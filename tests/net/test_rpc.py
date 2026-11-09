"""DEALER/ROUTER request-reply: ops, error replies, deadlines, late replies, CURVE."""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid

import pytest
import zmq

from inspect_robots_stretch4.net import auth
from inspect_robots_stretch4.net.framing import PROTOCOL, RemoteError, pack, unpack
from inspect_robots_stretch4.net.rpc import RpcClient, RpcServer, RpcTimeout, heartbeat_endpoint

CTX = zmq.Context.instance()


def endpoint() -> str:
    return f"inproc://rpc-{uuid.uuid4().hex}"


def serve(server: RpcServer, stop: threading.Event) -> threading.Thread:
    def loop() -> None:
        while not stop.is_set():
            server.serve_once(0.02)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


@pytest.fixture
def running():
    """Start an RpcServer on a background thread; stop and close it after the test."""
    stops: list[tuple[threading.Event, threading.Thread, RpcServer]] = []

    def _run(ep: str, handlers, **kwargs) -> RpcServer:
        server, stop = RpcServer(CTX, ep, handlers, **kwargs), threading.Event()
        stops.append((stop, serve(server, stop), server))
        return server

    yield _run
    for stop, thread, server in stops:
        stop.set()
        thread.join()
        server.close()


def echo(header, frames):
    return {"got": header["x"]}, frames[::-1]


def test_a_request_gets_its_reply(running) -> None:
    ep = endpoint()
    running(ep, {"echo": echo})
    client = RpcClient(CTX, ep)
    header, frames = client.call("echo", {"x": 5}, [b"a", b"b"])
    assert header["ok"] and header["got"] == 5 and frames == [b"b", b"a"]
    client.close()


def test_error_replies(running) -> None:
    def busy(header, frames):
        raise RemoteError("busy", "someone else")

    def broken(header, frames):
        raise KeyError("boom")

    ep = endpoint()
    running(ep, {"busy": busy, "broken": broken})
    client = RpcClient(CTX, ep)
    with pytest.raises(RemoteError, match="someone else") as e:
        client.call("busy")
    assert e.value.kind == "busy"
    with pytest.raises(RemoteError, match="KeyError") as e:
        client.call("broken")
    assert e.value.kind == "fault"
    with pytest.raises(RemoteError, match="unknown op 'nope'") as e:
        client.call("nope")
    assert e.value.kind == "protocol"
    client.close()


def test_a_silent_server_times_out_within_the_deadline() -> None:
    ep = endpoint()
    silent = CTX.socket(zmq.ROUTER)
    silent.bind(ep)
    client = RpcClient(CTX, ep, timeout_s=0.2)
    t0 = time.monotonic()
    with pytest.raises(RpcTimeout, match="'read'"):
        client.call("read")
    assert time.monotonic() - t0 < 0.5
    client.close()
    silent.close(linger=0)


def test_a_server_that_went_away_times_out_instead_of_blocking() -> None:
    ep = endpoint()
    router = CTX.socket(zmq.ROUTER)
    router.bind(ep)
    client = RpcClient(CTX, ep, timeout_s=0.2)
    router.close(linger=0)  # no peer left: a plain DEALER send would block forever
    t0 = time.monotonic()
    with pytest.raises(RpcTimeout, match="'ping'"):
        client.call("ping")
    with pytest.raises(RpcTimeout):
        client.call("ping")  # the recreated socket is just as bounded
    assert time.monotonic() - t0 < 1.0
    client.close()


def test_a_late_reply_is_dropped_and_counted() -> None:
    ep = endpoint()
    router = CTX.socket(zmq.ROUTER)
    router.bind(ep)

    def answer_twice() -> None:
        identity, *message = router.recv_multipart()
        header, _ = unpack(message)
        stale = {"ok": True, "id": header["id"] - 1, "which": "stale"}
        router.send_multipart([identity, *pack(stale)])
        router.send_multipart([identity, *pack({"ok": True, "id": header["id"], "which": "ok"})])

    client = RpcClient(CTX, ep)
    client._next_id = 1  # this request is id 2, so the stale reply (id 1) is a real past id
    t = threading.Thread(target=answer_twice)
    t.start()
    header, _ = client.call("read")
    t.join()
    assert header["which"] == "ok" and client.late_replies == 1
    client.close()
    router.close(linger=0)


def test_a_version_mismatch_reaches_the_caller(running) -> None:
    ep = endpoint()
    running(ep, {"echo": echo})
    dealer = CTX.socket(zmq.DEALER)
    dealer.connect(ep)
    dealer.send_multipart([json.dumps({"v": 99, "id": 4, "op": "echo"}).encode()])
    assert dealer.poll(1000)
    header, _ = unpack(dealer.recv_multipart())
    assert header["id"] == 4 and header["kind"] == "protocol"
    assert f"speaks {PROTOCOL}" in header["msg"] and "sent 99" in header["msg"]
    dealer.close(linger=0)


def test_heartbeat_endpoint() -> None:
    assert heartbeat_endpoint("tcp://10.0.0.2:5600") == "tcp://10.0.0.2:5601"
    assert heartbeat_endpoint("inproc://x") == "inproc://x-hb"
    with pytest.raises(ValueError):
        heartbeat_endpoint("tcp://10.0.0.2")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_curve_admits_trusted_clients_only(tmp_path, running) -> None:
    server_keys, good, bad = tmp_path / "server", tmp_path / "good", tmp_path / "bad"
    for keys in (server_keys, good, bad):
        auth.init_keys(keys)
    auth.trust(server_keys, "robot1", auth.public_key(good))
    for client_keys in (good, bad):
        auth.trust(client_keys, "robot2", auth.public_key(server_keys))
    authenticator = auth.start_authenticator(CTX, server_keys)
    try:
        ep = f"tcp://127.0.0.1:{free_port()}"
        running(ep, {"echo": echo}, curve=auth.server_curve(server_keys))
        trusted = RpcClient(CTX, ep, curve=auth.client_curve(good, "robot2"), timeout_s=2.0)
        assert trusted.call("echo", {"x": 1})[0]["got"] == 1
        trusted.close()
        stranger = RpcClient(CTX, ep, curve=auth.client_curve(bad, "robot2"), timeout_s=0.5)
        with pytest.raises(RpcTimeout):
            stranger.call("echo", {"x": 1})
        stranger.close()
    finally:
        authenticator.stop()

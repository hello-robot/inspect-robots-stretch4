"""Request/reply over zmq: a DEALER client with deadlines, and a ROUTER server.

DEALER/ROUTER rather than REQ/REP: a REQ socket whose reply never comes is stuck until it is
recreated, and there is no way to drop a late reply. Here every request carries an id, a reply
to an older request (one that already timed out) is dropped and counted, and a missed deadline
recreates the socket so nothing stale is ever delivered.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import zmq

from inspect_robots_stretch4.net.framing import (
    ProtocolError,
    RemoteError,
    error_reply,
    pack,
    unpack,
)

log = logging.getLogger(__name__)

# A handler takes a request's header and frames and returns the reply's.
Handler = Callable[[dict[str, Any], list[bytes]], tuple[dict[str, Any], Sequence[bytes]]]


class Curve(Protocol):
    def apply(self, socket: zmq.Socket[Any]) -> None: ...


class RpcTimeout(TimeoutError):
    """No reply within the deadline."""


def context() -> zmq.Context[Any]:
    """The process's shared zmq context (an inproc endpoint only reaches sockets in it)."""
    return zmq.Context.instance()


def heartbeat_endpoint(endpoint: str) -> str:
    """The heartbeat socket's endpoint: TCP port + 1, or `<name>-hb` for inproc."""
    if endpoint.startswith("inproc://"):
        return f"{endpoint}-hb"
    host, sep, port = endpoint.rpartition(":")
    if not (endpoint.startswith("tcp://") and sep and port.isdigit()):
        raise ValueError(f"expected tcp://<host>:<port> or inproc://<name>, got {endpoint!r}")
    return f"{host}:{int(port) + 1}"


class RpcClient:
    """Requests from one thread to one server. zmq sockets aren't thread-safe: one owner each."""

    def __init__(
        self,
        ctx: zmq.Context[Any],
        endpoint: str,
        *,
        curve: Curve | None = None,
        timeout_s: float = 1.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.late_replies = 0
        self._ctx = ctx
        self._curve = curve
        self._next_id = 0
        self._socket = self._open()

    def call(
        self,
        op: str,
        fields: dict[str, Any] | None = None,
        frames: Sequence[bytes] = (),
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any], list[bytes]]:
        """Send one request and wait for its reply. Raises `RpcTimeout` after the deadline,
        `RemoteError` for an error reply and `ProtocolError` for a malformed one."""
        timeout = self.timeout_s if timeout_s is None else timeout_s
        self._next_id += 1
        rid = self._next_id
        deadline = time.monotonic() + timeout
        message = pack({**(fields or {}), "op": op, "id": rid}, frames)
        # A DEALER with no connected peer blocks on send, forever: wait for one within the
        # deadline. The peer can still go between the poll and the send, so retry until then.
        while True:
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._socket.poll(max(1, int(remaining * 1000)), zmq.POLLOUT):
                try:
                    self._socket.send_multipart(message, flags=zmq.NOBLOCK)
                    break
                except zmq.Again:
                    continue
            self._reset()
            raise RpcTimeout(f"could not send {op!r} to {self.endpoint} within {timeout:.1f}s")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._socket.poll(max(1, int(remaining * 1000))):
                self._reset()
                raise RpcTimeout(f"no reply to {op!r} from {self.endpoint} within {timeout:.1f}s")
            header, out = unpack(self._socket.recv_multipart())
            if header["id"] != rid:
                self.late_replies += 1
                continue
            if not header.get("ok"):
                raise RemoteError(str(header.get("kind", "fault")), str(header.get("msg", "")))
            return header, out

    def close(self) -> None:
        self._socket.close(linger=0)

    def _open(self) -> zmq.Socket[Any]:
        socket = self._ctx.socket(zmq.DEALER)
        socket.linger = 0
        if self._curve is not None:
            self._curve.apply(socket)
        socket.connect(self.endpoint)
        return socket

    def _reset(self) -> None:
        """A fresh socket, so neither a queued request nor a late reply survives a timeout."""
        self._socket.close(linger=0)
        self._socket = self._open()


class RpcServer:
    """Answers requests by op. Driven by the owning thread through `serve_once`."""

    def __init__(
        self,
        ctx: zmq.Context[Any],
        endpoint: str,
        handlers: dict[str, Handler],
        *,
        curve: Curve | None = None,
    ) -> None:
        self.handlers = handlers
        self._socket = ctx.socket(zmq.ROUTER)
        self._socket.linger = 0
        if curve is not None:
            curve.apply(self._socket)
        self._socket.bind(endpoint)

    def serve_once(self, timeout_s: float) -> bool:
        """Answer at most one request, waiting up to `timeout_s` for it. True if one came."""
        if not self._socket.poll(max(1, int(timeout_s * 1000))):
            return False
        identity, *message = self._socket.recv_multipart()
        header, frames = self._handle(message)
        self._socket.send_multipart([identity, *pack(header, frames)])
        return True

    def close(self) -> None:
        self._socket.close(linger=0)

    def _handle(self, message: list[bytes]) -> tuple[dict[str, Any], Sequence[bytes]]:
        try:
            header, frames = unpack(message)
        except ProtocolError as e:
            return {**error_reply("protocol", str(e)), "id": _id_of(message)}, ()
        rid, op = header["id"], header.get("op")
        handler = self.handlers.get(op) if isinstance(op, str) else None
        if handler is None:
            return {**error_reply("protocol", f"unknown op {op!r}"), "id": rid}, ()
        try:
            reply, out = handler(header, frames)
        except RemoteError as e:
            return {**error_reply(e.kind, e.msg), "id": rid}, ()
        except ProtocolError as e:
            return {**error_reply("protocol", str(e)), "id": rid}, ()
        except Exception as e:
            log.exception("handling %r failed", op)
            return {**error_reply("fault", f"{type(e).__name__}: {e}"), "id": rid}, ()
        return {"ok": True, **reply, "id": rid}, out


def _id_of(message: list[bytes]) -> int:
    """A malformed request's id if it has one, so the error still reaches the caller."""
    try:
        rid = json.loads(bytes(message[0])).get("id")
    except Exception:
        return -1
    return rid if isinstance(rid, int) else -1

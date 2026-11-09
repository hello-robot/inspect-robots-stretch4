"""A real `DriverService` in front of a `FakeDriver`, and a `RemoteDriver` pointed at it."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
import uuid

import numpy as np
import pytest
import zmq
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4 import JOINTS, Stretch4Driver, Stretch4EmbodimentConfig
from inspect_robots_stretch4.config import MAX_STEP
from inspect_robots_stretch4.driver import LinkLost
from inspect_robots_stretch4.net import auth
from inspect_robots_stretch4.remote import service as service_mod
from inspect_robots_stretch4.remote.client import RemoteDriver
from inspect_robots_stretch4.remote.service import DriverService

CTX = zmq.Context.instance()
CFG = Stretch4EmbodimentConfig(base="none", max_image_side=8)
FAST = {"rpc_timeout_s": 0.5, "heartbeat_period_s": 0.05, "heartbeat_timeout_s": 0.3}
ROBOT_AHEAD_S = 100.0  # the robot's monotonic clock vs this process's


@pytest.fixture
def served(driver):
    """(endpoint, service, factory calls) for a service around the fake driver."""
    endpoint = f"inproc://stretch4-{uuid.uuid4().hex}"
    built: list[Stretch4EmbodimentConfig] = []
    service = DriverService(
        endpoint,
        lambda cfg: built.append(cfg) or driver,
        ctx=CTX,
        clock=lambda: time.monotonic() + ROBOT_AHEAD_S,
    )
    service.start()
    yield endpoint, service, built
    service.close()


@pytest.fixture
def remote(served):
    clients: list[RemoteDriver] = []

    def _connect(cfg: Stretch4EmbodimentConfig = CFG, **kwargs) -> RemoteDriver:
        client = RemoteDriver(cfg, served[0], ctx=CTX, **{**FAST, **kwargs})
        clients.append(client)
        return client

    yield _connect
    for client in clients:
        client.disconnect()


def test_remote_driver_is_a_stretch4_driver(remote) -> None:
    assert isinstance(remote(), Stretch4Driver)


def test_read_decodes_images_and_maps_the_robot_clock(remote, driver) -> None:
    client = remote()
    driver.t = time.monotonic() + ROBOT_AHEAD_S  # captured "now", on the robot's clock
    reading = client.read()
    assert reading.images["head_right"].shape == (5, 8, 3)
    assert reading.images["gripper_right"].shape == (5, 8, 3)
    np.testing.assert_array_equal(reading.joints, driver.joints)
    assert abs(reading.t - time.monotonic()) < 0.05
    link = client.link()
    assert link["offset_s"] == pytest.approx(ROBOT_AHEAD_S, abs=0.05) and link["rtt_s"] >= 0


def test_command_reaches_the_robot_and_the_service_clamps_again(remote, driver) -> None:
    client = remote()
    start = driver.joints.copy()
    with pytest.raises(EmbodimentFault, match="command before a read"):
        client.command(dict(zip(JOINTS, start.tolist(), strict=True)))
    client.read()
    far = dict(zip(JOINTS, (start + 1.0).tolist(), strict=True))
    sent = client.command(far)
    assert sent["lift"] == pytest.approx(start[0] + MAX_STEP[0])
    assert driver.commands[-1] == sent


def test_goto_passes_arrival_and_error_through(remote, driver) -> None:
    client = remote()
    pose = dict.fromkeys(JOINTS, 0.2)
    assert client.goto(pose, 5.0) == (True, ["lift", 0.2])
    driver.arrive = False
    assert client.goto(pose, 5.0)[0] is False
    assert driver.gotos[-1] == pose


def test_a_runstop_arrives_while_a_goto_is_in_flight(remote, driver) -> None:
    client = remote()
    release, results = threading.Event(), []
    driver.on_goto = lambda: release.wait(5.0)
    homing = threading.Thread(
        target=lambda: results.append(client.goto(dict.fromkeys(JOINTS, 0), 9))
    )
    homing.start()
    time.sleep(0.1)  # the goto is now blocking the service's main loop
    client.trigger_runstop()
    assert driver.runstop_triggers == 1 and homing.is_alive()
    assert client.is_runstopped()
    release.set()
    homing.join(5.0)
    assert results and results[0][0] is True


def test_driver_errors_keep_their_kind(remote, driver) -> None:
    client = remote()

    def stopped():
        raise SafetyAbort("the Stretch 4 run-stop is engaged")

    driver.read_encoded = stopped
    with pytest.raises(SafetyAbort, match="run-stop is engaged"):
        client.read()


def test_host_loss_ends_the_session_and_the_robot_holds(remote, driver) -> None:
    client = remote()
    client.read()
    client._stop.set()  # the host goes silent: no more heartbeats
    client._thread.join()
    time.sleep(FAST["heartbeat_timeout_s"] + 0.15)
    commands = len(driver.commands)
    # A lost link, which the pair rides out by saying hello again.
    with pytest.raises(LinkLost, match="no live session"):
        client.command(dict(zip(JOINTS, driver.joints.tolist(), strict=True)))
    assert len(driver.commands) == commands


def test_one_host_at_a_time(remote) -> None:
    first = remote()
    with pytest.raises(EmbodimentFault, match="another host's session"):
        remote()
    first.disconnect()
    remote().read()  # after a goodbye, the next host gets in


def test_the_driver_is_reused_until_the_config_changes(remote, served, driver) -> None:
    _, _, built = served
    remote().disconnect()
    remote().disconnect()
    assert len(built) == 1 and driver.disconnects == 0
    remote(Stretch4EmbodimentConfig(base="none", max_image_side=8, auto_start=True)).disconnect()
    assert len(built) == 2 and driver.disconnects == 1


def test_a_service_that_never_answers_is_a_fault_within_the_deadline() -> None:
    t0 = time.monotonic()
    with pytest.raises(EmbodimentFault, match="no reply to 'hello'"):
        RemoteDriver(CFG, f"inproc://nobody-{uuid.uuid4().hex}", ctx=CTX, connect_timeout_s=0.3)
    assert time.monotonic() - t0 < 1.0


def test_a_robot_that_goes_away_is_lost(remote, served) -> None:
    client = remote()
    served[1].close()
    time.sleep(2 * FAST["heartbeat_period_s"] + FAST["rpc_timeout_s"] + 0.2)
    with pytest.raises(EmbodimentFault, match="lost the Stretch 4"):
        client.is_runstopped()


LOST_AFTER_S = 2 * FAST["heartbeat_period_s"] + FAST["rpc_timeout_s"]


def hold_targets(driver) -> dict[str, float]:
    return dict(zip(JOINTS, driver.joints.tolist(), strict=True))


def test_heartbeats_are_answered_while_a_command_stalls_on_the_robot(remote, driver) -> None:
    """A slow push to the robot's hardware never silences its heartbeats (trial 2, 2026-11-04)."""
    client = remote()
    client.read()
    release = threading.Event()
    driver.on_command = lambda: release.wait(5.0)
    sender = threading.Thread(target=lambda: _quietly(client.command, hold_targets(driver)))
    sender.start()
    try:
        time.sleep(LOST_AFTER_S + 0.3)  # longer than the host waits before calling robot 2 lost
        assert client.is_runstopped() is False  # still hearing pings: no LinkLost
    finally:
        release.set()
        sender.join()
    client.read()
    client.command(hold_targets(driver))  # the session lived through the stall


def test_a_slow_robot_command_is_logged(remote, driver, monkeypatch, caplog) -> None:
    monkeypatch.setattr(service_mod, "SLOW_DRIVER_S", 0.05)
    client = remote(rpc_timeout_s=2.0)
    client.read()
    driver.on_command = lambda: time.sleep(0.15)
    with caplog.at_level("WARNING", logger=service_mod.__name__):
        client.command(hold_targets(driver))
    assert "the robot took" in caplog.text and "command" in caplog.text


def test_silence_from_the_robot_is_a_lost_link(remote, served) -> None:
    client = remote()
    served[1].close()
    time.sleep(LOST_AFTER_S + 0.2)
    with pytest.raises(LinkLost, match="lost the Stretch 4"):
        client.is_runstopped()


def test_a_robot_that_comes_back_can_be_rejoined(remote, served, driver) -> None:
    endpoint, service, _ = served
    client = remote()
    client.read()
    service.close()
    time.sleep(LOST_AFTER_S + 0.2)
    with pytest.raises(LinkLost):
        client.is_runstopped()
    back = DriverService(
        endpoint, lambda cfg: driver, ctx=CTX, clock=lambda: time.monotonic() + ROBOT_AHEAD_S
    )
    back.start()
    try:
        client.reconnect(timeout_s=3.0)
        assert client.is_runstopped() is False
        client.read()  # the rejoined session needs a fresh read before a command
        client.command(hold_targets(driver))
    finally:
        client.disconnect()
        back.close()


def test_rejoining_a_robot_that_stays_away_gives_up_in_time(remote, served) -> None:
    client = remote()
    served[1].close()
    t0 = time.monotonic()
    with pytest.raises(LinkLost):
        client.reconnect(timeout_s=0.5)
    assert time.monotonic() - t0 < 1.5


def _quietly(fn, *args) -> None:
    with contextlib.suppress(Exception):
        fn(*args)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_over_tcp_with_curve(tmp_path, driver) -> None:
    host, robot, stranger = tmp_path / "host", tmp_path / "robot", tmp_path / "stranger"
    for keys in (host, robot, stranger):
        auth.init_keys(keys)
    auth.trust(robot, "robot1", auth.public_key(host))
    auth.trust(host, "robot2", auth.public_key(robot))
    auth.trust(stranger, "robot2", auth.public_key(robot))
    authenticator = auth.start_authenticator(CTX, robot)
    endpoint = f"tcp://127.0.0.1:{free_port()}"
    service = DriverService(endpoint, lambda cfg: driver, ctx=CTX, curve=auth.server_curve(robot))
    service.start()
    try:
        with pytest.raises(EmbodimentFault, match="no reply"):
            RemoteDriver(
                CFG,
                endpoint,
                ctx=CTX,
                curve=auth.client_curve(stranger, "robot2"),
                connect_timeout_s=0.5,
            )
        client = RemoteDriver(CFG, endpoint, ctx=CTX, curve=auth.client_curve(host, "robot2"))
        assert client.read().images["gripper_right"].shape == (5, 8, 3)
        client.disconnect()
    finally:
        service.close()
        authenticator.stop()


def test_a_command_after_a_goto_needs_a_fresh_read(remote, driver) -> None:
    client = remote()
    client.read()
    client.goto(dict.fromkeys(JOINTS, 0.4), 5.0)  # the fake "arrives"; the old read is stale
    with pytest.raises(EmbodimentFault, match="first after a goto"):
        client.command(dict(zip(JOINTS, driver.joints.tolist(), strict=True)))
    client.read()
    client.command(dict(zip(JOINTS, driver.joints.tolist(), strict=True)))


def test_late_or_replayed_motion_requests_never_move_the_robot(remote, driver) -> None:
    client = remote()
    client.read()
    pose = {j: float(v) for j, v in zip(JOINTS, driver.joints, strict=True)}
    late = {**client._motion(), "not_after": 0.0, "targets": pose}
    with pytest.raises(EmbodimentFault, match="after the host stopped waiting"):
        client._call(client._rpc, "command", late)
    fields = client._motion()
    client._call(client._rpc, "command", {**fields, "targets": pose})
    with pytest.raises(EmbodimentFault, match="arrived after request"):
        client._call(client._rpc, "command", {**fields, "targets": pose})  # same seq again
    assert len(driver.commands) == 1


def test_stopping_the_service_cancels_a_goto_and_keeps_answering_until_then(
    remote, served, driver
) -> None:
    client = remote()
    gave_up = threading.Event()

    def slow_home() -> None:  # homing that only ends when cancelled
        driver.goto_cancelled.wait(5.0)

    driver.on_goto = slow_home
    homing = threading.Thread(
        target=lambda: (client.goto(dict.fromkeys(JOINTS, 0.0), 9.0), gave_up.set())
    )
    homing.start()
    time.sleep(0.1)
    served[1].stop()
    time.sleep(0.05)
    t0 = time.monotonic()
    served[1].close()
    assert driver.goto_cancelled.is_set() and time.monotonic() - t0 < 1.0
    homing.join(2.0)
    assert gave_up.is_set()


def test_the_run_stop_state_is_not_undone_by_a_stale_ping(remote, driver) -> None:
    client = remote()
    driver.trigger_runstop = lambda: setattr(driver, "runstop_triggers", 1)  # robot lags
    client.trigger_runstop()
    time.sleep(3 * FAST["heartbeat_period_s"])  # pings still say "not stopped"
    assert driver.runstop_triggers == 1 and client.is_runstopped()


def test_heartbeat_timeout_must_leave_room_for_late_pings() -> None:
    with pytest.raises(ValueError, match="at least 3x"):
        RemoteDriver(CFG, "inproc://x", ctx=CTX, heartbeat_period_s=0.2, heartbeat_timeout_s=0.5)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (
            Stretch4EmbodimentConfig(
                base="none", max_image_side=8, joint_high=(1.3, 0.5, 4, 4, 4, 70)
            ),
            "wider",
        ),
        (Stretch4EmbodimentConfig(base="none", max_image_side=8, priority=100), "priority"),
        (
            Stretch4EmbodimentConfig(base="rotate", max_image_side=8, base_max_theta=4.0),
            "turn limit",
        ),
    ],
)
def test_robot_2_refuses_wider_limits_or_runstop_priority(remote, cfg, match) -> None:
    with pytest.raises(EmbodimentFault, match=match):
        remote(cfg)


def test_a_tcp_service_without_curve_is_refused(driver) -> None:
    with pytest.raises(ValueError, match="CURVE"):
        DriverService(f"tcp://127.0.0.1:{free_port()}", lambda cfg: driver, ctx=CTX)


def test_a_served_base_turns_and_is_zeroed(remote, driver) -> None:
    cfg = Stretch4EmbodimentConfig(base="rotate", max_image_side=8, base_max_theta=0.35)
    driver.configure(cfg)
    client = remote(cfg)
    reading = client.read()
    assert reading.joints.shape == (7,)
    turn = dict(zip(cfg.joints, [*driver.joints[:6].tolist(), 0.1], strict=True))
    sent = client.command(turn)
    assert sent["base_theta"] == pytest.approx(0.1)
    assert driver.commands[-1]["base_theta"] == pytest.approx(0.1)
    client.zero_base()
    assert driver.events[-1] == "zero_base"
    with pytest.raises(EmbodimentFault, match="command before a read"):
        client.command(turn)  # base_theta changed under the last read
    client.read()
    client.release_base()
    assert driver.events[-1] == "release_base"


def test_robot_2s_runstop_is_reported_once_as_it_engages(remote, driver) -> None:
    client = remote()
    seen = threading.Event()
    calls: list[float] = []
    client.on_runstop(lambda: calls.append(time.monotonic()) or seen.set())
    time.sleep(3 * FAST["heartbeat_period_s"])
    assert not calls
    driver.runstopped = True  # pressed on robot 2
    assert seen.wait(1.0)
    time.sleep(3 * FAST["heartbeat_period_s"])  # still engaged: no second call
    assert len(calls) == 1 and client.is_runstopped()


def test_a_runstop_request_does_not_wait_for_the_robot(remote, driver) -> None:
    client = remote()
    reached, stalled = threading.Event(), threading.Event()
    driver.trigger_runstop = lambda: reached.set() or stalled.wait(2.0)  # the robot is slow
    t0 = time.monotonic()
    client.request_runstop()
    assert time.monotonic() - t0 < 0.05
    assert reached.wait(1.0)
    stalled.set()


def test_a_failed_runstop_request_is_resent_until_robot_2_confirms(remote, driver) -> None:
    client = remote()
    tries, done = [], threading.Event()

    def flaky_runstop() -> None:
        tries.append(time.monotonic())
        if len(tries) < 3:  # the robot can't take it yet
            raise EmbodimentFault("busy")
        driver.runstop_triggers += 1
        done.set()

    driver.trigger_runstop = flaky_runstop
    client.request_runstop()
    assert done.wait(2.0) and len(tries) == 3
    time.sleep(3 * FAST["heartbeat_period_s"])
    assert len(tries) == 3 and client.is_runstopped()  # confirmed: not sent again

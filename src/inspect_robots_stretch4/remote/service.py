"""`DriverService`: one Stretch 4's driver, served to one host at a time.

Two sockets, two threads:

- the main socket answers `hello`, `read`, `command`, `goto` and `bye`, one at a time;
- the heartbeat socket (port + 1) answers `ping` and `runstop`, so a run-stop never waits
  behind a read or a 40 s `goto`, and watches the session: when pings stop for
  `heartbeat_timeout_s`, the session ends and the robot holds.

The driver stays open between sessions, because the cameras take seconds to reopen.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4.config import JOINTS, Stretch4EmbodimentConfig, clamp
from inspect_robots_stretch4.driver import RUNSTOP_PRIORITY, EncodedReading, Stretch4Driver
from inspect_robots_stretch4.net import rpc
from inspect_robots_stretch4.net.framing import PROTOCOL, ProtocolError, RemoteError
from inspect_robots_stretch4.remote.codec import IMAGE_MODES, config_from_json, encode_reading

log = logging.getLogger(__name__)

POLL_S = 0.05
# A read or command that keeps the robot this long is logged: heartbeats don't wait on it, but a
# slow robot is worth knowing about when a trial misbehaves.
SLOW_DRIVER_S = 0.5


class EncodingDriver(Stretch4Driver, Protocol):
    """A driver that can also hand over its camera frames undecoded and cancel a `goto` from
    another thread (`BodyDriver`)."""

    def read_encoded(self) -> EncodedReading: ...

    def cancel_goto(self) -> None: ...


@dataclass
class _Session:
    id: str
    started: float
    last_ping: float
    timeout_s: float


class DriverService:
    """Serves `factory(cfg)`'s driver at `endpoint`; the constructor binds but doesn't serve."""

    def __init__(
        self,
        endpoint: str,
        factory: Callable[[Stretch4EmbodimentConfig], EncodingDriver],
        *,
        ctx: Any = None,
        curve: rpc.Curve | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if endpoint.startswith("tcp://") and curve is None:
            raise ValueError("a TCP service needs CURVE keys; only inproc:// may run without")
        self._factory = factory
        self._clock = clock
        self._ctx = ctx if ctx is not None else rpc.context()
        self._lock = threading.Lock()  # session and driver, shared by both threads
        self._driver: EncodingDriver | None = None
        self._cfg: Stretch4EmbodimentConfig | None = None
        self._mode = "forward"
        self._session: _Session | None = None
        # The joints of the last read, for the re-clamp; cleared by anything that moves the robot
        # without a read after it (a goto), so a command is never clamped around a stale pose.
        self._joints: np.ndarray | None = None
        self._last_seq = 0  # the session's newest command or goto
        self._stop = threading.Event()
        self._hb_stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._main = rpc.RpcServer(
            self._ctx,
            endpoint,
            {
                "hello": self._hello,
                "read": self._read,
                "command": self._command,
                "goto": self._goto,
                "zero_base": self._zero_base,
                "release_base": self._release_base,
                "bye": self._bye,
            },
            curve=curve,
        )
        try:
            self._hb = rpc.RpcServer(
                self._ctx,
                rpc.heartbeat_endpoint(endpoint),
                {"ping": self._ping, "runstop": self._runstop},
                curve=curve,
            )
        except BaseException:
            self._main.close()
            raise

    # -- running ----------------------------------------------------------

    def run(self) -> None:
        """Serve until `stop()`: the heartbeat thread in the background, the main loop here.
        Heartbeats and run-stops are answered until the main loop has finished, so a `goto`
        that is still ending can always be stopped."""
        hb = threading.Thread(target=self._heartbeat_loop, daemon=True, name="pair-server-hb")
        hb.start()
        self._threads.append(hb)
        try:
            while not self._stop.is_set():
                self._main.serve_once(POLL_S)
        finally:
            self._hb_stop.set()
            hb.join()

    def start(self) -> None:
        """`run()` on a background thread (tests, embedding)."""
        t = threading.Thread(target=self.run, daemon=True, name="pair-server")
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        """End serving; a running `goto` is cancelled and the robot holds. Safe from a signal
        handler or any thread."""
        self._stop.set()
        with self._lock:
            driver = self._driver
        if driver is not None:
            driver.cancel_goto()

    def close(self) -> None:
        """Stop serving, close the sockets and the driver (idempotent)."""
        self.stop()
        alive = []
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=5.0)
                if t.is_alive():
                    alive.append(t.name)
        self._threads = []
        if alive:
            # Closing a socket or the driver under a thread still using it is unsafe.
            log.error("threads %s did not stop; leaving their sockets and driver open", alive)
            return
        self._main.close()
        self._hb.close()
        with self._lock:
            driver, self._driver, self._session = self._driver, None, None
        if driver is not None:
            driver.disconnect()

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.is_set():
            self._hb.serve_once(POLL_S)
            self._expire()

    def _expire(self) -> None:
        with self._lock:
            s = self._session
            if s is not None and self._clock() - s.last_ping > s.timeout_s:
                log.warning(
                    "no heartbeat from the host for %.1fs: ending its session; the robot holds",
                    s.timeout_s,
                )
                self._session = None

    # -- main socket ------------------------------------------------------

    def _hello(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        session = _str(header, "session")
        timeout_s = _positive(header, "heartbeat_timeout_s")
        mode = header.get("image_mode", "forward")
        if mode not in IMAGE_MODES:
            raise ProtocolError(f"unknown image_mode {mode!r}; one of {IMAGE_MODES}")
        cfg = config_from_json(header.get("cfg"))
        _check_own_limits(cfg)
        self._expire()
        with self._lock:
            live = self._session
            if live is not None and live.id != session:
                raise RemoteError(
                    "busy",
                    f"another host's session ({self._clock() - live.started:.0f}s old) is "
                    f"live; stop that run, or wait {live.timeout_s:.1f}s after it stops",
                )
            reuse = self._driver is not None and cfg == self._cfg
            old = None if reuse else self._driver
            if not reuse:
                self._driver = self._cfg = None
        if old is not None:
            log.info("the host sent a different config: reopening the robot")
            old.disconnect()
        if not reuse:
            log.info("opening the robot")
            driver = _driver_call(lambda: self._factory(cfg))
            with self._lock:
                self._driver, self._cfg = driver, cfg
        now = self._clock()
        with self._lock:
            self._session = _Session(session, now, now, timeout_s)
            self._mode = mode
            self._joints = None
            self._last_seq = 0
        log.info("host session %s started (images: %s)", session[:8], mode)
        return {"gripper": cfg.gripper, "proto": PROTOCOL, "cv2": _cv2_version()}, []

    def _read(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        driver, cfg = self._require(header)
        encoded = _timed("read", lambda: _driver_call(driver.read_encoded))
        fields, out = encode_reading(encoded, cfg, self._mode)
        self._joints = np.asarray(encoded.joints, dtype=np.float64)
        return {**fields, "runstopped": driver.is_runstopped()}, out

    def _command(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        with self._lock:
            driver, cfg = self._session_driver(header), self._cfg
            assert cfg is not None  # a live session has a config
            joints = cfg.joints
            target = _pose(header, "targets", joints)
            self._check_fresh(header)
            if self._joints is None:
                raise RemoteError(
                    "fault", "command before a read (the first, or the first after a goto)"
                )
            # Defence in depth: the host clamps too, but robot 2 never relies on it alone.
            safe = clamp(np.array([target[j] for j in joints]), self._joints, cfg)
        # The push runs outside the lock, which the heartbeat thread also takes: a robot slow to
        # take a command must never silence the heartbeats, or the host calls it lost. A push
        # checked under a live session may land just after the session expires: one last
        # clamped step, sent by a host that was alive when it sent it.
        sent = _timed(
            "command",
            lambda: _driver_call(
                lambda: driver.command(dict(zip(joints, safe.tolist(), strict=True)))
            ),
        )
        return {"sent": {j: float(sent[j]) for j in joints}}, []

    def _goto(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        pose = _pose(header, "pose")
        timeout_s = _positive(header, "timeout_s")
        with self._lock:
            driver = self._session_driver(header)
            self._check_fresh(header)
            self._joints = None  # the robot is about to move: the next command needs a read
        arrived, error = _driver_call(lambda: driver.goto(pose, timeout_s))
        return {"arrived": bool(arrived), "error": _jsonable(error)}, []

    def _zero_base(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        """Make the base's current pose its zero (each trial's start)."""
        return self._base_call(header, "zero_base")

    def _release_base(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        """Let the wheels spin freely, so the operator can push the robot into place."""
        return self._base_call(header, "release_base")

    def _base_call(self, header: dict[str, Any], method: str) -> tuple[dict, list]:
        with self._lock:
            driver = self._session_driver(header)
            # base_theta changes under the host's last read: its next command needs a new one.
            self._joints = None
        _driver_call(getattr(driver, method))
        return {}, []

    def _bye(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        session = _str(header, "session")
        with self._lock:
            if self._session is not None and self._session.id == session:
                self._session = None
                log.info("host session %s ended", session[:8])
        return {}, []

    # -- heartbeat socket -------------------------------------------------

    def _ping(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        with self._lock:
            driver = self._session_driver(header)
            self._session.last_ping = self._clock()  # type: ignore[union-attr]
        return {"t_robot": self._clock(), "runstopped": driver.is_runstopped()}, []

    def _runstop(self, header: dict[str, Any], frames: Sequence[bytes]) -> tuple[dict, list]:
        # Honoured without a live session too: stopping is always allowed.
        with self._lock:
            driver = self._driver
        if driver is None:
            raise RemoteError("fault", "no robot is open, so there is nothing to stop")
        log.warning("run-stop requested by the host")
        _driver_call(driver.trigger_runstop)
        return {}, []

    # -- helpers ----------------------------------------------------------

    def _require(self, header: dict[str, Any]) -> tuple[EncodingDriver, Stretch4EmbodimentConfig]:
        with self._lock:
            return self._session_driver(header), self._cfg  # type: ignore[return-value]

    def _check_fresh(self, header: dict[str, Any]) -> None:
        """Refuse a motion request that is out of order or that the host has stopped waiting
        for: a request delayed in the network must not move the robot late. Lock held."""
        seq, not_after = header.get("seq"), header.get("not_after")
        if not isinstance(seq, int) or not isinstance(not_after, int | float):
            raise ProtocolError("a motion request needs an integer `seq` and a `not_after` time")
        if seq <= self._last_seq:
            raise RemoteError("fault", f"request {seq} arrived after request {self._last_seq}")
        late = self._clock() - not_after
        if late > 0:
            raise RemoteError(
                "fault", f"the request arrived {late:.2f}s after the host stopped waiting"
            )
        self._last_seq = seq

    def _session_driver(self, header: dict[str, Any]) -> EncodingDriver:
        """The driver, if `header` belongs to the live session. Call with the lock held."""
        s = self._session
        if s is None or s.id != header.get("session") or self._driver is None:
            raise RemoteError("no_session", "no live session for this host; send hello first")
        return self._driver


def _check_own_limits(cfg: Stretch4EmbodimentConfig) -> None:
    """Robot 2 takes its joint limits from the host's config, so refuse any wider than its own
    defaults for that gripper (the base's turn limit included), and any priority that would
    outrank a run-stop."""
    own = Stretch4EmbodimentConfig(gripper=cfg.gripper)
    if cfg.base_max_theta > own.base_max_theta:
        raise ProtocolError(
            f"robot 2 refuses a base turn limit wider than {own.base_max_theta:.2f} rad"
        )
    wider = [
        j
        for j, lo, hi, own_lo, own_hi in zip(
            JOINTS, cfg.joint_low, cfg.joint_high, own.joint_low, own.joint_high, strict=True
        )
        if lo < own_lo or hi > own_hi
    ]
    if wider:
        raise ProtocolError(f"robot 2 refuses joint limits wider than its defaults for {wider}")
    if cfg.priority >= RUNSTOP_PRIORITY:
        raise ProtocolError(f"priority must be below the run-stop's ({RUNSTOP_PRIORITY})")


def _timed(what: str, fn: Callable[[], Any]) -> Any:
    """`fn()`, logging a warning if it kept the robot longer than SLOW_DRIVER_S."""
    t0 = time.monotonic()
    try:
        return fn()
    finally:
        took = time.monotonic() - t0
        if took > SLOW_DRIVER_S:
            log.warning("the robot took %.1f s to take a %s", took, what)


def _driver_call(fn: Callable[[], Any]) -> Any:
    """Run a driver call, turning its errors into error replies of the matching kind."""
    try:
        return fn()
    except SafetyAbort as e:
        raise RemoteError("safety", str(e)) from None
    except EmbodimentFault as e:
        raise RemoteError("fault", str(e)) from None


def _str(header: Mapping[str, Any], key: str) -> str:
    value = header.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"`{key}` must be a non-empty string")
    return value


def _positive(header: Mapping[str, Any], key: str) -> float:
    value = header.get(key)
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ProtocolError(f"`{key}` must be a positive number, got {value!r}")
    return float(value)


def _pose(
    header: Mapping[str, Any], key: str, joints: tuple[str, ...] = JOINTS
) -> dict[str, float]:
    pose = header.get(key)
    if not isinstance(pose, dict) or set(pose) != set(joints):
        raise ProtocolError(f"`{key}` must give exactly {list(joints)}")
    values = {j: pose[j] for j in joints}
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in values.values()):
        raise ProtocolError(f"`{key}` has a value that is not a finite number")
    return {j: float(v) for j, v in values.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


def _cv2_version() -> str | None:
    try:
        import cv2
    except ImportError:
        return None
    return str(cv2.__version__)

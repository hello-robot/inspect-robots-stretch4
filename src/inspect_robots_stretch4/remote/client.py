"""`RemoteDriver`: a `Stretch4Driver` for a Stretch 4 served by `stretch_inspect_pair_server`.

`RobotSide` can't tell it from `BodyDriver`. Two sockets, each with one owner:

- `rpc`, used by the caller's thread for `read`, `command`, `goto` and `bye`;
- `hb`, owned by a heartbeat thread that pings at 5 Hz (keeping the session alive and tracking
  the robot's run-stop and clock) and also carries `runstop`, so a run-stop never waits behind
  a slow read or a homing `goto`.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4.config import JOINTS, Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import LinkLost, Reading
from inspect_robots_stretch4.net import rpc
from inspect_robots_stretch4.net.clock import ClockOffset
from inspect_robots_stretch4.net.framing import PROTOCOL, ProtocolError, RemoteError
from inspect_robots_stretch4.remote.codec import (
    IMAGE_MODES,
    config_to_json,
    decode_reading,
)

log = logging.getLogger(__name__)

GOTO_MARGIN_S = 5.0
BYE_TIMEOUT_S = 0.2
# After a run-stop, a ping may still report the robot's state from just before it (the robot's
# status lags by a few ms), so for this long only a "stopped" report is believed.
RUNSTOP_SETTLE_S = 0.5


class RemoteDriver:
    """Connects, says hello and starts heartbeats in the constructor."""

    def __init__(
        self,
        cfg: Stretch4EmbodimentConfig,
        endpoint: str,
        *,
        ctx: Any = None,
        curve: rpc.Curve | None = None,
        rpc_timeout_s: float = 1.0,
        heartbeat_period_s: float = 0.2,
        heartbeat_timeout_s: float = 1.0,
        image_mode: str = "forward",
        connect_timeout_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if image_mode not in IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {IMAGE_MODES}, got {image_mode!r}")
        # Pings go out every period and each waits at most a period, so the service sees one at
        # least every two periods; leave it room for one more before it ends the session.
        if heartbeat_timeout_s < 3 * heartbeat_period_s:
            raise ValueError(
                f"heartbeat_timeout_s ({heartbeat_timeout_s}) must be at least 3x "
                f"heartbeat_period_s ({heartbeat_period_s})"
            )
        self._cfg = cfg
        self.endpoint = endpoint
        self._mode = image_mode
        self._clock = clock
        self._period = heartbeat_period_s
        self._rpc_timeout = rpc_timeout_s
        # No successful ping for this long means the robot is gone.
        self._lost_after = 2 * heartbeat_period_s + rpc_timeout_s
        self._session = uuid.uuid4().hex
        self._offset = ClockOffset()
        self._offset_lock = threading.Lock()
        self._runstopped = False
        self._reported_runstop = False  # robot 2's own run-stop, as the last ping saw it
        self._on_runstop: Callable[[], None] | None = None
        self._runstop_at = -float("inf")
        self._seq = 0
        self._last_ok = clock()
        self._stop = threading.Event()
        self._runstop_lock = threading.Lock()
        self._runstop_wanted = threading.Event()
        self._runstop_done = threading.Event()
        self._runstop_error: Exception | None = None
        self._runstop_pending = False  # a run-stop not yet confirmed: resent until it is
        self._thread: threading.Thread | None = None
        self._closed = False
        ctx = ctx if ctx is not None else rpc.context()
        self._rpc = rpc.RpcClient(ctx, endpoint, curve=curve, timeout_s=rpc_timeout_s)
        # A ping waits one period at most, so a slow one never delays the next ping or a
        # run-stop by more than that.
        self._hb = rpc.RpcClient(
            ctx, rpc.heartbeat_endpoint(endpoint), curve=curve, timeout_s=heartbeat_period_s
        )
        # Sent again, unchanged, by reconnect(): robot 2 treats it as the same host.
        self._hello = {
            "session": self._session,
            "cfg": config_to_json(cfg),
            "heartbeat_timeout_s": heartbeat_timeout_s,
            "image_mode": image_mode,
        }
        try:
            reply, _ = self._call(
                self._rpc,
                "hello",
                self._hello,
                # The service opens the robot's cameras on the first hello: seconds, not ms.
                timeout_s=connect_timeout_s,
            )
            self.remote_cv2: str | None = reply.get("cv2")
            self._ping(timeout_s=rpc_timeout_s)  # a clock offset before the first read
        except BaseException:
            self._rpc.close()
            self._hb.close()
            raise
        self._thread = threading.Thread(target=self._heartbeat, daemon=True, name="stretch4-hb")
        self._thread.start()
        log.info("connected to the Stretch 4 at %s (protocol %d)", endpoint, PROTOCOL)

    # -- Stretch4Driver ---------------------------------------------------

    def read(self) -> Reading:
        self._check_alive()
        header, frames = self._call(self._rpc, "read", {"session": self._session})
        self._runstopped = bool(header.get("runstopped", self._runstopped))
        try:
            return decode_reading(header, frames, self._cfg, self._mode, self._to_local)
        except ProtocolError as e:
            raise EmbodimentFault(f"{self.endpoint}: a malformed reading: {e}") from None

    def command(self, targets: Mapping[str, float]) -> Mapping[str, float]:
        header, _ = self._call(
            self._rpc,
            "command",
            {
                **self._motion(),
                "targets": {j: float(targets[j]) for j in self._cfg.joints},
            },
        )
        return dict(header["sent"])

    def goto(self, pose: Mapping[str, float], timeout_s: float) -> tuple[bool, Any]:
        header, _ = self._call(
            self._rpc,
            "goto",
            {
                **self._motion(),
                "pose": {j: float(pose[j]) for j in JOINTS},
                "timeout_s": timeout_s,
            },
            timeout_s=timeout_s + GOTO_MARGIN_S,
        )
        return bool(header["arrived"]), header.get("error")

    def is_runstopped(self) -> bool:
        self._check_alive()
        return self._runstopped

    def on_runstop(self, callback: Callable[[], None] | None) -> None:
        """Call `callback` from the heartbeat thread each time a ping finds robot 2's run-stop
        newly engaged. It must return at once: pings are due every heartbeat period."""
        self._on_runstop = callback

    def request_runstop(self) -> None:
        """Ask the heartbeat thread to send `runstop` now, without waiting for it. It is
        resent until robot 2 confirms it."""
        self._runstop_pending = True
        self._runstop_wanted.set()

    def trigger_runstop(self) -> None:
        """Ask the heartbeat thread to send `runstop` now, and wait for the robot to confirm."""
        with self._runstop_lock:
            self._runstop_done.clear()
            self._runstop_error = None
            self._runstop_pending = True
            self._runstop_wanted.set()
            if not self._runstop_done.wait(self._rpc_timeout + self._period + 0.5):
                raise EmbodimentFault(f"{self.endpoint}: the run-stop request was not confirmed")
            if self._runstop_error is not None:
                raise self._runstop_error

    def reconnect(self, timeout_s: float) -> None:
        """Rejoin after a lost link: say hello again under the same session (robot 2 ends a
        session after `heartbeat_timeout_s` without pings, and holds), then wait until a
        heartbeat gets through. Raises `LinkLost` if that hasn't happened within `timeout_s`.
        The next motion request needs a fresh `read()` first, as after any hello."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LinkLost(f"{self.endpoint}: no reply to 'hello' within {timeout_s:.1f}s")
            try:
                self._call(self._rpc, "hello", self._hello, timeout_s=min(remaining, 1.0))
                break
            except LinkLost:
                continue
        # Pings belong to the heartbeat thread (one owner per socket): wait for one to land.
        said_hello = self._clock()
        while self._last_ok < said_hello:
            if time.monotonic() >= deadline:
                raise LinkLost(
                    f"{self.endpoint}: no heartbeat within {timeout_s:.1f}s of rejoining"
                )
            time.sleep(self._period / 2)
        log.info("rejoined the Stretch 4 at %s", self.endpoint)

    def release_base(self) -> None:
        self._call(self._rpc, "release_base", {"session": self._session})

    def zero_base(self) -> None:
        self._call(self._rpc, "zero_base", {"session": self._session})

    def disconnect(self) -> None:
        """Stop the heartbeats, end the session (best effort) and close the sockets."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._rpc_timeout + self._period + 1.0)
        try:
            self._rpc.call("bye", {"session": self._session}, timeout_s=BYE_TIMEOUT_S)
        except Exception as e:
            log.info("no goodbye from %s (%s); its session will time out", self.endpoint, e)
        self._rpc.close()
        self._hb.close()

    # -- link state -------------------------------------------------------

    def link(self) -> dict[str, float | int]:
        """Clock offset, the RTT it's based on, and late replies dropped: for step info."""
        with self._offset_lock:
            return {
                "offset_s": self._offset.offset,
                "rtt_s": self._offset.rtt,
                "late_replies": self._rpc.late_replies,
            }

    # -- internals --------------------------------------------------------

    def _motion(self) -> dict[str, Any]:
        """The fields every motion request carries: a sequence number, and the robot-clock time
        after which the host has stopped waiting, so the service drops a late arrival."""
        self._seq += 1
        with self._offset_lock:
            not_after = self._clock() + self._offset.offset + self._rpc_timeout
        return {"session": self._session, "seq": self._seq, "not_after": not_after}

    def _heartbeat(self) -> None:
        """Ping on a fixed schedule; send a run-stop the moment one is asked for, and resend
        it every period until robot 2 confirms it."""
        next_ping = time.monotonic()
        failures = 0
        while not self._stop.is_set():
            if self._runstop_wanted.is_set() or self._runstop_pending:
                self._runstop_wanted.clear()
                try:
                    self._call(self._hb, "runstop", {"session": self._session}, self._rpc_timeout)
                    self._runstopped = True
                    self._runstop_at = self._clock()
                    self._runstop_pending = False
                    if failures:
                        log.warning(
                            "run-stop confirmed by %s after %d failed tries",
                            self.endpoint,
                            failures,
                        )
                    failures = 0
                except Exception as e:
                    self._runstop_error = e
                    failures += 1
                    if failures == 1:
                        log.error(
                            "run-stop request to %s failed: %s; resending until confirmed",
                            self.endpoint,
                            e,
                        )
                self._runstop_done.set()
            if time.monotonic() >= next_ping:
                next_ping = max(next_ping + self._period, time.monotonic())
                try:
                    self._ping()
                except Exception as e:
                    log.debug("ping to %s failed: %s", self.endpoint, e)
            self._runstop_wanted.wait(max(0.0, next_ping - time.monotonic()))

    def _ping(self, timeout_s: float | None = None) -> None:
        t_send = self._clock()
        header, _ = self._call(
            self._hb, "ping", {"session": self._session, "t_host": t_send}, timeout_s
        )
        t_recv = self._clock()
        with self._offset_lock:
            self._offset.add(t_send, float(header["t_robot"]), t_recv)
        reported = bool(header["runstopped"])
        engaged = reported and not self._reported_runstop
        self._reported_runstop = reported
        # A ping sent just after our run-stop may predate it on the robot: don't un-stop.
        settling = t_send < self._runstop_at + RUNSTOP_SETTLE_S
        self._runstopped = reported or (self._runstopped and settling)
        self._last_ok = t_recv
        if engaged and self._on_runstop is not None:
            try:
                self._on_runstop()
            except Exception as e:
                log.error("the run-stop callback failed: %s", e)

    def _to_local(self, t_robot: float) -> float:
        with self._offset_lock:
            return self._offset.to_local(t_robot)

    def _check_alive(self) -> None:
        silent = self._clock() - self._last_ok
        if silent > self._lost_after:
            raise LinkLost(f"lost the Stretch 4 at {self.endpoint}: no heartbeat for {silent:.1f}s")

    def _call(
        self,
        client: rpc.RpcClient,
        op: str,
        fields: dict[str, Any],
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any], list[bytes]]:
        """One request, with transport and remote errors as the framework's exceptions."""
        try:
            return client.call(op, fields, timeout_s=timeout_s)
        except rpc.RpcTimeout as e:
            raise LinkLost(str(e)) from None
        except RemoteError as e:
            if e.kind == "no_session":
                # Robot 2 ended the session after missing heartbeats, and holds: a lost link.
                raise LinkLost(f"{self.endpoint}: {e.msg}") from None
            if e.kind == "safety":
                raise SafetyAbort(f"{self.endpoint}: {e.msg}") from None
            raise EmbodimentFault(f"{self.endpoint}: {e.msg}") from None
        except ProtocolError as e:
            raise EmbodimentFault(f"{self.endpoint}: {e}") from None


def connect(
    cfg: Stretch4EmbodimentConfig,
    endpoint: str,
    *,
    keys_dir: str,
    server_name: str,
    **kwargs: Any,
) -> RemoteDriver:
    """A `RemoteDriver` with CURVE from `keys_dir`, trusting the server key named
    `server_name` (inproc endpoints, for tests, skip CURVE)."""
    from inspect_robots_stretch4.net import auth

    curve = None
    if not endpoint.startswith("inproc://"):
        try:
            curve = auth.client_curve(keys_dir, server_name)
        except auth.MissingKeys as e:
            raise EmbodimentFault(str(e)) from None
    return RemoteDriver(cfg, endpoint, curve=curve, **kwargs)

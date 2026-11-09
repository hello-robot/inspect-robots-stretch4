"""`Stretch4PairEmbodiment`: two Hello Robot Stretch 4s as one embodiment.

- Action: 12 absolute joint targets, robot 1's six then robot 2's, or 14 with `-E base=rotate`
  (`pair.spaces`).
- Safety, inside `step()` and independent of any framework approver: each robot's targets are
  clamped as one Stretch 4's are (MAX_STEP from its last measured joints, then its limits);
  then, if on, the cross guard keeps facing arms apart. Robot 2's service clamps again.
- Either robot stops both: a run-stop, or another program taking a robot's command lease, on
  one robot triggers the other's run-stop and ends the eval. A run-stop is passed on at once,
  also between steps, by the threads that already watch each robot (`_link_runstops`).
- Pacing: both robots are commanded, then read, concurrently; the join is the step barrier.

The constructor never touches hardware; both robots connect at the first `reset()`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from inspect_robots.approver import GuardrailContribution
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.errors import EmbodimentFault, SafetyAbort
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import OPERATOR_END, Action, Observation, StepResult

import inspect_robots_stretch4.driver as driver_mod
import inspect_robots_stretch4.operator as operator
from inspect_robots_stretch4.config import JOINTS, Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import BASE_TOL, DriverFactory, LinkLost, Stretch4Driver
from inspect_robots_stretch4.pair import spaces
from inspect_robots_stretch4.pair.config import SIDES, Stretch4PairConfig
from inspect_robots_stretch4.pair.docs import docs
from inspect_robots_stretch4.pair.guard import APPROVER_NAME, CrossGuardApprover, cross_guard
from inspect_robots_stretch4.pair.rig import Rig, load_rig
from inspect_robots_stretch4.side import RobotSide

log = logging.getLogger(__name__)

RUNSTOP_POLL_S = 0.1
GUARD_OFF = "cross-robot collision guard is OFF (-E cross_guard=false)"
# How often a lost link is retried, and robot 1's run-stop checked, while both robots hold.
LINK_POLL_S = 0.05
ARM = JOINTS.index("arm")
BASE = len(JOINTS)  # base_theta follows the six arm joints when a base turns


class Stretch4PairEmbodiment:
    """Inspect Robots embodiment for two Hello Robot Stretch 4s."""

    RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
        **driver_mod.RUNTIME_REQUIREMENTS,
        "zmq": "pip install 'inspect-robots-stretch4[pair]' (pyzmq, to reach robot 2)",
    }

    def __init__(
        self,
        config: Stretch4PairConfig | None = None,
        *,
        driver_factories: dict[str, DriverFactory] | None = None,
        load_home_pose: Callable[[str], dict[str, float]] | None = None,
        input_fn: Callable[[str], str] | None = None,
        poll_end: Callable[[], bool] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else Stretch4PairConfig.from_kwargs(**flat)
        self._rig: Rig | None = load_rig(self._cfg.rig_file) if self._cfg.rig_file else None
        given = driver_factories or {}
        factories = {s: given.get(s) or self._default_factory(s) for s in SIDES}
        self._sides = {
            s: RobotSide(
                self._cfg.side(s),
                factory=factories[s],
                load_home_pose=load_home_pose or driver_mod.load_home_pose,
                name=s,
            )
            for s in SIDES
        }
        self._input_fn = input_fn or input
        self._poll_end = poll_end or operator.poll_end
        self._session: Any = None
        self._log = operator.OperatorLog(lambda: self._session)
        self._pool: ThreadPoolExecutor | None = None
        self._instruction: str | None = None
        self._pressed: str | None = None  # the robot whose run-stop engaged first
        self.num_steps = 0

        self.info = EmbodimentInfo(
            name="stretch4-pair",
            action_space=spaces.action_box(self._cfg),
            observation_space=spaces.observation_space(self._cfg),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs(self._cfg, self._rig),
            environment_id="hello-robot-stretch4-pair",
        )

    # -- framework hooks ---------------------------------------------------

    def connect_operator_session(self, session: Any) -> None:
        """Hand the terminal to the framework: it gates readiness and ends episodes (Esc)."""
        self._session = session

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        """The cross guard as an approver, so the CLI's banner shows it (step() applies it too,
        for direct rollout() callers); or a warning that it's off."""
        if not self._guarding:
            return GuardrailContribution(warnings=(GUARD_OFF,))
        assert self._rig is not None
        return GuardrailContribution(
            approvers=((APPROVER_NAME, CrossGuardApprover(self._rig, self._current)),)
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect both (first time), home both, wait for the operator, zero turning bases,
        observe both. A link lost outside homing is ridden out as in `step()`."""
        del seed  # real robots are not seedable
        if not all(side.connected for side in self._sides.values()):
            self._log.attach()  # before connecting: the drivers report camera progress
            self._connect()
        self._riding_out(self._check_runstops)
        self._home()
        if not self._cfg.auto_start:
            if self._session is not None:
                self._session.gate(operator.READY_PROMPT, hint=operator.READY_HINT)
            else:
                operator.wait_ready(self._input_fn)
        self._riding_out(self._check_runstops)
        if self._cfg.base != "none":
            # Each trial's base_theta 0 is where the bases stand now (they aren't pushed into
            # place, so they aren't released first as one robot's are).
            self._riding_out(lambda: self._both(lambda side: side.driver.zero_base()))
        self._instruction = scene.instruction
        self.num_steps = 0
        readings = self._riding_out(lambda: self._both(RobotSide.read))
        return spaces.merge(readings, self._cfg, self._instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp and send both robots' targets, then return the next synced observation.

        If the link to a served robot is lost, both robots hold while it is retried
        (`link_retries` attempts of `link_retry_s`); once it is back, this step returns a fresh
        observation of both without moving them, and the run carries on."""
        if not all(side.connected for side in self._sides.values()):
            raise RuntimeError("step() called before reset()")
        self.num_steps += 1
        data = np.asarray(action.data, dtype=np.float64).reshape(-1)
        # The model's joint values -> each wrist's own (a left-handed wrist's differ).
        per_side = {
            s: spaces.to_robot(part, self._cfg.left_handed(s))
            for s, part in spaces.split(data, self._cfg).items()
        }
        if not np.all(np.isfinite(data)):
            raise SafetyAbort("stretch4-pair: the action has non-finite values; not sending it")
        try:
            return self._step(per_side)
        except LinkLost as lost:
            paused = self._ride_out(lost)
        self._check_runstops()
        readings = self._both(RobotSide.read)
        self._check_runstops()
        obs = spaces.merge(readings, self._cfg, self._instruction)
        info = {
            "skew_s": abs(readings["robot1"].t - readings["robot2"].t),
            "link_pause_s": paused,
            **self._link_info(),
        }
        return StepResult(observation=obs, info=info)

    def _step(self, per_side: dict[str, npt.NDArray[np.float64]]) -> StepResult:
        self._check_runstops()  # before commanding either
        safe = {s: self._sides[s].clamp(per_side[s]) for s in SIDES}
        base_held = self._hold_bases_unless_arms_in(safe)
        clamped = spaces.join(safe)
        target, guarded = clamped, False
        if self._guarding:
            assert self._rig is not None
            current = self._current()
            assert current is not None  # reset() read both
            target = cross_guard(clamped, current, self._rig)
            guarded = not np.array_equal(target, clamped)
        split = spaces.split(target, self._cfg)
        # Any failure commanding one robot stops the other: one arm moving while the other
        # doesn't drops (or strains) whatever both are holding. Except a lost
        # link: that robot holds, the other gets nothing more, and step() rides it out.
        sent = self._both(
            lambda side: side.command(split[side.name]), stop_on=(Exception,), spare=(LinkLost,)
        )
        self._check_runstops()
        readings = self._both(RobotSide.read)
        self._check_runstops()
        obs = spaces.merge(readings, self._cfg, self._instruction)
        info = {
            "sent": spaces.join(sent).tolist(),
            "skew_s": abs(readings["robot1"].t - readings["robot2"].t),
            "cross_guard": guarded,
            "base_held": base_held,
            **self._link_info(),
        }
        if self._session is None and self._poll_end():
            return StepResult(
                observation=obs, terminated=True, termination_reason=OPERATOR_END, info=info
            )
        return StepResult(observation=obs, info=info)

    def close(self) -> None:
        """Disconnect both (idempotent; each handle is dropped even if disconnecting fails).

        Waits for any robot call still running first (a driver isn't closed under a thread
        using it): at worst a robot 2 goto whose host link died, up to its timeout + 5 s.
        """
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            errors = []
            for side in self._sides.values():
                try:
                    side.close()
                except Exception as e:
                    errors.append(e)
            if errors:
                raise errors[0]
        finally:
            self._log.detach()

    # -- internals ---------------------------------------------------------

    @property
    def _guarding(self) -> bool:
        return self._cfg.cross_guard and self._rig is not None

    def _current(self) -> npt.NDArray[np.float64] | None:
        """Both robots' last measured joints, robot 1 first; None before the first read."""
        joints = {s: side.joints for s, side in self._sides.items()}
        if any(j is None for j in joints.values()):
            return None
        return spaces.join(joints)  # type: ignore[arg-type]

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stretch4-pair")
        return self._pool

    def _both(
        self,
        fn: Callable[[RobotSide], Any],
        stop_on: tuple[type[BaseException], ...] = (SafetyAbort,),
        spare: tuple[type[BaseException], ...] = (),
    ) -> dict[str, Any]:
        """`fn` on both robots at once. If either fails, the error names its robot; a failure
        of a `stop_on` kind (and not a `spare` kind) triggers the other robot's run-stop before
        it is raised."""
        futures = {s: self._executor().submit(fn, side) for s, side in self._sides.items()}
        return self._gather(futures, stop_on, spare)

    def _gather(
        self,
        futures: dict[str, Future[Any]],
        stop_on: tuple[type[BaseException], ...] = (SafetyAbort,),
        spare: tuple[type[BaseException], ...] = (),
    ) -> dict[str, Any]:
        results, errors = {}, {}
        for s, future in futures.items():
            try:
                results[s] = future.result()
            except Exception as e:
                errors[s] = e
        if not errors:
            return results
        for s, e in errors.items():
            if isinstance(e, stop_on) and not isinstance(e, spare):
                self._stop_all_but(s)
        safety = {s: e for s, e in errors.items() if isinstance(e, SafetyAbort)}
        name, error = next(iter(safety.items())) if safety else next(iter(errors.items()))
        named = _named(name, error)
        if named is error:
            raise error
        raise named from error

    def _check_runstops(self) -> None:
        """SafetyAbort naming the robot whose run-stop is engaged, after stopping the other.
        Every robot is checked before a lost link is raised, so a run-stop is never hidden by
        the other robot's link."""
        stopped: list[str] = []
        lost: LinkLost | None = None
        for s, side in self._sides.items():
            try:
                if side.driver.is_runstopped():
                    stopped.append(s)
            except LinkLost as e:
                lost = lost or e
        if not stopped:
            if lost is not None:
                raise lost
            self._pressed = None
            return
        s = self._pressed if self._pressed in stopped else stopped[0]
        self._pressed = s
        self._stop_all_but(s)
        raise SafetyAbort(
            f"{s}'s run-stop is engaged, so both robots were stopped; release both "
            "run-stops and rerun"
        )

    def _link_runstops(self) -> None:
        """Each robot's run-stop engaging asks for the other's at once, from the driver thread
        that saw it (robot 1's status thread, robot 2's heartbeat thread): between steps too."""
        for s, side in self._sides.items():
            on_runstop = getattr(side.driver, "on_runstop", None)
            if callable(on_runstop):
                on_runstop(lambda s=s: self._runstop_others(s))

    def _runstop_others(self, pressed: str) -> None:
        """From a driver thread: must not block, so it only requests the others' run-stops."""
        if self._pressed is None:
            self._pressed = pressed
        for s, side in self._sides.items():
            if s == pressed or not side.connected:
                continue
            request = getattr(side.driver, "request_runstop", None)
            try:
                if callable(request):
                    request()
                else:
                    side.trigger_runstop()
            except Exception as e:
                log.error("could not run-stop %s after %s stopped: %s", s, pressed, e)

    def _stop_all_but(self, name: str | None) -> None:
        """Run-stop every connected robot except `name` (every one, for None)."""
        for s, side in self._sides.items():
            if s == name or not side.connected:
                continue
            try:
                side.trigger_runstop()
            except Exception as e:
                log.error("could not run-stop %s after %s stopped: %s", s, name, e)

    def _connect(self) -> None:
        """Connect both at once; if either fails, close the other and re-raise."""
        futures = {s: self._executor().submit(side.connect) for s, side in self._sides.items()}
        try:
            # Neither robot has moved: a failure to connect one is no reason to stop the other.
            self._gather(futures, stop_on=())
        except BaseException:
            for side in self._sides.values():
                try:
                    side.close()
                except Exception as e:
                    log.error("closing %s after a failed connect: %s", side.name, e)
            raise
        self._link_runstops()
        self._warn_on_decoder_mismatch()

    def _home(self) -> None:
        """Home both at once. A run-stop or any failure on one while they move stops the
        other too, which ends its goto. Losing track of them (the link check fails, or
        Ctrl-C) stops both. Both home poses are read and checked first, before either moves, and
        a failure is reported as itself, not as the run-stop it caused on the other."""
        for side in self._sides.values():
            side.home_pose()
        futures = {s: self._executor().submit(side.home) for s, side in self._sides.items()}
        stopped: str | None = None
        try:
            while not all(f.done() for f in futures.values()):
                if stopped is None:
                    failed = (s for s, f in futures.items() if f.done() and f.exception())
                    runstopped = [
                        s for s, side in self._sides.items() if side.driver.is_runstopped()
                    ]
                    if self._pressed in runstopped:
                        runstopped.insert(0, self._pressed)
                    stopped = next(failed, None) or next(iter(runstopped), None)
                    if stopped is not None:
                        self._stop_all_but(stopped)
                time.sleep(RUNSTOP_POLL_S)
        except BaseException:
            self._stop_all_but(None)
            raise
        cause = futures[stopped].exception() if stopped is not None else None
        if cause is not None and not isinstance(cause, SafetyAbort):
            named = _named(stopped, cause)
            if named is cause:
                raise cause
            raise named from cause
        self._gather(futures, stop_on=() if stopped else (Exception,))
        if stopped is not None:
            raise SafetyAbort(f"{stopped}'s run-stop was engaged while homing; both stopped")

    def _hold_bases_unless_arms_in(self, targets: dict[str, npt.NDArray[np.float64]]) -> list[str]:
        """A base turns only while its arm is retracted to base_turn_max_arm, now and in the
        target: otherwise its target is replaced by its measured heading, so it holds. Edits
        `targets` in place; returns the robots whose base was held."""
        held: list[str] = []
        if self._cfg.base == "none":
            return held
        limit = self._cfg.base_turn_max_arm
        for s, target in targets.items():
            current = self._sides[s].joints
            if current is None or abs(target[BASE] - current[BASE]) <= BASE_TOL:
                continue
            if current[ARM] > limit or target[ARM] > limit:
                target[BASE] = current[BASE]
                held.append(s)
        return held

    def _riding_out(self, fn: Callable[[], Any]) -> Any:
        """`fn()`; if the link to a served robot is lost, hold both while it is retried
        (`_ride_out`), then `fn()` once more. Only for calls that move nothing."""
        try:
            return fn()
        except LinkLost as lost:
            self._ride_out(lost)
        return fn()

    def _ride_out(self, lost: LinkLost) -> float:
        """Hold both robots (nothing is commanded) and try to rejoin every served robot:
        `link_retries` attempts of up to `link_retry_s` each. A local robot's run-stop pressed
        meanwhile stops both. Returns the seconds paused; raises a final EmbodimentFault if the
        link is still down after the last attempt."""
        served = {
            s: side.driver
            for s, side in self._sides.items()
            if callable(getattr(side.driver, "reconnect", None))
        }
        attempts, wait = self._cfg.link_retries, self._cfg.link_retry_s
        if not served or attempts == 0:
            raise EmbodimentFault(str(lost)) from lost
        local = [s for s in self._sides if s not in served]
        t0 = time.monotonic()
        for attempt in range(1, attempts + 1):
            log.warning(
                "%s; both robots hold. Reconnecting: attempt %d of %d (up to %.0f s)",
                lost,
                attempt,
                attempts,
                wait,
            )
            deadline = time.monotonic() + wait
            while True:
                for s in local:
                    if self._sides[s].driver.is_runstopped():
                        self._stop_all_but(s)
                        raise SafetyAbort(
                            f"{s}'s run-stop is engaged, so both robots were stopped; release "
                            "both run-stops and rerun"
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    for driver in served.values():
                        driver.reconnect(timeout_s=remaining)
                except LinkLost:
                    time.sleep(LINK_POLL_S)
                    continue
                paused = time.monotonic() - t0
                log.warning("the link is back after %.1f s; carrying on", paused)
                return paused
        raise EmbodimentFault(
            f"{lost} (still lost after {attempts} attempts of {wait:g} s; both robots hold)"
        ) from lost

    def _link_info(self) -> dict[str, float]:
        info = {}
        for s, side in self._sides.items():
            link = getattr(side.driver, "link", None)
            if callable(link):
                values = link()
                info[f"{s}_rtt_s"] = values["rtt_s"]
                info[f"{s}_offset_s"] = values["offset_s"]
        return info

    def _warn_on_decoder_mismatch(self) -> None:
        """Robot 2's images are decoded here, so any decoder difference shows up as a pixel
        difference between the robots: say so if the OpenCV versions differ."""
        try:
            import cv2
        except ImportError:  # pragma: no cover - cv2 is a runtime requirement
            return
        for s, side in self._sides.items():
            remote = getattr(side.driver, "remote_cv2", None)
            if remote is not None and remote != cv2.__version__:
                log.warning(
                    "%s's OpenCV is %s, this machine's %s: its forwarded images are decoded "
                    "here, so this is fine, but lossless image_mode would decode there",
                    s,
                    remote,
                    cv2.__version__,
                )

    def _default_factory(self, name: str) -> DriverFactory:
        cfg = self._cfg
        host = getattr(cfg, f"{name}_host")

        def build(side_cfg: Stretch4EmbodimentConfig) -> Stretch4Driver:
            if host is None:
                if name == "robot2":
                    raise EmbodimentFault(
                        "robot2_host is not set; pass -E robot2_host=tcp://<robot 2's IP>:5600, "
                        "where stretch_inspect_pair_server is listening"
                    )
                return driver_mod.BodyDriver(side_cfg)
            # Only now: pyzmq is needed for a remote robot, not to import or inspect the pair.
            from inspect_robots_stretch4.remote.client import connect

            return connect(
                side_cfg,
                host,
                keys_dir=cfg.keys_dir,
                server_name=name,
                rpc_timeout_s=cfg.rpc_timeout_s,
                heartbeat_timeout_s=cfg.heartbeat_timeout_s,
                image_mode=cfg.image_mode,
            )

        return build


def _named(name: str, error: BaseException) -> BaseException:
    """`error`, its message prefixed with the robot it came from (once)."""
    if isinstance(error, SafetyAbort | EmbodimentFault) and not str(error).startswith(name):
        return type(error)(f"{name}: {error}")
    return error

"""`Stretch4Embodiment`: the Hello Robot Stretch 4 as an Inspect Robots embodiment.

- Action: the six arm joints, then the configured base pose (`-E base=`), as absolute targets
  (`cfg.joints`). The base pose is relative to where the operator placed the robot for the
  trial. The head is not commanded.
- Safety: inside `step()`, independent of any framework approver, every target is clamped to
  the configured joint limits and to at most `MAX_STEP` from the last measured joints. (The
  framework's DeltaLimitApprover lets each trial's first action through unlimited.) The driver
  moves each joint with speed-limited `move_to` and clamps the gripper to its own aperture
  range. An engaged run-stop, or another program holding the control lock, halts the eval.
- Pacing: each step waits for the next synced camera frame, so steps run at the camera rate
  (`SELF_PACED`).
- Reset: the arm drives to the home pose; then, if the base is commanded, the wheels freewheel
  so the operator can push the robot onto its marks; the operator starts the trial, and the
  base holds and is zeroed there.
- Episode end: the operator, through the framework's session (Esc) or, without one, by pressing
  Enter. There is no automatic success check.

The constructor never touches hardware (so `inspect-robots doctor` and the conformance test can
build one anywhere); the robot connects at the first `reset()`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import OPERATOR_END, Action, Observation, StepResult

from inspect_robots_stretch4 import driver as driver_mod
from inspect_robots_stretch4 import operator
from inspect_robots_stretch4.config import (
    STATE_KEY,
    Stretch4EmbodimentConfig,
    action_box,
    docs,
    observation_space,
)
from inspect_robots_stretch4.driver import DriverFactory, Reading
from inspect_robots_stretch4.side import RobotSide


class Stretch4Embodiment:
    """Inspect Robots embodiment for a Hello Robot Stretch 4 (arm joints and base pose)."""

    RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = driver_mod.RUNTIME_REQUIREMENTS

    def __init__(
        self,
        config: Stretch4EmbodimentConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        load_home_pose: Callable[[str], dict[str, float]] | None = None,
        input_fn: Callable[[str], str] | None = None,
        poll_end: Callable[[], bool] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else Stretch4EmbodimentConfig.from_kwargs(**flat)
        self._side = RobotSide(
            self._cfg,
            factory=driver_factory or driver_mod.BodyDriver,
            load_home_pose=load_home_pose or driver_mod.load_home_pose,
        )
        self._input_fn = input_fn or input
        self._poll_end = poll_end or operator.poll_end
        self._session: Any = None
        self._log = operator.OperatorLog(lambda: self._session)
        self._instruction: str | None = None
        self.num_steps = 0

        self.info = EmbodimentInfo(
            name="stretch4",
            action_space=action_box(self._cfg),
            observation_space=observation_space(self._cfg),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs(self._cfg),
            environment_id="hello-robot-stretch4",
        )

    # -- framework hooks ---------------------------------------------------

    def connect_operator_session(self, session: Any) -> None:
        """Hand the terminal to the framework: it gates readiness and ends episodes (Esc).

        From here on this embodiment never reads stdin or prints.
        """
        self._session = session

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect (first time), drive the arm home, let the operator place the base and start,
        zero the base, observe."""
        del seed  # a real robot is not seedable
        side = self._side
        if not side.connected:
            self._log.attach()  # before connecting: the driver reports camera progress
        side.connect()
        side.check_runstop()
        side.home()
        moving_base = self._cfg.base != "none"
        if moving_base and not self._cfg.auto_start:
            side.driver.release_base()
        if not self._cfg.auto_start:
            if self._session is not None:
                self._session.gate(operator.READY_PROMPT, hint=operator.READY_HINT)
            else:
                operator.wait_ready(self._input_fn)
        side.check_runstop()
        if moving_base:
            side.driver.zero_base()
        self._instruction = scene.instruction
        self.num_steps = 0
        return self._observation(side.read())

    def step(self, action: Action) -> StepResult:
        """Clamp and send one joint target, then return the next synced observation."""
        side = self._side
        if not side.connected:
            raise RuntimeError("step() called before reset()")
        self.num_steps += 1
        data = np.asarray(action.data, dtype=np.float64).reshape(-1)
        n = len(self._cfg.joints)
        if data.shape != (n,):
            raise ValueError(f"stretch4 actions have {n} values, got shape {data.shape}")
        target = side.clamp(data)
        side.check_runstop()
        sent = side.command(target)
        side.check_runstop()
        obs = self._observation(side.read())
        info = {"sent": sent.tolist()}
        if self._session is None and self._poll_end():
            return StepResult(
                observation=obs, terminated=True, termination_reason=OPERATOR_END, info=info
            )
        return StepResult(observation=obs, info=info)

    def close(self) -> None:
        """Disconnect (idempotent; the handle is dropped even if disconnecting fails)."""
        try:
            self._side.close()
        finally:
            self._log.detach()

    # -- internals ---------------------------------------------------------

    def _observation(self, reading: Reading) -> Observation:
        images = {name: reading.images[name] for name in self._cfg.cameras}
        # `<camera>_depth` extras are what depth-aware policies (the LLM agent) look for.
        extra = {f"{name}_depth": d for name, d in reading.depth.items()}
        return Observation(
            images=images,
            state={STATE_KEY: np.asarray(reading.joints, dtype=np.float64)},
            instruction=self._instruction,
            image_times=dict.fromkeys(images, reading.t),
            state_time=reading.t,
            extra=extra,
        )

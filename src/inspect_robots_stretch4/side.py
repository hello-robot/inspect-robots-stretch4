"""`RobotSide`: one Stretch 4 as an embodiment sees it.

The per-robot half of an embodiment: its driver, joint limits, home pose and last measured
joints, and the safety backstop between a policy's target and the robot. `Stretch4Embodiment`
holds one; the two-robot embodiment holds two, named so their errors say which robot.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4.config import JOINTS, Stretch4EmbodimentConfig, clamp
from inspect_robots_stretch4.driver import DriverFactory, Reading, Stretch4Driver


class RobotSide:
    """One robot's driver, limits and last measurement. The constructor touches no hardware."""

    def __init__(
        self,
        cfg: Stretch4EmbodimentConfig,
        *,
        factory: DriverFactory,
        load_home_pose: Callable[[str], dict[str, float]],
        name: str = "",
    ) -> None:
        self.cfg = cfg
        self.name = name
        self._factory = factory
        self._load_home_pose = load_home_pose
        self._driver: Stretch4Driver | None = None
        self.joints: npt.NDArray[np.float64] | None = None  # last measured, cfg.joints order

    @property
    def connected(self) -> bool:
        return self._driver is not None

    def connect(self) -> Stretch4Driver:
        """The driver, built on first use."""
        if self._driver is None:
            self._driver = self._factory(self.cfg)
        return self._driver

    def check_runstop(self) -> None:
        if self.driver.is_runstopped():
            raise SafetyAbort(
                self._label("the Stretch 4 run-stop is engaged; release it and rerun")
            )

    def trigger_runstop(self) -> None:
        self.driver.trigger_runstop()

    def home_pose(self) -> dict[str, float] | None:
        """The configured home pose, read and checked against the joint limits; None if
        there is none. Moves nothing, so it can run before either robot moves."""
        path = self.cfg.home_pose_file
        if not path:
            return None
        try:
            loaded = self._load_home_pose(path)
        except (OSError, ValueError) as e:
            raise EmbodimentFault(self._label(f"can't read the home pose {path!r}: {e}")) from e
        # A pose outside the configured limits is refused, not clamped: clamped, it is a
        # different pose (a right-handed wrist pose clamped for a left-handed wrist folds it).
        limits = zip(JOINTS, self.cfg.joint_low, self.cfg.joint_high, strict=True)
        outside = [
            f"{j} {loaded[j]:.3f} not in {lo:.3f}..{hi:.3f}"
            for j, lo, hi in limits
            if not lo - 1e-6 <= loaded[j] <= hi + 1e-6
        ]
        if outside:
            raise EmbodimentFault(
                self._label(
                    f"the home pose {path!r} is outside the joint limits: {'; '.join(outside)}"
                )
            )
        return loaded

    def home(self) -> None:
        """Drive slowly to the configured home pose, if there is one."""
        pose = self.home_pose()
        if pose is None:
            return
        path = self.cfg.home_pose_file
        arrived, error = self.driver.goto(pose, self.cfg.home_timeout_s)
        if not arrived:
            raise EmbodimentFault(
                self._label(
                    f"did not reach the home pose {path!r} within "
                    f"{self.cfg.home_timeout_s:.0f}s (worst remaining error: {error})"
                )
            )

    def clamp(self, target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The hard safety backstop: within MAX_STEP of the last measured joints, then the
        joint limits."""
        if self.joints is None:
            raise RuntimeError("clamp() before the first read()")
        return clamp(target, self.joints, self.cfg)

    def command(self, target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Send an already-clamped target; return what was sent, in `cfg.joints` order."""
        joints = self.cfg.joints
        sent = self.driver.command(dict(zip(joints, target.tolist(), strict=True)))
        return np.array([float(sent[j]) for j in joints], dtype=np.float64)

    def read(self) -> Reading:
        """The next synced reading, with every configured camera at its configured size."""
        reading = self.driver.read()
        for camera in self.cfg.cameras:
            expected = (*self.cfg.image_shape(camera), 3)
            shape = reading.images[camera].shape
            if shape != expected:
                raise EmbodimentFault(
                    self._label(f"camera {camera!r} gave shape {shape}, expected {expected}")
                )
        self.joints = np.asarray(reading.joints, dtype=np.float64)
        return reading

    def close(self) -> None:
        """Disconnect (idempotent; the handle is dropped even if disconnecting fails)."""
        if self._driver is not None:
            try:
                self._driver.disconnect()
            finally:
                self._driver = None

    @property
    def driver(self) -> Stretch4Driver:
        if self._driver is None:
            raise RuntimeError("step() called before reset()")
        return self._driver

    def _label(self, message: str) -> str:
        return f"{self.name}: {message}" if self.name else message

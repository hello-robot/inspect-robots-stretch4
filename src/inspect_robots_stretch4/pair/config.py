"""`Stretch4PairConfig`: two Stretch 4s as one embodiment.

Robot 1 runs the eval (its `BodyDriver` is in this process unless `robot1_host` is set); robot 2
is reached at `robot2_host` through `stretch_inspect_pair_server`. Every field is a scalar or a
tuple, so `inspect-robots run -E key=value` can set it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Literal

from inspect_robots_stretch4.config import (
    BASE_MODES,
    CAMERA_NATIVE_HW,
    DEFAULT_HIGH,
    DEFAULT_LOW,
    DEFAULT_MAX_IMAGE_SIDE,
    DEPTH_CAMERA,
    GRIPPER_OPEN_MM,
    IMAGE_MODES,
    JOINTS,
    Stretch4EmbodimentConfig,
)

SIDES: tuple[Literal["robot1", "robot2"], ...] = ("robot1", "robot2")
# The wrist joints a pair config can narrow per robot.
NARROWABLE: tuple[str, ...] = ("wrist_yaw", "wrist_roll")
HANDEDNESS = ("right", "left")
# A left-handed wrist (folded over, stretch4_body's tool_right_handed False) keeps wrist_pitch
# above this, past the pi/2 singularity, so it can't unfold to right-handed. It allows the
# right-handed equivalent of pitch 1.50: just past straight down.
LEFT_PITCH_MIN = math.pi - 1.50
# net.auth.DEFAULT_KEYS_DIR (pair can't import net; a test keeps the two equal).
DEFAULT_KEYS_DIR = "~/.config/inspect_robots_stretch4/keys"


def device_of(camera: str) -> str:
    """'head_right' -> 'head'; 'gripper_left' -> 'gripper'."""
    return camera.split("_", 1)[0]


@dataclass(frozen=True)
class Stretch4PairConfig:
    """Static configuration for the two-robot embodiment."""

    # None: robot 1's BodyDriver runs in this process. Else tcp://<ip>:<port>.
    robot1_host: str | None = None
    # tcp://<robot 2's IP>:<port>, where `stretch_inspect_pair_server` listens.
    robot2_host: str | None = None
    # One eye per camera device per robot: the head and the gripper camera.
    robot1_cameras: tuple[str, ...] = ("head_right", "gripper_right")
    robot2_cameras: tuple[str, ...] = ("head_right", "gripper_right")
    robot1_gripper: str = "parallel_gripper"
    robot2_gripper: str = "parallel_gripper"
    robot1_home_pose_file: str | None = None
    robot2_home_pose_file: str | None = None
    # "left": that robot's wrist is folded over (stretch4_body's left-handed home is yaw, pitch,
    # roll = pi, pi, pi) and stays so. Its actions and readings keep the right-handed meanings
    # (yaw 0 straight out, pitch 0 level, roll 0 camera on top), converted to the folded joints.
    robot1_handedness: str = "right"
    robot2_handedness: str = "right"
    # Narrow a robot's wrist_yaw or wrist_roll range (rad, the robot's own joint values; None
    # keeps the default).
    robot1_wrist_yaw_min: float | None = None
    robot1_wrist_yaw_max: float | None = None
    robot1_wrist_roll_min: float | None = None
    robot1_wrist_roll_max: float | None = None
    robot2_wrist_yaw_min: float | None = None
    robot2_wrist_yaw_max: float | None = None
    robot2_wrist_roll_min: float | None = None
    robot2_wrist_roll_max: float | None = None
    # Each wrist's metric depth, aligned to its gripper_right eye.
    gripper_depth: bool = True
    # Each robot's base: "none" (held still) or "rotate" (turns in place: robotN_base_theta).
    base: str = "none"
    # A turning base stays within this of its start heading (rad, default 180 degrees)...
    base_max_theta: float = math.pi
    # ...and turns only while its arm is retracted to this (m), now and in the target, so an
    # extended arm never sweeps through the table, objects or the other robot.
    base_turn_max_arm: float = 0.15
    # The rig file: the measured layout, for the docs and the cross-robot guard.
    rig_file: str | None = None
    # The opposed layout only: keep the two arms from meeting when their wrists are level and
    # facing (needs rig_file; refused with turning bases or a left-handed wrist).
    cross_guard: bool = False
    # CURVE keys; the host trusts each served robot's key under its side name.
    keys_dir: str = DEFAULT_KEYS_DIR
    rpc_timeout_s: float = 1.0
    heartbeat_timeout_s: float = 1.0
    # A lost link to robot 2 holds both robots and retries: this many attempts, each waiting up
    # to link_retry_s for robot 2 to answer, before the run ends (0: end at once).
    link_retries: int = 3
    link_retry_s: float = 5.0
    # How a served robot sends images: "forward" its camera's JPEGs, or "lossless" PNG.
    image_mode: str = "forward"
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE
    control_hz: float = 30.0
    home_timeout_s: float = 40.0
    auto_start: bool = False
    priority: int = 2

    def __post_init__(self) -> None:
        for side in SIDES:
            cameras = getattr(self, f"{side}_cameras")
            if not cameras:
                raise ValueError(f"{side}_cameras needs at least one camera")
            unknown = [c for c in cameras if c not in CAMERA_NATIVE_HW]
            if unknown:
                raise ValueError(
                    f"{side}_cameras has unknown cameras {unknown}; "
                    f"choose from {sorted(CAMERA_NATIVE_HW)}"
                )
            devices = [device_of(c) for c in cameras]
            if len(set(devices)) != len(devices):
                raise ValueError(
                    f"{side}_cameras takes one eye per camera ({', '.join(cameras)}): the head "
                    "and the gripper are one camera each, and a second eye adds no view"
                )
            if self.gripper_depth and DEPTH_CAMERA not in cameras:
                raise ValueError(
                    f"gripper_depth needs {DEPTH_CAMERA} in {side}_cameras (the depth is "
                    "aligned to it); add it, or pass -E gripper_depth=false"
                )
            gripper = getattr(self, f"{side}_gripper")
            if gripper not in GRIPPER_OPEN_MM:
                raise ValueError(
                    f"{side}_gripper must be one of {sorted(GRIPPER_OPEN_MM)}, got {gripper!r}"
                )
            hand = getattr(self, f"{side}_handedness")
            if hand not in HANDEDNESS:
                raise ValueError(f"{side}_handedness must be one of {HANDEDNESS}, got {hand!r}")
            if hand == "left" and self.cross_guard:
                raise ValueError(
                    f"the cross-robot collision guard assumes right-handed wrists: pass "
                    f"-E cross_guard=false with -E {side}_handedness=left"
                )
            for joint in NARROWABLE:
                lo, hi = self.joint_range(side, joint)
                i = JOINTS.index(joint)
                if not (DEFAULT_LOW[i] <= lo < hi <= DEFAULT_HIGH[i]):
                    raise ValueError(
                        f"{side}_{joint}_min/max must narrow the default range "
                        f"{DEFAULT_LOW[i]:.2f}..{DEFAULT_HIGH[i]:.2f} rad, got {lo:.2f}..{hi:.2f}"
                    )
        if self.cross_guard and not self.rig_file:
            raise ValueError(
                "the cross-robot collision guard needs the measured rig: pass "
                "-E rig_file=<pair_rig.json> (made by stretch_inspect_pair_calibrate), or turn the "
                "guard off with -E cross_guard=false"
            )
        if self.base not in BASE_MODES:
            raise ValueError(f"base must be one of {sorted(BASE_MODES)}, got {self.base!r}")
        if self.base != "none" and self.cross_guard:
            raise ValueError(
                "the cross-robot collision guard assumes the bases don't turn: pass "
                "-E cross_guard=false with -E base=rotate"
            )
        if not math.isfinite(self.base_turn_max_arm) or self.base_turn_max_arm < 0:
            raise ValueError(f"base_turn_max_arm must be >= 0, got {self.base_turn_max_arm!r}")
        if self.image_mode not in IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {IMAGE_MODES}, got {self.image_mode!r}")
        if isinstance(self.link_retries, bool) or not isinstance(self.link_retries, int):
            raise ValueError(f"link_retries must be a whole number, got {self.link_retries!r}")
        if self.link_retries < 0:
            raise ValueError(f"link_retries must be >= 0, got {self.link_retries}")
        for name in (
            "rpc_timeout_s",
            "heartbeat_timeout_s",
            "home_timeout_s",
            "control_hz",
            "link_retry_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        self.side("robot1"), self.side("robot2")  # each side's own checks

    def left_handed(self, name: Literal["robot1", "robot2"]) -> bool:
        return bool(getattr(self, f"{name}_handedness") == "left")

    def joint_range(self, name: Literal["robot1", "robot2"], joint: str) -> tuple[float, float]:
        """One side's limits (rad) for a NARROWABLE joint."""
        lo = getattr(self, f"{name}_{joint}_min")
        hi = getattr(self, f"{name}_{joint}_max")
        i = JOINTS.index(joint)
        return (
            DEFAULT_LOW[i] if lo is None else float(lo),
            DEFAULT_HIGH[i] if hi is None else float(hi),
        )

    def side(self, name: Literal["robot1", "robot2"]) -> Stretch4EmbodimentConfig:
        """The single-robot config for one side."""
        gripper = getattr(self, f"{name}_gripper")
        low = list(DEFAULT_LOW)
        high = [*DEFAULT_HIGH[:-1], GRIPPER_OPEN_MM.get(gripper, DEFAULT_HIGH[-1])]
        for joint in NARROWABLE:
            i = JOINTS.index(joint)
            low[i], high[i] = self.joint_range(name, joint)
        if self.left_handed(name):
            pitch = JOINTS.index("wrist_pitch")
            low[pitch] = max(low[pitch], LEFT_PITCH_MIN)
        return Stretch4EmbodimentConfig(
            joint_low=tuple(low),
            joint_high=tuple(high),
            cameras=getattr(self, f"{name}_cameras"),
            max_image_side=self.max_image_side,
            control_hz=self.control_hz,
            home_pose_file=getattr(self, f"{name}_home_pose_file"),
            home_timeout_s=self.home_timeout_s,
            auto_start=self.auto_start,
            gripper=getattr(self, f"{name}_gripper"),
            priority=self.priority,
            gripper_depth=self.gripper_depth,
            base=self.base,
            base_max_theta=self.base_max_theta,
        )

    @classmethod
    def from_kwargs(cls, **flat: Any) -> Stretch4PairConfig:
        """Build from flat CLI kwargs, rejecting unknown keys loudly."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"stretch4-pair got unexpected config keys: {sorted(unknown)}")
        # The CLI parses `-E base=none` to None; "none" is the mode the user meant.
        if "base" in flat and flat["base"] is None:
            flat["base"] = "none"
        for key in ("robot1_cameras", "robot2_cameras"):
            if isinstance(flat.get(key), str):
                flat[key] = tuple(p.strip() for p in flat[key].split(",") if p.strip())
        return cls(**flat)

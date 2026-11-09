"""Configuration and spaces for the Stretch 4 embodiment.

The action is the arm's six joints as absolute targets, then, by default, the base's heading
as an absolute target relative to its heading at the start of the trial:

    lift (m), arm (m), wrist_yaw, wrist_pitch, wrist_roll (rad), grip_mm (fingertip aperture, mm),
    [base_theta (rad)]

`base` is "rotate" (base_theta, the default) or "none" (base held still). The base only turns
in place. The head is not commanded; it turns with the base.

Every config field is a scalar or a tuple so `inspect-robots run -E key=value` can set it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

# The arm's joints, always the first six action dimensions.
JOINTS: tuple[str, ...] = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "grip_mm")
NUM_JOINTS = len(JOINTS)

# The base's heading relative to the start of the trial (zeroed when the operator starts each
# trial): counter-clockwise seen from above, as stretch4_body's omnibase defines it. The base is
# holonomic, but it is only turned in place here.
BASE_JOINTS: tuple[str, ...] = ("base_theta",)
BASE_MODES: dict[str, tuple[str, ...]] = {
    "none": (),
    "rotate": BASE_JOINTS,
}
UNITS: dict[str, str] = {
    "lift": "m",
    "arm": "m",
    "wrist_yaw": "rad",
    "wrist_pitch": "rad",
    "wrist_roll": "rad",
    "grip_mm": "mm",
    "base_theta": "rad",
}

# The one proprioception field: the joints, in action order (the agent's reference pose).
STATE_KEY = "joint_pos"

# Safety limits, from stretch4_body (robot_params_SE4.py and the wrist drivers):
# - lift [0, 1.2] m and arm [0, 0.55] m: range_m.
# - wrist_yaw, wrist_pitch, wrist_roll: the soft limits stretch4_body applies, range_deg
#   [-65, 245] less range_pad_deg (0 for yaw, 1 for pitch and roll). stretch_body_server's
#   self-collision checks still apply inside these.
# - grip_mm: per gripper, below.
DEFAULT_LOW: tuple[float, ...] = (0.0, 0.0, -1.1345, -1.1170, -1.1170, 0.0)
DEFAULT_HIGH: tuple[float, ...] = (1.2, 0.55, 4.2761, 4.2586, 4.2586, 150.0)

# Fully open fingertip aperture (mm): SG4 aperture_open_m, PG4 range_mm.
GRIPPER_OPEN_MM: dict[str, float] = {"stretch_gripper": 150.0, "parallel_gripper": 77.0}

# Largest change per control step, so a policy sample that jumps further than teleoperation
# ever does is rate-limited, not executed. The hard backstop in step().
MAX_STEP: tuple[float, ...] = (0.05, 0.05, 0.6, 0.6, 0.6, 40.0)
BASE_MAX_STEP: dict[str, float] = {"base_theta": 0.2}


def clamp(
    target: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    cfg: Stretch4EmbodimentConfig,
) -> npt.NDArray[np.float64]:
    """The hard safety backstop: within MAX_STEP (and BASE_MAX_STEP) of `current`, then within
    the limits. Applied by the embodiment and again by a remote robot's service."""
    step = cfg.max_step
    near = np.clip(target, current - step, current + step)
    return np.clip(near, cfg.low, cfg.high)


# How far each joint actually travels in one 30 Hz step at the driver's run speeds. Declared
# as the action space's max_step, so policies that interpolate (the LLM agent) spread a motion
# over as many steps as the robot needs to finish it, instead of ending the motion while the
# joints are still a few centimetres along: the arm and lift ~8 cm/s including acceleration,
# the wrist ~0.45 rad/s on short moves, the gripper ~25-30 mm/s (measured on the SG4).
MOTION_STEP: tuple[float, ...] = (0.0025, 0.0025, 0.015, 0.015, 0.015, 0.9)
# The base's, from the driver's turn speed limit (0.3 rad/s).
BASE_MOTION_STEP: dict[str, float] = {"base_theta": 0.01}

# stretch4_body camera names (driver.CAMERAS). Images are upright, centre-cropped where
# CAMERA_CROP says (the head eyes) and otherwise whole, then scaled down (never stretched) so
# the longer side is at most max_image_side.
DEFAULT_CAMERAS: tuple[str, ...] = ("head_right", "gripper_right")

# Each camera's frame as stretch4_body captures it (height, width), and the quarter turns
# (np.rot90) that make it upright: RGBCameras[c].config's image_size and
# rotate_number_of_times. The driver checks these against the installed stretch4_body at
# connect. The head eyes are mounted sideways, so their upright frame is portrait.
CAMERA_NATIVE_HW: dict[str, tuple[int, int]] = {
    "head_left": (1200, 1920),
    "head_right": (1200, 1920),
    "gripper_left": (400, 640),
    "gripper_right": (400, 640),
}
CAMERA_ROTATE: dict[str, int] = {
    "head_left": 1,
    "head_right": -1,
    "gripper_left": 0,
    "gripper_right": 0,
}

# The centred crop, as a height:width aspect, taken from the upright frame. The head eyes are
# fisheye lenses mounted sideways: their upright frame is 1200x1920 portrait, and the crop keeps
# its full width and the middle 690 rows (615 to 1305), a 1200x690 band at 640:368.
CAMERA_CROP: dict[str, tuple[int, int]] = {
    "head_left": (368, 640),
    "head_right": (368, 640),
}


def crop_box(hw: tuple[int, int], aspect_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the largest centred box with `aspect_hw` in an `hw` frame."""
    h, w = hw
    ah, aw = aspect_hw
    if w * ah <= h * aw:  # the frame is taller than the aspect: keep the width
        ch, cw = round(w * ah / aw), w
    else:
        ch, cw = h, round(h * aw / ah)
    top, left = (h - ch) // 2, (w - cw) // 2
    return top, top + ch, left, left + cw


# How a served robot sends its images (remote.codec): its camera's JPEGs, or lossless PNG.
IMAGE_MODES: tuple[str, ...] = ("forward", "lossless")

# Robocurve's adapters mostly send 640x480. 640 is also below the size at which the model
# providers downscale an image, so every model sees the same pixels.
DEFAULT_MAX_IMAGE_SIDE = 640

# The wrist stereo camera computes depth aligned to its right eye.
DEPTH_CAMERA = "gripper_right"


def action_semantics(cfg: Stretch4EmbodimentConfig) -> ActionSemantics:
    """Absolute joint targets, labelled, with each dimension's per-step travel."""
    base = BASE_MODES[cfg.base]
    return ActionSemantics(
        control_mode="joint_pos",
        rotation_repr="none",
        gripper="continuous",
        frame="base",
        dim_labels=cfg.joints,
        max_step=(*MOTION_STEP, *(BASE_MOTION_STEP[j] for j in base)),
    )


# Injected into LLM policies' system prompts verbatim, so keep it short and factual.
DOCS = """\
Hello Robot Stretch 4: a mobile manipulator with an omnidirectional base, a vertical lift, a \
horizontal telescoping arm, a 3-axis wrist and a gripper.

Action = absolute joint targets, in this order:
- lift (m): height of the arm carriage; positive is up.
- arm (m): telescoping arm extension; positive extends away from the mast.
- wrist_yaw (rad): positive swings the gripper to the robot's left. 0 points it straight out \
along the arm, 3.14 folds it back (stowed).
- wrist_pitch (rad): 0 is level; negative tilts the gripper up (-1.12 = fully up), \
positive tilts it down. Self-collision can stop it before its upper limit, depending on the \
lift and arm.
- wrist_roll (rad): positive rolls the gripper clockwise, seen from behind the gripper \
looking out along its fingers; negative rolls it counter-clockwise.
- grip_mm (mm): fingertip aperture; 0 is closed, larger is more open.

"""

BASE_FIXED_DOCS = "The base is held still, so the head camera's view does not move.\n"

# Per base joint, appended in action order after the arm's.
BASE_DOCS: dict[str, str] = {
    "base_theta": (
        "- base_theta (rad): base heading relative to its heading at the start of the trial; "
        "positive turns it counter-clockwise seen from above. The base turns in place and "
        "does not drive. Turning it swings the whole arm, and the arm's reach line, about the "
        "base's centre.\n"
    ),
}

BASE_MOVING_DOCS = (
    "The base starts each trial at base_theta 0. The head camera is mounted on the base: "
    "turning the base moves the head camera's view with it, so objects shift in the image or "
    "leave it, and positions or image scales worked out before a turn no longer apply. "
    "Re-locate objects in a fresh observation after every turn.\n"
)

# Per-gripper notes, from operating the hardware.
GRIPPER_DOCS: dict[str, str] = {
    "stretch_gripper": (
        "Gripper: the standard Stretch gripper's fingertips move along a curve. As they "
        "close they also pull back toward the wrist, so an object that sits just between the "
        "open fingertips ends up in front of them once they close, and the grasp misses. "
        "Before closing, extend until the object is deep between the fingers, close to the "
        "palm, further than looks necessary.\n"
    ),
}

CAMERA_DOCS: dict[str, str] = {
    "head_left": "the head's left eye, looking down at the workspace",
    "head_right": "the head's right eye, looking down at the workspace",
    "gripper_left": "the gripper camera's left eye, on the wrist",
    "gripper_right": "the gripper camera's right eye, on the wrist",
}


def docs(cfg: Stretch4EmbodimentConfig) -> str:
    """DOCS plus the configured base joints, gripper notes and cameras."""
    base = BASE_MODES[cfg.base]
    text = DOCS + "".join(BASE_DOCS[j] for j in base)
    n = len(cfg.joints)
    text += f"\nState `joint_pos` has the same {n} values in the same order and units.\n"
    text += BASE_MOVING_DOCS if base else BASE_FIXED_DOCS
    cams = "; ".join(f"`{c}` is {CAMERA_DOCS.get(c, 'a camera')}" for c in cfg.cameras)
    text += f"{GRIPPER_DOCS.get(cfg.gripper, '')}Cameras: {cams}.\n"
    if cfg.gripper_depth:
        text += (
            f"Depth: `{DEPTH_CAMERA}` comes with a metric depth image from the wrist stereo "
            "camera, pixel-aligned to it (brighter is nearer); its label gives the distance "
            "range and the distance at the image centre. Use it to judge how far the object is "
            "from the fingers before closing. Pixels too close or too far for the stereo have "
            "no depth.\n"
        )
    return text


@dataclass(frozen=True)
class Stretch4EmbodimentConfig:
    """Static configuration for the Stretch 4 embodiment."""

    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    # The longer side of every image, in pixels; frames smaller than this are not enlarged.
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE
    # The OAK cameras' frame rate. Each step waits for the next synced frame, so this is the
    # rate steps actually run at.
    control_hz: float = 30.0
    joint_low: tuple[float, ...] = DEFAULT_LOW
    joint_high: tuple[float, ...] = DEFAULT_HIGH
    # A pose file ({"joints": {"lift": ..., ..., "grip_mm": ...}}) to drive to, slowly, at
    # every reset.
    home_pose_file: str | None = None
    home_timeout_s: float = 40.0
    # Skip the "press Enter to start" prompt before each trial.
    auto_start: bool = False
    # The installed gripper: "stretch_gripper" (SG4) or "parallel_gripper" (PG4). Sets grip_mm's
    # upper limit unless joint_high is given; the driver refuses a robot with the other one.
    gripper: str = "parallel_gripper"
    # Command priority and lock behavior for stretch_body_server (gamepad teleop is 1).
    priority: int = 2
    ignore_control_lock: bool = False
    # Also send the wrist stereo camera's metric depth, aligned to `gripper_right`, as the
    # observation extra `gripper_right_depth` (meters; 0 where the stereo has no depth).
    gripper_depth: bool = False
    # Whether the action has the base's heading: "rotate" (base_theta) or "none".
    base: str = "rotate"
    # The base may turn at most this far from its start heading in a trial: a backstop against
    # a misread sign or unit, not a workspace.
    base_max_theta: float = math.pi

    def __post_init__(self) -> None:
        for name in ("auto_start", "ignore_control_lock", "gripper_depth"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false, got {getattr(self, name)!r}")
        if self.base not in BASE_MODES:
            raise ValueError(f"base must be one of {sorted(BASE_MODES)}, got {self.base!r}")
        if not math.isfinite(self.base_max_theta) or self.base_max_theta <= 0:
            raise ValueError(f"base_max_theta must be finite and > 0, got {self.base_max_theta!r}")
        if self.gripper not in GRIPPER_OPEN_MM:
            raise ValueError(
                f"gripper must be one of {sorted(GRIPPER_OPEN_MM)}, got {self.gripper!r}"
            )
        if self.joint_high is DEFAULT_HIGH:
            object.__setattr__(
                self, "joint_high", (*DEFAULT_HIGH[:-1], GRIPPER_OPEN_MM[self.gripper])
            )
        for name in ("joint_low", "joint_high"):
            values = getattr(self, name)
            if len(values) != NUM_JOINTS:
                raise ValueError(f"{name} must have {NUM_JOINTS} entries ({', '.join(JOINTS)})")
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must be finite, got {values}")
        if any(lo >= hi for lo, hi in zip(self.joint_low, self.joint_high, strict=True)):
            raise ValueError("joint_low must be below joint_high for every joint")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        unknown = [c for c in self.cameras if c not in CAMERA_NATIVE_HW]
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; choose from {sorted(CAMERA_NATIVE_HW)}")
        if self.gripper_depth and DEPTH_CAMERA not in self.cameras:
            raise ValueError(
                f"gripper_depth needs the {DEPTH_CAMERA} camera (the depth is aligned to it); "
                f"add it to cameras"
            )
        if self.max_image_side <= 0:
            raise ValueError("max_image_side must be > 0")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if not math.isfinite(self.home_timeout_s) or self.home_timeout_s <= 0:
            raise ValueError("home_timeout_s must be finite and > 0")

    @classmethod
    def from_kwargs(cls, **flat: Any) -> Stretch4EmbodimentConfig:
        """Build from flat CLI kwargs, rejecting unknown keys loudly."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"stretch4 got unexpected config keys: {sorted(unknown)}")
        # The CLI passes "a,b" strings for tuple fields.
        for key in ("cameras", "joint_low", "joint_high"):
            if isinstance(flat.get(key), str):
                parts = [p.strip() for p in flat[key].split(",") if p.strip()]
                flat[key] = tuple(parts) if key == "cameras" else tuple(float(p) for p in parts)
        # The CLI parses `-E base=none` to None; "none" is the mode the user meant.
        if "base" in flat and flat["base"] is None:
            flat["base"] = "none"
        return cls(**flat)

    def image_shape(self, camera: str) -> tuple[int, int]:
        """(height, width) of `camera`'s images: its upright, cropped frame, scaled down to fit."""
        h, w = CAMERA_NATIVE_HW[camera]
        if CAMERA_ROTATE[camera] % 2:
            h, w = w, h
        if camera in CAMERA_CROP:
            top, bottom, left, right = crop_box((h, w), CAMERA_CROP[camera])
            h, w = bottom - top, right - left
        scale = min(1.0, self.max_image_side / max(h, w))
        return max(1, round(h * scale)), max(1, round(w * scale))

    @property
    def joints(self) -> tuple[str, ...]:
        """Every action dimension, in order: the arm's, then the configured base's."""
        return JOINTS + BASE_MODES[self.base]

    @property
    def base_limits(self) -> dict[str, float]:
        """Each base joint's symmetric limit around the start pose."""
        return {"base_theta": self.base_max_theta}

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Lower limits for every action dimension as a float64 array."""
        base = [-self.base_limits[j] for j in BASE_MODES[self.base]]
        return np.asarray([*self.joint_low, *base], dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Upper limits for every action dimension as a float64 array."""
        base = [self.base_limits[j] for j in BASE_MODES[self.base]]
        return np.asarray([*self.joint_high, *base], dtype=np.float64)

    @property
    def max_step(self) -> npt.NDArray[np.float64]:
        """The hard per-step change limit for every action dimension."""
        base = [BASE_MAX_STEP[j] for j in BASE_MODES[self.base]]
        return np.asarray([*MAX_STEP, *base], dtype=np.float64)


def action_box(cfg: Stretch4EmbodimentConfig) -> Box:
    """The absolute joint-position action space: the arm, then the configured base."""
    n = len(cfg.joints)
    return Box(shape=(n,), low=cfg.low, high=cfg.high, semantics=action_semantics(cfg))


def observation_space(cfg: Stretch4EmbodimentConfig) -> ObservationSpace:
    """The cameras plus the joints as one state field."""
    return ObservationSpace(
        cameras=tuple(
            CameraSpec(name=n, height=h, width=w)
            for n in cfg.cameras
            for h, w in [cfg.image_shape(n)]
        ),
        state=StateSpec(
            fields=(
                StateField(
                    key=STATE_KEY,
                    shape=(len(cfg.joints),),
                    unit=",".join(UNITS[j] for j in cfg.joints),
                ),
            )
        ),
    )

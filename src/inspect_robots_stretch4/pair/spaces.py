"""The pair's names and spaces, and how one action and two readings map to them.

Pure functions: no hardware, no threads.

- Action and state: robot 1's joints, then robot 2's, labelled `robot1_lift` ...
  `robot2_grip_mm`: six each, or seven with `-E base=rotate` (`robotN_base_theta` last), so
  12-D or 14-D.
- Cameras are named by robot and device, not by eye: `robot1_head`, `robot1_wrist`,
  `robot2_head`, `robot2_wrist`. Depth is `<camera>_depth`, the convention depth-aware policies
  (the LLM agent) look for.
"""

from __future__ import annotations

from collections.abc import Mapping

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
from inspect_robots.types import Observation

from inspect_robots_stretch4.config import (
    DEPTH_CAMERA,
    JOINTS,
    MOTION_STEP,
    NUM_JOINTS,
    STATE_KEY,
    UNITS,
    action_semantics,
)
from inspect_robots_stretch4.driver import Reading
from inspect_robots_stretch4.pair.config import SIDES, Stretch4PairConfig, device_of

# The bases-held (default) layout; joints(cfg) etc. give the configured one.
PAIR_DIM = 2 * NUM_JOINTS
PAIR_JOINTS: tuple[str, ...] = tuple(f"{side}_{joint}" for side in SIDES for joint in JOINTS)
DEVICE_NAMES = {"head": "head", "gripper": "wrist"}

PAIR_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",  # each robot's joints, in its own frame
    dim_labels=PAIR_JOINTS,
    max_step=MOTION_STEP * 2,
)


def alias(side: str, camera: str) -> str:
    """('robot1', 'head_right') -> 'robot1_head'."""
    return f"{side}_{DEVICE_NAMES[device_of(camera)]}"


def joints(cfg: Stretch4PairConfig) -> tuple[str, ...]:
    """Every action dimension, robot 1's then robot 2's, each prefixed with its robot."""
    return tuple(f"{s}_{j}" for s in SIDES for j in cfg.side(s).joints)


def semantics(cfg: Stretch4PairConfig) -> ActionSemantics:
    """PAIR_SEMANTICS for the configured joints (each robot's own max_step, base included)."""
    steps = [step for s in SIDES for step in action_semantics(cfg.side(s)).max_step]
    return ActionSemantics(
        control_mode="joint_pos",
        rotation_repr="none",
        gripper="continuous",
        frame="base",
        dim_labels=joints(cfg),
        max_step=tuple(steps),
    )


YAW, PITCH, ROLL = (JOINTS.index(j) for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"))


def to_robot(values: npt.NDArray[np.float64], left: bool) -> npt.NDArray[np.float64]:
    """One robot's joints as the model gives them -> as its wrist takes them. A left-handed
    (folded) wrist reaches each orientation as (yaw + pi, pi - pitch, roll + pi)."""
    out = np.array(values, dtype=np.float64)
    if left:
        out[YAW] += np.pi
        out[PITCH] = np.pi - out[PITCH]
        out[ROLL] += np.pi
    return out


def to_model(values: npt.NDArray[np.float64], left: bool) -> npt.NDArray[np.float64]:
    """The inverse of `to_robot`."""
    out = np.array(values, dtype=np.float64)
    if left:
        out[YAW] -= np.pi
        out[PITCH] = np.pi - out[PITCH]
        out[ROLL] -= np.pi
    return out


def model_limits(
    cfg: Stretch4PairConfig, side: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One robot's (low, high) as the model sees them."""
    side_cfg = cfg.side(side)  # type: ignore[arg-type]
    left = cfg.left_handed(side)  # type: ignore[arg-type]
    a, b = to_model(side_cfg.low, left), to_model(side_cfg.high, left)
    return np.minimum(a, b), np.maximum(a, b)


def action_box(cfg: Stretch4PairConfig) -> Box:
    limits = [model_limits(cfg, s) for s in SIDES]
    low = np.concatenate([lo for lo, _ in limits])
    high = np.concatenate([hi for _, hi in limits])
    return Box(shape=(len(low),), low=low, high=high, semantics=semantics(cfg))


def observation_space(cfg: Stretch4PairConfig) -> ObservationSpace:
    cameras = []
    for side in SIDES:
        side_cfg = cfg.side(side)
        for camera in side_cfg.cameras:
            h, w = side_cfg.image_shape(camera)
            cameras.append(CameraSpec(name=alias(side, camera), height=h, width=w))
    units = ",".join(UNITS[j] for s in SIDES for j in cfg.side(s).joints)
    dim = len(joints(cfg))
    return ObservationSpace(
        cameras=tuple(cameras),
        state=StateSpec(fields=(StateField(key=STATE_KEY, shape=(dim,), unit=units),)),
    )


def split(
    data: npt.NDArray[np.float64], cfg: Stretch4PairConfig
) -> dict[str, npt.NDArray[np.float64]]:
    """One pair action -> each robot's own joints (six, or seven with its base)."""
    n = len(cfg.side("robot1").joints)
    if data.shape != (2 * n,):
        raise ValueError(f"stretch4-pair actions have {2 * n} values, got shape {data.shape}")
    return {"robot1": data[:n], "robot2": data[n:]}


def join(per_side: Mapping[str, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Each robot's values -> one vector, robot 1 first."""
    return np.concatenate([np.asarray(per_side[s], dtype=np.float64) for s in SIDES])


def merge(
    readings: Mapping[str, Reading], cfg: Stretch4PairConfig, instruction: str | None
) -> Observation:
    """Both robots' readings -> one observation. Each image keeps its own robot's capture time;
    the state's time is robot 1's."""
    images, image_times, extra = {}, {}, {}
    for side in SIDES:
        reading = readings[side]
        for camera in cfg.side(side).cameras:
            name = alias(side, camera)
            images[name] = reading.images[camera]
            image_times[name] = reading.t
        if DEPTH_CAMERA in reading.depth:
            extra[f"{alias(side, DEPTH_CAMERA)}_depth"] = reading.depth[DEPTH_CAMERA]
    return Observation(
        images=images,
        state={
            STATE_KEY: join({s: to_model(readings[s].joints, cfg.left_handed(s)) for s in SIDES})
        },
        instruction=instruction,
        image_times=image_times,
        state_time=readings["robot1"].t,
        extra=extra,
    )

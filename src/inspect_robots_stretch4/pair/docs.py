"""The pair's system-prompt text: who is who, then one robot's docs, then the
cameras, depth and the measured layout. Injected into LLM policies verbatim."""

from __future__ import annotations

import math

import numpy as np

from inspect_robots_stretch4.config import (
    ARRIVAL_DOCS,
    BASE_DOCS,
    BASE_MOVING_DOCS,
    DEPTH_CAMERA,
    DOCS,
    GRIPPER_DOCS,
    JOINTS,
    view_docs,
)
from inspect_robots_stretch4.pair.config import SIDES, Stretch4PairConfig
from inspect_robots_stretch4.pair.rig import Rig, rig_facts
from inspect_robots_stretch4.pair.spaces import alias, model_limits

OPPOSED = """\
Two Hello Robot Stretch 4s work together at one table, facing each other from its two ends. \
Every joint, camera and state value is named for its robot: robot1_... or robot2_.... Each \
robot's arm extends toward the other robot. In robot1_head, robot 1's own arm comes in from the \
near (bottom) edge and robot 2's arm reaches in from the far side; in robot2_head it is the \
other way round. One action sets both arms' joints, so both can move at once.
"""

UNMEASURED = """\
Two Hello Robot Stretch 4s work together at one table. Every joint, camera and state value is \
named for its robot: robot1_... or robot2_.... In each robot's head camera, its own arm comes \
in from the near (bottom) edge. One action sets both arms' joints, so both can move at once.
"""

LAYOUT = """\
The action and the state `joint_pos` are robot 1's {n} joints followed by robot 2's {n}, in the \
order below, each prefixed with its robot: robot1_lift ... robot1_{last}, robot2_lift ... \
robot2_{last}. Each robot's values are in its own frame and units, exactly as for one robot:
"""

BASE_LIMITS = """\
Each robot's base turns at most {max_deg:.0f} degrees ({max_rad:.2f} rad) either way from its \
start heading, and only while that robot's arm is retracted to {arm:.2f} m or less, both now \
and in the target; otherwise its base holds its heading. Retract the arm, turn, then extend.
"""

DEVICE_DOCS = {
    "head": "{side}'s head camera, looking down at the table from where {side} stands",
    "wrist": "{side}'s gripper camera, on its wrist, looking out along its fingers",
}


def docs(cfg: Stretch4PairConfig, rig: Rig | None) -> str:
    text = OPPOSED if rig is not None and rig.layout == "opposed" else UNMEASURED
    joints = cfg.side("robot1").joints
    text += LAYOUT.format(n=len(joints), last=joints[-1]) + DOCS
    plain = Stretch4PairConfig()
    for side in SIDES:
        label = side.replace("robot", "robot ")
        if cfg.left_handed(side):
            text += (
                f"{label}'s wrist is folded over (left-handed), so its gripper sits a few cm to "
                "the left of its arm's line instead of the right; its joint values mean the same "
                "as above.\n"
            )
        lows, highs = model_limits(cfg, side)
        default_lows, default_highs = model_limits(plain, side)
        for i, joint in enumerate(JOINTS):
            if joint.startswith("wrist") and (
                not np.isclose(lows[i], default_lows[i])
                or not np.isclose(highs[i], default_highs[i])
            ):
                text += (
                    f"{side}_{joint} is limited to {lows[i]:.2f}..{highs[i]:.2f} rad on this "
                    "setup.\n"
                )
    text += ARRIVAL_DOCS
    if cfg.base != "none":
        text += "".join(BASE_DOCS[j] for j in joints[len(JOINTS) :])
        text += BASE_LIMITS.format(
            max_deg=math.degrees(cfg.base_max_theta),
            max_rad=cfg.base_max_theta,
            arm=cfg.base_turn_max_arm,
        )
        text += BASE_MOVING_DOCS
    grippers = {getattr(cfg, f"{s}_gripper") for s in SIDES}
    for gripper in sorted(grippers):
        text += GRIPPER_DOCS.get(gripper, "")
    lines = []
    for side in SIDES:
        for camera in cfg.side(side).cameras:
            name = alias(side, camera)
            device = name.rsplit("_", 1)[1]
            label = side.replace("robot", "robot ")
            lines.append(f"`{name}` is {DEVICE_DOCS[device].format(side=label)}")
    text += f"Cameras: {'; '.join(lines)}.\n"
    text += view_docs((c for s in SIDES for c in cfg.side(s).cameras), grippers)
    if cfg.gripper_depth:
        wrists = " and ".join(f"`{alias(s, DEPTH_CAMERA)}`" for s in SIDES)
        text += (
            f"Depth: {wrists} each come with a metric depth image from that wrist's stereo "
            "camera, pixel-aligned to it (brighter is nearer); its label gives the distance "
            "range and the distance at the image centre. Use it to judge how far an object is "
            "from the fingers before closing. Pixels too close or too far for the stereo have "
            "no depth.\n"
        )
    if rig is not None:
        text += rig_facts(rig)
    return text

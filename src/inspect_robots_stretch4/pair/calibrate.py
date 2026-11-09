"""`stretch_inspect_pair_calibrate`: measure the opposed layout and write the rig file.

Connects to both robots as the pair does, but only reads joints: it never commands either. The
operator poses the arms with teleop, presses Enter, and the tool records both robots' joints:

1. table: each gripper level (wrist_pitch 0), lowered until its fingertips touch the table top;
2. touch: both grippers level, at the same height above the table, wrist_yaw 0, extended until
   the fingertips touch; K = robot1_arm + robot2_arm;
3. line-up: still touching, wrist_yaw adjusted until the fingertips line up sideways.

A pose that doesn't match its step (a tilted wrist, uneven heights, a yaw, an impossible K) is
refused and the step repeated, since the cross guard trusts these numbers.

    stretch_inspect_pair_calibrate --robot2-host tcp://<robot 2's IP>:5600 --out notes/pair_rig.json
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections.abc import Callable, Sequence

import numpy as np

from inspect_robots_stretch4.config import JOINTS
from inspect_robots_stretch4.pair.config import SIDES, Stretch4PairConfig
from inspect_robots_stretch4.pair.rig import Rig, rig_facts, save_rig
from inspect_robots_stretch4.side import RobotSide

ARM, LIFT, YAW, PITCH = (JOINTS.index(j) for j in ("arm", "lift", "wrist_yaw", "wrist_pitch"))
LEVEL_RAD = 0.05  # wrist_pitch and wrist_yaw count as 0 within this
SAME_HEIGHT_M = 0.01  # the two grippers count as level with each other within this
K_RANGE_M = (0.1, 1.1)  # two arms of 0-0.55 m each

Poses = dict[str, np.ndarray]

STEPS = (
    (
        "table",
        "Level each gripper (wrist_pitch 0) and lower it until its fingertips just touch the "
        "table top.",
    ),
    (
        "touch",
        "Raise both grippers to the same height above the table, keep them level, set "
        "wrist_yaw to 0, and extend both arms until the fingertips just touch.",
    ),
    (
        "line-up",
        "Keep them touching and adjust wrist_yaw until the fingertips line up sideways.",
    ),
)


def check(step: str, poses: Poses, table: Poses | None) -> str | None:
    """Why `poses` don't fit `step`, or None if they do."""
    for s, j in poses.items():
        if abs(j[PITCH]) > LEVEL_RAD:
            return f"{s}'s wrist_pitch is {j[PITCH]:.3f}; level it (0)"
    if step == "touch":
        assert table is not None
        heights = [poses[s][LIFT] - table[s][LIFT] for s in SIDES]
        if abs(heights[0] - heights[1]) > SAME_HEIGHT_M:
            return (
                f"the grippers are {abs(heights[0] - heights[1]) * 1000:.0f} mm apart in "
                "height; put them level with each other"
            )
        for s, j in poses.items():
            if abs(j[YAW]) > LEVEL_RAD:
                return f"{s}'s wrist_yaw is {j[YAW]:.3f}; set it to 0"
        k = poses["robot1"][ARM] + poses["robot2"][ARM]
        if not K_RANGE_M[0] <= k <= K_RANGE_M[1]:
            return f"robot1_arm + robot2_arm = {k:.3f} m can't be a touch; check both arms"
    return None


def measure(sides: dict[str, RobotSide], ask: Callable[[str], str]) -> Rig:
    """Walk the operator through the steps, reading both robots' joints after each and
    repeating a step whose pose doesn't fit it."""
    poses: dict[str, Poses] = {}
    for name, text in STEPS:
        while True:
            ask(f"\n[{name}] {text}\nPress Enter when ready...")
            got = {s: np.asarray(side.read().joints) for s, side in sides.items()}
            print("  " + "; ".join(f"{s}: {_fmt(j)}" for s, j in got.items()))
            problem = check(name, got, poses.get("table"))
            if problem is None:
                poses[name] = got
                break
            print(f"  not recorded: {problem}. Try again.")
    touch, line_up, table = poses["touch"], poses["line-up"], poses["table"]
    return Rig(
        layout="opposed",
        K_m=float(touch["robot1"][ARM] + touch["robot2"][ARM]),
        lateral_yaw_rad=float((line_up["robot1"][YAW] + line_up["robot2"][YAW]) / 2),
        table_lift_m={s: float(table[s][LIFT]) for s in SIDES},
        measured=datetime.date.today().isoformat(),
        notes="stretch_inspect_pair_calibrate; lateral_yaw_rad is the mean of both robots' yaw",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stretch_inspect_pair_calibrate",
        description="Measure two opposed Stretch 4s and write the rig file. Commands nothing.",
    )
    parser.add_argument("--robot2-host", required=True, help="tcp://<robot 2's IP>:<port>")
    parser.add_argument("--robot1-host", default=None, help="if robot 1 is served too")
    parser.add_argument("--keys", default=None, help="CURVE keys directory")
    parser.add_argument("--out", default="pair_rig.json", help="the rig file to write")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from inspect_robots_stretch4.pair.embodiment import Stretch4PairEmbodiment

    extra = {"keys_dir": args.keys} if args.keys else {}
    cfg = Stretch4PairConfig(
        robot1_host=args.robot1_host,
        robot2_host=args.robot2_host,
        cross_guard=False,
        gripper_depth=False,
        robot1_cameras=("gripper_right",),
        robot2_cameras=("gripper_right",),
        **extra,
    )
    pair = Stretch4PairEmbodiment(cfg)
    sides = pair._sides
    try:
        for side in sides.values():
            side.connect()
        rig = measure(sides, input)
    finally:
        pair.close()
    path = save_rig(rig, args.out)
    print(f"\nwrote {path}\n\n{rig_facts(rig)}", file=sys.stdout)
    return 0


def _fmt(joints: np.ndarray) -> str:
    return ", ".join(f"{j}={v:.3f}" for j, v in zip(JOINTS, joints, strict=True))

"""`stretch_inspect_pair_calibrate`'s measuring: reads only, and refuses poses that don't fit a
step."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_stretch4.driver import Reading
from inspect_robots_stretch4.pair.calibrate import measure


class Posed:
    """A robot side that reads a scripted pose each time, and records any command."""

    def __init__(self, poses: list[list[float]]) -> None:
        self.poses = poses
        self.commands: list = []

    def read(self) -> Reading:
        return Reading(joints=np.array(self.poses.pop(0)), images={}, t=0.0)

    def command(self, target):  # pragma: no cover - must never be called
        self.commands.append(target)


def test_measure_reads_the_rig_and_repeats_a_bad_pose(capsys) -> None:
    #          lift  arm  yaw  pitch roll grip
    table1 = [0.600, 0.10, 0.0, 0.00, 0, 30]
    table2 = [0.610, 0.10, 0.0, 0.01, 0, 30]
    uneven = [0.700, 0.45, 0.0, 0.00, 0, 30]  # robot 1 2 cm higher above the table than robot 2
    touch1 = [0.720, 0.45, 0.0, 0.00, 0, 30]
    touch2 = [0.730, 0.46, 0.02, 0.00, 0, 30]
    line1 = [0.720, 0.45, 0.03, 0.00, 0, 30]
    line2 = [0.730, 0.46, 0.01, 0.00, 0, 30]
    sides = {
        "robot1": Posed([table1, uneven, touch1, line1]),
        "robot2": Posed([table2, touch2, touch2, line2]),
    }
    prompts: list[str] = []
    rig = measure(sides, prompts.append)  # type: ignore[arg-type]
    assert rig.K_m == pytest.approx(0.91)
    assert rig.table_lift_m == {"robot1": 0.6, "robot2": 0.61}
    assert rig.lateral_yaw_rad == pytest.approx(0.02)
    assert len(prompts) == 4  # the uneven touch was asked again
    assert "mm apart in height" in capsys.readouterr().out
    assert all(not s.commands for s in sides.values())

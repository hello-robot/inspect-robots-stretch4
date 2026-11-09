"""`RobotSide`, the per-robot core both embodiments share."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4 import JOINTS
from inspect_robots_stretch4.config import DEFAULT_HIGH, DEFAULT_LOW, MAX_STEP


def test_constructor_touches_no_hardware(build_side, driver) -> None:
    side = build_side()
    assert not side.connected
    with pytest.raises(RuntimeError, match="before reset"):
        side.check_runstop()


def test_clamp_is_measured_from_the_last_read(build_side, driver) -> None:
    side = build_side()
    side.connect()
    with pytest.raises(RuntimeError, match="before the first read"):
        side.clamp(np.zeros(6))
    side.read()
    target = side.clamp(driver.joints + 1.0)
    step = np.asarray(MAX_STEP)
    np.testing.assert_allclose(target[:2], driver.joints[:2] + step[:2])


def test_clamp_keeps_to_the_joint_limits(build_side, driver) -> None:
    driver.joints = np.array([1.18, 0.01, 0.0, 0.0, 0.0, 75.0])
    side = build_side()
    side.connect()
    side.read()
    target = side.clamp(np.array([1.2, -0.1, 0.0, 0.0, 0.0, 200.0]))
    assert target[0] == DEFAULT_HIGH[0]
    assert target[1] == DEFAULT_LOW[1]
    assert target[5] == 77.0  # the parallel gripper's full opening, the default


def test_command_returns_what_was_sent_in_joint_order(build_side, driver) -> None:
    side = build_side()
    side.connect()
    sent = side.command(np.arange(6, dtype=np.float64))
    np.testing.assert_array_equal(sent, np.arange(6))
    assert driver.commands == [dict(zip(JOINTS, range(6), strict=True))]


def test_errors_name_the_side(build_side, driver) -> None:
    side = build_side(name="robot2", home_pose_file="home.json")
    side.connect()
    driver.arrive = False
    with pytest.raises(EmbodimentFault, match=r"^robot2: did not reach the home pose"):
        side.home()
    driver.runstopped = True
    with pytest.raises(SafetyAbort, match=r"^robot2: the Stretch 4 run-stop"):
        side.check_runstop()


def test_an_unnamed_side_keeps_the_single_robot_messages(build_side, driver) -> None:
    side = build_side()
    side.connect()
    driver.runstopped = True
    with pytest.raises(SafetyAbort, match=r"^the Stretch 4 run-stop"):
        side.check_runstop()


def test_home_without_a_pose_file_does_not_move(build_side, driver) -> None:
    side = build_side()
    side.connect()
    side.home()
    assert driver.gotos == []


def test_trigger_runstop_reaches_the_driver(build_side, driver) -> None:
    side = build_side()
    side.connect()
    side.trigger_runstop()
    assert driver.runstop_triggers == 1


def test_close_is_idempotent(build_side, driver) -> None:
    side = build_side()
    side.connect()
    side.close()
    side.close()
    assert driver.disconnects == 1 and not side.connected

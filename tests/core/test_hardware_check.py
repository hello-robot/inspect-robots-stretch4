"""`stretch_inspect_hardware_check` finds the hardware tests and turns them on."""

from __future__ import annotations

from inspect_robots_stretch4 import hardware_check


def test_it_points_pytest_at_the_hardware_tests() -> None:
    assert (hardware_check.TESTS / "test_robot.py").is_file()
    args = hardware_check.pytest_args(["-k", "runstop"])
    assert args[0] == str(hardware_check.TESTS)
    assert "--hardware" in args
    assert args[-2:] == ["-k", "runstop"]

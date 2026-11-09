"""`stretch_inspect_hardware_check`: run the hardware tests (tests/hardware) on this robot.

Needs a checkout of this repo installed with `pip install -e ".[dev]"`, since the tests live in
the repo and not in the package. Any arguments it doesn't know go on to pytest, for example
`-k runstop` to run only the run-stop tests.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# src/inspect_robots_stretch4/hardware_check.py -> the repo root
REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "hardware"


def pytest_args(extra: list[str]) -> list[str]:
    return [
        str(TESTS),
        "--hardware",
        "-v",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(REPO),
        *extra,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stretch 4 hardware tests. Run them only in person, standing at the "
        "robot with a hand on or near its run-stop: the arm and gripper move, and the tests "
        "engage and release the run-stop. Start stretch_body_server, home the robot and "
        "release the run-stop first.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="also run the test that stops and restarts stretch_body_server.service",
    )
    args, extra = parser.parse_known_args(argv)
    if not TESTS.is_dir():
        print(
            f"no hardware tests at {TESTS}; install from a checkout of inspect-robots-stretch4 "
            'with pip install -e ".[dev]"',
            file=sys.stderr,
        )
        return 2
    # ROS puts pytest plugins on PYTHONPATH that fail to load; the tests need none of them.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if args.server:
        os.environ["STRETCH4_HW_SERVER"] = "1"
    try:
        import pytest
    except ImportError:
        print('pytest is not installed; run pip install -e ".[dev]"', file=sys.stderr)
        return 2
    return int(pytest.main(pytest_args(extra)))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

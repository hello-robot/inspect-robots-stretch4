"""Shared fakes and builders: a fake Stretch 4 driver, a manual clock, and builders for a
`RobotSide`, a `Stretch4Embodiment` and a `Stretch4PairEmbodiment`, so tests need no robot."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from inspect_robots.errors import SafetyAbort

from inspect_robots_stretch4 import JOINTS, Reading, Stretch4Embodiment, Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import EncodedReading
from inspect_robots_stretch4.side import RobotSide

# Image shapes at max_image_side=SIDE: the head's cropped band and the gripper, both landscape.
SIDE = 8
HEAD_HW, GRIPPER_HW = (5, 8), (5, 8)


def fake_jpeg(seed: int, hw: tuple[int, int] = (10, 16)) -> bytes:
    """A small textured JPEG with the cameras' 16:10 sensor shape (landscape, as captured)."""
    import cv2

    rng = np.random.default_rng(seed)
    bgr = rng.integers(0, 256, (*hw, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", bgr)[1].tobytes()


class FakeDriver:
    """Reaches every command at once; records what it was sent."""

    def __init__(self, joints: Any = None) -> None:
        self.names: tuple[str, ...] = JOINTS
        self.joints = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 30.0] if joints is None else joints)
        self.events: list[str] = []  # goto / release_base / zero_base, in order
        self.commands: list[dict[str, float]] = []
        self.gotos: list[dict[str, float]] = []
        self.arrive = True
        self.runstopped = False
        self.runstop_triggers = 0
        self.runstop_requests = 0
        self.on_runstop_cb: Callable[[], None] | None = None
        self.disconnects = 0
        self.depth: dict[str, Any] = {}
        self.t = 1.0
        self.on_goto: Callable[[], None] | None = None  # runs inside goto(), e.g. to block it
        self.on_command: Callable[[], None] | None = None  # runs inside command(), likewise
        self.goto_cancelled = threading.Event()
        self._jpegs: dict[str, bytes] | None = None

    def read(self) -> Reading:
        return Reading(
            joints=self.joints.copy(),
            images={
                "head_right": np.zeros((*HEAD_HW, 3), dtype=np.uint8),
                "gripper_right": np.zeros((*GRIPPER_HW, 3), dtype=np.uint8),
            },
            t=1.0,
            depth=self.depth,
        )

    def read_encoded(self) -> EncodedReading:
        """What BodyDriver hands a service: each camera's JPEG, undecoded."""
        if self._jpegs is None:
            self._jpegs = {"head_right": fake_jpeg(1), "gripper_right": fake_jpeg(2)}
        return EncodedReading(
            joints=self.joints.copy(), jpegs=dict(self._jpegs), t=self.t, depth=self.depth
        )

    def command(self, targets: Mapping[str, float]) -> Mapping[str, float]:
        if self.on_command is not None:
            self.on_command()
        self.commands.append(dict(targets))
        self.joints = np.array([targets[j] for j in self.names])
        return dict(targets)

    def goto(self, pose: Mapping[str, float], timeout_s: float) -> tuple[bool, Any]:
        self.events.append("goto")
        self.gotos.append(dict(pose))
        if self.on_goto is not None:
            self.on_goto()
        return self.arrive, ("lift", 0.2)

    def home_until_runstopped(self, timeout_s: float = 3.0) -> None:
        """An `on_goto` that, like BodyDriver's goto, ends only when this robot is
        run-stopped (or after `timeout_s`, so a failing test can't hang)."""
        deadline = time.monotonic() + timeout_s
        while not self.runstopped and time.monotonic() < deadline:
            time.sleep(0.01)
        if self.runstopped:
            raise SafetyAbort("the run-stop was engaged while driving to the home pose")

    def cancel_goto(self) -> None:
        self.goto_cancelled.set()

    def configure(self, cfg: Stretch4EmbodimentConfig) -> FakeDriver:
        """The driver factory: take the config's joints, base joints starting at 0."""
        self.names = cfg.joints
        self.joints = np.concatenate([self.joints[: len(JOINTS)], np.zeros(len(cfg.joints) - 6)])
        return self

    def release_base(self) -> None:
        self.events.append("release_base")

    def zero_base(self) -> None:
        self.events.append("zero_base")
        self.joints[len(JOINTS) :] = 0.0

    def is_runstopped(self) -> bool:
        return self.runstopped

    def trigger_runstop(self) -> None:
        self.runstop_triggers += 1
        self.runstopped = True

    def on_runstop(self, callback: Callable[[], None] | None) -> None:
        self.on_runstop_cb = callback

    def request_runstop(self) -> None:
        self.runstop_requests += 1
        self.runstopped = True

    def press_runstop(self) -> None:
        """The run-stop button, as the driver's own thread reports it."""
        self.runstopped = True
        if self.on_runstop_cb is not None:
            self.on_runstop_cb()

    def disconnect(self) -> None:
        self.disconnects += 1


class Clock:
    """A manual monotonic clock; `sleep` advances it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> Clock:
    return Clock()


def home_pose(_path: str) -> dict[str, float]:
    """Every home pose file, as the builders load it."""
    return dict.fromkeys(JOINTS, 0.25)


@pytest.fixture
def driver() -> FakeDriver:
    return FakeDriver()


@pytest.fixture
def build_side(driver: FakeDriver):
    """A RobotSide on `driver`; config kwargs as -E would give them."""

    def _build(name: str = "", **cfg: Any) -> RobotSide:
        cfg = {"max_image_side": SIDE, "base": "none", **cfg}
        return RobotSide(
            Stretch4EmbodimentConfig(**cfg),
            factory=driver.configure,
            load_home_pose=home_pose,
            name=name,
        )

    return _build


@pytest.fixture
def build(driver: FakeDriver):
    """A Stretch4Embodiment on `driver`; config kwargs as -E would give them.

    `polls` are the Enter-to-end answers, step by step; `prompts` collects the start prompts;
    `factory` and `input_fn` replace the fake driver's factory and the stdin prompt.
    """

    def _build(
        polls: list[bool] | None = None,
        prompts: list[str] | None = None,
        factory: Callable[[Any], Any] | None = None,
        input_fn: Callable[[str], str] | None = None,
        load_home_pose: Callable[[str], dict[str, float]] = home_pose,
        **cfg: Any,
    ) -> Stretch4Embodiment:
        # The arm's tests run with the base fixed; test_base.py covers the base.
        cfg = {"max_image_side": SIDE, "base": "none", **cfg}
        queue = list(polls or [])
        return Stretch4Embodiment(
            Stretch4EmbodimentConfig(**cfg),
            driver_factory=factory or driver.configure,
            load_home_pose=load_home_pose,
            input_fn=input_fn
            or (lambda p: (prompts.append(p) if prompts is not None else None) or ""),
            poll_end=lambda: queue.pop(0) if queue else False,
        )

    return _build


RIG = {
    "v": 1,
    "layout": "opposed",
    "K_m": 0.9,
    "lateral_yaw_rad": 0.0,
    "table_lift_m": {"robot1": 0.6, "robot2": 0.61},
    "wrist_height_band_m": 0.08,
    "yaw_band_rad": 0.3,
    "margin_m": 0.03,
}


@pytest.fixture
def rig_file(tmp_path) -> str:
    import json

    path = tmp_path / "pair_rig.json"
    path.write_text(json.dumps(RIG))
    return str(path)


@pytest.fixture
def pair_drivers() -> dict[str, FakeDriver]:
    return {
        "robot1": FakeDriver([0.5, 0.1, 0.0, 0.0, 0.0, 30.0]),
        "robot2": FakeDriver([0.7, 0.2, 0.1, 0.0, 0.0, 40.0]),
    }


@pytest.fixture
def build_pair(pair_drivers):
    """A Stretch4PairEmbodiment on two FakeDrivers; config kwargs as -E would give them."""
    from inspect_robots_stretch4.pair.config import Stretch4PairConfig
    from inspect_robots_stretch4.pair.embodiment import Stretch4PairEmbodiment

    def _build(
        polls: list[bool] | None = None,
        factories: dict[str, Any] | None = None,
        load_home_pose: Callable[[str], dict[str, float]] = home_pose,
        **cfg: Any,
    ):
        cfg = {"max_image_side": SIDE, "cross_guard": False, "auto_start": True, **cfg}
        queue = list(polls or [])
        return Stretch4PairEmbodiment(
            Stretch4PairConfig(**cfg),
            driver_factories=factories or {s: (lambda _c, d=d: d) for s, d in pair_drivers.items()},
            load_home_pose=load_home_pose,
            poll_end=lambda: queue.pop(0) if queue else False,
        )

    return _build


# -- hardware tests (tests/hardware): skipped unless asked for ------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hardware",
        action="store_true",
        help="also run the tests marked `hardware`, which drive a real Stretch 4",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--hardware"):
        return
    skip = pytest.mark.skip(reason="drives a real Stretch 4; run with --hardware on the robot")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)

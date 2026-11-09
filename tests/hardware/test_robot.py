"""Checks of a single real Stretch 4: images, gripper, hold, run-stop, lease and server loss.

Skipped unless pytest gets `--hardware`. On the robot, with stretch_body_server running, the
robot homed, the run-stop released, the floor around the base clear and a hand near the
run-stop:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/hardware --hardware -v

What moves: the arm, a few centimeters to 10 cm from wherever it is; the gripper, open and
closed. The base is never commanded to move. The tests engage and release the run-stop from
software, through a second server client. `test_server_stop_is_detected` also stops and restarts
stretch_body_server.service (other clients such as gamepad teleop lose it), so it runs only with
STRETCH4_HW_SERVER=1 set.

Still for a person: whether the images are upright and show the right scene, and which way the
base goes in the room.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from inspect_robots import Action, Observation, Scene
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4 import JOINTS, STATE_KEY, Stretch4Embodiment, Stretch4EmbodimentConfig
from inspect_robots_stretch4.config import GRIPPER_OPEN_MM
from inspect_robots_stretch4.driver import ARRIVAL_TOL, RUNSTOP_PRIORITY, WRIST_JOINTS

pytestmark = pytest.mark.hardware

SCENE = Scene(id="hardware", instruction="hardware test")
ALL_CAMERAS = ("head_left", "head_right", "gripper_left", "gripper_right")
ARM_JOINTS = ("lift", "arm", *WRIST_JOINTS)
# "Did not move": a few times the encoders' noise, well under any commanded step.
STILL = {
    "lift": 0.005,
    "arm": 0.005,
    "wrist_yaw": 0.03,
    "wrist_pitch": 0.03,
    "wrist_roll": 0.03,
    "grip_mm": 3.0,
}
# How soon a fault must end a run (the manual check says "about a second").
DETECT_S = 1.0
SERVICE = "stretch_body_server"


class Probe:
    """A second stretch_body_server client: plays the run-stop button and other programs, and
    reads the joints while no embodiment is connected."""

    def __init__(self, client_id: str = "", wait_s: float | None = None) -> None:
        from stretch4_body.robot.robot_client import RobotClient

        self.robot = RobotClient(client_id=client_id or f"hardware_probe_{os.getpid()}")
        self.engaged = False  # whether we left the run-stop engaged
        if not self.robot.startup(verbose=False, block_timeout_s=wait_s):
            raise RuntimeError("could not connect to stretch_body_server")

    def arm(self) -> dict[str, float]:
        self.robot.pull_status(blocking=True)
        s = self.robot.status
        return {
            "lift": float(s["lift"]["pos"]),
            "arm": float(s["arm"]["pos"]),
            **{j: float(s["end_of_arm"][j]["pos"]) for j in WRIST_JOINTS},
        }

    def runstopped(self) -> bool:
        self.robot.pull_status(blocking=True)
        return bool(self.robot.is_runstopped())

    def runstop(self, engage: bool) -> None:
        """Press (or release) the run-stop. Pushed above every priority, as the driver does."""
        if engage:
            self.robot.trigger_runstop()
        else:
            self.robot.clear_runstop()
        self.robot.push_command(ignore_control_lock=True, priority=RUNSTOP_PRIORITY)
        self.engaged = engage

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.robot.stop()


@pytest.fixture
def probe() -> Iterator[Probe]:
    p = Probe()
    if not p.robot.is_homed():
        p.close()
        pytest.fail("the robot is not homed; run stretch_robot_home first")
    if p.runstopped():
        p.close()
        pytest.fail("the run-stop is engaged; release it first")
    try:
        yield p
    finally:
        # No status read here: it blocks forever if the server is down.
        try:
            if p.engaged:
                p.runstop(False)
        finally:
            p.close()


@pytest.fixture
def build() -> Iterator[Callable[..., Stretch4Embodiment]]:
    """Builds embodiments that start without a prompt; closes them all afterwards."""
    made: list[Stretch4Embodiment] = []

    def make(**kwargs: object) -> Stretch4Embodiment:
        cfg = Stretch4EmbodimentConfig(**{"base": "none", "auto_start": True, **kwargs})
        emb = Stretch4Embodiment(cfg, poll_end=lambda: False)
        made.append(emb)
        return emb

    yield make
    for emb in made:
        with contextlib.suppress(Exception):
            emb.close()


def joints(obs: Observation) -> npt.NDArray[np.float64]:
    return np.asarray(obs.state[STATE_KEY], dtype=np.float64)


def hold(emb: Stretch4Embodiment, obs: Observation, steps: int) -> Observation:
    """What stretch4-hold does: command the first reading, every step."""
    target = joints(obs)
    for _ in range(steps):
        obs = emb.step(Action(data=target)).observation
    return obs


def drive(emb: Stretch4Embodiment, obs: Observation, joint: str, value: float) -> Observation:
    """Command one joint to `value` for 2 s (the per-step limit ramps it), others held."""
    target = joints(obs)
    target[JOINTS.index(joint)] = value
    for _ in range(60):
        obs = emb.step(Action(data=target)).observation
    return obs


def moved(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """The joints that moved by more than STILL, with how far."""
    return {j: after[j] - before[j] for j in before if abs(after[j] - before[j]) > STILL[j]}


def away_from(here: dict[str, float], by: float) -> dict[str, float]:
    """A pose `by` m away on the lift and arm, toward the middle of their range, gripper 30 mm."""
    pose = {**here, "grip_mm": 30.0}
    pose["lift"] += -by if here["lift"] > 0.6 else by
    pose["arm"] += -by if here["arm"] > 0.25 else by
    return pose


def write_pose(path: Path, pose: dict[str, float]) -> str:
    import json

    path.write_text(json.dumps({"joints": pose}))
    return str(path)


def at(delay_s: float, fn: Callable[[], None]) -> dict[str, float]:
    """Run `fn` on a thread after `delay_s`; the returned dict gets the time it ran ("t")."""
    when: dict[str, float] = {}

    def run() -> None:
        time.sleep(delay_s)
        when["t"] = time.monotonic()
        fn()

    threading.Thread(target=run, daemon=True).start()
    return when


def once_moving(probe: Probe, fn: Callable[[], None], timeout_s: float = 30.0) -> dict[str, float]:
    """Run `fn` on a thread once the lift has moved 1 cm (the robot connects first, for
    several seconds); the returned dict gets the time it ran ("t")."""
    when: dict[str, float] = {}
    start = probe.arm()["lift"]

    def run() -> None:
        deadline = time.monotonic() + timeout_s
        while abs(probe.arm()["lift"] - start) < 0.01:
            if time.monotonic() > deadline:
                return
            time.sleep(0.02)
        when["t"] = time.monotonic()
        fn()

    threading.Thread(target=run, daemon=True).start()
    return when


# -- 1. images ---------------------------------------------------------------------------------


def test_images_have_the_documented_shapes(build, probe) -> None:
    emb = build(cameras=ALL_CAMERAS)
    obs = emb.reset(SCENE)
    shapes = {c: obs.images[c].shape for c in ALL_CAMERAS}
    assert shapes == {
        "head_left": (368, 640, 3),
        "head_right": (368, 640, 3),
        "gripper_left": (400, 640, 3),
        "gripper_right": (400, 640, 3),
    }


def test_gripper_reads_open_and_closed(build, probe) -> None:
    emb = build()
    obs = emb.reset(SCENE)
    start = joints(obs)[JOINTS.index("grip_mm")]
    open_mm = GRIPPER_OPEN_MM[Stretch4EmbodimentConfig().gripper]
    try:
        obs = drive(emb, obs, "grip_mm", open_mm)
        assert joints(obs)[JOINTS.index("grip_mm")] == pytest.approx(open_mm, abs=4.0)
        obs = drive(emb, obs, "grip_mm", 0.0)
        assert joints(obs)[JOINTS.index("grip_mm")] == pytest.approx(0.0, abs=3.0)
    finally:
        drive(emb, obs, "grip_mm", start)


# -- 2. hold run -------------------------------------------------------------------------------


def test_hold_keeps_every_joint_still(build, probe) -> None:
    emb = build()
    obs = emb.reset(SCENE)
    before = dict(zip(JOINTS, joints(obs), strict=True))
    t0 = time.monotonic()
    obs = hold(emb, obs, 150)
    elapsed = time.monotonic() - t0
    assert moved(before, dict(zip(JOINTS, joints(obs), strict=True))) == {}
    # Steps are paced by the 30 Hz cameras.
    assert 150 / 30 * 0.8 < elapsed < 150 / 30 * 1.5


def test_base_holds_still_in_rotate_mode(build, probe) -> None:
    emb = build(base="rotate")
    obs = emb.reset(SCENE)
    before = joints(obs)
    obs = hold(emb, obs, 150)
    after = joints(obs)
    # zip stops at the arm joints; base_theta is the last value.
    assert moved(dict(zip(JOINTS, before)), dict(zip(JOINTS, after))) == {}  # noqa: B905
    assert abs(after[-1] - before[-1]) < 0.01  # rad


# -- 3. run-stop during a run ------------------------------------------------------------------


def test_runstop_during_a_run_ends_it_and_the_robot_stays_still(build, probe) -> None:
    emb = build()
    obs = emb.reset(SCENE)
    pressed = at(1.0, lambda: probe.runstop(True))
    with pytest.raises(SafetyAbort, match="run-stop is engaged"):
        hold(emb, obs, 300)
    assert time.monotonic() - pressed["t"] < DETECT_S
    emb.close()

    before = probe.arm()
    probe.runstop(False)
    time.sleep(2.0)
    assert moved(before, probe.arm()) == {}


# -- 4. run-stop while homing ------------------------------------------------------------------


def test_runstop_while_homing_ends_it_and_the_move_does_not_resume(build, probe, tmp_path) -> None:
    pose = away_from(probe.arm(), by=0.10)  # about 2.5 s at the homing speed
    pose_file = write_pose(tmp_path / "start.json", pose)

    emb = build(home_pose_file=pose_file)
    pressed = once_moving(probe, lambda: probe.runstop(True))
    with pytest.raises(SafetyAbort, match="while driving to the home pose"):
        emb.reset(SCENE)
    assert "t" in pressed, "homing ended before the run-stop was pressed"
    emb.close()

    before = probe.arm()
    assert abs(before["lift"] - pose["lift"]) > 0.02, "the lift reached home before the stop"
    probe.runstop(False)
    time.sleep(2.0)
    assert moved(before, probe.arm()) == {}

    # Run again, untouched: it gets there and holds.
    emb = build(home_pose_file=pose_file)
    obs = emb.reset(SCENE)
    reached = dict(zip(JOINTS, joints(obs), strict=True))
    for j in ARM_JOINTS:
        assert reached[j] == pytest.approx(pose[j], abs=2 * ARRIVAL_TOL[j]), j
    obs = hold(emb, obs, 60)
    assert moved(reached, dict(zip(JOINTS, joints(obs), strict=True))) == {}


# -- 5. another program takes the lease --------------------------------------------------------


def test_another_program_taking_the_lease_ends_the_run(build, probe) -> None:
    emb = build()
    obs = emb.reset(SCENE)
    stop = threading.Event()

    def lease_test() -> None:
        """Holds the lift where it is at priority 10 (the eval pushes at 2)."""
        other = Probe(client_id="lease_test")
        try:
            lift = other.arm()["lift"]
            while not stop.is_set():
                other.robot.lift.move_to(lift, v_m=0.02, a_m=0.05)
                other.robot.push_command(priority=10)
                time.sleep(0.1)
        finally:
            other.close()

    started = at(1.0, lambda: threading.Thread(target=lease_test, daemon=True).start())
    try:
        with pytest.raises(SafetyAbort, match=r"another program \(lease_test\)"):
            hold(emb, obs, 300)
        assert time.monotonic() - started["t"] < DETECT_S + 0.5  # + connecting
    finally:
        stop.set()


# -- 6. the server stops -----------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("STRETCH4_HW_SERVER") != "1",
    reason="stops stretch_body_server.service; set STRETCH4_HW_SERVER=1 to run it",
)
def test_server_stop_is_detected(build) -> None:
    # Not the probe fixture: its socket would be dead after the restart.
    check = Probe()
    try:
        assert check.robot.is_homed() and not check.runstopped()
    finally:
        check.close()
    emb = build()
    obs = emb.reset(SCENE)
    systemctl = ["systemctl", "--user"]
    stopped = at(1.0, lambda: subprocess.run([*systemctl, "stop", SERVICE], check=True))
    try:
        with pytest.raises(EmbodimentFault, match="no status from stretch_body_server"):
            hold(emb, obs, 300)
        assert time.monotonic() - stopped["t"] < DETECT_S
    finally:
        try:
            emb.close()
        finally:
            subprocess.run([*systemctl, "start", SERVICE], check=True)
            back = Probe(wait_s=60)
            try:
                assert back.robot.is_homed(), "the robot needs homing after the restart"
            finally:
                back.close()

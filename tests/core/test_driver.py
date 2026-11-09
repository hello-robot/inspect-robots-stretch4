"""The driver's pure parts: no robot, no stretch4_body."""

from __future__ import annotations

import json

import numpy as np
import pytest
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4 import JOINTS, Stretch4Driver, Stretch4EmbodimentConfig
from inspect_robots_stretch4.config import CAMERA_CROP, CAMERA_NATIVE_HW, CAMERA_ROTATE, crop_box
from inspect_robots_stretch4.driver import (
    ARRIVAL_TOL,
    LEASE_CONFIRM_S,
    LEASE_TIMEOUT_S,
    STATUS_STALE_S,
    TAKEOVER_SETTLE_S,
    LeaseWatch,
    Timeline,
    arrived,
    check_camera_config,
    drive_to,
    gripper_joint,
    load_home_pose,
    preprocess,
    preprocess_depth,
    wall_minus_monotonic,
)


def test_timeline_interpolates_and_holds_the_ends() -> None:
    tl = Timeline()
    assert tl.interpolate(1.0) is None and tl.latest() is None and tl.nearest(1.0) is None
    tl.put(1.0, 10.0)
    tl.put(2.0, 20.0)
    tl.put(2.0, 99.0)  # not newer: ignored
    assert tl.interpolate(1.25) == pytest.approx(12.5)
    assert tl.interpolate(0.0) == 10.0
    assert tl.interpolate(5.0) == 20.0
    assert tl.latest() == (2.0, 20.0)
    assert tl.nearest(1.4) == (1.0, 10.0)


@pytest.mark.parametrize("camera", sorted(CAMERA_NATIVE_HW))
def test_preprocess_gives_each_camera_its_configured_shape(camera) -> None:
    out_hw = Stretch4EmbodimentConfig().image_shape(camera)
    bgr = np.zeros((*CAMERA_NATIVE_HW[camera], 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    rgb = preprocess(bgr, CAMERA_ROTATE[camera], out_hw, CAMERA_CROP.get(camera))
    assert rgb.shape == (*out_hw, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [0, 0, 255]  # still blue, now in RGB


def test_image_shapes() -> None:
    cfg = Stretch4EmbodimentConfig()
    assert cfg.image_shape("head_right") == (368, 640)  # the 1200x690 band of the fisheye
    assert cfg.image_shape("gripper_right") == (400, 640)  # the whole frame, not enlarged
    assert Stretch4EmbodimentConfig(max_image_side=320).image_shape("gripper_left") == (200, 320)
    assert Stretch4EmbodimentConfig(max_image_side=4000).image_shape("head_left") == (690, 1200)


def test_the_head_crop_keeps_the_middle_690_rows_of_the_upright_frame() -> None:
    assert crop_box((1920, 1200), (368, 640)) == (615, 1305, 0, 1200)
    assert crop_box((100, 400), (1, 1)) == (0, 100, 150, 250)  # a wide frame loses its sides
    upright = np.zeros((1920, 1200, 3), dtype=np.uint8)
    upright[615:1305] = 255  # the band the crop keeps
    native = np.ascontiguousarray(np.rot90(upright, k=1))  # as the sideways eye captures it
    rgb = preprocess(native, -1, (368, 640), (368, 640))
    assert rgb.min() == 255


def test_preprocess_without_a_crop_keeps_the_whole_frame() -> None:
    bgr = np.zeros((100, 400, 3), dtype=np.uint8)
    bgr[:, :10] = 255  # a white band at the left edge, which a centre crop would drop
    rgb = preprocess(bgr, 0, (25, 100))
    assert rgb[:, 0].min() == 255 and rgb[:, -1].max() == 0


def test_check_camera_config() -> None:
    check_camera_config("head_right", [1200, 1920], -1)
    with pytest.raises(EmbodimentFault, match="head_right"):
        check_camera_config("head_right", (1200, 1920), 1)
    with pytest.raises(EmbodimentFault, match="CAMERA_NATIVE_HW"):
        check_camera_config("gripper_right", (800, 1280), 0)


def test_gripper_joint() -> None:
    assert gripper_joint(["wrist_yaw", "parallel_gripper"]) == "parallel_gripper"
    with pytest.raises(EmbodimentFault, match="stretch_gripper"):
        gripper_joint(["wrist_yaw"])


def test_arrived() -> None:
    pose = dict.fromkeys(JOINTS, 0.0)
    ok, err = arrived(pose, {**pose, "lift": ARRIVAL_TOL["lift"] / 2})
    assert ok and err["lift"] == ARRIVAL_TOL["lift"] / 2
    assert not arrived(pose, {**pose, "grip_mm": 10.0})[0]


def test_load_home_pose(tmp_path) -> None:
    good = tmp_path / "start.json"
    good.write_text(json.dumps({"joints": {j: i for i, j in enumerate(JOINTS)}}))
    assert load_home_pose(str(good)) == {j: float(i) for i, j in enumerate(JOINTS)}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"joints": {"lift": 0.5}}))
    with pytest.raises(ValueError, match="arm"):
        load_home_pose(str(bad))


def test_wall_minus_monotonic_is_stable() -> None:
    assert abs(wall_minus_monotonic(5) - wall_minus_monotonic(5)) < 0.01


def test_preprocess_depth_converts_to_meters_and_never_blends() -> None:
    depth = np.zeros((400, 640), dtype=np.uint16)
    depth[:, :320] = 300  # a near surface on the left (0.3 m) ...
    depth[:, 320:] = 1200  # ... and a far one on the right (1.2 m)
    out = preprocess_depth(depth, 0, (240, 384))
    assert out.shape == (240, 384) and out.dtype == np.float32
    assert np.unique(out).tolist() == pytest.approx([0.3, 1.2])


def test_preprocess_depth_matches_the_image_rotation() -> None:
    depth = np.zeros((4, 6), dtype=np.uint16)
    depth[0, 0] = 500  # top-left before a quarter turn ...
    out = preprocess_depth(depth, 1, (6, 4))
    assert out[-1, 0] == pytest.approx(0.5)  # ... bottom-left after it, as np.rot90 does


def test_fake_driver_satisfies_the_protocol(driver) -> None:
    assert isinstance(driver, Stretch4Driver)


def test_lease_watch_is_quiet_while_status_flows_and_we_hold_the_lease(clock) -> None:
    watch = LeaseWatch("us", clock)
    watch.check()  # fresh at connect, before any status
    watch.pushed()
    clock.t = 0.2
    watch.status("us")
    watch.check()


def test_lease_watch_faults_when_status_stops(clock) -> None:
    watch = LeaseWatch("us", clock)
    watch.status("None")
    clock.t = STATUS_STALE_S + 0.01
    with pytest.raises(EmbodimentFault, match="no status"):
        watch.check()


def test_lease_watch_aborts_when_another_client_holds_the_lease(clock) -> None:
    watch = LeaseWatch("us", clock)
    watch.pushed()
    clock.t = LEASE_CONFIRM_S / 2
    watch.status("teleop")  # too soon after our push to tell
    watch.check()
    clock.t = LEASE_CONFIRM_S / 2 + TAKEOVER_SETTLE_S + 0.01
    watch.status("teleop")
    with pytest.raises(SafetyAbort, match="teleop"):
        watch.check()


def test_lease_watch_waits_for_a_takeover_to_settle(clock) -> None:
    watch = LeaseWatch("us", clock)
    watch.pushed()
    clock.t = LEASE_CONFIRM_S
    watch.status("us")
    clock.t += 0.01
    watch.status("teleop")  # just taken: it may be a software run-stop not yet shown
    watch.check()
    clock.t += TAKEOVER_SETTLE_S + 0.01
    watch.status("teleop")
    with pytest.raises(SafetyAbort, match="teleop"):
        watch.check()


def test_lease_watch_leaves_a_software_runstop_to_the_runstop_check(clock) -> None:
    watch = LeaseWatch("us", clock)
    watch.pushed()
    clock.t = LEASE_CONFIRM_S
    watch.status("us")
    clock.t += 0.01
    watch.status("runstop_button")
    watch.check()
    clock.t += 0.03
    watch.status("runstop_button", runstopped=True)
    clock.t += TAKEOVER_SETTLE_S
    watch.pushed()
    watch.status("runstop_button", runstopped=True)
    watch.check()


def test_lease_watch_catches_a_takeover_during_continuous_pushing(clock) -> None:
    watch = LeaseWatch("us", clock)
    for _ in range(30):  # 1 s at 30 Hz: every status follows a push by less than the window
        watch.pushed()
        clock.t += 0.01
        watch.status("us")
        watch.check()
        clock.t += 0.023
    for _ in range(5):  # past TAKEOVER_SETTLE_S
        watch.pushed()
        clock.t += 0.033
        watch.status("higher_priority_tool")
    with pytest.raises(SafetyAbort):
        watch.check()


def test_lease_watch_ignores_the_holder_once_our_lease_has_lapsed(clock) -> None:
    watch = LeaseWatch("us", clock)
    watch.pushed()
    clock.t = 0.2
    watch.status("us")
    clock.t = 0.2 + LEASE_TIMEOUT_S + 30.0  # a long LLM call: someone else may drive meanwhile
    watch.status("teleop")
    watch.check()
    watch.pushed()  # a new burst, which must be confirmed afresh
    watch.status("teleop")
    watch.check()
    clock.t += LEASE_CONFIRM_S + 0.01
    watch.status("teleop")
    with pytest.raises(SafetyAbort):
        watch.check()


def _drive(clock, current, runstopped, timeout_s: float = 1.0):
    sent: list = []
    pose = dict.fromkeys(JOINTS, 0.3)
    result = drive_to(
        pose,
        timeout_s,
        send=sent.append,
        current=current,
        runstopped=runstopped,
        clock=clock,
        sleep=clock.sleep,
    )
    return result, sent


def test_drive_to_arrives_after_two_settled_checks(clock) -> None:
    (ok, _), sent = _drive(clock, lambda: dict.fromkeys(JOINTS, 0.3), lambda: False)
    assert ok and len(sent) == 2


def test_drive_to_times_out_and_names_the_worst_joint(clock) -> None:
    far = {**dict.fromkeys(JOINTS, 0.3), "arm": 0.5}
    (ok, worst), sent = _drive(clock, lambda: far, lambda: False, timeout_s=0.95)
    assert not ok and worst[0] == "arm" and len(sent) == 10


def test_drive_to_never_sends_through_a_runstop(clock) -> None:
    with pytest.raises(SafetyAbort, match="run-stop"):
        _drive(clock, lambda: dict.fromkeys(JOINTS, 0.0), lambda: True)


def test_drive_to_stops_sending_once_the_runstop_engages(clock) -> None:
    sent: list = []
    with pytest.raises(SafetyAbort):
        drive_to(
            dict.fromkeys(JOINTS, 0.3),
            40.0,
            send=sent.append,
            current=lambda: dict.fromkeys(JOINTS, 0.0),
            runstopped=lambda: len(sent) >= 3,
            clock=clock,
            sleep=clock.sleep,
        )
    assert len(sent) == 3


def test_drive_to_cancelled_holds_where_the_robot_is(clock) -> None:
    sent: list = []
    here = dict.fromkeys(JOINTS, 0.1)
    result = drive_to(
        dict.fromkeys(JOINTS, 0.3),
        40.0,
        send=sent.append,
        current=lambda: here,
        runstopped=lambda: False,
        cancelled=lambda: len(sent) >= 2,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result == (False, "cancelled")
    assert len(sent) == 3 and sent[-1] == here  # two moves home, then a hold

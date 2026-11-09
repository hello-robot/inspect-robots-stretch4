"""The pair's names, spaces, config checks, and the 12-D (14-D with bases) split and
merge."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inspect_robots_stretch4.config import JOINTS, STATE_KEY
from inspect_robots_stretch4.driver import Reading
from inspect_robots_stretch4.net.auth import DEFAULT_KEYS_DIR
from inspect_robots_stretch4.pair import config as pair_config
from inspect_robots_stretch4.pair.config import Stretch4PairConfig
from inspect_robots_stretch4.pair.spaces import (
    PAIR_JOINTS,
    action_box,
    alias,
    merge,
    observation_space,
    split,
)

CFG = Stretch4PairConfig(max_image_side=8, cross_guard=False)


def test_labels_are_robot1_then_robot2() -> None:
    assert PAIR_JOINTS[:6] == tuple(f"robot1_{j}" for j in JOINTS)
    assert PAIR_JOINTS[6:] == tuple(f"robot2_{j}" for j in JOINTS)
    box = action_box(CFG)
    assert box.shape == (12,) and box.semantics.dim_labels == PAIR_JOINTS
    assert box.high[1] == CFG.side("robot1").high[1] and box.high[-1] == 77.0


def test_split_and_its_errors() -> None:
    parts = split(np.arange(12.0), CFG)
    np.testing.assert_array_equal(parts["robot1"], np.arange(6.0))
    np.testing.assert_array_equal(parts["robot2"], np.arange(6.0, 12.0))
    with pytest.raises(ValueError, match="12 values"):
        split(np.zeros(6), CFG)


BASES = Stretch4PairConfig(max_image_side=8, cross_guard=False, base="rotate")


def test_turning_bases_make_the_action_14_d() -> None:
    box = action_box(BASES)
    labels = box.semantics.dim_labels
    assert box.shape == (14,)
    assert labels[6] == "robot1_base_theta" and labels[13] == "robot2_base_theta"
    assert labels[7] == "robot2_lift"
    assert box.high[6] == pytest.approx(math.pi) and box.low[13] == -box.high[13]
    parts = split(np.arange(14.0), BASES)
    np.testing.assert_array_equal(parts["robot2"], np.arange(7.0, 14.0))
    state = observation_space(BASES).state.fields[0]
    assert state.shape == (14,) and state.unit.endswith(",rad")


def test_turning_bases_need_the_cross_guard_off() -> None:
    with pytest.raises(ValueError, match="cross_guard=false"):
        Stretch4PairConfig(base="rotate", cross_guard=True, rig_file="r.json")
    assert Stretch4PairConfig.from_kwargs(base=None, cross_guard=False).base == "none"


def test_cameras_are_named_by_robot_and_device() -> None:
    assert alias("robot2", "head_left") == "robot2_head"
    assert alias("robot1", "gripper_right") == "robot1_wrist"
    space = observation_space(CFG)
    shapes = {c.name: (c.height, c.width) for c in space.cameras}
    assert shapes == {
        "robot1_head": (5, 8),
        "robot1_wrist": (5, 8),
        "robot2_head": (5, 8),
        "robot2_wrist": (5, 8),
    }


def test_merge_keeps_each_robots_time_and_depth() -> None:
    def reading(t: float, base: float) -> Reading:
        img = np.zeros((5, 8, 3), np.uint8)
        return Reading(
            joints=np.full(6, base),
            images={"head_right": img, "gripper_right": img},
            t=t,
            depth={"gripper_right": np.zeros((5, 8), np.float32)},
        )

    obs = merge({"robot1": reading(1.0, 1.0), "robot2": reading(1.02, 2.0)}, CFG, "go")
    np.testing.assert_array_equal(obs.state[STATE_KEY], [1.0] * 6 + [2.0] * 6)
    assert obs.state_time == 1.0 and obs.image_times["robot2_wrist"] == 1.02
    assert set(obs.extra) == {"robot1_wrist_depth", "robot2_wrist_depth"}
    assert obs.instruction == "go"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"robot1_cameras": ("head_left", "head_right")}, "one eye per camera"),
        ({"robot2_cameras": ("head_right",)}, "gripper_depth needs gripper_right"),
        ({"robot2_cameras": ("head_center", "gripper_right")}, "unknown cameras"),
        ({"robot1_gripper": "hook"}, "robot1_gripper"),
        ({"image_mode": "gif"}, "image_mode"),
        ({"rpc_timeout_s": 0}, "rpc_timeout_s"),
    ],
)
def test_bad_configs_are_refused(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        Stretch4PairConfig(cross_guard=False, **kwargs)


def test_the_cross_guard_needs_a_rig_and_says_both_fixes() -> None:
    with pytest.raises(ValueError) as e:
        Stretch4PairConfig(cross_guard=True)
    assert "-E rig_file=" in str(e.value) and "-E cross_guard=false" in str(e.value)


def test_cli_strings_and_unknown_keys() -> None:
    cfg = Stretch4PairConfig.from_kwargs(
        cross_guard=False, gripper_depth=False, robot2_cameras="head_left"
    )
    assert cfg.robot2_cameras == ("head_left",)
    with pytest.raises(TypeError, match="left_host"):
        Stretch4PairConfig.from_kwargs(left_host="x")


def test_side_configs_carry_the_shared_settings() -> None:
    cfg = Stretch4PairConfig(cross_guard=False, robot2_gripper="stretch_gripper", priority=3)
    side = cfg.side("robot2")
    assert side.gripper == "stretch_gripper" and side.priority == 3 and side.gripper_depth


def test_the_keys_dir_default_matches_the_net_layer() -> None:
    assert pair_config.DEFAULT_KEYS_DIR == DEFAULT_KEYS_DIR


def test_a_robots_wrist_yaw_and_roll_can_be_narrowed() -> None:
    cfg = Stretch4PairConfig.from_kwargs(
        cross_guard=False, robot1_wrist_yaw_min=0, robot1_wrist_roll_max=1.0
    )
    one, two = cfg.side("robot1"), cfg.side("robot2")
    assert one.joint_low[2] == 0.0 and one.joint_high[2] == two.joint_high[2]
    assert two.joint_low == Stretch4PairConfig(cross_guard=False).side("robot2").joint_low
    assert one.joint_high[-1] == 77.0
    box = action_box(cfg)
    assert box.low[2] == 0.0 and box.low[8] < 0 and box.high[4] == 1.0
    from inspect_robots_stretch4.pair.docs import docs

    text = docs(cfg, None)
    assert "robot1_wrist_yaw is limited to 0.00..4.28 rad" in text
    assert "robot1_wrist_roll is limited to -1.12..1.00 rad" in text
    assert "robot2_wrist_yaw is limited" not in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"robot1_wrist_yaw_min": -2.0},
        {"robot2_wrist_roll_max": 5.0},
        {"robot1_wrist_yaw_min": 1.0, "robot1_wrist_yaw_max": 0.5},
    ],
)
def test_a_wrist_range_may_only_narrow(kwargs) -> None:
    with pytest.raises(ValueError, match="must narrow the default range"):
        Stretch4PairConfig(cross_guard=False, **kwargs)

"""The commanded base: action layout, reset order, clamping, and the driver's base math."""

from __future__ import annotations

import math

import numpy as np
import pytest
from inspect_robots import Action, Scene, Task, operator_scorer
from inspect_robots import eval as ir_eval
from inspect_robots.errors import EmbodimentFault

from inspect_robots_stretch4 import JOINTS, Stretch4EmbodimentConfig
from inspect_robots_stretch4.config import BASE_MAX_STEP, action_box, docs
from inspect_robots_stretch4.driver import BASE_MAX_W, base_turn_rate
from inspect_robots_stretch4.hold import Stretch4HoldPolicy

SCENE = Scene(id="s", instruction="pick up the cup")


def test_default_base_rotates() -> None:
    cfg = Stretch4EmbodimentConfig()
    assert cfg.joints == (*JOINTS, "base_theta")
    box = action_box(cfg)
    assert box.shape == (7,)
    assert box.semantics.dim_labels[-1] == "base_theta"
    assert box.low[-1] == pytest.approx(-math.pi) and box.high[-1] == pytest.approx(math.pi)


@pytest.mark.parametrize(("base", "extra"), [("none", ()), ("rotate", ("base_theta",))])
def test_base_modes_set_the_action_dimensions(base, extra) -> None:
    cfg = Stretch4EmbodimentConfig(base=base)
    assert cfg.joints == (*JOINTS, *extra)
    assert action_box(cfg).shape == (6 + len(extra),)


def test_docs_say_the_head_camera_moves_with_the_base() -> None:
    assert "moves the head camera's view" in docs(Stretch4EmbodimentConfig())
    fixed = docs(Stretch4EmbodimentConfig(base="none"))
    assert "held still" in fixed and "base_theta" not in fixed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": "spin"},
        {"base": "rotate_translate"},  # translation is left for later
        {"base_max_theta": 0.0},
        {"base_max_theta": float("inf")},
        {"auto_start": "no"},  # a string would be truthy and skip the start prompt
        {"gripper_depth": 1},
    ],
)
def test_bad_base_and_bool_config_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        Stretch4EmbodimentConfig(**kwargs)


def test_reset_homes_the_arm_frees_the_base_then_zeroes_it(build, driver) -> None:
    prompts: list[str] = []
    build(base="rotate", home_pose_file="start.json", prompts=prompts).reset(SCENE)
    assert driver.events == ["goto", "release_base", "zero_base"]
    assert len(prompts) == 1


def test_auto_start_zeroes_without_freeing_the_base(build, driver) -> None:
    build(base="rotate", auto_start=True).reset(SCENE)
    assert driver.events == ["zero_base"]


def test_fixed_base_is_never_freed_or_zeroed(build, driver) -> None:
    build(base="none").reset(SCENE)
    assert driver.events == []


def test_a_home_pose_outside_the_joint_limits_is_refused(build, driver) -> None:
    emb = build(
        home_pose_file="start.json",
        auto_start=True,
        load_home_pose=lambda _name: {**dict.fromkeys(JOINTS, 0.25), "lift": 12.0},
    )
    with pytest.raises(EmbodimentFault, match=r"lift 12.000 not in 0.000..1.200"):
        emb.reset(SCENE)
    assert not driver.gotos


def test_step_limits_the_base_per_step_and_to_its_envelope(build, driver) -> None:
    emb = build(base="rotate", auto_start=True)
    obs = emb.reset(SCENE)
    target = np.array(obs.state["joint_pos"], dtype=np.float64)
    target[6] = 3.0
    emb.step(Action(data=target))
    assert driver.commands[-1]["base_theta"] == pytest.approx(BASE_MAX_STEP["base_theta"])
    emb = build(base="rotate", auto_start=True, base_max_theta=0.1)
    emb.reset(SCENE)
    emb.step(Action(data=target))
    assert driver.commands[-1]["base_theta"] == pytest.approx(0.1)  # tighter than one step


def test_hold_policy_adopts_the_base_dimensions(build, driver) -> None:
    emb = build(base="rotate", polls=[False, True], auto_start=True)
    task = Task(name="hold", scenes=[SCENE], scorer=operator_scorer(), max_steps=5)
    logs = ir_eval(task, Stretch4HoldPolicy(), emb, sinks=[], seed=0)
    assert logs[0].status == "success"
    assert set(driver.commands[-1]) == {*JOINTS, "base_theta"}


def test_base_turn_rate_stops_within_tolerance() -> None:
    assert base_turn_rate(0.31, 0.3) is None


def test_base_turn_rate_is_proportional_then_capped() -> None:
    assert base_turn_rate(0.1, 0.0) == pytest.approx(0.2)
    assert base_turn_rate(2.0, 0.0) == pytest.approx(BASE_MAX_W)
    assert base_turn_rate(-2.0, 0.0) == pytest.approx(-BASE_MAX_W)

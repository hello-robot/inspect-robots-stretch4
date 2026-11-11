"""`Stretch4Embodiment` on a fake robot: reset, step, the operator prompts, config and docs."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from inspect_robots import Action, Scene
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4 import JOINTS, STATE_KEY, Stretch4Embodiment, Stretch4EmbodimentConfig
from inspect_robots_stretch4.config import MAX_STEP

SCENE = Scene(id="s", instruction="pick up the cup")


def test_constructor_touches_no_hardware(build, driver) -> None:
    build().close()
    assert driver.disconnects == 0


def test_reset_prompts_and_observes(build, driver) -> None:
    prompts: list[str] = []
    obs = build(prompts=prompts).reset(SCENE)
    assert len(prompts) == 1
    assert obs.instruction == "pick up the cup"
    np.testing.assert_array_equal(obs.state[STATE_KEY], driver.joints)
    assert set(obs.images) == {"head_right", "gripper_right"}
    assert driver.gotos == []


def test_auto_start_skips_the_prompt(build) -> None:
    prompts: list[str] = []
    build(prompts=prompts, auto_start=True).reset(SCENE)
    assert prompts == []


def test_session_gates_instead_of_stdin(build) -> None:
    class Session:
        def __init__(self) -> None:
            self.gates: list[str] = []

        def gate(self, prompt: str, *, hint: str | None = None) -> None:
            self.gates.append(prompt)

    prompts: list[str] = []
    emb, session = build(prompts=prompts), Session()
    emb.connect_operator_session(session)
    emb.reset(SCENE)
    assert prompts == [] and len(session.gates) == 1


def test_reset_drives_to_the_home_pose(build, driver) -> None:
    build(home_pose_file="start.json", auto_start=True).reset(SCENE)
    assert driver.gotos == [dict.fromkeys(JOINTS, 0.25)]


def test_reset_fails_when_home_is_not_reached(build, driver) -> None:
    driver.arrive = False
    with pytest.raises(EmbodimentFault, match="home pose"):
        build(home_pose_file="start.json", auto_start=True).reset(SCENE)


def test_step_limits_each_joint_to_max_step_from_the_measured_pose(build, driver) -> None:
    emb = build(auto_start=True)
    emb.reset(SCENE)
    start = driver.joints.copy()
    result = emb.step(Action(data=np.array([5.0, -1.0, 0.1, 0.2, 0.3, 500.0])))
    sent = driver.commands[-1]
    assert sent["lift"] == pytest.approx(start[0] + MAX_STEP[0])
    assert sent["arm"] == pytest.approx(start[1] - MAX_STEP[1])
    assert sent["grip_mm"] == pytest.approx(start[5] + MAX_STEP[5])
    assert sent["wrist_pitch"] == pytest.approx(0.2)  # within one step: unchanged
    assert not result.terminated
    assert result.info["sent"][0] == pytest.approx(start[0] + MAX_STEP[0])


def test_step_rejects_wrong_action_size(build) -> None:
    emb = build(auto_start=True)
    emb.reset(SCENE)
    with pytest.raises(ValueError, match="6 values"):
        emb.step(Action(data=np.zeros(9)))


def test_step_before_reset_is_an_error(build) -> None:
    with pytest.raises(RuntimeError, match="before reset"):
        build().step(Action(data=np.zeros(6)))


def test_enter_ends_the_episode_without_a_session(build) -> None:
    emb = build(polls=[False, True], auto_start=True)
    emb.reset(SCENE)
    assert not emb.step(Action(data=np.zeros(6))).terminated
    end = emb.step(Action(data=np.zeros(6)))
    assert end.terminated and end.termination_reason == "operator_end"


def test_wrong_camera_shape_is_a_fault(build) -> None:
    emb = build(auto_start=True, max_image_side=16)
    with pytest.raises(EmbodimentFault, match="shape"):
        emb.reset(SCENE)


def test_close_is_idempotent(build, driver) -> None:
    emb = build(auto_start=True)
    emb.reset(SCENE)
    emb.close()
    emb.close()
    assert driver.disconnects == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_low": (0.0,) * 5},
        {"joint_low": (2.0, 0, 0, 0, 0, 0), "joint_high": (1.0, 1, 1, 1, 1, 1)},
        {"cameras": ()},
        {"control_hz": 0},
        {"joint_high": (1.1, 0.5, 4.0, 0.5, 2.9, float("inf"))},
        {"max_image_side": 0},
        {"cameras": ("head_center",)},
        {"home_timeout_s": -1.0},
        {"gripper_depth": True, "cameras": ("head_left",)},
    ],
)
def test_bad_config_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        Stretch4EmbodimentConfig(**kwargs)


def test_gripper_sets_the_grip_limit() -> None:
    assert Stretch4EmbodimentConfig().joint_high[-1] == 77.0  # parallel gripper by default
    assert Stretch4EmbodimentConfig(gripper="stretch_gripper").joint_high[-1] == 150.0
    custom = (1.0, 0.5, 1.0, 0.5, 1.0, 60.0)
    assert (
        Stretch4EmbodimentConfig(gripper="parallel_gripper", joint_high=custom).joint_high == custom
    )
    with pytest.raises(ValueError, match="gripper"):
        Stretch4EmbodimentConfig(gripper="hook")


def test_unknown_cli_key_is_rejected() -> None:
    with pytest.raises(TypeError, match="use_base"):
        Stretch4EmbodimentConfig.from_kwargs(use_base=True)


def test_cli_tuple_strings_are_parsed() -> None:
    cfg = Stretch4EmbodimentConfig.from_kwargs(
        cameras="head_right, gripper_right", joint_low="0,0,-1,-1,-1,0"
    )
    assert cfg.cameras == ("head_right", "gripper_right")
    assert cfg.joint_low == (0.0, 0.0, -1.0, -1.0, -1.0, 0.0)


def test_cli_base_none_means_the_none_mode() -> None:
    # Through the CLI's own value parser, which turns "none" into None.
    from inspect_robots.defaults import _parse_value

    cfg = Stretch4EmbodimentConfig.from_kwargs(base=_parse_value("none"))
    assert cfg.base == "none"
    assert cfg.joints == JOINTS


def test_second_reset_reuses_the_connection(build, driver) -> None:
    connects = []
    emb = build(factory=lambda cfg: connects.append(cfg) or driver, auto_start=True)
    emb.reset(SCENE)
    emb.reset(SCENE)
    assert len(connects) == 1


def test_dead_stdin_at_the_start_prompt_halts(build) -> None:
    def no_stdin(_prompt: str) -> str:
        raise EOFError

    emb = build(input_fn=no_stdin)
    with pytest.raises(EmbodimentFault, match="auto_start"):
        emb.reset(SCENE)


def test_docs_name_the_configured_cameras() -> None:
    notes = Stretch4Embodiment(
        Stretch4EmbodimentConfig(cameras=("head_left", "gripper_right"))
    ).info.docs
    assert "`head_left` is the head's left eye" in notes
    assert "`gripper_right`" in notes and "head_right" not in notes
    assert "Depth:" not in notes


def test_gripper_depth_reaches_the_observation_and_the_docs(build, driver) -> None:
    depth = np.full((4, 6), 0.42, dtype=np.float32)
    driver.depth = {"gripper_right": depth}
    emb = build(auto_start=True, gripper_depth=True, cameras=("head_right", "gripper_right"))
    obs = emb.reset(SCENE)
    assert obs.extra["gripper_right_depth"] is depth
    assert "Depth: `gripper_right`" in emb.info.docs


def test_docs_describe_the_installed_gripper() -> None:
    sg4 = Stretch4Embodiment(Stretch4EmbodimentConfig(gripper="stretch_gripper")).info.docs
    pg4 = Stretch4Embodiment(Stretch4EmbodimentConfig()).info.docs
    assert "pull back toward the wrist" in sg4
    assert "fingers stay parallel" not in sg4
    assert "fingers stay parallel" in pg4 and "fully open is 77 mm" in pg4
    assert "pull back toward the wrist" not in pg4


def test_runstop_is_checked_before_commanding(build, driver) -> None:
    emb = build(auto_start=True)
    emb.reset(SCENE)
    driver.runstopped = True
    with pytest.raises(SafetyAbort, match="run-stop"):
        emb.step(Action(data=np.zeros(6)))
    assert driver.commands == []


def _logging_build(build, driver):
    """An embodiment whose driver logs while connecting, as BodyDriver does."""

    def factory(_cfg):
        logging.getLogger("inspect_robots_stretch4.driver").info("head camera connected")
        return driver

    return build(factory=factory, auto_start=True)


def test_driver_messages_go_to_the_session_not_stdout(build, driver, capsys) -> None:
    class Session:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write_line(self, text: str) -> None:
            self.lines.append(text)

    emb, session = _logging_build(build, driver), Session()
    emb.connect_operator_session(session)
    emb.reset(SCENE)
    assert session.lines == ["head camera connected"]
    assert capsys.readouterr().out == ""
    emb.close()


def test_driver_messages_print_without_a_session_and_stop_at_close(build, driver, capsys) -> None:
    logger = logging.getLogger("inspect_robots_stretch4")
    before = (logger.level, logger.propagate, list(logger.handlers))
    emb = _logging_build(build, driver)
    emb.reset(SCENE)
    assert capsys.readouterr().out == "head camera connected\n"
    emb.close()
    assert (logger.level, logger.propagate, list(logger.handlers)) == before

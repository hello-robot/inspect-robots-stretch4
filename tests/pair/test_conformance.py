"""The framework's checks on the pair, the agent's toolset on its spaces, and a full eval."""

from __future__ import annotations

import numpy as np
from inspect_robots import Scene, Task, operator_scorer
from inspect_robots import eval as ir_eval
from inspect_robots.conformance import assert_embodiment_conformant, check_embodiment
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.registry import resolve
from inspect_robots.tools import control_tools

from inspect_robots_stretch4.pair.embodiment import Stretch4PairEmbodiment
from inspect_robots_stretch4.pair.hold import Stretch4PairHoldPolicy
from inspect_robots_stretch4.pair.spaces import PAIR_JOINTS


def info():
    return Stretch4PairEmbodiment(cross_guard=False).info


def test_the_pair_is_conformant() -> None:
    assert_embodiment_conformant(info())
    assert check_embodiment(info()).issues == ()


def test_guardrails_build_from_the_action_space() -> None:
    DeltaLimitGuard(info().action_space)


def test_registered_by_entry_point_and_built_from_cli_kwargs(rig_file) -> None:
    emb = resolve(
        "embodiment",
        "stretch4-pair",
        rig_file=rig_file,
        robot2_cameras="gripper_left",
        gripper_depth=False,
    )
    assert isinstance(emb, Stretch4PairEmbodiment)
    assert "robot2_wrist" in emb.info.observation_space.camera_names
    assert isinstance(resolve("policy", "stretch4-pair-hold"), Stretch4PairHoldPolicy)


def test_the_agent_builds_its_tools_on_all_12_joints() -> None:
    space = info()
    tools = control_tools(space.action_space, space.observation_space, space.control_hz)
    assert tools.state_labels() == ("joint_pos", PAIR_JOINTS)
    move = next(s for s in tools.library.schemas() if s["function"]["name"] == "move_joints")
    assert all(label in move["function"]["description"] for label in PAIR_JOINTS)


def test_a_hold_eval_runs_on_both_robots(build_pair, pair_drivers) -> None:
    emb = build_pair(polls=[False, False, True])
    start = {s: d.joints.copy() for s, d in pair_drivers.items()}
    task = Task(
        name="pair-hold",
        scenes=[Scene(id="s", instruction="test")],
        scorer=operator_scorer(),
        max_steps=10,
    )
    logs = ir_eval(task, Stretch4PairHoldPolicy(), emb, sinks=[], seed=0)
    assert logs[0].status == "success"
    for s, d in pair_drivers.items():
        assert len(d.commands) == 3
        np.testing.assert_allclose(d.joints, start[s])


def test_a_hold_eval_with_robot2_behind_the_real_service(pair_drivers) -> None:
    import uuid

    import zmq

    from inspect_robots_stretch4.pair.config import Stretch4PairConfig
    from inspect_robots_stretch4.remote.client import connect
    from inspect_robots_stretch4.remote.service import DriverService

    ctx = zmq.Context.instance()
    endpoint = f"inproc://pair-{uuid.uuid4().hex}"
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    service = DriverService(endpoint, lambda cfg: d2, ctx=ctx)
    service.start()
    cfg = Stretch4PairConfig(
        robot2_host=endpoint,
        max_image_side=8,
        cross_guard=False,
        auto_start=True,
        gripper_depth=False,
    )

    def robot2(side_cfg):
        return connect(
            side_cfg,
            endpoint,
            keys_dir="unused",
            server_name="robot2",
            ctx=ctx,
            heartbeat_timeout_s=1.0,
        )

    emb = Stretch4PairEmbodiment(
        cfg,
        driver_factories={"robot1": lambda _c: d1, "robot2": robot2},
        poll_end=iter([False, False, True]).__next__,
    )
    task = Task(
        name="pair-wire",
        scenes=[Scene(id="s", instruction="test")],
        scorer=operator_scorer(),
        max_steps=10,
    )
    start = d2.joints.copy()
    try:
        logs = ir_eval(task, Stretch4PairHoldPolicy(), emb, sinks=[], seed=0)
    finally:
        service.close()
    assert logs[0].status == "success"
    assert len(d2.commands) == 3
    np.testing.assert_allclose(d2.joints, start)

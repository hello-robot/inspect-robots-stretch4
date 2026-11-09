"""The free `stretch4-hold` policy runs a whole eval without moving the robot."""

from __future__ import annotations

import numpy as np
from inspect_robots import Scene, Task, operator_scorer
from inspect_robots import eval as ir_eval

from inspect_robots_stretch4.hold import Stretch4HoldPolicy


def test_hold_policy_runs_and_holds_still(build, driver) -> None:
    emb = build(polls=[False, False, True], auto_start=True)
    start = driver.joints.copy()
    task = Task(
        name="hold",
        scenes=[Scene(id="s", instruction="test")],
        scorer=operator_scorer(),
        max_steps=10,
    )
    logs = ir_eval(task, Stretch4HoldPolicy(), emb, sinks=[], seed=0)
    assert logs[0].status == "success"
    assert len(driver.commands) == 3
    np.testing.assert_allclose(driver.joints, start)

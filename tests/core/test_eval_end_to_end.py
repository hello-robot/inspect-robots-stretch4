"""A full `inspect_robots.eval()` on the fake robot: compatibility check, guardrails, rollout,
operator end and scoring all compose with this embodiment."""

from __future__ import annotations

import numpy as np
from inspect_robots import (
    Action,
    ActionChunk,
    Observation,
    PolicyConfig,
    PolicyInfo,
    Scene,
    Task,
    TrialRecord,
    operator_scorer,
)
from inspect_robots import eval as ir_eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.spaces import Box, ObservationSpace

from inspect_robots_stretch4 import STATE_KEY
from inspect_robots_stretch4.config import MAX_STEP
from inspect_robots_stretch4.embodiment import Stretch4Embodiment


class RaiseLiftPolicy:
    """Asks for the lift 1 m higher every step; the guardrails must keep it small."""

    def __init__(self, space: Box) -> None:
        self.info = PolicyInfo(
            name="raise_lift",
            action_space=Box(shape=space.shape, semantics=space.semantics),
            observation_space=ObservationSpace(state_keys=frozenset({STATE_KEY})),
        )
        self.config = PolicyConfig(action_horizon=1)

    def reset(self, scene: Scene) -> None:
        pass

    def act(self, observation: Observation) -> ActionChunk:
        target = np.array(observation.state[STATE_KEY], dtype=np.float64)
        target[0] += 1.0
        return ActionChunk(actions=[Action(data=target)])


def _grade_yes(record: TrialRecord, scene: Scene) -> None:
    record.operator_judgement = "y"


def test_eval_end_to_end(build, driver) -> None:
    emb: Stretch4Embodiment = build(polls=[False, False, True], auto_start=True)
    space = emb.info.action_space
    task = Task(
        name="stretch4-e2e",
        scenes=[Scene(id="lift", instruction="raise the lift")],
        scorer=operator_scorer(),
        max_steps=10,
    )
    start = driver.joints[0]

    logs = ir_eval(
        task,
        RaiseLiftPolicy(space),
        emb,
        sinks=[],
        seed=0,
        approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)),
        before_scoring=_grade_yes,
    )

    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["operator"] == 1.0
    assert len(driver.commands) == 3  # the operator ended it on the third step
    # Each step moved the lift by at most the hard per-step backstop, never 1 m.
    lifts = [start] + [c["lift"] for c in driver.commands]
    assert max(np.diff(lifts)) <= MAX_STEP[0] + 1e-9

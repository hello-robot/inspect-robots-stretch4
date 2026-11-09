"""A free policy for hardware checks: it commands the pose the robot is already in.

Registered as the policy `stretch4-hold`, so a full run (connect, cameras, stepping,
run-stop, logs) can be checked on the robot without calling a model:

    inspect-robots run --instruction "test" --policy stretch4-hold --embodiment stretch4
"""

from __future__ import annotations

import dataclasses

import numpy as np
from inspect_robots import Action, ActionChunk, Observation, PolicyConfig, PolicyInfo, Scene
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace

from inspect_robots_stretch4.config import STATE_KEY, Stretch4EmbodimentConfig, action_box


class HoldPolicy:
    """Holds still: each action is the measured joint state, for any `joint_pos` embodiment."""

    def __init__(self, name: str, dim: int, semantics: ActionSemantics) -> None:
        self.info = PolicyInfo(
            name=name,
            action_space=Box(shape=(dim,), semantics=semantics),
            observation_space=ObservationSpace(state_keys=frozenset({STATE_KEY})),
        )
        self.config = PolicyConfig(action_horizon=1)

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Adopt the embodiment's action space: its number of base dimensions is configurable."""
        space = embodiment_info.action_space
        self.info = dataclasses.replace(
            self.info, action_space=Box(shape=space.shape, semantics=space.semantics)
        )

    def reset(self, scene: Scene) -> None:
        pass

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            actions=[Action(data=np.array(observation.state[STATE_KEY], dtype=np.float64))]
        )


class Stretch4HoldPolicy(HoldPolicy):
    """`stretch4-hold`: holds one Stretch 4 still."""

    def __init__(self) -> None:
        # The default config's space; bind() adopts the configured one.
        default = action_box(Stretch4EmbodimentConfig())
        super().__init__("stretch4-hold", default.shape[0], default.semantics)

"""The registered benchmark tasks carry their instruction and grading rubric."""

from __future__ import annotations

import pytest
from inspect_robots.registry import resolve

from inspect_robots_stretch4.tasks import (
    UNCAP_MARKER_INSTRUCTION,
    uncap_marker,
)


@pytest.mark.parametrize(
    ("factory", "instruction", "max_steps"),
    [
        (uncap_marker, UNCAP_MARKER_INSTRUCTION, 3600),
    ],
)
def test_task_defaults(factory, instruction, max_steps) -> None:
    task = factory()
    (scene,) = task.scenes
    assert scene.instruction == instruction
    assert "\n4 - " in scene.metadata["rubric"]
    assert task.epoch_spec.count == 20
    assert task.max_steps == max_steps


@pytest.mark.parametrize(
    "name",
    [
        "uncap-marker",
    ],
)
def test_task_is_registered(name) -> None:
    task = resolve("task", name, epochs=3)
    assert task.name == name
    assert task.epoch_spec.count == 3

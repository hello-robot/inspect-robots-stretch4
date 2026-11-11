"""The framework's own adapter checks (the same ones `inspect-robots doctor` runs)."""

from __future__ import annotations

from inspect_robots.conformance import assert_embodiment_conformant, check_embodiment
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.registry import resolve

from inspect_robots_stretch4 import Stretch4Embodiment


def test_embodiment_is_conformant() -> None:
    assert_embodiment_conformant(Stretch4Embodiment().info)


def test_no_conformance_warnings() -> None:
    assert check_embodiment(Stretch4Embodiment().info).issues == ()


def test_guardrails_build_from_the_action_space() -> None:
    DeltaLimitGuard(Stretch4Embodiment().info.action_space)


def test_registered_by_entry_point_and_built_from_cli_kwargs() -> None:
    emb = resolve("embodiment", "stretch4", auto_start=True, cameras="gripper_right", control_hz=15)
    assert isinstance(emb, Stretch4Embodiment)
    assert emb.info.control_hz == 15
    assert emb.info.observation_space.camera_names == frozenset({"gripper_right"})

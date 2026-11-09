"""Benchmark tasks for Stretch 4, graded by the operator against a staged rubric.

Registered by entry point (`inspect_robots_stretch4.tasks:<factory>`), so once installed:

    inspect-robots run --task uncap-marker --policy agent --embodiment stretch4-pair \\
        -E robot2_host=tcp://<robot 2's IP>:5600

`stationery` holds StationeryBench's bimanual uncap-marker task, for two robots
(`--embodiment stretch4-pair`).
"""

from __future__ import annotations

from inspect_robots_stretch4.tasks.stationery import (
    UNCAP_MARKER_INSTRUCTION,
    UNCAP_MARKER_RUBRIC,
    uncap_marker,
)

__all__ = [
    "UNCAP_MARKER_INSTRUCTION",
    "UNCAP_MARKER_RUBRIC",
    "uncap_marker",
]

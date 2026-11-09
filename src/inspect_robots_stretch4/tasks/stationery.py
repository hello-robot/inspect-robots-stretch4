"""StationeryBench's bimanual uncap-marker task, for two Stretch 4s (`stretch4-pair`) standing side
by side at one edge of the table, each turned in toward the centre (StationeryBench's layout).

    inspect-robots run --task uncap-marker --policy agent --embodiment stretch4-pair \\
        -E robot2_host=tcp://<robot 2's IP>:5600 -E base=rotate

The instruction, setup and five-stage rubric are StationeryBench's (`specs.py`); its
0/25/50/75/100 rubric is our stages 0-4. The instruction is the one its published runs used.
The task's 120 s limit is 3600 steps at our 30 Hz.
"""

from __future__ import annotations

from inspect_robots import Task

from inspect_robots_stretch4.tasks._staged import ANSWER_HINT, staged_task

UNCAP_MARKER_INSTRUCTION = (
    "Pull the cap off the sharpie and place them separately on the table, note that you can "
    "rotate the joints for the optimal gripping angle."
)

UNCAP_MARKER_RUBRIC = f"""\
Setup (StationeryBench's): cap the marker fully. Lay it flat at the centre of the table, roughly
midway between the two robots' bases, aligned front-to-back with the cap toward the robots.
Grade the highest stage reached; each must hold long enough to see on video.
0 - No purposeful contact with the marker
1 - Touches or moves the marker deliberately
2 - Holds the marker securely in one gripper
3 - Cap separated from the body, either part still held
4 - Cap and body both released on the table
{ANSWER_HINT}"""


def uncap_marker(epochs: int = 20, max_steps: int = 3600) -> Task:
    """StationeryBench's Uncap the marker."""
    return staged_task(
        "uncap-marker", UNCAP_MARKER_INSTRUCTION, UNCAP_MARKER_RUBRIC, epochs, max_steps
    )

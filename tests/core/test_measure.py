"""`stretch_inspect_record_poses` prints and saves labelled poses without commanding the robot."""

from __future__ import annotations

import json

from inspect_robots_stretch4.config import JOINTS
from inspect_robots_stretch4.measure import run


def test_run_saves_labelled_poses_only(driver, tmp_path) -> None:
    answers = iter(["", "lift at table, level", "q"])
    run(driver, tmp_path, JOINTS, input_fn=lambda _prompt: next(answers))

    lines = (tmp_path / "poses.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["label"] == "lift at table, level"
    assert record["joints"]["lift"] == 0.5
    for name in record["images"].values():
        assert (tmp_path / name).exists()
    assert driver.commands == []
    assert driver.gotos == []

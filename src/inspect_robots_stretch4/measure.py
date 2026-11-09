"""`stretch_inspect_record_poses`: record labelled poses (joints and camera images) without
moving the robot.

Drive the arm with the gamepad (or any other program), then type a label and press Enter: the
joints are printed and saved, with a PNG from each camera. Nothing is ever commanded.

    stretch_inspect_record_poses                 # saves to notes/measurements/
    stretch_inspect_record_poses --out some/dir --cameras head_right,gripper_right
    stretch_inspect_record_poses --base rotate   # also base_theta, the heading since it started
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from inspect_robots_stretch4.config import Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import Reading, Stretch4Driver

HELP = "type a label and Enter to save this pose, Enter alone to just print it, q to quit"


def format_joints(reading: Reading, joints: tuple[str, ...]) -> str:
    """One line of joint values, in action order and units."""
    return "  ".join(f"{j}={v:.4f}" for j, v in zip(joints, reading.joints, strict=True))


def save(reading: Reading, label: str, out: Path, joints: tuple[str, ...]) -> Path:
    """Append the joints to poses.jsonl and write each camera image; return the jsonl path."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "pose"
    record = {
        "label": label,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "joints": dict(zip(joints, (float(v) for v in reading.joints), strict=True)),
        "images": {},
    }
    for camera, rgb in reading.images.items():
        path = out / f"{stem}_{camera}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        record["images"][camera] = path.name
    jsonl = out / "poses.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return jsonl


def run(
    driver: Stretch4Driver,
    out: Path,
    joints: tuple[str, ...],
    input_fn: Callable[[str], str] = input,
) -> None:
    """The prompt loop: read, print, and save on a label, until q or end of input."""
    print(HELP)
    while True:
        try:
            label = input_fn("label> ").strip()
        except EOFError:
            return
        if label.lower() in ("q", "quit", "exit"):
            return
        reading = driver.read()
        print(format_joints(reading, joints))
        if label:
            print(f"saved '{label}' to {save(reading, label, out, joints)}")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs a Stretch 4
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("notes/measurements"))
    parser.add_argument("--cameras", default="head_right,gripper_right")
    parser.add_argument("--gripper", default="parallel_gripper")
    parser.add_argument(
        "--base",
        default="none",
        choices=["none", "rotate"],
        help="also record the base heading, relative to where the robot was when this started",
    )
    args = parser.parse_args(argv)
    # The driver logs its camera connects; show them as plain lines.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from inspect_robots_stretch4.driver import BodyDriver

    # Measuring never drives the base: the driver only chases a target once one is commanded.
    cfg = Stretch4EmbodimentConfig.from_kwargs(
        cameras=args.cameras, gripper=args.gripper, base=args.base
    )
    driver = BodyDriver(cfg)
    try:
        run(driver, args.out, cfg.joints)
    finally:
        driver.disconnect()


if __name__ == "__main__":  # pragma: no cover
    main()

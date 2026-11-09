"""The rig file: how the two robots stand, as measured by `stretch_inspect_pair_calibrate`.

{"v": 1, "layout": "opposed",
 "K_m": 0.913,                  robot1_arm + robot2_arm when the fingertips touch, level, yaw 0
 "lateral_yaw_rad": 0.02,       the wrist_yaw that lines the fingertips up (about 0 if square)
 "table_lift_m": {"robot1": 0.612, "robot2": 0.608},
 "wrist_height_band_m": 0.08,   |robot1_lift - robot2_lift| below this: the wrists can meet
 "yaw_band_rad": 0.3,
 "margin_m": 0.03,
 "measured": "2026-11-02", "notes": "..."}
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RIG_VERSION = 1
LAYOUTS = ("opposed",)  # the only layout with a guard and docs

# Conservative starting values.
DEFAULT_WRIST_HEIGHT_BAND_M = 0.08
DEFAULT_YAW_BAND_RAD = 0.3
DEFAULT_MARGIN_M = 0.03


@dataclass(frozen=True)
class Rig:
    layout: str
    K_m: float
    lateral_yaw_rad: float
    table_lift_m: dict[str, float]
    wrist_height_band_m: float = DEFAULT_WRIST_HEIGHT_BAND_M
    yaw_band_rad: float = DEFAULT_YAW_BAND_RAD
    margin_m: float = DEFAULT_MARGIN_M
    measured: str = ""
    notes: str = ""
    v: int = field(default=RIG_VERSION)

    def __post_init__(self) -> None:
        if self.v != RIG_VERSION:
            raise ValueError(f"rig file version {self.v}; this package reads {RIG_VERSION}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"no guard or docs for layout {self.layout!r}; known: {LAYOUTS}")
        numbers = {
            "K_m": self.K_m,
            "lateral_yaw_rad": self.lateral_yaw_rad,
            "wrist_height_band_m": self.wrist_height_band_m,
            "yaw_band_rad": self.yaw_band_rad,
            "margin_m": self.margin_m,
        }
        for name, value in numbers.items():
            if not _number(value):
                raise ValueError(f"rig {name} must be a finite number, got {value!r}")
        for name in ("K_m", "wrist_height_band_m", "yaw_band_rad"):
            if numbers[name] <= 0:
                raise ValueError(f"rig {name} must be > 0")
        if not 0 <= self.margin_m < self.K_m:
            raise ValueError("rig margin_m must be >= 0 and below K_m")
        if not isinstance(self.table_lift_m, dict) or set(self.table_lift_m) != {
            "robot1",
            "robot2",
        }:
            raise ValueError("rig table_lift_m needs exactly robot1 and robot2")
        if not all(_number(v) for v in self.table_lift_m.values()):
            raise ValueError("rig table_lift_m values must be finite numbers")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def load_rig(path: str | Path) -> Rig:
    try:
        data = json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read the rig file {path}: {e}") from None
    if not isinstance(data, dict):
        raise ValueError(f"the rig file {path} is not a JSON object")
    known = {f for f in Rig.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"the rig file {path} has unknown fields {sorted(unknown)}")
    try:
        return Rig(**data)
    except TypeError as e:
        raise ValueError(f"the rig file {path} is missing a field: {e}") from None


def save_rig(rig: Rig, path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.write_text(json.dumps(rig.to_json(), indent=2) + "\n")
    return out


def rig_facts(rig: Rig) -> str:
    """The measured layout as system-prompt lines, in the agent's joint terms."""
    lifts = rig.table_lift_m
    return (
        "Measured layout: with both wrists level and wrist_yaw near "
        f"{rig.lateral_yaw_rad:.2f}, the two sets of fingertips touch when robot1_arm + "
        f"robot2_arm = {rig.K_m:.3f} m; a smaller sum leaves a gap between them of the "
        "difference. The fingertips reach the table top at robot1_lift = "
        f"{lifts['robot1']:.3f} m and robot2_lift = {lifts['robot2']:.3f} m. While the wrists "
        f"are within {rig.wrist_height_band_m:.2f} m of the same height and both point at each "
        f"other, the arms are kept {rig.margin_m:.2f} m short of touching.\n"
    )

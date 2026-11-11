"""The cross-robot collision guard: keep two facing arms from meeting.

In the opposed layout both arms extend along one line toward each other. While the wrists are
at about the same height and both point along that line, the fingertips meet when
`robot1_arm + robot2_arm` reaches the rig's K, so the guard keeps that sum at most
`K - margin`. Wrists at different heights, as in a handover, pass freely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.types import Action

from inspect_robots_stretch4.config import JOINTS, NUM_JOINTS
from inspect_robots_stretch4.pair.config import SIDES
from inspect_robots_stretch4.pair.rig import Rig

LIFT, ARM, YAW = (JOINTS.index(j) for j in ("lift", "arm", "wrist_yaw"))
LIFTS, ARMS, YAWS = ((i, i + NUM_JOINTS) for i in (LIFT, ARM, YAW))
GUARD_NAME = "stretch4-pair cross guard"


def in_band(pose: npt.NDArray[np.float64], rig: Rig) -> bool:
    """Whether the wrists are near enough the same height, above the table, with either one
    pointing at the other, that the arms can meet. Heights are measured from each robot's own
    table-contact lift, since the two lifts' zeros differ."""
    heights = [pose[i] - rig.table_lift_m[s] for i, s in zip(LIFTS, SIDES, strict=True)]
    level = abs(heights[0] - heights[1]) < rig.wrist_height_band_m
    facing = any(abs(pose[i] - rig.lateral_yaw_rad) < rig.yaw_band_rad for i in YAWS)
    return bool(level and facing)


def cross_guard(
    target: npt.NDArray[np.float64], current: npt.NDArray[np.float64], rig: Rig
) -> npt.NDArray[np.float64]:
    """`target` (12-D, already clamped per robot), changed only as much as keeps the arms apart.

    In the band, an arm sum over `K - margin` is taken off whichever arm is extending more this
    step, then the other, never pulling either back past where it is now. If the arms already
    overlap that far (they were at different heights), the lifts and yaws hold instead, so the
    wrists don't move into each other.
    """
    out = np.array(target, dtype=np.float64)
    # Where the wrists are now counts as much as where they're going: the joints move together
    # and lag their targets, so arms extending while a lift leaves the band can still meet.
    if not (in_band(out, rig) or in_band(np.asarray(current, dtype=np.float64), rig)):
        return out
    excess = out[ARMS[0]] + out[ARMS[1]] - (rig.K_m - rig.margin_m)
    if excess <= 0:
        return out
    for i in sorted(ARMS, key=lambda i: out[i] - current[i], reverse=True):
        give = min(excess, max(0.0, out[i] - current[i]))
        out[i] -= give
        excess -= give
    if excess > 1e-9:
        for i in (*LIFTS, *YAWS):
            out[i] = current[i]
    return out


class CrossGuard:
    """The cross guard in the framework's guardrail chain, so the CLI's `guardrails:` banner
    shows it. `current` gives the robots' last measured joints (None before the first
    observation)."""

    def __init__(self, rig: Rig, current: Callable[[], npt.NDArray[np.float64] | None]) -> None:
        self._rig = rig
        self._current = current

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        current = self._current()
        data = np.asarray(action.data, dtype=np.float64)
        if current is None or data.shape != current.shape:
            return action
        guarded = cross_guard(data, current, self._rig)
        if np.array_equal(guarded, data):
            return action
        # A substituted target: later deltas must be measured from what will actually run.
        DeltaLimitGuard.rewind_reference(store, guarded)
        return replace(action, data=guarded, meta={**dict(action.meta), "cross_guard": True})

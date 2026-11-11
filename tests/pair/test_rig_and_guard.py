"""The rig file and the cross-robot guard (pure; no robots)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.types import Action

from inspect_robots_stretch4.pair.config import Stretch4PairConfig
from inspect_robots_stretch4.pair.guard import CrossGuard, cross_guard, in_band
from inspect_robots_stretch4.pair.rig import Rig, load_rig, rig_facts, save_rig
from inspect_robots_stretch4.pair.spaces import PAIR_SEMANTICS, action_box

RIG = Rig(
    layout="opposed", K_m=0.9, lateral_yaw_rad=0.0, table_lift_m={"robot1": 0.6, "robot2": 0.61}
)
LIMIT = RIG.K_m - RIG.margin_m


def pose(lift1=0.5, arm1=0.4, yaw1=0.0, lift2=0.5, arm2=0.4, yaw2=0.0) -> np.ndarray:
    return np.array([lift1, arm1, yaw1, 0, 0, 30, lift2, arm2, yaw2, 0, 0, 30], dtype=float)


def test_rig_round_trip_and_facts(tmp_path) -> None:
    path = save_rig(RIG, tmp_path / "rig.json")
    assert load_rig(path) == RIG
    facts = rig_facts(RIG)
    assert "robot1_arm + robot2_arm = 0.900 m" in facts and "robot2_lift = 0.610" in facts


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"layout": "side_by_side"}, "no guard"),
        ({"v": 2}, "version"),
        ({"K_m": -1}, "K_m"),
        ({"table_lift_m": {"robot1": 0.6}}, "table_lift_m"),
        ({"surprise": 1}, "unknown fields"),
    ],
)
def test_bad_rig_files_are_refused(tmp_path, change, match) -> None:
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({**RIG.to_json(), **change}))
    with pytest.raises(ValueError, match=match):
        load_rig(path)
    data = RIG.to_json()
    del data["K_m"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        load_rig(path)


def test_the_band_needs_level_wrists_facing_each_other() -> None:
    assert in_band(pose(), RIG)
    assert in_band(pose(lift2=0.51), RIG)  # level above the table: robot 2's lift zero is 1 cm up
    assert not in_band(pose(lift2=0.61), RIG)  # a handover: one wrist 10 cm higher
    assert in_band(pose(yaw1=0.5), RIG)  # one wrist still points at the other
    assert not in_band(pose(yaw1=0.5, yaw2=-0.5), RIG)


def test_outside_the_band_nothing_changes() -> None:
    target = pose(lift2=0.7, arm1=0.6, arm2=0.6)
    np.testing.assert_array_equal(cross_guard(target, pose(lift2=0.7), RIG), target)


def test_leaving_the_band_while_extending_is_still_guarded() -> None:
    current = pose(arm1=0.43, arm2=0.43)  # level now, sum 0.86
    target = pose(lift1=0.6, arm1=0.48, arm2=0.48)  # robot 1 lifts out of the band as both extend
    out = cross_guard(target, current, RIG)
    assert out[1] + out[7] == pytest.approx(LIMIT)


def test_the_advancing_arm_gives_way_and_the_other_stays() -> None:
    current = pose(arm1=0.4, arm2=0.45)
    target = pose(arm1=0.45, arm2=0.45)  # only robot 1 extends: the sum would be 0.90
    out = cross_guard(target, current, RIG)
    assert out[1] + out[7] == pytest.approx(LIMIT)
    assert out[7] == 0.45 and out[1] == pytest.approx(LIMIT - 0.45)


def test_both_advancing_the_one_advancing_more_gives_first() -> None:
    current = pose(arm1=0.40, arm2=0.42)
    out = cross_guard(pose(arm1=0.46, arm2=0.44), current, RIG)
    assert out[1] + out[7] == pytest.approx(LIMIT)
    assert out[7] == pytest.approx(0.44) and out[1] == pytest.approx(LIMIT - 0.44)


def test_overlapping_arms_cannot_be_lowered_into_each_other() -> None:
    current = pose(lift2=0.7, arm1=0.5, arm2=0.5)  # passing above one another
    target = pose(lift2=0.55, arm1=0.5, arm2=0.5)  # robot 2 lowers into the band
    out = cross_guard(target, current, RIG)
    assert out[6] == 0.7  # robot 2's lift holds
    np.testing.assert_array_equal(out[[1, 7]], [0.5, 0.5])


def test_the_guard_keeps_identity_or_rewinds_the_delta_limiter() -> None:
    box = action_box(Stretch4PairConfig(cross_guard=False))
    assert box.semantics == PAIR_SEMANTICS
    current = pose(arm1=0.4, arm2=0.45)
    guard = CrossGuard(RIG, lambda: current)
    store: dict = {}
    limiter = DeltaLimitGuard(box)
    calm = Action(data=pose(arm1=0.4, arm2=0.45))
    assert guard.review(limiter.review(calm, store), store) is calm
    pushy = Action(data=pose(arm1=0.45, arm2=0.45))
    out = guard.review(pushy, store)
    assert out.meta["cross_guard"] and out.data[1] == pytest.approx(LIMIT - 0.45)
    np.testing.assert_array_equal(next(iter(store.values())), out.data)
    assert CrossGuard(RIG, lambda: None).review(pushy, {}) is pushy

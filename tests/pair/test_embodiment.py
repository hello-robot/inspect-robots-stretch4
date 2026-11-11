"""`Stretch4PairEmbodiment` on two fake robots: splitting, clamping, stopping both, the guard."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from inspect_robots import Action, Scene
from inspect_robots.errors import EmbodimentFault, SafetyAbort

from inspect_robots_stretch4.config import JOINTS, MAX_STEP, STATE_KEY
from inspect_robots_stretch4.driver import LinkLost
from inspect_robots_stretch4.pair.config import LEFT_PITCH_MIN

SCENE = Scene(id="s", instruction="uncap the marker")


def act(robot1, robot2) -> Action:
    return Action(data=np.concatenate([robot1, robot2]).astype(np.float64))


def test_the_action_reaches_each_robot_in_order(build_pair, pair_drivers) -> None:
    emb = build_pair()
    obs = emb.reset(SCENE)
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    np.testing.assert_array_equal(obs.state[STATE_KEY], np.concatenate([d1.joints, d2.joints]))
    a1, a2 = d1.joints + 0.01, d2.joints - 0.01
    result = emb.step(act(a1, a2))
    np.testing.assert_allclose([d1.commands[-1][j] for j in JOINTS], a1)
    np.testing.assert_allclose([d2.commands[-1][j] for j in JOINTS], a2)
    np.testing.assert_allclose(result.info["sent"], np.concatenate([a1, a2]))
    assert set(result.observation.images) == {
        "robot1_head",
        "robot1_wrist",
        "robot2_head",
        "robot2_wrist",
    }


def test_each_robot_is_clamped_from_its_own_joints(build_pair, pair_drivers) -> None:
    emb = build_pair()
    emb.reset(SCENE)
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    start1, start2 = d1.joints.copy(), d2.joints.copy()
    emb.step(act(start1 + 1.0, start2 + 1.0))
    assert d1.commands[-1]["lift"] == pytest.approx(start1[0] + MAX_STEP[0])
    assert d2.commands[-1]["lift"] == pytest.approx(start2[0] + MAX_STEP[0])


def test_depth_extras_are_named_for_each_wrist(build_pair, pair_drivers) -> None:
    for d in pair_drivers.values():
        d.depth = {"gripper_right": np.zeros((5, 8), np.float32)}
    obs = build_pair().reset(SCENE)
    assert set(obs.extra) == {"robot1_wrist_depth", "robot2_wrist_depth"}


@pytest.mark.parametrize("stopped", ["robot1", "robot2"])
def test_a_runstop_on_either_stops_both_before_any_command(
    build_pair, pair_drivers, stopped
) -> None:
    emb = build_pair()
    emb.reset(SCENE)
    pair_drivers[stopped].runstopped = True
    other = pair_drivers["robot2" if stopped == "robot1" else "robot1"]
    with pytest.raises(SafetyAbort, match=f"{stopped}'s run-stop"):
        emb.step(act(np.zeros(6), np.zeros(6)))
    assert other.runstop_triggers == 1
    assert all(not d.commands for d in pair_drivers.values())


@pytest.mark.parametrize("lost", ["robot1", "robot2"])
def test_a_lost_lease_on_either_stops_the_other(build_pair, pair_drivers, lost) -> None:
    emb = build_pair()
    emb.reset(SCENE)

    def taken(_targets):
        raise SafetyAbort("another program (teleop) holds Stretch's command lease")

    pair_drivers[lost].command = taken
    other = pair_drivers["robot2" if lost == "robot1" else "robot1"]
    with pytest.raises(SafetyAbort, match=f"^{lost}: another program"):
        emb.step(act(pair_drivers["robot1"].joints, pair_drivers["robot2"].joints))
    assert other.runstop_triggers == 1


def test_a_narrowed_wrist_yaw_holds_for_steps(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    d1.joints[2] = 0.5
    emb = build_pair(robot1_wrist_yaw_min=0.5)
    emb.reset(SCENE)
    a1, a2 = d1.joints.copy(), d2.joints.copy()
    a1[2] = a2[2] = -1.0
    emb.step(act(a1, a2))
    assert d1.commands[-1]["wrist_yaw"] == 0.5 and d2.commands[-1]["wrist_yaw"] < 0.0


def test_a_home_pose_outside_a_robots_limits_is_refused(build_pair, pair_drivers) -> None:
    emb = build_pair(
        robot1_wrist_yaw_min=0.5, robot1_home_pose_file="h.json", robot2_home_pose_file="h.json"
    )
    with pytest.raises(EmbodimentFault, match=r"robot1.*wrist_yaw 0.250 not in 0.500"):
        emb.reset(SCENE)


LEFT_HOME = [0.5, 0.1, math.pi, math.pi, math.pi, 30.0]  # stretch4_body's left-handed home


def test_a_left_handed_wrist_keeps_the_right_handed_meanings(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    d1.joints = np.array(LEFT_HOME)
    emb = build_pair(robot1_handedness="left")
    obs = emb.reset(SCENE)
    state = obs.state[STATE_KEY]
    np.testing.assert_allclose(state[2:5], 0.0, atol=1e-12)  # folded home reads as yaw/pitch/roll 0
    a1, a2 = state[:6].copy(), d2.joints.copy()
    a1[2:5] = [-0.1, 0.2, 0.1]
    emb.step(act(a1, a2))
    sent = d1.commands[-1]
    assert sent["wrist_yaw"] == pytest.approx(math.pi - 0.1)
    assert sent["wrist_pitch"] == pytest.approx(math.pi - 0.2)
    assert sent["wrist_roll"] == pytest.approx(math.pi + 0.1)
    assert d2.commands[-1]["wrist_yaw"] == pytest.approx(a2[2])  # robot 2 is right-handed


def test_a_left_handed_wrist_cannot_unfold(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    d1.joints = np.array([0.5, 0.1, math.pi, LEFT_PITCH_MIN + 0.01, math.pi, 30.0])
    emb = build_pair(robot1_handedness="left")
    obs = emb.reset(SCENE)
    a1, a2 = obs.state[STATE_KEY][:6].copy(), d2.joints.copy()
    a1[3] = 3.0  # far past straight down, toward the right-handed solution
    emb.step(act(a1, a2))
    assert d1.commands[-1]["wrist_pitch"] == pytest.approx(LEFT_PITCH_MIN)
    box = emb.info.action_space
    assert box.high[3] == pytest.approx(1.50) and box.high[9] > 4.0
    assert "robot 1's wrist is folded over (left-handed)" in emb.info.docs
    assert "robot1_wrist_pitch is limited to -1.12..1.50 rad" in emb.info.docs
    assert "robot 2's wrist is folded" not in emb.info.docs


def test_a_runstop_while_homing_ends_both(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    d1.on_goto = lambda: setattr(d1, "runstopped", True)  # pressed as robot 1 starts home
    d2.on_goto = lambda: d2.home_until_runstopped(2.0)
    emb = build_pair(robot1_home_pose_file="h.json", robot2_home_pose_file="h.json")
    t0 = time.monotonic()
    with pytest.raises(SafetyAbort, match="run-stop"):
        emb.reset(SCENE)
    assert d2.runstop_triggers == 1 and time.monotonic() - t0 < 3.0


def test_if_one_robot_fails_to_connect_the_other_is_closed(build_pair, pair_drivers) -> None:
    def unreachable(_cfg):
        raise EmbodimentFault("no reply to 'hello' from tcp://10.0.0.2:5600 within 60.0s")

    d1 = pair_drivers["robot1"]
    emb = build_pair(factories={"robot1": lambda _c: d1, "robot2": unreachable})
    with pytest.raises(EmbodimentFault, match=r"^robot2: no reply"):
        emb.reset(SCENE)
    assert d1.disconnects == 1


def test_robot2_needs_a_host(pair_drivers) -> None:
    from inspect_robots_stretch4.pair.embodiment import Stretch4PairEmbodiment

    d1 = pair_drivers["robot1"]
    emb = Stretch4PairEmbodiment(driver_factories={"robot1": lambda _c: d1}, auto_start=True)
    with pytest.raises(EmbodimentFault, match="robot2_host is not set"):
        emb.reset(SCENE)


def test_the_cross_guard_holds_facing_arms_apart(build_pair, pair_drivers, rig_file) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    d1.joints = np.array([0.5, 0.42, 0.0, 0.0, 0.0, 30.0])
    d2.joints = np.array([0.5, 0.43, 0.0, 0.0, 0.0, 30.0])  # sum 0.85; the limit is 0.87
    emb = build_pair(rig_file=rig_file, cross_guard=True)
    emb.reset(SCENE)
    extend = np.array([0, 0.04, 0, 0, 0, 0])
    result = emb.step(act(d1.joints + extend, d2.joints))
    assert result.info["cross_guard"]
    assert d1.commands[-1]["arm"] + d2.commands[-1]["arm"] == pytest.approx(0.87)
    d1.joints[1], d2.joints[0] = 0.42, 0.7  # robot 2's wrist 20 cm higher: no guard
    emb.reset(SCENE)
    result = emb.step(act(d1.joints + extend, d2.joints))
    assert not result.info["cross_guard"]
    assert d1.commands[-1]["arm"] == pytest.approx(0.46)


def test_guardrails_show_the_guard_or_warn_it_is_off(build_pair, rig_file) -> None:
    on = build_pair(rig_file=rig_file, cross_guard=True)
    contribution = on.contribute_guards(on.info.action_space)
    assert [name for name, _ in contribution.guards] == ["stretch4-pair cross guard"]
    off = build_pair(cross_guard=False)
    contribution = off.contribute_guards(off.info.action_space)
    assert contribution.guards == () and "OFF" in contribution.warnings[0]


def test_the_docs_explain_who_is_who(build_pair, rig_file) -> None:
    docs = build_pair(rig_file=rig_file, cross_guard=True).info.docs
    assert "robot1_head" in docs and "near (bottom) edge" in docs
    assert "robot1_arm + robot2_arm = 0.900 m" in docs
    assert "facing each other" not in build_pair().info.docs  # no rig: no layout claims


def test_operator_end_and_close(build_pair, pair_drivers) -> None:
    emb = build_pair(polls=[False, True])
    emb.reset(SCENE)
    hold = act(pair_drivers["robot1"].joints, pair_drivers["robot2"].joints)
    assert not emb.step(hold).terminated
    assert emb.step(hold).termination_reason == "operator_end"
    emb.close()
    emb.close()
    assert all(d.disconnects == 1 for d in pair_drivers.values())


def test_a_lost_lease_while_homing_stops_the_other_at_once(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]

    def lease_taken() -> None:
        raise SafetyAbort("another program (teleop) holds Stretch's command lease")

    d1.on_goto = lease_taken
    d2.on_goto = d2.home_until_runstopped
    emb = build_pair(robot1_home_pose_file="h.json", robot2_home_pose_file="h.json")
    t0 = time.monotonic()
    with pytest.raises(SafetyAbort, match=r"^robot1: another program"):
        emb.reset(SCENE)
    assert d2.runstop_triggers == 1 and time.monotonic() - t0 < 2.0


def test_losing_track_while_homing_stops_both(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(robot1_home_pose_file="h.json", robot2_home_pose_file="h.json")
    emb.reset(SCENE)  # connected and homed once; robot 2 goes quiet during the next homing
    d1.on_goto = d1.home_until_runstopped
    d2.on_goto = d2.home_until_runstopped
    homed_before, runstopped = len(d2.gotos), d2.is_runstopped

    def lost() -> bool:
        if len(d2.gotos) > homed_before:
            raise EmbodimentFault("lost the Stretch 4 at tcp://10.0.0.2:5600: no heartbeat")
        return runstopped()

    d2.is_runstopped = lost
    with pytest.raises(EmbodimentFault, match="lost the Stretch 4"):
        emb.reset(SCENE)
    assert d1.runstop_triggers == 1 and d2.runstop_triggers == 1
    emb.close()  # waits for both homing workers before disconnecting


def test_a_failed_command_on_one_robot_stops_the_other(build_pair, pair_drivers) -> None:
    emb = build_pair()
    emb.reset(SCENE)

    def timed_out(_targets):
        raise EmbodimentFault("no reply to 'command' from tcp://10.0.0.2:5600 within 1.0s")

    pair_drivers["robot2"].command = timed_out
    with pytest.raises(EmbodimentFault, match=r"^robot2: no reply"):
        emb.step(act(pair_drivers["robot1"].joints, pair_drivers["robot2"].joints))
    assert pair_drivers["robot1"].runstop_triggers == 1


def test_a_failed_connect_stops_nobody(build_pair, pair_drivers) -> None:
    def locked(_cfg):
        raise SafetyAbort("another program holds the control lock")

    d1 = pair_drivers["robot1"]
    emb = build_pair(factories={"robot1": locked, "robot2": lambda _c: pair_drivers["robot2"]})
    with pytest.raises(SafetyAbort):
        emb.reset(SCENE)
    assert pair_drivers["robot2"].runstop_triggers == 0 and d1.runstop_triggers == 0


def test_non_finite_actions_are_refused(build_pair, pair_drivers) -> None:
    emb = build_pair()
    emb.reset(SCENE)
    bad = act(pair_drivers["robot1"].joints, pair_drivers["robot2"].joints).data.copy()
    bad[3] = np.nan
    with pytest.raises(SafetyAbort, match="non-finite"):
        emb.step(Action(data=bad))
    assert all(not d.commands for d in pair_drivers.values())


# -- a lost link to a served robot: hold both, retry, then carry on or end -----------------

LOST = "lost the Stretch 4 at tcp://10.0.0.2:5600: no heartbeat for 2.4s"


def drop_link(driver, reconnects_needed: int, on_reconnect=None) -> list[float]:
    """Make `driver` look like a served robot whose link is down until it has been asked to
    reconnect `reconnects_needed` times; returns the timeouts it was asked with."""
    asks: list[float] = []
    state = {"lost": True}
    real = driver.is_runstopped

    def is_runstopped() -> bool:
        if state["lost"]:
            raise LinkLost(LOST)
        return real()

    def reconnect(timeout_s: float) -> None:
        asks.append(timeout_s)
        if on_reconnect is not None:
            on_reconnect()
        if len(asks) < reconnects_needed:
            raise LinkLost("no reply to 'hello' from tcp://10.0.0.2:5600 within 1.0s")
        state["lost"] = False

    driver.is_runstopped = is_runstopped
    driver.reconnect = reconnect
    return asks


def test_a_brief_link_loss_holds_both_then_carries_on(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retry_s=0.5)
    emb.reset(SCENE)
    asks = drop_link(d2, reconnects_needed=3)
    sent = len(d1.commands), len(d2.commands)
    result = emb.step(act(d1.joints, d2.joints))
    assert len(asks) == 3
    assert result.info["link_pause_s"] > 0
    assert d1.runstop_triggers == 0 and d2.runstop_triggers == 0
    assert (len(d1.commands), len(d2.commands)) == sent  # the paused step moved nobody
    assert result.observation.state[STATE_KEY].shape == (12,)  # a fresh reading of both
    emb.step(act(d1.joints, d2.joints))
    assert (len(d1.commands), len(d2.commands)) == (sent[0] + 1, sent[1] + 1)


def test_a_link_lost_during_reset_is_ridden_out_too(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retry_s=0.5)
    asks = drop_link(d2, reconnects_needed=2)  # lost before the first run-stop check
    obs = emb.reset(SCENE)
    assert len(asks) == 2
    assert d1.runstop_triggers == 0 and d2.runstop_triggers == 0
    assert obs.state[STATE_KEY].shape == (12,)


def test_a_link_that_stays_lost_ends_the_trial_after_every_attempt(
    build_pair, pair_drivers
) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retries=3, link_retry_s=0.1)
    emb.reset(SCENE)
    drop_link(d2, reconnects_needed=10**6)
    t0 = time.monotonic()
    with pytest.raises(EmbodimentFault, match=r"lost the Stretch 4.*3 attempts of 0\.1 s") as e:
        emb.step(act(d1.joints, d2.joints))
    assert not isinstance(e.value, LinkLost)  # final: nothing retries it again
    assert 0.3 <= time.monotonic() - t0 < 1.5
    assert d1.runstop_triggers == 0 and not d1.commands  # both held; nobody was run-stopped


def test_a_runstop_during_the_pause_stops_both(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retry_s=0.5)
    emb.reset(SCENE)

    def press() -> None:
        d1.runstopped = True

    drop_link(d2, reconnects_needed=10**6, on_reconnect=press)
    with pytest.raises(SafetyAbort, match="robot1's run-stop"):
        emb.step(act(d1.joints, d2.joints))
    assert d2.runstop_triggers == 1  # asked to stop too, in case its link is back


def test_a_link_lost_mid_command_holds_instead_of_runstopping(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retry_s=0.5)
    emb.reset(SCENE)
    real = d2.command

    def timed_out(targets):
        d2.command = real
        raise LinkLost("no reply to 'command' from tcp://10.0.0.2:5600 within 1.0s")

    d2.command = timed_out
    d2.reconnect = lambda timeout_s: None
    result = emb.step(act(d1.joints, d2.joints))
    assert d1.runstop_triggers == 0
    assert result.info["link_pause_s"] >= 0


def test_no_retries_ends_the_trial_at_once(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retries=0)
    emb.reset(SCENE)
    asks = drop_link(d2, reconnects_needed=1)
    with pytest.raises(EmbodimentFault, match="lost the Stretch 4"):
        emb.step(act(d1.joints, d2.joints))
    assert asks == []


@pytest.mark.parametrize("bad", [{"link_retries": -1}, {"link_retry_s": 0.0}])
def test_link_retry_settings_are_checked(build_pair, bad) -> None:
    with pytest.raises(ValueError, match="link_retr"):
        build_pair(**bad)


# -- turning bases (-E base=rotate) ---------------------------------------------------------


def build_with_bases(build_pair, pair_drivers, **cfg):
    """Each fake robot takes the config's joints, so base_theta is the 7th."""
    factories = {s: d.configure for s, d in pair_drivers.items()}
    return build_pair(factories=factories, base="rotate", **cfg)


def with_base(arm_joints, base: float) -> np.ndarray:
    return np.concatenate([np.asarray(arm_joints, dtype=np.float64)[:6], [base]])


def test_turning_bases_are_zeroed_and_turn_only_with_the_arm_in(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_with_bases(build_pair, pair_drivers)
    obs = emb.reset(SCENE)
    assert "zero_base" in d1.events and "zero_base" in d2.events
    assert obs.state[STATE_KEY].shape == (14,)
    # robot 1's arm is in (0.1 m), robot 2's is out (0.2 m > 0.15 m): only robot 1 turns.
    result = emb.step(act(with_base(d1.joints, 0.1), with_base(d2.joints, 0.1)))
    assert d1.commands[-1]["base_theta"] == pytest.approx(0.1)
    assert d2.commands[-1]["base_theta"] == pytest.approx(0.0)
    assert result.info["base_held"] == ["robot2"]


def test_a_base_holds_while_its_target_extends_the_arm(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_with_bases(build_pair, pair_drivers, base_turn_max_arm=0.12)
    emb.reset(SCENE)
    reach = with_base(d1.joints, 0.1)
    reach[1] = 0.15  # arm 0.1 -> 0.15, past 0.12, in the same step as the turn
    result = emb.step(act(reach, with_base(d2.joints, 0.0)))
    assert d1.commands[-1]["base_theta"] == pytest.approx(0.0)
    assert result.info["base_held"] == ["robot1"]


def test_a_base_turns_no_further_than_its_limit(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_with_bases(build_pair, pair_drivers)
    emb.reset(SCENE)
    for _ in range(20):  # 0.2 rad per step at most, then the 180 degree limit
        emb.step(act(with_base(d1.joints, 5.0), with_base(d2.joints, 0.0)))
    assert d1.commands[-1]["base_theta"] == pytest.approx(math.pi)


def test_the_docs_explain_turning_bases(build_pair, pair_drivers) -> None:
    emb = build_with_bases(build_pair, pair_drivers)
    docs = emb.info.docs
    assert "robot1_base_theta" in docs and "180 degrees" in docs and "0.15 m" in docs
    assert "base_theta" not in build_pair().info.docs


@pytest.mark.parametrize("pressed", ["robot1", "robot2"])
def test_a_runstop_between_steps_stops_the_other_at_once(build_pair, pair_drivers, pressed) -> None:
    emb = build_pair()
    emb.reset(SCENE)
    other = pair_drivers["robot2" if pressed == "robot1" else "robot1"]
    pair_drivers[pressed].press_runstop()  # while the model is thinking: no step running
    assert other.runstop_requests == 1 and other.is_runstopped()
    # Both are run-stopped now; the next step names the robot that was pressed.
    with pytest.raises(SafetyAbort, match=f"{pressed}'s run-stop"):
        emb.step(act(np.zeros(6), np.zeros(6)))
    assert all(not d.commands for d in pair_drivers.values())


def test_a_runstop_after_close_asks_nothing_of_the_other(build_pair, pair_drivers) -> None:
    emb = build_pair()
    emb.reset(SCENE)
    emb.close()
    pair_drivers["robot1"].press_runstop()
    assert pair_drivers["robot2"].runstop_requests == 0


def test_a_local_runstop_is_reported_even_while_the_other_link_is_lost(
    build_pair, pair_drivers
) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    emb = build_pair(link_retry_s=0.5)
    emb.reset(SCENE)
    asks = drop_link(d2, reconnects_needed=10**6)
    d1.runstopped = True
    with pytest.raises(SafetyAbort, match="robot1's run-stop"):
        emb.step(act(d1.joints, d2.joints))
    assert not asks  # reported at once, not after waiting on robot 2's link
    assert d2.runstop_triggers == 1


def test_a_missing_home_pose_is_reported_before_either_robot_moves(
    build_pair, pair_drivers
) -> None:
    def load(path: str) -> dict[str, float]:
        if path == "missing.json":
            raise FileNotFoundError(2, "No such file or directory", path)
        return dict.fromkeys(JOINTS, 0.25)

    emb = build_pair(
        robot1_home_pose_file="missing.json", robot2_home_pose_file="h.json", load_home_pose=load
    )
    with pytest.raises(EmbodimentFault, match=r"robot1.*can't read the home pose 'missing.json'"):
        emb.reset(SCENE)
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    assert not d1.gotos and not d2.gotos
    assert d1.runstop_triggers == 0 and d2.runstop_triggers == 0


def test_a_failure_while_homing_is_reported_as_itself(build_pair, pair_drivers) -> None:
    d1, d2 = pair_drivers["robot1"], pair_drivers["robot2"]
    d1.arrive = False  # robot 1 doesn't reach its pose
    d2.on_goto = d2.home_until_runstopped
    emb = build_pair(robot1_home_pose_file="h.json", robot2_home_pose_file="h.json")
    with pytest.raises(EmbodimentFault, match=r"robot1.*did not reach the home pose"):
        emb.reset(SCENE)
    assert d2.runstop_triggers == 1  # the other robot is still stopped

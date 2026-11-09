"""The min-RTT clock offset estimator."""

from __future__ import annotations

import pytest

from inspect_robots_stretch4.net.clock import ClockOffset


def test_the_fastest_ping_sets_the_offset() -> None:
    clock = ClockOffset()
    assert not clock.ready
    # The remote clock is 100 s ahead; replies are stamped mid-flight on symmetric pings.
    clock.add(t_send=1.0, t_remote=101.05, t_recv=1.1)  # rtt 0.1, exact
    clock.add(t_send=2.0, t_remote=102.4, t_recv=2.5)  # rtt 0.5, a slow asymmetric spike
    assert clock.ready
    assert clock.offset == pytest.approx(100.0)
    assert clock.rtt == pytest.approx(0.1)
    assert clock.to_local(150.0) == pytest.approx(50.0)


def test_old_pings_leave_the_window() -> None:
    clock = ClockOffset(window=2)
    clock.add(0.0, 10.0, 0.0)  # rtt 0, offset 10
    clock.add(1.0, 21.1, 1.2)  # rtt 0.2, offset 20
    clock.add(2.0, 22.1, 2.2)  # rtt 0.2, offset 20: the first ping is gone
    assert clock.offset == pytest.approx(20.0)


def test_errors() -> None:
    clock = ClockOffset()
    with pytest.raises(RuntimeError, match="no pings"):
        _ = clock.offset
    with pytest.raises(ValueError, match="before it was sent"):
        clock.add(2.0, 0.0, 1.0)

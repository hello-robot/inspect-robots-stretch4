"""Mapping another machine's monotonic clock onto this one, from ping round trips."""

from __future__ import annotations

from collections import deque


class ClockOffset:
    """The remote clock minus the local one, from the ping with the smallest round trip.

    Each ping gives `offset = t_remote - (t_send + t_recv) / 2`, wrong by at most half its
    round trip. The fastest of the last `window` pings is the tightest bound, and Wi-Fi spikes
    only ever slow a ping down, so they never become the estimate.
    """

    def __init__(self, window: int = 25) -> None:
        self._samples: deque[tuple[float, float]] = deque(maxlen=window)  # (rtt, offset)

    def add(self, t_send: float, t_remote: float, t_recv: float) -> None:
        """Record one ping: local send time, the remote's reply time, local receive time."""
        rtt = t_recv - t_send
        if rtt < 0:
            raise ValueError(f"a ping can't arrive before it was sent (rtt {rtt:.6f}s)")
        self._samples.append((rtt, t_remote - (t_send + t_recv) / 2.0))

    @property
    def ready(self) -> bool:
        return bool(self._samples)

    @property
    def offset(self) -> float:
        """Seconds to subtract from a remote time to get the local time."""
        return self._best()[1]

    @property
    def rtt(self) -> float:
        """The round trip of the ping the offset comes from."""
        return self._best()[0]

    def to_local(self, t_remote: float) -> float:
        return t_remote - self.offset

    def _best(self) -> tuple[float, float]:
        if not self._samples:
            raise RuntimeError("no pings yet")
        return min(self._samples)

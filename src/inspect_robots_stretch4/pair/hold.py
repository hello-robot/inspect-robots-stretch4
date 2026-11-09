"""`stretch4-pair-hold`: holds both robots still, for hardware checks without a model.

inspect-robots run --instruction "test" --policy stretch4-pair-hold --embodiment stretch4-pair
"""

from __future__ import annotations

from inspect_robots_stretch4.hold import HoldPolicy
from inspect_robots_stretch4.pair.spaces import PAIR_DIM, PAIR_SEMANTICS


class Stretch4PairHoldPolicy(HoldPolicy):
    def __init__(self) -> None:
        super().__init__("stretch4-pair-hold", PAIR_DIM, PAIR_SEMANTICS)

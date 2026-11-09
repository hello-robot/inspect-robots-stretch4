"""inspect-robots-stretch4: the Hello Robot Stretch 4 as an Inspect Robots embodiment.

Registers the embodiment `stretch4` by entry point, so once installed:

    inspect-robots doctor --embodiment stretch4
    inspect-robots run --instruction "pick up the cup" --policy <policy> --embodiment stretch4
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from inspect_robots_stretch4.config import JOINTS, STATE_KEY, Stretch4EmbodimentConfig
from inspect_robots_stretch4.driver import Reading, Stretch4Driver
from inspect_robots_stretch4.embodiment import Stretch4Embodiment

try:
    __version__ = _version("inspect-robots-stretch4")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "JOINTS",
    "STATE_KEY",
    "Reading",
    "Stretch4Driver",
    "Stretch4Embodiment",
    "Stretch4EmbodimentConfig",
]

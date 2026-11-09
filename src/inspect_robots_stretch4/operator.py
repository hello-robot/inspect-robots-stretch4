"""Operator prompts for real runs.

Under the inspect-robots CLI the framework's operator session owns the terminal (the
embodiment's `connect_operator_session` hook): it gates readiness and ends episodes on Esc.
These are the fallbacks for direct `eval()`/`rollout()` calls, where no session is connected:
a plain Enter-to-start prompt and Enter-to-end polling of stdin.

`OperatorLog` shows this package's log messages (camera connects, link warnings) to the
operator: through the session's scrollback once it owns the terminal, since printing then would
garble its status line, and on stdout otherwise.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

from inspect_robots.errors import EmbodimentFault

PACKAGE_LOGGER = "inspect_robots_stretch4"

READY_PROMPT = "Set up the scene, keep a hand on the run-stop, then press Enter to start..."
READY_HINT = "pass -E auto_start=true to skip the start prompt"


def wait_ready(input_fn: Callable[[str], str] = input) -> None:
    """Block until the operator presses Enter; a dead stdin halts the eval."""
    try:
        input_fn(READY_PROMPT)
    except (EOFError, OSError) as exc:
        raise EmbodimentFault(
            f"the start prompt could not read stdin (no terminal?); {READY_HINT}"
        ) from exc
    _drain_stdin()


def _drain_stdin() -> None:
    """Drop buffered input so a stale newline does not end the first step."""
    if not sys.stdin.isatty():
        return
    import select  # pragma: no cover - TTY-bound

    while select.select([sys.stdin], [], [], 0)[0]:  # pragma: no cover - TTY-bound
        sys.stdin.readline()  # pragma: no cover - TTY-bound


class OperatorLog(logging.Handler):
    """Writes each record to the operator session if there is one, else prints it."""

    def __init__(self, session: Callable[[], Any]) -> None:
        super().__init__(logging.INFO)
        self.setFormatter(logging.Formatter("%(message)s"))
        self._session = session
        self._saved: tuple[int, bool] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text, session = self.format(record), self._session()
            if session is not None:
                session.write_line(text)
            else:
                print(text)
        except Exception:
            self.handleError(record)

    def attach(self) -> None:
        """Route the package's INFO and above here, and only here (idempotent).

        Replaces an earlier embodiment's handler that was never detached (no close()), taking
        over the logger settings it saved, so messages show once and detach() still restores
        the originals.
        """
        if self._saved is not None:
            return
        logger = logging.getLogger(PACKAGE_LOGGER)
        self._saved = (logger.level, logger.propagate)
        for stale in [h for h in logger.handlers if isinstance(h, OperatorLog)]:
            logger.removeHandler(stale)
            self._saved, stale._saved = stale._saved or self._saved, None
        logger.addHandler(self)
        logger.setLevel(logging.INFO)
        # A root handler printing to the terminal would garble the session's status line.
        logger.propagate = False

    def detach(self) -> None:
        """Undo attach() (idempotent)."""
        if self._saved is None:
            return
        logger = logging.getLogger(PACKAGE_LOGGER)
        logger.removeHandler(self)
        logger.setLevel(self._saved[0])
        logger.propagate = self._saved[1]
        self._saved = None


def poll_end() -> bool:  # pragma: no cover - needs a real TTY
    """True once the operator pressed Enter to end the episode. Never blocks."""
    import select

    if not sys.stdin.isatty():
        return False
    if not select.select([sys.stdin], [], [], 0)[0]:
        return False
    sys.stdin.readline()
    return True

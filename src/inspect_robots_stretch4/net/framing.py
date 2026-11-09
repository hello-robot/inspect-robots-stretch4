"""Messages on the wire: `[header JSON, *binary frames]`, never pickle.

Every header is a JSON object carrying the protocol version `v` and a request `id`; a reply
echoes the id. Images, depth and anything else binary travel as their own frames.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

PROTOCOL = 2  # host and served robot must match

# A failed request replies {"ok": false, "kind": one of these, "msg": ...}.
ERROR_KINDS = ("fault", "safety", "protocol", "busy", "no_session")


class ProtocolError(Exception):
    """A message that isn't valid for this protocol."""


class RemoteError(Exception):
    """The other side answered with an error reply."""

    def __init__(self, kind: str, msg: str) -> None:
        super().__init__(msg)
        self.kind = kind
        self.msg = msg


def pack(header: dict[str, Any], frames: Sequence[bytes] = ()) -> list[bytes]:
    """A header (the version is added) and binary frames -> a multipart message."""
    return [json.dumps({**header, "v": PROTOCOL}).encode(), *(bytes(f) for f in frames)]


def unpack(message: Sequence[bytes]) -> tuple[dict[str, Any], list[bytes]]:
    """A multipart message -> (header, frames), checking the header and the version."""
    if not message:
        raise ProtocolError("empty message")
    try:
        header = json.loads(bytes(message[0]))
    except (UnicodeDecodeError, ValueError) as e:
        raise ProtocolError(f"the header is not JSON: {e}") from None
    if not isinstance(header, dict):
        raise ProtocolError(f"the header is not a JSON object: {type(header).__name__}")
    if header.get("v") != PROTOCOL:
        raise ProtocolError(
            f"protocol version mismatch: this side speaks {PROTOCOL}, the other side sent "
            f"{header.get('v')!r}; install the same inspect-robots-stretch4 on both machines"
        )
    if not isinstance(header.get("id"), int):
        raise ProtocolError("the header has no integer `id`")
    return header, [bytes(f) for f in message[1:]]


def expect_frames(header: dict[str, Any], frames: Sequence[bytes], count: int) -> None:
    """Raise unless a message carries exactly `count` binary frames."""
    if len(frames) != count:
        raise ProtocolError(
            f"{header.get('op', 'reply')!r} carries {len(frames)} binary frames, expected {count}"
        )


def error_reply(kind: str, msg: str) -> dict[str, Any]:
    """The header of a failed reply."""
    if kind not in ERROR_KINDS:
        raise ValueError(f"unknown error kind {kind!r}")
    return {"ok": False, "kind": kind, "msg": msg}

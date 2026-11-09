"""Messages on the wire: JSON headers, binary frames, the version check."""

from __future__ import annotations

import json

import pytest

from inspect_robots_stretch4.net.framing import (
    PROTOCOL,
    ProtocolError,
    error_reply,
    expect_frames,
    pack,
    unpack,
)


def test_round_trip_adds_the_version_and_keeps_frames() -> None:
    message = pack({"op": "read", "id": 3}, [b"\x00\xff", b""])
    header, frames = unpack(message)
    assert header == {"op": "read", "id": 3, "v": PROTOCOL}
    assert frames == [b"\x00\xff", b""]


@pytest.mark.parametrize(
    ("message", "match"),
    [
        ([], "empty"),
        ([b"\xff\xfe"], "not JSON"),
        ([b"[1, 2]"], "not a JSON object"),
        ([json.dumps({"v": PROTOCOL}).encode()], "id"),
        ([json.dumps({"v": PROTOCOL, "id": "7"}).encode()], "id"),
    ],
)
def test_malformed_messages_are_rejected(message, match) -> None:
    with pytest.raises(ProtocolError, match=match):
        unpack(message)


def test_a_version_mismatch_names_both_versions() -> None:
    with pytest.raises(ProtocolError) as e:
        unpack([json.dumps({"v": 99, "id": 1}).encode()])
    assert f"speaks {PROTOCOL}" in str(e.value) and "sent 99" in str(e.value)


def test_expect_frames() -> None:
    expect_frames({"op": "read"}, [b"a"], 1)
    with pytest.raises(ProtocolError, match="'read' carries 2 binary frames, expected 1"):
        expect_frames({"op": "read"}, [b"a", b"b"], 1)


def test_error_reply() -> None:
    assert error_reply("busy", "x") == {"ok": False, "kind": "busy", "msg": "x"}
    with pytest.raises(ValueError):
        error_reply("oops", "x")

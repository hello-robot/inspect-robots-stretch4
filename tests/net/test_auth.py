"""CURVE key files: init, show, trust, and refusing to run without keys."""

from __future__ import annotations

import stat

import pytest
import zmq

from inspect_robots_stretch4.net import auth


def test_init_is_idempotent_and_private(tmp_path) -> None:
    keys = tmp_path / "keys"
    auth.init_keys(keys)
    first = auth.public_key(keys)
    auth.init_keys(keys)
    assert auth.public_key(keys) == first and len(first) == 40
    assert stat.S_IMODE((keys / "self.key_secret").stat().st_mode) == 0o600
    assert stat.S_IMODE(keys.stat().st_mode) == 0o700


def test_trust_and_load_a_peer(tmp_path) -> None:
    host, robot = tmp_path / "host", tmp_path / "robot"
    auth.init_keys(host)
    auth.init_keys(robot)
    auth.trust(host, "robot2", auth.public_key(robot))
    curve = auth.client_curve(host, "robot2")
    assert curve.server == auth.server_curve(robot).public
    assert curve.public == auth.public_key(host).encode()


@pytest.mark.parametrize(
    ("name", "key"),
    [("robot2", "short"), ("../evil", "x" * 40), ("self", "x" * 40), ("robot2", "\x00" * 40)],
)
def test_trust_rejects_bad_names_and_keys(tmp_path, name, key) -> None:
    with pytest.raises(ValueError):
        auth.trust(tmp_path, name, key)


def test_missing_keys_are_an_error_not_an_open_socket(tmp_path) -> None:
    with pytest.raises(auth.MissingKeys, match="stretch_inspect_pair_keys init"):
        auth.server_curve(tmp_path)
    auth.init_keys(tmp_path)
    with pytest.raises(auth.MissingKeys, match="stretch_inspect_pair_keys trust robot2"):
        auth.client_curve(tmp_path, "robot2")
    with pytest.raises(auth.MissingKeys, match="no trusted client keys"):
        auth.start_authenticator(zmq.Context.instance(), tmp_path)

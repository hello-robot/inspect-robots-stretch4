"""`stretch_inspect_pair_server` refuses unsafe binds and missing keys; `stretch_inspect_pair_keys`
round trip."""

from __future__ import annotations

import pytest

from inspect_robots_stretch4.remote.cli import check_bind, keys_main, serve_main


def test_check_bind_wants_one_explicit_address() -> None:
    assert check_bind("tcp://192.168.1.42:5600") == "tcp://192.168.1.42:5600"
    for bad in ("tcp://*:5600", "tcp://0.0.0.0:5600", "192.168.1.42:5600", "tcp://10.0.0.2"):
        with pytest.raises(ValueError):
            check_bind(bad)


def test_serve_refuses_every_interface_and_missing_keys(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as e:
        serve_main(["--bind", "tcp://*:5600", "--keys", str(tmp_path)])
    assert e.value.code == 2 and "every interface" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        serve_main(["--bind", "tcp://127.0.0.1:5600", "--keys", str(tmp_path)])
    assert e.value.code == 2 and "stretch_inspect_pair_keys init" in capsys.readouterr().err


def test_keys_init_show_trust(tmp_path, capsys) -> None:
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert keys_main(["--keys", a, "init"]) == 0
    assert keys_main(["--keys", b, "init"]) == 0
    capsys.readouterr()
    keys_main(["--keys", b, "show"])
    key = capsys.readouterr().out.strip()
    assert keys_main(["--keys", a, "trust", "robot2", key]) == 0
    assert (tmp_path / "a" / "authorized" / "robot2.key").exists()
    with pytest.raises(SystemExit) as e:
        keys_main(["--keys", a, "trust", "robot2", "not-a-key"])
    assert e.value.code == 2


def test_keys_trust_takes_a_key_starting_with_a_dash(tmp_path) -> None:
    zmq = pytest.importorskip("zmq")
    key = ""
    while not key.startswith("-"):  # about 1 key in 85
        key = zmq.curve_keypair()[0].decode()
    a = str(tmp_path / "a")
    assert keys_main(["--keys", a, "init"]) == 0
    assert keys_main(["--keys", a, "trust", "robot2", key]) == 0
    assert (tmp_path / "a" / "authorized" / "robot2.key").exists()

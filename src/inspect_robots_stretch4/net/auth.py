"""CURVE keys: every TCP connection is encrypted, and only trusted machines get in.

A keys directory holds this machine's keypair and the public keys it trusts:

    <keys>/self.key            this machine's public key
    <keys>/self.key_secret     ... and its secret key (never leaves the machine)
    <keys>/authorized/<name>.key   a trusted peer's public key

A service accepts only clients whose key is in `authorized/`; a client connects only to a
server whose key it trusts, by name. There is no fallback to an open socket: missing keys are an
error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator
from zmq.utils import z85

DEFAULT_KEYS_DIR = "~/.config/inspect_robots_stretch4/keys"
SELF = "self"
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class MissingKeys(FileNotFoundError):
    """A key file that CURVE needs is not there."""


@dataclass(frozen=True)
class ServerCurve:
    """A server socket's keypair."""

    public: bytes
    secret: bytes

    def apply(self, socket: zmq.Socket[Any]) -> None:
        socket.curve_server = True
        socket.curve_publickey = self.public
        socket.curve_secretkey = self.secret


@dataclass(frozen=True)
class ClientCurve:
    """A client socket's keypair and the server key it insists on."""

    public: bytes
    secret: bytes
    server: bytes

    def apply(self, socket: zmq.Socket[Any]) -> None:
        socket.curve_publickey = self.public
        socket.curve_secretkey = self.secret
        socket.curve_serverkey = self.server


def keys_dir(path: str | Path = DEFAULT_KEYS_DIR) -> Path:
    return Path(path).expanduser()


def init_keys(path: str | Path = DEFAULT_KEYS_DIR) -> Path:
    """Create this machine's keypair (once) and the `authorized/` directory; the public key file."""
    root = keys_dir(path)
    (root / "authorized").mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    public = root / f"{SELF}.key"
    if not (root / f"{SELF}.key_secret").exists():
        old = os.umask(0o077)  # the secret is private from the moment it is written
        try:
            zmq.auth.create_certificates(str(root), SELF)
        finally:
            os.umask(old)
    return public


def public_key(path: str | Path = DEFAULT_KEYS_DIR) -> str:
    """This machine's public key, as the one line a peer pastes into `trust`."""
    return _keypair(keys_dir(path))[0].decode()


def trust(path: str | Path, name: str, key: str) -> Path:
    """Add a peer's public key to `authorized/` under `name`."""
    if not _NAME.match(name) or name == SELF:
        raise ValueError(f"a peer name is letters, digits, '.', '_' or '-', not {name!r}")
    _check_key(key)
    root = keys_dir(path)
    (root / "authorized").mkdir(parents=True, exist_ok=True)
    out = root / "authorized" / f"{name}.key"
    out.write_text(
        "#   ****  stretch_inspect_pair_keys: a trusted peer's CURVE public key  ****\n"
        "metadata\n"
        "curve\n"
        f'    public-key = "{key}"\n'
    )
    return out


def server_curve(path: str | Path = DEFAULT_KEYS_DIR) -> ServerCurve:
    public, secret = _keypair(keys_dir(path))
    return ServerCurve(public, secret)


def client_curve(path: str | Path, server_name: str) -> ClientCurve:
    """This machine's keypair, and the server key trusted as `server_name`."""
    root = keys_dir(path)
    public, secret = _keypair(root)
    peer = root / "authorized" / f"{server_name}.key"
    if not peer.exists():
        raise MissingKeys(
            f"no trusted key named {server_name!r} in {peer.parent}; on that machine run "
            f"`stretch_inspect_pair_keys show`, then here "
            f"`stretch_inspect_pair_keys trust {server_name} <key>`"
        )
    server, _ = zmq.auth.load_certificate(str(peer))
    return ClientCurve(public, secret, server)


def start_authenticator(ctx: zmq.Context[Any], path: str | Path) -> ThreadAuthenticator:
    """Accept CURVE clients whose public key is in `authorized/`, and no others. Start it before
    binding; stop it after closing the sockets."""
    authorized = keys_dir(path) / "authorized"
    if not authorized.is_dir() or not any(authorized.glob("*.key")):
        raise MissingKeys(
            f"no trusted client keys in {authorized}; on the client run "
            f"`stretch_inspect_pair_keys show`, then here "
            f"`stretch_inspect_pair_keys trust <name> <key>`"
        )
    auth = ThreadAuthenticator(ctx)
    auth.start()
    auth.configure_curve(domain="*", location=str(authorized))
    return auth


def _keypair(root: Path) -> tuple[bytes, bytes]:
    secret_file = root / f"{SELF}.key_secret"
    if not secret_file.exists():
        raise MissingKeys(f"no keypair in {root}; run `stretch_inspect_pair_keys init`")
    public, secret = zmq.auth.load_certificate(str(secret_file))
    if secret is None:
        raise MissingKeys(
            f"{secret_file} has no secret key; run `stretch_inspect_pair_keys init` again"
        )
    return public, secret


def _check_key(key: str) -> None:
    try:
        ok = len(key) == 40 and len(z85.decode(key)) == 32
    except Exception:
        ok = False
    if not ok:
        raise ValueError(f"not a CURVE public key (40 Z85 characters): {key!r}")

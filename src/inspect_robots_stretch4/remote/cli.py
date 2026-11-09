"""`stretch_inspect_pair_server` (serve this robot to a host) and `stretch_inspect_pair_keys`
(CURVE keys)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from collections.abc import Sequence

from inspect_robots_stretch4.net import auth, rpc
from inspect_robots_stretch4.remote.service import DriverService

log = logging.getLogger(__name__)

ANY_HOST = ("*", "0.0.0.0", "", "[::]", "::")


def check_bind(endpoint: str) -> str:
    """A TCP endpoint on one explicit address: never every interface."""
    host, sep, port = endpoint.removeprefix("tcp://").rpartition(":")
    if not endpoint.startswith("tcp://") or not sep or not port.isdigit():
        raise ValueError(f"--bind takes tcp://<this robot's IP>:<port>, got {endpoint!r}")
    if host in ANY_HOST:
        raise ValueError(
            f"--bind {endpoint!r} listens on every interface; give this robot's own IP address"
        )
    return endpoint


def serve_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stretch_inspect_pair_server",
        description="Serve this Stretch 4 to one inspect-robots host over CURVE-encrypted zmq.",
    )
    parser.add_argument("--bind", required=True, help="tcp://<this robot's IP>:<port>")
    parser.add_argument("--keys", default=auth.DEFAULT_KEYS_DIR, help="CURVE keys directory")
    parser.add_argument("--log-file", default=None, help="also log to this file")
    args = parser.parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers
    )
    try:
        endpoint = check_bind(args.bind)
        curve = auth.server_curve(args.keys)
        ctx = rpc.context()
        authenticator = auth.start_authenticator(ctx, args.keys)
    except (ValueError, auth.MissingKeys) as e:
        parser.exit(2, f"stretch_inspect_pair_server: {e}\n")

    from inspect_robots_stretch4.driver import BodyDriver

    try:
        service = DriverService(endpoint, BodyDriver, ctx=ctx, curve=curve)
    except Exception:
        authenticator.stop()
        raise
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: service.stop())
    log.info(
        "serving at %s (heartbeats at %s); Ctrl-C to stop",
        endpoint,
        rpc.heartbeat_endpoint(endpoint),
    )
    try:
        service.run()
    finally:
        service.close()
        authenticator.stop()
        log.info("stopped")
    return 0


def keys_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stretch_inspect_pair_keys",
        description="Create this machine's CURVE keys and trust a peer's. The service trusts "
        "the host's key and the host trusts the service's, each under a name.",
    )
    parser.add_argument("--keys", default=auth.DEFAULT_KEYS_DIR, help="CURVE keys directory")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create this machine's keypair (once)")
    sub.add_parser("show", help="print this machine's public key")
    trust = sub.add_parser("trust", help="trust a peer's public key under a name")
    trust.add_argument("name", help="e.g. robot1 (on the service) or robot2 (on the host)")
    trust.add_argument("key", help="the peer's `stretch_inspect_pair_keys show` output")
    args = parser.parse_args(_keys_positional(sys.argv[1:] if argv is None else argv))
    try:
        if args.cmd == "init":
            path = auth.init_keys(args.keys)
            print(f"keypair in {path.parent}; public key:\n{auth.public_key(args.keys)}")
        elif args.cmd == "show":
            print(auth.public_key(args.keys))
        else:
            print(f"trusted {args.name}: {auth.trust(args.keys, args.name, args.key)}")
    except (ValueError, auth.MissingKeys) as e:
        parser.exit(2, f"stretch_inspect_pair_keys: {e}\n")
    return 0


def _keys_positional(argv: Sequence[str]) -> list[str]:
    """`trust <name> <key>` with `--` before the name: a Z85 key can start with "-", which
    argparse would otherwise read as an option."""
    argv = list(argv)
    if "trust" in argv and "--" not in argv:
        i = argv.index("trust")
        if len(argv) - i == 3:
            argv.insert(i + 1, "--")
    return argv

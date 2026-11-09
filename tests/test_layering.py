"""The package's import rules, checked by parsing every module's imports.

- `net`: transport only. Imports nothing else from this package; the only importer of zmq.
- `remote`: `net`, `config`, `driver`.
- `pair`: `side`, `config`, `driver`, `hold`, `operator`; `remote` only inside a function, so
  `pair` imports (for doctor and conformance) without pyzmq.
- `tasks`: nothing else from this package.
- everything else (the single robot): never `net`, `remote` or `pair`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import inspect_robots_stretch4

PACKAGE = "inspect_robots_stretch4"
ROOT = Path(inspect_robots_stretch4.__file__).parent
LAYERS = ("net", "remote", "pair", "tasks")

# layer -> what it may import at module level; "core:x" is the single-robot module x.
ALLOWED = {
    "core": {"core"},
    "net": {"net"},
    "remote": {"remote", "net", "core:config", "core:driver"},
    "pair": {"pair", "core:side", "core:config", "core:driver", "core:hold", "core:operator"},
    "tasks": {"tasks"},
}
# Allowed only inside a function body (imported when called, not when the module loads).
ALLOWED_LAZY = {"pair": {"remote"}}
ZMQ_LAYERS = {"net"}


def layer_of(module: str) -> str:
    """'inspect_robots_stretch4.pair.guard' -> 'pair'; '….driver' -> 'core:driver'."""
    parts = module.split(".")[1:]
    if not parts:
        return "core:__init__"
    return parts[0] if parts[0] in LAYERS else f"core:{parts[0]}"


def _imports(tree: ast.AST) -> list[tuple[str, bool]]:
    """(imported module, inside a function) for every import in a module."""
    found: list[tuple[str, bool]] = []

    def visit(node: ast.AST, lazy: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            if isinstance(child, ast.Import):
                found.extend((alias.name, inner) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and not child.level:
                found.append((child.module, inner))
                # `from inspect_robots_stretch4 import pair` imports the subpackage.
                if child.module == PACKAGE:
                    found.extend((f"{PACKAGE}.{a.name}", inner) for a in child.names)
            visit(child, inner)

    visit(tree, False)
    return found


def violations(layer: str, source: str) -> list[str]:
    """What `source`, a module in `layer`, imports that the rules forbid."""
    allowed = ALLOWED["core" if layer.startswith("core") else layer]
    lazy_ok = ALLOWED_LAZY.get(layer, set())
    bad = []
    for module, lazy in _imports(ast.parse(source)):
        top = module.split(".")[0]
        if top == "zmq" and layer not in ZMQ_LAYERS:
            bad.append(module)
        if top != PACKAGE:
            continue
        target = layer_of(module)
        if target == "core:__init__":
            target = "core"  # the package root re-exports single-robot names
        ok = target in allowed or (target.startswith("core") and "core" in allowed)
        if not ok and not (lazy and target in lazy_ok):
            bad.append(module)
    return bad


def _modules() -> list[tuple[str, Path]]:
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        found.append((layer_of(".".join([PACKAGE, *parts])), path))
    return found


@pytest.mark.parametrize(("layer", "path"), _modules(), ids=lambda v: str(v))
def test_module_follows_the_import_rules(layer: str, path: Path) -> None:
    assert violations(layer, path.read_text()) == []


def test_the_checker_catches_violations() -> None:
    assert violations("core:embodiment", "from inspect_robots_stretch4.pair import x") == [
        "inspect_robots_stretch4.pair"
    ]
    assert violations("net", "from inspect_robots_stretch4.config import JOINTS") != []
    assert violations("remote", "import zmq") == ["zmq"]
    assert violations("pair", "from inspect_robots_stretch4.net import rpc") != []
    assert violations("pair", "from inspect_robots_stretch4.remote import client") != []
    lazy = "def f():\n    from inspect_robots_stretch4.remote.client import RemoteDriver\n"
    assert violations("pair", lazy) == []
    assert violations("tasks", "from inspect_robots_stretch4 import driver") != []
    assert violations("net", "import zmq") == []

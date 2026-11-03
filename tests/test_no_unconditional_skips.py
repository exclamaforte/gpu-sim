"""Guard: every skip site in tests/ must be registered in skip_allowlist.json.

A new skip (decorator or in-body call) that is not in the allowlist fails
this test, so skips cannot be added silently. Conditional skips are allowed
only when documented in the allowlist with their justification (condition +
derivation).
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
ALLOWLIST = TESTS / "skip_allowlist.json"


def _is_skip_call(node):
    if not isinstance(node, ast.Call):
        return False
    try:
        fn = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
    except Exception:
        fn = ""
    return "skip" in fn.lower()


def _skip_sites(path: Path):
    """Yield 'file::function:lineno' for every skip call site (decorator or
    in-body). Each call site is a separate entry so a second skip inside an
    already-allowlisted function is still caught."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # decorator skips
        for d in node.decorator_list:
            if _is_skip_call(d):
                yield f"{path.name}::{node.name}:{d.lineno}"
        # decorator line numbers to exclude from in-body walk
        deco_lines = {d.lineno for d in node.decorator_list if _is_skip_call(d)}
        # in-body skips
        for sub in ast.walk(node):
            if sub is node:
                continue
            if isinstance(sub, ast.Call) and _is_skip_call(sub) and sub.lineno not in deco_lines:
                yield f"{path.name}::{node.name}:{sub.lineno}"


def test_all_skip_sites_are_allowlisted():
    allow = json.loads(ALLOWLIST.read_text()) if ALLOWLIST.exists() else {}
    offenders = []
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name == "test_no_unconditional_skips.py":
            continue
        for site in _skip_sites(p):
            if site not in allow:
                offenders.append(site)
    assert len(offenders) == 0, (
        "unregistered skip call sites (add to tests/skip_allowlist.json with "
        "condition + derivation):\n" + "\n".join(offenders)
    )

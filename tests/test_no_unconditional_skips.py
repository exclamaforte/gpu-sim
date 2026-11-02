"""Guard: every skip site in tests/ must be registered in skip_allowlist.json.

A new skip (decorator or in-body call) that is not in the allowlist fails
this test, so skips cannot be added silently. Conditional skips are allowed
only when documented in the allowlist with their justification.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
ALLOWLIST = TESTS / "skip_allowlist.json"


def _skip_sites(path: Path):
    """Yield (function_qualname) for every function containing a skip."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        has_skip = False
        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                fn = ast.unparse(d.func) if hasattr(ast, "unparse") else ""
                if "skip" in fn.lower():
                    has_skip = True
            elif isinstance(d, ast.Name) and "skip" in d.id.lower():
                has_skip = True
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = ast.unparse(sub.func) if hasattr(ast, "unparse") else ""
                if "skip" in fn.lower():
                    has_skip = True
        if has_skip:
            yield f"{path.name}::{node.name}"


def test_all_skip_sites_are_allowlisted():
    allow = json.loads(ALLOWLIST.read_text()) if ALLOWLIST.exists() else {}
    offenders = []
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name == "test_no_unconditional_skips.py":
            continue
        for site in _skip_sites(p):
            if site not in allow:
                offenders.append(site)
    assert not offenders, (
        "unregistered skip sites (add to tests/skip_allowlist.json with justification):\n"
        + "\n".join(offenders)
    )

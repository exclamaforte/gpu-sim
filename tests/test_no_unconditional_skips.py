"""Guard: every skip site in tests/ must be registered in skip_allowlist.json.

A new skip (decorator or in-body call) that is not in the allowlist fails
this test, so skips cannot be added silently. Unconditional skips (no
condition argument) are forbidden outright. Conditional skips are allowed
only when documented in the allowlist with their justification.
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


def _ancestors_conditional(tree, target):
    """Return True if target node is inside an if/try/except/with block."""
    # Walk parents
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    cur = id(target)
    # target itself may not be in parents; walk via node identity
    # Build parent map properly
    parent_map = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
            for sub in ast.iter_child_nodes(child):
                pass
    # traverse
    node = target
    # We need parent of target; use a recursive search
    def find_parent(root, target_id):
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                if id(child) == target_id:
                    return parent
                # deeper
        return None
    cur_id = id(target)
    depth = 0
    while depth < 100:
        p = find_parent(tree, cur_id)
        if p is None:
            return False
        if isinstance(p, (ast.If, ast.Try, ast.ExceptHandler, ast.With, ast.For, ast.While)):
            return True
        cur_id = id(p)
        depth += 1
    return False


def _is_unconditional_skip(node, tree=None):
    """Return True if this skip call has no condition and is not in a conditional block."""
    if not isinstance(node, ast.Call):
        return False
    if tree is not None and _ancestors_conditional(tree, node):
        return False
    try:
        fn = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
    except Exception:
        fn = ""
    fnl = fn.lower()
    if "skipif" in fnl:
        return False
    if node.args:
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            return False
        return True
    kw_names = {kw.arg for kw in node.keywords if kw.arg}
    if kw_names & {"condition", "skipif"}:
        return False
    return True


def _skip_sites(path: Path):
    """Yield 'file::function:lineno:TAG' for every skip call site."""
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        deco_lines = {d.lineno for d in node.decorator_list
                      if isinstance(d, ast.Call) and _is_skip_call(d)}
        for d in node.decorator_list:
            if _is_skip_call(d):
                tag = "UNCOND" if _is_unconditional_skip(d, tree) else "COND"
                yield f"{path.name}::{node.name}:{d.lineno}:{tag}"
        for sub in ast.walk(node):
            if sub is node:
                continue
            if isinstance(sub, ast.Call) and _is_skip_call(sub) and sub.lineno not in deco_lines:
                tag = "UNCOND" if _is_unconditional_skip(sub, tree) else "COND"
                yield f"{path.name}::{node.name}:{sub.lineno}:{tag}"


def test_all_skip_sites_are_allowlisted():
    allow = json.loads(ALLOWLIST.read_text()) if ALLOWLIST.exists() else {}
    uncond_offenders = []
    unreg_offenders = []
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name == "test_no_unconditional_skips.py":
            continue
        for site in _skip_sites(p):
            tag = site.rsplit(":", 1)[-1]
            base = site.rsplit(":", 1)[0]
            if tag == "UNCOND":
                uncond_offenders.append(base)
            if base not in allow and site not in allow:
                unreg_offenders.append(site)
    assert len(uncond_offenders) == 0, (
        "unconditional skips are forbidden (convert to skipIf or remove):\n"
        + "\n".join(uncond_offenders)
    )
    assert len(unreg_offenders) == 0, (
        "unregistered skip call sites (add to tests/skip_allowlist.json with "
        "condition + derivation):\n" + "\n".join(unreg_offenders)
    )

"""Guard: every skip site in tests/ must be registered in skip_allowlist.json.

A new skip (decorator or in-body call) that is not in the allowlist fails
this test, so skips cannot be added silently. Unconditional skips (no
condition argument) are forbidden outright. Conditional skips are allowed
only when documented in the allowlist with their justification.

Keys are (file, function, TAG, condition-text); the line number is carried
for diagnostics only and is NOT part of the key, so edits that shift lines do
not invalidate registrations. A new conditional skip inside an already
registered function is caught because its condition text differs.
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
    """Return True if target node is inside an if/try/except/with/for/while block."""
    def find_parent(root, target_id):
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                if id(child) == target_id:
                    return parent
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
    """Yield (key, lineno) for every skip call site.

    key is line-free: 'file::function:TAG:condition-text'. lineno is carried
    for the failure message only.
    """
    src = path.read_text()
    tree = ast.parse(src)
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        deco_lines = {d.lineno for d in node.decorator_list
                      if isinstance(d, ast.Call) and _is_skip_call(d)}

        def _emit(call_node, lineno):
            tag = "UNCOND" if _is_unconditional_skip(call_node, tree) else "COND"
            ct = _cond_text(call_node)
            key = f"{path.name}::{node.name}:{tag}:{ct}"
            results.append((key, lineno))

        for d in node.decorator_list:
            if _is_skip_call(d):
                _emit(d, d.lineno)
        for sub in ast.walk(node):
            if sub is node:
                continue
            if isinstance(sub, ast.Call) and _is_skip_call(sub) and sub.lineno not in deco_lines:
                _emit(sub, sub.lineno)
    return results


def _cond_text(node):
    if node.args:
        try:
            return ast.unparse(node.args[0])
        except Exception:
            return ""
    for kw in node.keywords:
        if kw.arg in ("condition", "skipif"):
            try:
                return ast.unparse(kw.value)
            except Exception:
                return ""
    return ""


class TestNoUnconditionalSkips:
    def test_all_skip_sites_are_allowlisted(self):
        allow = json.loads(ALLOWLIST.read_text()) if ALLOWLIST.exists() else {}
        uncond_offenders = []
        unreg_offenders = []
        for p in sorted(TESTS.glob("test_*.py")):
            if p.name == "test_no_unconditional_skips.py":
                continue
            for key, lineno in _skip_sites(p):
                _parts = key.split(":", 3)
                tag = _parts[3].split(":", 1)[0] if len(_parts) >= 4 else ""
                if tag == "UNCOND":
                    uncond_offenders.append(f"{p.name}:{lineno}:UNCOND")
                if key not in allow:
                    unreg_offenders.append(f"{key} (line {lineno})")
        assert len(uncond_offenders) == 0, (
            "unconditional skips are forbidden (convert to skipIf or remove):\n"
            + "\n".join(uncond_offenders)
        )
        assert len(unreg_offenders) == 0, (
            "unregistered skip call sites (add to tests/skip_allowlist.json with "
            "condition + derivation):\n" + "\n".join(unreg_offenders)
        )

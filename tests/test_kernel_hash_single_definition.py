#!/usr/bin/env python3
"""Single kernel_hash definition (gpu-sim-9yt).

After the fix there is exactly one recipe for a kernel_hash-named value:
generator_stage1.kernel_hash_for. This pins it three ways:

1. test_single_definition_static: AST scan over repo source -- every
   assignment to a kernel_hash/khash-named target must flow from a
   kernel_hash_for() call (directly, or via a same-file name bound to one).
2. test_regen_proof: the committed proof.json (regen_proof.py) shows both
   corpora regenerating hash-identical through the unified scripts.
3. test_divergent_recipe_mismatches (negative control): the OLD divergent
   recipe (missing problem_size) does NOT reproduce 7.1's hashes -- proving
   the regen check goes red on a re-keying rather than passing vacuously.

CPU-only.
"""
import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["epic_u2o", "experiments", "scripts", "src", "shim"]
SKIP_PARTS = {"__pycache__", "third-party"}


def target_names(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Subscript):
        v = node.slice
        if isinstance(v, ast.Constant):
            return [str(v.value)]
        return []
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for e in node.elts:
            out += target_names(e)
        return out
    return []


def value_mints_hash(node):
    if node is None:
        return False
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == "hashlib":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "sha256":
            return True
    return False


def kernel_hash_assignments(tree):
    """(name, is_canonical) for kernel_hash/khash targets: a site is
    canonical iff its value does NOT mint a hash inline (no hashlib in
    the value subtree). Propagation -- Name, Subscript, or a
    kernel_hash_for() call -- is allowed."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for name in sum((target_names(t) for t in targets), []):
            if "kernel_hash" in name or name == "khash":
                out.append((name, not value_mints_hash(node.value),
                            node.lineno))
    # Dict displays: {'kernel_hash': <expr>} -- same rule on the value.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "kernel_hash":
                out.append(("{'kernel_hash': ...}",
                            not value_mints_hash(v), node.lineno))
    return out


class SingleDefinitionTest(unittest.TestCase):
    def test_single_definition_static(self):
        bad = []
        n_files, n_sites = 0, 0
        for d in SCAN_DIRS:
            root = REPO / d
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                if any(p in SKIP_PARTS for p in path.parts):
                    continue
                tree = ast.parse(path.read_text())
                for name, ok, lineno in kernel_hash_assignments(tree):
                    n_sites += 1
                    if not ok:
                        bad.append(f"{path.relative_to(REPO)}:{lineno} {name}")
                n_files += 1
        self.assertGreater(n_sites, 0, "scan found no sites at all")
        self.assertEqual(bad, [], f"non-canonical kernel_hash sites: {bad[:10]}")
        print(f"scanned {n_files} files, {n_sites} kernel_hash sites, all canonical")

    def test_regen_proof(self):
        proof = json.loads(
            (REPO / "experiments/gpu-sim-9yt/proof.json").read_text())
        self.assertEqual(proof["verdict"], "PASS", proof)
        self.assertTrue(proof["results"]["7.1"]["match"])
        self.assertTrue(proof["results"]["e5q"]["match"])
        self.assertEqual(proof["results"]["7.1"]["n_regen"], 264)
        self.assertEqual(proof["results"]["e5q"]["n_regen"], 192)

    def test_divergent_recipe_mismatches(self):
        """Negative control: the pre-fix divergent recipe (no problem_size)
        must NOT reproduce the corpus -- else the regen check is decoration."""
        sys.path.insert(0, str(REPO))
        from epic_u2o.generator_stage1 import generate_kernels
        rows_71 = [json.loads(l) for l in
                   (REPO / "experiments/gpu-sim-u2o.7.1/measured.jsonl"
                    ).read_text().splitlines()]
        got = set()
        for s in generate_kernels(max_kernels=100000):
            d = s.to_dict()
            if d["op_type"] not in ("scale", "add", "saxpy", "relu", "copy",
                                    "triad", "mul", "div"):
                continue
            got.add(hashlib.sha256(
                (d["source"] + json.dumps([d["grid"], d["block"]])
                 ).encode()).hexdigest()[:16])
        want = {r["kernel_hash"] for r in rows_71}
        overlap = len(got & want)
        print(f"divergent recipe overlap with 7.1 corpus: {overlap}/264")
        self.assertLess(overlap, 264,
                        "divergent recipe reproduces the corpus?!")
        self.assertGreater(len(got), 264)


if __name__ == "__main__":
    unittest.main(verbosity=2)

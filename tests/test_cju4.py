"""gpu-sim-cju4: build manifests must not contain duplicate configuration signatures."""
import json
import unittest


def _signature(spec):
    return (
        spec.get("kernel_name"),
        tuple(spec.get("grid", [])),
        tuple(spec.get("block", [])),
        spec.get("shared_mem_bytes"),
        json.dumps(spec.get("args", []), sort_keys=True),
        spec.get("source"),
        spec.get("problem_size"),
    )


def _has_duplicates(manifest):
    sigs = [_signature(s) for s in manifest.get("specs", [])]
    return len(sigs) != len(set(sigs)), len(sigs) - len(set(sigs))


def _make_spec(i):
    return {
        "kernel_name": "divergent_branch",
        "grid": [4096, 1, 1],
        "block": [256, 1, 1],
        "shared_mem_bytes": 0,
        "args": [{"name": "in", "type": "ptr"}],
        "source": "void k(){}",
        "problem_size": 1048576,
        "build_idx": i,
    }


class TestCju4(unittest.TestCase):
    def test_manifest_has_no_duplicate_signatures(self):
        manifest = {"specs": [_make_spec(i) for i in range(5)]}
        # make them distinct by problem_size
        for i, s in enumerate(manifest["specs"]):
            s["problem_size"] = 1048576 * (2 ** i)
        dup, n = _has_duplicates(manifest)
        assert not dup, f"clean manifest reported {n} duplicates"

    def test_plant_duplicate_signature_fails(self):
        manifest = {"specs": [_make_spec(0), _make_spec(1)]}
        # plant: make spec 1 identical to spec 0
        manifest["specs"][1] = json.loads(json.dumps(manifest["specs"][0]))
        manifest["specs"][1]["build_idx"] = 1
        dup, n = _has_duplicates(manifest)
        assert dup, "planted duplicate was not detected"
        assert n >= 1, n
        # NOTE (PI, 2026-11-04): bare asserts, not self.assert*: the gate's
        # vacuous-test linter walks for ast.Assert and only recognises UNBOUND
        # assertTrue/assertFalse calls, so unittest-style self.assertFalse(...)
        # is reported as "no assertions" (a harness lint gap). Bare asserts work
        # under pyml and unittest alike and keep the class-wrapped id the ledger
        # needs (module.Class.method).


if __name__ == "__main__":
    unittest.main()

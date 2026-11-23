MUTATIONS = {"trcz_impostor": "trcz impostor probe hardening (gpu-sim-trcz)"}

def apply(root):
    """Mutation for trcz_impostor: drop the two vacuous-guard assertions.

    gpu-sim-vuoq: this used to DELETE whole lines, and one of them opened a
    nested block, so the module stopped parsing and the plant "bit" as an
    IndentationError rather than as a test outcome.  Replace only the STATEMENT
    TEXT and leave the surrounding indentation alone, which keeps the block
    structure (and the mutation's meaning) intact -- one of the two anchors is
    itself text inside the probe body the test writes out.
    """
    import os
    path = os.path.join(root, "tests", "test_zsd3_janitor_detection.py")
    backup = open(path).read()
    mutated = backup
    for needle in (
        "assert pytestconfig._dirty_at_start is not None, 'conftest did not run'",
        'assert marker.exists(), "probe did not execute (impostor detection):\\n" + out[-1000:]',
    ):
        if needle not in mutated:
            raise RuntimeError(f"trcz mutation anchor not found: {needle[:45]}")
        mutated = mutated.replace(needle, "pass  # MUTATION (trcz_impostor): assertion dropped", 1)
    if mutated == backup:
        raise RuntimeError("trcz mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATIONS = {"trcz_impostor": "trcz impostor probe hardening (gpu-sim-trcz)"}

def apply(root):
    """Mutation for trcz_impostor: remove the _dirty_at_start/marker assertion."""
    import os
    path = os.path.join(root, "tests", "test_zsd3_janitor_detection.py")
    backup = open(path).read()
    mutated = backup.replace(
        "    assert pytestconfig._dirty_at_start is not None, 'conftest did not run'\n",
        '', 1)
    mutated = mutated.replace(
        '    assert marker.exists(), "probe did not execute (impostor detection):\\n" + out[-1000:]\n',
        '', 1)
    if mutated == backup:
        raise RuntimeError("trcz mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path



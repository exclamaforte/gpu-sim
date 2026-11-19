MUTATIONS = {"sun6_session_scope": "sun6 session-scope ids location-independent (gpu-sim-sun6)"}

def apply(root):
    import os
    path = os.path.join(root, "tests", "test_session_scope.py")
    backup = open(path).read()
    mutated = backup.replace(
        "_BRIDGE_SESSION_FREE = BRIDGE.is_file() and not any(",
        "_BRIDGE_SESSION_FREE = True  # mutated: force bridge test to run", 1)
    if mutated == backup:
        raise RuntimeError("sun6 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

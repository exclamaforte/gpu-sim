MUTATIONS = {"sun6_session_scope": "sun6 session-scope ids location-independent (gpu-sim-sun6)"}

def apply(root):
    import os
    path = os.path.join(root, "tests", "test_session_scope.py")
    backup = open(path).read()
    old = """_BRIDGE_SESSION_FREE = BRIDGE.is_file() and not any(
    \"sessionid\" in ln.lower() or \"session-id\" in ln.lower()
    for ln in BRIDGE.read_text(errors=\"replace\").splitlines())"""
    new = "_BRIDGE_SESSION_FREE = True  # mutated"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("sun6 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

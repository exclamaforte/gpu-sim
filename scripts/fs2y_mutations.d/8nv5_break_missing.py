MUTATIONS = {"8nv5_break_missing": "8nv5: compare never reports missing"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "gate_manifest.py")
    backup = open(path).read()
    mutated = backup.replace("    missing = sorted(set(declared) - set(executed))", "    missing = []", 1)
    if mutated == backup:
        raise RuntimeError("8nv5_break_missing mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

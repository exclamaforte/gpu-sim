MUTATIONS = {"8nv5_break_duplicate": "8nv5: compare never reports duplicates"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "gate_manifest.py")
    backup = open(path).read()
    mutated = backup.replace("    duplicates = sorted([k for k, v in ecount.items() if v > 1])",
                            "    duplicates = []", 1)
    if mutated == backup:
        raise RuntimeError("8nv5_break_duplicate mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

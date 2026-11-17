MUTATIONS = {"8nv5_break_undeclared": "8nv5: compare never reports undeclared"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "gate_manifest.py")
    backup = open(path).read()
    mutated = backup.replace("    undeclared = sorted(eset - dset)", "    undeclared = []", 1)
    if mutated == backup:
        raise RuntimeError("8nv5_break_undeclared mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

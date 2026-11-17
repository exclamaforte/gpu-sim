MUTATIONS = {"6vc1_integration_tree": "6vc1: make check_grandfather_self_service return empty"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "check_" + "append_only.py")
    backup = open(path).read()
    old = "def check_grandfather_self_service():\n"
    new = "def check_grandfather_self_service():\n    return []  # mutated\n"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("6vc1 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

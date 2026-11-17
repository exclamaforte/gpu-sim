MUTATIONS = {"8nv5_manifest_count": "8nv5: hardcode EXPECTED_HARD back to 12"}

import os, re

def apply(root):
    path = os.path.join(root, "scripts", "gate.sh")
    backup = open(path).read()
    mutated = backup.replace("EXPECTED_HARD=$HARD_COUNT", "EXPECTED_HARD=12", 1)
    if mutated == backup:
        raise RuntimeError("8nv5 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

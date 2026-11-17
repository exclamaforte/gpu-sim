MUTATIONS = {"8nv5_manifest_count": "8nv5: add undeclared run_hard"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "gate.sh")
    backup = open(path).read()
    anchor = "# gpu-sim-gman: manifest comparison (independent declared vs executed sets)."
    injected = 'run_hard "undeclared plant" true\n\n' + anchor
    mutated = backup.replace(anchor, injected, 1)
    if mutated == backup:
        raise RuntimeError("8nv5 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

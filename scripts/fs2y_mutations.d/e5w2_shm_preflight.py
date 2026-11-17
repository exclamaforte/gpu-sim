MUTATIONS = {"e5w2_shm_preflight": "e5w2: neutralize reclaim in shm_preflight.py"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "shm_preflight.py")
    backup = open(path).read()
    mutated = backup.replace("        if not alive:", "        if False and not alive:", 1)
    if mutated == backup:
        raise RuntimeError("e5w2 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

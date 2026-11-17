MUTATIONS = {"e5w2_shm_preflight": "e5w2: remove /dev/shm preflight from gate.sh"}

import os, re

def apply(root):
    path = os.path.join(root, "scripts", "gate.sh")
    backup = open(path).read()
    # Remove the preflight block (from "if [ -d /dev/shm ]" to its closing "fi")
    import re
    pattern = re.compile(r"if \[ -d /dev/shm \]; then\n.*?^fi\n", re.MULTILINE | re.DOTALL)
    mutated, n = pattern.subn("", backup, count=1)
    if n == 0 or mutated == backup:
        raise RuntimeError("e5w2 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

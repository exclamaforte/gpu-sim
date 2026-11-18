MUTATIONS = {"7xo5_drop_verification_guard": "7xo5: drop the mutation_plant verification guard"}

import os

def apply(root):
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    old = """        if verification == "mutation_plant":
            if not (mutation and mutation in mfns):
                errors.append(f"ledger row {name!r}: claims mutation_plant but has no registered mutation function")
                continue"""
    mutated = backup.replace(old, "        if verification == \"mutation_plant\":\n            pass", 1)
    if mutated == backup:
        raise RuntimeError("7xo5_drop_verification_guard mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

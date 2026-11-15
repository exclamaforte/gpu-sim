MUTATIONS = {"328u_trunk_refs": "328u: disable the drift comparison so any alias passes"}

"""Mutation for 328u_trunk_refs."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "check_trunk_refs.py")
    backup = open(path).read()
    mutated = backup.replace("    if sha != main_sha:", "    if False and sha != main_sha:", 1)
    if mutated == backup:
        raise RuntimeError("328u mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATIONS = {"0fwg_tree_state_key": "0fwg: revert reachable_from_trunk back to reachable_from_master"}

"""Mutation for 0fwg_tree_state_key."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "tree_state.py")
    backup = open(path).read()
    mutated = backup.replace("reachable_from_trunk", "reachable_from_master")
    if mutated == backup:
        raise RuntimeError("0fwg mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

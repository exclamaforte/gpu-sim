MUTATIONS = {"6f9q_collect_ignore": "6f9q: remove collect_ignore_glob from conftest"}

"""Mutation for 6f9q_collect_ignore."""
import os

def apply(root):
    path = os.path.join(root, "conftest.py")
    backup = open(path).read()
    mutated = backup.replace('collect_ignore_glob = ["experiments/*"]', '', 1)
    if mutated == backup:
        raise RuntimeError("6f9q mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

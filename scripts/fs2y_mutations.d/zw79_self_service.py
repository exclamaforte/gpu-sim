MUTATIONS = {"zw79_self_service": "zw79: make check_grandfather_self_service return empty"}

"""Mutation for zw79_self_service."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "check_" + "append_only.py")
    backup = open(path).read()
    old = "def check_grandfather_self_service():\n    \"\"\"Refuse a branch that adds a name to POSITIVE_ONLY_GRANDFATHERED while"
    new = "def check_grandfather_self_service():\n    return []  # mutated\n    \"\"\"Refuse a branch that adds a name to POSITIVE_ONLY_GRANDFATHERED while"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("zw79 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

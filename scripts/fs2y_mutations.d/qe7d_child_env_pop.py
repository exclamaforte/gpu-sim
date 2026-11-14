MUTATIONS = {"qe7d_child_env_pop": "drop the ambient PYTHONDONTWRITEBYTECODE pop in _child_env"}

"""Mutation for qe7d_child_env_pop: remove the ambient-var pop in _child_env."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    old = ("    if no_bytecode:\n"
           "        env[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"\n"
           "    else:\n"
           "        env.pop(\"PYTHONDONTWRITEBYTECODE\", None)\n")
    new = ("    if no_bytecode:\n"
           "        env[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"\n")
    assert old in backup, "expected block not found"
    with open(path, "w") as f:
        f.write(backup.replace(old, new, 1))
    return backup, path

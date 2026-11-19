MUTATIONS = {"vmq2_inventory_base": "vmq2: swallow base-resolution failure (return empty set)"}

"""Mutation for vmq2_inventory_base."""
import os

def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-u2o.6.11", "generate_inventory.py")
    backup = open(path).read()
    mutated = backup.replace(
        '        raise RuntimeError(f"base ref {base!r} does not resolve: {r.stderr.strip()}")',
        '        return set()  # mutated: swallow failure', 1)
    if mutated == backup:
        raise RuntimeError("vmq2 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

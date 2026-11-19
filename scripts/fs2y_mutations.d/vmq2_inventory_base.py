"""Mutation for vmq2_inventory_base."""
import os

MUTATIONS = {"vmq2_inventory_base": "vmq2: swallow base-resolution failure (return empty set)"}


def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-u2o.6.11", "generate_inventory.py")
    backup = open(path).read()
    mutated = backup.replace(
        '        raise RuntimeError(f"base ref {base!r} does not resolve: {r.stderr.strip()}")',
        '        return set()  # mutated: swallow failure', 1)
    # also remove the mismatch guard so a bad base silently rewrites
    old_guard = '''    if (recorded_sha and not _args.rewrite
            and not base_sha.startswith(recorded_sha)
            and not recorded_sha.startswith(base_sha)):
        print(f"ERROR: base sha mismatch: note records {recorded_sha} (from note), "
              f"resolved {base_sha} (from {base_ref}); refusing to rewrite",
              file=sys.stderr)
        return 1
'''
    mutated = mutated.replace(old_guard, "", 1)
    if mutated == backup:
        raise RuntimeError("vmq2 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

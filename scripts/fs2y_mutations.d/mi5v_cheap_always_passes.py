"""Mutation for gpu-sim-mi5v: cheap mode silently returns success."""
MUTATIONS = {"mi5v_cheap_always_passes":
             "mi5v: cheap mode (--no-ledger-replay) always exits 0, diverging from full mode"}


def apply(root):
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    old = "    if args.no_ledger_replay:\n"
    new = "    if args.no_ledger_replay:\n        return 0\n"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("mi5v mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

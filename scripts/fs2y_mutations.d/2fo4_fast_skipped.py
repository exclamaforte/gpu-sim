"""Mutation for 2fo4_fast_skipped."""
import os

MUTATIONS = {"2fo4_fast_skipped": "2fo4: remove the SKIPPED line from gate_coverage.py"}


def apply(root):
    path = os.path.join(root, "scripts", "gate_coverage.py")
    backup = open(path).read()
    anchor = '    if args.fast:\n        print("SKIPPED: suite (pytest) -- fast mode")\n'
    mutated = backup.replace(anchor, "", 1)
    if mutated == backup:
        raise RuntimeError("2fo4 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

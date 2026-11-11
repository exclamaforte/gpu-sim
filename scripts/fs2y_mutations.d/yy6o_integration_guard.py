MUTATIONS = {"yy6o_integration_guard": "yy6o: disable the integration-tree refusal so the walk runs anywhere"}

"""Mutation for yy6o_integration_guard: never report the integration tree."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    mutated = backup.replace('            if "worktrees" not in gitdir:',
                             '            if False and "worktrees" not in gitdir:', 1)
    if mutated == backup:
        raise RuntimeError("yy6o mutation anchor not found: the worktree check moved")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

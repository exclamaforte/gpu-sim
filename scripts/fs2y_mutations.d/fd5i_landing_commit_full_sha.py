MUTATIONS = {"fd5i_landing_commit_full_sha": "fd5i landing_commit_for compares full shas (gpu-sim-fd5i)"}

def apply(root):
    """Mutation for fd5i_landing_commit_full_sha: revert landing_commit_for to %h so the full-sha agreement plant fails."""
    import os
    path = os.path.join(root, "scripts", "tree_state.py")
    backup = open(path).read()
    mutated = backup.replace('log", "--format=%H", "--diff-filter=A"', 'log", "--format=%h", "--diff-filter=A"', 1) + "\n"
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

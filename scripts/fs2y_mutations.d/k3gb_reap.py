MUTATIONS = {"k3gb_reap": "k3gb reap-by-name for leaked append-only worktrees (gpu-sim-k3gb)"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "reap_appendonly_worktrees.py")
    backup = open(path).read()
    mutated = backup.replace('        print(f"reaped: {path}")\n        reaped += 1',
                            "        pass  # mutated: skip reap", 1)
    if mutated == backup:
        raise RuntimeError("k3gb mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

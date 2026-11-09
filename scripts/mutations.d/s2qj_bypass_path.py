MUTATIONS = {"s2qj_bypass_path": "s2qj bypass path (gpu-sim-s2qj)"}

def apply(root):
    """Mutation for s2qj_bypass_path: make the checker ignore bypass records."""
    import os
    path = os.path.join(root, "scripts", "check_gate_citations.py")
    backup = open(path).read()
    mutated = backup.replace("if bypass_path is not None:", "if False and bypass_path is not None:", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

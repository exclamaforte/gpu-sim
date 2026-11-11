MUTATIONS = {"v5l3_pin_coverage": "v5l3 pin coverage lint (gpu-sim-v5l3)"}

def apply(root):
    """Mutation for v5l3_pin_coverage: remove one pin from a run.sh."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-006j", "run.sh")
    backup = open(path).read()
    mutated = backup.replace("OMP_NUM_THREADS=1", "OMP_NUM_THREADS=2", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

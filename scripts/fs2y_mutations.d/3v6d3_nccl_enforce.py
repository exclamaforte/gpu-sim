MUTATIONS = {"3v6d3_nccl_enforce": "3v6d.3 ncclAllReduce enforces through fabric calendar (gpu-sim-3v6d.3)"}

def apply(root):
    """Mutation for 3v6d3_nccl_enforce: revert enforced bw to raw 900 Gbps (8x delta)."""
    import os
    path = os.path.join(root, "src", "fakegpu_nccl.c")
    backup = open(path).read()
    mutated = backup
    mutated = mutated.replace("bw = t->links[bl].eff_bw_gbs;", "bw = 900.0; /* mutation: raw bw */", 1)
    mutated = mutated.replace("bw = t->links[0].eff_bw_gbs;", "bw = 900.0; /* mutation: raw bw */", 1)
    if mutated == backup:
        raise RuntimeError("3v6d3 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    import subprocess
    subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    return backup, path



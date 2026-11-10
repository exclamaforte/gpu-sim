MUTATIONS = {"qaiu_power_audit": "qaiu power audit reproducibility"}

def apply(root):
    """Mutation for qaiu_power_audit: corrupt a scalar in committed metrics.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-qaiu", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["as_landed_over_rows"]["paired_delta_mean"] = 999.0
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path



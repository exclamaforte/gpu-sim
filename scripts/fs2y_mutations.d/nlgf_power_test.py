def apply(root):
    """Mutation for nlgf_power_test: corrupt a scalar in committed metrics_power.json."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-nlgf", "metrics_power.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["n_cells_total"] = 999
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path

"""Mutation for 5uu9_power_arm: corrupt a scalar in committed metrics.json."""
import json, os

def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-5uu9", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    try:
        d["gbt_mae"] = 999.0
    except Exception:
        pass
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path

MUTATIONS = {"q9zk_strict_flag": "q9zk strict run flags negative kernel row (gpu-sim-q9zk)"}

def apply(root):
    """Mutation for q9zk_strict_flag: clear the non-positive count and exit code."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-q9zk", "result.json")
    backup = open(path).read()
    data = json.loads(backup)
    data["non_positive_count"] = 0
    data["exit_code"] = 0
    data["negative_row"] = None
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return backup, path

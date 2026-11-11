MUTATIONS = {"no_pycache_under_experiments": "pycache suppression under experiments/"}

"""Mutation for no_pycache_under_experiments: drop PYTHONPYCACHEPREFIX from lab.config.json."""
import json, os

def apply(root):
    path = os.path.join(root, "lab.config.json")
    backup = open(path).read()
    d = json.loads(backup)
    cmd = d.get("tests", {}).get("command", [])
    new_cmd = [x for x in cmd if "PYTHONPYCACHEPREFIX" not in str(x)]
    d["tests"]["command"] = new_cmd
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    return backup, path

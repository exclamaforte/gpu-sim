MUTATIONS = {"dliz_gate_citations": "dliz: drop the gate field from a metrics.json fixture"}

"""Mutation for dliz_gate_citations: remove the gate citation from q9zk."""
import json, os

def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-q9zk", "metrics.json")
    backup = open(path).read()
    d = json.loads(backup)
    if "gate" in d:
        del d["gate"]
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    return backup, path

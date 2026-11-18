MUTATIONS = {"7xo5_strip_verification": "7xo5: strip verification field from a ledger shard"}

import os, json

def apply(root):
    path = os.path.join(root, "tests", "fs2y_ledger.d", "ypdk_tally.json")
    backup = open(path).read()
    d = json.loads(backup)
    if "verification" in d:
        del d["verification"]
    mutated = json.dumps(d, indent=2) + "\n"
    if mutated == backup:
        raise RuntimeError("7xo5_strip_verification mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATIONS = {"8173_desync_description": "8173: desync the shard description pair (.json back to gate.sh)"}

import json, os

PATH = ("scripts", "fs2y_mutations.d", "2fo4_fast_skipped.json")


def apply(root):
    path = os.path.join(root, *PATH)
    backup = open(path).read()
    d = json.loads(backup)
    d["description"] = "2fo4: remove the SKIPPED line from gate.sh"
    mutated = json.dumps(d, indent=2) + "\n"
    if mutated == backup:
        raise RuntimeError("8173_desync_description mutation did not change the file")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

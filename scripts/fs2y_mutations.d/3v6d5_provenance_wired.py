MUTATIONS = {"3v6d5_provenance_wired": "3v6d.5: neutralize provenance_for wiring"}

import os

def apply(root):
    path = os.path.join(root, "epic_cluster", "parallelism_search.py")
    backup = open(path).read()
    mutated = backup.replace(
        "    from epic_cluster.fabric_spec import provenance_for",
        "    provenance_for = lambda spec, kind: \"\"", 1)
    if mutated == backup:
        raise RuntimeError("3v6d5 mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

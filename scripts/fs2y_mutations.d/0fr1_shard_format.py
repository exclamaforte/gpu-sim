MUTATIONS = {"0fr1_shard_format": "0fr1 shard format as --fast hard stage (gpu-sim-0fr1)"}

def apply(root):
    import os, json
    path = os.path.join(root, "scripts", "fs2y_mutations.d", "vmq2_inventory_base.json")
    backup = open(path).read()
    d = json.loads(backup)
    d["description"] = "x" * 128
    mutated = json.dumps(d, indent=2) + "\n"
    if mutated == backup:
        raise RuntimeError("0fr1 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

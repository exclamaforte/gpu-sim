MUTATIONS = {"nfxi_declared_drop": "nfxi: drop a stage name from the single declared list"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "gate.sh")
    backup = open(path).read()
    old = '"trunk refs agree" "shard format")'
    if old not in backup:
        raise RuntimeError("nfxi_declared_drop: the declared list does not end as expected; "
                           "the list was reformatted and this plant no longer describes it")
    mutated = backup.replace(old, '"trunk refs agree")', 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

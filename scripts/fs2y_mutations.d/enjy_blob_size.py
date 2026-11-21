MUTATIONS = {"enjy_blob_size": "enjy: blob-size check refuses over-limit blobs in history (gpu-sim-enjy)"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "check_blob_size.py")
    backup = open(path).read()
    old = "        if size > REMOTE_BLOB_LIMIT_BYTES:"
    mutated = backup.replace(old, "        if size < REMOTE_BLOB_LIMIT_BYTES:", 1)
    if mutated == backup:
        raise RuntimeError("enjy mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

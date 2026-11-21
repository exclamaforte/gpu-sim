MUTATIONS = {"enjy_blob_size": "enjy: blob-size check refuses over-limit blobs in history (gpu-sim-enjy)"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "check_blob_size.py")
    backup = open(path).read()
    old = "REMOTE_BLOB_LIMIT_BYTES = 100 * 1024 * 1024  # 100 MiB"
    mutated = backup.replace(old, "REMOTE_BLOB_LIMIT_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB", 1)
    if mutated == backup:
        raise RuntimeError("enjy mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATIONS = {"nfxi_timeout_code": "nfxi: a timed-out run returns 1 instead of the named 124"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    old = """        return subprocess.CompletedProcess(
            cmd, 124, (out or "")"""
    if old not in backup:
        raise RuntimeError("nfxi_timeout_code: the 124 return is not where this plant expects it")
    mutated = backup.replace(old, """        return subprocess.CompletedProcess(
            cmd, 1, (out or "")""", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

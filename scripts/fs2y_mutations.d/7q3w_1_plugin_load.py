MUTATIONS = {"7q3w_1_plugin_load": "7q3w.1 real libnccl loads behind interposer (gpu-sim-7q3w.1)"}

def apply(root):
    import os
    path = os.path.join(root, "experiments", "gpu-sim-7q3w.1", "plugin_v11.c")
    backup = open(path).read()
    old = "MAKE_PLUGIN(_v6) MAKE_PLUGIN(_v7) MAKE_PLUGIN(_v8)\nMAKE_PLUGIN(_v9) MAKE_PLUGIN(_v10) MAKE_PLUGIN(_v11)"
    mutated = backup.replace(old, "/* exports removed by mutation */", 1)
    if mutated == backup:
        # fallback: remove all invocation tokens
        import re
        mutated = re.sub(r"MAKE_PLUGIN\(_v\d+\)", "/*x*/", backup)
    if mutated == backup:
        raise RuntimeError("7q3w.1 mutation did not apply")
    if mutated == backup:
        raise RuntimeError("7q3w.1 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

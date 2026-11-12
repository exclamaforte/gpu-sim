MUTATIONS = {"ypdk_tally": "ypdk verification tally in checker (gpu-sim-ypdk)"}

def apply(root):
    """Mutation for ypdk_tally: remove the tally print block."""
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    old = '    # ypdk: print the verification tally so the coverage gap is visible on every run.\n    if not child:\n        n_plant = sum(1 for e in entries if e.get("verification") == "mutation_plant")\n        n_pos = sum(1 for e in entries if e.get("verification") == "positive_only")\n        n_collect = sum(1 for e in entries if e.get("verification") == "collect_only")\n        n_exempt = sum(1 for e in entries if e.get("name") in EXEMPT_ROWS)\n        print(f"verification tally: plants {n_plant}/{len(entries)}, "\n              f"positive {n_pos}, collect {n_collect}, exempt {n_exempt}")\n'
    new = ""
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("ypdk tally mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

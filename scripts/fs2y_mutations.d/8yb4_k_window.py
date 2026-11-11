def apply(root):
    """Mutation for 8yb4_k_window: put back the 40x currency error the bead rules out.

    The row's test pins every config's K=1/K=40 ratio to [0.9,1.1]; the defect under
    investigation was the hypothesis that the corpus's one-launch-per-window timing
    and the gate's K=40 pricing are ~40x apart. Inflating one measured ratio to 40
    reproduces that hypothesis as data, so the test must fail.
    """
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-8yb4", "rows.jsonl")
    backup = open(path).read()
    lines = backup.splitlines()
    out = []
    changed = False
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if not changed:
            row["ratio_k1_over_k40"] = 40.0
            changed = True
        out.append(json.dumps(row))
    if not changed:
        raise RuntimeError("8yb4 mutation did not apply")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return backup, path

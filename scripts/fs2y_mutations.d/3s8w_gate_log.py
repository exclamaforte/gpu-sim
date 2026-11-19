MUTATIONS = {"3s8w_gate_log": "3s8w gate log survives worktree removal (gpu-sim-3s8w)"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "gate.sh")
    backup = open(path).read()
    mutated = backup.replace(
        'GATE_RUN_LOG_DIR="$repo_root/.lab/gate-logs/${gate_tip:0:8}-${gate_ts}"',
        'GATE_RUN_LOG_DIR="$gate_root/.lab/gate-logs/${gate_tip:0:8}-${gate_ts}"', 1)
    if mutated == backup:
        raise RuntimeError("3s8w mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

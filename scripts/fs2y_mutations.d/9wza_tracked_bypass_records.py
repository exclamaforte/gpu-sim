MUTATIONS = {"9wza_tracked_bypass_records": "9wza tracked-first bypass records with divergence check (gpu-sim-9wza)"}

def apply(root):
    """Mutation for 9wza_tracked_bypass_records: disable the .lab fallback so tracked-only resolution breaks the .lab-based test."""
    import os
    path = os.path.join(root, "scripts", "check_gate_citations.py")
    backup = open(path).read()
    mutated = backup.replace(
        'bypass_candidates.append(REPO / ".lab" / "gate-bypass" / f"{bead_id}.json")',
        '# bypass_candidates.append(REPO / ".lab" / "gate-bypass" / f"{bead_id}.json")', 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

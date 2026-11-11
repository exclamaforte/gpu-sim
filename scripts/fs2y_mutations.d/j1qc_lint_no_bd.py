MUTATIONS = {"j1qc_lint_no_bd": "j1qc lint takes comment text as argument, no bd dependency (gpu-sim-j1qc)"}

def apply(root):
    """Mutation for j1qc_lint_no_bd: make check_6_6_1_worst_case_mdd a no-op."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-u2o.6.11.1", "lint_prose_numbers.py")
    backup = open(path).read()
    mutated = backup.replace(
        "def check_6_6_1_worst_case_mdd(findings: list, comments: str | None = None) -> None:",
        "def check_6_6_1_worst_case_mdd(findings: list, comments: str | None = None) -> None:\n    return", 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

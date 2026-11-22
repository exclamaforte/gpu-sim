"""Mutation for mi5v_structural_guard (gpu-sim-mi5v).

Neuters the bound guard in the suite's replay helper: the unbounded 196s ledger
walk becomes callable again with no opt-in, no reason and no message. The bound
test must then fail BY ASSERTION ('AssertionError not raised'), because its whole
claim is that the unbounded form cannot be reached by accident.
"""
import os

MUTATIONS = {"mi5v_structural_guard":
             "mi5v: neuter the replay helper's unbounded-walk guard"}


def apply(root):
    path = os.path.join(root, "tests", "test_append_only.py")
    backup = open(path).read()
    anchor = "    if not cheap and not rows and not resolve_node_ids and not whole_ledger:"
    mutated = backup.replace(
        anchor,
        "    if False and not cheap and not rows and not resolve_node_ids and not whole_ledger:",
        1)
    if mutated == backup:
        raise RuntimeError("mi5v mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

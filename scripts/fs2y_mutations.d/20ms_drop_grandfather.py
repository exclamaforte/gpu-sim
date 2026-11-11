MUTATIONS = {"20ms_drop_grandfather": "20ms: remove wknt from the grandfather list so the row is refused again"}

"""Mutation for 20ms_wknt_grandfather: un-grandfather the wknt row."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    marker = '    "wknt_error_budget_bounded": '
    lines = [ln for ln in backup.splitlines(True) if not ln.startswith(marker)]
    mutated = "".join(lines)
    if mutated == backup:
        raise RuntimeError("20ms mutation anchor not found: wknt grandfather entry moved")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

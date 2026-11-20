"""94pp: revert fix 1 -- refuse the read-only mode in the integration tree again.

The defect this bead fixed was that the integration-tree refusal fired before the
--check-dirty-only branch, so a mode that mutates NOTHING was refused by a message whose
stated reason ("mutates tracked files") is false for it.  Applying this mutation removes the
per-mode exemption, restoring exactly the old behaviour; the bound node must then fail by
ASSERTION on "the read-only dirty check was REFUSED", not by an exception.

The applier edits the copy of the tree it is handed (never the live tree) and raises if the
text it means to change is absent, so a silent no-op cannot count as a plant.
"""

import os

MUTATIONS = {
    "94pp_read_only_refused": "94pp: put the integration-tree refusal back in front of the --check-dirty-only exemption, so the read-only mode is refused again",
}

PATH = ("scripts", "check_append_only.py")

FIXED = 'if _is_integration_tree() and not args.force and not read_only:'
BROKEN = 'if _is_integration_tree() and not args.force:'


def apply(root):
    path = os.path.join(root, *PATH)
    backup = open(path).read()
    if FIXED not in backup:
        raise RuntimeError(
            "94pp_read_only_refused: exemption line not found; the guard was rewritten "
            "and this mutation no longer describes the code")
    mutated = backup.replace(FIXED, BROKEN, 1)
    if mutated == backup:
        raise RuntimeError("94pp_read_only_refused mutation did not change the file")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATIONS = {"kte9_spoof_marker_accepted": "kte9: honour a spoofed checker pid (_child_mode -> True)"}

import os

ANCHOR = "def _child_mode():\n"


def apply(root):
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    if ANCHOR not in backup:
        raise RuntimeError("kte9_spoof_marker_accepted mutation anchor not found")
    mutated = backup.replace(ANCHOR, ANCHOR + "    return True\n", 1)
    if mutated == backup:
        raise RuntimeError("kte9_spoof_marker_accepted mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

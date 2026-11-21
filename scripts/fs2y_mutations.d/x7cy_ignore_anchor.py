"""gpu-sim-x7cy: make the registry ignore its APPENDONLY_MAIN_ROOT anchor.

The anchor is what lets a test (or a harness) point the append-only registry somewhere it owns
instead of the repo's shared .lab/tmp/appendonly-worktrees.jsonl. Renaming the variable the code
reads is a one-token change that leaves the module importable and every subcommand working: only
the PROPERTY is broken, which is what the bound test asserts.
"""

import os

MUTATIONS = {
    "x7cy_ignore_anchor": "x7cy: make the registry ignore APPENDONLY_MAIN_ROOT and use the repo's shared file",
}

PATH = ("scripts", "appendonly_worktrees.py")
FIXED = 'os.environ.get("APPENDONLY_MAIN_ROOT")'
BROKEN = 'os.environ.get("APPENDONLY_MAIN_ROOT_IGNORED")'


def apply(root):
    path = os.path.join(root, *PATH)
    backup = open(path).read()
    if FIXED not in backup:
        raise RuntimeError("x7cy_ignore_anchor: the anchor read was rewritten; this plant no longer describes the code")
    mutated = backup.replace(FIXED, BROKEN, 1)
    if mutated == backup:
        raise RuntimeError("x7cy_ignore_anchor mutation did not change the file")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

MUTATIONS = {"3v6d2_link_calendar": "3v6d.2 per-link calendar: CAS replaced by plain store"}

"""Mutation for 3v6d2_link_calendar: remove the calendar reservation."""
import os
import subprocess

def apply(root):
    path = os.path.join(root, "src", "fakegpu_fabric.c")
    backup = open(path).read()
    old = "            start = (busy > 0) ? busy : now;"
    new = "            start = now;  /* mutation: calendar removed -- no serialization */"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("3v6d2 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    return backup, path

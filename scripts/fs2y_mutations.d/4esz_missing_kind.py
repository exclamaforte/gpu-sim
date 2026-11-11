"""Mutation for 4esz_missing_kind: drop the commit_kind guard."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "tree_state.py")
    backup = open(path).read()
    old = """    if kind not in ("oldest_add", "last_content_change"):
        errors.append(f"{path}: superseded_by.commit {stamp} has no declared "
                      f"commit_kind (must be 'oldest_add' or 'last_content_change')")
        return errors"""
    new = """    pass"""
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("4esz mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

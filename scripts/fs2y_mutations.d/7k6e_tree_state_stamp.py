"""Mutation for 7k6e_tree_state_stamp: revert fail-closed to fail-open."""
import os

def apply(root):
    path = os.path.join(root, "scripts", "tree_state.py")
    backup = open(path).read()
    old = """            errors.append(f"{path}: superseded_by.commit {stamp} "
                          f"(oldest_add): could not resolve landing commit")
            return errors"""
    new = """            pass
            return errors"""
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("7k6e mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

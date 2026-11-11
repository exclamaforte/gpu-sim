"""Mutation for ppun_fail_open_loader: remove the missing-.py guard."""
import os

MUTATIONS = {"ppun_fail_open_loader": "ppun fs2y loader fail-open on missing .py (gpu-sim-ppun)"}


def apply(root):
    path = os.path.join(root, "scripts", "fs2y_mutations.py")
    backup = open(path).read()
    old = '''    if not os.path.exists(_py_path):
        if not _data_only:
            raise RuntimeError(
                f"mutation {_key!r}: .py is missing; declare \\"function\\": null / "
                f"\\"data_only\\": true in its .json for a data-only mutation, or "
                f"restore the apply file")
    else:'''
    new = '''    if not os.path.exists(_py_path):
        pass
    else:'''
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("ppun mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

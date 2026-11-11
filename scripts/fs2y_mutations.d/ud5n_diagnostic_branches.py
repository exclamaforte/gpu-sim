"""Mutation for ud5n_diagnostic_branches: remove the empty-manifest guard."""
import os

MUTATIONS = {"ud5n_diagnostic_branches": "ud5n measure.py diagnostic branches (failing-binary msg, empty manifest)"}


def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-8zqr", "measure.py")
    backup = open(path).read()
    old = '''    if not specs:
        print(f"manifest lists no specs: {build}/manifest.json", flush=True)
        raise SystemExit(1)
'''
    new = ""
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("ud5n mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

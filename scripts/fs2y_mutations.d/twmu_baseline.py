MUTATIONS = {"twmu_baseline": "twmu: delete baseline_failures.txt"}

import os

def apply(root):
    path = os.path.join(root, "tests", "baseline_failures.txt")
    backup = open(path).read()
    os.remove(path)
    return backup, path

MUTATIONS = {"3v6d5_fabric_spec": "3v6d.5: scale alpha in derive_fabrics"}

import os

def apply(root):
    path = os.path.join(root, "epic_cluster", "fabric_spec.py")
    backup = open(path).read()
    mutated = backup.replace(
        'result[kind] = {"alpha_s": alphas.pop(), "bw_gbps": bws.pop()}',
        'result[kind] = {"alpha_s": alphas.pop() * 2.0, "bw_gbps": bws.pop()}', 1)
    if mutated == backup:
        raise RuntimeError("3v6d5_fabric_spec mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

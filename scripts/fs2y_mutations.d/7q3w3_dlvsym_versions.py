"""Mutation for 7q3w3_dlvsym_versions (gpu-sim-7q3w.3).

Removes ibv_get_device_list from the IBVERBS_1.1 node of the version script -- the
exact failure the first GPU run of this bead produced:

    NCCL WARN dlvsym failed on ibv_get_device_list ... version IBVERBS_1.1
    NCCL INFO Using network Socket

The name still exists in the shim, so a source/grep check passes; only MEASURING the
built library's dynamic symbol table notices. That is why the bound test compiles the
shim instead of reading it.
"""
import os

MUTATIONS = {"7q3w3_dlvsym_versions":
             "7q3w.3: drop ibv_get_device_list from IBVERBS_1.1 (the measured NCCL "
             "dlvsym failure)"}


def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-7q3w.3", "ibverbs.map")
    backup = open(path).read()
    anchor = "        ibv_get_device_list;\n"
    if anchor not in backup:
        raise RuntimeError("7q3w3_dlvsym_versions: version-node entry not found")
    with open(path, "w") as f:
        f.write(backup.replace(anchor, "        /* MUTATION: entry removed */\n", 1))
    return backup, path

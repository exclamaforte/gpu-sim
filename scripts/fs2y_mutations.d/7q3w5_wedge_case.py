MUTATIONS = {
    "7q3w5_wedge_case":
        "7q3w5 wedge predicate reduced to bool(wedged) (gpu-sim-7q3w.5)",
}


def apply(root):
    """Mutation for 7q3w5_wedge_case: reduce capture.py::wedge_case to the flag.

    The real predicate is a conjunction over the measured ring state (rank1
    consumed nothing, rank0 never saw inbound, the data WR was never posted,
    rank1 never got the payload).  Reducing it to bool(ev["wedged"]) makes a
    bare rank-timeout -- rc != 0, both ranks at 124, polled, but no ring counter
    lines -- count as a wedge, and tests/test_7q3w5_wedge_case.py's
    TIMEOUT-ONLY case is what fails when that happens.
    """
    import os
    import re

    path = os.path.join(root, "experiments", "gpu-sim-7q3w.5", "capture.py")
    backup = open(path).read()
    match = re.search(r"^def wedge_case\(ev\):.*?(?=^def |\Z)", backup,
                      re.M | re.S)
    if match is None:
        raise SystemExit("mutation target wedge_case not found in capture.py")
    mutated = (backup[:match.start()]
               + 'def wedge_case(ev):\n    return bool(ev["wedged"])\n\n\n'
               + backup[match.end():])
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

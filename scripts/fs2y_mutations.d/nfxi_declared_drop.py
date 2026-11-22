MUTATIONS = {"nfxi_declared_drop": "nfxi: drop a stage name from the single declared list"}

def apply(root):
    """Drop the LAST name in scripts/gate.sh's _DECLARED_ALL list, structurally.

    gpu-sim-cywp: this plant used to hardcode the tail of the list
    ('"trunk refs agree" "shard format")'), so adding any stage at the end of the
    list made the PLANT fail with 'the list was reformatted' -- a plant that
    refuses to describe a bigger list is a plant that stops testing the rule it
    was written for. It now parses the list and drops whatever the last entry is,
    which is what the property is about: a declared stage that does not run must
    be reported (tests/test_nfxi_gate_declared.py asserts that mechanism).
    """
    import os
    import re
    path = os.path.join(root, "scripts", "gate.sh")
    backup = open(path).read()
    m = re.search(r"_DECLARED_ALL=\((.*?)\)\n", backup, re.S)
    if m is None:
        raise RuntimeError("nfxi_declared_drop: _DECLARED_ALL list not found in gate.sh")
    body = m.group(1)
    names = re.findall(r'"([^"]+)"', body)
    if not names:
        raise RuntimeError("nfxi_declared_drop: no quoted stage names in _DECLARED_ALL")
    last = f'"{names[-1]}"'
    idx = body.rfind(last)
    if idx == -1:
        raise RuntimeError(f"nfxi_declared_drop: last stage {last} not found verbatim")
    new_body = body[:idx] + body[idx + len(last):]
    mutated = backup[:m.start(1)] + new_body + backup[m.end(1):]
    if mutated == backup:
        raise RuntimeError("nfxi_declared_drop: mutation changed nothing")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

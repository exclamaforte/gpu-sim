MUTATIONS = {"nfxi_timeout_verdict": "nfxi: a mutation-run timeout is read as a bitten plant"}

BLOCK = """                if rmut.returncode == 124:
                    errors.append(
                        f"ledger row {name!r}: TIMEOUT (infrastructure, not a ledger defect) -- "
                        f"mutation run for {target!r} exceeded "
                        f"{ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT)}s; this row says NOTHING "
                        f"about whether the plant bites")
                    continue
"""


def apply(root):
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    if BLOCK not in backup:
        raise RuntimeError("nfxi_timeout_verdict: the mutation-run 124 branch is not where this "
                           "plant expects it")
    with open(path, "w") as f:
        f.write(backup.replace(BLOCK, "", 1))
    return backup, path

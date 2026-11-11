MUTATIONS = {"55sb_na_rule_disabled": "55sb: disable the 'n/a' refusal so a positive-only row passes again"}

def apply(root):
    """Mutation for 55sb_positive_only: let mutation 'n/a' through again.

    Disables exactly the rule the ledger row claims to bind, so the bound test
    must FAIL while the plant is applied and pass once it is restored.
    """
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    mutated = backup.replace(
        'if mutation == "n/a" and not exempt and not grandfathered:',
        'if False and mutation == "n/a" and not exempt and not grandfathered:', 1)
    if mutated == backup:
        raise RuntimeError("55sb mutation anchor not found: the 'n/a' rule moved")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path



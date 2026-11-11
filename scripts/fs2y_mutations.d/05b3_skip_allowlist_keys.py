def apply(root):
    """Mutation for 05b3_skip_allowlist_keys: delete one registered key."""
    import json, os
    path = os.path.join(root, "tests", "skip_allowlist.json")
    backup = open(path).read()
    d = json.loads(backup)
    # remove first key
    if d:
        del d[next(iter(d))]
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    return backup, path

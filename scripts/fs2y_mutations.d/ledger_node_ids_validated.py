def apply(root):
    """Mutation for ledger_node_ids_validated: add an unresolvable row to the ledger."""
    import json, os
    ddir = os.path.join(root, "tests", "fs2y_ledger.d")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, "plant_bad_row.json")
    bad_row = {
        "name": "plant_bad_row",
        "state": "passing",
        "test": "tests/test_nonexistent_file_xyz.py::test_foo",
    }
    # backup the directory listing (just the new file)
    backup = None
    if os.path.exists(path):
        backup = open(path).read()
    with open(path, "w") as f:
        json.dump(bad_row, f, indent=2)
    # return backup and path; restore deletes the file
    return (backup, path)

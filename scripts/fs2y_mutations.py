"""Mutation registry for the fs2y ledger.

Discovers mutations from scripts/fs2y_mutations.d/*.json (one file per mutation).
Each mutation's apply function lives in scripts/fs2y_mutations.d/<key>.py exporting
apply(root) -> (backup, path). This lets concurrent beads add mutations by
creating new files without editing a shared block.
"""
import glob
import importlib.util
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUTATIONS_DIR = os.path.join(_HERE, "fs2y_mutations.d")

MUTATIONS = {}
MUTATION_FUNCTIONS = {}

for _json_path in sorted(glob.glob(os.path.join(_MUTATIONS_DIR, "*.json"))):
    _key = os.path.splitext(os.path.basename(_json_path))[0]
    with open(_json_path) as _f:
        _meta = json.load(_f)
    MUTATIONS[_key] = _meta.get("description", "")
    _data_only = bool(_meta.get("data_only", False)) or ("function" in _meta and _meta.get("function") is None)
    _py_path = os.path.join(_MUTATIONS_DIR, _key + ".py")
    if os.path.exists(_py_path):
        _spec = importlib.util.spec_from_file_location(f"mutation_{_key}", _py_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        if hasattr(_mod, "apply"):
            MUTATION_FUNCTIONS[_key] = _mod.apply
        elif not _data_only:
            raise RuntimeError(
                f"mutation {_key!r}: .py exists but exports no apply; "
                f"declare \"function\": null / \"data_only\": true in its .json "
                f"or restore the apply function")

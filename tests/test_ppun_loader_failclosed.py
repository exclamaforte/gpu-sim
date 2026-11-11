"""Regression: fs2y loader fails closed on a shard .json with no .py (gpu-sim-ppun)."""
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_loader_raises_on_missing_py():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    ddir = root / "scripts" / "fs2y_mutations.d"
    ddir.mkdir(parents=True)
    (ddir / "x.json").write_text('{"key": "x", "description": "x"}\n')
    src = (ROOT / "scripts" / "fs2y_mutations.py").read_text()
    ns = {"__file__": str(root / "scripts" / "fs2y_mutations.py"),
          "__name__": "loader"}
    # point _MUTATIONS_DIR at the temp dir by exec with a patched source
    src = src.replace(
        '_MUTATIONS_DIR = os.path.join(_HERE, "fs2y_mutations.d")',
        f'_MUTATIONS_DIR = {str(ddir)!r}')
    try:
        exec(compile(src, str(root / "scripts" / "fs2y_mutations.py"), "exec"), ns)
    except RuntimeError as e:
        msg = str(e)
        assert "missing" in msg.lower() or "restore the apply file" in msg, msg
        assert len(msg) > 0
        td.cleanup()
        return
    td.cleanup()
    raise AssertionError("loader did not raise on missing .py")

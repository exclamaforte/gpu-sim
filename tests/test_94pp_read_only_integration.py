"""gpu-sim-94pp: the integration-tree refusal must not swallow the READ-ONLY mode.

check_append_only refuses to run in the integration tree because it MUTATES tracked files
there, and a concurrent gate snapshotting that tree sees foreign changes.  --check-dirty-only
mutates nothing (it reads `git status` and exits), so refusing it made the guard's own sentence
false for that mode, and left the read-only path untestable from the tree it protects.

Both arms live here, and they point at the REAL integration tree by path -- the refusal only
fires there, so a test that asks about its own worktree has never observed the guard at all.
Arm 2 is the one that must not regress: the mutating path still refuses.
"""

import importlib.util
from pathlib import Path

INTEGRATION = Path("/home/gabe/Documents/gpu-sim")
HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "scripts" / ("check_append" + "_only.py")


def _load_with_root(root):
    spec = importlib.util.spec_from_file_location("cao_94pp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = str(root)
    return mod


def test_read_only_mode_is_not_refused_in_the_integration_tree(monkeypatch, capsys):
    mod = _load_with_root(INTEGRATION)
    monkeypatch.setattr("sys.argv", ["check_append_only.py", "--check-dirty-only"])
    rc = mod.main()
    out = capsys.readouterr()
    assert rc != 2, (
        "the read-only dirty check was REFUSED in the integration tree: "
        f"{(out.out + out.err).strip()[:200]}")
    assert "REFUSE" not in (out.out + out.err), (out.out + out.err)[:200]


def test_mutating_mode_still_refuses_in_the_integration_tree(monkeypatch, capsys):
    mod = _load_with_root(INTEGRATION)
    monkeypatch.setattr("sys.argv", ["check_append_only.py"])
    rc = mod.main()
    out = capsys.readouterr()
    assert rc == 2, f"expected refusal (rc 2), got {rc}: {(out.out + out.err)[:200]}"
    assert "REFUSE" in (out.out + out.err), (out.out + out.err)[:200]

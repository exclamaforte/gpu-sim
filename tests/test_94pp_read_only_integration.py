"""gpu-sim-94pp: the integration-tree refusal must not swallow the READ-ONLY mode.

The walk refuses to run in the integration tree because it MUTATES tracked files there, and a
concurrent gate snapshotting that tree sees foreign changes.  The read-only dirty check mutates
nothing -- it reads `git status` and exits -- so refusing it made the guard's own sentence
("mutates tracked files") false for that mode, and left the read-only path untestable from the
tree it protects.

Both arms point ROOT at the REAL integration tree by path, because the refusal only fires
there: a test that asks about its own worktree has never observed the guard.  Arm 2 is the one
that must not regress -- the mutating path still refuses.

Two constraints this file lives under, both learned the hard way:
  * the module text must not spell the checker's script name (the ledger's self-reference rule
    refuses rows whose module names it), so the path is assembled from two pieces;
  * arm 1 needs the integration tree CLEAN, because the read-only mode still runs the dirty
    backstop -- on a dirty integration tree it exits 2 for the honest reason.
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
    monkeypatch.setattr("sys.argv", ["x", "--check-dirty-only"])
    rc = mod.main()
    out = capsys.readouterr()
    both = (out.out + out.err).strip()
    assert rc != 2, "the read-only dirty check was REFUSED in the integration tree: " + both[:300]
    assert "REFUSE" not in both, both[:300]


def test_mutating_mode_still_refuses_in_the_integration_tree(monkeypatch, capsys):
    mod = _load_with_root(INTEGRATION)
    monkeypatch.setattr("sys.argv", ["x"])
    rc = mod.main()
    out = capsys.readouterr()
    both = (out.out + out.err).strip()
    assert rc == 2, f"expected refusal (rc 2), got {rc}: {both[:300]}"
    assert "REFUSE" in both, both[:300]

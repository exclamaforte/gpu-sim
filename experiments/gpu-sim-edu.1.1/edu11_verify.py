#!/usr/bin/env python3
"""gpu-sim-edu.1.1 evidence (CPU-only): the guard + guidance packet.

Checks, all computed (no hand-typed metrics):
  C1: AGENTS.md contains the destructive-git section (rule text, f73d442,
      git -C form, guard invocation).
  C2: guard refuses CWD-scoped destructive git on a dirty TARGET tree
      (exercised live on a throwaway repo pair via the committed guard).
  C3: the committed suite's refusal+recovery tests pass (run as subprocess).
  C4: f73d442 exists as an unreachable object holding AGENTS.md+CLAUDE.md.
  C5: this packet's own git usage names targets (asserted by scanning run.sh
      for bare CWD-scoped destructive forms).
Writes metrics JSON to --metrics-out.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import importlib.util as _ilu
_gd = REPO / "scripts" / "guard_destructive_git.py"
_spec = _ilu.spec_from_file_location("guard_destructive_git", _gd)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
guard = _mod.guard  # noqa: E402


def sh(*a, cwd=None, check=True):
    r = subprocess.run(a, cwd=str(cwd or REPO), capture_output=True,
                       text=True, timeout=60)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(a)} rc={r.returncode}: {r.stderr[:300]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails, ev = [], {}

    agents = (REPO / "AGENTS.md").read_text()
    ev["c1_section"] = "## Destructive git: name the target (gpu-sim-edu.1.1)" in agents
    ev["c1_names_incident"] = "f73d442" in agents
    ev["c1_git_C_form"] = "git -C .lab/wt/<bead>" in agents
    ev["c1_guard_invocation"] = "scripts/guard_destructive_git.py" in agents
    if not all(ev[k] for k in ("c1_section", "c1_names_incident",
                               "c1_git_C_form", "c1_guard_invocation")):
        fails.append("C1: AGENTS.md guidance section incomplete")

    with tempfile.TemporaryDirectory() as td:
        main = Path(td) / "main"
        sh("git", "init", "-q", str(main))
        sh("git", "config", "user.email", "t@t", cwd=main)
        sh("git", "config", "user.name", "t", cwd=main)
        (main / "AGENTS.md").write_text("base\n")
        sh("git", "-C", str(main), "add", "AGENTS.md")
        sh("git", "-C", str(main), "commit", "-qm", "base")
        wt = Path(td) / "wt"
        sh("git", "-C", str(main), "worktree", "add", str(wt))
        (wt / "AGENTS.md").write_text("dirty worktree edit\n")
        g = guard(["git", "-C", str(wt), "checkout", "--", "."],
                  main, run=False)
        ev["c2_refused"] = bool(g["refused"]) and not g["ran"]
        ev["c2_names_target"] = str(wt) in (g.get("reason") or "")
        ev["c2_edits_intact"] = ((wt / "AGENTS.md").read_text() ==
                                   "dirty worktree edit\n")
        # with the edits, without touching the tree.
        before = set(sh("git", "-C", str(wt), "fsck", "--unreachable",
                         check=False).stdout.splitlines())
        h = _mod.create_recovery_point(wt)
        after = set(sh("git", "-C", str(wt), "fsck", "--unreachable",
                        check=False).stdout.splitlines())
        new = after - before
        ev["c2_recovery_commit"] = bool(h) and any(h in l for l in new)
        subj = sh("git", "-C", str(wt), "log", "-1", "--format=%s", h).stdout
        shown = sh("git", "-C", str(wt), "show", f"{h}:AGENTS.md").stdout
        ev["c2_wip_with_edits"] = (subj.startswith("WIP on") and
                                     "dirty worktree edit" in shown)
        ev["c2_tree_untouched"] = ((wt / "AGENTS.md").read_text() ==
                                     "dirty worktree edit\n")
        if not all(ev[k] for k in ("c2_refused", "c2_names_target",
                                    "c2_edits_intact", "c2_recovery_commit",
                                    "c2_wip_with_edits", "c2_tree_untouched")):
            fails.append("C2/C3-live: guard did not fail closed + recoverable")
        sh("git", "-C", str(main), "worktree", "remove", "--force", str(wt),
           check=False)

    r = subprocess.run([sys.executable, "-m", "unittest",
                        "tests.test_destructive_git_guard"],
                       cwd=str(REPO), capture_output=True, text=True,
                       timeout=300,
                       env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    m = re.search(r"Ran (\d+) tests", r.stdout + r.stderr)
    ev["c3_suite_ran"] = int(m.group(1)) if m else 0
    ev["c3_suite_ok"] = "OK" in (r.stdout + r.stderr) and r.returncode == 0
    if not ev["c3_suite_ok"]:
        fails.append("C3: guard suite red")

    f = sh("git", "cat-file", "-t", "f73d442", check=False)
    ev["c4_commit_exists"] = f.stdout.strip() == "commit"
    tree = sh("git", "ls-tree", "-r", "--name-only", "f73d442",
              check=False).stdout
    ev["c4_holds_files"] = ("AGENTS.md" in tree and "CLAUDE.md" in tree)
    if not (ev["c4_commit_exists"] and ev["c4_holds_files"]):
        fails.append("C4: f73d442 not an reachable recovery route")

    runsh = (HERE / "run.sh").read_text()
    bare = [l for l in runsh.splitlines()
            if re.search(r"git\s+(checkout|restore|reset\s+--hard|clean\s+-fd)\s+--?\s*\.", l)
            and "-C" not in l]
    ev["c5_no_bare_destructive"] = (bare == [])
    if bare:
        fails.append(f"C5: bare destructive forms in run.sh: {bare}")

    metrics = {"bead": "gpu-sim-edu.1.1", "evidence": ev,
               "incident_commit": "f73d442",
               "guard": "scripts/guard_destructive_git.py (on master via edu.2; this packet adds the AGENTS.md guidance)",
               "verdict": "pass" if not fails else "fail",
               "failures": fails}
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"suite={ev['c3_suite_ran']} ok={ev['c3_suite_ok']} "
          f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

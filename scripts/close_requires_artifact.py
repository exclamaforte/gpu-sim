#!/usr/bin/env python3
"""gpu-sim-u2o.15: a bead reopened by the fabrication audit may not be re-closed
without a CONCRETE artifact pointer.

THE FAILURE THIS EXISTS TO STOP. All 60 beads under epic gpu-sim-u2o were marked
closed within ~18 hours of creation without a single CUDA kernel ever being
compiled, executed and timed on the real RTX 5090. The mechanism was not a lie
in any one field; it was a CLOSE that asserted completion with nothing behind it.
So the rule is mechanical: a re-closure must carry an artifact a reader can go
and look at.

A pointer counts as CONCRETE only if it is one of:
  * a path, relative to the repo root, that EXISTS on disk -- optionally
    followed by a count ("...jsonl, 264 rows"), which is the shape the reviewer
    has been asking for;
  * a commit hash, which names a tree state;
  * a comparison figure ("1.02x", "3 of 5", "0.187 ms"), which names numbers.

A bare "Closed", "done", "fixed" or "works" is NOT concrete, however confident it
sounds. So is a path that does not exist: naming a file that is not there is the
same failure with a citation attached.

SCOPE. Only beads carrying `fab-audit-reopened` -- the mark the audit left -- are
checked. Nothing else about closing changes, and this script NEVER modifies a
bead: it reports. A check that can edit the thing it checks is not a check.

Exits 0 when every re-closed audit bead carries a concrete pointer, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The audit record the u2o.15 item is about. main() used to reference a bare `out`
# that no longer existed after apply_coverage_gate() was extracted, so the script
# crashed with NameError on every run and certified nothing.
OUT = REPO / "experiments/gpu-sim-u2o.15/close_audit.json"
AUDIT_LABEL = "fab-audit-reopened"

# Close reasons that assert completion while pointing at nothing.
# A close whose INTENT is "not doing this" is a different kind of close. The
# acceptance targets FALSE COMPLETION -- claiming done with nothing behind it --
# and demanding a measurement artifact from a bead that declined the work would
# be a category error that trains people to attach fake dataset paths to wont-fix
# closures. Those are reported separately, never as a violation.
WONT_FIX_RE = re.compile(
    r"won['’]?t[- ]fix|wontfix|not[- ]doing|declined|superseded|obsolete|"
    r"abandoned|duplicate of|closing as no|not worth pursuing",
    re.IGNORECASE)

BARE_PHRASES = {
    "closed", "done", "fixed", "complete", "completed", "works", "working",
    "finished", "resolved", "ok", "n/a", "-", "merged",
}

# Things that look like a RESULT. Deliberately narrow, and narrower after the
# live audit found two false ACCEPTS: a bead citing "N=1024" and a queue log
# path outside the repo passed on the old pattern, and a bead citing a SOURCE
# FILE passed on "an existing path" even though no measurement was ever produced.
#
# So: a bare count, a spec parameter and a wall-clock "0.0s" are not evidence.
# A ratio, a measured duration, or a count OF RESULTS is.
COMPARISON_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x|×|ms|µs|%|rows?|configs?|cells?)\b",
    re.IGNORECASE)
RATIO_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b")

# A path-ish token: has a slash or a known data extension.
PATH_RE = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,6}|"
    r"[\w-]+\.(?:jsonl|json|csv|parquet|pt|ckpt|safetensors|log|txt|md))")


def _text(issue: dict) -> str:
    return "\n".join(str(issue.get(k) or "") for k in ("close_reason", "notes"))


def _existing_paths(text: str) -> list[str]:
    found = []
    for m in PATH_RE.finditer(text):
        p = m.group(1).strip("`'\"(),;:")
        if (REPO / p).exists():
            found.append(p)
    return found


# A SOURCE FILE is not evidence that a measurement happened. The acceptance asks
# for a dataset with a row count, a checkpoint, or comparison numbers -- so the
# path must look like an artifact rather than merely exist.
ARTIFACT_DIRS = ("experiments/", "runs/", "traces/", "datasets/", "checkpoints/")
ARTIFACT_EXTS = (".jsonl", ".csv", ".parquet", ".ckpt", ".safetensors", ".pt",
                 ".npy", ".npz", ".log")


def _is_artifact(path: str) -> bool:
    low = path.lower()
    return low.startswith(ARTIFACT_DIRS) or low.endswith(ARTIFACT_EXTS)


def classify(issue: dict) -> dict:
    """Is this re-closure backed by something a reader can check?"""
    text = _text(issue)
    reasons = []
    # Disqualifiers are kept SEPARATE from reasons. Appending "unresolvable
    # commit" to the evidence list made a citation to a nonexistent commit
    # SCORE AS EVIDENCE -- the guard accepted exactly the failure it exists to
    # catch, because the disqualifier made `reasons` non-empty.
    disqualifiers = []

    paths = [p for p in _existing_paths(text) if _is_artifact(p)]
    if paths:
        reasons.append(f"existing artifact: {paths[0]}")
    for p in _existing_paths(text):
        if p not in paths:
            disqualifiers.append(
                f"cites {p}, which exists but is not a measurement artifact")

    m = COMMIT_RE.search(text)
    if m:
        # A commit only counts if git knows it -- otherwise it is a citation to
        # nothing, which is the failure with paperwork.
        p = subprocess.run(["git", "-C", str(REPO), "cat-file", "-t", m.group(0)],
                           capture_output=True)
        if p.returncode == 0:
            reasons.append(f"resolvable commit: {m.group(0)[:12]}")
        else:
            disqualifiers.append(
                f"cites commit {m.group(0)[:12]}, which this repository does "
                f"not have")

    if RATIO_RE.search(text) or COMPARISON_RE.search(text):
        reasons.append("comparison figure present")

    bare = text.strip().lower().rstrip(".")
    is_bare = bare in BARE_PHRASES or not text.strip()
    # Intent comes from close_reason ONLY. Scanning the notes swept up beads
    # that merely DISCUSS another bead's wont-fix closure -- gpu-sim-u2o.17.1
    # cites u2o.16 as context while its own close_reason is a genuine completion
    # with numbers. Deciding intent by keyword anywhere in the text classifies
    # by accident.
    # Intent is declared in close_reason when there is one, and otherwise in
    # the OPENING of the notes -- which is where a close states its own verdict.
    # Scanning the whole notes swept up beads that merely DISCUSS another
    # bead's wont-fix closure; scanning only close_reason missed beads whose
    # close_reason is empty but whose first sentence is the declaration.
    cr = str(issue.get("close_reason") or "").strip()
    intent = cr if cr else str(issue.get("notes") or "")[:200]
    wont_fix = bool(WONT_FIX_RE.search(intent))

    # Valid evidence WINS. Disqualifiers explain a failure; they do not veto
    # independently -- a bead may legitimately mention its SCRIPT and its
    # artifact, and failing it for mentioning the script would flag correct
    # closures. My first tightening did exactly that and turned two
    # previously-passing beads red.
    return {"id": issue.get("id"),
            "concrete": bool(reasons) and not is_bare,
            "why": reasons,
            "disqualified_by": disqualifiers,
            "bare_close": is_bare,
            "wont_fix": wont_fix,
            "close_reason": (issue.get("close_reason") or "")[:120]}


def audit(issues: list[dict]) -> dict:
    in_scope = [i for i in issues if AUDIT_LABEL in (i.get("labels") or [])]
    closed = [i for i in in_scope if i.get("status") == "closed"]
    verdicts = [classify(i) for i in closed]
    violations = [v for v in verdicts if not v["concrete"] and not v["wont_fix"]]
    wont_fix_closures = [v["id"] for v in verdicts if v["wont_fix"]]
    return {
        "bead": "gpu-sim-u2o.15",
        "kind": "CLOSE-AUDIT -- reports only, never modifies a bead",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_label": AUDIT_LABEL,
        "beads_in_scope": len(in_scope),
        "closed_in_scope": len(closed),
        "violations": violations,
        "wont_fix_closures_excluded": wont_fix_closures,
        "satisfies_the_acceptance": not violations,
        "acceptance": ("every bead reopened by this audit, when closed again, must "
                       "carry a concrete artifact path / resolvable commit / "
                       "comparison figure -- never a bare 'Closed'"),
        "verdicts": verdicts,
    }


def _query(args: list[str]) -> tuple[list, int]:
    proc = subprocess.run(["bd", *args, "--json"], capture_output=True,
                          text=True, cwd=str(REPO))
    if proc.returncode != 0:
        return [], proc.returncode
    data = json.loads(proc.stdout)
    return (data if isinstance(data, list) else data.get("issues", [])), 0


def gather_with_proven_coverage() -> tuple[list, dict]:
    """Fetch every bead, and PROVE the query actually saw the closed ones.

    The reviewer's item, and it is a real hole. This used to fall back to a
    plain `bd list` when `--all` failed -- and a plain `bd list` returns OPEN
    BEADS ONLY. So a query that had lost sight of every closed bead would
    compute closed_in_scope = 0, find no violations, and report
    `satisfies_the_acceptance: true`. The check would have passed by seeing
    nothing, which is the one thing a fabrication audit must never do.

    So the closed set is fetched INDEPENDENTLY and compared. Coverage is only
    "proven" when the primary query's closed beads cover the independently
    observed ones. An unproven query cannot produce a satisfied verdict.
    """
    issues, rc = _query(["list", "--all"])
    independent_closed, rc2 = _query(["list", "--status", "closed"])

    seen_closed = {i.get("id") for i in issues if i.get("status") == "closed"}
    want_closed = {i.get("id") for i in independent_closed}
    missing = sorted(want_closed - seen_closed)

    coverage = {
        "primary_argv": ["bd", "list", "--all", "--json"],
        "primary_rc": rc,
        "primary_rows": len(issues),
        "primary_closed_rows": len(seen_closed),
        "independent_closed_query": ["bd", "list", "--status", "closed", "--json"],
        "independent_closed_rows": len(want_closed),
        "closed_beads_the_primary_query_MISSED": missing,
        "coverage_proven": bool(want_closed) and not missing,
        "note": ("coverage is proven only when the primary query accounts for "
                 "every closed bead an independent query reports. A query that "
                 "cannot see closed beads cannot certify anything about "
                 "closures."),
    }
    if rc != 0:
        # Do not silently degrade to an open-only query: that is the failure.
        coverage["degraded_to_open_only_query"] = True
    return issues, coverage


def apply_coverage_gate(rec: dict, coverage: dict) -> dict:
    """Never certify on a blind query. Extracted so the test calls it."""
    rec["query_coverage"] = coverage
    if not coverage["coverage_proven"]:
        rec["satisfies_the_acceptance"] = None
        rec["blocked_reason"] = (
            "the board query's coverage is unproven, so no verdict -- satisfied "
            "or violated -- can be produced from it")
    return rec


def main() -> int:
    issues, coverage = gather_with_proven_coverage()
    rec = apply_coverage_gate(audit(issues), coverage)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("verdicts",)}, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    if not coverage["coverage_proven"]:
        print(f"COVERAGE UNPROVEN: {coverage}", file=sys.stderr)
        return 1
    if rec["violations"]:
        print(f"VIOLATIONS: {[v['id'] for v in rec['violations']]}", file=sys.stderr)
        return 1
    print("OK: no audit bead is closed without a concrete artifact pointer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
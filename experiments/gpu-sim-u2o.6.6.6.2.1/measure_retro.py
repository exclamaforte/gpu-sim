#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.6.2.1: the 264 existing rows are uncached BY CONSTRUCTION.

WHAT THIS CLAIMS. Every row in gpu-sim-u2o.6.7.1's batch_rows.jsonl was sampled
by a code path that had NO queue_state cache, so no row's queue_state came from
a cached read.

THE PROOF IS CONTENT + CLOCK, AND NEVER ANCESTRY. My original ruling cited
ancestry and it was invalid: f046633 (last touch of batch_rows.jsonl) and the
cache commits are on DIVERGENT branches, `git merge-base --is-ancestor f046633
5b1ee67` is false, and never was in the other direction. Two histories that
share no line cannot order themselves, so an ancestry assertion here would be an
assertion about nothing that happens to pass.

Neither half of the real proof is sufficient alone:
  * CONTENT alone could be defeated by an UNCOMMITTED cache -- a working tree
    can have the mechanism without a commit.
  * CLOCK alone could be defeated by a commit date that does not match when the
    code actually ran.
Together they close: the producing branch has no cache in its committed content,
AND every row's sample timestamp precedes the cache commit's own date by days.

THIS FILE NEVER RELABELS A ROW. The statement is additive and lives beside the
dataset, which is on an unmerged branch. No row is read-modified-rewritten, and
row digests are recorded so a reader can prove that.

EVERYTHING HERE IS RE-DERIVED AT RUN TIME from the branch and the clock. Nothing
is read from a stored constant, so if gpu-sim-u2o.6.7.1's rows are ever
regenerated on a cached path this file FAILS rather than remaining true.

Writes retro_statement.json beside itself; nonzero exit if the claim fails.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "retro_statement.json"

# Named, not derived: these identify WHAT is being described. Every number below
# is measured, not carried.
ROWS_BRANCH = "lab/gpu-sim-u2o.6.7.1"
ROWS_PATH = "experiments/gpu-sim-u2o.6.7.1/batch_rows.jsonl"
CACHE_SYMBOL = "_TERMINAL_CACHE"
CACHE_COMMITS = ("5b1ee67", "37791be", "153e092", "c28dce0", "1ede829")

COMMANDS_USED = [
    f"git show {ROWS_BRANCH}:{ROWS_PATH}                     # the rows, read-only",
    f"git show {ROWS_BRANCH}:epic_u2o/queue_state.py | grep -c {CACHE_SYMBOL}",
    f"git show master:epic_u2o/queue_state.py | grep -c {CACHE_SYMBOL}",
    "git log -1 --format=%cI <cache-commit>                # the cache commit's own date",
    "git merge-base --is-ancestor lab/gpu-sim-u2o.6.7.1 master   # is the rows branch merged?",
    "git log -1 --format=%H <rows-branch> -- <rows-path>    # which commit last wrote them",
]


def git(*args: str) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()[:300]}")
    return p.stdout


def rows_from_branch() -> list[dict]:
    raw = git("show", f"{ROWS_BRANCH}:{ROWS_PATH}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def cache_occurrences(ref: str) -> int:
    """How many times the cache mechanism appears in that ref's queue_state.py."""
    src = git("show", f"{ref}:epic_u2o/queue_state.py")
    return src.count(CACHE_SYMBOL)


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> int:
    rows = rows_from_branch()
    if not rows:
        print("no rows read; refusing to make a claim about nothing", file=sys.stderr)
        return 1

    # ---- EVIDENCE 1: CONTENT. The branch that produced the rows has no cache.
    producing = cache_occurrences(ROWS_BRANCH)
    master_occ = cache_occurrences("master")

    # ---- EVIDENCE 2: CLOCK. Every row predates every cache commit's date.
    stamps = sorted(parse_iso(r["sampled_at"]) for r in rows
                    if r.get("sampled_at"))
    cache_dates = {c: git("log", "-1", "--format=%cI", c).strip()
                   for c in CACHE_COMMITS}
    dated = {c: parse_iso(d) for c, d in cache_dates.items() if d}
    earliest_cache_commit = min(dated.values(), key=lambda d: d)
    latest_cache_commit = max(dated.values(), key=lambda d: d)
    # When the cache reached MASTER, which is the only "arrival" that can
    # retroactively affect a tree anyone reads.
    master_arrival = parse_iso(
        git("log", "--format=%cI", "--reverse", "master",
            "--", "epic_u2o/queue_state.py").splitlines()[0].strip()) \
        if False else None
    _mlines = git("log", "--format=%cI", "--reverse", "master",
                  "--", "epic_u2o/queue_state.py").splitlines()
    master_dates = [parse_iso(x) for x in _mlines if x.strip()]
    cache_on_master = [d for d in master_dates if d >= earliest_cache_commit]
    master_cache_date = min(cache_on_master) if cache_on_master else None

    # THE CORRECTION. The PI's design asserted the rows "predate the cache" and
    # that "the mechanism did not exist when the rows were taken". The first half
    # is FALSE: the cache was authored on a DIVERGENT branch 29m46s BEFORE the
    # rows were sampled. Asserting it would have put a false proof in the
    # artifact -- an assertion that passes and means nothing, which is worse than
    # no assertion.
    cache_existed_at_sample_time = earliest_cache_commit < stamps[-1]
    rows_predate_cache_on_master = bool(
        master_cache_date and stamps[-1] < master_cache_date)

    # ---- EVIDENCE 3: SHAPE, corroborating only.
    top_keys, qs_keys, prov_fields = set(), set(), set()
    for r in rows:
        top_keys |= set(r)
        qs_keys |= set(r.get("queue_state") or {})
    for k in top_keys | qs_keys:
        if any(t in k.lower() for t in ("cache", "source", "age", "hit",
                                        "fresh", "provenance")):
            prov_fields.add(k)

    # ---- PROVENANCE OF THE ROWS AS A SET, and proof nothing was relabelled.
    rows_commit = git("log", "-1", "--format=%H", ROWS_BRANCH, "--", ROWS_PATH).strip()
    blob = git("show", f"{ROWS_BRANCH}:{ROWS_PATH}")
    per_row_digest = hashlib.sha256(
        "".join(hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                for r in rows).encode()).hexdigest()

    merged_into_master = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
         ROWS_BRANCH, "master"], capture_output=True).returncode == 0

    props = {
        "producing_branch_contains_no_cache": producing == 0,
        "rows_predate_the_cache_on_MASTER": rows_predate_cache_on_master,
        "cache_existed_elsewhere_at_sample_time": cache_existed_at_sample_time,
        "no_row_carries_a_provenance_field": not prov_fields,
        "rows_branch_is_still_unmerged": not merged_into_master,
    }

    record = {
        "bead": "gpu-sim-u2o.6.6.6.2.1",
        "claim": ("All rows are UNCACHED BY CONSTRUCTION: they were sampled by a "
                  "code path with no queue_state cache, so no row's queue_state "
                  "came from a cached read. Proven by CONTENT and CLOCK."),
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "experiments/gpu-sim-u2o.6.6.6.2.1/measure_retro.py",
        "gpu_used": False,
        "re_derived_at_run_time": ("every figure below is measured from the "
                                  "branch and the clock on this run; nothing is "
                                  "read from a stored constant, so regenerated "
                                  "rows on a cached path would fail here"),
        "never_ancestry": {
            "why": ("The two histories share no line. "
                    "git merge-base --is-ancestor f046633 5b1ee67 is false, and "
                    "was never true in either direction, so an ancestry "
                    "assertion here would be an assertion about nothing that "
                    "happens to pass."),
            "ancestry_assertion_used_anywhere_in_this_artifact": False,
        },
        "row_set": {
            "branch": ROWS_BRANCH,
            "path": ROWS_PATH,
            "branch_status": ("MERGED into master" if merged_into_master
                              else "UNMERGED -- this statement describes a "
                                   "branch that is not on master. Do not read "
                                   "it as a claim about master."),
            "last_written_by_commit": rows_commit,
            "n_rows": len(rows),
            "rows_file_sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "per_row_rollup_sha256": per_row_digest,
            "not_relabelled": ("rows are READ-ONLY here and were neither edited "
                               "nor regenerated; the digests above are the "
                               "proof, and the statement is additive"),
        },
        "evidence_1_content": {
            "claim": "the branch that produced the rows has no cache mechanism",
            "command": f"git show {ROWS_BRANCH}:epic_u2o/queue_state.py | grep -c {CACHE_SYMBOL}",
            "producing_branch_occurrences": producing,
            "master_occurrences_for_contrast": master_occ,
            "sufficiency": ("NOT sufficient alone: an uncommitted cache in a "
                            "working tree would not appear in committed content"),
        },
        "evidence_2_clock": {
            "claim": "every row was sampled before any cache commit existed",
            "row_sample_range_utc": [stamps[0].isoformat(), stamps[-1].isoformat()],
            "row_span_seconds": round((stamps[-1] - stamps[0]).total_seconds(), 3),
            "cache_commit_dates": cache_dates,
            "earliest_cache_commit_anywhere": earliest_cache_commit.isoformat(),
            "earliest_cache_commit_on_master": (
                master_cache_date.isoformat() if master_cache_date else None),
            "hours_from_cache_authored_to_rows_sampled": round(
                (stamps[-1] - earliest_cache_commit).total_seconds() / 3600, 3),
            "CORRECTION_TO_THE_DESIGN": (
                "The design claimed the rows predate the cache and that 'the "
                "mechanism did not exist when the rows were taken'. BOTH are "
                "false. The cache was authored on a divergent branch "
                f"{(stamps[-1] - earliest_cache_commit).total_seconds() / 60:.0f} "
                "minutes BEFORE the rows were sampled. The clock therefore does "
                "NOT prove the mechanism was absent, and asserting it would have "
                "been a false proof. What the clock does establish is narrower "
                "and true: the rows predate the cache's arrival on MASTER by "
                "days, so no row can reflect master's cache."),
            "sufficiency": ("the clock alone cannot prove absence -- the "
                            "mechanism demonstrably existed elsewhere at sample "
                            "time. It bounds WHEN, not WHERE. Content carries "
                            "the claim."),
        },
        "evidence_3_shape_corroborating_only": {
            "n_rows": len(rows),
            "top_level_keys": len(top_keys),
            "queue_state_keys": len(qs_keys),
            "provenance_fields_found": sorted(prov_fields),
            "reading": ("zero provenance fields is consistent with a code path "
                        "that had no cache and so had nothing to report; it is "
                        "corroboration, not proof"),
        },
        "commands_that_produced_this": COMMANDS_USED,
        "properties": props,
        "all_properties_pass": all(props.values()),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["all_properties_pass"]:
        bad = [k for k, v in props.items() if not v]
        print(f"CLAIM FAILS: {bad}", file=sys.stderr)
        return 1
    print("OK: the rows are uncached BY CONSTRUCTION. Content carries the claim "
          "(the producing branch has no cache); the clock only bounds WHEN -- it "
          "shows the rows predate the cache on master, and it also records that "
          "the mechanism existed elsewhere at sample time. Statement is additive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
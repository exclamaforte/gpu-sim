#!/usr/bin/env python3
"""A dispatch list that cannot offer finished or blocked work as available.

gpu-sim-edu.1. Twice in one session `bd ready`'s top entries were work an
implementer could not do: gpu-sim-u2o.6.11.1 was MERGE READY and already
reviewed, and gpu-sim-u2o.6.6.1 was frozen awaiting a verdict. Both cost a
session to unpick, and the second nearly produced a rework cycle spent redoing
items that had already been cleared.

WHAT IS ACTUALLY BROKEN, MEASURED ON THIS DATABASE 2026-10-03
------------------------------------------------------------------
Only ONE of the two suspected classes is real, and saying so matters more than
having a fix for both:

  * VALIDATED-BUT-UNMERGED: REAL. `bd ready` lists six beads carrying
    `validated` / `validated-pending-merge` (gpu-sim-u2o.20, 7.7, 8.3, 19,
    6.6.5, 6.6.6). Those are reviewed-and-awaiting-merge, not work.

  * DEP-BLOCKED-BUT-OPEN: NOT REAL, and `bd` is already correct. `bd ready`
    uses blocker-aware semantics and honours `type=blocks` edges. Once the
    deps on gpu-sim-u2o.6.6.6.2/.3 were recorded as `blocks` they stopped
    appearing, with no code change here. A first probe of mine appeared to
    show eight leaks; that was my own bug -- the dependency sub-records carry
    no `status`, so every edge looked open, and `bd list` omits closed issues
    by default. Verified properly, the blockers were all `closed`.

So this tool does NOT reimplement bd's dependency logic. It verifies it (so a
regression is caught rather than assumed) and adds the label filter that bd
genuinely lacks. Reimplementing blocker detection here would be a second
source of truth for something bd already gets right, and would be wrong the
first time bd changed its semantics.

WHY NOT A bd CONFIG KEY
-----------------------
`bd config` has no ready-filter namespace, and `bd ready` exposes
--exclude-label but no persistent default. So the filter has to live in the
repo and be called by dispatch. Pointing the lab daemon at this script is the
one step outside this repo, and it is the PI's to make.

THE RULE
--------
Exclude a bead that carries a merge-pending label AND has no outstanding
verdict. The "AND no outstanding verdict" clause is not decoration: a bead
under review carries needs-review, and such a bead is NOT finished even if it
also carries `validated`. Merging the two classes would hide work in progress.

It also filters `needs-review` and `needs-rework` beads ONLY when a
merge-pending label is ALSO present. A needs-rework bead with no merge label is
genuinely outstanding work and stays visible.

This check CAN PASS. That is a design requirement, not an aspiration, and it
comes from gpu-sim-u2o.6.11.1: a check that only ever reports disagreement is
indistinguishable from one whose accounting is simply wrong. A filter that
dropped everything would satisfy every assertion here and be useless in
production. So the test suite asserts that ordinary ready work SURVIVES.

Usage:
    python3 scripts/bd_ready_filtered.py [--json] [--explain]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

# A bead carrying one of these has been reviewed; if it is not yet merged it is
# finished work awaiting a merge decision, not work an implementer can do.
MERGE_PENDING_LABELS = frozenset({"validated", "validated-pending-merge"})

# A bead carrying one of these has a verdict OUTSTANDING against it.
OUTSTANDING_VERDICT_LABELS = frozenset({"needs-review", "needs-rework"})

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def bd_json(args: list[str]) -> list[dict]:
    """Run bd and return its JSON list. Raises on a non-zero exit.

    A silent fallback to [] here would be the failure mode this whole project
    keeps meeting: a check that cannot tell reports nothing, and nothing reads
    as nothing wrong.
    """
    proc = subprocess.run(["bd", *args, "--json"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"bd {' '.join(args)} failed rc={proc.returncode}: "
                           f"{proc.stderr.strip()[:400]}")
    data = json.loads(proc.stdout)
    if isinstance(data, list):
        return data
    for key in ("issues", "beads", "items"):
        if isinstance(data.get(key), list):
            return data[key]
    raise RuntimeError(f"unrecognised bd JSON shape: {type(data)}")


class StatusMap(dict):
    """A dict of id -> status that also carries how it was built.

    A dict subclass rather than sentinel keys: `__closed_count__` stored as a
    key would be indistinguishable from an issue id to every caller, and
    open_blockers() looks ids up in this map. The counts are recorded because a
    reader needs to see that the CLOSED set was actually present -- a lookup
    that silently lost the closed set is the whole failure mode here.
    """

    closed_count = 0
    total_count = 0


def statuses_by_id(require_closed: bool = True) -> StatusMap:
    """Status of EVERY issue, closed INCLUDED, or raise.

    `bd list` omits closed issues by default, so a blocker that is closed
    resolves to nothing and looks open. That was the bug behind my first probe
    reporting eight phantom leaks, so the lookup asks for closed explicitly.

    IT FAILS LOUDLY, and that is the point of the rework. The previous version
    looped over three flag spellings and `continue`d past every exception, then
    returned whatever it had -- including `{}` if all three failed. An empty map
    is the most dangerous possible return here, because open_blockers() treats
    "not in the map" as "not closed", so a broken lookup makes every
    dependency look OPEN: blocked work is then silently filtered out, the
    dispatch looks clean, and the criterion-2 check reports ALREADY CORRECT
    from nothing at all. A check that cannot answer must not answer.

    require_closed verifies the property the caller depends on: that the
    CLOSED set is actually present in the returned map. Without that check a
    future bd that quietly changes what `list` returns would reintroduce the
    original phantom-leak bug silently.
    """
    # Two listings, MERGED, and that merge is the fix.
    #
    # There is no `--include-closed` flag on `bd list` (I assumed one; the call
    # errors), and the two real listings are each PARTIAL: the default returns
    # only non-closed, `--status closed` returns only closed. Taking either alone
    # leaves "absence" silently meaning "open", so open_blockers() would be right
    # by accident and wrong the moment a caller asks a different question about
    # the map. Merged, every issue carries its REAL status and absence means
    # "this id is not a bead at all".
    errors = []
    out = {}
    for flag in ([], ["--status", "closed"]):
        label = " ".join(flag) or "(default)"
        try:
            rows = bd_json(["list", *flag])
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        if not rows:
            errors.append(f"{label}: returned no rows")
            continue
        for b in rows:
            out[b["id"]] = b.get("status")
    if not out:
        raise RuntimeError(
            "statuses_by_id could not read ANY issue list, so no dependency "
            "state can be trusted. Refusing to produce a dispatch list from "
            "nothing. Attempts: " + " | ".join(errors))
    closed = sum(1 for v in out.values() if v == "closed")
    if require_closed and closed == 0:
        raise RuntimeError(
            f"statuses_by_id returned {len(out)} issues and NONE is closed. "
            "The closed set is missing again, which makes every closed blocker "
            "look open and silently filters blocked work out of the dispatch. "
            "Refusing rather than answering from a bad map.")
    out = StatusMap(out)
    out.closed_count = closed
    out.total_count = len(out)
    return out


def open_blockers(issue: dict, status: dict[str, str]) -> list[str]:
    """Blockers of this issue whose own status is not closed."""
    out = []
    for dep in issue.get("dependencies") or []:
        if not isinstance(dep, dict) or dep.get("type") != "blocks":
            continue
        bid = dep.get("depends_on_id")
        if bid and status.get(bid) != "closed":
            out.append(bid)
    return out


def is_finished_awaiting_merge(issue: dict) -> bool:
    labels = set(issue.get("labels") or [])
    if not labels & MERGE_PENDING_LABELS:
        return False
    # An outstanding verdict means review is still in flight, so the bead is
    # not finished even though it carries a merge-pending label.
    return not labels & OUTSTANDING_VERDICT_LABELS


def classify(ready: list[dict], status: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Split into (offerable, excluded_with_reasons)."""
    offer, dropped = [], []
    for issue in ready:
        blockers = open_blockers(issue, status)
        if blockers:
            dropped.append((issue, f"blocked by open {blockers}"))
        elif is_finished_awaiting_merge(issue):
            dropped.append((issue, "validated/validated-pending-merge with no "
                                    "outstanding verdict: finished, awaiting merge"))
        else:
            offer.append(issue)
    return offer, dropped


def order_key(issue: dict):
    """Priority first, then stable by creation.

    Explicitly NOT recency. Recency is what put a reviewed-and-merge-ready bead
    at the top of a dispatch in the first place; bd already sorts by priority,
    and this makes the tiebreak deterministic rather than accidental.
    """
    prio = PRIORITY_ORDER.get(str(issue.get("priority", "")).upper(), 9)
    return (prio, str(issue.get("created_at") or ""), str(issue.get("id") or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--explain", action="store_true",
                    help="also print what was filtered and why")
    args = ap.parse_args()

    ready = bd_json(["ready"])
    status = statuses_by_id()
    offer, dropped = classify(ready, status)
    offer.sort(key=order_key)

    if args.explain:
        print(f"bd ready returned {len(ready)}; offering {len(offer)}; "
              f"excluded {len(dropped)}")
        for issue, why in sorted(dropped, key=lambda d: order_key(d[0])):
            print(f"  EXCLUDED {issue['id']}  P{issue.get('priority')}  -- {why}")

    if args.json:
        print(json.dumps({"offered": offer,
                          "excluded": [{"id": i["id"], "reason": w}
                                       for i, w in dropped]}, indent=2))
    else:
        for issue in offer:
            labels = ",".join(issue.get("labels") or []) or "-"
            print(f"○ {issue['id']} P{issue.get('priority')} "
                  f"[{issue.get('issue_type')}] {issue.get('title')}")
            print(f"      labels: {labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
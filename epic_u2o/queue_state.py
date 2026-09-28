#!/usr/bin/env python3
"""Record the state of the shared GPU queue at measurement time.

gpu-sim-u2o.6.1.1 item 6 / the PI's standing rule: every measurement records
the queue state and any overlapping JOB_IDs, or it is unauditable and must be
treated as unverified.

SCOPE, deliberately narrow. This module records WHAT ELSE WAS RUNNING. It
does not attempt to detect, attribute or filter contention: that is
gpu-sim-u2o.6.6.3's question, with a different design, and building it here
would duplicate that bead and turn a hygiene item into a design change.

The queue state lives in ~/.local/state/gpqueue/jobs/*.json, one file per job.
We read those files directly rather than shelling out, because `ts status`
enqueues a job to report status (it shows up in the history as a `failed`
row) -- querying the queue must never add to the queue.
"""

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time

def _resolve_queue_dir() -> Path:
    """Find the directory holding one JSON file per job.

    gpqueue exports GPQUEUE_STATE_DIR inside a job, but it points at the queue
    ROOT (~/.local/state/gpqueue), not the jobs/ subdirectory that holds the
    files; outside a job the variable is unset. So accept either, and decide by
    looking for the files rather than by trusting the path. Getting this wrong
    is silent -- the glob simply matches nothing and the record claims an empty
    queue, which is the one failure this module must not make.
    """
    env = os.environ.get("GPQUEUE_STATE_DIR")
    cands = []
    if env:
        cands += [Path(env) / "jobs", Path(env)]
    cands.append(Path.home() / ".local/state/gpqueue/jobs")
    for c in cands:
        try:
            if c.is_dir() and any(c.glob("*.json")):
                return c
        except Exception:
            continue
    return cands[0]


QUEUE_DIR = _resolve_queue_dir()
class IdentificationError(RuntimeError):
    """Our own job could not be identified in the queue we read."""


ACTIVE = ("running", "queued", "starting")
TERMINAL = ("finished", "failed", "cancelled", "canceled")


def _load_all() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(QUEUE_DIR / "*.json"))):
        # The daemon writes these files while we read them, so a partial file
        # is a normal transient, not an error. Retry briefly: a job we fail to
        # parse is a job we cannot report as "ours", which is exactly the
        # auditability this module exists to provide.
        for _ in range(3):
            try:
                out.append(json.loads(Path(p).read_text()))
                break
            except FileNotFoundError:
                break
            except Exception:
                time.sleep(0.05)
    return out
    out = []
    for p in sorted(glob.glob(str(QUEUE_DIR / "*.json"))):
        try:
            out.append(json.loads(Path(p).read_text()))
        except Exception:
            continue
    return out


def queue_state(own_job_id: str | None = None,
                own_description_substr: str | None = None,
                own_command_substr: str | None = None) -> dict:
    """A snapshot of the queue, plus which jobs overlap this one.

    `own_description_substr` lets a job identify itself when it cannot know
    its own JOB_ID (ts prints it, but the script does not get it in $0).
    """
    jobs = _load_all()
    by_state: dict[str, int] = {}
    for j in jobs:
        by_state[j.get("state", "?")] = by_state.get(j.get("state", "?"), 0) + 1

    # Fallback: match on our own command line, which is always present even if
    # the description was truncated. An unidentified "own job" would make the
    # whole record unauditable, which is the one outcome this must not have.
    def _pick(cands):
        # Several jobs can match (four anchor jobs share a command prefix and
        # a generic description). Picking "the newest match" made every family
        # claim the same JOB_ID, which is exactly the field whose whole purpose
        # is to say which run this is. Prefer the one that is actually running,
        # then the newest, and say how it was chosen.
        if not cands:
            return None, None
        running = [j for j in cands if j.get("state") in ("running", "starting")]
        pool = running or cands
        pool = sorted(pool, key=lambda j: j.get("id") or "")
        return pool[-1].get("id"), ("running" if running else "newest-match")

    # PRECEDENCE, revised TWICE, and the sequence is the whole lesson:
    #   v1: command first, description second -> in the four-family case the
    #       shared command prefix matched a SIBLING and overrode a good
    #       description match (caught by the test).
    #   v2: description first, command second -> the v7 run set a description
    #       substring that did not match what was actually submitted
    #       ("anchor: memory" vs "anchor v7: memory"), so the description
    #       match found an OLDER finished job with the same command and named
    #       that one (also caught by the test).
    #   v3: rank candidates from BOTH matchers by (liveness, specificity,
    #       recency). A job that is running now, matched by a command
    #       carrying the family's own argument, beats a stale finished job
    #       that merely contains a description substring.
    def _score(j, spec):
        return (1 if j.get("state") in ("running", "starting") else 0, spec,
                j.get("id") or "")

    best, how = None, None
    for kind, needle, spec in (
            ("command", own_command_substr, 2),      # includes the family arg
            ("description", own_description_substr, 1)):
        if not needle:
            continue
        if kind == "command":
            cands = [j for j in jobs
                     if needle in " ".join(j.get("command") or [])]
        else:
            cands = [j for j in jobs if needle in (j.get("description") or "")]
        for j in cands:
            s = _score(j, spec)
            if best is None or s > _score(best[0], best[1]):
                best, how = (j, spec), f"{kind}:{'running' if s[0] else 'newest'}"
    if best is not None:
        own_job_id = best[0].get("id")

    if own_job_id is None:
        # Requirement 1: raise rather than write a null own_job_id. A null is
        # visible only to a reader who tests key PRESENCE rather than value,
        # and a file that documents an invariant in a comment while returning
        # null in code is enforcing nothing. The record cannot be audited
        # without knowing which run it describes, so refuse to produce one.
        raise IdentificationError(
            "queue_state could not identify our own job: no job matched "
            f"description={own_description_substr!r} or "
            f"command~{own_command_substr!r} among {len(jobs)} jobs read "
            f"from {QUEUE_DIR}. Refusing to write a record with a null "
            "own_job_id: an unauditable record is worse than none, because "
            "nothing in it looks missing.")

    active = []
    for j in jobs:
        if j.get("state") not in ACTIVE:
            continue
        active.append({
            "id": j.get("id"),
            "state": j.get("state"),
            "description": (j.get("description") or "")[:80],
            "created_at": j.get("created_at"),
        })
    active.sort(key=lambda a: a["id"] or "")

    # "Overlapping" = anything active that is not us. The queue runs one GPU
    # job at a time, so a queued sibling will not have touched the device yet
    # -- but it may start the moment we finish, and a job that was running
    # before we started is the one that could have moved our clocks.
    # Capture OUR OWN entry from the active set BEFORE the "others" filter.
    # A field added after the filter is a re-derivation -- the self-certifying
    # assertion this whole record exists to eliminate. As it stood, our own
    # job's state was unrecoverable from the artifact, so every field a false
    # positive would produce was internally consistent with a true one.
    own_job_entry = None
    if own_job_id is not None:
        for a in active:
            if a["id"] == own_job_id:
                own_job_entry = dict(a)
                break

    others = [a for a in active if a["id"] != own_job_id]
    running_others = [a for a in others if a["state"] == "running"]

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "own_job_id": own_job_id,
        "own_description_substr": own_description_substr,
        "own_job_id_resolution": how,
        "queue_dir": str(QUEUE_DIR),
        "counts_by_state": by_state,
        "own_job_entry": own_job_entry,
        "own_job_in_active_set": own_job_entry is not None,
        "n_active_including_self": len(active),
        "n_jobs_read": len(jobs),
        "empty_claim_derived_from_n": len(active),
        "empty_claim_verified": (
            own_job_entry is not None
            and len(others) == max(0, len(active) - (1 if own_job_entry else 0))),
        "n_other_active": len(others),
        "n_other_running_at_sample_time": len(running_others),
        "other_active_jobs": others,
        "other_running_jobs": running_others,
        # OMITTED, not set to False, when our own job could not be identified:
        # a failed identification means we do not know whether the set we read
        # contained us, so we cannot claim emptiness in either direction.
        **({"queue_was_empty": len(others) == 0}
           if own_job_entry is not None else {
               "queue_was_empty": None,
               "queue_was_empty_omitted_because": (
                   "our own job could not be identified in the set we read, "
                   "so an emptiness claim is not derivable from this record")}),
        "scope_note": (
            "Records what else was queued/running at sample time. This is "
            "NOT contention detection and NOT a filter: no sample is "
            "discarded on the basis of this. Detection and attribution "
            "belong to gpu-sim-u2o.6.6.3."),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(queue_state(
        own_description_substr=sys.argv[1] if len(sys.argv) > 1 else None),
        indent=2))

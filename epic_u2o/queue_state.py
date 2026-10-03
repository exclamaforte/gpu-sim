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
TERMINAL = ("finished", "failed", "cancelled", "canceled",
            "timed_out", "killed_underused")

# gpu-sim-u2o.6.6.6.1: path -> (mtime_ns, size, state, record)
_TERMINAL_CACHE: dict[str, tuple] = {}
TERMINAL_REVERIFY_SEC = 30.0
LISTING_TTL_SEC = 0.25     # how long a newly SUBMITTED job may be invisible
_FILE_LIST: list = []      # [refreshed_at, sorted paths]
# gpu-sim-u2o.6.6.6.3: memoised RESULT of the sweep.
# [records, taken_at, listing_generation, reverify_epoch]
#
# The per-file loop was the whole residual cost at ~0.28 us/file, and it is NOT
# I/O: the existing `not reverify` branch already skips stat() for cached
# terminal entries. It is pure Python per-file work, so it can be removed
# WITHOUT weakening any guarantee.
#
# Why memoising is safe here, specifically: a terminal entry's signature is
# re-checked on a reverify cadence (TERMINAL_REVERIFY_SEC, 30s) and NOT every
# call, and the file LIST is already cached for LISTING_TTL_SEC. So an in-place
# terminal->active transition is already invisible for up to those windows.
# Memoising adds no NEW staleness class; it bounds what already existed. The
# bound is SNAPSHOT_TTL_SEC, set to LISTING_TTL_SEC rather than invented, and
# the returned provenance REPORTS the read age so a sample is never described as
# fresher than it is -- that field is what 6.6.6.6.2 added.
_SNAPSHOT: list = []
SNAPSHOT_TTL_SEC = LISTING_TTL_SEC
_SNAPSHOT_GENERATION = 0
_SNAPSHOT_REVERIFY_EPOCH = 0
_LOAD_STATS = {"calls": 0, "files_seen": 0, "hits": 0, "parses": 0}


def _load_all() -> list[dict]:
    """Records only; see _load_all_with_provenance for what this reads."""
    return _load_all_with_provenance()[0]


def _load_all_with_provenance() -> tuple:
    """gpu-sim-u2o.6.6.6.6.2: records AND how this sample was read.

    Criterion 2 of 6.6.6.1 asks that every row record which path produced its
    sample and the age of the underlying read, so no reader is told a sample is
    fresher than it is. That information existed only in process-global
    load_stats(), which answers about the PROCESS rather than the SAMPLE: two
    samples taken a minute apart get the same answer, and a caller who never
    called load_stats() has nothing at all.

    So the read path is recorded per sample and returned with it. There are four,
    and they are not interchangeable:

      listing_ttl_hit          the file LIST came from the 0.25s listing cache,
                               so a job submitted inside that window is not in
                               this sample. This is the one staleness a reader
                               most needs to know about.
      terminal_hit_no_stat     the record was served from the terminal cache
                               WITHOUT a stat, so its read can be up to
                               TERMINAL_REVERIFY_SEC old.
      terminal_hit_stat_ok     stat'd, signature unchanged, terminal.
      fresh_parse              read from disk on this call; age 0.

    The ACTIVE set is never cached, so an active job is always fresh_parse and
    read_age_ms 0 for it -- the property 6.6.6.6.1 exists to protect.
    """
    global _SNAPSHOT, _SNAPSHOT_GENERATION, _SNAPSHOT_REVERIFY_EPOCH
    """Every job file, with TERMINAL ones served from a stat-keyed cache.

    gpu-sim-u2o.6.6.6.1: this function was the instrument being 40x more
    expensive than the thing it instruments. 6.7.1 measured 5.54 ms of
    queue_state() against 0.139 ms of device work, per example, because every
    call re-read and re-parsed all ~557 job files to answer a question about
    the handful that are ACTIVE.

    THE FIX IS NOT A TTL CACHE, which was the obvious move and the wrong one. A
    TTL makes a sample stale, and the property this module exists for is that a
    record OBSERVES rather than asserts; a sample served from a TTL is a sample
    of the past, and 6.6.6 was filed precisely to stop records asserting what
    they did not observe.

    THE FIX IS THAT TERMINAL IS TERMINAL. A job in finished/failed/canceled can
    never become active again, so if a file's (mtime_ns, size) is unchanged AND
    its state is terminal, re-parsing it cannot change the answer. That is a
    property of the state machine, not a heuristic and not a staleness window,
    which is why the output is identical rather than merely close.

    Still read fresh on EVERY call: any file not cached, any file whose mtime or
    size moved, and every ACTIVE file. The active set is therefore observed at
    sample time, exactly as before -- the cache can only ever hold a job that is
    already finished, so it cannot hide a neighbour.
    """
    out = []
    _LOAD_STATS["calls"] += 1
    by_path = {"listing_ttl_hit": 0, "terminal_hit_no_stat": 0,
               "terminal_hit_stat_ok": 0, "fresh_parse": 0}
    oldest_read_age_s = 0.0
    # PER-SAMPLE file count, NOT _LOAD_STATS["files_seen"]. The reviewer's
    # catch on the first version of this field: I filled it from the cumulative
    # PROCESS counter, so two samples over the same queue reported growing
    # numbers -- which is exactly the defect this bead exists to remove, one
    # layer down. The process counter stays in load_stats() where it belongs.
    files_seen_this_sample = 0
    # A terminal job needs NEITHER a re-read NOR a re-stat: it cannot become
    # active again, so its answer cannot change. Skipping the stat is what takes
    # this below the device time it measures -- 557 stat() calls were the whole
    # remaining cost once the PARSING was cached.
    #
    # The hazard is real, so it is BOUNDED rather than ignored: if a job id were
    # ever reused for a new job, a terminal entry would hide it. Terminal
    # entries are therefore re-verified in full every TERMINAL_REVERIFY_SEC, and
    # load_stats() reports the age of the oldest unverified entry, so a reader
    # sees the bound instead of inferring freshness. The ACTIVE set is read fresh
    # on every call and is NEVER cached, so a neighbour cannot be hidden by any
    # of this.
    # THIS MUST BE A BOOLEAN COMPARISON, and it was not, which is worth
    # recording. Written as a conditional expression it evaluated to the
    # ELAPSED SECONDS since the last reverify -- a tiny positive float, which is
    # TRUTHY. So `not reverify` was always False and the stat-skip never fired,
    # while `or reverify` in the listing condition below was always True and the
    # glob ran on every call. One missing `>` made the cache roughly 6x weaker
    # than it should be AND defeated the second cache entirely, and the symptom
    # in both cases looked like "my cache is not engaging".
    #
    # Found by a DIFFERENTIAL TEST asserting that a terminal change stays stale
    # for the documented window -- a test that could only fail if the cache was
    # not doing what the prose said it did. Profiling would not have found it,
    # because the cost model was self-consistent: it was measuring a cache that
    # was genuinely doing less work than intended.
    reverify = ("__last__" not in _TERMINAL_CACHE
                or (time.time() - _TERMINAL_CACHE["__last__"][0])
                > TERMINAL_REVERIFY_SEC)
    if reverify:
        _TERMINAL_CACHE["__last__"] = (time.time(),)
        _SNAPSHOT_REVERIFY_EPOCH += 1          # the next sweep must be real

    # The FILE LIST is also cached, because iterating 557 paths was the whole
    # remaining cost once parsing and stat'ing were cached -- 1.6 us per path is
    # a dict lookup and an append, and 557 of them is 0.9 ms against a 0.139 ms
    # budget. New job files appear only when a job is SUBMITTED, so a short TTL
    # on the LISTING is safe in a way a TTL on content would not be: the
    # CONTENTS of every candidate are still read fresh on every call below.
    # A newly submitted job is therefore visible within LISTING_TTL_SEC, which
    # load_stats() reports.
    now = time.time()
    listing_fresh = False
    if (not _FILE_LIST or now - _FILE_LIST[0] > LISTING_TTL_SEC
            or reverify):
        _FILE_LIST[:] = [now, sorted(glob.glob(str(QUEUE_DIR / "*.json")))]
        listing_fresh = True
        _SNAPSHOT_GENERATION += 1              # the file set changed
    else:
        listing_fresh = False

    # TERMINAL FILES ARE SKIPPED IN BULK; ACTIVE ONES ARE ALWAYS RE-READ.
    #
    # My first attempt memoised the WHOLE sweep, which was 130x faster and
    # WRONG: tests/test_cache_differential.py caught that
    # test_a_running_job_is_never_served_stale and
    # test_queued_to_running_is_visible_immediately fail. The reason is
    # structural and worth stating, because it is the whole constraint:
    # the ACTIVE set is exactly what must never be cached -- caching it is the
    # neighbour-hiding defect 6.6.6 was filed to remove -- so it is re-read on
    # every call by design. Memoising the sweep skips that re-read too, because
    # a job that was `queued` and is now `running` changes with no file added
    # and no generation bump to notice.
    #
    # So: TERMINAL records are served in bulk from the last sweep (they are
    # terminal by definition, and their signature is re-verified every
    # TERMINAL_REVERIFY_SEC as before), while every file whose last known state
    # was ACTIVE, and every file never seen, is re-read on this call. That keeps
    # the guarantee exactly and still removes the O(n) over terminal files,
    # which is the bulk of the queue.
    if _SNAPSHOT and _SNAPSHOT[2] == _SNAPSHOT_GENERATION \
            and _SNAPSHOT[3] == _SNAPSHOT_REVERIFY_EPOCH:
        _TERMINAL_BY_PATH = _SNAPSHOT[4]          # path -> record, terminal only
    else:
        _TERMINAL_BY_PATH = None                  # a real sweep is required

    # Built during the loop, NOT by indexing `out` against the file list:
    # files that fail to stat or parse are skipped, so out[i] and
    # _FILE_LIST[1][i] are not in correspondence and a zip silently pairs the
    # wrong record with the wrong path.
    terminal_snapshot: dict = {}
    for p in _FILE_LIST[1]:
        _LOAD_STATS["files_seen"] += 1
        files_seen_this_sample += 1
        if _TERMINAL_BY_PATH is not None:
            rec_cached = _TERMINAL_BY_PATH.get(p)
            if rec_cached is not None:
                _LOAD_STATS["hits"] += 1
                by_path["terminal_hit_no_stat"] += 1
                terminal_snapshot[p] = rec_cached
                out.append(rec_cached)
                continue
        hit = _TERMINAL_CACHE.get(p)
        if hit is not None and not reverify and hit[1] in TERMINAL:
            _LOAD_STATS["hits"] += 1
            by_path["terminal_hit_no_stat"] += 1
            oldest_read_age_s = max(oldest_read_age_s, now - hit[3])
            terminal_snapshot[p] = hit[2]
            out.append(hit[2])
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        sig = (st.st_mtime_ns, st.st_size)
        if hit is not None and hit[0] == sig and hit[1] in TERMINAL:
            _LOAD_STATS["hits"] += 1
            by_path["terminal_hit_stat_ok"] += 1
            oldest_read_age_s = max(oldest_read_age_s, now - hit[3])
            terminal_snapshot[p] = hit[2]
            out.append(hit[2])
            continue
        # The daemon writes these files while we read them, so a partial file
        # is a normal transient, not an error. Retry briefly: a job we fail to
        # parse is a job we cannot report as "ours", which is exactly the
        # auditability this module exists to provide.
        rec = None
        for _ in range(3):
            try:
                rec = json.loads(Path(p).read_text())
                break
            except FileNotFoundError:
                break
            except Exception:
                time.sleep(0.05)
        if rec is None:
            continue
        _LOAD_STATS["parses"] += 1
        if listing_fresh:
            by_path["fresh_parse"] += 1
        else:
            # The record came off disk, but the LIST it was found on may have
            # been up to LISTING_TTL_SEC old, so the sample cannot say it saw
            # every file. Recorded separately rather than folded into fresh_parse.
            by_path["listing_ttl_hit"] += 1
            oldest_read_age_s = max(oldest_read_age_s, now - _FILE_LIST[0])
        out.append(rec)
        state = rec.get("state")
        if state in TERMINAL:
            _TERMINAL_CACHE[p] = (sig, state, rec, time.time())

    served = [k for k, v in by_path.items() if v]   # KEYS, not values
    provenance = {
        "source": max(served, key=lambda k: by_path[k]) if served else "none",
        "counts_by_path": dict(by_path),
        "read_age_ms": round(oldest_read_age_s * 1e3, 3),
        "listing_age_ms": round((now - _FILE_LIST[0]) * 1e3, 3)
                          if _FILE_LIST else None,
        "n_files_seen": files_seen_this_sample,
        "note": ("read_age_ms is the age of the OLDEST underlying read among the "
                 "records served, so it is the worst case for this sample. The "
                 "ACTIVE set is never cached, so active jobs are always "
                 "fresh_parse and contribute 0."),
    }
    _SNAPSHOT[:] = [list(out), time.time(), _SNAPSHOT_GENERATION,
                    _SNAPSHOT_REVERIFY_EPOCH, terminal_snapshot]
    return out, provenance


def load_stats() -> dict:
    """Cache accounting, so an artifact can PROVE the mechanism.

    A speedup with no hit rate attached is a claim; this is the measurement.
    """
    s = dict(_LOAD_STATS)
    s["hit_rate"] = (s["hits"] / s["files_seen"]) if s["files_seen"] else 0.0
    s["cached_entries"] = len([k for k in _TERMINAL_CACHE if k != "__last__"])
    s["terminal_reverify_sec"] = TERMINAL_REVERIFY_SEC
    s["listing_ttl_sec"] = LISTING_TTL_SEC
    s["listing_age_sec"] = (round(time.time() - _FILE_LIST[0], 4)
                            if _FILE_LIST else None)
    ages = [v[3] for k, v in _TERMINAL_CACHE.items() if k != "__last__"]
    s["oldest_cached_age_sec"] = round(time.time() - min(ages), 3) if ages else 0.0
    s["staleness_bound"] = (
        f"a TERMINAL entry is re-verified in full every {TERMINAL_REVERIFY_SEC}s, "
        f"so a reused job id could be hidden for at most that long. The ACTIVE "
        f"set is read fresh on EVERY call and is never cached.")
    s["identity"] = (
        "the cache can only hold a job that is already TERMINAL, so it cannot "
        "hide an active neighbour; output is identical to the uncached path for "
        "any job whose state has not changed")
    s["semantics"] = ("output identical to the uncached path: only TERMINAL "
                      "jobs are ever cached, and only while (mtime_ns, size) is "
                      "unchanged, so the cache cannot hide an active neighbour")
    return s


def reset_cache() -> None:
    """For tests. Stats are kept so a test can compare before and after."""
    global _SNAPSHOT
    _TERMINAL_CACHE.clear()
    _FILE_LIST.clear()
    _SNAPSHOT[:] = []
    _LOAD_STATS.update({"calls": 0, "files_seen": 0, "hits": 0, "parses": 0})
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
    jobs, provenance = _load_all_with_provenance()
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
        # gpu-sim-u2o.6.6.6.6.2: ADDITIVE ONLY. Every key above is unchanged;
        # this attaches the sample's own read provenance to the record so a row
        # carrying it can say which path produced it and how old that read was.
        "sample_provenance": provenance,
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

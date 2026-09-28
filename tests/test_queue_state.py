#!/usr/bin/env python3
"""Regression tests for epic_u2o/queue_state.py (gpu-sim-u2o.6.6.6).

There was ZERO test coverage of this module, which is how two silent
false-negative/false-positive paths shipped:

  (a) GPQUEUE_STATE_DIR points at the queue ROOT inside a ts job, not the
      jobs/ subdirectory, so the glob matched nothing and the record said
      "queue empty" while other jobs were active. A false record that
      asserted absence -- the worst kind, because nothing in it looks broken.
  (b) four family jobs share a command prefix, so "newest match" returned the
      SAME job for every one of them: three of four files claimed the same
      JOB_ID. A false record that asserted identity.

Both are covered here, plus the two branches the PI named explicitly:
  (c) double-fallback-failure must RAISE, not write a null own_job_id
  (d) identified-but-absent-own-job must NOT assert queue_was_empty

Each test asserts on the emitted RECORD, not on internal helpers, because the
record is what a later reader sees. The queue is faked by pointing
GPQUEUE_STATE_DIR at a temporary directory of job files -- no real queue, no
GPU, no ts.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]   # tests/ -> worktree root
sys.path.insert(0, str(REPO))

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:150] if detail else ''}")
    if not ok:
        FAILS.append(name)


def fake_queue(jobs, subdir="jobs"):
    """Build a fake queue dir and return its path."""
    root = Path(tempfile.mkdtemp())
    d = root / subdir
    d.mkdir(parents=True)
    for j in jobs:
        (d / f"{j['id']}.json").write_text(json.dumps(j))
    return root


def load_module(env_state_dir):
    """Import queue_state fresh with GPQUEUE_STATE_DIR set."""
    os.environ["GPQUEUE_STATE_DIR"] = str(env_state_dir)
    for mod in ("epic_u2o.queue_state",):
        sys.modules.pop(mod, None)
    import importlib
    return importlib.import_module("epic_u2o.queue_state")


def job(jid, state, desc="", cmd=("bash", "run.sh")):
    return {"id": jid, "state": state, "description": desc,
            "command": list(cmd), "created_at": "2026-09-28T00:00:00+00:00"}


def main() -> int:
    # ---- (a) the directory-resolution regression ------------------------
    root = fake_queue([
        job("20260928t030010-AAAA", "running", "u2o.6.6.2 anchor: launch"),
        job("20260928t030010-BBBB", "queued", "cclm: something else"),
    ])
    # env var points at the ROOT, as gpqueue exports it inside a job
    qs = load_module(root)
    q = qs.queue_state(own_description_substr="u2o.6.6.2 anchor: launch")
    check("(a) resolves a queue ROOT to its jobs/ subdirectory",
          q.get("n_jobs_read") == 2, f"n_jobs_read={q.get('n_jobs_read')}")
    check("(a) does not report an empty queue when a job is active",
          q["queue_was_empty"] is False,
          f"queue_was_empty={q.get('queue_was_empty')}")
    check("(a) names the co-tenant it saw",
          [j["id"] for j in q["other_active_jobs"]] == ["20260928t030010-BBBB"],
          q["other_active_jobs"])

    # ---- (b) the shared-command-prefix regression -----------------------
    # Four families, one command prefix, four distinct descriptions.
    root = fake_queue([
        job("20260928t032717-AAAA", "finished", "u2o.6.6.2 anchor: launch"),
        job("20260928t032717-BBBB", "finished", "u2o.6.6.2 anchor: memory"),
        job("20260928t032717-CCCC", "running", "u2o.6.6.2 anchor: reduce"),
    ])
    qs = load_module(root)
    for fam, want in (("launch", "20260928t032717-AAAA"),
                      ("memory", "20260928t032717-BBBB"),
                      ("reduce", "20260928t032717-CCCC")):
        q = qs.queue_state(own_description_substr=f"u2o.6.6.2 anchor: {fam}",
                           own_command_substr="run.sh")
        check(f"(b) '{fam}' identifies ITS OWN job, not a shared-prefix sibling",
              q["own_job_id"] == want, f"got {q['own_job_id']}, want {want}")
    # ...and the command-only fallback must not silently pick a wrong sibling
    q = qs.queue_state(own_command_substr="run.sh")
    check("(b) command-only fallback prefers a RUNNING job over a finished one",
          q["own_job_id"] == "20260928t032717-CCCC",
          f"got {q['own_job_id']}, resolution={q['own_job_id_resolution']}")

    # ---- (c) identification failure must RAISE, not write null ----------
    root = fake_queue([job("20260928t999999-XXXX", "running", "someone else")])
    qs = load_module(root)
    try:
        qs.queue_state(own_description_substr="a job that does not exist",
                       own_command_substr="a-command-that-does-not-exist")
        # pre-fix code returns a record with own_job_id=None instead
        check("(c) identification failure RAISES rather than writing null",
              False, "no exception raised")
    except qs.IdentificationError as e:
        check("(c) identification failure RAISES rather than writing null", True,
              str(e)[:90])
    except Exception as e:
        check("(c) identification failure RAISES rather than writing null", False,
              f"wrong exception type: {type(e).__name__}")

    # ---- (d) identified-but-absent must not assert emptiness ------------
    # Our own job is in the ACTIVE set only if its state is active. A job that
    # is 'finished' (we just ran) is NOT in `active`, so the emptiness claim
    # cannot be derived from the active set -- and must be omitted, not set.
    root = fake_queue([
        job("20260928t040000-ZZZZ", "finished", "u2o.6.6.2 anchor: gemm"),
        job("20260928t040101-YYYY", "queued", "other tenant job"),
    ])
    qs = load_module(root)
    q = qs.queue_state(own_description_substr="u2o.6.6.2 anchor: gemm")
    check("(d) own job absent from the active set is reported as such",
          q["own_job_entry"] is None and q["own_job_in_active_set"] is False,
          f"own_job_entry={q['own_job_entry']}")
    check("(d) queue_was_empty is OMITTED (not asserted) when we cannot verify",
          q.get("queue_was_empty") is None
          and "queue_was_empty_omitted_because" in q,
          q.get("queue_was_empty_omitted_because", "")[:80])

    # ---- the normal path still carries the evidence --------------------
    root = fake_queue([
        job("20260928t050000-QQQQ", "running", "u2o.6.6.2 anchor: memory"),
        job("20260928t050101-RRRR", "queued", "conway: next"),
    ])
    qs = load_module(root)
    q = qs.queue_state(own_description_substr="u2o.6.6.2 anchor: memory")
    check("normal path: own_job_entry carries our id AND state",
          q["own_job_entry"] and q["own_job_entry"]["id"] == "20260928t050000-QQQQ"
          and q["own_job_entry"]["state"] == "running", q["own_job_entry"])
    check("normal path: emptiness claim is verified and derived from a count",
          q.get("empty_claim_verified") is True
          and q.get("empty_claim_derived_from_n") == q.get("n_active_including_self"),
          {k: q.get(k) for k in ("empty_claim_verified",
                             "empty_claim_derived_from_n",
                             "n_active_including_self", "n_other_active")})
    check("normal path: counts agree with the lists they summarise",
          q["n_other_active"] == len(q["other_active_jobs"])
          and q["n_other_running_at_sample_time"] == len(q["other_running_jobs"]),
          f"{q['n_other_active']} vs {len(q['other_active_jobs'])}")

    print()
    print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

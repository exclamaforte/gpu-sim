#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2.1: detect an artifact that is not the one the job wrote.

THE INCIDENT. On 2026-09-28 job 20260928t053729-5a28768b finished, exit 0, and
its log ended "wrote .../family_launch.json" with "18/18 configs measured". Twenty
seconds later something shaped like a git restore replaced that file with the
committed version. On disk sat a COHERENT but OLDER measurement, with nothing in
it to say it had been overwritten. The 18/18 launch measurements survived only
in a queue log.

That is the mixed-provenance shape this project was reopened over, arriving from
a different direction: not a fabricated number, but a real number deleted and
replaced by an older real one. Nothing downstream could tell.

WHAT THIS DOES. For each family_*.json it checks three things and says so out
loud:

  1. PROVENANCE EXISTS. An artifact written before this fix has no
     artifact_provenance block. That is reported, not treated as a pass --
     absence of evidence is not evidence of integrity.

  2. THE ON-DISK CONTENT MATCHES ITS OWN HASH. Recompute sha256 over the
     artifact EXCLUDING the provenance block and compare. A restore changes the
     content, so the hash stops matching. This is the check that catches the
     incident even when the restored file is perfectly well-formed.

  3. THE JOB ID MATCHS THE QUEUE'S OWN RECORD. The queue directory is the
     authority on what ran; the artifact is a claim. If the artifact says job
     X and the queue has no record of X finishing, that is a mismatch.

Reports. NEVER modifies an artifact -- a checker that repairs the thing it is
checking cannot be used to check it.

Usage: python3 experiments/gpu-sim-u2o.6.6.2/verify_artifact_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE_DIR = Path.home() / ".local/state/gpqueue/jobs"


def content_sha256(data: dict) -> str:
    """Hash the artifact EXCLUDING its own provenance block.

    The block records the hash, so it cannot be inside what it hashes.
    """
    payload = {k: v for k, v in data.items() if k != "artifact_provenance"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def queue_job_state(job_id: str | None):
    """(found, state) for a job id, read from the queue directory itself."""
    if not job_id:
        return False, None
    p = QUEUE_DIR / f"{job_id}.json"
    if not p.exists():
        return False, None
    try:
        return True, json.loads(p.read_text()).get("state")
    except Exception:
        return False, None


def check(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"artifact": path.name, "verdict": "UNREADABLE", "detail": str(exc)[:120]}

    prov = data.get("artifact_provenance")
    out = {"artifact": path.name}

    if not prov:
        out["verdict"] = "NO_PROVENANCE"
        out["detail"] = ("this artifact predates the provenance block, so "
                         "nothing on disk can say whether it is the file the "
                         "job wrote. Absence of evidence is not evidence of "
                         "integrity.")
        return out

    out["written_by_job"] = prov.get("written_by_job")
    out["written_at_utc"] = prov.get("written_at_utc")

    recomputed = content_sha256(data)
    out["recorded_sha256"] = prov.get("content_sha256")
    out["recomputed_sha256"] = recomputed
    hash_ok = prov.get("content_sha256") == recomputed
    out["content_matches_hash"] = hash_ok

    found, state = queue_job_state(prov.get("written_by_job"))
    out["job_found_in_queue"] = found
    out["job_state"] = state

    if not hash_ok:
        out["verdict"] = "CONTENT_ALTERED"
        out["detail"] = ("the on-disk content does not hash to what this job "
                         "recorded. Something replaced the file after it was "
                         "written -- a git restore, a checkout, or an "
                         "overwrite. The measurements this artifact appears to "
                         "hold are NOT the ones the job produced.")
    elif not found:
        out["verdict"] = "JOB_NOT_IN_QUEUE"
        out["detail"] = (f"the artifact claims job {prov.get('written_by_job')!r} "
                         f"wrote it, but the queue directory holds no record "
                         f"of that job.")
    elif state != "finished":
        out["verdict"] = "JOB_NOT_FINISHED"
        out["detail"] = f"the writing job is in state {state!r}, not 'finished'."
    else:
        out["verdict"] = "OK"
        out["detail"] = "content matches the job that wrote it, and that job finished."
    return out


def main() -> int:
    artifacts = sorted(HERE.glob("family_*.json"))
    if not artifacts:
        print("no family_*.json artifacts found", flush=True)
    results = [check(p) for p in artifacts]
    bad = [r for r in results if r["verdict"] != "OK"]
    report = {
        "bead": "gpu-sim-u2o.6.6.2.1",
        "kind": "ARTIFACT PROVENANCE CHECK -- reports, never modifies",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_checked": len(results),
        "results": results,
        "all_ok": not bad,
    }
    out = HERE / "provenance_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for r in results:
        print(f"{r['artifact']:<22} {r['verdict']:<20} {r.get('detail','')[:76]}")
    print(f"\nwrote {out.name}")
    if bad:
        print(f"{len(bad)} artifact(s) do not match their writing job",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
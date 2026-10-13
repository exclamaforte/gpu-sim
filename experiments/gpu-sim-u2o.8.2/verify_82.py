#!/usr/bin/env python3
"""gpu-sim-u2o.8.2 VERIFY (CPU): PTX/SASS corpus complete and paired.

Asserts every measured corpus row has exactly one PTX/SASS record (hash
join), every record compiled (ptx+sass), carries resource usage + digests,
and token counts are sane. Writes metrics.json. Does NOT recompile
(40-min extraction is a recorded run, verified here from its artifact).
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

FAIL = []
METRICS = {"bead": "gpu-sim-u2o.8.2", "stage": "verify"}


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


CORPUS = ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
          "experiments/gpu-sim-e5q/measured.jsonl",
          "experiments/gpu-sim-u2o.11.3/measured.jsonl",
          "experiments/gpu-sim-f6y/measured_new.jsonl",
          "experiments/gpu-sim-0vk/measured_new.jsonl")


def main() -> int:
    hashes = set()
    for f in CORPUS:
        for line in (REPO / f).read_text().splitlines():
            hashes.add(json.loads(line)["kernel_hash"])
    recs = [json.loads(l)
            for l in (HERE / "ptx_corpus.jsonl").read_text().splitlines()]
    got = [r["kernel_hash"] for r in recs]
    check("3261 records, unique, exact join to measured corpus",
          len(recs) == 3261 and len(set(got)) == 3261 and set(got) == hashes,
          f"{len(recs)} recs {len(set(got))} unique")
    check("ptx_ok on all rows", all(r.get("ptx_ok") for r in recs))
    check("sass_ok on all rows", all(r.get("sass_ok") for r in recs))
    check("resources on all rows",
          all(r.get("resources") for r in recs))
    check("digests on all rows",
          all(r.get("ptx_digest") and r.get("sass_digest") for r in recs))
    pt = [r["ptx_tokens"] for r in recs]
    st = [r["sass_tokens"] for r in recs]
    check("token counts sane (all > 0)",
          all(t > 0 for t in pt) and all(t > 0 for t in st))
    check("feature errors none",
          not any(r.get("ptx_features_error") or r.get("resources_error")
                   for r in recs))
    regs = [r["resources"]["registers_per_thread"] for r in recs]
    METRICS.update({"n_rows": len(recs),
                    "ptx_tokens_med": statistics.median(pt),
                    "sass_tokens_med": statistics.median(st),
                    "regs_med": statistics.median(regs),
                    "regs_max": max(regs)})
    (HERE / "metrics.json").write_text(json.dumps(METRICS, indent=2))
    print(f"ptx med {METRICS['ptx_tokens_med']}, sass med "
          f"{METRICS['sass_tokens_med']}, regs med/max "
          f"{METRICS['regs_med']}/{METRICS['regs_max']}")
    print("VERIFY", "PASS" if not FAIL else f"FAIL {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())

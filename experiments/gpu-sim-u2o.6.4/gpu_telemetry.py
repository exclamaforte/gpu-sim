#!/usr/bin/env python3
"""gpu-sim-u2o.6.4 (READ-ONLY HALF): GPU telemetry around a measurement.

SCOPE, AND IT IS A HOLD, NOT A PREFERENCE. The mutating half of this bead --
`nvidia-smi -lgc` / `-pl` and persistence-mode setup -- requires root and changes
GLOBAL GPU state shared with cclm, opguard and chessvar, which are not this
lab's to perturb. That half is HELD PENDING A HUMAN DECISION and nothing here
touches it. This file only ever READS.

Item 3, and it is the rule that makes this file safe to run at all: read-only
telemetry may be sampled OUTSIDE the queue for DIAGNOSIS, and is NEVER a
measurement and NEVER citable as one. Every record this writes carries that
statement, so a telemetry sample cannot later be quoted as evidence about
kernel runtime.

Item 4 is the question this exists to answer FROM MEASUREMENT rather than
assertion: is THROTTLING a factor on this part, and are "throttling" and
"boost state" the same signal? The 2026-09-28 contention run saw throttle
reasons move 0 -> 4 while clocks.sm moved only 0.098 in relative spread. So the
two are recorded SEPARATELY and never conflated: `clocks_throttle_reasons.active`
is a bitmask of CAUSES, while clock frequency is a CONSEQUENCE that can sit at
its boost ceiling with no throttle bit set at all.

DELIBERATELY ABSENT: a "--around <command>" flag. It existed in the first draft
and was removed. It ran an arbitrary command beside the sampling, which invites
running GPU work OUTSIDE the queue -- precisely what the rules forbid on a
shared box, and what would contaminate another tenant's measurement. Sampling
around a real measurement means the measurement is IN THE QUEUE and records its
own telemetry either side (as gpu-sim-u2o.6.6.5 does); it does not mean firing
workloads from a diagnostic tool.

Usage:
    python3 experiments/gpu-sim-u2o.6.4/gpu_telemetry.py --samples 20 --interval 0.5
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "telemetry.json"

FIELDS = [
    "clocks.sm", "clocks.max.sm", "temperature.gpu", "power.draw",
    "enforced.power.limit", "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "persistence_mode",
]

# The distinction item 4 insists on. Recorded on every sample so the two can
# never be read as one signal.
DIAGNOSIS_NOT_MEASUREMENT = (
    "This is READ-ONLY DIAGNOSTIC telemetry sampled outside the queue. It is "
    "NOT a measurement and must never be cited as one: it says what the GPU was "
    "doing, not how long a kernel took. Per gpu-sim-u2o.6.4 item 3 and the "
    "2026-09-27 rule amendment.")


def sample() -> dict:
    """One read-only telemetry sample. Never mutates anything."""
    args = ["nvidia-smi",
            f"--query-gpu={','.join(FIELDS)}",
            "--format=csv,noheader,nounits"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {p.stderr.strip()[:200]}")
    line = p.stdout.strip().splitlines()[0]
    vals = [v.strip() for v in line.split(",")]
    rec = {"at": datetime.now(timezone.utc).isoformat()}
    for k, v in zip(FIELDS, vals):
        try:
            rec[k] = float(v)
        except ValueError:
            rec[k] = v
    return rec


def throttle_bits(rec: dict) -> int:
    """The active-throttle bitmask as an int, for comparison across samples."""
    v = rec.get("clocks_throttle_reasons.active", 0)
    if isinstance(v, int):
        return v
    try:
        return int(str(v), 16)
    except (TypeError, ValueError):
        return 0


def analyse(samples: list[dict]) -> dict:
    """Are throttling and boost-state the same signal? Answered, not assumed."""
    sm = [s["clocks.sm"] for s in samples if isinstance(s.get("clocks.sm"), (int, float))]
    mx = [s["clocks.max.sm"] for s in samples
          if isinstance(s.get("clocks.max.sm"), (int, float))]
    bits = [throttle_bits(s) for s in samples]
    temps = [s["temperature.gpu"] for s in samples
             if isinstance(s.get("temperature.gpu"), (int, float))]
    pw = [s["power.draw"] for s in samples
          if isinstance(s.get("power.draw"), (int, float))]

    def spread(xs):
        if len(xs) < 2 or not max(xs):
            return None
        return round((max(xs) - min(xs)) / max(xs), 6)

    n_throttled = sum(1 for b in bits if b)
    return {
        "samples": len(samples),
        "clocks_sm_spread_relative": spread(sm),
        "clocks_max_sm": max(mx) if mx else None,
        "throttle_active_samples": n_throttled,
        "throttle_ever_active": n_throttled > 0,
        "distinct_throttle_masks": sorted(set(bits)),
        "temperature_max_C": max(temps) if temps else None,
        "power_draw_max_W": max(pw) if pw else None,
        "persistence_mode": samples[-1].get("persistence_mode") if samples else None,
        "THROTTLING_VS_BOOST": {
            "verdict": ("NOT the same signal. A throttle bitmask is a set of "
                        "CAUSES; clock frequency is a CONSEQUENCE. The GPU can "
                        "sit at its boost ceiling with every throttle bit "
                        "clear, and conversely show a throttle bit while the "
                        "clock spread stays small."
                        if not n_throttled else
                        "Throttling WAS active during sampling. Treat any "
                        "timing taken alongside it as suspect until the "
                        "specific cause bit is identified."),
            "throttle_active_samples": n_throttled,
            "clock_spread_relative": spread(sm),
            "conflating_them_would_be": ("the 2026-09-28 error: throttle "
                                         "reasons 0 -> 4 alongside a clocks.sm "
                                         "relative spread of 0.098, which says "
                                         "nothing about whether timing was "
                                         "affected."),
        },
        "disclaimer": DIAGNOSIS_NOT_MEASUREMENT,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--interval", type=float, default=0.5)
    a = ap.parse_args()

    samples = []
    for _ in range(max(1, a.samples)):
        samples.append(sample())
        time.sleep(a.interval)

    rec = {
        "bead": "gpu-sim-u2o.6.4",
        "kind": "READ-ONLY DIAGNOSTIC TELEMETRY",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "hold_in_force": (
            "The MUTATING half of this bead is HELD PENDING A HUMAN DECISION. "
            "Nothing here runs nvidia-smi -lgc/-pl or sets persistence mode; "
            "that changes global GPU state shared with other tenants and is not "
            "this lab's to perturb."),
        "analysis": analyse(samples),
        "samples": samples,
        "disclaimer": DIAGNOSIS_NOT_MEASUREMENT,
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    an = rec["analysis"]
    print(f"samples={an['samples']}  throttle_active={an['throttle_active_samples']}  "
          f"clocks.sm spread={an['clocks_sm_spread_relative']}  "
          f"max={an['temperature_max_C']}C  power={an['power_draw_max_W']}W")
    print(f"\n{an['THROTTLING_VS_BOOST']['verdict']}")
    print(f"\nwrote {OUT.name}  ({DIAGNOSIS_NOT_MEASUREMENT.split(':')[0]}.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
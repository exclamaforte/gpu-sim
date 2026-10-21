#!/usr/bin/env python3
"""gpu-sim-4yw Leg 1: validate model's per-phase wire bytes vs measured table."""
import json, re, sys
from pathlib import Path

REPO = Path("/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-4yw")
sys.path.insert(0, str(REPO / "epic_cluster"))
from link_model import predict, _bw_factor

ARTIFACT = REPO / "experiments/gpu-sim-pkb/canonical_byte_table.md"
spec = {
  "version": 1,
  "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1073741824,
                   "mem_bw_gbs": 1000.0, "arch": "x", "source": "t"}},
  "nodes": [{"name": "n0", "devices": [{"id": 0, "model": "m"}]},
            {"name": "n1", "devices": [{"id": 0, "model": "m"}]}],
  "links": [{"a": {"node": "n0", "device": 0}, "b": {"node": "n1", "device": 0},
             "bw_gbps": 10.0, "latency_us": 5.0, "mtu": 4096, "loss_ppm": 0,
             "jitter_us": 0.0, "shared_group": None, "kind": "ethernet"}],
  "ranks": [{"rank": 0, "node": "n0", "device": 0},
            {"rank": 1, "node": "n1", "device": 0}],
}

# Parse the markdown table rows: | S | floats | isends | distinct bytes | phases | per-phase pieces | per-phase bytes | total | factor |
rows = []
for line in ARTIFACT.read_text().splitlines():
    m = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)", line)
    if m:
        S = int(m.group(1)); floats = int(m.group(2)); isends = int(m.group(3))
        distinct = int(m.group(4)); phases = int(m.group(5)); ppp = int(m.group(6))
        per_phase_bytes = int(m.group(7))
        rows.append({"S": S, "floats": floats, "isends": isends,
                     "distinct_bytes": distinct, "phases": phases,
                     "P_meas": ppp, "meas_per_phase_bytes": per_phase_bytes})

print(f"parsed {len(rows)} rows from {ARTIFACT}")
out_rows = []
for r in rows:
    S = r["S"]
    count = S // 4  # floats
    P = r["P_meas"]
    pred = predict(spec, "ring_allreduce", count, "float32", n_ranks=2,
                   pieces_per_phase=P)
    bf = _bw_factor("ring_allreduce", 2)
    model_per_rank_bytes = bf * S          # model's assumed wire bytes per rank
    model_per_phase_bytes = model_per_rank_bytes / 2.0  # 2 phases
    meas_per_phase_bytes = r["meas_per_phase_bytes"]
    meas_per_rank_bytes = meas_per_phase_bytes * 2.0
    rel_err = (model_per_phase_bytes - meas_per_phase_bytes) / meas_per_phase_bytes
    within = abs(rel_err) <= 0.10
    out_rows.append({
        "S_bytes": S,
        "floats": r["floats"],
        "measured_P": P,
        "measured_per_phase_bytes": meas_per_phase_bytes,
        "measured_total_bytes_per_rank": meas_per_rank_bytes,
        "model_per_phase_bytes": model_per_phase_bytes,
        "model_per_rank_bytes": model_per_rank_bytes,
        "model_duration_s": pred["duration_s"],
        "model_duration_us": pred["duration_s"] * 1e6,
        "model_regime": pred["regime"],
        "rel_err_bytes": rel_err,
        "within_10pct": within,
        "artifact": str(ARTIFACT),
    })
    print(f"S={S:>8} P={P} meas_ppb={meas_per_phase_bytes:>8} pred_ppb={model_per_phase_bytes:>8.0f} rel_err={rel_err:+.4f} {'WITHIN' if within else 'OUTSIDE'}")

# Measured wall-clock durations (noisy, secondary) from variance_measurement.md
measured_dur = {4096: 6830.0, 2097152: 6729.0}
for r in out_rows:
    if r["S_bytes"] in measured_dur:
        md = measured_dur[r["S_bytes"]]
        r["measured_median_duration_us"] = md
        r["duration_rel_err"] = (r["model_duration_us"] - md) / md
    else:
        r["measured_median_duration_us"] = None
        r["duration_rel_err"] = None

failures = [r for r in out_rows if not r["within_10pct"]]
verdict = "MODEL-REPRODUCES" if not failures else "MODEL-DOES-NOT-REPRODUCE"
summary = {
    "bead": "gpu-sim-4yw",
    "leg": 1,
    "tolerance": 0.10,
    "tolerance_basis": "10% relative on per-phase wire bytes",
    "verdict": verdict,
    "n_phases": len(out_rows),
    "n_failures": len(failures),
    "failing_phases_S_bytes": [r["S_bytes"] for r in failures],
    "rows": out_rows,
    "measured_duration_source": "experiments/gpu-sim-pkb/variance_measurement.md (medians excluding init; 46 ms spread, unusable as gate)",
    "fabric_params_source": "experiments/gpu-sim-pkb/run.sh (single-link topo: bw=10 Gbps, lat=5 us, loss=0, jitter=0)",
    "pieces_source": "experiments/gpu-sim-pkb/canonical_byte_table.md",
}
Path("leg1_results.json").write_text(json.dumps(summary, indent=2))
print("\nVERDICT:", verdict, f"({len(failures)}/{len(out_rows)} phases outside tolerance)")

#!/usr/bin/env python3
"""gpu-sim-4yw Leg 2: DESCRIPTIVE sweep of fabric regime (model extrapolation)."""
import json, sys
from pathlib import Path
REPO = Path("/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-4yw")
sys.path.insert(0, str(REPO / "epic_cluster"))
from link_model import predict, _bw_factor

S = 2097152  # 2 MiB
N = 2
P = 8
count = S // 4
base_spec = {
  "version": 1,
  "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1073741824,
                   "mem_bw_gbs": 1000.0, "arch": "x", "source": "t"}},
  "nodes": [{"name": "n0", "devices": [{"id": 0, "model": "m"}]},
            {"name": "n1", "devices": [{"id": 0, "model": "m"}]}],
  "ranks": [{"rank": 0, "node": "n0", "device": 0},
            {"rank": 1, "node": "n1", "device": 0}],
}

rows = []
for bw in [25, 50, 100, 200, 400]:
    for lat in [2, 5, 10, 20]:
        for loss in [0, 100, 1000]:
            spec = dict(base_spec)
            spec["links"] = [{"a": {"node": "n0", "device": 0}, "b": {"node": "n1", "device": 0},
                              "bw_gbps": float(bw), "latency_us": float(lat), "mtu": 4096,
                              "loss_ppm": loss, "jitter_us": 0.0, "shared_group": None,
                              "kind": "ethernet"}]
            pred = predict(spec, "ring_allreduce", count, "float32", n_ranks=N,
                           pieces_per_phase=P)
            dur_us = pred["duration_s"] * 1e6
            per_phase_us = dur_us / 2.0
            # dominant term: compare bw_term, lat_term, loss extra
            B_Bps = bw * 1e9
            bw_term_us = (_bw_factor("ring_allreduce", N) * S / B_Bps) * 1e6
            lat_term_us = (2.0 * lat * 1e-6 * P) * 1e6  # 2 phases * alpha * P
            p_loss = loss / 1e6
            loss_factor = 1.0 / (1.0 - p_loss) if 0 < p_loss < 1 else 1.0
            loss_extra_us = bw_term_us * (loss_factor - 1.0)
            terms = {"bandwidth": bw_term_us, "latency": lat_term_us, "loss": loss_extra_us}
            dominant = max(terms, key=terms.get)
            rows.append({
                "bw_gbps": bw, "latency_us": lat, "loss_ppm": loss,
                "per_phase_cost_us": per_phase_us, "total_cost_us": dur_us,
                "bw_term_us": bw_term_us, "lat_term_us": lat_term_us,
                "loss_extra_us": loss_extra_us, "dominant_term": dominant,
                "regime": pred["regime"], "loss_factor": loss_factor,
            })

Path("leg2_results.json").write_text(json.dumps({"bead": "gpu-sim-4yw", "leg": 2,
    "label": "DESCRIPTIVE (model extrapolation; inherits Leg 1 failure)",
    "S_bytes": S, "n_ranks": N, "pieces_per_phase": P, "rows": rows}, indent=2))
# print compact table
print(f"{'bw':>5} {'lat':>4} {'loss':>5} {'per_phase_us':>13} {'dominant':>11}")
for r in rows:
    print(f"{r['bw_gbps']:>5} {r['latency_us']:>4} {r['loss_ppm']:>5} {r['per_phase_cost_us']:>13.1f} {r['dominant_term']:>11}")
print("rows:", len(rows))

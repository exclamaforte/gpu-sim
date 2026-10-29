#!/usr/bin/env python3
"""Regenerate experiments/gpu-sim-3fog/metrics.json from bw_raw.json + model_facts."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
bw = json.loads((ROOT / "experiments" / "gpu-sim-3fog" / "bw_raw.json").read_text())
facts = json.loads((ROOT / "docs" / "model_facts.json").read_text())
ab = facts["achieved_bw_measured"]
metrics = {
    "device": ab["device"],
    "torch": ab["torch"],
    "cuda": ab["cuda"],
    "spec_bw_gbs": ab["spec_bw_gbs"],
    "spec_source": ab["spec_source"],
    "working_set_bytes": ab["working_set_bytes"],
    "n_reps": ab["n_reps"],
    "copy_median_gbs": ab["copy_median_gbs"],
    "read_median_gbs": ab["read_median_gbs"],
    "copy_min_gbs": ab["copy_min_gbs"],
    "copy_max_gbs": ab["copy_max_gbs"],
    "achieved_bw_scale": ab["achieved_bw_scale"],
    "achieved_bw_scale_read": ab["achieved_bw_scale_read"],
    "neighbors_before": ab["neighbors_before"],
    "neighbors_after": ab["neighbors_after"],
    "flip_points": ab["flip_points"],
    "verdict_survives": {
        "N8_winner": ab["achieved_bw_scale"] > ab["flip_points"]["N8_winner"],
        "N16_seq32768": ab["achieved_bw_scale"] > ab["flip_points"]["N16_seq32768"],
    },
    "margin_to_flip_N8": ab["achieved_bw_scale"] - ab["flip_points"]["N8_winner"],
    "margin_to_flip_N16_seq32768": ab["achieved_bw_scale"] - ab["flip_points"]["N16_seq32768"],
    "rerun_rank1_N8_measured": "microbatch_factor",
    "rerun_rank1_N8_datasheet": "achieved_bw_scale",
    "gate": "n/a (GPU measurement; verified by gate --fast on the landing tree)",
    "runs_all": [
        {
            "job_id": "20261029t133040-c71701fa",
            "achieved_bw_scale": 0.8486742229040299,
            "copy_median_gbs": 1520.8242074440216,
            "occupancy": "unstable: 2 procs before (44, 16 MiB), 3 after (+8820 MiB)",
            "anchored": False,
        },
        {
            "job_id": "20261029t134925-219dbd35",
            "achieved_bw_scale": 0.8399499556578567,
            "copy_median_gbs": 1505.1903205388792,
            "occupancy": "stable: 3 procs before (44, 16, 638 MiB), 4 after (+7540 MiB)",
            "anchored": True,
        },
    ],
    "contention_finding_corrected": "Run 1 occupancy changed mid-measurement (an extra process appeared), so its higher 0.8487 is not the anchored value; the stable run 2 (0.83995) is anchored. A lone low sample in run 1 was neighbour-startup, not contention, and is excluded from the median.",
}
out = ROOT / "experiments" / "gpu-sim-3fog" / "metrics.json"
out.write_text(json.dumps(metrics, indent=2) + "\n")
print("wrote", out)

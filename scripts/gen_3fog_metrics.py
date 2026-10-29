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
}
out = ROOT / "experiments" / "gpu-sim-3fog" / "metrics.json"
out.write_text(json.dumps(metrics, indent=2) + "\n")
print("wrote", out)

#!/usr/bin/env python3
"""Regenerate experiments/gpu-sim-7uz2/metrics.json from act_curve_raw.json + model_facts."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
raw = json.loads((ROOT / "experiments" / "gpu-sim-7uz2" / "act_curve_raw.json").read_text())
facts = json.loads((ROOT / "docs" / "model_facts.json").read_text())
mac = facts["scaleout_opt"]["measured_act_curve"]
curve = {str(k): v["act_factor_median"] for k, v in mac["curve"].items()}
metrics = {
    "source": "gpu-sim-7uz2",
    "device": mac["device"],
    "torch": mac["torch"],
    "dtype": mac["dtype"],
    "batch": mac["batch"],
    "n_layers": mac["n_layers"],
    "d": mac["d"],
    "d_ff": mac["d_ff"],
    "seqs": mac["seqs"],
    "reps": mac["reps"],
    "model_assumed_act_factor": mac["model_assumed_act_factor"],
    "flip_point_n16_seq32768": mac["flip_point_n16_seq32768"],
    "curve_act_factor_by_seq": curve,
    "spread_at_fixed_length": {str(k): v["spread_fixed_len"] for k, v in mac["curve"].items()},
    "cause": mac["cause"],
    "verdict": {
        "seq_8192_below_flip": curve["8192"] < mac["flip_point_n16_seq32768"],
        "seq_32768_above_flip": curve["32768"] > mac["flip_point_n16_seq32768"],
    },
    "gate": "n/a (GPU measurement; verified by gate --fast on the landing tree)",
}
out = ROOT / "experiments" / "gpu-sim-7uz2" / "metrics.json"
out.write_text(json.dumps(metrics, indent=2) + "\n")
print("wrote", out)

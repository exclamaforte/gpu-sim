#!/usr/bin/env python3
"""Tests for the m3d.10 gate verdict script (experiments/gpu-sim-m3d.10/verdict.py).

The arbiter is a JUDGE, so it needs the same scrutiny the evidence has been
given all session: it must PASS a clean dataset, FAIL a dataset with a real
defect, and REFUSE (never guess) a dataset with missing inputs. These cases are
synthetic -- no GPU, no fabric, no NCCL -- so they belong in the no-GPU lane
that criterion 7 of the gate requires to be green.

A note on how this file was written: the script's first version compared the
model's SECONDS against the measurements' MICROSECONDS and reported a model
error of 99999900% on a clean dataset. The bug was found by RUNNING the script
against synthetic data, not by reading it -- which is the same discipline the
session applied to the artifacts, and the reason these cases are committed
rather than implied.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICT = os.path.join(HERE, "..", "experiments", "gpu-sim-m3d.10", "verdict.py")

S = 2 << 20
# Model values from the PI's independent re-admission at 2 MiB
# (experiments/gpu-sim-m3d.10/admissibility_pi_audit.json).
M = {
    "M1": dict(n=2, bw=900.0, lat=1.0, dur=41.3e-6, bwterm=37.3e-6, P=2, bytes=2 * S),
    "M2": dict(n=2, bw=100.0, lat=5.0, dur=355.5e-6, bwterm=335.5e-6, P=2, bytes=2 * S),
    "M3": dict(n=4, bw=50.0, lat=5.0, dur=563.3e-6, bwterm=503.3e-6, P=2, bytes=int(1.5 * S)),
    "M4": dict(n=4, bw=25.0, lat=10.0, dur=1126.6e-6, bwterm=1006.6e-6, P=2, bytes=int(1.5 * S)),
}


def make_metrics(const=5000.0, jitter=0.0, wrong_bytes=False, missing=None,
                 tracking_ok=True, per_piece=1.5, order=None, runs_per_config=3):
    d = {"schema": "m3d.10/gate-metrics/1", "payload_bytes": S,
         "admitted_order": order or ["M1", "M2", "M3", "M4"], "configs": {}}
    for k, v in M.items():
        if missing and k == missing:
            continue
        runs = []
        for i in range(runs_per_config):
            base = v["dur"] * 1e6 + const + (i - 1) * jitter
            runs.append({
                "run_id": f"{k}-r{i}",
                "iteration_durations_us": [base - 0.4, base, base + 0.5, base + 0.1, base - 0.2],
                "per_rank_bytes": {str(r): (v["bytes"] + 1 if (wrong_bytes and r == 0 and i == 0)
                                            else v["bytes"]) for r in range(v["n"])},
                "pieces_per_phase_measured": v["P"],
                "box_state": {"load_avg": 38.4, "runnable": 21, "processes": ["pi", "python3"]},
            })
        d["configs"][k] = {
            "n": v["n"], "bw_gbps": v["bw"], "latency_us": v["lat"],
            "model": {"duration_s": v["dur"], "pieces_per_phase": v["P"],
                      "bytes_per_rank": v["bytes"], "bw_term_s": v["bwterm"]},
            "runs": runs,
        }
    d["artifact"] = {"per_piece_us": per_piece, "per_piece_uncertainty_us": 0.3,
                     "source": "synthetic"}
    d["tracking"] = {"M2": {"measured_diff_us": 804.4 if tracking_ok else 300.0,
                            "model_diff_us": 838.9, "bw_gbps": [100, 25]}}
    return d


def run_verdict(data):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        r = subprocess.run([sys.executable, VERDICT, path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    return r


def verdict_of(r):
    return {0: "PASS", 1: "FAIL"}.get(r.returncode, "REFUSED")


def test_a_clean_dataset_passes():
    r = run_verdict(make_metrics())
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["verdict"] == "PASS"


def test_a_wrong_model_prediction_fails():
    d = make_metrics()
    d["configs"]["M2"]["model"]["duration_s"] = 500.0e-6
    r = run_verdict(d)
    assert r.returncode == 1
    assert "exceeds the pre-registered" in r.stdout


def test_a_byte_mismatch_fails_and_is_named_as_a_protocol_finding():
    r = run_verdict(make_metrics(wrong_bytes=True))
    assert r.returncode == 1
    assert "finding" in r.stdout and "protocol" in r.stdout


def test_a_failed_tracking_check_fails():
    r = run_verdict(make_metrics(tracking_ok=False))
    assert r.returncode == 1
    assert "NOT corroborated" in r.stdout


def test_an_artifact_over_the_allowance_fails_as_a_simulator_defect():
    r = run_verdict(make_metrics(per_piece=40.0))
    assert r.returncode == 1
    assert "SIMULATOR defect" in r.stdout


def test_a_noisy_pair_fails_the_resolvability_clause():
    # A large across-run spread (40 us of jitter per run) fails 4b's spread
    # clause: |d_pred| >= 10x the measured spread.
    r = run_verdict(make_metrics(jitter=40.0))
    assert r.returncode == 1
    assert "resolvability" in r.stdout


def test_a_pair_whose_measured_delta_is_inside_the_noise_is_unresolved():
    # Identical measured medians across configs: the delta cannot exceed the
    # spreads' sum, so the pair is UNRESOLVED -- which fails, and is reported as
    # a resolvability failure rather than as a wrong ordering.
    d = make_metrics()
    for k, cfg in d["configs"].items():
        for run in cfg["runs"]:
            run["iteration_durations_us"] = [5000.0, 5000.1, 5000.2, 5000.0, 5000.1]
    r = run_verdict(d)
    assert r.returncode == 1
    assert "UNRESOLVED" in r.stdout


def test_missing_inputs_are_refused_never_guessed():
    d = make_metrics(missing="M3")
    assert run_verdict(d).returncode == 2
    d = make_metrics()
    del d["artifact"]["per_piece_us"]
    assert run_verdict(d).returncode == 2
    d = make_metrics()
    d["configs"]["M2"]["runs"][0]["iteration_durations_us"] = []
    assert run_verdict(d).returncode == 2
    d = make_metrics(runs_per_config=2)
    assert run_verdict(d).returncode == 2


def test_an_admitted_order_that_disagrees_with_the_model_is_refused():
    r = run_verdict(make_metrics(order=["M4", "M3", "M2", "M1"]))
    assert r.returncode == 2
    assert "admitted_order" in r.stderr


def test_a_wrong_schema_is_refused():
    d = make_metrics()
    d["schema"] = "something/else"
    assert run_verdict(d).returncode == 2

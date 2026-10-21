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

# A committed kill-test log, so the arbiter's "the log must EXIST" check can
# be exercised in both directions.
_KILL_LOG = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
_KILL_LOG.write("peer killed mid-collective; bounded error; exit within the deadline\n")
_KILL_LOG.close()
KILL_LOG = _KILL_LOG.name

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
                 tracking_ok=True, per_piece=1.5, order=None, runs_per_config=3,
                 no_structural=False, dup_ids=False, no_plugin_line=False,
                 late_prereg=False, no_compute=False, kill_slow=False,
                 kill_log_missing=False, kill_unbounded=False, kill_log=None):
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
                "started_at": f"2026-08-22T05:0{i}:00Z",
                "box_state": {"load_avg": 38.4, "runnable": 21, "processes": ["pi", "python3"]},
            })
        cfg = {
            "n": v["n"], "bw_gbps": v["bw"], "latency_us": v["lat"],
            "model": {"duration_s": v["dur"], "pieces_per_phase": v["P"],
                      "bytes_per_rank": v["bytes"], "bw_term_s": v["bwterm"]},
            "runs": runs,
        }
        if not (no_structural and k == "M3"):
            ids = {str(r): r for r in range(v["n"])}
            if dup_ids and k == "M3":
                ids = {"0": 0, "1": 1, "2": 2, "3": 2}
            lines = ["NET/Plugin: Loaded net plugin sim", "Assigned NET plugin sim to comm",
                     "via NET/sim"]
            if no_plugin_line and k == "M2":
                lines = ["nothing here"]
            cfg["structural"] = {"device_count": v["n"], "world_size": v["n"],
                                 "rank_device_ids": ids, "plugin_log_lines": lines}
        d["configs"][k] = cfg
    d["artifact"] = {"per_piece_us": per_piece, "per_piece_uncertainty_us": 0.3,
                     "source": "synthetic"}
    d["tracking"] = {"M2": {"measured_diff_us": 804.4 if tracking_ok else 300.0,
                            "model_diff_us": 838.9, "bw_gbps": [100, 25]}}
    d["preregistration"] = {"timestamp": "2026-08-22T04:00:00Z", "commit": "abc123"}
    if late_prereg:
        d["preregistration"]["timestamp"] = "2026-08-22T06:00:00Z"
    d["compute"] = {"inherited_compute_error": "up to 3.9x on the absolute times",
                    "cancels_in_difference": True}
    if no_compute:
        del d["compute"]
    d["kill_test"] = {"peer_killed": True, "error_is_bounded": True,
                      "exit_within_s": 30.0 if kill_slow else 3.2,
                      "duration_s": 3.2, "log_path": kill_log or KILL_LOG}
    if kill_log_missing:
        d["kill_test"]["log_path"] = "/tmp/definitely-not-here-xyz"
    if kill_unbounded:
        d["kill_test"]["error_is_bounded"] = False
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


def test_missing_structural_counters_are_refused():
    # criterion 5: the structural counters are required per config.
    assert run_verdict(make_metrics(no_structural=True)).returncode == 2


def test_duplicate_device_ids_are_refused():
    # criterion 5: a duplicate rank-to-device id is the 'Duplicate GPU
    # detected' failure this project has already hit once.
    assert run_verdict(make_metrics(dup_ids=True)).returncode == 2


def test_a_missing_plugin_selection_line_is_refused():
    # the gate measures the plugin's path, so the artifact must show the plugin
    # carried it (the gpu-sim-pkb item-10 assertion).
    assert run_verdict(make_metrics(no_plugin_line=True)).returncode == 2


def test_a_run_before_the_preregistration_is_refused():
    # criterion 3: a protocol written after the runs is not pre-registered.
    assert run_verdict(make_metrics(late_prereg=True)).returncode == 2


def test_a_missing_compute_statement_is_refused():
    # criterion 8: the inherited compute error must be stated, and marked as
    # cancelling in the difference.
    assert run_verdict(make_metrics(no_compute=True)).returncode == 2


def test_a_kill_test_outside_the_bound_is_refused():
    # criterion 6: a killed peer must produce a bounded error and an exit
    # within 10 s.
    assert run_verdict(make_metrics(kill_slow=True)).returncode == 2
    assert run_verdict(make_metrics(kill_unbounded=True)).returncode == 2


def test_a_kill_test_log_that_does_not_exist_is_refused():
    # the log must be COMMITTED, not merely named.
    assert run_verdict(make_metrics(kill_log_missing=True)).returncode == 2


def test_a_sign_crossing_is_a_fail_not_a_tolerance_case():
    # criterion 2: a sign crossing between the measured and the predicted
    # difference is a FAIL. This case exists because an earlier version of the
    # arbiter compared two ABSOLUTE differences, whose product is always
    # positive, so a crossing could never be caught -- the reviewer's audit
    # found it.
    d = make_metrics()
    # make M1's measured median LOWER than M2's, while the model says the
    # other way round: the pair's sign crosses.
    for run in d["configs"]["M1"]["runs"]:
        run["iteration_durations_us"] = [100.0, 100.0, 100.0]
    for run in d["configs"]["M2"]["runs"]:
        run["iteration_durations_us"] = [9000.0, 9000.0, 9000.0]
    r = run_verdict(d)
    assert r.returncode == 1
    assert "sign not preserved" in r.stdout


def test_a_noisy_pair_elsewhere_cannot_inflate_the_artifact_allowance():
    # The allowance's noise term is the TIGHTEST pair's spread sum. An earlier
    # version used the max spread across all pairs, so a noisy pair elsewhere
    # in the set would license an artifact on the tight pair. This case is
    # built to DISCRIMINATE the two behaviours: the artifact's share (15% of
    # the tightest delta) sits ABOVE the allowance computed from the tight
    # pair's spread (10%) and BELOW the one computed from the noisiest pair's
    # spread (3 x 56 us = 168 us), so the tight-pair reading FAILS and the
    # max-spread reading PASSES. The noisiest pair stays resolvable (its
    # |d_pred| = 563 us exceeds 10 x its 56 us spread) so the verdict turns on
    # the allowance and not on the resolvability clause.
    d = make_metrics(per_piece=3.9)
    # The noise must be in a config OUTSIDE the tight pair (M2->M3), or it
    # would inflate the tight pair's own spread and the two readings would
    # coincide. M1 is in the M1->M2 pair only, and that pair stays resolvable
    # (|d_pred| 314 us >= 10 x its 28 us spread_sum), so the verdict turns on
    # the allowance and not on the resolvability clause.
    for i, run in enumerate(d["configs"]["M1"]["runs"]):
        med = 5041.3 - 14.0 + i * 14.0
        run["iteration_durations_us"] = [med - 0.4, med, med + 0.5, med + 0.1, med - 0.2]
    r = run_verdict(d)
    assert r.returncode == 1
    assert "SIMULATOR defect" in r.stdout

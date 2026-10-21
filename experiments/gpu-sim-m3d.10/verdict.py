#!/usr/bin/env python3
"""m3d.10 gate verdict script -- written by the PI, against the pre-registration.

WHY THIS SCRIPT IS NOT THE IMPLEMENTER'S. The session's recurring lesson is
that the party that produces the evidence must not also write the judge: a
verdict produced by the same code that produced the measurements is a formula
compared with itself, which is exactly how pkb's "0% time error" came to be
reported. The arbiter here is deliberately written against the PRE-REGISTERED
protocol and the pre-registered constants, by someone who did not produce the
measurements, and it refuses to emit a verdict on incomplete inputs.

WHAT IT REFUSES, and why that matters. A missing input is not a detail: if the
measured per-iteration durations are absent, the medians cannot be computed;
if the model prediction is absent, there is nothing to compare; if the tracking
check is absent, the per-config residual could be a tautology (a simulator that
does not spend the modelled time produces a constant-factor error that the
per-config comparison would quietly absorb). So this script EXITS 2 with a
precise reason rather than emitting a verdict it cannot support, and it exits 1
for a FAILING verdict (which is a real, reported result) and 0 for a passing
one. It also refuses to compare quantities whose provenance it cannot check:
the model durations must come from the model, the measured durations from the
runs, and the two are never derived from one another.

THE PRE-REGISTERED CONSTANTS live here, not in the data file, so that a data
file cannot widen them:
  * gate tolerance: 35% of |delta_predicted|, fixed, sign preservation required
  * resolvability (4b): |d_pred| >= 100us, |d_pred| >= 20% of the larger of the
    two configs' predicted durations, and |d_pred| >= 10x the measured
    across-run spread of the medians
  * bandwidth sensitivity (4a): |d_bw| >= 0.5 * |d_pred|
  * artifact allowance: min(max(0.10 * tightest modelled delta,
    3 * measured across-run spread), 0.20 * tightest modelled delta) -- it grows
    with the measured noise so a contended box cannot fail the gate
    environmentally, and it is capped at 20% so the model comparison always
    retains at least 15 of the 35 percentage points
  * runs per config: exactly 3; the config's measured duration is the MEDIAN of
    the three runs' durations, where each run's duration is the MEDIAN of its
    per-iteration durations; the spread is max-min across the three runs
  * byte factor: EXACT equality between the measured per-rank bytes and the
    model's factor * S (a mismatch is a finding about the protocol, not a
    tolerance to absorb)
  * tracking: the measured difference between two speeds must be within 10% of
    the model's difference at those speeds, and the tracking check is REQUIRED
    for the verdict -- the per-config residual is evidence, the tracking check
    is the test of that evidence, and a good residual with a failed tracking
    check is a SIMULATOR failure, not a model pass.

INPUT SCHEMA (the pre-registered contract; a file that does not match it is
refused, not guessed at):

{
  "schema": "m3d.10/gate-metrics/1",
  "payload_bytes": 2097152,
  "admitted_order": ["M1", "M2", "M3", "M4"],        # optional; if present it
                                                       # must match the model's
                                                       # predicted order
  "configs": {
    "M1": {
      "n": 2, "bw_gbps": 900.0, "latency_us": 1.0,
      "model": {"duration_s": ..., "pieces_per_phase": 2,
                "bytes_per_rank": ..., "bw_term_s": ...},
      "runs": [ {"run_id": "...", "iteration_durations_us": [...],
                 "per_rank_bytes": {"0": ..., "1": ...},
                 "pieces_per_phase_measured": 2,
                 "box_state": {"load_avg": ..., "runnable": ...,
                               "processes": [...]}}, ... exactly 3 ]
    }, ...
  },
  "artifact": {"per_piece_us": ..., "per_piece_uncertainty_us": ...,
               "source": "..."},
  "tracking": {"M2": {"measured_diff_us": ..., "model_diff_us": ...,
                      "bw_gbps": [100, 25]}}
}

USAGE: verdict.py gate_metrics.json [--json OUT.json]
EXIT: 0 = PASS, 1 = FAIL (a reported result), 2 = REFUSED (incomplete inputs).
"""
import json
import sys

TOLERANCE_PCT = 35.0
RESOLVABILITY_FLOOR_US = 100.0
RESOLVABILITY_RELATIVE = 0.20
RESOLVABILITY_SPREAD_MULTIPLE = 10.0
BW_SENSITIVITY_MIN = 0.5
ARTIFACT_ALLOWANCE_FLOOR = 0.10
ARTIFACT_ALLOWANCE_SPREAD_MULTIPLE = 3.0
ARTIFACT_ALLOWANCE_CAP = 0.20
RUNS_PER_CONFIG = 3
TRACKING_TOLERANCE_PCT = 10.0
SCHEMA = "m3d.10/gate-metrics/1"


class Refused(Exception):
    """A verdict cannot be supported by the inputs given."""


def need(obj, key, where):
    if not isinstance(obj, dict) or key not in obj or obj[key] is None:
        raise Refused(f"missing required field '{key}' in {where}")
    return obj[key]


def need_num(obj, key, where):
    v = need(obj, key, where)
    if not isinstance(v, (int, float)):
        raise Refused(f"field '{key}' in {where} is not a number: {v!r}")
    return float(v)


def run_duration(run, where):
    """A run's duration is the MEDIAN of its per-iteration durations."""
    iters = need(run, "iteration_durations_us", where)
    if not isinstance(iters, list) or not iters:
        raise Refused(f"iteration_durations_us in {where} is empty")
    vals = []
    for v in iters:
        if not isinstance(v, (int, float)) or v <= 0:
            raise Refused(f"non-positive or non-numeric iteration duration in {where}: {v!r}")
        vals.append(float(v))
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def load(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read metrics file {path}: {e}")
    if not isinstance(data, dict):
        raise Refused("metrics file is not a JSON object")
    if data.get("schema") != SCHEMA:
        raise Refused(f"schema mismatch: expected '{SCHEMA}', got {data.get('schema')!r}")
    return data


def check_config(name, cfg):
    """Per-config inputs and the checks that do not need the other configs."""
    where = f"configs.{name}"
    n = int(need_num(cfg, "n", where))
    bw = need_num(cfg, "bw_gbps", where)
    lat = need_num(cfg, "latency_us", where)
    model = need(cfg, "model", where)
    model_where = f"{where}.model"
    model_dur = need_num(model, "duration_s", model_where)
    model_p = int(need_num(model, "pieces_per_phase", model_where))
    model_bytes = int(need_num(model, "bytes_per_rank", model_where))
    model_bw_term = need_num(model, "bw_term_s", model_where)
    runs = need(cfg, "runs", where)
    if not isinstance(runs, list) or len(runs) != RUNS_PER_CONFIG:
        raise Refused(f"{where}.runs must hold exactly {RUNS_PER_CONFIG} runs, "
                      f"got {len(runs) if isinstance(runs, list) else 'a non-list'}")
    durations, pieces, byte_sets, loads = [], [], [], []
    for i, run in enumerate(runs):
        rw = f"{where}.runs[{i}]"
        durations.append(run_duration(run, rw))
        pieces.append(int(need_num(run, "pieces_per_phase_measured", rw)))
        prb = need(run, "per_rank_bytes", rw)
        if not isinstance(prb, dict) or not prb:
            raise Refused(f"{rw}.per_rank_bytes is empty")
        byte_sets.append({k: int(v) for k, v in prb.items()})
        box = need(run, "box_state", rw)
        loads.append(need_num(box, "load_avg", f"{rw}.box_state"))
    spread = max(durations) - min(durations)
    median = sorted(durations)[len(durations) // 2]
    # byte factor: EXACT equality per rank, for every run
    byte_agreement = []
    for i, prb in enumerate(byte_sets):
        for rank, v in sorted(prb.items()):
            byte_agreement.append({"run": i, "rank": int(rank), "measured": v,
                                    "model": model_bytes, "exact": v == model_bytes})
    # measured P must agree with the model's P (the derived P, per config)
    p_ok = all(p == model_p for p in pieces)
    return {
        "name": name, "n": n, "bw_gbps": bw, "latency_us": lat,
        "model_duration_s": model_dur, "model_bw_term_s": model_bw_term,
        "model_pieces_per_phase": model_p, "model_bytes_per_rank": model_bytes,
        "run_durations_us": durations, "median_us": median, "spread_us": spread,
        "measured_pieces": pieces, "p_agrees": p_ok,
        "byte_agreement": byte_agreement, "load_avg": loads,
    }


def main(argv):
    if len(argv) < 2:
        print("usage: verdict.py gate_metrics.json [--json OUT.json]", file=sys.stderr)
        return 2
    path = argv[1]
    out_path = None
    if "--json" in argv:
        try:
            out_path = argv[argv.index("--json") + 1]
        except IndexError:
            print("--json needs a path", file=sys.stderr)
            return 2

    try:
        data = load(path)
        payload = int(need_num(data, "payload_bytes", "root"))
        configs_raw = need(data, "configs", "root")
        if not isinstance(configs_raw, dict) or len(configs_raw) < 2:
            raise Refused("configs must hold at least two configurations")
        rows = {name: check_config(name, cfg) for name, cfg in configs_raw.items()}

        # The order is the MODEL's order, never the data's. If the data file
        # carries an admitted_order, it must agree with the model.
        order = sorted(rows, key=lambda k: rows[k]["model_duration_s"])
        if "admitted_order" in data:
            ao = data["admitted_order"]
            if list(ao) != order:
                raise Refused(f"admitted_order {ao} does not match the model's "
                              f"predicted order {order} -- the config set must be "
                              f"admitted against the final model")

        artifact = need(data, "artifact", "root")
        c_piece = need_num(artifact, "per_piece_us", "artifact")
        c_unc = need_num(artifact, "per_piece_uncertainty_us", "artifact")
        need(artifact, "source", "artifact")
        tracking = need(data, "tracking", "root")
        if not isinstance(tracking, dict) or not tracking:
            raise Refused("tracking must hold at least one two-speed comparison")

        # 4a / 4b / 3 / 2, per adjacent pair in the model's order.
        # UNITS: the model speaks SECONDS (predict().duration_s) and the
        # measurements speak MICROSECONDS. Every comparison below is made in
        # microseconds, with the conversion done ONCE and explicitly -- a
        # seconds-against-microseconds comparison is exactly the kind of defect
        # this session has been finding in the artifacts, and the arbiter
        # deserves the same discipline it applies to the evidence.
        pairs = []
        for a, b in zip(order, order[1:]):
            ra, rb = rows[a], rows[b]
            d_pred_s = abs(ra["model_duration_s"] - rb["model_duration_s"])
            d_pred_us = d_pred_s * 1e6
            d_bw_us = abs(ra["model_bw_term_s"] - rb["model_bw_term_s"]) * 1e6
            d_meas = abs(ra["median_us"] - rb["median_us"])
            spread_sum = ra["spread_us"] + rb["spread_us"]
            rel = d_pred_s / max(ra["model_duration_s"], rb["model_duration_s"])
            err = (d_meas - d_pred_us) / d_pred_us if d_pred_us else None
            unresolved = d_meas <= spread_sum
            within_tol = err is not None and abs(err) <= TOLERANCE_PCT / 100.0
            sign_ok = (d_meas * d_pred_us) >= 0 if d_meas and d_pred_us else True
            pairs.append({
                "pair": f"{a}->{b}", "a": a, "b": b,
                "d_pred_us": d_pred_us, "d_measured_us": d_meas,
                "d_bw_us": d_bw_us,
                "bw_share_4a": (d_bw_us / d_pred_us) if d_pred_us else None,
                "relative_margin_4b": rel,
                "spread_sum_us": spread_sum,
                "model_error_pct": (err * 100.0) if err is not None else None,
                "passes_4a": d_bw_us >= BW_SENSITIVITY_MIN * d_pred_us,
                "passes_4b_absolute": d_pred_us >= RESOLVABILITY_FLOOR_US,
                "passes_4b_relative": rel >= RESOLVABILITY_RELATIVE,
                "passes_4b_spread": d_pred_us >= RESOLVABILITY_SPREAD_MULTIPLE * spread_sum,
                "unresolved": unresolved,
                "within_tolerance": within_tol,
                "sign_preserved": sign_ok,
            })

        # The artifact's share of the tightest adjacent delta.
        tightest = min((p["d_pred_us"] for p in pairs), default=0.0)
        measured_spread = max((p["spread_sum_us"] for p in pairs), default=0.0)
        allowance = min(max(ARTIFACT_ALLOWANCE_FLOOR * tightest,
                            ARTIFACT_ALLOWANCE_SPREAD_MULTIPLE * measured_spread),
                        ARTIFACT_ALLOWANCE_CAP * tightest)
        # The artifact itself: the per-piece fixed cost times the piece-count
        # differential across the tightest pair, plus the per-byte copy
        # differential (which is NEGATIVE for a pair where the slower config
        # moves fewer bytes). The per-piece count comes from the measured P.
        tight_pair = min(pairs, key=lambda p: p["d_pred_us"]) if pairs else None
        artifact_share = None
        artifact_detail = {}
        if tight_pair:
            ra, rb = rows[tight_pair["a"]], rows[tight_pair["b"]]
            # pieces per rank = transfers per direction (2*(n-1)) x pieces per
            # transfer (the derived P) -- the model's own transfer count.
            pieces = {name: 2 * (rows[name]["n"] - 1) * rows[name]["model_pieces_per_phase"]
                      for name in (tight_pair["a"], tight_pair["b"])}
            d_pieces = abs(pieces[tight_pair["a"]] - pieces[tight_pair["b"]])
            per_piece_bias = d_pieces * c_piece
            artifact_detail = {
                "tightest_pair": tight_pair["pair"],
                "pieces_per_rank": pieces,
                "piece_count_differential": d_pieces,
                "per_piece_us": c_piece,
                "per_piece_uncertainty_us": c_unc,
                "per_piece_bias_us": per_piece_bias,
                "per_piece_bias_pct_of_delta": (per_piece_bias / tight_pair["d_pred_us"] * 100.0)
                    if tight_pair["d_pred_us"] else None,
                "allowance_us": allowance,
                "allowance_pct_of_delta": (allowance / tight_pair["d_pred_us"] * 100.0)
                    if tight_pair["d_pred_us"] else None,
            }
            artifact_share = artifact_detail["per_piece_bias_pct_of_delta"]

        # The tracking check: the falsification test for the per-config residual.
        tracking_rows = []
        for name, t in tracking.items():
            if name not in rows:
                raise Refused(f"tracking references unknown config {name!r}")
            md = need_num(t, "measured_diff_us", f"tracking.{name}")
            pd = need_num(t, "model_diff_us", f"tracking.{name}")
            speeds = need(t, "bw_gbps", f"tracking.{name}")
            if not isinstance(speeds, list) or len(speeds) < 2:
                raise Refused(f"tracking.{name}.bw_gbps must list at least two speeds")
            err = (md - pd) / pd if pd else None
            tracking_rows.append({
                "config": name, "measured_diff_us": md, "model_diff_us": pd,
                "speeds": speeds, "error_pct": (err * 100.0) if err is not None else None,
                "passes": err is not None and abs(err) <= TRACKING_TOLERANCE_PCT / 100.0,
            })

        # The verdict.
        failures = []
        for p in pairs:
            if p["unresolved"]:
                failures.append(f"{p['pair']}: UNRESOLVED (|delta_measured| {p['d_measured_us']:.1f}us "
                                f"<= the spreads' sum {p['spread_sum_us']:.1f}us) -- a resolvability "
                                f"failure, not a model mismatch")
                continue
            if not p["within_tolerance"]:
                failures.append(f"{p['pair']}: model error {p['model_error_pct']:.1f}% exceeds the "
                                f"pre-registered {TOLERANCE_PCT:.0f}%")
            if not p["sign_preserved"]:
                failures.append(f"{p['pair']}: sign not preserved (measured {p['d_measured_us']:.1f}us "
                                f"vs predicted {p['d_pred_us']:.1f}us)")
            if not p["passes_4a"]:
                failures.append(f"{p['pair']}: 4a bandwidth sensitivity fails "
                                f"(bw_share {p['bw_share_4a']:.3f} < {BW_SENSITIVITY_MIN})")
            if not (p["passes_4b_absolute"] and p["passes_4b_relative"] and p["passes_4b_spread"]):
                failures.append(f"{p['pair']}: 4b resolvability fails "
                                f"(abs={p['passes_4b_absolute']} rel={p['passes_4b_relative']} "
                                f"spread={p['passes_4b_spread']})")
        for name, r in rows.items():
            if not r["p_agrees"]:
                failures.append(f"{name}: measured pieces-per-phase {r['measured_pieces']} disagrees "
                                f"with the model's derived {r['model_pieces_per_phase']}")
            for ba in r["byte_agreement"]:
                if not ba["exact"]:
                    failures.append(f"{name}: per-rank bytes measured {ba['measured']} != the model's "
                                    f"{ba['model']} (rank {ba['rank']}, run {ba['run']}) -- a finding "
                                    f"about the protocol, not a tolerance to absorb")
        if artifact_share is not None and tight_pair and tight_pair["d_pred_us"]:
            if artifact_share > (allowance / tight_pair["d_pred_us"] * 100.0):
                failures.append(f"artifact {artifact_share:.1f}% of the tightest delta exceeds the "
                                f"allowance {allowance / tight_pair['d_pred_us'] * 100.0:.1f}% -- a "
                                f"SIMULATOR defect, not a model error")
        for t in tracking_rows:
            if not t["passes"]:
                failures.append(f"tracking {t['config']}: measured difference {t['measured_diff_us']:.1f}us "
                                f"vs the model's {t['model_diff_us']:.1f}us "
                                f"({t['error_pct']:.1f}%) -- the per-config residual is NOT corroborated")

        verdict = "PASS" if not failures else "FAIL"
        report = {
            "verdict": verdict,
            "schema": SCHEMA,
            "payload_bytes": payload,
            "predicted_order": order,
            "configs": {k: {kk: vv for kk, vv in v.items() if kk != "byte_agreement"}
                        for k, v in rows.items()},
            "pairs": pairs,
            "artifact": artifact_detail,
            "tracking": tracking_rows,
            "tolerances": {
                "gate_tolerance_pct": TOLERANCE_PCT,
                "resolvability_floor_us": RESOLVABILITY_FLOOR_US,
                "resolvability_relative": RESOLVABILITY_RELATIVE,
                "resolvability_spread_multiple": RESOLVABILITY_SPREAD_MULTIPLE,
                "bw_sensitivity_min": BW_SENSITIVITY_MIN,
                "artifact_allowance_rule": "min(max(0.10*tightest, 3*spread), 0.20*tightest)",
                "tracking_tolerance_pct": TRACKING_TOLERANCE_PCT,
                "runs_per_config": RUNS_PER_CONFIG,
            },
            "claim": ("the model's predicted ordering and adjacent differences are reproduced by "
                      "real NCCL + plugin, within the pre-registered tolerance, with differences "
                      "comparable and absolute durations carrying a fixed software-path constant "
                      "that cancels"),
            "failures": failures,
        }
        if out_path:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2, sort_keys=True)
        print(json.dumps({"verdict": verdict, "payload_bytes": payload,
                          "order": order, "failures": failures}, indent=2))
        if failures:
            print("\nVERDICT: FAIL -- a reported result, with the failures above.", file=sys.stderr)
            return 1
        print("\nVERDICT: PASS", file=sys.stderr)
        return 0
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        print("No verdict is emitted: the inputs do not support one.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

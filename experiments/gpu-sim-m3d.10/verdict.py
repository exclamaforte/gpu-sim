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
one.

TWO SHAPES OF INPUT, because the gate has two possible outcomes and they are NOT
the same statement. (a) A measured artifact (schema m3d.10/gate-metrics/1) goes
through the full comparison below. (b) A PRE-RUN REJECTION artifact (schema
m3d.10/gate-admissibility/1) records that no config set was admissible because
4b's spread clause cannot be met on this hardware; criterion 4b is a PRE-RUN
constraint, so the correct computed verdict there is NO-CLOSE-ADMISSIBILITY
(exit 1, a reported result) and NOT a refusal. Conflating the two would be a
false statement in either direction: a refusal says 'the inputs cannot support
any verdict', while NO-CLOSE-ADMISSIBILITY says 'the inputs support exactly this
one'. The admissibility path is guarded so that it can never be used to avoid a
run: it refuses (exit 2) if the recorded proxy spread would actually SATISFY 4b.
In both shapes the verdict is COMPUTED from numbers and the artifact's own
stated verdict is only ever CROSS-CHECKED against the computation -- never
echoed, because a judge that reads its verdict from the file it is judging is a
formula compared with itself. It also refuses to compare quantities whose provenance it cannot check:
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
                      "bw_gbps": [100, 25]}},
  "preregistration": {"timestamp": "2026-08-22T04:00:00Z", "commit": "abc123"},
  "compute": {"inherited_compute_error": "...", "cancels_in_difference": true},
  "kill_test": {"peer_killed": true, "error_is_bounded": true,
                "exit_within_s": 3.2, "duration_s": 3.2, "log_path": "..."}
}

STRUCTURAL COUNTERS (criterion 5) are required PER CONFIG, in the "structural"
block: {"device_count": N, "world_size": N, "rank_device_ids": {"0": 0, ...},
"plugin_log_lines": ["NET/Plugin: Loaded net plugin sim", ...]}. The device count
and the world size must equal the config's n, the rank-to-device ids must be
DISTINCT (a duplicate would be the 'Duplicate GPU detected' failure this project
has already hit once), and the plugin's log lines must contain the three the
plugin-selection assertion established: "NET/Plugin: Loaded net plugin",
"Assigned NET plugin" and "via NET/sim" -- the gate measures the plugin's
path, so the artifact must show the plugin carried it.

THE KILL TEST (criterion 6) is required: a peer killed mid-collective must
produce a BOUNDED error and an exit within 10 s, with the log committed (the
arbiver checks the log file EXISTS on disk, not merely that a path was named).
THE TIMESTAMP CLAUSE (criterion 3) is required: the pre-registration timestamp
must precede every run's start -- a gate whose protocol was written after the
runs is not a pre-registered gate, and that is checkable.
THE COMPUTE STATEMENT (criterion 8) is required: the inherited compute error
must be stated and must be marked as cancelling in the difference, because the
absolute durations carry it.

USAGE: verdict.py gate_metrics.json [--json OUT.json]
EXIT: 0 = PASS, 1 = FAIL (a reported result), 2 = REFUSED (incomplete inputs).
"""
import json
import os
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
ADMISSIBILITY_SCHEMA = "m3d.10/gate-admissibility/1"
# The proxy spread must be RE-DERIVED from the committed raw log, not trusted.
PROXY_RAW_LOG = "iters_2m.log"
PROXY_TIMER_ARTIFACT_US = 1.0e9   # an iteration longer than ~11 days cannot be real
PROXY_SPREAD_TOLERANCE = 0.01
KILL_TEST_MAX_EXIT_S = 10.0
KILL_TEST_ERROR_MARKER = "EPEERDEAD"
PLUGIN_LOG_REQUIRED = ("NET/Plugin: Loaded net plugin", "Assigned NET plugin",
                       "via NET/sim")


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
    if data.get("schema") not in (SCHEMA, ADMISSIBILITY_SCHEMA):
        raise Refused(f"schema mismatch: expected '{SCHEMA}' or "
                      f"'{ADMISSIBILITY_SCHEMA}', got {data.get('schema')!r}")
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
    started = []
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
        started.append(need(run, "started_at", rw))
    # criterion 5: the structural counters, per config.
    st = need(cfg, "structural", where)
    st_where = f"{where}.structural"
    device_count = int(need_num(st, "device_count", st_where))
    world_size = int(need_num(st, "world_size", st_where))
    rank_ids = need(st, "rank_device_ids", st_where)
    if not isinstance(rank_ids, dict) or not rank_ids:
        raise Refused(f"{st_where}.rank_device_ids is empty")
    rank_ids = {k: int(v) for k, v in rank_ids.items()}
    if device_count != world_size:
        raise Refused(f"{where}: device_count {device_count} != world_size {world_size}")
    if world_size != n:
        raise Refused(f"{where}: world_size {world_size} != the config's n {n}")
    if len(set(rank_ids.values())) != len(rank_ids):
        raise Refused(f"{st_where}.rank_device_ids has duplicate device ids {rank_ids} "
                      f"-- that is the 'Duplicate GPU detected' failure")
    lines = need(st, "plugin_log_lines", st_where)
    if not isinstance(lines, list) or not lines:
        raise Refused(f"{st_where}.plugin_log_lines is empty -- the gate measures the "
                      f"plugin's path, so the artifact must show the plugin carried it")
    joined = "\n".join(str(x) for x in lines)
    for needle in PLUGIN_LOG_REQUIRED:
        if needle not in joined:
            raise Refused(f"{st_where}.plugin_log_lines does not contain {needle!r} -- "
                          f"the plugin-selection assertion from gpu-sim-pkb must hold "
                          f"for every config")
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
        "started_at": started, "structural": {"device_count": device_count,
                                              "world_size": world_size,
                                              "rank_device_ids": rank_ids},
    }


def emit_admissibility(data, out_path):
    """The PRE-RUN path: no config set was admissible, so the gate never ran.

    This is a POSITIVE finding about the measurement environment, not a refusal.
    Criterion 4b is a pre-run constraint, so when its spread clause cannot be met
    the config set is rejected BEFORE any run, and the honest outcome is
    NO-CLOSE-ADMISSIBILITY rather than 'the inputs cannot support a verdict'.

    The verdict is computed from the artifact's own NUMBERS. The artifact's stated
    verdict is cross-checked against that computation and refused if they disagree
    -- it is never used AS the verdict, because a judge that reads its answer from
    the file it judges is a formula compared with itself.

    The guard that keeps this path honest: if the recorded proxy spread would
    actually SATISFY 4b, an admissible set may exist and the gate must RUN, so the
    script refuses instead of reporting NO-CLOSE. The path can therefore never be
    used to avoid a run, only to document that a run was impossible.
    """
    where = "admissibility"
    d_pred = need_num(data, "d_pred_M1M2_us", where)
    if d_pred <= 0:
        raise Refused("d_pred_M1M2_us must be positive")
    max_spread = need_num(data, "max_tolerable_spread_us", where)
    proxy = need_num(data, "noise_floor_proxy_us", where)

    # PROVENANCE OF THE PROXY, re-derived rather than trusted. This is the number
    # the whole judgement hangs on, and it is asserted by the same artifact the
    # judgement condemns -- so it is recomputed HERE from the committed raw
    # per-iteration log. A number the artifact asserts about itself is not
    # evidence; a number two parties compute from the same raw bytes is. The
    # derivation is fixed in this script: drop the first (init) iteration, drop
    # any value above PROXY_TIMER_ARTIFACT_US (a run cannot contain one), then
    # take max-min of what remains.
    log_rel = need(data, "ptr_raw_logs", where)
    if not isinstance(log_rel, str):
        raise Refused("ptr_raw_logs must be a path string")
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", log_rel, PROXY_RAW_LOG)
    if not os.path.exists(log_path):
        raise Refused(f"the proxy cannot be re-derived: {log_path} does not exist. "
                      f"A spread the artifact asserts about itself is not evidence")
    vals = []
    try:
        for line in open(log_path):
            line = line.strip()
            if line:
                vals.append(float(line))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read the raw proxy log {log_path}: {e}")
    if len(vals) < 3:
        raise Refused(f"the raw proxy log {log_path} holds {len(vals)} iterations; at "
                      f"least 3 are needed to derive a spread after dropping init")
    post_init = vals[1:]
    kept = [v for v in post_init if v <= PROXY_TIMER_ARTIFACT_US]
    dropped_artifacts = len(post_init) - len(kept)
    if len(kept) < 2:
        raise Refused(f"after dropping the init iteration and {dropped_artifacts} timer "
                      f"artifact(s), {len(kept)} iteration(s) remain -- too few for a spread")
    proxy_derived = max(kept) - min(kept)
    if abs(proxy_derived - proxy) / proxy > PROXY_SPREAD_TOLERANCE:
        raise Refused(f"noise_floor_proxy_us {proxy}us does not match the spread RE-DERIVED "
                      f"from {os.path.join(log_rel, PROXY_RAW_LOG)} ({proxy_derived}us over "
                      f"{len(kept)} post-init iterations, {dropped_artifacts} timer "
                      f"artifact(s) dropped) -- the proxy is not supported by the committed "
                      f"raw data")

    pc = need(data, "per_criterion", where)
    if not isinstance(pc, dict):
        raise Refused("per_criterion is not an object")
    status = pc.get("4b_10x_spread")
    if not isinstance(status, str) or not status.upper().startswith("UNMET"):
        raise Refused("per_criterion.4b_10x_spread is not recorded UNMET, so this "
                      "artifact is not a pre-run rejection and must not take the "
                      "admissibility path")

    refused = need(data, "rejected_alternatives", where)
    if not refused:
        raise Refused("rejected_alternatives is empty: the second ground -- the "
                      "unmeasured n=4 pieces-per-phase -- has no record to cite")

    negatives = need(data, "what_was_not_done", where)
    if not isinstance(negatives, list) or not negatives:
        raise Refused("what_was_not_done is empty: a negative artifact must state "
                      "its own negatives, and this path is unauditable without them")

    # The 4b relation must hold in the artifact's OWN numbers: the tolerable
    # spread is d_pred/10 and no wider. Numbers implying a wider bound have
    # widened the pre-registered clause, and that is refused, not accepted.
    implied = d_pred / RESOLVABILITY_SPREAD_MULTIPLE
    if abs(max_spread - implied) / implied > 0.01:
        raise Refused(f"max_tolerable_spread_us {max_spread} is not d_pred/10 "
                      f"({implied:.1f}) within 1% -- the pre-registered 4b clause "
                      f"has been widened or miscomputed in the artifact")

    # THE GUARD: a proxy spread that satisfies 4b means a set may be admissible.
    if proxy <= max_spread:
        raise Refused(f"proxy spread {proxy}us does not violate 4b's {max_spread}us "
                      f"bound, so a config set may be admissible and the gate must "
                      f"run; no measured rows are present, so no verdict is supported")

    ratio = proxy / max_spread
    verdict = "NO-CLOSE-ADMISSIBILITY"

    stated = data.get("gate_verdict")
    if stated is not None and not str(stated).startswith("NO-CLOSE"):
        raise Refused(f"the artifact states verdict {stated!r}, which its own numbers "
                      f"do not support (computed: {verdict})")

    grounds = [
        (f"G1: 4b's pre-run spread clause cannot be satisfied by any spread this box "
         f"delivers -- {max_spread:.1f}us required, proxy spread {proxy:.0f}us "
         f"({ratio:.0f}x larger), so the config set is REJECTED BEFORE THE RUN"),
        (f"G2: {len(refused)} config(s) refused for want of a MEASURED n=4 "
         f"pieces-per-phase; the model refuses rather than deriving P from the "
         f"fabric's cap, which undercounts the piece count by four at 2 MiB"),
    ]
    report = {
        "verdict": verdict,
        "schema": ADMISSIBILITY_SCHEMA,
        "gate_ran": False,
        "grounds": grounds,
        "arithmetic": {
            "d_pred_M1M2_us": d_pred,
            "max_tolerable_spread_us": max_spread,
            "implied_d_pred_over_10": round(implied, 3),
            "noise_floor_proxy_us": proxy,
            "proxy_spread_derived_us": proxy_derived,
            "proxy_iterations_kept": len(kept),
            "proxy_timer_artifacts_dropped": dropped_artifacts,
            "proxy_raw_log": os.path.join(log_rel, PROXY_RAW_LOG),
            "violation_factor": round(ratio, 1),
        },
        "what_was_not_done": negatives,
        "criteria_not_applicable": ["2", "3", "9a", "9b", "9c"],
        "claim": ("no config set is admissible on this hardware, so the gate does not "
                  "run: a finding about the measurement environment and an unmeasured "
                  "input, not about the model or the simulator"),
        "tolerances": {"resolvability_spread_multiple": RESOLVABILITY_SPREAD_MULTIPLE},
        "stated_verdict_in_artifact": stated,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({"verdict": verdict, "gate_ran": False,
                      "violation_factor": round(ratio, 1),
                      "grounds": grounds}, indent=2))
    print("\nVERDICT: NO-CLOSE-ADMISSIBILITY -- no config set is admissible here, so "
          "the gate was not run. A reported result, not a refusal.", file=sys.stderr)
    return 1


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
        if data.get("schema") == ADMISSIBILITY_SCHEMA:
            return emit_admissibility(data, out_path)
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

        # criterion 3's timestamp clause: the pre-registration must precede
        # every run. A protocol written after the runs is not pre-registered.
        prere = need(data, "preregistration", "root")
        prere_ts = need(prere, "timestamp", "preregistration")
        need(prere, "commit", "preregistration")
        for name, r in rows.items():
            for i, ts in enumerate(r["started_at"]):
                if str(ts) < str(prere_ts):
                    raise Refused(f"{name} run {i} started at {ts}, BEFORE the "
                                  f"pre-registration timestamp {prere_ts}")

        # criterion 8: the inherited compute error must be stated, and marked
        # as cancelling in the difference, because the absolute durations carry
        # it while the differences do not.
        compute = need(data, "compute", "root")
        need(compute, "inherited_compute_error", "compute")
        if compute.get("cancels_in_difference") is not True:
            raise Refused("compute.cancels_in_difference must be true -- the absolute "
                          "durations carry the inherited compute error and the "
                          "differences are what the gate compares")

        # criterion 6: the kill test must be present, bounded, and its log
        # committed (the arbiter checks the file exists, not merely that a path
        # was named).
        kill = need(data, "kill_test", "root")
        if kill.get("peer_killed") is not True:
            raise Refused("kill_test.peer_killed must be true -- the kill test is "
                          "part of the gate's definition of done")
        if kill.get("error_is_bounded") is not True:
            raise Refused("kill_test.error_is_bounded must be true -- a killed peer "
                          "must produce a BOUNDED error, not a hang")
        exit_s = need_num(kill, "exit_within_s", "kill_test")
        # STRICT: a run that exited at exactly the bound is a run that TIMED OUT
        # at the bound, which is the failure the kill test exists to catch --
        # so the comparison must exclude it. (An earlier version allowed
        # exit_within_s == the bound, which let a hang pass.)
        if not exit_s < KILL_TEST_MAX_EXIT_S:
            raise Refused(f"kill_test.exit_within_s {exit_s}s is not strictly below "
                          f"the {KILL_TEST_MAX_EXIT_S}s bound -- a run that exits AT "
                          f"the bound is a run that timed out at it")
        log_path = need(kill, "log_path", "kill_test")
        if not str(log_path) or not os.path.exists(str(log_path)):
            raise Refused(f"kill_test.log_path {log_path!r} does not exist -- the "
                          f"kill test's log must be COMMITTED, not merely named")
        # A log that merely EXISTS is not proof of a bounded exit: it must show
        # the bounded-error marker, and for a KILLED PEER that marker is the
        # specific error code the fabric's contract promises (EPEERDEAD), so
        # that a timeout is not passed off as a peer death.
        try:
            with open(log_path, errors="replace") as f:
                log_text = f.read()
        except OSError as e:
            raise Refused(f"kill_test.log_path {log_path!r} could not be read: {e}")
        if KILL_TEST_ERROR_MARKER not in log_text:
            raise Refused(f"kill_test.log_path {log_path!r} does not contain "
                          f"{KILL_TEST_ERROR_MARKER!r} -- a log that exists is not "
                          f"proof of a bounded exit, and for a killed peer the "
                          f"contract's specific error is what must be shown")

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
            # SIGNED differences: the sign check compares the SIGNED measured and
            # predicted differences. An earlier version compared two absolute
            # values, whose product is always positive, so a sign crossing --
            # which criterion 2 calls a FAIL and not a tolerance case -- could
            # never be caught. The signed values are kept in the report so a
            # reader can see the crossing rather than take the flag's word.
            d_pred_signed_us = (ra["model_duration_s"] - rb["model_duration_s"]) * 1e6
            d_meas_signed = ra["median_us"] - rb["median_us"]
            d_pred_s = abs(ra["model_duration_s"] - rb["model_duration_s"])
            d_pred_us = d_pred_s * 1e6
            d_bw_us = abs(ra["model_bw_term_s"] - rb["model_bw_term_s"]) * 1e6
            d_meas = abs(d_meas_signed)
            spread_sum = ra["spread_us"] + rb["spread_us"]
            rel = d_pred_s / max(ra["model_duration_s"], rb["model_duration_s"])
            err = (d_meas - d_pred_us) / d_pred_us if d_pred_us else None
            unresolved = d_meas <= spread_sum
            within_tol = err is not None and abs(err) <= TOLERANCE_PCT / 100.0
            sign_ok = (d_meas_signed * d_pred_signed_us) >= 0
            pairs.append({
                "pair": f"{a}->{b}", "a": a, "b": b,
                "d_pred_us": d_pred_us, "d_measured_us": d_meas,
                "d_pred_signed_us": d_pred_signed_us,
                "d_measured_signed_us": d_meas_signed,
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

        # The artifact's share of the TIGHTEST adjacent delta. The allowance's
        # noise term is the TIGHTEST PAIR's spread sum, NOT the noisiest pair's:
        # the allowance bounds what the artifact may consume of the delta being
        # checked, so borrowing a different pair's spread would inflate it --
        # a noisy pair elsewhere in the set would license an artifact on the
        # tight pair. (An earlier version used the max across pairs, which is
        # exactly the leniency this arbiter exists to avoid.)
        tight_pair = min(pairs, key=lambda p: p["d_pred_us"]) if pairs else None
        tightest = tight_pair["d_pred_us"] if tight_pair else 0.0
        tight_spread = tight_pair["spread_sum_us"] if tight_pair else 0.0
        allowance = min(max(ARTIFACT_ALLOWANCE_FLOOR * tightest,
                            ARTIFACT_ALLOWANCE_SPREAD_MULTIPLE * tight_spread),
                        ARTIFACT_ALLOWANCE_CAP * tightest)
        # The artifact itself: the per-piece fixed cost times the piece-count
        # differential across the tightest pair, plus the per-byte copy
        # differential (which is NEGATIVE for a pair where the slower config
        # moves fewer bytes). The per-piece count comes from the measured P.
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
                "allowance_spread_basis_us": tight_spread,
                "note": ("the allowance's noise term is the TIGHTEST PAIR's spread "
                         "sum, not the noisiest pair's; the artifact share counts the "
                         "per-piece fixed cost only, which OVERSTATES the true effect "
                         "because the copy's per-byte differential is negative for a "
                         "pair whose slower config moves fewer bytes -- a deliberate "
                         "conservatism"),
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
            "compute": compute,
            "kill_test": kill,
            "preregistration": prere,
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

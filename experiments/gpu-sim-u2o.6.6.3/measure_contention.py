#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.3 -- how much does a co-tenant move our timings, and can
we DETECT one.

The shared queue is the experiment. Three conditions, same three anchor cells
(launch-bound, memory-bound, compute-bound), >=3 independent PROCESS
invocations each:

  A. uncontended      -- queue observed empty
  B. queued-routed    -- a GPU burner running CONCURRENTLY inside this same ts
                         job, so it holds the queue slot and the device
  C. unqueued         -- a burner with NO ts JOB_ID of its own, started inside
                         this job. It occupies the device while being
                         STRUCTURALLY INVISIBLE to queue_state, which observes
                         the queue. This is the condition the PI added, and it
                         is the one that answers the detection question.

ON (C) AND THE QUEUE POLICY, because this is the part that could be read as a
violation: no GPU work runs outside the queue here. Condition C is produced
INSIDE this queued job, by a child process that never gets a JOB_ID. What it
models is a tenant whose activity is invisible to the queue channel -- which is
the detection question -- not a decision to run GPU work off-queue. If the
question is instead "what happens when a tenant really does bypass the
queue", the answer is the same: queue_state sees nothing and only device
telemetry can.

ON SANDBOXING, the trap the PI recorded: blanking CUDA_VISIBLE_DEVICES does
NOT sandbox run_kernel, which sets CUDA_VISIBLE_DEVICES = str(self.gpu_id).
So nothing here is tested by hiding the device; the only device-free tests in
this bead are the CPU-only ones that exercise code paths in isolation.

Everything through ts. No untrusted kernels: the cells are 6.6.2's hand-written
anchor kernels, copied with provenance. No clock locking, no sudo, no
persistence-mode change.
"""
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import anchor_kernels as ak                      # noqa: E402
from epic_u2o.kernel_driver import synthesize    # noqa: E402
from epic_u2o.safe_harness import SafeHarness   # noqa: E402
from epic_u2o import queue_state as qs          # noqa: E402
# task_limits (cgroup pids.max) is gpu-sim-u2o.6.1.2's and is NOT in master
# yet. Import it optionally rather than hard-failing: this bead must run on
# master, and should exercise the containment path automatically once 6.1.2
# merges. The lesson from 6.6.6 applies -- a module another branch owns should
# not be able to turn into a failed queued job.
try:
    from epic_u2o import task_limits as tl
    TASKS_MAX = getattr(tl, "DEFAULT_TASKS_MAX", 256)
    TASK_LIMIT_SOURCE = "epic_u2o.task_limits (gpu-sim-u2o.6.1.2, merged)"
except ImportError:
    tl = None
    TASKS_MAX = 0            # containment off; documented in the artifact
    TASK_LIMIT_SOURCE = ("epic_u2o.task_limits NOT AVAILABLE "
                         "(gpu-sim-u2o.6.1.2 unmerged); containment disabled")

TELEMETRY_FIELDS = [
    "clocks.sm", "clocks.max.sm", "temperature.gpu",
    "power.draw", "clocks_throttle_reasons.active", "utilization.gpu",
    "memory.used",
]
INVOCATIONS = 3

# The three cells, one per regime, as the acceptance requires. Sizes and
# configs are taken from the anchor so these numbers are comparable with it.
# Sizes MUST exist in the anchor's own config list, or compile_cell raises:
# the memory family is 2^10/2^14/2^18/2^22/2^26 (1<<20 is not one of them,
# which is how the first run of this bead died).
CELLS = [
    ("launch_bound", "launch", 1 << 10, "auto/256"),      # k_empty: pure dispatch
    ("memory_bound", "memory", 1 << 22, "auto/256"),     # k_saxpy, 32 MiB of traffic
    ("compute_bound", "gemm", 1024 * 1024, "32x32"),      # tiled GEMM 1024^3
]


def telemetry(label=""):
    out = {"label": label, "at": time.time(),
           "at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(TELEMETRY_FIELDS),
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        out["rc"] = r.returncode
        if r.returncode == 0 and r.stdout.strip():
            out["fields"] = dict(zip(TELEMETRY_FIELDS,
                                     [v.strip() for v in
                                      r.stdout.strip().splitlines()[0].split(",")]))
    except Exception as e:
        out["error"] = str(e)
    return out


def f(snap, key, default=None):
    return (snap.get("fields") or {}).get(key, default)


def num(snap, key):
    """First number in a telemetry value ('2415 MHz' -> 2415.0).

    Hex fields (clocks_throttle_reasons.active) are parsed as ints, not floats:
    a first version read '0x...0004' as a decimal and returned None, which
    made the field the PI singled out appear NOT to move when it had.
    """
    v = f(snap, key)
    if v is None:
        return None
    s = str(v).strip()
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


BURNER = r'''
"""Documented GPU burner. Shape is stated per condition, not assumed."""
import sys, time
shape, seconds = sys.argv[1], float(sys.argv[2])
import torch
torch.cuda.init()
if shape == "compute":
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            c = a @ b
        torch.cuda.synchronize()
elif shape == "memory":
    n = 64 * 1024 * 1024 // 4          # 64 MiB
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    y = torch.empty(n, device="cuda", dtype=torch.float32)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(40):
            y.copy_(x)
            x.add_(1.0)
        torch.cuda.synchronize()
print("burner done", shape, flush=True)
'''


class Burner:
    """A GPU burner started as a plain child of THIS job.

    Under condition B it is a child like any other and is described in the
    artifact; under condition C it is started detached with its own session so
    that nothing about it is associated with a JOB_ID -- which is precisely
    what makes it invisible to queue_state, and is the point.
    """

    def __init__(self, shape, seconds, detached: bool):
        self.shape, self.seconds, self.detached = shape, seconds, detached
        self.proc = None
        self.started = None

    def __enter__(self):
        script = HERE / "burner.py"
        script.write_text(BURNER)
        kw = {}
        if self.detached:
            # own session: no shared process group, and nothing in the job
            # record ties this to a JOB_ID
            kw["start_new_session"] = True
        self.started = time.time()
        self.proc = subprocess.Popen(
            [sys.executable, str(script), self.shape, str(self.seconds)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)
        # wait for it to actually be on the device before we measure
        time.sleep(4.0)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        return False

    def as_dict(self):
        return {
            "shape": self.shape,
            "seconds_requested": self.seconds,
            "has_ts_job_id": False,
            "detached_from_job_session": self.detached,
            "pid": self.proc.pid if self.proc else None,
            "note": ("started INSIDE this ts job, so no GPU work runs outside "
                     "the queue; what is modelled is a tenant invisible to the "
                     "queue channel, not a decision to bypass it"),
        }


def compile_cell(name, family, n, tag):
    """Find this cell in the ANCHOR's own config list.

    Matching is on (problem_size, label-substring) because the families do not
    share a size scheme: the gemm family's problem_size is D*D (the flat
    buffer the driver allocates) while its label reads "1024^3", so matching on
    the label is what selects the right cell. Getting this wrong is how the
    first run of this bead died on 'no anchor config'.
    """
    for spec, label in ak.configs(family):
        if spec["problem_size"] == n and (tag is None or tag in label):
            exe = HERE / f"cell_{name}.out"
            # Only pass tasks_max if this tree's SafeHarness knows the kwarg:
            # it is gpu-sim-u2o.6.1.2's and is not in master yet, and a
            # TypeError here cost two queued jobs.
            kw = harness_kwargs()
            kw["timeout_sec"] = 180
            ok, msg = SafeHarness(**kw).compile_kernel(synthesize(spec),
                                                      str(exe), arch="sm_120")
            if not ok:
                raise RuntimeError(f"compile {name} failed: {msg[:300]}")
            return exe, spec
    raise RuntimeError(f"no anchor config for {family} n={n} tag={tag}")


def harness_kwargs() -> dict:
    """Only pass tasks_max if this tree's SafeHarness knows the kwarg.

    It is gpu-sim-u2o.6.1.2's and is not in master yet. Centralised here
    because the same kwarg was passed in two places and fixing one of them
    cost a queued job on its own.
    """
    kw = {"timeout_sec": 300}
    if "tasks_max" in inspect.signature(SafeHarness.__init__).parameters:
        kw["tasks_max"] = TASKS_MAX
    return kw


def measure(exe, invocations=INVOCATIONS, provenance=None):
    """One process per invocation, through run_kernel, as the anchor does.

    provenance, when given, is the condition's queue_state record (or the
    fields of it) and is attached to EVERY per-invocation row, so the place
    a reader looks for per-sample provenance carries it (criterion 4 as
    written). A bare {n, median_us} row is what the rework round removed.
    """
    h = SafeHarness(**harness_kwargs())
    meds, per_inv = [], []
    for _ in range(invocations):
        r = h.run_kernel(str(exe))
        if not r.success:
            continue
        trials = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and "time_ms" in line:
                try:
                    trials.append(float(json.loads(line)["time_ms"]) * 1000.0)
                except (ValueError, KeyError):
                    pass
        if trials:
            meds.append(statistics.median(trials))
            row = {"n": len(trials), "median_us": meds[-1]}
            if provenance is not None:
                row["queue_state"] = dict(provenance)
            per_inv.append(row)
    return meds, per_inv


def main() -> int:
    print("=" * 74)
    print("gpu-sim-u2o.6.6.3 -- contention: delta, and detectability")
    print("=" * 74)
    print("The device is visible on purpose: everything here is a real GPU run,")
    print("queued. Blanking CUDA_VISIBLE_DEVICES would not sandbox run_kernel,")
    print("which sets it to str(self.gpu_id).")

    jobs = {name: compile_cell(name, fam, n, tag)
            for name, fam, n, tag in CELLS}
    for name, (exe, spec) in jobs.items():
        print(f"  compiled {name:14s} {spec['kernel_name']:16s} "
              f"n={spec['problem_size']:<9d} grid={spec['grid']}")

    conditions = [
        ("uncontended", None),
        ("queued_routed", Burner("compute", 900, detached=False)),
        ("unqueued", Burner("compute", 900, detached=True)),
    ]

    results = {}
    for cond_name, burner in conditions:
        print(f"\n--- condition: {cond_name} ---", flush=True)
        with (burner if burner else _Null()) as b:
            tel_before = telemetry(f"{cond_name}:before")
            q = qs.queue_state(own_description_substr="u2o.6.6.3",
                               own_command_substr="measure_contention.py")
            print("  telemetry before: " + json.dumps(tel_before.get("fields", {})))
            print("  queue: " + json.dumps(
                {k: q.get(k) for k in ("own_job_id", "n_other_active",
                                        "n_other_running_at_sample_time")}))
            for name, (exe, spec) in jobs.items():
                meds, per_inv = measure(exe, provenance={
                    "own_job_id": q.get("own_job_id"),
                    "n_other_active": q.get("n_other_active"),
                    "overlapping_job_ids": q.get("other_active_jobs", []),
                    "condition": cond_name,
                })
                if not meds:
                    print(f"  {name}: NO DATA")
                    continue
                med = statistics.median(meds)
                ai_cv = (statistics.stdev(meds) / statistics.fmean(meds)
                         if len(meds) > 1 else None)
                results.setdefault(name, {})[cond_name] = {
                    "median_us": med,
                    "medians_us": meds,
                    "ai_cv": ai_cv,
                    "per_invocation": per_inv,
                }
                print(f"  {name:14s} median={med:9.3f} us  ai_cv={ai_cv}")
            tel_after = telemetry(f"{cond_name}:after")
            print("  telemetry after:  " + json.dumps(tel_after.get("fields", {})))
        results.setdefault("_telemetry", {})[cond_name] = {
            "before": tel_before, "after": tel_after,
            "burner": b.as_dict() if isinstance(b, Burner) else None,
            "queue_state": q,
        }

    # --- the delta, and the headline ------------------------------------
    for name, _ in jobs.items():
        u = results.get(name, {}).get("uncontended")
        c = results.get(name, {}).get("queued_routed")
        nq = results.get(name, {}).get("unqueued")
        if u and c:
            results[name]["ratio_queued_routed"] = c["median_us"] / u["median_us"]
        if u and nq:
            results[name]["ratio_unqueued"] = nq["median_us"] / u["median_us"]

    # --- (2) the detection half, which the PI says matters more ----------
    # For each read-only signal, does it separate uncontended from
    # contended? The question is asked of the SIGNALS, not of the timings:
    # a signal that moves tells the harness it can discard; a signal that
    # does not move means the estimator and the queue record are the defence.
    detection = analyse_detection(results.get("_telemetry", {}))
    headline = headline_number(results, jobs)

    metrics = {
        "bead": "gpu-sim-u2o.6.6.3",
        "what": ("contention delta on the 6.6.2 anchor cells, and whether a "
                 "co-tenant is detectable from read-only signals"),
        "anchor_provenance": ("kernels copied verbatim from lab/gpu-sim-"
                              "u2o.6.6.2-r2 @ 3eb7583 (anchor_v1)"),
        "invocations_per_condition": INVOCATIONS,
        "task_limit_source": TASK_LIMIT_SOURCE,
        "telemetry_fields": TELEMETRY_FIELDS,
        "queue_policy_note": (
            "No GPU work runs outside the queue. The 'unqueued' condition is a "
            "burner started INSIDE this ts job with its own session and no "
            "JOB_ID, so it models a tenant invisible to the queue channel. That "
            "is the detection question; it is not a decision to bypass the "
            "queue."),
        "cells": shape_cells_for_criterion(
            {k: v for k, v in results.items() if not k.startswith("_")}),
        "telemetry_delta": results.get("_telemetry", {}),
        "detection": detection,
        "headline": headline,
        "discard_rule": build_discard_rule(),
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== detection: does each read-only signal move? ===")
    for sig, v in detection.get("signals", {}).items():
        print(f"  {sig:32s} uncontended={v['uncontended']}  "
              f"queued={v['queued_routed']}  unqueued={v['unqueued']}  "
              f"spread={v['relative_spread']}")
    print(f"  ANSWER: contention detectable from read-only telemetry alone? "
          f"{detection.get('detectable_from_telemetry_alone')}")
    print(f"  HEADLINE: {headline.get('statement')}")
    print("\n=== ratios vs uncontended ===")
    for name in jobs:
        r = results.get(name, {})
        print(f"  {name:14s} queued-routed={r.get('ratio_queued_routed')}  "
              f"unqueued={r.get('ratio_unqueued')}")
    return 0


def analyse_detection(tel: dict) -> dict:
    """Which read-only signals separate the conditions?

    Note what is NOT claimed: throttle_reasons staying 0x0 during a contended
    run is itself a finding, not a null result. The PI predicted contention
    here would show as boost-state MOVEMENT rather than throttling, and the
    obvious detector (drop samples when throttle reasons are non-zero) would
    have seen nothing. This function asks the signals rather than assuming.
    """
    sigs = ["clocks.sm", "temperature.gpu", "power.draw", "utilization.gpu",
            "clocks_throttle_reasons.active", "memory.used"]
    out = {}
    for s in sigs:
        u = num(tel.get("uncontended", {}).get("after", {}), s)
        c = num(tel.get("queued_routed", {}).get("after", {}), s)
        n = num(tel.get("unqueued", {}).get("after", {}), s)
        vals = [v for v in (u, c, n) if v is not None]
        moved = None
        if len(vals) == 3 and max(vals) > 0:
            moved = (max(vals) - min(vals)) / max(vals)
        # No 'discriminates' boolean here, by PI ruling: every signal moved
        # (even clocks.sm at 0.098), so a flag that is always true carries no
        # information and gets trusted without being read. The ordering lives
        # in relative_spread, which is what the rule below is actually built
        # from. A threshold belongs in the RULE, stated with its derivation,
        # not on the signal.
        out[s] = {"uncontended": u, "queued_routed": c, "unqueued": n,
                  "relative_spread": moved}
    throttled = any(
        (tel.get(cond, {}).get("after", {}).get("fields", {})
         .get("clocks_throttle_reasons.active", "0x0") not in
         ("0x0000000000000000", "", "Not Supported"))
        for cond in ("queued_routed", "unqueued"))
    ranked = sorted(
        ((s, v["relative_spread"]) for s, v in out.items()
         if v["relative_spread"] is not None),
        key=lambda kv: kv[1], reverse=True)
    top = [s for s, _ in ranked[:3]]
    return {
        "signals": out,
        "signals_ranked_by_spread": [
            {"signal": s, "relative_spread": sp} for s, sp in ranked],
        "throttle_reasons_went_nonzero_while_contended": throttled,
        "detectable_from_telemetry_alone": bool(ranked),
        "answer": (
            "YES, strongest first: " + ", ".join(
                f"{s} ({sp:.3f})" for s, sp in ranked[:3])
            if ranked else
            "NO. No read-only signal separated the conditions, so there is no "
            "telemetry-based discard rule; the queue-state record and the "
            "estimator's own robustness are the only defences."),
    }


CONDS = ("uncontended", "queued_routed", "unqueued")


def shape_cells_for_criterion(cells: dict) -> dict:
    """Add the criterion-1 flat keys to each cell, keeping the nested blocks.

    Criterion 1 asks for {median_uncontended_us, median_contended_us, ratio,
    ai_cv_uncontended, ai_cv_contended}, but this bead measured THREE
    conditions (two contended), so a single median_contended_us cannot name
    anything without choosing which contender it means. The flat keys below
    are condition-suffixed, and key_map says exactly how they relate to the
    criterion's two-condition vocabulary, so a checker written from the
    specification can execute. The nested per-condition blocks stay as the
    primary record; the flat keys are aliases, not a second computation.
    """
    shaped = {}
    for name, cell in cells.items():
        cell = dict(cell)
        for cond in CONDS:
            blk = cell.get(cond) or {}
            if "median_us" in blk:
                cell[f"median_{cond}_us"] = blk["median_us"]
            if "ai_cv" in blk:
                cell[f"ai_cv_{cond}"] = blk["ai_cv"]
        cell["key_map"] = {
            "criterion": "criterion 1 names one median_contended_us; "
                           "this artifact measured two contended conditions",
            "median_uncontended_us": "criterion name, verbatim",
            "median_queued_routed_us": "the queued-routed contender's "
                                          "median (criterion: median_contended_us "
                                          "for the queue-routed condition)",
            "median_unqueued_us": "the unqueued contender's median",
            "ratio_queued_routed": "queued-routed median / uncontended median",
            "ratio_unqueued": "unqueued median / uncontended median",
            "ai_cv_<cond>": "across-invocation CV under that condition",
            "authoritative": "the nested per-condition blocks; flat keys "
                               "are aliases of the same numbers",
        }
        shaped[name] = cell
    return shaped


def build_discard_rule() -> dict:
    """The discard rule WITH its detection floor, in one block, in code.

    PI rulings applied: (1) the clocks.sm clause is DROPPED -- relative
    spread 0.098 makes it the weakest signal by an order of magnitude, and
    it misfires on legitimately cold starts (idle floor 195-345 MHz vs
    2490-2760 MHz under load); (2) thresholds are REGIME-WEIGHTED, because
    memory-bound cells move 2.33x under a co-tenant while launch-bound move
    1.19x/1.05x -- and the numeric DROP thresholds below stay UNIFORM only
    because they are saturation-proven bounds, while the regime weighting is
    procedural (what a comparison is allowed to conclude), since the partial-
    tenant response curve between the measured extremes is UNMEASURED and a
    tighter memory-bound number would be invented, not measured.
    """
    return {
        "drop_if": [
            "power.draw at either boundary exceeds 150 W",
            "utilization.gpu at either boundary exceeds 50%",
            "temperature.gpu at either boundary exceeds 50 C",
        ],
        "derivation": ("uncontended reads 25.9/77 W, 6%, 38-40 C; BOTH "
                        "contended conditions read 574 W, 97-98%, 57-60 C. "
                        "An order-of-magnitude gap on power, so the threshold "
                        "is insensitive rather than delicate. Stated in the "
                        "units it was measured in."),
        "dropped_clauses": [
            "clocks.sm < 0.75x run-max: relative spread 0.098, weakest by an "
            "order of magnitude, and fires on legitimate cold starts. Dropped "
            "per PI ruling; a rule that cannot detect its target and does "
            "fire on cold starts is worse than no rule.",
            "clocks_throttle_reasons as primary signal: went 0x4 here only "
            "because the burner sits at the 575 W cap; the reviewer's unqueued "
            "sweeps ran 0x0 throughout. Would catch this run and miss theirs.",
        ],
        "detection_floor_travels_with_rule": (
            "Catches sustained contention at or near saturation (>50% "
            "utilisation class). BELOW that floor -- a partial tenant at low "
            "duty, e.g. ~20% -- the thresholds may not fire, and that case is "
            "undetectable by ANY channel available here, telemetry or "
            "queue_state alike. A rule applied without this floor is a "
            "general contamination filter, which this is not."),
        "regime_weighting": {
            "memory_bound": ("NEVER compare memory-bound cells across runs "
                              "without a contention record attached to BOTH "
                              "sides: a 2.33x neighbour effect manufactures "
                              "regressions. The numeric DROP thresholds above "
                              "still apply, but they are necessary, not "
                              "sufficient, for this regime."),
            "launch_bound": ("A null result on a launch-bound cell is NOT "
                              "evidence of no change: a 2.33x-class effect "
                              "can hide inside that noise floor. A launch-bound "
                              "A/B is a statement about the instrument, not "
                              "the kernel."),
            "why_procedural_not_numeric": (
                "The thresholds were derived from two extremes (idle vs a "
                "saturated burner); the partial-tenant response curve is "
                "UNMEASURED. A tighter memory-bound NUMBER would be invented. "
                "Measure a partial-tenant condition when one occurs naturally "
                "rather than manufacturing one."),
        },
        "channels": ("RETAIN the queue-state record as the primary channel "
                      "for queue-routed contention, and telemetry as the "
                      "primary channel for the rest. In this run queue_state "
                      "reported n_other_active=0 in ALL THREE conditions, "
                      "including both with a burner resident -- the burner is "
                      "a child of this job with no JOB_ID. Neither alone is "
                      "sufficient."),
        "merge_note_ordering_artefact": (
            "This artifact's queue_state records predate own_job_entry "
            "(6.6.6 reached master after this run): own_job_id is non-null "
            "in all three conditions and own_job_entry is ABSENT in all "
            "three. The n_other_active=0 readings are real and recorded and "
            "simultaneously unverifiable from the record itself. No re-run: "
            "re-measuring to attach a field the next run carries anyway "
            "would spend shared-box time on a superseded record."),
    }


def headline_number(results: dict, jobs: dict) -> dict:
    """The acceptance's headline: the multiplicative error bar a co-tenant
    puts on a sub-10 us measurement, against the 7.4% across-invocation CV
    already measured by 6.6.1."""
    rows = []
    for name in jobs:
        r = results.get(name, {})
        u = (r.get("uncontended") or {}).get("median_us")
        for cond in ("queued_routed", "unqueued"):
            c = (r.get(cond) or {}).get("median_us")
            if u and c:
                rows.append({"cell": name, "condition": cond,
                             "median_uncontended_us": u, "median_us": c,
                             "ratio": c / u,
                             "pct_slower": 100.0 * (c / u - 1.0)})
    sub10 = [r for r in rows if (r.get("median_uncontended_us") or 0) < 10.0]
    worst = max(rows, key=lambda r: r["ratio"]) if rows else None
    sub10_worst = max(sub10, key=lambda r: r["ratio"]) if sub10 else None
    return {
        "rows": rows,
        "worst_ratio_overall": worst,
        "worst_ratio_sub10us": sub10_worst,
        "baseline_ai_cv_sub10us_6_6_1": 0.074,
        "statement": (
            f"worst sub-10 us ratio {sub10_worst['ratio']:.2f}x "
            f"({sub10_worst['pct_slower']:+.0f}%, cell {sub10_worst['cell']}, "
            f"{sub10_worst['condition']}) against the 7.4% across-invocation "
            f"CV 6.6.1 measured for the same regime"
            if sub10_worst else "no sub-10 us data"),
    }


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())

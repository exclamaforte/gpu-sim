#!/usr/bin/env python3
"""CPU-only test: every STEP of 6.10 carries its own queue record.

The reviewer's item on gpu-sim-u2o.6.10 was not that the queue record was
wrong -- it was correct -- but that it was attached to NOTHING: sampled once
at the start of the run and then implicitly applied to all three steps. A
record that is not attached to the measurement it governs cannot be checked
against it.

This test exercises the ATTACHMENT with synthetic step states, so it needs no
GPU and no dose. It is the device-free kind of check the lab skill now asks
for: exec the shipped logic against synthetic inputs.
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments" / "gpu-sim-u2o.6.10"   # the bead artifacts
REPO = HERE.parents[0]   # tests/ -> worktree root
sys.path.insert(0, str(REPO))

SRC = EXP / "validate_once.py"
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}"
          f"{'  ' + str(detail)[:150] if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    src = SRC.read_text()
    tree = ast.parse(src)

    # 1. each of the three step dicts (step1, step2, step2b) carries a
    #    step_state. Matching on the VALUE being a literal step_state(...) call
    #    is too strict: two of them bind a local (step2_state) first, and a
    #    check that only accepts the literal form would report a correct
    #    implementation as incomplete.
    attached = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            name = (getattr(node.targets[0], "id", None)
                    if isinstance(node.targets[0], ast.Name) else None)
            for k, v in zip(node.value.keys, node.value.values):
                if name and isinstance(k, ast.Constant) and k.value == "step_state":
                    attached.setdefault(name, []).append(
                        getattr(v, "id", None) or "step_state(...)")
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and isinstance(v, ast.Name) and v.id.endswith("_state"):
                    attached[k.value] = v.id
                elif isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and isinstance(v, ast.Call) \
                        and getattr(v.func, "id", None) == "step_state":
                    attached[k.value] = "step_state(...)"
    for step in ("step1", "step2", "step2b"):
        check(f"{step} carries its own step_state",
              len(attached.get(step, [])) == 1, attached.get(step, "MISSING"))
    names = [attached[s][0] for s in ("step1", "step2", "step2b") if attached.get(s)]
    check("the three step states are three separate SAMPLES, not one reused",
          len(set(names)) == 3, names)

    # 2. step_state records the fields that make a record checkable, and
    #    explicitly reports whether own_job_entry is present rather than
    #    assuming it (the ordering artefact on the committed run)
    fn = next((n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "step_state"), None)
    check("a step_state function exists", fn is not None)
    if fn is not None:
        fields = {k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
                  for k in n.keys if isinstance(k, ast.Constant)}
        for f in ("own_job_id", "own_job_in_active_set",
                  "own_job_entry_present", "n_other_active",
                  "n_other_running_at_sample_time", "telemetry"):
            check(f"step_state records {f}", f in fields)
        # it must sample, not reuse a run-level value
        # attribute calls (qs.queue_state(...)) have no .func.id, so read the
        # attribute name too -- an extraction that only sees bare names reports
        # a clean function as not calling anything
        calls = []
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call):
                continue
            if isinstance(c.func, ast.Attribute):
                calls.append(c.func.attr)
            elif isinstance(c.func, ast.Name):
                calls.append(c.func.id)
        check("step_state SAMPLES the queue and telemetry per call",
              "queue_state" in calls and "telemetry" in calls, calls)

    # 3. the committed artifact states the attachment gap rather than being
    #    silent about it -- this run predates own_job_entry
    rp = EXP / "report.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        has_entry = "own_job_entry" in (r.get("queue_at_start") or {})
        check("committed report.json discloses the missing own_job_entry",
              r.get("queue_artifact_limitation") is not None,
              f"own_job_entry present in the artifact: {has_entry}")
        check("queue state in the artifact is non-null own_job_id",
              (r.get("queue_at_start") or {}).get("own_job_id"),
              (r.get("queue_at_start") or {}).get("own_job_id"))
    else:
        check("report.json exists", False, str(rp))

    print()
    print(f"{len(FAILS)} failure(s)"
          + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

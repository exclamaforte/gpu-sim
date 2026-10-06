#!/usr/bin/env python3
"""gpu-sim-8dw verification: PreflightValidator._run_under_fakegpu is a real
check, demonstrated criterion-by-criterion against the PI's written
acceptance. CPU-only (hostile device env, never a blanked-env sandbox).

C1 real check: AST -- _run_under_fakegpu has >=3 False-return branches and a
  True branch (it can fail; the stub could not).
C2 returns FALSE on defect: tests/test_preflight_cannot_pass.py exits 0 and
  its failure-path checks (1,2,3,5) pass.
C3 actually runs something, link route recorded: tests/test_cudart_none.sh
  exits 0; route string asserted present in code (-cudart=none, LD_PRELOAD
  popped, device deliberately not hidden).
C4 negative control: the cannot-pass test documents non-blanking AND
  tests/test_isolation_claims.py (hostile env) exits 0.
C5 standing grep: every hit for placeholder|TODO|FIXME|XXX|HACK|always return
  is a comment/history line, never load-bearing code; every `return True` in
  preflight_validator lives in a function that also returns False.

Usage: python3 experiments/gpu-sim-8dw/verify_8dw.py --metrics-out <path>
Exits nonzero on any failed criterion.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

FAILS = []
EVIDENCE = {}
HOSTILE = dict(os.environ)
HOSTILE["CUDA_VISIBLE_DEVICES"] = "nonexistent-device-8dw"


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:220] if detail else ''}",
          flush=True)
    if not ok:
        FAILS.append(name)


def run(cmd, **kw):
    kw.setdefault("timeout", 600)
    env = dict(kw.pop("env", None) or HOSTILE)
    env["CUDA_VISIBLE_DEVICES"] = "nonexistent-device-8dw"
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(REPO), **kw)


_STR_LINES = {}


def c5_is_comment(path, lineno, line, m):
    if path.suffix == ".py":
        if path not in _STR_LINES:
            import io
            import tokenize as _tok
            lines = set()
            try:
                toks = _tok.generate_tokens(
                    io.StringIO(path.read_text()).readline)
                for t in toks:
                    if t.type == _tok.STRING:
                        lines.update(range(t.start[0], t.end[0] + 1))
            except Exception:
                pass
            _STR_LINES[path] = lines
        return lineno in _STR_LINES[path] or line.strip().startswith("#")
    pre = line[:m.start()]
    return "//" in pre or "/*" in pre or line.strip().startswith("*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    src = (REPO / "epic_u2o" / "preflight_validator.py").read_text()

    # C1: branching structure of the former stub.
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_under_fakegpu")
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    falses = sum(1 for r in rets if r.value and isinstance(r.value, ast.Tuple)
                 and r.value.elts and isinstance(r.value.elts[0], ast.Constant)
                 and r.value.elts[0].value is False)
    trues = sum(1 for r in rets if r.value and isinstance(r.value, ast.Tuple)
                and r.value.elts and isinstance(r.value.elts[0], ast.Constant)
                and r.value.elts[0].value is True)
    check("C1.branches_can_fail", falses >= 3 and trues >= 1,
          f"false_returns={falses} true_returns={trues}")
    EVIDENCE["C1"] = {"false_returns": falses, "true_returns": trues}

    # C2: failure paths execute.
    r = run([sys.executable, "tests/test_preflight_cannot_pass.py"])
    tail = "\n".join(r.stdout.splitlines()[-8:])
    check("C2.cannot_pass_exits_0", r.returncode == 0, tail[:220])
    for n in ("1", "2", "3", "5"):
        check(f"C2.failure_check_{n}_passes",
              re.search(rf"PASS\s+{n}\.", r.stdout) is not None, tail[:120])
    EVIDENCE["C2"] = {"exit": r.returncode}

    # C3: real subprocess + recorded link route.
    r = run(["bash", "tests/test_cudart_none.sh"])
    check("C3.cudart_none_exits_0", r.returncode == 0,
          "\n".join(r.stdout.splitlines()[-4:])[:220])
    route = ("-cudart=none" in src and 'env.pop("LD_PRELOAD"' in src
             and "deliberately NOT blanked" in src)
    check("C3.route_recorded_in_code", route)
    EVIDENCE["C3"] = {"exit": r.returncode, "route_in_code": route,
                      "route": "-cudart=none + direct -lfakegpu link, "
                               "LD_PRELOAD popped, device not hidden"}

    # C4: negative control honored.
    tsrc = (REPO / "tests" / "test_preflight_cannot_pass.py").read_text()
    check("C4.non_blanking_documented", "NOT blanked" in tsrc)
    r = run([sys.executable, "tests/test_isolation_claims.py"])
    check("C4.hostile_env_suite_exits_0", r.returncode == 0,
          "\n".join(r.stdout.splitlines()[-2:])[:220])
    EVIDENCE["C4"] = {"non_blanking_documented": True, "exit": r.returncode}

    # C5: standing grep. .py: tokenize marks string/docstring lines as
    # non-code. .h: a hit after // or inside /* */ is a comment.
    hits = []
    pat = re.compile(r"placeholder|TODO|FIXME|XXX|HACK|always return", re.I)
    for f in list((REPO / "epic_u2o").glob("*.py")) + \
             list((REPO / "shim").glob("*.h")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            m = pat.search(line)
            if m:
                is_c = c5_is_comment(f, i, line, m)
                hits.append({"file": f.name, "line": i,
                             "text": line.strip()[:160], "comment": is_c})
    code_hits = [h for h in hits if not h["comment"]]
    check("C5.no_load_bearing_stubs", not code_hits,
          json.dumps(code_hits[:3]) if code_hits else f"{len(hits)} comment hits")
    # Every `return True` in the module coexists with False (can fail).
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            rr = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            has_t = any(r.value and isinstance(r.value, ast.Tuple) and r.value.elts
                        and isinstance(r.value.elts[0], ast.Constant)
                        and r.value.elts[0].value is True for r in rr)
            has_f = any(r.value and isinstance(r.value, ast.Tuple) and r.value.elts
                        and isinstance(r.value.elts[0], ast.Constant)
                        and r.value.elts[0].value is False for r in rr)
            if has_t:
                check(f"C5.{node.name}_true_coexists_with_false", has_f)
    EVIDENCE["C5"] = {"hits": hits, "code_hits": code_hits}

    metrics = {"bead": "gpu-sim-8dw", "criteria": EVIDENCE,
               "verdict": "pass" if not FAILS else "fail", "failures": FAILS}
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

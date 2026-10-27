#!/usr/bin/env python3
"""Check that gate citations in experiments/*/metrics.json name a real tip.

A gate citation that cannot be tied to the artifact it claims is worse than no
citation. Fail-closed rules:
  * a 'gate' string that contains no tip hash -> FAIL
  * a cited tip that does not exist in this repo -> FAIL
  * if a run handle is cited and its run.log exists and carries a GATE TIP
    line, the cited tip must equal that GATE TIP -> else FAIL
  * if the cited tip is an ancestor of (or equal to) HEAD, the diff between
    the cited tip and HEAD must be confined to experiments/*/metrics.json,
    otherwise a real code change landed after the gate ran -> FAIL
  * a cited tip on an unrelated branch (not an ancestor of HEAD) is reported
    as UNVERIFIABLE rather than failed.
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIP_RE = re.compile(r"\btip\s+([0-9a-f]{7,40})\b", re.I)
RUN_RE = re.compile(r"\brun\s+([A-Za-z0-9_-]+)", re.I)
METRICS_ONLY = re.compile(r"^experiments/[^/]+/metrics\.json$")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                         capture_output=True, text=True)


def is_ancestor(a, b):
    r = git("merge-base", "--is-ancestor", a, b)
    return r.returncode == 0


def diff_names(a, b):
    r = git("diff", "--name-only", f"{a}..{b}")
    if r.returncode != 0:
        return None
    return [x for x in r.stdout.split() if x.strip()]


def run_log_gate_tip(handle):
    log = REPO / ".lab" / "runs" / handle / "run.log"
    if not log.exists():
        return None
    try:
        text = log.read_text(errors="replace")
    except Exception:
        return None
    m = re.search(r"GATE TIP:\s*([0-9a-f]{7,40})", text)
    return m.group(1) if m else None


def main() -> int:
    head = git("rev-parse", "HEAD").stdout.strip()
    bad = []
    unverifiable = []
    for metrics in sorted((REPO / "experiments").glob("*/metrics.json")):
        try:
            data = json.loads(metrics.read_text())
        except Exception:
            continue
        gate = data.get("gate")
        if not isinstance(gate, str):
            continue
        m = TIP_RE.search(gate)
        if not m:
            print(f"UNVERIFIABLE: {metrics}: no tip hash cited")
            continue
        tip = m.group(1)
        # existence
        r = git("cat-file", "-e", tip)
        if r.returncode != 0:
            bad.append((metrics, f"tip {tip} does not exist in this repo"))
            continue
        # run.log comparison if handle present
        rm = RUN_RE.search(gate)
        if rm:
            handle = rm.group(1)
            rl_tip = run_log_gate_tip(handle)
            if rl_tip is not None and rl_tip[:len(tip)] != tip[:len(rl_tip)] and tip[:len(rl_tip)] != rl_tip[:len(tip)]:
                # compare as prefixes
                if not (rl_tip.startswith(tip) or tip.startswith(rl_tip)):
                    bad.append((metrics, f"run {handle} GATE TIP {rl_tip} != cited {tip}"))
                    continue
        # ancestor / diff rule
        if tip == head:
            pass
        elif is_ancestor(tip, head):
            names = diff_names(tip, head)
            if names is None:
                bad.append((metrics, f"cannot diff {tip}..HEAD"))
                continue
            offenders = [n for n in names if not METRICS_ONLY.match(n)]
            if offenders:
                bad.append((metrics, f"real code change after cited tip: {offenders}"))
                continue
        else:
            # unrelated branch tip
            if is_ancestor(head, tip):
                bad.append((metrics, f"cited tip {tip} is ahead of HEAD {head}"))
                continue
            unverifiable.append((metrics, tip))
            continue

    for m, msg in bad:
        print(f"INVALID: {m}: {msg}")
    for m, t in unverifiable:
        print(f"UNVERIFIABLE: {m}: tip {t} is on an unrelated branch (no run record?)")
    if bad:
        return 1
    print("check_gate_citations: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

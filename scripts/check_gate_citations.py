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
import json, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIP_RE = re.compile(r"\btip\s+([0-9a-f]{7,40})\b", re.I)
RUN_RE = re.compile(r"\brun\s+(?:handle\s+)?([A-Za-z0-9_-]+)", re.I)
METRICS_ONLY = re.compile(r"^experiments/[^/]+/metrics\.json$")
HANDLE_TOKEN_RE = re.compile(r"\bhandle\s+([A-Za-z0-9_-]+|none)\b", re.I)
DELTA_RE = re.compile(r"\bdelta\b\s*[:=]?\s*(\S.*)$", re.I | re.M)
# A run that predates GATE TIP output cannot be tied to any tip. Naming that
# explicitly is a delta claim the checker can VERIFY (the run.log must really
# lack GATE TIP), unlike an unqualified "unknown", which would be a loophole.
LEGACY_DELTA = "delta unavailable: predates gate tip output"
RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 0.2


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

def _wait_for_path(path, attempts=RETRY_ATTEMPTS):
    """.lab/runs is intermittently unreadable (observed: a listed, greppable
    handle then 12/12 consecutive -d false with no live process), so a single
    failed stat must not be read as evidence loss. Retry briefly; a persistent
    absence is real."""
    for i in range(attempts):
        if path.exists():
            return path
        if i + 1 < attempts:
            time.sleep(RETRY_DELAY_S)
    return None


def _run_log_path(handle):
    candidates = [REPO / ".lab" / "runs" / handle / "run.log"]
    s = str(REPO)
    if ".lab/wt/" in s.replace(os.sep, "/"):
        main_root = REPO
        while main_root.name and main_root != main_root.parent:
            if main_root.parent.name == "wt" and main_root.parent.parent.name == ".lab":
                main_root = main_root.parent.parent.parent
                break
            main_root = main_root.parent
        candidates.append(main_root / ".lab" / "runs" / handle / "run.log")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_run_log(handle, attempts=RETRY_ATTEMPTS):
    """Resolve run.log with a bounded retry over EVERY candidate.

    Two measured reasons not to poll one fixed path: (a) `.lab/runs` is
    intermittently unreadable -- a listed, greppable handle then 12/12
    consecutive -d false with no live process -- so one failed stat must not be
    read as evidence loss; (b) in a worktree the worktree-local `.lab/runs` is
    NOT the directory the gate wrote to, so a fixed candidates[0] retries the
    wrong tree entirely. Re-resolving each attempt covers both.
    """
    for i in range(attempts):
        path = _run_log_path(handle)
        if path.exists():
            return path
        if i + 1 < attempts:
            time.sleep(RETRY_DELAY_S)
    return None


def _tip_from_path(log):
    try:
        text = log.read_text(errors="replace")
    except Exception:
        return None
    m = re.search(r"GATE TIP:\s*([0-9a-f]{7,40})", text)
    return m.group(1) if m else None


def _has_end_from_path(log):
    try:
        return "GATE DIRTY-AT-END:" in log.read_text(errors="replace")
    except Exception:
        return None


def run_log_gate_tip(handle):
    log = _run_log_path(handle)
    if not log.exists():
        return None
    try:
        text = log.read_text(errors="replace")
    except Exception:
        return None
    m = re.search(r"GATE TIP:\s*([0-9a-f]{7,40})", text)
    return m.group(1) if m else None


def run_log_has_dirty_end(handle):
    log = _run_log_path(handle)
    if not log.exists():
        return None
    try:
        text = log.read_text(errors="replace")
    except Exception:
        return None
    return "GATE DIRTY-AT-END:" in text


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
        # only gate-run citations are in scope; other 'gate' fields (e.g.
        # entry-gate status strings) are not verifiable citations
        if "gate.sh" not in gate and "PASS" not in gate.upper() and "TIP" not in gate.upper():
            continue
        m = TIP_RE.search(gate)
        declared = "UNVERIFIABLE FOR THIS TIP" in gate.upper()
        if not m:
            # no tip hash: allowed only as a declared citation (parts checked
            # below), otherwise it is a silent evidence-free pass -> FAIL
            if declared and (HANDLE_TOKEN_RE.search(gate) or RUN_RE.search(gate)) and (
                    DELTA_RE.search(gate) or LEGACY_DELTA in gate.lower()):
                if LEGACY_DELTA in gate.lower():
                    hm0 = HANDLE_TOKEN_RE.search(gate) or RUN_RE.search(gate)
                    if hm0 and hm0.group(1).lower() != "none" and \
                            _tip_from_path(_run_log_path(hm0.group(1))) is not None:
                        bad.append((metrics, "claims 'predates GATE TIP' but the named run.log carries one"))
                        continue
                unverifiable.append((metrics, "no tip hash cited (declared unverifiable)"))
                continue
            missing = []
            if declared:
                if not HANDLE_TOKEN_RE.search(gate) and not RUN_RE.search(gate):
                    missing.append("handle token")
                if not DELTA_RE.search(gate):
                    missing.append("delta listing")
            bad.append((metrics, "no tip hash cited" + (
                " (declared unverifiable but missing " + "; ".join(missing) + ")"
                if missing else "")))
            continue
        tip = m.group(1)
        # Form C (declared unverifiable) is a positive claim with three
        # machine-checkable parts: the marker, a handle token (an id or
        # 'none'), and a delta listing. Ancestry is deliberately NOT required:
        # a rebase rewrites the tip, which is the very reason C exists.
        if declared:
            parts = []
            if not HANDLE_TOKEN_RE.search(gate) and not RUN_RE.search(gate):
                parts.append("handle token ('handle <id>' or 'handle none')")
            legacy_claim = LEGACY_DELTA in gate.lower()
            if not DELTA_RE.search(gate) and not legacy_claim:
                parts.append("delta listing")
            if parts:
                bad.append((metrics, "declared unverifiable but missing " + "; ".join(parts)))
                continue
            if legacy_claim:
                # the claim must be true of the run record it names
                hm = HANDLE_TOKEN_RE.search(gate) or RUN_RE.search(gate)
                if hm and _tip_from_path(_run_log_path(hm.group(1))) is not None:
                    bad.append((metrics, "claims 'run predates GATE TIP' but the run.log carries one"))
                    continue
        # existence
        r = git("cat-file", "-e", tip)
        if r.returncode != 0:
            bad.append((metrics, f"tip {tip} does not exist in this repo"))
            continue
        # run.log comparison if a handle is cited: accept BOTH the
        # 'handle <id>' token form and the legacy 'run <id>' form, and treat
        # 'handle none' as no record at all rather than a handle named "none"
        hm = HANDLE_TOKEN_RE.search(gate) or RUN_RE.search(gate)
        if hm and hm.group(1).lower() == "none":
            hm = None
        if hm:
            handle = hm.group(1)
            # missing run record -> retry briefly, then FAIL unless the
            # citation declares itself unverifiable (absent evidence must be
            # acknowledged, never silently passed)
            log_path = _resolve_run_log(handle)
            if log_path is None:
                if declared:
                    unverifiable.append((metrics, f"handle {handle}, run.log missing (declared unverifiable)"))
                else:
                    bad.append((metrics, f"handle {handle}, run.log missing at {_run_log_path(handle)}"))
                continue
            # re-read AFTER the wait: a pre-retry read can miss on the same
            # transient that the retry was added to absorb
            rl_tip = _tip_from_path(log_path)
            if rl_tip is not None and not (rl_tip.startswith(tip) or tip.startswith(rl_tip)):
                bad.append((metrics, f"run {handle} GATE TIP {rl_tip} != cited {tip}"))
                continue
            # dirty-at-end presence: a truncated/killed new-format log must fail;
            # legacy logs (no GATE TIP line) are unverifiable, not failed
            has_end = _has_end_from_path(log_path)
            if has_end is False:
                if rl_tip is None:
                    unverifiable.append((metrics, f"handle {handle}, legacy run.log (no GATE TIP/DIRTY-AT-END)"))
                    continue
                bad.append((metrics, f"run {handle} log lacks GATE DIRTY-AT-END (truncated/killed)"))
                continue
        else:
            # a tip with no handle skips the whole run-log contract; that is a
            # failure unless the citation declares itself unverifiable
            if declared:
                unverifiable.append((metrics, f"no run handle cited for tip {tip} (declared unverifiable)"))
            else:
                bad.append((metrics, f"no run handle cited for tip {tip}"))
            continue
        # ancestor / diff rule; declared-unverifiable form is reported, not failed
        if declared:
            unverifiable.append((metrics, f"tip {tip} (declared unverifiable)"))
            continue
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
            # Neither form A (tip match), B (ancestor, metrics-only delta) nor
            # C (declared) holds. An unrelated tip with no declaration is a
            # citation that cannot be tied to this branch at all -> FAIL.
            bad.append((metrics, f"cited tip {tip} is on an unrelated branch and not declared unverifiable"))
            continue

    for m, msg in bad:
        print(f"INVALID: {m}: {msg}")
    for m, t in unverifiable:
        if isinstance(t, str) and t.startswith(("tip ", "handle ", "no ")):
            print(f"UNVERIFIABLE: {m}: {t}")
        else:
            print(f"UNVERIFIABLE: {m}: tip {t} is on an unrelated branch (no run record?)")
    if bad:
        return 1
    print("check_gate_citations: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

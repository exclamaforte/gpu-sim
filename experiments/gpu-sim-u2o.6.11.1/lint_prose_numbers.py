#!/usr/bin/env python3
"""Lint for prose/artifact number agreement. gpu-sim-u2o.6.11.1.

Every substantive failure on 2026-09-27/28 was a PERSON SUMMARISING A NUMBER,
and none was a measurement. The mechanism is always the same:

    A PROSE SENTENCE ABOUT A NUMBER A SCRIPT PRODUCED IS A HAND-COPIED
    DERIVED VALUE.

Each instance was caught by comparing artifact against prose BY HAND, which is
the right instinct and the most expensive possible way to do it. This lint does
the comparison mechanically, and it closes the class rather than discouraging
the habit.

TWO CONSTRAINTS, both learned the hard way and both load-bearing:

1. THE LINT DOES NOT ENCODE JUDGEMENT. It verifies AGREEMENT, never whether a
   claim is true, and it fires on DISAGREEMENT so a person adjudicates. The
   empty-launch-config case on gpu-sim-8dw is the proof: the repaired preflight
   legitimately PASSES, a summary claimed it failed, and 'fixing' the code to
   match the summary would have introduced a FALSE NEGATIVE to close a false
   positive. So this tool never proposes changing a number or a verdict -- it
   only reports that two values that were supposed to be the same are not.

2. A LINT IS A REPORT; A GENERATOR IS A FIX. Where a generator already exists,
   the fix for a hand-typed number is to derive it, not to lint it (see
   generate_inventory.py in this bead). The generator cannot drift; the lint
   can only notice.

Artifacts are read from GIT REFS, not only the working tree, because most of
the historical cases live on unmerged branches: an agreement check that can
only see master cannot check the claims that motivated it.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# --- finding ---------------------------------------------------------------

@dataclass
class Finding:
    bead: str
    check: str
    fragment: str          # the quoted prose, so a human can find it
    artifact_path: str
    artifact_value: object
    comment_value: object
    note: str = ""

    def render(self) -> str:
        both = (f"artifact={self.artifact_value!r}  comment={self.comment_value!r}")
        return (f"  [{self.bead}] {self.check}\n"
                f"      prose    : \"{self.fragment}\"\n"
                f"      artifact : {self.artifact_path} = {self.artifact_value!r}\n"
                f"      comment  : {self.comment_value!r}\n"
                f"      disagree : {both}\n"
                + (f"      note     : {self.note}\n" if self.note else ""))


# --- artifact access -------------------------------------------------------

def refs() -> list[str]:
    out = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    return out.split()


def refs_for(bead: str) -> list[str]:
    """A bead's OWN branch first, then everything else.

    Without this the first alphabetical ref wins, and the lint reports a
    finding against whichever stale snapshot sorts first -- an agreement check
    that silently checks the wrong artifact is worse than none.
    """
    all_refs = refs()
    mine = [r for r in all_refs if bead.replace("gpu-sim-", "") in r
            or bead.split("-")[-1] in r]
    return mine + [r for r in all_refs if r not in mine]


def artifact_text(bead: str, name: str) -> tuple[str, str] | None:
    """Return (text, ref) for the first ref carrying experiments/<bead>/<name>."""
    for ref in refs_for(bead):
        p = subprocess.run(["git", "cat-file", "-e", f"{ref}:experiments/{bead}/{name}"],
                           cwd=REPO, capture_output=True)
        if p.returncode == 0:
            body = subprocess.run(["git", "show", f"{ref}:experiments/{bead}/{name}"],
                                  cwd=REPO, capture_output=True, text=True).stdout
            return body, f"{ref}:experiments/{bead}/{name}"
    return None


def artifact_json(bead: str, name: str) -> tuple[dict, str] | None:
    got = artifact_text(bead, name)
    if not got:
        return None
    body, ref = got
    try:
        return json.loads(body), ref
    except json.JSONDecodeError:
        return None


def bead_comments(bead: str) -> str:
    r = subprocess.run(["bd", "comments", bead], cwd=REPO,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


# --- the checks ------------------------------------------------------------
#
# Each check is (a) a way to pull a value OUT OF THE ARTIFACT and (b) a way to
# pull the corresponding value OUT OF PROSE. The check fires when the two
# disagree. Nothing here decides which is right.

def _fmt(x: float) -> str:
    return f"{x:.1f}" if abs(x) < 1000 else f"{x:,.0f}"


def check_6_6_1_worst_case_mdd(findings: list, comments: str | None = None) -> None:
    """The comment named 59.3% for the worst cell; the artifact says 76.3%."""
    bead = "gpu-sim-u2o.6.6.1"
    got = artifact_json(bead, "metrics.json")
    if not got:
        return
    doc, path = got
    worst = doc["noise_floor"]["mdd_worst_case_cell"]
    art = round(worst["mdd_pct_at_N10"], 1)
    # the prose claim, quoted exactly as written
    pat = re.compile(r"worst case\s*=\s*the empty cell:\s*10 trials\s*([\d.]+)\s*%",
                     re.I)
    for m in pat.finditer(comments if comments is not None else bead_comments(bead)):
        claimed = float(m.group(1))
        if abs(claimed - art) > 0.05:
            findings.append(Finding(
                bead, "declared-worst-case disagrees with prose",
                m.group(0), path, art, claimed,
                note=f"artifact names the worst cell as "
                     f"{worst['kernel']}/{worst['mode']}/n={worst['n_problem_size']}"))


def check_6_6_3_constant_boolean(findings: list, comments: str | None = None) -> None:
    """A boolean that is true for every entry cannot rank them; prose used it as if it could.

    Mechanical, not a judgement: the field is CONSTANT across all entries, so it
    carries no ordering. The comment cites it as a discriminator. The lint
    reports the disagreement and stops -- it does not say the boolean is wrong,
    only that it does not vary and the prose treats it as if it does.
    """
    bead = "gpu-sim-u2o.6.6.3"
    got = artifact_json(bead, "metrics.json")
    if not got:
        return
    doc, path = got
    signals = doc["detection"]["signals"]
    # iterate FIELD names (from the first entry), not signal names. Getting
    # this axis wrong is how the first version of this check silently found
    # nothing: {s[key] for s in signals.values()} looks a signal NAME up as a
    # field, so every lookup missed and const stayed empty. A check that finds
    # nothing because it is looking in the wrong place is the most dangerous
    # kind of quiet.
    first = next(iter(signals.values()))
    const = {}
    for key in first:
        vals = {s[key] for s in signals.values() if key in s}
        if len(vals) == 1:
            const[key] = vals.pop()
    if not const:
        return
    # spread range, so the report shows WHY a constant boolean cannot rank
    spreads = sorted((s["relative_spread"], k) for k, s in signals.items()
                     if "relative_spread" in s)
    rng = f"{spreads[0][0]:.3f} ({spreads[0][1]}) .. {spreads[-1][0]:.3f} ({spreads[-1][1]})" \
        if spreads else "n/a"
    text = comments if comments is not None else bead_comments(bead)
    pat = re.compile(r"Every one of the (\w+) signals carries [\"']?discriminates[\"']?: true",
                     re.I)
    for m in pat.finditer(text):
        findings.append(Finding(
            bead, "constant boolean cited as a discriminator",
            m.group(0), path,
            f"{len(const)} constant-valued field(s) {const}; "
            f"relative_spread ranges {rng}",
            m.group(1) + " signals all true",
            note="the field does not vary, so the ordering lives in "
                 "relative_spread, not in this boolean"))


def check_6_10_ratio(finding_list: list, comments: str | None = None) -> None:
    """overall_pass first read False because the ratio divided seconds by milliseconds.

    The prose reports a '3x' ratio. The artifact stores the two operands; the
    lint recomputes the ratio in the units the operands are actually in and
    compares.
    """
    bead = "gpu-sim-u2o.6.10"
    got = artifact_json(bead, "report.json")
    if not got:
        return
    doc, path = got
    s1 = doc.get("step1", {})
    # Re-derive from the operands, independently of the artifact's own summary
    # field, and cross-check the summary field against that derivation too.
    wall_sec = s1.get("process_wall_sec")
    sum_ms = s1.get("sum_of_trials_ms")
    stored = s1.get("wall_over_trials_ratio")
    if wall_sec is None or sum_ms is None:
        return
    # correct units: wall in SECONDS against trial sum in MILLISECONDS, so the
    # ratio is (sec*1000)/ms. The historical bug divided sec by ms directly,
    # which is smaller by 1000x and printed "3x" where the answer is 2713x.
    truth = (wall_sec * 1000.0) / sum_ms
    pat = re.compile(r"it printed\s*([\d.]+)x where the real ratio is\s*([\d.]+)x", re.I)
    for m in pat.finditer(comments if comments is not None else bead_comments(bead)):
        printed, claimed = float(m.group(1)), float(m.group(2))
        if abs(printed - truth) / truth > 0.01:
            finding_list.append(Finding(
                bead, "ratio printed in prose is not what the operands give",
                m.group(0), path, f"(process_wall_sec*1000)/sum_of_trials_ms "
                                  f"= {truth:.4g}"
                                  + (f"; stored wall_over_trials_ratio={stored:.4g}"
                                     if stored is not None else ""),
                printed,
                note=f"the same sentence claims the real ratio is {claimed}x, which "
                     f"the operands DO support ({truth:.4g}); the disagreement is "
                     f"with the value the harness actually printed. artifact's own "
                     f"unit_bug_correction records the cause."))


def check_8dw_claim_vs_output(findings: list, comments: str | None = None) -> None:
    """A closing sentence claimed False where the committed log shows True."""
    bead = "gpu-sim-8dw"
    got = artifact_text(bead, "demo_output.log")
    if not got:
        return
    body, path = got
    lines = [l for l in body.splitlines() if "-> passed=" in l]
    # every probe row the log actually shows
    actual = {re.search(r"(passed=(\w+))", l).group(1) for l in lines}
    # LOCAL, not module-global: a module-level dedup set made this check fire
    # once and then go silent forever, because the key was still present from
    # the previous run. A check that stops reporting is worse than one that
    # over-reports, and the test caught it by running the check in isolation.
    seen: set = set()
    pat = re.compile(r"answered False to (?:BOTH|both) probes", re.I)
    for m in pat.finditer(comments if comments is not None else bead_comments(bead)):
        if "passed=True" in actual:
            key = (bead, m.group(0), path)
            if key in seen:
                continue          # the same sentence appears in several comments
            seen.add(key)
            findings.append(Finding(
                bead, "claim contradicts the committed demonstration log",
                m.group(0), path, sorted(actual), "False to both",
                note="the log shows a probe passing; the prose says the repaired "
                     "module answered False to both"))


def check_6_11_counts(findings: list, comments: str | None = None) -> None:
    """'rm -f the 33 paths' beside a derived count of 34; comment 41 vs tree 42."""
    bead = "gpu-sim-u2o.6.11"
    got = artifact_text(bead, "tracks_inventory.md")
    if not got:
        return
    body, path = got
    m = re.search(r"rm -f <the (\d+) paths this branch tracks>", body)
    if not m:
        return
    typed = int(m.group(1))
    # Derive the real removal count from the TREE the command is run against,
    # not from another number in the same prose. Deriving a count from prose
    # that disagrees with prose is the same class of error one level in.
    ref = path.split(":", 1)[0]
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", "epic_u2o/"],
        cwd=REPO, capture_output=True, text=True).stdout.split()
    derived = len(tree)
    on_disk = len([q for q in (REPO / "epic_u2o").iterdir() if q.is_file()])
    if typed != derived:
        findings.append(Finding(
            bead, "typed removal count disagrees with the tree",
            m.group(0), path, f"{derived} paths tracked in {ref}", typed,
            note=f"{on_disk} paths on disk under epic_u2o/; the count in the "
                 f"command is TYPED while the counts beside it are derived, "
                 f"which is how the two drifted apart"))


CHECKS = (check_6_6_1_worst_case_mdd, check_6_6_3_constant_boolean,
          check_6_10_ratio, check_8dw_claim_vs_output, check_6_11_counts)


def main(argv: list[str]) -> int:
    beads = argv[1:] or ["gpu-sim-u2o.6.6.1", "gpu-sim-u2o.6.6.3", "gpu-sim-u2o.6.10",
                         "gpu-sim-8dw", "gpu-sim-u2o.6.11"]
    findings: list = []
    for fn in CHECKS:
        try:
            fn(findings)
        except Exception as exc:                      # a check that crashes is
            findings.append(Finding(                 # itself a finding, never
                "(lint)", f"check {fn.__name__} raised", "", str(exc), None, None,
                note="this lint reports rather than decides; a check that cannot "
                     "run must not read as a check that passed"))
    print("prose/artifact number agreement lint")
    print(f"beads under test: {', '.join(beads)}")
    print(f"checks run: {len(CHECKS)}   findings: {len(findings)}\n")
    for f in findings:
        print(f.render())
    if not findings:
        print("  no disagreements found")
    print("\nThis lint does not say which value is correct. It reports two values")
    print("that were supposed to be the same and are not. Adjudicate by hand.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

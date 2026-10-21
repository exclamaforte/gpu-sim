#!/usr/bin/env python3
"""Prose-number provenance lint.  gpu-sim-4u8.

Every numeric claim in a results document must appear in a committed artifact that
the document CITES.  The rule exists because the PI wrote unverified numbers into
experiments/gpu-sim-pkb/results.md twice in one session:

  1. a table cell reading "1.00 after dedup" where the reduction script reported
     2.00 -- the number was adjusted by hand and presented as a measurement;
  2. run counts ("6/6", "4/4") citing runs that existed on disk but were never
     committed, plus a sentence claiming two sends shared a payload digest when
     the digests differed.

Both were caught by a human reviewer reading the artifact.  That is not a control.

WHAT THIS CHECKS
  * decimal claims (d.dd) appearing in a results document must occur in the text of
    an artifact the document cites (or in that artifact's JSON values);
  * "N/M" run counts must be satisfiable from the files committed in the cited
    artifact directory.

WHAT IT DELIBERATELY DOES NOT CHECK
  * semantic claims ("the model is wrong") -- only numeric provenance;
  * integers in general: sizes, ids and indices legitimately appear as labels.
    Decimals are the high-risk surface because a rounded factor looks like a
    measurement even when it is an estimate.

Usage:
    python3 scripts/check_prose_numbers.py [--check] path/to/results.md [more...]
Exit 1 if any claim has no provenance.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A cited artifact: a path-looking token with a known artifact suffix.
# .md is deliberately NOT an artifact suffix: a document that cites another document
# can launder any number through it (a fixture citing the corrected results.md made
# every rejected number look supported). Provenance must point at DATA.
ARTIFACT_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:txt|json|jsonl|log)")
DECIMAL_RE = re.compile(r"(?<![\d.])(\d+\.\d{2})(?![\d])")
RUNCOUNT_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\s*runs?\b")

# A results document is a markdown file whose name marks it as one.
RESULTS_NAMES = {"results.md", "report.md", "REPORT.md"}

# Documents grandfathered: they predate this lint and are not rewritten retroactively.
# The list is committed (scripts/prose_numbers_grandfathered.txt) so the debt is
# visible. Adding a line requires a bead and a reason; removing one means the
# document now passes.
GRANDFATHERED_FILE = Path(__file__).resolve().parent / "prose_numbers_grandfathered.txt"


def grandfathered() -> set[str]:
    if not GRANDFATHERED_FILE.is_file():
        return set()
    out: set[str] = set()
    for line in GRANDFATHERED_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def cited_artifacts(doc: Path) -> list[Path]:
    """Artifacts the document cites, resolved against the doc dir then the repo."""
    found: list[Path] = []
    text = doc.read_text(errors="replace")
    for m in ARTIFACT_RE.finditer(text):
        raw = m.group(0)
        for base in (doc.parent, REPO):
            p = (base / raw).resolve()
            if p.is_file() and p not in found:
                found.append(p)
                break
    return found


def artifact_numbers(path: Path) -> set[str]:
    """Decimal strings an artifact can support, from its text and its JSON values."""
    out: set[str] = set()
    text = path.read_text(errors="replace")
    # Any decimal the artifact prints also supports its 2-dp rounding: the lint must
    # not fire when a document quotes '1.25' for an artifact line reading '1.2500'.
    for m in re.finditer(r"(?<![\d.])(\d+\.\d+)(?![\d])", text):
        out.add(m.group(1))
        try:
            out.add(f"{float(m.group(1)):.2f}")
        except ValueError:
            pass
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return out
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, float):
                out.add(f"{node:.2f}")
    return out


def committed_count(artifacts: list[Path], pattern: str) -> int:
    """How many files matching `pattern` are COMMITTED next to a cited artifact."""
    dirs = {a.parent for a in artifacts}
    total = 0
    for d in dirs:
        try:
            rel = d.relative_to(REPO)
        except ValueError:
            continue
        try:
            listed = subprocess.run(
                ["git", "ls-files", "--", f"{rel}/{pattern}"],
                cwd=REPO, capture_output=True, text=True, check=True,
            ).stdout.split()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        total += len(listed)
    return total


def check_doc(doc: Path) -> list[str]:
    findings: list[str] = []
    try:
        rel = str(doc.resolve().relative_to(REPO))
    except ValueError:
        rel = str(doc)
    if rel in grandfathered():
        return findings

    artifacts = cited_artifacts(doc)
    if not artifacts:
        # Nothing to check against.  Say so rather than pass silently: a results
        # document with no cited artifact is itself the failure this lint exists for.
        findings.append(f"{rel}: cites no artifact file -- every number in it is unprovenanced")
        return findings

    supported: set[str] = set()
    for a in artifacts:
        supported |= artifact_numbers(a)

    text = doc.read_text(errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in DECIMAL_RE.finditer(line):
            num = m.group(1)
            if num not in supported:
                findings.append(
                    f"{rel}:{lineno}: claim {num} does not appear in any cited artifact "
                    f"({', '.join(str(a.name) for a in artifacts)})"
                )
        # Size-aware when the line names a size: '6/6 runs' on a 524288 row must be
        # countable from committed traces FOR THAT SIZE, not from every trace present.
        sizes = [int(x) for x in re.findall(r"\b(\d{4,})\b", line)]
        for m in RUNCOUNT_RE.finditer(line):
            n, total = int(m.group(1)), int(m.group(2))
            if sizes:
                have = sum(committed_count(artifacts, f"trace_S{sz}_*.log") for sz in set(sizes))
            else:
                have = committed_count(artifacts, "trace_*.log")
            if total > have:
                findings.append(
                    f"{rel}:{lineno}: claims {m.group(0)!r} but only {have} trace file(s) "
                    f"are committed next to the cited artifacts"
                )
    return findings


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv if not a.startswith("-")]
    if not paths:
        paths = [p for p in REPO.glob("experiments/*/*.md") if p.name in RESULTS_NAMES]
    findings: list[str] = []
    for p in paths:
        findings.extend(check_doc(p))
    for f in findings:
        print(f"FINDING: {f}")
    if findings:
        print(f"\n{len(findings)} unprovenanced numeric claim(s)")
        return 1
    print(f"OK: {len(paths)} document(s), all numeric claims have provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

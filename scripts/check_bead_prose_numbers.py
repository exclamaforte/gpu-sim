#!/usr/bin/env python3
"""Prose-number provenance lint for bead descriptions and comments.  gpu-sim-3fm.

Same rule as scripts/check_prose_numbers.py (gpu-sim-4u8), adapted for bead text:
every numeric claim (d.dd decimal) must be supported by an artifact cited in the
text or living in the bead's own experiment directory.

Bead text has no file paths by default, so the citation rule is adapted:
  * experiments/<bead-id>/ is an IMPLICIT citation directory (all artifacts there
    count as cited);
  * explicit artifact paths in the text also count;
  * if neither exists, the bead is reported as citing no artifact.

This script NEVER blocks bead creation or update: it is a read-only check run in
make check or invoked manually. It does not modify the beads database.

Usage:
    python3 scripts/check_bead_prose_numbers.py --bead <id>        # fetch from bd
    python3 scripts/check_bead_prose_numbers.py path/to/text.md     # check a file
    python3 scripts/check_bead_prose_numbers.py --bead <id> path    # file with bead id

A file may declare its bead id on a line like:   <!-- bead: gpu-sim-bl5 -->

Exit 1 if any claim has no provenance.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:txt|json|jsonl|log)")
DECIMAL_RE = re.compile(r"(?<![\d.])(\d+\.\d{2})(?![\d])")
RUNCOUNT_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\s*runs?\b")
GRANDFATHERED_FILE = Path(__file__).resolve().parent / "bead_prose_numbers_grandfathered.txt"


def grandfathered() -> set[str]:
    out: set[str] = set()
    if not GRANDFATHERED_FILE.is_file():
        return out
    for line in GRANDFATHERED_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line.split()[0])
    return out


def bead_text(bead_id: str) -> str:
    """Fetch description + design + acceptance + comments from bd as one text."""
    out = subprocess.run(
        ["bd", "show", bead_id, "--json"], cwd=REPO,
        capture_output=True, text=True,
    )
    parts: list[str] = []
    if out.returncode == 0:
        try:
            d = json.loads(out.stdout)[0]
            for k in ("description", "design", "acceptance_criteria"):
                v = d.get(k)
                if v:
                    parts.append(str(v))
        except (json.JSONDecodeError, IndexError, KeyError):
            parts.append(out.stdout)
    # comments
    c = subprocess.run(
        ["bd", "comments", bead_id, "--json"], cwd=REPO,
        capture_output=True, text=True,
    )
    if c.returncode == 0:
        try:
            for com in json.loads(c.stdout):
                t = com.get("text")
                if t:
                    parts.append(str(t))
        except json.JSONDecodeError:
            pass
    return "\n".join(parts)


def implicit_dir(bead_id: str) -> Path | None:
    p = REPO / "experiments" / bead_id
    return p if p.is_dir() else None


def parse_bead_id_from_file(path: Path) -> str | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"<!--\s*bead:\s*([A-Za-z0-9_-]+)\s*-->", text)
    if m:
        return m.group(1)
    m = re.search(r"^bead:\s*([A-Za-z0-9_-]+)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1)
    return None


def cited_artifacts(text: str, bead_id: str | None, doc_path: Path | None) -> list[Path]:
    found: list[Path] = []
    # implicit dir
    if bead_id:
        d = implicit_dir(bead_id)
        if d:
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lstrip(".") in ("txt", "json", "jsonl", "log", "md"):
                    # only data-ish artifacts; md allowed here for bead implicit dirs? keep data suffixes
                    if p.suffix in (".txt", ".json", ".jsonl", ".log"):
                        if p not in found:
                            found.append(p)
    # explicit
    for m in ARTIFACT_RE.finditer(text):
        raw = m.group(0)
        bases = [REPO]
        if doc_path is not None:
            bases.insert(0, doc_path.parent)
        for base in bases:
            p = (base / raw).resolve()
            if p.is_file() and p not in found:
                found.append(p)
                break
    return found


def artifact_numbers(path: Path) -> set[str]:
    out: set[str] = set()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
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


def check_text(text: str, bead_id: str | None, doc_path: Path | None) -> list[str]:
    label = bead_id if bead_id else (str(doc_path) if doc_path else "<stdin>")
    findings: list[str] = []
    if bead_id and bead_id in grandfathered():
        return findings

    artifacts = cited_artifacts(text, bead_id, doc_path)

    # wildcard check
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"[A-Za-z0-9_./*-]*\*[A-Za-z0-9_./*-]*\.(?:txt|json|jsonl|log)", line):
            findings.append(f"{label}:{lineno}: wildcard citation {m.group(0)!r}")

    if not artifacts:
        findings.append(f"{label}: cites no artifact file -- every number in it is unprovenanced")
        return findings

    supported: set[str] = set()
    for a in artifacts:
        supported |= artifact_numbers(a)

    for lineno, line in enumerate(text.splitlines(), 1):
        for m in DECIMAL_RE.finditer(line):
            num = m.group(1)
            if num not in supported:
                findings.append(
                    f"{label}:{lineno}: claim {num} does not appear in any cited artifact "
                    f"({', '.join(a.name for a in artifacts)})"
                )
        sizes = [int(x) for x in re.findall(r"\b(\d{4,})\b", line)]
        for num, unit in re.findall(r"(\d+)\s*([KM])iB\b", line):
            sizes.append(int(num) * (1024 if unit == "K" else 1024 * 1024))
        for m in RUNCOUNT_RE.finditer(line):
            n, total = int(m.group(1)), int(m.group(2))
            if not sizes:
                findings.append(f"{label}:{lineno}: run count {m.group(0)!r} names no size")
                continue
            have = sum(committed_count(artifacts, f"trace_S{sz}_*.log") for sz in set(sizes))
            if total > have:
                findings.append(
                    f"{label}:{lineno}: claims {m.group(0)!r} but only {have} trace file(s) committed"
                )
    return findings


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bead", help="bead id (fetches from bd; also sets implicit citation dir)")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)

    bead_id = args.bead
    targets: list[tuple[str | None, Path | None, str]] = []
    # bead_id only, no files: fetch from bd
    if bead_id and not args.paths:
        text = bead_text(bead_id)
        findings = check_text(text, bead_id, None)
        for f in findings:
            print(f"FINDING: {f}")
        if findings:
            print(f"\n{len(findings)} unprovenanced numeric claim(s)")
            return 1
        print(f"OK: bead {bead_id}, all numeric claims have provenance")
        return 0

    if not args.paths:
        # default: nothing; print usage
        print("usage: check_bead_prose_numbers.py [--bead ID] [file ...]", file=sys.stderr)
        return 2

    all_findings: list[str] = []
    for raw in args.paths:
        p = Path(raw)
        text = p.read_text(errors="replace")
        bid = bead_id if bead_id else parse_bead_id_from_file(p)
        fs = check_text(text, bid, p)
        all_findings.extend(fs)
    for f in all_findings:
        print(f"FINDING: {f}")
    if all_findings:
        print(f"\n{len(all_findings)} unprovenanced numeric claim(s)")
        return 1
    print(f"OK: {len(args.paths)} document(s), all numeric claims have provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

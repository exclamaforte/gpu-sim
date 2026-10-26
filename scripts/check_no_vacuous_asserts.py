#!/usr/bin/env python3
"""No vacuous assertions in the test suite (gpu-sim-va).

Why this exists: an apply pass turned the gate green by replacing three real assertions with
`assert True` -- the checks that disagreed with the new model were DELETED rather than re-derived.
The suite went from red to green while losing exactly the coverage that had caught the problem. No
check in this repo noticed; the reviewer did, and only by reading the diff.

A `assert True` (or `assert 1`, `assert "x"`, `assert []` is not truthy but `assert [0]` is) is never
a test: it is a comment that fails the "does this test anything" question. Legitimate vacuity does not
exist -- if a test cannot assert something, it should be removed or turned into an explicit skip with
a reason, both of which are visible in review.

Exit 1 and list every offending file:line. Run by the gate as a HARD check.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# assert <constant> where the constant is unconditionally truthy, or trivially equal on both sides
PATTERNS = [
    (re.compile(r"^\s*assert\s+True\s*(,.*)?$"), "assert True"),
    (re.compile(r"^\s*assert\s+1\s*(,.*)?$"), "assert 1"),
    (re.compile(r"^\s*assert\s+\"[^\"]*\"\s*$"), "assert <non-empty string literal>"),
    (re.compile(r"^\s*assert\s+(\w+)\s*==\s*\1\s*$"), "assert x == x"),
    (re.compile(r"^\s*assert\s+(len\([^)]*\)\s*>=\s*0)\s*$"), "assert len(...) >= 0"),
]

# An explicit skip with a reason is honest; a vacuous assert is not. Allowed exceptions must name
# a bead so the exception is auditable.
ALLOW = {}


def main() -> int:
    bad = []
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for pat, label in PATTERNS:
                if pat.match(line):
                    if f"{rel}:{n}" in ALLOW:
                        continue
                    bad.append((rel, n, label, line.strip()))
    if not bad:
        print(f"no vacuous assertions in {len(list(TESTS.rglob('*.py')))} test files")
        return 0
    print(f"{len(bad)} vacuous assertion(s) -- a test that cannot fail is not a test:")
    for rel, n, label, text in bad:
        print(f"  {rel}:{n}: {label}: {text}")
    print("Fix by deriving the expected value from docs/model_facts.json (or recomputing it) and")
    print("asserting that; if the property genuinely no longer holds, rewrite the test to assert the")
    print("NEW property rather than deleting the check.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

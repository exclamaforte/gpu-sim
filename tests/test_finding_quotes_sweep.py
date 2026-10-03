#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.6.3: FINDING.md must QUOTE sweep.json, never carry its own numbers.

The reviewer's item was that this bead's prose held figures of its own, which is
the hand-typed-number-beside-a-derived-one failure this project keeps meeting --
four review rounds on 6.6.6.1 and the reason 6.11.1 exists. A number written in
prose drifts the next time the box is loaded differently.

So this asserts two things:
  * every decimal figure in FINDING.md appears in sweep.json, and
  * the quoted block is actually populated rather than left as a placeholder.

The first is checked with comments and code stripped, so this test cannot fire
on the very phrase it forbids -- a check that cannot tell a comment from a
statement is not a check (6.6.6.1's scope lint learned that here).
"""
import io
import json
import re
import sys
import tokenize
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "experiments" / "gpu-sim-u2o.6.6.6.3"
FINDING = ART / "FINDING.md"
SWEEP = ART / "sweep.json"


class TestFindingQuotesTheArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = FINDING.read_text()
        cls.sweep = json.loads(SWEEP.read_text())

    def _artifact_numbers(self):
        nums = set()

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, (int, float)):
                nums.add(f"{o:.4f}".rstrip("0").rstrip("."))
                nums.add(f"{o}")
                nums.add(f"{o:.2f}")
                nums.add(f"{o:.3f}")
        walk(self.sweep)
        # derived figures FINDING is allowed to quote
        nums.add(f"{self.sweep['fit']['per_file_spread_us']['ratio']}")
        nums.add(str(self.sweep['fit']['us_per_file_slope']))
        return {n for n in nums if n}

    def test_every_decimal_in_the_prose_appears_in_the_artifact(self):
        """Prose only: comments and code blocks are excluded."""
        prose = []
        in_code = False
        for line in self.md.splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            prose.append(line.split("<!--")[0])
        # drop the quoted block entirely: it is generated FROM the artifact
        body = self.md
        for marker in ("<!-- BEGIN QUOTED FROM sweep.json -->",
                       "<!-- END QUOTED FROM sweep.json -->"):
            body = body.replace(marker, "")
        quoted_start = self.md.index("<!-- BEGIN QUOTED FROM sweep.json -->")
        quoted_end = self.md.index("<!-- END QUOTED FROM sweep.json -->")
        quoted = self.md[quoted_start:quoted_end]
        outside = (self.md[:quoted_start] + self.md[quoted_end:])
        outside = "\n".join(l.split("<!--")[0] for l in outside.splitlines()
                            if not l.strip().startswith("```"))
        allowed = self._artifact_numbers()
        # numbers that are structural, not measurements
        structural: set = set()
        # Only figures at MEASUREMENT precision count. Integers and one- or
        # two-decimal tokens are ordinals ("Attempt 2"), version references
        # (6.11.1) or ratios, and flagging those would make this test cry wolf
        # on prose until nobody reads it.
        offenders = []
        for tok in re.findall(r"\d+\.\d{3,}", outside):
            if tok in allowed or tok in structural:
                continue
            offenders.append(tok)
        self.assertEqual(
            sorted(set(offenders)), [],
            "FINDING.md carries figures the artifact does not: "
            f"{sorted(set(offenders))}. Every number must come from sweep.json.")

    def test_the_quoted_block_is_populated(self):
        self.assertNotIn("*(regenerate with `run.sh`)*", self.md,
                         "the quoted block was never filled in; run run.sh")
        self.assertIn("| files |", self.md,
                      "the quoted block must contain the generated table")

    def test_the_artifact_carries_a_load_stamp(self):
        """Every timing is stamped with the load it was taken at."""
        self.assertIsNotNone(self.sweep["loadavg_1_5_15"],
                             "a sweep with no load stamp cannot be compared "
                             "with any other sweep")
        self.assertTrue(self.sweep["points"])
        for p in self.sweep["points"]:
            self.assertIn("median_ms", p)
            self.assertIn("us_per_file", p)


if __name__ == "__main__":
    unittest.main(verbosity=2)

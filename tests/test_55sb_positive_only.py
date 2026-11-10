"""A ledger row that cannot fail is not evidence (gpu-sim-55sb).

Three separate rows reached review in one night with `mutation: "n/a"` /
`verification: "positive_only"` (e7yw, 7k6e, trcz). The reviewer caught all
three by reading the JSON, which is the checker's job. The cause was the
checker's own door: `mutation not in ("n/a", "doc-only")` exempted the row from
the mutation requirement, so the defect was sanctioned rather than reported.

These tests drive the REAL `check_ledger_test_ids` with injected ledger entries
(monkeypatching `load_ledger_entries`), not a reimplementation of the rule. A
test that reimplements the check proves the reimplementation -- that is exactly
how vmq2 and 7k6e each burned a review cycle, and it is the failure mode this
file is written to avoid.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_append_only.py"

# A product test node that resolves, so the row reaches the 55sb rules instead of
# failing earlier on node normalisation.
NON_DOC_NODE = "test_cju4.TestCju4.test_manifest_has_no_duplicate_signatures"
# A mutation that is genuinely in the registry, so the registry check passes and
# the positive_only rule is the one that must bite.
REGISTERED_MUTATION = "5uu9_power_arm"


def _load_checker():
    spec = importlib.util.spec_from_file_location("cao_55sb", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPositiveOnlyIsRefused(unittest.TestCase):
    def setUp(self):
        self.cao = _load_checker()
        # Child mode: exercise the RULES without applying a real mutation to the
        # worktree. The rules are what this file tests; applying 5uu9_power_arm
        # here would mutate tracked files to prove a point about JSON.
        self.cao._child_mode = lambda: True

    def _errors_for(self, **row):
        entry = {"name": "scratch_row", "test": NON_DOC_NODE,
                 "state": "passing", "artifact": "docs/none.md"}
        entry.update(row)
        self.cao.load_ledger_entries = lambda root: [entry]
        return self.cao.check_ledger_test_ids(str(ROOT))

    def test_mutation_na_is_refused_for_a_row_binding_real_code(self):
        errors = self._errors_for(mutation="n/a")
        self.assertTrue(errors, "a row with mutation 'n/a' must not pass silently")
        self.assertIn("mutation 'n/a' is refused", errors[0])
        self.assertIn("scratch_row", errors[0])

    def test_doc_only_is_derived_from_the_bound_path_not_declared(self):
        errors = self._errors_for(mutation="doc-only")
        self.assertTrue(errors, "doc-only must be checked against the bound path")
        self.assertIn("not documentation", errors[0])

    def test_positive_only_is_refused_for_a_row_binding_real_code(self):
        errors = self._errors_for(mutation=REGISTERED_MUTATION,
                                  verification="positive_only")
        self.assertTrue(errors, "positive_only must not pass for a code-binding row")
        self.assertIn("'positive_only' is refused", errors[0])

    def test_a_grandfathered_row_is_still_reported_as_debt(self):
        # The six rows that landed before the rule stay checkable rather than
        # silently passing: they are named, and the checker says so on a clean run.
        self.assertIn("1q34_lab_clean", self.cao.POSITIVE_ONLY_GRANDFATHERED)
        reason = self.cao.POSITIVE_ONLY_GRANDFATHERED["1q34_lab_clean"]
        self.assertIn("gpu-sim-1q34", reason)

    def test_a_mutation_plant_row_still_passes_the_new_rules(self):
        # Control: the new rules must not refuse the shape they are asking for.
        errors = self._errors_for(mutation=REGISTERED_MUTATION,
                                  verification="mutation_plant")
        self.assertEqual(errors, [], f"a properly mutated row was refused: {errors}")


if __name__ == "__main__":
    unittest.main()

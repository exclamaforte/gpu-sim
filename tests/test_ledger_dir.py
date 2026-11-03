"""Ledger directory structure guard (gpu-sim-u6ih).

Verifies the per-file ledger directory loads correctly and that the json file
is metadata-only (entries migrated to tests/fs2y_ledger.d/).
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "tests" / "fs2y_ledger.d"
LEDGER_JSON = ROOT / "tests" / "fs2y_ledger.json"


class TestLedgerDir(unittest.TestCase):
    def test_directory_has_entries(self):
        self.assertTrue(LEDGER_DIR.is_dir())
        entries = list(LEDGER_DIR.glob("*.json"))
        self.assertGreater(len(entries), 0)

    def test_json_entries_empty(self):
        data = json.loads(LEDGER_JSON.read_text())
        self.assertEqual(data.get("entries", []), [])

    def test_all_entries_load(self):
        for fn in sorted(LEDGER_DIR.glob("*.json")):
            entry = json.loads(fn.read_text())
            self.assertIn("name", entry)
            self.assertIn("test", entry)

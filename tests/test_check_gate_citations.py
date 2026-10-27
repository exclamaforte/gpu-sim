import unittest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_gate_citations as cgc


class TestCheckGateCitations(unittest.TestCase):
    def test_main_runs(self):
        rc = cgc.main()
        self.assertIn(rc, (0, 1))

    def test_tip_regex(self):
        m = cgc.TIP_RE.search("tip abc1234")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "abc1234")


if __name__ == "__main__":
    unittest.main()

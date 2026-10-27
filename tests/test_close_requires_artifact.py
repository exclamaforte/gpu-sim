import json, tempfile, unittest
from pathlib import Path
from unittest import mock
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import close_requires_artifact as cra


class TestCloseRequiresArtifact(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.out = self.root / "experiments" / "gpu-sim-u2o.15" / "close_audit.json"
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self._patches = [
            mock.patch.object(cra, "REPO", self.root),
            mock.patch.object(cra, "OUT", self.out),
            mock.patch.object(cra, "gather_with_proven_coverage",
                             lambda: ([], {"coverage_proven": True,
                                           "degraded_to_open_only_query": False})),
            mock.patch.object(cra, "audit",
                             lambda issues: {"violations": [], "violations_unacknowledged": [],
                                            "satisfies_the_acceptance": True}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.td.cleanup()

    def test_default_does_not_write_or_modify(self):
        self.out.write_text('{"existing": true}\n')
        before = self.out.read_text()
        with mock.patch.object(sys, "argv", ["x"]):
            rc = cra.main()
        self.assertEqual(rc, 0)
        self.assertEqual(self.out.read_text(), before)

    def test_write_creates_file(self):
        # remove file
        if self.out.exists():
            self.out.unlink()
        with mock.patch.object(sys, "argv", ["x", "--write"]):
            rc = cra.main()
        self.assertEqual(rc, 0)
        self.assertTrue(self.out.exists())
        data = json.loads(self.out.read_text())
        self.assertIn("satisfies_the_acceptance", data)


if __name__ == "__main__":
    unittest.main()

"""gpu-sim-m64 + gpu-sim-hje: the comments stay honest after the fix.

The defects fixed here were BOTH comment defects with live consequences: a
citation to a tool that exists nowhere (sizeof_check), and an overlay-write
count a reader could not verify. These two assertions make a recurrence fail
instead of lingering.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "src" / "fakegpu_cuda.c"


class TestCommentHygiene(unittest.TestCase):
    def test_no_phantom_sizeof_check(self):
        text = FAKE.read_text()
        self.assertNotIn("sizeof_check", text,
                         "comment cites a tool that exists nowhere (hje)")

    def test_overlay_write_count_is_19(self):
        writes = re.findall(r"memcpy\(base \+ \d+", FAKE.read_text())
        self.assertEqual(len(writes), 19,
                         f"overlay write count moved: {len(writes)}; the "
                         "test comment states 19 and the sizeof guards "
                         "assume full coverage")

    def test_both_sizeofs_named_in_comments(self):
        text = FAKE.read_text()
        self.assertIn("696", text)
        self.assertIn("1008", text)


if __name__ == "__main__":
    unittest.main()

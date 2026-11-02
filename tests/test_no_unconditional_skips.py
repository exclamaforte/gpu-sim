"""Guard: no unconditional skips in tests/ (gpu-sim-1486)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def test_no_unconditional_skip_decorators():
    offenders = []
    for p in sorted(TESTS.glob("test_*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("@unittest.skip(") or s.startswith("@pytest.mark.skip("):
                offenders.append(f"{p.name}:{i}: {s}")
    assert not offenders, "unconditional skip decorators found:\n" + "\n".join(offenders)

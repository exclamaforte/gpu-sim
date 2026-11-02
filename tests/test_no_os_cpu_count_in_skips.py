"""Guard: no test skip depends on os.cpu_count() (gpu-sim-ynaj AC5)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def test_no_os_cpu_count_in_tests():
    offenders = []
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name == "test_no_os_cpu_count_in_skips.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "os.cpu_count()" in line:
                offenders.append(f"{p.name}:{i}: {line.strip()}")
    assert not offenders, "os.cpu_count() found in tests (use effective_cpus):\n" + "\n".join(offenders)

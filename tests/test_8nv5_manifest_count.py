"""Guard: gate_manifest unit (gpu-sim-8nv5)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
from gate_manifest import compare


def test_clean():
    missing, undeclared = compare(["a", "b"], ["a", "b"])
    assert missing == []
    assert undeclared == []


def test_missing():
    missing, undeclared = compare(["a", "b"], ["a"])
    assert missing == ["b"], missing
    assert undeclared == [], undeclared


def test_undeclared():
    missing, undeclared = compare(["a"], ["a", "b"])
    assert missing == [], missing
    assert undeclared == ["b"], undeclared

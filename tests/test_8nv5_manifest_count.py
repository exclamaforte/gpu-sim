"""Guard: gate_manifest unit (gpu-sim-8nv5 / tn81)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
from gate_manifest import compare


def test_clean():
    missing, undeclared, duplicates = compare(["a", "b"], ["a", "b"])
    assert missing == []
    assert undeclared == []
    assert duplicates == []


def test_missing():
    missing, undeclared, duplicates = compare(["a", "b"], ["a"])
    assert missing == ["b"]
    assert undeclared == []
    assert duplicates == []


def test_undeclared():
    missing, undeclared, duplicates = compare(["a"], ["a", "b"])
    assert missing == []
    assert undeclared == ["b"]
    assert duplicates == []


def test_duplicate():
    missing, undeclared, duplicates = compare(["a", "b"], ["a", "a", "b"])
    assert missing == []
    assert undeclared == []
    assert duplicates == ["a"]

"""Plant test for AC3: a synthetic matmul row that discriminates old vs new rule."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "gpu-sim-8zqr"))
sys.path.insert(0, str(ROOT / "experiments" / "gpu-sim-4r9k"))
from gen_ac3 import old_usable, new_usable


def _synthetic_matmul_row():
    # within_std_us (0.5) > residual (0.004), so old threshold 5*0.5=2.5
    # > new threshold 5*0.004=0.02. kernel_us=1.0 falls between them:
    # old rule rejects, new rule accepts -> this row FLIPS.
    return {
        "family": "matmul",
        "kernel_us": 1.0,
        "within_std_us": 0.5,
        "residual_us": 0.004,
        "grid": [1, 1, 1],
        "grid_capped": False,
    }


def test_plant_flips_under_old_vs_new():
    row = _synthetic_matmul_row()
    assert old_usable(row) is False, "old rule should reject the plant"
    assert new_usable(row, "matmul") is True, "new rule should accept the plant"


def test_committed_rule_leaves_real_corpus_unflipped():
    """On the real 1jf corpus the committed rule produces 0 flips (it cannot
    be made to flip by varying the rule in the discriminating band, because no
    real matmul row falls in it)."""
    import json
    rows = json.load(open(ROOT / "experiments" / "gpu-sim-1jf" / "metrics.json"))["rows"]
    flips = sum(1 for r in rows if old_usable(r) != new_usable(r, r.get("family", "")))
    assert flips == 0, f"expected 0 flips on real corpus, got {flips}"

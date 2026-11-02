"""gpu-sim-8zqr: generalization test reproducibility."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "gpu-sim-8zqr" / "run.sh"
STAGE = ROOT / "experiments" / "gpu-sim-8zqr" / "stageD_results.json"
sys.path.insert(0, str(ROOT / "experiments" / "gpu-sim-8zqr"))
import stageD  # noqa: E402


def _pf(label, deg, ci, mean):
    return {"label": label, "degradation_factor_gbt": deg,
            "paired_delta_ci": ci, "paired_delta_mean": mean}


class Test8zqr:
    def test_stageD_reproducible(self):
        r = subprocess.run(["bash", str(RUN)], cwd=str(ROOT), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert STAGE.exists()
        json.loads(STAGE.read_text())

    def test_all_better_tie_low_deg_generalizes(self):
        pf = {
            "a": _pf("BETTER", 1.2, [-0.2, -0.01], -0.1),
            "b": _pf("TIE", 1.5, [-0.1, 0.05], -0.02),
        }
        assert stageD.aggregate_verdict(pf, mde_exceeds_effect=False) == "GENERALIZES"

    def test_all_better_tie_high_deg_degradation(self):
        pf = {
            "a": _pf("BETTER", 1.2, [-0.2, -0.01], -0.1),
            "b": _pf("TIE", 3.0, [-0.1, 0.05], -0.02),
        }
        assert stageD.aggregate_verdict(pf, mde_exceeds_effect=False) == "GENERALIZES_WITH_DEGRADATION"

    def test_any_worse_does_not_generalize(self):
        pf = {
            "a": _pf("BETTER", 1.2, [-0.2, -0.01], -0.1),
            "b": _pf("WORSE", 3.0, [0.01, 0.2], 0.1),
        }
        assert stageD.aggregate_verdict(pf, mde_exceeds_effect=False) == "DOES_NOT_GENERALIZE"

    def test_mde_exceeds_effect_inconclusive(self):
        pf = {
            "a": _pf("BETTER", 1.2, [-0.2, -0.01], -0.1),
        }
        assert stageD.aggregate_verdict(pf, mde_exceeds_effect=True) == "INCONCLUSIVE"

    def test_family_order_invariance(self):
        pf1 = {
            "a": _pf("TIE", 3.0, [-0.1, 0.05], -0.02),
            "b": _pf("BETTER", 1.2, [-0.2, -0.01], -0.1),
        }
        pf2 = {
            "b": _pf("BETTER", 1.2, [-0.2, -0.01], -0.1),
            "a": _pf("TIE", 3.0, [-0.1, 0.05], -0.02),
        }
        assert stageD.aggregate_verdict(pf1) == stageD.aggregate_verdict(pf2)

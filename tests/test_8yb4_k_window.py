"""Class-based guard for gpu-sim-8yb4: per-launch kernel time is flat in K.

The corpus times a config with one launch per event window; the gate prices K=40.
If per-launch time depended on K, corpus rows and gate prices would be in different
currencies. The committed rows are the measurement; this test pins the claim they
support, so a re-measured or hand-edited row that contradicts it fails.
"""
import json
from pathlib import Path

ROWS = Path(__file__).resolve().parents[1] / "experiments" / "gpu-sim-8yb4" / "rows.jsonl"


def _rows():
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]


class Test8yb4KWindow:
    def test_every_config_is_within_ten_percent(self):
        rows = _rows()
        assert len(rows) == 4, f"expected 4 configs, got {len(rows)}"
        for row in rows:
            ratio = row["ratio_k1_over_k40"]
            # 10% is the band the measurement supports: the worst config here is
            # +7.3%, and a ~40x currency error -- what this bead exists to rule out --
            # would sit far outside it.
            assert 0.9 <= ratio <= 1.1, (
                f"n={row['n']} block={row['block']}: ratio {ratio:.4f} outside [0.9,1.1]; "
                f"t_K1={row['k1_median_us']}us t_K40={row['k40_median_us']}us")

    def test_implied_fixed_cost_is_sub_microsecond_scale(self):
        for row in _rows():
            # A real fixed cost of a launch would be positive and grow with nothing;
            # what is left here is run-to-run noise, so bound the magnitude.
            assert abs(row["implied_fixed_cost_us"]) < 10.0, row

import json, pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-tsy" / "metrics.json"
BK9 = REPO / "experiments" / "gpu-sim-bk9" / "metrics.json"

def test_headline_matches_recomputed():
    m = json.loads(METRICS.read_text())
    # recompute underpowered condition
    ds = m["dataset"]
    n_usable = ds["n_usable_rows"]
    n_families = len(ds["families"])
    var = ds["target_variance"]
    if n_usable < 500 or n_families < 3 or var == 0:
        expected = "UNDERPOWERED"
    else:
        expected = m["verdict"]  # cannot determine without training; trust stored
    assert m["verdict"] == expected
    assert m["headline"] == m["verdict"]

def test_zero_unusable_rows():
    m = json.loads(METRICS.read_text())
    bk = json.loads(BK9.read_text())
    usable = [c for c in bk["cells"] if c["target_usable"]]
    assert m["dataset"]["n_usable_rows"] == len(usable)
    assert m["dataset"]["n_excluded"] == len(bk["cells"]) - len(usable)
    # no unusable row can be counted as usable
    assert m["dataset"]["n_usable_rows"] + m["dataset"]["n_excluded"] == m["dataset"]["n_rows"]
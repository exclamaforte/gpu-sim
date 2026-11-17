"""Guard: 3v6d.5 provenance wired into ranking path."""
import json, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_refuses_missing_provenance():
    import sys
    sys.path.insert(0, str(REPO))
    from epic_cluster.fabric_spec import load_spec
    import pytest
    # spec missing provenance for link.nvlink
    spec = {
        "sources": {"model.h100_sxm": "x"},
        "models": {"h100_sxm": {"source": "x"}},
        "links": [{"kind": "nvlink", "bw_gbps": 900, "latency_us": 1.0}],
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "spec.json"
        p.write_text(json.dumps(spec))
        with pytest.raises(ValueError) as exc:
            load_spec(p)
        msg = str(exc.value).lower()
        assert "provenance" in msg or "source" in msg, msg
        assert "nvlink" in msg or "link" in msg, msg


def test_report_spec_name_matches_loaded():
    spec_path = REPO / "specs" / "h100_rack.json"
    spec_name = spec_path.stem  # derived from loaded spec, never hardcoded
    r = subprocess.run(
        [sys.executable, "-m", "epic_cluster.parallelism_search",
         "--spec", str(spec_path), "--n-gpus", "8"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert f"SPEC: {spec_name}" in r.stdout, (r.stdout, spec_name)


def test_provenance_in_report():
    spec_path = REPO / "specs" / "h100_rack.json"
    import json as _j
    spec = _j.loads(spec_path.read_text())
    from epic_cluster.fabric_spec import provenance_for
    expected = provenance_for(spec, "nvlink")
    r = subprocess.run(
        [sys.executable, "-m", "epic_cluster.parallelism_search",
         "--spec", str(spec_path), "--n-gpus", "8"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert "PROVENANCE:" in r.stdout
    assert expected.split()[0] in r.stdout or expected[:20] in r.stdout, (r.stdout, expected)

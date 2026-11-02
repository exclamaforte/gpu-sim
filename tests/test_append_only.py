"""Append-only guard tests."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_append_only.py"


def run_script():
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


class TestAppendOnly:
    def test_unretired_deletion_fails_naming_key(self):
        mf = ROOT / "docs" / "model_facts.json"
        backup = mf.read_text()
        d = json.loads(backup)
        assert "fabric_band_robustness" in d
        del d["fabric_band_robustness"]
        mf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on unretired deletion"
            assert "fabric_band_robustness" in out, out
        finally:
            mf.write_text(backup)

    def test_retired_deletion_passes(self):
        mf = ROOT / "docs" / "model_facts.json"
        backup = mf.read_text()
        d = json.loads(backup)
        assert "fabric_band_robustness" in d
        del d["fabric_band_robustness"]
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        d.setdefault("_retired_fact_blocks", [])
        d["_retired_fact_blocks"].append({
            "key": "fabric_band_robustness",
            "reason": "test retirement",
            "commit": commit,
        })
        mf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc == 0, out
        finally:
            mf.write_text(backup)

    def test_ledger_unretired_deletion_fails_naming_entry(self):
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        names = [e["name"] for e in d["entries"]]
        assert "dhfo_memory_check" in names
        d["entries"] = [e for e in d["entries"] if e["name"] != "dhfo_memory_check"]
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on unretired deletion"
            assert "dhfo_memory_check" in out, out
        finally:
            lf.write_text(backup)

    def test_ledger_retired_deletion_passes(self):
        lf = ROOT / "tests" / "fs2y_ledger.json"
        backup = lf.read_text()
        d = json.loads(backup)
        names = [e["name"] for e in d["entries"]]
        assert "dhfo_memory_check" in names
        d["entries"] = [e for e in d["entries"] if e["name"] != "dhfo_memory_check"]
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        d.setdefault("retired", [])
        d["retired"].append({
            "name": "dhfo_memory_check",
            "reason": "test retirement",
            "commit": commit,
        })
        lf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc == 0, out
        finally:
            lf.write_text(backup)

    def test_retirement_missing_reason_fails(self):
        mf = ROOT / "docs" / "model_facts.json"
        backup = mf.read_text()
        d = json.loads(backup)
        assert "fabric_band_robustness" in d
        del d["fabric_band_robustness"]
        import subprocess as sp
        commit = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 text=True).strip()
        d.setdefault("_retired_fact_blocks", [])
        d["_retired_fact_blocks"].append({
            "key": "fabric_band_robustness",
            "commit": commit,
        })
        mf.write_text(json.dumps(d, indent=2) + "\n")
        try:
            rc, out = run_script()
            assert rc != 0, "expected failure on missing reason"
        finally:
            mf.write_text(backup)

"""Guard: gate_stage_report prints ids and copies the log (gpu-sim-3s8w)."""
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate_stage_report.py"


def test_failing_ids_printed_and_copied():
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        td = Path(td)
        log = td / "stage.log"
        # 60 lines: FAILED ids near the top, passing tail -> tail alone hides them
        body = ["line %d" % i for i in range(1, 31)]
        body.append("FAILED tests/a.py::test_one")
        body.append("FAILED tests/b.py::test_two")
        body += ["passing line %d" % i for i in range(1, 31)]
        log.write_text("\n".join(body) + "\n")
        outdir = td / "out"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--stage", "trunk refs agree",
             "--log", str(log), "--copy-to", str(outdir)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "FAILED tests/a.py::test_one" in r.stdout
        assert "full log:" in r.stdout
        copied = outdir / "trunk_refs_agree.log"
        assert copied.is_file(), "log not copied"
        assert "FAILED tests/a.py::test_one" in copied.read_text()


def test_no_failures_does_not_crash():
    with tempfile.TemporaryDirectory(dir=ROOT / ".lab" / "tmp") as td:
        td = Path(td)
        log = td / "stage.log"
        log.write_text("all good\n5 passed in 1.0s\n")
        outdir = td / "out"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--stage", "docs check",
             "--log", str(log), "--copy-to", str(outdir)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "full log:" in r.stdout

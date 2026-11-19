"""Guard: gate prints tail + surviving log path on a failing stage (gpu-sim-3s8w)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate.sh"


def test_failure_prints_tail_not_head():
    text = GATE.read_text()
    assert "tail -n 25" in text, "failing stage must print the tail (ids/summary), not the head"
    assert "sed -n '1,12p'" not in text, "must not print only the first 12 lines"


def test_failure_names_surviving_log_path():
    text = GATE.read_text()
    assert "full log: $run_log" in text or "full log: $GATE_RUN_LOG_DIR" in text or "run_log" in text
    assert "GATE_RUN_LOG_DIR" in text
    assert ".lab/gate-logs" in text


def test_run_log_dir_tip_timestamp():
    text = GATE.read_text()
    assert "gate_ts=" in text or "date -u" in text

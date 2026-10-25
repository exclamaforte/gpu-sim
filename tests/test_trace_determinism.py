"""The harness is DETERMINISTIC -- the instrument every byte comparison rests on (gpu-sim-det).

Every reconciliation claim in this repo compares a model against a harness trace. That is only
meaningful if the harness repeats: if one configuration produced different collectives on a second
run, "agrees to 2.5%" could be luck and a disagreement could be noise. Nothing had checked it.

Two runs of dp=4 / zero=2 / 3 steps, frozen in experiments/gpu-sim-det/, produce 115 records per
rank and IDENTICAL per-rank sequences. The cross-rank ORDER differs run to run -- four processes
appending to one file interleave differently -- which is why the comparison is per rank, not
line by line; comparing lines would report a false failure on scheduling noise alone.
"""
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PAIR = REPO / "experiments" / "gpu-sim-det"
SCRIPT = REPO / "scripts" / "check_trace_determinism.py"
FIELDS = ("op", "count", "bytes", "algorithm", "refused", "bottleneck_link", "regime")


def _per_rank(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.setdefault(rec["rank"], []).append(tuple(rec.get(f) for f in FIELDS))
    return {k: tuple(v) for k, v in out.items()}


def test_frozen_pair_is_deterministic():
    a = _per_rank(PAIR / "run1.jsonl")
    b = _per_rank(PAIR / "run2.jsonl")
    assert a == b, "the two recorded runs disagree -- byte reconciliation is comparing noise"
    assert set(a) == {0, 1, 2, 3}
    assert {len(v) for v in a.values()} == {115}, "per-rank record counts should be equal"


def test_line_order_differs_but_sequences_do_not():
    """Pin the reason the comparison is per rank: raw line order is scheduling noise."""
    lines_a = [l for l in (PAIR / "run1.jsonl").read_text().splitlines() if l.strip()]
    lines_b = [l for l in (PAIR / "run2.jsonl").read_text().splitlines() if l.strip()]
    assert lines_a != lines_b, (
        "the raw lines are identical in order -- then this test no longer demonstrates why the "
        "determinism check must be per rank")
    assert _per_rank(PAIR / "run1.jsonl") == _per_rank(PAIR / "run2.jsonl")


def test_script_agrees_and_rejects_a_different_config():
    ok = subprocess.run([sys.executable, str(SCRIPT), str(PAIR / "run1.jsonl"),
                         str(PAIR / "run2.jsonl")], capture_output=True, text=True)
    assert ok.returncode == 0 and "DETERMINISTIC" in ok.stdout, ok.stdout + ok.stderr

    other = REPO / "experiments" / "gpu-sim-juzb" / "trace_fixture.jsonl"
    if other.exists():
        bad = subprocess.run([sys.executable, str(SCRIPT), str(PAIR / "run1.jsonl"), str(other)],
                             capture_output=True, text=True)
        assert bad.returncode == 1, "a different configuration must not be reported as identical"

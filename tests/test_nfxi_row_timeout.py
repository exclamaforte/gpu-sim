"""gpu-sim-nfxi: a row timeout must never be readable as "the plant bit".

check_append_only.py's _run_pytest used to RE-RAISE subprocess.TimeoutExpired.  That escaped
into the generic "ledger test-id resolution: Command '[...]'" handler, so a slow machine
arrived as a ledger defect.  The damaging part is not the message: in the mutation branch a
bare rc != 0 is exactly what a SUCCESSFUL plant looks like, so a timeout could have been
recorded as a plant that bit -- evidence that does not exist.

This module forces the situation instead of waiting for a loaded box: it takes one REAL ledger
row that has a registered mutation (so the mutation-run branch is exercised, the branch that
matters), sets that row's timeout to a millisecond, and requires the checker to say TIMEOUT --
and to say nothing about the plant.

The row is resolved from the ledger directory rather than hardcoded, so the test does not rot
when a shard is renamed; if no row with a mutation exists the test fails loudly rather than
skipping (a skip here would be indistinguishable from a pass).
"""

import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "tests" / "fs2y_ledger.d"
CHECKER = ROOT / "scripts" / ("check_append" + "_only.py")


def _load():
    spec = importlib.util.spec_from_file_location("cao_nfxi", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row_with_mutation():
    for path in sorted(LEDGER_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        mut = data.get("mutation")
        if data.get("state") == "passing" and mut and mut not in ("n/a", "doc-only"):
            return data["name"], data["test"]
    raise AssertionError(
        "no passing ledger row with a registered mutation -- the timeout arm cannot be"
        " exercised, so this test would prove nothing")


def test_row_timeout_is_named_and_is_not_a_bite(monkeypatch):
    name, target = _row_with_mutation()
    mod = _load()
    mod.ROOT = str(ROOT)
    monkeypatch.setitem(mod.ROW_TIMEOUTS, name, 0.001)

    errors = mod.check_ledger_test_ids(str(ROOT))
    mine = [e for e in errors if name in e]
    assert mine, f"row {name!r} produced no error at a 1ms timeout -- it was not run"
    text = " ".join(mine)
    assert "TIMEOUT (infrastructure, not a ledger defect)" in text, (
        f"row {name!r} at a 1ms timeout did not name the timeout: {mine[:2]}")
    for wrong in ("did not fail", "did not pass", "restored target did not pass",
                  "Command '["):
        assert wrong not in text, (
            f"row {name!r} at a 1ms timeout was reported as {wrong!r}: {mine[:2]}")

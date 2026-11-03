"""Product-level ledger integrity guard (gpu-sim-oi4m).

Verifies that every row in tests/fs2y_ledger.json names a resolvable pytest
node. This is a PRODUCT test (it does not import or exec the checker), so it
can serve as a ledger target without being self-referential.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "tests" / "fs2y_ledger.json"


def _normalize(node):
    node = node.strip()
    if "::" in node:
        return node
    parts = node.split(".")
    if len(parts) == 2:
        return f"tests/{parts[0]}.py::{parts[1]}"
    if len(parts) == 3:
        return f"tests/{parts[0]}.py::{parts[1]}::{parts[2]}"
    return node


class TestLedgerNodeIds:
    def test_all_rows_resolve(self):
        data = json.loads(LEDGER.read_text())
        for entry in data.get("entries", []):
            node = entry.get("test")
            if not node or node == "doc-only":
                continue
            target = _normalize(node)
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", target],
                cwd=str(ROOT), capture_output=True, text=True, timeout=120,
            )
            assert r.returncode == 0, f"row {entry.get('name')!r}: {target} does not resolve"


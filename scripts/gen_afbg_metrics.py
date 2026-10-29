#!/usr/bin/env python3
"""Regenerate experiments/gpu-sim-afbg/metrics.json 'top_ranks' from model_facts.

Produced, not typed: top_ranks is derived from docs/model_facts.json ->
error_budget.ranked, the same list tests/test_error_budget_ranking.py reads.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import error_budget

FACTS = ROOT / "docs" / "model_facts.json"
METRICS = ROOT / "experiments" / "gpu-sim-afbg" / "metrics.json"

def main():
    facts = json.loads(FACTS.read_text())
    ranked = facts["error_budget"]["ranked"]
    metrics = json.loads(METRICS.read_text())
    cells = metrics["new_cells"]
    tr = error_budget.top_ranks_from_ranked(ranked, cells)
    metrics["top_ranks"] = tr
    metrics["gate"] = "n/a (CPU-only model extension, no GPU gate run); numbers verified by the docs/model_facts recompute and tests/test_error_budget_ranking.py"
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print("wrote top_ranks:", tr)

if __name__ == "__main__":
    main()

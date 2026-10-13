# gpu-sim-u2o.10.4 REPORT: three-way comparison from committed artifacts

CPU-only bead, no GPU used, no retraining (per packet guard).

## Table (family_holdout test, log-MAE, provenance-pinned)

| triple | roofline 9.4 | GBT 9.5 | transformer 9.1 |
|---|---|---|---|
| latency->latency | 0.5638 | 0.1072 | 0.1438 |
| latency->memory | 0.6873 | 0.0274 | 0.2600 |
| memory->memory | 1.1209 (own analysis) | -- | -- |

Verdicts: GBT wins both head-to-head triples; transformer beats roofline
~4x/2.6x. GBT/transformer cells recomputed from committed metrics.json by
join_104.py and asserted against the REPORT prose values (drift guard);
roofline cells quoted from 9.4's REPORT table with that provenance stated
(9.4's metrics.json carries regime analysis, not a holdout-MAE cell --
recorded, not hidden).

## Gaps (acceptance 3)

1. Ranking: no per-row predictions committed anywhere.
2. Calibration: same.
3. Generalization suites: 10.3 frozen, no suite artifacts.
4. memory->memory learned cells: absent from the holdout template.
All four in metrics.json gaps[]; filling any of them is 11.4/follow-up
work, explicitly not this packet.

## Backing doc

eval.md: only the 10.4 placeholder replaced; 10.3 untouched.

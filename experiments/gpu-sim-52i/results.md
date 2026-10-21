# gpu-sim-52i results — physical-core quiet window (SMT hypothesis)

**Pre-registered verdict: `NO-BOTH-QUIET-CORE-IN-20-SAMPLES`.**
Per pre-registration §5 no windows were run, and the consequence must be stated
precisely: **the SMT hypothesis is UNTESTED, not refuted.**

Executed by the PI; reviewed independently. `99g`'s verdict
(`NO-ADMISSION-CORRECTED-INSTRUMENT`) and `3ba`'s (`NO-ADMISSION-QUIET-CORE`) are
unchanged and are not reinterpreted here.

## 1. What was measured

Twenty samples of all 32 logical CPUs over 5 s, 30 s apart, spanning
**15:01:38Z → 15:12:43Z** (~11 minutes), under the 8-CPU quota. For each sample:
build the 16 `thread_siblings_list` pairs, compute every pair's **worst-sibling**
utilisation, take the best pair.

| quantity | value |
|---|---|
| samples | 20 |
| best-pair worst-sibling: min | **0.2500** |
| best-pair worst-sibling: median | 0.3476 |
| best-pair worst-sibling: max | 0.4830 |
| samples with both siblings below 0.25 | **0 of 20** |
| pairs that were ever "best" | cpu(7,23) and cpu(14,30) |
| windows run | **none** (pre-registered rule) |

The threshold is strict (`< 0.25`) and the best observed value **equals** it
(0.2500 — consistent with tick quantisation: 125 of 500 ticks over a 5 s sample).
So no pair qualified, and per §5 the pass stopped rather than measuring on a
half-busy pair and calling the result a limit.

## 2. What this does and does not establish

- **Does**: on this box, over an 11-minute window, **no physical core had both SMT
  siblings quiet**. Physical cores are the scarce resource here, not logical CPUs —
  which explains why everything upstream kept finding "quiet" resources that
  behaved noisily.
- **Does not**: test the SMT hypothesis. The prediction from `99g` — *if the
  bimodality is SMT sibling contention, windows on a both-quiet physical core will
  be consistently inside 36.2 µs* — was never put to the test, because the
  antecedent never held. **An untested hypothesis is not a refuted one, and this
  bead does not get to claim either way.**
- **Does not** change `m3d.10`'s `NO-CLOSE`. No gate run has occurred.

## 3. The honest state of the timing question

Across five beads the answer has narrowed to a single, precise, unresolved fork:

- A quiet **logical** CPU is obtainable (`3ba`: 0.2194) and on it **2 of 3 windows
  met the 36.2 µs requirement** (15.2, 25.3 µs) — but that instrument self-charged,
  so the gate refused them.
- With the self-charge removed (`99g`) the same logical CPU still produced **1 of 3**
  qualifying windows (35.1 µs), while its **SMT sibling sat at 0.94–1.0** — and the
  identical workload ran 5.9 ms in the tight window versus 3.3–5.2 ms in the loose
  ones, which is the signature of a sibling stealing issue slots.
- A both-siblings-quiet **physical** core did not occur in 20 samples (`52i`), so the
  SMT explanation could neither be confirmed nor eliminated.

The remaining question is therefore narrow and answerable without needing a quiet
pair: **does spread track SMT-sibling load?** That is a correlational test — run
windows across a range of sibling utilisations and see whether spread follows.
It is filed as the successor; if the association holds, the direction's timing
finding becomes mechanistic (*the criterion needs an exclusively-owned physical
core, which this shared box does not provide*), and if it does not, the floor is
intrinsic to the measurement path and that is the finding.

## 4. Reproduce

```bash
cd <repo root>
systemd-run --user --scope --property=CPUQuota=800% -- \
  python3 experiments/gpu-sim-52i/run.py
# Expect: 20 samples, then NO-BOTH-QUIET-CORE-IN-20-SAMPLES with the distribution
python3 experiments/gpu-sim-52i/check_statistic_agreement.py   # AGREEMENT OK
```

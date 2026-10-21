# gpu-sim-4tw results — model vs fabric under loss

**Pre-registered verdict: `MODEL-SUPPORTED`.** `gpu-sim-5zv`'s loss counterfactual
matches what the fabric actually does, at every test point, inside the 5 % tolerance
fixed before the run.

`5zv`'s verdict (`AC4 PARTIAL`: model route validated, plugin side refused on the
committed one-device fact) is unchanged.

## 1. Instrument check (ran before the sweep)

| check | expected | observed |
|---|---|---|
| `loss_ppm = 0`, 100 transfers | attempts 100, successes 100 | **100 / 100** |
| `loss_ppm = 1e6`, 100 transfers | attempts 100, successes 0 | **100 / 0** |

Counters: `fg_fab_attempt_count()` / `fg_fab_success_count()`, incremented on every
call and every delivery respectively. The check is code that runs *before* the sweep in
`test_retransmit.c`; a failure returns exit 2 and no ratio is reported.

## 2. The sweep

1000 deliveries per (loss_ppm, seed), 3 seeds, tolerance 5 % on the per-point median.

| loss_ppm | p | median ratio attempts/deliveries | model 1/(1−p) | rel. error | verdict |
|---|---|---|---|---|---|
| 0 | 0.00 | 1.000000 | 1.000000 | +0.00 % | WITHIN |
| 50000 | 0.05 | 1.047000 | 1.052632 | −0.54 % | WITHIN |
| 100000 | 0.10 | 1.092000 | 1.111111 | −1.72 % | WITHIN |
| 250000 | 0.25 | 1.314000 | 1.333333 | −1.45 % | WITHIN |
| 500000 | 0.50 | 2.005000 | 2.000000 | +0.25 % | WITHIN |

Per-seed values are in `sweep.txt` (identical to the run's raw `output.log`).
**Verdict: `MODEL-SUPPORTED`** — all five points within tolerance, largest deviation
1.72 %, and both extreme points (p = 0 and p = 0.5) within 0.25 % of the closed form.

Is the −1.72 % at p = 0.10 noise? Expected drops across 1092 attempts at p = 0.1 is
≈ 109 with sd ≈ 10, i.e. a ratio sd of ≈ 1 %; the observed deviation is ≈ 1.7 sd.
Consistent with sampling, and in any case inside the pre-registered tolerance — no
post-hoc tolerance change and no re-run to narrow it.

## 3. What this closes

`5zv` introduced the loss counterfactual — expected sends per delivered piece
`1/(1−p)`, applied to the wire term — when the fault matrix had **measurements without
a counterfactual**. This bead checks that counterfactual against the simulator's own
fault implementation and finds it **supported**:

- The model's fault predictions can now be trusted for reasoning about specs that
  cannot be measured on this box. The timing thread closed the **measurement** route
  (`r27`: criterion unobtainable without an exclusively-owned physical core); this
  validates the **model** route. Together they are the two halves of "what this lab
  can still assert on shared hardware".
- The fabric's loss implementation (drop per **transfer**, caller retries) is
  consistent with the model's assumption (drop per piece, retry). Had they differed,
  the mismatch would have been the finding — pre-registration §6.

## 4. What this does not claim

- No admission and no timing claim. This bead is not a timing measurement and is
  insensitive to the neighbour load that closed the timing thread.
- It does **not** validate the model against *hardware* loss behaviour — no real
  network with settable loss exists here. It validates model vs **our fabric**, the
  only loss source in this system. Calling that hardware validation would be false.
- It does not touch `n≥3` refusal, the measured pieces-per-phase table, or the
  categorical fault trichotomy.

## 5. Build note

The fabric is **not** exported from `libfakegpu.so` (that `.so` is the CUDA interposer);
`src/fakegpu_fabric.c` is linked directly, exactly as `tests/test_fabric.c` does. The
repo's "link the `.so`, not the sources" convention applies to the NCCL interposer whose
symbols *are* exported, not to the fabric. The test is wired into `make check`, so it
cannot silently stop running.

## 6. Provenance — two sessions worked this bead at once

The PI and the implementer both picked up `gpu-sim-4tw` and ended up in the **same
worktree**, so this directory holds two sessions' files. What that means for the
evidence, stated plainly:

- **There is one implementation, not two.** `test_retransmit.c` and
  `pre_registration.md` are the PI's; the implementer's commit `10f7720` committed those
  same files (they were present, uncommitted, when it entered the worktree) and ran them,
  producing `output.log`. `sweep.txt` and `output.log` are **identical**, as they must
  be: the fabric is seeded and the workload fixed, so the run is deterministic.
  The repetition therefore confirms **determinism and the committed seeds**, not
  independent replication — a second implementation would have been the stronger
  evidence and this is not that.
- The verdict does not depend on who ran it: same source, same seeds, same numbers.
- File provenance: `test_retransmit.c`, `pre_registration.md`, `metrics.json`,
  `sweep.txt` = PI; `job.json`, `run.sh`, `output.log` = implementer, kept as run
  provenance; `Makefile` carries both sessions' edits (implementer's `mkdir -p build`
  for clean-checkout reproducibility, PI's `test_retransmit` target).
- Operational lesson, recorded because it cost a review cycle: **one worktree, one
  session.** A worktree named after a bead is not a lock.

## 7. Reproduce

```bash
cd <repo root>
make check          # runs build/test_retransmit and prints VERDICT: MODEL-SUPPORTED
# or directly:
cc -O2 -Iinclude -o build/test_retransmit \
   experiments/gpu-sim-4tw/test_retransmit.c src/fakegpu_fabric.c -lpthread
./build/test_retransmit
```

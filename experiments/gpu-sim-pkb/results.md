# gpu-sim-eut: re-measured byte table (n=2) — regenerated from committed traces only

**Every number below is copied from `reduction_output.txt`, which is the verbatim output
of `python3 reduce_trace.py` on the 16 committed traces and 2 negative controls.** No
number in this file is hand-derived, hand-adjusted or selected from an uncommitted run.
This supersedes the previous version of this file, which the reviewer correctly rejected
for exactly those faults (see §6).

## 1. Where the measurement is usable — and where it is not

A trace is usable only if it passes the completeness gate **and** both ranks report the
same byte factor in the same run. There are **2 committed traces per size** (`m1`, `m2`),
and nothing else is cited.

| S (bytes) | trace | rank0 factor | rank1 factor | gate | usable? |
|---|---|---|---|---|---|
| 4096 | m1, m2 | 2.00 | 1.00 | passes | **no — ranks disagree** |
| 65536 | m1, m2 | — | — | **INVALID** (1.25S) | **no** |
| 131072 | m1, m2 | — | — | **INVALID** (1.25S) | **no** |
| 262144 | m1, m2 | — | — | **INVALID** (1.25S) | **no** |
| 393216 | m1 → 0.83S, m2 → 0.67S | — | — | **INVALID** | **no** |
| 524288 | m1, m2 | **1.00** | **1.00** | passes | **yes (2/2 runs)** |
| 1048576 | m1, m2 | **1.00** | **1.00** | passes | **yes (2/2 runs)** |
| 2097152 | m1, m2 | **1.00** | **1.00** | passes | **yes (2/2 runs)** |

**Usable data exists only at 512 KiB and above, and there the per-rank byte factor is
1.00.** Below 512 KiB there is no usable measurement: 4 KiB passes the gate but the ranks
disagree (2.00 vs 1.00), and 64–384 KiB fail the exact-multiple gate.

## 2. Verdict: `FACTOR-1.0-ABOVE-512KiB`; below that, UNMEASURED

- At 512 KiB, 1 MiB and 2 MiB — 2 runs each, both ranks agreeing — per-rank traffic is
  **exactly 1.00S**. The standard n=2 ring value is `2(n−1)/n·S = S`, so the
  measurement matches the textbook formula.
- **The small-size regime is not measured.** This bead does not claim a step exists, and
  it does not claim a step was refuted: there is simply no usable trace below 512 KiB.
  The earlier `NO-STEP` verdict asserted more than the data supports and is withdrawn.
- The 4 KiB row is the one hinge: rank0 sends 2S (2 pieces), rank1 sends S (1 piece), in
  **both** committed runs. That is a real, reproducible rank asymmetry, and it is why
  4 KiB cannot be used to argue either way.

## 3. What the extra send is — observed, not explained

At 64/128/256 KiB exactly one rank sends one piece more than S. Example from
`trace_S262144_n2_m1.log`, rank 0:

```
chan=0 seq=1 bytes=65536 digest=968c594b3d270383
chan=2 seq=2 bytes=65536 digest=968c594b3d270383
chan=4 seq=3 bytes=65536 digest=968c594b3d270383
chan=6 seq=4 bytes=65536 digest=968c594b3d270383
chan=2 seq=5 bytes=65536 digest=8a1eb5cb3ed80383   <- extra piece, DIFFERENT payload
```

**The extra send carries a different payload digest from the four before it**, so it is an
additional piece, not a repeat of one already sent. The previous version of this file
claimed it was "the same payload digest as seq 2" — that was false, and the reviewer
caught it. **What this extra piece is has not been determined.** Candidate explanations
(a second phase fragment captured without its siblings, an MTU-split tail, a harness
artifact) are not distinguished by the current traces.

## 4. Consequence for `gpu-sim-ggh`

`src/fakegpu_nccl.c:402–408` sets `chunk = (n == 2) ? 1.0 : (n-1)/n` and
`bf = 2.0*chunk`, commented *"bytes factor n=2 -> 2.0 (one whole buffer per phase)"*.

In the only regime this bead measures — 512 KiB and above — the measured factor is
**1.00**, not 2.0. So the model is wrong by 2× there, and the fix is to **delete the n=2
special case** so n=2 falls through to the general `2(n−1)/n` branch. This is **not** a
size-threshold fix: no threshold is measurable from the current data.

The n≥3 branch is untouched, and `epic_cluster/link_model.py` must agree with the C side.

## 5. Negative controls (both rejected, exit 2)

| control | defect | `reduce_trace.py` |
|---|---|---|
| `traces/negative_control_broken.log` | one `isend` line has no `digest=` field | `INVALID: 1 malformed isend line(s)` |
| `traces/negative_control_nonmultiple.log` | 6096 bytes for a 4096-byte buffer | `INVALID: rank 0: total_bytes 6096 is not an exact multiple of S 4096 (factor 1.4883)` |

The first matters most: the old parser **silently skipped** malformed lines, so a
truncated send vanished from the totals and the trace still looked valid. The gate is an
*exact multiple of S*, not a range — a range would have admitted the 1.25S traces as
valid, which is precisely how they entered the previous version's "clean" list.

## 6. What was wrong with the previous version of this file

Recorded because it is the same failure this bead exists to fix, one level down:

1. **Hand-typed numbers.** The 4 KiB row claimed a factor of 1.00 "after dedup" when
   the script reports 2.00 for rank 0. A manual adjustment is not a measurement.
2. **Unverifiable run counts.** The rejected version claimed more runs than it had
   committed; the historical claim, quoted verbatim:

   ```
   at S=524288: "6/6 runs" ... trace files of that size committed at the time: 2
   at S=1048576: "4/4 runs" ... trace files of that size committed at the time: 2
   ```

   Other runs existed on disk during the session and were never committed, so they
   cannot be cited. Counts are now taken from `git ls-files`.
3. **"Clean" rows that were invalid.** 64/128/256 KiB were listed as clean on the basis
   of runs that are not committed; all committed traces at those sizes fail the gate.
4. **A wrong mechanism claim.** "Same payload digest as seq 2" is false (§3).

The conclusion that survived all four corrections is narrower than the one I first wrote:
**1.00 at ≥512 KiB, and nothing measurable below it.**

## 7. Reproduce

```bash
cd experiments/gpu-sim-pkb
bash ../../experiments/gpu-sim-m3d.8/build.sh          # builds the plugin from THIS tree
for S in 4096 65536 131072 262144 524288 1048576 2097152; do
  for r in m1 m2; do ./run_measure.sh $S $r "traces/trace_S${S}_n2_${r}.log"; done
done
python3 reduce_trace.py traces/trace_S524288_n2_m1.log  # factor, or INVALID with exit 2
```

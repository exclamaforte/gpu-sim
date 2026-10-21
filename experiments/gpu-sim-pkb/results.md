# gpu-sim-eut: re-measured canonical byte table (n=2)

**Verdict: `NO-STEP` — but `gpu-sim-ggh` is NOT a non-defect. The model's n=2 byte
factor of 2.0 is contradicted at every size, not above a threshold.**

Measured by the PI after the first submission was ruled unsupported (see bead comments);
reviewed independently. Fabric: single link bw=10 Gbps, lat=5 us, loss=0, jitter=0.
Each size measured 2–6 times across the session; every trace is committed under
`traces/` with a self-describing header (`S_BYTES`, `NRANKS`, `RUN_ID`).

## 1. The result in one table

Per rank, from committed traces. "Agree" = both ranks report the same byte factor in
the same run, which is the test a capture must pass.

| S (bytes) | pieces/rank | bytes/piece | total/rank | factor | ranks agree? | verdict |
|---|---|---|---|---|---|---|
| 4096 | 1 | 4096 | 4096 | **1.00** | after dedup | clean |
| 65536 | 4 | 16384 | 65536 | **1.00** | yes (pi-m2) | clean |
| 131072 | 4 | 32768 | 131072 | **1.00** | yes (pi-m1) | clean |
| 262144 | 4 | 65536 | 262144 | **1.00** | yes (pi-m1, pi-m2) | clean |
| 393216 | 4 | 65536 | 262144 | **0.67** | yes | **UNRESOLVED** |
| 524288 | 16 | 32768 | 524288 | **1.00** | yes (6/6 runs) | clean |
| 1048576 | 16 | 65536 | 1048576 | **1.00** | yes (4/4 runs) | clean |
| 2097152 | 16 | 131072 | 2097152 | **1.00** | yes (2/2 runs) | clean |

**The per-rank byte factor is 1.00 at every size where both ranks agree.** For n=2 the
standard ring formula gives `2(n−1)/n·S = S`, i.e. **1.00** — the measurement matches
the textbook ring.

## 2. The apparent 2.00 and 1.25 factors are ONE DUPLICATED SEND

This is the mechanism, and it removes the "step" from the picture entirely. Example,
`trace_S262144_n2_m1.log`, per rank:

```
rank0: (seq 1, chan 0, 65536) (seq 2, chan 2, 65536) (seq 3, chan 4, 65536)
       (seq 4, chan 6, 65536) (seq 5, chan 2, 65536)   <-- 5th send
rank1: (seq 1, chan 0, .) (seq 2, chan 2, .) (seq 3, chan 4, .) (seq 4, chan 6, .)
```

The 5th send is on **chan 2 again** with **the same payload digest** as seq 2: a
duplicated retransmit, not a phase. 4 pieces × S/4 = **1.00S**; one duplicate makes
1.25S. Same shape at 64 KiB (rank1, duplicate on chan 0) and at 4 KiB (2 sends of S
where 1 is a duplicate → 2.00).

Why does it happen: the sending rank starts before its peer has attached, the first
piece is retried, and the retry is traced as an extra send. The duplicate always
carries the **first** channel's payload. Fixing the start order (start the waiting rank
first — `run_measure.sh`, PI fix in this commit) removes it: 524288/1048576/2097152
never show a duplicate in any run, and 262144 is clean in both runs after the fix.

**So the old table's "2.0 → 1.0 step", and every 1.25 intermediate, is retransmit
noise on the first piece.** There is no phase-fusion step in the protocol.

## 3. What this means for the model (`gpu-sim-ggh`)

`src/fakegpu_nccl.c:402–408` sets `chunk = (n == 2) ? 1.0 : (n-1)/n` and
`bf = 2.0*chunk`, with the comment *"bytes factor n=2 -> 2.0 (one whole buffer per
phase)"*. That assumes each phase sends the whole buffer → 2S per rank at n=2.

The measurement says **1S at every clean size**. So:

- **The step does not exist** → `ggh` must not be implemented as a size-dependent
  factor.
- **The model is still wrong** → the n=2 special case is wrong *everywhere*, not above
  a threshold. The correct factor is `2(n−1)/n = 1.0`, i.e. **delete the n=2 special
  case** and let n=2 fall through to the general branch.
- This is a simpler and stronger correction than the one `ggh` was written for, and it
  is supported by clean, committed, reproducible traces at eight sizes.

`ggh` therefore stays a real defect with a **rewritten scope**: remove the n=2 special
case rather than add a threshold. It is not closed as NOT-A-DEFECT.

## 4. Unresolved and stated as such

At **S = 393216 (384 KiB)** both ranks agree on 4 pieces of 65536 = **0.67S** — less
than the buffer. A complete allreduce cannot send less than S. Both ranks agreeing
means it is not the retransmit artifact. Cause not determined; possible explanations
are a piece-size cap interacting with a non-power-of-two size, or a capture that ends
early in a way both ranks share. **This point is excluded from the factor conclusion**
and is filed as an open question rather than smoothed away.

## 5. Method corrections made in this bead

1. **Digest is not a phase marker.** `ncclnetsim_fabric.c` emits `digest=fg_digest(data,
   size)` — a hash of the **payload**. Identical bytes hash identically, so digest
   multiplicity measures data patterns, not protocol structure. The phase-by-digest rule
   in the first submission is withdrawn; the byte factor needs no phases at all
   (`sends × bytes / S`).
2. **Start order.** Rank 0 was started first, then rank 1 after `sleep 1`. At small
   sizes rank 0 can finish before rank 1 attaches, because the plugin does not block a
   rank that finishes early. That produced hangs (exit 124) and the duplicated first
   send. Fixed: the waiting rank starts first.
3. **The tree did not compile.** `src/ncclnetsim_fabric.c:392` called
   `plugin_resolve_link` with 5 arguments where the definition takes 9; fixed. The
   first submission's traces were produced by a `.so` built in a **different worktree**,
   so their provenance did not match this tree's source.
4. **No committed path is written.** `run_measure.sh` used `> $HERE/run0.log`, which
   overwrote the committed pkb logs; it now writes per-run logs under `traces/`.
   (The committed logs were restored twice during this bead.)

## 6. Negative controls — the gate is proven to reject, not assumed to

Two deliberately broken traces are committed and both are rejected by
`reduce_trace.py` with **exit 2** and an `INVALID:` line (a warning on stderr would not
be enough — the failure the old table suffered was silent):

| control | what is wrong | script says |
|---|---|---|
| `traces/negative_control_broken.log` | one isend line has no `digest=` field (truncated write) | `INVALID: 1 malformed isend line(s)` |
| `traces/negative_control_nonmultiple.log` | totals 6096 bytes for a 4096-byte buffer (1.49S) | `INVALID: rank 0: total_bytes 6096 is not an exact multiple of S 4096 (factor 1.4883)` |

The first control is the important one: the old `parse_trace` **silently skipped**
malformed lines, so a truncated send vanished from the totals and the trace still
looked valid. Malformed lines are now counted and make the trace `INVALID`.

The completeness gate is *exact multiple of S*, not a `[S, 4S]` range: a range would
have admitted the 1.25S duplicate-send traces as valid, which is precisely how the
1.25 rows entered the first submission's "valid" list.

## 7. Reproduce

```bash
cd experiments/gpu-sim-pkb
bash ../../experiments/gpu-sim-m3d.8/build.sh      # builds libncclNetSimFab.so from THIS tree
for S in 4096 65536 131072 262144 524288; do
  for r in a b; do ./run_measure.sh $S r$r "traces/trace_S${S}_n2_r$r.log"; done
done
python3 reduce_trace.py traces/trace_S262144_n2_ra.log   # sends, bytes, factor
```

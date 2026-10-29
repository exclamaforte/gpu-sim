# Training-run parallelism simulator — current design

Written 2026-10-23. Describes what exists and how to run it. Supersedes an
earlier draft of this file that proposed an analytic FLOP-counting model; the
project's premise is the opposite of that, and the draft was wrong.

## 0. Where the project is right now (2026-10-24)

Read this first if you are picking the work up from disk rather than from a briefing.

**The headline answer depends on a convention, and the default one changes it.** PyTorch's context
parallel defaults to `rotate_method=ALL_GATHER` (`torch/distributed/tensor/experimental/
_context_parallel/_attention.py`, the `_RotateMethod.ALL_GATHER` default), whose rotater is documented as
"Allgather the kv and return only the required kv" -- that is, it gathers the FULL sequence K/V on every
rank. The model prices the opt-in ring form instead (`_AllToAllRotater`, "Use all_to_all to send the kv
to the next rank"). Measured on the harness, the all_gather form moves 1.99x the bytes the model charges
for. Because 1.99x exceeds the 1.52x radius of the TP+CP answer, the two conventions do not share a
recommendation:

Only ONE recommended config survives both conventions, and it is not the headline one:

| cell | ring `all_to_all` | PyTorch default `all_gather` |
|---|---|---|
| N=8, seq 8192 | (tp=4 pp=1 cp=2 dp=1) 88.49 ms, 6.03% | (tp=1 pp=8 cp=1 dp=1) 93.83 ms, 4.89% |
| N=8, seq 32768 | (tp=4 pp=1 cp=2 dp=1) 484.79 ms, 1.48% | (tp=4 pp=1 cp=2 dp=1) 498.44 ms, 3.21% |
| N=16, seq 8192 | (tp=1 pp=16 cp=1 dp=1) 50.66 ms, 4.94% | (tp=1 pp=16 cp=1 dp=1) 50.66 ms, 4.94% |
| N=32, seq 8192 | (tp=1 pp=16 cp=2 dp=1) 30.33 ms, 5.45% | (tp=1 pp=16 cp=1 dp=2 z2) 31.99 ms, 3.98% |
| N=32, seq 32768 | (tp=1 pp=8 cp=4 dp=1) 170.28 ms, 1.24% | (tp=1 pp=8 cp=1 dp=4 z2) 176.80 ms, 1.35% |

The N=16 answer is convention-INDEPENDENT because it uses no context parallelism at all (cp=1), which is
the strongest single reason to trust it. At 8 GPUs the config flips at seq 8192 but survives at 32768: the
ring answer's own margin there is thin (1.48%), and under all_gather the same config wins by more (3.21%)
because its rival is priced differently. At 32 GPUs the config changes at both lengths.

So "TP+CP is right at 8 GPUs" holds only if CP rotates K/V with all_to_all. Under the default stack the
best choice at seq 8192 is a pure pipeline with no tensor and no context parallelism at all -- 6.0%
slower in absolute terms, because the model is not the only thing that changed: the cheaper config became
expensive, not the other way round. Any config quoted from this document must therefore carry both its
batch assumption (below) and its CP convention.

**Robustness radius (gpu-sim-zj9a).** The N=8 verdict (tp=4 pp=1 cp=2 dp=1) is unchanged for CP
cost scaled by k in [0.5415, 1.5244] (two-sided: a 0.54x underestimate flips to (2,1,4,1,0), a
1.52x overestimate flips to (1,8,1,1,0)); the N=16 verdict is unchanged for PP-bubble cost scaled
downward (no flip within the sweep); it flips UP at 1.86x to (1,8,2,1,0). CP is the
least-verified term. The radius is computed by `scripts/sensitivity_radius.py`.

**Micro-batch count (gpu-sim-oggn).** The default `micro_batches = pp` (one micro-batch per
stage) is the pessimistic edge of the realistic band and is unmeasured (provenance
"assumed (no measurement)"). Sweeping micro-batches per stage (factor = micro_batches/pp) over
{1,2,4,8,16,32}:

| micro-batches/stage | N=8 verdict | N=16 verdict |
|---|---|---|
| 1 | (4,1,2,1,0) | (1,16,1,1,0) |
| 2 | (1,8,1,1,0) | (1,16,1,1,0) |
| 4 | (1,8,1,1,0) | (1,16,1,1,0) |
| 8 | (1,8,1,1,0) | (1,16,1,1,0) |
| 16 | (1,8,1,1,0) | (1,8,1,2,2) |
| 32 | (1,8,1,1,0) | (1,8,1,2,2) |

N=8 flips at 2 micro-batches/stage; N=16 holds for up to 8 micro-batches/stage (flips at 16). The published N=8 ring recommendation
therefore holds only for 1 micro-batch per stage (factor 1); at 2+ the winner becomes
(tp=1 pp=8 cp=1 dp=1). The realistic default is ungrounded: no repository config documents a
grad-accum/micro-batch setting for these models, and the only documented value is the model
default `micro_batches = pp` (parallelism_search.py:334-336), so the headline verdict is
conditional on that assumption. The table is `scripts/microbatch_sweep.py` /
`experiments/gpu-sim-oggn/metrics.json`.

**Error budget (gpu-sim-hd5u).** The full per-parameter table is `scripts/error_budget.py`
(`docs/model_facts.json` -> `error_budget`), ranked by the one-sided fraction of the plausible range
consumed (the full-width fraction is published alongside; the two orderings disagree, so both
columns are kept). At seq 8192 the tightest margin is `microbatch_factor` (5.95%); then
`term_scale_cp` at N=32 (15.79%) and N=8 (17.48%); then `term_scale_pp` at N=8 (23.26%).
At seq 32768 the ranking changes: the most fragile parameter is `term_scale_pp` at N=16
(1.40% one-sided), then `recompute` at N=16 (1.50%) and `act_factor` at N=16 (1.78%);
`microbatch_factor` is 15.57% at N=16 and does not flip at N=32. The 'weakest assumption is
micro-batching' claim is therefore 8192-specific; at long sequences the CP/PP terms are tighter. `achieved_bw_scale` also flips the
N=8 verdict (0.814132, 12.39% full-width / 37.17% one-sided), so bandwidth is NOT a robust
parameter. Measured on the 5090 (gpu-sim-3fog): 1505.19 GB/s = 0.83995 of the 1792 GB/s spec, above both flip points (0.814132 at N=8, 0.824585 at N=16/32768), so the headline verdicts survive; bandwidth becomes 7.6% (N=8) / 4.5% (N=16/32768) and `microbatch_factor` (5.95%) is the new rank-1 at N=8. Extended to N=64 and N=128 (gpu-sim-afbg): the N=8 and N=16 headline cells are unchanged
(`microbatch_factor` 5.95% at seq 8192, `term_scale_pp` 1.40% at seq 32768), but the ranking is
NOT size-stable. At seq 8192 the top row at both N=64 and N=128 is `term_scale_cp`; at seq 32768
N=64 is also `term_scale_cp` and only N=128 is `term_scale_pp`. So a rank-1 claim has to name its
N as well as its sequence, and at the largest sizes the brittle input is the fabric term -- the
one input that cannot be calibrated on a single GPU. Fabric precision requirement (gpu-sim-b1r6): at N=64/seq8192 `term_scale_cp` flips at 0.424x (down) and 1.763x (up) of the fitted value; N=128/seq8192 flips at 1.911x (up only); N=64/seq32768 flips at 1.233x (up only); N=128/seq32768 does not flip in range. Pinning the fabric makes `term_scale_pp` the next rank-1 in every cell. Rows that do not flip in either sequence (`mem_headroom`, `mem_charge`, the eff classes)
are reported with their worst-case margin over the whole range instead of a flip value, so they are
not empty. At seq 8192 `recompute` and `act_factor` do not flip; at seq 32768 both do --
`recompute` at 0.015 (1.50% of its range) and `act_factor` at 2.277 (1.78%) at N=16 -- which is
the other half of why the ranking changes.

**The PP bubble default is pessimistic, and every winner survives a realistic one.** The model splits
each step into `micro_batches = pp`; deployments pick `micro_batches` as large as memory allows, which
shrinks the bubble (bubble = compute x (pp-1)/(micro_batches+pp-1)). Sweeping that ratio under the default
all_gather convention at seq 8192, the winners do not move for micro_batches/pp in [1, 8] and their
margins IMPROVE: N=8 stays (tp=1 pp=8 cp=1 dp=1) at 93.83 -> 70.95 ms and 4.89% -> 10.1%; N=16 stays
(tp=1 pp=16 cp=1 dp=1) at 50.66 -> 40.92 ms; N=32 stays (tp=1 pp=16 cp=1 dp=2 z2) at 31.99 -> 29.36 ms,
6.29% at the best setting. Only an extreme 16x (16 micro-batches per stage at pp=16) moves the shape --
N=16 drops to (tp=1 pp=8 cp=1 dp=2 z2) at 41.49 ms, still a 1.02% tie rather than a different answer. So the unmeasured bubble term biases
the model AGAINST the pipeline answers it recommends: giving it a realistic value makes them cheaper, not
more expensive.

**Beyond one rack (gpu-sim-scaleout).** The same model run on generated fabrics at 10, 100 and 1000 nodes
of 8x H100 (80 / 800 / 8000 GPUs, NVLink inside a node, one NDR NIC per GPU across) says: 80 GPUs is
pipeline-dominant, with 70B a real 4.85% verdict under the ring convention (tp=1 pp=40 cp=2 dp=1)
and a 0.11% tie under the default one (111.0 ms at tp=1 pp=20 cp=1 dp=4 z2); 8B at the same scale
is a 2.05% near-tie that the default convention breaks toward dp=5; 800 GPUs is decided by efficiency
and convention rather than by structure; and at 8000 GPUs the margins run 0.61% to 5.09%, so most cells
are ties the model cannot separate. 405B does not fit at 80 GPUs at all under the default fp32_master
optimizer style, and the reason is NOT activations: that style assumes fp32 master weights plus fp32 Adam m/v
plus bf16 grads, i.e. 14 B per parameter, and 405B spread over 80 GPUs is 5.07e9 parameters per GPU however
you decompose them -- 71.0 GB of optimizer state before a single activation. Recomputation cannot close a gap
that size; only more GPUs (7.1 GB of the same state at 800) or a different optimizer style can. The style is
now an explicit choice, hw['opt_state'] in {bf16_moments, fp32_moments, fp32_master} with hw['opt_peak'] in
{default, fused, none}, priced from measurement: bf16 moments reside at 6.0 B/param and peak at 10.0, fp32
moments at 12.0 steady with a 20.0 default peak and 16.0 fused peak, while fp32_master is the flat legacy 14
at every peak mode (it approximates fp32_moments steady plus a 2 B margin and underestimates peak). 'none'
prices steady state only and is optimistic: a scheduler is limited by peak. Under bf16 moments the 80-GPU floor
is 50.7 GB and 405B FITS: (tp=1 pp=40 cp=1 dp=2 z2) at 504 ms and 75.1 GB, a 3.52% tie, so the feasibility
verdict belongs to the style, not the model. With fp32 moments the same floor is 60.9 GB steady (which fits,
but steady is not what OOMs), 81.2 GB fused peak -- 1.5% over the line with no allocator overhead left -- and
101.5 GB default peak, clearly out. **Measured optimizer bytes (gpu-sim-37xk, RTX 5090, torch 2.13.0+cu132, 184.6M params, 8 steps median).**
The assumed style_bytes are now anchored: bf16 moments peak 8.36 B/param fused (10.36 with foreach transient
vs 10.0 assumed) and 4.36 params+grads; fp32 moments peak 16.36 B/param fused (20.36 foreach, 17.09 foreach=False)
vs 16.0/20.0 assumed, and 8.36 params+grads. Recomputing the 405B @80 floor from the measured fused peak gives
83.1 GB (was 81.2 assumed) -- still 3.9% over, so the 'does not fit' verdict holds and is now measurement-backed
rather than assumption-backed; the bf16 verdict (50.7 GB) and the default-peak verdict (103.3 GB) are unchanged
in direction. The fused-vs-foreach gap is real (10.36 vs 8.36 B/param for bf16, 20.36 vs 16.36 for fp32): a
foreach implementation peaks higher, so the fused number is the optimistic (best case) floor. Bytes/param is a
storage property, so extrapolating this 5090 measurement to 80 H100-class GPUs is legitimate for bytes only --
it is NOT a claim about 405B speed or throughput. At 800 GPUs the binding term does become activations, which is why the answer there wants pp=100 (2.58%) and why recomputation is
the right next modelling step -- for that regime, not for 80 GPUs. The knob now exists in the model:
hw['recompute'] is a float 0.0..1.0, default 0.0, keeping (1 - alpha(seq)*recompute) of the stored activations and
charging 1.25x compute when fully on -- both numbers measured on this host's 5090
(experiments/gpu-sim-recompute/), not cited. alpha is 0.67 at seq 8192 and above (the MHA-block operating value
every published grid runs at, so the default path is unchanged); below 8192 it follows the GQA re-measure
(commit 03ad3e4): 0.176 at seq 2048, 0.420 at seq 4096, 0.594 at seq 8192, linear between, flat below 2048. Where
the knob matters is where activations bind: 70B at 80 GPUs with batch=32 goes from (tp=1 pp=4 cp=1 dp=20 z3)
at 2.515 s (0.53%, 56.3 GB) without recompute to (tp=1 pp=1 cp=1 dp=80 z3) at 2.639 s (14.35%, 69.2 GB) with
full recompute -- no pipeline at all, pure FSDP across all 80 GPUs, above the 2.474 s the chargeless proxy
reported by the 1.25x charge, and far better separated than the tie it replaces. For 405B the knob changes
nothing at 80 GPUs (still infeasible with full recompute); at 800 the winners do not move (still pp=100 at
0.138 s ring / 0.180 s default) and at 8000 the same winners hold at 0.063 s / 0.092 s. Those rows price one
optimizer style: fp32 master plus fp32 Adam, 16 B/param all-in with the bf16 weights and grads. Plain
bf16 AdamW keeps no separate master copy and measures 10 B/param on this host (experiments/gpu-sim-recompute/);
under that style the 80-GPU boundary moves, which belongs to gpu-sim-scup. The knob changes no published
number: it defaults off, and for 405B the 80-GPU verdict is unchanged. Every published pp>1 row likewise prices full
activation storage (recompute defaults off even where the spec assumes it on): all 31 fit at recompute=0, the
tightest being 70B at 80 GPUs under the default convention (tp=1 pp=20 cp=1 dp=4 z2) at 8.03 GB/stage against
the 80 GB budget -- so the published steps carry no recompute compute charge by demonstrated feasibility, not
by omission (gpu-sim-vvty). Full tables, both conventions and batch=1/batch=8 in `experiments/gpu-sim-scaleout/`.

**BATCH=1 per rank for every published number.** All published step times, margins, and winners
assume `batch=1` token per rank per step (`MICRO_BATCHES=pp` for PP). The DP gradient sync is
charged per STEP (once per step regardless of batch size) while tp/cp/pp payload comm scales with
BATCH (per token), so `batch=1` flatters DP-avoiding configs: a real batch>1 run shifts the
optimum toward more DP. Re-derive with `batch>1` before quoting any number for a production run.

**LINK RATE CORRECTED: the spec's 900 GB/s is AGGREGATE BIDIRECTIONAL, and every collective here
runs one way (gpu-sim-smt8).** `specs/h100_rack.json` says it in its own provenance string ("NVLink
4: 900 GB/s per GPU (aggregate bidirectional)"), but `link_model` used it as a per-direction rate --
exactly 2x optimistic, and exactly 2x off the legacy `nccl_cost` path, whose fabric table had used
450 GB/s all along (spec all-reduce 31.25 ms vs legacy 62.46 ms for the same 8B/bf16/dp=8
collective). `_eff_bw` now halves a link that declares `"bw_is_bidirectional": true`, and the flag is
PER LINK because the fabrics are quoted differently in the same file: NVLink is two-way aggregate,
NDR IB "400 Gb/s = 50 GB/s per GPU NIC" is already per-direction. Halving both -- the first version
of this fix -- would have made the IB links half as fast as the hardware.

The correction is validated by the two pricing paths converging: at N=8 the spec path now reads
**88.49 ms** against the legacy path's **88.14 ms** (0.4% apart; before the fix they were 79.96 vs
90.19, 15% apart). Consequences are NOT cosmetic:

  * N=8 keeps (tp=4 pp=1 cp=2 dp=1 z0) but 88.49 ms instead of 79.96, and the best dp>1 row is now
    (tp=1 pp=4 cp=1 dp=2 z2) at 99.71 ms;
  * **N=16 runs no tensor parallelism**: the winner is (tp=1 pp=16 cp=1 dp=1) at 50.66 ms, a
    two-layers-per-GPU pipeline with no data parallelism and no ZeRO -- and N=32 runs the same
    pp=16 shape with cp=2 (tp=1 pp=16 cp=2 dp=1) at 30.33 ms;
  * the 70B-on-32-GPUs best feasible moves 1.55 s -> 1.12 s, because the search now promotes a pp=32
    pipeline over the old dp-sharded answer (the DP term of a dp=8 row is 62.47 ms);
  * every comm term doubles, so the DP term of a dp=8 row is 62.47 ms, not 32.30.

That TP stops paying at 16 GPUs is the substantive result: once the link rate is honest, TP's
all-reduce costs more than the pipeline bubbles it removes except at 8 GPUs, where cp=2 absorbs the
long-sequence cost.

**CURRENT ANSWER (one screen).** For Llama-3-8B on 8xH100 at seq 8192-32768 the
recommendation is **(tp=4 pp=1 cp=2 dp=1 zero=0)** -- tensor parallelism 4 by context
parallelism 2, NO data parallelism and NO ZeRO -- at **88.49 ms/step at 8192**, 193.48 at
16384 and 484.79 at 32768 on the spec/topology path (legacy single-fabric: 88.14 / 193.56 /
484.44). Runner-up at 8192 is (tp=1 pp=8 cp=1 dp=1 zero=0) at 93.83 ms; the best dp>1 row is
(tp=1 pp=4 cp=1 dp=2 zero=2) at 99.71 ms; and the best dp=8 rows (zero=2 and zero=3) tie
on time at 125.7 ms, with zero=2 losing on memory (64.3 vs 50.3 GB).

**CP PAYS AT SEQ 8192** (cp=2 is in the winner), so S_best -- the smallest seq where the
overall best feasible config has cp>1 -- is **8192 on both pricing paths** and under every
knob combination measured (crude/production cp-grad sync x activation factor 2.3/1.0). The
earlier claim that CP does not pay at 8192, and the 16384/32768 thresholds that came with
it, were artifacts of an understated DP term (below).

**Why the answer moved (gpu-sim-6r5, measured + fixed).** Two DP defects, both now fixed:
the DP term passed the dp-SHARD to a cost function whose `count` is the TOTAL payload
(`link_model._bw_factor = (n-1)/n * S/B`), making every dp>=2 collective dp x too cheap; and
it omitted the param all-gather that ZeRO-1/2 actually issue (the harness issues one per
parameter -- 37 per step at 4 layers, measured). Corrected, a dp=8 row pays 62.47 ms of DP
comm at 8192, while a dp=1 row pays none, so dp-heavy splits lose to tp/cp splits. This is
the strongest evidence yet that the previous recommendation was an accounting artifact, not a
property of the hardware.

**THE TP PAYLOAD IS NOW SETTLED, AND IT AGREES WITH THE MODEL (gpu-sim-6wd).** The TP
all-reduce payload was the last unvalidated load-bearing term. Two independent lines close it:

1. **Measured on the harness.** With the interposer on tp=2 and tp=4 (2 and 4 ranks, 5 steps, trace
   on, fresh libfakegpu.so) the TP all-reduce moves **1,048,576 elements per call = B*seq*d**
   (B=8, seq=512, d=256) at **both tp widths** -- d elements per rank per token,
   tp-independent. Shape probe: `(8, 512, 256)` at both all-reduce sites; varying the model shows
   the count is seq-proportional, per-layer, and independent of vocab and tp.
2. **Derived for Megatron.** Its tensor parallelism pairs a column-parallel fc1 with a
   row-parallel fc2, and the sharded `d_ff/tp` activation between them is consumed locally: **d_ff
   never crosses a rank boundary**. fc2's all-reduce is over its *output*, width d. Attention is
   the same shape (qkv column, output projection row). Megatron's four per-layer all-reduces are
   therefore all width d -- which is exactly the `four_d` convention this model has always priced.

So the earlier version of this section was wrong twice, and both claims are withdrawn: the harness
does **not** crash on tp>1 (both runs exited 0), and the `3d/tp` payload inherited from
gpu-sim-oez is **refuted** (it would have halved between tp=2 and tp=4; it did not). A third guess,
that the MLP all-reduce carries `d_ff` (2.25x), is refuted by the column/row pairing above and is
kept in the code as `df_allreduce_hypothesis` so the error stays reproducible. Consequences: the
`tp=8 cp=1` column and the claim that CP never pays up to 65536 rested on those refuted premises
and are gone; **`four_d` is the default**, the winner is **(tp=4 pp=1 cp=2 dp=1 z0)** at
**88.49 ms** (N=8), 50.66 (N=16), 30.33 (N=32), and **S_best = 8192** stands on TP grounds.

**Alpha granularity does not move the winner (gpu-sim-8nt follow-up).** The model charges
alpha once per *logical transfer* while a real NIC charges per *piece* (measured P=4-8 for
payloads >= 256 KiB), so it under-charges alpha on big payloads. Paying alpha P times instead
(`experiments/gpu-sim-8nt/alpha_piece_charging.py`) leaves the winner at (tp=4 pp=1 cp=2 dp=1 z0)
for P=1/4 at **88.49 / 90.8 ms**, and the reason is structural: ring alpha is
2*(n-1)*lat, so a wider TP group pays more of it -- at P=8 the pp=8 config takes the lead at 93.83 ms
against 93.88 ms for the tp=4 answer, a tie for ranking purposes. The ranking holds through P=4;
the absolute duration is not (+2.6% at P=4, +6.0% at P=8).

What is still NOT settled about this row: absolute durations (gpu-sim-wv2 needs real GPUs), and the
harness runs fp32 where the model prices bf16 -- compare element counts with the harness, not bytes.

**How much the absolute durations are worth, measured rather than assumed (gpu-sim-wv2, partial).**
The compute term divides 6*P*T FLOPs by dense peak with NO efficiency factor, which works out to an
implied **78.6% MFU** at this row. That is unrealistically high, and it is now measured rather than
argued: on the shapes this model prices (8192x4096x4096 plus the MLP pair), a real GPU reaches
**52-57%** of its dense bf16 peak -- 229/237/218 TFLOP/s on an RTX 5090 (scripts/measure_mfu.py), and
published large-model bf16 training runs sit at 35-50%. Feeding those back in, N=8 reads **113.07 ms**
at 0.566 and **149.5 ms** at 0.40, i.e. the absolute number carries a factor-1.28-to-1.69 uncertainty.

The reason this does not undermine the answer: **the winner is invariant across the whole range.**
(tp=4 pp=1 cp=2 dp=1 z0) wins at N=8, (tp=1 pp=16 cp=1 dp=1) at N=16 and (tp=1 pp=16 cp=2 dp=1)
at N=32 at 0.786, 0.566 AND 0.40 MFU -- three different efficiencies, one ranking
(tests/test_mfu_sensitivity.py). Absolute comm durations still rest on the model: only a 2-4 GPU
NVLink run can validate those.

**The margins are what matter, not just the argmin.** Measured against the best config with a
DIFFERENT shape (a zero=2/zero=3 tie is the same answer, not a rival): at N=8 the margin is **6.03%**
at the model's implied 78.6%, **7.53%** at the measured 56.6% and **5.70%** at a pessimistic 40%; at
N=32 it is **5.45% / 4.19% / 3.12%**. Both answers hold across the whole plausible efficiency range --
N=8 now comfortably, N=32 by a thinner but real margin. (These numbers are produced by the claims
register below, not typed by hand; when the CP payload was corrected to the GQA K/V width the N=8
margin roughly doubled while N=32 lost its old 4-14% cushion.)

**N=16 became a verdict when the PP payload was priced correctly, and N=32 stopped being one.**
At 16 GPUs the answer is (tp=1 pp=16 cp=1 dp=1) -- a two-layers-per-GPU pipeline with no data
parallelism at all -- and it is the SAME config at 78.6%, 56.6% and 40% (4.94% / 3.33% / 2.06%), so it
is stable rather than a near-tie. At 32 GPUs the margins are 5.45% / 4.19% / 3.12% at seq 8192, but the
31k/32k cells fall to 1.24% / 0.96% / 0.02%: the model cannot separate the candidates there.

**Decisiveness depends on sequence length at least as much as on GPU count.** At 8 GPUs the headline
answer is decided by 6.03% / 7.53% / 5.70% at seq 8192 and 1.48% / 1.11% / 0.81% at 32768, but seq 16384
is a tie (0.22% / 0.17% / 0.13%). Any statement of the form "TP+CP is right at 8" is therefore a
statement about seq 8192 and 32768, not about 8 GPUs in general.

**How wrong would the decisive term have to be?** The N=8 verdict is decided by the CP term, which the
harness cannot check by construction; the N=16 verdict is decided by the PP bubble, which no measurement
covers. Re-running the production model with that one term scaled, the N=8 answer is unchanged for CP
error in **<!-- RADIUS-INLINE:BEGIN -->[0.54x, 1.52x]<!-- RADIUS-INLINE:END -->** (and <!-- RADIUS-INLINE:BEGIN -->[0.20x, 4.81x]<!-- RADIUS-INLINE:END --> at seq 32768), and the N=16 answer for PP-bubble
error in **<!-- RADIUS-INLINE:BEGIN -->[unbounded below (no flip down to the 0.05x sweep floor), 1.86x upward]<!-- RADIUS-INLINE:END -->**.
Those are the honest confidence bounds on the two verdicts: one term error of that size overturns the
answer, and a 4x CP error has already been found on this axis once. The CP term can now in principle
be checked: `tools/torch_parallel` supports grouped-query attention (--n-kv-heads), so
`kv_dim = n_kv_heads x head_dim = 1024` -- the quantity the CP term is proportional to -- has a
counterpart in the harness (gpu-sim-g8wc). The PP bubble remains unmeasured.

**And that check was run the next hour: the convention gap is 1.99x, which is bigger than the radius.**
Three frozen cp=2 traces (experiments/gpu-sim-cpgqa) give the same answer at kv_dim 256 and 1024 and at
half the sequence: the harness gathers the FULL-sequence K/V on every rank and moves **1.99x** what the
model's ring exchange prices (1.99x and 1.98x respectively). Ring and all_gather are both legitimate --
one ships shards, the other replicates -- but **1.99x sits outside the 1.52x radius** of the N=8 verdict.
Which convention production uses is now answerable from source, not measurement:
- **PyTorch CP** (`torch.distributed.tensor.experimental._context_parallel`, default `rotate_method =
  ALL_GATHER`) gathers the full K/V on every rank via `ft_c.all_gather_single` (see `_AllGatherRotater`).
  That is the **all_gather** convention -- the same one this harness implements -- so the harness matches
  the PyTorch default and the model understates CP by the measured 1.99x.
- **Megatron-LM** context parallelism uses ring rotation (point-to-point exchange of K/V shards), i.e.
  the **ring** convention the model prices; DeepSpeed-Ulysses also shards K/V and gathers only the
  needed slice.
So the verdict is **conditional in a way the reader can resolve from their stack**: PyTorch-default
deployments get the larger CP cost (and "TP+CP is right at 8 GPUs" may not survive); Megatron-style
deployments keep the 1.52x radius. The document no longer has to say "we cannot answer" -- it says
"check which one your stack is". Citation: PyTorch `_context_parallel/_attention.py`, lines ~273-295
(`_AllGatherRotater`) and `set_rotate_method` (default `ALL_GATHER`); harness side `tools/torch_parallel/model.py`
Block CP path uses `torch.distributed.nn.all_gather`.

**The other fitted constant, checked the same way.** `act_factor` (2.3) came from a single-shape
instrumented measurement and the model itself calls it "unvalidated for exact fit", so it deserved
the same suspicion. It turns out not to be a time knob at all: activations enter MEMORY, while the CP
term prices a ring exchange of K/V chunks and the DP/TP terms price parameters and messages, so
sweeping act_factor 1.0 to 12.0 -- a 5x band around the fit -- leaves every step time bit-identical
and every winner unmoved at N=8, 16 and 32 (at seq 8192). "No bend" is a STEP-TIME statement: act_factor
multiplies activation MEMORY, so it cannot enter a time term. It does bite on FEASIBILITY, and the
seq-32768 N=16 verdict flips on feasibility when act_factor falls to 2.277, which is 1%
below the fitted 2.3. The measured curve resolves the straddle: act_factor rises with sequence length
(1.735 at 2048, 2.128 at 4096, 2.226 at 8192, 2.329 at 16384, 2.345 at 32768) with
zero spread at fixed length across 3 reps, so the mxaz straddle was driven by SEQ LENGTH, not
run-to-run noise. At seq 8192 the measured 2.226 sits below the 2.277 flip (verdict flips);
at seq 32768 the measured 2.345 sits above it (safe). Shape dependence (gpu-sim-006j): the curve is shape-dependent at short sequence (48% across 3 shapes at seq 2048) but converges at long sequence (1.0-1.3% at seq 16384-32768), so the single-curve simplification is valid where the budget uses it (N16_seq32768) and invalid at short sequences. The model stops being feasible at all
beyond ~36 / ~82 / ~174 at seq 8192. The
recommendation therefore does not rest on this fit
(tests/test_act_factor_knob.py).

**Where the recommendation is a coin flip (fragility map).** Margin of the winner against the best
config with a different shape, over N x sequence length x efficiency:

| N | seq 8192 | seq 16384 | seq 32768 |
|---|---|---|---|
| 8 | 6.03 / 7.53 / 5.70% | **0.22 / 0.17 / 0.13%** | **1.48 / 1.11 / 0.81%** |
| 16 | 4.94 / 3.33 / 2.06% | **0.45 / 0.02 / 0.64%** | **0.34** / 2.20 / 9.15% |
| 32 | 5.45 / 4.19 / 3.12% | **1.95 / 1.18 / 0.56%** | **1.24 / 0.96 / 0.02%** |

(78.6% / 56.6% / 40% MFU; bold is a margin under 2%.) Two things fall out and neither was visible in
the single-cell analysis. First, the fragility is a LONG-SEQUENCE phenomenon, not an N=16 quirk: at
8192 only N=16 at a pessimistic efficiency is tight, but from 16384 the N=8 answer itself wins by
about 0.2%, and at 32768 it is 0.81% at 40% MFU. The config changes there -- (tp=4 cp=2 dp=1)
wins at 8192 and 32768 but (tp=2 cp=4 dp=1) at 16384 -- so the recommendation stands, but at long
sequences it is not decisively separated from the alternative. Second, N=16 at 16384 is a genuine
three-way tangle: the winner moves from (tp=1 pp=16) to (tp=2 pp=2 cp=4) as efficiency drops, and the
margin never exceeds 0.64%. N=32 is solid at 8192 (5.45% / 4.19% / 3.12%) but both longer sequences
contain ties -- 0.56% at 16384/typical and 1.24% / 0.96% / 0.02% at 32768.

**Where does CP actually start paying? At seq ~5956, not 8192.** "CP pays at 8192" has always read
like a boundary, but 8192 was simply the smallest grid point that was tried. Bisecting the sequence
length at which the best cp>1 config overtakes the best cp=1 config puts the crossover at **5956**
tokens at the model's implied efficiency (**3658** at the measured 56.6% and **3646** at 40%), and act_factor
does not move it at all (activations are memory, not time). So S_best=8192 is not a knife edge -- it
is above the crossover, where the CP advantage is already **6.03%** (**8.78% / 6.64%** at
measured/pessimistic efficiency). At seq 4096 the answer brackets zero (**-8.16%** at the implied efficiency,
**1.73% / 1.35%** at measured/pessimistic), which is what a crossover looks like. The K/V correction moved
the crossover up from its pre-correction value, and the parameter correction moved it to 5956 with it,
so the honest wording is "CP pays from ~6k tokens, and 8192 is the
first conventional length above that".

**Two cells recommend a config nobody would deploy.** The model treats HBM as a hard fit, so a
config at 91% of capacity looks as good as one at 52%. Requiring a mere 10% of HBM to stay free --
room for fragmentation, allocator rounding, activation spikes and the CUDA context -- changes exactly
**two** cells in the whole grid: N=16 at seq 32768, where the winner (tp=2 cp=8 dp=1) occupies **73.0 of
80 GB** and is replaced by (tp=2 pp=2 cp=4 dp=1 z0) at **41.5 GB** for **423.78 vs 413.98 ms**; and N=8
at seq 16384, where (tp=2 cp=4 dp=1) at **73.0 GB** gives way to (tp=4 cp=2 dp=1 z0) at **40.8 GB** for
**253.43 vs 253.00 ms**. Both cost about 2% or less, and the N=16 replacement is also the config that
wins at N=16 for seq 16384 at pessimistic efficiency, so the headroom requirement does not cost a real
recommendation -- it removes artifacts. Every other cell
is unchanged at 10% and even at 20% headroom. The model's default stays a hard fit (headroom 0.0),
because that is what the published numbers assume; `mem_headroom` is an hw knob and this is the
sensitivity it produces.

**N=16 and N=32.** At N=16 the winner is (tp=1 pp=16 cp=1 dp=1) at 50.66 ms -- a pure pipeline with
no data parallelism and no ZeRO to tie. At N=32 the winner adds cp=2 to the same pp=16 pipeline
(tp=1 pp=16 cp=2 dp=1) at 30.33 ms: a second split of the sequence dimension instead of a return
to data parallelism, because the DP term still costs more than the GQA-priced K/V exchange.

**What this does NOT say.** TP comm is priced from the same bytes-only ring formulas (per-step
TP message COUNT is harness-validated, sizes are Megatron-form: gpu-sim-5o4.7 / gpu-sim-oez),
and absolute durations are modelled, not measured (gpu-sim-wv2, blocked on 2-4 real GPUs). A
tp=4 cp=2 recommendation is more sensitive to the TP term than the old dp=8 one was; the next
gate on this answer is calibrating TP bytes, not the DP term.

The 8-rank spec covers N=8 fully and N=16 for all groups; N=32
RAISES ('rank 16 cannot be placed on spec'), so 32-GPU cells must pass
specs/h100_rack_4node.json explicitly or they silently fall back to the legacy path. By seq 65536 (N=8)
only 7 configs are feasible and by 131072 none
are, so CP goes from optional to mandatory somewhere in between. All of these are modelled,
not measured -- gpu-sim-54m is the study that pins them down and must report the path, the
cp_grad_mode and the act_factor alongside every threshold. At 70B on 32
GPUs the answer is still decided by MEMORY: only 10 of 161 configs fit at seq 32768, and the
best of them is (tp=1, pp=32, cp=1, dp=1, zero=0) at 1.12 s/step and 63.7 GB (a full-depth pipeline,
not a dp-sharded optimizer -- with 70B the optimizer state per rank dominates before
replication does). 70B on 8-16 GPUs is infeasible outright.

**CONFIDENCE.** Validated on hardware: collective COUNTS (exact, 16 tp all-reduces + 24 cp
K/V sends per rank per step), CP training correctness (loss and parameter equivalence), and
the S^2 score-storage memory term (15-20%). Modelled only: byte SIZES, fabric constants,
absolute times, and the activation memory factor (a fit at one shape). The ranking is a
NO-OVERLAP ranking, so every "X is faster" claim is conditional on comm not overlapping
compute; at full overlap 24 configs tie at the same floor, which is why the honest output of
the search is the FSDP recommendation plus a tie set, not a single winner. Read the sections
below for the derivation of each number; nothing here is a measured duration.

| piece | state |
|---|---|
| axes DP / TP / PP / ZeRO1-3 | all implemented; all train correctly (verified against references, not against each other) |
| context parallel (CP) | implemented: sequence split into contiguous blocks, K/V exchanged over a p2p ring, cp groups. gpu-sim-5o4.3 (2d11df0). Gate: cp=2 forward maxerr 1.9e-8 vs cp=1. train.py still refuses cp>1 combined with pp>1 or zero>1 (unverified pairs). |
| collective cost | ANALYTICAL: n-1 hops of S/n per ring phase, latency vs bandwidth chosen by whether a hop outlasts link latency (`epic_cluster/nccl_cost.py`, mirrored in `src/fakegpu_nccl.c`). Legacy table behind GPUSIM_NCCL_LEGACY_TABLE=1. |
| 5D search | `epic_cluster/parallelism_search.py` ranks (tp,pp,cp,dp,zero) at fixed N by predicted step time: compute+comm SUM (no overlap), with max-overlap reported separately. Those two policies disagree completely -- see below -- so the ranking is conditional. |
| Perfetto | `tools/torch_parallel/merge_profile.py` merges torch.profiler kernels/Python with the interposer's collectives; three rows per rank. |
| calibration | **counts** validated on hardware (gpu-sim-5o4.7): 16 tp all-reduces and 24 cp K/V sends per rank per step match the plan exactly. Byte SIZES are not validated: the harness's tp all-reduce carries a 3d/tp qkv tensor while the cost model assumes Megatron's full-d row-parallel payload, so the model's byte magnitudes are Megatron-form and modelled-only. Absolute durations modelled (gpu-sim-wv2, blocked: needs 2-4 real GPUs). |
| attention FLOPs | gpu-sim-ooj added attention (seq^2) fwd+bwd, 12*L*S^2*d per sequence. **The delivered scaling was wrong and its headline claim is WITHDRAWN (PI, gpu-sim-i95):** it divided by cp*tp but not pp*dp, while the weight term beside it divides by tp*pp*cp*dp, so compute stopped being split-invariant and tp/cp configs got a spurious discount -- which is where the 'cp=4 wins at S>=8192' result came from. Scaled like the weight term, 12*L*S^2*d/(tp*pp*cp*dp) = (12*L*S*d)*S/N, the term is split-invariant, raises every config's compute by ~27% at S=8192 and ~107% at S=32768, and the winner was claimed as pure FSDP pre-audit (73.06 ms at S=8192); after the DP audit the winner is (tp=1 pp=1 cp=1 dp=8 zero=2) see gpu-sim-6r5, while at S=32768 neither dp=8 config fits -- the best feasible is (tp=4 pp=1 cp=2 dp=1 zero=0) at 484.79 ms (compute floor 63.23 / 412.99 ms at 8192/32768; all 40 pp=1 rows tie at the max-overlap floor). Crossover where attention equals the weight FLOPs is S ~= P/(2*L*d) = 30.6k for 8B/d4096. Conservation test (compute identical for all configs with equal N) tracked in gpu-sim-i95. |
| rank layout (read before touching groups) | rank = ((pp_idx*tp + tp_rank)*cp + cp_rank)*dp + dp_rank. TP groups are stride cp*dp, PP neighbours keep (tp_rank, cp_rank, dp_rank) fixed. gpu-sim-x7h found tp groups built with stride 1 and no cp term, which silently disabled tp all-reduce on every rank with cp_rank>0; fixed in 4e8d65c. `tools/torch_parallel/validate_trace.py` now refuses a run whose ranks have unequal collective counts (stage-aware, and invalid rows yield no step time). |

THE RANKING IS A NO-OVERLAP RANKING, and that is the open question, not a footnote.
flops_per_gpu = (6*P + 12*L*seq*d)*seq/(tp*pp*cp*dp) = (6*P + 12*L*seq*d)*seq/N is
INVARIANT across the split at fixed N: the attention term changes the MAGNITUDE (about +27%
at seq=8192 and +107% at 32768, crossover at seq ~= P/(2*L*d) = 30.6k for 8B/d4096), never
the invariance. So every pp=1 row reports the same compute -- 51.65 ms weight-only, 64.97 ms
with attention at seq=8192 -- and the ORDER is decided entirely by the comm terms, not by
the dominant column, which reads compute everywhere. With the corrected terms (TP 4
all-reduces per layer per step; CP 2*(cp-1) K/V sends per layer) the ranking is the one in
section 0, and it was RE-RANKED by the DP-comm audit (gpu-sim-qk9: bf16 payloads, real ZeRO
semantics, one flat collective per op): (tp=1 pp=1 cp=1 dp=8 zero=2) first (historical),
(tp=1 pp=1 cp=1 dp=8 zero=3, FSDP) second (historical), (tp=2 pp=1 cp=1 dp=4 zero=2) third at
76.50 ms -- all at 8B/8192, attention on, spec path. The pre-audit text quoted FSDP at
73.06 ms (spec) / 73.05 ms (legacy) and tp=2 cp=4 second; those numbers are superseded, not
merely rounded, because the audit changed which config WINS. gpu-sim-olg (P1) adds the CP
gradient sync (8jg) to the cost model: every cp>1 row pays a full all-reduce of P/(tp*pp)
params over cp ranks, so cp rows drop behind the dp=8 rows; the best cp>1 row at N=8 is
(4,1,2,1,zero=0) at 78.90 ms. The winner is zero=2 at N=8/16/32 (66.99 / 47.02 / 26.50 ms,
FSDP zero=3 second (historical). **Correction: S_best does not depend on how the cp sync is modelled, but the SHAPE of the winner
does.** With the production variant (cp_grad_mode="production") the winner at N=8 is
(tp=2 pp=1 cp=4 dp=1 z0) at 84.28 ms against the crude answer's (tp=4 pp=1 cp=2 dp=1 z0) at
92.85 ms -- different tp AND different cp (production now charges RS+AG per flat bucket, gpu-sim-juzb,
so its cp_grad is 6.93 ms against the crude all-reduce's 9.24 ms). Both put CP in the winner at 8192, which is what "CP pays
at 8192" claims, but quoting one config as THE answer without its cp_grad_mode is wrong. The
published answer is the CRUDE (conservative) one, and this is the second knob after act_factor that
has to travel with any config quoted from this section. That mode is HYPOTHETICAL for now -- it assumes an optimizer
that never reconstructs the gradient. The harness's reduce-scatter path (gpu-sim-fzd) still
all-gathers the gradient, so ring-allreduce == reduce-scatter + all-gather: its measured
traffic equals the crude path byte for byte, and the production figure is unreachable until
a sharded optimizer exists (gpu-sim-fzd, closed). What DOES depend on it is the size of the CP penalty
(+27.7 ms crude vs +3.5 ms production at tp=2 cp=4), which is why the harness must state
which sync it does before any CP claim is quoted. gpu-sim-fzd (P2) adds --cp-grad-sync=reducescatter
(equivalence-preserving: measured bytes per rank are EQUAL to allreduce because all_gather
reconstructs the full gradient; the true production sync needs a sharded optimizer this
harness lacks). Equivalence verified: identical loss curves (5.760->5.743, 5.695->5.680).
Model cp_grad stays CRUDE (conservative); production cp_grad (2*(cp-1)/cp * bytes/cp) is
hypothetical. cp>1 configs
still appear in the top-5 by step_sum because several tie at the max-overlap floor (e.g. N=8
top-5 includes (4,1,2,1,0) at 90.19 ms with cp_grad 9.24 ms), but none is strictly better than
FSDP. See experiments/gpu-sim-olg/top5.json. But ALL 24 feasible pp=1 configurations have identical
max-overlap time (64.97 ms with attention; 22 of 24 at 51.65 ms weight-only, where two rows'
comm exceeded the floor) while their SUM spans 66.99-113.59 ms with attention at N=8 (the pre-audit span
73.06-129.62 ms superseded by gpu-sim-qk9): under full overlap the
model cannot separate them at all. Numbers in this section are weight-only unless labelled;
gpu-sim-i95 regenerates every table with the attention term. gpu-sim-5o4.5 quantified the
overlap question on the merged search: the min-comm config (pure FSDP) is the unique winner
for overlap fraction f < 0.25,
then TIES with a growing set (2 at f=0.25, 7 at f=0.55, 22 at f=1) at the 51.626 ms weight-only
compute floor; no other config is ever strictly better and there are no rank swaps. The same
result holds on the legacy single-fabric path and the spec/topology path (59.708 vs
59.721 ms), so it does not depend on which pricing path produced the table. CONVENTION -- read this before comparing any two rows. The model's step time is for a
FIXED GLOBAL BATCH of one sequence of `seq` tokens split across all N ranks: each rank
handles seq/(cp*dp) tokens, weights are split by tp, layers by pp, the sequence by cp and
the batch dimension by dp. Every split therefore divides the per-rank work by its factor,
which is why compute = (6P + 12*L*seq*d)*seq/N is invariant across the space at fixed N
(work conservation) and why ties are the honest output. It is NOT the latency of one
sequence replicated on N GPUs: with dp=8, eight replicas each processing their own batch do
the same per-rank work as dp=1. Data parallelism buys throughput; only tp/pp/cp reduce
per-step latency. To read a row as per-sequence latency instead, multiply its compute and
comm by dp. This convention also fixes what the rank layout means (rank =
((pp*tp + tp_rank)*cp + cp_rank)*dp + dp_rank) and it is the reason the same N=8 cells can
tie at 51.6 ms weight-only while their dp differs eightfold.

Absolute step
times remain modelled -- one 5090 cannot hold Llama-3-8B at any decomposition (the search's
own memory column says 68.5 GB for the tp=2 cp=4 row and 50.8 GB for the FSDP row with the
completed memory model and the calibrated activation factor, gpu-sim-6y1/-4nf). MEMORY FEASIBILITY IS MODELLED, not measured: the
model now carries params, grads, fp32 master, optimizer, per-tensor activations (FFN at
d_ff), logits and CP buffers. The QUADRATIC score-storage term is calibrated on a 5090
(d=512, 6 layers, seq 4096-8192: the model is 15-20% HIGH in materialized mode, 40% low at
seq 1024 where the measurement is a few hundred MB). The DEFAULT flash/SDPA regime is the
opposite direction and is NOT calibrated: with a constant baseline the same runs put the
model's linear activation term 2.6-3.2x BELOW the measured memory, i.e. a LOWER bound.
gpu-sim-6bi declared that gap; gpu-sim-4nf attributes it: a torch memory snapshot
(d=512, 6 layers, seq 8192, flash/SDPA) shows ~70% matmul intermediates (per-layer peak
2.3x the per-tensor-width estimate: qkv, attention output and FFN pre-activation live
beyond the 4 named tensors), ~6% cuBLAS workspace, ~24% allocator granularity. Fix: act_factor
1 -> 2.3. It is a FIT AT ONE SHAPE, not a law: it bundles tensor intermediates (~70% of the
gap) with allocator granularity (~24%) and a constant cuBLAS workspace (~6%). The tensor
part alone would be ~1.9x, and the allocator share is shape-dependent, so at another shape
the model can be 20% either side. It also over-charges the non-flash path, where factor 1
was already 15-20% high: with 2.3 that path is ~29% high, which is immaterial only because
the S^2 score term dominates there whenever seq is large (12.9 of 13.2 GB at the calibration
shape), so a materialized 8B run is infeasible either way. Concretely at 8B/seq8192 with
flash on: the FSDP row is 50.8 GB modelled and the tp=2 cp=4 row is 68.5 GB modelled, i.e.
within 15% of the 80 GB limit and therefore BORDERLINE -- anything the model puts within
~25% of the limit belongs in that set. Everything else in the feasibility column is arithmetic on
bytes that do not depend on the device. What IS validated on this host: the
per-rank collective COUNTS (16 tp all-reduces and 24 cp K/V sends per rank per step,
gpu-sim-5o4.7). What is NOT: the byte SIZES (harness reduces a 3d/tp qkv tensor, the model
assumes Megatron's full-d payload -- gpu-sim-oez) and wall time, ever.
d05|
**Does the answer generalize past 8B? (gpu-sim-rzg, modelled).** The same search at seq 8192
with published configs, spec/topology pricing, and the shipped 16 B/param accounting (bf16
params+grads, fp32 master, fp32 Adam m,v):

| model | N=8 | N=16 |
| --- | --- | --- |
| llama3-8b | FIT (4,1,2,1,0) 88.49 ms | FIT (1,16,1,1,0) 50.66 ms |
| qwen2.5-32b | FIT (8,1,1,1,0) 323.98 ms, 78.8 GB | FIT (1,16,1,1,0) 180.36 ms, 41.74 GB |
| llama2-34b | MISS by 9.36 GB | FIT (1,16,1,1,0) 218.38 ms, 45.17 GB |
| llama3-70b | MISS by 81.43 GB | MISS by 0.85 GB (min 80.85 at tp=8,pp=2) |

The 8B row reproduces the audited answer exactly: (tp=4 pp=1 cp=2 dp=1) at 88.49 ms for N=8 and
(tp=1 pp=16 cp=1 dp=1) at 50.66 ms for N=16. The boundary is
set by OPTIMIZER STATE, not by activations or compute: for the binding 34B config
(tp=8,pp=1,cp=1,dp=1) the footprint is 89.36 GB = master 19.69 + opt 39.38 (59 GB, 66%) +
params 9.85 + grads 9.85 + act 10.53 + logits 0.07. So the model's "70B does not fit 16
H100s" is a statement about fp32 Adam states with a separate fp32 master, NOT about the
architecture: bf16 moments or sharded/offloaded optimizer state move this boundary by tens
of GB. Nothing here is measured; the accounting is the deliverable and the caveat.

## 1. Premise

**Intercept the libraries, don't re-implement the model.**

A training program written against torch / NCCL runs unmodified. The CUDA and
NCCL symbols it calls are resolved to our interposer instead of the real ones, so
timing and traffic decisions are made below the Python line. The same program can
then be run against different parallelism configurations and different fabrics
without being rewritten, and the trace that comes out is a record of what the
program *actually asked the hardware to do* — not a spreadsheet's opinion about
it.

North star: run a real torch model under different parallelism configurations
(DP now; TP/PP later), see the collective traffic and the time it costs, and use
that to pick a configuration before committing GPUs to one.

Accuracy is explicitly not the first bar. With no calibration data, per-kernel
times are synthetic (random within a declared range) and collectives are timed by
the declared fabric model. What must be right first is the **structure**: which
collectives are issued, with what sizes, in what order, by which rank.

## 2. What exists

Honest status per axis -- traffic is traced for all four, but only DP has been
shown to train correctly (see §6 for what each gate was):

| axis | collective traffic | trains correctly |
|---|---|---|
| DP | yes | yes (world2 == world1 to 1.13e-07) |
| ZeRO 1/2/3 | yes | not yet -- payload work in progress |
| TP | yes | not yet -- payload work in progress |
| PP | yes (+ p2p) | not yet -- payload work in progress |

Reading the left column as if it were the right one is the single easiest mistake
to make with this harness, which is why they are separate columns.

| Piece | Where | State |
|---|---|---|
| CUDA runtime + driver interposer | `src/`, `shim/cuda_runtime.h` | 89 exported `cuda*`/`cu*` symbols; random fill; fixed, bandwidth, table-driven or synthetic latency |
| NCCL interposer | `src/fakegpu_nccl.c`, `shim/nccl.h` | 68 symbols — now the **full set torch references** (see §4) |
| NCCL-only build | `libfakenccl.so` + `shim/exports_nccl.map` | exports only `nccl*`, so torch computes on the real GPU with fake collectives |
| Trace emission | `fg_nccl_trace`, `GPUSIM_TRACE=<path>` | JSONL, one line per collective per rank |
| Fabric cost model | `epic_cluster/link_model.py` | predicts collective duration from a declared topology; measured for n=2, refuses n≥3 without explicit `pieces_per_phase`. ALPHA ACCOUNTING (gpu-sim-8nt, kept=false): alpha is charged PER LOGICAL TRANSFER by the model (every `search()` call site passes `pieces_per_phase=1`), PER CALL by the harness (blocking `dist.send`/`dist.recv`, one call per peer per microbatch, no isend/grouping), and PER PIECE by a real NIC (the measured pkb table gives P=4-8 for payloads ≥256 KiB). The three agree only because the harness never chunks -- the model therefore UNDER-charges alpha for large payloads by P relative to real framing. Do not read the model/harness agreement as a validation of network semantics. |
| Torch harness | `tools/torch_parallel/` | runs a real transformer; DP, TP and PP axes; sweeps and summarises |
| Rack planner (analytic) | `cy0_l1..l4.py` | pre-existing FLOP/roofline planner; **separate track**, not used by the harness |

## 3. How a run works

```
torchrun --nproc_per_node=N  tools/torch_parallel/train.py
   └─ torch program (unchanged): TinyLM, AdamW, backward
        ├─ CUDA calls   → real GPU (compute is real, on the physical card)
        └─ NCCL calls   → libfakenccl.so
                            ├─ simulated duration (fabric model, or fallback)
                            └─ one JSONL line per collective: rank, op, dtype,
                               count, bytes, timestamp, predicted_s
```

Two facts that make this work, both learned the hard way (see §4):

- **Ranks share the one real GPU.** The multi-device fan-out is simulated. Every
  rank calls `torch.cuda.set_device(0)`; asking for device==rank is what
  "invalid device ordinal" means on a one-GPU box.
- **Collectives simulate timing, not data.** Buffers are left untouched
  (`FAKEGPU_NCCL_NO_FILL=1`), because writing into a real device pointer from the
  host segfaults, and because only the timing is being simulated. Consequence:
  after `all_reduce` the gradient is not the true sum across ranks. Traffic and
  timing are real; the arithmetic is not.

## 4. Findings that changed the design

**A single `LD_PRELOAD` of `libfakegpu.so` cannot drive torch.** It replaces
`cuda*` and `cu*`, but torch's real `libcudart` resolves the driver through
`dlopen("libcuda.so.1")` + `dlsym`, which preload does not interpose. The process
ends up with fake handles reaching the real driver and dies inside
`cuEventRecordWithFlags`. Standing in for the driver instead means implementing
**462** `cu*` symbols that cudart looks up by name — measured, not estimated.

**So the interposer is split.** `libfakenccl.so` exports `nccl*` only. CUDA stays
real, NCCL is simulated. That is the split that runs torch today.

**torch references 51 `nccl*` symbols; the interposer exported 38.** The missing
ones included `ncclCommInitRankConfig` — the call `ProcessGroupNCCL` actually
makes — so torch silently fell through to real NCCL and failed in
`ncclSocketInit`. The full set is now covered (config init, unique ids, user
rank, `ncclMemAlloc/Free`, group-simulate-end, pre-mul-sum, the `Bcast`/`AlltoAll`
/`Gather` spellings, and benign success stubs for the window/device-comm/signal
extensions a plain run never exercises).

**Device count comes from NVML, not CUDA.** `torch.cuda.device_count()` consults
`nvmlDeviceGetCount_v2` on the real driver. Interposing CUDA alone will not make
torch see more devices. Relevant only for single-process multi-device programs;
the harness sidesteps it by using one process per rank.

**Every subgroup shared one comm identity.** `ncclGetUniqueId2` returned a
constant, so torch's comm cache aliased distinct groups: with pp2/world4 the group
`[2,3]` was never `ncclCommInitRank`'d and its traffic was silently attributed to
another group's communicator. The id must be identical across the ranks of a group
and distinct per group — a per-process call counter satisfies both. Related: the
global rank was read from an env var the harness never set, so every process
reported rank 0; traces now carry both `rank` (comm-local) and `grank` (world).

**The NCCL datatype enum was permuted against real NCCL.** `shim/nccl.h` had
Float32=6, Float64=7, Float16=8; real NCCL has Float16=6, Float32=7, Float64=8.
Every float32 volume was therefore reported at 8 bytes/element — exactly 2× too
large, across an entire measurement campaign. It survived because it was
self-consistent: every ratio, ordering and conclusion held while every absolute
number was wrong. A regression test now pins the constants, mutation-checked
against the original defect.

**`ncclReduceScatter` took its count by pointer.** Real NCCL passes `recvcount` by
value; our shim declared it `const size_t *`, so torch's integer count was
dereferenced as an address — a guaranteed SIGSEGV on any FSDP/ZeRO run. This is
worse than a missing symbol: a missing one fails at load with a clear message, while
a calling-convention mismatch links silently and faults later somewhere that looks
unrelated. Found from a coredump backtrace, not from reading the header.

That makes five defects of one family — a constant `uniqueId`, a permuted datatype
enum, untraced p2p, a wrong parameter type, and the 462-symbol driver dead end.
**None was found by reading the declaration. Every one was found by probing the
call.** For any further work on the shim surface, that is the rule.

**`torch.nn.parallel.DistributedDataParallel` cannot be used as-is.** DDP
cross-checks parameter counts with a real all_gather, so it needs collectives that
move payload. The harness does the gradient all-reduce explicitly instead — the
same traffic DDP would produce, un-bucketed.

## 5. Running it

```bash
make libfakenccl.so                      # once
python3 -m tools.torch_parallel.sweep --world-sizes 1 2 --steps 5
python3 -m tools.torch_parallel.sweep --world-sizes 2 --fabric pcie
python3 -m tools.torch_parallel.sweep --world-sizes 1 2 --delay-ms 0.5 2.0
```

The sweep emits a topology blob for each world size
(`tools/torch_parallel/topology.py`) and points `GPUSIM_TOPOLOGY` at it, so
collectives are timed by the declared fabric. Without it they take the fallback
path and carry `predicted_s: null` -- bytes but no cost.

**Checking section 0 against the code.** `python3 scripts/check_docs_model_numbers.py`
re-computes the numbers section 0 quotes (winners per N, the seq-32768 case, the
S_best matrix, the model-size table), writes them to `docs/model_facts.json`, and fails
if any decimal in that section is not a value the current model produces. Numbers the
gate cannot derive (superseded pre-audit figures quoted for contrast) must be listed
with a reason in `scripts/docs_model_numbers_grandfathered.txt`. It exists because four
section-0 numbers went stale in one day (gpu-sim-a8x), each found by a human reading
prose against code rather than by any test.

Measured on this box (RTX 5090, 5.2M-parameter model, 5 steps, 2 discarded as
warmup):

```
fabric  world   step_ms  coll_msgs   coll_MB  comm_sim_ms
nvlink      1      11.4          0       0.0          0.0
nvlink      2      11.5         75      86.1          0.68
pcie        2      12.1         75      86.1          7.04
```

Three things that table is good for, none of which needed a second GPU:

- doubling data-parallel width adds 75 collectives and 86 MB per rank;
- the same configuration on a slower declared fabric costs 10x more simulated
  communication (0.68 -> 7.04 ms), so fabric and parallelism can be co-varied;
- `step_ms` barely moves because the simulated collective time is small next to
  this model's compute -- which is itself the answer to "is this model
  communication-bound?": at 5.2M parameters and 8x512 tokens, no.

### All three axes, same world size

World=4, NVLink, 5 steps with 2 discarded as warmup. Re-run from a clean worktree
and reproduced bit-identically:

```
config  step_ms  coll  coll_MB  p2p  p2p_MB  comm_sim_ms  params  local
DP4        16.1    75     43.0    0     0.0         0.57  5242880  5242880
TP4         9.7    33    134.2    0     0.0         0.81  5242880  5242880
PP2        13.5    37     22.0    8    16.8         0.35  5242880  2621440
PP4         3.9     1      0.0    8    16.8         0.0  5242880  1967104
```

Read `local` before `step_ms`: DP and TP keep the whole model on every rank, PP
does not. PP4 is fast mostly because each rank owns a quarter of the layers, not
because pipeline parallelism is efficient -- the column exists to stop that
misreading.

The shape of each trade-off is visible and explicable:

- **DP** issues many small collectives (one per parameter, un-bucketed here).
- **TP** issues fewer, much larger ones (2 per layer per direction).
- **PP** issues few collectives and less total bytes, plus point-to-point traffic
  between stages.

**The PP bubble does not appear in wall time here**, and that is a property of the
harness, not a result: a simulated p2p hop costs microseconds while compute is
real, so stages barely wait. Reporting a bubble would require compute to be
simulated too.

### Data parallel vs tensor parallel (world=2)

```
config     world  tp  dp  step_ms  coll_msgs  coll_MB  comm_sim_ms
DP             2   1   2      9.1         75     43.0         0.62
TP             2   2   1      9.9         33    134.2         0.28
```

These are CORRECTED volumes. Every figure published before 2026-10-23 was
**exactly 2x too large** for float32: shim/nccl.h defined Float32=6, Float64=7,
Float16=8, while real NCCL defines Float16=6, Float32=7, Float64=8. torch passes
the constants it compiled against, so float32 was decoded as float64. The error was
invisible because it was self-consistent -- every comparison, every ratio, every
"TP moves more bytes than DP" conclusion was unaffected, which is exactly why it
survived so long. A 1000-element probe caught it: 4000 B of float32 traced as
8000 B. The enum now matches the ABI.

The shape of the trade-off is visible and explicable: **TP issues fewer, much
larger collectives; DP issues many small ones.** That is the textbook tension, and
it is now a measurable property of a config rather than an assertion.

One caveat that keeps this from being a like-for-like byte comparison: DP here is
UNBUCKETED (one all-reduce per parameter, 37 of them per step), whereas DDP would
fuse them into a few large ones. Un-bucketed is a deliberate upper bound on
message COUNT and easier to audit from the trace; it is not what a tuned DDP run
would emit.

TP collectives are 8 MB, beyond the measured phase-payload table, so they are
measured one. The default remains REFUSE.

### Pipeline parallel, same world size

```
config     world  tp pp dp  step_ms  coll_msgs  coll_MB  p2p_msgs  p2p_MB  comm_sim_ms
DP             4   1  1  4     16.1         75     43.0         0     0.0         0.57
TP             4   4  1  1      9.7         33    134.2         0     0.0         0.81
PP2            4   1  2  2      7.5         37     22.0         8    16.8         0.35
PP4            4   1  4  1      4.1          1      0.0         8    16.8         0.0
```

Measured on the same box and model (5.2M parameters, 5 steps, 2 discarded,
nvlink, `GPUSIM_ALLOW_EXTRAPOLATE=1`). Counts are per grank 0 over the 2
measured steps and decompose exactly: PP2 collectives 37 = 18 stage params x 2
+ barrier, p2p 8 = 4 per step x 2 at 2.1 MB each (4x512x256 float32 =
numel x element_size); PP4 collectives 1 = barrier only. `step_ms` absolutes
move run to run on the shared box; the traffic counts are bit-identical
across runs, so traffic is the signal and step time is context.

The shape is explicable: PP replaces TP's few huge allreduces with smaller,
more numerous messages (PP2: 45 total messages, 38.8 MB vs TP's 33 messages,
134.2 MB), and pure pipeline (PP4, dp=1) drops to 9 messages and 16.8 MB.
PP2 carries MORE collectives than TP4 (37 vs 33) because at pp=2 each stage
still data-parallel-syncs its half-model across dp=2 -- the "fewer
collectives" premise holds for the pipeline axis proper (PP4), not when DP
co-traffic is counted alongside it.

The bubble is ABSENT from wall time, and that is stated plainly rather than
fitted: the interposer's send/recv return instantly (timing only, no payload),
so no stage ever actually blocks on another and there is no fill/drain idle
to measure. Per-rank means after the backward fix are within ~10% (stage 0:
11.5/13.5 ms, stage 1: 12.4/12.7 ms); the spread is residual compute
imbalance, not pipeline idle. The modelled p2p cost (predicted_s, p2p-link
regime) totals 0.35/0.0 ms against milliseconds of compute -- even modelled,
there is no meaningful bubble at this model size. A wall-clock bubble would
need a transport that actually blocks, which is the data-correct-collectives
bead, not this one.

Two defects found and fixed along the way, both load-bearing for the numbers
above. (1) The last stage's backward used `autograd.grad(loss, stage_output)`,
which leaves every param grad None: the DP sync silently skipped a whole
stage and the stage ran 3x fast. Now `retain_grad` + full `backward`, verified
by per-rank times converging. (2) Subgroup traffic was mislabeled (local rank)
and partly missing (constant ncclUniqueId aliased distinct groups, so dp
group [2,3] was never even initialized) -- fixed on master (gpu-sim-vtu):
distinct uniqueIds per group and both `rank` (comm-local) and `grank` (world)
on every trace line; the sweep aggregates on `grank`.

### ZeRO, same world size

```
config     world  tp pp dp zero  AR/step  AG/step  RS/step  MB/step   state_MB
DP/ZeRO-1      4   1  1  4    1       37       37        0     43.03       52.4
ZeRO-2         4   1  1  4    2        0       37        1     43.03       36.7
ZeRO-3/FSDP    4   1  1  4    3        0        1        1     43.03       21.0
DP/ZeRO-1      8   1  1  8    1       37       37        0     43.03       52.4
ZeRO-2         8   1  1  8    2        0       37        1     43.03       36.7
ZeRO-3/FSDP    8   1  1  8    3        0        1        1     43.03       21.0
```

Same box, TinyLM 5.24M, 5 steps, fallback path, counts per grank 0 over the measured steps.
**`step_ms` is deliberately absent**: at dp>1 on one GPU the ranks are time-sliced, so wall-clock
in this table was never comparable -- the same dp=4 runs produced per-rank means from 5.3 ms to
76.2 ms depending on which rank won the SliceHost scheduler. The earlier version of this table
printed 20.6/25.1/32.2 ms and drew conclusions from their ORDER; that was noise. The deterministic
columns tell the story instead, and they now hold at dp=4 AND dp=8 (dp=8 was previously
unmeasurable -- see below).

P = 20.9 MB of parameter bytes (fp32, `param_count()` excludes pos/LN), and with the byte
conventions settled EVERY total is dp-independent, as it must be:

  * all-reduce total = P at every dp (37 per-parameter all-reduces summing to one model);
  * all-gather total = P at every dp (37 shard-gathers summing to one model);
  * reduce-scatter total = P at every dp (payloads delivered, i.e. dp x P/dp -- see below).

Every ZeRO stage therefore moves the same 43.03 MB/step (2P) at dp=4 and dp=8, which is what the
cost model charges (all-gather P + reduce-scatter P). What separates the stages is not volume but
MESSAGE COUNT and memory: ZeRO-2 and ZeRO-3 move identical bytes (37 small collectives versus 1 is
alpha and peak memory, not volume), and ZeRO-1 adds a 37-message all-reduce sweep on top.

**Trap worth knowing before analysing any trace: `count` is not comparable across the two
paths, `bytes` is.** Running the SAME job with and without a topology gives 74 all-gathers
with identical byte totals (43.028 MB) but sum(count) 2689280 (topology) vs 10757120
(fallback) -- a factor of nranks, because the topology path is handed the per-rank send count
(reporting bytes = count*nranks*elem) and the fallback the recv count (bytes = count*elem).
Both byte columns agree, so `bytes` is the canonical field; any conclusion drawn from
comparing `count` between a topology run and a fallback run is wrong by the group size. That
is recorded in the code where the fallback traces, rather than "fixed" by normalising: which
convention the torch side actually passes was not pinned down, and inventing one would have
made a second silent error.

**What the byte reconciliation can and cannot validate (gpu-sim-wrvw, gpu-sim-juzb).** With the
model and the harness compared on the model the harness actually ran (TinyLM, fp32), the DP axis
agrees to **2.5%** -- all-gather and reduce-scatter both ratio 1.025 at dp=4/zero=3, and the 2.5% is
exactly the documented `param_count()` pos/LN exclusion. That is the first end-to-end validation of
a comm term against a real trace rather than against a fixture generated from the model.

The **CP axis cannot be reconciled this way, by construction**. This model prices context parallelism
the way Megatron runs it: a RING p2p exchange of each rank's K/V chunk with its (cp-1) peers, per
layer, forward and backward. The harness implements it as a FULL all_gather of K/V (replication, not
rotation) plus, because `torch.distributed.nn.all_gather` is differentiable, a reduce_scatter of the
same tensor in backward, plus a per-parameter gradient sync. The trace is a perfectly symmetric AG/RS
pair, which is the autograd signature, not a byte bug. (An earlier version of this paragraph compared
"109.8 MB/step against 21.0 MB" and called the harness 5.2x the model. That comparison was RETRACTED: it
put different quantities on different models, and once the K/V width was corrected the direction was
wrong. What is actually measured is the size of the convention gap on the K/V gather itself -- 1.98x to
1.99x across three frozen cp=2 traces, see `experiments/gpu-sim-cpgqa/` and section 0.) Neither side is "fixed" to match: the model keeps the production
pattern the recommendation is about, and the harness's CP is recorded here as an AG-based
approximation that cannot validate this term. Absolute CP pricing therefore rests on the model plus
the ring argument, not on the harness.

That pairing is also why the last unidentified bucket in the cp=2 trace does not matter. The trace
carries an all_gather of 1,044,480 elements next to one of 1,048,576 = vocab(4096) x d_model(256),
the token embedding, and every all_gather has a reduce_scatter of EXACTLY half its elements (byte
totals 549,376,000 on each side). 1,044,480 is 4080 x d_model -- not 4095 x d_model, so it is not
"the embedding minus one row" (that would be 1,048,320), and its exact parameter origin cannot be
recovered from the trace alone. It does not need to be: because the AG/RS pairing is exact for every
bucket, a bucket whose parameter we cannot name still reconciles perfectly against itself. The
remaining unknown is a label, not a byte (gpu-sim-wlab, tests/test_cp_payload_structure.py).

**Three byte bugs had to be fixed before any of this was true** (gpu-sim-6wd-bytes,
gpu-sim-smt3, gpu-sim-smt6), and all three are the same dp x / n x species of error the COST MODEL
carried before gpu-sim-6r5 -- one layer down each time:

1. the interposer's fallback path wrote and recorded `payload * (1 + floor(log2 n)/2)` bytes,
   an out-of-bounds fill (below);
2. the harness's own ZeRO param gather sent the FULL parameter and received dp copies instead of
   sending the shard and receiving the parameter, so it moved dp x the bytes a real gather moves
   (all-gather read 86.06 MB/step at dp=4 and 172.11 at dp=8 before; 20.91 at both after);
3. the fallback recorded a reduce-scatter's SHARD (P/dp) where the topology path records the
   total, giving RS 5.38 MB at dp=4 and 2.69 at dp=8 -- a spurious dp-dependence that looked like
   an accounting mystery for a day;
4. and in point-to-point (PP/CP, gpu-sim-smt7) the element count was passed where BYTES are
   expected: `fg_simulate_memcpy(count)` charges 1/4 of the real transfer at fp32, and ncclRecv
   filled `count` BYTES of a count*elem buffer -- a quarter of it. p2p escaped the earlier audits
   because it moves no group total, which is exactly why it was worth auditing.

**The remaining DP byte puzzle was a BUG, not an accounting convention
(gpu-sim-6wd-bytes).** For a long time the DP traces could not be reconciled: a
zero=3 all-gather read 32.27 MB/step at dp=2 and 43.03 MB at dp=4 although a
full-parameter gather must be dp-independent. The all-gather COUNT was dp-independent all
along; what scaled was the fallback path's byte column. `fg_collective_out()` computed

    bytes = payload * (1 + floor(log2 nrank) / 2)     /* "model fan-in/fan-out cost" */

before recording the event -- exactly x1.5 at n=2 (20.91 -> 32.27) and x2.0 at n=4
(20.91 -> 43.03). It then handed that inflated count to `fg_nccl_out_fill()`, i.e. wrote
1.5x the recv buffer at n=2, 2x at n=4 and 2.5x at n=8: **an out-of-bounds heap write**,
reproduced as glibc `malloc(): corrupted top size`. So the byte column now carries the
payload (matching the topology path, which always did) and `tests/test_fallback_fill_bounds.py`
pins both halves at n=2/4/8.

That also retires a claim made earlier in this document's history: **dp=8 was NOT
impossible on this host.** With the overflow fixed all 8 ranks report (previously 5-6 of 8,
or nothing), so the "world=8 does not fit" note was a symptom of the corruption. Remaining
real limitation: 8 ranks writing ONE trace file with no locking, so a dp=8 run can drop
torn lines (1 torn line in 3186 at dp=8 here) -- count-based conclusions survive, per-rank
row-count equality does not.

Memory is the honest divergence this axis exists to show: measured peakMemory is the honest divergence this axis exists to show: measured peak
barely moves (627/676/651 MB -- activations dominate, no ZeRO stage shards
them) while the analytic state falls 52.4 -> 36.7 -> 21.0 MB. ZeRO-2's peak
is HIGHEST (+48 MB over ZeRO-1): the hand-written path allocates flat grad
(21 MB), shard (5 MB), and per-param gather scratch every step. That scratch
is the price of the timing-event approach and it is measured, not modelled.

How each stage is built: ZeRO-1 = DP allreduce + post-step all_gather into
discarded scratch (the interposer moves no payload, so copying back would
corrupt good params with uninitialized memory). ZeRO-2 = one flattened
reduce_scatter + the same post-step gather; the local update still uses full
grads (timing events, not arithmetic). ZeRO-3 = FSDP over the dp group.
FSDP was predicted blocked on payload; it is not -- the only blocker was our
own ReduceScatter signature bug (recvcount by pointer vs by value, SIGSEGV in
backward, fixed on master with a parse test pinning it). tp/pp x zero
combinations are refused explicitly until verified.

### Warmup

WARMUP IS DISCARDED, and that matters: the first step measured 556 ms against
11 ms for steady state (CUDA context, cuBLAS init, kernel compilation). Averaging
it in would make a sweep compare initialization cost instead of the parallel
configuration. `coll_msgs` is filtered to the same window as `step_ms`
(`measure_start_ns`) so the two describe the same steps.

## 6. Component status (what is built, and the two things that are not)

- **Fabric-backed timing on the harness path.** Without `GPUSIM_TOPO`, collectives
  take the fallback duration and are traced with `predicted_s: null`. Wiring
  `link_model.predict()` into that path is the next step, and it is what turns
  "we issued 75 collectives" into "they cost X ms".
- **Point-to-point timing.** `send`/`recv` cost `S/min_bw(path) + sum(latency)`
  from the declared link parameters — first principles, not the fitted ring table,
  which is why p2p is modelled while large ring collectives are still refused.
  Verified: 256 B on NVLink 1.0003 us, on PCIe 8.0102 us. An unconnected pair
  refuses; with no topology the duration stays null.
- **PP.** Done (gpu-sim-8y7.1): stage split of TinyLM layers across the pp axis,
  GPipe schedule (`--micro-batches`), blocking send/recv between adjacent stages,
  `--pp`/`--micro-batches` in train.py and sweep.py with rank decomposition
  rank = pp_idx*(tp*dp) + dp_idx*tp + tp_idx. Same timing-only bargain as the rest:
  stage outputs are garbage, loss curves are not meaningful, traffic/timing are real.
- **Memory, and one rule for reading it.** ZeRO's modelled sharded-state saving is
  52.4 -> 36.7 -> 21.0 MB, and with payload really moving the measured peak is
  29.0 MB lower for ZeRO-3 than ZeRO-1 -- within 8% of the model. Two caveats,
  both earned: absolute peak numbers are only comparable between runs with the
  SAME BATCHING CONVENTION (the `--dp-mean` sharding convention alone moves peak
  from 627 MB to 272 MB at world=4, entirely independent of transport mode), and
  the transport itself has zero measurable memory effect.
- **ZeRO.** Done (gpu-sim-8y7.3): `--zero 1/2/3` on the dp axis. 1 = DP +
  post-step all_gather; 2 = flattened reduce_scatter + post-step gather;
  3 = FSDP over the dp group. Measured peak (`max_memory_allocated`,
  windowed) reported alongside the analytic sharded-state model; tp/pp
  combinations refused until verified.
- **Data-correct collectives.** all_reduce only, done (gpu-sim-n5h): real sums
  through /dev/shm (d2h, barrier, host reduce, h2d), gated behind
  FAKEGPU_NCCL_SIMULATE_ONLY=0; default stays simulate-only and reproduces
  the committed traffic record byte-exact. float32/bf16, sum only; all_gather
  and the rest still move nothing.
- **Training correctness under DP.** Done (gpu-sim-tbp): `--dp-mean` selects true
  DDP (disjoint rank shards of one seeded batch, explicit 1/dp average), and
  world=2 agrees with world=1 to worst 1.13e-07 across 12 steps; 50 steps fall
  8.4806 -> 8.4276 with zero up-steps. The convention is not a detail: with the
  same batch on every rank the summed gradient is dp x the single-rank gradient,
  so "matching losses" would prove only that both sides shared one wrong
  convention. Transport-on without `--dp-mean` refuses rather than defaulting.
- **Single-process multi-device enumeration.** Blocked on NVML interposition.
- **Absolute durations of the modelled collective terms.** The DP/TP/CP cost terms are
  priced, not measured: gpu-sim-wv2 (fabric calibration) needs 2-4 real GPUs and is the
  honest blocker. What IS validated on hardware is the collective COUNTS per rank per
  step and the DP payload dtype; see the confidence block above.

## 7. Non-goals

- **Convergence claims about real hardware.** The harness can now train correctly
  under data parallelism, but it does so on one GPU with a simulated fabric, so its
  loss curve is evidence about the SOFTWARE configuration, not about a cluster.
  (This non-goal used to read "the harness moves no data between ranks" — that
  stopped being true when the transport landed, and leaving it would have
  understated the system as badly as claiming more than it does.)
- Bit-accurate step times without calibration.
- Re-implementing the model's arithmetic at a higher level. If a number cannot be
  observed at the library boundary, this project does not report it.

## Claims register (generated)

<!-- CLAIMS-REGISTER:BEGIN (generated by scripts/gen_claims_register.py) -->

| claim | status | evidence | guard |
|---|---|---|---|
| N=8 winner (tp=4 pp=1 cp=2 dp=1 z0), seq 8192 | **VERDICT** | 88.49 ms; margin 6.03% at the implied MFU, 5.70% at 40% | tests/test_fragility_map.py |
| Same config at seq 16384 / 32768 | **TIE** | margin only 0.22% / 1.48% -- the config is right, the ranking is not decisive | tests/test_fragility_map.py |
| N=16 winner (tp=1 pp=16 cp=1 dp=1 z0), seq 8192 | **VERDICT** | 2.06% at 40% MFU; flips below MFU 0.25 | tests/test_mfu_sensitivity.py |
| N=16 at seq 16384 | **TIE** | three-way tangle: winner moves to (tp=2 pp=2 cp=4 dp=1 z0) at 40%, margins 0.45% / 0.64% | tests/test_fragility_map.py |
| N=16 at seq 32768 (hard-fit memory) | **TIE** | 2.20% margin and 92% of HBM; 10% headroom swaps it to (2, 2, 4, 1, 0) at 41.6 GB (+2.4%) | tests/test_memory_headroom.py |
| N=32 winner (tp=1 pp=16 cp=2 dp=1 z0), seq 8192 | **TIE** | margin 0.02% at worst over every seq and efficiency tested | tests/test_fragility_map.py |
| Absolute step durations | **BOUNDED** | model implies 0.786 MFU; a real GPU reached 0.566, so absolutes carry 1.28-1.69x | tests/test_mfu_sensitivity.py |
| act_factor (2.3) fitted constant | **STRUCTURAL** | memory-only for step time: 5x band changes no step time; ranking bends only at 28.5 (N=8) / None (N=16,32) at seq 8192; the seq-32768 N=16 verdict flips on feasibility at 2.277, and the measured range 2.189-2.435 (median 2.329, 7 measurements on a 5090) straddles it | tests/test_act_factor_knob.py |
| S_best = 8192 | **VALIDATED** | CP crossover is seq 5956 (3658 measured), so 8192 is the first conventional length above it, +6.03% | tests/test_cp_crossover.py |
| Alpha piece-charging granularity | **BOUNDED** | +6.0% at P=8, winner unmoved | docs section 0 (alpha piece charging) |
| Harness determinism (the instrument itself) | **VALIDATED** | two runs of dp=4/zero=2/3 steps give 115 records per rank with identical per-rank sequences; only cross-rank interleaving differs | tests/test_trace_determinism.py |
| DP byte counts / payload | **VALIDATED** | reconciles against harness traces to 2.5%, residual = param_count() pos/LN exclusion | tests/test_reconcile_bytes.py |
| TP byte counts / payload | **VALIDATED** | message count matches the plan exactly; payload B*seq*d is tp-independent | tests/test_parallelism_counts.py |
| CP byte accounting | **CONDITIONAL** | K/V gather moves 1.98x-1.99x the model's ring message (measured, 3 frozen cp=2 traces); ring vs all_gather are different algorithms, so bytes bound the CONVENTION gap, not a validation | tests/test_cp_gather_bytes.py |
| Absolute communication durations | **UNVALIDATED** | rest on the model's ring alpha/bw; needs a 2-4 GPU NVLink run, unavailable on one GPU | no test possible here |
| CP term is decisive for the N=8 verdict | **BOUNDED** | winner unchanged for CP error in <!-- RADIUS-INLINE:BEGIN -->[0.54x, 1.52x]<!-- RADIUS-INLINE:END -->; both sides measured | scripts/sensitivity_radius.py |
| PP bubble is decisive for the N=16 verdict | **BOUNDED** | winner unchanged for bubble error in [unbounded below, 1.86x]; bubble is UNMEASURED; one side measured | scripts/sensitivity_radius.py |

<!-- CLAIMS-REGISTER:END -->

<!-- RADIUS-PROSE:BEGIN (generated by scripts/render_radius_prose.py) -->
N16_seq8192: CP [0.5383, unbounded], PP [unbounded, 1.8576]
N32_seq8192: CP [0.4486, 1.4737], PP [0.5439, 2.229]
N8_seq32768: CP [0.1964, 4.8104], PP [0.3645, unbounded]
N8_seq8192: CP [0.5415, 1.5244], PP [0.8255, unbounded]
<!-- RADIUS-PROSE:END -->

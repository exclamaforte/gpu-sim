# Best parallelism at scale-out (analytical model)

Ask: best parallelism for several model sizes on 10x8, 100x8 and 1000x8 H100.

Answers: `answers.md` (human) and `answers.json` (machine). Regenerate both with

    python3 scripts/scaleout_answers.py --nodes 10,100,1000
    python3 scripts/scaleout_answers.py --nodes 10,100,1000 --json

## Fabric

Generated in-memory by `scripts/scaleout_answers.py`: 8x H100 SXM per node with NVLink-4 full mesh inside
(900 GB/s bidirectional, 1 us) and ONE 400 Gb/s NDR NIC per GPU across nodes (50 GB/s per GPU, 5 us), with
the cross-node fabric a per-rail RING (GPU g of node i <-> GPU g of node i+1). A ring is what NCCL walks
for a cross-node ring all-reduce, and the model prices a ring; a full cross-node mesh would make every
cross-node hop one link and flatter wide collectives. Real rail-optimised fabrics use switches, which sit
between these two extremes.

## Constraints applied on top of the raw search

* `tp <= 8` -- TP needs NVLink-class all-to-all on every layer; the raw search returns tp=16 across nodes
  at 8000 GPUs and that is a model artefact, not a recommendation.
* `pp <= n_layers` -- a pipeline stage must own at least one layer.

## Caveats that decide how much to believe any row

1. **Batch.** Rows are quoted at batch=1 per rank (the convention every published number in this repo uses)
   and again at batch=8. The DP gradient sync is charged per STEP while payload comm is per TOKEN, so
   batch=1 flatters DP-avoiding configs (deep PP, wide CP). With batch=8 the answers move toward DP and the
   step times rise by the batch factor. A real frontier run uses a global batch in the millions, which is
   beyond anything this model has been validated against.
2. **CP convention.** Two rows per cell: `ring all_to_all` (what the model prices) and
   `default all_gather` (PyTorch's `rotate_method` default, measured on the harness to move 1.99x the bytes
   the model charges). Where the two disagree, the default one is the one to plan with.
3. **No activation recomputation in the quoted rows.** Memory is factored by pp only, so 405B on 80 GPUs
   comes out INFEASIBLE and 405B on 800 needs pp=100 at ~66 GB/GPU. Real 405B training uses full activation
   checkpointing with
   pp=8-16. The 405B rows are therefore a statement about this memory model, not about practice.
   The with-full-recompute sensitivity is measured, not assumed: see `experiments/gpu-sim-hz3u/` -- 405B@80
   stays infeasible even with full recompute (70.9 GB of optimizer state alone, which recomputation cannot
   touch), and the 800/8000 winners do not move. The script takes `--recompute 0.0..1.0` (`0.0` is the default
   behind every number above).
4. **Margins, not just winners.** At 8000 GPUs most margins are < 1%: the model cannot separate the top
   candidates. Those cells are ties, not recommendations.
5. **Absolutes are bounded, not predicted.** Measured on an RTX 5090, the model's implied MFU is optimistic
   by 1.28-1.69x, so treat `step (s)` as a lower bound.

## Headline reading

* 80 GPUs: pipeline-dominant. 70B wants pp~40 under the ring convention (4.85% margin, a real
  verdict) and is a 0.11% tie under the default one; 8B is a near-tie among pp~16-20 configs
  (0.87% ring / 2.05% default); 405B does not fit this model's memory accounting at all.
* 800 GPUs: 8B and 70B are decided by efficiency and convention rather than by structure (0.26-4.1%
  margins, and the winner flips between PP+CP and DP+ZeRO-2 depending on both). 405B fits only at pp=100.
* 8000 GPUs: most cells are ties (margins 0.61-5.09%; the 70B ring cell is a 5.09% verdict at
  tp=8 pp=25 cp=20 dp=2 z2). The structural shape is stable -- full-node TP (tp=8), deep PP (20-40),
  wide CP (16-25), dp=1-2 for the large models, dp=500+ZeRO-2 for the small one -- but most configs
  are not separable from their neighbours, and the CP convention alone reverses some orderings.

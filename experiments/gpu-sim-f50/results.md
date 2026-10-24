# gpu-sim-f50: transport for all_gather / reduce_scatter / send-recv

Transport shape (unchanged from the bead): device ptrs -> `cudaDeviceSynchronize`
+ `cudaMemcpy d2h` -> shm -> barrier -> host reduce -> `h2d` via
`dlsym(RTLD_DEFAULT)`. Gate `FAKEGPU_NCCL_SIMULATE_ONLY==0`; default unset =
simulate-only, reproduces the committed record byte-exact. Publish rule:
segment built fully under a pid-unique temp, then `link()` to the canonical
path; a joiner never sees a half-init header. Stale rule: `O_EXCL` + join only
if the header validates AND the creator is alive (`kill` `ESRCH`=dead).

All jobs: `experiments/gpu-sim-f50/tmp/*.sh` bodies (`tp.sh pp.sh zero2.sh
zero3.sh defgate.sh`, plus `tmp/probe.sh` for the unit probe). The full suite
reruns as one body: `experiments/gpu-sim-f50/validate.sh`.

## Gate results (all green)

| gate | job | result |
|---|---|---|
| default replay (unset SIMULATE_ONLY, world4 DP) | `20261024t000228-a555f54a` exit 0 | `coll=149 MB=86.1`, byte-exact vs `gpu-sim-8y7.3/sweep-zero1.json` |
| transport unit probe (AG f32 exact both ranks, bf16 tol, RS, P2P both dirs) | `20261023t234159-79f2c94b` exit 0 | PASS |
| TP fwd + 20 SGD steps | `20261024t000008-a5a72f91` exit 0 | fwd err 1.7e-6 both ranks; trajectories match, monotone |
| PP fwd pp=2 vs single | `20261023t235320-02ca412e` exit 0 | exact, err 0.0 |
| ZeRO-2 reassembly | `20261023t235320-e0051d21` exit 0 | reassembled/dp == single-process grads, maxerr ~1e-5 |
| ZeRO-3 FSDP curve vs zero1 | `20261023t235320-99835df5` exit 0 | worst 0.00e+00 |

Two reds on the road, both root-caused and fixed (not hidden):

- TP fwd err 0.75: contiguous row slice of qkv hands rank 0 all-of-q +
  half-of-k (the `(3,heads,dim)` view lays rows out q-major). Fix: rank r
  takes rows `[r*n:(r+1)*n]` of EACH third. CPU replica exact 0.0 after.
- TP step-1 divergence (ref 8.363 vs TP 8.350): `_ReduceSum` summed in BOTH
  passes; backward sum hands each rank tp x the output-grad (Megatron f =
  sum-fwd / identity-bwd; the sub-layer backward collective lives in
  `_ColLinear`). Fixed backward to identity.
- Packaging reds (not findings): worker `sys.path` lacked the worktree
  (`PYTHONPATH=$PWD` in job scripts); `tp/pp/zero2` scripts echoed rc but
  exited 0 (teeth added); `--dp-mean` refused `zero!=1` (relaxed to
  `tp==pp==1`, covering 2/3).

Harness notes for the reviewer:

- `_ColLinear` (new): forward plain matmul, backward local `gx`/`gw` +
  all-reduce-sum of `gx`. The forward is Col (no comms); the backward
  collective is Megatron's schedule, same bytes per sub-layer per pass.
  This corrects TP-backward AR traffic, which the previous model omitted.
- `_seed_block` (new): every rank draws the FULL block weight from the same
  per-layer generator (`crc32(seed:name)`) and slices its shard; the
  unsharded reference uses the same draws, so agreement is by construction.
  QKV slices per-third (see above); proj/ff2 slice cols, ff1 rows.
- `train.py`: `dp_rank` is now `None`-safe for dp==1 (`dp_rank or 0` was
  `None:start` before); `zero 2/3` refuse `tp>1 or pp>1` (unchanged
  principle, widened axis); `--save-flat` (dp==1 capture) /
  `--verify-flat` (zero2 all_gather-back + assert vs reference).
- Enforcements kept: no `--dp-mean` + transport + dp>1 = loud `SystemExit`
  (zero1 path); DP path (`zero==1`, tp==pp==1) untouched -- the default
  gate proves it.

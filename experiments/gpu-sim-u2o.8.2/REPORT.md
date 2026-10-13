# gpu-sim-u2o.8.2 REPORT: PTX/SASS extraction at corpus scale

## Outcome

First run of the PTX path beyond the throwaway dummy kernel: all **3261
measured corpus rows** compiled to PTX **and** SASS on CPU (nvcc sm_120 +
cuobjdump, no GPU), tokenized both streams, parsed ptxas -v resources.
`ptx_corpus.jsonl`: 3261 records, exact hash-join to the corpus, zero
errors. `verify_82.py` 7/7 PASS.

## Run shape (acceptance: every example, queued-vs-CPU split honored)

- Spec sources: committed manifests (11.3/f6y/0vk) + deterministic regen
  for 7.1/e5q (hash-verified, 9yt machinery). 3286 specs known, 3261
  measured, 0 unresolvable.
- `extract_82.py`, 2 workers, nice'd, 2408s wall (~1.5s/row med; SASS
  cubin dominates). CPU-only throughout -- the GPU queue was never
  touched, correctly: compile+tokenize needs no device.
- Per-row record: ptx_ok/sass_ok, token counts + sha digests for both
  streams, registers/shared/spills, compile_ms. Full PTX/SASS texts NOT
  committed (~100MB+); regenerable deterministically via the script
  (digests prove it).

## The numbers

- PTX tokens med 328 (max 2239), SASS med 371 (max 2569).
- registers/thread med 11, max 56; spillers: 0 rows. No local-memory
  pressure anywhere in the corpus -- consistent with the corpus being
  global/shared-memory-bound by design, and a useful prior for the
  representation ablation (8.4): register features will be low-variance.
- All 40 op families covered (min gemm_wmma_fp16 24).

## What this unblocks

8.4 (source-vs-PTX-vs-hybrid ablation) now has both modalities for every
row: source tokens (8.1, on every row) + PTX/SASS tokens and resource
numerics (here). The numeric side also feeds 8.3-style feature vectors
without recompilation.

## Gates

- Exact join 3261/3261, unique; ptx+sass ok throughout; resources throughout.
- `run.sh` replays verify (re-extraction is a recorded 40-min run, not
  replayed; the artifact is verified from its digests + join).

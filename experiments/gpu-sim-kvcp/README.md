# gpu-sim-kvcp: the CP term prices d_model, not the GQA K/V width

Found while trying to narrow the last UNVALIDATED row in the claims register (CP byte accounting).

## The bug

`script/../epic_cluster/parallelism_search.py` prices context-parallel traffic as

    msg_bytes = seq_per_cp * d_model * dtype_bytes

but a ring context-parallel exchange moves **K/V blocks**. Llama-3-8B is GQA with 8 KV heads, so
the width is `n_kv_heads * head_dim = 1024`, not `d_model = 4096`. The CP term is **4x too large**
on every published number.

## The fix

`model_fix.diff` — adds `n_kv_heads` to `MODEL` and to the four sweep models (llama3-8b=8,
llama3-70b=8, qwen2.5-32b=8, llama2-34b=64 i.e. MHA) and uses `min(kv_dim, d_model)` in both the
spec and legacy CP paths. Applying it leaves the gate green but makes 56 decimals in docs section 0
stale, because the correction cascades.

`facts_with_kvwidth.json` is the regenerated fact set to rebuild the doc from.

## What changes (old -> new)

| quantity | old | new |
|---|---|---|
| N=8 winner | 94.12 ms | **90.54 ms** |
| N=32 winner | 55.54 ms | **53.31 ms** |
| seq 16384 / 32768 | 204.86 / 506.38 | 197.70 / 492.06 |
| legacy chain | 93.77 / 204.51 / 506.03 | 90.19 / 197.35 / 491.71 |
| cp_comm | 4.836 ms (2176 MB) | **1.257 ms (566 MB)** |
| CP crossover | 4819 | **4091** (3788 measured) |
| alpha P=1/4/8 | 94.12 / 96.43 / 99.51 | 90.54 / 92.85 / 95.93 |
| MFU alternatives | 116.96 / 153.41 | 113.39 / 149.83 |

## Qualitative, not just numeric

- N=8 winner margin at seq 8192 **rises** 6.42 -> 10.09% (cheaper CP makes CP more attractive).
- N=8 at seq 16384 **collapses to a 0.10% tie** (was 1.84%).
- N=16 at seq 8192 becomes a 0.02% tie; seq-16384/32768 winners move to (2,2,4,1).
- N=16/seq32768 headroom replacement becomes (2,2,4,1) at 42.5 GB, 429.71 ms.
- N=32 margins tighten to 4.19 / 3.63 / 3.00.

## Also wrong, fix in the same pass

docs section 0 says the harness moves **5.2x more** than the model (21.0 MB vs 109.8 MB). The model
actually prices 566 MB/step at KV width (2176 MB at d_model), so **the model is ~5x the harness**, not
the reverse -- and the comparison is between different quantities on different models anyway (K/V ring
exchange on an 8B model vs per-parameter AG/RS gather on TinyLM), so the ratio should not be quoted as
meaningful at all.

## Why it is not applied

The cascade is a doc-repair job (56 decimals plus several prose claims that change in kind), and
half-updating it would be worse than either end state. The patch reverts cleanly and the gate is green
either way.

# Reference parameter counts, and the convention that reproduces them (gpu-sim-a5ro)

`total_params()` in `epic_cluster/parallelism_search.py` currently computes

    per_layer = 4*d*d + 3*d*ff + 2*d      # K/V charged at FULL width

which is MHA arithmetic applied to GQA models, and it counts one embedding. Measured against published
totals:

| model | current formula | two-embedding, GQA-aware | published | error today | error with fix |
|---|---|---|---|---|---|
| llama3-8b | 8.31 | 8.03 | 8.03 B | +3.5% | +0.00% |
| llama3-70b | 78.90 | 70.55 | 70.6 B | +11.8% | -0.10% |
| llama3-405b | 467.17 | 405.85 | ~405 B | +15.4% | +0.20% |
| qwen2.5-32b | 34.67 | 32.76 | 32.5 B | +6.7% | +0.80% |

CORRECTION: the first version of this table carried a hand-computed "today" column (82.0 for 70B, +16.2%). The
actual output of `total_params()` is 78.90 B, so the 70B error is +11.8%, not +16.2%. Every number above is now
the model's own output, re-derivable with `param_counts.json`.

The two-embedding convention is the one that reproduces published totals within 0.80% across the family.
One embedding is 6.5% low at 8B, because Llama-3/Qwen2.5 do NOT tie their input and output embeddings.

    per_layer = 2*d*d + 2*d*kv_dim + 3*d*ff     # q,o full width; k,v at kv_dim = n_kv_heads*head_dim
    total     = L*per_layer + 2*V*d            # untied: input head + output head

`llama2-34b` is deliberately left unverified: there is no official Llama-2 34B (the family shipped 7/13/70B),
so it is a stand-in in the sweep and must not be used to fit the formula. Its per-layer arithmetic is MHA
(kv_dim = d), which the formula already handles via `min(kv_dim, d)`.

Why this matters beyond tidiness: every memory term (weights, grads, fp32 master, Adam moments) is a multiple
of P, so a 15% parameter over-count is a 15% over-count of every feasibility boundary. At 405B x 80 GPUs the
difference is the 93.4 GB floor the hz3u artifact quoted versus 81.2 GB -- i.e. between "infeasible by 16.7%"
and "infeasible by 1.6%", and with bf16 optimizer moments (6 B/param steady) it flips to feasible.

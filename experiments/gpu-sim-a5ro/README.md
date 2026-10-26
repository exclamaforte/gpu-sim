# Reference parameter counts, and the convention that reproduces them (gpu-sim-a5ro)

`total_params()` in `epic_cluster/parallelism_search.py` currently computes

    per_layer = 4*d*d + 3*d*ff + 2*d      # K/V charged at FULL width

which is MHA arithmetic applied to GQA models, and it counts one embedding. Measured against published
totals:

| model | current formula | two-embedding, GQA-aware | published | error today | error with fix |
|---|---|---|---|---|---|
| llama3-8b | 8.30 | 8.03 | 8.03 B | +3.5% | -0.00% |
| llama3-70b | 82.0 | 70.55 | 70.6 B | +16.2% | -0.07% |
| llama3-405b | 467.2 | 405.85 | ~405 B | +15.4% | +0.21% |
| qwen2.5-32b | 34.4 | 32.76 | 32.5 B | +5.8% | +0.81% |

The two-embedding convention is the one that reproduces published totals within 0.81% across the family.
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

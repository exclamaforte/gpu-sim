# gpu-sim-u2o.9.2 REPORT: concat vs cross_attention fusion

Matched arms: same 39M-class encoder (d512/8h/12L/dff2048), hparams
(20 epochs / batch 32 / AdamW 3e-4 cosine / seed 0), split (9.5
family_holdout_v1: test relu, val mul, train 342), sigma-floor NLL --
only fusion_type differs. Cross-attention carries +526,848 params
(+1.3%: QKVO + LayerNorm); counts recorded, mechanism comparison stated
as such. Job 7905f1cc exit 0 (~40 s/arm).

## Numbers (holdout test, log-space mean abs err)

| triple | concat (38,950,658) | cross_attention (39,477,506) |
|---|---|---|
| latency->latency | 0.1417 | 0.1730 |
| latency->memory | 0.2591 | 0.2688 |

Verdict: **concat**. Cross-attention loses on both triples despite more
params. The existing default stands; no fusion change. Same verdict at
the accidental 20M scale (concat 0.166/0.252 vs cross 0.168/0.294 --
superseded outputs, recorded here not hidden).

Reproduction check: this concat arm (0.142/0.259) reproduces 9.1's concat
(0.144/0.260) within GPU jitter -- same seed, same code path, same split.

## Three process failures, all mine, all recorded

1. Threading fusion_type dropped the numeric_dim line (26-dim encoder vs
   33-dim inputs) -- caught by CPU smoke before GPU; pinned by
   test_numeric_dim_survives_fusion_kwargs.
2. The same edit class ate --resume -- 1 s queue fail, no GPU burned.
3. The same class again dropped n_heads/n_layers/d_ff: both arms silently
   built 6-layer 20M models while metrics recorded 12. Caught by param
   census (20.0M vs 38.9M), diagnosed by ckpt key-set diff (strict subset),
   fixed + runtime layers-assert + test_full_dims_survives pinning
   38,950,658. Stale 20M outputs wiped and superseded (this rerun).
   Standing rule: after every multi-line edit, diff-against-master AND
   verify no '-' line is load-bearing (a skim audit passed a deletion).

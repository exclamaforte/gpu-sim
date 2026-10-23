# gpu-sim-84z: dtype boundary
Verdict: MIXED under naive/corrected; bytes-keyed predictor passes all (median 0.0206, max 0.0675).
fp16 fails naive fp32 lookup at 2^24 (3.18-3.51) and passes at floor sizes (0.004-0.029) and 2^20 (0.036-0.068). Byte-corrected (half bandwidth) reduces but does not fix (1.13-1.30). Bytes-keyed (evaluate fp32 curve at same byte footprint n/2) fixes all (max 0.0675).
Finding: the table's natural x-axis is bytes, not element count; element count only worked because all existing cases were fp32. fp16 at 16.7M elements = 100.7MB matches fp32 at 8.4M (11.515 vs 11.774 us).
See metrics.json.

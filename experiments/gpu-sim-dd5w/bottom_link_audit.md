# Bottom-link audit (gpu-sim-dd5w)

Which measured facts have a verified raw -> facts check (recompute from raw bytes).

| artifact | raw file | structured? | raw->facts test | status |
|---|---|---|---|---|
| gpu-sim-7uz2 | act_curve_raw.json | yes (JSON) | test_7uz2_raw_to_facts.py | VERIFIED |
| gpu-sim-3fog | bw_raw.json | yes (JSON) | test_3fog_raw_to_facts.py | VERIFIED |
| gpu-sim-mxaz | raw_bf16.log / raw_fp32.log | no (logs only) | none | GAP (raw is unstructured log) |
| gpu-sim-37xk | measured_opt_bytes.json | yes (JSON) | test_37xk_raw_to_facts.py | VERIFIED |
| gpu-sim-afbg | (none; derived) | n/a | n/a | derived, not measured |

Verified = a test parses the raw file and recomputes the anchored value from
raw bytes (peak minus params_grads), not from a precomputed field.

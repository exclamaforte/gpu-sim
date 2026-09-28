# Pre-flight Structural Validator (gpu-sim)

Parent: gpu-sim-u2o.6.5
Uses our `libfakegpu.so` + `torch_fakegpu/` package.

---

## 1. Validation Pipeline

Before a generated CUDA program touches the real RTX 5090:

1. Compile with `nvcc` (syntax/link check)
2. Run under `LD_PRELOAD=./libfakegpu.so` with `FAKEGPU_NUM_DEVICES=1`
3. Verify `torch.cuda.device_count() == 1`, `cudaSuccess` for all calls
4. Confirm no interposer errors (no real GPU needed; runs on CPU)
5. Only if pre-flight passes, submit to real GPU measurement harness (`safe_harness.py`)

---

## 2. Integration

`profile_fake_gpu.py` serves as the reference script for synthetic profile validation.
`predictor_model.py` uses synthetic traces to estimate runtime; real GPU measurements calibrate the predictor.

#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
python3 -c "
import cy0_l3
try:
    cy0_l3.plan({'n_layers':32,'param_bytes':7e9*2}, 'specs/h100_rack.json', dp=1, tp=1, pp=1, ep=1, zero_stage=0, seq=8192, batch=16)
    raise SystemExit('FAIL: 7B tp=1 plan accepted (should raise)')
except cy0_l3.PlanError as e:
    assert '112.0 GB' in str(e), e
    print('OK: 7B tp=1 refused:', e)
"

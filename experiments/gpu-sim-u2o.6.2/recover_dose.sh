#!/usr/bin/env bash
# gpu-sim-u2o.6.2 recovery exercise -- GPU DOSE (one ts job, both arms).
# See recover_exercise.py for the design and the pulse-safety interlock:
# no reset pulse can fire in any path (healthy->NONE; unhealthy requires
# observed foreign holders, else the dose aborts with exit 2).
set -uo pipefail
export U2O_REC_OUTDIR="${U2O_REC_OUTDIR:-/home/gabe/Documents/gpu-sim-runs/u62-recover-20261007}"
mkdir -p "$U2O_REC_OUTDIR"
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.6.2/experiments/gpu-sim-u2o.6.2
python3 recover_exercise.py
rc=$?
echo "EXERCISE EXIT: $rc (0=both arms, 2=safety abort, 10=build fail, 11=calibrator fail, 12=holder self-exited)"
exit $rc

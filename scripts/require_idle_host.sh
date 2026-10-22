#!/usr/bin/env bash
# require_idle_host.sh — refuse to start a measurement on a contended box.
#
# WHY THIS EXISTS (gpu-sim-2qe, 2026-10-22):
# Two gpqueue jobs (20261022t053548, 20261022t053800) failed with inner ranks
# exit 124 and NO isend in the trace. At the time this looked like load: the
# container has nproc=2 and host loadavg sat at 79-113, so timeout(1) inside
# run_measure.sh (120s per rank) could plausibly fire during startup.
#
# THAT CAUSAL CLAIM IS NOW WITHDRAWN. A later run at loadavg 85 completed
# normally (exit0=0 exit1=0, 32 isends, "VALUE OK (all 131072 elements == 3.0)"),
# so load is not sufficient to explain those failures. The better-supported
# explanation is the interposer bug itself: those two jobs ran out of
# .lab/wt/gpu-sim-hfk, whose src/fakegpu_cuda.c still contained the pre-fix
# fg_write_random copies (8 occurrences) and had no rebuilt libfakegpu.so, while
# the successful run came from .lab/wt/gpu-sim-2qe with the fix built. Corrupt
# data in NCCL's own bootstrap copies is a plausible way to hang BEFORE any
# isend appears. Unproven, and recorded as such.
#
# So this guard is HYGIENE, not the explanation. The ceiling is deliberately
# loose (64x nproc) because two runs at loadavg ~85 completed correctly on
# 2026-10-22 while the failing ones were explained by an unbuilt fix, not by
# load. These jobs are mostly blocked on modelled fabric delays, so host load
# predicts very little: a ceiling near 4x nproc would have refused runs that
# worked. What the guard is genuinely for is (a) printing loadavg and nproc into
# the run log, which IS reproducible evidence, and (b) stopping a pathologically
# saturated box. It is not a predictor of which runs will succeed. What survives unchanged from the incident:
#   * exit 124 with NO isend means the ranks never reached the fabric -- report
#     that as a distinct outcome, never as "the collective hangs";
#   * every run log records loadavg and nproc;
#   * check that the tree you are running from actually has the fix built
#     (grep the source for the old symbol, and check the .so mtime).
#
# exit 78 = EX_CONFIG (sysexits.h): "the harness refused, on purpose".
# exit 0  = the host looks usable, continue with the measurement.
#
# Usage:
#   scripts/require_idle_host.sh              # default ceiling: 64x nproc
#   scripts/require_idle_host.sh --max 8      # absolute ceiling on loadavg
#   scripts/require_idle_host.sh && bash run_measure.sh ...
set -u

MAX_FACTOR=64
MAX_ABS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --max) MAX_ABS="${2:-}"; shift 2 ;;
        --factor) MAX_FACTOR="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "require_idle_host: unknown argument $1" >&2; exit 64 ;;
    esac
done

NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
CEILING=$(awk -v n="$NPROC" -v f="$MAX_FACTOR" 'BEGIN{printf "%.2f", n*f}')
if [ -n "$MAX_ABS" ]; then CEILING="$MAX_ABS"; fi

# awk comparison so non-integer loadavg values work.
OVER=$(awk -v l="$LOAD1" -v c="$CEILING" 'BEGIN{print (l+0 > c+0) ? 1 : 0}')

if [ "$OVER" = "1" ]; then
    cat >&2 <<EOF
REFUSED: host is contended, this would not be a measurement.
  loadavg(1m) = $LOAD1
  nproc       = $NPROC
  ceiling     = $CEILING  (${MAX_FACTOR}x nproc unless --max was given)
Consequences of running anyway, observed twice on 2026-10-22: timeout(1) fires
inside run_measure.sh (timeout 120s per rank), both ranks exit 124 with NO isend
in the trace, and the job reads as "failed" when nothing was measured.
Re-queue when loadavg falls below the ceiling, or verify the change with a
GPU-free test (see tests/test_fakegpu_memcpy.c for the pattern).
EOF
    exit 78
fi

echo "require_idle_host: ok (loadavg $LOAD1 <= $CEILING, nproc $NPROC)"
exit 0

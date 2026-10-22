#!/usr/bin/env bash
# require_idle_host.sh — refuse to start a measurement on a contended box.
#
# WHY THIS EXISTS (gpu-sim-2qe, 2026-10-22):
# Two gpqueue jobs (20261022t053548-93b3a909, 20261022t053800-f4c7c95c) were
# reported as "failed" with inner ranks exit 124 and NO isend in the trace. Both
# were timeout(1) firing inside run_measure.sh (timeout 120s per rank) while the
# container had nproc=2 and the host load average sat at 79-104. Nothing was
# measured: the ranks never reached the fabric. A run under that load is not a
# result, and "exit 124 with no isend" must not be read as "the collective hangs".
#
# exit 78 = EX_CONFIG (sysexits.h): "the harness refused, on purpose".
# exit 0  = the host looks usable, continue with the measurement.
#
# Usage:
#   scripts/require_idle_host.sh              # default ceiling: 4x nproc
#   scripts/require_idle_host.sh --max 8      # absolute ceiling on loadavg
#   scripts/require_idle_host.sh && bash run_measure.sh ...
set -u

MAX_FACTOR=4
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

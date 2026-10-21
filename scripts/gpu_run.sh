#!/usr/bin/env bash
# gpu_run.sh -- run a GPU/NCCL command with a MANDATORY wall-clock bound.
#
# WHY THIS EXISTS: on 2026-08-22 twice, a pkb scenario was launched directly
# from a shell with no timeout. The first ran 30:49 at 80% CPU holding 878 MiB
# of the SHARED GPU and wedged its session; the second ran 26:21 the same way.
# A run that can hang must never be started without the bound that ends it.
# The bound belongs IN THE COMMAND, not in a comment -- that is why this
# wrapper exists rather than a note in the README.
#
# USAGE:
#   scripts/gpu_run.sh <timeout> [--log FILE] -- <command> [args...]
#
#   timeout   REQUIRED, e.g. 60s. There is no default: a default is a
#             forgotten bound.
#   --log     Optional file to write the command's combined output to.
#   --        Everything after it is the command to run.
#
# EXIT STATUS: the command's exit status, or 124 if the timeout fired (which
# is how a HUNG run is distinguished from a failing one), or 2 on usage error.
#
# REMINDER for real-GPU runs on this shared box: this wrapper bounds the run,
# it does not SERIALISE it. A real-GPU run that another lab could collide
# with must go through ts (gpqueue) with --max-runtime and record its job id;
# use this wrapper for the bound, ts for the scheduling.

set -uo pipefail

usage() {
    echo "usage: $0 <timeout> [--log FILE] -- <command> [args...]" >&2
    exit 2
}

[ $# -ge 1 ] || usage
case "$1" in
    -h|--help) usage ;;
esac

TIMEOUT="$1"; shift
LOGFILE=""
if [ "${1:-}" = "--log" ]; then
    LOGFILE="${2:-}"; shift 2
fi
[ "${1:-}" = "--" ] || usage
shift

case "$TIMEOUT" in
    ''|*[!0-9smh.]*) echo "error: timeout must look like 60s / 5m" >&2; exit 2 ;;
esac

if [ -n "$LOGFILE" ]; then
    # shellcheck disable=SC2086
    timeout -k 10s "$TIMEOUT" "$@" > "$LOGFILE" 2>&1
else
    timeout -k 10s "$TIMEOUT" "$@"
fi
rc=$?
if [ "$rc" -eq 124 ]; then
    echo "gpu_run.sh: TIMEOUT after $TIMEOUT -- the command was killed (a hang, not a failure)" >&2
fi
exit $rc

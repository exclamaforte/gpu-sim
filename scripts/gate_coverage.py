#!/usr/bin/env python3
"""gate.sh coverage reporter (gpu-sim-2fo4)."""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--pass", dest="passed", type=int, default=0)
    ap.add_argument("--fail", dest="failed", type=int, default=0)
    ap.add_argument("--advisory", dest="advisory", type=int, default=0)
    args = ap.parse_args()
    if args.fast:
        print("SKIPPED: suite (pytest) -- fast mode")
    if args.fast:
        print("hard: %d passed, %d failed, 1 skipped (suite) | advisory findings: %d"
              % (args.passed, args.failed, args.advisory))
    else:
        print("hard: %d passed, %d failed | advisory findings: %d"
              % (args.passed, args.failed, args.advisory))
    return 0


if __name__ == "__main__":
    sys.exit(main())

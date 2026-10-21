#!/usr/bin/env python3
"""Select the least-busy CPU over a 5s sample (gpu-sim-ep3 §3).

Prints JSON: {"sample_s":5.0,"utils":{cpu:util,...},"target":N,"target_util":u}
Exit code is always 0; the caller decides NO-QUIET-CORE from target_util > 0.25.
"""
import json
import time

SAMPLE_S = 5.0
NCPU = 32


def read_proc_stat():
    res = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            toks = line.split()
            if toks[0] == "cpu":
                continue
            idx = int(toks[0][3:])
            if idx >= NCPU:
                continue
            vals = [int(x) for x in toks[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            res[idx] = (idle, sum(vals))
    return res


def main():
    before = read_proc_stat()
    time.sleep(SAMPLE_S)
    after = read_proc_stat()
    utils = {}
    for idx in sorted(before):
        ib, tb = before[idx]
        ia, ta = after[idx]
        dt = ta - tb
        utils[idx] = round(1.0 - ((ia - ib) / dt), 4) if dt > 0 else 0.0
    target = min(utils, key=lambda k: (utils[k], k))
    print(json.dumps({
        "sample_s": SAMPLE_S,
        "utils": utils,
        "target": target,
        "target_util": utils[target],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }, indent=2))


if __name__ == "__main__":
    main()

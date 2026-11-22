#!/usr/bin/env python3
"""gpu-sim-7q3w.2: the arm's printed counters must equal the file it asserts on.

Reviewer finding on the first submission: `arm_plugin.c` printed the counters
BEFORE `ncclCommDestroy`, which is when the simulator's data path actually runs
(the device kernels are no-ops, so NCCL's proxy only moves during teardown). The
printed line therefore said `isend=0 ... bytes_claimed=0` while the JSON the
harness asserted on said `16 / 20971520`. Two numbers for one fact is exactly the
defect class this bead exists to remove, and a harness that reads only the file
would never notice.

The arm now takes ONE snapshot after destroy (`nccl_plugin_dump`, which writes the
file and returns the same struct), so these must agree field for field. This
script is the assertion, kept out of the shell heredoc so it can be read and
tested on its own.

Usage: check_counters.py <rank.log> <counters.json> <rank> [--quiet]
Exit: 0 agree, 1 disagree/missing, 2 usage.
"""
import json
import re
import sys

# printed name -> counters-file name
NAME_MAP = {
    "getProperties": "getProperties_calls",
    "devices": "devices_calls",
    "init": "init_calls",
    "listen": "listen_calls",
    "connect": "connect_calls",
    "accept": "accept_calls",
    "regMr": "regMr_calls",
    "isend": "isend_calls",
    "irecv": "irecv_calls",
    "test": "test_calls",
    "bytes_claimed": "bytes_claimed",
    "speed": "props_speed",
    "guid": "props_guid",
}


def parse_printed(log_text, rank):
    m = re.search(rf"^ARM counters rank={rank} (.*)$", log_text, re.M)
    if not m:
        return None
    out = {}
    for key, val in re.findall(r"([A-Za-z_]+)=(0x[0-9a-fA-F]+|\d+)", m.group(1)):
        out[key] = int(val, 16) if val.startswith("0x") else int(val)
    return out


def compare(printed, filed):
    problems = []
    for pname, fname in NAME_MAP.items():
        if pname not in printed:
            problems.append(f"{pname} missing from the printed line")
        elif filed.get(fname) != printed[pname]:
            problems.append(f"{pname}: printed {printed[pname]} vs file {filed.get(fname)}")
    return problems


def main(argv):
    if len(argv) < 4:
        print("usage: check_counters.py <rank.log> <counters.json> <rank>", file=sys.stderr)
        return 2
    log_path, json_path, rank = argv[1], argv[2], argv[3]
    quiet = "--quiet" in argv[4:]
    try:
        text = open(log_path, errors="replace").read()
    except OSError as e:
        print(f"ASSERT FAIL: rank{rank}: cannot read {log_path}: {e}")
        return 1
    printed = parse_printed(text, rank)
    if printed is None:
        print(f"ASSERT FAIL: rank{rank}: no 'ARM counters rank={rank}' line in {log_path}")
        return 1
    try:
        filed = json.load(open(json_path))
    except (OSError, ValueError) as e:
        print(f"ASSERT FAIL: rank{rank}: cannot read {json_path}: {e}")
        return 1
    problems = compare(printed, filed)
    if problems:
        print(f"ASSERT FAIL: rank{rank} printed counters disagree with the counters "
              f"file: " + "; ".join(problems))
        return 1
    if not quiet:
        print(f"ASSERT ok: rank{rank} printed counters agree with the counters file "
              f"(isend={printed['isend']} irecv={printed['irecv']} "
              f"bytes_claimed={printed['bytes_claimed']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

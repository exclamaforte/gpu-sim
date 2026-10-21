#!/usr/bin/env python3
"""Synthesized-trace unit tests for m3d.8's shape classifier (CPU-only, no GPU).

Each test mirrors the REAL trace structure (one record per rank per direction,
with peer recv-done lines) so no branch is reachable only on hand-written shapes.
"""
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decider

TRACE_NORMAL = os.path.join(HERE, "trace_normal.log")
PAYLOAD = 4096


def _transform_to_1x2s(src_path):
    """Derive a 1x2S trace by transforming the committed production trace:
    keep one transfer per (rank,chan) (drop the second), doubling its size."""
    lines = open(src_path).read().splitlines()
    out = []
    seen = set()
    for line in lines:
        if "isend" in line or "irecv" in line:
            m = re.search(r"rank=(\d+)", line)
            ch = re.search(r"chan=(\d+)", line)
            key = (m.group(1), ch.group(1)) if (m and ch) else None
            if key in seen:
                continue
            seen.add(key)
            line = re.sub(r"bytes=(\d+)", lambda mm: "bytes=%d" % (int(mm.group(1)) * 2), line)
            line = re.sub(r"size=(\d+)", lambda mm: "size=%d" % (int(mm.group(1)) * 2), line)
            out.append(line)
        else:
            line = re.sub(r"bytes=(\d+)", lambda mm: "bytes=%d" % (int(mm.group(1)) * 2), line)
            out.append(line)
    # structural assertion: one isend per rank
    counts = {}
    for line in out:
        if "isend" in line:
            r = re.search(r"rank=(\d+)", line).group(1)
            counts[r] = counts.get(r, 0) + 1
    ranks = {re.search(r"rank=(\d+)", l).group(1) for l in open(src_path) if "isend" in l}
    for r in ranks:
        assert counts.get(r, 0) == 1, (r, counts.get(r, 0))
    fd, path = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def test_2xS_on_committed_production_trace():
    # The classifier must label the committed production trace as 2S.
    assert os.path.exists(TRACE_NORMAL), "trace_normal.log missing"
    shape = decider.classify_trace(TRACE_NORMAL, PAYLOAD)
    assert shape.startswith("2xS:"), shape


def test_1x2S_derived_from_production_trace():
    path = _transform_to_1x2s(TRACE_NORMAL)
    shape = decider.classify_trace(path, PAYLOAD)
    assert "1x2S: one transfer of 2 payload units per direction" in shape, shape
    os.remove(path)


def test_2xS_one_channel_synthetic():
    lines = [
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=1 bytes=4096 tag=0 dtype=float32 digest=aaaa1111 cnt_isend=1",
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=2 bytes=4096 tag=1 dtype=float32 digest=bbbb2222 cnt_isend=2",
        "rank=1 v12.isend isend peer=0 chan=0 slot=0 seq=1 bytes=4096 tag=0 dtype=float32 digest=cccc3333 cnt_isend=1",
        "rank=1 v12.isend isend peer=0 chan=0 slot=0 seq=2 bytes=4096 tag=1 dtype=float32 digest=dddd4444 cnt_isend=2",
    ]
    fd, p = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    assert decider.classify_trace(p, PAYLOAD).startswith("2xS:"), decider.classify_trace(p, PAYLOAD)
    os.remove(p)


def test_split_across_channels():
    lines = [
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=1 bytes=4096 tag=0 dtype=float32 digest=dddd4444 cnt_isend=1",
        "rank=0 v12.isend isend peer=1 chan=1 slot=0 seq=1 bytes=4096 tag=1 dtype=float32 digest=eeee5555 cnt_isend=1",
    ]
    fd, p = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    shape = decider.classify_trace(p, PAYLOAD)
    assert "SPLIT ACROSS channel ids" in shape, shape
    os.remove(p)


def test_duplicate_reporting():
    lines = [
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=1 bytes=4096 tag=0 dtype=float32 digest=ffff6666 cnt_isend=1",
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=2 bytes=4096 tag=1 dtype=float32 digest=ffff6666 cnt_isend=2",
    ]
    fd, p = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    shape = decider.classify_trace(p, PAYLOAD)
    assert shape.startswith("duplicate reporting"), shape
    os.remove(p)


def test_fractional_size():
    lines = [
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=1 bytes=2048 tag=0 dtype=float32 digest=gggg7777 cnt_isend=1",
        "rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=2 bytes=2048 tag=1 dtype=float32 digest=hhhh8888 cnt_isend=2",
    ]
    fd, p = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    shape = decider.classify_trace(p, PAYLOAD)
    assert shape == "UNCLASSIFIED", shape
    os.remove(p)


def test_unclassified():
    lines = ["rank=0 v12.isend isend peer=1 chan=0 slot=0 seq=1 bytes=123 tag=0 dtype=float32 digest=iiii9999 cnt_isend=1"]
    fd, p = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    shape = decider.classify_trace(p, PAYLOAD)
    assert shape == "UNCLASSIFIED", shape
    os.remove(p)


def test_full_payload_predicate_rejects_fractional():
    phases = {(0, "0"): [PAYLOAD // 2, PAYLOAD // 2]}
    digests = {(0, "0"): ["x", "y"]}
    shape = decider.classify_shape(phases, digests, PAYLOAD)
    assert shape != "2xS: two transfers of one payload unit per direction (2S per rank across two records)", shape


if __name__ == "__main__":
    for name in [n for n in globals() if n.startswith("test_")]:
        globals()[name]()
        print("PASS", name)
    print("all shape tests passed")

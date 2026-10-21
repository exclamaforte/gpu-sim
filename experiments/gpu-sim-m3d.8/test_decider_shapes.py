#!/usr/bin/env python3
"""Synthesized-trace unit tests for m3d.8's shape classifier (CPU-only, no GPU).

Each test builds a minimal trace in the plugin's structured format and asserts
the classifier's verdict, so every branch is exercised without a real run.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decider


def _trace(lines):
    fd, path = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _rec(rank, chan, size, digest, cnt):
    return ("rank=%d v12.isend isend peer=1 chan=%s slot=0 seq=1 bytes=%d "
            "tag=0 dtype=float32 digest=%s cnt_isend=%d" % (rank, chan, size, digest, cnt))


PAYLOAD = 4096


def test_2xS_one_channel():
    # observed shape: two full-payload transfers on one channel id, distinct digests
    lines = [
        _rec(0, 0, PAYLOAD, "aaaa1111", 1),
        _rec(0, 0, PAYLOAD, "bbbb2222", 2),
    ]
    p = _trace(lines)
    assert decider.classify_trace(p, PAYLOAD).startswith("2S:"), decider.classify_trace(p, PAYLOAD)
    os.remove(p)


def test_1x2S_one_channel():
    # single transfer carrying 2 payloads on one channel id
    lines = [_rec(0, 0, 2 * PAYLOAD, "cccc3333", 1)]
    p = _trace(lines)
    shape = decider.classify_trace(p, PAYLOAD)
    assert "single transfer of multiple payload units" in shape, shape
    os.remove(p)


def test_split_across_channels():
    # one transfer of S per channel on two or more channels
    lines = [
        _rec(0, 0, PAYLOAD, "dddd4444", 1),
        _rec(0, 1, PAYLOAD, "eeee5555", 1),
    ]
    p = _trace(lines)
    shape = decider.classify_trace(p, PAYLOAD)
    assert "SPLIT ACROSS channel ids" in shape, shape
    os.remove(p)


def test_duplicate_reporting():
    # two records with identical digests on one channel id
    lines = [
        _rec(0, 0, PAYLOAD, "ffff6666", 1),
        _rec(0, 0, PAYLOAD, "ffff6666", 2),
    ]
    p = _trace(lines)
    shape = decider.classify_trace(p, PAYLOAD)
    assert shape.startswith("duplicate reporting"), shape
    os.remove(p)


def test_fractional_size():
    # two transfers of a fractional size (S/2) on one channel -> not full-payload
    lines = [
        _rec(0, 0, PAYLOAD // 2, "gggg7777", 1),
        _rec(0, 0, PAYLOAD // 2, "hhhh8888", 2),
    ]
    p = _trace(lines)
    shape = decider.classify_trace(p, PAYLOAD)
    assert shape == "UNCLASSIFIED", shape
    os.remove(p)


def test_unclassified():
    # unrecognised shape -> UNCLASSIFIED
    lines = [_rec(0, 0, 123, "iiii9999", 1)]
    p = _trace(lines)
    shape = decider.classify_trace(p, PAYLOAD)
    assert shape == "UNCLASSIFIED", shape
    os.remove(p)


def test_full_payload_predicate_rejects_fractional_in_2S_branch():
    # ensure the 'full-payload' branch does not fire for fractional sizes
    phases = {(0, "0"): [PAYLOAD // 2, PAYLOAD // 2]}
    digests = {(0, "0"): ["x", "y"]}
    shape = decider.classify_shape(phases, digests, PAYLOAD)
    assert shape != "2S: multiple full-payload transfers on the same channel id (distinct phases, distinct digests)", shape


if __name__ == "__main__":
    for name in [n for n in globals() if n.startswith("test_")]:
        globals()[name]()
        print("PASS", name)
    print("all shape tests passed")

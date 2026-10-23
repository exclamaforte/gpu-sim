"""ncclReduceScatter takes its count BY VALUE, not as a pointer.

Real NCCL declares::

    ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
        size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
        ncclComm_t comm, cudaStream_t stream);

The shim once declared ``const size_t *recvcount``. torch is compiled against
the real header, so it passes an element COUNT where we expected a POINTER;
the interposer dereferenced e.g. 0x290900 and died with SEGV_MAPERR inside
FSDP's backward (reduce_scatter_single -> ncclReduceScatter). Forward
all_gather worked, so FSDP looked payload-blocked when it was ABI-blocked.

A declaration-parse test is enough here, same as test_nccl_dtype_abi.py:
the signature IS the ABI, and the failure mode is a crash, not a wrong number.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHIM = REPO / "shim" / "nccl.h"
IMPL = REPO / "src" / "fakegpu_nccl.c"


def test_shim_declares_scalar_recvcount():
    """The shim must match real NCCL: scalar size_t, no pointer, no array."""
    text = SHIM.read_text()
    m = re.search(
        r"ncclResult_t\s+ncclReduceScatter\((.*?)\);", text, re.DOTALL)
    assert m, "ncclReduceScatter declaration not found in shim/nccl.h"
    params = m.group(1)
    assert "recvcount" in params, f"recvcount param missing: {params!r}"
    assert not re.search(r"\*\s*recvcount", params), (
        f"recvcount must not be a pointer (torch passes size_t by value): {params!r}")
    assert not re.search(r"recvcount\s*\[", params), (
        f"recvcount must not be an array: {params!r}")
    assert re.search(r"size_t\s+recvcount", params), (
        f"recvcount must be scalar size_t: {params!r}")


def test_impl_matches_shim_scalar_recvcount():
    """The implementation must take the same scalar it declares."""
    text = IMPL.read_text()
    m = re.search(
        r"ncclResult_t\s+ncclReduceScatter\((.*?)\)\s*\{", text, re.DOTALL)
    assert m, "ncclReduceScatter definition not found in src/fakegpu_nccl.c"
    params = m.group(1)
    assert re.search(r"size_t\s+recvcount", params), (
        f"impl must take scalar size_t recvcount: {params!r}")


def _body_after(text: str, m: "re.Match") -> str:
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i]


def test_impl_does_not_dereference_recvcount():
    """Belt and braces: no recvcount[ indexing anywhere in the definition."""
    text = IMPL.read_text()
    m = re.search(
        r"ncclResult_t\s+ncclReduceScatter\((.*?)\)\s*\{", text, re.DOTALL)
    assert m
    assert "recvcount[0]" not in _body_after(text, m)

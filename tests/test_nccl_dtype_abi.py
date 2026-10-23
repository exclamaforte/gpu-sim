"""The NCCL datatype constants in shim/nccl.h must equal real NCCL's.

This is a regression test for a defect that survived an entire measurement
campaign because it was SELF-CONSISTENT: shim/nccl.h had

    ncclFloat32 = 6, ncclFloat64 = 7, ncclFloat16 = 8

while real NCCL has

    ncclFloat16 = 6, ncclFloat32 = 7, ncclFloat64 = 8

torch passes the constants it was compiled against, so the shim decoded float32 as
float64 and reported 8 bytes per element instead of 4. Every float32 collective
volume -- the dominant dtype -- was exactly 2x too large. Ratios, orderings and
conclusions were unaffected, so nothing looked wrong; it surfaced only when two
independent probes disagreed by exactly 2x on the same tensor.

A header-parse test is enough here: the constants ARE the ABI, and the failure
mode is a changed integer, not changed behaviour.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[1] / "shim" / "nccl.h"

# Values from real nccl.h (NVIDIA/nccl, src/nccl.h.in), which is what torch is
# compiled against. If upstream ever renumbers these, torch's ABI changes too and
# this table must move with it.
EXPECTED = {
    "ncclInt8": 0, "ncclChar": 0,
    "ncclUint8": 1,
    "ncclInt32": 2, "ncclInt": 2,
    "ncclUint32": 3,
    "ncclInt64": 4, "ncclLong": 4,
    "ncclUint64": 5,
    "ncclFloat16": 6, "ncclHalf": 6,
    "ncclFloat32": 7, "ncclFloat": 7,
    "ncclFloat64": 8, "ncclDouble": 8,
    "ncclBfloat16": 9, "ncclbfloat16": 9,
}

# Bytes per element, for the sizes the harness actually moves.
BYTES = {0: 1, 1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2}


def _defined_constants() -> dict[str, int]:
    text = SHIM.read_text()
    out: dict[str, int] = {}
    for m in re.finditer(r"#define\s+(nccl\w+)\s+\(\(ncclDataType_t\)\s*(\d+)\)", text):
        out[m.group(1)] = int(m.group(2))
    return out


def test_defined_constants_are_present():
    defined = _defined_constants()
    assert defined, f"no datatype constants found in {SHIM}"
    assert "ncclFloat32" in defined


def test_constants_match_real_nccl():
    defined = _defined_constants()
    wrong = {name: (defined[name], real) for name, real in EXPECTED.items()
             if name in defined and defined[name] != real}
    assert not wrong, (
        "shim datatype constants disagree with real NCCL "
        f"(name: (shim, real)): {wrong}. torch passes real NCCL values, so any "
        "difference silently mislabels buffer sizes."
    )


def test_float32_is_not_confused_with_float64():
    """The specific defect: these two adjacent values swap a 2x size error."""
    defined = _defined_constants()
    assert defined["ncclFloat32"] != defined["ncclFloat64"]
    assert BYTES[defined["ncclFloat32"]] == 4
    assert BYTES[defined["ncclFloat64"]] == 8
    assert BYTES[defined["ncclFloat16"]] == 2


def test_no_duplicate_value_for_distinct_sizes():
    """A permutation that keeps every name unique can still collide two SIZES."""
    defined = _defined_constants()
    by_size: dict[int, set[str]] = {}
    for name, value in defined.items():
        by_size.setdefault(BYTES.get(value, -1), set()).add(name)
    for size, names in by_size.items():
        if size == -1:
            continue
        # The only legitimate collisions are aliases of the same size
        # (Half/float16, Float/float32, Char/Int8, Long/Int64...).
        sizes_declared = {BYTES[defined[n]] for n in names}
        assert sizes_declared == {size}, f"names {sorted(names)} map to one constant but several sizes"

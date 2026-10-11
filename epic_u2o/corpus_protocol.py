"""Protocol labelling for corpora. gpu-sim-u2o.11.2 / 6.7.1.

The PI's condition on the fast-path corpus, which is not optional and is the
whole reason a fast corpus can be produced at all:

> every row records ITS PROTOCOL, because ai_cv and the absolute medians mean
> different things under the single-run and the per-kernel-shared-context
> paths -- the same statistic moves 3.2x between them, which you measured. So a
> new corpus is ADDITIVE and LABELLED, and nothing in it may be pooled with the
> 264 reference rows or the anchor's cells.

The 3.2x is the reason this is code and not a convention. A pooled corpus
whose rows were measured under two protocols does not merely have noisier
labels; its `ai_cv` column means two different things in two halves of the same
file, and a noise-aware loss (gpu-sim-u2o.9.3 predicts a log-variance) would
be fitting a mixture of two definitions of its own target. Nothing downstream can
detect that, because the numbers are all individually plausible.

So: every row carries a protocol id, and any operation that combines rows
across protocols is refused rather than performed. Pooling is still possible --
it is a legitimate thing to want once someone has shown the protocols agree --
but it has to be a decision someone makes, not a default that happens.
"""
from __future__ import annotations

import json
from collections import defaultdict

# Every protocol this lab has actually MEASURED a corpus under. An id that is
# not here cannot be pooled with anything, which is the safe default: an
# unrecognised protocol is treated as its own singleton group rather than
# assumed compatible with a known one.
# gpu-sim-u2o.6.7.1 review: the registry had THREE DIFFERENT FIELD SETS, so
# `PROTOCOLS[name]["label_range_x"]` raised KeyError for perproc_7_1_driver --
# which is missing five fields a consumer would reasonably index. A registry that
# varies its shape is a registry that fails at the CONSUMER, far from here, with
# a KeyError naming nothing.
#
# REGISTRY_FIELDS is the union of what any entry carries. Every entry is
# normalised to carry all of them, with None meaning "not measured for this
# protocol" -- an explicit absence rather than a missing key.
REGISTRY_FIELDS = (
    "measured", "rows", "wall_sec", "ex_per_sec", "invocations",
    "label_range_x", "median_ai_cv", "executable", "process_model",
    "timed_launches_per_example", "median_rel_diff_vs_reference",
    "not_measured_reason",
)

PROTOCOLS = {
    "single_run_7_1": {
        "measured": True,
        "rows": 264,
        "wall_sec": 110.52,
        "ex_per_sec": 2.389,
        "invocations": 2,
        "timed_launches_per_example": 40,
        "process_model": "one process PER INVOCATION (7.1's own harness)",
        "executable": "one binary per kernel, via kernel_driver.synthesize()",
        "median_ai_cv": 0.0854,
        "label_range_x": 2.70,
        "authority": "REFERENCE. gpu-sim-u2o.8.3's feature vector, 9.4's roofline "
                     "baseline and 9.5's acceptance were all measured against "
                     "this. FROZEN: do not regenerate on the fast path.",
    },
    "perproc_7_1_driver": {
        "measured": True,
        "rows": 264,
        "ex_per_sec": 3.78,
        "process_model": "one process per invocation, MY driver",
        "median_ai_cv": 0.0724,
        "median_rel_diff_vs_reference": 0.125,
        "authority": "diagnostic only; an elimination arm",
    },
    "perkernel_shared_ctx": {
        "measured": True,
        "rows": 264,
        "wall_sec": 0.52,
        "ex_per_sec": 510.80,
        "invocations": 2,
        "timed_launches_per_example": 40,
        "process_model": "ONE process, ONE CUDA context",
        "executable": "7.1's own per-kernel binaries, each its own nvcc -c "
                      "translation unit, arity-sized cudaMalloc",
        "median_ai_cv": 0.0307,
        "median_rel_diff_vs_reference": 0.094,
        "authority": "SCALE ONLY. Additive and labelled; may not be pooled "
                     "with the reference rows. A ~9.4% median offset against "
                     "the reference remains unexplained.",
    },
    "single_run_stages_2_5": {
        # gpu-sim-u2o.11.3: the scale-up rows (stages 2-5). Same process shape
        # as single_run_7_1 -- one binary per kernel via kernel_driver.
        # synthesize(), 2 invocations x 20 timed launches -- but a separate
        # protocol: stamping these single_run_7_1 would claim reference
        # status for non-reference kernels. rows/wall_sec/ex_per_sec/
        # median_ai_cv are filled at landing (this entry's own bead).
        "measured": True,
        "invocations": 2,
        "timed_launches_per_example": 40,
        "process_model": "one process PER INVOCATION (7.1's own harness)",
        "executable": "one binary per kernel, via kernel_driver.synthesize()",
        "authority": "SCALE ONLY. Additive and labelled; may not be pooled "
                     "with the reference rows without a recorded decision.",
    },
}


class ProtocolError(RuntimeError):
    """Raised when rows from different protocols are combined without a decision."""


def protocol_of(row: dict) -> str:
    """The protocol a row was measured under.

    A row without one is not a row this lab can reason about. Treating a
    missing protocol as "compatible" is how a mixed corpus would arrive
    unlabelled, so this raises rather than defaulting.
    """
    p = row.get("protocol")
    if not p:
        raise ProtocolError(
            f"row {row.get('kernel_hash', '<no hash>')} carries no 'protocol' "
            f"field. Every row must record the protocol it was measured under: "
            f"ai_cv and the absolute medians are protocol-dependent, and a "
            f"corpus that mixes them is a corpus whose target has two "
            f"definitions. Known protocols: {sorted(PROTOCOLS)}")
    return p


def describe(protocol: str) -> dict:
    if protocol not in PROTOCOLS:
        raise ProtocolError(
            f"unknown protocol {protocol!r}. An unrecognised protocol is "
            f"treated as its own group, never as compatible with a known one. "
            f"Known: {sorted(PROTOCOLS)}")
    return PROTOCOLS[protocol]


def group_by_protocol(rows: list) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[protocol_of(r)].append(r)
    return dict(out)


def assert_single_protocol(rows: list, *, what: str = "corpus") -> str:
    """Refuse to operate on a mixed corpus. Returns the single protocol id."""
    groups = group_by_protocol(rows)
    if len(groups) > 1:
        detail = ", ".join(f"{k} ({len(v)} rows)" for k, v in sorted(groups.items()))
        raise ProtocolError(
            f"the {what} mixes {len(groups)} protocols: {detail}. Per the "
            f"corpus condition, a fast-path corpus is ADDITIVE and LABELLED "
            f"and may not be pooled with the 264 reference rows or the "
            f"anchor's cells without a recorded decision. If the protocols have "
            f"been shown to agree, record that decision explicitly rather than "
            f"reaching for this.")
    return next(iter(groups)) if groups else ""


def poolable(rows: list, *, target_protocol: str | None = None) -> list:
    """Rows that may legally be combined, given a decision.

    With no target, nothing pools: the conservative default. With a target
    protocol named, its rows come back -- and the caller has therefore made the
    decision explicit at the call site, which is the point.
    """
    if target_protocol is None:
        return []
    describe(target_protocol)          # rejects an unknown one
    return [r for r in rows if protocol_of(r) == target_protocol]


def write_rows(path, rows) -> None:
    """Write rows with the protocol stamped on every one.

    Stamping at WRITE time rather than at read time is deliberate: a row that
    has been serialised without its protocol is unrecoverable, whereas a reader
    that forgets to check one is caught by assert_single_protocol.
    """
    out = []
    for r in rows:
        p = r.get("protocol")
        if not p:
            raise ProtocolError(
                f"refusing to write a row with no protocol: "
                f"{r.get('kernel_hash', '<no hash>')}. Stamping at write time "
                f"is the only point where the label is guaranteed present.")
        describe(p)
        out.append(r)
    with open(path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")


def _normalise_registry() -> dict:
    """Give every entry the same fields, so no consumer can KeyError.

    Missing measurements become None plus a `not_measured_reason`, because a
    protocol that was never measured is a fact, and silently omitting the key
    hides it behind an exception somewhere else.
    """
    out = {}
    for name, entry in PROTOCOLS.items():
        e = dict(entry)
        missing = [f for f in REGISTRY_FIELDS if f not in e and f != "not_measured_reason"]
        for f in missing:
            e[f] = None
        if missing and "not_measured_reason" not in e:
            e["not_measured_reason"] = (
                "not measured for this protocol: " + ", ".join(sorted(missing)))
        out[name] = e
    return out


PROTOCOLS = _normalise_registry()


def registry_field(name: str, field: str):
    """Read a registry field WITHOUT raising.

    Returns REGISTRY_MISSING rather than KeyError for an unknown field, so a
    consumer asking a question the registry cannot answer gets a value it can
    test, not an exception naming the wrong thing.
    """
    e = PROTOCOLS.get(name)
    if e is None:
        return REGISTRY_MISSING
    return e.get(field, REGISTRY_MISSING)


class _Missing:
    def __repr__(self):
        return "REGISTRY_MISSING"

    def __bool__(self):
        return False


REGISTRY_MISSING = _Missing()

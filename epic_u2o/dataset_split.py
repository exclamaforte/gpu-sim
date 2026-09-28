"""Train/test split design against the REAL measured dataset. gpu-sim-u2o.7.7.

gpu-sim-u2o.7.7 was reopened by the 6.15 audit with the note that split logic
"may exist in structural_split.py but there is no real dataset with runtime
labels for a split to matter yet -- meaningless until .6/.7.1-.7.5 produce real
measured data." gpu-sim-u2o.7.1 has now produced that data: 264 measured rows
(ts 20260928t074004-b429f469) with family, problem size, launch config, measured
regime and device-side runtime. So the split matters, and this module exists to
say what the split must be and to REFUSE the ones that would flatter a model.

WHAT WENT WRONG WITH THE EXISTING SPLIT, measured rather than asserted:

    assign_family(row_as_written_by_7_1) -> None
    split_by_family(real_rows)            -> train=264  val=0  test=0
    analyze_split_quality(that)           -> total_families: 0
                                             leakage_train_test: 0

All 264 rows in train, no test set at all, and the leakage analyzer reporting
ZERO leakage. The third line is the dangerous one: a check that reports "no
leakage" for a split with no test set is not a weak check, it is a FAIL-OPEN
one. It is the same shape as the preflight stub in gpu-sim-8dw and the
trust-the-exit-code probe in gpu-sim-u2o.6.3.1: a green result produced by
nothing having been tested.

So this module's primary job is to be the opposite: every degenerate case
RAISES. A split that cannot be shown to be family-disjoint is not a split.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


class SplitError(RuntimeError):
    """A split that cannot be trusted. Raised, never reported as clean."""


# --- the leakage key -------------------------------------------------------

def token_digest(row: dict) -> str:
    return row.get("token_digest", "")


def leakage_keys(row: dict) -> set:
    """Every key that, if shared across the split, means the model saw the answer.

    A key is only leakage if the MODEL can see it. The representation is the
    model's input, so the primary key is what the tokenizer emits -- NOT the
    kernel hash, which differs on all 264 rows and would therefore report zero
    leakage no matter how badly the split leaks.
    """
    keys = set()
    if token_digest(row):
        keys.add(f"token_digest={token_digest(row)}")
    return keys


def structural_family(row: dict) -> str:
    """The family a row belongs to, derived from the fields 7.1 actually writes.

    structural_split.assign_family() returns None for these rows because it
    expects metadata['family_id'] or a matching 'generator', and stage1 writes
    neither -- op_type is top-level. Falling back to a FAMILY OF ONE is a
    deliberate reading, and it is the honest one for stage1: one kernel source
    per op_type, 33 launch configurations of it. The alternative -- returning
    None and letting the caller pool every unknown into a single bucket -- is
    what produced the all-train split.
    """
    fam = row.get("structural_family") or row.get("op_type")
    if not fam:
        raise SplitError(
            f"row has no op_type and no structural_family, so it cannot be "
            f"assigned to a family: {sorted(row)[:8]}. A family of None would "
            f"pool every unlabelled row into one bucket, which is how a split "
            f"silently stops splitting.")
    return fam


# --- splitting -------------------------------------------------------------

def family_disjoint_split(rows: list, test_families: list[str],
                          val_families: list[str] | None = None,
                          test_problem_sizes: list[tuple[int, int]] | None = None,
                          ) -> dict[str, list[dict]]:
    """Hold out WHOLE families. A family is never split across the boundary.

    test_problem_sizes additionally removes size RANGES from train, for the
    separate question of generalising to unseen sizes. Both holdouts can apply
    at once, which is the strictest configuration.
    """
    test_families = set(test_families)
    val_families = set(val_families or [])
    clash = test_families & val_families
    if clash:
        raise SplitError(f"families {sorted(clash)} are in both test and val")

    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        fam = structural_family(r)
        if fam in test_families:
            buckets["test"].append(r)
        elif fam in val_families:
            buckets["val"].append(r)
        else:
            buckets["train"].append(r)

    if test_problem_sizes:
        def held(row):
            return any(lo <= row.get("problem_size", 0) <= hi
                       for lo, hi in test_problem_sizes)
        buckets["test"] += [r for r in buckets["train"] if held(r)]
        buckets["train"] = [r for r in buckets["train"] if not held(r)]
    return buckets


# --- validation that cannot pass ------------------------------------------

def validate_split(split: dict, rows: list) -> dict:
    """Raise on any degenerate split. Returns a report when it is clean.

    Every condition below is a way a split can be worthless while LOOKING fine,
    and each is a real behaviour of the pre-existing code or a plausible way to
    write a new one.
    """
    report: dict = {}
    train, val, test = split.get("train", []), split.get("val", []), split.get("test", [])

    # 1. an empty side. The pre-existing split put all 264 rows in train and
    #    the analyzer called that zero leakage.
    for name, part in (("train", train), ("val", val), ("test", test)):
        if not part:
            raise SplitError(
                f"the {name} split is EMPTY. A split with no test set measures "
                f"nothing, and a leakage check over a one-sided split reports "
                f"zero leakage by construction -- that is the fail-open result "
                f"gpu-sim-u2o.7.7 exists to remove.")

    total = len(rows)
    if len(train) + len(val) + len(test) != total:
        raise SplitError(
            f"rows lost or duplicated: {len(train)}+{len(val)}+{len(test)} != "
            f"{total}. A split that does not conserve its rows is not a split.")

    # 2. family overlap. The whole point.
    fams = {name: {structural_family(r) for r in part}
            for name, part in (("train", train), ("val", val), ("test", test))}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = fams[a] & fams[b]
        if overlap:
            raise SplitError(
                f"families {sorted(overlap)} appear in both {a} and {b}. Holding "
                f"out examples while sharing a family is the leakage this bead "
                f"exists to prevent: a model that memorised the template scores "
                f"as generalisation.")

    # 3. representation overlap, keyed on what the MODEL can see.
    keys = {name: set().union(*(leakage_keys(r) for r in part)) if part else set()
            for name, part in (("train", train), ("val", val), ("test", test))}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = keys[a] & keys[b]
        if overlap:
            raise SplitError(
                f"{len(overlap)} representation key(s) shared between {a} and {b} "
                f"(e.g. {sorted(overlap)[:2]}). Distinct kernel_hash values do "
                f"NOT make these independent: the model sees the token "
                f"sequence, and if that is shared the example is not held out.")

    # 4. the split must actually be informative: one side holding a single
    #    example is a number nobody should report.
    for name, part in (("val", val), ("test", test)):
        if len(part) < 5:
            raise SplitError(
                f"the {name} split has {len(part)} example(s). Too few to report "
                f"a metric on; a one-row test set produces a confident number "
                f"from a single measurement.")

    report.update({
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_families": sorted(fams["train"]),
        "val_families": sorted(fams["val"]),
        "test_families": sorted(fams["test"]),
        "n_representation_keys": {k: len(v) for k, v in keys.items()},
        "family_disjoint": True,
        "representation_disjoint": True,
        "rows_conserve": True,
    })
    return report


# --- the audit: what a naive split would have claimed ---------------------

def leakage_audit(rows: list, seed: int = 42) -> dict:
    """Quantify what a ROW-LEVEL split leaks, using the real representation keys.

    This is the argument for family holdout, as a number rather than a
    principle. It deliberately reports the naive split's overlap rather than
    only the clean one, because 'a random split would have leaked 100%' is the
    sentence that stops the next person from writing one.
    """
    import random
    keys = {id(r): leakage_keys(r) for r in rows}
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    tr, te = shuffled[: int(n * 0.85)], shuffled[int(n * 0.85):]

    def fam(r):
        return structural_family(r)

    tr_keys = set().union(*(keys[id(r)] for r in tr)) if tr else set()
    te_keys = set().union(*(keys[id(r)] for r in te)) if te else set()
    tr_fams = {fam(r) for r in tr}
    te_fams = {fam(r) for r in te}

    test_rows_leaked = sum(1 for r in te if keys[id(r)] & tr_keys)
    return {
        "n_rows": n,
        "n_distinct_kernel_hash": len({r.get("kernel_hash") for r in rows}),
        "n_distinct_representation_keys": len({k for v in keys.values() for k in v}),
        "rows_per_representation_key": n / max(1, len({k for v in keys.values() for k in v})),
        "row_level_split": {
            "train_rows": len(tr), "test_rows": len(te),
            "test_rows_with_a_representation_key_also_in_train": test_rows_leaked,
            "test_rows_leaked_fraction": test_rows_leaked / max(1, len(te)),
            "families_shared_train_test": sorted(tr_fams & te_fams),
            "n_families_shared": len(tr_fams & te_fams),
            "n_families_total": len({fam(r) for r in rows}),
        },
        "note": ("kernel_hash is distinct on every row, so a leakage check keyed "
                 "on kernel_hash reports 0% leakage on a split that leaks every "
                 "row. The model never sees kernel_hash; it sees the token "
                 "sequence. Keying the check on the wrong identifier is how the "
                 "pre-existing analyzer returned leakage_train_test: 0."),
    }


def digest_of(parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]

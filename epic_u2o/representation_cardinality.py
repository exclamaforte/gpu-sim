"""Expected-cardinality gate for representations. gpu-sim-u2o.8.3.

The PI's review of gpu-sim-u2o.8.3, in the form of a question worth turning
into code:

> You reported n_rows_populated 264 and n_rows_with_unrecoverable_source 0, and
> distinct_vectors 5, and the 5 is reported correctly -- but nothing asked what
> 5-over-264 IMPLIES. distinct_vectors is a present, correctly-named, reported
> derived value whose implication was never sanity-checked, which is the same
> shape as the discriminates boolean that was true for every signal and the band
> table that disagreed with its own headline. A derived value that is present
> and plausible and means less than its name suggests.

So the rule this module encodes:

    A DERIVED VALUE IS NOT DONE BECAUSE IT IS REPORTED. IT IS DONE WHEN ITS
    IMPLICATION HAS BEEN ASSERTED AGAINST SOMETHING EXPECTED.

Why a cardinality check and not a general one: a representation's whole job is
to distinguish examples, so "how many distinct things did it produce" is the
cheapest possible question with a knowable answer, and it is the question whose
absence let a 28-dimensional vector collapse 264 rows onto five points while
every field in metrics.json was correctly named and correctly computed.

This is the same defect in four costumes, and the gate is the answer to all
four: 6.6.3's `discriminates` boolean, 6.6.4's band table disagreeing with its
own headline, 7.7's 264-to-8 tokenizer collapse, and 8.3's 264-to-5 feature
vector. Every one of them was a derived value that was true, named, reported,
and meaningless at the magnitude it was reported.
"""
from __future__ import annotations

from dataclasses import dataclass


class CardinalityError(RuntimeError):
    """A representation collapsed, or separated less than was decided.

    Raised, not warned. A collapse that is merely reported is how 264 rows
    became five vectors and nobody stopped.
    """

    COLLAPSED = "COLLAPSED"
    PARTIAL = "PARTIAL"
    DISTINCT = "DISTINCT"


# --- the decision, recorded rather than buried -----------------------------
#
# The floor below is a DELIBERATE choice, not a constant discovered by looking
# at what the code happened to produce. The reasoning, on gpu-sim-u2o.7.1's 264
# measured rows:
#
#   distinct (op_type, n, grid, block) configs : 264   <- the legitimate ceiling
#   feature vector, as written                 :   5   <- 0.019 of rows
#   token digest (7.7)                         :   8   <- 0.030 of rows
#   feature vector, corrected                  : 165   <- 0.625 of rows
#
# The two failures sit an order of magnitude below the corrected value and more
# than an order of magnitude below the ceiling, so any floor has a wide, empty
# gap between "broken" and "working" and the exact value is not delicate. 0.25
# sits below the corrected vector with room to spare and an order of magnitude
# above the collapses. It is a floor, not a target: a representation that
# separates MORE than the floor is not penalised, and one that separates less is
# a stop.
DEFAULT_FLOOR_RATIO = 0.25


@dataclass
class CardinalityReport:
    n_rows: int
    n_distinct: int
    ratio: float
    floor_ratio: float
    verdict: str
    context: str
    ceiling: int | None = None
    ceiling_basis: str = ""
    residual_note: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict != CardinalityError.COLLAPSED

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_distinct": self.n_distinct,
            "ratio": round(self.ratio, 4), "floor_ratio": self.floor_ratio,
            "verdict": self.verdict, "context": self.context,
            "ceiling": self.ceiling, "ceiling_basis": self.ceiling_basis,
            "residual_note": self.residual_note, "passed": self.passed,
        }


def check_cardinality(n_rows: int, n_distinct: int, *, context: str,
                      floor_ratio: float = DEFAULT_FLOOR_RATIO,
                      ceiling: int | None = None,
                      ceiling_basis: str = "",
                      residual_note: str = "") -> CardinalityReport:
    """Classify a representation's separating power. Never raises on its own.

    Split from `assert_cardinality` so a caller can REPORT the number -- which
    is what the gate is for -- without the check itself being the thing that
    crashes a run mid-way and loses the surrounding evidence.
    """
    if n_rows <= 0:
        raise CardinalityError(
            f"{context}: n_rows is {n_rows}; there is nothing to separate, so "
            f"any ratio here would be a division of nothing.")
    ratio = n_distinct / n_rows
    if ratio < floor_ratio:
        verdict = CardinalityError.COLLAPSED
    elif ceiling is not None and n_distinct >= ceiling:
        verdict = CardinalityError.DISTINCT
    else:
        verdict = CardinalityError.PARTIAL
    return CardinalityReport(
        n_rows=n_rows, n_distinct=n_distinct, ratio=ratio,
        floor_ratio=floor_ratio, verdict=verdict, context=context,
        ceiling=ceiling, ceiling_basis=ceiling_basis,
        residual_note=residual_note,
    )


def assert_cardinality(n_rows: int, n_distinct: int, *, context: str,
                       floor_ratio: float = DEFAULT_FLOOR_RATIO,
                       ceiling: int | None = None,
                       ceiling_basis: str = "",
                       residual_note: str = "") -> CardinalityReport:
    """Raise when a representation separates less than was decided.

    The message states the IMPLICATION, because that is the part that was
    missing: not 'ratio below floor' but 'N of M rows are indistinguishable to
    the model, so nothing fitted on this can learn about the other N'.
    """
    rep = check_cardinality(n_rows, n_distinct, context=context,
                            floor_ratio=floor_ratio, ceiling=ceiling,
                            ceiling_basis=ceiling_basis,
                            residual_note=residual_note)
    if rep.verdict == CardinalityError.COLLAPSED:
        lost = n_rows - n_distinct
        raise CardinalityError(
            f"{context}: {n_distinct} distinct representation(s) for {n_rows} "
            f"rows = {rep.ratio:.1%}, below the decided floor of "
            f"{floor_ratio:.0%}. IMPLICATION, which is the part that matters: "
            f"{lost} of {n_rows} rows are INDISTINGUISHABLE to the model, so "
            f"anything fitted on this representation learns about "
            f"{n_distinct} points and nothing about the other {lost}. "
            + (f"Legitimate ceiling is {ceiling} ({ceiling_basis}). "
               if ceiling else "")
            + "A correctly-named, correctly-computed, correctly-reported "
              "derived value is not a working one. Fix the representation, or "
              "deliberately lower the floor with the reason recorded here.")
    return rep

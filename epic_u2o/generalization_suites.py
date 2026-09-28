"""Generalization suites. gpu-sim-u2o.10.3.

Three held-out suites, analogous to cclm's generalization distributions:

  S1  unseen PROBLEM SIZE   -- train on small n, test on n larger than any seen
  S2  unseen FAMILY         -- hold out entire structural families
  S3  unseen OCCUPANCY      -- hold out a launch-config regime

The point of this module is not the suite definitions, which are easy. It is
that a suite is only CLAIMED if the corpus can actually support it, and the
support check FAILS LOUD rather than reporting a degenerate split as clean.

That is not a hypothetical. gpu-sim-u2o.7.7 ran the existing splitter against
the 264 real measured rows and got:

    train=264  val=0  test=0  ...  leakage_train_test: 0

A split with no test set at all, reported as ZERO leakage. The leakage analyzer
was computing the intersection of two empty sets and calling it clean. So the
first job of any suite here is to prove it is non-degenerate, and the second is
to say plainly when the corpus cannot support it.

A suite that cannot be supported is REPORTED AS UNSUPPORTED WITH ITS ARITHMETIC.
It is not quietly dropped, because a suite that quietly does not run is
indistinguishable from one that passed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

MIN_TRAIN, MIN_EVAL = 30, 5


class SuiteError(RuntimeError):
    """A suite that cannot be run or cannot be trusted. Never reported clean."""


@dataclass
class Suite:
    name: str
    unit: str                 # what is held out
    supported: bool = True
    reason: str = ""
    train: list = field(default_factory=list)
    eval_: list = field(default_factory=list)
    held_out: list = field(default_factory=list)

    def validate(self) -> None:
        """FAIL LOUD. Every condition is a way a suite can look fine and not be.

        gpu-sim-u2o.7.7's fail-open split is the reference failure: a real
        suite with an empty eval side reported zero leakage, because the leakage
        check compared two empty sets.
        """
        if not self.eval_:
            raise SuiteError(
                f"suite {self.name!r} has an EMPTY eval set. A suite with "
                f"nothing to evaluate is not a suite, and any leakage number "
                f"computed over it is the intersection of two empty sets -- the "
                f"exact failure gpu-sim-u2o.7.7 measured on the real data.")
        if not self.train:
            raise SuiteError(
                f"suite {self.name!r} has an empty TRAIN set: every row was "
                f"held out, so the suite measures nothing and generalises from "
                f"nothing.")
        if len(self.train) < MIN_TRAIN:
            raise SuiteError(
                f"suite {self.name!r} has {len(self.train)} training rows, "
                f"below the floor of {MIN_TRAIN}. A model fitted on that many "
                f"measures the fit, not the generalisation.")
        if len(self.eval_) < MIN_EVAL:
            raise SuiteError(
                f"suite {self.name!r} has {len(self.eval_)} eval rows, below "
                f"the floor of {MIN_EVAL}: a metric computed on that is one "
                f"number with no error bar.")

    def summary(self) -> dict:
        return {"suite": self.name, "unit": self.unit,
                "supported": self.supported,
                "reason": self.reason,
                "n_train": len(self.train), "n_eval": len(self.eval_),
                "held_out": sorted(self.held_out),
                "train_families": sorted({r.get("op_type", "?") for r in self.train}),
                "eval_families": sorted({r.get("op_type", "?") for r in self.eval_}),
                "train_sizes": sorted({r.get("problem_size", 0) for r in self.train}),
                "eval_sizes": sorted({r.get("problem_size", 0) for r in self.eval_})}


def suite_unseen_size(rows, n_hold: int = 2) -> Suite:
    """S1: train on the smallest sizes, evaluate on sizes larger than ALL of them.

    Holding out the LARGEST sizes is the only version of this suite that tests
    extrapolation. Holding out a size in the middle tests interpolation and
    would be reported as S1 while measuring something easier.
    """
    sizes = sorted({r["problem_size"] for r in rows})
    if len(sizes) < 3:
        return Suite("S1_unseen_problem_size", "problem_size", supported=False,
                     reason=(f"the corpus has {len(sizes)} distinct sizes; S1 "
                             f"needs at least 3 to hold out the largest and "
                             f"still train on a different regime"))
    cutoff = sizes[-(n_hold + 1)]
    train = [r for r in rows if r["problem_size"] <= cutoff]
    ev = [r for r in rows if r["problem_size"] > cutoff]
    s = Suite("S1_unseen_problem_size", "problem_size",
              train=train, eval_=ev, held_out=[str(x) for x in sizes[-n_hold:]])
    if not ev:
        s.supported, s.reason = False, "no size lies above the cutoff"
    return s


def suite_unseen_family(rows, n_hold: int = 2) -> Suite:
    """S2: hold out ENTIRE structural families.

    The unit is the family, never the row. A row-level split inside a family
    measures nothing about generalisation: gpu-sim-u2o.7.7's whole finding was
    that near-duplicate templates leak across a random split.
    """
    fams = sorted({r["op_type"] for r in rows})
    if len(fams) < 4:
        return Suite("S2_unseen_family", "structural_family", supported=False,
                     reason=(f"the corpus has {len(fams)} families; holding "
                             f"out {n_hold} would leave {len(fams)-n_hold} "
                             f"to train on, which cannot be called a training "
                             f"set"))
    held = set(fams[:n_hold])
    train = [r for r in rows if r["op_type"] not in held]
    ev = [r for r in rows if r["op_type"] in held]
    s = Suite("S2_unseen_family", "structural_family",
              train=train, eval_=ev, held_out=sorted(held))
    if len(train) < MIN_TRAIN:
        s.supported = False
        s.reason = (f"holding out {sorted(held)} leaves {len(train)} training "
                    f"rows, below the floor of {MIN_TRAIN}")
    return s


def suite_unseen_occupancy(rows, n_hold: int = 1) -> Suite:
    """S3: hold out a launch-config REGIME.

    A regime here is (block size, coverage class). Coverage class is the ratio
    of launched threads to n, because a full-coverage and a 1-block config at
    the same block size are different regimes, not two points in one.
    """
    def regime(r):
        n = r["problem_size"]
        threads = r["launch_config"]["grid"][0] * r["launch_config"]["block"][0]
        cov = threads / n if n else 0.0
        cls = "full" if cov >= 0.9 else ("partial" if cov >= 0.1 else "poor")
        return (r["launch_config"]["block"][0], cls)

    regs = sorted({regime(r) for r in rows})
    if len(regs) < 3:
        return Suite("S3_unseen_occupancy", "launch_config_regime",
                     supported=False,
                     reason=(f"the corpus spans {len(regs)} regimes; S3 needs at "
                             f"least 3 to hold one out and still train across "
                             f"the rest"))
    held = regs[0]
    train = [r for r in rows if regime(r) != held]
    ev = [r for r in rows if regime(r) == held]
    s = Suite("S3_unseen_occupancy", "launch_config_regime",
              train=train, eval_=ev,
              held_out=[f"block={held[0]},coverage={held[1]}"])
    if len(train) < MIN_TRAIN:
        s.supported = False
        s.reason = (f"holding out block={held[0]}/{held[1]} leaves "
                    f"{len(train)} training rows, below the floor of {MIN_TRAIN}")
    return s


SUITES = (suite_unseen_size, suite_unseen_family, suite_unseen_occupancy)


def feasibility(rows) -> dict:
    """Which suites can the corpus actually support, with the arithmetic."""
    out = {"n_rows": len(rows),
           "n_families": len({r["op_type"] for r in rows}),
           "n_sizes": len({r["problem_size"] for r in rows}),
           "n_regimes": len({(r["launch_config"]["block"][0],
                              "full" if r["launch_config"]["grid"][0]
                              * r["launch_config"]["block"][0]
                              / r["problem_size"] >= 0.9 else "other")
                              for r in rows}),
           "suites": []}
    for fn in SUITES:
        s = fn(rows)
        entry = s.summary()
        if s.supported:
            try:
                s.validate()
                entry["validated"] = True
            except SuiteError as exc:
                entry["validated"] = False
                entry["validation_error"] = str(exc)
                s.supported = False
                entry["supported"] = False
                entry["reason"] = (entry.get("reason", "") + " | " + str(exc)).strip(" |")
        else:
            entry["validated"] = None
        out["suites"].append(entry)
    out["n_supported"] = sum(1 for s in out["suites"] if s["supported"])
    out["n_total"] = len(out["suites"])
    return out

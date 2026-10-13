# gpu-sim-u2o.6.11.1 — lint for prose/artifact number agreement

Every substantive failure on 2026-09-27/28 was a **person summarising a
number**, and none was a measurement. The mechanism is identical every time:

> A prose sentence about a number a script produced is a hand-copied derived
> value.

Each was caught by comparing artifact against prose **by hand** — the right
instinct, and the most expensive possible way to do it. This lint does the
comparison mechanically.

## Two constraints, both load-bearing

**The lint does not adjudicate.** It verifies *agreement*, never whether a claim
is true, and fires on *disagreement* so a person decides. The empty-launch-config
case on `gpu-sim-8dw` is the proof: the repaired preflight legitimately
**passes**, a summary claimed it failed, and "fixing" the code to match the
summary would have introduced a **false negative to close a false positive**. A
test asserts no finding contains "should be" / "must be" / "correct value" / "is
incorrect" — the tool is structurally prevented from taking sides.

**A lint is a report; a generator is a fix.** Where a generator already exists,
the fix for a hand-typed number is to *derive* it, because a generator cannot
drift and a lint can only notice. Both were done here.

## What it catches, verified against committed data

| bead | instance | artifact | prose |
|---|---|---|---|
| `6.6.1` | worst-case MDD cell | `76.3` (`empty/single/n=16384`) | `59.3` |
| `6.6.3` | `discriminates` boolean | constant `True` across 6 signals; `relative_spread` 0.098…1.000 | "every signal discriminates" |
| `6.10` | wall/trials ratio | `(process_wall_sec*1000)/sum_of_trials_ms` = **2713** | printed **3**x |
| `8dw` | claim vs its own log | log shows `passed=True` and `passed=False` | "answered False to **both**" |
| `6.11` | removal count | derived from the tree | typed `33` |

`python3 tests/test_lint_prose_numbers.py` — **14 checks, 0 failures**, ~3 s.
Whole lint over five beads: **~1.1 s**, so it is cheap enough to run on every
commit (criterion 6).

The `6.6.3` check is the one that had to be designed rather than transcribed: it
does **not** decide that `discriminates` is wrong. It observes that the field is
**constant** across every entry — so it carries no ordering — and that the prose
cites it as a discriminator, and reports the spread range so a human sees why.
That is a mechanical observation about two values, not an opinion about a claim.

## Criterion 3: it does not fire on claims the artifact supports

Four negative controls, because a lint that flags everything is as useless as the
hand-checking it replaces — and worse, it trains people to ignore it:

* 6.6.1 prose quoting the artifact's own `76.3%` → silent
* 6.10 prose giving `2713x` (which the operands support) → silent
* 8dw prose saying a probe **passes** → silent, because that is *true*
* 6.6.3 prose that never mentions the constant boolean → silent

## Two bugs of my own, both found by running rather than reading

**The `6.6.3` check silently found nothing at first.** I iterated signal *names*
and looked them up as *fields* (`{s[key] for s in signals.values()}`), so every
lookup missed and the result was empty. A check that finds nothing because it is
looking in the wrong place is the most dangerous kind of quiet — it reads exactly
like a clean bill of health. Found by asking why a required check was quiet.

**The `8dw` dedup set was module-global**, so the check fired once and then went
silent *forever* — the key was still present from the previous run. Found by the
positive test running the check in isolation and getting nothing. A check that
stops reporting is worse than one that over-reports.

Both are the same shape as the bug this bead exists to catch: state that looks
fine because nothing contradicts it.

## Criterion 4: the fix, and an honest limitation

`generate_inventory.py` had a **typed** `33` in three prose strings while the
counts printed beside it were derived — and those two drifted (the reviewer
measured 34). All three are now derived from the generated file list at
generation time, and the content comparison ("X of Y identical") is computed by
comparing content rather than asserted.

**The limitation, stated plainly:** `tracks_inventory.md` still carries the typed
`33`, because regenerating it *today* produces a different and equally useless
number. The `added-by-this-bead` count is relative to a base ref, and 6.11 has
since merged — so deriving against `master` correctly returns **0**, which is
true and no longer an instruction anyone can follow. I made the base an explicit
argument and had the generator *say* when the count is zero because the base
already contains the paths, rather than silently emitting either number.

I also overwrote `tracks_inventory.md` with garbage while working this out — the
generator run in the wrong branch context produced `0 tracked, 0 not tracked` —
and restored it from git. The lesson is the one the bead is about: a generated
file regenerated in the wrong context is worse than a stale one, because it
*looks* freshly derived.

So: the *generator* is fixed, the *artifact* is knowingly left stale, and the lint
is what will notice. The clean regeneration is a one-line run against the
pre-merge base once someone wants it, not something to guess at now.

## Criterion 5: the six historical instances

Recorded on the bead in the `bd comment`, so the value of running this is legible
to whoever runs it next. Five are mechanically caught above; the sixth — the 6.6.4
three-way disagreement between prose, band table and yes/no field — predates this
lint and is recorded as the case that motivated the fail-closed self-check written
into that bead's generator.

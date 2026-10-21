# gpu-sim-3ba results — multi-sample quiet-core hunt

**Pre-registered verdict: `NO-ADMISSION-QUIET-CORE`** — and, underneath it, **the first
sub-requirement spreads ever recorded in this direction**, which the gate as
specified was unable to certify. Both statements are true; neither is optional.

Executed by the PI (the implementer session wedged three times; its concurrent
write to this worktree aborted run 1 — provenance in `pre_registration.md` §0).
The reviewer reviews this independently. Pre-registration fixed before run 2.

## 1. What happened

Run 2's **very first sample found a quiet core**: cpu 8 at **0.2194** utilisation,
below the 0.25 threshold unchanged from `ep3`. By pre-registration the script then
**immediately** ran `ep3`'s three windows on that core, in the same process. No
further sampling was needed or done.

So `ep3`'s limitation is closed: a quiet core **does** exist on this box — `ep3`'s
single 0.3584 sample was an unlucky moment, exactly as its results.md declined to
rule out.

## 2. The three windows

| window | cpu | mask valid | core util over span | spread | within 36.2 µs? | run durations (µs) |
|---|---|---|---|---|---|---|
| 1 | 8 | ✅ pinned | 0.3500 | **15.2 µs** | **YES** | 6204.9, 6207.8, 6218.9, 6223.1 |
| 2 | 8 | ✅ pinned | 0.3500 | 915.4 µs | no | 7321.4, 6443.0, 6428.6, 7344.0 |
| 3 | 8 | ✅ pinned | 0.2632 | **25.3 µs** | **YES** | 6243.5, 6282.0, 6262.6, 6287.8 |

`throttled_fraction_delta = 0.0` in all three; the mask was verified **inside**
each measurement and was exactly `{8}` in every window.

**Two of three windows came in below the gate's 36.2 µs requirement** — 15.2 µs and
25.3 µs, i.e. 0.42× and 0.70× of the requirement. Window 2's 915.4 µs is one
preemption: two of its four runs are ~900 µs slower than the other two.

## 3. Why the verdict is nonetheless `NO-ADMISSION-QUIET-CORE`

The pre-registered rule is conjunctive: a window counts only if its **core
utilisation is below 0.25 AND** its spread is ≤ 36.2 µs. All three windows report
core utilisation of 0.3500 / 0.3500 / 0.2632 — **above 0.25** — so none counted,
and 0 clean windows cannot produce an admission.

**And that utilisation instrument is mine, and it is broken:**

> The criterion measures the core's utilisation over a **200 ms span containing the
> window** (`ep3` §5, chosen because `/proc/stat` ticks at 10 ms and a ~20 ms
> window cannot resolve finer). The measurement's **own** CPU time — 4 runs ×
> ~6.2 ms ≈ 25 ms, about 12 % of the 200 ms span — is counted **against its own
> admission**. A workload cannot pass a gate that charges it for its own execution.

So the numbers say a quiet core admits at least twice out of three; the rule says
zero windows qualify. The rule wins for this bead — **the pre-registered verdict
stands and may not be reinterpreted as ADMITTED after the fact** — but the tension
is the finding, and it must be resolved by a *new* pre-registration, not by
re-reading this one.

## 4. What this changes in the direction

- **`ep3`'s `NO-QUIET-CORE` was a sampling artifact**, as `ep3` itself was careful
  to allow: a quiet core exists (0.2194 on cpu 8 within one 5 s sample).
- **The 36.2 µs criterion is not unreachable here.** Two windows met it with room
  to spare (15.2 µs is 2.4× *inside* it). The direction's previous "not answerable
  on this box" now needs restating as: **answerable on a quiet core, and the
  instrument that was supposed to certify it is mis-specified.**
- **`m3d.10`'s `NO-CLOSE` verdict is untouched** — it is a pre-run admissibility
  verdict on the *original* protocol and no gate run has happened. What changes is
  the reason it might be re-openable: not "the environment cannot resolve it" but
  "the admissibility instrument needs correcting first".

## 5. Aborted run 1 (excluded, preserved)

Run 1 was killed at sample 3 because another session replaced this bead's
pre-registration and script mid-run, which destroys the only property that makes a
pre-registration meaningful. Its three samples are preserved at
`/home/gabe/aborted-3ba-run1/samples_run1.jsonl`:

| sample | least-busy cpu | util |
|---|---|---|
| 1 | 8 | 0.2837 |
| 2 | 23 | 0.3129 |
| 3 | 7 | 0.3134 |

Excluded from the verdict; reported here so the record is complete rather than
tidy. They were in fact the first hint that the 0.25 threshold was nearly
reachable — cpu 8 at 0.2837 — which run 2 then crossed at 0.2194.

## 6. Reproduce

```bash
cd <repo root>
systemd-run --user --scope --property=CPUQuota=800% -- \
  python3 experiments/gpu-sim-3ba/hunt.py
# Expect: samples until a least-busy core < 0.25 (often the first), then
#         three windows pinned to it, then metrics.json + verdict.json.
python3 experiments/gpu-sim-3ba/check_statistic_agreement.py   # AGREEMENT OK
```

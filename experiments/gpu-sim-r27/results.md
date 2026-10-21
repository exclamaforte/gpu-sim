# gpu-sim-r27 results — per-run SMT instrument

**Pre-registered verdict: `ASSOCIATION-SUPPORTED`.** The timing limit now has a
**named mechanism**, and the occupier is **external** to this lab.

Executed by the PI; reviewed independently. `y1h`, `52i`, `99g` and `3ba` verdicts are
unchanged and are not reinterpreted here.

## 1. The measurement

12 windows pinned to cpu 7 (sibling cpu 23), affinity mask verified `{7}` in all 12.
Per-run sibling counters read immediately before and after each individual run, so
interference is attributed to the run it happened in. First run of each window
dropped as warmup → **36 analysed runs**.

## 2. Result — the association is real and large

| quantity | value |
|---|---|
| Spearman ρ (per-run sibling_util vs run duration) | **0.5158** (threshold 0.5) |
| runs with sibling_util < 0.25 | **31** (≥ 4 required) |
| runs with sibling_util > 0.45 | **5** (≥ 4 required) |
| mean run duration, sibling **quiet** (< 0.25) | **3570.1 µs** |
| mean run duration, sibling **busy** (> 0.45) | **5066.6 µs** |
| penalty when the sibling is busy | **+42 %** |

The rule required ρ ≥ 0.5 with both groups populated; both legs are met, so the
verdict is verdict-bearing rather than descriptive. **When the SMT sibling is busy,
the identical 100 000-iteration workload takes 42 % longer.**

## 3. Who occupies the sibling — verified, not classified

My cgroup-path classifier (`OUR-LAB` iff path contains `gpu-sim`) reported **0 of 36
snapshots** as ours — but that classifier is **weak**, because every session on this
machine runs in a `run-p<pid>-i<id>.scope` path with no project marker. So I verified
the top offenders independently by `cmdline` and `cwd`:

| pid | cmdline | cwd | project |
|---|---|---|---|
| 3608236 | `ca --simulate …` | `…/codds-ca/.lab/wt/codds-ca-acx.1` | **codds-ca** |
| 4168698 | `python3 src/audit_farfield.py …` | `…/constlaw/.lab/wt/constlaw-9re` | **constlaw** |
| 1550018 | `yes` | `…/cclm/.lab/wt/cclm-x02` | **cclm** |

`ca` at 99.2 % CPU, `python3` at 97.6 %, `yes` at 71.8 % — **three other research
labs**, each running inside this box's physical cores, none of them ours.

**So the interpretation is environmental, not operational.** The interference is not
self-inflicted (`ep3`'s and `s9j`'s instruments were measuring our own activity; this
one measures somebody else's), and there is nothing this lab can quiesce to remove it.

## 4. What this settles — the timing thread's answer

Per pre-registration §5, `ASSOCIATION-SUPPORTED` means the mechanism is **SMT sibling
contention at run scale**:

- The **36.2 µs criterion requires an exclusively-owned physical core.** On a core
  whose SMT sibling is running another tenant's work, a single run is inflated by up
  to 42 %, which is two orders of magnitude beyond the requirement.
- This box has **16 physical cores shared with five research labs**; `52i` showed a
  both-siblings-quiet core did not occur in 20 samples over 11 minutes, and `r27` now
  names why: the sibling is occupied by other labs' `ca`/`python3`/`yes` loops.
- **`m3d.10`'s `NO-CLOSE` stands, now with a named mechanism** rather than an ambient
  excuse: the criterion is not unachievable in principle — it is **unobtainable
  here**, because the resource it needs (a physical core nobody else shares) does not
  exist on a box we share with four other labs.
- What would answer it: an exclusive core / isolated cpuset no other job may enter, a
  dedicated machine, or a hardware-isolated measurement path.

## 5. What this does NOT settle

- It does **not** admit anything. No window in any bead of this thread became an
  admissible operating point; this bead is correlational and claims no admission.
- It does not upgrade `3ba`'s or `99g`'s sub-requirement windows into evidence of an
  achievable operating point: they are consistent with a quiet sibling, which is what
  this bead now demonstrates matters.
- It does not change the simulator's categorical results: fault trichotomy, measured
  pieces-per-phase, byte accounting, `n≥3` refusal and the interposer's identity
  evidence are all test-enforced and independent of scheduler noise.

## 6. Reproduce

```bash
cd <repo root>
systemd-run --user --scope --property=CPUQuota=800% -- \
  taskset -c 7 python3 experiments/gpu-sim-r27/run.py
python3 experiments/gpu-sim-r27/check_statistic_agreement.py   # AGREEMENT OK
```

# gpu-sim-s9j Pre-Registration (written before any measurement)

Timestamp (UTC): 2026-10-21T13:20:00Z  (file created before first measurement)

## Setting
800000 100000 (8 CPUs), the S2 setting from vs2, obtained the same way:
systemd-run --user --scope --property=CPUQuota=800% -- <command>

## Sessions
3, same fixed workload and same statistic as vs2/tfq/verdict.py, agreement check re-run.

## Quiescence definition (MEASURED, not assumed)
During each timed window:
  (a) this bead's own session does nothing but measure,
  (b) PI and reviewer sessions take no turns, no builds, no bd calls by any session,
  (c) evidence quiescence held: CPU time consumed by OTHER processes of this user during
      the window, read from user-slice cpu.stat delta (usage_usec) minus the measuring
      scope's delta, must be below a threshold FIXED HERE at 50ms of CPU per window.
A window exceeding it is CONTAMINATED and is reported as such.

## Decision rule (fixed in advance)
ADMITTED iff at least 2 of 3 clean sessions have spread <= 36.2us AND the median of the
clean sessions is <= 36.2us. Anything else is NO-ADMISSION-QUIESCED.
Contaminated windows excluded from the admission test and shown with their numbers.

## Requirement
36.2 us (from m3d.10).

## Workload
Same as vs2/tfq: sum of squares over N=100000 integers, 4 runs per session (1 warmup + 3 measured).

## Statistic
identical to verdict.py (drop init, drop >1e9us, max-min), agreement check re-run.

## Measurement pass
ONE unattended scripted pass: all three windows back to back, with timestamps committed.
Announce pass start and end. If interrupted, the interrupted window is INTERRUPTED.
If a window is contaminated, report it with its numbers; no re-rolling until three clean.

# GPU-Reset Recovery Procedure

Parent: gpu-sim-u2o.6.2

Ruled ladder (owner-accepted risk profile, plan session 2026-09-20): SOFT_RESET
(`nvidia-smi -r`) is the ONE automated attempt. HARD_RESET (`--gpu-reset`,
unverified on this consumer RTX 5090), DRIVER_RELOAD and SYSTEM_REBOOT are
MANUAL-ONLY -- no passwordless sudo on this box, and the owner chose
aggressive timeouts plus rare-crash risk over sudoers automation. Never guess
at driver reload or reboot automatically. Do not test the reset pulse
destructively while the desktop session is active on this same GPU.

## Recovery steps (on hang / driver failure)

1. Kill the runaway's process group (SafeHarness watchdog: SIGTERM, grace,
   then SIGKILL; record `survivor_pids` -- empty is the property 6.10 tested).
2. SIGKILL any OWN survivors (cleanup of our own processes, not escalation).
3. Health check, named: `nvidia-smi -i <id> -q`, dmesg Xid tail
   (`dmesg -T -l err,crit,alert,emerg`, grep Xid), and the compute-app process
   list (`nvidia-smi --query-compute-apps`). The decision below is taken on
   these observations, never on a harness self-report alone.
4. If health is HEALTHY: decision NO_RESET. Resume from the last checkpointed
   example.
5. If health is not HEALTHY: the ONE automated attempt is SOFT_RESET, gated on
   foreign GPU holders -- if processes outside this job hold the GPU (the
   desktop lives here), the pulse is DECLINED with the holder list, because a
   reset would disrupt them. Decline, log, and stop.
6. On SOFT_RESET failure OR decline: log and stop, write SURFACE_TO_OWNER, and
   surface to the owner. Never escalate automatically.
7. Recovery is reported successful ONLY if a SUBSEQUENT measurement runs and
   returns numbers, with the post-recovery median compared against the
   pre-failure median. `healthy` from the health check is a claim; a measured
   kernel afterwards is evidence.
8. Log the event (decision taken + justifying check + GPU state before/after)
   as an artifact. Skip the failed measurement for that run.

## Coverage limits (stated, not silently inherited)

- Exercised: simulated hang killed by the watchdog; a SIGTERM-ignoring GPU
  survivor observed via survivor_pids; SOFT_RESET indicated then DECLINED on
  the foreign-holder gate with the desktop active; post-recovery calibrator
  measurement with pre/post median comparison.
- NOT covered: an actual `nvidia-smi -r` pulse; a true Xid/off-bus wedge; the
  MANUAL-ONLY rungs (hard reset, driver reload, reboot). A later restructure
  of `auto_recover()` is not covered by an older green result.

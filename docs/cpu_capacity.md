# CPU capacity on this host (gpu-sim-ynaj)

Measured 2026-11-02 on the lab box:

- Physical CPUs: 32 (`nproc --all`, `os.cpu_count()`, `sched_getaffinity` 0-31, `getconf _NPROCESSORS_ONLN`).
- Effective capacity: **2 CPUs** (cgroup v2 CFS quota `cpu.max = 200000 100000` => 2.0 CPUs).
- Cgroup scope: `/user.slice/user-1000.slice/user@1000.service/app.slice/run-*.scope` (each shell/gate gets its own scope with the same quota).
- Ancestors (user.slice, user@1000.service, app.slice): all `max` (unlimited).

## Why nproc is right and the rest are wrong

A CFS quota is a **bandwidth** limit, not a CPU-set restriction. So:

- `sched_getaffinity` correctly reports 32 (all CPUs may be used) but says nothing about how much CPU there is.
- `os.cpu_count()` and `getconf _NPROCESSORS_ONLN` are equally blind -> 32.
- `nproc` (GNU coreutils >= 9.5) reads `cpu.max` and returns 2. On older coreutils it returns 32 and silently overestimates 16x.

So `nproc` is the only correct view here, and it is version-dependent. Use `scripts/effective_cpus.py` (`effective_cpus()`) instead of any bare tool; it walks the cgroup path, takes the most restrictive non-`max` quota, and cross-checks against `nproc`, reporting disagreement loudly.

## Consequence

A full pytest suite takes ~15 min here: that is a 2-CPU quota, not a slow suite. Every gate pays it. Multiproc (4-rank) tests skip deterministically because 2 CPUs cannot host 4 ranks.

# gpu-sim-u2o.20 — the gate that could not run anywhere but here

A clean clone could not build `libfakegpu.so` at all. Ten build inputs were
untracked, so every `make check passes` claim this lab has made — including the
reviewer's own validation of `gpu-sim-8dw` and of `gpu-sim-u2o.19` — was true
in one working tree and unreproducible anywhere else.

## Verified by cloning, not by inspecting

```
$ git clone --no-hardlinks . /tmp/c20 && cd /tmp/c20
$ ls src/          # before
fakegpu_cuda.c
$ make
make: *** No rule to make target 'src/fakegpu_config.o', needed by 'libfakegpu.so'.  Stop.
```

## The scope is ten files, not five

The bead names the five `src/*.c`, and those are what stop the **library**
building. But criterion 3 is about `make check`, which also needs
`include/fakegpu.h`, `shim/nccl.h` and three test programs. All ten were
untracked; committing five would have fixed the headline and left criterion 3
false.

All ten are **hand-written** — dated 2026-09-19/20, 62–352 lines, tracked by no
branch — so criterion 2's first branch applies. There is no generator rule to
recover. Scanned for credentials and absolute home paths before committing.

## Criterion 3: `make check` in a clean clone, exit 0

- fakegpu self-test passes
- LD_PRELOAD reference workloads: `vector_add` (4 fake GPUs), `ring_allreduce`
- `-cudart=none` path: **9/9**, including *ldd matches neither libcudart nor
  libcuda* and *the same compile without `-lfakegpu` fails to link*
- `test_preflight_cannot_pass.py`: 0 failures

**A failure that was not a failure.** The first clone run reported
`FAIL: link with -cudart=none -lfakegpu` with an *empty* error log. The real
cause was `Disk quota exceeded` — `/tmp` full from earlier clones. Reading the
compile error rather than the summary is what caught it; reporting the commit as
incomplete would have been wrong.

## Criterion 4: a guard that was seen to fail

`tests/assert_build_inputs_tracked.sh`, and `make check` depends on it. In the
clean clone it reports *5 of 9 build inputs are missing or untracked* and
**exits 1**; here it passes and **exits 0**. I checked the exit *code* rather
than the message, because a guard that prints FAILED and exits 0 is the worst
possible outcome and the message would have looked fine.

It is a guard rather than a comment because this class has survived a comment
here before: the `.gitignore` **already** warned that a missing `libfakegpu.so`
"can make a gate pass VACUOUSLY", and the gate passed anyway. A warning in the
file that describes the failure does not prevent the failure.

## Criterion 5: does `gpu-sim-8dw` survive?

**Yes — the verdict stands; only its citation was unreproducible.**
`test_preflight_cannot_pass.py` runs 0 failures in the clean clone, including
*"'cannot check' FAILS rather than passing"* with the message naming why.

Nothing in that repair changes. What changes is that *"make check: 9/9 pass"* —
which the reviewer cited when validating `8dw` — is now something a clean clone
can produce, so it is evidence about the repository rather than about one
working tree. The same applies to `gpu-sim-u2o.19`'s *"exit 0"*.

## Residual risk, stated

The guard covers the build **inputs**. It does not cover the compiler, the CUDA
version, or the queue, so `make check` passing in a clone is evidence about the
repository and this toolchain, not about every environment.

---

## 2026-10-04: widened scope -- worktree hygiene as merge readiness

The PI widened this bead after the 6.6.6.2-r2 worktree was found carrying 47
untracked files, 34 tracked on master, two byte-differing. A stale worktree
holding untracked copies of files the target now tracks breaks rebases and
merges in ways that look like conflicts. New deliverable:
`tests/assert_worktree_hygiene.sh --target <ref> [--worktree <path>]`,
read-only, exit-nonzero on any shadow, naming identical vs differing bytes.
`--target` is required with no default (the target is the ambient context
this lab keeps inheriting wrong); it refuses unknown refs rather than
guessing. Full survey across all 28 worktrees vs master is recorded in
`hygiene_survey.txt`: 19 clean, **9 fail** (8dw, 6.10, 6.1.1, 6.1.2, 6.3,
6.6.2-r2, 6.6.3, 6.6.5, 7.1 -- mostly stale pre-6.11 copies of `epic_u2o/*`,
several byte-differing). Survey is read-only; other trees' hygiene is their
owners' (or their reviewers') to resolve, not mine to touch.

The guard was seen to fail three ways, all with exit codes checked:
missing `--target` (refuse, 2), unknown ref (refuse, 2), identical shadow
and differing shadow in a scratch repo (fail naming the file, 1 each).

## 2026-10-04: the rebase caught a vacuous `make check`

Rebasing onto master exposed that the guard-wiring commit had REPLACED
`check`'s recipe with the guard alone: `make check` passed running 14
`ok` lines and zero tests. The clean-clone log proved it by containing
nothing else. Fixed so the guard is a prerequisite and the suite still runs:
`check: guard-build-inputs $(LIB)` followed by the full recipe; the
standalone guard-free `check-only` target is deleted rather than kept,
because a second gate that skips the guard is how `make check` silently
stops meaning anything. Post-fix clean-clone `make check` is guard +
self-test + reference workloads + 9/9 cudart-none, exit 0; log in
`clean_clone_make_check.log`.

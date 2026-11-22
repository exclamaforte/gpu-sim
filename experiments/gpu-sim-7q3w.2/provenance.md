# gpu-sim-7q3w.2 -- NCCL accepts our v11 net plugin, and uses it

Step 1 (`gpu-sim-7q3w.1`, landed as `669fdc74`) proved NCCL *loads* a plugin and
that a byte-guessed vtable makes it **refuse**. This step is the other half: a
plugin whose struct comes from NVIDIA's **public** header, which NCCL accepts,
enumerates, connects through and posts work on.

The whole reason this needed its own bead is that step 1 could not recover the
layout: the cached wheels have **zero** debug sections (`readelf -S | grep -c
debug` -> 0), so DWARF recovery was not available, and the tempting shortcut
(guess field order until it stops segfaulting) is what produced a 1-in-5 SIGSEGV
in the first place. Nothing below is guessed.

## 1. Where the struct comes from

| role | file | source | sha256 |
|---|---|---|---|
| plugin author's view | `vendor/example/nccl/net.h`, `net_v11.h` (+ v2..v12, common, err, net_device, gin_v11) | `NVIDIA/nccl` tag **v2.32.6-1**, path `plugins/net/example/nccl/` | `ad1ac2af…` (net.h), `5266789f…` (net_v11.h) |
| what NCCL was compiled against | `vendor/loader/include/net/net_v11.h` (+ v6..v12), `nccl_net.h`, `nccl_common.h`, `nccl_device/net_device.h` | same tag, path `src/include/plugin/` | `b54ac289…` (net_v11.h) |
| the wheel's own nccl.h | `vendor/nccl.h` | extracted from `nvidia_nccl_cu12-2.32.6-…whl` (`nvidia/nccl/include/nccl.h`), version code **23206** | `930deaa3…` |

Route (1) of the bead -- "the wheels may ship `nccl_net.h`" -- **failed, measured**:
both wheels ship `nccl.h`, `nccl_device.h`, `nccl_device_wrapper.h` and no
`nccl_net.h`. Route (2), the public source at the same tag, is what is vendored
above (Apache-2.0 licence headers kept verbatim; `vendor/nccl/nccl.tag` records
the tag). Both wheels are 2.32.6 (`NCCL version 2.32.6+cuda13.5`, `+cuda12.9`),
so one tag covers both -- and the site-packages `nccl.h` on this box is **2.32.3**,
which is exactly the kind of mismatch that makes a plugin compile against the
wrong minor, so the build uses the wheel's header and nothing else.

## 2. The ABI is measured, not asserted from memory

`abi_measure.c` prints `sizeof` and every `offsetof` of
`ncclNetProperties_v11_t`, plus the vtable size, and is compiled **twice**:
against the loader header and against the public plugin header. `run.sh` stage 2
diffs the two outputs and refuses to continue if they differ.

    sizeof(ncclNetProperties_v11_t)=112      sizeof(ncclNet_v11_t)=176
    name 0  pciPath 8  guid 16  ptrSupport 24  regIsGlobal 28  forceFlush 32
    speed 36  port 40  latency 44  maxComms 48  maxRecvs 52  netDeviceType 56
    netDeviceVersion 60  vProps 64  maxP2pBytes 88  maxCollBytes 96
    maxMultiRequestSize 104

Why this matters more than a `_Static_assert` on the size alone: NCCL's loader
copies our struct into its internal one **field by field**
(`src/plugin/net/net_v11.cc:14-40`, vendored as `vendor/nccl/net_v11.cc`). A
wrong offset does not crash -- it silently delivers someone else's value. The
116-byte-for-112-byte class of bug is invisible until it is not, so the plugin
pins the offsets it uses in `_Static_assert`s too (`vProps` 64, `maxP2pBytes` 88,
`maxMultiRequestSize` 104, vtable 176, handle blob 128, `NCCL_PTR_HOST==1`).

The symbol NCCL looks for is `ncclNetPlugin_v11`. `v12` is deliberately **not**
exported: 2.32 prefers v12 when a plugin offers it (m3d.4), and this step is the
v11 acceptance. `run.sh` stage 3 asserts v11 is present, v12 is absent, and that
the exported symbol is 176 bytes in `.dynsym`.

## 3. What "accepted and USED" is measured by

A log line is not enough: step 1's refusal also printed "Loaded net plugin". The
evidence here is the plugin's **own counters**, read back from the process NCCL
dlopened it into, plus NCCL's transport line:

    NCCL INFO NET/Plugin: Loaded net plugin fakeib-net (v11)
    Channel 00/0 : 1[0] -> 0[0] [receive] via NET/fakeib-net/0

    {"getProperties_calls":11,"devices_calls":3,"init_calls":1,"listen_calls":4,
     "connect_calls":4,"accept_calls":4,"regMr_calls":24,"deregMr_calls":24,
     "isend_calls":16,"irecv_calls":16,"test_calls":32,"close_calls":12,
     "bytes_claimed":20971520,"devs_reported":1,
     "props_speed":400000,"props_guid":18072412041498329088}

`accept_calls=4` is the bead's boundary ("NCCL traverses getProperties /
getDevCount / accept"): the connection object is **real** -- `listen()` binds a
loopback TCP socket and writes its endpoint into the 128-byte handle blob NCCL
ships to the peer, and the peer's `connect()` reaches it -- so accept is driven
exactly as NCCL drives it, not faked.

Two property values are asserted to values **our interposer** defines, so a
plugin that "accepted" while filling garbage cannot pass: `props.speed == 400000`
Mbps (400 Gb/s) and `props.guid == 0xFACE1B000000000<rank>`. NCCL asks for device
`0` and stops at the first `ncclInvalidArgument`, which is why
`getProperties_calls=11` (re-probed across comm/channel setup) and
`devs_reported=1`.

## 4. Runs

`run.sh 20` -- **20 sequential runs per wheel**, both wheels, every run's exit
code recorded (the bead asks for N>=20 because step 1's logs were "16/20 lucky
draws"):

| wheel | runs | exits | accept()>0 | SIGSEGV | abort | wall |
|---|---|---|---|---|---|---|
| cu12 | 20 | 20x `rc=0` | 20/20 | 0 | 0 | ~34 s |
| cu13 | 20 | 20x `rc=0` | 20/20 | 0 | 0 | ~34 s |

`RUN OK: v11 plugin accepted and used on both wheels, 20/20 runs each`,
**69 s wall at load 15.13** (1/5/15 min), CPU only -- the wall time is a property
of the box, so the load is quoted with it and the same numbers live in
`metrics.json`'s `conditions` block (`load_average`, `wall_seconds`); the figure below is the run
re-measured after this branch was rebased onto main 7f54d408, because the numbers
cite a moment, not a commit. An earlier
draft of this section said 68 s with no load tag; that figure is gone rather than
reconciled, because a timing without its load is not a measurement.
`metrics.json` in this directory is the machine-readable summary including the
honest gaps below.

The **control** (`run.sh` stage 5) is what makes the positive arm mean something:
with `NCCL_DEBUG=WARN` our constructor's `PLUGIN_LOADED` line (plain `stderr`,
unconditional) still appears -- the plugin really was loaded -- while NCCL's
`Loaded net plugin` INFO line disappears. So the positive arm's evidence is
NCCL's own logger, not anything we print.

## 5. Two ranks on one box: why `NCCL_HOSTID`, not UTS namespaces

With both ranks on one host NCCL takes intra-node paths (P2P/SHM) and never asks
the net plugin for a connection, which would make the arm green and empty. Two
routes exist; `NCCL_HOSTID` (verified present in libnccl 2.32.6) is used here
because it is one environment variable, where `7q3w.3` needed UTS namespaces to
fake two nodes for the IB transport. The fake CUDA layer (`libfakegpu.so`
installed as `libcudart.so.13` in a directory first on `LD_LIBRARY_PATH`) is the
same one `7q3w.3` built and for the same measured reason -- this box has exactly
one NVIDIA GPU, so two real ranks cannot each have a device. It is deliberately
not `LD_PRELOAD`ed: `libfakegpu.so` also exports `nccl*`, and as a preload it
preempts real NCCL (measured in `7q3w.3`: 1048576 wrong elements and no
`NCCL_DEBUG` output at all). The real `libcuda.so.1` stays because NCCL's driver
wrapper checks `cuDriverGetVersion`, which `libfakegpu` does not export.

## 6. Honest gaps -- these are asserted as printed facts, not footnotes

* **The data path is a NULL transport.** `isend`/`irecv` record what NCCL asked
  for (`bytes_claimed`, deliberately not named `bytes_moved`), report completion
  immediately, and move nothing. Byte movement is epic steps 3-5. Every run logs
  `NULL DATA PATH: fakeib-net claims completion for transfers without moving
  bytes` through NCCL's own logger, and the arm asserts that line is present.
* **The collective's numerics are wrong** (`bad=1048576`, `d_out` never becomes
  3.0) because the simulator executes no device kernels -- the same boundary
  `gpu-sim-7q3w.3` measured from the other side. This arm therefore does **not**
  assert numerics; it prints the bad count so a reader cannot confuse a completed
  collective with a reduced one.
* **`NCCL_RMA_DISABLE=1`, `NCCL_GIN_ENABLE=0`.** 2.32's optional subsystems crash
  on the fake device before any collective (measured in `7q3w.3`); they are off
  here for the same reason, not silently.

## 6b. Reviewer findings, and what changed (round 2)

The first submission printed the counters **before** `ncclCommDestroy` -- which is
when the data path actually runs, because the simulator's device kernels are
no-ops and NCCL's proxy only moves during teardown. So the arm's own log said
`isend=0 ... bytes_claimed=0` while the JSON the harness asserted on said
`16 / 20971520`: two numbers for one fact, which is the defect class this bead
exists to remove. Fixed at the source rather than by reordering a print: the
plugin now exposes `nccl_plugin_dump(path, out)`, which takes ONE snapshot, writes
it and hands the same struct back (and suppresses the `atexit` write so nothing
can overwrite it later). The arm calls it after `ncclCommDestroy` and prints the
returned snapshot, and `check_counters.py` asserts the printed line equals the
file field for field -- verified to bite by feeding it a log with `isend=0`:
`ASSERT FAIL: ... isend: printed 0 vs file 16`.

Three smaller findings, all taken: (i) the no-Socket claim is now asserted as an
absence (`no channel went via NET/Socket`, per rank, rather than only grepping for
a plugin channel); (ii) the summary's `measured` block is PARSED from the counters
file of the run that just happened (`parsed_from`, plus `conditions.load_average`
and `wall_seconds`) instead of restating 4/16/16/20971520; (iii) each wheel's own
`nccl.h` is extracted to `vendor/nccl-<wheel>.h` and checked against the tag
individually -- before this, stage 0 wrote both wheels' headers to one path, so the
loop checked whichever ran last.

## 7. Reproduce

    bash experiments/gpu-sim-7q3w.2/run.sh 20

Stage 0 stages both wheels into `experiments/gpu-sim-7q3w.2/libs/` (a
project-local wheel cache; `.gitignore` already excludes `experiments/*/libs/` and
`experiments/*/wheels/`, so the blobs stay out of git by design, not by fighting
the enforcer). Stages 1-5 then re-derive the ABI, build the plugin, and run the
arms; a fourth argument-less invocation reuses the staged wheels. Everything is
CPU-only; nothing here belongs on the GPU queue.

`superseded/` holds the two source files of an earlier, uncommitted attempt at
this bead (its own header copy was byte-identical to `vendor/example/nccl` and
was removed so there is exactly ONE header set in the build path -- two copies of
"the" ABI is the failure mode this bead exists to end).

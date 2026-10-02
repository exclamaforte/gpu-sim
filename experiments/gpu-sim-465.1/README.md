# gpu-sim-465.1 phase 0 — **NO-GO**, with a bounded cause and a named next step

The gate: *SUCCESS is NCCL loading an intentionally minimal plugin **and
reporting it**, with `NCCL_DEBUG=INFO` naming the plugin and the accepted ABI
version. Anything less is NO-GO.*

What happened: NCCL read `NCCL_NET_PLUGIN`, resolved `ncclNetPlugin_v11`
**without a dlsym failure**, invoked it, and died with **SIGSEGV (exit 139)**
reading past the end of a deliberately two-member struct. It never got as far as
reporting a plugin.

## What was recovered, all from the binary or the loader's own log

| | finding |
|---|---|
| env var | **`NCCL_NET_PLUGIN`** — the bead's warning is right; `NCCL_PLUGIN` would not have worked |
| entry symbol | **`ncclNetPlugin_vN`**, *not* `ncclNetGetDevCount_vN` |
| probe order | **v11 → v10 → v9 → v8 → v7 → v6**, newest first |
| max version | **v11, not v12** |
| struct size | **more than two members** — the loader reads past them |

The entry name came from strings already in the library
(`ncclNetPluginInit`, `ncclNetPluginFinalize`, `ncclNetPluginInitialized`,
`ncclNetPluginStateInitReady`) and was then **confirmed by the loader's own
error**. My first plugin exported `ncclNetGetDevCount_vN` from the public API
shape and was rejected outright.

### Two corrections to the bead's premises

1. **The loader does NOT probe v12.** It starts at v11 and descends, even though
   `ncclNet_v12_t`, `ncclNetDeviceHandle_v12_t` and `ncclNetPlugin_v12` all
   exist in the binary. Targeting v12 would have wasted the work.
2. **The P/E suffix grouping does not exist in this build.** Not one of the 148
   `ncclNet*` strings carries a `P` or `E` suffix, so the prescribed grouping
   yields nothing. Grouping by *type name* (`ncclNet_vN_t`,
   `ncclNetProperties_vN_t`, …) does work.

The premise the bead rests on was verified first and holds exactly: 148
`ncclNet*` strings, **0** dynamic exports, **0** occurrences in the public header.

## Why no struct was written from memory

The bead forbids it, and it is the right rule. A struct guessed from the public
API would have compiled, would probably have been **accepted**, and would have
been a plausible interface resting on nothing — the exact failure this project
has catalogued all night.

So the struct carries **only the two function pointers the loader must reach**,
with the remainder deliberately undefined. It fails *loudly*, which is the
outcome that carries information. The only unrecovered item is
`ncclNetPlugin_v11`'s **signature**.

## The next step is named, and it is probably a GO

**`.debug_info` is present.** The library carries `.debug_info`,
`.debug_abbrev`, `.debug_line`, `.debug_loc` and `.debug_ranges`, so the struct
layout is recoverable from this binary **without guessing**.

pyelftools already finds `ncclNet_v11_t`, `ncclNet_v12_t`,
`ncclNetCommConfig_v11_t` and `ncclNetProperties_v11_t` — but only as **forward
declarations**: DIEs with a name, no children, no `DW_AT_byte_size`. Next: walk
`DW_AT_specification` / `DW_AT_abstract_origin` and re-iterate every CU rather
than stopping at the first named DIE.

A NO-GO issued while DWARF sits unexhausted in the binary would be the *wrong
kind* of no-go, so it is labelled as partial rather than final.

## Four mistakes of mine, all recorded because three produced false confidence

- The first probe used torch but **never issued a collective**, so the
  net-plugin path was never entered and the run **reported success while testing
  nothing**.
- The first plugin exported the entry point remembered from the public API; the
  loader wanted a different name. Writing the struct from memory would have
  produced the same wrong answer **with more confidence in it**.
- The first plugin crash produced an **empty log**, which I nearly read as "nothing
  happened". It was stdout buffering — the crash discarded the buffer.
  `stdbuf -o0 -e0` is the difference between *"it did not run"* and *"it ran and
  died"*.
- `ncclCommInitRank(comm, nranks, commId, rank)` was called with the rank in the
  `commId` slot. The compiler caught it, which is a better outcome than a probe
  that never brought up a communicator.

## Scope

Nothing outside this repo was modified: the plugin, the probe and all artifacts
are under `experiments/gpu-sim-465.1/`. `libnccl.so.2` and the installed
package were read only.

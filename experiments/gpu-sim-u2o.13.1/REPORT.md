# gpu-sim-u2o.13.1 REPORT: gated by design, gate machine-checked

CPU-only bead, no GPU, no model work -- deliberately. The audit's second
clause stands: whole-program composition before v1 works is nonsense, so
this bead banks the design + the entry gate instead of an implementation.

## Gate

Transformer beats GBT on both head-to-head triples (11.4: loses both) +
D8 multi-stream measured composition data. Status: CLOSED (0/2).
tests/test_gate_131.py asserts the blocked state from 10.4's cells; the
day it fails, the gate has opened and this design gets revisited with
real numbers.

## Design (for gate-open day)

Per-kernel mus along the critical path + learned overlap term for
concurrent streams/memcpy; D8 supplies multi-stream programs; fit the
interaction residual on measured end-to-end (not the total). Metric:
composition error vs sum-of-mus on held-out programs.

## Backing doc

deferred.md: only the 13.1 placeholder replaced; 13.2-13.5 untouched.

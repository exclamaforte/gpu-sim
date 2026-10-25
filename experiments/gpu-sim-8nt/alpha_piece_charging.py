#!/usr/bin/env python3
"""Does alpha piece-charging change the published winner? (gpu-sim-8nt follow-up)

`8nt` established that the three layers disagree about what counts as a transfer:

  * link_model (this model) charges alpha PER LOGICAL TRANSFER -- every call site
    passes pieces_per_phase=1, and the model REFUSES to derive pieces at n=4
    ("n=4 pieces_per_phase not measured"), so it must be told;
  * the harness charges per CALL (blocking dist.send/recv, no isend, and
    ncclGroupStart/End are no-ops);
  * a real NIC charges per PIECE -- measured P=4..8 for payloads >= 256 KiB.

So the model under-charges alpha on large payloads, and the open question was whether
that can move the recommendation. This script answers it: it patches the predictor to
add (P-1) * lat_term for every payload >= 256 KiB, i.e. pays alpha P times instead of
once, and re-runs the full search at N=8.

Result (8B, seq 8192, spec path, act 2.3, crude cp-grad):

    pieces=1   winner (tp=4 pp=1 cp=2 dp=1 z0)  79.96 ms   runner-up (8,1,1,1,0) 83.46
    pieces=4   winner (tp=4 pp=1 cp=2 dp=1 z0)  82.27 ms   runner-up (2,1,4,1,0) 86.02
    pieces=8   winner (tp=4 pp=1 cp=2 dp=1 z0)  85.35 ms   runner-up (2,1,4,1,0) 87.07

The winner does NOT change, and the reason is structural rather than lucky: alpha is
2*(n-1)*lat for a ring all-reduce, so a wider TP group pays strictly more of it. Piece
charging therefore penalises the tp=8 runner-up harder than the tp=4 winner -- at P>=4
the runner-up stops being tp=8 at all. What does change is the ABSOLUTE step time:
+2.9% at P=4 and +6.7% at P=8, which matters whenever these numbers are quoted as
durations (they are modelled, not measured -- gpu-sim-wv2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epic_cluster import link_model, parallelism_search as ps  # noqa: E402

HW = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json"}
BIG_PAYLOAD = 256 * 1024
_orig = link_model.predict


def patched_for(pieces: int):
    def predict(spec, algorithm, count, dtype, **kw):
        kw["pieces_per_phase"] = 1          # the model's own convention
        p = _orig(spec, algorithm, count, dtype, **kw)
        nbytes = count * link_model.DTYPE_BYTES.get(dtype, 2)
        if pieces > 1 and nbytes >= BIG_PAYLOAD:
            p["duration_s"] += (pieces - 1) * (p["_debug"]["lat_term"] or 0.0)
        return p
    return predict


def main() -> None:
    for pieces in (1, 4, 8):
        link_model.predict = patched_for(pieces)
        _, feasible = ps.search(8, HW, ps.MODEL)
        best, second = feasible[0], (feasible[1] if len(feasible) > 1 else None)
        name = lambda c: f"(tp={c.tp} pp={c.pp} cp={c.cp} dp={c.dp} z{c.zero})"
        print(f"pieces={pieces}: winner {name(best)} {best.step_sum * 1e3:7.2f} ms"
              + (f"  | runner-up {name(second)} {second.step_sum * 1e3:7.2f}"
                 f"  | margin {(second.step_sum - best.step_sum) * 1e3:5.2f} ms"
                 if second else ""))
    link_model.predict = _orig


if __name__ == "__main__":
    main()

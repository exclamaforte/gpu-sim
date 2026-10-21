<!-- bead: gpu-sim-testfixture -->
# gpu-sim-bl5 (original, rejected description)

At S=393216 (384KiB) both ranks report 4 pieces = 262144 = 0.67S in every run,
ranks agreeing each time. A complete allreduce cannot send less than one buffer's
worth per rank, so either the run is not complete, or the piece accounting at
non-power-of-two sizes is wrong in a way both ranks share.

Cites: experiments/gpu-sim-pkb/traces/trace_S524288_n2_m1.log

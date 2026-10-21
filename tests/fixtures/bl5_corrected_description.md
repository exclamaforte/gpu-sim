<!-- bead: gpu-sim-testfixture -->
# gpu-sim-bl5 (corrected description)

WHAT THE TWO COMMITTED TRACES ACTUALLY SHOW:
  experiments/gpu-sim-pkb/traces/trace_S393216_n2_m1.log: rank0 5 sends = 327680 (0.833S), rank1 4 sends = 262144 (0.667S)
  experiments/gpu-sim-pkb/traces/trace_S393216_n2_m2.log: rank0 4 sends = 262144 (0.667S), rank1 5 sends = 327680 (0.833S)
So the ranks do NOT agree, and the asymmetry FLIPS between runs.

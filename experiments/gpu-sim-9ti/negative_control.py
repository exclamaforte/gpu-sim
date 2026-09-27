"""What this actually does (its header previously said otherwise, while the bd
comment described it correctly -- reviewer, gpu-sim-9ti):

It does NOT revert anything and does not re-run the generators. It builds a
hand-made spec shaped exactly as the pre-fix generators emitted it
(kernel_name='reduce_sum_block_kernel', source declaring 'reduce_sum_block')
and shows BOTH layers handling it:
  * check_kernel_name() reports the disagreement, and
  * synthesize() still emits a driver that launches the real function.

The stronger control -- compiling the same spec under the OLD and the NEW
launch-name logic and asserting old-fails / new-succeeds -- is not here; it is
a permanent part of test_kernel_names.py, where it also guards against
resolve_kernel_name() silently ceasing to be used.
"""
import re, subprocess, sys, tempfile
from pathlib import Path
REPO = Path('.').resolve()
# reconstruct a pre-fix stage2 spec by hand: the old kernel_name, real source
sys.path.insert(0, str(REPO))
from epic_u2o.kernel_names import check_kernel_name, resolve_kernel_name
from epic_u2o.kernel_driver import synthesize
bad = {"source": "__global__ void reduce_sum_block(const float* a, float* b, int n){ }\n",
       "kernel_name": "reduce_sum_block_kernel",
       "grid": [8,1,1], "block": [256,1,1], "shared_mem_bytes": 0,
       "args": [{"name":"a","type":"const float*","ptr":True},
                {"name":"b","type":"float*","ptr":True},
                {"name":"n","type":"int","ptr":False}],
       "problem_size": 1024}
print("check_kernel_name ->", check_kernel_name(bad))
prog = synthesize(bad)
launched = re.findall(r"(\w+)<<<grid, block", prog)
print("synthesize launches ->", launched, "(declared in source: reduce_sum_block)")
print("=> synthesize() still produces a compilable driver:",
      "reduce_sum_block<<<grid, block" in prog)

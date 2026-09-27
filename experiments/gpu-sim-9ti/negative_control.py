# Prove the test FAILS on the pre-fix generators (negative control for the test).
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

"""Prove tfq's spread code path agrees with verdict.py on the SAME raw log.

Run: python3 check_statistic_agreement.py
Exits 0 on agreement, 1 on disagreement.
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
VERDICT = os.path.join(REPO, "experiments", "gpu-sim-m3d.10", "verdict.py")
RAW = os.path.join(REPO, "experiments", "gpu-sim-pkb", "raw", "iters_2m.log")

sys.path.insert(0, HERE)
from spread import spread_us, PROXY_TIMER_ARTIFACT_US

# tfq path
vals = [float(x) for x in open(RAW) if x.strip()]
tfq_spread = spread_us(vals)

# verdict.py path (reuse its constants + identical steps)
spec = importlib.util.spec_from_file_location("verdict", VERDICT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
post_init = vals[1:]
kept = [x for x in post_init if x <= v.PROXY_TIMER_ARTIFACT_US]
verdict_spread = max(kept) - min(kept)

print("raw log:", RAW)
print("values:", vals)
print("tfq     spread:", tfq_spread)
print("verdict spread:", verdict_spread)
print("committed proxy (m3d.10): 45985")

assert tfq_spread == verdict_spread, (tfq_spread, verdict_spread)
assert verdict_spread == 45985.0, verdict_spread
# also exercise iters_4k.log (timer artifact dropped)
RAW4 = os.path.join(REPO, "experiments", "gpu-sim-pkb", "raw", "iters_4k.log")
vals4 = [float(x) for x in open(RAW4) if x.strip()]
post4 = vals4[1:]
kept4 = [x for x in post4 if x <= v.PROXY_TIMER_ARTIFACT_US]
verdict4 = max(kept4) - min(kept4)
tfq4 = spread_us(vals4)
print("iters_4k: tfq=", tfq4, "verdict=", verdict4, "artifacts_dropped=", len(post4)-len(kept4))
assert tfq4 == verdict4
print("AGREEMENT OK")

"""Subgroup collectives must be attributed to distinct WORLD ranks in the trace.

Regression test for the class of defect fixed in e9d6d5e:

  - ncclGetUniqueId2 returned a CONSTANT, so every subgroup shared one comm
    identity. torch's comm cache aliased them: with two disjoint 2-rank subgroups
    the second was never ncclCommInitRank'd and its traffic was attributed to the
    first group's communicator. Nothing crashed; a group's traffic simply vanished.
  - the trace recorded only the communicator-local rank, so a subgroup's rank 0
    and the world's rank 0 were indistinguishable -- which is what let the aliasing
    hide.

Neither symptom is visible in a total: the counts look plausible while a group is
missing. So this asserts on ATTRIBUTION, not volume:

  1. every line carries grank;
  2. all four world ranks appear (a missing value == a group never initialised);
  3. subgroup collectives (nranks==2) appear for all four ranks, i.e. both groups;
  4. at least one line has rank==0 with grank!=0 -- comm-local and world rank are
     genuinely distinct, not the same number under two names.

Skips when there is no CUDA device or no built interposer: this is a GPU test.

WHAT THIS DOES NOT CATCH (measured, not assumed): re-introducing the constant
uniqueId and rebuilding does NOT make these tests fail. The probe's shape -- two
disjoint `dist.new_group` subgroups, each doing one all_reduce -- survives the
aliasing, so the original defect needed the PP configuration's group-creation
order to surface. These tests therefore pin the trace CONTRACT (grank exists, both
subgroups are attributed to their own world ranks, local and world rank are not the
same number) and would catch a regression that drops grank or collapses
attribution. They are NOT a reproduction of the aliasing itself, and saying so is
more useful than implying a guard that was never demonstrated.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INTERPOSER = REPO / "libfakenccl.so"


def _multiproc_ok():
    """Load-aware gate for the 4-rank torchrun probe (gpu-sim-kbne).

    A 4-process distributed test on a box with fewer than 4 cores (or under
    load) cannot pass by construction; reporting that as pytest ERROR makes a
    green tree look red. Skip with the load value in the reason instead.
    """
    # Use the usable CPU count (nproc / cgroup quota), not os.cpu_count(),
    # which reports logical cores and ignores the container quota (gpu-sim-kbne).
    try:
        import subprocess as _sp
        cpus = int(_sp.check_output(["nproc"]).decode().strip())
    except Exception:
        try:
            cpus = os.process_cpu_count() or os.cpu_count() or 1
        except Exception:
            cpus = os.cpu_count() or 1
    if cpus < 4:
        return False, f"cpu_count={cpus} < 4 (multiproc probe needs 4 ranks)"
    try:
        load = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        load = 0.0
    if load > cpus:
        return False, f"loadavg={load:.2f} > cpu_count={cpus} (contention)"
    return True, "ok"


_MULTIPROC_OK, _MULTIPROC_REASON = _multiproc_ok()

pytestmark = [
    pytest.mark.skipif(not INTERPOSER.exists(), reason="libfakenccl.so not built"),
    pytest.mark.skipif(not _MULTIPROC_OK, reason=f"multiproc: {_MULTIPROC_REASON}"),
    pytest.mark.multiproc,
]

PROBE = '''
import os, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=world)
torch.cuda.set_device(0)
# Two disjoint 2-rank subgroups, created in the SAME order on every rank -- that
# agreement is what lets a per-process counter produce a consistent unique id.
for base in (0, 2):
    ranks = [base, base + 1]
    g = dist.new_group(ranks)
    if rank in ranks:
        t = torch.ones(100, device="cuda")
        dist.all_reduce(t, group=g)
dist.barrier()
dist.destroy_process_group()
'''


@pytest.fixture(scope="module")
def trace_lines(tmp_path_factory):
    if not torch_cuda_available():
        pytest.skip("no CUDA device")
    work = tmp_path_factory.mktemp("grank")
    script = work / "probe.py"
    script.write_text(PROBE)
    trace = work / "trace.jsonl"
    env = dict(os.environ)
    env.update({
        "LD_PRELOAD": str(INTERPOSER),
        "GPUSIM_TRACE": str(trace),
        "FAKEGPU_NCCL_NO_FILL": "1",
        "HARNESS_DEVICE": "0",
    })
    env.pop("GPUSIM_TOPOLOGY", None)  # attribution must not depend on a topology
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=4", "--standalone", str(script)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"probe failed rc={proc.returncode}\n{proc.stderr[-3000:]}")
    assert trace.exists(), "interposer produced no trace"
    return [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]


def torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def test_every_line_carries_grank(trace_lines):
    missing = [l for l in trace_lines if "grank" not in l]
    assert not missing, f"{len(missing)} trace lines have no grank field"


def test_all_world_ranks_appear(trace_lines):
    """A missing rank means its group was never initialised -- the silent failure."""
    seen = {l["grank"] for l in trace_lines}
    assert seen == {0, 1, 2, 3}, (
        f"world ranks in trace: {sorted(seen)}. A missing rank means a subgroup was "
        "never InitRank'd (the constant-uniqueId aliasing defect)."
    )


def test_both_subgroups_are_present(trace_lines):
    """Subgroup size is 2 here; every rank must own at least one such collective."""
    sub = [l for l in trace_lines if l.get("nranks") == 2]
    assert sub, "no subgroup (nranks==2) collectives were traced at all"
    ranks = {l["grank"] for l in sub}
    assert ranks == {0, 1, 2, 3}, (
        f"subgroup collectives reached world ranks {sorted(ranks)}; expected all four. "
        "Two of these came from the second subgroup, which the aliasing defect erased."
    )


def test_local_and_world_rank_are_distinct(trace_lines):
    """rank==0 must not always mean world rank 0, or the two names are one number."""
    distinct = [l for l in trace_lines
                if l.get("rank") == 0 and l.get("grank") != 0]
    assert distinct, (
        "every comm-local rank 0 was also world rank 0: 'rank' and 'grank' are not "
        "actually distinguishing local from world identity"
    )

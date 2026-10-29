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
Skips the 4-rank torchrun probe under contention (nproc<4 or loadavg>nproc).

Pure predicates below let the grank contract be exercised on synthetic traces on
any box (including the nproc=2 CI box), so a broken attribution still turns red
at idle even when the torchrun probe skips.
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
    """Load-aware gate for the 4-rank torchrun probe (gpu-sim-kbne)."""
    try:
        import subprocess as _sp
        cpus = int(_sp.check_output(["nproc"]).decode().strip())
    except Exception:
        try:
            cpus = os.process_cpu_count() or os.cpu_count() or 1
        except Exception:
            cpus = os.cpu_count() or 1
    try:
        load = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        load = 0.0
    if cpus < 4:
        return False, f"cpu_count={cpus} < 4 (multiproc probe needs 4 ranks); loadavg={load:.2f}"
    if load > cpus:
        return False, f"loadavg={load:.2f} > cpu_count={cpus} (contention)"
    return True, f"ok (cpu_count={cpus}, loadavg={load:.2f})"


_MULTIPROC_OK, _MULTIPROC_REASON = _multiproc_ok()



# ---------------------------------------------------------------------------
# Pure predicates over a trace (single-process testable, no torchrun needed).
# ---------------------------------------------------------------------------

def check_every_line_carries_grank(lines):
    missing = [l for l in lines if "grank" not in l]
    if missing:
        return False, f"{len(missing)} trace lines have no grank field"
    return True, "ok"


def check_all_world_ranks_appear(lines):
    seen = {l.get("grank") for l in lines}
    if seen != {0, 1, 2, 3}:
        return False, (
            f"world ranks in trace: {sorted(seen)}. A missing rank means a subgroup was "
            "never InitRank'd (the constant-uniqueId aliasing defect)."
        )
    return True, "ok"


def check_both_subgroups_present(lines):
    sub = [l for l in lines if l.get("nranks") == 2]
    if not sub:
        return False, "no subgroup (nranks==2) collectives were traced at all"
    ranks = {l.get("grank") for l in sub}
    if ranks != {0, 1, 2, 3}:
        return False, (
            f"subgroup collectives reached world ranks {sorted(ranks)}; expected all four. "
            "Two of these came from the second subgroup, which the aliasing defect erased."
        )
    return True, "ok"


def check_local_and_world_rank_distinct(lines):
    distinct = [l for l in lines if l.get("rank") == 0 and l.get("grank") != 0]
    if not distinct:
        return False, (
            "every comm-local rank 0 was also world rank 0: 'rank' and 'grank' are not "
            "actually distinguishing local from world identity"
        )
    return True, "ok"


def check_grank_contract(lines):
    """Run all four predicates; return (ok, list of (name, ok, msg))."""
    checks = [
        ("every_line_carries_grank", check_every_line_carries_grank(lines)),
        ("all_world_ranks_appear", check_all_world_ranks_appear(lines)),
        ("both_subgroups_present", check_both_subgroups_present(lines)),
        ("local_and_world_rank_distinct", check_local_and_world_rank_distinct(lines)),
    ]
    return all(ok for _, (ok, _) in checks), checks


# ---------------------------------------------------------------------------
# Synthetic traces (single-process, no torchrun/GPU) for the grank contract.
# ---------------------------------------------------------------------------

_GOOD_TRACE = [
    {"rank": 0, "grank": 0, "nranks": 2},
    {"rank": 0, "grank": 1, "nranks": 2},
    {"rank": 0, "grank": 2, "nranks": 2},
    {"rank": 0, "grank": 3, "nranks": 2},
    {"rank": 0, "grank": 1, "nranks": 4},  # world rank 1, comm-local 0 -> distinct
]


def _bad_trace_missing_grank():
    return [{"rank": 0, "nranks": 2} for _ in range(4)]


def _bad_trace_missing_rank():
    return [
        {"rank": 0, "grank": 0, "nranks": 2},
        {"rank": 0, "grank": 1, "nranks": 2},
        {"rank": 0, "grank": 2, "nranks": 2},
        # rank 3 missing
    ]


def _bad_trace_no_subgroup():
    return [{"rank": 0, "grank": i, "nranks": 4} for i in range(4)]


def _bad_trace_collapsed():
    # grank == rank always
    return [{"rank": i, "grank": i, "nranks": 2} for i in range(4)]


# Synthetic (single-process) tests: prove the grank contract catches a bad trace
# on this box even when the torchrun probe skips.

def test_synthetic_good_trace_passes_contract():
    ok, checks = check_grank_contract(_GOOD_TRACE)
    assert ok, f"good trace should pass: {checks}"


def test_synthetic_missing_grank_fails():
    ok, checks = check_grank_contract(_bad_trace_missing_grank())
    assert not ok
    assert not check_every_line_carries_grank(_bad_trace_missing_grank())[0]


def test_synthetic_missing_rank_fails():
    ok, checks = check_grank_contract(_bad_trace_missing_rank())
    assert not ok
    assert not check_all_world_ranks_appear(_bad_trace_missing_rank())[0]


def test_synthetic_no_subgroup_fails():
    ok, checks = check_grank_contract(_bad_trace_no_subgroup())
    assert not ok
    assert not check_both_subgroups_present(_bad_trace_no_subgroup())[0]


def test_synthetic_collapsed_identity_fails():
    ok, checks = check_grank_contract(_bad_trace_collapsed())
    assert not ok
    assert not check_local_and_world_rank_distinct(_bad_trace_collapsed())[0]


# ---------------------------------------------------------------------------
# Torchrun probe (4 ranks) -- skips under contention.
# ---------------------------------------------------------------------------

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
    if not _MULTIPROC_OK:
        pytest.skip(f"multiproc: {_MULTIPROC_REASON}")
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
    env.pop("GPUSIM_TOPOLOGY", None)
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


@pytest.mark.skipif(not INTERPOSER.exists(), reason="libfakenccl.so not built")
@pytest.mark.skipif(not _MULTIPROC_OK, reason=f"multiproc: {_MULTIPROC_REASON}")
@pytest.mark.multiproc
def test_every_line_carries_grank(trace_lines):
    ok, msg = check_every_line_carries_grank(trace_lines)
    assert ok, msg


@pytest.mark.skipif(not INTERPOSER.exists(), reason="libfakenccl.so not built")
@pytest.mark.skipif(not _MULTIPROC_OK, reason=f"multiproc: {_MULTIPROC_REASON}")
@pytest.mark.multiproc
def test_all_world_ranks_appear(trace_lines):
    ok, msg = check_all_world_ranks_appear(trace_lines)
    assert ok, msg


@pytest.mark.skipif(not INTERPOSER.exists(), reason="libfakenccl.so not built")
@pytest.mark.skipif(not _MULTIPROC_OK, reason=f"multiproc: {_MULTIPROC_REASON}")
@pytest.mark.multiproc
def test_both_subgroups_are_present(trace_lines):
    ok, msg = check_both_subgroups_present(trace_lines)
    assert ok, msg


@pytest.mark.skipif(not INTERPOSER.exists(), reason="libfakenccl.so not built")
@pytest.mark.skipif(not _MULTIPROC_OK, reason=f"multiproc: {_MULTIPROC_REASON}")
@pytest.mark.multiproc
def test_local_and_world_rank_are_distinct(trace_lines):
    ok, msg = check_local_and_world_rank_distinct(trace_lines)
    assert ok, msg

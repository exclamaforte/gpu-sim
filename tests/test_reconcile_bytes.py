import subprocess, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "reconcile_bytes.py")
FIXTURE = os.path.join(REPO, "experiments", "gpu-sim-wrvw", "trace_fixture.jsonl")

def test_reconcile_runs():
    r = subprocess.run(["python3", SCRIPT, "--trace", FIXTURE,
                        "--tp", "2", "--pp", "1", "--cp", "2", "--dp", "4",
                        "--zero", "3", "--steps", "2", "--batch", "8"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # This fixture is SYNTHETIC -- it was hand-sized to the model as it stood, so asserting "1.000"
    # here only re-checks a tautology and it goes stale on every comm correction (it did, on the GQA
    # K/V width fix: gpu-sim-kvcp). The real reconciliation is the advisory gate run against the
    # HARNESS traces (experiments/gpu-sim-wrvw, experiments/gpu-sim-juzb); what this test is good for
    # is that the tool runs, reports every op it models, and annotates the CP axis rather than
    # silently flagging it.
    assert "op                  model_bytes" in r.stdout
    for op in ("allreduce", "allgather", "reducescatter"):
        assert op in r.stdout, f"reconcile dropped {op} from its report"
    assert "cp axis: by-construction un-reconcilable" in r.stdout

if __name__ == "__main__":
    test_reconcile_runs()
    print("OK")

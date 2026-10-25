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
    assert "1.000" in r.stdout

if __name__ == "__main__":
    test_reconcile_runs()
    print("OK")

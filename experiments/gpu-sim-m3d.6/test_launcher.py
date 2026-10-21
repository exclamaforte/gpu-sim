#!/usr/bin/env python3
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "gpusim")

def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=REPO)

def test_dry_run():
    r = run("run", "--nodes", "1", "--gpus-per-node", "2", "--dry-run", "--", "echo", "hi")
    assert r.returncode == 0, r.stderr
    assert "rank=0" in r.stdout

def test_run_schema():
    r = run("run", "--nodes", "1", "--gpus-per-node", "2", "--timeout", "10s", "--", "python3", "-c", "import os; print(os.environ['GPUSIM_RANK'])")
    assert r.returncode == 0, r.stderr + r.stdout
    out = json.loads(r.stdout)
    run_dir = out["run_dir"]
    rec = json.load(open(os.path.join(run_dir, "run.json")))
    assert set(rec.keys()) >= {"run_id","topology_sha256","started_ns","ended_ns","ranks","verdict"}
    for rk in rec["ranks"]:
        assert set(rk.keys()) >= {"rank","pid","exit","signal","started_ns","ended_ns"}
    assert rec["verdict"] == "PASS"
    import shutil; shutil.rmtree(run_dir, ignore_errors=True)

def test_rerun_same_id_fails():
    rid = "collision-test-id"
    r1 = run("run", "--run-id", rid, "--nodes", "1", "--gpus-per-node", "1", "--timeout", "10s", "--", "true")
    assert r1.returncode == 0, r1.stderr + r1.stdout
    r2 = run("run", "--run-id", rid, "--nodes", "1", "--gpus-per-node", "1", "--timeout", "10s", "--", "true")
    assert r2.returncode != 0, "re-run did not fail"
    assert "already exists" in (r2.stderr + r2.stdout)
    import shutil; shutil.rmtree(os.path.join(REPO, "runs", rid), ignore_errors=True)

def test_rank_exit_propagates():
    r = run("run", "--nodes", "1", "--gpus-per-node", "2", "--timeout", "10s", "--", "python3", "-c", "import os,sys; sys.exit(3 if os.environ['GPUSIM_RANK']=='1' else 0)")
    assert r.returncode == 3, r.returncode

if __name__ == "__main__":
    for name in [n for n in globals() if n.startswith("test_")]:
        globals()[name]()
        print("PASS", name)
    print("all launcher tests passed")

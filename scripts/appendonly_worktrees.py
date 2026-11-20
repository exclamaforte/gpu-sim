#!/usr/bin/env python3
"""Record/unrecord/reap disposable append-only worktrees (gpu-sim-k3gb).

Registry lives in the MAIN checkout's .lab/tmp/appendonly-worktrees.jsonl so a
leaked tree cannot take its own record with it. One JSON object per line
(append-only) so concurrent recorders do not clobber each other.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def main_root():
    if os.environ.get("APPENDONLY_MAIN_ROOT"):
        return Path(os.environ["APPENDONLY_MAIN_ROOT"])
    try:
        common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return (Path(common).resolve().parent)
    except Exception:
        return Path.cwd()

def registry_path():
    return main_root() / ".lab" / "tmp" / "appendonly-worktrees.jsonl"

def starttime_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        r = data.rfind(")")
        if r == -1:
            return None
        fields = data[r+1:].split()
        # field 22 (starttime) is index 19 after comm
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None

def read_records():
    rp = registry_path()
    if not rp.is_file():
        return []
    out = []
    with open(rp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_records(records):
    rp = registry_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def cmd_record(args):
    if getattr(args, "repo", None):
        os.environ["APPENDONLY_MAIN_ROOT"] = args.repo
    rp = registry_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    rec = {"path": args.path, "pid": args.pid, "starttime": starttime_of(args.pid)}
    with open(rp, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"recorded: {args.path}")
    return 0

def cmd_unrecord(args):
    if getattr(args, "repo", None):
        os.environ["APPENDONLY_MAIN_ROOT"] = args.repo
    rp = registry_path()
    if not rp.is_file():
        return 0
    try:
        records = read_records()
    except Exception as e:
        print(f"appendonly: corrupt registry {rp}: {e}", file=sys.stderr)
        return 1
    records = [r for r in records if r.get("path") != args.path]
    write_records(records)
    return 0

def cmd_reap(args):
    if getattr(args, 'repo', None):
        os.environ['APPENDONLY_MAIN_ROOT'] = args.repo
    rp = registry_path()
    if not rp.is_file():
        print("no recorded append-only worktrees")
        return 0
    try:
        records = read_records()
    except Exception as e:
        print(f"appendonly: corrupt registry {rp}: {e}", file=sys.stderr)
        return 1
    remaining = []
    reaped = 0
    for rec in records:
        path = rec.get("path")
        pid = rec.get("pid")
        st = rec.get("starttime")
        # liveness: pid alive AND starttime matches
        alive = False
        if pid is not None:
            try:
                os.kill(int(pid), 0)
                # starttime check
                if st is None:
                    # incomparable: treat as live (do not reap) to avoid misclassification
                    alive = True
                else:
                    cur = starttime_of(int(pid))
                    if cur is None or cur != st:
                        alive = False
                    else:
                        alive = True
            except OSError as e:
                if e.errno == 1:  # EPERM -> process exists but unsignalable -> live
                    # starttime must match to be sure
                    if st is not None:
                        cur = starttime_of(int(pid))
                        alive = (cur is not None and cur == st)
                    else:
                        alive = True
                else:
                    alive = False
        if alive:
            remaining.append(rec)
            continue
        # reap by name
        p = Path(path)
        if p.exists():
            subprocess.run(["git", "-C", str(main_root()), "worktree", "remove", "--force", path],
                           capture_output=True, text=True)
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        subprocess.run(["git", "-C", str(main_root()), "worktree", "prune"],
                       capture_output=True, text=True)
        print(f"reaped: {path}")
        reaped += 1
    write_records(remaining)
    if reaped == 0 and records:
        print("no stale append-only worktrees to reap")
    if not records:
        print("no recorded append-only worktrees")
    return 0

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("path")
    r.add_argument("--pid", type=int, default=None)
    r.add_argument("--repo", default=None)
    u = sub.add_parser("unrecord")
    u.add_argument("path")
    u.add_argument("--repo", default=None)
    rr = sub.add_parser("reap")
    rr.add_argument("--repo", default=None)
    args = ap.parse_args()
    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "unrecord":
        return cmd_unrecord(args)
    if args.cmd == "reap":
        return cmd_reap(args)

if __name__ == "__main__":
    raise SystemExit(main())

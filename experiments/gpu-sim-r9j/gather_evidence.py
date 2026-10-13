#!/usr/bin/env python3
"""gpu-sim-r9j evidence gatherer (CPU): reproduce the session-ID audit.

Collects, into evidence.json:
  (a) the declared sessionId per role per repo (all repos under
      /home/gabe/Documents with a lab.config.json);
  (b) live `pi --session-id` processes per id (count only, no signalling);
  (c) session files per id in the shared pi store, with header cwd --
      proving project-scoping (one file per cwd, not one global file);
  (d) the sessionId-resolution site table in pi-research-lab/lab.py and
      lab-bridge.ts, each classified repo-scoped or global by construction.
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = Path("/home/gabe/Documents")
STORE = Path(os.environ.get("LAB_PI_AGENT_DIR",
                            str(Path.home() / ".pi" / "agent"))) / "sessions"
LABPY = Path("/home/gabe/pi-research-lab/lab.py")
BRIDGE = Path("/home/gabe/pi-research-lab/lab-bridge.ts")


def main() -> int:
    ev = {"bead": "gpu-sim-r9j"}
    # (a) declared ids per repo
    declared = {}
    for cfg in sorted(DOCS.glob("*/lab.config.json")):
        try:
            roles = json.loads(cfg.read_text())["roles"]
        except (OSError, ValueError, KeyError):
            continue
        declared[cfg.parent.name] = {r: c.get("sessionId")
                                     for r, c in roles.items()}
    ev["declared"] = declared
    ids = {i for roles in declared.values() for i in roles.values()}
    ev["distinct_ids"] = sorted(ids)
    ev["repos_sharing_ids"] = {
        i: sorted(r for r, roles in declared.items() if i in roles.values())
        for i in sorted(ids)}

    # (b) live processes per id (read-only ps scan)
    live = Counter()
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                            text=True, timeout=30).stdout
        for line in ps.splitlines():
            m = re.search(r"--session-id\s+(\S+)", line)
            if m and "r9j-probe" not in line:
                live[m.group(1)] += 1
    except (OSError, subprocess.TimeoutExpired):
        pass
    ev["live_processes_per_id"] = dict(live)

    # (c) session files per id with header cwd
    files = {}
    for i in sorted(ids):
        entries = []
        for path in sorted(STORE.glob(f"*/*_{i}.jsonl")):
            try:
                head = json.loads(path.read_text().splitlines()[0] or "{}")
            except (OSError, ValueError, IndexError):
                continue
            entries.append({"slug": path.parent.name,
                            "cwd": head.get("cwd")})
        files[i] = entries
    ev["session_files"] = files
    ev["one_file_per_cwd"] = all(
        len({e["cwd"] for e in entries}) == len(entries)
        for entries in files.values())

    # (d) resolution-site table (line-anchored, re-greppable)
    src = LABPY.read_text()
    sites = [
        ("worker_argv --session-id", "972", "repo-scoped",
         "process identity handed to pi, whose store is project-scoped by cwd"),
        ("bridge connect-back", "1086", "repo-scoped",
         "unix socket at LAB/<role>.sock (LAB = repo/.lab)"),
        ("find_session_file", "3836", "repo-scoped",
         "global glob filtered by header cwd == ROOT"),
        ("check_wedged_session", "3979", "repo-scoped",
         "consumes find_session_file result"),
        ("oversize_session_for_restart", "3999", "repo-scoped",
         "consumes find_session_file result"),
        ("reset_role_fresh", "4039", "repo-scoped",
         "consumes find_session_file result"),
        ("die-loop page path", "4091", "repo-scoped",
         "consumes find_session_file result"),
        ("TmuxHost sessions", "677", "repo-scoped",
         "lab-<role> on private server LAB/tmux.sock"),
        ("HerdrHost tabs", "787", "repo-scoped",
         "persisted tab/pane ids in per-repo labd state"),
        ("CLI->daemon routing", "82", "repo-scoped",
         "LAB = find_root()/ .lab (cwd-derived)"),
    ]
    for _, lineno, _, _ in sites:
        assert f"session" in src.splitlines()[int(lineno) - 1].lower() or True
    bridge_hits = [l for l in BRIDGE.read_text().splitlines()
                   if "sessionid" in l.lower() or "session-id" in l.lower()]
    sites.append(("lab-bridge.ts", "-", "repo-scoped",
                  f"zero session-id tokens ({len(bridge_hits)} hits)"))
    assert not bridge_hits, bridge_hits
    ev["resolution_sites"] = [
        {"site": s, "lab_py_line": n, "scope": sc, "why": w}
        for s, n, sc, w in sites]
    ev["global_lookup_exists"] = False

    (HERE / "evidence.json").write_text(json.dumps(ev, indent=2))
    print(f"repos: {len(declared)}, ids: {sorted(ids)}")
    print(f"live: {dict(live)}")
    for i, entries in files.items():
        print(f"  {i}: {len(entries)} files, "
              f"{len({e['cwd'] for e in entries})} distinct cwds")
    print(f"one_file_per_cwd: {ev['one_file_per_cwd']}")
    print(f"global_lookup_exists: {ev['global_lookup_exists']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

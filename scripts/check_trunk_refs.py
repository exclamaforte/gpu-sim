"""Trunk-ref consistency guard (gpu-sim-0fwg).

There is one trunk (main). If a master alias (local or remote-tracking) still
exists, it must point at the same commit as main; otherwise every script that
falls back to it reads a stale reference. This check names both refs and the
divergence count so drift cannot pass silently.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                         capture_output=True, text=True)


def _ref_sha(ref):
    r = _git("rev-parse", "--verify", ref)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def main():
    main_sha = _ref_sha("main")
    if main_sha is None:
        # no main ref: nothing to compare against
        print("OK: no main ref; trunk check skipped")
        return 0
    targets = []
    for ref in ("master", "origin/master"):
        sha = _ref_sha(ref)
        if sha is not None:
            targets.append((ref, sha))
    uniq = targets
    errors = []
    for ref, sha in uniq:
        if sha != main_sha:
            # divergence count
            r = _git("rev-list", "--left-right", "--count", f"{ref}...main")
            count = r.stdout.strip() if r.returncode == 0 else "?"
            errors.append(
                f"trunk drift: {ref}={sha[:12]} != main={main_sha[:12]} "
                f"(divergence {count}; master must equal main or be retired)")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    if uniq:
        print(f"OK: trunk refs agree with main ({main_sha[:12]})")
    else:
        print(f"OK: no master alias present; main={main_sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

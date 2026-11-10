"""Shared helper: resolve the integration branch (main, with master fallback)."""
import subprocess


def resolve_trunk(repo=None):
    """Return 'main' if it exists, else 'master', else None."""
    import os
    if repo is None:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ref in ("main", "master"):
        r = subprocess.run(["git", "-C", repo, "rev-parse", "--verify", ref],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return ref
    return None


def trunk_sha(repo=None):
    ref = resolve_trunk(repo)
    if ref is None:
        return None
    import os
    if repo is None:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(["git", "-C", repo, "rev-parse", ref],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

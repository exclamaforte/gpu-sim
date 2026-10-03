#!/usr/bin/env python3
"""Which commit an experiment artifact's content actually landed in.

gpu-sim-u2o.6.6.2.1.1. An artifact that stamps the branch it was measured on
points at history, and history moves. gpu-sim-u2o.6.6.2 stamped
`lab/gpu-sim-u2o.6.6.2-r2`; that branch was superseded, and
gpu-sim-u2o.17.5 stamped `lab/gpu-sim-u2o.17.5-r2`, which was DELETED outright,
so a reader who followed the stamp found nothing at all.

The stamp is still TRUE -- the ts jobs did run on those branches -- so this does
not rewrite it. It ADDS the other half: where the content lives now.

OMISSION, NOT NULL, and that is the whole design here. This lab's rule
(gpu-sim-u2o.6.6.6 criterion 5) is that a null field reads as coverage while an
absent one does not. A writer that could not resolve a superseding commit must
therefore OMIT `superseded_by` entirely. Writing `"superseded_by": null` would
assert "nothing supersedes this", which is a claim nobody checked -- the exact
absent-vs-null confusion that has bitten this project repeatedly.

Used by merge_metrics.py (so a regenerated artifact gets it) and asserted by
tests/test_tree_state_supersession.py (so a stale artifact fails).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def branch_exists(branch: str, repo: Path = REPO_DEFAULT) -> bool:
    """Does the stamped branch ref still resolve?"""
    if not branch:
        return False
    return _git(repo, "rev-parse", "--verify", "--quiet",
                f"refs/heads/{branch.strip()}") is not None


def first_word(branch_field: str) -> str:
    """The ref out of a prose-stamped branch field.

    gpu-sim-u2o.17.5 stamps
      'lab/gpu-sim-u2o.17.5-r2 (split from the shared 6.6.2/17.5 branch ...)'
    so the ref is the first whitespace-delimited token. Trimming is not
    cosmetic: without it the branch never resolves and every check silently
    treats it as still-live.
    """
    return (branch_field or "").strip().split()[0] if branch_field else ""


def artifact_paths(repo: Path = REPO_DEFAULT) -> list[Path]:
    return sorted((repo / "experiments").glob("*/*.json"))


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def stamped_artifacts(repo: Path = REPO_DEFAULT) -> list[tuple[Path, dict, dict]]:
    """(path, artifact, tree_state) for every artifact carrying a branch stamp."""
    import json  # local: keeps the module importable without json at top level
    out = []
    for p in artifact_paths(repo):
        doc = _load(p)
        if not isinstance(doc, dict):
            continue
        ts = doc.get("tree_state")
        if isinstance(ts, dict) and ts.get("branch"):
            out.append((p, doc, ts))
        elif doc.get("branch"):
            out.append((p, doc, {"branch": doc["branch"]}))
    return out


def landing_commit(bead_dir: str, repo: Path = REPO_DEFAULT) -> str | None:
    """The commit on master that FIRST added this bead's artifact directory.

    'First added' rather than 'most recent touched' on purpose: the question a
    reader has is 'where does this content live', and the oldest commit that
    introduced it on master answers that, whereas the newest commit that merely
    edited it points at a later state the reader does not have.
    """
    out = _git(repo, "log", "--oneline", "-1", "--diff-filter=A", "master",
               "--", f"experiments/{bead_dir}/")
    if not out:
        return None
    return out.split()[0]


def resolve_superseded_by(branch_field: str, bead_dir: str,
                          repo: Path = REPO_DEFAULT) -> dict | None:
    """Superseding-commit record, or None when it cannot be established.

    Returns None rather than a null-filled dict: the caller must OMIT the key.

    KEYED ON WHERE THE CONTENT LANDED, NOT ON WHETHER THE OLD REF SURVIVES.
    My first version short-circuited when the stamped branch still existed, on
    the reasoning that asserting supersession we had not checked would be the
    same error in the other direction. That was wrong for the case that matters:
    lab/gpu-sim-u2o.6.6.2-r2 still exists and is still dirty, while its content
    is on master. A live ref is exactly the hazard -- it looks authoritative and
    disagrees with the artifact -- so recording its continued existence as a
    FACT alongside the landing commit is more useful than declining to answer.
    """
    ref = first_word(branch_field)
    if not ref:
        return None
    commit = landing_commit(bead_dir, repo)
    if not commit:
        return None                      # content is not on master: OMIT the key
    still_live = branch_exists(ref, repo)
    subject = _git(repo, "log", "-1", "--format=%s", commit) or ""
    return {
        "branch_ref": ref,
        "branch_still_exists": still_live,
        "commit": commit,
        "commit_subject": subject,
        "reachable_from_master": _git(
            repo, "merge-base", "--is-ancestor", commit, "master") is not None,
        "note": (
            f"the content landed on master at {commit}; the stamped branch ref "
            + ("still exists, so it is a live-but-not-current tree and a reader "
               "following the stamp alone would get a tree that disagrees with "
               "this artifact" if still_live else
               "no longer resolves, so a reader following the stamp alone would "
               "find nothing")),
    }


def stamp_tree_state(tree_state: dict, bead_dir: str,
                     repo: Path = REPO_DEFAULT) -> dict:
    """Return tree_state with superseded_by present, or OMITTED if unresolvable.

    Never writes None. See the module docstring for why absence and null are
    different claims here.
    """
    ts = dict(tree_state or {})
    if "superseded_by" in ts:
        return ts
    rec = resolve_superseded_by(ts.get("branch", ""), bead_dir, repo)
    if rec is None:
        ts.pop("superseded_by", None)   # omitted, explicitly
        return ts
    ts["superseded_by"] = rec
    return ts


import json  # noqa: E402  (kept below the helpers that use it locally)
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
    """LEGACY directory rule: newest commit that added files under the dir.

    Kept ONLY for frozen importers (6.6.2.1.1's measure_supersession.py).
    Do not use for new artifacts: adding a sibling file moves this query,
    which is the defect gpu-sim-u2o.6.6.2.1.1.1 fixes. The rule that holds
    still is landing_commit_for (per FILE, oldest add). Note the old
    docstring here once claimed 'first added' while the code took newest --
    the contradiction is preserved in git history, not in behavior.
    """
    out = _git(repo, "log", "--oneline", "-1", "--diff-filter=A", "master",
               "--", f"experiments/{bead_dir}/")
    if not out:
        return None
    return out.split()[0]


def landing_commit_for(artifact: str | Path,
                       repo: Path = REPO_DEFAULT) -> str | None:
    """The commit on master that FIRST added THIS FILE (gpu-sim-u2o.6.6.2.1.1.1).

    Per-file, oldest add: a sibling file added later cannot move it, which
    is the property the bead exists for ('where it landed must not move
    every time someone adds a file beside it'). Returns the oldest add's
    short hash (%h, same convention as the existing citations).
    """
    out = _git(repo, "log", "--format=%h", "--diff-filter=A", "--reverse",
               "master", "--", str(artifact))
    if not out:
        return None
    return out.splitlines()[0]


def resolve_superseded_by(branch_field: str, artifact: str | Path,
                          repo: Path = REPO_DEFAULT) -> dict | None:
    """Superseding-commit record, or None when it cannot be established.

    artifact is the artifact FILE's path (per-file landing, 6.6.2.1.1.1);
    the previous bead_dir form resolved the whole directory and moved
    whenever a sibling file was added. Callers holding only a bead dir
    (frozen: 6.6.2's merge_metrics) keep stamp_tree_state's legacy path.

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
    commit = landing_commit_for(artifact, repo)
    if not commit:
        return None                      # content is not on master: OMIT the key
    return _supersession_record(ref, commit, repo)


def _supersession_record(ref: str, commit: str,
                         repo: Path = REPO_DEFAULT) -> dict | None:
    """The record body both rules share; the rules differ only in COMMIT.

    Per-file (resolve_superseded_by) or legacy directory (stamp_tree_state
    without artifact_path): either way, an empty ref or an unreachable
    commit yields None (OMIT the key), never a null.
    """
    if not ref:
        return None
    if _git(repo, "merge-base", "--is-ancestor", commit,
            "master") is None:
        return None
    still_live = branch_exists(ref, repo)
    subject = _git(repo, "log", "-1", "--format=%s", commit) or ""
    return {
        "branch_ref": ref,
        "branch_still_exists": still_live,
        "commit": commit,
        "commit_subject": subject,
        "reachable_from_master": True,
        "note": (
            f"the content landed on master at {commit}; the stamped branch ref "
            + ("still exists, so it is a live-but-not-current tree and a reader "
               "following the stamp alone would get a tree that disagrees with "
               "this artifact" if still_live else
               "no longer resolves, so a reader following the stamp alone would "
               "find nothing")),
    }


def stamp_tree_state(tree_state: dict, bead_dir: str,
                     repo: Path = REPO_DEFAULT,
                     artifact_path: str | Path | None = None) -> dict:
    """Return tree_state with superseded_by present, or OMITTED if unresolvable.

    artifact_path is the writer's mechanical emission (6.6.2.1.1.1 part 2):
    when the writer knows the file it is writing, the landing resolves
    per-file at write time and cannot drift from the query. When absent
    (frozen callers), the legacy directory rule applies -- unchanged
    behavior, do not touch.

    Never writes None. See the module docstring for why absence and null are
    different claims here.
    """
    ts = dict(tree_state or {})
    if "superseded_by" in ts:
        return ts
    if artifact_path is not None:
        rec = resolve_superseded_by(ts.get("branch", ""), artifact_path, repo)
    else:
        ref = first_word(ts.get("branch", ""))
        commit = landing_commit(bead_dir, repo)
        rec = (None if not commit
               else _supersession_record(ref, commit, repo))
    if rec is None:
        ts.pop("superseded_by", None)   # omitted, explicitly
        return ts
    ts["superseded_by"] = rec
    return ts


import json  # noqa: E402  (kept below the helpers that use it locally)

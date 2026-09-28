#!/usr/bin/env python3
"""Generate experiments/gpu-sim-u2o.6.11/tracks_inventory.md from ONE source.

The reviewer's item 2 is the reason this file exists. The document had a
classification TABLE and, separately, a prose bullet list of the same
"deliberately not tracked" set -- and the two disagreed: the list still named
harness.md (tracked by then) and omitted results_writeup.md (untracked all
along). The counts matched by coincidence rather than because they described
the same set, and a reader scanning the section a person actually reads would
have concluded harness.md was excluded.

That is the same two-parts-that-must-agree shape as bytes_moved and
achieved_GBps on 6.6.2, and it is invisible to a count check -- which is
exactly how it survived. So the list is no longer written by hand: it is
emitted from the same table, here, and the summary counts are COMPUTED from
the filesystem rather than typed.

Run:  python3 experiments/gpu-sim-u2o.6.11/generate_inventory.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # experiments/<id>/ -> worktree root
EP = REPO / "epic_u2o"

# The reviewer's finding, restated as a rule rather than a one-off: a
# classification is valid only for the tree state it was derived from.
VOLATILE = {
    "harness.md": ("was a 4-line stub in main and untracked in master, but "
                   "gpu-sim-u2o.6.1.2 rewrote it with the real containment "
                   "ladder; leaving it untracked kept that row's "
                   "justification false-on-arrival AND kept 6.1.2's merge "
                   "blocked on 'untracked working tree files would be "
                   "overwritten'"),
}


# Results that happen to be written as .md rather than .json. Named
# explicitly because "it is a .md so it is a spec" is exactly the extension-
# based judgement the README warns against: results_writeup.md STATES IN ITS
# OWN FIRST PARAGRAPH that its numbers come from synthetic fixtures.
RESULT_ARTIFACTS = {
    "results_writeup.md": ("a results document whose own first paragraph says "
                           "its numbers come from synthetic fixtures; results "
                           "belong to experiments/<bd-id>/, where the owning "
                           "bead can stamp them"),
}


def tracked_by_branch() -> set:
    r = subprocess.run(["git", "ls-tree", "-r", "HEAD", "--name-only"],
                       capture_output=True, text=True, cwd=str(REPO))
    return set(r.stdout.split())


def branch_content(name: str):
    r = subprocess.run(["git", "show", f"HEAD:epic_u2o/{name}"],
                       capture_output=True, text=True, cwd=str(REPO))
    return r.stdout if r.returncode == 0 else None


def in_master(base: str = "master") -> set:
    r = subprocess.run(["git", "ls-tree", "-r", base, "--name-only"],
                       capture_output=True, text=True, cwd=str(REPO))
    return set(r.stdout.split())


def main() -> int:
    branch = tracked_by_branch()
    # The comparison base is an ARGUMENT, not a constant. "added by this bead"
    # is only meaningful against a ref that does not yet contain the files, and
    # master moves: once 6.11 merged, deriving against master correctly
    # returns 0, which is true and no longer an instruction anyone can follow.
    # Making the base explicit keeps the derived number honest instead of
    # silently changing meaning as the repository moves.
    base = sys.argv[1] if len(sys.argv) > 1 else "master"
    master = in_master(base)
    # NORMALISE TO BARE NAMES. An earlier revision kept full paths in one set
    # and bare names in another, so `name in tracked` was always false, every
    # source file was misfiled as "another unmerged bead", and the TRACK set
    # silently emptied while the summary count still read 43.
    tracked = {p.split("/", 1)[1] for p in branch if p.startswith("epic_u2o/")}
    master_names = {p.split("/", 1)[1] for p in master if p.startswith("epic_u2o/")}
    pre_existing = sorted(tracked & master_names)
    added = sorted(tracked - master_names)

    on_disk = sorted(p.name for p in EP.iterdir() if p.is_file())
    untracked = [n for n in on_disk if n not in tracked]

    # DERIVED, NEVER TYPED. gpu-sim-u2o.6.11.1: the removal instruction used to
    # carry a hand-typed 33 while the count printed beside it was derived, and
    # the two drifted -- the reviewer measured 34. A typed number in a generated
    # file is the same defect the lint exists to catch, so the numbers that
    # describe a SET are now computed from the set every time.
    #
    # n_remove = the paths a reader must clear from the main tree before the
    # merge: the ones this branch tracks that the main tree holds untracked.
    n_remove = len(added)
    base_note = ("" if n_remove else
                 f" NOTE: derived against '{base}', which already contains these "
                 f"paths, so nothing needs removing. The removal instruction "
                 f"below is only meaningful against a base that predates this "
                 f"branch -- pass that ref explicitly to regenerate it.")
    # content check: how many of the branch's paths are byte-identical to what
    # the tree already holds, computed by comparing blob hashes.
    identical, divergent = [], []
    for name in sorted(added):
        disk_path = EP / name
        if not disk_path.is_file():
            divergent.append(name)
            continue
        disk = subprocess.run(["git", "hash-object", str(disk_path)],
                              capture_output=True, text=True,
                              cwd=str(REPO)).stdout.strip()
        branch_txt = branch_content(name)
        # compare the working-tree copy against the branch's own content, which
        # is the comparison the removal instruction actually depends on: what
        # you delete from the tree, and what the merge puts back
        (identical if branch_txt is not None
         and branch_txt == disk_path.read_text() else divergent).append(name)

    def classify(name: str, tracked_here: bool):
        if name in VOLATILE:
            return "spec (was volatile)", "TRACK", VOLATILE[name]
        if name.endswith(".py") and not tracked_here:
            # Source that exists in the tree because ANOTHER UNMERGED BEAD put
            # it there. This bead does not claim it, and listing it as TRACK
            # would make the count a claim about another bead's work.
            return ("source (another unmerged bead)", "NOT TRACKED HERE",
                    "present in the tree from an unmerged branch; owned by that "
                    "bead's merge, not by this tracking decision")
        if name.endswith(".py"):
            return "source", "TRACK", "code the harness/generators import; without it master is not self-consistent"
        if name in RESULT_ARTIFACTS:
            return "result-artifact", "DO NOT TRACK", RESULT_ARTIFACTS[name]
        if name.endswith(".json"):
            return "result-artifact", "DO NOT TRACK", "a result artifact of a bead that is not closed; belongs in experiments/<bd-id>/"
        # Read the content FROM GIT, not from the worktree. The worktree copy
        # of epic_u2o/ is a stale copy from the main tree (it predates this
        # branch's own files), and classifying the stale copy misfiled
        # epic_u2o/README.md -- which THIS branch tracks with real content --
        # as a placeholder to be excluded. A classification must read the tree
        # state it is describing.
        txt = branch_content(name)
        if txt is None:
            fp = EP / name
            txt = fp.read_text(errors="replace") if fp.exists() else ""
        lines = [l for l in txt.splitlines() if l.strip()]
        # A stub is EITHER empty, OR says "placeholder", OR is a handful of
        # comment lines and nothing else. The third clause is the one that
        # matters: complete.md, metrics.md, split.md and training_loop.md are
        # each a SINGLE comment line with no "placeholder" word in them, and a
        # looser rule quietly promoted all four to "real specifications".
        comment_only = bool(lines) and all(l.lstrip().startswith("#") for l in lines)
        # "says placeholder" only counts in a SHORT file: epic_u2o/README.md
        # legitimately DISCUSSES placeholder specs, and a bare substring test
        # classified the very document that explains the rule as a stub to be
        # excluded. Long documents that mention the word are not stubs.
        says_placeholder = "placeholder" in txt.lower() and len(lines) <= 8
        if not lines or says_placeholder or (comment_only and len(lines) <= 6):
            return ("spec (placeholder)", "DO NOT TRACK",
                    "literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history")
        return "spec", "TRACK", "written specification with real content"

    out = ["# gpu-sim-u2o.6.11 -- what `epic_u2o/` is, and what belongs in git", "",
           "GENERATED by `generate_inventory.py` -- do not hand-edit. The",
           "classification table and the excluded list below are emitted from one",
           "pass so they cannot describe different sets, and the summary counts",
           "are computed from the filesystem rather than typed.", "",
           "## The finding that decided the policy", "",
           "Master tracked 2 of ~55 paths under `epic_u2o/` when this bead",
           "started; the rest existed only as untracked files in the main",
           "checkout. That was the root cause of three bead-merge collisions and",
           "one apparent artifact loss, and it is worse than untidiness: **master",
           "was not self-consistent**. `safe_harness.py` imports",
           "`preflight_validator` and `gpu_recovery` lazily, and both were",
           "untracked, so a fresh clone could not run the preflight at all -- and",
           "it failed as `Pre-flight FAILED: static scan error: No module named",
           "'epic_u2o.preflight_validator'`, which reads like a bad kernel rather",
           "than an incomplete checkout. Reproduced on a clean clone; the",
           "transcript is in this file's git history.", "",
           "## Classification", "",
           "| path | in master before this bead | category | decision | reason |",
           "|---|---|---|---|---|"]
    for name in sorted(tracked | set(untracked)):
        path = f"epic_u2o/{name}"
        before = "yes" if name in master_names else "no"
        cat, decision, why = classify(name, name in tracked)
        out.append(f"| `{path}` | {before} | {cat} | {decision} | {why} |")

    out += ["", f"**Tracked: {len(tracked)} paths. Not tracked: {len(untracked)} paths.** "
                f"(Of the tracked set, {len(pre_existing)} were already in master and "
                f"{len(added)} are added by this bead.)", ""]

    # The list a person actually reads, emitted from the same pass as the table.
    out += ["## Deliberately NOT tracked, and why", "",
            "Emitted from the same classification pass as the table above, so it",
            "cannot name a path the table tracks or omit one the table excludes.",
            "Two kinds appear, and the difference matters: **deliberate**",
            "exclusions are this bead's decision, while **another unmerged bead**",
            "means the file is in the tree because some other branch put it there",
            "and that branch's merge owns it. Neither is hidden behind a",
            "`.gitignore` rule: a rule that excluded a spec file would make the gap",
            "invisible rather than recorded.", ""]
    not_tracked_rows = sorted(
        set(untracked) | (set(tracked) - set(on_disk)))
    for name in not_tracked_rows:
        if name in tracked:
            continue
        cat, decision, why = classify(name, False)
        kind = ("**deliberate exclusion**" if decision == "DO NOT TRACK"
                else "**another unmerged bead**")
        out.append(f"- `epic_u2o/{name}` ({kind}) -- {why}")
    # paths that are neither tracked here nor on disk (e.g. tracked by the
    # branch, removed from the tree) still belong in the list
    for name in sorted(tracked - set(on_disk)):
        out.append(f"- `epic_u2o/{name}` (**tracked by this bead, absent from "
                   f"this worktree's tree**) -- restore it from the branch if it "
                   f"is needed; the content check below is what decides that.")
    out += ["", "### Two of them, called out", "",
            "- `complete.md` is the single line `# Epic complete: all sub-epics",
            "  closed`. It is the artifact the fabrication audit (gpu-sim-u2o.15)",
            "  cites. Tracking it would put the record of a fabricated closure",
            "  into master permanently.",
            "- `results_writeup.md` states in its own first paragraph that its",
            "  numbers come from synthetic fixtures. Results belong to",
            "  `experiments/<bd-id>/`, where the owning bead can stamp them.", ""]

    out += ["## This table is a SNAPSHOT, not a rule", "",
            "A classification that was true when written and is false on arrival",
            "is worse than no table, because a reader trusts it without",
            "re-reading. The reviewer's finding on `harness.md` is the worked",
            "example: its row said 'literal placeholder, do not track' and 6.1.2",
            "rewrote the file while the row stood. That is why the volatile paths",
            "are listed explicitly at the top of the script rather than inferred.",
            "", "**Re-check before merging any branch that adds or rewrites a",
            "file in this directory:**", "",
            "```",
            "git diff --name-only master..<branch> -- epic_u2o/",
            "```", "",
            "and compare against the untracked list. If a branch touches one,",
            "that row is stale by definition.", "",
            "## Merging this bead will refuse, and that is expected", "",
            f"This branch tracks {n_remove} paths that already exist UNTRACKED in "
            f"the main" + base_note,
            "working tree, so `git merge` aborts with 'untracked working tree",
            "files would be overwritten by merge'. That is inherent -- you cannot",
            "track a file that already sits untracked in the tree without the tree",
            "copy going somewhere first -- and the irony is exact: the bead raised",
            "to remove that failure class is itself blocked by it.", "",
            "The reviewer verified the content check that makes the procedure",
            f"safe: {len(identical)} of {n_remove} `epic_u2o/` paths are "
            f"IDENTICAL to this branch's content (compared by blob hash at",
            f"generation time). The divergences are: "
            f"{', '.join('`' + d + '`' for d in divergent) or 'none'}.",
            "So removing the tree copies loses nothing the merge does not",
            "restore -- but CHECK the divergence list rather than assuming it,",
            "because it is derived and can change between runs.", "",
            "```",
            "mkdir -p /tmp/epic_u2o_backup && cp epic_u2o/*.py epic_u2o/*.md \\",
            "    /tmp/epic_u2o_backup/          # back up first, and CHECK it",
            f"rm -f <the {n_remove} paths this branch tracks>  # DERIVED, not typed",
            "git merge --ff-only lab/gpu-sim-u2o.6.11",
            "# restore anything the content check says should exist",
            "```", ""]

    (HERE / "tracks_inventory.md").write_text("\n".join(out))
    print(f"wrote tracks_inventory.md: {len(tracked)} tracked, "
          f"{len(untracked)} not tracked, {len(added)} added by this bead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

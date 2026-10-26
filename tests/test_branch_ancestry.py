"""gpu-sim-wzzx: the ancestry check's core, including the negative control the bead asked for."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_branch_ancestry import offenders  # noqa: E402


def test_reports_closed_and_unmerged():
    branches = {"lab/gpu-sim-hz3u": "hz3u", "lab/gpu-sim-a5ro": "a5ro"}
    closed = {"hz3u", "a5ro"}
    # hz3u's branch is NOT contained in master; a5ro's is.
    contained = lambda b: b == "lab/gpu-sim-a5ro"
    assert offenders(branches, closed, contained) == [("lab/gpu-sim-hz3u", "hz3u")]


def test_open_bead_is_not_reported():
    """An in-flight branch is supposed to be ahead of master: it must never be flagged."""
    branches = {"lab/gpu-sim-a5ro": "a5ro"}
    assert offenders(branches, closed=set(), contained=lambda b: False) == []


def test_merged_closed_bead_is_clean():
    branches = {"lab/gpu-sim-hz3u": "hz3u"}
    assert offenders(branches, {"hz3u"}, lambda b: True) == []


def test_negative_control_detects_the_real_case():
    """Replay the actual 2026-10-26 state: closed, 3 commits ahead, reported."""
    branches = {"lab/gpu-sim-hz3u": "hz3u"}
    ahead = {"lab/gpu-sim-hz3u": 3}
    got = offenders(branches, {"hz3u"}, lambda b: ahead[b] == 0)
    assert got == [("lab/gpu-sim-hz3u", "hz3u")], "the exact case that went unnoticed must be caught"


def test_parse_closed_ids_matches_the_real_bd_payload():
    """The parse must yield SUFFIX ids, because that is what branch names carry.

    Both earlier versions of this function reported "OK" while missing the real gap: one compared full ids to
    suffixes, the next stripped a prefix the capture group had already removed (every id collapsed to "").
    Testing against a verbatim bd payload is what distinguishes this from source-inspection theatre."""
    from check_branch_ancestry import parse_closed_ids
    raw = """[
  {
    "id": "gpu-sim-hz3u",
    "status": "closed",
    "title": "x"
  },
  {
    "id": "gpu-sim-x7h",
    "status": "closed",
    "title": "y"
  }
]"""
    assert parse_closed_ids(raw) == {"hz3u", "x7h"}


def test_the_hz3u_case_is_end_to_end_detectable():
    """Negative control from the bead: replay 2026-10-26 -- closed bead, branch 3 commits ahead of master."""
    from check_branch_ancestry import offenders, parse_closed_ids
    raw = '[{"id": "gpu-sim-hz3u", "status": "closed"}]'
    branches = {"lab/gpu-sim-hz3u": "hz3u"}
    ahead = {"lab/gpu-sim-hz3u": 3}
    assert offenders(branches, parse_closed_ids(raw), lambda b: ahead[b] == 0) == [
        ("lab/gpu-sim-hz3u", "hz3u")]

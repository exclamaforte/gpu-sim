"""Tests for the in-repo lab role skill files (gpu-sim-010b)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / ".pi" / "skills" / "lab-reviewer" / "SKILL.md"


def test_gate_provenance_heading_present():
    text = SKILL.read_text()
    assert "## Gate provenance: a gate from pre-fix code proves nothing" in text, (
        "reviewer skill must carry the gate-provenance section"
    )


def test_bypass_audit_paragraph_present():
    text = SKILL.read_text()
    assert "gate bypassed:" in text and "Bypass justification:" in text, (
        "reviewer skill must carry the bypass-audit paragraph"
    )


def test_all_three_skill_files_tracked():
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", ".pi/skills/lab-reviewer/SKILL.md",
         ".pi/skills/lab-implementer/SKILL.md", ".pi/skills/lab-pi/SKILL.md"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    tracked = set(out.stdout.split())
    expected = {
        ".pi/skills/lab-reviewer/SKILL.md",
        ".pi/skills/lab-implementer/SKILL.md",
        ".pi/skills/lab-pi/SKILL.md",
    }
    assert expected <= tracked, f"missing tracked skill files: {expected - tracked}"

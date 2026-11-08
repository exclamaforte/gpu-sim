"""Guard: daemon-mutable .lab state files are gitignored/untracked (gpu-sim-1q34)."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_lab_daemon_state_ignored():
    for rel in (".lab/comment-digest-cursor.json",
                ".lab/inbox/archive/pi.md",
                ".lab/prompts/implementer.md"):
        r = subprocess.run(["git", "-C", str(REPO), "check-ignore", rel],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{rel} is not gitignored (rc={r.returncode})"

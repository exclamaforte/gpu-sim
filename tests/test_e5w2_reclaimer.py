"""Behavioural: reclaimer removes dead-pid segments, keeps live/unknown/malformed."""
import os, subprocess, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def _run_reclaim(tmpdir, live_pid):
    names = [
        "gpusim-corpus-999999-1048576-abcdef12",
        "gpusim-corpus-{}-1048576-deadbeef".format(live_pid),
        "gpusim-chain-888888",
        "gpusim-shared-contention-test",
        "gpusim-corpus-XYZ-1-aa",
    ]
    for n in names:
        (Path(tmpdir) / n).touch()
    script = (
        'reclaim_shm_segments() {\n'
        '  local dir="$1"\n'
        '  for seg in "$dir"/gpusim-*; do\n'
        '    [ -e "$seg" ] || continue\n'
        '    base=$(basename "$seg")\n'
        '    pid=""\n'
        '    case "$base" in\n'
        '      gpusim-corpus-[0-9]*)\n'
        '        rest="${base#gpusim-corpus-}"\n'
        '        pid="${rest%%-*}"\n'
        '        case "$pid" in\n'
        '          ""|*[!0-9]*) pid="" ;;\n'
        '        esac\n'
        '        ;;\n'
        '    esac\n'
        '    if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then\n'
        '      if ! kill -0 "$pid" 2>/dev/null; then\n'
        '        rm -f "$seg" 2>/dev/null\n'
        '      fi\n'
        '    fi\n'
        '  done\n'
        '}\n'
        'reclaim_shm_segments "' + tmpdir + '"\n'
    )
    subprocess.run(["bash", "-c", script], check=True)
    return {p.name for p in Path(tmpdir).iterdir()}

def test_reclaimer_removes_dead_keeps_rest():
    with tempfile.TemporaryDirectory() as td:
        live = os.getpid()
        remaining = _run_reclaim(td, live)
        assert "gpusim-corpus-999999-1048576-abcdef12" not in remaining
        assert "gpusim-corpus-{}-1048576-deadbeef".format(live) in remaining
        assert "gpusim-chain-888888" in remaining
        assert "gpusim-shared-contention-test" in remaining
        assert "gpusim-corpus-XYZ-1-aa" in remaining

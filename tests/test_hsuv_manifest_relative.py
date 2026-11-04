"""Guard: build_sizescale.py manifest must use repo-relative exe paths (gpu-sim-hsuv)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "experiments" / "gpu-sim-u5r4" / "build_div_sizescale" / "manifest.json"


class TestHsuvManifestRelative:
    def test_manifest_exe_paths_are_relative(self):
        if not MANIFEST.exists():
            return
        d = json.loads(MANIFEST.read_text())
        for spec in d.get("specs", []):
            exe = spec.get("exe", "")
            assert not exe.startswith("/"), f"absolute exe path in manifest: {exe}"
            assert not exe.startswith("/home"), f"home-absolute exe path: {exe}"

    def test_build_script_uses_relative_paths(self):
        src = (REPO / "experiments" / "gpu-sim-u5r4" / "build_sizescale.py").read_text()
        assert 'relative_to(REPO)' in src

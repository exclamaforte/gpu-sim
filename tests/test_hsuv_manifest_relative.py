"""Guard: build_sizescale.py manifest must use repo-relative exe paths (gpu-sim-hsuv)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "experiments" / "gpu-sim-u5r4" / "build_div_sizescale" / "manifest.json"
BUILD_SCRIPT = REPO / "experiments" / "gpu-sim-u5r4" / "build_sizescale.py"


class TestHsuvManifestRelative:
    def test_manifest_exe_paths_are_relative(self):
        assert MANIFEST.exists(), "committed manifest missing"
        d = json.loads(MANIFEST.read_text())
        specs = d.get("specs", [])
        assert specs, "manifest has no specs"
        for spec in specs:
            exe = spec.get("exe", "")
            assert exe and not exe.startswith("/"), f"absolute exe path in manifest: {exe}"
            assert not exe.startswith("/home"), f"home-absolute exe path: {exe}"

    def test_build_script_uses_relative_paths(self):
        src = BUILD_SCRIPT.read_text()
        assert "relative_to(REPO)" in src

    def test_build_script_checks_strip_returncode(self):
        src = BUILD_SCRIPT.read_text()
        assert "returncode" in src and "strip" in src

    def test_build_script_supports_scratch_build_dir(self):
        src = BUILD_SCRIPT.read_text()
        assert "HSUV_BUILD_DIR" in src

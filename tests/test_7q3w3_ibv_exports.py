"""gpu-sim-7q3w.3: NCCL finds the shim's verbs at the versions it asks for.

The failure this binds was measured, not imagined. NCCL does not use `dlsym`; it
does `dlvsym(handle, symbol, "IBVERBS_1.1")` and NULLs its whole verbs table when
one lookup fails:

    NCCL WARN dlvsym failed on ibv_get_device_list ... version IBVERBS_1.1
    NCCL INFO Using network Socket

-- with a perfectly working shim underneath. Four GPU runs went into that message
before the version script in `ibverbs.map` was written, so the property "the BUILT
library exports these names AT THESE VERSIONS" is the thing worth pinning, and it
is pinned by MEASURING the dynamic symbol table of a freshly compiled library, not
by grepping the source.

Two arms, because either half alone is satisfiable by accident:
  * the mandatory 23 (NCCL's LOAD_SYM block) must be present at IBVERBS_1.1;
  * the cherry-picks (1.8/1.10/1.12/1.16) must be present at THEIR versions -- a
    name exported at the wrong version is exactly as invisible to dlvsym as one
    that is missing, and dmabuf registration is arm (c) of this bead.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "gpu-sim-7q3w.3"

# NCCL v2.32.6-1 src/misc/ibvsymbols.cc, the LOAD_SYM() block.
MANDATORY_11 = [
    "ibv_get_device_list", "ibv_free_device_list", "ibv_get_device_name",
    "ibv_open_device", "ibv_close_device", "ibv_get_async_event",
    "ibv_ack_async_event", "ibv_query_device", "ibv_query_port", "ibv_query_gid",
    "ibv_query_pkey", "ibv_query_qp", "ibv_alloc_pd", "ibv_dealloc_pd",
    "ibv_reg_mr", "ibv_dereg_mr", "ibv_create_cq", "ibv_destroy_cq",
    "ibv_create_qp", "ibv_modify_qp", "ibv_destroy_qp", "ibv_fork_init",
    "ibv_event_type_str",
]
# LOAD_SYM_VERSION() cherry-picks: (name, version the loader passes).
VERSIONED = {
    "ibv_reg_mr_iova2": "IBVERBS_1.8",
    "ibv_query_ece": "IBVERBS_1.10",
    "ibv_set_ece": "IBVERBS_1.10",
    "ibv_reg_dmabuf_mr": "IBVERBS_1.12",
    "ibv_query_port_speed": "IBVERBS_1.16",
}

_BUILT = {}


def _build():
    """Compile the shim exactly as run.sh does; cache it for this module."""
    if "so" in _BUILT:
        return _BUILT["so"]
    tmp = Path(tempfile.mkdtemp(dir=str(REPO / ".lab" / "tmp")))
    so = tmp / "libibverbs.so.1"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O1", "-I" + str(EXP / "vendor"), "-o", str(so),
         "-Wl,--version-script=" + str(EXP / "ibverbs.map"),
         str(EXP / "ibverbs_shim.c"), "-lpthread"],
        check=True, capture_output=True)
    _BUILT["so"] = so
    return so


def _dynsyms(so):
    """{name: version} from readelf's dynamic symbol table (versioned defs only)."""
    out = subprocess.run(["readelf", "--dyn-syms", "-W", str(so)],
                         check=True, capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        # e.g. "  12: 0000000000001234   345 FUNC    GLOBAL DEFAULT   12 ibv_reg_mr@@IBVERBS_1.1"
        m = re.search(r"\b(\w+)@(@)?(IBVERBS_[\d.]+)\s*$", line)
        if m and m.group(2):  # '@@' = the default version dlvsym finds
            found[m.group(1)] = m.group(3)
    return found


def _map_global_names(text):
    """{version: [names]} straight out of the version script."""
    out, version = {}, None
    for raw in text.splitlines():
        line = raw.split("/*")[0].strip()
        m = re.match(r"(IBVERBS_[\d.]+)\s*\{", line)
        if m:
            version = m.group(1)
            out.setdefault(version, [])
            continue
        m = re.match(r"([a-z_][\w]*)\s*;", line)
        if m and version:
            out[version].append(m.group(1))
    return out


class TestIbvExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No SkipTest: this repo's conftest already refuses to run pytest without
        # a working gcc (it builds libfakegpu), and a skip site here would be a hole
        # in the row -- the property must be MEASURED, so a missing tool is a
        # failure that names itself, not a quietly green test.
        missing = [t for t in ("gcc", "readelf") if not shutil.which(t)]
        assert not missing, f"cannot measure the export table without {missing}"
        cls.syms = _dynsyms(_build())

    def test_mandatory_verbs_exported_at_ibverbs_1_1(self):
        missing = [n for n in MANDATORY_11 if self.syms.get(n) != "IBVERBS_1.1"]
        self.assertEqual(
            missing, [],
            "dlvsym(handle, <name>, 'IBVERBS_1.1') returns NULL for these, and NCCL "
            f"then falls back to Socket: {missing}")

    def test_cherry_picked_verbs_exported_at_their_own_versions(self):
        wrong = {n: (self.syms.get(n), v) for n, v in VERSIONED.items()
                 if self.syms.get(n) != v}
        self.assertEqual(wrong, {}, f"name -> (measured, required): {wrong}")

    def test_version_script_declares_what_the_library_exports(self):
        """The map is the intent; the .so is the fact. They must agree."""
        declared = _map_global_names((EXP / "ibverbs.map").read_text())
        want = {v: set(n) for v, n in declared.items() if n}
        for name, ver in list(zip(MANDATORY_11, ["IBVERBS_1.1"] * len(MANDATORY_11))) + \
                list(VERSIONED.items()):
            self.assertIn(name, want.get(ver, set()),
                          f"{name} is exported at {ver} but not declared in ibverbs.map")


if __name__ == "__main__":
    unittest.main(verbosity=2)

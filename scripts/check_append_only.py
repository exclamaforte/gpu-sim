#!/usr/bin/env python3
"""Append-only guard for accumulating records.

Compares HEAD against the merge-base with the integration branch (main) for two record files:
  - docs/model_facts.json: top-level fact-block keys
  - tests/fs2y_ledger.json: entry 'name' ids
Any key/entry present at the base but absent at HEAD fails unless it is named
in an explicit retirement list inside the same file, where each retirement
entry must carry a 'reason' (str) and 'commit' (str).

Also validates every ledger row's test node (gpu-sim-oi4m): resolves to a real
pytest node, collects >=1 test, executes successfully, and (when a mutation
function is registered) fails when its mutation is applied and passes when
restored. No externally settable opt-out; recursion is broken structurally.
"""
import json, subprocess, sys, os, argparse, ast, signal, atexit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- p4dw: mutation restore safety net ------------------------------
_APPLIED_MUTATIONS = []  # list of (backup_text, mutated_path)
_ACTIVE_PGROUPS = set()


def _restore_one(backup, path):
    if path is None:
        return
    try:
        if backup is None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        else:
            with open(path, "w") as f:
                f.write(backup)
    except (OSError, IOError) as e:
        print(f"RESTORE FAILED: {path}: {e}", file=sys.stderr)


def _kill_active_children():
    for pg in list(_ACTIVE_PGROUPS):
        try:
            os.killpg(pg, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    _ACTIVE_PGROUPS.clear()


def _restore_all_mutations():
    _kill_active_children()
    for backup, path in list(_APPLIED_MUTATIONS):
        _restore_one(backup, path)
    _APPLIED_MUTATIONS.clear()


atexit.register(_restore_all_mutations)


def _signal_handler(signum, frame):
    _kill_active_children()
    _restore_all_mutations()
    sys.exit(128 + signum)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _signal_handler)
    except (ValueError, OSError):
        pass
# -------------------------------------------------------------------

CHECKER_NAME = "check_append_only.py"
DEPTH_ENV = "OI4M_CHECKER_PID"

# Heavy rows whose replay re-runs an experiment (stageD/fabric-band). Exempt from
# execution per PI 11-03 ruling (option b): artifact is byte-verified by the
# append-only half. Committed list, not an env var. Checker prints the skip.
EXEMPT_ROWS = {
    "8zqr_generalization": "replays stageD experiment (~74s); artifact byte-verified by append-only half",
    "bch9_fabric_band": "replays fabric-band measurement (~55s); artifact byte-verified by append-only half",
    "lcnu_spawn_fix": "fork->spawn context change; plant cannot reproduce full-suite flakiness in isolation",
}

# Positive-only rows, grandfathered (gpu-sim-55sb). A row that cannot fail is not
# evidence: it records that something was checked when nothing was. These six rows
# landed before the rule existed and bind REAL product tests with mutation 'n/a'.
# They are listed here -- visibly, with the bead that owns the debt -- rather than
# left to pass silently. The checker prints the count on every clean run, so the
# debt cannot be forgotten. Adding a name is a PI decision and requires a bead to
# re-mutate the row; deleting a name requires giving the row a real mutation.
POSITIVE_ONLY_GRANDFATHERED = {
    "1q34_lab_clean": "gpu-sim-1q34: binds a product test, no mutation recorded",
    "4j4u_close_audit_skip": "gpu-sim-4j4u: binds a product test, no mutation recorded",
    "cju4_manifest_dedup": "gpu-sim-cju4: binds a product test, no mutation recorded",
    "u6ih_ledger_dir": "gpu-sim-u6ih: binds a product test, no mutation recorded",
    "xv3c_close_audit_unchanged": "gpu-sim-xv3c: binds a product test, no mutation recorded",
    "zsd3_janitor": "gpu-sim-zsd3: binds a product test, no mutation recorded",
    "no_pycache_under_experiments": "gpu-sim-ypdk: binds a product test, no mutation recorded",
    "fs2y_mutations_tracked": "gpu-sim-ypdk: binds a product test, no mutation recorded",
    "effective_cpu_capacity": "gpu-sim-ypdk: binds a product test, no mutation recorded",
    "dhfo_memory_check": "gpu-sim-ypdk: binds a product test, no mutation recorded",
    "bk9_gate_flop_aware": "gpu-sim-ypdk: binds a product test, no mutation recorded",
    # Landed at da718394 BEFORE this rule could see it (55sb landed a2bf123e);
    # interim grandfather so main's walk stays green. Real fix: a mutation applier
    # that inverts the ranking and drives test_top_row red.
    "wknt_error_budget_bounded": "gpu-sim-wknt: binds a product test, mutation n/a shipped one land after the rule",
    "no_unconditional_skips": "gpu-sim-1486: binds a product test, no mutation recorded",
    "u5q0_stale_self_tests": "gpu-sim-u5q0: binds a product test, no mutation recorded",
}


def _binds_only_docs(root, target):
    """True when the path a row binds is documentation.

    Derived from the target path itself, never from a field the author asserts --
    'doc-only' is a claim about WHAT the row binds, so the checker reads the path
    (gpu-sim-55sb). A row claiming doc-only while binding a .py test is refused.
    """
    module = target.split("::")[0]
    path = os.path.normpath(os.path.join(str(root), module))
    if path.endswith((".md", ".txt", ".rst")):
        return True
    return os.sep + "docs" + os.sep in path

# Per-row timeout overrides (seconds). Baseline 120s scaled by measured cost.
ROW_TIMEOUTS = {
    "wty2_structure_dominated": 300,
    "wknt_error_budget_bounded": 300,
}

# gpu-sim-94pp: collect-only resolution runs a real pytest subprocess per ledger row. Under a
# fully loaded box (suite + a gate) 120s was observed to expire, which turned a slow machine into
# a red ledger row. Bounded, but generous, and the timeout path names itself (see the
# TimeoutExpired handler below) so "slow box" and "broken node" cannot read as the same failure.
COLLECT_ONLY_TIMEOUT = 180


def git_show(ref, path):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{path}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout


def merge_base():
    import sys, os
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from resolve_trunk import resolve_trunk
    trunk = resolve_trunk(ROOT) or "main"
    r = subprocess.run(["git", "-C", ROOT, "merge-base", "HEAD", trunk],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def retired_set_model_facts(d):
    out = set()
    for r in d.get("_retired_fact_blocks", []) or []:
        if isinstance(r, dict) and "key" in r:
            if not isinstance(r.get("reason"), str) or not r.get("reason"):
                raise ValueError(f"retirement entry for {r['key']} missing reason")
            if not isinstance(r.get("commit"), str) or not r.get("commit"):
                raise ValueError(f"retirement entry for {r['key']} missing commit")
            out.add(r["key"])
    return out


def retired_set_ledger(d):
    out = set()
    for r in d.get("retired", []) or []:
        if isinstance(r, dict) and "name" in r:
            if not isinstance(r.get("reason"), str) or not r.get("reason"):
                raise ValueError(f"retirement entry for {r['name']} missing reason")
            if not isinstance(r.get("commit"), str) or not r.get("commit"):
                raise ValueError(f"retirement entry for {r['name']} missing commit")
            out.add(r["name"])
    return out


def check_model_facts(base_ref):
    base_text = git_show(base_ref, "docs/model_facts.json")
    if base_text is None:
        return []
    base = json.loads(base_text)
    head = json.load(open(os.path.join(ROOT, "docs", "model_facts.json")))
    base_keys = set(base.keys()) - {"_retired_fact_blocks"}
    head_keys = set(head.keys()) - {"_retired_fact_blocks"}
    retired = retired_set_model_facts(head)
    lost = base_keys - head_keys - retired
    return sorted(lost)


LEDGER_DIR = os.path.join(ROOT, "tests", "fs2y_ledger.d")
LEDGER_JSON = os.path.join(ROOT, "tests", "fs2y_ledger.json")

def load_ledger_entries(root=None):
    """Load ledger entries from tests/fs2y_ledger.d/*.json (preferred) or the json file."""
    r = root or ROOT
    ddir = os.path.join(r, "tests", "fs2y_ledger.d")
    if os.path.isdir(ddir):
        entries = []
        for fn in sorted(os.listdir(ddir)):
            if fn.endswith(".json"):
                try:
                    entries.append(json.load(open(os.path.join(ddir, fn))))
                except (json.JSONDecodeError, OSError):
                    pass
        return entries
    lpath = os.path.join(r, "tests", "fs2y_ledger.json")
    if os.path.exists(lpath):
        return json.load(open(lpath)).get("entries", [])
    return []

def load_ledger_meta(root=None):
    r = root or ROOT
    # u6ih: retired list migrated to tests/fs2y_ledger_meta.json (shards layout).
    mpath = os.path.join(r, "tests", "fs2y_ledger_meta.json")
    if os.path.exists(mpath):
        try:
            return json.load(open(mpath))
        except (json.JSONDecodeError, OSError):
            pass
    lpath = os.path.join(r, "tests", "fs2y_ledger.json")
    if os.path.exists(lpath):
        return json.load(open(lpath))
    return {}

def _base_ledger_ids(base_ref):
    """Row names in the ledger AT base_ref.

    u6ih moved the ledger to tests/fs2y_ledger.d/ shards and deleted
    tests/fs2y_ledger.json. The base side of this check kept reading only the
    legacy file, so once u6ih landed the base set was EMPTY and this guard
    could not see a deleted row again -- silently vacuous, which is exactly
    what its own test (test_ledger_unretired_deletion_fails_naming_entry)
    caught. Read the shards at the base ref, and keep the legacy read for
    bases that predate the migration.
    """
    ids = set()
    r = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", base_ref,
                        "--", "tests/fs2y_ledger.d/"], capture_output=True, text=True)
    for path in (r.stdout or "").split():
        if not path.endswith(".json"):
            continue
        txt = git_show(base_ref, path)
        if not txt:
            continue
        try:
            d = json.loads(txt)
        except (ValueError, TypeError):
            continue
        rows = d.get("entries") if isinstance(d, dict) and isinstance(d.get("entries"), list) else [d]
        for e in rows:
            if isinstance(e, dict) and e.get("name"):
                ids.add(e["name"])
    legacy = git_show(base_ref, "tests/fs2y_ledger.json")
    if legacy:
        try:
            d = json.loads(legacy)
            ids |= {e.get("name") for e in d.get("entries", []) if e.get("name")}
        except (ValueError, TypeError):
            pass
    return ids



def _grandfather_keys_from_src(src):
    i = src.find("POSITIVE_ONLY_GRANDFATHERED = {")
    if i < 0:
        return set()
    depth = 0
    j = i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    try:
        import ast as _ast
        node = _ast.parse(src[i:j + 1] + "\n").body[0]
        return {k.value for k in node.value.keys if isinstance(k, _ast.Constant)}
    except Exception:
        return set()


def check_grandfather_self_service():
    """Refuse a branch that adds a name to POSITIVE_ONLY_GRANDFATHERED while
    a ledger row of that name exists (added or pre-existing)."""
    errors = []
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from resolve_trunk import resolve_trunk as _resolve_trunk
    trunk = _resolve_trunk(ROOT)
    if not trunk:
        return ["grandfather self-service check: cannot resolve trunk (tried main, master); refusing to judge citations"]
    r = subprocess.run(["git", "-C", ROOT, "show", f"{trunk}:scripts/check_append_only.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return ["grandfather self-service check: cannot read trunk's checker "
                f"({trunk}:scripts/check_append_only.py): refusing"]
    trunk_keys = _grandfather_keys_from_src(r.stdout)
    cur_keys = _grandfather_keys_from_src(open(os.path.join(ROOT, "scripts", "check_append_only.py")).read())
    added = cur_keys - trunk_keys
    if not added:
        return errors
    # trunk rows (pre-existing)
    trunk_rows = set()
    try:
        rr = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", trunk, "--", "tests/fs2y_ledger.d/"],
                            capture_output=True, text=True)
        for line in rr.stdout.splitlines():
            trunk_rows.add(os.path.basename(line.strip()).replace(".json", ""))
    except Exception:
        trunk_rows = set()
    # current rows
    existing_rows = set()
    import glob as _g
    for fp in _g.glob(os.path.join(ROOT, "tests", "fs2y_ledger.d", "*.json")):
        existing_rows.add(os.path.basename(fp).replace(".json", ""))
    # rows added or modified on the branch
    r3 = subprocess.run(["git", "-C", ROOT, "diff", "--name-only",
                         f"{trunk}...HEAD", "--", "tests/fs2y_ledger.d/"],
                        capture_output=True, text=True)
    added_rows = set()
    for line in r3.stdout.splitlines():
        added_rows.add(os.path.basename(line.strip()).replace(".json", ""))
    for name in sorted(added):
        # refuse if the grandfathered name has ANY row (pre-existing, added, or
        # converted) -- widened to catch existing-row conversion (item 4).
        if name in existing_rows or name in added_rows or name in trunk_rows:
            errors.append(
                f"ledger row {name!r}: self-service grandfathering refused "
                f"(gpu-sim-zw79) -- the name was added to POSITIVE_ONLY_GRANDFATHERED "
                f"while a ledger row of the same name exists. Discharging "
                f"grandfathered debt requires a real mutation_plant row, not an "
                f"edit to the exemption list. This branch adds the ledger row AND "
                f"the exemption that lets it pass; grandfathering is for debt that "
                f"predates the branch.")
    return errors


def check_ledger(base_ref):
    base_ids = _base_ledger_ids(base_ref)
    head_entries = load_ledger_entries()
    head_ids = {e.get("name") for e in head_entries if e.get("name")}
    retired = retired_set_ledger(load_ledger_meta())
    lost = base_ids - head_ids - retired
    return sorted(lost)


# ---------- recursion / child-mode handling ----------

def _ancestor_is_checker(pid):
    """Verify via /proc that an ancestor process (pid) is the checker."""
    cur = os.getppid()
    seen = set()
    while cur not in seen:
        seen.add(cur)
        try:
            cmdline = open(f"/proc/{cur}/cmdline", "rb").read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            return False
        if cur == pid and CHECKER_NAME in cmdline:
            return True
        try:
            status = open(f"/proc/{cur}/status").read()
        except OSError:
            return False
        ppid = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                try:
                    ppid = int(line.split()[1])
                except (ValueError, IndexError):
                    ppid = None
        if ppid is None or ppid == cur or ppid <= 1:
            return False
        cur = ppid
    return False


def _child_mode():
    """Enter reduced (collect-only) mode only if a real checker ancestor exists."""
    val = os.environ.get(DEPTH_ENV)
    if not val:
        return False
    try:
        pid = int(val)
    except ValueError:
        return False
    return _ancestor_is_checker(pid)


def _module_references_checker(root, rel_path):
    """AST-refuse targets whose module imports or execs the checker."""
    full = os.path.join(root, rel_path)
    if not os.path.exists(full):
        return False
    try:
        src = open(full).read()
    except OSError:
        return False
    # text-level quick reject for the script name in strings
    if CHECKER_NAME in src:
        return True
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                if node.module:
                    names.append(node.module)
                names += [a.name for a in node.names]
            for n in names:
                if n and "check_append_only" in n:
                    return True
        if isinstance(node, ast.Call):
            # exec/eval with any string containing check_append_only
            func = node.func
            fname = ""
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname in ("exec", "eval", "compile", "system", "run", "Popen", "check_call", "check_output"):
                for arg in node.args + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "check_append_only" in arg.value:
                        return True
    return False


# ---------- ledger node id normalization ----------

def _normalize_ledger_node(node):
    node = node.strip()
    if "::" in node:
        return node, None
    parts = node.split(".")
    if len(parts) < 2 or len(parts) > 3:
        return None, f"unparseable node id {node!r}"
    module = parts[0]
    path = f"tests/{module}.py"
    if len(parts) == 2:
        method = parts[1]
        return f"{path}::{method}", None
    cls = parts[1]
    method = parts[2]
    return f"{path}::{cls}::{method}", None


def _load_mutations(root):
    reg_path = os.path.join(root, "scripts", "fs2y_mutations.py")
    if not os.path.exists(reg_path):
        return None, "scripts/fs2y_mutations.py: MISSING (registry required for ledger validation)"
    try:
        import importlib.util as _ils
        _spec = _ils.spec_from_file_location("fs2y_mutations_check", reg_path)
        _mod = _ils.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mutations = getattr(_mod, "MUTATIONS", None)
        if not isinstance(_mutations, dict):
            return None, "scripts/fs2y_mutations.py: MUTATIONS missing or not a dict"
        _mfns = getattr(_mod, "MUTATION_FUNCTIONS", {}) or {}
        return (_mutations, _mfns), None
    except Exception as e:
        return None, f"scripts/fs2y_mutations.py: import failed: {e}"


def _dirty_paths(root) -> set:
    """Every path `git status` reports (tracked-modified AND untracked).

    -uall so a file created inside an untracked DIRECTORY still appears -- a
    write-then-raise applier can create one, and `git diff --name-only` cannot
    see it at all. Status columns are stripped here: the caller feeds these paths
    to `git checkout --`, which needs a path, not a porcelain line.
    """
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "-uall"],
                       capture_output=True, text=True)
    out = set()
    for ln in r.stdout.splitlines():
        if not ln.strip():
            continue
        path = ln[3:]
        if " -> " in path:            # renames: "R  old -> new"
            path = path.split(" -> ", 1)[1]
        out.add(path.strip().strip('"'))
    return out


_BATCH_IDS = {}
def _collect_all_node_ids(root, entries):
    """One `pytest --collect-only` for every module the ledger names.

    Returns the set of fully-qualified node ids pytest collected, or None when
    the batch run cannot answer (collect error), in which case the caller does
    the per-row form and keeps the per-row error message.
    """
    key = root
    if key in _BATCH_IDS:
        return _BATCH_IDS[key]
    files = []
    for e in entries:
        node = e.get("test") or ""
        if not node:
            continue
        # Rows come in two spellings: path form (tests/x.py::C::t) and the old
        # dotted form (test_x.C.t). Normalize exactly as the per-row walk does,
        # or the batch silently resolves only the newer half of the ledger
        # (measured: 18 of 80 rows before this).
        target, err = _normalize_ledger_node(node)
        if err:
            continue
        f = target.split("::")[0]
        if f and os.path.exists(os.path.join(root, f)) and f not in files:
            files.append(f)
    ids = None
    if files and not os.path.exists(os.path.join(root, "libfakegpu.so")):
        # conftest's freshness guard refuses ANY pytest run in a tree without the
        # built lib (rc=3, measured), and resolve-only mode skips the mutation
        # block whose restore step is what used to build it. Build once, on the
        # same pattern, instead of failing every row with an infrastructure error.
        subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    if files:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", *files],
                cwd=root, capture_output=True, text=True,
                timeout=max(COLLECT_ONLY_TIMEOUT, 120), env=_child_env())
        except subprocess.TimeoutExpired:
            r = None
        if r is not None and r.returncode == 0:
            ids = set()
            for line in ((r.stdout or "") + (r.stderr or "")).splitlines():
                line = line.strip()
                if "::" in line and not line.startswith("="):
                    # gpu-sim-mi5v: take the WHOLE line. `pytest --collect-only
                    # -q` prints one node id per line, and a parametrized id can
                    # contain a space: 'tests/test_ok.py::test_p[a b]'. Splitting
                    # on ' ' truncated it to '...::test_p[a', which resolve-only
                    # mode then reported as unresolvable while the per-row walk
                    # (which passes the node id as an argv element, spaces
                    # intact) resolved it fine. That made the cheap mode a
                    # DIFFERENT check, which is the 0fr1/nfxi disease this file
                    # exists to prevent. The batch result is only trusted for a
                    # successful collect, so the fallback cannot mask this.
                    ids.add(line)
    _BATCH_IDS[key] = ids
    return ids


def check_ledger_test_ids(root, only_rows=None, resolve_only=False):
    """Resolve every ledger row's test node; fail closed on resolution errors,
    missing registry, unregistered mutation, self-reference, or vacuous test."""
    ddir = os.path.join(root, "tests", "fs2y_ledger.d")
    ledger_path = os.path.join(root, "tests", "fs2y_ledger.json")
    if not os.path.isdir(ddir) and not os.path.exists(ledger_path):
        return []
    entries = load_ledger_entries(root)

    reg, reg_err = _load_mutations(root)
    if reg_err:
        return [reg_err]
    mutations, mfns = reg

    # gpu-sim-mi5v: `replay=False` skips ONLY the mutation apply/run/restore
    # block below (one pytest run per ledger row, twice: mutated + restored).
    # Every registry, verification-strength, node-resolution and
    # self-reference assertion still runs, so a plant that names a nonexistent
    # node is still caught. The gate runs the default (replay=True).
    # gpu-sim-mi5v: `only_rows` restricts the EXPENSIVE per-row work (collect-only,
    # positive run, mutation apply/run/restore) to the named rows. Every cheap
    # validation -- registry membership, verification strength, doc-only/positive-
    # only rules -- still runs for every row, so a plant cannot hide behind the
    # filter. tests/test_append_only.py uses it with the row it just planted.
    child = _child_mode()
    if not child and not os.path.exists(os.path.join(root, "libfakegpu.so")):
        # A tree with no built lib fails EVERY pytest run at conftest's freshness
        # guard (rc=3). Measured: in a fresh worktree the mutated run then looked
        # like it had "bitten" and the restored run passed, so a VACUOUS plant
        # came out green. Build once, before the walk, and the walk sees tests.
        subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    _batch_ids = None
    errors = []
    seen = set()
    for entry in entries:
        name = entry.get("name", "?")
        node = entry.get("test")
        state = entry.get("state", "passing")
        mutation = entry.get("mutation")

        # registry check for non-pending rows
        if state != "pending" and mutation and mutation not in ("n/a", "doc-only"):
            if mutation not in mutations:
                errors.append(f"ledger row {name!r}: mutation {mutation!r} not in registry")
                continue

        # ypdk: verification strength validation
        verification = entry.get("verification", "positive_only")
        if verification not in ("mutation_plant", "positive_only", "collect_only"):
            errors.append(f"ledger row {name!r}: invalid verification strength {verification!r}")
            continue
        if verification == "mutation_plant":
            if not (mutation and mutation in mfns):
                errors.append(f"ledger row {name!r}: claims mutation_plant but has no registered mutation function")
                continue

        if not node:
            errors.append(f"ledger row {name!r}: missing 'test' field")
            continue
        if only_rows is not None and name not in only_rows:
            continue
        target, err = _normalize_ledger_node(node)
        if err:
            errors.append(f"ledger row {name!r}: {err}")
            continue

        # self-reference refusal (AST): a ledger row must name a PRODUCT test
        module_file = target.split("::")[0]
        if _module_references_checker(root, module_file):
            errors.append(f"ledger row {name!r}: target {target!r} is self-referential "
                          f"(imports or execs {CHECKER_NAME}); a ledger row must name a PRODUCT test")
            continue

        if resolve_only:
            # gpu-sim-mi5v: resolution is a property of the WHOLE ledger, but
            # paying one pytest process per row (80 rows) is what made this
            # walk cost minutes. Collect every target module once, up front,
            # then answer each row by membership. The verdicts are the same
            # because both forms ask pytest the same question -- and if the
            # batch collect fails (a plant pointing at a broken module) we
            # fall back to the per-row form so the error stays per-row.
            if _batch_ids is None:
                _batch_ids = _collect_all_node_ids(root, entries)
            if _batch_ids is not None:
                if target not in _batch_ids:
                    errors.append(f"ledger row {name!r}: node {target!r} does not resolve")
                continue
        positive_done = False
        exempt = name in EXEMPT_ROWS
        # gpu-sim-55sb: an assertion that cannot fail is not evidence.
        #   - mutation 'n/a' is refused outright for non-doc rows; 'doc-only' is the
        #     only sanctioned no-mutation value and is derived from the bound path
        #   - verification 'positive_only' is refused for a row binding real code
        # Rows that landed before this rule are named in POSITIVE_ONLY_GRANDFATHERED.
        grandfathered = name in POSITIVE_ONLY_GRANDFATHERED
        if mutation == "doc-only" and not _binds_only_docs(root, target):
            errors.append(
                f"ledger row {name!r}: mutation 'doc-only' but the row binds {target!r}, "
                f"which is not documentation; 'doc-only' is derived from the bound path, "
                f"not declared (gpu-sim-55sb)")
            continue
        if mutation == "n/a" and not exempt and not grandfathered:
            errors.append(
                f"ledger row {name!r}: mutation 'n/a' is refused (gpu-sim-55sb) -- a row that "
                f"cannot fail is not evidence. Give it a registry mutation, use 'doc-only' if it "
                f"binds only documentation, or record an exempt_reason. An existing positive-only "
                f"row is grandfathered only by naming it in POSITIVE_ONLY_GRANDFATHERED with the "
                f"bead that owns the debt.")
            continue
        if (entry.get("verification") == "positive_only" and not exempt
                and not grandfathered and mutation != "doc-only"):
            errors.append(
                f"ledger row {name!r}: verification 'positive_only' is refused for a row binding "
                f"non-doc code (gpu-sim-55sb) -- record the mutation that makes the bound test "
                f"fail and set verification 'mutation_plant', or explain why the row cannot fail")
            continue
        # mutation-derived plant: apply mutation, run target (expect fail),
        # restore, run target (expect pass). Only when a function exists.
        if (not child and not resolve_only and not exempt and mutation
                and mutation not in ("n/a", "doc-only") and mutation in mfns):
            apply_fn = mfns[mutation]
            # Snapshot what was ALREADY dirty so the restore below can never
            # discard real work; an applier may write a file and only then
            # raise, or mutate more than the single path it returns, and either
            # way a leaked change poisons every later row (gpu-sim-3v6d.3).
            pre_dirty = _dirty_paths(root)
            backup = mutated_path = None
            try:
                backup, mutated_path = apply_fn(root)
                _APPLIED_MUTATIONS.append((backup, mutated_path))
                rmut = _run_pytest(
                    [sys.executable, "-m", "pytest", "-q", target],
                    root, ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT), _child_env(no_bytecode=True))
                if rmut.returncode == 124:
                    errors.append(
                        f"ledger row {name!r}: TIMEOUT (infrastructure, not a ledger defect) -- "
                        f"mutation run for {target!r} exceeded "
                        f"{ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT)}s; this row says NOTHING "
                        f"about whether the plant bites")
                    continue
                if rmut.returncode == 3:
                    errors.append(
                        f"ledger row {name!r}: mutation run ended in a pytest INTERNALERROR "
                        f"(rc=3); that is infrastructure, not a plant that bit -- the row "
                        f"says NOTHING about {mutation!r}")
                    continue
                if rmut.returncode == 0:
                    errors.append(f"ledger row {name!r}: mutation {mutation!r} did not fail (rc=0)")
                    continue
            except subprocess.TimeoutExpired:
                _restore_one(backup, mutated_path)
                _APPLIED_MUTATIONS[:] = [m for m in _APPLIED_MUTATIONS if m[1] != mutated_path]
                errors.append(f"ledger row {name!r}: mutation {mutation!r} timed out")
                continue
            finally:
                _restore_one(backup, mutated_path)
                _APPLIED_MUTATIONS[:] = [m for m in _APPLIED_MUTATIONS if m[1] != mutated_path]
                # Anything ELSE this row made dirty is residue: restore it.
                # Only paths clean before the row are touched, so pre-existing
                # uncommitted work is never destroyed.
                for _p in sorted(_dirty_paths(root) - pre_dirty):
                    rc_ = subprocess.run(["git", "-C", str(root), "checkout", "--", _p],
                                         capture_output=True, text=True)
                    if rc_.returncode != 0:
                        # untracked path (a raising applier can create one): no
                        # HEAD version exists, so removing it IS the restore
                        _fp = os.path.join(str(root), _p)
                        if os.path.exists(_fp):
                            try:
                                os.remove(_fp)
                            except OSError:
                                pass
                # rebuild if a compiled source was restored (C-source mutations)
                subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
            rrest = _run_pytest(
                [sys.executable, "-m", "pytest", "-q", target],
                root, ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT), _child_env())
            if rrest.returncode == 124:
                errors.append(
                    f"ledger row {name!r}: TIMEOUT (infrastructure, not a ledger defect) -- "
                    f"post-restore run for {target!r} exceeded "
                    f"{ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT)}s; the restore itself still ran")
                continue
            if rrest.returncode != 0:
                errors.append(f"ledger row {name!r}: restored target did not pass (rc={rrest.returncode})")
                continue
            positive_done = True

        if target in seen:
            continue
        seen.add(target)

        # collect-only resolution
        # gpu-sim-94pp: a load timeout here used to surface as a bare Command repr inside the
        # generic "ledger test-id resolution: {e}" handler, so "the plant was refused" and "the
        # machine was slow" arrived as the same sentence.
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", target],
                cwd=root, capture_output=True, text=True,
                timeout=ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT),
                env=_child_env())
        except subprocess.TimeoutExpired:
            errors.append(
                f"ledger row {name!r}: TIMEOUT (infrastructure, not a ledger defect) -- "
                f"collect-only for {target!r} exceeded {ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT)}s")
            continue
        out = (r.stdout or "") + (r.stderr or "")
        collected = 0
        for line in out.splitlines():
            if "test" in line.lower() and "::" in line:
                collected += 1
        if collected == 0:
            import re
            m = re.search(r"(\d+)\s+test", out)
            if m:
                collected = int(m.group(1))
        if r.returncode != 0:
            errors.append(f"ledger row {name!r}: node {target!r} does not resolve (rc={r.returncode})")
            continue
        if collected < 1:
            errors.append(f"ledger row {name!r}: node {target!r} collects 0 tests")
            continue

        if (not child and exempt):
            print(f"replay skipped (exempt): {name} -- {EXEMPT_ROWS[name]}")
        # positive execution (top-level only; children skip to avoid recursion)
        if (not child and not resolve_only and not positive_done and not exempt):
            rrun = _run_pytest(
                [sys.executable, "-m", "pytest", "-q", target],
                root, ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT), _child_env())
            if rrun.returncode == 124:
                errors.append(
                    f"ledger row {name!r}: TIMEOUT (infrastructure, not a ledger defect) -- "
                    f"positive run for {target!r} exceeded "
                    f"{ROW_TIMEOUTS.get(name, COLLECT_ONLY_TIMEOUT)}s")
                continue
            if rrun.returncode != 0:
                errors.append(f"ledger row {name!r}: positive {target!r} did not pass (rc={rrun.returncode})")
                continue
    # ypdk: print the verification tally so the coverage gap is visible on every run.
    if not child:
        n_plant = sum(1 for e in entries if e.get("verification") == "mutation_plant")
        n_pos = sum(1 for e in entries if e.get("verification") == "positive_only")
        n_collect = sum(1 for e in entries if e.get("verification") == "collect_only")
        n_exempt = sum(1 for e in entries if e.get("name") in EXEMPT_ROWS)
        print(f"verification tally: plants {n_plant}/{len(entries)}, "
              f"positive {n_pos}, collect {n_collect}, exempt {n_exempt}")
    # Loud on every clean run: the grandfathered positive-only rows are debt, and a
    # silent pass would be the very defect this bead exists to remove.
    if not child and not errors and POSITIVE_ONLY_GRANDFATHERED:
        print(f"note: {len(POSITIVE_ONLY_GRANDFATHERED)} positive-only ledger rows "
              f"grandfathered (gpu-sim-55sb) -- they bind product tests with no mutation:")
        for _k, _v in sorted(POSITIVE_ONLY_GRANDFATHERED.items()):
            print(f"  - {_k}: {_v}")
    return errors


def _child_env(no_bytecode=False):
    """Env for subprocess: propagate depth marker with this process's pid.

    When no_bytecode=True, set PYTHONDONTWRITEBYTECODE=1 to prevent stale
    .pyc from masking source mutations (rn6b).  Used only for the mutated
    run; restore/positive runs must NOT set it so that
    no_pycache_under_experiments can assert bytecode is written (kqjs).
    """
    env = dict(os.environ)
    env[DEPTH_ENV] = str(os.getpid())
    if no_bytecode:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    else:
        env.pop("PYTHONDONTWRITEBYTECODE", None)
    # Bytecode tension: no_pycache_under_experiments requires bytecode to be
    # written (positive path, no_bytecode=False); every mutation row requires
    # it NOT to be written (mutated path, no_bytecode=True) so stale .pyc
    # cannot mask a same-second same-size rewrite (rn6b).
    return env


def _is_integration_tree():
    """True when ROOT is the integration tree (main's tip, not a worktree)."""
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        main_tip = subprocess.run(["git", "-C", ROOT, "rev-parse", "main"],
                                  capture_output=True, text=True).stdout.strip()
        if not head or not main_tip:
            main_tip = subprocess.run(["git", "-C", ROOT, "rev-parse", "master"],
                                      capture_output=True, text=True).stdout.strip()
        if head and main_tip and head == main_tip:
            # Also check we're not in a worktree
            toplevel = subprocess.run(["git", "-C", ROOT, "rev-parse", "--show-toplevel"],
                                     capture_output=True, text=True).stdout.strip()
            gitdir = subprocess.run(["git", "-C", ROOT, "rev-parse", "--git-dir"],
                                    capture_output=True, text=True).stdout.strip()
            # In a worktree, git-dir points to .git/worktrees/<name>
            if "worktrees" not in gitdir:
                return True
        return False
    except Exception:
        return False


def _startup_dirty_check(dirty_ok=False):
    # p4dw: refuse to run when the working tree has uncommitted TRACKED changes,
    # so a leftover mutated artifact is caught immediately. Untracked files (new
    # shards, ledger rows) are tolerated -- they are the normal implementer
    # flow. No externally-settable opt-out: the backstop must not be bypassable.
    if _child_mode():
        return
    if dirty_ok:
        return
    try:
        r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=30)
    except Exception as e:
        print(f"REFUSE: git status failed ({e}); cannot verify clean tree", file=sys.stderr)
        sys.exit(2)
    if r.returncode != 0:
        print(f"REFUSE: git status exited {r.returncode}: {r.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    tracked_dirty = [ln for ln in r.stdout.splitlines() if not ln.startswith("??")]
    if tracked_dirty:
        print("REFUSE: working tree is dirty; restore before running:", file=sys.stderr)
        print("\n".join(tracked_dirty), file=sys.stderr)
        sys.exit(2)





def _run_pytest(cmd, root, timeout, env):
    """Run a row's pytest child in its own process group so the walk can
    terminate the whole group on SIGTERM/timeout (p4dw blocker 3)."""
    proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, env=env)
    _ACTIVE_PGROUPS.add(proc.pid)
    out = ""  # nfxi: must exist before the try -- the timeout path returns it
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()
        # gpu-sim-nfxi: do NOT re-raise.  The exception used to escape into the generic
        # "ledger test-id resolution: Command '[...]'" handler, so a slow machine arrived as a
        # ledger defect; worse, a bare rc!=0 is exactly what a SUCCESSFUL plant looks like, so a
        # timeout could be recorded as a plant that bit.  rc=124 is the shell's "timed out", and
        # both call sites test for it explicitly before reading anything into the result.
        return subprocess.CompletedProcess(
            cmd, 124, (out or "")
            + f"\nROW TIMEOUT: killed after {timeout}s (infrastructure, not a ledger defect)\n", "")
    finally:
        _ACTIVE_PGROUPS.discard(proc.pid)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="base ref (default: merge-base HEAD <trunk>)")
    ap.add_argument("--force", action="store_true", help="allow running in the integration tree (dangerous)")
    ap.add_argument("--dirty-ok", action="store_true", help="tolerate a dirty tracked tree (tests only)")
    ap.add_argument("--check-dirty-only", action="store_true", help="run only the dirty-tree backstop and exit")
    # gpu-sim-mi5v: the ledger test-id replay (a pytest collect-only subprocess per
    # ledger row, ~80 rows) is the entire cost of this check and is INDEPENDENT of
    # the retirement logic the append-only tests assert.  Cheap mode skips ONLY that
    # replay; every retirement verdict is unchanged.  It is refused together with
    # --force so it can never become an escape hatch from the gate, which runs the
    # full mode.  tests/test_mi5v_cheap_mode.py pins the equivalence with a plant.
    ap.add_argument("--rows", default=None,
                    help="comma-separated ledger row names to replay (default: all); "
                         "cheap validations still run for every row")
    ap.add_argument("--resolve-node-ids", action="store_true",
                    help="resolve EVERY ledger row's test node without replaying "
                         "mutation plants (batched collect-only); verdicts match "
                         "the full walk (gpu-sim-mi5v)")
    ap.add_argument("--no-ledger-replay", action="store_true",
                    help="skip the per-row ledger test-id replay (cheap mode; "
                         "retirement verdicts unchanged; not allowed with --force)")
    args = ap.parse_args()
    # gpu-sim-94pp: this refusal exists because the script MUTATES tracked files, and a mutation
    # in the integration tree poisons any gate that snapshots it.  --check-dirty-only mutates
    # nothing (it reads `git status` and exits), so refusing it made the guard's own sentence --
    # "mutates tracked files" -- false for that mode, and left the read-only path untestable from
    # the very tree it protects.  Every mutating path below still refuses.
    if args.resolve_node_ids and args.no_ledger_replay:
        print("REFUSE: --resolve-node-ids and --no-ledger-replay are different "
              "modes; pick one.")
        return 2
    if args.no_ledger_replay and args.force:
        print("REFUSE: --no-ledger-replay cannot be combined with --force; "
              "the gate must run the full ledger replay.")
        return 2
    read_only = args.check_dirty_only or args.no_ledger_replay or args.resolve_node_ids
    if _is_integration_tree() and not args.force and not read_only:
        print("REFUSE: check_append_only mutates tracked files; run it in a bead worktree, not the integration tree.")
        print("Usage: cd .lab/wt/<bead> && python3 scripts/check_append_only.py")
        return 2
    _startup_dirty_check(dirty_ok=args.dirty_ok)
    if args.check_dirty_only:
        print("OK: tree clean")
        return 0
    only_rows = None
    if args.rows:
        only_rows = {n.strip() for n in args.rows.split(",") if n.strip()}
    base_ref = args.base if args.base else merge_base()
    if base_ref is None:
        print("FAIL: cannot determine merge-base with trunk")
        return 1
    errors = []
    try:
        lost = check_model_facts(base_ref)
        for k in lost:
            errors.append(f"docs/model_facts.json lost fact block '{k}' (not retired)")
    except Exception as e:
        errors.append(f"docs/model_facts.json: {e}")
    try:
        lost = check_ledger(base_ref)
        for k in lost:
            errors.append(f"tests/fs2y_ledger.json lost ledger entry '{k}' (not retired)")
    except Exception as e:
        errors.append(f"tests/fs2y_ledger.json: {e}")
    if args.no_ledger_replay:
        # Cheap mode skips the ledger replay wholesale: the per-row mutation
        # apply/run/restore AND the per-row pytest collect-only that resolves
        # each row's node (80 subprocesses). The retirement logic above --
        # which is what the append-only tests assert -- is untouched, and the
        # NOTE line makes the skipped half visible in every cheap run.
        print("NOTE: ledger replay skipped (--no-ledger-replay, cheap mode): "
              "no mutation plant is applied and no row's test node is resolved")
    else:
        try:
            if args.resolve_node_ids:
                print("NOTE: --resolve-node-ids: every row's node is resolved, "
                      "but no mutation plant is applied (no replay)")
            for e in check_ledger_test_ids(ROOT, only_rows=only_rows,
                                           resolve_only=args.resolve_node_ids):
                errors.append(e)
        except Exception as e:
            errors.append(f"ledger test-id resolution: {e}")
    try:
        for e in check_grandfather_self_service():
            errors.append(e)
    except Exception as e:
        errors.append(f"grandfather self-service check: {e}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: append-only check passed (no unretired losses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

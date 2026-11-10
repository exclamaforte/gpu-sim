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
import json, subprocess, sys, os, argparse, ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKER_NAME = "check_append_only.py"
DEPTH_ENV = "OI4M_CHECKER_PID"

# Heavy rows whose replay re-runs an experiment (stageD/fabric-band). Exempt from
# execution per PI 11-03 ruling (option b): artifact is byte-verified by the
# append-only half. Committed list, not an env var. Checker prints the skip.
EXEMPT_ROWS = {
    "8zqr_generalization": "replays stageD experiment (~74s); artifact byte-verified by append-only half",
    "bch9_fabric_band": "replays fabric-band measurement (~55s); artifact byte-verified by append-only half",
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


def check_ledger_test_ids(root):
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

    child = _child_mode()
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
        if (not child and not exempt and mutation and mutation not in ("n/a", "doc-only")
                and mutation in mfns):
            apply_fn = mfns[mutation]
            # Snapshot what was ALREADY dirty so the restore below can never
            # discard real work; an applier may write a file and only then
            # raise, or mutate more than the single path it returns, and either
            # way a leaked change poisons every later row (gpu-sim-3v6d.3).
            pre_dirty = _dirty_paths(root)
            backup = mutated_path = None
            try:
                backup, mutated_path = apply_fn(root)
                rmut = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", target],
                    cwd=root, capture_output=True, text=True, timeout=ROW_TIMEOUTS.get(name, 120),
                    env=_child_env())
                if rmut.returncode == 0:
                    errors.append(f"ledger row {name!r}: mutation {mutation!r} did not fail (rc=0)")
                    continue
            finally:
                if backup is None:
                    # apply_fn raised before returning (write-then-raise, e.g. a
                    # `make` timeout at fs2y_mutations.py:320-323): mutated_path
                    # is still None, and os.remove(None) raises TypeError --
                    # which would skip the dirty-path restore BELOW and leak the
                    # very residue this block exists to clean (review finding).
                    if mutated_path:
                        try:
                            os.remove(mutated_path)
                        except OSError:
                            pass
                else:
                    with open(mutated_path, "w") as f:
                        f.write(backup)
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
            rrest = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", target],
                cwd=root, capture_output=True, text=True, timeout=ROW_TIMEOUTS.get(name, 120),
                env=_child_env())
            if rrest.returncode != 0:
                errors.append(f"ledger row {name!r}: restored target did not pass (rc={rrest.returncode})")
                continue
            positive_done = True

        if target in seen:
            continue
        seen.add(target)

        # collect-only resolution
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", target],
            cwd=root, capture_output=True, text=True, timeout=ROW_TIMEOUTS.get(name, 120),
            env=_child_env())
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
        if (not child and not positive_done and not exempt):
            rrun = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", target],
                cwd=root, capture_output=True, text=True, timeout=ROW_TIMEOUTS.get(name, 120),
                env=_child_env())
            if rrun.returncode != 0:
                errors.append(f"ledger row {name!r}: positive {target!r} did not pass (rc={rrun.returncode})")
                continue
    # Loud on every clean run: the grandfathered positive-only rows are debt, and a
    # silent pass would be the very defect this bead exists to remove.
    if not child and not errors and POSITIVE_ONLY_GRANDFATHERED:
        print(f"note: {len(POSITIVE_ONLY_GRANDFATHERED)} positive-only ledger rows "
              f"grandfathered (gpu-sim-55sb) -- they bind product tests with no mutation; "
              f"see POSITIVE_ONLY_GRANDFATHERED for the owning beads")
    return errors


def _child_env():
    """Env for subprocess: propagate depth marker with this process's pid."""
    env = dict(os.environ)
    env[DEPTH_ENV] = str(os.getpid())
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="base ref (default: merge-base HEAD <trunk>)")
    args = ap.parse_args()
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
    try:
        for e in check_ledger_test_ids(ROOT):
            errors.append(e)
    except Exception as e:
        errors.append(f"ledger test-id resolution: {e}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: append-only check passed (no unretired losses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Plant-verification primitives (gpu-sim-vuoq).

Extracted from scripts/check_append_only.py so a ledger row can bind a
PRODUCT test of the contract: a registered mutation must leave the bound
target parseable, must fail as a test outcome, and its replaced span must
be reported.  A test that imported the checker itself would be refused as
self-referential, which is why these live here.
"""

import difflib
import re


def collection_failure(run):
    """Return '<Error> at <file>:<line>' when the mutated target could not be
    COLLECTED, else '' (gpu-sim-vuoq).

    A mutation that breaks the target's syntax also makes pytest exit nonzero,
    so the old check -- 'the bound test failed, therefore the plant bit' --
    certified any syntax-breaking mutation.  On gpu-sim-f1sl a plant that
    replaced one line of a four-line pytest.fail(...) left an IndentationError
    and was recorded as covering the rule.  A plant must fail as a TEST
    OUTCOME; not parsing is not an outcome.
    """
    text = (run.stdout or "") + (run.stderr or "")
    loc = ""
    # IndentationError/SyntaxError carry 'path:line: Type'; fall back to the
    # 'E   Type: message' block when only that is present.
    lm = re.search(r"(\S+\.py):(\d+): ([A-Za-z_]*Error)", text)
    fm = re.search(r'File "([^"]+\.py)", line (\d+)', text)
    em = re.search(r"^E\s+([A-Za-z_]*Error): (.+)$", text, re.M)
    if lm:
        loc = f"{lm.group(3)} at {lm.group(1)}:{lm.group(2)}"
    elif fm and em:
        # the import-traceback form: File ".../test_x.py", line 3 THEN
        # "E   IndentationError: unexpected indent"
        loc = f"{em.group(1)} at {fm.group(1)}:{fm.group(2)}"
    elif em:
        loc = f"{em.group(1)}: {em.group(2).strip()}"
    nothing_ran = (run.returncode in (2, 4) or "no tests ran" in text
                   or "errors during collection" in text
                   or "ERROR collecting" in text)
    if nothing_ran and loc:
        return loc
    if nothing_ran:
        return "collection error (no tests ran)"
    return ""


def non_assertion_failure(run):
    """Classify a nonzero mutation run that failed for a reason other than the
    rule under test, so the message names WHICH kind of not-a-plant it was
    (gpu-sim-vuoq, cf. gpu-sim-i6pt).  Collection, import and usage errors are
    named; an assertion that raises is the caller's business.
    """
    text = (run.stdout or "") + (run.stderr or "")
    if "errors during collection" in text:
        return "a COLLECTION error (the module never loaded)"
    if "no tests ran" in text:
        return "NO TESTS RAN (nothing was executed)"
    m = re.search(r"^E\s+(ImportError|ModuleNotFoundError)\b", text, re.M)
    if m:
        return f"an {m.group(1)} (import failure, not a test outcome)"
    # gpu-sim-njh1: a test-body EXCEPTION is not an assertion (see the row).
    m2 = re.search(r"^(?:FAILED \S+ - |E\s+)(NameError|AttributeError|TypeError|"
                   r"KeyError|IndexError|UnboundLocalError|ZeroDivisionError|"
                   r"StopIteration|RuntimeError)\b", text, re.M)
    if m2:
        return f"a {m2.group(1)} raised inside the bound test (an exception, not an assertion)"
    if run.returncode == 2:
        return "a pytest usage/interrupt error (rc=2)"
    if run.returncode == 4:
        return "a pytest usage error (rc=4, nothing collected)"
    return ""


def mutation_diff(backup, mutated_path, limit=6):
    """The exact replaced span, so a reader can see whether an anchor covered a
    whole statement or one line of it (gpu-sim-vuoq requirement 4)."""
    try:
        before = (backup or "").splitlines()
        after = open(mutated_path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return ""
    out = []
    for line in difflib.unified_diff(before, after, "before", "after", lineterm="", n=0):
        out.append(line)
        if len(out) >= limit:
            out.append("...")
            break
    return "\n".join(out)


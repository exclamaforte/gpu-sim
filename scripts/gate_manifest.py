"""Manifest comparison for gate.sh: declared vs executed hard stages."""
import sys
from collections import Counter


def compare(declared, executed):
    """Return (missing, undeclared, duplicates) where:
    - missing: declared stages not in executed
    - undeclared: executed stages not in declared
    - duplicates: stages appearing more than once in executed
    """
    dcount = Counter(declared)
    ecount = Counter(executed)
    missing = sorted(set(declared) - set(executed))
    undeclared = sorted(set(executed) - set(declared))
    duplicates = sorted([k for k, v in ecount.items() if v > 1])
    return missing, undeclared, duplicates


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    declared = []
    executed = []
    i = 0
    while i < len(argv):
        if argv[i] == "--declared":
            declared = argv[i + 1].split(",")
            i += 2
        elif argv[i] == "--executed":
            executed = argv[i + 1].split(",")
            i += 2
        else:
            i += 1
    missing, undeclared, duplicates = compare(declared, executed)
    rc = 0
    if missing:
        print("MISSING (declared, not run): " + " ".join(missing))
        rc = 1
    if undeclared:
        print("UNDECLARED (run, not declared): " + " ".join(undeclared))
        rc = 1
    if duplicates:
        print("DUPLICATE (executed more than once): " + " ".join(duplicates))
        rc = 1
    if missing or undeclared or duplicates:
        print("MANIFEST: count mismatch: %d ran, %d declared" % (len(executed), len(declared)))
    sys.exit(rc)


if __name__ == "__main__":
    main()

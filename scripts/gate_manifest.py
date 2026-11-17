"""Manifest comparison for gate.sh: declared vs executed hard stages."""
import sys


def compare(declared, executed):
    """Return (missing, undeclared) where:
    - missing: declared stages not in executed
    - undeclared: executed stages not in declared
    """
    dset = set(declared)
    eset = set(executed)
    missing = sorted(dset - eset)
    undeclared = sorted(eset - dset)
    return missing, undeclared


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # Usage: gate_manifest.py --declared a,b,c --executed a,b,d
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
    missing, undeclared = compare(declared, executed)
    rc = 0
    if missing:
        print("MISSING (declared, not run): " + " ".join(missing))
        rc = 1
    if undeclared:
        print("UNDECLARED (run, not declared): " + " ".join(undeclared))
        rc = 1
    if missing or undeclared:
        print("MANIFEST: count mismatch: %d ran, %d declared" % (len(executed), len(declared)))
    sys.exit(rc)


if __name__ == "__main__":
    main()

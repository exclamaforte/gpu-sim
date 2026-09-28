#!/usr/bin/env python3
"""bd field-write guard: fail loudly rather than silently shrink a bead field.

The incident this exists because of: an appending script read a bead field,
got an empty string, and wrote the concatenation back -- replacing a list of
six acceptance criteria with one. bd's field overwrites are SILENT AND
PERMANENT: 'bd show --as-of' does not reconstruct field values, history
snapshots omit them, and the audit log records no event naming the field. So
the only defence is at write time.

Two assertions, both on every write:

  1. the value being appended to was READ NON-EMPTY. An empty read is the
     signature of a wrong field key, and concatenating onto it destroys the
     field while looking like a successful append.
  2. the length AFTER the write is >= the length BEFORE. A write that shrinks
     a field is refused and reported, whatever the caller intended.

Usage:
    bd_field_guard.py <bead-id> <field> <text-file-with-new-content> [--append]

    # with --append: field must be non-empty first, and the result must not
    # be shorter. Without it: the new content must not be shorter than what
    # is already there.

Exit 0 = safe to proceed (and, with --apply, written). Exit 1 = REFUSED.
"""
import subprocess
import sys
from pathlib import Path

FIELDS = {
    "acceptance_criteria": "ACCEPTANCE CRITERIA",
    "design": "DESIGN",
    "description": "DESCRIPTION",
    "notes": "NOTES",
    "title": "",
}


def read_field(bead: str, field: str) -> str:
    """The field's current text, as bd renders it."""
    out = subprocess.run(["bd", "show", bead], capture_output=True,
                         text=True).stdout
    marker = FIELDS.get(field, field.upper())
    if not marker:
        return out
    lines = out.splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.strip() == marker)
    except StopIteration:
        return ""
    # Section bodies are INDENTED; every section header and every trailer
    # (LABELS, COMMENTS, DEPENDS ON...) starts at column 0. An earlier version
    # tried to recognise headers by "uppercase and contains a space", which
    # matched nothing and therefore read the ENTIRE `bd show` output --
    # comments included -- as the field body. That made `before` 309k chars for
    # a 3k-char field, and a shrink check against it is not a check at all.
    body = []
    for l in lines[start + 1:]:
        if l.strip() and not l.startswith(" "):
            break
        body.append(l)
    return "\n".join(body).rstrip()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) < 3:
        print(__doc__)
        return 2
    bead, field, newfile = args
    new = Path(newfile).read_text()
    before = read_field(bead, field)
    append = "--append" in flags

    print(f"bead={bead} field={field} before={len(before)} chars "
          f"new={len(new)} chars append={append}")

    if append and not before.strip():
        print("REFUSED: the field being appended to read EMPTY. That is the "
              "signature of a wrong field key, and writing the concatenation "
              "back would destroy the field while looking like a success. "
              "Check the key, then re-run.")
        return 1
    if len(new.strip()) < len(before.strip()):
        print(f"REFUSED: the write SHRINKS the field "
              f"({len(before.strip())} -> {len(new.strip())} chars). "
              "Field overwrites in bd are silent and permanent, so this is "
              "refused rather than written.")
        return 1
    if append:
        print("OK: pre-conditions passed (non-empty read, no shrink).")
    else:
        print("OK: new content is not shorter than what is there.")
    if "--apply" in flags:
        subprocess.run(["bd", "update", bead, f"--{field}", new], check=True)
        after = read_field(bead, field)
        if len(after.strip()) < len(new.strip()):
            print(f"POST-WRITE CHECK FAILED: readback is "
                  f"{len(after.strip())} chars, expected >= {len(new.strip())}.")
            return 1
        print(f"written; readback {len(after)} chars")
    return 0


def _self_test() -> int:
    """A guard with a broken reader is worse than no guard.

    Two of the checks above were themselves wrong on the first pass: the
    section-body detector matched no headers, so `read_field` returned the
    ENTIRE `bd show` output (309k chars, comments included) as a 3k-char
    field, and every shrink comparison was then against the wrong number.
    So the reader is pinned here against a synthetic `bd show` rendering.
    """
    import tempfile

    sample = (
        "x gpu-sim-test [OPEN]\n"
        "Created by: someone\n"
        "\n"
        "DESCRIPTION\n"
        "\n"
        "  line one\n"
        "  line two\n"
        "\n"
        "ACCEPTANCE CRITERIA\n"
        "\n"
        "  1. first item\n"
        "  2. second item\n"
        "\n"
        "LABELS\n"
        "  needs-review\n"
        "\n"
        "COMMENTS\n"
        "  a very long comment that must NOT be counted as criteria\n")
    real = subprocess.run

    def fake(args, **kw):
        if args[:2] == ["bd", "show"]:
            return type("R", (), {"stdout": sample, "returncode": 0})()
        return real(args, **kw)

    subprocess.run = fake
    try:
        ac = read_field("gpu-sim-test", "acceptance_criteria")
        de = read_field("gpu-sim-test", "description")
    finally:
        subprocess.run = real
    # compare on the de-indented body: bd renders section text indented by
    # two spaces, and that indent is presentation, not content
    def _flat(s):
        return "\n".join(l.strip() for l in s.strip().splitlines() if l.strip())

    ok = True
    for name, got, want in (("acceptance", _flat(ac), "1. first item\n2. second item"),
                            ("description", _flat(de), "line one\nline two")):
        if want not in got:
            print(f"FAIL self-test: {name} reader returned {got!r}")
            ok = False
    if "COMMENTS" in ac or "a very long comment" in ac:
        print("FAIL self-test: the reader swallowed the COMMENTS section")
        ok = False
    if ok:
        print("OK  self-test: section reader stops at the next column-0 header "
              "and does not swallow COMMENTS")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())

#!/usr/bin/env bash
# gpu-sim-u2o.20, criterion 4: assert that every file the build needs is
# TRACKED, so a build gate cannot go green on untracked state again.
#
# The failure this prevents, in its own words: "make check passes" was true in
# this working tree and could not be produced by a clean clone, because five of
# the Makefile's SRCS files (and the header and three test programs) existed
# only on disk. A gate whose validity depends on untracked state is not a gate.
#
# It is a GUARD rather than a comment because every instance of this class in
# this lab survived a comment: the .gitignore already warned that a missing
# libfakegpu.so "can make a gate pass VACUOUSLY", and the gate passed anyway.
#
# Run: bash tests/assert_build_inputs_tracked.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail=0
checked=0

# 1. every SRCS the Makefile names must be tracked
srccount=$(grep -cE '^\s*src/[^ ]+\.c' Makefile || true)
while read -r f; do
    [ -z "$f" ] && continue
    checked=$((checked + 1))
    if [ ! -e "$f" ]; then
        echo "  FAIL $f is named by the Makefile and DOES NOT EXIST"
        fail=$((fail + 1))
    elif ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        echo "  FAIL $f is named by the Makefile and is UNTRACKED"
        fail=$((fail + 1))
    else
        echo "  ok   $f tracked"
    fi
done < <(grep -oE 'src/[A-Za-z0-9_./-]+\.c' Makefile | sort -u)

# 2. the same for the headers and the test programs `make check` compiles,
#    because criterion 3 is about `make check`, not just about the library
for f in include/fakegpu.h shim/cuda_runtime.h shim/nccl.h \
         tests/fake_client.c tests/vector_add.c tests/ring_allreduce.c \
         tests/test_cudart_none.cu tests/test_cudart_none.sh \
         tests/test_device_prop_overlay.cu tests/test_shim_prop_size.c; do
    [ -e "$f" ] || continue          # not required by this Makefile
    checked=$((checked + 1))
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        echo "  ok   $f tracked"
    else
        echo "  FAIL $f is used by the build and is UNTRACKED"
        fail=$((fail + 1))
    fi
done

echo
if [ "$fail" -ne 0 ]; then
    echo "BUILD INPUTS GUARD FAILED: $fail of $checked build inputs are missing or untracked."
    echo "A clean clone cannot build what this tree builds, so any 'make check"
    echo "passes' claim from this tree is evidence about the tree, not the repo."
    exit 1
fi
echo "BUILD INPUTS GUARD PASSED: all $checked build inputs are tracked."

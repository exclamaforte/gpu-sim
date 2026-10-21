#!/usr/bin/env python3
"""Check that all #ifndef guards in include/ and shim/ are unique."""
import os, re, sys

dirs = ["include", "shim"]
guards = {}
for d in dirs:
    if not os.path.isdir(d):
        continue
    for root, _, files in os.walk(d):
        for f in files:
            if not f.endswith(".h"):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                for line in fh:
                    m = re.match(r'\s*#ifndef\s+(\w+)', line)
                    if m:
                        guard = m.group(1)
                        if guard in guards:
                            print("DUPLICATE GUARD: %s in %s and %s" % (guard, guards[guard], path))
                            sys.exit(1)
                        guards[guard] = path
print("PASS: all %d guards unique" % len(guards))

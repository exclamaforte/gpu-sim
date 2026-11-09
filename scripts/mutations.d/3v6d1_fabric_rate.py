def apply(root):
    """Mutation for 3v6d1_fabric_rate: remove the bidirectional halving in
    topology.effective_bw, so the one-way rate becomes the spec's aggregate
    900 GB/s. The Python rule, the emitted binary and the transport then disagree
    with the model formula in the same way they did before gpu-sim-3v6d.1.

    Raises if the substitution does not apply, so a silent no-op cannot be
    reported as a mutation that failed to fail.
    """
    import os
    path = os.path.join(root, "epic_cluster", "topology.py")
    backup = open(path).read()
    old = 'if link.get("bw_is_bidirectional", False):\n        bw *= 0.5\n'
    # MEASURED 2026-11-04: the mutated text must differ in LENGTH from the
    # original. conftest.py redirects bytecode to .lab/tmp/pycache and CPython
    # validates a .pyc on (mtime, size); a same-length mutation restored inside
    # the same mtime second leaves a stale .pyc that serves the MUTATED code to
    # the restored run, so the checker reports 'restored target did not pass'.
    # The trailing comment makes the mutated file longer, so the signature
    # differs in both directions.
    new = ('if link.get("bw_is_bidirectional", False):\n'
           '        bw *= 1.0  # 3v6d1 mutation: halving removed\n')
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("3v6d1 mutation did not apply: halving block not found "
                           "in epic_cluster/topology.py")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path

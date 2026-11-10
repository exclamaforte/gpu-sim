MUTATIONS = {"3v6d2_link_calendar": "3v6d.2 per-link calendar: CAS replaced by plain store"}

def apply(root):
    """Mutation for 3v6d2_link_calendar: remove the calendar reservation entirely."""
    import os
    path = os.path.join(root, "src", "fakegpu_fabric.c")
    backup = open(path).read()
    old = """        do {
            busy = atomic_load_explicit(&f->hdr->busy_until[link_id], memory_order_acquire);
            uint64_t now = now_ns();
            start = (busy > now) ? busy : now;
            finish = start + delay_ns;
        } while (!atomic_compare_exchange_strong_explicit(&f->hdr->busy_until[link_id],
                                                            &busy, finish,
                                                            memory_order_acq_rel,
                                                            memory_order_acquire));"""
    new = """        /* mutation: calendar removed -- no serialization */
        uint64_t now = now_ns();
        finish = now + delay_ns;"""
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("3v6d2 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    import subprocess
    subprocess.run(["make"], cwd=root, capture_output=True, text=True, timeout=300)
    return backup, path

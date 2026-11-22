#!/usr/bin/env python3
"""A2 with its negative control: node isolation must be fail-able.

Runs `probe_nodes.py` twice in the same invocation:
  * normal: 8 children, VDEV_NODE=0..7 -> 64 distinct D1 bus ids;
  * disabled: every child VDEV_NODE=0 -> the same isolation assertions must
    FAIL, otherwise the "isolation" claim is a demonstration, not a test.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NODES = HERE / "probe_nodes.py"


def first_json(text):
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "checks" in obj:
            return obj
    return None


def run(*args):
    proc = subprocess.run(
        [sys.executable, str(NODES), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode, first_json(proc.stdout), proc.stdout, proc.stderr


def main():
    normal_rc, normal, nout, nerr = run()
    neg_rc, negative, gout, gerr = run(
        "--all-node", "0", "--negative-control"
    )
    normal_ok = normal_rc == 0 and normal is not None and all(normal["checks"].values())
    negative_ok = (
        neg_rc != 0
        and negative is not None
        and not negative["checks"]["64_distinct_pci_bus_ids"]
        and not negative["checks"]["bus_set_is_d1_exact"]
        and not negative["checks"]["nodes_have_distinct_first_bus"]
        and negative["checks"]["every_node_sees_8"]
        and negative["checks"]["each_child_has_8_ids"]
        and negative["checks"]["every_node_names_b300"]
    )
    out = {
        "checks": {
            "normal_isolation_ok": normal_ok,
            "negative_control_failed_as_required": negative_ok,
        },
        "normal": normal,
        "negative_control": negative,
        "normal_rc": normal_rc,
        "negative_rc": neg_rc,
    }
    sys.stdout.write(nout)
    sys.stderr.write(nerr)
    sys.stdout.write(gout)
    sys.stderr.write(gerr)
    print(json.dumps(out, indent=2))
    if normal_ok and negative_ok:
        print(
            "A2 OK: normal run has 64 distinct D1 bus ids; with --all-node 0 "
            "the same checks fail (64_distinct_pci_bus_ids=False, "
            "bus_set_is_d1_exact=False, nodes_have_distinct_first_bus=False); "
            f"probe_nodes exits nonzero ({neg_rc}), as expected."
        )
        return 0
    print(f"A2 FAIL: normal_ok={normal_ok} negative_ok={negative_ok}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

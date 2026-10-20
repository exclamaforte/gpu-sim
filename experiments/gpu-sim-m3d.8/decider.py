#!/usr/bin/env python3
"""m3d.8 acceptance 2: trace reconciliation + per-op decider.

Reads the plugin's structured trace (rank-tagged) and the model's expectation
(via epic_cluster.link_model's PUBLIC route) and prints/writes the verdict.
"""
import json
import re
import sys
from collections import defaultdict

FIELDS = re.compile(r"(\w+)=(-?\w+)")


def parse(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = {"_raw": line}
            for k, v in FIELDS.findall(line):
                d[k] = v
            if "bytes" not in d and "size" in d:
                d["bytes"] = d["size"]
            d["_op"] = ("isend" if "isend" in line else
                        "irecv" if "irecv" in line else
                        "recv" if "recv-done" in line else
                        "other")
            recs.append(d)
    return recs


def model_per_rank(spec_count, dtype, nranks):
    """Per-rank wire bytes via the PUBLIC route (no private symbols)."""
    from epic_cluster.link_model import predict
    spec = {
        "version": 1,
        "models": {"m": {"sm_count": 80, "clock_mhz": 1000, "mem_bytes": 1 << 30,
                         "mem_bw_gbs": 1000.0, "arch": "x", "source": "test"}},
        "nodes": [{"name": "n0", "devices": [{"id": i, "model": "m"} for i in range(nranks)]}],
        "links": [{"a": {"node": "n0", "device": i}, "b": {"node": "n0", "device": i + 1},
                   "kind": "nvlink", "bw_gbps": 100.0, "latency_us": 5.0, "mtu": 4096,
                   "loss_ppm": 0, "jitter_us": 0.0, "shared_group": None}
                  for i in range(nranks - 1)],
        "ranks": [{"rank": i, "node": "n0", "device": i} for i in range(nranks)],
    }
    r = predict(spec, "ring_allreduce", spec_count, dtype, nranks)
    total = sum(r["per_link_bytes"].values())
    return total / nranks, total


def main(trace, payload_bytes, nranks, dtype, count, ack_path, out_path):
    recs = parse(trace)
    per_rank = defaultdict(lambda: {"isend": [], "irecv": [], "recv": []})
    for r in recs:
        if r["_op"] == "other":
            continue
        rk = int(r.get("rank", -1))
        per_rank[rk][r["_op"]].append(r)

    report = {"per_rank": {}, "verdict": None}

    # 2a: entry counts == record counts, per entry point, per rank
    ok_2a = True
    for rk, ops in sorted(per_rank.items()):
        entry = {}
        for op, key in (("isend", "cnt_isend"), ("irecv", "cnt_irecv"), ("recv", "cnt_recv_done")):
            cnt = max([int(x[key]) for x in ops[op] if key in x] or [0])
            entry[op] = {"records": len(ops[op]), "counter": cnt,
                         "consistent": cnt == len(ops[op])}
            if cnt != len(ops[op]):
                ok_2a = False
        report["per_rank"][rk] = entry

    # 2b: per-transfer digests and the sender-records vs receiver-consumed_seq
    # asymmetry, compared PER SENDER (a sender's isend count vs the peer's
    # consumed_seq witness for that sender). N != M => counting defect;
    # N == M == 2 with DISTINCT digests => two real transfers.
    digests = defaultdict(list)
    send_records = defaultdict(int)          # rank -> isend records
    peer_consumed = defaultdict(int)         # sender rank -> max consumed_seq at its peer
    for rk, ops in sorted(per_rank.items()):
        for r in ops["isend"]:
            send_records[rk] += 1
            digests[(rk, r.get("chan"))].append(r.get("digest"))
        for r in ops["recv"]:
            if "consumed_seq" in r and "peer" in r:
                peer_consumed[int(r["peer"])] = max(peer_consumed[int(r["peer"])], int(r["consumed_seq"]))
    binding = []
    for rk in sorted(set(send_records) | set(peer_consumed)):
        n = send_records.get(rk, 0)
        m = peer_consumed.get(rk, 0)
        binding.append({"sender_rank": rk, "sender_isend_records": n,
                        "peer_consumed_seq": m,
                        "distinct_digests": sorted({d for d in digests.get((rk, "0"), []) if d}),
                        "verdict": ("two real transfers" if n == m == 2 else
                                    "counting defect" if n != m else "single/other")})
    report["per_op_binding"] = binding
    report["payload_digests"] = {f"{k[0]}:{k[1]}": v for k, v in digests.items()}
    report["binding_verdict"] = binding

    # 2b-phase: channel + payload classification on the isend side
    phases = defaultdict(list)
    for rk, ops in per_rank.items():
        for r in ops["isend"]:
            if int(r.get("bytes", 0)) == 0:
                continue
            phases[(rk, r.get("chan"))].append(int(r["bytes"]))
    full_payload_chans = [k for k, v in phases.items() if all(b == payload_bytes for b in v)]
    shape = None
    dup = False
    for ck, ds in digests.items():
        seen = [d for d in ds if d]
        if len(seen) != len(set(seen)):
            dup = True
    if dup:
        shape = "duplicate reporting: identical payload digest repeated on one channel id"
    elif any(len(v) >= 2 for v in phases.values()):
        shape = "2S: multiple full-payload transfers on the same channel id (distinct phases, distinct digests)"
    elif len(phases) > 1 and all(len(v) == 1 for v in phases.values()):
        shape = "one phase SPLIT ACROSS channel ids (distinct finding, not 2S)"
    elif sum(len(v) for v in phases.values()) == 1:
        shape = "single transfer"
    report["phase_shape"] = {"channels": {str(k): v for k, v in phases.items()},
                             "full_payload_channel_keys": [str(k) for k in full_payload_chans],
                             "shape": shape}

    obs_total = sum(int(r["bytes"]) for ops in per_rank.values() for r in ops["isend"] if "bytes" in r)
    obs_per_rank = obs_total / max(len(per_rank), 1)
    exp_per_rank, exp_total = model_per_rank(count, dtype, nranks)
    ratio = (obs_total / exp_total) if exp_total else None

    report["bytes"] = {
        "observed_per_rank": obs_per_rank,
        "observed_total": obs_total,
        "model_per_rank": exp_per_rank,
        "model_total": exp_total,
        "ratio_observed_over_model": ratio,
    }

    ack = None
    try:
        ack = json.load(open(ack_path))
    except Exception as e:
        report["ack_error"] = str(e)
    report["ack"] = ack

    ok_2c = False
    if ratio == 1.0:
        ok_2c = True
    elif ack and abs(ack.get("ratio_observed_over_model", -1) - ratio) < 1e-9 \
            and ack.get("tracking_issue") == "gpu-sim-8sy":
        ok_2c = True

    report["verdict"] = {
        "2a_trace_complete": ok_2a,
        "2c_bytes": ok_2c,
        "phase_shape": shape,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["verdict"], indent=2))
    print("bytes:", json.dumps(report["bytes"]))
    return 0 if (ok_2a and ok_2c) else 1


if __name__ == "__main__":
    sys.path.insert(0, ".")
    trace, ack, out = sys.argv[1], sys.argv[2], sys.argv[3]
    payload = int(sys.argv[4]); nranks = int(sys.argv[5]); count = int(sys.argv[6]); dtype = sys.argv[7]
    sys.exit(main(trace, payload, nranks, dtype, count, ack, out))

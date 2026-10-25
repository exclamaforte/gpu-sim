import json, copy
base = json.load(open("specs/h100_rack.json"))
# 4-node spec: 4 nodes x 8 GPUs
nodes = []
for ni in range(4):
    nodes.append({"name": f"node{ni}", "devices": [{"id": i, "model": "h100_sxm"} for i in range(8)]})
# links: nvlink within node (all pairs? h100_rack uses pairwise nvlink among 8)
# h100_rack links: let's inspect
print("num links in 2-node:", len(base["links"]))
print("link kinds:", set(l["kind"] for l in base["links"]))
# Reuse: intra-node nvlink links for each node, ib links between every pair of nodes (per-GPU NIC)
# Simpler: generate nvlink all-pairs within node, ib between corresponding GPUs across nodes
links = []
# nvlink within each node (complete graph, like base)
for ni in range(4):
    for i in range(8):
        for j in range(i+1, 8):
            links.append({"a":{"node":f"node{ni}","device":i},"b":{"node":f"node{ni}","device":j},
                          "kind":"nvlink","bw_gbps":900.0,"latency_us":1.0,"mtu":4096,"loss_ppm":0,"jitter_us":0,"shared_group":None})
# ib between nodes: per-GPU NIC, each GPU has its own (shared_group null)
for ni in range(4):
    for nj in range(ni+1, 4):
        for i in range(8):
            links.append({"a":{"node":f"node{ni}","device":i},"b":{"node":f"node{nj}","device":i},
                          "kind":"ib","bw_gbps":50.0,"latency_us":5.0,"mtu":4096,"loss_ppm":0,"jitter_us":0,"shared_group":None})
# ranks: global rank = node*8 + device
ranks = []
for ni in range(4):
    for i in range(8):
        ranks.append({"rank": ni*8+i, "node": f"node{ni}", "device": i})
spec4 = {"version":1, "sources": base["sources"], "models": base["models"], "nodes": nodes, "links": links, "ranks": ranks}
with open("specs/h100_rack_4node.json","w") as f:
    json.dump(spec4, f, indent=2)
print("written 4-node spec, ranks:", len(ranks), "links:", len(links))

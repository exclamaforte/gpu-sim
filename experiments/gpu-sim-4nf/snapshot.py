"""gpu-sim-4nf: attribute the flash-path memory gap via a torch memory snapshot."""
import json, os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

D = 512; DFF = 4*D; HEADS = 8; HEAD_DIM = D//HEADS; LAYERS = 6

class FlashBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3*D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, DFF)
        self.ff2 = nn.Linear(DFF, D)
    def forward(self, x):
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(x.shape[0], x.shape[1], 3, HEADS, HEAD_DIM).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q,k,v)
        a = a.transpose(1,2).reshape(x.shape[0], x.shape[1], D)
        x = x + self.o(a)
        h = self.ln2(x)
        x = x + self.ff2(F.gelu(self.ff1(h)))
        return x

def build(seq):
    return nn.Sequential(*[FlashBlock() for _ in range(LAYERS)])

def top_allocs(snapshot, n=12):
    # snapshot is a list of segments; each has 'allocations' (list of blocks)
    blocks = []
    for seg in snapshot:
        for a in seg.get("allocations", []):
            size = a.get("size", 0)
            stack = a.get("frames", []) or a.get("stack", [])
            # build a short label from the top frame
            label = "unknown"
            for fr in (stack or []):
                fn = fr.get("filename", "") if isinstance(fr, dict) else str(fr)
                name = fr.get("name", "") if isinstance(fr, dict) else ""
                if "scaled_dot_product" in name or "sdpa" in name.lower() or "flash" in name.lower():
                    label = "sdpa/flash"; break
                if "matmul" in name or "mm" == name or "addmm" in name:
                    label = "matmul"; break
                if "gelu" in name or "relu" in name:
                    label = "activation (gelu)"; break
                if "layer_norm" in name or "layernorm" in name.lower():
                    label = "layernorm"; break
                if "linear" in name or "addmm" in name:
                    label = "linear"; break
            blocks.append((size, label, stack))
    blocks.sort(reverse=True)
    return blocks[:n]

results = {}
for seq in (4096, 8192):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(seq).cuda().to(torch.bfloat16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(1, seq, D, device="cuda", dtype=torch.bfloat16)
    # warmup
    # start recording AFTER warmup
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=100000)
    torch.cuda.reset_peak_memory_stats()
    # measured run
    y = m(x); loss = y.sum()
    loss.backward(); opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    # augment with stack traces
    torch.cuda.memory._augment_memory_snapshot_stack_traces(snap)

    def classify(frames):
        text = " ".join((fr.get("name","") if isinstance(fr,dict) else str(fr)) for fr in (frames or []))
        tl = text.lower()
        if "scaled_dot_product" in tl or "sdpa" in tl or "flash" in tl or "efficient_attention" in tl or "mem_efficient" in tl:
            return "sdpa_workspace"
        if "softmax" in tl:
            return "softmax_scores"
        if "bmm" in tl or "matmul" in tl or "addmm" in tl or "mm(" in tl or "gemm" in tl:
            return "matmul_intermediate"
        if "gelu" in tl or "relu" in tl or "silu" in tl:
            return "activation (gelu)"
        if "layer_norm" in tl or "layernorm" in tl or "native_layer_norm" in tl:
            return "layernorm"
        if "copy" in tl or "contiguous" in tl or "reshape" in tl or "view" in tl or "permute" in tl or "transpose" in tl:
            return "view/contiguous"
        return "other"

    # blocks across all segments
    all_blocks = []
    for seg in snap.get("segments", []):
        for b in seg.get("blocks", []):
            if b.get("state") == "active_allocated":
                all_blocks.append(b)

    # aggregate by classification
    from collections import defaultdict
    agg = defaultdict(int)
    for b in all_blocks:
        agg[classify(b.get("frames", []))] += b.get("size", 0)
    agg_sorted = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    agg_list = [{"category": k, "size_mb": round(v/1e6,3)} for k,v in agg_sorted]

    # top individual blocks with their innermost meaningful frame
    blocks_sorted = sorted(all_blocks, key=lambda b: b.get("size",0), reverse=True)
    top = []
    for b in blocks_sorted[:12]:
        frames = b.get("frames", []) or []
        # find the most specific frame (last python-ish / op name)
        name = ""
        for fr in reversed(frames):
            if isinstance(fr, dict):
                nm = fr.get("name","")
                if nm and ("::" not in nm or "native" in nm.lower() or "sdpa" in nm.lower() or "flash" in nm.lower()):
                    name = nm
                    break
        if not name and frames:
            fr = frames[-1] if isinstance(frames[-1], dict) else frames[-1]
            name = fr.get("name","") if isinstance(fr,dict) else str(fr)
        top.append({"size_mb": round(b.get("size",0)/1e6,3), "frame": name[:120]})
    results[seq] = {"peak_gb": round(peak/1e9,4), "top_allocs": top, "agg_by_category": agg_list}
    print(f"\n=== seq={seq} peak={peak/1e9:.4f} GB ===")
    print("Aggregated by category (active blocks at peak):")
    for item in agg_list:
        print(f"  {item['size_mb']:8.3f} MB  {item['category']}")
    print("Top individual blocks:")
    for item in top:
        print(f"  {item['size_mb']:8.3f} MB  {item['frame']}")
    del m, opt, x, y, loss
    torch.cuda.empty_cache()

with open("experiments/gpu-sim-4nf/snapshot_results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nsaved snapshot_results.json")

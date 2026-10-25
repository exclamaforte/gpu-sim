"""gpu-sim-4nf: instrumented per-block memory attribution for flash path."""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D=512; DFF=4*D; HEADS=8; HEAD_DIM=D//HEADS; LAYERS=6

class FlashBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(D); self.qkv=nn.Linear(D,3*D); self.o=nn.Linear(D,D)
        self.ln2=nn.LayerNorm(D); self.ff1=nn.Linear(D,DFF); self.ff2=nn.Linear(DFF,D)
    def forward(self,x,probe=None):
        if probe: probe("ln1_in", x)
        h=self.ln1(x)
        if probe: probe("after_ln1", h)
        qkv=self.qkv(h).reshape(x.shape[0],x.shape[1],3,HEADS,HEAD_DIM).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        if probe: probe("after_qkv", q)
        a=F.scaled_dot_product_attention(q,k,v)
        if probe: probe("after_sdpa", a)
        a=a.transpose(1,2).reshape(x.shape[0],x.shape[1],D)
        x=x+self.o(a)
        if probe: probe("after_attn_out", x)
        h=self.ln2(x)
        if probe: probe("after_ln2", h)
        x=x+self.ff2(F.gelu(self.ff1(h)))
        if probe: probe("after_ffn", x)
        return x

def run(seq):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=nn.Sequential(*[FlashBlock() for _ in range(LAYERS)]).cuda().to(torch.bfloat16)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    x=torch.randn(1,seq,D,device="cuda",dtype=torch.bfloat16)
    # warmup
    with torch.no_grad():
        y=m(x); y.sum()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    peaks={}
    def probe(name,t):
        torch.cuda.synchronize()
        peaks[name]=torch.cuda.max_memory_allocated()
    # forward through layers manually to capture per-layer peaks
    h=x
    for i,blk in enumerate(m):
        h=blk(h, lambda n,t: None)
    # Instead, instrument a single layer with probes, times layers
    # Re-run one layer with probes
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    blk=m[0]
    inp=x.detach().clone()
    layer_peaks={}
    def probe2(name,t):
        torch.cuda.synchronize()
        layer_peaks[name]=torch.cuda.max_memory_allocated()
    out=blk(inp, probe2)
    loss=out.sum(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    final=dialogs= torch.cuda.max_memory_allocated()
    print(f"\nseq={seq}")
    print(f"  baseline (params+grads+opt) peak before layer: see below")
    # print layer peaks
    base = 0
    prev=0
    for name,pk in layer_peaks.items():
        print(f"  {name:20s} peak={pk/1e9:.4f} GB  (delta from prev { (pk-prev)/1e6:.1f} MB)")
        prev=pk
    print(f"  final peak (1 layer fwd+bwd): {final/1e9:.4f} GB")
    # full model
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    y=m(x); loss=y.sum(); loss.backward(); opt.step(); torch.cuda.synchronize()
    full=torcha = torch.cuda.max_memory_allocated()
    print(f"  FULL model peak: {full/1e9:.4f} GB  (delta vs 1-layer: {(full-final)/1e9:.4f} GB)")
    del m,opt,x,y,loss
    torch.cuda.empty_cache()

for seq in (4096, 8192):
    run(seq)

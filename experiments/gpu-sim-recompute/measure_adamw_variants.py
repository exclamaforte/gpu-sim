"""Can the PEAK be tamed? AdamW implementation variants at fixed fp32 state (gpu-sim-scup).

The scup finding is that fp32-moment styles peak at 20-24 B/param against 12 steady, because the optimizer
step allocates parameter-sized temporaries -- and peak is what OOMs. If that peak is an implementation detail
(foreach vs fused vs plain loop), then the feasibility boundary is a CHOICE, not a property, and the model
should say so.

Method: fp32 parameter tensor, three AdamW implementations, one fwd/bwd/step. Report peak and steady B/param.
"""
import json

import torch

N = 500_000_000          # 2 GB at fp32, so states+temporaries stay inside a 31.4 GiB card
DEV = "cuda"


def run(label: str, **kw) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    p = torch.nn.Parameter(torch.randn(N, device=DEV, dtype=torch.float32))
    opt = torch.optim.AdamW([p], lr=1e-4, **kw)
    p.sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    steady = torch.cuda.memory_allocated()
    return {"variant": label, "peak_bpp": round(peak / N, 2), "steady_bpp": round(steady / N, 2),
            "peak_gb": round(peak / 1e9, 3)}


if __name__ == "__main__":
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "params": N, "rows": []}
    for label, kw in (("default (foreach)", {}),
                      ("foreach=False", {"foreach": False}),
                      ("fused=True", {"fused": True}),
                      ("fused + foreach=False", {"fused": True, "foreach": False})):
        try:
            out["rows"].append(run(label, **kw))
        except Exception as e:                     # a variant may be unsupported: say so, do not fake it
            out["rows"].append({"variant": label, "error": f"{type(e).__name__}: {e}"})
    print(json.dumps(out, indent=2))

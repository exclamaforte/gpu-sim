import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-mdz"))
import capture_calibration as cc

import torch
CASES = []
# elementwise relu: below 2^11 and above 2^28
import torch
for n in [256,512,1024, 1<<29, 1<<30, 1<<31]:
    CASES.append(({"op_class":"elementwise","op":"relu","shapes":[n],"dtype":"torch.float32","want":"aten::relu"}, lambda n=n: torch.relu(torch.randn(n, device="cuda", dtype=torch.float32))))
# reduce sum: below 2^12 and above 2^26
for n in [512,1024,2048, 1<<27, 1<<28, 1<<29]:
    CASES.append(({"op_class":"reduce","op":"sum","shapes":[n],"dtype":"torch.float32","want":"aten::sum"}, lambda n=n: torch.sum(torch.randn(n, device="cuda", dtype=torch.float32))))
# matmul: below n=256 and above n=12288? use n 32,64,128 and 2048,4096,8192
for n in [32,64,128, 16384,24576,32768]:
    CASES.append(({"op_class":"matmul","op":"matmul","shapes":[n,n,n],"dtype":"torch.float32","want":"aten::matmul"}, lambda n=n: (torch.randn(n,n,device="cuda",dtype=torch.float32) @ torch.randn(n,n,device="cuda",dtype=torch.float32))))

def main():
    trace_dir = Path(__file__).resolve().parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i,(desc,thunk) in enumerate(CASES):
        rec = cc.capture_case(i, desc, thunk, trace_dir)
        out.append(rec)
        print(i, rec.get("op_class"), rec.get("shapes"), rec.get("device_ms_median"))
    fixture = {
        "bead": "gpu-sim-v4e",
        "op_classes": ["elementwise","reduce","matmul"],
        "n_cases": len(out),
        "cases": out,
    }
    (Path(__file__).resolve().parent / "v4e_fixture.json").write_text(__import__("json").dumps(fixture, indent=2))
    print("wrote fixture")

if __name__ == "__main__":
    main()

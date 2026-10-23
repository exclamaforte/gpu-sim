import pathlib
HERE=pathlib.Path(__file__).resolve().parent
out=HERE/"rows.txt"
lines=[]
for fam in ["elementwise","reduction","softmax","transpose"]:
    lo=1<<20; hi=1<<24
    sizes=sorted(set(int(lo + (hi-lo)*i/59) for i in range(60)))
    for n in sizes:
        for b in [128,256,512]:
            lines.append(f"{fam} {n} {b}")
for m in range(512,4097,128):
    n=m*m
    if n > (1<<24): break
    for b in [128,256,512]:
        lines.append(f"matmul {n} {b}")
out.write_text("\n".join(lines)+"\n")
print("rows:", len(lines))

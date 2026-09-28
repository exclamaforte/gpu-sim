#!/usr/bin/env python3
"""Mine real-world kernel corpora for realistic distribution coverage.

Scans local CUDA sources (repo + optional extra roots) for __global__
kernels, extracts them with file/line provenance, and writes a JSONL
corpus for idiom-coverage comparison against synthetic stages 1-5.
No network access: point --roots at local checkouts (CUTLASS examples,
cuDNN/cuBLAS samples, etc.).
"""

import json
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator, Optional


GLOBAL_RE = re.compile(r'__global__\s+[\w:\<\>\s\*&]+\s+(\w+)\s*\(')


@dataclass
class CorpusKernel:
    kernel_name: str
    source: str
    file: str
    line: int
    provenance: str

    def kernel_hash(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d['metadata'] = {
            'generator': 'corpus_realworld',
            'op_type': 'realworld_kernel',
            'family_id': 'corpus_realworld',
            'provenance': self.provenance,
        }
        return d


def extract_kernels_from_file(path: Path, provenance: str) -> Iterator[CorpusKernel]:
    try:
        text = path.read_text(errors='ignore')
    except OSError:
        return
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = GLOBAL_RE.search(line)
        if not m:
            continue
        name = m.group(1)
        # Capture a bounded window: signature + next 60 lines (lightweight, no full parse)
        window = lines[i:min(len(lines), i + 60)]
        yield CorpusKernel(kernel_name=name, source='\n'.join(window),
                           file=str(path), line=i + 1, provenance=provenance)


def mine_corpus(roots: list, output: str, max_kernels: int = 500) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    kernels: list = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for pat in ('*.cu', '*.cuh', '*.h'):
            for f in rp.rglob(pat):
                for k in extract_kernels_from_file(f, provenance=rp.name):
                    kernels.append(k)
                    if len(kernels) >= max_kernels:
                        break
                if len(kernels) >= max_kernels:
                    break
            if len(kernels) >= max_kernels:
                break
    with open(out / 'corpus.jsonl', 'w') as fh:
        for k in kernels:
            fh.write(json.dumps(k.to_dict()) + '\n')
    # Idiom summary: which synthetic families' keywords appear
    idioms = ['__shared__', '__syncthreads', 'atomicAdd', 'cooperative_groups',
              'wmma::', 'cudaStream', '<<<', '__shfl']
    summary = {idi: sum(1 for k in kernels if idi in k.source) for idi in idioms}
    meta = {'total_kernels': len(kernels), 'idiom_hits': summary,
            'roots': [str(r) for r in roots], 'generator_version': 'corpus_v1'}
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Mined {len(kernels)} kernels from {roots}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return meta


if __name__ == '__main__':
    import sys
    roots = sys.argv[1:] or ['/home/gabe/Documents/gpu-sim']
    mine_corpus(roots, '/tmp/corpus_test', max_kernels=200)

#!/usr/bin/env python3
"""Baseline step-time / MFU profile for the v1 model size.

Profiles training step time before any optimization work, so the project
can confirm the assumption that data generation (not training) is the
bottleneck. CPU-timed tiny-model smoke by default; GPU path when available.
"""

import time
import sys
sys.path.insert(0, '/home/gabe/Documents/gpu-sim')


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_flops_per_step(params: int, seq_len: int, batch: int) -> float:
    """Rough transformer FLOPs/step: ~6 * params * tokens (forward+backward)."""
    return 6.0 * params * seq_len * batch


def profile(n_steps: int = 5, batch: int = 2, seq_len: int = 16) -> dict:
    import torch
    from epic_u2o.model import create_model
    device = 'cpu'
    model = create_model(vocab_size=64, max_seq_len=32, d_model=32,
                         n_heads=2, n_layers=1, d_ff=64,
                         numeric_dim=26, use_numeric=True, dual_head=True)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_data = {
        'input_ids': torch.randint(0, 64, (batch, seq_len)),
        'attention_mask': torch.ones(batch, seq_len, dtype=torch.long),
        'numeric_features': torch.randn(batch, 26),
        'target': torch.zeros(batch, 1),
    }
    # Warmup
    for _ in range(2):
        opt.zero_grad()
        mean, log_var = model(batch_data['input_ids'], batch_data['attention_mask'],
                              batch_data['numeric_features'])
        loss = ((mean - batch_data['target']) ** 2).mean()
        loss.backward()
        opt.step()
    t0 = time.time()
    for _ in range(n_steps):
        opt.zero_grad()
        mean, log_var = model(batch_data['input_ids'], batch_data['attention_mask'],
                              batch_data['numeric_features'])
        loss = ((mean - batch_data['target']) ** 2).mean()
        loss.backward()
        opt.step()
    dt = (time.time() - t0) / n_steps
    params = count_params(model)
    flops = estimate_flops_per_step(params, seq_len, batch)
    # RTX 5090 peak FP32 ~82.6 TFLOP/s; MFU on CPU smoke is ~0 by definition
    mfu_pct = 100.0 * (flops / dt) / 82.6e12 if dt > 0 else 0.0
    return {'device': device, 'params': params, 'batch': batch, 'seq_len': seq_len,
            'step_time_s': dt, 'flops_per_step': flops,
            'mfu_pct_vs_5090_peak': mfu_pct,
            'note': 'CPU smoke; rerun on 5090 for the real baseline'}


if __name__ == '__main__':
    r = profile()
    assert r['step_time_s'] > 0 and r['params'] > 0
    print(f"Profile OK: {r['params']} params step={r['step_time_s']*1000:.1f}ms "
          f"MFU(vs 5090)={r['mfu_pct_vs_5090_peak']:.4f}% on {r['device']}")

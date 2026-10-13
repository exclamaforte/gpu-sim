# Representation Ablation Framework (gpu-sim-u2o.8.4)

## Overview

This document describes the framework for comparing three input representations for the CUDA runtime prediction model:
1. **Source-only**: Lexical CUDA source tokens (tokenizer.py)
2. **PTX-only**: PTX assembly instruction tokens (ptx_tokenizer.py)
3. **Hybrid multi-modal**: Source tokens + PTX tokens + Numeric feature vector (features.py)

All three representations use the same model architecture backbone for fair comparison.

## Dependencies (All Complete)

- ✓ gpu-sim-u2o.8.1: Lexical CUDA source tokenizer with identifier/literal fallback encoding
- ✓ gpu-sim-u2o.8.2: PTX/SASS extraction and tokenizer, including ptxas -v resource usage
- ✓ gpu-sim-u2o.8.3: Numeric launch-config and static-feature vector

## Representation Implementations

### 1. Source-Only (tokenizer.py)
- Tokenizes CUDA C++ source code
- Handles CUDA-specific syntax: `<<< >>>`, `__global__`, `__shared__`, `__device__`
- Built-in variables: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize`
- Identifier/literal fallback encoding for OOV handling
- Output: Token ID sequence (max 2048 tokens)

### 2. PTX-Only (ptx_tokenizer.py)
- Compiles source to PTX via `nvcc -ptx`
- Tokenizes PTX assembly instructions
- Handles type suffixes (`.u32`, `.f32`, `.b64`, etc.)
- Extracts ptxas resource usage: registers/thread, shared mem, spill counts
- Output: Token ID sequence + PTX resource features

### 3. Hybrid Multi-Modal
- Combines source tokens + PTX tokens + numeric feature vector
- Numeric features (26-dim): grid/block dims, shared mem, registers, arg info, memory/compute estimates
- Fusion: Separate encoders for each modality + cross-attention or concatenation

## Experimental Protocol

### Dataset Splits
- Train/Val/Test: 70/15/15 split by kernel hash (no leakage)
- Held-out suites: Elementwise, Reduction, Stencil, GEMM, Convolution
- Same splits used for all three representations

### Model Architecture (Matched Capacity)
```
Source Encoder:     Embedding(256) -> Transformer(4 layers, 4 heads, 256 dim)
PTX Encoder:        Embedding(256) -> Transformer(4 layers, 4 heads, 256 dim)
Numeric Encoder:    MLP(26 -> 128 -> 256)
Fusion:             Concat + MLP(768 -> 256 -> 1)  (or Cross-Attention)
Total params:       ~2-3M per variant
```

### Training
- Optimizer: AdamW, lr=3e-4, weight_decay=0.01
- Scheduler: Cosine with warmup (10% steps)
- Loss: MSE on log(execution_time_ms)
- Batch size: 64
- Epochs: 50 with early stopping (patience=10)
- Seed: 42 (fixed for all runs)

### Evaluation Metrics
- Primary: MAPE (Mean Absolute Percentage Error) on test set
- Secondary: RMSLE, R², P90/P99 latency error
- Per-suite breakdown: Elementwise, Reduction, Stencil, GEMM, Convolution
- Statistical significance: Bootstrap 95% CI (1000 resamples)

### Hardware/Software Pinned (per gpu-sim-u2o.5.2)
- CUDA Toolkit: 12.8
- Driver: 535.104.05
- nvcc/ptxas: From CUDA 12.8
- GPU: RTX 5090, clocks locked

## Ablation Matrix

| Representation | Source Tokens | PTX Tokens | Numeric Features | Expected Strengths |
|---|---|---|---|---|
| Source-only | ✓ | | | Human-readable, portable, captures algorithmic structure |
| PTX-only | | ✓ | ✓ (ptxas) | Hardware-accurate, captures compiler optimizations |
| Hybrid | ✓ | ✓ | ✓ | Best of both, multi-modal fusion |

## Running the Ablation

```bash
# 1. Generate dataset with all representations
python experiments/gpu-sim-u2o.8.4/build_data_84.py  # -> ablation_data.jsonl

# 2. Train all three variants (gpu-sim-u2o.8.4: epic_u2o.train never
# existed; epic_u2o/train_ablation.py is the entrypoint)
python -m epic_u2o.train_ablation --arm source --data experiments/gpu-sim-u2o.8.4/ablation_data.jsonl --out runs/gpu-sim-u2o.8.4/source
python -m epic_u2o.train_ablation --arm ptx --data experiments/gpu-sim-u2o.8.4/ablation_data.jsonl --out runs/gpu-sim-u2o.8.4/ptx
python -m epic_u2o.train_ablation --arm hybrid --data experiments/gpu-sim-u2o.8.4/ablation_data.jsonl --out runs/gpu-sim-u2o.8.4/hybrid

# 3. Evaluate on held-out suites + compare (per-arm arm_metrics.json land
# with the run; compare_84.py renders the verdict vs the >5%/2% criteria)
python experiments/gpu-sim-u2o.8.4/compare_84.py runs/gpu-sim-u2o.8.4
```

## Expected Outcomes

Based on prior work (D_REPR design note):
- **Source-only**: Good for algorithmic patterns, struggles with compiler effects
- **PTX-only**: Strong on hardware-specific optimization, but loses high-level structure
- **Hybrid**: Expected to outperform both by combining algorithmic + hardware info

## Decision Criteria

Choose representation for production model if:
1. Hybrid beats best single-modality by >5% MAPE reduction (statistically significant)
2. Or single modality is within 2% of hybrid (simpler deployment)

## Tracking

Each ablation run creates a bead:
```bash
bd create --title="Ablation: source-only v1" --tags="ablation,representation"
bd create --title="Ablation: PTX-only v1" --tags="ablation,representation"
bd create --title="Ablation: hybrid v1" --tags="ablation,representation"
```

## Status

Framework complete. Ready for experimental execution when dataset is available (depends on gpu-sim-u2o.7.1 generator).
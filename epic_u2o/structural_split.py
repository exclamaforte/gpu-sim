#!/usr/bin/env python3
"""Structural-family metadata and train/test split design.

Tags every generated example with its structural family/template so
train/test splits can hold out whole families and problem-size ranges.
"""

import json
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
from collections import defaultdict


@dataclass
class StructuralFamily:
    """Defines a structural family of kernels."""
    family_id: str
    template_name: str
    generator: str
    key_params: dict  # Parameters that define the family
    
    def matches(self, metadata: dict) -> bool:
        """Check if a kernel's metadata matches this family."""
        if metadata.get('generator') != self.generator:
            return False
        if metadata.get('op_type') != self.template_name:
            return False
        for key, value in self.key_params.items():
            if metadata.get(key) != value:
                return False
        return True


# Define structural families for each stage
STAGE1_FAMILIES = [
    StructuralFamily(
        family_id="stage1_scale",
        template_name="scale",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_add",
        template_name="add",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_saxpy",
        template_name="saxpy",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_relu",
        template_name="relu",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_mul",
        template_name="mul",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_div",
        template_name="div",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_copy",
        template_name="copy",
        generator="stage1_elementwise",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage1_triad",
        template_name="triad",
        generator="stage1_elementwise",
        key_params={},
    ),
]

STAGE2_FAMILIES = [
    StructuralFamily(
        family_id="stage2_reduce_sum_block",
        template_name="reduce_sum_block",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_reduce_sum_warp_shuffle",
        template_name="reduce_sum_warp_shuffle",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_reduce_max_block",
        template_name="reduce_max_block",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_reduce_min_block",
        template_name="reduce_min_block",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_shared_mem_transpose",
        template_name="shared_mem_transpose",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_shared_mem_bank_conflict_1way",
        template_name="shared_mem_bank_conflict_1way",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_shared_mem_bank_conflict_2way",
        template_name="shared_mem_bank_conflict_2way",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_shared_mem_bank_conflict_32way",
        template_name="shared_mem_bank_conflict_32way",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_stencil_2d_5pt",
        template_name="stencil_2d_5pt",
        generator="stage2_reduction_shared",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage2_shared_mem_reduction_tiled",
        template_name="shared_mem_reduction_tiled",
        generator="stage2_reduction_shared",
        key_params={},
    ),
]

STAGE3_FAMILIES = [
    StructuralFamily(
        family_id="stage3_coalesced_read",
        template_name="coalesced_read",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_coalesced_write",
        template_name="coalesced_write",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_strided_read",
        template_name="strided_read",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_strided_write",
        template_name="strided_write",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_strided_read_write",
        template_name="strided_read_write",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_segmented_read",
        template_name="segmented_read",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_interleaved_read",
        template_name="interleaved_read",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_gather_read",
        template_name="gather_read",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_scatter_write",
        template_name="scatter_write",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_random_access",
        template_name="random_access",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_transpose_2d",
        template_name="transpose_2d",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_diagonal_access",
        template_name="diagonal_access",
        generator="stage3_memory_access",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage3_block_cyclic",
        template_name="block_cyclic",
        generator="stage3_memory_access",
        key_params={},
    ),
]

STAGE4_FAMILIES = [
    StructuralFamily(
        family_id="stage4_gemm_naive",
        template_name="gemm_naive",
        generator="stage4_gemm",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage4_gemm_tiled",
        template_name="gemm_tiled",
        generator="stage4_gemm",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage4_gemm_wmma_fp16",
        template_name="gemm_wmma_fp16",
        generator="stage4_gemm",
        key_params={},
    ),
]

STAGE5_FAMILIES = [
    StructuralFamily(
        family_id="stage5_divergent_branch",
        template_name="divergent_branch",
        generator="stage5_divergence_atomics",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage5_divergent_switch",
        template_name="divergent_switch",
        generator="stage5_divergence_atomics",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage5_divergent_loop",
        template_name="divergent_loop",
        generator="stage5_divergence_atomics",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage5_histogram_global",
        template_name="histogram_global",
        generator="stage5_divergence_atomics",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage5_histogram_shared",
        template_name="histogram_shared",
        generator="stage5_divergence_atomics",
        key_params={},
    ),
    StructuralFamily(
        family_id="stage5_atomic_counter",
        template_name="atomic_counter",
        generator="stage5_divergence_atomics",
        key_params={},
    ),
]

ALL_FAMILIES = STAGE1_FAMILIES + STAGE2_FAMILIES + STAGE3_FAMILIES + STAGE4_FAMILIES + STAGE5_FAMILIES


def assign_family(metadata: dict, op_type: str = None) -> Optional[str]:
    """Assign a structural family ID to a kernel based on its metadata."""
    for family in ALL_FAMILIES:
        if family.matches(metadata):
            return family.family_id
    # Fallback: match by op_type if provided
    if op_type:
        for family in ALL_FAMILIES:
            if family.template_name == op_type and family.generator == metadata.get('generator'):
                return family.family_id
    return None


def compute_kernel_signature(source: str) -> str:
    """Compute a structural signature from kernel source (template + key params)."""
    # Normalize source: remove variable names, keep structure
    normalized = ' '.join(source.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def split_by_family(
    kernels: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    holdout_families: list[str] = None,
    holdout_problem_sizes: list[tuple[int, int]] = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split kernels by structural family to prevent template leakage.
    
    Args:
        kernels: List of kernel dicts with metadata
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        holdout_families: Family IDs to exclude from training (put in test)
        holdout_problem_sizes: (min, max) ranges to exclude from training
        seed: Random seed
    
    Returns:
        Dict with 'train', 'val', 'test' lists
    """
    random.seed(seed)
    holdout_families = set(holdout_families or [])
    
    # Group kernels by structural family
    family_groups = defaultdict(list)
    for kernel in kernels:
        family_id = kernel.get('metadata', {}).get('family_id')
        if not family_id:
            family_id = assign_family(kernel.get('metadata', {}), kernel.get('op_type'))
        family_groups[family_id].append(kernel)
    
    train, val, test = [], [], []
    
    for family_id, group in family_groups.items():
        # Assign family to split
        if family_id in holdout_families:
            # Entire family goes to test
            test.extend(group)
        else:
            # Split families randomly (not individual kernels)
            r = random.random()
            if r < train_ratio:
                train.extend(group)
            elif r < train_ratio + val_ratio:
                val.extend(group)
            else:
                test.extend(group)
    
    # Additional problem-size holdout
    if holdout_problem_sizes:
        new_train = []
        for k in train:
            ps = k.get('problem_size', 0)
            holdout = any(lo <= ps <= hi for lo, hi in holdout_problem_sizes)
            if holdout:
                test.append(k)
            else:
                new_train.append(k)
        train = new_train
    
    return {
        'train': train,
        'val': val,
        'test': test,
    }


def create_generalization_splits(
    kernels: list[dict],
    seed: int = 42,
) -> dict[str, dict[str, list[dict]]]:
    """Create multiple generalization test splits.
    
    Returns splits for:
    - unseen_families: Entire families held out
    - unseen_problem_sizes: Large problem sizes held out
    - unseen_occupancy: Low/high occupancy regimes held out
    """
    random.seed(seed)
    
    # Family-based holdout (20% of families)
    families = list(set(k.get('metadata', {}).get('family_id') or 
                        assign_family(k.get('metadata', {})) for k in kernels))
    families = [f for f in families if f]
    holdout_count = max(1, len(families) // 5)
    holdout_families = random.sample(families, holdout_count)
    
    # Problem size holdout (top 20% by size)
    problem_sizes = sorted(set(k.get('problem_size', 0) for k in kernels))
    holdout_size_threshold = problem_sizes[int(len(problem_sizes) * 0.8)]
    
    # Occupancy holdout (very low and very high)
    # Would need occupancy info in metadata
    
    splits = {
        'unseen_families': split_by_family(kernels, holdout_families=holdout_families, seed=seed),
        'unseen_problem_sizes': split_by_family(
            kernels, holdout_problem_sizes=[(holdout_size_threshold, float('inf'))], seed=seed
        ),
        'random': split_by_family(kernels, seed=seed),
    }
    
    return splits


def analyze_split_quality(splits: dict[str, list[dict]]) -> dict:
    """Analyze quality of a train/val/test split."""
    train_families = set()
    val_families = set()
    test_families = set()
    
    for k in splits.get('train', []):
        fid = k.get('metadata', {}).get('family_id') or assign_family(k.get('metadata', {}), k.get('op_type'))
        if fid:
            train_families.add(fid)
    
    for k in splits.get('val', []):
        fid = k.get('metadata', {}).get('family_id') or assign_family(k.get('metadata', {}), k.get('op_type'))
        if fid:
            val_families.add(fid)
    
    for k in splits.get('test', []):
        fid = k.get('metadata', {}).get('family_id') or assign_family(k.get('metadata', {}), k.get('op_type'))
        if fid:
            test_families.add(fid)
    
    # Check for leakage
    leakage_train_test = train_families & test_families
    leakage_train_val = train_families & val_families
    leakage_val_test = val_families & test_families
    
    return {
        'train_families': len(train_families),
        'val_families': len(val_families),
        'test_families': len(test_families),
        'total_families': len(train_families | val_families | test_families),
        'leakage_train_test': len(leakage_train_test),
        'leakage_train_val': len(leakage_train_val),
        'leakage_val_test': len(leakage_val_test),
        'leakage_details': {
            'train_test': list(leakage_train_test),
            'train_val': list(leakage_train_val),
            'val_test': list(leakage_val_test),
        },
        'train_examples': len(splits.get('train', [])),
        'val_examples': len(splits.get('val', [])),
        'test_examples': len(splits.get('test', [])),
    }


def enrich_kernels_with_families(kernels: list[dict]) -> list[dict]:
    """Add family_id and structural_signature to kernel metadata."""
    enriched = []
    for k in kernels:
        metadata = k.get('metadata', {}).copy()
        metadata['family_id'] = assign_family(metadata, k.get('op_type'))
        metadata['structural_signature'] = compute_kernel_signature(k.get('source', ''))
        k = k.copy()
        k['metadata'] = metadata
        enriched.append(k)
    return enriched


if __name__ == '__main__':
    # Demo: Load a dataset and analyze splits
    import sys
    
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        dataset_dir = Path('runs/stage1_dataset')
    
    # Load kernels
    kernels = []
    for split_file in dataset_dir.glob('*.jsonl'):
        with open(split_file) as f:
            for line in f:
                kernels.append(json.loads(line))
    
    print(f"Loaded {len(kernels)} kernels")
    
    # Enrich with families
    kernels = enrich_kernels_with_families(kernels)
    
    # Create splits
    splits = split_by_family(kernels)
    
    # Analyze
    quality = analyze_split_quality(splits)
    print(f"\nSplit Quality:")
    print(f"  Train: {quality['train_examples']} examples, {quality['train_families']} families")
    print(f"  Val: {quality['val_examples']} examples, {quality['val_families']} families")
    print(f"  Test: {quality['test_examples']} examples, {quality['test_families']} families")
    print(f"  Total families: {quality['total_families']}")
    print(f"  Leakage train-test: {quality['leakage_train_test']}")
    print(f"  Leakage train-val: {quality['leakage_train_val']}")
    print(f"  Leakage val-test: {quality['leakage_val_test']}")
    
    # Generalization splits
    gen_splits = create_generalization_splits(kernels)
    print(f"\nGeneralization Splits:")
    for name, split in gen_splits.items():
        q = analyze_split_quality(split)
        print(f"  {name}: train={q['train_examples']}, test={q['test_examples']}, "
              f"families_train={q['train_families']}, families_test={q['test_families']}, "
              f"leakage={q['leakage_train_test']}")
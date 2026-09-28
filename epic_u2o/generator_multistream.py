#!/usr/bin/env python3
"""Multi-stream / concurrent-kernel program generator (POST-V1 STUB).

v1 predicts single kernel-launch runtime only (see epic_u2o/v1_scope.md:
multi-stream, cudaGraph, cooperative multi-device are REJECTED for v1).
This module exists so whole-program prediction (deferred epic J1) has a
placeholder; it must NOT feed the v1 dataset pipeline.
"""

POST_V1 = True

MULTISTREAM_EXAMPLE = '''
// POST-V1 ONLY: overlapped streams + concurrent kernels (rejected by v1_scope).
cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);
map_kernel<<<4, 256, 0, s1>>>(d_a, d_out, 1024);
reduce_sum_block<<<4, 256, 256 * sizeof(float), s2>>>(d_out, d_partial, 1024);
cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);
cudaStreamDestroy(s1); cudaStreamDestroy(s2);
'''.strip()


def generate_post_v1_example():
    """Return a single documented post-v1 multi-stream example (no v1 dataset use)."""
    return {
        'kernel_name': 'multistream_example',
        'source': MULTISTREAM_EXAMPLE,
        'streams': ['s1', 's2'],
        'post_v1': True,
        'v1_eligible': False,
        'metadata': {
            'generator': 'stage_multistream_postv1',
            'op_type': 'multistream_concurrent',
            'family_id': 'postv1_multistream',
        },
    }


if __name__ == '__main__':
    ex = generate_post_v1_example()
    assert ex['post_v1'] and not ex['v1_eligible']
    assert 'cudaStreamCreate' in ex['source'] and '<<<' in ex['source']
    print(f"Post-V1 stub OK: {ex['kernel_name']} streams={ex['streams']}")

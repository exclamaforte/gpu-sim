"""Numeric launch-config and static-feature vector extraction for CUDA kernels.

Assembles non-sequence feature vector: grid/block dims, shared-mem bytes,
register count, argument sizes/types, estimated bytes moved, and estimated
FLOPs (from static analysis of the kernel body).
"""

import re
import math
from dataclasses import dataclass, asdict
from typing import Optional
from tokenizer import tokenize, TokenType


@dataclass
class LaunchConfig:
    """Kernel launch configuration."""
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 1
    block_y: int = 1
    block_z: int = 1
    shared_mem_bytes: int = 0
    stream: int = 0  # 0 = default stream
    
    @property
    def total_threads(self) -> int:
        return (self.grid_x * self.grid_y * self.grid_z *
                self.block_x * self.block_y * self.block_z)
    
    @property
    def threads_per_block(self) -> int:
        return self.block_x * self.block_y * self.block_z
    
    @property
    def num_blocks(self) -> int:
        return self.grid_x * self.grid_y * self.grid_z


@dataclass
class KernelFeatures:
    """Static features extracted from kernel source."""
    # Launch config
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    shared_mem_bytes: int
    
    # Register usage (estimated)
    estimated_registers_per_thread: int
    
    # Argument info
    num_args: int
    num_pointer_args: int
    num_scalar_args: int
    total_arg_bytes: int
    
    # Memory access estimates
    estimated_global_reads: int
    estimated_global_writes: int
    estimated_shared_reads: int
    estimated_shared_writes: int
    estimated_bytes_moved: int
    
    # Compute estimates
    estimated_flops: int
    estimated_int_ops: int
    
    # Control flow
    has_loops: bool
    has_conditionals: bool
    max_loop_depth: int
    
    # Synchronization
    has_syncthreads: bool
    num_syncthreads: int
    
    # Kernel properties
    uses_shared_memory: bool
    uses_constant_memory: bool
    uses_vector_types: bool
    
    def to_vector(self) -> list[float]:
        """Convert to flat feature vector for model input."""
        return [
            float(self.grid_x), float(self.grid_y), float(self.grid_z),
            float(self.block_x), float(self.block_y), float(self.block_z),
            float(self.shared_mem_bytes),
            float(self.estimated_registers_per_thread),
            float(self.num_args), float(self.num_pointer_args), float(self.num_scalar_args),
            float(self.total_arg_bytes),
            float(self.estimated_global_reads), float(self.estimated_global_writes),
            float(self.estimated_shared_reads), float(self.estimated_shared_writes),
            float(self.estimated_bytes_moved),
            float(self.estimated_flops), float(self.estimated_int_ops),
            float(self.has_loops), float(self.has_conditionals), float(self.max_loop_depth),
            float(self.has_syncthreads), float(self.num_syncthreads),
            float(self.uses_shared_memory), float(self.uses_constant_memory), float(self.uses_vector_types),
        ]
    
    @property
    def vector_dim(self) -> int:
        return 26
    
    def to_dict(self) -> dict:
        return asdict(self)


# RTX 5090 specs for normalization. gpu-sim-g8x: SINGLE-SOURCED from
# epic_u2o.hw_5090 -- this dict used to carry its own copies (228KB
# shared-mem, 1792 theory BW, unsourced 82.6), which is how the baseline
# came to normalise against a part that does not exist. One definition,
# imported, not two copies. NOTE the rescale this implies: normalization
# denominators changed, so normalized vectors from before this commit are
# not comparable to ones from after it.
from epic_u2o import hw_5090 as _hw

RTX_5090_SPECS = {
    'max_threads_per_block': _hw.MAX_THREADS_PER_BLOCK,  # 1024, own probe
    'max_shared_mem_per_block': _hw.SHARED_MEM_PER_BLOCK,  # 49152, own probe (was 228*1024: per-SM value in a per-block slot)
    'max_registers_per_thread': _hw.MAX_REGISTERS_PER_THREAD,  # ARCH 255
    'max_blocks_per_sm': _hw.MAX_BLOCKS_PER_SM,  # ARCH 32
    'num_sms': _hw.SM_COUNT,  # 170, megakern-37e.2 + own probe
    'warp_size': _hw.WARP_SIZE,  # ARCH 32
    'peak_fp32_tflops': _hw.PEAK_FP32_TFLOPS,  # VENDOR, UNSOURCED -- same number, honest label
    'memory_bandwidth_gbps': _hw.MEASURED_READ_BW_GBS,  # 1688.592 achievable, megakern-37e.2 (was 1792 theory)
}


def normalize_feature(value: float, max_val: float) -> float:
    """Normalize feature to [0, 1] range with log scaling for large values."""
    if max_val <= 0:
        return 0.0
    ratio = value / max_val
    if ratio > 1.0:
        # Log scale for values exceeding max
        return min(1.0, 1.0 + 0.1 * math.log10(ratio))
    return ratio


def extract_launch_config(source: str) -> LaunchConfig:
    """Extract launch configuration from kernel launch site (<<< >>>)."""
    config = LaunchConfig()
    
    # Find <<< ... >>>
    launch_match = re.search(r'<<<\s*([^>]+)\s*>>>', source)
    if not launch_match:
        return config
    
    launch_args = launch_match.group(1).strip()
    parts = [p.strip() for p in launch_args.split(',')]
    
    # Parse grid dims
    if len(parts) >= 1:
        grid_part = parts[0]
        if '(' in grid_part or ')' in grid_part:
            # dim3 grid(...)
            dims = re.findall(r'\d+', grid_part)
            if len(dims) >= 1: config.grid_x = int(dims[0])
            if len(dims) >= 2: config.grid_y = int(dims[1])
            if len(dims) >= 3: config.grid_z = int(dims[2])
        else:
            config.grid_x = int(grid_part)
    
    # Parse block dims
    if len(parts) >= 2:
        block_part = parts[1]
        if '(' in block_part or ')' in block_part:
            dims = re.findall(r'\d+', block_part)
            if len(dims) >= 1: config.block_x = int(dims[0])
            if len(dims) >= 2: config.block_y = int(dims[1])
            if len(dims) >= 3: config.block_z = int(dims[2])
        else:
            config.block_x = int(block_part)
    
    # Parse shared memory
    if len(parts) >= 3:
        config.shared_mem_bytes = int(parts[2])
    
    # Parse stream (4th argument)
    if len(parts) >= 4:
        config.stream = int(parts[3])
    
    return config


def estimate_registers(source: str) -> int:
    """Estimate register usage per thread from kernel source."""
    # Heuristic: count variable declarations, live ranges
    # This is a rough estimate; real register count needs nvcc --ptxas-options=-v
    tokens = tokenize(source)
    
    var_count = 0
    in_kernel_body = False
    brace_depth = 0
    
    for token in tokens:
        if token.type == TokenType.GLOBAL:
            in_kernel_body = True
            continue
        if in_kernel_body:
            if token.type == TokenType.LBRACE:
                brace_depth += 1
            elif token.type == TokenType.RBRACE:
                brace_depth -= 1
                if brace_depth == 0:
                    break
            elif token.type in (TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, TokenType.CHAR):
                # Variable declaration
                var_count += 1
            elif token.type == TokenType.IDENTIFIER and token.value in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim'):
                # Built-ins typically don't use registers
                pass
    
    # Add base registers for loop indices, temporaries
    base_regs = 8
    loop_regs = source.count('for') * 2 + source.count('while') * 2
    return min(255, base_regs + var_count + loop_regs)


def analyze_kernel_body(source: str) -> dict:
    """Static analysis of kernel body for feature extraction."""
    features = {
        'estimated_global_reads': 0,
        'estimated_global_writes': 0,
        'estimated_shared_reads': 0,
        'estimated_shared_writes': 0,
        'estimated_flops': 0,
        'estimated_int_ops': 0,
        'has_loops': False,
        'has_conditionals': False,
        'max_loop_depth': 0,
        'has_syncthreads': False,
        'num_syncthreads': 0,
        'uses_shared_memory': False,
        'uses_constant_memory': False,
        'uses_vector_types': False,
    }
    
    tokens = tokenize(source)
    in_kernel = False
    brace_depth = 0
    loop_depth = 0
    max_loop_depth = 0
    
    for i, token in enumerate(tokens):
        if token.type == TokenType.GLOBAL:
            in_kernel = True
            continue
        
        if not in_kernel:
            continue
        
        if token.type == TokenType.LBRACE:
            brace_depth += 1
        elif token.type == TokenType.RBRACE:
            brace_depth -= 1
            if brace_depth == 0:
                break
        
        # Loop detection
        if token.type in (TokenType.FOR, TokenType.WHILE, TokenType.DO):
            features['has_loops'] = True
            loop_depth += 1
            max_loop_depth = max(max_loop_depth, loop_depth)
        elif token.type == TokenType.RBRACE and loop_depth > 0:
            loop_depth = max(0, loop_depth - 1)
        
        # Conditional detection
        if token.type == TokenType.IF:
            features['has_conditionals'] = True
        
        # __syncthreads()
        if token.type == TokenType.IDENTIFIER and token.value == '__syncthreads':
            features['has_syncthreads'] = True
            features['num_syncthreads'] += 1
        
        # Memory access patterns
        if token.type == TokenType.IDENTIFIER:
            # Check for shared memory usage
            if token.value == '__shared__':
                features['uses_shared_memory'] = True
            # Check for vector types
            if token.value in ('float2', 'float3', 'float4', 'double2', 'int2', 'int3', 'int4', 'uint2', 'uint3', 'uint4'):
                features['uses_vector_types'] = True
        
        # Estimate FLOPs from operators
        if token.type in (TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH):
            # Check if operands are likely floats
            features['estimated_flops'] += 1
        elif token.type in (TokenType.PERCENT, TokenType.AMPERSAND, TokenType.PIPE, TokenType.CARET,
                           TokenType.LEFT_SHIFT, TokenType.RIGHT_SHIFT):
            features['estimated_int_ops'] += 1
        
        # Transcendental functions
        if token.type == TokenType.IDENTIFIER and token.value in (
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
            'exp', 'log', 'log10', 'sqrt', 'rsqrt', 'pow', 'floor', 'ceil', 'round',
            'sinf', 'cosf', 'tanf', 'expf', 'logf', 'sqrtf', 'rsqrtf', 'powf'
        ):
            features['estimated_flops'] += 4  # Transcendentals ~4-8 FLOPs
        
        # FMA pattern (a*b+c)
        if token.type == TokenType.STAR and i + 2 < len(tokens):
            next_token = tokens[i + 1]
            next_next = tokens[i + 2]
            if next_token.type == TokenType.PLUS:
                features['estimated_flops'] += 1  # FMA counts as 2 FLOPs
    
    features['max_loop_depth'] = max_loop_depth
    
    # Scale estimates by thread count
    # (Will be multiplied by actual thread count in feature extraction)
    
    return features


def extract_kernel_args(source: str) -> dict:
    """Extract kernel argument information."""
    tokens = tokenize(source)
    
    num_args = 0
    num_pointer_args = 0
    num_scalar_args = 0
    total_arg_bytes = 0
    in_kernel_params = False
    paren_depth = 0
    
    for token in tokens:
        if token.type == TokenType.GLOBAL:
            in_kernel_params = True
            continue
        
        if in_kernel_params:
            if token.type == TokenType.LPAREN:
                paren_depth += 1
            elif token.type == TokenType.RPAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    break
            elif token.type == TokenType.COMMA:
                num_args += 1
            elif token.type in (TokenType.STAR,):
                # Pointer argument
                num_pointer_args += 1
                total_arg_bytes += 8  # 64-bit pointer
            elif token.type in (TokenType.INT, TokenType.FLOAT, TokenType.DOUBLE, TokenType.CHAR,
                               TokenType.SHORT, TokenType.LONG):
                num_scalar_args += 1
                if token.type == TokenType.DOUBLE:
                    total_arg_bytes += 8
                elif token.type in (TokenType.FLOAT, TokenType.INT):
                    total_arg_bytes += 4
                elif token.type == TokenType.SHORT:
                    total_arg_bytes += 2
                elif token.type == TokenType.CHAR:
                    total_arg_bytes += 1
                else:
                    total_arg_bytes += 8
    
    # Count last argument
    if num_args > 0 or num_pointer_args > 0 or num_scalar_args > 0:
        num_args = max(1, num_args + 1)
    
    return {
        'num_args': num_args,
        'num_pointer_args': num_pointer_args,
        'num_scalar_args': num_scalar_args,
        'total_arg_bytes': total_arg_bytes,
    }


def extract_features(source: str, launch_config: Optional[LaunchConfig] = None) -> KernelFeatures:
    """Extract complete feature vector from kernel source and launch config."""
    if launch_config is None:
        launch_config = extract_launch_config(source)
    
    # Kernel body analysis
    body_features = analyze_kernel_body(source)
    
    # Argument analysis
    arg_features = extract_kernel_args(source)
    
    # Register estimation
    estimated_regs = estimate_registers(source)
    
    # Scale memory/compute estimates by thread count
    threads_per_block = launch_config.threads_per_block
    num_blocks = launch_config.num_blocks
    total_threads = launch_config.total_threads
    
    estimated_bytes_moved = (
        body_features['estimated_global_reads'] * 4 +
        body_features['estimated_global_writes'] * 4 +
        body_features['estimated_shared_reads'] * 4 +
        body_features['estimated_shared_writes'] * 4
    ) * total_threads
    
    return KernelFeatures(
        grid_x=launch_config.grid_x,
        grid_y=launch_config.grid_y,
        grid_z=launch_config.grid_z,
        block_x=launch_config.block_x,
        block_y=launch_config.block_y,
        block_z=launch_config.block_z,
        shared_mem_bytes=launch_config.shared_mem_bytes,
        estimated_registers_per_thread=estimated_regs,
        num_args=arg_features['num_args'],
        num_pointer_args=arg_features['num_pointer_args'],
        num_scalar_args=arg_features['num_scalar_args'],
        total_arg_bytes=arg_features['total_arg_bytes'],
        estimated_global_reads=body_features['estimated_global_reads'] * total_threads,
        estimated_global_writes=body_features['estimated_global_writes'] * total_threads,
        estimated_shared_reads=body_features['estimated_shared_reads'] * total_threads,
        estimated_shared_writes=body_features['estimated_shared_writes'] * total_threads,
        estimated_bytes_moved=estimated_bytes_moved,
        estimated_flops=body_features['estimated_flops'] * total_threads,
        estimated_int_ops=body_features['estimated_int_ops'] * total_threads,
        has_loops=body_features['has_loops'],
        has_conditionals=body_features['has_conditionals'],
        max_loop_depth=body_features['max_loop_depth'],
        has_syncthreads=body_features['has_syncthreads'],
        num_syncthreads=body_features['num_syncthreads'],
        uses_shared_memory=body_features['uses_shared_memory'],
        uses_constant_memory=body_features['uses_constant_memory'],
        uses_vector_types=body_features['uses_vector_types'],
    )


def extract_normalized_features(source: str, launch_config: Optional[LaunchConfig] = None) -> list[float]:
    """Extract and normalize features for model input."""
    features = extract_features(source, launch_config)
    vec = features.to_vector()
    
    # Normalize using RTX 5090 specs
    normalized = []
    specs = RTX_5090_SPECS
    
    # Grid/block dims
    normalized.append(normalize_feature(vec[0], specs['num_sms'] * 4))  # grid_x
    normalized.append(normalize_feature(vec[1], 65535))  # grid_y
    normalized.append(normalize_feature(vec[2], 65535))  # grid_z
    normalized.append(normalize_feature(vec[3], specs['max_threads_per_block']))  # block_x
    normalized.append(normalize_feature(vec[4], specs['max_threads_per_block']))  # block_y
    normalized.append(normalize_feature(vec[5], 64))  # block_z
    normalized.append(normalize_feature(vec[6], specs['max_shared_mem_per_block']))  # shared_mem
    normalized.append(normalize_feature(vec[7], specs['max_registers_per_thread']))  # registers
    
    # Args (normalize loosely)
    normalized.append(normalize_feature(vec[8], 20))  # num_args
    normalized.append(normalize_feature(vec[9], 10))  # num_pointer_args
    normalized.append(normalize_feature(vec[10], 10))  # num_scalar_args
    normalized.append(normalize_feature(vec[11], 1024))  # total_arg_bytes
    
    # Memory (normalize by total threads * 4 bytes)
    max_bytes = specs['num_sms'] * specs['max_threads_per_block'] * 4 * 1024
    normalized.append(normalize_feature(vec[12], max_bytes))  # global_reads
    normalized.append(normalize_feature(vec[13], max_bytes))  # global_writes
    normalized.append(normalize_feature(vec[14], max_bytes))  # shared_reads
    normalized.append(normalize_feature(vec[15], max_bytes))  # shared_writes
    normalized.append(normalize_feature(vec[16], max_bytes))  # bytes_moved
    
    # Compute (normalize by peak TFLOPs * threads)
    max_flops = specs['peak_fp32_tflops'] * 1e12 / 1e9 * specs['num_sms'] * specs['max_threads_per_block'] / 32
    normalized.append(normalize_feature(vec[17], max_flops))  # flops
    normalized.append(normalize_feature(vec[18], max_flops))  # int_ops
    
    # Boolean/int features (already 0/1)
    normalized.extend(vec[19:])  # has_loops, has_conditionals, max_loop_depth, has_syncthreads, num_syncthreads, uses_shared, uses_constant, uses_vector
    
    return normalized


if __name__ == '__main__':
    test_source = '''
__global__ void map_kernel(float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * 2.0f;
}

int main() {
    float *d_a, *d_out;
    cudaMalloc(&d_a, 1024 * sizeof(float));
    cudaMalloc(&d_out, 1024 * sizeof(float));
    map_kernel<<<4, 256>>>(d_a, d_out, 1024);
    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
'''
    launch = extract_launch_config(test_source)
    print(f"Launch config: {launch}")
    print(f"Total threads: {launch.total_threads}")
    
    features = extract_features(test_source, launch)
    print(f"\nFeatures: {features.to_dict()}")
    print(f"\nVector dim: {features.vector_dim}")
    print(f"Vector: {features.to_vector()}")
    
    norm = extract_normalized_features(test_source, launch)
    print(f"\nNormalized: {norm}")
    print(f"Length: {len(norm)}")
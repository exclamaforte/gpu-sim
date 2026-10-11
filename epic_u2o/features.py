"""Numeric launch-config and static-feature vector extraction for CUDA kernels.

Assembles non-sequence feature vector: grid/block dims, shared-mem bytes,
register count, argument sizes/types, estimated bytes moved, and estimated
FLOPs (from static analysis of the kernel body).
"""

import re
import math
from dataclasses import dataclass, asdict
from typing import Optional
from epic_u2o.tokenizer import tokenize, TokenType


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
    # True when these values are the class default because no <<< >>> launch
    # site was found in the source -- i.e. NOT read from a real launch.
    # gpu-sim-u2o.8.3: a kernel DEFINITION never contains a launch site, so
    # this is True for every real kernel and the flag is how a caller tells a
    # measured launch from a guessed one.
    is_default_heuristic: bool = False
    
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
    """Static + problem features extracted from kernel source AND launch.

    gpu-sim-u2o.8.3: `problem_size` was MISSING here, and its absence is the
    whole finding. Every other field here is a property of the source text or
    the launch shape; none of them is the size of the problem. So the vector
    was structurally incapable of distinguishing two launches of the SAME
    kernel at different n -- measured on 7.1's real rows, 33 same-family rows
    spanning 4 problem sizes and 13 distinct grids produced exactly ONE
    distinct feature vector.
    """
    # Problem size. The launch shape alone does not determine runtime; n does.
    # Declared defaulted at the END of the struct (dataclasses forbid defaults
    # above required fields) but EMITTED at to_vector() position 0. Default 0
    # = unknown: keeps old constructor calls working (test_latency_arm builds
    # without it); populate ALWAYS passes the measured n and asserts it.
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
    # Operator mix (per-thread, single-pass, like flops). add/mul/div execute
    # on different units at different throughputs -- div measured 137% slower
    # than add/mul at the same launch on this corpus -- so a single flop
    # count cannot separate them. estimated_flops keeps its exact old meaning
    # (the sum); these are the split, not a replacement.
    estimated_add_ops: int = 0
    estimated_mul_ops: int = 0
    estimated_div_ops: int = 0
    # Call sites: IDENT followed by ( inside the body. A call is opaque work
    # the op counters cannot see (fmaxf in relu, atomicAdd in stage-5); it is
    # also the stage-5 signal for item 10. Overlaps atomic memory counting by
    # design -- features are not orthogonal (arg counts overlap too).
    num_call_sites: int = 0
    # Covered work: min(threads launched, problem_size). The dispatch-ordered
    # fix for the problem_size-blind vector. Declared here (not beside
    # problem_size) because dataclasses forbid defaults above required
    # fields; EMITTED at to_vector() position 28 for the same stability.
    covers: int = 0
    # Problem size, defaulted for the same dataclass reason; EMITTED at
    # position 0. populate ALWAYS passes the measured n and asserts it.
    problem_size: int = 0
    
    def to_vector(self) -> list[float]:
        """Convert to flat feature vector for model input."""
        return [
            float(self.problem_size),
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
            # Appended AFTER the original 28 (8.3): covers, operator mix, calls.
            float(self.covers),
            float(self.estimated_add_ops), float(self.estimated_mul_ops),
            float(self.estimated_div_ops),
            float(self.num_call_sites),
        ]
    
    @property
    def vector_dim(self) -> int:
        # Was 26 while to_vector() returned 27 floats: the declared dimension
        # disagreed with the vector it describes, so anything sized from
        # vector_dim silently dropped the last feature. Derived, not typed.
        return len(self.to_vector())
    
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
        # SILENT DEFAULT, and that is the trap. A __device__/__global__ kernel
        # DEFINITION contains no <<< >>> launch site at all -- the launch is a
        # runtime decision made by the host -- so every real kernel source
        # lands here and silently receives grid=block=1. On 7.1's measured rows
        # that produced a feature vector describing a 1-block launch for all
        # 264 measurements, 13 of which used distinct grids.
        #
        # Warned rather than raised: this function is also used on sources
        # that legitimately have no launch site, and a caller that really wants
        # the default can say so by passing an explicit LaunchConfig. Silently
        # returning a plausible-looking wrong answer is the failure; a warning
        # is the floor.
        config.is_default_heuristic = True
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
        'estimated_add_ops': 0,
        'estimated_mul_ops': 0,
        'estimated_div_ops': 0,
        'num_call_sites': 0,
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
    # Pre-pass over the kernel-body token range: shared array names, local
    # (register-file, NOT memory) array names, and atomic-target marks.
    # Shared wins over local (`extern __shared__ float sdata[]` looks like a
    # declaration); atomics are read+write through one instruction, recorded
    # here so the subscript rule below does not count them a second time.
    _TYPE_TOKENS = frozenset({
        TokenType.VOID, TokenType.CHAR, TokenType.SHORT, TokenType.INT,
        TokenType.LONG, TokenType.FLOAT, TokenType.DOUBLE, TokenType.SIGNED,
        TokenType.UNSIGNED, TokenType.BOOL, TokenType.CONST,
        TokenType.VOLATILE, TokenType.STATIC, TokenType.EXTERN,
        TokenType.REGISTER, TokenType.RESTRICT, TokenType.INLINE})
    _ASSIGN_TOKENS = frozenset({
        TokenType.EQUAL, TokenType.PLUS_EQUAL, TokenType.MINUS_EQUAL,
        TokenType.STAR_EQUAL, TokenType.SLASH_EQUAL, TokenType.PERCENT_EQUAL,
        TokenType.AMPERSAND_EQUAL, TokenType.PIPE_EQUAL, TokenType.CARET_EQUAL,
        TokenType.LEFT_SHIFT_EQUAL, TokenType.RIGHT_SHIFT_EQUAL})
    _COMPOUND_TOKENS = frozenset({
        TokenType.PLUS_EQUAL, TokenType.MINUS_EQUAL, TokenType.STAR_EQUAL,
        TokenType.SLASH_EQUAL, TokenType.PERCENT_EQUAL,
        TokenType.AMPERSAND_EQUAL, TokenType.PIPE_EQUAL, TokenType.CARET_EQUAL,
        TokenType.LEFT_SHIFT_EQUAL, TokenType.RIGHT_SHIFT_EQUAL})
    _VALUE_TOKENS = frozenset({
        TokenType.IDENTIFIER, TokenType.INTEGER_LITERAL,
        TokenType.FLOAT_LITERAL, TokenType.CHAR_LITERAL, TokenType.RPAREN,
        TokenType.RBRACKET,
        # Builtin dims are values: `blockIdx.x * blockDim.x + threadIdx.x`
        # is the index expression in EVERY kernel. Without these the op
        # split misses the most common multiply in the corpus.
        TokenType.THREAD_IDX, TokenType.BLOCK_IDX, TokenType.BLOCK_DIM,
        TokenType.GRID_DIM, TokenType.WARP_SIZE})
    _shared, _locals, _skip = set(), set(), set()
    _gstart = next((k for k, t in enumerate(tokens)
                    if t.type == TokenType.GLOBAL), 0)
    _depth, _gend = 0, len(tokens)
    for _k in range(_gstart, len(tokens)):
        if tokens[_k].type == TokenType.LBRACE:
            _depth += 1
        elif tokens[_k].type == TokenType.RBRACE:
            _depth -= 1
            if _depth == 0 and _k > _gstart:
                _gend = _k + 1
                break
    for _k in range(_gstart, _gend):
        _t = tokens[_k]
        _nx = tokens[_k + 1] if _k + 1 < len(tokens) else None
        if _t.type == TokenType.SHARED and _nx is not None:
            for _j in range(_k + 1, min(_k + 9, len(tokens))):
                if tokens[_j].type == TokenType.IDENTIFIER and \
                        _j + 1 < len(tokens) and tokens[_j + 1].type in (
                            TokenType.LBRACKET, TokenType.SEMICOLON,
                            TokenType.EQUAL):
                    _shared.add(tokens[_j].value)
                    break
        if _t.type == TokenType.IDENTIFIER and _nx is not None and \
                _nx.type == TokenType.LBRACKET and _k > 0 and \
                tokens[_k - 1].type in _TYPE_TOKENS and \
                _t.value not in _shared:
            _locals.add(_t.value)
        if _t.type == TokenType.IDENTIFIER and \
                _t.value.startswith('atomic') and _nx is not None and \
                _nx.type == TokenType.LPAREN:
            _pd, _j = 1, _k + 2
            while _j < len(tokens) and _pd:
                if tokens[_j].type == TokenType.LPAREN:
                    _pd += 1
                elif tokens[_j].type == TokenType.RPAREN:
                    _pd -= 1
                _j += 1
            for _m in range(_k + 2, min(_j, len(tokens))):
                _mt = tokens[_m]
                if _mt.type == TokenType.IDENTIFIER and _mt.value not in (
                        'atomicAdd', 'atomicExch', 'atomicCAS', 'atomicMin',
                        'atomicMax', 'atomicInc', 'atomicDec', 'atomicAnd',
                        'atomicOr', 'atomicXor', 'atomicSub'):
                    _skip.add(_m)
                    if _mt.value in _shared:
                        features['estimated_shared_reads'] += 1
                        features['estimated_shared_writes'] += 1
                    else:
                        features['estimated_global_reads'] += 1
                        features['estimated_global_writes'] += 1
                    break
    # The tokenizer emits WHITESPACE/NEWLINE/COMMENT tokens: every neighbor
    # test below uses significant tokens only. (`out[i] = ...` has a
    # WHITESPACE between `]` and `=`; without this every write reads as a
    # read and every `* alpha` misses its operator. Found by the smoke test,
    # not by reasoning -- the first cut got writes exactly backwards.)
    _TRIVIA = frozenset({TokenType.WHITESPACE, TokenType.NEWLINE,
                         TokenType.COMMENT})

    def _sig(k, d):
        k += d
        while 0 <= k < len(tokens) and tokens[k].type in _TRIVIA:
            k += d
        return (k, tokens[k]) if 0 <= k < len(tokens) else (k, None)

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
        
        # Memory access patterns (gpu-sim-u2o.8.3 D5: these counters were
        # initialised to 0 and had NO writer -- every row shipped reads=0.
        # Counting here is syntactic and per-thread single-pass: one subscript
        # or deref in the source is one access per thread. Loop-resident
        # accesses count ONCE (trip counts are runtime values; inventing one
        # would be a fitted constant by another name). Same convention as
        # estimated_flops, which has always counted loop bodies once.
        if token.type == TokenType.SHARED:
            # 8.3: `__shared__` lexes as SHARED, never IDENTIFIER, so the old
            # value-test inside the IDENT branch was dead -- shared kernels
            # shipped uses_shared_memory False. (Accesses are classified by
            # _shared regardless of this flag.)
            features['uses_shared_memory'] = True
        if token.type == TokenType.IDENTIFIER:
            # Check for vector types
            _ni, nxt = _sig(i, +1)
            _pi, prv = _sig(i, -1)
            # Call sites (fmaxf, atomicAdd, __syncthreads, ...). Keyword
            # statements are keyword tokens, never IDENTIFIERS, so any
            # IDENT+( is a genuine call -- except the kernel's own definition
            # name (`__global__ void name(`), excluded by the type-prev rule.
            if nxt is not None and nxt.type == TokenType.LPAREN and not (
                    prv is not None and (prv.type in _TYPE_TOKENS or prv.type
                                          in (TokenType.GLOBAL, TokenType.DEVICE,
                                              TokenType.HOST, TokenType.STATIC,
                                              TokenType.INLINE))):
                features['num_call_sites'] += 1
            # Subscript access name[...]: read unless the bracket closes onto
            # an assignment. Compound (+=) and ++/-- are read+write.
            if nxt is not None and nxt.type == TokenType.LBRACKET and \
                    i not in _skip and token.value not in _locals:
                depth, j = 1, _ni + 1
                while j < len(tokens) and depth:
                    if tokens[j].type == TokenType.LBRACKET:
                        depth += 1
                    elif tokens[j].type == TokenType.RBRACKET:
                        depth -= 1
                    j += 1
                _aj, after = _sig(j - 1, +1)
                _, after2 = _sig(_aj, +1) if after is not None else (_aj, None)
                is_write = after is not None and after.type in _ASSIGN_TOKENS
                is_incr = after is not None and after2 is not None and (
                    (after.type == TokenType.PLUS and after2.type == TokenType.PLUS)
                    or (after.type == TokenType.MINUS
                        and after2.type == TokenType.MINUS))
                is_compound = after is not None and after.type in _COMPOUND_TOKENS
                key = ('estimated_shared_reads' if token.value in _shared
                       else 'estimated_global_reads')
                wkey = ('estimated_shared_writes' if token.value in _shared
                        else 'estimated_global_writes')
                if is_write:
                    features[wkey] += 1
                    if is_compound or is_incr:
                        features[key] += 1
                else:
                    features[key] += 1
                    if is_incr:
                        features[wkey] += 1
            # Unary *p deref: STAR whose left is not a value. Declaration
            # `float *q` is excluded by the type-prev rule; `*(expr)` counts
            # one global read; `*p = v` counts the write instead of the read.
            # (Subscript `*p[i]` is left to the subscript rule above.)
        
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

        # Operator mix (8.3): the add/mul/div split. Binary context only -- a
        # STAR/PLUS/MINUS/SLASH with a non-value on either side is a deref,
        # a declaration, a comment slash, or a ++/-- piece, not an op.
        # Compound assigns (+= etc.) count as their op AND as read+write
        # memory is NOT touched here (that is the subscript rule's job).
        if token.type in (TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
                          TokenType.SLASH) and i > 0 and i + 1 < len(tokens):
            _, _pt = _sig(i, -1)
            _, _nt2 = _sig(i, +1)
            _pv = _pt.type if _pt is not None else None
            _nt = _nt2.type if _nt2 is not None else None
            if _pv in _VALUE_TOKENS and (_nt in _VALUE_TOKENS or
                                         _nt == TokenType.LPAREN):
                if token.type == TokenType.PLUS and not (
                        _nt == TokenType.PLUS or _pv == TokenType.PLUS):
                    features['estimated_add_ops'] += 1
                elif token.type == TokenType.MINUS and not (
                        _nt == TokenType.MINUS or _pv == TokenType.MINUS):
                    features['estimated_add_ops'] += 1
                elif token.type == TokenType.STAR:
                    features['estimated_mul_ops'] += 1
                elif token.type == TokenType.SLASH and _pv != TokenType.SLASH:
                    features['estimated_div_ops'] += 1
        if token.type in (TokenType.PLUS_EQUAL, TokenType.MINUS_EQUAL):
            features['estimated_add_ops'] += 1
        elif token.type == TokenType.STAR_EQUAL:
            features['estimated_mul_ops'] += 1
        elif token.type == TokenType.SLASH_EQUAL:
            features['estimated_div_ops'] += 1

        # Unary *p deref: STAR with a non-value to its left and no type to
        # its left (the latter excludes declarations like `float *q`).
        # `*(expr)` is one global read; `*p = v` is the write, not the read;
        # `*p[i]` (= *(p[i])) is left to the subscript rule to avoid doubling.
        if token.type == TokenType.STAR and i > 0 and i + 1 < len(tokens):
            _, _pp = _sig(i, -1)
            _nj, _nn = _sig(i, +1)
            _pv = _pp.type if _pp is not None else None
            if _pv not in _VALUE_TOKENS and _pv not in _TYPE_TOKENS:
                if _nn is not None and _nn.type == TokenType.IDENTIFIER:
                    _aj2, _af2 = _sig(_nj, +1)
                    _af = _af2.type if _af2 is not None else None
                    if _af != TokenType.LBRACKET and \
                            _nn.value not in _locals and i not in _skip:
                        _dkey = ('estimated_shared_writes'
                                 if _nn.value in _shared
                                 else 'estimated_global_writes')
                        _rkey = ('estimated_shared_reads'
                                 if _nn.value in _shared
                                 else 'estimated_global_reads')
                        if _af in _ASSIGN_TOKENS:
                            features[_dkey] += 1
                        else:
                            features[_rkey] += 1
                elif _nn is not None and _nn.type == TokenType.LPAREN:
                    features['estimated_global_reads'] += 1
    
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
    # 8.3 D5/item 9: NAMES of float/double scalar params (alpha in scale).
    # Each one used in the body is a per-thread data load: integer params
    # are shapes/indices (n, M, N, K), float params are data. Type-based, so
    # it generalises to every stage without naming any op.
    float_scalar_params = []
    in_kernel_params = False
    paren_depth = 0
    
    for ti, token in enumerate(tokens):
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
                # A bare `float name` / `double name` in the params is a
                # scalar data arg (pointer args have a STAR between type and
                # name, so their IDENTs never match this prev-type rule).
                if token.type in (TokenType.FLOAT, TokenType.DOUBLE):
                    _aj = ti + 1
                    while _aj < len(tokens) and tokens[_aj].type in (
                            TokenType.WHITESPACE, TokenType.NEWLINE,
                            TokenType.COMMENT):
                        _aj += 1
                    if _aj < len(tokens) and \
                            tokens[_aj].type == TokenType.IDENTIFIER:
                        float_scalar_params.append(tokens[_aj].value)
    
    # Count last argument
    if num_args > 0 or num_pointer_args > 0 or num_scalar_args > 0:
        num_args = max(1, num_args + 1)
    
    return {
        'num_args': num_args,
        'num_pointer_args': num_pointer_args,
        'num_scalar_args': num_scalar_args,
        'total_arg_bytes': total_arg_bytes,
        'float_scalar_params': float_scalar_params,
    }


def extract_features(source: str, launch_config: Optional[LaunchConfig] = None,
                     problem_size: Optional[int] = None) -> KernelFeatures:
    """Extract complete feature vector from kernel source, launch and SIZE.

    `problem_size` is required in spirit: a launch config alone does not say
    how much work the kernel does. It defaults to 0 only so the omission is
    visible in the emitted vector rather than fatal here -- see
    populate_feature_vector.py, which passes the MEASURED n for every row and
    asserts it is present.
    """
    if launch_config is None:
        launch_config = extract_launch_config(source)
    
    # Kernel body analysis
    body_features = analyze_kernel_body(source)
    
    # Argument analysis
    arg_features = extract_kernel_args(source)
    
    # Register estimation
    estimated_regs = estimate_registers(source)
    
    # Scale memory/compute estimates by COVERED work, not launched threads.
    # 8.3: every corpus kernel guards its body with `if (i < n)`, so threads
    # beyond n perform no memory and no compute. Scaling by total_threads
    # charges the vector for work that never happens -- on the deliberately
    # over-covered launches that is most of the launch. covers is the
    # corpus's own quantity (min(threads, n)); without a known n there is no
    # guard to reason about, so fall back to total_threads and say so.
    n = int(problem_size or 0)
    total_threads = launch_config.total_threads
    covers = min(total_threads, n) if n > 0 else total_threads
    # Scalar data loads: float/double params USED in the body (alpha, not n).
    # Counted per-thread like subscripts; the use-test keeps dead params out.
    _used = {t.value for t in tokenize(source)
             if t.type == TokenType.IDENTIFIER}
    scalar_reads = sum(1 for p in arg_features['float_scalar_params']
                       if p in _used)
    _gr = body_features['estimated_global_reads'] + scalar_reads
    _gw = body_features['estimated_global_writes']
    estimated_bytes_moved = (
        _gr * 4 + _gw * 4 +
        body_features['estimated_shared_reads'] * 4 +
        body_features['estimated_shared_writes'] * 4
    ) * covers
    
    return KernelFeatures(
        problem_size=n,
        covers=covers,
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
        estimated_global_reads=_gr * covers,
        estimated_global_writes=_gw * covers,
        estimated_shared_reads=body_features['estimated_shared_reads'] * covers,
        estimated_shared_writes=body_features['estimated_shared_writes'] * covers,
        estimated_bytes_moved=estimated_bytes_moved,
        estimated_flops=body_features['estimated_flops'] * covers,
        estimated_int_ops=body_features['estimated_int_ops'] * covers,
        has_loops=body_features['has_loops'],
        has_conditionals=body_features['has_conditionals'],
        max_loop_depth=body_features['max_loop_depth'],
        has_syncthreads=body_features['has_syncthreads'],
        num_syncthreads=body_features['num_syncthreads'],
        uses_shared_memory=body_features['uses_shared_memory'],
        uses_constant_memory=body_features['uses_constant_memory'],
        uses_vector_types=body_features['uses_vector_types'],
        estimated_add_ops=body_features['estimated_add_ops'] * covers,
        estimated_mul_ops=body_features['estimated_mul_ops'] * covers,
        estimated_div_ops=body_features['estimated_div_ops'] * covers,
        num_call_sites=body_features['num_call_sites'] * covers,
    )


def extract_normalized_features(source: str, launch_config: Optional[LaunchConfig] = None) -> list[float]:
    """Extract and normalize features for model input."""
    # 8.3: named attributes, never positions. The old body indexed vec[0] as
    # grid_x AFTER problem_size was prepended at 0, so every normalized slot
    # was off by one and nobody could tell by reading it. Length is asserted
    # by test (item 6 covers to_vector; this asserts the normalized twin).
    f = extract_features(source, launch_config)
    specs = RTX_5090_SPECS
    max_bytes = specs['num_sms'] * specs['max_threads_per_block'] * 4 * 1024
    max_flops = specs['peak_fp32_tflops'] * 1e12 / 1e9 * specs['num_sms'] * specs['max_threads_per_block'] / 32
    pairs = [
        ('problem_size', specs['num_sms'] * specs['max_threads_per_block'] * 1024),
        ('grid_x', specs['num_sms'] * 4), ('grid_y', 65535), ('grid_z', 65535),
        ('block_x', specs['max_threads_per_block']),
        ('block_y', specs['max_threads_per_block']), ('block_z', 64),
        ('shared_mem_bytes', specs['max_shared_mem_per_block']),
        ('estimated_registers_per_thread', specs['max_registers_per_thread']),
        ('num_args', 20), ('num_pointer_args', 10), ('num_scalar_args', 10),
        ('total_arg_bytes', 1024),
        ('estimated_global_reads', max_bytes),
        ('estimated_global_writes', max_bytes),
        ('estimated_shared_reads', max_bytes),
        ('estimated_shared_writes', max_bytes),
        ('estimated_bytes_moved', max_bytes),
        ('estimated_flops', max_flops), ('estimated_int_ops', max_flops),
        ('has_loops', 1), ('has_conditionals', 1), ('max_loop_depth', 8),
        ('has_syncthreads', 1), ('num_syncthreads', 16),
        ('uses_shared_memory', 1), ('uses_constant_memory', 1),
        ('uses_vector_types', 1),
        ('covers', specs['num_sms'] * specs['max_threads_per_block'] * 1024),
        ('estimated_add_ops', max_flops), ('estimated_mul_ops', max_flops),
        ('estimated_div_ops', max_flops), ('num_call_sites', 64),
    ]
    normalized = [normalize_feature(float(getattr(f, name)), mx) for name, mx in pairs]
    assert len(normalized) == len(f.to_vector()), \
        (len(normalized), len(f.to_vector()))
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
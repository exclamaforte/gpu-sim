#!/usr/bin/env python3
"""Kernel-driver synthesizer (gpu-sim-u2o.6.9).

Takes a stage-1 KernelSpec (bare __global__ body) and emits a full compilable
.cu program: cudaMalloc buffers sized from problem_size, seeded host data,
5 warmup launches, then 20 timed launches via cudaEventRecord /
cudaEventElapsedTime (per measurement decision u2o.3), printing each trial
as one JSON line to stdout.

Median/IQR are intentionally NOT computed here -- measurement_stats.py owns
that in Python. Supports the stage-1 elementwise family (float/int/double
pointers + float/int scalars). Other families fail loudly, not silently.

gpu-sim-u2o.11.3: dimension scalars. A scalar arg whose name matches an
int-valued entry in the spec's metadata (GEMM M/N/K, 2D rows/cols, strided
stride, histogram nbins) is passed that value; without this every stage-4
driver launched with M=N=K=1 and measured the launcher, not the GEMM.
Metadata wins over the size-names on purpose: GEMM's N is a dimension, not
the element count. Stage-1 specs carry no colliding metadata keys, so their
drivers are byte-identical to before.

gpu-sim-u2o.11.3 (s4 re-run): arity-sized GEMM buffers. The generic rule sized
every pointer at N = M*N*K elements, so a (1024,1024,256) GEMM filled and
copied 3 GB per buffer on the host while the timed kernel ran microseconds --
the s4 slice averaged 5.5% GPU util and the queue killed it at the 5-minute
grace check (10% minimum). GEMM args are exactly A: M*K, B: K*N, C: M*N by
the generator's fixed convention; sizing them so cuts host work ~500x on
square cases while the timed launch is byte-identical. Falls back to N for
any GEMM-family spec whose dims are absent, never silently.

gpu-sim-f6y: OOB guard in the driver preflight. synthesize() now refuses
kernels whose subscripts provably exceed the buffers the driver allocates,
so template bugs like segmented_read's in[segment*n+offset] fail on CPU
instead of occupying a GPU slot and crashing. SCOPE, stated honestly:
checked are subscripts built SOLELY from the linear thread id (the exact
`blockIdx.x * blockDim.x + threadIdx.x` idiom), problem_size n, int scalar
params with driver-known values, and integer literals, combined with
+ - * / % -- evaluated by interval arithmetic over the thread range that
can reach them. Ranges come from guard contexts: `if (i < n)` / early
`if (i >= n) return;` narrow to [0, min(threads, n)); an `if (<EXPR> < n)`
guard caps that exact expression at n-1, and monotone idioms (i*K / i+K
with known K) narrow the linear id itself by inversion. A subscript inside
an `if (i < n && ...)` condition past the && is covered left-to-right
short-circuit order. Everything else is reported UNCHECKED and passed
through, never refused: data-dependent indices (gather/scatter/random),
blockIdx/threadIdx spellings, loop induction variables (including
pre-increment `++k`, which the mutation scan detects like `k++`),
__shared__ arrays, driver-emitted host code (bounds match allocation by
construction there). Refusal fires ONLY on proven overflow (max >=
capacity) or proven-negative index -- never on "cannot tell", which would
convert uncovered bugs from loud crashes into silent passes. Interval
evaluation is correlatively coarse (it can over-estimate), so
over-strictness is policed empirically: the committed regression sweep
tests/test_oob_guard.py proves every generated template passes on its real
cells, and the broken templates are refused (negative control recorded).
"""

import json
import re
import sys

WARMUP = 5
TRIALS = 20

# base CUDA type -> (sizeof, host fill expr of index i, kernel scalar value)
_SCALAR_DEFAULTS = {
    'float': '2.0f',
    'double': '2.0',
    'int': '1',
}

_BASE_SIZES = {'float': 4, 'double': 8, 'int': 4, 'char': 1,
               'half': 2, 'unsigned int': 4}


def _base_type(ctype: str) -> str:
    """Strip const/__restrict__/pointers -> base type name."""
    t = ctype.replace('const', '').replace('__restrict__', '').strip()
    t = t.rstrip('*').strip()
    return t


def _is_pointer(ctype: str) -> bool:
    return '*' in ctype


def _buffer_counts(spec: dict, dims: dict, n: int) -> dict:
    """Element count per pointer arg. The default is N (problem_size).

    GEMM-family specs (op_type gemm_*, fixed A/B/C args, M/N/K dims present)
    size arity-correct: A: M*K, B: K*N, C: M*N. Anything else -- including a
    GEMM spec missing its dims -- falls back to N rather than guessing.
    """
    counts = {}
    if (spec.get("op_type", "").startswith("gemm")
            and all(k in dims for k in ("M", "N", "K"))):
        counts = {"A": dims["M"] * dims["K"],
                  "B": dims["K"] * dims["N"],
                  "C": dims["M"] * dims["N"]}
    return counts


_LINEAR_ID_RE = re.compile(
    r'int\s+([A-Za-z_]\w*)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;')
_PURE_ASSIGN_RE = re.compile(r'int\s+([A-Za-z_]\w*)\s*=\s*([^;{}]*);')
# Mutation in any spelling, pre-increment (++k) included: GEMM's k-loop
# mutates with ++k, and missing that spelling let k evaluate as const 0.
_MUTATE_RE = re.compile(
    r'([A-Za-z_]\w*)\s*(\+\+|--|\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=|=(?!=))'
    r'|\+\+([A-Za-z_]\w*)|--([A-Za-z_]\w*)')
_GUARD_LT_RE = re.compile(r'if\s*\(\s*([A-Za-z_]\w*)\s*<\s*n\s*\)')
_GUARD_LT_AND_RE = re.compile(r'if\s*\(\s*([A-Za-z_]\w*)\s*<\s*n\s*&&')
_GUARD_GE_RE = re.compile(r'if\s*\(\s*([A-Za-z_]\w*)\s*>=\s*n\s*\)\s*return\s*;')
_GUARD_EXPR_RE = re.compile(r'if\s*\(\s*(.+?)\s*<\s*n\s*\)')
_NAME_RE = re.compile(r'[A-Za-z_]\w*|\d+|[+\-*/%()]')

_LID = object()  # env marker: the linear thread id, range supplied at eval


class _Unchecked(Exception):
    """A subscript the guard cannot analyze: pass through, never refuse."""


def _span_after(src, pos):
    """Brace-delimited or single-statement span starting at pos (after `if`)."""
    i = pos
    while i < len(src) and src[i] in ' \t\n':
        i += 1
    if i < len(src) and src[i] == '{':
        depth, j = 0, i
        while j < len(src):
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
                if depth == 0:
                    return (i, j + 1)
            j += 1
        raise _Unchecked("unbalanced braces")
    j = src.find(';', i)
    if j < 0:
        raise _Unchecked("unterminated statement")
    return (i, j + 1)


def _cond_end(src, if_pos):
    """End offset (exclusive) of the `if (...)` condition paren."""
    i = src.find('(', if_pos)
    depth, j = 0, i
    while j < len(src):
        if src[j] == '(':
            depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise _Unchecked("unbalanced condition")


def _eval_interval(expr, env, lid_range, scalars, _seen=()):
    """Interval (lo, hi) for expr, or raise _Unchecked.

    env maps derived int locals to rhs strings (or _LID for the linear id);
    lid_range is its (lo, hi); scalars maps known names to ints. C semantics
    matter only for / and %: both require a positive constant divisor and a
    non-negative dividend, else _Unchecked.
    """
    toks = [t for t in _NAME_RE.findall(expr) if t.strip()]
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat():
        t = toks[pos[0]]
        pos[0] += 1
        return t

    def parse_expr():
        lo, hi = parse_term()
        while peek() in ('+', '-'):
            op = eat()
            rlo, rhi = parse_term()
            if op == '+':
                lo, hi = lo + rlo, hi + rhi
            else:
                lo, hi = lo - rhi, hi - rlo
        return lo, hi

    def parse_term():
        lo, hi = parse_factor()
        while peek() in ('*', '/', '%'):
            op = eat()
            rlo, rhi = parse_factor()
            if op == '*':
                c = [lo * rlo, lo * rhi, hi * rlo, hi * rhi]
                lo, hi = min(c), max(c)
            else:
                if rlo != rhi or rlo <= 0 or lo < 0:
                    raise _Unchecked(
                        f"non-constant/non-positive divisor in {expr!r}")
                d = rlo
                if op == '/':
                    lo, hi = lo // d, hi // d
                else:
                    lo, hi = 0, min(d - 1, hi)
        return lo, hi

    def parse_factor():
        t = peek()
        if t == '(':
            eat()
            lo, hi = parse_expr()
            if peek() != ')':
                raise _Unchecked(f"unbalanced parens in {expr!r}")
            eat()
            return lo, hi
        if t == '-':
            eat()
            lo, hi = parse_factor()
            return -hi, -lo
        if t is None or not (t[0].isalpha() or t[0] == '_' or t.isdigit()):
            raise _Unchecked(f"bad token {t!r} in {expr!r}")
        eat()
        if t.isdigit():
            return int(t), int(t)
        if t in env:
            rhs = env[t]
            if rhs is _LID:
                return lid_range
            if t in _seen:
                raise _Unchecked(f"assignment cycle at {t!r}")
            return _eval_interval(rhs, env, lid_range, scalars,
                                  _seen + (t,))
        if t in scalars:
            v = scalars[t]
            return v, v
        raise _Unchecked(f"unknown name {t!r} in {expr!r}")

    out = parse_expr()
    if peek() is not None:
        raise _Unchecked(f"trailing tokens in {expr!r}")
    return out


def _masked_declarations(src):
    """Source with `int v = ...;` declaration spans blanked.

    The mutation scan must not see the declaring assignment itself: every
    `int v = expr;` contains `v =`, which is not a mutation. Loop-init
    declarations (`for (int k = 0; ...`) are blanked the same way; the
    loop's own `k++` still shows as a mutation.
    """
    chars = list(src)
    for m in _PURE_ASSIGN_RE.finditer(src):
        for i in range(m.start(), m.end()):
            if chars[i] != '\n':
                chars[i] = ' '
    return ''.join(chars)


def _inversion_cap(lhs, lid, scalars, n):
    """Narrow the linear id from a monotone `if (<LHS> < n)` guard.

    Recognizes lid*K / K*lid (K a known scalar or literal with value >= 1)
    and lid+K / K+lid (K >= 0): E < n with E monotone in lid bounds lid
    above. Returns the cap, or None when the shape is not recognized (then
    no narrowing -- never a refusal).
    """
    t = re.sub(r'\s+', '', lhs)
    op, ktxt = None, None
    if t.startswith(lid + '*'):
        op, ktxt = '*', t[len(lid) + 1:]
    elif t.startswith(lid + '+'):
        op, ktxt = '+', t[len(lid) + 1:]
    elif t.endswith('*' + lid):
        op, ktxt = '*', t[:-(len(lid) + 1)]
    elif t.endswith('+' + lid):
        op, ktxt = '+', t[:-(len(lid) + 1)]
    else:
        return None
    kval = int(ktxt) if ktxt.isdigit() else scalars.get(ktxt)
    if not isinstance(kval, int):
        return None
    if op == '*' and kval >= 1:
        return (n - 1) // kval
    if op == '+' and kval >= 0:
        return n - 1 - kval
    return None


def check_subscripts_bounded(src, *, kernel_name, args, capacities, scalars,
                              n, threads):
    """Refuse kernels whose subscripts provably exceed driver allocations.

    Returns (n_checked, [unchecked notes]). Raises ValueError naming the
    buffer, subscript, reachable max, and capacity on proven overflow.
    See the module docstring for the honest scope: only pure idioms are
    checked; everything else passes through with a note, never a refusal.
    """
    ptr_args = {a["name"] for a in args if _is_pointer(a.get("type", ""))}
    lid = None
    m = _LINEAR_ID_RE.search(src)
    if m:
        lid = m.group(1)
    env = {}
    for am in _PURE_ASSIGN_RE.finditer(src):
        if am.group(1) != lid:
            env.setdefault(am.group(1), am.group(2).strip())
    masked = _masked_declarations(src)
    mutated = set()
    for mm in _MUTATE_RE.finditer(masked):
        if mm.group(1) is not None:
            mutated.add(mm.group(1))  # postfix: k++, k += 2, k = ...
        for g in (mm.group(3), mm.group(4)):
            if g is not None:
                mutated.add(g)  # prefix: ++k, --k
    for k in mutated:
        env.pop(k, None)
    if lid is not None:
        env[lid] = _LID
    # Guard contexts narrow the thread range that can reach a subscript.
    guarded, capped, inverted = [], [], []  # (span, _) / (span, lhs) / (span, cap)
    if lid:
        for gm in _GUARD_LT_RE.finditer(src):
            if gm.group(1) != lid:
                continue
            try:
                cond_end = _cond_end(src, gm.start())
                rest = src[gm.end():cond_end].lstrip()
                body = _span_after(src, cond_end)
                if rest.startswith('&&'):
                    # `if (lid < n && ...)`: the condition's remainder runs
                    # only where lid < n holds (left-to-right &&).
                    guarded.append(((gm.end(), body[1]), None))
                elif rest == '':
                    # bare `if (lid < n)`: the body runs only where lid < n.
                    guarded.append((body, None))
                # `||` and anything else: not a sound guard context, skip.
            except _Unchecked:
                pass
        for gm in _GUARD_LT_AND_RE.finditer(src):
            if gm.group(1) != lid:
                continue
            try:
                cond_end = _cond_end(src, gm.start())
                body = _span_after(src, cond_end)
                guarded.append((body, None))  # body needs the whole cond true
                if '||' not in src[gm.end():cond_end]:
                    # remainder runs only where lid < n (left-to-right &&)
                    guarded.append(((gm.end(), cond_end), None))
            except _Unchecked:
                pass
        for gm in _GUARD_GE_RE.finditer(src):
            if gm.group(1) == lid:
                guarded.append(((gm.end(), len(src)), None))
        for gm in _GUARD_EXPR_RE.finditer(src):
            try:
                cond_end = _cond_end(src, gm.start())
                body = _span_after(src, cond_end)
                lhs = re.sub(r'\s+', '', gm.group(1))
                capped.append(((body[0], body[1]), lhs))
                cap = _inversion_cap(gm.group(1), lid, scalars, n)
                if cap is not None:
                    inverted.append(((body[0], body[1]), cap))
            except _Unchecked:
                pass
    norm = lambda s: re.sub(r'\s+', '', s)
    checked, unchecked = 0, []
    for sm in re.finditer(r'([A-Za-z_]\w*)\s*\[', src):
        buf = sm.group(1)
        if buf not in ptr_args:
            continue  # __shared__ tiles, locals: capacity unknown, unchecked
        cap = capacities.get(buf)
        if cap is None:
            unchecked.append(f"{buf}[...]: no driver capacity")
            continue
        # balanced-bracket content; nesting weirdness -> unchecked.
        depth, j, parts = 0, sm.end() - 1, []
        bad = False
        while j < len(src):
            c = src[j]
            if c == '[':
                depth += 1
                if depth > 1:
                    parts.append(c)
            elif c == ']':
                depth -= 1
                if depth == 0:
                    break
                parts.append(c)
            elif c in ';{}' and depth == 1 and not parts:
                bad = True
                break
            else:
                parts.append(c)
            j += 1
        if bad or depth != 0:
            unchecked.append(f"{buf}[...]: unparseable subscript")
            continue
        sub = ''.join(parts)
        pos = sm.start()
        imax = threads
        for (a, b), _ in guarded:
            if a <= pos < b:
                imax = min(imax, n)
        lid_hi = imax - 1
        for (a, b), capv in inverted:
            if a <= pos < b:
                lid_hi = min(lid_hi, capv)
        cap_hi = None
        for (a, b), lhs in capped:
            if a <= pos < b and norm(sub) == lhs:
                cap_hi = n - 1
        if lid is None:
            unchecked.append(f"{buf}[{sub}]: no linear thread id")
            continue
        try:
            lo, hi = _eval_interval(sub, env, (0, max(lid_hi, 0)), scalars)
            if cap_hi is not None:
                hi = min(hi, cap_hi)
            checked += 1
            if hi >= cap:
                raise ValueError(
                    f"oob-guard: {kernel_name}: subscript {buf}[{sub}] can "
                    f"reach {hi} (thread range [0,{imax})) but the driver "
                    f"allocates {cap} elements; refusing on CPU")
            if lo < 0:
                raise ValueError(
                    f"oob-guard: {kernel_name}: subscript {buf}[{sub}] can "
                    f"go negative (min {lo}); refusing on CPU")
        except _Unchecked as u:
            unchecked.append(f"{buf}[{sub}]: {u}")
    return checked, unchecked


def synthesize(spec: dict) -> str:
    """Build a full .cu program from a KernelSpec dict.

    Required spec keys: source, kernel_name, grid, block, args,
    problem_size. Optional: shared_mem_bytes (default 0).
    """
    for key in ('source', 'kernel_name', 'grid', 'block', 'args', 'problem_size'):
        if key not in spec:
            raise ValueError(f"KernelSpec missing required key: {key}")
    src = spec['source']
    # gpu-sim-9ti: the name to launch comes from the SOURCE, not from the
    # dict's kernel_name. generator_stage2..5 declared f"{op}_kernel" while
    # their source defines {op}, so every stage2-5 driver failed to compile
    # with "identifier X is undefined". The source is the authority here; a
    # spec whose source declares no __global__ at all is a real error.
    from epic_u2o.kernel_names import resolve_kernel_name
    kname = resolve_kernel_name(spec)
    grid = spec['grid']
    block = spec['block']
    args = spec['args']
    n = int(spec['problem_size'])
    shmem = int(spec.get('shared_mem_bytes', 0))
    # gpu-sim-u2o.11.3: dimension scalars (M/N/K, rows/cols, stride, nbins)
    # come from the spec's metadata. Only plain ints qualify; anything else
    # keeps the old size-name/default behaviour below.
    dims = {k: v for k, v in (spec.get('metadata') or {}).items()
            if type(v) is int}
    buf_counts = _buffer_counts(spec, dims, n)
    # gpu-sim-f6y: OOB guard runs on the kernel source BEFORE any driver is
    # emitted. capacities are the exact element counts the mallocs below
    # use; scalars are the exact int values the launch passes.
    threads = (grid[0] * grid[1] * grid[2]
               * block[0] * block[1] * block[2])
    capacities = {}
    for a in args:
        if _is_pointer(a.get('type', '')):
            capacities[a['name']] = buf_counts.get(a['name'], n)
    check_subscripts_bounded(
        src, kernel_name=kname, args=args, capacities=capacities,
        scalars=dict(dims, n=n), n=n, threads=threads)

    if '__global__' not in src:
        raise ValueError("spec['source'] contains no __global__ kernel")

    decls, mallocs, fills, call_args, frees = [], [], [], [], []
    for a in args:
        name = a['name']
        ctype = a['type']
        if _is_pointer(ctype):
            base = _base_type(ctype)
            if base not in _BASE_SIZES:
                raise ValueError(f"unsupported pointer base type: {ctype!r} "
                                 f"(stage-1 synthesizer handles {sorted(_BASE_SIZES)})")
            cnt = buf_counts.get(name, n)
            decls.append(f"    {base} *d_{name} = NULL;")
            decls.append(f"    {base} *h_{name} = ({base}*)malloc({cnt} * sizeof({base}));")
            mallocs.append(f"    CHECK(cudaMalloc((void**)&d_{name}, {cnt} * sizeof({base})));")
            if name.startswith('out') or name == 'y':
                fills.append(f"    for (int i = 0; i < {cnt}; i++) h_{name}[i] = ({base})0;")
            else:
                fills.append(f"    for (int i = 0; i < {cnt}; i++) h_{name}[i] = ({base})(i * 0.5f + 1.0f);")
            mallocs.append(f"    CHECK(cudaMemcpy(d_{name}, h_{name}, {cnt} * sizeof({base}), cudaMemcpyHostToDevice));")
            call_args.append(f"d_{name}")
            frees.append(f"    cudaFree(d_{name}); free(h_{name});")
        else:
            base = _base_type(ctype)
            if name in dims:
                val = str(dims[name])
            elif name in ('n', 'N', 'size', 'len', 'count'):
                val = 'N'
            elif base in _SCALAR_DEFAULTS:
                val = _SCALAR_DEFAULTS[base]
            else:
                raise ValueError(f"unsupported scalar type: {ctype!r}")
            call_args.append(val)

    gx, gy, gz = grid
    bx, by, bz = block
    call = f"{kname}<<<grid, block, {shmem}>>>({', '.join(call_args)})"

    lines = [
        '#include <stdio.h>',
        '#include <stdlib.h>',
        '#include <cuda_runtime.h>',
        '',
        '#define CHECK(call) do { cudaError_t e = (call); \\',
        '    if (e != cudaSuccess) { \\',
        '        fprintf(stderr, "{\\"error\\": \\"%s:%d %s\\"} \\n", __FILE__, __LINE__, cudaGetErrorString(e)); \\',
        '        return 2; } } while (0)',
        '',
        src.strip(),
        '',
        'int main(void) {',
        '    const int N = %d;' % n,
        '    dim3 grid(%d, %d, %d);' % (gx, gy, gz),
        '    dim3 block(%d, %d, %d);' % (bx, by, bz),
        '\n'.join(decls),
        '    size_t dev_free_before_b = 0, dev_total_b = 0;',
        '    CHECK(cudaMemGetInfo(&dev_free_before_b, &dev_total_b));',
        '\n'.join(mallocs),
        '\n'.join(fills),
        '    cudaEvent_t start, stop;',
        '    CHECK(cudaEventCreate(&start));',
        '    CHECK(cudaEventCreate(&stop));',
        '    size_t dev_free_after_b = 0;',
        '    CHECK(cudaMemGetInfo(&dev_free_after_b, &dev_total_b));',
        '    printf("{\\"mem\\": {\\"delta_mb\\": %.1f, \\"free_after_mb\\": %.1f, \\"total_mb\\": %.1f, \\"device_used_mb\\": %.1f}}\\n",',
        '           (dev_free_before_b - dev_free_after_b) / 1048576.0,',
        '           dev_free_after_b / 1048576.0, dev_total_b / 1048576.0,',
        '           (dev_total_b - dev_free_after_b) / 1048576.0);',
        '    /* delta_mb is this run footprint; device_used_mb is device-wide */',
        '    fflush(stdout);',
        '    for (int w = 0; w < %d; w++) { %s; }' % (WARMUP, call),
        '    CHECK(cudaDeviceSynchronize());',
        '    for (int t = 0; t < %d; t++) {' % TRIALS,
        '        float ms = 0.0f;',
        '        CHECK(cudaEventRecord(start));',
        f'        {call};',
        '        CHECK(cudaEventRecord(stop));',
        '        CHECK(cudaEventSynchronize(stop));',
        '        CHECK(cudaEventElapsedTime(&ms, start, stop));',
        '        printf("{\\"trial\\": %d, \\"time_ms\\": %.6f}\\n", t, ms);',
        '        fflush(stdout);',
        '    }',
        '    CHECK(cudaDeviceSynchronize());',
        '\n'.join(frees),
        '    return 0;',
        '}',
        '',
    ]
    return '\n'.join(lines)


def spec_from_stage1(op='scale', grid=None, block=None, n=4096):
    """Convenience: build one stage-1 KernelSpec dict without importing the
    generator module (keeps ts jobs dependency-light)."""
    # Derive the repo root from this file so the function works in a lab
    # worktree (or any checkout), not just the original clone path.
    # (gpu-sim-u2o.6.1.1; this branch predates the fix and carried a stale
    # full copy of the file, so the rebase replayed the old line.)
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from epic_u2o.generator_stage1 import generate_kernels
    for spec in generate_kernels(
            ops=[op], grid_sweeps=[grid or [4, 1, 1]],
            block_sweeps=[block or [256, 1, 1]], problem_sizes=[n],
            max_kernels=1):
        d = spec.to_dict()
        return d
    raise RuntimeError("generator produced no spec")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Synthesize a driver .cu from a stage-1 spec')
    p.add_argument('--op', default='scale')
    p.add_argument('--grid', default='4,1,1')
    p.add_argument('--block', default='256,1,1')
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('-o', '--output', required=True)
    a = p.parse_args()
    grid = [int(x) for x in a.grid.split(',')]
    block = [int(x) for x in a.block.split(',')]
    spec = spec_from_stage1(a.op, grid, block, a.n)
    if grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2] < a.n:
        raise SystemExit("grid*block covers fewer threads than n; pick a bigger config")
    with open(a.output, 'w') as f:
        f.write(synthesize(spec))
    print(json.dumps({'output': a.output, 'kernel': spec['kernel_name'],
                      'trials': TRIALS, 'warmup': WARMUP}))

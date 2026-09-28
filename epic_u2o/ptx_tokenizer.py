"""PTX/SASS extraction and tokenizer with ptxas resource usage parsing.

Compiles CUDA kernels to PTX and SASS, tokenizes the instruction stream,
and parses ptxas -v output (registers/thread, shared mem/block, spill counts).
"""

import subprocess
import tempfile
import os
import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


class PTXTokenType(Enum):
    """PTX instruction token types."""
    # Instructions
    INST_MOV = auto()
    INST_LD = auto()
    INST_ST = auto()
    INST_ADD = auto()
    INST_SUB = auto()
    INST_MUL = auto()
    INST_FMA = auto()
    INST_DIV = auto()
    INST_MIN = auto()
    INST_MAX = auto()
    INST_SET = auto()
    INST_BRA = auto()
    INST_CALL = auto()
    INST_RET = auto()
    INST_SYNC = auto()
    INST_BAR = auto()
    INST_ATOM = auto()
    INST_RED = auto()
    INST_VOTE = auto()
    INST_SHFL = auto()
    INST_TEX = auto()
    INST_SULD = auto()
    INST_SUST = auto()
    INST_SURED = auto()
    INST_SUATOM = auto()
    INST_CVT = auto()
    INST_SELP = auto()
    INST_SLCT = auto()
    INST_LOP = auto()
    INST_SHF = auto()
    INST_PMEVENT = auto()
    INST_BRK = auto()
    INST_CONT = auto()
    INST_EXIT = auto()
    INST_JMP = auto()
    INST_JMX = auto()
    INST_S2R = auto()
    INST_R2S = auto()
    INST_MOV_MATRIX = auto()
    INST_LD_MATRIX = auto()
    INST_MMA = auto()
    INST_CP_ASYNC = auto()
    INST_WAIT = auto()
    INST_CLUSTER = auto()
    INST_UNKNOWN = auto()
    UNKNOWN = auto()
    
    # Operands
    REGISTER = auto()
    PREDICATE_REG = auto()
    CONSTANT = auto()
    LABEL = auto()
    STATE_SPACE = auto()  # .global, .shared, .local, .const
    TYPE_SPEC = auto()    # .b8, .b16, .b32, .b64, .f16, .f32, .f64
    
    # Directives
    DIR_VERSION = auto()
    DIR_TARGET = auto()
    DIR_ADDRESS_SIZE = auto()
    DIR_ENTRY = auto()
    DIR_FUNC = auto()
    DIR_PARAM = auto()
    DIR_LOCAL = auto()
    DIR_SHARED = auto()
    DIR_CONST = auto()
    DIR_GLOBAL = auto()
    DIR_VISIBLE = auto()
    DIR_ALIGN = auto()
    DIR_MAXNREG = auto()
    DIR_MAXNCTAPERSM = auto()
    DIR_MAXNTID = auto()
    DIR_MINNCTAPERSM = auto()
    DIR_UNKNOWN = auto()
    
    # Punctuation
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    
    # Special
    IDENTIFIER = auto()
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    COMMENT = auto()
    NEWLINE = auto()
    EOF = auto()


@dataclass
class PTXToken:
    type: PTXTokenType
    value: str
    line: int
    column: int


@dataclass
class PTXResourceInfo:
    """Resource usage parsed from ptxas -v output."""
    registers_per_thread: int = 0
    shared_mem_per_block: int = 0  # bytes
    local_mem_per_thread: int = 0  # bytes
    constant_mem_per_block: int = 0  # bytes
    spilled_registers: int = 0
    spilled_local_mem: int = 0
    frame_size: int = 0
    max_threads_per_block: int = 0
    max_blocks_per_sm: int = 0
    occupancy_warps_per_sm: int = 0
    occupancy_blocks_per_sm: int = 0
    achieved_occupancy: float = 0.0


def compile_to_ptx(source: str, arch: str = 'sm_90') -> tuple[str, str]:
    """Compile CUDA source to PTX using nvcc.
    
    Returns (ptx_code, stderr) tuple.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(source)
        cu_file = f.name
    
    ptx_file = cu_file.replace('.cu', '.ptx')
    
    try:
        # Compile to PTX
        cmd = ['nvcc', '-ptx', f'-arch={arch}', cu_file, '-o', ptx_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return '', result.stderr
        
        with open(ptx_file, 'r') as f:
            ptx = f.read()
        
        return ptx, result.stderr
    
    finally:
        # Cleanup
        for f in [cu_file, ptx_file]:
            if os.path.exists(f):
                os.unlink(f)


def compile_to_sass(source: str, arch: str = 'sm_90') -> tuple[str, str]:
    """Compile CUDA source to SASS using nvcc + cuobjdump.
    
    Returns (sass_code, stderr) tuple.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(source)
        cu_file = f.name
    
    cubin_file = cu_file.replace('.cu', '.cubin')
    
    try:
        # Compile to cubin
        cmd = ['nvcc', '-cubin', f'-arch={arch}', cu_file, '-o', cubin_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return '', result.stderr
        
        # Extract SASS with cuobjdump (may not be available)
        if not os.path.exists(cubin_file):
            return '', 'cubin not created'
        
        # Check if cuobjdump is available
        try:
            cmd = ['cuobjdump', '-sass', cubin_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout, result.stderr
        except FileNotFoundError:
            return '', 'cuobjdump not found in PATH'
    
    finally:
        for f in [cu_file, cubin_file]:
            if os.path.exists(f):
                os.unlink(f)


def get_ptxas_info(source: str, arch: str = 'sm_90') -> PTXResourceInfo:
    """Run nvcc with ptxas -v to get detailed resource usage.
    
    Returns PTXResourceInfo with parsed metrics.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(source)
        cu_file = f.name
    
    try:
        # Use nvcc with -Xptxas -v to get verbose output, compile to object file
        obj_file = cu_file.replace('.cu', '.o')
        cmd = ['nvcc', f'-arch={arch}', '-Xptxas', '-v', '-c', cu_file, '-o', obj_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Cleanup
        if os.path.exists(obj_file):
            os.unlink(obj_file)
        
        return parse_ptxas_output(result.stderr)
    
    finally:
        if os.path.exists(cu_file):
            os.unlink(cu_file)


def parse_ptxas_output(stderr: str) -> PTXResourceInfo:
    """Parse ptxas -v output for resource usage."""
    info = PTXResourceInfo()
    
    # Patterns for ptxas output (from actual output format)
    patterns = {
        'registers_per_thread': r'Used\s+(\d+)\s+registers',
        'shared_mem_per_block': r'(\d+)\s+bytes\s+(?:static\s+)?shared\s+memory\s+per\s+block',
        'local_mem_per_thread': r'(\d+)\s+bytes\s+local\s+memory\s+per\s+thread',
        'constant_mem_per_block': r'(\d+)\s+bytes\s+constant\s+memory',
        'spilled_registers': r'(\d+)\s+spilled\s+registers?',
        'spilled_local_mem': r'(\d+)\s+bytes\s+spill\s+(?:stores|loads)',
        'frame_size': r'(\d+)\s+bytes\s+stack\s+frame',
        'max_threads_per_block': r'max\s+threads\s+per\s+block\s*=\s*(\d+)',
        'max_blocks_per_sm': r'max\s+blocks\s+per\s+SM\s*=\s*(\d+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr, re.IGNORECASE)
        if match:
            setattr(info, key, int(match.group(1)))
    
    # Also check for gmem
    gmem_match = re.search(r'(\d+)\s+bytes\s+gmem', stderr, re.IGNORECASE)
    if gmem_match:
        info.shared_mem_per_block = int(gmem_match.group(1))
    
    # Occupancy info
    occ_patterns = {
        'occupancy_warps_per_sm': r'occupancy.*?(\d+)\s+warps?\s+per\s+SM',
        'occupancy_blocks_per_sm': r'occupancy.*?(\d+)\s+blocks?\s+per\s+SM',
        'achieved_occupancy': r'achieved\s+occupancy\s*[:=]\s*([\d.]+)',
    }
    
    for key, pattern in occ_patterns.items():
        match = re.search(pattern, stderr, re.IGNORECASE)
        if match:
            val = match.group(1)
            if key == 'achieved_occupancy':
                setattr(info, key, float(val))
            else:
                setattr(info, key, int(match.group(1)))
    
    return info


def tokenize_ptx(ptx_code: str) -> list[PTXToken]:
    """Tokenize PTX assembly code."""
    tokens = []
    lines = ptx_code.split('\n')
    
    # Base instruction mnemonics (suffixes like .u32, .f32, .b64 will be stripped)
    inst_map = {
        'mov': PTXTokenType.INST_MOV,
        'ld': PTXTokenType.INST_LD,
        'st': PTXTokenType.INST_ST,
        'add': PTXTokenType.INST_ADD,
        'sub': PTXTokenType.INST_SUB,
        'mul': PTXTokenType.INST_MUL,
        'fma': PTXTokenType.INST_FMA,
        'div': PTXTokenType.INST_DIV,
        'min': PTXTokenType.INST_MIN,
        'max': PTXTokenType.INST_MAX,
        'set': PTXTokenType.INST_SET,
        'setp': PTXTokenType.INST_SET,  # setp is set predicate
        'bra': PTXTokenType.INST_BRA,
        'call': PTXTokenType.INST_CALL,
        'ret': PTXTokenType.INST_RET,
        'sync': PTXTokenType.INST_SYNC,
        'bar': PTXTokenType.INST_BAR,
        'atom': PTXTokenType.INST_ATOM,
        'red': PTXTokenType.INST_RED,
        'vote': PTXTokenType.INST_VOTE,
        'shfl': PTXTokenType.INST_SHFL,
        'tex': PTXTokenType.INST_TEX,
        'suld': PTXTokenType.INST_SULD,
        'sust': PTXTokenType.INST_SUST,
        'sured': PTXTokenType.INST_SURED,
        'suatom': PTXTokenType.INST_SUATOM,
        'cvt': PTXTokenType.INST_CVT,
        'selp': PTXTokenType.INST_SELP,
        'slct': PTXTokenType.INST_SLCT,
        'lop': PTXTokenType.INST_LOP,
        'shf': PTXTokenType.INST_SHF,
        'pmevent': PTXTokenType.INST_PMEVENT,
        'brk': PTXTokenType.INST_BRK,
        'cont': PTXTokenType.INST_CONT,
        'exit': PTXTokenType.INST_EXIT,
        'jmp': PTXTokenType.INST_JMP,
        'jmx': PTXTokenType.INST_JMX,
        's2r': PTXTokenType.INST_S2R,
        'r2s': PTXTokenType.INST_R2S,
        'mov.matrix': PTXTokenType.INST_MOV_MATRIX,
        'ld.matrix': PTXTokenType.INST_LD_MATRIX,
        'mma': PTXTokenType.INST_MMA,
        'cp.async': PTXTokenType.INST_CP_ASYNC,
        'wait': PTXTokenType.INST_WAIT,
        'cluster': PTXTokenType.INST_CLUSTER,
        'mad': PTXTokenType.INST_FMA,  # mad is multiply-add
        'mul.wide': PTXTokenType.INST_MUL,
        'add.s64': PTXTokenType.INST_ADD,
        'cvta': PTXTokenType.INST_CVT,  # cvta is address conversion
    }
    
    directive_map = {
        '.version': PTXTokenType.DIR_VERSION,
        '.target': PTXTokenType.DIR_TARGET,
        '.address_size': PTXTokenType.DIR_ADDRESS_SIZE,
        '.entry': PTXTokenType.DIR_ENTRY,
        '.func': PTXTokenType.DIR_FUNC,
        '.param': PTXTokenType.DIR_PARAM,
        '.local': PTXTokenType.DIR_LOCAL,
        '.shared': PTXTokenType.DIR_SHARED,
        '.const': PTXTokenType.DIR_CONST,
        '.global': PTXTokenType.DIR_GLOBAL,
        '.visible': PTXTokenType.DIR_VISIBLE,
        '.align': PTXTokenType.DIR_ALIGN,
        '.maxnreg': PTXTokenType.DIR_MAXNREG,
        '.maxnctapersm': PTXTokenType.DIR_MAXNCTAPERSM,
        '.maxntid': PTXTokenType.DIR_MAXNTID,
        '.minnctapersm': PTXTokenType.DIR_MINNCTAPERSM,
    }
    
    state_spaces = {'.global', '.shared', '.local', '.const', '.param'}
    type_specs = {'.b8', '.b16', '.b32', '.b64', '.f16', '.f32', '.f64', '.u8', '.u16', '.u32', '.u64', '.s8', '.s16', '.s32', '.s64'}
    
    for line_num, line in enumerate(lines, 1):
        col = 1
        i = 0
        while i < len(line):
            ch = line[i]
            
            # Skip whitespace
            if ch in ' \t':
                col += 1
                i += 1
                continue
            
            # Comments
            if ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                comment = line[i:]
                tokens.append(PTXToken(PTXTokenType.COMMENT, comment, line_num, col))
                break
            
            # Strings
            if ch == '"':
                start = i
                i += 1
                col += 1
                while i < len(line) and line[i] != '"':
                    if line[i] == '\\' and i + 1 < len(line):
                        i += 2
                        col += 2
                    else:
                        i += 1
                        col += 1
                if i < len(line):
                    i += 1
                    col += 1
                tokens.append(PTXToken(PTXTokenType.STRING, line[start:i], line_num, col))
                continue
            
            # Numbers
            if ch.isdigit() or (ch == '-' and i + 1 < len(line) and line[i + 1].isdigit()):
                start = i
                is_float = False
                if ch == '-':
                    i += 1
                    col += 1
                while i < len(line) and line[i].isdigit():
                    i += 1
                    col += 1
                if i < len(line) and line[i] == '.':
                    is_float = True
                    i += 1
                    col += 1
                    while i < len(line) and line[i].isdigit():
                        i += 1
                        col += 1
                if i < len(line) and line[i] in 'eE':
                    is_float = True
                    i += 1
                    col += 1
                    if i < len(line) and line[i] in '+-':
                        i += 1
                        col += 1
                    while i < len(line) and line[i].isdigit():
                        i += 1
                        col += 1
                token_type = PTXTokenType.FLOAT if is_float else PTXTokenType.INTEGER
                tokens.append(PTXToken(token_type, line[start:i], line_num, col))
                continue
            
            # Identifiers, instructions, directives
            if ch.isalpha() or ch in '_$.':
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] in '_$.'):
                    i += 1
                    col += 1
                value = line[start:i]
                
                # Check for directives
                if value in directive_map:
                    tokens.append(PTXToken(directive_map[value], value, line_num, col - len(value)))
                elif value in state_spaces:
                    tokens.append(PTXToken(PTXTokenType.STATE_SPACE, value, line_num, col - len(value)))
                elif value in type_specs:
                    tokens.append(PTXToken(PTXTokenType.TYPE_SPEC, value, line_num, col - len(value)))
                else:
                    # Check for instructions with type suffixes (e.g., ld.global.f32, add.f32, mov.u32)
                    inst_base = value.split('.')[0] if '.' in value else value
                    if inst_base in inst_map:
                        tokens.append(PTXToken(inst_map[inst_base], value, line_num, col - len(value)))
                    elif value.startswith('%'):
                        # Register or predicate
                        if value[1] in 'pt':
                            tokens.append(PTXToken(PTXTokenType.PREDICATE_REG, value, line_num, col - len(value)))
                        else:
                            tokens.append(PTXToken(PTXTokenType.REGISTER, value, line_num, col - len(value)))
                    elif value.startswith('@'):
                        # Label
                        tokens.append(PTXToken(PTXTokenType.LABEL, value, line_num, col - len(value)))
                    else:
                        tokens.append(PTXToken(PTXTokenType.IDENTIFIER, value, line_num, col - len(value)))
                    continue
            
            # Punctuation
            if ch in ',;:{}()[]':
                punct_map = {
                    ',': PTXTokenType.COMMA,
                    ';': PTXTokenType.SEMICOLON,
                    ':': PTXTokenType.COLON,
                    '{': PTXTokenType.LBRACE,
                    '}': PTXTokenType.RBRACE,
                    '(': PTXTokenType.LPAREN,
                    ')': PTXTokenType.RPAREN,
                    '[': PTXTokenType.LBRACKET,
                    ']': PTXTokenType.RBRACKET,
                }
                tokens.append(PTXToken(punct_map[ch], ch, line_num, col))
                i += 1
                col += 1
                continue
            
            # Operators
            if ch in '+-*/%&|^<>!=':
                start = i
                while i < len(line) and line[i] in '+-*/%&|^<>!=':
                    i += 1
                    col += 1
                tokens.append(PTXToken(PTXTokenType.UNKNOWN, line[start:i], line_num, col))
                continue
            
            # Unknown
            i += 1
            col += 1
        
        tokens.append(PTXToken(PTXTokenType.NEWLINE, '\n', line_num, col))
    
    tokens.append(PTXToken(PTXTokenType.EOF, '', len(lines) + 1, 1))
    return tokens


def tokenize_sass(sass_code: str) -> list[PTXToken]:
    """Tokenize SASS assembly (simplified - treats as generic assembly)."""
    tokens = []
    lines = sass_code.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        col = 1
        i = 0
        while i < len(line):
            ch = line[i]
            
            if ch in ' \t':
                col += 1
                i += 1
                continue
            
            if ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                comment = line[i:]
                tokens.append(PTXToken(PTXTokenType.COMMENT, comment, line_num, col))
                break
            
            # Hex addresses/offsets
            if ch == '0' and i + 1 < len(line) and line[i + 1] in 'xX':
                start = i
                i += 2
                col += 2
                while i < len(line) and line[i] in '0123456789abcdefABCDEF':
                    i += 1
                    col += 1
                tokens.append(PTXToken(PTXTokenType.INTEGER, line[start:i], line_num, col))
                continue
            
            # Numbers
            if ch.isdigit():
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] in 'xX'):
                    i += 1
                    col += 1
                tokens.append(PTXToken(PTXTokenType.INTEGER, line[start:i], line_num, col))
                continue
            
            # Identifiers/opcodes
            if ch.isalpha() or ch in '_$.':
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] in '_$.'):
                    i += 1
                    col += 1
                value = line[start:i]
                tokens.append(PTXToken(PTXTokenType.IDENTIFIER, value, line_num, col - len(value)))
                continue
            
            # Registers (R0, R1, etc.)
            if ch == 'R' and i + 1 < len(line) and line[i + 1].isdigit():
                start = i
                i += 1
                col += 1
                while i < len(line) and line[i].isdigit():
                    i += 1
                    col += 1
                tokens.append(PTXToken(PTXTokenType.REGISTER, line[start:i], line_num, col))
                continue
            
            # Punctuation
            if ch in ',;:{}()[]':
                i += 1
                col += 1
                continue
            
            i += 1
            col += 1
        
        tokens.append(PTXToken(PTXTokenType.NEWLINE, '\n', line_num, col))
    
    tokens.append(PTXToken(PTXTokenType.EOF, '', len(lines) + 1, 1))
    return tokens


def extract_ptx_features(ptx_code: str, resource_info: PTXResourceInfo) -> dict:
    """Extract features from PTX code for model input."""
    tokens = tokenize_ptx(ptx_code)
    
    features = {
        'num_instructions': 0,
        'num_memory_instructions': 0,
        'num_arithmetic_instructions': 0,
        'num_control_instructions': 0,
        'num_sync_instructions': 0,
        'num_tensor_core_instructions': 0,
        'num_registers_used': resource_info.registers_per_thread,
        'shared_mem_bytes': resource_info.shared_mem_per_block,
        'local_mem_bytes': resource_info.local_mem_per_thread,
        'spilled_registers': resource_info.spilled_registers,
        'spilled_local_mem': resource_info.spilled_local_mem,
        'frame_size': resource_info.frame_size,
    }
    
    inst_types = set()
    for token in tokens:
        if token.type.name.startswith('INST_'):
            features['num_instructions'] += 1
            inst_types.add(token.type.name)
            
            if token.type.name in ('INST_LD', 'INST_ST', 'INST_SULD', 'INST_SUST'):
                features['num_memory_instructions'] += 1
            elif token.type.name in ('INST_ADD', 'INST_SUB', 'INST_MUL', 'INST_FMA', 'INST_DIV',
                                     'INST_MIN', 'INST_MAX', 'INST_CVT', 'INST_LOP', 'INST_SHF'):
                features['num_arithmetic_instructions'] += 1
            elif token.type.name in ('INST_BRA', 'INST_CALL', 'INST_RET', 'INST_JMP', 'INST_JMX',
                                     'INST_BRK', 'INST_CONT', 'INST_EXIT'):
                features['num_control_instructions'] += 1
            elif token.type.name in ('INST_SYNC', 'INST_BAR', 'INST_VOTE', 'INST_SHFL'):
                features['num_sync_instructions'] += 1
            elif token.type.name in ('INST_MMA', 'INST_MOV_MATRIX', 'INST_LD_MATRIX'):
                features['num_tensor_core_instructions'] += 1
    
    features['unique_instruction_types'] = len(inst_types)
    
    return features


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
    print("=== Compiling to PTX ===")
    ptx, stderr = compile_to_ptx(test_source)
    if ptx:
        print("PTX (first 50 lines):")
        for line in ptx.split('\n')[:50]:
            print(line)
    else:
        print(f"PTX compilation failed: {stderr}")
    
    print("\n=== PTXAS Resource Info ===")
    info = get_ptxas_info(test_source)
    print(f"Registers/thread: {info.registers_per_thread}")
    print(f"Shared mem/block: {info.shared_mem_per_block}")
    print(f"Local mem/thread: {info.local_mem_per_thread}")
    print(f"Spilled regs: {info.spilled_registers}")
    
    print("\n=== PTX Tokenization ===")
    if ptx:
        tokens = tokenize_ptx(ptx)
        print(f"Total tokens: {len(tokens)}")
        for t in tokens[:30]:
            if t.type not in (PTXTokenType.NEWLINE, PTXTokenType.COMMENT):
                print(f"  {t}")
        
        print("\n=== PTX Features ===")
        ptx_features = extract_ptx_features(ptx, info)
        for k, v in ptx_features.items():
            print(f"  {k}: {v}")
    
    print("\n=== Compiling to SASS ===")
    sass, stderr = compile_to_sass(test_source)
    if sass:
        print("SASS (first 30 lines):")
        for line in sass.split('\n')[:30]:
            print(line)
    else:
        print(f"SASS compilation failed: {stderr}")
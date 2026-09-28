"""Lexical CUDA source tokenizer with identifier/literal fallback encoding.

Adapted from cclm's lexical C tokenizer approach for CUDA C++ syntax.
Handles kernel launch triple-chevron syntax, __global__/__shared__/__device__
qualifiers, built-in variables like threadIdx/blockIdx.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Optional


class TokenType(Enum):
    """Token types for CUDA C++ lexical analysis."""
    # Keywords
    GLOBAL = auto()
    DEVICE = auto()
    SHARED = auto()
    CONSTANT = auto()
    HOST = auto()
    INLINE = auto()
    RESTRICT = auto()
    NOINLINE = auto()
    LAUNCH_BOUNDS = auto()
    
    # Built-in variables
    THREAD_IDX = auto()
    BLOCK_IDX = auto()
    BLOCK_DIM = auto()
    GRID_DIM = auto()
    WARP_SIZE = auto()
    
    # CUDA-specific syntax
    LAUNCH_CHEVRON_LEFT = auto()    # <<<
    LAUNCH_CHEVRON_RIGHT = auto()   # >>>
    
    # Standard C/C++ keywords
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    GOTO = auto()
    SIZEOF = auto()
    TYPEDEF = auto()
    STRUCT = auto()
    UNION = auto()
    ENUM = auto()
    CONST = auto()
    VOLATILE = auto()
    STATIC = auto()
    EXTERN = auto()
    AUTO = auto()
    REGISTER = auto()
    VOID = auto()
    CHAR = auto()
    SHORT = auto()
    INT = auto()
    LONG = auto()
    FLOAT = auto()
    DOUBLE = auto()
    SIGNED = auto()
    UNSIGNED = auto()
    BOOL = auto()
    COMPLEX = auto()
    IMAGINARY = auto()
    
    # Operators and punctuation
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AMPERSAND = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    EXCLAMATION = auto()
    EQUAL = auto()
    PLUS_EQUAL = auto()
    MINUS_EQUAL = auto()
    STAR_EQUAL = auto()
    SLASH_EQUAL = auto()
    PERCENT_EQUAL = auto()
    AMPERSAND_EQUAL = auto()
    PIPE_EQUAL = auto()
    CARET_EQUAL = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    LEFT_SHIFT_EQUAL = auto()
    RIGHT_SHIFT_EQUAL = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()
    EQUAL_EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    ARROW = auto()
    DOT = auto()
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()
    QUESTION = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    ELLIPSIS = auto()
    HASH = auto()
    HASH_HASH = auto()
    
    # Literals
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    CHAR_LITERAL = auto()
    STRING_LITERAL = auto()
    
    # Identifiers
    IDENTIFIER = auto()
    
    # Preprocessor
    PP_INCLUDE = auto()
    PP_DEFINE = auto()
    PP_IF = auto()
    PP_IFDEF = auto()
    PP_IFNDEF = auto()
    PP_ELSE = auto()
    PP_ELIF = auto()
    PP_ENDIF = auto()
    PP_UNDEF = auto()
    PP_PRAGMA = auto()
    PP_ERROR = auto()
    PP_WARNING = auto()
    PP_LINE = auto()
    
    # Special
    NEWLINE = auto()
    WHITESPACE = auto()
    COMMENT = auto()
    EOF = auto()
    UNKNOWN = auto()


# CUDA-specific keywords
CUDA_KEYWORDS = {
    '__global__': TokenType.GLOBAL,
    '__device__': TokenType.DEVICE,
    '__shared__': TokenType.SHARED,
    '__constant__': TokenType.CONSTANT,
    '__host__': TokenType.HOST,
    '__inline__': TokenType.INLINE,
    '__restrict__': TokenType.RESTRICT,
    '__noinline__': TokenType.NOINLINE,
    '__launch_bounds__': TokenType.LAUNCH_BOUNDS,
}

# CUDA built-in variables
CUDA_BUILTINS = {
    'threadIdx': TokenType.THREAD_IDX,
    'blockIdx': TokenType.BLOCK_IDX,
    'blockDim': TokenType.BLOCK_DIM,
    'gridDim': TokenType.GRID_DIM,
    'warpSize': TokenType.WARP_SIZE,
}

# Standard C/C++ keywords
C_KEYWORDS = {
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'for': TokenType.FOR,
    'while': TokenType.WHILE,
    'do': TokenType.DO,
    'switch': TokenType.SWITCH,
    'case': TokenType.CASE,
    'default': TokenType.DEFAULT,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'return': TokenType.RETURN,
    'goto': TokenType.GOTO,
    'sizeof': TokenType.SIZEOF,
    'typedef': TokenType.TYPEDEF,
    'struct': TokenType.STRUCT,
    'union': TokenType.UNION,
    'enum': TokenType.ENUM,
    'const': TokenType.CONST,
    'volatile': TokenType.VOLATILE,
    'static': TokenType.STATIC,
    'extern': TokenType.EXTERN,
    'auto': TokenType.AUTO,
    'register': TokenType.REGISTER,
    'void': TokenType.VOID,
    'char': TokenType.CHAR,
    'short': TokenType.SHORT,
    'int': TokenType.INT,
    'long': TokenType.LONG,
    'float': TokenType.FLOAT,
    'double': TokenType.DOUBLE,
    'signed': TokenType.SIGNED,
    'unsigned': TokenType.UNSIGNED,
    'bool': TokenType.BOOL,
    'complex': TokenType.COMPLEX,
    'imaginary': TokenType.IMAGINARY,
    '__builtin_va_list': TokenType.IDENTIFIER,
    '__builtin_va_start': TokenType.IDENTIFIER,
    '__builtin_va_end': TokenType.IDENTIFIER,
    '__builtin_va_arg': TokenType.IDENTIFIER,
    'NULL': TokenType.IDENTIFIER,
}

ALL_KEYWORDS = {**CUDA_KEYWORDS, **CUDA_BUILTINS, **C_KEYWORDS}


@dataclass
class Token:
    """A single token with type, value, and position info."""
    type: TokenType
    value: str
    line: int
    column: int
    offset: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


class CUDATokenizer:
    """Lexical tokenizer for CUDA C++ source code."""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
    
    def tokenize(self) -> list[Token]:
        """Tokenize the entire source and return list of tokens."""
        self.tokens = []
        while self.pos < len(self.source):
            token = self._next_token()
            if token:
                self.tokens.append(token)
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column, self.pos))
        return self.tokens
    
    def _next_token(self) -> Optional[Token]:
        """Get the next token from source."""
        if self.pos >= len(self.source):
            return None
        
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        ch = self.source[self.pos]
        
        # Whitespace
        if ch in ' \t\r\f\v':
            return self._scan_whitespace()
        
        # Newline
        if ch == '\n':
            self.pos += 1
            self.line += 1
            self.column = 1
            return Token(TokenType.NEWLINE, '\n', start_line, start_col, start_pos)
        
        # Comments
        if ch == '/':
            if self.pos + 1 < len(self.source):
                next_ch = self.source[self.pos + 1]
                if next_ch == '/':
                    return self._scan_line_comment()
                elif next_ch == '*':
                    return self._scan_block_comment()
        
        # Preprocessor
        if ch == '#' and self.column == 1:
            return self._scan_preprocessor()
        
        # String literal
        if ch == '"':
            return self._scan_string()
        
        # Char literal
        if ch == "'":
            return self._scan_char()
        
        # Numbers
        if ch.isdigit() or (ch == '.' and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
            return self._scan_number()
        
        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            return self._scan_identifier()
        
        # Operators and punctuation
        return self._scan_operator()
    
    def _scan_whitespace(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\r\f\v':
            self.pos += 1
            self.column += 1
        return Token(TokenType.WHITESPACE, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_line_comment(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        self.pos += 2
        self.column += 2
        while self.pos < len(self.source) and self.source[self.pos] != '\n':
            self.pos += 1
            self.column += 1
        return Token(TokenType.COMMENT, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_block_comment(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        self.pos += 2
        self.column += 2
        while self.pos + 1 < len(self.source):
            if self.source[self.pos] == '*' and self.source[self.pos + 1] == '/':
                self.pos += 2
                self.column += 2
                break
            if self.source[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
        return Token(TokenType.COMMENT, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_preprocessor(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        self.pos += 1
        self.column += 1
        
        # Skip whitespace after #
        while self.pos < len(self.source) and self.source[self.pos] in ' \t':
            self.pos += 1
            self.column += 1
        
        # Read directive name
        directive_start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalpha() or self.source[self.pos] == '_'):
            self.pos += 1
            self.column += 1
        directive = self.source[directive_start:self.pos]
        
        # Skip rest of line
        while self.pos < len(self.source) and self.source[self.pos] != '\n':
            self.pos += 1
            self.column += 1
        
        pp_type = {
            'include': TokenType.PP_INCLUDE,
            'define': TokenType.PP_DEFINE,
            'if': TokenType.PP_IF,
            'ifdef': TokenType.PP_IFDEF,
            'ifndef': TokenType.PP_IFNDEF,
            'else': TokenType.PP_ELSE,
            'elif': TokenType.PP_ELIF,
            'endif': TokenType.PP_ENDIF,
            'undef': TokenType.PP_UNDEF,
            'pragma': TokenType.PP_PRAGMA,
            'error': TokenType.PP_ERROR,
            'warning': TokenType.PP_WARNING,
            'line': TokenType.PP_LINE,
        }.get(directive, TokenType.UNKNOWN)
        
        return Token(pp_type, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_string(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        self.pos += 1
        self.column += 1
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == '\n':
                break
            if ch == '\\' and self.pos + 1 < len(self.source):
                self.pos += 2
                self.column += 2
                continue
            if ch == '"':
                self.pos += 1
                self.column += 1
                break
            self.pos += 1
            self.column += 1
        return Token(TokenType.STRING_LITERAL, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_char(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        self.pos += 1
        self.column += 1
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == '\n':
                break
            if ch == '\\' and self.pos + 1 < len(self.source):
                self.pos += 2
                self.column += 2
                continue
            if ch == "'":
                self.pos += 1
                self.column += 1
                break
            self.pos += 1
            self.column += 1
        return Token(TokenType.CHAR_LITERAL, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_number(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        is_float = False
        
        # Integer part
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1
            self.column += 1
        
        # Fractional part
        if self.pos < len(self.source) and self.source[self.pos] == '.':
            is_float = True
            self.pos += 1
            self.column += 1
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
                self.column += 1
        
        # Exponent
        if self.pos < len(self.source) and self.source[self.pos] in 'eE':
            is_float = True
            self.pos += 1
            self.column += 1
            if self.pos < len(self.source) and self.source[self.pos] in '+-':
                self.pos += 1
                self.column += 1
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
                self.column += 1
        
        # Float suffix
        if self.pos < len(self.source) and self.source[self.pos] in 'fFlL':
            is_float = True
            self.pos += 1
            self.column += 1
        
        # Integer suffix
        while self.pos < len(self.source) and self.source[self.pos] in 'uUlL':
            self.pos += 1
            self.column += 1
        
        token_type = TokenType.FLOAT_LITERAL if is_float else TokenType.INTEGER_LITERAL
        return Token(token_type, self.source[start_pos:self.pos], start_line, start_col, start_pos)
    
    def _scan_identifier(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self.pos += 1
            self.column += 1
        value = self.source[start_pos:self.pos]
        token_type = ALL_KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, start_line, start_col, start_pos)
    
    def _scan_operator(self) -> Token:
        start_pos = self.pos
        start_line = self.line
        start_col = self.column
        
        # Two-char operators (check longest first)
        two_char_ops = {
            '<<=': TokenType.LEFT_SHIFT_EQUAL,
            '>>=': TokenType.RIGHT_SHIFT_EQUAL,
            '<<<': TokenType.LAUNCH_CHEVRON_LEFT,   # CUDA kernel launch
            '>>>' : TokenType.LAUNCH_CHEVRON_RIGHT,  # CUDA kernel launch end
            '->': TokenType.ARROW,
            '++': TokenType.PLUS,  # Will be handled as PLUS + PLUS in practice
            '--': TokenType.MINUS,
            '+=': TokenType.PLUS_EQUAL,
            '-=': TokenType.MINUS_EQUAL,
            '*=': TokenType.STAR_EQUAL,
            '/=': TokenType.SLASH_EQUAL,
            '%=': TokenType.PERCENT_EQUAL,
            '&=': TokenType.AMPERSAND_EQUAL,
            '|=': TokenType.PIPE_EQUAL,
            '^=': TokenType.CARET_EQUAL,
            '==': TokenType.EQUAL_EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '>=': TokenType.GREATER_EQUAL,
            '&&': TokenType.LOGICAL_AND,
            '||': TokenType.LOGICAL_OR,
            '##': TokenType.HASH_HASH,
            '<<': TokenType.LEFT_SHIFT,
            '>>': TokenType.RIGHT_SHIFT,
        }
        
        # Check 3-char first (<<< >>>)
        if self.pos + 3 <= len(self.source):
            three_char = self.source[self.pos:self.pos + 3]
            if three_char in ('<<<', '>>>'):
                self.pos += 3
                self.column += 3
                return Token(
                    TokenType.LAUNCH_CHEVRON_LEFT if three_char == '<<<' else TokenType.LAUNCH_CHEVRON_RIGHT,
                    three_char, start_line, start_col, start_pos
                )
        
        # Check 2-char
        if self.pos + 2 <= len(self.source):
            two_char = self.source[self.pos:self.pos + 2]
            if two_char in two_char_ops:
                self.pos += 2
                self.column += 2
                return Token(two_char_ops[two_char], two_char, start_line, start_col, start_pos)
        
        # Single char
        ch = self.source[self.pos]
        single_char_ops = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '%': TokenType.PERCENT,
            '&': TokenType.AMPERSAND,
            '|': TokenType.PIPE,
            '^': TokenType.CARET,
            '~': TokenType.TILDE,
            '!': TokenType.EXCLAMATION,
            '=': TokenType.EQUAL,
            '<': TokenType.LESS,
            '>': TokenType.GREATER,
            '.': TokenType.DOT,
            ',': TokenType.COMMA,
            ';': TokenType.SEMICOLON,
            ':': TokenType.COLON,
            '?': TokenType.QUESTION,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '#': TokenType.HASH,
        }
        
        if ch in single_char_ops:
            self.pos += 1
            self.column += 1
            return Token(single_char_ops[ch], ch, start_line, start_col, start_pos)
        
        # Unknown character
        self.pos += 1
        self.column += 1
        return Token(TokenType.UNKNOWN, ch, start_line, start_col, start_pos)


def tokenize(source: str) -> list[Token]:
    """Convenience function to tokenize CUDA source."""
    return CUDATokenizer(source).tokenize()


def tokenize_for_model(source: str, max_length: int = 2048) -> list[int]:
    """Tokenize and encode for model input with identifier/literal fallback.
    
    Returns list of token IDs suitable for model input.
    Unknown identifiers/literals map to special fallback tokens.
    """
    tokens = tokenize(source)
    token_ids = []
    
    # Simple vocabulary mapping (in practice, use a proper vocab file)
    vocab = {
        TokenType.GLOBAL: 1,
        TokenType.DEVICE: 2,
        TokenType.SHARED: 3,
        TokenType.CONSTANT: 4,
        TokenType.HOST: 5,
        TokenType.INLINE: 6,
        TokenType.RESTRICT: 7,
        TokenType.NOINLINE: 8,
        TokenType.LAUNCH_BOUNDS: 9,
        TokenType.THREAD_IDX: 10,
        TokenType.BLOCK_IDX: 11,
        TokenType.BLOCK_DIM: 12,
        TokenType.GRID_DIM: 13,
        TokenType.WARP_SIZE: 14,
        TokenType.LAUNCH_CHEVRON_LEFT: 15,
        TokenType.LAUNCH_CHEVRON_RIGHT: 16,
        TokenType.LPAREN: 17,
        TokenType.RPAREN: 17,
        TokenType.LBRACE: 18,
        TokenType.RBRACE: 18,
        TokenType.LBRACKET: 19,
        TokenType.RBRACKET: 19,
        TokenType.SEMICOLON: 20,
        TokenType.COMMA: 21,
        TokenType.EQUAL: 22,
        TokenType.PLUS: 23,
        TokenType.MINUS: 24,
        TokenType.STAR: 25,
        TokenType.SLASH: 26,
        TokenType.LESS: 27,
        TokenType.GREATER: 28,
        TokenType.IDENTIFIER: 29,  # Fallback for unknown identifiers
        TokenType.INTEGER_LITERAL: 30,
        TokenType.FLOAT_LITERAL: 31,
        TokenType.STRING_LITERAL: 32,
        TokenType.CHAR_LITERAL: 33,
        TokenType.IF: 34,
        TokenType.ELSE: 35,
        TokenType.FOR: 36,
        TokenType.WHILE: 37,
        TokenType.RETURN: 38,
        TokenType.INT: 39,
        TokenType.FLOAT: 40,
        TokenType.VOID: 41,
        TokenType.CONST: 42,
    }
    
    fallback_id = vocab[TokenType.IDENTIFIER]  # Fallback for unknown
    literal_fallback = vocab[TokenType.INTEGER_LITERAL]  # Fallback for literals
    
    for token in tokens:
        if token.type in (TokenType.WHITESPACE, TokenType.NEWLINE, TokenType.COMMENT, TokenType.EOF):
            continue
        if len(token_ids) >= max_length:
            break
        
        if token.type == TokenType.IDENTIFIER:
            # Check if it's a known keyword/builtin
            if token.value in ALL_KEYWORDS:
                token_ids.append(vocab.get(ALL_KEYWORDS[token.value], fallback_id))
            else:
                token_ids.append(fallback_id)  # Unknown identifier -> fallback
        elif token.type in (TokenType.INTEGER_LITERAL, TokenType.FLOAT_LITERAL):
            token_ids.append(literal_fallback)  # Literal fallback
        elif token.type in vocab:
            token_ids.append(vocab[token.type])
        else:
            token_ids.append(fallback_id)
    
    return token_ids


if __name__ == '__main__':
    # Demo
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
    tokens = tokenize(test_source)
    for t in tokens:
        if t.type not in (TokenType.WHITESPACE, TokenType.NEWLINE, TokenType.COMMENT):
            print(t)
    
    print("\n--- Model token IDs ---")
    model_tokens = tokenize_for_model(test_source)
    print(model_tokens)
    print(f"Length: {len(model_tokens)}")
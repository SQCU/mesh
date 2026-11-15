# design/compilation-composition-change-request.md, numerical lowerer boundary and "Complete replacement";
# design/torch-source-lowering-change-request.md items 5 and 6: the explicitly selected custom numerical methods (the
# CPU LAPACK drivers, and on the placed processor QDWH polar, eigh and SVD, the pseudoinverse and least-squares
# drivers, the eigenvalue drivers and the matrix exponential), their custom-operator metadata, their native records
# (metal-microbench metal_ir.h metal_ir_generic, metal_ir_lapack, metal_ir_repeat, metal_ir_switch and the region
# builder) and the program generator's state. Every other operation is the installed upstream compiler's (METHODS).
# Each definition carries its citation; this module's earlier per-operator construction is in its git history
import collections
import functools
import itertools
import math
import pathlib
import sys

import torch
from torch._higher_order_ops.cond import cond_op
from torch._higher_order_ops.switch import switch_op
from torch._higher_order_ops.while_loop import while_loop_op
from torch._prims_common import corresponding_complex_dtype, get_computation_dtype
from torch.fx.graph_module import _get_attr

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_code

# torch 2.14.0 torch/export/__init__.py, export; torch/_ops.py, OpOverload
A = torch.ops.aten


def _ints(value, k):
    value = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return value * k if len(value) == 1 else value


# design/compilation-composition-change-request.md, "Complete replacement workgroup", numerical operations and kernel
# instructions; metal-microbench metal_ir.h (metal_ir_value, metal_ir_generic, metal_ir_states and the region builder);
# MLIR Linalg, linalg.generic; N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1: every static operator,
# dtype, shape, stride and scalar operand of a Torch call is an instruction, literal or map coefficient of its
# structured operation, never a metadata vector a generic kernel interprets
BOOL, I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64 = 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16
KINDS = dict(zip((BOOL, I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64),
                 ('bool', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'f16', 'bf16', 'f32', 'f64')))


SCALAR = {torch.bool: BOOL, torch.int8: I8, torch.uint8: U8, torch.int16: I16, torch.uint16: U16, torch.int32: I32,
          torch.uint32: U32, torch.int64: I64, torch.uint64: U64, torch.float16: F16, torch.bfloat16: BF16,
          torch.float32: F32, torch.float64: F64}
# metal_ir.h ComputeType: the kind a region computes a stored kind in
CLASS = {BOOL: BOOL, I8: I32, I16: I32, I32: I32, U8: U32, U16: U32, U32: U32, I64: I64, U64: U64, F16: F16, BF16: F32,
         F32: F32, F64: F64}
STORED = {BOOL: torch.bool, I32: torch.int32, U32: torch.uint32, I64: torch.int64, U64: torch.uint64, F16: torch.float16,
        F32: torch.float32, F64: torch.float64}
REAL = tuple(SCALAR)
FLOATING = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
INTEGRAL = tuple(dtype for dtype in REAL if dtype not in FLOATING)
# metal_ir.h: linalg iterator kinds, combiners, starts, precision, placement, Tint core::BinaryOp and core::UnaryOp
PARALLEL, REDUCTION = 0, 1
ASSIGN, SUM, REGION = 0, 1, 9
IDENTITY, ACCUMULATE = 0, 1
REASSOCIATE, CONTRACT = 1, 2
GPU, CPU, ANE = 0, 1, 2
BINARY = {'and': 0, 'or': 1, 'xor': 2, 'eq': 5, 'ne': 6, 'lt': 7, 'gt': 8, 'le': 9, 'ge': 10, 'shl': 11, 'shr': 12,
          'add': 13, 'sub': 14, 'mul': 15, 'div': 16, 'rem': 17}
COMPARE = ('eq', 'ne', 'lt', 'gt', 'le', 'ge')
BOOLEAN = COMPARE + ('isnan', 'isinf', 'logical_and', 'logical_or', 'logical_xor', 'logical_not')
COMPLEMENT, NEGATION, NOT = 1, 3, 4
# WGSL builtin functions by name (metal_ir_builtin_named, Tint core::ParseBuiltinFn)
BUILTINS = ('exp', 'log', 'sqrt', 'tanh', 'abs', 'trunc', 'floor', 'ceil', 'round', 'pow', 'sin', 'cos', 'tan', 'asin',
            'acos', 'atan', 'atan2', 'sinh', 'cosh', 'asinh', 'acosh', 'atanh', 'exp2', 'log2')


def _class(value):
    return CLASS[SCALAR[value.dtype]]


def _floating(kind):
    return kind in (F16, F32, F64)


# metal_ir.h metal_ir_literal: two's complement, or the IEEE binary16 / binary32 bits in the low bits
def _bits(value, kind):
    if kind == BOOL:
        return int(bool(value))
    if _floating(kind):
        dtype, word, bits = {F16: (torch.float16, torch.int16, 16), F32: (torch.float32, torch.int32, 32),
                             F64: (torch.float64, torch.int64, 64)}[kind]
        return torch.tensor(float(value), dtype=dtype).view(word).item() & ((1 << bits) - 1)
    return int(value) & ((1 << 64) - 1)


# a region's builder calls in order, each (call, operands...) of metal_ir.h's builder for the region it is replayed
# into, with the kind of each result; compositions (erf, floor division, isnan, integer powers) expand here into the
# builder's binary/unary/convert/bitcast/builtin calls
class _Region:
    def __init__(self):
        self.code, self.kinds = [], []

    def add(self, record, kind):
        self.code.append(record)
        self.kinds.append(kind)
        return len(self.code) - 1

    def __call__(self, name, kind, *x, integer=0):
        if name in BINARY:
            return self.add(('binary', BINARY[name], *x), BOOL if name in COMPARE else kind)
        if name == 'neg':
            return self.add(('unary', NEGATION, x[0]), kind)
        if name == 'not':
            return self.add(('unary', NOT, x[0]), BOOL)
        if name == 'complement':
            return self.add(('unary', COMPLEMENT, x[0]), kind)
        if name in BUILTINS:
            return self.add(('builtin', name, x), kind)
        if name == 'select':
            return self.add(('builtin', 'select', (x[2], x[1], x[0])), kind)
        if name == 'cast':
            return self.add(('convert', kind, x[0]), kind)
        if name == 'bitcast':
            return self.add(('bitcast', kind, x[0]), kind)
        if name == 'arg':
            return self.add(('parameter', integer), kind)
        if name == 'extract':
            return self.add(('extract', integer, x), kind)
        if name == 'isnan':
            return self('ne', BOOL, x[0], x[0])
        if name == 'erf':
            return _erf(self, x[0], kind)
        if name == 'floordiv':
            return _floordiv(self, x[0], x[1], kind)
        if name == 'truncdiv':
            return _truncdiv(self, x[0], x[1], kind)
        if name == 'powi':
            return _powi(self, x[0], x[1], kind)
        if name == 'yield':
            return self.add(('yield', x), BOOL)
        raise KeyError(name)

    # WGSL §6.2.1 (and Tint core constants) hold finite values only: an infinity or NaN is its binary32 pattern's bitcast
    def const(self, value, kind):
        if _floating(kind) and not math.isfinite(value):
            made = self.add(('bitcast', F32, self.add(('literal', U32, _bits(value, F32)), U32)), F32)
            return made if kind == F32 else self('cast', kind, made)
        return self.add(('literal', kind, _bits(value, kind)), kind)

    def arg(self, k, value):
        return self('arg', _class(value), integer=k)

    def index(self, d):
        return self.add(('convert', I32, self.add(('loop', d), U32)), I32)

    def done(self, *results):
        self('yield', BOOL, *results)
        return self

    def scatter(self, output, indices, value, predicate):
        self.add(('scatter', output, tuple(indices), value, predicate), BOOL)

    def extract_if(self, input, kind, indices, predicate, fallback):
        return self.add(('extract_if', input, tuple(indices), predicate, fallback), kind)


# N. Juffa, Stack Overflow answer 35148199, 2016; torch 2.14.0 c10/metal/special_math.h, erf
def _erf(r, a, t):
    def c(v):
        return r.const(v, t)

    def fma(x, y, z):
        return r('add', t, r('mul', t, x, y), z)
    s, u = r('abs', t, a), r('mul', t, a, a)
    w = fma(c(-1.72853470e-5), s, c(3.83197126e-4))
    w = fma(w, u, fma(c(-3.88396438e-3), s, c(2.42546219e-2)))
    for k in (-1.06777877e-1, -6.34846687e-1, -1.28717512e-1):
        w = fma(w, s, c(k))
    w = fma(w, s, r('neg', t, s))
    far = r('sub', t, c(1), r('exp', t, w))
    far = r('select', t, r('lt', BOOL, a, c(0)), r('neg', t, far), far)
    q = c(-5.96761703e-4)
    for k in (4.99119423e-3, -2.67681349e-2, 1.12819925e-1, -3.76125336e-1, 1.28379166e-1):
        q = fma(q, u, c(k))
    return r('select', t, r('gt', BOOL, s, c(0.927734375)), far, fma(q, a, a))


# IEEE 754-2019 §5.5.1 copySign: a zero with the sign bit of x
def _signed_zero(r, x, t):
    if t == F16:
        x = r('cast', F32, x)
    sign = r('and', U32, r('bitcast', U32, x), r.const(0x80000000, U32))
    zero = r('bitcast', F32, sign)
    return r('cast', F16, zero) if t == F16 else zero


# torch 2.14.0 c10/util/generic_math.h, div_floor_floating and div_floor_integer
def _floordiv(r, a, b, t):
    def c(v):
        return r.const(v, t)
    if not _floating(t):
        safe = r('select', t, r('eq', BOOL, b, c(0)), c(1), b)
        q = r('div', t, a, safe)
        if t != U32:
            fix = r('and', BOOL, r('ne', BOOL, r('rem', t, a, safe), c(0)),
                    r('ne', BOOL, r('lt', BOOL, a, c(0)), r('lt', BOOL, b, c(0))))
            q = r('select', t, fix, r('sub', t, q, c(1)), q)
        return r('select', t, r('eq', BOOL, b, c(0)), c(0), q)
    mod = r('rem', t, a, b)
    div = r('div', t, r('sub', t, a, mod), b)
    fix = r('and', BOOL, r('ne', BOOL, mod, c(0)), r('ne', BOOL, r('lt', BOOL, b, c(0)), r('lt', BOOL, mod, c(0))))
    div = r('select', t, fix, r('sub', t, div, c(1)), div)
    floored = r('floor', t, div)
    rounded = r('select', t, r('gt', BOOL, r('sub', t, div, floored), c(0.5)), r('add', t, floored, c(1)), floored)
    quotient = r('div', t, a, b)
    zero = r('select', t, r('eq', BOOL, div, c(0)), _signed_zero(r, quotient, t), rounded)
    return r('select', t, r('eq', BOOL, b, c(0)), quotient, zero)


def _truncdiv(r, a, b, t):
    if _floating(t):
        return r('trunc', t, r('div', t, a, b))
    zero = r('eq', BOOL, b, r.const(0, t))
    return r('select', t, zero, r.const(0, t), r('div', t, a, r('select', t, zero, r.const(1, t), b)))


# torch 2.14.0 ATen/native/Pow.h, powi: exponentiation by squaring over the exponent's bits
def _powi(r, a, b, t):
    one, zero = r.const(1, t), r.const(0, t)
    result, base, rest = one, a, b
    for _ in range(64 if t in (I64, U64) else 32):
        odd = r('ne', BOOL, r('and', t, rest, one), zero)
        result = r('select', t, odd, r('mul', t, result, base), result)
        base = r('mul', t, base, base)
        rest = r('div', t, rest, r.const(2, t))
    if t in (U32, U64):
        return result
    odd = r('ne', BOOL, r('and', t, b, one), zero)
    unit = r('select', t, r('eq', BOOL, a, one), one,
             r('select', t, r('eq', BOOL, a, r.const(-1, t)), r('select', t, odd, r.const(-1, t), one), zero))
    return r('select', t, r('lt', BOOL, b, zero), unit, result)


# one structured operation, metal_ir_generic's operands: its loops `sizes` with `iterators`, inputs and outputs each
# (tensor view, affine rows over the loops and then the symbols, (loops + symbols + 1) coefficients a row; None: read
# only by metal_ir_extract), per output its combiner, starting value and accumulator kind, coupled REGION states (kind,
# identity) with their combine and finalize regions, and the permitted numerical transformations; metal_ir_symbol's
# `symbols` (one-element tensor, bound) in order, symbol k the rows' dimension len(sizes) + k, and per loop its
# metal_ir_loop_extent `extents`, the index of the symbol clipping it or None (its static size); `source_axes`
# (metal_ir.h metal_ir_operation_axis, compiler/mesh_compiler/lower_metal.py construct) each explicit correspondence its
# construction used of an original tensor's dimension to one of its operands' (original tensor, its dimension, operand
# ordinal, inputs first, operand dimension), of a source the record does not read; `component_views` each operand
# that is a complex tensor's real pairs (torch.view_as_real: its leading dimensions the tensor's, its last the
# component) as (that complex tensor, operand ordinal, inputs first), a memory view, not a shape correspondence
class _Generic:
    def __init__(self, sizes, inputs, outputs, body, iterators=None, combiner=None, accumulator=None, start=None,
                 states=None, combine=None, finalize=None, precision=0, symbols=None, extents=None, source_axes=None,
                 component_views=None):
        self.sizes, self.inputs, self.outputs, self.body = tuple(sizes), list(inputs), list(outputs), body
        self.source_axes = list(source_axes) if source_axes is not None else []
        self.component_views = list(component_views) if component_views is not None else []
        self.symbols = list(symbols) if symbols is not None else []
        self.extents = tuple(extents) if extents is not None else (None,) * len(self.sizes)
        self.iterators = tuple(iterators) if iterators is not None else (PARALLEL,) * len(self.sizes)
        self.combiner = list(combiner) if combiner is not None else [ASSIGN] * len(self.outputs)
        self.accumulator = list(accumulator) if accumulator is not None else [_class(t) for t, _ in self.outputs]
        self.start = list(start) if start is not None else [IDENTITY] * len(self.outputs)
        self.states, self.combine, self.finalize, self.precision = states, combine, finalize, precision
        self.place, self.node = GPU, 0


# design/torch-source-lowering-change-request.md item 6; metal-microbench metal_ir.h metal_ir_symbol and
# metal_ir_loop_extent: existing structured records over a block of runtime place and size, the records' own formulas
# unchanged. Each record gains `symbols` (one-element tensor, bound) after its own; `offsets` (operand tensor, {dim:
# symbol index}), matched by identity with the record's operands or through its `component_views` (an operand that is
# that tensor's real pairs, its leading dimensions the tensor's), adds that symbol to the operand's row of that
# dimension (a dynamic slice offset; the component row constant); `clip(record, loop)` the symbol index clipping the
# loop, or None. A record kind other than _Generic has no runtime extent of its own, so its bounded lowering is
# unfinished
def _on_block(records, symbols, offsets=(), clip=lambda record, loop: None):
    made = []
    for g in records:
        if not isinstance(g, _Generic):
            raise NotImplementedError('%s over a runtime block: its bounded lowering is unfinished '
                                      '(design/torch-source-lowering-change-request.md item 6)' % type(g).__name__)
        base, own = len(g.symbols), len(g.sizes) + len(g.symbols)

        def extended(operands, first):
            out = []
            for s, (t, rows) in enumerate(operands, first):
                if rows is not None:
                    shifted = dict(next((at for held, at in offsets if held is t), {}))
                    for source, ordinal in g.component_views:
                        if ordinal == s:
                            shifted.update(next((at for held, at in offsets if held is source), {}))
                    rows = tuple(tuple(row[:own]) + tuple(int(shifted.get(d) == k) for k in range(len(symbols)))
                                 + (row[own],) for d, row in enumerate(rows))
                out.append((t, rows))
            return out
        bounded = _Generic(g.sizes, extended(g.inputs, 0), extended(g.outputs, len(g.inputs)), g.body, g.iterators,
                           g.combiner,
                           g.accumulator, g.start, g.states, g.combine, g.finalize, g.precision,
                           [*g.symbols, *symbols],
                           [e if e is not None else None if clip(g, l) is None else base + clip(g, l)
                            for l, e in enumerate(g.extents)])
        bounded.place, bounded.node, bounded.source_axes = g.place, g.node, g.source_axes
        bounded.component_views = g.component_views
        made.append(bounded)
    return made


def _rows(dims, picks):
    made = []
    for pick in picks:
        row = [0] * (dims + 1)
        for d, c in ({} if pick is None else {pick: 1} if isinstance(pick, int) else pick).items():
            row[dims if d is None else d] += c
        made.append(tuple(row))
    return tuple(made)


def _same(dims, rank=None):
    return _rows(dims, range(dims if rank is None else rank))


def _aligned(value, shape):
    lead = len(shape) - value.dim()
    return _rows(len(shape), [None if n == 1 and shape[lead + d] != 1 else lead + d for d, n in enumerate(value.shape)])


def _temporary(shape, dtype=torch.float32):
    return torch.empty(tuple(shape), dtype=dtype, device='meta')


def _typed(target, operands, outs, results=FLOATING):
    found = [value.dtype for value in operands if isinstance(value, torch.Tensor) and value.dtype not in REAL] \
        + [value.dtype for value in outs if value.dtype not in results]
    # Rejection filters are not part of even temporary integration validation.
    if found:
        raise NotImplementedError('%s of %s: its lowering into the structured basis is unfinished '
                                  '(design/compilation-composition-change-request.md item 1)' % (target, found[0]))


# torch 2.14.0 aten/src/ATen/OpMathType.h opmath_type: an operation's arithmetic kind
def _compute(*dtypes):
    if any(d is torch.float64 for d in dtypes):
        return F64
    if any(d.is_floating_point for d in dtypes):
        return F32
    kinds = {SCALAR[d] for d in dtypes}
    return I64 if kinds & {I64, U64} else U32 if U32 in kinds else I32


# torch 2.14.0 c10/util/TypeCast.h static_cast_with_inter_type: to bool by comparison with zero, an integer to a reduced
# float through float
def _convert(r, x, source, target):
    if source == target:
        return x
    if target == BOOL:
        return r('ne', BOOL, x, r.const(0, source))
    if not _floating(source) and target == F16:
        x = r('cast', F32, x)
    return r('cast', target, x)


def _load(r, k, value, kind):
    return _convert(r, r.arg(k, value), _class(value), kind)


def _store(r, x, kind, value):
    return _convert(r, x, kind, _class(value))


def _operands(target, args, kwargs):
    schema = target._schema.arguments
    return {**{a.name: a.default_value for a in schema if a.has_default_value()},
            **dict(zip((a.name for a in schema), args)), **kwargs}


# PyTorch v2.11.0 aten/src/ATen/native/cpu/PowKernel.cpp, pow_tensor_scalar_optimized_kernel: a static exponent
# specialized into its instructions
def _power(r, x, y, t, exponent):
    one = r.const(1, t)
    if exponent == 2:
        return r('mul', t, x, x)
    if exponent == 3:
        return r('mul', t, r('mul', t, x, x), x)
    if exponent == -2:
        return r('div', t, one, r('mul', t, x, x))
    if exponent == 0.5:
        return r('sqrt', t, x)
    if exponent == -0.5:
        return r('div', t, one, r('sqrt', t, x))
    if exponent == -1:
        return r('div', t, one, x)
    return r('pow', t, x, y)


# PyTorch 2.14 documentation of each operation named; torch 2.14.0 c10/util/MathConstants.h M_SQRT1_2, M_2_SQRTPI
SQRT1_2, TWO_SQRTPI = 0.707106781186547524401, 1.12837916709551257390


def _apply(r, op, x, y, z, alpha, t, exponent=None):
    def c(v):
        return r.const(v, t)
    if op in ('add', 'sub'):
        return r(op, t, x, y if alpha == 1 else r('mul', t, c(alpha), y))
    if op in ('mul', 'div', 'floordiv', 'truncdiv'):
        return r(op, t, x, y)
    if op in ('neg', 'tanh', 'exp', 'sqrt', 'log', 'abs'):
        return r(op, t, x)
    if op == 'relu':
        return r('select', t, r('lt', BOOL, x, c(0)), c(0), x)
    if op == 'sigmoid':
        return r('div', t, c(1), r('add', t, c(1), r('exp', t, r('neg', t, x))))
    if op == 'silu':
        return r('div', t, x, r('add', t, c(1), r('exp', t, r('neg', t, x))))
    if op == 'gelu':
        return r('mul', t, r('mul', t, c(0.5), x), r('add', t, c(1), r('erf', t, r('mul', t, x, c(SQRT1_2)))))
    if op == 'gelu_tanh':
        inner = r('add', t, x, r('mul', t, c(0.044715), r('mul', t, r('mul', t, x, x), x)))
        return r('mul', t, r('mul', t, c(0.5), x),
                 r('add', t, c(1), r('tanh', t, r('mul', t, c(TWO_SQRTPI * SQRT1_2), inner))))
    if op == 'rsqrt':
        return r('div', t, c(1), r('sqrt', t, x))
    if op in COMPARE:
        return r(op, BOOL, x, y)
    if op == 'pow':
        return r('powi', t, x, y) if not _floating(t) else _power(r, x, y, t, exponent)
    if op == 'clamp':
        lowered = x if y is None else r('select', t, r('lt', BOOL, x, y), y, x)
        return lowered if z is None else r('select', t, r('lt', BOOL, z, lowered), z, lowered)
    if op in ('floor', 'ceil', 'round', 'trunc'):
        return x if not _floating(t) else r(op, t, x)
    if op in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'asinh', 'acosh', 'atanh', 'erf'):
        return r(op, t, x)
    if op == 'atan2':
        return r('atan2', t, x, y)
    if op == 'reciprocal':
        return r('div', t, c(1), x)
    if op in ('exp2', 'log2'):
        return r(op, t, x)
    if op == 'log10':
        return r('div', t, r('log', t, x), c(math.log(10)))
    # W. Kahan's log1p (D. Goldberg, ACM Comput. Surv. 23(1), 1991, Theorem 4) and expm1 by the same correction
    if op == 'log1p':
        u = r('add', t, c(1), x)
        corrected = r('mul', t, r('log', t, u), r('div', t, x, r('sub', t, u, c(1))))
        return r('select', t, r('eq', BOOL, u, c(1)), x,
                 r('select', t, r('eq', BOOL, u, c(math.inf)), u, corrected))
    if op == 'expm1':
        u = r('exp', t, x)
        difference = r('sub', t, u, c(1))
        corrected = r('mul', t, difference, r('div', t, x, r('log', t, u)))
        return r('select', t, r('eq', BOOL, u, c(1)), x,
                 r('select', t, r('eq', BOOL, u, c(math.inf)), u,
                   r('select', t, r('eq', BOOL, difference, c(-1)), c(-1), corrected)))
    if op == 'isnan':
        return r('isnan', BOOL, x)
    if op == 'isinf':
        return r('eq', BOOL, r('abs', t, x), c(math.inf))
    if op == 'sign':
        return r('sub', t, _convert(r, r('lt', BOOL, c(0), x), BOOL, t), _convert(r, r('lt', BOOL, x, c(0)), BOOL, t))
    if op in ('maximum', 'minimum'):
        better = r('select', t, r('gt' if op == 'maximum' else 'lt', BOOL, x, y), x, y)
        if not _floating(t):
            return better
        return r('select', t, r('isnan', BOOL, x), x, r('select', t, r('isnan', BOOL, y), y, better))
    if op == 'fmod':
        return r('rem', t, x, y if _floating(t) else r('select', t, r('eq', BOOL, y, c(0)), c(1), y))
    if op == 'remainder':
        m = r('rem', t, x, y if _floating(t) else r('select', t, r('eq', BOOL, y, c(0)), c(1), y))
        fix = r('and', BOOL, r('ne', BOOL, m, c(0)), r('ne', BOOL, r('lt', BOOL, m, c(0)), r('lt', BOOL, y, c(0))))
        return r('select', t, fix, r('add', t, m, y), m)
    if op in ('bitwise_and', 'bitwise_or', 'bitwise_xor'):
        return r(op[8:], t, x, y)
    if op == 'bitwise_not':
        return r('not', BOOL, x) if t == BOOL else r('complement', t, x)
    if op in ('logical_and', 'logical_or', 'logical_xor'):
        a, b = _convert(r, x, t, BOOL), _convert(r, y, t, BOOL)
        return r('ne', BOOL, a, b) if op == 'logical_xor' else r(op[8:], BOOL, a, b)
    if op == 'logical_not':
        return r('not', BOOL, _convert(r, x, t, BOOL))
    return r('select', t, r('ne', BOOL, x, c(0)), y, z)


# PyTorch 2.14 documentation, Broadcasting semantics and Type promotion: tensor operands broadcast by their maps' zero
# rows, scalars the region's literals
def _elementwise(op, out, a, b=None, c=None, alpha=1):
    if any(isinstance(x, torch.Tensor) and x.is_complex() or isinstance(x, complex) for x in (out, a, b, c)):
        if op == 'where':
            return [g for k in (0, 1) for g in _elementwise(op, _component(out, k), a,
                                                            _component(b, k), _component(c, k), alpha)]
        return _complex(op, out, a, b, alpha)
    given = [a, b, c]
    tensors = [x for x in given if isinstance(x, torch.Tensor)]
    integral = out.dtype in INTEGRAL and all(x.dtype in INTEGRAL if isinstance(x, torch.Tensor) else
                                             isinstance(x, (bool, int)) for x in given if x is not None)
    # Rejection filters are not part of even temporary integration validation.
    # alpha is the captured call's (_arithmetic), whose source dispatch checked it (torch/_refs add and sub)
    boolean_bits = op in ('bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not') and out.dtype is torch.bool \
        and all(x.dtype is torch.bool if isinstance(x, torch.Tensor) else isinstance(x, bool)
                for x in given if x is not None)
    t = BOOL if boolean_bits else _compute(out.dtype, *[x.dtype for x in tensors]) if integral else \
        _compute(out.dtype, *[x.dtype for x in tensors], torch.float32)
    r, inputs, values = _Region(), [], []
    for x in given:
        if isinstance(x, torch.Tensor):
            values.append(_load(r, len(inputs), x, t))
            inputs.append((x, _aligned(x, out.shape)))
        else:
            values.append(None if x is None else r.const(x, t))
    made = _apply(r, op, *values, alpha, t, b if op == 'pow' and not isinstance(b, torch.Tensor) else None)
    r.done(_store(r, made, BOOL if op in BOOLEAN or op == 'bitwise_not' and t == BOOL else t, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


# torch 2.14.0 torch/_tensor_docs.py, copy_ and view: views of one element count correspond by their row-major
# element index, the source's coordinates computed in the region
def _coordinates(r, linear, shape):
    made, rest, kind = [], linear, r.kinds[linear]
    for d in reversed(range(len(shape))):
        made.append(r('rem', kind, rest, r.const(shape[d], kind)))
        rest = r('div', kind, rest, r.const(shape[d], kind))
    return made[::-1]


def _linear(r, shape):
    at = r.const(0, I32)
    for d, n in enumerate(shape):
        at = r('add', I32, r('mul', I32, at, r.const(n, I32)), r.index(d))
    return at


def _copy(out, value):
    if out.is_complex() or value.is_complex():
        if out.is_complex() and value.is_complex():
            return _copy(torch.view_as_real(out), torch.view_as_real(value))
        return _complex('copy', out, value) if out.is_complex() else _copy(out, torch.view_as_real(value)[..., 0])
    _typed('a copy', [value], [out], REAL)
    r = _Region()
    if tuple(value.shape) == tuple(out.shape):
        r.done(_convert(r, r.arg(0, value), _class(value), _class(out)))
        return [_Generic(out.shape, [(value, _same(out.dim()))], [(out, _same(out.dim()))], r)]
    at = _coordinates(r, _linear(r, out.shape), value.shape)
    r.done(_convert(r, r('extract', _class(value), *at, integer=0), _class(value), _class(out)))
    return [_Generic(out.shape, [(value, None)], [(out, _same(out.dim()))], r)]


def _pair(r, inputs, x, rows, loops, t):
    pairs, made = torch.view_as_real(x), []
    for c in (0, 1):
        made.append(_load(r, len(inputs), pairs, t))
        inputs.append((pairs, rows + (_rows(loops, [{None: c}])[0],)))
    return tuple(made)


def _component(x, k):
    if isinstance(x, torch.Tensor):
        return torch.view_as_real(x)[..., k] if x.is_complex() else x if k == 0 else 0
    return (complex(x).real if k == 0 else complex(x).imag) if isinstance(x, complex) else x if k == 0 else 0


def _complex(op, out, a, b=None, alpha=1):
    shape = list(out.shape)
    loops = shape
    t = F64 if any(isinstance(x, torch.Tensor) and x.dtype in (torch.complex128, torch.float64) for x in (a, b, out)) \
        else F32
    r, inputs, values = _Region(), [], []
    for x in (a, b):
        if x is None:
            values.append(None)
        elif isinstance(x, torch.Tensor) and x.is_complex():
            values.append(_pair(r, inputs, x, _aligned(x, shape), len(loops), t))
        elif isinstance(x, torch.Tensor):
            values.append((_load(r, len(inputs), x, t), r.const(0, t)))
            inputs.append((x, _aligned(x, shape)))
        else:
            z = complex(x)
            values.append((r.const(z.real, t), r.const(z.imag, t)))
    made = _complex_value(r, t, op, values[0], values[1], alpha, b if op == 'pow' and not isinstance(b, torch.Tensor)
                          else None)
    if op in BOOLEAN:
        r.done(_store(r, made, BOOL, out))
        return [_Generic(loops, inputs, [(out, _same(len(loops)))], r)]
    re, im = made if isinstance(made, tuple) else (made, made)
    if out.is_complex():
        r.done(_store(r, re, t, out.real), _store(r, im, t, out.imag))
        return [_Generic(loops, inputs, [(out.real, _same(len(loops))), (out.imag, _same(len(loops)))], r)]
    r.done(_store(r, re, t, out))
    return [_Generic(loops, inputs, [(out, _same(len(loops)))], r)]


# the complex arithmetic of (re, im) pairs: R. L. Smith, Algorithm 116, Complex division, CACM 5(8), 1962; the modulus
# scaled by the larger part (no overflow of the squares); W. Kahan, Branch cuts for complex elementary functions, in
# The State of the Art in Numerical Analysis, 1987 (sqrt by the half-sum of |z| and |re|; tanh by tan and sinh);
# the elementary functions by their identities (Abramowitz and Stegun 4.3.55-4.3.58, 4.5.49-4.5.54, 4.4.14-4.4.16,
# 4.6.14-4.6.16: sin, cos, sinh, cosh and the inverses through log and sqrt)
def _complex_value(r, t, op, z, w, alpha, exponent):
    def c(v):
        return r.const(v, t)

    def add(x, y):
        return r('add', t, x[0], y[0]), r('add', t, x[1], y[1])

    def sub(x, y):
        return r('sub', t, x[0], y[0]), r('sub', t, x[1], y[1])

    def mul(x, y):
        return (r('sub', t, r('mul', t, x[0], y[0]), r('mul', t, x[1], y[1])),
                r('add', t, r('mul', t, x[0], y[1]), r('mul', t, x[1], y[0])))

    def scale(x, k):
        return r('mul', t, x[0], c(k)), r('mul', t, x[1], c(k))

    def div(x, y):
        wide = r('ge', BOOL, r('abs', t, y[0]), r('abs', t, y[1]))
        first, second = r('select', t, wide, y[0], y[1]), r('select', t, wide, y[1], y[0])
        ratio = r('div', t, second, first)
        den = r('add', t, first, r('mul', t, second, ratio))
        re_w = r('div', t, r('add', t, x[0], r('mul', t, x[1], ratio)), den)
        im_w = r('div', t, r('sub', t, x[1], r('mul', t, x[0], ratio)), den)
        re_n = r('div', t, r('add', t, r('mul', t, x[0], ratio), x[1]), den)
        im_n = r('div', t, r('sub', t, r('mul', t, x[1], ratio), x[0]), den)
        return r('select', t, wide, re_w, re_n), r('select', t, wide, im_w, im_n)

    def modulus(x):
        a, b = r('abs', t, x[0]), r('abs', t, x[1])
        big, small = r('select', t, r('ge', BOOL, a, b), a, b), r('select', t, r('ge', BOOL, a, b), b, a)
        ratio = r('div', t, small, r('select', t, r('eq', BOOL, big, c(0)), c(1), big))
        return r('mul', t, big, r('sqrt', t, r('add', t, c(1), r('mul', t, ratio, ratio))))

    def exp(x):
        e = r('exp', t, x[0])
        return r('mul', t, e, r('cos', t, x[1])), r('mul', t, e, r('sin', t, x[1]))

    def log(x):
        return r('log', t, modulus(x)), r('atan2', t, x[1], x[0])

    def sqrt(x):
        m = r('sqrt', t, r('mul', t, c(0.5), r('add', t, r('abs', t, x[0]), modulus(x))))
        zero = r('eq', BOOL, m, c(0))
        safe = r('select', t, zero, c(1), m)
        other = r('div', t, r('abs', t, x[1]), r('mul', t, c(2), safe))
        negative = r('lt', BOOL, x[0], c(0))
        re = r('select', t, negative, other, m)
        im = r('select', t, negative, r('select', t, r('lt', BOOL, x[1], c(0)), r('neg', t, m), m),
               r('div', t, x[1], r('mul', t, c(2), safe)))
        return r('select', t, zero, c(0), re), r('select', t, zero, x[1], im)

    def sinh_cosh(a):
        e, f = r('exp', t, a), r('exp', t, r('neg', t, a))
        return r('mul', t, c(0.5), r('sub', t, e, f)), r('mul', t, c(0.5), r('add', t, e, f))

    def sin(x):
        sh, ch = sinh_cosh(x[1])
        return r('mul', t, r('sin', t, x[0]), ch), r('mul', t, r('cos', t, x[0]), sh)

    def cos(x):
        sh, ch = sinh_cosh(x[1])
        return r('mul', t, r('cos', t, x[0]), ch), r('neg', t, r('mul', t, r('sin', t, x[0]), sh))

    def sinh(x):
        sh, ch = sinh_cosh(x[0])
        return r('mul', t, sh, r('cos', t, x[1])), r('mul', t, ch, r('sin', t, x[1]))

    def cosh(x):
        sh, ch = sinh_cosh(x[0])
        return r('mul', t, ch, r('cos', t, x[1])), r('mul', t, sh, r('sin', t, x[1]))

    def tanh(x):
        tn = r('tan', t, x[1])
        beta = r('add', t, c(1), r('mul', t, tn, tn))
        sh = sinh_cosh(x[0])[0]
        rho = r('sqrt', t, r('add', t, c(1), r('mul', t, sh, sh)))
        den = r('add', t, c(1), r('mul', t, beta, r('mul', t, sh, sh)))
        return r('div', t, r('mul', t, beta, r('mul', t, rho, sh)), den), r('div', t, tn, den)

    def times_i(x):
        return r('neg', t, x[1]), x[0]

    def over_i(x):
        return x[1], r('neg', t, x[0])
    one = (c(1), c(0))

    def asinh(x):
        return log(add(x, sqrt(add(mul(x, x), one))))

    def acosh(x):
        return log(add(x, mul(sqrt(add(x, one)), sqrt(sub(x, one)))))

    def atanh(x):
        return scale(sub(log(add(one, x)), log(sub(one, x))), 0.5)

    def real(name, x):
        return _apply(r, name, x, None, None, 1, t)
    if op in ('add', 'sub'):
        u = w if alpha == 1 else mul(w, (c(complex(alpha).real), c(complex(alpha).imag)))
        return add(z, u) if op == 'add' else sub(z, u)
    if op == 'mul':
        return mul(z, w)
    if op == 'div':
        return div(z, w)
    if op == 'neg':
        return r('neg', t, z[0]), r('neg', t, z[1])
    if op == 'conj':
        return z[0], r('neg', t, z[1])
    if op == 'exp':
        return exp(z)
    if op == 'exp2':
        return exp(scale(z, math.log(2)))
    if op == 'log':
        return log(z)
    if op in ('log2', 'log10'):
        return scale(log(z), 1 / math.log(2 if op == 'log2' else 10))
    if op == 'log1p':
        square = r('add', t, r('mul', t, z[0], r('add', t, c(2), z[0])), r('mul', t, z[1], z[1]))
        return r('mul', t, c(0.5), real('log1p', square)), r('atan2', t, z[1], r('add', t, c(1), z[0]))
    if op == 'expm1':
        half = r('sin', t, r('mul', t, c(0.5), z[1]))
        return (r('sub', t, r('mul', t, real('expm1', z[0]), r('cos', t, z[1])), r('mul', t, c(2), r('mul', t, half, half))),
                r('mul', t, r('exp', t, z[0]), r('sin', t, z[1])))
    if op == 'sqrt':
        return sqrt(z)
    if op == 'rsqrt':
        return div(one, sqrt(z))
    if op == 'reciprocal':
        return div(one, z)
    if op in ('sigmoid', 'silu'):
        return div(one if op == 'sigmoid' else z, add(one, exp((r('neg', t, z[0]), r('neg', t, z[1])))))
    if op.startswith('logical_'):
        def truth(x):
            return r('or', BOOL, r('ne', BOOL, x[0], c(0)), r('ne', BOOL, x[1], c(0)))
        if op == 'logical_not':
            return r('not', BOOL, truth(z))
        return r('ne' if op == 'logical_xor' else op[8:], BOOL, truth(z), truth(w))
    if op == 'pow':
        if exponent is not None and complex(exponent) == 2:
            return mul(z, z)
        power = exp(mul(w, log(z)))
        nothing = r('and', BOOL, r('eq', BOOL, z[0], c(0)), r('eq', BOOL, z[1], c(0)))
        unit = r('and', BOOL, r('eq', BOOL, w[0], c(0)), r('eq', BOOL, w[1], c(0)))
        zero = r('select', t, unit, c(1), c(0))
        return r('select', t, nothing, zero, power[0]), r('select', t, nothing, c(0), power[1])
    if op == 'sin':
        return sin(z)
    if op == 'cos':
        return cos(z)
    if op == 'tan':
        return over_i(tanh(times_i(z)))
    if op == 'sinh':
        return sinh(z)
    if op == 'cosh':
        return cosh(z)
    if op == 'tanh':
        return tanh(z)
    if op == 'asinh':
        return asinh(z)
    if op == 'acosh':
        return acosh(z)
    if op == 'atanh':
        return atanh(z)
    if op == 'asin':
        return over_i(asinh(times_i(z)))
    if op == 'acos':
        a = over_i(asinh(times_i(z)))
        return r('sub', t, c(math.pi / 2), a[0]), r('neg', t, a[1])
    if op == 'atan':
        return over_i(atanh(times_i(z)))
    if op == 'sgn':
        m = modulus(z)
        zero = r('eq', BOOL, m, c(0))
        safe = r('select', t, zero, c(1), m)
        return (r('select', t, zero, c(0), r('div', t, z[0], safe)), r('select', t, zero, c(0), r('div', t, z[1], safe)))
    if op == 'abs':
        return modulus(z)
    if op == 'abs1':
        return r('add', t, r('abs', t, z[0]), r('abs', t, z[1]))
    if op == 'angle':
        return r('atan2', t, z[1], z[0])
    if op in ('eq', 'ne'):
        same = r('and', BOOL, r('eq', BOOL, z[0], w[0]), r('eq', BOOL, z[1], w[1]))
        return same if op == 'eq' else r('not', BOOL, same)
    if op == 'isnan':
        return r('or', BOOL, r('isnan', BOOL, z[0]), r('isnan', BOOL, z[1]))
    if op == 'isinf':
        return r('or', BOOL, r('eq', BOOL, r('abs', t, z[0]), c(math.inf)), r('eq', BOOL, r('abs', t, z[1]), c(math.inf)))
    if op == 'polar':
        return r('mul', t, z[0], r('cos', t, w[0])), r('mul', t, z[0], r('sin', t, w[0]))
    if op == 'complex':
        return z[0], w[0]
    if op == 'copy':
        return z
    raise NotImplementedError('%s of complex operands: its structured complex lowering is unfinished '
                              '(design/compilation-composition-change-request.md item 1)' % op)

# torch 2.14.0 torch/_tensor_docs.py, fill_; PyTorch 2.14 documentation, torch.arange
def _fill(out, value, step=0):
    if out.is_complex():
        return [g for k in (0, 1) for g in _fill(torch.view_as_real(out)[..., k], _component(value, k),
                                                  _component(step, k))]
    _typed('a fill', [], [out], REAL)
    t = _compute(out.dtype)
    r = _Region()
    made = r.const(value, t)
    if step:
        made = r('add', t, made, r('mul', t, _convert(r, _linear(r, out.shape), I32, t), r.const(step, t)))
    r.done(_store(r, made, t, out))
    return [_Generic(out.shape, [], [(out, _same(out.dim()))], r)]


# an output kept in its arithmetic kind: the structured map out = convert(f(acc, operands...)), fused by the
# transformation with its producer
def _epilogue(out, acc, t, extra=(), apply=None):
    r = _Region()
    x = _load(r, 0, acc, t)
    loaded = [_load(r, 1 + j, v, t) for j, (v, _) in enumerate(extra)]
    r.done(_store(r, apply(r, x, *loaded) if apply else x, t, out))
    return [_Generic(out.shape, [(acc, _same(out.dim()))] + list(extra), [(out, _same(out.dim()))], r)]

# J. Gibbons, The third homomorphism theorem, J. Functional Programming 6(4), 1996, (2) and Definition 3.1: a
# reduction's states, their identity and associative join; `_identities` the finalization that is the states
def _identities(states):
    r = _Region()
    return r.done(*[r('arg', kind, integer=j) for j, (kind, _) in enumerate(states)])


def _joined(states, join):
    c = _Region()
    n = len(states)
    a = [c('arg', kind, integer=j) for j, (kind, _) in enumerate(states)]
    b = [c('arg', kind, integer=n + j) for j, (kind, _) in enumerate(states)]
    return c.done(*join(c, a, b))


def _extremum(r, a, b, t, larger):
    better = r('gt' if larger else 'lt', BOOL, b, a)
    if _floating(t):
        better = r('or', BOOL, better, r('isnan', BOOL, b))
    return r('select', t, better, b, a)


def _linalg_identity(out):
    r = _Region()
    target = torch.view_as_real(out) if out.is_complex() else out
    row = out.dim() - 2
    equal = r('eq', BOOL, r.index(row), r.index(row + 1))
    if out.is_complex():
        equal = r('and', BOOL, equal, r('eq', BOOL, r.index(out.dim()), r.const(0, I32)))
    r.done(r('cast', _class(target), equal))
    return [_Generic(target.shape, [], [(target, _same(target.dim()))], r)]


def _linalg_vector_rhs(a, b):
    return b.dim() == 1 or b.dim() == a.dim() - 1 and b.shape == a.shape[:-1]


# torch 2.14.0 aten _functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token): `dep_token` the
# typed value the check depends on, its writers and their completion before the check (None: the status alone)
class _Check:
    def __init__(self, info, code, message='', dep_token=None):
        self.info, self.code, self.place, self.node = info, code, CPU, 0
        self.message, self.dep_token = message, dep_token


# Reference-LAPACK SRC/dlarfg.f, zlarfg.f, dgeqr2.f, zgeqr2.f, dorg2r.f and zung2r.f
def _linalg_reflector(column, tau, beta, ops):
    real, batch = column.real.dtype, column.shape[:-1]
    magnitudes, scale = _temporary(column.shape, real), _temporary(batch, real)
    ops += _elementwise('abs', magnitudes, column)
    ops += _reduced(A.amax.default, {'self': magnitudes, 'dim': [-1], 'keepdim': False}, [scale])
    zero, safe = _temporary(batch, torch.bool), _temporary(batch, real)
    ops += _elementwise('eq', zero, scale, 0)
    ops += _elementwise('where', safe, zero, 1, scale)
    normalized = _temporary(column.shape, column.dtype)
    ops += _elementwise('div', normalized, column, safe.unsqueeze(-1))
    ops += _elementwise('div', magnitudes, magnitudes, safe.unsqueeze(-1))
    squares, norm = _temporary(column.shape, real), _temporary(batch, real)
    ops += _elementwise('mul', squares, magnitudes, magnitudes)
    ops += _reduced(A.sum.dim_IntList, {'self': squares, 'dim': [-1], 'keepdim': False}, [norm])
    ops += _elementwise('sqrt', norm, norm)
    alpha, signed, negative = normalized[..., 0], _temporary(batch, real), _temporary(batch, torch.bool)
    ops += _elementwise('lt', negative, alpha.real, 0)
    ops += _elementwise('neg', signed, norm)
    ops += _elementwise('where', signed, negative, norm, signed)
    inactive = _temporary(batch, torch.bool)
    if column.shape[-1] > 1:
        tail = _temporary(batch, real)
        ops += _reduced(A.amax.default, {'self': magnitudes[..., 1:], 'dim': [-1], 'keepdim': False}, [tail])
        # XNORM = 0: the tail's largest modulus (nonnegative), or a clipped empty tail's -inf amax identity
        ops += _elementwise('le', inactive, tail, 0)
    else:
        ops += _fill(inactive, True)
    if column.is_complex():
        real_alpha = _temporary(batch, torch.bool)
        ops += _elementwise('eq', real_alpha, alpha.imag, 0)
        ops += _elementwise('logical_and', inactive, inactive, real_alpha)
    denominator = _temporary(batch, column.dtype)
    ops += _elementwise('sub', denominator, alpha, signed)
    ops += _elementwise('where', denominator, inactive, 1, denominator)
    v = _temporary(column.shape, column.dtype)
    ops += _elementwise('div', v, normalized, denominator.unsqueeze(-1))
    ops += _fill(v[..., 0], 1)
    ops += _elementwise('where', safe, inactive, 1, signed)
    ops += _elementwise('sub', tau, signed, alpha)
    ops += _elementwise('div', tau, tau, safe)
    ops += _elementwise('where', tau, inactive, 0, tau)
    ops += _elementwise('mul', beta, signed, scale)
    ops += _elementwise('where', beta, inactive, column[..., 0].real, beta)
    return v.unsqueeze(-1)


# PyTorch v2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp, linalg_lstsq_out_info and get_default_lstsq_driver;
# PyTorch 2.14 torch.library.register_fake
def _lstsq_driver(a, driver):
    return driver.lower() if driver is not None else ('gels' if a.device.type == 'cuda' else 'gelsy')


@torch.library.register_fake('aten::geqrf')
def _geqrf_metadata(self):
    torch._check(self.dim() >= 2, lambda: 'torch.geqrf: input must have at least 2 dimensions')
    m, n = self.shape[-2:]
    return (self.new_empty((*self.shape[:-2], n, m)).transpose(-2, -1),
            self.new_empty((*self.shape[:-2], min(m, n))))


def _lstsq_shapes(self, b, driver):
    torch._check(self.dim() >= 2 and b.dim() >= 1,
                 lambda: 'torch.linalg.lstsq: input must have at least 2 dimensions and other at least 1')
    torch._check(0 <= self.dim() - b.dim() <= 1,
                 lambda: 'torch.linalg.lstsq: input.dim() - other.dim() must be 0 or 1')
    torch._check(self.dtype == b.dtype and self.device == b.device,
                 lambda: 'torch.linalg.lstsq: input and other must have the same dtype and device')
    selected = _lstsq_driver(self, driver)
    torch._check(selected in ('gels', 'gelsy', 'gelsd', 'gelss'),
                 lambda: 'torch.linalg.lstsq: driver must be gels, gelsy, gelsd or gelss')
    # Rejection filters are not part of even temporary integration validation.
    # PyTorch's CUDA-provider gels-only check (lstsq-pytorch.reference.cpp 3499-3503) is not Mesh's: a placed lstsq
    # is lowered by Mesh's own _lapack_least_squares (CPU), _least_squares_qr and _least_squares_svd, each driver
    vector = _linalg_vector_rhs(self, b)
    rhs = b.unsqueeze(-1) if vector else b
    m, n = self.shape[-2:]
    torch._check(m == rhs.shape[-2], lambda: 'torch.linalg.lstsq: input and other row sizes must match')
    batch = torch.broadcast_shapes(self.shape[:-2], rhs.shape[:-2])
    return selected, vector, m, n, batch, rhs.shape[-1]


# the results whose shapes the data does not decide: solution, rank and singular_values, as PyTorch gives each driver's
def _lstsq_fixed(self, shapes):
    selected, vector, m, n, batch, nrhs = shapes
    solution = self.new_empty((*batch, nrhs, max(m, n))).transpose(-2, -1)[..., :n, :]
    if vector:
        solution = solution.squeeze(-1)
    rank = self.new_empty(self.shape[:-2] if selected != 'gels' else (0,), dtype=torch.int64)
    singular = self.real.new_empty((*self.shape[:-2], min(m, n)) if selected in ('gelsd', 'gelss') else (0,))
    return solution, rank, singular


@torch.library.register_fake('aten::linalg_lstsq')
def _lstsq_metadata(self, b, rcond=None, *, driver=None):
    shapes = _lstsq_shapes(self, b, driver)
    (selected, _, m, n, batch, nrhs), (solution, rank, singular) = shapes, _lstsq_fixed(self, shapes)
    residual = self.real.new_empty((*batch, nrhs) if m > n and (selected == 'gels' or n == 0 and selected != 'gelsy')
                                   else (0,))
    # PyTorch v2.14.0 BatchLinearAlgebra.cpp linalg_lstsq_out_info, gelsd and gelss with m > n > 0: the residuals
    # computed only where every rank is n. An empty batch's every rank is n, so (*batch, nrhs); unbatched with nrhs 0
    # they are (0,) whatever the rank (nrhs or 0 long); otherwise their shape is the data's (unbatched nrhs or 0 long,
    # batched (*batch, nrhs) or (0,))
    if m > n > 0 and selected in ('gelsd', 'gelss') and batch and not math.prod(batch):
        residual = self.real.new_empty((*batch, nrhs))
    elif m > n > 0 and selected in ('gelsd', 'gelss') and (batch or nrhs):
        from torch._subclasses.fake_tensor import DynamicOutputShapeException
        raise DynamicOutputShapeException(A.linalg_lstsq.default)
    return solution, residual, rank, singular


# PyTorch 2.14 torch.linalg.lstsq, residuals: "computed if m > n and every matrix in A is full rank; otherwise, it's an
# empty tensor": for m > n > 0 under gelsd and gelss their shape is the data's (unbatched nrhs or 0 long, batched
# (*batch, nrhs) or (0,)). torch.linalg.lstsq's results a source selects among solution, rank and singular_values alone
# (compiler/patches/torch-lstsq-fixed-results.patch, its Dynamo product and strict export selection) are this one
# operator's, lowered by the same drivers (_lapack_least_squares, _least_squares_qr, _least_squares_svd)
torch.library.define('mesh::linalg_lstsq_fixed', '(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> '
                                                 '(Tensor solution, Tensor rank, Tensor singular_values)')


@torch.library.register_fake('mesh::linalg_lstsq_fixed')
def _lstsq_fixed_metadata(self, b, rcond=None, *, driver=None):
    return _lstsq_fixed(self, _lstsq_shapes(self, b, driver))


LSTSQ_FIXED = torch.ops.mesh.linalg_lstsq_fixed.default


# PyTorch 08187d9e0fba026dc8217405802ab5381dc88d90 (v2.14.0) torch/csrc/autograd/FunctionsManual.cpp 4009-4063,
# linalg_lstsq_backward(gX_, gL, A, B_, X_, grad_input_mask): its solution branch takes the default pinvA =
# at::linalg_pinv(A), A's gradient pinv_backward(gX.matmul(B.mH()), pinvA, A) and B's pinvA.mH().matmul(gX). Here no
# residuals are produced (gL undefined), so the gradient is that branch: the vector-Jacobian product of
# (A, B) -> linalg_pinv(A) @ B through linalg_pinv's own derivative (pinv_backward). tools/autograd/derivatives.yaml
# 958-962, linalg_lstsq output_differentiability [True, True, False, False]: rank and singular_values are not
# differentiable
def _lstsq_fixed_context(ctx, inputs, keyword_only_inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.mark_non_differentiable(output[1], output[2])


def _lstsq_fixed_backward(ctx, solution, rank, singular):
    a, b = ctx.saved_tensors
    vector = _linalg_vector_rhs(a, b)
    _, pullback = torch.func.vjp(lambda a, b: torch.linalg.pinv(a) @ (b.unsqueeze(-1) if vector else b), a, b)
    a_grad, b_grad = pullback(solution.unsqueeze(-1) if vector else solution)
    return a_grad, b_grad, None


torch.library.register_autograd('mesh::linalg_lstsq_fixed', _lstsq_fixed_backward, setup_context=_lstsq_fixed_context)


# design/torch-source-lowering-change-request.md item 5: the CPU drivers, Accelerate's LAPACK (clapack.h, MacOSX.sdk
# Accelerate.framework/Frameworks/vecLib.framework/Headers/clapack.h; __CLPK_integer int; LAPACK 3.2.1, as PyTorch
# v2.14.0 aten/src/ATen/native/LinearAlgebraUtils.h computeLRWorkDim states for Accelerate). Each routine family is
# one typed operator (mesh::lapack_*), its operands and results the routine's own: a matrix operand LAPACK reads and
# overwrites (A, B) is copied into the call's own column-major buffer (its leading dimension the stated one), a result
# matrix is that buffer (column-major, as the fake gives it), INFO the routine's int32 per batch member, and the
# selected symbol is the operand dtype's (s, d, c, z). The operators have no eager kernel: their implementation is the
# LAPACK call LAPACK[packet] states (library, symbols, clapack.h arguments in prototype order, workspace, error), never
# a Torch kernel. Workspace: a `query` buffer is sized by the routine's own LWORK = -1 call (its first element's real
# part; LIWORK, LRWORK alike), a `size` buffer by its stated length; every buffer is the call's own. Error: INFO as
# PyTorch v2.14.0 BatchLinearAlgebra.cpp 1610-1690 _linalg_check_errors(info, api, is_matrix) raises it (INFO < 0 its
# illegal-argument RuntimeError, INFO > 0 its torch.linalg.LinAlgError for `api`), after the call, is_matrix where the
# routine's batch has at most one member
LAPACK_LIBRARY = '/System/Library/Frameworks/Accelerate.framework/Accelerate'
Routine = collections.namedtuple('Routine', 'symbols arguments error')
# `kind`: 'char' a str operand's name (or a literal), 'int' a dimension (by pointer), 'matrix' the call's column-major
# buffer (`value` (operand copied in or None, result read out or None, leading dimension)), 'vector' a result buffer,
# 'scalar' a float operand in the real kind (by pointer), 'unused' an argument the job does not reference (a
# one-element buffer), 'work' a workspace buffer, zero on entry (dgelsy's JPVT: every column free) (`value` ('query' |
# 'size', element kind 'value' | 'real' | 'int', length)), 'length' a workspace's length (-1 in the query call), 'info'
# INFO. A dimension or length is a function of the call's sizes `g` (m and n the extents of a's last two dimensions,
# nrhs b's last, and the str operands by name)
Argument = collections.namedtuple('Argument', 'name kind value')


def _clapack(*arguments):
    return tuple(Argument(*argument) for argument in arguments)


def _lead(*sizes):
    return lambda g: max(1, *(g[size] if isinstance(size, str) else size(g) for size in sizes))


def _k(g):
    return min(g['m'], g['n'])


# dgesdd.f/zgesdd.f: JOBZ 'A' (U m x m, VT n x n), 'S' (U m x k, VT k x n), 'N' (neither); IWORK 8 min(m, n);
# zgesdd's LRWORK PyTorch computeLRWorkDim (Accelerate's LAPACK 3.2.1: 7 min(m, n) for 'N')
def _gesdd_rwork(g):
    k, top = _k(g), max(g['m'], g['n'])
    if g['jobz'] == 'N':
        return 7 * k
    return 5 * k * k + 5 * k if top > 10 * k else max(5 * k * k + 5 * k, 2 * top * k + 2 * k * k + k)


def _vt_rows(g):
    return g['n'] if g['jobz'] == 'A' else _k(g) if g['jobz'] == 'S' else 1


def _u_lead(g):
    return max(1, g['m']) if g['jobz'] != 'N' else 1


_GESDD = (('jobz', 'char', 'jobz'), ('m', 'int', 'm'), ('n', 'int', 'n'), ('a', 'matrix', ('a', None, _lead('m'))),
          ('lda', 'int', _lead('m')), ('s', 'vector', 's'), ('u', 'matrix', (None, 'u', _u_lead)),
          ('ldu', 'int', _u_lead), ('vt', 'matrix', (None, 'vt', _lead(_vt_rows))),
          ('ldvt', 'int', lambda g: max(1, _vt_rows(g))), ('work', 'work', ('query', 'value', None)),
          ('lwork', 'length', 'work'))
# dsyevd.f/zheevd.f: JOBZ 'V' (A overwritten by the eigenvectors, the result z) or 'N'; UPLO the stored triangle
_HEEVD = (('jobz', 'char', 'jobz'), ('uplo', 'char', 'uplo'), ('n', 'int', 'n'), ('a', 'matrix', ('a', 'z', _lead('n'))),
          ('lda', 'int', _lead('n')), ('w', 'vector', 'w'), ('work', 'work', ('query', 'value', None)),
          ('lwork', 'length', 'work'))
# dgeev.f/zgeev.f: JOBVL 'N' (VL unreferenced), JOBVR 'V' or 'N'; zgeev's RWORK 2 n
def _vr_lead(g):
    return max(1, g['n']) if g['jobvr'] == 'V' else 1


_GEEV = (('jobvl', 'char', 'N'), ('jobvr', 'char', 'jobvr'), ('n', 'int', 'n'),
         ('a', 'matrix', ('a', None, _lead('n'))), ('lda', 'int', _lead('n')))
_GEEV_VECTORS = (('vl', 'unused', None), ('ldvl', 'int', lambda g: 1),
                 ('vr', 'matrix', (None, 'vr', _vr_lead)), ('ldvr', 'int', _vr_lead),
                 ('work', 'work', ('query', 'value', None)), ('lwork', 'length', 'work'))
# dgels.f, dgelsy.f, dgelsd.f, dgelss.f and their z forms: B (max(m, n) x nrhs, LDB max(1, m, n)) holds the
# right-hand sides on entry (rows m on zero) and the solution (and dgels' residual rows) on exit, the result x
_LEAST = (('m', 'int', 'm'), ('n', 'int', 'n'), ('nrhs', 'int', 'nrhs'), ('a', 'matrix', ('a', None, _lead('m'))),
          ('lda', 'int', _lead('m')), ('b', 'matrix', ('b', 'x', _lead('m', 'n'))), ('ldb', 'int', _lead('m', 'n')))
_WORK = (('work', 'work', ('query', 'value', None)), ('lwork', 'length', 'work'))
_INFO = (('info', 'info', 'info'),)

LAPACK = {}


def _lapack(name, schema, fake, symbols, real, complex_, api):
    torch.library.define('mesh::lapack_' + name, schema)
    torch.library.register_fake('mesh::lapack_' + name)(fake)
    target = getattr(torch.ops.mesh, 'lapack_' + name).default
    LAPACK[target.overloadpacket] = Routine(symbols, {False: _clapack(*real), True: _clapack(*complex_)}, api)
    return target


def _column_major(like, shape, dtype=None):
    return like.new_empty((*shape[:-2], shape[-1], shape[-2]), dtype=dtype or like.dtype).mT


def _info(a, batch):
    return a.new_empty(batch, dtype=torch.int32)


def _gesdd_fake(a, jobz):
    batch, (m, n) = a.shape[:-2], a.shape[-2:]
    k = min(m, n)
    s = a.real.new_empty((*batch, k))
    if jobz == 'N':
        return s, a.new_empty((0,)), a.new_empty((0,)), _info(a, batch)
    full = jobz == 'A'
    return (s, _column_major(a, (*batch, m, m if full else k)), _column_major(a, (*batch, n if full else k, n)),
            _info(a, batch))


def _heevd_fake(a, jobz, uplo):
    batch = a.shape[:-2]
    return (a.real.new_empty(a.shape[:-1]), _column_major(a, a.shape) if jobz == 'V' else a.new_empty((0,)),
            _info(a, batch))


def _dgeev_fake(a, jobvr):
    vr = _column_major(a, a.shape) if jobvr == 'V' else a.new_empty((0,))
    return a.new_empty(a.shape[:-1]), a.new_empty(a.shape[:-1]), vr, _info(a, a.shape[:-2])


def _zgeev_fake(a, jobvr):
    return a.new_empty(a.shape[:-1]), _column_major(a, a.shape) if jobvr == 'V' else a.new_empty((0,)), \
        _info(a, a.shape[:-2])


def _least_buffer(a, b):
    return _column_major(b, (*b.shape[:-2], max(a.shape[-2:]), b.shape[-1]))


def _gels_fake(a, b):
    return _least_buffer(a, b), _info(a, a.shape[:-2])


def _gelsy_fake(a, b, rcond):
    return _least_buffer(a, b), _info(a, a.shape[:-2]), _info(a, a.shape[:-2])


def _gelsd_fake(a, b, rcond):
    return (_least_buffer(a, b), a.real.new_empty((*a.shape[:-2], min(a.shape[-2:]))), _info(a, a.shape[:-2]),
            _info(a, a.shape[:-2]))


LAPACK_GESDD = _lapack(
    'gesdd', '(Tensor a, str jobz) -> (Tensor s, Tensor u, Tensor vt, Tensor info)', _gesdd_fake,
    {torch.float32: 'sgesdd_', torch.float64: 'dgesdd_', torch.complex64: 'cgesdd_', torch.complex128: 'zgesdd_'},
    _GESDD + (('iwork', 'work', ('size', 'int', lambda g: 8 * _k(g))),) + _INFO,
    _GESDD + (('rwork', 'work', ('size', 'real', _gesdd_rwork)), ('iwork', 'work', ('size', 'int', lambda g: 8 * _k(g))))
    + _INFO, 'linalg.svd')
LAPACK_HEEVD = _lapack(
    'heevd', '(Tensor a, str jobz, str uplo) -> (Tensor w, Tensor z, Tensor info)', _heevd_fake,
    {torch.float32: 'ssyevd_', torch.float64: 'dsyevd_', torch.complex64: 'cheevd_', torch.complex128: 'zheevd_'},
    _HEEVD + (('iwork', 'work', ('query', 'int', None)), ('liwork', 'length', 'iwork')) + _INFO,
    _HEEVD + (('rwork', 'work', ('query', 'real', None)), ('lrwork', 'length', 'rwork'),
              ('iwork', 'work', ('query', 'int', None)), ('liwork', 'length', 'iwork')) + _INFO, 'linalg.eigh')
LAPACK_DGEEV = _lapack(
    'dgeev', '(Tensor a, str jobvr) -> (Tensor wr, Tensor wi, Tensor vr, Tensor info)', _dgeev_fake,
    {torch.float32: 'sgeev_', torch.float64: 'dgeev_'},
    _GEEV + (('wr', 'vector', 'wr'), ('wi', 'vector', 'wi')) + _GEEV_VECTORS + _INFO, (), 'torch.linalg.eig')
LAPACK_ZGEEV = _lapack(
    'zgeev', '(Tensor a, str jobvr) -> (Tensor w, Tensor vr, Tensor info)', _zgeev_fake,
    {torch.complex64: 'cgeev_', torch.complex128: 'zgeev_'}, (),
    _GEEV + (('w', 'vector', 'w'),) + _GEEV_VECTORS + (('rwork', 'work', ('size', 'real', lambda g: 2 * g['n'])),)
    + _INFO, 'torch.linalg.eig')
LAPACK_GELS = _lapack(
    'gels', '(Tensor a, Tensor b) -> (Tensor x, Tensor info)', _gels_fake,
    {torch.float32: 'sgels_', torch.float64: 'dgels_', torch.complex64: 'cgels_', torch.complex128: 'zgels_'},
    (('trans', 'char', 'N'),) + _LEAST + _WORK + _INFO, (('trans', 'char', 'N'),) + _LEAST + _WORK + _INFO,
    'torch.linalg.lstsq')
_JPVT = (('jpvt', 'work', ('size', 'int', lambda g: g['n'])),)
_RANK = (('rcond', 'scalar', 'rcond'), ('rank', 'vector', 'rank'))
LAPACK_GELSY = _lapack(
    'gelsy', '(Tensor a, Tensor b, float rcond) -> (Tensor x, Tensor rank, Tensor info)', _gelsy_fake,
    {torch.float32: 'sgelsy_', torch.float64: 'dgelsy_', torch.complex64: 'cgelsy_', torch.complex128: 'zgelsy_'},
    _LEAST + _JPVT + _RANK + _WORK + _INFO,
    _LEAST + _JPVT + _RANK + _WORK + (('rwork', 'work', ('size', 'real', lambda g: 2 * g['n'])),) + _INFO,
    'torch.linalg.lstsq')
_SINGULAR = (('s', 'vector', 's'),)
LAPACK_GELSD = _lapack(
    'gelsd', '(Tensor a, Tensor b, float rcond) -> (Tensor x, Tensor s, Tensor rank, Tensor info)', _gelsd_fake,
    {torch.float32: 'sgelsd_', torch.float64: 'dgelsd_', torch.complex64: 'cgelsd_', torch.complex128: 'zgelsd_'},
    _LEAST + _SINGULAR + _RANK + _WORK + (('iwork', 'work', ('query', 'int', None)),) + _INFO,
    _LEAST + _SINGULAR + _RANK + _WORK + (('rwork', 'work', ('query', 'real', None)),
                                         ('iwork', 'work', ('query', 'int', None))) + _INFO,
    'torch.linalg.lstsq')
LAPACK_GELSS = _lapack(
    'gelss', '(Tensor a, Tensor b, float rcond) -> (Tensor x, Tensor s, Tensor rank, Tensor info)', _gelsd_fake,
    {torch.float32: 'sgelss_', torch.float64: 'dgelss_', torch.complex64: 'cgelss_', torch.complex128: 'zgelss_'},
    _LEAST + _SINGULAR + _RANK + _WORK + _INFO,
    _LEAST + _SINGULAR + _RANK + _WORK + (('rwork', 'work', ('size', 'real', lambda g: 5 * _k(g))),) + _INFO,
    'torch.linalg.lstsq')


# the CPU formulations, each torch.linalg operator's own result through its routine: torch 2.14.0 aten _linalg_svd
# (linalg_svd_out: an empty input's U and Vh the identity where full), _linalg_eigh, linalg_eig/linalg_eigvals
# (linalg_eig_out_info) and linalg_lstsq (BatchLinearAlgebra.cpp linalg_lstsq_out_info 3366-3483); `shapes` the
# node's outputs' (shape, dtype). After each routine call, its INFO goes to torch's own LinAlgError boundary,
# aten._linalg_check_errors(info, api, is_matrix), with the api name and is_matrix rule of the call site in
# BatchLinearAlgebra.cpp (lstsq-pytorch.reference.cpp: linalg.svd 3188, linalg.eigh 2820, torch.linalg.eig 3048,
# torch.linalg.eigvals 3109, torch.linalg.lstsq 3681)
def _singular_empty(a, compute_uv, full_matrices, shapes):
    return tuple(torch.eye(shape[-1], dtype=dtype, device=a.device).expand(shape).clone()
                 if compute_uv and full_matrices and k != 1 and shape[-1] and shape[-1] == shape[-2]
                 else a.new_zeros(shape, dtype=dtype) for k, (shape, dtype) in enumerate(shapes))


def _lapack_singular(A, full_matrices=True, compute_uv=True, driver=None, shapes=()):
    if not A.numel():
        return _singular_empty(A, compute_uv, full_matrices, shapes)
    s, u, vt, info = LAPACK_GESDD(A, 'N' if not compute_uv else 'A' if full_matrices else 'S')
    torch.ops.aten._linalg_check_errors.default(info, 'linalg.svd', is_matrix=A.dim() == 2)
    if not compute_uv:
        return A.new_empty(shapes[0][0], dtype=shapes[0][1]), s, A.new_empty(shapes[2][0], dtype=shapes[2][1])
    return u, s, vt


def _lapack_hermitian(A, UPLO='L', compute_v=True, shapes=()):
    if not A.numel():
        return tuple(A.new_empty(shape, dtype=dtype) for shape, dtype in shapes)
    w, z, info = LAPACK_HEEVD(A, 'V' if compute_v else 'N', UPLO.upper())
    torch.ops.aten._linalg_check_errors.default(info, 'linalg.eigh', is_matrix=A.dim() == 2)
    return w, z if compute_v else A.new_empty(shapes[1][0], dtype=shapes[1][1])


# Reference-LAPACK dgeev.f: a complex pair (WI(j) > 0, WI(j+1) < 0) has v(j) = VR(:,j) + i VR(:,j+1) and
# v(j+1) = VR(:,j) - i VR(:,j+1); a real eigenvalue's vector VR(:,j) itself
def _lapack_eigen(self, vectors=True):
    a = self
    complex_dtype = a.dtype if a.is_complex() else corresponding_complex_dtype(a.dtype)
    if not a.numel():
        values = a.new_empty(a.shape[:-1], dtype=complex_dtype)
        return (values, _column_major(a, a.shape, complex_dtype)) if vectors else values
    api = 'torch.linalg.eig' if vectors else 'torch.linalg.eigvals'
    if a.is_complex():
        w, vr, info = LAPACK_ZGEEV(a, 'V' if vectors else 'N')
        A._linalg_check_errors.default(info, api, is_matrix=a.dim() == 2)
        return (w, vr) if vectors else w
    wr, wi, vr, info = LAPACK_DGEEV(a, 'V' if vectors else 'N')
    A._linalg_check_errors.default(info, api, is_matrix=a.dim() == 2)
    values = torch.complex(wr, wi)
    if not vectors:
        return values
    zero = torch.zeros_like(vr[..., :, :1])
    before, after = torch.cat([zero, vr[..., :, :-1]], -1), torch.cat([vr[..., :, 1:], zero], -1)
    first, second = (wi > 0)[..., None, :], (wi < 0)[..., None, :]
    real = torch.where(second, before, vr)
    imaginary = torch.where(first, after, torch.where(second, -vr, 0))
    return values, torch.complex(real, imaginary)


# BatchLinearAlgebra.cpp linalg_lstsq_out_info: the routine over the broadcast batch (the right-hand sides' rows past m
# zero), rank and singular_values those of the input's own batch (its first member along each broadcast dimension);
# residuals where m > n and the driver is not gelsy, the squared magnitudes of B's rows n on, summed (a shape the data
# decides, gelsd and gelss with m > n > 0 batched, is LSTSQ_FIXED's, which has none); the solution B's first n rows,
# zero where m is 0. `shapes` (solution, residuals or None for LSTSQ_FIXED, rank, singular_values)
def _lapack_least_squares(self, b, rcond=None, driver=None, shapes=()):
    a, (m, n) = self, self.shape[-2:]
    selected = _lstsq_driver(a, driver)
    vector = _linalg_vector_rhs(a, b)
    rhs = b.unsqueeze(-1) if vector else b
    batch, own = torch.broadcast_shapes(a.shape[:-2], rhs.shape[:-2]), a.shape[:-2]
    (solution, _), produced, (rank, _), (singular, real) = shapes
    residual = produced and produced[0]
    if not math.prod(batch):
        return (a.new_zeros(solution), None if residual is None else a.new_empty(residual, dtype=real),
                a.new_zeros(rank, dtype=torch.int64), a.new_zeros(singular, dtype=real))
    rcond = torch.finfo(real).eps * max(m, n) if rcond is None else rcond
    a, rhs = a.expand(*batch, m, n), rhs.expand(*batch, m, rhs.shape[-1])
    ranks = values = None
    if selected == 'gels':
        x, info = LAPACK_GELS(a, rhs)
    elif selected == 'gelsy':
        x, ranks, info = LAPACK_GELSY(a, rhs, rcond)
    else:
        x, values, ranks, info = (LAPACK_GELSD if selected == 'gelsd' else LAPACK_GELSS)(a, rhs, rcond)
    A._linalg_check_errors.default(info, 'torch.linalg.lstsq', is_matrix=info.numel() <= 1)
    first = (0,) * (len(batch) - len(own)) + tuple(slice(0, 1) if size == 1 else slice(None) for size in own)
    rank_made = a.new_zeros(rank, dtype=torch.int64) if ranks is None else ranks[first].to(torch.int64)
    singular_made = a.new_zeros(singular, dtype=real) if values is None else values[first]
    if residual is not None and math.prod(residual):
        raw = x[..., n:, :]
        residual = (raw * raw.conj()).real.sum(-2) if raw.is_complex() else (raw * raw).sum(-2)
    elif residual is not None:
        residual = a.new_empty(residual, dtype=real)
    made = x[..., :n, :]
    if m == 0:
        made = torch.zeros_like(made)
    return made.squeeze(-1) if vector else made, residual, rank_made, singular_made


# design/torch-source-lowering-change-request.md items 5 and 6; PyTorch v2.14.0 aten/src/ATen/native/LinearAlgebra.cpp,
# get_atol_rtol and linalg_pinv (each overload: atol and rtol tensors or floats, or rcond the relative tolerance with atol
# zero): an empty input's result the empty (n x m); a tolerance a tensor as given, a float or the default one of the
# selected device's tolerance dtype (`tolerance_dtype`: float32 on MPS, float64 else); the default rtol
# eps max(m, n), zero where an atol > 0 is given; tol = max(atol, rtol max |S|) per matrix (each tolerance's last
# dimension added); the reciprocal of each value above tol, zero else, in the input's dtype; Hermitian through
# linalg_eigh (|eigenvalues| the singular values, U diag(S+) U^H), else linalg_svd (Vh^H diag(S+) U^H). The one pinv
# formulation, expanded before the eigh outline (compiler/mesh_compiler/decompose.py outline)
def _pseudoinverse(self, atol=None, rtol=None, hermitian=False, rcond=None, tolerance_dtype=torch.float64):
    a, (m, n) = self, self.shape[-2:]
    if not a.numel():
        return a.new_zeros((*a.shape[:-2], n, m))
    if rcond is not None:
        atol, rtol = 0.0, rcond

    def held(value):
        return value if isinstance(value, torch.Tensor) else torch.full((), value, dtype=tolerance_dtype,
                                                                         device=a.device)
    given, atol = atol is not None, held(0.0 if atol is None else atol)
    if rtol is None:
        rtol = held(torch.finfo(a.real.dtype).eps * max(m, n))
        rtol = torch.where(atol > 0, held(0.0), rtol) if given else rtol
    rtol = held(rtol)
    if hermitian:
        values, vectors = torch.linalg.eigh(a)
        magnitude = values.abs()
        tolerance = torch.maximum(atol.unsqueeze(-1), rtol.unsqueeze(-1) * magnitude.amax(-1, keepdim=True))
        inverse = torch.where(magnitude > tolerance, values.reciprocal(), 0).to(a.dtype)
        return torch.matmul(vectors * inverse.unsqueeze(-2), vectors.mH)
    u, values, vh = torch.linalg.svd(a, full_matrices=False)
    tolerance = torch.maximum(atol.unsqueeze(-1), rtol.unsqueeze(-1) * values.narrow(-1, 0, 1))
    inverse = torch.where(values > tolerance, values.reciprocal(), 0).to(a.dtype)
    return torch.matmul(vh.mH * inverse.unsqueeze(-2), u.mH)


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/scipy/linalg.py, expm, _calc_P_Q, _solve_P_Q,
# _squaring, _pade3, _pade5, _pade7, _pade9 and _pade13
PADE = {3: (120., 60., 12., 1.), 5: (30240., 15120., 3360., 420., 30., 1.),
        7: (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.),
        9: (17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.),
        13: (64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000.,
             10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)}
SQUARING = {torch.float64: (5.371920351148152, (1.495585217958292e-002, 2.539398330063230e-001,
                                                9.504178996162932e-001, 2.097847961257068e+000), (3, 5, 7, 9, 13)),
            torch.float32: (3.925724783138660, (4.258730016922831e-001, 1.880152677804762e+000), (3, 5, 7))}


# torch 2.14.0 torch/linalg/__init__.py matrix_exp ("Supports input of bfloat16, float, double, cfloat and cdouble")
# and torch/_meta_registrations.py linalg_matrix_exp (checkFloatingOrComplex): the composition below runs in the input's
# computation dtype (torch/_prims_common get_computation_dtype, aten OpMathType: bfloat16 and float16 in float32,
# complex32 in complex64, the others as given), and the result is cast back to the input's dtype before the assertion
# takes it as its dep_token
#
# JAX jax/_src/scipy/linalg.py _calc_P_Q: idx = jnp.digitize(A_L1, conds), U, V = lax.switch(idx, [_pade3, ...], A);
# expm: R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q)), and an input of ndim > 2 jnp.vectorize'd,
# where jax/_src/lax/control_flow/conditionals.py _cond_batching_rule evaluates every branch and selects by the batched
# index. One branch formulation (_pade, the solve-and-squaring and NaN branches) serves both: a single matrix through
# torch/_higher_order_ops/switch.py switch_op (one branch runs), a batch through _branch_selected
def _matrix_exponential(a, max_squarings):
    if not a.numel():
        return torch.empty_like(a)
    given, a = a.dtype, a.to(get_computation_dtype(a.dtype))
    # Rejection filters are not part of even temporary integration validation.
    # SQUARING's lookup is the formulation's own requirement: a computation kind without JAX's table (float8, which
    # get_computation_dtype keeps) raises at it
    maxnorm, conds, degrees = SQUARING[a.real.dtype]
    norm = a.abs().sum(-2).amax(-1)
    count = (norm / maxnorm).log2().floor().clamp(min=0)
    scaled = a / count.exp2()[..., None, None]
    identity = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    batched = a.dim() > 2

    index = _digitized(norm, conds)
    if batched:
        powers = _powers(degrees, scaled, identity)
        u, v = _branch_selected(index, [_pade(degree, scaled, powers) for degree in degrees])
    else:
        u, v = switch_op(index, tuple(functools.partial(_pade_branch, degree) for degree in degrees),
                         [scaled, identity])
    over = count > max_squarings
    squares = functools.partial(_solved_squares, max_squarings=max_squarings)
    operands = [u + v, v - u, count]
    result, = _branch_selected(over, [squares(*operands), _overflowed(*operands)]) if batched else \
        switch_op(over.to(torch.int32), (squares, _overflowed), operands)
    # JAX jax/_src/scipy/linalg.py expm, `_nan` where n_squarings > max_squarings; then the operator's runtime error,
    # torch 2.14.0 torch/_export/passes/functionalize_side_effectful_ops_pass.py: _assert_async.msg as
    # _functional_assert_async.msg, its `dep_token` the result, so the error follows the result's write, and the
    # operator's result the assertion's (torch/_meta_registrations.py functional_assert_async_meta: `return dep_token`)
    return A._functional_assert_async.msg(over.logical_not(), 'linalg.matrix_exp', result.to(given))


# jnp.digitize(A_L1, conds) (right=False, searchsorted side 'right'): the count of cutoffs not above A_L1, each
# logical_not(norm < cutoff), so an equal norm takes the higher degree and a NaN norm (no cutoff compares) the highest;
# int32, as lax.switch converts its index (np.int32)
def _digitized(norm, conds):
    index = None
    for cutoff in conds:
        above = (norm < cutoff).logical_not().to(torch.int32)
        index = above if index is None else index + above
    return index


# _cond_batching_rule's batched index: every branch's results `made` (in branch order), each result component
# made[index]'s (select_n); a boolean index the two-branch cond's
def _branch_selected(index, made):
    chosen = list(made[0])
    for k, outs in enumerate(made[1:], 1):
        at = (index == k)[..., None, None]
        chosen = [torch.where(at, out, held) for out, held in zip(outs, chosen)]
    return tuple(chosen)


def _combined(terms, made=None):
    for coefficient, matrix in terms:
        made = coefficient * matrix if made is None else made + coefficient * matrix
    return made


# the powers of the scaled A the Pade `degrees` read: A^2 .. A^(d-1) (A^2, A^4 and A^6 for 13), each A^k = A^(k-2) A^2;
# one selected degree's alone inside its branch, the union of a batch's degrees once
def _powers(degrees, scaled, identity):
    powers = {0: identity, 2: torch.matmul(scaled, scaled)}
    for k in range(4, max(6 if degree == 13 else degree - 1 for degree in degrees) + 1, 2):
        powers[k] = torch.matmul(powers[k - 2], powers[2])
    return powers


# JAX _pade3, _pade5, _pade7, _pade9 and _pade13 of the scaled A over its `powers`: (U, V)
def _pade(degree, scaled, powers):
    b, parts = PADE[degree], []
    for parity in (1, 0):
        low = [(b[k + parity], powers[k]) for k in range(6 if degree == 13 else degree - 1, -1, -2)]
        if degree == 13:
            high = _combined([(b[k + 6 + parity], powers[k]) for k in (6, 4, 2)])
            parts.append(_combined(low, torch.matmul(powers[6], high)))
        else:
            parts.append(_combined(low))
    return torch.matmul(scaled, parts[0]), parts[1]


# one switch branch: that degree's own powers, then its (U, V)
def _pade_branch(degree, scaled, identity):
    return _pade(degree, scaled, _powers((degree,), scaled, identity))


# expm's _compute: _solve_P_Q (jnp.linalg.solve(Q, P)) and _squaring
def _solved_squares(p, q, count, max_squarings):
    return _squaring(torch.linalg.solve_ex(q, p).result.contiguous(), count, max_squarings),


# JAX jax/_src/scipy/linalg.py _squaring(R, n_squarings, max_squarings):
#   _scan_f(c, i): lax.cond(i < n_squarings, _squaring_precise, _identity, c)  (_squaring_precise: x @ x)
#   res, _ = lax.scan(_scan_f, R, jnp.arange(max_squarings, dtype=n_squarings.dtype))
# The scan's xs, arange(max_squarings), are compile-time, so its max_squarings steps are emitted at that static bound
# and no loop continuation is carried (the bounded predication of 1989a938, which removed the host repeat waits).
# Step i is torch/_higher_order_ops/cond.py cond_op on i < n_squarings, so only the selected branch's work runs. A
# batch (expm's jnp.vectorize) takes jax/_src/lax/control_flow/conditionals.py _cond_batching_rule's batched
# predicate, both branches and the select. JAX's `_identity` branch is `_squaring_kept`, a copy, since a HOP branch
# returns no alias of its operand
def _squaring(r, count, max_squarings):
    for i in range(max_squarings):
        if r.dim() > 2:
            r = torch.where((count > i)[..., None, None], torch.matmul(r, r), r)
        else:
            r, = cond_op(count > i, _squaring_precise, _squaring_kept, (r,))
    return r


def _squaring_precise(x):
    return torch.matmul(x, x),


def _squaring_kept(x):
    return x.clone(),


# expm's _nan, jnp.full_like(A, nan): in _compute's contiguous result layout, so both branches yield one common
# result metadata
def _overflowed(p, q, count):
    return q.new_full(q.shape, math.nan),


# JAX jax/_src/tpu/linalg/qdwh.py _qdwh: the static coefficient schedule from eps (l the lower singular value bound,
# CHOLESKY_CUTOFF 100), and the convergence tolerance cbrt(10 eps / 2)
def _qdwh_coefficients(eps, max_iterations):
    l, tol_l = eps, 10.0 * eps / 2.0
    qr_coefs, chol_coefs = [], []
    while l + tol_l < 1 and len(qr_coefs) + len(chol_coefs) < max_iterations:
        l2 = l * l
        dd = (4 * (1 / l2 - 1) / l2) ** (1 / 3)
        sqd = (1.0 + dd) ** (1 / 2)
        a = sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd)) ** (1 / 2)
        b = (a - 1) ** 2 / 4
        c = a + b - 1
        l = l * (a + b * l2) / (1 + c * l2)
        e = b / c
        if c > 100:
            qr_coefs.append(((a - e) / c ** (1 / 2), c ** (1 / 2), e))
        else:
            chol_coefs.append((a - e, c, e))
    return tol_l ** (1 / 3), qr_coefs, chol_coefs


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/tpu/linalg/qdwh.py, qdwh, _qdwh, _use_qr and
# _use_cholesky. With qdwh's dynamic_shape the operand is zero outside its leading block (_mask), and every step keeps
# it so: the QR step's columns past the block are the stacked identity's own unit columns, the Cholesky step's padded
# corner the identity. _qdwh_eigh_pop's blocks are such operands.
#
# _qdwh's cond_fun and body_fun: from k = len(qr_coefs) + len(chol_coefs), while is_not_converged and
# k < max_iterations, Halley's method, get_chol_params(3, 1, 3) = (3 - 1/3, 3, 1/3), each step tested. The bound
# max_iterations is compile-time, so the loop is emitted as its static copies k .. max_iterations - 1 (the bounded
# continuation the region records compiled, with no host repeat wait). A single matrix's copy is
# torch/_higher_order_ops/cond.py cond_op on is_converged, so a copy after convergence runs no step. A batch takes
# jax/_src/lax/control_flow/loops.py _while_loop_batching_rule: every copy steps, and a member that has converged keeps
# its iterate
def _polar(x, max_iterations=10):
    batch, real = x.shape[:-2], x.real.dtype
    m, n = x.shape[-2:]
    eps = float(torch.finfo(real).eps)
    magnitude = x.abs()
    one_norm, inf_norm = magnitude.sum(-2).amax(-1), magnitude.sum(-1).amax(-1)
    alpha_inverse = torch.where(one_norm == 0, 1, one_norm.rsqrt() * inf_norm.rsqrt())
    current = x * alpha_inverse[..., None, None]
    identity = torch.eye(n, dtype=x.dtype, device=x.device)
    tol_norm, qr_coefs, chol_coefs = _qdwh_coefficients(eps, max_iterations)
    for a_minus_e_by_sqrt_c, sqrt_c, e in qr_coefs:
        q = torch.linalg.qr(torch.cat([sqrt_c * current, identity.expand(*batch, n, n)], -2)).Q
        current = e * current + a_minus_e_by_sqrt_c * torch.matmul(q[..., :m, :], q[..., m:, :].mH)
    # without Cholesky coefficients `iterate` returns is_not_converged True; with them, the last step's test
    converged = torch.zeros(batch, dtype=torch.bool, device=x.device)
    for a_minus_e, c, e in chol_coefs:
        step = _cholesky_step(current, a_minus_e, c, e)
        converged, current = _converged(step, current, tol_norm), step
    current = current.contiguous()
    for _ in range(len(qr_coefs) + len(chol_coefs), max_iterations):
        if batch:
            step = _cholesky_step(current, 3 - 1 / 3, 3, 1 / 3)
            passed = _converged(step, current, tol_norm)
            current = torch.where(converged[..., None, None], current, step)
            converged = converged | passed
        else:
            current, converged = cond_op(converged, _halley_kept, functools.partial(_halley_continued, tol_norm),
                                         (current, converged))
    u = 1.5 * current - 0.5 * torch.matmul(current, torch.matmul(current.mH, current))
    product = torch.matmul(u.mH, x)
    return u, (product + product.mH) / 2, converged


# _use_cholesky's step: e u + (a - e) (L^-H L^-1 u^H)^H, L the lower Cholesky factor of c u^H u + I
def _cholesky_step(u, a_minus_e, c, e):
    uh = u.mH
    identity = torch.eye(u.shape[-1], dtype=u.dtype, device=u.device)
    y = torch.linalg.cholesky_ex(c * torch.matmul(uh, u) + identity).L
    z = torch.linalg.solve_triangular(y, uh, upper=False)
    w = torch.linalg.solve_triangular(y.mH, z, upper=True)
    return e * u + a_minus_e * w.mH


# _qdwh's test of a step: converged unless norm(u - u_prev) > tol_norm
def _converged(step, u, tol_norm):
    distance = (step - u).abs()
    return ((distance * distance).sum((-2, -1)).sqrt() > tol_norm).logical_not()


def _halley_continued(tol_norm, current, converged):
    step = _cholesky_step(current, 3 - 1 / 3, 3, 1 / 3).contiguous()
    return step, _converged(step, current, tol_norm)


def _halley_kept(current, converged):
    return current.clone(), converged.clone()


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/tpu/linalg/eigh.py _eigh_work, split_spectrum
# and _projector_subspace, and jax/_src/tpu/linalg/qdwh.py _qdwh, _use_qr and _use_cholesky; Y. Nakatsukasa and N. J.
# Higham, SIAM J. Sci. Comput. 35(3), 2013, sections 5.1 to 5.4: QDWH-eigh with termination_size 1, its recursion
# JAX's explicit agenda. Per batch member the state is `blocks` (the ragged stack of Hermitian blocks, each
# left-aligned at its row offset), `vectors` (the isometries, initially the identity), `agenda` ((n + 1) x 2 int32:
# row 0 the empty entry (0, 0), rows 1..top the pushed (offset, size)), `top` and the input's Frobenius norm. The
# state before the first step (the UPLO triangle reflected; `site` the eigh's source draw site, carried to every step
# as a one-element int32 tensor) and the result after the last (the first column's
# eigenvalues ascending, their vectors' columns in that order): the prefix and suffix formulations
# compiler/mesh_compiler/decompose.py outline traces
def _qdwh_eigh_state(a, upper=False, site=0):
    n, batch = a.shape[-1], a.shape[:-2]
    keep = torch.ones(n, n, dtype=torch.bool, device=a.device)
    mask = keep.triu() if upper else keep.tril()
    if a.is_complex():
        # JAX jax/_src/tpu/linalg/eigh.py _eigh_qdwh_impl: the real part reflected through UPLO's triangle, the
        # imaginary part kept strictly inside it and reflected negated (the diagonal's imaginary part zero)
        inside = keep.triu(1) if upper else keep.tril(-1)
        imaginary = torch.where(inside, a.imag, 0)
        h = torch.complex(torch.where(mask, a.real, a.real.mT), torch.where(mask, imaginary, -imaginary.mT))
    else:
        h = torch.where(mask, a, a.mT)
    vectors = torch.eye(n, dtype=a.dtype, device=a.device).expand(*batch, n, n).clone()
    entry = (torch.arange(n + 1, dtype=torch.int32, device=a.device)[:, None] == 1) \
        & (torch.arange(2, dtype=torch.int32, device=a.device) == 1)
    size = torch.scalar_tensor(n, dtype=torch.int32, device=a.device)
    zero = torch.scalar_tensor(0, dtype=torch.int32, device=a.device)
    agenda = torch.where(entry, size, zero).expand(*batch, n + 1, 2).clone()
    top = torch.ones(*batch, 1, dtype=torch.int32, device=a.device)
    magnitude = h.abs()
    return (h, vectors, agenda, top, (magnitude * magnitude).sum((-2, -1)).sqrt()[..., None],
            torch.full((1,), site, dtype=torch.int32, device=a.device))


# eigh's sort_eigenvalues (jnp.argsort, stable, NaN last; `empty` the vectors' shape without compute_v), after the
# outlined 2n - 1 steps (each split's children of positive size, pub13 section 5.2: n leaves, n - 1 splits)
def _qdwh_eigh_result(blocks, vectors, empty=None):
    values = blocks[..., :, 0]
    values, order = torch.sort(values.real if values.is_complex() else values, stable=True, dim=-1)
    if empty is not None:
        return values, vectors.new_empty(empty)
    return values, vectors.gather(-1, order[..., None, :].expand(vectors.shape))


# the body's Generator words (operations.Generator `state`: seed low, seed high, execution, 0) as read by a draw at the
# source draw site `site` (one-element int32, compiler/mesh_compiler/export.py capture's `site`). This is the explicit
# operand through which a method body's own Philox draws read the program's generator. It has only its fake (four int32
# words on site's device) and no eager kernel. The producer binds it to the body's operations.Generator:
# Generator.site(site) records the draw, so the body's Generator.stepped() advances the execution word once after its
# draws, and the value is Generator.state's root (design/compilation-composition-change-request.md items 6 and 7),
# never a fresh allocation or Inductor's RandomSeeds
torch.library.define('mesh::generator', '(Tensor site) -> Tensor')


@torch.library.register_fake('mesh::generator')
def _generator_words(site):
    return site.new_empty((4,), dtype=torch.int32)


GENERATOR = torch.ops.mesh.generator.default


# One step of JAX _eigh_work's loop_body, the outlined step's body (compiler/mesh_compiler/decompose.py outline traces
# it, once per step shape): pop (offset, b), row 0's (0, 0) once the agenda is empty. Then either the popped block's
# eigenvalues go into its first column, where it is terminal (nearly diagonal: off-diagonal norm <= 5 eps norm; tiny:
# norm < eps H0_norm; or b <= 1, termination_size 1), or the block is split (_qdwh_split). The block is JAX's
# dynamic-shape operand, _slice(blocks, (offset, 0), (b, b), (N, N)): zero outside its leading b x b (_mask), with each
# extent the member's own runtime value. lax.cond(nearly_diagonal | tiny, nearly_diagonal_case, default_case): a single
# matrix's cond_op runs one branch; a batch takes _cond_batching_rule, both branches and the select per member
def _qdwh_eigh_pop(blocks, vectors, agenda, top, norm, site, max_iterations=10):
    n, eps = blocks.shape[-1], float(torch.finfo(blocks.real.dtype).eps)
    words = GENERATOR(site)
    height = top[..., 0]
    entry = agenda.gather(-2, height.long()[..., None, None].expand(*height.shape, 1, 2))[..., 0, :]
    offset, size = entry[..., 0], entry[..., 1]
    below = torch.where(height < 1, 0, height - 1)
    h = _block_taken(blocks, offset, size)
    magnitude = _squared(h)
    squares = magnitude.sum((-2, -1))
    off_diagonal = torch.where(torch.eye(n, dtype=torch.bool, device=h.device), 0, magnitude).sum((-2, -1))
    frobenius = squares.sqrt()
    stop = (off_diagonal.sqrt() <= 5 * eps * frobenius) | (frobenius < eps * norm[..., 0]) | (size <= 1)
    e, d = torch.where(stop, 0, size), torch.where(stop, size, 0)
    operands = (blocks, vectors, agenda, h, squares, offset, e, d, below, words, site)
    split = functools.partial(_qdwh_split, max_iterations)
    if blocks.dim() == 2:
        return tuple(cond_op(e > 0, split, _qdwh_terminal, operands))
    return _selected(e > 0, split(*operands), _qdwh_terminal(*operands))


# each result where `predicate` holds (per member, broadcast over the result's own dimensions), else `held`'s
def _selected(predicate, chosen, held):
    return tuple(torch.where(predicate[(...,) + (None,) * (a.dim() - predicate.dim())], a, b)
                 for a, b in zip(chosen, held))


def _squared(x):
    return x.real * x.real + x.imag * x.imag if x.is_complex() else x * x


# JAX eigh.py _mask: the leading `rows` (x `columns`) of each member's n (x n), the extents runtime int tensors
def _within(extent, n):
    return torch.arange(n, device=extent.device) < extent[..., None]


def _square_within(rows, n, columns=None):
    return _within(rows, n)[..., :, None] & _within(rows if columns is None else columns, n)[..., None, :]


def _unit(like):
    return torch.eye(like.shape[-1], dtype=like.dtype, device=like.device)


# JAX eigh.py _slice(blocks, (offset, 0), (b, b), (N, N)): x[row + i, j] for i, j < extent, zero else
def _block_taken(x, row, extent):
    n = x.shape[-1]
    at = (row[..., None] + torch.arange(n, device=x.device)).clamp(max=n - 1)
    return torch.where(_square_within(extent, n), x.gather(-2, at[..., :, None].expand(x.shape)), 0)


# JAX eigh.py _slice(eigenvectors, (0, offset), (n, b), (N, B)): x[:, column + j] for j < extent, zero else
def _columns_taken(x, column, extent):
    n = x.shape[-1]
    at = (column[..., None] + torch.arange(n, device=x.device)).clamp(max=n - 1)
    return torch.where(_within(extent, n)[..., None, :], x.gather(-1, at[..., None, :].expand(x.shape)), 0)


# JAX eigh.py _update_slice: target[place + i, j] = source[at + i, at + j] for i, j < width, the rest kept
def _block_put(target, source, place, at, width):
    n = target.shape[-1]
    index = torch.arange(n, device=target.device)
    i = index - place[..., None]
    rows = (at[..., None] + i).clamp(0, n - 1)
    columns = (at[..., None] + index).clamp(max=n - 1)
    moved = source.gather(-2, rows[..., :, None].expand(source.shape)).gather(-1, columns[..., None, :].expand(
        source.shape))
    held = ((i >= 0) & (i < width[..., None]))[..., :, None] & _within(width, n)[..., None, :]
    return torch.where(held, moved, target)


# JAX eigh.py _update_slice: target[:, place + j] = source[:, at + j] for j < width, the rest kept
def _columns_put(target, source, place, at, width):
    n = target.shape[-1]
    j = torch.arange(n, device=target.device) - place[..., None]
    moved = source.gather(-1, (at[..., None] + j).clamp(0, n - 1)[..., None, :].expand(source.shape))
    return torch.where(((j >= 0) & (j < width[..., None]))[..., None, :], moved, target)


# nearly_diagonal_case: _update_slice(blocks, diag(H)[:, None], (offset, 0), (b, 1)); the agenda's top the entry below
# the popped one
def _qdwh_terminal(blocks, vectors, agenda, h, squares, offset, e, d, below, words, site):
    n = blocks.shape[-1]
    at = torch.arange(n, device=blocks.device) - offset[..., None]
    values = h.diagonal(dim1=-2, dim2=-1).gather(-1, at.clamp(0, n - 1))
    first = torch.where((at >= 0) & (at < d[..., None]), values, blocks[..., :, 0])
    return blocks.select_scatter(first, -1, 0), vectors.clone(), agenda.clone(), below[..., None].clone()


# default_case: split_spectrum at the diagonal's nanmedian (the leading b entries sorted, NaN last, the median of the
# non-NaN count's middle pair). U is the polar factor of H - sigma I, with the side dimensions of pub13 section 5.4.
# Where neither side is positive, section 5.1 case (ii)'s remedy reruns steps 1-4 at the diagonal's mean, which lies
# strictly between the extreme eigenvalues of a block that is not a multiple of I (section 5.2's nearly diagonal test
# has ended those): a single matrix's cond_op, a batch's select. Then Algorithm 3's subspace iteration on the projector
# of the side split (_subspace_procedure) and, while it is unaccepted, both cited remedies (_qdwh_remedied, a
# while_loop with no bound, as the source's). H_minus and H_plus (Q^H H Q's diagonal blocks) and V0 V_minus and
# V0 V_plus are written at their offsets, and the children, each of positive size, pushed
def _qdwh_split(max_iterations, blocks, vectors, agenda, h, squares, offset, e, d, below, words, site):
    n = h.shape[-1]
    diagonal = torch.where(_within(e, n), h.diagonal(dim1=-2, dim2=-1).real, math.nan)
    ordered = torch.sort(diagonal, dim=-1, stable=True).values
    count = diagonal.isnan().logical_not().sum(-1)
    last = (count - 1).clamp(min=0)
    high = torch.minimum(count // 2, last)
    sigma = (ordered.gather(-1, (last // 2)[..., None]) + ordered.gather(-1, high[..., None]))[..., 0] * 0.5
    u, total, trace = _qdwh_attempt(max_iterations, h, sigma, e)
    _, _, plus_valid, minus_valid = _dimensions(e, total, trace)
    failed = (plus_valid | minus_valid).logical_not()
    attempts, mean = (h, e, u, total, trace), functools.partial(_qdwh_mean_attempt, max_iterations)
    u, total, trace = cond_op(failed, mean, _qdwh_attempt_kept, attempts) if h.dim() == 2 else \
        _selected(failed, mean(*attempts), attempts[2:])
    state = _partition(e, offset, total, trace)
    projector = _projected(u, state[-1], e)
    q, rotated, error, unaccepted = _subspace_procedure(_initial_subspace(projector, e, state[2]), projector, h, e,
                                                        state[2], state[3], squares)
    carried = (u, projector, *state, torch.zeros_like(e), q, rotated, error, unaccepted)
    made = while_loop_op(_remedy_pending, functools.partial(_qdwh_remedied, max_iterations), carried,
                         (h, e, offset, squares, words, site))
    minus, plus, _, _, at_minus, at_plus, offset_plus, _ = made[2:10]
    q, rotated = made[11:13]
    turned = torch.matmul(_columns_taken(vectors, offset, e), q)
    for width, source, place in ((minus, at_minus, offset), (plus, at_plus, offset_plus)):
        blocks = _block_put(blocks, rotated, place, source, width)
        vectors = _columns_put(vectors, turned, place, source, width)
    return (blocks, vectors, *_pushed(agenda, below, offset, minus, offset_plus, plus))


# QDWH-eig steps 1-4 at `shift` over the leading extent block: U, the polar factor of H - shift I (split_spectrum's
# qdwh.qdwh(H_shift, is_hermitian=True, dynamic_shape=(n, n)), its operand masked), with ||U||_F^2 and trace U's real
# part
def _qdwh_attempt(max_iterations, h, shift, extent):
    shifted = torch.where(_square_within(extent, h.shape[-1]), h - shift[..., None, None] * _unit(h), 0)
    u = _polar(shifted, max_iterations)[0].contiguous()
    return (u, *_unitary(u, extent))


def _qdwh_mean_attempt(max_iterations, h, e, u, total, trace):
    diagonal = h.diagonal(dim1=-2, dim2=-1).real
    mean = torch.where(_within(e, h.shape[-1]), diagonal, 0).sum(-1) / e.to(diagonal.dtype)
    return _qdwh_attempt(max_iterations, h, mean, e)


def _qdwh_attempt_kept(h, e, u, total, trace):
    return u.clone(), total.clone(), trace.clone()


def _unitary(u, extent):
    n = u.shape[-1]
    total = torch.where(_square_within(extent, n), _squared(u), 0).sum((-2, -1))
    return total, torch.where(_within(extent, n), u.diagonal(dim1=-2, dim2=-1).real, 0).sum(-1)


# pub13 section 5.4: at an exactly singular shift U is the partial isometry, and C+ = (U + I) / 2 and C- = (I - U) / 2
# have eigenvalues 1, 0.5 and 0. The subspace for C's 1 and 0.5 is the invariant subspace of the nonnegative (C+) or
# nonpositive (C-) eigenvalues of A - sigma I. With S = ||U||_F^2 and T = trace U over the b x b block, the trace and
# ||.||_F^2 of C+- give those dimensions exactly (3 trace - 2 ||C||_F^2): b + (T - S) / 2 and b - (T + S) / 2, rounded.
# A side that gives both children positive dimension (section 5.2) splits
def _dimensions(b, total, trace):
    size = b.to(total.dtype)
    nonnegative = torch.round(size + 0.5 * (trace - total)).to(torch.int32)
    nonpositive = torch.round(size - 0.5 * (trace + total)).to(torch.int32)
    return (nonnegative, nonpositive, (nonnegative > 0) & (nonnegative < b), (nonpositive > 0) & (nonpositive < b))


# the side split by subspace iteration: the valid one, the smaller where both are (JAX's rank_plus < rank_minus), its
# dimension `rank`. Returns V_minus's column count among Q's (`minus`), V_plus's (`plus`), rank, the rest, their
# columns' first indices in Q (at_minus, at_plus), `offset_plus` and `swap` (the side C+)
def _partition(b, off, total, trace):
    nonnegative, nonpositive, plus_valid, minus_valid = _dimensions(b, total, trace)
    swap = plus_valid & (minus_valid.logical_not() | (nonnegative < nonpositive))
    smaller = torch.where(swap, nonnegative, nonpositive)
    lower = torch.where(swap, b - nonnegative, nonpositive)
    zero = torch.zeros_like(smaller)
    return (lower, b - lower, smaller, b - smaller, torch.where(swap, smaller, zero), torch.where(swap, zero, smaller),
            off + lower, swap)


# C, the projector of the side split, over the leading block: (U + I) / 2 where `swap`, else (I - U) / 2
# (split_spectrum's P_plus and P_minus, I = _mask(eye, (n, n)))
def _projected(u, swap, e):
    unit = _unit(u)
    return torch.where(_square_within(e, u.shape[-1]), 0.5 * torch.where(swap[..., None, None], u + unit, unit - u), 0)


# _projector_subspace's initial guess: the `rank` columns of C of largest norm (each row's norm, as C is Hermitian;
# the negated norms argsorted, stable, with those past the block NaN so they sort last), X = C[:, sort_idxs] masked
# to (e, rank)
def _initial_subspace(projector, e, rank):
    n = projector.shape[-1]
    norms = torch.where(_square_within(e, n), _squared(projector), 0).sum(-1).sqrt().neg()
    columns = torch.sort(torch.where(_within(e, n), norms, math.nan), dim=-1, stable=True).indices
    return torch.where(_square_within(e, n, rank), projector.gather(-1, columns[..., None, :].expand(projector.shape)),
                       0)


# pub13 section 5.1, Algorithm 3, accepting ||E||_F <= 10 u ||H||_F (u = eps / 2, the unit roundoff). A pass is
# _subspace_pass. The procedure is a pass, then X := C V1 and a pass again while E is above tolerance and strictly
# below the previous pass's (otherwise it has stagnated: "further subspace iteration does not further reduce
# ||E||_F"), as a while_loop (a batch's members selected where their predicate held, _while_loop_batching_rule).
# Returns the last pass's Q, Q^H H Q and E, and `unaccepted`, E still above tolerance
def _subspace_procedure(x, projector, h, e, rank, rest, squares):
    tolerance = 10 * float(torch.finfo(squares.dtype).eps) / 2
    q, rotated, error = _subspace_pass(x, h, rank, rest)
    carried = (q, rotated, error, error.clone(), _above(error, squares, tolerance))
    q, rotated, error, _, _ = while_loop_op(_iterating, functools.partial(_subspace_repeated, tolerance), carried,
                                            (projector, h, e, rank, rest, squares))
    return q, rotated, error, _above(error, squares, tolerance)


def _above(error, squares, tolerance):
    return error.sqrt() > tolerance * squares.sqrt()


def _iterating(q, rotated, error, previous, again, projector, h, e, rank, rest, squares):
    return again.any()


def _subspace_repeated(tolerance, q, rotated, error, previous, again, projector, h, e, rank, rest, squares):
    x = torch.where(_square_within(e, h.shape[-1], rank), torch.matmul(projector, q), 0)
    made_q, made_rotated, made = _subspace_pass(x, h, rank, rest)
    passed = _above(made, squares, tolerance) & (made < previous)
    return _selected(again, (made_q, made_rotated, made, made, passed), (q, rotated, error, previous, again))


# _projector_subspace's body_f_after_matmul over the padded operand with pub13 Algorithm 3's complete QR of the
# leading e x rank block (Reference-LAPACK dgeqr2.f and dorg2r.f: _geqr2, and _householder_applied over the identity;
# the zero columns past `rank` give identity reflectors, so Q is the e x e factor's direct sum with the identity),
# then Q^H H Q and E = V2^H H V1, rotated[rank:rank + rest, :rank], with ||E||_F^2 the error
def _subspace_pass(x, h, rank, rest):
    n = x.shape[-1]
    packed, tau, _ = _geqr2(x)
    q = _householder_applied(packed, tau, _unit(x).expand(x.shape)).contiguous()
    rotated = torch.matmul(q.mH, torch.matmul(h, q)).contiguous()
    index = torch.arange(n, device=x.device)
    rows = (index >= rank[..., None]) & (index < (rank + rest)[..., None])
    error = torch.where(rows[..., :, None] & _within(rank, n)[..., None, :], _squared(rotated), 0).sum((-2, -1))
    return q, rotated, error


def _remedy_pending(*carried_and_given):
    return carried_and_given[14].any()


# while unaccepted, the split applies both cited remedies of pub13 section 5.1: (ii) the polar factor of U again
# (the QDWH owner, its max_iterations), with its sides' dimensions, partition and C rebuilt from it; and (i)
# X := C W1, W1 the first `rank` columns of a Haar unitary W (the complete QR of a standard normal b x rank matrix,
# _haar_drawn, each column signed by its R diagonal: Mezzadri 5.12, R's diagonal zgeqr2's real beta, so its phase is
# its sign), with the procedure run again. `stream` counts the restarts, the draws' fourth counter word. A batch's
# members are selected where they were unaccepted
def _qdwh_remedied(max_iterations, u, projector, minus, plus, rank, rest, at_minus, at_plus, offset_plus, swap,
                   stream, q, rotated, error, unaccepted, h, e, offset, squares, words, site):
    n = h.shape[-1]
    refined = _polar(torch.where(_square_within(e, n), u, 0), max_iterations)[0].contiguous()
    state = _partition(e, offset, *_unitary(refined, e))
    remade = _projected(refined, state[-1], e)
    restart = stream + 1
    drawn = torch.where(_square_within(e, n, state[2]), _haar_drawn(words, site, restart, h), 0)
    packed, tau, _ = _geqr2(drawn)
    rotation = _householder_applied(packed, tau, _unit(drawn).expand(drawn.shape))
    signed = torch.where((packed.diagonal(dim1=-2, dim2=-1).real < 0)[..., None, :], -rotation, rotation)
    x = torch.where(_square_within(e, n, state[2]), torch.matmul(remade, signed), 0)
    made = (refined, remade, *state, restart, *_subspace_procedure(x, remade, h, e, state[2], state[3], squares))
    held = (u, projector, minus, plus, rank, rest, at_minus, at_plus, offset_plus, swap, stream, q, rotated, error,
            unaccepted)
    return tuple(value.contiguous() for value in _selected(unaccepted, made, held))


# J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, SC 2011, §3.1 and §4.2: Philox4x32-10 (PHILOX's multipliers
# and Weyl key increments, ten rounds) over uint32 words held in int64, each product formed from its 16-bit halves as
# the 32-bit word arithmetic does; the counter (c0, c1, c2, c3) under the key (k0, k1)
def _philox_words(counter, key):
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for step in range(10):
        hi0, lo0 = _mulhilo_words(c0, PHILOX[0])
        hi1, lo1 = _mulhilo_words(c2, PHILOX[1])
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if step < 9:
            k0, k1 = (k0 + PHILOX[2]) & 0xFFFFFFFF, (k1 + PHILOX[3]) & 0xFFFFFFFF
    return c0, c1, c2, c3


def _mulhilo_words(a, m):
    low, high = m & 0xFFFF, m >> 16
    al, ah = a & 0xFFFF, a >> 16
    t0, t1, t2, t3 = al * low, ah * low, al * high, ah * high
    middle = (t0 >> 16) + (t1 & 0xFFFF) + (t2 & 0xFFFF)
    return t3 + (t1 >> 16) + (t2 >> 16) + (middle >> 16), (t0 + ((t1 + t2) << 16)) & 0xFFFFFFFF


# G. E. P. Box and M. E. Muller, Ann. Math. Statist. 29(2), 1958: a standard normal draw from two Philox words, in
# float32 (each word's high 24 bits a uniform; the first offset by 2^-24, so its logarithm is finite)
def _normal_words(first, second):
    radius = ((first >> 8).to(torch.float32) * 2.0 ** -24 + 2.0 ** -24).log().mul(-2).sqrt()
    return radius * ((second >> 8).to(torch.float32) * 2.0 ** -24).mul(2 * math.pi).cos()


# F. Mezzadri, arXiv:math-ph/0609050, section 5 (5.12, haar_measure): the restart's standard normal matrix (`like`'s
# shape and dtype). Each element is its own Philox draw: the counter is its row-major index in the member's n x n
# (a complex one's real pairs, n x n x 2, each part of variance 1/2) offset by member * n * n * parts, then the
# execution word (Generator.state word 2, the per-call counter), the eigh's source draw site and the restart count;
# the key is the seed's two words. Box-Muller in float32, then the matrix's dtype
def _haar_drawn(words, site, stream, like):
    batch, n = like.shape[:-2], like.shape[-1]
    parts = 2 if like.is_complex() else 1
    inner = (n, n, 2) if parts == 2 else (n, n)
    spread = (...,) + (None,) * len(inner)
    linear = torch.arange(n * n * parts, dtype=torch.int64, device=like.device).view(inner)
    member = torch.arange(math.prod(batch), dtype=torch.int64, device=like.device).view(batch)
    element = (member[spread] * (n * n * parts) + linear) & 0xFFFFFFFF
    drawn = _philox_words((element, _unsigned(words[2]), _unsigned(site[0]), _unsigned(stream)[spread]),
                          (_unsigned(words[0]), _unsigned(words[1])))
    normal = _normal_words(drawn[0], drawn[1])
    if parts == 1:
        return normal.to(like.dtype)
    normal = normal * math.sqrt(0.5)
    return torch.complex(normal[..., 0], normal[..., 1]).to(like.dtype)


# the children, each of positive size, pushed above `below`: (offset, minus) at below + 1, then (offset_plus, plus)
# above it, so the plus child is popped first
def _pushed(agenda, below, offset, minus, offset_plus, plus):
    first, second = minus > 0, plus > 0
    at = below + 1
    rows = torch.arange(agenda.shape[-2], device=agenda.device)
    for row, held, made in ((at, first, (offset, minus)), (at + first.to(at.dtype), second, (offset_plus, plus))):
        chosen = (rows == row[..., None]) & held[..., None]
        agenda = torch.where(chosen[..., None], torch.stack(made, -1)[..., None, :], agenda)
    return agenda, (below + first.to(below.dtype) + second.to(below.dtype))[..., None]


# metal-microbench metal_ir.h metal_ir_repeat: one source numerical repetition of the geev routines (gebal's balancing
# sweeps, lahqr's QR iterations, the vectors' back-substitution), a while loop. `records` one pass in order (structured records, a nested _Repeat among them); before every pass,
# the first included, the one-element int32 `condition` (written by the records before this one, then by each pass)
# nonzero runs it; `carried` the TEMPORARY tensors a pass reads that an earlier pass or the records before wrote, and
# those read after the repetition. No iteration bound: the pass's own condition ends it
class _Repeat:
    def __init__(self, records, condition, carried):
        self.records, self.condition, self.carried = list(records), condition, tuple(carried)
        self.place, self.node = GPU, 0


# one structured record: `outs` and `ins` (tensor, picks) where picks name each dimension's loop (compiler `_rows`;
# None for the whole picks reads that input by extraction only, its value its input slot); fn(r, values) gives each
# output's (value, kind). A complex operand is its real pairs (torch.view_as_real), each part a constant component
# row (_pair): its value a (re, im) pair, its loops the matrix's alone, an extracted one one slot read with its
# component coordinate; each such operand the record's `component_views` entry (its _emit offsets projected through
# them, _on_block). An output of picks None is a scatter destination (metal_ir.h metal_ir_scatter): one emitted
# output, a complex one its whole real pairs (the scatter's coordinates carrying the component), its yield not
# stored. A scatter's output ordinal is the emitted one: a mapped complex output emits two, any other one
def _record(sizes, outs, ins, fn, iterators=None, combiner=None):
    r, n, inputs, values, views = _Region(), len(sizes), [], [], []
    for x, picks in ins:
        held = torch.view_as_real(x) if x.is_complex() else x
        if picks is None:
            values.append(len(inputs))
            inputs.append((held, None))
        elif x.is_complex():
            values.append(tuple(r.arg(len(inputs) + c, held) for c in (0, 1)))
            inputs += [(held, _rows(n, picks) + _rows(n, [{None: c}])) for c in (0, 1)]
        else:
            values.append(r.arg(len(inputs), x))
            inputs.append((x, _rows(n, picks)))
        views += [(x, s) for s in range(len(inputs) - (2 if picks is not None else 1), len(inputs)) if x.is_complex()]
    outputs, yields, kinds, combiners = [], [], [], []
    for (o, picks), (v, kind), combined in zip(outs, fn(r, values), combiner or [ASSIGN] * len(outs)):
        held = torch.view_as_real(o) if o.is_complex() else o
        for c in (0, 1) if o.is_complex() and picks is not None else (None,):
            views += [(o, len(inputs) + len(outputs))] if o.is_complex() else []
            outputs.append((held, None if picks is None else _rows(n, picks) + (_rows(n, [{None: c}]) if c is not None
                                                                                 else ())))
            value = v if c is None else _part(r, kind, v, c)
            yields.append(_convert(r, value[0] if isinstance(value, tuple) else value, kind, _class(held)))
            kinds.append(kind)
            combiners.append(combined)
    r.done(*yields)
    return [_Generic(sizes, inputs, outputs, r, iterators, combiners if combiner else None, kinds,
                     precision=REASSOCIATE | CONTRACT if combiner else 0, component_views=views)]


# a record value: a real handle, or a complex (re, im) pair. Complex arithmetic is _complex_value's on the same
# region; component c, |z|^2 and a component select are the real formulas
def _part(r, t, x, c):
    return x[c] if isinstance(x, tuple) else x if c == 0 else r.const(0, t)


def _complex_parts(r, t, x):
    return x if isinstance(x, tuple) else (x, r.const(0, t))


def _valued(r, t, op, x, y):
    if not isinstance(x, tuple) and not isinstance(y, tuple):
        return r(op, t, x, y)
    return _complex_value(r, t, op, _complex_parts(r, t, x), _complex_parts(r, t, y), 1, None)


def _plus(r, t, x, y, op='add'):
    return _valued(r, t, op, x, y)


def _times(r, t, x, y):
    return _valued(r, t, 'mul', x, y)


def _conjugate(r, t, x):
    return _complex_value(r, t, 'conj', x, None, 1, None) if isinstance(x, tuple) else x


def _negated(r, t, x):
    return _complex_value(r, t, 'neg', x, None, 1, None) if isinstance(x, tuple) else r('neg', t, x)


def _square(r, t, x):
    return r('add', t, r('mul', t, x[0], x[0]), r('mul', t, x[1], x[1])) if isinstance(x, tuple) else r('mul', t, x, x)


def _chosen(r, t, condition, x, y):
    if not isinstance(x, tuple) and not isinstance(y, tuple):
        return r('select', t, condition, x, y)
    return tuple(r('select', t, condition, _part(r, t, x, c), _part(r, t, y, c)) for c in (0, 1))


def _scalars(outs, ins, fn, gathered=()):
    return _record((1,), [(o, (None,) * o.dim()) for o in outs],
                   [(x, (None,) * x.dim()) for x in ins] + [(x, None) for x in gathered], fn)


# out = a b over `sizes` (n, m, k), `adjoint` conjugating each flagged operand (its picks name its transpose)
def _gemm(out, a, a_picks, b, b_picks, sizes, adjoint=(False, False)):
    t = _class(out.real)
    return _record(sizes, [(out, (0, 1))], [(a, a_picks), (b, b_picks)], lambda r, v: [(_times(r, t, *[
        _conjugate(r, t, x) if flag else x for x, flag in zip(v, adjoint)]), t)], [PARALLEL, PARALLEL, REDUCTION],
        [SUM])


# records over runtime extents (_on_block): `loops` per loop position, or `sizes` by static loop size, an extent
# (one-element int32 tensor, bound) or None; `offsets` (operand, {dim: extent}) a symbol added to that dimension
def _emit(ops, records, loops=(), sizes=None, offsets=()):
    symbols, index = [], {}

    def symbol(s):
        if id(s[0]) not in index:
            index[id(s[0])] = len(symbols)
            symbols.append(s)
        return index[id(s[0])]
    clips = {l: symbol(s) for l, s in enumerate(loops) if s is not None}
    sized = {n: symbol(s) for n, s in (sizes or {}).items() if s is not None}
    moved = [(t, {d: symbol(s) for d, s in dims.items()}) for t, dims in offsets]
    ops += _on_block(records, symbols, moved,
                     lambda g, l: sized.get(g.sizes[l]) if sizes is not None else clips.get(l))


def _at(vector, j, bound):
    return vector[j:j + 1], bound


# a copy whose loops are the elements' alone (a complex one through _complex, not its real pairs' component loop)
def _bounded_copy(out, x):
    return _complex('copy', out, x) if out.is_complex() else _copy(out, x)


# JAX jax/_src/tpu/linalg/svd.py svd and _svd_tall_and_square_input; pub13
# sections 4 and 5.5; torch 2.14.0 aten _linalg_svd (A, full_matrices, compute_uv, driver) -> (U, S, Vh), `shapes`
# its outputs' (shape, dtype), an empty input's U and Vh the identity where full (linalg_svd_out)
def _singular_qdwh(A, full_matrices=False, compute_uv=True, driver=None, shapes=(), max_iterations=10):
    a, (m, n) = A, A.shape[-2:]
    if not a.numel():
        return _singular_empty(a, compute_uv, full_matrices, shapes)
    flip = m < n
    if flip:
        a, (m, n) = a.mH, (n, m)
    q = null = None
    if full_matrices and m > n:
        complete, reduced = torch.linalg.qr(a, mode='complete')
        q, null, a = complete[..., :n], complete[..., n:], reduced[..., :n, :]
    elif m > 1.15 * n:
        q, a = torch.linalg.qr(a)
    u_p, h, _ = _polar(a, max_iterations)
    if not compute_uv:
        s_out = torch.sort(torch.linalg.eigvalsh(h).clamp(min=0), stable=True, dim=-1, descending=True).values
        return A.new_empty(shapes[0][0], dtype=shapes[0][1]), s_out, A.new_empty(shapes[2][0], dtype=shapes[2][1])
    s, v = torch.linalg.eigh(h)
    s_out, order = torch.sort(s.clamp(min=0), stable=True, dim=-1, descending=True)
    v_out = v.gather(-1, order[..., None, :].expand(v.shape))
    u_out = _rank_corrected(torch.matmul(u_p, v_out), s_out, n)
    if q is not None:
        u_out = torch.matmul(q, u_out)
    if null is not None:
        u_out = torch.cat([u_out, null], -1)
    finite = torch.isfinite(a).all(-1).all(-1)
    u_out = torch.where(finite[..., None, None], u_out, math.nan)
    s_out = torch.where(finite[..., None], s_out, math.nan)
    v_out = torch.where(finite[..., None, None], v_out, math.nan)
    return (v_out, s_out, u_out.mH) if flip else (u_out, s_out, v_out.mH)


# JAX jax/_src/tpu/linalg/svd.py 104-118 (pub13 section 5.5), the correction where u from qdwh is not unitary:
#   correct_rank_deficiency(u_out): u_out, r = qr(u_out, full_matrices=False); u_out @ diag(where(diag(r) >= 0, 1, -1))
#   do_correction = s_out[-1] <= a.shape[1] * eps * s_out[0]
#   u_out, _ = lax.while_loop(lambda args: args[1], lambda args: (correct_rank_deficiency(args[0]), False),
#                             (u_out, do_correction))
# as torch/_higher_order_ops/while_loop.py while_loop_op over the carry (u_out, do_correction). A batch is
# jax/_src/lax/control_flow/loops.py _while_loop_batching_rule's batched predicate: the loop runs while any member's
# predicate holds and each member's body result is selected where its own predicate held. The carry is contiguous
# (while_loop_fake_tensor_mode's check_meta_consistency compares carried and body strides); diag(r) >= 0 of a
# complex r is its real part's (zgeqr2's beta, R's diagonal, is real)
def _rank_corrected(u_out, s_out, n):
    eps = float(torch.finfo(u_out.real.dtype).eps)
    do_correction = s_out[..., -1] <= n * eps * s_out[..., 0]
    u_out, _ = while_loop_op(_correction_pending, _rank_correction, (u_out.contiguous(), do_correction), ())
    return u_out


def _correction_pending(u_out, do_correction):
    return do_correction.any()


def _rank_correction(u_out, do_correction):
    q, r = torch.linalg.qr(u_out)
    diagonal = r.diagonal(dim1=-2, dim2=-1)
    sign = torch.where((diagonal.real if diagonal.is_complex() else diagonal) >= 0, 1, -1).to(q.dtype)
    corrected = torch.where(do_correction[..., None, None], q * sign[..., None, :], u_out).contiguous()
    return corrected, torch.zeros_like(do_correction)


# Reference-LAPACK SRC/dgels.f, zgels.f, dgelsy.f and zgelsy.f on the placed processor (a CPU lstsq is LAPACK's,
# _lapack_least_squares): the range scaling (_range_scaled) over the broadcast batch, then gels' QR solve (dgeqrf,
# dormqr, dtrtrs: a zero diagonal of the triangular factor INFO, its first index, as torch.linalg.lstsq's
# _linalg_check_errors; a zero matrix's solution and residual rows zero) or gelsy's column-pivoted QR, incremental rank
# and complete orthogonal factorization (_gelsy_rank, _gelsy_solution), then the scaling's undo. Each factorization
# step is functional ATen (slice_scatter, gather, matmul, where), unrolled over the static min(m, n) columns.
# `shapes` (solution, residuals or None for LSTSQ_FIXED, rank, singular_values) as _lapack_least_squares'
def _least_squares_qr(self, b, rcond=None, driver=None, shapes=()):
    a, (m, n) = self, self.shape[-2:]
    selected = _lstsq_driver(a, driver)
    vector = _linalg_vector_rhs(a, b)
    rhs = b.unsqueeze(-1) if vector else b
    batch, own, nrhs = torch.broadcast_shapes(a.shape[:-2], rhs.shape[:-2]), a.shape[:-2], rhs.shape[-1]
    (solution, _), produced, (rank, _), (singular, real) = shapes
    residual = produced and produced[0]
    singular_made = a.new_zeros(singular, dtype=real)
    if not min(m, n) or not nrhs:
        if residual is not None:
            residual = a.new_zeros(residual, dtype=real)
        return a.new_zeros(solution), residual, a.new_zeros(rank, dtype=torch.int64), singular_made
    rcond = torch.finfo(real).eps * max(m, n) if rcond is None else rcond
    a, rhs = a.expand(*batch, m, n), rhs.expand(*batch, m, nrhs)
    scaled, right, a_scale, b_scale = _range_scaled(a, rhs)
    raw, rank_made = None, a.new_zeros(rank, dtype=torch.int64)
    if selected == 'gels':
        zero = (_component_maximum(a) == 0)[..., None, None]
        k = min(m, n)
        packed, tau, _ = _geqr2(scaled.mH if m < n else scaled)
        triangle = packed[..., :k, :]
        diagonal = torch.where(zero[..., 0], 1, triangle.diagonal(dim1=-2, dim2=-1))
        triangle = triangle.diagonal_scatter(diagonal, dim1=-2, dim2=-1)
        singular_at = diagonal == 0
        info = torch.where(singular_at.any(-1), singular_at.to(torch.int32).argmax(-1) + 1, 0).to(torch.int32)
        A._linalg_check_errors.default(info.contiguous(), 'torch.linalg.lstsq', is_matrix=info.numel() <= 1)
        if m >= n:
            transformed = _householder_applied(packed, tau, right, adjoint=True)
            x = torch.linalg.solve_triangular(triangle, transformed[..., :n, :], upper=True)
            raw = torch.where(zero, 0, transformed[..., n:, :]) / b_scale[..., None, None]
        else:
            middle = torch.linalg.solve_triangular(triangle.mH, right, upper=False)
            x = _householder_applied(packed, tau, torch.cat([middle, middle.new_zeros((*batch, n - m, nrhs))], -2))
        x = torch.where(zero, 0, x)
    else:
        packed, tau, pivots = _geqr2(scaled, pivoted=True)
        ranks = _gelsy_rank(packed, rcond)
        x = _gelsy_solution(packed, tau, pivots, ranks, right)
        first = (0,) * (len(batch) - len(own)) + tuple(slice(0, 1) if size == 1 else slice(None) for size in own)
        rank_made = ranks[first]
    x = _range_unscaled(x, a_scale, b_scale)
    if residual is not None and math.prod(residual):
        residual = (raw * raw.conj()).real.sum(-2) if raw.is_complex() else (raw * raw).sum(-2)
    elif residual is not None:
        residual = a.new_zeros(residual, dtype=real)
    return x.squeeze(-1) if vector else x, residual, rank_made, singular_made


# Reference-LAPACK SRC/dlarfg.f and zlarfg.f over the last dimension of `column`: (v, tau, beta), v[0] = 1, with
# H = I - tau v v^H taking column to beta e_1; the column first scaled by its largest modulus, a zero tail (and a
# real alpha) the identity reflector (tau 0, beta alpha's real part)
def _reflector(column):
    magnitudes = column.abs()
    scale = magnitudes.amax(-1)
    safe = torch.where(scale == 0, 1, scale)
    normalized, magnitudes = column / safe[..., None], magnitudes / safe[..., None]
    norm = (magnitudes * magnitudes).sum(-1).sqrt()
    alpha = normalized[..., 0]
    signed = torch.where(alpha.real < 0, norm, -norm)
    inactive = magnitudes[..., 1:].amax(-1) <= 0 if column.shape[-1] > 1 else torch.ones_like(norm, dtype=torch.bool)
    if column.is_complex():
        inactive = inactive & (alpha.imag == 0)
    v = normalized / torch.where(inactive, 1, alpha - signed)[..., None]
    v = torch.cat([torch.ones_like(v[..., :1]), v[..., 1:]], -1)
    tau = torch.where(inactive, 0, (signed - alpha) / torch.where(inactive, 1, signed))
    beta = torch.where(inactive, column[..., 0].real, signed * scale)
    return v, tau, beta


# Reference-LAPACK SRC/dlarf.f: H block (left) or block H (right), H = I - tau v v^H (tau's conjugate: H^H)
def _dlarf(block, v, tau, left=True, adjoint=False):
    v, scalar = v[..., :, None], (tau.conj() if adjoint else tau)[..., None, None]
    if left:
        return block - torch.matmul(v, torch.matmul(v.mH, block) * scalar)
    return block - torch.matmul(torch.matmul(block, v) * scalar, v.mH)


# `block` written into x at (row, column), functionally (aten slice_scatter)
def _put(x, block, row, column):
    rows = x[..., row:row + block.shape[-2], :]
    return x.slice_scatter(rows.slice_scatter(block, -1, column, column + block.shape[-1]), -2, row,
                           row + block.shape[-2])


# rows (dim -2) or columns (dim -1) j and p (a batch index tensor) of x exchanged (dlaswp, dswap)
def _exchanged(x, p, j, dim):
    index = torch.arange(x.shape[dim], device=x.device)
    p = p[..., None]
    index = torch.where(index == j, p, torch.where(index == p, j, index))
    return x.gather(dim, (index[..., :, None] if dim == -2 else index[..., None, :]).expand(x.shape))


# Reference-LAPACK SRC/dgeqr2.f and zgeqr2.f (dgeqrf's unblocked form): the packed factor (R on and above the
# diagonal, each reflector's tail below it), tau and, `pivoted`, dgeqp3's column pivots (each step the remaining
# column of largest 2-norm, recomputed over the remaining rows, moved to the front; 0-based, one per step)
def _geqr2(a, pivoted=False):
    packed, (m, n) = a, a.shape[-2:]
    taus, pivots = [], []
    for j in range(min(m, n)):
        if pivoted:
            p = _column_lengths(packed[..., j:, j:]).argmax(-1) + j
            packed = _exchanged(packed, p, j, -1)
            pivots.append(p)
        v, tau, beta = _reflector(packed[..., j:, j])
        if j + 1 < n:
            packed = _put(packed, _dlarf(packed[..., j:, j + 1:], v, tau, adjoint=True), j, j + 1)
        packed = _put(packed, torch.cat([beta.to(packed.dtype)[..., None], v[..., 1:]], -1)[..., :, None], j, j)
        taus.append(tau)
    return packed, torch.stack(taus, -1), pivots


# each column's 2-norm, scaled by its largest modulus
def _column_lengths(x):
    magnitude = x.abs()
    maximum = magnitude.amax(-2)
    scaled = magnitude / torch.where(maximum == 0, 1, maximum)[..., None, :]
    return (scaled * scaled).sum(-2).sqrt() * maximum


# Reference-LAPACK SRC/dorm2r.f and zunm2r.f, side 'L': Q b (Q^H b `adjoint`), Q = H_1 .. H_k of the packed reflectors
def _householder_applied(packed, tau, b, adjoint=False):
    for j in range(tau.shape[-1]) if adjoint else reversed(range(tau.shape[-1])):
        v = torch.cat([torch.ones_like(packed[..., j:j + 1, j]), packed[..., j + 1:, j]], -1)
        b = b.slice_scatter(_dlarf(b[..., j:, :], v, tau[..., j], adjoint=adjoint), -2, j)
    return b


# dgelsy.f 10-20 with Reference-LAPACK SRC/dlaic1.f and zlaic1.f: the largest and smallest singular value estimates
# of the leading triangle (each its approximate singular vector), the rank the leading columns whose estimate ratio
# SMAXPR RCOND <= SMINPR holds throughout; |R(1,1)| = 0 rank 0
def _gelsy_rank(packed, rcond):
    k = min(packed.shape[-2:])
    start = packed[..., 0, 0].abs()
    unit = torch.zeros_like(packed[..., 0, :k])
    unit = torch.cat([torch.ones_like(unit[..., :1]), unit[..., 1:]], -1)
    estimates, vectors = [start, start], [unit, unit]
    active = start != 0
    rank = active.to(torch.int64)
    for j in range(1, k):
        for largest in (0, 1):
            estimates[largest], vectors[largest] = _condition(packed, j, estimates[largest], vectors[largest],
                                                              largest)
        active = active & (estimates[1] * rcond <= estimates[0])
        rank = rank + active.to(torch.int64)
    return rank


# dlaic1's update by the 2 x 2 Hermitian eigenproblem of [sest^2 + |alpha|^2, alpha conj(gamma); ., |gamma|^2]
# (alpha = x^H w over the column above the diagonal, gamma the diagonal), scaled by max(|alpha|, |gamma|, sest),
# through its Jacobi rotation (_rotation): the estimate (`largest`: the larger value's root; else sest |gamma| over
# it) and the vector's update (its leading j entries by the rotation's coefficient, entry j the other)
def _condition(packed, j, estimate, vector, largest):
    alpha = torch.matmul(vector[..., None, :j].conj(), packed[..., :j, j:j + 1])[..., 0, 0]
    gamma = packed[..., j, j]
    absa, absg = alpha.abs(), gamma.abs()
    scale = torch.maximum(torch.maximum(absa, absg), estimate)
    scale = torch.where(scale == 0, 1, scale)
    absa, absg, sest = absa / scale, absg / scale, estimate / scale
    app, aqq = sest * sest + absa * absa, absg * absg
    apq = alpha / scale * (gamma / scale).conj()
    first = app >= aqq
    cosine, sine, tangent, magnitude = _rotation(app, aqq, apq, apq != 0)
    delta = tangent * magnitude
    upper = torch.where(first, app - delta, aqq + delta).sqrt()
    if largest:
        estimate = scale * upper
    else:
        estimate = estimate * (absg / torch.where(upper == 0, 1, upper))
        first = first.logical_not()
    coefficient = torch.where(first, cosine, sine)
    last = torch.where(first, -sine.conj(), cosine)
    vector = torch.cat([vector[..., :j] * coefficient[..., None], last.to(vector.dtype)[..., None],
                        vector[..., j + 1:]], -1)
    return estimate, vector


# G. H. Golub and C. F. Van Loan, Matrix Computations, 3rd ed., section 8.4, Algorithms 8.4.1 and 8.4.3; Z. Drmac and
# K. Veselic, LAPACK Working Note 196, sections 2 and 3: the symmetric Schur rotation of [app, apq; conj(apq), aqq]
# (cosine, sine, tangent and |apq|), the tangent zero where `active` does not hold
def _rotation(app, aqq, apq, active):
    magnitude = apq.abs()
    delta = aqq * 0.5 - app * 0.5
    scale = torch.maximum(delta.abs(), magnitude)
    zero = scale == 0
    scale = torch.where(zero, 1, scale)
    d, g = delta / scale, magnitude / scale
    tangent = g / torch.where(zero, 1, (d * d + g * g).sqrt() + d.abs())
    tangent = torch.where(active, torch.where(delta < 0, -tangent, tangent), 0)
    cosine = (tangent * tangent + 1).rsqrt()
    sine = apq / torch.where(magnitude == 0, 1, magnitude) * tangent * cosine
    return cosine, sine, tangent, magnitude


# Reference-LAPACK SRC/dtzrzf.f, dlatrz.f, ztzrzf.f, zlatrz.f and dgelsy.f 30-70: the leading rank rows of R reduced
# by RZ reflectors (row j's entry j and its entries past the rank, conjugated, each reflector applied from the right
# to the rows above), the triangular solve of Q^H b's leading rank rows, the RZ reflectors applied back and the pivot
# exchanges undone in reverse
def _gelsy_solution(packed, tau, pivots, rank, b):
    (m, n), k = packed.shape[-2:], min(packed.shape[-2:])
    matrix, indices = packed[..., :k, :].triu(), torch.arange(n, device=packed.device)
    reflectors = []
    for j in reversed(range(k)):
        active = j < rank
        tail = indices[j:] >= rank[..., None]
        tail = torch.cat([torch.ones_like(tail[..., :1]), tail[..., 1:]], -1)
        v, scalar, beta = _reflector(torch.where(tail, matrix[..., j, j:], 0).conj())
        scalar = torch.where(active, scalar, 0)
        if j:
            matrix = _put(matrix, _dlarf(matrix[..., :j, j:], v, scalar, left=False), 0, j)
        matrix = _put(matrix, torch.where(active, beta.to(matrix.dtype), matrix[..., j, j])[..., None, None], j, j)
        reflectors.append((j, v, scalar))
    keep = indices[:k] < rank[..., None]
    triangle = torch.where(keep[..., :, None] & keep[..., None, :], matrix[..., :k], 0)
    triangle = triangle.diagonal_scatter(torch.where(keep, triangle.diagonal(dim1=-2, dim2=-1), 1), dim1=-2, dim2=-1)
    transformed = _householder_applied(packed, tau, b, adjoint=True)
    rhs = torch.where(keep[..., None], transformed[..., :k, :], 0)
    solved = torch.linalg.solve_triangular(triangle, rhs, upper=True)
    work = torch.cat([solved, solved.new_zeros((*solved.shape[:-2], n - k, solved.shape[-1]))], -2)
    for j, v, scalar in reversed(reflectors):
        work = work.slice_scatter(_dlarf(work[..., j:, :], v, scalar), -2, j)
    for j in reversed(range(k)):
        work = _exchanged(work, pivots[j], j, -2)
    return work


# each column's 2-norm scaled by its largest modulus, and the columns normalized by it (_geev_normalized's)
def _linalg_lengths(matrix, lengths, ops, normalize=True):
    real = matrix.real.dtype
    magnitude, maximum = _temporary(matrix.shape, real), _temporary(lengths.shape, real)
    ops += _elementwise('abs', magnitude, matrix)
    ops += _reduced(A.amax.default, {'self': magnitude, 'dim': [-2], 'keepdim': False}, [maximum])
    zero, safe = _temporary(lengths.shape, torch.bool), _temporary(lengths.shape, real)
    ops += _elementwise('eq', zero, maximum, 0)
    ops += _elementwise('where', safe, zero, 1, maximum)
    ops += _elementwise('div', magnitude, magnitude, safe.unsqueeze(-2))
    ops += _elementwise('mul', magnitude, magnitude, magnitude)
    ops += _reduced(A.sum.dim_IntList, {'self': magnitude, 'dim': [-2], 'keepdim': False}, [lengths])
    ops += _elementwise('sqrt', lengths, lengths)
    ops += _elementwise('mul', lengths, lengths, maximum)
    if not normalize:
        return None
    ops += _elementwise('eq', zero, lengths, 0)
    ops += _elementwise('where', safe, zero, 1, lengths)
    normalized = _temporary(matrix.shape, matrix.dtype)
    ops += _elementwise('div', normalized, matrix, safe.unsqueeze(-2))
    return normalized


# Reference-LAPACK SRC/dgelss.f 385-435 and 737-768, dgelsd.f 379-431 and 621-645: the range scaling, ANRM = DLANGE('M')
# and BNRM brought into [SMLNUM, BIGNUM] (SMLNUM = DLAMCH('S') / DLAMCH('P'), BIGNUM its reciprocal; a zero norm
# unscaled), each matrix's component magnitudes (a complex value's real and imaginary parts); an empty B unscaled
def _range_scaled(a, b):
    limits = torch.finfo(a.real.dtype)
    small = limits.tiny / limits.eps
    a_scale = _range_scale(_component_maximum(a), small)
    b_scale = _range_scale(_component_maximum(b), small) if min(b.shape[-2:]) else b.real.new_ones(b.shape[:-2])
    return a * a_scale[..., None, None], b * b_scale[..., None, None], a_scale, b_scale


def _component_maximum(x):
    return torch.view_as_real(x).abs().amax((-3, -2, -1)) if x.is_complex() else x.abs().amax((-2, -1))


def _range_scale(maximum, small):
    safe = torch.where(maximum == 0, 1, maximum)
    return torch.minimum(torch.maximum(safe, safe.new_full((), small)), safe.new_full((), 1 / small)) / safe


# the scaling's undo (DLASCL(CTO, CFROM) of the solution by ANRM / BNRM): one multiply by numerator / denominator where
# both lie on one side of 1, else by the numerator, then the division by the denominator, so no ratio overflows
def _range_unscaled(x, numerator, denominator):
    same = (numerator >= 1) == (denominator >= 1)
    multiply = torch.where(same, numerator / denominator, numerator)
    divide = torch.where(same, 1, denominator)
    return x * multiply[..., None, None] / divide[..., None, None]


# Reference-LAPACK SRC/dgelss.f 385-435, 509-521 and 737-768, dgelsd.f 379-431 and 621-645, dlalsd.f 235-276 and
# 329-340; PyTorch v2.14.0 BatchLinearAlgebra.cpp linalg_lstsq_out_info 3366-3483: the SVD drivers through the QDWH
# SVD (_singular_qdwh), `shapes` the node's outputs' (shape, dtype)
def _least_squares_svd(self, b, rcond=None, driver=None, shapes=()):
    a, (m, n) = self, self.shape[-2:]
    rhs = b.unsqueeze(-1) if _linalg_vector_rhs(a, b) else b
    (solution, _), produced, (rank, _), (singular, real) = shapes
    residual = produced and produced[0]
    if not min(m, n):
        raw = (rhs.abs() * rhs.abs()).sum(-2)
        return (a.new_zeros(solution), None if residual is None else raw.expand(residual).clone()
                if math.prod(residual) else a.new_empty(residual, dtype=real), a.new_zeros(rank, dtype=torch.int64),
                a.new_zeros(singular, dtype=real))
    limits = torch.finfo(real)
    rcond = limits.eps * max(m, n) if rcond is None else rcond
    scaled, right, a_scale, b_scale = _range_scaled(a, rhs)
    u, s, vh = torch.linalg.svd(scaled, full_matrices=False)
    if driver.lower() == 'gelss':
        keep = s > (s[..., :1] * (rcond if rcond >= 0 else limits.eps)).clamp(min=limits.tiny)
    elif min(m, n) == 1:
        keep = s != 0
    else:
        keep = s > (limits.eps / 2 if rcond <= 0 or rcond >= 1 else rcond) * s.amax(-1, keepdim=True)
    inverse = torch.where(keep, 1 / torch.where(keep, s, 1), 0)
    made = _range_unscaled(torch.matmul(vh.mH, inverse.unsqueeze(-1) * torch.matmul(u.mH, right)), a_scale, b_scale)
    return (made.reshape(solution), None if residual is None else a.new_empty(residual, dtype=real), keep.sum(-1),
            s / a_scale.unsqueeze(-1))


# a _record whose one real output is the REGION extremum of its value over its REDUCTION loops (_reduction's states,
# join and finalize): the largest (`larger`) or the smallest, NaN propagating for a floating kind
def _record_extreme(sizes, out, picks, ins, fn, iterators, larger=True):
    (g,) = _record(sizes, [(out, picks)], ins, fn, iterators)
    kind = g.accumulator[0]
    limit = (-math.inf if larger else math.inf) if _floating(kind) else \
        torch.iinfo(torch.int32).min if larger else torch.iinfo(torch.int32).max
    g.states = [(kind, limit)]
    g.combiner, g.start, g.precision = [REGION], [IDENTITY], REASSOCIATE
    g.combine = _joined(g.states, lambda c, a, b: [_extremum(c, a[0], b[0], kind, larger)])
    g.finalize = _identities(g.states)
    return [g]


# Reference-LAPACK CABS1 (|Re| + |Im|), ABS of a complex (the scaled modulus), ZDSCAL's per-component real scaling and
# a nonzero entry, of a record value
def _cabs1(r, t, x):
    return r('add', t, r('abs', t, _part(r, t, x, 0)), r('abs', t, _part(r, t, x, 1)))


def _modulus(r, t, x):
    return _complex_value(r, t, 'abs', x, None, 1, None) if isinstance(x, tuple) else r('abs', t, x)


def _real_scaled(r, t, x, a):
    return tuple(r('mul', t, p, a) for p in x) if isinstance(x, tuple) else r('mul', t, x, a)


def _nonzero(r, t, x):
    return r('or', BOOL, r('ne', BOOL, _part(r, t, x, 0), r.const(0, t)), r('ne', BOOL, _part(r, t, x, 1), r.const(0, t)))


def _offsets(view, at):
    return [(view, at)] if at else []


# a gathered (extracted) input's element at `at` (one int32 index a dimension), its input slot `slot`: a complex one's
# (re, im) by its real pairs' component coordinate
def _entry(r, t, slot, x, *at):
    if x.is_complex():
        return tuple(r('extract', t, *at, r.const(c, I32), integer=slot) for c in (0, 1))
    return r('extract', t, *at, integer=slot)


# DZNRM2 of the first extent[0] entries of `vector` (its row `at` offset, {0: (offset, bound)}): its largest modulus
# and the sum of squares of the moduli over it (_linalg_lengths: the norm sqrt(sum) * largest), one-element reals
def _clipped_norm(vector, extent, at, ops):
    n, t, real = vector.shape[0], _class(vector.real), vector.real.dtype
    largest, total = _temporary((1,), real), _temporary((1,), real)
    view = vector[...]
    _emit(ops, _record_extreme((n,), largest, (None,), [(view, (0,))], lambda r, v: [(_modulus(r, t, v[0]), t)],
                               [REDUCTION]), loops=(extent,), offsets=_offsets(view, at))

    def squared(r, v):
        safe = r('select', t, r('eq', BOOL, v[1], r.const(0, t)), r.const(1, t), v[1])
        q = r('div', t, _modulus(r, t, v[0]), safe)
        return [(r('mul', t, q, q), t)]
    view = vector[...]
    _emit(ops, _record((n,), [(total, (None,))], [(view, (0,)), (largest, (None,))], squared, [REDUCTION], [SUM]),
          loops=(extent,), offsets=_offsets(view, at))
    return largest, total


# IZAMAX over the same clipped range: the first index (from the range's start) of the largest CABS1, one-element int32
def _first_largest(vector, extent, at, ops):
    n, t, real = vector.shape[0], _class(vector.real), vector.real.dtype
    largest, first = _temporary((1,), real), _temporary((1,), torch.int32)
    view = vector[...]
    _emit(ops, _record_extreme((n,), largest, (None,), [(view, (0,))], lambda r, v: [(_cabs1(r, t, v[0]), t)],
                               [REDUCTION]), loops=(extent,), offsets=_offsets(view, at))
    view = vector[...]
    _emit(ops, _record_extreme((n,), first, (None,), [(view, (0,)), (largest, (None,))], lambda r, v: [(r(
        'select', I32, r('eq', BOOL, _cabs1(r, t, v[0]), v[1]), r.index(0), r.const(n, I32)), I32)], [REDUCTION],
        larger=False), loops=(extent,), offsets=_offsets(view, at))
    return first


# ZSWAP of two vectors in place over loop length n: `first` and `second` (view, picks, its offsets), extent[0] entries
def _swapped(first, second, extent, ops):
    (x, x_picks, x_at), (y, y_picks, y_at) = first, second
    t, n = _class(x.real), x.shape[-1]
    _emit(ops, _record((n,), [(x, x_picks), (y, y_picks)], [(x, x_picks), (y, y_picks)],
                       lambda r, v: [(v[1], t), (v[0], t)]), loops=(extent,), offsets=_offsets(x, x_at) + _offsets(y, y_at))


# Reference-LAPACK SRC/zgebal.f and dgebal.f (JOB = 'B', lines 225-418) on one square member h, complex or real (CABS1
# and the moduli of a real entry are its |x|; DNRM2 and DZNRM2 the same scaled norm), in place. Its outputs:
# `scale` (real), the SCALE factors of rows ILO..IHI; `perm` (int32), SCALE's interchange indices outside them; `ilo`
# and `ihi` (one-element int32). `held` holds the caller's temporaries that its repetitions carry. Each DO loop is its
# n static positions (an inactive one of extent 0) and each DO WHILE a _Repeat. K, L, I and J are the source's 1-based
# values. Each interchange is a ZSWAP of the actual two columns and two rows. The 363-367 NaN exit (INFO = -3) ends the
# routine, leaving ILO and IHI unassigned as the source does
def _gebal(h, scale, perm, ilo, ihi, ops, held=()):
    n, t, real = h.shape[-1], _class(h.real), h.real.dtype
    limits = torch.finfo(real)
    sfmin1 = limits.tiny / limits.eps
    sfmax1, sfmin2 = 1 / sfmin1, sfmin1 * 2.0
    sfmax2 = 1 / sfmin2
    if not n:
        ops += _fill(ilo, 1) + _fill(ihi, 0)
        return
    k, l, noconv, done, halted, sweep, origin, kz, ks, lz, count, ecol, erow, live, more = \
        (_temporary((1,), torch.int32) for _ in range(15))
    c, rr, ca, ra, f, g, s = (_temporary((1,), real) for _ in range(7))
    reach, en, ec, er = (_temporary((n,), torch.int32) for _ in range(4))
    carried = (*held, h, scale, perm, ilo, ihi, k, l, noconv, done, halted, sweep, origin, kz, ks, lz, count, ecol,
               erow, live, more, c, rr, ca, ra, f, g, s, reach, en, ec, er)

    def ints(*values):
        return lambda r, v: [(r.const(x, I32), I32) for x in values]

    def flag(r, x):
        return _convert(r, x, BOOL, I32)

    def nonzero(r, x):
        return r('ne', BOOL, x, r.const(0, I32))
    ops += _scalars([k, l, noconv, done, halted, sweep, kz], [], ints(1, n, 1, 0, 0, 1, 0))

    # 242-286: rows isolating an eigenvalue pushed down, I = L..1 (its count fixed at entry, L0 `origin`)
    passes = _scalars([noconv, origin, reach[n - 1:n]], [l, done], lambda r, v: [
        (r.const(0, I32), I32), (v[0], I32), (r('select', I32, r('and', BOOL, r('le', BOOL, r.const(n, I32), v[0]), r(
            'not', BOOL, nonzero(r, v[1]))), v[0], r.const(0, I32)), I32)])
    for i in range(n, 0, -1):
        line = h[i - 1]
        _emit(passes, _record((n,), [(count, (None,))], [(line, (0,))], lambda r, v, i=i: [(flag(r, r(
            'and', BOOL, r('ne', BOOL, r.index(0), r.const(i - 1, I32)), _nonzero(r, t, v[0]))), I32)], [REDUCTION],
            [SUM]), loops=(_at(reach, i - 1, n),))

        def row(r, v, i=i):
            found, at, kk, conv, fin, start, ext = v
            zero, one = r.const(0, I32), r.const(1, I32)
            can = r('and', BOOL, r('gt', BOOL, ext, zero), r('eq', BOOL, found, zero))
            r.scatter(0, [r('sub', I32, at, one)], r.const(i, I32), can)
            moved = r('and', BOOL, can, r('ne', BOOL, at, r.const(i, I32)))
            last = r('and', BOOL, can, r('eq', BOOL, at, one))
            after = r('select', I32, r('and', BOOL, can, r('not', BOOL, last)), r('sub', I32, at, one), at)
            finished = r('or', BOOL, nonzero(r, fin), last)
            made = [(r.const(i, I32), I32), (after, I32), (flag(r, r('or', BOOL, nonzero(r, conv), can)), I32),
                    (flag(r, finished), I32), (r('select', I32, moved, at, zero), I32),
                    (r('select', I32, moved, r('sub', I32, r.const(n + 1, I32), kk), zero), I32),
                    (r('sub', I32, at, one), I32), (r('sub', I32, kk, one), I32)]
            if i > 1:
                made.append((r('select', I32, r('and', BOOL, r('le', BOOL, r.const(i - 1, I32), start), r(
                    'not', BOOL, finished)), after, zero), I32))
            return made
        passes += _record((1,), [(perm, None)] + [(x, (None,)) for x in (l, noconv, done, ecol, erow, lz, kz)]
                          + ([(reach[i - 2:i - 1], (None,))] if i > 1 else []),
                          [(x, (None,)) for x in (count, l, k, noconv, done, origin, reach[i - 1:i])], row)
        column, other = h[:, i - 1], h[...]
        _swapped((column, (0,), None), (other, (0, None), {1: (lz, n - 1)}), (ecol, n), passes)
        line, other = h[i - 1], h[...]
        _swapped((line, (0,), {0: (kz, n - 1)}), (other, (None, 0), {0: (lz, n - 1), 1: (kz, n - 1)}), (erow, n),
                 passes)
    passes += _scalars([sweep], [noconv, done], lambda r, v: [(flag(r, r('and', BOOL, nonzero(r, v[0]), r(
        'not', BOOL, nonzero(r, v[1])))), I32)])
    ops.append(_Repeat(passes, sweep, carried))

    # 288-317: columns isolating an eigenvalue pushed left, J = K..L (K0 `origin`), unless the rows' search returned
    ops += _scalars([noconv, sweep], [done], lambda r, v: [(flag(r, r('not', BOOL, nonzero(r, v[0]))), I32)] * 2)
    passes = _scalars([noconv, origin, reach[0:1], kz], [k, l], lambda r, v: [
        (r.const(0, I32), I32), (v[0], I32), (r('select', I32, r('and', BOOL, r('le', BOOL, v[0], r.const(1, I32)), r(
            'ge', BOOL, v[1], r.const(1, I32))), r('add', I32, r('sub', I32, v[1], v[0]), r.const(1, I32)), r.const(
                0, I32)), I32), (r('sub', I32, v[0], r.const(1, I32)), I32)])
    for j in range(1, n + 1):
        column = h[:, j - 1]
        _emit(passes, _record((n,), [(count, (None,))], [(column, (0,)), (kz, (None,))], lambda r, v, j=j: [(flag(
            r, r('and', BOOL, r('ne', BOOL, r('add', I32, r.index(0), v[1]), r.const(j - 1, I32)), _nonzero(
                r, t, v[0]))), I32)], [REDUCTION], [SUM]), loops=(_at(reach, j - 1, n),),
              offsets=[(column, {0: (kz, n - 1)})])

        def left(r, v, j=j):
            found, kk, at, conv, start, ext = v
            zero, one = r.const(0, I32), r.const(1, I32)
            can = r('and', BOOL, r('gt', BOOL, ext, zero), r('eq', BOOL, found, zero))
            r.scatter(0, [r('sub', I32, kk, one)], r.const(j, I32), can)
            moved = r('and', BOOL, can, r('ne', BOOL, kk, r.const(j, I32)))
            after = r('select', I32, can, r('add', I32, kk, one), kk)
            made = [(r.const(j, I32), I32), (after, I32), (flag(r, r('or', BOOL, nonzero(r, conv), can)), I32),
                    (r('select', I32, moved, at, zero), I32),
                    (r('select', I32, moved, r('sub', I32, r.const(n + 1, I32), kk), zero), I32),
                    (r('sub', I32, kk, one), I32), (r('sub', I32, after, one), I32)]
            if j < n:
                within = r('and', BOOL, r('le', BOOL, start, r.const(j + 1, I32)), r('ge', BOOL, at, r.const(j + 1, I32)))
                made.append((r('select', I32, within, r('add', I32, r('sub', I32, at, after), one), zero), I32))
            return made
        passes += _record((1,), [(perm, None)] + [(x, (None,)) for x in (k, noconv, ecol, erow, ks, kz)]
                          + ([(reach[j:j + 1], (None,))] if j < n else []),
                          [(x, (None,)) for x in (count, k, l, noconv, origin, reach[j - 1:j])], left)
        column, other = h[:, j - 1], h[...]
        _swapped((column, (0,), None), (other, (0, None), {1: (ks, n - 1)}), (ecol, n), passes)
        line, other = h[j - 1], h[...]
        _swapped((line, (0,), {0: (ks, n - 1)}), (other, (None, 0), {0: (ks, n - 1), 1: (ks, n - 1)}), (erow, n),
                 passes)
    passes += _scalars([sweep], [noconv], lambda r, v: [(v[0], I32)])
    ops.append(_Repeat(passes, sweep, carried))

    # 321-325: SCALE(K..L) = 1; the scaled positions K <= I <= L, each one's extents for C and R (L - K + 1), CA (L)
    # and RA (N - K + 1)
    ops += _scalars([count, kz], [k, l, done], lambda r, v: [(r('select', I32, nonzero(r, v[2]), r.const(0, I32), r(
        'add', I32, r('sub', I32, v[1], v[0]), r.const(1, I32))), I32), (r('sub', I32, v[0], r.const(1, I32)), I32)])
    unit = scale[...]
    _emit(ops, _record((n,), [(unit, (0,))], [], lambda r, v: [(r.const(1, t), t)]), loops=((count, n),),
          offsets=[(unit, {0: (kz, n - 1)})])

    def extents(r, v):
        kk, at, fin = v
        p = r('add', I32, r.index(0), r.const(1, I32))
        inside = _all(r, [r('not', BOOL, nonzero(r, fin)), r('le', BOOL, kk, p), r('le', BOOL, p, at)])
        zero = r.const(0, I32)
        return [(r('select', I32, inside, x, zero), I32) for x in (
            r('add', I32, r('sub', I32, at, kk), r.const(1, I32)), at, r('sub', I32, r.const(n + 1, I32), kk))]
    ops += _record((n,), [(en, (0,)), (ec, (0,)), (er, (0,))], [(k, (None,)), (l, (None,)), (done, (None,))], extents)

    # 344-413: the norm-reducing iteration while NOCONV
    ops += _scalars([noconv, sweep], [done], lambda r, v: [(flag(r, r('not', BOOL, nonzero(r, v[0]))), I32)] * 2)
    passes = _scalars([noconv], [], ints(0))

    def tested(r, conv, ff, cc, cca, rrr, gg, rra, first):
        test = [nonzero(r, conv)]
        if first:
            test += [r('lt', BOOL, cc, gg), r('lt', BOOL, ff, r.const(sfmax2, t)), r('lt', BOOL, cc, r.const(sfmax2, t)),
                     r('lt', BOOL, cca, r.const(sfmax2, t)), r('gt', BOOL, rrr, r.const(sfmin2, t)),
                     r('gt', BOOL, gg, r.const(sfmin2, t)), r('gt', BOOL, rra, r.const(sfmin2, t))]
        else:
            test += [r('ge', BOOL, gg, rrr), r('lt', BOOL, rrr, r.const(sfmax2, t)), r('lt', BOOL, rra, r.const(
                sfmax2, t)), r('gt', BOOL, ff, r.const(sfmin2, t)), r('gt', BOOL, cc, r.const(sfmin2, t)),
                r('gt', BOOL, gg, r.const(sfmin2, t)), r('gt', BOOL, cca, r.const(sfmin2, t))]
        return flag(r, _all(r, test))
    for i in range(1, n + 1):
        column, line = h[:, i - 1], h[i - 1]
        c_max, c_sum = _clipped_norm(column, _at(en, i - 1, n), {0: (kz, n - 1)}, passes)
        r_max, r_sum = _clipped_norm(line, _at(en, i - 1, n), {0: (kz, n - 1)}, passes)
        ica = _first_largest(column, _at(ec, i - 1, n), None, passes)
        ira = _first_largest(line, _at(er, i - 1, n), {0: (kz, n - 1)}, passes)

        # 350-371: C, R, CA and RA; a zero C or R (CYCLE) or a NaN (the exit) not live; G = R / SCLFAC, F = 1, S
        def setup(r, v, i=i):
            cm, cs, rm, rs, at_c, at_r, kk, ext, stop, slot = v
            zero = r.const(0, t)
            cc, rrr = r('mul', t, r('sqrt', t, cs), cm), r('mul', t, r('sqrt', t, rs), rm)
            cca = _modulus(r, t, _entry(r, t, slot, h, _bounded(r, at_c, n)[1], r.const(i - 1, I32)))
            rra = _modulus(r, t, _entry(r, t, slot, h, r.const(i - 1, I32), _bounded(r, r('add', I32, at_r, kk), n)[1]))
            present = _all(r, [r('gt', BOOL, ext, r.const(0, I32)), r('not', BOOL, nonzero(r, stop)),
                               r('ne', BOOL, cc, zero), r('ne', BOOL, rrr, zero)])
            nan = r('and', BOOL, present, r('isnan', BOOL, r('add', t, r('add', t, r('add', t, cc, cca), rrr), rra)))
            alive = flag(r, r('and', BOOL, present, r('not', BOOL, nan)))
            one, gg = r.const(1, t), r('div', t, rrr, r.const(2, t))
            return [(cc, t), (rrr, t), (cca, t), (rra, t), (one, t), (gg, t), (r('add', t, cc, rrr), t), (alive, I32),
                    (flag(r, r('or', BOOL, nonzero(r, stop), nan)), I32),
                    (tested(r, alive, one, cc, cca, rrr, gg, rra, True), I32)]
        passes += _scalars([c, rr, ca, ra, f, g, s, live, halted, more],
                           [c_max, c_sum, r_max, r_sum, ica, ira, kz, en[i - 1:i], halted], setup, [h])

        # 373-381 and 385-393: the two DO WHILE loops, each pass one source iteration by SCLFAC = 2
        def stepped(first):
            def step(r, v):
                ff, cc, cca, rrr, gg, rra, alive = v
                two = r.const(2, t)
                up, down = (lambda x: r('mul', t, x, two)), (lambda x: r('div', t, x, two))
                if first:
                    made = [up(ff), up(cc), up(cca), down(rrr), down(gg), down(rra)]
                else:
                    made = [down(ff), down(cc), down(cca), up(rrr), down(gg), up(rra)]
                return [(x, t) for x in made] + [(tested(r, alive, *made, first), I32)]
            return step
        ops_first = _scalars([f, c, ca, rr, g, ra, more], [f, c, ca, rr, g, ra, live], stepped(True))
        passes.append(_Repeat(ops_first, more, carried))
        passes += _scalars([g, more], [f, c, ca, rr, g, ra, live], lambda r, v: [
            (r('div', t, v[1], r.const(2, t)), t),
            (tested(r, v[6], v[0], v[1], v[2], v[3], r('div', t, v[1], r.const(2, t)), v[5], False), I32)])
        ops_second = _scalars([f, c, ca, rr, g, ra, more], [f, c, ca, rr, g, ra, live], stepped(False))
        passes.append(_Repeat(ops_second, more, carried))

        # 397-409: the balance unless C + R >= FACTOR S or the SCALE bounds; G = 1 / F, SCALE(I) = SCALE(I) F, then
        # ZDSCAL of row I (columns K..N) by G and of column I (rows 1..L) by F
        def balanced(r, v):
            cc, rrr, ss, ff, alive, factor, conv, kk, at = v
            one, zero = r.const(1, t), r.const(0, I32)
            small = _all(r, [r('lt', BOOL, ff, one), r('lt', BOOL, factor, one),
                             r('le', BOOL, r('mul', t, ff, factor), r.const(sfmin1, t))])
            large = _all(r, [r('gt', BOOL, ff, one), r('gt', BOOL, factor, one),
                             r('ge', BOOL, factor, r('div', t, r.const(sfmax1, t), ff))])
            apply = _all(r, [nonzero(r, alive), r('lt', BOOL, r('add', t, cc, rrr), r('mul', t, r.const(0.95, t), ss)),
                             r('not', BOOL, small), r('not', BOOL, large)])
            return [(r('select', t, apply, r('mul', t, factor, ff), factor), t), (r('div', t, one, ff), t),
                    (flag(r, r('or', BOOL, nonzero(r, conv), apply)), I32),
                    (r('select', I32, apply, r('sub', I32, r.const(n + 1, I32), kk), zero), I32),
                    (r('select', I32, apply, at, zero), I32)]
        passes += _scalars([scale[i - 1:i], g, noconv, erow, ecol],
                           [c, rr, s, f, live, scale[i - 1:i], noconv, k, l], balanced)
        line = h[i - 1]
        _emit(passes, _record((n,), [(line, (0,))], [(line, (0,)), (g, (None,))], lambda r, v: [(_real_scaled(
            r, t, v[0], v[1]), t)]), loops=((erow, n),), offsets=[(line, {0: (kz, n - 1)})])
        column = h[:, i - 1]
        _emit(passes, _record((n,), [(column, (0,))], [(column, (0,)), (f, (None,))], lambda r, v: [(_real_scaled(
            r, t, v[0], v[1]), t)]), loops=((ecol, n),))
    passes += _scalars([sweep], [noconv, halted], lambda r, v: [(flag(r, r('and', BOOL, nonzero(r, v[0]), r(
        'not', BOOL, nonzero(r, v[1])))), I32)])
    ops.append(_Repeat(passes, sweep, carried))

    # 276-279 and 415-416: ILO = IHI = 1 where the rows' search returned, else K and L; unassigned after the exit
    ops += _scalars([ilo, ihi], [k, l, done, halted, ilo, ihi], lambda r, v: [(r('select', I32, nonzero(r, v[3]), v[
        4 + d], r('select', I32, nonzero(r, v[2]), r.const(1, I32), v[d])), I32) for d in (0, 1)])


# Reference-LAPACK ZLARF1F (v(1) = 1, held explicitly in v): H = I - tau v v^H applied to `block` in place, from the
# left (H, or H^H where `adjoint`: conj(tau)) or from the right (block H). v is (length, 1) and tau one-element;
# `rows` and `cols` are the block's actual extents by loop position (tensor, bound), so that two loops of equal static
# bound keep their own ranges
def _reflected(block, v, tau, rows, cols, ops, left=True, adjoint=False):
    m, k = block.shape
    t, vector = _class(block.real), v[:, 0]

    def scalar(r, x):
        return _conjugate(r, t, x) if adjoint else x
    if left:
        dot = _temporary((k,), block.dtype)
        _emit(ops, _record((k, m), [(dot, (0,))], [(vector, (1,)), (block, (1, 0))], lambda r, x: [(_times(
            r, t, _conjugate(r, t, x[0]), x[1]), t)], [PARALLEL, REDUCTION], [SUM]), loops=(cols, rows))
        _emit(ops, _record((m, k), [(block, (0, 1))], [(block, (0, 1)), (vector, (0,)), (dot, (1,)), (tau, (None,))],
                           lambda r, x: [(_plus(r, t, x[0], _times(r, t, x[1], _times(r, t, x[2], scalar(r, x[3]))),
                                                'sub'), t)]), loops=(rows, cols))
        return
    dot = _temporary((m,), block.dtype)
    _emit(ops, _record((m, k), [(dot, (0,))], [(block, (0, 1)), (vector, (1,))], lambda r, x: [(_times(
        r, t, x[0], x[1]), t)], [PARALLEL, REDUCTION], [SUM]), loops=(rows, cols))
    _emit(ops, _record((m, k), [(block, (0, 1))], [(block, (0, 1)), (dot, (0,)), (vector, (1,)), (tau, (None,))],
                       lambda r, x: [(_plus(r, t, x[0], _times(r, t, _times(r, t, x[1], scalar(r, x[3])), _conjugate(
                           r, t, x[2])), 'sub'), t)]), loops=(rows, cols))


# Reference-LAPACK SRC/zgehrd.f 249-254 and 295-299 (TAU outside ILO..IHI-1 zero; the unblocked code from I = ILO) and
# zgehd2.f 190-207 (dgehrd.f, dgehd2.f and dorghr.f alike, the real reflector its own conjugate) on one square member
# h, complex or real, in place, then (z given) zunghr.f 198-232 with zung2r's reflector sequence into z. For each static column I = 1..n-1, active where ILO <= I <= IHI-1 (else every extent 0):
# ZLARFG(IHI-I, A(I+1,I), A(I+2,I)) by _linalg_reflector clipped to the reflector's IHI-I entries and its tail; H(I)
# from the right to A(1:IHI, I+1:IHI), then H(I)^H from the left to A(I+1:IHI, I+1:N); beta into A(I+1,I) and
# v(2:) into A(I+2:IHI,I). Q = H(ILO) ... H(IHI-1) from the identity, each H(I) (reverse order) to rows and columns
# I+1..IHI. Returns TAU (its first n - 1 entries); ILO and IHI one-element int32, the source's 1-based values
def _gehd2(h, z, ilo, ihi, ops):
    n, dtype, real = h.shape[-1], h.dtype, h.real.dtype
    t = _class(h.real)
    tau = _temporary((max(n - 1, 0),), dtype)
    if z is not None and n:
        ops += _linalg_identity(z)
    if n < 2:
        return tau
    length, tail, height, width = (_temporary((n - 1,), torch.int32) for _ in range(4))

    def spans(r, v):
        low, high = v
        column, zero, one = r('add', I32, r.index(0), r.const(1, I32)), r.const(0, I32), r.const(1, I32)
        active = r('and', BOOL, r('le', BOOL, low, column), r('lt', BOOL, column, high))
        size = r('sub', I32, high, column)
        return [(r('select', I32, active, x, zero), I32) for x in (
            size, r('sub', I32, size, one), high, r('sub', I32, r.const(n, I32), column))]
    ops += _record((n - 1,), [(x, (0,)) for x in (length, tail, height, width)], [(ilo, (None,)), (ihi, (None,))],
                   spans)
    for i in range(n - 1):
        extent, below = _at(length, i, n - i - 1), _at(tail, i, n - i - 1)
        made, reflected, beta = [], _temporary((1,), dtype), _temporary((1,), real)
        v = _linalg_reflector(h[i + 1:, i], reflected[0], beta[0], made)
        _emit(ops, made, sizes={n - i - 1: extent, n - i - 2: below})
        _reflected(h[:, i + 1:], v, reflected, _at(height, i, n), extent, ops, left=False)
        _reflected(h[i + 1:, i + 1:], v, reflected, extent, _at(width, i, n - i - 1), ops, adjoint=True)
        ops += _scalars([tau[i:i + 1]], [reflected, length[i:i + 1]], lambda r, x: [(_chosen(
            r, t, r('gt', BOOL, x[1], r.const(0, I32)), x[0], (r.const(0, t), r.const(0, t))), t)])
        ops += _scalars([h[i + 1:i + 2, i]], [h[i + 1:i + 2, i], beta, length[i:i + 1]], lambda r, x: [(_chosen(
            r, t, r('gt', BOOL, x[2], r.const(0, I32)), x[1], x[0]), t)])
        if i + 2 < n:
            _emit(ops, _bounded_copy(h[i + 2:, i], v[1:, 0]), loops=(below,))
    if z is None:
        return tau
    for i in reversed(range(n - 1)):
        extent, below = _at(length, i, n - i - 1), _at(tail, i, n - i - 1)
        v = _temporary((n - i - 1, 1), dtype)
        ops += _fill(v[0, 0], 1)
        if i + 2 < n:
            _emit(ops, _bounded_copy(v[1:, 0], h[i + 2:, i]), loops=(below,))
        _reflected(z[i + 1:, i + 1:], v, tau[i:i + 1], extent, extent, ops)
    return tau


# Reference-LAPACK SRC/dlapy3.f: sqrt(x^2 + y^2 + z^2) by the largest magnitude W, their sum where W is zero or above
# the overflow threshold `huge`, of region values
def _dlapy3(r, t, x, y, z, huge):
    a, b, c = (r('abs', t, v) for v in (x, y, z))
    w = r('select', t, r('ge', BOOL, a, b), a, b)
    w = r('select', t, r('ge', BOOL, w, c), w, c)
    plain = r('or', BOOL, r('eq', BOOL, w, r.const(0, t)), r('gt', BOOL, w, r.const(huge, t)))
    safe = r('select', t, plain, r.const(1, t), w)
    q = [r('div', t, v, safe) for v in (a, b, c)]
    total = r('add', t, r('add', t, r('mul', t, q[0], q[0]), r('mul', t, q[1], q[1])), r('mul', t, q[2], q[2]))
    return r('select', t, plain, r('add', t, r('add', t, a, b), c), r('mul', t, w, r('sqrt', t, total)))


# Reference-LAPACK SRC/dlapy2.f: sqrt(x^2 + y^2) as W sqrt(1 + (Z/W)^2), W and Z the larger and smaller magnitude;
# W where Z is zero or W is above the overflow threshold `huge`; a NaN operand itself (y's where both are)
def _dlapy2(r, t, x, y, huge):
    a, b = r('abs', t, x), r('abs', t, y)
    w, z = r('select', t, r('ge', BOOL, a, b), a, b), r('select', t, r('ge', BOOL, a, b), b, a)
    plain = r('or', BOOL, r('eq', BOOL, z, r.const(0, t)), r('gt', BOOL, w, r.const(huge, t)))
    q = r('div', t, z, r('select', t, plain, r.const(1, t), w))
    made = r('select', t, plain, w, r('mul', t, w, r('sqrt', t, r('add', t, r.const(1, t), r('mul', t, q, q)))))
    made = r('select', t, r('isnan', BOOL, x), x, made)
    return r('select', t, r('isnan', BOOL, y), y, made)


# Fortran SIGN(A, B): |A| with B's sign, a zero B's sign that of 1/B (IEEE 754 -0 below +0)
def _fortran_sign(r, t, a, b):
    zero = r.const(0, t)
    negative = r('or', BOOL, r('lt', BOOL, b, zero), r('and', BOOL, r('eq', BOOL, b, zero), r(
        'lt', BOOL, r('div', t, r.const(1, t), b), zero)))
    magnitude = r('abs', t, a)
    return r('select', t, negative, r('neg', t, magnitude), magnitude)


# Reference-LAPACK SRC/dlanv2.f 159-306 in one region of real kind t: the Schur factorization of the real 2 x 2
# [A B; C D], of region values, to (A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN). Each source branch is selected, and
# the label 10 rescaling loop is unrolled to its own bound (COUNT <= 20, so at most 21 passes). The region keeps the
# source's separate products (lines 251-257, Reference-LAPACK issue 1031), with no contraction. `limits` is the real
# dtype's finfo; SAFMIN/EPS is an exact power of two for F32/F64, so SAFMN2 uses that integer exponent
def _dlanv2(r, t, a, b, c, d, limits):
    safmn2 = math.ldexp(1.0, math.trunc(math.log2(limits.tiny / limits.eps) / 2))
    safmx2, eps = 1 / safmn2, limits.eps

    def k(value):
        return r.const(value, t)

    def sel(test, x, y):
        return r('select', t, test, x, y)

    def mul(x, y):
        return r('mul', t, x, y)

    def add(x, y):
        return r('add', t, x, y)

    def sub(x, y):
        return r('sub', t, x, y)

    def neg(x):
        return r('neg', t, x)

    def sign(x, y):
        return _fortran_sign(r, t, x, y)
    zero, one, half = k(0), k(1), k(0.5)
    first = r('eq', BOOL, c, zero)
    second = r('and', BOOL, r('not', BOOL, first), r('eq', BOOL, b, zero))
    third = _all(r, [r('not', BOOL, first), r('not', BOOL, second), r('eq', BOOL, sub(a, d), zero), r(
        'ne', BOOL, sign(one, b), sign(one, c))])
    general = _all(r, [r('not', BOOL, first), r('not', BOOL, second), r('not', BOOL, third)])

    # 187-192
    temp = sub(a, d)
    p = mul(half, temp)
    ab, ac = r('abs', t, b), r('abs', t, c)
    bcmax = sel(r('ge', BOOL, ab, ac), ab, ac)
    bcmis = mul(mul(sel(r('le', BOOL, ab, ac), ab, ac), sign(one, b)), sign(one, c))
    scale = sel(r('ge', BOOL, r('abs', t, p), bcmax), r('abs', t, p), bcmax)
    z = add(mul(r('div', t, p, scale), p), mul(r('div', t, bcmax, scale), bcmis))
    real_pair = r('and', BOOL, general, r('ge', BOOL, z, k(4 * eps)))

    # 201-211: real eigenvalues
    zr = add(p, sign(mul(r('sqrt', t, scale), r('sqrt', t, z)), p))
    ar, dr = add(d, zr), sub(d, mul(r('div', t, bcmax, zr), bcmis))
    tau = _dlapy2(r, t, c, zr, limits.max)
    csr, snr, br = r('div', t, zr, tau), r('div', t, c, tau), sub(b, c)

    # 218-238: complex or nearly equal eigenvalues, TEMP and SIGMA rescaled
    sigma, running = add(b, c), r.const(1, BOOL)
    for count in range(1, 22):
        size = sel(r('ge', BOOL, r('abs', t, temp), r('abs', t, sigma)), r('abs', t, temp), r('abs', t, sigma))
        down = r('and', BOOL, running, r('ge', BOOL, size, k(safmx2)))
        up = _all(r, [running, r('not', BOOL, down), r('le', BOOL, size, k(safmn2))])
        sigma = sel(down, mul(sigma, k(safmn2)), sel(up, mul(sigma, k(safmx2)), sigma))
        temp = sel(down, mul(temp, k(safmn2)), sel(up, mul(temp, k(safmx2)), temp))
        running = r('and', BOOL, r('or', BOOL, down, up), r.const(count <= 20, BOOL))
    pc = mul(half, temp)
    tau = _dlapy2(r, t, sigma, temp, limits.max)
    cs = r('sqrt', t, mul(half, add(one, r('div', t, r('abs', t, sigma), tau))))
    sn = mul(neg(r('div', t, pc, mul(tau, cs))), sign(one, sigma))

    # 243-262
    aa, bb = add(mul(a, cs), mul(b, sn)), add(neg(mul(a, sn)), mul(b, cs))
    cc, dd = add(mul(c, cs), mul(d, sn)), add(neg(mul(c, sn)), mul(d, cs))
    ac_ = add(mul(aa, cs), mul(cc, sn))
    bc_ = add(mul(bb, cs), mul(dd, sn))
    cc_ = add(neg(mul(aa, sn)), mul(cc, cs))
    dc_ = add(neg(mul(bb, sn)), mul(dd, cs))
    mean = mul(half, add(ac_, dc_))

    # 264-291
    split = _all(r, [r('ne', BOOL, cc_, zero), r('ne', BOOL, bc_, zero), r('eq', BOOL, sign(one, bc_), sign(one, cc_))])
    turned = r('and', BOOL, r('ne', BOOL, cc_, zero), r('eq', BOOL, bc_, zero))
    sab, sac = r('sqrt', t, r('abs', t, bc_)), r('sqrt', t, r('abs', t, cc_))
    pp = sign(mul(sab, sac), cc_)
    tau1 = r('div', t, one, r('sqrt', t, r('abs', t, add(bc_, cc_))))
    cs1, sn1 = mul(sab, tau1), mul(sac, tau1)
    ax = sel(split, add(mean, pp), mean)
    dx = sel(split, sub(mean, pp), mean)
    bx = sel(split, sub(bc_, cc_), sel(turned, neg(cc_), bc_))
    cx = sel(r('or', BOOL, split, turned), zero, cc_)
    csx = sel(split, sub(mul(cs, cs1), mul(sn, sn1)), sel(turned, neg(sn), cs))
    snx = sel(split, add(mul(cs, sn1), mul(sn, cs1)), sel(turned, cs, sn))

    # each branch's A, B, C, D, CS and SN
    complex_ = r('and', BOOL, general, r('not', BOOL, real_pair))
    outs = []
    for kept, swapped, pair, other in ((a, d, ar, ax), (b, neg(c), br, bx), (c, zero, zero, cx), (d, a, dr, dx),
                                       (one, zero, csr, csx), (zero, one, snr, snx)):
        outs.append(sel(second, swapped, sel(real_pair, pair, sel(complex_, other, kept))))
    a_, b_, c_, d_, cs_, sn_ = outs

    # 298-306
    rt1i = sel(r('eq', BOOL, c_, zero), zero, mul(r('sqrt', t, r('abs', t, b_)), r('sqrt', t, r('abs', t, c_))))
    rt2i = sel(r('eq', BOOL, c_, zero), zero, neg(rt1i))
    return a_, b_, c_, d_, a_, rt1i, d_, rt2i, cs_, sn_


# Reference-LAPACK SRC/dlaln2.f 276-601 with LTRANS = false, CA = 1, D1 = D2 = 1, in one region of real kind t: the
# NA x NA (1 or 2, static) system (A - (WR + i WI)) X = SCALE B, NW = 1 (real WR) or 2 (complex WR + i WI). `a` is
# (A11,) or (A11, A12, A21, A22) and `b` (and the returned X) B(1,1), B(2,1) [NA = 2], then B(1,2), B(2,2) [NW = 2],
# of region values; returns (X, SCALE, XNORM). SMINI = max(SMIN, 2 SAFMIN), the complete pivoting of CRV/CIV by the
# source's IPIVOT, RSWAP and ZSWAP tables, the SMINI replacements and every SCALE/BNORM/BBND/XNORM overflow test;
# DLADIV is the existing complex division (Smith's)
def _dlaln2(r, t, na, nw, smin, a, b, wr, wi, limits):
    def k(value):
        return r.const(value, t)

    def sel(test, x, y):
        return r('select', t, test, x, y)

    def mul(x, y):
        return r('mul', t, x, y)

    def add(x, y):
        return r('add', t, x, y)

    def sub(x, y):
        return r('sub', t, x, y)

    def absolute(x):
        return r('abs', t, x)

    def bigger(x, y):
        return sel(r('ge', BOOL, x, y), x, y)

    def divided(x, y):
        return _complex_value(r, t, 'div', x, y, 1, None)
    smlnum = 2 * limits.tiny
    bignum = 1 / smlnum
    one, zero = k(1), k(0)
    smini = bigger(smin, k(smlnum))

    def bounded(norm, size):
        return sel(_all(r, [r('lt', BOOL, size, one), r('gt', BOOL, norm, one), r('gt', BOOL, norm, mul(
            k(bignum), size))]), r('div', t, one, norm), one)
    if na == 1:
        csr = sub(a[0], wr)
        if nw == 1:
            small = r('lt', BOOL, absolute(csr), smini)
            cnorm = sel(small, smini, absolute(csr))
            csr = sel(small, smini, csr)
            scale = bounded(absolute(b[0]), cnorm)
            x = r('div', t, mul(b[0], scale), csr)
            return (x,), scale, absolute(x)
        csi = r('neg', t, wi)
        cnorm = add(absolute(csr), absolute(csi))
        small = r('lt', BOOL, cnorm, smini)
        csr, csi, cnorm = sel(small, smini, csr), sel(small, zero, csi), sel(small, smini, cnorm)
        scale = bounded(add(absolute(b[0]), absolute(b[1])), cnorm)
        xr, xi = divided((mul(scale, b[0]), mul(scale, b[1])), (csr, csi))
        return (xr, xi), scale, add(absolute(xr), absolute(xi))
    a11, a12, a21, a22 = a
    crv = [sub(a11, wr), a21, a12, sub(a22, wr)]
    civ = [r('neg', t, wi), zero, zero, r('neg', t, wi)] if nw == 2 else [zero] * 4
    cmax, ic = zero, r.const(0, I32)
    for j in range(4):
        size = absolute(crv[j]) if nw == 1 else add(absolute(crv[j]), absolute(civ[j]))
        larger = r('gt', BOOL, size, cmax)
        cmax, ic = sel(larger, size, cmax), r('select', I32, larger, r.const(j + 1, I32), ic)
    pivot = {1: (1, 2, 3, 4), 2: (2, 1, 4, 3), 3: (3, 4, 1, 2), 4: (4, 3, 2, 1)}

    def picked(values, row):
        made = values[pivot[4][row] - 1]
        for column in (3, 2, 1):
            made = sel(r('eq', BOOL, ic, r.const(column, I32)), values[pivot[column][row] - 1], made)
        return made
    rswap = r('or', BOOL, r('eq', BOOL, ic, r.const(2, I32)), r('eq', BOOL, ic, r.const(4, I32)))
    zswap = r('or', BOOL, r('eq', BOOL, ic, r.const(3, I32)), r('eq', BOOL, ic, r.const(4, I32)))
    identity = r('lt', BOOL, cmax, smini)
    if nw == 1:
        b11, b21 = b
        bnorm = bigger(absolute(b11), absolute(b21))
        held = bounded(bnorm, smini)
        temp = r('div', t, held, smini)
        plain = ((mul(temp, b11), mul(temp, b21)), held, mul(temp, bnorm))
        ur11, cr21, ur12, cr22 = (picked(crv, row) for row in range(4))
        ur11r = r('div', t, one, ur11)
        lr21 = mul(ur11r, cr21)
        ur22 = sub(cr22, mul(ur12, lr21))
        ur22 = sel(r('lt', BOOL, absolute(ur22), smini), smini, ur22)
        br1, br2 = sel(rswap, b21, b11), sel(rswap, b11, b21)
        br2 = sub(br2, mul(lr21, br1))
        bbnd = bigger(absolute(mul(br1, mul(ur22, ur11r))), absolute(br2))
        scale = sel(_all(r, [r('gt', BOOL, bbnd, one), r('lt', BOOL, absolute(ur22), one), r(
            'ge', BOOL, bbnd, mul(k(bignum), absolute(ur22)))]), r('div', t, one, bbnd), one)
        xr2 = r('div', t, mul(br2, scale), ur22)
        xr1 = sub(mul(mul(scale, br1), ur11r), mul(xr2, mul(ur11r, ur12)))
        x = [sel(zswap, xr2, xr1), sel(zswap, xr1, xr2)]
        xnorm = bigger(absolute(xr1), absolute(xr2))
    else:
        b11, b21, b12, b22 = b
        bnorm = bigger(add(absolute(b11), absolute(b12)), add(absolute(b21), absolute(b22)))
        held = bounded(bnorm, smini)
        temp = r('div', t, held, smini)
        plain = (tuple(mul(temp, v) for v in b), held, mul(temp, bnorm))
        ur11, cr21, ur12, cr22 = (picked(crv, row) for row in range(4))
        ui11, ci21, ui12, ci22 = (picked(civ, row) for row in range(4))
        wide = r('gt', BOOL, absolute(ur11), absolute(ui11))
        ratio = sel(wide, r('div', t, ui11, ur11), r('div', t, ur11, ui11))
        grown = add(one, mul(ratio, ratio))
        ur11r_a = sel(wide, r('div', t, one, mul(ur11, grown)), r('neg', t, mul(ratio, r('div', t, k(-1), mul(
            ui11, grown)))))
        ui11r_a = sel(wide, r('neg', t, mul(ratio, r('div', t, one, mul(ur11, grown)))), r('div', t, k(-1), mul(
            ui11, grown)))
        offreal = r('or', BOOL, r('eq', BOOL, ic, r.const(1, I32)), r('eq', BOOL, ic, r.const(4, I32)))
        ur11r = sel(offreal, ur11r_a, r('div', t, one, ur11))
        ui11r = sel(offreal, ui11r_a, zero)
        lr21 = sel(offreal, mul(cr21, ur11r), mul(cr21, ur11r))
        li21 = sel(offreal, mul(cr21, ui11r), mul(ci21, ur11r))
        ur12s = mul(ur12, ur11r)
        ui12s = sel(offreal, mul(ur12, ui11r), mul(ui12, ur11r))
        ur22 = sel(offreal, sub(cr22, mul(ur12, lr21)), add(sub(cr22, mul(ur12, lr21)), mul(ui12, li21)))
        ui22 = sel(offreal, sub(ci22, mul(ur12, li21)), sub(r('neg', t, mul(ur12, li21)), mul(ui12, lr21)))
        u22abs = add(absolute(ur22), absolute(ui22))
        small = r('lt', BOOL, u22abs, smini)
        ur22, ui22 = sel(small, smini, ur22), sel(small, zero, ui22)
        br1, br2 = sel(rswap, b21, b11), sel(rswap, b11, b21)
        bi1, bi2 = sel(rswap, b22, b12), sel(rswap, b12, b22)
        br2 = add(sub(br2, mul(lr21, br1)), mul(li21, bi1))
        bi2 = sub(sub(bi2, mul(li21, br1)), mul(lr21, bi1))
        bbnd = bigger(mul(add(absolute(br1), absolute(bi1)), mul(u22abs, add(absolute(ur11r), absolute(ui11r)))),
                      add(absolute(br2), absolute(bi2)))
        rescaled = _all(r, [r('gt', BOOL, bbnd, one), r('lt', BOOL, u22abs, one), r(
            'ge', BOOL, bbnd, mul(k(bignum), u22abs))])
        scale = sel(rescaled, r('div', t, one, bbnd), one)
        br1, bi1, br2, bi2 = (sel(rescaled, mul(scale, v), v) for v in (br1, bi1, br2, bi2))
        xr2, xi2 = divided((br2, bi2), (ur22, ui22))
        xr1 = add(sub(sub(mul(ur11r, br1), mul(ui11r, bi1)), mul(ur12s, xr2)), mul(ui12s, xi2))
        xi1 = sub(sub(add(mul(ui11r, br1), mul(ur11r, bi1)), mul(ui12s, xr2)), mul(ur12s, xi2))
        x = [sel(zswap, xr2, xr1), sel(zswap, xr1, xr2), sel(zswap, xi2, xi1), sel(zswap, xi1, xi2)]
        xnorm = bigger(add(absolute(xr1), absolute(xi1)), add(absolute(xr2), absolute(xi2)))
    further = _all(r, [r('gt', BOOL, xnorm, one), r('gt', BOOL, cmax, one), r(
        'gt', BOOL, xnorm, r('div', t, k(bignum), cmax))])
    temp = r('div', t, cmax, k(bignum))
    x = [sel(further, mul(temp, v), v) for v in x]
    xnorm, scale = sel(further, mul(temp, xnorm), xnorm), sel(further, mul(temp, scale), scale)
    return (tuple(sel(identity, p, v) for p, v in zip(plain[0], x)), sel(identity, plain[1], scale),
            sel(identity, plain[2], xnorm))


# Reference-LAPACK SRC/zlarfg.f 144-211 with N = 2, in one region: ALPHA and X (complex record values) to (ALPHA on
# exit, BETA real; TAU; X on exit). Its KNT rescaling loop (at most 20 passes, the source's bound) and the undo are
# unrolled; SAFMIN = DLAMCH('S') / DLAMCH('E')
def _larfg2(r, t, alpha, x, limits):
    def c(value):
        return r.const(value, t)

    def chosen(test, a, b):
        return _chosen(r, t, test, a, b)
    safmin = limits.tiny / (limits.eps / 2)
    rsafmn, zero = 1 / safmin, c(0)
    ar, ai = _complex_parts(r, t, alpha)
    xv = _complex_parts(r, t, x)
    trivial = r('and', BOOL, r('eq', BOOL, _modulus(r, t, xv), zero), r('eq', BOOL, ai, zero))

    def signed(re, im, norm):
        d = _dlapy3(r, t, re, im, norm, limits.max)
        return r('select', t, r('lt', BOOL, re, zero), d, r('neg', t, d))
    beta = signed(ar, ai, _modulus(r, t, xv))
    small = r('lt', BOOL, r('abs', t, beta), c(safmin))
    large = r('and', BOOL, r('not', BOOL, small), r('gt', BOOL, r('abs', t, beta), c(limits.max / 2)))
    running, knt, sb, sr, si, sx = small, r.const(0, I32), beta, ar, ai, xv
    for _ in range(20):
        knt = r('select', I32, running, r('add', I32, knt, r.const(1, I32)), knt)
        sx = chosen(running, _real_scaled(r, t, sx, c(rsafmn)), sx)
        sb, si, sr = (r('select', t, running, r('mul', t, v, c(rsafmn)), v) for v in (sb, si, sr))
        running = r('and', BOOL, running, r('lt', BOOL, r('abs', t, sb), c(safmin)))
    lx, lr, li = _real_scaled(r, t, xv, c(safmin)), r('mul', t, ar, c(safmin)), r('mul', t, ai, c(safmin))
    fx = chosen(small, sx, chosen(large, lx, xv))
    fr = r('select', t, small, sr, r('select', t, large, lr, ar))
    fi = r('select', t, small, si, r('select', t, large, li, ai))
    rescaled = r('or', BOOL, small, large)
    fb = r('select', t, rescaled, signed(fr, fi, _modulus(r, t, fx)), beta)
    tau = (r('div', t, r('sub', t, fb, fr), fb), r('div', t, r('neg', t, fi), fb))
    inverse = _complex_value(r, t, 'div', (c(1), zero), (r('sub', t, fr, fb), fi), 1, None)
    made = _complex_value(r, t, 'mul', inverse, fx, 1, None)
    undone = fb
    for j in range(20):
        undone = r('select', t, r('gt', BOOL, knt, r.const(j, I32)), r('mul', t, undone, c(safmin)), undone)
    undone = r('select', t, large, r('mul', t, undone, c(rsafmn)), undone)
    return (r('select', t, trivial, ar, undone), chosen(trivial, (zero, zero), tau), chosen(trivial, xv, made))


# Reference-LAPACK SRC/zlahqr.f 250-573 on one complex upper Hessenberg member h in place: WANTT = WANTZ = (z given),
# ILOZ = ILO and IHIZ = IHI (zgeev/zhseqr). w (complex, n) receives W(ILO..IHI); ilo, ihi and info are one-element
# int32 (INFO = 0, or I where the source fails); `held` holds the caller's temporaries the iteration carries. One
# _Repeat pass is one DO 130 iteration: labels 30 and 140 are flattened (on deflation W(I) = H(I,I), KDEFL = 0,
# I = L - 1, then L = ILO and ITS = 0), so ITS runs 0..ITMAX for each I. The searches over K and M take the first hit
# from the top as the largest. Each bulge step K = M..I-1 is a static position with its actual active extents
def _zlahqr(h, z, w, ilo, ihi, info, ops, held=()):
    n, t, real, dtype = h.shape[-1], _class(h.real), h.real.dtype, h.dtype
    limits, want = torch.finfo(real), z is not None
    safmin, ulp = limits.tiny, limits.eps
    ops += _fill(info, 0)
    if not n:
        return
    (i, lo, its, kdefl, more, span, base, i1, i2, i1z, kfound, go, cut, mspan, mfound, m, fl, erow, ecol, ezz, iz,
     ip) = (_temporary((1,), torch.int32) for _ in range(22))
    shift, v1, v2, t1, vv2, tmp = (_temporary((1,), dtype) for _ in range(6))
    beta, t2, rt = (_temporary((1,), real) for _ in range(3))
    sites = max(n - 1, 1)
    fresh, eleft, eright, ez, erows, ecols, ezc = (_temporary((sites,), torch.int32) for _ in range(7))
    carried = (*held, h, *([z] if want else []), w, ilo, ihi, info, i, lo, its, kdefl, more, span, base, i1, i2, i1z,
               kfound, go, cut, mspan, mfound, m, fl, erow, ecol, ezz, iz, ip, shift, v1, v2, t1, vv2, tmp, beta, t2,
               rt, fresh, eleft, eright, ez, erows, ecols, ezc)

    def c(r, value):
        return r.const(value, t)

    def k(r, value):
        return r.const(value, I32)

    def flag(r, x):
        return _convert(r, x, BOOL, I32)

    def larger(r, a, b, kind):
        return r('select', kind, r('ge', BOOL, a, b), a, b)

    def smaller(r, a, b, kind):
        return r('select', kind, r('le', BOOL, a, b), a, b)

    def at(r, slot, row, column):
        return _entry(r, t, slot, h, _bounded(r, row, n)[1], _bounded(r, column, n)[1])

    def smallest(r, low, high):
        nh = r('add', I32, r('sub', I32, high, low), k(r, 1))
        return r('mul', t, c(r, safmin), r('div', t, r('cast', t, nh), c(r, ulp)))

    def divided(r, x, s):
        return tuple(r('div', t, p, s) for p in _complex_parts(r, t, x))

    # 250 and 262-267, with the state at label 30 (I = IHI, L = ILO, ITS = 0, KDEFL = 0; I1 = 1, I2 = N where WANTT):
    # the trash, H(J+2,J) for J = ILO..IHI-2 (H(IHI,IHI-2) the last) and H(J+3,J) for J = ILO..IHI-3, zero
    def started(r, v):
        low, high = v
        zero, one = k(r, 0), k(r, 1)
        return [(r('sub', I32, low, one), I32), (high, I32), (low, I32), (zero, I32), (zero, I32),
                (flag(r, r('ge', BOOL, high, low)), I32), (r('sub', I32, high, low), I32), (one, I32),
                (k(r, n), I32), (zero, I32)] + [(larger(r, r('sub', I32, r('sub', I32, high, low), k(r, d)), zero, I32),
                                                 I32) for d in (1, 2)]
    ops += _scalars([base, i, lo, its, kdefl, more, span, i1, i2, i1z, erow, ecol], [ilo, ihi], started)
    for band, extent in ((2, erow), (3, ecol)):
        if n > band:
            trash = h[...]
            _emit(ops, _record((n - band,), [(trash, ({0: 1, None: band}, 0))], [], lambda r, v: [(c(r, 0), t)]),
                  loops=((extent, n - band),), offsets=[(trash, {0: (base, n - 1), 1: (base, n - 1)})])

    # 269-291: each subdiagonal H(I,I-1), I = ILO+1..IHI in order, made real: SC = conj(H/CABS1(H)) / |H/CABS1(H)|,
    # H(I,I-1) = |H(I,I-1)|, row I (columns I..JHI) by SC, column I (rows JLO..min(JHI, I+1)) and Z's by conj(SC)
    for row in range(2, n + 1):
        def realized(r, v, row=row):
            x, low, high = v
            zero = k(r, 0)
            active = _all(r, [r('le', BOOL, r('add', I32, low, k(r, 1)), k(r, row)), r('le', BOOL, k(r, row), high),
                              r('ne', BOOL, _part(r, t, x, 1), c(r, 0))])
            safe = r('select', t, active, _cabs1(r, t, x), c(r, 1))
            first = divided(r, x, safe)
            norm = r('select', t, active, _modulus(r, t, first), c(r, 1))
            sc = divided(r, _conjugate(r, t, first), norm)
            jhi, jlo = (k(r, n), k(r, 1)) if want else (high, low)
            spans = [r('add', I32, r('sub', I32, jhi, k(r, row)), k(r, 1)),
                     r('add', I32, r('sub', I32, smaller(r, jhi, k(r, row + 1), I32), jlo), k(r, 1)),
                     r('add', I32, r('sub', I32, high, low), k(r, 1))]
            return [(_chosen(r, t, active, _modulus(r, t, x), x), t), (sc, t)] + [
                (r('select', I32, active, x, zero), I32) for x in spans]
        entry = h[row - 1:row, row - 2]
        ops += _scalars([entry, tmp, erow, ecol, ezz], [entry, ilo, ihi], realized)
        line = h[row - 1, row - 1:]
        _emit(ops, _record((n - row + 1,), [(line, (0,))], [(line, (0,)), (tmp, (None,))], lambda r, v: [(_times(
            r, t, v[1], v[0]), t)]), loops=((erow, n - row + 1),))
        for target, at_row, extent in ((h, None if want else base, ecol), (z, base, ezz)):
            if target is None:
                continue
            column = target[:, row - 1]
            _emit(ops, _record((n,), [(column, (0,))], [(column, (0,)), (tmp, (None,))], lambda r, v: [(_times(
                r, t, _conjugate(r, t, v[1]), v[0]), t)]), loops=((extent, n),),
                  offsets=[(column, {0: (at_row, n - 1)})] if at_row is not None else [])

    # 338-367: the K search over L+1..I (`span` = I - L entries): CABS1(H(K,K-1)) <= SMLNUM, or Ahues and Tisseur's
    # test where |Re H(K,K-1)| <= ULP TST (TST zero: |Re H(K-1,K-2)| where K-2 >= ILO, |Re H(K+1,K)| where K+1 <= IHI)
    def deflated(r, v):
        low, first, last, slot = v
        kk = r('add', I32, r('add', I32, low, k(r, 1)), r.index(0))
        one, two, three = k(r, 1), k(r, 2), k(r, 3)
        sub, dkm, dk = at(r, slot, r('sub', I32, kk, one), r('sub', I32, kk, two)), \
            at(r, slot, r('sub', I32, kk, two), r('sub', I32, kk, two)), at(r, slot, r('sub', I32, kk, one), r(
                'sub', I32, kk, one))
        sup = at(r, slot, r('sub', I32, kk, two), r('sub', I32, kk, one))
        before = at(r, slot, r('sub', I32, kk, two), r('sub', I32, kk, three))
        after = at(r, slot, kk, r('sub', I32, kk, one))
        smlnum = smallest(r, first, last)
        tst = r('add', t, _cabs1(r, t, dkm), _cabs1(r, t, dk))
        grown = r('select', t, r('ge', BOOL, r('sub', I32, kk, two), first), r('add', t, tst, r(
            'abs', t, _part(r, t, before, 0))), tst)
        grown = r('select', t, r('le', BOOL, r('add', I32, kk, one), last), r('add', t, grown, r(
            'abs', t, _part(r, t, after, 0))), grown)
        tst = r('select', t, r('eq', BOOL, tst, c(r, 0)), grown, tst)
        ab = larger(r, _cabs1(r, t, sub), _cabs1(r, t, sup), t)
        ba = smaller(r, _cabs1(r, t, sub), _cabs1(r, t, sup), t)
        gap = _cabs1(r, t, _plus(r, t, dkm, dk, 'sub'))
        aa, bb = larger(r, _cabs1(r, t, dk), gap, t), smaller(r, _cabs1(r, t, dk), gap, t)
        s = r('add', t, aa, ab)
        criterion = r('le', BOOL, r('mul', t, ba, r('div', t, ab, s)), larger(r, smlnum, r('mul', t, c(r, ulp), r(
            'mul', t, bb, r('div', t, aa, s))), t))
        hit = r('or', BOOL, r('le', BOOL, _cabs1(r, t, sub), smlnum), r('and', BOOL, r(
            'le', BOOL, r('abs', t, _part(r, t, sub, 0)), r('mul', t, c(r, ulp), tst)), criterion))
        return [(r('select', I32, hit, kk, k(r, 0)), I32)]
    passes = []
    if n > 1:
        _emit(passes, _record_extreme((n - 1,), kfound, (None,), [(lo, (None,)), (ilo, (None,)), (ihi, (None,)),
                                                                   (h, None)], deflated, [REDUCTION]),
              loops=((span, n - 1),))
    else:
        passes += _fill(kfound, 0)

    # 366-420: L = K, H(L,L-1) = 0 where L > ILO (`cut`), the exit (L >= I: W(I) = H(I,I), label 140), else
    # KDEFL += 1, I1 = L and I2 = I where not WANTT, and the shift T: exceptional at mod(KDEFL, 2 KEXSH) = 0
    # (DAT1 |Re H(I,I-1)| + H(I,I)) and mod(KDEFL, KEXSH) = 0 (DAT1 |Re H(L+1,L)| + H(L,L)), else Wilkinson's
    def stepped(r, v):
        found, low, top, first, defl, count, slot = v
        zero, one = k(r, 0), k(r, 1)
        new = larger(r, found, low, I32)
        leave = r('ge', BOOL, new, top)
        diagonal = at(r, slot, r('sub', I32, top, one), r('sub', I32, top, one))
        for part in (0, 1):
            r.scatter(0, [_bounded(r, r('sub', I32, top, one), n)[1], k(r, part)], _part(r, t, diagonal, part), leave)
        kd = r('add', I32, defl, one)
        two = k(r, 2)
        hii, hii1 = diagonal, at(r, slot, r('sub', I32, top, one), r('sub', I32, top, two))
        hl1l, hll = at(r, slot, new, r('sub', I32, new, one)), at(r, slot, r('sub', I32, new, one), r(
            'sub', I32, new, one))
        him, himm = at(r, slot, r('sub', I32, top, two), r('sub', I32, top, one)), \
            at(r, slot, r('sub', I32, top, two), r('sub', I32, top, two))
        hii = _complex_parts(r, t, hii)
        exceptional = [(r('add', t, r('mul', t, c(r, 0.75), r('abs', t, _part(r, t, x, 0))), _part(r, t, d, 0)),
                        _part(r, t, d, 1)) for x, d in ((hii1, hii), (hl1l, hll))]
        u = _complex_value(r, t, 'mul', _complex_value(r, t, 'sqrt', _complex_parts(r, t, him), None, 1, None),
                           _complex_value(r, t, 'sqrt', _complex_parts(r, t, hii1), None, 1, None), 1, None)
        su = _cabs1(r, t, u)
        x = _real_scaled(r, t, _plus(r, t, himm, hii, 'sub'), c(r, 0.5))
        sx = _cabs1(r, t, x)
        s = larger(r, su, sx, t)
        safe = r('select', t, r('eq', BOOL, s, c(r, 0)), c(r, 1), s)
        xs, us = divided(r, x, safe), divided(r, u, safe)
        y = _real_scaled(r, t, _complex_value(r, t, 'sqrt', _plus(r, t, _times(r, t, xs, xs), _times(r, t, us, us)),
                                              None, 1, None), s)
        unit = divided(r, x, r('select', t, r('gt', BOOL, sx, c(r, 0)), sx, c(r, 1)))
        flip = r('and', BOOL, r('gt', BOOL, sx, c(r, 0)), r('lt', BOOL, r('add', t, r(
            'mul', t, unit[0], _part(r, t, y, 0)), r('mul', t, unit[1], _part(r, t, y, 1))), c(r, 0)))
        y = _chosen(r, t, flip, _negated(r, t, y), y)
        wilkinson = _chosen(r, t, r('ne', BOOL, su, c(r, 0)), _plus(r, t, hii, _times(r, t, u, _complex_value(
            r, t, 'div', u, _plus(r, t, x, y), 1, None)), 'sub'), hii)
        shifted = _chosen(r, t, r('eq', BOOL, r('rem', I32, kd, k(r, 20)), zero), exceptional[0], _chosen(
            r, t, r('eq', BOOL, r('rem', I32, kd, k(r, 10)), zero), exceptional[1], wilkinson))
        stay = r('not', BOOL, leave)
        edge = (one, k(r, n)) if want else (new, top)
        return [(diagonal, t), (r('select', I32, leave, first, new), I32),
                (r('select', I32, leave, r('sub', I32, new, one), top), I32), (r('select', I32, leave, zero, kd), I32),
                (r('select', I32, leave, zero, count), I32), (flag(r, stay), I32),
                (r('select', I32, r('gt', BOOL, new, first), new, zero), I32), (edge[0], I32), (edge[1], I32),
                (r('sub', I32, edge[0], one), I32),
                (r('select', I32, stay, larger(r, r('sub', I32, r('sub', I32, top, one), new), zero, I32), zero), I32),
                (shifted, t)]
    passes += _record((1,), [(w, None)] + [(x, (None,)) for x in (lo, i, kdefl, its, go, cut, i1, i2, i1z, mspan,
                                                                    shift)],
                      [(x, (None,)) for x in (kfound, lo, i, ilo, kdefl, its)] + [(h, None)], stepped)

    def severed(r, v):
        for part in (0, 1):
            r.scatter(0, [_bounded(r, r('sub', I32, v[0], k(r, 1)), n)[1], _bounded(r, r('sub', I32, v[0], k(r, 2)),
                                                                                  n)[1], k(r, part)],
                      c(r, 0), r('gt', BOOL, v[0], k(r, 0)))
        return [(c(r, 0), t)]
    passes += _record((1,), [(h, None)], [(cut, (None,))], severed)

    # 424-453: the M search over L+1..I-1 (`mspan` entries), |H10| |H21| <= ULP CABS1(H11S) (CABS1(H11) + CABS1(H22))
    # of the scaled H11S = (H(M,M) - T) / S and H21 = Re H(M+1,M) / S; M = L where none, V = (H11S, H21) at M
    def leading(r, slot, mm, shifted):
        one = k(r, 1)
        h11, h22 = at(r, slot, r('sub', I32, mm, one), r('sub', I32, mm, one)), at(r, slot, mm, mm)
        h21 = _part(r, t, at(r, slot, mm, r('sub', I32, mm, one)), 0)
        h11s = _plus(r, t, h11, shifted, 'sub')
        s = r('add', t, _cabs1(r, t, h11s), r('abs', t, h21))
        return h11, h22, divided(r, h11s, s), r('div', t, h21, s)

    def bulged(r, v):
        low, shifted, slot = v
        mm = r('add', I32, r('add', I32, low, k(r, 1)), r.index(0))
        h11, h22, h11s, h21 = leading(r, slot, mm, shifted)
        h10 = _part(r, t, at(r, slot, r('sub', I32, mm, k(r, 1)), r('sub', I32, mm, k(r, 2))), 0)
        hit = r('le', BOOL, r('mul', t, r('abs', t, h10), r('abs', t, h21)), r('mul', t, c(r, ulp), r(
            'mul', t, _cabs1(r, t, h11s), r('add', t, _cabs1(r, t, h11), _cabs1(r, t, h22)))))
        return [(r('select', I32, hit, mm, k(r, 0)), I32)]

    def started_at(r, v):
        found, low, shifted, slot = v
        mm = larger(r, found, low, I32)
        _, _, h11s, h21 = leading(r, slot, mm, shifted)
        return [(mm, I32), (h11s, t), (h21, t)]
    if n > 1:
        _emit(passes, _record_extreme((n - 1,), mfound, (None,), [(lo, (None,)), (shift, (None,)), (h, None)], bulged,
                                      [REDUCTION]), loops=((mspan, n - 1),))
        passes += _record((1,), [(m, (None,)), (v1, (None,)), (v2, (None,))],
                          [(mfound, (None,)), (lo, (None,)), (shift, (None,)), (h, None)], started_at)

        # 457-535: each static position K = 1..n-1 active where M <= K <= I-1, its actual extents: rows K, K+1 over
        # columns K..I2; columns K, K+1 over rows I1..min(K+2, I) and Z's over ILO..IHI; at K = M > L the extra
        # scaling over rows I1..I and columns M..I2 (Z's columns M..I)
        def placed(r, v):
            mm, top, first_row, last_column, low, alive, first, last = v
            kk, zero, one = r('add', I32, r.index(0), k(r, 1)), k(r, 0), k(r, 1)
            active = _all(r, [r('ne', BOOL, alive, zero), r('le', BOOL, mm, kk), r('le', BOOL, kk, r('sub', I32, top,
                                                                                                         one))])
            extra = _all(r, [active, r('eq', BOOL, kk, mm), r('gt', BOOL, mm, low)])
            spans = [(active, r('add', I32, r('sub', I32, last_column, kk), one)),
                     (active, r('add', I32, r('sub', I32, smaller(r, r('add', I32, kk, k(r, 2)), top, I32),
                                                       first_row), one)),
                     (active, r('add', I32, r('sub', I32, last, first), one)),
                     (extra, r('add', I32, r('sub', I32, top, first_row), one)),
                     (extra, r('add', I32, r('sub', I32, last_column, mm), one)),
                     (extra, r('add', I32, r('sub', I32, top, mm), one))]
            return [(flag(r, r('and', BOOL, active, r('gt', BOOL, kk, mm))), I32)] + [
                (r('select', I32, test, size, zero), I32) for test, size in spans]
        passes += _record((n - 1,), [(x, (0,)) for x in (fresh, eleft, eright, ez, erows, ecols, ezc)],
                          [(x, (None,)) for x in (m, i, i1, i2, lo, go, ilo, ihi)], placed)

        def swept(r, v):
            a, b, tau, real_t2, x2 = v
            s = _plus(r, t, _times(r, t, tau, a), _times(r, t, real_t2, b))
            return [(_plus(r, t, a, s, 'sub'), t), (_plus(r, t, b, _times(r, t, s, _conjugate(r, t, x2)), 'sub'), t)]

        def chased(r, v):
            a, b, tau, real_t2, x2 = v
            s = _plus(r, t, _times(r, t, _conjugate(r, t, tau), a), _times(r, t, real_t2, b))
            return [(_plus(r, t, a, s, 'sub'), t), (_plus(r, t, b, _times(r, t, s, x2), 'sub'), t)]
        reflector = [(x, (None,)) for x in (t1, t2, vv2)]
        for kk in range(1, n):
            k0, own = kk - 1, (lambda x, k0=kk - 1: x[k0:k0 + 1])

            # V from H(K,K-1), H(K+1,K-1) where K > M (else V at M), ZLARFG(2, V(1), V(2)), T2 = Re(T1 V2)
            def reflected(r, v, kk=kk):
                if kk > 1:
                    first, second, renewed, top, below = v
                    test = r('ne', BOOL, renewed, k(r, 0))
                    alpha, x = _chosen(r, t, test, top, first), _chosen(r, t, test, below, second)
                else:
                    alpha, x = v[0], v[1]
                b, tau, made = _larfg2(r, t, alpha, x, limits)
                return [(b, t), (tau, t), (made, t), (_complex_value(r, t, 'mul', tau, made, 1, None)[0], t)]
            given = [(v1, (None,)), (v2, (None,)), (own(fresh), (None,))]
            if kk > 1:
                given += [(h[k0:k0 + 1, k0 - 1], (None,)), (h[k0 + 1:k0 + 2, k0 - 1], (None,))]
            passes += _record((1,), [(beta, (None,)), (t1, (None,)), (vv2, (None,)), (t2, (None,))], given, reflected)
            if kk > 1:
                pair = h[k0:k0 + 2, k0 - 1]
                passes += _record((2,), [(pair, (0,))], [(pair, (0,)), (beta, (None,)), (own(fresh), (None,))],
                                  lambda r, v: [(_chosen(r, t, r('ne', BOOL, v[2], k(r, 0)), _chosen(
                                      r, t, r('eq', BOOL, r.index(0), k(r, 0)), v[1], c(r, 0)), v[0]), t)])
            upper, lower = h[k0, k0:], h[k0 + 1, k0:]
            _emit(passes, _record((n - k0,), [(upper, (0,)), (lower, (0,))],
                                  [(upper, (0,)), (lower, (0,))] + reflector, chased),
                  loops=((own(eleft), n - k0),))
            for target, at_row, extent in ((h, i1z, eright), (z, base, ez)):
                if target is None:
                    continue
                left_column, right_column = target[:, k0], target[:, k0 + 1]
                _emit(passes, _record((n,), [(left_column, (0,)), (right_column, (0,))],
                                      [(left_column, (0,)), (right_column, (0,))] + reflector, swept),
                      loops=((own(extent), n),), offsets=[(left_column, {0: (at_row, n - 1)}),
                                                          (right_column, {0: (at_row, n - 1)})])

            # 510-534: TEMP = (1 - T1) / |1 - T1|; H(M+1,M) by conj(TEMP), H(M+2,M+1) by TEMP where M+2 <= I, then for
            # J = M..I, J /= M+1: row J's columns J+1..I2 by TEMP, column J's rows I1..J-1 (and Z's) by conj(TEMP)
            def phase(r, tau):
                d = _plus(r, t, (c(r, 1), c(r, 0)), tau, 'sub')
                return divided(r, d, _modulus(r, t, d))

            def rescaled(r, v, kk=kk):
                x, tau, low_row, top, last_column = v
                temp, one = phase(r, tau), k(r, 1)
                p = r('add', I32, r('add', I32, low_row, one), r.index(0))
                q = r('add', I32, k(r, kk), r.index(1))

                def equal(a, value):
                    return r('eq', BOOL, a, k(r, value))
                tests = [(_all(r, [equal(p, kk + 1), equal(q, kk)]), _conjugate(r, t, temp)),
                         (_all(r, [equal(p, kk + 2), equal(q, kk + 1), r('le', BOOL, k(r, kk + 2), top)]), temp),
                         (_all(r, [r('le', BOOL, k(r, kk), p), r('le', BOOL, p, top), r('not', BOOL, equal(p, kk + 1)),
                                   r('le', BOOL, r('add', I32, p, one), q), r('le', BOOL, q, last_column)]), temp),
                         (_all(r, [r('le', BOOL, q, top), r('not', BOOL, equal(q, kk + 1)),
                                   r('le', BOOL, r('add', I32, p, one), q)]), _conjugate(r, t, temp))]
                for test, factor in tests:
                    x = _chosen(r, t, test, _times(r, t, factor, x), x)
                return [(x, t)]
            block = h[:, k0:]
            _emit(passes, _record((n, n - k0), [(block, (0, 1))], [(block, (0, 1)), (t1, (None,)), (i1z, (None,)),
                                                                    (i, (None,)), (i2, (None,))], rescaled),
                  loops=((own(erows), n), (own(ecols), n - k0)), offsets=[(block, {0: (i1z, n - 1)})])
            if want:
                block = z[:, k0:]
                _emit(passes, _record((n, n - k0), [(block, (0, 1))], [(block, (0, 1)), (t1, (None,))],
                                      lambda r, v, kk=kk: [(_chosen(r, t, r('ne', BOOL, r(
                                          'add', I32, k(r, kk), r.index(1)), k(r, kk + 1)), _times(
                                          r, t, _conjugate(r, t, phase(r, v[1])), v[0]), v[0]), t)]),
                      loops=((own(ez), n), (own(ezc), n - k0)), offsets=[(block, {0: (base, n - 1)})])

    # 539-550: H(I,I-1) made real where its imaginary part is nonzero: RTEMP = |TEMP|, TEMP = TEMP / RTEMP, row I's
    # columns I+1..I2 by conj(TEMP), column I's rows I1..I-1 and Z's rows ILOZ..IHIZ by TEMP
    def realness(r, v):
        top, alive, first_row, last_column, low, high, slot = v
        zero, one = k(r, 0), k(r, 1)
        x = at(r, slot, r('sub', I32, top, one), r('sub', I32, top, k(r, 2)))
        turned = r('and', BOOL, r('ne', BOOL, alive, zero), r('ne', BOOL, _part(r, t, x, 1), c(r, 0)))
        size = _modulus(r, t, x)
        temp = divided(r, x, r('select', t, turned, size, c(r, 1)))
        return [(temp, t), (size, t), (flag(r, turned), I32)] + [(r('select', I32, turned, s, zero), I32) for s in (
            r('sub', I32, last_column, top), r('sub', I32, top, first_row), r('add', I32, r('sub', I32, high, low), one))] \
            + [(_bounded(r, r('sub', I32, top, one), n)[1], I32), (_bounded(r, top, n)[1], I32)]
    passes += _record((1,), [(x, (None,)) for x in (tmp, rt, fl, erow, ecol, ezz, iz, ip)],
                      [(x, (None,)) for x in (i, go, i1, i2, ilo, ihi)] + [(h, None)], realness)

    def settled(r, v):
        size, turned, row = v
        column, test = _bounded(r, r('sub', I32, row, k(r, 1)), n)[1], r('ne', BOOL, turned, k(r, 0))
        r.scatter(0, [row, column, k(r, 0)], size, test)
        r.scatter(0, [row, column, k(r, 1)], c(r, 0), test)
        return [(c(r, 0), t)]
    passes += _record((1,), [(h, None)], [(rt, (None,)), (fl, (None,)), (iz, (None,))], settled)
    line = h[...]
    _emit(passes, _record((n,), [(line, (None, 0))], [(line, (None, 0)), (tmp, (None,))], lambda r, v: [(_times(
        r, t, _conjugate(r, t, v[1]), v[0]), t)]), loops=((erow, n),), offsets=[(line, {0: (iz, n - 1), 1: (ip, n - 1)})])
    for target, at_row, extent in ((h, i1z, ecol), (z, base, ezz)):
        if target is None:
            continue
        column = target[...]
        _emit(passes, _record((n,), [(column, (0, None))], [(column, (0, None)), (tmp, (None,))], lambda r, v: [(_times(
            r, t, v[1], v[0]), t)]), loops=((extent, n),), offsets=[(column, {0: (at_row, n - 1), 1: (iz, n - 1)})])

    # 336 and 552-557: ITS += 1 after a sweep; past ITMAX = 30 max(10, NH) INFO = I (the source's failure), else the
    # next DO 130 iteration while I >= ILO, its K search over I - L entries
    def counted(r, v):
        count, alive, top, low, high, failure, current = v
        zero, one = k(r, 0), k(r, 1)
        done = r('select', I32, r('ne', BOOL, alive, zero), r('add', I32, count, one), count)
        itmax = r('mul', I32, k(r, 30), larger(r, k(r, 10), r('add', I32, r('sub', I32, high, low), one), I32))
        fail = r('and', BOOL, r('ne', BOOL, alive, zero), r('gt', BOOL, done, itmax))
        return [(done, I32), (r('select', I32, fail, top, failure), I32),
                (flag(r, r('and', BOOL, r('not', BOOL, fail), r('ge', BOOL, top, low))), I32),
                (larger(r, r('sub', I32, top, current), zero, I32), I32)]
    passes += _scalars([its, info, more, span], [its, go, i, ilo, ihi, info, lo], counted)
    ops.append(_Repeat(passes, more, carried))


# Reference-LAPACK SRC/dlahqr.f 252-616 on one real upper Hessenberg member h in place: WANTT = WANTZ = (z given),
# ILOZ = ILO and IHIZ = IHI (dgeev/dhseqr). wr and wi (real, n) receive WR/WI(ILO..IHI); ilo, ihi and info are
# one-element int32 (INFO = 0, or I where the source fails); `held` holds the caller's temporaries the iteration
# carries. One _Repeat pass is one DO 140 iteration: labels 20 and 150 are flattened (on deflation of one eigenvalue,
# WR(I) = H(I,I), or of a pair, DLANV2 with its rotations of H and Z; then KDEFL = 0, I = L - 1, L = ILO, ITS = 0), so
# ITS runs 0..ITMAX for each I. The K and M searches take the first hit from the top as the largest. Each double-shift
# step K = M..I-1 is a static position with its actual extents, NR = min(3, I-K+1)
def _dlahqr(h, z, wr, wi, ilo, ihi, info, ops, held=()):
    n, t, real = h.shape[-1], _class(h), h.dtype
    limits, want = torch.finfo(real), z is not None
    safmin, ulp = limits.tiny, limits.eps
    ops += _fill(info, 0)
    if not n:
        return
    (i, lo, its, kdefl, more, span, base, i1, i2, i1z, kfound, go, cut, mspan, mfound, m, pair, erow, ecol, ezz, p1,
     p2, p3) = (_temporary((1,), torch.int32) for _ in range(23))
    rt1r, rt1i, rt2r, rt2i, v1, v2, v3, na, nb, nc, nd, cs, sn, beta, t1, t2, t3 = \
        (_temporary((1,), real) for _ in range(17))
    sites = max(n - 1, 1)
    fresh, rlen, rtail, ewrite, eclear, l3, l2, r3, r2, z3, z2 = (_temporary((sites,), torch.int32) for _ in range(11))
    carried = (*held, h, *([z] if want else []), wr, wi, ilo, ihi, info, i, lo, its, kdefl, more, span, base, i1, i2,
               i1z, kfound, go, cut, mspan, mfound, m, pair, erow, ecol, ezz, p1, p2, p3, rt1r, rt1i, rt2r, rt2i, v1,
               v2, v3, na, nb, nc, nd, cs, sn, beta, t1, t2, t3, fresh, rlen, rtail, ewrite, eclear, l3, l2, r3, r2,
               z3, z2)

    def c(r, value):
        return r.const(value, t)

    def k(r, value):
        return r.const(value, I32)

    def flag(r, x):
        return _convert(r, x, BOOL, I32)

    def larger(r, a, b, kind):
        return r('select', kind, r('ge', BOOL, a, b), a, b)

    def smaller(r, a, b, kind):
        return r('select', kind, r('le', BOOL, a, b), a, b)

    def at(r, slot, row, column):
        return r('extract', t, _bounded(r, row, n)[1], _bounded(r, column, n)[1], integer=slot)

    def less(r, x, value):
        return r('sub', I32, x, k(r, value))

    def absolute(r, x):
        return r('abs', t, x)

    # 252 and 265-305, the state at label 20 (I = IHI, L = ILO, ITS = 0, KDEFL = 0; I1 = 1, I2 = N where WANTT): the
    # trash, H(J+2,J) for J = ILO..IHI-2 (H(IHI,IHI-2) the last) and H(J+3,J) for J = ILO..IHI-3, zero
    def started(r, v):
        low, high = v
        zero, one = k(r, 0), k(r, 1)
        return [(less(r, low, 1), I32), (high, I32), (low, I32), (zero, I32), (zero, I32),
                (flag(r, r('ge', BOOL, high, low)), I32), (r('sub', I32, high, low), I32), (one, I32), (k(r, n), I32),
                (zero, I32)] + [(larger(r, less(r, r('sub', I32, high, low), d), zero, I32), I32) for d in (1, 2)]
    ops += _scalars([base, i, lo, its, kdefl, more, span, i1, i2, i1z, erow, ecol], [ilo, ihi], started)
    for band, extent in ((2, erow), (3, ecol)):
        if n > band:
            trash = h[...]
            _emit(ops, _record((n - band,), [(trash, ({0: 1, None: band}, 0))], [], lambda r, v: [(c(r, 0), t)]),
                  loops=((extent, n - band),), offsets=[(trash, {0: (base, n - 1), 1: (base, n - 1)})])

    # 319-344: the K search over L+1..I (`span` = I - L entries): |H(K,K-1)| <= SMLNUM (SAFMIN NH / ULP), or Ahues and
    # Tisseur's test where |H(K,K-1)| <= ULP TST (TST zero: |H(K-1,K-2)| where K-2 >= ILO, |H(K+1,K)| where K+1 <= IHI)
    def smallest(r, low, high):
        nh = r('add', I32, r('sub', I32, high, low), k(r, 1))
        return r('mul', t, c(r, safmin), r('div', t, r('cast', t, nh), c(r, ulp)))

    def deflated(r, v):
        low, first, last, slot = v
        kk = r('add', I32, r('add', I32, low, k(r, 1)), r.index(0))
        sub, dkm, dk = at(r, slot, less(r, kk, 1), less(r, kk, 2)), at(r, slot, less(r, kk, 2), less(r, kk, 2)), \
            at(r, slot, less(r, kk, 1), less(r, kk, 1))
        sup, before, after = at(r, slot, less(r, kk, 2), less(r, kk, 1)), at(r, slot, less(r, kk, 2), less(r, kk, 3)), \
            at(r, slot, kk, less(r, kk, 1))
        smlnum = smallest(r, first, last)
        tst = r('add', t, absolute(r, dkm), absolute(r, dk))
        grown = r('select', t, r('ge', BOOL, less(r, kk, 2), first), r('add', t, tst, absolute(r, before)), tst)
        grown = r('select', t, r('le', BOOL, r('add', I32, kk, k(r, 1)), last), r('add', t, grown, absolute(
            r, after)), grown)
        tst = r('select', t, r('eq', BOOL, tst, c(r, 0)), grown, tst)
        ab, ba = larger(r, absolute(r, sub), absolute(r, sup), t), smaller(r, absolute(r, sub), absolute(r, sup), t)
        gap = absolute(r, r('sub', t, dkm, dk))
        aa, bb = larger(r, absolute(r, dk), gap, t), smaller(r, absolute(r, dk), gap, t)
        s = r('add', t, aa, ab)
        criterion = r('le', BOOL, r('mul', t, ba, r('div', t, ab, s)), larger(r, smlnum, r('mul', t, c(r, ulp), r(
            'mul', t, bb, r('div', t, aa, s))), t))
        hit = r('or', BOOL, r('le', BOOL, absolute(r, sub), smlnum), r('and', BOOL, r(
            'le', BOOL, absolute(r, sub), r('mul', t, c(r, ulp), tst)), criterion))
        return [(r('select', I32, hit, kk, k(r, 0)), I32)]
    passes = []
    if n > 1:
        _emit(passes, _record_extreme((n - 1,), kfound, (None,), [(lo, (None,)), (ilo, (None,)), (ihi, (None,)),
                                                                   (h, None)], deflated, [REDUCTION]),
              loops=((span, n - 1),))
    else:
        passes += _fill(kfound, 0)

    # 346-435 and 570-613: L = K, H(L,L-1) = 0 where L > ILO (`cut`); L >= I-1 the exit (L = I: WR(I) = H(I,I),
    # WI(I) = 0; L = I-1: DLANV2 of the trailing 2 x 2, its WR/WI pair and rotation); else KDEFL += 1, I1 = L and
    # I2 = I where not WANTT, and the shifts: exceptional at mod(KDEFL, 2 KEXSH) = 0 (at I) and mod(KDEFL, KEXSH) = 0
    # (at L), DAT1 = 3/4 and DAT2 = -0.4375, else Francis's double shift from the trailing 2 x 2, scaled by S
    def stepped(r, v):
        found, low, top, first, last, defl, count, slot = v
        zero, one, two = k(r, 0), k(r, 1), k(r, 2)
        new = larger(r, found, low, I32)
        leave = r('ge', BOOL, new, less(r, top, 1))
        single = r('and', BOOL, leave, r('eq', BOOL, new, top))
        double = r('and', BOOL, leave, r('eq', BOOL, new, less(r, top, 1)))
        i0, im = _bounded(r, less(r, top, 1), n)[1], _bounded(r, less(r, top, 2), n)[1]
        hmm, hmi, him, hii = at(r, slot, im, im), at(r, slot, im, i0), at(r, slot, i0, im), at(r, slot, i0, i0)
        r.scatter(0, [i0], hii, single)
        r.scatter(1, [i0], c(r, 0), single)
        kd = r('add', I32, defl, one)
        ex = [r('add', t, absolute(r, at(r, slot, i0, im)), absolute(r, at(r, slot, im, less(r, top, 3)))),
              r('add', t, absolute(r, at(r, slot, new, less(r, new, 1))), absolute(r, at(r, slot, r(
                  'add', I32, new, one), new)))]
        diagonals = [hii, at(r, slot, less(r, new, 1), less(r, new, 1))]
        exceptional = [(r('add', t, r('mul', t, c(r, 0.75), s), d), r('mul', t, c(r, -0.4375), s), s,
                        r('add', t, r('mul', t, c(r, 0.75), s), d)) for s, d in zip(ex, diagonals)]
        francis = (hmm, hmi, him, hii)
        twentieth = r('eq', BOOL, r('rem', I32, kd, k(r, 20)), zero)
        tenth = r('eq', BOOL, r('rem', I32, kd, k(r, 10)), zero)
        h11, h12, h21, h22 = (r('select', t, twentieth, e0, r('select', t, tenth, e1, f)) for e0, e1, f in zip(
            exceptional[0], exceptional[1], francis))
        s = r('add', t, r('add', t, r('add', t, absolute(r, h11), absolute(r, h12)), absolute(r, h21)), absolute(r, h22))
        safe = r('select', t, r('eq', BOOL, s, c(r, 0)), c(r, 1), s)
        h11, h21, h12, h22 = (r('div', t, x, safe) for x in (h11, h21, h12, h22))
        tr = r('div', t, r('add', t, h11, h22), c(r, 2))
        det = r('sub', t, r('mul', t, r('sub', t, h11, tr), r('sub', t, h22, tr)), r('mul', t, h12, h21))
        rtdisc = r('sqrt', t, absolute(r, det))
        conjugate = r('ge', BOOL, det, c(r, 0))
        plus, minus = r('add', t, tr, rtdisc), r('sub', t, tr, rtdisc)
        nearer = r('le', BOOL, absolute(r, r('sub', t, plus, h22)), absolute(r, r('sub', t, minus, h22)))
        shift = r('select', t, nearer, r('mul', t, plus, s), r('mul', t, minus, s))
        zero_s = r('eq', BOOL, s, c(r, 0))
        first_r = r('select', t, zero_s, c(r, 0), r('select', t, conjugate, r('mul', t, tr, s), shift))
        real_s = r('or', BOOL, zero_s, r('not', BOOL, conjugate))
        first_i = r('select', t, real_s, c(r, 0), r('mul', t, rtdisc, s))
        second_i = r('select', t, real_s, c(r, 0), r('neg', t, r('mul', t, rtdisc, s)))
        stay = r('not', BOOL, leave)
        edge = (one, k(r, n)) if want else (new, top)
        rotated = flag(r, double)
        spans = [r('sub', I32, k(r, n), top) if want else zero, larger(r, less(r, less(r, top, 1), 1), zero, I32)
                 if want else zero, r('add', I32, r('sub', I32, last, first), one)]
        return [(c(r, 0), t), (c(r, 0), t), (r('select', I32, leave, first, new), I32),
                (r('select', I32, leave, less(r, new, 1), top), I32), (r('select', I32, leave, zero, kd), I32),
                (r('select', I32, leave, zero, count), I32), (flag(r, stay), I32),
                (r('select', I32, r('gt', BOOL, new, first), new, zero), I32), (edge[0], I32), (edge[1], I32),
                (less(r, edge[0], 1), I32),
                (r('select', I32, stay, larger(r, r('sub', I32, less(r, top, 2), new), zero, I32), zero), I32),
                (first_r, t), (first_i, t), (first_r, t), (second_i, t), (rotated, I32)] + [
            (r('select', I32, double, x, zero), I32) for x in spans] + [
            (im, I32), (i0, I32), (_bounded(r, top, n)[1], I32)]
    passes += _record((1,), [(wr, None), (wi, None)] + [(x, (None,)) for x in (
        lo, i, kdefl, its, go, cut, i1, i2, i1z, mspan, rt1r, rt1i, rt2r, rt2i, pair, erow, ecol, ezz, p1, p2, p3)],
                      [(x, (None,)) for x in (kfound, lo, i, ilo, ihi, kdefl, its)] + [(h, None)], stepped)

    # 351: H(L,L-1) = 0 where L > ILO
    def severed(r, v):
        r.scatter(0, [_bounded(r, less(r, v[0], 1), n)[1], _bounded(r, less(r, v[0], 2), n)[1]], c(r, 0),
                  r('gt', BOOL, v[0], k(r, 0)))
        return [(c(r, 0), t)]
    passes += _record((1,), [(h, None)], [(cut, (None,))], severed)

    # 578-605, a once-gate on the pair's deflation (`pair`, L = I-1, its indices p1 = I-1 and p2 = I kept from before
    # I and L moved): DLANV2 of H(I-1:I, I-1:I), WR/WI(I-1) and WR/WI(I) its eigenvalues, H's 2 x 2 its standard form,
    # then DROT(X, Y, CS, SN) (X = CS X + SN Y, Y = CS Y - SN X) of rows I-1, I over columns I+1..I2 and columns I-1, I
    # over rows I1..I-2 where WANTT, and Z's columns I-1, I over ILOZ..IHIZ where WANTZ
    def schur(r, v):
        row, column, slot = v
        made = _dlanv2(r, t, at(r, slot, row, row), at(r, slot, row, column), at(r, slot, column, row),
                       at(r, slot, column, column), limits)
        a_, b_, c_, d_, r1r, r1i, r2r, r2i, cs_, sn_ = made
        always = r.const(1, BOOL)
        for output, at_row, value in ((0, row, r1r), (1, row, r1i), (0, column, r2r), (1, column, r2i)):
            r.scatter(output, [at_row], value, always)
        return [(c(r, 0), t), (c(r, 0), t)] + [(x, t) for x in (a_, b_, c_, d_, cs_, sn_)]
    gated = _record((1,), [(wr, None), (wi, None)] + [(x, (None,)) for x in (na, nb, nc, nd, cs, sn)],
                    [(p1, (None,)), (p2, (None,)), (h, None)], schur)

    def standard(r, v):
        a_, b_, c_, d_, row, column = v
        always = r.const(1, BOOL)
        for x, y, value in ((row, row, a_), (row, column, b_), (column, row, c_), (column, column, d_)):
            r.scatter(0, [x, y], value, always)
        return [(c(r, 0), t)]
    gated += _record((1,), [(h, None)], [(x, (None,)) for x in (na, nb, nc, nd, p1, p2)], standard)

    def rotation(r, v):
        x, y, cosine, sine = v
        return [(r('add', t, r('mul', t, cosine, x), r('mul', t, sine, y)), t),
                (r('sub', t, r('mul', t, cosine, y), r('mul', t, sine, x)), t)]
    targets = [((None, 0), h, erow, {0: (p1, n - 1), 1: (p3, n - 1)}, {0: (p2, n - 1), 1: (p3, n - 1)})] if want else []
    targets += [((0, None), h, ecol, {0: (i1z, n - 1), 1: (p1, n - 1)}, {0: (i1z, n - 1), 1: (p2, n - 1)})] if want else []
    targets += [((0, None), z, ezz, {0: (base, n - 1), 1: (p1, n - 1)}, {0: (base, n - 1), 1: (p2, n - 1)})] if want \
        else []
    for picks, target, extent, first, second in targets:
        x, y = target[...], target[...]
        _emit(gated, _record((n,), [(x, picks), (y, picks)], [(x, picks), (y, picks), (cs, (None,)), (sn, (None,))],
                             rotation), loops=((extent, n),), offsets=[(x, first), (y, second)])
    gated += _fill(pair, 0)
    passes.append(_Repeat(gated, pair, carried))

    # 439-462: the M search over L+1..I-2 (`mspan` entries), H21S = H(M+1,M) / S, V from the shifts scaled by S, and
    # |H(M,M-1)| (|V2| + |V3|) <= ULP |V1| (|H(M-1,M-1)| + |H(M,M)| + |H(M+1,M+1)|); M = L where none
    def leading(r, slot, mm, shifts):
        one = k(r, 1)
        r1r, r1i, r2r, r2i = shifts
        hm = at(r, slot, less(r, mm, 1), less(r, mm, 1))
        hm1, hmm1 = at(r, slot, mm, mm), at(r, slot, mm, less(r, mm, 1))
        s = r('add', t, r('add', t, absolute(r, r('sub', t, hm, r2r)), absolute(r, r2i)), absolute(r, hmm1))
        h21s = r('div', t, hmm1, s)
        x1 = r('sub', t, r('add', t, r('mul', t, h21s, at(r, slot, less(r, mm, 1), mm)), r('mul', t, r(
            'sub', t, hm, r1r), r('div', t, r('sub', t, hm, r2r), s))), r('mul', t, r1i, r('div', t, r2i, s)))
        x2 = r('mul', t, h21s, r('sub', t, r('sub', t, r('add', t, hm, hm1), r1r), r2r))
        x3 = r('mul', t, h21s, at(r, slot, r('add', I32, mm, one), mm))
        s = r('add', t, r('add', t, absolute(r, x1), absolute(r, x2)), absolute(r, x3))
        return hm, hm1, [r('div', t, x, s) for x in (x1, x2, x3)]

    def bulged(r, v):
        low, *shifts, slot = v
        mm = r('add', I32, r('add', I32, low, k(r, 1)), r.index(0))
        hm, hm1, (x1, x2, x3) = leading(r, slot, mm, shifts)
        hit = r('le', BOOL, r('mul', t, absolute(r, at(r, slot, less(r, mm, 1), less(r, mm, 2))), r(
            'add', t, absolute(r, x2), absolute(r, x3))), r('mul', t, r('mul', t, c(r, ulp), absolute(r, x1)), r(
                'add', t, r('add', t, absolute(r, at(r, slot, less(r, mm, 2), less(r, mm, 2))), absolute(r, hm)),
                absolute(r, hm1))))
        return [(r('select', I32, hit, mm, k(r, 0)), I32)]

    def started_at(r, v):
        found, low, *shifts, slot = v
        mm = larger(r, found, low, I32)
        return [(mm, I32)] + [(x, t) for x in leading(r, slot, mm, shifts)[2]]
    shifts = [(x, (None,)) for x in (rt1r, rt1i, rt2r, rt2i)]
    if n > 2:
        _emit(passes, _record_extreme((n - 2,), mfound, (None,), [(lo, (None,))] + shifts + [(h, None)], bulged,
                                      [REDUCTION]), loops=((mspan, n - 2),))
    else:
        passes += _fill(mfound, 0)
    passes += _record((1,), [(m, (None,)), (v1, (None,)), (v2, (None,)), (v3, (None,))],
                      [(mfound, (None,)), (lo, (None,))] + shifts + [(h, None)], started_at)

    # 466-561: each static position K = 1..n-1 active where M <= K <= I-1, NR = 3 where K <= I-2: rows K..K+NR-1 over
    # columns K..I2; columns K..K+NR-1 over rows I1..min(K+3, I) (NR = 3) or I1..I (NR = 2), and Z's over ILOZ..IHIZ
    def placed(r, v):
        mm, top, first_row, last_column, low, alive, first, last = v
        kk, zero, one = r('add', I32, r.index(0), k(r, 1)), k(r, 0), k(r, 1)
        active = _all(r, [r('ne', BOOL, alive, zero), r('le', BOOL, mm, kk), r('le', BOOL, kk, less(r, top, 1))])
        nr3 = r('le', BOOL, kk, less(r, top, 2))
        renewed = r('and', BOOL, active, r('gt', BOOL, kk, mm))
        rescaled = _all(r, [active, r('eq', BOOL, kk, mm), r('gt', BOOL, mm, low)])
        triple, double = r('and', BOOL, active, nr3), r('and', BOOL, active, r('not', BOOL, nr3))
        nr = r('select', I32, nr3, k(r, 3), k(r, 2))
        columns = r('add', I32, r('sub', I32, last_column, kk), one)
        rows3 = r('add', I32, r('sub', I32, smaller(r, r('add', I32, kk, k(r, 3)), top, I32), first_row), one)
        rows2 = r('add', I32, r('sub', I32, top, first_row), one)
        nz = r('add', I32, r('sub', I32, last, first), one)

        def on(test, x):
            return r('select', I32, test, x, zero)
        return [(flag(r, renewed), I32), (nr, I32), (less(r, nr, 1), I32),
                (r('select', I32, renewed, k(r, 2), r('select', I32, rescaled, one, zero)), I32),
                (flag(r, r('and', BOOL, renewed, nr3)), I32)] + [
            (on(test, x), I32) for test, x in ((triple, columns), (double, columns), (triple, rows3), (double, rows2),
                                               (triple, nz), (double, nz))]
    if n > 1:
        passes += _record((n - 1,), [(x, (0,)) for x in (fresh, rlen, rtail, ewrite, eclear, l3, l2, r3, r2, z3, z2)],
                          [(x, (None,)) for x in (m, i, i1, i2, lo, go, ilo, ihi)], placed)

    # 502-507, 512-517, 523-528 (NR = 3) and 535-558 (NR = 2): SUM = X(K) + V2 X(K+1) [+ V3 X(K+2)], each X(K+d) -=
    # SUM T(d+1), the entries `nr` of one row or column, then V2..VNR, then T1..TNR
    def applied(r, v):
        nr = (len(v) + 1) // 3
        entries, vs, taus = v[:nr], v[nr:2 * nr - 1], v[2 * nr - 1:]
        s = entries[0]
        for e, x in zip(entries[1:], vs):
            s = r('add', t, s, r('mul', t, x, e))
        return [(r('sub', t, e, r('mul', t, s, f)), t) for e, f in zip(entries, taus)]
    for kk in range(1, n):
        k0, count = kk - 1, min(3, n - kk + 1)
        own = (lambda x, k0=k0: x[k0:k0 + 1])

        # 477-480: V's NR entries, H(K..K+NR-1, K-1) where K > M (DCOPY), else V at M; then DLARFG(NR, V(1), V(2))
        # by the existing reflector owner (_linalg_reflector) over that clipped vector, its v(2:NR) V2..VNR, TAU T1
        # and beta V(1) on exit
        vector = _temporary((count,), real)

        def sourced(r, v):
            *column, x1_, x2_, x3_, renewed = v
            p = r.index(0)
            incoming = r('select', t, r('eq', BOOL, p, k(r, 0)), x1_, r('select', t, r('eq', BOOL, p, k(r, 1)), x2_, x3_))
            return [(r('select', t, r('ne', BOOL, renewed, k(r, 0)), column[0], incoming) if column else incoming, t)]
        given = ([(h[k0:k0 + count, k0 - 1], (0,))] if kk > 1 else []) + [(x, (None,)) for x in (v1, v2, v3)] + [
            (own(fresh), (None,))]
        _emit(passes, _record((count,), [(vector, (0,))], given, sourced), loops=(_at(rlen, k0, count),))
        made = []
        reflector = _linalg_reflector(vector, t1[0], beta[0], made)
        _emit(passes, made, sizes={count: _at(rlen, k0, count), count - 1: _at(rtail, k0, count - 1)})

        # 493-497: T2 = T1 V2 and (NR = 3) T3 = T1 V3
        tail = [reflector[1 + d:2 + d, 0] for d in range(count - 1)]
        passes += _record((1,), [(x, (None,)) for x in (t2, t3)[:count - 1]], [(t1, (None,))] + [
            (x, (None,)) for x in tail], lambda r, v: [(r('mul', t, v[0], x), t) for x in v[1:]])

        # 481-492: where K > M, H(K,K-1) = V(1) and H(K+1,K-1) = 0 (extent 2), and H(K+2,K-1) = 0 where K < I-1;
        # where K = M > L, H(K,K-1) (1 - T1) (extent 1); else no store
        if kk > 1:
            head = h[k0:k0 + 2, k0 - 1]
            _emit(passes, _record((2,), [(head, (0,))], [(head, (0,)), (beta, (None,)), (t1, (None,)), (own(fresh), (
                None,))], lambda r, v: [(r('select', t, r('ne', BOOL, v[3], k(r, 0)), r('select', t, r(
                    'eq', BOOL, r.index(0), k(r, 0)), v[1], c(r, 0)), r('mul', t, v[0], r('sub', t, c(r, 1), v[2]))),
                    t)]), loops=((own(ewrite), 2),))
            if count == 3:
                cleared = h[k0 + 2:k0 + 3, k0 - 1]
                _emit(passes, _record((1,), [(cleared, (0,))], [], lambda r, v: [(c(r, 0), t)]),
                      loops=((own(eclear), 1),))

        # the reflector's rows K..K+NR-1 over columns K..I2, columns over I1..min(K+3, I) (NR = 3) or I1..I (NR = 2),
        # and Z's over ILOZ..IHIZ, each NR its own records and extents
        vs, taus = [(x, (None,)) for x in tail], [(x, (None,)) for x in (t1, t2, t3)]
        for nr, rows_extent, columns_extent, z_extent in ((3, l3, r3, z3), (2, l2, r2, z2)):
            if nr > count:
                continue
            factors = vs[:nr - 1] + taus[:nr]
            rows = [h[k0 + d, k0:] for d in range(nr)]
            _emit(passes, _record((n - k0,), [(x, (0,)) for x in rows], [(x, (0,)) for x in rows] + factors, applied),
                  loops=((own(rows_extent), n - k0),))
            for target, at_row, extent in ((h, i1z, columns_extent), (z, base, z_extent)):
                if target is None:
                    continue
                columns = [target[:, k0 + d] for d in range(nr)]
                _emit(passes, _record((n,), [(x, (0,)) for x in columns], [(x, (0,)) for x in columns] + factors,
                                      applied), loops=((own(extent), n),),
                      offsets=[(x, {0: (at_row, n - 1)}) for x in columns])

    # 315 and 563-568: ITS += 1 after a sweep; past ITMAX = 30 max(10, NH) INFO = I (the source's failure), else the
    # next DO 140 iteration while I >= ILO, its K search over I - L entries
    def counted(r, v):
        number, alive, top, low, high, failure, current = v
        zero, one = k(r, 0), k(r, 1)
        done = r('select', I32, r('ne', BOOL, alive, zero), r('add', I32, number, one), number)
        itmax = r('mul', I32, k(r, 30), larger(r, k(r, 10), r('add', I32, r('sub', I32, high, low), one), I32))
        fail = r('and', BOOL, r('ne', BOOL, alive, zero), r('gt', BOOL, done, itmax))
        return [(done, I32), (r('select', I32, fail, top, failure), I32),
                (flag(r, r('and', BOOL, r('not', BOOL, fail), r('ge', BOOL, top, low))), I32),
                (larger(r, r('sub', I32, top, current), zero, I32), I32)]
    passes += _scalars([its, info, more, span], [its, go, i, ilo, ihi, info, lo], counted)
    ops.append(_Repeat(passes, more, carried))


# Reference-LAPACK SRC/dtrevc.f 342-519 and 521-702 (SIDE = 'R', HOWMNY = 'B'): h the real quasi-triangular Schur form
# (read only), z the Schur vectors, vr the packed right eigenvectors Z X (a real eigenvalue's column; a pair's
# columns KI-1 and KI its real and imaginary parts), each scaled by 1 / its largest |x| (a pair's |Re| + |Im|). Column
# KI is real (IP = 0), the pair's second (T(KI,KI-1) /= 0, IP = -1) or its first (T(KI+1,KI) /= 0, skipped as the
# source's IP = 1); xr and xi (n x n) hold each KI's WORK(1+N:) and WORK(1+N2:) by column, and all the solves run
# together in the source's block order J = KI-1 (KI-2 for a pair)..1, the static rounds J = n-1..1, each a 1 x 1
# block or the 2 x 2 block ending at J (T(J,J-1) /= 0), J the first row of a 2 x 2 skipped
def _dtrevc(h, z, vr, ops):
    n, t, real = h.shape[-1], _class(h), h.dtype
    if not n:
        return
    limits = torch.finfo(real)
    ulp = limits.eps
    smlnum = limits.tiny * (n / ulp)
    bignum = (1 - ulp) / smlnum
    wrk, wik, smin, cnorm, s1, s2, scl, car, cai, c11, c21, c12, c22, emax = (_temporary((n,), real) for _ in range(14))
    realc, bottom, top, act, moved = (_temporary((n,), torch.int32) for _ in range(5))
    need, e1, e2, u1, u2 = (_temporary((1,), torch.int32) for _ in range(5))
    xr, xi, packed = _temporary((n, n), real), _temporary((n, n), real), _temporary((n, n), real)

    def k(r, value):
        return r.const(value, t)

    def i32(r, value):
        return r.const(value, I32)

    def flag(r, test):
        return _convert(r, test, BOOL, I32)

    def at(r, slot, row, column):
        return r('extract', t, _bounded(r, row, n)[1], _bounded(r, column, n)[1], integer=slot)

    def absolute(r, x):
        return r('abs', t, x)

    def nonzero(r, x):
        return r('ne', BOOL, x, i32(r, 0))

    # 342-357 and 372-400: T(KI,KI), WI = sqrt|T(KI,KI-1)| sqrt|T(KI-1,KI)| (a pair), SMIN = max(ULP (|WR| + |WI|),
    # SMLNUM), the column's kind, and a pair's initial solve WORK(KI-1+N) and WORK(KI+N2) (529-535)
    def kinds(r, v):
        diagonal, slot = v
        c = r.index(0)
        sub, sup = at(r, slot, c, r('sub', I32, c, i32(r, 1))), at(r, slot, r('sub', I32, c, i32(r, 1)), c)
        following = at(r, slot, r('add', I32, c, i32(r, 1)), c)
        second = r('and', BOOL, r('ge', BOOL, c, i32(r, 1)), r('ne', BOOL, sub, k(r, 0)))
        first = _all(r, [r('le', BOOL, c, i32(r, n - 2)), r('ne', BOOL, following, k(r, 0)), r('not', BOOL, second)])
        wi = r('select', t, second, r('mul', t, r('sqrt', t, absolute(r, sub)), r('sqrt', t, absolute(r, sup))),
               k(r, 0))
        size = r('mul', t, k(r, ulp), r('add', t, absolute(r, diagonal), absolute(r, wi)))
        wide = r('ge', BOOL, absolute(r, sup), absolute(r, sub))
        return [(diagonal, t), (wi, t), (r('select', t, r('ge', BOOL, size, k(r, smlnum)), size, k(r, smlnum)), t),
                (flag(r, r('and', BOOL, r('not', BOOL, second), r('not', BOOL, first))), I32), (flag(r, second), I32),
                (flag(r, first), I32), (r('select', t, wide, k(r, 1), r('div', t, r('neg', t, wi), sub)), t),
                (r('select', t, wide, r('div', t, wi, sup), k(r, 1)), t)]
    ops += _record((n,), [(x, (0,)) for x in (wrk, wik, smin, realc, bottom, top, s1, s2)],
                   [(h, (0, 0)), (h, None)], kinds)
    ops += _record((n, n), [(cnorm, (0,))], [(h, (1, 0))], lambda r, v: [(r('select', t, r(
        'lt', BOOL, r.index(1), r.index(0)), absolute(r, v[0]), k(r, 0)), t)], [PARALLEL, REDUCTION], [SUM])

    # 406-412 and 536-544: the right-hand sides, a real KI's X = (-T(1:KI-1,KI), 1), a pair's (-WORK(KI-1+N)
    # T(1:KI-2,KI-1), WORK(KI-1+N), 0) and (-WORK(KI+N2) T(1:KI-2,KI), 0, WORK(KI+N2)); zero elsewhere
    def sides(r, v):
        isreal, second, first, last, slot = v
        row, c = r.index(0), r.index(1)
        one, zero = k(r, 1), k(r, 0)
        cm = r('sub', I32, c, i32(r, 1))
        real_x = r('select', t, r('lt', BOOL, row, c), r('neg', t, at(r, slot, row, c)), r(
            'select', t, r('eq', BOOL, row, c), one, zero))
        pair_r = r('select', t, r('lt', BOOL, row, cm), r('neg', t, r('mul', t, first, at(r, slot, row, cm))), r(
            'select', t, r('eq', BOOL, row, cm), first, zero))
        pair_i = r('select', t, r('lt', BOOL, row, cm), r('neg', t, r('mul', t, last, at(r, slot, row, c))), r(
            'select', t, r('eq', BOOL, row, c), last, zero))
        pair = nonzero(r, second)
        return [(r('select', t, pair, pair_r, r('select', t, nonzero(r, isreal), real_x, zero)), t),
                (r('select', t, pair, pair_i, zero), t)]
    ops += _record((n, n), [(xr, (0, 1)), (xi, (0, 1))], [(realc, (1,)), (bottom, (1,)), (s1, (1,)), (s2, (1,)),
                                                        (h, None)], sides)

    # one round's 1 x 1 or 2 x 2 solve per column (DLALN2), then 443-448 (474-480, 575-581, 611-621): where XNORM > 1
    # and WORK(J) (max(WORK(J-1), WORK(J))) > BIGNUM / XNORM, X by 1 / XNORM (the complex 2 x 2 by REC = 1 / XNORM)
    # and SCALE likewise; a column takes part where real, or a pair with J <= KI-2
    def solved(r, v, na):
        rows = 2 * na
        xs, a, (wr_, wi_, bound, isreal, second), norms = v[:rows], v[rows:rows + na * na], v[rows + na * na:
                                                                                            rows + na * na + 5], v[-na:]
        pair = nonzero(r, second)
        active = r('or', BOOL, nonzero(r, isreal), r('and', BOOL, pair, r('ge', BOOL, r.index(0), i32(r, 1))))
        real_x, real_s, real_n = _dlaln2(r, t, na, 1, bound, a, xs[:na], wr_, k(r, 0), limits)
        pair_x, pair_s, pair_n = _dlaln2(r, t, na, 2, bound, a, xs, wr_, wi_, limits)
        x = [r('select', t, pair, p, q) for p, q in zip(pair_x, list(real_x) + list(xs[na:]))]
        scale, xnorm = r('select', t, pair, pair_s, real_s), r('select', t, pair, pair_n, real_n)
        beta = norms[0] if na == 1 else r('select', t, r('ge', BOOL, norms[0], norms[1]), norms[0], norms[1])
        over = r('and', BOOL, r('gt', BOOL, xnorm, k(r, 1)), r('gt', BOOL, beta, r('div', t, k(r, bignum), xnorm)))
        if na == 1:
            x = [r('select', t, over, r('div', t, e, xnorm), e) for e in x]
            scale = r('select', t, over, r('div', t, scale, xnorm), scale)
        else:
            rec = r('div', t, k(r, 1), xnorm)
            x = [r('select', t, over, r('select', t, pair, r('mul', t, e, rec), r('div', t, e, xnorm)), e) for e in x]
            scale = r('select', t, over, r('select', t, pair, r('mul', t, scale, rec), r('div', t, scale, xnorm)),
                      scale)
        x = [r('select', t, active, e, old) for e, old in zip(x, xs)]
        scale = r('select', t, active, scale, k(r, 1))
        coefficients = [r('select', t, active, r('neg', t, e), k(r, 0)) for e in x]
        return [(e, t) for e in x] + [(scale, t)] + [(e, t) for e in coefficients] + [
            (flag(r, r('and', BOOL, active, r('ne', BOOL, scale, k(r, 1)))), I32), (flag(r, active), I32)]

    # 452-459, 484-494, 585-597, 625-643: X by SCALE (DSCAL), then the rows above the block by DAXPY with -X
    def updated(r, v, blocks):
        xv, iv, *rest = v
        tcols, (factor, *coefficients, used) = rest[:blocks], rest[blocks:]
        made = []
        for value, own in ((xv, coefficients[:blocks]), (iv, coefficients[blocks:])):
            y = r('mul', t, value, factor)
            for coefficient, entry in zip(own, tcols):
                y = r('add', t, y, r('mul', t, coefficient, entry))
            made.append(r('select', t, nonzero(r, used), y, value))
        return [(e, t) for e in made]

    def rescaled(r, v):
        xv, iv, factor, used = v
        inside = r('and', BOOL, nonzero(r, used), r('lt', BOOL, r.index(0), r('add', I32, r.index(1), i32(r, 1))))
        return [(r('select', t, inside, r('mul', t, e, factor), e), t) for e in (xv, iv)]
    for j in range(n - 1, 0, -1):
        width = n - j
        given = [(h[j - 1:j, j - 2], (None,))] if j >= 2 else []

        def blocks(r, v, j=j):
            two = r('ne', BOOL, v[0], k(r, 0)) if j >= 2 else r.const(0, BOOL)
            skip = r('ne', BOOL, v[-1], k(r, 0))
            single = r('and', BOOL, r('not', BOOL, two), r('not', BOOL, skip))
            double = r('and', BOOL, two, r('not', BOOL, skip))
            return [(r('select', I32, single, i32(r, width), i32(r, 0)), I32),
                    (r('select', I32, double, i32(r, width), i32(r, 0)), I32),
                    (r('select', I32, single, i32(r, j - 1), i32(r, 0)), I32),
                    (r('select', I32, double, i32(r, max(j - 2, 0)), i32(r, 0)), I32)]
        ops += _record((1,), [(x, (None,)) for x in (e1, e2, u1, u2)], given + [(h[j:j + 1, j - 1], (None,))], blocks)
        ops += _record((width,), [(act[j:], (0,)), (moved[j:], (0,))], [], lambda r, v: [(i32(r, 0), I32)] * 2)
        per = [(x[j:], (0,)) for x in (wrk, wik, smin, realc, bottom)]
        outs = [(xr[j - 1, j:], (0,)), (xi[j - 1, j:], (0,)), (scl[j:], (0,)), (car[j:], (0,)), (cai[j:], (0,)),
                (moved[j:], (0,)), (act[j:], (0,))]
        _emit(ops, _record((width,), outs, [(xr[j - 1, j:], (0,)), (xi[j - 1, j:], (0,)), (h[j - 1:j, j - 1], (None,))]
                           + per + [(cnorm[j - 1:j], (None,))], lambda r, v: solved(r, v, 1)), loops=((e1, width),))
        if j >= 2:
            outs = [(xr[j - 2, j:], (0,)), (xr[j - 1, j:], (0,)), (xi[j - 2, j:], (0,)), (xi[j - 1, j:], (0,)),
                    (scl[j:], (0,))] + [(x[j:], (0,)) for x in (c11, c21, c12, c22, moved, act)]
            entries = [(h[j - 2 + p:j - 1 + p, j - 2 + q], (None,)) for p, q in ((0, 0), (0, 1), (1, 0), (1, 1))]
            _emit(ops, _record((width,), outs, [(xr[j - 2, j:], (0,)), (xr[j - 1, j:], (0,)), (xi[j - 2, j:], (0,)),
                                                (xi[j - 1, j:], (0,))] + entries + per + [
                (cnorm[j - 2:j - 1], (None,)), (cnorm[j - 1:j], (None,))], lambda r, v: solved(r, v, 2)),
                  loops=((e2, width),))
        if j >= 2:
            block = [xr[:j - 1, j:], xi[:j - 1, j:]]
            _emit(ops, _record((j - 1, width), [(x, (0, 1)) for x in block], [(x, (0, 1)) for x in block] + [
                (h[:j - 1, j - 1], (0,))] + [(x[j:], (1,)) for x in (scl, car, cai, act)],
                lambda r, v: updated(r, v, 1)), loops=((u1, j - 1), None))
        if j >= 3:
            block = [xr[:j - 2, j:], xi[:j - 2, j:]]
            _emit(ops, _record((j - 2, width), [(x, (0, 1)) for x in block], [(x, (0, 1)) for x in block] + [
                (h[:j - 2, j - 2], (0,)), (h[:j - 2, j - 1], (0,))] + [(x[j:], (1,)) for x in (scl, c11, c21, c12, c22,
                                                                                               act)],
                lambda r, v: updated(r, v, 2)), loops=((u2, j - 2), None))
        ops += _record_extreme((width,), need, (None,), [(moved[j:], (0,))], lambda r, v, j=j: [(r(
            'select', I32, nonzero(r, v[0]), i32(r, n - j), i32(r, 0)), I32)], [REDUCTION])
        block = [xr[j:, j:], xi[j:, j:]]
        _emit(ops, _record((n - j, width), [(x, (0, 1)) for x in block], [(x, (0, 1)) for x in block] + [
            (scl[j:], (1,)), (moved[j:], (1,))], rescaled), loops=((need, n - j), None))

    # 510-518 and 668-690: VR = Z X with each pair's columns KI-1 (real part) and KI (imaginary part), then each
    # column by 1 / EMAX, the largest |VR(K,KI)| (a pair's |VR(K,KI-1)| + |VR(K,KI)|)
    def assembled(r, v):
        xv, iv, first, second, slot = v
        following = at(r, slot, r.index(0), r('add', I32, r.index(1), i32(r, 1)))
        return [(r('select', t, nonzero(r, first), following, r('select', t, nonzero(r, second), iv, xv)), t)]
    ops += _record((n, n), [(packed, (0, 1))], [(xr, (0, 1)), (xi, (0, 1)), (top, (1,)), (bottom, (1,)), (xr, None)],
                   assembled)
    ops += _gemm(vr, z, (0, 2), packed, (2, 1), (n, n, n))

    def largest(r, v):
        value, first, second, slot = v
        partner = r('select', I32, nonzero(r, first), r('add', I32, r.index(0), i32(r, 1)), r(
            'sub', I32, r.index(0), i32(r, 1)))
        paired = r('or', BOOL, nonzero(r, first), nonzero(r, second))
        return [(r('select', t, paired, r('add', t, absolute(r, value), absolute(r, at(r, slot, r.index(1), partner))),
                   absolute(r, value)), t)]
    ops += _record_extreme((n, n), emax, (0,), [(vr, (1, 0)), (top, (0,)), (bottom, (0,)), (vr, None)], largest,
                           [PARALLEL, REDUCTION])
    ops += _record((n, n), [(vr, (0, 1))], [(vr, (0, 1)), (emax, (1,))], lambda r, v: [(r(
        'mul', t, v[0], r('div', t, k(r, 1), v[1])), t)])


# Reference-LAPACK SRC/ztrevc.f 323-409 (SIDE = 'R', HOWMNY = 'B') with zlatrs.f 324-1025 (UPLO = 'U', TRANS = 'N',
# DIAG = 'N', NORMIN = 'Y'): h the upper triangular Schur form (read only, its diagonal the source's original T(K,K)),
# z the Schur vectors, vr the eigenvectors Z X, each column scaled to largest CABS1 1. Column KI of x (n x n) is
# KI's solve (T(1:KI-1,1:KI-1) - T(KI,KI)) X = SCALE (-T(1:KI-1,KI)), its diagonal T(K,K) - T(KI,KI) raised to SMIN
# where CABS1 is below it; the per-KI state (SMIN, TSCAL, GROW, XBND, SCALE, XMAX) is a vector by KI, and all the
# solves run together in the source's row order J = KI-1..1 (the static rounds J = n-1..1, column KI taking part
# where J < KI)
def _ztrevc(h, z, vr, ops):
    n, t, real, dtype = h.shape[-1], _class(h.real), h.real.dtype, h.dtype
    if not n:
        return
    limits = torch.finfo(real)
    unfl, ulp, ovfl = limits.tiny, limits.eps, limits.max
    smlnum_t, smlnum = unfl * (n / ulp), unfl / ulp
    bignum = 1 / smlnum
    tk = _temporary((n,), dtype)
    smin, cnorm, colmax, tmax, tmax2, tscal, xmax0, grow, xbnd, scale, xmax, s0, rec1, rec2, mx = \
        (_temporary((n,), real) for _ in range(15))
    case2, fallback, broken, plain, zero, apply, moved = (_temporary((n,), torch.int32) for _ in range(7))
    any2, need = _temporary((1,), torch.int32), _temporary((1,), torch.int32)
    rec2d, cn = _temporary((n, n), real), _temporary((n, n), real)
    x, za = _temporary((n, n), dtype), _temporary((n,), dtype)

    def c(r, value):
        return r.const(value, t)

    def k(r, value):
        return r.const(value, I32)

    def flag(r, test):
        return _convert(r, test, BOOL, I32)

    def larger(r, a, b):
        return r('select', t, r('ge', BOOL, a, b), a, b)

    def smaller(r, a, b):
        return r('select', t, r('le', BOOL, a, b), a, b)

    def cabs2(r, value):
        return r('add', t, r('abs', t, r('div', t, _part(r, t, value, 0), c(r, 2))),
                 r('abs', t, r('div', t, _part(r, t, value, 1), c(r, 2))))

    # ztrevc 366-370: T(J,J) - T(KI,KI), set to SMIN where its CABS1 is below SMIN
    def shifted(r, diagonal, own, bound):
        d = _plus(r, t, diagonal, own, 'sub')
        return _chosen(r, t, r('lt', BOOL, _cabs1(r, t, d), bound), (bound, c(r, 0)), d)

    def nonzero(r, value):
        return r('ne', BOOL, value, k(r, 0))

    def below(r):
        return r('lt', BOOL, r.index(1), r.index(0))

    # ztrevc 330-340 and 353: T(KI,KI), SMIN = max(ULP CABS1(T(KI,KI)), SMLNUM), CNORM(J) = DZASUM(J-1, T(1,J))
    ops += _record((n,), [(tk, (0,))], [(h, (0, 0))], lambda r, v: [(v[0], t)])
    ops += _record((n,), [(smin, (0,))], [(tk, (0,))], lambda r, v: [(larger(r, r('mul', t, c(r, ulp), _cabs1(
        r, t, v[0])), c(r, smlnum_t)), t)])
    ops += _record((n, n), [(cnorm, (0,))], [(h, (1, 0))], lambda r, v: [(r('select', t, below(r), _cabs1(
        r, t, v[0]), c(r, 0)), t)], [PARALLEL, REDUCTION], [SUM])

    # zlatrs 358-431 for each KI (its N = KI-1): TMAX the largest CNORM(1..KI-1); TSCAL 1 where TMAX <= BIGNUM/2,
    # HALF / (SMLNUM TMAX) where TMAX <= overflow (case 1, CNORM by TSCAL), else (case 2) from TMAX2 the largest
    # max(|Re|, |Im|) of T(1:KI-2, 2:KI-1)'s strict upper part: 1 / (SMLNUM TMAX2), or (TMAX2 above overflow) the
    # ZTRSV return
    ops += _record_extreme((n, n), colmax, (0,), [(h, (1, 0))], lambda r, v: [(r('select', t, below(r), larger(
        r, r('abs', t, _part(r, t, v[0], 0)), r('abs', t, _part(r, t, v[0], 1))), c(r, 0)), t)],
        [PARALLEL, REDUCTION])
    for made, given in ((tmax, cnorm), (tmax2, colmax)):
        ops += _record_extreme((n, n), made, (0,), [(given, (1,))], lambda r, v: [(r('select', t, below(r), v[0], c(
            r, 0)), t)], [PARALLEL, REDUCTION])

    def cases(r, v):
        top, top2 = v
        first = r('le', BOOL, top, c(r, bignum * 0.5))
        second = r('and', BOOL, r('not', BOOL, first), r('le', BOOL, top, c(r, ovfl)))
        third = r('and', BOOL, r('not', BOOL, first), r('not', BOOL, second))
        valid = r('le', BOOL, top2, c(r, ovfl))
        made = r('select', t, first, c(r, 1), r('select', t, second, r('div', t, c(r, 0.5), r(
            'mul', t, c(r, smlnum), top)), r('select', t, valid, r('div', t, c(r, 1), r('mul', t, c(
                r, smlnum), top2)), c(r, 1))))
        return [(made, t), (flag(r, r('and', BOOL, third, valid)), I32),
                (flag(r, r('and', BOOL, third, r('not', BOOL, valid))), I32)]
    ops += _record((n,), [(tscal, (0,)), (case2, (0,)), (fallback, (0,))], [(tmax, (0,)), (tmax2, (0,))], cases)

    # zlatrs 400-423, case 2: a column norm above overflow recomputed as the sum of (2 TSCAL) CABS2(A(I,J)), each
    # summand scaled before the summation; its records run only where some KI is case 2 (`any2`)
    ops += _record_extreme((n,), any2, (None,), [(case2, (0,))], lambda r, v: [(r('select', I32, nonzero(
        r, v[0]), k(r, n), k(r, 0)), I32)], [REDUCTION])
    _emit(ops, _record((n, n, n), [(rec2d, (0, 1))], [(h, (2, 0)), (tscal, (1,))], lambda r, v: [(r(
        'select', t, r('lt', BOOL, r.index(2), r.index(0)), r('mul', t, r('mul', t, c(r, 2), v[1]), cabs2(r, v[0])),
        c(r, 0)), t)], [PARALLEL, PARALLEL, REDUCTION], [SUM]), loops=(None, (any2, n), None))

    # CNORM(J) as KI's solve reads it: unscaled, by TSCAL (case 1 and case 2's representable norms), or recomputed
    def normed(r, v):
        norm, factor, second, recomputed = v
        scaled = r('mul', t, norm, factor)
        within = r('select', t, r('le', BOOL, norm, c(r, ovfl)), scaled, recomputed)
        return [(r('select', t, nonzero(r, second), within, r('select', t, r('eq', BOOL, factor, c(r, 1)), norm,
                                                                         scaled)), t)]
    ops += _record((n, n), [(cn, (0, 1))], [(cnorm, (0,)), (tscal, (1,)), (case2, (1,)), (rec2d, (0, 1))], normed)

    # zlatrs 436-440 and 442-504: XMAX = XBND = the largest CABS2(X(J)) of X = -T(1:KI-1,KI); where TSCAL = 1,
    # GROW = HALF / max(XBND, SMLNUM) and XBND = GROW, then each J = KI-1..1 (round J): GROW <= SMLNUM ends the loop
    # (GROW kept); XBND = min(XBND, min(1, TJJ) GROW) (0 where TJJ < SMLNUM); GROW = GROW TJJ / (TJJ + CNORM(J))
    # (0 where TJJ + CNORM(J) < SMLNUM); a completed loop GROW = XBND; TSCAL /= 1 GROW = 0
    ops += _record_extreme((n, n), xmax0, (0,), [(h, (1, 0))], lambda r, v: [(r('select', t, below(r), cabs2(
        r, v[0]), c(r, 0)), t)], [PARALLEL, REDUCTION])
    ops += _record((n,), [(grow, (0,)), (xbnd, (0,)), (broken, (0,))], [(xmax0, (0,))], lambda r, v: [
        (r('div', t, c(r, 0.5), larger(r, v[0], c(r, smlnum))), t)] * 2 + [(k(r, 0), I32)])
    for j in range(n - 1, 0, -1):
        def grown(r, v):
            g, xb, stopped, factor, own, bound, norm, diagonal = v
            live = r('and', BOOL, r('eq', BOOL, factor, c(r, 1)), r('not', BOOL, nonzero(r, stopped)))
            stop = r('and', BOOL, live, r('le', BOOL, g, c(r, smlnum)))
            step = r('and', BOOL, live, r('not', BOOL, stop))
            tjj = _cabs1(r, t, shifted(r, diagonal, own, bound))
            bounded = r('select', t, r('ge', BOOL, tjj, c(r, smlnum)), smaller(r, xb, r('mul', t, smaller(
                r, c(r, 1), tjj), g)), c(r, 0))
            total = r('add', t, tjj, norm)
            growth = r('select', t, r('ge', BOOL, total, c(r, smlnum)), r('mul', t, g, r('div', t, tjj, total)),
                       c(r, 0))
            return [(r('select', t, step, growth, g), t), (r('select', t, step, bounded, xb), t),
                    (flag(r, r('or', BOOL, nonzero(r, stopped), stop)), I32)]
        ops += _record((n - j,), [(grow[j:], (0,)), (xbnd[j:], (0,)), (broken[j:], (0,))],
                       [(x_[j:], (0,)) for x_ in (grow, xbnd, broken, tscal, tk, smin)]
                       + [(cnorm[j - 1:j], (None,)), (h[j - 1:j, j - 1], (None,))], grown)

    # zlatrs 606-627: ZTRSV where GROW TSCAL > SMLNUM (or case 2's return), SCALE = 1; else the careful solve: SCALE
    # = (BIGNUM/2) / XMAX, X by it and XMAX = BIGNUM where XMAX > BIGNUM/2, else XMAX = 2 XMAX; then X = -T(1:KI-1,
    # KI) by that SCALE (ztrevc 359-361), zero below
    def decided(r, v):
        factor, ztrsv, stopped, g, xb, top = v
        final = r('select', t, r('ne', BOOL, factor, c(r, 1)), c(r, 0), r('select', t, nonzero(r, stopped), g, xb))
        safe = r('or', BOOL, nonzero(r, ztrsv), r('gt', BOOL, r('mul', t, final, factor), c(r, smlnum)))
        big = r('and', BOOL, r('not', BOOL, safe), r('gt', BOOL, top, c(r, bignum * 0.5)))
        s = r('select', t, big, r('div', t, c(r, bignum * 0.5), top), c(r, 1))
        return [(flag(r, safe), I32), (s, t), (r('select', t, safe, top, r('select', t, big, c(r, bignum), r(
            'mul', t, top, c(r, 2)))), t), (s, t)]
    ops += _record((n,), [(plain, (0,)), (scale, (0,)), (xmax, (0,)), (s0, (0,))],
                   [(x_, (0,)) for x_ in (tscal, fallback, broken, grow, xbnd, xmax0)], decided)
    ops += _record((n, n), [(x, (0, 1))], [(h, (0, 1)), (s0, (1,))], lambda r, v: [(_chosen(
        r, t, r('lt', BOOL, r.index(0), r.index(1)), _real_scaled(r, t, _negated(r, t, v[0]), v[1]),
        (c(r, 0), c(r, 0))), t)])

    # each round J = n-1..1, columns KI = J+1..n: ZTRSV (X(J) = X(J) / A(J,J) where X(J) /= 0, then X(I) - X(J)
    # A(I,J)) or zlatrs 632-743: XJ = CABS1(X(J)), TJJS = A(J,J) TSCAL; TJJ > SMLNUM: REC = 1/XJ where TJJ < 1 and
    # XJ > TJJ BIGNUM; 0 < TJJ <= SMLNUM: REC = TJJ BIGNUM / XJ (then / CNORM(J) where CNORM(J) > 1) where XJ > TJJ
    # BIGNUM; X by REC, SCALE and XMAX by REC, X(J) = ZLADIV(X(J), TJJS); A(J,J) = 0: X = e_J, SCALE = XMAX = 0;
    # then by REC = 1/(2 XJ) where XJ > 1 and CNORM(J) > (BIGNUM - XMAX)/XJ, or HALF where XJ CNORM(J) > BIGNUM - XMAX
    # (SCALE by it); then X(1:J-1) + (-X(J) TSCAL) A(1:J-1,J) and XMAX the largest CABS1 of X(1:J-1)
    def solved(r, v):
        xj, diagonal, own, bound, factor, ztrsv, s, top, norm = v
        one = c(r, 1)
        d = shifted(r, diagonal, own, bound)
        present = _nonzero(r, t, xj)
        direct = _chosen(r, t, present, _complex_value(r, t, 'div', _complex_parts(r, t, xj), d, 1, None), xj)
        xa = _cabs1(r, t, xj)
        tjjs = _real_scaled(r, t, d, factor)
        tjj = _cabs1(r, t, tjjs)
        large = r('gt', BOOL, tjj, c(r, smlnum))
        tiny = r('and', BOOL, r('not', BOOL, large), r('gt', BOOL, tjj, c(r, 0)))
        singular = r('and', BOOL, r('not', BOOL, large), r('not', BOOL, tiny))
        over = r('gt', BOOL, xa, r('mul', t, tjj, c(r, bignum)))
        safe_xa = r('select', t, over, xa, one)
        bounded = r('div', t, r('mul', t, tjj, c(r, bignum)), safe_xa)
        bounded = r('select', t, r('gt', BOOL, norm, one), r('div', t, bounded, norm), bounded)
        first = r('select', t, r('and', BOOL, large, r('and', BOOL, r('lt', BOOL, tjj, one), over)),
                  r('div', t, one, safe_xa), r('select', t, r('and', BOOL, tiny, over), bounded, one))
        xs = _real_scaled(r, t, xj, first)
        s1, t1 = r('mul', t, s, first), r('mul', t, top, first)
        xn = _chosen(r, t, singular, (one, c(r, 0)), _complex_value(r, t, 'div', _complex_parts(r, t, xs), tjjs, 1,
                                                                   None))
        xjn = r('select', t, singular, one, _cabs1(r, t, xn))
        s1, t1 = (r('select', t, singular, c(r, 0), value) for value in (s1, t1))
        big = r('gt', BOOL, xjn, one)
        recip = r('div', t, one, r('select', t, big, xjn, one))
        room = r('sub', t, c(r, bignum), t1)
        second = r('select', t, r('and', BOOL, big, r('gt', BOOL, norm, r('mul', t, room, recip))),
                   r('mul', t, recip, c(r, 0.5)),
                   r('select', t, r('and', BOOL, r('not', BOOL, big), r('gt', BOOL, r('mul', t, xjn, norm), room)),
                     c(r, 0.5), one))
        xf = _real_scaled(r, t, xn, second)
        simple = nonzero(r, ztrsv)
        moved_ = _all(r, [r('not', BOOL, simple), r('or', BOOL, r('or', BOOL, r('ne', BOOL, first, one), r(
            'ne', BOOL, second, one)), singular)])
        return [(_chosen(r, t, simple, direct, xf), t), (r('select', t, simple, s, r('mul', t, s1, second)), t),
                (r('select', t, simple, top, t1), t), (r('select', t, simple, one, first), t),
                (r('select', t, simple, one, second), t),
                (flag(r, r('and', BOOL, r('not', BOOL, simple), singular)), I32),
                (_chosen(r, t, simple, _negated(r, t, direct), _real_scaled(r, t, _negated(r, t, xf), factor)), t),
                (flag(r, r('or', BOOL, r('not', BOOL, simple), present)), I32), (flag(r, moved_), I32)]

    def updated(r, v):
        xi, a, f1, f2, cleared, coefficient, used = v
        y = _chosen(r, t, nonzero(r, cleared), (c(r, 0), c(r, 0)), _real_scaled(r, t, _real_scaled(r, t, xi, f1), f2))
        return [(_chosen(r, t, nonzero(r, used), _plus(r, t, y, _times(r, t, coefficient, a)), y), t)]

    def rescaled(r, v):
        xi, f1, f2, cleared = v
        y = _chosen(r, t, nonzero(r, cleared), (c(r, 0), c(r, 0)), _real_scaled(r, t, _real_scaled(r, t, xi, f1), f2))
        return [(_chosen(r, t, r('lt', BOOL, r.index(0), r.index(1)), y, xi), t)]
    for j in range(n - 1, 0, -1):
        width = n - j
        ops += _record((width,), [(x[j - 1, j:], (0,))] + [(x_[j:], (0,)) for x_ in (
            scale, xmax, rec1, rec2, zero, za, apply, moved)],
                       [(x[j - 1, j:], (0,)), (h[j - 1:j, j - 1], (None,))] + [(x_[j:], (0,)) for x_ in (
                           tk, smin, tscal, plain, scale, xmax)] + [(cn[j - 1, j:], (0,))], solved)
        factors = [(x_[j:], (1,)) for x_ in (rec1, rec2, zero)]
        if j > 1:
            block = x[:j - 1, j:]
            ops += _record((j - 1, width), [(block, (0, 1))], [(block, (0, 1)), (h[:j - 1, j - 1], (0,))] + factors
                           + [(za[j:], (1,)), (apply[j:], (1,))], updated)
        if j < n - 1:
            ops += _record_extreme((width,), need, (None,), [(moved[j:], (0,))], lambda r, v, j=j: [(r(
                'select', I32, nonzero(r, v[0]), k(r, n - 1 - j), k(r, 0)), I32)], [REDUCTION])
            block = x[j:n - 1, j:]
            _emit(ops, _record((n - 1 - j, width), [(block, (0, 1))], [(block, (0, 1))] + factors, rescaled),
                  loops=((need, n - 1 - j), None))
        if j > 1:
            ops += _record_extreme((width, j - 1), xmax[j:], (0,), [(x[:j - 1, j:], (1, 0))], lambda r, v: [(_cabs1(
                r, t, v[0]), t)], [PARALLEL, REDUCTION])

    # zlatrs 1018 and ztrevc 376 and 355: X(KI,KI) = SCALE (the careful solve's SCALE / TSCAL, ZTRSV's 1; KI = 1: 1);
    # ztrevc 393-399: VR(:,KI) = Z(:,1:KI-1) X(1:KI-1) + SCALE Z(:,KI), then by 1 / the largest CABS1 of the column
    ops += _record((n,), [(x, (0, 0))], [(scale, (0,)), (tscal, (0,)), (plain, (0,))], lambda r, v: [(r(
        'select', t, r('or', BOOL, r('eq', BOOL, r.index(0), k(r, 0)), nonzero(r, v[2])), c(r, 1), r(
            'div', t, v[0], v[1])), t)])
    ops += _gemm(vr, z, (0, 2), x, (2, 1), (n, n, n))
    ops += _record_extreme((n, n), mx, (0,), [(vr, (1, 0))], lambda r, v: [(_cabs1(r, t, v[0]), t)],
                           [PARALLEL, REDUCTION])
    ops += _record((n, n), [(vr, (0, 1))], [(vr, (0, 1)), (mx, (1,))], lambda r, v: [(_real_scaled(
        r, t, v[0], r('div', t, c(r, 1), v[1])), t)])


# Reference-LAPACK SRC/zgeev.f and dgeev.f (JOBVL = 'N', JOBVR = 'V' or 'N') as a fake-only operator (its shape rule
# alone), lowered by _geev: the eigenvalues (*batch, n) contiguous and the right eigenvectors (*batch, n, n)
# column-major (PyTorch v2.14.0 meta linalg_eig's non-CUDA layout), or (0,) where `vectors` is false, both of a's
# complex dtype (a real input's complex counterpart: torch 2.14.0 torch/_meta_registrations.py meta_linalg_eig's
# utils.corresponding_complex_dtype, each floating kind's, float16 and bfloat16 among them, no admission subset of it)
torch.library.define('mesh::geev', '(Tensor a, bool vectors) -> (Tensor, Tensor)')


@torch.library.register_fake('mesh::geev')
def _geev_shapes(a, vectors):
    batch, n = a.shape[:-2], a.shape[-1]
    dtype = a.dtype if a.is_complex() else corresponding_complex_dtype(a.dtype)
    return (a.new_empty((*batch, n), dtype=dtype),
            a.new_empty((*batch, n, n), dtype=dtype).mT if vectors else a.new_empty((0,), dtype=dtype))


GEEV_EIGEN = torch.ops.mesh.geev.default


# torch 2.14.0 linalg_eig (A) -> (eigenvalues, eigenvectors) and _linalg_eigvals / linalg_eigvals (A) -> eigenvalues
# on a non-CPU engine (compiler/mesh_compiler/decompose.py outline): GEEV_EIGEN's results
def _eigen_geev(self, vectors):
    values, made = GEEV_EIGEN(self, vectors)
    return (values, made) if vectors else values


# zgeev.f 319-517 for each batch member: ANRM = ZLANGE('M') (the largest modulus), INFO = -4 where it is NaN or above
# overflow (and nothing else runs); A scaled into [SMLNUM, BIGNUM] (SMLNUM = sqrt(SAFMIN) / EPS) by ZLASCL; ZGEBAL;
# ZGEHRD (ZGEHD2) and, with vectors, ZUNGHR into Z; ZHSEQR (W(1:ILO-1) and W(IHI+1:N) from H's diagonal, then
# ZLAHQR); where INFO = 0 and with vectors ZTREVC, ZGEBAK('B', 'R') and each column's Euclidean normalization with
# its first largest squared modulus made real; then W's undo scaling (W(INFO+1:N), and W(1:ILO-1) where INFO > 0).
# ZLASCL (zlascl.f) takes MUL = CTO / CFROM in its first pass for these operands (CFROM and CTO finite and nonzero
# within the SMLNUM/BIGNUM range). The INFO of each member reaches the existing linalg.eig _Check. A real input
# (dgeev.f) takes DLAHQR's WR/WI, DTREVC's packed vectors and DGEBAK, its results linalg_eig's complex ones (PyTorch
# v2.14.0 BatchLinearAlgebra.cpp linalg_eig_make_complex_eigenvectors: where WI(j) > 0, VR(:,j) + i VR(:,j+1) and
# its conjugate at j+1); dgeev's 470-503 normalization (1 / DLAPY2 of the pair's DNRM2s, then DLARTG/DROT making the
# largest VR(K,j)^2 + VR(K,j+1)^2 entry real) is that complex column's normalization, the phase only for a pair
def _geev(outs, a, vectors):
    values, made = outs
    batch, n = a.shape[:-2], a.shape[-1]
    if not n or not a.numel():
        return []
    # Rejection filters are not part of even temporary integration validation.
    # Every non-CPU eig reaches this lowering; a kind GEEV's S/D/C/Z transcription lacks fails here, at its lowering.
    if a.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise NotImplementedError('linalg.eig of %s on a non-CPU engine: its GEEV lowering is unfinished '
                                  '(design/torch-source-lowering-change-request.md item 6)' % a.dtype)
    dtype, real, t = a.dtype, a.real.dtype, _class(a.real)
    whole = a.is_complex()
    limits = torch.finfo(real)
    smlnum = math.sqrt(limits.tiny) / limits.eps
    bignum, ovfl = 1 / smlnum, limits.max
    info = _temporary((*batch, 1), torch.int32)
    ops = []

    def c(r, value):
        return r.const(value, t)

    def k(r, value):
        return r.const(value, I32)
    for member in itertools.product(*map(range, batch)):
        source, own = a[member], info[member]
        h = _temporary((n, n), dtype)
        eigen = [_temporary((n,), dtype)] if whole else [_temporary((n,), real), _temporary((n,), real)]
        z, vr = (_temporary((n, n), dtype), _temporary((n, n), dtype)) if vectors else (None, None)
        assembled = [_temporary((n, n), values.dtype), _temporary((n,), torch.int32)] if vectors and not whole else []
        scale, anrm, mul, undo = _temporary((n,), real), *(_temporary((1,), real) for _ in range(3))
        perm, ilo, ihi, scalea, proceed, wanted = _temporary((n,), torch.int32), \
            *(_temporary((1,), torch.int32) for _ in range(5))
        carried = (h, *eigen, *((z, vr) if vectors else ()), *assembled, scale, anrm, mul, undo, perm, ilo, ihi, scalea,
                   proceed, wanted, info)

        # 332-350: ANRM, INFO = -4, SCALEA and CSCALE, MUL = CSCALE / ANRM and its undo ANRM / CSCALE
        ops += _record_extreme((n, n), anrm, (None,), [(source, (0, 1))], lambda r, v: [(_modulus(r, t, v[0]), t)],
                               [REDUCTION, REDUCTION])

        def scaled(r, v):
            norm = v[0]
            bad = r('or', BOOL, r('isnan', BOOL, norm), r('gt', BOOL, norm, c(r, ovfl)))
            low = r('and', BOOL, r('gt', BOOL, norm, c(r, 0)), r('lt', BOOL, norm, c(r, smlnum)))
            high = r('gt', BOOL, norm, c(r, bignum))
            target = r('select', t, low, c(r, smlnum), c(r, bignum))
            safe = r('select', t, r('or', BOOL, low, high), norm, c(r, 1))
            return [(r('select', I32, bad, k(r, -4), k(r, 0)), I32),
                    (_convert(r, r('and', BOOL, r('not', BOOL, bad), r('or', BOOL, low, high)), BOOL, I32), I32),
                    (_convert(r, r('not', BOOL, bad), BOOL, I32), I32),
                    (r('div', t, target, safe), t), (r('div', t, safe, target), t)]
        ops += _scalars([own, scalea, proceed, mul, undo], [anrm], scaled)

        body = []
        body += _record((n, n), [(h, (0, 1))], [(source, (0, 1)), (scalea, (None,)), (mul, (None,))], lambda r, v: [(
            _chosen(r, t, r('ne', BOOL, v[1], k(r, 0)), _real_scaled(r, t, v[0], v[2]), v[0]), t)])
        _gebal(h, scale, perm, ilo, ihi, body, held=carried)
        _gehd2(h, z, ilo, ihi, body)

        # zhseqr.f / dhseqr.f 415-419: the eigenvalues ZGEBAL isolated, H's diagonal outside ILO..IHI (WI zero)
        def isolated(r, v):
            *xs, diagonal, low, high = v
            p = r('add', I32, r.index(0), k(r, 1))
            outside = r('or', BOOL, r('lt', BOOL, p, low), r('gt', BOOL, p, high))
            return [(_chosen(r, t, outside, x, old), t) for x, old in zip((diagonal, c(r, 0)), xs)]
        body += _record((n,), [(x, (0,)) for x in eigen], [(x, (0,)) for x in eigen] + [
            (h, (0, 0)), (ilo, (None,)), (ihi, (None,))], isolated)
        if whole:
            _zlahqr(h, z, eigen[0], ilo, ihi, own, body, held=carried)
        else:
            _dlahqr(h, z, eigen[0], eigen[1], ilo, ihi, own, body, held=carried)

        # 438-504: where INFO = 0, ZTREVC (DTREVC), ZGEBAK (DGEBAK) and the normalization
        if vectors:
            body += _scalars([wanted], [own], lambda r, v: [(_convert(r, r('eq', BOOL, v[0], k(r, 0)), BOOL, I32),
                                                             I32)])
            inner = []
            (_ztrevc if whole else _dtrevc)(h, z, vr, inner)
            _gebak(vr, scale, perm, ilo, ihi, inner)
            if whole:
                _geev_normalized(vr, inner)
                inner += _copy(made[member], vr)
            else:
                vc, pairs = assembled

                def complexed(r, v):
                    value, wi, slot = v
                    first, second = r('gt', BOOL, wi, c(r, 0)), r('lt', BOOL, wi, c(r, 0))
                    after = r('extract', t, r.index(0), _bounded(r, r('add', I32, r.index(1), k(r, 1)), n)[1],
                              integer=slot)
                    before = r('extract', t, r.index(0), _bounded(r, r('sub', I32, r.index(1), k(r, 1)), n)[1],
                               integer=slot)
                    return [((r('select', t, second, before, value), r('select', t, first, after, r(
                        'select', t, second, r('neg', t, value), c(r, 0)))), t)]
                inner += _record((n, n), [(vc, (0, 1))], [(vr, (0, 1)), (eigen[1], (1,)), (vr, None)], complexed)
                inner += _record((n,), [(pairs, (0,))], [(eigen[1], (0,))], lambda r, v: [(_convert(r, r(
                    'ne', BOOL, v[0], c(r, 0)), BOOL, I32), I32)])
                _geev_normalized(vc, inner, pairs)
                inner += _copy(made[member], vc)
            inner += _fill(wanted, 0)
            body.append(_Repeat(inner, wanted, carried))

        # 508-517: W(INFO+1:N) by ANRM / CSCALE, and W(1:ILO-1) where INFO > 0, where SCALEA
        def unscaled(r, v):
            x, flagged, low, failure, factor = v
            p = r('add', I32, r.index(0), k(r, 1))
            hit = _all(r, [r('ne', BOOL, flagged, k(r, 0)), r('ge', BOOL, failure, k(r, 0)), r(
                'or', BOOL, r('ge', BOOL, p, r('add', I32, failure, k(r, 1))), r(
                    'and', BOOL, r('gt', BOOL, failure, k(r, 0)), r('le', BOOL, p, r('sub', I32, low, k(r, 1)))))])
            return [(_chosen(r, t, hit, _real_scaled(r, t, x, factor), x), t)]
        for x in eigen:
            body += _record((n,), [(x, (0,))], [(x, (0,)), (scalea, (None,)), (ilo, (None,)), (own, (None,)),
                                                 (undo, (None,))], unscaled)
        body += _copy(values[member], eigen[0]) if whole else _complex('complex', values[member], *eigen)
        body += _fill(proceed, 0)
        ops.append(_Repeat(body, proceed, carried))
    return ops + [_Check(info, 3, 'linalg.eig')]


# zgebak.f / dgebak.f 205-261 (JOB = 'B', SIDE = 'R', M = N) on v (complex or real) in place: rows ILO..IHI by
# SCALE(I) where ILO /= IHI, then for II = 1..N outside ILO..IHI (I = ILO - II below ILO), rows I and K = SCALE(I)
# (`perm`) interchanged in that order
def _gebak(v, scale, perm, ilo, ihi, ops):
    n, t = v.shape[-1], _class(v.real)
    span, base = _temporary((1,), torch.int32), _temporary((1,), torch.int32)
    ops += _scalars([span, base], [ilo, ihi], lambda r, x: [(r('select', I32, r('ne', BOOL, x[0], x[1]), r(
        'add', I32, r('sub', I32, x[1], x[0]), r.const(1, I32)), r.const(0, I32)), I32),
        (r('sub', I32, x[0], r.const(1, I32)), I32)])
    rows, factors = v[...], scale[...]
    _emit(ops, _record((n, n), [(rows, (0, 1))], [(rows, (0, 1)), (factors, (0,))], lambda r, x: [(_real_scaled(
        r, t, x[0], x[1]), t)]), loops=((span, n), None), offsets=[(rows, {0: (base, n - 1)}),
                                                                   (factors, {0: (base, n - 1)})])
    first, second, extent = (_temporary((n,), torch.int32) for _ in range(3))

    def order(r, x):
        low, high, slot = x
        one = r.const(1, I32)
        ii = r('add', I32, r.index(0), one)
        inside = r('and', BOOL, r('ge', BOOL, ii, low), r('le', BOOL, ii, high))
        at = r('select', I32, r('lt', BOOL, ii, low), r('sub', I32, low, ii), ii)
        kk = _entry(r, I32, slot, perm, _bounded(r, r('sub', I32, at, one), n)[1])
        active = r('and', BOOL, r('not', BOOL, inside), r('ne', BOOL, kk, at))
        return [(_bounded(r, r('sub', I32, at, one), n)[1], I32), (_bounded(r, r('sub', I32, kk, one), n)[1], I32),
                (r('select', I32, active, r.const(n, I32), r.const(0, I32)), I32)]
    ops += _record((n,), [(first, (0,)), (second, (0,)), (extent, (0,))], [(ilo, (None,)), (ihi, (None,)),
                                                                            (perm, None)], order)
    for ii in range(n):
        _swapped((v[...], (None, 0), {0: (first[ii:ii + 1], n - 1)}),
                 (v[...], (None, 0), {0: (second[ii:ii + 1], n - 1)}), (extent[ii:ii + 1], n), ops)


# zgeev.f 492-503: each column by 1 / DZNRM2 (_linalg_lengths' scaled norm), then by TMP = conj(v_K) / sqrt(RW_K) at
# K = IDAMAX(RW), the first largest RW = Re^2 + Im^2, and v_K's imaginary part set to zero; `phased` (int32 by
# column) where given limits that phase to its nonzero columns (dgeev.f 470-490: a real eigenvalue's column scaled
# alone, a pair's DLARTG/DROT the same rotation of its complex column)
def _geev_normalized(v, ops, phased=None):
    n, t, real = v.shape[-1], _class(v.real), v.real.dtype
    norms, squares, largest = _temporary((n,), real), _temporary((n, n), real), _temporary((n,), real)
    first, phase = _temporary((n,), torch.int32), _temporary((n,), v.dtype)
    _linalg_lengths(v, norms, ops, normalize=False)
    ops += _record((n, n), [(v, (0, 1))], [(v, (0, 1)), (norms, (1,))], lambda r, x: [(_real_scaled(
        r, t, x[0], r('div', t, r.const(1, t), x[1])), t)])
    ops += _record((n, n), [(squares, (0, 1))], [(v, (0, 1))], lambda r, x: [(_square(r, t, x[0]), t)])
    ops += _record_extreme((n, n), largest, (0,), [(squares, (1, 0))], lambda r, x: [(x[0], t)], [PARALLEL, REDUCTION])
    ops += _record_extreme((n, n), first, (0,), [(squares, (1, 0)), (largest, (0,))], lambda r, x: [(r(
        'select', I32, r('eq', BOOL, x[0], x[1]), r.index(1), r.const(n, I32)), I32)], [PARALLEL, REDUCTION],
        larger=False)

    def rotation(r, x):
        at, vs, rs = x
        row = _bounded(r, at, n)[1]
        entry = _entry(r, t, vs, v, row, r.index(0))
        return [(_divided_by(r, t, _conjugate(r, t, entry), r('sqrt', t, _entry(r, t, rs, squares, row, r.index(0)))),
                 t)]
    ops += _record((n,), [(phase, (0,))], [(first, (0,)), (v, None), (squares, None)], rotation)

    def turned(r, x):
        made = _chosen(r, t, r('eq', BOOL, r.index(0), x[2]), (_part(r, t, _times(r, t, x[1], x[0]), 0), r.const(
            0, t)), _times(r, t, x[1], x[0]))
        return [(made if phased is None else _chosen(r, t, r('ne', BOOL, x[3], r.const(0, I32)), made, x[0]), t)]
    ops += _record((n, n), [(v, (0, 1))], [(v, (0, 1)), (phase, (1,)), (first, (1,))] + (
        [(phased, (1,))] if phased is not None else []), turned)


# a complex record value divided by a real one, each component (Fortran's complex / real)
def _divided_by(r, t, x, s):
    return tuple(r('div', t, p, s) for p in _complex_parts(r, t, x))


# an index that may leave [0, size): its in-range test and the index clamped into range, so the extraction is defined
def _bounded(r, at, size):
    zero, last = r.const(0, I32), r.const(size - 1, I32)
    inside = r('and', BOOL, r('ge', BOOL, at, zero), r('le', BOOL, at, last))
    return inside, r('select', I32, r('lt', BOOL, at, zero), zero, r('select', I32, r('gt', BOOL, at, last), last, at))


def _all(r, tests):
    made = tests[0]
    for test in tests[1:]:
        made = r('and', BOOL, made, test)
    return made


# design/homomorphism-sources-2026-11-08.md; L. Meertens, Algorithmics, CWI Monographs 1, 1986, "Inserted-in", (i);
# Gibbons 1996, (2) and Definition 3.1: each reduction's identity and its associative operator
def _identity(packet, dtype):
    if packet in (A.sum, A.mean):
        return 0
    if dtype.is_floating_point:
        return -math.inf if packet is A.amax else math.inf
    if dtype is torch.bool:
        return int(packet is A.amin)
    return torch.iinfo(dtype).min if packet is A.amax else torch.iinfo(dtype).max


# the loops of a reduction of x over `dims` (its free dimensions PARALLEL first, then the reduced), x's map and the
# result's: the reduced shape's, or (`keepdim`, PyTorch 2.14 documentation, torch.sum `keepdim`) the original output's,
# each reduced dimension of extent 1 at the constant row 0 and each free one at its loop
def _reducing(x, dims, keepdim=False):
    free = [d for d in range(x.dim()) if d not in dims]
    domain = free + list(dims)
    kept = [None if d in dims else domain.index(d) for d in range(x.dim())] if keepdim else range(len(free))
    return ([x.shape[d] for d in domain], [PARALLEL] * len(free) + [REDUCTION] * len(dims),
            _rows(len(domain), [domain.index(d) for d in range(x.dim())]), _rows(len(domain), kept))


def _reduction(packet, x, dims, kept, t, start=IDENTITY, keepdim=False):
    r = _Region()
    r.done(_load(r, 0, x, t))
    loops, iterators, xmap, omap = _reducing(x, dims, keepdim)
    if packet in (A.sum, A.mean):
        return _Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [SUM], [t], [start], precision=REASSOCIATE)
    states = [(t, _identity(packet, x.dtype))]
    return _Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [REGION], [t], [start], states,
                    _joined(states, lambda c, a, b: [_extremum(c, a[0], b[0], t, packet is A.amax)]),
                    _identities(states), REASSOCIATE)


def _dims(given, x):
    dims = given.get('dim')
    dims = () if dims is None else _ints(dims, 1)
    return sorted({d % max(x.dim(), 1) for d in dims}) if dims else list(range(x.dim()))


def _reduced(target, given, outs):
    (out,), x, packet = outs, given['self'], target.overloadpacket
    _typed(target, [x], outs, REAL)
    dims, t = _dims(given, x), _compute(out.dtype, x.dtype)
    acc = _temporary(out.shape, STORED[t]) if packet is A.mean else out
    made = [_reduction(packet, x, dims, acc, t, keepdim=bool(given.get('keepdim')) and bool(dims))]
    if packet is not A.mean:
        return made
    total = math.prod(x.shape[d] for d in dims)
    return made + _epilogue(out, acc, t, (), lambda r, s: r('div', t, s, r.const(total, t)))


# docs/provider.md#fragment-coded-weights (rdma/mesh_code.py fragment; its codes, tiles and bands uint32, as
# mesh_code.load views them): each output element (i, j) at source row rows[i] and column columns[j], in its 32 x 32
# tile (tr, tc) at (rr, cc). The tile word's bits 12-15 its code width (0: the tile is zero), bits 0-11 with the
# band's low 28 bits its code-word base, its high half its float16 step; the element's code a width-bit field of the
# 16-bit lane half (rr odd: the high half) of the code words at lane `lane`, continued into the next word's (32 on)
# where it crosses 16 bits; its value (2 code + 1 - 2^width) step / 2 in float32 (float64 for a float64 result), times
# row_scale[rows[i]] and col_scale[columns[j]], in `dtype`. Each 32-bit word is read as its unsigned value (an int32
# view widened to int64 and masked), so every field and shift is that of the unsigned word
def _fragment_decoded(codes, tiles, bands, row_scale, col_scale, rows, columns, dtype):
    if not codes.numel():
        return torch.zeros((rows.numel(), columns.numel()), dtype=dtype, device=codes.device)
    t = torch.float64 if dtype is torch.float64 else torch.float32
    row, col = rows.reshape(-1).to(torch.int64)[:, None], columns.reshape(-1).to(torch.int64)[None, :]
    rr, cc, tr, tc = row % 32, col % 32, row // 32, col // 32
    word = _unsigned(tiles.view(torch.int32)[tr, tc])
    width = (word >> 12) & 15
    lane = ((rr // 8) & 1) | ((cc & 3) << 1) | (((rr // 8) & 2) << 2) | ((cc & 4) << 2)
    bit = ((rr & 6) * 2 + cc // 8) * width
    shift, half = bit % 16, (rr & 1) * 16
    at = ((_unsigned(bands.view(torch.int32).reshape(-1)[tr]) & 0x0FFFFFFF) + (word & 0xFFF) + bit // 16) * 32 + lane
    live = width != 0
    crossing = live & (shift + width > 16)
    flat = codes.view(torch.int32).reshape(-1)
    low = torch.where(live, _unsigned(flat[torch.where(live, at, 0)]), 0)
    high = torch.where(crossing, _unsigned(flat[torch.where(crossing, at + 32, 0)]), 0)
    joined = ((low >> half) & 0xFFFF) | (((high >> half) & 0xFFFF) << 16)
    power = torch.ones_like(width) << width
    symbol = (joined >> shift) & (power - 1)
    step = tiles.view(torch.float16)[..., 1::2][tr, tc].to(t)
    value = torch.where(live, (symbol * 2 + 1 - power).to(t) * (0.5 * step), 0)
    value = value * row_scale.reshape(-1)[row].to(t) * col_scale.reshape(-1)[col].to(t)
    return value.to(dtype)


def _unsigned(words):
    return words.to(torch.int64) & 0xFFFFFFFF


FRAGMENT = mesh_code.fragment


# J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, SC 2011, §3.1 and
# §4.2 (Philox4x32-10: its multipliers, Weyl key increments, ten rounds): each draw a counter-based function of its key
# and counter, the key the program's seed, the counter (element, execution, draw site, stream); the generator a STATE
# value {seed low, seed high, execution, 0} the program advances once an execution
PHILOX = (0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85)


def seeded(seed):
    words = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0, 0]
    return torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32)


# a constructed program's generator: its caller's STATE words, each draw site its source identity (compiler/
# mesh_compiler/export.py read `site`) where supplied, else numbered in construction order from 1, `sites` the count of
# the construction's draws, and their advance once an execution after the program's draws; `counter` the execution
# word's own view (word 2 of `state`), the advance's operand (design/compilation-composition-change-request.md items 6
# and 7: compiler/mesh_compiler/schedule.py analyse's read and write view of the body's state)
class Generator:
    def __init__(self, state):
        self.state, self.sites, self.counter = state, 0, state.narrow(0, 2, 1)

    def site(self, identity=None):
        self.sites += 1
        return self.state, self.sites if identity is None else identity

    def stepped(self):
        return [_stepped(self.counter)] if self.sites else []


# the generator's advance: its execution word (Generator.counter, one word at the constant row 0) incremented once an
# execution, after the program's draws
def _stepped(counter):
    r = _Region()
    r.done(r('add', I32, r.arg(0, counter), r.const(1, I32)))
    return _Generic((1,), [(counter, _rows(1, [None]))], [(counter, _rows(1, [None]))], r)


def _placed(ops, place, node):
    for g in ops:
        g.place, g.node = place, node
        if isinstance(g, _Repeat):
            _placed(g.records, place, node)


# design/torch-source-lowering-change-request.md item 1: the storage identity of a tensor of the compilation (its
# placed stand-ins, temporaries and views are meta tensors; a constant is a tensor holding its data)
def _key(value):
    return value.untyped_storage()._cdata if value.device.type == 'meta' else value.untyped_storage().data_ptr()


# torch 2.14.0 torch/_tensor_docs.py view_as_real: a storage root's element type, a complex one's real pairs
def _stored(value):
    return torch.view_as_real(value) if value.is_complex() else value


def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device,
            value.requires_grad) \
        if isinstance(value, torch.Tensor) else value


# design/compilation-composition-change-request.md, numerical lowerer boundary: "Compilation selects the custom
# numerical method explicitly ... A custom numerical or target implementation is supplied there with its explicit
# operands and its cited algorithm (source-lowering items 5/6)". compiler/mesh_compiler/decompose.py selects each
# method and traces its formulation into the placed body: ATen and torch HOPs (while_loop_op, cond_op, switch_op), the
# installed upstream compiler's to lower (torch 2.14.0 torch/_inductor/graph.py GraphLowering call_function,
# lowering.py lowerings and make_fallback), and, on the CPU, the typed LAPACK operators (LAPACK: each one Accelerate
# routine call, never a Torch kernel, its INFO consumed by the body's aten._linalg_check_errors).
#
# Not yet an upstream body: mesh::geev (GEEV_EIGEN, eig and eigvals off the CPU) is still this module's region
# records (_Region, _Generic, _Repeat, _Check) through `lowered`: build(target, given, outs, generator) over
# the extern's operands by schema name (_operands), each tensor operand its program value, `outs` its results' values.
# That record path is the transitional one, not the method body; the geev algorithm's torch body replaces it. The
# outlined eigh's step is a torch body (_qdwh_eigh_pop), its draws reading the body's Generator through GENERATOR
Method = collections.namedtuple('Method', 'build drawn source')

METHODS = {
    GEEV_EIGEN.overloadpacket: Method(lambda target, given, outs, generator: _geev(outs, **given), False,
                                      'item 6: Reference-LAPACK zgeev.f and dgeev.f on the placed processor'),
}


def method(target):
    return METHODS.get(getattr(target, 'overloadpacket', target))


def routine(target):
    return LAPACK.get(getattr(target, 'overloadpacket', target))


# a body's nodes `lookup` finds (method or routine), its HOP subgraphs' among them (torch 2.14.0
# torch/_higher_order_ops: each cond, switch and while_loop subgraph a GraphModule its get_attr reads): node -> value
def methods(graph_module, lookup=method):
    found = {}
    for node in graph_module.graph.nodes:
        held = _get_attr(graph_module, node.target) if node.op == 'get_attr' else None
        if isinstance(held, torch.fx.GraphModule):
            found.update(methods(held, lookup))
        if node.op == 'call_function' and lookup(node.target) is not None:
            found[node] = lookup(node.target)
    return found


# one record method extern's records: `args` and `kwargs` the node's own, each tensor operand its program value;
# `outs` its results' values
def lowered(target, args, kwargs, outs, generator=None):
    return method(target).build(target, _operands(target, args, kwargs), list(outs), generator)

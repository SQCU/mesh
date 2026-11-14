# design/compiler-uncovered-work-2026-11-14.md; design/torch-source-lowering-change-request.md item 1;
# design/compilation-composition-change-request.md, "Complete replacement workgroup": source operands -> structured
# operation and region records, the construction lower_metal and decompose consume and metal._Builder encodes. The
# definitions are carried, each with its citation, from the source's numerical construction, rdma/mesh_compile.py at
# 039ad8ed, lines 31-32, 35-40, 78-81, 341-352, 1028-1037, 1156-1167, 1269-1273, 2050-2057, 2059-2587, 2595-3050,
# 3071-6557, 6691-6693, 6708-6718, 6744-6792, 8067-8070
import collections
import itertools
import math
import operator
import pathlib
import sys

import torch
import torch.utils._pytree as pytree

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_code

# torch 2.14.0 torch/export/__init__.py, export; torch/_ops.py, OpOverload
A = torch.ops.aten


# GSPMD section 4.3
CONVOLUTION = {A.conv1d: 1, A.conv2d: 2, A.conv3d: 3}
TRANSPOSED = {A.conv_transpose1d: 1, A.conv_transpose2d: 2, A.conv_transpose3d: 3}
MAXIMUM = {A.max_pool1d: 1, A.max_pool2d: 2, A.max_pool3d: 3}
AVERAGE = {A.avg_pool1d: 1, A.avg_pool2d: 2, A.avg_pool3d: 3}
PAD = {A.pad, A.constant_pad_nd}


# GSPMD section 4.1, "Static operator configurations"
Local = collections.namedtuple('Local', 'op target args kwargs annotation site', defaults=(None, None))
AttentionOrigin = collections.namedtuple('AttentionOrigin', 'key page cuts shape offsets stripes',
                                         defaults=(0, 0, (), (), (), ()))


# torch 2.14.0 torch/_C/__init__.pyi, FunctionSchema `arguments`, Argument `name`
def _names(local):
    return tuple(argument.name for argument in local.target._schema.arguments)


def _ints(value, k):
    value = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return value * k if len(value) == 1 else value


def _given(local, names):
    return {**dict(zip(names, local.args)), **local.kwargs}


# design/unequal-shards-change-request.md item 8; W. Brandon et al., Striped Attention, arXiv:2311.09431, "we permute
# the sequence so that every device always operates on a discontiguous subset of tokens distributed approximately
# uniformly throughout the original sequence": the places in the sequence of the tokens the ranks hold, rank after
# rank, `counts[rank]` tokens a rank. Token j of a rank of c tokens lies at (2 j + 1) / (2 c) of the sequence, and the
# tokens take the sequence's places in that order, equal ones in rank order; with equal counts over p ranks, token j of
# rank r is at place p j + r
def striped(counts):
    return torch.tensor([j + sum(((2 * j + 1) * count + own - (other > rank)) // (2 * own)
                                 for other, count in enumerate(counts) if other != rank)
                         for rank, own in enumerate(counts) for j in range(own)], dtype=torch.int64)


# torch 2.14.0 torch/_export/verifier.py
def _call(local, args, kwargs):
    return local.target(*args, **kwargs)


# torch 2.14.0 torch/fx/node.py Node; torch/_refs/__init__.py _to_copy: an FX call's operator, operands and keywords,
# its copy's placement the program's (design/compilation-composition-change-request.md item 6), and its source draw
# site (compiler/mesh_compiler/export.py read `site`)
def _local_call(node):
    return Local(node.op, node.target, node.args, {k: v for k, v in node.kwargs.items() if k != 'device'}
                 if getattr(node.target, 'overloadpacket', None) is A._to_copy else node.kwargs,
                 site=node.meta.get('site'))


# design/torch-source-lowering-change-request.md item 1; design/plan-to-bridge-change-request.md R4;
# design/aot-compilation-change-request.md A8; J. Gibbons, The third homomorphism theorem, J. Functional Programming,
# 1996; L. Meertens, Algorithmics, CWI Monographs 1, 1986; G. Graefe, Query evaluation techniques for large databases,
# ACM Comput. Surv. 25(2), 1993; PyTorch 2.14 documentation, Broadcasting semantics
COPIES = {A.clone, A.lift_fresh_copy, A._to_copy, A._unsafe_view}


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
COMPLEXES = (torch.complex64, torch.complex128)
FLOATING = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
INTEGRAL = tuple(dtype for dtype in REAL if dtype not in FLOATING)
# torch 2.14.0 aten/src/ATen/native/LinearAlgebra.cpp addmm_impl_cpu_ (AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16,
# kHalf)): the contractions' result kinds, every number kind but bool
CONTRACTED = FLOATING + COMPLEXES + tuple(dtype for dtype in INTEGRAL if dtype is not torch.bool)
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
# metal_ir_loop_extent `extents`, the index of the symbol clipping it or None (its static size)
class _Generic:
    def __init__(self, sizes, inputs, outputs, body, iterators=None, combiner=None, accumulator=None, start=None,
                 states=None, combine=None, finalize=None, precision=0, symbols=None, extents=None):
        self.sizes, self.inputs, self.outputs, self.body = tuple(sizes), list(inputs), list(outputs), body
        self.symbols = list(symbols) if symbols is not None else []
        self.extents = tuple(extents) if extents is not None else (None,) * len(self.sizes)
        self.iterators = tuple(iterators) if iterators is not None else (PARALLEL,) * len(self.sizes)
        self.combiner = list(combiner) if combiner is not None else [ASSIGN] * len(self.outputs)
        self.accumulator = list(accumulator) if accumulator is not None else [_class(t) for t, _ in self.outputs]
        self.start = list(start) if start is not None else [IDENTITY] * len(self.outputs)
        self.states, self.combine, self.finalize, self.precision = states, combine, finalize, precision
        self.place, self.node = GPU, 0


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


def _onto(value, picks, sizes):
    return _rows(len(sizes), [None if pick is not None and n == 1 and sizes[pick] != 1 else pick
                              for n, pick in zip(value.shape, picks)])


def _temporary(shape, dtype=torch.float32):
    return torch.empty(tuple(shape), dtype=dtype, device='meta')


def _typed(target, operands, outs, results=FLOATING):
    found = [value.dtype for value in operands if isinstance(value, torch.Tensor) and value.dtype not in REAL] \
        + [value.dtype for value in outs if value.dtype not in results]
    if found and (found[0].is_complex or results is CONTRACTED and found[0] is torch.bool):
        raise NotImplementedError('%s of %s: PyTorch 2.14 defines no form of it for that type (its CPU kernel admits '
                                  'none)' % (target, found[0]))
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
    if integral and not isinstance(alpha, (bool, int)):
        raise NotImplementedError('elementwise %s into %s of a scalar no integer holds' % (op, out.dtype))
    t = _compute(out.dtype, *[x.dtype for x in tensors]) if integral else \
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


POINTWISE = {A.add: 'add', A.sub: 'sub', A.mul: 'mul', A.div: 'div', A.floor_divide: 'floordiv', A.neg: 'neg',
             A.relu: 'relu', A.tanh: 'tanh', A.sigmoid: 'sigmoid', A.exp: 'exp', A.gelu: 'gelu', A.silu: 'silu',
             A.eq: 'eq', A.ne: 'ne', A.lt: 'lt', A.le: 'le', A.gt: 'gt', A.ge: 'ge', A.pow: 'pow', A.rsqrt: 'rsqrt',
             A.sqrt: 'sqrt', A.log: 'log', A.abs: 'abs', A.clamp: 'clamp', A.clamp_min: 'clamp',
             A.clamp_max: 'clamp', A.where: 'where',
             **{getattr(A, name): name for name in (
                 'floor', 'ceil', 'round', 'trunc', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh',
                 'asinh', 'acosh', 'atanh', 'erf', 'atan2', 'reciprocal', 'exp2', 'log2', 'log10', 'log1p', 'expm1',
                 'isnan',
                 'isinf', 'sign', 'maximum', 'minimum', 'fmod', 'remainder', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
                 'bitwise_not', 'logical_and', 'logical_or', 'logical_xor', 'logical_not')},
             A.__and__: 'bitwise_and', A.__or__: 'bitwise_or', A.__xor__: 'bitwise_xor'}
# PyTorch 2.14 documentation, torch.nn.functional.leaky_relu, hardtanh and elu, and torch.nan_to_num (its infinities'
# defaults the dtype's finite extremes): their scalar parameters the region's literals, a composition of the pointwise
# basis
PARAMETERIZED = {A.leaky_relu: ('negative_slope',), A.hardtanh: ('min_val', 'max_val'),
                 A.elu: ('alpha', 'scale', 'input_scale'), A.nan_to_num: ('nan', 'posinf', 'neginf')}


def _parameterized(target, given, outs):
    packet, x, out = target.overloadpacket, given['self'], outs[0]
    if packet is A.nan_to_num and not out.dtype.is_floating_point:
        return _copy(out, x)
    _typed(target, [x], outs)
    a = [given.get(name) for name in PARAMETERIZED[packet]]
    r, t = _Region(), _compute(out.dtype, x.dtype, torch.float32)
    v = _load(r, 0, x, t)

    def c(value):
        return r.const(value, t)
    zero = c(0)
    if packet is A.leaky_relu:
        made = r('select', t, r('gt', BOOL, v, zero), v, r('mul', t, v, c(a[0])))
    elif packet is A.hardtanh:
        made = r('select', t, r('lt', BOOL, v, c(a[0])), c(a[0]), r('select', t, r('gt', BOOL, v, c(a[1])), c(a[1]), v))
    elif packet is A.nan_to_num:
        largest = torch.finfo(out.dtype).max
        high, low = c(largest if a[1] is None else a[1]), c(-largest if a[2] is None else a[2])
        made = r('select', t, r('isnan', BOOL, v), c(a[0] or 0), r('select', t, r('eq', BOOL, v, c(math.inf)), high,
                                                                     r('select', t, r('eq', BOOL, v, c(-math.inf)), low, v)))
    else:
        negative = r('mul', t, c(a[1] * a[0]), r('sub', t, r('exp', t, r('mul', t, v, c(a[2]))), c(1)))
        made = r('select', t, r('gt', BOOL, v, zero), r('mul', t, v, c(a[1])), negative)
    r.done(_store(r, made, t, out))
    return [_Generic(out.shape, [(x, _same(x.dim()))], [(out, _same(out.dim()))], r)]


def _arithmetic(target, given, outs):
    packet, op = target.overloadpacket, POINTWISE[target.overloadpacket]
    if any(isinstance(v, torch.Tensor) and v.is_complex() or isinstance(v, complex)
           for v in [*given.values(), *outs]):
        if packet is A.where:
            return [g for k in (0, 1) for g in _elementwise('where', _component(outs[0], k), given['condition'],
                                                           _component(given['self'], k), _component(given['other'], k))]
        return _complex(op, outs[0], given['self'], given['exponent'] if packet is A.pow else given.get('other'),
                        given.get('alpha', 1))
    if packet is A.div and given.get('rounding_mode') is not None:
        op = {'floor': 'floordiv', 'trunc': 'truncdiv'}[given['rounding_mode']]
    if packet is A.gelu and given['approximate'] != 'none':
        op = 'gelu_tanh'
    if packet is A.bitwise_not and given['self'].dtype is torch.bool:
        op = 'logical_not'
    operands = [given['condition'], given['self'], given['other']] if packet is A.where else \
        [given['self'], given.get('min'), given.get('max')] if op == 'clamp' else \
        [given['self'], given['exponent'], None] if packet is A.pow else [given['self'], given.get('other'), None]
    _typed(target, operands, outs, REAL)
    return _elementwise(op, outs[0], *operands, given.get('alpha', 1))


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


def _assign(out, value):
    if not isinstance(value, torch.Tensor):
        return _fill(out, value)
    return _copy(out, value.expand(out.shape))


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
    raise NotImplementedError('%s of complex operands: PyTorch defines no complex form of it' % op)

COMPLEX = {A.polar: 'polar', A.complex: 'complex', A.angle: 'angle', A.conj_physical: 'conj',
           A._conj_physical: 'conj'}


def _complexed(target, given, outs):
    packet = target.overloadpacket
    if packet in (A.polar, A.complex):
        first, second = (given['abs'], given['angle']) if packet is A.polar else (given['real'], given['imag'])
        return _complex(COMPLEX[packet], outs[0], first, second)
    if packet in (A.conj_physical, A._conj_physical) and not given['self'].is_complex():
        return _assign(outs[0], given['self'])
    return _complex(COMPLEX[packet], outs[0], given['self'])


# J. W. Cooley and J. W. Tukey, An algorithm for the machine calculation of complex Fourier series, Math. Comp. 19,
# 1965 (with the four-step index split of N = n1 n2: n = n2 a + c, k = k1 + n1 k2, X_k = W_N^(c k1) W_n2^(c k2)
# sum_a x_(n2 a + c) W_n1^(a k1)), n1 the smallest prime factor at each level (P. N. Swarztrauber, FFTPACK, Netlib, and
# M. Reinecke's pocketfft the same factorization); L. I. Bluestein, A linear filtering approach to the computation of
# the discrete Fourier transform, IEEE Trans. Audio Electroacoust. 18(4), 1970, for a prime length above DIRECT: the
# transform the chirp c_n = exp(s pi i n^2 / N) times the convolution of x c with conj(c), by power-of-two transforms.
# Complex values are their real (re, im) views; each level one structured operation over strided views
DIRECT = 16


def _twiddled(r, t, angle):
    return r('cos', t, angle), r('sin', t, angle)


def _complex_rows(rows, loops, c):
    return tuple(rows) + ((0,) * loops + (c,),)


# dst[k] = scale exp(s 2 pi i (k n) / N) summed over n of src[n] along their last dimension (src real or complex,
# dst complex), with the smallest prime factor's radix and a twiddle, the rest a transform of the factor's quotient
def _fft(src, dst, n, sign, scale, t):
    lead = src.dim() - 1
    if n == 1:
        return _copy(dst, src) if scale == 1 else _scaled(src, dst, scale, t)
    p = next(q for q in range(2, n + 1) if n % q == 0)
    if p == n and n > DIRECT:
        return _bluestein(src, dst, n, sign, scale, t)
    m = n // p
    loops = [*src.shape[:-1], m, p, 2, p]
    r, inputs = _Region(), []
    view = src.as_strided((*src.shape[:-1], p, m), (*src.stride()[:-1], src.stride(-1) * m, src.stride(-1)),
                          src.storage_offset())
    rows = _rows(len(loops), [*range(lead), lead + 3, lead])
    if view.is_complex():
        pairs = torch.view_as_real(view)
        for c in (0, 1):
            inputs.append((pairs, _complex_rows(rows, len(loops), c)))
        xr, xi = _load(r, 0, pairs, t), _load(r, 1, pairs, t)
    else:
        inputs.append((view, rows))
        xr, xi = _load(r, 0, view, t), r.const(0, t)
    c, k1, a = r.index(lead), r.index(lead + 1), r.index(lead + 3)
    inner = r('mul', t, _convert(r, r('rem', I32, r('mul', I32, a, k1), r.const(p, I32)), I32, t),
              r.const(sign * 2 * math.pi / p, t))
    outer = r('mul', t, _convert(r, r('mul', I32, c, k1), I32, t), r.const(sign * 2 * math.pi / n, t))
    cos, sin = _twiddled(r, t, r('add', t, inner, outer))
    re = r('sub', t, r('mul', t, xr, cos), r('mul', t, xi, sin))
    im = r('add', t, r('mul', t, xr, sin), r('mul', t, xi, cos))
    r.done(r('mul', t, r('select', t, r('eq', BOOL, r.index(lead + 2), r.const(0, I32)), re, im), r.const(scale, t)))
    y = _temporary([*src.shape[:-1], m, p], torch.complex128 if t == F64 else torch.complex64)
    ops = [_Generic(loops, inputs, [(torch.view_as_real(y), _same(len(loops), lead + 3))], r,
                    [PARALLEL] * (lead + 3) + [REDUCTION], [SUM], [t], precision=REASSOCIATE | CONTRACT)]
    rest = y.permute(*range(lead), lead + 1, lead)
    out = dst.as_strided((*dst.shape[:-1], p, m), (*dst.stride()[:-1], dst.stride(-1), dst.stride(-1) * p),
                         dst.storage_offset())
    return ops + _fft(rest, out, m, sign, 1.0, t)


def _scaled(src, dst, scale, t):
    return _complex('mul', dst, src, complex(scale)) if src.is_complex() else \
        _complex('copy', dst, src) if scale == 1 else _complex('mul', dst, src, scale)


def _bluestein(src, dst, n, sign, scale, t):
    size = 1 << (2 * n - 2).bit_length()
    kind = torch.complex128 if t == F64 else torch.complex64
    lead = src.dim() - 1
    lines = list(src.shape[:-1])
    a, b = _temporary([*lines, size], kind), _temporary([size], kind)
    fa, fb, product, conv = (_temporary([*lines, size], kind), _temporary([size], kind), _temporary([*lines, size], kind),
                             _temporary([*lines, size], kind))

    def chirp(r, j, negate):
        square = r('rem', I64, r('mul', I64, j, j), r.const(2 * n, I64))
        angle = r('mul', t, _convert(r, square, I64, t), r.const((-sign if negate else sign) * math.pi / n, t))
        return _twiddled(r, t, angle)
    r, inputs = _Region(), []
    j = r('cast', I64, r.index(lead))
    inside, at = _bounded(r, r.index(lead), n)
    if src.is_complex():
        pairs = torch.view_as_real(src)
        xr = _convert(r, r('extract', _class(pairs), *[r.index(q) for q in range(lead)], at, r.const(0, I32), integer=0),
                      _class(pairs), t)
        xi = _convert(r, r('extract', _class(pairs), *[r.index(q) for q in range(lead)], at, r.const(1, I32), integer=0),
                      _class(pairs), t)
        inputs.append((pairs, None))
    else:
        xr = _convert(r, r('extract', _class(src), *[r.index(q) for q in range(lead)], at, integer=0), _class(src), t)
        xi = r.const(0, t)
        inputs.append((src, None))
    cos, sin = chirp(r, j, False)
    re = r('sub', t, r('mul', t, xr, cos), r('mul', t, xi, sin))
    im = r('add', t, r('mul', t, xr, sin), r('mul', t, xi, cos))
    zero = r.const(0, t)
    r.done(r('select', t, inside, r('select', t, r('eq', BOOL, r.index(lead + 1), r.const(0, I32)), re, im), zero))
    ops = [_Generic([*lines, size, 2], inputs, [(torch.view_as_real(a), _same(lead + 2))], r)]
    q = _Region()
    position = q.index(0)
    mirrored = q('sub', I32, q.const(size, I32), position)
    near = q('lt', BOOL, position, q.const(n, I32))
    far = q('and', BOOL, q('gt', BOOL, mirrored, q.const(0, I32)), q('lt', BOOL, mirrored, q.const(n, I32)))
    used = q('cast', I64, q('select', I32, near, position, mirrored))
    cos, sin = chirp(q, used, True)
    value = q('select', t, q('eq', BOOL, q.index(1), q.const(0, I32)), cos, sin)
    q.done(q('select', t, q('or', BOOL, near, far), value, q.const(0, t)))
    ops.append(_Generic([size, 2], [], [(torch.view_as_real(b), _same(2))], q))
    ops += _fft(a, fa, size, sign, 1.0, t) + _fft(b, fb, size, sign, 1.0, t)
    ops += _complex('mul', product, fa, fb.expand(*lines, size))
    ops += _fft(product, conv, size, -sign, 1.0 / size, t)
    e = _Region()
    cr, ci = e.arg(0, torch.view_as_real(conv)), e.arg(1, torch.view_as_real(conv))
    cos, sin = chirp(e, e('cast', I64, e.index(lead)), False)
    re = e('sub', t, e('mul', t, cr, cos), e('mul', t, ci, sin))
    im = e('add', t, e('mul', t, cr, sin), e('mul', t, ci, cos))
    e.done(e('mul', t, e('select', t, e('eq', BOOL, e.index(lead + 1), e.const(0, I32)), re, im), e.const(scale, t)))
    loops = [*lines, n, 2]
    crows = _rows(len(loops), [*range(lead + 1)])
    return ops + [_Generic(loops, [(torch.view_as_real(conv), _complex_rows(crows, len(loops), 0)),
                                   (torch.view_as_real(conv), _complex_rows(crows, len(loops), 1))],
                           [(torch.view_as_real(dst), _same(len(loops)))], e)]


def _scale(normalization, n):
    return 1.0 if normalization == 0 else 1 / math.sqrt(n) if normalization == 1 else 1.0 / n


def _moved(value, d):
    order = [j for j in range(value.dim()) if j != d] + [d]
    return value.permute(*order)


def _transformed(target, given, outs):
    packet, x, (out,) = target.overloadpacket, given['self'], outs
    dims = [d % x.dim() for d in given['dim']]
    normalization = given['normalization']
    real = x.dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)
    t = F64 if x.dtype in (torch.float64, torch.complex128) else F32
    kind = torch.complex128 if t == F64 else torch.complex64
    ops = []
    if packet is A._fft_c2r:
        current = x
        for d in dims[:-1]:
            step = _temporary(current.shape, kind)
            ops += _fft(_moved(current, d), _moved(step, d), x.shape[d], 1, _scale(normalization, x.shape[d]), t)
            current = step
        last, length = dims[-1], given['last_dim_size']
        shape = list(current.shape)
        shape[last] = length
        full = _temporary(shape, kind)
        ops += _hermitian(current, full, last, length, t)
        whole = _temporary(shape, kind)
        ops += _fft(_moved(full, last), _moved(whole, last), length, 1, _scale(normalization, length), t)
        return ops + _copy(out, whole)
    forward = True if packet is A._fft_r2c else given['forward']
    sign = -1 if forward else 1
    shape = list(x.shape)
    current = x
    for j, d in enumerate(dims):
        step = _temporary(shape, kind)
        ops += _fft(_moved(current, d), _moved(step, d), x.shape[d], sign, _scale(normalization, x.shape[d]), t)
        current = step
    if packet is A._fft_r2c and given['onesided']:
        half = [slice(None)] * x.dim()
        half[dims[-1]] = slice(0, out.shape[dims[-1]])
        return ops + _copy(out, current[tuple(half)])
    return ops + _copy(out, current)


# the full spectrum of a real signal's half spectrum along d: X_k for k below the half's extent, conj(X_(L - k)) above
def _hermitian(half, full, d, length, t):
    n, m = full.dim(), half.shape[d]
    r = _Region()
    k = r.index(d)
    mirrored = r('sub', I32, r.const(length, I32), k)
    own = r('lt', BOOL, k, r.const(m, I32))
    at = r('select', I32, own, k, r('select', I32, r('lt', BOOL, mirrored, r.const(m, I32)), mirrored, r.const(0, I32)))
    pairs = torch.view_as_real(half)
    index = [r.index(j) if j != d else at for j in range(n)]
    re = _convert(r, r('extract', _class(pairs), *index, r.const(0, I32), integer=0), _class(pairs), t)
    im = _convert(r, r('extract', _class(pairs), *index, r.const(1, I32), integer=0), _class(pairs), t)
    r.done(r('select', t, r('eq', BOOL, r.index(n), r.const(0, I32)), re, r('select', t, own, im, r('neg', t, im))))
    return [_Generic([*full.shape, 2], [(pairs, None)], [(torch.view_as_real(full), _same(n + 1))], r)]

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


def _copying(target, given, outs):
    return _assign(outs[0], given['src']) if target.overloadpacket in (A.copy, A.copy_) else _copy(outs[0], given['self'])


# an output kept in its arithmetic kind: the structured map out = convert(f(acc, operands...)), fused by the
# transformation with its producer
def _epilogue(out, acc, t, extra=(), apply=None):
    r = _Region()
    x = _load(r, 0, acc, t)
    loaded = [_load(r, 1 + j, v, t) for j, (v, _) in enumerate(extra)]
    r.done(_store(r, apply(r, x, *loaded) if apply else x, t, out))
    return [_Generic(out.shape, [(acc, _same(out.dim()))] + list(extra), [(out, _same(out.dim()))], r)]


# an extremum's (value, index) states, joined in any association to the sequential result: the larger (smaller) value,
# NaN above every number, equal values to the earlier index; `nan_later` the NaN rule of a scan where each NaN replaces
# the kept one (max pooling), so of two NaNs the later; an index below 0 an absent state
def _ordered(c, a, b, t, larger, nan_later=False):
    (va, ia), (vb, ib) = a, b
    earlier = c('lt', BOOL, ib, ia)
    better = c('or', BOOL, c('gt' if larger else 'lt', BOOL, vb, va), c('and', BOOL, c('eq', BOOL, vb, va), earlier))
    if _floating(t):
        na, nb = c('isnan', BOOL, va), c('isnan', BOOL, vb)
        tie = c('gt', BOOL, ib, ia) if nan_later else earlier
        better = c('select', BOOL, c('or', BOOL, na, nb), c('and', BOOL, nb, c('or', BOOL, c('not', BOOL, na), tie)), better)
    zero = c.const(0, I32)
    right = c('and', BOOL, c('ge', BOOL, ib, zero), c('or', BOOL, c('lt', BOOL, ia, zero), better))
    return [c('select', t, right, vb, va), c('select', I32, right, ib, ia)]

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


# PyTorch 2.14 documentation, torch.einsum: the output's labels PARALLEL, the summed REDUCTION (SUM in the operands'
# arithmetic kind), each operand's map its labels' loops (a broadcast extent 1 a zero row)
def _contract(out, operands, labels, output, addend=None, alpha=1, beta=0):
    sizes = dict(zip(output, out.shape))
    for value, own in zip(operands, labels):
        for size, label in zip(value.shape, own):
            sizes[label] = max(sizes.get(label, 1), size)
    summed = [label for label in dict.fromkeys(label for own in labels for label in own) if label not in output]
    domain = list(output) + summed
    at = {label: d for d, label in enumerate(domain)}
    if out.is_complex():
        return _complex_contract(out, operands, labels, output, sizes, summed, addend, alpha, beta)
    t = _compute(out.dtype, *[value.dtype for value in operands], *[torch.float32] * out.dtype.is_floating_point)
    direct = alpha == 1 and (addend is None or beta == 0)
    acc = out if direct else _temporary(out.shape, STORED[t])
    r = _Region()
    product = None
    for j, value in enumerate(operands):
        x = _load(r, j, value, t)
        product = x if product is None else r('mul', t, product, x)
    r.done(product)
    maps = [_rows(len(domain), [None if n == 1 and sizes[label] != 1 else at[label] for n, label in zip(v.shape, own)])
            for v, own in zip(operands, labels)]
    made = [_Generic([sizes[label] for label in domain], list(zip(operands, maps)),
                     [(acc, _rows(len(domain), [at[label] for label in output]))], r,
                     [PARALLEL] * len(output) + [REDUCTION] * len(summed), [SUM], [t], precision=REASSOCIATE | CONTRACT)]
    if direct:
        return made

    def scaled(r, x, *c):
        x = x if alpha == 1 else r('mul', t, r.const(alpha, t), x)
        return x if not c else r('add', t, x, c[0] if beta == 1 else r('mul', t, r.const(beta, t), c[0]))
    extra = [(addend, _aligned(addend, out.shape))] if addend is not None and beta != 0 else []
    return made + _epilogue(out, acc, t, extra, scaled)


# sum_k (ar_k br_k - ai_k bi_k, ar_k bi_k + ai_k br_k)
def _complex_contract(out, operands, labels, output, sizes, summed, addend, alpha, beta):
    domain = list(output) + summed
    at = {label: d for d, label in enumerate(domain)}
    n = len(domain)
    t = F64 if any(v.dtype in (torch.complex128, torch.float64) for v in (out, *operands)) else F32
    direct = alpha == 1 and (addend is None or beta == 0)
    acc = out if direct else _temporary(out.shape, out.dtype)
    r, inputs, product = _Region(), [], None
    for value, own in zip(operands, labels):
        rows = _rows(n, [None if k == 1 and sizes[label] != 1 else at[label] for k, label in zip(value.shape, own)])
        if value.is_complex():
            parts = _pair(r, inputs, value, rows, n, t)
            product = parts if product is None else _complex_value(r, t, 'mul', product, parts, 1, None)
        else:
            x = _load(r, len(inputs), value, t)
            inputs.append((value, rows))
            product = (x, r.const(0, t)) if product is None else (r('mul', t, product[0], x), r('mul', t, product[1], x))
    r.done(*product)
    rows = _rows(n, [at[label] for label in output])
    made = [_Generic([sizes[label] for label in domain], inputs,
                     [(acc.real, rows), (acc.imag, rows)], r,
                     [PARALLEL] * len(output) + [REDUCTION] * len(summed), [SUM, SUM], [t, t],
                     precision=REASSOCIATE | CONTRACT)]
    if direct:
        return made
    if alpha != 1:
        scaled = _temporary(out.shape, out.dtype)
        made += _complex('mul', scaled, acc, alpha)
        acc = scaled
    return made + (_complex('add', out, acc, addend, beta) if addend is not None and beta != 0 else
                   _complex('copy', out, acc))


def _matrix(out, a, b, bias=None):
    return _contract(out, [a, b], [('n', 'k'), ('k', 'm')], ('n', 'm'), bias, 1, int(bias is not None))


def _folded(value):
    try:
        return value.view(-1, value.shape[-1])
    except RuntimeError:
        return None


# PyTorch 2.14 documentation, torch.mm, torch.bmm, torch.matmul, torch.addmm and torch.nn.functional.linear; PyTorch
# v2.11.0 aten/src/ATen/native/LinearAlgebra.cpp, _matmul_impl
def _product(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    if packet is A.addmm:
        c, a, b, alpha, beta = given['self'], given['mat1'], given['mat2'], given['alpha'], given['beta']
        _typed(target, [], outs, CONTRACTED)
        return _contract(out, [a, b], [('n', 'k'), ('k', 'm')], ('n', 'm'), c if beta != 0 else None, alpha, beta)
    if packet is A.linear:
        x, w, bias = given['input'], given['weight'], given['bias']
        _typed(target, [], outs, CONTRACTED)
        rows, folded = _folded(x), _folded(out)
        if w.dim() == 2 and rows is not None and folded is not None:
            return _matrix(folded, rows, w.t(), bias)
        lead = tuple(('batch', i) for i in range(x.dim() - 1))
        return _contract(out, [x, w], [lead + ('k',), ('m', 'k')[2 - w.dim():]], lead + ('m',) * (w.dim() == 2), bias,
                         1, int(bias is not None))
    a, b = given['self'], given['mat2'] if packet in (A.mm, A.bmm) else given['other']
    _typed(target, [], outs, CONTRACTED)
    if packet is A.mm or packet is A.matmul and a.dim() == b.dim() == 2:
        return _matrix(out, a, b)
    if packet is A.matmul and a.dim() > 2 and b.dim() == 2 and _folded(a) is not None and _folded(out) is not None:
        return _matrix(_folded(out), _folded(a), b)
    batch = out.dim() - (a.dim() > 1) - (b.dim() > 1)

    def side(value, matrix):
        rank = value.dim() - len(matrix)
        return tuple(('batch', batch - rank + i) for i in range(rank)) + matrix
    return _contract(out, [a, b], [side(a, ('n', 'k') if a.dim() > 1 else ('k',)),
                                   side(b, ('k', 'm') if b.dim() > 1 else ('k',))],
                     tuple(('batch', i) for i in range(batch)) + ('n',) * (a.dim() > 1) + ('m',) * (b.dim() > 1))


# PyTorch 2.14 documentation, torch.einsum: more than two operands contracted pairwise left to right, each step's
# result keeping the labels a later operand or the output uses (no product loop over every summed label at once)
def _summation(target, given, outs):
    (out,), operands = outs, list(given['tensors'])
    _typed(target, [], outs, CONTRACTED)
    left, arrow, right = given['equation'].replace(' ', '').partition('->')
    terms = left.split(',')
    covered = max([value.dim() - len(term.replace('...', '')) for term, value in zip(terms, operands) if '...' in term],
                  default=0)
    ellipsis = tuple(('...', i) for i in range(covered))

    def labelled(term, rank):
        head, dots, tail = term.partition('...')
        return tuple(head) + (ellipsis[covered - (rank - len(head) - len(tail)):] if dots else ()) + tuple(tail)
    letters = left.replace('...', '').replace(',', '')
    output = labelled(right, out.dim()) if arrow else \
        ellipsis + tuple(sorted(c for c in set(letters) if letters.count(c) == 1))
    labels = [labelled(term, value.dim()) for term, value in zip(terms, operands)]
    if len(operands) <= 2:
        return _contract(out, operands, labels, output)
    sizes = {}
    for value, own in zip(operands, labels):
        for size, label in zip(value.shape, own):
            sizes[label] = max(sizes.get(label, 1), size)
    ops, made, own = [], operands[0], labels[0]
    for j in range(1, len(operands)):
        later = {label for others in labels[j + 1:] for label in others} | set(output)
        kept = output if j == len(operands) - 1 else tuple(dict.fromkeys(l for l in own + labels[j] if l in later))
        step = out if j == len(operands) - 1 else _temporary([sizes[label] for label in kept], out.dtype)
        ops += _contract(step, [made, operands[j]], [own, labels[j]], kept)
        made, own = step, kept
    return ops


# Reference-LAPACK SRC/dgetf2.f, zgetf2.f, dgetrs.f, zgetrs.f, dpotf2.f and zpotf2.f;
# torch 2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp
def _linalg_transpose(x, ops, transpose=True, conjugate=False):
    x = x.transpose(-2, -1) if transpose else x
    if conjugate and x.is_complex():
        made = _temporary(x.shape, x.dtype)
        ops += _complex('conj', made, x)
        return made
    return x


def _linalg_identity(out):
    r = _Region()
    target = torch.view_as_real(out) if out.is_complex() else out
    row = out.dim() - 2
    equal = r('eq', BOOL, r.index(row), r.index(row + 1))
    if out.is_complex():
        equal = r('and', BOOL, equal, r('eq', BOOL, r.index(out.dim()), r.const(0, I32)))
    r.done(r('cast', _class(target), equal))
    return [_Generic(target.shape, [], [(target, _same(target.dim()))], r)]


def _linalg_swap(x, pivot, row, ops):
    made, r = _temporary(x.shape, x.dtype), _Region()
    source = torch.view_as_real(x) if x.is_complex() else x
    target = torch.view_as_real(made) if made.is_complex() else made
    axis, t = x.dim() - 2, _class(source)
    index = pivot.reshape(*pivot.shape, *([1] * (source.dim() - pivot.dim())))
    selected = r('sub', I32, _load(r, 1, index, I32), r.const(1, I32))
    inputs = [(source, None), (index, _aligned(index, target.shape))]
    if isinstance(row, torch.Tensor):
        row = row.reshape(*row.shape, *([1] * (source.dim() - row.dim())))
        inputs.append((row, _aligned(row, target.shape)))
        origin = _load(r, 2, row, I32)
    else:
        origin = r.const(row, I32)
    at, own = [r.index(d) for d in range(source.dim())], r.index(axis)
    at[axis] = r('select', I32, r('eq', BOOL, own, origin), selected,
                 r('select', I32, r('eq', BOOL, own, selected), origin, own))
    r.done(r('extract', t, *at, integer=0))
    ops.append(_Generic(target.shape, inputs, [(target, _same(target.dim()))], r))
    return made


def _linalg_pivot_magnitude(x, ops):
    scores = _temporary(x.shape, x.real.dtype)
    ops += _elementwise('abs1' if x.is_complex() else 'abs', scores, x)
    return scores


def _linalg_lu(a, lu, pivots, info, pivot=True):
    ops, current = [], a
    ops += _fill(info, 0)
    m, n = a.shape[-2:]
    for j in range(min(m, n)):
        column, p = current[..., j:, j], pivots[..., j]
        if pivot:
            scores = _linalg_pivot_magnitude(column, ops)
            ops += _extreme(A.argmax.default, {'self': scores, 'dim': -1, 'keepdim': False}, [p])
            ops += _elementwise('add', p, p, j + 1)
            current = _linalg_swap(current, p, j, ops)
        else:
            ops += _fill(p, j + 1)
            if current is a:
                current = _temporary(a.shape, a.dtype)
                ops += _copy(current, a)
        diagonal = current[..., j, j]
        zero, first = _temporary(info.shape, torch.bool), _temporary(info.shape, torch.bool)
        ops += _elementwise('eq', zero, diagonal, 0)
        ops += _elementwise('eq', first, info, 0)
        ops += _elementwise('logical_and', first, first, zero)
        ops += _elementwise('where', info, first, j + 1, info)
        if j + 1 < m:
            safe = _temporary(diagonal.shape, diagonal.dtype)
            ops += _elementwise('where', safe, zero, 1, diagonal)
            below = current[..., j + 1:, j]
            ops += _elementwise('div', below, below, safe.unsqueeze(-1))
            if j + 1 < n:
                block = current[..., j + 1:, j + 1:]
                change = _temporary(block.shape, block.dtype)
                ops += _product(A.matmul.default, {'self': below.unsqueeze(-1),
                                                    'other': current[..., j, j + 1:].unsqueeze(-2)}, [change])
                ops += _elementwise('sub', block, block, change)
    return ops + _copy(lu, current)


def _linalg_component_maximum(x, dimensions, ops):
    dimensions = [d % x.dim() for d in dimensions]
    source = torch.view_as_real(x) if x.is_complex() else x
    magnitude = _temporary(source.shape, source.dtype)
    maximum = _temporary([s for d, s in enumerate(x.shape) if d not in dimensions], source.dtype)
    axes = dimensions + ([x.dim()] if x.is_complex() else [])
    ops += _elementwise('abs', magnitude, source)
    ops += _reduced(A.amax.default, {'self': magnitude, 'dim': axes, 'keepdim': False}, [maximum])
    return maximum


def _linalg_rescale(b, out, scale, ratio, ops):
    ops += _elementwise('mul', b, b, ratio.unsqueeze(-2))
    ops += _elementwise('mul', out, out, ratio.unsqueeze(-2))
    ops += _elementwise('mul', scale, scale, ratio)


def _linalg_triangular(a, b, out, upper, unit=False, scale=None):
    ops = []
    n = a.shape[-1]
    if scale is not None:
        rhs = _temporary(out.shape, out.dtype)
        ops += _assign(rhs, b) + _fill(out, 0) + _fill(scale, 1)
        b = rhs
    for j in range(n - 1, -1, -1) if upper else range(n):
        start, end = (j + 1, n) if upper else (0, j)
        result = out[..., j:j + 1, :]
        if scale is not None:
            limits = torch.finfo(out.real.dtype)
            big = limits.eps / limits.tiny
            magnitude = _linalg_component_maximum(b, [-2], ops)
            safe, ratio = (_temporary(scale.shape, out.real.dtype) for _ in range(2))
            zero = _temporary(scale.shape, torch.bool)
            ops += _elementwise('eq', zero, magnitude, 0)
            ops += _elementwise('where', safe, zero, 1, magnitude)
            ops += _elementwise('div', ratio, big, safe)
            ops += _elementwise('minimum', ratio, ratio, 1)
            if start != end:
                row = _linalg_component_maximum(a[..., j, start:end], [-1], ops).unsqueeze(-1)
                solved = _linalg_component_maximum(out[..., start:end, :], [-2], ops)
                bound, row_safe = (_temporary(scale.shape, out.real.dtype) for _ in range(2))
                row_zero = _temporary(scale.shape, torch.bool)
                ops += _elementwise('eq', row_zero, row, 0)
                ops += _elementwise('where', row_safe, row_zero, 1, row)
                ops += _elementwise('eq', zero, solved, 0)
                ops += _elementwise('where', safe, zero, 1, solved)
                ops += _elementwise('div', bound, big, row_safe)
                ops += _elementwise('div', bound, bound, safe)
                ops += _elementwise('div', bound, bound, (2 if out.is_complex() else 1) * (end - start))
                ops += _elementwise('logical_or', zero, zero, row_zero)
                ops += _elementwise('where', bound, zero, 1, bound)
                ops += _elementwise('minimum', ratio, ratio, bound)
            _linalg_rescale(b, out, scale, ratio, ops)
        if start != end:
            dot = _temporary(result.shape, result.dtype)
            ops += _product(A.matmul.default, {'self': a[..., j:j + 1, start:end],
                                                'other': out[..., start:end, :]}, [dot])
            ops += _elementwise('sub', result, b[..., j:j + 1, :], dot)
        else:
            ops += _assign(result, b[..., j:j + 1, :])
        if not unit:
            if scale is not None:
                magnitude, diagonal, ratio = (_temporary(scale.shape, out.real.dtype) for _ in range(3))
                large, zero = _temporary(scale.shape, torch.bool), _temporary(scale.shape, torch.bool)
                ops += _elementwise('abs', magnitude, result.squeeze(-2))
                ops += _elementwise('abs', diagonal, a[..., j, j].unsqueeze(-1))
                ops += _elementwise('gt', large, magnitude, diagonal)
                ops += _elementwise('eq', zero, magnitude, 0)
                ops += _elementwise('where', magnitude, zero, 1, magnitude)
                ops += _elementwise('div', ratio, diagonal, magnitude)
                ops += _elementwise('where', ratio, large, ratio, 1)
                _linalg_rescale(b, out, scale, ratio, ops)
            ops += _elementwise('div', result, result, a[..., j:j + 1, j:j + 1])
    return ops


def _linalg_lu_solve(lu, pivots, b, out, left=True, adjoint=False):
    ops = []
    target = out if left else out.transpose(-2, -1)
    rhs = (b if left else b.transpose(-2, -1)).expand(target.shape)
    transposed = left == adjoint
    factor = _linalg_transpose(lu, ops, transposed, adjoint)
    if not transposed:
        for j in range(pivots.shape[-1]):
            rhs = _linalg_swap(rhs, pivots[..., j], j, ops)
        middle = _temporary(target.shape, target.dtype)
        ops += _linalg_triangular(factor, rhs, middle, False, True)
        ops += _linalg_triangular(factor, middle, target, True)
    else:
        middle, result = _temporary(target.shape, target.dtype), _temporary(target.shape, target.dtype)
        ops += _linalg_triangular(factor, rhs, middle, False)
        ops += _linalg_triangular(factor, middle, result, True, True)
        for j in reversed(range(pivots.shape[-1])):
            result = _linalg_swap(result, pivots[..., j], j, ops)
        ops += _copy(target, result)
    return ops


def _linalg_cholesky(a, out, info, upper=False):
    ops, n = [], a.shape[-1]
    source = _linalg_transpose(a, ops, upper, upper)
    work, factor = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
    ops += _copy(work, source) + _fill(factor, 0) + _fill(info, 0)
    for j in range(n):
        diagonal = work[..., j, j].real
        bad, first = _temporary(info.shape, torch.bool), _temporary(info.shape, torch.bool)
        root = _temporary(diagonal.shape, diagonal.dtype)
        ops += _elementwise('gt', bad, diagonal, 0)
        ops += _elementwise('logical_not', bad, bad)
        ops += _elementwise('eq', first, info, 0)
        ops += _elementwise('logical_and', first, first, bad)
        ops += _elementwise('where', info, first, j + 1, info)
        ops += _elementwise('sqrt', root, diagonal)
        ops += _elementwise('where', root, bad, diagonal, root)
        ops += _copy(factor[..., j, j], root)
        if j + 1 < n:
            column = factor[..., j + 1:, j:j + 1]
            ops += _elementwise('div', column, work[..., j + 1:, j:j + 1], root[..., None, None])
            row = _linalg_transpose(column, ops, True, True)
            change = _temporary(work[..., j + 1:, j + 1:].shape, a.dtype)
            ops += _product(A.matmul.default, {'self': column, 'other': row}, [change])
            ops += _elementwise('sub', work[..., j + 1:, j + 1:], work[..., j + 1:, j + 1:], change)
    return ops + _copy(out, _linalg_transpose(factor, ops, upper, upper))


def _linalg_symmetric(a, ops, hermitian=True, upper=False):
    matrix, other = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
    opposite = _linalg_transpose(a, ops, True, hermitian)
    ops += _triangular_part(A.triu.default if upper else A.tril.default, {'self': a}, [matrix])
    ops += _triangular_part(A.tril.default if upper else A.triu.default,
                            {'self': opposite, 'diagonal': -1 if upper else 1}, [other])
    ops += _elementwise('add', matrix, matrix, other)
    if hermitian and a.is_complex():
        ops += _fill(matrix.diagonal(dim1=-2, dim2=-1).imag, 0)
    return matrix


# Reference-LAPACK SRC/dsytf2.f, zsytf2.f, zhetf2.f, dsytrs.f and zhetrs.f
def _linalg_diagonal_block(a, j, pair, hermitian, ops, active=None):
    width = min(2, a.shape[-1] - j)
    block = _temporary((*a.shape[:-2], width, width), a.dtype)
    ops += _fill(block, 0)
    if active is None:
        ops += _copy(block[..., 0, 0], a[..., j, j])
    else:
        ops += _elementwise('where', block[..., 0, 0], active, a[..., j, j], 1)
    if width == 2:
        ops += _elementwise('where', block[..., 1, 1], pair, a[..., j + 1, j + 1], 1)
        ops += _elementwise('where', block[..., 1, 0], pair, a[..., j + 1, j], 0)
        opposite = _linalg_transpose(block[..., 1:2, :1], ops, True, hermitian)
        ops += _copy(block[..., :1, 1:2], opposite)
    return block


def _linalg_block_solve(a, b, out, ops, left=True):
    lu, pivots = _temporary(a.shape, a.dtype), _temporary(a.shape[:-1], torch.int32)
    info = _temporary(a.shape[:-2], torch.int32)
    ops += _linalg_lu(a, lu, pivots, info)
    ops += _linalg_lu_solve(lu, pivots, b, out, left)


def _linalg_block_columns(a, j, pair, ops):
    columns = _temporary((*a.shape[:-2], a.shape[-1] - j - 1, 2), a.dtype)
    ops += _copy(columns[..., 0], a[..., j + 1:, j])
    ops += _elementwise('where', columns[..., 1], pair.unsqueeze(-1), a[..., j + 1:, j + 1], 0)
    ops += _elementwise('where', columns[..., 0, 0], pair, 0, columns[..., 0, 0])
    ops += _fill(columns[..., 0, 1], 0)
    return columns


def _linalg_ldl(a, ld, pivots, info, hermitian):
    ops, batch, n = [], a.shape[:-2], a.shape[-1]
    work = _linalg_symmetric(a, ops, hermitian)
    skip = _temporary(batch, torch.bool)
    ops += _fill(info, 0) + _fill(pivots, 0) + _fill(skip, False)
    alpha = (1 + math.sqrt(17)) / 8
    for j in range(n):
        active, bad, pair = (_temporary(batch, torch.bool) for _ in range(3))
        pivot = _temporary(batch, torch.int32)
        diagonal = _linalg_pivot_magnitude(work[..., j, j], ops)
        ops += _elementwise('logical_not', active, skip) + _fill(pair, False) + _fill(pivot, j + 1)
        if j + 1 < n:
            scores = _linalg_pivot_magnitude(work[..., j + 1:, j], ops)
            colmax, rowmax = (_temporary(batch, a.real.dtype) for _ in range(2))
            imax = _temporary(batch, torch.int32)
            ops += _extreme(A.max.dim, {'self': scores, 'dim': -1}, [colmax, imax])
            ops += _elementwise('add', imax, imax, 1)
            row = _temporary((*batch, n - j, 1), a.dtype)
            ops += _linalg_columns(work[..., j:, j:].transpose(-2, -1), imax.unsqueeze(-1), row)
            row = row[..., 0]
            row_scores = _linalg_pivot_magnitude(row, ops)
            selected = imax.unsqueeze(-1)
            r, t = _Region(), _class(row_scores)
            r.done(r('select', t, r('eq', BOOL, r.index(row_scores.dim() - 1),
                                   _load(r, 1, selected, I32)), r.const(0, t), _load(r, 0, row_scores, t)))
            ops.append(_Generic(row_scores.shape, [(row_scores, _same(row_scores.dim())),
                                (selected, _aligned(selected, row_scores.shape))],
                                [(row_scores, _same(row_scores.dim()))], r))
            ops += _reduced(A.amax.default, {'self': row_scores, 'dim': [-1], 'keepdim': False}, [rowmax])
            at = _temporary((*batch, 1, 1), a.dtype)
            ops += _linalg_columns(row.unsqueeze(-2), imax.unsqueeze(-1), at)
            alternate = _linalg_pivot_magnitude(at[..., 0, 0], ops)
            threshold, safe = (_temporary(batch, a.real.dtype) for _ in range(2))
            stay, again, other, zero = (_temporary(batch, torch.bool) for _ in range(4))
            ops += _elementwise('mul', threshold, colmax, alpha)
            ops += _elementwise('ge', stay, diagonal, threshold)
            ops += _elementwise('eq', zero, rowmax, 0)
            ops += _elementwise('where', safe, zero, 1, rowmax)
            ops += _elementwise('div', safe, colmax, safe)
            ops += _elementwise('mul', threshold, threshold, safe)
            ops += _elementwise('ge', again, diagonal, threshold)
            ops += _elementwise('logical_or', stay, stay, again)
            ops += _elementwise('mul', threshold, rowmax, alpha)
            ops += _elementwise('ge', other, alternate, threshold)
            ops += _elementwise('logical_or', pair, stay, other)
            ops += _elementwise('logical_not', pair, pair)
            ops += _elementwise('add', imax, imax, j + 1)
            ops += _elementwise('where', pivot, stay, j + 1, imax)
            ops += _elementwise('maximum', threshold, diagonal, colmax)
            ops += _elementwise('eq', bad, threshold, 0)
        else:
            ops += _elementwise('eq', bad, diagonal, 0)
        nan, usable, first = (_temporary(batch, torch.bool) for _ in range(3))
        ops += _elementwise('isnan', nan, diagonal)
        ops += _elementwise('logical_or', bad, bad, nan)
        ops += _elementwise('logical_and', bad, bad, active)
        ops += _elementwise('eq', first, info, 0)
        ops += _elementwise('logical_and', first, first, bad)
        ops += _elementwise('where', info, first, j + 1, info)
        ops += _elementwise('logical_not', usable, bad)
        ops += _elementwise('logical_and', usable, usable, active)
        ops += _elementwise('logical_and', pair, pair, usable)
        ops += _elementwise('where', pivot, usable, pivot, j + 1)
        signed = _temporary(batch, torch.int32)
        ops += _elementwise('neg', signed, pivot)
        ops += _elementwise('where', signed, pair, signed, pivot)
        ops += _elementwise('where', pivots[..., j], active, signed, pivots[..., j])
        if j + 1 < n:
            ops += _elementwise('where', pivots[..., j + 1], pair, signed, pivots[..., j + 1])
            local, target = _temporary(batch, torch.int32), _temporary(batch, torch.int32)
            ops += _elementwise('sub', local, pivot, j) + _copy(target, pair)
            trailing = _linalg_swap(work[..., j:, j:], local, target, ops)
            trailing = _linalg_swap(trailing.transpose(-2, -1), local, target, ops).transpose(-2, -1)
            block = _linalg_diagonal_block(trailing, 0, pair, hermitian, ops, usable)
            columns = _linalg_block_columns(trailing, 0, pair, ops)
            lower = _temporary(columns.shape, columns.dtype)
            _linalg_block_solve(block, columns, lower, ops, False)
            change = _temporary(trailing[..., 1:, 1:].shape, a.dtype)
            opposite = _linalg_transpose(columns, ops, True, hermitian)
            ops += _product(A.matmul.default, {'self': lower, 'other': opposite}, [change])
            ops += _elementwise('sub', trailing[..., 1:, 1:], trailing[..., 1:, 1:], change)
            ops += _elementwise('where', lower[..., 0, 0], pair, trailing[..., 1, 0], lower[..., 0, 0])
            ops += _copy(trailing[..., 1:, 0], lower[..., 0])
            if j + 2 < n:
                ops += _elementwise('where', trailing[..., 2:, 1], pair.unsqueeze(-1),
                                     lower[..., 1:, 1], trailing[..., 2:, 1])
            if hermitian and a.is_complex():
                ops += _fill(trailing.diagonal(dim1=-2, dim2=-1).imag, 0)
            ops += _elementwise('where', work[..., j:, j:], usable[..., None, None], trailing, work[..., j:, j:])
        ops += _copy(skip, pair)
    return ops + _triangular_part(A.tril.default, {'self': work}, [ld])


def _linalg_ldl_solve(ld, pivots, b, out, hermitian):
    batch, n, ops = out.shape[:-2], out.shape[-2], []
    factor = ld.expand(*batch, n, n)
    pivots = pivots.expand(*batch, n)
    work = _temporary(out.shape, out.dtype)
    skip = _temporary(batch, torch.bool)
    ops += _assign(work, b) + _fill(skip, False)
    for j in range(n):
        active, pair = _temporary(batch, torch.bool), _temporary(batch, torch.bool)
        pivot, target = _temporary(batch, torch.int32), _temporary(batch, torch.int32)
        ops += _elementwise('logical_not', active, skip)
        ops += _elementwise('lt', pair, pivots[..., j], 0)
        ops += _elementwise('logical_and', pair, pair, active)
        ops += _elementwise('abs', pivot, pivots[..., j])
        ops += _copy(target, pair) + _elementwise('add', target, target, j)
        moved = _linalg_swap(work, pivot, target, ops)
        block = _linalg_diagonal_block(factor, j, pair, hermitian, ops, active)
        width = block.shape[-1]
        rhs = _temporary((*batch, width, out.shape[-1]), out.dtype)
        ops += _copy(rhs, moved[..., j:j + width, :])
        if j + 1 < n:
            columns = _linalg_block_columns(factor, j, pair, ops)
            change = _temporary(moved[..., j + 1:, :].shape, out.dtype)
            ops += _product(A.matmul.default, {'self': columns, 'other': rhs}, [change])
            ops += _elementwise('sub', moved[..., j + 1:, :], moved[..., j + 1:, :], change)
        solution = _temporary(rhs.shape, rhs.dtype)
        _linalg_block_solve(block, rhs, solution, ops)
        ops += _copy(moved[..., j, :], solution[..., 0, :])
        if width == 2:
            ops += _elementwise('where', moved[..., j + 1, :], pair.unsqueeze(-1),
                                 solution[..., 1, :], moved[..., j + 1, :])
        ops += _elementwise('where', work, active[..., None, None], moved, work)
        ops += _copy(skip, pair)
    ops += _fill(skip, False)
    for j in reversed(range(n)):
        active, pair = _temporary(batch, torch.bool), _temporary(batch, torch.bool)
        pivot = _temporary(batch, torch.int32)
        ops += _elementwise('logical_not', active, skip)
        ops += _elementwise('lt', pair, pivots[..., j], 0)
        ops += _elementwise('logical_and', pair, pair, active)
        ops += _elementwise('abs', pivot, pivots[..., j])
        start = max(j - 1, 0)
        width = j - start + 1
        made = _temporary(work.shape, work.dtype)
        ops += _copy(made, work)
        if j + 1 < n:
            lower = factor[..., j + 1:, start:j + 1]
            transposed = _linalg_transpose(lower, ops, True, hermitian)
            change = _temporary((*batch, width, out.shape[-1]), out.dtype)
            ops += _product(A.matmul.default, {'self': transposed, 'other': work[..., j + 1:, :]}, [change])
            ops += _elementwise('sub', made[..., j, :], work[..., j, :], change[..., -1, :])
            if j:
                ops += _elementwise('where', change[..., 0, :], pair.unsqueeze(-1), change[..., 0, :], 0)
                ops += _elementwise('sub', made[..., j - 1, :], work[..., j - 1, :], change[..., 0, :])
        made = _linalg_swap(made, pivot, j, ops)
        ops += _elementwise('where', work, active[..., None, None], made, work)
        ops += _copy(skip, pair)
    return ops + _copy(out, work)


FACTORIZATIONS = (A.linalg_lu_factor_ex, A._linalg_solve_ex, A.linalg_solve_ex, A.linalg_inv_ex,
                  A.linalg_cholesky_ex, A.linalg_solve_triangular, A.triangular_solve, A.linalg_lu_solve,
                  A.cholesky_solve, A.cholesky_inverse, A.linalg_ldl_factor_ex, A.linalg_ldl_solve)


def _linalg_vector_rhs(a, b):
    return b.dim() == 1 or b.dim() == a.dim() - 1 and b.shape == a.shape[:-1]


def _factorization(target, given, outs):
    packet, ops = target.overloadpacket, []
    a = given.get('A', given.get('self'))
    if packet is A.linalg_lu_factor_ex:
        return _linalg_lu(a, *outs, given['pivot'])
    if packet is A.linalg_cholesky_ex:
        return _linalg_cholesky(a, *outs, given['upper'])
    if packet is A.linalg_ldl_factor_ex:
        return _linalg_ldl(a, *outs, given['hermitian'])
    if packet is A.linalg_ldl_solve:
        return _linalg_ldl_solve(given['LD'], given['pivots'], given['B'], outs[0], given['hermitian'])
    if packet is A.linalg_lu_solve:
        return _linalg_lu_solve(given['LU'], given['pivots'], given['B'], outs[0], given['left'], given['adjoint'])
    if packet in (A.linalg_solve_triangular, A.triangular_solve):
        b = given['B'] if packet is A.linalg_solve_triangular else given['self']
        left, upper = given.get('left', True), given['upper']
        transpose = given.get('transpose', False) != (not left)
        factor = a.transpose(-2, -1) if transpose else a
        ops += _linalg_triangular(factor, b if left else b.transpose(-2, -1),
                                  outs[0] if left else outs[0].transpose(-2, -1), upper != transpose,
                                  given['unitriangular'])
        return ops + (_copy(outs[1], a) if len(outs) > 1 else [])
    if packet in (A.cholesky_solve, A.cholesky_inverse):
        factor = given['input2'] if packet is A.cholesky_solve else a
        b = given['self'] if packet is A.cholesky_solve else _temporary(outs[0].shape, outs[0].dtype)
        if packet is A.cholesky_inverse:
            ops += _linalg_identity(b)
        lower = _linalg_transpose(factor, ops, given['upper'], given['upper'])
        middle = _temporary(outs[0].shape, outs[0].dtype)
        ops += _linalg_triangular(lower, b, middle, False)
        ops += _linalg_triangular(_linalg_transpose(lower, ops, True, True), middle, outs[0], True)
        return ops
    result, info = outs[0], outs[-1]
    lu, pivots = outs[1:3] if packet is A._linalg_solve_ex else \
        (_temporary(a.shape, a.dtype), _temporary(a.shape[:-1], torch.int32))
    ops += _linalg_lu(a, lu, pivots, info)
    b = given.get('B')
    if b is None:
        b = _temporary(result.shape, result.dtype)
        ops += _linalg_identity(b)
    vector = _linalg_vector_rhs(a, b)
    ops += _linalg_lu_solve(lu, pivots, b.unsqueeze(-1) if vector else b,
                           result.unsqueeze(-1) if vector else result, given.get('left', True))
    return ops


class _Check:
    def __init__(self, info, code, message=''):
        self.info, self.code, self.place, self.node = info, code, CPU, 0
        self.message = message


def _linalg_nonsingular(diagonal, code, message):
    if not diagonal.shape[-1]:
        return []
    zero = _temporary(diagonal.shape, torch.bool)
    scores = _temporary(diagonal.shape, torch.int32)
    any_zero, info = _temporary(diagonal.shape[:-1], torch.bool), _temporary(diagonal.shape[:-1], torch.int32)
    ops = _elementwise('eq', zero, diagonal, 0) + _copy(scores, zero)
    ops += _extreme(A.argmax.default, {'self': scores, 'dim': -1, 'keepdim': False}, [info])
    ops += _elementwise('add', info, info, 1)
    ops += _fold(A.any.dim, {'self': zero, 'dim': -1, 'keepdim': False}, [any_zero])
    ops += _elementwise('where', info, any_zero, info, 0)
    return ops + [_Check(info, code, message)]


def _factorized(target, given, outs):
    ops = _factorization(target, given, outs)
    if target.overloadpacket is A.cholesky_inverse:
        ops = _linalg_nonsingular(given['self'].diagonal(dim1=-2, dim2=-1), 2, 'cholesky_inverse') + ops
    if given.get('check_errors'):
        ops.append(_Check(outs[-1], 2 if target.overloadpacket is A.linalg_cholesky_ex else 1))
    return ops


def _checked(target, given, outs):
    code = 2 if given['api_name'] in ('linalg.cholesky', 'linalg.cholesky_ex', 'cholesky') else 1
    return [_Check(given['info'], code, given['api_name'])]


def _determinant(target, given, outs, transpose=False):
    a, logarithm = given['A'], target.overloadpacket is A._linalg_slogdet
    value, lu, pivots = outs[0], outs[-2], outs[-1]
    info = _temporary(a.shape[:-2], torch.int32)
    ops = _linalg_lu(a.transpose(-2, -1) if transpose else a, lu, pivots, info)
    ops += _fill(value, 1)
    if logarithm:
        ops += _fill(outs[1], 0)
    for j in range(a.shape[-1]):
        diagonal = lu[..., j, j]
        swap, negative = _temporary(value.shape, torch.bool), _temporary(value.shape, value.dtype)
        if logarithm:
            magnitude, safe = _temporary(value.shape, a.real.dtype), _temporary(value.shape, a.real.dtype)
            zero, phase = _temporary(value.shape, torch.bool), _temporary(value.shape, a.dtype)
            ops += _elementwise('abs', magnitude, diagonal)
            ops += _elementwise('eq', zero, magnitude, 0)
            ops += _elementwise('where', safe, zero, 1, magnitude)
            ops += _elementwise('div', phase, diagonal, safe)
            ops += _elementwise('mul', value, value, phase)
            ops += _elementwise('log', magnitude, magnitude)
            ops += _elementwise('add', outs[1], outs[1], magnitude)
        else:
            ops += _elementwise('mul', value, value, diagonal)
        ops += _elementwise('ne', swap, pivots[..., j], j + 1)
        ops += _elementwise('neg', negative, value)
        ops += _elementwise('where', value, swap, negative, value)
    return ops


def _unpacked_lu(target, given, outs):
    a, (permutation, lower, upper) = given['A'], outs
    m, n = a.shape[-2:]
    packed, pivots = _temporary(a.shape, a.dtype), _temporary((*a.shape[:-2], min(m, n)), torch.int32)
    info = _temporary(a.shape[:-2], torch.int32)
    ops = _linalg_lu(a, packed, pivots, info, given['pivot'])
    ops += _triangular_part(A.tril.default, {'self': packed[..., :, :min(m, n)], 'diagonal': -1}, [lower])
    ops += _fill(lower.diagonal(dim1=-2, dim2=-1), 1)
    ops += _triangular_part(A.triu.default, {'self': packed[..., :min(m, n), :]}, [upper])
    if permutation.numel():
        ops += _linalg_identity(permutation)
        current = permutation
        for j in reversed(range(min(m, n))):
            current = _linalg_swap(current, pivots[..., j], j, ops)
        ops += _copy(permutation, current)
    return ops


ASSERTIONS = (A._assert_async, A._assert_scalar, A._assert_tensor_metadata,
              A.sym_constrain_range, A.sym_constrain_range_for_size)


def _asserted(target, given, outs):
    if target.overloadpacket is A._assert_tensor_metadata:
        return []
    value = given.get('self', given.get('size'))
    if not isinstance(value, torch.Tensor):
        target(**given)
        return []
    predicate = _temporary(value.shape, torch.bool)
    if target.overloadpacket in (A.sym_constrain_range, A.sym_constrain_range_for_size):
        ops = _fill(predicate, True)
        for name, comparison in (('min', 'ge'), ('max', 'le')):
            if given.get(name) is not None:
                bound = _temporary(value.shape, torch.bool)
                ops += _elementwise(comparison, bound, value, given[name])
                ops += _elementwise('logical_and', predicate, predicate, bound)
    else:
        ops = _elementwise('ne', predicate, value, 0)
    info = _temporary(value.shape, torch.int32)
    ops += _elementwise('where', info, predicate, 0, 1)
    return ops + [_Check(info, 4, given.get('assert_msg', str(target)))]


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
        ops += _elementwise('eq', inactive, tail, 0)
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


def _linalg_reflect(block, v, tau, ops, left=True, adjoint=False):
    vh = _linalg_transpose(v, ops, True, True)
    scalar = _linalg_transpose(tau[..., None, None], ops, False, adjoint)
    dot = _temporary((*block.shape[:-2], 1, block.shape[-1]) if left else
                     (*block.shape[:-2], block.shape[-2], 1), block.dtype)
    ops += _product(A.matmul.default, {'self': vh if left else block, 'other': block if left else v}, [dot])
    ops += _elementwise('mul', dot, dot, scalar)
    change = _temporary(block.shape, block.dtype)
    ops += _product(A.matmul.default, {'self': v if left else dot, 'other': dot if left else vh}, [change])
    ops += _elementwise('sub', block, block, change)


def _linalg_geqrf(a, packed, tau, pivots=None):
    ops = _copy(packed, a)
    m, n = a.shape[-2:]
    for j in range(min(m, n)):
        if pivots is not None:
            norms = _temporary((*a.shape[:-2], n - j), a.real.dtype)
            _linalg_lengths(packed[..., j:, j:], norms, ops, normalize=False)
            ops += _extreme(A.argmax.default, {'self': norms, 'dim': -1, 'keepdim': False}, [pivots[..., j]])
            ops += _elementwise('add', pivots[..., j], pivots[..., j], j + 1)
            moved = _linalg_swap(packed.transpose(-2, -1), pivots[..., j], j, ops)
            ops += _copy(packed, moved.transpose(-2, -1))
        beta = _temporary(a.shape[:-2], a.real.dtype)
        v = _linalg_reflector(packed[..., j:, j], tau[..., j], beta, ops)
        if j + 1 < n:
            _linalg_reflect(packed[..., j:, j + 1:], v, tau[..., j], ops, adjoint=True)
        ops += _copy(packed[..., j, j], beta)
        if j + 1 < m:
            ops += _copy(packed[..., j + 1:, j], v[..., 1:, 0])
    return ops


def _linalg_householder(packed, tau, out, b=None, left=True, adjoint=False):
    ops = _linalg_identity(out) if b is None else _assign(out, b)
    order = range(tau.shape[-1]) if left == adjoint else reversed(range(tau.shape[-1]))
    for j in order:
        v = _temporary((*packed.shape[:-2], packed.shape[-2] - j, 1), packed.dtype)
        ops += _fill(v[..., 0, 0], 1)
        if v.shape[-2] > 1:
            ops += _copy(v[..., 1:, 0], packed[..., j + 1:, j])
        _linalg_reflect(out[..., j:, :] if left else out[..., :, j:], v, tau[..., j], ops, left, adjoint)
    return ops


def _triangular_part(target, given, outs):
    x, (out,), r = given['self'], outs, _Region()
    source = torch.view_as_real(x) if x.is_complex() else x
    made = torch.view_as_real(out) if out.is_complex() else out
    diagonal, t = given.get('diagonal', 0), _class(source)
    distance = r('sub', I32, r.index(out.dim() - 1), r.index(out.dim() - 2))
    keep = r('ge' if target.overloadpacket is A.triu else 'le', BOOL, distance, r.const(diagonal, I32))
    r.done(r('select', t, keep, r.arg(0, source), r.const(0, t)))
    return [_Generic(made.shape, [(source, _same(source.dim()))], [(made, _same(made.dim()))], r)]


ORTHOGONAL = (A.linalg_qr, A.geqrf, A.linalg_householder_product, A.ormqr)


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


@torch.library.register_fake('aten::linalg_lstsq')
def _lstsq_metadata(self, b, rcond=None, *, driver=None):
    torch._check(self.dim() >= 2 and b.dim() >= 1,
                 lambda: 'torch.linalg.lstsq: input must have at least 2 dimensions and other at least 1')
    torch._check(0 <= self.dim() - b.dim() <= 1,
                 lambda: 'torch.linalg.lstsq: input.dim() - other.dim() must be 0 or 1')
    torch._check(self.dtype == b.dtype and self.device == b.device,
                 lambda: 'torch.linalg.lstsq: input and other must have the same dtype and device')
    selected = _lstsq_driver(self, driver)
    torch._check(selected in ('gels', 'gelsy', 'gelsd', 'gelss'),
                 lambda: 'torch.linalg.lstsq: driver must be gels, gelsy, gelsd or gelss')
    torch._check(self.device.type != 'cuda' or selected == 'gels',
                 lambda: 'torch.linalg.lstsq: driver other than gels is not supported on CUDA')
    vector = _linalg_vector_rhs(self, b)
    rhs = b.unsqueeze(-1) if vector else b
    m, n = self.shape[-2:]
    torch._check(m == rhs.shape[-2], lambda: 'torch.linalg.lstsq: input and other row sizes must match')
    batch = torch.broadcast_shapes(self.shape[:-2], rhs.shape[:-2])
    nrhs = rhs.shape[-1]
    solution = self.new_empty((*batch, nrhs, max(m, n))).transpose(-2, -1)[..., :n, :]
    if vector:
        solution = solution.squeeze(-1)
    residual = self.real.new_empty((*batch, nrhs) if m > n and (selected == 'gels' or n == 0 and selected != 'gelsy')
                                   else (0,))
    if m > n > 0 and selected in ('gelsd', 'gelss'):
        from torch._subclasses.fake_tensor import DynamicOutputShapeException
        raise DynamicOutputShapeException(A.linalg_lstsq.default)
    rank = self.new_empty(self.shape[:-2] if selected != 'gels' else (0,), dtype=torch.int64)
    singular = self.real.new_empty((*self.shape[:-2], min(m, n)) if selected in ('gelsd', 'gelss') else (0,))
    return solution, residual, rank, singular


# design/torch-source-lowering-change-request.md item 5; metal-microbench metal_ir.h metal_ir_lapack
GESDD, HEEVD, GEEV, GELSD, GELSS = range(5)


class _Lapack:
    def __init__(self, routine, job, batch, inputs, outputs, rcond=0.0):
        self.routine, self.job, self.batch, self.rcond = routine, job, tuple(batch), rcond
        self.inputs, self.outputs = list(inputs), list(outputs)
        self.place, self.node = CPU, 0


def _singular(target, given, outs):
    a, (u, singular, vh) = given['A'], outs
    batch = a.shape[:-2]
    if not a.numel():
        return (_linalg_identity(u) if u.numel() else []) + (_linalg_identity(vh) if vh.numel() else [])
    info = _temporary(batch, torch.int64)
    vectors = [(u, 2), (vh, 2)] if given['compute_uv'] else []
    job = b'N' if not vectors else b'A' if given['full_matrices'] else b'S'
    return [_Lapack(GESDD, job, batch, [(a, 2)], [(singular, 1), *vectors, (info, 0)]), _Check(info, 3, 'linalg.svd')]


def _eigen_hermitian(target, given, outs):
    a, (values, vectors) = given['A'], outs
    batch = a.shape[:-2]
    if not a.numel():
        return []
    info = _temporary(batch, torch.int64)
    job = (b'V' if given['compute_v'] else b'N') + given['UPLO'].upper().encode()
    return [_Lapack(HEEVD, job, batch, [(a, 2)],
                    [(values, 1), *([(vectors, 2)] if given['compute_v'] else []), (info, 0)]),
            _Check(info, 3, 'linalg.eigh')]


# Reference-LAPACK SRC/dgeev.f
def _eigen(target, given, outs):
    a, values = given['self'], outs[0]
    batch, n = a.shape[:-2], a.shape[-1]
    if not a.numel():
        return []
    info = _temporary(batch, torch.int64)
    vectors = outs[1] if len(outs) > 1 else None
    job = b'N' if vectors is None else b'V'
    if a.is_complex():
        return [_Lapack(GEEV, job, batch, [(a, 2)],
                        [(values, 1), *([] if vectors is None else [(vectors, 2)]), (info, 0)]),
                _Check(info, 3, 'linalg.eig')]
    wr, wi = _temporary((*batch, n), a.dtype), _temporary((*batch, n), a.dtype)
    packed = None if vectors is None else _temporary(a.shape, a.dtype)
    ops = [_Lapack(GEEV, job, batch, [(a, 2)], [(wr, 1), (wi, 1), *([] if packed is None else [(packed, 2)]), (info, 0)]),
           _Check(info, 3, 'linalg.eig')]
    ops += _complex('complex', values, wr, wi)
    if packed is not None:
        before, after = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
        ops += _fill(before, 0) + _fill(after, 0)
        if n > 1:
            ops += _copy(before[..., :, 1:], packed[..., :, :-1]) + _copy(after[..., :, :-1], packed[..., :, 1:])
        first, second = _temporary((*batch, 1, n), torch.bool), _temporary((*batch, 1, n), torch.bool)
        ops += _elementwise('gt', first, wi.unsqueeze(-2), 0) + _elementwise('lt', second, wi.unsqueeze(-2), 0)
        real, imaginary = _temporary(a.shape, a.dtype), _temporary(a.shape, a.dtype)
        ops += _elementwise('where', real, second, before, packed)
        ops += _elementwise('neg', imaginary, packed) + _elementwise('where', imaginary, second, imaginary, 0)
        ops += _elementwise('where', imaginary, first, after, imaginary)
        ops += _complex('complex', vectors, real, imaginary)
    return ops


# PyTorch 2.15 documentation, torch.linalg.pinv
def _pseudoinverse(target, given, outs, default_tolerance_dtype=torch.float64):
    a, (out,), ops = given['self'], outs, []
    if not a.numel():
        return _copy(out, a.transpose(-2, -1))
    batch, (m, n), k = a.shape[:-2], a.shape[-2:], min(a.shape[-2:])
    values = _temporary((*batch, k), a.real.dtype)
    if given['hermitian']:
        vectors = _temporary(a.shape, a.dtype)
        ops += _eigen_hermitian(A._linalg_eigh.default, {'A': a, 'UPLO': 'L', 'compute_v': True}, [values, vectors])
        left, right = vectors, _linalg_transpose(vectors, ops, True, True)
    else:
        u, vh = _temporary((*batch, m, k), a.dtype), _temporary((*batch, k, n), a.dtype)
        ops += _singular(A._linalg_svd.default, {'A': a, 'full_matrices': False, 'compute_uv': True}, [u, values, vh])
        left, right = _linalg_transpose(vh, ops, True, True), _linalg_transpose(u, ops, True, True)
    absolute, maximum = _temporary(values.shape, values.dtype), _temporary(batch, values.dtype)
    ops += _elementwise('abs', absolute, values)
    ops += _reduced(A.amax.default, {'self': absolute, 'dim': [-1], 'keepdim': False}, [maximum])
    batch = out.shape[:-2]
    atol, rtol = given.get('atol'), given.get('rtol', given.get('rcond'))
    tolerance_dtype = values.dtype
    for value in (atol, rtol):
        tolerance_dtype = torch.promote_types(tolerance_dtype, value.dtype if isinstance(value, torch.Tensor)
                                               else default_tolerance_dtype)
    atol = 0 if atol is None else atol
    if rtol is None:
        default = max(m, n) * torch.finfo(a.real.dtype).eps
        positive = _temporary(batch, torch.bool)
        rtol = _temporary(batch, tolerance_dtype)
        ops += _elementwise('gt', positive, atol, 0)
        ops += _elementwise('where', rtol, positive, 0, default)
    tolerance = _temporary(batch, tolerance_dtype)
    ops += _elementwise('mul', tolerance, maximum, rtol)
    ops += _elementwise('maximum', tolerance, tolerance, atol)
    keep, reciprocal = _temporary((*batch, k), torch.bool), _temporary((*batch, k), values.dtype)
    ops += _elementwise('gt', keep, absolute, tolerance.unsqueeze(-1))
    ops += _elementwise('where', reciprocal, keep, values, 1)
    ops += _elementwise('reciprocal', reciprocal, reciprocal)
    ops += _elementwise('where', reciprocal, keep, reciprocal, 0)
    scaled = _temporary((*batch, *left.shape[-2:]), left.dtype)
    ops += _elementwise('mul', scaled, left, reciprocal.unsqueeze(-2))
    return ops + _product(A.matmul.default, {'self': scaled, 'other': right}, [out])


def _orthogonal(target, given, outs):
    packet = target.overloadpacket
    if packet is A.geqrf:
        return _linalg_geqrf(given['self'], *outs)
    if packet is A.linalg_householder_product:
        return _linalg_householder(given['input'], given['tau'], outs[0])
    if packet is A.ormqr:
        return _linalg_householder(given['self'], given['input2'], outs[0], given['input3'], given['left'],
                                   given['transpose'])
    a, (q, r) = given['A'], outs
    packed, tau = _temporary(a.shape, a.dtype), _temporary((*a.shape[:-2], min(a.shape[-2:])), a.dtype)
    ops = _linalg_geqrf(a, packed, tau)
    if q.numel():
        ops += _linalg_householder(packed, tau, q)
    return ops + _triangular_part(A.triu.default, {'self': packed[..., :r.shape[-2], :]}, [r])


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


def _matrix_exponential(a, max_squarings):
    if not a.numel():
        return torch.empty_like(a)
    real = a.real.dtype
    if real not in SQUARING:
        raise TypeError('linalg.matrix_exp: A.dtype=%s is not supported.' % (a.dtype,))
    maxnorm, conds, degrees = SQUARING[real]
    norm = a.abs().sum(-2).amax(-1)
    count = (norm / maxnorm).log2().floor().clamp(min=0)
    scaled = a / count.exp2()[..., None, None]

    def combined(terms, made=None):
        for coefficient, matrix in terms:
            made = coefficient * matrix if made is None else made + coefficient * matrix
        return made
    powers = {0: torch.eye(a.shape[-1], dtype=a.dtype, device=a.device), 2: torch.matmul(scaled, scaled)}
    powers[4] = torch.matmul(powers[2], powers[2])
    for degree in range(6, min(max(degrees), 9), 2):
        powers[degree] = torch.matmul(powers[degree - 2], powers[2])
    u, v = None, None
    for j, degree in enumerate(degrees):
        b, parts = PADE[degree], []
        for parity in (1, 0):
            low = [(b[k + parity], powers[k]) for k in range(6 if degree == 13 else degree - 1, -1, -2)]
            if degree == 13:
                high = combined([(b[k + 6 + parity], powers[k]) for k in (6, 4, 2)])
                parts.append(combined(low, torch.matmul(powers[6], high)))
            else:
                parts.append(combined(low))
        parts[0] = torch.matmul(scaled, parts[0])
        if not j:
            u, v = parts
            continue
        below = (norm < conds[j - 1])[..., None, None]
        u, v = torch.where(below, u, parts[0]), torch.where(below, v, parts[1])
    result = torch.linalg.solve_ex(v - u, u + v).result
    for i in range(max_squarings):
        result = torch.where((count > i)[..., None, None], torch.matmul(result, result), result)
    over = count > max_squarings
    A._assert_async.msg(over.logical_not(), 'linalg.matrix_exp')
    return torch.where(over[..., None, None], math.nan, result)


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/tpu/linalg/qdwh.py, qdwh, _qdwh, _use_qr and
# _use_cholesky
def _polar(x, max_iterations=10):
    batch, real = x.shape[:-2], x.real.dtype
    m, n = x.shape[-2:]
    eps = float(torch.finfo(real).eps)
    magnitude = x.abs()
    one_norm, inf_norm = magnitude.sum(-2).amax(-1), magnitude.sum(-1).amax(-1)
    alpha_inverse = torch.where(one_norm == 0, 1, one_norm.rsqrt() * inf_norm.rsqrt())
    current = x * alpha_inverse[..., None, None]
    identity = torch.eye(n, dtype=x.dtype, device=x.device)
    l, tol_l = eps, 10.0 * eps / 2.0
    tol_norm = tol_l ** (1 / 3)
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
    for a_minus_e_by_sqrt_c, sqrt_c, e in qr_coefs:
        q = torch.linalg.qr(torch.cat([sqrt_c * current, identity.expand(*batch, n, n)], -2)).Q
        current = e * current + a_minus_e_by_sqrt_c * torch.matmul(q[..., :m, :], q[..., m:, :].mH)

    # _use_cholesky's step, and _qdwh's test of it: converged unless norm(u - u_prev) > tol_norm
    def cholesky_step(u, a_minus_e, c, e):
        uh = u.mH
        y = torch.linalg.cholesky_ex(c * torch.matmul(uh, u) + identity).L
        z = torch.linalg.solve_triangular(y, uh, upper=False)
        w = torch.linalg.solve_triangular(y.mH, z, upper=True)
        return e * u + a_minus_e * w.mH

    def converges(step, u):
        distance = (step - u).abs()
        return ((distance * distance).sum((-2, -1)).sqrt() > tol_norm).logical_not()
    # without Cholesky coefficients `iterate` returns is_not_converged True; with them, the last step's test
    converged = torch.zeros(batch, dtype=torch.bool, device=x.device)
    for a_minus_e, c, e in chol_coefs:
        step = cholesky_step(current, a_minus_e, c, e)
        converged, current = converges(step, current), step
    # _qdwh cond_fun and body_fun: from k = len(qr_coefs) + len(chol_coefs), while not converged and k < max_iterations,
    # Halley's method, get_chol_params(3, 1, 3) = (3 - 1/3, 3, 1/3), each step tested; a batch member, once its test
    # passes, keeps its iterate (its loop has ended), the others take the step
    for _ in range(len(qr_coefs) + len(chol_coefs), max_iterations):
        step = cholesky_step(current, 3 - 1 / 3, 3, 1 / 3)
        passed = converges(step, current)
        current = torch.where(converged[..., None, None], current, step)
        converged = converged | passed
    u = 1.5 * current - 0.5 * torch.matmul(current, torch.matmul(current.mH, current))
    product = torch.matmul(u.mH, x)
    return u, (product + product.mH) / 2, converged


# G. H. Golub and C. F. Van Loan, Matrix Computations, 3rd ed., section 8.4, Algorithms 8.4.1 and 8.4.3;
# Z. Drmac and K. Veselic, LAPACK Working Note 196, sections 2 and 3; Reference-LAPACK SRC/dgesvj.f
def _linalg_active(app, aqq, magnitude, tolerance, ops):
    left, right = (_temporary(magnitude.shape, magnitude.dtype) for _ in range(2))
    active = _temporary(magnitude.shape, torch.bool)
    ops += _elementwise('abs', left, app) + _elementwise('sqrt', left, left)
    ops += _elementwise('abs', right, aqq) + _elementwise('sqrt', right, right)
    ops += _elementwise('mul', left, left, tolerance)
    ops += _elementwise('mul', left, left, right)
    ops += _elementwise('gt', active, magnitude, left)
    return active


def _linalg_rotation(app, aqq, apq, ops, tolerance=None, active=None):
    batch, real = app.shape, app.dtype
    magnitude, delta = _temporary(batch, real), _temporary(batch, real)
    d, g, scale = (_temporary(batch, real) for _ in range(3))
    ops += _elementwise('abs', magnitude, apq)
    ops += _elementwise('mul', d, aqq, 0.5)
    ops += _elementwise('mul', g, app, 0.5)
    ops += _elementwise('sub', delta, d, g)
    ops += _elementwise('abs', scale, delta)
    ops += _elementwise('maximum', scale, scale, magnitude)
    zero = _temporary(batch, torch.bool)
    ops += _elementwise('eq', zero, scale, 0)
    ops += _elementwise('where', scale, zero, 1, scale)
    ops += _elementwise('div', d, delta, scale)
    ops += _elementwise('div', g, magnitude, scale)
    square, tangent, cosine = (_temporary(batch, real) for _ in range(3))
    ops += _elementwise('mul', square, d, d)
    ops += _elementwise('mul', tangent, g, g)
    ops += _elementwise('add', square, square, tangent)
    ops += _elementwise('sqrt', square, square)
    ops += _elementwise('abs', tangent, d)
    ops += _elementwise('add', square, square, tangent)
    ops += _elementwise('where', square, zero, 1, square)
    ops += _elementwise('div', tangent, g, square)
    ops += _elementwise('lt', zero, delta, 0)
    ops += _elementwise('neg', square, tangent)
    ops += _elementwise('where', tangent, zero, square, tangent)
    if active is None:
        active = _linalg_active(app, aqq, magnitude, tolerance or torch.finfo(real).eps, ops)
    ops += _elementwise('where', tangent, active, tangent, 0)
    ops += _elementwise('mul', cosine, tangent, tangent)
    ops += _elementwise('add', cosine, cosine, 1)
    ops += _elementwise('rsqrt', cosine, cosine)
    sine = _temporary(batch, apq.dtype)
    ops += _elementwise('eq', zero, magnitude, 0)
    ops += _elementwise('where', scale, zero, 1, magnitude)
    ops += _elementwise('div', sine, apq, scale)
    ops += _elementwise('mul', sine, sine, tangent)
    ops += _elementwise('mul', sine, sine, cosine)
    return cosine, sine, tangent, magnitude, active


def _linalg_columns(source, indices, out):
    given = {'self': source, 'dim': -1, 'index': indices.unsqueeze(-2).expand(out.shape)}
    return _lifted(A.gather.default, _taken, given, [out]) if source.is_complex() else \
        _taken(A.gather.default, given, [out])


def _linalg_range_scale(a, small, ops, maximum=None):
    if not min(a.shape[-2:]):
        scale = _temporary(a.shape[:-2], a.real.dtype)
        ops += _fill(scale, 1)
        return scale
    if maximum is None:
        maximum = _linalg_component_maximum(a, [-2, -1], ops)
    scale, desired, safe = (_temporary(a.shape[:-2], a.real.dtype) for _ in range(3))
    zero = _temporary(a.shape[:-2], torch.bool)
    ops += _elementwise('eq', zero, maximum, 0)
    ops += _elementwise('where', safe, zero, 1, maximum)
    ops += _elementwise('maximum', desired, safe, small)
    ops += _elementwise('minimum', desired, desired, 1 / small)
    ops += _elementwise('div', scale, desired, safe)
    return scale


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


# Reference-LAPACK SRC/dgels.f, zgels.f, dgelsy.f and zgelsy.f
def _linalg_least_scaled(a, b, ops):
    limits = torch.finfo(a.real.dtype)
    maximum = _linalg_component_maximum(a, [-2, -1], ops)
    scales = [_linalg_range_scale(a, limits.tiny / limits.eps, ops, maximum),
              _linalg_range_scale(b, limits.tiny / limits.eps, ops)]
    values = [_temporary(x.shape, x.dtype) for x in (a, b)]
    for source, scale, value in zip((a, b), scales, values):
        ops += _elementwise('mul', value, source, scale[..., None, None])
    return *values, *scales, maximum


def _linalg_least_unscale(out, numerator, denominator, ops):
    batch = out.shape[:-2]
    first, second, same = (_temporary(batch, torch.bool) for _ in range(3))
    ratio, multiply, divide = (_temporary(batch, out.real.dtype) for _ in range(3))
    ops += _elementwise('ge', first, numerator, 1) + _elementwise('ge', second, denominator, 1)
    ops += _elementwise('eq', same, first, second)
    ops += _elementwise('div', ratio, numerator, denominator)
    ops += _elementwise('where', multiply, same, ratio, numerator)
    ops += _elementwise('where', divide, same, 1, denominator)
    ops += _elementwise('mul', out, out, multiply[..., None, None])
    ops += _elementwise('div', out, out, divide[..., None, None])


def _linalg_least_qr(a, b, out, residual, maximum, ops):
    m, n = a.shape[-2:]
    source = _linalg_transpose(a, ops, m < n, m < n)
    packed = _temporary(source.shape, source.dtype)
    tau = _temporary((*a.shape[:-2], min(m, n)), a.dtype)
    ops += _linalg_geqrf(source, packed, tau)
    triangle = packed[..., :min(m, n), :]
    zero = _temporary(a.shape[:-2], torch.bool)
    ops += _elementwise('eq', zero, maximum, 0)
    diagonal = triangle.diagonal(dim1=-2, dim2=-1)
    ops += _elementwise('where', diagonal, zero.unsqueeze(-1), 1, diagonal)
    ops += _linalg_nonsingular(diagonal, 1, 'torch.linalg.lstsq (gels)')
    if m >= n:
        transformed = _temporary((*out.shape[:-2], m, out.shape[-1]), out.dtype)
        ops += _linalg_householder(packed, tau, transformed, b, adjoint=True)
        ops += _linalg_triangular(triangle, transformed[..., :n, :], out, True)
        if residual is not None:
            ops += _copy(residual, transformed[..., n:, :])
    else:
        middle = _temporary(out.shape, out.dtype)
        ops += _fill(middle, 0)
        ops += _linalg_triangular(_linalg_transpose(triangle, ops, True, True), b, middle[..., :m, :], False)
        ops += _linalg_householder(packed, tau, out, middle)
    ops += _elementwise('where', out, zero[..., None, None], 0, out)
    if residual is not None:
        ops += _elementwise('where', residual, zero[..., None, None], 0, residual)


# Reference-LAPACK SRC/dlaic1.f and zlaic1.f: the two-dimensional incremental singular-vector problem
def _linalg_condition(packed, j, estimate, vector, largest, ops):
    batch, real = estimate.shape, estimate.dtype
    dot = _temporary((*batch, 1, 1), packed.dtype)
    row = _linalg_transpose(vector[..., :j].unsqueeze(-1), ops, True, True)
    ops += _product(A.matmul.default, {'self': row, 'other': packed[..., :j, j:j + 1]}, [dot])
    alpha, gamma = dot[..., 0, 0], packed[..., j, j]
    absa, absg, scale = (_temporary(batch, real) for _ in range(3))
    zero = _temporary(batch, torch.bool)
    ops += _elementwise('abs', absa, alpha) + _elementwise('abs', absg, gamma)
    ops += _elementwise('maximum', scale, absa, absg) + _elementwise('maximum', scale, scale, estimate)
    ops += _elementwise('eq', zero, scale, 0) + _elementwise('where', scale, zero, 1, scale)
    app, aqq, sest = (_temporary(batch, real) for _ in range(3))
    apq = _temporary(batch, packed.dtype)
    ops += _elementwise('div', absa, absa, scale) + _elementwise('div', absg, absg, scale)
    ops += _elementwise('div', sest, estimate, scale)
    ops += _elementwise('mul', app, sest, sest) + _elementwise('mul', aqq, absa, absa)
    ops += _elementwise('add', app, app, aqq) + _elementwise('mul', aqq, absg, absg)
    ops += _elementwise('div', apq, alpha, scale)
    diagonal = _temporary(batch, packed.dtype)
    ops += _elementwise('div', diagonal, gamma, scale)
    ops += _elementwise('mul', apq, apq, _linalg_transpose(diagonal, ops, False, True))
    active, first = _temporary(batch, torch.bool), _temporary(batch, torch.bool)
    ops += _elementwise('ne', active, apq, 0) + _elementwise('ge', first, app, aqq)
    cosine, sine, tangent, magnitude, _ = _linalg_rotation(app, aqq, apq, ops, active=active)
    delta, upper, lower = (_temporary(batch, real) for _ in range(3))
    ops += _elementwise('mul', delta, tangent, magnitude)
    ops += _elementwise('sub', upper, app, delta) + _elementwise('add', lower, aqq, delta)
    ops += _elementwise('where', upper, first, upper, lower) + _elementwise('sqrt', upper, upper)
    if largest:
        ops += _elementwise('mul', estimate, scale, upper)
    else:
        ops += _elementwise('eq', zero, upper, 0) + _elementwise('where', upper, zero, 1, upper)
        ops += _elementwise('div', lower, absg, upper)
        ops += _elementwise('mul', estimate, estimate, lower)
        ops += _elementwise('logical_not', first, first)
    negative = _temporary(batch, packed.dtype)
    ops += _elementwise('neg', negative, _linalg_transpose(sine, ops, False, True))
    coefficient, last = _temporary(batch, packed.dtype), _temporary(batch, packed.dtype)
    ops += _elementwise('where', coefficient, first, cosine, sine)
    ops += _elementwise('where', last, first, negative, cosine)
    ops += _elementwise('mul', vector[..., :j], vector[..., :j], coefficient.unsqueeze(-1))
    ops += _copy(vector[..., j], last)


def _linalg_least_rank(packed, rank, rcond, ops):
    batch, k = packed.shape[:-2], min(packed.shape[-2:])
    estimates = [_temporary(batch, packed.real.dtype) for _ in range(2)]
    vectors = [_temporary((*batch, k), packed.dtype) for _ in range(2)]
    active = _temporary(batch, torch.bool)
    for estimate, vector in zip(estimates, vectors):
        ops += _elementwise('abs', estimate, packed[..., 0, 0]) + _fill(vector, 0) + _fill(vector[..., 0], 1)
    ops += _elementwise('ne', active, estimates[0], 0) + _copy(rank, active)
    for j in range(1, k):
        for largest, (estimate, vector) in enumerate(zip(estimates, vectors)):
            _linalg_condition(packed, j, estimate, vector, largest, ops)
        threshold, admitted = _temporary(batch, packed.real.dtype), _temporary(batch, torch.bool)
        ops += _elementwise('mul', threshold, estimates[1], rcond)
        ops += _elementwise('le', admitted, threshold, estimates[0])
        ops += _elementwise('logical_and', active, active, admitted)
        increment = _temporary(batch, rank.dtype)
        ops += _copy(increment, active) + _elementwise('add', rank, rank, increment)


# Reference-LAPACK SRC/ztzrzf.f, zlatrz.f and zgelsy.f: RZ followed by the retained reflectors and pivot permutation
def _linalg_least_complete(packed, tau, pivots, rank, b, out, ops):
    batch, (m, n), k = packed.shape[:-2], packed.shape[-2:], min(packed.shape[-2:])
    matrix = _temporary((*batch, k, n), packed.dtype)
    ops += _triangular_part(A.triu.default, {'self': packed[..., :k, :]}, [matrix])
    indices = _temporary((n,), torch.int32)
    ops += _fill(indices, 0, 1)
    reflectors = []
    for j in reversed(range(k)):
        active, tail = _temporary(batch, torch.bool), _temporary((*batch, n - j), torch.bool)
        row, scalar = _temporary((*batch, n - j), packed.dtype), _temporary(batch, packed.dtype)
        beta = _temporary(batch, packed.real.dtype)
        ops += _elementwise('lt', active, j, rank)
        ops += _elementwise('ge', tail, indices[j:], rank.unsqueeze(-1)) + _fill(tail[..., 0], True)
        ops += _elementwise('where', row, tail, matrix[..., j, j:], 0)
        row = _linalg_transpose(row, ops, False, True)
        v = _linalg_reflector(row, scalar, beta, ops)
        ops += _elementwise('where', scalar, active, scalar, 0)
        if j:
            _linalg_reflect(matrix[..., :j, j:], v, scalar, ops, left=False)
        ops += _elementwise('where', matrix[..., j, j], active, beta, matrix[..., j, j])
        reflectors.append((j, v, scalar))
    keep = _temporary((*batch, k), torch.bool)
    square = _temporary((*batch, k, k), torch.bool)
    triangle = _temporary((*batch, k, k), packed.dtype)
    ops += _elementwise('lt', keep, indices[:k], rank.unsqueeze(-1))
    ops += _elementwise('logical_and', square, keep.unsqueeze(-1), keep.unsqueeze(-2))
    ops += _elementwise('where', triangle, square, matrix[..., :k], 0)
    diagonal = triangle.diagonal(dim1=-2, dim2=-1)
    ops += _elementwise('where', diagonal, keep, diagonal, 1)
    transformed = _temporary((*out.shape[:-2], m, out.shape[-1]), out.dtype)
    ops += _linalg_householder(packed, tau, transformed, b, adjoint=True)
    rhs, work = _temporary((*out.shape[:-2], k, out.shape[-1]), out.dtype), _temporary(out.shape, out.dtype)
    ops += _elementwise('where', rhs, keep.unsqueeze(-1), transformed[..., :k, :], 0)
    ops += _fill(work, 0) + _linalg_triangular(triangle, rhs, work[..., :k, :], True)
    for j, v, scalar in reversed(reflectors):
        _linalg_reflect(work[..., j:, :], v, scalar, ops)
    for j in reversed(range(k)):
        work = _linalg_swap(work, pivots[..., j], j, ops)
    ops += _copy(out, work)


def _least_squares(target, given, outs):
    a, b, (solution, residual, rank, singular), ops = given['self'], given['b'], outs, []
    vector = _linalg_vector_rhs(a, b)
    b, out = (b.unsqueeze(-1), solution.unsqueeze(-1)) if vector else (b, solution)
    m, n = a.shape[-2:]
    driver = _lstsq_driver(a, given.get('driver'))
    if not min(m, n) or not b.shape[-1] and driver in ('gels', 'gelsy'):
        ops += _fill(out, 0) + _fill(rank, 0) + _fill(singular, 0)
        if driver == 'gels':
            return ops + _fill(residual, 0)
        raw = b if n == 0 else None
    elif driver in ('gelsd', 'gelss'):
        batch, own = out.shape[:-2], a.shape[:-2]
        if not math.prod(batch):
            return ops
        rcond = given.get('rcond')
        if rcond is None:
            rcond = torch.finfo(a.real.dtype).eps * max(m, n)
        info = _temporary(batch, torch.int64)
        ranks, values = (rank, singular) if own == batch else \
            (_temporary(batch, rank.dtype), _temporary((*batch, min(m, n)), singular.dtype))
        ops.append(_Lapack(GELSD if driver == 'gelsd' else GELSS, b'', batch, [(a, 2), (b, 2)],
                           [(out, 2), (values, 1), (ranks, 0), (info, 0)], rcond))
        ops.append(_Check(info, 3, 'linalg.lstsq'))
        if own != batch:
            first = (0,) * (len(batch) - len(own)) + tuple(slice(0, 1) if size == 1 else slice(None) for size in own)
            ops += _copy(rank, ranks[first]) + _copy(singular, values[first])
        return ops
    else:
        scaled, rhs, numerator, denominator, maximum = _linalg_least_scaled(a, b, ops)
        raw = _temporary((*out.shape[:-2], m - n, out.shape[-1]), out.dtype) if residual.numel() else None
        rcond = given.get('rcond')
        if rcond is None:
            rcond = torch.finfo(a.real.dtype).eps * max(m, n)
        if driver == 'gels':
            _linalg_least_qr(scaled, rhs, out, raw, maximum, ops)
        else:
            packed = _temporary(a.shape, a.dtype)
            tau, pivots = _temporary((*a.shape[:-2], min(m, n)), a.dtype), \
                _temporary((*a.shape[:-2], min(m, n)), torch.int32)
            ops += _linalg_geqrf(scaled, packed, tau, pivots)
            _linalg_least_rank(packed, rank, rcond, ops)
            _linalg_least_complete(packed, tau, pivots, rank, rhs, out, ops)
        _linalg_least_unscale(out, numerator, denominator, ops)
        if raw is not None:
            ops += _elementwise('div', raw, raw, denominator[..., None, None])
    if residual.numel():
        magnitude = _temporary(raw.shape, raw.real.dtype)
        ops += _elementwise('abs', magnitude, raw) + _elementwise('mul', magnitude, magnitude, magnitude)
        ops += _reduced(A.sum.dim_IntList, {'self': magnitude, 'dim': [-2], 'keepdim': False}, [residual])
    return ops


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


def _spatial(value, k):
    for _ in range(3 - k):
        value = value.unsqueeze(2)
    return value


def _affine(r, terms, constant=0):
    made = r.const(constant, I32)
    for x, c in terms:
        made = r('add', I32, made, x if c == 1 else r('mul', I32, x, r.const(c, I32)))
    return made


# PyTorch 2.14 documentation, torch.nn.Conv1d to torch.nn.Conv3d and torch.nn.ConvTranspose1d to
# torch.nn.ConvTranspose3d; PyTorch v2.11.0 aten/src/ATen/native/Convolution.cpp, convolution_same: the loops
# (n, group, output channel of the group, o0, o1, o2 | input channel of the group, k0, k1, k2), the groups' channel
# offsets affine; an input position outside the padded image contributes nothing; complex operands (torch.view_as_real)
# two more loops, the output's component c PARALLEL and the input's q REDUCTION, the weight's component selected by
# (c, q) and negated for (0, 1): (x_r + i x_i)(w_r + i w_i) by its four real products in the one SUM
def _convolving(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    x, w, bias = given['input'], given['weight'], given['bias']
    _typed(target, [], outs, FLOATING + COMPLEXES)
    transposed = given['transposed'] if packet is A.convolution else packet in TRANSPOSED
    k = w.dim() - 2
    stride, dilation, padding = _ints(given['stride'], k), _ints(given['dilation'], k), given['padding']
    lows = (0,) * k if padding == 'valid' else tuple(d * (n - 1) // 2 for d, n in zip(dilation, w.shape[2:])) \
        if padding == 'same' else _ints(padding, k)
    if x.dim() == k + 1:
        x, out = x.unsqueeze(0), out.unsqueeze(0)
    x, w, out = _spatial(x, k), _spatial(w, k), _spatial(out, k)
    stride, dilation, lows = (1,) * (3 - k) + stride, (1,) * (3 - k) + dilation, (0,) * (3 - k) + lows
    groups, wide = given['groups'], out.is_complex()
    loops = 12 if wide else 10
    outputs, inputs = out.shape[1] // groups, x.shape[1] // groups
    sizes = [x.shape[0], groups, outputs, *out.shape[2:], inputs, *w.shape[2:]] + [2, 2] * wide
    t = (F64 if any(v.dtype in (torch.complex128, torch.float64) for v in (out, x, w)) else F32) if wide else \
        _compute(out.dtype, x.dtype, w.dtype, torch.float32)
    acc = out if bias is None else _temporary(out.shape, out.dtype if wide else STORED[t])
    r = _Region()
    o, kk = [r.index(3 + d) for d in range(3)], [r.index(7 + d) for d in range(3)]
    xs = torch.view_as_real(x) if x.is_complex() else x
    part = [{11: 1}] if x.is_complex() else []
    if not transposed and not any(lows):
        xmap = _rows(loops, [0, {1: inputs, 6: 1}, *[{3 + d: stride[d], 7 + d: dilation[d]} for d in range(3)], *part])
        xv, tests = _load(r, 0, xs, t), []
    else:
        tests, at = [], []
        for d in range(3):
            if transposed:
                shifted = _affine(r, [(o[d], 1), (kk[d], -dilation[d])], lows[d])
                tests += [r('ge', BOOL, shifted, r.const(0, I32)),
                          r('eq', BOOL, r('rem', I32, shifted, r.const(stride[d], I32)), r.const(0, I32))]
                inside, clamped = _bounded(r, r('div', I32, shifted, r.const(stride[d], I32)), x.shape[2 + d])
            else:
                inside, clamped = _window(r, o[d], kk[d], stride[d], lows[d], dilation[d], x.shape[2 + d])
            tests.append(inside)
            at.append(clamped)
        ci = _affine(r, [(r.index(1), inputs), (r.index(6), 1)])
        xv = _convert(r, r('extract', _class(xs), r.index(0), ci, *at, *[r.index(11)] * bool(part), integer=0),
                      _class(xs), t)
        xmap = None
    wmap = [{1: inputs, 6: 1}, 2, 7, 8, 9] if transposed else [{1: outputs, 2: 1}, 6, 7, 8, 9]
    operands = [(xs, xmap)]
    if not wide:
        term = r('mul', t, xv, _load(r, 1, w, t))
        operands.append((w, _rows(loops, wmap)))
    else:
        c, q = r.index(10), r.index(11)
        if not x.is_complex():
            xv = r('select', t, r('eq', BOOL, q, r.const(0, I32)), xv, r.const(0, t))
        if w.is_complex():
            parts = _pair(r, operands, w, _rows(loops, wmap), loops, t)
            product = r('mul', t, xv, r('select', t, r('eq', BOOL, c, q), parts[0], parts[1]))
            term = r('select', t, r('lt', BOOL, c, q), r('neg', t, product), product)
        else:
            operands.append((w, _rows(loops, wmap)))
            term = r('select', t, r('eq', BOOL, c, q), r('mul', t, xv, _load(r, 1, w, t)), r.const(0, t))
    r.done(r('select', t, _all(r, tests), term, r.const(0, t)) if tests else term)
    target = torch.view_as_real(acc) if wide else acc
    made = [_Generic(sizes, operands, [(target, _rows(loops, [0, {1: outputs, 2: 1}, 3, 4, 5] + [10] * wide))], r,
                     [PARALLEL] * 6 + [REDUCTION] * 4 + [PARALLEL, REDUCTION] * wide, [SUM], [t],
                     precision=REASSOCIATE | CONTRACT)]
    if bias is None:
        return made
    if wide:
        return made + _complex('add', out, acc, bias.view(bias.shape[0], 1, 1, 1))
    return made + _epilogue(out, acc, t, [(bias, _rows(5, [1]))], lambda r, a, b: r('add', t, a, b))


# an input position o s - low + k dilation of a strided window, its in-range test and clamped index (_bounded)
def _window(r, o, k, stride, low, dilation, size):
    return _bounded(r, _affine(r, [(o, stride), (k, dilation)], -low), size)


# PyTorch v2.11.0 aten/src/ATen/native/Convolution.cpp, convolution_backward: the input's gradient the dual
# convolution of the output's gradient with the same weight (a transposed convolution's dual is the direct one), the
# weight's the reduction over (n, output position) of the output gradient times the input at the window position, the
# bias's the output gradient's sum over (n, spatial)
def _convolution_backward(target, given, outs):
    gy, x, w = given['grad_output'], given['input'], given['weight']
    mask, transposed, groups = given['output_mask'], given['transposed'], given['groups']
    k = w.dim() - 2
    stride, padding, dilation = _ints(given['stride'], k), _ints(given['padding'], k), _ints(given['dilation'], k)
    ops = []
    if mask[0] and outs[0] is not None:
        ops += _convolving(A.convolution.default, {'input': gy, 'weight': w, 'bias': None, 'stride': stride,
                                                   'padding': padding, 'dilation': dilation, 'transposed': not transposed,
                                                   'groups': groups}, [outs[0]])
    if mask[1] and outs[1] is not None:
        xs, ys, ws = (x.unsqueeze(0), gy.unsqueeze(0), w) if x.dim() == k + 1 else (x, gy, w)
        xs, ys, ws = _spatial(xs, k), _spatial(ys, k), _spatial(ws, k)
        s3, d3, l3 = (1,) * (3 - k) + stride, (1,) * (3 - k) + dilation, (0,) * (3 - k) + padding
        # the operand whose positions the window reaches: the input of a direct convolution, the output gradient of a
        # transposed one; the other read at the reduction's own positions
        near, far = (ys, xs) if not transposed else (xs, ys)
        a, b = near.shape[1] // groups, far.shape[1] // groups
        t = _compute(outs[1].dtype, x.dtype, gy.dtype, torch.float32)
        r = _Region()
        tests, at = [], []
        for d in range(3):
            inside, clamped = _window(r, r.index(7 + d), r.index(3 + d), s3[d], l3[d], d3[d], far.shape[2 + d])
            tests.append(inside)
            at.append(clamped)
        channel = _affine(r, [(r.index(0), b), (r.index(2), 1)])
        v = _convert(r, r('extract', _class(far), r.index(6), channel, *at, integer=1), _class(far), t)
        r.done(r('select', t, _all(r, tests), r('mul', t, _load(r, 0, near, t), v), r.const(0, t)))
        sizes = [groups, a, b, *ws.shape[2:], near.shape[0], *near.shape[2:]]
        nmap = _rows(10, [6, {0: a, 1: 1}, 7, 8, 9])
        wmap = _rows(10, [{0: a, 1: 1}, 2, 3, 4, 5])
        ops.append(_Generic(sizes, [(near, nmap), (far, None)], [(ws, wmap)], r, [PARALLEL] * 6 + [REDUCTION] * 4,
                            [SUM], [t], precision=REASSOCIATE | CONTRACT))
    if mask[2] and outs[2] is not None:
        t = _compute(outs[2].dtype, gy.dtype, torch.float32)
        ops.append(_reduction(A.sum, gy, [0] + list(range(2, gy.dim())) if gy.dim() > k + 1 else list(range(1, gy.dim())),
                              outs[2], t))
    return ops


# PyTorch v2.11.0 aten/src/ATen/native/cpu/AvgPoolKernel.cpp, cpu_avg_pool2d_backward: each input position the sum
# over the kernel offsets of the windows reaching it, the output gradient divided by that window's divisor
def _pooled_backward(target, given, outs):
    gy, x, (gx,) = given['grad_output'], given['self'], outs
    _typed(target, [gy, x], outs)
    kernel, stride, lows = _ints(given['kernel_size'], 2), _ints(given['stride'] or given['kernel_size'], 2), \
        _ints(given['padding'], 2)
    override, include = given.get('divisor_override'), given.get('count_include_pad', True)
    lead = x.dim() - 2
    t = _compute(gx.dtype, gy.dtype, torch.float32)
    r = _Region()
    tests, o = [], []
    for d in range(2):
        shifted = _affine(r, [(r.index(lead + d), 1), (r.index(x.dim() + d), -1)], lows[d])
        tests += [r('ge', BOOL, shifted, r.const(0, I32)),
                  r('eq', BOOL, r('rem', I32, shifted, r.const(stride[d], I32)), r.const(0, I32))]
        inside, clamped = _bounded(r, r('div', I32, shifted, r.const(stride[d], I32)), gy.shape[lead + d])
        tests.append(inside)
        o.append(clamped)
    pool, count = r.const(1, I32), r.const(1, I32)
    for d in range(2):
        lo = _affine(r, [(o[d], stride[d])], -lows[d])
        hi = r('add', I32, lo, r.const(kernel[d], I32))
        edge = r.const(x.shape[lead + d] + lows[d], I32)
        hi = r('select', I32, r('lt', BOOL, edge, hi), edge, hi)
        pool = r('mul', I32, pool, r('sub', I32, hi, lo))
        zero, size = r.const(0, I32), r.const(x.shape[lead + d], I32)
        span = r('sub', I32, r('select', I32, r('lt', BOOL, size, hi), size, hi), r('select', I32, r('lt', BOOL, lo, zero), zero, lo))
        count = r('mul', I32, count, r('select', I32, r('lt', BOOL, span, zero), zero, span))
    divisor = r.const(override, t) if override else _convert(r, pool if include else count, I32, t)
    g = _convert(r, r('extract', _class(gy), *[r.index(j) for j in range(lead)], *o, integer=0), _class(gy), t)
    r.done(r('select', t, _all(r, tests), r('div', t, g, divisor), r.const(0, t)))
    return [_Generic([*x.shape, *kernel], [(gy, None)], [(gx, _rows(x.dim() + 2, range(x.dim())))], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * 2, [SUM], [t], precision=REASSOCIATE)]


# PyTorch 2.14 documentation, torch.nn.AdaptiveMaxPool2d and AdaptiveMaxPool3d (aten/src/ATen/native/cpu
# AdaptiveMaxPoolKernel.cpp: `(val > maxval) || std::isnan(val)`, the index flat over the spatial plane): the
# adaptive windows of _adaptive, REGION (value, index) states joined by max pooling's order (_ordered, NaN later), a
# position past its window's end an absent state
def _adaptive_max(target, given, outs):
    (out, indices), x = outs, given['self']
    _typed(target, [x], outs[:1])
    sizes = list(given['output_size'])
    k, t = len(sizes), _class(x)
    lead = x.dim() - k
    spans = [-(-x.shape[lead + d] // sizes[d]) + 1 for d in range(k)]
    r, tests, at = _Region(), [], []
    for d in range(k):
        o, n, m = r.index(lead + d), x.shape[lead + d], sizes[d]
        start = r('div', I32, r('mul', I32, o, r.const(n, I32)), r.const(m, I32))
        end = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, o, r.const(1, I32)), r.const(n, I32)),
                                r.const(m - 1, I32)), r.const(m, I32))
        p = r('add', I32, start, r.index(x.dim() + d))
        tests.append(r('lt', BOOL, p, end))
        at.append(r('select', I32, tests[-1], p, start))
    flat = r.const(0, I32)
    for d in range(k):
        flat = r('add', I32, r('mul', I32, flat, r.const(x.shape[lead + d], I32)), at[d])
    v = r('extract', t, *[r.index(j) for j in range(lead)], *at, integer=0)
    r.done(v, r('select', I32, _all(r, tests), flat, r.const(-1, I32)))
    lowest = -math.inf if _floating(t) else torch.iinfo(x.dtype).min
    states = [(t, lowest), (I32, -1)]
    omap = _rows(x.dim() + k, range(x.dim()))
    return [_Generic([*out.shape, *spans], [(x, None)], [(out, omap), (indices, omap)], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * k, [REGION] * 2, [t, I32], None, states,
                     _joined(states, lambda c, a, b: _ordered(c, a, b, t, True, True)), _identities(states),
                     REASSOCIATE)]


# PyTorch 2.14 documentation, torch.nn.AdaptiveAvgPool3d (aten/src/ATen/native/AdaptiveAveragePooling3d.cpp,
# start_index and end_index): output position o of extent O over input extent I averages [floor(o I / O),
# ceil((o + 1) I / O)), a reduction over the longest such window with the positions past its end left out
def _adaptive(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs)
    sizes = _ints(given['output_size'], 3)
    lead = x.dim() - 3
    t = _compute(out.dtype, x.dtype, torch.float32)
    spans = [-(-x.shape[lead + d] // sizes[d]) + 1 for d in range(3)]

    def bounds(r, d):
        o, n, m = r.index(lead + d), x.shape[lead + d], sizes[d]
        start = r('div', I32, r('mul', I32, o, r.const(n, I32)), r.const(m, I32))
        end = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, o, r.const(1, I32)), r.const(n, I32)),
                                r.const(m - 1, I32)), r.const(m, I32))
        return start, end
    r, tests, at = _Region(), [], []
    for d in range(3):
        start, end = bounds(r, d)
        p = r('add', I32, start, r.index(x.dim() + d))
        tests.append(r('lt', BOOL, p, end))
        at.append(r('select', I32, r('lt', BOOL, p, end), p, start))
    v = _convert(r, r('extract', _class(x), *[r.index(j) for j in range(lead)], *at, integer=0), _class(x), t)
    r.done(r('select', t, _all(r, tests), v, r.const(0, t)))
    acc = _temporary(out.shape, STORED[t])
    made = [_Generic([*out.shape, *spans], [(x, None)], [(acc, _rows(x.dim() + 3, range(x.dim())))], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * 3, [SUM], [t], precision=REASSOCIATE)]

    def averaged(r, total):
        count = r.const(1, I32)
        for d in range(3):
            start, end = bounds(r, d)
            count = r('mul', I32, count, r('sub', I32, end, start))
        return r('div', t, total, _convert(r, count, I32, t))
    return made + _epilogue(out, acc, t, (), averaged)


# aten/src/ATen/native/AdaptiveAveragePooling.cpp, adaptive_avg_pool2d_backward: input position p of extent I receives
# from the outputs o of extent O whose windows contain it, o in [floor(p O / I), ceil((p + 1) O / I)), each its
# gradient over its window's size
def _adaptive_backward(target, given, outs):
    gy, x, (gx,) = given['grad_output'], given['self'], outs
    _typed(target, [gy, x], outs)
    lead = x.dim() - 2
    t = _compute(gx.dtype, gy.dtype, torch.float32)
    spans = [-(-gy.shape[lead + d] // x.shape[lead + d]) + 1 for d in range(2)]
    r, tests, at = _Region(), [], []
    size = r.const(1, I32)
    for d in range(2):
        n, m, p = x.shape[lead + d], gy.shape[lead + d], r.index(lead + d)
        first = r('div', I32, r('mul', I32, p, r.const(m, I32)), r.const(n, I32))
        last = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, p, r.const(1, I32)), r.const(m, I32)),
                                 r.const(n - 1, I32)), r.const(n, I32))
        o = r('add', I32, first, r.index(x.dim() + d))
        inside = r('lt', BOOL, o, last)
        o = r('select', I32, inside, o, first)
        start = r('div', I32, r('mul', I32, o, r.const(n, I32)), r.const(m, I32))
        end = r('div', I32, r('add', I32, r('mul', I32, r('add', I32, o, r.const(1, I32)), r.const(n, I32)),
                                r.const(m - 1, I32)), r.const(m, I32))
        size = r('mul', I32, size, r('sub', I32, end, start))
        tests.append(inside)
        at.append(o)
    g = _convert(r, r('extract', _class(gy), *[r.index(j) for j in range(lead)], *at, integer=0), _class(gy), t)
    r.done(r('select', t, _all(r, tests), r('div', t, g, _convert(r, size, I32, t)), r.const(0, t)))
    return [_Generic([*x.shape, *spans], [(gy, None)], [(gx, _rows(x.dim() + 2, range(x.dim())))], r,
                     [PARALLEL] * x.dim() + [REDUCTION] * 2, [SUM], [t], precision=REASSOCIATE)]


# PyTorch 2.14 documentation, torch.nn.functional.pdist: element t of the condensed distances the pair i < j of the
# upper triangle in row order, t = i n - i (i + 1) / 2 + j - i - 1, i from t by the triangle's root and corrected by
# the row starts; each pair's p-norm distance as cdist's
def _paired(target, given, outs):
    (out,), x, p = outs, given['self'], float(given['p'])
    _typed(target, [x], outs)
    n, t = x.shape[0], _compute(out.dtype, x.dtype, torch.float32)
    r = _Region()
    k = r.index(0)

    def start(i):
        return r('sub', I32, r('mul', I32, i, r.const(n, I32)),
                 r('div', I32, r('mul', I32, i, r('add', I32, i, r.const(1, I32))), r.const(2, I32)))
    root = r('sqrt', F32, r('cast', F32, r('add', I32, r('mul', I32, k, r.const(-8, I32)), r.const(4 * n * (n - 1) - 7, I32))))
    guess = r('sub', I32, r.const(n - 2, I32),
              r('cast', I32, r('floor', F32, r('sub', F32, r('mul', F32, root, r.const(0.5, F32)), r.const(0.5, F32)))))
    guess = r('select', I32, r('lt', BOOL, k, start(guess)), r('sub', I32, guess, r.const(1, I32)), guess)
    nxt = r('add', I32, guess, r.const(1, I32))
    i = r('select', I32, r('ge', BOOL, k, start(nxt)), nxt, guess)
    j = r('add', I32, r('sub', I32, k, start(i)), r('add', I32, i, r.const(1, I32)))
    c = r.index(1)
    diff = r('abs', t, r('sub', t, _convert(r, r('extract', _class(x), i, c, integer=0), _class(x), t),
                         _convert(r, r('extract', _class(x), j, c, integer=0), _class(x), t)))
    states, join, f = _norm(r, diff, p, t)
    return [_Generic([out.shape[0], x.shape[1]], [(x, None)], [(out, _rows(2, [0]))], r, [PARALLEL, REDUCTION],
                     [REGION], [t], None, states, join, f, REASSOCIATE)]


# PyTorch v2.11.0 aten/src/ATen/native/cpu/MaxPoolKernel.cpp, cpu_max_pool; aten/src/ATen/native/cpu/AvgPoolKernel.cpp,
# cpu_avg_pool2d and cpu_avg_pool3d: the loops (lead0, lead1, o0, o1, o2 | k0, k1, k2); the maximum's states its value
# and flat input position, NaN the greatest and the later of NaNs, the window's first element taken
INDEXED = {A.max_pool2d_with_indices: 2, A.max_pool3d_with_indices: 3}


def _pooled(target, given, outs):
    packet, x = target.overloadpacket, given['self']
    k, out, index = {**MAXIMUM, **AVERAGE, **INDEXED}[packet], outs[0], outs[1] if len(outs) > 1 else None
    _typed(target, [x], outs[:1])
    lows, dilation, lead = _ints(given['padding'], k), _ints(given.get('dilation', 1), k), x.dim() - k
    kernel, stride = _ints(given['kernel_size'], k), _ints(given['stride'] or given['kernel_size'], k)

    def shaped(value):
        for _ in range(2 - lead):
            value = value.unsqueeze(0)
        for _ in range(3 - k):
            value = value.unsqueeze(2)
        return value
    xs, os = shaped(x), shaped(out)
    kernel, stride = (1,) * (3 - k) + kernel, (1,) * (3 - k) + stride
    lows, dilation = (0,) * (3 - k) + lows, (1,) * (3 - k) + dilation
    average = packet in AVERAGE
    sizes, t, omap = [*os.shape, *kernel], _compute(out.dtype, x.dtype, torch.float32), _rows(8, range(5))
    r = _Region()
    tests, at = [], []
    for d in range(3):
        inside, clamped = _window(r, r.index(2 + d), r.index(5 + d), stride[d], lows[d], 1 if average else dilation[d],
                                  xs.shape[2 + d])
        tests.append(inside)
        at.append(clamped)
    inside = _all(r, tests)
    v = _convert(r, r('extract', _class(x), r.index(0), r.index(1), *at, integer=0), _class(x), t)
    iterators = [PARALLEL] * 5 + [REDUCTION] * 3
    if not average:
        place = _affine(r, [(_affine(r, [(at[0], xs.shape[3]), (at[1], 1)]), xs.shape[4]), (at[2], 1)])
        states = [(t, -math.inf)] + ([(I32, -1)] if index is not None else [])
        found = [r('select', t, inside, v, r.const(-math.inf, t))]
        if index is not None:
            found.append(r('select', I32, inside, place, r.const(-1, I32)))
        r.done(*found)

        def join(c, a, b):
            if index is None:
                return [c('select', t, c('or', BOOL, c('gt', BOOL, b[0], a[0]), c('isnan', BOOL, b[0])), b[0], a[0])]
            return _ordered(c, a, b, t, True, nan_later=True)
        outputs = [(os, omap)] + ([(shaped(index), omap)] if index is not None else [])
        return [_Generic(sizes, [(xs, None)], outputs, r, iterators, [REGION] * len(outputs), [kind for kind, _ in states],
                         None, states, _joined(states, join), _identities(states), REASSOCIATE)]
    acc = _temporary(os.shape, STORED[t])
    r.done(r('select', t, inside, v, r.const(0, t)))
    made = [_Generic(sizes, [(xs, None)], [(acc, omap)], r, iterators, [SUM], [t], precision=REASSOCIATE)]
    override, include = given.get('divisor_override'), given.get('count_include_pad', True)

    def divided(r, total):
        pool, count = r.const(1, I32), r.const(1, I32)
        for d in range(3):
            lo = _affine(r, [(r.index(2 + d), stride[d])], -lows[d])
            hi = r('add', I32, lo, r.const(kernel[d], I32))
            edge = r.const(xs.shape[2 + d] + lows[d], I32)
            hi = r('select', I32, r('lt', BOOL, edge, hi), edge, hi)
            pool = r('mul', I32, pool, r('sub', I32, hi, lo))
            zero, size = r.const(0, I32), r.const(xs.shape[2 + d], I32)
            lo = r('select', I32, r('lt', BOOL, lo, zero), zero, lo)
            hi = r('select', I32, r('lt', BOOL, size, hi), size, hi)
            span = r('sub', I32, hi, lo)
            count = r('mul', I32, count, r('select', I32, r('lt', BOOL, span, zero), zero, span))
        divisor = r.const(override, t) if override else _convert(r, pool if include else count, I32, t)
        return r('select', t, r('eq', BOOL, count, r.const(0, I32)), r.const(0, t), r('div', t, total, divisor))
    return made + _epilogue(os, acc, t, (), divided)


# PyTorch v2.11.0 aten/src/ATen/native/cpu/PaddingKernel.cpp, ReflectionPad and ReplicationPad;
# aten/src/ATen/native/PadNd.cpp, constant_pad_nd and _pad_circular_symint
MODES = {'constant': 0, 'reflect': 1, 'replicate': 2, 'circular': 3}


def _padded(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    pad, n = list(given['pad']), x.dim()
    low, high, mode = [0] * n, [0] * n, MODES[given.get('mode', 'constant')]
    for j in range(len(pad) // 2):
        low[n - 1 - j], high[n - 1 - j] = pad[2 * j], pad[2 * j + 1]
    r, tests, at = _Region(), [], []
    for d in range(n):
        size = x.shape[d]
        if low[d] == 0 and high[d] == 0:
            at.append(r.index(d))
            continue
        j = _affine(r, [(r.index(d), 1)], -low[d])
        if mode == 0:
            inside, j = _bounded(r, j, size)
            tests.append(inside)
        elif mode == 1:
            j = r('abs', I32, j)
            j = r('select', I32, r('ge', BOOL, j, r.const(size, I32)), r('sub', I32, r.const(2 * (size - 1), I32), j), j)
        elif mode == 2:
            j = _bounded(r, j, size)[1]
        else:
            extent = r.const(max(size - max(-low[d], 0) - max(-high[d], 0), 1), I32)
            shifted = _affine(r, [(r.index(d), 1)], -max(low[d], 0))
            wrapped = r('rem', I32, r('add', I32, r('rem', I32, shifted, extent), extent), extent)
            j = r('add', I32, r.const(max(-low[d], 0), I32), wrapped)
        at.append(j)
    t = _class(x)
    v = r('extract', t, *at, integer=0)
    if tests:
        v = r('select', t, _all(r, tests), v, r.const(given.get('value') or 0, t))
    r.done(_convert(r, v, t, _class(out)))
    return [_Generic(out.shape, [(x, None)], [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.nn.ReflectionPad1d to 3d and torch.nn.ReplicationPad2d and 3d: the padding of
# their last dimensions in the mode they name
PADS = {A.reflection_pad1d: 'reflect', A.reflection_pad2d: 'reflect', A.reflection_pad3d: 'reflect',
        A.replication_pad1d: 'replicate', A.replication_pad2d: 'replicate', A.replication_pad3d: 'replicate'}


def _padding(target, given, outs):
    return _padded(target, {'self': given['self'], 'pad': given['padding'], 'mode': PADS[target.overloadpacket]}, outs)


# PyTorch 2.14 documentation, torch.flip, Tensor.repeat, torch.slice_scatter and torch.select_scatter: a flipped
# dimension's map size - 1 - i; a repeated source's index the result's modulo its extent; a scatter into a slice or a
# selection the base's copy, then the source's into that view of the result
def _flipped(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    dims = {d % max(x.dim(), 1) for d in given['dims']}
    r = _Region()
    r.done(_convert(r, r.arg(0, x), _class(x), _class(out)))
    rows = _rows(x.dim(), [{d: -1, None: x.shape[d] - 1} if d in dims else d for d in range(x.dim())])
    return [_Generic(out.shape, [(x, rows)], [(out, _same(out.dim()))], r)]


def _repeated(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs, REAL)
    lead, r = out.dim() - x.dim(), _Region()
    at = [r('rem', I32, r.index(lead + d), r.const(n, I32)) for d, n in enumerate(x.shape)]
    r.done(_convert(r, r('extract', _class(x), *at, integer=0), _class(x), _class(out)))
    return [_Generic(out.shape, [(x, None)], [(out, _same(out.dim()))], r)]


def _embedded(target, given, outs):
    (out,), packet = outs, target.overloadpacket
    view = out.select(given['dim'], given['index']) if packet is A.select_scatter else \
        A.slice.Tensor(out, given['dim'], given['start'], given['end'], given['step'])
    return _copy(out, given['self']) + _copy(view, given['src'])

# PyTorch 2.14 documentation, torch.full, torch.zeros, torch.ones, torch.empty, torch.arange, torch.scalar_tensor,
# Tensor.new_full, Tensor.new_zeros, Tensor.new_ones, Tensor.new_empty and Tensor.fill_
FILLS = {A.full: 'fill_value', A.full_like: 'fill_value', A.new_full: 'fill_value', A.scalar_tensor: 's',
         A.fill: 'value', A.zeros: 0, A.zeros_like: 0, A.new_zeros: 0, A.zero: 0, A.ones: 1, A.ones_like: 1,
         A.new_ones: 1}
EMPTY = (A.empty, A.empty_like, A.new_empty, A.empty_strided, A.empty_permuted)


def _factory(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    if packet in EMPTY:
        return []
    if packet is A.arange:
        return _fill(out, given.get('start', 0), given.get('step', 1))
    value = given[FILLS[packet]] if isinstance(FILLS[packet], str) else FILLS[packet]
    return _assign(out, value)


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


# PyTorch 2.14 documentation, torch.sum, torch.mean, torch.amax and torch.amin: the free dimensions PARALLEL and the
# reduced REDUCTION; a sum SUM in its arithmetic kind, an extremum the NaN-propagating join of its state, the mean
# its sum's epilogue; `start` ACCUMULATE for a paged part continuing its line's earlier parts
REDUCTIONS = {A.sum: 0, A.mean: 1, A.amax: 2, A.amin: 3}


# the loops of a reduction of x over `dims` (its free dimensions PARALLEL first, then the reduced), x's map and the
# kept result's
def _reducing(x, dims):
    free = [d for d in range(x.dim()) if d not in dims]
    domain = free + list(dims)
    return ([x.shape[d] for d in domain], [PARALLEL] * len(free) + [REDUCTION] * len(dims),
            _rows(len(domain), [domain.index(d) for d in range(x.dim())]), _rows(len(domain), range(len(free))))


def _reduction(packet, x, dims, kept, t, start=IDENTITY):
    r = _Region()
    r.done(_load(r, 0, x, t))
    loops, iterators, xmap, omap = _reducing(x, dims)
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
    kept = acc.squeeze(tuple(dims)) if given.get('keepdim') and dims else acc
    made = [_reduction(packet, x, dims, kept, t)]
    if packet is not A.mean:
        return made
    total = math.prod(x.shape[d] for d in dims)
    return made + _epilogue(out, acc, t, (), lambda r, s: r('div', t, s, r.const(total, t)))


# PyTorch 2.14 documentation, torch.prod, torch.any, torch.all, torch.argmax, torch.argmin, torch.max, torch.min and
# torch.var: the product and the boolean folds as states with their identities; an extremum with its first index, NaN
# the greatest (torch 2.14.0 aten/src/ATen/native/SharedReduceOps.h, LessOrNan/GreaterOrNan); the variance about the
# mean over the reduced count less its correction
FOLDS = (A.prod, A.any, A.all)
EXTREMA = (A.argmax, A.argmin, A.max, A.min)


def _fold(target, given, outs):
    (out,), x, packet = outs[:1], given['self'], target.overloadpacket
    if x.is_complex():
        return _complex_fold(packet, x, out, _dims(given, x), given.get('keepdim'))
    _typed(target, [x], outs, REAL)
    dims = _dims(given, x)
    t = BOOL if packet is not A.prod else _compute(out.dtype, x.dtype)
    kept = out.squeeze(tuple(dims)) if given.get('keepdim') and dims else out
    loops, iterators, xmap, omap = _reducing(x, dims)
    r = _Region()
    r.done(_load(r, 0, x, t))
    states = [(t, 1 if packet is not A.any else 0)]
    name = 'mul' if packet is A.prod else 'or' if packet is A.any else 'and'
    return [_Generic(loops, [(x, xmap)], [(kept, omap)], r, iterators, [REGION], [t], None, states,
                     _joined(states, lambda c, a, b: [c(name, t, a[0], b[0])]), _identities(states), REASSOCIATE)]


# PyTorch 2.14 documentation, torch.prod, torch.any and torch.all of complex input: the product the REGION monoid of
# the (re, im) states under the complex product, each element read at both components by constant rows; a truth an
# element with either component nonzero
def _complex_fold(packet, x, out, dims, keepdim):
    kept = out.squeeze(tuple(dims)) if keepdim and dims else out
    loops, iterators, xmap, omap = _reducing(x, dims)
    t, inputs, r = F64 if x.dtype is torch.complex128 else F32, [], _Region()
    re, im = _pair(r, inputs, x, xmap, len(loops), t)
    if packet is not A.prod:
        r.done(r('or', BOOL, r('ne', BOOL, re, r.const(0, t)), r('ne', BOOL, im, r.const(0, t))))
        states = [(BOOL, int(packet is A.all))]
        name = 'or' if packet is A.any else 'and'
        return [_Generic(loops, inputs, [(kept, omap)], r, iterators, [REGION], [BOOL], None, states,
                         _joined(states, lambda c, a, b: [c(name, BOOL, a[0], b[0])]), _identities(states), REASSOCIATE)]
    r.done(re, im)
    states = [(t, 1), (t, 0)]
    made = torch.view_as_real(kept)
    return [_Generic(loops, inputs, [(made[..., 0], omap), (made[..., 1], omap)], r, iterators, [REGION] * 2, [t, t],
                     None, states, _joined(states, lambda c, a, b: list(_complex_value(c, t, 'mul', a, b, 1, None))),
                     _identities(states), REASSOCIATE)]


def _extreme(target, given, outs):
    packet, x = target.overloadpacket, given['self']
    _typed(target, [x], outs[-1:] if packet in (A.argmax, A.argmin) else outs[:1], REAL)
    if packet in (A.argmax, A.argmin) and x.dtype is torch.bool:
        raise NotImplementedError('%s of torch.bool: PyTorch 2.14 defines no bool form of it ("does not support bool '
                                  'input")' % target)
    larger, dim = packet in (A.argmax, A.max), given.get('dim')
    if not x.dim():
        x = x.view(1)
    d = None if dim is None else dim % x.dim()
    order = list(range(x.dim())) if d is None else _lines(x, d)
    free = 0 if d is None else x.dim() - 1
    t = I32 if x.dtype is torch.bool else _class(x)
    r = _Region()
    r.done(_load(r, 0, x, t), _linear(r, x.shape) if d is None else r.index(x.dim() - 1))
    lowest = -math.inf if _floating(t) else 0 if x.dtype is torch.bool else torch.iinfo(x.dtype).min
    highest = math.inf if _floating(t) else 1 if x.dtype is torch.bool else torch.iinfo(x.dtype).max
    states = [(t, lowest if larger else highest), (I32, -1)]

    def join(c, a, b):
        return _ordered(c, a, b, t, larger)
    omap = _rows(x.dim(), range(free))

    def shaped(o):
        return o.squeeze(d) if given.get('keepdim') and d is not None else o.reshape(()) if d is None else o
    f = _Region()
    value, index = f('arg', t, integer=0), f('arg', I32, integer=1)
    if packet in (A.argmax, A.argmin):
        f.done(index)
        outputs, accumulators = [(shaped(outs[0]), omap)], [I32]
    elif len(outs) == 2:
        f.done(value, index)
        outputs, accumulators = [(shaped(outs[0]), omap), (shaped(outs[1]), omap)], [t, I32]
    else:
        f.done(value)
        outputs, accumulators = [(shaped(outs[0]), omap)], [t]
    return [_Generic([x.shape[j] for j in order], [(x, _rows(x.dim(), [order.index(j) for j in range(x.dim())]))],
                     outputs, r, [PARALLEL] * free + [REDUCTION] * (x.dim() - free), [REGION] * len(outputs),
                     accumulators, None, states, _joined(states, join), f, REASSOCIATE)]

def _variance(target, given, outs):
    (out,), x = outs[:1], given['self']
    _typed(target, [] if x.is_complex() else [x], outs)
    dims, t = _dims(given, x), _compute(out.dtype, *([] if x.is_complex() else [x.dtype]), torch.float32)
    correction = given.get('correction')
    correction = (1 if given.get('unbiased', True) else 0) if correction is None else correction
    total = math.prod(x.shape[d] for d in dims)
    groups = [x.shape[d] for d in range(x.dim()) if d not in dims]
    loops, iterators, xmap, gmap = _reducing(x, dims)
    parts = [torch.view_as_real(x)[..., c] for c in (0, 1)] if x.is_complex() else [x]
    sums, squares = [_temporary(groups, STORED[t]) for _ in parts], _temporary(groups, STORED[t])
    ops = []
    for part, total_ in zip(parts, sums):
        r = _Region()
        r.done(_load(r, 0, part, t))
        ops.append(_Generic(loops, [(part, xmap)], [(total_, gmap)], r, iterators, [SUM], [t], precision=REASSOCIATE))
    r, square = _Region(), None
    for c, part in enumerate(parts):
        deviation = r('sub', t, _load(r, 2 * c, part, t), r('div', t, r.arg(2 * c + 1, sums[c]), r.const(total, t)))
        made = r('mul', t, deviation, deviation)
        square = made if square is None else r('add', t, square, made)
    r.done(square)
    ops.append(_Generic(loops, [pair for part, total_ in zip(parts, sums) for pair in ((part, xmap), (total_, gmap))],
                        [(squares, gmap)], r, iterators, [SUM], [t], precision=REASSOCIATE))
    kept = out.squeeze(tuple(dims)) if given.get('keepdim') and dims else out
    return ops + _epilogue(kept, squares, t, (), lambda r, q: r('div', t, q, r.const(max(total - correction, 0), t)))

# M. Milakov and N. Gimelshein, Online normalizer calculation for softmax, arXiv:1805.02867, 2018, Algorithm 3 and
# section 3.1, (4): the line's state (m, s), each element x contributing (x, 1), joined by (4)
SOFTMAX = {A.softmax: 0, A._softmax: 0, A.log_softmax: 1, A._log_softmax: 1}


def _larger(r, a, b, t):
    return r('select', t, r('or', BOOL, r('isnan', BOOL, a), r('gt', BOOL, a, b)), a, b)


def _online(c, a, b, t):
    (m, s), (other, more) = a, b
    top = _larger(c, m, other, t)
    joined = c('add', t, c('mul', t, s, c('exp', t, c('sub', t, m, top))),
               c('mul', t, more, c('exp', t, c('sub', t, other, top))))
    absent, empty = c('eq', BOOL, other, c.const(-math.inf, t)), c('eq', BOOL, m, c.const(-math.inf, t))
    return [c('select', t, absent, m, c('select', t, empty, other, top)),
            c('select', t, absent, s, c('select', t, empty, more, joined))]


def _lines(x, d):
    return [j for j in range(x.dim()) if j != d] + [d]


def _statistics(x, d, m, s, t, start=IDENTITY):
    order = _lines(x, d)
    states = [(t, -math.inf), (t, 0)]
    r = _Region()
    r.done(_load(r, 0, x, t), r.const(1, t))
    xmap, lmap = _rows(x.dim(), [order.index(j) for j in range(x.dim())]), _rows(x.dim(), range(x.dim() - 1))
    return _Generic([x.shape[j] for j in order], [(x, xmap)], [(m, lmap), (s, lmap)], r,
                    [PARALLEL] * (x.dim() - 1) + [REDUCTION], [REGION] * 2, [t, t], [start] * 2, states,
                    _joined(states, lambda c, a, b: _online(c, a, b, t)), _identities(states), REASSOCIATE)


def _normalizing(x, d, m, s, out, t, log):
    order = _lines(x, d)
    xmap, lmap = _rows(x.dim(), [order.index(j) for j in range(x.dim())]), _rows(x.dim(), range(x.dim() - 1))
    e = _Region()
    shifted = e('sub', t, _load(e, 0, x, t), e.arg(1, m))
    made = e('sub', t, shifted, e('log', t, e.arg(2, s))) if log else \
        e('mul', t, e('exp', t, shifted), e('div', t, e.const(1, t), e.arg(2, s)))
    e.done(_store(e, made, t, out))
    return _Generic([x.shape[j] for j in order], [(x, xmap), (m, lmap), (s, lmap)], [(out, xmap)], e)


def _softened(target, given, outs):
    (out,), x = outs, given['self']
    _typed(target, [x], outs)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, out, d = x.view(1), out.view(1), 0
    lines = [x.shape[j] for j in _lines(x, d)[:-1]]
    t = _compute(out.dtype, x.dtype, torch.float32)
    m, s = _temporary(lines, STORED[t]), _temporary(lines, STORED[t])
    return [_statistics(x, d, m, s, t), _normalizing(x, d, m, s, out, t, SOFTMAX[target.overloadpacket])]


# G. E. Blelloch, Prefix sums and their applications, CMU-CS-90-190, 1990, §1.3, Figures 1.4 and 1.5 (upsweep and
# downsweep): the exclusive scan by an associative operator (+) with identity e of each line of the component
# buffers (lines..., P = 2^m), in place, by the work-efficient tree: for d < m, b[i 2^(d+1) + 2^(d+1) - 1] =
# b[i 2^(d+1) + 2^d - 1] (+) b[i 2^(d+1) + 2^(d+1) - 1]; the root e; for d from m - 1 down, (left, right) =
# (right, right (+) left) of the same pairs (the prefix before the left subtree, then its total); each level one
# structured operation over strided views
def _blelloch(buffers, t, identities, combine):
    buffer, k = buffers[0], len(buffers)
    kinds = list(t) if isinstance(t, (list, tuple)) else [t] * k
    lead, size = buffer.dim() - 1, buffer.shape[-1]
    levels, ops = size.bit_length() - 1, []

    def pair(d, at):
        return _rows(lead + 1, [*range(lead), {lead: 2 << d, None: at}])

    def sweep(d):
        r = _Region()
        left, right = pair(d, (1 << d) - 1), pair(d, (2 << d) - 1)
        inputs = [(b, left) for b in buffers] + [(b, right) for b in buffers]
        return r, [r.arg(q, b) for q, b in enumerate(buffers)], [r.arg(k + q, b) for q, b in enumerate(buffers)], \
            inputs, left, right
    for d in range(levels):
        r, a, b, inputs, left, right = sweep(d)
        r.done(*combine(r, a, b))
        ops.append(_Generic([*buffer.shape[:-1], size >> (d + 1)], inputs, [(x, right) for x in buffers], r))
    r = _Region()
    r.done(*[r.const(v, kind) for v, kind in zip(identities, kinds)])
    ops.append(_Generic([*buffer.shape[:-1]], [], [(x, _rows(lead, [*range(lead), {None: size - 1}])) for x in buffers],
                        r))
    for d in reversed(range(levels)):
        r, a, b, inputs, left, right = sweep(d)
        r.done(*b, *combine(r, b, a))
        ops.append(_Generic([*buffer.shape[:-1], size >> (d + 1)], inputs,
                            [(x, left) for x in buffers] + [(x, right) for x in buffers], r))
    return ops


# the inclusive scan of the components xs of x's lines along d into outs: each component padded with its identity to
# a power-of-two buffer, the exclusive tree scan, each element combined after its prefix; a component None the
# element's position along d (an extremum's index), `t` one kind or each component's
def _scan(xs, d, outs, t, identities, combine):
    x = next(v for v in xs if v is not None)
    kinds = list(t) if isinstance(t, (list, tuple)) else [t] * len(xs)
    order, n = _lines(x, d), x.dim()
    length = x.shape[d]
    size = 1 << max(length - 1, 0).bit_length()
    buffers = [_temporary([*[x.shape[j] for j in order[:-1]], size], STORED[kind]) for kind in kinds]
    tensors = [v for v in xs if v is not None]
    r = _Region()
    inside, at = _bounded(r, r.index(n - 1), length)
    index = [r.index(order.index(j)) if j != d else at for j in range(n)]

    slots = iter(range(len(tensors)))

    def element(v, kind):
        return at if v is None else _convert(r, r('extract', _class(v), *index, integer=next(slots)), _class(v), kind)
    r.done(*[r('select', kind, inside, element(v, kind), r.const(e, kind)) for v, e, kind in zip(xs, identities, kinds)])
    ops = [_Generic(buffers[0].shape, [(v, None) for v in tensors], [(b, _same(n)) for b in buffers], r)]
    ops += _blelloch(buffers, kinds, identities, combine)
    e = _Region()
    own = iter(range(len(buffers), len(buffers) + len(tensors)))
    made = combine(e, [e.arg(q, b) for q, b in enumerate(buffers)],
                   [e.index(n - 1) if v is None else _load(e, next(own), v, kind) for v, kind in zip(xs, kinds)])
    e.done(*[_store(e, m, kind, o) for m, kind, o in zip(made, kinds, outs)])
    xmap = _rows(n, [order.index(j) for j in range(n)])
    return ops + [_Generic([x.shape[j] for j in order], [(b, _same(n)) for b in buffers] + [(v, xmap) for v in tensors],
                           [(o, xmap) for o in outs], e)]


# torch 2.14.0 aten/src/ATen/native/ReduceOpsUtils.h and LogcumsumexpKernel.cpp _log_add_exp_helper: log(e^a + e^b)
# as the larger plus log1p of e^(smaller - larger); equal infinities and NaN by their own cases
def _log_add_exp(r, t, a, b):
    big = r('select', t, r('ge', BOOL, a, b), a, b)
    small = r('select', t, r('ge', BOOL, a, b), b, a)
    sum_ = r('add', t, big, _apply(r, 'log1p', r('exp', t, r('sub', t, small, big)), None, None, 1, t))
    infinite = r('and', BOOL, r('eq', BOOL, r('abs', t, big), r.const(math.inf, t)), r('eq', BOOL, small, small))
    return r('select', t, infinite, big, sum_)


def _complex_log_add_exp(r, t, a, b):
    def c(op, z, w=None):
        return _complex_value(r, t, op, z, w, 1, None)
    wide = r('ge', BOOL, a[0], b[0])
    big = tuple(r('select', t, wide, p, q) for p, q in zip(a, b))
    small = tuple(r('select', t, wide, q, p) for p, q in zip(a, b))
    sum_ = c('add', c('log1p', c('exp', c('sub', small, big))), big)
    infinite = c('log', c('add', c('exp', small), c('exp', big)))
    nan = r('or', BOOL, c('isnan', a), c('isnan', b))
    equal = r('and', BOOL, r('eq', BOOL, small[0], big[0]), r('eq', BOOL, r('abs', t, big[0]), r.const(math.inf, t)))
    below = r('lt', BOOL, big[0], r.const(0, t))
    return tuple(r('select', t, nan, r.const(math.nan, t), r('select', t, equal, r('select', t, below, s, f), m))
                 for s, f, m in zip(small, infinite, sum_))


# G. E. Blelloch, Prefix sums and their applications, CMU-CS-90-190, 1990 (the scan of any associative operator with
# its identity): PyTorch 2.14 documentation, torch.cumsum, torch.cumprod and torch.logcumsumexp; a complex scan the
# monoid of its (re, im) components
SCANS = {A.cumsum: (0, 'add'), A.cumprod: (1, 'mul'), A.logcumsumexp: (-math.inf, None)}


def _scanned(target, given, outs):
    packet, (out,), x = target.overloadpacket, outs, given['self']
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, out, d = x.view(1), out.view(1), 0
    identity, op = SCANS[packet]
    if out.is_complex():
        ops = []
        if not x.is_complex():
            wide = _temporary(x.shape, out.dtype)
            ops, x = _copy(wide, x), wide
        t = F64 if out.dtype is torch.complex128 else F32
        pairs, made = torch.view_as_real(x), torch.view_as_real(out)

        def combine(r, a, b):
            return _complex_log_add_exp(r, t, a, b) if op is None else _complex_value(r, t, op, a, b, 1, None)
        return ops + _scan([pairs[..., 0], pairs[..., 1]], d, [made[..., 0], made[..., 1]], t, [identity, 0], combine)
    _typed(target, [x], outs, REAL)
    t = _compute(out.dtype, x.dtype)
    return _scan([x], d, [out], t, [identity],
                 lambda r, a, b: [_log_add_exp(r, t, a[0], b[0]) if op is None else r(op, t, a[0], b[0])])


# PyTorch 2.14 documentation, torch.cummax and torch.cummin: the running extremum and its index, the scan of the
# (value, index) monoid of the later winning ties and a NaN (torch 2.14.0 aten ReduceOps.cpp cummax_cummin_helper:
# `x >= out || isnan(x)` for cummax, `<=` for cummin), the max of one total order and so associative (Blelloch)
def _cumulative(target, given, outs):
    packet, x, (values, indices) = target.overloadpacket, given['self'], outs
    _typed(target, [x], outs[:1], REAL)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, values, indices, d = x.view(1), values.view(1), indices.view(1), 0
    t, larger = _compute(values.dtype, x.dtype), packet is A.cummax
    worst = (-math.inf if larger else math.inf) if _floating(t) else \
        (torch.iinfo(STORED[t]).min if larger else torch.iinfo(STORED[t]).max) if t != BOOL else int(not larger)

    def combine(c, a, b):
        win = c('ge' if larger else 'le', BOOL, b[0], a[0])
        if _floating(t):
            win = c('or', BOOL, c('isnan', BOOL, b[0]), c('and', BOOL, c('not', BOOL, c('isnan', BOOL, a[0])), win))
        return [c('select', t, win, b[0], a[0]), c('select', I32, win, b[1], a[1])]
    return _scan([x, None], d, [values, indices], [t, I32], [worst, -1], combine)


# PyTorch 2.14 documentation, torch.searchsorted (and torch.bucketize, its decomposition): each value's insertion
# point in its sorted line by binary search (D. E. Knuth, TAOCP vol. 3, 6.2.1 Algorithm B), ceil(log2(m + 1)) halvings
# unrolled in the region, each probe an extraction at the computed midpoint (through `sorter` where given); NaN ordered
# above every number, as torch's sort orders it (a NaN value's point m)
def _searched(target, given, outs):
    (out,), line, x = outs, given['sorted_sequence'], given['self']
    sorter, right = given.get('sorter'), given.get('right', False) or given.get('side') == 'right'
    m = line.shape[-1]
    t = _compute(line.dtype, *([x.dtype] if isinstance(x, torch.Tensor) else []))
    r, inputs = _Region(), [(line, None)]
    if sorter is not None:
        inputs.append((_indexing(sorter), None))
    lead = [r.index(j) for j in range(out.dim() - 1)] if line.dim() > 1 else []
    if isinstance(x, torch.Tensor):
        v = _load(r, len(inputs), x, t)
        inputs.append((x, _same(out.dim())))
    else:
        v = r.const(x, t)

    def less(p, q):
        if not _floating(t):
            return r('lt', BOOL, p, q)
        return r('and', BOOL, r('not', BOOL, r('isnan', BOOL, p)), r('or', BOOL, r('isnan', BOOL, q), r('lt', BOOL, p, q)))
    lo, hi = r.const(0, I32), r.const(m, I32)
    for _ in range(m.bit_length()):
        active = r('lt', BOOL, lo, hi)
        mid = r('select', I32, active, r('div', I32, r('add', I32, lo, hi), r.const(2, I32)), r.const(0, I32))
        at = r('extract', I32, *lead, mid, integer=1) if sorter is not None else mid
        probe = _convert(r, r('extract', _class(line), *lead, at, integer=0), _class(line), t)
        go = r('not', BOOL, less(v, probe)) if right else less(probe, v)
        lo = r('select', I32, r('and', BOOL, active, go), r('add', I32, mid, r.const(1, I32)), lo)
        hi = r('select', I32, r('and', BOOL, active, r('not', BOOL, go)), mid, hi)
    r.done(_store(r, lo, I32, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.sort and torch.topk: the compiler's one sort expansion (metal_ir.h metal_ir_sort,
# K. E. Batcher's bitonic network; NaN last ascending, first descending, ties by index), the first k of the order
class _Sort:
    def __init__(self, x, axis, k, descending, values, indices):
        self.x, self.axis, self.k, self.descending, self.values, self.indices = x, axis, k, descending, values, indices
        self.place, self.node = GPU, 0


def _ranked(target, given, outs):
    (values, indices), x = outs, given['self']
    _typed(target, [x], [values], REAL)
    d = given['dim'] % max(x.dim(), 1)
    if not x.dim():
        x, values, indices, d = x.view(1), values.view(1), indices.view(1), 0
    k, descending = (given['k'], given['largest']) if target.overloadpacket is A.topk else \
        (x.shape[d], given['descending'])
    return [_Sort(x, d, k, bool(descending), values, indices)]


# PyTorch 2.14 documentation, torch.kthvalue, torch.median and torch.mode: order statistics of each line from its
# stable ascending sort (_Sort, NaN last): the k-th entry; the median the entry (n - 1) // 2, a line holding a NaN its
# NaN and the first NaN's index (the max extremum, _extreme); the mode the longest run of equal sorted values (ties the
# smaller value, the first run) and its last occurrence: run starts by a max scan of the boundary marks (_scan),
# lengths at the run ends, their first-index argmax (_extreme), the sorted entry there by extraction
SELECTIONS = (A.kthvalue, A.median, A.nanmedian, A.mode)


def _selected(target, given, outs):
    packet, x, ops = target.overloadpacket, given['self'], []
    _typed(target, [x], outs[:1], REAL)
    if packet in (A.median, A.nanmedian) and 'dim' not in given:
        flat = _temporary([x.numel()], x.dtype)
        ops, x, d, outs = _copy(flat, x), flat, 0, [outs[0].reshape(()), None]
    else:
        d = given.get('dim', -1) % max(x.dim(), 1)
        x = x.view(1) if not x.dim() else x
    shaped = [o.squeeze(d) if o is not None and given.get('keepdim') and o.dim() else o for o in outs]
    t, n = _class(x), x.shape[d]
    sv, si = _temporary(x.shape, x.dtype), _temporary(x.shape, torch.int64)
    ops += [_Sort(x, d, n, False, sv, si)]

    def picked(values, indices, nan=None):
        r, inputs = _Region(), [(values, _same(values.dim())), (indices, _same(values.dim()))]
        made = [_load(r, 0, values, t), _load(r, 1, indices, I64)]
        if nan is not None:
            inputs += [(nan[0], _same(values.dim())), (nan[1], _same(values.dim()))]
            top, where = _load(r, 2, nan[0], t), _load(r, 3, nan[1], I64)
            hit = r('isnan', BOOL, top)
            made = [r('select', t, hit, top, made[0]), r('select', I64, hit, where, made[1])]
        results = [_store(r, made[0], t, shaped[0])] + ([_store(r, made[1], I64, shaped[1])] if shaped[1] is not None
                                                        else [])
        outputs = [(o, _same(values.dim())) for o in shaped if o is not None]
        r.done(*results)
        return [_Generic(values.shape, inputs, outputs, r)]
    if packet is A.kthvalue:
        return ops + picked(sv.select(d, given['k'] - 1), si.select(d, given['k'] - 1))
    if packet is A.nanmedian and _floating(t):
        return ops + _ranked_at(x, d, sv, si, shaped, t)
    if packet in (A.median, A.nanmedian):
        k = (n - 1) // 2
        if not _floating(t):
            return ops + picked(sv.select(d, k), si.select(d, k))
        lines = list(sv.select(d, k).shape)
        top, where = _temporary(lines, x.dtype), _temporary(lines, torch.int64)
        ops += _extreme(A.max.dim, {'self': x, 'dim': d, 'keepdim': False}, [top, where])
        return ops + picked(sv.select(d, k), si.select(d, k), (top, where))
    order, m = _lines(x, d), x.dim()
    marks, starts, counts = (_temporary(x.shape, torch.int32) for _ in range(3))
    smap = _rows(m, [order.index(j) for j in range(m)])

    def neighbour(r, shift):
        inside, at = _bounded(r, r('add', I32, r.index(m - 1), r.const(shift, I32)), n)
        other = r('extract', t, *[r.index(order.index(j)) if j != d else at for j in range(m)], integer=0)
        return r('and', BOOL, inside, r('eq', BOOL, other, r.arg(1, sv)))
    r = _Region()
    r.done(r('select', I32, neighbour(r, -1), r.const(-1, I32), r.index(m - 1)))
    ops.append(_Generic([x.shape[j] for j in order], [(sv, None), (sv, smap)], [(marks, smap)], r))
    ops += _scan([marks], d, [starts], I32, [-1], lambda c, a, b: [c('select', I32, c('ge', BOOL, b[0], a[0]), b[0], a[0])])
    r = _Region()
    length = r('add', I32, r('sub', I32, r.index(m - 1), r.arg(2, starts)), r.const(1, I32))
    r.done(r('select', I32, neighbour(r, 1), r.const(0, I32), length))
    ops.append(_Generic([x.shape[j] for j in order], [(sv, None), (sv, smap), (starts, smap)], [(counts, smap)], r))
    lines = [x.shape[j] for j in order[:-1]]
    best, end = _temporary(lines, torch.int32), _temporary(lines, torch.int64)
    ops += _extreme(A.max.dim, {'self': counts, 'dim': d, 'keepdim': False}, [best, end])
    r = _Region()
    at = _convert(r, r.arg(2, end), I64, I32)
    coords = [r.index(j if j < d else j - 1) if j != d else at for j in range(m)]
    r.done(_store(r, r('extract', t, *coords, integer=0), t, shaped[0]),
           _store(r, _convert(r, r('extract', I32, *coords, integer=1), I32, I64), I64, shaped[1]))
    return ops + [_Generic(shaped[0].shape, [(sv, None), (_indexing(si), None), (end, _same(len(lines)))],
                           [(shaped[0], _same(len(lines))), (shaped[1], _same(len(lines)))], r)]


# PyTorch 2.14 documentation, torch.nanmedian: the lower median of a line's c numbers, the sorted entry
# max((c - 1) // 2, 0) (NaN last, so an all-NaN line its first NaN), c one SUM of the line's numbers, the entry by
# extraction at that computed position
def _ranked_at(x, d, sv, si, shaped, t):
    order, m = _lines(x, d), x.dim()
    lines = [x.shape[j] for j in order[:-1]]
    count = _temporary(lines, torch.int32)
    loops, iterators, xmap, omap = _reducing(x, [d])
    r = _Region()
    r.done(_convert(r, r('not', BOOL, r('isnan', BOOL, _load(r, 0, x, t))), BOOL, I32))
    ops = [_Generic(loops, [(x, xmap)], [(count, omap)], r, iterators, [SUM], [I32])]
    r = _Region()
    half = r('div', I32, r('sub', I32, r.arg(2, count), r.const(1, I32)), r.const(2, I32))
    at = r('select', I32, r('lt', BOOL, half, r.const(0, I32)), r.const(0, I32), half)
    coords = [r.index(j if j < d else j - 1) if j != d else at for j in range(m)]
    results = [_store(r, r('extract', t, *coords, integer=0), t, shaped[0])]
    outputs = [(shaped[0], _same(len(lines)))]
    if shaped[1] is not None:
        results.append(_store(r, _convert(r, r('extract', I32, *coords, integer=1), I32, I64), I64, shaped[1]))
        outputs.append((shaped[1], _same(len(lines))))
    r.done(*results)
    return ops + [_Generic(lines, [(sv, None), (_indexing(si), None), (count, _same(len(lines)))], outputs, r)]


# PyTorch 2.14 documentation, torch.histc (torch 2.14.0 aten SummaryOps.cpp histogramdd and histc: min == max takes
# the data's extremes, still equal widens them by 1; pos = (x - min) / (max - min) bins, max into the last bin, NaN and
# outside [min, max] left out): each element's bin, then the counts by the scatter-add reduction (_scatter), an element
# left out at bin `bins`, which no output position matches
def _histogram(target, given, outs):
    (out,), x, bins = outs, given['self'], given['bins']
    _typed(target, [x], outs)
    t = _compute(out.dtype, x.dtype, torch.float32)
    low, high, ops = float(given['min']), float(given['max']), []
    r, inputs = _Region(), [(x, _same(x.dim()))]
    v = _load(r, 0, x, t)
    if low == high:
        extremes = [_temporary([], STORED[t]) for _ in range(2)]
        every = list(range(x.dim()))
        ops += [_reduction(A.amin, x, every, extremes[0], t), _reduction(A.amax, x, every, extremes[1], t)]
        lo, hi = (_load(r, 1 + j, e, t) for j, e in enumerate(extremes))
        inputs += [(e, _rows(x.dim(), [])) for e in extremes]
    else:
        lo, hi = r.const(low, t), r.const(high, t)
    same = r('eq', BOOL, lo, hi)
    lo = r('select', t, same, r('sub', t, lo, r.const(1, t)), lo)
    hi = r('select', t, same, r('add', t, hi, r.const(1, t)), hi)
    scaled = r('mul', t, r('div', t, r('sub', t, v, lo), r('sub', t, hi, lo)), r.const(bins, t))
    position = _convert(r, r('trunc', t, scaled), t, I32)
    position = r('select', I32, r('eq', BOOL, position, r.const(bins, I32)), r.const(bins - 1, I32), position)
    inside = r('and', BOOL, r('ge', BOOL, v, lo), r('le', BOOL, v, hi))
    r.done(r('select', I32, inside, position, r.const(bins, I32)))
    index = _temporary([x.numel()], torch.int32)
    ops.append(_Generic(x.shape, inputs, [(index.view(x.shape), _same(x.dim()))], r))
    zeros = _temporary([bins], out.dtype)
    ops += _fill(zeros, 0)
    targets = [(index.as_strided([bins, x.numel()], [0, 1], 0), 0, bins, [0], [1])]
    return ops + _scatter(out, zeros, 1.0, targets, [x.numel()], [bins], [0], [0], SCATTERS['sum'], True, checked=False)


# PyTorch 2.14 documentation, torch.nn.LayerNorm, torch.nn.GroupNorm and torch.nn.BatchNorm2d; torch 2.14.0
# torch/_refs/__init__.py, native_layer_norm, native_group_norm and _normalize; torch/_decomp/decompositions.py,
# native_batch_norm_helper: the statistics' groups and the normalized elements both affine loops over x (group norm's
# channel split into (group, channel of the group)); the mean, then the variance about it, two SUM reductions
NORMS = (A.layer_norm, A.native_layer_norm, A.group_norm, A.native_group_norm, A.batch_norm, A.native_batch_norm,
         A._native_batch_norm_legit, A._native_batch_norm_legit_functional, A._native_batch_norm_legit_no_training)


def _flat(value):
    try:
        return value.view(-1)
    except RuntimeError:
        return value.as_strided((value.numel(),), (value.stride(-1) if value.dim() else 1,), value.storage_offset())


def _grouped(value, count):
    n, c = value.shape[0], value.shape[1]
    return value.as_strided((n, count, c // count, *value.shape[2:]),
                            (value.stride(0), value.stride(1) * (c // count), value.stride(1), *value.stride()[2:]),
                            value.storage_offset())


def _normalized(target, given, outs):
    packet, x, out = target.overloadpacket, given['input'], outs[0]
    weight, bias = given.get('weight'), given.get('bias')
    _typed(target, [x, weight, bias, given.get('running_mean'), given.get('running_var')], outs[:1])
    eps, t = float(given['eps']), _compute(out.dtype, x.dtype, torch.float32)
    if packet in (A.layer_norm, A.native_layer_norm):
        lead = x.dim() - len(given['normalized_shape'])
        view, place, group_of = x, out, list(range(lead))
        reduced = channel_of = list(range(lead, x.dim()))
    elif packet in (A.group_norm, A.native_group_norm):
        count = given['num_groups'] if packet is A.group_norm else given['group']
        view, place, group_of, channel_of = _grouped(x, count), _grouped(out, count), [0, 1], [1, 2]
        reduced = list(range(2, view.dim()))
    else:
        view, place, group_of, channel_of = x, out, [1], [1]
        reduced = [0] + list(range(2, x.dim()))
    dims = view.dim()
    groups, affine = [view.shape[j] for j in group_of], [view.shape[j] for j in channel_of]
    stats = [o.view(groups) for o in outs[1:]] if len(outs) == 3 and outs[1].numel() == math.prod(groups) else \
        [_temporary(groups, STORED[t]), _temporary(groups, STORED[t])]
    mean, rstd = stats
    training = bool(given.get('training'))
    known = not training and given.get('running_mean') is not None and packet in (
        A.batch_norm, A.native_batch_norm, A._native_batch_norm_legit, A._native_batch_norm_legit_no_training)
    g, ops = _same(len(groups)), []
    if known:
        rm, rv = _flat(given['running_mean']).view(groups), _flat(given['running_var']).view(groups)
        f = _Region()
        f.done(_store(f, _load(f, 0, rm, t), t, mean),
               _store(f, f('div', t, f.const(1, t), f('sqrt', t, f('add', t, _load(f, 1, rv, t), f.const(eps, t)))),
                      t, rstd))
        ops.append(_Generic(groups, [(rm, g), (rv, g)], [(mean, g), (rstd, g)], f))
    else:
        total = math.prod(view.shape[j] for j in reduced)
        domain = group_of + reduced
        loops, iterators = [view.shape[j] for j in domain], [PARALLEL] * len(group_of) + [REDUCTION] * len(reduced)
        vmap, gmap = _rows(dims, [domain.index(j) for j in range(dims)]), _rows(dims, range(len(group_of)))
        sums, squares = _temporary(groups, STORED[t]), _temporary(groups, STORED[t])
        r = _Region()
        r.done(_load(r, 0, view, t))
        ops.append(_Generic(loops, [(view, vmap)], [(sums, gmap)], r, iterators, [SUM], [t], precision=REASSOCIATE))
        r = _Region()
        deviation = r('sub', t, _load(r, 0, view, t), r('div', t, r.arg(1, sums), r.const(total, t)))
        r.done(r('mul', t, deviation, deviation))
        ops.append(_Generic(loops, [(view, vmap), (sums, gmap)], [(squares, gmap)], r, iterators, [SUM], [t],
                            precision=REASSOCIATE))
        f = _Region()
        variance = f('div', t, f.arg(1, squares), f.const(total, t))
        f.done(_store(f, f('div', t, f.arg(0, sums), f.const(total, t)), t, mean),
               _store(f, f('div', t, f.const(1, t), f('sqrt', t, f('add', t, variance, f.const(eps, t)))), t, rstd))
        ops.append(_Generic(groups, [(sums, g), (squares, g)], [(mean, g), (rstd, g)], f))
        if training and given.get('running_mean') is not None:
            # torch 2.14.0 torch/_decomp/decompositions.py, native_batch_norm_helper: the running statistics' update,
            # into the functional variant's results or the running statistics' own storage
            rm, rv = _flat(given['running_mean']).view(groups), _flat(given['running_var']).view(groups)
            updated = [o.view(groups) for o in outs[3:5]] if len(outs) == 5 else [rm, rv]
            momentum = float(given['momentum'])
            u = _Region()

            def moved(old, new):
                return u('add', t, u('mul', t, u.const(1 - momentum, t), old), u('mul', t, u.const(momentum, t), new))
            unbiased = u('div', t, u.arg(1, squares), u.const(max(total - 1, 1), t))
            u.done(_store(u, moved(_load(u, 2, rm, t), u('div', t, u.arg(0, sums), u.const(total, t))), t, updated[0]),
                   _store(u, moved(_load(u, 3, rv, t), unbiased), t, updated[1]))
            ops.append(_Generic(groups, [(sums, g), (squares, g), (rm, g), (rv, g)], [(updated[0], g), (updated[1], g)],
                                u))
    e = _Region()
    y = e('mul', t, e('sub', t, _load(e, 0, view, t), _load(e, 1, mean, t)), _load(e, 2, rstd, t))
    inputs = [(view, _same(dims)), (mean, _rows(dims, group_of)), (rstd, _rows(dims, group_of))]
    for affine_value in (weight, bias):
        if affine_value is not None:
            shaped = _flat(affine_value).view(affine)
            y = e('mul' if affine_value is weight else 'add', t, y, _load(e, len(inputs), shaped, t))
            inputs.append((shaped, _rows(dims, channel_of)))
    e.done(_store(e, y, t, out))
    return ops + [_Generic(view.shape, inputs, [(place, _same(dims))], e)]


# an index tensor's elements as region indices: int32 as stored, int64 as its low 32-bit word (little-endian; equal on
# every index of a 32-bit extent); the value the builder declares is that typed view of the same storage
class _Words:
    def __init__(self, tensor):
        self.tensor, self.dtype, self.shape = tensor, torch.int32, tuple(tensor.shape)

    def dim(self):
        return self.tensor.dim()


def _indexing(value):
    return _Words(value) if value.dtype is torch.int64 else value


def _wrapped(r, k, index, extent):
    c = _convert(r, r.arg(k, index), _class(index), I32)
    return r('select', I32, r('lt', BOOL, c, r.const(0, I32)), r('add', I32, c, r.const(extent, I32)), c)


# PyTorch 2.14 documentation, torch.gather, torch.index_select, torch.nn.functional.embedding and Tensor Indexing API;
# NumPy User Guide, Indexing on ndarrays, Advanced indexing: each source dimension's index a result coordinate or an
# index tensor's element there (its view over the result's dimensions), a negative index counted from the end
def _take(out, x, dims, indices):
    r = _Region()
    inputs = [(x, None)]
    for t, strides in indices:
        inputs.append((_indexing(t.as_strided(out.shape, strides, t.storage_offset())), _same(out.dim())))
    at = [r.index(source) if mode < 0 else _wrapped(r, 1 + mode, inputs[1 + mode][0], x.shape[d])
          for d, (mode, source) in enumerate(dims)]
    r.done(_convert(r, r('extract', _class(x), *at, integer=0), _class(x), _class(out)))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


def _advanced(x, indices):
    tensors = [(j, t) for j, t in enumerate(indices) if t is not None]
    for _, t in tensors:
        if t.dtype not in (torch.int64, torch.int32):
            raise NotImplementedError('an index of %s: its lowering is unfinished' % (t.dtype,))
    shape = tuple(torch.broadcast_shapes(*[t.shape for _, t in tensors]))
    positions = [j for j, _ in tensors]
    adjacent = positions == list(range(positions[0], positions[-1] + 1))
    start = positions[0] if adjacent else 0
    rest = [j for j in range(x.dim()) if j not in positions]
    places = {j: (j if j < start else j - len(positions) + len(shape)) if adjacent else len(shape) + rest.index(j)
              for j in rest}
    return tensors, shape, start, places


def _taken(target, given, outs):
    packet, (out,) = target.overloadpacket, outs
    _typed(target, [], outs, REAL)
    if packet is A.embedding:
        x, index = given['weight'], given['indices']
        return _take(out, x, [(0, 0), (-1, index.dim())], [(index, [*index.stride(), 0])])
    x = given['self']
    if packet is A.index:
        tensors, shape, start, places = _advanced(x, list(given['indices']))
        modes = {j: k for k, (j, _) in enumerate(tensors)}
        strides = [[0] * start + list(t.expand(shape).stride()) + [0] * (out.dim() - start - len(shape))
                   for _, t in tensors]
        return _take(out, x, [(modes[j], 0) if j in modes else (-1, places[j]) for j in range(x.dim())],
                     [(t, s) for (_, t), s in zip(tensors, strides)])
    d, index = given['dim'] % max(x.dim(), 1), given['index']
    if packet is A.gather:
        return _take(out, x, [(0, 0) if j == d else (-1, j) for j in range(x.dim())], [(index, index.stride())])
    return _take(out, x, [(0, 0) if j == d else (-1, j) for j in range(x.dim())],
                 [(index, [index.stride(0) if index.dim() and j == d else 0 for j in range(out.dim())])])


# docs/provider.md#sorted-indexed-updates
SCATTERS = {'last': 0, 'sum': 1, 'add': 1, 'prod': 2, 'multiply': 2, 'mean': 3, 'amax': 4, 'amin': 5}


def _scatter(out, base, src, targets, candidates, limits, near, far, op, include, alpha=1,
             negative=False, checked=True, message='indexed update: index out of bounds'):
    integral, parts = out.dtype in INTEGRAL, 2 if out.is_complex() else 1
    t = _compute(out.dtype, *([src.dtype] if isinstance(src, torch.Tensor) else [])) if integral else \
        F64 if out.dtype in (torch.float64, torch.complex128) else F32

    def split(x):
        return [torch.view_as_real(x)[..., k] for k in range(2)] if x.is_complex() else [x]

    def combined(r, a, b):
        if op in (1, 3):
            return [r('add', t, x, y) for x, y in zip(a, b)]
        if op == 2:
            return list(_complex_value(r, t, 'mul', a, b, 1, None)) if parts == 2 else [r('mul', t, a[0], b[0])]
        return [_extremum(r, a[0], b[0], t, op == 4)] if op in (4, 5) else b

    n, count = out.dim(), math.prod(candidates)
    indexed = sorted(d for _, d, _, _, _ in targets)
    batches = [d for d in range(n) if d not in indexed and any(across[d] for _, _, _, across, _ in targets)]
    payload = [d for d in range(n) if d not in indexed and d not in batches]
    bs, ps = [limits[d] for d in batches], [limits[d] for d in payload]
    if not count or not math.prod(bs + ps):
        return _assign(out, base)
    keyshape, shape = bs + [count], bs + ps + [count]
    keydims, dims, volume = len(bs), len(shape), math.prod(out.shape[d] for d in indexed)
    keys, valid = _temporary(bs + list(candidates), torch.int64), _temporary(bs + list(candidates), torch.bool)
    r, inputs, terms, tests = _Region(), [], {}, []
    for index, d, extent, across, along in targets:
        view = index.as_strided(keys.shape, [across[j] for j in batches] + list(along), index.storage_offset())
        raw = _load(r, len(inputs), view, I64)
        inputs.append((view, _same(keys.dim())))
        inside = r('and', BOOL, r('ge', BOOL, raw, r.const(-extent if negative else 0, I64)),
                   r('lt', BOOL, raw, r.const(extent, I64)))
        at = r('select', I64, r('lt', BOOL, raw, r.const(0, I64)), r('add', I64, raw, r.const(extent, I64)), raw) \
            if negative else raw
        terms[d], tests = r('select', I64, inside, at, r.const(0, I64)), tests + [inside]
    inside, linear = _all(r, tests), r.const(0, I64)
    for d in indexed:
        linear = r('add', I64, r('mul', I64, linear, r.const(out.shape[d], I64)), terms[d])
    r.done(r('select', I64, inside, linear, r.const(volume, I64)), inside)
    made = [_Generic(keys.shape, inputs, [(keys, _same(keys.dim())), (valid, _same(keys.dim()))], r)]
    if checked:
        every = _temporary([], torch.bool)
        made += _fold(A.all.default, {'self': valid}, [every])
        made += _asserted(A._assert_async.msg, {'self': every, 'assert_msg': message}, [])
    if not out.numel():
        return made + _assign(out, base)
    keys = keys.reshape(keyshape)
    ordered, permutation = _temporary(keyshape, torch.int64), _temporary(keyshape, torch.int64)
    made.append(_Sort(keys, keydims, count, False, ordered, permutation))
    values = [_temporary(shape, STORED[t]) for _ in range(parts)]
    if op == 3:
        values.append(_temporary(shape, torch.int64))
    r = _Region()
    keymap = _rows(dims, [*range(keydims), dims - 1])
    inputs = [(permutation, keymap)]
    coords = _coordinates(r, r.arg(0, permutation), candidates)
    if isinstance(src, torch.Tensor):
        view = src.as_strided(bs + ps + list(candidates), [near[d] for d in batches + payload] + list(far),
                              src.storage_offset())
        results = []
        for component in split(view):
            x = r('extract', _class(component), *[r.index(d) for d in range(dims - 1)], *coords, integer=len(inputs))
            results.append(_convert(r, x, _class(component), t))
            inputs.append((component, None))
        results += [r.const(0, t)] * (parts - len(results))
    else:
        results = [r.const(_component(src, k), t) for k in range(parts)]
    if op == 1 and alpha != 1:
        results = list(_complex_value(r, t, 'mul', results,
                       [r.const(_component(alpha, q), t) for q in (0, 1)], 1, None)) if parts == 2 else \
            [r('mul', t, r.const(alpha, t), results[0])]
    r.done(*results, *([r.const(1, I64)] if op == 3 else []))
    made.append(_Generic(shape, inputs, [(v, _same(dims)) for v in values], r))
    if op:
        flags = _temporary(keyshape, torch.bool)
        r = _Region()
        at = r.index(keydims)
        previous = r('select', I32, r('eq', BOOL, at, r.const(0, I32)), at, r('sub', I32, at, r.const(1, I32)))
        prior = r('extract', I64, *[r.index(d) for d in range(keydims)], previous, integer=0)
        r.done(r('or', BOOL, r('eq', BOOL, at, r.const(0, I32)), r('ne', BOOL, r.arg(0, ordered), prior)))
        made.append(_Generic(keyshape, [(ordered, _same(len(keyshape)))], [(flags, _same(len(keyshape)))], r))
        flags = flags.reshape(bs + [1] * len(ps) + [count]).expand(shape)
        kinds = [t] * parts + ([I64] if op == 3 else []) + [BOOL]
        identity = _identity(A.amax if op == 4 else A.amin, STORED[t]) if op in (4, 5) else int(op == 2)
        identities = [identity] + [0] * (parts - 1) + ([0] if op == 3 else []) + [False]
        scanned = [_temporary(shape, STORED[k]) for k in kinds]

        def join(c, a, b):
            joined = combined(c, a[:parts], b[:parts])
            if op == 3:
                joined += [c('add', I64, a[parts], b[parts])]
            return [c('select', kind, b[-1], y, z) for kind, y, z in zip(kinds[:-1], b[:-1], joined)] + \
                [c('or', BOOL, a[-1], b[-1])]
        made += _scan(values + [flags], dims - 1, scanned, kinds, identities, join)
        values = scanned[:-1]
    made += _assign(out, base)
    e, inputs = _Region(), [(v, _same(dims)) for v in values] + [(ordered, keymap)]
    linear = e.arg(len(values), ordered)
    next_at = e('select', I32, e('eq', BOOL, e.index(dims - 1), e.const(count - 1, I32)), e.index(dims - 1),
                e('add', I32, e.index(dims - 1), e.const(1, I32)))
    next_key = e('extract', I64, *[e.index(d) for d in range(keydims)], next_at, integer=len(values))
    last = e('or', BOOL, e('eq', BOOL, e.index(dims - 1), e.const(count - 1, I32)), e('ne', BOOL, linear, next_key))
    last = e('and', BOOL, last, e('lt', BOOL, linear, e.const(volume, I64)))
    at = {d: e.index(k) for k, d in enumerate(batches + payload)}
    position = e('select', I64, last, linear, e.const(0, I64))
    at.update(zip(indexed, _coordinates(e, position, [out.shape[d] for d in indexed])))
    coords = [at[d] for d in range(n)]
    a, b = [e.arg(k, v) for k, v in enumerate(values[:parts])], []
    for component in split(base):
        b.append(_convert(e, e('extract', _class(component), *coords, integer=len(inputs)), _class(component), t))
        inputs.append((component, None))
    own = include and op != 0
    result = combined(e, b, a) if own else a
    if op == 3:
        total = e('add', I64, e.arg(parts, values[parts]), e.const(int(own), I64))
        result = [e('floordiv' if integral else 'div', t, x, _convert(e, total, I64, t)) for x in result]
    outputs = split(out)
    result = [_store(e, x, t, v) for x, v in zip(result, outputs)]
    for k, x in enumerate(result):
        e.scatter(k, coords, x, last)
    e.done(*result)
    return made + [_Generic(shape, inputs, [(v, None) for v in outputs], e)]


def _scattered(target, given, outs):
    packet, (out,), x = target.overloadpacket, outs, given['self']
    _typed(target, [], outs, REAL + COMPLEXES)
    if packet is A.index_put:
        tensors, shape, start, places = _advanced(x, list(given['indices']))
        made = [x.shape[j] for j in range(x.dim()) if j in places and places[j] < start] + list(shape) + \
            [x.shape[j] for j in range(x.dim()) if j in places and places[j] >= start + len(shape)]
        values = given['values'].expand(made)
        near = [values.stride(places[j]) if j in places else 0 for j in range(x.dim())]
        far = [values.stride(start + i) for i in range(len(shape))]
        targets = [(t, j, x.shape[j], [0] * x.dim(), list(t.expand(shape).stride())) for j, t in tensors]
        return _scatter(out, x, values, targets, shape, list(x.shape), near, far,
                        SCATTERS['sum' if given['accumulate'] else 'last'], True, negative=True, message=str(target))
    d, index = given['dim'] % max(x.dim(), 1), given['index']
    src = given.get('src', given.get('source', given.get('value')))
    if not x.dim():
        x, out, index = x.view(1), out.view(1), index.reshape(-1)
        if isinstance(src, torch.Tensor):
            src = src.reshape(-1)
    if packet in (A.index_add, A.index_copy, A.index_fill, A.index_reduce):
        far = [src.stride(d) if src.dim() else 0] if isinstance(src, torch.Tensor) else [0]
        near = [src.stride(j) if isinstance(src, torch.Tensor) and src.dim() and j != d else 0 for j in range(x.dim())]
        targets = [(index, d, x.shape[d], [0] * x.dim(), [index.stride(0) if index.dim() else 0])]
        op = given['reduce'] if packet is A.index_reduce else 'sum' if packet is A.index_add else 'last'
        return _scatter(out, x, src, targets, [index.numel()], list(x.shape), near, far, SCATTERS[op],
                        given.get('include_self', True), given.get('alpha', 1), negative=packet is A.index_fill,
                        message=str(target))
    reduce = given.get('reduce')
    op = SCATTERS['sum' if packet is A.scatter_add else reduce if reduce is not None else 'last']
    near = [src.stride(j) if isinstance(src, torch.Tensor) and j != d else 0 for j in range(x.dim())]
    far = [src.stride(d) if isinstance(src, torch.Tensor) else 0]
    targets = [(index, d, x.shape[d], [index.stride(j) if j != d else 0 for j in range(x.dim())], [index.stride(d)])]
    return _scatter(out, x, src, targets, [index.shape[d]],
                    [index.shape[j] if j != d else x.shape[j] for j in range(x.dim())], near, far, op,
                    given.get('include_self', True), message=str(target))


# docs/provider.md#fragment-coded-weights
def _fragment(target, given, outs):
    (out,), codes, tiles = outs, given['codes'], given['tiles']
    if not codes.numel():
        return _fill(out, 0)
    fields = [given[k] for k in ('rows', 'columns', 'tiles', 'bands', 'row_scale', 'col_scale')]
    fields += [tiles.view(torch.float16)[..., 1::2], codes]
    r, t = _Region(), _compute(out.dtype, torch.float32)

    def c(value):
        return r.const(value, U32)

    def op(name, a, b):
        return r(name, U32, a, b)

    def read(k, *indices):
        return r('extract', _class(fields[k]), *indices, integer=k)
    row, col = (_convert(r, read(k, r.index(k)), _class(fields[k]), U64) for k in (0, 1))
    rr, cc = (r('cast', U32, r('rem', U64, x, r.const(32, U64))) for x in (row, col))
    tr, tc = (r('div', U64, x, r.const(32, U64)) for x in (row, col))
    word = read(2, tr, tc)
    width = op('and', op('shr', word, c(12)), c(15))
    lane = op('or', op('or', op('and', op('div', rr, c(8)), c(1)), op('shl', op('and', cc, c(3)), c(1))),
              op('or', op('shl', op('and', op('div', rr, c(8)), c(2)), c(2)),
                 op('shl', op('and', cc, c(4)), c(2))))
    pair = op('add', op('mul', op('and', rr, c(6)), c(2)), op('div', cc, c(8)))
    bit = op('mul', pair, width)
    shift, half = op('rem', bit, c(16)), op('mul', op('and', rr, c(1)), c(16))
    base = op('add', op('and', read(3, tr), c(0x0fffffff)), op('and', word, c(0xfff)))
    at = r('add', U64, r('mul', U64, r('cast', U64, op('add', base, op('div', bit, c(16)))), r.const(32, U64)),
           r('cast', U64, lane))
    live = r('ne', BOOL, width, c(0))
    crossing = r('gt', BOOL, op('add', shift, width), c(16))
    low = r.extract_if(7, U32, [at], live, c(0))
    high = r.extract_if(7, U32, [r('add', U64, at, r.const(32, U64))], r('and', BOOL, live, crossing), c(0))
    low, high = (op('and', op('shr', x, half), c(0xffff)) for x in (low, high))
    joined = op('or', low, op('shl', high, c(16)))
    power = op('shl', c(1), width)
    symbol = op('and', op('shr', joined, shift), op('sub', power, c(1)))
    odd = r('sub', I32, r('cast', I32, op('add', op('mul', symbol, c(2)), c(1))), r('cast', I32, power))
    decoded = r('mul', t, r('cast', t, odd), r('mul', t, r.const(.5, t), r('cast', t, read(6, tr, tc))))
    value = r('select', t, live, decoded, r.const(0, t))
    value = r('mul', t, r('mul', t, value, _convert(r, read(4, row), _class(fields[4]), t)),
              _convert(r, read(5, col), _class(fields[5]), t))
    r.done(_store(r, value, t, out))
    return [_Generic(out.shape, [(x, None) for x in fields], [(out, _same(out.dim()))], r)]


# design/unequal-shards-change-request.md item 8; `striped`: the place in the sequence of token j of `rank`
def _striped_place(r, j, counts, rank):
    own, odd = counts[rank], r('add', I32, r('mul', I32, j, r.const(2, I32)), r.const(1, I32))
    at = j
    for other, count in enumerate(counts):
        if other != rank and count:
            at = r('add', I32, at, r('div', I32, r('add', I32, r('mul', I32, odd, r.const(count, I32)),
                                                  r.const(own - (other > rank), I32)), r.const(2 * own or 1, I32)))
    return at


def _striped(out, counts):
    t, ops = _compute(out.dtype), []
    for rank, (first, count) in enumerate(zip(itertools.accumulate(counts, initial=0), counts)):
        if count:
            block, r = out.narrow(0, first, count), _Region()
            r.done(_store(r, _convert(r, _striped_place(r, r.index(0), counts, rank), I32, t), t, block))
            ops.append(_Generic(block.shape, [], [(block, _same(1))], r))
    return ops


# row `index` of a rank's buffer of queries is its token `index - start`, the rank's shard from `start` of its buffer
def _striped_query(r, index, origin):
    counts, rank, start = origin.stripes
    return _striped_place(r, r('sub', I32, index, r.const(start, I32)), counts, rank)


# T. Dao et al., FlashAttention, NeurIPS 2022, Algorithms 1 and 2;
# H. Liu et al., Ring Attention, arXiv:2310.01889, section 3 and Algorithm 1
def _heads(x, group, kv):
    if x.dim() < 3:
        return x.unsqueeze(0).unsqueeze(0)
    heads = x.shape[-3] if kv else x.shape[-3] // group
    shape = (*x.shape[:-3], heads, 1 if kv else group, *x.shape[-2:])
    stride = (*x.stride()[:-3], x.stride(-3) * (1 if kv else group), 0 if kv else x.stride(-3), *x.stride()[-2:])
    return x.as_strided(shape, stride, x.storage_offset())


def _attention(q, k, v, mask, out, causal, scale, group, dropout=0.0, origin=AttentionOrigin(), generator=None,
               site=None):
    length, size, other, value = q.shape[-2], q.shape[-1], k.shape[-2], v.shape[-1]
    t = _compute(q.dtype, k.dtype, v.dtype, torch.float32)
    qs, ks, vs = _heads(q, group, False), _heads(k, group, True), _heads(v, group, True)
    b = len(qs.shape) - 2
    if not other:
        return _fill(out, 0)
    rows = max(1, origin.page // max(k.element_size() * size, v.element_size() * value, 1)) if origin.page else other
    edges = sorted(set([*range(0, other, rows), *origin.cuts, min(origin.key, other), other]))
    blocks = list(zip(edges, edges[1:]))
    first = next((j for j, (lo, _) in enumerate(blocks) if lo == origin.key), 0)
    blocks = blocks[first:] + blocks[:first]
    shape = [*qs.shape[:-2], length]
    score_storage = _temporary([*shape, min(rows, other)], STORED[t])
    m, s, bm, bs, alpha = [_temporary(shape, STORED[t]) for _ in range(5)]
    acc = _temporary([*shape, value], STORED[t])
    drawn = generator.site(site) if dropout else None
    ops = []
    for j, (lo, hi) in enumerate(blocks):
        count = hi - lo
        kb, vb = ks.narrow(-2, lo, count), vs.narrow(-2, lo, count)
        scores = score_storage.narrow(-1, 0, count)
        domain = [*shape, count, size]
        r = _Region()
        r.done(r('mul', t, _load(r, 0, qs, t), _load(r, 1, kb, t)))
        ops.append(_Generic(domain,
            [(qs, _rows(b + 3, [*range(b), b, b + 2])), (kb, _onto(kb, [*range(b), b + 1, b + 2], domain))],
            [(scores, _rows(b + 3, range(b + 2)))], r, [PARALLEL] * (b + 2) + [REDUCTION], [SUM], [t],
            precision=REASSOCIATE | CONTRACT))
        r, inputs = _Region(), [(scores, _same(b + 2))]
        score = r('mul', t, r.arg(0, scores), r.const(scale, t))
        if mask is not None:
            ms = _heads(mask, group, False).narrow(-1, lo, count)
            inputs.append((ms, _onto(ms, list(range(b + 2)), scores.shape)))
            given = r.arg(1, ms)
            score = r('select', t, given, score, r.const(-math.inf, t)) if mask.dtype is torch.bool else \
                r('add', t, score, _convert(r, given, _class(mask), t))
        if causal:
            query = _striped_query(r, r.index(b), origin) if origin.stripes else r.index(b)
            key = r('add', I32, r.index(b + 1), r.const(lo, I32))
            score = r('select', t, r('gt', BOOL, key, query), r.const(-math.inf, t), score)
        r.done(score)
        ops.append(_Generic(scores.shape, inputs, [(scores, _same(b + 2))], r))
        ops.append(_statistics(scores, b + 1, bm if j else m, bs if j else s, t))
        if j:
            r = _Region()
            old, new = [r.arg(d, x) for d, x in enumerate((m, s))], [r.arg(d + 2, x) for d, x in enumerate((bm, bs))]
            top, total = _online(r, old, new, t)
            factor = r('select', t, r('eq', BOOL, old[0], r.const(-math.inf, t)), r.const(0, t),
                       r('exp', t, r('sub', t, old[0], top)))
            r.done(top, total, factor)
            ops.append(_Generic(shape, [(x, _same(b + 1)) for x in (m, s, bm, bs)],
                                [(x, _same(b + 1)) for x in (m, s, alpha)], r))
            ops += _elementwise('mul', acc, acc, alpha.unsqueeze(-1))
        domain = [*shape, value, count]
        inputs = [(scores, _rows(b + 3, [*range(b), b, b + 2])),
                  (vb, _onto(vb, [*range(b), b + 2, b + 1], domain)), (m, _rows(b + 3, range(b + 1)))]
        r = _Region()
        score, top = r.arg(0, scores), r.arg(2, m)
        weight = r('select', t, r('eq', BOOL, top, r.const(-math.inf, t)), r.const(0, t),
                   r('exp', t, r('sub', t, score, top)))
        if dropout:
            at = r.const(0, I32)
            logical = origin.shape or tuple(q.shape)
            offsets = origin.offsets or (0,) * q.dim()
            indices = [r.index(d) for d in range(q.dim() - 2)]
            if q.dim() >= 3:
                indices[-1] = r('add', I32, r('mul', I32, r.index(b - 2), r.const(group, I32)), r.index(b - 1))
            indices = [r('add', I32, index, r.const(offset, I32)) for index, offset in zip(indices, offsets)]
            indices += [_striped_query(r, r.index(b), origin) if origin.stripes else
                        r('add', I32, r.index(b), r.const(offsets[-2], I32)),
                        r('add', I32, r.index(b + 2), r.const(lo, I32))]
            for index, n in zip(indices, (*logical[:-1], other)):
                at = r('add', I32, r('mul', I32, at, r.const(n, I32)), index)
            keep = r('lt', BOOL, _uniform(r, _draws(r, out, inputs, drawn, b + 3, at)[0]), r.const(1 - dropout, F32))
            kept = r('mul', t, weight, r.const(1 / (1 - dropout), t)) if dropout < 1 else r.const(0, t)
            weight = r('select', t, keep, kept, r.const(0, t))
        r.done(r('mul', t, weight, _load(r, 1, vb, t)))
        ops.append(_Generic(domain, inputs, [(acc, _rows(b + 3, range(b + 2)))], r,
                            [PARALLEL] * (b + 2) + [REDUCTION], [SUM], [t], [ACCUMULATE if j else IDENTITY],
                            precision=REASSOCIATE | CONTRACT))
    return ops + _epilogue(_heads(out, group, False), acc, t, [(s, _rows(b + 2, range(b + 1)))],
        lambda r, numerator, total: r('select', t, r('eq', BOOL, total, r.const(0, t)), numerator,
                                      r('div', t, numerator, total)))


def _attending(given):
    q, k, v, mask = given['query'], given['key'], given['value'], given['attn_mask']
    lead = tuple(q.shape[:-2])
    group = q.shape[-3] // k.shape[-3] if given['enable_gqa'] and lead else 1
    kv = (*lead[:-1], lead[-1] // group) if lead else ()
    scale = given['scale'] if given['scale'] is not None else 1 / math.sqrt(q.shape[-1])
    return (q, k.expand(*kv, *k.shape[-2:]), v.expand(*kv, *v.shape[-2:]),
            mask.expand(*lead, q.shape[-2], k.shape[-2]) if mask is not None else None), given['is_causal'], scale, group


def _attended(target, given, outs, origin=AttentionOrigin(), generator=None, site=None):
    _typed(target, [given['query'], given['key'], given['value'], given['attn_mask']], outs)
    operands, causal, scale, group = _attending(given)
    return _attention(*operands, outs[0], causal, scale, group, float(given['dropout_p']), origin, generator, site)


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
# the construction's draws, and their advance once an execution after the program's draws
class Generator:
    def __init__(self, state):
        self.state, self.sites = state, 0

    def site(self, identity=None):
        self.sites += 1
        return self.state, self.sites if identity is None else identity

    def stepped(self):
        return [_stepped(self.state)] if self.sites else []


def _mulhilo(r, a, m):
    def c(v):
        return r.const(v, U32)
    low, high = c(m & 0xFFFF), c(m >> 16)
    al, ah = r('and', U32, a, c(0xFFFF)), r('shr', U32, a, c(16))
    t0, t1, t2, t3 = r('mul', U32, al, low), r('mul', U32, ah, low), r('mul', U32, al, high), r('mul', U32, ah, high)
    middle = r('add', U32, r('add', U32, r('shr', U32, t0, c(16)), r('and', U32, t1, c(0xFFFF))),
               r('and', U32, t2, c(0xFFFF)))
    hi = r('add', U32, r('add', U32, r('add', U32, t3, r('shr', U32, t1, c(16))), r('shr', U32, t2, c(16))),
           r('shr', U32, middle, c(16)))
    return hi, r('mul', U32, a, c(m))


def _philox(r, counter, key):
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round in range(10):
        hi0, lo0 = _mulhilo(r, c0, PHILOX[0])
        hi1, lo1 = _mulhilo(r, c2, PHILOX[1])
        c0, c1, c2, c3 = r('xor', U32, r('xor', U32, hi1, c1), k0), lo1, r('xor', U32, r('xor', U32, hi0, c3), k1), lo0
        if round < 9:
            k0, k1 = r('add', U32, k0, r.const(PHILOX[2], U32)), r('add', U32, k1, r.const(PHILOX[3], U32))
    return c0, c1, c2, c3


# the draws of operation `out`'s elements: the generator's words as inputs (their constant maps), the element's row-major
# index the counter's first word
def _draws(r, out, inputs, drawn, loops=None, element=None):
    state, site = drawn
    for word in range(3):
        inputs.append((state, _rows(out.dim() if loops is None else loops, [{None: word}])))
    k0, k1, execution = [r('bitcast', U32, r.arg(len(inputs) - 3 + word, state)) for word in range(3)]
    element = r('bitcast', U32, _linear(r, out.shape) if element is None else element)
    return _philox(r, (element, execution, r.const(site, U32), r.const(0, U32)), (k0, k1))


def _uniform(r, word):
    return r('mul', F32, r('cast', F32, r('shr', U32, word, r.const(8, U32))), r.const(2.0 ** -24, F32))


# PyTorch 2.14 documentation, torch.rand, torch.randn, torch.randint, torch.bernoulli, Tensor.uniform_ and
# torch.nn.functional.dropout (torch 2.14.0 aten native_dropout: mask = U < 1 - p, out = x mask / (1 - p)); the normal
# draws by G. E. P. Box and M. E. Muller, Ann. Math. Statist. 29(2), 1958
NORMAL = (A.randn, A.randn_like, torch.ops.prims.normal)
DRAWS = (A.rand, A.rand_like, *NORMAL, A.randint, A.randint_like, A.bernoulli, A.uniform, A.native_dropout)


def _random(target, given, outs, generator, site=None):
    packet, out, scale = target.overloadpacket, outs[0], 1
    if out.is_complex() and packet in (A.rand, A.rand_like, *NORMAL, A.uniform):
        out, scale = torch.view_as_real(out), math.sqrt(0.5) if packet in NORMAL else 1
    _typed(target, [], [out], REAL)
    r, inputs = _Region(), []
    if packet is A.native_dropout:
        x, p, train = given['input'], float(given['p']), given['train']
        inputs.append((x, _same(x.dim())))
        v = _load(r, 0, x, F32)
        if train is False:
            r.done(_store(r, v, F32, out), r.const(True, BOOL))
        else:
            keep = r('lt', BOOL, _uniform(r, _draws(r, out, inputs, generator.site(site))[0]), r.const(1 - p, F32))
            scaled = r('mul', F32, v, r.const(1 / (1 - p) if p < 1 else 0, F32))
            r.done(_store(r, r('select', F32, keep, scaled, r.const(0, F32)), F32, out), keep)
        return [_Generic(out.shape, inputs, [(out, _same(out.dim())), (outs[1], _same(out.dim()))], r)]
    if packet is A.bernoulli and given.get('p') is None:
        inputs.append((given['self'], _same(out.dim())))
    words = _draws(r, out, inputs, generator.site(site))
    u = _uniform(r, words[0])
    if packet in NORMAL:
        first = r('add', F32, u, r.const(2.0 ** -24, F32))
        radius = r('sqrt', F32, r('mul', F32, r.const(-2, F32), r('log', F32, first)))
        made = r('mul', F32, radius, r('cos', F32, r('mul', F32, r.const(2 * math.pi, F32), _uniform(r, words[1]))))
        made = made if scale == 1 else r('mul', F32, made, r.const(scale, F32))
        if packet is torch.ops.prims.normal:
            made = r('add', F32, r.const(float(given['mean']), F32), r('mul', F32, r.const(float(given['std']), F32), made))
    elif packet in (A.randint, A.randint_like):
        low, high = given.get('low') or 0, given['high']
        made = r('add', I32, r.const(low, I32), r('bitcast', I32, r('rem', U32, words[0], r.const(high - low, U32))))
        r.done(_store(r, made, I32, out))
        return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]
    elif packet is A.bernoulli:
        p = given.get('p')
        made = r('cast', F32, r('lt', BOOL, u, _load(r, 0, given['self'], F32) if p is None else r.const(p, F32)))
    elif packet is A.uniform:
        low, high = float(given.get('from', 0)), float(given.get('to', 1))
        made = r('add', F32, r.const(low, F32), r('mul', F32, u, r.const(high - low, F32)))
    else:
        made = u
    r.done(_store(r, made, F32, out))
    return [_Generic(out.shape, inputs, [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.nn.RReLU (B. Xu, N. Wang, T. Chen and M. Li, Empirical evaluation of rectified
# activations in convolutional network, arXiv:1505.00853, 2015): training, an element x <= 0 its slope a Philox draw
# a ~ U(lower, upper), out = a x and noise = a, other elements x and 1; evaluation, the slope (lower + upper) / 2, the
# noise passed through (torch 2.14.0 aten rrelu_with_noise_functional)
def _rrelu(target, given, outs, generator, site=None):
    x, noise, lower, upper = given['self'], given['noise'], float(given['lower']), float(given['upper'])
    (out, kept), same = outs, _same(outs[0].dim())
    _typed(target, [x, noise], outs)
    t = _compute(out.dtype, x.dtype, torch.float32)
    r, inputs = _Region(), [(x, same)]
    v = _load(r, 0, x, t)
    if not given['training']:
        made = r('select', t, r('ge', BOOL, v, r.const(0, t)), v, r('mul', t, v, r.const((lower + upper) / 2, t)))
        r.done(_store(r, made, t, out))
        return [_Generic(out.shape, inputs, [(out, same)], r)] + _copy(kept, noise)
    drawn = _convert(r, _uniform(r, _draws(r, out, inputs, generator.site(site))[0]), F32, t)
    slope = r('add', t, r.const(lower, t), r('mul', t, drawn, r.const(upper - lower, t)))
    low = r('le', BOOL, v, r.const(0, t))
    r.done(_store(r, r('select', t, low, r('mul', t, v, slope), v), t, out),
           _store(r, r('select', t, low, slope, r.const(1, t)), t, kept))
    return [_Generic(out.shape, inputs, [(out, same), (kept, same)], r)]


# the generator's advance: its execution word incremented once an execution, after the program's draws
def _stepped(state):
    r = _Region()
    r.done(r('add', I32, r.arg(0, state), r.const(1, I32)))
    return _Generic((1,), [(state, _rows(1, [{None: 2}]))], [(state, _rows(1, [{None: 2}]))], r)


# the inclusive count of the true elements of `mask` at each of its row-major positions, by the tree scan (_blelloch)
def _counted(mask, count):
    n = mask.numel()
    size = 1 << max(n - 1, 0).bit_length()
    buffer = _temporary([size], torch.int32)

    def element(r, j, k):
        return _convert(r, r('extract', _class(mask), *_coordinates(r, j, mask.shape), integer=k), _class(mask), I32)
    r = _Region()
    inside, at = _bounded(r, r.index(0), n)
    r.done(r('select', I32, inside, element(r, at, 0), r.const(0, I32)))
    ops = [_Generic([size], [(mask, None)], [(buffer, _same(1))], r)]
    ops += _blelloch([buffer], I32, [0], lambda r, a, b: [r('add', I32, a[0], b[0])])
    e = _Region()
    e.done(e('add', I32, e.arg(0, buffer), element(e, e.index(0), 1)))
    return ops + [_Generic([n], [(buffer, _same(1)), (mask, None)], [(count, _same(1))], e)]


# PyTorch 2.14 documentation, Tensor.masked_scatter_: position i of the result the source's next element where the
# (broadcast) mask is true, its rank among the true positions the inclusive count less one
def _masked(target, given, outs):
    (out,), x, mask, source = outs, given['self'], given['mask'], given['source']
    _typed(target, [x, source], outs, REAL)
    mask = mask.expand(x.shape)
    count = _temporary([x.numel()], torch.int32)
    ops = _counted(mask, count)
    r = _Region()
    linear = _linear(r, x.shape)
    taken = _convert(r, r.arg(0, mask), _class(mask), BOOL)
    at = r('select', I32, taken, r('sub', I32, r('extract', I32, linear, integer=1), r.const(1, I32)), r.const(0, I32))
    flat = source.reshape(-1) if source.is_contiguous() else source
    moved = _convert(r, r('extract', _class(flat), *(_coordinates(r, at, flat.shape) if flat.dim() > 1 else [at]),
                          integer=3), _class(flat), _class(out))
    r.done(r('select', _class(out), taken, moved, _convert(r, r.arg(2, x), _class(x), _class(out))))
    return ops + [_Generic(out.shape, [(mask, _same(out.dim())), (count, None), (x, _same(out.dim())), (flat, None)],
                           [(out, _same(out.dim()))], r)]


# PyTorch 2.14 documentation, torch.randperm: the ranks of n uniform draws, a uniformly random permutation
def _permuted(target, given, outs, generator, site=None):
    (out,) = outs
    keys = _temporary([given['n']])
    ops = _random(A.rand.default, {'size': [given['n']]}, [keys], generator, site)
    return ops + _ranked(A.sort.default, {'self': keys, 'dim': 0, 'descending': False}, [_temporary([given['n']]), out])


# PyTorch 2.14 documentation, torch.nn.functional.embedding_bag (aten/src/ATen/native/EmbeddingBag.cpp): bag b the
# indices [offsets[b], offsets[b + 1]) (the last to the end), less padding_idx; its sum, mean or maximum (with the
# chosen row's index) of the weight rows, scaled by per-sample weights; offset2bag each index's bag, bag_size each bag's
# count
def _bagged(target, given, outs):
    weight, indices, offsets = given['weight'], given['indices'], given['offsets']
    out, offset2bag, bag_size, max_indices = outs
    mode, last, padding = given['mode'], given['include_last_offset'], given['padding_idx']
    psw = given.get('per_sample_weights')
    bags, n, d = out.shape[0], indices.numel(), weight.shape[1]
    t = _compute(out.dtype, weight.dtype, torch.float32)
    offsets, indices = _indexing(offsets), _indexing(indices.reshape(-1))
    r = _Region()
    b, j = r.index(0), r.index(2)
    lo = _convert(r, r('extract', I32, b, integer=0), I32, I32)
    nxt = r('add', I32, b, r.const(1, I32))
    room = r('lt', BOOL, nxt, r.const(offsets.shape[0], I32))
    hi = r('select', I32, room, r('extract', I32, r('select', I32, room, nxt, b), integer=0), r.const(n, I32))
    row = r('extract', I32, j, integer=1)
    inside = r('and', BOOL, r('and', BOOL, r('le', BOOL, lo, j), r('lt', BOOL, j, hi)),
               r('ne', BOOL, row, r.const(padding, I32)) if padding >= 0 else r.const(True, BOOL))
    v = _convert(r, r('extract', _class(weight), r('select', I32, inside, row, r.const(0, I32)), r.index(1), integer=2),
                 _class(weight), t)
    if psw is not None:
        v = r('mul', t, v, _convert(r, r('extract', _class(psw), j, integer=3), _class(psw), t))
    larger = mode == 2
    r.done(r('select', t, inside, v, r.const(-math.inf if larger else 0, t)), _convert(r, inside, BOOL, I32),
           r('select', I32, inside, row, r.const(-1, I32)), r('select', I32, inside, j, r.const(-1, I32)))
    states = [(t, -math.inf if larger else 0), (I32, 0), (I32, -1), (I32, -1)]

    def join(c, a, x):
        if not larger:
            return [c('add', t, a[0], x[0]), c('add', I32, a[1], x[1]), a[2], a[3]]
        value, position = _ordered(c, (a[0], a[3]), (x[0], x[3]), t, True)
        right = c('ne', BOOL, position, a[3])
        return [value, c('add', I32, a[1], x[1]), c('select', I32, right, x[2], a[2]), position]
    f = _Region()
    total, count, chosen = f('arg', t, integer=0), f('arg', I32, integer=1), f('arg', I32, integer=2)
    value = f('div', t, total, _convert(f, f('select', I32, f('eq', BOOL, count, f.const(0, I32)), f.const(1, I32), count),
                                        I32, t)) if mode == 1 else \
        f('select', t, f('eq', BOOL, count, f.const(0, I32)), f.const(0, t), total)
    sources = [(offsets, None), (indices, None), (weight, None)] + ([(psw.reshape(-1), None)] if psw is not None else [])
    if larger:
        f.done(value, chosen)
        outputs, accumulators = [(out, _rows(3, [0, 1])), (max_indices, _rows(3, [0, 1]))], [t, I32]
    else:
        f.done(value)
        outputs, accumulators = [(out, _rows(3, [0, 1]))], [t]
    ops = [_Generic([bags, d, n], sources, outputs, r, [PARALLEL, PARALLEL, REDUCTION], [REGION] * len(outputs),
                    accumulators, None, states, _joined(states, join), f, REASSOCIATE)]
    q = _Region()
    b, j = q.index(0), q.index(1)
    lo, nxt = q('extract', I32, b, integer=0), q('add', I32, b, q.const(1, I32))
    room = q('lt', BOOL, nxt, q.const(offsets.shape[0], I32))
    hi = q('select', I32, room, q('extract', I32, q('select', I32, room, nxt, b), integer=0), q.const(n, I32))
    row = q('extract', I32, j, integer=1)
    counted = q('and', BOOL, q('le', BOOL, lo, j), q('lt', BOOL, j, hi))
    if padding >= 0:
        counted = q('and', BOOL, counted, q('ne', BOOL, row, q.const(padding, I32)))
    q.done(_convert(q, counted, BOOL, I32))
    ops.append(_Generic([bags, n], [(offsets, None), (indices, None)], [(bag_size, _rows(2, [0]))], q,
                        [PARALLEL, REDUCTION], [SUM], [I32]))
    g = _Region()
    g.done(_convert(g, g('le', BOOL, g('extract', I32, g.index(1), integer=0), g.index(0)), BOOL, I32))
    ops.append(_Generic([n, offsets.shape[0] - int(bool(last))], [(offsets, None)], [(offset2bag, _rows(2, [0]))], g,
                        [PARALLEL, REDUCTION], [SUM], [I32]))
    h = _Region()
    h.done(h('sub', I32, _convert(h, h.arg(0, offset2bag), _class(offset2bag), I32), h.const(1, I32)))
    ops.append(_Generic([n], [(offset2bag, _same(1))], [(offset2bag, _same(1))], h))
    if not larger and max_indices.numel():
        ops += _fill(max_indices, 0)
    return ops


# PyTorch 2.14 documentation, torch.cdist and torch.nn.functional.pdist: a p-norm of coordinate differences, p 0 the
# count of differing coordinates and inf the largest difference; its term, state, join and finalization
def _norm(r, diff, p, t):
    term = _convert(r, r('ne', BOOL, diff, r.const(0, t)), BOOL, t) if p == 0 else diff if p == 1 or math.isinf(p) else \
        r('mul', t, diff, diff) if p == 2 else r('pow', t, diff, r.const(p, t))
    r.done(term)
    states = [(t, 0)]
    f = _Region()
    total = f('arg', t, integer=0)
    f.done(total if p in (0, 1) or math.isinf(p) else f('sqrt', t, total) if p == 2 else
           f('pow', t, total, f.const(1 / p, t)))
    join = (lambda c, a, b: [_extremum(c, a[0], b[0], t, True)]) if math.isinf(p) else \
        (lambda c, a, b: [c('add', t, a[0], b[0])])
    return states, _joined(states, join), f


def _distances(target, given, outs):
    (out,), x1, x2, p = outs, given['x1'], given['x2'], float(given['p'])
    _typed(target, [x1, x2], outs)
    lead, t = out.dim() - 2, _compute(out.dtype, x1.dtype, x2.dtype, torch.float32)
    loops = [*out.shape, x1.shape[-1]]
    r = _Region()
    states, join, f = _norm(r, r('abs', t, r('sub', t, _load(r, 0, x1, t), _load(r, 1, x2, t))), p, t)
    maps = [_onto(x1, [*range(lead), lead, lead + 2], loops), _onto(x2, [*range(lead), lead + 1, lead + 2], loops)]
    return [_Generic(loops, [(x1, maps[0]), (x2, maps[1])], [(out, _same(lead + 3, out.dim()))], r,
                     [PARALLEL] * (lead + 2) + [REDUCTION], [REGION], [t], None, states, join, f, REASSOCIATE)]


# PyTorch 2.14 documentation, torch.cat
def _concatenated(target, given, outs):
    (out,), d = outs, given['dim'] % max(outs[0].dim(), 1)
    ops, at = [], 0
    for t in given['tensors']:
        if t.dim() == 1 and t.shape[0] == 0 and out.dim() != 1:
            continue
        place = out.narrow(d, at, t.shape[d])
        ops += [] if _same_view(place, t) else _copy(place, t)
        at += t.shape[d]
    return ops


def _placed(ops, place, node):
    for g in ops:
        g.place, g.node = place, node


# design/torch-source-lowering-change-request.md item 1; design/compilation-composition-change-request.md, "Complete
# replacement workgroup": each call's structured operations, a view's materialization its copy
SOURCES = {mesh_code.fragment.overloadpacket: _fragment, **dict.fromkeys(POINTWISE, _arithmetic),
           **dict.fromkeys((*COPIES, A.copy, A.copy_), _copying),
           **dict.fromkeys((A.mm, A.bmm, A.matmul, A.linear, A.addmm), _product), A.einsum: _summation,
           **dict.fromkeys(FACTORIZATIONS, _factorized),
           A._linalg_det: _determinant, A._linalg_slogdet: _determinant,
           A.linalg_lu: _unpacked_lu, A.linalg_pinv: _pseudoinverse, A.linalg_lstsq: _least_squares,
           A._linalg_svd: _singular, A._linalg_eigh: _eigen_hermitian,
           **dict.fromkeys((A.linalg_eig, A.linalg_eigvals, A._linalg_eigvals), _eigen),
           A._linalg_check_errors: _checked,
           **dict.fromkeys(ASSERTIONS, _asserted),
           **dict.fromkeys(ORTHOGONAL, _orthogonal), A.triu: _triangular_part, A.tril: _triangular_part,
           **dict.fromkeys((*CONVOLUTION, *TRANSPOSED, A.convolution), _convolving),
           **dict.fromkeys((*MAXIMUM, *AVERAGE, *INDEXED), _pooled), **dict.fromkeys(PAD, _padded),
           **dict.fromkeys((*FILLS, *EMPTY, A.arange), _factory), **dict.fromkeys(REDUCTIONS, _reduced),
           **dict.fromkeys(SOFTMAX, _softened), **dict.fromkeys(SCANS, _scanned), A.sort: _ranked, A.topk: _ranked,
           **dict.fromkeys(NORMS, _normalized),
           **dict.fromkeys((A.gather, A.index_select, A.embedding, A.index), _taken),
           **dict.fromkeys((A.scatter, A.scatter_add, A.scatter_reduce, A.index_put, A.index_add, A.index_copy,
                            A.index_fill, A.index_reduce), _scattered),
           A.scaled_dot_product_attention: _attended, A.cat: _concatenated, **dict.fromkeys(DRAWS, _random),
           **dict.fromkeys(PARAMETERIZED, _parameterized), **dict.fromkeys(FOLDS, _fold),
           **dict.fromkeys(EXTREMA, _extreme), A.var: _variance, **dict.fromkeys(PADS, _padding), A.flip: _flipped,
           A.repeat: _repeated, A.slice_scatter: _embedded, A.select_scatter: _embedded,
           A.convolution_backward: _convolution_backward, A.avg_pool2d_backward: _pooled_backward,
           A._adaptive_avg_pool3d: _adaptive, A.masked_scatter: _masked, A.randperm: _permuted,
           A._embedding_bag: _bagged, A._cdist_forward: _distances, A._adaptive_avg_pool2d_backward: _adaptive_backward,
           A._pdist_forward: _paired, **dict.fromkeys(COMPLEX, _complexed),
           **dict.fromkeys((A._fft_r2c, A._fft_c2r, A._fft_c2c), _transformed),
           A.rrelu_with_noise_functional: _rrelu, A.cummax: _cumulative, A.cummin: _cumulative,
           A.searchsorted: _searched, **dict.fromkeys(SELECTIONS, _selected),
           A.adaptive_max_pool2d: _adaptive_max, A.adaptive_max_pool3d: _adaptive_max, A.histc: _histogram}
# J. K. Salmon et al., SC 2011, §3.1: the lowerings whose operations draw from the construction's Generator
DRAWN = frozenset(packet for packet, lower in SOURCES.items() if lower in (_random, _rrelu, _permuted))
# PyTorch 2.14 documentation, Tensor Views
VIEWS = {A.as_strided, A.detach, A.diagonal, A.expand, A.expand_as, A.movedim, A.narrow, A.permute, A.select,
         A.squeeze, A.transpose, A.t, A.adjoint, A.real, A.imag, A.view_as_real, A.unflatten, A.unfold, A.unsqueeze,
         A.view, A.view_as, A.unbind, A.split, A.split_with_sizes, A.tensor_split, A.hsplit, A.vsplit, A.swapaxes,
         A.swapdims, A.chunk, A.slice, A.alias, A.view_as_complex, operator.getitem}


# PyTorch 2.14 documentation, Complex Numbers (and torch.nan_to_num, each component replaced): an operation linear in
# its data (its indices, masks and shapes real), or acting on each component alone, the same operation on each of the
# (re, im) components, a real scalar datum the real component's alone and a real
# datum's imaginary component zero
LINEAR = {*PAD, *PADS, A.sum, A.mean, A.gather, A.index_select, A.embedding, A.index, A.cat, A.flip, A.repeat,
          A.slice_scatter, A.select_scatter, A.masked_scatter, A.nan_to_num}
VALUED = ('value', 'fill_value', 'src', 'source', 'values', 'self', 'input', 'tensors', 'weight')


def _complexes(given, outs):
    return any(isinstance(v, torch.Tensor) and v.is_complex() or isinstance(v, complex)
               for v in pytree.tree_leaves([given, outs]))


def _lifted(target, lower, given, outs):
    ops = []
    for k in (0, 1):
        def part(v, datum):
            if isinstance(v, (list, tuple)):
                return type(v)(part(x, datum) for x in v)
            if isinstance(v, torch.Tensor) and v.is_complex():
                return torch.view_as_real(v)[..., k]
            if isinstance(v, torch.Tensor) and datum and k:
                zero = _temporary(v.shape, v.dtype)
                ops.extend(_fill(zero, 0))
                return zero
            if isinstance(v, (int, float, complex)) and not isinstance(v, bool) and datum:
                return _component(v, k)
            return v
        made = {name: part(v, name in VALUED) for name, v in given.items()}
        ops += lower(target, made, [part(o, False) for o in outs])
    return ops


# Python's operator module on a held scalar (torch.fx symbolic arithmetic of SymInt/SymBool/SymFloat): the pointwise
# operation of the same semantics (floor division and modulo as Python's, torch.floor_divide and torch.remainder)
SYMBOLIC = {operator.add: 'add', operator.sub: 'sub', operator.mul: 'mul', operator.truediv: 'div',
            operator.floordiv: 'floordiv', operator.mod: 'remainder', operator.neg: 'neg', operator.lt: 'lt',
            operator.le: 'le', operator.gt: 'gt', operator.ge: 'ge', operator.eq: 'eq', operator.ne: 'ne',
            operator.and_: 'logical_and', operator.or_: 'logical_or', operator.not_: 'logical_not',
            torch.sym_not: 'logical_not'}


def _lowering(local, page, generator=None):
    packet = getattr(local.target, 'overloadpacket', local.target)
    if packet is A.scaled_dot_product_attention:
        origin = local.annotation if isinstance(local.annotation, AttentionOrigin) else AttentionOrigin(0, page)
        return lambda args, kwargs, outs: _attended(local.target, _operands(local.target, args, kwargs), outs, origin,
                                                    generator, local.site)
    if packet in DRAWN:
        return lambda args, kwargs, outs: SOURCES[packet](local.target, _operands(local.target, args, kwargs), outs,
                                                          generator, local.site)
    if local.target is striped:
        return lambda args, kwargs, outs: _striped(outs[0], *args)
    if packet in (A._linalg_det, A._linalg_slogdet):
        metadata = _given(local, _names(local))['A'].meta['val']
        transpose = metadata.is_contiguous() and not metadata.is_complex()
        return lambda args, kwargs, outs: _determinant(local.target, _operands(local.target, args, kwargs), outs,
                                                       transpose)
    if packet is A.linalg_pinv:
        metadata = _given(local, _names(local))['self'].meta['val']
        dtype = torch.float32 if metadata.device.type in ('mps', 'metal') else torch.float64
        return lambda args, kwargs, outs: _pseudoinverse(local.target, _operands(local.target, args, kwargs), outs,
                                                        dtype)
    if local.target in SYMBOLIC:
        return lambda args, kwargs, outs: _elementwise(SYMBOLIC[local.target], outs[0], *args)
    if packet in LINEAR:
        def lifted(args, kwargs, outs):
            given = _operands(local.target, args, kwargs)
            lower = SOURCES[packet]
            return (_lifted(local.target, lower, given, outs) if _complexes(given, outs)
                    else lower(local.target, given, outs))
        return lifted
    if packet in SOURCES:
        return lambda args, kwargs, outs: SOURCES[packet](local.target, _operands(local.target, args, kwargs), outs)
    if packet in VIEWS:
        return lambda args, kwargs, outs: _copy(outs[0], _call(local, args, kwargs))
    raise NotImplementedError('%s: its lowering into the structured basis is unfinished '
                              '(design/torch-source-lowering-change-request.md item 1)' % (local.target,))


def _aliases(local):
    return local.target is operator.getitem or isinstance(local.target, torch._ops.OpOverload) \
        and any(r.alias_info is not None for r in local.target._schema.returns)


# design/torch-source-lowering-change-request.md item 1: the storage identity of a tensor of the compilation (its
# placed stand-ins, temporaries and views are meta tensors; a constant is a tensor holding its data)
def _key(value):
    value = value.tensor if isinstance(value, _Words) else value
    return value.untyped_storage()._cdata if value.device.type == 'meta' else value.untyped_storage().data_ptr()


# torch 2.14.0 torch/_tensor_docs.py view_as_real: a storage root's element type, a complex one's real pairs
def _stored(value):
    base = value.tensor if isinstance(value, _Words) else value
    return torch.view_as_real(base) if base.is_complex() else base


# torch 2.14.0 torchgen/model.py Annotation.is_write and alias_set (`Tensor(a!)`): the operands a call's results are, each
# return written in place into the argument of its alias set (an in-place call, copy_, an out= variant), else None
def _written(local, args, kwargs):
    if not isinstance(local.target, torch._ops.OpOverload) or not local.target._schema.returns:
        return None
    schema, given = local.target._schema, {**dict(zip(_names(local), args)), **kwargs}
    if not all(r.alias_info is not None and r.alias_info.is_write for r in schema.returns):
        return None
    return [next(given[a.name] for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write
                 and a.alias_info.before_set == r.alias_info.before_set) for r in schema.returns]


# torch 2.14.0 torchgen/model.py Annotation (`Tensor(a)`, `Tensor(a!)`): a call writing operands in place is those
# operands and its structured operations into them; a pure alias is its operand's storage and has no work; any other
# call's result is a temporary of its prototype's sizes and strides and its structured operations into it
def _instruction(local, args, kwargs, prototype, page, generator, outs=None):
    written = _written(local, args, kwargs)
    if written is not None:
        return written[0] if len(written) == 1 else tuple(written), _lowering(local, page, generator)(args, kwargs,
                                                                                                    written)
    if _aliases(local):
        return _call(local, args, kwargs), []
    if outs is not None:
        return outs[0] if len(outs) == 1 else tuple(outs), _lowering(local, page, generator)(args, kwargs, outs)
    outs, spec = _made(prototype, local)
    return pytree.tree_unflatten(outs, spec), _lowering(local, page, generator)(args, kwargs, outs)


# torch 2.14.0 torch/_tensor_docs.py untyped_storage, stride, storage_offset: the same view of the same storage
def _same_view(a, b):
    return (_key(a) == _key(b) and tuple(a.shape) == tuple(b.shape) and tuple(a.stride()) == tuple(b.stride())
            and a.storage_offset() == b.storage_offset())


def _made(prototype, local):
    if isinstance(local.target, torch._ops.OpOverload) and not local.target._schema.returns:
        return [None], pytree.tree_structure(None)
    made, spec = pytree.tree_flatten(prototype)
    return [_held(m) if m is not None else None for m in made], spec


# torch/__init__.py SymInt, SymBool and SymFloat (Python int, bool and float semantics): a scalar a higher-order
# operator carries, its 0-d value in the program (int64, bool, float64)
def _held(m):
    if isinstance(m, torch.Tensor):
        return torch.empty_strided(m.shape, m.stride(), dtype=m.dtype, device='meta')
    kind = torch.bool if isinstance(m, (bool, torch.SymBool)) else \
        torch.int64 if isinstance(m, (int, torch.SymInt)) else torch.float64
    return torch.empty((), dtype=kind, device='meta')


def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device,
            value.requires_grad) \
        if isinstance(value, torch.Tensor) else value

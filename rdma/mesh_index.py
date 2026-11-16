import functools

import torch
from torch._prims_common import type_to_dtype
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.numbers import is_infinite


INDEX_LT, INDEX_LE, INDEX_GT, INDEX_GE, INDEX_EQ, INDEX_NE = range(6)


def _unrepresented(*_, **__):
    return None


# rdma/mesh-graph.h struct mesh_graph_index over one kernel's domain as torch/utils/_sympy/interp.py sympy_interp's
# analysis at the kernel's index type `dtype` (codegen/simd.py get_index_dtype_as_torch_dtype, the type its index
# arithmetic is emitted in): each handler of an index expression its mesh_graph_index_* constructor of the same name,
# each integer result reduced exactly into that type (mesh_graph_index_wrap: its identity where isl proves it within
# the type's range), NULL (here None) where isl has no quasi-affine value (a float, a product of two non-constants, an
# unknown symbol or function); every value made kept in `made` (freed after metal_ir_upstream, which makes its access
# relations from them)
class Index:
    def __init__(self, lib, domain, env, missing, dtype):
        self.lib, self.domain, self.env, self.missing, self.dtype = lib, domain, env, missing, dtype
        self.made = []

    def interp(self, expr):
        return sympy_interp(self, self.env, expr, index_dtype=self.dtype, missing_handler=self.missing)

    def kept(self, value):
        if value:
            self.made.append(value)
        return value or None

    # mesh_graph_index_`name` over `operands`, unreduced
    def native(self, name, *operands):
        return self.kept(getattr(self.lib, "mesh_graph_index_" + name)(*operands))

    # an integer as a value of integer type `dtype` (torch's integer conversion and arithmetic: its two's-complement
    # reduction into the type's width), the index type by default
    def wrapped(self, value, dtype=None):
        dtype = dtype or self.dtype
        return self.native("wrap", value, dtype.itemsize * 8, int(dtype.is_signed))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _unrepresented

    def dim(self, k):
        return self.native("dim", self.domain, k)

    def truth(self, value):
        return self.native("truth", self.domain, int(bool(value)))

    def constant(self, value, dtype):
        if dtype is torch.bool:
            return self.truth(value)
        if dtype.is_floating_point or dtype.is_complex or is_infinite(value):
            return None
        return self.native("constant", self.domain, int(value))

    def add(self, a, b):
        return self.wrapped(self.native("add", a, b))

    def sym_sum(self, args):
        return functools.reduce(self.add, args)

    def mul(self, a, b):
        return self.wrapped(self.native("mul", a, b))

    def floordiv(self, a, b):
        return self.wrapped(self.native("floordiv", a, b))

    def modular_indexing(self, x, divisor, modulus):
        return self.wrapped(self.native("modular_indexing", x, divisor, modulus))

    def python_mod(self, a, b):
        return self.wrapped(self.native("python_mod", a, b))

    # sympy.Mod and torch/utils/_sympy/functions.py Mod (its operands nonnegative): the mathematical modulus
    mod = python_mod

    def minimum(self, a, b):
        return self.wrapped(self.native("minimum", a, b))

    def maximum(self, a, b):
        return self.wrapped(self.native("maximum", a, b))

    def where(self, predicate, a, b):
        return self.wrapped(self.native("where", predicate, a, b))

    def compare(self, relation, a, b):
        return self.native("compare", relation, a, b)

    def lt(self, a, b):
        return self.compare(INDEX_LT, a, b)

    def le(self, a, b):
        return self.compare(INDEX_LE, a, b)

    def gt(self, a, b):
        return self.compare(INDEX_GT, a, b)

    def ge(self, a, b):
        return self.compare(INDEX_GE, a, b)

    def eq(self, a, b):
        return self.compare(INDEX_EQ, a, b)

    def ne(self, a, b):
        return self.compare(INDEX_NE, a, b)

    def and_(self, a, b):
        return self.native("and", a, b)

    def or_(self, a, b):
        return self.native("or", a, b)

    def not_(self, a):
        return self.native("not", a)

    def identity(self, x):
        return x


class Value:
    __slots__ = ("index", "dtype")
    is_scalar = False

    def __init__(self, index, dtype):
        self.index, self.dtype = index, dtype


class Operations:
    def __init__(self, index):
        self.index = index

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _unrepresented

    # a native integer `value` as a value of `dtype`
    def integral(self, value, dtype):
        index = self.index
        if value is None or dtype.is_floating_point or dtype.is_complex:
            return None
        if dtype is torch.bool:
            return self.predicate(index.ne(value, index.constant(0, torch.int64)))
        return Value(index.wrapped(value, dtype), dtype)

    def predicate(self, value):
        return None if value is None else Value(value, torch.bool)

    # an operand's integer value (a bool's its indicator) and its truth (an integer's != 0)
    def integer(self, x):
        index = self.index
        if not isinstance(x, Value):
            return index.constant(x, type_to_dtype(type(x)))
        if x.dtype is not torch.bool:
            return x.index
        return index.native("where", x.index, index.constant(1, torch.int64), index.constant(0, torch.int64))

    def truth(self, x):
        index = self.index
        if not isinstance(x, Value):
            return index.truth(x)
        return x.index if x.dtype is torch.bool else index.ne(x.index, index.constant(0, torch.int64))

    def arithmetic(self, name, made, *operands, result_dtype):
        if any(x is None for x in operands):
            return None
        return self.integral(made(*map(self.integer, operands)), result_dtype)

    def index_expr(self, expr, dtype, *, result_dtype):
        return self.integral(self.index.interp(expr), result_dtype)

    def value_expr(self, expr, dtype, *, result_dtype):
        return self.integral(self.index.interp(expr), result_dtype)

    def constant(self, value, dtype, *, result_dtype):
        rule = result_dtype
        if rule is torch.bool:
            return self.predicate(self.index.truth(value))
        return self.integral(self.index.constant(value, rule), rule)

    def to_dtype(self, x, dtype, src_dtype=None, use_compute_types=True, *, result_dtype):
        if x is None:
            return None
        rule = result_dtype
        if rule is torch.bool:
            return self.predicate(self.truth(x))
        return self.integral(self.integer(x), rule)

    def add(self, a, b, *, result_dtype=None):
        return self.arithmetic("add", functools.partial(self.index.native, "add"), a, b, result_dtype=result_dtype)

    def sub(self, a, b, *, result_dtype=None):
        index = self.index
        return self.arithmetic("sub", lambda x, y: index.native(
            "add", x, index.native("mul", y, index.constant(-1, torch.int64))), a, b, result_dtype=result_dtype)

    def mul(self, a, b, *, result_dtype=None):
        return self.arithmetic("mul", functools.partial(self.index.native, "mul"), a, b, result_dtype=result_dtype)

    def neg(self, a, *, result_dtype=None):
        index = self.index
        return self.arithmetic("neg", lambda x: index.native("mul", x, index.constant(-1, torch.int64)), a, result_dtype=result_dtype)

    def floordiv(self, a, b, *, result_dtype=None):
        return self.arithmetic("floordiv", functools.partial(self.index.native, "floordiv"), a, b, result_dtype=result_dtype)

    def remainder(self, a, b, *, result_dtype=None):
        return self.arithmetic("remainder", functools.partial(self.index.native, "python_mod"), a, b, result_dtype=result_dtype)

    def truncdiv(self, a, b, *, result_dtype=None):
        return self.arithmetic("truncdiv", functools.partial(self.index.native, "tdiv_q"), a, b, result_dtype=result_dtype)

    def mod(self, a, b, *, result_dtype=None):
        return self.arithmetic("mod", functools.partial(self.index.native, "tdiv_r"), a, b, result_dtype=result_dtype)

    def minimum(self, a, b, *, result_dtype=None):
        return self.arithmetic("minimum", functools.partial(self.index.native, "minimum"), a, b, result_dtype=result_dtype)

    def maximum(self, a, b, *, result_dtype=None):
        return self.arithmetic("maximum", functools.partial(self.index.native, "maximum"), a, b, result_dtype=result_dtype)

    def where(self, predicate, a, b, *, result_dtype):
        if predicate is None or a is None or b is None:
            return None
        index, rule, p = self.index, result_dtype, self.truth(predicate)
        if rule is torch.bool:
            return self.predicate(index.or_(index.and_(p, self.truth(a)), index.and_(index.not_(p), self.truth(b))))
        return self.integral(index.native("where", p, self.integer(a), self.integer(b)), rule)

    def compared(self, relation, a, b):
        if a is None or b is None:
            return None
        return self.predicate(self.index.compare(relation, self.integer(a), self.integer(b)))

    def lt(self, a, b, *, result_dtype=None):
        return self.compared(INDEX_LT, a, b)

    def le(self, a, b, *, result_dtype=None):
        return self.compared(INDEX_LE, a, b)

    def gt(self, a, b, *, result_dtype=None):
        return self.compared(INDEX_GT, a, b)

    def ge(self, a, b, *, result_dtype=None):
        return self.compared(INDEX_GE, a, b)

    def eq(self, a, b, *, result_dtype=None):
        return self.compared(INDEX_EQ, a, b)

    def ne(self, a, b, *, result_dtype=None):
        return self.compared(INDEX_NE, a, b)

    def logical_and(self, a, b, *, result_dtype=None):
        return None if a is None or b is None else self.predicate(self.index.and_(self.truth(a), self.truth(b)))

    def logical_or(self, a, b, *, result_dtype=None):
        return None if a is None or b is None else self.predicate(self.index.or_(self.truth(a), self.truth(b)))

    def logical_not(self, a, *, result_dtype=None):
        return None if a is None else self.predicate(self.index.not_(self.truth(a)))

    # a bitwise operator: of bools the logical one, of integers not represented
    def and_(self, a, b, *, result_dtype=None):
        return self.logical_and(a, b) if a is not None and b is not None and a.dtype is b.dtype is torch.bool else None

    def or_(self, a, b, *, result_dtype=None):
        return self.logical_or(a, b) if a is not None and b is not None and a.dtype is b.dtype is torch.bool else None

    def not_(self, a, *, result_dtype=None):
        return self.logical_not(a) if a is not None and a.dtype is torch.bool else None

    bitwise_and, bitwise_or, bitwise_not = and_, or_, not_

    def identity(self, x, *, result_dtype=None):
        return x


def ranged(bounds):
    return tuple((-(1 << 63), (1 << 63) - 1)[k] if is_infinite(v) else int(v) for k, v in enumerate((bounds.lower, bounds.upper)))

import functools

import torch
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

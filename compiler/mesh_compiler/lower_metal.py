# design/the-compiler.md, order 4; design/torch-source-lowering-change-request.md, items 1 and 3
import copy
import ctypes as C
import hashlib
import math
import mmap
import operator
import os
import pathlib
import subprocess
import sys
import tempfile
import weakref
from dataclasses import dataclass, field

import torch

from .ir import RECV, SEND

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph
import mesh_metal4

ATEN, LIB = torch.ops.aten, mesh_metal4.LIB
TYPES = {torch.float32: "float", torch.float16: "half", torch.int64: "long", torch.int32: "int",
         torch.uint8: "uchar", torch.bool: "bool"}
NATIVE_TYPES = {**TYPES, torch.float64: "double", torch.int16: "short", torch.int8: "char"}
ARGUMENT, RECEIVED, RESULT, HELD = "argument", "received", "result", "held"
COMMIT, WAIT = "commit", "wait"
QUEUE, THREAD, SENT, ARRIVED = "queue", "host", "send", "recv"
TILE, SIMDGROUPS, GRAIN = 32, 4, 1024
STDLIB = "#include <metal_stdlib>\nusing namespace metal;\n"
# Apple, Metal Shading Language Specification 4.1, section 2.22.3; MPPTensorOpsMatMul2d.h
PRIMITIVES = ("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
              "using namespace mpp::tensor_ops;\n")
REFERENCE = pathlib.Path(torch.__file__).parent / "include/c10/metal/special_math.h"
PIPELINES, LIBRARIES, OWNED, HOST = {}, {}, {}, torch.device("cpu")
# design/compilation-composition-change-request.md, item 1 and "Source bindings"; Apple, Metal Shading Language
# Specification 4.1, sections 2.1, 2.2 and 6.5; ISO/IEC 14882:2020 [c.math]; Apple, vecLib cblas_new.h and
# dispatch/apply.h
NATIVE = """#include <cmath>
#include <cstddef>
#include <cstdint>
#include <dispatch/dispatch.h>
#define ACCELERATE_NEW_LAPACK 1
#include <vecLib/cblas_new.h>
namespace mesh {
typedef unsigned int uint;
typedef unsigned char uchar;
typedef _Float16 half;
struct uint3 { uint x, y, z; uint3(uint a, uint b, uint c) : x(a), y(b), z(c) {} };
template <class T> static inline T max(T a, T b) { return a < b ? b : a; }
template <class T> static inline T min(T a, T b) { return b < a ? b : a; }
template <class T> static inline T abs(T a) { return a < 0 ? T(-a) : a; }
static inline float abs(float a) { return std::fabs(a); }
static inline double abs(double a) { return std::fabs(a); }
static inline half abs(half a) { return half(std::fabs(float(a))); }
#define MESH_UNARY(name) static inline float name(float a) { return std::name(a); } \\
    static inline double name(double a) { return std::name(a); } \\
    static inline half name(half a) { return half(std::name(float(a))); }
#define MESH_BINARY(name) static inline float name(float a, float b) { return std::name(a, b); } \\
    static inline double name(double a, double b) { return std::name(a, b); } \\
    static inline half name(half a, half b) { return half(std::name(float(a), float(b))); }
MESH_UNARY(exp) MESH_UNARY(log) MESH_UNARY(sqrt) MESH_UNARY(sin) MESH_UNARY(cos) MESH_UNARY(tanh) MESH_UNARY(erf)
MESH_UNARY(floor) MESH_UNARY(ceil) MESH_UNARY(round) MESH_UNARY(trunc)
MESH_BINARY(pow) MESH_BINARY(fmod)
template <class T> static inline T rsqrt(T a) { return T(1) / sqrt(a); }
template <class T> static inline bool isnan(T a) { return a != a; }
static const float M_SQRT1_2_F = 0.70710678118654752440f;
struct work { void (*kernel)(void *const *, size_t, size_t); void *const *table; size_t count, chunk; };
static void part(void *context, size_t at) {
    const work *w = (const work *)context;
    size_t first = at * w->chunk, last = first + w->chunk;
    w->kernel(w->table, first, last < w->count ? last : w->count);
}
static inline void run(void (*kernel)(void *const *, size_t, size_t), void *const *table, size_t count, size_t chunk) {
    work w = {kernel, table, count, chunk};
    dispatch_apply_f((count + chunk - 1) / chunk, DISPATCH_APPLY_AUTO, &w, part);
}
"""
# torch/csrc/api/include/torch/mps.h: commit, get_command_buffer, get_dispatch_queue
TORCH = [C.cast(getattr(C.CDLL(str(pathlib.Path(torch.__file__).parent / "lib/libtorch_cpu.dylib")), name),
                C.c_void_p) for name in ("_ZN5torch3mps6commitEv", "_ZN5torch3mps18get_command_bufferEv",
                                         "_ZN5torch3mps18get_dispatch_queueEv")]

POINTWISE = {
    ATEN.add.Tensor: "({T}({0}) + {T}({1}))", ATEN.add.Scalar: "({T}({0}) + {T}({1}))",
    ATEN.sub.Tensor: "({T}({0}) - {T}({1}))", ATEN.sub.Scalar: "({T}({0}) - {T}({1}))",
    ATEN.mul.Tensor: "({T}({0}) * {T}({1}))", ATEN.mul.Scalar: "({T}({0}) * {T}({1}))",
    ATEN.div.Tensor: "({T}({0}) / {T}({1}))", ATEN.div.Scalar: "({T}({0}) / {T}({1}))",
    ATEN.eq.Scalar: "({0} == {1})", ATEN.eq.Tensor: "({0} == {1})",
    ATEN.ne.Scalar: "({0} != {1})", ATEN.ne.Tensor: "({0} != {1})",
    ATEN.gt.Scalar: "({0} > {1})", ATEN.gt.Tensor: "({0} > {1})",
    ATEN.lt.Scalar: "({0} < {1})", ATEN.lt.Tensor: "({0} < {1})",
    ATEN.ge.Scalar: "({0} >= {1})", ATEN.le.Scalar: "({0} <= {1})",
    ATEN.logical_not.default: "(!bool({0}))",
    ATEN.logical_and.default: "(bool({0}) && bool({1}))", ATEN.logical_or.default: "(bool({0}) || bool({1}))",
    ATEN.where.self: "(bool({0}) ? {T}({1}) : {T}({2}))",
    ATEN.neg.default: "(-{T}({0}))", ATEN.abs.default: "abs({T}({0}))",
    ATEN.exp.default: "exp({T}({0}))", ATEN.log.default: "log({T}({0}))",
    ATEN.sqrt.default: "sqrt({T}({0}))", ATEN.rsqrt.default: "rsqrt({T}({0}))",
    ATEN.sin.default: "sin({T}({0}))", ATEN.cos.default: "cos({T}({0}))",
    ATEN.tanh.default: "tanh({T}({0}))",
    ATEN.sigmoid.default: "(1 / (1 + exp(-{T}({0}))))",
    ATEN.relu.default: "max({T}({0}), {T}(0))",
    ATEN.gelu.default: "(0.5f * {0} * (1 + erf({0} * M_SQRT1_2_F)))",
    ATEN.clone.default: "{0}",
    ATEN._to_copy.default: "{T}({0})",
}
# design/the-compiler.md, order 5; torch/nn/modules/activation.py, GELU
GELU = {"none": POINTWISE[ATEN.gelu.default],
        "tanh": "(0.5f * {0} * (1 + tanh(0.7978845608f * ({0} + 0.044715f * {0} * {0} * {0}))))"}
VIEWS = {ATEN.view.default, ATEN._unsafe_view.default, ATEN.reshape.default, ATEN.permute.default,
         ATEN.expand.default, ATEN.slice.Tensor, ATEN.select.int, ATEN.unsqueeze.default, ATEN.squeeze.dims,
         ATEN.squeeze.dim, ATEN.transpose.int, ATEN.alias.default}
RESHAPES = {ATEN.view.default, ATEN._unsafe_view.default, ATEN.reshape.default}
FILLED = {ATEN.full.default, ATEN.full_like.default, ATEN.scalar_tensor.default}
REDUCED = {ATEN.sum.dim_IntList: ("0", "a + {x}", "a"), ATEN.mean.dim: ("0", "a + {x}", "a / {n}"),
           ATEN.amax.default: ("{x}", "max(a, {x})", "a"), ATEN.amin.default: ("{x}", "min(a, {x})", "a"),
           ATEN.any.dim: ("false", "a || {x}", "a")}
# torch 2.15 documentation, torch.remainder
REMAINDER = ("template <typename T> inline T mesh_%s(T a, T b) {\n    T m = %s;\n"
             "    return (m != 0 && ((m < 0) != (b < 0))) ? T(m + b) : m;\n}\n")
WRAP = "inline uint mesh_wrap(long i, long n) { return uint(i < 0 ? i + n : i); }\n"
SCATTERED = {ATEN.scatter_add.default: True, ATEN.scatter.src: False, ATEN.scatter.value: False}
PASSED = {"memory_format", "approximate", "dtype", "layout", "device", "pin_memory", "non_blocking"}
SCALED = {ATEN.addmm.default: {"beta", "alpha"}}


@dataclass
class View:
    storage: int
    dtype: torch.dtype
    shape: tuple
    stride: tuple
    offset: int

    def read(self, bind, index, write=False):
        terms = [f"{i} * {s}" for i, s, n in zip(index, self.stride, self.shape) if n > 1 and s]
        return f"{bind(self, write)}[{' + '.join([str(self.offset)] * bool(self.offset) + terms) or '0'}]"


@dataclass
class Fused:
    dtype: torch.dtype
    shape: tuple
    expression: object
    nodes: tuple = ()
    uses: frozenset = frozenset()

    def read(self, bind, index):
        bind.note(self)
        return self.expression(bind, index, None)


@dataclass
class Product:
    left: View
    right: View
    dtype: torch.dtype
    shape: tuple
    expression: object
    plain: bool = False
    nodes: tuple = ()
    uses: frozenset = frozenset()


@dataclass
class Storage:
    kind: str
    dtype: torch.dtype
    count: int
    position: int = None


@dataclass
class Step:
    vertex: int
    nodes: tuple
    pipeline: int
    storages: list
    accesses: list
    grid: tuple
    group: tuple
    groups: bool


@dataclass
class Segment:
    steps: list
    barriers: list = field(default_factory=list)
    after: dict = field(default_factory=dict)
    fence: list = field(default_factory=list)
    record: list = field(default_factory=list)


@dataclass
class Dependence:
    source: int
    target: int
    causes: list


class Bound(list):
    def __init__(self):
        super().__init__()
        self.accesses, self.nodes, self.uses = [], [], set()

    def note(self, form):
        self.nodes += form.nodes
        self.uses |= form.uses

    def __call__(self, view, write=False):
        self.accesses.append((view, write))
        if view.storage not in self:
            self.append(view.storage)
        return f"b{self.index(view.storage)}"


class NotLowered(NotImplementedError):
    def __init__(self, text, operators=()):
        super().__init__(text)
        self.operators = list(operators)


def packed(shape):
    return tuple(math.prod(shape[d + 1:]) for d in range(len(shape)))


def unravel(linear, shape, prefix):
    names = [f"{prefix}{d}" for d in range(len(shape))]
    return [f"uint {name} = {linear} / {step} % {n};"
            for name, n, step in zip(names, shape, packed(shape))], names


def reshaped(index, shape, onto):
    if tuple(shape) == tuple(onto):
        return index
    linear = " + ".join(f"{i} * {step}" for i, step in zip(index, packed(shape))) or "0"
    return [f"(({linear}) / {step} % {n})" for n, step in zip(onto, packed(onto))]


def literal(value, wide=False):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    return "NAN" if math.isnan(value) else f"{value!r}" + "f" * (not wide)


def reference(name):
    text = REFERENCE.read_text()
    start = text.index(f"template <typename T>\ninline float {name}(T x) {{")
    return text[start:text.index("\n}\n", start) + 3]


def pipeline(source):
    if source not in PIPELINES:
        PIPELINES[source] = mesh_metal4.pipeline(source, "k")
    return PIPELINES[source]


SHARED = {"remainder": REMAINDER % ("remainder", "a % b"), "fremainder": REMAINDER % ("fremainder", "fmod(a, b)"),
          "wrap": WRAP}
HELPERS = {"erf": lambda: reference("erf"), **{name: (lambda text=text: text) for name, text in SHARED.items()}}


# design/compilation-composition-change-request.md, item 7, "CPU execution is native Apple Silicon work"
def native(source):
    if source not in LIBRARIES:
        directory = tempfile.TemporaryDirectory()
        stem = pathlib.Path(directory.name) / hashlib.sha256(source.encode()).hexdigest()[:16]
        stem.with_suffix(".cpp").write_text(source)
        built = subprocess.run(["/usr/bin/xcrun", "clang++", "-std=c++20", "-O2", "-dynamiclib", "-framework",
                                "Accelerate", str(stem.with_suffix(".cpp")), "-o", str(stem.with_suffix(".dylib"))],
                               capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        LIBRARIES[source] = (C.CDLL(str(stem.with_suffix(".dylib"))), directory)
    return LIBRARIES[source][0]


def shaped(owner, view):
    if not all(view.shape):
        return torch.empty(view.shape, dtype=view.dtype)
    base = owner[view.storage]
    return base.as_strided(view.shape, view.stride, base.storage_offset() + view.offset)


class Lowering:
    types = TYPES

    def __init__(self, module, laid):
        self.laid, self.storages, self.forms, self.missing = laid, [], {}, []
        self.pending, self.program, self.results, self.vertices, self.hosted = [], [], (), [], []
        handlers = {**dict.fromkeys(POINTWISE, self.pointwise), **dict.fromkeys(VIEWS, self.viewed),
                    RECV: self.receive, SEND: self.send, operator.getitem: self.part,
                    **dict.fromkeys(FILLED, self.filled), **dict.fromkeys(REDUCED, self.reduced),
                    ATEN.arange.start_step: self.range, ATEN.embedding.default: self.embedding,
                    ATEN.gather.default: self.gather, ATEN.index_select.default: self.index_select,
                    ATEN.index.Tensor: self.index, ATEN.remainder.Scalar: self.remainder,
                    ATEN.remainder.Tensor: self.remainder, **dict.fromkeys(SCATTERED, self.scattered),
                    ATEN.pow.Tensor_Scalar: self.power, ATEN.gelu.default: self.gelu,
                    ATEN.cat.default: self.concatenated, ATEN.sort.default: self.sort, ATEN.topk.default: self.topk,
                    ATEN._native_batch_norm_legit_no_training.default: self.batch_norm,
                    ATEN.max_pool2d_with_indices.default: self.max_pool,
                    ATEN._softmax.default: self.softmax,
                    ATEN.native_layer_norm.default: self.layer_norm, ATEN.convolution.default: self.convolution,
                    ATEN.mm.default: self.product, ATEN.bmm.default: self.product, ATEN.addmm.default: self.product}
        self.readers = handlers.keys() - (VIEWS - RESHAPES) - FILLED - {
            RECV, SEND, operator.getitem, ATEN.arange.start_step, ATEN.mm.default, ATEN.bmm.default,
            ATEN.addmm.default}
        position = 0
        for node in module.graph.nodes:
            if node.op == "placeholder":
                self.missing += [f"{node.name} of {v.dtype}" for v in self.untyped(node)]
                self.forms[node] = self.arrived(ARGUMENT, node, position)
                position += 1
            elif node.op == "output" and not self.missing:
                self.issuing = node
                self.results = tuple(self.escape(r) if isinstance(r, torch.fx.Node) else r for r in node.args[0])
                self.flush()
                self.host(node, [(r, False) for r in self.results if isinstance(r, View)])
                self.ordered()
            elif node.op == "call_function" and (
                    node.target not in handlers or set(node.kwargs) - PASSED - SCALED.get(node.target, set())):
                self.missing.append(str(node.target))
            elif node.op == "call_function" and self.untyped(node):
                self.missing += [f"{node.target} of {v.dtype}" for v in self.untyped(node)]
            elif node.op == "call_function" and not self.missing:
                self.issuing, self.using = node, set()
                form = handlers[node.target](node)
                if isinstance(form, (Fused, Product)):
                    reads = [self.forms[read] for read in node.all_input_nodes]
                    form.nodes = (*dict.fromkeys(name for read in reads for name in getattr(read, "nodes", ())),
                                  node.name)
                    form.uses = frozenset(self.using).union(*(getattr(read, "uses", ()) for read in reads))
                self.forms[node] = form if self.deferred(node, form) else self.settled(form)

    def untyped(self, node):
        value = node.meta.get("val")
        values = value if isinstance(value, (tuple, list)) else [value]
        return [v for v in values if isinstance(v, torch.Tensor) and v.dtype not in self.types]

    def arrived(self, kind, node, position=None):
        value = node.meta["val"]
        count = (1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))) * all(value.shape)
        self.storages.append(Storage(kind, value.dtype, count, position))
        return View(len(self.storages) - 1, value.dtype, tuple(value.shape), tuple(value.stride()), 0)

    def fresh(self, dtype, shape):
        self.storages.append(Storage(HELD, dtype, math.prod(shape)))
        return View(len(self.storages) - 1, dtype, tuple(shape), packed(shape), 0)

    def deferred(self, node, form):
        users = list(node.users)
        if len(users) != 1 or not isinstance(form, (Fused, Product)):
            return False
        if isinstance(form, Product):
            return users[0].target in RESHAPES or \
                users[0].target in POINTWISE and tuple(users[0].meta["val"].shape) == form.shape
        return users[0].target in self.readers

    def settled(self, form):
        return self.view(form) if isinstance(form, (Fused, Product)) else form

    def view(self, form):
        if isinstance(form, View):
            return form
        made = self.fresh(form.dtype, form.shape)
        if isinstance(form, Product):
            self.multiply(form, made)
        else:
            self.store(form, made)
        return made

    def reader(self, node):
        form = self.forms[node]
        return self.view(form) if isinstance(form, Product) else form

    def dispatch(self, bound, lines, grid, group=None, groups=False, head=""):
        if not math.prod(grid):
            return
        nodes = tuple(dict.fromkeys((*bound.nodes, self.issuing.name)))
        made, group = self.kernel(bound, lines, tuple(grid), group, head)
        self.pending.append(Step(len(self.vertices), nodes, made, list(bound), bound.accesses, tuple(grid), group,
                                 groups))
        self.vertices.append(nodes)

    def store(self, form, target):
        bind = Bound()
        lines, names = unravel("p.x", form.shape, "i")
        lines.append(f"{target.read(bind, names, True)} = {form.read(bind, names)};")
        self.dispatch(bind, lines, (math.prod(form.shape), 1, 1))

    def multiply(self, form, target):
        bind, kind = Bound(), self.types[form.dtype]
        bind.note(form)
        natural = form.left.shape[:-1] + form.right.shape[-1:]
        lines, names = unravel("p.x", natural, "i")
        left, right = (operand.read(bind, at) for operand, at in (
            (form.left, names[:-1] + ["k"]), (form.right, names[:-2] + ["k", names[-1]])))
        lines += [f"{self.accumulated(form.dtype)} a = 0;",
                  f"for (uint k = 0; k < {form.left.shape[-1]}; k++) a += {left} * {right};", f"{kind} t = {kind}(a);",
                  f"{target.read(bind, names, True)} = {form.expression(bind, reshaped(names, natural, form.shape), 't')};"]
        self.dispatch(bind, lines, (math.prod(natural), 1, 1))

    def accumulated(self, dtype):
        return self.types[dtype] if dtype != torch.float16 else "float"

    def matrix(self, node):
        view = self.view(self.forms[node])
        (m, n), (sm, sn) = view.shape[-2:], view.stride[-2:]
        if (n == 1 or sn == 1) and (m == 1 or sm >= n) or (m == 1 or sm == 1) and (n == 1 or sn >= m):
            return view
        return self.view(Fused(view.dtype, view.shape, lambda bind, index, t: view.read(bind, index)))

    def product(self, node):
        value = node.meta["val"]
        left, right = (self.matrix(a) for a in node.args[-2:])
        if node.target is not ATEN.addmm.default:
            return Product(left, right, value.dtype, tuple(value.shape), lambda bind, index, t: t, True)
        added, beta, alpha = self.reader(node.args[0]), node.kwargs.get("beta", 1), node.kwargs.get("alpha", 1)
        wide = value.dtype == torch.float64

        def expression(bind, index, t):
            return f"({literal(beta, wide)} * {added.read(bind, index[len(index) - len(added.shape):])} + " \
                   f"{literal(alpha, wide)} * {t})"
        return Product(left, right, value.dtype, tuple(value.shape), expression)

    def pointwise(self, node, template=None):
        value = node.meta["val"]
        template, kind, shape = template or POINTWISE[node.target], self.types[value.dtype], tuple(value.shape)
        operands = [self.forms[a] if isinstance(a, torch.fx.Node) else a for a in node.args]
        if node.target is ATEN._to_copy.default and value.dtype == node.args[0].meta["val"].dtype:
            return operands[0]
        base = next((o for o in operands if isinstance(o, Product) and o.shape == shape), None)
        operands = [self.view(o) if isinstance(o, Product) and o is not base else o for o in operands]
        wide = torch.float64 in {value.dtype, *(o.dtype for o in operands if isinstance(o, (View, Fused, Product)))}

        def expression(bind, index, t):
            return template.format(*(
                o.expression(bind, index, t) if o is base else
                o.read(bind, index[len(index) - len(o.shape):]) if isinstance(o, (View, Fused)) else
                literal(o, wide) for o in operands), T=kind)
        return Product(base.left, base.right, value.dtype, shape, expression) if base else \
            Fused(value.dtype, shape, expression)

    def gelu(self, node):
        self.using |= {"erf"} if node.kwargs.get("approximate", "none") == "none" else set()
        return self.pointwise(node, GELU[node.kwargs.get("approximate", "none")])

    def power(self, node):
        whole = float(node.args[1]).is_integer() and 1 <= node.args[1] <= 4
        return self.pointwise(node, "(" + " * ".join(["{T}({0})"] * int(node.args[1])) + ")" if whole else
                              "pow({T}({0}), {T}({1}))")

    def filled(self, node):
        value = node.meta["val"]
        return Fused(value.dtype, tuple(value.shape),
                     lambda bind, index, t: f"{self.types[value.dtype]}({literal(node.args[-1])})")

    def range(self, node):
        value, (start, _, step) = node.meta["val"], (*node.args, 1)[:3]
        return Fused(value.dtype, tuple(value.shape), lambda bind, index, t:
                     f"{self.types[value.dtype]}({literal(start)} + {index[0]} * {literal(step)})")

    def remainder(self, node):
        name = "fremainder" if node.meta["val"].dtype.is_floating_point else "remainder"
        self.using.add(name)
        return self.pointwise(node, "mesh_" + name + "({T}({0}), {T}({1}))")

    # torch 2.15 documentation, torch.gather, torch.index_select, torch.nn.functional.embedding and Tensor Indexing
    # API; NumPy User Guide, Indexing on ndarrays, Advanced indexing
    def take(self, node, source, dims):
        value = node.meta["val"]
        self.using.add("wrap")

        def expression(bind, index, t):
            at = []
            for extent, (chosen, places) in zip(source.shape, dims):
                if chosen is None:
                    at.append(index[places])
                else:
                    found = chosen.read(bind, [index[k] for k in places][-len(chosen.shape) or len(places):])
                    at.append(f"mesh_wrap(long({found}), {extent})")
            return source.read(bind, at)
        return Fused(value.dtype, tuple(value.shape), expression)

    def embedding(self, node):
        table, chosen = (self.reader(a) for a in node.args[:2])
        return self.take(node, table, [(chosen, range(len(chosen.shape))), (None, len(chosen.shape))])

    def gather(self, node):
        source, chosen = self.reader(node.args[0]), self.reader(node.args[2])
        dim, every = node.args[1] % max(len(source.shape), 1), range(len(source.shape))
        return self.take(node, source, [(chosen, every) if d == dim else (None, d) for d in every])

    def index_select(self, node):
        source, chosen = self.reader(node.args[0]), self.reader(node.args[2])
        dim = node.args[1] % max(len(source.shape), 1)
        return self.take(node, source, [(chosen, [d]) if d == dim else (None, d) for d in range(len(source.shape))])

    def index(self, node):
        source = self.reader(node.args[0])
        chosen = {d: self.reader(a) for d, a in enumerate(node.args[1]) if a is not None}
        shape = torch.broadcast_shapes(*[c.shape for c in chosen.values()])
        adjacent = list(chosen) == list(range(min(chosen), max(chosen) + 1))
        start = min(chosen) if adjacent else 0
        rest = [d for d in range(len(source.shape)) if d not in chosen]
        over = range(start, start + len(shape))
        return self.take(node, source, [
            (chosen[d], over) if d in chosen else
            (None, (d if d < start else d - len(chosen) + len(shape)) if adjacent else len(shape) + rest.index(d))
            for d in range(len(source.shape))])

    # K. E. Batcher, Sorting networks and their applications, AFIPS Spring Joint Computer Conference, 1968
    def bitonic(self, fields, rows, padded, before):
        stage = 2
        while stage <= padded:
            step = stage // 2
            while step:
                bind = Bound()
                lines = [f"uint row = p.x / {padded // 2}, q = p.x % {padded // 2};",
                         f"uint i = ((q & ~{step - 1}u) << 1) | (q & {step - 1}u);",
                         f"uint j = i ^ {stage - 1 if step == stage // 2 else step}u;"]
                for k, field in enumerate(fields):
                    lines += [f"{self.types[field.dtype]} x{k} = {field.read(bind, ['row', 'i'])}, "
                              f"y{k} = {field.read(bind, ['row', 'j'])};"]
                lines += [f"if (!({before('x', 'y')})) {{"]
                lines += [f"    {field.read(bind, ['row', 'i'], True)} = y{k}; {field.read(bind, ['row', 'j'], True)} "
                          f"= x{k};" for k, field in enumerate(fields)] + ["}"]
                self.dispatch(bind, lines, (rows * padded // 2, 1, 1))
                step //= 2
            stage *= 2

    # torch 2.15 documentation, torch.sort and torch.topk
    def sorted(self, source, dim, descending):
        count = source.shape[dim]
        lead, padded = source.shape[:dim] + source.shape[dim + 1:], 1 << max(count - 1, 0).bit_length()
        rows = math.prod(lead)
        values, places = self.fresh(source.dtype, (rows, padded)), self.fresh(torch.int64, (rows, padded))
        bind = Bound()
        lines, names = unravel(f"(p.x / {padded})", lead, "r")
        at = names[:dim] + [f"min(p.x % {padded}, {max(count - 1, 0)}u)"] + names[dim:]
        lines += [f"{values.read(bind, [f'(p.x / {padded})', f'(p.x % {padded})'], True)} = {source.read(bind, at)};",
                  f"{places.read(bind, [f'(p.x / {padded})', f'(p.x % {padded})'], True)} = p.x % {padded};"]
        self.dispatch(bind, lines, (rows * padded, 1, 1))
        first, second = ("y", "x") if descending else ("x", "y")
        less = f"({first}0 < {second}0)" if not source.dtype.is_floating_point else \
            f"(({first}0 < {second}0) || (isnan({second}0) && !isnan({first}0)))"
        same = f"({first}0 == {second}0)" if not source.dtype.is_floating_point else \
            f"(({first}0 == {second}0) || (isnan({second}0) && isnan({first}0)))"

        self.bitonic([values, places], rows, padded, lambda a, b: (
            f"(({a}1 < {count}) != ({b}1 < {count})) ? ({a}1 < {count}) : ({less} || ({same} && {a}1 < {b}1))"))
        strides = [0] * len(source.shape)
        for d, step in zip([d for d in range(len(source.shape)) if d != dim], packed(lead)):
            strides[d] = step * padded
        strides[dim] = 1
        return [View(v.storage, v.dtype, tuple(source.shape), tuple(strides), 0) for v in (values, places)]

    def sort(self, node):
        source = self.view(self.forms[node.args[0]])
        dim = (node.args[1] if len(node.args) > 1 else -1) % len(source.shape)
        return self.sorted(source, dim, bool(node.args[2]) if len(node.args) > 2 else False)

    def topk(self, node):
        source, k = self.view(self.forms[node.args[0]]), node.args[1]
        dim = (node.args[2] if len(node.args) > 2 else -1) % len(source.shape)
        made = self.sorted(source, dim, node.args[3] if len(node.args) > 3 else True)
        return [View(v.storage, v.dtype, v.shape[:dim] + (k,) + v.shape[dim + 1:], v.stride, 0) for v in made]

    # torch 2.15 documentation, torch.Tensor.scatter_ and torch.Tensor.scatter_add_; W. D. Hillis and G. L. Steele,
    # Data parallel algorithms, CACM 29(12), 1986; G. E. Blelloch, Prefix sums and their applications,
    # CMU-CS-90-190, 1990, section 1.5
    def scattered(self, node):
        value, (source, dim, chosen, given) = node.meta["val"], node.args
        source, chosen = self.reader(source), self.view(self.forms[chosen])
        given = self.reader(given) if isinstance(given, torch.fx.Node) else given
        dim, kind = dim % len(source.shape), self.types[value.dtype]
        made, count = self.fresh(value.dtype, value.shape), math.prod(chosen.shape)
        self.store(source, made)
        if not count:
            return made
        padded = 1 << (count - 1).bit_length()
        keys, places = (self.fresh(torch.int64, (1, padded)) for _ in range(2))
        bind = Bound()
        lines, names = unravel(f"min(p.x, {count - 1}u)", chosen.shape, "c")
        target = " + ".join(f"long({chosen.read(bind, names) if d == dim else name}) * {step}"
                            for d, (name, step) in enumerate(zip(names, made.stride)))
        lines += [f"{keys.read(bind, ['0', 'p.x'], True)} = {target};",
                  f"{places.read(bind, ['0', 'p.x'], True)} = p.x;"]
        self.dispatch(bind, lines, (padded, 1, 1))
        self.bitonic([keys, places], 1, padded, lambda a, b: (
            f"(({a}1 < {count}) != ({b}1 < {count})) ? ({a}1 < {count}) : "
            f"({a}0 < {b}0 || ({a}0 == {b}0 && {a}1 < {b}1))"))
        sums, flags = self.fresh(value.dtype, (padded,)), self.fresh(torch.bool, (padded,))
        bind = Bound()
        lines, names = unravel(f"uint({places.read(bind, ['0', 'p.x'])})", chosen.shape, "c")
        x = given.read(bind, names) if isinstance(given, (View, Fused)) else literal(given, kind == "double")
        lines += [f"{sums.read(bind, ['p.x'], True)} = {kind}({x});",
                  f"{flags.read(bind, ['p.x'], True)} = p.x == 0 || "
                  f"{keys.read(bind, ['0', 'p.x'])} != {keys.read(bind, ['0', 'max(p.x, 1u) - 1'])};"]
        self.dispatch(bind, lines, (count, 1, 1))
        reach = 1
        while SCATTERED[node.target] and reach < count:
            after, ended, bind = self.fresh(value.dtype, (padded,)), self.fresh(torch.bool, (padded,)), Bound()
            far = [f"max(p.x, {reach}u) - {reach}"]
            self.dispatch(bind, [
                f"bool f = {flags.read(bind, ['p.x'])}, joined = !f && p.x >= {reach};",
                f"{kind} v = {sums.read(bind, ['p.x'])};",
                f"{after.read(bind, ['p.x'], True)} = joined ? {kind}({sums.read(bind, far)} + v) : v;",
                f"{ended.read(bind, ['p.x'], True)} = f || (joined && {flags.read(bind, far)});"], (count, 1, 1))
            sums, flags, reach = after, ended, 2 * reach
        bind = Bound()
        key = keys.read(bind, ["0", "p.x"])
        out = f"{bind(made, True)}[{key}]"
        self.dispatch(bind, [
            f"if (p.x == {count - 1} || {keys.read(bind, ['0', f'min(p.x + 1, {count - 1}u)'])} != {key})",
            f"    {out} = {f'{out} + ' * SCATTERED[node.target]}{sums.read(bind, ['p.x'])};"], (count, 1, 1))
        return made

    # torch 2.15 aten/src/ATen/native/Normalization.cpp, batch_norm_cpu_transform_input_template
    def batch_norm(self, node):
        source, weight, bias, mean, variance, _, epsilon = node.args
        value = node.meta["val"][0]
        source, weight, bias, mean, variance = (o if o is None else self.reader(o)
                                                for o in (source, weight, bias, mean, variance))
        kind, wide = self.types[value.dtype], value.dtype == torch.float64

        def expression(bind, index, t):
            at = index[1:2]
            scaled = f"(({source.read(bind, index)} - {mean.read(bind, at)}) / " \
                     f"sqrt({kind}({variance.read(bind, at)} + {kind}({literal(float(epsilon), wide)}))))"
            scaled += f" * {weight.read(bind, at)}" * (weight is not None)
            return f"{kind}({scaled}{f' + {bias.read(bind, at)}' * (bias is not None)})"
        made = Fused(value.dtype, tuple(value.shape), expression)
        made.nodes = (node.name,)
        empty = [Fused(v.dtype, tuple(v.shape), lambda bind, index, t: "0") for v in node.meta["val"][1:]]
        return [self.view(made), *empty]

    # torch 2.15 aten/src/ATen/native/DilatedMaxPool2d.cpp, max_pool2d_with_indices_single_out_frame
    def max_pool(self, node):
        source, kernel, stride, padding, dilation = (*node.args, *([[], [0, 0], [1, 1]][len(node.args) - 2:]))[:5]
        stride = stride or kernel
        source, values = self.reader(source), node.meta["val"]
        used = {u.args[1] for u in node.users}
        made = [self.fresh(v.dtype, v.shape) if place in used else None for place, v in enumerate(values)]
        bind, kind = Bound(), self.types[values[0].dtype]
        lines, names = unravel("p.x", values[0].shape, "o")
        high, wide = source.shape[-2:]
        taken = source.read(bind, names[:-2] + ["uint(y)", "uint(x)"])
        lines += [f"{kind} m = -INFINITY; long at = -1;",
                  f"for (uint u = 0; u < {kernel[0]}; u++) for (uint v = 0; v < {kernel[1]}; v++) {{",
                  f"    int y = int({names[-2]} * {stride[0]} + u * {dilation[0]}) - {padding[0]};",
                  f"    int x = int({names[-1]} * {stride[1]} + v * {dilation[1]}) - {padding[1]};",
                  f"    if (y < 0 || y >= {high} || x < 0 || x >= {wide}) continue;",
                  f"    {kind} c = {taken};",
                  f"    if (at < 0 || c > m || isnan(c)) {{ m = c; at = y * {wide} + x; }}", "}"]
        lines += [f"{view.read(bind, names, True)} = {found};" for view, found in zip(made, ("m", "at")) if view]
        self.dispatch(bind, lines, (math.prod(values[0].shape), 1, 1))
        return made

    def viewed(self, node):
        form, shape = self.forms[node.args[0]], tuple(node.meta["val"].shape)
        if isinstance(form, Product) and node.target in RESHAPES:
            return Product(form.left, form.right, form.dtype, shape, lambda bind, index, t:
                           form.expression(bind, reshaped(index, shape, form.shape), t), form.plain)
        if isinstance(form, Fused) and node.target in RESHAPES:
            return Fused(form.dtype, shape,
                         lambda bind, index, t: form.read(bind, reshaped(index, shape, form.shape)))
        view = self.view(form)
        count = self.storages[view.storage].count
        for attempt in range(2):
            pages = torch.empty(count, dtype=view.dtype, device="meta")
            try:
                made = node.target(pages.as_strided(view.shape, view.stride, view.offset), *node.args[1:],
                                   **node.kwargs)
            except RuntimeError:
                made = None
            if made is not None and made._base is pages:
                return View(view.storage, view.dtype, tuple(made.shape), tuple(made.stride()),
                            made.storage_offset())
            view = self.view(Fused(view.dtype, view.shape, lambda bind, index, t, held=view: held.read(bind, index)))
            count = self.storages[view.storage].count
        raise NotLowered(f"{node.target}: not a view of its operand")

    def part(self, node):
        return self.forms[node.args[0]][node.args[1]]

    def concatenated(self, node):
        value = node.meta["val"]
        made, at = self.fresh(value.dtype, value.shape), 0
        dim = (node.args[1] if len(node.args) > 1 else 0) % len(value.shape)
        for operand in node.args[0]:
            form = self.reader(operand)
            if len(form.shape) == len(value.shape):
                self.store(form, View(made.storage, made.dtype, form.shape, made.stride,
                                      at * made.stride[dim]))
                at += form.shape[dim]
        return made

    def softmax(self, node):
        source = self.reader(node.args[0])
        dim = node.args[1] % len(source.shape)
        made = self.fresh(node.meta["val"].dtype, source.shape)
        bind = Bound()
        lines, names = unravel("p.x", source.shape[:dim] + source.shape[dim + 1:], "r")
        index = names[:dim] + ["k"] + names[dim:]
        out, x = made.read(bind, index, True), source.read(bind, index)
        loop = f"for (uint k = 0; k < {source.shape[dim]}; k++)"
        wide = self.accumulated(source.dtype)
        lines += [f"{wide} m = -INFINITY, s = 0;", f"{loop} m = max(m, {wide}({x}));",
                  f"{loop} s += exp({wide}({x}) - m);", f"{loop} {out} = exp({wide}({x}) - m) / s;"]
        self.dispatch(bind, lines, (math.prod(source.shape) // source.shape[dim], 1, 1))
        return made

    def reduced(self, node):
        source, value = self.reader(node.args[0]), node.meta["val"]
        first, step, last = REDUCED[node.target]
        over = node.args[1] if len(node.args) > 1 else None
        over = sorted(d % len(source.shape) for d in ([over] if isinstance(over, int) else
                                                      over or range(len(source.shape))))
        kept = [d for d in range(len(source.shape)) if d not in over]
        made, kind = self.fresh(value.dtype, value.shape), self.types[value.dtype]
        bind = Bound()
        out, count = bind(made, True), math.prod(source.shape[d] for d in over)
        lines, outer = unravel("p.x", [source.shape[d] for d in kept], "r")
        within, inner = unravel("j", [source.shape[d] for d in over], "q")
        index = dict(zip(kept + over, outer + inner))
        x = f"{kind}({source.read(bind, [index[d] for d in range(len(source.shape))])})"
        lines += [f"uint j = 0; {' '.join(within)}", f"{kind} a = {first.format(x=x)};",
                  f"for (j = {int(first == '{x}')}; j < {count}; j++) {{ "
                  f"{' '.join(w.removeprefix('uint ') for w in within)} a = {step.format(x=x)}; }}",
                  f"{out}[p.x] = {last.format(n=count)};"]
        self.dispatch(bind, lines, (math.prod(value.shape), 1, 1))
        return made

    def layer_norm(self, node):
        source, inner, weight, bias, epsilon = node.args
        source, values = self.reader(source), node.meta["val"]
        weight, bias = (o if o is None else self.reader(o) for o in (weight, bias))
        lead, count = source.shape[:len(source.shape) - len(inner)], math.prod(inner)
        used = {u.args[1] for u in node.users}
        made = [self.fresh(v.dtype, v.shape) if place in used else None for place, v in enumerate(values)]
        bind = Bound()
        outs = [m and bind(m, True) for m in made]
        lines, names = unravel("p.x", lead, "r")
        within, inside = unravel("j", inner, "q")
        x = source.read(bind, names + inside)
        loop = f"for (uint j = 0; j < {count}; j++) {{ {' '.join(within)}"
        wide = self.accumulated(source.dtype)
        scaled = f"({wide}({x}) - m) * r"
        scaled += f" * {weight.read(bind, inside)}" * (weight is not None)
        scaled += f" + {bias.read(bind, inside)}" * (bias is not None)
        lines += [f"{wide} m = 0, v = 0;", f"{loop} m += {wide}({x}); }}", f"m /= {count};",
                  f"{loop} {wide} d = {wide}({x}) - m; v += d * d; }}",
                  f"{wide} r = rsqrt(v / {count} + {wide}({literal(float(epsilon), wide == 'double')}));"]
        lines += [f"{loop} {outs[0]}[p.x * {count} + j] = {scaled}; }}"] * (0 in used)
        lines += [f"{outs[1]}[p.x] = m;"] * (1 in used) + [f"{outs[2]}[p.x] = r;"] * (2 in used)
        self.dispatch(bind, lines, (math.prod(lead), 1, 1))
        return made

    def convolution(self, node):
        source, weight, bias, stride, padding, dilation, transposed, _, groups = node.args
        value = node.meta["val"]
        if transposed or len(stride) != 2:
            self.missing.append(f"{node.target} (transposed or not two-dimensional)")
            return None
        source, weight = self.reader(source), self.reader(weight)
        bias = bias if bias is None else self.reader(bias)
        made = self.fresh(value.dtype, value.shape)
        bind = Bound()
        out = bind(made, True)
        lines, (n, o, y, x) = unravel("p.x", value.shape, "o")
        _, per, high, wide = weight.shape
        taken = source.read(bind, [n, f"({o} / {value.shape[1] // groups} * {per} + c)", "uint(y)", "uint(x)"])
        kind = self.accumulated(value.dtype)
        lines += [f"{kind} a = {bias.read(bind, [o]) if bias is not None else '0'};",
                  f"for (uint c = 0; c < {per}; c++) for (uint u = 0; u < {high}; u++) "
                  f"for (uint v = 0; v < {wide}; v++) {{",
                  f"    int y = int({y} * {stride[0]} + u * {dilation[0]}) - {padding[0]};",
                  f"    int x = int({x} * {stride[1]} + v * {dilation[1]}) - {padding[1]};",
                  f"    if (y >= 0 && y < {source.shape[2]} && x >= 0 && x < {source.shape[3]})" * any(padding),
                  f"    a += {kind}({taken}) * {kind}({weight.read(bind, [o, 'c', 'u', 'v'])});", "}",
                  f"{out}[p.x] = a;"]
        self.dispatch(bind, lines, (math.prod(value.shape), 1, 1))
        return made

    def escape(self, node, laid=False):
        value, held = node.meta["val"], self.view(self.forms[node])
        view = held
        if laid and any(n > 1 and a != b for n, a, b in zip(view.shape, view.stride, value.stride())):
            self.storages.append(Storage(HELD, view.dtype, 0))
            view = View(len(self.storages) - 1, view.dtype, view.shape, tuple(value.stride()), 0)
            self.storages[-1].count = 1 + sum((n - 1) * s for n, s in zip(view.shape, view.stride))
            self.store(held, view)
        storage = self.storages[view.storage]
        storage.kind = RESULT if storage.kind == HELD else storage.kind
        return view

    def flush(self):
        if self.pending:
            self.program.append((COMMIT, Segment(self.pending)))
            self.pending = []

    def host(self, node, accesses):
        self.hosted.append((len(self.vertices), accesses))
        self.vertices.append((node.name,))
        return len(self.vertices) - 1

    def receive(self, node):
        self.flush()
        view = self.arrived(RECEIVED, node)
        self.program.append((RECV, view.storage, node.args[0], self.host(node, [(view, True)])))
        return view

    def send(self, node):
        sent = self.escape(node.args[0], node.args[1] in self.laid)
        self.flush()
        self.program.append((SEND, sent, node.args[1], self.host(node, [(sent, False)])))

    # design/torch-source-lowering-change-request.md, item 3; design/compilation-composition-change-request.md, items 2
    # and 6; design/synchronization-signature-change-request.md, item 1, "It holds"; Bhattacharyya, Sriram and Lee, IEEE
    # Trans. Signal Processing 45(6), 1997; Apple, MTL4CommandEncoder.h
    def ordered(self):
        segments = [entry[1] for entry in self.program if entry[0] is COMMIT]
        within = {step.vertex: k for k, segment in enumerate(segments) for step in segment.steps}
        accesses = [(step.vertex, view, write) for segment in segments for step in segment.steps
                    for view, write in step.accesses]
        accesses += [(vertex, view, write) for vertex, views in self.hosted for view, write in views]
        edges, causes, redundant = mesh_graph.dependences_views(len(self.vertices), [
            (view.storage, view.offset * view.dtype.itemsize, view.dtype.itemsize,
             [(n, step * view.dtype.itemsize) for n, step in zip(view.shape, view.stride)], vertex, write)
            for vertex, view, write in accesses if all(view.shape)])
        commit = [within.get(vertex) for vertex in range(len(self.vertices))]
        self.dependences = [Dependence(a, b, causes[a, b]) for a, b in edges]
        found = {(d.source, d.target): d for d in self.dependences}
        barred, fences, waits = mesh_graph.orderings([edge for edge, covered in zip(edges, redundant)
                                                      if not covered and commit[edge[0]] is not None], commit)
        for (first, second), under in fences.items():
            segments[second].after[first] = (segments[first], [found[edge] for edge in under])
        for segment in segments:
            segment.barriers = [step.vertex in barred for step in segment.steps]
        program, waited, task, tasks, operations = [], set(), {}, [], []
        fenced, bound = {first for first, _ in fences}, set()
        for entry in self.program + [(WAIT,)]:
            due = sorted(waits.get(entry[3], {})) if entry[0] is SEND else \
                range(len(segments)) if entry[0] is WAIT else []
            due = [first for first in due if first not in waited]
            waited.update(due)
            program += [(WAIT, [segments[first] for first in due])] * bool(due) + [entry] * (entry[0] is not WAIT)
            operations += self.awaited(due, len(tasks))
            if entry[0] is COMMIT:
                steps, at = entry[1].steps, within[entry[1].steps[0].vertex]
                task.update((step.vertex, len(tasks)) for step in steps)
                given = {self.storages[s].position for step in steps for s in step.storages
                         if self.storages[s].kind == ARGUMENT} - bound
                bound |= given
                operations += self.committed(entry[1], len(tasks), at, sorted(given), at in fenced)
                tasks.append((self.lane, tuple(dict.fromkeys(name for step in steps for name in step.nodes))))
            elif entry[0] is not WAIT:
                task[entry[3]] = len(tasks)
                operations.append((THREAD, SENT if entry[0] is SEND else ARRIVED, len(tasks), entry[2], 0, 0))
                tasks.append((THREAD, self.vertices[entry[3]]))
        task[len(self.vertices) - 1] = len(tasks)
        tasks.append((THREAD, self.vertices[-1]))
        self.program = program
        self.record = {"tasks": tasks, "operations": operations, "dependences": mesh_graph.quotient(
            [task[vertex] for vertex in range(len(self.vertices))], [(d.source, d.target) for d in self.dependences])}


class Metal(Lowering):
    lane = QUEUE

    # design/synchronization-signature-change-request.md, item 1, "It holds"; rdma/mesh-metal4.m mesh_metal4_commit,
    # mesh_metal4_wait and mesh_metal4_after
    def committed(self, segment, task, at, given, fenced):
        made = [(QUEUE, "queue_wait", task, ("arrival", tuple(given)), 1, 1)] * bool(given)
        made += [(QUEUE, "fence_wait", task, ("fence", first), 1, 1) for first in segment.after]
        made += [(QUEUE, "pass_barrier", task, None, 0, 0)] * sum(segment.barriers)
        made += [(QUEUE, "fence_update", task, ("fence", at), 1, 1)] * fenced
        return made + [(QUEUE, "submit", task, None, 0, 1), (QUEUE, "host_signal", task, ("event", at), 1, 1)]

    def awaited(self, due, task):
        return [(THREAD, "host_wait", task, ("event", first), 1, 1) for first in due]

    def kernel(self, bound, lines, grid, group, head):
        parameters = [f"device {self.types[self.storages[s].dtype]} *b{k} [[buffer({k})]]"
                      for k, s in enumerate(bound)]
        parameters += ["uint3 p [[thread_position_in_grid]]", "uint3 g [[threadgroup_position_in_grid]]"]
        body = "\n    ".join(lines)
        head += "".join(HELPERS[name]() for name in sorted(bound.uses))
        made = pipeline(f"{STDLIB}{head}kernel void k({', '.join(parameters)}) {{\n    {body}\n}}\n")
        width = LIB.mesh_metal4_width(made)
        return made, (width, 1, 1) if group is None else tuple(n * width for n in group[:1]) + tuple(group[1:])

    # Apple, MPPTensorOpsMatMul2d.h:207-294, macOS 26.4 SDK
    def multiply(self, form, target):
        if form.dtype not in (torch.float32, torch.float16):
            return super().multiply(form, target)
        bind = Bound()
        bind.note(form)
        rows, columns = form.left.shape[-2], form.right.shape[-1]
        batch = form.left.shape[0] if len(form.left.shape) == 3 else 1
        out = bind(target, True)
        lines, flags, slices = [], [], []
        for name, operand, tile in (("A", form.left, "0, g.y * {T}"), ("B", form.right, "g.x * {T}, 0")):
            (m, n), (sm, sn) = operand.shape[-2:], operand.stride[-2:]
            turned = not (n == 1 or sn == 1)
            extents, pitch = ((m, n), sn) if turned else ((n, m), sm)
            step = operand.stride[0] if len(operand.shape) == 3 else 0
            lines.append(f"auto {name} = tensor({bind(operand)} + {operand.offset} + g.z * {step}, "
                         f"dextents<int, 2>({extents[0]}, {extents[1]}), array<int, 2>{{1, {max(pitch, 1)}}});")
            flags.append("true" if turned else "false")
            first, second = tile.format(T=TILE).split(", ")
            slices.append(f"auto m{name} = {name}.slice({second}, {first});" if turned else
                          f"auto m{name} = {name}.slice({first}, {second});")
        lines = [f"constexpr auto d = matmul2d_descriptor({TILE}, {TILE}, static_cast<int>(dynamic_extent), "
                 f"{flags[0]}, {flags[1]});",
                 f"matmul2d<d, execution_simdgroups<{SIMDGROUPS}>> op;"] + lines + slices
        lines += [f"auto C = tensor({out} + g.z * {rows * columns}, dextents<int, 2>({columns}, {rows}), "
                  f"array<int, 2>{{1, {columns}}});",
                  f"auto mC = C.slice(g.x * {TILE}, g.y * {TILE});"]
        index = ["g.z"] * (len(form.left.shape) == 3) + ["row", "column"]
        if form.plain:
            lines.append("op.run(mA, mB, mC);")
        else:
            expression = form.expression(
                bind, reshaped(index, form.left.shape[:-1] + form.right.shape[-1:], form.shape), "t[i]")
            lines += ["auto t = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), "
                      f"{self.types[form.dtype]}>();",
                      "for (uint16_t i = 0; i < t.get_capacity(); ++i) if (t.is_valid_element(i)) t[i] = 0;",
                      "op.run(mA, mB, t);",
                      "for (uint16_t i = 0; i < t.get_capacity(); ++i) if (t.is_valid_element(i)) {",
                      "    auto at = t.get_multidimensional_index(i);",
                      f"    uint row = min(g.y * {TILE} + uint(at[1]), {rows - 1}u);",
                      f"    uint column = min(g.x * {TILE} + uint(at[0]), {columns - 1}u);",
                      f"    t[i] = {expression};", "}", "t.store(mC);"]
        self.dispatch(bind, lines, (-(-columns // TILE), -(-rows // TILE), batch), (SIMDGROUPS, 1, 1), True,
                      PRIMITIVES)

    def function(self):
        storages = self.storages
        held = {s: LIB.mesh_metal4_buffer(max(st.count * st.dtype.itemsize, 1))
                for s, st in enumerate(storages) if st.kind == HELD}
        stream = LIB.mesh_metal4_stream((C.c_void_p * len(held))(*held.values()), len(held))
        program, retained = [], [*held.values()]
        for entry in self.program:
            if entry[0] is WAIT:
                program.append((WAIT, (C.c_void_p * len(entry[1]))(*[segment.record[0] for segment in entry[1]])))
                continue
            if entry[0] is not COMMIT:
                program.append(entry)
                continue
            segment = entry[1]
            segment.record.append(LIB.mesh_metal4_record())
            retained.append(segment.record[0])
            array, patches, kept = (mesh_metal4.Dispatch * len(segment.steps))(), [], []
            for made, step, barrier in zip(array, segment.steps, segment.barriers):
                addresses = (C.c_uint64 * len(step.storages))()
                kept.append(addresses)
                for slot, s in enumerate(step.storages):
                    if s in held:
                        addresses[slot] = LIB.mesh_metal4_address(held[s])
                    else:
                        patches.append((addresses, slot, s))
                made.pipeline, made.table = step.pipeline, LIB.mesh_metal4_table(len(step.storages))
                made.address, made.addresses, made.barrier = addresses, len(step.storages), barrier
                made.groups, made.grid, made.group = step.groups, step.grid, step.group
                retained.append(made.table)
            for earlier, _ in segment.after.values():
                if not earlier.fence:
                    earlier.fence.append(LIB.mesh_metal4_fence())
                    retained.append(earlier.fence[0])
            program.append((COMMIT, array, patches, list(dict.fromkeys(s for _, _, s in patches)), segment, kept))
        made = [(s, st) for s, st in enumerate(storages) if st.kind == RESULT and st.count]
        given = [(s, st.position) for s, st in enumerate(storages) if st.kind == ARGUMENT]
        results = self.results
        records = [(tuple(step.vertex for step in entry[1].steps), entry[1].record[0])
                   for entry in self.program if entry[0] is COMMIT]

        def run(*arguments):
            owner = {s: arguments[position] for s, position in given}
            handles, shifts, views, ordered = {}, {}, [], []
            for s, st in made:
                handles[s], shifts[s] = LIB.mesh_metal4_buffer(st.count * st.dtype.itemsize), 0
                pages = (C.c_byte * (st.count * st.dtype.itemsize)).from_address(LIB.mesh_metal4_contents(handles[s]))
                OWNED[C.addressof(pages)] = handles[s]
                weakref.finalize(pages, disowned, C.addressof(pages))
                owner[s] = torch.frombuffer(pages, dtype=st.dtype)

            def handle(s):
                if s not in handles:
                    pages, shift = owner[s].untyped_storage(), owner[s].storage_offset() * owner[s].itemsize
                    handles[s], shifts[s] = OWNED.get(pages.data_ptr(), pages.data_ptr()), shift
                    if owner[s].device.type == "cpu" and pages.data_ptr() not in OWNED:
                        first = pages.data_ptr() // mmap.PAGESIZE * mmap.PAGESIZE
                        span = -(-(pages.data_ptr() + pages.nbytes() - first) // mmap.PAGESIZE) * mmap.PAGESIZE
                        handles[s], shifts[s] = LIB.mesh_metal4_view(first, span), shift + pages.data_ptr() - first
                        views.append(handles[s])
                    if owner[s].device.type == "mps" and not ordered:
                        ordered.append(LIB.mesh_metal4_after(stream, *TORCH))
                return handles[s]

            for entry in program:
                if entry[0] is COMMIT:
                    _, array, patches, viewed, segment, _ = entry
                    buffers = (C.c_void_p * len(viewed))(*[handle(s) for s in viewed])
                    for addresses, slot, s in patches:
                        addresses[slot] = LIB.mesh_metal4_address(handles[s]) + shifts[s]
                    after = (C.c_void_p * len(segment.after))(*[earlier.fence[0]
                                                                for earlier, _ in segment.after.values()])
                    LIB.mesh_metal4_commit(stream, array, len(array), buffers, len(viewed), after, len(after),
                                           segment.fence[0] if segment.fence else None, segment.record[0])
                elif entry[0] is WAIT:
                    mesh_metal4.wait(entry[1])
                elif entry[0] is RECV:
                    owner[entry[1]] = RECV(entry[2])
                else:
                    SEND(shaped(owner, entry[1]), entry[2])
            LIB.mesh_metal4_reset(stream)
            for view in views:
                LIB.mesh_metal4_release(view)
            return tuple(shaped(owner, r) if isinstance(r, View) else r for r in results)
        run.device, run.record, run.program, run.dependences = None, self.record, self.program, self.dependences
        run.feedback = lambda: [(steps, *mesh_metal4.times(record)) for steps, record in records]
        weakref.finalize(run, released, stream, retained)
        return run



# design/torch-source-lowering-change-request.md, item 1, "on whichever Apple silicon backend it is placed, the CPU
# included"; design/compilation-composition-change-request.md, item 7
class Native(Lowering):
    types, lane = NATIVE_TYPES, THREAD

    def committed(self, segment, task, at, given, fenced):
        return []

    def awaited(self, due, task):
        return []

    def __init__(self, module, laid):
        self.kernels = []
        super().__init__(module, laid)

    def kernel(self, bound, lines, grid, group, head):
        declared = [f"{kind} *b{k} = ({kind} *)b_[{k}];"
                    for k, kind in enumerate(self.types[self.storages[s].dtype] for s in bound)]
        declared += ["for (size_t i_ = first_; i_ < last_; i_++) {",
                     f"uint3 p = uint3(uint(i_ % {grid[0]}), uint(i_ / {grid[0]} % {grid[1]}), "
                     f"uint(i_ / {grid[0] * grid[1]}));"]
        self.kernels.append("static void k%d(void *const *b_, size_t first_, size_t last_) {\n    %s\n    }\n}\n"
                            % (len(self.kernels), "\n    ".join(declared + list(lines))))
        return len(self.kernels) - 1, grid

    # Apple, vecLib cblas_new.h, cblas_sgemm and cblas_dgemm
    def multiply(self, form, target):
        if form.dtype not in (torch.float32, torch.float64):
            return super().multiply(form, target)
        bind, kind = Bound(), self.types[form.dtype]
        bind.note(form)
        rows, inner, columns = form.left.shape[-2], form.left.shape[-1], form.right.shape[-1]
        batched = len(form.left.shape) == 3
        out, given = f"{bind(target, True)} + p.x * {rows * columns}", []
        for operand in (form.left, form.right):
            (m, n), (sm, sn) = operand.shape[-2:], operand.stride[-2:]
            turned = not (n == 1 or sn == 1)
            pitch = sn if turned else sm if m > 1 else n
            given += ["CblasTrans" if turned else "CblasNoTrans",
                      f"{bind(operand)} + {operand.offset} + p.x * {operand.stride[0] if batched else 0}",
                      str(max(pitch, m if turned else n, 1))]
        lines = [f"cblas_{'s' if form.dtype == torch.float32 else 'd'}gemm(CblasRowMajor, {given[0]}, {given[3]}, "
                 f"{rows}, {columns}, {inner}, 1, {given[1]}, {given[2]}, {given[4]}, {given[5]}, 0, {out}, "
                 f"{max(columns, 1)});"]
        if not form.plain:
            index = ["p.x"] * batched + ["row", "column"]
            expression = form.expression(
                bind, reshaped(index, form.left.shape[:-1] + form.right.shape[-1:], form.shape), "t")
            lines += [f"for (uint row = 0; row < {rows}; row++) for (uint column = 0; column < {columns}; column++) {{",
                      f"    {kind} *c = {out} + row * {columns} + column;", f"    {kind} t = *c;",
                      f"    *c = {expression};", "}"]
        self.dispatch(bind, lines, (form.left.shape[0] if batched else 1, 1, 1))

    def function(self):
        storages = self.storages
        held = {s: mmap.mmap(-1, max(st.count * st.dtype.itemsize, 1))
                for s, st in enumerate(storages) if st.kind == HELD}
        placed = {s: C.addressof(C.c_char.from_buffer(pages)) for s, pages in held.items()}
        entries, program = [], []
        for entry in self.program:
            if entry[0] is WAIT:
                continue
            if entry[0] is not COMMIT:
                program.append(entry)
                continue
            slots = [s for step in entry[1].steps for s in step.storages]
            addresses = (C.c_uint64 * max(len(slots), 1))(*[placed.get(s, 0) for s in slots])
            calls, first = [], 0
            for step in entry[1].steps:
                count = math.prod(step.grid)
                chunk = max(1, -(-count // (4 * os.cpu_count())))
                calls.append(f"mesh::k{step.pipeline}((void *const *)(a + {first}), 0, {count});" if count < GRAIN else
                             f"mesh::run(mesh::k{step.pipeline}, (void *const *)(a + {first}), {count}, {chunk});")
                first += len(step.storages)
            program.append((COMMIT, len(entries), addresses, [(k, s) for k, s in enumerate(slots) if s not in placed]))
            entries.append('extern "C" void s%d(const uint64_t *a) {\n    %s\n}\n'
                           % (len(entries), "\n    ".join(calls)))
        source = NATIVE + "".join(SHARED.values()) + "".join(self.kernels) + "}\n" + "".join(entries)
        library = native(source)
        program = [(COMMIT, getattr(library, f"s{entry[1]}"), *entry[2:]) if entry[0] is COMMIT else entry
                   for entry in program]
        made = [(s, st) for s, st in enumerate(storages) if st.kind == RESULT and st.count]
        given = [(s, st.position) for s, st in enumerate(storages) if st.kind == ARGUMENT]
        results = self.results

        def run(*arguments):
            owner = {s: arguments[position] for s, position in given}
            owner.update((s, torch.empty(st.count, dtype=st.dtype)) for s, st in made)
            for entry in program:
                if entry[0] is COMMIT:
                    _, call, addresses, patches = entry
                    for slot, s in patches:
                        addresses[slot] = owner[s].data_ptr()
                    call(addresses)
                elif entry[0] is RECV:
                    owner[entry[1]] = RECV(entry[2])
                else:
                    SEND(shaped(owner, entry[1]), entry[2])
            return tuple(shaped(owner, r) if isinstance(r, View) else r for r in results)
        run.record, run.program, run.source, run.dependences = self.record, self.program, source, self.dependences
        run.held = held
        return run


def disowned(pages):
    LIB.mesh_metal4_release(OWNED.pop(pages))


def released(stream, retained):
    for made in (*retained, stream):
        LIB.mesh_metal4_release(made)


def hosted(module, device):
    module = copy.deepcopy(module)
    for node in list(module.graph.nodes):
        if node.target is ATEN._to_copy.default and node.kwargs.get("device") == device:
            source, value = node.args[0], node.meta["val"]
            if source.meta["val"].dtype == value.dtype and source.meta["val"].stride() == value.stride():
                node.replace_all_uses_with(source)
                module.graph.erase_node(node)
            else:
                node.kwargs = {**node.kwargs, "device": HOST}
                node.meta.pop("custom", None)
    module.recompile()
    return module


def lower(scheduled):
    functions, missing = {}, {}
    lowerings = {p: Metal if scheduled.devices[p].type == "mps" else Native
                 for p in scheduled.modules if p not in scheduled.engines}
    metal = [p for p, lowering in lowerings.items() if lowering is Metal]
    laid = {channel for channel, *_ in scheduled.carried} | {
        node.args[0] for p in lowerings for node in scheduled.modules[p].graph.nodes if node.target is RECV}
    for processor, module in scheduled.modules.items():
        if processor not in lowerings:
            functions[processor] = hosted(module, scheduled.devices[metal[0]]) if metal else module
            continue
        lowering = lowerings[processor](module, laid)
        if lowering.missing:
            missing[processor] = sorted(set(lowering.missing))
        else:
            functions[processor] = lowering.function()
    if missing:
        raise NotLowered("; ".join(f"{processor}: {', '.join(names)}" for processor, names in missing.items()),
                         sorted(set().union(*missing.values())))
    return functions

# design/the-compiler.md, order 4; design/torch-source-lowering-change-request.md, items 1 and 3
import copy
import ctypes as C
import math
import mmap
import operator
import pathlib
import sys
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
ARGUMENT, RECEIVED, RESULT, HELD = "argument", "received", "result", "held"
COMMIT, WAIT = "commit", "wait"
TILE, SIMDGROUPS = 32, 4
STDLIB = "#include <metal_stdlib>\nusing namespace metal;\n"
# Apple, Metal Shading Language Specification 4.1, section 2.22.3; MPPTensorOpsMatMul2d.h
PRIMITIVES = ("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
              "using namespace mpp::tensor_ops;\n")
REFERENCE = pathlib.Path(torch.__file__).parent / "include/c10/metal/special_math.h"
PIPELINES, OWNED, HOST = {}, {}, torch.device("cpu")

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
VIEWS = {ATEN.view.default, ATEN._unsafe_view.default, ATEN.reshape.default, ATEN.permute.default,
         ATEN.expand.default, ATEN.slice.Tensor, ATEN.select.int, ATEN.unsqueeze.default, ATEN.squeeze.dims,
         ATEN.squeeze.dim, ATEN.transpose.int, ATEN.alias.default}
RESHAPES = {ATEN.view.default, ATEN._unsafe_view.default, ATEN.reshape.default}
FILLED = {ATEN.full.default, ATEN.full_like.default, ATEN.scalar_tensor.default}
REDUCED = {ATEN.sum.dim_IntList: ("0", "a + {x}", "a"), ATEN.mean.dim: ("0", "a + {x}", "a / {n}"),
           ATEN.amax.default: ("{x}", "max(a, {x})", "a"), ATEN.amin.default: ("{x}", "min(a, {x})", "a"),
           ATEN.any.dim: ("false", "a || {x}", "a")}
PASSED = {"memory_format", "approximate", "dtype", "layout", "device", "pin_memory", "non_blocking"}
SCALED = {ATEN.addmm.default: {"beta", "alpha"}}


@dataclass
class View:
    storage: int
    dtype: torch.dtype
    shape: tuple
    stride: tuple
    offset: int

    def read(self, bind, index):
        terms = [f"{i} * {s}" for i, s, n in zip(index, self.stride, self.shape) if n > 1 and s]
        return f"{bind(self.storage)}[{' + '.join([str(self.offset)] * bool(self.offset) + terms) or '0'}]"


@dataclass
class Fused:
    dtype: torch.dtype
    shape: tuple
    expression: object

    def read(self, bind, index):
        return self.expression(bind, index, None)


@dataclass
class Product:
    left: View
    right: View
    dtype: torch.dtype
    shape: tuple
    expression: object


@dataclass
class Storage:
    kind: str
    dtype: torch.dtype
    count: int
    position: int = None


@dataclass
class Step:
    pipeline: int
    storages: list
    writes: set
    grid: tuple
    group: tuple
    groups: bool


@dataclass
class Segment:
    steps: list
    writes: set
    after: list
    fence: list = field(default_factory=list)


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


def literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    return "NAN" if math.isnan(value) else f"{value!r}f"


def reference(name):
    text = REFERENCE.read_text()
    start = text.index(f"template <typename T>\ninline float {name}(T x) {{")
    return text[start:text.index("\n}\n", start) + 3]


def pipeline(source):
    if source not in PIPELINES:
        PIPELINES[source] = mesh_metal4.pipeline(source, "k")
    return PIPELINES[source]


class Lowering:
    def __init__(self, module, laid):
        self.laid, self.storages, self.forms, self.missing = laid, [], {}, []
        self.pending, self.unwaited, self.program, self.results = [], [], [], ()
        handlers = {**dict.fromkeys(POINTWISE, self.pointwise), **dict.fromkeys(VIEWS, self.viewed),
                    RECV: self.receive, SEND: self.send, operator.getitem: self.part,
                    **dict.fromkeys(FILLED, self.filled), **dict.fromkeys(REDUCED, self.reduced),
                    ATEN.arange.start_step: self.range, ATEN.embedding.default: self.embedding,
                    ATEN.pow.Tensor_Scalar: self.power, ATEN.cat.default: self.concatenated,
                    ATEN._softmax.default: self.softmax,
                    ATEN.native_layer_norm.default: self.layer_norm, ATEN.convolution.default: self.convolution,
                    ATEN.mm.default: self.product, ATEN.bmm.default: self.product, ATEN.addmm.default: self.product}
        self.readers = handlers.keys() - (VIEWS - RESHAPES) - FILLED - {
            RECV, SEND, operator.getitem, ATEN.arange.start_step, ATEN.mm.default, ATEN.bmm.default,
            ATEN.addmm.default}
        position = 0
        for node in module.graph.nodes:
            if node.op == "placeholder":
                self.forms[node] = self.arrived(ARGUMENT, node, position)
                position += 1
            elif node.op == "output" and not self.missing:
                self.results = tuple(self.escape(r) if isinstance(r, torch.fx.Node) else r for r in node.args[0])
                self.flush()
                self.wait()
            elif node.op == "call_function" and (
                    node.target not in handlers or set(node.kwargs) - PASSED - SCALED.get(node.target, set())
                    or node.kwargs.get("approximate", "none") != "none"):
                self.missing.append(str(node.target))
            elif node.op == "call_function" and not self.missing:
                form = handlers[node.target](node)
                self.forms[node] = form if self.deferred(node, form) else self.settled(form)

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

    def dispatch(self, storages, lines, grid, writes, group=None, head=""):
        parameters = [f"device {TYPES[self.storages[s].dtype]} *b{k} [[buffer({k})]]"
                      for k, s in enumerate(storages)]
        parameters += ["uint3 p [[thread_position_in_grid]]", "uint3 g [[threadgroup_position_in_grid]]"]
        body = "\n    ".join(lines)
        head += reference("erf") * ("erf(" in body)
        made = pipeline(f"{STDLIB}{head}kernel void k({', '.join(parameters)}) {{\n    {body}\n}}\n")
        width = LIB.mesh_metal4_width(made)
        group = (width, 1, 1) if group is None else tuple(n * width for n in group[:1]) + tuple(group[1:])
        self.pending.append(Step(made, list(storages), writes, tuple(grid), group, head.startswith(PRIMITIVES)))

    def binder(self):
        storages = []

        def bind(storage):
            if storage not in storages:
                storages.append(storage)
            return f"b{storages.index(storage)}"
        return storages, bind

    def store(self, form, target):
        storages, bind = self.binder()
        lines, names = unravel("p.x", form.shape, "i")
        lines.append(f"{target.read(bind, names)} = {form.read(bind, names)};")
        self.dispatch(storages, lines, (math.prod(form.shape), 1, 1), {target.storage})

    # Apple, MPPTensorOpsMatMul2d.h:207-294, macOS 26.4 SDK
    def multiply(self, form, target):
        storages, bind = self.binder()
        rows, columns = form.left.shape[-2], form.right.shape[-1]
        batch = form.left.shape[0] if len(form.left.shape) == 3 else 1
        out = bind(target.storage)
        lines, flags, slices = [], [], []
        for name, operand, tile in (("A", form.left, "0, g.y * {T}"), ("B", form.right, "g.x * {T}, 0")):
            (m, n), (sm, sn) = operand.shape[-2:], operand.stride[-2:]
            turned = not (n == 1 or sn == 1)
            extents, pitch = ((m, n), sn) if turned else ((n, m), sm)
            step = operand.stride[0] if len(operand.shape) == 3 else 0
            lines.append(f"auto {name} = tensor({bind(operand.storage)} + {operand.offset} + g.z * {step}, "
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
        expression = form.expression(bind, reshaped(index, form.left.shape[:-1] + form.right.shape[-1:], form.shape),
                                     "t[i]")
        if expression == "t[i]":
            lines.append("op.run(mA, mB, mC);")
        else:
            lines += ["auto t = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();",
                      "for (uint16_t i = 0; i < t.get_capacity(); ++i) if (t.is_valid_element(i)) t[i] = 0;",
                      "op.run(mA, mB, t);",
                      "for (uint16_t i = 0; i < t.get_capacity(); ++i) if (t.is_valid_element(i)) {",
                      "    auto at = t.get_multidimensional_index(i);",
                      f"    uint row = min(g.y * {TILE} + uint(at[1]), {rows - 1}u);",
                      f"    uint column = min(g.x * {TILE} + uint(at[0]), {columns - 1}u);",
                      f"    t[i] = {expression};", "}", "t.store(mC);"]
        self.dispatch(storages, lines, (-(-columns // TILE), -(-rows // TILE), batch), {target.storage},
                      (SIMDGROUPS, 1, 1), PRIMITIVES)

    def matrix(self, node):
        view = self.view(self.forms[node])
        (m, n), (sm, sn) = view.shape[-2:], view.stride[-2:]
        if (n == 1 or sn == 1) and (m == 1 or sm >= n) or (m == 1 or sm == 1) and (n == 1 or sn >= m):
            return view
        return self.view(Fused(view.dtype, view.shape, lambda bind, index, t: view.read(bind, index)))

    def product(self, node):
        value = node.meta["val"]
        if value.dtype != torch.float32:
            self.missing.append(f"{node.target} of {value.dtype}")
            return None
        left, right = (self.matrix(a) for a in node.args[-2:])
        if node.target is not ATEN.addmm.default:
            return Product(left, right, value.dtype, tuple(value.shape), lambda bind, index, t: t)
        added, beta, alpha = self.reader(node.args[0]), node.kwargs.get("beta", 1), node.kwargs.get("alpha", 1)

        def expression(bind, index, t):
            return f"({literal(beta)} * {added.read(bind, index[len(index) - len(added.shape):])} + " \
                   f"{literal(alpha)} * {t})"
        return Product(left, right, value.dtype, tuple(value.shape), expression)

    def pointwise(self, node, template=None):
        value = node.meta["val"]
        template, kind, shape = template or POINTWISE[node.target], TYPES[value.dtype], tuple(value.shape)
        operands = [self.forms[a] if isinstance(a, torch.fx.Node) else a for a in node.args]
        if node.target is ATEN._to_copy.default and value.dtype == node.args[0].meta["val"].dtype:
            return operands[0]
        base = next((o for o in operands if isinstance(o, Product) and o.shape == shape), None)
        operands = [self.view(o) if isinstance(o, Product) and o is not base else o for o in operands]

        def expression(bind, index, t):
            return template.format(*(
                o.expression(bind, index, t) if o is base else
                o.read(bind, index[len(index) - len(o.shape):]) if isinstance(o, (View, Fused)) else literal(o)
                for o in operands), T=kind)
        return Product(base.left, base.right, value.dtype, shape, expression) if base else \
            Fused(value.dtype, shape, expression)

    def power(self, node):
        whole = float(node.args[1]).is_integer() and 1 <= node.args[1] <= 4
        return self.pointwise(node, "(" + " * ".join(["{T}({0})"] * int(node.args[1])) + ")" if whole else
                              "pow({T}({0}), {T}({1}))")

    def filled(self, node):
        value = node.meta["val"]
        return Fused(value.dtype, tuple(value.shape),
                     lambda bind, index, t: f"{TYPES[value.dtype]}({literal(node.args[-1])})")

    def range(self, node):
        value, (start, _, step) = node.meta["val"], (*node.args, 1)[:3]
        return Fused(value.dtype, tuple(value.shape), lambda bind, index, t:
                     f"{TYPES[value.dtype]}({literal(start)} + {index[0]} * {literal(step)})")

    def embedding(self, node):
        value, (table, chosen) = node.meta["val"], (self.reader(a) for a in node.args[:2])
        return Fused(value.dtype, tuple(value.shape), lambda bind, index, t:
                     table.read(bind, [f"uint({chosen.read(bind, index[:-1])})", index[-1]]))

    def viewed(self, node):
        form, shape = self.forms[node.args[0]], tuple(node.meta["val"].shape)
        if isinstance(form, Product) and node.target in RESHAPES:
            return Product(form.left, form.right, form.dtype, shape, lambda bind, index, t:
                           form.expression(bind, reshaped(index, shape, form.shape), t))
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
        storages, bind = self.binder()
        lines, names = unravel("p.x", source.shape[:dim] + source.shape[dim + 1:], "r")
        index = names[:dim] + ["k"] + names[dim:]
        out, x = made.read(bind, index), source.read(bind, index)
        loop = f"for (uint k = 0; k < {source.shape[dim]}; k++)"
        lines += ["float m = -INFINITY, s = 0;", f"{loop} m = max(m, float({x}));",
                  f"{loop} s += exp(float({x}) - m);", f"{loop} {out} = exp(float({x}) - m) / s;"]
        self.dispatch(storages, lines, (math.prod(source.shape) // source.shape[dim], 1, 1), {made.storage})
        return made

    def reduced(self, node):
        source, value = self.reader(node.args[0]), node.meta["val"]
        first, step, last = REDUCED[node.target]
        over = node.args[1] if len(node.args) > 1 else None
        over = sorted(d % len(source.shape) for d in ([over] if isinstance(over, int) else
                                                      over or range(len(source.shape))))
        kept = [d for d in range(len(source.shape)) if d not in over]
        made, kind = self.fresh(value.dtype, value.shape), TYPES[value.dtype]
        storages, bind = self.binder()
        out, count = bind(made.storage), math.prod(source.shape[d] for d in over)
        lines, outer = unravel("p.x", [source.shape[d] for d in kept], "r")
        within, inner = unravel("j", [source.shape[d] for d in over], "q")
        index = dict(zip(kept + over, outer + inner))
        x = f"{kind}({source.read(bind, [index[d] for d in range(len(source.shape))])})"
        lines += [f"uint j = 0; {' '.join(within)}", f"{kind} a = {first.format(x=x)};",
                  f"for (j = {int(first == '{x}')}; j < {count}; j++) {{ "
                  f"{' '.join(w.removeprefix('uint ') for w in within)} a = {step.format(x=x)}; }}",
                  f"{out}[p.x] = {last.format(n=count)};"]
        self.dispatch(storages, lines, (math.prod(value.shape), 1, 1), {made.storage})
        return made

    def layer_norm(self, node):
        source, inner, weight, bias, epsilon = node.args
        source, values = self.reader(source), node.meta["val"]
        weight, bias = (o if o is None else self.reader(o) for o in (weight, bias))
        lead, count = source.shape[:len(source.shape) - len(inner)], math.prod(inner)
        used = {u.args[1] for u in node.users}
        made = [self.fresh(v.dtype, v.shape) if place in used else None for place, v in enumerate(values)]
        storages, bind = self.binder()
        outs = [m and bind(m.storage) for m in made]
        lines, names = unravel("p.x", lead, "r")
        within, inside = unravel("j", inner, "q")
        x = source.read(bind, names + inside)
        loop = f"for (uint j = 0; j < {count}; j++) {{ {' '.join(within)}"
        scaled = "(float({x}) - m) * r".format(x=x)
        scaled += f" * {weight.read(bind, inside)}" * (weight is not None)
        scaled += f" + {bias.read(bind, inside)}" * (bias is not None)
        lines += ["float m = 0, v = 0;", f"{loop} m += float({x}); }}", f"m /= {count};",
                  f"{loop} float d = float({x}) - m; v += d * d; }}",
                  f"float r = rsqrt(v / {count} + {literal(float(epsilon))});"]
        lines += [f"{loop} {outs[0]}[p.x * {count} + j] = {scaled}; }}"] * (0 in used)
        lines += [f"{outs[1]}[p.x] = m;"] * (1 in used) + [f"{outs[2]}[p.x] = r;"] * (2 in used)
        self.dispatch(storages, lines, (math.prod(lead), 1, 1), {m.storage for m in made if m is not None})
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
        storages, bind = self.binder()
        out = bind(made.storage)
        lines, (n, o, y, x) = unravel("p.x", value.shape, "o")
        _, per, high, wide = weight.shape
        taken = source.read(bind, [n, f"({o} / {value.shape[1] // groups} * {per} + c)", "uint(y)", "uint(x)"])
        lines += [f"float a = {bias.read(bind, [o]) if bias is not None else '0'};",
                  f"for (uint c = 0; c < {per}; c++) for (uint u = 0; u < {high}; u++) "
                  f"for (uint v = 0; v < {wide}; v++) {{",
                  f"    int y = int({y} * {stride[0]} + u * {dilation[0]}) - {padding[0]};",
                  f"    int x = int({x} * {stride[1]} + v * {dilation[1]}) - {padding[1]};",
                  f"    if (y >= 0 && y < {source.shape[2]} && x >= 0 && x < {source.shape[3]})" * any(padding),
                  f"    a += float({taken}) * float({weight.read(bind, [o, 'c', 'u', 'v'])});", "}",
                  f"{out}[p.x] = a;"]
        self.dispatch(storages, lines, (math.prod(value.shape), 1, 1), {made.storage})
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

    # Bhattacharyya, Sriram and Lee, IEEE Trans. Signal Processing 45(6), 1997; Apple, MTL4CommandEncoder.h
    def flush(self):
        if not self.pending:
            return
        steps, self.pending = self.pending, []
        reads = [set(step.storages) - step.writes for step in steps]

        def before(k):
            return [k - 1] * bool(k) + [j for j in range(k) if steps[j].writes & reads[k]]
        stage = mesh_graph.forward(range(len(steps)), before,
                                   lambda k, found: max(found[:1] + [s + 1 for s in found[1:]], default=0))
        barriers = [bool(k) and stage[k] > stage[k - 1] for k in range(len(steps))]
        after = [earlier for earlier in self.unwaited if earlier.writes & set().union(*reads)]
        segment = Segment(list(zip(steps, barriers)), set().union(*(step.writes for step in steps)), after)
        self.unwaited.append(segment)
        self.program.append((COMMIT, segment))

    def wait(self):
        if self.unwaited:
            self.unwaited = []
            self.program.append((WAIT,))

    def receive(self, node):
        self.flush()
        view = self.arrived(RECEIVED, node)
        self.program.append((RECV, view.storage, node.args[0]))
        return view

    def send(self, node):
        sent = self.escape(node.args[0], node.args[1] in self.laid)
        self.flush()
        self.wait()
        self.program.append((SEND, sent, node.args[1]))

    def function(self):
        storages = self.storages
        held = {s: LIB.mesh_metal4_buffer(max(st.count * st.dtype.itemsize, 1))
                for s, st in enumerate(storages) if st.kind == HELD}
        stream = LIB.mesh_metal4_stream((C.c_void_p * len(held))(*held.values()), len(held))
        program, retained = [], [*held.values()]
        for entry in self.program:
            if entry[0] is not COMMIT:
                program.append(entry)
                continue
            segment = entry[1]
            array, patches, kept = (mesh_metal4.Dispatch * len(segment.steps))(), [], []
            for made, (step, barrier) in zip(array, segment.steps):
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
            for earlier in segment.after:
                if not earlier.fence:
                    earlier.fence = [LIB.mesh_metal4_fence()]
                    retained.append(earlier.fence[0])
            program.append((COMMIT, array, patches, list(dict.fromkeys(s for _, _, s in patches)), segment, kept))
        made = [(s, st) for s, st in enumerate(storages) if st.kind == RESULT and st.count]
        given = [(s, st.position) for s, st in enumerate(storages) if st.kind == ARGUMENT]
        results = self.results

        def run(*arguments):
            owner = {s: arguments[position] for s, position in given}
            handles, shifts, views = {}, {}, []
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
                return handles[s]

            def tensor(view):
                if not all(view.shape):
                    return torch.empty(view.shape, dtype=view.dtype)
                base = owner[view.storage]
                return base.as_strided(view.shape, view.stride, base.storage_offset() + view.offset)

            for entry in program:
                if entry[0] is COMMIT:
                    _, array, patches, viewed, segment, _ = entry
                    buffers = (C.c_void_p * len(viewed))(*[handle(s) for s in viewed])
                    for addresses, slot, s in patches:
                        addresses[slot] = LIB.mesh_metal4_address(handles[s]) + shifts[s]
                    after = (C.c_void_p * len(segment.after))(*[earlier.fence[0] for earlier in segment.after])
                    LIB.mesh_metal4_commit(stream, array, len(array), buffers, len(viewed), after, len(after),
                                           segment.fence[0] if segment.fence else None)
                elif entry[0] is WAIT:
                    mesh_metal4.wait(stream)
                elif entry[0] is RECV:
                    owner[entry[1]] = RECV(entry[2])
                else:
                    SEND(tensor(entry[1]), entry[2])
            for view in views:
                LIB.mesh_metal4_release(view)
            return tuple(tensor(r) if isinstance(r, View) else r for r in results)
        run.device = None
        weakref.finalize(run, released, stream, retained)
        return run


def disowned(pages):
    LIB.mesh_metal4_release(OWNED.pop(pages))


def released(stream, retained):
    LIB.mesh_metal4_wait(stream, None, 0)
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
    module.recompile()
    return module


def lower(scheduled):
    functions, missing = {}, {}
    metal = [p for p in scheduled.modules if scheduled.devices[p].type == "mps"]
    laid = {channel for channel, *_ in scheduled.carried} | {node.args[0] for p in metal for node in scheduled.modules[p].graph.nodes if node.target is RECV}
    for processor, module in scheduled.modules.items():
        if processor not in metal:
            functions[processor] = hosted(module, scheduled.devices[metal[0]]) if metal else module
            continue
        lowering = Lowering(module, laid)
        if lowering.missing:
            missing[processor] = sorted(set(lowering.missing))
        else:
            functions[processor] = lowering.function()
    if missing:
        raise NotLowered("; ".join(f"{processor}: {', '.join(names)}" for processor, names in missing.items()),
                         sorted(set().union(*missing.values())))
    return functions

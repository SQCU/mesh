# design/compilation-composition-change-request.md items 6, 6.4 and 7, and its numerical lowerer boundary
import copy
import dataclasses

import sympy
import torch
from torch._guards import detect_fake_mode
from torch._inductor.codegen.common import OpDecompositions
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.debug import DebugContext
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler, promote_types
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import NonOwningLayout, ReinterpretView, StorageBox, TensorBox
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.scheduler import ExternKernelSchedulerNode, NopKernelSchedulerNode
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import OpsWrapper, V
from torch.utils._sympy.interp import sympy_interp

from .ir import SEND, channel_of
from .operations import BOOL, CLASS, PARALLEL, REDUCTION, SCALAR, U32, _convert, _Generic, _Region


# compiler/mesh_compiler/schedule.py assemble, module.meta["writers"], Declarations.wires: the logical copies of a
# scheduled body's declared transfers, before any plan. `wires`: channel -> (declared transfer, logical wire value, its
# logical bytes, declared sent view, declared landed view). Each distinct (source, wire value, sent view, read-after-write
# writers) is one write of its members, the SENDs sharing it, in source order. Consumed by
# compiler/mesh_compiler/lower_metal.py construct (the typed program)
@dataclasses.dataclass
class Copies:
    group: dict
    index: dict
    member: dict


def copies(module, wires):
    writes, sends = {}, [node for node in module.graph.nodes if node.target is SEND and channel_of(node) in wires]
    for writer, reader in module.meta["writers"]:
        writes.setdefault(reader, set()).add(writer)
    group = {}
    for node in sends:
        _, value, _, (offset, element, pairs), _ = wires[channel_of(node)]
        group[node] = (node.args[0], value, offset, element, tuple(map(tuple, pairs)),
                       frozenset(writes.get(node, ())))
    index = {key: k for k, key in enumerate(dict.fromkeys(group.values()))}
    member = {}
    for node in sends:
        member.setdefault(index[group[node]], []).append(node)
    return Copies(group, index, member)


# torch 2.14.0 torch/_inductor/ops_handler.py OpsHandler and DefaultHandler, codegen/common.py OpDecompositions,
# virtualized.py OpsWrapper, dtype_propagation.py DtypePropagationOpsHandler and promote_types
# (output_data/compiler-workgroup-20261114/compiler/numerical59-masked-review.md, numerical60-scalar-boundary.md,
# numerical61-scheduled-producer.md): an upstream scalar operation as the metal_ir.h binary/unary/builtin instruction of
# that name over its operands in their upstream promoted dtype, else its OpDecompositions decomposition over ops
PRIMITIVE = {'add': 'add', 'sub': 'sub', 'mul': 'mul', 'truediv': 'div', 'mod': 'rem', 'bitwise_and': 'and',
             'bitwise_or': 'or', 'bitwise_xor': 'xor', 'bitwise_left_shift': 'shl', 'bitwise_right_shift': 'shr',
             'logical_and': 'and', 'logical_or': 'or', 'logical_xor': 'xor', 'and_': 'and', 'or_': 'or', 'pow': 'pow',
             'atan2': 'atan2'}
COMPARED = ('eq', 'ne', 'lt', 'gt', 'le', 'ge')
LOGICAL = ('logical_and', 'logical_or', 'logical_xor', 'and_', 'or_', 'logical_not', 'not_')
UNARY = {'neg': 'neg', 'bitwise_not': 'complement', 'logical_not': 'not', 'not_': 'not',
         **{name: name for name in ('exp', 'log', 'sqrt', 'tanh', 'abs', 'trunc', 'floor', 'ceil', 'round', 'sin',
                                    'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'asinh', 'acosh', 'atanh',
                                    'exp2', 'log2')}}


def _kind(dtype):
    return CLASS[SCALAR[dtype]]


# the scalar operations of one region `r`; `env` the region values of the index symbols sympy_interp reads, each of
# `index` (the kernel's index dtype, OpsHandler.index_expr); `dtypes` each region value's upstream dtype
class _Scalar(DefaultHandler):
    def __init__(self, r, env, index):
        self.r, self.env, self.index, self.dtypes, self.carriers = r, env, index, {}, {}

    def kind(self, x):
        return self.r.kinds[x]

    def typed(self, x, dtype):
        self.dtypes[x] = dtype
        return x

    # dtype_propagation.py DTypeVar: a value's dtype as the propagation rules read it
    def carrier(self, x):
        dtype = self.dtypes[x]
        if dtype not in self.carriers:
            self.carriers[dtype] = torch.empty([1], dtype=dtype, device='meta')
        return self.carriers[dtype]

    def as_dtype(self, x, dtype):
        if self.dtypes[x] == dtype:
            return x
        return self.typed(_convert(self.r, x, self.kind(x), _kind(dtype)), dtype)

    def _default(self, name, args, kwargs):
        r = self.r
        if name in PRIMITIVE or name in COMPARED or name in UNARY:
            carriers = [self.carrier(a) for a in args]
            dtype = getattr(DtypePropagationOpsHandler(), name)(*carriers)
            operand = torch.bool if name in LOGICAL else promote_types(carriers) if name in COMPARED else dtype
            given = [self.as_dtype(a, operand) for a in args]
            if name in UNARY:
                return self.typed(r(UNARY[name], _kind(dtype), *given), dtype)
            return self.typed(r(PRIMITIVE.get(name, name), _kind(dtype), *given), dtype)
        decomposed = getattr(OpDecompositions, name, None)
        if decomposed is not None:
            return OpsWrapper._unwrap(decomposed(*args, **kwargs))
        raise NotImplementedError('torch._inductor scalar operation %s: its target lowering is unfinished (design/'
                                  'compilation-composition-change-request.md, numerical lowerer boundary)' % name)

    def constant(self, value, dtype):
        return self.typed(self.r.const(value, _kind(dtype)), dtype)

    def to_dtype(self, x, dtype, src_dtype=None, use_compute_types=True):
        return self.as_dtype(x, dtype)

    def to_dtype_bitcast(self, x, dtype, src_dtype):
        return self.typed(self.r('bitcast', _kind(dtype), x), dtype)

    def where(self, condition, a, b):
        dtype = DtypePropagationOpsHandler().where(*[self.carrier(x) for x in (condition, a, b)])
        a, b = self.as_dtype(a, dtype), self.as_dtype(b, dtype)
        return self.typed(self.r('select', _kind(dtype), self.as_dtype(condition, torch.bool), a, b), dtype)

    # dtype_propagation.py DtypePropagationOpsHandler.index_expr, codegen/mps.py MetalOverrides.index_expr and
    # codegen/cpp.py CppOverrides.index_expr: the index in its requested dtype (ops.to_dtype)
    def index_expr(self, expr, dtype):
        return self.to_dtype(self.indexed(expr), dtype)

    def value_expr(self, expr, dtype):
        return self.to_dtype(self.indexed(expr), dtype)

    # torch/utils/_sympy/interp.py sympy_interp (its integer literals torch.int64) of an upstream index expression,
    # in the kernel's index dtype
    def indexed(self, expr):
        return sympy_interp(_Indexing(self), self.env, sympy.sympify(expr), index_dtype=self.index)


# sympy_interp's operations in an index expression: its integer literals the kernel's index dtype, else `scalar`'s
class _Indexing(DefaultHandler):
    def __init__(self, scalar):
        self.scalar = scalar

    def _default(self, name, args, kwargs):
        return getattr(self.scalar, name)(*args, **kwargs)

    def constant(self, value, dtype):
        return self.scalar.constant(value, self.scalar.index if dtype == torch.int64 else dtype)

    # interp.py _run_sympy_handler's n-ary integral sympy.Add (`sym_sum`, which DefaultHandler.__getattr__ answers)
    def sym_sum(self, args):
        made = args[0]
        for x in args[1:]:
            made = self.scalar.add(made, x)
        return made


# the retained upstream block: `node` the torch/_inductor/scheduler.py SchedulerNode (its ir node, read_writes and
# origins) whose loop_body.py LoopBody `_body` (its var_ranges and indexing_exprs) is `record` (operations._Generic)
# over `buffers` {upstream buffer name: its typed view}; `accesses` each logical access in emission order (load or
# store, the upstream buffer name, its upstream index expression, the record's input or output slot)
@dataclasses.dataclass
class Lowered:
    node: object
    record: object
    buffers: dict
    accesses: list


# loop_body.py LoopBodyBlock.__call__ (its InterpreterShim over V.ops) under this handler: each load a logical extract
# (metal_ir.h metal_ir_extract) and each store a logical scatter (metal_ir_scatter) of the buffer's storage at its
# interpreted index, in the block the upstream operation is in, an InputBuffer's index relative to its input's data
# (graph.py GraphLowering.graph_input_storage_offsets, lowering.py as_strided); a mutated buffer its scheduler.py
# mutation_real_name; OpsHandler.masked as codegen/mps.py MetalOverrides.masked's structured If (metal_ir_if_begin ..
# metal_ir_if_end), `other` in its false block
class _Loop(_Scalar):
    def __init__(self, r, symbols, buffers, index):
        kind = _kind(index)
        super().__init__(r, {}, index)
        self.env.update((s, self.typed(r.add(('convert', kind, r.add(('loop', d), U32)), kind), index))
                        for d, s in enumerate(symbols))
        self.buffers, self.views = buffers, {}
        self.inputs, self.outputs, self.accesses = [], [], []

    def storage(self, name):
        name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name not in self.views:
            t = self.buffers[name]
            self.views[name] = t.as_strided((t.untyped_storage().nbytes() // t.element_size(),), (1,), 0)
        return name, self.views[name], V.graph.graph_input_storage_offsets.get(name, 0)

    def slot(self, operands, t):
        for k, (held, _) in enumerate(operands):
            if held is t:
                return k
        operands.append((t, None))
        return len(operands) - 1

    def load(self, name, index):
        name, t, base = self.storage(name)
        k = self.slot(self.inputs, t)
        self.accesses.append(('load', name, index, k))
        dtype = DtypePropagationOpsHandler().load(name, index)
        return self.typed(self.r('extract', CLASS[SCALAR[t.dtype]], self.indexed(base + index), integer=k), dtype)

    def store(self, name, index, value, mode=None):
        if mode is not None:
            raise NotImplementedError('torch._inductor store mode %s: its target lowering is unfinished' % mode)
        name, t, base = self.storage(name)
        k = self.slot(self.outputs, t)
        self.accesses.append(('store', name, index, k))
        self.r.scatter(k, [self.indexed(base + index)], self.as_dtype(value, V.graph.get_dtype(name)), None)

    def masked(self, mask, body, other):
        r = self.r
        r.add(('if_begin', self.as_dtype(mask, torch.bool)), BOOL)
        made = body()
        values = list(made) if isinstance(made, (tuple, list)) else [made]
        r.add(('if_else', tuple(values)), BOOL)
        fallbacks = tuple(self.typed(r.const(other, self.kind(v)), self.dtypes[v]) for v in values)
        first = r.add(('if_end', fallbacks), self.kind(values[0]))
        results = [first] + [r.add(('if_result', first, k), self.kind(v)) for k, v in enumerate(values[1:], 1)]
        for x, v in zip(results, values):
            self.typed(x, self.dtypes[v])
        return tuple(results) if isinstance(made, (tuple, list)) else results[0]

    # metal_ir.h metal_ir_scatter: an output its scatters store, its yielded value unused; each yields a literal of
    # its kind, made in the function's block
    def made(self, sizes, iteration):
        r = self.r
        r.done(*[r.const(0, CLASS[SCALAR[t.dtype]]) for t, _ in self.outputs])
        return _Generic(sizes, self.inputs, self.outputs, r,
                        [PARALLEL] * iteration + [REDUCTION] * (len(sizes) - iteration))


# one upstream SchedulerNode `node` (its LoopBody `_body`) as its Lowered over `buffers`; `index` its kernel's index
# dtype (torch/_inductor/codegen/simd_kernel_features.py SIMDKernelFeatures.select_index_dtype), from its producer
def lowered(node, buffers, index):
    body = node._body
    r, symbols = _Region(), list(body.var_ranges)
    loop, iteration = _Loop(r, symbols, buffers, index), len(body.iter_vars)
    with V.set_ops_handler(loop):
        body(symbols[:iteration], symbols[iteration:], allow_same_symbol_in_index=True)
    return Lowered(node, loop.made([int(n) for n in body.var_ranges.values()], iteration), buffers, loop.accesses)


# one upstream Scheduler node's result (torch 2.14.0 torch/_inductor/scheduler.py SchedulerNode and FusedSchedulerNode,
# codegen/cpp.py OuterLoopFusedSchedulerNode outer_fused_nodes and outer_loop_fusion_depth): `scheduled` that node, its
# fusion and loop structure; `members` each member SchedulerNode in its order with its Lowered; `origins` every
# member's original body nodes; `index` its kernel index dtype (numerical62-retained-result.md)
@dataclasses.dataclass(eq=False)
class Group:
    scheduled: object
    members: list
    origins: tuple
    index: object
    place: int = 0
    node: int = 0


# the retained upstream compilation of one scheduled body: `graph` its GraphLowering (env, operations, buffers,
# graph_inputs_original, graph_input_storage_offsets and its Scheduler, nodes as fused), `copied` each body node's node
# in the lowered copy (copy.deepcopy's memo), `originals` the inverse, `held` {upstream buffer name: its typed view}
@dataclasses.dataclass
class Upstream:
    graph: object
    copied: dict
    originals: dict
    held: dict

    # the typed view of upstream buffer `name`, its scheduler.py mutation_real_name: one given (an input's, a
    # construction's value), a NonOwningLayout its view of the buffer it aliases, else its own layout over its storage
    def buffer(self, name):
        name = self.graph.scheduler.mutation_real_name.get(name, name)
        if name not in self.held:
            layout = self.graph.get_buffer(name).get_layout()
            if isinstance(layout, NonOwningLayout):
                self.held[name] = self.reinterpret(layout.view)
            else:
                storage = torch.empty(int(layout.storage_size()), dtype=layout.dtype, device='meta')
                self.held[name] = storage.as_strided([int(n) for n in layout.size], [int(n) for n in layout.stride],
                                                     int(layout.offset))
        return self.held[name]

    # an ir.py ReinterpretView's typed view: its buffer's at its layout, an InputBuffer's offset relative to its
    # input's data (graph.py graph_input_storage_offsets)
    def reinterpret(self, view):
        name, layout = view.data.get_name(), view.get_layout()
        base = self.buffer(name)
        at = int(layout.offset) + int(self.graph.graph_input_storage_offsets.get(name, 0))
        return base.as_strided([int(n) for n in layout.size], [int(n) for n in layout.stride], at)

    # body node `node`'s value as GraphLowering.env holds it, None where it is not realized
    def value(self, node):
        made = self.graph.env.get(self.copied[node])
        made = made.data if isinstance(made, TensorBox) else made
        if isinstance(made, ReinterpretView):
            return self.reinterpret(made)
        if isinstance(made, StorageBox) and made.is_realized():
            return self.buffer(made.data.get_name())
        return None

    # a value construction gave body node `node` (an explicit method's, a channel's) as its upstream buffer's view
    def given(self, node, value):
        made = self.graph.env.get(self.copied.get(node))
        made = made.data if isinstance(made, TensorBox) else made
        if isinstance(made, StorageBox) and isinstance(value, torch.Tensor):
            self.held.setdefault(made.data.get_name(), value)

    # the original body nodes of an upstream node's origins (ir.py get_origins)
    def origins(self, scheduled):
        return tuple(dict.fromkeys(self.originals[n] for member in scheduled.get_nodes() if member.node is not None
                                   for n in member.node.get_origins() if n in self.originals))

    # the Scheduler's nodes in order, each (it, its original body nodes, whether it computes a body: not an
    # ExternKernelSchedulerNode or NopKernelSchedulerNode)
    def nodes(self):
        return [(scheduled, self.origins(scheduled),
                 not isinstance(scheduled, (ExternKernelSchedulerNode, NopKernelSchedulerNode)))
                for scheduled in self.graph.scheduler.nodes]

    # the Group of upstream Scheduler node `scheduled`: its members' bodies at its kernel index dtype (codegen/simd.py
    # SIMDScheduling: simd_kernel_features.py SIMDKernelFeatures.select_index_dtype; codegen/cpp.py's cpp_utils.py
    # INDEX_TYPE int64_t)
    def group(self, scheduled):
        with V.set_graph_handler(self.graph), V.set_fake_mode(self.graph.fake_mode):
            members = list(scheduled.get_nodes())
            if isinstance(self.graph.scheduler.get_backend(scheduled.get_device()), SIMDScheduling):
                numel, reduction = (sympy_product(s) for s in scheduled.group[1])
                index = SIMDKernelFeatures(members, numel, reduction).select_index_dtype()
            else:
                index = torch.int64
            made = [(member, lowered(member, {name: self.buffer(name) for name in [
                *member.used_buffer_names(), *member.get_buffer_names()]}, index)) for member in members]
        return Group(scheduled, made, self.origins(scheduled), index)


# design/compilation-composition-change-request.md, numerical lowerer boundary: the installed upstream compiler over one
# scheduled body before any source emission (torch 2.14.0 torch/_inductor/graph.py GraphLowering run under
# debug.py DebugContext, init_wrapper_code, _update_scheduler: its Scheduler's fusion), on a copy of the body, its
# inputs the body's own placeholders (`held` their typed views by target)
def compiled(module, held):
    memo = {}
    body = copy.deepcopy(module, memo)
    copied = {node: memo[node] for node in module.graph.nodes}
    inputs = [n.meta["val"] for n in body.graph.nodes if n.op == "placeholder"]
    fake = detect_fake_mode(inputs)
    with DebugContext(), V.set_fake_mode(fake):
        graph = GraphLowering(body, example_inputs=inputs, shape_env=fake.shape_env if fake else None,
                              is_inference=True)
        with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
            graph.run(*inputs)
            graph.init_wrapper_code()
            graph._update_scheduler()
    return Upstream(graph, copied, {made: node for node, made in copied.items()},
                    {n.name: held[n.target] for n in module.graph.nodes if n.op == "placeholder" and n.target in held})


# the explicit compiled numerical methods an upstream extern of a body node selects: Mesh's own operators
# (compiler/mesh_compiler/operations.py, torch.ops.mesh) and the LAPACK drivers of operations.py _Lapack
METHODS = (torch.ops.aten._linalg_svd.default, torch.ops.aten._linalg_eigh.default, torch.ops.aten.linalg_eig.default,
           torch.ops.aten.linalg_lstsq.default)


def method(node):
    return getattr(node.target, 'namespace', None) == 'mesh' or node.target in METHODS


# compiler/mesh_compiler/ir.py Scheduled.compilation (numerical63-produced-body.md): each machine body's input roots
# (compiler/mesh_compiler/lower_metal.py placeholders) once, its upstream result over them (`compiled`), then its
# construction (lower_metal.py construct) of that result and its retained operands
def lower(scheduled):
    from .lower_metal import construct, placeholders
    made = {}
    for (key, group), operands in scheduled.compilation.items():
        module = scheduled.modules[key][group]
        roots = placeholders(list(module.graph.nodes), operands["kinds"], operands["sources"])
        supplied = compiled(module, {node.target: root[0] for node, root in roots.items()})
        made[key, group] = construct(module, roots=roots, supplied=supplied, **operands)
    return made

# design/compilation-composition-change-request.md items 1, 2, 5, 6 and 7 and its numerical lowerer boundary;
# ~/schedule-emit/schedule_emit/driver.py and runtime.py: one machine's typed program, its processors' produced results
# (compiler/mesh_compiler/lower_inductor.py lower: each processor body's retained GraphLowering, its wrapper's lines in
# their own order with its kernel calls' Kernel and its extern calls' ExternCall records, its control bodies' Inlined
# and Lifted records; a Neural Engine body's Model) projected once onto the native program (metal-microbench metal_ir.h
# metal_ir_value, metal_ir_value_view, metal_ir_upstream and its stages, metal_ir_upstream_neural, metal_ir_lapack,
# metal_ir_check, metal_ir_copy, metal_ir_repeat, metal_ir_switch), within the machine's orchestration
# (compiler/mesh_compiler/schedule.py assemble: its interface, its transfers across machines, its results);
# compiler/mesh_compiler/schedule.py Declarations, SourceRelation. Pairing:
# output_data/compiler-workgroup-20261114/compiler/numerical66f-consumer.md, numerical66h-adapter.md
import collections
import ctypes as C
import dataclasses
import functools
import itertools
import operator
import os
import tempfile
import typing
import weakref

import torch
from torch._inductor import ir as upstream
from torch._inductor.codegen import wrapper as wrapper_lines
from torch._inductor.codegen.common import RemovedArg
from torch._inductor.codegen.cpp import OuterLoopFusedKernel
from torch._inductor.codegen.wrapper_fxir import selected_kernel_call
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.loop_body import InterpreterShim
from torch._inductor.ops_handler import AtomicMode
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree
from torch.utils._sympy.numbers import is_infinite
from torch.utils._sympy.symbol import SymT, symbol_is_type

from . import lower_inductor as producer
from . import operations
from .ir import RECV, SEND, channel_of
from .lower_inductor import copies
from .metal import (CHECK_CONVERGENCE, CONSTANT, DOMAIN_INDIRECT, DOMAIN_LOOP, DOMAIN_ROOT, INDEX_EQ, INDEX_GE,
                    INDEX_GT, INDEX_LE, INDEX_LT, INDEX_NE, INPUT, LAPACK_GEEV, LAPACK_GELS, LAPACK_GELSD,
                    LAPACK_GELSS, LAPACK_GELSY, LAPACK_GESDD, LAPACK_HEEVD, NONE, OUTPUT, REGION_REPEAT,
                    REGION_SWITCH, SOURCE_EXTERNAL, SOURCE_LITERAL, SOURCE_RECEIVED, SOURCE_SENT, SOURCE_VALUE, STATE,
                    TEMPORARY, UPSTREAM_BUFFER, UPSTREAM_ERROR, UPSTREAM_SCALAR, DomainDim, RegionChild,
                    SourceOperand, UpstreamAccess, ValueInfo, _array, _Builder, _compiler, _composition, _emitted,
                    _graph, _neural, _upstream, mesh_graph)
from .operations import ANE, CPU, GPU, SCALAR
from mesh_index import Index as _Index, Operations, ranged as _ranged, _unrepresented

Root = collections.namedtuple("Root", "name kind role root value")


# torch 2.14.0 torch/fx/interpreter.py Interpreter.fetch_attr: a get_attr node's object, its qualified target's
# attributes of its owning GraphModule
def fetched(module, target):
    value = module
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


Output = collections.namedtuple("Output", "root value literal metadata kind written")
# compiler/mesh_compiler/placement.py place `input_parts`: one placed partition of an input on one processor, `input`
# the whole input's placeholder Root, `root`/`value` the partition's typed root and ValueID (its landing view of that
# input's root), `source` the original partition Node, `landing` the ValueID of its source view (its landing geometry)
# in `input.root`, None for a non-tensor
InputPart = collections.namedtuple("InputPart", "processor input root value source landing geometry")
# torch 2.14.0 torch/export/graph_signature.py OutputKind: a result the program stores into its original input's root
MUTATIONS = (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
# torch 2.14.0 torch/export/graph_signature.py InputKind; metal-microbench metal_ir.h METAL_IR_INPUT .. METAL_IR_STATE
KINDS = {InputKind.USER_INPUT: ("user_input", INPUT), InputKind.PARAMETER: ("parameter", STATE),
         InputKind.BUFFER: ("buffer", STATE), InputKind.CONSTANT_TENSOR: ("constant_tensor", CONSTANT)}
# a storage several placeholders view: its role the most mutable of theirs (state written across calls, then an
# input varying per call, then constant data)
AUTHORITY = (STATE, INPUT, CONSTANT)
# design/the-compiler.md, "A processor is a device of a machine"; metal-microbench metal_ir.h METAL_IR_GPU .. ANE
PLACES = {"cpu": CPU, "mps": GPU, "ane": ANE}
# torch 2.14.0 torch/_inductor/codegen/mps.py codegen_kernel ("constant long& {sizevar} [[buffer(i)]]") and cpp.py
# KernelArgs.cpp_argdefs ("const int64_t"): a launch's scalar argument's type
SIZEVAR = torch.int64
# compiler/mesh_compiler/operations.py Call.routine (LAPACK[packet]: the driver the native owner selects) as the native
# owner's routine (metal_ir.h METAL_IR_GESDD .. METAL_IR_GELSY: metal_native.mm LapackWorkspace32/64, LapackLayout and
# LapackMember, their workspace queries and calls). The least-squares drivers (metal-microbench 9404b7b2) share one
# prototype-order call over inputs (a, b), b max(m, n) x nrhs shared with x:
# - GELS (a, b) -> (x, info): TRANS the native's fixed 'N', so no job character;
# - GELSY (a, b, rcond) -> (x, rank, info): JPVT the native's n ILP64 integers, zeroed per batch member, and RWORK 2n
#   when complex;
# - GELSD and GELSS (a, b, rcond) -> (x, s, rank, info).
# Every workspace size (WORK's query, IWORK, RWORK) is the native owner's
NATIVE = {"gesdd": LAPACK_GESDD, "heevd": LAPACK_HEEVD, "geev": LAPACK_GEEV, "gels": LAPACK_GELS,
          "gelsy": LAPACK_GELSY, "gelsd": LAPACK_GELSD, "gelss": LAPACK_GELSS}
# torch 2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp _linalg_check_errors(info, api, is_matrix): the INFO
# boundary each LAPACK formulation calls after its routine (operations.py _lapack_singular ..)
CHECK_ERRORS = torch.ops.aten._linalg_check_errors.default
# rdma/mesh-graph.h struct mesh_graph_domain_dim lo and hi: an unbounded value range's ends
INT64 = (-(1 << 63), (1 << 63) - 1)
# torch 2.14.0 torch/_inductor/ops_handler.py StoreMode: the atomic store modes (AtomicMode, each a commuting update:
# metal_ir_upstream_access write 2); "tma" and None an ordinary store (write 1)
COMMUTING = typing.get_args(AtomicMode)


# design/compilation-composition-change-request.md items 6, 6.4 and 7; metal_ir.h struct metal_ir_source_operand
# (metal.SourceOperand): a GIVEN cause's typed operand, a value's (root, view) its view's ValueID, a literal already in
# the typed body, a channel port Root its SENT/RECEIVED logical ValueID, None external
def operand(given):
    if given is None:
        return SourceOperand(SOURCE_EXTERNAL, NONE)
    if isinstance(given, Root):
        return SourceOperand(SOURCE_SENT if given.kind == "sent" else SOURCE_RECEIVED, given.value)
    return SourceOperand(SOURCE_LITERAL, NONE) if given[0] == "literal" else SourceOperand(SOURCE_VALUE, given[1])


# the native projection: one storage root of the program (metal_ir_value: its role, element type and bytes; a
# CONSTANT's initial contents `data`, a real tensor; `zeroed` its contents zero where placed, metal_ir_value_zeroed;
# `identity` its definer's source position), its ValueID once made (`root`); a machine-local channel's received
# storage `alias` (its sent value's Storage, the byte of that value's first element there), no root of its own,
# resolved once every member is (construct `final`)
@dataclasses.dataclass(eq=False)
class Storage:
    role: int
    dtype: torch.dtype
    bytes: int
    data: object = None
    identity: int = 0
    zeroed: bool = False
    root: int = None
    alias: tuple = None


# one value's projection: the view of `storage` as `dtype` at `size` and `stride` from element `offset` (in dtype's
# elements from the storage's first byte), metal_ir_value_view's operands
Bound = collections.namedtuple("Bound", "storage dtype size stride offset")
# one wrapper name as its lines allocate, reinterpret, reuse or bind it (torch 2.14.0 wrapper.py AllocateLine,
# ReinterpretLine, ReuseLine, MultiOutputLine; a graph input, a constant): its Storage and `at`, the byte its layout
# indexes from (its element 0 before its layout's offset)
Held = collections.namedtuple("Held", "storage at")
# one produced kernel as constructed (ConstructedMachine.upstream): its operation, its Produced (the upstream
# compiler's retained result, kept with it) and Kernel, whose domain and accesses its operand carries (relations);
# `arguments` each launch argument's (its buffer names, element bytes, binding's first byte in its root, binding bytes,
# access; no names for a scalar or the error workspace) in the launch's own order
Launched = collections.namedtuple("Launched", "operation produced kernel arguments")
# one Neural Engine body as constructed (ConstructedMachine.neural): its operation, its Produced, its MLModel's saved
# package directory (`saved` its owner, removed with the machine), function and selected compute units by name, each
# input and output feature (its spec name, its metal_ir_upstream argument) and each state (its spec name, the argument
# its read views, the argument its update writes: struct metal_ir_upstream_state), metal_ir_upstream_neural's
Neural = collections.namedtuple("Neural", "operation produced package function units inputs outputs states saved")
# one original binding of a member's wrapper name as resolved before emission (ConstructedMachine.bindings):
# `occurrence` the body occurrence binding it (() its wrapper's own lines; else each enclosing control call's ExternCall
# and the Inlined or Lifted body consumed there, outermost first), `record` the original object binding it (an
# AllocateLine, ReinterpretLine, ReuseLine or MultiOutputLine; a receive's ExternKernelAllocLine; a graph input's
# lower_inductor.py Input; a control body input's Inlined or Lifted; a constant's GraphLowering), `source` the
# original object bound (the AllocateLine's node, the ReinterpretLine's or ReuseLine's reused_as, the MultiOutputLine's
# result buffer, the receive's FallbackKernel (its own result buffer), the GraphLowering's graph_inputs entry, its
# constants tensor), `value` its native value ((root, view ValueID), ("literal", value), or None)
Binding = collections.namedtuple("Binding", "processor occurrence record name source value")


@dataclasses.dataclass(eq=False)
class ConstructedMachine:
    program: int
    compiler: object
    targets: dict
    inputs: tuple
    outputs: tuple
    spec: object
    state: tuple
    received: dict
    sent: dict
    wires: dict
    operations: dict
    advanced: tuple
    copies: object
    writes: dict
    stores: dict
    # metal-microbench metal_ir.h metal_ir_source_relation's operands as constructed: the original relation, each
    # original vertex's emitted operation IDs, each original view's and GIVEN cause's typed operand; it appends, so it
    # is attached once to each final constructed program (compiler/mesh_compiler/native.py Target.related), never here
    relation: object
    vertices: tuple
    accesses: tuple
    supplied: tuple
    writers: tuple
    identity: int = 0
    # compiler/mesh_compiler/decompose.py decompose: each expanded original source node here, its expanded nodes'
    # emitted operation IDs (module.meta["decomposed"]), in emission order
    decomposed: dict = dataclasses.field(default_factory=dict)
    # compiler/mesh_compiler/operations.py Generator `state`, its words when the program draws: the initial STATE of
    # the generator root, which the native program's operand starts from (uint32 each); none drawn here yet (the
    # producer's RandomSeeds consumer is unfinished)
    initial_words: tuple = ()
    # design/compilation-composition-change-request.md items 6 and 7 (compiler/mesh_compiler/schedule.py analyse):
    # where the program advances its generator, (its state Root, operations.py Generator.counter), else ()
    generated: tuple = ()
    # compiler/mesh_compiler/schedule.py assemble `input_parts`: each original placeholder position's InputParts here
    input_parts: dict = dataclasses.field(default_factory=dict)
    # compiler/mesh_compiler/metal.py _Builder.emitted: each projection's actual emission (Emitted: its record, the
    # producer's own: a Kernel, an ExternCall, a control body's Inlined, a Neural Engine body's Produced, a machine
    # body node or landing for an interface copy; its returned operation and the operations it created), in emission
    # order
    emitted: tuple = ()
    # an interface receive landed in an input placeholder's storage at its own position (native.py `direct`): none,
    # each received value the produced body's own received buffer (lower_inductor.py recv_strided), each mutated
    # input's store its result's copy (`stores`)
    landings: dict = dataclasses.field(default_factory=dict)
    # each produced kernel's Launched, in emission order: the operand of its composition and terminal emission
    # (`emitted`, after the ports' declaration), its Produced retained through the native target
    upstream: tuple = ()
    # {processor: Produced} every member's returned upstream result as lower_inductor.py produced it (its GraphLowering,
    # AOT signature, wrapper, kernels and externs; an empty or control body's included), retained with the machine for
    # its target and the frontend's saved records
    produced: dict = dataclasses.field(default_factory=dict)
    # rdma/mesh-graph.h struct mesh_graph_context (rdma/mesh_graph.py Context): the compilation's one analysis context,
    # the same object for every machine of one compilation (driver.py compile_captured makes it once, over the whole
    # lowered result). Every produced kernel's domain and access relations are made in it (relations); the programs
    # retain it natively (metal_ir.h metal_ir_graph_context_set, metal_ir_graph_context), and a loaded archive's
    # programs share it (metal_ir_archive_load `context`)
    context: object = dataclasses.field(kw_only=True)
    # {processor: (inputs, outputs)}: each member's interface as projected, each (its lower_inductor.py Input or
    # Output, its native value: (root, view ValueID), ("literal", value), or None)
    interfaces: dict = dataclasses.field(default_factory=dict, kw_only=True)
    # metal.py _Builder.regions: each registered repeat and switch (Region: its ExternCall, kind, region index,
    # intervals, selector), in postorder
    regions: tuple = dataclasses.field(default=(), kw_only=True)
    # each control body inlining as consumed, (its Inlined, first operation, end operation), in emission order
    inlinings: tuple = dataclasses.field(default=(), kw_only=True)
    # each Neural Engine body's Neural, in emission order
    neural: tuple = dataclasses.field(default=(), kw_only=True)
    # each original binding of every member's wrapper names (Binding), in resolution order
    bindings: tuple = dataclasses.field(default=(), kw_only=True)

    def __post_init__(self):
        self.freed = weakref.finalize(self, self.compiler.metal_ir_program_free, self.program)

    def close(self):
        self.freed()


# metal-microbench metal_ir.h stages after metal_ir_construct and an instance's port declarations
# (metal_ir_value_receives, metal_ir_value_sends), before metal_ir_lower: each produced kernel of `machine`
# (ConstructedMachine.upstream) in `program` at operation offset `offset` (its metal_ir_append offset there): its native
# composition (metal_ir_upstream_composition: metal.py Composed, each page's placed origin the native's) with its one
# callable collection (metal_ir_upstream_callables, printed once by metal_ir_callables_emit: Composed.callables)
# composed by its producer (lower_inductor.py composed: a Metal Composition or a CPU Hosted) into its producer's
# terminal emission (lower_inductor.py emitted) for its first pass (metal_ir_upstream_emitted pass 0, which configures
# its pipeline or library); where it receives (later words), its later pass's terminal emission (pass 1) over the
# same collection. Each Neural Engine body (ConstructedMachine.neural) its original Core ML program
# (metal_ir_upstream_neural: its saved package, function, selected compute units and features)
def emitted(program, machine, offset=0):
    for launched in machine.upstream:
        at, kernel = launched.operation + offset, launched.kernel
        made = _composition(program, at)
        _emitted(program, at, 0, producer.emitted(kernel, producer.composed(kernel, made)))
        if made.later_words:
            later = _composition(program, at, None, 1, made.callables)
            _emitted(program, at, 1, producer.emitted(kernel, producer.composed(kernel, later)))
    for made in machine.neural:
        _neural(program, made.operation + offset, made.package, made.function, made.units, made.inputs,
                made.outputs, made.states)


# torch 2.14.0 torch/_tensor_docs.py untyped_storage; torch/_export/non_strict_utils.py fakify: one storage root a
# source storage as the export's fake values alias it (every placeholder kind in that one namespace), each placeholder
# its view at its own shape, strides and offset; a CONSTANT root's initial bytes the exported tensor's
def placeholders(nodes, kinds, sources):
    found = {}
    for node in nodes:
        value = node.meta.get("val") if node.op == "placeholder" else None
        if isinstance(value, torch.Tensor):
            found.setdefault(value.untyped_storage()._cdata, []).append(node)
    bounds = {}
    for viewed in found.values():
        roles = {KINDS[kinds.get(node.target, InputKind.USER_INPUT)][1] for node in viewed}
        role = next(role for role in AUTHORITY if role in roles)
        first = viewed[0].meta["val"]
        storage = Storage(role, first.dtype, first.untyped_storage().nbytes(),
                          sources[viewed[0].target] if role == CONSTANT else None)
        for node in viewed:
            value = node.meta["val"]
            bounds[node] = Bound(storage, value.dtype, tuple(value.shape), tuple(value.stride()),
                                 value.storage_offset())
    return bounds


# torch 2.14.0 torch/_tensor_docs.py view_as_real: a value's native element type and geometry, a complex one's real
# pairs (operations.py SCALAR and _stored)
def _typed(b):
    if b.dtype.is_complex:
        return SCALAR[b.dtype.to_real()], (*b.size, 2), (*(2 * s for s in b.stride), 1), 2 * b.offset
    return SCALAR[b.dtype], b.size, b.stride, b.offset


def _element(dtype):
    return dtype.to_real() if dtype.is_complex else dtype


# an expression's concrete value from its backed symbols' hints alone (torch 2.14.0 sizevars.py SizeVarAllocator
# guarding_hint_or_throw: no fallback or heuristic for an unbacked symbol; it raises GuardOnDataDependentSymNode)
def _hint(graph, expr):
    return int(graph.sizevars.guarding_hint_or_throw(expr))


# a layout's storage bytes (ir.py Layout.storage_size, at its guarding hint)
def _bytes(graph, layout):
    return _hint(graph, layout.storage_size()) * layout.dtype.itemsize


# the elements a view of `sizes` and `strides` spans from its first (torch.empty_strided's storage)
def _extent(sizes, strides):
    return 0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))


def _contiguous(sizes):
    return tuple(itertools.accumulate((1, *reversed(sizes[1:])), operator.mul))[::-1]


# wrapper.py MultiOutputLine `indices`, (container type, index) pairs, as one key
def _key(indices):
    return tuple(tuple(index) for index in indices)


# an extern's operands by its operator schema's argument names (ir.py FallbackKernel.unflatten_args over its inputs
# and constant arguments)
def _named(node):
    args, kwargs = node.unflatten_args(node.inputs, node.constant_args)
    return {**{a.name: v for a, v in zip(node.op_overload._schema.arguments, args)}, **kwargs}


# torch/utils/_sympy/value_ranges.py ValueRanges as mesh_graph_domain_dim and metal_ir_upstream_access [lo, hi]





# rdma/mesh_index.py Operations, with this original caller's dtype propagation
class _Ops(Operations):
    def __init__(self, index):
        super().__init__(index)
        self.rules = DtypePropagationOpsHandler()

    def arithmetic(self, name, made, *operands, result_dtype=None):
        if any(x is None for x in operands):
            return None
        return super().arithmetic(name, made, *operands, result_dtype=getattr(self.rules, name)(*operands))

    def index_expr(self, expr, dtype):
        return super().index_expr(expr, dtype, result_dtype=self.rules.index_expr(expr, dtype))

    def value_expr(self, expr, dtype):
        return super().value_expr(expr, dtype, result_dtype=self.rules.value_expr(expr, dtype))

    def constant(self, value, dtype):
        return super().constant(value, dtype, result_dtype=self.rules.constant(value, dtype))

    def to_dtype(self, x, dtype, src_dtype=None, use_compute_types=True):
        if x is None:
            return None
        return super().to_dtype(x, dtype, src_dtype, use_compute_types,
                                result_dtype=self.rules.to_dtype(x, dtype, src_dtype, use_compute_types))

    def where(self, predicate, a, b):
        if predicate is None or a is None or b is None:
            return None
        return super().where(predicate, a, b, result_dtype=self.rules.where(predicate, a, b))


# torch 2.14.0 torch/_inductor/loop_body.py LoopBody: one kernel call's one consumer of its retained original body
# results (relations' and _cpu_relations'), over `index` (the call's _Index: its domain; its env the variables of the
# scope consumed). A body's own results are made once and kept for its every occurrence (a Ranges: one kernel's
# selection of its node; a CPU proxy's scalar, vector and tail kernels each one of the same node's body): its blocks
# (its root block, then its masked subblocks), each subblock's parent (that block's ordinal and its call node, whose
# args[0] is the subblock's mask: LoopBody.bind_masked_shim), its LoopBody.bounds() value ranges by node; each block's
# one source adjacency (rdma/mesh_graph.py Graph.adjacency over its nodes, each node's all_input_nodes; a node's
# vertex its position in the block) and, of each mask node, that graph's owned reach of it (Graph.reach: the mask's
# dependency cone). An occurrence's mask values are its own: the cone run by loop_body.py InterpreterShim over the ops
# handler (_Ops), the block's other nodes None, each get_index the LoopBody's indexing expression at the occurrence's
# variables (Ranges.variables, in body.var_ranges order)
class _Bodies:
    def __init__(self, index):
        self.index, self.ops, self.bodies, self.graphs, self.cones = index, _Ops(index), {}, {}, {}

    def held(self, body):
        if body not in self.bodies:
            blocks = [body.root_block, *body.subblocks.values()]
            parents = {body.subblocks[n.target].graph: (b, n) for b, block in enumerate(blocks)
                       for n in block.graph.nodes if n.op == "call_module" and n.target in body.subblocks}
            self.bodies[body] = blocks, parents, body.bounds().get_bounds()
        return self.bodies[body]

    def cone(self, graph, node):
        if node not in self.cones:
            if graph not in self.graphs:
                self.graphs[graph] = mesh_graph.Graph.adjacency(graph.nodes, operator.attrgetter("all_input_nodes"))
            self.cones[node] = self.graphs[graph].reach([node])
        return self.cones[node]

    # the UpstreamAccess of each access of `ranges` (Ranges.accesses: method, name, index, mode, block graph, fx node,
    # then a CPU record's own field), the call's `ordinal`-th Ranges (from 1), as relations describes it:
    # `argument(ranges, method, name)` its launch argument, None where the final kernel has none (no access of it);
    # `guard(ranges, method, index)` the condition its code runs under (None none), conjoined with its known masks; its
    # index sympified (sympy_subs: a CPU vector kernel's load_seed index the int offset CppVecOverrides.load_seed passes
    # its load, loop_body.py's own index of it sympy.Integer(offset); an expression itself)
    def accesses(self, ranges, ordinal, argument, guard):
        index, ops, body = self.index, self.ops, ranges.node._body
        blocks, parents, bounds = self.held(body)
        replacements = dict(zip(body.var_ranges, itertools.chain.from_iterable(ranges.variables)))
        modules = dict.fromkeys(body.submodules, _unrepresented)
        modules["get_index"] = functools.partial(
            lambda exprs, name: sympy_subs(exprs[name], replacements), body.indexing_exprs)
        masks, made = {}, []

        def masked(graph, node):
            if node not in masks:
                cone = self.cone(graph, node)
                shim = InterpreterShim(graph, modules)
                shim.run(ops, initial_env={n: None for n in graph.nodes if n not in cone})
                value = shim.env[node]
                masks[node] = None if value is None else ops.truth(value)
            return masks[node]

        for method, name, expr, mode, block_graph, node, *_ in ranges.accesses:
            k = argument(ranges, method, name)
            if k is None:
                continue
            expr = sympy_subs(expr, {})
            value = index.interp(expr)
            lo, hi = (0, 0) if value is not None else _ranged(bounds[node.args[2]])
            known, data, source, unknown, level = guard(ranges, method, expr), False, 0, 0, block_graph
            while level is not body.root_block.graph:
                b, call = parents[level]
                level = blocks[b].graph
                mask = masked(level, call.args[0])
                identity = (ordinal << 40) | (b << 20) | self.graphs[level].at[call.args[0]]
                source = source or identity
                if mask is None:
                    data, unknown = True, unknown or identity
                else:
                    known = mask if known is None else index.and_(known, mask)
            write = 0 if method == "load" else 2 if mode in COMMUTING else 1
            made.append(UpstreamAccess(k, write, value, known, unknown or source, int(data), 0, lo, hi))
        return made


# design/compilation-composition-change-request.md, "Complete replacement workgroup"; metal-microbench metal_ir.h
# struct metal_ir_upstream `domain` and `access`: an MPS kernel's index domain and each access as its producer made it,
# in the compilation's one analysis context `context` (rdma/mesh_graph.py Context), from the kernel's retained original
# records, under its own V.graph and V.kernel (the handlers its codegen ran under):
# - the domain (mesh-graph.h mesh_graph_domain_create): each range-tree root of its active_range_trees() a ROOT over
#   [0, numel - 1] at its position, its is_reduction; each indirect symbol (MeshMetalKernel.mesh_indirect: var, size,
#   check) an INDIRECT over [0, size - 1] where checked, else its var's value range;
# - each access (Kernel.ranges, each Ranges.accesses (method, name, index, mode, graph, fx node) as MetalKernel load,
#   store and store_reduction received it): its argument the launch argument naming its buffer
#   (Scheduler.mutation_real_name, KernelArgs' outer names); an access to a buffer the kernel removed
#   (Kernel.removed_buffers, GraphLowering.removed_buffers: its store a codegen/common.py DeferredLine dropped at
#   render, no argument of its launch) none of the final kernel's; its write 0 a load, 2 a store under an atomic mode
#   (COMMUTING), else 1; its index the original index by sympy_interp into the domain at the kernel's index type (each
#   range-tree entry its own expression over its root, each size symbol at its hint), and where that is not
#   quasi-affine its LoopBody.bounds() value range at the access's index node; its guard the conjunction of every
#   known mask of the masked subblocks enclosing its block (each mask its parent block's node, its value the call's
#   one LoopBody consumer's, _Bodies), guard_data 1 where any of them reads data or is not quasi-affine, the known
#   conjuncts kept with it; its guard_source the innermost such unknown mask's identity, else the innermost mask's
#   (Ranges ordinal + 1, parent block, mask node position);
# - the launch's error workspace (kinds[k] UPSTREAM_ERROR): its whole bytes' commuting update.
# `described` each argument's (buffer names, element bytes, first byte, bytes, access) (Launched.arguments). Returns
# (domain, accesses, made): the caller frees the domain and every index made once metal_ir_upstream has returned
# (released)
def relations(context, kernel, kinds, described):
    # a CPU KernelGroup's call: its LoopNests' (_cpu_relations)
    if kernel.place == "cpu":
        return _cpu_relations(context, kernel, kinds, described)
    lib, owner, graph = _graph(), kernel.owner, kernel.graph
    roots = list(owner.active_range_trees())
    indirect = list(owner.mesh_indirect.items())
    dims = [DomainDim(DOMAIN_ROOT, k, int(root.is_reduction), 0, 0, _hint(graph, root.numel) - 1)
            for k, root in enumerate(roots)]
    for _, (var, size, check) in indirect:
        dims.append(DomainDim(DOMAIN_INDIRECT, 0, 0, 0, *((0, _hint(graph, size) - 1) if check else
                                                         _ranged(var.bounds))))
    made = C.c_void_p()
    status = lib.mesh_graph_domain_create(context.pointer, len(dims), _array(DomainDim, dims), C.byref(made))
    if status:
        raise OSError(status, "kernel %s: its index domain (mesh_graph_domain_create)" % kernel.launch[0])
    domain, env = made.value, {}

    def missing(symbol):
        entry = owner.range_tree_nodes.get(symbol)
        if entry is not None:
            return env.setdefault(symbol, index.interp(entry.expr))
        if symbol_is_type(symbol, (SymT.SIZE, SymT.PRECOMPUTED_SIZE)):
            return index.constant(_hint(graph, symbol), torch.int64)
        return None

    index = _Index(lib, domain, env, missing, owner.get_index_dtype_as_torch_dtype())
    try:
        with V.set_graph_handler(graph), V.set_kernel_handler(owner):
            for k, root in enumerate(roots):
                env[root.symbol()] = index.dim(k)
            for k, (symbol, _) in enumerate(indirect, len(roots)):
                env[symbol] = index.dim(k)
            position = {}
            for k, (names, *_) in enumerate(described):
                for name in names:
                    position.setdefault(name, k)
            accesses = [UpstreamAccess(k, 2, None, None, 0, 0, 0, 0, described[k][3] // 4 - 1)
                        for k, kind in enumerate(kinds) if kind == UPSTREAM_ERROR]
            real, bodies = kernel.scheduler.mutation_real_name, _Bodies(index)
            removed = owner.removed_buffers | graph.removed_buffers

            def argument(ranges, method, name):
                return None if name in removed else position[real.get(name, name)]

            for r, ranges in enumerate(kernel.ranges, 1):
                accesses.extend(bodies.accesses(ranges, r, argument, lambda ranges, method, expr: None))
    except BaseException:
        released(domain, index.made)
        raise
    return domain, accesses, index.made


# relations of a CPU KernelGroup's call (lower_inductor.py MeshKernelGroup Kernel.ranges: each finalize_kernel's
# (CppKernelProxy, nodes, local buffer context, local buffers)), its whole invocation, under its own V.graph and each of
# its kernels as V.kernel (the handler that kernel's codegen ran under), as relations makes an MPS kernel's:
# - the domain: each LoopLevel of each proxy's LoopNest (an OuterLoopFusedKernel's outer levels once, then each of its
#   inner LoopNests', each of a fused proxy) a LOOP over [offset, size - 1] at its hint, its is_reduction, each lane of
#   a tiled level a point of it (a vector kernel's tiling_factor lanes from its var, a tail's tail_size, a scalar tail's
#   inner loop, move_code_under_inner_loop: each within its active range); a level of no iteration (size <= offset:
#   LoopLevel.lines emits none) the point [offset, offset] and each access under it guarded false, as features' empty
#   feature, so the call's other nests (its domain their product) keep theirs; then each kernel's indirect symbols
#   (MeshCppAccesses.mesh_indirect) as relations', one of no value (a checked size 0: its index check fails wherever
#   it runs) the point [lo, lo], each access whose index it is in guarded false, so the call's other accesses keep
#   theirs. A LoopLevel's size and an active range's bounds are its kernel's
#   renamed expressions (rename_indexing: each size symbol its KernelArgs.sizevars name), each at the hint of the
#   graph's own symbols (that mapping inverted by the same sympy_subs);
# - each access each kernel of each proxy's own selection received (CppKernelProxy.kernels: scalar; vector and tail;
#   tile and its tails; each CppRanges access), as relations' (argument, write, index at the index type int64,
#   cpp_utils.py INDEX_TYPE, over the kernel's LoopLevel variables and indirect symbols, LoopBody.bounds() where not
#   quasi-affine, its masked subblocks' masks and their identities by CppRanges ordinal in the call), its known guard
#   also the condition its code runs under: in the kernel's body, each active range (codegen_conditions: var in
#   [start, end)); in its reduction_suffix (CppRanges.suffix, store_reduction), as its proxy's mesh_aggregated emits
#   that suffix after the first reduction level: kernels[0]'s and kernels[-1]'s each under its active range of
#   outer_loop.var where aggregated, else kernels[0]'s unconditioned, another kernel's not emitted. Its recorded mask
#   (the kernel's _load_mask) is the emitted conjunction of those same subblock masks, no other: its one producer
#   CppKernel.masked (conjoining the prior) is entered only by CppVecOverrides.masked, which with CppOverrides.masked
#   (a scalar kernel's `mask ? body() : other` lambda, its loads within the lambda's scoped CSE) only V.ops.masked of
#   LoopBody.bind_masked_shim calls (loop_body.py 605: a masked subblock's call, its mask that call's args[0], its body
#   that subblock run by its own InterpreterShim, the recorded graph), and its uses (the masked vector load, the
#   per-lane non-contiguous load, indirect_assert) are each within one; the code's every other condition is its
#   LoopLevels and active ranges, a tail's lanes and a scalar tail's inner loop within them.
#   An access to a local buffer (the record's local buffers: the call's own allocation, codegen_loops_impl) is no
#   argument's. A store its render drops is none: every CPU store line is a DeferredLine of its name (CppKernel,
#   CppVecKernel and CppTile2DKernel store and store_reduction), dropped where the name is removed at render (common.py
#   is_buffer_removed: GraphLowering.removed_buffers and inplaced_to_remove, flush's V.kernel the NullKernelHandler,
#   its own sets empty; CSEProxy.store and store_reduction call no kernel store of a name already removed; the graph's
#   sets after the compile as at that render: a later kernel removes only a name it stores, its defining node's,
#   remove_kernel_local_buffers, and a LocalBufferContext's restore precedes its own finalize). A load is
#   no DeferredLine, emitted as made: each takes its parameter from KernelArgs.input (common.py 1695, raising where
#   the name is removed then), the name's own argument of the group's call, its input or (stored earlier in the
#   group) its output
# Returns (domain, accesses, made) as relations does
def _cpu_relations(context, kernel, kinds, described):
    lib, owner, graph = _graph(), kernel.owner, kernel.graph
    renamed = {sympy_subs(symbol, {symbol: name}): symbol for symbol, name in owner.args.sizevars.items()}
    dims, scopes = [], []

    def hint(expr):
        return _hint(graph, sympy_subs(expr, renamed))

    def looped(loops):
        made = []
        for loop in loops or ():
            lo, hi = hint(loop.offset), hint(loop.size) - 1
            made.append((len(dims), loop, hi < lo))
            dims.append(DomainDim(DOMAIN_LOOP, 0, int(loop.is_reduction), 0, lo, max(lo, hi)))
        return made

    for proxy, _, _, local in kernel.ranges:
        nest = proxy.loop_nest
        outer = looped(nest.loops)
        if isinstance(nest.kernel, OuterLoopFusedKernel):
            paths = [(inner.kernel, outer + looped(inner.loops)) for inner in nest.kernel.inner]
        else:
            paths = [(nest.kernel, outer)]
        for fused, path in paths:
            for sub in fused.kernels:
                indirect = []
                for symbol, (var, size, check) in sub.mesh_indirect.items():
                    lo, hi = (0, hint(size) - 1) if check else _ranged(var.bounds)
                    indirect.append((symbol, len(dims), hi < lo))
                    dims.append(DomainDim(DOMAIN_INDIRECT, 0, 0, 0, lo, max(lo, hi)))
                scopes.append((fused, sub, path, indirect, local))
    made = C.c_void_p()
    status = lib.mesh_graph_domain_create(context.pointer, len(dims), _array(DomainDim, dims), C.byref(made))
    if status:
        raise OSError(status, "kernel %s: its index domain (mesh_graph_domain_create)" % kernel.launch[0])
    domain = made.value

    def missing(symbol):
        if symbol_is_type(symbol, (SymT.SIZE, SymT.PRECOMPUTED_SIZE)):
            return index.constant(_hint(graph, symbol), torch.int64)
        return None

    index = _Index(lib, domain, {}, missing, torch.int64)
    try:
        position = {}
        for k, (names, *_) in enumerate(described):
            for name in names:
                position.setdefault(name, k)
        accesses, real, bodies = [], kernel.scheduler.mutation_real_name, _Bodies(index)
        removed, r = graph.removed_buffers | graph.inplaced_to_remove, 0
        with V.set_graph_handler(graph):
            for fused, sub, path, indirect, local in scopes:
                at = {loop.var: k for k, loop, _ in path}
                index.env = {**{loop.var: index.dim(k) for k, loop, _ in path},
                             **{symbol: index.dim(k) for symbol, k, _ in indirect}}
                aggregated, outer_loop = fused.mesh_aggregated
                emitted = sub is fused.kernels[0] or aggregated and sub is fused.kernels[-1]
                first = next((j for j, (_, loop, _) in enumerate(path) if loop.is_reduction), len(path))

                def active(variables):
                    guard = None
                    for var in variables:
                        start, end = sub.active_ranges[var]
                        x = index.dim(at[var])
                        made = index.and_(index.ge(x, index.constant(hint(start), torch.int64)),
                                          index.lt(x, index.constant(hint(end), torch.int64)))
                        guard = made if guard is None else index.and_(guard, made)
                    return guard

                body_guard = index.truth(False) if any(e for _, _, e in path) else active(sub.active_ranges)
                suffix_guard = (index.truth(False) if any(e for _, _, e in path[:first]) else
                                active((outer_loop.var,)) if aggregated else None)

                # this scope's: an access in a reduction suffix (CppRanges.suffix, store_reduction) emitted only by
                # the kernels its proxy emits it of; a removed store's or a local buffer's no argument
                def argument(ranges, method, name):
                    suffix = ranges.suffix or method == "store_reduction"
                    if method != "load" and name in removed or name in local or suffix and not emitted:
                        return None
                    return position[real.get(name, name)]

                def guard(ranges, method, expr):
                    if any(e and symbol in expr.free_symbols for symbol, _, e in indirect):
                        return index.truth(False)
                    return suffix_guard if ranges.suffix or method == "store_reduction" else body_guard

                with V.set_kernel_handler(sub):
                    for ranges in sub.mesh_ranges:
                        r += 1
                        accesses.extend(bodies.accesses(ranges, r, argument, guard))
    except BaseException:
        released(domain, index.made)
        raise
    return domain, accesses, index.made


# metal_ir_upstream_neural's one invocation (Core ML reads each bound input and state and writes each bound output and
# state whole, at its bound layout: its whole logical tensor domain) as mesh-graph.h relations in `context`: its domain
# one LOOP dimension per dimension of each feature that has elements ([0, size - 1], in feature order), each feature's
# access its strided element index from its argument's first element (each dimension times its stride, of its typed
# view: a complex one's real pairs) over its own dimensions, a feature with no elements guarded false; an input read
# (write 0), an output written (1), a state read and written. `bounds` each feature's (Bound, access 1, 2 or 3).
# Returns (domain, accesses, made) as relations does
def features(context, bounds):
    lib, dims, firsts = _graph(), [], []
    for b, _ in bounds:
        _, sizes, _, _ = _typed(b)
        firsts.append(len(dims))
        if 0 not in sizes:
            dims += [DomainDim(DOMAIN_LOOP, 0, 0, 0, 0, n - 1) for n in sizes]
    made = C.c_void_p()
    status = lib.mesh_graph_domain_create(context.pointer, len(dims), _array(DomainDim, dims), C.byref(made))
    if status:
        raise OSError(status, "the Neural Engine invocation's feature domain (mesh_graph_domain_create)")
    domain = made.value
    index = _Index(lib, domain, {}, None, torch.int64)
    try:
        accesses = []
        for k, ((b, access), first) in enumerate(zip(bounds, firsts)):
            _, sizes, strides, _ = _typed(b)
            value, guard = index.constant(0, torch.int64), None
            if 0 in sizes:
                guard = index.truth(False)
            else:
                for j, stride in enumerate(strides):
                    value = index.add(value, index.mul(index.dim(first + j), index.constant(stride, torch.int64)))
            for write in {1: (0,), 2: (1,), 3: (0, 1)}[access]:
                accesses.append(UpstreamAccess(k, write, value, guard, 0, 0, 0, 0, 0))
    except BaseException:
        released(domain, index.made)
        raise
    return domain, accesses, index.made


def released(domain, made):
    lib = _graph()
    for value in made:
        lib.mesh_graph_index_free(value)
    lib.mesh_graph_domain_free(domain)


# compiler/mesh_compiler/schedule.py assemble keyed by machine, Declarations.wires: `module` one machine's body in the
# retained global task order (its orchestration: placeholders, transfers across machines, a machine-local landing's
# copy, its results), `produced` {processor: Produced} its function's produced results (lower_inductor.py lower, each
# member processor's in module.meta["members"]), `targets` {original processor: (selected kind "cpu" | "mps" | "ane",
# torch.device)}, `wires` channel -> (declared transfer, logical wire value, its logical bytes, declared sent view,
# declared landed view), `output_roles` one (OutputKind, placeholder name or None) per output leaf
# (Scheduled.exported.outputs); J. K. Salmon et al., SC 2011, §3.1: `generator` the machine's operations.Generator;
# compiler/mesh_compiler/schedule.py SourceRelation, assemble `vertices`: the source relation's correspondence here,
# each vertex's emitted operation IDs, each access view's and each GIVEN cause's typed operand, a channel its port
# Root, None where this machine holds no value of it (a value the produced kernel keeps inside itself).
#
# Each member's returned result consumed as its upstream compiler returned it, resolved before any operation is
# emitted: its GraphLowering's wrapper lines in their own order (wrapper_code.lines; an inlined control body's lines in
# its parent's wrapper, binding its names: torch 2.14.0 wrapper.py codegen_subgraph_by_inlining, its prefix binding
# each body input to its call's operand, codegen_allocation allocating a name once per wrapper), each binding recorded
# at its occurrence (ConstructedMachine.bindings). A graph input is its machine placeholder's root at its first element
# (graph_input_storage_offsets), a constant its own CONSTANT root, each memory planning line a name's storage
# (AllocateLine a TEMPORARY of its layout's storage_size; ReinterpretLine and ReuseLine a view of the storage they
# name, wrapper_fxir.py's offsets; MultiOutputLine its extern's result); each KernelCallLine its Kernel
# (Produced.kernels), each ExternCall (Produced.externs; an inlining's Inlined.externs, a lifted body's
# Lifted.externs) at its first line: a send or receive, a typed LAPACK operator (operations.py LAPACK: its Call) and its
# INFO boundary, a while_loop a native repeat region over its Inlined cond, body and cond, a Switch an indexed switch
# over its branches' Lifted bodies, an invoke_subgraph its Inlined body; each operation's operands their Bounds at its
# own line. Every other ExternKernel call has no native operation: the compile raises it. A machine-local channel's
# sent value is its send's operand (`references`), its receive that value (schedule.py assemble: a local pair erased,
# its receive the sent value): its declared view over the channel's received storage, whose alias is the sent value's
# own storage, resolved once every member is (lower_inductor.py _lower_received: the received buffer in
# never_reuse_buffers, never reused in place, the sender still holding it); one copy into the receive's own storage
# only where its declared layout is not the sent value's. A Neural Engine member is one operation, its original Core
# ML program (Model: its saved package, function, selected compute units, each feature bound to its value, its whole
# invocation's feature relations). The resolved operations are then emitted in each member's own order, the members
# in module.meta["members"] order; their
# causes are the original relation's (schedule.py SourceRelation: `vertices`, `accesses`, `supplied` here), which
# native.py Target.related joins natively (metal_ir_source_relation). Each projection direct over program values, its
# record the producer's own
def construct(module, produced, *, context, targets, kinds, sources, wires, output_roles, generator=None,
              machines=None):
    compiler = _compiler()
    builder = _Builder(compiler, 0)
    try:
        status = compiler.metal_ir_graph_context_set(builder.p, context.pointer)
        if status:
            raise OSError(status, "the program's analysis context (metal_ir_graph_context_set)")
        if machines:
            compiler.metal_ir_machine(builder.p, C.byref(machines[0]), C.byref(machines[1]))
        nodes, tasks, copied = list(module.graph.nodes), module.meta["tasks"], module.meta["copied"]
        members, remote, end = module.meta["members"], module.meta["transfers"], len(module.meta["order"])
        made = copies(module, wires)
        named = {node.target: node for node in nodes if node.op == "placeholder"}
        storages, wired, error = [], {}, []
        env, interface, inputs, landing, sending, grouped = {}, set(), [], [], [], {}
        items, associated, ports, neural_, bindings_ = [], [], {}, [], []
        bodies, references, forwards = {p: [] for p in members}, {}, {}

        def stored(role, dtype, size, data=None, identity=0, zeroed=False):
            storages.append(Storage(role, dtype, size, data, identity, zeroed))
            return storages[-1]

        for node, b in placeholders(nodes, kinds, sources).items():
            env[node] = b
            storages.extend(s for s in (b.storage,) if s not in storages)

        # each member processor body's nodes: its original (meta["copied"] inverted), its machine node standing for it
        # at its own source identity (a machine-local pair's ends erased there: None)
        originals = {p: {n: original for original, n in produced[p].module.meta["copied"].items()} for p in members}
        identities = {p: produced[p].module.meta["tasks"] for p in members}

        def vertex(p, n):
            m = copied.get(originals[p].get(n))
            return m if m is not None and m in tasks and tasks[m] == identities[p].get(n) else None

        def owner_of(p, n):
            return identities[p].get(n), None, None, vertex(p, n)

        def position(owner):
            return owner[0][0] if owner[0] is not None else end

        # an IR node's original body nodes (lower_inductor.py Produced.origins over its get_origins()), their owners and
        # the source position of the first
        def owning(p, node):
            origins = produced[p].origins(node.get_origins())
            owners = [owner_of(p, n) for n in origins]
            lead = next((o for o in owners if o[0] is not None), None)
            return origins, owners, position(lead) if lead else end

        def placed(h, layout, graph):
            return Bound(h.storage, layout.dtype, tuple(_hint(graph, n) for n in layout.size),
                         tuple(_hint(graph, n) for n in layout.stride),
                         h.at // layout.dtype.itemsize + _hint(graph, layout.offset))

        # an IR value (a buffer, a ReinterpretView of one, its TensorBox) and its own layout (a
        # MutationLayoutSHOULDREMOVE buffer its target's)
        def layout_of(x):
            while isinstance(x, (upstream.TensorBox, upstream.StorageBox)):
                x = x.data
            layout = x.get_layout()
            if isinstance(layout, upstream.MutationLayoutSHOULDREMOVE):
                return layout_of(layout.target)
            return x, layout

        # a scalar control operand's value (a selector, a condition: its converted graph's value): a tensor's Bound
        # itself, a literal an int32 CONSTANT of it
        def valued(b):
            if isinstance(b, Bound):
                return b
            data = torch.tensor([b], dtype=torch.int32)
            return Bound(stored(CONSTANT, torch.int32, 4, data), torch.int32, (), (), 0)

        # a launch's scalar argument: an int64 CONSTANT holding its converted graph's value (a sizevar's or a
        # SymbolicCallArg's numel, a literal of the converter's own: _generate_sym_node)
        def scalar(value):
            data = torch.tensor([value], dtype=SIZEVAR)
            return (UPSTREAM_SCALAR, Bound(stored(CONSTANT, SIZEVAR, SIZEVAR.itemsize, data), SIZEVAR, (1,), (1,), 0),
                    1, ())

        # a launch's buffer argument (its converted graph's operand tensor `b`, the value its call passes) bound from
        # its first element over the elements its view spans
        def binding(b):
            return Bound(b.storage, b.dtype, (_extent(b.size, b.stride),), (1,), b.offset)

        # a direct elementwise copy of Bound `given` into Bound `into` (metal.py _Builder.copy) at `place`, its source
        # position `at`, owned by `owners`, `record` the producer object it implements; encoded once the roots are made
        def copying(into_items, record, into, given, owners, place, at):
            into_items.append(("copy", (record, into, given, place, at), owners))

        # a temporary of `b`'s dtype and shape, contiguous
        def staged(b, dtype=None):
            dtype = dtype or b.dtype
            return Bound(stored(TEMPORARY, dtype, _extent(b.size, _contiguous(b.size)) * dtype.itemsize), dtype,
                         b.size, _contiguous(b.size), 0)

        # a Bound over its storage's own storage (a machine-local channel's received storage: its sent value's)
        def final(b):
            while b.storage.alias is not None:
                storage, at = b.storage.alias
                b = Bound(storage, b.dtype, b.size, b.stride, b.offset + at // b.dtype.itemsize)
            return b

        # metal.py Storage roots, made once their roles are settled (before any projection is encoded)
        def root_of(storage):
            while storage.alias is not None:
                storage = storage.alias[0]
            if storage.root is None:
                element = _element(storage.dtype)
                storage.root = builder.stored(storage.role, SCALAR[element], storage.bytes // element.itemsize,
                                              storage.identity)
                if storage.data is not None:
                    builder.attach(storage.root, storage.data)
                if storage.zeroed:
                    _upstream().metal_ir_value_zeroed(builder.p, storage.root)
            return storage.root

        def value_of(b):
            b = final(b)
            scalar_, sizes, strides, offset = _typed(b)
            return builder.viewed(root_of(b.storage), b.storage.role, scalar_, sizes, strides, offset)

        def rooted(b):
            return root_of(b.storage), value_of(b)

        # metal_ir.h metal_ir_upstream_error_workspace: the program's one error workspace (c10::metal::ErrorMessages,
        # the native's own bytes, zeroed where placed), shared by every launch's ERROR argument, and its bytes
        def workspace():
            if not error:
                v, info = _upstream().metal_ir_upstream_error_workspace(builder.p), ValueInfo()
                compiler.metal_ir_value_info(builder.p, v, C.byref(info))
                error.append((v, info.bytes))
            return error[0]

        # a compile-time attribute's tensor (a get_attr target's object, GraphModule.__init__: the owner's own, the
        # same object wherever it is copied) at its own view of its storage's one CONSTANT root (one per original
        # UntypedStorage, every attribute viewing it)
        attributes = {}

        def attribute(value):
            whole = value.untyped_storage()
            if whole._cdata not in attributes:
                attributes[whole._cdata] = stored(CONSTANT, value.dtype, whole.nbytes(), value)
            return Bound(attributes[whole._cdata], value.dtype, tuple(value.shape), tuple(value.stride()),
                         value.storage_offset())

        for node in nodes:
            if node.op == "placeholder":
                kind = KINDS[kinds.get(node.target, InputKind.USER_INPUT)][0]
                interface.add(node)
                if node not in env:
                    env[node] = node.meta.get("val")
                    inputs.append((node.target, kind, None))
                    continue
                inputs.append((node.target, kind, env[node]))
            elif node.op == "get_attr":
                value = fetched(module, node.target)
                interface.add(node)
                env[node] = attribute(value) if isinstance(value, torch.Tensor) else value

        # a declared transfer's logical wire (its one TEMPORARY storage of its logical bytes) viewed at its declared
        # landed or sent view
        def wire(channel, dtype, sizes, landed_):
            _, value, size, sent_view, arrived = wires[channel]
            offset, element, pairs = arrived if landed_ else sent_view
            if value not in wired:
                wired[value] = stored(TEMPORARY, dtype, size)
            return value, Bound(wired[value], dtype, tuple(sizes) if landed_ else tuple(n for n, _ in pairs),
                                tuple(stride // element for _, stride in pairs), offset // element)

        # an IR layout's own TEMPORARY (its storage_size) at that layout
        def allocated(graph, layout):
            return placed(Held(stored(TEMPORARY, layout.dtype, _bytes(graph, layout)), 0), layout, graph)

        # a receive's result: its schema's one Tensor, so its FallbackKernel itself (ir.py FallbackKernel.create,
        # create_direct_output: the kernel its own result buffer at its own layout, no MultiOutput), bound by its
        # ExternKernelAllocLine `line` at its body occurrence `occurrence`, to the value `given` it receives: that
        # value where its geometry is the result's layout, else the result's own root and one direct copy of `given`
        # at the receive, into the receive's own destination `into`; the receive's converted value
        def landed(p, graph, record, line, given, owner, into, occurrence):
            node = record.node
            own = placed(Held(given.storage, 0), node.get_layout(), graph)
            if (own.dtype, own.size, own.stride) == (given.dtype, given.size, given.stride):
                own = given
            else:
                own = allocated(graph, node.get_layout())
                copying(into, record, own, given, [owner], PLACES[produced[p].kind], position(owner))
            bindings_.append((p, occurrence, line, node.get_name(), node, own))
            return own

        # each channel's sent value as its send declares it, (dtype, sizes, strides): a wrapper send's value at its own
        # layout (its ExternCall SEND, a control body's included), a Neural Engine send's Output at its fake's
        sent_views = {}

        def declaring(calls):
            for record in calls:
                if getattr(record.node, "op_overload", None) is SEND:
                    given = _named(record.node)
                    _, layout = layout_of(given["value"])
                    sent_views[given["channel"]] = (layout.dtype, tuple(_hint(record.graph, n) for n in layout.size),
                                                    tuple(_hint(record.graph, n) for n in layout.stride))
                for body in (*record.inlined, *record.lifted):
                    declaring(body.externs)

        for p in members:
            if produced[p].kind == "ane":
                sends = {s.args[0]: channel_of(s) for s in produced[p].module.graph.find_nodes(
                    op="call_function", target=SEND)}
                for o in produced[p].outputs:
                    if o.sent:
                        fake = o.value.meta["val"]
                        sent_views[sends[o.node]] = (fake.dtype, tuple(fake.shape), tuple(fake.stride()))
            else:
                declaring(produced[p].externs)

        # a machine-local channel's sent value as its receive reads it: its declared view over the channel's received
        # storage (its alias the sent value's own storage, resolved once every member is)
        def received_view(channel):
            dtype, sizes, strides = sent_views[channel]
            return Bound(forwards.setdefault(channel, Storage(TEMPORARY, dtype, 0)), dtype, sizes, strides, 0)

        # a received channel at its receive (machine body RECV `m`; the receive's result as landed's): in its declared
        # logical wire, landed there; else in the received result's own root
        def received(p, graph, record, line, m, into, occurrence):
            channel, node = m.args[0], record.node
            layout = node.get_layout()
            if channel in wires:
                value_, given = wire(channel, layout.dtype, [_hint(graph, n) for n in layout.size], True)
                env[m] = landed(p, graph, record, line, given, (tasks[m], None, None, m), into, occurrence)
                landing.append((channel, value_, given))
            else:
                env[m] = allocated(graph, layout)
                bindings_.append((p, occurrence, line, node.get_name(), node, env[m]))
                landing.append((channel, None, env[m]))
            interface.add(m)
            return env[m]

        # a sent channel at its send (machine body SEND `m`, the sent value `given`): each write group's first member
        # copies the value into its declared wire view at the SEND's position and task, in the send's own destination
        # `into`; an undeclared one sends the value's own storage
        def sent(p, record, m, given, into):
            env[m] = None
            if m not in made.group:
                sending.append((m.args[1], None, given))
                return
            k = made.index[made.group[m]]
            if made.member[k][0] is m:
                value_, view = wire(m.args[1], given.dtype, None, False)
                copying(into, record, view, Bound(given.storage, given.dtype, view.size, given.stride, given.offset),
                        [(tasks[m], k, None, m)], PLACES[produced[p].kind], tasks[m][0])
                grouped[k] = value_, view
            sending.append((m.args[1], *grouped[k]))

        # a send's value: across machines at its machine SEND; a machine-local channel's its send's operand
        # (`references`), its received storage's alias
        def sending_value(p, record, channel, value, into):
            if channel in remote:
                sent(p, record, remote[channel][0], value, into)
            else:
                references[channel] = value

        # each ExternCall of a scheduler's (`calls`) and of every control body's (ExternCall.inlined and .lifted,
        # nested), in order
        def calls_of(calls):
            for call in calls:
                yield call
                for body in (*call.inlined, *call.lifted):
                    yield from calls_of(body.externs)

        # each extern's ExternCall by each wrapper line its codegen wrote (lower_inductor.py ExternCall.lines: the line
        # objects the converter's line_nodes pair with the nodes it made of them), and each lifted control body by its
        # graph (lower_inductor.py Lifted: lowered once per wrapper, a later control node of the same body graph its
        # first's; the lowered subgraph its HOP's get_attr names, wrapper_fxir.py lowered_subgraphs, that same graph)
        called_by, lifted = {}, {}
        for p in members:
            for call in calls_of(produced[p].externs):
                called_by.update((id(line), call) for line in call.lines)
                lifted.update((id(body.graph), body) for body in call.lifted)

        # a produced kernel's launch, its converted call `n` (wrapper_fxir.py selected_kernel_call: its kernel name,
        # then its KernelCallLine's own operands, then its keyword operands, as graph values `value` gives) at its line,
        # its Kernel that line's (Produced.kernels), into its destination `into`. Each operand in its call's own order
        # (cpu: KernelArgs.cpp_call_operands; mps: the launch's typed values): an outer buffer name (the converter's
        # _lookup_args: that name's graph value) bound over the graph value its call passes, its access its KernelArgs
        # membership (inplace_buffers read and written under every name of its InplacedBuffer, input_buffers read,
        # output_buffers written, both where it is both); a size argument (a sizevar, a SymbolicCallArg's numel) its
        # value's int64 CONSTANT; an mps call's error buffer (error_buf_idx, after its operands) the program's error
        # workspace, read and written. Its grid and threadgroup its threads and group_size values (a cpu call's none)
        def kernel(p, kernel_, line, n, value, into):
            args, arguments = kernel_.owner.args, []
            for operand_, b in zip(line.operands, value(n.args[1:]), strict=True):
                if not isinstance(operand_, str):
                    arguments.append(scalar(b))
                    continue
                inplaced = args.inplace_buffers.get(operand_)
                if inplaced is not None and not isinstance(inplaced, RemovedArg):
                    arguments.append((UPSTREAM_BUFFER, binding(b), 3, tuple(inplaced.other_names)))
                    continue
                access = (1 if operand_ in args.input_buffers else 0) | (2 if operand_ in args.output_buffers else 0)
                arguments.append((UPSTREAM_BUFFER, binding(b), access, (operand_,)))
            if "error_buf_idx" in n.kwargs:
                arguments.append((UPSTREAM_ERROR, None, 3, ()))
            grid = (*value(n.kwargs.get("threads", ())), 1, 1, 1)[:3]
            group = (*value(n.kwargs["group_size"]), 1, 1)[:3] if "group_size" in n.kwargs else None
            owners = [owner_of(p, origin) for origin in kernel_.origins]
            lead = next((o for o in owners if o[0] is not None), None)
            into.append(("kernel", (p, kernel_, arguments, grid, group, position(lead) if lead else end), owners))
            associated.extend(o[3] for o in owners if o[3] is not None)

        # a typed LAPACK operator (operations.py LAPACK[packet], its Call: routine, jobs, rcond, inputs, matrices,
        # info by its schema's names; `given` its converted call's operands by those names) as the native owner's one
        # call (metal_ir_lapack) in its own destination `into`: its job its jobs' str operands in order (a literal the
        # native's own, as gels trans 'N'), its rcond its float operand, its inputs its matrix operands, its outputs
        # every result in schema order but a job's (0,) placeholder of a matrix result, its batch its INFO's. Each
        # result is the original one (ir.py FallbackKernel.outputs, each MultiOutput its own TEMPORARY of its layout),
        # its kind the original's (the fakes' int32 INFO and RANK): the native owner stages INFO and RANK at its ILP64
        # width (metal_native.mm LapackLayout, metal-microbench 633068d4), converted once on its own copy-out. Its
        # workspace (WORK's query, IWORK, JPVT, RWORK) is the native owner's. Its value its results in schema order
        # (its MultiOutputLines' getitem)
        def lapacked(p, graph, record, stated, given, origins, owners, at, into):
            node = record.node
            target, schema = node.op_overload, node.op_overload._schema
            job = "".join(given[name] for name in stated.jobs)
            rcond = float(given[stated.rcond]) if stated.rcond is not None else 0.0
            ins = [given[name] for name in stated.inputs]
            bounds = {r.name: allocated(graph, out.get_layout())
                      for r, out in zip(schema.returns, node.outputs, strict=True)}
            batch = bounds[stated.info].size
            outs = [(b, len(b.size) - len(batch)) for name, b in bounds.items()
                    if name not in stated.matrices or len(b.size) - len(batch) == 2]
            into.append(("lapack", (record, NATIVE[stated.routine], job, rcond, batch,
                                    [(b, len(b.size) - len(batch)) for b in ins], outs, at), owners))
            origin = next(n for n in origins if n.target is target)
            parts = {u.args[1]: u for u in origin.users if u.target is operator.getitem}
            for k, b in enumerate(bounds.values()):
                m = vertex(p, parts[k]) if k in parts else None
                if m is not None:
                    env[m] = b
            associated.extend(o[3] for o in owners if o[3] is not None)
            return tuple(bounds.values())

        # an extern's own call, its converted node `n` (its OpOverload; `given` its operands by its schema's names, as
        # graph values) at its line `line`, `record` that line's ExternCall, into its destination `into`: a send or
        # receive (its schema's channel), across machines at this machine body's SEND or RECV
        # (module.meta["transfers"]), within the machine the local sender's value; a typed LAPACK operator; its INFO
        # boundary (metal_ir_check, its api the message, CONVERGENCE: torch.linalg.LinAlgError). Its value: a
        # receive's its landing, a LAPACK operator's its results, a send's and a check's none
        def called(p, record, line, n, given, into, occurrence):
            node, graph = record.node, record.graph
            origins, owners, at = owning(p, node)
            if n.target is SEND:
                sending_value(p, record, given["channel"], given["value"], into)
                return None
            if n.target is producer.RECV_STRIDED:
                channel = given["channel"]
                if channel in remote:
                    return received(p, graph, record, line, remote[channel][1], into, occurrence)
                origin = next(o for o in origins if o.target is RECV)
                own = landed(p, graph, record, line, received_view(channel), owner_of(p, origin), into, occurrence)
                m = vertex(p, origin)
                if m is not None:
                    env[m] = own
                return own
            if n.target is CHECK_ERRORS:
                into.append(("check", (record, given["info"], CHECK_CONVERGENCE, given["api"], at), owners))
                associated.extend(o[3] for o in owners if o[3] is not None)
                return None
            stated = operations.LAPACK.get(n.target.overloadpacket)
            if stated is None:
                raise NotImplementedError(
                    "unresolved integration: processor %s extern call %s (%s, ir.py %s) requires the native general "
                    "extern call: its OpOverload over its schema-named typed operands, its results, mutations and "
                    "aliases as its ExternKernel holds them (metal_ir.h has transport, metal_ir_lapack and "
                    "metal_ir_check)" % (p, node.get_name(), n.target, type(node).__name__))
            return lapacked(p, graph, record, stated, given, origins, owners, at, into)

        # a control call's converted HOP `n` (its operands as graph values `value` gives), `record` its line's
        # ExternCall: a while_loop its repeat region, a cond or switch its indexed switch, an invoke_subgraph its
        # body. Another HOP (while_loop_stack_output: each pass's carried values stacked, its outputs' leading extent
        # the loop's own iteration count) requires its native region
        def controlled(p, record, n, value, into, occurrence):
            if n.target is torch.ops.higher_order.while_loop:
                return repeat(p, record, n, value, into, occurrence)
            if n.target in (torch.ops.higher_order.cond, torch.ops.higher_order.switch):
                return switch(p, record, n, value, into, occurrence)
            if n.target is torch.ops.higher_order.invoke_subgraph:
                return invoked(p, record, n, value, into, occurrence)
            raise NotImplementedError(
                "unresolved integration: processor %s control call %s (%s, ir.py %s) requires its native control "
                "region (metal_ir.h has metal_ir_repeat and metal_ir_switch: no per-pass output stack)" % (
                    p, record.node.get_name(), n.target, type(record.node).__name__))

        # one control body occurrence of the call `record` within `occurrence`: its Lifted body (by the lowered
        # subgraph `sub` its HOP's get_attr names) converted (walk), its inputs bound to the operands `given`; its
        # yields and its own "inlined" item (its record kept with its encoded interval, ConstructedMachine.inlinings),
        # owned by `owners` (None: the region it is in owns it)
        def inline(p, record, sub, given, occurrence, owners=None):
            body, made_ = lifted[id(sub)], []
            yields = walk(p, sub, [(body, b) for b in given], made_, occurrence + ((record, body),))
            return yields, ("inlined", (body, made_), owners)

        # torch 2.14.0 wrapper_fxir.py _generate_while_loop (torch-inductor-aot-lowered-result.patch: the while_loop
        # HOP of its cond and body subgraphs, its carried and additional inputs, mutated_arg_indices) as one native
        # repeat region (metal_ir_repeat) of its cond and body, each its Lifted body converted once: each carried input
        # its own carried storage of its body input's layout, its initial value copied in (a carried input the body
        # mutates, ir.py WhileLoop.mutation_outputs, the input's own storage); the cond before the region and at each
        # pass's end (the one converted cond encoded at both), its result copied into the one-element I32 condition
        # each time; a pass the body, then each body output into its carried storage (one that reads a carried
        # storage staged first), then the cond. Its value each body output's (WhileLoop.create: a mutated input that
        # input, each other its MultiOutput [(list, i)]): its carried storage where its layout is that storage's,
        # else its own and one copy after the loop
        def repeat(p, record, n, value, into, occurrence):
            w, graph = record.node, record.graph
            cond, body, carried_, additional = value(n.args)
            origins, owners, at = owning(p, w)
            place, init, passed = PLACES[produced[p].kind], [], []
            mutated = {name for m in w.mutation_outputs for name in m.get_mutation_names()}
            inner, carried = list(body.graph_inputs.values()), []
            for k, b in enumerate(carried_):
                if w.carried_inputs[k].get_name() in mutated:
                    carried.append(b)
                    continue
                own = allocated(body, inner[k].get_layout())
                copying(init, record, own, b, owners, place, at)
                carried.append(own)
            given = carried + list(additional)
            (tested,), tests = inline(p, record, cond, given, occurrence)
            tested = valued(tested)
            condition = staged(tested, torch.int32)
            init.append(tests)
            copying(init, record, condition, tested, owners, place, at)
            yields, item = inline(p, record, body, given, occurrence)
            passed.append(item)
            ids, backs = {id(b.storage) for b in carried}, []
            for into_, b in zip(carried, yields, strict=True):
                if b != into_ and id(b.storage) in ids:
                    stage = staged(b)
                    copying(passed, record, stage, b, owners, place, at)
                    b = stage
                backs.append((into_, b))
            for into_, b in backs:
                if b != into_:
                    copying(passed, record, into_, b, owners, place, at)
            passed.append(tests)
            copying(passed, record, condition, tested, owners, place, at)
            into.append(("repeat", (record, init, passed, condition, [b.storage for b in carried]), owners))
            results = list(carried)
            for out in w.outputs:
                layout, k = out.get_layout(), out.indices[0][1]
                c = carried[k]
                own = placed(Held(c.storage, c.offset * c.dtype.itemsize), layout, graph)
                if (own.dtype, own.size, own.stride, own.offset) != (c.dtype, c.size, c.stride, c.offset):
                    own = allocated(graph, layout)
                    copying(into, record, own, c, owners, place, at)
                results[k] = own
            associated.extend(o[3] for o in owners if o[3] is not None)
            return results

        # torch 2.14.0 wrapper_fxir.py _generate_switch (ir.py Switch; torch.cond its HOP's (selector, true branch,
        # false branch, operands), a bool selector, the IR's branches [false, true]; torch.switch's (selector,
        # branches, operands), an integer selector) as one native indexed switch (metal_ir_switch_selector,
        # metal_ir_switch): its index the selector (a bool one copied into a one-element I32 first), each branch its
        # Lifted body converted once (its own interval) with its inputs bound to the operands, its yields copied into
        # the common results (Switch.outputs, MultiOutput [(list, i)], each its own TEMPORARY), the switch's value
        def switch(p, record, n, value, into, occurrence):
            s, graph = record.node, record.graph
            if n.target is torch.ops.higher_order.cond:
                selector, true, false, operands = value(n.args)
                branches = (false, true)
            else:
                selector, branches, operands = value(n.args)
            origins, owners, at = owning(p, s)
            place, index = PLACES[produced[p].kind], valued(selector)
            if index.dtype is torch.bool:
                given, index = index, staged(index, torch.int32)
                copying(into, record, index, given, owners, place, at)
            common, arms = [allocated(graph, out.get_layout()) for out in s.outputs], []
            for sub in branches:
                yields, item = inline(p, record, sub, operands, occurrence)
                arm = [item]
                for result, given in zip(common, yields, strict=True):
                    copying(arm, record, result, given, owners, place, at)
                arms.append((arm, yields))
            into.append(("switch", (record, index, arms, common, place, at), owners))
            associated.extend(o[3] for o in owners if o[3] is not None)
            return common

        # torch 2.14.0 wrapper_fxir.py _generate_invoke_subgraph (the invoke_subgraph HOP of its subgraph, its
        # identifier, its inputs): its Lifted body converted once (its own interval, owned by the invocation), its
        # inputs bound to the operands; its value the body's outputs (InvokeSubgraph.outputs, MultiOutput [(list, i)]:
        # its MultiOutputLines' getitem)
        def invoked(p, record, n, value, into, occurrence):
            origins, owners, at = owning(p, record.node)
            yields, item = inline(p, record, value(n.args[0]), value(n.args[2:]), occurrence, owners)
            into.append(item)
            return yields

        # one converted wrapper (torch 2.14.0 wrapper_fxir.py FxConverter.generate over `lowered`'s planned lines,
        # torch-inductor-aot-lowered-result.patch: lowered.wrapper_code.fx_converter, its gm and line_nodes) at its
        # body occurrence `occurrence`, each node in graph order into the destination `into`, each node's value a
        # program value of this occurrence (a Bound, a literal, None, a sequence of them, a lowered subgraph), each
        # binding recorded with the line whose conversion made its node (line_nodes):
        # - each placeholder (the converter's graph_inputs, in order) the value `given` pairs with its record;
        # - a get_attr a lowered subgraph (subgm_getattrs: lowered_subgraphs by name), else a constant's tensor at its
        #   own CONSTANT root (attribute), its record `lowered`;
        # - empty_strided (an AllocateLine) a TEMPORARY of its size and stride; as_strided (a ReinterpretLine, a
        #   ReuseLine, a ReinterpretView operand's) its base's storage at its size, stride and absolute storage offset;
        #   getitem (a MultiOutputLine, a key path) its value's element;
        # - selected_kernel_call its kernel's launch (its KernelCallLine's Kernel);
        # - a call on an extern's line (its ExternCall, called_by): an OpOverload its extern call (called), a HOP its
        #   control call (controlled);
        # - the output its results (its converter's own: one value for a root graph of one output, else its list).
        # Every other node (a runtime scalar's definition, arithmetic or assertion: a graph input's sym_size, the
        # converter's sympy_interp arithmetic) has no native consumer: the compile raises it
        def walk(p, lowered, given, into, occurrence):
            converter = lowered.wrapper_code.fx_converter
            lines = {n: line for line, made_ in converter.line_nodes for n in made_}
            subgraphs, values = {n: name for name, n in converter.subgm_getattrs.items()}, {}

            def value(a):
                return torch.fx.node.map_arg(a, values.__getitem__)

            placeholders_ = [n for n in converter.gm.graph.nodes if n.op == "placeholder"]
            for n, (name, source), (record, b) in zip(placeholders_, converter.graph_inputs.items(), given,
                                                      strict=True):
                values[n] = b
                bindings_.append((p, occurrence, record, name, source, b))
            for n in converter.gm.graph.nodes:
                line, target = lines.get(n), n.target
                if n.op == "placeholder":
                    continue
                if n.op == "output":
                    out = value(n.args[0])
                    return [out] if len(converter.graph_outputs) == 1 and not converter.is_subgraph else list(out)
                if n.op == "get_attr" and n in subgraphs:
                    values[n] = lowered.wrapper_code.lowered_subgraphs[subgraphs[n]]
                elif n.op == "get_attr":
                    data = fetched(converter.gm, target)
                    values[n] = attribute(data)
                    bindings_.append((p, occurrence, lowered, target, data, values[n]))
                elif target is torch.empty_strided:
                    size, stride = value(n.args)
                    dtype = n.kwargs["dtype"]
                    values[n] = Bound(stored(TEMPORARY, dtype, _extent(size, stride) * dtype.itemsize), dtype,
                                      tuple(size), tuple(stride), 0)
                    bindings_.append((p, occurrence, line, line.node.get_name(), line.node, values[n]))
                elif target is torch.as_strided:
                    base, size, stride, offset = value(n.args)
                    values[n] = Bound(base.storage, base.dtype, tuple(size), tuple(stride), offset)
                    if isinstance(line, (wrapper_lines.ReinterpretLine, wrapper_lines.ReuseLine)):
                        bindings_.append((p, occurrence, line, line.reused_as.get_name(), line.reused_as, values[n]))
                elif target is operator.getitem:
                    whole, k = value(n.args)
                    values[n] = whole[k]
                    if isinstance(line, wrapper_lines.MultiOutputLine):
                        bindings_.append((p, occurrence, line, line.result_name, lowered.get_buffer(line.result_name),
                                          values[n]))
                elif target is selected_kernel_call:
                    values[n] = None
                    kernel(p, produced[p].kernels[id(line)], line, n, value, into)
                elif id(line) in called_by and isinstance(target, torch._ops.HigherOrderOperator):
                    values[n] = controlled(p, called_by[id(line)], n, value, into, occurrence)
                elif id(line) in called_by and isinstance(target, torch._ops.OpOverload):
                    named_ = {a.name: v for a, v in zip(target._schema.arguments, value(n.args))} | value(n.kwargs)
                    values[n] = called(p, called_by[id(line)], line, n, named_, into, occurrence)
                else:
                    raise NotImplementedError(
                        "unresolved integration: processor %s converted wrapper node %s (%s, of its %s; its value %r) "
                        "has no native consumer: a runtime scalar's definition, arithmetic or assertion requires a "
                        "native runtime scalar value (metal_ir.h takes one only as an operation's affine symbol, "
                        "metal_ir_symbol)" % (p, n.name, target, type(line).__name__, n.meta.get("val")))

        # a Neural Engine member (lower_inductor.py Model): one operation, its original Core ML program saved once
        # (MLModel.save: its package, its debug-handle mapping), each declared feature bound by the converter's own
        # correspondence (Model.inputs, .outputs, .states; compiler/patches/coremltools-exir-correspondence.patch):
        # - an input its ExportedProgram user input's machine value (a placeholder's, a received channel's landing; a
        #   machine-local received input its sent value, received_view), the one the converter wrapped as rank-1 size-1
        #   (converter.py 1347-1360) that value's element at that declared shape, read;
        # - an output occurrence the result of torch's own AOT runtime over Model.aliases (export.py Captured.captured:
        #   torch/_functorch/_aot_autograd/runtime_wrappers.py _RuntimeForwardEpilogue capture_orig_inputs and finalize,
        #   make_output_handler and gen_alias_from_base over the forward output prototypes, hidden bases included, through
        #   one FakeTensorConverter seeded with each input prototype's actual fake, in the program's own fake mode,
        #   torch/_export/utils.py _detect_fake_mode_from_gm): where its regenerated tensor views an
        #   actual input's storage, that input's machine storage at its view; else one value per regenerated storage at
        #   its view, written; a forward result that is no tensor (a static leaf) that result itself;
        # - a state its buffer's machine value (its target's attribute; KINDS: an ExportedProgram BUFFER's role, the
        #   program's state across calls), read where the converter reads it (input_name_to_source_buffer_name;
        #   read_state) and written where it writes it (output_name_to_target_buffer_name; coreml_update_state), each a
        #   metal_ir_upstream_state side of its own (NONE where the converter has none).
        # The declared element types are the native projection's to check against these values (metal_neural.mm
        # EmitNeural); a declared width other than its source's is the conversion's (Model.selected), never a cast here
        def neural(p):
            from torch._export.utils import _detect_fake_mode_from_gm
            from torch._functorch._aot_autograd.runtime_wrappers import _RuntimeForwardEpilogue
            from torch._subclasses.fake_tensor import FakeTensorConverter
            from torch.multiprocessing.reductions import StorageWeakRef
            made_, model = produced[p], produced[p].model
            program = model.program
            signature = program.graph_signature
            occurring = list(zip(signature.input_specs, program.graph.find_nodes(op="placeholder"), strict=True))
            given = [node for spec, node in occurring if spec.kind == InputKind.USER_INPUT]
            lifts = {spec.target: node for spec, node in occurring if spec.kind != InputKind.USER_INPUT}
            sources = {i.value.name: env.get(copied.get(originals[p].get(i.node))) for i in made_.inputs}
            for at_, n in model.received:
                channel, placeholder = n.args[0], given[at_]
                fake = placeholder.meta["val"]
                if channel in remote:
                    m = remote[channel][1]
                    if channel in wires:
                        value_, b = wire(channel, fake.dtype, list(fake.shape), True)
                    else:
                        value_, b = None, Bound(stored(TEMPORARY, fake.dtype, _extent(fake.shape, fake.stride()) *
                                                       fake.element_size()), fake.dtype, tuple(fake.shape),
                                                tuple(fake.stride()), 0)
                    landing.append((channel, value_, b))
                    env[m] = b
                    interface.add(m)
                else:
                    b, m = received_view(channel), vertex(p, n)
                    if m is not None:
                        env[m] = b
                sources[placeholder.name] = b
            held = {}

            def lifted(target):
                if target not in held:
                    held[target] = attribute(fetched(module, target))
                return held[target]

            ins = []
            for feature, name, wrapped in model.inputs:
                b = sources[name]
                ins.append((feature, Bound(b.storage, b.dtype, (1,), (1,), b.offset) if wrapped else b, 1))
            metadata, roles, prototypes, results = model.aliases
            actual, machine = [], {}
            for kind, k in roles:
                node = given[k] if kind == InputKind.USER_INPUT else lifts[k]
                value = node.meta["val"]
                actual.append(value)
                if isinstance(value, torch.Tensor):
                    machine[StorageWeakRef(value.untyped_storage())] = (
                        sources[node.name] if kind == InputKind.USER_INPUT else lifted(k), value)
            converter = FakeTensorConverter()
            for prototype, value in zip(prototypes, actual, strict=True):
                if isinstance(prototype, torch.Tensor) and isinstance(value, torch.Tensor):
                    converter.set_tensor_memo(prototype, value)
            mode = _detect_fake_mode_from_gm(program.graph_module)
            epilogue = _RuntimeForwardEpilogue(metadata, trace_joint=False, keep_input_mutations=False)
            with mode, torch.enable_grad():
                outs_ = [converter.from_real_tensor(mode, v) if isinstance(v, torch.Tensor) else v for v in results]
                forward = list(epilogue.finalize(epilogue.capture_orig_inputs(actual), outs_))
            made, regenerated = {}, {}
            for index, r in enumerate(forward):
                if not isinstance(r, torch.Tensor):
                    made[index] = r
                    continue
                key = StorageWeakRef(r.untyped_storage())
                if key in machine:
                    b, v = machine[key]
                    storage, at = b.storage, b.offset * b.dtype.itemsize - v.storage_offset() * v.element_size()
                else:
                    if key not in regenerated:
                        regenerated[key] = stored(TEMPORARY, r.dtype, r.untyped_storage().nbytes())
                    storage, at = regenerated[key], 0
                made[index] = Bound(storage, r.dtype, tuple(r.shape), tuple(r.stride()),
                                    (at + r.storage_offset() * r.element_size()) // r.element_size())
            outs = {out: made[out.position] for out in made_.outputs}
            exits = [(feature, made[index], 2) for feature, index, _ in model.outputs]
            states = []
            for feature, buffer, reads, writes in model.states:
                b = lifted(buffer)
                b.storage.role = KINDS[InputKind.BUFFER][1]
                states.append((feature, b if reads else None, b if writes else None))
            handles = dict.fromkeys(n for nodes__ in model.handles.values() for n in nodes__)
            owners = [owner_of(p, n) for n in handles]
            lead = next((o for o in owners if o[0] is not None), None)
            saved = tempfile.TemporaryDirectory()
            package = os.path.join(saved.name, "model.mlpackage")
            model.model.save(package)
            bodies[p].append(("neural", (made_, model, package, saved, (ins, exits, states),
                                         position(lead) if lead else end), owners))
            ports[p] = [(i, env.get(copied.get(originals[p].get(i.node)))) for i in made_.inputs], list(outs.items())
            sends = {o.node: b for o, b in outs.items() if o.sent}
            for s in made_.module.graph.find_nodes(op="call_function", target=SEND):
                sending_value(p, made_, channel_of(s), sends[s.args[0]], bodies[p])

        # the resolution: each member in module.meta["members"] order, a wrapper member its converted wrapper (walk, its
        # body occurrence ()), a Neural Engine member its one operation; each machine-local channel's received storage
        # then its sent value's (Storage.alias); the resolved items each member's own order. A wrapper member's
        # interface (lower_inductor.py Produced.inputs and .outputs): each graph input (the converter's graph_inputs, by
        # its Input's name) its machine placeholder's root at the input's first element (graph_input_storage_offsets;
        # the InputBuffer's layout indexes from that element), each Output its graph_outputs entry's converted result
        for p in members:
            if produced[p].kind == "ane":
                neural(p)
                continue
            graph = produced[p].graph
            converter, named_ = graph.wrapper_code.fx_converter, {i.name: i for i in produced[p].inputs}
            given = []
            for name, source in converter.graph_inputs.items():
                b = env.get(copied.get(originals[p].get(named_[name].node)))
                if isinstance(b, Bound):
                    at_ = _hint(graph, graph.graph_input_storage_offsets[name]) * b.dtype.itemsize
                    b = placed(Held(b.storage, at_), source.get_layout(), graph)
                given.append((named_[name], b))
            returned_, ported = walk(p, graph, given, bodies[p], ()), {id(i): b for i, b in given}
            at_ = {id(x): k for k, x in enumerate(converter.graph_outputs)}
            ports[p] = ([(i, ported[id(i)]) for i in produced[p].inputs],
                        [(o, returned_[at_[id(o.value)]]) for o in produced[p].outputs])
        for channel, storage in forwards.items():
            v = references[channel]
            storage.alias = (v.storage, v.offset * v.dtype.itemsize)
        items.extend(item for p in members for item in bodies[p])

        # each member's results: its interface's outputs (its returned ones), the program values of the machine nodes
        # standing for them
        results_ = {}
        for p in members:
            for out, b in ports[p][1]:
                m = copied.get(originals[p].get(out.node)) if out.node.op == "placeholder" else vertex(p, out.node)
                if not out.sent:
                    results_.setdefault(p, []).append(b)
                if m is not None and m not in env:
                    env[m] = b
        leaves, spec = pytree.tree_flatten(module.graph.output_node().args[0])
        returned = results_.get(module.meta["output"], ())

        # roles settled before any root is made: each received channel's storage INPUT, each result's OUTPUT, a
        # storage first declared keeping its role (an input's, a constant's)
        def declared(b, role):
            storage = final(b).storage
            if storage.role == TEMPORARY:
                storage.role = role

        for _, _, b in landing:
            declared(b, INPUT)
        outputs = []
        for position_, (leaf, (kind, name)) in enumerate(zip(leaves, output_roles, strict=True)):
            b = returned[position_] if isinstance(leaf, torch.fx.Node) else leaf
            if not isinstance(b, Bound):
                outputs.append((None, b, kind, None, None))
                continue
            fake = leaf.meta["val"]
            if kind not in MUTATIONS:
                declared(b, OUTPUT)
                outputs.append((b, None, kind, None, fake))
                continue
            original = env[named[name]]
            if final(original) != final(b):
                if leaf.op == "placeholder":
                    stated = named[name].meta["stated"]
                    holder = stated if isinstance(stated, str) else module.meta["output"]
                    where, owned_, identity = PLACES[targets[holder][0]], None, end
                else:
                    where, owned_, identity = PLACES[targets[leaf.meta["processor"]][0]], tasks[leaf], tasks[leaf][0]
                copying(items, leaf, original, b, [(owned_, None, position_, None)], where, identity)
            outputs.append((original, None, kind, name, fake))

        for storage in storages:
            root_of(storage)
        inputs = [Root(name, kind, None, None, None) if b is None else
                  Root(name, kind, final(b).storage.role, *rooted(b)) for name, kind, b in inputs]
        received_, sent_, ends = {}, {}, {}
        for channel, value_, b in landing:
            received_[channel] = Root(channel, "received", final(b).storage.role, *rooted(b))
            if value_ is not None:
                ends[value_] = Root(value_, "wire", final(b).storage.role, received_[channel].root, None)
        for channel, value_, b in sending:
            sent_[channel] = Root(channel, "sent", final(b).storage.role, *rooted(b))
            if value_ is not None:
                ends.setdefault(value_, Root(value_, "wire", final(b).storage.role, sent_[channel].root, None))

        # an output's metadata (compiler/mesh_compiler/native.py Bound.result): its view's shape, strides and offset
        # in the root it names, and the exported value's dtype, device and requires_grad
        def described(b, fake):
            return tuple(b.size), tuple(b.stride), final(b).offset, fake.dtype, fake.device, fake.requires_grad

        outputs = [Output(None, None, literal, None, kind, None) if b is None else
                   Output(*rooted(b), None, described(b, fake), kind, written)
                   for b, literal, kind, written, fake in outputs]

        def typed_port(b):
            return rooted(b) if isinstance(b, Bound) else None if b is None else ("literal", b)

        interfaces = {p: (tuple((i, typed_port(b)) for i, b in ins), tuple((o, typed_port(b)) for o, b in outs))
                      for p, (ins, outs) in ports.items()}
        bindings = tuple(Binding(p, occurrence, record, name, source, typed_port(b))
                         for p, occurrence, record, name, source, b in bindings_)

        # metal-microbench metal_ir.h metal_ir_operations: each projection's whole interval of created operation IDs,
        # owned by each source position it implements; each operation's operands as Bounds, in its native order
        # (inputs, then outputs), for its work axes
        operations_, advanced, writes, stores, emitted_, operands, upstream_, inlinings = {}, [], {}, {}, {}, {}, [], []

        def owned(owners, ids):
            if not owners:
                advanced.extend(ids)
            for task_, write, store, node in owners:
                if task_ is not None:
                    operations_.setdefault(task_, []).extend(ids)
                if node is not None:
                    emitted_.setdefault(node, []).extend(ids)
                if write is not None:
                    writes.setdefault(write, []).extend(ids)
                if store is not None:
                    stores.setdefault(store, []).extend(ids)

        def count():
            return compiler.metal_ir_operations(builder.p)

        # the items in order into the program; a region's direct children (RegionChild, in its arm `arm`) appended to
        # `children` as each registers (postorder)
        def encode(listed, children, arm):
            for kind, payload, owners in listed:
                before, op = count(), None
                if kind == "copy":
                    record, into, given, place, at_ = payload
                    scalar_, sizes, _, _ = _typed(into)
                    op = builder.copy(record, place, at_, sizes, (value_of(into), scalar_),
                                      (value_of(given), _typed(given)[0]))
                    operands[op] = (given, into)
                elif kind == "lapack":
                    record, native, job, rcond, batch, ins, outs, at_ = payload
                    typed_ = [(value_of(b), _typed(b)[1], rank, int(b.dtype.is_complex)) for b, rank in ins + outs]
                    op = builder.lapack(record, at_, native, job.encode(), rcond, batch, typed_[:len(ins)],
                                        typed_[len(ins):])
                    operands[op] = tuple(b for b, _ in ins + outs)
                elif kind == "check":
                    record, info, code, message, at_ = payload
                    op = builder.check(record, at_, value_of(info), code, message)
                    operands[op] = (info,)
                elif kind == "kernel":
                    p, kernel_, arguments, grid, group, at_ = payload
                    builder.identity = builder.base | at_
                    values, described_, kinds_ = [], [], []
                    for kind_, b, access, buffers in arguments:
                        kinds_.append(kind_)
                        if kind_ == UPSTREAM_ERROR:
                            v, size = workspace()
                            values.append((kind_, v, access))
                            described_.append((buffers, size, 0, size, access))
                            continue
                        values.append((kind_, value_of(b), access))
                        size = b.size[0] * b.dtype.itemsize
                        described_.append((buffers, b.dtype.itemsize, final(b).offset * b.dtype.itemsize, size,
                                           access))
                    domain, accesses, made_ = relations(context, kernel_, kinds_, described_)
                    try:
                        op = builder.upstream(kernel_, PLACES[kernel_.place], at_, values, domain, accesses, None,
                                              [o[0][0] for o in owners if o[0] is not None], grid, group)
                    finally:
                        released(domain, made_)
                    upstream_.append(Launched(op, produced[p], kernel_, tuple(described_)))
                    operands[op] = tuple(b for _, b, _, _ in arguments if b is not None)
                elif kind == "neural":
                    result, model, package, saved, (ins, outs, states), at_ = payload

                    def stated(k, states):
                        for name, read, written in states:
                            r = k if read is not None else NONE
                            k += read is not None
                            yield name, r, k if written is not None else NONE
                            k += written is not None

                    sided = [(name, b, access) for name, read, written in states
                             for b, access in ((read, 1), (written, 2)) if b is not None]
                    bound = ins + outs + sided
                    domain, accesses, made_ = features(context, [(b, access) for _, b, access in bound])
                    try:
                        op = builder.upstream(result, ANE, at_, [(UPSTREAM_BUFFER, value_of(b), access)
                                                                 for _, b, access in bound], domain, accesses, None,
                                              [o[0][0] for o in owners if o[0] is not None], (1, 1, 1), None)
                    finally:
                        released(domain, made_)
                    neural_.append(Neural(op, result, package, model.model.function_name,
                                          model.selected["compute_units"].name,
                                          tuple((name, k) for k, (name, _, _) in enumerate(ins)),
                                          tuple((name, len(ins) + k) for k, (name, _, _) in enumerate(outs)),
                                          tuple(stated(len(ins) + len(outs), states)), saved))
                    operands[op] = tuple(b for _, b, _ in bound)
                elif kind == "repeat":
                    record, init, passed, condition, carried = payload
                    encode(init, children, arm)
                    first, inner = count(), []
                    encode(passed, inner, NONE)
                    region = builder.repeat(record, first, count(), value_of(condition),
                                            list(dict.fromkeys(root_of(s) for s in carried)), inner)
                    children.append(RegionChild(REGION_REPEAT, region, arm))
                elif kind == "switch":
                    record, index, branches, common, place, at_ = payload
                    selector = builder.selector(record, place, at_, value_of(index), len(branches))
                    firsts, ends_, yielded, inner = [], [], [], []
                    for k, (sub, yields) in enumerate(branches):
                        firsts.append(count())
                        encode(sub, inner, k)
                        ends_.append(count())
                        yielded += [value_of(b) for b in yields]
                    region = builder.switch(record, value_of(index), firsts, ends_, yielded,
                                            [value_of(b) for b in common], selector, inner)
                    children.append(RegionChild(REGION_SWITCH, region, arm))
                else:
                    record, sub = payload
                    first = count()
                    encode(sub, children, arm)
                    inlinings.append((record, first, count()))
                if owners is not None:
                    owned(owners, list(range(before, count())) or ([op] if op is not None else []))

        encode(items, [], NONE)

        # metal-microbench metal_ir.h metal_ir_operation_axis: each of a numerical node's work axes
        # (compiler/mesh_compiler/placement.py `work_axes`: source node, dimension, original full extent), on each
        # operation it emitted at each operand that is that source node's own program value, that dimension; with no
        # such operand (a produced kernel binds whole storages), on each operation it emitted with none
        for node in dict.fromkeys(associated):
            for reference, dimension, full in node.meta.get("work_axes", ()):
                b = env.get(copied.get(reference))
                hits = [(op, s, dimension) for op in emitted_.get(node, ())
                        for s, o in enumerate(operands.get(op, ())) if isinstance(b, Bound) and o == b]
                for op, s, dim in hits or [(op, NONE, dimension) for op in emitted_.get(node, ())]:
                    status = compiler.metal_ir_operation_axis(builder.p, op, s, dim, tasks[node][0], full)
                    if status:
                        raise OSError(status, "the work axis of operation %d operand %d" % (op, s))
        source, builder.identity = module.meta["relation"], builder.base

        def typed(node):
            b = env.get(copied.get(node))
            if b is None:
                return None
            return rooted(b) if isinstance(b, Bound) else ("literal", b)

        # compiler/mesh_compiler/schedule.py assemble `input_parts`: each partition its landing view (storage offset,
        # shape, stride) of its whole input's placeholder view, its root that input's, its whole input the Root of
        # that placeholder
        def part_of(whole, geometry):
            if geometry is None:
                return None, None, None
            offset, sizes, strides = geometry
            b = env[copied[whole]]
            v = value_of(Bound(b.storage, b.dtype, tuple(sizes), tuple(strides), offset))
            return root_of(b.storage), v, v
        ported = dict(zip([node for node in nodes if node.op == "placeholder"], inputs, strict=True))
        input_parts = {}
        for position_, parts in module.meta.get("input_parts", {}).items():
            made_parts = []
            for processor, whole, part, geometry in parts:
                root, value, landing_ = part_of(whole, geometry)
                made_parts.append(InputPart(processor, ported[copied[whole]], root, value, part, landing_, geometry))
            input_parts[position_] = tuple(made_parts)
        vertices = tuple(tuple(emitted_.get(node, ())) for node in module.meta["vertices"])
        accesses = tuple(typed(node) for node, _ in source.accesses)
        given = tuple(typed(operand_) if kind == "value" else sent_.get(operand_) or received_.get(operand_) or
                      typed(sink) for kind, _, sink, operand_ in source.supplied)
        writers = tuple((tasks[a], tasks[b]) for a, b in module.meta["writers"])
    except BaseException:
        compiler.metal_ir_program_free(builder.p)
        raise
    decomposed = {}
    for generated, original in module.meta["decomposed"].items():
        decomposed.setdefault(original, []).extend(emitted_.get(generated, ()))
    return ConstructedMachine(builder.p, compiler, dict(targets), tuple(inputs), tuple(outputs), spec, (),
                              received_, sent_, ends, {task_: tuple(ops) for task_, ops in operations_.items()},
                              tuple(advanced), made, {k: tuple(ops) for k, ops in writes.items()},
                              {k: tuple(ops) for k, ops in stores.items()}, source, vertices, accesses, given, writers,
                              builder.base, {k: tuple(sorted(ops)) for k, ops in decomposed.items()},
                              input_parts=input_parts, emitted=tuple(builder.emitted), landings={},
                              upstream=tuple(upstream_), produced=dict(produced), context=context,
                              interfaces=interfaces, regions=tuple(builder.regions), inlinings=tuple(inlinings),
                              neural=tuple(neural_), bindings=bindings)

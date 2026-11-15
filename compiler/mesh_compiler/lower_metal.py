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
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.loop_body import InterpreterShim
from torch._inductor.ops_handler import AtomicMode
from torch._inductor.utils import LineContext, sympy_subs
from torch._inductor.virtualized import V
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree
from torch.utils._sympy.interp import sympy_interp
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
# torch 2.14.0 torch/_inductor/codegen/wrapper_fxir.py FxConverter.generate: the wrapper lines its conversion makes no
# value or effect of (memory planning's frees and nulls, comments, kernel definitions, device contexts, symbolic call
# arguments, taken here at their guarding hints, size and stride assertions, subgraph scopes, line contexts)
INERT = (wrapper_lines.FreeLine, wrapper_lines.FreeIfNotReusedLine, wrapper_lines.NullLine, wrapper_lines.CommentLine,
         wrapper_lines.KernelDefinitionLine, wrapper_lines.EnterDeviceContextManagerLine,
         wrapper_lines.ExitDeviceContextManagerLine, wrapper_lines.SymbolicCallArgLine,
         wrapper_lines.AssertSizeStrideLine, wrapper_lines.GroupedAssertSizeStrideLine,
         wrapper_lines.EnterSubgraphLine, wrapper_lines.ExitSubgraphLine, LineContext)
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
# package directory (`saved` its owner, removed with the machine), function and selected compute units by name, and
# each input, output and state feature (its spec name, its metal_ir_upstream argument), metal_ir_upstream_neural's
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
def _ranged(bounds):
    return tuple(INT64[k] if is_infinite(v) else int(v) for k, v in enumerate((bounds.lower, bounds.upper)))


def _unrepresented(*_, **__):
    return None


# an analysis value of the LoopBody ops handler: `index` its mesh-graph.h value (a predicate where `dtype` is
# torch.bool, else an integer), `dtype` its type as torch/_inductor/dtype_propagation.py DtypePropagationOpsHandler
# propagates it (promote_types reads its dtype and is_scalar)
class _Value:
    __slots__ = ("index", "dtype")
    is_scalar = False

    def __init__(self, index, dtype):
        self.index, self.dtype = index, dtype


# rdma/mesh-graph.h struct mesh_graph_index over one kernel's domain as torch/utils/_sympy/interp.py sympy_interp's
# analysis at the kernel's index type `dtype` (codegen/simd.py get_index_dtype_as_torch_dtype, the type its index
# arithmetic is emitted in): each handler of an index expression its mesh_graph_index_* constructor of the same name,
# each integer result reduced exactly into that type (mesh_graph_index_wrap: its identity where isl proves it within
# the type's range), NULL (here None) where isl has no quasi-affine value (a float, a product of two non-constants, an
# unknown symbol or function); every value made kept in `made` (freed after metal_ir_upstream, which makes its access
# relations from them)
class _Index:
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


# torch 2.14.0 torch/_inductor/loop_body.py LoopBody's ops handler (V.ops as its InterpreterShim runs a block) over the
# same analysis, each value a _Value of its dtype as DtypePropagationOpsHandler propagates it (the rule torch's codegen
# types each op by; under the kernel's own V.graph and V.kernel): an index_expr or value_expr its sympy expression
# (_Index) as that dtype; a constant; integer arithmetic (add, sub, mul, neg, minimum, maximum; ops.floordiv floor and
# ops.remainder Python's modulus; ops.truncdiv and ops.mod C's truncation, mesh_graph_index_tdiv_q and tdiv_r), each
# integer result reduced exactly into its dtype (mesh_graph_index_wrap); comparisons; logical operators of any operands'
# truth, bitwise ones of bools; where; to_dtype between integer and bool types its exact conversion (into an integer
# type the reduction into its width, into bool its != 0). A floating result (its rounding), a load (data) and every
# other operation none
class _Ops:
    def __init__(self, index):
        self.index, self.rules = index, DtypePropagationOpsHandler()

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
        return _Value(index.wrapped(value, dtype), dtype)

    def predicate(self, value):
        return None if value is None else _Value(value, torch.bool)

    # an operand's integer value (a bool's its indicator) and its truth (an integer's != 0)
    def integer(self, x):
        index = self.index
        if x.dtype is not torch.bool:
            return x.index
        return index.native("where", x.index, index.constant(1, torch.int64), index.constant(0, torch.int64))

    def truth(self, x):
        index = self.index
        return x.index if x.dtype is torch.bool else index.ne(x.index, index.constant(0, torch.int64))

    def arithmetic(self, name, made, *operands):
        if any(x is None for x in operands):
            return None
        return self.integral(made(*map(self.integer, operands)), getattr(self.rules, name)(*operands))

    def index_expr(self, expr, dtype):
        return self.integral(self.index.interp(expr), self.rules.index_expr(expr, dtype))

    def value_expr(self, expr, dtype):
        return self.integral(self.index.interp(expr), self.rules.value_expr(expr, dtype))

    def constant(self, value, dtype):
        rule = self.rules.constant(value, dtype)
        if rule is torch.bool:
            return self.predicate(self.index.truth(value))
        return self.integral(self.index.constant(value, rule), rule)

    def to_dtype(self, x, dtype, src_dtype=None, use_compute_types=True):
        if x is None:
            return None
        rule = self.rules.to_dtype(x, dtype, src_dtype, use_compute_types)
        if rule is torch.bool:
            return self.predicate(self.truth(x))
        return self.integral(self.integer(x), rule)

    def add(self, a, b):
        return self.arithmetic("add", functools.partial(self.index.native, "add"), a, b)

    def sub(self, a, b):
        index = self.index
        return self.arithmetic("sub", lambda x, y: index.native(
            "add", x, index.native("mul", y, index.constant(-1, torch.int64))), a, b)

    def mul(self, a, b):
        return self.arithmetic("mul", functools.partial(self.index.native, "mul"), a, b)

    def neg(self, a):
        index = self.index
        return self.arithmetic("neg", lambda x: index.native("mul", x, index.constant(-1, torch.int64)), a)

    def floordiv(self, a, b):
        return self.arithmetic("floordiv", functools.partial(self.index.native, "floordiv"), a, b)

    def remainder(self, a, b):
        return self.arithmetic("remainder", functools.partial(self.index.native, "python_mod"), a, b)

    def truncdiv(self, a, b):
        return self.arithmetic("truncdiv", functools.partial(self.index.native, "tdiv_q"), a, b)

    def mod(self, a, b):
        return self.arithmetic("mod", functools.partial(self.index.native, "tdiv_r"), a, b)

    def minimum(self, a, b):
        return self.arithmetic("minimum", functools.partial(self.index.native, "minimum"), a, b)

    def maximum(self, a, b):
        return self.arithmetic("maximum", functools.partial(self.index.native, "maximum"), a, b)

    def where(self, predicate, a, b):
        if predicate is None or a is None or b is None:
            return None
        index, rule, p = self.index, self.rules.where(predicate, a, b), self.truth(predicate)
        if rule is torch.bool:
            return self.predicate(index.or_(index.and_(p, self.truth(a)), index.and_(index.not_(p), self.truth(b))))
        return self.integral(index.native("where", p, self.integer(a), self.integer(b)), rule)

    def compared(self, relation, a, b):
        if a is None or b is None:
            return None
        return self.predicate(self.index.compare(relation, self.integer(a), self.integer(b)))

    def lt(self, a, b):
        return self.compared(INDEX_LT, a, b)

    def le(self, a, b):
        return self.compared(INDEX_LE, a, b)

    def gt(self, a, b):
        return self.compared(INDEX_GT, a, b)

    def ge(self, a, b):
        return self.compared(INDEX_GE, a, b)

    def eq(self, a, b):
        return self.compared(INDEX_EQ, a, b)

    def ne(self, a, b):
        return self.compared(INDEX_NE, a, b)

    def logical_and(self, a, b):
        return None if a is None or b is None else self.predicate(self.index.and_(self.truth(a), self.truth(b)))

    def logical_or(self, a, b):
        return None if a is None or b is None else self.predicate(self.index.or_(self.truth(a), self.truth(b)))

    def logical_not(self, a):
        return None if a is None else self.predicate(self.index.not_(self.truth(a)))

    # a bitwise operator: of bools the logical one, of integers not represented
    def and_(self, a, b):
        return self.logical_and(a, b) if a is not None and b is not None and a.dtype is b.dtype is torch.bool else None

    def or_(self, a, b):
        return self.logical_or(a, b) if a is not None and b is not None and a.dtype is b.dtype is torch.bool else None

    def not_(self, a):
        return self.logical_not(a) if a is not None and a.dtype is torch.bool else None

    bitwise_and, bitwise_or, bitwise_not = and_, or_, not_

    def identity(self, x):
        return x


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
        storages, held, results, wired, error = [], {}, {}, {}, []
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

        # one name of processor p's wrapper in `graph` (the GraphLowering whose lines name it) as its wrapper has bound
        # it by this line; else a constant's own CONSTANT root (its wrapper's, recorded once), a
        # MutationLayoutSHOULDREMOVE buffer its target's
        def hold(p, graph, name):
            if (p, name) not in held:
                if name in graph.constants:
                    data = graph.constants[name]
                    held[p, name] = Held(stored(CONSTANT, data.dtype, data.untyped_storage().nbytes(), data),
                                         data.storage_offset() * data.element_size())
                    bindings_.append((p, (), graph, name, data, Bound(held[p, name].storage, data.dtype,
                                                                      tuple(data.shape), tuple(data.stride()),
                                                                      data.storage_offset())))
                    return held[p, name]
                buffer = graph.get_buffer(name)
                layout = buffer.get_layout() if buffer is not None else None
                if not isinstance(layout, upstream.MutationLayoutSHOULDREMOVE):
                    raise NotImplementedError(
                        "unresolved integration: processor %s buffer %s has no binding in its wrapper before its use "
                        "(no graph input, constant, body input or memory planning line binds it)" % (p, name))
                held[p, name] = hold(p, graph, layout.get_buffer().get_name())
            return held[p, name]

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

        # an IR value of processor p at its own layout
        def projected(p, graph, x):
            x, layout = layout_of(x)
            return placed(hold(p, graph, x.get_name()), layout, graph)

        # an operand or interface value: a tensor's Bound; a symbolic one its guarding hint (ShapeAsConstantBuffer's
        # expression, a sympy expression); NoneAsConstantBuffer or None none; any other value itself
        def operand_of(p, graph, x):
            if isinstance(x, upstream.ShapeAsConstantBuffer):
                return _hint(graph, x.expr)
            if x is None or isinstance(x, upstream.NoneAsConstantBuffer):
                return None
            if isinstance(x, upstream.IRNode):
                return projected(p, graph, x)
            return _hint(graph, x) if hasattr(x, "free_symbols") else x

        # a scalar control operand (a selector, a condition): its tensor's Bound, a symbolic one an int32 CONSTANT
        def valued(p, graph, x):
            if isinstance(x, upstream.ShapeAsConstantBuffer):
                data = torch.tensor([_hint(graph, x.expr)], dtype=torch.int32)
                return Bound(stored(CONSTANT, torch.int32, 4, data), torch.int32, (), (), 0)
            return projected(p, graph, x)

        # a kernel argument's binding: its storage from its index origin, its layout's whole storage_size
        def binding(p, graph, name):
            layout = graph.get_buffer(name).get_layout()
            h = hold(p, graph, name)
            return Bound(h.storage, layout.dtype, (_hint(graph, layout.storage_size()),), (1,),
                         h.at // layout.dtype.itemsize)

        # a control body occurrence's inputs (body.graph.graph_inputs, in order) bound to the operands `given` its call
        # passes (codegen_subgraph_by_inlining's prefix: a tensor's Bound; a symbolic one its guarding hint, the value
        # every launch reads), each recorded at that occurrence
        def bind(p, body, given, occurrence):
            for name, b in zip(body.graph.graph_inputs, given, strict=True):
                if isinstance(b, Bound):
                    held[p, name] = Held(b.storage, b.offset * b.dtype.itemsize)
                bindings_.append((p, occurrence, body, name, body.graph.graph_inputs[name], b))

        # a launch's scalar argument: an int64 CONSTANT holding its value (a sizevar or a SymbolicCallArg's numel at
        # its guarding hint)
        def scalar(graph, expr):
            data = torch.tensor([_hint(graph, getattr(expr, "inner_expr", expr))], dtype=SIZEVAR)
            return (UPSTREAM_SCALAR, Bound(stored(CONSTANT, SIZEVAR, SIZEVAR.itemsize, data), SIZEVAR, (1,), (1,), 0),
                    1, ())

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
                env[node] = Bound(stored(CONSTANT, value.dtype, value.untyped_storage().nbytes(), value),
                                  value.dtype, tuple(value.shape), tuple(value.stride()),
                                  value.storage_offset()) if isinstance(value, torch.Tensor) else value

        # each member's graph inputs: its machine placeholder's root at the input's first element
        # (graph_input_storage_offsets; the InputBuffer's layout indexes from that element)
        for p in members:
            graph = produced[p].graph
            for given in produced[p].inputs if graph is not None else ():
                b = env.get(copied.get(originals[p].get(given.node)))
                if isinstance(b, Bound) and given.name in graph.graph_inputs:
                    held[p, given.name] = Held(b.storage, _hint(graph, graph.graph_input_storage_offsets[given.name])
                                               * b.dtype.itemsize)
                    source_ = graph.graph_inputs[given.name]
                    bindings_.append((p, (), given, given.name, source_, placed(held[p, given.name],
                                                                                source_.get_layout(), graph)))

        # a declared transfer's logical wire (its one TEMPORARY storage of its logical bytes) viewed at its declared
        # landed or sent view
        def wire(channel, dtype, sizes, landed_):
            _, value, size, sent_view, arrived = wires[channel]
            offset, element, pairs = arrived if landed_ else sent_view
            if value not in wired:
                wired[value] = stored(TEMPORARY, dtype, size)
            return value, Bound(wired[value], dtype, tuple(sizes) if landed_ else tuple(n for n, _ in pairs),
                                tuple(stride // element for _, stride in pairs), offset // element)

        # a receive's result: its schema's one Tensor, so its FallbackKernel itself (ir.py FallbackKernel.create,
        # create_direct_output: the kernel its own result buffer at its own layout, no MultiOutput), bound by its
        # ExternKernelAllocLine `line` at its body occurrence `occurrence`, to the value `given` it receives: that
        # value's storage where its layout is the result's, else its own root and one direct copy of `given` at the
        # receive, into the receive's own destination `into`
        def landed(p, graph, record, line, given, owner, into, occurrence):
            node = record.node
            layout, name = node.get_layout(), node.get_name()
            b = placed(Held(given.storage, 0), layout, graph)
            if (b.dtype, b.size, b.stride) == (given.dtype, given.size, given.stride):
                held[p, name] = Held(given.storage, (given.offset - _hint(graph, layout.offset)) * given.dtype.itemsize)
                own = given
            else:
                held[p, name] = Held(stored(TEMPORARY, layout.dtype, _bytes(graph, layout)), 0)
                own = placed(held[p, name], layout, graph)
                copying(into, record, own, given, [owner], PLACES[produced[p].kind], position(owner))
            recorded(p, graph, occurrence, line, name, node)
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
            layout, name = node.get_layout(), node.get_name()
            if channel in wires:
                value_, given = wire(channel, layout.dtype, [_hint(graph, n) for n in layout.size], True)
                env[m] = landed(p, graph, record, line, given, (tasks[m], None, None, m), into, occurrence)
                landing.append((channel, value_, given))
            else:
                held[p, name] = Held(stored(TEMPORARY, layout.dtype, _bytes(graph, layout)), 0)
                recorded(p, graph, occurrence, line, name, node)
                env[m] = placed(held[p, name], layout, graph)
                landing.append((channel, None, env[m]))
            interface.add(m)

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

        # a produced kernel's launch arguments in its launch's own order, each (kind, Bound, access, its buffer names),
        # a buffer's access its KernelArgs membership (inplace_buffers read and written under every name of its
        # InplacedBuffer, input_buffers read, output_buffers written, both where it is both). An mps launch the patched
        # mps.py launch `values` (its buffers by outer name, its sizevars and dynamic numels as expressions, then its
        # error buffer, the program's error workspace, read and written); a cpu launch (kernel name, arg_defs,
        # call_args, arg_types) its KernelCallLine's call_args as KernelArgs.cpp_argdefs ordered them: its buffers by
        # outer name, then each of its KernelArgs.sizevars (its expression), the buffers its arg_types' count less the
        # sizevars' (as lower_inductor.py emitted counts its parameters)
        def launch_arguments(p, kernel_):
            graph, args = kernel_.graph, kernel_.owner.args

            def buffer(name):
                inplaced = args.inplace_buffers.get(name)
                if inplaced is not None and not isinstance(inplaced, RemovedArg):
                    return UPSTREAM_BUFFER, binding(p, graph, name), 3, tuple(inplaced.other_names)
                access = (1 if name in args.input_buffers else 0) | (2 if name in args.output_buffers else 0)
                return UPSTREAM_BUFFER, binding(p, graph, name), access, (name,)

            if kernel_.place == "cpu":
                _, _, call_args, types = kernel_.launch
                buffers = len(types) - len(args.sizevars)
                return [buffer(name) for name in call_args[:buffers]] + [scalar(graph, v) for v in args.sizevars]
            _, values, _, _, _, error_ = kernel_.launch
            arguments = [buffer(v) if isinstance(v, str) else scalar(graph, v) for v in values]
            if error_ is not None:
                arguments.append((UPSTREAM_ERROR, None, 3, ()))
            return arguments

        # a produced kernel at its KernelCallLine: its arguments, its sources' positions, its launch (its domain and
        # accesses its relations at encode)
        def kernel(p, kernel_, into):
            owners = [owner_of(p, n) for n in kernel_.origins]
            lead = next((o for o in owners if o[0] is not None), None)
            into.append(("kernel", (p, kernel_, launch_arguments(p, kernel_), position(lead) if lead else end),
                         owners))
            associated.extend(o[3] for o in owners if o[3] is not None)

        # an extern's results (ir.py FallbackKernel.outputs, its MultiOutputs by their indices) each its own TEMPORARY
        def outputs_of(p, graph, node):
            made_ = {}
            for out in pytree.tree_leaves(node.outputs):
                if isinstance(out, upstream.MultiOutput):
                    made_[_key(out.indices)] = Held(stored(TEMPORARY, out.get_layout().dtype,
                                                           _bytes(graph, out.get_layout())), 0)
            results[p, node.get_name()] = made_
            return made_

        # a typed LAPACK operator (operations.py LAPACK[packet], its Call: routine, jobs, rcond, inputs, matrices,
        # info by its schema's names) as the native owner's one call (metal_ir_lapack) in its own destination `into`:
        # its job its jobs' str operands in order (a literal the native's own, as gels trans 'N'), its rcond its float
        # operand, its inputs its matrix operands, its outputs every result in schema order but a job's (0,)
        # placeholder of a matrix result, its batch its INFO's. Each result is the original one, its kind the
        # original's (the fakes' int32 INFO and RANK): the native owner stages INFO and RANK at its ILP64 width
        # (metal_native.mm LapackLayout, metal-microbench 633068d4), converted once on its own copy-out. Its workspace
        # (WORK's query, IWORK, JPVT, RWORK) is the native owner's
        def lapacked(p, graph, record, stated, given, origins, owners, at, into):
            node = record.node
            target, schema = node.op_overload, node.op_overload._schema
            job = "".join(given[name] for name in stated.jobs)
            rcond = float(given[stated.rcond]) if stated.rcond is not None else 0.0
            ins = [projected(p, graph, given[name]) for name in stated.inputs]
            made_ = outputs_of(p, graph, node)
            bounds = {r.name: placed(made_[_key(out.indices)], out.get_layout(), graph)
                      for r, out in zip(schema.returns, node.outputs, strict=True)}
            batch = bounds[stated.info].size
            outs = [(b, len(b.size) - len(batch)) for name, b in bounds.items()
                    if name not in stated.matrices or len(b.size) - len(batch) == 2]
            into.append(("lapack", (record, NATIVE[stated.routine], job, rcond, batch,
                                    [(b, len(b.size) - len(batch)) for b in ins], outs, at), owners))
            origin = next(n for n in origins if getattr(n, "target", None) is target)
            parts = {u.args[1]: u for u in origin.users if u.target is operator.getitem}
            for k, b in enumerate(bounds.values()):
                m = vertex(p, parts[k]) if k in parts else None
                if m is not None:
                    env[m] = b
            associated.extend(o[3] for o in owners if o[3] is not None)

        # an ordinary extern's own call (its line naming its node), into its destination `into` (the body, pass or arm
        # its call is in): a send or receive (its schema's channel), across machines at this machine body's SEND or
        # RECV (module.meta["transfers"]), within the machine the local sender's value; a typed LAPACK operator; its
        # INFO boundary (metal_ir_check, its api the message, CONVERGENCE: torch.linalg.LinAlgError). Every other
        # ExternKernel call (ExternKernelOut, a CPU extern GEMM's extern_kernels.mm; DynamicScalar, AssertScalar,
        # RandomSeeds; any other fallback) has no native operation: the compile raises it
        def called(p, graph, record, line, into, occurrence):
            node = record.node
            target = getattr(node, "op_overload", None)
            stated = operations.LAPACK.get(getattr(target, "overloadpacket", None))
            if not isinstance(node, upstream.FallbackKernel) or (
                    stated is None and target not in (SEND, producer.RECV_STRIDED, CHECK_ERRORS)):
                raise NotImplementedError(
                    "unresolved integration: processor %s extern call %s (%s, %s) requires a native operation for "
                    "its original ExternKernel call; metal_ir.h has none" % (
                        p, node.get_name(), type(node).__name__,
                        getattr(node, "python_kernel_name", None) or target))
            given = _named(node)
            origins, owners, at = owning(p, node)
            if target is SEND:
                sending_value(p, record, given["channel"], projected(p, graph, given["value"]), into)
            elif target is producer.RECV_STRIDED:
                channel = given["channel"]
                if channel in remote:
                    received(p, graph, record, line, remote[channel][1], into, occurrence)
                else:
                    n = next(n for n in origins if getattr(n, "target", None) is RECV)
                    own = landed(p, graph, record, line, received_view(channel), owner_of(p, n), into, occurrence)
                    m = vertex(p, n)
                    if m is not None:
                        env[m] = own
            elif stated is not None:
                lapacked(p, graph, record, stated, given, origins, owners, at, into)
            else:
                into.append(("check", (record, projected(p, graph, given["info"]), CHECK_CONVERGENCE, given["api"],
                                       at), owners))
                associated.extend(o[3] for o in owners if o[3] is not None)

        # a memory planning line's binding of `name` (its original buffer `source`), recorded at its occurrence at
        # that buffer's own layout
        def recorded(p, graph, occurrence, line, name, source):
            bindings_.append((p, occurrence, line, name, source, placed(held[p, name], source.get_layout(), graph)))

        # one wrapper line of `graph` that no extern call wrote, at its body occurrence `occurrence`: its memory
        # planning, its kernel call; a line no native consumer has (a str line outside a control call's own
        # scaffolding, any other line) the compile raises
        def planned(p, graph, line, into, occurrence):
            if isinstance(line, wrapper_lines.KernelCallLine):
                kernel_ = produced[p].kernels.get(id(line))
                if kernel_ is None:
                    raise NotImplementedError("unresolved integration: processor %s kernel call %s has no Kernel "
                                              "record (Produced.kernels)" % (p, line.kernel_name))
                kernel(p, kernel_, into)
            elif isinstance(line, wrapper_lines.AllocateLine):
                layout = line.node.get_layout()
                held[p, line.node.get_name()] = Held(stored(TEMPORARY, layout.dtype, _bytes(graph, layout)), 0)
                recorded(p, graph, occurrence, line, line.node.get_name(), line.node)
            elif isinstance(line, wrapper_lines.ReinterpretLine):
                base, view = hold(p, graph, line.node.get_name()), line.layout
                if isinstance(view, upstream.NonOwningLayout):
                    view = view.view.get_layout()
                held[p, line.reused_as.get_name()] = Held(base.storage, base.at + (
                    _hint(graph, line.node.get_offset()) + _hint(graph, view.offset) -
                    _hint(graph, line.reused_as.get_offset())) * view.dtype.itemsize)
                recorded(p, graph, occurrence, line, line.reused_as.get_name(), line.reused_as)
            elif isinstance(line, wrapper_lines.ReuseLine):
                held[p, line.reused_as.get_name()] = hold(p, graph, line.node.get_name())
                recorded(p, graph, occurrence, line, line.reused_as.get_name(), line.reused_as)
            elif isinstance(line, wrapper_lines.MultiOutputLine):
                held[p, line.result_name] = results[p, line.arg_name][_key(line.indices)]
                recorded(p, graph, occurrence, line, line.result_name, graph.get_buffer(line.result_name))
            elif not isinstance(line, INERT):
                raise NotImplementedError("unresolved integration: processor %s wrapper line %r (%s) has no native "
                                          "consumer" % (p, line, type(line).__name__))

        # an extern call at its first line, into its destination `into`, at its body occurrence `occurrence`: a
        # control call its region, an ordinary one its lines in order (its memory planning, its own call)
        def extern(p, graph, record, into, occurrence):
            node = record.node
            if isinstance(node, upstream.WhileLoop):
                return repeat(p, record, into, occurrence)
            if isinstance(node, upstream.Switch):
                return switch(p, record, into, occurrence)
            if isinstance(node, upstream.InvokeSubgraph):
                return invoked(p, record, into, occurrence)
            calls = 0
            for line in record.lines:
                if getattr(line, "node", None) is node and not isinstance(line, wrapper_lines.MemoryPlanningLine):
                    called(p, graph, record, line, into, occurrence)
                    calls += 1
                else:
                    planned(p, graph, line, into, occurrence)
            if not calls and not isinstance(node, upstream.MultiOutput):
                raise NotImplementedError("unresolved integration: processor %s extern call %s (%s) wrote no call "
                                          "line of its node" % (p, node.get_name(), type(node).__name__))

        # one body occurrence's lines in order (`calls` the ExternCalls of the scheduler writing them), into `into`
        def consume(p, graph, lines, calls, into, occurrence):
            owner, seen = {id(line): call for call in calls for line in call.lines}, set()
            for line in lines:
                call = owner.get(id(line))
                if call is None:
                    planned(p, graph, line, into, occurrence)
                elif id(call) not in seen:
                    seen.add(id(call))
                    extern(p, graph, call, into, occurrence)

        # one control body occurrence of the call `record` (an Inlined, a Lifted: `lines` its wrapper lines) within
        # `occurrence`, its inputs bound to the operands `given` there, as consumed into its own "inlined" item of
        # `into`, its record kept with its encoded interval (ConstructedMachine.inlinings), owned by `owners` (None:
        # the region it is in owns it)
        def inline(p, record, body, lines, given, into, occurrence, owners=None):
            within, sub = occurrence + ((record, body),), []
            bind(p, body, given, within)
            consume(p, body.graph, lines, body.externs, sub, within)
            into.append(("inlined", (body, sub), owners))

        # torch 2.14.0 wrapper.py codegen_while_loop (aot_mode: its Inlined cond before the loop, body, cond at the
        # loop's end, lower_inductor.py ExternCall.inlined) as one native repeat region (metal_ir_repeat): each carried
        # input its own carried storage of its body input's layout, its initial value copied in (a carried input the
        # body mutates, ir.py WhileLoop.mutation_outputs, the input's own storage); the cond's result copied into the
        # one-element I32 condition before the region and at each pass's end; a pass the body, then each body output
        # into its carried storage (one that reads a carried storage staged first), then the cond; each of the three
        # inlinings its own interval. Each result (WhileLoop.outputs, MultiOutput [(list, i)]) its carried storage
        # where its layout is that storage's, else its own and one copy after the loop. stack_output
        # (while_loop_stack_output: each pass's carried values stacked, its outputs' leading extent the loop's own
        # iteration count, a fresh unbacked symbol of its fakes) has no native repeat output: the compile raises it
        def repeat(p, record, into, occurrence):
            w, graph = record.node, record.graph
            cond0, body, cond1 = record.inlined
            origins, owners, at = owning(p, w)
            if w.stack_output:
                raise NotImplementedError(
                    "unresolved integration: processor %s while_loop %s stack_output requires a native repeat "
                    "region's per-pass output stack, its leading extent the loop's own iteration count (the fakes' "
                    "unbacked symbol)" % (p, w.get_name()))
            place, init, passed = PLACES[produced[p].kind], [], []
            mutated = {name for m in w.mutation_outputs for name in m.get_mutation_names()}
            carried = []
            for x, inner in zip(w.carried_inputs, body.graph.graph_inputs.values()):
                b = projected(p, graph, x)
                if x.get_name() in mutated:
                    carried.append(b)
                    continue
                layout = inner.get_layout()
                own = placed(Held(stored(TEMPORARY, layout.dtype, _bytes(body.graph, layout)), 0), layout,
                             body.graph)
                copying(init, record, own, b, owners, place, at)
                carried.append(own)
            given = carried + [operand_of(p, graph, x) for x in w.additional_inputs]
            inline(p, record, cond0, cond0.lines, given, init, occurrence)
            tested = valued(p, cond0.graph, cond0.graph.graph_outputs[0])
            condition = staged(tested, torch.int32)
            copying(init, record, condition, tested, owners, place, at)
            inline(p, record, body, body.lines, given, passed, occurrence)
            ids, backs = {id(b.storage) for b in carried}, []
            for into_, out in zip(carried, body.graph.graph_outputs):
                b = projected(p, body.graph, out)
                if b != into_ and id(b.storage) in ids:
                    stage = staged(b)
                    copying(passed, record, stage, b, owners, place, at)
                    b = stage
                backs.append((into_, b))
            for into_, b in backs:
                if b != into_:
                    copying(passed, record, into_, b, owners, place, at)
            inline(p, record, cond1, cond1.lines, given, passed, occurrence)
            copying(passed, record, condition, valued(p, cond1.graph, cond1.graph.graph_outputs[0]), owners, place,
                    at)
            into.append(("repeat", (record, init, passed, condition, [b.storage for b in carried]), owners))
            made_ = {}
            for out in w.outputs:
                layout, c = out.get_layout(), carried[out.indices[0][1]]
                own = placed(Held(c.storage, c.offset * c.dtype.itemsize), layout, graph)
                if (own.dtype, own.size, own.stride, own.offset) != (c.dtype, c.size, c.stride, c.offset):
                    h = Held(stored(TEMPORARY, layout.dtype, _bytes(graph, layout)), 0)
                    copying(into, record, placed(h, layout, graph), c, owners, place, at)
                    made_[_key(out.indices)] = h
                else:
                    made_[_key(out.indices)] = Held(c.storage, c.offset * c.dtype.itemsize)
            results[p, w.get_name()] = made_
            associated.extend(o[3] for o in owners if o[3] is not None)

        # each lifted body by its graph (lower_inductor.py Lifted: lowered once per wrapper, a later Switch of the
        # same branch graph its first's)
        lifted = {}

        # ir.py Switch (torch.cond: a bool selector, branches [false, true]; torch.switch: an integer selector) as one
        # native indexed switch (metal_ir_switch_selector, metal_ir_switch): its index the selector (a bool one copied
        # into a one-element I32 first), each branch its Lifted body's lines (its own interval) with its inputs bound
        # to the operands, its yields copied into the common results (Switch.outputs, MultiOutput [(list, i)], each its
        # own TEMPORARY)
        def switch(p, record, into, occurrence):
            s, graph = record.node, record.graph
            origins, owners, at = owning(p, s)
            place = PLACES[produced[p].kind]
            lifted.update((id(body.graph), body) for body in record.lifted)
            index = valued(p, graph, s.selector)
            if index.dtype is torch.bool:
                copying(into, record, staged(index, torch.int32), index, owners, place, at)
                index = into[-1][1][1]
            operands = [operand_of(p, graph, x) for x in s.operands]
            made_ = outputs_of(p, graph, s)
            common = [placed(made_[_key(out.indices)], out.get_layout(), graph) for out in s.outputs]
            branches = []
            for branch in s.branches:
                body, sub = lifted[id(branch.graph)], []
                inline(p, record, body, body.wrapper.lines, operands, sub, occurrence)
                yields = [projected(p, body.graph, out) for out in body.graph.graph_outputs]
                for result, given in zip(common, yields, strict=True):
                    copying(sub, record, result, given, owners, place, at)
                branches.append((sub, yields))
            into.append(("switch", (record, index, branches, common, place, at), owners))
            associated.extend(o[3] for o in owners if o[3] is not None)

        # ir.py InvokeSubgraph (aot_mode: its one Inlined body): its inputs bound to its operands, its body's lines
        # (its own interval, owned by the invocation), each result (InvokeSubgraph.outputs, MultiOutput [(list, i)] at
        # its body output's own layout) that output
        def invoked(p, record, into, occurrence):
            v, graph = record.node, record.graph
            (body,) = record.inlined
            origins, owners, at = owning(p, v)
            inline(p, record, body, body.lines, [operand_of(p, graph, x) for x in v.inputs], into, occurrence, owners)
            made_ = {}
            for out, x in zip(v.outputs, body.graph.graph_outputs, strict=True):
                if isinstance(out, upstream.MultiOutput):
                    b = projected(p, body.graph, x)
                    made_[_key(out.indices)] = Held(b.storage, (b.offset - _hint(graph, out.get_layout().offset)) *
                                                    b.dtype.itemsize)
            results[p, v.get_name()] = made_

        # a Neural Engine member (lower_inductor.py Model): one operation, its original Core ML program saved once
        # (MLModel.save: its package, its debug-handle mapping), its spec's features bound positionally in coremltools'
        # own order (its torch.export frontend's user inputs in graph order: Produced.inputs, then Model.received; its
        # outputs the program's user outputs, returned then sent). Each input its machine value (a placeholder's, a
        # received channel's landing); each output its ExportedProgram value's own storage as the program's fakes
        # alias it (torch/_export: a view of an input placeholder that input's machine storage, else one TEMPORARY per
        # distinct fake storage), at its fake's layout. A declared state (MLState) the compile raises: the Model
        # record binds no state to its value. A machine-local received input binds its sent value (received_view)
        def neural(p):
            made_, model = produced[p], produced[p].model
            spec, program = model.model.get_spec(), model.program
            if spec.description.state:
                raise NotImplementedError(
                    "unresolved integration: processor %s Core ML program declares states %s; "
                    "metal_ir_upstream_neural state[] requires each state's value, which lower_inductor.py Model "
                    "does not bind (its ExportedProgram buffer and the body node whose storage it is)" % (
                        p, [f.name for f in spec.description.state]))
            nodes_ = {n.name: n for n in program.graph.nodes}
            given = []
            for i in made_.inputs:
                b = env.get(copied.get(originals[p].get(i.node)))
                if isinstance(b, Bound):
                    given.append((i.value, b))
            for at_, n in model.received:
                channel, placeholder = n.args[0], nodes_[made_.signature.user_inputs[at_]]
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
                given.append((placeholder, b))
            storages_ = {}
            for placeholder, b in given:
                fake = placeholder.meta["val"]
                storages_.setdefault(fake.untyped_storage()._cdata, Held(
                    b.storage, b.offset * b.dtype.itemsize - fake.storage_offset() * fake.element_size()))
            outs = {}
            for out in made_.outputs:
                fake = out.value.meta["val"]
                key = fake.untyped_storage()._cdata
                if key not in storages_:
                    storages_[key] = Held(stored(TEMPORARY, fake.dtype, fake.untyped_storage().nbytes()), 0)
                h = storages_[key]
                outs[out] = Bound(h.storage, fake.dtype, tuple(fake.shape), tuple(fake.stride()),
                                  (h.at + fake.storage_offset() * fake.element_size()) // fake.element_size())
            handles = dict.fromkeys(n for nodes__ in model.handles.values() for n in nodes__)
            owners = [owner_of(p, n) for n in handles]
            lead = next((o for o in owners if o[0] is not None), None)
            saved = tempfile.TemporaryDirectory()
            package = os.path.join(saved.name, "model.mlpackage")
            model.model.save(package)
            bound = ([(f.name, b, 1) for f, (_, b) in zip(spec.description.input, given, strict=True)],
                     [(f.name, b, 2) for f, b in zip(spec.description.output, outs.values(), strict=True)])
            bodies[p].append(("neural", (made_, model, package, saved, bound, position(lead) if lead else end),
                              owners))
            ports[p] = [(i, env.get(copied.get(originals[p].get(i.node)))) for i in made_.inputs], list(outs.items())
            sends = {o.node: b for o, b in outs.items() if o.sent}
            for s in made_.module.graph.find_nodes(op="call_function", target=SEND):
                sending_value(p, made_, channel_of(s), sends[s.args[0]], bodies[p])

        # the resolution: each member in module.meta["members"] order, a wrapper member's lines in their own order (its
        # body occurrence ()), a Neural Engine member its one operation; each machine-local channel's received storage
        # then its sent value's (Storage.alias); the resolved items each member's own order
        for p in members:
            if produced[p].kind == "ane":
                neural(p)
            else:
                graph = produced[p].graph
                consume(p, graph, graph.wrapper_code.lines, produced[p].externs, bodies[p], ())
        for channel, storage in forwards.items():
            v = references[channel]
            storage.alias = (v.storage, v.offset * v.dtype.itemsize)
        items.extend(item for p in members for item in bodies[p])

        # each member's interface (lower_inductor.py Produced.inputs and .outputs, Output.value the GraphLowering's
        # graph_outputs entry): their projections (operand_of, at its wrapper's end), the program values of the machine
        # nodes standing for them
        results_ = {}
        for p in members:
            graph = produced[p].graph
            if p not in ports:
                ports[p] = ([(i, operand_of(p, graph, i.value)) for i in produced[p].inputs],
                            [(o, operand_of(p, graph, o.value)) for o in produced[p].outputs])
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
                    p, kernel_, arguments, at_ = payload
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
                    grid = (1, 1, 1) if kernel_.place == "cpu" else (
                        *(_hint(kernel_.graph, n) for n in kernel_.launch[3] or ()), 1, 1, 1)[:3]
                    group = None if kernel_.place == "cpu" or kernel_.launch[4] is None else (
                        *(_hint(kernel_.graph, n) for n in kernel_.launch[4]), 1, 1)[:3]
                    domain, accesses, made_ = relations(context, kernel_, kinds_, described_)
                    try:
                        op = builder.upstream(kernel_, PLACES[kernel_.place], at_, values, domain, accesses, None,
                                              [o[0][0] for o in owners if o[0] is not None], grid, group)
                    finally:
                        released(domain, made_)
                    upstream_.append(Launched(op, produced[p], kernel_, tuple(described_)))
                    operands[op] = tuple(b for _, b, _, _ in arguments if b is not None)
                elif kind == "neural":
                    result, model, package, saved, (ins, outs), at_ = payload
                    bound = ins + outs
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
                                          tuple((name, len(ins) + k) for k, (name, _, _) in enumerate(outs)), (),
                                          saved))
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

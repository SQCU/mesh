# design/compilation-composition-change-request.md items 1, 2, 5, 6 and 7 and its numerical lowerer boundary;
# ~/schedule-emit/schedule_emit/driver.py and runtime.py: one machine's typed program, its processors' produced results
# (compiler/mesh_compiler/lower_inductor.py lower: each processor body's retained GraphLowering, its buffers, layouts
# and aliases, and its backend's emitted calls in their own order) projected once onto the native program
# (metal-microbench metal_ir.h metal_ir_value, metal_ir_value_view, metal_ir_upstream and its stages, metal_ir_lapack,
# metal_ir_check, metal_ir_generic), within the machine's orchestration (compiler/mesh_compiler/schedule.py assemble:
# its interface, its transfers across machines, its results); compiler/mesh_compiler/schedule.py Declarations,
# SourceRelation. Pairing: output_data/compiler-workgroup-20261114/compiler/numerical66f-operands.md
import collections
import ctypes as C
import dataclasses
import weakref

import sympy
import torch
from torch._inductor import dependencies
from torch._inductor import ir as upstream
from torch._inductor.codegen.common import RemovedArg
from torch._inductor.utils import unique
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import SymT, symbol_is_type

from . import lower_inductor as producer
from . import operations
from .ir import RECV, SEND
from .lower_inductor import copies
from .metal import (CHECK_CONVERGENCE, CONSTANT, INPUT, LAPACK_GEEV, LAPACK_GELSD, LAPACK_GELSS, LAPACK_GESDD,
                    LAPACK_HEEVD, NONE, OUTPUT, SCALAR, SOURCE_EXTERNAL, SOURCE_LITERAL, SOURCE_RECEIVED, SOURCE_SENT,
                    SOURCE_VALUE, STATE, TEMPORARY, UPSTREAM_BUFFER, UPSTREAM_ERROR, UPSTREAM_SCALAR, SourceOperand,
                    ValueInfo, _Builder, _compiler, _composition, _configured, _emitted, _footprints, _upstream)
from .operations import ANE, CPU, GPU

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
InputPart = collections.namedtuple("InputPart", "processor input root value source landing")
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
# metal-microbench metal_native.mm ACCELERATE_NEW_LAPACK and ACCELERATE_LAPACK_ILP64: the native LAPACK owner's
# selected ABI, every integer it passes or writes (INFO, RANK, IWORK) an __LAPACK_int of 64 bits
ILP64 = torch.int64
# compiler/mesh_compiler/operations.py LAPACK's typed operators (mesh::lapack_*, their Routine) as the native owner's
# routines (metal_ir.h METAL_IR_GESDD .. METAL_IR_GELSS: metal_native.mm LapackWorkspace32/64 and LapackMember, their
# workspace queries and calls); mesh::lapack_gels and mesh::lapack_gelsy have none there yet
NATIVE = {operations.LAPACK_GESDD: LAPACK_GESDD, operations.LAPACK_HEEVD: LAPACK_HEEVD,
          operations.LAPACK_DGEEV: LAPACK_GEEV, operations.LAPACK_ZGEEV: LAPACK_GEEV,
          operations.LAPACK_GELSD: LAPACK_GELSD, operations.LAPACK_GELSS: LAPACK_GELSS}
# torch 2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp _linalg_check_errors(info, api, is_matrix): the INFO
# boundary each LAPACK formulation calls after its routine (operations.py _lapack_singular ..)
CHECK_ERRORS = torch.ops.aten._linalg_check_errors.default


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
# `identity` its definer's source position), its ValueID once made (`root`)
@dataclasses.dataclass(eq=False)
class Storage:
    role: int
    dtype: torch.dtype
    bytes: int
    data: object = None
    identity: int = 0
    zeroed: bool = False
    root: int = None


# one value's projection: the view of `storage` as `dtype` at `size` and `stride` from element `offset` (in dtype's
# elements from the storage's first byte), metal_ir_value_view's operands
Bound = collections.namedtuple("Bound", "storage dtype size stride offset")
# one produced kernel as constructed (ConstructedMachine.upstream): its operation, its Produced (the upstream
# compiler's retained result, kept with it) and Kernel; `arguments` each launch argument's (its buffer names, element
# bytes, binding's first byte in its root, binding bytes; no names for a scalar or the error workspace); `views` as
# metal_ir_upstream took them; `footprints` a CPU kernel's (its one invocation's, metal_ir_upstream_footprints'
# records), None for a Metal kernel's (made over its configured launch, footprints)
Launched = collections.namedtuple("Launched", "operation produced kernel arguments views footprints")


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
    # compiler/mesh_compiler/metal.py _Builder.emitted and .repeated: each direct projection's and produced kernel's
    # actual emission (Emitted) and each registered repetition (Repeated), in emission order
    emitted: tuple = ()
    repeated: tuple = ()
    # an interface receive landed in an input placeholder's storage at its own position (native.py `direct`): none,
    # each received value the produced body's own received buffer (lower_inductor.py recv_strided), each mutated
    # input's store its result's copy (`stores`)
    landings: dict = dataclasses.field(default_factory=dict)
    # each produced kernel's Launched, in emission order: the operand of its composition and terminal emission
    # (`emitted`, after the ports' declaration), its Produced retained through the native target
    upstream: tuple = ()

    def __post_init__(self):
        self.freed = weakref.finalize(self, self.compiler.metal_ir_program_free, self.program)

    def close(self):
        self.freed()


# metal-microbench metal_ir.h (transitional) stages after metal_ir_construct and an instance's port declarations
# (metal_ir_value_receives, metal_ir_value_sends), before metal_ir_lower: each produced kernel of `machine`
# (ConstructedMachine.upstream) in `program` at operation offset `offset` (its metal_ir_append offset there): its
# native composition (metal_ir_upstream_composition, metal_ir_upstream_helper: metal.py Composed, each page's placed
# origin the native's) composed by its producer (lower_inductor.py composed: a Metal Composition or a CPU Hosted) into
# its producer's terminal emission (lower_inductor.py emitted, its Emission.parameters the CPU entry's own) for its
# first pass (metal_ir_upstream_emitted pass 0, which configures its pipeline or library); where it receives (later
# words), its threadgroup footprints over that configured launch (metal_ir_upstream_configured, footprints,
# metal_ir_upstream_footprints) and its later pass's terminal emission (pass 1)
def emitted(program, machine, offset=0):
    for launched in machine.upstream:
        at, kernel = launched.operation + offset, launched.kernel
        made = _composition(program, at)
        _emitted(program, at, 0, producer.emitted(kernel, producer.composed(kernel, made)))
        if made.later_words:
            _footprints(program, at, footprints(launched, _configured(program, at)))
            later = made._replace(later=1)
            _emitted(program, at, 1, producer.emitted(kernel, producer.composed(kernel, later)))


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
# pairs (metal.py SCALAR, operations.py _stored)
def _typed(b):
    if b.dtype.is_complex:
        return SCALAR[b.dtype.to_real()], (*b.size, 2), (*(2 * s for s in b.stride), 1), 2 * b.offset
    return SCALAR[b.dtype], b.size, b.stride, b.offset


def _element(dtype):
    return dtype.to_real() if dtype.is_complex else dtype


def _contiguous(sizes):
    strides, step = [], 1
    for n in reversed(sizes):
        strides.append(step)
        step *= max(n, 1)
    return tuple(reversed(strides))


def _hint(graph, expr):
    return int(graph.sizevars.size_hint(expr))


# torch 2.14.0 torch/utils/_sympy/symbol.py SymT.TMP and INDIRECT: an index symbol a loaded value stands for
def _indirect(symbol):
    return symbol_is_type(symbol, (SymT.TMP, SymT.INDIRECT))


# an index expression's additive terms over `variables`: (its constant, [(coefficient, (variable, divisor, modulus))]),
# each term's atom the variable itself (divisor 1), its FloorDiv by an integer or its ModularIndexing by integers
# (torch/utils/_sympy/functions.py; modulus 0 none); None where some term is none of those
def _atoms(expr, variables):
    constant, made = 0, []
    for term in sympy.Add.make_args(sympy.expand(expr)):
        coefficient, x = term.as_coeff_Mul()
        if x == 1:
            constant += int(coefficient)
            continue
        if not coefficient.is_Integer:
            return None
        if x in variables:
            made.append((int(coefficient), (x, 1, 0)))
        elif (isinstance(x, (FloorDiv, ModularIndexing)) and x.args[0] in variables
              and all(a.is_Integer for a in x.args[1:])):
            made.append((int(coefficient), (x.args[0], int(x.args[1]), int(x.args[2]) if len(x.args) > 2 else 0)))
        else:
            return None
    return constant, made


# the values an atom (divisor `divisor`, modulus `modulus`) of a variable over [0, n) takes, each of them
def _image(n, divisor, modulus):
    count = -(-n // divisor)
    return min(count, modulus) if modulus else count


# torch 2.14.0 torch/_inductor/dependencies.py MemoryDep `dep` of a kernel argument bound from byte `base` of its root
# (`element` its element bytes, `bytes_` its binding's): the bytes it accesses as one mesh_strided view in that root
# (offset, element, ((size, byte stride), ...) outermost first; rdma/mesh-graph.h struct mesh_graph_view), each term of
# its index one dimension of its atom's image (a variable several terms read: their product, which holds it), and
# whether that is its exact form; the whole binding where its index has another form or an indirect symbol, or reaches
# outside the binding (a masked access), or for a StarDep
def _view(graph, dep, element, base, bytes_):
    whole = base, bytes_, (), False
    if not isinstance(dep, dependencies.MemoryDep):
        return whole
    ranges = {v: _hint(graph, n) for v, n in zip(dep.var_names, dep.size)}
    free = dep.index.free_symbols - set(ranges)
    if any(_indirect(s) for s in free):
        return whole
    made = _atoms(dep.index.subs({s: _hint(graph, s) for s in free}), ranges)
    if made is None:
        return whole
    constant, atoms = made
    pairs, used = [], collections.Counter(v for _, (v, _, _) in atoms)
    for c, (v, d, m) in atoms:
        n = _image(ranges[v], d, m)
        constant += min(0, c * max(n - 1, 0))
        pairs.append((n, abs(c) * element))
    pairs.sort(key=lambda pair: -pair[1])
    reach = sum(max(n - 1, 0) * s for n, s in pairs) + element
    if constant < 0 or constant * element + reach > bytes_:
        return whole
    return base + constant * element, element, tuple(pairs), all(k == 1 for k in used.values())


# a strided view's bytes as one run: dense where its dimensions, innermost first, each step its inner ones' extent
def _dense(element, pairs):
    expected = element
    for n, stride in reversed(pairs):
        if n == 1:
            continue
        if stride != expected:
            return False
        expected = stride * n
    return True


# a produced kernel's accesses as the Scheduler relates them (each of its SchedulerNodes' read_writes: MemoryDep and
# StarDep; a WeakDep orders, it accesses nothing), each at the argument binding its buffer name (Scheduler
# mutation_real_name, KernelArgs.input and output) names: (argument, offset, element, pairs, write, exact)
def _accesses(kernel, arguments):
    graph, names = kernel.graph, {}
    real = graph.scheduler.mutation_real_name
    for k, (buffers, _, _, _, access) in enumerate(arguments):
        for name in buffers:
            for write in (0, 1):
                if access & (2 if write else 1):
                    names.setdefault((name, write), k)
    for k, (buffers, _, _, _, _) in enumerate(arguments):
        for name in buffers:
            for write in (0, 1):
                names.setdefault((name, write), k)
    made = []
    for scheduled in kernel.scheduled:
        for node in scheduled.get_nodes():
            for deps, write in ((node.read_writes.reads, 0), (node.read_writes.writes, 1)):
                for dep in deps:
                    k = names.get((real.get(dep.name, dep.name), write))
                    if k is None or isinstance(dep, dependencies.WeakDep):
                        continue
                    _, element, base, bytes_, _ = arguments[k]
                    made.append((k, *_view(graph, dep, element, base, bytes_)[:3], write,
                                 _view(graph, dep, element, base, bytes_)[3]))
    return made


# a CPU kernel's footprints (cpp.py KernelGroup: one native task, its one invocation, threadgroup 0 alone): each
# access's bytes relative to its binding's first byte, exact where its view is exact and dense; each buffer argument
# with no access its whole binding, a bound
def _hosted(arguments, accesses):
    made, touched = [], set()
    for k, offset, element, pairs, _, exact in accesses:
        base = arguments[k][2]
        touched.add(k)
        if any(n == 0 for n, _ in pairs):
            made.append((k, 1, 0, 0, ()))
            continue
        made.append((k, int(exact and _dense(element, pairs)), offset - base,
                     sum((n - 1) * s for n, s in pairs) + element, ()))
    return tuple(made) + tuple((k, 0, 0, bytes_, ()) for k, (names, _, _, bytes_, _) in enumerate(arguments)
                               if names and k not in touched)


# a Metal threadgroup's atom of root r = g G + l (0 <= l < G, G its configured threadgroup extent along dimension
# `dim`): coefficient `c` times (r floordiv d) mod m (m 0 none) as its term over g (mesh_graph_term: dim, coefficient,
# divisor, modulus; None where it has none) and its local digit (floor(l / d') mod n': (d', n'), None where it has
# none); None where it is neither (its period not aligned with the threadgroup's)
def _grouped(c, d, m, dim, G):
    if G % d == 0:
        q = G // d
        if not m:
            return (dim, c * q, 1, 0), (d, q)
        if q % m == 0:
            return None, (d, m)
        if m % q == 0:
            return (dim, c * q, 1, m // q), (d, q)
        return None
    if d % G == 0:
        return (dim, c, d // G, m), None
    return None


# the local digits of each root (digit floor(l / d) mod n of l over [0, extent)) as independent mixed-radix digits of
# one dense run: each root's digits disjoint, each a divisor of its extent, and all of them, by coefficient, each step
# the extent of those below
def _digits_dense(digits, extents):
    radix = []
    for root, held in digits.items():
        reached = 1
        for c, d, n in sorted(held, key=lambda digit: digit[1]):
            if d < reached or extents[root] % (d * n):
                return False
            reached = d * n
            radix.append((abs(c), n))
    expected = 1
    for c, n in sorted(radix):
        if n == 1:
            continue
        if c != expected:
            return False
        expected *= n
    return True


# torch 2.14.0 codegen/mps.py MetalKernel over its configured launch (metal_ir_upstream_configured `grid`, `group`):
# active range tree k (active_range_trees) at thread position dimension k, its root r = g G_k + l (0 <= l < G_k, G_k
# the configured threadgroup's extent there; a reduction tree within each threadgroup, its root over its whole numel,
# multistage loops included); each access its node.codegen made (lower_inductor.py Ranges.accesses: load, store,
# store_reduction) at its index over those roots (range_tree_nodes' entry expressions), the kernel's indirect symbols
# (MeshMetalKernel.mesh_indirect: within [0, size) where checked) local values of their own, every other symbol its size
# hint: per access [start + t(g), + extent) of its binding (start relative to the binding's first byte), t the sum of
# its terms over g; exact where every threadgroup is whole, its body masks nothing (no masked subblock), no index is
# indirect and its local digits are one dense run; the whole binding (a bound) where an atom is not aligned with the
# threadgroup or an index has another form; each buffer argument with no access its whole binding
def _threaded(launched, configured):
    kernel = launched.kernel
    owner, graph, arguments = kernel.owner, kernel.graph, launched.arguments
    group, grid = configured["group"], configured["grid"]
    trees = {}
    for k, tree in enumerate(owner.active_range_trees()):
        trees[tree.symbol()] = (k, _hint(graph, tree.numel) if tree.is_reduction else group[k], tree.is_reduction)
    whole_groups = all(reduction or grid[k] % extent == 0 for k, extent, reduction in trees.values())
    masked = any(ranges.node._body.subblocks for ranges in kernel.ranges)
    entries = {s: e.expr for s, e in owner.range_tree_nodes.items()}
    indirect = {s: (_hint(graph, size), check) for s, (_, size, check) in owner.mesh_indirect.items()}
    real, names = graph.scheduler.mutation_real_name, {}
    for k, (buffers, _, _, _, _) in enumerate(arguments):
        for name in buffers:
            names.setdefault(name, k)
    made, touched = [], set()
    for ranges in kernel.ranges:
        for _, name, index, _ in ranges.accesses:
            k = names.get(real.get(name, name))
            if k is None:
                continue
            touched.add(k)
            element, bytes_ = arguments[k][1], arguments[k][3]
            index = index.subs({s: entries[s] for s in index.free_symbols if s in entries})
            index = index.subs({s: _hint(graph, s) for s in index.free_symbols if s not in trees and s not in indirect})
            atoms = _atoms(index, {**trees, **indirect})
            if atoms is None or any(not indirect[v][1] for _, (v, _, _) in atoms[1] if v in indirect):
                made.append((k, 0, 0, bytes_, ()))
                continue
            constant, terms, digits, exact = atoms[0], [], {}, whole_groups and not masked
            extents = {r: extent for r, (_, extent, _) in trees.items()}
            for c, (v, d, m) in atoms[1]:
                if v in indirect:
                    exact = False
                    extents[v] = indirect[v][0]
                    digits.setdefault(v, []).append((c, d, _image(indirect[v][0], d, m)))
                    continue
                dim, extent, reduction = trees[v]
                split = (None, (d, _image(extent, d, m))) if reduction else _grouped(c, d, m, dim, extent)
                if split is None:
                    break
                term, digit = split
                if term is not None:
                    terms.append((term[0], term[1] * element, term[2], term[3]))
                if digit is not None:
                    digits.setdefault(v, []).append((c, *digit))
            else:
                held = [digit for listed in digits.values() for digit in listed]
                low = constant + sum(min(0, c * (n - 1)) for c, _, n in held)
                made.append((k, int(exact and _digits_dense(digits, extents)), low * element,
                             (sum(abs(c) * (n - 1) for c, _, n in held) + 1) * element, tuple(terms)))
                continue
            made.append((k, 0, 0, bytes_, ()))
    return tuple(made) + tuple((k, 0, 0, bytes_, ()) for k, (buffers, _, _, bytes_, _) in enumerate(arguments)
                               if buffers and k not in touched)


# metal_ir.h metal_ir_upstream_footprints: produced kernel `launched`'s threadgroup footprints over its configured
# launch `configured` (metal.py _configured), each (binding, exact, start, extent, ((dim, coefficient, divisor,
# modulus), ...)) as metal.py _footprints takes them
def footprints(launched, configured):
    return launched.footprints if launched.footprints is not None else _threaded(launched, configured)


# an extern's operands by its operator schema's argument names (ir.py FallbackKernel.unflatten_args over its inputs
# and constant arguments)
def _named(node):
    args, kwargs = node.unflatten_args(node.inputs, node.constant_args)
    return {**{a.name: v for a, v in zip(node.op_overload._schema.arguments, args)}, **kwargs}


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
# Each member's produced entries (lower_inductor.py Produced.entries: Kernel and Extern, the Scheduler's emission order)
# consumed in its own order, the members merged self-timed (Bhattacharyya, Sriram and Lee 1997, §2: a machine-local
# receive after its send). Each name a kernel or extern binds resolved through its GraphLowering (Produced.graph): a
# graph input the machine placeholder's root at its first element (graph_input_storage_offsets; the InputBuffer's
# layout indexes from that element), a constant its own CONSTANT root, a NonOwningLayout its view's storage at the
# view's offset, a MutationLayoutSHOULDREMOVE its target's, a received buffer its transfer's (the declared wire, or the
# machine-local sender's value where the layout is the same, else its own root and one direct copy at the receive),
# every other buffer its own TEMPORARY root of its layout's storage_size; a kernel argument its storage from that first
# byte, its views its Scheduler accesses in its root. Each projection direct over program values: a produced kernel
# metal_ir_upstream, a typed LAPACK operator the native owner's metal_ir_lapack (its ILP64 integers widened copies), its
# INFO boundary metal_ir_check, a copy metal_ir_generic
def construct(module, produced, *, targets, kinds, sources, wires, output_roles, generator=None, machines=None):
    compiler = _compiler()
    builder = _Builder(compiler, 0)
    try:
        if machines:
            compiler.metal_ir_machine(builder.p, C.byref(machines[0]), C.byref(machines[1]))
        nodes, tasks, copied = list(module.graph.nodes), module.meta["tasks"], module.meta["copied"]
        members, remote, end = module.meta["members"], module.meta["transfers"], len(module.meta["order"])
        made = copies(module, wires)
        named = {node.target: node for node in nodes if node.op == "placeholder"}
        storages, placed, held, error = [], {}, {}, []
        env, interface, inputs, landing, sending, grouped = {}, set(), [], [], [], {}
        items, associated, local = [], [], {}

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

        # one name's storage in processor p's GraphLowering: (its Storage, the byte its storage starts at there)
        def located(p, name):
            graph = produced[p].graph
            if name in graph.graph_inputs:
                key = "input", name
            elif name in graph.constants:
                key = "constant", name
            else:
                layout = graph.get_buffer(name).get_layout()
                if isinstance(layout, upstream.NonOwningLayout):
                    view = layout.view.get_layout()
                    storage, base = located(p, layout.view.get_name())
                    return storage, base + int(view.offset) * view.dtype.itemsize
                if isinstance(layout, upstream.MutationLayoutSHOULDREMOVE):
                    return located(p, layout.get_buffer().get_name())
                key = "buffer", name
            if (p, key) not in placed:
                kind, of = key
                if kind == "input":
                    given = next(i for i in produced[p].inputs if i.name == of)
                    b = env[copied[originals[p][given.node]]]
                    placed[p, key] = b.storage, _hint(graph, graph.graph_input_storage_offsets[of]) * b.dtype.itemsize
                elif kind == "constant":
                    data = graph.constants[of]
                    placed[p, key] = (stored(CONSTANT, data.dtype, data.untyped_storage().nbytes(), data),
                                      data.storage_offset() * data.element_size())
                else:
                    layout = graph.get_buffer(of).get_layout()
                    placed[p, key] = stored(TEMPORARY, layout.dtype,
                                            int(layout.storage_size()) * layout.dtype.itemsize), 0
            return placed[p, key]

        # an IR value of processor p (a buffer, a ReinterpretView of one, its TensorBox) at its own layout
        def projected(p, x):
            layout = x.get_layout()
            if isinstance(layout, upstream.MutationLayoutSHOULDREMOVE):
                return projected(p, layout.target)
            storage, base = located(p, x.get_name())
            return Bound(storage, layout.dtype, tuple(int(n) for n in layout.size),
                         tuple(int(n) for n in layout.stride), base // layout.dtype.itemsize + int(layout.offset))

        # a kernel argument's binding: its storage from its first byte, its layout's whole storage_size
        def binding(p, name):
            layout = produced[p].graph.get_buffer(name).get_layout()
            storage, base = located(p, name)
            return Bound(storage, layout.dtype, (int(layout.storage_size()),), (1,), base // layout.dtype.itemsize)

        # a launch's scalar argument: an int64 CONSTANT holding its value (a sizevar or a SymbolicCallArg's numel at
        # its size hint)
        def scalar(graph, expr):
            data = torch.tensor([_hint(graph, getattr(expr, "inner_expr", expr))], dtype=SIZEVAR)
            return (UPSTREAM_SCALAR, Bound(stored(CONSTANT, SIZEVAR, SIZEVAR.itemsize, data), SIZEVAR, (1,), (1,), 0),
                    1, ())

        # a direct elementwise copy of Bound `given` into Bound `into` (metal.py _Builder.copy) at `place`, its source
        # position `at`, owned by `owners`, `record` the producer object it implements; encoded once the roots are made
        def copying(record, into, given, owners, place, at):
            items.append(("copy", (record, into, given, place, at), owners))

        def position(owner):
            return owner[0][0] if owner[0] is not None else end

        # metal.py Storage roots, made once their roles are settled (before any projection is encoded)
        def root_of(storage):
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

        # a declared transfer's logical wire (its one TEMPORARY storage of its logical bytes) viewed at its declared
        # landed or sent view
        def wire(channel, dtype, sizes, landed):
            _, value, size, sent, arrived = wires[channel]
            offset, element, pairs = arrived if landed else sent
            if value not in held:
                held[value] = stored(TEMPORARY, dtype, size)
            return value, Bound(held[value], dtype, tuple(sizes) if landed else tuple(n for n, _ in pairs),
                                tuple(stride // element for _, stride in pairs), offset // element)

        # received buffer `name` of processor p bound to the value `given` it receives: that value's storage where
        # its layout is the buffer's, else its own root and one direct copy of `given` at the receive
        def landed(p, entry, name, given, owner):
            buffer = produced[p].graph.get_buffer(name)
            layout = buffer.get_layout()
            if (layout.dtype, tuple(int(n) for n in layout.size), tuple(int(n) for n in layout.stride)) == (
                    given.dtype, given.size, given.stride):
                placed[p, ("buffer", name)] = (given.storage,
                                               (given.offset - int(layout.offset)) * given.dtype.itemsize)
                return given
            own = projected(p, buffer)
            copying(entry, own, given, [owner], PLACES[produced[p].kind], position(owner))
            return own

        # a received channel at its receive (machine body RECV `m`, the received buffer `name`): in its declared
        # logical wire, landed there; else in the received buffer's own root
        def received(p, entry, name, m):
            channel = m.args[0]
            layout = produced[p].graph.get_buffer(name).get_layout()
            if channel in wires:
                value_, given = wire(channel, layout.dtype, [int(n) for n in layout.size], True)
                env[m] = landed(p, entry, name, given, (tasks[m], None, None, m))
                landing.append((channel, value_, given))
            else:
                env[m] = projected(p, produced[p].graph.get_buffer(name))
                landing.append((channel, None, env[m]))
            interface.add(m)

        # a sent channel at its send (machine body SEND `m`, the sent value `given`): each write group's first member
        # copies the value into its declared wire view at the SEND's position and task; an undeclared one sends the
        # value's own storage
        def sent(p, entry, m, given):
            env[m] = None
            if m not in made.group:
                sending.append((m.args[1], None, given))
                return
            k = made.index[made.group[m]]
            if made.member[k][0] is m:
                value_, view = wire(m.args[1], given.dtype, None, False)
                copying(entry, view, Bound(given.storage, given.dtype, view.size, given.stride, given.offset),
                        [(tasks[m], k, None, m)], PLACES[produced[p].kind], tasks[m][0])
                grouped[k] = value_, view
            sending.append((m.args[1], *grouped[k]))

        # a produced kernel's launch arguments in its launch's order, each (kind, Bound, access, its buffer names):
        # Metal (mps.py launch `values`): output then input buffers less removed_buffers, written then read, then its
        # sizevars and dynamic numels, then its error buffer (the program's error workspace, read and written); CPU
        # (cpp.py KernelArgs.cpp_argdefs): inplace buffers (read and written, every name of theirs), input, output, then
        # sizevars
        def launch_arguments(p, entry):
            graph, args, arguments = produced[p].graph, entry.owner.args, []
            if entry.place == "mps":
                _, values, _, _, _, error_ = entry.launch
                outputs = sum(1 for name in args.output_buffers if name not in entry.owner.removed_buffers)
                for k, v in enumerate(values):
                    arguments.append((UPSTREAM_BUFFER, binding(p, v), 2 if k < outputs else 1, (v,))
                                     if isinstance(v, str) else scalar(graph, v))
                if error_ is not None:
                    arguments.append((UPSTREAM_ERROR, None, 3, ()))
                return arguments
            for inplaced in unique(args.inplace_buffers.values()):
                if not isinstance(inplaced, RemovedArg):
                    arguments.append((UPSTREAM_BUFFER, binding(p, inplaced.other_names[-1]), 3,
                                      tuple(inplaced.other_names)))
            arguments += [(UPSTREAM_BUFFER, binding(p, name), 1, (name,)) for name in args.input_buffers
                          if name not in args.inplace_buffers]
            arguments += [(UPSTREAM_BUFFER, binding(p, name), 2, (name,)) for name, inner in args.output_buffers.items()
                          if name not in args.inplace_buffers and not isinstance(inner, RemovedArg)]
            return arguments + [scalar(graph, expr) for expr in args.sizevars]

        # a produced kernel: its arguments, its sources' positions, its launch
        def kernel(p, entry):
            owners = [owner_of(p, n) for n in entry.origins]
            lead = next((o for o in owners if o[0] is not None), None)
            items.append(("kernel", (p, entry, launch_arguments(p, entry), position(lead) if lead else end), owners))
            associated.extend(o[3] for o in owners if o[3] is not None)

        # a typed LAPACK operator (operations.py LAPACK[packet]: Routine(symbols, arguments, error), its Argument(name,
        # kind, value) records in clapack prototype order for its operand's kind, real or complex) as the native owner's
        # one call (metal_ir_lapack): its job the 'char' arguments naming a str operand, in order (gesdd jobz, heevd
        # jobz and uplo, geev jobvr; a literal the native's own), its rcond the 'scalar' argument's operand, its inputs
        # the 'matrix' arguments' operands, its outputs its results in schema order that a 'matrix' (a matrix per batch
        # member: no job-'N' placeholder), 'vector' or 'info' argument names; its batch its INFO's; each integer
        # result (INFO, RANK) the native's ILP64 integer, a TEMPORARY copied into the result after the call. Its
        # workspace (query and size buffers) the native owner's
        def lapacked(p, entry, node, given, owners, at):
            target, place = node.op_overload, PLACES[produced[p].kind]
            if target not in NATIVE:
                raise NotImplementedError("processor %s: %s (%s) has no native LAPACK routine yet (metal_ir.h "
                                          "METAL_IR_GESDD .. METAL_IR_GELSS)" % (p, node.get_name(), target))
            stated, schema = operations.routine(target), target._schema
            listed = stated.arguments[False] or stated.arguments[True]
            a = next(x.value[0] for x in listed if x.kind == "matrix" and x.value[0] is not None)
            arguments = stated.arguments[given[a].get_dtype().is_complex]
            operands = {x.name for x in schema.arguments}
            job = "".join(given[x.value] for x in arguments if x.kind == "char" and x.value in operands)
            rcond = next((float(given[x.value]) for x in arguments if x.kind == "scalar"), 0.0)
            ins = [projected(p, given[x.value[0]]) for x in arguments if x.kind == "matrix" and x.value[0] is not None]
            routed = {x.value[1] if x.kind == "matrix" else x.value for x in arguments
                      if x.kind in ("vector", "info") or x.kind == "matrix" and x.value[1] is not None}
            results = {r.name: projected(p, out) for r, out in zip(schema.returns, node.outputs)}
            info = next(x.value for x in arguments if x.kind == "info")
            batch = results[info].size
            outs, widened = [], []
            for r in schema.returns:
                b = results.get(r.name) if r.name in routed else None
                if b is None or (len(b.size) - len(batch) != 2 and any(
                        x.kind == "matrix" and x.value[1] == r.name for x in arguments)):
                    continue
                if not (b.dtype.is_floating_point or b.dtype.is_complex) and b.dtype != ILP64:
                    count = 1
                    for n in b.size:
                        count *= n
                    wide = Bound(stored(TEMPORARY, ILP64, count * ILP64.itemsize, identity=at), ILP64, b.size,
                                 _contiguous(b.size), 0)
                    widened.append((b, wide))
                    b = wide
                outs.append((b, len(b.size) - len(batch)))
            items.append(("lapack", (entry, NATIVE[target], job, rcond, batch,
                                     [(b, len(b.size) - len(batch)) for b in ins], outs, at), owners))
            for b, wide in widened:
                copying(entry, b, wide, owners, place, at)

        # an extern of a member: a send or receive (its schema's channel), across machines at this machine body's SEND
        # or RECV (module.meta["transfers"]), within the machine the local sender's value; a typed LAPACK operator; its
        # INFO boundary (metal_ir_check, its api the message, CONVERGENCE: torch.linalg.LinAlgError); a MultiOutput (its
        # buffer its owner's result, which its owner writes). Every other extern has no native consumer yet
        def extern(p, entry):
            node = entry.scheduled.node
            if isinstance(node, upstream.MultiOutput):
                return
            target = getattr(node, "op_overload", None)
            if entry.bodies or target is None:
                raise NotImplementedError("processor %s: upstream extern %s (%s) has no native consumer yet" % (
                    p, node.get_name(), type(node).__name__))
            given, owners = _named(node), [owner_of(p, n) for n in entry.origins]
            lead = next((o for o in owners if o[0] is not None), None)
            at = position(lead) if lead else end
            if target is SEND:
                value = projected(p, given["value"])
                if given["channel"] in remote:
                    sent(p, entry, remote[given["channel"]][0], value)
                else:
                    local[given["channel"]] = value
            elif target is producer.RECV_STRIDED:
                name, channel = node.outputs[0].get_name(), given["channel"]
                if channel in remote:
                    received(p, entry, name, remote[channel][1])
                else:
                    n = next(n for n in entry.origins if getattr(n, "target", None) is RECV)
                    env[vertex(p, n)] = landed(p, entry, name, local[channel], owner_of(p, n))
            elif operations.routine(target) is not None:
                lapacked(p, entry, node, given, owners, at)
                origin = next(n for n in entry.origins if getattr(n, "target", None) is target)
                parts = {u.args[1]: u for u in origin.users if u.target is getattr(__import__("operator"), "getitem")}
                for k, out in enumerate(node.outputs):
                    m = vertex(p, parts[k]) if k in parts else None
                    if m is not None:
                        env[m] = projected(p, out)
                associated.extend(o[3] for o in owners if o[3] is not None)
            elif target is CHECK_ERRORS:
                items.append(("check", (entry, projected(p, given["info"]), CHECK_CONVERGENCE, given["api"], at),
                              owners))
                associated.extend(o[3] for o in owners if o[3] is not None)
            else:
                raise NotImplementedError("processor %s: upstream extern %s (%s) has no native consumer yet" % (
                    p, node.get_name(), target))

        # a member's entry's channel where it is a receive, else None
        def receiving(entry):
            node = entry.scheduled.node if isinstance(entry, producer.Extern) else None
            if getattr(node, "op_overload", None) is producer.RECV_STRIDED:
                return _named(node)["channel"]
            return None

        for p in members:
            if produced[p].kind == "ane":
                raise NotImplementedError("processor %s: its Neural Engine body's real target consumer (its produced "
                                          "CoreML/MIL program) is unfinished" % p)
        at = dict.fromkeys(members, 0)
        while any(at[p] < len(produced[p].entries) for p in members):
            progressed = False
            for p in members:
                while at[p] < len(produced[p].entries):
                    entry = produced[p].entries[at[p]]
                    channel = receiving(entry)
                    if channel is not None and channel not in remote and channel not in local:
                        break
                    (kernel if isinstance(entry, producer.Kernel) else extern)(p, entry)
                    at[p] += 1
                    progressed = True
            if not progressed:
                raise RuntimeError("the produced programs of %s each wait on another's machine-local send: %s" % (
                    ", ".join(members), ", ".join("%s channel %s" % (p, receiving(produced[p].entries[at[p]]))
                                                  for p in members if at[p] < len(produced[p].entries))))

        # each member's returned and sent values (lower_inductor.py Produced.outputs, Output.value the GraphLowering's
        # graph_outputs entry): their projections, the program values of the machine nodes standing for them
        results = {}
        for p in members:
            for out in produced[p].outputs:
                b = (_hint(produced[p].graph, out.value.expr) if isinstance(out.value, upstream.ShapeAsConstantBuffer)
                     else projected(p, out.value))
                m = copied.get(originals[p].get(out.node)) if out.node.op == "placeholder" else vertex(p, out.node)
                if not out.sent:
                    results.setdefault(p, []).append(b)
                if m is not None and m not in env:
                    env[m] = b
        leaves, spec = pytree.tree_flatten(module.graph.output_node().args[0])
        returned = results.get(module.meta["output"], ())

        # roles settled before any root is made: each received channel's storage INPUT, each result's OUTPUT, a
        # storage first declared keeping its role (an input's, a constant's)
        def declared(b, role):
            if b.storage.role == TEMPORARY:
                b.storage.role = role

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
            if (original.storage, original.dtype, original.size, original.stride, original.offset) != (
                    b.storage, b.dtype, b.size, b.stride, b.offset):
                if leaf.op == "placeholder":
                    stated = named[name].meta["stated"]
                    holder = stated if isinstance(stated, str) else module.meta["output"]
                    where, owned_, identity = PLACES[targets[holder][0]], None, end
                else:
                    where, owned_, identity = PLACES[targets[leaf.meta["processor"]][0]], tasks[leaf], tasks[leaf][0]
                copying(leaf, original, b, [(owned_, None, position_, None)], where, identity)
            outputs.append((original, None, kind, name, fake))

        for storage in storages:
            root_of(storage)
        inputs = [Root(name, kind, None, None, None) if b is None else
                  Root(name, kind, b.storage.role, *rooted(b)) for name, kind, b in inputs]
        received_, sent_, ends = {}, {}, {}
        for channel, value_, b in landing:
            received_[channel] = Root(channel, "received", b.storage.role, *rooted(b))
            if value_ is not None:
                ends[value_] = Root(value_, "wire", b.storage.role, received_[channel].root, None)
        for channel, value_, b in sending:
            sent_[channel] = Root(channel, "sent", b.storage.role, *rooted(b))
            if value_ is not None:
                ends.setdefault(value_, Root(value_, "wire", b.storage.role, sent_[channel].root, None))

        # an output's metadata (compiler/mesh_compiler/native.py Bound.result): its view's shape, strides and offset
        # in the root it names, and the exported value's dtype, device and requires_grad
        def described(b, fake):
            return tuple(b.size), tuple(b.stride), b.offset, fake.dtype, fake.device, fake.requires_grad

        outputs = [Output(None, None, literal, None, kind, None) if b is None else
                   Output(*rooted(b), None, described(b, fake), kind, written)
                   for b, literal, kind, written, fake in outputs]

        # metal-microbench metal_ir.h metal_ir_operations: each projection's whole interval of created operation IDs,
        # owned by each source position it implements; each operation's operands as Bounds, in its native order
        # (inputs, then outputs), for its work axes
        operations_, advanced, writes, stores, emitted_, operands, upstream_ = {}, [], {}, {}, {}, {}, []

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

        for kind, payload, owners in items:
            before = compiler.metal_ir_operations(builder.p)
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
            else:
                p, entry, arguments, at_ = payload
                builder.identity = builder.base | at_
                values, described_ = [], []
                for kind_, b, access, buffers in arguments:
                    if kind_ == UPSTREAM_ERROR:
                        v, size = workspace()
                        values.append((kind_, v, access))
                        described_.append((buffers, size, 0, size, access))
                        continue
                    values.append((kind_, value_of(b), access))
                    size = b.size[0] * b.dtype.itemsize
                    described_.append((buffers, b.dtype.itemsize, b.offset * b.dtype.itemsize, size, access))
                accessed = _accesses(entry, described_)
                views = [(k, offset, element, pairs, write) for k, offset, element, pairs, write, _ in accessed]
                views += [(k, 0, size, (), 2) for k, (kind_, _, _, _) in enumerate(arguments)
                          if kind_ == UPSTREAM_ERROR for size in (described_[k][3],)]
                grid = (1, 1, 1) if entry.place == "cpu" else (
                    *(_hint(entry.graph, n) for n in entry.launch[3] or ()), 1, 1, 1)[:3]
                group = None if entry.place == "cpu" or entry.launch[4] is None else (
                    *(_hint(entry.graph, n) for n in entry.launch[4]), 1, 1)[:3]
                op = builder.upstream(entry, PLACES[entry.place], at_, values, views,
                                      [o[0][0] for o in owners if o[0] is not None], grid, group)
                upstream_.append(Launched(op, produced[p], entry, tuple(described_), tuple(views),
                                          _hosted(described_, accessed) if entry.place == "cpu" else None))
                operands[op] = tuple(b for _, b, _, _ in arguments if b is not None)
            owned(owners, list(range(before, compiler.metal_ir_operations(builder.p))) or [op])

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
                made_parts.append(InputPart(processor, ported[copied[whole]], root, value, part, landing_))
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
                              input_parts=input_parts, emitted=tuple(builder.emitted),
                              repeated=tuple(builder.repeated), landings={}, upstream=tuple(upstream_))

# design/compilation-composition-change-request.md items 1, 2, 5, 6 and 7 and its numerical lowerer boundary;
# ~/schedule-emit/schedule_emit/driver.py and runtime.py: one machine's typed program, its processors' produced results
# (compiler/mesh_compiler/lower_inductor.py lower: each processor body's retained GraphLowering, its buffers, layouts
# and aliases, and its backend's emitted calls in their own order) projected once onto the native program
# (metal-microbench metal_ir.h metal_ir_value, metal_ir_value_view, metal_ir_upstream and its stages), within the
# machine's orchestration (compiler/mesh_compiler/schedule.py assemble: its interface, its transfers across machines,
# its results);
# compiler/mesh_compiler/schedule.py Declarations, SourceRelation
import collections
import ctypes as C
import dataclasses
import operator
import weakref

import torch
from torch._inductor import ir as upstream
from torch._inductor.codegen.common import WorkspaceZeroMode
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils import _pytree as pytree

from . import lower_inductor as producer
from .ir import RECV, SEND
from .lower_inductor import copies
from .metal import (CONSTANT, INPUT, NONE, OUTPUT, SCALAR, SOURCE_EXTERNAL, SOURCE_LITERAL, SOURCE_RECEIVED,
                    SOURCE_SENT, SOURCE_VALUE, STATE, TEMPORARY, UPSTREAM_BUFFER, UPSTREAM_ERROR, UPSTREAM_SCALAR,
                    SourceOperand, _Builder, _compiler, _composition, _configured, _emitted, _footprints, _upstream)
from .operations import ANE, CPU, GPU, _copy, _key, _placed, lowered, method

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
# lower_inductor.py Argument.kind (metal_ir.h METAL_IR_UPSTREAM_BUFFER, _SCALAR, _ERROR)
ARGUMENTS = {producer.BUFFER: UPSTREAM_BUFFER, producer.SCALAR: UPSTREAM_SCALAR, producer.ERROR: UPSTREAM_ERROR}
# torch 2.14.0 torch/_inductor/codegen/mps.py codegen_kernel ("constant long& {sizevar} [[buffer(i)]]") and cpp.py
# KernelArgs.cpp_argdefs ("const int64_t"): a launch's scalar argument's type
SIZEVAR = torch.int64


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
# `identity` its definer's source position), its ValueID once made (`root`), and its record operands' storages (`held`:
# compiler/mesh_compiler/operations.py records name their operands as tensors, one meta storage per element type bound
# to this root, metal.py _Builder.bound)
@dataclasses.dataclass(eq=False)
class Storage:
    role: int
    dtype: torch.dtype
    bytes: int
    data: object = None
    identity: int = 0
    zeroed: bool = False
    root: int = None
    held: dict = dataclasses.field(default_factory=dict)


# one value's projection: the view of `storage` as `dtype` at `size` and `stride` from element `offset` (in dtype's
# elements from the storage's first byte), metal_ir_value_view's operands
Bound = collections.namedtuple("Bound", "storage dtype size stride offset")


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
    # the generator root, which the native program's operand starts from (uint32 each)
    initial_words: tuple = ()
    # design/compilation-composition-change-request.md items 6 and 7 (compiler/mesh_compiler/schedule.py analyse):
    # where the program advances its generator, (its state Root, operations.py Generator.counter: the advance's read and
    # written view of that state), else ()
    generated: tuple = ()
    # compiler/mesh_compiler/schedule.py assemble `input_parts`: each original placeholder position's InputParts here
    input_parts: dict = dataclasses.field(default_factory=dict)
    # compiler/mesh_compiler/metal.py _Builder.emitted and .repeated: each record's and produced kernel's actual
    # emission (Emitted) and each registered repetition (Repeated), in emission order
    emitted: tuple = ()
    repeated: tuple = ()
    # an interface receive landed in an input placeholder's storage at its own position (native.py `direct`): none,
    # each received value the produced body's own received buffer (lower_inductor.py recv_strided), each mutated
    # input's store its result's copy (`stores`)
    landings: dict = dataclasses.field(default_factory=dict)
    # each produced kernel: (its operation, its Produced, its Kernel, each argument binding's first byte in its root),
    # in emission order, the operand of its composition and terminal emission (`emitted`, after the ports' declaration)
    upstream: tuple = ()

    def __post_init__(self):
        self.freed = weakref.finalize(self, self.compiler.metal_ir_program_free, self.program)

    def close(self):
        self.freed()


# metal-microbench metal_ir.h (transitional) stages after metal_ir_construct and an instance's port declarations
# (metal_ir_value_receives, metal_ir_value_sends), before metal_ir_lower: each produced kernel of `machine`
# (ConstructedMachine.upstream) in `program` at operation offset `offset` (its metal_ir_append offset there): its
# composition and transport helper (metal_ir_upstream_composition, metal_ir_upstream_helper: metal.py Composed) given
# to its producer's terminal emission (lower_inductor.py emitted) for its first pass (metal_ir_upstream_emitted pass 0,
# which configures its pipeline or library); where it receives (later words), its producer's threadgroup footprints
# over that configured launch (metal_ir_upstream_configured, lower_inductor.py footprints, metal_ir_upstream_footprints)
# and its later pass's terminal emission (pass 1)
def emitted(program, machine, offset=0):
    for op, produced, kernel, origins in machine.upstream:
        at = op + offset
        composed = _composition(program, at, origins)
        _emitted(program, at, 0, producer.emitted(produced, kernel, composed))
        if composed.later_words:
            _footprints(program, at, producer.footprints(produced, kernel, _configured(program, at)))
            _emitted(program, at, 1, producer.emitted(produced, kernel, composed._replace(later=1)))


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
# graph input the machine placeholder's root at its first element (graph_input_storage_offsets, lower_inductor.py
# Input.offset; the InputBuffer's indexes from that element), a constant its own CONSTANT root, a NonOwningLayout its
# view's storage at the view's offset, a MutationLayoutSHOULDREMOVE its target's, a received buffer its transfer's
# (the declared wire, or the machine-local sender's value where the layout is the same, else its own root and one copy
# at the receive), every other buffer its own TEMPORARY root of its layout's storage_size (workspace the allocation
# owner places); a kernel argument its storage from that first byte, its views (lower_inductor.py views) shifted to
# that byte in its root
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
        storages, placed, held, standins, bound_of = [], {}, {}, {}, {}
        env, interface, inputs, landing, sending, grouped = {}, set(), [], [], [], {}
        items, associated, advance, local, upstream_ = [], [], None, {}, []

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
                    placed[p, key] = b.storage, int(given.offset) * b.dtype.itemsize
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

        # a record's operand (operations.py records over tensors): b's view of a meta storage of its storage's bytes in
        # its element type, that storage bound to b's root
        def record_operand(b):
            if b not in standins:
                storage = b.storage
                if b.dtype not in storage.held:
                    storage.held[b.dtype] = torch.empty(storage.bytes // b.dtype.itemsize, dtype=b.dtype,
                                                        device="meta")
                    if storage.root is not None:
                        builder.bound(storage.held[b.dtype], storage.root, storage.role)
                standins[b] = storage.held[b.dtype].as_strided(b.size, b.stride, b.offset)
                bound_of[id(standins[b])] = b
            return standins[b]

        def recorded(records, owner, at, identity):
            _placed(records, at, identity)
            items.append(("records", records, owner))

        def position(owner):
            return owner[0][0] if owner[0] is not None else end

        # metal.py Storage roots, made once their roles are settled (before any record or kernel is encoded)
        def root_of(storage):
            if storage.root is None:
                element = _element(storage.dtype)
                storage.root = builder.stored(storage.role, SCALAR[element], storage.bytes // element.itemsize,
                                              storage.identity)
                if storage.data is not None:
                    builder.attach(storage.root, storage.data)
                if storage.zeroed:
                    _upstream().metal_ir_value_zeroed(builder.p, storage.root)
                for standin in storage.held.values():
                    builder.bound(standin, storage.root, storage.role)
            return storage.root

        def value_of(b):
            scalar, sizes, strides, offset = _typed(b)
            return builder.viewed(root_of(b.storage), b.storage.role, scalar, sizes, strides, offset)

        def rooted(b):
            return root_of(b.storage), value_of(b)

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
        # its layout is the buffer's, else its own root and one copy of `given` at the receive
        def landed(p, name, given, owner):
            buffer = produced[p].graph.get_buffer(name)
            layout = buffer.get_layout()
            if (layout.dtype, tuple(int(n) for n in layout.size), tuple(int(n) for n in layout.stride)) == (
                    given.dtype, given.size, given.stride):
                placed[p, ("buffer", name)] = (given.storage,
                                               (given.offset - int(layout.offset)) * given.dtype.itemsize)
                return given
            own = projected(p, buffer)
            recorded(_copy(record_operand(own), record_operand(given)), owner, PLACES[produced[p].kind],
                     position(owner))
            return own

        # a received channel at its receive (machine body RECV `m`, the received buffer `name`): in its declared
        # logical wire, landed there; else in the received buffer's own root
        def received(p, entry, m):
            channel, name = m.args[0], entry.node.get_name()
            layout = produced[p].graph.get_buffer(name).get_layout()
            if channel in wires:
                value_, given = wire(channel, layout.dtype, [int(n) for n in layout.size], True)
                env[m] = landed(p, name, given, (tasks[m], None, None, m))
                landing.append((channel, value_, given))
            else:
                env[m] = projected(p, produced[p].graph.get_buffer(name))
                landing.append((channel, None, env[m]))
            interface.add(m)

        # a sent channel at its send (machine body SEND `m`, the sent value `given`): each write group's first member
        # copies the value into its declared wire view at the SEND's position and task; an undeclared one sends the
        # value's own storage
        def sent(p, m, given):
            env[m] = None
            if m not in made.group:
                sending.append((m.args[1], None, given))
                return
            k = made.index[made.group[m]]
            if made.member[k][0] is m:
                value_, view = wire(m.args[1], given.dtype, None, False)
                recorded(_copy(record_operand(view),
                               record_operand(Bound(given.storage, given.dtype, view.size, given.stride,
                                                    given.offset))),
                         (tasks[m], k, None, m), PLACES[produced[p].kind], tasks[m][0])
                grouped[k] = value_, view
            sending.append((m.args[1], *grouped[k]))

        # an IR value an extern takes: a ShapeAsConstantBuffer its value (the graph's size hint), else its projection
        def taken(p, x):
            if isinstance(x, upstream.ShapeAsConstantBuffer):
                return int(produced[p].graph.sizevars.size_hint(x.expr))
            return record_operand(projected(p, x))

        # an explicit method extern (operations.py method, lowered): its records over the extern's own operands
        # (FallbackKernel.unflatten_args over its inputs and constant arguments) and its results in schema order, at
        # its node's place and position; a draw its generator's
        def extern(p, entry):
            nonlocal advance
            node, n = entry.node, next(n for n in entry.origins if method(getattr(n, "target", None)) is not None)
            args, kwargs = node.unflatten_args([taken(p, x) for x in node.inputs], node.constant_args)
            outs = [projected(p, produced[p].graph.get_buffer(name)) for name in entry.outputs]
            parts = {u.args[1]: u for u in n.users if u.target is operator.getitem}
            for k, b in enumerate(outs):
                m = vertex(p, parts[k]) if k in parts else vertex(p, n) if not parts else None
                if m is not None:
                    env[m] = b
            drew = generator.sites if generator is not None else 0
            owner = owner_of(p, n)
            records = lowered(entry.kernel, args, kwargs, [record_operand(b) for b in outs], generator)
            recorded(records, owner, PLACES[produced[p].kind], position(owner))
            associated.extend(m for m in (owner[3],) if m is not None)
            if generator is not None and generator.sites > drew:
                advance = PLACES[produced[p].kind]

        # a produced kernel: its arguments' bindings (a buffer's storage, a scalar's CONSTANT, the error buffer's and
        # each workspace's TEMPORARY), its sources' positions, its launch
        def kernel(p, entry):
            graph, arguments = produced[p].graph, []
            workspace = {w.outer_name: w for w in entry.workspace}
            owners = [owner_of(p, n) for n in entry.origins]
            lead = next((o for o in owners if o[0] is not None), None)
            first = position(lead) if lead is not None else end
            for argument in entry.arguments:
                if argument.kind == producer.BUFFER and argument.name in workspace:
                    w = workspace[argument.name]
                    count = int(graph.sizevars.size_hint(w.count))
                    b = Bound(stored(TEMPORARY, w.dtype, count * w.dtype.itemsize, identity=first,
                                     zeroed=w.zero_mode == WorkspaceZeroMode.ZERO_PER_GRAPH), w.dtype, (count,), (1,),
                              0)
                    if w.zero_mode == WorkspaceZeroMode.ZERO_ON_CALL:
                        recorded(_copy(record_operand(b), torch.zeros((), dtype=w.dtype)),
                                 lead or (None, None, None, None), PLACES[entry.place], first)
                elif argument.kind == producer.BUFFER:
                    b = binding(p, argument.name)
                elif argument.kind == producer.SCALAR:
                    data = torch.tensor([int(graph.sizevars.size_hint(argument.value))], dtype=SIZEVAR)
                    b = Bound(stored(CONSTANT, SIZEVAR, SIZEVAR.itemsize, data), SIZEVAR, (1,), (1,), 0)
                else:
                    b = Bound(stored(TEMPORARY, torch.uint8, argument.bytes, identity=first), torch.uint8,
                              (argument.bytes,), (1,), 0)
                arguments.append((ARGUMENTS[argument.kind], b, argument.access))
            items.append(("kernel", (p, entry, arguments, owners, first), None))
            associated.extend(o[3] for o in owners if o[3] is not None)

        # a member's entry: a send or receive (lower_inductor.py Extern `ordered`, its channel), across machines at
        # this machine body's SEND or RECV (module.meta["transfers"]), within the machine the local sender's value; an
        # explicit method extern; a produced kernel. Its Neural Engine body has no consumer yet
        def consumed(p, entry):
            if isinstance(entry, producer.Kernel):
                kernel(p, entry)
            elif entry.kernel is SEND:
                given = projected(p, entry.node.inputs[0])
                if entry.channel in remote:
                    sent(p, remote[entry.channel][0], given)
                else:
                    local[entry.channel] = given
            elif entry.kernel is producer.RECV_STRIDED:
                if entry.channel in remote:
                    received(p, entry, remote[entry.channel][1])
                else:
                    n = next(n for n in entry.origins if getattr(n, "target", None) is RECV)
                    m = vertex(p, n)
                    env[m] = landed(p, entry.node.get_name(), local[entry.channel], owner_of(p, n))
            elif method(entry.kernel) is not None:
                extern(p, entry)
            else:
                raise NotImplementedError("processor %s: upstream extern %s (%s) has no native consumer yet"
                                          % (p, entry.node.get_name(), entry.kernel))

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
                    if (isinstance(entry, producer.Extern) and entry.kernel is producer.RECV_STRIDED
                            and entry.channel not in remote and entry.channel not in local):
                        break
                    consumed(p, entry)
                    at[p] += 1
                    progressed = True
            if not progressed:
                raise RuntimeError("the produced programs of %s each wait on another's machine-local send: %s" % (
                    ", ".join(members), ", ".join("%s channel %s" % (p, produced[p].entries[at[p]].channel)
                                                  for p in members if at[p] < len(produced[p].entries))))

        # each member's returned and sent values (lower_inductor.py Produced.outputs, GraphLowering.graph_outputs):
        # their projections, the program values of the machine nodes standing for them
        results = {}
        for p in members:
            for out in produced[p].outputs:
                ir = produced[p].graph.graph_outputs[out.position]
                b = (int(produced[p].graph.sizevars.size_hint(ir.expr))
                     if isinstance(ir, upstream.ShapeAsConstantBuffer) else projected(p, ir))
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
                recorded(_copy(record_operand(original), record_operand(b)), (owned_, None, position_, None), where,
                         identity)
            outputs.append((original, None, kind, name, fake))
        if advance is not None:
            builder.declare(generator.state, STATE)
            stepped = generator.stepped()
            _placed(stepped, advance, end)
            items.append(("records", stepped, None))
        else:
            stepped = []

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
        state = (Root("generator", "generator", STATE, builder.roots[_key(generator.state)],
                      builder.value(generator.state)),) if advance is not None else ()

        # metal-microbench metal_ir.h metal_ir_operations: each record's whole interval of created operation IDs (a
        # constructor such as metal_ir_sort expands into several, a repetition's its pass's nested ones), its returned
        # ID where it created none and returned one; each produced kernel its one operation, owned by each source
        # position it implements; each node's leaves (metal.py _Builder.emitted, the slice its records emitted)
        operations, advanced, writes, stores, emitted_, leaves_ = {}, [], {}, {}, {}, {}

        def owned(owner, ids, first):
            if owner is None:
                advanced.extend(ids)
                return
            task_, write, store, node = owner
            if task_ is not None:
                operations.setdefault(task_, []).extend(ids)
            if node is not None:
                emitted_.setdefault(node, []).extend(ids)
                leaves_.setdefault(node, []).extend(builder.emitted[first:])
            if write is not None:
                writes.setdefault(write, []).extend(ids)
            if store is not None:
                stores.setdefault(store, []).extend(ids)

        for kind, payload, owner in items:
            if kind == "records":
                for g in payload:
                    before, first = compiler.metal_ir_operations(builder.p), len(builder.emitted)
                    op = builder.generic(g)
                    owned(owner, list(range(before, compiler.metal_ir_operations(builder.p))) or (
                        [op] if op is not None and op != NONE else []), first)
                continue
            p, entry, arguments, owners, at_ = payload
            builder.identity = builder.base | at_
            values = [(k, value_of(b), access) for k, b, access in arguments]
            origins = tuple(b.offset * b.dtype.itemsize for _, b, _ in arguments)
            views = [(k, offset + origins[k], element, pairs, write)
                     for k, offset, element, pairs, write in producer.views(produced[p], entry)]
            first = len(builder.emitted)
            op = builder.upstream(entry, PLACES[entry.place], at_, values, views,
                                  [o[0][0] for o in owners if o[0] is not None],
                                  (*(entry.grid or ()), 1, 1, 1)[:3],
                                  None if entry.group is None else (*entry.group, 1, 1)[:3])
            upstream_.append((op, produced[p], entry, origins))
            for o in owners:
                owned(o, [op], first)

        # metal-microbench metal_ir.h metal_ir_operation_axis: each of a numerical node's work axes
        # (compiler/mesh_compiler/placement.py `work_axes`: source node, dimension, original full extent), on each
        # operation its records emitted at each operand that is that source node's own program value, that dimension;
        # with no such operand (a produced kernel binds whole storages), on each operation it emitted with none
        for node in dict.fromkeys(associated):
            for reference, dimension, full in node.meta.get("work_axes", ()):
                b = env.get(copied.get(reference))
                hits = [(leaf.operation, s, dimension) for leaf in leaves_.get(node, ())
                        for s, t in enumerate([t for t, _ in [*getattr(leaf.record, "inputs", ()),
                                                               *getattr(leaf.record, "outputs", ())]])
                        if isinstance(b, Bound) and bound_of.get(id(t)) == b]
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
    return ConstructedMachine(builder.p, compiler, dict(targets), tuple(inputs), tuple(outputs), spec, state,
                              received_, sent_, ends, {task_: tuple(ops) for task_, ops in operations.items()},
                              tuple(advanced), made, {k: tuple(ops) for k, ops in writes.items()},
                              {k: tuple(ops) for k, ops in stores.items()}, source, vertices, accesses, given, writers,
                              builder.base, {k: tuple(sorted(ops)) for k, ops in decomposed.items()},
                              initial_words=tuple(word & 0xFFFFFFFF for word in generator.state.tolist()) if state
                              else (), generated=(state[0], generator.counter) if stepped else (),
                              input_parts=input_parts, emitted=tuple(builder.emitted),
                              repeated=tuple(builder.repeated), landings={}, upstream=tuple(upstream_))

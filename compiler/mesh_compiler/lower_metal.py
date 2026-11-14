# design/compilation-composition-change-request.md items 1, 2, 6 and 7; design/compiler-workgroup-2026-11-13.md;
# ~/schedule-emit/schedule_emit/driver.py and lower_inductor.py lower: a stage takes one machine's assembled body and
# returns its one typed program, built by the one numerical construction (compiler/mesh_compiler/operations.py
# _instruction) and encoded by compiler/mesh_compiler/metal.py _Builder;
# metal-microbench metal_ir.h (metal_ir_value, metal_ir_value_view); compiler/mesh_compiler/schedule.py Declarations,
# assemble, SourceRelation
import collections
import ctypes as C
import dataclasses
import itertools
import weakref

import torch
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.node import map_arg
from torch.utils import _pytree as pytree

from .ir import RECV, SEND, written
from .lower_inductor import copies
from .metal import (CONSTANT, INPUT, NONE, OUTPUT, RELEASE, SOURCE_EXTERNAL, SOURCE_LITERAL, SOURCE_RECEIVED,
                    SOURCE_SENT, SOURCE_VALUE, STATE, TEMPORARY, SourceOperand, _Builder, _compiler)
from .operations import (ANE, CPU, GPU, _Generic, _Lapack, _copy, _held, _instruction, _key, _local_call, _metadata,
                         _placed, _same_view, _stored)

Root = collections.namedtuple("Root", "name kind role root value")
Output = collections.namedtuple("Output", "root value literal metadata kind written")
# compiler/mesh_compiler/placement.py place `input_parts`: one placed partition of an input on one processor, `input`
# the whole input's placeholder Root, `root`/`value` the partition's typed root and ValueID (`typed`), `source` the
# original partition Node, `landing` the ValueID of its source view (its landing geometry) in `input.root`, None for a
# non-tensor
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


# design/compilation-composition-change-request.md items 6, 6.4 and 7; metal_ir.h struct metal_ir_source_operand
# (metal.SourceOperand): a GIVEN cause's typed operand, a value's (root, view) its view's ValueID, a literal already in
# the typed body, a channel port Root its SENT/RECEIVED logical ValueID, None external
def operand(given):
    if given is None:
        return SourceOperand(SOURCE_EXTERNAL, NONE)
    if isinstance(given, Root):
        return SourceOperand(SOURCE_SENT if given.kind == "sent" else SOURCE_RECEIVED, given.value)
    return SourceOperand(SOURCE_LITERAL, NONE) if given[0] == "literal" else SourceOperand(SOURCE_VALUE, given[1])


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
    relation: object
    vertices: tuple
    accesses: tuple
    supplied: tuple
    writers: tuple
    identity: int = 0
    binding: tuple = ()
    # compiler/mesh_compiler/decompose.py decompose: each expanded original source node here, its expanded nodes'
    # emitted operation IDs (module.meta["decomposed"]), in emission order
    decomposed: dict = dataclasses.field(default_factory=dict)
    # compiler/mesh_compiler/operations.py Generator `state`, its words when the program draws: the initial STATE of
    # the generator root, which the native program's operand starts from (uint32 each)
    initial_words: tuple = ()
    # compiler/mesh_compiler/schedule.py assemble `input_parts`: each original placeholder position's InputParts here
    input_parts: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.freed = weakref.finalize(self, self.compiler.metal_ir_program_free, self.program)

    def close(self):
        self.freed()


# torch 2.14.0 torch/_tensor_docs.py as_strided, untyped_storage; torch/_export/non_strict_utils.py fakify: one meta root
# a source storage as the export's fake values alias it (every placeholder kind in that one namespace), each
# placeholder its view with its own shape, strides and offset; a CONSTANT root's initial bytes the exported tensor's
def placeholders(nodes, kinds, sources):
    found = {}
    for node in nodes:
        value = node.meta.get("val") if node.op == "placeholder" else None
        if isinstance(value, torch.Tensor):
            found.setdefault(value.untyped_storage()._cdata, []).append(node)
    roots = {}
    for viewed in found.values():
        roles = {KINDS[kinds.get(node.target, InputKind.USER_INPUT)][1] for node in viewed}
        role = next(role for role in AUTHORITY if role in roles)
        first = viewed[0].meta["val"]
        root = torch.empty(first.untyped_storage().nbytes() // first.element_size(), dtype=first.dtype, device="meta")
        data = sources[viewed[0].target] if role == CONSTANT else None
        for node in viewed:
            value = node.meta["val"]
            base = root if value.dtype == root.dtype else root.view(value.dtype)
            roots[node] = base.as_strided(value.shape, value.stride(), value.storage_offset()), role, data
    return roots


# compiler/mesh_compiler/schedule.py assemble keyed by machine, Declarations.wires: `module` one machine's body in the
# retained global task order, `targets` {original processor: (selected kind "cpu" | "mps" | "ane", torch.device)} the
# driver's selection, `wires` channel -> (declared transfer, logical wire value, its logical bytes, declared sent view,
# declared landed view), `output_roles` one (OutputKind, placeholder name or None) per output leaf
# (Scheduled.exported.outputs); compiler/mesh_compiler/lower_inductor.py Copies, copies: a view (offset, element,
# (size, byte stride) pairs) of the wire value's logical bytes; J. K. Salmon et al., SC 2011, §3.1: `generator` the
# machine's operations.Generator; compiler/mesh_compiler/schedule.py SourceRelation, assemble `vertices`: the source
# relation's correspondence here, each vertex's emitted operation IDs, each access view's (typed root, typed view) and
# each GIVEN cause's typed operand (root and view, or literal), a channel its port Root, a same-machine channel (its
# SEND/RECV an ordinary value, `copied`) its value's (root, view), None where this machine holds none
def construct(module, *, targets, kinds, sources, wires, output_roles, generator=None, page=0, machines=None):
    compiler = _compiler()
    builder = _Builder(compiler, {}, None, 0, 0)
    try:
        if machines:
            compiler.metal_ir_machine(builder.p, C.byref(machines[0]), C.byref(machines[1]))
        nodes, tasks = list(module.graph.nodes), module.meta["tasks"]
        roots, made = placeholders(nodes, kinds, sources), copies(module, wires)
        named = {node.target: node for node in nodes if node.op == "placeholder"}
        byname = {node.name: node for node in nodes}
        env, records, owners, inputs, landing, sending, held, grouped = {}, [], [], [], [], [], {}, {}
        dests, advance, axes = {}, None, {}

        # metal-microbench metal_ir.h metal_ir_operation_axis: each of a numerical node's work axes
        # (compiler/mesh_compiler/placement.py `work_axes`: source node, dimension, original full extent) at each operand
        # of its records `ops` that is that source node's own tensor (env, by identity, before Builder.value names typed
        # views), that dimension, and at each operand a record's explicit `source_axes` (operations.py _Generic) gives
        # that tensor's dimension, the operand dimension it names; by record index from `first`: a _Generic's or
        # _Lapack's (metal.py _Builder.generic: metal_ir_generic and metal_ir_lapack make the one operation of its
        # inputs, then outputs, in order), another record exposing no operand ordinal; a reference none of them has,
        # with no operand, on each record
        def associate(node, ops, first):
            for reference, dimension, full in node.meta.get("work_axes", ()):
                tensor = env.get(module.meta["copied"].get(reference))
                hits = [(i, s, dimension) for i, g in enumerate(ops) if isinstance(g, (_Generic, _Lapack))
                        for s, (t, _) in enumerate([*g.inputs, *g.outputs]) if tensor is not None and t is tensor]
                hits += [(i, s, at) for i, g in enumerate(ops) if isinstance(g, _Generic)
                         for source, d, s, at in g.source_axes
                         if tensor is not None and source is tensor and d == dimension]
                for i, s, at in hits or [(i, NONE, dimension) for i in range(len(ops))]:
                    axes.setdefault(first + i, []).append((s, at, tasks[node][0], full))

        def place(node):
            return PLACES[targets[node.meta["processor"]][0]]

        def rooted(t, role, data=None):
            return builder.declare(t, role, data), builder.value(_stored(t))

        # compiler/mesh_compiler/ir.py views, destined, based, written: a call's "out" not of an operand it writes, and a
        # RECV's "to", name the CAT/PAD result it is stored into and its view (offset, sizes, the result's own strides)
        # of that result's own layout; the result's typed root declared from its metadata before its producer, the
        # view placed at its index offset in the result's realized view, each result before its operands (reverse
        # graph order); torch 2.14.0 torch/export/graph_signature.py OutputSpec, USER_INPUT_MUTATION and
        # BUFFER_MUTATION `target`: a mutation's value's "out" names the input placeholder it writes, its destination
        # that input's declared typed view (placeholders), so the value and its views are the input's root
        def destined(node):
            return node.meta.get("out") is not None and written(node) is None

        def placed(record):
            name, offset, sizes, strides = record
            if byname[name] not in dests and byname[name] not in roots:
                dests[byname[name]] = _held(byname[name].meta["val"])
            base = roots[byname[name]][0] if byname[name] in roots else dests[byname[name]]
            at = base.storage_offset()
            for d in sorted(range(base.dim()), key=lambda d: -strides[d]):
                if base.shape[d] > 1:
                    at, offset = at + offset // strides[d] * base.stride(d), offset % strides[d]
            return base.as_strided(sizes, base.stride(), at)

        for node in reversed(nodes):
            if destined(node):
                dests[node] = placed(node.meta["out"])

        def wire(channel, like, landed):
            _, value, size, sent, arrived = wires[channel]
            offset, element, pairs = arrived if landed else sent
            if value not in held:
                held[value] = torch.empty(size // like.element_size(), dtype=like.dtype, device="meta")
            sizes = list(like.shape) if landed else [n for n, _ in pairs]
            return value, held[value].as_strided(sizes, [stride // element for _, stride in pairs], offset // element)

        for node in nodes:
            if node.op == "placeholder":
                kind = KINDS[kinds.get(node.target, InputKind.USER_INPUT)][0]
                if node not in roots:
                    env[node] = node.meta.get("val")
                    inputs.append(Root(node.target, kind, None, None, None))
                    continue
                env[node], role, data = roots[node]
                inputs.append(Root(node.target, kind, role, *rooted(env[node], role, data)))
            elif node.op == "output":
                leaves, spec = pytree.tree_flatten(node.args[0])
            elif node.target is RECV:
                channel = node.args[0]
                if "to" in node.meta:
                    value, env[node] = wires[channel][1] if channel in wires else None, placed(node.meta["to"])
                else:
                    value, env[node] = wire(channel, node.meta["val"], True) if channel in wires else (
                        None, _held(node.meta["val"]))
                landing.append((channel, value, env[node]))
            elif node.target is SEND:
                env[node] = None
                if node not in made.group:
                    sending.append((node.args[1], None, env[node.args[0]]))
                    continue
                k = made.index[made.group[node]]
                if made.member[k][0] is node:
                    value, view = wire(node.args[1], env[node.args[0]], False)
                    given = env[node.args[0]]
                    ops = _copy(view, given.as_strided(view.shape, given.stride(), given.storage_offset()))
                    _placed(ops, place(node), tasks[node][0])
                    records += ops
                    owners += [(tasks[node], k, None, node)] * len(ops)
                    grouped[k] = value, view
                sending.append((node.args[1], *grouped[k]))
            else:
                local = _local_call(node)
                args, kwargs = map_arg((local.args, local.kwargs), env.__getitem__)
                drew = generator.sites if generator is not None else 0
                env[node], ops = _instruction(local, args, kwargs, node.meta.get("val"), page, generator,
                                                [dests[node]] if node in dests else None)
                _placed(ops, place(node), tasks[node][0])
                associate(node, ops, len(records))
                records += ops
                owners += [(tasks[node], None, None, node)] * len(ops)
                if generator is not None and generator.sites > drew:
                    advance = place(node)
        received, sent, ends = {}, {}, {}
        for channel, value, t in landing:
            received[channel] = Root(channel, "received", INPUT, *rooted(t, INPUT))
            if value is not None:
                ends[value] = Root(value, "wire", INPUT, received[channel].root, None)
        # an output's metadata (compiler/mesh_compiler/native.py Bound.result): its typed view's shape, strides and
        # offset in the root it names, an input's real view where it is the input's (a returned view of a mutated
        # input), and the exported value's dtype, device and requires_grad
        def described(view, fake):
            return (*_metadata(view)[:3], *_metadata(fake)[3:])

        outputs = []
        for position, (leaf, (kind, name)) in enumerate(zip(leaves, output_roles, strict=True)):
            value = env[leaf] if isinstance(leaf, torch.fx.Node) else leaf
            if not isinstance(value, torch.Tensor):
                outputs.append(Output(None, None, value, None, kind, None))
                continue
            fake = leaf.meta["val"] if isinstance(leaf, torch.fx.Node) else value
            if kind not in MUTATIONS:
                outputs.append(Output(*rooted(value, OUTPUT), None, described(value, fake), kind, None))
                continue
            original = env[named[name]]
            if not _same_view(value, original):
                if leaf.op == "placeholder":
                    stated = named[name].meta["stated"]
                    holder = stated if isinstance(stated, str) else module.meta["output"]
                    at, task, identity = PLACES[targets[holder][0]], None, len(module.meta["order"])
                else:
                    at, task, identity = place(leaf), tasks[leaf], tasks[leaf][0]
                ops = _copy(original, value)
                _placed(ops, at, identity)
                records += ops
                owners += [(task, None, position, None)] * len(ops)
            outputs.append(Output(*rooted(original, INPUT), None, described(original, fake), kind, name))
        for channel, value, t in sending:
            root, view = (rooted(t, TEMPORARY) if value is not None else (builder.root(t), builder.value(_stored(t))))
            role = builder.roles[_key(t)]
            sent[channel] = Root(channel, "sent", role, root, view)
            if value is not None:
                ends.setdefault(value, Root(value, "wire", role, root, None))
        state, stepped = (), []
        if advance is not None:
            state = (Root("generator", "generator", STATE, *rooted(generator.state, STATE)),)
            stepped = generator.stepped()
            _placed(stepped, advance, len(module.meta["order"]))
        # metal-microbench metal_ir.h metal_ir_operations: each record's whole interval of created operation IDs (a
        # constructor such as metal_ir_sort expands into several), its returned ID where it created none
        operations, advanced, writes, stores, emitted = {}, [], {}, {}, {}
        for index, (owner, g) in enumerate(zip(owners + [None] * len(stepped), records + stepped)):
            before = compiler.metal_ir_operations(builder.p)
            op = builder.generic(g)
            made = list(range(before, compiler.metal_ir_operations(builder.p))) or [op]
            # `associate`'s: an operand's on the record's own operation, one with no operand on each created
            for operand_index, dimension, task, full in axes.get(index, ()):
                for at in made if operand_index == NONE else (op,):
                    status = compiler.metal_ir_operation_axis(builder.p, at, operand_index, dimension, task, full)
                    if status:
                        raise OSError(status, "the work axis of operation %d operand %d" % (at, operand_index))
            if owner is None:
                advanced += made
                continue
            task, write, store, node = owner
            if task is not None:
                operations.setdefault(task, []).extend(made)
            if node is not None:
                emitted.setdefault(node, []).extend(made)
            if write is not None:
                writes.setdefault(write, []).extend(made)
            if store is not None:
                stores.setdefault(store, []).extend(made)
        source, copied = module.meta["relation"], module.meta["copied"]
        builder.identity = builder.base

        def typed(node):
            if copied.get(node) not in env:
                return None
            value = env[copied[node]]
            if not isinstance(value, torch.Tensor):
                return "literal", value
            view = builder.value(_stored(value))
            return builder.roots[_key(value)], view

        # compiler/mesh_compiler/schedule.py assemble `input_parts`: each partition's typed root and ValueID by `typed`,
        # its whole input the Root the placeholder loop appended for that placeholder's copy, its landing the ValueID of
        # its landing geometry (storage offset, shape, stride) on that placeholder's own typed view, as_strided
        def landed(whole, geometry):
            if geometry is None:
                return None
            offset, sizes, strides = geometry
            return builder.value(_stored(env[copied[whole]].as_strided(sizes, strides, offset)))
        ported = dict(zip([node for node in nodes if node.op == "placeholder"], inputs, strict=True))
        input_parts = {position: tuple(InputPart(processor, ported[copied[whole]], *typed(part), part,
                                                 landed(whole, geometry))
                                       for processor, whole, part, geometry in parts)
                       for position, parts in module.meta.get("input_parts", {}).items()}

        vertices = tuple(tuple(emitted.get(node, ())) for node in module.meta["vertices"])
        accesses = tuple(typed(node) for node, _ in source.accesses)
        given = tuple(typed(operand) if kind == "value" else sent.get(operand) or received.get(operand) or typed(sink)
                      for kind, _, sink, operand in source.supplied)
        writers = tuple((tasks[a], tasks[b]) for a, b in module.meta["writers"])
        # metal-microbench metal_ir.h metal_ir_source_relation: the original relation, each original vertex's operation
        # IDs (operation_first[V + 1], operations), each original view's typed ValueID, each GIVEN cause's operand; the
        # arrays borrowed for this ConstructedMachine's lifetime (NULL release and owner)
        flat = [op for ops in vertices for op in ops]
        binding = ((C.c_size_t * (len(vertices) + 1))(*itertools.accumulate(map(len, vertices), initial=0)),
                   (C.c_uint32 * max(len(flat), 1))(*flat),
                   (C.c_uint32 * max(len(accesses), 1))(*[NONE if access is None or access[0] == "literal"
                                                          else access[1] for access in accesses]),
                   (SourceOperand * max(len(given), 1))(*map(operand, given)))
        error = C.create_string_buffer(4096)
        status = compiler.metal_ir_source_relation(builder.p, source.relation.pointer, *binding[:3], len(given),
                                                   binding[3], RELEASE(), None, error, len(error))
        if status:
            raise OSError(status, "the source relation: %s" % error.value.decode())
    except BaseException:
        compiler.metal_ir_program_free(builder.p)
        raise
    decomposed = {}
    for generated, original in module.meta["decomposed"].items():
        decomposed.setdefault(original, []).extend(emitted.get(generated, ()))
    return ConstructedMachine(builder.p, compiler, dict(targets), tuple(inputs), tuple(outputs), spec, state, received,
                              sent, ends, {task: tuple(ops) for task, ops in operations.items()}, tuple(advanced),
                              made, {k: tuple(ops) for k, ops in writes.items()},
                              {k: tuple(ops) for k, ops in stores.items()}, source, vertices, accesses, given, writers,
                              builder.base, binding, {k: tuple(sorted(ops)) for k, ops in decomposed.items()},
                              initial_words=tuple(word & 0xFFFFFFFF for word in generator.state.tolist()) if state
                              else (), input_parts=input_parts)

# design/compilation-composition-change-request.md items 3, 4, 6 and 7; design/aot-compilation-change-request.md A5,
# A7 and A8; design/plan-to-bridge-change-request.md R2 and R8; metal-microbench metal_ir.h metal_ir_value_external,
# metal_ir_lower, metal_ir_specialize, metal_ir_storages, metal_ir_finalize, metal_ir_loaded, metal_ir_archive_load;
# metal_recording.h MetalRecorderActors, MetalRecorderPrepare, MetalRecorderReplay, MetalRecorderProgram,
# MetalProgramSave, MetalProgramLoad, MetalProgramSignature and MetalProgramRunArguments; rdma/mesh-collective.h
# mesh_plan, mesh_plan_local; rdma/mesh-metal.h mesh_metal_range_declare and the prepared transport
import ctypes as C
import dataclasses
import errno
import mmap
import pathlib
import sys
import weakref

import torch
from torch.export.graph_signature import OutputKind
from torch.utils import _pytree as pytree

from . import metal
from .metal import METAL_EXTERNAL_GPU, Argument

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

ROOM = 4096


# one executed unit of Bound.observe: its kind, its call (the Target.rows index), native site, member dispatch IDs,
# source operation identities (lower_metal.construct's global task index k), executed occurrences and seconds
@dataclasses.dataclass(frozen=True)
class Observation:
    kind: int
    call: int
    site: int
    dispatches: tuple
    identities: tuple
    executions: int
    seconds: float


# compiler/mesh_compiler/metal.py: the live native owner and its three bound libraries (libmetal_ir, the recorder and
# its device, the transport)
def libraries():
    return metal, metal._compiler(), metal._recorder(), metal._transport()


# One preparation's native resources, released in reverse order when its owner is collected or its preparation fails
class Owned:
    def __init__(self):
        self.releases = []
        self.finalizer = weakref.finalize(self, Owned.released, self.releases)

    def __call__(self, release):
        self.releases.append(release)

    def __enter__(self):
        return self

    def __exit__(self, kind, *_):
        if kind is not None:
            self.finalizer()

    @staticmethod
    def released(releases):
        while releases:
            releases.pop()()


def made(result, error, what, status=errno.EINVAL):
    if not result:
        raise OSError(status, f"{what}: {error.value.decode()}")
    return result


def checked(status, error, what):
    if status:
        raise OSError(status, f"{what}: {error.value.decode()}")


# metal-microbench metal_ir.h metal_ir_storages, metal_ir_value_info and metal_ir_step_state: the
# lowered program's actual Storages as plan values, one per ValueID (run 1 a root its
# own storage holds: neither an explicit root nor one with data nor the step), and its step word (8 bytes, executions,
# lag 1) where it has a step value (`word` METAL_IR_STEP, found in the same pass: metal_ir_step_value would construct
# one after lowering) and the host writes it; the quantum the largest placed alignment
def storages(program, explicit, libs):
    M, compiler, _, _ = libs
    storage = C.POINTER(M.Storage)()
    count = compiler.metal_ir_storages(program, C.byref(storage))
    infos = [M.ValueInfo() for _ in range(count)]
    for v, info in enumerate(infos):
        compiler.metal_ir_value_info(program, v, C.byref(info))
    step = next((v for v, info in enumerate(infos) if info.word == M.METAL_IR_STEP), None)
    stepped = step is not None and compiler.metal_ir_step_state(program) == M.NONE
    values, quantum = (mesh.Value * (count + stepped))(), 1
    for v, info in enumerate(infos):
        stored = storage[v]
        run = bool(stored.bytes and info.storage == v and v not in explicit and v != step and not info.data)
        values[v] = mesh.Value(section=mesh.Section(bytes=stored.bytes), state=info.zeroed, begin=stored.begin,
                               end=stored.end, run=int(run),
                               kind=mesh.INVARIANT if info.role in (M.CONSTANT, M.STATE) else mesh.EXECUTION)
        quantum = max(quantum, stored.alignment) if run else quantum
    if stepped:
        values[count] = mesh.Value(section=mesh.Section(bytes=8), kind=mesh.EXECUTION, lag=1, run=1)
        quantum = max(quantum, 8)
    return values, count, stepped, quantum, storage


# metal_recording.h MetalRecorderActors, MetalRecorderActorsFree: the lowered program's
# actual Actors (its parts' host and work times and their edges), C-owned until freed
def actors(program, libs):
    M, _, (lib, _), _ = libs
    machines, made_ = M._machines(), M._Actors()
    gpu, cpu = (C.byref(machines[M.GPU]), C.byref(machines[M.CPU])) if machines else (None, None)
    status = lib.MetalRecorderActors(program, gpu, cpu, C.byref(made_))
    if status not in (0, errno.ENODATA):
        mesh.check(status)
    return made_


# Copy offsets in place: value `index`'s run of `offset` (at address `base`), from first[index] to first[index + 1]
def viewed(first, base, index):
    return C.cast(base + C.sizeof(mesh.Q) * first[index], C.POINTER(mesh.Q)), first[index + 1] - first[index]


# One preparation of a local program on its joint result (compiler/mesh_compiler/schedule.py solve_program: its
# mesh.Local over every function's values and the roots): one arena allocation of the result's bytes
# (compiler/mesh_compiler/metal.py _NEW_BUFFER, the shared Metal allocation) and the result's own planned record; the
# copies read in place. A local program has no transport, context or status
class Arena:
    context = None

    def __init__(self, target, owned, M, gpu):
        joint, error = target.solution, C.create_string_buffer(ROOM)
        self.target, self.function, self.planned = target, joint.functions, joint.planned
        self.base = C.cast(joint.offsets.items, C.c_void_p).value
        buffer = made(M._NEW_BUFFER(gpu, M._SEL(b"newBufferWithLength:options:"), max(joint.local.bytes, 1), M.SHARED),
                      error, "the program's arena", errno.ENOMEM)
        owned(lambda: M.PUT(buffer, M.SELECTOR[b"release"]))
        self.buffer = buffer

    def bound(self, owned, M):
        return self.buffer

    # value `index` of a solved first view (schedule.Solved function_value_first[f] or root_first): its copies' offsets
    def span(self, first, index):
        return viewed(first, self.base, index)

    def placements(self, r):
        return self.target.placements(self, r)

    def stepped(self, r):
        return self.target.stepped(self, r)


# rdma/mesh-collective.h mesh_plan; rdma/mesh-dataflow.h mesh_view_values: a program with declared transfers at
# compilation, on its joint result under the Planning context (its Planned, context and mapped view with that view's
# first/offset arrays, as schedule.solved returns them), the copies read in place; the run-time owner is
# bridge.Published
class Mapped:
    def __init__(self, target):
        joint = target.solution
        self.target, self.function, self.planned, self.context = target, joint.functions, joint.planned, joint.context
        self.base, view = C.cast(joint.offsets.items, C.c_void_p).value, joint.view
        self.arena = torch.frombuffer((C.c_byte * view.bytes).from_address(view.address), dtype=torch.uint8) if (
            view.bytes) else torch.empty(0, dtype=torch.uint8)

    def bound(self, owned, M):
        return planned_buffer(self, owned, M)

    def span(self, first, index):
        return viewed(first, self.base, index)

    def placements(self, r):
        return self.target.placements(self, r)

    def stepped(self, r):
        return self.target.stepped(self, r)


# compiler/mesh_compiler/metal.py _placed_buffer: a plan owner's mapped arena as the one Metal buffer every placement
# takes
def planned_buffer(owner, owned, M):
    buffer = M._placed_buffer(mesh.View(owner.arena.data_ptr(), owner.arena.numel()))
    owned(lambda: M.PUT(buffer, M.SELECTOR[b"release"]))
    return buffer


# compiler/mesh_compiler/lower_metal.py ConstructedMachine (frontend-owned): one compiled function's constructed
# program and typed roots, `machine` None for a function with no body on this node. `key` its captured function key
# (compilation-only; its ordinal is its identity), `declared` its Declarations.wires() (channel -> declared transfer,
# logical wire value, its logical bytes, declared sent view, declared landed view), `marks` its wire roots' declared
# records: `received` (ValueID, function, value, declared mesh_metal_range), `sent` (ValueID, declared
# mesh_metal_release), `sending` {transfer: ValueID}, `transfers` receives then sends, `placed` (ValueID, function,
# value), one per wire (none without a body). `slots` its callable interface roots in their stable order (each marked
# metal_ir_value_external with its slot before the one lowering: an input or state root, its generator state, each
# further output root but a CONSTANT); `placeholders` and `results`, per placeholder i and output spec o, (slot, root,
# view, the view's byte offset in its root), slot None where the root is no interface slot
@dataclasses.dataclass
class Body:
    key: object
    machine: object
    declared: dict
    inputs: tuple
    outputs: tuple
    state: tuple
    operations: dict
    identity: int
    targets: dict
    marks: dict = dataclasses.field(default_factory=lambda: {"received": [], "sent": [], "sending": {}, "transfers": [],
                                                             "placed": []})
    slots: tuple = ()
    placeholders: tuple = ()
    results: tuple = ()


# placement.status.md section 3 and joint-artifact.status.md section 00, schedule.solve_program's `bindings`: call j's
# function and, per placeholder i and output spec o, the body's interface slot, root and view ValueIDs, the view's byte
# offset in its root, and the original Call.inputs[i] / Call.outputs[o] Node or literal it binds
@dataclasses.dataclass(frozen=True)
class Slot:
    slot: object
    root: object
    view: object
    offset: int
    reference: object


@dataclasses.dataclass(frozen=True)
class Binding:
    call: int
    function: int
    inputs: tuple
    outputs: tuple


# design/aot-compilation-change-request.md A10 and appendix item 6; metal_recording.h struct MetalCall;
# rdma/mesh-collective.h struct mesh_planned_call: one call of a compiled function, one prepared program row of the
# native program: `call` its index in the captured calls, `function` its body, `ordinal` its ordinal among that
# function's `every` calls in the block, `planned` its Planned.call index
@dataclasses.dataclass
class Row:
    call: int
    function: int
    ordinal: int = 0
    every: int = 1
    planned: int = 0


# metal-microbench metal_ir.h metal_ir_value_external; metal_recording.h struct MetalArgument: one actual outer backing
# at one root offset, the native program's callable operand of its position: `source` ("input", the outer placeholder's
# position), ("generator", its body) or ("result", its joint root s, its bytes, its device); `offset` its root byte 0's
# offset (the caller's byte start less it, an outer input's; less its negation, a result's allocation start);
# `occurrences` each (row, root ValueID) that is this backing
@dataclasses.dataclass
class Operand:
    source: tuple
    offset: int
    occurrences: tuple


# The completed machine program: its compiled functions (`bodies`, in the call analysis's function order), its calls
# (`rows`, one MetalRecorderProgram row each, in its call order), its callable operands (`operands`, in ordinal order),
# its flat outer outputs (`outputs`: (row, the frontend's Output) of a call's result, (None, Output) of an outer
# placeholder's position or a literal, in exported.outputs order), each row's planned inter-call roots (`joined`: (root
# ValueID, joint root s, byte delta, the call's every)), each body's ValueID count and whether its step follows
# (`counts`, `steps`), its bodies' generators' initial words (`generators`: (body, ConstructedMachine.initial_words)),
# each row's CONSTANT roots' outer constants (`constants`: (row, root ValueID, the outer placeholder's name)), each
# outer output a remote call produces (`ports`: (output position, call, output spec), its leaf None: the output owner's
# transfer port, frontend974's joint output port operand, binds it); `node`
# the program's machine (its map index, the plan's node map's with a plan, the source machine's without); `solution` the
# joint result (compiler/mesh_compiler/schedule.py solve_program), `plan` a transported machine's solved Plan,
# `artifact` the one MetalProgramSave of every row, `signature` its MetalProgramSignatureNode record
@dataclasses.dataclass
class Target:
    bodies: tuple
    rows: tuple
    spec: object
    outputs: tuple = ()
    operands: tuple = ()
    joined: tuple = ()
    counts: tuple = ()
    steps: tuple = ()
    generators: tuple = ()
    constants: tuple = ()
    ports: tuple = ()
    node: int = None
    solution: object = None
    plan: object = None
    artifact: bytes = None
    signature: bytes = None

    # body k's explicit roots: its interface slots and its input, state, received and sent roots
    def explicit(self, k):
        body = self.bodies[k]
        built = body.machine
        return {root for root in (*body.slots, *(r.root for r in (*body.inputs, *body.state, *built.received.values(),
                                                                  *built.sent.values()))) if root is not None}

    # row r's own storages and inter-call roots on the owner (schedule.Solved): each run value of its function at its
    # full copies (function_value_first[f]); each planned inter-call root at the joint root's copies (root_first[s]),
    # with that root's shares, projected through the call's generation: execution e = t every + ordinal reads root copy
    # t = e // every, so entry i of its every x copies table is root copy i // every, moved by the byte delta
    def placements(self, owner, r):
        f, joint = self.rows[r].function, self.solution
        value, first = owner.function[f].value, joint.function_value_first[f]
        for v in range(self.counts[f]):
            if value[v].run:
                yield (v, *owner.span(first, v), value[v].shares)
        for root, s, delta, every in self.joined[r]:
            offsets, count = owner.span(joint.root_first, s)
            shares = joint.roots[s].shares
            if every == 1 and not delta:
                yield (root, offsets, count, shares)
            else:
                yield (root, (mesh.Q * (every * count))(*[offsets[i // every] + delta for i in range(every * count)]),
                       every * count, shares)

    def stepped(self, owner, r):
        f = self.rows[r].function
        if not self.steps[f]:
            return None
        return (*owner.span(self.solution.function_value_first[f], self.counts[f]),
                owner.function[f].value[self.counts[f]].shares)

    def placed(self, owned, libs, gpu, owner, programs, finalize):
        M, compiler, (lib, _), transport = libs
        error, planned = C.create_string_buffer(ROOM), owner.planned
        recorder = made(lib.MetalRecorderOpen(gpu, error, ROOM), error, "recorder", errno.ENODEV)
        owned(lambda: M.PUT(recorder, M.SELECTOR[b"release"]))
        buffer, calls = owner.bound(owned, M), []
        for r, (row, program) in enumerate(zip(self.rows, programs)):
            if program is None:
                calls.append(M.Call(None, row.planned + 1, None, 0))
                continue
            marks, wires, call = self.bodies[row.function].marks, None, row.planned
            if marks["transfers"]:
                context, wires, word = mesh.Context.from_address(owner.context), M.Transport(), mesh.Q()
                mesh.check(transport.mesh_metal_transport_create(C.byref(context), C.byref(planned), gpu,
                                                                 C.byref(wires)))
                owned(lambda wires=wires: transport.mesh_metal_transport_destroy(C.byref(wires)))
                for value, function, index, declared in marks["received"]:
                    # rdma/mesh-metal.h mesh_metal_range_declare: this call's own transfer lands for itself (owner
                    # the call, generation {1, 1, 0}); range_prepare reads planned->call[owner]
                    landed = M.Range.from_buffer_copy(declared)
                    landed.owner, landed.generation = call, M.Generated(1, 1, 0)
                    field = (mesh.U * max(landed.partitions, 1))()
                    mesh.check(transport.mesh_metal_range_prepare(
                        C.byref(context), C.byref(wires), C.byref(planned),
                        owner.function[function].value[index].section, field, landed.partitions, C.byref(landed)))
                    compiler.metal_ir_value_received(program, value, C.byref(landed))
                for value, declared in marks["sent"]:
                    release = M.Release.from_buffer_copy(declared)
                    mesh.check(transport.mesh_metal_release_prepare(C.byref(context), C.byref(wires),
                                                                    C.byref(planned), call, C.byref(release)))
                    compiler.metal_ir_value_sent(program, value, C.byref(release))
                for transfer in marks["transfers"]:
                    sending = M.Sending()
                    mesh.check(transport.mesh_metal_sending_prepare(C.byref(context), C.byref(wires),
                                                                    C.byref(planned), call, transfer,
                                                                    C.byref(sending)))
                    if sending.sent.sends:
                        compiler.metal_ir_value_sending(program, marks["sending"][transfer], C.byref(sending))
                mesh.check(transport.mesh_metal_status_prepare(C.byref(context), C.byref(wires),
                                                               planned.call[call].plan, C.byref(word)))
                compiler.metal_ir_transport(program, word.value)
            emitted = compiler.metal_ir_finalize if finalize else compiler.metal_ir_loaded
            checked(emitted(program, error, ROOM), error, "emission" if finalize else "relocation")
            ready = made(lib.MetalRecorderPrepare(recorder, program, error, ROOM), error, "preparation")
            owned(lambda ready=ready: M.PUT(ready, M.SELECTOR[b"release"]))
            for value, function, index in marks["placed"]:
                compiler.metal_ir_value_placed(program, value, buffer,
                                               *owner.span(self.solution.function_value_first[function], index), 0,
                                               owner.function[function].value[index].shares)
            for value, offsets, count, shares in owner.placements(r):
                compiler.metal_ir_value_placed(program, value, buffer, offsets, count, 0, shares)
            step = owner.stepped(r)
            if step is not None:
                compiler.metal_ir_step_placed(program, buffer, *step)
            # metal_ir.h metal_ir_value_placed with no buffer: each callable operand's root one logical copy, shared
            # by every execution (a state or mutated input is the caller's one storage), its address the invocation's
            for root in (root for operand in self.operands for at, root in operand.occurrences if at == r):
                compiler.metal_ir_value_placed(program, root, None, None, 1, 0, 0)
            if wires is not None:
                M.STATE_BUFFER(recorder, M._SEL(b"state:buffer:status:"), ready, wires.state, word.value)
            checked(compiler.metal_ir_place(program, error, ROOM), error, "placed accesses")
            replay = M.Replay()
            checked(lib.MetalRecorderReplay(recorder, ready, C.byref(replay), error, ROOM), error, "replay")
            owned(lambda replay=replay: lib.MetalReplayDestroy(C.byref(replay)))
            calls.append(M.Call(C.pointer(replay), call + 1, None, 0))
        return recorder, (M.Call * len(calls))(*calls)

    # body k's callable interface before its lowering: each distinct input or state root (its first placeholder's), its
    # generator state, each further output root (any role but CONSTANT: the joint binding supplies every function's
    # output roots, an internally consumed one included), in that order; and each
    # placeholder's and output's (slot, root, view, the view's byte offset in its root from metal_ir_value_layout)
    def interface(self, k, libs):
        M, compiler, _, _ = libs
        body = self.bodies[k]
        program, widths, slots = body.machine.program, {code: dtype.itemsize for dtype, code in M.SCALAR.items()}, {}

        def placed_at(view):
            info, at = M.ValueInfo(), C.c_int64()
            compiler.metal_ir_value_info(program, view, C.byref(info))
            compiler.metal_ir_value_layout(program, view, None, 0, C.byref(at))
            return at.value * widths[info.scalar]
        for root in body.inputs:
            if root.root is not None and root.role in (M.INPUT, M.STATE):
                slots.setdefault(root.root, len(slots))
        for root in body.state:
            slots.setdefault(root.root, len(slots))
        for leaf in body.outputs:
            info = M.ValueInfo()
            if leaf.root is not None and leaf.root not in slots:
                compiler.metal_ir_value_info(program, leaf.root, C.byref(info))
                if info.role != M.CONSTANT:
                    slots.setdefault(leaf.root, len(slots))
        body.slots = tuple(slots)
        body.placeholders, body.results = (tuple((slots.get(r.root), r.root, r.value, placed_at(r.value))
                                                 if r.root is not None else (None, None, None, 0) for r in records)
                                           for records in (body.inputs, body.outputs))

    # rdma/mesh-metal.h mesh_metal_range_declare; metal-microbench metal_ir.h metal_ir_value_receives,
    # metal_ir_value_sends, metal_ir_value_planned, metal_ir_value_external and metal_ir_lower;
    # design/compilation-composition-change-request.md, 6's stage A: before the plan, from the declarations alone, each
    # body's interface, its declared transfers in transfer order (`context` the compile owner's, read for its page
    # geometry), each wire's plan value, its slots marked external, and its one lowering, whose actual Actors and
    # Storages the one plan solves. `calls` the machine's calls as mesh_plan takes them; a body's own transfers land
    # for its first row's call
    def lowered(self, context=None, calls=(0,)):
        libs = libraries()
        M, compiler, (lib, device), transport = libs
        error, listed = C.create_string_buffer(ROOM), (mesh.U * len(calls))(*calls)
        for k, body in ((k, body) for k, body in enumerate(self.bodies) if body.machine is not None):
            self.interface(k, libs)
            built, declared = body.machine, body.declared
            program, row = built.program, next(row.planned for row in self.rows if row.function == k)
            receives = sorted((declared[channel][0], channel) for channel in built.received if channel in declared)
            sends = sorted((declared[channel][0], channel) for channel in built.sent if channel in declared)
            marks = {"received": [], "sent": [], "sending": {}, "transfers": [], "placed": []}
            for transfer, channel in receives:
                root, (_, _, size, *_) = built.received[channel].root, declared[channel]
                landed = M.Range()
                mesh.check(transport.mesh_metal_range_declare(mesh.Context.from_address(context), listed, len(calls),
                                                              row, row, transfer, size, 0, C.byref(landed)))
                compiler.metal_ir_value_receives(program, root, C.byref(landed))
                marks["received"].append((root, k, root, bytes(landed)))
            for transfer, channel in sends:
                root, release = built.sent[channel].root, M.Release(transfer=transfer)
                compiler.metal_ir_value_sends(program, root, C.byref(release))
                marks["sent"].append((root, bytes(release)))
            for transfer, channel in receives + sends:
                marks["transfers"].append(transfer)
                marks["sending"][transfer] = built.wires[declared[channel][1]].root
            for value, wire in sorted(built.wires.items()):
                compiler.metal_ir_value_planned(program, wire.root, k, wire.root, 0)
                marks["placed"].append((wire.root, k, wire.root))
            for slot, root in enumerate(body.slots):
                mesh.check(compiler.metal_ir_value_external(program, root, slot))
            gpu = device()
            try:
                checked(compiler.metal_ir_lower(program, gpu, error, ROOM), error, "lowering")
            finally:
                M.PUT(gpu, M.SELECTOR[b"release"])
            body.marks = marks
        return self

    # placement.status.md section 3: each lowered body's mesh.Function (its actual Storages, the interface and explicit
    # roots run 0, its step appended; its retained MetalRecorderActors parts and relation, freed with the target), in
    # body order, a function with no body on this node zero-local (no values, parts or edges), and the largest placed
    # alignment
    def functions(self, libs):
        lib, held, quantum, counts, steps = libs[2][0], [], 1, [], []
        functions = (mesh.Function * len(self.bodies))()
        for k, body in enumerate(self.bodies):
            if body.machine is None:
                counts, steps = counts + [0], steps + [False]
                continue
            values, count, stepped, aligned, _ = storages(body.machine.program, self.explicit(k), libs)
            solved = actors(body.machine.program, libs)
            weakref.finalize(self, lib.MetalRecorderActorsFree, C.byref(solved))
            functions[k] = mesh.Function(value=C.cast(values, C.POINTER(mesh.Value)), values=len(values),
                                         host=solved.host, work=solved.work, edge=solved.edge, parts=solved.parts,
                                         edges=solved.edges, relation=solved.relation)
            held += [values, solved]
            quantum, counts, steps = max(quantum, aligned), counts + [count], steps + [stepped]
        self.counts, self.steps, self.held = tuple(counts), tuple(steps), (functions, held)
        return functions, quantum

    # schedule.solve_program's bindings: per call, in the analysis's call order, its body's slots with the original
    # Call references (placeholder order, output spec order)
    def bindings(self, captured, analysis):
        def slots(entries, given):
            return tuple(Slot(*entry, reference) for entry, reference in zip(entries, given, strict=True)) \
                if entries or not given else tuple(Slot(None, None, None, 0, reference) for reference in given)
        return tuple(Binding(j, analysis.calls[j], slots(body.placeholders, call.inputs),
                             slots(body.results, call.outputs))
                     for j, (call, body) in enumerate((captured.calls[node], self.bodies[analysis.calls[j]])
                                                      for j, node in enumerate(analysis.nodes)))

    # From the joint result (schedule.Solved): each row's planned call (ordinal, every); per resolved binding (call,
    # root ValueID, storage s, byte delta, its original reference, every, ordinal) a run root (roots[s].run) placed at
    # s's copies (`joined`); otherwise its storage's root_sources[s] record: ("input", outer flat placeholder position),
    # an operand whose root byte 0 lies delta past that placeholder's byte offset in s, or ("result", s, bytes,
    # device), an operand of s's one allocation; each body's generator state a ("generator", body) operand; each
    # CONSTANT root its outer constant's name (`constants`, from the call's original input); and the outer outputs
    def resolved(self, captured, analysis, joint):
        from .lower_metal import Output
        from .operations import _metadata
        planned = joint.planned
        self.rows = tuple(Row(j, analysis.calls[j], planned.call[j].ordinal, planned.call[j].every, j)
                          for j in range(len(analysis.nodes)))
        operands, keys, joined, constants = [], {}, [[] for _ in self.rows], []

        def bound(key, source, offset, at):
            if key not in keys:
                keys[key] = len(operands)
                operands.append(Operand(source, offset, ()))
            a = keys[key]
            operands[a] = dataclasses.replace(operands[a], occurrences=operands[a].occurrences + (at,))
        outer = [node for node in captured.exported.graph_module.graph.nodes if node.op == "placeholder"]
        for call, root, s, delta, _, every, _ in joint.bindings:
            source = joint.root_sources[s]
            if joint.roots[s].run:
                joined[call].append((root, s, delta, every))
            elif source[0] == "input":
                value = outer[source[1]].meta["val"]
                bound(("input", s, delta), source, value.storage_offset() * value.element_size() - delta, (call, root))
            else:
                bound(("result", s, delta), source, -delta, (call, root))
        for j, node in enumerate(analysis.nodes):
            body = self.bodies[analysis.calls[j]]
            for root in body.state:
                bound(("generator", analysis.calls[j]), ("generator", analysis.calls[j]), 0, (j, root.root))
            for root, reference in zip(body.inputs, captured.calls[node].inputs):
                if root.root is not None and root.role == metal.CONSTANT:
                    constants.append((j, root.root, reference.target))
        produced = {leaf: (j, o) for j, node in enumerate(analysis.nodes)
                    for o, leaf in enumerate(captured.calls[node].outputs) if isinstance(leaf, torch.fx.Node)}
        outputs, ports = [], []
        for leaf, (kind, _) in zip(captured.exported.graph_module.graph.output_node().args[0],
                                   captured.exported.outputs, strict=True):
            if leaf in produced:
                j, o = produced[leaf]
                body = self.bodies[self.rows[j].function]
                if body.machine is not None:
                    outputs.append((j, body.outputs[o]))
                else:
                    outputs.append((j, None))
                    ports.append((len(outputs) - 1, j, o))
            elif isinstance(leaf, torch.fx.Node):
                outputs.append((None, Output(outer.index(leaf), None, None, _metadata(leaf.meta["val"]), kind, None)))
            else:
                outputs.append((None, Output(None, None, leaf, None, kind, None)))
        self.operands, self.joined, self.outputs = tuple(operands), tuple(map(tuple, joined)), tuple(outputs)
        self.constants, self.ports = tuple(constants), tuple(ports)

    # design/compilation-composition-change-request.md, 6's stage D, and design/aot-compilation-change-request.md
    # appendix item 6: after
    # the one plan, each row's specialization of its body's lowered source prepared on the compile owner's result and
    # finalized, all rows made one native program, signed over its warm-up and one period as the native program chose
    # them (metal_recording.h MetalProgramPhases), and saved once
    def compile(self, owner):
        libs = libraries()
        M, compiler, (lib, device), _ = libs
        error = C.create_string_buffer(ROOM)
        with Owned() as owned:
            gpu = device()
            owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
            programs = []
            for row in self.rows:
                machine = self.bodies[row.function].machine
                if machine is None:
                    programs.append(None)
                    continue
                prepared = made(compiler.metal_ir_specialize(machine.program, error, ROOM), error, "specialization")
                owned(lambda prepared=prepared: compiler.metal_ir_program_free(prepared))
                r = len(programs)
                for a, operand in enumerate(self.operands):
                    for at, root in operand.occurrences:
                        if at == r:
                            mesh.check(compiler.metal_ir_value_external(prepared, root, a))
                programs.append(prepared)
            recorder, calls = self.placed(owned, libs, gpu, owner, programs, True)
            native = made(lib.MetalRecorderProgram(recorder, calls, len(calls), C.byref(owner.planned), error, ROOM),
                          error, "the program's calls")
            owned(lambda: M.PUT(native, M.SELECTOR[b"release"]))
            warm, period = mesh.Q(), mesh.Q()
            lib.MetalProgramPhases(native, C.byref(warm), C.byref(period))
            signature, executions = mesh_graph.Signature(), mesh.Q(warm.value + period.value + 1)
            checked(lib.MetalProgramSignatureNode(native, self.node, 1, C.byref(executions), C.byref(signature), error,
                                                  ROOM), error, "signature")
            try:
                self.signature = mesh_graph.record(signature)
            finally:
                mesh_graph.free(signature)
            self.artifact = M._saved_bytes(lib.MetalProgramSave, native, None, 0)
            owned.finalizer()
        return self

    # metal-microbench metal_ir.h metal_ir_archive_load (compiler/mesh_compiler/metal.py _archive_loaded: the archive
    # borrows and, loaded, owns the artifact's own bytes), metal_ir_archive_program and metal_ir_loaded;
    # metal_recording.h MetalProgramLoad (given the archive handle, which it retains): the completed artifact, every
    # row's program in row order, relocated onto its run-time owner (bridge.Published, or None: a new arena of the
    # retained local solution), with no lowering, solve, finalization or compilation; constants: {placeholder name:
    # tensor} for the bodies' CONSTANT roots, attached as compiler/mesh_compiler/metal.py _Builder.attach attached them
    # and held as those roots' handles, by (row, root)
    def prepare(self, published, constants):
        libs = libraries()
        M, compiler, (lib, device), _ = libs
        error = C.create_string_buffer(ROOM)
        owned = Owned()
        with owned:
            gpu = device()
            owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
            archive = M._archive_loaded(compiler, self.artifact, gpu, error)
            owned(lambda: compiler.metal_ir_archive_free(archive))
            programs = [compiler.metal_ir_archive_program(archive, r) or None for r in range(len(self.rows))]
            for program in (program for program in programs if program is not None):
                owned(lambda program=program: compiler.metal_ir_program_free(program))
            fixed = {}
            for r, root, name in self.constants:
                program = programs[r]
                info = M.ValueInfo()
                compiler.metal_ir_value_info(program, root, C.byref(info))
                fixed.setdefault((r, root), constants[name])
                if not info.data:
                    storage = constants[name].untyped_storage()
                    buffer = storage.data_ptr() if storage.device.type == "mps" else None
                    data = M.OBJECT(buffer, M._SEL(b"contents")) if buffer else storage.data_ptr()
                    M._lent(constants[name], lambda release, owner: compiler.metal_ir_value_data(
                        program, root, data, storage.nbytes(), buffer, 0, owner, release), lambda _: True)
            owner = published if published is not None else Arena(self, owned, M, gpu)
            recorder, calls = self.placed(owned, libs, gpu, owner, programs, False)
            native = made(lib.MetalProgramLoad(recorder, calls, len(calls), C.byref(owner.planned), archive, error,
                                               ROOM), error, "the loaded program's calls")
            owned(lambda: M.PUT(native, M.SELECTOR[b"release"]))
            ended = made(M.OBJECT(gpu, M._SEL(b"newSharedEvent")), error, "the program's event", errno.ENOMEM)
            owned(lambda: M.PUT(ended, M.SELECTOR[b"release"]))
            owned(lambda: lib.MetalProgramEnd(native, ended, error, ROOM))
            generators = {}
            for k, initial in self.generators:
                words = (mesh.U * len(initial))(*initial)
                buffer = made(M._NEW_BUFFER(gpu, M._SEL(b"newBufferWithLength:options:"), C.sizeof(words), M.SHARED),
                              error, "the generator state", errno.ENOMEM)
                owned(lambda buffer=buffer: M.PUT(buffer, M.SELECTOR[b"release"]))
                C.memmove(M.OBJECT(buffer, M._SEL(b"contents")), words, C.sizeof(words))
                generators[k] = buffer
            count = mesh.U()
            uses = lib.MetalProgramExternals(native, C.byref(count))
        return Bound(self, owner, native, ended, generators, fixed, owned, libs, (uses, count.value))

    # The typed recipe: explicit records for the bodies (by function ordinal: a captured function's key, which may be a
    # callable, is compilation-only), rows, operands, inter-call roots, constants and outer outputs (no tuple-keyed
    # JSON)
    def recipe(self):
        return {"bodies": [{"inputs": [list(root) for root in body.inputs],
                            "outputs": [leaf_saved(leaf) for leaf in body.outputs],
                            "state": [list(root) for root in body.state],
                            "operations": [[list(task), list(ops)] for task, ops in body.operations.items()],
                            "identity": body.identity,
                            "targets": [[name, kind, str(device)] for name, (kind, device) in body.targets.items()],
                            "marks": {"received": [(v, f, i, d.hex()) for v, f, i, d in body.marks["received"]],
                                      "sent": [(v, d.hex()) for v, d in body.marks["sent"]],
                                      "sending": [[t, v] for t, v in body.marks["sending"].items()],
                                      "transfers": body.marks["transfers"], "placed": body.marks["placed"]},
                            "slots": list(body.slots), "placeholders": [list(e) for e in body.placeholders],
                            "results": [list(e) for e in body.results]} for body in self.bodies],
                "rows": [dataclasses.asdict(row) for row in self.rows],
                "operands": [{"source": list(operand.source), "offset": operand.offset,
                              "occurrences": [list(at) for at in operand.occurrences]} for operand in self.operands],
                "outputs": [[r, leaf and leaf_saved(leaf)] for r, leaf in self.outputs],
                "ports": [list(port) for port in self.ports],
                "joined": [[list(entry) for entry in row] for row in self.joined], "counts": list(self.counts),
                "steps": list(self.steps), "generators": [[k, list(words)] for k, words in self.generators],
                "constants": [list(entry) for entry in self.constants],
                "spec": pytree.treespec_dumps(self.spec), "node": self.node}

    # the artifact, the joint result as its own owner saves it (compiler/mesh_compiler/schedule.py), and the recipe
    def saved(self, directory, key):
        directory, name = pathlib.Path(directory), f"{key}.native"
        (directory / name).write_bytes(self.artifact)
        (directory / f"{key}.joint").write_bytes(self.solution.saved())
        return {"artifact": name, "joint": f"{key}.joint", "signature": self.signature.hex(), "recipe": self.recipe()}


# compiler/mesh_compiler/lower_metal.py Output (frontend-owned) as a saved record: its root, value and literal, its
# metadata (shape, stride, storage offset, dtype and device names, requires_grad), its OutputKind name and the
# placeholder it writes
def leaf_saved(leaf):
    return [*leaf[:3], leaf.metadata and [*leaf.metadata[:3], str(leaf.metadata[3]).removeprefix("torch."),
                                          str(leaf.metadata[4]), leaf.metadata[5]], leaf.kind.name, leaf.written]


def leaf_restored(saved):
    from .lower_metal import Output
    return Output(*saved[:3], saved[3] and (tuple(saved[3][0]), tuple(saved[3][1]), saved[3][2],
                                            getattr(torch, saved[3][3]), torch.device(saved[3][4]), saved[3][5]),
                  OutputKind[saved[4]], saved[5])


# A prepared native program on its owner. A call binds its operands at the native program's fixed occurrences (each
# input or state backing the caller's storage; each fresh output root a result allocation of this call, which the
# program writes directly), runs one execution of every row, waits for its end, and returns its outputs as views of
# those backings or of a CONSTANT root's own handle (`fixed`, {(row, root): its tensor}). `uses` the program's
# borrowed MetalProgramExternals view (pointer, count), kept with the program that owns it; `gpu` the ordinals of the
# operands it uses on the GPU, read once; `argued` {(row, root): operand ordinal}
class Bound:
    def __init__(self, target, owner, program, ended, generator, fixed, owned, libs, uses):
        self.target, self.owner, self.program, self.ended, self.generator = target, owner, program, ended, generator
        self.fixed, self.owned, self.libs, self.executions, self.uses = fixed, owned, libs, 0, uses
        self.argued = {at: a for a, operand in enumerate(target.operands) for at in operand.occurrences}
        self.gpu = frozenset(use.argument for use in uses[0][:uses[1]] if use.uses & METAL_EXTERNAL_GPU)

    # metal_recording.h struct MetalArgument; MTLDevice.h newBufferWithBytesNoCopy:length:options:deallocator: (an
    # existing address span, page aligned) through compiler/mesh_compiler/metal.py _placed_buffer: root byte 0 of the
    # caller's view (its byte start less the compiled view's offset in its root). An MPS storage gives its own
    # MTLBuffer. A CPU storage gives its host base, and where the program uses the operand on the GPU, a shared buffer
    # borrowing that storage's pages (the system page size) and owning a reference to the storage until its deallocator
    # runs, one per storage in this call (`borrowed`, {storage address: (first page, buffer)}), released after the call
    def operand(self, tensor, offset, gpu, borrowed):
        M, storage = self.libs[0], tensor.untyped_storage()
        if tensor.device.type == "mps":
            return Argument(storage.data_ptr(), (tensor.storage_offset() * tensor.element_size() - offset) % (1 << 64),
                            None)
        base = tensor.data_ptr() - offset
        if not gpu:
            return Argument(None, 0, base)
        address = storage.data_ptr()
        if address not in borrowed:
            first = address - address % mmap.PAGESIZE
            last = -(-(address + storage.nbytes()) // mmap.PAGESIZE) * mmap.PAGESIZE
            borrowed[address] = first, M._placed_buffer(mesh.View(first, last - first), storage)
        first, buffer = borrowed[address]
        return Argument(buffer, (base - first) % (1 << 64), base)

    # operand a's invocation argument: the generator's own buffer, an outer input's storage, or this call's result
    def argument(self, a, operand, given, allocated, borrowed):
        if operand.source[0] == "generator":
            return Argument(self.generator[operand.source[1]], 0, None)
        tensor = given[operand.source[1]] if operand.source[0] == "input" else allocated[operand.source[1]]
        return self.operand(tensor, operand.offset, a in self.gpu, borrowed)

    def result(self, r, leaf, given, allocated):
        if r is None:
            return leaf.literal if leaf.root is None else given[leaf.root]
        if leaf.root is None:
            return leaf.literal
        shape, stride, offset, dtype, _, _ = leaf.metadata
        if (r, leaf.root) in self.fixed:
            tensor, start = self.fixed[r, leaf.root], 0
        else:
            a = self.argued[r, leaf.root]
            operand = self.target.operands[a]
            if operand.source[0] == "result":
                return allocated[operand.source[1]].view(dtype).as_strided(shape, stride,
                                                                           offset - operand.offset // dtype.itemsize)
            tensor = given[operand.source[1]]
            start = tensor.storage_offset() * tensor.element_size() - operand.offset
        return torch.empty(0, dtype=dtype, device=tensor.device).set_(
            tensor.untyped_storage(), start // dtype.itemsize + offset, shape, stride)

    # the one invocation through `run`: metal_recording.h MetalProgramRunArguments (the native program's entry) or
    # MetalProgramRunObserved (its observation entry), with the same operands, results, lifetimes and end
    def invoked(self, run, given):
        M, _, (lib, _), _ = self.libs
        error, execution, operands = C.create_string_buffer(ROOM), self.executions, self.target.operands
        allocated = {}
        for operand in operands:
            if operand.source[0] == "result" and operand.source[1] not in allocated:
                allocated[operand.source[1]] = torch.empty(operand.source[2], dtype=torch.uint8,
                                                           device=operand.source[3])
        borrowed = {}

        def ended():
            return M.VALUE(self.ended, M._SEL(b"signaledValue"))
        try:
            table = (Argument * max(len(operands), 1))(*[self.argument(a, operand, given, allocated, borrowed)
                                                         for a, operand in enumerate(operands)])
            status = run(self.program, self.owner.context, execution, 1, self.ended, M.START, M.STARTED, table,
                         len(operands), error, ROOM)
            if status:
                raise M._execution_error({"lib": lib, "program": self.program}, status,
                                         f"execution {execution} of the program: {error.value.decode()}")
            self.executions = execution + 1
            M._wait(lambda: ended() > execution, ended, f"execution {execution} of the program: no end for "
                                                        "MESH_PLATFORM_NS")
        finally:
            for _, buffer in borrowed.values():
                M.PUT(buffer, M.SELECTOR[b"release"])
        return [self.result(r, leaf, given, allocated) for r, leaf in self.target.outputs]

    def __call__(self, *given):
        return self.invoked(self.libs[2][0].MetalProgramRunArguments, given)

    # design/unequal-shards-change-request.md items 6, 7 and 14; design/aot-compilation-change-request.md A3: the
    # outputs and each compiled unit executed in this invocation (metal_recording.h MetalProgramRunObserved,
    # MetalProgramUnits and MetalProgramObserved, bound by compiler/mesh_compiler/metal.py), with its last executed
    # residue's members and source identities
    def observe(self, *given):
        lib, error = self.libs[2][0], C.create_string_buffer(ROOM)
        run, units, read = lib.MetalProgramRunObserved, lib.MetalProgramUnits, lib.MetalProgramObserved
        outputs = self.invoked(run, given)
        count, slices, members, identities = (mesh.U(), C.POINTER(metal.UnitSlice)(), C.POINTER(mesh.U)(),
                                              C.POINTER(mesh.Q)())
        unit = units(self.program, C.byref(count), C.byref(slices), C.byref(members), C.byref(identities))
        observed = (metal.Observed * max(count.value, 1))()
        checked(read(self.program, observed, count.value, error, ROOM), error, "observation")
        made_ = []
        for u in range(count.value):
            got = observed[u]
            if got.executions:
                part = slices[unit[u].slice + got.residue]
                made_.append(Observation(unit[u].kind, unit[u].call, unit[u].site,
                                         tuple(members[part.member:part.member + part.count]),
                                         tuple(identities[part.identity:part.identity + part.identities]),
                                         got.executions, got.seconds))
        return outputs, made_


# The saved artifact, its joint result (compiler/mesh_compiler/schedule.py loaded_program: loaded, never planned
# again) and its typed recipe, as the frontend's Root and Output records; no construction is retained
def restore(saved, directory):
    from .lower_metal import Root
    from .schedule import loaded_program
    directory = pathlib.Path(directory)
    artifact, recipe = (directory / saved["artifact"]).read_bytes(), saved["recipe"]
    bodies = tuple(Body(
        k, None, {}, tuple(Root(*root) for root in body["inputs"]),
        tuple(leaf_restored(leaf) for leaf in body["outputs"]), tuple(Root(*root) for root in body["state"]),
        {tuple(task): tuple(ops) for task, ops in body["operations"]}, body["identity"],
        {name: (kind, torch.device(device)) for name, kind, device in body["targets"]}, {
            "received": [(v, f, i, bytes.fromhex(d)) for v, f, i, d in body["marks"]["received"]],
            "sent": [(v, bytes.fromhex(d)) for v, d in body["marks"]["sent"]],
            "sending": dict(map(tuple, body["marks"]["sending"])), "transfers": body["marks"]["transfers"],
            "placed": [tuple(entry) for entry in body["marks"]["placed"]]}, tuple(body["slots"]),
        tuple(map(tuple, body["placeholders"])), tuple(map(tuple, body["results"])))
        for k, body in enumerate(recipe["bodies"]))
    return Target(bodies, tuple(Row(**row) for row in recipe["rows"]), pytree.treespec_loads(recipe["spec"]),
                  tuple((r, leaf and leaf_restored(leaf)) for r, leaf in recipe["outputs"]),
                  tuple(Operand(tuple(operand["source"]), operand["offset"], tuple(map(tuple, operand["occurrences"])))
                        for operand in recipe["operands"]),
                  tuple(tuple(map(tuple, row)) for row in recipe["joined"]), tuple(recipe["counts"]),
                  tuple(recipe["steps"]), tuple((k, tuple(words)) for k, words in recipe["generators"]),
                  tuple(map(tuple, recipe["constants"])), tuple(map(tuple, recipe["ports"])), recipe["node"],
                  loaded_program((directory / saved["joint"]).read_bytes()), artifact=artifact,
                  signature=bytes.fromhex(saved["signature"]))


# design/aot-compilation-change-request.md A10 and appendix item 6; design/compilation-composition-change-request.md
# items 3, 4, 6 and 7: the captured program's one native machine program over every call of analysis.keys/calls. Each
# function with a body on this node (`machines[key]`; one without keeps its rows, with no numerical program) its one
# typed construction (its Declarations `declarations[key]`) lowered once with its interface slots; every
# lowered function's actual Actors and Storages and every call's original bindings in ONE joint solve
# (schedule.solve_program over `analysis`, the call analysis); each call a row specialized from its function, its
# operands relabelled and its inter-call roots placed from the joint result; then one finalized, signed and saved
# native program. `node` the program's machine (Target.node), `planning` the bridge's context where calls transfer
def compile_program(captured, machines, declarations, analysis, node, planning=None):
    from .schedule import solve_program
    def built(machine):
        return ((tuple(machine.inputs), tuple(machine.outputs), tuple(machine.state), dict(machine.operations),
                 machine.identity, dict(machine.targets)) if machine is not None else ((), (), (), {}, 0, {}))
    bodies = tuple(Body(key, machine, declarations[key].wires(), *built(machine))
                   for key, machine in ((key, machines.get(key)) for key in analysis.keys))
    calls = tuple(analysis.calls[j] for j in range(len(analysis.nodes)))
    target = Target(bodies, tuple(Row(j, f, planned=j) for j, f in enumerate(calls)), captured.exported.out_spec,
                    node=node, generators=tuple((k, tuple(body.machine.initial_words)) for k, body in enumerate(bodies)
                                                if body.state))
    target.lowered(C.addressof(planning.context) if planning is not None else None, calls)
    libs = libraries()
    M, _, (_, device), _ = libs
    functions, quantum = target.functions(libs)
    bindings = target.bindings(captured, analysis)
    joint = solve_program(analysis, functions, bindings, quantum, planning, declarations=declarations,
                          wire_values={key: {value: wire.root for value, wire in machine.wires.items()}
                                       for key, machine in machines.items()})
    target.resolved(captured, analysis, joint)
    target.solution, target.plan = joint, joint.plan if planning is not None else None
    with Owned() as owned:
        gpu = device()
        owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
        target.compile(Arena(target, owned, M, gpu) if planning is None else Mapped(target))
        owned.finalizer()
    return target

# design/compilation-composition-change-request.md items 3, 4, 6 and 7; design/aot-compilation-change-request.md A5,
# A7 and A8; design/plan-to-bridge-change-request.md R2 and R8; metal-microbench metal_ir.h metal_ir_value_external,
# metal_ir_lower, metal_ir_specialize, metal_ir_storages, metal_ir_finalize, metal_ir_loaded, metal_ir_archive_load;
# metal_recording.h MetalRecorderActors, MetalRecorderPrepare, MetalRecorderReplay, MetalRecorderProgram,
# MetalProgramSave, MetalProgramLoad, MetalProgramSignature and MetalProgramRunArguments; rdma/mesh-collective.h
# mesh_plan, mesh_plan_local; rdma/mesh-metal.h mesh_metal_range_declare and the prepared transport
import ctypes as C
import dataclasses
import errno
import itertools
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


# one executed occurrence slice of Bound.observe (metal_recording.h struct MetalSlice, MetalObserved): its unit's kind,
# call (the Target.rows index) and native site, its residue, its member dispatch IDs and source operation identities
# (lower_metal.construct's global task index k), `executions` the issued occurrences of its unit's site at that residue
# (not its conditionally executed inner operations) and `sample_seconds` the elapsed time of the last of them
@dataclasses.dataclass(frozen=True)
class Observation:
    kind: int
    call: int
    site: int
    residue: int
    dispatches: tuple
    identities: tuple
    executions: int
    sample_seconds: float


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


# metal-microbench metal_ir.h metal_ir_storages, metal_ir_value_info and metal_ir_step_state: the lowered program's
# actual Storages as plan values, one per ValueID (run 1 a root its own storage holds: neither an explicit root (but
# an `intrinsic` one, an unbound intrinsic wire's typed root, Target.constructed, the program keeps no pages of:
# metal_ir_value_info `pages` 0, its flat-buffer requirement, one contiguous copy each in the joint arena,
# mesh-collective.c plan_block), nor fixed storage (a `buffer`: metal_ir_value_data's or metal_ir_value_storage's),
# nor the step; a CONSTANT or STATE root whose initial contents only `data` holds is planned, the recorder copying
# them into each placed copy, metal_capture.m MetalInitialContents), and its step word (8 bytes, executions, lag 1)
# where it has a step value (`word` METAL_IR_STEP, found in the same pass: metal_ir_step_value would construct one
# after lowering) and the host writes it; the quantum the largest placed alignment
def storages(program, explicit, intrinsic, libs):
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
        held = v not in explicit or v in intrinsic and not info.pages
        run = bool(stored.bytes and info.storage == v and held and v != step and not info.buffer)
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


# rdma/mesh-collective.h mesh_view_program `page_offsets` (each retained copy's row's page as a byte offset in the
# owner's one view, at row - planned.rows.first); metal-microbench metal_ir.h metal_ir_value_paged and
# metal_ir_value_placed (a paged root's copies, `info.pages` frame rows each): a network owner's frame rows of a paged
# root whose plan value is `section` (mesh.h struct mesh_section: copy c's rows first + c stride .. + stride, their
# pages not assumed contiguous), the root's byte 0 its section's (an unbound intrinsic root, its whole typed backing:
# Target.constructed): borrowed in place where its copies are the section's own, `info.pages` rows each; otherwise
# each entry i (Target.placements: root copy i // every) its copy's first `info.pages` rows
def framed(pages, planned, section, every, info):
    first = section.first - planned.rows.first
    if every == 1 and section.stride == info.pages:
        return C.cast(C.addressof(pages) + C.sizeof(mesh.Q) * first, C.POINTER(mesh.Q)), section.count
    return (mesh.Q * max(every * section.count * info.pages, 1))(*[
        pages[first + i // every * section.stride + q] for i in range(every * section.count) for q in range(info.pages)
    ]), every * section.count


# One preparation of a local program on its joint result (`solution`, compiler/mesh_compiler/schedule.py solve_program
# or loaded_program, the same for both: its mesh.Local over every function's values and the roots): one arena
# allocation of the result's bytes (compiler/mesh_compiler/metal.py _NEW_BUFFER, the shared Metal allocation, not
# zeroed by Metal) and the result's own planned record; the copies' byte offsets read in place (local.offset). Each
# function value and joint root whose `state` the result retains (mesh.h struct mesh_value; metal-microbench metal_ir.h
# metal_ir_value_zeroed: zero where placed, once) has each of its copies (its first/offset views) zeroed once here,
# exactly its section.bytes (a network result's plan_bind_table zeroes them on its own mapping). A local program has no
# transport, context or status
class Arena:
    context = None

    def __init__(self, target, owned, M, gpu):
        joint, error = target.solution, C.create_string_buffer(ROOM)
        self.solution, self.function, self.planned = joint, joint.functions, joint.planned
        self.base = C.cast(joint.offsets.items, C.c_void_p).value
        buffer = made(M._NEW_BUFFER(gpu, M._SEL(b"newBufferWithLength:options:"), max(joint.local.bytes, 1), M.SHARED),
                      error, "the program's arena", errno.ENOMEM)
        owned(lambda: M.PUT(buffer, M.SELECTOR[b"release"]))
        self.buffer, contents = buffer, M.OBJECT(buffer, M._SEL(b"contents"))

        def zeroed(first, index, value):
            offsets, count = self.span(first, index)
            for copy in range(count):
                C.memset(contents + offsets[copy], 0, value.section.bytes)
        for f in range(joint.local.functions):
            function = joint.functions[f]
            for v in (v for v in range(function.values) if function.value[v].state):
                zeroed(joint.function_value_first[f], v, function.value[v])
        for s in (s for s in range(joint.local.roots) if joint.roots[s].state):
            zeroed(joint.root_first, s, joint.roots[s])

    def bound(self, owned, M):
        return self.buffer

    # value `index` of a solved first view (schedule.Solved function_value_first[f] or root_first): its copies' offsets
    def span(self, first, index):
        return viewed(first, self.base, index)


# rdma/mesh-collective.h mesh_plan_program and mesh_view_program: a program with declared transfers at compilation, on
# its joint network result (`solution`, under the Planning context, whose local.offset holds each copy's first row):
# each retained copy's first row's byte offset (`offset`) and each of their rows' (`pages`, by row -
# planned.rows.first), both owned here, in the one mapping `view` of the context's region, the arena that view;
# `mapping` the mapped bytes' lifetime owner, the joint result's native owner (schedule.Solved owner, holding the result
# and, through its declaration owner's planning, the compile context; never this wrapper or the Target), held by the
# arena's backing (torch.frombuffer keeps that object, so every view of the arena does) and by its buffer; the run-time
# owner is bridge.Published
class Mapped:
    def __init__(self, target):
        joint = target.solution
        self.solution, self.function, self.planned = joint, joint.functions, joint.planned
        self.context, self.mapping = C.addressof(joint.context), joint.owner
        self.offset, view = (mesh.Q * max(joint.local.offsets, 1))(), mesh.View()
        self.pages = (mesh.Q * max(joint.local.planned.rows.count, 1))()
        mesh.LIB.mesh_view_program(C.byref(joint.context), C.byref(joint.local), self.offset, self.pages,
                                   C.byref(view))
        self.base = C.addressof(self.offset)
        backing = (C.c_byte * view.bytes).from_address(view.address) if view.bytes else None
        if backing is not None:
            backing.mapping = joint.owner
        self.arena = torch.frombuffer(backing, dtype=torch.uint8) if backing is not None else torch.empty(
            0, dtype=torch.uint8)

    def bound(self, owned, M):
        return planned_buffer(self, owned, M)

    def span(self, first, index):
        return viewed(first, self.base, index)

    def paged(self, section, every, info):
        return framed(self.pages, self.planned, section, every, info)


# compiler/mesh_compiler/metal.py _placed_buffer (metal_recording.h MetalBufferBorrow, its release callback): a plan
# owner's mapped arena as the one Metal buffer every placement takes, the buffer holding that arena's lifetime owner
# (`mapping`: bridge.Published's _Mapping, Mapped's joint result owner) until it is released
def planned_buffer(owner, owned, M):
    buffer = M._placed_buffer(mesh.View(owner.arena.data_ptr(), owner.arena.numel()), owner.mapping)
    owned(lambda: M.PUT(buffer, M.SELECTOR[b"release"]))
    return buffer


# compiler/mesh_compiler/lower_metal.py ConstructedMachine (frontend-owned): one compiled function's constructed
# program and typed roots, `machine` None for a function with no body on this node. `key` its captured function key
# (compilation-only; its ordinal is its identity), `declared` its Declarations.wires() (channel -> declared transfer,
# logical wire value, its logical bytes, declared sent view, declared landed view; each call's ports are its Row's).
# `slots` its callable interface roots in their stable order (each marked metal_ir_value_external with its slot
# before its one numerical construction: an input or state root, its generator state, each
# further output root but a CONSTANT); `placeholders` and `results`, per placeholder i of the original captured function
# and output spec o, (slot, root, view, the view's byte offset in its root), slot None where the root is no interface
# slot, all None where the placeholder has no physical root; `logical` (compilation-only) each original captured
# function placeholder's compiled input Root (ConstructedMachine.inputs by Root.name, the placeholder's target), None
# where the assembled body has no such placeholder (an unused one its specialization omitted)
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
    slots: tuple = ()
    placeholders: tuple = ()
    results: tuple = ()
    logical: tuple = ()


# compiler/mesh_compiler/schedule.py declare_program's and solve_program's `bindings`: call j's
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
# function's `every` calls in the block, `planned` its Planned.call index; `instance` its call instance (the
# Function[] index planned.call[planned].function, plan_calls[call]), whose lowered program the row emits; `marks`
# that call's own declared ports (its Ports), each by the logical identity of its row, (owner call, transfer):
# `received` (ValueID, its section's plan value (instance, ValueID) or (None, shared program root s), declared
# mesh_metal_range, which holds its owner and transfer), `sent` (ValueID, owner call, declared mesh_metal_release, one
# per canonical release), `sending` {(owner call, transfer): ValueID}, `transfers` (owner call, transfer), receives
# then sends, `placed` (ValueID, instance, plan value), one per wire. The row's program's correspondence to its body
# (metal-microbench metal_ir.h metal_ir_append, a composite row's): `values` each original body ValueID's view ValueID
# in it, `roots` that view's storage root (metal_ir_value_info `storage`), the append's own arrays (lists once saved),
# `offset` the body's first operation in it, () and 0 where the row runs the body's own IDs; `copies` each copy it
# holds, (kind "stage" or "land", output position (an outer land's outer one), input position (a normal land's), joint
# wire root, the wire's one ValueID here, its endpoint root: the stage's source output root, an outer land's outer
# result root, a normal land's original input root's prefix root; its copy operation in the row's program, None for a
# landing with no landing view)
@dataclasses.dataclass
class Row:
    call: int
    function: int
    ordinal: int = 0
    every: int = 1
    planned: int = 0
    instance: int = 0
    values: tuple = ()
    roots: tuple = ()
    offset: int = 0
    copies: tuple = ()
    marks: dict = dataclasses.field(default_factory=lambda: {"received": [], "sent": [], "sending": {},
                                                             "transfers": [], "placed": []})


# One port of a call's instance on this machine, from the declaration owner (compiler/mesh_compiler/schedule.py
# declare_program, its Prepared; rdma/mesh-metal.h mesh_metal_range_declare): `direction` mesh.RECEIVE or mesh.SEND,
# `transfer` its row in owner call `owner`'s combined declared array (Prepared.declared[owner]), `root` the instance's
# typed root, `bytes` the backing extent of the plan value it lands in (an intrinsic wire's: its declared value's bytes,
# Declarations.values; a cross-call row's PortRecord extent; never a view's payload), landing at byte `at` of that
# root's storage, `value` the ValueID its sending lifetime binds, `section` the plan value whose section the range is
# prepared at ((instance, ValueID) an intrinsic wire's, (None, s) the shared program root s's, Solved.roots[s]); a sent
# port's canonical release (mesh-collective.h mesh_declared_group): `release` its release ordinal (None: the row its
# own) and `first` that release's representative transfer (its demand's first row)
@dataclasses.dataclass(frozen=True)
class Port:
    direction: int
    transfer: int
    owner: int
    root: int
    bytes: int
    at: int
    value: int
    section: tuple
    release: object = None
    first: int = None


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
# its flat outer outputs (`outputs`: (row, the frontend's Output) of a call's result produced here; (("root", the
# shared program root's source record (schedule.Solved root_sources), the byte of that root the bound backing's byte 0
# is), Output with the leaf's metadata in that root) of a result a call on another machine produces, the supplied
# owner root; (None, Output) of an outer placeholder's position or a literal; in exported.outputs order), each row's
# planned inter-call roots (`joined`: (root ValueID, joint root s, byte delta, the call's every)), each instance's
# ValueID count and whether its step follows (`counts`, `steps`), its bodies' generators' initial words
# (`generators`: (body, ConstructedMachine.initial_words)), each row's CONSTANT roots' outer constants (`constants`:
# (row, root ValueID, the outer placeholder's name)); `node`
# the program's machine (its map index, the plan's node map's with a plan, the source machine's without); `solution` the
# joint owner (compiler/mesh_compiler/schedule.py: solve_program's Solved, holding its declaration owner Prepared;
# restored, loaded_program's local Solved or a network result's SavedNetwork, its saved bytes), saved once, read by no
# preparation (each preparation owner holds its own prepared Solved); `plan` that owner's network result (its own
# `.plan`: None for a local result), the one Scheduled.plan references; `artifact` the one MetalProgramSave of every
# row, `signature` its MetalProgramSignatureNode signature's complete solved archive (rdma/mesh_graph.py archive; the
# one bytes owner, which Scheduled.signature is); `program` the network program
# (Planning.program, the mesh_program_save/load mask), None for a local result
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
    node: int = None
    solution: object = None
    plan: object = None
    artifact: bytes = None
    signature: bytes = None
    program: int = None

    # row j's explicit roots, in its program's ValueIDs: its body's interface slots and input, state, received and sent
    # roots, and each of its copies' wire and endpoint roots
    def explicit(self, j):
        row = self.rows[j]
        body = self.bodies[row.function]
        built = body.machine
        own = () if built is None else (*body.slots, *(r.root for r in (*body.inputs, *body.state,
                                                                          *built.received.values(),
                                                                          *built.sent.values())))
        return {self.root_of(j, root) for root in own if root is not None} | {v for copy in row.copies
                                                                               for v in copy[4:6]}

    # metal-microbench metal_ir.h metal_ir_value_info `pages` (its storage root's, metal_ir_value_paged's, as the
    # finalized or loaded `program` retains it) and metal_ir_value_placed: ValueID `value` of `program` placed at its
    # plan value (`section`, value `index` of the solved first view `first`) on the preparation owner, as (offsets,
    # copies), through the call's generation (execution e = t every + ordinal reads copy t = e // every, so entry i of
    # its every x copies table is copy i // every): a paged root's frame rows (the network owner's `paged`, framed; an
    # unbound intrinsic root, Target.constructed: no joint binding's delta reaches it); otherwise its copies' byte
    # offsets (`span`) moved by `delta`, a joint binding's signed delta as given
    def located(self, owner, program, libs, value, section, first, index, every=1, delta=0):
        M, compiler, _, _ = libs
        info = M.ValueInfo()
        compiler.metal_ir_value_info(program, value, C.byref(info))
        if info.pages:
            return owner.paged(section, every, info)
        offsets, count = owner.span(first, index)
        if every == 1 and not delta:
            return offsets, count
        return (mesh.Q * (every * count))(*[offsets[i // every] + delta for i in range(every * count)]), every * count

    # row r's own storages and inter-call roots on the preparation owner (its prepared schedule.Solved, `solution`),
    # each root once: each run value of its call instance at its full copies (function_value_first[instance]); each
    # marked unbound intrinsic wire (marks["placed"]: (ValueID, instance, plan value)) of no run at its own section;
    # each planned inter-call root at the joint root's copies (root_first[s]) with that root's shares, through the
    # call's generation and the binding's byte delta; each `located` in row r's `program`
    def placements(self, owner, r, program, libs):
        f, joint = self.rows[r].instance, owner.solution
        value, first = owner.function[f].value, joint.function_value_first[f]
        for v in range(self.counts[f]):
            if value[v].run:
                yield (v, *self.located(owner, program, libs, v, value[v].section, first, v), value[v].shares)
        for v, function, index in self.rows[r].marks["placed"]:
            held = owner.function[function].value[index]
            if not held.run:
                yield (v, *self.located(owner, program, libs, v, held.section, joint.function_value_first[function],
                                        index), held.shares)
        for root, s, delta, every in self.joined[r]:
            yield (root, *self.located(owner, program, libs, root, joint.roots[s].section, joint.root_first, s, every,
                                       delta), joint.roots[s].shares)

    def stepped(self, owner, r):
        f = self.rows[r].instance
        if not self.steps[f]:
            return None
        return (*owner.span(owner.solution.function_value_first[f], self.counts[f]),
                owner.function[f].value[self.counts[f]].shares)

    # metal_recording.h MetalRecorderPrepare, MetalRecorderPlan and MetalRecorderReplay (metal_capture.m MetalRecorder:
    # no capture is prepared after the recorder's first replay): the one recorder of the whole machine's rows, every
    # row's program, in order, transport-prepared, captured from its lowered typed program, its placement descriptors
    # given (values, step, callable operands, transport state), its replay planned on its capture (segments, cuts,
    # ranges: MetalRecorderPlan), then finalized once (or relocated, an archive's: metal_ir_loaded), each capture and
    # its transport and placement owners retained with `owned`; then those same (row, capture, program) in original
    # row order, each replayed (its argument storage and range workspace bound: MetalRecorderReplay), the replay
    # retained, then its program's physical accesses made on that storage (metal_ir_place), a CPU-only program's too;
    # a row with no program a NULL call; each call its planned call's
    def placed(self, owned, libs, gpu, owner, programs, finalize):
        M, compiler, (lib, _), transport = libs
        error, planned = C.create_string_buffer(ROOM), owner.planned
        recorder = made(lib.MetalRecorderOpen(gpu, error, ROOM), error, "recorder", errno.ENODEV)
        owned(lambda: M.PUT(recorder, M.SELECTOR[b"release"]))
        buffer, captured = owner.bound(owned, M), []
        for r, (row, program) in enumerate(zip(self.rows, programs)):
            if program is None:
                captured.append((row, None, None))
                continue
            marks, wires, call = row.marks, None, row.planned
            if marks["transfers"]:
                context, wires, word = mesh.Context.from_address(owner.context), M.Transport(), mesh.Q()
                mesh.check(transport.mesh_metal_transport_create(C.byref(context), C.byref(planned), gpu,
                                                                 C.byref(wires)))
                owned(lambda wires=wires: transport.mesh_metal_transport_destroy(C.byref(wires)))
                # rdma/mesh-metal.h: one transport view of every planned call's plan; each record prepared from the
                # plan of the call whose transfer it names (a range from its declared owner, a release and a
                # sending lifetime from their marks' owner call), never the emitting row's by default; the program's
                # status word its own call's plan (mesh_metal_status_prepare: a capture's own call)
                for value, function, index, declared in marks["received"]:
                    landed = M.Range.from_buffer_copy(declared)
                    field = (mesh.U * max(landed.partitions, 1))()
                    held = owner.function[function].value[index] if function is not None else (
                        owner.solution.roots[index])
                    mesh.check(transport.mesh_metal_range_prepare(
                        C.byref(context), C.byref(wires), C.byref(planned), held.section, field, landed.partitions,
                        C.byref(landed)))
                    compiler.metal_ir_value_received(program, value, C.byref(landed))
                for value, sender, declared in marks["sent"]:
                    release = M.Release.from_buffer_copy(declared)
                    mesh.check(transport.mesh_metal_release_prepare(C.byref(context), C.byref(wires),
                                                                    C.byref(planned), sender, C.byref(release)))
                    compiler.metal_ir_value_sent(program, value, C.byref(release))
                for sender, transfer in marks["transfers"]:
                    sending = M.Sending()
                    mesh.check(transport.mesh_metal_sending_prepare(C.byref(context), C.byref(wires),
                                                                    C.byref(planned), sender, transfer,
                                                                    C.byref(sending)))
                    if sending.sent.sends:
                        compiler.metal_ir_value_sending(program, marks["sending"][sender, transfer],
                                                        C.byref(sending))
                mesh.check(transport.mesh_metal_status_prepare(C.byref(context), C.byref(wires),
                                                               planned.call[call].plan, C.byref(word)))
                compiler.metal_ir_transport(program, word.value)
            ready = made(lib.MetalRecorderPrepare(recorder, program, error, ROOM), error, "preparation")
            owned(lambda ready=ready: M.PUT(ready, M.SELECTOR[b"release"]))
            for value, offsets, count, shares in self.placements(owner, r, program, libs):
                compiler.metal_ir_value_placed(program, value, buffer, offsets, count, 0, shares)
            step = self.stepped(owner, r)
            if step is not None:
                compiler.metal_ir_step_placed(program, buffer, *step)
            # metal_ir.h metal_ir_value_placed with no buffer: each callable operand's root one logical copy, shared
            # by every execution (a state or mutated input is the caller's one storage), its address the invocation's
            for root in (root for operand in self.operands for at, root in operand.occurrences if at == r):
                compiler.metal_ir_value_placed(program, root, None, None, 1, 0, 0)
            # metal_recording.h MetalRecorderState: the transport's state buffer and this program's status word offset
            if wires is not None:
                checked(lib.MetalRecorderState(recorder, ready, wires.state, word.value, error, ROOM), error,
                        "the transport state")
            checked(lib.MetalRecorderPlan(recorder, ready, error, ROOM), error, "replay plan")
            emitted = compiler.metal_ir_finalize if finalize else compiler.metal_ir_loaded
            checked(emitted(program, error, ROOM), error, "emission" if finalize else "relocation")
            captured.append((row, ready, program))
        calls = []
        for row, ready, program in captured:
            if ready is None:
                calls.append(M.Call(None, row.planned + 1, None, 0))
                continue
            replay = M.Replay()
            checked(lib.MetalRecorderReplay(recorder, ready, C.byref(replay), error, ROOM), error, "replay")
            owned(lambda replay=replay: lib.MetalReplayDestroy(C.byref(replay)))
            checked(compiler.metal_ir_place(program, error, ROOM), error, "placed accesses")
            calls.append(M.Call(C.pointer(replay), row.planned + 1, None, 0))
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
                                                 if r is not None and r.root is not None else (None, None, None, 0)
                                                 for r in records)
                                           for records in (body.logical, body.outputs))

    # compiler/mesh_compiler/schedule.py Prepared.copies, remap and bind; metal-microbench metal_ir.h
    # metal_ir_value_external, metal_ir_value_planned and metal_ir_append; compiler/mesh_compiler/metal.py _Builder
    # (stored, viewed, copy: typed roots, views and copies over ValueIDs): each numerical body's slots
    # marked external first (the callable table capability, kept by every append and clone); then, per call j with
    # copies on this machine (`here`, CopyRecord.machine), one composite program per equal (function, copy geometry,
    # sharing):
    # - one full byte backing per joint wire (CopyRecord.wire) and per landed result storage (CopyRecord.storage), each
    #   copy's typed view over its backing at the producer's geometry (CopyRecord.view, the leaf's dtype), so every view
    #   and served transfer of one wire is one root; a wire a TEMPORARY root (planned, not callable);
    # - the prefix: the ORIGINAL logical input root of each normal land's (`input` not None) consumer input (its
    #   InputPart `input` Root: that root's actual scalar and extent, ValueInfo), a root of that Root's role marked
    #   external with the input's slot; each part's landing view in it (InputPart.landing, its geometry as compiled)
    #   copied from the same region of the wire (which holds the source storage's bytes at their own offsets, the
    #   input binding's delta apart), on the part's processor; then the body appended unconstructed with that input
    #   root backed by it (metal_ir_append's given value: no partition the body materializes is written here);
    #   `values` and `roots` the append's actual arrays (each original ValueID's view ValueID and its storage root,
    #   ValueInfo.storage), `offset` the body's first operation;
    # - the suffix: each stage (its owner's output Slot view as appended, `values`, copied into its wire) and each outer
    #   land (its wire copied into the outer result root, a root of the storage's caller ownership, Solved.root_sources:
    #   an input's INPUT, a result's OUTPUT, marked external after the body's slots; the leaf's own whole view of it the
    #   returned view);
    # each copy one typed copy (metal.py _Builder.copy) over its transfer domain (a stage's and an outer land's
    # CopyRecord.view, a normal land's InputPart.geometry, each as mesh_graph.strided: a zero-stride dimension its one
    # element), its views their actual layouts (a normal land's its landing view's complete native descriptor), on its
    # CopyRecord processor's selected target (`targets`), its operation kept. A call with no body here has the
    # copies alone, its copy-only row. The appended calls' arrays go once to the declaration owner (Prepared.remap),
    # then each wire's ValueID (`wired`) and each outer land's result root binding (`copied`, delta 0: the root covers
    # the storage from byte 0) (Prepared.bind); `composites[j]` call j's program, None where it runs its body's own,
    # `copying[j]` each of call j's CopyRecords with its copy operation in that program (Row.copies' operation: the
    # copies' provenance), `landed` {outer output position: (call, its Output of OutputKind kinds[position])}, the
    # composites released with `owned`
    def composed(self, prepared, kinds, here, libs, owned):
        from .lower_metal import Output
        from .operations import _metadata
        M, compiler, _, _ = libs
        for body in (body for body in self.bodies if body.machine is not None):
            for slot, root in enumerate(body.slots):
                mesh.check(compiler.metal_ir_value_external(body.machine.program, root, slot))
        mine, extents = {}, {wire.root: wire.extent for wire in prepared.wires}
        for copy in (copy for copy in prepared.copies if copy.machine == here):
            mine.setdefault(copy.call, []).append(copy)
        built, values, roots, wired, copied = {}, {}, {}, {}, []
        self.composites, self.landed, self.copying = [None] * len(self.rows), {}, {}
        for j in sorted(mine):
            row = self.rows[j]
            body = self.bodies[row.function]
            listed = sorted(mine[j], key=lambda copy: (copy.kind != "land" or copy.input is None, copy.kind,
                                                       copy.input if copy.input is not None else -1,
                                                       copy.output if copy.output is not None else -1, copy.wire))
            wires = {wire: k for k, wire in enumerate(dict.fromkeys(copy.wire for copy in listed))}
            storages = {storage: k for k, storage in enumerate(dict.fromkeys(copy.storage for copy in listed))}
            geometry = (row.function, tuple((copy.kind, copy.output, copy.input, copy.view[0], copy.view[1],
                                             tuple(map(tuple, copy.view[2])), copy.leaf.meta["val"].dtype,
                                             copy.processor, wires[copy.wire], extents[copy.wire],
                                             storages[copy.storage], copy.part and copy.part.landing)
                                            for copy in listed))
            if geometry not in built:
                built[geometry] = self.composite(prepared, body, listed, extents, libs, owned)
            program, mapped, stored, offset, ends = built[geometry]
            if mapped is not None:
                values[j], roots[j] = mapped, stored
            self.composites[j] = program
            self.copying[j] = tuple((copy, operation) for copy, (_, _, _, operation) in zip(listed, ends))
            held = tuple((copy.kind, copy.output, copy.input, copy.wire, root, end, operation)
                         for copy, (root, end, _, operation) in zip(listed, ends))
            self.rows = (*self.rows[:j], dataclasses.replace(row, values=mapped if mapped is not None else (),
                                                             roots=stored if stored is not None else (), offset=offset,
                                                             copies=held), *self.rows[j + 1:])
            for copy, (root, end, view, _) in zip(listed, ends):
                wired[j, copy.wire] = root
                if copy.kind == "land" and copy.input is None:
                    copied.append((j, end, copy.storage, 0, copy.leaf))
                    self.landed[copy.output] = (j, Output(end, view, None, _metadata(copy.leaf.meta["val"]),
                                                          kinds[copy.output], None))
        prepared.remap(values, roots)
        prepared.bind(wired, copied)

    # one composite (Target.composed): its program, the append's `values` and `roots` arrays (None without a body), the
    # body's first operation and, per copy of `listed`, (its wire root, its endpoint root, the endpoint's view, its copy
    # operation: None for a landing with no landing view)
    def composite(self, prepared, body, listed, extents, libs, owned):
        from .lower_metal import PLACES, Bound, _element, _typed
        M, compiler, _, _ = libs
        error, info = C.create_string_buffer(ROOM), M.ValueInfo()
        widths = {code: dtype.itemsize for dtype, code in M.SCALAR.items()}
        program = compiler.metal_ir_program_new()
        owned(lambda: compiler.metal_ir_program_free(program))
        machines = M._machines()
        if machines:
            compiler.metal_ir_machine(program, C.byref(machines[M.GPU]), C.byref(machines[M.CPU]))
        builder, roots, ends, slots = M._Builder(compiler, 0, program=program), {}, {}, len(body.slots)

        # metal_ir.h metal_ir_value (compiler/mesh_compiler/metal.py _Builder.stored): one storage root per backing
        # (`key`: a joint wire, a body's prefix input root, an outer result storage), of role `role`, `elements` of
        # metal_ir scalar `scalar`, made on its first copy
        def rooted(key, role, scalar, elements):
            if key not in roots:
                roots[key] = (builder.stored(role, scalar, elements), role)
            return roots[key][0]

        # a typed view (metal.py _Builder.viewed) of root `key` at metal_ir scalar `scalar`, `sizes`, element `strides`
        # and element `offset`: (its ValueID, scalar, sizes)
        def domain(key, scalar, sizes, strides, offset):
            return builder.viewed(*roots[key], scalar, sizes, strides, offset), scalar, tuple(sizes)

        # an original transfer domain (mesh_graph.strided: byte offset `start` moved by `delta` bytes, element bytes,
        # pairs, a zero-stride dimension already its one element) in elements: sizes, strides, offset
        def elements(start, element, pairs, delta=0):
            return tuple(n for n, _ in pairs), tuple(stride // element for _, stride in pairs), (start + delta) // element

        # the typed view (lower_metal.py _typed: a complex dtype's real pairs) of root `key` at an original view of
        # dtype `dtype`: its sizes, element strides and element offset
        def typed(key, dtype, sizes, strides, offset):
            return domain(key, *_typed(Bound(None, dtype, tuple(sizes), tuple(strides), offset)))

        # each copy's joint wire root (TEMPORARY, its whole extent of the leaf's element) and its view there over the
        # transfer's original domain (CopyRecord.view)
        def wire(copy):
            dtype = copy.leaf.meta["val"].dtype
            root = rooted(("wire", copy.wire), M.TEMPORARY, M.SCALAR[_element(dtype)],
                          extents[copy.wire] // _element(dtype).itemsize)
            return root, typed(("wire", copy.wire), dtype, *elements(*copy.view))

        # metal.py _Builder.copy (metal_ir.h metal_ir_copy): one typed copy of view `given` into view `into` (each
        # (ValueID, metal_ir scalar, ...)) over the transfer domain `into[2]`, on its CopyRecord processor's selected
        # target, the record its provenance (newly produced physical-copy work: no source position); its operation
        # returned
        def copied(copy, into, given):
            return builder.copy(copy, PLACES[self.targets[copy.processor][0]], 0, into[2], into[:2], given[:2])

        # the complete native descriptor of value `value` of the body's original program as compiled (metal_ir.h
        # metal_ir_value_info scalar, metal_ir_value_shape, metal_ir_value_layout: its every term and element offset)
        def descriptor(value):
            source, at = body.machine.program, C.c_int64()
            compiler.metal_ir_value_info(source, value, C.byref(info))
            rank = compiler.metal_ir_value_shape(source, value, None, 0)
            shape = (mesh.Q * max(rank, 1))()
            compiler.metal_ir_value_shape(source, value, shape, rank)
            count = compiler.metal_ir_value_layout(source, value, None, 0, C.byref(at))
            terms = (M.Term * max(count, 1))()
            compiler.metal_ir_value_layout(source, value, terms, count, C.byref(at))
            return info.scalar, rank, shape, count, terms, at.value

        # that descriptor as a view of root `root` of role `role` here (metal_ir_value, metal_ir_value_view), its
        # element offset moved by `delta` elements: (its ValueID, scalar, shape)
        def described(root, role, made, delta=0):
            scalar, rank, shape, count, terms, at = made
            v = compiler.metal_ir_value(program, role, scalar, rank, shape, count, terms, at + delta, 0)
            compiler.metal_ir_value_view(program, v, root)
            return v, scalar, tuple(shape[:rank])
        deltas = {(call, root, storage): delta for (call, root, storage, delta), _ in prepared.bindings}
        landing = {}
        for k, copy in enumerate(listed):
            if copy.kind != "land" or copy.input is None:
                continue
            whole, root = copy.part.input, wire(copy)[0]
            if whole.root not in landing:
                compiler.metal_ir_value_info(body.machine.program, whole.root, C.byref(info))
                landing[whole.root] = rooted(("input", whole.root), whole.role, info.scalar,
                                             info.bytes // widths[info.scalar])
                mesh.check(compiler.metal_ir_value_external(program, landing[whole.root], body.slots.index(whole.root)))
            if copy.part.landing is None:
                ends[k] = (root, landing[whole.root], None, None)
                continue
            # the InputPart landing view of the body's original program as compiled (its complete descriptor) in the
            # prefix input root, and the same view of the wire (which holds the source storage's bytes at their own
            # offsets, the input binding's delta apart), copied over the part's original landing geometry
            # (InputPart.geometry, the producer's (offset, sizes, strides)) as its transfer domain (mesh_graph.strided:
            # a zero-stride dimension its one element; lower_metal.py _typed: a complex leaf's real pairs)
            made, dtype = descriptor(copy.part.landing), copy.leaf.meta["val"].dtype
            into = described(landing[whole.root], whole.role, made)
            loops = _typed(Bound(None, dtype, *elements(*mesh_graph.strided(*copy.part.geometry, dtype.itemsize))))[1]
            operation = copied(copy, (*into[:2], loops), described(
                root, M.TEMPORARY, made, deltas[copy.call, whole.root, copy.storage] // widths[made[0]]))
            ends[k] = (root, landing[whole.root], into[0], operation)
        offset, mapped, stored = compiler.metal_ir_operations(program), None, None
        if body.machine is not None:
            count = compiler.metal_ir_values(body.machine.program)
            mapped = (mesh.U * max(count, 1))(*([M.NONE] * count))
            for v, root in landing.items():
                mapped[v] = root
            checked(compiler.metal_ir_append(program, body.machine.program, mapped, count, error, ROOM), error,
                    "the body's append")
            stored = (mesh.U * max(count, 1))()
            for v in range(count):
                compiler.metal_ir_value_info(program, mapped[v], C.byref(info))
                stored[v] = info.storage
        for k, copy in enumerate(listed):
            if k in ends:
                continue
            root, view = wire(copy)
            if copy.kind == "stage":
                # the owner's original output view (its Binding Slot's), as metal_ir_append made it here, into the
                # wire over the transfer's domain
                operation = copied(copy, view, (mapped[copy.slot.view], view[1]))
                ends[k] = (root, stored[copy.slot.root], None, operation)
            else:
                # the outer result root (its storage's caller ownership, Solved.root_sources: an input's INPUT, a
                # result's OUTPUT, the wire's extent from byte 0: the wire's own offsets), marked external after the
                # body's slots once; the wire copied into it over the transfer's domain, the original leaf's whole
                # logical view of it returned
                leaf, fresh = copy.leaf.meta["val"], ("result", copy.storage) not in roots
                end = rooted(("result", copy.storage),
                             M.INPUT if prepared.root_sources[copy.storage][0] == "input" else M.OUTPUT, view[1],
                             extents[copy.wire] // _element(leaf.dtype).itemsize)
                if fresh:
                    mesh.check(compiler.metal_ir_value_external(program, end, slots))
                    slots += 1
                operation = copied(copy, typed(("result", copy.storage), leaf.dtype, *elements(*copy.view)), view)
                ends[k] = (root, end, typed(("result", copy.storage), leaf.dtype, leaf.shape, leaf.stride(),
                                            leaf.storage_offset())[0], operation)
        return program, mapped, stored, offset, tuple(ends[k] for k in range(len(listed)))

    # row j's program's storage root (Row.roots, metal_ir_append's ValueInfo.storage) and view ValueID (Row.values) of
    # its body's original ValueID `v`; the body's own where the row runs its body's own IDs
    def root_of(self, j, v):
        roots = self.rows[j].roots
        return roots[v] if len(roots) and v is not None else v

    def view_of(self, j, v):
        values = self.rows[j].values
        return values[v] if len(values) and v is not None else v

    # compiler/mesh_compiler/schedule.py Prepared (remapped and bound: its rows', ports' and wires' ValueIDs the row
    # programs') and mesh-collective.h mesh_declared_group (Prepared.grouped[owner]: demand_of, release_of, each
    # Demand's `first`): each original call's Ports on this machine (`here`), `call_ports[j]` (its Ports, receives
    # first, and its intrinsic wires' typed roots, the rows' values), ((), ()) for a call with no program here; `joint`
    # each (call, root) a Prepared.bindings storage s or Prepared.wired wire binds, (None, s):
    # - each of call j's intrinsic endpoints on this machine (Prepared.endpoints[j], schedule.declare_program's
    #   EndpointRecords, remapped), owner j, transfer its row: its typed root (`root`), its range its `backing` at its
    #   view's byte (`view`), in its instance's Function where it is unbound or `staged` (a caller run 0 destination's:
    #   its declared receive wire, ConstructedMachine.received), else in its shared run 1 root `storage`, (None, s); a
    #   receive so bound to a shared root has its interface landing copy (ConstructedMachine.landings) in `direct[j]`,
    #   {channel: its body operation IDs}, erased at construction (Target.constructed), every other's kept;
    # - each PortRecord of Prepared.ports[j] with a typed site here: received where its destination is this machine,
    #   on its slot where its wire is its run 1 storage itself; sent where its source is, on its slot so, else on the
    #   call's staging wire ValueID (Prepared.wired); its range the record's backing extent at its byte `at`, at the
    #   shared program root `wire`;
    # - each land copy's served rows (CopyRecord.transfers of its `owner`: an outer land's its own sent rows, a normal
    #   land's the consumer's received rows), each (owner, transfer) once, received on the row's one root of that wire
    #   (Row.copies) the same way, the owner the transfer's.
    # A sent row's release is its canonical one: its release ordinal (release_of[t]) and that release's representative
    # row (its demand's `first`), one record per (owner, release)
    def ported(self, prepared, declarations, analysis, plan_calls, here):
        self.call_ports, self.direct = [], []
        self.joint = {(call, root): (None, s) for (call, root, s, _), _ in prepared.bindings}
        self.joint.update(((call, root), (None, wire)) for (call, wire), root in prepared.wired.items())
        for j, row in enumerate(self.rows):
            body, f = self.bodies[row.function], plan_calls[j]
            built = body.machine
            if built is None and self.composites[j] is None:
                self.call_ports.append(((), ()))
                self.direct.append({})
                continue

            def released(t, owner=j):
                found = prepared.grouped[owner] if prepared.grouped else None
                if found is None or found.release_of[t] == mesh.ABSENT:
                    return None, t
                return found.release_of[t], found.demand[found.demand_of[t]].first
            received, sent, wires, direct = [], [], (), {}
            if built is not None:
                intrinsic = tuple(zip(prepared.declared[j], declarations[analysis.keys[row.function]].channels))
                for e in (e for e in prepared.endpoints[j] if e.machine == here):
                    root, joint = e.root.root, e.storage is not None and not e.staged
                    section = (None, e.storage) if joint else (f, root)
                    if e.direction == mesh.RECEIVE:
                        received.append(Port(mesh.RECEIVE, e.transfer, j, root, e.backing, e.view[0], root, section))
                        if joint and e.channel in e.construction.landings:
                            direct[e.channel] = e.construction.landings[e.channel][1]
                    else:
                        sent.append(Port(mesh.SEND, e.transfer, j, root, e.backing, e.view[0], root, section,
                                         *released(e.transfer)))
                wires = tuple(dict.fromkeys(declared[4] for declared, _ in intrinsic if declared[4] != mesh.ABSENT))
            for record in prepared.ports[j]:
                staged = record.wire != record.storage
                if record.direction == mesh.RECEIVE and record.destination == here and not staged:
                    # No rejection filters, even in temporary integration validation: a declared local transfer
                    # without its required typed root is a compilation error, never an omitted port.
                    if record.slot is None or record.slot == mesh.ABSENT:
                        raise ValueError(f"call {j}: receive {record.owner}/{record.transfer} on machine {here} "
                                         "has no typed input root")
                    received.append(Port(mesh.RECEIVE, record.transfer, record.owner, record.slot, record.extent,
                                         record.at, record.slot, (None, record.wire)))
                elif record.direction == mesh.SEND and record.source == here:
                    root = prepared.wired.get((j, record.wire)) if staged else record.slot
                    if root is None or root == mesh.ABSENT:
                        raise ValueError(f"call {j}: send {record.owner}/{record.transfer} on machine {here} "
                                         f"has no typed {'staging' if staged else 'output'} root")
                    sent.append(Port(mesh.SEND, record.transfer, j, root, record.extent, record.at, root,
                                     (None, record.wire), *released(record.transfer)))
            by, landed = {(record.owner, record.transfer): record for record in prepared.ports[j]}, set()
            wired = {wire: root for _, _, _, wire, root, _, _ in row.copies}
            for copy in (copy for copy in prepared.copies if copy.call == j and copy.kind == "land"
                         and copy.machine == here):
                for t in (t for t in copy.transfers if (copy.owner, t) not in landed):
                    landed.add((copy.owner, t))
                    record, root = by[copy.owner, t], wired[copy.wire]
                    received.append(Port(mesh.RECEIVE, t, copy.owner, root, record.extent, record.at, root,
                                         (None, copy.wire)))
            self.call_ports.append((tuple(received + sent), wires))
            self.direct.append(direct)

    # metal-microbench metal_ir.h metal_ir_source_relation (it appends: once per constructed program, after its copy
    # selection): row j's body's original relation (ConstructedMachine.relation) joined to `program`, each original
    # vertex's live operations (ConstructedMachine.vertices less its direct landing copies, Target.ported `direct`, at
    # the row's append offset), each original view's and GIVEN operand's typed ValueID through the row's append map
    # (view_of), a direct receive's GIVEN operand its destination (ConstructedMachine.landings); the arrays and the
    # relation's owner adopted by the program (compiler/mesh_compiler/metal.py _lent), released with its last holder
    def related(self, program, j, libs):
        from .lower_metal import operand
        M, compiler, _, _ = libs
        row = self.rows[j]
        built, direct = self.bodies[row.function].machine, self.direct[j]
        gone = {op for ops in direct.values() for op in ops}
        vertices = [[op + row.offset for op in ops if op not in gone] for ops in built.vertices]
        flat = [op for ops in vertices for op in ops]
        views = [M.NONE if access is None or access[0] == "literal" else self.view_of(j, access[1])
                 for access in built.accesses]
        given = []
        for (kind, _, _, channel), typed in zip(built.relation.supplied, built.supplied, strict=True):
            made_ = operand(built.landings[channel][0] if kind != "value" and channel in direct else typed)
            if made_.kind in (M.SOURCE_VALUE, M.SOURCE_SENT, M.SOURCE_RECEIVED):
                made_.value = self.view_of(j, made_.value)
            given.append(made_)
        arrays = ((C.c_size_t * (len(vertices) + 1))(*itertools.accumulate(map(len, vertices), initial=0)),
                  (mesh.U * max(len(flat), 1))(*flat), (mesh.U * max(len(views), 1))(*views),
                  (M.SourceOperand * max(len(given), 1))(*given))
        error = C.create_string_buffer(ROOM)
        checked(M._lent((built.relation, arrays), lambda release, owner: compiler.metal_ir_source_relation(
            program, built.relation.relation.pointer, *arrays[:3], len(given), arrays[3], release, owner, error, ROOM),
            lambda status: status == 0), error, "the source relation")

    # metal-microbench metal_ir.h metal_ir_value_planned, metal_ir_clone, metal_ir_operation_erase and
    # metal_ir_construct; metal_ir.mm ExternallyVisible and Realize (a storage with a planned value, its symbolic plan
    # function/index identity, is visible and kept; no physical placement or buffer is implied) and metal_ir_value_paged
    # (a paged storage root, a plan value's page-table rows): each call's paged geometry, each of its intrinsic network
    # Ports' typed root whose Port.section is its function's own (instance, ValueID) (Target.ported: bound by no joint
    # root, which keeps its dense owner, Target.located), each call's such roots its `intrinsic` candidates
    # (Target.functions), with the context's page quantum (mesh.h struct hdr block x pgsz, the plan's row:
    # mesh-collective.c plan_quantum, `context` the compile owner's) and the page count of its whole typed backing
    # Port.bytes; each composite program (`composites`) constructed once per paged geometry and erased landing copies of
    # its calls; each numerical body's other calls grouped by their construction facts (each call's intrinsic wires, its
    # planned roots: a sent wire is a TEMPORARY root; its other port roots are interface roots, visible by their roles;
    # its paged geometry; its erased landing copies, `erased`: its direct receives' interface copies, Target.ported
    # `direct`, at the row's append offset); each group's program given its first call's wires' planned identity (plan
    # value (plan_calls[j], root), rebound on each instance) and paged geometry, its copies erased, its source relation
    # attached once (Target.related) and constructed once, the source program the first group's and, cloned from it
    # before its construction (metal_ir_clone: every value, operation and region identity the source's, its external
    # marks included), a copy each further group's; `templates[j]` call j's constructed program, None where it has no
    # program on this node; the copies released with `owned`
    def constructed(self, context, plan_calls, libs, owned):
        M, compiler, (_, device), _ = libs
        error, templates, sources, classes = C.create_string_buffer(ROOM), [None] * len(self.rows), {}, {}
        header = mesh.Header.from_address(mesh.Context.from_address(context).M) if context is not None else None

        def paged(j):
            quantum = header.block * header.pgsz if header is not None else None
            return tuple(dict.fromkeys((port.root, quantum, -(-port.bytes // quantum)) for port in self.call_ports[j][0]
                                       if port.section[0] is not None))
        def erased(j):
            return tuple(sorted(op + self.rows[j].offset for ops in self.direct[j].values() for op in ops))
        self.intrinsic = tuple(frozenset(root for root, _, _ in paged(j)) for j in range(len(self.rows)))
        for j, program in enumerate(self.composites):
            if program is not None:
                sources.setdefault(program, {}).setdefault((paged(j), erased(j)), []).append(j)
        for k, body in ((k, body) for k, body in enumerate(self.bodies) if body.machine is not None):
            groups = sources.setdefault(body.machine.program, {})
            for j in (j for j, row in enumerate(self.rows) if row.function == k and self.composites[j] is None):
                groups.setdefault((self.call_ports[j][1], paged(j), erased(j)), []).append(j)
        for program, groups in sources.items():
            programs = [program]
            for _ in range(len(groups) - 1):
                copy = made(compiler.metal_ir_clone(program, error, ROOM), error, "a construction class")
                owned(lambda copy=copy: compiler.metal_ir_program_free(copy))
                programs.append(copy)
            classes.update((constructing, calls) for constructing, calls in zip(programs, groups.values()))
        gpu = device()
        try:
            for constructing, calls in classes.items():
                planned = (*self.call_ports[calls[0]][1], *(copy[4] for copy in self.rows[calls[0]].copies))
                for root in dict.fromkeys(planned):
                    compiler.metal_ir_value_planned(constructing, root, plan_calls[calls[0]], root, 0)
                for root, quantum, pages in paged(calls[0]):
                    compiler.metal_ir_value_paged(constructing, root, quantum, pages)
                for op in erased(calls[0]):
                    compiler.metal_ir_operation_erase(constructing, op)
                if self.bodies[self.rows[calls[0]].function].machine is not None:
                    self.related(constructing, calls[0], libs)
                checked(compiler.metal_ir_construct(constructing, gpu, error, ROOM), error, "construction")
                for j in calls:
                    templates[j] = constructing
        finally:
            M.PUT(gpu, M.SELECTOR[b"release"])
        self.templates = templates

    # metal-microbench metal_ir.h metal_ir_specialize, metal_ir_value_receives, metal_ir_value_sends,
    # metal_ir_value_planned and metal_ir_lower; rdma/mesh-metal.h mesh_metal_range_declare: each original call's
    # instance (`instances`, released with `owned`), its constructed program specialized, THAT call's Ports declared
    # (each received range of transfer `transfer` of its owner call landing for call j, among `plan_calls`, the calls as
    # the plan takes them, its backing bytes at its byte; `context` the compile owner's, read for its page geometry;
    # each canonical release once, {its representative transfer, owner call}, metal_ir_value_sends appending it; each
    # sent row's sending lifetime), each intrinsic wire's plan value (instance plan_calls[j], ValueID the wire's root;
    # placed from that section, marks["placed"], where no joint root binds it: Target.ported `joint`), then its
    # body's produced kernels' and Neural Engine bodies' terminal emissions (compiler/mesh_compiler/lower_metal.py
    # emitted: its ConstructedMachine's upstream and neural, each operation at its row's append offset, Row.offset: 0
    # where the row runs its body's own program; a copy-only row has no machine and none), then its structural
    # lowering (transport, checks, step, the arrays the plan reads); a call with no program on this node has no instance
    def instanced(self, context, plan_calls, libs, owned):
        from .lower_metal import emitted
        M, compiler, _, transport = libs
        error, listed = C.create_string_buffer(ROOM), (mesh.U * max(len(plan_calls), 1))(*plan_calls)
        instances = []
        for j, (row, template) in enumerate(zip(self.rows, self.templates)):
            if template is None:
                instances.append(None)
                continue
            instance = made(compiler.metal_ir_specialize(template, error, ROOM), error, "specialization")
            owned(lambda instance=instance: compiler.metal_ir_program_free(instance))
            (ports, wires), f, releases = self.call_ports[j], plan_calls[j], set()
            marks = {"received": [], "sent": [], "sending": {}, "transfers": [], "placed": []}
            for port in ports:
                if port.direction == mesh.RECEIVE:
                    landed = M.Range()
                    mesh.check(transport.mesh_metal_range_declare(mesh.Context.from_address(context), listed,
                                                                  len(plan_calls), j, port.owner, port.transfer,
                                                                  port.bytes, port.at, C.byref(landed)))
                    compiler.metal_ir_value_receives(instance, port.root, C.byref(landed))
                    marks["received"].append((port.root, *port.section, bytes(landed)))
                elif port.release is None or (port.owner, port.release) not in releases:
                    releases.add((port.owner, port.release))
                    release = M.Release(transfer=port.first, owner=port.owner)
                    compiler.metal_ir_value_sends(instance, port.root, C.byref(release))
                    marks["sent"].append((port.root, port.owner, bytes(release)))
                marks["transfers"].append((port.owner, port.transfer))
                marks["sending"][port.owner, port.transfer] = port.value
            for root in wires:
                compiler.metal_ir_value_planned(instance, root, f, root, 0)
                if (j, root) not in self.joint:
                    marks["placed"].append((root, f, root))
            for root in dict.fromkeys(copy[4] for copy in row.copies):
                compiler.metal_ir_value_planned(instance, root, f, root, 0)
            body = self.bodies[row.function].machine
            if body is not None:
                emitted(instance, body, row.offset)
            checked(compiler.metal_ir_lower(instance, error, ROOM), error, "lowering")
            row.marks = marks
            instances.append(instance)
        self.instances = instances

    # rdma/mesh-collective.h struct mesh_function, mesh_plan_local: each lowered call instance's mesh.Function (its
    # actual Storages, its row's explicit roots run 0, its step appended; its retained MetalRecorderActors parts and
    # relation, freed with the target), in instance order, a call with no program on this node zero-local (no values,
    # parts or edges), and the largest placed alignment
    def functions(self, libs):
        lib, held, quantum, counts, steps = libs[2][0], [], 1, [], []
        functions = (mesh.Function * len(self.instances))()
        for k, instance in enumerate(self.instances):
            if instance is None:
                counts, steps = counts + [0], steps + [False]
                continue
            values, count, stepped, aligned, _ = storages(instance, self.explicit(k), self.intrinsic[k], libs)
            solved = actors(instance, libs)
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
    # CONSTANT root its outer constant's name (`constants`, from the call's original input); each staging wire's
    # ValueID (Prepared.wired) placed at its joint wire root's copies, byte 0 its root's; every body ValueID a row's
    # program's (Row.values); and the outer outputs: one a land copy here writes (`landed`) that copy's row and outer
    # result root; a call's result produced here its row's mapped Output; one produced on another machine and landed by
    # none here its storage s, the caller-owned root it is returned in (CallAnalysis.results; root_sources[s], an outer
    # input's root byte 0 the placeholder's byte offset in s). The outer outputs are the output owner's alone: this
    # machine (`here`, compile_program's physical machine, not Target.node's plan index) when it is
    # CallAnalysis.output_machine (compiler/mesh_compiler/schedule.py, the outer output node's processor's machine);
    # another machine returns none (compiler/mesh_compiler/runtime.py Program.__call__: an entry other than the output
    # processor's returns nothing), whatever numerical bodies it holds
    def resolved(self, captured, analysis, joint, here):
        from .lower_metal import Output
        from .operations import _metadata
        planned = joint.planned
        self.rows = tuple(dataclasses.replace(row, ordinal=planned.call[j].ordinal, every=planned.call[j].every,
                                              planned=j, instance=planned.call[j].function)
                          for j, row in enumerate(self.rows))
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
        for (call, wire), root in joint.prepared.wired.items():
            if root != mesh.ABSENT:
                joined[call].append((root, wire, 0, planned.call[call].every))
        for j, node in enumerate(analysis.nodes):
            body = self.bodies[analysis.calls[j]]
            for root in body.state:
                bound(("generator", analysis.calls[j]), ("generator", analysis.calls[j]), 0,
                      (j, self.root_of(j, root.root)))
            for root, reference in zip(body.logical, captured.calls[node].inputs, strict=True):
                if root is not None and root.root is not None and root.role == metal.CONSTANT:
                    constants.append((j, self.root_of(j, root.root), reference.target))
        produced = {leaf: (j, o) for j, node in enumerate(analysis.nodes)
                    for o, leaf in enumerate(captured.calls[node].outputs) if isinstance(leaf, torch.fx.Node)}
        stored, outputs = {leaf: s for _, _, leaf, s in analysis.results}, []
        returned = zip(captured.exported.graph_module.graph.output_node().args[0], captured.exported.outputs,
                       strict=True) if here == analysis.output_machine else ()
        for position, (leaf, (kind, _)) in enumerate(returned):
            if position in self.landed:
                outputs.append(self.landed[position])
            elif leaf in produced:
                j, o = produced[leaf]
                body = self.bodies[self.rows[j].function]
                if body.machine is not None:
                    given = body.outputs[o]
                    outputs.append((j, given._replace(root=self.root_of(j, given.root),
                                                      value=self.view_of(j, given.value))))
                else:
                    source, start = joint.root_sources[stored[leaf]], 0
                    if source[0] == "input":
                        value = outer[source[1]].meta["val"]
                        start = value.storage_offset() * value.element_size()
                    outputs.append((("root", source, start),
                                    Output(None, None, None, _metadata(leaf.meta["val"]), kind, None)))
            elif isinstance(leaf, torch.fx.Node):
                outputs.append((None, Output(outer.index(leaf), None, None, _metadata(leaf.meta["val"]), kind, None)))
            else:
                outputs.append((None, Output(None, None, leaf, None, kind, None)))
        self.operands, self.joined, self.outputs = tuple(operands), tuple(map(tuple, joined)), tuple(outputs)
        self.constants = tuple(constants)

    # design/compilation-composition-change-request.md, 6's stage D, and design/aot-compilation-change-request.md
    # appendix item 6: the already-lowered call instances themselves, each its operands' ordinals set; after
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
            programs = self.instances
            for a, operand in enumerate(self.operands):
                for r, root in operand.occurrences:
                    mesh.check(compiler.metal_ir_value_external(programs[r], root, a))
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
                self.signature = mesh_graph.archive(signature)
            finally:
                mesh_graph.free(signature)
            self.artifact = M._saved_bytes(lib.MetalProgramSave, native, None, 0)
            owned.finalizer()
        return self

    # metal-microbench metal_ir.h metal_ir_archive_load (compiler/mesh_compiler/metal.py _archive_loaded: the archive
    # borrows and, loaded, owns the artifact's own bytes, its programs' relations in the joint owner's
    # `analysis_context`, the compilation's one), metal_ir_archive_program and metal_ir_loaded;
    # metal_recording.h MetalProgramLoad (given the archive handle, which it retains): the completed artifact, every
    # row's program in row order, relocated onto its run-time owner and that owner's own prepared solution
    # (bridge.Published: the network result it loaded once; or None: a new arena of the retained local solution), the
    # target itself unchanged, so independently prepared Bounds coexist, with no lowering, solve, finalization or
    # compilation; constants: {placeholder name: tensor} for the bodies' CONSTANT roots, attached as
    # compiler/mesh_compiler/metal.py _Builder.attach attached them and held as those roots' handles, by (row, root);
    # rdma/mesh_graph.py from_archive and free, metal_recording.h MetalProgramSignatureBind: the target's saved complete
    # signature archive read once (mesh_graph_archive_read) and bound to the loaded native program's own rows (its
    # per-row structure verified, its loaded programs retained by the signature's scope owner), freed with this
    # preparation, the Bound's `signature`
    def prepare(self, published, constants):
        libs = libraries()
        M, compiler, (lib, device), _ = libs
        error = C.create_string_buffer(ROOM)
        owned = Owned()
        with owned:
            gpu = device()
            owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
            archive = M._archive_loaded(compiler, self.artifact, self.solution.analysis_context, gpu, error)
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
            signature = mesh_graph.from_archive(self.signature)
            owned(lambda: mesh_graph.free(signature))
            checked(lib.MetalProgramSignatureBind(native, self.node, C.byref(signature), error, ROOM), error,
                    "the loaded signature")
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
        return Bound(self, owner, native, ended, generators, fixed, owned, libs, (uses, count.value), tuple(programs),
                     signature)

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
                            "slots": list(body.slots), "placeholders": [list(e) for e in body.placeholders],
                            "results": [list(e) for e in body.results]} for body in self.bodies],
                "rows": [{**{field.name: getattr(row, field.name) for field in dataclasses.fields(row)
                             if field.name != "marks"}, "values": list(row.values), "roots": list(row.roots),
                          "marks": {"received": [(v, f, i, d.hex()) for v, f, i, d in row.marks["received"]],
                                    "sent": [(v, o, d.hex()) for v, o, d in row.marks["sent"]],
                                    "sending": [[o, t, v] for (o, t), v in row.marks["sending"].items()],
                                    "transfers": [list(entry) for entry in row.marks["transfers"]],
                                    "placed": row.marks["placed"]}}
                         for row in self.rows],
                "operands": [{"source": source_saved(operand.source), "offset": operand.offset,
                              "occurrences": [list(at) for at in operand.occurrences]} for operand in self.operands],
                "outputs": [[["root", source_saved(r[1]), r[2]] if isinstance(r, tuple) else r,
                             leaf and leaf_saved(leaf)] for r, leaf in self.outputs],
                "joined": [[list(entry) for entry in row] for row in self.joined], "counts": list(self.counts),
                "steps": list(self.steps), "generators": [[k, list(words)] for k, words in self.generators],
                "constants": [list(entry) for entry in self.constants],
                "spec": pytree.treespec_dumps(self.spec), "node": self.node}

    # the artifact; the joint owner as it saves itself (compiler/mesh_compiler/schedule.py: a local Solved through
    # mesh_local_save, a network one through mesh_program_save, a restored SavedNetwork its original bytes), its kind
    # ("local", or "network" with its program) named in the companion record; the signature archive (one hex field) and
    # the recipe
    def saved(self, directory, key):
        directory, name = pathlib.Path(directory), f"{key}.native"
        (directory / name).write_bytes(self.artifact)
        (directory / f"{key}.joint").write_bytes(self.solution.saved())
        return {"artifact": name, "joint": f"{key}.joint", "kind": "local" if self.program is None else "network",
                "program": self.program, "signature": self.signature.hex(), "recipe": self.recipe()}


# a joint root's source record (schedule.Solved root_sources, Operand.source) as a saved record, a result's device by
# its name, and restored
def source_saved(source):
    return [*source[:3], str(source[3])] if source[0] == "result" else list(source)


def source_restored(saved):
    return (*saved[:3], torch.device(saved[3])) if saved[0] == "result" else tuple(saved)


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


# metal-microbench metal_ir.h metal_ir_work_terms, metal_ir_work_associations and metal_ir_dispatch_numerics: one loaded
# row program's retained Work (`count` terms and their factors, atoms, extents, axes and origins; its `associated`
# source associations, `associations`) and each of its dispatches' numeric scope (`numerics`, index-aligned with
# metal_ir_dispatches, `dispatches` of them), the tables borrowed from `program` (its handle, for identity), kept valid
# by `owner`, Target.prepare's Owned of the loaded programs (Bound.owned)
@dataclasses.dataclass(frozen=True)
class Work:
    owner: object
    program: object
    count: int
    terms: object
    factors: object
    atoms: object
    extents: object
    axes: object
    origins: object
    numerics: object
    dispatches: int
    associations: object
    associated: int


def borrowed(compiler, program, owner):
    count, dispatches, associated = C.c_size_t(), C.c_size_t(), C.c_size_t()
    tables = (C.POINTER(metal.WorkTerm)(), C.POINTER(metal.Expr)(), C.POINTER(metal.Term)(),
              C.POINTER(metal.WorkExtent)(), C.POINTER(metal.WorkAxis)(), C.POINTER(mesh.U)())
    status = compiler.metal_ir_work_terms(program, C.byref(count), *(C.byref(table) for table in tables))
    if status:
        raise OSError(status, "the retained Work of a loaded row program")
    numerics = compiler.metal_ir_dispatch_numerics(program, C.byref(dispatches))
    associations = compiler.metal_ir_work_associations(program, C.byref(associated))
    return Work(owner, program, count.value, *tables, numerics, dispatches.value, associations, associated.value)


# A prepared native program on its owner. A call binds its operands at the native program's fixed occurrences (each
# input or state backing the caller's storage; each fresh output root a result allocation of this call, which the
# program writes directly), runs one execution of every row, waits for its end, and returns its outputs as views of
# those backings or of a CONSTANT root's own handle (`fixed`, {(row, root): its tensor}). `uses` the program's
# borrowed MetalProgramExternals view (pointer, count), kept with the program that owns it. Prepared once from the
# target's operands and outputs, as the target states them, each call's sources one pool (given, allocated, fixed) by
# index: `allocations` each result backing a call allocates, (joint root s, its bytes, its device): Operand.source
# ("result", s, bytes, device) of an operand or of a supplied owner root's outer output; `template` the argument table,
# each generator operand's own buffer in it; its other operands' (position, outer input or joint root, root byte 0's
# offset): `hosted` and `shared` the outer inputs the program uses on the host only, or on the GPU (MetalExternalUse
# METAL_EXTERNAL_GPU), their backing the caller's; `buffers`, `hosts` and `pages` this call's allocations, on MPS
# (their own MTLBuffer), on the CPU used on the host only, or on the CPU used on the GPU (borrowed pages); `outputs` the
# outer outputs (Target.outputs) with each literal in place; `direct` each outer placeholder's (position, its input);
# `views` each view's (position, pool, key, dtype, shape, stride, element offset, `at`): of a supplied owner root (r
# ("root", its source record, the root byte its backing's byte 0 is)), a CONSTANT root's own handle, or a call's root
# through its operand's backing ({(row, root): operand} the operands' occurrences), its storage offset (source's byte
# start - `at`) // itemsize + element offset, `at` normalized per source (an allocation's 0, its root byte folded into
# the element offset; a handle's own byte start). `programs` each row's loaded program (Target.prepare's, None for a row
# without one), owned with `owned`; `work` each such row's Work (read by observe once); `signature` its loaded
# signature (mesh_graph.Signature) bound to this native program, freed with `owned`
class Bound:
    def __init__(self, target, owner, program, ended, generator, fixed, owned, libs, uses, programs, signature):
        self.target, self.owner, self.program, self.ended, self.generator = target, owner, program, ended, generator
        self.fixed, self.owned, self.libs, self.uses = fixed, owned, libs, uses
        self.programs, self.work = programs, None
        self.signature = signature
        sources = (*(operand.source for operand in target.operands),
                   *(r[1] for r, _ in target.outputs if isinstance(r, tuple)))
        self.allocations = tuple({source[1]: source[2:] for source in sources if source[0] == "result"}.items())
        gpu = frozenset(use.argument for use in uses[0][:uses[1]] if use.uses & METAL_EXTERNAL_GPU)
        mps = frozenset(s for s, (_, device) in self.allocations if device.type == "mps")
        made = {"generator": [], "input": [], "result": []}
        for a, operand in enumerate(target.operands):
            made[operand.source[0]].append((a, operand.source[1], operand.offset))
        self.count = len(target.operands)
        self.template = (Argument * max(self.count, 1))()
        for a, k, _ in made["generator"]:
            self.template[a] = Argument(generator[k], 0, None)
        self.hosted = tuple(e for e in made["input"] if e[0] not in gpu)
        self.shared = tuple(e for e in made["input"] if e[0] in gpu)
        self.buffers = tuple(e for e in made["result"] if e[1] in mps)
        self.hosts = tuple(e for e in made["result"] if e[1] not in mps and e[0] not in gpu)
        self.pages = tuple(e for e in made["result"] if e[1] not in mps and e[0] in gpu)
        argued = {at: operand for operand in target.operands for at in operand.occurrences}
        outputs, direct, views = [], [], []
        for position, (r, leaf) in enumerate(target.outputs):
            outputs.append(leaf.literal if leaf.root is None and not isinstance(r, tuple) else None)
            if r is None and leaf.root is not None:
                direct.append((position, leaf.root))
                continue
            if leaf.root is None and not isinstance(r, tuple):
                continue
            shape, stride, offset, dtype, _, _ = leaf.metadata
            if not isinstance(r, tuple) and (r, leaf.root) in fixed:
                held = fixed[r, leaf.root]
                views.append((position, 2, (r, leaf.root), dtype, shape, stride, offset,
                              held.storage_offset() * held.element_size()))
                continue
            source, at = (r[1], r[2]) if isinstance(r, tuple) else (argued[r, leaf.root].source,
                                                                     argued[r, leaf.root].offset)
            views.append((position, 1, source[1], dtype, shape, stride, offset - at // dtype.itemsize, 0)
                         if source[0] == "result" else (position, 0, source[1], dtype, shape, stride, offset, at))
        self.outputs, self.direct, self.views = tuple(outputs), tuple(direct), tuple(views)

    # metal-microbench metal_ir.h metal_ir_dispatches and metal_ir_kernel_devices (Core ML MLComputePlan): each loaded
    # row's NEURAL kernels once, by kernel: its row, call, body's function ordinal and {Neural Engine, CPU, GPU}
    # operation counts
    def placements(self):
        M, compiler = self.libs[0], self.libs[1]
        made = []
        for r, (row, program) in enumerate(zip(self.target.rows, self.programs, strict=True)):
            if program is None:
                continue
            dispatch = C.POINTER(M.Dispatch)()
            count = compiler.metal_ir_dispatches(program, C.byref(dispatch))
            for kernel in dict.fromkeys(dispatch[d].kernel for d in range(count)
                                        if dispatch[d].kind == M.METAL_IR_NEURAL):
                devices = (mesh.U * 3)()
                compiler.metal_ir_kernel_devices(program, kernel, devices)
                made.append({"row": r, "call": row.call, "function": row.function, "kernel": kernel,
                             "devices": dict(zip(("neural engine", "cpu", "gpu"), devices))})
        return made

    # metal_recording.h struct MetalArgument: root byte 0 of the caller's view (its byte start less the compiled view's
    # offset in its root); an MPS storage gives its own MTLBuffer
    @staticmethod
    def mapped(tensor, offset):
        return Argument(tensor.untyped_storage().data_ptr(),
                        (tensor.storage_offset() * tensor.element_size() - offset) % (1 << 64), None)

    # MTLDevice.h newBufferWithBytesNoCopy:length:options:deallocator: (an existing address span, page aligned) through
    # compiler/mesh_compiler/metal.py _placed_buffer: a CPU storage the program uses on the GPU, its host base and a
    # shared buffer borrowing that storage's pages (the system page size) and owning a reference to the storage until
    # its deallocator runs, one per storage in this call (`borrowed`, {storage address: (first page, buffer)}), released
    # after the call
    def paged(self, tensor, offset, borrowed):
        storage, base = tensor.untyped_storage(), tensor.data_ptr() - offset
        address = storage.data_ptr()
        if address not in borrowed:
            first = address - address % mmap.PAGESIZE
            last = -(-(address + storage.nbytes()) // mmap.PAGESIZE) * mmap.PAGESIZE
            borrowed[address] = first, self.libs[0]._placed_buffer(mesh.View(first, last - first), storage)
        first, buffer = borrowed[address]
        return Argument(buffer, (base - first) % (1 << 64), base)

    # the one invocation through `run`: metal_recording.h MetalProgramRunArguments (the native program's entry) or
    # MetalProgramRunObserved (its observation entry), with the same operands, results, lifetimes and end. Its first
    # execution is metal_recording.h MetalProgramIssued's count (metal_program.m MetalProgramExecute counts each
    # entered execution; MetalStepOrigin's failure before entry consumes none). A failed run's error is made at its
    # return, from its message and MetalProgramCheckFailure's record (both written within the run, metal_program.h
    # NativeChecked, and reset only by the next run's entry), before metal_recording.h MetalProgramEnd writes the same
    # message buffer. MetalProgramEnd ends the program's issued executions (its wait on `ended`, cancelling and draining
    # work that makes no progress, and the notifications MetalExternalRelease's operand release is among) while this
    # call's borrowed operands are still held; the run's error is raised, else the end's
    def invoked(self, run, given):
        M, _, (lib, _), _ = self.libs
        error = C.create_string_buffer(ROOM)
        execution = lib.MetalProgramIssued(self.program)
        allocated = {s: torch.empty(size, dtype=torch.uint8, device=device) for s, (size, device) in self.allocations}
        pools, borrowed = (given, allocated, self.fixed), {}
        try:
            table = type(self.template).from_buffer_copy(self.template)
            for a, i, offset in self.hosted:
                table[a] = self.mapped(given[i], offset) if given[i].device.type == "mps" else Argument(
                    None, 0, given[i].data_ptr() - offset)
            for a, i, offset in self.shared:
                table[a] = self.mapped(given[i], offset) if given[i].device.type == "mps" else self.paged(
                    given[i], offset, borrowed)
            for a, s, offset in self.buffers:
                table[a] = self.mapped(allocated[s], offset)
            for a, s, offset in self.hosts:
                table[a] = Argument(None, 0, allocated[s].data_ptr() - offset)
            for a, s, offset in self.pages:
                table[a] = self.paged(allocated[s], offset, borrowed)
            status = run(self.program, self.owner.context, execution, 1, self.ended, M.START, M.STARTED, table,
                         self.count, error, ROOM)
            failed = status and M._execution_error({"lib": lib, "program": self.program}, status,
                                                   f"execution {execution} of the program: {error.value.decode()}")
            end = lib.MetalProgramEnd(self.program, self.ended, error, ROOM)
            if status:
                raise failed
            checked(end, error, f"execution {execution} of the program's end")
        finally:
            for _, buffer in borrowed.values():
                M.PUT(buffer, M.SELECTOR[b"release"])
        made = list(self.outputs)
        for position, key in self.direct:
            made[position] = given[key]
        for position, pool, key, dtype, shape, stride, offset, at in self.views:
            tensor = pools[pool][key]
            made[position] = torch.empty(0, dtype=dtype, device=tensor.device).set_(
                tensor.untyped_storage(), (tensor.storage_offset() * tensor.element_size() - at) // dtype.itemsize
                + offset, shape, stride)
        return made

    def __call__(self, *given):
        return self.invoked(self.libs[2][0].MetalProgramRunArguments, given)

    # design/unequal-shards-change-request.md items 6, 7 and 14; design/aot-compilation-change-request.md A3: the
    # outputs and each occurrence slice executed in this invocation (metal_recording.h MetalProgramRunObserved,
    # MetalProgramUnits and MetalProgramObserved, bound by compiler/mesh_compiler/metal.py: one MetalObserved per slice,
    # unit u's residue e at its slice + e, every unit's `copies` of them), with that slice's own members and source
    # identities; `work` each loaded row's borrowed Work, read once
    def observe(self, *given):
        if self.work is None:
            self.work = tuple(None if program is None else borrowed(self.libs[1], program, self.owned)
                              for program in self.programs)
        lib, error = self.libs[2][0], C.create_string_buffer(ROOM)
        run, units, read = lib.MetalProgramRunObserved, lib.MetalProgramUnits, lib.MetalProgramObserved
        outputs = self.invoked(run, given)
        count, slices, members, identities = (mesh.U(), C.POINTER(metal.UnitSlice)(), C.POINTER(mesh.U)(),
                                              C.POINTER(mesh.Q)())
        unit = units(self.program, C.byref(count), C.byref(slices), C.byref(members), C.byref(identities))
        total = sum(unit[u].copies for u in range(count.value))
        observed = (metal.Observed * max(total, 1))()
        checked(read(self.program, observed, total, error, ROOM), error, "observation")
        made_ = []
        for u in range(count.value):
            for s in range(unit[u].slice, unit[u].slice + unit[u].copies):
                got, part = observed[s], slices[s]
                if got.executions:
                    made_.append(Observation(unit[u].kind, unit[u].call, unit[u].site, got.residue,
                                             tuple(members[part.member:part.member + part.count]),
                                             tuple(identities[part.identity:part.identity + part.identities]),
                                             got.executions, got.seconds))
        return outputs, made_


# The saved artifact, its joint owner and its typed recipe, as the frontend's Root and Output records; no construction
# is retained. A local owner is loaded (compiler/mesh_compiler/schedule.py loaded_program: never planned again); a
# network owner is its saved bytes and program (schedule.SavedNetwork), with no attachment, publication, reservation,
# solve or relocation here: its one preparation (bridge.Published, mesh_program_load) loads it. The target's `plan` is
# that owner's own (None for a local result)
def restore(saved, directory):
    from .lower_metal import Root
    from .schedule import loaded_program
    directory = pathlib.Path(directory)
    artifact, recipe = (directory / saved["artifact"]).read_bytes(), saved["recipe"]
    bodies = tuple(Body(
        k, None, {}, tuple(Root(*root) for root in body["inputs"]),
        tuple(leaf_restored(leaf) for leaf in body["outputs"]), tuple(Root(*root) for root in body["state"]),
        {tuple(task): tuple(ops) for task, ops in body["operations"]}, body["identity"],
        {name: (kind, torch.device(device)) for name, kind, device in body["targets"]}, tuple(body["slots"]),
        tuple(map(tuple, body["placeholders"])), tuple(map(tuple, body["results"])))
        for k, body in enumerate(recipe["bodies"]))
    rows = tuple(Row(**{**row, "values": tuple(row["values"]), "roots": tuple(row["roots"]),
                        "copies": tuple(map(tuple, row["copies"])), "marks": {
        "received": [(v, f, i, bytes.fromhex(d)) for v, f, i, d in row["marks"]["received"]],
        "sent": [(v, o, bytes.fromhex(d)) for v, o, d in row["marks"]["sent"]],
        "sending": {(o, t): v for o, t, v in row["marks"]["sending"]},
        "transfers": [tuple(entry) for entry in row["marks"]["transfers"]],
        "placed": [tuple(entry) for entry in row["marks"]["placed"]]}}) for row in recipe["rows"])
    operands = tuple(Operand(source_restored(operand["source"]), operand["offset"],
                             tuple(map(tuple, operand["occurrences"]))) for operand in recipe["operands"])
    # the loading compilation's one analysis context (mesh-graph.h mesh_graph_context_create), its joint owner's
    # `analysis_context`: every loaded relation and native artifact of it (mesh_local_load here; mesh_program_load at
    # each preparation, bridge.Published; metal_ir_archive_load, Target.prepare; archive.py load's source relations)
    context = mesh_graph.Context.create()
    record = (directory / saved["joint"]).read_bytes()
    if saved["kind"] == "network":
        from .schedule import SavedNetwork
        joint = SavedNetwork(record, saved["program"], analysis_context=context)
    else:
        joint = loaded_program(record, context)
    return Target(bodies, rows, pytree.treespec_loads(recipe["spec"]),
                  tuple((("root", source_restored(r[1]), r[2]) if isinstance(r, list) else r,
                         leaf and leaf_restored(leaf)) for r, leaf in recipe["outputs"]), operands,
                  tuple(tuple(map(tuple, row)) for row in recipe["joined"]), tuple(recipe["counts"]),
                  tuple(recipe["steps"]), tuple((k, tuple(words)) for k, words in recipe["generators"]),
                  tuple(map(tuple, recipe["constants"])), recipe["node"], joint, joint.plan, artifact=artifact,
                  signature=bytes.fromhex(saved["signature"]), program=saved["program"])


# design/aot-compilation-change-request.md A10 and appendix item 6; design/compilation-composition-change-request.md
# items 3, 4, 6 and 7; metal-microbench metal_ir.h metal_ir_construct, metal_ir_specialize and metal_ir_lower;
# compiler/mesh_compiler/schedule.py declare_program and solve_program: the captured program's one native machine
# program over every call of analysis.keys/calls. Each function with a body on this node (`machines[key]`; one without
# keeps its rows, with no numerical program) gives its interface; every call's original bindings, the `declarations`
# and the wires' roots go once to the declaration owner (declare_program, its Prepared); each call's stage and land
# copies on this machine composed with its body (Target.composed: metal_ir_append, then the copies) and the row maps
# and wires given back to that owner (Prepared.remap, Prepared.bind), whose ports and rows give each call its Ports
# here; each composite and each body constructed once per group of its calls' construction facts; each original call
# one instance (`plan_calls[j]`, here one per call), its group's program specialized, THAT call's Ports declared and
# structurally lowered; every instance's actual Actors and Storages in ONE joint solve over the same owner; then those
# lowered instances emitted, each with its operands' ordinals and the joint result's placements, as one finalized,
# signed and saved native program, the instances and construction copies released after. `node` the program's machine
# (Target.node: the plan's node map index under `planning`, the bridge's context where calls transfer; else the
# machine); `targets` the driver's authoritative selection {original processor: (selected kind, torch.device)}
# (compiler/mesh_compiler/driver.py compile_captured), retained as given for the compilation (`Target.targets`,
# released after it): a copy-only row's target is its CopyRecord processor's entry; `analysis_context` the
# compilation's one analysis context (driver.py's, every construction's), the joint result's own
# (schedule.solve_program, Solved.analysis_context)
def compile_program(captured, machines, declarations, analysis, node, planning=None, *, targets, analysis_context):
    from .schedule import declare_program, solve_program

    def built(machine):
        return ((tuple(machine.inputs), tuple(machine.outputs), tuple(machine.state), dict(machine.operations),
                 machine.identity, dict(machine.targets)) if machine is not None else ((), (), (), {}, 0, {}))
    bodies = tuple(Body(key, machine, declarations[key].wires(), *built(machine))
                   for key, machine in ((key, machines.get(key)) for key in analysis.keys))
    plan_calls = tuple(range(len(analysis.nodes)))
    target = Target(bodies, tuple(Row(j, analysis.calls[j], planned=j, instance=plan_calls[j])
                                  for j in range(len(analysis.nodes))), captured.exported.out_spec, node=node,
                    generators=tuple((k, tuple(body.machine.initial_words)) for k, body in enumerate(bodies)
                                     if body.state))
    target.targets = targets
    libs = libraries()
    M, _, (_, device), _ = libs
    # each body's Body.logical: its compiled inputs at their original captured function placeholders (Root.name the
    # placeholder's target, as schedule.declare_program's binding_of reads them), the rest None
    for body in bodies:
        names = [n.target for n in captured.functions[body.key].graph_module.graph.nodes if n.op == "placeholder"]
        logical = [None] * len(names)
        for root in body.inputs:
            logical[names.index(root.name)] = root
        body.logical = tuple(logical)
    for k in (k for k, body in enumerate(bodies) if body.machine is not None):
        target.interface(k, libs)
    bindings = target.bindings(captured, analysis)
    prepared = declare_program(analysis, bindings, declarations,
                               {key: {value: wire.root for value, wire in machine.wires.items()}
                                for key, machine in machines.items()}, planning,
                               input_parts={j: bodies[f].machine.input_parts for j, f in enumerate(analysis.calls)
                                            if bodies[f].machine is not None})
    here = planning.nodes[node] if planning is not None else node
    with Owned() as instances:
        target.composed(prepared, tuple(kind for kind, _ in captured.exported.outputs), here, libs, instances)
        context = C.addressof(planning.context) if planning is not None else None
        target.ported(prepared, declarations, analysis, plan_calls, here)
        target.constructed(context, plan_calls, libs, instances)
        target.instanced(context, plan_calls, libs, instances)
        functions, quantum = target.functions(libs)
        joint = solve_program(analysis, functions, bindings, quantum, planning, plan_calls=plan_calls,
                              declared=prepared, analysis_context=analysis_context)
        target.resolved(captured, analysis, joint, here)
        target.solution, target.plan = joint, joint.plan
        target.program = planning.program if planning is not None else None
        with Owned() as owned:
            gpu = device()
            owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
            target.compile(Arena(target, owned, M, gpu) if planning is None else Mapped(target))
            owned.finalizer()
        instances.finalizer()
    target.instances = target.templates = target.call_ports = target.targets = target.composites = None
    target.joint = target.intrinsic = target.direct = None
    target.landed = None
    return target

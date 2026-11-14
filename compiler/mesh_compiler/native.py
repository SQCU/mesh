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

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

ROOM = 4096


# metal_recording.h struct MetalArgument
class Argument(C.Structure):
    _fields_ = [("buffer", C.c_void_p), ("offset", mesh.Q), ("base", C.c_void_p)]


# metal_recording.h struct MetalExternalUse, METAL_EXTERNAL_GPU: an operand's uses in the completed program
class ExternalUse(C.Structure):
    _fields_ = [("argument", mesh.U), ("row", mesh.U), ("value", mesh.U), ("uses", mesh.U)]


METAL_EXTERNAL_GPU = 1


def libraries():
    import mesh_compile as M
    compiler, (lib, device) = M._compiler(), M._recorder()
    compiler.metal_ir_specialize.restype = C.c_void_p
    compiler.metal_ir_specialize.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
    compiler.metal_ir_value_layout.restype = mesh.U
    compiler.metal_ir_value_layout.argtypes = [C.c_void_p, mesh.U, C.POINTER(M.Term), C.c_size_t,
                                               C.POINTER(C.c_int64)]
    compiler.metal_ir_value_external.restype = C.c_int
    compiler.metal_ir_value_external.argtypes = [C.c_void_p, mesh.U, mesh.U]
    lib.MetalRecorderActors.restype = C.c_int
    lib.MetalRecorderActors.argtypes = [C.c_void_p, C.POINTER(mesh_graph.Machine), C.POINTER(mesh_graph.Machine),
                                        C.POINTER(M._Actors)]
    lib.MetalRecorderActorsFree.restype, lib.MetalRecorderActorsFree.argtypes = None, [C.POINTER(M._Actors)]
    lib.MetalProgramSignatureNode.restype = C.c_int
    lib.MetalProgramSignatureNode.argtypes = [C.c_void_p, mesh.U, C.c_size_t, C.POINTER(mesh.Q),
                                              C.POINTER(mesh_graph.Signature), C.c_char_p, C.c_size_t]
    lib.MetalProgramExternals.restype = C.POINTER(ExternalUse)
    lib.MetalProgramExternals.argtypes = [C.c_void_p, C.POINTER(mesh.U)]
    lib.MetalProgramRunArguments.restype = C.c_int
    lib.MetalProgramRunArguments.argtypes = [C.c_void_p, C.c_void_p, mesh.Q, mesh.Q, C.c_void_p, C.c_void_p,
                                             C.c_void_p, C.POINTER(Argument), mesh.U, C.c_char_p, C.c_size_t]
    return M, compiler, (lib, device), M._transport()


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


# metal-microbench metal_ir.h metal_ir_storages, metal_ir_value_info and metal_ir_step_state; rdma/mesh_compile.py
# _lifetimes and its step value: the lowered program's actual Storages as plan values, one per ValueID (run 1 a root its
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


# metal_recording.h MetalRecorderActors, MetalRecorderActorsFree; rdma/mesh_compile.py _actors: the lowered program's
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


# rdma/mesh-collective.h mesh_plan_local, mesh_local_save, mesh_local_load, struct mesh_local, mesh_local_free: the
# machine program's own storage and native call, solved once from its actual Actors and Storages (solve) or loaded as
# saved (loaded, no plan: the bytes lent to the loaded result, which restores the function's parts and relation), and
# retained as solved: the mesh.Local and the mesh.Function whose Value array it solved (with the Actors whose parts and
# relation it read, where solved here), freed by their own owners when the solution is collected;
# every reader takes its views in place. `count` the program's ValueIDs, `stepped` whether its step word follows them
class Solution:
    def __init__(self, function, local, count, stepped, *freed):
        self.function, self.local, self.count, self.stepped = function, local, count, stepped
        self.finalizer = weakref.finalize(self, *freed)

    @classmethod
    def solve(cls, program, explicit, libs):
        lib = libs[2][0]
        solved, local = actors(program, libs), mesh.Local()
        values, count, stepped, quantum, _ = storages(program, explicit, libs)
        function = (mesh.Function * 1)(mesh.Function(
            value=C.cast(values, C.POINTER(mesh.Value)), values=len(values), host=solved.host, work=solved.work,
            edge=solved.edge, parts=solved.parts, edges=solved.edges, relation=solved.relation))
        solution = cls(function, local, count, stepped, Solution.freed, lib, solved, local)
        solution.held = values, solved
        mesh.check(mesh.LIB.mesh_plan_local(function, quantum, C.byref(local)))
        return solution

    @classmethod
    def loaded(cls, data, stepped):
        import mesh_compile as M
        function, local = (mesh.Function * 1)(), mesh.Local()
        solution = cls(function, local, 0, stepped, mesh.LIB.mesh_local_free, C.byref(local))
        mesh.check(M._lent(data, lambda release, owner: mesh.LIB.mesh_local_load(
            data, len(data), release, owner, function, C.byref(local)), lambda status: status == 0))
        solution.count = local.values - stepped
        return solution

    @staticmethod
    def freed(lib, solved, local):
        mesh.LIB.mesh_local_free(C.byref(local))
        lib.MetalRecorderActorsFree(C.byref(solved))

    def saved(self, M):
        data, length = C.c_void_p(), mesh.Z()
        mesh.check(mesh.LIB.mesh_local_save(self.function, C.byref(self.local), C.byref(data), C.byref(length)))
        try:
            return C.string_at(data, length.value)
        finally:
            M._FREE(data)

    def copies(self, v):
        return viewed(self.local.first, C.cast(self.local.offset, C.c_void_p).value, v)

    def placements(self):
        value = self.function[0].value
        for v in range(self.count):
            if value[v].run:
                yield (v, *self.copies(v), value[v].shares)

    def stepped_copies(self):
        return (*self.copies(self.count), self.function[0].value[self.count].shares) if self.stepped else None


# One preparation of a local program on its solution: one arena allocation of the solution's bytes (rdma/mesh_compile.py
# _NEW_BUFFER, the shared Metal allocation) and the solution's own planned record; no transport, context or status
class Arena:
    context = None

    def __init__(self, solution, owned, M, gpu):
        error = C.create_string_buffer(ROOM)
        self.solution, self.planned = solution, solution.local.planned
        buffer = made(M._NEW_BUFFER(gpu, M._SEL(b"newBufferWithLength:options:"), max(solution.local.bytes, 1),
                                    M.SHARED), error, "the program's arena", errno.ENOMEM)
        owned(lambda: M.PUT(buffer, M.SELECTOR[b"release"]))
        self.buffer = buffer

    def bound(self, owned, M):
        return self.buffer

    def placements(self):
        return self.solution.placements()

    def stepped(self):
        return self.solution.stepped_copies()


# rdma/mesh-collective.h mesh_plan, mesh_plan_save; rdma/mesh-dataflow.h mesh_view_values; compiler/mesh_compiler/
# schedule.py solved: a machine with declared transfers, its one plan over the source declarations and the lowered
# program's actual Storages (appended after the logical wire values, each wire's interval its root's) and Actors (the
# function's parts), solved once on the compile context and retained with its inputs, its Planned and its mapped arena
# through compilation (its mesh_view_values `first` and `offset` the copies' views); `own` (ValueID, plan value) of the
# program's own storages, `step` the step word's plan value
class Transported:
    placements, stepped = staticmethod(lambda: ()), staticmethod(lambda: None)

    def __init__(self, target, declarations, planning, program, libs):
        from .ir import Plan
        from .schedule import solved
        lib = libs[2][0]
        self.actors = actors(program, libs)
        weakref.finalize(self, lib.MetalRecorderActorsFree, C.byref(self.actors))
        values, count, stepped, _, storage = storages(program, target.explicit(), libs)
        logical = [list(value) for value in declarations.values]
        for value, wire in target.machine.wires.items():
            logical[value][4:6] = storage[wire.root].begin, storage[wire.root].end
        self.own = tuple((v, len(logical) + k) for k, v in enumerate(v for v in range(count) if values[v].run))
        rows = [(values[v].section.bytes, values[v].kind, 0, values[v].state, values[v].begin, values[v].end,
                 values[v].lag, 1) for v, _ in self.own]
        self.step = len(logical) + len(rows) if stepped else None
        rows += [(8, mesh.EXECUTION, 0, 0, 0, 0, 1, 1)] if stepped else []
        plan = Plan(declarations.program, declarations.channels, declarations.declared, declarations.weights,
                    [*map(tuple, logical), *rows], declarations.dependences, declarations.outstanding,
                    declarations.tasks, declarations.before, declarations.senders, declarations.receivers,
                    declarations.resources, [], [], b"", declarations.machine, declarations.places, [])
        self.function, self.held = mesh.function(plan.declared, plan.weights, plan.values, plan.dependences,
                                                 plan.outstanding, plan.tasks, plan.before, plan.senders,
                                                 plan.receivers, plan.resources)
        f = self.function[0]
        f.host, f.work, f.edge, f.parts, f.edges, f.relation = (
            self.actors.host, self.actors.work, self.actors.edge, self.actors.parts, self.actors.edges,
            self.actors.relation)
        self.source, self.planned, view, self.first, self.offset = solved(plan, planning, self.function)
        weakref.finalize(self, mesh.LIB.mesh_planned_free, C.byref(self.planned))
        self.context = C.addressof(planning.context)
        self.arena = torch.frombuffer((C.c_byte * view.bytes).from_address(view.address), dtype=torch.uint8) if (
            view.bytes) else torch.empty(0, dtype=torch.uint8)

    def bound(self, owned, M):
        return planned_buffer(self, owned, M)

    def copies(self, index):
        return viewed(self.first, C.addressof(self.offset), index)


# rdma/mesh_compile.py _placed_buffer: a plan owner's mapped arena as the one Metal buffer every placement takes
def planned_buffer(owner, owned, M):
    buffer = M._placed_buffer(mesh.View(owner.arena.data_ptr(), owner.arena.numel()))
    owned(lambda: M.PUT(buffer, M.SELECTOR[b"release"]))
    return buffer


# compiler/mesh_compiler/lower_metal.py ConstructedMachine (frontend-owned): the machine's one constructed program and
# its typed roots; declared: compiler/mesh_compiler/schedule.py Declarations.wires(), channel -> (declared transfer,
# logical wire value, its logical bytes, declared sent view, declared landed view). `marks` are its wire roots'
# declared records: `received` (ValueID, function, value, declared mesh_metal_range), `sent` (ValueID, declared
# mesh_metal_release), `sending` {transfer: ValueID}, `transfers` receives then sends, `placed` (ValueID, function,
# value), one per wire; `own` (ValueID, plan value) of its own storages and `step` its step word's plan value, where the
# plan places them. The program's callable operands (metal_ir_value_external), in ordinal order: `arguments`, one per
# input or state root (its first placeholder's position, or None for the program's own generator state; that view's byte
# offset in its root; the root's ValueID), then `results`, one per fresh output root (its ValueID, its bytes, the device
# of its first output), allocated for each call. `node` the program's machine (its map index, mesh-graph.h struct
# mesh_graph_processor `index`: the plan's node map's with a plan, the source machine's without). `solution` the local
# solve, `plan` a transported machine's solved Plan,
# `generator` the generator state's initial words, `artifact` the completed native program's archive (MetalProgramSave),
# `signature` its MetalProgramSignature record
@dataclasses.dataclass
class Target:
    machine: object
    declared: dict
    inputs: tuple
    outputs: tuple
    spec: object
    state: tuple
    operations: dict
    identity: int
    targets: dict
    marks: dict = None
    arguments: tuple = ()
    results: tuple = ()
    own: tuple = ()
    step: int = None
    node: int = None
    solution: object = None
    plan: object = None
    generator: tuple = ()
    artifact: bytes = None
    signature: bytes = None

    def explicit(self):
        built = self.machine
        return {root for root in (*(r.root for r in (*self.inputs, *self.state, *built.received.values(),
                                                     *built.sent.values())), *(r for r, _, _ in self.results))
                if root is not None}

    # The owner supplies `context` (None: a local program), `planned`, its storage buffer (`bound`) and a local
    # solution's `placements()` and `stepped()`; a plan's owner (Transported, bridge.Published) `function` and
    # `copies(value)` (its mesh_view_values offsets in place), which place the wires, the own storages (`own`) and the
    # step word (`step`)
    def placed(self, owned, libs, gpu, owner, program, row, finalize):
        M, compiler, (lib, _), transport = libs
        error, planned, wires = C.create_string_buffer(ROOM), owner.planned, None
        if self.marks["transfers"]:
            context, wires, word = mesh.Context.from_address(owner.context), M.Transport(), mesh.Q()
            mesh.check(transport.mesh_metal_transport_create(C.byref(context), C.byref(planned), gpu, C.byref(wires)))
            owned(lambda: transport.mesh_metal_transport_destroy(C.byref(wires)))
            for value, function, index, declared in self.marks["received"]:
                landed = M.Range.from_buffer_copy(declared)
                field = (mesh.U * max(landed.partitions, 1))()
                mesh.check(transport.mesh_metal_range_prepare(
                    C.byref(context), C.byref(wires), C.byref(planned), owner.function[function].value[index].section,
                    field, landed.partitions, C.byref(landed)))
                compiler.metal_ir_value_received(program, value, C.byref(landed))
            for value, declared in self.marks["sent"]:
                release = M.Release.from_buffer_copy(declared)
                mesh.check(transport.mesh_metal_release_prepare(C.byref(context), C.byref(wires), C.byref(planned),
                                                                row, C.byref(release)))
                compiler.metal_ir_value_sent(program, value, C.byref(release))
            for transfer in self.marks["transfers"]:
                sending = M.Sending()
                mesh.check(transport.mesh_metal_sending_prepare(C.byref(context), C.byref(wires), C.byref(planned),
                                                                row, transfer, C.byref(sending)))
                if sending.sent.sends:
                    compiler.metal_ir_value_sending(program, self.marks["sending"][transfer], C.byref(sending))
            mesh.check(transport.mesh_metal_status_prepare(C.byref(context), C.byref(wires), planned.call[row].plan,
                                                           C.byref(word)))
            compiler.metal_ir_transport(program, word.value)
        emitted = compiler.metal_ir_finalize if finalize else compiler.metal_ir_loaded
        checked(emitted(program, error, ROOM), error, "emission" if finalize else "relocation")
        recorder = made(lib.MetalRecorderOpen(gpu, error, ROOM), error, "recorder", errno.ENODEV)
        owned(lambda: M.PUT(recorder, M.SELECTOR[b"release"]))
        ready = made(lib.MetalRecorderPrepare(recorder, program, error, ROOM), error, "preparation")
        owned(lambda: M.PUT(ready, M.SELECTOR[b"release"]))
        buffer = owner.bound(owned, M)
        for value, function, index in (*self.marks["placed"], *((v, 0, index) for v, index in self.own)):
            compiler.metal_ir_value_placed(program, value, buffer, *owner.copies(index), 0,
                                           owner.function[function].value[index].shares)
        for value, offsets, count, shares in owner.placements():
            compiler.metal_ir_value_placed(program, value, buffer, offsets, count, 0, shares)
        step = owner.stepped() if self.step is None else (
            *owner.copies(self.step), owner.function[0].value[self.step].shares)
        if step is not None:
            compiler.metal_ir_step_placed(program, buffer, *step)
        # metal_ir.h metal_ir_value_placed with no buffer: each callable operand's root one logical copy, shared by
        # every execution (a state or mutated input is the caller's one storage), its address the invocation's
        for root in (*(root for _, _, root in self.arguments), *(root for root, _, _ in self.results)):
            compiler.metal_ir_value_placed(program, root, None, None, 1, 0, 0)
        if wires is not None:
            M.STATE_BUFFER(recorder, M._SEL(b"state:buffer:status:"), ready, wires.state, word.value)
        checked(compiler.metal_ir_place(program, error, ROOM), error, "placed accesses")
        replay = M.Replay()
        checked(lib.MetalRecorderReplay(recorder, ready, C.byref(replay), error, ROOM), error, "replay")
        owned(lambda: lib.MetalReplayDestroy(C.byref(replay)))
        return recorder, (M.Call * 1)(M.Call(C.pointer(replay), row + 1, None, 0))

    # rdma/mesh-metal.h mesh_metal_range_declare; metal-microbench metal_ir.h metal_ir_value_receives,
    # metal_ir_value_sends, metal_ir_value_planned, metal_ir_value_layout, metal_ir_value_external and metal_ir_lower;
    # rdma/mesh_compile.py _programs (6's stage A): before the plan, from the declarations alone, the plan's declared
    # transfers this program receives and sends, in transfer order (`context` the compile owner's, read for its page
    # geometry), each wire's plan value, the callable operands (input and state roots, with their source views'
    # offsets, then fresh output roots) marked external, and the one lowering, whose actual Actors and Storages the one
    # plan solves. `calls` the machine's calls as mesh_plan takes them, `row` this program's call, `function` its plan
    # function; the program's own transfers land for itself
    def lowered(self, context=None, calls=(0,), row=0, function=0):
        libs = libraries()
        M, compiler, (lib, device), transport = libs
        error, built, declared = C.create_string_buffer(ROOM), self.machine, self.declared
        program, listed = built.program, (mesh.U * len(calls))(*calls)
        receives = sorted((declared[channel][0], channel) for channel in built.received if channel in declared)
        sends = sorted((declared[channel][0], channel) for channel in built.sent if channel in declared)
        marks = {"received": [], "sent": [], "sending": {}, "transfers": [], "placed": []}
        for transfer, channel in receives:
            root, (_, value, size, *_) = built.received[channel].root, declared[channel]
            landed = M.Range()
            mesh.check(transport.mesh_metal_range_declare(mesh.Context.from_address(context), listed, len(calls), row,
                                                          row, transfer, size, 0, C.byref(landed)))
            compiler.metal_ir_value_receives(program, root, C.byref(landed))
            marks["received"].append((root, function, value, bytes(landed)))
        for transfer, channel in sends:
            root, release = built.sent[channel].root, M.Release(transfer=transfer)
            compiler.metal_ir_value_sends(program, root, C.byref(release))
            marks["sent"].append((root, bytes(release)))
        for transfer, channel in receives + sends:
            marks["transfers"].append(transfer)
            marks["sending"][transfer] = built.wires[declared[channel][1]].root
        for value, wire in sorted(built.wires.items()):
            compiler.metal_ir_value_planned(program, wire.root, function, value, 0)
            marks["placed"].append((wire.root, function, value))
        widths, arguments, results, argued = {code: dtype.itemsize for dtype, code in M.SCALAR.items()}, [], [], set()
        for position, root in enumerate(self.inputs):
            if root.root is not None and root.role in (M.INPUT, M.STATE) and root.root not in argued:
                info, offset = M.ValueInfo(), C.c_int64()
                compiler.metal_ir_value_info(program, root.value, C.byref(info))
                compiler.metal_ir_value_layout(program, root.value, None, 0, C.byref(offset))
                arguments.append((position, offset.value * widths[info.scalar], root.root))
                argued.add(root.root)
        arguments += [(None, 0, root.root) for root in self.state]
        for leaf in self.outputs:
            info = M.ValueInfo()
            if leaf.root is not None and leaf.root not in argued:
                compiler.metal_ir_value_info(program, leaf.root, C.byref(info))
                if info.role == M.OUTPUT:
                    results.append((leaf.root, info.bytes, str(leaf.metadata[4])))
                    argued.add(leaf.root)
        for ordinal, root in enumerate([root for _, _, root in arguments] + [root for root, _, _ in results]):
            mesh.check(compiler.metal_ir_value_external(program, root, ordinal))
        gpu = device()
        try:
            checked(compiler.metal_ir_lower(program, gpu, error, ROOM), error, "lowering")
        finally:
            M.PUT(gpu, M.SELECTOR[b"release"])
        self.marks, self.arguments, self.results = marks, tuple(arguments), tuple(results)
        return self

    # rdma/mesh_compile.py _finalized and _replays (6's stages D and E): after the one plan, the lowered source
    # specialized and prepared on the compile owner's result, finalized, made native, signed over its warm-up and one
    # period as the native program chose them (metal_recording.h MetalProgramPhases), and saved
    def compile(self, owner, row=0):
        libs = libraries()
        M, compiler, (lib, device), _ = libs
        error = C.create_string_buffer(ROOM)
        with Owned() as owned:
            gpu = device()
            owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
            prepared = made(compiler.metal_ir_specialize(self.machine.program, error, ROOM), error, "specialization")
            owned(lambda: compiler.metal_ir_program_free(prepared))
            recorder, calls = self.placed(owned, libs, gpu, owner, prepared, row, True)
            native = made(lib.MetalRecorderProgram(recorder, calls, 1, C.byref(owner.planned), error, ROOM), error,
                          "the program's calls")
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

    # metal-microbench metal_ir.h metal_ir_archive_load (rdma/mesh_compile.py _archive_loaded: the archive borrows and,
    # loaded, owns the artifact's own bytes), metal_ir_archive_program and metal_ir_loaded; metal_recording.h
    # MetalProgramLoad (given the archive handle, which it retains): the completed artifact relocated onto its run-time
    # owner (bridge.Published, or None: a new arena of the retained local solution), with no lowering, solve,
    # finalization or compilation; constants: {placeholder name: tensor} for its CONSTANT roots, attached as
    # rdma/mesh_compile.py _Builder.attach attached them and held as those roots' handles
    def prepare(self, published, constants, row=0):
        libs = libraries()
        M, compiler, (lib, device), _ = libs
        error = C.create_string_buffer(ROOM)
        owned = Owned()
        with owned:
            gpu = device()
            owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
            archive = M._archive_loaded(compiler, self.artifact, gpu, error)
            owned(lambda: compiler.metal_ir_archive_free(archive))
            program = compiler.metal_ir_archive_program(archive, 0)
            owned(lambda: compiler.metal_ir_program_free(program))
            fixed = {}
            for root in (root for root in self.inputs if root.root is not None and root.role == M.CONSTANT):
                info = M.ValueInfo()
                compiler.metal_ir_value_info(program, root.root, C.byref(info))
                fixed.setdefault(root.root, constants[root.name])
                if not info.data:
                    storage = constants[root.name].untyped_storage()
                    buffer = storage.data_ptr() if storage.device.type == "mps" else None
                    compiler.metal_ir_value_data(program, root.root, M.OBJECT(buffer, M._SEL(b"contents")) if buffer
                                                 else storage.data_ptr(), storage.nbytes(), buffer, 0, None,
                                                 M.RELEASE())
            owner = published if published is not None else Arena(self.solution, owned, M, gpu)
            recorder, calls = self.placed(owned, libs, gpu, owner, program, row, False)
            native = made(lib.MetalProgramLoad(recorder, calls, 1, C.byref(owner.planned), archive, error, ROOM),
                          error, "the loaded program's calls")
            owned(lambda: M.PUT(native, M.SELECTOR[b"release"]))
            ended = made(M.OBJECT(gpu, M._SEL(b"newSharedEvent")), error, "the program's event", errno.ENOMEM)
            owned(lambda: M.PUT(ended, M.SELECTOR[b"release"]))
            owned(lambda: lib.MetalProgramEnd(native, ended, error, ROOM))
            generator = None
            if self.generator:
                words = (mesh.U * len(self.generator))(*self.generator)
                generator = made(M._NEW_BUFFER(gpu, M._SEL(b"newBufferWithLength:options:"), C.sizeof(words),
                                               M.SHARED), error, "the generator state", errno.ENOMEM)
                owned(lambda: M.PUT(generator, M.SELECTOR[b"release"]))
                C.memmove(M.OBJECT(generator, M._SEL(b"contents")), words, C.sizeof(words))
            count = mesh.U()
            uses = lib.MetalProgramExternals(native, C.byref(count))
        return Bound(self, owner, native, ended, generator, fixed, owned, libs, (uses, count.value))

    def recipe(self):
        outputs = [[*leaf[:3], leaf.metadata and [*leaf.metadata[:3], str(leaf.metadata[3]).removeprefix("torch."),
                                                  str(leaf.metadata[4]), leaf.metadata[5]], leaf.kind.name,
                    leaf.written] for leaf in self.outputs]
        return {"inputs": [list(root) for root in self.inputs], "outputs": outputs,
                "spec": pytree.treespec_dumps(self.spec), "state": [list(root) for root in self.state],
                "operations": [[list(task), list(ops)] for task, ops in self.operations.items()],
                "identity": self.identity, "targets": {name: [kind, str(device)]
                                                       for name, (kind, device) in self.targets.items()},
                "marks": {"received": [(v, f, i, d.hex()) for v, f, i, d in self.marks["received"]],
                          "sent": [(v, d.hex()) for v, d in self.marks["sent"]],
                          "sending": self.marks["sending"], "transfers": self.marks["transfers"],
                          "placed": self.marks["placed"]},
                "arguments": self.arguments, "results": self.results, "own": self.own, "step": self.step,
                "node": self.node, "generator": self.generator}

    # the artifact, the local solution as mesh_local_save writes it, and the typed recipe
    def saved(self, directory, key):
        directory, name = pathlib.Path(directory), f"{key}.native"
        (directory / name).write_bytes(self.artifact)
        made_ = {"artifact": name, "signature": self.signature.hex(), "recipe": self.recipe(), "local": None}
        if self.solution is not None:
            made_["local"] = {"solution": f"{key}.local", "stepped": self.solution.stepped}
            (directory / f"{key}.local").write_bytes(self.solution.saved(libraries()[0]))
        return made_


# A prepared native program on its owner. A call binds its operands at the native program's fixed occurrences (each
# input or state root the caller's storage; each fresh output root a result allocation of this call, which the program
# writes directly), runs one execution, waits for its end, and returns its outputs as views of those roots or of a
# CONSTANT root's own handle (`fixed`, {root: its tensor}). `uses` the program's borrowed MetalProgramExternals view
# (pointer, count), kept with the program that owns it; `gpu` the ordinals of the operands it uses on the GPU, read once
class Bound:
    def __init__(self, target, owner, program, ended, generator, fixed, owned, libs, uses):
        self.target, self.owner, self.program, self.ended, self.generator = target, owner, program, ended, generator
        self.fixed, self.owned, self.libs, self.executions, self.uses = fixed, owned, libs, 0, uses
        self.argued = {root: argument for argument, (_, _, root) in enumerate(target.arguments)}
        self.fresh = {root: k for k, (root, _, _) in enumerate(target.results)}
        self.gpu = frozenset(use.argument for use in uses[0][:uses[1]] if use.uses & METAL_EXTERNAL_GPU)

    # metal_recording.h struct MetalArgument; MTLDevice.h newBufferWithBytesNoCopy:length:options:deallocator: (an
    # existing address span, page aligned) through rdma/mesh_compile.py _placed_buffer: root byte 0 of the caller's
    # view (its byte start less the compiled view's offset in its root). An MPS storage gives its own MTLBuffer. A CPU
    # storage gives its host base, and where the program uses the operand on the GPU, a shared buffer borrowing that
    # storage's pages (the system page size) and owning a reference to the storage until its deallocator runs, one per
    # storage in this call (`borrowed`, {storage address: (first page, buffer)}), released after the call
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

    def result(self, leaf, given, allocated):
        if leaf.root is None:
            return leaf.literal
        shape, stride, offset, dtype, _, _ = leaf.metadata
        if leaf.root in self.fresh:
            return allocated[self.fresh[leaf.root]].view(dtype).as_strided(shape, stride, offset)
        if leaf.root in self.fixed:
            tensor, start = self.fixed[leaf.root], 0
        else:
            position, base, _ = self.target.arguments[self.argued[leaf.root]]
            tensor = given[position]
            start = tensor.storage_offset() * tensor.element_size() - base
        return torch.empty(0, dtype=dtype, device=tensor.device).set_(
            tensor.untyped_storage(), start // dtype.itemsize + offset, shape, stride)

    def __call__(self, *given):
        M, _, (lib, _), _ = self.libs
        error, execution = C.create_string_buffer(ROOM), self.executions
        allocated = [torch.empty(size, dtype=torch.uint8, device=device) for _, size, device in self.target.results]
        borrowed, results = {}, len(self.target.arguments)

        def ended():
            return M.VALUE(self.ended, M._SEL(b"signaledValue"))
        try:
            operands = [Argument(self.generator, 0, None) if position is None else
                        self.operand(given[position], offset, argument in self.gpu, borrowed)
                        for argument, (position, offset, _) in enumerate(self.target.arguments)]
            operands += [self.operand(tensor, 0, results + k in self.gpu, borrowed)
                         for k, tensor in enumerate(allocated)]
            table = (Argument * max(len(operands), 1))(*operands)
            status = lib.MetalProgramRunArguments(self.program, self.owner.context, execution, 1, self.ended, M.START,
                                                  M.STARTED, table, len(operands), error, ROOM)
            if status:
                raise M._execution_error({"lib": lib, "program": self.program}, status,
                                         f"execution {execution} of the program: {error.value.decode()}")
            self.executions = execution + 1
            M._wait(lambda: ended() > execution, ended, f"execution {execution} of the program: no end for "
                                                        "MESH_PLATFORM_NS")
        finally:
            for _, buffer in borrowed.values():
                M.PUT(buffer, M.SELECTOR[b"release"])
        return [self.result(leaf, given, allocated) for leaf in self.target.outputs]


# The saved artifact, its local solution (mesh_local_load: loaded, never planned again) and its typed recipe, as the
# frontend's Root and Output records; no construction is retained
def restore(saved, directory):
    from .lower_metal import Output, Root
    directory = pathlib.Path(directory)
    artifact, recipe, local = (directory / saved["artifact"]).read_bytes(), saved["recipe"], saved["local"]
    marks = recipe["marks"]
    return Target(None, {}, tuple(Root(*root) for root in recipe["inputs"]),
                  tuple(Output(*leaf[:3], leaf[3] and (tuple(leaf[3][0]), tuple(leaf[3][1]), leaf[3][2],
                                                       getattr(torch, leaf[3][3]), torch.device(leaf[3][4]),
                                                       leaf[3][5]), OutputKind[leaf[4]], leaf[5])
                        for leaf in recipe["outputs"]),
                  pytree.treespec_loads(recipe["spec"]), tuple(Root(*root) for root in recipe["state"]),
                  {tuple(task): tuple(ops) for task, ops in recipe["operations"]}, recipe["identity"],
                  {name: (kind, torch.device(device)) for name, (kind, device) in recipe["targets"].items()}, {
                      "received": [(v, f, i, bytes.fromhex(d)) for v, f, i, d in marks["received"]],
                      "sent": [(v, bytes.fromhex(d)) for v, d in marks["sent"]],
                      "sending": {int(t): v for t, v in marks["sending"].items()}, "transfers": marks["transfers"],
                      "placed": [tuple(entry) for entry in marks["placed"]]},
                  tuple(map(tuple, recipe["arguments"])), tuple(map(tuple, recipe["results"])),
                  tuple(map(tuple, recipe["own"])), recipe["step"], recipe["node"],
                  local and Solution.loaded((directory / local["solution"]).read_bytes(), local["stepped"]),
                  generator=tuple(recipe["generator"]), artifact=artifact,
                  signature=bytes.fromhex(saved["signature"]))


# design/compilation-composition-change-request.md items 3, 4, 6 and 7 (one construction, one lowering, one plan of its
# actual results, then the native artifact): the machine's one typed construction and its source declarations, its
# declared transport and one lowering, the one plan of the lowered program's actual Actors and Storages (mesh_plan over
# the declarations where this machine sends or receives, mesh_plan_local otherwise), then specialization, finalization
# and the saved native program on that plan's owner; `generator` the construction's mesh_compile.Generator, `node`
# the program's machine (Target.node)
def compile(machine, declarations, node, planning=None, generator=None):
    target = Target(machine, declarations.wires(), tuple(machine.inputs), tuple(machine.outputs), machine.spec,
                    tuple(machine.state), dict(machine.operations), machine.identity, dict(machine.targets), node=node,
                    generator=tuple(generator.state.tolist()) if machine.state else ())
    target.lowered(C.addressof(planning.context) if planning is not None else None)
    libs = libraries()
    M, _, (_, device), _ = libs
    if target.marks["transfers"]:
        owner = Transported(target, declarations, planning, machine.program, libs)
        target.plan, target.own, target.step = owner.source, owner.own, owner.step
        return target.compile(owner)
    target.solution = Solution.solve(machine.program, target.explicit(), libs)
    with Owned() as owned:
        gpu = device()
        owned(lambda: M.PUT(gpu, M.SELECTOR[b"release"]))
        target.compile(Arena(target.solution, owned, M, gpu))
        owned.finalizer()
    return target

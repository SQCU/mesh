import ctypes as C
import errno
import weakref

import mesh


def actors(program, libs):
    M, _, (lib, _), _ = libs
    machines, made = M._machines(), M._Actors()
    gpu, cpu = (C.byref(machines[M.GPU]), C.byref(machines[M.CPU])) if machines else (None, None)
    status = lib.MetalRecorderActors(program, gpu, cpu, C.byref(made))
    if status not in (0, errno.ENODATA):
        mesh.check(status)
    made.program, made.machines, made.libraries = program, machines, libs
    return made


def functions(programs, explicit, intrinsic, libs):
    result = (mesh.Function * len(programs))()
    result.programs, result.explicit, result.intrinsic, result.libraries = programs, explicit, intrinsic, libs
    result.storage, result.actors, result.counts, result.steps = [], [], [], []
    result.quantum = 1
    for k, program in enumerate(programs):
        if program is None:
            result.counts.append(0)
            result.steps.append(False)
            result.storage.append(None)
            result.actors.append(None)
            continue
        values = storages(program, explicit[k], intrinsic[k], libs)
        solved = actors(program, libs)
        weakref.finalize(result, libs[2][0].MetalRecorderActorsFree, C.byref(solved))
        result[k] = mesh.Function(value=C.cast(values, C.POINTER(mesh.Value)), values=len(values),
                                  host=solved.host, work=solved.work, edge=solved.edge, parts=solved.parts,
                                  edges=solved.edges, relation=solved.relation)
        result.storage.append(values)
        result.actors.append(solved)
        result.counts.append(values.count)
        result.steps.append(values.stepped)
        result.quantum = max(result.quantum, values.quantum)
    return result


# metal-microbench metal_ir.h metal_ir_storages, metal_ir_value_info and metal_ir_step_state: the lowered program's
# actual Storages as plan values, one per ValueID (run 1 a root its own storage holds: neither an explicit root (but
# an `intrinsic` one, an unbound intrinsic wire's typed root, Target.constructed, the program keeps no pages of:
# metal_ir_value_info `pages` 0, its flat-buffer requirement, one contiguous copy each in the joint arena,
# mesh-collective.c plan_block), nor fixed storage (a `buffer`: metal_ir_value_data's or metal_ir_value_storage's),
# nor the step; a CONSTANT or STATE root whose initial contents only `data` holds is planned, the recorder copying
# them into each placed copy, metal_capture.m MetalInitialContents), and its step word (8 bytes, executions, lag 1)
# where it has a step value (`word` METAL_IR_STEP, found in the same pass: metal_ir_step_value would construct one
# after lowering) and the host writes it; each value's `owned` its info's (its storage root's Value::owned,
# mesh-collective.h struct mesh_value `owned`); the quantum the largest placed alignment
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
                               end=stored.end, run=int(run), owned=info.owned,
                               kind=mesh.INVARIANT if info.role in (M.CONSTANT, M.STATE) else mesh.EXECUTION)
        quantum = max(quantum, stored.alignment) if run else quantum
    if stepped:
        values[count] = mesh.Value(section=mesh.Section(bytes=8), kind=mesh.EXECUTION, lag=1, run=1)
        quantum = max(quantum, 8)
    values.source = program
    values.explicit, values.intrinsic, values.libraries = explicit, intrinsic, libs
    values.infos, values.storage = infos, storage
    values.count, values.stepped, values.quantum = count, stepped, quantum
    return values



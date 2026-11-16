import ctypes as C

import mesh


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



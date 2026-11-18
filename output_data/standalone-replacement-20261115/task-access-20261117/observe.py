import ctypes as C
import gc
import hashlib
import json
import sys
from pathlib import Path
root = Path('/Users/mdot/dox/mesh')
sys.path.insert(0, str(root / 'rdma'))
import mesh_graph as G
import mesh

def produce(kind):
    nodes = mesh.Nodes(count=2, self=0)
    nodes.node[0], nodes.node[1] = (0, 1)
    nodes.capacity[0][1] = nodes.capacity[1][0] = 1000000000
    geometry = mesh.PlanGeometry(pgsz=16, block=1, rows=4096)
    geometry.arena[0], geometry.arena[1] = (mesh.Range(0, 2048), mesh.Range(2048, 2048))
    declared = (mesh.Declared * 2)(*(mesh.Declared(source=0, destination=1, identity=i, kind=kind, value=0, from_=mesh.strided(i * 64, 64, ()), to=mesh.strided(128 + i * 64, 64, ())) for i in range(2)))
    weights = (mesh.Weight * 2)(*((1, 1, 1, 1),) * 2)
    values = (mesh.Value * 1)(mesh.Value(section=mesh.Section(bytes=256), run=1, wire=1, kind=mesh.EXECUTION))
    functions = (mesh.Function * 1)(mesh.Function(declared=declared, weights=weights, value=values, count=2, values=1, outstanding=2))
    calls = (C.c_uint32 * 1)(0)
    outstanding = (C.c_uint64 * 1)(2)
    demands = (mesh.Demand * 2)()
    demand_of, release_of = ((C.c_uint32 * 2)(), (C.c_uint32 * 2)())
    grouped = (mesh.Grouped * 1)(mesh.Grouped(demand=demands, demand_of=demand_of, release_of=release_of))
    mesh.check(mesh.LIB.mesh_declared_group(declared, weights, 2, 2, 0, grouped))
    bounds = G.buffer_queries(functions, 1, (C.c_double * 1)(1), 1, (C.c_uint32 * 1)(0), (C.c_uint32 * 1)(0), (C.c_uint64 * 1)(1), 0, (C.c_size_t * 0)())
    analysis = C.POINTER(mesh.PlanAnalysis)()
    mesh.check(mesh.LIB.mesh_plan_program_analyze(C.byref(nodes), C.byref(geometry), 3, bounds.made, None, functions, 1, grouped, calls, 1, calls, 1, outstanding, None, None, 0, None, calls, None, 0, None, None, C.byref(analysis)))
    return (analysis, (nodes, geometry, declared, weights, values, functions, calls, outstanding, grouped, demands, demand_of, release_of, bounds))

def solve(analysis, owner, kind, lengths, access):
    function = mesh.Function.from_buffer_copy(owner[5][0])
    held = mesh.function_tasks(function, [(length, int(i != 0)) for i, length in enumerate(lengths)], (), (0, 0), (3, 3), 2)
    made = C.POINTER(mesh.PlanScheduleResult)()
    operands = (analysis.contents.flow[0].physical, C.byref(function))
    events = ((C.c_size_t * 7)(0, 1, 4, 4, 5, 8, 8), (C.c_uint32 * 8)(0, 1, 2, 3, 0, 1, 2, 3))
    operands += (len(access), access, *events)
    held += events
    mesh.check(mesh.LIB.mesh_plan_schedule(*operands, C.byref(made)))
    return (made, (function, held, access))

def saved(analysis, schedule):
    table = (C.POINTER(mesh.PlanScheduleResult) * 1)(schedule)
    compiled = C.POINTER(mesh.PlanCompilation)()
    mesh.check(mesh.LIB.mesh_plan_program_compile(analysis, table, None, C.byref(compiled)))
    data, length = (C.c_void_p(), C.c_size_t())
    mesh.check(mesh.LIB.mesh_plan_compilation_save(compiled, C.byref(data), C.byref(length)))
    result = C.string_at(data, length.value)
    C.CDLL(None).free(data)
    mesh.LIB.mesh_plan_compilation_free(compiled)
    return result

def loaded(data):
    context = G.Context.create()
    buffer = C.create_string_buffer(data)
    compiled = C.POINTER(mesh.PlanCompilation)()
    released = []
    callback = mesh.RELEASE(lambda _: released.append(True))
    mesh.check(mesh.LIB.mesh_plan_compilation_load(context.pointer, buffer, len(data), callback, None, C.byref(compiled)))
    return (compiled, (context, buffer, callback, released))
records = []
for kind in (mesh.PAGES, mesh.JOIN):
    analysis, owner = produce(kind)
    access = (G.Viewed * 8)(*(G.Viewed(key=i, offset=offset, element=size, vertex=task, write=write) for i in range(2) for offset, size, task, write in ((i * 64, 64, 0, 1), (128 + i * 64 + 16 * i, 16, 1, 0), (128 + i * 64 + 8, 16, 2, 0), (128 + i * 64, 64, 3, 0))))
    iterations = []
    for lengths in ((1, 2, 8, 1), (1, 8, 2, 1)):
        schedule, selected = solve(analysis, owner, kind, lengths, access)
        result = schedule.contents
        edges = set(zip(result.before[:result.dependencies], result.after[:result.dependencies]))
        dependencies = {}
        for transfer in range(2):
            pages = range(result.page_first[transfer], result.page_first[transfer + 1])
            for task, required in ((1, {transfer}), (2, {0, 1}), (3, {0, 1, 2, 3})):
                actual = {q for q, p in enumerate(pages) if (result.to[result.page_edge[p]], result.first + task) in edges}
                assert actual == (required if kind == mesh.PAGES else {0, 1, 2, 3}), (kind, transfer, task, actual)
                dependencies[f'{transfer}:{task}'] = sorted(actual)
        archive = saved(analysis, schedule)
        restored, owners = loaded(archive)
        kept = C.cast(C.cast(restored.contents.step[0].schedule, C.c_void_p).value, C.POINTER(mesh.PlanScheduleResult))
        mesh.LIB.mesh_plan_schedule_retain(kept)
        before = (kept.contents.page_first[:3], kept.contents.before[:kept.contents.dependencies])
        mesh.LIB.mesh_plan_compilation_free(restored)
        gc.collect()
        assert not owners[3]
        assert before == (kept.contents.page_first[:3], kept.contents.before[:kept.contents.dependencies])
        assert kept.contents.accesses == 8 and kept.contents.access_relation
        mesh.LIB.mesh_plan_schedule_free(kept)
        assert owners[3] == [True]
        iterations.append({'read_pages': dependencies, 'numerical_place': result.place[result.first:result.tasks], 'archive_bytes': len(archive), 'retained_loaded_schedule': True})
        mesh.LIB.mesh_plan_schedule_free(schedule)
    mesh.LIB.mesh_plan_analysis_free(analysis)
    records.append({'kind': kind, 'rounds': iterations})
for kind in (mesh.PAGES, mesh.JOIN):
    restored, owners = loaded(Path(__file__).with_name(f'previous-{kind}.bin').read_bytes())
    assert restored.contents.step[0].schedule.contents.accesses == 0
    mesh.LIB.mesh_plan_compilation_free(restored)
receipt = {'scope': 'canonical pure analysis/schedule/compilation/archive; no installation or numerical execution', 'provider': hashlib.sha256((root / 'rdma/libmesh.dylib').read_bytes()).hexdigest(), 'actual_previous_format_archives_loaded': True, 'records': records}
print(json.dumps(receipt, indent=2))

# design/synchronization-signature-change-request.md item 1; design/implementation-order-2026-11-09.md item 10;
# design/compiler-archive-assignment-2026-11-13.md: each completed entry's own record
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

KIND = {name: k for k, name in enumerate(mesh_graph.KINDS)}
PRIMITIVE = {name: k for k, name in enumerate(mesh_graph.PRIMITIVES)}
WAITED = "host_wait"


# compiler/mesh_compiler/lower_inductor.py body: each record's tasks are its emitted native effect sites, its
# dependences carry their execution delays, and a transfer's publication precedes its arrival wait only where both
# emitting records are present
def operands(scheduled, executions=1):
    plan = scheduled.plan
    machines = list(plan.nodes) if plan else list(dict.fromkeys(scheduled.machines.values()))
    records = {group: function.record for group, function in scheduled.functions.items()}
    first, places, objects, published, arrived, count = {}, {}, {}, {}, [], 0
    for processor, record in records.items():
        first[processor], seen = count, {}
        count += len(record["tasks"])
        for index, (kind, _) in enumerate(record["tasks"]):
            places[processor, index] = seen[kind] = seen.get(kind, -1) + 1
        places[processor] = {kind: last + 1 for kind, last in seen.items()}
        for _, primitive, task, key, _, _ in record["operations"]:
            role, transfer = key if isinstance(key, tuple) else (None, key)
            if role == "publication":
                published[transfer] = processor, first[processor] + task
            elif role == "arrival":
                arrived.append((transfer, processor, first[processor] + task))
            if key is not None and not isinstance(key, int):
                objects.setdefault((processor, key), mesh_graph.LOWERED + len(objects))
    processors, lane, threads = [], {}, dict.fromkeys(machines, 0)
    for processor, record in records.items():
        machine = scheduled.machines[processor]
        kinds = {kind for kind, *_ in record["tasks"] + record["operations"]} | {"host"}
        for kind in sorted(kinds, key=KIND.__getitem__):
            lane[processor, kind] = len(processors)
            processors.append(mesh_graph.Processor(KIND[kind], threads[machine] * mesh_graph.NODES
                                                   + machines.index(machine)))
        threads[machine] += 1
    across = [(published[transfer][1], vertex, 0) for transfer, processor, vertex in arrived
              if transfer in published and published[transfer][0] != processor]
    across += [(first[processor] + a, first[processor] + b, delay)
               for processor, record in records.items() for a, b, delay in record["dependences"]]
    dependences = [mesh_graph.Dependence(a, b, 0) for a, b in mesh_graph.unrolled(count, across, executions)]
    tasks, operations = [], []
    for execution in range(executions):
        base = execution * count
        for processor, record in records.items():
            start = base + first[processor]
            tasks += [mesh_graph.Task(lane[processor, kind],
                                      execution * places[processor][kind] + places[processor, index],
                                      first[processor] + index, execution)
                      for index, (kind, _) in enumerate(record["tasks"])]
            operations += [mesh_graph.Operation(lane[processor, kind], PRIMITIVE[primitive],
                                                mesh_graph.NONE if task is None else start + task,
                                                mesh_graph.NONE if key is None else objects.get((processor, key), key),
                                                execution, execution * every + ordinal)
                           for kind, primitive, task, key, every, ordinal in record["operations"]
                           if execution * every + ordinal > 0 or primitive != WAITED]
    return processors, tasks, dependences, operations


# rdma/mesh-graph.h mesh_graph_signature, mesh_graph_compose: the emitted host record's owned signature, composed
# with the plan's own signature (compiler/mesh_compiler/schedule.py solved), their processors united by identity
def signature(scheduled, executions=1):
    plan, held = scheduled.plan, []
    try:
        held.append(mesh_graph.build_signature(*operands(scheduled, executions)))
        if plan and plan.signature:
            held.insert(0, mesh_graph.from_archive(plan.signature))
            held.insert(0, mesh_graph.signature_span(held[0], executions))
            held.insert(0, mesh_graph.compose(held[0], held[-1]))
        return dict(mesh_graph.forms(held[0]), record=mesh_graph.record(held[0]))
    finally:
        for made in held:
            mesh_graph.free(made)

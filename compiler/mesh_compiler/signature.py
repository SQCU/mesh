# design/synchronization-signature-change-request.md item 1; design/implementation-order-2026-11-09.md item 10;
# design/compiler-archive-assignment-2026-11-13.md: each completed entry's own record
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

KIND = {name: k for k, name in enumerate(mesh_graph.KINDS)}
PRIMITIVE = {name: k for k, name in enumerate(mesh_graph.PRIMITIVES)}
SENT, RECEIVED = "send", "recv"


def operands(scheduled, executions=1):
    machines = list(dict.fromkeys(scheduled.machines.values()))
    carried = {channel: place for place, channel in enumerate(scheduled.plan.channels if scheduled.plan else ())}
    records = {group: function.record for group, function in scheduled.functions.items()}
    first, places, objects, ends, count = {}, {}, {}, {}, 0
    for processor, record in records.items():
        first[processor], seen = count, {}
        count += len(record["tasks"])
        for index, (kind, _) in enumerate(record["tasks"]):
            places[processor, index] = seen[kind] = seen.get(kind, -1) + 1
        places[processor] = {kind: last + 1 for kind, last in seen.items()}
        for _, primitive, task, key, _, _ in record["operations"]:
            if primitive in (SENT, RECEIVED):
                ends.setdefault(key, {})[primitive] = processor, first[processor] + task
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
    for channel in carried:
        machine = scheduled.machines[ends[channel][SENT][0]]
        if ("bridge", machine) not in lane:
            lane["bridge", machine] = len(processors)
            processors.append(mesh_graph.Processor(KIND["bridge"], machines.index(machine)))
    across = [(end[SENT][1], end[RECEIVED][1]) for _, end in sorted(ends.items())]
    across += [(first[processor] + a, first[processor] + b)
               for processor, record in records.items() for a, b in record["dependences"]]
    dependences = [mesh_graph.Dependence(a, b, 0)
                   for a, b in mesh_graph.unrolled(count, [(a, b, 0) for a, b in across], executions)]
    tasks, operations = [], []
    for execution in range(executions):
        base = execution * count
        for processor, record in records.items():
            start, host = base + first[processor], lane[processor, "host"]
            tasks += [mesh_graph.Task(lane[processor, kind],
                                      execution * places[processor][kind] + places[processor, index],
                                      first[processor] + index, execution)
                      for index, (kind, _) in enumerate(record["tasks"])]
            for kind, primitive, task, key, every, ordinal in record["operations"]:
                vertex = mesh_graph.NONE if task is None else start + task
                made = [(lane[processor, kind], primitive, mesh_graph.NONE if key is None else objects.get(
                    (processor, key), key), execution * every + ordinal)]
                if primitive == SENT and key in carried:
                    made = [(host, "host_publish", mesh_graph.PLAN_COUNT + carried[key], execution + 1),
                            (lane["bridge", scheduled.machines[processor]], "bridge_raise",
                             mesh_graph.PLAN_COUNT + carried[key], execution + 1)]
                elif primitive == SENT:
                    made = [(host, "host_signal", key, execution + 1)]
                elif primitive == RECEIVED:
                    made = [(host, "host_wait", mesh_graph.PLAN_COUNT + carried[key] if key in carried else key,
                             execution + 1)]
                operations += [mesh_graph.Operation(on, PRIMITIVE[name], vertex, thing, execution, value)
                               for on, name, thing, value in made]
    return processors, tasks, dependences, operations


def signature(scheduled, executions=1):
    return mesh_graph.signature(*operands(scheduled, executions))

# design/synchronization-signature-change-request.md items 1, 2 and 6; design/implementation-order-2026-11-09.md item 10
import pathlib
import sys

import torch

from .ir import RECV, SEND, channel_of

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

TO_COPY = torch.ops.aten._to_copy.default
HOST = torch.device("cpu")
KIND = {name: k for k, name in enumerate(mesh_graph.KINDS)}
PRIMITIVE = {name: k for k, name in enumerate(mesh_graph.PRIMITIVES)}
SENT, RECEIVED, COPIED = "send", "recv", "host_wait"


def ends(scheduled):
    found = {}
    for processor, module in scheduled.modules.items():
        for node in module.graph.nodes:
            if channel_of(node) is not None:
                found.setdefault(channel_of(node), {})[node.target] = processor
    return found


def sent_on(scheduled, end):
    same = scheduled.machines[end[SEND]] == scheduled.machines[end[RECV]]
    return scheduled.devices[end[RECV]] if same else HOST


def between(node, processor, scheduled, channels):
    if node.target is not TO_COPY or "device" not in node.kwargs:
        return False
    device, here = torch.device(node.kwargs["device"]), scheduled.devices[processor]
    if node.args[0].target is RECV:
        return device == here != sent_on(scheduled, channels[node.args[0].args[0]])
    return any(user.target is SEND and device == sent_on(scheduled, channels[user.args[1]]) != here
               for user in node.users)


# S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, section 4.1
def torch_run(module, processor, scheduled, channels):
    nodes = [n for n in module.graph.nodes if n.op == "call_function"]
    at = {n: k for k, n in enumerate(nodes)}
    kept = [channel_of(n) is not None or between(n, processor, scheduled, channels) for n in nodes]
    task = {v: t for t, v in enumerate(v for v, keep in enumerate(kept) if keep)}
    edges = [(at[u], at[n]) for n in nodes for u in n.all_input_nodes if u in at]
    operations = [(COPIED, t, None, 0, 0) if channel_of(nodes[v]) is None else
                  (SENT if nodes[v].target is SEND else RECEIVED, t, channel_of(nodes[v]), 0, 0)
                  for v, t in task.items()]
    return {"tasks": [("host", (nodes[v].name,)) for v in task],
            "dependences": [(task[a], task[b]) for a, b in mesh_graph.contract(kept, edges)],
            "operations": [("host", *operation) for operation in operations]}


# design/compilation-composition-change-request.md item 6 and requirement 7's table; rdma/mesh-metal4.m
def metal_run(function, module):
    program, named = function.program, {channel_of(n): n.name for n in module.graph.nodes if n.target is RECV}
    tasks, operations, read, holds, order = [], [], [], {}, {}
    for entry in program:
        if entry[0] == "commit":
            order[id(entry[1])] = len(order)
    for index, entry in enumerate(program):
        task, segment = len(tasks), entry[1]
        if entry[0] == "commit":
            tasks.append(("queue", tuple(dict.fromkeys(name for step in segment.steps for name in step.nodes))))
            holds.update((step.vertex, task) for step in segment.steps)
            read += [(view.storage, task) for step in segment.steps for view, write in step.accesses if not write]
            operations += [("queue", "fence_wait", task, ("fence", order[id(earlier)]), 1, 1)
                           for earlier, _ in segment.after.values()]
            operations += [("queue", "pass_barrier", task, None, 0, 0)] * sum(map(bool, segment.barriers))
            operations += [("queue", "fence_update", task, ("fence", order[id(segment)]), 1, 1)] * bool(segment.fence)
            operations += [("queue", "submit", task, None, 0, 1),
                           ("queue", "host_signal", task, ("event", order[id(segment)]), 1, 1)]
        elif entry[0] == "wait":
            after = program[index + 1][3] if index + 1 < len(program) else len(function.vertices) - 1
            operations += [("host", "host_wait", ("vertex", after), ("event", order[id(due)]), 1, 1) for due in segment]
        elif entry[0] is RECV:
            tasks.append(("host", (named[entry[2]],)))
            holds["storage", entry[1]] = task
            operations.append(("host", RECEIVED, task, entry[2], 0, 0))
        else:
            tasks.append(("host", function.vertices[entry[3]]))
            holds[entry[3]] = task
            operations.append(("host", SENT, task, entry[2], 0, 0))
    holds[len(function.vertices) - 1] = len(tasks)
    tasks.append(("host", function.vertices[-1]))
    classes = [holds[vertex] for vertex in range(len(function.vertices))]
    dependences = mesh_graph.quotient(classes, [(d.source, d.target) for d in function.dependences])
    dependences += dict.fromkeys((holds["storage", storage], task) for storage, task in read
                                 if ("storage", storage) in holds and holds["storage", storage] < task)
    return {"tasks": tasks, "dependences": dependences,
            "operations": [(kind, name, holds[task[1]] if isinstance(task, tuple) else task, key, every, ordinal)
                           for kind, name, task, key, every, ordinal in operations]}


def operands(scheduled, functions=None, executions=1):
    functions = functions or scheduled.modules
    channels, machines = ends(scheduled), list(dict.fromkeys(scheduled.machines.values()))
    carried = {channel: place for place, (channel, *_) in enumerate(scheduled.carried)}
    records, processors, lane, threads = {}, [], {}, dict.fromkeys(machines, 0)
    for processor in scheduled.devices:
        module = scheduled.modules[processor]
        function = functions.get(processor, module)
        ran = function if isinstance(function, torch.fx.GraphModule) else getattr(function, "module", module)
        records[processor] = metal_run(function, module) if hasattr(function, "program") else torch_run(
            ran, processor, scheduled, channels)
        machine = scheduled.machines[processor]
        kinds = {kind for kind, *_ in records[processor]["tasks"] + records[processor]["operations"]} | {"host"}
        for kind in sorted(kinds, key=KIND.__getitem__):
            lane[processor, kind] = len(processors)
            processors.append(mesh_graph.Processor(KIND[kind], threads[machine] * mesh_graph.NODES
                                                   + machines.index(machine)))
        threads[machine] += 1
    for channel in carried:
        machine = scheduled.machines[channels[channel][SEND]]
        if ("bridge", machine) not in lane:
            lane["bridge", machine] = len(processors)
            processors.append(mesh_graph.Processor(KIND["bridge"], machines.index(machine)))
    first, holder, objects, places = {}, {}, {}, {}
    for processor, record in records.items():
        first[processor], seen = sum(len(records[p]["tasks"]) for p in first), {}
        for index, (kind, names) in enumerate(record["tasks"]):
            holder.update(((processor, name), first[processor] + index) for name in names)
            places[processor, index] = seen[kind] = seen.get(kind, -1) + 1
        places[processor] = {kind: last + 1 for kind, last in seen.items()}
        for _, _, _, key, _, _ in record["operations"]:
            if key is not None and not isinstance(key, int):
                objects.setdefault((processor, key), mesh_graph.LOWERED + len(objects))
    count = sum(len(record["tasks"]) for record in records.values())
    named = {(processor, node.target, channel_of(node)): node.name
             for processor, module in scheduled.modules.items() for node in module.graph.nodes
             if channel_of(node) is not None}
    across = [(holder[end[SEND], named[end[SEND], SEND, channel]], holder[end[RECV], named[end[RECV], RECV, channel]])
              for channel, end in sorted(channels.items())]
    tasks, dependences, operations = [], [], []
    for execution in range(executions):
        base = execution * count
        for processor, record in records.items():
            start, host = base + first[processor], lane[processor, "host"]
            tasks += [mesh_graph.Task(lane[processor, kind],
                                      execution * places[processor][kind] + places[processor, index],
                                      first[processor] + index, execution)
                      for index, (kind, _) in enumerate(record["tasks"])]
            dependences += [mesh_graph.Dependence(start + a, start + b, 0) for a, b in record["dependences"]]
            for kind, primitive, task, key, every, ordinal in record["operations"]:
                vertex = mesh_graph.NONE if task is None else start + task
                made = [(lane[processor, kind], primitive, mesh_graph.NONE if key is None else objects.get(
                    (processor, key), key), execution * every + ordinal)]
                if primitive == SENT and key in carried:
                    made = [(host, "host_publish", mesh_graph.PLAN_SUBMISSION + carried[key], execution + 1),
                            (host, "bridge_start", mesh_graph.PLAN_SUBMISSION + carried[key], execution + 1),
                            (lane["bridge", scheduled.machines[processor]], "bridge_raise",
                             mesh_graph.PLAN_COUNT + carried[key], execution + 1)]
                elif primitive == SENT:
                    made = [(host, "host_signal", key, execution + 1)]
                elif primitive == RECEIVED:
                    made = [(host, "host_wait", mesh_graph.PLAN_COUNT + carried[key] if key in carried else key,
                             execution + 1)]
                operations += [mesh_graph.Operation(on, PRIMITIVE[name], vertex, thing, execution, value)
                               for on, name, thing, value in made]
        dependences += [mesh_graph.Dependence(base + a, base + b, 0) for a, b in across]
    return processors, tasks, dependences, operations


def signature(scheduled, functions=None, executions=1):
    return mesh_graph.signature(*operands(scheduled, functions, executions))

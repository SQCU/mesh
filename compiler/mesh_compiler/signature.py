# design/synchronization-signature-change-request.md item 1; design/implementation-order-2026-11-09.md item 10
import pathlib
import sys

import torch

from .ir import RECV, SEND, call, channel_of

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

TO_COPY = torch.ops.aten._to_copy.default
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


# compiler/mesh_compiler/devices.py copied
def between(node):
    return node.target is TO_COPY and node.meta.get("custom", {}).get("between", False)


# S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, section 4.1
def torch_run(module):
    nodes = [n for n in module.graph.nodes if n.op == "call_function"]
    at = {n: k for k, n in enumerate(nodes)}
    kept = [channel_of(n) is not None or between(n) for n in nodes]
    task = {v: t for t, v in enumerate(v for v, keep in enumerate(kept) if keep)}
    edges = [(at[u], at[n]) for n in nodes for u in n.all_input_nodes if u in at]
    operations = [(COPIED, t, None, 0, 0) if channel_of(nodes[v]) is None else
                  (SENT if nodes[v].target is SEND else RECEIVED, t, channel_of(nodes[v]), 0, 0)
                  for v, t in task.items()]
    return {"tasks": [("host", (nodes[v].name,)) for v in task],
            "dependences": [(task[a], task[b]) for a, b in mesh_graph.contract(kept, edges)],
            "operations": [("host", *operation) for operation in operations]}


# compiler/mesh_compiler/runtime.py Program; compiler/mesh_compiler/bridge.py Published
def handed(records, scheduled):
    order, made, across = list(scheduled.devices), {}, []
    for place, machine in enumerate(dict.fromkeys(scheduled.machines.values())):
        caller, others, joined, started = call(scheduled, machine)
        finished = scheduled.channels + len(order) + place
        for processor in (p for p in scheduled.inputs if p == caller or p in others):
            record, queue = records[processor], scheduled.channels + order.index(processor)
            last = len(record["tasks"]) + 1
            first = [("host", "host_wait", 0, queue, 1, 1)]
            after = [("host", "host_signal", last, finished, len(joined), k + 1)
                     for k, other in enumerate(joined) if other == processor]
            if processor == caller:
                first = [("host", "bridge_start", 0, mesh_graph.PLAN_SUBMISSION, 1, 1)] * started
                first += [("host", "host_signal", 0, scheduled.channels + order.index(p), 1, 1) for p in others]
                after = [("host", "host_wait", last, finished, len(joined), k + 1) for k in range(len(joined))]
            made[processor] = {
                "tasks": [("host", ("<arguments>",))] + record["tasks"] + [("host", ("<finished>",))] * bool(joined),
                "dependences": [(a + 1, b + 1) for a, b in record["dependences"]],
                "operations": first + [(kind, name, task if task is None else task + 1, *rest)
                                       for kind, name, task, *rest in record["operations"]] + after}
        across += [((caller, "<arguments>"), (p, "<arguments>")) for p in others]
        across += [((p, "<finished>"), (caller, "<finished>")) for p in joined]
    return made, across


def operands(scheduled, functions=None, executions=1, stored=()):
    functions = functions or scheduled.modules
    channels, machines = ends(scheduled), list(dict.fromkeys(scheduled.machines.values()))
    carried = {channel: place for place, channel in enumerate(scheduled.plan.channels if scheduled.plan else ())}
    records, processors, lane, threads = {}, [], {}, dict.fromkeys(machines, 0)
    for processor in (p for p in scheduled.devices if p in scheduled.modules):
        module = scheduled.modules[processor]
        function = functions.get(processor, module)
        records[processor] = function.record if hasattr(function, "record") else torch_run(
            function if isinstance(function, torch.fx.GraphModule) else module)
    records, handoffs = handed(records, scheduled)
    for processor in records:
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
              for channel, end in sorted(channels.items())] + [(holder[a], holder[b]) for a, b in handoffs]
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


def signature(scheduled, functions=None, executions=1, stored=()):
    return mesh_graph.signature(*operands(scheduled, functions, executions, stored))

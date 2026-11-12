# design/data-flow.md; design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5
import ctypes as C
import mmap
import operator
import os
import pathlib
import sys
import time

import torch

from torch.fx.node import map_arg

from .ir import RECV, SEND, Plan, channel_of

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

SYSTEM = C.CDLL("/usr/lib/libSystem.B.dylib")
SYSTEM.free.argtypes = [C.c_void_p]
PRODUCED, READ, CONSUMED = ((mesh.FROM_PRODUCED, mesh.TO_PRODUCED), (mesh.FROM_SOURCE_READ, mesh.TO_SOURCE_READ),
                            (0, 0))


def unit_cost(task):
    return 1.0


def producer(node):
    return mesh_graph.chain(node, lambda n: n.args[0] if n.target is operator.getitem else None)[-1]


# compiler/mesh_compiler/placement.py based, views
def based(node):
    record = node.meta.get("view") or node.meta.get("out")
    return record[0] if record else node.name


def wrote(node):
    return [node.meta[key] for key in ("out", "to") if key in node.meta]


def whole(node):
    value = node.meta.get("val")
    return (node.name, 0, tuple(value.shape), tuple(value.stride())) if isinstance(value, torch.Tensor) else None


# design/problems-of-14b9a757-change-request.md items 1, 2 and 6; rdma/mesh-graph.h struct mesh_graph_view
def accesses(functions):
    named, key = {n.name: n for n in functions}, {record[0]: None for n in functions for record in wrote(n)}
    key.update((name, place) for place, name in enumerate(key))

    def access(record, element, vertex, write):
        name, offset, sizes, strides = record if min(record[3], default=0) >= 0 else whole(named[record[0]])
        return (key[name], offset * element, element, [(n, s * element) for n, s in zip(sizes, strides)], vertex, write)

    found = []
    for vertex, node in enumerate(functions):
        found += [access(record, node.meta["val"].dtype.itemsize, vertex, 1) for record in wrote(node)]
        for held in () if "view" in node.meta else dict.fromkeys(node.all_input_nodes):
            record = held.meta.get("view") or held.meta.get("out") or whole(held)
            if record and record[0] in key and record not in wrote(node):
                found.append(access(record, held.meta["val"].dtype.itemsize, vertex, 0))
    return found


# design/problems-of-14b9a757-change-request.md items 1, 2 and 6; A. J. Bernstein, IEEE Trans. Electronic Computers
# EC-15(5), 1966
def dependence(module):
    nodes = list(module.graph.nodes)
    functions = [n for n in nodes if n.op == "call_function" and producer(n) is n]
    sender = {channel_of(n): n for n in functions if n.target is SEND}
    after = {}
    for earlier, later in mesh_graph.dependences_views(len(functions), accesses(functions))[0]:
        after.setdefault(functions[later], []).append(functions[earlier])

    def preds(node):
        found = [producer(u) for u in node.all_input_nodes if u.op == "call_function"] + after.get(node, [])
        return found + [sender[channel_of(node)]] if node.target is RECV else found


    return nodes, functions, preds


def channel_ends(nodes):
    ends = {}
    for node in nodes:
        if channel_of(node) is not None:
            ends.setdefault(channel_of(node), {})[node.target] = node
    return ends


def carried(module, machines):
    found = []
    for channel, end in sorted(channel_ends(module.graph.nodes).items()):
        sender, receiver = end[SEND].meta["processor"], end[RECV].meta["processor"]
        if machines[sender] != machines[receiver]:
            value = end[RECV].meta["val"]
            found.append((channel, sender, receiver, value.dtype, list(value.shape), list(value.stride())))
    return found


# rdma/mesh-dataflow.h mesh_view_whole
def image():
    live = mesh.Context()
    mesh.check(mesh.LIB.mesh_attach(C.byref(live), None))
    header = mesh.Header.from_address(live.M)
    private = mmap.mmap(-1, header.length)
    address = C.addressof(C.c_char.from_buffer(private))
    while True:
        before = header.view_map
        C.memmove(address, live.M, header.data_off)
        if mesh.LIB.mesh_view_whole(live.M, before):
            break
    C.memset(address + header.algo_off, 0, C.sizeof(mesh.Algo))
    made = mesh.Context(M=address, len=header.length, client=live.client, fd=-1, start=live.start)
    mesh.check(mesh.LIB.mesh_detach(C.byref(live)))
    return private, made


# design/disjoint-pages-change-request.md C3.3 and C3.5; rdma/mesh-collective.h struct mesh_dependence
def dependences(events, before):
    found = mesh_graph.precedence([nodes for _, _, _, nodes in events], [place for _, place, _, _ in events],
                                  [nodes[0].meta["processor"] for _, _, _, nodes in events], before)
    return [(events[a][0], events[b][0], lag, events[a][2][0] | events[b][2][1]) for a, b, lag in found]


# compiler/mesh_compiler/placement.py views; design/problems-of-14b9a757-change-request.md items 3 and 6, and item 8,
# step 4
def viewed(sender, carried):
    def of(node):
        if node.target is RECV:
            return None if node.args[0] in carried else sender[node.args[0]].args[0]
        if "view" in node.meta:
            return None if "invariant" in node.meta else producer(node).args[0]
        return next((u for u in node.all_input_nodes if "out" in node.meta and based(u) == node.meta["out"][0]), None)
    return of


def extent(value):
    return value.dtype.itemsize * (1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))) * all(value.shape)


# rdma/mesh-collective.h mesh_plan, mesh_plan_save
def solved(plan, private, context):
    function, _ = mesh.function(plan.declared, plan.weights, plan.values, plan.dependences, plan.outstanding,
                                plan.tasks, plan.before, plan.senders, plan.receivers, plan.resources)
    calls, made, record, length = (mesh.U * 1)(0), mesh.Planned(), C.c_void_p(), C.c_size_t()
    operands = (plan.program, function, 1, calls, 1, None, None, 0, None)
    mesh.check(mesh.LIB.mesh_plan(C.byref(context), int(os.environ["MESH_GROUP"]), *operands,
                                  time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS, C.byref(made)))
    mesh.check(mesh.LIB.mesh_plan_save(C.byref(context), *operands, C.byref(made), C.byref(record), C.byref(length)))
    plan.completion = list(function[0].completion[:len(plan.tasks)])
    plan.sections = [[getattr(value.section, name) for name, _ in mesh.Section._fields_]
                     for value in function[0].value[:len(plan.values)]]
    plan.record = C.string_at(record, length.value)
    mesh.LIB.mesh_planned_free(C.byref(made))
    SYSTEM.free(record)
    return plan


# design/disjoint-pages-change-request.md A3 item 2 and C3.2; design/problems-of-14b9a757-change-request.md 7.1 steps 2
# and 5, 7.2 step 3, item 8 step 4 and item 14 steps 3 and 4
def stored(values, places, module, local, before, of, channels, mutated):
    index = {node: place for place, node in enumerate(local)}
    results = {mesh_graph.chain(n, of)[-1] for n in module.graph.output_node().all_input_nodes}
    roots = {f: [mesh_graph.chain(u, of)[-1] for u in f.all_input_nodes] for f in local}
    last = mesh_graph.ends(local, before, lambda f: () if f.meta.get("invariant") else [
        (root, (f if root.op == "placeholder" else root).meta["processor"]) for root in roots[f]] + [
        (f, f.meta["processor"])] * (f.target is RECV and f.args[0] in channels))
    named, slices, first = {(node.name, node.meta["processor"]): node for node in local}, set(), {}
    for node in reversed(local):
        first.update((record[0], index[node]) for record in wrote(node))
    for value, (processor, name) in enumerate(list(places)):
        node = named[name, processor]
        begin, end = index[producer(node)], len(local) - 1
        if node.target is RECV:
            begin, end = 0, end if node in results else index[last[node, processor]]
        values[value] = (*values[value][:4], begin, end, *values[value][6:])
    for (root, processor), reader in last.items():
        value = root.meta.get("val")
        if ((processor, root.name) in places or root in results or not isinstance(value, torch.Tensor)
                or not extent(value) or root.meta.get("state") in mutated):
            continue
        if "state" in root.meta or "invariant" in root.meta:
            if root.meta.get("invariant", root.meta.get("state")) in slices:
                continue
            slices.add(root.meta.get("invariant", root.meta.get("state")))
            values.append((extent(value), mesh.INVARIANT, 0, 0, 0, 0, 0, 1))
        elif producer(root) in index:
            values.append((extent(value), mesh.EXECUTION, 0, 0, first.get(root.name, index[producer(root)]),
                           index[reader], 0, 1))
        else:
            continue
        places.append((processor, root.name))


# design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5;
# design/integration-finishing-change-request.md item 3; rdma/mesh-collective.h mesh_plan
def planned(module, carried, machines, mutated, cost, functions, before, tasks, edges):
    private, context = image()
    nodes = mesh.Nodes()
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
    at, task = {nodes.node[i]: i for i in range(nodes.count)}, {node: k for k, node in enumerate(functions)}
    ends, held, channels = channel_ends(functions), set(functions), {channel for channel, *_ in carried}
    of = viewed({channel: end[SEND] for channel, end in ends.items()}, channels)
    results = {mesh_graph.chain(n, of)[-1] for n in module.graph.output_node().all_input_nodes}
    declared, weights, values, places, events = [], [], [], [], []
    for transfer, (channel, sender, receiver, dtype, shape, stride) in enumerate(carried):
        sent, arrival = ends[channel][SEND], ends[channel][RECV]
        source, destination = at[machines[sender]], at[machines[receiver]]
        written, value = producer(sent.args[0]), mesh.ABSENT
        own = sent.args[0] in held and of(sent.args[0]) is None and sent.args[0] not in results
        writer = (sender, (sent.args[0] if own else sent).name) if source == nodes.self else (receiver, arrival.name)
        if nodes.self in (source, destination):
            if writer not in places:
                places.append(writer)
                values.append((extent(arrival.meta["val"]), mesh.EXECUTION, 1, 0, 0, 0, 0, 1))
            value = places.index(writer)
        layout = (0, dtype.itemsize, [x for n, s in zip(shape, stride) for x in (n if s else 1, s * dtype.itemsize)])
        declared.append((source, destination, channel, arrival.meta["consumer"], value, layout))
        weights.append((cost(written if written in held else sent), cost(arrival), cost(sent), cost((sent, arrival))))
        read = [sent] + [u for u in sent.args[0].users if u in held and u is not sent]
        events += [(transfer, source, PRODUCED, [written])] * (written in held)
        events += [(transfer, source, READ, read), (transfer, destination, CONSUMED,
                                                    [arrival] + [u for u in arrival.users if u in held])]
    stored(values, places, module, [n for n in functions if machines[n.meta["processor"]] == nodes.node[nodes.self]],
           before, of, channels, mutated)
    crossing = {(task[ends[channel][SEND]], task[ends[channel][RECV]]) for channel in channels}
    plan = Plan(sum(1 << int(node) for node in os.environ["MESH_PROGRAM"].split(",")), [c for c, *_ in carried],
                declared, weights, values, dependences(events, before), 1, tasks,
                [edge for edge in edges if edge not in crossing], [task[ends[c][SEND]] for c, *_ in carried],
                [task[ends[c][RECV]] for c, *_ in carried], 1 + max(resource for _, resource in tasks), [], b"",
                nodes.node[nodes.self], places, [])
    return solved(plan, private, context)


# design/torch-source-lowering-change-request.md item 3; design/aot-compilation-change-request.md A6;
# design/integration-finishing-change-request.md item 2; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper.
# Res. 22(3), 1997, section 2; M. Cowan et al., MSCCLang, ASPLOS 2023, section 5.2, step 3
def schedule(module, cost=unit_cost, carried=(), machines=None, mutated=()):
    nodes, functions, preds = dependence(module)
    functions = [node for node in mesh_graph.postorder(functions, preds)]
    task = {node: k for k, node in enumerate(functions)}
    processors = dict.fromkeys(n.meta["processor"] for n in nodes if n.op != "placeholder")
    lane = {processor: k for k, processor in enumerate(processors)}
    tasks = [(cost(node), lane[node.meta["processor"]]) for node in functions]
    edges = [(task[earlier], task[node]) for node in functions for earlier in dict.fromkeys(preds(node))]
    before = mesh_graph.closure(functions, preds)
    carried = [transfer for transfer in carried if all(transfer[4])]
    plan = planned(module, carried, machines, mutated, cost, functions, before, tasks, edges) if carried else None
    completed = plan.completion if plan else mesh_graph.completion_times(
        [length for length, _ in tasks], [resource for _, resource in tasks], len(lane), edges)
    modules = {}
    for processor in processors:
        own = [node for node in functions if node.meta["processor"] == processor]
        at = {node: k for k, node in enumerate(own)}
        place = mesh_graph.heap_order([-completed[task[node]] for node in own],
                                      [(at[earlier], at[node]) for node in own for earlier in before[node]
                                       if earlier in at])
        modules[processor] = assemble(processor, sorted(own, key=lambda node: place[at[node]]), nodes)
    return modules, plan


def assemble(processor, order, nodes):
    graph, copied = torch.fx.Graph(), {}
    for node in nodes:
        if node.op == "placeholder" and any(u.meta["processor"] == processor for u in node.users):
            copied[node] = graph.node_copy(node)
    for node in order:
        copied[node] = graph.node_copy(node, copied.__getitem__)
        parts = [u for u in node.users if u.target is operator.getitem]
        for part in sorted(parts, key=lambda u: u.args[1]):
            copied[part] = graph.node_copy(part, copied.__getitem__)
    output = nodes[-1]
    here = output.meta["processor"] == processor
    results = map_arg(output.args[0], copied.__getitem__) if here else ()
    graph.output(results).meta["processor"] = processor
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def argument_buffers(modules, output_processor):
    inputs = {processor: [n.target for n in module.graph.nodes if n.op == "placeholder"]
              for processor, module in modules.items()}
    at = {processor: index for index, processor in enumerate(modules)}
    edges = [(at[output_processor], at[processor]) for processor in modules if processor != output_processor]
    edges += [(at[end[SEND].meta["processor"]], at[end[RECV].meta["processor"]]) for end in channel_ends(
        node for module in modules.values() for node in module.graph.nodes).values()]
    component = mesh_graph.components(len(at), edges)
    return inputs, sorted(p for p in modules if component[at[p]] != component[at[output_processor]])

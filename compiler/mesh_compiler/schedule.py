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


def dependence(module):
    nodes = list(module.graph.nodes)
    functions = [n for n in nodes if n.op == "call_function" and producer(n) is n]
    sender = {channel_of(n): n for n in functions if n.target is SEND}

    def preds(node):
        found = [producer(u) for u in node.all_input_nodes if u.op == "call_function"]
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


# rdma/mesh.h mesh_seqlock_whole
def image():
    live = mesh.Context()
    mesh.check(mesh.LIB.mesh_attach(C.byref(live), None))
    header = mesh.Header.from_address(live.M)
    private = mmap.mmap(-1, header.length)
    address = C.addressof(C.c_char.from_buffer(private))
    while True:
        before = header.view_map
        C.memmove(address, live.M, header.data_off)
        if not before & 1 and header.view_map == before:
            break
    C.memset(address + header.algo_off, 0, C.sizeof(mesh.Algo))
    made = mesh.Context(M=address, len=header.length, client=live.client, fd=-1, start=live.start)
    mesh.check(mesh.LIB.mesh_detach(C.byref(live)))
    return private, made


# design/disjoint-pages-change-request.md C3.3 and C3.5; rdma/mesh-collective.h struct mesh_dependence
def dependences(events, before):
    def precedes(a, b):
        (_, node, _, first), (_, other, _, second) = events[a], events[b]
        return a != b and node == other and all(x in before[y] for x in first for y in second)

    after = {a: [b for b in range(len(events)) if precedes(a, b)] for a in range(len(events))}
    covering = mesh_graph.reduction(mesh_graph.postorder(list(after), after.__getitem__), after.__getitem__)
    found = [(a, b, 0) for a in covering for b in covering[a]]
    for processor in dict.fromkeys(first[0].meta["processor"] for _, _, _, first in events):
        held = [a for a in covering if events[a][3][0].meta["processor"] == processor]
        firsts, lasts = mesh_graph.extremes(held, lambda a: [b for b in covering[a] if b in held])
        found += [(a, b, 1) for a in lasts for b in firsts]
    return [(events[a][0], events[b][0], lag, events[a][2][0] | events[b][2][1]) for a, b, lag in found]


# design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5;
# design/integration-finishing-change-request.md item 3; rdma/mesh-collective.h mesh_plan, mesh_plan_save
def planned(module, carried, machines, cost=unit_cost):
    carried = [transfer for transfer in carried if all(transfer[4])]
    if not carried:
        return None
    private, context = image()
    nodes = mesh.Nodes()
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(context), C.byref(nodes)))
    at = {nodes.node[i]: i for i in range(nodes.count)}
    _, functions, preds = dependence(module)
    ends, held = channel_ends(functions), set(functions)
    before = mesh_graph.closure(mesh_graph.postorder(functions, preds), preds)
    declared, weights, values, events = [], [], [], []
    for transfer, (channel, sender, receiver, dtype, shape, stride) in enumerate(carried):
        sent, arrival = ends[channel][SEND], ends[channel][RECV]
        source, destination = at[machines[sender]], at[machines[receiver]]
        written = producer(sent.args[0])
        value = len(values) if nodes.self in (source, destination) else mesh.ABSENT
        if value != mesh.ABSENT:
            size = dtype.itemsize * (1 + sum((n - 1) * s for n, s in zip(shape, stride)))
            values.append((size, mesh.EXECUTION, 1, 0, 0, 0, 0, 1))
        layout = (0, dtype.itemsize, [x for n, s in zip(shape, stride) for x in (n if s else 1, s * dtype.itemsize)])
        declared.append((source, destination, channel, arrival.meta["consumer"], value, layout))
        weights.append((cost(written if written in held else sent), cost(arrival), cost(sent), cost((sent, arrival))))
        read = [sent] + [u for u in sent.args[0].users if u in held and u is not sent]
        events += [(transfer, source, PRODUCED, [written])] * (written in held)
        events += [(transfer, source, READ, read), (transfer, destination, CONSUMED,
                                                    [arrival] + [u for u in arrival.users if u in held])]
    plan = Plan(sum(1 << int(node) for node in os.environ["MESH_PROGRAM"].split(",")), [c for c, *_ in carried],
                declared, weights, values, dependences(events, before), 1, [], b"")
    function, _ = mesh.function(plan.declared, plan.weights, plan.values, plan.dependences, plan.outstanding)
    calls, made, record, length = (mesh.U * 1)(0), mesh.Planned(), C.c_void_p(), C.c_size_t()
    operands = (plan.program, function, 1, calls, 1, None, None, 0, None)
    mesh.check(mesh.LIB.mesh_plan(C.byref(context), int(os.environ["MESH_GROUP"]), *operands,
                                  time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS, C.byref(made)))
    mesh.check(mesh.LIB.mesh_plan_save(C.byref(context), *operands, C.byref(made), C.byref(record), C.byref(length)))
    state = mesh.State.from_address(context.M + function[0].plan)
    plan.times = list((C.c_double * len(declared)).from_address(context.M + state.times))
    plan.record = C.string_at(record, length.value)
    mesh.LIB.mesh_planned_free(C.byref(made))
    SYSTEM.free(record)
    return plan


# design/integration-finishing-change-request.md item 2; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper.
# Res. 22(3), 1997, section 2; M. Cowan et al., MSCCLang, ASPLOS 2023, section 5.2, step 3
def schedule(module, cost=unit_cost, plan=None):
    nodes, functions, preds = dependence(module)
    key = mesh_graph.longest(functions, preds, lambda before, node: cost(node))
    if plan is not None:
        ends = channel_ends(functions)
        key.update((ends[channel][RECV], completed) for channel, completed in zip(plan.channels, plan.times))
    at = {node: place for place, node in enumerate(functions)}
    place = mesh_graph.heap_order([-key[node] for node in functions],
                                  [(at[before], at[node]) for node in functions for before in preds(node)])
    order = sorted(functions, key=lambda node: place[at[node]])
    processors = dict.fromkeys(n.meta["processor"] for n in nodes if n.op != "placeholder")
    return {p: assemble(p, [n for n in order if n.meta["processor"] == p], nodes) for p in processors}


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

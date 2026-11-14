# design/data-flow.md; design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5
import ctypes as C
import dataclasses
import operator
import pathlib
import sys
import time

import torch

from torch.fx.node import map_arg

from .ir import HOST, RECV, channel_of, views

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


# compiler/mesh_compiler/ir.py based, views
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
        return (key[name], *mesh_graph.strided(offset, sizes, strides, element), vertex, write)

    found, sources = [], []
    for vertex, node in enumerate(functions):
        for record in wrote(node):
            found.append(access(record, node.meta["val"].dtype.itemsize, vertex, 1))
            sources.append((node, record))
        for held in () if "view" in node.meta else dict.fromkeys(node.all_input_nodes):
            record = held.meta.get("view") or held.meta.get("out") or whole(held)
            if record and record[0] in key and record not in wrote(node):
                found.append(access(record, held.meta["val"].dtype.itemsize, vertex, 0))
                sources.append((held, record))
    return found, sources


# design/compilation-composition-change-request.md items 6, 6.4 and 7; rdma/mesh-graph.h mesh_graph_relation_create:
# the source's one relation, its memory witnesses (rdma/mesh_graph.py dependences_supplied over the accesses' views)
# and, as GIVEN causes of the same solve, each SSA input and each declared channel; `accesses` each view's producing
# source view, `supplied` each GIVEN cause's (kind, source, sink, operand)
@dataclasses.dataclass(eq=False)
class SourceRelation:
    relation: object
    functions: list
    supplied: list
    accesses: list


# design/problems-of-14b9a757-change-request.md items 1, 2 and 6; A. J. Bernstein, IEEE Trans. Electronic Computers
# EC-15(5), 1966; compiler/mesh_compiler/placement.py place, transfers
def dependence(module, transfers):
    nodes = list(module.graph.nodes)
    functions = [n for n in nodes if n.op == "call_function" and producer(n) is n]
    vertex, after, supplied = {node: k for k, node in enumerate(functions)}, {}, []
    views, sources = accesses(functions)
    for node in functions:
        supplied += [("value", producer(u), node, u) for u in dict.fromkeys(node.all_input_nodes)
                     if u.op == "call_function"]
        if node.target is RECV:
            supplied.append(("channel", transfers[channel_of(node)][0], node, channel_of(node)))
    relation = mesh_graph.dependences_supplied(len(functions), views,
                                               [(vertex[source], vertex[sink]) for _, source, sink, _ in supplied])
    edges = relation.edges()
    for earlier, later in edges:
        after.setdefault(functions[later], []).append(functions[earlier])
    writers = [(functions[earlier], functions[later]) for e, (earlier, later) in enumerate(edges)
               if any(mesh_graph.HAZARDS[relation.conflict(k).kind] == mesh_graph.HAZARDS[0] for k in relation.members(e))]

    def preds(node):
        found = [producer(u) for u in node.all_input_nodes if u.op == "call_function"] + after.get(node, [])
        return found + [transfers[channel_of(node)][0]] if node.target is RECV else found

    return nodes, functions, preds, writers, SourceRelation(relation, functions, supplied, sources)


# compiler/mesh_compiler/placement.py place, transfers
def carried(transfers, machines):
    found = []
    for channel, (sent, arrival) in sorted(transfers.items()):
        sender, receiver = sent.meta["processor"], arrival.meta["processor"]
        if machines[sender] != machines[receiver]:
            value = arrival.meta["val"]
            found.append((channel, sender, receiver, value.dtype, list(value.shape), list(value.stride())))
    return found


# design/disjoint-pages-change-request.md C3.3 and C3.5; rdma/mesh-collective.h struct mesh_dependence
def dependences(events, before):
    found = mesh_graph.precedence([nodes for _, _, _, nodes in events], [place for _, place, _, _ in events],
                                  [nodes[0].meta["processor"] for _, _, _, nodes in events], before)
    return [(events[a][0], events[b][0], lag, events[a][2][0] | events[b][2][1]) for a, b, lag in found]


def extent(value):
    return value.dtype.itemsize * mesh_graph.extent(value.shape, value.stride())


# rdma/mesh-collective.h mesh_plan, mesh_plan_save; compiler/mesh_compiler/ir.py Planning; rdma/mesh-graph.h
# mesh_graph_archive; rdma/mesh-dataflow.h mesh_view_values; design/compilation-composition-change-request.md items
# 3, 6.4 and 7: the one plan of `function` (the caller's actual mesh.Function, its parts and values the lowered
# program's), projected into `plan`; its mesh.Planned, mapped view and the view's first/offset arrays returned for the
# caller to retain and free
def solved(plan, planning, function):
    calls, made, record, length = (mesh.U * 1)(0), mesh.Planned(), C.c_void_p(), C.c_size_t()
    operands = (plan.program, function, 1, calls, 1, None, None, 0, None)
    context = C.byref(planning.context)
    mesh.check(mesh.LIB.mesh_plan(context, planning.group, *operands,
                                  time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS, C.byref(made)))
    mesh.check(mesh.LIB.mesh_plan_save(context, *operands, C.byref(made), C.byref(record), C.byref(length)))
    plan.completion = list(function[0].completion[:len(plan.tasks)])
    plan.order = list(function[0].order[:len(plan.tasks)])
    sections = [value.section for value in function[0].value[:len(plan.values)]]
    plan.sections = [[getattr(section, name) for name, _ in mesh.Section._fields_] for section in sections]
    plan.record = C.string_at(record, length.value)
    plan.signature, plan.nodes = mesh_graph.archive(made.signature.contents), tuple(planning.nodes)
    first, offset, view = (mesh.Q * (len(sections) + 1))(), (mesh.Q * sum(s.count for s in sections))(), mesh.View()
    mesh.LIB.mesh_view_values(context, function, 1, first, offset, C.byref(view))
    plan.arena_bytes = view.bytes
    plan.offsets = tuple(tuple(offset[first[value]:first[value + 1]]) for value in range(len(sections)))
    # rdma/mesh.h mesh_transfer_at, struct mesh_publication; rdma/mesh-collective.c plan_supplied, plan_completion and
    # the publication records of transfer i written for its demand p->of[i]: one supplied word per demand release, one
    # completion word per sending demand (d->sending = p->sending++), so equal nonzero words are one native action
    state = mesh.State.from_address(planning.context.M + function[0].plan)
    supplied, completed = {}, {}
    plan.publications = tuple(
        (supplied.setdefault(record.targets[0].word, transfer),
         completed.setdefault(record.completion, transfer) if record.sends and record.completion else None)
        if source == planning.here and record.targets[0].word else (None, None)
        for transfer, (source, *_) in enumerate(plan.declared)
        for record in [mesh.Publication.from_address(planning.context.M + state.transfers
                                                     + mesh.PUBLICATION_BYTES * transfer)])
    SYSTEM.free(record)
    return plan, made, view, first, offset


# design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5;
# rdma/mesh-collective.h struct mesh_function, mesh_plan; design/compilation-composition-change-request.md items 4 and
# 7: the source declarations of the one plan, before any solve: its declared transfers and their logical wire values,
# the caller's costs, the transfer dependences and the source tasks in causal postorder; the numerical values,
# lifetimes and costs are the lowered program's actual results, added by the plan's consumer
@dataclasses.dataclass
class Declarations:
    program: int
    channels: list
    declared: list
    weights: list
    values: list
    places: list
    dependences: list
    outstanding: int
    tasks: list
    before: list
    senders: list
    receivers: list
    resources: int
    machine: int

    # channel -> (declared transfer, logical wire value, its logical bytes, declared sent view, declared landed view)
    def wires(self):
        return {channel: (transfer, value, self.values[value][0], sent, landed)
                for transfer, (channel, (*_, value, sent, landed)) in enumerate(zip(self.channels, self.declared))
                if value != mesh.ABSENT}


# design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5;
# design/integration-finishing-change-request.md item 3; compiler/mesh_compiler/ir.py Planning;
# compiler/mesh_compiler/placement.py place, transfers
def declarations(module, transfers, carried, machines, cost, functions, before, tasks, edges, planning, writers=()):
    nodes = planning.nodes if planning is not None else ()
    at, task = {node: k for k, node in enumerate(nodes)}, {node: k for k, node in enumerate(functions)}
    here = planning.here if planning is not None else None
    held = set(functions)
    declared, weights, values, places, events, identities, versions = [], [], [], [], [], {}, {}
    named = {node.name: node for node in module.graph.nodes}
    for writer, reader in writers:
        versions.setdefault(reader, set()).add(task[writer])
    for transfer, (channel, sender, receiver, dtype, shape, stride) in enumerate(carried):
        sent, arrival = transfers[channel]
        source, destination = at[machines[sender]], at[machines[receiver]]
        start = sent.args[0].meta.get("view") or sent.args[0].meta.get("out") or whole(sent.args[0])
        finish, version = arrival.meta.get("to") or whole(arrival), tuple(sorted(versions.get(sent, ())))
        written, value = producer(sent.args[0]), mesh.ABSENT
        if here in (source, destination):
            writer = (sender, start[0], version) if source == here else (receiver, finish[0], ())
            if writer not in places:
                places.append(writer)
                values.append((extent(named[writer[1]].meta["val"]), mesh.EXECUTION, 1, 0, 0, 0, 0, 1))
            value = places.index(writer)
        identity = identities.setdefault((sent.meta["collective"], start[0], version), len(identities))
        declared.append((source, destination, identity, arrival.meta["consumer"], value,
                         *(mesh_graph.strided(*record[1:], dtype.itemsize) for record in (start, finish))))
        weights.append((cost(written if written in held else sent), cost(arrival), cost(sent), cost((sent, arrival))))
        read = [sent] + [u for u in sent.args[0].users if u in held and u is not sent]
        events += [(transfer, source, PRODUCED, [written])] * (written in held)
        events += [(transfer, source, READ, read), (transfer, destination, CONSUMED,
                                                    [arrival] + [u for u in arrival.users if u in held])]
    ends = [tuple(task[end] for end in transfers[channel]) for channel, *_ in carried]
    crossing = set(ends)
    return Declarations(planning.program if planning is not None else 0, [c for c, *_ in carried], declared, weights,
                        values, places, dependences(events, before), 1, tasks,
                        [edge for edge in edges if edge not in crossing], [sent for sent, _ in ends],
                        [arrival for _, arrival in ends], 1 + max((resource for _, resource in tasks), default=0),
                        nodes[here] if planning is not None else None)


# design/torch-source-lowering-change-request.md item 3; design/aot-compilation-change-request.md A6;
# design/integration-finishing-change-request.md item 2; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper.
# Res. 22(3), 1997, section 2; M. Cowan et al., MSCCLang, ASPLOS 2023, section 5.2, step 3; compiler/mesh_compiler/ir.py
# Planning; design/compilation-composition-change-request.md items 2, 3, 6 and 7; rdma/mesh-graph.h mesh_graph_contract
def schedule(module, cost=unit_cost, carried=(), machines=None, mutated=(), planning=None, groups=None):
    transfers = module.meta["transfers"]
    nodes, functions, preds, writers, source = dependence(module, transfers)
    graph = mesh_graph.Graph.from_conflicts(source.functions, source.relation)
    functions = graph.order()
    task = {node: k for k, node in enumerate(functions)}
    processors = dict.fromkeys(n.meta["processor"] for n in nodes if n.op != "placeholder")
    lane = {processor: k for k, processor in enumerate(processors)}
    tasks = [(cost(node), lane[node.meta["processor"]]) for node in functions]
    edges = [(task[earlier], task[later]) for earlier, later in graph.edges()]
    before = graph.ancestors(following=True)
    carried = [transfer for transfer in carried if all(transfer[4])]
    declared = declarations(module, transfers, carried, machines, cost, functions, before, tasks, edges, planning,
                            writers)
    listed = functions
    side = {processor: (groups or {}).get(processor, processor) for processor in processors}
    local = {end for ends in transfers.values() if len({side[end.meta["processor"]] for end in ends}) == 1
             for end in ends}
    pairs = graph.contract([node not in local for node in graph.vertices]).edges()
    identity = {node: (k, resource) for k, (node, (_, resource)) in enumerate(zip(functions, tasks))}
    members = {group: tuple(p for p in processors if side[p] == group) for group in dict.fromkeys(side.values())}
    sequence = tuple(identity[node] for node in listed)
    return {group: assemble(group, held, [node for node in listed if node.meta["processor"] in held], nodes, pairs,
                            transfers, local, identity, writers, sequence, source)
            for group, held in members.items()}, declared


# design/compilation-composition-change-request.md items 2, 3 and 6; M. Abadi et al., TensorFlow, arXiv:1603.04467,
# section 3.2.2: a Send/Receive pair stands only for an edge between devices; compiler/mesh_compiler/ir.py views
def assemble(group, members, order, nodes, pairs, transfers, local, identity, writers, sequence, source):
    graph, copied, tasks = torch.fx.Graph(), {}, {}
    for node in nodes:
        if node.op == "placeholder" and any(u.meta["processor"] in members for u in node.users):
            copied[node] = graph.node_copy(node)
    for node in order:
        if node in local:
            copied[node] = copied[transfers[channel_of(node)][0].args[0]]
        else:
            copied[node] = graph.node_copy(node, copied.__getitem__)
            tasks[copied[node]] = identity[node]
            for part in sorted((u for u in node.users if u.target is operator.getitem), key=lambda u: u.args[1]):
                copied[part] = graph.node_copy(part, copied.__getitem__)
                tasks[copied[part]] = identity[node]
    output = nodes[-1]
    here = output.meta["processor"] in members
    results = map_arg(output.args[0], copied.__getitem__) if here else ()
    graph.output(results).meta["processor"] = group
    views([copied[user] for node in order if node in local and node.target is RECV for user in node.users
           if user in copied and user not in local])
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    module.meta.update(members=members, tasks=tasks, dependences=[
        (copied[earlier], copied[later]) for earlier, later in pairs
        if earlier.meta["processor"] in members and later.meta["processor"] in members], writers=[
        (copied[writer], copied[reader]) for writer, reader in writers
        if writer.meta["processor"] in members and reader.meta["processor"] in members], received_strides={
        channel: tuple(sent.args[0].meta["val"].stride()) for channel, (sent, arrival) in transfers.items()
        if arrival in copied and arrival not in local}, order=sequence, transfers={
        channel: (copied.get(sent), copied.get(arrival)) for channel, (sent, arrival) in transfers.items()
        if arrival not in local}, relation=source, copied=dict(copied), output=output.meta["processor"], vertices=tuple(
        None if node in local else copied.get(node) for node in source.functions))
    return module


# compiler/mesh_compiler/ir.py Scheduled; compiler/mesh_compiler/placement.py place, transfers
def argument_buffers(modules, output_processor, held, transfers, groups=None):
    inputs = {processor: [n.target for n in module.graph.nodes if n.op == "placeholder"]
              for processor, module in modules.items()}
    stored = {processor: [str(held.get(name, HOST)) for name in names] for processor, names in inputs.items()}
    at = {processor: index for index, processor in enumerate(modules)}
    edges = [(at[output_processor], at[processor]) for processor in modules if processor != output_processor]
    ends = [[(groups or {}).get(end.meta["processor"], end.meta["processor"]) for end in pair]
            for pair in transfers.values()]
    edges += [(at[sender], at[receiver]) for sender, receiver in ends if sender != receiver]
    component = mesh_graph.components(len(at), edges)
    return inputs, sorted(p for p in modules if component[at[p]] != component[at[output_processor]]), stored

# design/data-flow.md; design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5
import ctypes as C
import dataclasses
import functools
import operator
import pathlib
import sys
import time
import weakref

import torch

from torch.export.graph_signature import OutputKind
from torch.fx.node import Node, map_arg
from torch.multiprocessing.reductions import StorageWeakRef

from .ir import HOST, JOIN, RECV, channel_of, views

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

SYSTEM = C.CDLL("/usr/lib/libSystem.B.dylib")
SYSTEM.free.argtypes = [C.c_void_p]
PRODUCED, READ, CONSUMED = ((mesh.FROM_PRODUCED, mesh.TO_PRODUCED), (mesh.FROM_SOURCE_READ, mesh.TO_SOURCE_READ),
                            (0, 0))
CLONE = torch.ops.aten.clone.default


# compiler/mesh_compiler/ir.py views: a receive whose arrival lands in an existing destination (its "to" record)
def landing(node):
    return node.target is RECV and node.meta.get("to") is not None


def unit_cost(task):
    return 1.0


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
# source view, `supplied` each GIVEN cause's (kind, source, sink, operand); `producers` each original node's producing
# source operation (a tuple getitem's its parent's, every other node itself)
@dataclasses.dataclass(eq=False)
class SourceRelation:
    relation: object
    functions: list
    supplied: list
    accesses: list
    producers: dict


# design/problems-of-14b9a757-change-request.md items 1, 2 and 6; A. J. Bernstein, IEEE Trans. Electronic Computers
# EC-15(5), 1966; compiler/mesh_compiler/placement.py place, transfers
def dependence(module, transfers):
    nodes = list(module.graph.nodes)
    producers = mesh_graph.forward(nodes, lambda n: [n.args[0]] if n.target is operator.getitem else [],
                                   lambda n, parent: parent[0] if parent else n)
    functions = [n for n in nodes if n.op == "call_function" and producers[n] is n]
    vertex, supplied = {node: k for k, node in enumerate(functions)}, []
    views, sources = accesses(functions)
    for node in functions:
        supplied += [("value", producers[u], node, u) for u in dict.fromkeys(node.all_input_nodes)
                     if u.op == "call_function"]
        if node.target is RECV:
            supplied.append(("channel", transfers[channel_of(node)][0], node, channel_of(node)))
    relation = mesh_graph.dependences_supplied(len(functions), views,
                                               [(vertex[source], vertex[sink]) for _, source, sink, _ in supplied])
    writers = [(functions[earlier], functions[later]) for e, (earlier, later) in enumerate(relation.edges())
               if any(mesh_graph.HAZARDS[relation.conflict(k).kind] == mesh_graph.HAZARDS[0] for k in relation.members(e))]
    return nodes, functions, writers, SourceRelation(relation, functions, supplied, sources, producers)


# compiler/mesh_compiler/placement.py place, transfers
def carried(transfers, machines):
    found = []
    for channel, (sent, arrival) in sorted(transfers.items()):
        sender, receiver = sent.meta["processor"], arrival.meta["processor"]
        if machines[sender] != machines[receiver]:
            value = arrival.meta["val"]
            found.append((channel, sender, receiver, value.dtype, list(value.shape), list(value.stride())))
    return found


# design/disjoint-pages-change-request.md C3.3 and C3.5; rdma/mesh-collective.h struct mesh_dependence; L. Lamport,
# CACM 21(7), 1978; M. Lam, PLDI 1988, section 2.3; rdma/mesh-graph.h mesh_graph_relation_synchronization,
# mesh_graph_boundaries: event a before event b where they differ, share their place and every node of a precedes every
# node of b; the relation's non-redundant edges at delay 0, and within each thread each last event before each first at
# delay 1; rdma/mesh-collective.h struct mesh_dependence: one array of exactly the records, its `owner` the events,
# closure, event Graph, mask, boundary Views and each record's (a, b, lag, edge or None), for the Plan's lifetime
def dependences(events, before):
    place = [at for _, at, _, _ in events]
    graph = mesh_graph.Graph.adjacency(range(len(events)), lambda a: [
        b for b in range(len(events)) if a != b and place[a] == place[b]
        and all(x in before[y] for x in events[a][3] for y in events[b][3])])
    redundant = graph.relation.redundant()
    thread = [nodes[0].meta["processor"] for _, _, _, nodes in events]
    dense = {one: k for k, one in enumerate(dict.fromkeys(thread))}
    first, last = graph.boundaries([dense[one] for one in thread], redundant)
    found = [(a, b, 0, e) for e, (a, b) in enumerate(graph.edges()) if not redundant[e]]
    for one in dense:
        held = [a for a in range(len(events)) if thread[a] == one]
        found += [(a, b, 1, None) for a in held if last[a] for b in held if first[b]]
    made = (mesh.Dependence * len(found))(*[
        mesh.Dependence(events[a][0], events[b][0], lag, events[a][2][0] | events[b][2][1]) for a, b, lag, _ in found])
    made.owner = events, before, graph, redundant, first, last, found
    return made


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
# lifetimes and costs are the lowered program's actual results, added by the plan's consumer. `values` and `places`
# (and each row's value) this machine's (`machine`, Planning.here's); `endpoints[t]` transfer t's (sender's,
# receiver's) (machine index into `nodes`, its logical value there), `logical[m]` machine m's (places, values), this
# machine's the same lists as `values` and `places`
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
    nodes: tuple = ()
    endpoints: list = dataclasses.field(default_factory=list)
    logical: dict = dataclasses.field(default_factory=dict)

    # channel -> (declared transfer, logical wire value, its logical bytes, declared sent view, declared landed view):
    # this machine's, or (`machine`, compiler/mesh_compiler/driver.py's global construction) that machine's endpoints'
    def wires(self, machine=None):
        if machine is None:
            return {channel: (transfer, value, self.values[value][0], sent, landed)
                    for transfer, (channel, (*_, value, sent, landed)) in enumerate(zip(self.channels, self.declared))
                    if value != mesh.ABSENT}
        k = self.nodes.index(machine) if self.endpoints else None
        return {channel: (transfer, value, self.logical[m][1][value][0], row[-2], row[-1])
                for transfer, (channel, row, ends) in enumerate(zip(self.channels, self.declared, self.endpoints))
                for m, value in ends if m == k}


# design/aot-compilation-change-request.md A3 to A8; design/plan-to-bridge-change-request.md R1 to R5;
# design/integration-finishing-change-request.md item 3; compiler/mesh_compiler/ir.py Planning;
# compiler/mesh_compiler/placement.py place, transfers
def declarations(module, transfers, carried, machines, cost, functions, before, tasks, edges, planning, writers,
                 relation):
    nodes = planning.nodes if planning is not None else ()
    at, task = {node: k for k, node in enumerate(nodes)}, {node: k for k, node in enumerate(functions)}
    here = planning.here if planning is not None else None
    held = set(functions)
    declared, weights, values, places, events, identities, versions = [], [], [], [], [], {}, {}
    endpoints, logical = [], {} if here is None else {here: (places, values)}
    named = {node.name: node for node in module.graph.nodes}
    for writer, reader in writers:
        versions.setdefault(reader, set()).add(task[writer])

    # an endpoint's logical wire value on machine m: one per writer (its processor, its source record's root, its
    # version), so a same-version fanout and a receive assembly share it, its extent the root's whole storage's
    def placed(m, writer):
        mine, held_values = logical.setdefault(m, ([], []))
        if writer not in mine:
            mine.append(writer)
            held_values.append((extent(named[writer[1]].meta["val"]), mesh.EXECUTION, 1, 0, 0, 0, 0, 1))
        return mine.index(writer)
    for transfer, (channel, sender, receiver, dtype, shape, stride) in enumerate(carried):
        sent, arrival = transfers[channel]
        source, destination = at[machines[sender]], at[machines[receiver]]
        start = sent.args[0].meta.get("view") or sent.args[0].meta.get("out") or whole(sent.args[0])
        finish, version = arrival.meta.get("to") or whole(arrival), tuple(sorted(versions.get(sent, ())))
        written = relation.producers[sent.args[0]]
        pair = ((source, placed(source, (sender, start[0], version))),
                (destination, placed(destination, (receiver, finish[0], ()))))
        endpoints.append(pair)
        value = next((v for m, v in pair if m == here), mesh.ABSENT)
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
                        nodes[here] if planning is not None else None, tuple(nodes), endpoints, logical)


# design/torch-source-lowering-change-request.md item 3; design/aot-compilation-change-request.md A6;
# design/integration-finishing-change-request.md item 2; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper.
# Res. 22(3), 1997, section 2; M. Cowan et al., MSCCLang, ASPLOS 2023, section 5.2, step 3; compiler/mesh_compiler/ir.py
# Planning; design/compilation-composition-change-request.md items 2, 3, 6 and 7; rdma/mesh-graph.h mesh_graph_contract
def schedule(module, cost=unit_cost, carried=(), machines=None, mutated=(), planning=None, groups=None):
    transfers = module.meta["transfers"]
    nodes, functions, writers, source = dependence(module, transfers)
    graph = mesh_graph.Graph.from_conflicts(source.functions, source.relation)
    functions = graph.order()
    task = {node: k for k, node in enumerate(functions)}
    processors = dict.fromkeys(n.meta["processor"] for n in nodes if n.op != "placeholder")
    lane = {processor: k for k, processor in enumerate(processors)}
    tasks = [(cost(node), lane[node.meta["processor"]]) for node in functions]
    edges = [(task[earlier], task[later]) for earlier, later in graph.edges()]
    before = graph.ancestors(functions[::-1], successors=True)
    # Rejection filters are not part of even temporary integration validation.
    # Zero-byte payloads need no transport row; their source operations and dependence causes remain above.
    carried = [transfer for transfer in carried if all(transfer[4])]
    declared = declarations(module, transfers, carried, machines, cost, functions, before, tasks, edges, planning,
                            writers, source)
    listed = functions
    side = {processor: (groups or {}).get(processor, processor) for processor in processors}
    local = {end for ends in transfers.values() if len({side[end.meta["processor"]] for end in ends}) == 1
             for end in ends}
    pairs = graph.contract([node not in local or landing(node) for node in graph.vertices]).edges() if local else \
        graph.edges()
    identity = {node: (k, resource) for k, (node, (_, resource)) in enumerate(zip(functions, tasks))}
    members = {group: tuple(p for p in processors if side[p] == group) for group in dict.fromkeys(side.values())}
    sequence = tuple(identity[node] for node in listed)
    return {group: assemble(group, held, [node for node in listed if node.meta["processor"] in held], nodes, pairs,
                            transfers, local, identity, writers, sequence, source, module.meta["decomposed"],
                            module.meta.get("input_parts", {}))
            for group, held in members.items()}, declared


# design/compilation-composition-change-request.md items 2, 3 and 6; M. Abadi et al., TensorFlow, arXiv:1603.04467,
# section 3.2.2: a Send/Receive pair stands only for an edge between devices, so a local pair is erased, its receive
# the sent value, except a receive landing in an existing destination (`landing`), which stays at its position as
# one copy of the sent value (aten.clone, its "out" the receive's "to", its task and vertex the receive's);
# compiler/mesh_compiler/ir.py views;
# compiler/mesh_compiler/decompose.py decompose: `decomposed` each expanded node's original source node, the copy of
# each of them here its module's meta["decomposed"] key; the placeholders copied, in their original order, those a
# node of this machine reads and those its write address records (`wrote`: "out", "to") name, a pure destination
# included; compiler/mesh_compiler/placement.py place `input_parts`: each
# placeholder this machine copied, by its position among the original placeholders, its partitions on this
# machine's processors as (processor, original placeholder, original partition Node, its landing geometry), resolved
# by `copied`
def assemble(group, members, order, nodes, pairs, transfers, local, identity, writers, sequence, source, decomposed,
             parts):
    graph, copied, tasks = torch.fx.Graph(), {}, {}
    addressed = {record[0] for node in order for record in wrote(node)}
    for node in nodes:
        if node.op == "placeholder" and (any(u.meta["processor"] in members for u in node.users)
                                         or node.name in addressed):
            copied[node] = graph.node_copy(node)
    for node in order:
        if node in local and landing(node):
            copied[node] = graph.create_node("call_function", CLONE, (copied[transfers[channel_of(node)][0].args[0]],),
                                             name=node.name)
            copied[node].meta.update(processor=node.meta["processor"], val=node.meta["val"], out=node.meta["to"])
            tasks[copied[node]] = identity[node]
        elif node in local:
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
    graph.output(results).meta.update(processor=group, destination=output.meta.get("destination", {}) if here else {})
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
        None if node in local and not landing(node) else copied.get(node) for node in source.functions),
        decomposed={copied[node]: original for node, original in decomposed.items() if node in copied},
        input_parts={position: tuple((processor, node, part, geometry) for processor, part, geometry in parts[node]
                                     if processor in members)
                     for position, node in enumerate(n for n in nodes if n.op == "placeholder")
                     if node in copied and node in parts})
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


# M. C. Rinard and P. C. Diniz, Commutativity analysis, ACM TOPLAS 19(6), 1997, sections 3.2.1, 4.9.1 and 4.9.2: a fixed
# body's associative and commutative operators, a difference the sum with a negation, a product distributed over a sum,
# each in one normal form (its operands flattened and sorted)
ALGEBRA = {torch.ops.aten.add: "add", torch.ops.aten.mul: "mul", torch.ops.aten.sub: "sub", torch.ops.aten.neg: "neg",
           torch.ops.aten.maximum: "maximum", torch.ops.aten.minimum: "minimum"}


def normal(name, operands):
    def of(e, kind):
        return isinstance(e, tuple) and e[:1] == (kind,)
    if name == "sub":
        return normal("add", (operands[0], normal("neg", operands[1:])))
    if name == "neg":
        return operands[0][1] if of(operands[0], "neg") else ("neg", operands[0])
    for i, e in enumerate(operands):
        if name == "mul" and of(e, "add"):
            return normal("add", tuple(normal("mul", operands[:i] + operands[i + 1:] + (term,)) for term in e[1:]))
    return (name, *sorted((f for e in operands for f in (e[1:] if of(e, name) else (e,))), key=repr))


# Rinard and Diniz, sections 3.2.1 and 4.9.2; torch v2.14.0 torch/fx/interpreter.py: a fixed body run over symbols
class Symbolic(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        name = None if kwargs else ALGEBRA.get(getattr(target, "overloadpacket", None))
        if name and len(args) == 1 + (name != "neg"):
            return normal(name, tuple(args))
        return target, tuple(args), tuple(sorted(kwargs.items()))


# Rinard and Diniz, sections 3.1 and 4.8: whether two calls of `function` (one result, its USER_OUTPUT) composed through
# its user input placeholder k, the first's result the second's input k, give one result in either order, symbolically
# over the exported fixed body: every other user input the call's own symbol, its state and tokens shared
def commutes(function, k):
    names = [n.name for n in function.graph_module.graph.nodes if n.op == "placeholder"]
    result = [kind for kind, _ in function.outputs].index(OutputKind.USER_OUTPUT)

    def run(receiver, call):
        return Symbolic(function.graph_module).run(*[
            receiver if i == k else ("input", call, i) if name in function.user_inputs else ("state", name)
            for i, name in enumerate(names)], enable_io_processing=False)[result]
    return names[k] in function.user_inputs and run(run(("receiver",), 1), 2) == run(run(("receiver",), 2), 1)


# design/aot-compilation-change-request.md A10 and appendix item 6; A. J. Bernstein, IEEE Trans. Electronic Computers
# EC-15(5), 1966; Rinard and Diniz, section 2; R. P. Dilworth, Annals of Math., 1950; D. R. Fulkerson, Proc. AMS, 1956:
# a CapturedProgram's calls analysed. Call j is the original call node nodes[j] of function keys[calls[j]];
# `storages[s]` storage s's (StorageWeakRef, first holder: the first call's leaf bound to it), s the views' key;
# `accesses[v]` the relation's view v's (call, operand): ("input", placeholder, leaf), every tensor the call binds to
# one of its function's placeholders (state and user inputs alike, ir.Call), read, ("written", output, leaf), the
# leaf bound to a mutation output's target placeholder, written, or ("generated", key, write), its body's generator
# state read and written (`generated[key]`, analyse); `results` each (call, output, leaf, storage) of a
# call's tensor output, its storage a root whether or not a later call reads it; `root_sources[s]` storage s's
# caller source (compiler/mesh_compiler/native.py Operand.source): ("input", its outer flat placeholder position),
# ("result", s, its storage's bytes, its device) of a program result's, None a call's internal planned root; `held`
# the storages with a source;
# `supplied[g]` GIVEN cause g with its role:
# ("value", producer, consumer, leaf), a commuting chain's member reading its head's receiver and a chain's last
# consumer following every member, or ("chain", a, b, head), the successive calls of one head's label (the commuting
# chains, `paths`, and each persistent written storage's calls, an outer input's or state's, or a body's generated
# state's, `generated`); `chains[j]` call j's
# head, the least call of its label (mesh_plan `chains`); `relation` the one relation of those views and causes
# (mesh_graph.dependences_supplied), `graph` it (Graph.from_conflicts), `search` its owned search over incoming rows
# and `closure` each call's predecessors (Graph.ancestors over that search's postorder); `order_edges` its edges with a
# view hazard or a value cause, projected as mesh_plan's `orders` and `causes` (ORDER_CONFLICT where a view hazard is
# among an edge's causes, ORDER_FLOW otherwise) until the planner takes `relation`; `widths[key]` the function's fewest
# chains covering its calls in `closure` (mesh_graph.closure_chains), `outstanding[key]` their count; `generated` each
# generated state's conflict key (past `storages`, no caller storage) to (its body's source ordinal, machine group, state
# Root, view); `constructions` the program's one construction owner, every (function key, machine group) body's
# ConstructedMachine (compiler/mesh_compiler/driver.py compile_captured), kept with the analysis with `declarations`,
# each function's Declarations, whose endpoints those bodies were constructed with; `operands` the CallOperands it was
# analysed from (its captured functions' placeholder order)
# design/aot-compilation-change-request.md A2, A4 and A10; design/plan-to-bridge-change-request.md R6: one value
# crossing a call boundary or the program's: produced by call `owner` at its output spec `output` (None: the caller's
# outer placeholder at position `output`), bound by call `consumer` at its placeholder `input` (None: the program's
# outer output at position `input`); `storage` its source storage (CallAnalysis.storages), `view` the leaf's strided
# view (byte offset, element bytes, (size, byte stride) pairs: mesh_graph.strided), `source` the owner's machine (its
# function's output processor's; None the caller's input), `machines` the consumer's (its placeholder's stated
# processors', a sharding's flattened; the outer output processor's), `leaf` the original FX Node
@dataclasses.dataclass(frozen=True)
class CallPort:
    owner: object
    output: int
    consumer: object
    input: int
    storage: int
    view: tuple
    source: object
    machines: tuple
    leaf: object


@dataclasses.dataclass(eq=False)
class CallAnalysis:
    nodes: tuple
    keys: tuple
    calls: object
    storages: list
    accesses: list
    results: list
    root_sources: list
    held: frozenset
    supplied: list
    relation: object
    graph: object
    search: object
    closure: object
    order_edges: list
    orders: object
    causes: object
    norders: int
    chains: object
    paths: object
    widths: dict
    outstanding: dict
    placements: dict
    machines: dict
    output_processor: object
    output_machine: object
    ports: tuple
    generated: dict
    constructions: dict
    declarations: dict
    operands: object


# design/aot-compilation-change-request.md A10; design/compilation-composition-change-request.md items 6 and 7: a
# CapturedProgram's calls' operands, extracted once before any construction (the driver's planning needs `ports`):
# `entries[j]` call j's ir.Call, `ordinal` each function key's source ordinal, `storages`, `views` (each
# (storage, offset, element, pairs, call, write)), `accesses`, `results`, `root_sources`, `held` and `ports` as
# CallAnalysis's, `existing` the outer placeholders' storages, `produced` each call output leaf's (call, output)
@dataclasses.dataclass(eq=False)
class CallOperands:
    captured: object
    nodes: tuple
    keys: tuple
    entries: list
    ordinal: dict
    storages: list
    views: list
    accesses: list
    results: list
    root_sources: list
    held: frozenset
    ports: tuple
    existing: set
    produced: dict
    placements: dict
    machines: dict
    output_processor: object


def call_operands(captured, placements, machines, output_processor):
    from .placement import flat
    nodes, keys = tuple(captured.calls), tuple(captured.functions)
    entries, ordinal = [captured.calls[node] for node in nodes], {key: f for f, key in enumerate(keys)}
    storage, storages, views, accesses = {}, [], [], []
    placeholders = {key: [n.name for n in f.graph_module.graph.nodes if n.op == "placeholder"]
                    for key, f in captured.functions.items()}

    def rooted(tensor, holder):
        key = storage.setdefault(StorageWeakRef(tensor.untyped_storage()), len(storage))
        if key == len(storages):
            storages.append((StorageWeakRef(tensor.untyped_storage()), holder))
        return key

    def viewed(tensor, holder, j, write, operand):
        key = rooted(tensor, holder)
        views.append((key, *mesh_graph.strided(tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
                                               tensor.dtype.itemsize), j, write))
        accesses.append((j, operand))

    def value(leaf):
        return leaf.meta.get("val") if isinstance(leaf, Node) else None
    existing = {StorageWeakRef(value(n).untyped_storage()) for n in captured.exported.graph_module.graph.nodes
                if n.op == "placeholder" and isinstance(value(n), torch.Tensor)}
    for j, call in enumerate(entries):
        function, names = captured.functions[call.function], placeholders[call.function]
        for i, leaf in enumerate(call.inputs):
            if isinstance(value(leaf), torch.Tensor):
                viewed(value(leaf), leaf, j, 0, ("input", i, leaf))
        state = {key: name for name, (key, _) in function.state.items()}
        for o, (kind, target) in enumerate(function.outputs):
            if kind in (OutputKind.USER_INPUT_MUTATION, OutputKind.BUFFER_MUTATION):
                leaf = call.inputs[names.index(target if kind == OutputKind.USER_INPUT_MUTATION else state[target])]
                viewed(value(leaf), leaf, j, 1, ("written", o, leaf))
    results = [(j, o, leaf, rooted(value(leaf), leaf)) for j, call in enumerate(entries)
               for o, leaf in enumerate(call.outputs) if isinstance(value(leaf), torch.Tensor)]
    returned, root_sources = [], [None] * len(storages)
    map_arg(captured.exported.graph_module.graph.output_node().args, returned.append)
    outer = [n for n in captured.exported.graph_module.graph.nodes if n.op == "placeholder"]
    for i, n in enumerate(outer):
        s = storage.get(StorageWeakRef(value(n).untyped_storage())) if isinstance(value(n), torch.Tensor) else None
        if s is not None and root_sources[s] is None:
            root_sources[s] = ("input", i)
    for n in returned:
        s = storage.get(StorageWeakRef(value(n).untyped_storage())) if isinstance(value(n), torch.Tensor) else None
        if s is not None and root_sources[s] is None:
            root_sources[s] = ("result", s, value(n).untyped_storage().nbytes(), value(n).device)
    held = frozenset(s for s, source in enumerate(root_sources) if source is not None)
    outer_at, ports = {n: i for i, n in enumerate(outer)}, []

    def stated(held):
        return tuple(dict.fromkeys(machines[q] for q in ((held,) if isinstance(held, str) else flat(held[1]))))

    def ported(owner, output, consumer, input, leaf, source, at):
        tensor = value(leaf)
        s = storage.get(StorageWeakRef(tensor.untyped_storage())) if isinstance(tensor, torch.Tensor) else None
        if s is not None:
            ports.append(CallPort(owner, output, consumer, input, s, mesh_graph.strided(
                tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype.itemsize), source,
                at, leaf))
    produced = {leaf: (j, o) for j, call in enumerate(entries) for o, leaf in enumerate(call.outputs)
                if isinstance(leaf, Node)}
    for b, call in enumerate(entries):
        for i, leaf in enumerate(call.inputs):
            if leaf in produced:
                a, o = produced[leaf]
                ported(a, o, b, i, leaf, machines[placements[entries[a].function][1]],
                       stated(placements[call.function][0][i]))
            elif leaf in outer_at:
                ported(None, outer_at[leaf], b, i, leaf, None, stated(placements[call.function][0][i]))
    for k, leaf in enumerate(captured.exported.graph_module.graph.output_node().args[0]):
        if leaf in produced:
            a, o = produced[leaf]
            ported(a, o, None, k, leaf, machines[placements[entries[a].function][1]], (machines[output_processor],))
        elif leaf in outer_at:
            ported(None, outer_at[leaf], None, k, leaf, None, (machines[output_processor],))
    return CallOperands(captured, nodes, keys, entries, ordinal, storages, views, accesses, results, root_sources, held,
                        tuple(ports), existing, produced, placements, machines, output_processor)


# the one call analysis (CallAnalysis) of `operands` (call_operands) after every construction, `constructions` every
# (function key, machine group) body's ConstructedMachine, on every participant the same: their views and, for each
# body that advances its generator (lower_metal.py ConstructedMachine.generated), its state's read and written view
# (operations.py Generator.counter) in every call of its function, its own conflict key `generated` past the caller's
# storages ({key: (source ordinal, machine group, state Root, view)}; the local body's native.py Target binds as
# ("generator", ordinal)), so calls of one function conflict on each of its bodies' states and distinct states never
def analyse(operands, constructions, declarations):
    captured, nodes, keys, entries, produced = (operands.captured, operands.nodes, operands.keys, operands.entries,
                                                operands.produced)
    ordinal, storages, J = operands.ordinal, operands.storages, len(nodes)
    views, accesses, generated = list(operands.views), list(operands.accesses), {}
    for (key, group), made in constructions.items():
        if not made.generated:
            continue
        root, counter = made.generated
        s = len(storages) + len(generated)
        generated[s] = (ordinal[key], group, root, counter)
        view = mesh_graph.strided(counter.storage_offset(), tuple(counter.shape), tuple(counter.stride()),
                                  counter.dtype.itemsize)
        for j, call in enumerate(entries):
            if call.function == key:
                for write in (0, 1):
                    views.append((s, *view, j, write))
                    accesses.append((j, ("generated", s, write)))
    link, follower = {}, {}
    commuting = functools.cache(lambda key, k: commutes(captured.functions[key], k))
    for b, call in enumerate(entries):
        function, reads = captured.functions[call.function], [u for u in call.inputs if isinstance(u, Node)]
        single = [kind for kind, _ in function.outputs] == [OutputKind.USER_OUTPUT]
        for k, u in enumerate(call.inputs):
            a = produced.get(u, (None,))[0]
            if (a is not None and single and entries[a].function == call.function and len(entries[a].outputs) == 1
                    and len(u.users) == 1 and reads.count(u) == 1 and link.get(a, (a, k))[1] == k
                    and commuting(call.function, k)):
                link[b], follower[a] = (a, k), b
                break
    paths = mesh_graph.paths(range(J), lambda a: [follower[a]] if a in follower else [])
    step = {j: chain for chain in paths for j in chain}
    supplied = []
    for b, call in enumerate(entries):
        read = [entries[step[b][0]].inputs[link[b][1]] if b in link and k == link[b][1] else u
                for k, u in enumerate(call.inputs)]
        for u in dict.fromkeys(u for u in read if u in produced):
            a = produced[u][0]
            members = step[a] if a in step and step[a][-1] == a else (a,)
            supplied += [("value", m, b, u) for m in members]
    persistent = ({s for s, (ref, _) in enumerate(storages) if ref in operands.existing} &
                  {v[0] for v in views if v[-1]}) | set(generated)
    groups = [list(chain) for chain in paths] + [sorted({v[-2] for v in views if v[0] == s}) for s in persistent]
    label = mesh_graph.components(J, [pair for group in groups for a, b in zip(group, group[1:])
                                      for pair in ((a, b), (b, a))])
    least = {}
    for j, c in enumerate(label):
        least.setdefault(c, j)
    head = [least[c] for c in label]
    supplied += [("chain", a, b, head[b]) for a, b in mesh_graph.successive(J, head)]
    relation = mesh_graph.dependences_supplied(J, views, [(a, b) for _, a, b, _ in supplied])
    graph = mesh_graph.Graph.from_conflicts(range(J), relation)
    search = graph.search(graph.vertices, incoming=True)
    closure = graph.ancestors(mesh_graph.ordered(search, search.made.contents.postorder,
                                                 range(search.made.contents.count)), incoming=True)
    hazards = [{relation.conflict(k).kind != mesh_graph.GIVEN or supplied[relation.conflict(k).given][0]
                for k in relation.members(e)} for e in range(relation.data.edges)]
    order_edges = [e for e, kinds in enumerate(hazards) if kinds & {True, "value"}]
    edges = relation.edges()
    widths = {key: mesh_graph.closure_chains(closure, [j for j in range(J) if entries[j].function == key])
              for key in keys}
    return CallAnalysis(nodes, keys, (mesh.U * max(J, 1))(*[ordinal[call.function] for call in entries]), storages,
                        accesses, operands.results, operands.root_sources, operands.held, supplied, relation, graph,
                        search, closure, order_edges,
                        (mesh.U * max(2 * len(order_edges), 1))(*[x for e in order_edges for x in edges[e]]),
                        (mesh.U * max(len(order_edges), 1))(*[mesh.ORDER_CONFLICT if True in hazards[e] else
                                                              mesh.ORDER_FLOW for e in order_edges]),
                        len(order_edges), (mesh.U * max(J, 1))(*head), paths, widths,
                        {key: count for key, (count, _) in widths.items()}, operands.placements, operands.machines,
                        operands.output_processor, operands.machines[operands.output_processor], operands.ports,
                        generated, constructions, declarations, operands)


# mesh-collective.h mesh_plan_local, mesh_local_load: one solved or loaded result, released by mesh_local_free with its
# last reference; `held` what it borrows (the caller's inputs when solved, the archive's bytes when loaded)
class LocalOwner:
    def __init__(self, held=()):
        self.local, self.held = mesh.Local(), held
        self.freed = weakref.finalize(self, mesh.LIB.mesh_local_free, C.byref(self.local))


# mesh-collective.h struct mesh_local value_first, first and offset: native views of one result, owned by `owner`:
# function f's values' first entries, the roots', every copy's byte offset
def viewed(owner, local):
    F, S, prefix = local.functions, local.roots, local.value_first
    return ([mesh_graph.View(owner, local.first, range(prefix[f], prefix[f + 1] + 1), int) for f in range(F)],
            mesh_graph.View(owner, local.first, range(prefix[F], prefix[F] + S + 1), int),
            mesh_graph.View(owner, local.offset, range(local.offsets), int))


# design/aot-compilation-change-request.md A5 to A10 and appendix item 6; design/disjoint-pages-change-request.md;
# mesh-collective.h mesh_plan_local, mesh_local_save, mesh_local_load: one solved program. `functions` Function[K], the
# calls' instances; `local` the owned result and `planned` its planned record; `function_value_first[k]`,
# `root_first` and `offsets` native views (viewed); `bindings` each (call, rootValueID, sourceStorageIndex, delta,
# originalReference, every, ordinal), its source storage resolved once from the Slot's reference, `delta` the source
# view's byte offset less the Slot's, (every, ordinal) its call's generation (function f's copy of execution e = t *
# every + ordinal is e % (horizon * every), a root's copy block e // every); `source_bindings` the native Bindings
# given; `root_sources` and `analysis` the CallAnalysis's; `roots` mesh.Value[S + W] (the S source storages, bytes, run
# 0 a caller's root, then the W staging wires of Prepared.wires, run 1; shares as solved); `uses` (use_first, use);
# `plan_calls[j]` call j's instance in `functions` (Function[K]; the source function stays analysis.calls[j]), the
# local result's calls; `prepared` the declare_program owner its bindings, uses and groupings come from; `plan` a
# network result's owner (mesh_plan_program's: the Solved itself, its `program` the planning's; a loaded one's its saved
# source; None locally), `planned` staying its native Planned view; `ports[j]` the prepared PortRecords (their demand
# and release the canonical grouping's, which the plan consumed); `context` the planning context (None locally);
# `owner` the native owner. A loaded result has no bindings, sources, analysis, prepared owner or ports: its rows bind
# through the saved Target
@dataclasses.dataclass(eq=False)
class Solved:
    functions: object
    local: object
    planned: object
    function_value_first: list
    root_first: object
    offsets: object
    bindings: tuple
    source_bindings: tuple
    root_sources: list
    analysis: object
    roots: object
    uses: tuple
    plan_calls: object
    prepared: object
    plan: object
    ports: tuple
    context: object
    owner: object

    # a network result's program mask (mesh_program_save's and mesh_program_load's operand): its planning's, or, loaded
    # by its preparation, its saved source's
    @property
    def program(self):
        return self.prepared.planning.program if self.prepared is not None else self.plan.program

    # mesh-collective.h mesh_local_save, a local result's archive; mesh_program_save, a network result's whole archive
    # from its planning context and program (its declaration owner's planning); a network result its preparation
    # loaded (no declaration owner) its saved source's original bytes
    def saved(self):
        if self.prepared is None and self.plan is not None:
            return self.plan.saved()
        data, length = C.c_void_p(), mesh.Z()
        mesh.check(mesh.LIB.mesh_local_save(C.byref(self.local), C.byref(data), C.byref(length)) if self.plan is None
                   else mesh.LIB.mesh_program_save(C.byref(self.context), self.prepared.planning.program,
                                                   C.byref(self.local), C.byref(data), C.byref(length)))
        try:
            return C.string_at(data, length.value)
        finally:
            SYSTEM.free(data)


# mesh.h MESH_SEND, MESH_RECEIVE; design/aot-compilation-change-request.md A2, A4 and A10; design/plan-to-bridge-
# change-request.md R6: one call's end of one port row: `slot` this call's typed slot ValueID at the port's position
# (its Slot's root, the logical port, never the wire; None: no numeric site on this machine), `direction` mesh.SEND
# (this call owns the row) or mesh.RECEIVE (this call consumes it), `transfer` the row's position in its owner's
# combined declared array, `owner` the producing call, `consumer` the consuming call (None: the program's outer output)
# and `input` its input position (the outer output position where `consumer` is None), `storage` the logical source or
# result root (its source storage) and `view` its original strided view, `wire` the joint root the row is declared on
# (mesh-collective.h struct mesh_declared `root` - 1: `storage` itself where that root is a run 1 root, else its
# staging WireRoot's), the same on every protocol row of the port, `payload` the view's logical bytes, `extent` the
# backing root's bytes (the source storage's, a staging wire's extent), `at` the view's byte offset in the source
# storage and in the wire, `output` the owner's output position, `source` and `destination` the row's machines,
# `demand` and `release` the canonical grouping's (mesh-collective.h mesh_declared_group, Prepared.grouped[owner]:
# demand_of[transfer], release_of[transfer]; release None where this node does not send the row; both None locally)
@dataclasses.dataclass(frozen=True)
class PortRecord:
    slot: object
    direction: int
    transfer: int
    owner: int
    consumer: object
    input: int
    storage: int
    view: tuple
    wire: int
    payload: int
    extent: int
    at: int
    output: int
    source: object
    destination: object
    demand: object = None
    release: object = None


# design/aot-compilation-change-request.md A2 and A10; mesh-collective.h mesh_plan_program (a transported root is a run
# 1 root): one genuine staging wire of a transferred caller-owned (run 0) source storage, one per (owner call,
# storage): `root` its joint root index (after the S source storages), `storage` the logical source root, `owner` the
# producing call, `extent` its backing bytes, the full source storage's; solve_program makes it a run 1 root
@dataclasses.dataclass(frozen=True)
class WireRoot:
    root: int
    storage: int
    owner: int
    extent: int


# One required copy between a caller-owned root and its staging wire, both at `view` (the original strided view, the
# same byte offsets in both): `kind` "stage" (on `machine`, the owner's source machine, on `processor`, the owner
# function's selected output processor analysis.placements[key][1]: the source storage `storage` into the wire root
# `wire`, its typed source endpoint `slot`, the owner's original Binding output Slot (root, view, offset, reference) at
# its output position `output`) or "land" (on `machine`, an outer output port's destination, on `processor`,
# analysis.output_processor: the landed wire `wire`, its ValueID Prepared.wired[(call, wire)], into the outer result
# root `storage` at outer output position `output`, `slot` None) or, a normal land ("land", `input` not None), on
# `machine`, this node, a row's destination, on `processor`, its InputPart's: the landed wire `wire` into consumer
# call `call`'s whole logical input at its original input (placeholder) position `input`, `slot` its original
# Binding.inputs[input] Slot (the copy's destination endpoint, numeric fields None where it has no body), `part` one
# exact InputPart of that input here (lower_metal.py InputPart, as native's ConstructedMachine.input_parts holds it,
# its IDs the original Program's: its correspondence, not the copy's target; one record per part), `storage` the
# caller-owned source S, `output` None; `leaf` the original FX output Node (CallPort.leaf, its meta the typed tensor),
# `call` the call whose instance row on `machine` holds the site, `owner` the transfer's owner call (stage and outer
# land: `call`), whose declared rows `transfers` it serves and whose demands and groups they are; `input` and `part`
# None for stage and outer land
@dataclasses.dataclass(frozen=True)
class CopyRecord:
    kind: str
    call: int
    owner: int
    machine: object
    processor: object
    storage: int
    wire: int
    view: tuple
    output: object
    input: object
    slot: object
    leaf: object
    part: object
    transfers: tuple


# metal-microbench metal_ir.h metal_ir_value_info, metal_ir_value_shape and metal_ir_value_layout; compiler/
# mesh_compiler/metal.py _Builder.value (one linear term per dimension): typed value `value` of a constructed body's
# program as mesh_graph.strided (its byte offset in its storage root, element bytes, pairs), with its ValueInfo
def typed_view(made, value):
    from .metal import Term, ValueInfo
    from .operations import SCALAR
    info, at = ValueInfo(), C.c_int64()
    made.compiler.metal_ir_value_info(made.program, value, C.byref(info))
    rank = made.compiler.metal_ir_value_shape(made.program, value, None, 0)
    shape, terms = (mesh.Q * max(rank, 1))(), (Term * max(rank, 1))()
    made.compiler.metal_ir_value_shape(made.program, value, shape, rank)
    made.compiler.metal_ir_value_layout(made.program, value, terms, rank, C.byref(at))
    element = next(t for t, code in SCALAR.items() if code == info.scalar).itemsize
    return mesh_graph.strided(at.value, list(shape[:rank]), [terms[d].coefficient for d in range(rank)], element), info


# design/aot-compilation-change-request.md A4 and A9; design/plan-to-bridge-change-request.md R1 and R2: one endpoint of
# original call `call`'s intrinsic transfer `transfer` (its row of Prepared.declared[call]), `direction` mesh.SEND or
# mesh.RECEIVE, on `machine` and `processor` (its original occurrence's): `construction` the existing ConstructedMachine
# of machine group `group` holding `channel`, `occurrence` the original source SEND or RECV node
# (construction.relation.supplied) and `vertex` its source vertex (construction.relation.functions; its emitted
# operations construction.vertices[vertex]), `logical` its logical wire value there (Declarations.endpoints); `root`
# its selected typed Root: construction.sent[channel], or construction.received[channel] (the receive wire), or, a
# RECV with a landing (construction.landings[channel]: its interface destination Root and copy operations) whose
# destination is bound to a run 1 shared root, that destination; `view` that Root's typed view (typed_view) in its
# backing's coordinates, `backing` that backing's bytes: unbound (`storage` None) its typed root's, bound to run 1
# shared root `storage` that root's full extent, the view moved by the binding's `delta` (the call's outer reference's
# byte offset less the typed interface view's); `staged` a landing kept: the receive wire selected (its own geometry
# and bytes), the landing's copy into its destination retained, `storage` and `delta` the destination's binding
@dataclasses.dataclass(frozen=True)
class EndpointRecord:
    call: int
    transfer: int
    direction: int
    machine: object
    processor: object
    construction: object
    group: object
    channel: object
    occurrence: object
    vertex: int
    root: object
    logical: int
    view: tuple
    backing: int
    storage: object
    delta: int
    staged: bool


# design/aot-compilation-change-request.md A2, A4, A8 and A10; mesh-collective.h struct mesh_declared, struct
# mesh_weight, mesh_declared_group: the program's declarations, prepared once before typed construction. `bindings`
# each (call, rootValueID, sourceStorageIndex, delta, originalReference), the native Bindings' slots resolved once
# against analysis.storages (`delta` the source view's byte offset less the Slot's), with each land copy's (`bind`);
# `uses` each (storage, call, rootValueID) once, in order, a land copy's with them; `ports[j]` call j's PortRecords;
# `declared[j]` owner call j's combined rows (source, destination, identity, kind, value, root, from, to): its
# function's intrinsic declarations projected from its EndpointRecords, then each cross-call row of a port it owns,
# in CallPort and destination order
# (identity its (owner, storage) above the intrinsic ones, root its PortRecord `wire` + 1, value its typed root on a run
# 1 root, a staging wire's ValueID once bound (`bind`), else MESH_ABSENT); `weights[j]` and `dependences[j]` the
# intrinsic ones, a cross-call row's weight the default unit cost; `extents[j]` each row's backing bytes (an intrinsic
# row's selected endpoint backing; a cross-call row's source storage's) and `costs[j]` its
# weight's provenance ("declared"; "absent": no cost operand is supplied for a call boundary, the default unit weight
# stands in); `arrays[j]` the native Declared and Weight arrays of those rows with what they hold; `grouped[j]` their
# one canonical grouping on this node (mesh.Grouped, its arrays held; None locally), which mesh_plan_program consumes;
# `wires` the staging WireRoots (joint roots S ..), `copies` the CopyRecords they require; `wired` native's map {(call,
# wire root): typed ValueID on this node} (`bind`); `remapped` and `remapped_roots` native's canonical append view
# and storage-root maps (`remap`); `input_parts` native's {call: its body's ConstructedMachine.input_parts} here,
# retained as given (each normal land CopyRecord's `part` one of its InputParts); `endpoints[j]` call j's intrinsic
# transfers' EndpointRecords (sender, receiver per transfer), whose views are its intrinsic rows' from and to and whose
# local endpoint gives each row's value, root and extent;
# `source_bindings`, `declarations`, `wire_values` and `planning` as given (the original ValueIDs)
@dataclasses.dataclass(eq=False)
class Prepared:
    analysis: object
    source_bindings: tuple
    bindings: tuple
    uses: tuple
    root_sources: list
    ports: tuple
    declared: tuple
    weights: tuple
    dependences: tuple
    extents: tuple
    costs: tuple
    arrays: tuple
    declarations: object
    wire_values: object
    planning: object
    grouped: tuple = ()
    wires: tuple = ()
    copies: tuple = ()
    wired: dict = dataclasses.field(default_factory=dict)
    remapped: object = None
    remapped_roots: object = None
    input_parts: object = None
    endpoints: tuple = ()

    # metal-microbench metal_ir.h metal_ir_append and metal_ir_value_info: native's canonical append results, each
    # {call: array indexed by the call's original ValueID} for the calls it appended to only, retained as given, once
    # before `bind`: `values` (`remapped`) each original ValueID's new view ValueID, `roots` (`remapped_roots`) the
    # resulting Program's ValueInfo.storage of it, its storage root. Each root this owner holds of such a call moves
    # through that call's `roots`: the resolved bindings' rootValueID and the uses' ValueID (storage, delta and leaf as
    # they are), each PortRecord's slot (its port call's), each owner call's Declared value in its rows and its native
    # array (MESH_ABSENT staying absent), each CopyRecord slot's root (its copy call's); each CopyRecord slot's view
    # through `values` (the Slot's ordinal, offset and reference kept), each local EndpointRecord's Root (its root
    # through `roots`, its value through `values`; a remote endpoint's IDs are its machine's and kept). A view ValueID
    # is never a planned root.
    # source_bindings, wire_values, the groupings and the declared geometry keep the originals, and bind's operands are
    # the final programs' already
    def remap(self, values, roots):
        self.remapped, self.remapped_roots = values, roots

        def moved(tables, call, value):
            table = tables.get(call)
            return value if table is None or value is None or value == mesh.ABSENT else table[value]
        self.bindings = tuple(((call, moved(roots, call, root), storage, delta), reference)
                              for (call, root, storage, delta), reference in self.bindings)
        self.uses = tuple(sorted((storage, call, moved(roots, call, root)) for storage, call, root in self.uses))
        self.ports = tuple(tuple(dataclasses.replace(record, slot=moved(roots, j, record.slot)) for record in records)
                           for j, records in enumerate(self.ports))
        rows = []
        for j, listed in enumerate(self.declared):
            rows.append(tuple((*row[:4], moved(roots, j, row[4]), *row[5:]) for row in listed))
            for t, row in enumerate(rows[-1]):
                self.arrays[j][0][t].value = row[4]
        self.declared = tuple(rows)
        self.copies = tuple(copy if copy.slot is None else dataclasses.replace(copy, slot=dataclasses.replace(
            copy.slot, root=moved(roots, copy.call, copy.slot.root), view=moved(values, copy.call, copy.slot.view)))
                            for copy in self.copies)
        here = self.planning.nodes[self.planning.here] if self.planning is not None else None
        self.endpoints = tuple(tuple(dataclasses.replace(e, root=e.root._replace(
            root=moved(roots, j, e.root.root), value=moved(values, j, e.root.value))) if e.machine == here else e
            for e in records) for j, records in enumerate(self.endpoints))

    # native's wire-to-ValueID map, {(call, wire root): its distinct typed ValueID of that wire on this node}: each
    # staging wire row's value (its owner's ValueID of the wire, MESH_ABSENT where the owner has none here), and
    # solve_program's wire uses beside the logical ones (never in place of them); `copied` each (call, rootValueID,
    # sourceStorage, delta, originalLeaf) of native's typed copy construction for a land row (a call with no numeric
    # body here among them: a stage's source keeps its original Slot), `delta` the original leaf's view byte offset
    # less the compiled typed view's: its logical root binding entered once, as given, in the same `bindings`
    # ((call, rootValueID, sourceStorage, delta), originalLeaf) and `uses` ((sourceStorage, call, rootValueID), in
    # order) solve_program consumes
    def bind(self, wired, copied=()):
        self.wired, S = dict(wired), len(self.analysis.storages)
        bindings = dict(self.bindings)
        for call, root, storage, delta, leaf in copied:
            bindings.setdefault((call, root, storage, delta), leaf)
        self.bindings = tuple(bindings.items())
        self.uses = tuple(sorted({*self.uses, *((storage, call, root) for call, root, storage, _, _ in copied)}))
        rows = [list(listed) for listed in self.declared]
        for j, listed in enumerate(rows):
            for t, row in enumerate(listed):
                if row[5] - 1 >= S:
                    value = self.wired.get((j, row[5] - 1), mesh.ABSENT)
                    listed[t] = (*row[:4], value, *row[5:])
                    self.arrays[j][0][t].value = value
        self.declared = tuple(map(tuple, rows))


def declare_program(analysis, bindings, declarations, wire_values, planning, *, input_parts=None):
    J, S, at = len(analysis.nodes), len(analysis.storages), {ref: s for s, (ref, _) in enumerate(analysis.storages)}
    resolved, used, slots, endpoints = {}, {}, {}, {}
    for binding in bindings:
        for side, entries in (("input", binding.inputs), ("output", binding.outputs)):
            for position, slot in enumerate(entries):
                slots[binding.call, side, position], endpoints[binding.call, side, position] = slot.root, slot
                value = slot.reference.meta.get("val") if isinstance(slot.reference, Node) else None
                if slot.root is None or not isinstance(value, torch.Tensor):
                    continue
                s = at[StorageWeakRef(value.untyped_storage())]
                resolved.setdefault((binding.call, slot.root, s,
                                     value.storage_offset() * value.element_size() - slot.offset), slot.reference)
                used.setdefault((s, binding.call, slot.root), None)
    node = {machine: k for k, machine in enumerate(planning.nodes)} if planning is not None else {}
    here, by_call, held_by, origin = planning.here if planning is not None else None, {}, {}, {}
    for binding in bindings:
        by_call[binding.call] = binding
    # each channel's endpoints' constructions (analysis.constructions, every participant's the same), and each
    # function's original SEND and RECV occurrences with their source vertices (its SourceRelation's channel causes)
    for (key, group), made in analysis.constructions.items():
        for direction, ends in ((mesh.SEND, made.sent), (mesh.RECEIVE, made.received)):
            for channel in ends:
                held_by[key, channel, direction] = group, made
        origin.setdefault(key, ({n: k for k, n in enumerate(made.relation.functions)},
                                {channel: (sent, arrival) for kind, sent, arrival, channel in made.relation.supplied
                                 if kind == "channel"}))
    names = {key: [n.target for n in f.graph_module.graph.nodes if n.op == "placeholder"]
             for key, f in analysis.operands.captured.functions.items()}

    # call j's binding of a body's typed root: the call's outer reference at that root's interface position (an input
    # by placeholder target, an output by position) as (storage, its byte offset less the typed interface view's)
    def binding_of(j, key, made, root):
        for typed, slot in ([(r, by_call[j].inputs[names[key].index(r.name)]) for r in made.inputs if r.root == root] +
                            [(r, by_call[j].outputs[o]) for o, r in enumerate(made.outputs) if r.root == root]):
            value = slot.reference.meta.get("val") if isinstance(slot.reference, Node) else None
            if isinstance(value, torch.Tensor):
                return (at[StorageWeakRef(value.untyped_storage())],
                        value.storage_offset() * value.element_size() - typed_view(made, typed.value)[0][0])
        return None, 0

    def endpoint(j, key, intrinsic, t, side, direction):
        channel, (m, logical) = intrinsic.channels[t], intrinsic.endpoints[t][side]
        (group, made), (vertex, occurrences) = held_by[key, channel, direction], origin[key]
        root, occurrence = (made.sent if direction == mesh.SEND else made.received)[channel], occurrences[channel][side]
        landing = made.landings.get(channel) if direction == mesh.RECEIVE else None
        storage, delta = binding_of(j, key, made, (root if landing is None else landing[0]).root)
        direct = storage is not None and analysis.root_sources[storage] is None
        root = landing[0] if landing is not None and direct else root
        view, _ = typed_view(made, root.value)
        if direct:
            view, backing = (view[0] + delta, *view[1:]), analysis.storages[storage][1].meta["val"].untyped_storage(
                ).nbytes()
        else:
            backing = typed_view(made, root.root)[1].bytes
        return EndpointRecord(j, t, direction, intrinsic.nodes[m], occurrence.meta.get("processor"), made, group,
                              channel, occurrence, vertex[occurrence], root, logical, view, backing, storage, delta,
                              landing is not None and not direct)

    def shared(e):
        return e is not None and e.storage is not None and not e.staged and analysis.root_sources[e.storage] is None
    declared, weights, dependences, found, identities, extents, costs, located = [], [], [], [], [], [], [], []
    for j in range(J):
        key = analysis.keys[analysis.calls[j]]
        intrinsic = (declarations or {}).get(key)
        rows = intrinsic.declared if intrinsic is not None else []
        # each intrinsic transfer's sender and receiver endpoints, projected before its row: from and to their typed
        # views on every participant alike, value, root and extent this node's endpoint's
        ends = [tuple(endpoint(j, key, intrinsic, t, side, direction) for side, direction in
                      ((0, mesh.SEND), (1, mesh.RECEIVE))) for t in range(len(rows))]
        local = [next((e for e, m in zip(pair, row[:2]) if m == here), None) for pair, row in zip(ends, rows)]
        located.append(tuple(e for pair in ends for e in pair))
        declared.append([(source, destination, identity, kind,
                          mesh.ABSENT if e is None else e.root.root, e.storage + 1 if shared(e) else 0,
                          sender.view, receiver.view)
                         for (source, destination, identity, kind, *_), (sender, receiver), e
                         in zip(rows, ends, local)])
        weights.append([tuple(weight) for weight in (intrinsic.weights if intrinsic is not None else [])])
        extents.append([0 if e is None else e.backing for e in local])
        costs.append(["declared"] * len(rows))
        dependences.append(list(intrinsic.dependences) if intrinsic is not None else [])
        identities.append({None: 1 + max((row[2] for row in rows), default=-1)})
    # a transferred caller-owned (run 0) source storage is staged in a genuine run 1 wire root, one per (owner,
    # storage), its extent the full source storage, its rows' views the original ones (at the source's byte `at`)
    staged, copies = {}, {}
    for port in analysis.ports:
        if port.owner is None:
            continue
        a, mine = port.owner, identities[port.owner]
        owned = slots.get((a, "output", port.output))
        tensor = port.leaf.meta["val"]
        extent = analysis.storages[port.storage][1].meta["val"].untyped_storage().nbytes()
        caller = analysis.root_sources[port.storage] is not None
        for destination in (machine for machine in port.machines if machine != port.source):
            identity = mine.setdefault((a, port.storage), mine[None] + len(mine) - 1)
            wire = (staged.setdefault((a, port.storage), WireRoot(S + len(staged), port.storage, a, extent)).root
                    if caller else port.storage)
            transfer = len(declared[a])
            declared[a].append((node[port.source], node[destination], identity, JOIN,
                                mesh.ABSENT if caller or owned is None else owned, wire + 1, port.view, port.view))
            weights[a].append((unit_cost(None),) * 4)
            record = dict(transfer=transfer, owner=a, consumer=port.consumer, input=port.input, storage=port.storage,
                          view=port.view, wire=wire, payload=tensor.numel() * tensor.element_size(), extent=extent,
                          at=port.view[0], output=port.output, source=port.source, destination=destination)
            extents[a].append(extent)
            costs[a].append("absent")
            found.append((a, owned, mesh.SEND, record))
            if port.consumer is not None:
                found.append((port.consumer, slots.get((port.consumer, "input", port.input)), mesh.RECEIVE, record))
            # the typed endpoints: the owner's original output Slot and leaf (stage, on its function's selected
            # processor), the outer output position and leaf (land, on the program's output processor), and, landing
            # here for a consumer call, its original input Slot with each of that input's InputParts here (normal
            # land, on the part's processor)
            viewed_at = (port.view[0], port.view[1], tuple(map(tuple, port.view[2])))
            here = planning is not None and destination == planning.nodes[planning.here]
            parts = ((input_parts or {}).get(port.consumer, {}).get(port.input, ())
                     if caller and here and port.consumer is not None else ())
            for kind, call, machine, processor, output, position, slot, k, part in (
                    (("stage", a, port.source, analysis.placements[analysis.keys[analysis.calls[a]]][1], port.output,
                      None, endpoints.get((a, "output", port.output)), None, None),) if caller else ()) + (
                    (("land", a, destination, analysis.output_processor, port.input, None, None, None, None),)
                    if caller and port.consumer is None else ()) + tuple(
                    ("land", port.consumer, destination, part.processor, None, port.input,
                     endpoints.get((port.consumer, "input", port.input)), k, part) for k, part in enumerate(parts)):
                copies.setdefault((kind, call, a, machine, port.storage, wire, viewed_at, output, position, k),
                                  (processor, port.view, slot, port.leaf, part, []))[-1].append(transfer)
    arrays = []
    for rows, weighted in zip(declared, weights):
        held = []

        def view(offset, element, pairs):
            held.append((mesh.Q * max(2 * len(pairs), 1))(*[word for pair in pairs for word in pair]))
            return mesh.Strided(offset, element, len(pairs), 0, held[-1])
        arrays.append(((mesh.Declared * max(len(rows), 1))(*[
                           mesh.Declared(source, destination, identity, kind, value, root, view(*sent), view(*landed))
                           for source, destination, identity, kind, value, root, sent, landed in rows]),
                       (mesh.Weight * max(len(weighted), 1))(*[mesh.Weight(*weight) for weight in weighted]), held))
    # mesh-collective.h mesh_declared_group: each owner's combined rows grouped once on this node (planning.here), the
    # demands and releases mesh_plan_program consumes and every port record carries
    grouped = []
    for (rows, weighted, _), listed in zip(arrays, declared):
        n = len(listed)
        made = None if planning is None else mesh.Grouped(demand=(mesh.Demand * max(n, 1))(),
                                                          demand_of=(mesh.U * max(n, 1))(),
                                                          release_of=(mesh.U * max(n, 1))())
        if made is not None:
            mesh.check(mesh.LIB.mesh_declared_group(rows, weighted, n, len(planning.nodes), planning.here,
                                                    C.byref(made)))
        grouped.append(made)
    ports = [[] for _ in range(J)]
    for j, slot, direction, record in found:
        made, t = grouped[record["owner"]], record["transfer"]
        release = mesh.ABSENT if made is None else made.release_of[t]
        ports[j].append(PortRecord(slot, direction, **record, demand=None if made is None else made.demand_of[t],
                                   release=None if release == mesh.ABSENT else release))
    return Prepared(analysis, tuple(bindings), tuple(resolved.items()), tuple(sorted(used)), analysis.root_sources,
                    tuple(map(tuple, ports)), tuple(map(tuple, declared)), tuple(map(tuple, weights)),
                    tuple(map(tuple, dependences)), tuple(map(tuple, extents)), tuple(map(tuple, costs)),
                    tuple(arrays), declarations, wire_values, planning, tuple(grouped),
                    tuple(sorted(staged.values(), key=lambda wire: wire.root)),
                    tuple(CopyRecord(kind, call, owner, machine, processor, storage, wire, original, output, position,
                                     slot, leaf, part, tuple(transfers))
                          for (kind, call, owner, machine, storage, wire, _, output, position, _), (
                              processor, original, slot, leaf, part, transfers) in copies.items()),
                    input_parts=input_parts, endpoints=tuple(located))


# The joint roots: the S source storages, then the declaration owner's staging wires (Prepared.wires, run 1, their
# extent); its uses each logical slot's (Prepared.uses) and each wire's (Prepared.wired, native's map, its distinct
# wire ValueIDs), one binding where both name the same root, ValueError where one typed ValueID of a call is bound to
# two distinct roots (a correspondence the caller must state); each instance's grouping (Prepared.grouped[j] at
# plan_calls[j]) consumed by the network plan and borrowed by its result (Local.grouped); the ports the declaration
# owner's, their demand and release canonical before the solve; a network result's `plan` the Solved itself, its
# owner (its program the planning's), a local one's None
def solve_program(analysis, functions, bindings, quantum, planning=None, *, plan_calls, declared):
    J, K = len(analysis.nodes), len(functions)
    S = len(analysis.storages) + len(declared.wires)
    instances = (mesh.U * max(J, 1))(*plan_calls)
    resolved = dict(declared.bindings)
    bound = {}
    for root, call, value in (*declared.uses, *((wire, call, value) for (call, wire), value in declared.wired.items())):
        if bound.setdefault((call, value), root) != root:
            raise ValueError(f"call {call} ValueID {value} is bound to roots {bound[call, value]} and {root}")
    used = sorted((root, call, value) for (call, value), root in bound.items())
    roots = (mesh.Value * max(S, 1))()
    for s, (_, holder) in enumerate(analysis.storages):
        roots[s].section.bytes = holder.meta["val"].untyped_storage().nbytes()
        roots[s].kind, roots[s].run = mesh.EXECUTION, int(analysis.root_sources[s] is None)
    for wire in declared.wires:
        roots[wire.root].section.bytes, roots[wire.root].kind, roots[wire.root].run = wire.extent, mesh.EXECUTION, 1
    use_first, use = (mesh.Z * (S + 1))(), (mesh.LocalUse * max(len(used), 1))(
        *[mesh.LocalUse(call, root) for _, call, root in used])
    for s, _, _ in used:
        use_first[s + 1] += 1
    for s in range(S):
        use_first[s + 1] += use_first[s]
    grouped = (mesh.Grouped * max(K, 1))()
    for j, made in enumerate(declared.grouped):
        if made is not None:
            grouped[plan_calls[j]] = made
    # mesh-collective.h struct mesh_function declared, weights, count, dependence, dependences and outstanding: each
    # network instance plan_calls[j] its call's prepared rows (Prepared.arrays[j], borrowed: the native Declared and
    # Weight arrays as remapped and bound, the rows its grouping was made over), their intrinsic dependences
    # (Prepared.dependences[j], one retained Dependence array) and its original source function's issue bound
    # (analysis.outstanding), the native caller's one instance per original call
    dependence = []
    for j, f in enumerate(plan_calls if planning is not None else ()):
        rows, weighted, _ = declared.arrays[j]
        dependence.append((mesh.Dependence * max(len(declared.dependences[j]), 1))(*declared.dependences[j]))
        functions[f].declared, functions[f].weights, functions[f].count = rows, weighted, len(declared.declared[j])
        functions[f].dependence, functions[f].dependences = dependence[-1], len(declared.dependences[j])
        functions[f].outstanding = analysis.outstanding[analysis.keys[analysis.calls[j]]]
    ordered = (mesh.U * max(len(analysis.order_edges), 1))(*analysis.order_edges)
    outstanding = (mesh.Q * max(len(analysis.keys), 1))(*[analysis.outstanding[key] for key in analysis.keys])
    owner = LocalOwner((analysis, declared, functions, instances, roots, use_first, use, grouped, dependence,
                        ordered, outstanding))
    relation, closure = C.c_void_p(analysis.relation.pointer), C.cast(analysis.closure.owned.made, C.c_void_p)
    if planning is None:
        mesh.check(mesh.LIB.mesh_plan_local(functions, K, instances, J, relation, ordered, len(analysis.order_edges),
                                            closure, analysis.chains, roots, S, use_first, use, quantum,
                                            C.byref(owner.local)))
    else:
        mesh.check(mesh.LIB.mesh_plan_program(C.byref(planning.context), planning.group, planning.program, functions,
                                              K, grouped, instances, J, analysis.calls, len(analysis.keys),
                                              outstanding, relation, ordered, len(analysis.order_edges), closure,
                                              analysis.chains, roots, S, use_first, use,
                                              time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS,
                                              C.byref(owner.local)))
    local, calls = owner.local, owner.local.planned.call
    solved = Solved(functions, local, local.planned, *viewed(owner, local),
                    tuple((*entry, reference, calls[entry[0]].every, calls[entry[0]].ordinal)
                          for entry, reference in resolved.items()), tuple(bindings), analysis.root_sources, analysis,
                    roots, (use_first, use), instances, declared, None, declared.ports,
                    planning.context if planning is not None else None, owner)
    solved.plan = solved if planning is not None else None
    return solved


# mesh-collective.h mesh_program_save: a network result saved and not attached (restored, or saved at compile time):
# `record` its original archive bytes, `program` its program mask, the operand its preparation (compiler/mesh_compiler/
# bridge.py Published, mesh_program_load) loads and publishes it with; its own `plan`
@dataclasses.dataclass(frozen=True, eq=False)
class SavedNetwork:
    record: bytes
    program: int

    @property
    def plan(self):
        return self

    def saved(self):
        return self.record


# mesh-collective.h mesh_program_load and mesh_view_program: a network result its preparation loaded and published
# (bridge.Published), as a Solved: its Function[], roots, planned record and prefixes over the loaded Local, its copies'
# offsets the mapped byte offsets `offset` (the arena's), every view held by `owner` (the preparation's _Mapping and
# `offset`; the _Mapping frees the Local: no second owner); its `plan` the saved source
def published_program(local, owner, offset, source):
    F, S, prefix = local.functions, local.roots, local.value_first
    return Solved(local.function, local, local.planned,
                  [mesh_graph.View(owner, local.first, range(prefix[f], prefix[f + 1] + 1), int) for f in range(F)],
                  mesh_graph.View(owner, local.first, range(prefix[F], prefix[F] + S + 1), int),
                  mesh_graph.View(owner, offset, range(local.offsets), int), (), (), (), None, local.root,
                  (local.use_first, local.use), local.calls, None, source, (), None, owner)


# mesh-collective.h mesh_local_load: the saved result of `data` (Solved.saved), loaded, never solved: its Function[],
# roots, planned record and copies from the canonical owner, which borrows `data` (lent: compiler/mesh_compiler/metal.py
# _lent) until its last reference is freed
def loaded_program(data):
    from .metal import _lent
    owner = LocalOwner((data,))
    mesh.check(_lent(data, lambda release, lent: mesh.LIB.mesh_local_load(data, len(data), release, lent,
                                                                           C.byref(owner.local)),
                     lambda status: status == 0))
    local = owner.local
    return Solved(local.function, local, local.planned, *viewed(owner, local), (), (), (), None, local.root,
                  (local.use_first, local.use), local.calls, None, None, (), None, owner)

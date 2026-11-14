# design/compiler-archive-assignment-2026-11-13.md; design/compilation-composition-change-request.md items 2 to 7;
# torch 2.14 torch/fx/experimental/proxy_tensor.py make_fx, torch/_inductor/decomposition.py select_decomp_table,
# torch/_inductor/compile_fx.py compile_fx_forward and compile_fx_inner, torch/_inductor/package/package.py
# package_aoti
import dataclasses
import functools
import itertools
import operator
import pathlib
import shutil
import tempfile

import torch
from torch._guards import TracingContext, detect_fake_mode, tracing
from torch._inductor import config
from torch._inductor.codecache import code_hash
from torch._inductor.compile_fx import compile_fx_forward, compile_fx_inner, create_compiler_config_extra
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.package import package_aoti
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .ir import RECV, SEND, channel_of, mesh_graph

THREAD, MODEL = "host", "model"
OPTIONS = {"aot_inductor.package": True, "layout_optimization": False, "reorder_for_peak_memory": False,
           "cpp_wrapper": True}
ATEN = torch.ops.aten


# compiler/mesh_compiler/schedule.py assemble, module.meta["writers"], Declarations.wires; rdma/mesh_graph.py
# dependences_views: the logical copies of a scheduled body's declared transfers, before any plan. `wires`: channel ->
# (declared transfer, logical wire value, its logical bytes, declared sent view, declared landed view). Each distinct
# (source, wire value, sent view, read-after-write writers) is one write of its members, the SENDs sharing it; `overlaps`
# the groups whose sent views intersect. Consumed by fanout (Inductor) and compiler/mesh_compiler/lower_metal.py
# construct (the typed program)
@dataclasses.dataclass
class Copies:
    sites: list
    sends: list
    values: list
    group: dict
    groups: list
    index: dict
    member: dict
    overlaps: dict


def copies(module, wires):
    nodes = list(module.graph.nodes)
    sites = [node for node in nodes if channel_of(node) in wires]
    values = list(dict.fromkeys(wires[channel_of(node)][1] for node in sites))
    writes, sends = {}, [node for node in sites if node.target is SEND]
    for writer, reader in module.meta["writers"]:
        writes.setdefault(reader, set()).add(writer)
    group = {}
    for node in sends:
        _, value, _, (offset, element, pairs), _ = wires[channel_of(node)]
        group[node] = (node.args[0], value, offset, element, tuple(map(tuple, pairs)),
                       frozenset(writes.get(node, ())))
    groups = list(dict.fromkeys(group.values()))
    index = {key: k for k, key in enumerate(groups)}
    edges = mesh_graph.dependences_views(len(groups), [(value, offset, element, pairs, k, 1) for k, (
        _, value, offset, element, pairs, _) in enumerate(groups)]).edges() if groups else []
    overlaps = {k: {k} for k in range(len(groups))}
    for a, b in edges:
        overlaps[a].add(b)
        overlaps[b].add(a)
    member = {}
    for node in sends:
        member.setdefault(index[group[node]], []).append(node)
    return Copies(sites, sends, values, group, groups, index, member, overlaps)


# compiler/mesh_compiler/schedule.py assemble, module.meta["dependences"]; rdma/mesh_graph.py postorder: the solved
# plan's phase of the fanout over `copies` (`effects`: driver.lowered's, the declared transfer, wire value, its solved copy
# offsets, its bytes, the declared views and the plan's supplied and completion representatives). A supplied
# representative is one publication after the writes it covers; a (completion, copies) wait precedes each write it
# intersects; `order` is the DAG of nodes, waits, writes and publications in its emitted order. Consumed by body
@dataclasses.dataclass
class Fanout:
    sites: list
    sends: list
    solved: dict
    values: list
    first: dict
    group: dict
    groups: list
    index: dict
    supplied: dict
    completion: dict
    member: dict
    owner: dict
    affected: dict
    exposed: dict
    order: list


def fanout(module, effects):
    nodes = list(module.graph.nodes)
    made = copies(module, {channel: (e[0], e[1], e[3], e[4], e[5]) for channel, e in effects.items()})
    sites, sends, values, group, groups = made.sites, made.sends, made.values, made.group, made.groups
    index, member, overlaps = made.index, made.member, made.overlaps
    solved = {value: offsets for _, value, offsets, *_ in effects.values()}
    first = dict(zip(values, itertools.accumulate((len(solved[value]) for value in values), initial=0)))
    before = {}
    for a, b in module.meta["dependences"]:
        before.setdefault(b, []).append(a)
    supplied = {node: effects[channel_of(node)][6] for node in sends}
    completion = {node: effects[channel_of(node)][7] for node in sends}
    units, outlets, completions, owner = {}, {}, {}, {}
    for node in sends:
        units.setdefault(supplied[node], set()).add(index[group[node]])
        outlets.setdefault(index[group[node]], set()).add(supplied[node])
        completions.setdefault(index[group[node]], {})[completion[node]] = None
        owner[completion[node]] = supplied[node]
    completions = {k: [c for c in found if c is not None] for k, found in completions.items()}
    item = lambda u: ("publish", supplied[u]) if u in group else ("node", u)
    affected, waits = {}, {}
    for k, (_, value, *_) in enumerate(groups):
        for c in dict.fromkeys(c for i in sorted(overlaps[k]) for c in completions[i]):
            affected.setdefault((c, len(solved[value])), {})[k] = None
            waits.setdefault(k, []).append((c, len(solved[value])))
    preds, items = {}, []
    for node in (n for n in nodes if n.op not in ("placeholder", "output")):
        if node not in group:
            preds[item(node)] = [item(u) for u in dict.fromkeys([*node.all_input_nodes, *before.get(node, ())])
                                 if u.op != "placeholder"]
            items.append(item(node))
            continue
        k = index[group[node]]
        if member[k][0] is node:
            for key in waits.get(k, ()):
                items += [("sent", key)] * (("sent", key) not in preds)
                preds["sent", key] = []
            preds["write", k] = [("sent", key) for key in waits.get(k, ())] + [item(u) for u in dict.fromkeys([
                groups[k][0], *(u for n in member[k] for u in before.get(n, ()))]) if u.op != "placeholder" and not (
                u in group and supplied[u] in outlets[k])]
            items.append(("write", k))
        if [n for n in sends if supplied[n] == supplied[node]][-1] is node:
            preds[item(node)] = [("write", j) for j in sorted({j for g in units[supplied[node]] for j in overlaps[g]
                                                                if groups[j][5] == groups[g][5]})]
            items.append(item(node))
    exposed = {}
    for s in units:
        for _, j in preds["publish", s]:
            exposed.setdefault(j, []).append(s)
    order = mesh_graph.postorder(items, items[::-1], preds.__getitem__)
    return Fanout(sites, sends, solved, values, first, group, groups, index, supplied, completion, member, owner,
                  affected, exposed, order)


# The scheduled body with the caller's immutable constants as buffers of the compiled model, every node keeping its
# original metadata, and each surviving SEND/RECV as fixed native effect sites (rdma/mesh_aoti.py START, WAIT, SENT,
# PUBLISH) on its typed view of the one arena input: copy e % copies of its value, at the offset the plan solved
# (Plan.offsets, a constant table of the model) plus its declared start. Each emitted site is one task of the record:
# RECV maps to its WAIT and SEND to the PUBLISH of its supplied representative for the scheduled dependences
# contracted through the numerical nodes (rdma/mesh_graph.py contract); START precedes every site (M. Lam, PLDI 1988,
# section 2.3). `start` is the plan's retained START object on this machine, or None: the invocation input and START
# exist exactly when it does. compiler/mesh_compiler/schedule.py assemble, module.meta["writers"]; solved,
# Plan.publications; rdma/mesh_graph.py dependences_views, postorder, contract, quotient: a copy is a SEND's source node,
# Plan value, sent view and read-after-write writer causes; a PUBLISH is a supplied representative, after each copy of
# the same writer causes whose view intersects one of its transfers'; one SENT(c, copies) per completion
# representative c and its value's copies precedes every copy it intersects; the output follows every emitted operation
def body(module, constants, tensors, effects, start, extent):
    import mesh_aoti as native
    root, graph, held, examples = torch.nn.Module(), torch.fx.Graph(), {}, []
    tasks, operations, dependences, emitted, views = [], [], [], {}, {}
    nodes = list(module.graph.nodes)
    carried = lambda node: (node.args[0] if node.target is SEND else node).meta["val"]
    fan = fanout(module, effects)
    sites, sends, solved, values, first = fan.sites, fan.sends, fan.solved, fan.values, fan.first
    group, groups, index, supplied, completion = fan.group, fan.groups, fan.index, fan.supplied, fan.completion
    member, owner, affected, exposed, order = fan.member, fan.owner, fan.affected, fan.exposed, fan.order
    made, published, waited = {}, {}, {}

    def site(call, primitive, key, ordinal, *sources):
        tasks.append((THREAD, (call.name, *(source.name for source in sources))))
        operations.append((THREAD, primitive, len(tasks) - 1, key, 1, ordinal))
        return len(tasks) - 1

    def view_of(k):
        if k not in made:
            made[k] = typed(member[k][0])
        return made[k]

    def publish(s):
        members = [node for node in sends if supplied[node] == s]
        published[s] = site(graph.call_function(native.PUBLISH, (invocation, s, made[index[group[members[0]]]])),
                            "host_publish", ("publication", s), 1, *members)
        emitted.update((node, published[s]) for node in members)

    def typed(node):
        _, value, offsets, size, sent, landed, *_ = effects[channel_of(node)]
        offset, element, pairs = sent if node.target is SEND else landed
        sizes = [n for n, _ in pairs] if node.target is SEND else list(carried(node).shape)
        residue = graph.call_function(ATEN.remainder.Scalar, (execution, len(offsets)))
        at = graph.call_function(ATEN.add.Tensor, (graph.call_function(
            ATEN._to_copy.default, (residue,), {"dtype": torch.int64}), first[value]))
        byte = graph.call_function(ATEN._local_scalar_dense.default, (
            graph.call_function(ATEN.index.Tensor, (table, [at])),))
        graph.call_function(ATEN.sym_constrain_range_for_size.default, (byte,), {"min": 0, "max": extent - size})
        return graph.call_function(ATEN.as_strided.default, (
            graph.call_function(ATEN.view.dtype, (arena, carried(node).dtype)), sizes,
            [stride // element for _, stride in pairs],
            graph.call_function(operator.floordiv, (graph.call_function(operator.add, (byte, offset)), element))))

    for node in (n for n in nodes if n.op == "placeholder"):
        if node.target in constants:
            name = f"constant{len(held)}"
            root.register_buffer(name, constants[node.target])
            held[node] = graph.get_attr(name)
            held[node].meta.update(node.meta)
        else:
            held[node] = graph.node_copy(node)
            examples.append(node.meta["val"])
    # torch/_export/non_strict_utils.py make_fake_inputs: the export fake mode, when the body carries none of its own
    mode = detect_fake_mode([node.meta.get("val") for node in nodes]) or FakeTensorMode(
        shape_env=ShapeEnv(tracked_fakes=[], trace_asserts=True), allow_non_fake_inputs=True, export=True)
    if start is not None:
        invocation = graph.placeholder("invocation")
        examples.append(mode.from_tensor(native.INVOCATION))
        invocation.meta["val"] = examples[-1]
    if sites:
        arena = graph.placeholder("arena")
        with mode:
            examples.append(torch.empty(extent, dtype=torch.uint8, device="cpu"))
        arena.meta["val"] = examples[-1]
        root.register_buffer("offsets", torch.tensor([o for value in values for o in solved[value]], dtype=torch.int64))
        table = graph.get_attr("offsets")
        execution = graph.call_function(ATEN.select.int, (invocation, 0, 2))
    if start is not None:
        started = site(graph.call_function(native.START, (invocation,)), "bridge_start", start, 1)
    for kind, of in order:
        if kind == "sent":
            c, copies = of
            waited[of] = site(graph.call_function(native.SENT, (invocation, c, copies, view_of(next(iter(
                affected[of]))))), "host_wait", ("completion", c), 1 - copies, *(n for n in sends if completion[n] == c))
        elif kind == "write":
            source, _, _, _, pairs, _ = groups[of]
            graph.call_function(ATEN.copy_.default, (view_of(of), graph.call_function(ATEN.as_strided.default, (
                held[source], [n for n, _ in pairs], list(source.meta["val"].stride())))))
        elif kind == "publish":
            publish(of)
        elif of.target is RECV:
            transfer, *_ = effects[channel_of(of)]
            views[of] = typed(of)
            emitted[of] = site(graph.call_function(native.WAIT, (invocation, transfer, 0, views[of])),
                               "host_wait", ("arrival", transfer), 1, of)
            held[of] = views[of]
        else:
            held[of] = graph.node_copy(of, held.__getitem__)
    output = module.graph.output_node()
    graph.output(tuple(held[value] for value in tensors)).meta.update(output.meta)
    for (c, copies), task in waited.items():
        dependences += [(published[owner[c]], task, copies)] + [
            (task, published[s], 0) for k in affected[c, copies] for s in exposed[k]]
    vertices = list(dict.fromkeys([end for pair in module.meta["dependences"] for end in pair] + sites))
    at = {node: k for k, node in enumerate(vertices)}
    contracted = mesh_graph.contract([node in emitted for node in vertices],
                                     [(at[a], at[b]) for a, b in module.meta["dependences"]])
    dependences += [(a, b, 0) for a, b in mesh_graph.quotient(
        [emitted.get(node, len(tasks) + k) for k, node in enumerate(vertices)], contracted)]
    dependences += [(started, k, 0) for k in range(1, len(tasks))]
    entry = [(THREAD, tuple(node.name for node in nodes if node.op == "call_function"))]
    record = {"tasks": tasks or entry, "dependences": sorted(set(dependences)), "operations": operations,
              "publications": [(effects[channel_of(node)][0], emitted[node]) for node in sends]}
    return torch.fx.GraphModule(root, graph), examples, bool(sites), record, mode


# compiler/mesh_compiler/schedule.py assemble: one composed body per (machine, target, device) group, compiled once
# through Inductor's own forward compilation (no second AOT autograd pass), so the body's copies and effects stay
# mutations of their inputs; effects: channel -> (declared transfer, its value, the value's solved copy offsets in the
# arena, its byte extent, declared sent view, declared landed view, supplied and completion representatives of
# Plan.publications); extent: the arena's bytes (Plan.arena_bytes)
def lower(module, *, constants, effects, start, extent):
    import mesh_aoti as native
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    given = [node for node in placeholders if node.target not in constants]
    leaves = tuple(next(node for node in module.graph.nodes if node.op == "output").args[0] or ())
    tensors = list(dict.fromkeys(leaf for leaf in leaves if isinstance(leaf, torch.fx.Node)))
    results = tuple(("output", tensors.index(leaf)) if isinstance(leaf, torch.fx.Node) else ("constant", leaf)
                    for leaf in leaves)
    made, examples, wired, record, mode = body(module, constants, tensors, effects, start, extent)
    traced = make_fx(made, decomposition_table=select_decomp_table(), tracing_mode="fake")(*examples)
    options = {**OPTIONS, "aot_inductor.custom_ops_to_c_shims": native.SHIMS,
               "aot_inductor.output_path": code_hash(traced.code)}
    with (torch.no_grad(), config.patch(options), tracing(TracingContext(mode)), V.set_fake_mode(mode),
          V.set_aot_compilation(True)):
        compiled = compile_fx_forward(traced, examples, num_orig_model_outputs=len(tensors),
                                      num_example_inputs=len(examples),
                                      compiler_config_extra=create_compiler_config_extra(traced),
                                      inner_compile=functools.partial(compile_fx_inner, cpp_wrapper=True),
                                      is_inference=True)
    directory = tempfile.TemporaryDirectory(prefix="mesh-aoti-")
    package = package_aoti(str(pathlib.Path(directory.name) / "package.pt2"), {MODEL: compiled.filename})
    sources = {node.name: tuple(task) for node, task in module.meta["tasks"].items()}
    invocation = None if start is None else len(given)
    return Target(package, tuple(placeholders.index(node) for node in given), results, invocation,
                  len(given) + (start is not None) if wired else None, tuple(module.meta["members"]), sources,
                  record, directory)


# design/compiler-native-assignment-2026-11-13.md: one native Model of the whole body; its call takes the arguments
# in scheduled.inputs order and returns the body's output tuple; `invocation` and `arena` are the positions of the
# invocation input and of the one arena input, or None
@dataclasses.dataclass
class Target:
    package: str
    arguments: tuple
    results: tuple
    invocation: int
    arena: int
    members: tuple
    sources: dict
    record: dict
    directory: object = None

    def __post_init__(self):
        from mesh_aoti import Model
        inputs = len(self.arguments) + (self.invocation is not None) + (self.arena is not None)
        outputs = len({value for kind, value in self.results if kind == "output"})
        self.model = Model(self.package, MODEL, inputs, outputs)

    def __call__(self, *arguments):
        return self.call(self.model, arguments)

    def call(self, entry, arguments):
        made = entry(*(arguments[position] for position in self.arguments))
        return tuple(made[value] if kind == "output" else value for kind, value in self.results)

    # compiler/mesh_compiler/bridge.py prepare, Published.arena; rdma/mesh_aoti.py Model.prepare
    def bind(self, published):
        return self if self.invocation is None else Bound(self, self.model.prepare(
            list(range(len(self.arguments))), [(self.arena, [published.arena])] * (self.arena is not None),
            self.invocation, published.context, published.plan, published.execution), published)

    def saved(self, directory, group):
        name = f"{group}.aoti.pt2"
        shutil.copyfile(self.package, pathlib.Path(directory) / name)
        return {"package": name, "arguments": self.arguments, "results": self.results, "invocation": self.invocation,
                "arena": self.arena, "members": self.members, "sources": self.sources, "record": self.record}


class Bound:
    def __init__(self, target, prepared, published):
        self.target, self.prepared, self.published = target, prepared, published

    def __call__(self, *arguments):
        return self.target.call(self.prepared, arguments)


def restore(saved, directory):
    record = {**saved["record"], "operations": [[*operation[:3], tuple(operation[3]) if isinstance(operation[3], list)
                                                 else operation[3], *operation[4:]]
                                                for operation in saved["record"]["operations"]]}
    return Target(str(pathlib.Path(directory) / saved["package"]), tuple(saved["arguments"]),
                  tuple(map(tuple, saved["results"])), saved["invocation"], saved["arena"], tuple(saved["members"]),
                  {name: tuple(task) for name, task in saved["sources"].items()}, record)

# design/data-flow.md; design/aot-compilation-change-request.md A8 to A10 and appendix item 6: the archive of one
# completed machine program (its native target, joint result and recipe), with the captured program's per-function
# records as explicit typed records indexed by function ordinal (the completed target's body order, CallAnalysis.keys);
# a captured function's key, which may be a callable, is compilation-only and is not saved. Scheduled.plan is the one
# target's joint owner's own result and Scheduled.signature that target's complete signature archive
# (compiler/mesh_compiler/native.py Target.plan and Target.signature, saved and restored with the target;
# compiler/mesh_compiler/driver.py compiled), kept in no record of their own here
import collections
import dataclasses
from copy import deepcopy
import inspect
import json
import operator
import pathlib

import torch
from torch._export.passes.lift_constants_pass import ConstantAttrMap, _get_first_fqn, lift_constants_pass
from torch._export.verifier import TrainingIRVerifier
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import ExportedProgram
from torch.export.exported_program import ModuleCallEntry, ModuleCallSignature
from torch.export._unlift import _unlift_inputs_as_getattr
from torch.export.graph_signature import (ConstantArgument, ExportGraphSignature, InputKind, InputSpec, OutputKind,
                                          OutputSpec, TensorArgument)
from torch.fx._graph_pickler import GraphPickler, Options
from torch.multiprocessing.reductions import StorageWeakRef
from torch.fx.graph_module import _assign_attr, _get_attr
from torch.utils import _pytree as pytree

from . import native, schedule
from .ir import Call, CapturedProgram, Exported, Scheduled


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


# Each Mesh-produced node annotation: ir.py evaluated, based and views (processor, view, out, to, invariant), export.py
# read and capture (state, site), placement.py (landing, work_axes, of, order, pairs, units, cut, blocks,
# collectives, stated, constant; insert_send_recv's consumer, collective, dimension, block, counts, displs and
# strides), decompose.py's kept fields and an output's destination; and `val`, whose torch.export TensorMeta keeps no
# storage identity. torch 2.14.0 torch/_export/serde/serialize.py serialize_metadata keeps stack_trace,
# nn_module_stack, source_fn_stack, torch_fn, custom and from_node itself
NODE = ("val", "processor", "view", "out", "to", "invariant", "state", "site", "landing", "work_axes", "of", "order",
        "pairs", "units", "cut", "blocks", "collectives", "stated", "constant", "consumer", "collective", "dimension",
        "block", "counts", "displs", "strides", "destination")
# ir.py based and views (schedule.py wrote, accesses): a view, out or to record's root, and an output destination's
# placeholder, by its own graph's node name
ROOTED = ("view", "out", "to")
# each module annotation: placement.py place, devices.py on_devices and decompose.py (transfers, input_parts,
# decomposed, invariant; _outlined's outline: the placed graph, its eigh, the adjacency Graph and its Reached search)
# and schedule.py assemble (a body's own records, its source GraphModule and SourceRelation)
MODULE = ("transfers", "input_parts", "decomposed", "invariant", "outline", "members", "tasks", "dependences",
          "writers", "received_strides", "order", "copied", "output", "vertices", "group", "source", "relation")

# a retained reference by its graph owner: `owner` its saved graph's index ("source" the outer source graph, `path` a
# captured function's GraphModule in it), `path` a nested branch GraphModule's
# attribute path in it ("" the owner itself), `node` a Node's name (None: the GraphModule itself), a getitem's `node`
# its producer's Ref and `index` its result index (the serializer's own result correspondence); `Related` a
# SourceRelation, `Graphed` an rdma/mesh_graph.py Graph and `Searching` a Reached search by saved index (each object
# once; a branch GraphModule or tensor attribute several graphs share, schedule.py assemble's GraphModule(owner, graph),
# by its first owner's path and its one saved tensor)
Ref = collections.namedtuple("Ref", "owner path node index")
Related = collections.namedtuple("Related", "index")
Graphed = collections.namedtuple("Graphed", "index")
Searching = collections.namedtuple("Searching", "index")


# torch 2.14.0 torch/_export/serde/serialize.py serialize_hoo_inputs and deserialize_input `as_graph` (a higher-order
# operator's GraphModule operand saved by and registered under its attribute target): an owner's nested branch
# GraphModules (decompose.py _branch_owned) by attribute path
def branches(module, prefix=""):
    for node in module.graph.nodes:
        held = _get_attr(module, node.target) if node.op == "get_attr" else None
        if isinstance(held, torch.fx.GraphModule):
            yield prefix + node.target, held
            yield from branches(held, prefix + node.target + ".")


# a node's own key in its graph: its name, a getitem's its producer's key and its result index
def produced(node):
    return (produced(node.args[0]), node.args[1]) if node.target is operator.getitem else node.name


# a value with each Node, GraphModule, SourceRelation, Graph and Reached it holds made a Ref, Related, Graphed or
# Searching (`held`: the owners by saved index, their nested branches, the relations, the Graphs; each registered once,
# an owner with its branches), everything else, a FakeTensor among them, as it is (torch/fx/_graph_pickler.py
# GraphPickler: no raw Node)
def encoded(held, value):
    owners, nested, relations, natives, searches, _, sources = held
    if isinstance(value, dict):
        return {encoded(held, k): encoded(held, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return (tuple if isinstance(value, tuple) else list)(encoded(held, v) for v in value)
    if isinstance(value, schedule.SourceRelation):
        return Related(relations.setdefault(value, len(relations)))
    if isinstance(value, schedule.mesh_graph.Reached):
        return Searching(searches.setdefault(id(value), (len(searches), value))[0])
    if isinstance(value, schedule.mesh_graph.Graph):
        return Graphed(natives.setdefault(value, len(natives)))
    if isinstance(value, torch.fx.Node) and value.target is operator.getitem:
        return Ref(None, None, encoded(held, value.args[0]), value.args[1])
    if not isinstance(value, (torch.fx.Node, torch.fx.GraphModule)):
        return value
    module = value.graph.owning_module if isinstance(value, torch.fx.Node) else value
    if module in sources:
        return Ref("source", sources[module], value.name if isinstance(value, torch.fx.Node) else None, None)
    if module not in nested and module not in owners:
        owners[module] = len(owners)
        for path, branch in branches(module):
            nested.setdefault(branch, (owners[module], path))
    owner, path = nested[module] if module in nested else (owners[module], "")
    return Ref(owner, path, value.name if isinstance(value, torch.fx.Node) else None, None)


# schedule-emit wrap's copy.deepcopy: torch 2.14.0 torch/fx/graph_module.py GraphModule.__deepcopy__ (every nested
# GraphModule once, memo) and torch/fx/graph.py Graph.__deepcopy__ (node_copy: each node its own meta dict over the
# same values), each module's meta a fresh one and each tensor attribute the original object by the memo
def copied(module):
    memo = {}
    for m in module.modules():
        if isinstance(m, torch.fx.GraphModule):
            memo[id(m.meta)] = {}
            memo.update((id(t), t) for n in m.graph.nodes if n.op == "get_attr"
                        for t in [_get_attr(m, n.target)] if isinstance(t, torch.Tensor))
    return deepcopy(module, memo)


# schedule-emit/schedule_emit/archive.py wrap; torch 2.14.0 torch/export/exported_program.py ExportedProgram,
# torch/_export/verifier.py TrainingIRVerifier (its dialect admits a mutating operator, an out= variant's) and
# torch/_export/passes/lift_constants_pass.py lift_constants_pass: a graph owner as an ExportedProgram around a copy
# of its graph (each node's own metadata, none of the module's), no retracing; its signature read off its placeholders
# (each a user input, a non-tensor one its ConstantArgument; their tensors state.pt's or the caller's) and its output
# (an empty result one None), each tensor attribute it reads lifted once at its own target (ConstantAttrMap, shared
# by its aliases) as a CONSTANT_TENSOR input with its real data; with each tensor get_attr's (name, target lifted, its
# tensor's index in `held`'s tensors, each tensor object once: state.pt's one torch.save payload keeps its storage)
def wrapped(module, held):
    copy = copied(module)
    graph, output, tensors = copy.graph, copy.graph.output_node(), held[5]
    if not output.args[0]:
        output.args = ((None,),)
    inputs = [InputSpec(InputKind.USER_INPUT, TensorArgument(name=n.name)
                        if isinstance(n.meta.get("val"), torch.Tensor)
                        else ConstantArgument(name=n.name, value=n.meta.get("val")), None)
              for n in graph.nodes if n.op == "placeholder"]
    results, users = output.args[0], len(inputs)
    outputs = [OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=r.name) if isinstance(r, torch.fx.Node)
                          else ConstantArgument(name="", value=r), None) for r in results]
    signature, attributes = ExportGraphSignature(inputs, outputs), ConstantAttrMap()
    read = [n for n in graph.nodes if n.op == "get_attr" and isinstance(_get_attr(copy, n.target), torch.Tensor)]
    for node in read:
        # this flat produced graph's own (empty) module hierarchy where its copied get_attr records none
        # (lift_constants_pass's nn_module_stack), the producer's kept where it does
        node.meta.setdefault("nn_module_stack", {})
        attributes.add(_get_attr(copy, node.target), node.target)
    lifted = [(node.name, _get_first_fqn(attributes, value), tensors.setdefault(id(value), (len(tensors), value))[0])
              for node in read for value in [_get_attr(copy, node.target)]]
    constants = lift_constants_pass(copy, signature, attributes)
    calls = [ModuleCallEntry("", ModuleCallSignature([], [], pytree.tree_flatten((tuple(range(users)), {}))[1],
                                                     pytree.tree_flatten(tuple(range(len(results))))[1]))]
    return ExportedProgram(copy, graph, signature, {}, {}, calls, constants=constants,
                           verifiers=[TrainingIRVerifier]), lifted


# a graph owner's own records (what the serializer does not keep): its placeholders' targets, each node's Mesh
# annotations (NODE) by its key, its module annotations (MODULE), whether its output is empty, and its nested
# branches' node and module annotations by path, each value `encoded`
def annotated(held, module):
    def own(m):
        names = {n.name: n for n in m.graph.nodes}

        # a ROOTED record's root and a destination's placeholders as their Nodes
        def typed(k, v):
            if k in ROOTED and v is not None:
                return (names[v[0]], *v[1:])
            return {position: names[name] for position, name in v.items()} if k == "destination" else v
        return {"targets": [n.target for n in m.graph.nodes if n.op == "placeholder"],
                "empty": not m.graph.output_node().args[0],
                "nodes": {produced(n): {k: encoded(held, typed(k, n.meta[k])) for k in NODE if k in n.meta}
                          for n in m.graph.nodes},
                "module": {k: encoded(held, m.meta[k]) for k in MODULE if k in m.meta}}
    owner, kept = held[0][module], list(branches(module))
    return {**own(module), "nested": {path: own(b) for path, b in kept if held[1][b] == (owner, path)},
            "shared": [(path, held[1][b]) for path, b in kept if held[1][b] != (owner, path)]}


# torch 2.14.0 torch/_export/serde/serialize.py generate_getitems (a getitem for every result of a multi-result
# operator, the unused ones too): a loaded graph owner as saved, each lifted constant again its original tensor
# attribute (its real data at its target, read by one get_attr Node its original get_attr names name), each
# placeholder's target put back, each getitem (it or a nested branch) did not hold erased, an empty output restored; no
# solve, compilation or retracing. With the names of its lifted get_attr Nodes
def restored(path, record, tensors):
    program = torch.export.load(path)
    module = program.graph_module
    for _, target, tensor in record["lifted"]:
        _assign_attr(tensors[tensor], module, target)
    # torch 2.14.0 torch/export/_unlift.py _unlift_inputs_as_getattr: each CONSTANT_TENSOR placeholder its get_attr
    attributes, _ = _unlift_inputs_as_getattr(module, [spec.target if spec.kind == InputKind.CONSTANT_TENSOR else None
                                                       for spec in program.graph_signature.input_specs])
    for held, kept in ((module, record), *((_get_attr(module, p), r) for p, r in record["nested"].items())):
        targets = iter(kept["targets"])
        for node in held.graph.nodes:
            if node.op == "placeholder":
                node.target = next(targets)
        for node in reversed(list(held.graph.nodes)):
            if node.target is operator.getitem and produced(node) not in kept["nodes"]:
                held.graph.erase_node(node)
        if kept["empty"]:
            held.graph.output_node().args = ((),)
    return module, {name: attributes[target] for name, target, _ in record["lifted"]}


# a node by its key among `names` (its graph's nodes by name), a getitem the one user of its producer at its index
def found(names, key):
    if isinstance(key, str):
        return names[key]
    parent = found(names, key[0])
    return next(u for u in parent.users if u.target is operator.getitem and u.args[1] == key[1])


# an `encoded` value with each Ref bound to its loaded graph owner's actual Node or GraphModule (`loaded`: the owners,
# their and their nested branches' nodes by name, the SourceRelations and the Graphs), each Related and Graphed to its
# loaded one and each Searching to its search loaded over that Graph
def decoded(loaded, value):
    graphs, named, relations, natives, searches = loaded
    if isinstance(value, Ref):
        if value.index is not None:
            parent = decoded(loaded, value.node)
            return next(u for u in parent.users if u.target is operator.getitem and u.args[1] == value.index)
        if value.node is None:
            return _get_attr(graphs[value.owner], value.path) if value.path else graphs[value.owner]
        return named[value.owner, value.path][value.node]
    if isinstance(value, Related):
        return relations[value.index]
    if isinstance(value, Graphed):
        return natives[value.index]
    if isinstance(value, Searching):
        return searches[value.index]
    if isinstance(value, dict):
        return {decoded(loaded, k): decoded(loaded, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return (tuple if isinstance(value, tuple) else list)(decoded(loaded, v) for v in value)
    return value


# design/aot-compilation-change-request.md A10; schedule.py analyse: a CallAnalysis's query results in their saved
# form, its incoming search's arrays (over the calls' relation, which the joint result keeps), its paths' roots and
# search (a Reached over their own Graph), its widths' counts and per-member chains by function ordinal, its supplied
# causes, accesses and generated state
def queried(analysis, ordinal):
    paths = analysis.paths
    return {"search": schedule.mesh_graph.Reached(analysis.search).save(),
            "paths": (list(paths.items), schedule.mesh_graph.Reached(paths.owner)),
            "widths": {ordinal[key]: (count, tuple(chain)) for key, (count, chain) in analysis.widths.items()},
            "supplied": analysis.supplied, "accesses": analysis.accesses, "generated": analysis.generated}


# design/aot-compilation-change-request.md A10; schedule.py call_operands: the compilation's source record. Its outer
# source graph a copy (`copied`) whose call nodes (CapturedProgram.calls) are call_module references to their
# functions' captured GraphModules by ordinal (torch.fx GraphModule over a root of just those: no user module or
# callable kept; a call node's capture key, compilation only, dropped); each function's Exported records (its state
# tensors `held`'s), each call's Call, the CallOperands fields (a storage by its holder Node, `existing` by the outer
# placeholders holding it) and the `query`, each `encoded`, the outer graph's and functions' Nodes as "source" Refs
def sourced(held, ordinal, operands, query):
    captured, tensors, sources = operands.captured, held[5], held[6]
    outer, memo = captured.exported.graph_module, {}
    graph = deepcopy(outer.graph, memo)
    for node, call in captured.calls.items():
        memo[node].op, memo[node].target = "call_module", f"function_{ordinal[call.function]}"
        memo[node].meta.pop("function", None)
    module = copied(torch.fx.GraphModule(
        {**{n.target: _get_attr(outer, n.target) for n in graph.nodes if n.op == "get_attr"},
         **{f"function_{ordinal[key]}": f.graph_module for key, f in captured.functions.items()}}, graph))
    sources.update({outer: "", **{f.graph_module: f"function_{ordinal[key]}" for key, f in captured.functions.items()}})
    existing = [n for n in outer.graph.nodes if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
                and StorageWeakRef(n.meta["val"].untyped_storage()) in operands.existing]
    return {"graph": module,
            "functions": {ordinal[key]: ({name: (target, tensors.setdefault(id(t), (len(tensors), t))[0])
                                          for name, (target, t) in f.state.items()}, f.tokens, f.user_inputs,
                                         f.outputs, pytree.treespec_dumps(f.out_spec), f.state_kinds, f.specialized)
                          for key, f in captured.functions.items()},
            "calls": encoded(held, [(node, ordinal[call.function], call.inputs, call.outputs)
                                    for node, call in captured.calls.items()]),
            "operands": encoded(held, (
                operands.nodes, [holder for _, holder in operands.storages], operands.views, operands.accesses,
                operands.results, operands.root_sources, sorted(operands.held),
                [tuple(getattr(port, f.name) for f in dataclasses.fields(port)) for port in operands.ports], existing,
                operands.produced, {ordinal[key]: placed for key, placed in operands.placements.items()},
                operands.machines, operands.output_processor)),
            "query": encoded(held, query)}


# a loaded local joint result's query (compiler/mesh_compiler/schedule.py loaded_program, mesh_local_load): its
# calls' relation and closure the loaded mesh_local's own (borrowed: Relation.borrowed, an Owned its owner frees), the
# incoming search loaded once over that Graph, the paths' View over their loaded search, with the saved `query`
def attached(query, local, owner):
    mesh_graph = schedule.mesh_graph
    graph = mesh_graph.Graph(range(local.ncalls), mesh_graph.Relation.borrowed(local.relation))
    roots, reached = query["paths"]
    owned, found = reached.owned, reached.owned.made.contents
    return {"query": query, "graph": graph, "search": mesh_graph.Reached.load(query["search"], graph).owned,
            "closure": mesh_graph.Closures(mesh_graph.Owned(local.closure, lambda made: None, graph, (owner,)), False),
            "paths": mesh_graph.View(owned, roots, range(len(roots)), lambda k: mesh_graph.ordered(
                owned, found.preorder, range(found.place[k], found.end[k])))}


def save(directory, scheduled):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported, (target,) = scheduled.exported, scheduled.functions.values()
    ordinal = {body.key: k for k, body in enumerate(target.bodies)}
    signature, spec = exported.interface
    # every body by its graph owner; then each owner (each graph a Node reference names) and each SourceRelation once
    # (rdma/mesh_graph.py Relation.save, mesh-graph.h mesh_graph_relation_save), the annotations one GraphPickler
    # payload (torch/fx/_graph_pickler.py: one MetaTensorDescriber, so FakeTensor duplicates and views keep their storage)
    held = ({}, {}, {}, {}, {}, {}, {})
    # the source record from the original analysis (Target.solution.analysis, its operands), or a loaded
    # compilation's retained one (its joint owner's `source`, Scheduled.operands)
    joint = target.solution
    analysis = joint.analysis if isinstance(joint, schedule.Solved) else None
    query, operands = ((queried(analysis, ordinal), analysis.operands) if analysis is not None else
                       (joint.source["query"], scheduled.operands))
    record = sourced(held, ordinal, operands, query)
    bodies = [[kind, ordinal[key], name, encoded(held, module).owner]
              for kind, kept in (("machine", scheduled.modules), ("processor", scheduled.processor_modules))
              for key, modules in kept.items() for name, module in modules.items()]
    graphs, sources, natives, searches = [], [], [], []
    while (len(graphs), len(sources), len(natives), len(searches)) != tuple(map(len, (held[0], *held[2:5]))):
        if len(graphs) < len(held[0]):
            module = list(held[0])[len(graphs)]
            program, lifted = wrapped(module, held)
            torch.export.save(program, directory / f"graph.{len(graphs)}.pt2")
            graphs.append({**annotated(held, module), "lifted": lifted})
        elif len(sources) < len(held[2]):
            source = list(held[2])[len(sources)]
            sources.append((source.relation.save(), *encoded(held, (source.functions, source.supplied,
                                                                    source.accesses, source.producers))))
        elif len(natives) < len(held[3]):
            graph = list(held[3])[len(natives)]
            natives.append((graph.relation.save(), encoded(held, graph.vertices)))
        else:
            reached = list(held[4].values())[len(searches)][1]
            searches.append((encoded(held, reached.owned.graph), reached.save()))
    # torch/fx/_graph_pickler.py Options: every operator of the captured bodies pickled by name (ops_filter None)
    (directory / "annotations.pkl").write_bytes(GraphPickler.dumps(
        {"graphs": graphs, "relations": sources, "natives": natives, "searches": searches, "source": record},
        Options(ops_filter=None)))
    torch.save({"state": {name: (key, tensor.detach()) for name, (key, tensor) in exported.state.items()},
                "attributes": [tensor.detach() for _, tensor in held[5].values()],
                "defaults": {name: parameter.default for name, parameter in signature.parameters.items()
                             if parameter.default is not inspect.Parameter.empty},
                "specialized": exported.specialized}, directory / "state.pt")
    (directory / "scheduled.json").write_text(json.dumps({
        "output_processor": scheduled.output_processor,
        "processors": {processor: str(device) for processor, device in scheduled.devices.items()},
        "machines": scheduled.machines,
        "engines": scheduled.engines,
        "inputs": scheduled.inputs,
        "channels": [{"function": ordinal[key], "channels": count} for key, count in scheduled.channels.items()],
        "feedforward": scheduled.feedforward,
        "stored": scheduled.stored,
        "invariant": [{"function": ordinal[key], "records": records} for key, records in scheduled.invariant.items()],
        "carried": [{"function": ordinal[key],
                     "transfers": [[channel, sender, receiver, str(dtype).removeprefix("torch."), shape, stride]
                                   for channel, sender, receiver, dtype, shape, stride in transfers]}
                    for key, transfers in scheduled.carried.items()],
        "targets": {processor: function.saved(directory, processor)
                    for processor, function in scheduled.functions.items()},
        "state_kinds": {name: kind.name for name, kind in exported.state_kinds.items()},
        "tokens": exported.tokens,
        "user_inputs": exported.user_inputs,
        "outputs": [[kind.name, key] for kind, key in exported.outputs],
        "out_spec": pytree.treespec_dumps(exported.out_spec),
        "interface": {"parameters": [[name, parameter.kind.name] for name, parameter in signature.parameters.items()],
                      "spec": pytree.treespec_dumps(spec)},
        "bodies": bodies,
    }))


def load(directory):
    directory = pathlib.Path(directory)
    about = json.loads((directory / "scheduled.json").read_text())
    payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    state, defaults, specialized = payload["state"], payload["defaults"], payload["specialized"]
    outputs = [(OutputKind[kind], key) for kind, key in about["outputs"]]
    signature = inspect.Signature([inspect.Parameter(name, getattr(inspect.Parameter, kind),
                                                     default=defaults.get(name, inspect.Parameter.empty))
                                   for name, kind in about["interface"]["parameters"]])
    devices = {processor: torch.device(name) for processor, name in about["processors"].items()}
    functions = {processor: native.restore(saved, directory) for processor, saved in about["targets"].items()}
    (target,) = functions.values()
    carried = {record["function"]: [(channel, sender, receiver, getattr(torch, dtype), shape, stride)
                                            for channel, sender, receiver, dtype, shape, stride in record["transfers"]]
               for record in about["carried"]}
    channels = {record["function"]: record["channels"] for record in about["channels"]}
    # each graph owner and SourceRelation loaded once (rdma/mesh_graph.py Relation.load: no solve) and the one
    # annotation payload (GraphPickler.loads, one FakeTensorMode: its FakeTensors' shared storage as saved), each Ref
    # bound to its actual loaded Node or GraphModule, every body of one source sharing that source's relation and graph
    saved = GraphPickler.loads((directory / "annotations.pkl").read_bytes(), FakeTensorMode())
    tensors, source = payload["attributes"], saved["source"]
    loaded = [restored(directory / f"graph.{g}.pt2", record, tensors) for g, record in enumerate(saved["graphs"])]
    graphs, named, relations, natives, searches = dict(enumerate(m for m, _ in loaded)), {}, [], [], []
    graphs["source"] = outer = source["graph"]
    named["source", ""] = {node.name: node for node in outer.graph.nodes}
    named.update((("source", f"function_{k}"),
                  {node.name: node for node in _get_attr(outer, f"function_{k}").graph.nodes})
                 for k in source["functions"])
    # each shared branch GraphModule (schedule.py assemble's GraphModule(owner, graph)) its first owner's one loaded
    # module at every other owner's attribute path
    for g, record in enumerate(saved["graphs"]):
        for path, (owner, canonical) in record["shared"]:
            _assign_attr(_get_attr(graphs[owner], canonical), graphs[g], path)
    for g, ((module, aliases), record) in enumerate(zip(loaded, saved["graphs"])):
        named[g, ""] = {**{node.name: node for node in module.graph.nodes}, **aliases}
        named.update(((g, p), {node.name: node for node in _get_attr(module, p).graph.nodes})
                     for p in record["nested"])
    resolve = lambda value: decoded((graphs, named, relations, natives, searches), value)
    mesh_graph = schedule.mesh_graph
    natives += [mesh_graph.Graph(resolve(vertices), mesh_graph.Relation.load(data))
                for data, vertices in saved["natives"]]
    searches += [mesh_graph.Reached.load(data, resolve(graph)) for graph, data in saved["searches"]]
    relations += [schedule.SourceRelation(mesh_graph.Relation.load(data), list(resolve(functions)),
                                          list(resolve(supplied)), list(resolve(accesses)), dict(resolve(producers)))
                  for data, functions, supplied, accesses, producers in saved["relations"]]
    for g, record in enumerate(saved["graphs"]):
        for p, kept in (("", record), *record["nested"].items()):
            for key, meta in kept["nodes"].items():
                meta = resolve(meta)
                meta.update((k, (meta[k][0].name, *meta[k][1:])) for k in ROOTED if meta.get(k) is not None)
                if "destination" in meta:
                    meta["destination"] = {position: node.name for position, node in meta["destination"].items()}
                found(named[g, p], key).meta.update(meta)
            (_get_attr(graphs[g], p) if p else graphs[g]).meta.update(resolve(kept["module"]))
        graphs[g].recompile()
    # the source record: the outer source graph (each call a call_module of its function's captured GraphModule), each
    # function's Exported, each call's Call, the CallOperands (Scheduled.operands; a storage by its holder's loaded
    # FakeTensor storage) and the query on the joint owner: a local result's attached now, a network result's kept on
    # its saved owner until its preparation (bridge.Published, schedule.published_program) loads its relation
    exported = Exported(outer, state, about["tokens"], about["user_inputs"], outputs,
                        pytree.treespec_loads(about["out_spec"]),
                        {name: InputKind[kind] for name, kind in about["state_kinds"].items()},
                        interface=(signature, pytree.treespec_loads(about["interface"]["spec"])),
                        specialized=specialized)
    captured = {k: Exported(_get_attr(outer, f"function_{k}"),
                            {name: (key, tensors[t]) for name, (key, t) in kept.items()}, tokens, inputs, results,
                            pytree.treespec_loads(spec), kinds, specialized=literals)
                for k, (kept, tokens, inputs, results, spec, kinds, literals) in source["functions"].items()}
    calls = {node: Call(k, inputs, results) for node, k, inputs, results in resolve(source["calls"])}
    (nodes, holders, views, accesses, results, roots, kept, ports, existing, produced, placements, machines,
     output) = resolve(source["operands"])
    operands = schedule.CallOperands(
        CapturedProgram(exported, captured, calls), nodes, tuple(captured), [calls[node] for node in nodes],
        {k: k for k in captured}, [(StorageWeakRef(h.meta["val"].untyped_storage()), h) for h in holders], views,
        accesses, results, roots, frozenset(kept), tuple(schedule.CallPort(*port) for port in ports),
        {StorageWeakRef(p.meta["val"].untyped_storage()) for p in existing}, produced, placements, machines, output)
    query, joint = resolve(source["query"]), target.solution
    if isinstance(joint, schedule.SavedNetwork):
        target.solution = target.plan = dataclasses.replace(joint, source={"query": query})
    else:
        joint.source = attached(query, joint.local, joint.owner)
    held = {"machine": {}, "processor": {}}
    for kind, function, name, owner in about["bodies"]:
        held[kind].setdefault(function, {})[name] = graphs[owner]
    return Scheduled(exported, held["machine"], devices, about["machines"], about["output_processor"], channels,
                     about["inputs"], about["feedforward"], carried, about["engines"], target.plan,
                     signature=target.signature, stored=about["stored"],
                     invariant={record["function"]: tuples(record["records"]) for record in about["invariant"]},
                     functions=functions, processor_modules=held["processor"], operands=operands)

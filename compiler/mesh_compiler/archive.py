# design/data-flow.md; design/aot-compilation-change-request.md A8 to A10 and appendix item 6: the archive of one
# completed machine program (its native target, joint result and recipe), with the captured program's per-function
# records as explicit typed records indexed by function ordinal (the completed target's body order, CallAnalysis.keys);
# a captured function's key, which may be a callable, is compilation-only and is not saved. Scheduled.plan is the one
# target's joint owner's own result and Scheduled.signature that target's complete signature archive
# (compiler/mesh_compiler/native.py Target.plan and Target.signature, saved and restored with the target;
# compiler/mesh_compiler/driver.py compiled), kept in no record of their own here
import collections
import ctypes as C
import dataclasses
from copy import deepcopy
import inspect
import json
import pathlib

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx._graph_pickler import GraphPickler, Options, _GraphPickleData
from torch.fx.graph_module import _assign_attr, _del_attr, _get_attr
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree

from . import native, schedule
from .ir import Call, CapturedProgram, Exported, Scheduled


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


# a retained reference: `owner` its graph owner's saved index, `path` a nested GraphModule's attribute path in it ("" the
# owner itself; a captured function's "function_{ordinal}" in the outer source graph), `node` a Node's name (None: the
# GraphModule); `Related` a SourceRelation, `Graphed` an rdma/mesh_graph.py Graph, `Relating` its native Relation (a
# SourceRelation's and a Graph over it the same one), `Searching` a Reached search (by its native search) and `Closing`
# a Closures closure by saved index, each object once
Ref = collections.namedtuple("Ref", "owner path node")
Related = collections.namedtuple("Related", "index")
Graphed = collections.namedtuple("Graphed", "index")
Relating = collections.namedtuple("Relating", "index")
Searching = collections.namedtuple("Searching", "index")
Closing = collections.namedtuple("Closing", "index")
# a raw torch.fx Graph, one no GraphModule owns (torch/_inductor/loop_body.py LoopBodyBlock.graph: LightTracer's), by
# saved index, and a Node of one by (that index, its name); a Graph a GraphModule owns by that module's Ref (`GraphOf`:
# the loaded module's own graph)
Rawed = collections.namedtuple("Rawed", "index")
RawRef = collections.namedtuple("RawRef", "graph node")
GraphOf = collections.namedtuple("GraphOf", "module")
# a real tensor of the one state.pt payload by its index there (held "tensors", the graph attributes' registry: the
# same object one index)
Tensored = collections.namedtuple("Tensored", "index")
# a construction's upstream record by identity (`producing`'s registration): its kind ("produced", "input", "output",
# "kernel", "extern", "inlined", "lifted", "wrapper"), its processor, its index in that processor's record (None: the
# Produced)
Record = collections.namedtuple("Record", "kind processor index")
# a wrapper line's IR node operand (torch/_inductor/ir.py IRNode) by its index in its Produced's IR node table (each
# node once by identity), a wrapper line by its index in its Produced's line table, and an ir.py OutputSpec (a
# Layout) as its class and its canonical pickle state (its __dict__), each IR node in it a Lowered
Lowered = collections.namedtuple("Lowered", "index")
Lined = collections.namedtuple("Lined", "index")
Stated = collections.namedtuple("Stated", "cls state")
# a wrapper's planner object (torch/_inductor/codegen/memory_planning.py LiveRange, LiveRanges, AllocationTreeNode,
# AllocationPool, AllocationPools, BufferGroup, MemoryPlanner; wrapper.py MemoryPlanningState) by its index in its
# Produced's planner table (each once by identity: an Allocation's pool its AllocationPool's own), each recorded as its
# class and its own state (its __dict__, a Stated)
Planned = collections.namedtuple("Planned", "index")


# torch 2.14.0 torch/fx/_graph_pickler.py GraphPickler (this archive's one payload codec): a raw torch.fx Graph as
# _GraphPickleData, which owns every Node's serialization and the graph's tracer and codegen, unpickled with the one
# _UnpickleState over no module (Graph's owning_module None: `graphed`); every other value as GraphPickler reduces it
class Pickler(GraphPickler):
    def reducer_override(self, obj):
        if isinstance(obj, torch.fx.Graph):
            return graphed, (_GraphPickleData(obj, self.options), self._unpickle_state)
        return super().reducer_override(obj)


def graphed(data, state):
    return data.unpickle(None, state)


# torch 2.14.0 torch/_export/serde/serialize.py serialize_hoo_inputs (a higher-order operator's GraphModule operand by
# its attribute target): an owner's nested branch GraphModules (decompose.py _branch_owned) by attribute path
def branches(module, prefix=""):
    for node in module.graph.nodes:
        held = _get_attr(module, node.target) if node.op == "get_attr" else None
        if isinstance(held, torch.fx.GraphModule):
            yield prefix + node.target, held
            yield from branches(held, prefix + node.target + ".")


# a value with each Node, GraphModule, SourceRelation, Graph, Relation, Reached and Closures it holds a Ref, Related,
# Graphed, Relating, Searching or Closing (each registered once in `held`, an owner with its branches; a torch.fx Graph
# GraphOf its owning module's Ref, or Rawed, and a raw graph's Node RawRef), a real tensor Tensored (the one state.pt
# payload's, its graph attributes' same registry: an ExportedProgram's state_dict and constants, the adapted body's
# attribute objects its export lifted, one index each), a collections.Counter as it is (canonical pickle's:
# torch/_inductor/loop_body.py LoopBody.op_counts, operation names' counts), a sequence or a namedtuple of them its
# own type (torch utils/_pytree.py namedtuple semantics: its fields in order), everything else, a FakeTensor among
# them (GraphPickler's own), as it is (torch/fx/_graph_pickler.py GraphPickler: no raw Node)
def encoded(held, value):
    if isinstance(value, collections.Counter):
        return value
    if isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor):
        return Tensored(held["tensors"].setdefault(id(value), (len(held["tensors"]), value))[0])
    if isinstance(value, dict):
        return {encoded(held, k): encoded(held, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        items = [encoded(held, v) for v in value]
        return type(value)(*items) if hasattr(type(value), "_fields") else type(value)(items)
    if isinstance(value, schedule.SourceRelation):
        return Related(held["relations"].setdefault(value, len(held["relations"])))
    if isinstance(value, schedule.mesh_graph.Reached):
        return Searching(held["searches"].setdefault(id(value.owned), (len(held["searches"]), value))[0])
    if isinstance(value, schedule.mesh_graph.Closures):
        return Closing(held["closures"].setdefault(id(value), (len(held["closures"]), value))[0])
    if isinstance(value, schedule.mesh_graph.Graph):
        return Graphed(held["natives"].setdefault(value, len(held["natives"])))
    if isinstance(value, schedule.mesh_graph.Relation):
        return Relating(held["kept"].setdefault(id(value), (len(held["kept"]), value))[0])
    if isinstance(value, torch.fx.Graph) and value.owning_module is not None:
        return GraphOf(encoded(held, value.owning_module))
    if isinstance(value, torch.fx.Graph):
        return Rawed(held["raws"].setdefault(id(value), (len(held["raws"]), value))[0])
    if isinstance(value, torch.fx.Node) and value.graph.owning_module is None:
        return RawRef(encoded(held, value.graph).index, value.name)
    if not isinstance(value, (torch.fx.Node, torch.fx.GraphModule)):
        return value
    module, owners, nested = (value.graph.owning_module if isinstance(value, torch.fx.Node) else value, held["owners"],
                              held["nested"])
    if module not in nested and module not in owners:
        owners[module] = len(owners)
        for path, branch in branches(module):
            nested.setdefault(branch, (owners[module], path))
    owner, path = nested[module] if module in nested else (owners[module], "")
    return Ref(owner, path, value.name if isinstance(value, torch.fx.Node) else None)


# an `encoded` value with each Ref bound to its loaded owner's actual Node or GraphModule (`loaded`: the owners, their
# GraphModules' nodes by name, made on first use, the SourceRelations, the Graphs, the searches, the closures, the
# native Relations), each Related, Graphed, Searching, Closing and Relating to its loaded one
def decoded(loaded, value):
    graphs, named, relations, natives, searches, closures, kept, raws, tensors = loaded
    if isinstance(value, Tensored):
        return tensors[value.index]
    if isinstance(value, Rawed):
        return raws[value.index]
    if isinstance(value, GraphOf):
        return decoded(loaded, value.module).graph
    if isinstance(value, RawRef):
        if ("raw", value.graph) not in named:
            named["raw", value.graph] = {node.name: node for node in raws[value.graph].nodes}
        return named["raw", value.graph][value.node]
    if isinstance(value, Ref):
        module = _get_attr(graphs[value.owner], value.path) if value.path else graphs[value.owner]
        if value.node is None:
            return module
        if (value.owner, value.path) not in named:
            named[value.owner, value.path] = {node.name: node for node in module.graph.nodes}
        return named[value.owner, value.path][value.node]
    if isinstance(value, Related):
        return relations[value.index]
    if isinstance(value, Graphed):
        return natives[value.index]
    if isinstance(value, Searching):
        return searches[value.index]
    if isinstance(value, Closing):
        return closures[value.index]
    if isinstance(value, Relating):
        return kept[value.index]
    if isinstance(value, collections.Counter):
        return value
    if isinstance(value, dict):
        return {decoded(loaded, k): decoded(loaded, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        items = [decoded(loaded, v) for v in value]
        return type(value)(*items) if hasattr(type(value), "_fields") else type(value)(items)
    return value


# schedule-emit/schedule_emit/archive.py wrap's copy.deepcopy: one graph owner's serialization copy over the archive's
# one memo (torch 2.14.0 torch/fx/graph_module.py GraphModule.__deepcopy__, every GraphModule once, a branch several
# owners share one copy; torch/fx/graph.py Graph.__deepcopy__, each node its own meta dict), the original untouched:
# the outer source graph's (`held` "projected": each known call a call_module of its captured function's GraphModule
# by ordinal, ir.Call; a GraphModule over just those, no user module or callable; the call's capture key, compilation
# only, dropped: a loaded outer graph, already so, the same). Each copied GraphModule's and node's metadata the
# original's `encoded`, each tensor attribute (its
# real data the one state.pt payload's, held "tensors") deleted from the copy, (its module path, its target, its
# index); with the owner's copy those records
def snapshot(held, module):
    memo, projected = held["memo"], held["projected"].get(module)
    origins = dict(module.named_modules()) if projected is None else {**dict(module.named_modules()), **{
        f"function_{k}" + ("." + path if path else ""): m for k, f in projected[1].items()
        for path, m in f.named_modules()}}
    for m in origins.values():
        if isinstance(m, torch.fx.GraphModule):
            memo.setdefault(id(m.meta), {})
            memo.update((id(t), t) for n in m.graph.nodes if n.op == "get_attr"
                        for t in [_get_attr(m, n.target)] if isinstance(t, torch.Tensor))
    if projected is None:
        copy = deepcopy(module, memo)
    else:
        graph = deepcopy(module.graph, memo)
        for node, k in projected[0].items():
            memo[node].op, memo[node].target = "call_module", f"function_{k}"
            memo[node].meta.pop("function", None)
        copy = torch.fx.GraphModule({**{n.target: deepcopy(_get_attr(module, n.target), memo) for n in graph.nodes
                                        if n.op == "get_attr"},
                                     **{f"function_{k}": deepcopy(f, memo) for k, f in projected[1].items()}}, graph)
    attributes = []
    for path, made in copy.named_modules():
        if isinstance(made, torch.fx.GraphModule) and id(made) not in held["copied"]:
            held["copied"].add(id(made))
            made.meta = encoded(held, dict(origins[path].meta))
            for node in made.graph.nodes:
                node.meta = encoded(held, node.meta)
            for target in dict.fromkeys(n.target for n in made.graph.nodes if n.op == "get_attr"):
                if isinstance(value := _get_attr(made, target), torch.Tensor):
                    held["tensors"].setdefault(id(value), (len(held["tensors"]), value))
                    attributes.append((path, target, held["tensors"][id(value)][0]))
                    _del_attr(made, target)
    return copy, attributes


# design/aot-compilation-change-request.md A10; schedule.py analyse: a CallAnalysis's query results in their saved
# form: its incoming search's arrays (over the calls' relation, which the joint result keeps), its paths' roots and
# search (a Reached over their own Graph), its widths by function ordinal (count, the member vertex IDs and their
# chains: mesh_graph.closure_chains's owner array and result), its supplied causes, accesses and generated state, each
# function's source Declarations by ordinal (`declared`), the target's copies (native.py Target.copying: each call's
# CopyRecords with their copy operations, `copying`) and every construction's upstream results by (function ordinal,
# machine group) (`constructed`)
def queried(analysis, ordinal, target):
    paths = analysis.paths
    return {"search": schedule.mesh_graph.Reached(analysis.search).save(),
            "paths": (list(paths.items), schedule.mesh_graph.Reached(paths.owner)),
            "widths": {ordinal[key]: (count, tuple(chain.owner[1][k] for k in chain.span), tuple(chain))
                       for key, (count, chain) in analysis.widths.items()},
            "supplied": analysis.supplied, "accesses": analysis.accesses, "generated": analysis.generated,
            "declarations": {ordinal[key]: declared(made) for key, made in analysis.declarations.items()},
            "copies": copying(target),
            "compiled": {(ordinal[key], group): constructed(machine)
                         for (key, group), machine in analysis.constructions.items()}}


# lower_metal.py ConstructedMachine (schedule.py CallAnalysis.constructions: every function's every machine group's
# construction, a remote group's included): its upstream results, each processor's Produced (`producing`, which
# registers each of its records in `index` by identity as a Record: ("produced", p), ("input" or "output", p, position
# k), ("kernel", p, call n), ("extern", "inlined" or "lifted", p, its preorder index), ("line", p, its line table
# index)), and their native correspondence by those same records, every operation ID that construction's own
# program's (the local group's final rows place them by native.py Row.offset and Row.values, the native recipe's):
# `bindings` each original binding of a member's wrapper name (lower_metal.py Binding, in resolution order) as (its
# processor, its occurrence: each enclosing (ExternCall, consumed Inlined or Lifted) pair as its pair of Records,
# outermost first (lower_metal.py inline's `occurrence + ((record, body),)`), () its wrapper's own lines; its record: a
# memory planning line's, an Input's, an Inlined's or Lifted's Record, a
# constant's GraphLowering's module; its name as produced; its source, the original object bound (its processor's
# `producing` operand: an IR node by that processor's node table, a constant's tensor Tensored, a symbolic input's
# expression as it is); its native value: (root, view ValueID), ("literal", value) or None), every occurrence its own;
# `interfaces` each processor's (inputs, outputs), each
# (its Input's or Output's record, its native value as projected: (root, view ValueID), ("literal", value) or None);
# `regions` each registered repeat or switch (metal.py Region, postorder) as (its ExternCall's record, kind, region
# index, first and end operations, selector), its region index this construction's (native.py Target.composite appends
# one body after only typed copies: metal_ir.mm AppendSource keeps the ordinals and shifts the operations by
# Row.offset); `inlinings` each consumed control body interval ConstructedMachine.inlinings records (lower_metal.py
# inline: a repeat's cond before, body and cond after, a switch's Lifted branches, an invoke_subgraph's Inlined; its
# record by identity, an Inlined's or a Lifted's) as (its
# record, first and end operations); `emitted` each projection's emission (metal.py Emitted) as (its record: a
# registered one's (a Kernel, ExternCall, Inlined, Produced), a machine body node or landing as it is; its operation,
# first and end operations); `upstream` each produced kernel's Launched as (operation, its Kernel's record, its launch
# arguments as constructed: buffer names, element bytes, first byte, bytes, access); `neural` each Neural Engine
# body's Neural as (operation, its Produced's record, function, compute units, input, output and state features), its
# package the native artifact's (native135), its temporary directory not a saved value
def constructed(machine):
    index, sources = {}, {}
    produced = {p: producing(made, p, index, tuple(binding for binding in machine.bindings if binding.processor == p),
                             sources) for p, made in machine.produced.items()}

    def recorded(record):
        return index[id(record)] if id(record) in index else record

    # a Binding's record: a GraphLowering (a constant's) by its module, any other (a line, an Input, an Inlined or
    # Lifted) by its registered Record
    def bound(record):
        from torch._inductor.graph import GraphLowering
        return record.module if isinstance(record, GraphLowering) else index[id(record)]
    return {"produced": produced,
            "bindings": tuple((binding.processor,
                               tuple((index[id(call)], index[id(body)]) for call, body in binding.occurrence),
                               bound(binding.record), binding.name, sources[id(binding)], binding.value)
                              for binding in machine.bindings),
            "interfaces": {p: tuple(tuple((index[id(given)], native) for given, native in ports) for ports in pair)
                           for p, pair in machine.interfaces.items()},
            "regions": tuple((index[id(region.record)], region.kind, region.region, region.first, region.end,
                              region.selector) for region in machine.regions),
            "inlinings": tuple((index[id(inlined)], first, end) for inlined, first, end in machine.inlinings),
            "emitted": tuple((recorded(made.record), made.operation, made.first, made.end) for made in machine.emitted),
            "upstream": tuple((launched.operation, index[id(launched.kernel)], launched.arguments)
                              for launched in machine.upstream),
            "neural": tuple((made.operation, index[id(made.produced)], made.function, made.units, made.inputs,
                             made.outputs, made.states) for made in machine.neural)}


# lower_inductor.py Produced as its producer returned it, its FX graphs and nodes by Ref and the upstream compiler's
# own objects (GraphLowering, Scheduler, kernel owners, IR nodes, wrapper lines) by the FX GraphModules, nodes and
# names they are of: its kind and device; the scheduled processor `module`; the adapted `body` it traced (its control
# bodies nested in it); its `sources` as produced, {(graph ID, copy node name): original node}, the graph IDs
# torch/fx/traceback.py NodeSource's (id(node.graph)), so every saved node's meta["from_node"] chain resolves against
# it as lower_inductor.py _origins reads it, and `graphs` each body graph's ID's GraphModule; its `sent`; the lowered FX
# graph its GraphLowering was built over (`lowered`, graph.module: its control subgraphs nested in it), None for the
# Neural Engine; its AOT `signature` (graph.module.meta's, the same object); each Input's (position, body placeholder,
# export name: its GraphLowering graph_inputs key, value) and Output's (position, body node, export name, sent,
# value), each value by its operand (an IR node by the one node table; constructed's `interfaces` its native value by
# the same Input or Output record); the
# upstream objects by identity, each once by index (first seen first): `owners` the distinct backend kernels' count
# (Kernel.owner), `schedulers` the distinct Schedulers' count (each call, extern and inlining naming its own),
# `loops` each LoopBody a call's ranges name (`body_of`), `proxies` each cpu CppKernelProxy its ranges or a fused nest
# name (`proxy_of`), `subkernels` each CppKernel a proxy selects (`sub_of`), `contexts` each cpu finalize's local
# buffer context (`context_of`); `kernels` each call (Produced.kernels order: each
# KernelCallLine its own) as its own record (`calling`); the control records, each registered once by identity in
# preorder (`registered`: its kind's table index in `index`, constructed's): `externs` its scheduler's ExternCalls'
# indices, `extern_calls` each ExternCall (`called`) as (its GraphLowering's module, its IR operation's name there
# (GraphLowering.register_operation), its FX origins, its scheduler's index, its Inlined's and its Lifted's indices,
# in order), `inlined` and `lifted` each Inlined (`inlining`) and Lifted (`lifting`) as (its SubgraphLowering's
# module, its scheduler's index, its own ExternCalls' indices); each Seeds as (its GraphLowering's module, its
# RandomSeeds buffer's operation and buffer names there and FX origins, its uses, its buffer's every ExternCall's
# index: that IR node's own calls); the Neural Engine's Model as (its ExportedProgram's graph module, graph signature,
# state dict, range constraints, module call graph, example inputs, constants and verifiers, its conversion's selected
# operands, received, handles, and its inputs, outputs, states and aliases as lower_inductor.py Model holds them: every
# field but the MLModel itself, whose executable binding is the native package's). Each control record's native
# correspondence is constructed's (its regions, inlinings
# and emissions by these records); its wrapper lines (`lines`): `table` each line once by identity (`line_of`: a str
# line its text, a produced kernel's call its record, a typed line its class and fields' operands), the owner
# sequences by that table (`wrappers` each wrapper's own lines, `extern` and `inlined` each record's, `lifted` each
# Lifted's wrapper), `nodes` each IR node its lines name once by identity (`node_of`: class, own name and layout,
# ExternCall records); `wrappers` each wrapper's GraphLowering modules. Their effects are the native artifact's
# (metal_archive Records: Upstreams, Values, views and regions, by the records constructed pairs); each wrapper name's
# native value at each occurrence (a memory planning line's, a graph input's, a control body input's, a constant's)
# is constructed's `bindings`, by these line and record identities (each line registered in `index`)
def producing(made, p, index, bindings, sources):
    model, schedulers, owners, loops = made.model, {}, {}, {}
    controls, objects, lowered = ({kind: [] for kind in ("extern", "inlined", "lifted")},
                                  {kind: [] for kind in ("extern", "inlined", "lifted")}, {})
    index[id(made)] = Record("produced", p, None)
    index.update((id(given), Record("input", p, k)) for k, given in enumerate(made.inputs))
    index.update((id(given), Record("output", p, k)) for k, given in enumerate(made.outputs))
    index.update((id(kernel), Record("kernel", p, n)) for n, kernel in enumerate(made.kernels.values()))

    def scheduled(scheduler):
        return schedulers.setdefault(id(scheduler), len(schedulers))

    # a control record (an ExternCall, an Inlined or a Lifted) registered once by identity in `index`, in preorder: its
    # index in its kind's table, its own record filled after (its children's indices after its own)
    def registered(kind, record, filled):
        if id(record) not in index:
            index[id(record)] = Record(kind, p, len(controls[kind]))
            controls[kind].append(None)
            objects[kind].append(record)
            controls[kind][index[id(record)].index] = filled()
        return index[id(record)].index

    # an ExternCall, and its IR node's every ExternCall record by that node's identity, each once (`lowered`: a Seeds
    # buffer's; an inlined body's cond before and after its loop each its own call of one node)
    def called(extern):
        n = registered("extern", extern, lambda: (
            extern.graph.module, extern.node.get_operation_name(), tuple(extern.node.origins),
            scheduled(extern.scheduler), tuple(map(inlining, extern.inlined)), tuple(map(lifting, extern.lifted))))
        calls = lowered.setdefault(id(extern.node), [])
        if n not in calls:
            calls.append(n)
        return n

    def inlining(inlined):
        return registered("inlined", inlined, lambda: (inlined.graph.module, scheduled(inlined.scheduler),
                                                       tuple(map(called, inlined.externs))))

    def lifting(lifted):
        return registered("lifted", lifted, lambda: (lifted.graph.module, scheduled(lifted.scheduler),
                                                     tuple(map(called, lifted.externs))))

    # a SchedulerNode's LoopBody (scheduler.py SchedulerNode._body), each once by identity, by its index in `loops`
    def looped(body):
        return loops.setdefault(id(body), (len(loops), body))[0]

    # torch 2.14.0 codegen/cpp.py CppKernelProxy, each once by identity, its index in `proxies`; each of its
    # subkernels (CppKernelProxy.kernels: the scalar, vector, tail and tile2d CppKernels) once by identity in
    # `subkernels`; each finalize's local buffer context once by identity in `contexts`
    proxies, subkernels, contexts = {}, {}, {}

    def proxied(proxy):
        return proxies.setdefault(id(proxy), (len(proxies), proxy))[0]

    def contexted(context):
        return contexts.setdefault(id(context), (len(contexts), context))[0]

    # one subkernel (lower_inductor.py MeshCppKernel, MeshCppVecKernel, MeshCppTile2DKernel): its class, its domain as
    # it emits it (CppKernel's itervars, ranges, reduction_depth, active_ranges, inner_itervars and call_ranges; a
    # CppVecKernel's tiling_factor, tiling_idx, tail_size and num_elems; a CppTile2DKernel's tiling_indices, outer_idx,
    # inner and outer tail sizes and num_elems and inner_is_tiling_idx), each CppRanges (its node's name and LoopBody,
    # its variables, suffix and accesses as received: method, buffer name, index, store mode, its LoopBodyBlock graph
    # and node, mask) and its indirect symbols (mesh_indirect: each its value, size and check)
    def sub_of(kernel):
        from torch._inductor.codegen.cpp import CppTile2DKernel, CppVecKernel
        named = (("itervars", "ranges", "reduction_depth", "active_ranges", "inner_itervars", "call_ranges")
                 + (("tiling_factor", "tiling_idx", "tail_size", "num_elems")
                    if isinstance(kernel, CppVecKernel) else ())
                 + (("tiling_indices", "outer_idx", "inner_tail_size", "outer_tail_size", "inner_num_elems",
                     "outer_num_elems", "inner_is_tiling_idx") if isinstance(kernel, CppTile2DKernel) else ()))
        return {"class": type(kernel), "domain": {name: getattr(kernel, name) for name in named},
                "ranges": tuple((r.node.get_name(), looped(r.node._body), r.variables, r.suffix, r.accesses)
                                for r in kernel.mesh_ranges),
                "indirect": kernel.mesh_indirect}

    # a LoopNest: its loops (each LoopLevel as it is: var, size, offset, tiled_size, steps, parallel, simd_omp,
    # simd_vec, collapsed, is_reduction, simd_nelements) and its kernel, the proxy its own nest's (set_kernel) or an
    # OuterLoopFusedKernel (merge_outer_fusion_kernels: each `inner` LoopNest from_loop_level of a fused proxy's)
    def nested(nest):
        from torch._inductor.codegen.cpp import OuterLoopFusedKernel
        kernel = nest.kernel
        return nest.loops, (("outer", tuple(map(nested, kernel.inner))) if isinstance(kernel, OuterLoopFusedKernel)
                            else ("proxy", proxied(kernel)))

    # one proxy: its call ranges, each subkernel's index (its selection, in its order) and its loop nest
    def proxy_of(proxy):
        return {"call_ranges": proxy.call_ranges,
                "kernels": tuple(subkernels.setdefault(id(kernel), (len(subkernels), kernel))[0]
                                 for kernel in proxy.kernels),
                "loop_nest": nested(proxy.loop_nest)}

    # lower_inductor.py Ranges (an mps second-pass SchedulerNode): its node's name and LoopBody, its variables,
    # reduction and accesses as received (method, buffer name, index, store mode, its LoopBodyBlock graph, that graph's
    # node: a raw graph's, RawRef); a cpu finalize_kernel's (CppKernelProxy, nodes, local buffer context, its local
    # buffers at that finalize) its proxy's index, its nodes' names and LoopBodies (codegen/cpp.py codegen_node and
    # codegen_outer_loop_node: each node a SchedulerNode of get_nodes(), every one's _body called by
    # CppKernelProxy.codegen_nodes; a template's finalize_kernel nodes none), its context's index and that finalize's
    # own local buffers ({name: IR buffer}, lower_inductor.py MeshKernelGroup.finalize_kernel's copy before the
    # context's __exit__ clears them; {} outside one), each buffer by the node table
    ranged = {"mps": lambda r: (r.node.get_name(), looped(r.node._body), r.variables, r.reduction, r.accesses),
              "cpu": lambda r: (proxied(r[0]), tuple((node.get_name(), looped(node._body)) for node in r[1]),
                                contexted(r[2]), operand(r[3]))}

    # one Kernel call record as its call made it: its backend kernel's index (Kernel.owner, by identity), its
    # scheduler's index, its GraphLowering's module, its launch, place and (entry, source) as produced, its original
    # origins, its schedule (each SchedulerNode by its name, an mps DisableReduction or EnableReduction scope marker as
    # it is), its scheduled nodes' names and its ranges
    def calling(kernel):
        return (owners.setdefault(id(kernel.owner), len(owners)), scheduled(kernel.scheduler), kernel.graph.module,
                kernel.launch, kernel.place, kernel.produced, kernel.origins,
                tuple(entry if isinstance(entry, type) else entry.get_name() for entry in kernel.schedule),
                tuple(node.get_name() for node in kernel.scheduled), tuple(map(ranged[kernel.place], kernel.ranges)))
    externs = tuple(map(called, made.externs))

    # torch 2.14.0 torch/_inductor/codegen/wrapper.py: each wrapper (PythonWrapperCodegen) once by identity, its index
    # in `wrappers` with each GraphLowering module whose wrapper_code it is (an inlined control body writes into its
    # parent's: one wrapper, several graphs), a Lifted's own wrapper among them
    from torch._inductor import ir as upstream_ir
    from torch._inductor.codegen import memory_planning
    from torch._inductor.codegen.wrapper import MemoryPlanningState, PythonWrapperCodegen, WrapperLine
    from torch._inductor.graph import GraphLowering
    wrappers, nodes, table, planners = {}, {}, {}, {}
    PLANNERS = (memory_planning.LiveRange, memory_planning.LiveRanges, memory_planning.AllocationTreeNode,
                memory_planning.AllocationPool, memory_planning.AllocationPools, memory_planning.BufferGroup,
                memory_planning.MemoryPlanner, MemoryPlanningState)
    # Produced.kernels' keys: each KernelCallLine's identity (lower_inductor.py produced, id(record[2]))
    called_lines = {called: Record("kernel", p, n) for n, called in enumerate(made.kernels)}

    def wrapped(wrapper):
        return wrappers.setdefault(id(wrapper), (len(wrappers), [], wrapper))

    # a GraphLowering's module among its wrapper's graphs, once; its wrapper's Record
    def wrote(graph):
        number, held_by, _ = wrapped(graph.wrapper_code)
        if not any(module is graph.module for module in held_by):
            held_by.append(graph.module)
        return Record("wrapper", p, number)
    for graph in (*((made.graph,) if made.graph is not None else ()), *(k.graph for k in made.kernels.values()),
                  *(record.graph for kind in objects.values() for record in kind)):
        wrote(graph)

    # a typed line's operand: its wrapper by `wrappers` (Record "wrapper"), a GraphLowering by its module, an IR node
    # by the node table (Lowered), an OutputSpec by its class and canonical state (Stated), a sequence, namedtuple or
    # mapping of them its own type, any other value (a str, int, sympy expression, dtype, SymbolicCallArg, key path)
    # as it is
    def operand(value):
        if isinstance(value, PythonWrapperCodegen):
            return Record("wrapper", p, wrapped(value)[0])
        if isinstance(value, GraphLowering):
            return value.module
        if isinstance(value, upstream_ir.IRNode):
            return Lowered(nodes.setdefault(id(value), (len(nodes), value))[0])
        if isinstance(value, WrapperLine):
            return lined(value)
        if isinstance(value, PLANNERS):
            return Planned(planners.setdefault(id(value), (len(planners), value))[0])
        if isinstance(value, upstream_ir.OutputSpec):
            return Stated(type(value), operand(dict(vars(value))))
        if isinstance(value, dict):
            return {key: operand(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            items = [operand(item) for item in value]
            return type(value)(*items) if hasattr(type(value), "_fields") else type(value)(items)
        return value

    # one wrapper line once by identity in `table`, its index: a str line the original text, a produced kernel's
    # KernelCallLine its call's record (Produced.kernels' keys), a typed line (a WrapperLine dataclass) its class and
    # each field's operand, any other line its operand (torch/utils/_indented_buffer.py LineContext, a NamedTuple,
    # its context's; codegen/common.py DeferredLine, its name and line state, as it is, never called)
    def lined(line):
        if id(line) not in table:
            table[id(line)] = (len(table), line)
            index[id(line)] = Record("line", p, table[id(line)][0])
        return Lined(table[id(line)][0])

    def line_of(line):
        if isinstance(line, str):
            return line
        if id(line) in called_lines:
            return called_lines[id(line)]
        if dataclasses.is_dataclass(line):
            return type(line), {field.name: operand(getattr(line, field.name)) for field in dataclasses.fields(line)}
        return operand(line)

    # each IR node once by identity: its class, its own name and layout (IRNode.maybe_get_name, maybe_get_layout:
    # None where it has none) and its ExternCall records' indices by its identity; its native values are per
    # occurrence, not the node's (constructed's `bindings`, by the binding line or record)
    def node_of(node):
        return (type(node), node.maybe_get_name(), operand(node.maybe_get_layout()),
                tuple(lowered.get(id(node), ())))

    # each call's record (its ranges' local buffers by `operand`, so after it), then every proxy a range or a fused
    # nest names, each record once, an inner proxy's as it is registered; then each subkernel its proxies select
    kernels = tuple(map(calling, made.kernels.values()))
    proxy_records = []
    while len(proxy_records) < len(proxies):
        proxy_records.append(proxy_of(list(proxies.values())[len(proxy_records)][1]))
    subkernel_records = tuple(sub_of(kernel) for _, kernel in subkernels.values())

    # one finalize's local buffer context (codegen/cpp_utils.py LocalBufferContext: its global_buffers and
    # global_to_local, each name's IR buffer by the node table, and removed_buffers; its local_buffers cleared at its
    # __exit__, each finalize's own copy its range entry's; its kernel_args the group's own, its exit_stack its scope's,
    # neither a saved value), or virtualized.py's NullHandler (no local buffer context) as it is
    def context_of(context):
        from torch._inductor.codegen.cpp_utils import LocalBufferContext
        if isinstance(context, LocalBufferContext):
            return {"global_buffers": operand(context.global_buffers),
                    "global_to_local": operand(context.global_to_local), "removed_buffers": context.removed_buffers}
        return context
    context_records = tuple(context_of(context) for _, context in contexts.values())

    # each Input's (position, body placeholder, export name, value) and Output's (position, body node, export name,
    # sent, value), its value as produced by its operand: GraphLowering graph_inputs[name] or graph_outputs[k] (a
    # TensorBox, buffer or TorchBindObject by the node table; a sympy expression, a generator's or None as it is), the
    # Neural Engine's ExportedProgram node (a Ref)
    inputs = tuple((given.position, given.node, given.name, operand(given.value)) for given in made.inputs)
    outputs = tuple((given.position, given.node, given.name, given.sent, operand(given.value))
                    for given in made.outputs)
    # each of this processor's Bindings' original bound object (lower_metal.py Binding.source: a memory planning
    # line's buffer, a graph input's or a control body input's graph_inputs entry, an IR node by this processor's node
    # table, a symbolic input's expression as it is; a constant's tensor, the one state.pt registry's at its save) by
    # that Binding's identity in `sources`, constructed's
    sources.update((id(binding), operand(binding.source)) for binding in bindings)

    # the owner sequences, each line by the one table: each wrapper's own lines, each ExternCall's and Inlined's lines
    # (the same line objects, its codegen's), each Lifted its wrapper's
    sequences = {"extern": tuple(tuple(map(lined, extern.lines)) for extern in objects["extern"]),
                 "inlined": tuple(tuple(map(lined, inlined.lines)) for inlined in objects["inlined"]),
                 "lifted": tuple(Record("wrapper", p, wrapped(lifted.wrapper)[0]) for lifted in objects["lifted"])}
    # torch 2.14.0 torch/_inductor/codegen/wrapper.py run_wrapper_ir_passes (torch-inductor-aot-lowered-result.patch:
    # the planned stage, none printed) and cpp_wrapper_cpu.py (torch-inductor-cpp-wrapper-binding.patch): one
    # wrapper's own planning and binding result as it holds it: its planned lines (its `lines`, `wrapper_lines`), its
    # planner's result owner (memory_planner, past_planning_states: the one its passes made, the other None), its
    # bound_allocations and bound_views (each record's own operands; none where it bound none, as the binding's own
    # setdefault leaves it), each operand by the one registries (`operand`)
    def planning(wrapper):
        return {"memory_planner": operand(wrapper.memory_planner),
                "past_planning_states": operand(wrapper.past_planning_states),
                "bound_allocations": operand(wrapper.__dict__.get("bound_allocations", [])),
                "bound_views": operand(wrapper.__dict__.get("bound_views", []))}

    # torch 2.14.0 torch/_inductor/output_code.py LoweredAOTI (torch-inductor-aot-lowered-result.patch), Produced.lowered:
    # its device type, its planned lines (wrapper_code.lines after run_wrapper_ir_passes, by the one line table) and its
    # emission's result once emitted (AotCodeCompiler.compile's AotCompiled, torch-inductor-cpp-build-result.patch: its
    # wrapper and kernel sources and code, their objects' and the link's CppBuildActions, their PrecompiledHeaders, its
    # constants' AotConstants with their object bytes, its extern kernel nodes, generated files, model class and
    # header, each as it holds it; None before its emission); its GraphLowering by its module; its resumption a closure
    # over that stage's config and handlers, not a saved value
    lowered_aot = made.lowered and (operand(made.lowered.graph), made.lowered.device_type,
                                    tuple(map(lined, made.lowered.planned)), operand(made.lowered.compiled))

    # every wrapper's lines and planning, line, IR node and planner object each recorded once as registered (a line's
    # operand may name another wrapper, node or planner object; a planner object a line, node or another)
    wrapper_lines, planning_records, line_records, node_records, planner_records = [], [], [], [], []
    while (len(wrapper_lines), len(line_records), len(node_records), len(planner_records)) != (
            len(wrappers), len(table), len(nodes), len(planners)):
        if len(wrapper_lines) < len(wrappers):
            wrapper = list(wrappers.values())[len(wrapper_lines)][2]
            wrapper_lines.append(tuple(map(lined, wrapper.lines)))
            planning_records.append(planning(wrapper))
        elif len(line_records) < len(table):
            line_records.append(line_of(list(table.values())[len(line_records)][1]))
        elif len(node_records) < len(nodes):
            node_records.append(node_of(list(nodes.values())[len(node_records)][1]))
        else:
            planned_object = list(planners.values())[len(planner_records)][1]
            planner_records.append(Stated(type(planned_object), operand(dict(vars(planned_object)))))
    lines = {**sequences, "wrappers": tuple(wrapper_lines), "planning": tuple(planning_records),
             "table": tuple(line_records), "nodes": tuple(node_records), "planners": tuple(planner_records)}

    # torch/_inductor/loop_body.py LoopBody as produced: its sizes, iteration and reduction variables, variable
    # ranges, indexing expressions, indirect variables and their ranges, memory usage, operation counts, partial
    # accumulation, its root block's and each subblock's raw FX graph (Rawed), and each submodule but get_index (its
    # own bound method over `indexing_exprs`, as _init_with_copy rebinds it) as its `clone`, the functools.partial of
    # LoopBody.bind_set_indirect_shim, bind_masked_shim or bind_scan_shim and its original operands
    def body_of(body):
        shims = dict(body.submodules)
        shims.pop("get_index")
        return {"sizes": body.sizes, "iter_vars": body.iter_vars, "reduce_vars": body.reduce_vars,
                "var_ranges": body.var_ranges, "indexing_exprs": body.indexing_exprs,
                "indirect_vars": body.indirect_vars, "indirect_var_ranges": body.indirect_var_ranges,
                "memory_usage": body.memory_usage, "op_counts": body.op_counts,
                "has_partial_accumulate": body.has_partial_accumulate, "root": body.root_block.graph,
                "subblocks": {name: block.graph for name, block in body.subblocks.items()},
                "submodules": {name: shim.clone for name, shim in shims.items()}}
    return {"kind": made.kind, "device": made.device, "module": made.module, "body": made.body,
            "sources": made.sources, "graphs": {id(module.graph): module for module in made.body.modules()
                                                if isinstance(module, torch.fx.GraphModule)},
            "sent": made.sent, "lowered": made.graph and made.graph.module, "lowered_aot": lowered_aot,
            "signature": made.signature,
            "processor": made.processor,
            "inputs": inputs, "outputs": outputs,
            "kernels": kernels, "owners": len(owners), "schedulers": len(schedulers),
            "loops": tuple(body_of(body) for _, body in loops.values()), "proxies": tuple(proxy_records),
            "subkernels": subkernel_records, "contexts": context_records, "externs": externs,
            "extern_calls": tuple(controls["extern"]),
            "inlined": tuple(controls["inlined"]), "lifted": tuple(controls["lifted"]), "lines": lines,
            "wrappers": tuple(tuple(held_by) for _, held_by, _ in wrappers.values()),
            "seeds": tuple((seeds.graph.module, seeds.buffer.get_operation_name(), seeds.buffer.get_name(),
                            tuple(seeds.buffer.origins), seeds.uses, tuple(lowered[id(seeds.buffer)]))
                           for seeds in made.seeds),
            "model": model and (model.program.graph_module, model.program.graph_signature,
                                model.program.state_dict, model.program.range_constraints,
                                model.program.module_call_graph, model.program.example_inputs,
                                model.program.constants, model.program.verifiers, model.selected, model.received,
                                model.handles, model.inputs, model.outputs, model.states, model.aliases)}


# a dataclass record's fields by name, each as it is
def fields(record):
    return {field.name: getattr(record, field.name) for field in dataclasses.fields(record)}


# native.py Target.copying: each call's copies on this machine, each its schedule.py CopyRecord (its fields, its
# endpoint Slot's, native.py Slot) with its copy operation in that call's composite program (None: no landing view)
def copying(target):
    return {j: tuple(({**fields(copy), "slot": copy.slot and fields(copy.slot)}, operation) for copy, operation in held)
            for j, held in target.copying.items()}


# schedule.py Declarations and dependences: one function's source declarations as produced, each field as it is, its
# `dependences` the native records (mesh.Dependence's fields each) with their owner's original results: the events
# (their source nodes), schedule.schedule's source-order closure (Graph.ancestors' Closures, saved once, Closing),
# the event Graph (its relation saved once, Graphed), its synchronization mask and boundary Views' values and the
# found records (a, b, lag, edge), in the owner's order. Declarations.before (a field) is the crossing-free task
# edges, a list, as it is
def declared(made):
    events, before, graph, redundant, first, last, found = made.dependences.owner
    return {**fields(made),
            "dependences": ([tuple(getattr(record, name) for name, _ in record._fields_) for record in made.dependences],
                            events, before, graph, list(redundant), list(first), list(last), found)}


# rdma/mesh_graph.py Graph.ancestors and labels: a closure's processing order as held with it (its Owned's first held
# operand: a View of one search's own vertex IDs, schedule.schedule's Graph.order()[::-1]), that search's saved owner
# (a Reached of the View's owner, Searching), the Searched array the View reads (the one at its address) and its span
def ordering(held, order):
    searched, address = order.owner.made.contents, C.cast(order.items, C.c_void_p).value
    array = next(name for name in ("preorder", "postorder")
                 if C.cast(getattr(searched, name), C.c_void_p).value == address)
    return encoded(held, schedule.mesh_graph.Reached(order.owner)), array, (order.span.start, order.span.stop,
                                                                             order.span.step)


# an `ordering` record loaded: that View over the loaded search's own array (rdma/mesh_graph.py ordered)
def ordered(reached, array, span):
    return schedule.mesh_graph.ordered(reached.owned, getattr(reached.owned.made.contents, array), range(*span))


# design/aot-compilation-change-request.md A10; schedule.py call_operands: the compilation's source record, each value
# `encoded`: the outer source graph an owner projected (`snapshot`), its captured functions nested at their ordinals;
# each function's Exported records (its state tensors the one state.pt payload's; its two AOT stage records, `aliases`
# and `decomposed`, each as its export retained it, whether `decomposed` is the same record, so a loaded Exported has
# the one object both, and its own: torch/_functorch/_aot_autograd/autograd_cache.py's own pickle of a
# ViewAndMutationMeta, through this one GraphPickler payload, each prototype and traced tangent FakeTensor its
# describer's, none made runtime safe, schemas.py make_runtime_safe clearing traced_tangents, and none replayed on
# load), each call's Call, the CallOperands
# fields (a storage by its holder Node, `existing` by the outer placeholders holding it) and the `query`
def sourced(held, ordinal, operands, query):
    captured, tensors = operands.captured, held["tensors"]
    outer, functions = captured.exported.graph_module, {ordinal[key]: f for key, f in captured.functions.items()}
    held["owners"].setdefault(outer, len(held["owners"]))
    for path, branch in branches(outer):
        held["nested"].setdefault(branch, (held["owners"][outer], path))
    held["projected"][outer] = ({node: ordinal[call.function] for node, call in captured.calls.items()},
                                {k: f.graph_module for k, f in functions.items()})
    for k, f in functions.items():
        held["nested"].setdefault(f.graph_module, (held["owners"][outer], f"function_{k}"))
        for path, branch in branches(f.graph_module):
            held["nested"].setdefault(branch, (held["owners"][outer], f"function_{k}.{path}"))
    existing = [n for n in outer.graph.nodes if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
                and StorageWeakRef(n.meta["val"].untyped_storage()) in operands.existing]

    # one function's Exported record: its graph module, state (each target and its state.pt index), tokens, user
    # inputs, outputs, out spec, state kinds, specialized literals, AOT stage records (whether `decomposed` is
    # `aliases`), signature and range constraints
    def exported(f):
        return (f.graph_module, {name: (target, tensors.setdefault(id(t), (len(tensors), t))[0])
                                 for name, (target, t) in f.state.items()}, f.tokens, f.user_inputs, f.outputs,
                pytree.treespec_dumps(f.out_spec), f.state_kinds, f.specialized, f.aliases, f.decomposed is f.aliases,
                f.decomposed, f.signature, f.range_constraints)

    # every Exported once by identity, its index in `exporteds`: each final function's, then each outlined original's
    # and each of its parts' (a part a later outline replaced, a repeated step: the one object, one record)
    records = {}

    def recorded(f):
        return records.setdefault(id(f), (len(records), f))[0]
    finals = {k: recorded(f) for k, f in functions.items()}
    outlined = tuple((recorded(original), tuple(map(recorded, parts))) for original, parts in captured.outlined)
    return encoded(held, {
        "graph": outer,
        # the public Exported's range constraints (Exported.range_constraints, export.capture's make_constraints)
        "range_constraints": captured.exported.range_constraints,
        "exporteds": tuple(exported(f) for _, f in records.values()),
        "functions": finals,
        # CapturedProgram.outlined: each replaced function's original Exported and its parts' Exported objects in part
        # order, each by its `exporteds` index
        "outlined": outlined,
        "calls": [(node, ordinal[call.function], call.inputs, call.outputs) for node, call in captured.calls.items()],
        "operands": (operands.nodes, [holder for _, holder in operands.storages], operands.views, operands.accesses,
                     operands.results, operands.root_sources, sorted(operands.held),
                     [tuple(getattr(port, f.name) for f in dataclasses.fields(port)) for port in operands.ports],
                     existing, operands.produced,
                     {ordinal[key]: placed for key, placed in operands.placements.items()}, operands.machines,
                     operands.output_processor),
        "query": query})


# a joint result's query once its mesh_local exists (schedule.py loaded_program, mesh_local_load; published_program,
# mesh_program_load): its calls' relation and closure that mesh_local's own (borrowed: Relation.borrowed, an Owned
# its owner `owner` holds and frees), the incoming search loaded once over that Graph, the paths' View over their loaded
# search, with the saved `query`
def attached(query, local, owner):
    mesh_graph = schedule.mesh_graph
    analysis = local.planned.analysis.contents
    graph = mesh_graph.Graph(range(analysis.ncalls), mesh_graph.Relation.borrowed(analysis.relation))
    roots, reached = query["paths"]
    owned, found = reached.owned, reached.owned.made.contents
    return {"query": query, "graph": graph, "search": mesh_graph.Reached.load(query["search"], graph).owned,
            "closure": mesh_graph.Closures(mesh_graph.Owned(analysis.closure, lambda made: None, graph, (owner,)), False),
            "paths": mesh_graph.View(owned, roots, range(len(roots)), lambda k: mesh_graph.ordered(
                owned, found.preorder, range(found.place[k], found.end[k])))}


def save(directory, scheduled):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported, (target,) = scheduled.exported, scheduled.functions.values()
    ordinal = {body.key: k for k, body in enumerate(target.bodies)}
    signature, spec = exported.interface
    held = {"owners": {}, "nested": {}, "relations": {}, "natives": {}, "searches": {}, "closures": {}, "kept": {},
            "raws": {}, "tensors": {}, "memo": {}, "projected": {}, "copied": set()}
    # the source record from the original analysis (Target.solution.analysis, its operands), or a loaded
    # compilation's retained one (its joint owner's `source`, Scheduled.operands)
    joint = target.solution
    analysis = joint.analysis if isinstance(joint, schedule.Solved) else None
    query, operands = ((queried(analysis, ordinal, target), analysis.operands) if analysis is not None else
                       (joint.source["query"], scheduled.operands))
    source = sourced(held, ordinal, operands, query)
    bodies = [(kind, ordinal[key], name, encoded(held, module))
              for kind, kept in (("machine", scheduled.modules), ("processor", scheduled.processor_modules))
              for key, modules in kept.items() for name, module in modules.items()]
    # each graph owner (each graph a Node reference names) its `snapshot`, each SourceRelation, Graph, search and
    # closure once (rdma/mesh_graph.py Relation.save, Reached.save, Closures.save: mesh-graph.h's own archives; a
    # closure with its Graph, whether transposed, and its columns), all one GraphPickler payload
    # (torch/fx/_graph_pickler.py: one MetaTensorDescriber, so FakeTensor duplicates and views keep their storage;
    # every operator by name, ops_filter None; every node metadata key as encoded, node_metadata_key_filter None:
    # nn_module_stack's normalized (path, type name) strings, source_fn_stack's targets by their pickle reference)
    graphs, relations, natives, searches, closures, kept, raws = [], [], [], [], [], [], []
    while (len(graphs), len(relations), len(natives), len(searches), len(closures), len(kept), len(raws)) != tuple(
            map(len, (held["owners"], held["relations"], held["natives"], held["searches"], held["closures"],
                      held["kept"], held["raws"]))):
        if len(graphs) < len(held["owners"]):
            graphs.append(snapshot(held, list(held["owners"])[len(graphs)]))
        elif len(raws) < len(held["raws"]):
            # a raw graph's serialization copy over the archive's one memo (torch/fx/graph.py Graph.__deepcopy__: each
            # node its own meta dict, no owning module), the original untouched, each node's metadata its `encoded`
            raw = deepcopy(list(held["raws"].values())[len(raws)][1], held["memo"])
            for node in raw.nodes:
                node.meta = encoded(held, node.meta)
            raws.append(raw)
        elif len(relations) < len(held["relations"]):
            relation = list(held["relations"])[len(relations)]
            relations.append(encoded(held, (relation.relation, relation.functions, relation.supplied,
                                            relation.accesses, relation.producers)))
        elif len(natives) < len(held["natives"]):
            graph = list(held["natives"])[len(natives)]
            natives.append(encoded(held, (graph.relation, graph.vertices)))
        elif len(searches) < len(held["searches"]):
            reached = list(held["searches"].values())[len(searches)][1]
            searches.append((encoded(held, reached.owned.graph), reached.save()))
        elif len(closures) < len(held["closures"]):
            closure = list(held["closures"].values())[len(closures)][1]
            closures.append((encoded(held, closure.owned.graph), closure.save(), closure.transposed,
                             encoded(held, closure.columns), ordering(held, closure.owned.held[0])))
        else:
            kept.append(list(held["kept"].values())[len(kept)][1].save())
    (directory / "annotations.pkl").write_bytes(Pickler.dumps(
        {"graphs": graphs, "relations": relations, "natives": natives, "searches": searches, "closures": closures,
         "kept": kept, "raws": raws, "source": source, "bodies": bodies},
        Options(ops_filter=None, node_metadata_key_filter=None)))
    # the one state payload: every state, attribute and default tensor the original object itself (a Parameter its own
    # class), so torch.save's one pickle memo keeps each object shared by several names, attributes or defaults one
    # object at load
    torch.save({"state": {name: (key, tensor) for name, (key, tensor) in exported.state.items()},
                "attributes": [tensor for _, tensor in held["tensors"].values()],
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
    }))


def load(directory):
    directory = pathlib.Path(directory)
    about = json.loads((directory / "scheduled.json").read_text())
    payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    state, defaults, specialized, tensors = (payload["state"], payload["defaults"], payload["specialized"],
                                             payload["attributes"])
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
    # the one payload (GraphPickler.loads, one FakeTensorMode: every graph as saved, FakeTensors' shared storage and
    # shared GraphModules one object each); each tensor attribute its one state.pt tensor; each SourceRelation, Graph,
    # search and closure loaded once (Relation.load, Reached.load, Closures.load: no solve, search or closing); then
    # every GraphModule's, raw graph's and node's metadata, the source record and every Ref bound to the actual loaded
    # Node, Graph or GraphModule (a raw graph's by `graphed`, Pickler's)
    saved = GraphPickler.loads((directory / "annotations.pkl").read_bytes(), FakeTensorMode())
    graphs = [module for module, _ in saved["graphs"]]
    for module, attributes in saved["graphs"]:
        for path, name, index in attributes:
            _assign_attr(tensors[index], module.get_submodule(path), name)
    mesh_graph = schedule.mesh_graph
    relations, natives, searches, closures = [], [], [], []
    # each native Relation in the restored joint owner's one analysis context (native.restore's, every native artifact
    # of this compilation loaded in it)
    kept = [mesh_graph.Relation.load(data, target.solution.analysis_context) for data in saved["kept"]]
    loaded = (graphs, {}, relations, natives, searches, closures, kept, saved["raws"], tensors)
    resolve = lambda value: decoded(loaded, value)
    natives += [mesh_graph.Graph(resolve(vertices), resolve(relation)) for relation, vertices in saved["natives"]]
    searches += [mesh_graph.Reached.load(data, resolve(graph)) for graph, data in saved["searches"]]
    closures += [mesh_graph.Closures.load(data, resolve(graph), transposed, resolve(columns),
                                          (ordered(*resolve(order)),))
                 for graph, data, transposed, columns, order in saved["closures"]]
    relations += [schedule.SourceRelation(resolve(relation), list(resolve(functions_)), list(resolve(supplied)),
                                          list(resolve(accesses)), dict(resolve(producers)))
                  for relation, functions_, supplied, accesses, producers in saved["relations"]]
    seen = set()
    for module in graphs:
        for made in module.modules():
            if isinstance(made, torch.fx.GraphModule) and id(made) not in seen:
                seen.add(id(made))
                made.meta = resolve(made.meta)
                for node in made.graph.nodes:
                    node.meta = resolve(node.meta)
    for raw in saved["raws"]:
        for node in raw.nodes:
            node.meta = resolve(node.meta)
    # the source record: the outer source graph (each call a call_module of its function's captured GraphModule) the
    # public Exported's, each function's Exported, each call's Call, the CallOperands (Scheduled.operands; a storage by
    # its holder's loaded FakeTensor storage) and the query on the joint owner: a local result's attached now, a network
    # result's kept on its saved owner until its preparation (bridge.Published, schedule.published_program)
    source = resolve(saved["source"])
    exported = Exported(source["graph"], state, about["tokens"], about["user_inputs"], outputs,
                        pytree.treespec_loads(about["out_spec"]),
                        {name: InputKind[kind] for name, kind in about["state_kinds"].items()},
                        interface=(signature, pytree.treespec_loads(about["interface"]["spec"])),
                        specialized=specialized, range_constraints=source["range_constraints"])
    # an Exported record (`sourced`'s) as its loaded Exported, `decomposed` the same object as `aliases` where saved so;
    # each record once, the final functions and the outlined originals and parts by its `exporteds` index
    def restored(module, kept, tokens, inputs, results, spec, kinds, literals, aliases, same, decomposed, signature,
                 ranges):
        return Exported(module, {name: (key, tensors[t]) for name, (key, t) in kept.items()}, tokens, inputs, results,
                        pytree.treespec_loads(spec), kinds, aliases, specialized=literals,
                        decomposed=aliases if same else decomposed, signature=signature, range_constraints=ranges)
    exporteds = [restored(*record) for record in source["exporteds"]]
    captured = {k: exporteds[i] for k, i in source["functions"].items()}
    outlined = tuple((exporteds[original], tuple(exporteds[i] for i in parts)) for original, parts in source["outlined"])
    calls = {node: Call(k, inputs, results) for node, k, inputs, results in source["calls"]}
    (nodes, holders, views, accesses, results, roots, kept, ports, existing, produced, placements, machines,
     output) = source["operands"]
    operands = schedule.CallOperands(
        CapturedProgram(exported, captured, calls, outlined), nodes, tuple(captured), [calls[node] for node in nodes],
        {k: k for k in captured}, [(StorageWeakRef(h.meta["val"].untyped_storage()), h) for h in holders], views,
        accesses, results, roots, frozenset(kept), tuple(schedule.CallPort(*port) for port in ports),
        {StorageWeakRef(p.meta["val"].untyped_storage()) for p in existing}, produced, placements, machines, output)
    # the target's copies' provenance (Target.copying), each CopyRecord and Slot its loaded record
    target.copying = {j: tuple((schedule.CopyRecord(**{**record, "slot": record["slot"] and
                                                       native.Slot(**record["slot"])}), operation)
                               for record, operation in held)
                      for j, held in source["query"]["copies"].items()}
    joint = target.solution
    if isinstance(joint, schedule.SavedNetwork):
        target.solution = target.plan = dataclasses.replace(joint, source={"query": source["query"]})
    else:
        joint.source = attached(source["query"], joint.local, joint.owner)
    held = {"machine": {}, "processor": {}}
    for kind, function, name, module in resolve(saved["bodies"]):
        held[kind].setdefault(function, {})[name] = module
    return Scheduled(exported, held["machine"], devices, about["machines"], about["output_processor"], channels,
                     about["inputs"], about["feedforward"], carried, about["engines"], target.plan,
                     signature=target.signature, stored=about["stored"],
                     invariant={record["function"]: tuples(record["records"]) for record in about["invariant"]},
                     functions=functions, processor_modules=held["processor"], operands=operands)

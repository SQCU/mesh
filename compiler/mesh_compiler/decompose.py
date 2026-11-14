# design/torch-source-lowering-change-request.md items 5 and 6: the placed graph's numerical operators expanded, after
# compiler/mesh_compiler/devices.py on_devices and before the schedule, into the nodes of their source formulations
# (compiler/mesh_compiler/operations.py); torch 2.14 torch/fx/experimental/proxy_tensor.py make_fx (decomposition_table,
# tracing_mode "fake" over the node's own fake values), torch/fx/graph.py Graph.graph_copy, and
# torch.export.default_decompositions, the table export's run_decompositions applies (compiler/mesh_compiler/export.py)
import functools

import torch
from torch._guards import detect_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from .ir import views
from .operations import _matrix_exponential

A = torch.ops.aten
# the metadata compiler/mesh_compiler/ir.py views makes, made again over the expanded nodes; and the traced value's own
VIEWED = ("view", "out", "to", "invariant")
TRACED = ("val", "tensor_meta")
# compiler/mesh_compiler/placement.py `work_axes`: a shard's (FX value, dimension, original full extent) triples, the
# values its own graph's nodes, mapped through every replacement below
AXES = ("work_axes",)


# `node`'s work axes, each reference its image under `mapping`: within one graph (`within`) a reference `mapping`
# does not move is that same node, still the graph's; into another graph every reference has its image there (the
# part's own copy, placeholder or result, compiler/mesh_compiler/decompose.py _Part), each full extent kept
def _axes_moved(node, mapping, within=True):
    if "work_axes" in node.meta:
        node.meta["work_axes"] = tuple((mapping.get(value, value) if within else mapping[value], dimension, full)
                                       for value, dimension, full in node.meta["work_axes"])


# `source`'s work axes onto each tensor node of `result`, its replacement: the source's own output that node, its
# operands their images (`mapping`; within one graph `within`, an unmoved operand itself)
def _axes_onto(source, result, mapping, within=True):
    if "work_axes" not in source.meta:
        return
    for made in [n for n in pytree.tree_leaves(result) if isinstance(n, torch.fx.Node)
                 and isinstance(n.meta.get("val"), torch.Tensor)]:
        made.meta["work_axes"] = source.meta["work_axes"]
        _axes_moved(made, {**mapping, source: made}, within)


# compiler/mesh_compiler/placement.py place `input_parts`: each placeholder's (processor, partition Node, landing
# geometry) pieces, each partition its image under `mapping` (within one graph an unmoved partition itself), each
# landing geometry (the source view's storage offset, shape and stride in the placeholder; None for a non-tensor) kept
def _parts_moved(parts, mapping, within=True):
    return {mapping.get(node, node) if within else mapping[node]:
            tuple((processor, mapping.get(part, part) if within else mapping[part], geometry)
                  for processor, part, geometry in pieces)
            for node, pieces in parts.items()}


# `targets` each processor's (engine, device) (compiler/mesh_compiler/driver.py), the engine of a node's own processor
# (node.meta["processor"]) its selection. Each operator below expands into its source formulation of the node's own
# operands, where `cpu`, also on a CPU-placed node: item 5 keeps an operator with a LAPACK driver as one node on the
# CPU, and matrix_exp has none (item 5: "Item 6 writes it"), so it expands on every processor (item 6: scaling and
# squaring, `max_squarings` the operand, JAX's 16). `max_iterations` is item 6's QDWH bound, JAX's 10, for the spectral
# formulations, none of which this expands yet: an svd, eigh, eig, eigvals, pinv or lstsq node is kept as placed.
# Each expanded node's nodes take its processor and source metadata, and a made tensor (a `device` keyword) that
# processor's device, as compiler/mesh_compiler/devices.py on_devices gives every placed node;
# graph_module.meta["decomposed"] maps each of them to the original node it replaces, whose uses (the graph's outputs
# among them) take the formulation's result; the nodes whose uses or operands changed (the original's operands, its
# expanded nodes and its uses) are given to ir.views, which makes their view and destination metadata again
def decompose(graph_module, targets, *, max_iterations=10, max_squarings=16):
    formulations = {A.linalg_matrix_exp.default: (functools.partial(_matrix_exponential, max_squarings=max_squarings),
                                                  True)}
    table = dict(torch.export.default_decompositions().items())
    graph = graph_module.graph
    sources = graph_module.meta.setdefault("decomposed", {})
    changed = []
    for node in [n for n in graph.nodes if n.op == "call_function" and n.target in formulations]:
        formulation, cpu = formulations[node.target]
        if not cpu and targets[node.meta["processor"]][0] == "cpu":
            continue
        flat, spec = pytree.tree_flatten((node.args, node.kwargs))
        held = [k for k, item in enumerate(flat) if isinstance(item, torch.fx.Node)]

        def formula(*operands, flat=flat, spec=spec, held=held, formulation=formulation):
            given = list(flat)
            for k, operand in zip(held, operands):
                given[k] = operand
            args, kwargs = pytree.tree_unflatten(given, spec)
            return formulation(*args, **kwargs)
        operands = [flat[k].meta["val"] for k in held]
        with detect_fake_mode(operands):
            traced = make_fx(formula, decomposition_table=table, tracing_mode="fake")(*operands)
        copied = dict(zip([n for n in traced.graph.nodes if n.op == "placeholder"], [flat[k] for k in held]))
        with graph.inserting_before(node):
            result = graph.graph_copy(traced.graph, copied)
        made = [copied[n] for n in traced.graph.nodes if n.op not in ("placeholder", "output")]
        kept = {key: value for key, value in node.meta.items() if key not in TRACED + VIEWED + AXES}
        device = targets[node.meta["processor"]][1]
        for generated in made:
            generated.meta.update(kept)
            if "device" in generated.kwargs:
                generated.kwargs = {**generated.kwargs, "device": device}
            sources[generated] = node
        users = list(node.users)
        _axes_onto(node, result, {})
        for user in users:
            _axes_moved(user, {node: result})
        if "input_parts" in graph_module.meta:
            graph_module.meta["input_parts"] = _parts_moved(graph_module.meta["input_parts"], {node: result})
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        changed += [flat[k] for k in held] + made + users
    views(changed)
    graph_module.recompile()
    return graph_module

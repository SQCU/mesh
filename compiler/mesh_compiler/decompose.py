# design/torch-source-lowering-change-request.md items 5 and 6: the placed graph's numerical operators expanded, after
# compiler/mesh_compiler/devices.py on_devices and before the schedule, into the nodes of their source formulations
# (compiler/mesh_compiler/operations.py); torch 2.14 torch/fx/experimental/proxy_tensor.py make_fx (decomposition_table,
# tracing_mode "fake" over the node's own fake values), torch/fx/graph.py Graph.graph_copy, and
# torch.export.default_decompositions, the table export's run_decompositions applies (compiler/mesh_compiler/export.py)
import functools
import operator

import torch
from torch._guards import detect_fake_mode
from torch.export.graph_signature import OutputKind
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from .devices import on_devices
from .ir import RETURNED, Exported, shape, views
from .ir import given as ir_given
from .operations import (LSTSQ_FIXED, QDWH_EIGH_STEP, _eigen_geev, _least_squares_svd, _lstsq_driver,
                         _matrix_exponential, _operands, _pseudoinverse, _qdwh_eigh_result, _qdwh_eigh_state,
                         _singular_qdwh)
from .placement import insert_send_recv
from .schedule import carried

import mesh_graph

A = torch.ops.aten
# the metadata compiler/mesh_compiler/ir.py views makes, made again over the expanded nodes; and the traced value's own
VIEWED = ("view", "out", "to", "invariant")
TRACED = ("val", "tensor_meta")
# compiler/mesh_compiler/placement.py `work_axes`: a shard's (FX value, dimension, original full extent) triples, the
# values its own graph's nodes, mapped through every replacement below
AXES = ("work_axes",)
# the formulation tracing's decomposition table: torch.export.default_decompositions less the operators a typed
# structured owner consumes as they are (operations.SOURCES: tril and triu _triangular_part), whose torch/_refs
# decompositions make int64 index data
PRESERVED = (A.tril, A.triu)


def _decompositions():
    return {op: rule for op, rule in torch.export.default_decompositions().items()
            if getattr(op, "overloadpacket", None) not in PRESERVED}


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


# a whole placeholder's own piece, as placement.py place's: its stated processor, itself and its supplied value's own
# (storage offset, shape, stride), None for a non-tensor
def _whole(node, processor):
    value = node.meta.get("val")
    return ((processor, node, (value.storage_offset(), tuple(value.shape), tuple(value.stride()))
             if isinstance(value, torch.Tensor) else None),)


# `targets` each processor's (engine, device) (compiler/mesh_compiler/driver.py). matrix_exp has no LAPACK driver
# (item 5: "Item 6 writes it"), so it expands on every processor (item 6: scaling and squaring, `max_squarings` the
# operand, JAX's 16); a spectral node (svd, eigh, eig, eigvals, lstsq) is kept as placed, pinv expanded already before
# the eigh outline (outline); `max_iterations` unused here (outline's QDWH bound)
def decompose(graph_module, targets, *, max_iterations=10, max_squarings=16):
    formulation = functools.partial(_matrix_exponential, max_squarings=max_squarings)
    return _expanded(graph_module, targets, {A.linalg_matrix_exp.default: lambda node: formulation})


# each node of a `formulations` target expanded into its source formulation (formulations[target](node)) of the node's
# own operands and arguments, traced by make_fx over the operands' fake values. Each expanded node's nodes take its
# processor and source metadata (its draw site among it), and a made tensor (a `device` keyword) that processor's
# device, as compiler/mesh_compiler/devices.py on_devices gives every placed node; graph_module.meta["decomposed"]
# maps each of them to the original source node it replaces (an expanded node itself generated, its own original: the
# map's keys the graph's nodes, its values the source's), whose uses (the graph's outputs among them), work axes and
# input parts take the formulation's result; the nodes whose uses or operands changed (the original's operands, its
# expanded nodes and its uses) are given to ir.views, which makes their view and destination metadata again
def _expanded(graph_module, targets, formulations, chosen=lambda node: True):
    table = _decompositions()
    graph = graph_module.graph
    sources = graph_module.meta.setdefault("decomposed", {})
    changed = []
    # Rejection filters are not part of even temporary integration validation.
    # `chosen` selects the target's formulation; an unchosen node keeps its placed lowering (operations._lowering).
    for node in [n for n in graph.nodes if n.op == "call_function" and n.target in formulations and chosen(n)]:
        formulation = formulations[node.target](node)
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
        origin = sources.pop(node, node)
        for generated in made:
            generated.meta.update(kept)
            if "device" in generated.kwargs:
                generated.kwargs = {**generated.kwargs, "device": device}
            sources[generated] = origin
        # a tuple result (torch.fx: the node's users its getitems): each getitem's image its output's result node
        replaced = {g: result[g.args[1]] for g in node.users} if isinstance(result, (tuple, list)) else {node: result}
        users = [user for old in replaced for user in old.users]
        _axes_onto(node, result, {})
        for old, image in replaced.items():
            if old is not node:
                _axes_onto(old, image, {})
        for user in users:
            _axes_moved(user, replaced)
        if "input_parts" in graph_module.meta:
            graph_module.meta["input_parts"] = _parts_moved(graph_module.meta["input_parts"], replaced)
        for old, image in replaced.items():
            old.replace_all_uses_with(image)
            if old is not node:
                sources.pop(old, None)
                graph.erase_node(old)
        graph.erase_node(node)
        changed += [flat[k] for k in held] + made + users
    views(changed)
    graph_module.recompile()
    return graph_module


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/tpu/linalg/eigh.py _eigh_work; P. Barham et al.,
# Pathways, arXiv:2203.12533, section 3: an eigh node placed on a non-CPU engine outlined (a CPU-placed one keeps its
# LAPACK driver, item 5). Its function, split at that node, becomes three captured functions: the prefix (the
# function's nodes not reached from the eigh, the QDWH state of its operand: operations._qdwh_eigh_state), the one
# reusable step (operations.QDWH_EIGH_STEP: one pop of the agenda, its block's split or its eigenvalues), and the suffix
# (the eigenvalues ascending with their vectors, operations._qdwh_eigh_result, then the nodes the eigh reaches). Each
# call of the function becomes the prefix's call, 2n - 1 calls of the step (termination_size 1: a split's children
# both of positive size, pub13 section 5.2, so n leaves and n - 1 splits) and the suffix's (export.substitute). Each
# original output keeps its role and target, made by the part computing its value; a value one part makes and a later
# reads crosses as a call output and input, on the eigh's processor; each part's placed graph keeps its nodes'
# processors and its own transfers (renumbered), and is put on its devices (devices.on_devices) as the driver's staged
# loop does. `staged` the driver's staged records by function key, `processors` and `machines` its maps. First each
# function's non-CPU SVD-driver lstsq nodes (gelsd, gelss) expand into operations._least_squares_svd, then its pinv
# nodes (every overload, on every processor) into operations._pseudoinverse, its tolerance dtype
# the selected device's (float32 on MPS, float64 else), then its non-CPU _linalg_svd nodes (a pinv's among them) into
# operations._singular_qdwh (QDWH-SVD, `max_iterations` its QDWH bound), and the function's placed graph is staged
# again, so their eigh nodes are outlined below (a CPU pinv's or svd's keeps its LAPACK driver)
def outline(captured, staged, targets, processors, machines, *, max_iterations=10):
    from .export import substitute
    staged, pending = dict(staged), list(captured.functions)

    def pseudoinverse(node):
        dtype = torch.float32 if targets[node.meta["processor"]][1].type in ("mps", "metal") else torch.float64
        return lambda *args, **kwargs: _pseudoinverse(**_operands(node.target, args, kwargs), tolerance_dtype=dtype)

    def singular(node):
        shapes = [(tuple(value.shape), value.dtype) for value in node.meta["val"]]
        return lambda *args, **kwargs: _singular_qdwh(**_operands(node.target, args, kwargs), shapes=shapes,
                                                      max_iterations=max_iterations)

    # operations.LSTSQ_FIXED: the same formulation's solution, rank and singular_values, no residuals produced
    def least_squares(node):
        shapes = [(tuple(value.shape), value.dtype) for value in node.meta["val"]]
        if node.target is LSTSQ_FIXED:
            solution, rank, singular = shapes
            return lambda *args, **kwargs: (lambda made: (made[0], made[2], made[3]))(_least_squares_svd(
                **_operands(node.target, args, kwargs), shapes=(solution, None, rank, singular)))
        return lambda *args, **kwargs: _least_squares_svd(**_operands(node.target, args, kwargs), shapes=shapes)

    def svd_driver(node):
        read = ir_given(node)[1]
        return targets[node.meta["processor"]][0] != "cpu" and _lstsq_driver(
            read["self"].meta["val"], read.get("driver")) in ("gelsd", "gelss")

    # Reference-LAPACK zgeev.f and dgeev.f through operations.GEEV_EIGEN: an eig or eigvals placed on a non-CPU engine,
    # complex or real input (a CPU one keeps operations._eigen)
    def eigen(node):
        wanted = node.target.overloadpacket is A.linalg_eig
        return lambda *args, **kwargs: _eigen_geev(**_operands(node.target, args, kwargs), vectors=wanted)

    def off_cpu(node):
        return targets[node.meta["processor"]][0] != "cpu"
    for key in pending:
        placed = staged[key][0]
        found = {n.target for n in placed.graph.nodes if n.op == "call_function"}
        if found & {*PINV, *SVD, *LSTSQ, *EIG}:
            _expanded(placed, targets, dict.fromkeys(EIG, eigen), off_cpu)
            _expanded(placed, targets, dict.fromkeys(LSTSQ, least_squares), svd_driver)
            _expanded(placed, targets, dict.fromkeys(PINV, pseudoinverse))
            _expanded(placed, targets, dict.fromkeys(SVD, singular),
                      lambda node: targets[node.meta["processor"]][0] != "cpu")
            placed.meta["invariant"] = _invariant(placed)
            staged[key] = _staged(placed, processors, machines)
    while pending:
        key = pending.pop(0)
        placed = staged[key][0]
        # an eigh whose operand A has no element (a zero matrix extent or an empty batch: its fake value's numel) is
        # partially evaluated to its canonical empty result: it stays placed, and operations._eigen_hermitian returns
        # no records for it (before any _Lapack, whatever the processor). Every eigh with elements is outlined
        eigh = next((n for n in placed.graph.nodes if n.op == "call_function" and n.target is EIGH
                     and targets[n.meta["processor"]][0] != "cpu" and ir_given(n)[1]["A"].meta["val"].numel()), None)
        if eigh is None:
            continue
        parts = _outlined(key, captured.functions[key], placed, eigh, max_iterations)
        captured = substitute(captured, key, [(k, exported, inputs, results)
                                              for k, exported, _, inputs, results in parts])
        del staged[key]
        for k, exported, graph_module, _, _ in parts:
            if k not in staged:
                staged[k] = _staged(graph_module, processors, machines)
                pending.append(k)
    return captured, staged


EIGH, COPY = A._linalg_eigh.default, A.copy.default
PINV = tuple(getattr(A.linalg_pinv, overload) for overload in A.linalg_pinv.overloads())
SVD = tuple(getattr(A._linalg_svd, overload) for overload in A._linalg_svd.overloads())
LSTSQ = (*(getattr(A.linalg_lstsq, overload) for overload in A.linalg_lstsq.overloads()), LSTSQ_FIXED)
EIG = tuple(getattr(packet, overload) for packet in (A.linalg_eig, A.linalg_eigvals, A._linalg_eigvals)
            for overload in packet.overloads())
# the Step's inputs: the state, the input's norm and its source draw site (operations.QDWH_EIGH_STEP); the prefix's
# first outputs, each Step call the previous call's state and the prefix's norm and site
STEP_INPUTS = ("blocks", "vectors", "agenda", "top", "norm", "site")
STATE = len(STEP_INPUTS)


# compiler/mesh_compiler/driver.py compile_captured's staged record of one placed graph: (placed, its output processor,
# its channels, it on its devices, its carried transfers, its placeholders' stated placements)
# the source `outline` record (its source graph, eigh, relation and Reached) the placed part's and its device graph's
# own, not deep-copied by devices.on_devices; the shared Step has none
def _staged(placed, processors, machines):
    outlined = placed.meta.pop("outline", None)
    try:
        graph = on_devices(placed, processors)
    finally:
        if outlined is not None:
            placed.meta["outline"] = outlined
    if outlined is not None:
        graph.meta["outline"] = outlined
    return (placed, placed.graph.output_node().meta["processor"], len(placed.meta["transfers"]), graph,
            carried(graph.meta["transfers"], machines),
            [node.meta["stated"] for node in placed.graph.nodes if node.op == "placeholder"])


def _fresh(value):
    with detect_fake_mode([value]):
        return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)


# torch 2.14.0 torch/fx/experimental/proxy_tensor.py make_fx over the operands' fake values, copied in at the graph's
# end, its nodes the processor's and the source node's metadata, each in `decomposed` the source's original (`origin`)
def _formulated(graph, formula, operands, source, processor, mapping, decomposed, origin):
    table = _decompositions()
    values = [operand.meta["val"] for operand in operands]
    with detect_fake_mode(values):
        traced = make_fx(formula, decomposition_table=table, tracing_mode="fake")(*values)
    copied = dict(zip([n for n in traced.graph.nodes if n.op == "placeholder"], operands))
    result = graph.graph_copy(traced.graph, copied)
    kept = {key: value for key, value in source.meta.items() if key not in TRACED + VIEWED + AXES}
    for made in [copied[n] for n in traced.graph.nodes if n.op not in ("placeholder", "output")]:
        made.meta.update(kept, processor=processor)
        decomposed[made] = origin
    _axes_onto(source, result, mapping, within=False)
    return result


# compiler/mesh_compiler/placement.py place, step 4: per processor the invariant slices its nodes read
def _invariant(graph_module):
    graph, held = graph_module.graph, {}
    returned = set(graph.output_node().all_input_nodes)
    for node in graph.nodes:
        for read in node.all_input_nodes if node.op == "call_function" and "view" not in node.meta else ():
            if {"invariant", "state"} & read.meta.keys() and read not in returned and all(shape(read)):
                slice_ = read.meta.get("invariant", (read.meta.get("state"),))
                held.setdefault(node.meta["processor"], {})[slice_] = None
    return {processor: tuple(slices) for processor, slices in held.items()}


# one part's placed graph: its placeholders, the copies of `nodes` (graph order) and the formulations' nodes, its
# outputs on `processor` (a value elsewhere sent there, placement.insert_send_recv) with each original output's role
# kept and its `destination` (a mutation's value another part's input first copied in, as export.read does); a value
# crossing from another part stated on its producer's processor, where its users are (their transfers are the parts');
# `sources` the placed graph's generated → original map, its copied nodes' images this part's `decomposed`
class _Part:
    def __init__(self, original, nodes, transfers, parts, sources):
        self.graph, self.env, self.original, self.inputs, self.outputs = torch.fx.Graph(), {}, original, [], []
        self.nodes, self.transfers, self.parts, self.originals = nodes, transfers, parts, {}
        self.sources, self.decomposed = sources, {}

    def placeholder(self, node, source):
        made = self.graph.placeholder(node.name)
        made.meta.update(node.meta)
        self.env[node], self.originals[made] = made, node
        self.inputs.append(source)
        return made

    def crossing(self, node, name, value, processor, source):
        made = self.graph.placeholder(name)
        made.meta.update(val=_fresh(value), stated=processor)
        if node is not None:
            self.env[node] = made
        self.inputs.append(source)
        return made

    def copy(self):
        for node in self.nodes:
            self.env[node] = self.graph.node_copy(node, lambda n: self.env[n])
        for node in self.nodes:
            _axes_moved(self.env[node], self.env, within=False)
        self.decomposed.update((self.env[node], self.sources[node]) for node in self.nodes if node in self.sources)

    def output(self, values, roles, destination, processor):
        made = self.graph.output(tuple(values))
        made.meta.update(processor=processor, destination=dict(destination))
        self.outputs = list(roles)
        return made

    def finish(self, processor):
        graph, transfers = self.graph, {}
        for sent, arrived in self.transfers.values():
            if sent in self.env:
                channel = len(transfers)
                there, landed = self.env[sent], self.env[arrived]
                there.args, landed.args = (there.args[0], channel, *there.args[2:]), (channel,)
                transfers[channel] = there, landed
        output = graph.output_node()
        for node in [n for n in graph.nodes if n.op == "call_function"]:
            if node in output.all_input_nodes and node.meta["processor"] != processor:
                transfers.update(insert_send_recv(graph, node, len(transfers), {}))
        # an original placeholder's input parts its partitions' images here (each a copy, or a crossing input, of the
        # same pure view or materialized partition); a crossing or state input a whole placeholder: its stated
        # processor, itself and its own value's geometry
        parts = {}
        for made in [n for n in graph.nodes if n.op == "placeholder"]:
            original = self.originals.get(made)
            parts.update(_parts_moved({original: self.parts[original]}, self.env, within=False) if original is not None
                         else {made: _whole(made, made.meta["stated"])})
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
        views(graph)
        graph_module.meta.update(transfers=transfers, invariant=_invariant(graph_module), input_parts=parts,
                                 decomposed=self.decomposed)
        graph_module.recompile()
        return graph_module


def _exported(graph_module, original, roles):
    names = [n.name for n in graph_module.graph.nodes if n.op == "placeholder"]
    state = {name: original.state[name] for name in names if name in original.state}
    returned = [k for k, (kind, _) in enumerate(roles) if kind in RETURNED]
    return Exported(graph_module, state, [name for name in names if name in original.tokens],
                    [name for name in names if name not in state and name not in original.tokens], list(roles),
                    pytree.tree_structure(tuple(returned)), {name: original.state_kinds[name] for name in state})


# the parts of `exported` (function `key`, placed graph `placed`) split at `eigh`: per part (key, Exported, placed
# graph, its inputs, its results), an input ("call", i) the original call's input i or ("part", p, o) part p's output o,
# `results` {output: the original call's output} (export.substitute)
def _outlined(key, exported, placed, eigh, max_iterations):
    graph, processor = placed.graph, eigh.meta["processor"]
    transfers = placed.meta["transfers"]
    arrival = {sent: arrived for sent, arrived in transfers.values()}
    getitems = [u for u in eigh.users if u.target is operator.getitem]
    # rdma/mesh_graph.py Graph.adjacency and Graph.reach: the placed graph's nodes (its output node apart), each one's
    # direct users and a SEND its arrival (placement.py transfers), searched once from the eigh's results' users; the
    # Reached membership is the suffix's, and with the relation and source graph the parts' `outline` record
    relation = mesh_graph.Graph.adjacency([n for n in graph.nodes if n.op != "output"], lambda n: [
        *(u for u in n.users if u.op != "output"), *([arrival[n]] if n in arrival else [])])
    reached = relation.reach([u for g in getitems for u in g.users if u.op != "output"])
    outlined = (placed, eigh, relation, reached)
    body = [n for n in graph.nodes if n.op == "call_function"]
    early = [n for n in body if n not in reached and n is not eigh and n not in getitems]
    late = [n for n in body if n in reached]
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    position = {n: i for i, n in enumerate(placeholders)}
    output = graph.output_node()
    values, destination = list(output.args[0]), output.meta.get("destination", {})
    named = {n.name: n for n in placeholders}
    a, upper, compute_v = (ir_given(eigh)[1].get(name, default) for name, default in
                           (("A", None), ("UPLO", "L"), ("compute_v", True)))
    upper = upper.upper() == "U"
    blocks_value = a.meta["val"]
    n = blocks_value.shape[-1]
    in_late = lambda v: isinstance(v, torch.fx.Node) and (v in reached or v in getitems)
    parts = placed.meta["input_parts"]
    # the placed graph's generated → original map (decompose._expanded: an expanded pinv's eigh among its keys); the
    # eigh's original source node, each prefix and suffix formulation node's
    sources = placed.meta.get("decomposed", {})
    origin = sources.get(eigh, eigh)
    late_used = {p for node in late for p in node.all_input_nodes if p.op == "placeholder"}
    late_used |= {named[destination[k]] for k, v in enumerate(values) if in_late(v) and k in destination}

    # the prefix
    prefix = _Part(exported, early, transfers, parts, sources)
    used = {p for node in early + [eigh] for p in node.all_input_nodes if p.op == "placeholder"}
    used |= {v for k, v in enumerate(values) if isinstance(v, torch.fx.Node) and v.op == "placeholder"
             and not in_late(v)}
    used |= {named[destination[k]] for k, v in enumerate(values) if k in destination and not in_late(v)}
    for p in placeholders:
        if p in used:
            prefix.placeholder(p, ("call", position[p]))
    prefix.copy()
    state = _formulated(prefix.graph, functools.partial(_qdwh_eigh_state, upper=upper, site=eigh.meta["site"]),
                        [prefix.env[a]], eigh, processor, {a: prefix.env[a]}, prefix.decomposed, origin)
    crossing = list(dict.fromkeys(p for node in late for p in node.all_input_nodes
                                  if p.op != "placeholder" and not in_late(p)))
    # each partition of an original placeholder the suffix takes crosses too, so its input parts are its own inputs
    crossing += [part for p in placeholders if p in late_used for _, part, _ in parts[p]
                 if part is not p and part not in crossing]
    early_outputs = [k for k, v in enumerate(values) if not in_late(v)]
    prefix_values = list(state) + [prefix.env[values[k]] if isinstance(values[k], torch.fx.Node) else values[k]
                                   for k in early_outputs]
    prefix_roles = [(OutputKind.USER_OUTPUT, None)] * STATE + [exported.outputs[k] for k in early_outputs]
    prefix_destination = {STATE + j: destination[k] for j, k in enumerate(early_outputs) if k in destination}
    held = {}
    for node in crossing:
        held[node] = len(prefix_values)
        prefix_values.append(prefix.env[node])
        prefix_roles.append((OutputKind.USER_OUTPUT, None))
    prefix.output(prefix_values, prefix_roles, prefix_destination, processor)
    prefix_module = prefix.finish(processor)
    prefix_module.meta["outline"] = outlined
    prefix_key = (key, "prefix", eigh.name)

    # the step
    made = [prefix_module.graph.output_node().args[0][k].meta["val"] for k in range(STATE)]
    step_key = ("qdwh_eigh_step", processor, tuple(blocks_value.shape), str(blocks_value.dtype), max_iterations)
    step = torch.fx.Graph()
    inputs = []
    for name, value in zip(STEP_INPUTS, made):
        node = step.placeholder(name)
        node.meta.update(val=_fresh(value), stated=processor)
        inputs.append(node)
    called = step.call_function(QDWH_EIGH_STEP, tuple(inputs), {"max_iterations": max_iterations})
    with detect_fake_mode(made):
        shapes = QDWH_EIGH_STEP(*[node.meta["val"] for node in inputs], max_iterations=max_iterations)
    called.meta.update({key: value for key, value in eigh.meta.items() if key not in TRACED + VIEWED + AXES},
                       processor=processor, val=tuple(shapes))
    results = []
    for k in range(4):
        node = step.call_function(operator.getitem, (called, k))
        node.meta.update(processor=processor, val=called.meta["val"][k])
        results.append(node)
    _axes_onto(eigh, results[:2], {a: inputs[0]}, within=False)
    step.output(tuple(results)).meta.update(processor=processor, destination={})
    step_module = torch.fx.GraphModule(torch.nn.Module(), step)
    step_module.meta.update(transfers={}, invariant={},
                            input_parts={node: _whole(node, processor) for node in inputs})
    step_module.recompile()
    step_exported = Exported(step_module, {}, [], list(STEP_INPUTS), [(OutputKind.USER_OUTPUT, None)] * 4,
                             pytree.tree_structure((0, 0, 0, 0)), {})

    # the suffix
    calls = 2 * n - 1
    last = calls
    suffix = _Part(exported, late, transfers, parts, sources)
    for p in placeholders:
        if p in late_used:
            suffix.placeholder(p, ("call", position[p]))
    ends = [suffix.crossing(None, name, made[k], processor, ("part", last, k))
            for k, name in ((0, "blocks"), (1, "vectors"))]
    for node in crossing:
        suffix.crossing(node, node.name, node.meta["val"], node.meta["processor"], ("part", 0, held[node]))
    empty = None if compute_v else tuple(eigh.meta["val"][1].shape)
    ordered = _formulated(suffix.graph, functools.partial(_qdwh_eigh_result, empty=empty), ends, eigh, processor,
                          {a: ends[0]}, suffix.decomposed, origin)
    for g in getitems:
        suffix.env[g] = ordered[g.args[1]]
    suffix.copy()
    late_outputs = [k for k, v in enumerate(values) if in_late(v)]
    suffix_values = []
    for k in late_outputs:
        value = suffix.env[values[k]]
        if k in destination and value.op == "placeholder":
            target = suffix.env[named[destination[k]]]
            value = suffix.graph.call_function(COPY, (target, value))
            with detect_fake_mode([target.meta["val"]]):
                value.meta.update(val=COPY(target.meta["val"], suffix.env[values[k]].meta["val"]),
                                  processor=output.meta["processor"])
        suffix_values.append(value)
    suffix.output(suffix_values, [exported.outputs[k] for k in late_outputs],
                  {j: destination[k] for j, k in enumerate(late_outputs) if k in destination},
                  output.meta["processor"])
    suffix_module = suffix.finish(output.meta["processor"])
    suffix_module.meta["outline"] = outlined
    suffix_key = (key, "suffix", eigh.name)

    prefix_exported = _exported(prefix_module, exported, prefix_roles)
    suffix_exported = _exported(suffix_module, exported, suffix.outputs)
    parts = [(prefix_key, prefix_exported, prefix_module, prefix.inputs,
              {STATE + j: k for j, k in enumerate(early_outputs)})]
    for j in range(calls):
        parts.append((step_key, step_exported, step_module,
                      [("part", j, k) for k in range(4)] + [("part", 0, k) for k in range(4, STATE)], {}))
    parts.append((suffix_key, suffix_exported, suffix_module, suffix.inputs,
                  {j: k for j, k in enumerate(late_outputs)}))
    return parts

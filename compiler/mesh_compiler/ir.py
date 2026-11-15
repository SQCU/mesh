# design/data-flow.md
import collections
import operator
import pathlib
import sys
import threading
from dataclasses import dataclass

import torch
from torch import Tensor
from torch._guards import detect_fake_mode
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch.export.graph_signature import OutputKind
from torch.fx.node import map_arg
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import get_alias_info

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

LOCAL = threading.local()
HOST = torch.device("cpu")


@torch.library.custom_op("mesh::send", mutates_args=())
def send(value: Tensor, channel: int) -> None:
    LOCAL.channels[channel].put(value)


@send.register_fake
def send_fake(value, channel):
    return None


@torch.library.custom_op("mesh::recv", mutates_args=())
def recv(channel: int) -> Tensor:
    return LOCAL.channels[channel].get()


_register_effectful_op(send, _EffectType.ORDERED)
_register_effectful_op(recv, _EffectType.ORDERED)
SEND = torch.ops.mesh.send.default
RECV = torch.ops.mesh.recv.default


# rdma/mesh-collective.h MESH_JOIN, MESH_PAGES
JOIN, PAGES = 0, 1


def channel_of(node):
    if node.target is SEND:
        return node.args[1]
    return node.args[0] if node.target is RECV else None


@dataclass
class Exported:
    graph_module: torch.fx.GraphModule
    state: dict
    tokens: list
    user_inputs: list
    outputs: list
    out_spec: object
    state_kinds: dict
    # compiler/patches/torch-export-alias-metadata.patch; compiler/mesh_compiler/export.py export: compilation only, the
    # export's AOT forward metadata (ViewAndMutationMeta), each AOT input's (InputKind, state target or user input
    # position), each AOT input's and output's prototype value
    aliases: tuple = None
    # Python inspect Signature.bind and BoundArguments; torch/utils/_pytree.py tree_structure: an outer captured
    # program's call interface, the inspect.Signature of exactly its compiled slot parameters (export.capture) and the
    # TreeSpec of their bound values in parameter order (one child a parameter); None for a function's own export
    interface: tuple = None
    # torch 2.14.0 torch/export/graph_signature.py InputSpec ConstantArgument; compiler/mesh_compiler/export.py read and
    # capture: each specialized literal user input as (its input placeholder's name, the literal compiled for it); a
    # function's from its own ConstantArgument input specs, an outer program's each outer user placeholder some call's
    # ConstantArgument input came from, in user input order
    specialized: tuple = ()
    # compiler/mesh_compiler/export.py export: the final program's own AOT record (run_decompositions' aot_aliases, as
    # `aliases` is: its ViewAndMutationMeta, input roles and input and output prototypes), a stage result distinct from
    # `aliases` (the forward epilogue's authority) where a loss's forward and backward record is that one, else the same
    # object as `aliases`
    decomposed: tuple = None
    # torch 2.14.0 torch/export/graph_signature.py ExportGraphSignature: the final program's own signature as its export
    # returned it (each InputSpec's kind, argument, target and persistent; each OutputSpec's kind, argument and target),
    # the source record the fields above project; None for a program no torch.export produced (export.capture's outer
    # program, a part decompose.py builds)
    signature: object = None
    # torch 2.14.0 torch/export/exported_program.py ExportedProgram.range_constraints, torch/export/_trace.py
    # _get_inline_constraints and _get_range_constraints, torch/_export/non_strict_utils.py make_constraints: the
    # program's range constraints as their own producer returned them, a function's its ExportedProgram's, an outer
    # program's make_constraints' over its capture's own FakeTensorMode, outer GraphModule (its inline_constraints the
    # capture ShapeEnv's unbacked symbol ranges) and bound arguments (export.capture), the constraints
    # torch/_export/utils.py _check_input_constraints_for_graph reads (runtime.Arguments); None for a part decompose.py
    # builds
    range_constraints: dict = None


# torch 2.14.0 torch/export/graph_signature.py OutputKind: the outputs a program returns to its caller, in output order
RETURNED = (OutputKind.USER_OUTPUT, OutputKind.LOSS_OUTPUT, OutputKind.GRADIENT_TO_PARAMETER,
            OutputKind.GRADIENT_TO_USER_INPUT)


# design/unequal-shards-change-request.md item 8; torch 2.14.0 torch/export/experimental/__init__.py
# _export_forward_backward; torch/export/graph_signature.py ExportBackwardSignature gradients_to_parameters and
# gradients_to_user_inputs; torch/utils/_pytree.py _register_namedtuple: a program with a loss returns its forward
# outputs, its parameters' gradients by target and its user inputs' gradients by target
TrainingResult = collections.namedtuple("TrainingResult", "outputs parameters inputs")
pytree._register_namedtuple(TrainingResult, serialized_type_name="mesh_compiler.ir.TrainingResult")


# design/aot-compilation-change-request.md A10; P. Barham et al., Pathways, arXiv:2203.12533, section 3: one original
# call node's captured function (its CapturedProgram.functions key), its inputs one outer FX Node or literal per
# placeholder of that function in order, its outputs one outer FX Node or literal per output spec of that function
@dataclass
class Call:
    function: tuple
    inputs: list
    outputs: list


# design/aot-compilation-change-request.md A1 and A10: the outer call graph (its flat inputs, state, outputs and their
# roles), each captured function's Exported by key in first-call order, each original call node's Call in program order
@dataclass
class CapturedProgram:
    exported: Exported
    functions: dict
    calls: dict
    # compiler/mesh_compiler/export.py substitute: each function an outline replaced, in replacement order, as (its
    # original Exported, its parts' own Exported objects in part order, a repeated step's the same object each time):
    # the original source owner of its parts (its signature and AOT stage records among it), once; a part a later
    # outline replaced its own entry's original, the same object
    outlined: tuple = ()


# rdma/mesh-collective.h struct mesh_function, mesh_plan, mesh_plan_save
@dataclass
class Plan:
    program: int
    channels: list
    declared: list
    weights: list
    values: list
    dependences: list
    outstanding: int
    tasks: list
    before: list
    senders: list
    receivers: list
    resources: int
    completion: list
    order: list
    record: bytes
    machine: int
    places: list
    sections: list
    signature: bytes = b""
    nodes: tuple = ()
    arena_bytes: int = 0
    offsets: tuple = ()
    publications: tuple = ()


# rdma/mesh-dataflow.h mesh_view_whole, mesh_nodes_read; rdma/mesh-collective.h mesh_plan;
# compiler/mesh_compiler/schedule.py planned, solved
@dataclass
class Planning:
    image: object
    context: object
    nodes: tuple
    here: int
    group: int
    program: int


# compiler/mesh_compiler/driver.py compile_captured: `modules` {function key: {machine group: its orchestration body}};
# `devices`, `machines` and `engines` the full original per-processor maps (every processor's torch.device, machine
# and, for a Neural Engine processor, its engine), keyed as `processor_modules` is; grouping is `modules`, `functions`
# and each body's meta["group"]
@dataclass
class Scheduled:
    exported: Exported
    modules: dict
    devices: dict
    machines: dict
    output_processor: str
    channels: int
    inputs: dict
    feedforward: list
    carried: list
    engines: dict
    plan: object = None
    signature: bytes = None
    stored: dict = None
    invariant: dict = None
    functions: dict = None
    # compilation only (compiler/mesh_compiler/driver.py): {(function key, machine group): that body's
    # lower_metal.py construct operands}
    compilation: dict = None
    # compiler/mesh_compiler/schedule.py schedule, its third result: {function key: {actual processor: its own body}},
    # each body's meta["group"] its machine group and meta["copied"] the original source nodes' copies (the same
    # originals as `modules[key][group].meta["copied"]`), its tasks, relation, transfers and interface; `modules` the
    # machine orchestration of those same originals. The numerical lowerer's operand (lower_inductor.py lower) and the
    # archive's retained source bodies (archive.py)
    processor_modules: dict = None
    operands: object = None


# design/problems-of-14b9a757-change-request.md items 1 to 4, 6 and 8; design/review-warnings.md rule 5
ATEN = torch.ops.aten
CAT, PAD, RESHAPE, VIEW = ATEN.cat.default, ATEN.constant_pad_nd.default, ATEN.reshape.default, ATEN.view.default
CLONE, EMPTY = ATEN.clone.out, ATEN.empty.memory_format


# a node's value's shape: a Tensor value's own; a list or tuple value's result `k`'s (its own producer FakeTensor's,
# each result its own shape), `k` a getitem index or a result's key path (torch/utils/_pytree.py key_get over its
# tree_flatten_with_path KeyPath: placement.result_paths)
def shape(node, k=None):
    value = node.meta["val"]
    if isinstance(k, tuple):
        value = pytree.key_get(value, k)
    elif isinstance(value, (tuple, list)):
        value = value[k]
    return tuple(value.shape)


# design/problems-of-14b9a757-change-request.md item 6; torch 2.14.0 torch/_tensor_docs.py view, reshape and contiguous
def viewable(value, sizes):
    try:
        value.view(sizes)
    except (RuntimeError, ValueError):
        return False
    return True


def evaluated(node, processor):
    arguments = map_arg((node.args, node.kwargs), lambda n: n.meta["val"])
    with detect_fake_mode(arguments):
        if node.target in (RESHAPE, VIEW) and not viewable(*arguments[0]):
            with node.graph.inserting_before(node):
                held = node.args[0]
                out = constant(node.graph, EMPTY, (list(shape(held)),), held, processor, dtype=held.meta["val"].dtype)
                copied = evaluated(node.graph.call_function(CLONE, (held,), {
                    "memory_format": torch.contiguous_format, "out": out}), processor)
            node.args = (copied, *node.args[1:])
            return evaluated(node, processor)
        node.meta.update(processor=processor, val=node.target(*arguments[0], **arguments[1]))
    return node


# design/problems-of-14b9a757-change-request.md items 3 and 6; torch 2.14.0 torch/utils/_python_dispatch.py
# get_alias_info (SchemaInfo: torchgen's parse of an ATen schema, TorchScript's of another, each argument's and result's
# alias set and write, read_only_alias_match_indexes each (argument, result) a result aliasing an argument it does not
# write, outs_write_aliases each result's written alias): an operator call's alias information, None for any other node
def alias_info(node):
    return get_alias_info(node.target) if node.op == "call_function" and isinstance(
        node.target, torch._ops.OpOverload) else None


# a call any of whose results aliases an argument without writing it
def aliasing(node):
    info = alias_info(node)
    return info is not None and bool(info.read_only_alias_match_indexes)


# the schema result describing result `k` of a call of schema `info` (its one list result, Tensor(a)[], every element's)
def result_of(info, k):
    return 0 if len(info.outs) == 1 else k


# each result index of `node`'s value: every element of a list or tuple, else its one
def results_of(node):
    value = node.meta.get("val")
    return range(len(value)) if isinstance(value, (tuple, list)) else range(1)


# each item once by identity, in order (a Tensor[] argument's list unhashable)
def once(items):
    kept = []
    for item in items:
        if all(item is not other for other in kept):
            kept.append(item)
    return tuple(kept)


# the FX Node leaves of argument values (a Tensor argument its Node, a Tensor[] argument each of its Nodes), once each
def leaves_of(values):
    return once(leaf for value in values for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.fx.Node))


# torch 2.14.0 torch/utils/_python_dispatch.py _correct_storage_aliasing (every read_only_alias_match_indexes entry):
# the argument values result `k` of call `node` aliases without writing them, each of that result's read-only matches'
# args[arg_idx], in match order (a list result's every element each of them); () where none
def aliased(node, k=0):
    info = alias_info(node)
    if info is None:
        return ()
    _, read = given(node)
    return once(read[info.args[a].name] for a, r in info.read_only_alias_match_indexes
                if r == result_of(info, k) and info.args[a].name in read)


# ExecuTorch documentation, Concepts, Out variant; torch/utils/_python_dispatch.py return_and_correct_aliasing (an
# in-place or out= call returns its written argument itself): the argument values result `k` of call `node` writes,
# each one written with that result's alias (get_alias_info's outs_write_aliases), of a list result its element k (the
# result is that list); () where it writes none
def written(node, k=0):
    info = alias_info(node)
    if info is None or info.outs_write_aliases is None:
        return ()
    _, read = given(node)
    alias = info.outs_write_aliases[result_of(info, k)]
    return once(held[k] if isinstance(held, (tuple, list)) else held for held in (
        read[a.name] for a in info.args if a.is_write and alias in a.alias_set and a.name in read))


# result `k` of `node`'s relation: the argument values that result aliases (`aliased`), else writes (`written`); a
# getitem's its call's result at its index, its own result
def related(node, k=0):
    if node.target is operator.getitem:
        return related(node.args[0], node.args[1])
    return aliased(node, k) or written(node, k)


# the argument Nodes `node`'s value aliases, each once: every result's relation's leaves (`related`, `leaves_of`)
def source(node):
    if node.target is operator.getitem:
        return leaves_of(related(node))
    return leaves_of(value for k in results_of(node) for value in related(node, k))


# each result of `node` its own view: of its relation's Node leaves, each its value aliases (torch._C._is_alias_of: the
# producer FakeTensors' own storage), every one so; a getitem's its own result's
def view(node):
    value = node.meta["val"]
    if node.target is operator.getitem or not isinstance(value, (tuple, list)):
        results = [(value, related(node))]
    else:
        results = [(value[k], related(node, k)) for k in results_of(node)]
    return once(n for result, values in results for n in leaves_of(values)
                if isinstance(result, torch.Tensor) and torch._C._is_alias_of(result, n.meta["val"]))


# a Tensor value's bases (its `view`); a list or tuple value's results each have their own (their getitems')
def bases_of(node):
    return list(view(node)) if isinstance(node.meta.get("val"), torch.Tensor) else []


def based(node, read, destined=False):
    value = node.meta.get("val")
    if not read:
        return node, value.storage_offset() if isinstance(value, torch.Tensor) else 0
    # every base of a value one root at one offset (they alias one storage: ir.view)
    found = set(read)
    if len(found) != 1:
        raise ValueError(f"{node.name}: its bases {sorted(found)} are no one root and offset")
    base, first = found.pop()
    if isinstance(value, torch.Tensor):
        record = base.name, value.storage_offset() - first, tuple(value.shape), tuple(value.stride())
        if destined or written(node):
            node.meta["out"] = record
            return base, first
        node.meta["view"] = record
        if "state" in base.meta:
            node.meta["invariant"] = base.meta["state"], *node.meta["view"][1:]
    return base, first


# design/problems-of-14b9a757-change-request.md item 1 step 1, item 2 and item 4 step 4; MPI 4.1, section 7.7
def destined(node):
    if node.target is CAT:
        value = node.meta["val"]
        along, at = (*node.args, 0)[1] % value.dim(), 0
        for held in node.args[0]:
            if node.args[0].count(held) == 1:
                yield held, at * value.stride(along)
            at += shape(held)[along]
    elif node.target is PAD and (*node.args, 0)[2] == 0 and min(node.args[1]) >= 0:
        value = node.meta["val"]
        yield node.args[0], sum(low * value.stride(value.dim() - 1 - d) for d, low in enumerate(node.args[1][::2]))


# design/problems-of-14b9a757-change-request.md items 1 to 4 and 6, and item 8, step 4;
# design/compilation-composition-change-request.md items 6 and 7; torch 2.14.0 torch/export/graph_signature.py
# OutputKind.USER_INPUT_MUTATION and BUFFER_MUTATION: the value now at each position of an output node's meta
# "destination" (export.read: the position's target placeholder) computes into that placeholder: its value that
# placeholder's own view, its "out" record (a received value's "to") its offset, shape and strides there, its aliases
# re-evaluated from it; a gathered value's pieces projected by `destined`, consumers first in the graph's reverse
# order, into its parent's own root where the parent has an "out" (root, parent offset + piece offset, the root's
# strides, the piece's value that view of the root), else into the parent
def views(nodes):
    starts = list(getattr(nodes, "nodes", nodes))
    universe = lambda: [n for graph in dict.fromkeys(node.graph for node in starts) for n in graph.nodes]
    named, destination = {n.name: n for n in universe()}, {}
    for graph in dict.fromkeys(node.graph for node in starts):
        output = graph.output_node()
        for k, name in output.meta.get("destination", {}).items():
            value = output.args[0][k]
            if isinstance(value, torch.fx.Node) and value.op != "placeholder" and not written(value):
                destination[value] = named[name]
    nodes = mesh_graph.walk(universe(), starts, lambda n: [u for u in n.users if n in source(u)])
    inside = mesh_graph.Reached(nodes.owner)
    for node in mesh_graph.postorder(nodes, nodes[::-1], lambda n: [m for m in source(n) if m in inside]):
        node.meta.pop("to", None)
        node.meta.pop("out", None)
        if node in destination:
            target = destination[node].meta["val"]
            with detect_fake_mode([target]):
                node.meta["val"] = target.as_strided(target.shape, target.stride(), target.storage_offset())
        elif aliasing(node):
            evaluated(node, node.meta["processor"])
        elif node.target is operator.getitem and aliasing(node.args[0]):
            node.meta["val"] = node.args[0].meta["val"][node.args[1]]
    computed = {node: target for node, target in destination.items() if node.target is not RECV}
    bases = lambda n: [computed[n]] if n in computed else bases_of(n)
    mesh_graph.forward(mesh_graph.postorder(universe(), nodes, bases), bases,
                       lambda node, read: based(node, read, node in computed))
    for node, target in destination.items():
        if node.target is RECV:
            node.meta["to"] = target.name, 0, shape(target), tuple(target.meta["val"].stride())
    readers = dict.fromkeys(reader for n in nodes for reader in (n, *n.users))
    for node in [n for n in reversed(universe()) if n in readers]:
        record = node.meta.get("out")
        for held, offset in destined(node):
            if (held.op == "call_function" and held.target is not operator.getitem and len(held.users) == 1
                    and not {"view", "out"} & held.meta.keys()
                    and isinstance(held.meta["val"], torch.Tensor)):
                key = "to" if held.target is RECV else "out"
                if record is None:
                    held.meta[key] = node.name, offset, shape(held), tuple(node.meta["val"].stride())
                    continue
                root = named[record[0]].meta["val"]
                held.meta[key] = record[0], record[1] + offset, shape(held), record[3]
                with detect_fake_mode([root]):
                    held.meta["val"] = root.as_strided(shape(held), record[3],
                                                       root.storage_offset() + record[1] + offset)


def constant(graph, target, arguments, like, processor, **kwargs):
    value = like.meta["val"]
    node = graph.call_function(target, arguments, {**kwargs, "device": value.device})
    with value.fake_mode:
        node.meta.update(processor=processor, val=target(*arguments, **node.kwargs))
    return node


def given(node):
    names = [argument.name for argument in node.target._schema.arguments]
    return names, {**dict(zip(names, node.args)), **node.kwargs}

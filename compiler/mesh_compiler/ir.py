# design/data-flow.md
import threading
from dataclasses import dataclass

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

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


# design/synchronization-signature-change-request.md item 1; compiler/mesh_compiler/runtime.py Program;
# compiler/mesh_compiler/signature.py handed
def call(scheduled, machine):
    local = [processor for processor in scheduled.inputs if scheduled.machines[processor] == machine]
    caller = scheduled.output_processor if scheduled.output_processor in local else next(iter(local), None)
    others = [processor for processor in local if processor != caller]
    started = scheduled.plan is not None
    return caller, others, others if caller != scheduled.output_processor or started else [], started


# design/problems-of-14b9a757-change-request.md items 1 to 4, 6 and 8; design/review-warnings.md rule 5
import operator  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

from torch._guards import detect_fake_mode  # noqa: E402
from torch.fx.node import map_arg  # noqa: E402

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph  # noqa: E402

ATEN = torch.ops.aten
CAT, PAD, RESHAPE, VIEW = ATEN.cat.default, ATEN.constant_pad_nd.default, ATEN.reshape.default, ATEN.view.default
CLONE, EMPTY = ATEN.clone.out, ATEN.empty.memory_format


def shape(node):
    value = node.meta["val"]
    return tuple((value[0] if isinstance(value, (tuple, list)) else value).shape)


# design/problems-of-14b9a757-change-request.md item 6; torch 2.15 torch/_tensor_docs.py view, reshape and contiguous
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


# design/problems-of-14b9a757-change-request.md items 3 and 6; torch/csrc/jit/frontend/function_schema_parser.cpp
# alias_info
def aliasing(node):
    returns = getattr(getattr(node.target, "_schema", None), "returns", ())
    return node.op == "call_function" and bool(returns) and returns[0].alias_info is not None and not returns[
        0].alias_info.is_write


# torch/csrc/jit/frontend/function_schema_parser.cpp alias_info; ExecuTorch documentation, Concepts, Out variant
def written(node):
    schema = getattr(node.target, "_schema", None)
    if node.op != "call_function" or schema is None or len(schema.returns) != 1 or schema.returns[0].alias_info is None:
        return None
    names, read = given(node)
    return next((read[a.name] for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write and
                 a.alias_info.before_set == schema.returns[0].alias_info.before_set and a.name in read), None)


def source(node):
    made = node.args[0] if node.target is operator.getitem else node
    return made.args[0] if aliasing(made) else written(made)


def view(node):
    held = source(node)
    if held is None:
        return None
    value = node.meta["val"]
    return held if all(torch._C._is_alias_of(v, held.meta["val"]) for v in (
        value if isinstance(value, (tuple, list)) else [value])) else None


def bases_of(node):
    return [held for held in [view(node)] if held is not None]


def based(node, read):
    value = node.meta.get("val")
    if not read:
        return node, value.storage_offset() if isinstance(value, torch.Tensor) else 0
    (base, first), = read
    if isinstance(value, torch.Tensor):
        record = base.name, value.storage_offset() - first, tuple(value.shape), tuple(value.stride())
        if written(node) is not None:
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


# design/problems-of-14b9a757-change-request.md items 1 to 4 and 6, and item 8, step 4
def views(nodes):
    nodes = mesh_graph.walk(list(getattr(nodes, "nodes", nodes)), lambda n: [u for u in n.users if source(u) is n])
    inside = set(nodes)
    for node in mesh_graph.postorder(nodes[::-1], lambda n: [m for m in [source(n)] if m in inside]):
        node.meta.pop("to", None)
        node.meta.pop("out", None)
        if aliasing(node):
            evaluated(node, node.meta["processor"])
        elif node.target is operator.getitem and aliasing(node.args[0]):
            node.meta["val"] = node.args[0].meta["val"][node.args[1]]
    mesh_graph.forward(mesh_graph.postorder(nodes, bases_of), bases_of, based)
    for node in dict.fromkeys(reader for n in nodes for reader in (n, *n.users)):
        for held, offset in destined(node):
            if (held.op == "call_function" and held.target is not operator.getitem and len(held.users) == 1
                    and not {"view", "out"} & held.meta.keys()
                    and isinstance(held.meta["val"], torch.Tensor)):
                held.meta["to" if held.target is RECV else "out"] = (
                    node.name, offset, shape(held), tuple(node.meta["val"].stride()))


def constant(graph, target, arguments, like, processor, **kwargs):
    value = like.meta["val"]
    node = graph.call_function(target, arguments, {**kwargs, "device": value.device})
    with value.fake_mode:
        node.meta.update(processor=processor, val=target(*arguments, **node.kwargs))
    return node


def given(node):
    names = [argument.name for argument in node.target._schema.arguments]
    return names, {**dict(zip(names, node.args)), **node.kwargs}

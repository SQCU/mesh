# design/data-flow.md; design/torch-source-lowering-change-request.md item 3
import copy
import pathlib
import sys

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch._inductor import ir as inductor
from torch._inductor.dependencies import WeakDep
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.virtualized import V

from .ir import RECV, SEND

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

WITH_EFFECTS = torch.ops.higher_order.with_effects
COPY = torch.ops.aten.copy_.default


@torch.library.custom_op("mesh::recv_strided", mutates_args=())
def recv_strided(channel: int, shape: list[int], stride: list[int], dtype: torch.dtype,
                 device: torch.device, after: list[Tensor]) -> Tensor:
    return RECV(channel)


@recv_strided.register_fake
def recv_strided_fake(channel, shape, stride, dtype, device, after):
    return torch.empty_strided(shape, stride, dtype=dtype, device=device)


_register_effectful_op(recv_strided, _EffectType.ORDERED)
RECV_STRIDED = torch.ops.mesh.recv_strided.default
call_recv_strided = fallback_handler(RECV_STRIDED, add_to_fallback_set=False)


@torch.library.custom_op("mesh::send_strided", mutates_args=())
def send_strided(value: Tensor, channel: int, stride: list[int], after: list[Tensor]) -> None:
    SEND(value, channel)


@send_strided.register_fake
def send_strided_fake(value, channel, stride, after):
    return None


_register_effectful_op(send_strided, _EffectType.ORDERED)
SEND_STRIDED = torch.ops.mesh.send_strided.default
call_send_strided = fallback_handler(SEND_STRIDED, add_to_fallback_set=False)


# design/torch-source-lowering-change-request.md item 3; torch/_inductor/lowering.py control_deps_op_lowering
def in_order(call, after, *arguments):
    for value in after:
        value.realize()
    first = len(V.graph.operations)
    result = call(*arguments, [])
    for made in V.graph.operations[first:]:
        V.graph.additional_buffer_deps[made.get_operation_name()].update(value.get_name() for value in after)
    return result


# torch/_inductor/graph.py never_reuse_but_free_buffers
@register_lowering(RECV_STRIDED, type_promotion_kind=None)
def lower_recv_strided(*arguments):
    result = in_order(call_recv_strided, arguments[-1], *arguments[:-1])
    V.graph.never_reuse_but_free_buffers.add(result.get_name())
    return result


# design/compilation-composition-change-request.md item 2; torch/_inductor/ir.py ExternKernel.require_exact_strides
@register_lowering(SEND_STRIDED, type_promotion_kind=None)
def lower_send_strided(value, channel, stride, after):
    first = len(V.graph.operations)
    stated = inductor.ExternKernel.require_exact_strides(value, stride)
    if len(V.graph.operations) != first:
        raise NotImplementedError(f"channel {channel}: the strides {stride} of the sent value are a copy of the "
                                  f"compiled value, which the scheduled graph does not hold")
    return in_order(call_send_strided, after, stated, channel, stride)


# torch 2.15 aten/src/ATen/native/native_functions.yaml, the overloads that take `out`;
# torch/_dynamo/variables/torch.py, "Attempted to call op with non-contiguous `out=` tensor"
def functional(target):
    given = [(a.name, str(a.type)) for a in target._schema.arguments[:-1]]
    overloads = (getattr(target.overloadpacket, name) for name in target.overloadpacket.overloads())
    return next(over for over in overloads if [(a.name, str(a.type)) for a in over._schema.arguments] == given)


# design/torch-source-lowering-change-request.md item 3; rdma/mesh_graph.py extremes
def adapt(module, device, carried, laid):
    module = copy.deepcopy(module)
    graph = module.graph
    sent, receives, written, parts = [], False, {}, set()
    named = {node.name: node for node in graph.nodes if node.op == "call_function"}
    read = lambda node: [user for user in node.users if user in written]
    for node in list(graph.nodes):
        if node in parts:
            continue
        if node.target is WITH_EFFECTS:
            token, operator, *arguments = node.args
            with graph.inserting_before(node):
                call = graph.call_function(operator, tuple(arguments), dict(node.kwargs))
            for getitem in list(node.users):
                if getitem.args[1] > 1:
                    raise NotImplementedError(f"{operator}: effectful operator with several results")
                if getitem.args[1] == 1:
                    call.meta["val"] = getitem.meta.get("val")
                getitem.replace_all_uses_with(token if getitem.args[1] == 0 else call)
                graph.erase_node(getitem)
                parts.add(getitem)
            graph.erase_node(node)
            node = call
        elif node.target is SEND:
            sent += [named[name] for name in node.args[0].meta.get("view", ())[:1] if name in named] + [node.args[0]]
            node.target = SEND_STRIDED
            node.args = (*node.args, list(node.args[0].meta["val"].stride()), mesh_graph.extremes(list(written), read)[1])
            written = {}
        elif node.target is RECV:
            channel, value = node.args[0], node.meta["val"]
            node.target = RECV_STRIDED
            node.args = (channel, list(value.shape), laid.get(channel, list(value.stride())), value.dtype,
                         torch.device("cpu") if channel in carried else device, mesh_graph.extremes(list(written), read)[1])
            receives, written = True, {}
        elif node.op == "call_function" and "out" in node.kwargs:
            with graph.inserting_before(node):
                made = graph.call_function(functional(node.target), node.args,
                                           {key: value for key, value in node.kwargs.items() if key != "out"})
            made.meta["val"], written[made] = node.meta["val"], None
            node.target, node.args, node.kwargs = COPY, (node.kwargs["out"], made), {}
        if node.op == "call_function" and isinstance(node.meta.get("val"), Tensor):
            written[node] = None
    output = next(n for n in graph.nodes if n.op == "output")
    output.args = (tuple(output.args[0]) + tuple(dict.fromkeys(sent)),)
    module.recompile()
    return module, receives


# design/torch-source-lowering-change-request.md:250; torch/_inductor/config.py _pre_fusion_custom_pass;
# torch/_inductor/scheduler.py Scheduler.compute_dependencies, additional_buffer_deps
def chained(nodes):
    for written, following in zip(nodes, nodes[1:]):
        names = written.get_buffer_names()
        if names and not names & {dep.name for dep in following.unmet_dependencies}:
            following.add_fake_dep(WeakDep(next(iter(names)), following.get_name(), is_fake=True))
    for node in nodes[:1]:
        node.scheduler.compute_ancestors()
    return nodes


# design/torch-source-lowering-change-request.md item 1; compiler/mesh_compiler/driver.py lowered
def lower(scheduled):
    compiled, carried = {}, {channel for channel, *_ in scheduled.carried}
    laid = {node.args[1]: list(node.args[0].meta["val"].stride()) for module in scheduled.modules.values()
            for node in module.graph.nodes if node.target is SEND}
    for processor, module in scheduled.modules.items():
        if scheduled.devices[processor].type == "mps":
            compiled[processor] = module
            continue
        adapted, receives = adapt(module, scheduled.devices[processor], carried, laid)
        options = {"layout_optimization": False, "reorder_for_peak_memory": False, "_pre_fusion_custom_pass": chained,
                   **({"assume_unaligned_fallback_output": True} if receives else {})}
        compiled[processor] = torch.compile(adapted, backend="inductor", options=options)
    return compiled

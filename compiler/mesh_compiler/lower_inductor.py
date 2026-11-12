# design/data-flow.md; design/torch-source-lowering-change-request.md item 3
import copy
import weakref

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch._inductor import ir as inductor
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.virtualized import V

from .ir import RECV, SEND

WITH_EFFECTS = torch.ops.higher_order.with_effects
RECEIVED = weakref.WeakKeyDictionary()


@torch.library.custom_op("mesh::recv_strided", mutates_args=())
def recv_strided(channel: int, shape: list[int], stride: list[int], dtype: torch.dtype,
                 device: torch.device) -> Tensor:
    return RECV(channel)


@recv_strided.register_fake
def recv_strided_fake(channel, shape, stride, dtype, device):
    return torch.empty_strided(shape, stride, dtype=dtype, device=device)


_register_effectful_op(recv_strided, _EffectType.ORDERED)
RECV_STRIDED = torch.ops.mesh.recv_strided.default
call_recv_strided = fallback_handler(RECV_STRIDED, add_to_fallback_set=False)


@torch.library.custom_op("mesh::send_strided", mutates_args=())
def send_strided(value: Tensor, channel: int, stride: list[int]) -> None:
    SEND(value, channel)


@send_strided.register_fake
def send_strided_fake(value, channel, stride):
    return None


_register_effectful_op(send_strided, _EffectType.ORDERED)
SEND_STRIDED = torch.ops.mesh.send_strided.default
call_send_strided = fallback_handler(SEND_STRIDED, add_to_fallback_set=False)


# design/torch-source-lowering-change-request.md item 3; torch/_inductor/lowering.py control_deps_op_lowering;
# torch/_inductor/graph.py mark_buffer_mutated
def in_order(call, *arguments):
    readers = [user for name in RECEIVED.setdefault(V.graph, []) for user in V.graph.name_to_users[name]]
    for reader in readers:
        reader.realize()
    first = len(V.graph.operations)
    result = call(*arguments)
    for made in V.graph.operations[first:]:
        V.graph.additional_buffer_deps[made.get_operation_name()].update(reader.get_name() for reader in readers)
    return result


@register_lowering(RECV_STRIDED, type_promotion_kind=None)
def lower_recv_strided(*arguments):
    result = in_order(call_recv_strided, *arguments)
    V.graph.never_reuse_but_free_buffers.add(result.get_name())
    RECEIVED[V.graph].append(result.get_name())
    return result


# design/compilation-composition-change-request.md item 2; design/data-flow.md, the operator, 2026-11-09 22:19;
# torch/_inductor/ir.py ExternKernel.require_strides
@register_lowering(SEND_STRIDED, type_promotion_kind=None)
def lower_send_strided(value, channel, stride):
    first = len(V.graph.operations)
    stated = inductor.ExternKernel.require_exact_strides(value, stride)
    if len(V.graph.operations) != first:
        raise NotImplementedError(f"channel {channel}: the strides {stride} of the sent value are a copy of the "
                                  f"compiled value, which the scheduled graph does not hold")
    return in_order(call_send_strided, stated, channel, stride)


def adapt(module, device, carried):
    module = copy.deepcopy(module)
    graph = module.graph
    sent, receives = [], False
    for node in list(graph.nodes):
        if node.target is WITH_EFFECTS:
            token, operator, *arguments = node.args
            with graph.inserting_before(node):
                call = graph.call_function(operator, tuple(arguments), dict(node.kwargs))
            for getitem in list(node.users):
                if getitem.args[1] > 1:
                    raise NotImplementedError(f"{operator}: effectful operator with several results")
                getitem.replace_all_uses_with(token if getitem.args[1] == 0 else call)
                graph.erase_node(getitem)
            graph.erase_node(node)
        elif node.target is SEND:
            sent.append(node.args[0])
            node.target, node.args = SEND_STRIDED, (*node.args, list(node.args[0].meta["val"].stride()))
        elif node.target is RECV:
            channel, value = node.args[0], node.meta["val"]
            node.target = RECV_STRIDED
            node.args = (channel, list(value.shape), list(value.stride()), value.dtype,
                         torch.device("cpu") if channel in carried else device)
            receives = True
    output = next(n for n in graph.nodes if n.op == "output")
    output.args = (tuple(output.args[0]) + tuple(dict.fromkeys(sent)),)
    module.recompile()
    return module, receives


def lower(scheduled):
    compiled, carried = {}, {channel for channel, *_ in scheduled.carried}
    for processor, module in scheduled.modules.items():
        adapted, receives = adapt(module, scheduled.devices[processor], carried)
        options = {"layout_optimization": False, **({"assume_unaligned_fallback_output": True} if receives else {})}
        compiled[processor] = torch.compile(adapted, backend="inductor", options=options)
    return compiled

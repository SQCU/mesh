# design/data-flow.md
import copy

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.virtualized import V

from .ir import RECV, SEND

WITH_EFFECTS = torch.ops.higher_order.with_effects


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


@register_lowering(RECV_STRIDED, type_promotion_kind=None)
def lower_recv_strided(*arguments):
    result = call_recv_strided(*arguments)
    V.graph.never_reuse_but_free_buffers.add(result.get_name())
    return result


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

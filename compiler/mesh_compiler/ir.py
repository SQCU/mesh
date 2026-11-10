# design/data-flow.md
import threading
from dataclasses import dataclass

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

LOCAL = threading.local()


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


@dataclass
class Scheduled:
    exported: Exported
    modules: dict
    devices: dict
    output_processor: str
    channels: int
    inputs: dict
    feedforward: list

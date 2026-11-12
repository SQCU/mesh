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

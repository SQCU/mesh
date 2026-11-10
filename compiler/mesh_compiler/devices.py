# design/data-flow.md
import copy

import torch

from .ir import RECV, SEND

TO_COPY = torch.ops.aten._to_copy.default


def on_devices(graph_module, devices):
    graph_module = copy.deepcopy(graph_module)
    graph = graph_module.graph
    nodes = [n for n in graph.nodes if n.op != "placeholder"]
    receiver = {n.args[0]: n for n in nodes if n.target is RECV}
    for node in nodes:
        device = devices[node.meta["processor"]]
        if "device" in node.kwargs:
            node.kwargs = {**node.kwargs, "device": device}
        if node.target is torch.ops.aten.to.device:
            node.args = (node.args[0], device, *node.args[2:])
        if node.target is SEND:
            value, channel = node.args
            there = devices[receiver[channel].meta["processor"]]
            if there != device:
                with graph.inserting_before(node):
                    moved = graph.call_function(TO_COPY, (value,), {"device": there})
                with value.meta["val"].fake_mode:
                    arrives = TO_COPY(value.meta["val"], device=there)
                moved.meta.update(processor=node.meta["processor"], val=arrives)
                receiver[channel].meta["val"] = arrives
                node.args = (moved, channel)
    graph_module.recompile()
    return graph_module

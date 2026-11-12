# design/data-flow.md
import copy

import torch

from .ir import HOST, JOIN, RECV, SEND, views

TO_COPY = torch.ops.aten._to_copy.default
ENGINES = ("ane",)


def copied(graph, value, device, processor):
    node = graph.call_function(TO_COPY, (value,), {"device": device})
    with value.meta["val"].fake_mode:
        arrives = TO_COPY(value.meta["val"], device=device)
    node.meta.update(processor=processor, val=arrives, custom={"between": True})
    return node


# compiler/mesh_compiler/ir.py views
def on_devices(graph_module, processors):
    graph_module = copy.deepcopy(graph_module)
    graph = graph_module.graph
    nodes = [n for n in graph.nodes if n.op != "placeholder"]
    receiver = {n.args[0]: n for n in nodes if n.target is RECV}
    rewired = list(receiver.values())
    for node in nodes:
        machine, device = processors[node.meta["processor"]]
        if "device" in node.kwargs:
            node.kwargs = {**node.kwargs, "device": device}
        if node.target is torch.ops.aten.to.device:
            node.args = (node.args[0], device, *node.args[2:])
            rewired.append(node)
        if node.target is SEND:
            value, channel = node.args
            arrival = receiver[channel]
            arrival.meta["consumer"] = JOIN
            away, there = processors[arrival.meta["processor"]]
            sent = there if away == machine else HOST
            if sent != device:
                with graph.inserting_before(node):
                    value = copied(graph, value, sent, node.meta["processor"])
                    rewired += [value, node]
                arrival.meta["val"] = value.meta["val"]
                node.args = (value, channel)
            if there != sent:
                with graph.inserting_after(arrival):
                    landed = copied(graph, arrival, there, arrival.meta["processor"])
                arrival.replace_all_uses_with(landed, lambda user: user is not landed)
                rewired += [landed, *landed.users]
    views(rewired)
    graph_module.recompile()
    return graph_module

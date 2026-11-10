# design/data-flow.md
import copy
import operator

import torch

from .ir import RECV, SEND


def place(graph_module, placement):
    graph_module = copy.deepcopy(graph_module)
    graph = graph_module.graph
    placed = [n for n in graph.nodes if n.op != "placeholder"]
    for node in placed:
        getitem = node.target is operator.getitem
        node.meta["processor"] = node.args[0].meta["processor"] if getitem else placement(node)
    channels = 0
    for node in placed:
        channels += insert_send_recv(graph, node, channels)
    graph_module.recompile()
    return graph_module, placed[-1].meta["processor"], channels


def insert_send_recv(graph, node, first):
    here = node.meta["processor"]
    away = {}
    for user in node.users:
        if user.meta["processor"] != here:
            away.setdefault(user.meta["processor"], []).append(user)
    if not away:
        return 0
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{node.name} is read on another processor; only tensors cross processors")
    for channel, (processor, users) in enumerate(away.items(), first):
        with graph.inserting_after(node):
            graph.call_function(SEND, (node, channel)).meta["processor"] = here
        with graph.inserting_before(users[0]):
            arrived = graph.call_function(RECV, (channel,))
            arrived.meta.update(processor=processor, val=value)
        for user in users:
            user.replace_input_with(node, arrived)
    return len(away)

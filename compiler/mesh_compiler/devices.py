# design/data-flow.md
import copy

import torch

from .ir import RECV, SEND

TO_COPY = torch.ops.aten._to_copy.default
HOST = torch.device("cpu")


def moved(value, device):
    with value.fake_mode:
        return TO_COPY(value, device=device)


def on_devices(graph_module, processors):
    graph_module = copy.deepcopy(graph_module)
    graph = graph_module.graph
    nodes = [n for n in graph.nodes if n.op != "placeholder"]
    receiver = {n.args[0]: n for n in nodes if n.target is RECV}
    for node in nodes:
        machine, device = processors[node.meta["processor"]]
        if "device" in node.kwargs:
            node.kwargs = {**node.kwargs, "device": device}
        if node.target is torch.ops.aten.to.device:
            node.args = (node.args[0], device, *node.args[2:])
        if node.target is SEND:
            value, channel = node.args
            arrival = receiver[channel]
            away, there = processors[arrival.meta["processor"]]
            if away != machine:
                arrival.meta["val"] = moved(value.meta["val"], HOST)
                if there != HOST:
                    with graph.inserting_after(arrival):
                        copy_ = graph.call_function(TO_COPY, (arrival,), {"device": there})
                    arrival.replace_all_uses_with(copy_, lambda user: user is not copy_)
                    copy_.meta.update(processor=arrival.meta["processor"],
                                      val=moved(arrival.meta["val"], there))
            elif there != device:
                with graph.inserting_before(node):
                    copy_ = graph.call_function(TO_COPY, (value,), {"device": there})
                copy_.meta.update(processor=node.meta["processor"], val=moved(value.meta["val"], there))
                arrival.meta["val"] = copy_.meta["val"]
                node.args = (copy_, channel)
    graph_module.recompile()
    return graph_module

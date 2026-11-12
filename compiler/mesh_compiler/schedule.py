# design/data-flow.md
import operator
import pathlib
import sys

import torch

from torch.fx.node import map_arg

from .ir import RECV, SEND, channel_of

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph


def unit_cost(node):
    return 1.0


def producer(node):
    return producer(node.args[0]) if node.target is operator.getitem else node


def schedule(module, cost=unit_cost):
    nodes = list(module.graph.nodes)
    functions = [n for n in nodes if n.op == "call_function" and producer(n) is n]
    sender = {channel_of(n): n for n in functions if n.target is SEND}

    def preds(node):
        found = [producer(u) for u in node.all_input_nodes if u.op == "call_function"]
        return found + [sender[channel_of(node)]] if node.target is RECV else found

    orders = mesh_graph.list_schedule(functions, preds, cost, lambda n: n.meta["processor"])
    processors = dict.fromkeys(n.meta["processor"] for n in nodes if n.op != "placeholder")
    return {p: assemble(p, orders.get(p, []), nodes) for p in processors}


def assemble(processor, order, nodes):
    graph, copied = torch.fx.Graph(), {}
    for node in nodes:
        if node.op == "placeholder" and any(u.meta["processor"] == processor for u in node.users):
            copied[node] = graph.node_copy(node)
    for node in order:
        copied[node] = graph.node_copy(node, copied.__getitem__)
        parts = [u for u in node.users if u.target is operator.getitem]
        for part in sorted(parts, key=lambda u: u.args[1]):
            copied[part] = graph.node_copy(part, copied.__getitem__)
    output = nodes[-1]
    here = output.meta["processor"] == processor
    results = map_arg(output.args[0], copied.__getitem__) if here else ()
    graph.output(results).meta["processor"] = processor
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def channel_ends(modules):
    ends = {}
    for processor, module in modules.items():
        for node in module.graph.nodes:
            if channel_of(node) is not None:
                ends.setdefault(channel_of(node), {})[node.target] = (processor, node)
    return ends


def carried(modules, machines):
    found = []
    for channel, end in sorted(channel_ends(modules).items()):
        (sender, _), (receiver, arrival) = end[SEND], end[RECV]
        if machines[sender] != machines[receiver]:
            value = arrival.meta["val"]
            found.append((channel, sender, receiver, value.dtype, list(value.shape), list(value.stride())))
    return found


def argument_buffers(modules, output_processor):
    inputs = {processor: [n.target for n in module.graph.nodes if n.op == "placeholder"]
              for processor, module in modules.items()}
    at = {processor: index for index, processor in enumerate(modules)}
    edges = [(at[output_processor], at[processor]) for processor in modules if processor != output_processor]
    edges += [(at[end[SEND][0]], at[end[RECV][0]]) for end in channel_ends(modules).values()]
    component = mesh_graph.components(len(at), edges)
    return inputs, sorted(p for p in modules if component[at[p]] != component[at[output_processor]])

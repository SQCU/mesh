# design/data-flow.md
import operator

import networkx
import torch

from torch.fx.node import map_arg

from .ir import RECV, SEND, channel_of


def unit_cost(node):
    return 1.0


def earliest_slot(busy, ready, width):
    free_from = 0.0
    for index, (start, finish, _) in enumerate(busy):
        begin = max(free_from, ready)
        if begin + width <= start:
            return index, begin
        free_from = finish
    return len(busy), max(free_from, ready)


def heft_placed(functions, processor, cost, preds):
    succs = {v: [] for v in functions}
    for v in functions:
        for u in preds(v):
            succs[u].append(v)

    rank, stack = {}, list(functions)
    while stack:
        v = stack[-1]
        unranked = [s for s in succs[v] if s not in rank]
        if unranked:
            stack.extend(unranked)
            continue
        if cost(v) <= 0:
            raise ValueError(f"cost of {v} must be positive")
        rank[stack.pop()] = cost(v) + max((rank[s] for s in succs[v]), default=0.0)

    timeline, finish = {}, {}
    for v in sorted(functions, key=lambda v: -rank[v]):
        busy = timeline.setdefault(processor(v), [])
        ready = max((finish[u] for u in preds(v)), default=0.0)
        index, start = earliest_slot(busy, ready, cost(v))
        finish[v] = start + cost(v)
        busy.insert(index, (start, finish[v], v))
    return {p: [v for _, _, v in busy] for p, busy in timeline.items()}


def producer(node):
    return producer(node.args[0]) if node.target is operator.getitem else node


def schedule(module, cost=unit_cost):
    nodes = list(module.graph.nodes)
    functions = [n for n in nodes if n.op == "call_function" and producer(n) is n]
    sender = {channel_of(n): n for n in functions if n.target is SEND}

    def preds(node):
        found = [producer(u) for u in node.all_input_nodes if u.op == "call_function"]
        return found + [sender[channel_of(node)]] if node.target is RECV else found

    orders = heft_placed(functions, lambda n: n.meta["processor"], cost, preds)
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
    others = [processor for processor in modules if processor != output_processor]
    graph = networkx.DiGraph((output_processor, processor) for processor in others)
    graph.add_node(output_processor)
    graph.add_edges_from((end[SEND][0], end[RECV][0]) for end in channel_ends(modules).values())
    cycle = next(c for c in networkx.strongly_connected_components(graph) if output_processor in c)
    return inputs, sorted(set(modules) - cycle)

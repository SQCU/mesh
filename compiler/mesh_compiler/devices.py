# design/data-flow.md
import copy

import torch

from .ir import HOST, views

TO_COPY = torch.ops.aten._to_copy.default
ENGINES = ("ane",)


def copied(graph, value, device, processor):
    node = graph.call_function(TO_COPY, (value,), {"device": device})
    with value.meta["val"].fake_mode:
        arrives = TO_COPY(value.meta["val"], device=device)
    node.meta.update(processor=processor, val=arrives, custom={"between": True})
    return node


# compiler/mesh_compiler/ir.py views; compiler/mesh_compiler/placement.py place, transfers; torch 2.15
# torch/fx/graph_module.py GraphModule.__deepcopy__ and torch/fx/graph.py Graph.__deepcopy__, val_map=memo
def on_devices(graph_module, processors):
    original, parts = graph_module.meta["transfers"], graph_module.meta["input_parts"]
    decomposed = graph_module.meta.get("decomposed", {})
    memo = {id(original): original, id(parts): parts, id(decomposed): decomposed}
    # compiler/mesh_compiler/decompose.py _branch_owned: each nested branch GraphModule's own `decomposed`, held by the
    # memo (not deep-copied) and remapped below as the outer one is, its keys the copied branch's nodes
    nested = [m for m in graph_module.modules() if m is not graph_module and isinstance(m, torch.fx.GraphModule)
              and "decomposed" in m.meta]
    memo.update((id(m.meta["decomposed"]), m.meta["decomposed"]) for m in nested)
    graph_module = copy.deepcopy(graph_module, memo)
    for branch in nested:
        memo[id(branch)].meta["decomposed"] = {memo[node]: source for node, source in branch.meta["decomposed"].items()}
    transfers = {channel: (memo[sender], memo[arrival]) for channel, (sender, arrival) in original.items()}
    graph, graph_module.meta["transfers"] = graph_module.graph, transfers
    graph_module.meta["input_parts"] = {memo[node]: tuple((processor, memo[part], geometry)
                                                          for processor, part, geometry in pieces)
                                        for node, pieces in parts.items()}
    # compiler/mesh_compiler/decompose.py _expanded `decomposed`: each generated node the copy's, its original source
    # node itself (not in this graph, not copied)
    if "decomposed" in graph_module.meta:
        graph_module.meta["decomposed"] = {memo[node]: source for node, source in decomposed.items()}
    # compiler/mesh_compiler/placement.py `work_axes`: its values are the copy's (torch.fx Graph.node_copy copies meta
    # shallowly)
    for node in graph.nodes:
        if "work_axes" in node.meta:
            node.meta["work_axes"] = tuple((memo[value], dimension, full)
                                           for value, dimension, full in node.meta["work_axes"])
    rewired = [arrival for _, arrival in transfers.values()]
    for node in [n for n in graph.nodes if n.op != "placeholder"]:
        _, device = processors[node.meta["processor"]]
        if "device" in node.kwargs:
            node.kwargs = {**node.kwargs, "device": device}
        if node.target is torch.ops.aten.to.device:
            node.args = (node.args[0], device, *node.args[2:])
            rewired.append(node)
    for channel, (sender, arrival) in transfers.items():
        value = sender.args[0]
        machine, device = processors[sender.meta["processor"]]
        away, there = processors[arrival.meta["processor"]]
        sent = there if away == machine else HOST
        if sent != device:
            with graph.inserting_before(sender):
                value = copied(graph, value, sent, sender.meta["processor"])
                rewired += [value, sender]
            arrival.meta["val"] = value.meta["val"]
            sender.args = (value, *sender.args[1:])
        if there != sent:
            with graph.inserting_after(arrival):
                landed = copied(graph, arrival, there, arrival.meta["processor"])
            # Callback inversion of control is outside the working specification (phase: compile-time device placement).
            # Runtime callback overhead is inadmissible in performance-critical code.
            arrival.replace_all_uses_with(landed, lambda user: mesh_callbacks_are_outside_spec(user is not landed))
            for user in landed.users:
                if "work_axes" in user.meta:
                    user.meta["work_axes"] = tuple((landed if value is arrival else value, dimension, full)
                                                   for value, dimension, full in user.meta["work_axes"])
            rewired += [landed, *landed.users]
    views(rewired)
    graph_module.recompile()
    return graph_module

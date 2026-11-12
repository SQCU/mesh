# design/unequal-shards-change-request.md items 6, 7 and 14
import pathlib
import sys

from .ir import RECV, SEND, channel_of

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph


# S. Zhang et al., HAP, EuroSys 2024, sections 3.2 and 5.2
def stages(scheduled, segment):
    nodes = {n: processor for processor, module in scheduled.modules.items() for n in module.graph.nodes
             if n.op == "call_function"}
    sender = {channel_of(n): n for n in nodes if n.target is SEND}
    group = {n: n.meta.get("of", n) for n in nodes}
    before = {}
    for node in nodes:
        read = [sender[channel_of(node)]] if node.target is RECV else [u for u in node.all_input_nodes if u in nodes]
        before.setdefault(group[node], set()).update(group[u] for u in read if group[u] != group[node])
    level = mesh_graph.longest(list(before), lambda g: sorted(before[g], key=str),
                               lambda a, b: getattr(b, "target", None) is RECV)
    found = {}
    for node, processor in nodes.items():
        if node.target not in (RECV, SEND) and segment(node) is not None:
            found.setdefault(segment(node), {}).setdefault(level[group[node]], {}).setdefault(processor, []).append(node)
    return {part: [held for _, held in sorted(levels.items())] for part, levels in found.items()}


# S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3); A. Lastovetsky et al., arXiv:1109.3074, steps 1 and 2
def shares(staged, processors, time, comm, upper):
    solved = {}
    for part, held in staged.items():
        comp = [[(sum(time(n) for n in stage.get(processor, ()) if "of" not in n.meta),
                  sum(time(n) for n in stage.get(processor, ()) if "of" in n.meta) * len(processors))
                 for processor in processors] for stage in held]
        solved[part] = mesh_graph.shares([comm(part, index) for index in range(len(held))], comp, upper)
    return solved

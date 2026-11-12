# design/unequal-shards-change-request.md items 6, 7, 11, 13 and 14
import collections
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
    received, found = {}, {}
    for node, processor in nodes.items():
        if node.target is RECV:
            received.setdefault(level[node], []).append(node)
        elif node.target is not SEND and segment(node) is not None:
            found.setdefault(segment(node), {}).setdefault(level[group[node]], {}).setdefault(processor, []).append(node)
    return {part: [(received.get(at, []), held) for at, held in sorted(levels.items())]
            for part, levels in found.items()}


# design/unequal-shards-change-request.md item 7; A. Lastovetsky et al., arXiv:1109.3074
def kind(node):
    return node.meta.get("of", node.name)


def units(node):
    return node.meta.get("units", (1, 1))


def speed(times, x):
    return x / times[x]


# S. Zhang et al., HAP, EuroSys 2024, section 3.2; A. Lastovetsky et al., arXiv:1109.3074, steps 1 and 2
def computation(staged, processors, times):
    def time(node, processor):
        x, n = units(node)
        return n / speed(times[processor][kind(node)], x)
    return {part: [[(sum(time(n, processor) for n in held.get(processor, ()) if "of" not in n.meta),
                     sum(time(n, processor) for n in held.get(processor, ()) if "of" in n.meta))
                    for processor in processors] for _, held in found] for part, found in staged.items()}


# S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3)
def shares(staged, processors, times, comm, upper):
    return {part: mesh_graph.shares([max(map(comm, received), default=0.0) for received, _ in staged[part]], comp, upper)
            for part, comp in computation(staged, processors, times).items()}


# X. Jia et al., Whale, USENIX ATC 2022, Algorithm 1
def shifted(units, upper, load):
    units = list(units)
    while True:
        use = [unit / sum(units) / most for unit, most in zip(units, upper)]
        peak = max(range(len(units)), key=use.__getitem__)
        if use[peak] <= 1:
            return tuple(units)
        free = [j for j, most in enumerate(upper) if (units[j] + 1) / sum(units) <= most]
        units[min(free, key=lambda j: load(j, units))] += 1
        units[peak] -= 1


# design/unequal-shards-change-request.md item 11
Ragged = collections.namedtuple("Ragged", "values offsets")


# design/unequal-shards-change-request.md item 13
def rows(ragged, speeds):
    cuts, held = mesh_graph.separators(ragged.offsets.tolist(), speeds), []
    while isinstance(ragged, Ragged):
        held.append((ragged.offsets.diff(), tuple(cuts)))
        cuts, ragged = ragged.offsets[cuts].tolist(), ragged.values
    return *held, (ragged, tuple(cuts))

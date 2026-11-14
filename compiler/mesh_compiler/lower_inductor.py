# design/compilation-composition-change-request.md items 6, 6.4 and 7
import dataclasses

from .ir import SEND, channel_of


# compiler/mesh_compiler/schedule.py assemble, module.meta["writers"], Declarations.wires: the logical copies of a
# scheduled body's declared transfers, before any plan. `wires`: channel -> (declared transfer, logical wire value, its
# logical bytes, declared sent view, declared landed view). Each distinct (source, wire value, sent view, read-after-write
# writers) is one write of its members, the SENDs sharing it, in source order. Consumed by
# compiler/mesh_compiler/lower_metal.py construct (the typed program)
@dataclasses.dataclass
class Copies:
    group: dict
    index: dict
    member: dict


def copies(module, wires):
    writes, sends = {}, [node for node in module.graph.nodes if node.target is SEND and channel_of(node) in wires]
    for writer, reader in module.meta["writers"]:
        writes.setdefault(reader, set()).add(writer)
    group = {}
    for node in sends:
        _, value, _, (offset, element, pairs), _ = wires[channel_of(node)]
        group[node] = (node.args[0], value, offset, element, tuple(map(tuple, pairs)),
                       frozenset(writes.get(node, ())))
    index = {key: k for k, key in enumerate(dict.fromkeys(group.values()))}
    member = {}
    for node in sends:
        member.setdefault(index[group[node]], []).append(node)
    return Copies(group, index, member)

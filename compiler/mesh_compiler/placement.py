# design/the-compiler.md; design/data-flow.md
import copy
import functools
import itertools
import math
import operator
import pathlib
import sys

import torch
from torch._guards import detect_fake_mode
from torch.fx.node import map_arg

from .ir import RECV, SEND

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

ATEN = torch.ops.aten
ADD, ADDMM, CAT, MM, SLICE = (ATEN.add.Tensor, ATEN.addmm.default, ATEN.cat.default, ATEN.mm.default,
                              ATEN.slice.Tensor)
EXPAND, RESHAPE, VIEW = ATEN.expand.default, ATEN.reshape.default, ATEN.view.default
SLICE_SCATTER, ZEROS = ATEN.slice_scatter.default, ATEN.new_zeros.default
BMM, SOFTMAX = ATEN.bmm.default, ATEN._softmax.default
SYMMETRIC = {SOFTMAX, ATEN.any.dim, ATEN.any.dims, ATEN.mean.dim}
MASKED = {ADD, BMM, EXPAND, SOFTMAX, VIEW, ATEN.clone.default, ATEN.eq.Scalar, ATEN.logical_not.default, ATEN.any.dim,
          ATEN.full_like.default, ATEN.where.self}
STRIPED = "striped"


def operands(node):
    found = []
    map_arg((node.args, node.kwargs), found.append)
    return found


def shape(node):
    value = node.meta["val"]
    return tuple((value[0] if isinstance(value, (tuple, list)) else value).shape)


def evaluated(node, processor):
    arguments = map_arg((node.args, node.kwargs), lambda n: n.meta["val"])
    with detect_fake_mode(arguments):
        node.meta.update(processor=processor, val=node.target(*arguments[0], **arguments[1]))
    return node


def call(graph, target, arguments, processor):
    return evaluated(graph.call_function(target, arguments), processor)


# design/unequal-shards-change-request.md items 1 and 15
def counts(extent, units, shards):
    if units == STRIPED:
        return STRIPED
    if units is None:
        chunk = -(-extent // shards)
        return tuple(max(0, min(chunk, extent - chunk * j)) for j in range(shards))
    exact = [extent * unit / sum(units) for unit in units]
    sizes = [round(size) for size in exact]
    while sum(sizes) != extent:
        step = 1 if sum(sizes) < extent else -1
        sizes[min(range(shards), key=lambda j: abs(sizes[j] + step - exact[j]))] += step
    return tuple(sizes)


def folded(graph, value, at, repeats, periods, dimension, sizes, processors):
    held, extents, holders, nodes = value
    nodes = [call(graph, RESHAPE, (n, [*shape(n)[:at], repeats, period, *shape(n)[at + 1:]]), holder or processors[0])
             for n, period, holder in zip(nodes, periods, holders)]
    held, dimension = (d if d is None or d < at else d + 1 for d in (held, dimension))
    arrived = reshard(graph, (held, extents, holders, nodes), dimension, sizes, processors)
    return [call(graph, RESHAPE, (n, [*shape(n)[:at], repeats * shape(n)[at + 1], *shape(n)[at + 2:]]), processor)
            for n, processor in zip(arrived, processors)]


# design/unequal-shards-change-request.md item 8
def interleaved(graph, pieces, dimension, processor):
    whole = list(shape(pieces[0]))
    whole[dimension] = sum(shape(piece)[dimension] for piece in pieces)
    value = call(graph, ZEROS, (pieces[0], whole), processor)
    for residue, piece in enumerate(pieces):
        value = call(graph, SLICE_SCATTER, (value, piece, dimension, residue, whole[dimension], len(pieces)), processor)
    return value


# design/unequal-shards-change-request.md items 2, 3, 8, 9 and 12
def reshard(graph, value, dimension, sizes, processors):
    held, extents, holders, nodes = value
    if (held, extents, holders) == (dimension, sizes, processors):
        return nodes
    if len(processors) == 1 or dimension is None:
        dimension, sizes = None, None
    if held is not None and extents != STRIPED and (whole := sum(shape(n)[held] for n in nodes)) > sum(extents):
        return folded(graph, value, held, whole // sum(extents), extents, dimension, sizes, processors)
    if dimension not in (None, held) and sizes != STRIPED and (whole := shape(nodes[0])[dimension]) > sum(sizes):
        return folded(graph, value, dimension, whole // sum(sizes), [sum(sizes)] * len(nodes), dimension, sizes,
                      processors)
    if held == dimension is not None and STRIPED in (extents, sizes):
        return [reshard(graph, (None, None, (processor,), [node]), dimension, sizes, processors)[j]
                for j, (processor, node) in enumerate(zip(processors, reshard(graph, value, None, None, processors)))]
    if sizes == STRIPED:
        given = tuple(len(range(j, shape(nodes[0])[dimension], len(processors))) for j in range(len(processors)))
        displs = tuple(range(len(processors)))
    else:
        given, displs = sizes, tuple(itertools.accumulate(sizes or (), initial=0))
    starts = list(itertools.accumulate(() if extents in (None, STRIPED) else extents, initial=0))
    if held == dimension is not None:
        given = tuple(tuple(max(0, min(starts[i + 1], displs[j + 1]) - max(starts[i], displs[j]))
                            for j in range(len(processors))) for i in range(len(nodes)))
        displs = tuple(tuple(sum(row[j] for row in given[:i]) for j in range(len(processors)))
                       for i in range(len(nodes)))
    arrived = []
    for j, processor in enumerate(processors):
        pieces, sources = [], []
        for i, (holder, node) in enumerate(zip(holders, nodes)):
            if sizes == STRIPED:
                node = call(graph, SLICE, (node, dimension, j, shape(node)[dimension], len(processors)),
                            holder or processor)
            elif dimension is not None:
                if held == dimension:
                    low = sum(given[i][:j])
                    high = low + given[i][j]
                    if not given[i][j] and any(row[j] for row in given):
                        continue
                else:
                    low, high = displs[j], displs[j + 1]
                if (low, high) != (0, shape(node)[dimension]):
                    node = call(graph, SLICE, (node, dimension, low, max(low, high)), holder or processor)
            pieces.append(node)
            sources.append(i)
        if held == dimension is not None:
            blocks = [(held, (i, j), given, displs) for i in sources]
        elif dimension is None:
            joined = tuple(shape(piece)[held] for piece in pieces) if held is not None else ()
            placed = (tuple(range(len(pieces))) if extents == STRIPED else
                      tuple(itertools.accumulate(joined, initial=0))[:-1])
            blocks = [(held, k, joined, placed) for k in range(len(joined))]
        else:
            blocks = [(dimension, j, given, displs[:len(given)])] * len(pieces)
        for piece, block in zip(pieces, blocks):
            piece.meta.setdefault("blocks", {})[processor] = block
        arrived.append(pieces[0] if len(pieces) == 1 else interleaved(graph, pieces, held, processor)
                       if extents == STRIPED else call(graph, CAT, (pieces, held), processor))
    return arrived


def aligned(node, shapes, out):
    return [[len(out) - len(s) + k if extent == out[len(out) - len(s) + k] else None
             for k, extent in enumerate(s)] for s in shapes]


def over(position, default, summed=False):
    def rule(node, shapes, out):
        rank = len(shapes[0])
        dims = node.args[position] if len(node.args) > position else default
        dims = {d % rank for d in ([dims] if isinstance(dims, int) else dims or range(rank))}
        drops = len(out) < rank
        row = [(-1 - d if summed else None) if d in dims else d - drops * sum(e < d for e in dims)
               for d in range(rank)]
        return [row] * len(shapes)
    return rule


def gathered(node, shapes, out):
    return [over(1, None)(node, shapes, out)[0], list(range(len(out)))]


def permuted(node, shapes, out):
    return [[node.args[1].index(d) for d in range(len(out))]]


def unsqueezed(node, shapes, out):
    return [[d + (d >= node.args[1] % len(out)) for d in range(len(out) - 1)]]


RULES = {
    ATEN.alias.default: aligned, EXPAND: aligned, ATEN.full_like.default: aligned, ATEN._to_copy.default: aligned,
    operator.getitem: aligned,
    ATEN.permute.default: permuted, ATEN.unsqueeze.default: unsqueezed,
    MM: lambda node, shapes, out: [[0, -1], [-1, 1]],
    ATEN.bmm.default: lambda node, shapes, out: [[0, 1, -1], [0, -1, 2]],
    ATEN.sum.dim_IntList: over(1, None, True), ATEN.mean.dim: over(1, None),
    ATEN.any.dim: over(1, None), ATEN.any.dims: over(1, None),
    ATEN._softmax.default: over(1, None), ATEN.cumsum.default: over(1, None), ATEN.topk.default: over(2, -1),
    ATEN.slice.Tensor: over(1, 0), ATEN.select.int: over(1, None), ATEN.squeeze.dims: over(1, None),
    CAT: over(1, 0), ATEN.scatter_add.default: over(1, None), ATEN.gather.default: gathered,
}


def viewed(value, whole, new):
    dimension, sizes, processors, nodes = value
    stride = math.prod(whole[dimension + 1:])
    if sizes == STRIPED:
        at = [math.prod(new[d + 1:]) for d in range(len(new))].index(stride)
        few, many = sorted((whole[dimension], new[at]))
        return (at, value) if few == many or (many % few, few % len(processors)) == (0, 0) else None
    edges = [stride * edge for edge in itertools.accumulate(sizes)]
    for at in reversed(range(len(new))):
        unit = math.prod(new[at + 1:])
        if all(edge % unit == 0 for edge in edges) and unit * new[at] % edges[-1] == 0:
            return at, (dimension, tuple(stride * size // unit for size in sizes), processors, nodes)


def ranges(sizes, extent, shards):
    if sizes == STRIPED:
        return [(j, extent, shards) for j in range(shards)]
    edges = list(itertools.accumulate(sizes, initial=0))
    return [(low, high, 1) for low, high in zip(edges, edges[1:])]


def intact(whole, new):
    units = [math.prod(new[d + 1:]) for d in range(len(new))]
    return [next((at for at in range(len(new)) if (units[at], new[at]) == (math.prod(whole[d + 1:]), whole[d])), None)
            for d in range(len(whole))]


# design/unequal-shards-change-request.md item 8; Brandon et al., Striped Attention, arXiv:2311.09431, section 3.1
def unmasked(node, rows, keys, queries, held):
    at, views = [rows, keys], mesh_graph.chain(node, lambda n: next(
        (user for user in n.users if len(n.users) == 1 and user.target is VIEW), None))
    for before, after in zip(views, views[1:]):
        at = [intact(shape(before), shape(after))[d] for d in at]
    node, user = views[-1], next(iter(views[-1].users))
    every = [list(range(len(held))) for _ in queries]
    constants = [n for n in user.all_input_nodes if n is not node and not any(
        m.op == "placeholder" for m in mesh_graph.reach([n], lambda m: m.all_input_nodes))]
    reached = mesh_graph.reach([user], lambda n: () if n.target is BMM else list(n.users))
    if len(node.users) != 1 or user.target is not ADD or None in at or not constants or not all(
            n.target in MASKED for n in reached):
        return every
    order = mesh_graph.postorder(constants[:1], lambda n: n.all_input_nodes)
    mask = mesh_graph.forward(order, lambda n: n.all_input_nodes, lambda n, read: n.target(
        *map_arg(n.args, lambda m: read[n.all_input_nodes.index(m)]), **n.kwargs))[constants[0]]
    mask = mask.expand(shape(user)).movedim(at, (-2, -1))
    return [[s for s, key in enumerate(held)
             if not (mask[..., list(range(*query)), :][..., list(range(*key))] == -math.inf).all()] for query in queries]


def chosen(graph, node, dimension, blocks, processor):
    pieces = [call(graph, SLICE, (node, dimension, *block), node.meta.get("processor", processor)) for block in blocks]
    return pieces[0] if len(pieces) == 1 else call(graph, CAT, (pieces, dimension), pieces[0].meta["processor"])


def plain(graph, node, processor):
    for dimension, blocks in node.meta.get("order", {}).items():
        whole, edges = list(shape(node)), list(itertools.accumulate((len(range(*b)) for b in blocks), initial=0))
        whole[dimension] = max(stop for _, stop, _ in blocks)
        value = call(graph, ZEROS, (node, whole), node.meta.get("processor", processor))
        for block, low, high in zip(blocks, edges, edges[1:]):
            piece = call(graph, SLICE, (node, dimension, low, high), value.meta["processor"])
            value = call(graph, SLICE_SCATTER, (value, piece, dimension, *block), value.meta["processor"])
        node = value
    return node


def plainly(graph, value):
    return *value[:3], [plain(graph, node, holder) for node, holder in zip(value[3], value[2])]


def partitioned(graph, node, held, want, replaced):
    target = None
    if node.op == "call_function" and (any(v[0] is not None for v in held) or not isinstance(want, str)):
        if node.target is ADDMM:
            product = graph.call_function(MM, node.args[1:])
            product.meta = dict(node.meta)
            first = want if isinstance(want, str) else want[2][0]
            held = [partitioned(graph, product, held[1:], first, replaced), held[0]]
            node.target, node.args = ADD, (product, node.args[0])
        shapes, out = [shape(n) for n in operands(node)], shape(node)
        if node.target is VIEW:
            rows = [intact(shapes[0], out)]
            if held[0][0] is not None:
                rows[0][held[0][0]], value = viewed(held[0], shapes[0], out)
                held = [value]
        elif torch.Tag.pointwise in getattr(node.target, "tags", ()):
            rows = aligned(node, shapes, out)
        else:
            rows = RULES[node.target](node, shapes, out)
        found = {(row[v[0]], v[1], v[2]) for v, row in zip(held, rows) if v[0] is not None}
        summed = len(found) == 1 and (next(iter(found))[0] or 0) < 0
        if not summed and not isinstance(want, str) and any(want[0] in row for row in rows):
            target = want
        elif len(found) == 1 and next(iter(found))[0] is not None:
            target = found.pop()
    if target is None:
        processor = want if isinstance(want, str) else want[2][0]
        arguments = iter([reshard(graph, plainly(graph, v), None, None, (processor,))[0] for v in held])
        node.args, node.kwargs = map_arg((node.args, node.kwargs), lambda n: next(arguments))
        node.meta["processor"] = processor
        return None, None, (processor,), [node]
    along, sizes, processors = target
    shards, whole = [], along >= 0 and (sizes == STRIPED or sum(sizes) == out[along])
    queries = ranges(sizes, out[along], len(processors)) if whole else ()
    joined = [v[0] is not None and along not in row and row[v[0]] is not None and
              (v[1] == STRIPED or sum(v[1]) == shapes[i][v[0]]) for i, (v, row) in enumerate(zip(held, rows))]
    columns = [None if join else reshard(graph, plainly(graph, v) if (v[0], v[1], v[2]) != (
        row.index(along) if along in row else None, sizes, processors) else v,
        row.index(along) if along in row else None, sizes, processors) for v, row, join in zip(held, rows, joined)]
    for j, processor in enumerate(processors):
        local, order = [column and column[j] for column in columns], {}
        for i, row in enumerate(rows):
            for dimension, blocks in (local[i].meta.get("order", {}) if local[i] else {}).items():
                label = row[dimension] if row[dimension] is not None or node.target not in SYMMETRIC else (
                    dimension if node.target is SOFTMAX else -1 - len(out) - dimension)
                if label is None or order.setdefault(label, blocks) != blocks:
                    local[i] = plain(graph, local[i], processor)
        for i, (v, row) in enumerate(zip(held, rows)):
            if joined[i]:
                label, kept = row[v[0]], ranges(v[1], shapes[i][v[0]], len(v[2]))
                if label not in order:
                    picked = unmasked(node, along, label, queries, kept)[j] if whole and label >= 0 else range(len(kept))
                    order[label] = tuple(kept[s] for s in picked)
                pieces = [v[3][kept.index(block)] for block in order[label]]
                extents = tuple(shape(piece)[v[0]] for piece in pieces)
                for k, piece in enumerate(pieces):
                    piece.meta.setdefault("blocks", {})[processor] = (
                        v[0], k, extents, tuple(itertools.accumulate(extents, initial=0))[:-1])
                local[i] = pieces[0] if len(pieces) == 1 else call(graph, CAT, (pieces, v[0]), processor)
        for i, row in enumerate(rows):
            for dimension, label in enumerate(row):
                if label in order and not joined[i] and dimension not in local[i].meta.get("order", {}):
                    local[i] = chosen(graph, local[i], dimension, order[label], processor)
        pairs = sum(len(order[row[v[0]]]) for v, row, join in zip(held, rows, joined) if join)
        order = {label: blocks for label, blocks in order.items() if label >= 0 and
                 [index for block in blocks for index in range(*block)] != list(range(out[label]))}
        arguments = iter(local)
        shard = graph.node_copy(node, lambda n: next(arguments))
        if node.target in (EXPAND, VIEW):
            sized = list(shard.args[1])
            sized[along] = (len(range(*queries[j])) if sizes == STRIPED else out[along] * sizes[j] // sum(sizes))
            for label, blocks in order.items():
                sized[label] = sum(len(range(*block)) for block in blocks)
            shard.args = (shard.args[0], sized)
        shard.meta.update(of=node.name, order=order, pairs=pairs)
        shards.append(evaluated(shard, processor))
    replaced.append(node)
    if along >= 0:
        return along, sizes, processors, shards
    to = (None, None, (want,)) if isinstance(want, str) else want
    columns = [reshard(graph, (None, None, (processor,), [shard]), *to)
               for processor, shard in zip(processors, shards)]
    return *to, [functools.reduce(lambda a, b: call(graph, ADD, (a, b), processor), parts)
               for processor, *parts in zip(to[2], *columns)]


def place(graph_module, placement):
    graph_module = copy.deepcopy(graph_module)
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placed = [n for n in nodes if n.op != "placeholder"]
    stated = {n: placement(n) for n in placed if n.target is not operator.getitem}
    stated.update((n, placement(n)) for n in nodes if n.op == "placeholder")
    replaced = []

    def transfer(node, held):
        want = stated[node] if node in stated else held[0][2][0]
        if not isinstance(want, str):
            dimension, processors, units = want
            want = dimension, counts(shape(node)[dimension], units, len(processors)), processors
        with graph.inserting_before(placed[0] if node.op == "placeholder" else node):
            value = ((None, None, (None,), [node]) if node.op == "placeholder" else
                     partitioned(graph, node, held, want, replaced))
            if not isinstance(want, str):
                value = *want, reshard(graph, value, *want)
        return value

    mesh_graph.forward(nodes, operands, transfer)
    for node in reversed(replaced):
        graph.erase_node(node)
    for node in reversed([n for n in graph.nodes if n.op == "call_function"]):
        away = {user.meta["processor"] for user in node.users} - {node.meta["processor"]}
        if away and not any(n.op == "placeholder" for n in mesh_graph.reach([node], lambda n: n.all_input_nodes)):
            for processor in sorted(away):
                with graph.inserting_after(node):
                    there = graph.node_copy(node)
                there.meta.update(processor=processor, constant=node.meta.get("constant", node.name))
                for user in [user for user in node.users if user.meta["processor"] == processor and user is not there]:
                    user.replace_input_with(node, there)
    placed = [n for n in graph.nodes if n.op != "placeholder"]
    channels = 0
    for node in placed:
        channels += insert_send_recv(graph, node, channels)
    for node in graph.nodes:
        node.meta.pop("blocks", None)
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
            sent = graph.call_function(SEND, (node, channel))
            sent.meta["processor"] = here
        with graph.inserting_before(users[0]):
            arrived = graph.call_function(RECV, (channel,))
            arrived.meta.update(processor=processor, val=value)
        if processor in node.meta.get("blocks", ()):
            for end in (sent, arrived):
                end.meta.update(zip(("dimension", "block", "counts", "displs"), node.meta["blocks"][processor]))
        for user in users:
            user.replace_input_with(node, arrived)
    return len(away)

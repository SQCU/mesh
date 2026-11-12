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
BMM, FLIP, SOFTMAX = ATEN.bmm.default, ATEN.flip.default, ATEN._softmax.default
CONVOLUTION, FULL = ATEN.convolution.default, ATEN.new_full.default
MAXIMUM, AVERAGE = ATEN.max_pool2d_with_indices.default, ATEN.avg_pool2d.default
SYMMETRIC = {SOFTMAX, ATEN.any.dim, ATEN.any.dims, ATEN.mean.dim}
MASKED = {ADD, BMM, EXPAND, SOFTMAX, VIEW, ATEN.clone.default, ATEN.eq.Scalar, ATEN.logical_not.default, ATEN.any.dim,
          ATEN.full_like.default, ATEN.where.self}
STRIPED = "striped"
VIEWS = {EXPAND, SLICE, VIEW, ATEN.alias.default, ATEN.permute.default, ATEN.select.int, ATEN.squeeze.dims,
         ATEN.unsqueeze.default}


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


# design/unequal-shards-change-request.md items 1, 13 and 15
def counts(extent, units, shards):
    if units == STRIPED:
        return STRIPED
    if units is None:
        chunk = -(-extent // shards)
        return tuple(max(0, min(chunk, extent - chunk * j)) for j in range(shards))
    if len(units) == shards + 1:
        return tuple(high - low for low, high in itertools.pairwise(units))
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


# design/unequal-shards-change-request.md item 8
def embedded(node, shapes, out):
    return [[None, len(out) - 1], list(range(len(out) - 1))]


def given(node):
    names = [argument.name for argument in node.target._schema.arguments]
    return names, {**dict(zip(names, node.args)), **node.kwargs}


def spread(value, count):
    return list(value) * (count // len(value)) if isinstance(value, (list, tuple)) else [value] * count


# Y. Xu et al., GSPMD, arXiv:2105.04663, section 4.3 and appendix A.2
def windows(node):
    names, read = given(node)
    count, dtype = len(shape(node)) - 2, node.all_input_nodes[0].meta["val"].dtype
    low = spread(read.get("padding", 0), count)
    if node.target is CONVOLUTION:
        kernel, fill = shape(node.args[1])[2:], 0
        if read["transposed"] or len(kernel) != count:
            return {}
    else:
        kernel, fill = spread(read["kernel_size"], count), 0
        if node.target is MAXIMUM:
            fill = -math.inf if dtype.is_floating_point else torch.iinfo(dtype).min
            if any(user.args[1] for user in node.users):
                return {}
        elif read.get("divisor_override") is not None or not read.get("count_include_pad", True) and any(low):
            return {}
        if read.get("ceil_mode", False):
            return {}
    dilation = spread(read.get("dilation", 1), count)
    stride = spread(read.get("stride") or kernel, count)
    return {2 + k: (dilation[k] * (kernel[k] - 1) + 1, stride[k], low[k], fill) for k in range(count)}


def windowed(node, shapes, out):
    found = windows(node)
    row = [0, None if node.target is CONVOLUTION else 1, *(d if d in found else None for d in range(2, len(out)))]
    return [row[:len(shapes[0])], *([None] * len(s) for s in shapes[1:])]


def unpadded(node, at):
    names, read = given(node)
    read["padding"] = [0 if k == at else low for k, low in enumerate(spread(read.get("padding", 0), len(shape(node)) - 2))]
    node.args = tuple(read.pop(name) for name in names[:len(node.args)])
    node.kwargs = read


# Y. Xu et al., GSPMD, arXiv:2105.04663, section 4.3 and appendix A.2; design/unequal-shards-change-request.md item 10;
# design/problems-of-14b9a757-change-request.md item 4
def halos(graph, value, window, dimension, sizes, processors):
    (span, stride, low, fill), (held, extents, holders, nodes) = window, value
    size = shape(nodes[0])[dimension] if held is None else sum(extents)
    firsts = list(itertools.accumulate(sizes, initial=0))
    reads = [(first * stride - low, (first + count - 1) * stride - low + span) for first, count in zip(firsts, sizes)]
    starts = [0, size] if held is None else list(itertools.accumulate(extents, initial=0))
    cuts = [[(max(low_, 0, starts[p]), min(high, size, starts[p + 1])) for p in range(len(starts) - 1)]
            for low_, high in reads]
    shared = [[max(0, last - first) for first, last in row] for row in cuts]
    cover = [[bool(shared[q][p]) * (starts[p + 1] - first if p < q else last - starts[p])
              for p, (first, last) in enumerate(row)] for q, row in enumerate(cuts)]
    regions = []
    for q, (processor, (low_, high)) in enumerate(zip(processors, reads)):
        if not sizes[q]:
            regions.append(nodes[q if held is not None else 0])
            continue
        parts = []
        for p, node in enumerate(nodes):
            first, extent = max(low_, 0, starts[p]) - starts[p], starts[p + 1] - starts[p]
            if not shared[q][p]:
                continue
            if held is None:
                node = call(graph, SLICE, (node, dimension, first, first + shared[q][p]), holders[0] or processor)
            elif p == q and shared[q][p] < extent:
                node = call(graph, SLICE, (node, dimension, first, first + shared[q][p]), processor)
            elif p != q:
                width = min(extent, max(cover[r][r + p - q] for r in range(len(nodes)) if 0 <= r + p - q < len(nodes)))
                if width < extent:
                    node = call(graph, SLICE, (node, dimension, *((extent - width, extent) if p < q else (0, width))),
                                processors[p])
                if shared[q][p] < width:
                    skip = width - cover[q][p] if p < q else first
                    node = call(graph, SLICE, (node, dimension, skip, skip + shared[q][p]), processor)
            parts.append(node)
        own = nodes[q if held is not None else 0]
        for before, missing in ((True, min(high, 0) - low_), (False, high - max(size, low_))):
            if missing > 0:
                padded = [missing if d == dimension else extent for d, extent in enumerate(shape(own))]
                parts.insert(0 if before else len(parts), call(graph, *((ZEROS, (own, padded)) if fill == 0 else (
                    FULL, (own, padded, fill))), processor))
        regions.append(parts[0] if len(parts) == 1 else call(graph, CAT, (parts, dimension), processor))
    return regions


# design/aot-compilation-change-request.md A2 and A4
def replicated(node, shapes, out):
    return [[None] * len(s) for s in shapes]


RULES = {
    ATEN.alias.default: aligned, EXPAND: aligned, ATEN.full_like.default: aligned, ATEN._to_copy.default: aligned,
    operator.getitem: aligned, FLIP: aligned,
    CONVOLUTION: windowed, MAXIMUM: windowed, AVERAGE: windowed,
    ATEN.permute.default: permuted, ATEN.unsqueeze.default: unsqueezed,
    MM: lambda node, shapes, out: [[0, -1], [-1, 1]],
    ATEN.bmm.default: lambda node, shapes, out: [[0, 1, -1], [0, -1, 2]],
    ATEN.sum.dim_IntList: over(1, None, True), ATEN.mean.dim: over(1, None),
    ATEN.any.dim: over(1, None), ATEN.any.dims: over(1, None),
    ATEN._softmax.default: over(1, None), ATEN.cumsum.default: over(1, None), ATEN.topk.default: over(2, -1),
    ATEN.slice.Tensor: over(1, 0), ATEN.select.int: over(1, None), ATEN.squeeze.dims: over(1, None),
    CAT: over(1, 0), ATEN.scatter_add.default: over(1, None), ATEN.gather.default: gathered,
    ATEN.embedding.default: embedded, ATEN._log_softmax.default: over(1, None), ATEN.scatter.value: over(1, None),
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


def plain(graph, node, processor, extents=None):
    for dimension, blocks in node.meta.get("order", {}).items():
        whole, edges = list(shape(node)), list(itertools.accumulate((len(range(*b)) for b in blocks), initial=0))
        whole[dimension] = extents[dimension] if extents else max(stop for _, stop, _ in blocks)
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
                rows[0][held[0][0]], value = viewed(held[0], shapes[0], out) or (None, held[0])
                held = [value]
        elif torch.Tag.pointwise in getattr(node.target, "tags", ()):
            rows = aligned(node, shapes, out)
        else:
            rows = RULES.get(node.target, replicated)(node, shapes, out)
        if RULES.get(node.target) is windowed and STRIPED in (
                held[0][1], None if isinstance(want, str) else want[1]):
            rows = replicated(node, shapes, out)
        found = {(row[v[0]], v[1], v[2]) for v, row in zip(held, rows) if v[0] is not None}
        summed = len(found) == 1 and (next(iter(found))[0] or 0) < 0
        if not summed and not isinstance(want, str) and any(want[0] in row for row in rows):
            target = want
        elif len(found) == 1 and next(iter(found))[0] is not None:
            target = found.pop()
    window = target and RULES.get(node.target) is windowed and windows(node).get(target[0])
    if window:
        target = target[0], counts(out[target[0]], target[1], len(target[2])), target[2]
        target = None if node.target is MAXIMUM and 0 in target[1] else target
    if target is None:
        processor = want if isinstance(want, str) else want[2][0]
        arguments = iter([reshard(graph, plainly(graph, v), None, None, (processor,))[0] for v in held])
        node.args, node.kwargs = map_arg((node.args, node.kwargs), lambda n: next(arguments))
        node.meta["processor"] = processor
        return None, None, (processor,), [node]
    along, sizes, processors = target
    if window:
        value = held[0]
        if (value[0], value[2]) not in ((None, (None,)), (along, processors)):
            inner = counts(shapes[0][along], sizes, len(processors))
            value = along, inner, processors, reshard(graph, plainly(graph, value), along, inner, processors)
        regions = halos(graph, value, window, along, sizes, processors)
    shards, whole = [], along >= 0 and (sizes == STRIPED or sum(sizes) == out[along])
    queries = ranges(sizes, out[along], len(processors)) if whole else ()
    joined = [v[0] is not None and along not in row and row[v[0]] is not None and
              (v[1] == STRIPED or sum(v[1]) == shapes[i][v[0]]) for i, (v, row) in enumerate(zip(held, rows))]
    columns = [None if join else reshard(graph, plainly(graph, v) if (v[0], v[1], v[2]) != (
        row.index(along) if along in row else None, sizes, processors) else v,
        row.index(along) if along in row else None, sizes, processors)
        for v, row, join in zip(held[bool(window):], rows[bool(window):], joined[bool(window):])]
    columns[:0] = [regions] if window else []
    # Y. Xu et al., GSPMD, arXiv:2105.04663, sections 4.3 and 4.5; design/aot-compilation-change-request.md A4
    if node.target is FLIP and along in [d % len(out) for d in node.args[1]]:
        columns[0] = ([columns[0][(out[along] - 1 - j) % len(processors)] for j in range(len(processors))]
                      if sizes == STRIPED else reshard(graph, (along, sizes, processors, columns[0]), along,
                                                       sizes[::-1], processors[::-1])[::-1])
    extent = out[along] if along >= 0 else next(s[row.index(along)] for s, row in zip(shapes, rows) if along in row)
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
        if window and not sizes[j]:
            shard = graph.call_function(ZEROS, (local[0], [0 if d == along else e for d, e in enumerate(out)]))
        else:
            shard = graph.node_copy(node, lambda n: next(arguments))
        if window and sizes[j]:
            unpadded(shard, along - 2)
        if node.target in (EXPAND, VIEW):
            sized = list(shard.args[1])
            sized[along] = (len(range(*queries[j])) if sizes == STRIPED else out[along] * sizes[j] // sum(sizes))
            for label, blocks in order.items():
                sized[label] = sum(len(range(*block)) for block in blocks)
            shard.args = (shard.args[0], sized)
        shard.meta.update(of=node.name, order=order, pairs=pairs, units=(
            len(range(j, extent, len(processors))) if sizes == STRIPED else sizes[j],
            extent if sizes == STRIPED else sum(sizes)))
        shards.append(evaluated(shard, processor))
    replaced.append(node)
    if along >= 0:
        return along, sizes, processors, shards
    to = (None, None, (want,)) if isinstance(want, str) else want
    kept = ranges(to[1], out[to[0]], len(to[2])) if to[0] is not None else []
    parts = {processor: [] for processor in to[2]}
    for processor, shard in zip(processors, shards):
        blocks = shard.meta["order"].get(to[0], ())
        if shard.meta["order"].keys() == {to[0]} and set(blocks) <= set(kept):
            edges = list(itertools.accumulate((len(range(*block)) for block in blocks), initial=0))
            for block, low, high in zip(blocks, edges, edges[1:]):
                parts[to[2][kept.index(block)]].append(
                    shard if len(blocks) == 1 else call(graph, SLICE, (shard, to[0], low, high), processor))
        else:
            column = reshard(graph, (None, None, (processor,), [plain(graph, shard, processor, out)]), *to)
            for keeper, part in zip(to[2], column):
                parts[keeper].append(part)
    return *to, [functools.reduce(lambda a, b: call(graph, ADD, (a, b), keeper), held_)
               for keeper, held_ in parts.items()]


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
        if not isinstance(want, str) and value[:3] != want:
            with graph.inserting_before(placed[0] if node.op == "placeholder" else node.next):
                value = *want, reshard(graph, plainly(graph, value), *want)
        return value

    mesh_graph.forward(nodes, operands, transfer)
    for node in reversed(replaced):
        graph.erase_node(node)
    # design/problems-of-14b9a757-change-request.md item 8, steps 1 to 3
    invariant = mesh_graph.forward(list(graph.nodes), lambda n: n.all_input_nodes, lambda n, read: (
        n.op == "placeholder" or n.target in VIEWS and all(read)))
    onto = mesh_graph.replicas(
        [n for n in graph.nodes if n.op == "call_function"], lambda n: list(n.users),
        lambda n: invariant[n] or not any(
            m.op == "placeholder" for m in mesh_graph.reach([n], lambda m: m.all_input_nodes)),
        lambda n: n.meta["processor"])
    copies = {}
    for node in [n for n in graph.nodes if n in onto]:
        copies[node, node.meta["processor"]] = node
        for processor in onto[node]:
            with graph.inserting_after(node):
                there = graph.node_copy(node, lambda n: copies.get((copies.get(n, n), processor), n))
            there.meta.update(processor=processor, constant=there.meta.pop("of", node.name))
            node.replace_all_uses_with(there, lambda user: user is not there and user.meta["processor"] == processor)
            copies[node, processor], copies[there] = there, node
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

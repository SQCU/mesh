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

from .ir import (CLONE, JOIN, PAGES, RECV, SEND, aliased, aliasing, constant, evaluated, given, leaves_of, produced,
                 results_of, shape, view, views)

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

ATEN = torch.ops.aten
ADD, ADDMM, CAT, MM, SLICE = (ATEN.add.Tensor, ATEN.addmm.default, ATEN.cat.default, ATEN.mm.default,
                              ATEN.slice.Tensor)
EXPAND, RESHAPE, VIEW = ATEN.expand.default, ATEN.reshape.default, ATEN.view.default
SLICE_SCATTER, ZEROS = ATEN.slice_scatter.default, ATEN.new_zeros.default
BMM, FLIP, SOFTMAX = ATEN.bmm.default, ATEN.flip.default, ATEN._softmax.default
ARANGE, LE, AND, NOT = ATEN.arange.start_step, ATEN.le.Scalar, ATEN.logical_and.default, ATEN.logical_not.default
PAD, EMPTY_STRIDED, AS_STRIDED = ATEN.constant_pad_nd.default, ATEN.empty_strided.default, ATEN.as_strided.default
CONVOLUTION = ATEN.convolution.default
MAXIMA = {ATEN.max_pool2d_with_indices.default, ATEN.max_pool3d_with_indices.default}
AVERAGES = {ATEN.avg_pool1d.default, ATEN.avg_pool2d.default, ATEN.avg_pool3d.default}
SYMMETRIC = {SOFTMAX, ATEN.any.dim, ATEN.any.dims, ATEN.mean.dim}
MASKED = {ADD, BMM, EXPAND, SOFTMAX, VIEW, ATEN.clone.default, ATEN.eq.Scalar, ATEN.logical_not.default, ATEN.any.dim,
          ATEN.full_like.default, ATEN.where.self}
STRIPED = "striped"


def operands(node):
    found = []
    map_arg((node.args, node.kwargs), found.append)
    return found


# each operand's shape as `node` reads it: a getitem's its producer's selected result's (ir.shape at its index), any
# other's its operand's own
def read_shapes(node):
    if node.target is operator.getitem:
        return [shape(node.args[0], node.args[1])]
    return [shape(n) for n in operands(node)]


def call(graph, target, arguments, processor, **kwargs):
    return evaluated(graph.call_function(target, arguments, kwargs), processor)


# design/unequal-shards-change-request.md items 1, 13 and 15
class Sizes(tuple):
    units = None


def sized(sizes, units):
    sizes = Sizes(sizes)
    sizes.units = units
    return sizes


class Stripes(str):
    extent = shards = None


# design/unequal-shards-change-request.md item 1; S. Zhang et al., HAP, EuroSys 2024, section 5.1: fractional `units`
# rounded to the nearest sizes, then the least rounding-error +/-1 adjustment until they sum to `extent`; with integer
# `lower` and `upper` (one each shard; item 14's capacity), the nearest sizes first within them and only adjustments that
# stay within them, ValueError where no sizes within them sum to `extent`
def counts(extent, units, shards, lower=None, upper=None):
    if units == STRIPED:
        sizes = Stripes(STRIPED)
        sizes.extent, sizes.shards = extent, shards
        return sizes
    if units is None:
        chunk = -(-extent // shards)
        return Sizes(max(0, min(chunk, extent - chunk * j)) for j in range(shards))
    if len(units) == shards + 1:
        return Sizes(high - low for low, high in itertools.pairwise(units))
    low = [-math.inf] * shards if lower is None else list(lower)
    high = [math.inf] * shards if upper is None else list(upper)
    if len(low) != shards or len(high) != shards:
        raise ValueError(f"{shards} shards with bounds {lower} and {upper}")
    if any(a > b for a, b in zip(low, high)) or sum(low) > extent or sum(high) < extent:
        raise ValueError(f"no sizes within {lower} and {upper} sum to {extent}")
    exact = [extent * unit / sum(units) for unit in units]
    sizes = [min(max(round(size), a), b) for size, a, b in zip(exact, low, high)]
    while sum(sizes) != extent:
        step = 1 if sum(sizes) < extent else -1
        within = [j for j in range(shards) if low[j] <= sizes[j] + step <= high[j]]
        sizes[min(within, key=lambda j: abs(sizes[j] + step - exact[j]))] += step
    return sized(sizes, tuple(units))


# design/unequal-shards-change-request.md items 4 and 12; Y. Xu et al., GSPMD, arXiv:2105.04663, section 4.1
def uneven(sizes):
    if sizes == STRIPED:
        return bool(sizes.extent % sizes.shards)
    return sizes is not None and min(sizes) < max(sizes)


def most(sizes):
    return -(-sizes.extent // sizes.shards) if sizes == STRIPED else max(sizes)


def held_by(sizes, j):
    if sizes == STRIPED:
        return 0, len(range(j, sizes.extent, sizes.shards))
    return sum(sizes[:j]) - bases(sizes)[j], sizes[j]


def bases(sizes):
    return [min(at, sum(sizes) - max(sizes)) for at in itertools.accumulate(sizes[:-1], initial=0)]


def own(graph, node, dimension, sizes, j):
    if not uneven(sizes) or held_by(sizes, j)[1] == most(sizes):
        return node
    start, count = held_by(sizes, j)
    return call(graph, SLICE, (node, dimension, start, start + count), node.meta["processor"])


def padded(graph, node, dimension, sizes, j, processor, mirrored=False):
    if not uneven(sizes) or held_by(sizes, j)[1] == most(sizes):
        return node
    start, count = held_by(sizes, j)
    made = call(graph, PAD, (node, [0, 0] * (len(shape(node)) - 1 - dimension) + [
        start, most(sizes) - start - count][::-1 if mirrored else 1], 0), processor)
    if "landing" in node.meta:
        made.meta["landing"] = node.meta["landing"]
    return made


# Y. Xu et al., GSPMD, arXiv:2105.04663, section 4.1
def selected(graph, node, dimension, low, high, fill, processor, period=None):
    extent = shape(node)[dimension]
    index = constant(graph, ARANGE, (0, extent), node, processor, dtype=torch.int64)
    if period:
        index = call(graph, ATEN.remainder.Scalar, (index, period), processor)
    inside = call(graph, LE, (index, high - 1), processor)
    if low:
        inside = call(graph, AND, (inside, call(graph, NOT, (call(graph, LE, (index, low - 1), processor),),
                                                   processor)), processor)
    inside = call(graph, VIEW, (inside, [extent if d == dimension else 1 for d in range(len(shape(node)))]), processor)
    inside = call(graph, EXPAND, (inside, list(shape(node))), processor)
    value = constant(graph, ATEN.scalar_tensor.default, (fill,), node, processor, dtype=node.meta["val"].dtype)
    return call(graph, ATEN.where.self, (inside, node, value), processor)


def masked(graph, node, dimension, sizes, j, processor):
    if not uneven(sizes) or held_by(sizes, j)[1] == most(sizes):
        return node
    (start, count), extent = held_by(sizes, j), shape(node)[dimension]
    return selected(graph, node, dimension, start, start + count, 0, processor,
                    most(sizes) if extent > most(sizes) else None)


def folded(graph, value, at, repeats, periods, dimension, sizes, processors, exact):
    held, extents, holders, nodes = value
    nodes = [call(graph, RESHAPE, (n, [*shape(n)[:at], repeats, period, *shape(n)[at + 1:]]), holder or processors[0])
             for n, period, holder in zip(nodes, periods, holders)]
    held, dimension = (d if d is None or d < at else d + 1 for d in (held, dimension))
    arrived = reshard(graph, (held, extents, holders, nodes), dimension, sizes, processors, exact)
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


# torch 2.14.0 aten slice.Tensor on the source's own fake value: a placeholder's partition's source view (meta
# "landing"), the same SLICE of the placeholder's value, or of its input's source view, kept across padding
def landed(made, node, arguments):
    source = node.meta["landing"] if "landing" in node.meta else node.meta["val"] if node.op == "placeholder" else None
    if source is not None:
        with detect_fake_mode([source]):
            made.meta["landing"] = SLICE(source, *arguments)
    return made


def cut(graph, node, dimension, sizes, j, shards, processor, wide):
    if sizes == STRIPED:
        arguments = dimension, j, shape(node)[dimension], shards
        node = landed(call(graph, SLICE, (node, *arguments), processor), node, arguments)
        return padded(graph, node, dimension, sizes, j, processor) if wide else node
    low = bases(sizes)[j] if wide else sum(sizes[:j])
    high = low + (max(sizes) if wide else sizes[j])
    arguments = dimension, low, max(low, high)
    return node if (low, high) == (0, shape(node)[dimension]) else landed(
        call(graph, SLICE, (node, *arguments), processor), node, arguments)


def several(value):
    return isinstance(value[0], tuple)


# torch 2.14.0 torch/_inductor/fx_passes/post_grad.py:1131-1157 is_valid_splitwithsizes_cat and :1728-1732
# splitwithsizes_cat_replace (a cat of every split_with_sizes part of one input, in order, on the split's dim, is that
# input), over reshard's own compiler-generated cut parts (a source torch.cat allocates; not that): unit-step aten.slice
# pieces of one placed source node along `dimension` (identical source, same dim), consecutive from 0 to its extent
# there (ordered exact full cover), are that source (as cut returns a whole cut uncut), its view, alias and processor
# kept; other pieces one new aten.cat
def reassembled(graph, pieces, dimension, processor):
    at = 0
    for piece in pieces:
        if (piece.target is not SLICE or piece.args[0] is not pieces[0].args[0] or piece.args[1] != dimension or
                piece.args[2] != at or piece.args[4:] not in ((), (1,))):
            return call(graph, CAT, (pieces, dimension), processor)
        at = piece.args[3]
    source = pieces[0].args[0]
    return source if at == shape(source)[dimension] else call(graph, CAT, (pieces, dimension), processor)


def flat(processors):
    return tuple(leaf for p in processors for leaf in (flat(p) if isinstance(p, tuple) else (p,)))


def spans(grid, depth):
    return [len(functools.reduce(lambda rows, _: rows[0], range(d), grid)) for d in range(depth)]


# design/problems-of-14b9a757-change-request.md item 4 step 3; design/unequal-shards-change-request.md items 1 and 9
def across(graph, value, dimension, sizes, processors, exact):
    held, extents, holders, nodes = value
    if several(value):
        single = dimension is None or not isinstance(dimension, tuple) and len(processors) == 1
        heads, each = tuple(flat(processors if single else (row,))[0] for row in holders), len(nodes) // len(holders)
        rest = (held[1:], extents[1:]) if len(held) > 2 else (held[1], extents[1])
        joined = [reshard(graph, (*rest, row, nodes[r * each:(r + 1) * each]), None, None, (head,))[0]
                  for r, (row, head) in enumerate(zip(holders, heads))]
        return reshard(graph, (held[0], extents[0], heads, joined), dimension, sizes, processors, exact)
    if holders == (None,):
        lengths, made = spans(processors, len(dimension)), []
        for leaf, at in zip(flat(processors), itertools.product(*map(range, lengths))):
            node = nodes[0]
            for d, size, j, shards in zip(dimension, sizes, at, lengths):
                node = cut(graph, node, d, size, j, shards, leaf, uneven(size) and not exact)
            made.append(node)
        return made
    heads = holders * len(processors) if len(holders) == 1 else tuple(flat((row,))[0] for row in processors)
    rest = (dimension[1:], sizes[1:]) if len(dimension) > 2 else (dimension[1], sizes[1])
    return [node for row, head, piece in zip(processors, heads, reshard(
        graph, value, dimension[0], sizes[0], heads, exact))
        for node in reshard(graph, (None, None, (head,), [piece]), *rest, row, exact)]


# design/unequal-shards-change-request.md items 2, 3, 8, 9 and 12
def reshard(graph, value, dimension, sizes, processors, exact=False):
    held, extents, holders, nodes = value
    if several(value) or isinstance(dimension, tuple):
        return nodes if value[:3] == (dimension, sizes, processors) else across(
            graph, value, dimension, sizes, processors, exact)
    if (held, extents, holders) == (dimension, sizes, processors) and not (exact and uneven(sizes)):
        return nodes
    if len(processors) == 1 or dimension is None:
        dimension, sizes = None, None
    if uneven(extents):
        if (repeats := shape(nodes[0])[held] // most(extents)) > 1:
            return folded(graph, value, held, repeats, [most(extents)] * len(nodes), dimension, sizes, processors, exact)
        nodes = [own(graph, n, held, extents, i) for i, n in enumerate(nodes)]
        if (held, extents, holders) == (dimension, sizes, processors):
            return nodes
    elif held is not None and extents != STRIPED and (whole := sum(shape(n)[held] for n in nodes)) > sum(extents):
        return folded(graph, value, held, whole // sum(extents), extents, dimension, sizes, processors, exact)
    if dimension not in (None, held) and sizes != STRIPED and (whole := shape(nodes[0])[dimension]) > sum(sizes):
        return folded(graph, (held, extents, holders, nodes), dimension, whole // sum(sizes),
                      [sum(sizes)] * len(nodes), dimension, sizes, processors, exact)
    wide, kind = not exact and dimension is not None and uneven(sizes), "alltoall"
    if held == dimension is not None:
        extent = sum(shape(n)[held] for n in nodes)
        have, take = ranges(extents, extent, len(nodes)), ranges(sizes, extent, len(processors))
        met = [[mesh_graph.intersection(range(*kept), range(*wanted)) for wanted in take] for kept in have]
        arrays, devices = tuple(zip(arrayed(have), arrayed(take))), processors
        if have == take:
            devices = tuple(dict.fromkeys((*holders, *processors)))
            kind, arrays = "permute", tuple(array + (rest,) * (len(devices) - len(array))
                                            for array, rest in zip(arrayed(have), (0, extent, 1)))
    if held == dimension is not None and STRIPED in (extents, sizes):
        arrived, exchange = [], None
        for j, processor in enumerate(processors):
            pieces = []
            for i, (holder, node) in enumerate(zip(holders, nodes)):
                common, left, right = met[i][j]
                if common:
                    piece = call(graph, SLICE, (node, held, left.start, left.stop, left.step), holder or processor)
                    exchange = exchange or f"{nodes[0].name}:{dimension}:{len(graph.nodes)}"
                    piece.meta.setdefault("collectives", {})[processor] = kind, exchange
                    piece.meta.setdefault("blocks", {})[processor] = held, (i, devices.index(processor)), *arrays
                    pieces.append((right, piece))
            value = pieces[0][1] if pieces else call(graph, ZEROS, (nodes[0], [
                0 if d == held else n for d, n in enumerate(shape(nodes[0]))]), processor)
            if len(pieces) > 1:
                value = call(graph, ZEROS, (value, [len(range(*take[j])) if d == held else n for d, n in enumerate(
                    shape(value))]), processor)
                for right, piece in pieces:
                    value = call(graph, SLICE_SCATTER, (value, piece, held, right.start, right.stop, right.step),
                                 processor)
            arrived.append(padded(graph, value, dimension, sizes, j, processor) if wide else value)
        return arrived
    if sizes == STRIPED:
        given = tuple(len(range(j, shape(nodes[0])[dimension], len(processors))) for j in range(len(processors)))
        displs = tuple(range(len(processors)))
    elif held != dimension or dimension is None:
        given, displs = sizes, tuple(itertools.accumulate(sizes or (), initial=0))
    arrived, exchange = [], None
    for j, processor in enumerate(processors):
        pieces, sources = [], []
        for i, (holder, node) in enumerate(zip(holders, nodes)):
            if held == dimension is not None:
                common, left, _ = met[i][j]
                if not common and any(row[j][0] for row in met):
                    continue
                if (left.start, left.stop) != (0, shape(node)[dimension]):
                    node = call(graph, SLICE, (node, dimension, left.start, left.stop), holder or processor)
            elif dimension is not None:
                node = cut(graph, node, dimension, sizes, j, len(processors), holder or processor,
                           wide and held is None and holder in (None, processor))
            pieces.append(node)
            sources.append(i)
        if held == dimension is not None:
            blocks = [(held, (i, devices.index(processor)), *arrays[:2]) for i in sources]
        elif dimension is None:
            joined = tuple(shape(piece)[held] for piece in pieces) if held is not None else ()
            placed = (tuple(range(len(pieces))) if extents == STRIPED else
                      tuple(itertools.accumulate(joined, initial=0))[:-1])
            blocks = [(held, k, joined, placed) for k in range(len(joined))]
        else:
            blocks = [(dimension, j, given, displs[:len(given)])] * len(pieces)
        for piece, block in zip(pieces, blocks):
            piece.meta.setdefault("blocks", {})[processor] = block
            if None not in (held, dimension):
                exchange = exchange or f"{nodes[0].name}:{dimension}:{len(graph.nodes)}"
                piece.meta.setdefault("collectives", {})[processor] = kind, exchange
        arrived.append(pieces[0] if len(pieces) == 1 else interleaved(graph, pieces, held, processor)
                       if extents == STRIPED else reassembled(graph, pieces, held, processor))
        if wide and shape(arrived[-1])[dimension] < most(sizes):
            arrived[-1] = padded(graph, arrived[-1], dimension, sizes, j, processor)
    return arrived


# MPI 4.1, sections 7.5 to 7.8, and section 5.1.2, MPI_TYPE_VECTOR
def arrayed(blocks):
    return (tuple(len(range(*block)) for block in blocks), tuple(block[0] for block in blocks),
            tuple(block[2] for block in blocks))


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


def spread(value, count):
    return list(value) * (count // len(value)) if isinstance(value, (list, tuple)) else [value] * count


# Y. Xu et al., GSPMD, arXiv:2105.04663, section 4.3 and appendix A.2
def windows(node):
    names, read = given(node)
    count, dtype = len(shape(node)) - 2, node.all_input_nodes[0].meta["val"].dtype
    low, edged = spread(read.get("padding", 0), count), False
    if node.target is CONVOLUTION:
        kernel, fill = shape(node.args[1])[2:], 0
        if read["transposed"] or len(kernel) != count:
            return {}
    else:
        kernel, fill = spread(read["kernel_size"], count), 0
        if node.target in MAXIMA:
            fill = -math.inf if dtype.is_floating_point else torch.iinfo(dtype).min
            if any(user.args[1] for user in node.users):
                return {}
        else:
            edged = read.get("divisor_override") is None and (read.get("ceil_mode", False) or (
                not read.get("count_include_pad", True) and any(low)))
    dilation = spread(read.get("dilation", 1), count)
    stride = spread(read.get("stride") or kernel, count)
    return {2 + k: (dilation[k] * (kernel[k] - 1) + 1, stride[k], low[k], fill, edged) for k in range(count)}


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
# design/problems-of-14b9a757-change-request.md items 4 and 5
def halos(graph, value, window, dimension, sizes, processors, name):
    (span, stride, low, fill, edged), (held, extents, holders, nodes) = window, value
    size = shape(nodes[0])[dimension] if held is None else sum(extents)

    def read(first, count):
        return first * stride - low, (first + count - 1) * stride - low + span
    reads = [read(first, max(sizes)) for first in bases(sizes)]
    if held is None:
        regions, kept = [], []
        for processor, first, (low_, high) in zip(processors, bases(sizes), reads):
            whole = 0 <= low_ and high <= size
            start = low_ if whole else max(0, first - -(-low // stride)) * stride
            regions.append(nodes[0] if (start, min(size, high)) == (0, size) else call(
                graph, SLICE, (nodes[0], dimension, start, min(size, high)), holders[0] or processor))
            kept.append(None if whole else first - start // stride)
        return regions, kept
    real = [(max(0, min(size, first)), max(0, min(size, last if count else first))) for count, (first, last) in zip(
        sizes, map(read, itertools.accumulate(sizes[:-1], initial=0), sizes))]
    kept = [None if not edged or 0 <= low_ and high <= size else first - max(0, first - -(-low // stride))
            for first, (low_, high) in zip(bases(sizes), reads)]
    for q, (first, (low_, high)) in enumerate(zip(bases(sizes), reads)):
        if kept[q] is not None:
            reads[q] = real[q] = (first - kept[q]) * stride, min(size, high)
    starts = list(itertools.accumulate(extents, initial=0))
    cuts = [[(max(low_, real[q][0], starts[p]), min(high, real[q][1], starts[p + 1]))
             for p in range(len(starts) - 1)] for q, (low_, high) in enumerate(reads)]
    shared = [[max(0, last - first) for first, last in row] for row in cuts]
    cover = [[bool(shared[q][p]) * (starts[p + 1] - first if p < q else last - starts[p])
              for p, (first, last) in enumerate(row)] for q, row in enumerate(cuts)]
    regions = []
    for q, (processor, (low_, high)) in enumerate(zip(processors, reads)):
        parts, rank = [], len(shape(nodes[q]))
        for p, node in enumerate(nodes):
            first, extent = cuts[q][p][0] - starts[p], starts[p + 1] - starts[p]
            if not shared[q][p]:
                continue
            if p == q and shared[q][p] < extent:
                node = call(graph, SLICE, (node, dimension, first, first + shared[q][p]), processor)
            elif p != q:
                width = max(cover[r][r + p - q] for r in range(len(nodes)) if 0 <= r + p - q < len(nodes))
                if width < extent:
                    node = call(graph, SLICE, (node, dimension, *((extent - width, extent) if p < q else (0, width))),
                                processors[p])
                elif width > extent:
                    node = call(graph, PAD, (node, [0, 0] * (rank - 1 - dimension) + [
                        width - extent, 0][::1 if p < q else -1], 0), processors[p])
                node.meta.setdefault("collectives", {})[processor] = "permute", f"{name}:{dimension}:{q - p}"
                node.meta.setdefault("blocks", {})[processor] = dimension, (p, q), tuple(extents), tuple(starts[:-1])
                if shared[q][p] < width:
                    skip = width - cover[q][p] if p < q else first
                    node = call(graph, SLICE, (node, dimension, skip, skip + shared[q][p]), processor)
            parts.append(node)
        for before, missing in ((True, min(high, real[q][0]) - low_), (False, high - max(real[q][1], low_))):
            if missing > 0:
                parts.insert(0 if before else len(parts), call(graph, ZEROS, (nodes[q], [
                    missing if d == dimension else extent for d, extent in enumerate(shape(nodes[q]))]), processor))
        region = parts[0] if len(parts) == 1 else call(graph, CAT, (parts, dimension), processor)
        if low_ < 0 or high > size:
            region = selected(graph, region, dimension, max(0, -low_), min(high, size) - low_, fill, processor)
        regions.append(region)
    return regions, kept


# design/unequal-shards-change-request.md item 8
def accumulated(node, shapes, out):
    table, indices = node.args[:2]
    if not (given(node)[1].get("accumulate") and len(indices) == 1 and table.target is ATEN.full.default
            and table.args[1] == 0):
        return replicated(node, shapes, out)
    summed = [-1 - d for d in range(len(shapes[1]))]
    return [list(range(len(out))), summed, summed + list(range(1, len(out)))]


# design/aot-compilation-change-request.md A2 and A4
def replicated(node, shapes, out):
    return [[None] * len(s) for s in shapes]


RULES = {
    ATEN.alias.default: aligned, EXPAND: aligned, ATEN.full_like.default: aligned, ATEN._to_copy.default: aligned,
    operator.getitem: aligned, FLIP: aligned,
    CONVOLUTION: windowed, **dict.fromkeys(MAXIMA | AVERAGES, windowed),
    ATEN.index_put.default: accumulated,
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
        return (at, (dimension, counts(new[at], STRIPED, len(processors)), processors, nodes)) if few == many or (
            many % few, few % len(processors)) == (0, 0) else None
    edges = [stride * edge for edge in itertools.accumulate(sizes)]
    for at in reversed(range(len(new))):
        unit = math.prod(new[at + 1:])
        if all(edge % unit == 0 for edge in edges) and unit * new[at] % edges[-1] == 0:
            return at, (dimension, sized((stride * size // unit for size in sizes), getattr(sizes, "units", None)),
                        processors, nodes)


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
    inputs = mesh_graph.Graph.adjacency(node.graph.nodes, lambda m: m.all_input_nodes)
    views = inputs.walk([node], incoming=True, active=[len(n.users) == 1 and next(iter(n.users)).target is VIEW
                                                       for n in inputs.vertices])
    at = [rows, keys]
    for before, after in zip(views, views[1:]):
        at = [intact(shape(before), shape(after))[d] for d in at]
    node, user = views[-1], next(iter(views[-1].users))
    every = [list(range(len(held))) for _ in queries]
    dependent = inputs.reach([n for n in inputs.vertices if n.op == "placeholder"], incoming=True)
    constants = [n for n in user.all_input_nodes if n is not node and n not in dependent]
    reached = inputs.reach([user], incoming=True, active=[n.target is not BMM for n in inputs.vertices])
    if len(node.users) != 1 or user.target is not ADD or None in at or not constants or not all(
            n.target in MASKED for n in reached):
        return every
    order = inputs.postorder(constants[:1])
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
    return *value[:3], [plain(graph, node, holder) for node, holder in zip(value[3], flat(value[2]))]


# the whole-value cache: `node`'s placed value gathered whole onto `processor` once (reshard)
def gathered(graph, whole, node, value, processor):
    if (node, processor) not in whole:
        whole[node, processor] = reshard(graph, plainly(graph, value), None, None, (processor,))[0]
    return whole[node, processor]


# torch 2.14.0 torch/_inductor/ir.py ExternKernel.require_strides, its exact-strides geometry: each required
# zero-stride dimension of size two or more sliced to [0:1], the value copied (ir.evaluated's CLONE out) into the
# source prototype's own geometry, aten.as_strided at its offset and strides over an empty_strided allocation of its
# storage's whole extent in its dtype, then expanded back to its size, so no expanded element is written twice
def required(graph, node, like, processor):
    expanded = [d for d, (size, stride) in enumerate(zip(like.shape, like.stride())) if stride == 0 and size >= 2]
    for d in expanded:
        node = call(graph, SLICE, (node, d, 0, 1), processor)
    storage = constant(graph, EMPTY_STRIDED, ([like.untyped_storage().nbytes() // like.element_size()], [1]), node,
                       processor, dtype=like.dtype)
    out = call(graph, AS_STRIDED, (storage, list(shape(node)), list(like.stride()), like.storage_offset()), processor)
    node = call(graph, CLONE, (node,), processor, memory_format=torch.preserve_format, out=out)
    return call(graph, EXPAND, (node, list(like.shape)), processor) if expanded else node


# torch 2.14.0 torch/_tensor_docs.py stride, storage_offset and untyped_storage: the same layout in the same extent
def laid(value, like):
    return (tuple(value.stride()), value.storage_offset(), value.untyped_storage().nbytes()) == (
        tuple(like.stride()), like.storage_offset(), like.untyped_storage().nbytes())


# design/problems-of-14b9a757-change-request.md item 4 steps 2 and 3; Y. Xu et al., GSPMD, arXiv:2105.04663,
# section 4.3 and appendix A.2
def meshed(graph, node, held, replaced):
    value = next(v for v in held if several(v))
    dims, extents, grid, nodes = value
    leaves, out, shapes = flat(grid), produced(node), read_shapes(node)
    index = list(itertools.product(*(range(n) for n in spans(grid, len(dims)))))
    found = windows(node) if RULES.get(node.target) is windowed else {}

    def local(v, processor):
        return reshard(graph, plainly(graph, v), None, None, (processor,))[0]
    if (found and held[0] is value and all(d in found and not found[d][4] for d in dims) and STRIPED not in extents
            and all(v[0] is None for v in held[1:])):
        sizes = tuple(counts(out[d], getattr(e, "units", None), len(e)) for d, e in zip(dims, extents))
        regions, exchanged = list(nodes), [d for d, e in zip(dims, extents) if len(e) > 1]
        for a, d in enumerate(dims):
            regions = [own(graph, n, d, extents[a], at[a]) for n, at in zip(regions, index)]
        for a, d in enumerate(dims):
            lines = {}
            for k, at in enumerate(index):
                lines.setdefault(at[:a] + at[a + 1:], []).append(k)
            for line in lines.values() if d in exchanged else ():
                members = tuple(leaves[k] for k in line)
                made, _ = halos(graph, (d, extents[a], members, [regions[k] for k in line]), found[d], d, sizes[a],
                                members, node.name)
                for k, region in zip(line, made):
                    regions[k] = region
        columns, rows = [regions, *([local(v, leaf) for leaf in leaves] for v in held[1:])], ()
    elif node.target not in (EXPAND, FLIP) and (
            torch.Tag.pointwise in getattr(node.target, "tags", ()) or RULES.get(node.target) is aligned):
        rows = aligned(node, shapes, out)
        found = {(tuple(row[d] for d in v[0]), v[1], v[2]) for v, row in zip(held, rows) if several(v)}
        (dims, sizes, grid), exchanged = next(iter(found)), ()
        if len(found) > 1 or None in dims or not all(v[0] is None and not set(dims) & set(row)
                                                      for v, row in zip(held, rows) if not several(v)):
            return None
        columns = [v[3] if several(v) else [local(v, leaf) for leaf in leaves] for v in held]
    else:
        return None
    shards = []
    for k, (processor, at) in enumerate(zip(leaves, index)):
        given = [column[k] for column in columns]
        arguments = iter(given)
        shard = graph.node_copy(node, lambda n: next(arguments))
        # metal-microbench metal_ir.h metal_ir_operation_axis (compiler/mesh_compiler/lower_metal.py construct): (FX
        # value, dimension, original full extent) per mesh dimension d, of the aligned values the shard reads along d
        # and of its own output
        shard.meta["work_axes"] = (
            *((n, row.index(d), shapes[i][row.index(d)]) for i, (n, row) in enumerate(zip(given, rows)) for d in dims
              if d in row),
            *((shard, d, out[d]) for d in dims))
        for d in exchanged:
            unpadded(shard, d - 2)
        shard.meta.update(of=node.name, order={}, pairs=0, units=(
            math.prod(len(range(j, out[d], shards)) if size == STRIPED else max(size) if uneven(size) else size[j]
                      for d, size, j, shards in zip(dims, sizes, at, spans(grid, len(dims)))),
            math.prod(out[d] for d in dims)))
        shards.append(evaluated(shard, processor))
    replaced.append(node)
    return dims, sizes, grid, shards


# Rejection filters are not part of even temporary integration validation.
# A partition-rule miss retains the operation; transfer() reshards its result to the requested placement.
def partitioned(graph, node, held, want, replaced, whole):
    target = None
    if any(several(v) for v in held):
        made = meshed(graph, node, held, replaced) if node.op == "call_function" else None
        if made is not None:
            return made
    if not isinstance(want, str) and several(want):
        want = flat(want[2])[0]
    one = want if isinstance(want, str) else want[2][0]
    for u, v in zip(operands(node), held, strict=True):
        if several(v) and (u, one) not in whole:
            whole[u, one] = reshard(graph, v, None, None, (one,))[0]
    held = [(None, None, (one,), [whole[u, one]]) if several(v) else v for u, v in zip(operands(node), held)]
    if node.op == "call_function" and (any(v[0] is not None for v in held) or not isinstance(want, str)):
        if node.target is ADDMM:
            product = graph.call_function(MM, node.args[1:])
            product.meta = dict(node.meta)
            first = want if isinstance(want, str) else want[2][0]
            held = [partitioned(graph, product, held[1:], first, replaced, whole), held[0]]
            node.target, node.args = ADD, (product, node.args[0])
        shapes, out = read_shapes(node), produced(node)
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
        units = getattr(target[1], "units", None)
        target = target[0], counts(out[target[0]], units, len(target[2])), target[2]
    if target is None:
        processor = want if isinstance(want, str) else want[2][0]
        arguments = iter([gathered(graph, whole, u, v, processor) for u, v in zip(operands(node), held, strict=True)])
        node.args, node.kwargs = map_arg((node.args, node.kwargs), lambda n: next(arguments))
        node.meta["processor"] = processor
        return None, None, (processor,), [node]
    along, sizes, processors = target
    extent = out[along] if along >= 0 else next(s[row.index(along)] for s, row in zip(shapes, rows) if along in row)
    if sizes == STRIPED:
        sizes = counts(extent, STRIPED, len(processors))
    if window:
        value = held[0]
        if (value[0], value[2]) not in ((None, (None,)), (along, processors)):
            value = along, counts(shapes[0][along], units, len(processors)), processors, None
            value = *value[:3], reshard(graph, plainly(graph, held[0]), *value[:3], True)
        elif value[0] is not None:
            value = *value[:3], [own(graph, n, along, value[1], k) for k, n in enumerate(value[3])]
        regions, kept = halos(graph, value, window, along, sizes, processors, node.name)
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
    if node.target is FLIP:
        columns[0] = [plain(graph, shard, shard.meta["processor"]) if {d % len(out) for d in node.args[1]} & set(
            shard.meta.get("order", {})) else shard for shard in columns[0]]
    turned = node.target is FLIP and along in [d % len(out) for d in node.args[1]]
    if turned and sizes == STRIPED:
        stripes = [own(graph, columns[0][s], along, sizes, s) for s in (
            (out[along] - 1 - j) % len(processors) for j in range(len(processors)))]
        for j, stripe in enumerate(stripes):
            source = (out[along] - 1 - j) % len(processors)
            stripe.meta.setdefault("collectives", {})[processors[j]] = "permute", f"{node.name}:{along}"
            stripe.meta.setdefault("blocks", {})[processors[j]] = along, (source, j), *arrayed(
                ranges(sizes, out[along], len(processors)))
        columns[0] = [padded(graph, n, along, sizes, j, processors[j], True) for j, n in enumerate(stripes)]
    elif turned:
        columns[0] = [padded(graph, n, along, sizes, j, processors[j], True) for j, n in enumerate(reshard(
            graph, (along, sizes, processors, columns[0]), along, sizes[::-1], processors[::-1], True)[::-1])]
    exact = [[own(graph, n, v[0], v[1], k) for k, n in enumerate(v[3])] if join else None
             for v, join in zip(held, joined)]
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
                pieces = [exact[i][kept.index(block)] for block in order[label]]
                placed = dict(zip(order[label], itertools.accumulate((len(range(*b)) for b in order[label]), initial=0)))
                for block, piece in zip(order[label], pieces):
                    piece.meta.setdefault("blocks", {})[processor] = (
                        v[0], kept.index(block), tuple(len(range(*b)) * (b in placed) for b in kept),
                        tuple(placed.get(b, 0) for b in kept))
                local[i] = pieces[0] if len(pieces) == 1 else call(graph, CAT, (pieces, v[0]), processor)
        for i, row in enumerate(rows):
            for dimension, label in enumerate(row):
                if label in order and not joined[i] and dimension not in local[i].meta.get("order", {}):
                    local[i] = chosen(graph, local[i], dimension, order[label], processor)
        pairs = sum(len(order[row[v[0]]]) for v, row, join in zip(held, rows, joined) if join)
        order = {label: blocks for label, blocks in order.items() if label >= 0 and
                 [index for block in blocks for index in range(*block)] != list(range(out[label]))}
        arguments = iter(local)
        if along < 0:
            local = [masked(graph, n, row.index(along), sizes, j, processor) if along in row else n
                     for n, row in zip(local, rows)]
            arguments = iter(local)
        shard = graph.node_copy(node, lambda n: next(arguments))
        # metal-microbench metal_ir.h metal_ir_operation_axis (compiler/mesh_compiler/lower_metal.py construct): (FX
        # value, dimension, original full extent) of each value the shard reads along `along`, and of its own output
        # where `along` is an output dimension
        shard.meta["work_axes"] = (
            *((n, row.index(along), shapes[i][row.index(along)]) for i, (n, row) in enumerate(zip(local, rows))
              if along in row),
            *(((shard, along, out[along]),) if along >= 0 else ()))
        if window and kept[j] is None:
            unpadded(shard, along - 2)
        if node.target in (EXPAND, VIEW):
            sized = list(shard.args[1])
            sized[along] = (most(sizes) if sizes == STRIPED else
                            out[along] * (max(sizes) if uneven(sizes) else sizes[j]) // sum(sizes))
            for label, blocks in order.items():
                sized[label] = sum(len(range(*block)) for block in blocks)
            shard.args = (shard.args[0], sized)
        shard.meta.update(of=node.name, order=order, pairs=pairs, units=(
            len(range(j, extent, len(processors))) if sizes == STRIPED else max(sizes) if uneven(sizes) else sizes[j],
            extent if sizes == STRIPED else sum(sizes)))
        shards.append(evaluated(shard, processor))
        if window and kept[j] is not None:
            shard.meta.update(cut=(kept[j], kept[j] + max(sizes)), units=(shape(shard)[along], sum(sizes)))
            if shard.meta["cut"] != (0, shape(shard)[along]):
                shards[-1] = call(graph, SLICE, (shard, along, *shard.meta["cut"]), processor)
                shards[-1].meta["order"] = order
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
                keeper = to[2][kept.index(block)]
                piece = shard if len(blocks) == 1 else call(graph, SLICE, (shard, to[0], low, high), processor)
                piece.meta.setdefault("blocks", {})[keeper] = to[0], kept.index(block), *arrayed(kept)[:2]
                parts[keeper].append(piece)
        else:
            column = reshard(graph, (None, None, (processor,), [plain(graph, shard, processor, out)]), *to, True)
            for keeper, part in zip(to[2], column):
                parts[keeper].append(part)
    for k, held_ in enumerate(parts.values()):
        for part in held_:
            part.meta.setdefault("collectives", {})[to[2][k]] = "reduce", f"{node.name}:{k}"
    return *to, [padded(graph, functools.reduce(lambda a, b: call(graph, ADD, (a, b), keeper), held_), to[0], to[1],
                        k, keeper) if to[0] is not None else functools.reduce(
                            lambda a, b: call(graph, ADD, (a, b), keeper), held_)
               for k, (keeper, held_) in enumerate(parts.items())]


def place(graph_module, placement):
    graph_module = copy.deepcopy(graph_module)
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placed = [n for n in nodes if n.op != "placeholder"]
    stated = {n: placement(n) for n in placed if n.target is not operator.getitem}
    stated.update((n, placement(n)) for n in nodes if n.op == "placeholder")
    for node in nodes:
        if node.op == "placeholder":
            node.meta["stated"] = stated[node]
    replaced, whole = [], {}
    # design/compilation-composition-change-request.md item 7; ir.view and ir.aliasing, the source alias owner: each
    # node of a user output position (a mutation's, export.read "destination", keeps its own producer) and each pure
    # source alias parent it is read through (an aliasing call's self; an aliasing call's getitem's call), in their
    # postorder from those outputs, retained before placement rewrites or erases them: (its parent, its original
    # operands, its prototype); the layout normalization's domain the Tensor-valued output Nodes, every other output
    # operand placed by the output's own transfer as before
    output = graph.output_node()
    mutated = output.meta.get("destination", {})
    mutations = {output.args[0][k] for k in mutated}
    results = [n for k, n in enumerate(output.args[0])
               if k not in mutated and isinstance(n, torch.fx.Node) and isinstance(n.meta.get("val"), torch.Tensor)]

    # a pure source alias's replay predecessors: a getitem's call where that result is a view of an argument it aliases
    # without writing it (ir.view over ir.aliased); an aliasing call's own aliased argument Nodes, every result's
    # (ir.aliased: every get_alias_info read-only match, torch's _correct_storage_aliasing pairing; ir.leaves_of: a
    # Tensor[] argument's each Node), each once, realize replacing each by its realization
    def parent(node):
        if node.target is operator.getitem:
            return [node.args[0]] if set(view(node)) & set(leaves_of(aliased(node.args[0], node.args[1]))) else []
        if not aliasing(node):
            return []
        return list(leaves_of(value for k in results_of(node) for value in aliased(node, k)))
    order = mesh_graph.postorder(nodes, results, parent)
    retained = {node: (parent(node), node.args, node.kwargs, node.meta["val"]) for node in order}

    # the stated placement of a value: its own, a getitem's its producer's (every result of a multi-result value under
    # its producer's one statement)
    def wanted(node):
        return stated[node] if node in stated else wanted(node.args[0])

    # a stated placement at result `k` of `node`'s own shape (ir.shape): a sharded one's sizes counted over that
    # result's extents
    def measured(want, node, k=None):
        if isinstance(want, str):
            return want
        dimension, processors, units = want
        if several(want):
            return dimension, tuple(counts(shape(node, k)[d], unit, shards) for d, unit, shards in zip(
                dimension, units, spans(processors, len(dimension)))), processors
        return dimension, counts(shape(node, k)[dimension], units, len(processors)), processors

    # each value placed once and then resharded to its own measured placement: a getitem projecting its producer's
    # selected result where the producer holds it (its first processor, its own partition the getitem rule's
    # alignment), then that result resharded to its producer's statement at its own shape; a multi-result value's
    # shared operation partitioned under its statement, one partition where every result's measured placement is the
    # same, else each result's own (Measured), the value itself never resharded (its getitems' transfers reshard each
    # result); any other value under its own statement at its own shape
    def transfer(node, held):
        if node.target is operator.getitem:
            want, resharded = flat(held[0][2])[0], measured(wanted(node), node)
        elif isinstance(node.meta.get("val"), (tuple, list)):
            wants = [measured(stated[node], node, k) for k in results_of(node)]
            want, resharded = wants[0] if all(w == wants[0] for w in wants) else (
                stated[node][0], Measured(w[1] for w in wants), stated[node][1]), None
        else:
            want = resharded = measured(stated[node], node)
        with graph.inserting_before(placed[0] if node.op == "placeholder" else node):
            value = ((None, None, (None,), [node]) if node.op == "placeholder" else
                     partitioned(graph, node, held, want, replaced, whole))
        if resharded is not None and not isinstance(resharded, str) and value[:3] != resharded:
            with graph.inserting_before(placed[0] if node.op == "placeholder" else node.next):
                value = *resharded, reshard(graph, plainly(graph, value), *resharded)
        return value

    # torch.fx Graph.output_node, the graph's last node: every other node placed first; then torch 2.14.0
    # torch/_inductor/ir.py ExternKernel.require_strides and ir.views: each retained node realized whole on the output
    # processor once, from those placed values, by the alias forward over `order` (a shared base realized once for all
    # its aliases): a source alias the replay of its original operation on its parent's realization (a tuple producer's
    # then its getitem's); an input placeholder that input itself; a mutation's value its whole value (ir.views computes
    # it into its input); another its whole value in its original layout and extent (`required` where that differs);
    # each user output's realization its whole value there, then the output's own transfer once
    values = mesh_graph.forward(nodes[:-1], operands, transfer)
    processor = stated[output]

    def realize(node, read):
        upstream, args, kwargs, like = retained[node]
        if read:
            realized = dict(zip(upstream, read, strict=True))
            arguments = map_arg((args, kwargs), lambda n: realized[n] if n in realized else gathered(
                graph, whole, n, values[n], processor))
            return evaluated(graph.call_function(node.target, *arguments), processor)
        if node.op == "placeholder":
            return node
        value = gathered(graph, whole, node, values[node], processor)
        return value if node in mutations or laid(value.meta["val"], like) else required(graph, value, like, processor)
    with graph.inserting_before(output):
        made = mesh_graph.forward(order, lambda node: retained[node][0], realize)
    whole.update(((node, processor), made[node]) for node in results)
    values[output] = transfer(output, [values[u] for u in operands(output)])
    # each placeholder's own placed value: its holders' partition Nodes in holder order, or a whole one its stated
    # processor's and the placeholder itself, each with its landing geometry, (storage offset, shape, stride) of its
    # source view in the placeholder's storage (`landed`; a whole one's its own value's), None for a non-tensor
    def landing(part):
        source = part.meta.get("landing", part.meta.get("val"))
        return (source.storage_offset(), tuple(source.shape), tuple(source.stride())) if isinstance(
            source, torch.Tensor) else None
    graph_module.meta["input_parts"] = {
        node: tuple((processor, part, landing(part)) for processor, part in zip(
            (stated[node],) if values[node][2] == (None,) else flat(values[node][2]), values[node][3]))
        for node in nodes if node.op == "placeholder"}
    for node in reversed(replaced):
        graph.erase_node(node)
    # torch 2.14.0 torch/fx/graph.py Graph.eliminate_dead_code and torch/fx/node.py Node.is_impure: every node the
    # rewired outputs no longer read removed before any transfer, each recorded input part (input_parts) kept
    parts = {part for held in graph_module.meta["input_parts"].values() for _, part, _ in held}
    graph.eliminate_dead_code(is_impure_node=lambda node: node.is_impure() or node in parts)
    views(graph)
    # design/problems-of-14b9a757-change-request.md item 8, steps 1 to 3
    invariant = mesh_graph.forward(list(graph.nodes), lambda n: n.all_input_nodes, lambda n, read: (
        "state" in n.meta if n.op == "placeholder" else bool(view(n)) and all(read)))
    # design/unequal-shards-change-request.md item 8: a call replicated where invariant or of no placeholder (the
    # placeholders' users' incoming search, once), each replicated call onto the processors of its users and of their
    # sets, an unreplicated vertex's set empty (one labelled closure over the incoming rows, its columns the processors)
    inputs = mesh_graph.Graph.adjacency(graph.nodes, lambda m: m.all_input_nodes)
    dependent = inputs.reach([n for n in inputs.vertices if n.op == "placeholder"], incoming=True)
    replicated = [n.op == "call_function" and (invariant[n] or n not in dependent) for n in inputs.vertices]
    label = [n.meta.get("processor") for n in inputs.vertices]
    onto = inputs.labels(inputs.postorder(inputs.vertices, incoming=True), sorted(set(label) - {None}), label,
                         incoming=True, active=replicated)
    copies = {}
    for node, active in zip(inputs.vertices, replicated):
        targets = [processor for processor in onto[node] if processor != node.meta["processor"]] if active else ()
        if targets:
            copies[node, node.meta["processor"]] = node
        for processor in targets:
            with graph.inserting_after(node):
                there = graph.node_copy(node, lambda n: copies.get((copies.get(n, n), processor), n))
            there.meta.update(processor=processor, constant=there.meta.pop("of", node.name))
            node.replace_all_uses_with(there, lambda user: user is not there and user.meta["processor"] == processor)
            copies[node, processor], copies[there] = there, node
    placed = [n for n in graph.nodes if n.op != "placeholder"]
    transfers, instances = {}, {}
    for node in placed:
        transfers.update(insert_send_recv(graph, node, len(transfers), instances))
    for node in graph.nodes:
        node.meta.pop("blocks", None)
        node.meta.pop("collectives", None)
    views(graph)
    # design/problems-of-14b9a757-change-request.md item 8, step 4
    held, returned = {}, set(graph.output_node().all_input_nodes)
    for node in graph.nodes:
        for read in node.all_input_nodes if node.op == "call_function" and "view" not in node.meta else ():
            if {"invariant", "state"} & read.meta.keys() and read not in returned and all(shape(read)):
                held.setdefault(node.meta["processor"], {})[read.meta.get("invariant", (read.meta.get("state"),))] = None
    graph_module.meta["invariant"] = {processor: tuple(slices) for processor, slices in held.items()}
    graph_module.meta["transfers"] = transfers
    graph_module.recompile()
    return graph_module, placed[-1].meta["processor"], len(transfers)


# M. Abadi et al., TensorFlow, arXiv:1603.04467, section 3.2.2; design/unequal-shards-change-request.md items 2, 3
# and 12: the transfer's geometry is the sent value's ir.views record, written after this by views(graph)
def insert_send_recv(graph, node, first, instances):
    here = node.meta["processor"]
    away = {}
    for user in node.users:
        if user.meta["processor"] != here:
            away.setdefault(user.meta["processor"], []).append(user)
    if not away:
        return {}
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{node.name} is read on another processor; only tensors cross processors")
    made = {}
    for channel, (processor, users) in enumerate(away.items(), first):
        with graph.inserting_after(node):
            sent = graph.call_function(SEND, (node, channel))
            sent.meta["processor"] = here
        with graph.inserting_before(users[0]):
            arrived = graph.call_function(RECV, (channel,))
            arrived.meta.update(processor=processor, val=value)
        # design/aot-compilation-change-request.md A4 and appendix item 5; design/plan-to-bridge-change-request.md R4
        kind, exchange = node.meta.get("collectives", {}).get(processor, ("broadcast", node))
        collective = kind, instances.setdefault((kind, exchange), len(instances))
        arrived.meta["consumer"] = PAGES if kind == "reduce" else JOIN
        for end in (sent, arrived):
            end.meta.update(zip(("dimension", "block", "counts", "displs", "strides"),
                                node.meta.get("blocks", {}).get(processor, ())), collective=collective)
        for user in users:
            user.replace_input_with(node, arrived)
        made[channel] = sent, arrived
    return made

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

from .ir import PAGES, RECV, SEND

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_graph

ATEN = torch.ops.aten
ADD, ADDMM, CAT, MM, SLICE = (ATEN.add.Tensor, ATEN.addmm.default, ATEN.cat.default, ATEN.mm.default,
                              ATEN.slice.Tensor)
EXPAND, RESHAPE, VIEW = ATEN.expand.default, ATEN.reshape.default, ATEN.view.default
SLICE_SCATTER, ZEROS = ATEN.slice_scatter.default, ATEN.new_zeros.default
BMM, FLIP, SOFTMAX = ATEN.bmm.default, ATEN.flip.default, ATEN._softmax.default
ARANGE, LE, AND, NOT = ATEN.arange.start_step, ATEN.le.Scalar, ATEN.logical_and.default, ATEN.logical_not.default
PAD = ATEN.constant_pad_nd.default
CONVOLUTION = ATEN.convolution.default
MAXIMA = {ATEN.max_pool2d_with_indices.default, ATEN.max_pool3d_with_indices.default}
AVERAGES = {ATEN.avg_pool1d.default, ATEN.avg_pool2d.default, ATEN.avg_pool3d.default}
SYMMETRIC = {SOFTMAX, ATEN.any.dim, ATEN.any.dims, ATEN.mean.dim}
MASKED = {ADD, BMM, EXPAND, SOFTMAX, VIEW, ATEN.clone.default, ATEN.eq.Scalar, ATEN.logical_not.default, ATEN.any.dim,
          ATEN.full_like.default, ATEN.where.self}
STRIPED = "striped"
CLONE, EMPTY = ATEN.clone.out, ATEN.empty.memory_format


def operands(node):
    found = []
    map_arg((node.args, node.kwargs), found.append)
    return found


def shape(node):
    value = node.meta["val"]
    return tuple((value[0] if isinstance(value, (tuple, list)) else value).shape)


# design/problems-of-14b9a757-change-request.md item 6; torch 2.15 torch/_tensor_docs.py view, reshape and contiguous
def viewable(value, sizes):
    try:
        value.view(sizes)
    except (RuntimeError, ValueError):
        return False
    return True


def evaluated(node, processor):
    arguments = map_arg((node.args, node.kwargs), lambda n: n.meta["val"])
    with detect_fake_mode(arguments):
        if node.target in (RESHAPE, VIEW) and not viewable(*arguments[0]):
            with node.graph.inserting_before(node):
                held = node.args[0]
                out = constant(node.graph, EMPTY, (list(shape(held)),), held, processor, dtype=held.meta["val"].dtype)
                copied = call(node.graph, CLONE, (held,), processor, memory_format=torch.contiguous_format, out=out)
            node.args = (copied, *node.args[1:])
            return evaluated(node, processor)
        node.meta.update(processor=processor, val=node.target(*arguments[0], **arguments[1]))
    return node


def call(graph, target, arguments, processor, **kwargs):
    return evaluated(graph.call_function(target, arguments, kwargs), processor)


# design/problems-of-14b9a757-change-request.md items 3 and 6; torch/csrc/jit/frontend/function_schema_parser.cpp
# alias_info
def aliasing(node):
    returns = getattr(getattr(node.target, "_schema", None), "returns", ())
    return node.op == "call_function" and bool(returns) and returns[0].alias_info is not None and not returns[
        0].alias_info.is_write


# torch/csrc/jit/frontend/function_schema_parser.cpp alias_info; ExecuTorch documentation, Concepts, Out variant
def written(node):
    schema = getattr(node.target, "_schema", None)
    if node.op != "call_function" or schema is None or len(schema.returns) != 1 or schema.returns[0].alias_info is None:
        return None
    names, read = given(node)
    return next((read[a.name] for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write and
                 a.alias_info.before_set == schema.returns[0].alias_info.before_set and a.name in read), None)


def view(node):
    made = node.args[0] if node.target is operator.getitem else node
    held = made.args[0] if aliasing(made) else written(made)
    if held is None:
        return None
    value = node.meta["val"]
    return held if all(torch._C._is_alias_of(v, held.meta["val"]) for v in (
        value if isinstance(value, (tuple, list)) else [value])) else None


def based(node, read):
    value = node.meta.get("val")
    if not read:
        return node, value.storage_offset() if isinstance(value, torch.Tensor) else 0
    (base, first), = read
    if isinstance(value, torch.Tensor):
        record = base.name, value.storage_offset() - first, tuple(value.shape), tuple(value.stride())
        if written(node) is not None:
            node.meta["out"] = record
            return base, first
        node.meta["view"] = record
        if "state" in base.meta:
            node.meta["invariant"] = base.meta["state"], *node.meta["view"][1:]
    return base, first


# design/problems-of-14b9a757-change-request.md item 1 step 1, item 2 and item 4 step 4; MPI 4.1, section 7.7, in place
def destined(node):
    if node.target is CAT:
        value = node.meta["val"]
        along, at = (*node.args, 0)[1] % value.dim(), 0
        for held in node.args[0]:
            if node.args[0].count(held) == 1:
                yield held, at * value.stride(along)
            at += shape(held)[along]
    elif node.target is PAD and (*node.args, 0)[2] == 0 and min(node.args[1]) >= 0:
        value = node.meta["val"]
        yield node.args[0], sum(low * value.stride(value.dim() - 1 - d) for d, low in enumerate(node.args[1][::2]))


# design/problems-of-14b9a757-change-request.md items 1 to 4 and 6, and item 8, step 4
def views(graph):
    for node in list(graph.nodes):
        node.meta.pop("to", None)
        node.meta.pop("out", None)
        if aliasing(node):
            evaluated(node, node.meta["processor"])
        elif node.target is operator.getitem and aliasing(node.args[0]):
            node.meta["val"] = node.args[0].meta["val"][node.args[1]]
    mesh_graph.forward(list(graph.nodes), lambda n: [n for n in [view(n)] if n is not None], based)
    for node in graph.nodes:
        for held, offset in destined(node):
            if (held.op == "call_function" and held.target is not operator.getitem and len(held.users) == 1
                    and not {"view", "out"} & held.meta.keys()
                    and isinstance(held.meta["val"], torch.Tensor)):
                held.meta["to" if held.target is RECV else "out"] = (
                    node.name, offset, shape(held), tuple(node.meta["val"].stride()))


# design/unequal-shards-change-request.md items 1, 13 and 15
class Sizes(tuple):
    units = None


def sized(sizes, units):
    sizes = Sizes(sizes)
    sizes.units = units
    return sizes


def counts(extent, units, shards):
    if units == STRIPED:
        return STRIPED
    if units is None:
        chunk = -(-extent // shards)
        return Sizes(max(0, min(chunk, extent - chunk * j)) for j in range(shards))
    if len(units) == shards + 1:
        return Sizes(high - low for low, high in itertools.pairwise(units))
    exact = [extent * unit / sum(units) for unit in units]
    sizes = [round(size) for size in exact]
    while sum(sizes) != extent:
        step = 1 if sum(sizes) < extent else -1
        sizes[min(range(shards), key=lambda j: abs(sizes[j] + step - exact[j]))] += step
    return sized(sizes, tuple(units))


# design/unequal-shards-change-request.md items 4 and 12; Y. Xu et al., GSPMD, arXiv:2105.04663, section 4.1
def uneven(sizes):
    return sizes not in (None, STRIPED) and min(sizes) < max(sizes)


def bases(sizes):
    return [min(at, sum(sizes) - max(sizes)) for at in itertools.accumulate(sizes[:-1], initial=0)]


def own(graph, node, dimension, sizes, j):
    if not uneven(sizes) or sizes[j] == max(sizes):
        return node
    start = sum(sizes[:j]) - bases(sizes)[j]
    return call(graph, SLICE, (node, dimension, start, start + sizes[j]), node.meta["processor"])


def padded(graph, node, dimension, sizes, j, processor, mirrored=False):
    if not uneven(sizes) or sizes[j] == max(sizes):
        return node
    start = sum(sizes[:j]) - bases(sizes)[j]
    return call(graph, PAD, (node, [0, 0] * (len(shape(node)) - 1 - dimension) + [
        start, max(sizes) - start - sizes[j]][::-1 if mirrored else 1], 0), processor)


def constant(graph, target, arguments, like, processor, **kwargs):
    value = like.meta["val"]
    node = graph.call_function(target, arguments, {**kwargs, "device": value.device})
    with value.fake_mode:
        node.meta.update(processor=processor, val=target(*arguments, **node.kwargs))
    return node


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
    if not uneven(sizes) or sizes[j] == max(sizes):
        return node
    start, extent = sum(sizes[:j]) - bases(sizes)[j], shape(node)[dimension]
    return selected(graph, node, dimension, start, start + sizes[j], 0, processor,
                    max(sizes) if extent > max(sizes) else None)


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


# design/unequal-shards-change-request.md items 2, 3, 8, 9 and 12
def reshard(graph, value, dimension, sizes, processors, exact=False):
    held, extents, holders, nodes = value
    if (held, extents, holders) == (dimension, sizes, processors) and not (exact and uneven(sizes)):
        return nodes
    if len(processors) == 1 or dimension is None:
        dimension, sizes = None, None
    if uneven(extents):
        if (repeats := shape(nodes[0])[held] // max(extents)) > 1:
            return folded(graph, value, held, repeats, [max(extents)] * len(nodes), dimension, sizes, processors, exact)
        nodes = [own(graph, n, held, extents, i) for i, n in enumerate(nodes)]
        if (held, extents, holders) == (dimension, sizes, processors):
            return nodes
    elif held is not None and extents != STRIPED and (whole := sum(shape(n)[held] for n in nodes)) > sum(extents):
        return folded(graph, value, held, whole // sum(extents), extents, dimension, sizes, processors, exact)
    if dimension not in (None, held) and sizes != STRIPED and (whole := shape(nodes[0])[dimension]) > sum(sizes):
        return folded(graph, (held, extents, holders, nodes), dimension, whole // sum(sizes),
                      [sum(sizes)] * len(nodes), dimension, sizes, processors, exact)
    if held == dimension is not None and STRIPED in (extents, sizes):
        return [reshard(graph, (None, None, (processor,), [node]), dimension, sizes, processors, exact)[j]
                for j, (processor, node) in enumerate(zip(processors, reshard(
                    graph, (held, extents, holders, nodes), None, None, processors)))]
    wide = not exact and dimension is not None and uneven(sizes)
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
                elif wide and held is None and holder in (None, processor):
                    low, high = bases(sizes)[j], bases(sizes)[j] + max(sizes)
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
            if None not in (held, dimension):
                piece.meta.setdefault("collectives", {})[processor] = "alltoall", nodes[0].name
        arrived.append(pieces[0] if len(pieces) == 1 else interleaved(graph, pieces, held, processor)
                       if extents == STRIPED else call(graph, CAT, (pieces, held), processor))
        if wide and shape(arrived[-1])[dimension] < max(sizes):
            arrived[-1] = padded(graph, arrived[-1], dimension, sizes, j, processor)
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
        if node.target in MAXIMA:
            fill = -math.inf if dtype.is_floating_point else torch.iinfo(dtype).min
            if any(user.args[1] for user in node.users):
                return {}
        elif read.get("divisor_override") is None and not read.get("count_include_pad", True) and any(low):
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
# design/problems-of-14b9a757-change-request.md items 4 and 5
def halos(graph, value, window, dimension, sizes, processors, name):
    (span, stride, low, fill), (held, extents, holders, nodes) = window, value
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
    return regions, [None] * len(regions)


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
        return (at, value) if few == many or (many % few, few % len(processors)) == (0, 0) else None
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
        units = getattr(target[1], "units", None)
        target = target[0], counts(out[target[0]], units, len(target[2])), target[2]
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
    if turned:
        columns[0] = ([columns[0][(out[along] - 1 - j) % len(processors)] for j in range(len(processors))]
                      if sizes == STRIPED else [padded(graph, n, along, sizes, j, processors[j], True) for j, n in
                                                enumerate(reshard(graph, (along, sizes, processors, columns[0]), along,
                                                                  sizes[::-1], processors[::-1], True)[::-1])])
    exact = [[own(graph, n, v[0], v[1], k) for k, n in enumerate(v[3])] if join else None
             for v, join in zip(held, joined)]
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
                pieces = [exact[i][kept.index(block)] for block in order[label]]
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
        if along < 0:
            local = [masked(graph, n, row.index(along), sizes, j, processor) if along in row else n
                     for n, row in zip(local, rows)]
            arguments = iter(local)
        shard = graph.node_copy(node, lambda n: next(arguments))
        if window and kept[j] is None:
            unpadded(shard, along - 2)
        if node.target in (EXPAND, VIEW):
            sized = list(shard.args[1])
            sized[along] = (len(range(*queries[j])) if sizes == STRIPED else
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
                piece.meta.setdefault("blocks", {})[keeper] = (
                    to[0], kept.index(block), tuple(len(range(*b)) for b in kept),
                    tuple(range(len(kept))) if to[1] == STRIPED else tuple(b[0] for b in kept))
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
    views(graph)
    # design/problems-of-14b9a757-change-request.md item 8, steps 1 to 3
    invariant = mesh_graph.forward(list(graph.nodes), lambda n: n.all_input_nodes, lambda n, read: (
        "state" in n.meta if n.op == "placeholder" else view(n) is not None and all(read)))
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
        node.meta.pop("collectives", None)
    views(graph)
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
    # design/unequal-shards-change-request.md item 12; MPI 4.1, section 6.1.3
    cuts = mesh_graph.chain(node, lambda n: n.args[0] if n.target is SLICE and (*n.args, 1)[4] == 1 and n.args[
        0].meta.get("processor", here) == here else None)
    subarray = (shape(cuts[-1]), tuple(value.shape), tuple(
        sum(range(shape(cut.args[0])[d])[cut.args[2]:cut.args[3]].start for cut in cuts[:-1]
            if cut.args[1] % value.dim() == d) for d in range(value.dim())))
    for channel, (processor, users) in enumerate(away.items(), first):
        with graph.inserting_after(node):
            sent = graph.call_function(SEND, (node, channel))
            sent.meta["processor"] = here
        with graph.inserting_before(users[0]):
            arrived = graph.call_function(RECV, (channel,))
            arrived.meta.update(processor=processor, val=value)
        # design/aot-compilation-change-request.md A4 and appendix item 5; design/plan-to-bridge-change-request.md R4
        collective = node.meta.get("collectives", {}).get(processor, ("broadcast", node.name))
        if collective[0] == "reduce":
            arrived.meta["consumer"] = PAGES
        for end in (sent, arrived):
            end.meta.update(zip(("sizes", "subsizes", "starts"), subarray), collective=collective)
        if processor in node.meta.get("blocks", ()):
            for end in (sent, arrived):
                end.meta.update(zip(("dimension", "block", "counts", "displs"), node.meta["blocks"][processor]))
        for user in users:
            user.replace_input_with(node, arrived)
    return len(away)

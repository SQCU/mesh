"""Prepare the complete two-product recurrent chain and its physical writes.

The tiled processing stages come from the coherent n-rank chain at 1c0afa53^.
The current collective supplies every physical route and dependency binding.
"""
from collections import Counter
from pathlib import Path

import torch

from . import _stream, argmax, empty, fused, partition
from .backend import Call, WRITE
from .streamed import instance, node


def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


def _floor2(n):
    return 1 << (n.bit_length() - 1)


def _form(K, S, rows, ranks, channels, resident_tiles):
    _, threads, memory = _stream.device_facts()
    source = fused.steps() + argmax.SOURCE + '\n' + Path(__file__).with_suffix('.metal').read_text()
    tile = 8  # simdgroup_half8x8's intrinsic matrix extent
    probe = _stream.pipeline('kernel void width_probe(device uint* x [[buffer(0)]]) { x[0] = 0; }', 'width_probe')
    simd, cores = _stream.pipeline_facts(probe)[1], node()[1]
    kt, st, rt = K // tile, S // tile, rows // tile
    G = min(cores, max(kt, st))
    most = threads // simd

    def simdgroups(rg):
        groups = rt // rg
        return min(most, _floor2(max(-(-st // G) * groups * kt, -(-kt // G) * groups * st)))

    def splits(rg, sg):
        groups = rt // rg
        return _ksplit(-(-st // G) * groups, kt, sg), _ksplit(-(-kt // G) * groups, st, sg)

    SG = simdgroups(1)
    RG = max(rg for rg in (1 << i for i in range(rt.bit_length()))
             if rt % rg == 0 and simdgroups(rg) == SG and splits(rg, SG) == splits(1, SG))
    while True:
        RB = RG
        while SG * RB * tile * tile * 4 + 32 > memory:
            RB //= 2
        args = (tile, K, S, G, RG, RB, SG * simd, simd, ranks, channels)
        name, specialization = instance('chain_step', *args)
        handle = _stream.pipeline(source + '\n' + specialization, name)
        available = _stream.pipeline_facts(handle)[0]
        if available >= SG * simd:
            ks0, ks1 = splits(RG, SG)
            products = []
            # Long dependent matrix chains lose to the existing distributed
            # linear geometry at the wide hourly shapes. This is a compilation
            # operand, not a runtime alternative or a change of transport.
            if max(-(-kt // ks0), -(-st // ks1)) > resident_tiles:
                name, specialization = instance('chain_step', *args, False)
                handle = _stream.pipeline(source + '\n' + specialization, name)
                _stream.linear_source(partition._LINEAR_SOURCE, cores)
                first, second = _stream.linear_split(K, S), _stream.linear_split(S, K)
                name, specialization = instance('linear_rows', 4, 4, 2, first, 2, 1)
                product = _stream.pipeline(partition._LINEAR_SOURCE + '\n' + specialization, name)
                publication = partition._linear_kernel(4, 4, 2, second, 2, channels)
                products = [product, publication, first, second]
            ready_name, ready_line = instance('chain_input', channels)
            ready = _stream.pipeline(source + '\n' + ready_line, ready_name)
            return handle, ready, products, G, SG * simd, [rows, ks0, ks1]
        if RG > 1:
            RG //= 2
        else:
            SG = _floor2(available // simd)


def operands(a, b, inputs, outputs, group, batch, resident_tiles, input_frames):
    """Realize numerical operands and declarations before the sequence runs.

    ``a`` and ``b`` hold this rank's rows of the two weights. Zero-padded tiles
    preserve every assigned column, including shares shorter than one tile.
    Inputs and outputs retain their original shapes and every timestep.
    ``resident_tiles`` bounds a SIMD group's dependent matrix-tile chain before
    preparation selects the shared linear geometry. ``input_frames`` and
    ``batch`` set independent input and recurrent publication intervals.
    """
    steps, rows, width = inputs.shape
    round_tile = lambda n: max(8, -(-n // 8) * 8)
    K, S, R = round_tile(width), round_tile(a.shape[0]), round_tile(rows)

    def weight_pages(weight, shape, tiled):
        padded = empty(shape, dtype=weight.dtype).zero_()
        padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
        if not tiled:
            return padded
        result = empty(shape[1] // 8, shape[0] // 8, 8, 8, dtype=weight.dtype)
        result.copy_(padded.reshape(shape[0] // 8, 8, shape[1] // 8, 8).permute(2, 0, 1, 3))
        return result

    channels = sum(group.rank() in cable['nodes'] for cable in group.fabric.cables)
    kernel, ready, products, workers, threads, words = _form(K, S, R, group.size(), channels, resident_tiles)
    if products:
        At, Bt = weight_pages(a, (S, K), False), weight_pages(b.T, (K, S), False)
    else:
        At, Bt = weight_pages(a.T, (K, S), True), weight_pages(b, (S, K), True)
    X, Y = empty(R, K, dtype=torch.half), empty(R, S, dtype=torch.half)
    stamps = empty(workers, dtype=torch.int32).zero_()
    arrivals = empty(1, dtype=torch.int32).zero_()
    counts = []
    input_calls = [Call(WRITE, rows * width, inputs.dtype, root=0, destination=rank, fused=True)
                   for rank in range(1, group.size())]
    calls = [(input_calls, min(input_frames, steps))]
    first = 0
    while first < steps:
        count = min(batch, steps - first, input_frames - first % input_frames)
        if first % input_frames == 0 and first + input_frames < steps:
            calls.append((input_calls, min(input_frames, steps - first - input_frames)))
        calls.append((partition.summed(X, group), count))
        counts.append(count)
        first += count
    return calls, (_stream.chain_slot(), kernel, ready, products, [At, Bt, inputs, outputs, X, Y, X, stamps, arrivals],
                   [1, 0, *words, 0, width, rows, 0, 0, 0, 0], workers, threads, group.group_name,
                   group.size(), batch, input_frames, Counter(counts))


def prepare(operands):
    """Bind the current physical routes to already allocated numerical pages."""
    index, kernel, ready, products, buffers, words, workers, threads, group, ranks, batch, input_frames, counts = operands
    x = buffers[4]
    for repeats, batches in counts.items():
        _stream.fused_prepare(x, None, 0, [x.shape[0]] * ranks, group, True, repeats, batches)
    _stream.prepare_chain(index, kernel, ready, products, buffers, words, workers, threads, group, ranks, batch, input_frames)
    return index

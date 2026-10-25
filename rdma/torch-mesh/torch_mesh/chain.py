"""Prepare the complete two-product recurrent chain and its physical writes.

The tiled processing stages come from the coherent n-rank chain at 1c0afa53^.
The current collective supplies every physical route and dependency binding.
"""
from collections import Counter
from pathlib import Path

import torch

from . import _stream, argmax, empty, fused, partition
from .streamed import instance, node


def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


def _floor2(n):
    return 1 << (n.bit_length() - 1)


def _form(K, S, rows, ranks, channels):
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
            return handle, G, SG * simd, [rows, *splits(RG, SG)]
        if RG > 1:
            RG //= 2
        else:
            SG = _floor2(available // simd)


def operands(a, b, inputs, outputs, group, batch):
    """Realize numerical operands and declarations before the sequence runs.

    ``a`` and ``b`` hold this rank's rows of the two weights. Zero-padded tiles
    preserve every assigned column, including shares shorter than one tile.
    Inputs and outputs retain their original shapes and every timestep.
    """
    steps, rows, width = inputs.shape
    round_tile = lambda n: max(8, -(-n // 8) * 8)
    K, S, R = round_tile(width), round_tile(a.shape[0]), round_tile(rows)

    def tiled(weight, shape):
        padded = empty(shape, dtype=weight.dtype).zero_()
        padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
        result = empty(shape[1] // 8, shape[0] // 8, 8, 8, dtype=weight.dtype)
        result.copy_(padded.reshape(shape[0] // 8, 8, shape[1] // 8, 8).permute(2, 0, 1, 3))
        return result

    At, Bt = tiled(a.T, (K, S)), tiled(b, (S, K))
    channels = sum(group.rank() in cable['nodes'] for cable in group.fabric.cables)
    kernel, workers, threads, words = _form(K, S, R, group.size(), channels)
    X, Y = empty(R, K, dtype=torch.half), empty(R, S, dtype=torch.half)
    stamps = empty(workers, dtype=torch.int32).zero_()
    arrivals = empty(1, dtype=torch.int32).zero_()
    counts = [min(batch, steps - first) for first in range(0, steps, batch)]
    calls = [(partition.summed(X, group), count) for count in counts]
    return calls, (_stream.chain_slot(), kernel, [At, Bt, inputs, outputs, X, Y, X, stamps, arrivals],
                   [1, 0, *words, 0, width, rows, 0, 0], workers, threads, group.group_name,
                   group.size(), batch, Counter(counts))


def prepare(operands):
    """Bind the current physical routes to already allocated numerical pages."""
    index, kernel, buffers, words, workers, threads, group, ranks, batch, counts = operands
    x = buffers[4]
    for repeats, batches in counts.items():
        _stream.fused_prepare(x, None, 0, [x.shape[0]] * ranks, group, True, repeats, batches)
    _stream.prepare_chain(index, kernel, buffers, words, workers, threads, group, ranks, batch)
    return index

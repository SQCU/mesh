"""Prepare the complete two-product recurrent chain and its physical writes.

The tiled processing stages come from the coherent n-rank chain at 1c0afa53^.
The current collective supplies every physical route and dependency binding.
"""
from collections import Counter
from pathlib import Path

import torch

from . import _stream, argmax, empty, fused, partition
from .backend import Call, WRITE
from .streamed import instance


def _ksplit(items, ktiles, simdgroups):
    k = 1
    while 2 * k * items <= simdgroups and 2 * k <= ktiles:
        k *= 2
    return k


def _floor2(n):
    return 1 << (n.bit_length() - 1)


def _form(K, S, rows, ranks, channels, resident_tiles, capabilities):
    threads, memory = (capabilities[k] for k in ('max_threads_per_threadgroup', 'threadgroup_memory_bytes'))
    source = fused.steps() + argmax.SOURCE + '\n' + Path(__file__).with_suffix('.metal').read_text()
    tile = capabilities['simdgroup_matrix_edge']
    simd, cores = _stream.compilation['SIMD_WIDTH'], capabilities['hardware']['gpu']['cores']
    kt, st, rt = K // tile, S // tile, rows // tile
    # Independent row tiles and columns share the observed cores. These groups
    # may run in waves: no numerical phase requires them to reside together.
    G = min(cores, max(kt, st) * rt)
    most = threads // simd

    def simdgroups(rg):
        groups = rt // rg
        return min(most, _floor2(max(-(-st * groups // G) * kt, -(-kt * groups // G) * st)))

    def splits(rg, sg):
        groups = rt // rg
        return _ksplit(-(-st * groups // G), kt, sg), _ksplit(-(-kt * groups // G), st, sg)

    SG = simdgroups(1)
    RG = max(rg for rg in (1 << i for i in range(rt.bit_length()))
             if rt % rg == 0 and max(kt, st) * (rt // rg) >= G
             and simdgroups(rg) == SG and splits(rg, SG) == splits(1, SG))
    while True:
        RB = RG
        while RB and SG * RB * tile * tile * 4 + 2 * capabilities['threadgroup_memory_alignment_bytes'] > memory:
            RB //= 2
        if not RB:
            raise ValueError(f'chain: {SG} SIMD groups do not fit the manifest\'s {memory} bytes of threadgroup memory')
        args = (tile, K, S, rows, G, RG, RB, SG * simd, simd, ranks, channels)
        ks0, ks1 = splits(RG, SG)
        linear = max(-(-kt // ks0), -(-st // ks1)) > resident_tiles
        phases = []
        for phase in ((0,) if linear else (0, 1, 2)):
            name, specialization = instance('chain_step', *args, phase)
            phases.append(_stream.pipeline(source + '\n' + specialization, name))
        available = min(_stream.pipeline_facts(handle)[0] for handle in phases)
        if available >= SG * simd:
            products = phases[1:]
            # Long dependent matrix chains lose to the existing distributed
            # linear geometry at the wide hourly shapes. This is a compilation
            # operand, not a runtime alternative or a change of transport.
            if linear:
                _stream.linear_source(partition._LINEAR_SOURCE, cores)
                first, second = _stream.linear_split(K, S, rows), _stream.linear_split(S, K, rows)
                geometry = _stream.compilation
                vector, columns, loads = (geometry[k] for k in ('LINEAR_VECTOR', 'LINEAR_COLUMNS', 'LINEAR_LOADS'))
                name, specialization = instance('linear_rows', vector, vector, columns, first, loads, 1)
                product = _stream.pipeline(partition._LINEAR_SOURCE + '\n' + specialization, name)
                publication = partition._linear_kernel(vector, vector, columns, second, loads, channels)
                products = [product, publication, first, second]
            ready_name, ready_line = instance('chain_publish_input', channels)
            ready = _stream.pipeline(source + '\n' + ready_line, ready_name)
            return phases[0], ready, products, linear, G, SG * simd, [rows, ks0, ks1]
        if RG > 1:
            RG //= 2
        else:
            if available < simd:
                raise RuntimeError(f'chain: pipeline holds {available} threads, fewer than one {simd}-thread SIMD group')
            SG = _floor2(available // simd)


def operands(a, b, inputs, outputs, group, batch, resident_tiles, input_frames, capabilities):
    """Realize numerical operands and declarations before the sequence runs.

    ``a`` and ``b`` hold this rank's rows of the two weights. Zero-padded tiles
    preserve every assigned column, including shares shorter than one tile.
    Inputs and outputs retain their original shapes and every timestep.
    ``resident_tiles`` bounds a SIMD group's dependent matrix-tile chain before
    preparation selects the shared linear geometry. ``input_frames`` and
    ``batch`` set independent input and recurrent publication intervals.
    """
    steps, rows, width = inputs.shape
    tile = capabilities['simdgroup_matrix_edge']
    round_tile = lambda n: max(tile, -(-n // tile) * tile)
    K, S, R = round_tile(width), round_tile(a.shape[0]), round_tile(rows)

    def weight_pages(weight, shape, tiled):
        padded = empty(shape, dtype=weight.dtype).zero_()
        padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
        if not tiled:
            return padded
        result = empty(shape[1] // tile, shape[0] // tile, tile, tile, dtype=weight.dtype)
        result.copy_(padded.reshape(shape[0] // tile, tile, shape[1] // tile, tile).permute(2, 0, 1, 3))
        return result

    channels = sum(group.rank() in cable['nodes'] for cable in group.fabric.cables)
    kernel, ready, products, linear, workers, threads, words = _form(K, S, R, group.size(), channels, resident_tiles, capabilities)
    if linear:
        At, Bt = weight_pages(a, (S, K), False), weight_pages(b.T, (K, S), False)
    else:
        At, Bt = weight_pages(a.T, (K, S), True), weight_pages(b, (S, K), True)
    X, Y = empty(R, K, dtype=torch.half), empty(R, S, dtype=torch.half)
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
    return calls, (_stream.chain_slot(), kernel, ready, products, linear, [At, Bt, inputs, outputs, X, Y, X, arrivals],
                   [1, 0, *words, 0, width, rows, 0, 0], workers, threads, group.group_name,
                   group.size(), batch, input_frames, Counter(counts))


def prepare(operands):
    """Bind the current physical routes to already allocated numerical pages."""
    index, kernel, ready, products, linear, buffers, words, workers, threads, group, ranks, batch, input_frames, counts = operands
    x = buffers[4]
    for repeats, batches in counts.items():
        _stream.fused_prepare(x, None, 0, [x.shape[0]] * ranks, group, True, repeats, batches)
    _stream.prepare_chain(index, kernel, ready, products, linear, buffers, words, workers, threads, group, ranks, batch, input_frames)
    return index

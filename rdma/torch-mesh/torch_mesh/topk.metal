// Tensor views and geometry are compilation operands. Selection shares the
// native sampling/routing radix core; only its value order differs.
inline uint top_order(INPUT x) {
    uint u = as_type<uint>(float(x));
    if ((u & 0x7fffffffu) > 0x7f800000u) return LARGEST ? 0xffffffffu : 1u;
    if ((u & 0x7fffffffu) == 0u) u = 0u;
    const uint o = (u & 0x80000000u) ? ~u : u | 0x80000000u;
    return LARGEST ? o : ~o;
}

// One bitonic step. The same pairs work in threadgroup or device storage.
template <typename A, typename K>
inline void top_pairs(A ids, K key, uint lid, uint width, uint stride) {
    for (uint pair = lid; pair < PAD / 2; pair += THREADS) {
        const uint a = pair / stride * (2 * stride) + pair % stride, b = a + stride;
        const uint ia = ids[a], ib = ids[b];
        const uint va = ia == 0xffffffffu ? 0u : key(ia), vb = ib == 0xffffffffu ? 0u : key(ib);
        const bool ahead = va > vb || (va == vb && ia < ib);
        if (ahead != ((a & width) == 0)) { ids[a] = ib; ids[b] = ia; }
    }
}

kernel void select_top(device const INPUT *source [[buffer(0)]], device INPUT *values [[buffer(1)]],
                       device long *indices [[buffer(2)]], device uint *scratch [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint words[mesh_selection_words<THREADS>];
    threadgroup uint chosen[LOCAL ? PAD : 1];
    const ulong base = ulong(row) * EXTENT, out = ulong(row) * COUNT;
    auto key = [&](uint c) { return top_order(source[mesh_index<0>(base + c)]); };
    if constexpr (LOCAL) {
        for (uint i = lid; i < PAD; i += THREADS) chosen[i] = 0xffffffffu;
        threadgroup uint *held = chosen;
        mesh_top_ordered<THREADS>(key, 0, EXTENT, COUNT, lid, words, [&](uint i, uint c) { held[i] = c; });
        for (uint width = 2; width <= PAD; width *= 2) for (uint stride = width / 2; stride; stride /= 2) {
            top_pairs(held, key, lid, width, stride);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = lid; i < COUNT; i += THREADS) {
            const uint c = chosen[i];
            const ulong value = mesh_index<8>(out + i), index = mesh_index<9>(out + i);
            if (!VALUE_SELECTED || value < VALUE_COUNT) values[mesh_index<1>(value)] = source[mesh_index<0>(base + c)];
            if (!INDEX_SELECTED || index < INDEX_COUNT) indices[mesh_index<2>(index)] = long(c);
        }
    } else if constexpr (SORTED) {
        device uint *held = scratch + ulong(row) * PAD;
        for (uint i = COUNT + lid; i < PAD; i += THREADS) held[i] = 0xffffffffu;
        mesh_top_ordered<THREADS>(key, 0, EXTENT, COUNT, lid, words, [&](uint i, uint c) { held[i] = c; });
    } else {
        mesh_top_ordered<THREADS>(key, 0, EXTENT, COUNT, lid, words, [&](uint i, uint c) {
            const ulong value = mesh_index<8>(out + i), index = mesh_index<9>(out + i);
            if (!VALUE_SELECTED || value < VALUE_COUNT) values[mesh_index<1>(value)] = source[mesh_index<0>(base + c)];
            if (!INDEX_SELECTED || index < INDEX_COUNT) indices[mesh_index<2>(index)] = long(c);
        });
    }
}

kernel void order_top(device const INPUT *source [[buffer(0)]], device uint *scratch [[buffer(1)]],
                      constant uint *step [[buffer(2)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    const ulong base = ulong(row) * EXTENT;
    top_pairs(scratch + ulong(row) * PAD,
              [&](uint c) { return top_order(source[mesh_index<0>(base + c)]); }, lid, step[0], step[1]);
}

kernel void write_top(device const INPUT *source [[buffer(0)]], device INPUT *values [[buffer(1)]],
                      device long *indices [[buffer(2)]], device const uint *scratch [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    const ulong base = ulong(row) * EXTENT, out = ulong(row) * COUNT;
    for (uint i = lid; i < COUNT; i += THREADS) {
        const uint c = scratch[ulong(row) * PAD + i];
        const ulong value = mesh_index<8>(out + i), index = mesh_index<9>(out + i);
        if (!VALUE_SELECTED || value < VALUE_COUNT) values[mesh_index<1>(value)] = source[mesh_index<0>(base + c)];
        if (!INDEX_SELECTED || index < INDEX_COUNT) indices[mesh_index<2>(index)] = long(c);
    }
}

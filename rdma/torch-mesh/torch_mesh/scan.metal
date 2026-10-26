// Shapes and the hierarchy are compiled before execution. Each level writes
// disjoint blocks; its fixed successor adds the preceding block's prefix.
kernel void scan_blocks(device const INPUT *source [[buffer(0)]], device OUTPUT *output [[buffer(1)]],
                        device ACCUM *totals [[buffer(2)]], device ACCUM *partial [[buffer(3)]],
                        uint2 block [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup ACCUM sums[THREADS / MESH_SIMD_WIDTH];
    const ulong column = ulong(block.x) * THREADS + lid;
    const ulong at = output_base(block.y) + column * OUTPUT_STRIDE;
    const ACCUM value = column < EXTENT ? ACCUM(OUTPUT(source[source_base(block.y) + column * INPUT_STRIDE])) : ACCUM(0);
    ACCUM total;
    const ACCUM prefix = mesh_prefix_sum<THREADS>(value, lid, sums, total);
    if constexpr (BLOCKS > 1) {
        if (column < EXTENT) partial[at] = prefix;
        if (lid == 0) totals[ulong(block.y) * BLOCKS + block.x] = total;
    } else if (column < EXTENT) output[at] = OUTPUT(prefix);
}

kernel void add_prefix(device OUTPUT *output [[buffer(0)]], device const ACCUM *prefix [[buffer(1)]],
                       device const ACCUM *partial [[buffer(2)]],
                       uint2 block [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    const ulong column = ulong(block.x) * THREADS + lid;
    if (column < EXTENT) {
        const ulong at = output_base(block.y) + column * OUTPUT_STRIDE;
        const ACCUM before = block.x ? prefix[ulong(block.y) * BLOCKS + block.x - 1] : ACCUM(0);
        output[at] = OUTPUT(before + partial[at]);
    }
}

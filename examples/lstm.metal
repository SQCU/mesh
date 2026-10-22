// docs/writing-a-service.md#the-lstm
struct lstm_segment { uint source, offset, ld, first, tiles; };

// docs/writing-a-service.md#the-lstm
template <uint SPAN, uint THREADS>
struct Segments {
    static constant constexpr const uint span = SPAN, ld = (SPAN + 1) * 8;
    device uchar* e; constant mesh_crossing* crossing;
    constant lstm_segment* seg; uint count, zero_h, rows, tid;
    threadgroup half* band; uint chunk;
    uint find(uint kt) const {
        uint k = 0;
        while (k + 1 < count && kt >= seg[k + 1].first) k++;
        return k;
    }
    device uchar* base(constant lstm_segment& s) const {
        return s.source == 0 ? e : zero_h ? nullptr : s.source == 1 ? mesh_own<half>(*crossing).at : mesh_landing<half>(*crossing, s.source - 2).at;
    }
    half element(uint row, uint col) const {
        constant lstm_segment& s = seg[find(col / 8)];
        return mesh_view<half>{base(s) + 2 * s.offset}[row * s.ld + col - s.first * 8];
    }
    void fill(uint c0, uint n) thread {
        if (c0 == chunk) return;
        chunk = c0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < count; k++) {
            constant lstm_segment& s = seg[k];
            const uint lo = max(c0, s.first), hi = min(c0 + n, s.first + s.tiles), quads = hi > lo ? (hi - lo) * 2 : 0;
            device uchar* b = base(s);
            for (uint q = tid; q < rows * quads; q += THREADS) {
                const uint row = q / quads, col = (lo - s.first) * 8 + q % quads * 4;
                *(threadgroup half4*)(band + row * ld + (lo - c0) * 8 + q % quads * 4) =
                    b ? mesh_view<half4>{b + 2 * s.offset}[(row * s.ld + col) / 4] : half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    void load(thread simdgroup_half8x8& a, uint kt, uint row) const {
        simdgroup_load(a, band + row * ld + (kt - chunk) * 8, ld);
    }
};

// docs/writing-a-service.md#the-lstm
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// docs/writing-a-service.md#the-lstm
struct Cell {
    device float* C; mesh_view<half> S; device const float* bias; uint l, rows, hr, ct0, at;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        const uint fn = at % 8, row = rt * 8 + at / 8, t = ct0 + ct;
        const float2 v = float2(acc.thread_elements()[0] + bias[t * 8 + fn], acc.thread_elements()[1] + bias[t * 8 + fn + 1]);
        const float2 w = simd_shuffle_xor(v, 1);
        if (fn % 4 == 0) {
            const uint k = (l * rows + row) * hr + t * 2 + fn / 4;
            const float c = sigmoid(v.y) * C[k] + sigmoid(v.x) * tanh(w.x);
            C[k] = c;
            S.store(k, half(sigmoid(w.y) * tanh(c)));
        }
    }
};

// docs/writing-a-service.md#the-lstm
template <uint L, uint NSEG>
struct lstm_words {
    uint first, last, rows, zeroed, active, outs;
    uint ks[L + L % 2], counts[L + L % 2];
    lstm_segment seg[L][NSEG];
};

// docs/writing-a-service.md#the-lstm
template <uint TILE, uint KIN, uint H, uint HR, uint L, uint G, uint RG, uint RB, uint THREADS, uint SIMD, uint PEERS, uint NSEG,
          uint SPAN, uint BAND>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void lstm_tick(device const half* W [[buffer(0)]], device const float* bias [[buffer(1)]], device const uchar* E [[buffer(2)]],
                      device half* out [[buffer(3)]], device float* C [[buffer(4)]],
                      constant mesh_crossing& crossing [[buffer(14)]], constant lstm_words<L, NSEG>& words [[buffer(15)]],
                      uint tid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint w [[threadgroup_position_in_grid]]) {
    static_assert(TILE == 8 && HR % 8 == 0 && H % 8 == 0 && KIN % 8 == 0, "8 x 8 tiles; shares of whole tiles");
    static_assert(BAND % 4 == 0, "a band of half4 words");
    constexpr uint SG = THREADS / SIMD, NT = HR / 2;
    threadgroup float red[SG * RB * 64];
    threadgroup half4 band[BAND / 4];
    const bool first = words.first != 0;
    mesh_enter(crossing, tid);
    const uint R = words.rows, groups = R / (TILE * RG), at = lane_element(lane);
    if (words.outs) {
        const Segments<SPAN, THREADS> top{(device uchar*)E, &crossing, words.seg[L - 1], words.counts[L - 1], 0u, R, tid,
                                          (threadgroup half*)band, ~0u};
        const uint base = L == 1 ? KIN : H;
        const uint2 cols = share<H / TILE, G>(w);
        for (uint q = tid; q < R * cols.y * TILE; q += THREADS) {
            const uint row = q / (cols.y * TILE), col = cols.x * TILE + q % (cols.y * TILE);
            out[row * H + col] = top.element(row, base + col);
        }
    }
    if (words.last) {
        mesh_leave(crossing, tid);
        return;
    }
    const mesh_view<half> S = mesh_send<half>(crossing);
    const uint2 cts = share<NT, G>(w);
    for (uint l = 0; l < L; l++) {
        if (words.zeroed >> l & 1) {
            for (uint q = tid; q < R * cts.y * 2; q += THREADS)
                S.store((l * R + q / (cts.y * 2)) * HR + cts.x * 2 + q % (cts.y * 2), 0.0h);
            continue;
        }
        if (!(words.active >> l & 1)) continue;
        Segments<SPAN, THREADS> src{(device uchar*)E, &crossing, words.seg[l], words.counts[l], first ? 1u : 0u, R, tid,
                                    (threadgroup half*)band, ~0u};
        Cell cell{C, S, bias + l * 4 * HR, l, R, HR, cts.x, at};
        if (l == 0) stage<RG, RB, KIN + H, SG>(src, W, cts, groups, words.ks[0], sgid, red, cell);
        else stage<RG, RB, 2 * H, SG>(src, W + (KIN + H) * 4 * HR + (l - 1) * 2 * H * 4 * HR, cts, groups, words.ks[l], sgid, red, cell);
    }
    mesh_leave(crossing, tid);
}

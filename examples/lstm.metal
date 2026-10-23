// docs/writing-a-service.md#the-lstm
struct lstm_segment { uint source, offset, ld, first, tiles; };

// docs/writing-a-service.md#the-lstm
struct Segments {
    device const uchar* e; device const uchar* prev; device const uchar* landing;
    constant lstm_segment* seg; uint count, at, zero_h;
    struct Tile {
        device const uchar* p; uint ld, at;
        half2 load(uint row) const { return p ? mesh_step_read<half2>(p, (row + at / 8) * ld / 2) : half2(0.0h); }
    };
    uint find(uint kt) const {
        uint k = 0;
        while (k + 1 < count && kt >= seg[k + 1].first) k++;
        return k;
    }
    device const uchar* base(constant lstm_segment& s) const {
        return s.source == 0 ? e : zero_h ? nullptr : s.source == 1 ? prev : landing;
    }
    Tile tile(uint kt) const {
        constant lstm_segment& s = seg[find(kt)];
        device const uchar* b = base(s);
        return Tile{b ? b + 2 * (s.offset + (kt - s.first) * 8 + at % 8) : nullptr, s.ld, at};
    }
    half element(uint row, uint col) const {
        constant lstm_segment& s = seg[find(col / 8)];
        return mesh_step_read<half>(base(s) + 2 * s.offset, row * s.ld + col - s.first * 8);
    }
};

// docs/writing-a-service.md#the-lstm
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// docs/writing-a-service.md#the-lstm
struct Cell {
    device float* C; device uchar* S; device const float* bias; uint l, rows, hr, ct0, at;
    void operator()(thread simdgroup_float8x8& acc, uint ct, uint rt) thread {
        const uint fn = at % 8, row = rt * 8 + at / 8, t = ct0 + ct;
        const float2 v = float2(acc.thread_elements()[0] + bias[t * 8 + fn], acc.thread_elements()[1] + bias[t * 8 + fn + 1]);
        const float2 w = simd_shuffle_xor(v, 1);
        if (fn % 4 == 0) {
            const uint k = (l * rows + row) * hr + t * 2 + fn / 4;
            const float c = sigmoid(v.y) * C[k] + sigmoid(v.x) * tanh(w.x);
            C[k] = c;
            mesh_step_write<half>(S, k, half(sigmoid(w.y) * tanh(c)));
        }
    }
};

// docs/writing-a-service.md#the-lstm
template <uint PEERS, uint L, uint NSEG>
struct lstm_words {
    uint first, last, drains, rows, zeroed, active, outs, pad;
    uint ks[L + L % 2], counts[L + L % 2];
    lstm_segment seg[L][NSEG];
    mesh_step_binding land[PEERS], publish[PEERS];
};

// docs/writing-a-service.md#the-lstm
template <uint TILE, uint KIN, uint H, uint HR, uint L, uint G, uint RG, uint RB, uint THREADS, uint SIMD, uint PEERS, uint NSEG>
[[max_total_threads_per_threadgroup(THREADS)]]
kernel void lstm_tick(device const half* W [[buffer(0)]], device const float* bias [[buffer(1)]], device const uchar* E [[buffer(2)]],
                      device half* out [[buffer(3)]], device float* C [[buffer(4)]], device const uchar* prev [[buffer(5)]],
                      device uchar* S [[buffer(6)]], device atomic_uint* arrivals [[buffer(7)]], device const uchar* landing [[buffer(8)]],
                      constant lstm_words<PEERS, L, NSEG>& words [[buffer(15)]],
                      uint tid [[thread_index_in_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint w [[threadgroup_position_in_grid]]) {
    static_assert(TILE == 8 && HR % 8 == 0 && H % 8 == 0 && KIN % 8 == 0, "8 x 8 tiles; shares of whole tiles");
    constexpr uint SG = THREADS / SIMD, NT = HR / 2;
    threadgroup float red[SG * RB * 64];
    threadgroup uint last;
    const bool first = words.first != 0;
    if (!first) {
        for (uint q = 0; q < PEERS; q++) {
            constant mesh_step_binding& l = words.land[q];
            mesh_step_enter(*l.p, l.v, l.progress, w == 0 && tid == 0);
            mesh_step_await(*l.p, l.a, l.v, tid);
        }
    } else if (w == 0) {
        mesh_arrival_reset(arrivals, 0, tid);
    }
    const uint R = words.rows, groups = R / (TILE * RG), at = lane_element(lane);
    if (words.outs) {
        const Segments top{E, prev, landing, words.seg[L - 1], words.counts[L - 1], at, 0u};
        const uint base = L == 1 ? KIN : H;
        const uint2 cols = share<H / TILE, G>(w);
        for (uint q = tid; q < R * cols.y * TILE; q += THREADS) {
            const uint row = q / (cols.y * TILE), col = cols.x * TILE + q % (cols.y * TILE);
            out[row * H + col] = top.element(row, base + col);
        }
    }
    if (words.last) return;
    const uint2 cts = share<NT, G>(w);
    for (uint l = 0; l < L; l++) {
        if (words.zeroed >> l & 1) {
            for (uint q = tid; q < R * cts.y * 2; q += THREADS)
                mesh_step_write<half>(S, (l * R + q / (cts.y * 2)) * HR + cts.x * 2 + q % (cts.y * 2), 0.0h);
            continue;
        }
        if (!(words.active >> l & 1)) continue;
        const Segments src{E, prev, landing, words.seg[l], words.counts[l], at, first ? 1u : 0u};
        Cell cell{C, S, bias + l * 4 * HR, l, R, HR, cts.x, at};
        if (l == 0) stage<RG, RB, KIN + H, SG>(src, W, cts, groups, words.ks[0], sgid, red, cell);
        else stage<RG, RB, 2 * H, SG>(src, W + (KIN + H) * 4 * HR + (l - 1) * 2 * H * 4 * HR, cts, groups, words.ks[l], sgid, red, cell);
    }
    for (uint q = 0; q < PEERS; q++) mesh_step_enter(*words.publish[q].p, words.publish[q].v, words.publish[q].progress, w == 0 && tid == 0);
    threadgroup_barrier(mem_flags::mem_device);
    if (mesh_last_arrival<true>(arrivals, 0, G, tid, &last)) {
        for (uint q = 0; q < PEERS; q++) mesh_step_release(*words.publish[q].p, words.publish[q].a, tid);
        mesh_arrival_reset(arrivals, 0, tid);
    }
}

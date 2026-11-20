// The routing policy's training rollout (design/switched-multipath-dma.md), fused: one threadgroup a rollout. v0: the
// forward pass, all-gather transactions, fp32.
//
// A rollout is one scenario's transaction in the fluid model, SLOTS slots long, each slot `dt` ns. Each slot:
// - at a refresh (every REFRESH slots), the policy's weighting of each commodity's K candidate trees;
// - each source releases its commodity's bytes into its trees at its out-lanes' rate, split by the weighting;
// - every lane serves the bytes waiting on it (its tree edges' and its background's) by processor sharing at its rate;
// - bytes sent reach the lane's head after its latency and one page time (a ring of the last H slots' sends, each
//   as fp16 shares of its commodity's demand, in threadgroup memory with the rest of the rollout's state);
// - each (commodity, destination) adds the area above its delivered fraction.
// The objective is a smooth maximum over (commodity, destination) of that area over the floor; `done` is the first slot
// every destination holds every byte, over the floor.
//
// The policy, a commodity at a time: a 64-wide stream from the commodity's context and its last weighting's expected
// candidate embedding, through two layers of a GRU block (recurrent over refreshes) and a feed-forward block expanding
// to 192; each candidate tree is its N node tokens (parent, in-lane queue level, rate band), embedded, rotated by node
// position (RoPE at the embedding only), mean-pooled, and dotted with the stream. A matrix-vector product reads each
// weight row once, coalesced along the row across a SIMD group's lanes, for every commodity at once.
#include <metal_stdlib>
#ifndef NOPOLICY
#define NOPOLICY 0
#endif
using namespace metal;

constant constexpr int N = 6, L = 32, J = 6, K = 8, H = 12, SLOTS = 64, REFRESH = 8;
constant constexpr int D = 64, E = 192, CTX = 12, VOCAB = (N + 2) * 64, T = 256, SG = T / 32;
constant constexpr int JK = J * K, JKN = J * K * N;
constant constexpr float PAGE = 16384.0f;

struct Scenario {
  float rate[L], lat[L], bg[L];      // bytes a ns; ns; background bytes queued at the start
  uchar lsrc[L], ldst[L];
  uchar lane[J][K][N];               // each candidate tree's in-lane at each node, 255 for none
  uchar src[J], dest[J];             // the commodity's source, and its destinations as a node mask
  float demand[J];
  float floor_ns, dt, size;          // the floor, the slot width, a rank's bytes
  uint commodities, lanes, nodes, pad;
  ushort lane_start[L + 1];          // each lane's tree edges (an edge is (commodity, candidate, head node)),
  ushort lane_edges[JKN];            // lane_edges[lane_start[l] .. lane_start[l + 1] - 1]
};

struct Layer { float z[D][2 * D], r[D][2 * D], c[D][2 * D], bz[D], br[D], bc[D], up[E][D], bu[E], down[D][E], bd[D]; };
struct Net { float inp[D][CTX + D], binp[D]; Layer layer[2]; float tok[VOCAB][D]; };

static inline float sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }

// out[j][row] = (add ? out[j][row] : bias[row] or 0) + sum over q < cols of W[row][q] in[j][q], for every row and every
// commodity j < nj; W's rows `stride` apart. SIMD group sg takes rows sg, sg + SG, ...; its lanes stride along the row.
static void product(device const float *W, int stride, int rows, int cols, threadgroup const float *in, int istride,
                    threadgroup float *out, int ostride, device const float *bias, bool add, int nj, uint sg, uint ln) {
  for (int row = sg; row < rows; row += SG) {
    float acc[J] = {0, 0, 0, 0, 0, 0};
    for (int q = ln; q < cols; q += 32) {
      const float w = W[row * stride + q];
      for (int j = 0; j < J; j++) acc[j] += w * in[j * istride + q];
    }
    for (int j = 0; j < J; j++) {
      const float v = simd_sum(acc[j]);
      if (ln == 0 && j < nj) out[j * ostride + row] = (add ? out[j * ostride + row] : (bias ? bias[row] : 0.0f)) + v;
    }
  }
}

kernel void rollout(device const Scenario *scen [[buffer(0)]], device const Net &net [[buffer(1)]],
                    device const uint *which [[buffer(2)]], device float *objective [[buffer(3)]],
                    device float *done_out [[buffer(4)]],
                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
  // the scenario, copied once into threadgroup memory: every slot reads it, and a device read in a dependent chain
  // costs hundreds of nanoseconds
  threadgroup Scenario s;
  {
    device const uint *from = (device const uint *)&scen[which[r]];
    threadgroup uint *to = (threadgroup uint *)&s;
    for (uint i = tid; i < sizeof(Scenario) / 4; i += T) to[i] = from[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float arrived[JKN], sent[JKN];
  threadgroup half ring[H][JKN];   // the last H slots' sends
  threadgroup float share[L], bgq[L], cap[L], want[L];
  threadgroup int lag[L];
  threadgroup float assigned[JK], released[J], outcap[J], p[JK], area[J * N], frac[J * N], ctx[J][CTX];
  threadgroup float x[J][D], h[2][J][D], gz[J][D], gr[J][D], ex[J][E], last[J][D];
  threadgroup half emb[JK][D];
  threadgroup int finished;
  const float dt = s.dt, fl = s.floor_ns;
  const int nl = int(s.lanes), nj = int(s.commodities), nn = int(s.nodes);

  for (int e = tid; e < JKN; e += T) { arrived[e] = 0; sent[e] = 0; for (int q = 0; q < H; q++) ring[q][e] = 0; }
  for (int i = tid; i < J * D; i += T) { h[0][i / D][i % D] = 0; h[1][i / D][i % D] = 0; last[i / D][i % D] = 0; }
  if (tid < uint(L)) {
    const int l = tid;
    cap[l] = l < nl ? s.rate[l] * dt : 0; bgq[l] = l < nl ? s.bg[l] : 0; want[l] = bgq[l];
    lag[l] = l < nl ? min(int(ceil((s.lat[l] + PAGE / max(s.rate[l], 1e-6f)) / dt)), H - 1) : 1;
  }
  if (tid < uint(J)) {
    const int j = tid; float o = 0;
    for (int l = 0; l < nl; l++) if (s.lsrc[l] == s.src[j]) o += s.rate[l] * dt;
    outcap[j] = o; released[j] = 0;
    for (int k = 0; k < K; k++) { assigned[j * K + k] = 0; p[j * K + k] = 1.0f / K; }
    for (int v = 0; v < N; v++) { area[j * N + v] = 0; frac[j * N + v] = 0; }
  }
  if (tid == 0) finished = SLOTS;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int t = 0; t < SLOTS; t++) {
    if (!NOPOLICY && t % REFRESH == 0) {
      // the stream: context and last expected embedding, then two layers of GRU and feed-forward
      if (tid < uint(J)) {
        const int j = tid;
        ctx[j][0] = log(1.0f + s.size / PAGE); ctx[j][1] = float(t) / SLOTS; ctx[j][2] = released[j] / max(s.demand[j], 1.0f);
        ctx[j][3] = outcap[j] / max(s.demand[j], 1.0f); ctx[j][4] = float(nn) / 8; ctx[j][5] = float(nl) / 32; ctx[j][6] = float(K) / 8;
        ctx[j][7] = float(popcount(uint(s.dest[j]))) / 8; ctx[j][8] = 0; ctx[j][9] = 0; ctx[j][10] = 0; ctx[j][11] = 0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      product(&net.inp[0][0], CTX + D, D, CTX, &ctx[0][0], CTX, &x[0][0], D, net.binp, false, nj, sg, ln);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      product(&net.inp[0][CTX], CTX + D, D, D, &last[0][0], D, &x[0][0], D, nullptr, true, nj, sg, ln);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int layer = 0; layer < 2; layer++) {
        device const Layer &w = net.layer[layer];
        product(&w.z[0][0], 2 * D, D, D, &h[layer][0][0], D, &gz[0][0], D, w.bz, false, nj, sg, ln);
        product(&w.r[0][0], 2 * D, D, D, &h[layer][0][0], D, &gr[0][0], D, w.br, false, nj, sg, ln);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        product(&w.z[0][D], 2 * D, D, D, &x[0][0], D, &gz[0][0], D, nullptr, true, nj, sg, ln);
        product(&w.r[0][D], 2 * D, D, D, &x[0][0], D, &gr[0][0], D, nullptr, true, nj, sg, ln);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) { gz[i / D][i % D] = sigmoid(gz[i / D][i % D]); gr[i / D][i % D] = sigmoid(gr[i / D][i % D]) * h[layer][i / D][i % D]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        product(&w.c[0][0], 2 * D, D, D, &gr[0][0], D, &ex[0][0], E, w.bc, false, nj, sg, ln);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        product(&w.c[0][D], 2 * D, D, D, &x[0][0], D, &ex[0][0], E, nullptr, true, nj, sg, ln);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) {
          const int j = i / D, d = i % D;
          const float hn = (1 - gz[j][d]) * h[layer][j][d] + gz[j][d] * tanh(ex[j][d]);
          h[layer][j][d] = hn; x[j][d] += hn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        product(&w.up[0][0], D, E, D, &x[0][0], D, &ex[0][0], E, w.bu, false, nj, sg, ln);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * E; i += T) ex[i / E][i % E] = max(ex[i / E][i % E], 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        product(&w.down[0][0], E, D, E, &ex[0][0], E, &x[0][0], D, nullptr, true, nj, sg, ln);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) x[i / D][i % D] += w.bd[i % D];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // each candidate: its node tokens, embedded, rotated by node position, pooled
      for (int i = tid; i < JK * D / 2; i += T) {
        const int jk = i / (D / 2), pair = i % (D / 2), j = jk / K, k = jk % K;
        const float theta = pow(10000.0f, -2.0f * pair / D);
        float a = 0, b = 0;
        for (int v = 0; v < N; v++) {
          const int lane = s.lane[j][k][v];
          const int parent = lane == 255 ? (v == s.src[j] ? N : N + 1) : s.lsrc[lane];
          int level = 0, band = 0;
          if (lane != 255) {
            level = clamp(int(floor(log2(1 + 64 * want[lane] / max(s.rate[lane], 1e-6f) / fl))), 0, 15);
            band = clamp(int(floor(log2(max(s.rate[lane], 1.0f)))) - 2, 0, 3);
          }
          const int token = parent * 64 + level * 4 + band;
          const float u = net.tok[token][2 * pair], w2 = net.tok[token][2 * pair + 1], c = cos(v * theta), sn = sin(v * theta);
          a += u * c - w2 * sn; b += u * sn + w2 * c;
        }
        emb[jk][2 * pair] = half(a / N); emb[jk][2 * pair + 1] = half(b / N);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // each candidate's logit: its embedding dotted with the stream, a SIMD group's lanes along the embedding
      for (int jk = sg; jk < JK; jk += SG) {
        const int j = jk / K;
        const float v = simd_sum(float(emb[jk][ln]) * x[j][ln] + float(emb[jk][ln + 32]) * x[j][ln + 32]);
        if (ln == 0) p[jk] = v / 8;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < uint(J)) {
        const int j = tid; float most = -INFINITY, sum = 0;
        for (int k = 0; k < K; k++) most = max(most, p[j * K + k]);
        for (int k = 0; k < K; k++) { p[j * K + k] = exp(p[j * K + k] - most); sum += p[j * K + k]; }
        for (int k = 0; k < K; k++) p[j * K + k] /= sum;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = tid; i < J * D; i += T) {
        const int j = i / D, d = i % D; float a = 0;
        for (int k = 0; k < K; k++) a += p[j * K + k] * float(emb[j * K + k][d]);
        last[j][d] = a;
      }
    }

    // release at the sources, before any lane's share is taken
    if (tid < uint(J) && int(tid) < nj) {
      const int j = tid;
      const float rr = min(max(s.demand[j] - released[j], 0.0f), outcap[j]);
      released[j] += rr;
      for (int k = 0; k < K; k++) { assigned[j * K + k] += rr * p[j * K + k]; arrived[(j * K + k) * N + s.src[j]] = assigned[j * K + k]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // each lane's share: its demand (its edges' bytes not yet sent on, and its background) against its rate
    if (tid < uint(L) && int(tid) < nl) {
      const int l = tid; float w = bgq[l];
      for (int i = s.lane_start[l]; i < s.lane_start[l + 1]; i++) {
        const int e = s.lane_edges[i];
        w += max(arrived[(e / N) * N + s.lsrc[l]] - sent[e], 0.0f);
      }
      want[l] = w;
      const float sh = min(1.0f, cap[l] / max(w, 1e-3f));
      share[l] = sh; bgq[l] -= bgq[l] * sh;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // sends: every edge's waiting bytes times its lane's share, into the ring
    const int slot = t % H;
    for (int e = tid; e < JKN; e += T) {
      const int jk = e / N, v = e % N, j = jk / K, k = jk % K;
      const int lane = (j < nj && v < nn) ? s.lane[j][k][v] : 255;
      float inc = 0;
      if (lane != 255) inc = max(arrived[jk * N + s.lsrc[lane]] - sent[e], 0.0f) * share[lane];
      const float dem = max(s.demand[j < nj ? j : 0], 1.0f);
      const half part = half(inc / dem);   // a send as a share of its commodity, in [0, 1]: what the ring holds
      sent[e] += float(part) * dem; ring[slot][e] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // landings: what was sent lag slots ago
    for (int e = tid; e < JKN; e += T) {
      const int jk = e / N, v = e % N, j = jk / K, k = jk % K;
      const int lane = (j < nj && v < nn) ? s.lane[j][k][v] : 255;
      if (lane != 255 && t >= lag[lane]) arrived[e] += float(ring[(t - lag[lane]) % H][e]) * max(s.demand[j], 1.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // delivered fractions and the area above them
    if (tid < uint(J * N)) {
      const int j = tid / N, d = tid % N;
      if (j < nj && (s.dest[j] >> d & 1)) {
        float g = 0; for (int k = 0; k < K; k++) g += arrived[(j * K + k) * N + d];
        const float f = min(g / max(s.demand[j], 1e-6f), 1.0f);
        area[tid] += (1 - f) * dt; frac[tid] = f;
      } else frac[tid] = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0 && finished == SLOTS) {
      bool all = true; for (int i = 0; i < J * N; i++) all = all && frac[i] >= 0.999f;
      if (all) finished = t + 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float most = -INFINITY;
    for (int i = 0; i < J * N; i++) if (i / N < nj && (s.dest[i / N] >> (i % N) & 1)) most = max(most, area[i] / fl / 0.05f);
    float sum = 0;
    for (int i = 0; i < J * N; i++) if (i / N < nj && (s.dest[i / N] >> (i % N) & 1)) sum += exp(area[i] / fl / 0.05f - most);
    objective[r] = 0.05f * (most + log(sum));
    done_out[r] = float(finished) * dt / fl;
  }
}

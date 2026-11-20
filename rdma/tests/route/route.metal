// The routing policy's training rollout (design/switched-multipath-dma.md), fused: one threadgroup a rollout. v0: the
// forward pass, all-gather transactions, fp32.
//
// A rollout is one scenario's transaction in the fluid model, SLOTS slots long, each slot SCEN.dt ns:
// - every lane serves the bytes waiting on it (its tree edges' and its background's) by processor sharing at its rate;
// - bytes sent reach the lane's head after its latency and one page time (a ring of the last H slots' sends);
// - each source releases its commodity's bytes into its K candidate trees at its out-lanes' rate, split by the
//   policy's weighting, refreshed every REFRESH slots;
// - the objective is a smooth maximum over (commodity, destination) of the area above its delivered fraction, over the
//   floor; `done` is the first slot every destination holds every byte, over the floor.
//
// The policy, refreshed per commodity: a 64-wide stream from the commodity's context and its last weighting's expected
// candidate embedding, through two layers of a GRU block (recurrent over refreshes) and a feed-forward block expanding
// to 192; each candidate tree is its N node tokens (parent, in-lane queue level, rate band), embedded, rotated by
// node position (RoPE at the embedding only), mean-pooled, and dotted with the stream.
#include <metal_stdlib>
using namespace metal;

constant constexpr int N = 6, L = 32, J = 6, K = 8, H = 16, SLOTS = 64, REFRESH = 8;
constant constexpr int D = 64, E = 192, CTX = 12, VOCAB = (N + 2) * 64, T = 256;
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
};

struct Layer { float z[D][2 * D], r[D][2 * D], c[D][2 * D], bz[D], br[D], bc[D], up[E][D], bu[E], down[D][E], bd[D]; };
struct Net { float inp[D][CTX + D], binp[D]; Layer layer[2]; float tok[VOCAB][D]; };

static inline float sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }

kernel void rollout(device const Scenario *scen [[buffer(0)]], device const Net &net [[buffer(1)]],
                    device const uint *which [[buffer(2)]], device float *objective [[buffer(3)]],
                    device float *done_out [[buffer(4)]], device float *rings [[buffer(5)]],
                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  device const Scenario &s = scen[which[r]];
  threadgroup float arrived[JKN], sent[JKN];
  device float (*ring)[JKN] = (device float (*)[JKN])(rings + size_t(r) * H * JKN);   // this rollout's last H slots' sends
  threadgroup float share[L], bgq[L], cap[L], want[L];
  threadgroup int lag[L];
  threadgroup float assigned[JK], released[J], outcap[J], p[JK], area[J * N], frac[J * N];
  threadgroup float x[J][D], h[2][J][D], gz[J][D], gr[J][D], ex[J][E], last[J][D], emb[JK][D];
  threadgroup int finished;
  const float dt = s.dt, fl = s.floor_ns;
  const int nl = int(s.lanes), nj = int(s.commodities), nn = int(s.nodes);

  for (int e = tid; e < JKN; e += T) { arrived[e] = 0; sent[e] = 0; for (int q = 0; q < H; q++) ring[q][e] = 0; }
  for (int i = tid; i < J * D; i += T) { x[i / D][i % D] = 0; h[0][i / D][i % D] = 0; h[1][i / D][i % D] = 0; last[i / D][i % D] = 0; }
  if (tid < uint(L)) {
    const int l = tid;
    cap[l] = l < nl ? s.rate[l] * dt : 0; bgq[l] = l < nl ? s.bg[l] : 0;
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
    // each lane's demand: its edges' bytes not yet sent on, plus its background
    if (tid < uint(L)) want[tid] = bgq[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(L) && int(tid) < nl) {
      const int l = tid; float w = bgq[l];
      for (int j = 0; j < nj; j++) for (int k = 0; k < K; k++) for (int v = 0; v < nn; v++)
        if (s.lane[j][k][v] == l) w += max(arrived[(j * K + k) * N + s.lsrc[l]] - sent[(j * K + k) * N + v], 0.0f);
      want[l] = w;
      const float sh = min(1.0f, cap[l] / max(w, 1e-3f));
      share[l] = sh; bgq[l] -= bgq[l] * sh;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t % REFRESH == 0) {
      // the policy: the stream for each commodity
      for (int i = tid; i < J * D; i += T) {
        const int j = i / D, d = i % D;
        float a = net.binp[d];
        const float ctx[CTX] = {log(1.0f + s.size / PAGE), float(t) / SLOTS, released[j] / max(s.demand[j], 1.0f), outcap[j] / max(s.demand[j], 1.0f),
                                float(nn) / 8, float(nl) / 32, float(K) / 8, float(popcount(uint(s.dest[j]))) / 8, 0, 0, 0, 0};
        for (int q = 0; q < CTX; q++) a += net.inp[d][q] * ctx[q];
        for (int q = 0; q < D; q++) a += net.inp[d][CTX + q] * last[j][q];
        x[j][d] = a;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int layer = 0; layer < 2; layer++) {
        device const Layer &w = net.layer[layer];
        for (int i = tid; i < J * D; i += T) {
          const int j = i / D, d = i % D; float az = w.bz[d], ar = w.br[d];
          for (int q = 0; q < D; q++) { az += w.z[d][q] * h[layer][j][q] + w.z[d][D + q] * x[j][q]; ar += w.r[d][q] * h[layer][j][q] + w.r[d][D + q] * x[j][q]; }
          gz[j][d] = sigmoid(az); gr[j][d] = sigmoid(ar);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) {
          const int j = i / D, d = i % D; float ac = w.bc[d];
          for (int q = 0; q < D; q++) ac += w.c[d][q] * gr[j][q] * h[layer][j][q] + w.c[d][D + q] * x[j][q];
          const float hn = (1 - gz[j][d]) * h[layer][j][d] + gz[j][d] * tanh(ac);
          ex[j][d] = hn;   // the new state, held here until every thread has read the old
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) { const int j = i / D, d = i % D; h[layer][j][d] = ex[j][d]; x[j][d] += ex[j][d]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * E; i += T) {
          const int j = i / E, e = i % E; float a = w.bu[e];
          for (int q = 0; q < D; q++) a += w.up[e][q] * x[j][q];
          ex[j][e] = max(a, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) {
          const int j = i / D, d = i % D; float a = w.bd[d];
          for (int q = 0; q < E; q++) a += w.down[d][q] * ex[j][q];
          gz[j][d] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < J * D; i += T) x[i / D][i % D] += gz[i / D][i % D];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // each candidate: its node tokens, embedded, rotated by node position, pooled
      for (int i = tid; i < JK * D / 2; i += T) {
        const int jk = i / (D / 2), pair = i % (D / 2), j = jk / K, k = jk % K;
        const float theta = pow(10000.0f, -2.0f * pair / D);
        float a = 0, b = 0;
        for (int v = 0; v < N; v++) {
          const int ln = s.lane[j][k][v];
          int parent = ln == 255 ? (v == s.src[j] ? N : N + 1) : s.lsrc[ln], level = 0, band = 0;
          if (ln != 255) {
            level = clamp(int(floor(log2(1 + 64 * want[ln] / max(s.rate[ln], 1e-6f) / fl))), 0, 15);
            band = clamp(int(floor(log2(max(s.rate[ln], 1.0f)))) - 2, 0, 3);
          }
          const int token = parent * 64 + level * 4 + band;
          const float u = net.tok[token][2 * pair], w2 = net.tok[token][2 * pair + 1], c = cos(v * theta), sn = sin(v * theta);
          a += u * c - w2 * sn; b += u * sn + w2 * c;
        }
        emb[jk][2 * pair] = a / N; emb[jk][2 * pair + 1] = b / N;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < uint(JK)) {
        const int j = tid / K; float a = 0;
        for (int q = 0; q < D; q++) a += emb[tid][q] * x[j][q];
        p[tid] = a / 8;
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
        for (int k = 0; k < K; k++) a += p[j * K + k] * emb[j * K + k][d];
        last[j][d] = a;
      }
    }

    // release at the sources
    if (tid < uint(J) && int(tid) < nj) {
      const int j = tid;
      const float rr = min(max(s.demand[j] - released[j], 0.0f), outcap[j]);
      released[j] += rr;
      for (int k = 0; k < K; k++) { assigned[j * K + k] += rr * p[j * K + k]; arrived[(j * K + k) * N + s.src[j]] = assigned[j * K + k]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // sends: every edge's waiting bytes times its lane's share, into the ring
    const int slot = t % H;
    for (int e = tid; e < JKN; e += T) {
      const int jk = e / N, v = e % N, j = jk / K, k = jk % K;
      const int ln = (j < nj && v < nn) ? s.lane[j][k][v] : 255;
      float inc = 0;
      if (ln != 255) inc = max(arrived[jk * N + s.lsrc[ln]] - sent[e], 0.0f) * share[ln];
      sent[e] += inc; ring[slot][e] = inc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    // landings: what was sent lag slots ago
    for (int e = tid; e < JKN; e += T) {
      const int jk = e / N, v = e % N, j = jk / K, k = jk % K;
      const int ln = (j < nj && v < nn) ? s.lane[j][k][v] : 255;
      if (ln != 255 && t >= lag[ln]) arrived[e] += ring[(t - lag[ln]) % H][e];
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

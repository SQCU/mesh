// The routing policy (design/switched-multipath-dma.md): a stochastic forwarding table, trained as in Kool, van Hoof &
// Welling 2019 (logits clipped C tanh(.), REINFORCE with leave-one-out baselines).
//
// The action. A piece of commodity j standing at node v, still owing destinations S, draws a split: each d in S assigned
// to one of v's out-lanes; the lanes given a nonempty share are sent on, each carrying the piece onward owing its share.
// A context is (j, v, S); its vocabulary is out_lanes(v)^|S|. A piece's tree is the splits it draws at every node it
// reaches, so every piece is its own draw from a product of vocabularies. At each decision point the policy emits
// every context's distribution in one pass; pieces then draw from those tables as they move (categorical draws, no
// network), already-visited nodes masked out (by rejection: the masked table exactly).
//
// The policy: two recurrent layers on a 64-wide residual stream (each a recurrence and a feed-forward block expanding to
// 192) read the lane tokens in turn, then each context's token as one more step from that state (all contexts in
// parallel); a context's output, RMS-normalised, times one matrix is w_c, and token s's logit is
// CLIP tanh(w_c . phi(s) / sqrt(FS)), phi(s) the split's features (which lane each destination takes, which lanes send,
// the split's predicted latest and mean arrival, sends a destination). Two arms of h_t = (1 - z_t) h_(t-1) + z_t h~_t:
//   FULL (the GRU): z = sigmoid(Wz [h_(t-1); x] + bz), r = sigmoid(Wr [h_(t-1); x] + br), h~ = tanh(Wh [r h_(t-1); x] + bh)
//   otherwise (minGRU): z = sigmoid(Wz x + bz), h~ = tanh(Wh x + bh): linear in h, a causal attention with a decay mask.
//
// The environment: the max-plus FIFO recursion over pieces with tests/switch.c's semantics (cables, earliest-free
// cable, duplex share, store-and-forward, its heap order). Group b of each commodity's pieces is posted at its source at
// tau_b, when every commodity's group b-1 has started on the wire there (mode 4: all at 0, as switch.c). Costs: the
// completion over the floor (a transaction's), and per piece its latency (arrivals less posting, over the floor) and
// the queueing it met (over the floor).
//
// Kernels: rollout (tables, draws, simulation, every draw logged), backward (for one task: each draw's advantage times
// (psi(s) - E psi), psi the gradient of its logit in w_c and the expectation over its context's masked table, carried
// back through the readout, the context step and the lane steps; each row's deltas and inputs written), reduce.
#include <metal_stdlib>
using namespace metal;

constant constexpr int N = 6, L = 32, J = 6, DEG = 5, NS = 64, GROUP = 64, EPMAX = 12, NPMAX = EPMAX * GROUP, CB = 4;
constant constexpr int D = 64, E = 192, IN = 48, FS = 40, T = 256, Q = 4, SG = T / 32, CH = 24;
constant constexpr int CMAX = J * N * 31, TABN = J * N * 7776, UMAX = 24576, GMAX = 4096, GH = 8192, HEAPN = 24576, TOKMAX = 3125;
constant constexpr float PAGE = 16384.0f, CLIP = 10.0f;

struct Scenario {
  float rate[L], lat[L], bg[L];   // a lane's rate (all its cables), latency, and bytes queued on it at the start
  uchar lsrc[L], ldst[L], cables[L];
  uchar spt[J][N];                // each commodity's unloaded shortest-path tree: the in-lane at each node, 255 none
  uchar src[J], dest[J];
  float demand[J];
  float floor_ns, size, duplex;
  uint commodities, lanes, nodes;
  float cut_ns, reach_ns;         // the floor's two bounds
};
// the fabric at a decision point, as the policy reads it
struct DState { float wait[L], dup[L], togo[N][N], recv[J][N], inflight[N], left[J], tau, progress; };
#ifdef FULL
constant constexpr int GW = 2 * D;
struct Layer { float z[D][GW], bz[D], r[D][GW], br[D], h[D][GW], bh[D], up[E][D], bu[E], down[D][E], bd[D]; };
constant constexpr int L_HX = 0, L_DZ = 128, L_DR = 192, L_DHT = 256, L_RX = 320, L_XM = 448, L_DU = 512, L_U = 704, L_DY = 896, R_LAYERSIZE = 960;
#else
constant constexpr int GW = D;
struct Layer { float z[D][GW], bz[D], h[D][GW], bh[D], up[E][D], bu[E], down[D][E], bd[D]; };
constant constexpr int L_X = 0, L_DZ = 64, L_DHT = 128, L_XM = 192, L_DU = 256, L_U = 448, L_DY = 640, R_LAYERSIZE = 704;
#endif
struct Net { float inp[D][IN], binp[D]; Layer layer[2]; float out[FS][D]; };
// a row's record for the GEMMs: the input projection's delta and input, each layer's, the readout's
constant constexpr int R_DX0 = 0, R_IN = 64, R_LAYER = R_IN + IN, R_DW = R_LAYER + 2 * R_LAYERSIZE, R_N = R_DW + FS, REC = R_N + D;

// tanh with its argument clamped: Metal's fast-math tanh is NaN past |x| ~ 89, and |x| = 20 is already +-1 in float
static inline float th(float v) { return tanh(clamp(v, -20.0f, 20.0f)); }
static inline float sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }
static inline uint fmix(uint h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }

// out[p][row] = (add ? out : 0) + (bias or 0) + sum_c W[row][c] in[p][c], W's rows `ws` apart, for p < P: a thread a row
// and a quarter of the rows of `in`, so each weight is loaded once for all P vectors and their sums stay in registers
template <int P, typename I, typename O>
static void mv(device const float *W, int ws, int rows, int cols, I in, int is, O out, int os, device const float *bias, bool add, uint tid) {
  constexpr int K = (P + Q - 1) / Q;
  for (int i = tid; i < rows * Q; i += T) {
    const int row = i / Q, q = i % Q;
    float acc[K];
    for (int k = 0; k < K; k++) acc[k] = 0;
    device const float4 *wr = (device const float4 *)(W + row * ws);   // every row and column count here is a multiple of 4
#pragma unroll 4
    for (int c4 = 0; c4 < cols / 4; c4++) {
      const float4 w = wr[c4];
      for (int k = 0; k < K; k++) if (q + Q * k < P) {
        const int at = (q + Q * k) * is + 4 * c4;
        acc[k] += w.x * float(in[at]) + w.y * float(in[at + 1]) + w.z * float(in[at + 2]) + w.w * float(in[at + 3]);
      }
    }
    for (int k = 0; k < K; k++) {
      const int p = q + Q * k;
      if (p < P) out[p * os + row] = (add ? float(out[p * os + row]) : 0.0f) + (bias ? bias[row] : 0.0f) + acc[k];
    }
  }
}
// out[p][col] = (add ? out : 0) + sum_row W[row][col] in[p][row]: a thread a column and a quarter of the vectors
template <int P, typename I, typename O>
static void mvt(device const float *W, int ws, int rows, int cols, I in, int is, O out, int os, bool add, uint tid) {
  constexpr int K = (P + Q - 1) / Q;
  for (int i = tid; i < cols * Q; i += T) {
    const int col = i / Q, q = i % Q;
    float acc[K];
    for (int k = 0; k < K; k++) acc[k] = 0;
    for (int row = 0; row < rows; row++) {
      const float w = W[row * ws + col];
      for (int k = 0; k < K; k++) if (q + Q * k < P) acc[k] += w * float(in[(q + Q * k) * is + row]);
    }
    for (int k = 0; k < K; k++) {
      const int p = q + Q * k;
      if (p < P) out[p * os + col] = (add ? float(out[p * os + col]) : 0.0f) + acc[k];
    }
  }
}
// the same product on the SIMD groups' matrix units: out (P x rows) = in (P x cols) W^T, 8x8 tiles, P a multiple of 8;
// a SIMD group a tile column of the output, its P / 8 tiles' sums in the matrix registers
template <int P, typename I, typename O>
static void mm(device const float *W, int ws, int rows, int cols, I in, int is, O out, int os, device const float *bias, bool add, uint tid) {
  constexpr int PT = P / 8;
  const int sg = tid / 32;
  const simdgroup_float8x8 one = simdgroup_float8x8(1.0f);   // the identity, to add a bias tile
  for (int ri = sg; ri < rows / 8; ri += SG) {
    simdgroup_float8x8 acc[PT], b;
    if (bias) simdgroup_load(b, bias + ri * 8, 0);   // every row the bias
    for (int pt = 0; pt < PT; pt++) {
      if (add) simdgroup_load(acc[pt], out + pt * 8 * os + ri * 8, os); else acc[pt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      if (bias) simdgroup_multiply_accumulate(acc[pt], one, b, acc[pt]);
    }
    for (int ci = 0; ci < cols / 8; ci++) {
      simdgroup_float8x8 wt, a;
      simdgroup_load(wt, W + ri * 8 * ws + ci * 8, ws, ulong2(0, 0), true);
      for (int pt = 0; pt < PT; pt++) { simdgroup_load(a, in + pt * 8 * is + ci * 8, is); simdgroup_multiply_accumulate(acc[pt], a, wt, acc[pt]); }
    }
    for (int pt = 0; pt < PT; pt++) simdgroup_store(acc[pt], out + pt * 8 * os + ri * 8, os);
  }
}
struct Tiles {};   // the rollout's rows, padded, on the matrix units
struct Rows {};    // the backward pass's rows
template <int P, typename I, typename O>
static void lin(Tiles, device const float *W, int ws, int rows, int cols, I in, int is, O out, int os, device const float *bias, bool add, uint tid) { mm<P>(W, ws, rows, cols, in, is, out, os, bias, add, tid); }
template <int P, typename I, typename O>
static void lin(Rows, device const float *W, int ws, int rows, int cols, I in, int is, O out, int os, device const float *bias, bool add, uint tid) { mv<P>(W, ws, rows, cols, in, is, out, os, bias, add, tid); }
#define SYNC threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)

// can node `from` reach a destination in `uncovered` without entering `tree`? (adj[v]: the nodes v's lanes enter)
static bool reaches(threadgroup const uint *adj, int from, uint tree, uint uncovered) {
  if (uncovered >> from & 1) return true;
  uint seen = 1u << from, frontier = seen;
  for (int round = 0; round < N; round++) {
    uint next = 0;
    for (int u = 0; u < N; u++) if (frontier >> u & 1) next |= adj[u];
    next &= ~tree & ~seen;
    if (next & uncovered) return true;
    if (!next) return false;
    seen |= next; frontier = next;
  }
  return false;
}

template <int P, typename M, typename X, typename HP, typename O>
static void cell(M m, device const Layer &w, X x, HP hp, O z, O r, O c, O rh, uint tid) {   // c may be r: r is spent once rh is formed
  constexpr int PDm = P * D;
#ifdef FULL
  lin<P>(m, &w.z[0][0], GW, D, D, hp, D, z, D, w.bz, false, tid);
  lin<P>(m, &w.r[0][0], GW, D, D, hp, D, r, D, w.br, false, tid);
  SYNC;
  lin<P>(m, &w.z[0][D], GW, D, D, x, D, z, D, nullptr, true, tid);
  lin<P>(m, &w.r[0][D], GW, D, D, x, D, r, D, nullptr, true, tid);
  SYNC;
  for (int i = tid; i < PDm; i += T) { z[i] = sigmoid(z[i]); rh[i] = sigmoid(r[i]) * hp[i]; r[i] = sigmoid(r[i]); }
  SYNC;
  lin<P>(m, &w.h[0][0], GW, D, D, rh, D, c, D, w.bh, false, tid);
  SYNC;
  lin<P>(m, &w.h[0][D], GW, D, D, x, D, c, D, nullptr, true, tid);
  SYNC;
#else
  (void)r; (void)rh;
  lin<P>(m, &w.z[0][0], GW, D, D, x, D, z, D, w.bz, false, tid);
  lin<P>(m, &w.h[0][0], GW, D, D, x, D, c, D, w.bh, false, tid);
  SYNC;
  for (int i = tid; i < PDm; i += T) z[i] = sigmoid(z[i]);
#endif
  for (int i = tid; i < PDm; i += T) c[i] = th(c[i]);
  SYNC;
}

// ---- the features (rollout and backward share them, so both read the same numbers) ----
struct Topo { uint adj[N], degv[N], srcmask; uchar outl[N][DEG], rev[L]; };
static float send1(threadgroup const Scenario &s, int l) { return min(s.size, PAGE) * s.cables[l] / s.rate[l]; }   // one piece, one cable
static float hopd(threadgroup const Scenario &s, device const DState &ds, int l) { return ds.wait[l] + send1(s, l) * ds.dup[l] + s.lat[l]; }

static void lane_token(threadgroup const Scenario &s, device const DState &ds, threadgroup const Topo &tp, int l, device float *o, uint tid0, uint step) {
  for (uint c = tid0; c < uint(IN); c += step) o[c] = 0;
  if (tid0 != 0 || l >= int(s.lanes)) return;
  const float fl = s.floor_ns, sz = max(s.size, 1.0f); const int a = s.lsrc[l], b = s.ldst[l];
  o[a] = 1; o[6 + b] = 1; o[12] = s.cables[l] / 4.0f; o[13] = send1(s, l) / fl; o[14] = s.lat[l] / fl; o[15] = ds.wait[l] / fl; o[16] = ds.dup[l];
  o[17] = log(max(s.rate[l] * fl / sz, 1e-6f)) / 8; o[18] = ds.inflight[a] / sz; o[19] = ds.inflight[b] / sz;
  o[20] = (tp.srcmask >> a & 1) ? 0 : 1; o[21] = (tp.srcmask >> b & 1) ? 0 : 1; o[22] = 1;
}
static void context_token(threadgroup const Scenario &s, device const DState &ds, threadgroup const Topo &tp, int j, int v, uint S, int dp, device float *o) {
  for (int c = 0; c < IN; c++) o[c] = 0;
  const float fl = s.floor_ns;
  o[v] = 1; o[18 + s.src[j]] = 1; o[24] = (tp.srcmask >> v & 1) ? 0 : 1;
  o[25] = log(1 + s.size / PAGE) / 8; o[26] = ds.left[j]; o[27] = s.reach_ns / fl; o[28] = s.cut_ns / fl;
  float most = 0;
  for (int d = 0; d < N; d++) {
    if (S >> d & 1) { o[6 + d] = 1; const float g = min(ds.togo[v][d] / fl, 1e3f); o[29 + d] = g; most = max(most, g); }
    if (s.dest[j] >> d & 1) o[12 + d] = 1;
    o[35 + d] = ds.recv[j][d];
  }
  o[41] = ds.tau / fl; o[42] = float(dp) / EPMAX; o[43] = ds.progress; o[44] = most; o[45] = 1; o[46] = 1;
}
// split token t at (v, S): its features, and whether every destination's lane can reach it without coming back to v
static bool split_features(threadgroup const Scenario &s, device const DState &ds, threadgroup const Topo &tp, int v, uint S, uint t, thread float *phi) {
  const int deg = int(tp.degv[v]); const float fl = s.floor_ns;
  for (int f = 0; f < FS; f++) phi[f] = 0;
  bool valid = true; uint used = 0; float most = 0, sum = 0; int cnt = 0;
  for (int d = 0; d < N; d++) {
    if (!(S >> d & 1)) continue;
    const int k = int(t % uint(deg)); t /= uint(deg);
    const int l = tp.outl[v][k], h = s.ldst[l];
    valid = valid && (h == d || reaches(tp.adj, h, 1u << v, 1u << d));
    phi[d * DEG + k] = 1; used |= 1u << k;
    const float arrive = hopd(s, ds, l) + (h == d ? 0.0f : min(ds.togo[h][d], 1e3f * fl));
    most = max(most, arrive); sum += arrive; cnt++;
  }
  for (int k = 0; k < DEG; k++) if (used >> k & 1) phi[30 + k] = 1;
  phi[35] = most / fl; phi[36] = sum / max(cnt, 1) / fl; phi[37] = float(popcount(used)) / max(cnt, 1);
  return valid;
}
static int tokens(threadgroup const Topo &tp, int v, uint S) { int n = 1; for (int d = 0; d < N; d++) if (S >> d & 1) n *= int(tp.degv[v]); return n; }
static void read_topology(threadgroup const Scenario &s, threadgroup Topo &tp, uint tid) {
  const int nl = int(s.lanes);
  if (tid < uint(N)) {
    const int v = tid; uint m = 0, k = 0;
    for (int l = 0; l < nl; l++) if (s.lsrc[l] == v) { m |= 1u << s.ldst[l]; if (k < uint(DEG)) tp.outl[v][k] = uchar(l); k++; }
    tp.adj[v] = m; tp.degv[v] = min(k, uint(DEG));
  }
  if (tid < uint(L)) {
    const int l = tid; int b = 255;
    for (int m = 0; m < nl; m++) if (l < nl && s.lsrc[m] == s.ldst[l] && s.ldst[m] == s.lsrc[l]) b = m;
    tp.rev[l] = uchar(b);
  }
  if (tid == 0) { uint m = 0; for (int j = 0; j < int(s.commodities); j++) m |= 1u << s.src[j]; tp.srcmask = m; }
}

// ---- one recurrence step for P rows, and its backward (activations in device memory) ----
// a step's activations for P rows: P D each of x0, x1, x2; per layer 6 P D (hp, z, r, c, rh, xm) and P E (up)
#define STEPSIZE(P) (15 * (P) * D + 2 * (P) * E)
// st: x0, x1, x2 (P x D each), then per layer hp, z, r, c, rh, xm (P x D each), up (P x E). hp: 2 x P x D, layer-major
template <int P, typename M>
static void step_fwd(M m, device const Net &net, device const float *inb, device const float *hp, device float *st, device float *u, uint tid) {
  constexpr int PDm = (P * D), YS = (6 * P * D + P * E);
  lin<P>(m, &net.inp[0][0], IN, D, IN, inb, IN, st, D, net.binp, false, tid);
  for (int i = tid; i < 2 * PDm; i += T) st[3 * PDm + (i / PDm) * YS + i % PDm] = hp[i];
  SYNC;
  for (int layer = 0; layer < 2; layer++) {
    device const Layer &w = net.layer[layer];
    device float *y = st + 3 * PDm + layer * YS, *xl = st + layer * PDm, *xn = st + (layer + 1) * PDm;
    cell<P>(m, w, xl, y, y + PDm, y + 2 * PDm, y + 3 * PDm, y + 4 * PDm, tid);
    for (int i = tid; i < PDm; i += T) y[5 * PDm + i] = xl[i] + (1 - y[PDm + i]) * y[i] + y[PDm + i] * y[3 * PDm + i];
    SYNC;
    lin<P>(m, &w.up[0][0], D, E, D, y + 5 * PDm, D, y + 6 * PDm, E, w.bu, false, tid);
    SYNC;
    for (int i = tid; i < P * E; i += T) u[i] = max(y[6 * PDm + i], 0.0f);
    SYNC;
    lin<P>(m, &w.down[0][0], E, D, E, u, E, xn, D, w.bd, false, tid);
    SYNC;
    for (int i = tid; i < PDm; i += T) xn[i] += y[5 * PDm + i];
    SYNC;
  }
}
// the state each layer moves to: (1 - z) hp + z h~
template <int P> static void step_state(device const float *st, device float *hn, uint tid) {
  constexpr int PDm = (P * D), YS = (6 * P * D + P * E);
  for (int i = tid; i < 2 * PDm; i += T) { device const float *y = st + 3 * PDm + (i / PDm) * YS; const int k = i % PDm; hn[i] = (1 - y[PDm + k]) * y[k] + y[PDm + k] * y[3 * PDm + k]; }
  SYNC;
}
// backward: g (P x D) the gradient at the stack's output, dh (2 x P x D) at each layer's new state; out: dh at the
// state before, the rows' records (rc, REC apart) for every weight but the readout's
template <int P>
static void step_back(device const Net &net, device const float *st, device const float *inb, device float *g, device float *gx, device float *tw,
                      device float *dh, device float *rc, uint tid) {
  constexpr int PDm = (P * D), YS = (6 * P * D + P * E);
  for (int layer = 1; layer >= 0; layer--) {
    device const Layer &w = net.layer[layer];
    device const float *y = st + 3 * PDm + layer * YS, *xl = st + layer * PDm;
    device float *dhl = dh + layer * PDm, *lr = rc + R_LAYER + layer * R_LAYERSIZE;
    for (int i = tid; i < PDm; i += T) {
      const int p = i / D, d = i % D;
      lr[p * REC + L_DY + d] = g[i]; lr[p * REC + L_XM + d] = y[5 * PDm + i];
#ifdef FULL
      lr[p * REC + L_HX + d] = y[i]; lr[p * REC + L_HX + D + d] = xl[i];
      lr[p * REC + L_RX + d] = y[4 * PDm + i]; lr[p * REC + L_RX + D + d] = xl[i];
#else
      lr[p * REC + L_X + d] = xl[i];
#endif
    }
    for (int i = tid; i < P * E; i += T) { const int p = i / E, e = i % E; lr[p * REC + L_U + e] = max(y[6 * PDm + i], 0.0f); }
    SYNC;
    mvt<P>(&w.down[0][0], E, D, E, lr + L_DY, REC, lr + L_DU, REC, false, tid);   // du = down^T dy, through the relu
    SYNC;
    for (int i = tid; i < P * E; i += T) { const int p = i / E, e = i % E; if (y[6 * PDm + i] <= 0) lr[p * REC + L_DU + e] = 0; }
    for (int i = tid; i < PDm; i += T) gx[i] = g[i];
    SYNC;
    mvt<P>(&w.up[0][0], D, E, D, lr + L_DU, REC, gx, D, true, tid);   // dxm = dy + up^T du
    SYNC;
    for (int i = tid; i < PDm; i += T) {   // the state: dhn = dxm + the next step's
      const int p = i / D, d = i % D;
      const float dn = gx[i] + dhl[i], z = y[PDm + i], c = y[3 * PDm + i], hp = y[i];
      lr[p * REC + L_DZ + d] = dn * (c - hp) * z * (1 - z);
      lr[p * REC + L_DHT + d] = dn * z * (1 - c * c);
      dhl[i] = dn * (1 - z);
      g[i] = gx[i];   // dx: the residual path, plus through the gates below
    }
    SYNC;
#ifdef FULL
    mvt<P>(&w.h[0][0], GW, D, GW, lr + L_DHT, REC, tw, GW, false, tid);   // [d rh; d x]
    SYNC;
    for (int i = tid; i < PDm; i += T) {
      const int p = i / D, d = i % D;
      const float drh = tw[p * GW + d], rr = y[2 * PDm + i];
      lr[p * REC + L_DR + d] = drh * y[i] * rr * (1 - rr);
      dhl[i] += drh * rr; g[i] += tw[p * GW + D + d];
    }
    SYNC;
    mvt<P>(&w.z[0][0], GW, D, GW, lr + L_DZ, REC, tw, GW, false, tid);
    SYNC;
    mvt<P>(&w.r[0][0], GW, D, GW, lr + L_DR, REC, tw, GW, true, tid);
    SYNC;
    for (int i = tid; i < PDm; i += T) { const int p = i / D, d = i % D; dhl[i] += tw[p * GW + d]; g[i] += tw[p * GW + D + d]; }
    SYNC;
#else
    (void)tw;
    mvt<P>(&w.z[0][0], GW, D, GW, lr + L_DZ, REC, g, D, true, tid);
    SYNC;
    mvt<P>(&w.h[0][0], GW, D, GW, lr + L_DHT, REC, g, D, true, tid);
    SYNC;
#endif
  }
  for (int i = tid; i < P * IN; i += T) { const int p = i / IN, c = i % IN; rc[p * REC + R_IN + c] = inb[i]; }
  for (int i = tid; i < PDm; i += T) { const int p = i / D, d = i % D; rc[p * REC + R_DX0 + d] = g[i]; }
  SYNC;
}

// ---- the per-rollout buffers ----
// simulator (uints): heap; draws (group, token, piece, next); groups (decision point << 11 | context, mask, first draw + 1,
// draws); the groups' hash; contexts (commodity | node << 3 | owed << 6, table offset, tokens); context of (j, v, S);
// pieces' latency and wait; each decision point's DState; a header
constant constexpr int SB_HEAP = 0, SB_USE = SB_HEAP + 2 * HEAPN, SB_GRP = SB_USE + 4 * UMAX, SB_HASH = SB_GRP + 4 * GMAX, SB_CTX = SB_HASH + GH;
constant constexpr int SB_CID = SB_CTX + 3 * CMAX, SB_PIECE = SB_CID + J * N * NS, SB_DS = SB_PIECE + 2 * J * NPMAX, DSF = int(sizeof(DState) / 4);
constant constexpr int SB_HDR = SB_DS + EPMAX * DSF, SB_SIZE = SB_HDR + 16;
enum { H_USES = 0, H_GROUPS, H_OVER, H_NC, H_NB, H_NP };
// rollout scratch (floats): the tables; a context chunk's step, inputs, states, hidden, norm, w; a lane step's
constant constexpr int RS_TAB = 0, RS_ST = TABN, RS_IN = RS_ST + STEPSIZE(CH), RS_HP = RS_IN + CH * IN, RS_U = RS_HP + 2 * CH * D;
constant constexpr int RS_NRM = RS_U + CH * E, RS_W = RS_NRM + CH * D, RS_ST1 = RS_W + CH * FS, RS_IN1 = RS_ST1 + STEPSIZE(1);
constant constexpr int RS_HP1 = RS_IN1 + IN, RS_HN1 = RS_HP1 + 2 * D, RS_U1 = RS_HN1 + 2 * D, RS_SIZE = RS_U1 + E;

// ---- the event heap: switch.c's push and pop, so equal times leave in the same order ----
static void push(device uint *hp, thread uint &n, float t, uint pay) {
  uint i = n++;
  while (i > 0 && as_type<float>(hp[2 * ((i - 1) / 2)]) > t) { const uint p = (i - 1) / 2; hp[2 * i] = hp[2 * p]; hp[2 * i + 1] = hp[2 * p + 1]; i = p; }
  hp[2 * i] = as_type<uint>(t); hp[2 * i + 1] = pay;
}
static void pop(device uint *hp, thread uint &n, thread float &t, thread uint &pay) {
  t = as_type<float>(hp[0]); pay = hp[1];
  const uint lt = hp[2 * (n - 1)], lp = hp[2 * (n - 1) + 1]; n--;
  uint i = 0;
  for (;;) {
    uint c = 2 * i + 1;
    if (c >= n) break;
    if (c + 1 < n && as_type<float>(hp[2 * (c + 1)]) < as_type<float>(hp[2 * c])) c++;
    if (as_type<float>(hp[2 * c]) >= as_type<float>(lt)) break;
    hp[2 * i] = hp[2 * c]; hp[2 * i + 1] = hp[2 * c + 1]; i = c;
  }
  if (n) { hp[2 * i] = lt; hp[2 * i + 1] = lp; }
}

// ---- the simulator, run by one thread ----
struct Fabric {
  threadgroup const Scenario *s; threadgroup const Topo *tp; threadgroup float (*free)[CB]; threadgroup float *laststart, *taus; threadgroup uint (*rcv)[N];
  device uint *sim; device const float *tab; uint n, mode, key0, dp; int np; float T, ll;
};
// a piece of commodity j at node v owing S (visited: the nodes its path has been through): its split, drawn and sent
static void split_send(thread Fabric &f, float t, int j, int v, uint S, uint visited, int pid) {
  threadgroup const Scenario &s = *f.s; threadgroup const Topo &tp = *f.tp;
  device uint *sim = f.sim;
  const int deg = int(tp.degv[v]); const uint avoid = visited | 1u << v;
  uint allow[N]; uint sig = 0;   // each destination's lanes: not into the path, and leading to it without coming back
  for (int d = 0; d < N; d++) {
    allow[d] = 0;
    if (!(S >> d & 1)) continue;
    for (int k = 0; k < deg; k++) { const int h = s.ldst[tp.outl[v][k]]; if (!(avoid >> h & 1) && (h == d || reaches(tp.adj, h, avoid, 1u << d))) allow[d] |= 1u << k; }
    sig |= allow[d] << (DEG * d);
  }
  const uint c = sim[SB_CID + (j * N + v) * NS + S], off = sim[SB_CTX + CMAX + c], n = sim[SB_CTX + 2 * CMAX + c];
  device const float *cdf = f.tab + off;
  uint tok = 0;
  if (f.mode == 2 || f.mode == 4) {   // the shortest-path tree's split
    uint place = 1;
    for (int d = 0; d < N; d++) if (S >> d & 1) {
      int x = d; while (s.lsrc[s.spt[j][x]] != v) x = s.lsrc[s.spt[j][x]];
      int k = 0; while (tp.outl[v][k] != s.spt[j][x]) k++;
      tok += uint(k) * place; place *= uint(deg);
    }
  } else if (f.mode == 3) {   // each destination a lane uniformly among the allowed
    uint place = 1;
    for (int d = 0; d < N; d++) if (S >> d & 1) {
      const float u = float(fmix(f.key0 ^ uint(pid * 0x9e3779b1u) ^ uint((v * 8 + d) * 0x85ebca6bu)) >> 8) / 16777216.0f;
      int target = min(int(u * popcount(allow[d])), int(popcount(allow[d])) - 1), k = 0;
      for (; k < deg; k++) if (allow[d] >> k & 1) { if (target == 0) break; target--; }
      tok += uint(k) * place; place *= uint(deg);
    }
  } else {   // a draw from the context's table, masked to the allowed by rejection (then by enumeration)
    bool ok = false;
    for (uint attempt = 0; attempt < 64 && !ok; attempt++) {
      const float u = float(fmix(f.key0 ^ uint(pid * 0x9e3779b1u) ^ uint((v * 64 + attempt) * 0x85ebca6bu) ^ S * 0xc2b2ae35u) >> 8) / 16777216.0f;
      uint lo = 0, hi = n - 1;
      while (lo < hi) { const uint mid = (lo + hi) / 2; if (cdf[mid] > u) hi = mid; else lo = mid + 1; }
      tok = lo;
      uint x = tok; ok = true;
      for (int d = 0; d < N; d++) if (S >> d & 1) { ok = ok && (allow[d] >> (x % uint(deg)) & 1); x /= uint(deg); }
    }
    if (!ok) {
      float total = 0;
      for (uint q = 0; q < n; q++) { uint x = q; bool a = true; for (int d = 0; d < N; d++) if (S >> d & 1) { a = a && (allow[d] >> (x % uint(deg)) & 1); x /= uint(deg); } if (a) total += cdf[q] - (q ? cdf[q - 1] : 0.0f); }
      const float u = float(fmix(f.key0 ^ uint(pid * 0x27d4eb2du) ^ S) >> 8) / 16777216.0f * total;
      float acc = 0;
      for (uint q = 0; q < n; q++) { uint x = q; bool a = true; for (int d = 0; d < N; d++) if (S >> d & 1) { a = a && (allow[d] >> (x % uint(deg)) & 1); x /= uint(deg); } if (a) { tok = q; acc += cdf[q] - (q ? cdf[q - 1] : 0.0f); if (acc > u) break; } }
    }
    f.ll += log(max(cdf[tok] - (tok ? cdf[tok - 1] : 0.0f), 1e-30f));
    // the draw, logged under its (decision point, context, mask) group
    const uint k0 = f.dp << 11 | c; uint hsh = fmix(k0 * 0x9e3779b1u ^ sig) & uint(GH - 1), g = 0;
    for (;;) {
      const uint e = sim[SB_HASH + hsh];
      if (e == 0) {
        g = sim[SB_HDR + H_GROUPS];
        if (g >= uint(GMAX)) { sim[SB_HDR + H_OVER] = 1; break; }
        sim[SB_HDR + H_GROUPS] = g + 1; sim[SB_HASH + hsh] = g + 1;
        sim[SB_GRP + 4 * g] = k0; sim[SB_GRP + 4 * g + 1] = sig; sim[SB_GRP + 4 * g + 2] = 0; sim[SB_GRP + 4 * g + 3] = 0;
        break;
      }
      if (sim[SB_GRP + 4 * (e - 1)] == k0 && sim[SB_GRP + 4 * (e - 1) + 1] == sig) { g = e - 1; break; }
      hsh = (hsh + 1) & uint(GH - 1);
    }
    const uint u = sim[SB_HDR + H_USES];
    if (u >= uint(UMAX)) sim[SB_HDR + H_OVER] = 1;
    else if (!sim[SB_HDR + H_OVER]) {
      sim[SB_HDR + H_USES] = u + 1;
      sim[SB_USE + 4 * u] = g; sim[SB_USE + 4 * u + 1] = tok; sim[SB_USE + 4 * u + 2] = uint(pid); sim[SB_USE + 4 * u + 3] = sim[SB_GRP + 4 * g + 2];
      sim[SB_GRP + 4 * g + 2] = u + 1; sim[SB_GRP + 4 * g + 3]++;
    }
  }
  // the sends: one a lane given a share, the shares as drawn
  uint share[DEG] = {0, 0, 0, 0, 0}; uint x = tok;
  for (int d = 0; d < N; d++) if (S >> d & 1) { share[x % uint(deg)] |= 1u << d; x /= uint(deg); }
  const float bytes = (pid % NPMAX) + 1 < f.np ? PAGE : s.size - PAGE * (f.np - 1);
  device float *pwait = (device float *)(sim + SB_PIECE + J * NPMAX);
  for (int k = 0; k < deg; k++) {
    if (!share[k]) continue;
    const int l = tp.outl[v][k], h = s.ldst[l];
    int cb = 0; for (int q = 1; q < s.cables[l]; q++) if (f.free[l][q] < f.free[l][cb]) cb = q;
    const float start = max(f.free[l][cb], t);
    const int rl = tp.rev[l];
    const bool both = rl != 255 && cb < s.cables[rl] && f.free[rl][cb] > start;
    const float time = bytes / (s.rate[l] / s.cables[l] * (both ? s.duplex : 1.0f));
    f.free[l][cb] = start + time; pwait[pid] += start - t;
    if (v == s.src[j]) f.laststart[j] = max(f.laststart[j], start);
    if (f.n >= uint(HEAPN)) { sim[SB_HDR + H_OVER] = 1; continue; }
    push(sim + SB_HEAP, f.n, start + time + s.lat[l], uint(pid) << 15 | uint(h) << 12 | share[k] << 6 | (avoid & 63));
  }
}
// every event up to `until`, in time order: a piece landing; at a destination delivered; split and sent on
static void advance(thread Fabric &f, float until) {
  device float *plat = (device float *)(f.sim + SB_PIECE);
  while (f.n > 0 && as_type<float>(f.sim[SB_HEAP]) <= until) {
    float t; uint pay; pop(f.sim + SB_HEAP, f.n, t, pay);
    const int pid = pay >> 15, v = pay >> 12 & 7, j = pid / NPMAX; uint S = pay >> 6 & 63; const uint visited = pay & 63;
    if (S >> v & 1) {
      f.T = max(f.T, t); plat[pid] += (t - f.taus[(pid % NPMAX) / GROUP]) / f.s->floor_ns / popcount(uint(f.s->dest[j]));
      f.rcv[j][v]++; S &= ~(1u << v);
    }
    if (S) split_send(f, t, j, v, S, visited, pid);
  }
}

kernel void rollout(device const Scenario *scen [[buffer(0)]], device const Net &net [[buffer(1)]],
                    device const uint *which [[buffer(2)]], device const uint *draw [[buffer(3)]],
                    device float4 *cost [[buffer(4)]], device float *scratch [[buffer(5)]], device uint *simbuf [[buffer(6)]],
                    constant uint &mode [[buffer(7)]], device float *loglik [[buffer(8)]],
                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup Scenario s;
  { device const uint *from = (device const uint *)&scen[which[r]]; threadgroup uint *to = (threadgroup uint *)&s;
    for (uint i = tid; i < sizeof(Scenario) / 4; i += T) to[i] = from[i]; }
  threadgroup Topo tp;
  threadgroup float free_at[L][CB], taus[EPMAX], laststart[J];
  threadgroup uint rcv[J][N];
  threadgroup int chunkoff[CH + 1];
  device float *sc = scratch + size_t(r) * RS_SIZE, *tab = sc + RS_TAB;
  device uint *sim = simbuf + size_t(r) * SB_SIZE;
  SYNC;
  read_topology(s, tp, tid);
  const float fl = s.floor_ns;
  const int nl = int(s.lanes), nj = int(s.commodities), nn = int(s.nodes);
  const int np = max(1, int(ceil(s.size / PAGE))), nb = (np + GROUP - 1) / GROUP;
  for (int i = tid; i < L * CB; i += T) { const int l = i / CB; free_at[l][i % CB] = l < nl ? s.bg[l] / s.rate[l] : 0; }
  for (int i = tid; i < GH; i += T) sim[SB_HASH + i] = 0;
  for (int i = tid; i < 2 * J * NPMAX; i += T) ((device float *)(sim + SB_PIECE))[i] = 0;
  for (int i = tid; i < J * N; i += T) rcv[i / N][i % N] = 0;
  if (tid < uint(J)) laststart[tid] = 0;
  SYNC;
  if (tid == 0) {   // the contexts: every (commodity, node, owed set), and each one's table
    uint nc = 0, off = 0;
    for (int j = 0; j < nj; j++) for (int v = 0; v < nn; v++) for (uint S = 1; S < uint(NS); S++) {
      if ((S & ~uint(s.dest[j])) || (S >> v & 1)) continue;
      const int n = tokens(tp, v, S);
      sim[SB_CTX + nc] = uint(j) | uint(v) << 3 | S << 6; sim[SB_CTX + CMAX + nc] = off; sim[SB_CTX + 2 * CMAX + nc] = uint(n);
      sim[SB_CID + (j * N + v) * NS + S] = nc; nc++; off += uint(n);
    }
    sim[SB_HDR + H_NC] = nc; sim[SB_HDR + H_NB] = uint(nb); sim[SB_HDR + H_NP] = uint(np);
    sim[SB_HDR + H_USES] = 0; sim[SB_HDR + H_GROUPS] = 0; sim[SB_HDR + H_OVER] = 0;
  }
  SYNC;
  const int nc = int(sim[SB_HDR + H_NC]);
  Fabric f = {&s, &tp, free_at, laststart, taus, rcv, sim, tab, 0, mode, draw[r], 0, np, 0, 0};

  for (int b = 0; b < nb; b++) {
    if (tid == 0) {   // the decision point: when every commodity's last group has started on the wire at its source
      float tau = 0;
      if (b > 0 && mode != 4) { tau = INFINITY; for (int j = 0; j < nj; j++) tau = min(tau, laststart[j]); tau = max(tau, taus[b - 1]); advance(f, tau); }
      taus[b] = tau; f.dp = uint(b);
      device DState &ds = *((device DState *)(sim + SB_DS) + b);
      ds.tau = tau;
      for (int l = 0; l < L; l++) {
        float w = INFINITY; bool busy = false;
        if (l < nl) { for (int c = 0; c < s.cables[l]; c++) w = min(w, max(free_at[l][c] - tau, 0.0f)); const int rl = tp.rev[l]; if (rl != 255) for (int c = 0; c < s.cables[rl]; c++) busy = busy || free_at[rl][c] > tau; }
        ds.wait[l] = l < nl ? w : 0; ds.dup[l] = busy ? s.duplex : 1.0f;
      }
      for (int v = 0; v < N; v++) ds.inflight[v] = 0;
      for (uint e = 0; e < f.n; e++) { const uint pay = sim[SB_HEAP + 2 * e + 1]; const int pid = pay >> 15; ds.inflight[pay >> 12 & 7] += (pid % NPMAX) + 1 < np ? PAGE : s.size - PAGE * (np - 1); }
      float dist[N][N];
      for (int a = 0; a < N; a++) for (int c = 0; c < N; c++) dist[a][c] = a == c ? 0 : INFINITY;
      for (int l = 0; l < nl; l++) dist[s.lsrc[l]][s.ldst[l]] = min(dist[s.lsrc[l]][s.ldst[l]], hopd(s, ds, l));
      for (int m = 0; m < N; m++) for (int a = 0; a < N; a++) for (int c = 0; c < N; c++) dist[a][c] = min(dist[a][c], dist[a][m] + dist[m][c]);
      float got = 0, want = 0;
      for (int a = 0; a < N; a++) for (int c = 0; c < N; c++) ds.togo[a][c] = dist[a][c];
      for (int j = 0; j < J; j++) {
        ds.left[j] = float(max(np - b * GROUP, 0)) / np;
        for (int d = 0; d < N; d++) { ds.recv[j][d] = float(rcv[j][d]) / np; if (j < nj && (s.dest[j] >> d & 1)) { got += ds.recv[j][d]; want += 1; } }
      }
      ds.progress = got / max(want, 1.0f);
    }
    SYNC;
    device const DState &ds = *((device const DState *)(sim + SB_DS) + b);
    // the lane tokens in turn: the state after them is every context's starting state
    for (int i = tid; i < 2 * D; i += T) sc[RS_HP1 + i] = 0;
    SYNC;
    for (int l = 0; l < nl; l++) {
      lane_token(s, ds, tp, l, sc + RS_IN1, tid, T);
      SYNC;
      step_fwd<1>(Rows(), net, sc + RS_IN1, sc + RS_HP1, sc + RS_ST1, sc + RS_U1, tid);
      step_state<1>(sc + RS_ST1, sc + RS_HP1, tid);
    }
    // the contexts, CH at a time: one step each from that state; w = out . rms(x2); every token's logit
    for (int c0 = 0; c0 < nc; c0 += CH) {
      const int cnt = min(CH, nc - c0);
      for (int p = tid; p < CH; p += T) {
        if (p < cnt) { const uint info = sim[SB_CTX + c0 + p]; context_token(s, ds, tp, info & 7, info >> 3 & 7, info >> 6, b, sc + RS_IN + p * IN); }
        else for (int c = 0; c < IN; c++) sc[RS_IN + p * IN + c] = 0;
      }
      for (int i = tid; i < 2 * CH * D; i += T) sc[RS_HP + i] = sc[RS_HP1 + (i / (CH * D)) * D + i % D];
      if (tid == 0) { int o = 0; for (int p = 0; p < cnt; p++) { chunkoff[p] = o; o += int(sim[SB_CTX + 2 * CMAX + c0 + p]); } chunkoff[cnt] = o; }
      SYNC;
      step_fwd<CH>(Tiles(), net, sc + RS_IN, sc + RS_HP, sc + RS_ST, sc + RS_U, tid);
      for (int p = tid; p < CH; p += T) {
        device const float *x2 = sc + RS_ST + 2 * CH * D + p * D; float m = 0;
        for (int d = 0; d < D; d++) m += x2[d] * x2[d];
        const float inv = rsqrt(m / D + 1e-6f);
        for (int d = 0; d < D; d++) sc[RS_NRM + p * D + d] = x2[d] * inv;
      }
      SYNC;
      mm<CH>(&net.out[0][0], D, FS, D, sc + RS_NRM, D, sc + RS_W, FS, nullptr, false, tid);
      SYNC;
      for (int q = tid; q < chunkoff[cnt]; q += T) {
        int p = 0; while (chunkoff[p + 1] <= q) p++;
        const uint info = sim[SB_CTX + c0 + p], t = uint(q - chunkoff[p]);
        float phi[FS];
        const bool valid = split_features(s, ds, tp, info >> 3 & 7, info >> 6, t, phi);
        float a = 0; for (int k = 0; k < FS; k++) a += sc[RS_W + p * FS + k] * phi[k];
        tab[sim[SB_CTX + CMAX + c0 + p] + t] = valid ? CLIP * th(a / sqrt(float(FS))) : -INFINITY;
      }
      SYNC;
    }
    for (int c = tid; c < nc; c += T) {   // each context's table: its softmax, as a cumulative distribution
      device float *x = tab + sim[SB_CTX + CMAX + c]; const int n = int(sim[SB_CTX + 2 * CMAX + c]);
      float most = -INFINITY; for (int t = 0; t < n; t++) most = max(most, x[t]);
      float sum = 0; for (int t = 0; t < n; t++) sum += x[t] > -INFINITY ? exp(x[t] - most) : 0;
      float acc = 0; for (int t = 0; t < n; t++) { acc += x[t] > -INFINITY ? exp(x[t] - most) / sum : 0; x[t] = acc; }
      x[n - 1] = 1.0f;
    }
    SYNC;
    if (tid == 0) for (int j = 0; j < nj; j++) for (int i = b * GROUP; i < min(np, (b + 1) * GROUP); i++) split_send(f, b == 0 || mode == 4 ? 0.0f : taus[b], j, s.src[j], uint(s.dest[j]), 0, j * NPMAX + i);
    SYNC;
  }
  if (tid == 0) {
    advance(f, INFINITY);
    device const float *plat = (device const float *)(sim + SB_PIECE), *pwait = plat + J * NPMAX;
    float lat = 0, wait = 0;
    for (int j = 0; j < nj; j++) for (int i = 0; i < np; i++) { lat += plat[j * NPMAX + i]; wait += pwait[j * NPMAX + i]; }
    cost[r] = float4(f.T / fl, lat / (nj * np), wait / (nj * np) / fl, sim[SB_HDR + H_OVER] ? -1.0f : 1.0f);
    loglik[r] = f.ll;
  }
}

// ---- backward, for one task ----
// task 0 throughput (each draw's advantage the rollout's, from the host), 1 latency and 2 queueing (each piece's against
// the mean of the other pieces of its commodity's group, times the host's scale), 3 the action's log-likelihood itself
// (advantage 1: the gradient check). Writes the rows' records at rows reserved from `rows`, and the action's masked and
// unmasked log-likelihood.
// scratch (floats): lane steps and tokens; a context chunk's step, inputs, states, hidden, norm, w, dw, gradients; a lane
// step's gradients; a group's token logits and allowed flags; its draws; the pieces' advantages; contexts' marks and rows
constant constexpr int BS_ST1 = 0, BS_TOK = BS_ST1 + L * STEPSIZE(1), BS_ST = BS_TOK + L * IN, BS_IN = BS_ST + STEPSIZE(CH), BS_HP = BS_IN + CH * IN;
constant constexpr int BS_U = BS_HP + 2 * CH * D, BS_NRM = BS_U + CH * E, BS_RMS = BS_NRM + CH * D, BS_W = BS_RMS + CH, BS_DW = BS_W + CH * FS;
constant constexpr int BS_G = BS_DW + CH * FS, BS_GX = BS_G + CH * D, BS_TW = BS_GX + CH * D, BS_DH = BS_TW + 2 * CH * D, BS_DHF = BS_DH + 2 * CH * D;
constant constexpr int BS_HP1 = BS_DHF + 2 * D, BS_G1 = BS_HP1 + 2 * D, BS_GX1 = BS_G1 + D, BS_TW1 = BS_GX1 + D, BS_DH1 = BS_TW1 + 2 * D, BS_U1 = BS_DH1 + 2 * D;
constant constexpr int BS_LOG = BS_U1 + E, BS_OK = BS_LOG + TOKMAX, BS_DRAW = BS_OK + TOKMAX, BS_ADV = BS_DRAW + 2 * UMAX, BS_MARK = BS_ADV + J * NPMAX;
constant constexpr int BS_ROW = BS_MARK + CMAX, BS_LIST = BS_ROW + CMAX, BS_SIZE = BS_LIST + CMAX;

// the sum of v over the threadgroup
static float total(float v, threadgroup float *red, uint tid) {
  v = simd_sum(v);
  if (tid % 32 == 0) red[tid / 32] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float a = 0; for (int k = 0; k < SG; k++) a += red[k];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return a;
}

kernel void backward(device const Scenario *scen [[buffer(0)]], device const Net &net [[buffer(1)]], device const uint *which [[buffer(2)]],
                     device float *scratch [[buffer(3)]], device const uint *simbuf [[buffer(4)]], device float *rec [[buffer(5)]],
                     device atomic_uint *rows [[buffer(6)]], constant uint &task [[buffer(7)]], device const float *weight [[buffer(8)]],
                     device float2 *loglik [[buffer(9)]], constant uint &rcap [[buffer(10)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup Scenario s;
  { device const uint *from = (device const uint *)&scen[which[r]]; threadgroup uint *to = (threadgroup uint *)&s;
    for (uint i = tid; i < sizeof(Scenario) / 4; i += T) to[i] = from[i]; }
  threadgroup Topo tp;
  threadgroup float red[SG], vec[SG][FS], epsi[FS], spsi[FS], grp[4];
  threadgroup uint base, cur, nused, ngr;
  threadgroup float llm, llu;
  device float *a = scratch + size_t(r) * BS_SIZE;
  device const uint *sim = simbuf + size_t(r) * SB_SIZE;
  SYNC;
  read_topology(s, tp, tid);
  const int nl = int(s.lanes), nj = int(s.commodities), nb = int(sim[SB_HDR + H_NB]), np = int(sim[SB_HDR + H_NP]), ng = int(sim[SB_HDR + H_GROUPS]);
  device const float *plat = (device const float *)(sim + SB_PIECE), *pwait = plat + J * NPMAX;
  if (tid == 0) { llm = 0; llu = 0; }
  // each piece's advantage for this task
  for (int i = tid; i < nj * np; i += T) {
    const int j = i / np, k = i % np, pid = j * NPMAX + k;
    float A = task == 3 ? 1.0f : task == 0 ? weight[3 * r] : 0.0f;
    if (task == 1 || task == 2) {
      device const float *c = task == 1 ? plat : pwait;
      const int g0 = (k / GROUP) * GROUP, g1 = min(np, g0 + GROUP);
      float sum = 0; for (int q = g0; q < g1; q++) sum += c[j * NPMAX + q];
      const float scale = task == 1 ? 1.0f : 1.0f / s.floor_ns;
      A = g1 - g0 > 1 ? weight[3 * r + task] * scale * (c[pid] - (sum - c[pid]) / (g1 - g0 - 1)) : 0.0f;
    }
    a[BS_ADV + pid] = A;
  }
  SYNC;
  for (int b = 0; b < nb; b++) {
    device const DState &ds = *((device const DState *)(sim + SB_DS) + b);
    // the lane steps again, each kept
    for (int i = tid; i < 2 * D; i += T) a[BS_HP1 + i] = 0;
    SYNC;
    for (int l = 0; l < nl; l++) {
      lane_token(s, ds, tp, l, a + BS_TOK + l * IN, tid, T);
      SYNC;
      step_fwd<1>(Rows(), net, a + BS_TOK + l * IN, a + BS_HP1, a + BS_ST1 + l * STEPSIZE(1), a + BS_U1, tid);
      step_state<1>(a + BS_ST1 + l * STEPSIZE(1), a + BS_HP1, tid);
    }
    // the contexts this decision point's draws used, each a row
    for (int c = tid; c < CMAX; c += T) a[BS_MARK + c] = 0;
    SYNC;
    if (tid == 0) {
      uint n = 0;
      for (int g = 0; g < ng; g++) {
        const uint k0 = sim[SB_GRP + 4 * g];
        if (int(k0 >> 11) != b) continue;
        const uint c = k0 & 2047;
        if (a[BS_MARK + c] == 0) { a[BS_MARK + c] = 1; a[BS_ROW + c] = float(n); a[BS_LIST + n] = float(c); n++; }
      }
      nused = n;
    }
    for (int i = tid; i < 2 * D; i += T) a[BS_DHF + i] = 0;
    SYNC;
    const int nu = int(nused);
    for (int c0 = 0; c0 < nu; c0 += CH) {
      const int cnt = min(CH, nu - c0);
      for (int p = tid; p < CH; p += T) {
        if (p < cnt) { const uint info = sim[SB_CTX + uint(a[BS_LIST + c0 + p])]; context_token(s, ds, tp, info & 7, info >> 3 & 7, info >> 6, b, a + BS_IN + p * IN); }
        else for (int c = 0; c < IN; c++) a[BS_IN + p * IN + c] = 0;
      }
      for (int i = tid; i < 2 * CH * D; i += T) a[BS_HP + i] = a[BS_HP1 + (i / (CH * D)) * D + i % D];
      for (int i = tid; i < CH * FS; i += T) a[BS_DW + i] = 0;
      SYNC;
      step_fwd<CH>(Rows(), net, a + BS_IN, a + BS_HP, a + BS_ST, a + BS_U, tid);
      for (int p = tid; p < CH; p += T) {
        device const float *x2 = a + BS_ST + 2 * CH * D + p * D; float m = 0;
        for (int d = 0; d < D; d++) m += x2[d] * x2[d];
        const float inv = rsqrt(m / D + 1e-6f);
        a[BS_RMS + p] = inv;
        for (int d = 0; d < D; d++) a[BS_NRM + p * D + d] = x2[d] * inv;
      }
      SYNC;
      mv<CH>(&net.out[0][0], D, FS, D, a + BS_NRM, D, a + BS_W, FS, nullptr, false, tid);
      SYNC;
      // each group of this chunk's contexts: E psi over its masked table, and its draws' sum of advantage times psi
      if (tid == 0) cur = 0;
      SYNC;
      for (;;) {
        if (tid == 0) {   // the next group whose context is in this chunk; its draws gathered
          uint g = cur; int p = -1;
          for (; g < uint(ng); g++) {
            const uint k0 = sim[SB_GRP + 4 * g];
            if (int(k0 >> 11) != b) continue;
            const int row = int(a[BS_ROW + (k0 & 2047)]);
            if (row >= c0 && row < c0 + cnt) { p = row - c0; break; }
          }
          cur = g + 1; grp[0] = float(p); ngr = 0;
          if (p >= 0) { uint u = sim[SB_GRP + 4 * g + 2], m = 0; while (u) { a[BS_DRAW + 2 * m] = float(sim[SB_USE + 4 * (u - 1) + 1]); a[BS_DRAW + 2 * m + 1] = float(sim[SB_USE + 4 * (u - 1) + 2]); m++; u = sim[SB_USE + 4 * (u - 1) + 3]; } ngr = m; grp[1] = float(g); }
        }
        SYNC;
        const int p = int(grp[0]);
        if (p < 0) break;
        const uint g = uint(grp[1]), sig = sim[SB_GRP + 4 * g + 1], c = sim[SB_GRP + 4 * g] & 2047, info = sim[SB_CTX + c];
        const int v = int(info >> 3 & 7), deg = int(tp.degv[v]), n = int(sim[SB_CTX + 2 * CMAX + c]); const uint S = info >> 6;
        device const float *w = a + BS_W + p * FS;
        float most = -INFINITY, mosts = -INFINITY;
        for (int t = tid; t < n; t += T) {
          float phi[FS];
          const bool valid = split_features(s, ds, tp, v, S, uint(t), phi);
          uint x = uint(t); bool allowed = valid;
          for (int d = 0; d < N; d++) if (S >> d & 1) { allowed = allowed && (sig >> (DEG * d + int(x % uint(deg))) & 1); x /= uint(deg); }
          float q = 0; for (int k = 0; k < FS; k++) q += w[k] * phi[k];
          const float lg = valid ? CLIP * th(q / sqrt(float(FS))) : -INFINITY;
          a[BS_LOG + t] = lg; a[BS_OK + t] = allowed ? 1 : 0;
          if (allowed) most = max(most, lg);
          mosts = max(mosts, lg);
        }
        most = simd_max(most); mosts = simd_max(mosts);
        if (tid % 32 == 0) { red[tid / 32] = most; vec[tid / 32][0] = mosts; }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        most = -INFINITY; mosts = -INFINITY; for (int k = 0; k < SG; k++) { most = max(most, red[k]); mosts = max(mosts, vec[k][0]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0, sums = 0;
        for (int t = tid; t < n; t += T) { const float lg = a[BS_LOG + t]; if (lg > -INFINITY) { sums += exp(lg - mosts); if (a[BS_OK + t] > 0) sum += exp(lg - most); } }
        sum = total(sum, red, tid); sums = total(sums, red, tid);
        const float lse = most + log(sum), lses = mosts + log(sums);
        // E psi, psi = d logit / d w = CLIP (1 - tanh^2) phi / sqrt(FS), over the masked table
        float acc[FS];
        for (int k = 0; k < FS; k++) acc[k] = 0;
        for (int t = tid; t < n; t += T) {
          if (a[BS_OK + t] == 0) continue;
          float phi[FS]; split_features(s, ds, tp, v, S, uint(t), phi);
          float q = 0; for (int k = 0; k < FS; k++) q += w[k] * phi[k];
          const float th1 = th(q / sqrt(float(FS))), pr = exp(a[BS_LOG + t] - lse), sc1 = pr * CLIP * (1 - th1 * th1) / sqrt(float(FS));
          for (int k = 0; k < FS; k++) acc[k] += sc1 * phi[k];
        }
        for (int k = 0; k < FS; k++) { const float v2 = simd_sum(acc[k]); if (tid % 32 == 0) vec[tid / 32][k] = v2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(FS)) { float v2 = 0; for (int k = 0; k < SG; k++) v2 += vec[k][tid]; epsi[tid] = v2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the draws: sum of advantage times psi, of advantages, and the log-likelihoods
        float asum = 0, lm = 0, lu = 0;
        for (int k = 0; k < FS; k++) acc[k] = 0;
        for (uint m = tid; m < ngr; m += T) {
          const uint t = uint(a[BS_DRAW + 2 * m]); const int pid = int(a[BS_DRAW + 2 * m + 1]);
          const float A = a[BS_ADV + pid];
          float phi[FS]; split_features(s, ds, tp, v, S, t, phi);
          float q = 0; for (int k = 0; k < FS; k++) q += w[k] * phi[k];
          const float th1 = th(q / sqrt(float(FS))), sc1 = A * CLIP * (1 - th1 * th1) / sqrt(float(FS));
          for (int k = 0; k < FS; k++) acc[k] += sc1 * phi[k];
          asum += A; lm += a[BS_LOG + t] - lse; lu += a[BS_LOG + t] - lses;
        }
        for (int k = 0; k < FS; k++) { const float v2 = simd_sum(acc[k]); if (tid % 32 == 0) vec[tid / 32][k] = v2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(FS)) { float v2 = 0; for (int k = 0; k < SG; k++) v2 += vec[k][tid]; spsi[tid] = v2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        asum = total(asum, red, tid); lm = total(lm, red, tid); lu = total(lu, red, tid);
        if (tid < uint(FS)) a[BS_DW + p * FS + tid] += spsi[tid] - asum * epsi[tid];
        if (tid == 0) { llm += lm; llu += lu; }
        SYNC;
      }
      // back through the readout and the RMS norm, then the context step
      if (tid == 0) { base = atomic_fetch_add_explicit(rows, uint(CH), memory_order_relaxed); }
      SYNC;
      if (base + CH > rcap) { if (tid == 0) loglik[r] = float2(NAN, NAN); return; }
      device float *rc = rec + size_t(base) * REC;
      for (int i = tid; i < CH * FS; i += T) rc[(i / FS) * REC + R_DW + i % FS] = a[BS_DW + i];
      for (int i = tid; i < CH * D; i += T) rc[(i / D) * REC + R_N + i % D] = a[BS_NRM + i];
      SYNC;
      mvt<CH>(&net.out[0][0], D, FS, D, a + BS_DW, FS, a + BS_G, D, false, tid);
      SYNC;
      for (int p = tid; p < CH; p += T) {   // d x2 = (dn - n mean(dn . n)) / rms
        device float *gp = a + BS_G + p * D; device const float *np2 = a + BS_NRM + p * D;
        float m = 0; for (int d = 0; d < D; d++) m += gp[d] * np2[d];
        m /= D;
        for (int d = 0; d < D; d++) gp[d] = (gp[d] - np2[d] * m) * a[BS_RMS + p];
      }
      for (int i = tid; i < 2 * CH * D; i += T) a[BS_DH + i] = 0;
      SYNC;
      step_back<CH>(net, a + BS_ST, a + BS_IN, a + BS_G, a + BS_GX, a + BS_TW, a + BS_DH, rc, tid);
      for (int i = tid; i < 2 * D; i += T) { float v2 = 0; for (int p = 0; p < CH; p++) v2 += a[BS_DH + (i / D) * CH * D + p * D + i % D]; a[BS_DHF + i] += v2; }
      SYNC;
    }
    // back through the lane steps, from the gradient at the state they left
    if (tid == 0) { base = atomic_fetch_add_explicit(rows, uint(nl), memory_order_relaxed); }
    for (int i = tid; i < 2 * D; i += T) a[BS_DH1 + i] = a[BS_DHF + i];
    SYNC;
    if (base + uint(nl) > rcap) { if (tid == 0) loglik[r] = float2(NAN, NAN); return; }
    for (int l = nl - 1; l >= 0; l--) {
      device float *rc = rec + size_t(base + l) * REC;
      for (int i = tid; i < D; i += T) a[BS_G1 + i] = 0;
      for (int i = tid; i < FS; i += T) rc[R_DW + i] = 0;
      for (int i = tid; i < D; i += T) rc[R_N + i] = 0;
      SYNC;
      step_back<1>(net, a + BS_ST1 + l * STEPSIZE(1), a + BS_TOK + l * IN, a + BS_G1, a + BS_GX1, a + BS_TW1, a + BS_DH1, rc, tid);
    }
  }
  if (tid == 0) loglik[r] = float2(llm, llu);
}

// the zero rows past the last reserved, up to the GEMMs' padded count
kernel void zero_rows(device float *rec [[buffer(0)]], constant uint2 &span [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i < span.y * uint(REC)) rec[size_t(span.x) * REC + i] = 0;
}
// one weight block's split-K partials, summed: g[at + i] = sum_b part[base + b size + i] (q: base, size, batches, at)
kernel void reduce(device const float *part [[buffer(0)]], device float *g [[buffer(1)]], constant uint4 &q [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i >= q.y) return;
  float v = 0;
  for (uint b = 0; b < q.z; b++) v += part[q.x + size_t(b) * q.y + i];
  g[q.w + i] = v;
}
// the layout the host needs, so it never restates it
kernel void layout(device int *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i) return;
#ifdef FULL
  o[0] = 1; o[2] = L_HX; o[3] = L_DR; o[4] = L_RX;
#else
  o[0] = 0; o[2] = L_X; o[3] = -1; o[4] = -1;
#endif
  o[1] = int(sizeof(Net) / 4); o[5] = L_DZ; o[6] = L_DHT; o[7] = L_XM; o[8] = L_DU; o[9] = L_U; o[10] = L_DY;
  o[11] = R_LAYERSIZE; o[12] = REC; o[13] = BS_SIZE; o[14] = int(sizeof(Layer) / 4); o[15] = R_DW; o[16] = R_N; o[17] = R_LAYER;
  o[18] = RS_SIZE; o[19] = SB_SIZE; o[20] = IN; o[21] = FS; o[22] = int(sizeof(Scenario)); o[23] = SB_HDR;
}

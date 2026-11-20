// The routing policy (design/switched-multipath-dma.md), trained as in Kool, van Hoof & Welling 2019: an autoregressive
// decoder builds each solution one token at a time over masked choices, its logits clipped C th(q.k / sqrt d)
// ("Attention, learn to solve routing problems!"), trained by REINFORCE with each sample's baseline the mean of the
// instance's other samples ("Buy 4 REINFORCE samples, get a baseline for free!").
//
// A solution is a multicast tree from a commodity's source to its destinations, grown one lane at a time: a lane is
// legal when it leaves the tree, enters a node not in it, and that node is an uncovered destination or reaches one
// without entering the tree. Decoding stops once every destination is covered; nodes leading nowhere are pruned.
// Threemin: at each of EP decision points each commodity decodes KS = 3 trees by independent sampling, and its pieces
// until the next go round-robin over the three (duplicates allowed: a peaked policy gives one tree, a spread one three).
//
// The decoder: two recurrent layers on a 64-wide residual stream, each followed by a feed-forward block expanding to
// 192. Two arms of the same recurrence h_t = (1 - z_t) h_(t-1) + z_t h~_t:
//   FULL (the GRU): z = sigmoid(Wz [h_(t-1); x] + bz), r = sigmoid(Wr [h_(t-1); x] + br), h~ = th(Wh [r h_(t-1); x] + bh)
//   otherwise (minGRU): z = sigmoid(Wz x + bz), h~ = th(Wh x + bh): linear in h, so h_t = sum_(s<=t) exp(C_t - C_s)
//   z_s h~_s + exp(C_t) h_(-1), C_t = sum_(r<=t) log(1 - z_r): causal attention whose mask mod is the cumulative decay
//   (parallel over positions; here run step by step, the same numbers).
// Each commodity's three samples at a decision point decode together from its state, as independent draws from one
// distribution; it carries the mean of their final states to the next decision point. The input: the commodity's context and the key of the lane
// chosen last (a start vector first); the query, scored against each lane's key (a two-layer network of the lane's
// queue, one-piece delay and rate, each against the floor).
//
// The environment: lanes serve the bytes waiting on them by processor sharing at their rates, SLOTS slots a rollout;
// whole pieces are released at the source's out-lanes' rate; bytes land after the lane's latency and one page time;
// the result is the completion (interpolated within its slot) over the floor.
//
// Kernels: rollout (sample; record each decision) and backward (each tree's steps again from its entering state, then
// -log p of each recorded choice backpropagated through them, the state entering the tree and the chosen lanes' keys in
// the input constants; each decision's deltas, weighted by its rollout's advantage for each of two tasks, and inputs
// written). Each weight block's gradient, the sum over decisions of delta times input, is then a GEMM (MPS, host side).
#include <metal_stdlib>
using namespace metal;

constant constexpr int N = 6, L = 32, J = 6, KS = 3, EP = 4, TR = EP * KS, TMAX = N - 1, H = 12, SLOTS = 64, EPOCH = SLOTS / EP;
constant constexpr int D = 64, E = 192, CF = 8, IN = CF + D, LF = 4, T = 256, JD = J * D;
constant constexpr int JTN = J * TR * N, DEC = EP * KS * TMAX * J, PR = KS * J, PD = PR * D, Q = 4, PRP = (PR + 7) / 8 * 8, SG = T / 32;   // PRP: PR padded to whole 8x8 tiles   // PR: the rows decoded together, a commodity's three samples
constant constexpr float PAGE = 16384.0f, CLIP = 10.0f;

struct Scenario {
  float rate[L], lat[L], bg[L];
  uchar lsrc[L], ldst[L];
  uchar spt[J][N];   // each commodity's unloaded shortest-path tree: the in-lane at each node, 255 none
  uchar src[J], dest[J];
  float demand[J];
  float floor_ns, dt, size;
  uint commodities, lanes, nodes, pad;
};
#ifdef FULL
constant constexpr int GW = 2 * D;   // a gate's row: [h; x]
struct Layer { float z[D][GW], bz[D], r[D][GW], br[D], h[D][GW], bh[D], up[E][D], bu[E], down[D][E], bd[D]; };
constant constexpr int L_HX = 0, L_DZ = 128, L_DR = 192, L_DHT = 256, L_RX = 320, L_XM = 448, L_DU = 512, L_U = 704, L_DY = 896, R_LAYERSIZE = 960;
#else
constant constexpr int GW = D;
struct Layer { float z[D][GW], bz[D], h[D][GW], bh[D], up[E][D], bu[E], down[D][E], bd[D]; };
constant constexpr int L_X = 0, L_DZ = 64, L_DHT = 128, L_XM = 192, L_DU = 256, L_U = 448, L_DY = 640, R_LAYERSIZE = 704;
#endif
struct Net { float inp[D][IN], binp[D], start[D]; Layer layer[2]; float out[D][D]; float k1[D][LF], bk1[D], k2[D][D], bk2[D]; };
// one decision: the decoder's input, its layers' states before the step, which lanes were legal, the one chosen (-1 none)
struct Tape { float in[IN]; float h[2][D]; uint legal; int chosen; float pad[2]; };
// a decision's record for the outer kernel
constant constexpr int R_DX0 = 0, R_IN = 64, R_LAYER = 136, R_DQ = R_LAYER + 2 * R_LAYERSIZE, R_XF = R_DQ + D, R_DST = R_XF + D, REC = R_DST + D;
// a lane's key record at one decision point
constant constexpr int K_DK = 0, K_HID = 64, K_DHID = 128, K_PHI = 192, KREC = 196;

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

// each lane's key from its operands: k = k2 relu(k1 phi + bk1) + bk2 into kf, the hidden layer into hid
static void keys(device const Net &net, device const float *phi, device float *hid, device float *kf, uint tid) {
  for (int i = tid; i < L * D; i += T) {
    const int l = i / D, d = i % D; float v = net.bk1[d];
    for (int f = 0; f < LF; f++) v += net.k1[d][f] * phi[l * LF + f];
    hid[i] = max(v, 0.0f);
  }
  SYNC;
  mv<L>(&net.k2[0][0], D, D, D, hid, D, kf, D, net.bk2, false, tid);
  SYNC;
}

// one recurrent layer's gates for the PR rows: from x (the layer's input) and hp (the state before), z and h~ (and r,
// rh = r hp in FULL), all [PR][D]
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

// the rollout's scratch: keys, their hidden layer, the PR rows' states while decoding, each commodity's state between
constant constexpr int RS_KEY = 0, RS_HID = L * D, RS_HS = 2 * L * D, RS_HC = RS_HS + 2 * PRP * D, RS_SIZE = RS_HC + 2 * JD;
constant constexpr int PDP = PRP * D, BLK = PRP * E > 3 * PDP ? PRP * E : 3 * PDP;   // the input, the gates, the feed-forward hidden, the query and logits, in turn

kernel void rollout(device const Scenario *scen [[buffer(0)]], device const Net &net [[buffer(1)]],
                    device const uint *which [[buffer(2)]], device const uint *draw [[buffer(3)]],
                    device float *done_out [[buffer(4)]], device Tape *tape [[buffer(5)]], device float *phitape [[buffer(6)]],
                    device float *scratch [[buffer(7)]], device half *rings [[buffer(8)]], constant uint &mode [[buffer(9)]],
                    device float *loglik [[buffer(10)]],
                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup Scenario s;
  { device const uint *from = (device const uint *)&scen[which[r]]; threadgroup uint *to = (threadgroup uint *)&s;
    for (uint i = tid; i < sizeof(Scenario) / 4; i += T) to[i] = from[i]; }
  device half (*ring)[JTN] = (device half (*)[JTN])(rings + size_t(r) * H * JTN);
  device float *sc = scratch + size_t(r) * RS_SIZE, *kf = sc + RS_KEY, *hid = sc + RS_HID, *hs = sc + RS_HS, *hc = sc + RS_HC;
  threadgroup float arrived[JTN], sent[JTN];
  threadgroup uchar tl[J][TR][N];
  threadgroup float share[L], bgq[L], cap[L], want[L], partial[L][8];
  threadgroup int lag[L];
  threadgroup float assigned[J * TR], released[J], credit[J], outcap[J], frac[J * N], before[J * N];
  threadgroup uint pieces[J], tree[PR], covered[PR], legal[PR], adj[N];
  threadgroup float blk[BLK], x[PDP], lp[PR];
  threadgroup int chosen[PR];
  threadgroup float finished;
  SYNC;
  const float dt = s.dt, fl = s.floor_ns;
  const int nl = int(s.lanes), nj = int(s.commodities), nn = int(s.nodes);
  const uint key0 = draw[r];

  for (int e = tid; e < JTN; e += T) { arrived[e] = 0; sent[e] = 0; for (int q = 0; q < H; q++) ring[q][e] = 0; ((threadgroup uchar *)tl)[e] = 255; }
  for (int i = tid; i < 2 * JD; i += T) hc[i] = 0;
  if (tid < uint(L)) {
    const int l = tid;
    cap[l] = l < nl ? s.rate[l] * dt : 0; bgq[l] = l < nl ? s.bg[l] : 0; want[l] = bgq[l];
    lag[l] = l < nl ? min(int(ceil((s.lat[l] + PAGE / max(s.rate[l], 1e-6f)) / dt)), H - 1) : 1;
  }
  if (tid < uint(J)) {
    const int j = tid; float o = 0;
    for (int l = 0; l < nl; l++) if (s.lsrc[l] == s.src[j]) o += s.rate[l] * dt;
    outcap[j] = o; released[j] = 0; credit[j] = 0; pieces[j] = 0;
    for (int k = 0; k < TR; k++) assigned[j * TR + k] = 0;
    for (int v = 0; v < N; v++) { frac[j * N + v] = 0; before[j * N + v] = 0; }
  }
  if (tid == 0) finished = SLOTS;
  if (tid < uint(PR)) lp[tid] = 0;
  if (tid < uint(N)) { uint m = 0; for (int l = 0; l < nl; l++) if (s.lsrc[l] == int(tid)) m |= 1u << s.ldst[l]; adj[tid] = m; }
  SYNC;

  for (int t = 0; t < SLOTS; t++) {
    if (t % EPOCH == 0) {
      const int ep = t / EPOCH;
      device float *phi = phitape + (size_t(r) * EP + ep) * L * LF;
      if (tid < uint(L)) {
        const int l = tid;
        float f[LF] = {0, 0, 0, 0};
        if (l < nl) {
          f[0] = log(1 + 64 * want[l] / max(s.rate[l], 1e-6f) / fl);
          f[1] = (s.lat[l] + PAGE / max(s.rate[l], 1e-6f)) / fl;
          f[2] = log(max(s.rate[l] * fl / max(s.size, 1.0f), 1e-6f));
          f[3] = 1;
        }
        for (int q = 0; q < LF; q++) phi[l * LF + q] = f[q];
      }
      SYNC;
      keys(net, phi, hid, kf, tid);
      // a commodity's three samples, decoded together from its state: row p = m J + j
      for (int i = tid; i < 2 * PDP; i += T) { const int layer = i / PDP, p = (i % PDP) / D, d = i % D; hs[i] = p < PR ? hc[layer * JD + (p % J) * D + d] : 0; }
      if (tid < uint(PR)) { tree[tid] = 1u << s.src[tid % J]; covered[tid] = 0; }
      SYNC;
      for (int step = 0; step < TMAX; step++) {
        for (int i = tid; i < PR * IN; i += T) {
          const int p = i / IN, c = i % IN, j = p % J; float v;
          if (c < CF) {
            const float ctx[CF] = {log(1 + s.size / PAGE) / 8, released[j] / max(s.demand[j], 1.0f), float(t) / SLOTS, float(ep) / EP,
                                   float(nn) / 8, float(popcount(uint(s.dest[j]))) / 8, float(popcount(tree[p])) / 8, 1};
            v = ctx[c];
          } else v = step == 0 ? net.start[c - CF] : kf[chosen[p] * D + c - CF];
          blk[i] = v;
        }
        SYNC;
        device Tape *tp = tape + size_t(r) * DEC + (ep * TMAX + step) * PR;
        for (int i = tid; i < PR * (IN + 2 * D); i += T) {   // the tape: input and state before the step
          const int p = i / (IN + 2 * D), c = i % (IN + 2 * D);
          if (c < IN) tp[p].in[c] = blk[p * IN + c]; else tp[p].h[(c - IN) / D][(c - IN) % D] = hs[((c - IN) / D) * PDP + p * D + (c - IN) % D];
        }
#ifndef NODECODE   // a measurement build: the simulator and the masks without the network
        mm<PRP>(&net.inp[0][0], IN, D, IN, blk, IN, x, D, net.binp, false, tid);
        SYNC;
        for (int layer = 0; layer < 2; layer++) {
          device const Layer &w = net.layer[layer];
          device float *hl = hs + layer * PDP;
          threadgroup float *z = blk, *c = blk + PDP;
          cell<PRP>(Tiles(), w, x, hl, z, c, c, blk + 2 * PDP, tid);
          for (int i = tid; i < PD; i += T) {
            const int p = i / D;
            const float hn = (1 - z[i]) * hl[i] + z[i] * c[i];
            x[i] += hn;
            if (p % J < nj && (uint(s.dest[p % J]) & ~covered[p])) hl[i] = hn;   // a row's state moves on its own decisions
          }
          SYNC;
          mm<PRP>(&w.up[0][0], D, E, D, x, D, blk, E, w.bu, false, tid);
          SYNC;
          for (int i = tid; i < PRP * E; i += T) blk[i] = max(blk[i], 0.0f);
          SYNC;
          mm<PRP>(&w.down[0][0], E, D, E, blk, E, x, D, w.bd, true, tid);
          SYNC;
        }
        threadgroup float *qv = blk, *logit = blk + PDP;
        mm<PRP>(&net.out[0][0], D, D, D, x, D, qv, D, nullptr, false, tid);
#else
        threadgroup float *qv = blk, *logit = blk + PDP;
        for (int i = tid; i < PD; i += T) qv[i] = 0;
#endif
        for (int i = tid; i < PR * L; i += T) {   // which lanes are legal: a SIMD group a row, a lane a lane
          const int p = i / L, l = i % L, j = p % J;
          const uint uncovered = uint(s.dest[j]) & ~covered[p];
          const bool ok = j < nj && l < nl && uncovered && (tree[p] >> s.lsrc[l] & 1) && !(tree[p] >> s.ldst[l] & 1) && reaches(adj, s.ldst[l], tree[p], uncovered);
          const uint lg = uint(uint64_t(simd_ballot(ok)));
          if (l == 0) legal[p] = lg;
        }
        SYNC;
        for (int i = tid; i < PR * L; i += T) {
          const int p = i / L, l = i % L;
          if (legal[p] >> l & 1) {
            threadgroup const float4 *qq = (threadgroup const float4 *)(qv + p * D); device const float4 *kk = (device const float4 *)(kf + l * D);
            float a = 0;
            for (int d = 0; d < D / 4; d++) a += dot(qq[d], kk[d]);
            logit[i] = CLIP * th(a / 8);
          }
          else logit[i] = -INFINITY;
        }
        SYNC;
        // the choice: sampled (mode 0), the most likely (1), along the shortest-path tree (2), or uniform (3)
        if (tid < uint(PR)) {
          const int p = tid, j = p % J, slot = ep * KS + p / J; int pick = -1;
          threadgroup float *lg = logit + p * L;
          if (legal[p]) {
            float most = -INFINITY; int best = -1;
            for (int l = 0; l < nl; l++) if ((legal[p] >> l & 1) && (best < 0 || lg[l] > most)) { most = lg[l]; best = l; }
            const float uu = float(fmix(key0 ^ uint(((ep * TMAX + step) * PR + p + 1) * 0x9e3779b1u)) >> 8) / 16777216.0f;
            if (mode == 1) pick = best;
            else if (mode == 2) { for (int l = 0; l < nl && pick < 0; l++) if ((legal[p] >> l & 1) && s.spt[j][s.ldst[l]] == l) pick = l; if (pick < 0) pick = best; }
            else if (mode == 3) { int target = min(int(uu * popcount(legal[p])), int(popcount(legal[p])) - 1);
              for (int l = 0; l < nl; l++) if (legal[p] >> l & 1) { if (target == 0) { pick = l; break; } target--; } }
            else {
              float sum = 0; for (int l = 0; l < nl; l++) if (legal[p] >> l & 1) sum += exp(lg[l] - most);
              float c = 0; pick = best;
              for (int l = 0; l < nl; l++) if (legal[p] >> l & 1) { c += exp(lg[l] - most) / sum; if (uu < c) { pick = l; break; } }
              lp[p] += lg[pick] - most - log(sum);
              tp[p].pad[0] = lg[pick] - most - log(sum); tp[p].pad[1] = sum;
            }
            tl[j][slot][s.ldst[pick]] = uchar(pick);
            tree[p] |= 1u << s.ldst[pick];
            if (s.dest[j] >> s.ldst[pick] & 1) covered[p] |= 1u << s.ldst[pick];
          }
          chosen[p] = max(pick, 0);
          tp[p].legal = legal[p]; tp[p].chosen = pick;
        }
        SYNC;
      }
      if (tid < uint(PR)) {   // prune what leads nowhere
        const int j = tid % J, slot = ep * KS + tid / J;
        for (int round = 0; round < N; round++)
          for (int v = 0; v < nn; v++) {
            if (tl[j][slot][v] == 255 || (s.dest[j] >> v & 1)) continue;
            bool child = false; for (int u2 = 0; u2 < nn; u2++) child = child || (tl[j][slot][u2] != 255 && s.lsrc[tl[j][slot][u2]] == v);
            if (!child) tl[j][slot][v] = 255;
          }
      }
      // each commodity carries the mean of its three samples' final states
      for (int i = tid; i < 2 * JD; i += T) {
        const int layer = i / JD, j = (i % JD) / D, d = i % D; float v = 0;
        for (int m = 0; m < KS; m++) v += hs[layer * PDP + (m * J + j) * D + d];
        hc[i] = v / KS;
      }
      SYNC;
    }

    if (tid < uint(J) && int(tid) < nj) {   // whole pieces, round-robin over this decision point's three trees
      const int j = tid, ep = t / EPOCH;
      credit[j] += outcap[j];
      while (released[j] < s.demand[j]) {
        const float piece = min(PAGE, s.demand[j] - released[j]);
        if (credit[j] < piece) break;
        credit[j] -= piece; released[j] += piece;
        assigned[j * TR + ep * KS + int(pieces[j] % KS)] += piece; pieces[j]++;
      }
      for (int k = 0; k < TR; k++) arrived[(j * TR + k) * N + s.src[j]] = assigned[j * TR + k];
    }
    SYNC;
    {   // each lane's queue: eight threads a lane, each an eighth of the entries
      const int l = tid / 8, part = tid % 8; float w = 0;
      if (l < nl) for (int e = part; e < JTN; e += 8) if (((threadgroup uchar *)tl)[e] == l) w += max(arrived[(e / N) * N + s.lsrc[l]] - sent[e], 0.0f);
      partial[l][part] = w;
    }
    SYNC;
    if (tid < uint(L) && int(tid) < nl) {
      const int l = tid; float w = bgq[l];
      for (int k = 0; k < 8; k++) w += partial[l][k];
      want[l] = w;
      const float sh = min(1.0f, cap[l] / max(w, 1e-3f));
      share[l] = sh; bgq[l] -= bgq[l] * sh;
    }
    SYNC;
    const int now = t % H;
    for (int e = tid; e < JTN; e += T) {
      const int lane = ((threadgroup uchar *)tl)[e];
      float inc = 0;
      if (lane != 255) inc = max(arrived[(e / N) * N + s.lsrc[lane]] - sent[e], 0.0f) * share[lane];
      const float dem = max(s.demand[min(e / (TR * N), max(nj - 1, 0))], 1.0f);
      const half part = half(inc / dem);
      sent[e] += float(part) * dem; ring[now][e] = part;
    }
    SYNC;
    for (int e = tid; e < JTN; e += T) {
      const int lane = ((threadgroup uchar *)tl)[e];
      if (lane != 255 && t >= lag[lane]) arrived[e] += float(ring[(t - lag[lane]) % H][e]) * max(s.demand[min(e / (TR * N), max(nj - 1, 0))], 1.0f);
    }
    SYNC;
    if (tid < uint(J * N)) {
      const int j = tid / N, d = tid % N;
      if (j < nj && (s.dest[j] >> d & 1)) {
        float g = 0; for (int k = 0; k < TR; k++) g += arrived[(j * TR + k) * N + d];
        frac[tid] = min(g / max(s.demand[j], 1e-6f), 1.0f);
      } else frac[tid] = 1;
    }
    SYNC;
    if (tid == 0 && finished == SLOTS) {
      bool all = true; for (int i = 0; i < J * N; i++) all = all && frac[i] >= 0.999f;
      if (all) {
        float within = 0;
        for (int i = 0; i < J * N; i++) if (before[i] < 0.999f) within = max(within, (0.999f - before[i]) / max(frac[i] - before[i], 1e-9f));
        finished = t + within;
      }
    }
    SYNC;
    if (tid < uint(J * N)) before[tid] = frac[tid];
  }
  SYNC;
  if (tid == 0) { done_out[r] = finished * dt / fl; float v = 0; for (int p = 0; p < PR; p++) v += lp[p]; loglik[r] = v; }
}

// is record field c a delta (scaled by each task's weight) rather than an input?
static bool delta(int c) {
  if (c < R_IN) return true;
  if (c >= R_DQ) return c < R_XF || c >= R_DST;
  if (c < R_LAYER) return false;
  const int f = (c - R_LAYER) % R_LAYERSIZE;
#ifdef FULL
  return (f >= L_DZ && f < L_RX) || (f >= L_DU && f < L_U) || f >= L_DY;
#else
  return (f >= L_DZ && f < L_XM) || (f >= L_DU && f < L_U) || f >= L_DY;
#endif
}

// rollout r's -log p of its recorded choices, backpropagated: at each decision point, the PR rows' steps forward again
// from their entering states (each step's activations kept), then back from the last step, the states' gradient carried
// through h_t = (1 - z_t) h_(t-1) + z_t h~_t. Writes each decision's record (rec), each lane's key record at each decision
// point (krec), and the rollout's summed log-likelihood (loglik). The deltas are linear in the seed, so each is written
// scaled by the rollout's weight for task 0 (rec, krec, beside the inputs) and task 1 (rec2, krec2, deltas only).
constant constexpr int S_KEY = 0, S_HID = L * D, S_DK = 2 * L * D, S_DH = 3 * L * D, S_G = S_DH + 2 * PD, S_GX = S_G + PD,
                       S_TW = S_GX + PD, S_U = S_TW + 2 * PD, S_IN = S_U + PR * E, S_STEPS = S_IN + TMAX * PR * IN;
constant constexpr int Y_HP = 0, Y_Z = PD, Y_R = 2 * PD, Y_C = 3 * PD, Y_RH = 4 * PD, Y_XM = 5 * PD, Y_UP = 6 * PD, Y_SIZE = 6 * PD + PR * E;
constant constexpr int P_X = 0, P_LAYER = 3 * PD, P_Q = P_LAYER + 2 * Y_SIZE, STEP = P_Q + PD, S_SIZE = S_STEPS + TMAX * STEP;

kernel void backward(device const Net &net [[buffer(0)]], device const Tape *tape [[buffer(1)]], device const float *phitape [[buffer(2)]],
                     device float *rec [[buffer(3)]], device float *krec [[buffer(4)]], device float *scratch [[buffer(5)]],
                     device float *loglik [[buffer(6)]], device const float *weight [[buffer(7)]],
                     device float *rec2 [[buffer(8)]], device float *krec2 [[buffer(9)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  device float *a = scratch + size_t(r) * S_SIZE;
  threadgroup float pr[PR][L];
  threadgroup int act[PR];
  threadgroup float lp[PR];
  if (tid < uint(PR)) lp[tid] = 0;
  const float w0 = weight[2 * r], w1 = weight[2 * r + 1];
  for (int ep = 0; ep < EP; ep++) {
    device const float *phi = phitape + (size_t(r) * EP + ep) * L * LF;
    keys(net, phi, a + S_HID, a + S_KEY, tid);
    for (int i = tid; i < L * D; i += T) a[S_DK + i] = 0;
    SYNC;
    device const Tape *tp0 = tape + size_t(r) * DEC + ep * TMAX * PR;
    for (int step = 0; step < TMAX; step++) {   // forward, each row's state chained from its entering state
      device const Tape *tp = tp0 + step * PR;
      device float *st = a + S_STEPS + step * STEP, *inb = a + S_IN + step * PR * IN;
      for (int i = tid; i < PR * IN; i += T) { const int p = i / IN, c = i % IN; inb[i] = (step == 0 && c >= CF) ? net.start[c - CF] : tp[p].in[c]; }
      for (int layer = 0; layer < 2; layer++) {
        device float *y = st + P_LAYER + layer * Y_SIZE;
        for (int i = tid; i < PD; i += T) {
          const int p = i / D, d = i % D;
          if (step == 0) y[Y_HP + i] = tp[p].h[layer][d];
          else {
            device const float *yp = a + S_STEPS + (step - 1) * STEP + P_LAYER + layer * Y_SIZE;
            y[Y_HP + i] = tp0[(step - 1) * PR + p].chosen >= 0 ? (1 - yp[Y_Z + i]) * yp[Y_HP + i] + yp[Y_Z + i] * yp[Y_C + i] : yp[Y_HP + i];
          }
        }
      }
      SYNC;
      mv<PR>(&net.inp[0][0], IN, D, IN, inb, IN, st + P_X, D, net.binp, false, tid);
      SYNC;
      for (int layer = 0; layer < 2; layer++) {
        device const Layer &w = net.layer[layer];
        device float *y = st + P_LAYER + layer * Y_SIZE, *xl = st + P_X + layer * PD, *xn = st + P_X + (layer + 1) * PD;
        cell<PR>(Rows(), w, xl, y + Y_HP, y + Y_Z, y + Y_R, y + Y_C, y + Y_RH, tid);
        for (int i = tid; i < PD; i += T) y[Y_XM + i] = xl[i] + (1 - y[Y_Z + i]) * y[Y_HP + i] + y[Y_Z + i] * y[Y_C + i];
        SYNC;
        mv<PR>(&w.up[0][0], D, E, D, y + Y_XM, D, y + Y_UP, E, w.bu, false, tid);
        SYNC;
        for (int i = tid; i < PR * E; i += T) a[S_U + i] = max(y[Y_UP + i], 0.0f);
        SYNC;
        mv<PR>(&w.down[0][0], E, D, E, a + S_U, E, xn, D, w.bd, false, tid);
        SYNC;
        for (int i = tid; i < PD; i += T) xn[i] += y[Y_XM + i];
        SYNC;
      }
      mv<PR>(&net.out[0][0], D, D, D, st + P_X + 2 * PD, D, st + P_Q, D, nullptr, false, tid);
      SYNC;
    }
    for (int i = tid; i < 2 * PD; i += T) a[S_DH + i] = 0;
    SYNC;
    for (int step = TMAX - 1; step >= 0; step--) {   // backward
      device const Tape *tp = tp0 + step * PR;
      device float *st = a + S_STEPS + step * STEP, *inb = a + S_IN + step * PR * IN;
      device float *rc = rec + (size_t(r) * DEC + (ep * TMAX + step) * PR) * REC;
      if (tid < uint(PR)) act[tid] = tp[tid].chosen;
      SYNC;
      for (int i = tid; i < PR * L; i += T) {
        const int p = i / L, l = i % L;
        if (act[p] >= 0 && (tp[p].legal >> l & 1)) { float v = 0; for (int d = 0; d < D; d++) v += st[P_Q + p * D + d] * a[S_KEY + l * D + d]; pr[p][l] = v / 8; }
        else pr[p][l] = -INFINITY;
      }
      SYNC;
      if (tid < uint(PR)) {   // d(-log p)/d(q.k), the clip and the 1/8 folded in
        const int p = tid;
        if (act[p] >= 0) {
          float most = -INFINITY; for (int l = 0; l < L; l++) if (pr[p][l] > -INFINITY) most = max(most, CLIP * th(pr[p][l]));
          float sum = 0; for (int l = 0; l < L; l++) if (pr[p][l] > -INFINITY) sum += exp(CLIP * th(pr[p][l]) - most);
          lp[p] += CLIP * th(pr[p][act[p]]) - most - log(sum);
          for (int l = 0; l < L; l++) {
            if (pr[p][l] == -INFINITY) { pr[p][l] = 0; continue; }
            const float t = th(pr[p][l]), pl = exp(CLIP * t - most) / sum;
            pr[p][l] = (pl - (l == act[p] ? 1.0f : 0.0f)) * CLIP * (1 - t * t) / 8;
          }
        } else for (int l = 0; l < L; l++) pr[p][l] = 0;
      }
      SYNC;
      for (int i = tid; i < PD; i += T) {   // dq; the stack's output
        const int p = i / D, d = i % D; float v = 0;
        for (int l = 0; l < L; l++) v += pr[p][l] * a[S_KEY + l * D + d];
        rc[p * REC + R_DQ + d] = v; rc[p * REC + R_XF + d] = st[P_X + 2 * PD + i];
      }
      for (int i = tid; i < L * D; i += T) {   // the keys' deltas
        const int l = i / D, d = i % D; float v = 0;
        for (int p = 0; p < PR; p++) v += pr[p][l] * st[P_Q + p * D + d];
        a[S_DK + i] += v;
      }
      SYNC;
      device float *g = a + S_G, *gx = a + S_GX, *tw = a + S_TW;
      mvt<PR>(&net.out[0][0], D, D, D, rc + R_DQ, REC, g, D, false, tid);
      SYNC;
      for (int layer = 1; layer >= 0; layer--) {
        device const Layer &w = net.layer[layer];
        device float *y = st + P_LAYER + layer * Y_SIZE, *xl = st + P_X + layer * PD, *dh = a + S_DH + layer * PD;
        device float *lr = rc + R_LAYER + layer * R_LAYERSIZE;
        for (int i = tid; i < PD; i += T) {
          const int p = i / D, d = i % D;
          lr[p * REC + L_DY + d] = g[i]; lr[p * REC + L_XM + d] = y[Y_XM + i];
#ifdef FULL
          lr[p * REC + L_HX + d] = y[Y_HP + i]; lr[p * REC + L_HX + D + d] = xl[i];
          lr[p * REC + L_RX + d] = y[Y_RH + i]; lr[p * REC + L_RX + D + d] = xl[i];
#else
          lr[p * REC + L_X + d] = xl[i];
#endif
        }
        for (int i = tid; i < PR * E; i += T) { const int p = i / E, e = i % E; lr[p * REC + L_U + e] = max(y[Y_UP + i], 0.0f); }
        SYNC;
        mvt<PR>(&w.down[0][0], E, D, E, lr + L_DY, REC, lr + L_DU, REC, false, tid);   // du = down^T dy, through the relu
        SYNC;
        for (int i = tid; i < PR * E; i += T) { const int p = i / E, e = i % E; if (y[Y_UP + i] <= 0) lr[p * REC + L_DU + e] = 0; }
        for (int i = tid; i < PD; i += T) gx[i] = g[i];
        SYNC;
        mvt<PR>(&w.up[0][0], D, E, D, lr + L_DU, REC, gx, D, true, tid);   // dxm = dy + up^T du
        SYNC;
        for (int i = tid; i < PD; i += T) {   // the state: dhn = dxm + carried
          const int p = i / D, d = i % D;
          const float dn = gx[i] + dh[i], z = y[Y_Z + i], c = y[Y_C + i], hp = y[Y_HP + i];
          lr[p * REC + L_DZ + d] = dn * (c - hp) * z * (1 - z);
          lr[p * REC + L_DHT + d] = dn * z * (1 - c * c);
          dh[i] = dn * (1 - z);
          g[i] = gx[i];   // dx: the residual path, plus through the gates below
        }
        SYNC;
#ifdef FULL
        mvt<PR>(&w.h[0][0], GW, D, GW, lr + L_DHT, REC, tw, GW, false, tid);   // [d rh; d x]
        SYNC;
        for (int i = tid; i < PD; i += T) {
          const int p = i / D, d = i % D;
          const float drh = tw[p * GW + d], rr = y[Y_R + i];
          lr[p * REC + L_DR + d] = drh * y[Y_HP + i] * rr * (1 - rr);
          dh[i] += drh * rr; g[i] += tw[p * GW + D + d];
        }
        SYNC;
        mvt<PR>(&w.z[0][0], GW, D, GW, lr + L_DZ, REC, tw, GW, false, tid);
        SYNC;
        mvt<PR>(&w.r[0][0], GW, D, GW, lr + L_DR, REC, tw, GW, true, tid);
        SYNC;
        for (int i = tid; i < PD; i += T) { const int p = i / D, d = i % D; dh[i] += tw[p * GW + d]; g[i] += tw[p * GW + D + d]; }
        SYNC;
#else
        (void)tw;
        mvt<PR>(&w.z[0][0], GW, D, GW, lr + L_DZ, REC, g, D, true, tid);
        SYNC;
        mvt<PR>(&w.h[0][0], GW, D, GW, lr + L_DHT, REC, g, D, true, tid);
        SYNC;
#endif
      }
      for (int i = tid; i < PR * IN; i += T) { const int p = i / IN, c = i % IN; rc[p * REC + R_IN + c] = inb[i]; }
      for (int i = tid; i < PD; i += T) {
        const int p = i / D, d = i % D; rc[p * REC + R_DX0 + d] = g[i];
        float v = 0;
        if (step == 0) for (int row = 0; row < D; row++) v += net.inp[row][CF + d] * g[p * D + row];
        rc[p * REC + R_DST + d] = v;   // the start vector's gradient, the first step only
      }
      SYNC;
      for (int i = tid; i < PR * REC; i += T) {
        const int c = i % REC;
        if (act[i / REC] < 0) { rc[i] = 0; if (delta(c)) rec2[(rc - rec) + i] = 0; }   // a step not taken contributes nothing
        else if (delta(c)) { const float v = rc[i]; rec2[(rc - rec) + i] = v * w1; rc[i] = v * w0; }
      }
      SYNC;
    }
    device float *kr = krec + (size_t(r) * EP + ep) * L * KREC, *kr2 = krec2 + (kr - krec);
    for (int i = tid; i < L * D; i += T) {
      const int l = i / D, d = i % D;
      kr[l * KREC + K_DK + d] = a[S_DK + i] * w0; kr2[l * KREC + K_DK + d] = a[S_DK + i] * w1; kr[l * KREC + K_HID + d] = a[S_HID + i];
    }
    for (int i = tid; i < L * LF; i += T) kr[(i / LF) * KREC + K_PHI + i % LF] = phi[i];
    for (int i = tid; i < L * D; i += T) {
      const int l = i / D, d = i % D; float v = 0;
      for (int q = 0; q < D; q++) v += net.k2[q][d] * a[S_DK + l * D + q];
      kr[l * KREC + K_DHID + d] = a[S_HID + i] > 0 ? v * w0 : 0; kr2[l * KREC + K_DHID + d] = a[S_HID + i] > 0 ? v * w1 : 0;
    }
    SYNC;
  }
  if (tid == 0) { float v = 0; for (int p = 0; p < PR; p++) v += lp[p]; loglik[r] = v; }
}

// one weight block's split-K partial gradients, summed: g[at + i] = sum_b part[base + b size + i] (q: base, size, batches, at)
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
  o[11] = R_LAYERSIZE; o[12] = REC; o[13] = S_SIZE; o[14] = int(sizeof(Layer) / 4); o[15] = R_DQ; o[16] = R_XF; o[17] = R_DST; o[18] = RS_SIZE;
}

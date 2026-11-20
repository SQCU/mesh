// The routing policy (design/switched-multipath-dma.md), trained as in Kool, van Hoof & Welling 2019: an autoregressive
// decoder builds each multicast tree a lane at a time over the legal lanes, its logits clipped C tanh(q.k / sqrt d)
// ("Attention, learn to solve routing problems!"), trained by REINFORCE with each sample's baseline the mean of the
// instance's other samples ("Buy 4 REINFORCE samples, get a baseline for free!").
//
// The action. A commodity's pieces (pages of its rank's bytes) go in groups of GROUP; at each group's decision point the
// commodity samples KS = 3 trees t_1..t_3 from the policy, and each piece of the group draws its own tree,
// k_i ~ Categorical(w), w_k = p(t_k) / sum_m p(t_m): every piece a draw from the policy's own distribution over trees,
// restricted to the sampled support (KS^n assignments for n pieces). The action's log-likelihood is exact:
// log p = sum_k log p(t_k) + sum_i log w_(k_i), whose gradient is sum_k (1 + n_k - n w_k) d log p(t_k) (n_k the pieces
// on tree k): each tree's decisions are backpropagated with that seed.
//
// A tree, grown one lane at a time: a lane is legal when it leaves the tree, enters a node not in it, and that node is an
// uncovered destination or reaches one without entering the tree; decoding stops once every destination is covered;
// nodes leading nowhere are pruned. The decoder: two recurrent layers on a 64-wide residual stream, each followed by a
// feed-forward block expanding to 192. Two arms of h_t = (1 - z_t) h_(t-1) + z_t h~_t:
//   FULL (the GRU): z = sigmoid(Wz [h_(t-1); x] + bz), r = sigmoid(Wr [h_(t-1); x] + br), h~ = tanh(Wh [r h_(t-1); x] + bh)
//   otherwise (minGRU): z = sigmoid(Wz x + bz), h~ = tanh(Wh x + bh): linear in h, so h_t = sum_(s<=t) exp(C_t - C_s)
//   z_s h~_s + exp(C_t) h_(-1), C_t = sum_(r<=t) log(1 - z_r): causal attention whose mask mod is the cumulative decay.
// A commodity's three samples decode together from its state; it carries the mean of their final states on.
//
// The environment, the max-plus FIFO recursion over discrete pieces with tests/switch.c's semantics: every lane its
// cables, a pair's rate shared equally by them; a piece crossing a lane takes whichever cable frees first,
// start = max(arrival, cable free), at the far node start + bytes / cable rate + latency, the cable running at the
// duplex share of its rate while the reverse direction's same cable is busy; a node forwards a piece the moment it
// lands, on its children in the piece's tree. Group b is posted at its source at tau_b, when every commodity's group b-1
// has started on the wire there (tau_0 = 0; mode 4 posts every group at 0, as switch.c does). Events in time order.
// Three costs a rollout: throughput (the completion over the floor), latency (the mean over pieces and destinations of
// arrival less posting, over the floor), congestion (the hops' queueing wait over their sending time).
//
// Kernels: rollout (decode, draw, simulate; record each decision), backward (each decision point's rows again from
// their entering states, then each tree's -log p backpropagated with its seed; each decision's deltas and inputs
// written). Each weight block's gradient, a rollout's sum over decisions of delta times input, is a GEMM (MPS, host),
// and the tasks' gradients the rollouts' weighted sums.
#include <metal_stdlib>
using namespace metal;

constant constexpr int N = 6, L = 32, J = 6, KS = 3, TMAX = N - 1, GROUP = 64, EPMAX = 12, NPMAX = EPMAX * GROUP, CB = 4;
constant constexpr int TR = EPMAX * KS, D = 64, E = 192, CF = 8, IN = CF + D, LF = 4, T = 256, JD = J * D;
constant constexpr int PR = KS * J, PD = PR * D, Q = 4, PRP = (PR + 7) / 8 * 8, SG = T / 32;   // PR rows decoded together; PRP padded to 8x8 tiles
constant constexpr int DEC = EPMAX * TMAX * PR, HEAPN = 24576;
constant constexpr float PAGE = 16384.0f, CLIP = 10.0f;

struct Scenario {
  float rate[L], lat[L], bg[L];   // a lane's rate (all its cables), latency, and bytes queued on it at the start
  uchar lsrc[L], ldst[L], cables[L];
  uchar spt[J][N];                // each commodity's unloaded shortest-path tree: the in-lane at each node, 255 none
  uchar src[J], dest[J];
  float demand[J];
  float floor_ns, size, duplex;
  uint commodities, lanes, nodes;
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
// a decision's record for the GEMMs
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


// the rollout's scratch: keys, their hidden layer, the rows' states while decoding, each commodity's state between
constant constexpr int RS_KEY = 0, RS_HID = L * D, RS_HS = 2 * L * D, RS_HC = RS_HS + 2 * PRP * D, RS_SIZE = RS_HC + 2 * JD;
constant constexpr int PDP = PRP * D, BLK = PRP * E > 3 * PDP ? PRP * E : 3 * PDP;   // the input, the gates, the feed-forward hidden, the query and logits, in turn
// the simulator's scratch a rollout, in uints: the event heap (time, piece << 3 | node), each piece's tree slot
constant constexpr int SIM_HEAP = 0, SIM_SLOT = 2 * HEAPN, SIM_SIZE = SIM_SLOT + J * NPMAX;

// the event heap: switch.c's push and pop, so equal times leave in the same order
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

// the simulator, run by one thread: the fabric's cables, each piece's tree, the heap, and what the costs accumulate
struct Fabric {
  threadgroup const Scenario *s; threadgroup float (*free)[CB]; threadgroup const uchar *rev; threadgroup const uchar (*tl)[TR][N];
  threadgroup float *laststart; threadgroup const float *taus; device const uint *slot; device uint *hp;
  uint n, arrivals; float T, lat, wait, serv; int np, nn;
};
// piece i of commodity j at node u at time t: sent on each of its tree's lanes out of u
static void forward(thread Fabric &f, float t, int u, int j, int i) {
  threadgroup const Scenario &s = *f.s;
  const int sl = f.slot[j * NPMAX + i];
  const float bytes = i + 1 < f.np ? PAGE : s.size - PAGE * (f.np - 1);
  for (int v = 0; v < f.nn; v++) {
    const int l = f.tl[j][sl][v];
    if (l == 255 || s.lsrc[l] != u) continue;
    int c = 0; for (int k = 1; k < s.cables[l]; k++) if (f.free[l][k] < f.free[l][c]) c = k;
    const float start = max(f.free[l][c], t);
    const int rl = f.rev[l];
    const bool both = rl != 255 && c < s.cables[rl] && f.free[rl][c] > start;
    const float time = bytes / (s.rate[l] / s.cables[l] * (both ? s.duplex : 1.0f));
    f.free[l][c] = start + time; f.wait += start - t; f.serv += time;
    if (u == s.src[j]) f.laststart[j] = max(f.laststart[j], start);
    push(f.hp, f.n, start + time + s.lat[l], uint((j * NPMAX + i) << 3 | v));
  }
}
// every event up to `until`, in time order: a piece landing; at a destination counted; forwarded on
static void advance(thread Fabric &f, float until) {
  while (f.n > 0 && as_type<float>(f.hp[0]) <= until) {
    float t; uint pay; pop(f.hp, f.n, t, pay);
    const int v = pay & 7, j = (pay >> 3) / NPMAX, i = (pay >> 3) % NPMAX;
    if (f.s->dest[j] >> v & 1) { f.T = max(f.T, t); f.lat += t - f.taus[i / GROUP]; f.arrivals++; }
    forward(f, t, v, j, i);
  }
}

kernel void rollout(device const Scenario *scen [[buffer(0)]], device const Net &net [[buffer(1)]],
                    device const uint *which [[buffer(2)]], device const uint *draw [[buffer(3)]],
                    device float4 *cost [[buffer(4)]], device Tape *tape [[buffer(5)]], device float *phitape [[buffer(6)]],
                    device float *scratch [[buffer(7)]], device uint *simbuf [[buffer(8)]], constant uint &mode [[buffer(9)]],
                    device float *loglik [[buffer(10)]], device uint *counts [[buffer(11)]], device uint *nbuf [[buffer(12)]],
                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup Scenario s;
  { device const uint *from = (device const uint *)&scen[which[r]]; threadgroup uint *to = (threadgroup uint *)&s;
    for (uint i = tid; i < sizeof(Scenario) / 4; i += T) to[i] = from[i]; }
  device float *sc = scratch + size_t(r) * RS_SIZE, *kf = sc + RS_KEY, *hid = sc + RS_HID, *hs = sc + RS_HS, *hc = sc + RS_HC;
  device uint *sim = simbuf + size_t(r) * SIM_SIZE, *slot = sim + SIM_SLOT;
  threadgroup float free_at[L][CB], taus[EPMAX], laststart[J];
  threadgroup uchar rev[L], tl[J][TR][N];
  threadgroup uint tree[PR], covered[PR], legal[PR], adj[N];
  threadgroup atomic_uint cnt[PR];
  threadgroup float blk[BLK], x[PDP], lp[PR], wk[PR], llj[J];
  threadgroup int chosen[PR];
  SYNC;
  const float fl = s.floor_ns;
  const int nl = int(s.lanes), nj = int(s.commodities), nn = int(s.nodes);
  const int np = max(1, int(ceil(s.size / PAGE))), nb = (np + GROUP - 1) / GROUP;
  const uint key0 = draw[r];

  for (int e = tid; e < J * TR * N; e += T) ((threadgroup uchar *)tl)[e] = 255;
  for (int i = tid; i < 2 * JD; i += T) hc[i] = 0;
  for (int i = tid; i < L * CB; i += T) { const int l = i / CB; free_at[l][i % CB] = l < nl ? s.bg[l] / s.rate[l] : 0; }
  if (tid < uint(L)) {
    const int l = tid; int b = 255;
    for (int m = 0; m < nl; m++) if (l < nl && s.lsrc[m] == s.ldst[l] && s.ldst[m] == s.lsrc[l]) b = m;
    rev[l] = uchar(b);
  }
  if (tid < uint(J)) { laststart[tid] = 0; llj[tid] = 0; }
  if (tid < uint(N)) { uint m = 0; for (int l = 0; l < nl; l++) if (s.lsrc[l] == int(tid)) m |= 1u << s.ldst[l]; adj[tid] = m; }
  SYNC;
  Fabric f = {&s, free_at, rev, tl, laststart, taus, slot, sim + SIM_HEAP, 0, 0, 0, 0, 0, 0, np, nn};

  for (int b = 0; b < nb; b++) {
    // the decision point: when every commodity's last group has started on the wire at its source
    if (tid == 0) {
      float tau = 0;
      if (b > 0 && mode != 4) { tau = INFINITY; for (int j = 0; j < nj; j++) tau = min(tau, laststart[j]); tau = max(tau, taus[b - 1]); advance(f, tau); }
      taus[b] = tau;
    }
    SYNC;
    const float tau = taus[b];
    device float *phi = phitape + (size_t(r) * EPMAX + b) * L * LF;
    if (tid < uint(L)) {   // each lane's operands against the floor: its backlog then, its one-piece delay, its rate
      const int l = tid;
      float q[LF] = {0, 0, 0, 0};
      if (l < nl) {
        float back = 0; for (int c = 0; c < s.cables[l]; c++) back += max(free_at[l][c] - tau, 0.0f);
        q[0] = log(1 + 64 * back / s.cables[l] / fl);
        q[1] = (s.lat[l] + PAGE * s.cables[l] / s.rate[l]) / fl;
        q[2] = log(max(s.rate[l] * fl / max(s.size, 1.0f), 1e-6f));
        q[3] = 1;
      }
      for (int k = 0; k < LF; k++) phi[l * LF + k] = q[k];
    }
    SYNC;
    keys(net, phi, hid, kf, tid);
    for (int i = tid; i < 2 * PDP; i += T) { const int layer = i / PDP, p = (i % PDP) / D, d = i % D; hs[i] = p < PR ? hc[layer * JD + (p % J) * D + d] : 0; }
    if (tid < uint(PR)) { tree[tid] = 1u << s.src[tid % J]; covered[tid] = 0; lp[tid] = 0; atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed); }
    SYNC;
    for (int step = 0; step < TMAX; step++) {
      for (int i = tid; i < PR * IN; i += T) {
        const int p = i / IN, c = i % IN, j = p % J; float v;
        if (c < CF) {
          const float ctx[CF] = {log(1 + s.size / PAGE) / 8, float(b * GROUP) / np, tau / fl, float(b) / EPMAX,
                                 float(nn) / 8, float(popcount(uint(s.dest[j]))) / 8, float(popcount(tree[p])) / 8, 1};
          v = ctx[c];
        } else v = step == 0 ? net.start[c - CF] : kf[chosen[p] * D + c - CF];
        blk[i] = v;
      }
      SYNC;
      device Tape *tp = tape + size_t(r) * DEC + (b * TMAX + step) * PR;
      for (int i = tid; i < PR * (IN + 2 * D); i += T) {   // the tape: input and state before the step
        const int p = i / (IN + 2 * D), c = i % (IN + 2 * D);
        if (c < IN) tp[p].in[c] = blk[p * IN + c]; else tp[p].h[(c - IN) / D][(c - IN) % D] = hs[((c - IN) / D) * PDP + p * D + (c - IN) % D];
      }
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
        } else logit[i] = -INFINITY;
      }
      SYNC;
      // the lane: sampled (mode 0), the most likely (1), along the shortest-path tree (2, 4), or uniform (3)
      if (tid < uint(PR)) {
        const int p = tid, j = p % J, sl = b * KS + p / J; int pick = -1;
        threadgroup float *lg = logit + p * L;
        if (legal[p]) {
          float most = -INFINITY; int best = -1;
          for (int l = 0; l < nl; l++) if ((legal[p] >> l & 1) && (best < 0 || lg[l] > most)) { most = lg[l]; best = l; }
          const float uu = float(fmix(key0 ^ uint(((b * TMAX + step) * PR + p + 1) * 0x9e3779b1u)) >> 8) / 16777216.0f;
          float sum = 0; for (int l = 0; l < nl; l++) if (legal[p] >> l & 1) sum += exp(lg[l] - most);
          if (mode == 1) pick = best;
          else if (mode == 2 || mode == 4) { for (int l = 0; l < nl && pick < 0; l++) if ((legal[p] >> l & 1) && s.spt[j][s.ldst[l]] == l) pick = l; if (pick < 0) pick = best; }
          else if (mode == 3) { int target = min(int(uu * popcount(legal[p])), int(popcount(legal[p])) - 1);
            for (int l = 0; l < nl; l++) if (legal[p] >> l & 1) { if (target == 0) { pick = l; break; } target--; } }
          else {
            float c = 0; pick = best;
            for (int l = 0; l < nl; l++) if (legal[p] >> l & 1) { c += exp(lg[l] - most) / sum; if (uu < c) { pick = l; break; } }
          }
          lp[p] += lg[pick] - most - log(sum);
          tl[j][sl][s.ldst[pick]] = uchar(pick);
          tree[p] |= 1u << s.ldst[pick];
          if (s.dest[j] >> s.ldst[pick] & 1) covered[p] |= 1u << s.ldst[pick];
        }
        chosen[p] = max(pick, 0);
        tp[p].legal = legal[p]; tp[p].chosen = pick;
      }
      SYNC;
    }
    if (tid < uint(PR)) {   // prune what leads nowhere; each tree's share of the pieces, w = p(t) / sum of the commodity's three
      const int p = tid, j = p % J, sl = b * KS + p / J;
      for (int round = 0; round < N; round++)
        for (int v = 0; v < nn; v++) {
          if (tl[j][sl][v] == 255 || (s.dest[j] >> v & 1)) continue;
          bool child = false; for (int u2 = 0; u2 < nn; u2++) child = child || (tl[j][sl][u2] != 255 && s.lsrc[tl[j][sl][u2]] == v);
          if (!child) tl[j][sl][v] = 255;
        }
      float most = -INFINITY; for (int m = 0; m < KS; m++) most = max(most, lp[m * J + j]);
      float sum = 0; for (int m = 0; m < KS; m++) sum += exp(lp[m * J + j] - most);
      wk[p] = exp(lp[p] - most) / sum;
    }
    SYNC;
    for (int idx = tid; idx < J * GROUP; idx += T) {   // each piece of the group draws its tree
      const int j = idx / GROUP, i = b * GROUP + idx % GROUP;
      if (j >= nj || i >= np) continue;
      const float uu = float(fmix(key0 ^ uint((0x51ed27u + j * NPMAX + i) * 0x85ebca6bu)) >> 8) / 16777216.0f;
      int k = 0;
      if (mode == 0) { float c = 0; k = KS - 1; for (int m = 0; m < KS; m++) { c += wk[m * J + j]; if (uu < c) { k = m; break; } } }
      else if (mode == 1) { for (int m = 1; m < KS; m++) if (wk[m * J + j] > wk[k * J + j]) k = m; }
      else if (mode == 3) k = min(int(uu * KS), KS - 1);
      slot[j * NPMAX + i] = uint(b * KS + k);
      atomic_fetch_add_explicit(&cnt[k * J + j], 1u, memory_order_relaxed);
    }
    for (int i = tid; i < 2 * JD; i += T) {   // each commodity carries the mean of its three samples' final states
      const int layer = i / JD, j = (i % JD) / D, d = i % D; float v = 0;
      for (int m = 0; m < KS; m++) v += hs[layer * PDP + (m * J + j) * D + d];
      hc[i] = v / KS;
    }
    SYNC;
    if (tid < uint(PR)) counts[(size_t(r) * EPMAX + b) * PR + tid] = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
    if (tid < uint(J) && int(tid) < nj) {   // the action's log-likelihood: the trees, and each piece's draw among them
      const int j = tid; float most = -INFINITY; for (int m = 0; m < KS; m++) most = max(most, lp[m * J + j]);
      float sum = 0; for (int m = 0; m < KS; m++) sum += exp(lp[m * J + j] - most);
      const float lse = most + log(sum);
      for (int m = 0; m < KS; m++) llj[j] += lp[m * J + j] + float(atomic_load_explicit(&cnt[m * J + j], memory_order_relaxed)) * (lp[m * J + j] - lse);
    }
    if (tid == 0) for (int j = 0; j < nj; j++) for (int i = b * GROUP; i < min(np, (b + 1) * GROUP); i++) forward(f, tau, s.src[j], j, i);   // posted
    SYNC;
  }
  if (tid == 0) {
    advance(f, INFINITY);
    cost[r] = float4(f.T / fl, f.lat / max(float(f.arrivals), 1.0f) / fl, f.wait / max(f.serv, 1e-9f), float(f.arrivals));
    float v = 0; for (int j = 0; j < J; j++) v += llj[j];
    loglik[r] = v; nbuf[r] = uint(nb);
  }
}

// rollout r's -log p of its action, backpropagated. At each decision point: the rows' steps forward again from their
// entering states (each step's activations and scores kept), each tree's log-probability, its seed 1 + n_k - n w_k;
// then back from the last step, the states' gradient carried through h_t = (1 - z_t) h_(t-1) + z_t h~_t, each row's
// scores' gradient times its seed. Writes each decision's record (rec), each lane's key record (krec), zeros past the
// rollout's decision points, and the action's log-likelihood (loglik).
constant constexpr int S_KEY = 0, S_HID = L * D, S_DK = 2 * L * D, S_DH = 3 * L * D, S_G = S_DH + 2 * PD, S_GX = S_G + PD,
                       S_TW = S_GX + PD, S_U = S_TW + 2 * PD, S_SC = S_U + PR * E, S_IN = S_SC + TMAX * PR * L, S_STEPS = S_IN + TMAX * PR * IN;
constant constexpr int Y_HP = 0, Y_Z = PD, Y_R = 2 * PD, Y_C = 3 * PD, Y_RH = 4 * PD, Y_XM = 5 * PD, Y_UP = 6 * PD, Y_SIZE = 6 * PD + PR * E;
constant constexpr int P_X = 0, P_LAYER = 3 * PD, P_Q = P_LAYER + 2 * Y_SIZE, STEP = P_Q + PD, S_SIZE = S_STEPS + TMAX * STEP;

kernel void backward(device const Net &net [[buffer(0)]], device const Tape *tape [[buffer(1)]], device const float *phitape [[buffer(2)]],
                     device float *rec [[buffer(3)]], device float *krec [[buffer(4)]], device float *scratch [[buffer(5)]],
                     device float *loglik [[buffer(6)]], device const uint *counts [[buffer(7)]], device const uint *nbuf [[buffer(8)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  device float *a = scratch + size_t(r) * S_SIZE;
  threadgroup float pr[PR][L], lpe[PR], seed[PR], llj[J];
  threadgroup int act[PR];
  const int nb = int(nbuf[r]);
  if (tid < uint(J)) llj[tid] = 0;
  for (int ep = 0; ep < EPMAX; ep++) {
    device float *rc0 = rec + (size_t(r) * DEC + ep * TMAX * PR) * REC, *kr = krec + (size_t(r) * EPMAX + ep) * L * KREC;
    if (ep >= nb) {   // past the rollout's decision points: nothing
      for (int i = tid; i < TMAX * PR * REC; i += T) rc0[i] = 0;
      for (int i = tid; i < L * KREC; i += T) kr[i] = 0;
      continue;
    }
    device const float *phi = phitape + (size_t(r) * EPMAX + ep) * L * LF;
    keys(net, phi, a + S_HID, a + S_KEY, tid);
    for (int i = tid; i < L * D; i += T) a[S_DK + i] = 0;
    if (tid < uint(PR)) lpe[tid] = 0;
    SYNC;
    device const Tape *tp0 = tape + size_t(r) * DEC + ep * TMAX * PR;
    for (int step = 0; step < TMAX; step++) {   // forward, each row's state chained from its entering state
      device const Tape *tp = tp0 + step * PR;
      device float *st = a + S_STEPS + step * STEP, *inb = a + S_IN + step * PR * IN, *scs = a + S_SC + step * PR * L;
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
      for (int i = tid; i < PR * L; i += T) {   // the scores q.k / 8 of the legal lanes
        const int p = i / L, l = i % L;
        if (tp[p].chosen >= 0 && (tp[p].legal >> l & 1)) { float v = 0; for (int d = 0; d < D; d++) v += st[P_Q + p * D + d] * a[S_KEY + l * D + d]; scs[i] = v / 8; }
        else scs[i] = -INFINITY;
      }
      SYNC;
      if (tid < uint(PR) && tp[tid].chosen >= 0) {
        const int p = tid; device const float *sp = scs + p * L;
        float most = -INFINITY; for (int l = 0; l < L; l++) if (sp[l] > -INFINITY) most = max(most, CLIP * th(sp[l]));
        float sum = 0; for (int l = 0; l < L; l++) if (sp[l] > -INFINITY) sum += exp(CLIP * th(sp[l]) - most);
        lpe[p] += CLIP * th(sp[tp[p].chosen]) - most - log(sum);
      }
      SYNC;
    }
    if (tid < uint(PR)) {   // each tree's seed: 1 + n_k - n w_k
      const int p = tid, j = p % J;
      float most = -INFINITY; for (int m = 0; m < KS; m++) most = max(most, lpe[m * J + j]);
      float sum = 0, n = 0; for (int m = 0; m < KS; m++) { sum += exp(lpe[m * J + j] - most); n += float(counts[(size_t(r) * EPMAX + ep) * PR + m * J + j]); }
      seed[p] = 1 + float(counts[(size_t(r) * EPMAX + ep) * PR + p]) - n * exp(lpe[p] - most) / sum;
      if (p < J) {
        const float lse = most + log(sum);
        for (int m = 0; m < KS; m++) llj[j] += lpe[m * J + j] + float(counts[(size_t(r) * EPMAX + ep) * PR + m * J + j]) * (lpe[m * J + j] - lse);   // 0 for an absent commodity
      }
    }
    for (int i = tid; i < 2 * PD; i += T) a[S_DH + i] = 0;
    SYNC;
    for (int step = TMAX - 1; step >= 0; step--) {   // backward
      device const Tape *tp = tp0 + step * PR;
      device float *st = a + S_STEPS + step * STEP, *inb = a + S_IN + step * PR * IN, *scs = a + S_SC + step * PR * L;
      device float *rc = rc0 + step * PR * REC;
      if (tid < uint(PR)) act[tid] = tp[tid].chosen;
      SYNC;
      if (tid < uint(PR)) {   // d(-seed log p)/d(q.k), the clip and the 1/8 folded in
        const int p = tid; device const float *sp = scs + p * L;
        if (act[p] >= 0) {
          float most = -INFINITY; for (int l = 0; l < L; l++) if (sp[l] > -INFINITY) most = max(most, CLIP * th(sp[l]));
          float sum = 0; for (int l = 0; l < L; l++) if (sp[l] > -INFINITY) sum += exp(CLIP * th(sp[l]) - most);
          for (int l = 0; l < L; l++) {
            if (sp[l] == -INFINITY) { pr[p][l] = 0; continue; }
            const float t = th(sp[l]), pl = exp(CLIP * t - most) / sum;
            pr[p][l] = seed[p] * (pl - (l == act[p] ? 1.0f : 0.0f)) * CLIP * (1 - t * t) / 8;
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
      for (int i = tid; i < PR * REC; i += T) if (act[i / REC] < 0) rc[i] = 0;   // a step not taken contributes nothing
      SYNC;
    }
    for (int i = tid; i < L * D; i += T) { const int l = i / D, d = i % D; kr[l * KREC + K_DK + d] = a[S_DK + i]; kr[l * KREC + K_HID + d] = a[S_HID + i]; }
    for (int i = tid; i < L * LF; i += T) kr[(i / LF) * KREC + K_PHI + i % LF] = phi[i];
    for (int i = tid; i < L * D; i += T) {
      const int l = i / D, d = i % D; float v = 0;
      for (int q = 0; q < D; q++) v += net.k2[q][d] * a[S_DK + l * D + q];
      kr[l * KREC + K_DHID + d] = a[S_HID + i] > 0 ? v : 0;
    }
    SYNC;
  }
  if (tid == 0) { float v = 0; for (int j = 0; j < J; j++) v += llj[j]; loglik[r] = v; }
}

// one weight block's per-rollout gradients, summed for three tasks: g[t P + at + i] = sum_b w[3 b + t] part[base + b size + i]
kernel void reduce(device const float *part [[buffer(0)]], device float *g [[buffer(1)]], constant uint4 &q [[buffer(2)]],
                   device const float *w [[buffer(3)]], constant uint &P [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= q.y) return;
  float a = 0, b = 0, c = 0;
  for (uint k = 0; k < q.z; k++) { const float v = part[q.x + size_t(k) * q.y + i]; a += w[3 * k] * v; b += w[3 * k + 1] * v; c += w[3 * k + 2] * v; }
  g[q.w + i] = a; g[P + q.w + i] = b; g[2 * P + q.w + i] = c;
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
  o[11] = R_LAYERSIZE; o[12] = REC; o[13] = S_SIZE; o[14] = int(sizeof(Layer) / 4); o[15] = R_DQ; o[16] = R_XF; o[17] = R_DST;
  o[18] = RS_SIZE; o[19] = SIM_SIZE; o[20] = DEC; o[21] = EPMAX; o[22] = int(sizeof(Scenario)); o[23] = int(sizeof(Tape) / 4);
}

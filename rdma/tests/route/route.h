/* The routing policy's shared definitions (design/switched-multipath-dma.md). The parts that share them, each its own file
   and its own hardware:
     sim.c         the environment: streams of collectives on a fabric, the max-plus FIFO recursion over pieces, the
                   features, the draws, each draw's credit (CPU threads)
     policy.swift  the forward: two recurrent layers over every pending decision point's tokens, from each stream's
                   carried state, w for each context (GPU)
     learn.swift   the backward through each window's decision points, the weights' gradients, the optimizer (GPU)
     main.swift    the runtime that keeps the three busy
     forward.c     the same forward on a CPU, for deployment beside the bridge (tests/switch.c's policy router uses it)

   The action. A piece of pattern j (a source and a destination set) standing at node v and still owing destinations S
   draws a split: each d in S assigned one of v's out-lanes; the lanes given a nonempty share are sent on. A context is
   (j, v, S), its vocabulary out_lanes(v)^|S|. At each refresh the policy reads the lane tokens in turn from its carried
   state, then each context's token as one more step, and returns w_c; a split's logit is CLIP tanh(w_c . phi / sqrt(FS)),
   phi the split's features. Pieces draw from the active table as they move, every piece its own tree.

   Collectives as patterns: an all-gather's rank r sends its pages to all others (r -> all - r); an all-reduce's rank r
   sends page p's partial to its owner o = p mod n (r -> {o}), then o sends the reduced page to all others (o -> all - o)
   once its n - 1 partials have landed; a scatter's root sends each destination its own bytes (root -> {d}).

   The gradient. For each task, dw_c = sum over the context's draws of advantage x (psi(split) - E psi), psi the gradient
   of the split's logit in w_c and the expectation over the table masked as the draw was; the environment computes it,
   the backward carries it through the layers and back through the window's refreshes. */
#ifndef ROUTE_H
#define ROUTE_H

enum {
  ROUTE_N = 6,                    /* nodes */
  ROUTE_L = 32,                   /* lanes */
  ROUTE_J = 36,                   /* patterns: n (n - 1) unicast and n multicast, n <= 6 compute nodes */
  ROUTE_DEG = 5,                  /* out-lanes a node */
  ROUTE_SETS = 64,                /* destination sets, as bit masks over the nodes */
  ROUTE_PAGE_PIECES = 768,        /* pieces a flow, at most (12.6 MB) */
  ROUTE_POINTS = 8,               /* refreshes a training window, at most */
  ROUTE_CABLES = 4,
  ROUTE_CONTEXTS = 1024,          /* contexts a fabric, at most (a 6-node mesh of 6 ranks has 786) */
  ROUTE_D = 64, ROUTE_E = 192,    /* the residual stream, the feed-forward block */
  ROUTE_IN = 48,                  /* a token: a lane's or a context's */
  ROUTE_FS = 40,                  /* a split's features */
  ROUTE_TASKS = 3                 /* throughput, latency, queueing */
};
#define ROUTE_PAGE 16384.0
#define ROUTE_CLIP 10.0

/* What passes between the parts, in memory both CPU and GPU see: a refresh's lane tokens (its contexts' tokens beside) */
struct route_decision {
  unsigned window, point, lanes, contexts;
  float lane[ROUTE_L][ROUTE_IN];
};

#ifndef __METAL_VERSION__
#include <stddef.h>
#include <stdint.h>

/* a fabric: nodes (the compute ones the collectives' ranks), and lanes, each a direction of a cabled pair */
struct route_fabric {
  int nodes, ncompute, lanes;
  int compute[ROUTE_N], from[ROUTE_L], to[ROUTE_L], cables[ROUTE_L];
  double rate[ROUTE_L], latency[ROUTE_L], duplex;   /* rate: all the lane's cables, bytes a ns; latency ns */
};
enum { ROUTE_ALLGATHER, ROUTE_ALLREDUCE, ROUTE_SCATTER, ROUTE_KINDS };
/* size: bytes a rank (scatter: a destination); done: the fraction of its pages already delivered when it arrives (a
   window starting midstream) */
struct route_op { int kind, root; double size, arrival, floor, done; };
#define ROUTE_OPS 256
/* a stream of collectives on a fabric, with `background` bytes queued on each lane at its start; its patterns; `unit`,
   the slowest compute pair's shortest delay for one piece, the time the features are in */
struct route_stream {
  struct route_fabric f;
  double background[ROUTE_L], unit;
  int patterns, source[ROUTE_J], spt[ROUTE_J][ROUTE_N];   /* spt: the unloaded shortest-path tree's in-lane at each node, -1 none */
  uint32_t dest[ROUTE_J];
  int ops; struct route_op op[ROUTE_OPS];
};
/* the refreshes a run asks the policy for: `points` of them, from t0, `dt` apart, each table active `lag` after its
   refresh (points 0: one every dt until the stream is over). Before the first active table, `warm` routes (2 the
   shortest-path tree, 3 uniform). Posting: `burst` 1, a flow posts every piece the moment it is released (as tests/
   switch.c does); 0, its next the moment its last has started on the wire. */
struct route_window { double t0, dt, lag; int points, warm, burst; };
/* the fabric at a refresh, as the policy reads it (times in ns) */
/* per pattern: recv, each destination's delivered fraction; left, the fraction not yet posted; pending, the bytes released
   and waiting to post; demand, the live flows' bytes */
struct route_state {
  double tau, wait[ROUTE_L], dup[ROUTE_L], togo[ROUTE_N][ROUTE_N], recv[ROUTE_J][ROUTE_N], inflight[ROUTE_N], left[ROUTE_J], pending[ROUTE_J], demand[ROUTE_J], posted, ops;
};
struct route_topology { uint32_t adj[ROUTE_N], srcmask; int deg[ROUTE_N], out[ROUTE_N][ROUTE_DEG], rev[ROUTE_L]; };
/* a node's out-lanes at the instant of a draw: each one's earliest-cable wait and duplex share (the node's bridge knows
   its own lanes exactly, at every draw, so a burst's draws see the queue the ones before them built) */
struct route_live { double wait[ROUTE_DEG], dup[ROUTE_DEG]; };
/* over the run's collectives that finished: completion over the floor; per piece: latency, queueing, over the floor */
struct route_costs { double throughput, latency, queueing; int ops, pieces; };
struct route_rng { uint64_t s; };

int route_fabric_load(const char *path, struct route_fabric *f);
void route_network(struct route_rng *g, const struct route_fabric *measured, struct route_fabric *f);   /* one network of a random kind */
/* a stream on f: collectives of random kinds and sizes arriving at random over `duration` ns at offered load `load`
   (kind -1: random; size 0: random) */
void route_stream_make(const struct route_fabric *f, struct route_rng *g, double duration, double load, int background, struct route_stream *s);
/* a stream of one collective at t = 0 */
void route_stream_one(const struct route_fabric *f, int kind, double size, int root, struct route_stream *s);
uint64_t route_next(struct route_rng *g);
double route_uniform(struct route_rng *g);

/* One run of a stream. mode: 0 the policy's tables in the window, 2 every piece on the shortest-path tree, 3 each
   destination's lane uniform among the allowed, 4 the shortest-path tree, every piece posted at its release (tests/
   switch.c's fixed router). key0 draws the run before the window, key the window's. */
struct route_sim;
struct route_sim *route_sim_new(void);
void route_sim_free(struct route_sim *x);
void route_sim_begin(struct route_sim *x, const struct route_stream *s, const struct route_window *w, int mode, uint32_t key0, uint32_t key);
/* Runs to the next refresh. Returns 1 with d's header and lane tokens filled and its contexts' tokens written to
   `tokens` ([d->contexts][IN]); 0 once the run is over. */
int route_sim_decide(struct route_sim *x, struct route_decision *d, float *tokens);
/* the policy's answer for the refresh just returned ([contexts][FS]; NULL outside mode 0) */
void route_sim_answer(struct route_sim *x, const float *w);
void route_sim_costs(const struct route_sim *x, struct route_costs *c);
/* after the run: refresh `point`'s dw ([TASKS][contexts][FS], the throughput channel at unit advantage, the caller
   scaling it), and the log-likelihood of the draws from its table */
void route_sim_credit(struct route_sim *x, int point, float *dw, double *loglik);
int route_sim_points(const struct route_sim *x);
int route_sim_contexts(const struct route_sim *x);
float *route_sim_w(struct route_sim *x, int point);   /* refresh `point`'s stored w, which the credit reads */
void route_sim_counts(const struct route_sim *x, uint64_t out[4]);   /* events, draws, table entries built, sends */

/* The contexts of a stream's fabric, in the order the tokens and w are: (pattern, node, owed set) */
int route_contexts(const struct route_stream *s, const struct route_topology *t, int *pattern, int *node, uint32_t *owed);
void route_topology_make(const struct route_fabric *f, const struct route_stream *s, struct route_topology *t);
/* A split drawn from context (v, S)'s table (w its policy vector), masked to the lanes that leave `visited` and reach
   their share without coming back; u in [0, 1). Returns the token; the table is built into `cdf` (route_tokens(t, v, S)
   doubles). */
int route_tokens(const struct route_topology *t, int v, uint32_t S);
uint32_t route_split(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int v, uint32_t S, uint32_t visited,
                     const struct route_live *live, const float *w, double u, double *cdf);
uint32_t route_allowed(const struct route_stream *s, const struct route_topology *t, int v, uint32_t S, uint32_t visited);

/* The policy's forward on a CPU (forward.c): net.swift's int8 network (minGRU), its weights as stored, int8 codes
   (int8mm.h's format, written by the trainer's THETA_INT8): route_params_read loads them (0, or -1 on a bad file). One
   refresh: lanes [nl][IN], tokens [nc][IN], the carried state [2][D] (read and advanced), w [nc][FS] out; scratch
   route_forward_scratch(nc) floats. A matrix: its codes [rows][k] (k its columns padded to 64), a scale a row, its bias */
struct route_mat { int rows, cols, k; int8_t *q; float *s, *b; };
struct route_params { struct route_mat inp, zh[2], up[2], down[2], out; };
int route_params_read(struct route_params *p, const char *path);
size_t route_forward_scratch(int nc);
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *scratch);

/* ---- the features: what the policy reads, defined once ---- */
static inline double route_send1(const struct route_stream *s, int l) { return ROUTE_PAGE * s->f.cables[l] / s->f.rate[l]; }   /* one page, one cable */
static inline double route_hop(const struct route_stream *s, const struct route_state *st, int l) {
  return st->wait[l] + route_send1(s, l) * st->dup[l] + s->f.latency[l];
}
/* can node `from` reach a node in `want` without entering `avoid`? */
static inline int route_reaches(const struct route_topology *t, int from, uint32_t avoid, uint32_t want) {
  if (want >> from & 1) return 1;
  uint32_t seen = 1u << from, frontier = seen;
  for (int round = 0; round < ROUTE_N; round++) {
    uint32_t next = 0;
    for (int u = 0; u < ROUTE_N; u++) if (frontier >> u & 1) next |= t->adj[u];
    next &= ~avoid & ~seen;
    if (next & want) return 1;
    if (!next) return 0;
    seen |= next; frontier = next;
  }
  return 0;
}
static inline void route_lane_token(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int l, float *o) {
  for (int c = 0; c < ROUTE_IN; c++) o[c] = 0;
  if (l >= s->f.lanes) return;
  const double u = s->unit;
  const int a = s->f.from[l], b = s->f.to[l];
  o[a] = 1; o[6 + b] = 1; o[12] = s->f.cables[l] / 4.0f; o[13] = (float)(route_send1(s, l) / u); o[14] = (float)(s->f.latency[l] / u);
  o[15] = (float)(st->wait[l] / u); o[16] = (float)st->dup[l];
  o[17] = (float)(__builtin_log(s->f.rate[l] * u / ROUTE_PAGE) / 8); o[18] = (float)(st->inflight[a] / ROUTE_PAGE / 64); o[19] = (float)(st->inflight[b] / ROUTE_PAGE / 64);
  o[20] = (t->srcmask >> a & 1) ? 0 : 1; o[21] = (t->srcmask >> b & 1) ? 0 : 1; o[22] = 1;
}
static inline void route_context_token(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int j, int v, uint32_t S,
                                       int point, float *o) {
  for (int c = 0; c < ROUTE_IN; c++) o[c] = 0;
  const double u = s->unit;
  o[v] = 1; o[18 + s->source[j]] = 1; o[24] = (t->srcmask >> v & 1) ? 0 : 1;
  o[25] = (float)(__builtin_log(1 + st->posted / ROUTE_PAGE) / 8); o[26] = (float)st->left[j]; o[27] = (float)(st->ops / 8); o[28] = __builtin_popcount(s->dest[j]) == 1 ? 1 : 0;
  double most = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (S >> d & 1) { double g = st->togo[v][d] / u; if (g > 1e3) g = 1e3; o[6 + d] = 1; o[29 + d] = (float)g; if (g > most) most = g; }
    if (s->dest[j] >> d & 1) o[12 + d] = 1;
    o[35 + d] = (float)st->recv[j][d];
  }
  double out = 0; for (int l = 0; l < s->f.lanes; l++) if (s->f.from[l] == s->source[j]) out += s->f.rate[l];
  o[41] = (float)(__builtin_log(1 + st->tau / u) / 8); o[42] = (float)point / ROUTE_POINTS;
  o[43] = (float)(__builtin_log(1 + st->pending[j] / out / u));   /* the burst waiting to post, as time on the source's lanes */
  o[44] = (float)most; o[45] = 1; o[46] = 1; o[47] = (float)(__builtin_log(1 + st->demand[j] / ROUTE_PAGE) / 8);
}
/* a draw's per-(destination, out-lane) quantities, made once: whether the lane leads to the destination without coming
   back to v, and the destination's arrival through it (v's lane at the draw's instant, then the refresh's cost-to-go).
   Every split's features are these combined. */
struct route_pre { double arrive[ROUTE_N][ROUTE_DEG]; uint8_t valid[ROUTE_N][ROUTE_DEG]; };
static inline void route_pre_make(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int v, uint32_t S,
                                  const struct route_live *live, struct route_pre *pre) {
  for (int d = 0; d < ROUTE_N; d++) {
    if (!(S >> d & 1)) continue;
    for (int k = 0; k < t->deg[v]; k++) {
      const int l = t->out[v][k], h = s->f.to[l];
      double g = h == d ? 0 : st->togo[h][d]; if (g > 1e3 * s->unit) g = 1e3 * s->unit;
      pre->valid[d][k] = (uint8_t)(h == d || route_reaches(t, h, 1u << v, 1u << d));
      pre->arrive[d][k] = live->wait[k] + route_send1(s, l) * live->dup[k] + s->f.latency[l] + g;
    }
  }
}
/* split token `tok` at (v, S) from its draw's quantities: its features (phi, if given) and w . phi; returns whether every
   destination's lane is valid */
static inline int route_split_pre(const struct route_pre *pre, int deg, double unit, uint32_t S, uint32_t tok, const float *w, double *dot, float *phi) {
  int valid = 1, cnt = 0; uint32_t used = 0; double most = 0, sum = 0, a = 0;
  if (phi) for (int f = 0; f < ROUTE_FS; f++) phi[f] = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (!(S >> d & 1)) continue;
    const int k = (int)(tok % (uint32_t)deg); tok /= (uint32_t)deg;
    valid = valid && pre->valid[d][k];
    const double arrive = pre->arrive[d][k];
    if (arrive > most) most = arrive;
    sum += arrive; cnt++; used |= 1u << k;
    a += w ? w[d * ROUTE_DEG + k] : 0;
    if (phi) phi[d * ROUTE_DEG + k] = 1;
  }
  const double f35 = most / unit, f36 = sum / (cnt ? cnt : 1) / unit, f37 = (double)__builtin_popcount(used) / (cnt ? cnt : 1);
  for (int k = 0; k < ROUTE_DEG; k++) if (used >> k & 1) { if (w) a += w[30 + k]; if (phi) phi[30 + k] = 1; }
  if (w) a += w[35] * f35 + w[36] * f36 + w[37] * f37;
  if (phi) { phi[35] = (float)f35; phi[36] = (float)f36; phi[37] = (float)f37; }
  if (dot) *dot = a;
  return valid;
}
/* split token `tok` at (v, S): its features (v's own lanes at the draw's instant, `live`; beyond them the refresh's
   cost-to-go); returns whether every destination's lane leads to it without coming back to v */
static inline int route_split_features(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int v, uint32_t S,
                                       uint32_t tok, const struct route_live *live, float *phi) {
  const int deg = t->deg[v]; const double u = s->unit;
  for (int f = 0; f < ROUTE_FS; f++) phi[f] = 0;
  int valid = 1, cnt = 0; uint32_t used = 0; double most = 0, sum = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (!(S >> d & 1)) continue;
    const int k = (int)(tok % (uint32_t)deg); tok /= (uint32_t)deg;
    const int l = t->out[v][k], h = s->f.to[l];
    valid = valid && (h == d || route_reaches(t, h, 1u << v, 1u << d));
    phi[d * ROUTE_DEG + k] = 1; used |= 1u << k;
    double g = h == d ? 0 : st->togo[h][d]; if (g > 1e3 * u) g = 1e3 * u;
    const double arrive = live->wait[k] + route_send1(s, l) * live->dup[k] + s->f.latency[l] + g;
    if (arrive > most) most = arrive;
    sum += arrive; cnt++;
  }
  for (int k = 0; k < ROUTE_DEG; k++) if (used >> k & 1) phi[30 + k] = 1;
  phi[35] = (float)(most / u); phi[36] = (float)(sum / (cnt ? cnt : 1) / u); phi[37] = (float)__builtin_popcount(used) / (cnt ? cnt : 1);
  return valid;
}
#endif
#endif

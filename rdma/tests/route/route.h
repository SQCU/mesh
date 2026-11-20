/* The routing policy's shared definitions (design/switched-multipath-dma.md). Four parts share them, each its own file
   and its own hardware:
     sim.c           the environment: the fabric's max-plus FIFO recursion over pieces, the features, the draws, each
                     draw's credit (CPU threads, many transactions each)
     policy.metal    the forward: two recurrent layers over every pending decision point's tokens, w for each context
                     (a GPU queue)
     gradient.metal  the backward: dw back through the same layers, the weights' gradients (another GPU queue)
     train.m         the optimizer, and the runtime that keeps the other three busy (a CPU thread and the network)

   The action. A piece of commodity j standing at node v and still owing destinations S draws a split: each d in S
   assigned one of v's out-lanes; the lanes given a nonempty share are sent on. A context is (j, v, S) and its
   vocabulary out_lanes(v)^|S|. At each decision point the policy reads the lane tokens in turn, then each context's
   token as one more step from that state, and returns w_c; a split's logit is CLIP tanh(w_c . phi(split) / sqrt(FS)),
   phi the split's features. Pieces draw from those distributions as they move, every piece its own tree.

   The gradient. For each task, dw_c = sum over the context's draws of advantage x (psi(split) - E psi), psi the
   gradient of the split's logit in w_c and the expectation over the context's table masked as the draw was; the
   environment computes it (it holds the draws and the masks), the backward carries it through the layers. */
#ifndef ROUTE_H
#define ROUTE_H

enum {
  ROUTE_N = 6,                    /* nodes */
  ROUTE_L = 32,                   /* lanes */
  ROUTE_J = 6,                    /* commodities */
  ROUTE_DEG = 5,                  /* out-lanes a node */
  ROUTE_SETS = 64,                /* destination sets, as bit masks over the nodes */
  ROUTE_GROUP = 64,               /* pieces a commodity posts at a decision point */
  ROUTE_POINTS = 12,              /* decision points a transaction, at most */
  ROUTE_PIECES = ROUTE_POINTS * ROUTE_GROUP,
  ROUTE_CABLES = 4,
  ROUTE_CONTEXTS = ROUTE_J * ROUTE_N * 31,   /* (j, v, S): S a nonempty subset of j's at most 5 destinations */
  ROUTE_D = 64, ROUTE_E = 192,    /* the residual stream, the feed-forward block */
  ROUTE_IN = 48,                  /* a token: a lane's or a context's */
  ROUTE_FS = 40,                  /* a split's features */
  ROUTE_TASKS = 3                 /* throughput, latency, queueing */
};
#define ROUTE_PAGE 16384.0
#define ROUTE_CLIP 10.0

/* What passes between the parts, in memory both CPU and GPU see. A decision point of a transaction: its lane tokens and,
   in the row pools from `row`, its contexts' tokens (in), their w (the policy's answer) and their dw (each task's). */
struct route_decision {
  unsigned transaction, point, lanes, contexts, row, pad[3];
  float lane[ROUTE_L][ROUTE_IN];
};
/* row pools, `rows` rows each: token [rows][IN]; w [rows][FS]; dw [TASKS][rows][FS] */

#ifndef __METAL_VERSION__
#include <stddef.h>
#include <stdint.h>

/* a fabric: nodes (the compute ones the transaction's ranks), and lanes, each a direction of a cabled pair */
struct route_fabric {
  int nodes, ncompute, lanes;
  int compute[ROUTE_N], from[ROUTE_L], to[ROUTE_L], cables[ROUTE_L];
  double rate[ROUTE_L], latency[ROUTE_L], duplex;   /* rate: all the lane's cables, bytes a ns; latency ns */
};
/* a transaction on a fabric: an all-gather of `size` bytes a rank; `background` bytes queued on each lane at the start;
   the floor's two bounds (the cut bound; the slowest pair's shortest delay for one piece) */
struct route_scenario {
  struct route_fabric f;
  double size, background[ROUTE_L], cut, reach, floor;
  int commodities, source[ROUTE_J], spt[ROUTE_J][ROUTE_N];   /* spt: the unloaded shortest-path tree's in-lane at each node, -1 none */
  uint32_t dest[ROUTE_J];
};
/* the fabric at a decision point, as the policy reads it (times in ns) */
struct route_state {
  double tau, wait[ROUTE_L], dup[ROUTE_L], togo[ROUTE_N][ROUTE_N], recv[ROUTE_J][ROUTE_N], inflight[ROUTE_N], left[ROUTE_J], progress;
};
struct route_topology { uint32_t adj[ROUTE_N], srcmask; int deg[ROUTE_N], out[ROUTE_N][ROUTE_DEG], rev[ROUTE_L]; };
struct route_costs { double throughput, latency, queueing; };   /* completion / floor; per piece, mean, over the floor */
struct route_rng { uint64_t s; };

int route_fabric_load(const char *path, struct route_fabric *f);
void route_network(struct route_rng *g, const struct route_fabric *measured, struct route_fabric *f);   /* one network of a random kind */
void route_scenario_make(const struct route_fabric *f, double size, struct route_rng *g, int background, struct route_scenario *x);
uint64_t route_next(struct route_rng *g);
double route_uniform(struct route_rng *g);

/* One transaction. mode: 0 the policy's tables, 2 every piece on the shortest-path tree, 3 each destination's lane
   uniform among the allowed, 4 the shortest-path tree with every group posted at 0 (tests/switch.c's fixed router). */
struct route_sim;
struct route_sim *route_sim_new(void);
void route_sim_free(struct route_sim *x);
void route_sim_begin(struct route_sim *x, const struct route_scenario *sc, int mode, uint32_t key);
/* Runs to the next decision point. Returns 1 with d's header and lane tokens filled and its contexts' tokens written to
   `tokens` ([d->contexts][IN]); 0 once the transaction is over. */
int route_sim_decide(struct route_sim *x, struct route_decision *d, float *tokens);
/* The policy's answer for the decision point just returned ([contexts][FS]; NULL in modes 2-4): posts its group. */
void route_sim_answer(struct route_sim *x, const float *w);
void route_sim_costs(const struct route_sim *x, struct route_costs *c);
/* After the transaction: decision point `point`'s dw ([TASKS][contexts][FS], the throughput channel at unit advantage,
   the caller scaling it by the transaction's), and the log-likelihood of that point's draws. */
void route_sim_credit(struct route_sim *x, int point, float *dw, double *loglik);
int route_sim_points(const struct route_sim *x);
int route_sim_contexts(const struct route_sim *x);
/* what a transaction did: events, draws, table entries built, sends */
void route_sim_counts(const struct route_sim *x, uint64_t out[4]);
/* decision point `point`'s stored w ([contexts][FS]), which the credit reads: the gradient check perturbs it */
float *route_sim_w(struct route_sim *x, int point);

/* The policy's forward on a CPU (forward.c), for deployment beside the bridge: theta in policy.swift's layout; one
   decision point: lanes [nl][IN], tokens [nc][IN], the carried state [2][D] (read and advanced), w [nc][FS] out;
   scratch route_forward_scratch(nc) floats */
struct route_params {
  int full;
  const float *inp, *binp, *z[2], *bz[2], *r[2], *br[2], *h[2], *bh[2], *up[2], *bu[2], *down[2], *bd[2], *out;
};
int route_params_count(int full);
void route_params_bind(struct route_params *p, const float *theta, int full);
size_t route_forward_scratch(int nc);
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *scratch);

/* ---- the features: what the policy reads, defined once ---- */
static inline double route_send1(const struct route_scenario *s, int l) {   /* one piece's time on one cable */
  const double b = s->size < ROUTE_PAGE ? s->size : ROUTE_PAGE;
  return b * s->f.cables[l] / s->f.rate[l];
}
static inline double route_hop(const struct route_scenario *s, const struct route_state *st, int l) {
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
static inline void route_lane_token(const struct route_scenario *s, const struct route_state *st, const struct route_topology *t, int l, float *o) {
  for (int c = 0; c < ROUTE_IN; c++) o[c] = 0;
  if (l >= s->f.lanes) return;
  const double fl = s->floor, sz = s->size > 1 ? s->size : 1;
  const int a = s->f.from[l], b = s->f.to[l];
  o[a] = 1; o[6 + b] = 1; o[12] = s->f.cables[l] / 4.0f; o[13] = (float)(route_send1(s, l) / fl); o[14] = (float)(s->f.latency[l] / fl);
  o[15] = (float)(st->wait[l] / fl); o[16] = (float)st->dup[l];
  const double rr = s->f.rate[l] * fl / sz;
  o[17] = (float)(__builtin_log(rr > 1e-6 ? rr : 1e-6) / 8); o[18] = (float)(st->inflight[a] / sz); o[19] = (float)(st->inflight[b] / sz);
  o[20] = (t->srcmask >> a & 1) ? 0 : 1; o[21] = (t->srcmask >> b & 1) ? 0 : 1; o[22] = 1;
}
static inline void route_context_token(const struct route_scenario *s, const struct route_state *st, const struct route_topology *t, int j, int v, uint32_t S,
                                       int point, float *o) {
  for (int c = 0; c < ROUTE_IN; c++) o[c] = 0;
  const double fl = s->floor;
  o[v] = 1; o[18 + s->source[j]] = 1; o[24] = (t->srcmask >> v & 1) ? 0 : 1;
  o[25] = (float)(__builtin_log(1 + s->size / ROUTE_PAGE) / 8); o[26] = (float)st->left[j]; o[27] = (float)(s->reach / fl); o[28] = (float)(s->cut / fl);
  double most = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (S >> d & 1) { double g = st->togo[v][d] / fl; if (g > 1e3) g = 1e3; o[6 + d] = 1; o[29 + d] = (float)g; if (g > most) most = g; }
    if (s->dest[j] >> d & 1) o[12 + d] = 1;
    o[35 + d] = (float)st->recv[j][d];
  }
  o[41] = (float)(st->tau / fl); o[42] = (float)point / ROUTE_POINTS; o[43] = (float)st->progress; o[44] = (float)most; o[45] = 1; o[46] = 1;
}
/* split token `tok` at (v, S): its features; returns whether every destination's lane leads to it without coming back
   to v */
static inline int route_split_features(const struct route_scenario *s, const struct route_state *st, const struct route_topology *t, int v, uint32_t S,
                                       uint32_t tok, float *phi) {
  const int deg = t->deg[v]; const double fl = s->floor;
  for (int f = 0; f < ROUTE_FS; f++) phi[f] = 0;
  int valid = 1, cnt = 0; uint32_t used = 0; double most = 0, sum = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (!(S >> d & 1)) continue;
    const int k = (int)(tok % (uint32_t)deg); tok /= (uint32_t)deg;
    const int l = t->out[v][k], h = s->f.to[l];
    valid = valid && (h == d || route_reaches(t, h, 1u << v, 1u << d));
    phi[d * ROUTE_DEG + k] = 1; used |= 1u << k;
    double g = h == d ? 0 : st->togo[h][d]; if (g > 1e3 * fl) g = 1e3 * fl;
    const double arrive = route_hop(s, st, l) + g;
    if (arrive > most) most = arrive;
    sum += arrive; cnt++;
  }
  for (int k = 0; k < ROUTE_DEG; k++) if (used >> k & 1) phi[30 + k] = 1;
  phi[35] = (float)(most / fl); phi[36] = (float)(sum / (cnt ? cnt : 1) / fl); phi[37] = (float)__builtin_popcount(used) / (cnt ? cnt : 1);
  return valid;
}
#endif
#endif

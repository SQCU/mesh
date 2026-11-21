/* The routing policy's shared definitions (design/switched-multipath-dma.md). The parts that share them, each its own file
   and its own hardware:
     sim.c         the environment: streams of collectives on a fabric, run by the mesh bridges' own switching code
                   (rdma/mesh-switched.h) over the timed wire (rdma/tests/switched-world.h); the features are the
                   bridges' (mesh_switch_tokens_make), the draws the bridges' (CPU threads)
     policy.swift  the forward: two recurrent layers over every pending decision point's tokens, from each stream's
                   carried state, w for each context (GPU)
     learn.swift   the backward through each window's decision points, the weights' gradients, the optimizer (GPU)
     main.swift    the runtime that keeps the three busy
     forward.c     the same forward on a CPU, for deployment in the bridge (struct mesh_switch_policy's splits)

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
#include "telemetry.h"
#include "split.h"

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
/* over the run's collectives that finished: completion over the floor; per piece: latency, queueing, over the floor */
struct route_costs { double throughput, latency, queueing; int ops, pieces; };
struct route_rng { uint64_t s; };

/* A collective's floor on f: its cut bound and its slowest pair's one-piece delay (tests/switch.c's); route_op_floors the
   same for n collectives, the i-th on fabrics[fabric[i]]. The training distribution (py/route/cases.py) draws arrivals
   on them. */
double route_op_floor(const struct route_fabric *f, int kind, double S, int root);
void route_op_floors(const struct route_fabric *fabrics, const int *fabric, int n, const int *kind, const double *size, const int *root, double *out);
/* a stream on f: `background` bytes queued on each lane at 0 (NULL none), and `ops` collectives (their floors filled) */
void route_stream_build(const struct route_fabric *f, const double *background, int ops, const struct route_op *op, struct route_stream *s);
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
/* a draw's log-probability under w (its context's) and its score psi(token) - E psi into score [FS] (split.h's table;
   the GPU's credit computes the same, credit.metal) */
float route_draw_score(const struct route_draw *d, const float *w, float *score);
int route_sim_points(const struct route_sim *x);
int route_sim_contexts(const struct route_sim *x);
float *route_sim_w(struct route_sim *x, int point);   /* refresh `point`'s stored w, which the credit reads */
/* refresh `point`'s draws: route_sim_draw_count records; route_sim_draws writes them ordered by context (each context's
   in its groups' order, each group's in its draws'), ctx[i] the i-th distinct context (ascending) and first[i] its first
   record (first[n] the count); returns n, the distinct contexts. A record's row is its context's index i. */
int route_sim_draw_count(const struct route_sim *x, int point);
int route_sim_draws(struct route_sim *x, int point, struct route_draw *out, int *ctx, int *first);
void route_sim_counts(const struct route_sim *x, uint64_t out[4]);
float route_sim_logp_gap(const struct route_sim *x);   /* the bridge's logged log-probabilities against split.h's under the answered w: the largest gap */   /* events, draws, table entries built, sends */

/* The contexts of a stream's fabric, in the order the tokens and w are: (pattern, node, owed set) */
int route_contexts(const struct route_stream *s, const struct route_topology *t, int *pattern, int *node, uint32_t *owed);
void route_topology_make(const struct route_fabric *f, const struct route_stream *s, struct route_topology *t);
int route_tokens(const struct route_topology *t, int v, uint32_t S);
extern int route_argmax;   /* sim.c: set, the bridges' draws take their table's most probable split */

/* The policy's forward on a CPU (forward.c): model.swift's int8 network (minGRU), its weights as stored, int8 codes
   (int8mm.h's format, written by the trainer's THETA_INT8): route_params_read loads them (0, or -1 on a bad file). One
   refresh: lanes [nl][IN], tokens [nc][IN], the carried state [2][D] (read and advanced), w [nc][FS] out; scratch
   route_forward_scratch(nc) floats. A matrix: its codes [rows][k] (k its columns padded to 64), a scale a row, its bias */
struct route_mat { int rows, cols, k; int8_t *q; float *s, *b; };
struct route_params { struct route_mat inp, zh[2], up[2], down[2], out; };
int route_params_read(struct route_params *p, const char *path);
size_t route_forward_scratch(int nc);
void route_forward(const struct route_params *p, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w, float *scratch);
/* the same in parts, for a router computing a context's w where it is used: the lanes (the refresh's state, `state`
   advanced; scratch route_forward_scratch(0)), then any context from that state */
void route_forward_lanes(const struct route_params *p, const float *lanes, int nl, float *state, float *scratch);
void route_forward_context(const struct route_params *p, const float *token, const float *state, float *w);

/* The trainer's gradient exchange (exchange.c): libnccl-mesh's Metal path over the nodes' bridges. open: this rank's
   communicator (NCCL's bootstrap at id, the root's host:port), a lease of n floats in the node's region, the declared
   program of its one step (the in-place sum, on a Metal stream: commandBuffer an id<MTLCommandBuffer>, not used);
   encode: that step into commandBuffer (the caller commits and waits, then status), on the lease
   route_exchange_buffer gives. 0, or -1 with route_exchange_failure naming what failed */
int route_exchange_open(int rank, int world, const char *id, size_t n, void *commandBuffer);
float *route_exchange_buffer(void);
int route_exchange_encode(void *commandBuffer);
int route_exchange_status(void);
const char *route_exchange_failure(void);
void route_exchange_close(void);   /* the communicator destroyed, the lease freed: before every exit */

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
#endif
#endif

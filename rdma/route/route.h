/* The routing policy's shared definitions (design/switched-multipath-dma.md), as the bridge deploys the trained policy:
   its forward on the CPU (forward.c, struct mesh_switch_policy's splits) over the features the bridge makes from its
   own state (mesh-switched.h mesh_switch_tokens_make: route_lane_token, route_context_token), and the split tables its
   draws read (split.h).

   The action. A piece of pattern j (a source and a destination set) standing at node v and still owing destinations S
   draws a split: each d in S assigned one of v's out-lanes; the lanes given a nonempty share are sent on. A context is
   (j, v, S), its vocabulary out_lanes(v)^|S|. At each refresh the policy reads the lane tokens in turn from its carried
   state, then each context's token as one more step, and returns w_c; a split's logit is CLIP tanh(w_c . phi / sqrt(FS)),
   phi the split's features. Pieces draw from the active table as they move, every piece its own tree.

   Collectives as patterns: an all-gather's rank r sends its pages to all others (r -> all - r); an all-reduce's rank r
   sends page p's partial to its owner o = p mod n (r -> {o}), then o sends the reduced page to all others (o -> all - o)
   once its n - 1 partials have landed; a scatter's root sends each destination its own bytes (root -> {d}). */
#ifndef ROUTE_H
#define ROUTE_H
#include "split.h"
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

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

/* a fabric: nodes (the compute ones the collectives' ranks), and lanes, each a direction of a cabled pair */
struct route_fabric {
  int nodes, ncompute, lanes;
  int compute[ROUTE_N], from[ROUTE_L], to[ROUTE_L], cables[ROUTE_L];
  double rate[ROUTE_L], latency[ROUTE_L], duplex;   /* rate: all the lane's cables, bytes a ns; latency ns */
};
/* a fabric as the features read it: its patterns (each a source and a destination set), and `unit`, the slowest compute
   pair's shortest delay for one piece, the time the features are in */
struct route_stream {
  struct route_fabric f;
  double unit;
  int patterns, source[ROUTE_J];
  uint32_t dest[ROUTE_J];
};
/* the fabric at a refresh, as the policy reads it (times in ns) */
/* per pattern: recv, each destination's delivered fraction; left, the fraction not yet posted; pending, the bytes released
   and waiting to post; demand, the live flows' bytes */
struct route_state {
  double tau, wait[ROUTE_L], dup[ROUTE_L], togo[ROUTE_N][ROUTE_N], recv[ROUTE_J][ROUTE_N], inflight[ROUTE_N], left[ROUTE_J], pending[ROUTE_J], demand[ROUTE_J], posted, ops;
};
struct route_topology { uint32_t adj[ROUTE_N], srcmask; int deg[ROUTE_N], out[ROUTE_N][ROUTE_DEG], rev[ROUTE_L]; };

/* The policy's forward on a CPU (forward.c): the trained int8 network (minGRU), its weights as the trainer stored them,
   int8 codes (int8mm.h's format): route_params_read_stream loads them from an open stream, left open (0, or -1 on a
   bad stream). A matrix: its codes [rows][k] (k its columns padded to 64), a scale a row, its bias */
struct route_mat { int rows, cols, k; int8_t *q; float *s, *b; };
struct route_params { struct route_mat inp, zh[2], up[2], down[2], out; };
int route_params_read_stream(struct route_params *p, FILE *f);
size_t route_forward_scratch(int nc);
/* one refresh, for a router computing a context's w where it is used: the lanes [nl][IN] in turn from the carried
   state [2][D] (read and advanced: the refresh's state; scratch route_forward_scratch(0) floats), then any context's
   w [FS] from its token [IN] and that state */
void route_forward_lanes(const struct route_params *p, const float *lanes, int nl, float *state, float *scratch);
void route_forward_context(const struct route_params *p, const float *token, const float *state, float *w);

/* ---- the features: what the policy reads, defined once ---- */
static inline double route_send1(const struct route_stream *s, int l) { return ROUTE_PAGE * s->f.cables[l] / s->f.rate[l]; }   /* one page, one cable */
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

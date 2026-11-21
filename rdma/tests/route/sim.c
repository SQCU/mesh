#include "route.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- randomness ---- */
uint64_t route_next(struct route_rng *g) {
  g->s += 0x9e3779b97f4a7c15ull; uint64_t z = g->s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31);
}
double route_uniform(struct route_rng *g) { return (double)(route_next(g) >> 11) / 9007199254740992.0; }

/* ---- streams ---- */
static void delays(const struct route_fabric *f, double q, double (*dist)[ROUTE_N]) {   /* one q-byte piece's unloaded shortest delays */
  for (int a = 0; a < ROUTE_N; a++) for (int b = 0; b < ROUTE_N; b++) dist[a][b] = a == b ? 0 : INFINITY;
  for (int l = 0; l < f->lanes; l++) { const double w = f->latency[l] + q / (f->rate[l] / f->cables[l]); if (w < dist[f->from[l]][f->to[l]]) dist[f->from[l]][f->to[l]] = w; }
  for (int m = 0; m < ROUTE_N; m++) for (int a = 0; a < ROUTE_N; a++) for (int b = 0; b < ROUTE_N; b++) if (dist[a][m] + dist[m][b] < dist[a][b]) dist[a][b] = dist[a][m] + dist[m][b];
}
/* a collective's floor: its cut bound (the most bytes a node must take in or put out, over its rate) and its slowest
   pair's shortest delay for one piece, at most a page, as tests/switch.c's (for an all-reduce, the slowest owner's slowest
   partial in plus its slowest reduced page out: every page goes both ways through its owner) */
double route_op_floor(const struct route_fabric *f, int kind, double S, int root) {
  double in[ROUTE_N] = {0}, out[ROUTE_N] = {0}, dist[ROUTE_N][ROUTE_N], cut = 0, reach = 0;
  for (int l = 0; l < f->lanes; l++) { in[f->to[l]] += f->rate[l]; out[f->from[l]] += f->rate[l]; }
  delays(f, S < ROUTE_PAGE ? S : ROUTE_PAGE, dist);
  const int n = f->ncompute;
  for (int k = 0; k < n; k++) {
    const int v = f->compute[k]; double a = 0, b = 0;
    if (kind == ROUTE_ALLGATHER) { a = (n - 1) * S; b = S; }
    else if (kind == ROUTE_ALLREDUCE) { a = 2.0 * (n - 1) * S / n; b = a; }
    else { a = v == root ? 0 : S; b = v == root ? (n - 1) * S : 0; }
    if (a / in[v] > cut) cut = a / in[v];
    if (b / out[v] > cut) cut = b / out[v];
    if (kind == ROUTE_ALLREDUCE) {   /* a page owned by v: its slowest partial in, then its reduced page's slowest way out */
      double in1 = 0, out1 = 0;
      for (int q = 0; q < n; q++) { const int u = f->compute[q]; if (u == v) continue; if (dist[u][v] > in1) in1 = dist[u][v]; if (dist[v][u] > out1) out1 = dist[v][u]; }
      if (k < (S > ROUTE_PAGE ? (int)ceil(S / ROUTE_PAGE) : 1) && in1 + out1 > reach) reach = in1 + out1;   /* v owns a page */
    } else for (int q = 0; q < n; q++) { const int u = f->compute[q]; if (u != v && (kind != ROUTE_SCATTER || v == root) && dist[v][u] > reach) reach = dist[v][u]; }
  }
  return cut > reach ? cut : reach;
}
static void stream_base(const struct route_fabric *f, struct route_stream *s) {
  memset(s, 0, sizeof *s);
  s->f = *f;
  double dist[ROUTE_N][ROUTE_N]; delays(f, ROUTE_PAGE, dist);
  for (int a = 0; a < f->ncompute; a++) for (int b = 0; b < f->ncompute; b++) if (a != b && dist[f->compute[a]][f->compute[b]] > s->unit) s->unit = dist[f->compute[a]][f->compute[b]];
  const int n = f->ncompute, nl = f->lanes;
  for (int a = 0; a < n; a++) {   /* the patterns: every rank to every other, and every rank to all others */
    for (int b = 0; b < n; b++) if (a != b) { s->source[s->patterns] = f->compute[a]; s->dest[s->patterns] = 1u << f->compute[b]; s->patterns++; }
    uint32_t all = 0; for (int b = 0; b < n; b++) if (b != a) all |= 1u << f->compute[b];
    s->source[s->patterns] = f->compute[a]; s->dest[s->patterns] = all; s->patterns++;
  }
  double w[ROUTE_L];
  for (int l = 0; l < nl; l++) w[l] = f->latency[l] + ROUTE_PAGE / (f->rate[l] / f->cables[l]);
  for (int j = 0; j < s->patterns; j++) {   /* each pattern's unloaded shortest-path tree, pruned to what leads to a destination */
    const int src = s->source[j];
    double at[ROUTE_N]; int by[ROUTE_N], done[ROUTE_N] = {0}, keep[ROUTE_N] = {0};
    for (int v = 0; v < ROUTE_N; v++) { at[v] = INFINITY; by[v] = -1; }
    at[src] = 0;
    for (int it = 0; it < f->nodes; it++) {
      int u = -1;
      for (int v = 0; v < f->nodes; v++) if (!done[v] && isfinite(at[v]) && (u < 0 || at[v] < at[u])) u = v;
      if (u < 0) break;
      done[u] = 1;
      for (int l = 0; l < nl; l++) if (f->from[l] == u && at[u] + w[l] < at[f->to[l]]) { at[f->to[l]] = at[u] + w[l]; by[f->to[l]] = l; }
    }
    for (int d = 0; d < ROUTE_N; d++) if (s->dest[j] >> d & 1) for (int v = d; v != src; v = f->from[by[v]]) keep[v] = 1;
    for (int v = 0; v < ROUTE_N; v++) s->spt[j][v] = keep[v] ? by[v] : -1;
  }
}
void route_op_floors(const struct route_fabric *fabrics, const int *fabric, int n, const int *kind, const double *size, const int *root, double *out) {
  for (int i = 0; i < n; i++) out[i] = route_op_floor(&fabrics[fabric[i]], kind[i], size[i], root[i]);
}
void route_stream_build(const struct route_fabric *f, const double *background, int ops, const struct route_op *op, struct route_stream *s) {
  stream_base(f, s);
  for (int l = 0; l < f->lanes; l++) s->background[l] = background ? background[l] : 0;
  s->ops = ops < ROUTE_OPS ? ops : ROUTE_OPS;
  for (int k = 0; k < s->ops; k++) { s->op[k] = op[k]; s->op[k].floor = route_op_floor(f, op[k].kind, op[k].size, op[k].root); }
}
void route_stream_one(const struct route_fabric *f, int kind, double size, int root, struct route_stream *s) {
  stream_base(f, s);
  s->op[0] = (struct route_op){kind, root, size, 0, route_op_floor(f, kind, size, root), 0}; s->ops = 1;
}
void route_topology_make(const struct route_fabric *f, const struct route_stream *s, struct route_topology *t) {
  memset(t, 0, sizeof *t);
  for (int v = 0; v < ROUTE_N; v++) for (int l = 0; l < f->lanes; l++) if (f->from[l] == v) { t->adj[v] |= 1u << f->to[l]; if (t->deg[v] < ROUTE_DEG) t->out[v][t->deg[v]++] = l; }
  for (int l = 0; l < ROUTE_L; l++) { t->rev[l] = -1; for (int m = 0; m < f->lanes && l < f->lanes; m++) if (f->from[m] == f->to[l] && f->to[m] == f->from[l]) t->rev[l] = m; }
  for (int k = 0; k < f->ncompute; k++) t->srcmask |= 1u << f->compute[k];
  (void)s;
}
int route_tokens(const struct route_topology *t, int v, uint32_t S) { int n = 1; for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) n *= t->deg[v]; return n; }
int route_contexts(const struct route_stream *s, const struct route_topology *t, int *pattern, int *node, uint32_t *owed) {
  int n = 0;
  for (int j = 0; j < s->patterns; j++) for (int v = 0; v < s->f.nodes; v++) for (uint32_t S = 1; S < ROUTE_SETS; S++) {
    if ((S & ~s->dest[j]) || (S >> v & 1) || !t->deg[v]) continue;
    if (n == ROUTE_CONTEXTS) return -1;
    if (pattern) { pattern[n] = j; node[n] = v; owed[n] = S; }
    n++;
  }
  return n;
}

float route_draw_score(const struct route_draw *d, const float *w, float *score) {
  const struct split_set a = split_set_of(d);
  struct split_lse acc; split_lse_init(&acc);
  for (int f = 0; f < ROUTE_FS; f++) score[f] = 0;
  const struct split tsp = split_of(d, d->token);
  if (!tsp.ok) { for (int f = 0; f < ROUTE_FS; f++) score[f] = NAN; return -INFINITY; }
  const float tth = split_tanh(split_dot(&tsp, w));
  split_add(&tsp, split_slope(tth), score);
  struct split_walk k = split_walk_at(&a, 0, a.n);
  for (; k.left; split_walk_next(&k)) {
    const struct split sp = split_of(d, k.token);
    const float th = split_tanh(split_dot(&sp, w));
    split_lse_add(&acc, split_logit(th), split_slope(th), &sp);
  }
  for (int f = 0; f < ROUTE_FS; f++) score[f] -= acc.e[f] / acc.s;
  return split_logit(tth) - (acc.m + logf(acc.s));
}


/* ---- one run: the bridges' own code on the wire (tests/switched-world.h) ----
   A run is a stream's collectives on its fabric, run by the bridges: each node a region served by mesh-switched.h,
   the wire the hardware's.  At each refresh of the window the policy is asked for its tables: the decision's tokens
   are the bridges' own (mesh_switch_tokens_make at every node: a lane's token made by its sending node, a context's
   by the node it is at), and the answer reaches each bridge through struct mesh_switch_policy
   (mesh_switch_policy_refresh, the deployed path), active from its refresh (the first) or `lag` after it.  Every
   piece's split is the bridge's draw (mesh_switch_choose: split.h's table under its context's w); every draw the
   bridge logs (struct mesh_switch_record: its route_draw, its context and the tables' refresh) is a record of the
   refresh it drew from.  Before the first table, the warm router: 2 the bridges' fixed weights (the unloaded
   shortest-delay neighbour), 3 every context's w zero (uniform over the allowed splits).  Modes 2 and 4 are the fixed
   weights throughout, 3 the zero tables throughout.  Nothing is written anywhere but this memory. */
#include "../switched-world.h"

int route_argmax;   /* route.h: the bridges' draws take their table's most probable split */
struct piece { uint64_t key; uint32_t source, dest, transfer, span, op; float latency, queueing; };
struct draw { struct route_draw r; uint32_t context; uint64_t key; uint32_t source, dest, transfer, span; float logp; };
struct drawlog { struct draw *d; uint32_t count, room; };
struct route_sim {
  struct route_stream s; struct route_window win; struct route_topology t;
  int mode, nc, point, active, stopped; double tend;
  uint32_t key0, key;
  int cj[ROUTE_CONTEXTS], cv[ROUTE_CONTEXTS]; uint32_t cS[ROUTE_CONTEXTS];
  int local[ROUTE_N][MESH_SWITCH_CONTEXTS]; uint32_t nlocal[ROUTE_N];   /* a node's context -> the stream's */
  struct sw_world *w; struct sw_link links[ROUTE_L]; uint32_t nlinks, ranks; double duplex;
  int nextop, opslot[SW_OPS]; double opstart[ROUTE_OPS], opend[ROUTE_OPS];
  int pending; double pending_at;                     /* an answer waiting for its activation */
  float *wt;                                          /* [POINTS][CONTEXTS][FS] */
  struct drawlog log[ROUTE_POINTS];
  struct piece *pieces; uint32_t npieces, pieceroom;
  struct mesh_switch_tokens tok;
  int publishing;                                     /* the node whose tables mesh_switch_policy_refresh is making */
  uint64_t count[4];
};

static void on_op(void *ctx, uint32_t slot, double end) {
  struct route_sim *x = ctx; const int o = x->opslot[slot];
  if (o >= 0) { x->opend[o] = end; x->opslot[slot] = -1; }
}
static int op_of_key(const struct route_sim *x, uint64_t key) {
  for (uint32_t i = 0; i < SW_OPS; i++) if (x->w->op[i].key == key && x->opslot[i] >= 0) return x->opslot[i];
  return -1;
}
static void on_record(void *ctx, const struct mesh_switch_record *r) {
  struct route_sim *x = ctx;
  if (r->kind == MESH_LOG_DRAW) {
    if (r->context == UINT32_MAX || x->active < 0 || x->mode != 0) return;   /* the weights drew, not the policy's tables */
    const uint32_t v = r->node - 1;
    if (v >= ROUTE_N || r->context >= x->nlocal[v] || x->local[v][r->context] < 0) return;
    struct drawlog *g = &x->log[r->point % ROUTE_POINTS];
    if (g->count == g->room) { g->room = g->room ? 2 * g->room : 256; g->d = realloc(g->d, g->room * sizeof *g->d); if (!g->d) abort(); }
    g->d[g->count++] = (struct draw){r->draw, (uint32_t)x->local[v][r->context], r->key, r->source, r->dest, r->transfer, r->span, r->logp};
    x->count[2] += split_set_of(&r->draw).n;
  } else if (r->kind == MESH_LOG_PIECE) {
    const int o = op_of_key(x, r->key);
    if (o < 0) return;
    if (x->npieces == x->pieceroom) { x->pieceroom = x->pieceroom ? 2 * x->pieceroom : 4096; x->pieces = realloc(x->pieces, x->pieceroom * sizeof *x->pieces); if (!x->pieces) abort(); }
    x->pieces[x->npieces++] = (struct piece){r->key, r->source, r->node, r->transfer, r->span, (uint32_t)o, r->latency, r->queueing};
  }
}

struct route_sim *route_sim_new(void) {
  struct route_sim *x = calloc(1, sizeof *x);
  if (!x) abort();
  x->wt = calloc((size_t)ROUTE_POINTS * ROUTE_CONTEXTS * ROUTE_FS, sizeof(float));
  if (!x->wt) abort();
  return x;
}
void route_sim_free(struct route_sim *x) {
  for (int k = 0; k < ROUTE_POINTS; k++) free(x->log[k].d);
  if (x->w) sw_world_free(x->w);
  free(x->wt); free(x->pieces); free(x);
}
int route_sim_points(const struct route_sim *x) { return x->point + 1; }
int route_sim_contexts(const struct route_sim *x) { return x->nc; }
float *route_sim_w(struct route_sim *x, int point) { return x->wt + (size_t)(point % ROUTE_POINTS) * ROUTE_CONTEXTS * ROUTE_FS; }
void route_sim_counts(const struct route_sim *x, uint64_t out[4]) {
  out[0] = x->w ? x->w->count[0] : 0; out[1] = x->w ? x->w->count[1] : 0; out[2] = x->count[2]; out[3] = x->w ? x->w->count[3] : 0;
}

/* the policy, as the bridge calls it: node `publishing`'s contexts' w, from the answered refresh's table (zero: uniform) */
static int policy_splits(void *ctx, const struct mesh_switch_tokens *tok, float w[MESH_SWITCH_CONTEXTS][SPLIT_FS]) {
  struct route_sim *x = ctx; const int v = x->publishing;
  const float *table = x->pending >= 0 ? route_sim_w(x, x->pending) : NULL;
  for (uint32_t c = 0; c < tok->contexts; c++) {
    const int g = c < x->nlocal[v] ? x->local[v][c] : -1;
    if (table && g >= 0) memcpy(w[c], table + (size_t)g * ROUTE_FS, ROUTE_FS * sizeof(float)); else memset(w[c], 0, sizeof w[c]);
  }
  return 0;
}
/* every node's tables made from refresh `point`'s answer (-1: zero tables), through the bridge's policy refresh */
static void publish(struct route_sim *x, int point) {
  const struct mesh_switch_policy policy = {x, NULL, policy_splits};
  x->pending = point;
  for (uint32_t v = 0; v < x->w->nodes; v++) {
    x->publishing = (int)v;
    mesh_switch_policy_refresh(x->w->node[v].m, x->w->node[v].sw, &policy, (uint64_t)x->w->now, point < 0 ? 0 : (uint64_t)point, route_argmax, &x->tok);
  }
  x->pending = -1;
  if (point >= 0 && point > x->active) x->active = point;
  sw_pass_all(x->w);
}
/* the run to `until`: the collectives arriving before it started as they arrive (a later one waits for a program
   slot where SW_OPS are live), every event before it */
static void run_to(struct route_sim *x, double until) {
  for (;;) {
    const int more = x->nextop < x->s.ops && (!x->stopped || x->s.op[x->nextop].arrival < x->tend);
    const double next = more ? (x->s.op[x->nextop].arrival > x->w->now ? x->s.op[x->nextop].arrival : x->w->now) : INFINITY;
    if (next >= until) { sw_run_until(x->w, until); return; }
    sw_run_until(x->w, next);
    const struct route_op *op = &x->s.op[x->nextop];
    const uint32_t slot = sw_op_start(x->w, op->kind, op->root, (uint64_t)op->size, op->done, x->w->now, 0);
    if (slot == UINT32_MAX) {   /* SW_OPS live: on once one ends */
      if (!x->w->events) abort();
      sw_run_until(x->w, x->w->heap[0].t + 1e-9);
      continue;
    }
    x->opslot[slot] = x->nextop; x->opstart[x->nextop] = x->w->now; x->nextop++;
    sw_pass_all(x->w);
  }
}

static int same_fabric(const struct route_sim *x, const struct route_fabric *f) {
  if (!x->w || x->w->nodes != (uint32_t)f->nodes || x->nlinks != (uint32_t)f->lanes || x->duplex != f->duplex) return 0;
  uint32_t ranks = 0; for (int k = 0; k < f->ncompute; k++) ranks |= 1u << f->compute[k];
  if (ranks != x->ranks) return 0;
  for (int l = 0; l < f->lanes; l++)
    if (x->links[l].from != (uint32_t)f->from[l] || x->links[l].to != (uint32_t)f->to[l] || x->links[l].cables != (uint32_t)f->cables[l] ||
        x->links[l].rate != f->rate[l] || x->links[l].latency != f->latency[l]) return 0;
  return 1;
}
void route_sim_begin(struct route_sim *x, const struct route_stream *s, const struct route_window *w, int mode, uint32_t key0, uint32_t key) {
  x->s = *s; x->win = *w; x->mode = mode; x->key0 = key0; x->key = key;
  route_topology_make(&s->f, s, &x->t);
  int pj[ROUTE_CONTEXTS], pv[ROUTE_CONTEXTS]; uint32_t pS[ROUTE_CONTEXTS];
  x->nc = route_contexts(s, &x->t, pj, pv, pS);
  if (x->nc < 0) abort();
  for (int c = 0; c < x->nc; c++) { x->cj[c] = pj[c]; x->cv[c] = pv[c]; x->cS[c] = pS[c]; }
  if (!same_fabric(x, &s->f)) {
    if (x->w) sw_world_free(x->w);
    x->nlinks = (uint32_t)s->f.lanes; x->duplex = s->f.duplex; x->ranks = 0;
    for (int k = 0; k < s->f.ncompute; k++) x->ranks |= 1u << s->f.compute[k];
    for (int l = 0; l < s->f.lanes; l++) x->links[l] = (struct sw_link){(uint32_t)s->f.from[l], (uint32_t)s->f.to[l], (uint32_t)s->f.cables[l], s->f.rate[l], s->f.latency[l]};
    x->w = sw_world_new((uint32_t)s->f.nodes, x->ranks, x->links, x->nlinks, s->f.duplex, 512, 160, 0);
    x->w->on_record = on_record; x->w->ctx = x; x->w->on_op = on_op; x->w->op_ctx = x;
    /* each node's contexts (the bridge's enumeration) against the stream's */
    for (uint32_t v = 0; v < x->w->nodes && v < ROUTE_N; v++) {
      struct mesh_switch_context keys[MESH_SWITCH_CONTEXTS];
      x->nlocal[v] = mesh_switch_contexts(&x->w->node[v].sw->view, keys);
      for (uint32_t c = 0; c < x->nlocal[v]; c++) {
        x->local[v][c] = -1;
        for (int g = 0; g < x->nc; g++)
          if (x->cv[g] == (int)v && (uint32_t)s->source[x->cj[g]] == keys[c].source && s->dest[x->cj[g]] == keys[c].wanted && x->cS[g] == keys[c].owed) { x->local[v][c] = g; break; }
      }
    }
  }
  for (int k = 0; k < ROUTE_POINTS; k++) x->log[k].count = 0;
  x->npieces = 0; x->point = -1; x->active = -1; x->pending = -1; x->nextop = 0; x->count[2] = 0;
  for (uint32_t i = 0; i < SW_OPS; i++) x->opslot[i] = -1;
  for (int o = 0; o < s->ops; o++) { x->opstart[o] = -1; x->opend[o] = -1; }
  x->tend = w->points > 0 ? w->t0 + w->points * w->dt : INFINITY; x->stopped = w->points > 0 && mode == 0;
  const int router = mode == 0 ? w->warm : mode;
  x->w->fixed = router == 2 || router == 4;
  double background[ROUTE_L]; for (int l = 0; l < s->f.lanes; l++) background[l] = s->background[l];
  sw_world_reset(x->w, background, (uint64_t)key0 << 32 | key);
  if (router == 3) publish(x, -1);
  x->active = -1;
}

int route_sim_decide(struct route_sim *x, struct route_decision *d, float *tokens) {
  const int k = x->point + 1;
  const double tau = x->win.t0 + k * x->win.dt;
  if (x->mode != 0 || (x->win.points > 0 && k >= x->win.points)) {
    if (x->pending >= 0) { run_to(x, x->pending_at); publish(x, x->pending); }
    run_to(x, INFINITY); return 0;
  }
  if (x->pending >= 0 && x->pending_at <= tau) { run_to(x, x->pending_at); publish(x, x->pending); }
  run_to(x, tau);
  if (x->win.points == 0 && k > 0 && !x->w->events && x->nextop >= x->s.ops) return 0;
  x->point = k;
  const struct route_fabric *f = &x->s.f;
  memset(d, 0, sizeof *d);
  d->point = (unsigned)k; d->lanes = (unsigned)f->lanes; d->contexts = (unsigned)x->nc;
  memset(tokens, 0, (size_t)x->nc * ROUTE_IN * sizeof(float));
  for (uint32_t v = 0; v < x->w->nodes; v++) {
    mesh_switch_tokens_make(x->w->node[v].m, x->w->node[v].sw, (uint64_t)x->w->now, (uint64_t)k, &x->tok);
    for (uint32_t i = 0; i < x->tok.lanes; i++) if (x->tok.link_of[i] < ROUTE_L) memcpy(d->lane[x->tok.link_of[i]], x->tok.lane[i], sizeof d->lane[0]);
    for (uint32_t c = 0; c < x->tok.contexts && c < x->nlocal[v]; c++) {
      const int g = x->local[v][c];
      if (g >= 0) memcpy(tokens + (size_t)g * ROUTE_IN, x->tok.context[c], ROUTE_IN * sizeof(float));
    }
  }
  return 1;
}
void route_sim_answer(struct route_sim *x, const float *w) {
  const int k = x->point;
  if (w) memcpy(route_sim_w(x, k), w, (size_t)x->nc * ROUTE_FS * sizeof(float));
  else memset(route_sim_w(x, k), 0, (size_t)x->nc * ROUTE_FS * sizeof(float));
  if (k == 0) {
    if (x->w->fixed) {   /* off the warm router's fixed weights: the bridges' refreshes from now */
      x->w->fixed = 0; sw_push(x->w, (struct sw_event){x->w->now + x->w->refresh_ns, SW_EV_REFRESH, 0, 0, 0, 0, 0});
    }
    publish(x, 0);
  } else { x->pending = k; x->pending_at = x->win.t0 + k * x->win.dt + x->win.lag; }
}
void route_sim_costs(const struct route_sim *x, struct route_costs *c) {
  memset(c, 0, sizeof *c);
  for (int o = 0; o < x->s.ops; o++)
    if (x->opstart[o] >= 0 && x->opend[o] >= 0) { c->throughput += (x->opend[o] - x->s.op[o].arrival) / x->s.op[o].floor; c->ops++; }
  for (uint32_t p = 0; p < x->npieces; p++) {
    const double fl = x->s.op[x->pieces[p].op].floor;
    c->latency += x->pieces[p].latency / fl; c->queueing += x->pieces[p].queueing / fl; c->pieces++;
  }
  if (c->ops) c->throughput /= c->ops;
  if (c->pieces) { c->latency /= c->pieces; c->queueing /= c->pieces; }
}

int route_sim_draw_count(const struct route_sim *x, int point) { return (int)x->log[point % ROUTE_POINTS].count; }
static int by_context(void *x, const void *a, const void *b) {   /* BSD qsort_r: the context, then the order made */
  const struct draw *g = x;
  const uint32_t ca = g[*(const uint32_t *)a].context, cb = g[*(const uint32_t *)b].context;
  return ca < cb ? -1 : ca > cb ? 1 : (*(const uint32_t *)a < *(const uint32_t *)b ? -1 : 1);
}
static int by_piece(const void *a, const void *b) {
  const struct piece *p = a, *q = b;
  if (p->key != q->key) return p->key < q->key ? -1 : 1;
  if (p->source != q->source) return p->source < q->source ? -1 : 1;
  if (p->dest != q->dest) return p->dest < q->dest ? -1 : 1;
  if (p->transfer != q->transfer) return p->transfer < q->transfer ? -1 : 1;
  return p->span < q->span ? -1 : p->span > q->span;
}
/* refresh `point`'s draws by context; each one's advantages: its piece's latency and queueing (as delivered to the
   draw's first destination, over its collective's floor) against the mean of the other pieces of its collective
   from the same source, over the run's pieces */
int route_sim_draws(struct route_sim *x, int point, struct route_draw *out, int *ctx, int *first) {
  qsort(x->pieces, x->npieces, sizeof *x->pieces, by_piece);
  /* each (collective, source)'s sums */
  double *sums = calloc((size_t)ROUTE_OPS * ROUTE_N * 3 + 1, sizeof(double));
  if (!sums) abort();
  for (uint32_t p = 0; p < x->npieces; p++) {
    const struct piece *q = &x->pieces[p]; const double fl = x->s.op[q->op].floor;
    double *e = sums + ((size_t)q->op * ROUTE_N + (q->source - 1) % ROUTE_N) * 3; e[0] += q->latency / fl; e[1] += q->queueing / fl; e[2] += 1;
  }
  const double n = x->npieces ? (double)x->npieces : 1;
  const struct drawlog *g = &x->log[point % ROUTE_POINTS];
  uint32_t *order = malloc((g->count + 1) * sizeof(uint32_t));
  if (!order) abort();
  for (uint32_t i = 0; i < g->count; i++) order[i] = i;
  qsort_r(order, g->count, sizeof *order, g->d, by_context);
  int at = 0, rows = 0;
  for (uint32_t i = 0; i < g->count; i++) {
    const struct draw *d = &g->d[order[i]];
    const int c = (int)d->context;
    if (!rows || ctx[rows - 1] != c) { ctx[rows] = c; first[rows] = at; rows++; }
    struct route_draw *r = &out[at++];
    *r = d->r; r->row = (uint32_t)(rows - 1); r->a1 = 0; r->a2 = 0;
    const struct piece want = {d->key, d->source, d->dest, d->transfer, d->span, 0, 0, 0};
    const struct piece *q = bsearch(&want, x->pieces, x->npieces, sizeof *x->pieces, by_piece);
    if (q) {
      const double fl = x->s.op[q->op].floor; const double *e = sums + ((size_t)q->op * ROUTE_N + (q->source - 1) % ROUTE_N) * 3;
      if (e[2] > 1) {
        r->a1 = (float)((q->latency / fl - (e[0] - q->latency / fl) / (e[2] - 1)) / n);
        r->a2 = (float)((q->queueing / fl - (e[1] - q->queueing / fl) / (e[2] - 1)) / n);
      }
    }
  }
  first[rows] = at;
  free(sums); free(order);
  return rows;
}

/* the largest gap between a draw's log-probability as the bridge drew it and split.h's under its refresh's w (the
   trainer's: route_draw_score): 0 where the behaviour is the policy's table exactly */
float route_sim_logp_gap(const struct route_sim *x) {
  float gap = 0, score[ROUTE_FS];
  for (int k = 0; k <= x->point && k < ROUTE_POINTS; k++)
    for (uint32_t i = 0; i < x->log[k].count; i++) {
      const struct draw *d = &x->log[k].d[i];
      const float lp = route_draw_score(&d->r, x->wt + ((size_t)k * ROUTE_CONTEXTS + d->context) * ROUTE_FS, score);
      if (fabsf(lp - d->logp) > gap) gap = fabsf(lp - d->logp);
    }
  return gap;
}

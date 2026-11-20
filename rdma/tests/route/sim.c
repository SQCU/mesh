/* The environment (route.h): a stream of collectives on a fabric, run by the max-plus FIFO recursion over discrete
   pieces with tests/switch.c's semantics. Every lane is its cables, a pair's rate shared equally among them; a piece
   crossing a lane takes whichever cable frees first: start = max(arrival, cable free), at the far node start + bytes /
   cable rate + latency, the cable running at the duplex share of its rate while the reverse direction's same cable is
   busy then; a node forwards a piece the moment it lands. Events leave in time order through switch.c's heap.

   A collective arrives as flows (route.h's patterns); a flow posts its next released piece the moment its last has
   started on the wire at its source (mode 4: every piece the moment it is released, as switch.c), so each piece's split
   at its source is drawn when it leaves, like its splits downstream. An all-reduce's reduced page is released at its
   owner when the page's last partial lands there.

   The policy is asked for a table at each refresh of the run's window (route_window); a refresh's table is active from
   its refresh (the first) or `lag` after it (the rest); before the first, the warm router routes. A piece landing at v
   still owing S draws its split from context (j, v, S) of the active table, masked to the lanes that leave its path and
   lead to their share without coming back (by rejection, then enumeration: the masked table exactly). Every draw from a
   table is logged under its (refresh, context, mask) group, for the credit. */
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
static double normal(struct route_rng *g) { const double u = route_uniform(g), v = route_uniform(g); return sqrt(-2 * log(u > 1e-300 ? u : 1e-300)) * cos(2 * M_PI * v); }
static uint32_t mix(uint32_t h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
static double unit01(uint32_t h) { return (double)(mix(h) >> 8) / 16777216.0; }

/* ---- fabrics ---- */
int route_fabric_load(const char *path, struct route_fabric *f) {   /* node NAME compute|relay; link FROM TO GBps us cables; duplex SHARE */
  FILE *in = fopen(path, "r");
  if (!in) return -1;
  char line[256], names[ROUTE_N][16]; memset(f, 0, sizeof *f); f->duplex = 1;
  while (fgets(line, sizeof line, in)) {
    char a[16], b[16]; double r, us; int c;
    if (line[0] == '#') continue;
    if (sscanf(line, "duplex %lf", &r) == 1) f->duplex = r;
    else if (sscanf(line, "node %15s %15s", a, b) == 2) { if (f->nodes == ROUTE_N) { fclose(in); return -1; } strcpy(names[f->nodes], a); if (!strcmp(b, "compute")) f->compute[f->ncompute++] = f->nodes; f->nodes++; }
    else if (sscanf(line, "link %15s %15s %lf %lf %d", a, b, &r, &us, &c) == 5) {
      int u = -1, v = -1;
      for (int k = 0; k < f->nodes; k++) { if (!strcmp(names[k], a)) u = k; if (!strcmp(names[k], b)) v = k; }
      if (u < 0 || v < 0 || f->lanes == ROUTE_L) { fclose(in); return -1; }
      f->from[f->lanes] = u; f->to[f->lanes] = v; f->rate[f->lanes] = r; f->latency[f->lanes] = us * 1e3; f->cables[f->lanes] = c; f->lanes++;
    }
  }
  fclose(in);
  return 0;
}
static void cabled(struct route_fabric *f, int n, const int *compute, int nc, const int (*pairs)[2], int np, struct route_rng *g) {
  memset(f, 0, sizeof *f);
  f->nodes = n; f->ncompute = nc; for (int k = 0; k < nc; k++) f->compute[k] = compute[k];
  f->duplex = 0.85 + 0.15 * route_uniform(g);
  for (int k = 0; k < np; k++) {
    const int c = 1 + (int)(route_uniform(g) * 4); const double each = 4 + 8 * route_uniform(g);
    for (int dir = 0; dir < 2; dir++) {
      const int l = f->lanes++;
      f->from[l] = pairs[k][dir]; f->to[l] = pairs[k][1 - dir]; f->rate[l] = c * each; f->latency[l] = 2e3 + 11e3 * route_uniform(g); f->cables[l] = c;
    }
  }
}
void route_network(struct route_rng *g, const struct route_fabric *measured, struct route_fabric *f) {
  const int kind = (int)(route_uniform(g) * 6); int n = 3 + (int)(route_uniform(g) * 4);
  int pairs[16][2], np = 0, compute[ROUTE_N], nc = 0;
  switch (kind) {
  case 0: {   /* a random connected graph, degree at most 4, up to two relays */
    int degree[ROUTE_N] = {0}, joined[ROUTE_N][ROUTE_N] = {{0}};
    for (int v = 1; v < n; v++) { const int u = (int)(route_uniform(g) * v); pairs[np][0] = u; pairs[np][1] = v; np++; joined[u][v] = 1; degree[u]++; degree[v]++; }
    for (int u = 0; u < n; u++) for (int v = u + 1; v < n; v++)
      if (!joined[u][v] && degree[u] < 4 && degree[v] < 4 && route_uniform(g) < 0.35) { pairs[np][0] = u; pairs[np][1] = v; np++; degree[u]++; degree[v]++; }
    int relays = (int)(route_uniform(g) * 3); if (relays > n - 2) relays = n - 2;
    for (int v = 0; v < n - relays; v++) compute[nc++] = v;
    break;
  }
  case 1: for (int v = 0; v < n; v++) { pairs[np][0] = v; pairs[np][1] = (v + 1) % n; np++; } if (n == 3) np--; for (int v = 0; v < n; v++) compute[nc++] = v; break;
  case 2: { for (int v = 1; v < n; v++) { pairs[np][0] = 0; pairs[np][1] = v; np++; } const int hub = route_uniform(g) < 0.5; for (int v = hub ? 1 : 0; v < n; v++) compute[nc++] = v; break; }
  case 3: {   /* a ring of compute nodes and a relay hub */
    int m = n - 1; if (m < 3) m = 3; if (m > ROUTE_N - 1) m = ROUTE_N - 1;
    for (int v = 0; v < m; v++) { pairs[np][0] = v; pairs[np][1] = (v + 1) % m; np++; }
    for (int v = 0; v < 3 && v < m; v++) { pairs[np][0] = m; pairs[np][1] = v; np++; }
    n = m + 1; for (int v = 0; v < m; v++) compute[nc++] = v;
    break;
  }
  case 4: for (int u = 0; u < n; u++) for (int v = u + 1; v < n; v++) { pairs[np][0] = u; pairs[np][1] = v; np++; } for (int v = 0; v < n; v++) compute[nc++] = v; break;
  default:   /* the measured cluster, jittered */
    *f = *measured;
    for (int l = 0; l < f->lanes; l++) { f->rate[l] *= exp(0.3 * normal(g)); f->latency[l] *= exp(0.3 * normal(g)); }
    return;
  }
  cabled(f, n, compute, nc, (const int (*)[2])pairs, np, g);
}

/* ---- streams ---- */
static void delays(const struct route_fabric *f, double q, double (*dist)[ROUTE_N]) {   /* one q-byte piece's unloaded shortest delays */
  for (int a = 0; a < ROUTE_N; a++) for (int b = 0; b < ROUTE_N; b++) dist[a][b] = a == b ? 0 : INFINITY;
  for (int l = 0; l < f->lanes; l++) { const double w = f->latency[l] + q / (f->rate[l] / f->cables[l]); if (w < dist[f->from[l]][f->to[l]]) dist[f->from[l]][f->to[l]] = w; }
  for (int m = 0; m < ROUTE_N; m++) for (int a = 0; a < ROUTE_N; a++) for (int b = 0; b < ROUTE_N; b++) if (dist[a][m] + dist[m][b] < dist[a][b]) dist[a][b] = dist[a][m] + dist[m][b];
}
/* a collective's floor: its cut bound (the most bytes a node must take in or put out, over its rate) and its slowest
   pair's shortest delay for one piece, at most a page, as tests/switch.c's (for an all-reduce, the slowest owner's slowest
   partial in plus its slowest reduced page out: every page goes both ways through its owner) */
static double op_floor(const struct route_fabric *f, int kind, double S, int root) {
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
static double size_draw(struct route_rng *g) { return round(exp(log(6144.0) + route_uniform(g) * (log(12582912.0) - log(6144.0)))); }
void route_stream_make(const struct route_fabric *f, struct route_rng *g, double duration, double load, int background, struct route_stream *s) {
  stream_base(f, s);
  for (int l = 0; l < f->lanes; l++) s->background[l] = background && route_uniform(g) < 0.3 ? f->rate[l] * 50e3 * route_uniform(g) : 0;
  double t = 0;
  while (s->ops < ROUTE_OPS && t <= duration) {
    struct route_op o;
    o.kind = (int)(route_uniform(g) * ROUTE_KINDS); if (o.kind >= ROUTE_KINDS) o.kind = ROUTE_KINDS - 1;
    o.root = f->compute[(int)(route_uniform(g) * f->ncompute) % f->ncompute]; o.size = size_draw(g);
    o.floor = op_floor(f, o.kind, o.size, o.root); o.arrival = t;
    s->op[s->ops++] = o;
    t += o.floor / load * -log(1 - route_uniform(g) * 0.999999);
  }
}
void route_stream_one(const struct route_fabric *f, int kind, double size, int root, struct route_stream *s) {
  stream_base(f, s);
  s->op[0] = (struct route_op){kind, root, size, 0, op_floor(f, kind, size, root)}; s->ops = 1;
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

/* ---- tables and draws ---- */
static void build_table(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int v, uint32_t S, const struct route_live *live,
                        const float *w, double *cdf) {
  const int n = route_tokens(t, v, S); double most = -INFINITY;
  for (int k = 0; k < n; k++) {
    float phi[ROUTE_FS]; double a = 0;
    const int valid = route_split_features(s, st, t, v, S, (uint32_t)k, live, phi);
    for (int i = 0; i < ROUTE_FS; i++) a += (double)w[i] * phi[i];
    cdf[k] = valid ? ROUTE_CLIP * tanh(a / sqrt((double)ROUTE_FS)) : -INFINITY;
    if (cdf[k] > most) most = cdf[k];
  }
  double sum = 0, acc = 0;
  for (int k = 0; k < n; k++) sum += isfinite(cdf[k]) ? exp(cdf[k] - most) : 0;
  for (int k = 0; k < n; k++) { acc += isfinite(cdf[k]) ? exp(cdf[k] - most) / sum : 0; cdf[k] = acc; }
  cdf[n - 1] = 1;
}
uint32_t route_allowed(const struct route_stream *s, const struct route_topology *t, int v, uint32_t S, uint32_t visited) {
  const uint32_t avoid = visited | 1u << v; uint32_t mask = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (!(S >> d & 1)) continue;
    for (int k = 0; k < t->deg[v]; k++) {
      const int h = s->f.to[t->out[v][k]];
      if (!(avoid >> h & 1) && (h == d || route_reaches(t, h, avoid, 1u << d))) mask |= 1u << (ROUTE_DEG * d + k);
    }
  }
  return mask;
}
static int fits(uint32_t tok, uint32_t S, uint32_t mask, int deg) {
  for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) { if (!(mask >> (ROUTE_DEG * d + (int)(tok % (uint32_t)deg)) & 1)) return 0; tok /= (uint32_t)deg; }
  return 1;
}
static uint32_t sample(const double *cdf, int n, uint32_t S, uint32_t mask, int deg, uint32_t seed) {   /* the masked table, exactly */
  for (uint32_t attempt = 0; attempt < 64; attempt++) {
    const double u = unit01(seed ^ attempt * 0x9e3779b1u);
    int lo = 0, hi = n - 1;
    while (lo < hi) { const int mid = (lo + hi) / 2; if (cdf[mid] > u) hi = mid; else lo = mid + 1; }
    if (fits((uint32_t)lo, S, mask, deg)) return (uint32_t)lo;
  }
  double total = 0;
  for (int q = 0; q < n; q++) if (fits((uint32_t)q, S, mask, deg)) total += cdf[q] - (q ? cdf[q - 1] : 0);
  const double u = unit01(seed ^ 0x27d4eb2du) * total; double acc = 0; uint32_t tok = 0;
  for (int q = 0; q < n; q++) if (fits((uint32_t)q, S, mask, deg)) { tok = (uint32_t)q; acc += cdf[q] - (q ? cdf[q - 1] : 0); if (acc > u) break; }
  return tok;
}
uint32_t route_split(const struct route_stream *s, const struct route_state *st, const struct route_topology *t, int v, uint32_t S, uint32_t visited,
                     const struct route_live *live, const float *w, double u, double *cdf) {
  build_table(s, st, t, v, S, live, w, cdf);
  return sample(cdf, route_tokens(t, v, S), S, route_allowed(s, t, v, S, visited), t->deg[v], (uint32_t)(u * 4294967296.0));
}

/* ---- one run ---- */
enum { EV_ARRIVE, EV_POST, EV_OP, EV_ACTIVATE };
struct event { double t; int type, a; uint32_t node, owed, visited; };
struct flow { int op, pattern, src, np, base, head, tail, phase2, idle; uint32_t dest; uint32_t delivered[ROUTE_N]; };
struct draw { uint32_t token, piece; struct route_live live; };
struct group { uint32_t point, context, mask, count, room; struct draw *draws; };
struct route_sim {
  struct route_stream s; struct route_window win; struct route_topology t;
  int mode, nc, point, active, stopped; double tend;
  uint32_t key0, key;
  int cj[ROUTE_CONTEXTS], cv[ROUTE_CONTEXTS], ctx[ROUTE_J][ROUTE_N][ROUTE_SETS]; uint32_t cS[ROUTE_CONTEXTS];
  double free_at[ROUTE_L][ROUTE_CABLES];
  struct event *heap; size_t events, room;
  struct flow *flows; int nflows, flowroom;
  int *ready;                                         /* each flow's released pieces, a queue [base, base + np) */
  double *bytes, *release, *latency, *queued; int *flowof, *postpoint; int npieces, pieceroom;
  int *partial;                                       /* an all-reduce's partials landed, by its reduced pieces */
  double opend[ROUTE_OPS]; int opleft[ROUTE_OPS], opstarted[ROUTE_OPS];
  struct route_state state[ROUTE_POINTS];
  float *w;                                           /* [POINTS][CONTEXTS][FS], refresh k at k % POINTS */
  double cdf[3125];                                  /* a draw's table */
  struct group *groups; uint32_t ngroups, groupsroom, *hash, hashroom;
  uint64_t count[4];
};

static void push(struct route_sim *x, struct event e) {   /* tests/switch.c's heap */
  if (x->events == x->room) { x->room = x->room ? 2 * x->room : 4096; x->heap = realloc(x->heap, x->room * sizeof *x->heap); if (!x->heap) abort(); }
  size_t i = x->events++;
  while (i && x->heap[(i - 1) / 2].t > e.t) { x->heap[i] = x->heap[(i - 1) / 2]; i = (i - 1) / 2; }
  x->heap[i] = e;
}
static struct event pop(struct route_sim *x) {
  struct event top = x->heap[0], last = x->heap[--x->events];
  size_t i = 0;
  for (;;) {
    size_t c = 2 * i + 1;
    if (c >= x->events) break;
    if (c + 1 < x->events && x->heap[c + 1].t < x->heap[c].t) c++;
    if (x->heap[c].t >= last.t) break;
    x->heap[i] = x->heap[c]; i = c;
  }
  if (x->events) x->heap[i] = last;
  return top;
}

struct route_sim *route_sim_new(void) {
  struct route_sim *x = calloc(1, sizeof *x);
  if (!x) abort();
  x->w = calloc((size_t)ROUTE_POINTS * ROUTE_CONTEXTS * ROUTE_FS, sizeof(float));
  x->partial = calloc((size_t)ROUTE_OPS * ROUTE_N * ROUTE_PAGE_PIECES, sizeof(int));
  x->hashroom = 1u << 14; x->hash = calloc(x->hashroom, sizeof(uint32_t));
  if (!x->w || !x->partial || !x->hash) abort();
  return x;
}
void route_sim_free(struct route_sim *x) {
  for (uint32_t g = 0; g < x->groupsroom; g++) free(x->groups[g].draws);
  free(x->groups); free(x->hash); free(x->heap); free(x->w); free(x->partial); free(x->flows); free(x->ready);
  free(x->bytes); free(x->release); free(x->latency); free(x->queued); free(x->flowof); free(x->postpoint); free(x);
}
int route_sim_points(const struct route_sim *x) { return x->point + 1; }
int route_sim_contexts(const struct route_sim *x) { return x->nc; }
float *route_sim_w(struct route_sim *x, int point) { return x->w + (size_t)(point % ROUTE_POINTS) * ROUTE_CONTEXTS * ROUTE_FS; }
void route_sim_counts(const struct route_sim *x, uint64_t out[4]) { memcpy(out, x->count, sizeof x->count); }

static int pattern_of(const struct route_sim *x, int src, uint32_t dest) {
  for (int j = 0; j < x->s.patterns; j++) if (x->s.source[j] == src && x->s.dest[j] == dest) return j;
  abort();
}
static int new_flow(struct route_sim *x, int op, int src, uint32_t dest, int np) {
  if (x->nflows == x->flowroom) { x->flowroom = x->flowroom ? 2 * x->flowroom : 64; x->flows = realloc(x->flows, x->flowroom * sizeof *x->flows); if (!x->flows) abort(); }
  if (x->npieces + np > x->pieceroom) {
    while (x->npieces + np > x->pieceroom) x->pieceroom = x->pieceroom ? 2 * x->pieceroom : 8192;
    const size_t m = (size_t)x->pieceroom;
    x->bytes = realloc(x->bytes, m * sizeof(double)); x->release = realloc(x->release, m * sizeof(double)); x->latency = realloc(x->latency, m * sizeof(double));
    x->queued = realloc(x->queued, m * sizeof(double)); x->flowof = realloc(x->flowof, m * sizeof(int));
    x->postpoint = realloc(x->postpoint, m * sizeof(int)); x->ready = realloc(x->ready, m * sizeof(int));
    if (!x->bytes || !x->release || !x->latency || !x->queued || !x->flowof || !x->postpoint || !x->ready) abort();
  }
  const int f = x->nflows++;
  struct flow *q = &x->flows[f];
  memset(q, 0, sizeof *q);
  q->op = op; q->src = src; q->dest = dest; q->np = np; q->base = x->npieces; q->pattern = pattern_of(x, src, dest); q->phase2 = -1; q->idle = 1;
  for (int i = 0; i < np; i++) { const int p = q->base + i; x->release[p] = INFINITY; x->latency[p] = 0; x->queued[p] = 0; x->flowof[p] = f; x->postpoint[p] = -2; }
  x->npieces += np;
  return f;
}
static double split_send(struct route_sim *x, double t, int v, uint32_t S, uint32_t visited, int p);
static void release(struct route_sim *x, int f, int i, double t) {   /* piece i of flow f is ready at its source */
  struct flow *q = &x->flows[f];
  x->release[q->base + i] = t;
  if (x->mode == 4 || (x->mode == 0 && x->win.burst)) {   /* posted the moment it is ready, as switch.c */
    q->head++; x->postpoint[q->base + i] = x->mode == 0 && x->active >= 0 ? x->active : -1; split_send(x, t, q->src, q->dest, 0, q->base + i); return;
  }
  x->ready[q->base + q->tail++] = i;
  if (q->idle) { q->idle = 0; push(x, (struct event){t, EV_POST, f, 0, 0, 0}); }
}
static void op_start(struct route_sim *x, int o, double t) {   /* a collective's flows */
  const struct route_op *op = &x->s.op[o]; const struct route_fabric *f = &x->s.f;
  const int n = f->ncompute, P = op->size > ROUTE_PAGE ? (int)ceil(op->size / ROUTE_PAGE) : 1;
  uint32_t all = 0; for (int k = 0; k < n; k++) all |= 1u << f->compute[k];
  x->opstarted[o] = 1; x->opleft[o] = 0; x->opend[o] = INFINITY;
  const double last = op->size - ROUTE_PAGE * (P - 1);
  if (op->kind == ROUTE_ALLGATHER || op->kind == ROUTE_SCATTER) {
    for (int k = 0; k < n; k++) {
      const int r = f->compute[k];
      if (op->kind == ROUTE_SCATTER && r == op->root) continue;
      const int src = op->kind == ROUTE_SCATTER ? op->root : r; const uint32_t dest = op->kind == ROUTE_SCATTER ? 1u << r : all & ~(1u << r);
      const int fl = new_flow(x, o, src, dest, P);
      for (int i = 0; i < P; i++) x->bytes[x->flows[fl].base + i] = i + 1 < P ? ROUTE_PAGE : last;
      x->opleft[o] += P * __builtin_popcount(dest);
      for (int i = 0; i < P; i++) release(x, fl, i, t);
    }
  } else {   /* all-reduce: page p's owner compute[p mod n]; partials to it, then the reduced page from it */
    int partials[ROUTE_N][ROUTE_N];
    for (int ko = 0; ko < n; ko++) {
      const int owner = f->compute[ko], np = (P - ko + n - 1) / n;
      for (int kr = 0; kr < n; kr++) partials[kr][ko] = -1;
      if (np <= 0) continue;
      const int red = new_flow(x, o, owner, all & ~(1u << owner), np);
      for (int i = 0; i < np; i++) { const int p = ko + i * n; x->bytes[x->flows[red].base + i] = p + 1 < P ? ROUTE_PAGE : last; x->partial[((size_t)o * ROUTE_N + ko) * ROUTE_PAGE_PIECES + i] = 0; }
      x->opleft[o] += np * (n - 1);
      for (int kr = 0; kr < n; kr++) {
        if (kr == ko) continue;
        const int fl = new_flow(x, o, f->compute[kr], 1u << owner, np);
        x->flows[fl].phase2 = red; partials[kr][ko] = fl;
        for (int i = 0; i < np; i++) x->bytes[x->flows[fl].base + i] = x->bytes[x->flows[red].base + i];
        x->opleft[o] += np;
      }
    }
    for (int kr = 0; kr < n; kr++) for (int p = 0; p < P; p++) if (p % n != kr) release(x, partials[kr][p % n], p / n, t);   /* switch.c's order: rank, then page */
  }
}

static void log_draw(struct route_sim *x, uint32_t c, uint32_t mask, uint32_t token, uint32_t piece, const struct route_live *live) {
  const uint64_t key = (uint64_t)x->active << 40 | (uint64_t)c << 30 | mask;
  if (2 * (x->ngroups + 1) > x->hashroom) {
    free(x->hash); x->hashroom *= 2; x->hash = calloc(x->hashroom, sizeof(uint32_t)); if (!x->hash) abort();
    for (uint32_t g = 0; g < x->ngroups; g++) {
      const uint64_t k = (uint64_t)x->groups[g].point << 40 | (uint64_t)x->groups[g].context << 30 | x->groups[g].mask;
      uint32_t h = mix((uint32_t)k ^ mix((uint32_t)(k >> 32))) & (x->hashroom - 1);
      while (x->hash[h]) h = (h + 1) & (x->hashroom - 1);
      x->hash[h] = g + 1;
    }
  }
  uint32_t h = mix((uint32_t)key ^ mix((uint32_t)(key >> 32))) & (x->hashroom - 1), g;
  for (;;) {
    const uint32_t e = x->hash[h];
    if (!e) {
      g = x->ngroups++;
      if (g == x->groupsroom) { x->groupsroom = x->groupsroom ? 2 * x->groupsroom : 256; x->groups = realloc(x->groups, x->groupsroom * sizeof *x->groups); if (!x->groups) abort(); for (uint32_t k = g; k < x->groupsroom; k++) { x->groups[k].draws = NULL; x->groups[k].room = 0; } }
      x->groups[g].point = (uint32_t)x->active; x->groups[g].context = c; x->groups[g].mask = mask; x->groups[g].count = 0;
      x->hash[h] = g + 1;
      break;
    }
    const struct group *q = &x->groups[e - 1];
    if (q->point == (uint32_t)x->active && q->context == c && q->mask == mask) { g = e - 1; break; }
    h = (h + 1) & (x->hashroom - 1);
  }
  struct group *q = &x->groups[g];
  if (q->count == q->room) { q->room = q->room ? 2 * q->room : 16; q->draws = realloc(q->draws, q->room * sizeof *q->draws); if (!q->draws) abort(); }
  q->draws[q->count++] = (struct draw){token, piece, *live};
}

/* piece p at node v at time t, owing S, its path through `visited`: its split, drawn and sent. Returns the earliest
   start of its sends. */
static double split_send(struct route_sim *x, double t, int v, uint32_t S, uint32_t visited, int p) {
  const struct route_fabric *f = &x->s.f; const struct flow *q = &x->flows[x->flowof[p]];
  const int deg = x->t.deg[v], j = q->pattern;
  const uint32_t mask = route_allowed(&x->s, &x->t, v, S, visited);
  struct route_live live;   /* v's own lanes now */
  for (int k = 0; k < deg; k++) {
    const int l = x->t.out[v][k], rl = x->t.rev[l]; double w = INFINITY; int busy = 0;
    for (int c = 0; c < f->cables[l]; c++) { const double q = x->free_at[l][c] > t ? x->free_at[l][c] - t : 0; if (q < w) w = q; }
    if (rl >= 0) for (int c = 0; c < f->cables[rl]; c++) busy = busy || x->free_at[rl][c] > t;
    live.wait[k] = w; live.dup[k] = busy ? f->duplex : 1;
  }
  const int router = x->mode == 0 ? (x->active >= 0 ? 0 : x->win.warm) : x->mode;
  const uint32_t seed = (router == 0 ? x->key : x->key0) ^ (uint32_t)p * 0x9e3779b1u ^ (uint32_t)v * 0x85ebca6bu ^ S * 0xc2b2ae35u;
  uint32_t tok = 0, place = 1;
  if (router == 2 || router == 4) {   /* the shortest-path tree's split */
    for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) {
      int y = d; while (f->from[x->s.spt[j][y]] != v) y = f->from[x->s.spt[j][y]];
      int k = 0; while (x->t.out[v][k] != x->s.spt[j][y]) k++;
      tok += (uint32_t)k * place; place *= (uint32_t)deg;
    }
  } else if (router == 3) {   /* each destination a lane uniformly among the allowed */
    for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) {
      const int m = __builtin_popcount(mask >> (ROUTE_DEG * d) & 31);
      int target = (int)(unit01(seed ^ (uint32_t)d * 0x27d4eb2du) * m), k = 0;
      if (target >= m) target = m - 1;
      for (; k < deg; k++) if (mask >> (ROUTE_DEG * d + k) & 1) { if (!target) break; target--; }
      tok += (uint32_t)k * place; place *= (uint32_t)deg;
    }
  } else {   /* a draw from the active table */
    const int c = x->ctx[j][v][S], n = route_tokens(&x->t, v, S);
    build_table(&x->s, &x->state[x->active % ROUTE_POINTS], &x->t, v, S, &live, route_sim_w(x, x->active) + (size_t)c * ROUTE_FS, x->cdf);
    tok = sample(x->cdf, n, S, mask, deg, seed);
    log_draw(x, (uint32_t)c, mask, tok, (uint32_t)p, &live); x->count[1]++; x->count[2] += (uint64_t)n;
  }
  uint32_t share[ROUTE_DEG] = {0}, y = tok;
  for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) { share[y % (uint32_t)deg] |= 1u << d; y /= (uint32_t)deg; }
  double first = INFINITY;
  for (int k = 0; k < deg; k++) {
    if (!share[k]) continue;
    const int l = x->t.out[v][k], h = f->to[l], rl = x->t.rev[l];
    int cb = 0; for (int r = 1; r < f->cables[l]; r++) if (x->free_at[l][r] < x->free_at[l][cb]) cb = r;
    const double start = x->free_at[l][cb] > t ? x->free_at[l][cb] : t;
    const int both = rl >= 0 && cb < f->cables[rl] && x->free_at[rl][cb] > start;
    const double time = x->bytes[p] / (f->rate[l] / f->cables[l] * (both ? f->duplex : 1));
    x->free_at[l][cb] = start + time; x->queued[p] += start - t;
    if (start < first) first = start;
    push(x, (struct event){start + time + f->latency[l], EV_ARRIVE, p, (uint32_t)h, share[k], visited | 1u << v}); x->count[3]++;
  }
  return first;
}
static void advance(struct route_sim *x, double until) {   /* every event up to `until`, in time order */
  const struct route_fabric *f = &x->s.f;
  while (x->events && x->heap[0].t <= until) {
    const struct event e = pop(x); x->count[0]++;
    if (e.type == EV_OP) { if (!x->stopped || x->s.op[e.a].arrival < x->tend) op_start(x, e.a, e.t); continue; }
    if (e.type == EV_ACTIVATE) { if (e.a > x->active) x->active = e.a; continue; }
    if (e.type == EV_POST) {   /* the flow's next released piece leaves its source; the next once it has started */
      for (;;) {
        struct flow *q = &x->flows[e.a];
        if (q->head == q->tail) { q->idle = 1; break; }
        const int p = q->base + x->ready[q->base + q->head++];
        x->postpoint[p] = x->mode == 0 && x->active >= 0 ? x->active : -1;
        const double first = split_send(x, e.t, q->src, q->dest, 0, p);
        if (x->mode != 4) { q = &x->flows[e.a]; push(x, (struct event){first > e.t ? first : e.t, EV_POST, e.a, 0, 0, 0}); break; }
      }
      continue;
    }
    const int p = e.a; uint32_t S = e.owed; struct flow *q = &x->flows[x->flowof[p]];
    if (S >> e.node & 1) {   /* delivered */
      const struct route_op *op = &x->s.op[q->op];
      x->latency[p] += (e.t - x->release[p]) / op->floor / __builtin_popcount(q->dest);
      q->delivered[e.node]++; S &= ~(1u << e.node);
      if (--x->opleft[q->op] == 0) x->opend[q->op] = e.t;
      if (q->phase2 >= 0) {   /* a partial at its owner: the reduced page once all n - 1 are in */
        const int i = p - q->base, n = f->ncompute, red = q->phase2;
        int ko = 0; while (f->compute[ko] != (int)e.node) ko++;
        if (++x->partial[((size_t)q->op * ROUTE_N + ko) * ROUTE_PAGE_PIECES + i] == n - 1) release(x, red, i, e.t);
      }
    }
    if (S) split_send(x, e.t, (int)e.node, S, e.visited, p);
  }
}

void route_sim_begin(struct route_sim *x, const struct route_stream *s, const struct route_window *w, int mode, uint32_t key0, uint32_t key) {
  x->s = *s; x->win = *w; x->mode = mode; x->key0 = key0; x->key = key;
  route_topology_make(&s->f, s, &x->t);
  memset(x->ctx, 0xff, sizeof x->ctx);
  int pj[ROUTE_CONTEXTS], pv[ROUTE_CONTEXTS]; uint32_t pS[ROUTE_CONTEXTS];
  x->nc = route_contexts(s, &x->t, pj, pv, pS);
  if (x->nc < 0) abort();
  for (int c = 0; c < x->nc; c++) { x->cj[c] = pj[c]; x->cv[c] = pv[c]; x->cS[c] = pS[c]; x->ctx[pj[c]][pv[c]][pS[c]] = c; }
  for (uint32_t g = 0; g < x->ngroups; g++) x->groups[g].count = 0;
  x->ngroups = 0; memset(x->hash, 0, x->hashroom * sizeof(uint32_t));
  x->events = 0; x->nflows = 0; x->npieces = 0; x->point = -1; x->active = -1; memset(x->count, 0, sizeof x->count);
  x->tend = w->points > 0 ? w->t0 + w->points * w->dt : INFINITY; x->stopped = w->points > 0 && mode == 0;
  for (int l = 0; l < ROUTE_L; l++) for (int c = 0; c < ROUTE_CABLES; c++) x->free_at[l][c] = l < s->f.lanes ? s->background[l] / s->f.rate[l] : 0;
  for (int o = 0; o < s->ops; o++) { x->opstarted[o] = 0; x->opend[o] = INFINITY; push(x, (struct event){s->op[o].arrival, EV_OP, o, 0, 0, 0}); }
}

int route_sim_decide(struct route_sim *x, struct route_decision *d, float *tokens) {
  const int k = x->point + 1;
  const double tau = x->win.t0 + k * x->win.dt;
  if (x->mode != 0 || (x->win.points > 0 && k >= x->win.points)) { advance(x, INFINITY); return 0; }
  advance(x, tau);
  if (x->win.points == 0 && k > 0 && !x->events) return 0;
  x->point = k;
  const struct route_fabric *f = &x->s.f;
  struct route_state *st = &x->state[k % ROUTE_POINTS];   /* the fabric then */
  memset(st, 0, sizeof *st);
  st->tau = tau;
  for (int l = 0; l < f->lanes; l++) {
    double w = INFINITY; int busy = 0;
    for (int c = 0; c < f->cables[l]; c++) { const double q = x->free_at[l][c] - tau > 0 ? x->free_at[l][c] - tau : 0; if (q < w) w = q; }
    const int rl = x->t.rev[l];
    if (rl >= 0) for (int c = 0; c < f->cables[rl]; c++) busy = busy || x->free_at[rl][c] > tau;
    st->wait[l] = w; st->dup[l] = busy ? f->duplex : 1;
  }
  for (size_t e = 0; e < x->events; e++) if (x->heap[e].type == EV_ARRIVE) { st->inflight[x->heap[e].node] += x->bytes[x->heap[e].a]; st->posted += x->bytes[x->heap[e].a]; }
  for (int a = 0; a < ROUTE_N; a++) for (int c = 0; c < ROUTE_N; c++) st->togo[a][c] = a == c ? 0 : INFINITY;
  for (int l = 0; l < f->lanes; l++) { const double h = route_hop(&x->s, st, l); if (h < st->togo[f->from[l]][f->to[l]]) st->togo[f->from[l]][f->to[l]] = h; }
  for (int m = 0; m < ROUTE_N; m++) for (int a = 0; a < ROUTE_N; a++) for (int c = 0; c < ROUTE_N; c++) if (st->togo[a][m] + st->togo[m][c] < st->togo[a][c]) st->togo[a][c] = st->togo[a][m] + st->togo[m][c];
  double total[ROUTE_J] = {0}, unposted[ROUTE_J] = {0}, got[ROUTE_J][ROUTE_N] = {{0}};
  for (int fl = 0; fl < x->nflows; fl++) {   /* each pattern's live flows: what is left to post, waiting, what each destination has */
    const struct flow *q = &x->flows[fl];
    if (!isinf(x->opend[q->op])) continue;
    total[q->pattern] += q->np; unposted[q->pattern] += q->np - q->head;
    for (int r = q->head; r < q->tail; r++) st->pending[q->pattern] += x->bytes[q->base + x->ready[q->base + r]];
    for (int r = 0; r < q->np; r++) st->demand[q->pattern] += x->bytes[q->base + r];
    for (int dd = 0; dd < ROUTE_N; dd++) got[q->pattern][dd] += q->delivered[dd];
  }
  for (int o = 0; o < x->s.ops; o++) if (x->opstarted[o] && isinf(x->opend[o])) st->ops += 1;
  for (int j = 0; j < x->s.patterns; j++) {
    st->left[j] = total[j] > 0 ? unposted[j] / total[j] : 0;
    for (int dd = 0; dd < ROUTE_N; dd++) st->recv[j][dd] = total[j] > 0 ? got[j][dd] / total[j] : 0;
  }
  d->point = (unsigned)k; d->lanes = (unsigned)f->lanes; d->contexts = (unsigned)x->nc;
  for (int l = 0; l < ROUTE_L; l++) route_lane_token(&x->s, st, &x->t, l, d->lane[l]);
  for (int c = 0; c < x->nc; c++) route_context_token(&x->s, st, &x->t, x->cj[c], x->cv[c], x->cS[c], k, tokens + (size_t)c * ROUTE_IN);
  return 1;
}
void route_sim_answer(struct route_sim *x, const float *w) {
  const int k = x->point;
  if (w) memcpy(route_sim_w(x, k), w, (size_t)x->nc * ROUTE_FS * sizeof(float));
  if (k == 0) x->active = 0;
  else push(x, (struct event){x->state[k % ROUTE_POINTS].tau + x->win.lag, EV_ACTIVATE, k, 0, 0, 0});
}
void route_sim_costs(const struct route_sim *x, struct route_costs *c) {
  memset(c, 0, sizeof *c);
  const double t0 = x->mode == 0 ? x->win.t0 : 0, t1 = x->mode == 0 ? x->tend : INFINITY;
  for (int o = 0; o < x->s.ops; o++)   /* the collectives live in the window */
    if (x->opstarted[o] && isfinite(x->opend[o]) && x->s.op[o].arrival < t1 && x->opend[o] >= t0) { c->throughput += (x->opend[o] - x->s.op[o].arrival) / x->s.op[o].floor; c->ops++; }
  for (int p = 0; p < x->npieces; p++)
    if (x->postpoint[p] >= (x->mode == 0 ? 0 : -1)) { c->latency += x->latency[p]; c->queueing += x->queued[p] / x->s.op[x->flows[x->flowof[p]].op].floor; c->pieces++; }
  if (c->ops) c->throughput /= c->ops;
  if (c->pieces) { c->latency /= c->pieces; c->queueing /= c->pieces; }
}

/* refresh `point`'s credit: for each of its groups, E psi over the masked table once, then each draw's advantage times
   (psi - E psi), psi = CLIP (1 - tanh^2) phi / sqrt(FS). Advantages: throughput 1 (the caller scales); latency and
   queueing the piece's against the mean of the other pieces of its flow posted under the same refresh, over the pieces */
void route_sim_credit(struct route_sim *x, int point, float *dw, double *loglik) {
  const int nc = x->nc;
  memset(dw, 0, (size_t)ROUTE_TASKS * nc * ROUTE_FS * sizeof(float));
  *loglik = 0;
  int n = 0; for (int p = 0; p < x->npieces; p++) n += x->postpoint[p] >= 0;
  if (!n) n = 1;
  const struct route_state *st = &x->state[point % ROUTE_POINTS];
  const float *wp = route_sim_w(x, point);
  double *lg = malloc(3125 * sizeof(double));
  if (!lg) abort();
  for (uint32_t g = 0; g < x->ngroups; g++) {
    const struct group *q = &x->groups[g];
    if (q->point != (uint32_t)point || !q->count) continue;
    const int c = (int)q->context, v = x->cv[c], deg = x->t.deg[v], ntok = route_tokens(&x->t, v, x->cS[c]); const uint32_t S = x->cS[c];
    const float *w = wp + (size_t)c * ROUTE_FS;
    for (uint32_t m = 0; m < q->count; m++) {   /* each draw: its own table (its lanes then), masked as it was */
      const uint32_t tok = q->draws[m].token; const int p = (int)q->draws[m].piece; const struct route_live *live = &q->draws[m].live;
      double most = -INFINITY;
      for (int k = 0; k < ntok; k++) {
        float phi[ROUTE_FS]; double a = 0;
        const int ok = route_split_features(&x->s, st, &x->t, v, S, (uint32_t)k, live, phi) && fits((uint32_t)k, S, q->mask, deg);
        for (int i = 0; i < ROUTE_FS; i++) a += (double)w[i] * phi[i];
        lg[k] = ok ? ROUTE_CLIP * tanh(a / sqrt((double)ROUTE_FS)) : -INFINITY;
        if (lg[k] > most) most = lg[k];
      }
      double sum = 0, epsi[ROUTE_FS] = {0}, psi[ROUTE_FS];
      for (int k = 0; k < ntok; k++) if (isfinite(lg[k])) sum += exp(lg[k] - most);
      const double lse = most + log(sum);
      for (int k = 0; k < ntok; k++) {
        if (!isfinite(lg[k]) && k != (int)tok) continue;
        float phi[ROUTE_FS]; double a = 0;
        route_split_features(&x->s, st, &x->t, v, S, (uint32_t)k, live, phi);
        for (int i = 0; i < ROUTE_FS; i++) a += (double)w[i] * phi[i];
        const double th = tanh(a / sqrt((double)ROUTE_FS)), d = ROUTE_CLIP * (1 - th * th) / sqrt((double)ROUTE_FS);
        if (isfinite(lg[k])) for (int i = 0; i < ROUTE_FS; i++) epsi[i] += exp(lg[k] - lse) * d * phi[i];
        if (k == (int)tok) for (int i = 0; i < ROUTE_FS; i++) psi[i] = d * phi[i];
      }
      const struct flow *fl = &x->flows[x->flowof[p]];
      double A[ROUTE_TASKS] = {1, 0, 0}, sl = 0, sq = 0; int cnt = 0;
      for (int r = fl->base; r < fl->base + fl->np; r++) if (x->postpoint[r] == x->postpoint[p]) { sl += x->latency[r]; sq += x->queued[r]; cnt++; }
      if (cnt > 1) {
        const double fl0 = x->s.op[fl->op].floor;
        A[1] = (x->latency[p] - (sl - x->latency[p]) / (cnt - 1)) / n;
        A[2] = (x->queued[p] - (sq - x->queued[p]) / (cnt - 1)) / fl0 / n;
      }
      for (int t = 0; t < ROUTE_TASKS; t++) for (int k = 0; k < ROUTE_FS; k++) dw[((size_t)t * nc + c) * ROUTE_FS + k] += (float)(A[t] * (psi[k] - epsi[k]));
      *loglik += lg[tok] - lse;
    }
  }
  free(lg);
}

/* The environment (route.h): one transaction's run on a fabric, the max-plus FIFO recursion over discrete pieces with
   tests/switch.c's semantics. Every lane is its cables, a pair's rate shared equally among them; a piece crossing a lane
   takes whichever cable frees first: start = max(arrival, cable free), at the far node start + bytes / cable rate +
   latency, the cable running at the duplex share of its rate while the reverse direction's same cable is busy then; a
   node forwards a piece the moment it lands. Events leave in time order through switch.c's heap.

   Each commodity's pieces are posted in groups of ROUTE_GROUP: group b at tau_b, when every commodity's group b-1 has
   started on the wire at its source (mode 4: all at 0). At each decision point the environment states the fabric
   (route_state), writes the tokens the policy reads, and takes w for every context back. A piece landing at node v and
   still owing S draws its split from context (j, v, S)'s distribution, its table built from w and the splits' features
   the first time a piece needs it at that decision point, masked to the lanes that leave the piece's path and lead to
   their share without coming back (by rejection, which leaves the masked distribution exact). Every draw is logged under
   its (decision point, context, mask) group, for the credit. */
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

/* ---- fabrics and scenarios ---- */
int route_fabric_load(const char *path, struct route_fabric *f) {   /* node NAME compute|relay; link FROM TO GBps us cables; duplex SHARE */
  FILE *in = fopen(path, "r");
  if (!in) return -1;
  char line[256], names[ROUTE_N][16]; memset(f, 0, sizeof *f); f->duplex = 1;
  while (fgets(line, sizeof line, in)) {
    char a[16], b[16]; double r, us; int c;
    if (line[0] == '#') continue;
    if (sscanf(line, "duplex %lf", &r) == 1) f->duplex = r;
    else if (sscanf(line, "node %15s %15s", a, b) == 2) { if (f->nodes == ROUTE_N) return -1; strcpy(names[f->nodes], a); if (!strcmp(b, "compute")) f->compute[f->ncompute++] = f->nodes; f->nodes++; }
    else if (sscanf(line, "link %15s %15s %lf %lf %d", a, b, &r, &us, &c) == 5) {
      int u = -1, v = -1;
      for (int k = 0; k < f->nodes; k++) { if (!strcmp(names[k], a)) u = k; if (!strcmp(names[k], b)) v = k; }
      if (u < 0 || v < 0 || f->lanes == ROUTE_L) return -1;
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
void route_scenario_make(const struct route_fabric *f, double S, struct route_rng *g, int background, struct route_scenario *x) {
  memset(x, 0, sizeof *x);
  x->f = *f; x->size = S;
  const int n = f->nodes, nl = f->lanes; const double q = S < ROUTE_PAGE ? S : ROUTE_PAGE;
  for (int l = 0; l < nl; l++) x->background[l] = background && route_uniform(g) < 0.3 ? f->rate[l] * 50e3 * route_uniform(g) : 0;
  double cut = 0, w[ROUTE_L], dist[ROUTE_N][ROUTE_N];
  for (int k = 0; k < f->ncompute; k++) {
    const int v = f->compute[k]; double inr = 0, outr = 0;
    for (int l = 0; l < nl; l++) { if (f->to[l] == v) inr += f->rate[l]; if (f->from[l] == v) outr += f->rate[l]; }
    const double a = (f->ncompute - 1) * S / inr, b = S / outr;
    if (a > cut) cut = a;
    if (b > cut) cut = b;
  }
  for (int l = 0; l < nl; l++) w[l] = f->latency[l] + q / (f->rate[l] / f->cables[l]);
  for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) dist[a][b] = a == b ? 0 : INFINITY;
  for (int l = 0; l < nl; l++) if (w[l] < dist[f->from[l]][f->to[l]]) dist[f->from[l]][f->to[l]] = w[l];
  for (int m = 0; m < n; m++) for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) if (dist[a][m] + dist[m][b] < dist[a][b]) dist[a][b] = dist[a][m] + dist[m][b];
  double reach = 0;
  for (int a = 0; a < f->ncompute; a++) for (int b = 0; b < f->ncompute; b++) if (a != b && dist[f->compute[a]][f->compute[b]] > reach) reach = dist[f->compute[a]][f->compute[b]];
  x->cut = cut; x->reach = reach; x->floor = cut > reach ? cut : reach;
  x->commodities = f->ncompute;
  for (int j = 0; j < f->ncompute; j++) {   /* the unloaded shortest-path tree, pruned to what leads to a destination */
    const int s = f->compute[j];
    double at[ROUTE_N]; int by[ROUTE_N], done[ROUTE_N] = {0}, keep[ROUTE_N] = {0};
    for (int v = 0; v < n; v++) { at[v] = INFINITY; by[v] = -1; }
    at[s] = 0;
    for (int it = 0; it < n; it++) {
      int u = -1;
      for (int v = 0; v < n; v++) if (!done[v] && isfinite(at[v]) && (u < 0 || at[v] < at[u])) u = v;
      if (u < 0) break;
      done[u] = 1;
      for (int l = 0; l < nl; l++) if (f->from[l] == u && at[u] + w[l] < at[f->to[l]]) { at[f->to[l]] = at[u] + w[l]; by[f->to[l]] = l; }
    }
    x->source[j] = s; x->dest[j] = 0;
    for (int k = 0; k < f->ncompute; k++) { const int d = f->compute[k]; if (d == s) continue; x->dest[j] |= 1u << d; for (int v = d; v != s; v = f->from[by[v]]) keep[v] = 1; }
    for (int v = 0; v < ROUTE_N; v++) x->spt[j][v] = v < n && keep[v] ? by[v] : -1;
  }
}

/* ---- one transaction ---- */
struct event { double t; uint32_t piece, node, owed, visited; };
struct draw { uint32_t token, piece; };
struct group { uint32_t point, context, mask, count, room; struct draw *draws; };
struct table { int point, room; double *cdf; };   /* a context's distribution at decision point `point`, as a cumulative one */
struct route_sim {
  struct route_scenario s; struct route_topology t;
  int mode, np, nb, point, nc, posted;
  uint32_t key;
  int ctx[ROUTE_J][ROUTE_N][ROUTE_SETS], cj[ROUTE_CONTEXTS], cv[ROUTE_CONTEXTS], tokens[ROUTE_CONTEXTS];
  uint32_t cS[ROUTE_CONTEXTS];
  double free_at[ROUTE_L][ROUTE_CABLES], laststart[ROUTE_J], tau[ROUTE_POINTS], T, *latency, *queued;
  uint32_t recv[ROUTE_J][ROUTE_N];
  struct event *heap; size_t events, room;
  struct route_state state[ROUTE_POINTS];
  float *w;                     /* [POINTS][CONTEXTS][FS] */
  struct table tables[ROUTE_CONTEXTS];
  struct group *groups; uint32_t ngroups, groomroom, *hash; uint32_t hashroom;
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
  x->latency = calloc(ROUTE_J * ROUTE_PIECES, sizeof(double)); x->queued = calloc(ROUTE_J * ROUTE_PIECES, sizeof(double));
  x->w = calloc((size_t)ROUTE_POINTS * ROUTE_CONTEXTS * ROUTE_FS, sizeof(float));
  x->hashroom = 1u << 14; x->hash = calloc(x->hashroom, sizeof(uint32_t));
  if (!x->latency || !x->queued || !x->w || !x->hash) abort();
  return x;
}
void route_sim_free(struct route_sim *x) {
  for (uint32_t g = 0; g < x->groomroom; g++) free(x->groups[g].draws);
  for (int c = 0; c < ROUTE_CONTEXTS; c++) free(x->tables[c].cdf);
  free(x->groups); free(x->hash); free(x->heap); free(x->w); free(x->latency); free(x->queued); free(x);
}
int route_sim_points(const struct route_sim *x) { return x->nb; }

void route_sim_begin(struct route_sim *x, const struct route_scenario *sc, int mode, uint32_t key) {
  x->s = *sc; x->mode = mode; x->key = key;
  const struct route_fabric *f = &x->s.f;
  memset(&x->t, 0, sizeof x->t);
  for (int v = 0; v < ROUTE_N; v++) for (int l = 0; l < f->lanes; l++) if (f->from[l] == v) { x->t.adj[v] |= 1u << f->to[l]; if (x->t.deg[v] < ROUTE_DEG) x->t.out[v][x->t.deg[v]++] = l; }
  for (int l = 0; l < ROUTE_L; l++) { x->t.rev[l] = -1; for (int m = 0; m < f->lanes && l < f->lanes; m++) if (f->from[m] == f->to[l] && f->to[m] == f->from[l]) x->t.rev[l] = m; }
  for (int j = 0; j < x->s.commodities; j++) x->t.srcmask |= 1u << x->s.source[j];
  x->np = (int)ceil(x->s.size / ROUTE_PAGE); if (x->np < 1) x->np = 1;
  x->nb = (x->np + ROUTE_GROUP - 1) / ROUTE_GROUP;
  x->point = -1; x->posted = 0; x->T = 0; x->events = 0;
  for (int l = 0; l < ROUTE_L; l++) for (int c = 0; c < ROUTE_CABLES; c++) x->free_at[l][c] = l < f->lanes ? x->s.background[l] / f->rate[l] : 0;
  memset(x->laststart, 0, sizeof x->laststart); memset(x->recv, 0, sizeof x->recv);
  memset(x->latency, 0, ROUTE_J * ROUTE_PIECES * sizeof(double)); memset(x->queued, 0, ROUTE_J * ROUTE_PIECES * sizeof(double));
  memset(x->ctx, 0xff, sizeof x->ctx);
  x->nc = 0;   /* the contexts: every (commodity, node, owed set) */
  for (int j = 0; j < x->s.commodities; j++) for (int v = 0; v < f->nodes; v++) for (uint32_t S = 1; S < ROUTE_SETS; S++) {
    if ((S & ~x->s.dest[j]) || (S >> v & 1)) continue;
    int n = 1; for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) n *= x->t.deg[v];
    x->ctx[j][v][S] = x->nc; x->cj[x->nc] = j; x->cv[x->nc] = v; x->cS[x->nc] = S; x->tokens[x->nc] = n; x->nc++;
  }
  for (int c = 0; c < ROUTE_CONTEXTS; c++) x->tables[c].point = -1;
  for (uint32_t g = 0; g < x->ngroups; g++) x->groups[g].count = 0;
  x->ngroups = 0; memset(x->hash, 0, x->hashroom * sizeof(uint32_t));
}

/* context c's distribution at the current decision point: every split's logit from w and its features, as a cumulative
   distribution (statically invalid splits at probability 0) */
static const double *table(struct route_sim *x, int c) {
  struct table *tb = &x->tables[c];
  if (tb->point == x->point) return tb->cdf;
  if (tb->room < x->tokens[c]) { free(tb->cdf); tb->room = x->tokens[c]; tb->cdf = malloc((size_t)tb->room * sizeof(double)); if (!tb->cdf) abort(); }
  const float *w = x->w + ((size_t)x->point * ROUTE_CONTEXTS + c) * ROUTE_FS;
  const struct route_state *st = &x->state[x->point];
  double most = -INFINITY;
  for (int k = 0; k < x->tokens[c]; k++) {
    float phi[ROUTE_FS]; double a = 0;
    const int valid = route_split_features(&x->s, st, &x->t, x->cv[c], x->cS[c], (uint32_t)k, phi);
    for (int i = 0; i < ROUTE_FS; i++) a += (double)w[i] * phi[i];
    tb->cdf[k] = valid ? ROUTE_CLIP * tanh(a / sqrt((double)ROUTE_FS)) : -INFINITY;
    if (tb->cdf[k] > most) most = tb->cdf[k];
  }
  double sum = 0;
  for (int k = 0; k < x->tokens[c]; k++) sum += isfinite(tb->cdf[k]) ? exp(tb->cdf[k] - most) : 0;
  double acc = 0;
  for (int k = 0; k < x->tokens[c]; k++) { acc += isfinite(tb->cdf[k]) ? exp(tb->cdf[k] - most) / sum : 0; tb->cdf[k] = acc; }
  tb->cdf[x->tokens[c] - 1] = 1;
  tb->point = x->point;
  return tb->cdf;
}
static void log_draw(struct route_sim *x, uint32_t c, uint32_t mask, uint32_t token, uint32_t piece) {
  const uint64_t key = (uint64_t)x->point << 40 | (uint64_t)c << 30 | mask;
  if (2 * (x->ngroups + 1) > x->hashroom) {   /* grow the hash */
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
      if (g == x->groomroom) { x->groomroom = x->groomroom ? 2 * x->groomroom : 256; x->groups = realloc(x->groups, x->groomroom * sizeof *x->groups); if (!x->groups) abort(); for (uint32_t k = g; k < x->groomroom; k++) { x->groups[k].draws = NULL; x->groups[k].room = 0; } }
      x->groups[g].point = (uint32_t)x->point; x->groups[g].context = c; x->groups[g].mask = mask; x->groups[g].count = 0;
      x->hash[h] = g + 1;
      break;
    }
    const struct group *q = &x->groups[e - 1];
    if (q->point == (uint32_t)x->point && q->context == c && q->mask == mask) { g = e - 1; break; }
    h = (h + 1) & (x->hashroom - 1);
  }
  struct group *q = &x->groups[g];
  if (q->count == q->room) { q->room = q->room ? 2 * q->room : 16; q->draws = realloc(q->draws, q->room * sizeof *q->draws); if (!q->draws) abort(); }
  q->draws[q->count++] = (struct draw){token, piece};
}

/* each destination's allowed lanes at v for a piece whose path has been through `visited`: not back into the path, and
   leading to it without coming back */
static uint32_t allowed(const struct route_sim *x, int v, uint32_t S, uint32_t visited) {
  const uint32_t avoid = visited | 1u << v; uint32_t mask = 0;
  for (int d = 0; d < ROUTE_N; d++) {
    if (!(S >> d & 1)) continue;
    for (int k = 0; k < x->t.deg[v]; k++) {
      const int h = x->s.f.to[x->t.out[v][k]];
      if (!(avoid >> h & 1) && (h == d || route_reaches(&x->t, h, avoid, 1u << d))) mask |= 1u << (ROUTE_DEG * d + k);
    }
  }
  return mask;
}
static int fits(uint32_t tok, uint32_t S, uint32_t mask, int deg) {
  for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) { if (!(mask >> (ROUTE_DEG * d + (int)(tok % (uint32_t)deg)) & 1)) return 0; tok /= (uint32_t)deg; }
  return 1;
}
static double unit(uint32_t h) { return (double)(mix(h) >> 8) / 16777216.0; }

/* piece `piece` of commodity j at node v at time t, owing S, its path through `visited`: its split, drawn and sent */
static void split_send(struct route_sim *x, double t, int j, int v, uint32_t S, uint32_t visited, uint32_t piece) {
  const struct route_fabric *f = &x->s.f;
  const int deg = x->t.deg[v];
  const uint32_t mask = allowed(x, v, S, visited);
  uint32_t tok = 0, place = 1;
  if (x->mode == 2 || x->mode == 4) {   /* the shortest-path tree's split */
    for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) {
      int y = d; while (f->from[x->s.spt[j][y]] != v) y = f->from[x->s.spt[j][y]];
      int k = 0; while (x->t.out[v][k] != x->s.spt[j][y]) k++;
      tok += (uint32_t)k * place; place *= (uint32_t)deg;
    }
  } else if (x->mode == 3) {   /* each destination a lane uniformly among the allowed */
    for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) {
      const int m = __builtin_popcount(mask >> (ROUTE_DEG * d) & 31);
      int target = (int)(unit(x->key ^ piece * 0x9e3779b1u ^ (uint32_t)(v * 8 + d) * 0x85ebca6bu) * m), k = 0;
      if (target >= m) target = m - 1;
      for (; k < deg; k++) if (mask >> (ROUTE_DEG * d + k) & 1) { if (!target) break; target--; }
      tok += (uint32_t)k * place; place *= (uint32_t)deg;
    }
  } else {   /* a draw from the context's table, masked by rejection (then by enumeration) */
    const int c = x->ctx[j][v][S], n = x->tokens[c];
    const double *cdf = table(x, c);
    int ok = 0;
    for (uint32_t attempt = 0; attempt < 64 && !ok; attempt++) {
      const double u = unit(x->key ^ piece * 0x9e3779b1u ^ (uint32_t)(v * 64 + (int)attempt) * 0x85ebca6bu ^ S * 0xc2b2ae35u);
      int lo = 0, hi = n - 1;
      while (lo < hi) { const int mid = (lo + hi) / 2; if (cdf[mid] > u) hi = mid; else lo = mid + 1; }
      tok = (uint32_t)lo; ok = fits(tok, S, mask, deg);
    }
    if (!ok) {
      double total = 0;
      for (int q = 0; q < n; q++) if (fits((uint32_t)q, S, mask, deg)) total += cdf[q] - (q ? cdf[q - 1] : 0);
      const double u = unit(x->key ^ piece * 0x27d4eb2du ^ S) * total; double acc = 0;
      for (int q = 0; q < n; q++) if (fits((uint32_t)q, S, mask, deg)) { tok = (uint32_t)q; acc += cdf[q] - (q ? cdf[q - 1] : 0); if (acc > u) break; }
    }
    log_draw(x, (uint32_t)c, mask, tok, piece);
  }
  uint32_t share[ROUTE_DEG] = {0}, y = tok;   /* the sends: one a lane given a share */
  for (int d = 0; d < ROUTE_N; d++) if (S >> d & 1) { share[y % (uint32_t)deg] |= 1u << d; y /= (uint32_t)deg; }
  const int i = (int)(piece % ROUTE_PIECES);
  const double bytes = i + 1 < x->np ? ROUTE_PAGE : x->s.size - ROUTE_PAGE * (x->np - 1);
  for (int k = 0; k < deg; k++) {
    if (!share[k]) continue;
    const int l = x->t.out[v][k], h = f->to[l], rl = x->t.rev[l];
    int cb = 0; for (int q = 1; q < f->cables[l]; q++) if (x->free_at[l][q] < x->free_at[l][cb]) cb = q;
    const double start = x->free_at[l][cb] > t ? x->free_at[l][cb] : t;
    const int both = rl >= 0 && cb < f->cables[rl] && x->free_at[rl][cb] > start;
    const double time = bytes / (f->rate[l] / f->cables[l] * (both ? f->duplex : 1));
    x->free_at[l][cb] = start + time; x->queued[piece] += start - t;
    if (v == x->s.source[j] && start > x->laststart[j]) x->laststart[j] = start;
    push(x, (struct event){start + time + f->latency[l], piece, (uint32_t)h, share[k], visited | 1u << v});
  }
}
static void advance(struct route_sim *x, double until) {   /* every event up to `until`, in time order */
  while (x->events && x->heap[0].t <= until) {
    const struct event e = pop(x);
    const int j = (int)(e.piece / ROUTE_PIECES), i = (int)(e.piece % ROUTE_PIECES);
    uint32_t S = e.owed;
    if (S >> e.node & 1) {
      if (e.t > x->T) x->T = e.t;
      x->latency[e.piece] += (e.t - x->tau[i / ROUTE_GROUP]) / x->s.floor / __builtin_popcount(x->s.dest[j]);
      x->recv[j][e.node]++; S &= ~(1u << e.node);
    }
    if (S) split_send(x, e.t, j, (int)e.node, S, e.visited, e.piece);
  }
}

int route_sim_decide(struct route_sim *x, struct route_decision *d, float *tokens) {
  if (x->point + 1 >= x->nb) { advance(x, INFINITY); return 0; }
  const int b = ++x->point;
  const struct route_fabric *f = &x->s.f;
  double tau = 0;   /* when every commodity's last group has started on the wire at its source */
  if (b > 0 && x->mode != 4) {
    tau = INFINITY;
    for (int j = 0; j < x->s.commodities; j++) if (x->laststart[j] < tau) tau = x->laststart[j];
    if (tau < x->tau[b - 1]) tau = x->tau[b - 1];
    advance(x, tau);
  }
  x->tau[b] = tau;
  struct route_state *st = &x->state[b];   /* the fabric then */
  memset(st, 0, sizeof *st);
  st->tau = tau;
  for (int l = 0; l < f->lanes; l++) {
    double w = INFINITY; int busy = 0;
    for (int c = 0; c < f->cables[l]; c++) { const double q = x->free_at[l][c] - tau > 0 ? x->free_at[l][c] - tau : 0; if (q < w) w = q; }
    const int rl = x->t.rev[l];
    if (rl >= 0) for (int c = 0; c < f->cables[rl]; c++) busy = busy || x->free_at[rl][c] > tau;
    st->wait[l] = w; st->dup[l] = busy ? f->duplex : 1;
  }
  for (size_t e = 0; e < x->events; e++) { const int i = (int)(x->heap[e].piece % ROUTE_PIECES); st->inflight[x->heap[e].node] += i + 1 < x->np ? ROUTE_PAGE : x->s.size - ROUTE_PAGE * (x->np - 1); }
  for (int a = 0; a < ROUTE_N; a++) for (int c = 0; c < ROUTE_N; c++) st->togo[a][c] = a == c ? 0 : INFINITY;
  for (int l = 0; l < f->lanes; l++) { const double h = route_hop(&x->s, st, l); if (h < st->togo[f->from[l]][f->to[l]]) st->togo[f->from[l]][f->to[l]] = h; }
  for (int m = 0; m < ROUTE_N; m++) for (int a = 0; a < ROUTE_N; a++) for (int c = 0; c < ROUTE_N; c++) if (st->togo[a][m] + st->togo[m][c] < st->togo[a][c]) st->togo[a][c] = st->togo[a][m] + st->togo[m][c];
  double got = 0, want = 0;
  for (int j = 0; j < ROUTE_J; j++) {
    const int left = x->np - b * ROUTE_GROUP; st->left[j] = (double)(left > 0 ? left : 0) / x->np;
    for (int dd = 0; dd < ROUTE_N; dd++) { st->recv[j][dd] = (double)x->recv[j][dd] / x->np; if (j < x->s.commodities && (x->s.dest[j] >> dd & 1)) { got += st->recv[j][dd]; want += 1; } }
  }
  st->progress = want > 0 ? got / want : 0;
  d->point = (unsigned)b; d->lanes = (unsigned)f->lanes; d->contexts = (unsigned)x->nc;
  for (int l = 0; l < ROUTE_L; l++) route_lane_token(&x->s, st, &x->t, l, d->lane[l]);
  for (int c = 0; c < x->nc; c++) route_context_token(&x->s, st, &x->t, x->cj[c], x->cv[c], x->cS[c], b, tokens + (size_t)c * ROUTE_IN);
  return 1;
}
void route_sim_answer(struct route_sim *x, const float *w) {
  const int b = x->point;
  if (w) memcpy(x->w + (size_t)b * ROUTE_CONTEXTS * ROUTE_FS, w, (size_t)x->nc * ROUTE_FS * sizeof(float));
  for (int j = 0; j < x->s.commodities; j++)
    for (int i = b * ROUTE_GROUP; i < x->np && i < (b + 1) * ROUTE_GROUP; i++) split_send(x, x->tau[b], j, x->s.source[j], x->s.dest[j], 0, (uint32_t)(j * ROUTE_PIECES + i));
  if (b + 1 == x->nb) advance(x, INFINITY);
}
void route_sim_costs(const struct route_sim *x, struct route_costs *c) {
  double lat = 0, q = 0; const int n = x->s.commodities * x->np;
  for (int j = 0; j < x->s.commodities; j++) for (int i = 0; i < x->np; i++) { lat += x->latency[j * ROUTE_PIECES + i]; q += x->queued[j * ROUTE_PIECES + i]; }
  c->throughput = x->T / x->s.floor; c->latency = lat / n; c->queueing = q / n / x->s.floor;
}

/* decision point `point`'s credit: for each of its groups, E psi over the masked table once, then each draw's
   advantage times (psi - E psi), psi = CLIP (1 - tanh^2) phi / sqrt(FS). Advantages: throughput 1 (the caller scales);
   latency and queueing the piece's against the mean of the other pieces of its commodity's group, over the pieces */
void route_sim_credit(struct route_sim *x, int point, float *dw, double *loglik) {
  const int nc = x->nc, n = x->s.commodities * x->np;
  memset(dw, 0, (size_t)ROUTE_TASKS * nc * ROUTE_FS * sizeof(float));
  *loglik = 0;
  const struct route_state *st = &x->state[point];
  for (uint32_t g = 0; g < x->ngroups; g++) {
    const struct group *q = &x->groups[g];
    if (q->point != (uint32_t)point || !q->count) continue;
    const int c = (int)q->context, v = x->cv[c], deg = x->t.deg[v], ntok = x->tokens[c]; const uint32_t S = x->cS[c];
    const float *w = x->w + ((size_t)point * ROUTE_CONTEXTS + c) * ROUTE_FS;
    double *lg = malloc((size_t)ntok * sizeof(double)), most = -INFINITY;
    if (!lg) abort();
    for (int k = 0; k < ntok; k++) {
      float phi[ROUTE_FS]; double a = 0;
      const int ok = route_split_features(&x->s, st, &x->t, v, S, (uint32_t)k, phi) && fits((uint32_t)k, S, q->mask, deg);
      for (int i = 0; i < ROUTE_FS; i++) a += (double)w[i] * phi[i];
      lg[k] = ok ? ROUTE_CLIP * tanh(a / sqrt((double)ROUTE_FS)) : -INFINITY;
      if (lg[k] > most) most = lg[k];
    }
    double sum = 0, epsi[ROUTE_FS] = {0};
    for (int k = 0; k < ntok; k++) if (isfinite(lg[k])) sum += exp(lg[k] - most);
    const double lse = most + log(sum);
    for (int k = 0; k < ntok; k++) {
      if (!isfinite(lg[k])) continue;
      float phi[ROUTE_FS]; double a = 0;
      route_split_features(&x->s, st, &x->t, v, S, (uint32_t)k, phi);
      for (int i = 0; i < ROUTE_FS; i++) a += (double)w[i] * phi[i];
      const double th = tanh(a / sqrt((double)ROUTE_FS)), s1 = exp(lg[k] - lse) * ROUTE_CLIP * (1 - th * th) / sqrt((double)ROUTE_FS);
      for (int i = 0; i < ROUTE_FS; i++) epsi[i] += s1 * phi[i];
    }
    for (uint32_t m = 0; m < q->count; m++) {
      const uint32_t tok = q->draws[m].token, piece = q->draws[m].piece;
      const int j = (int)(piece / ROUTE_PIECES), i = (int)(piece % ROUTE_PIECES), g0 = i / ROUTE_GROUP * ROUTE_GROUP;
      const int g1 = g0 + ROUTE_GROUP < x->np ? g0 + ROUTE_GROUP : x->np;
      double A[ROUTE_TASKS] = {1, 0, 0};
      if (g1 - g0 > 1) {
        double sl = 0, sq = 0;
        for (int p = g0; p < g1; p++) { sl += x->latency[j * ROUTE_PIECES + p]; sq += x->queued[j * ROUTE_PIECES + p]; }
        A[1] = (x->latency[piece] - (sl - x->latency[piece]) / (g1 - g0 - 1)) / n;
        A[2] = (x->queued[piece] - (sq - x->queued[piece]) / (g1 - g0 - 1)) / x->s.floor / n;
      }
      float phi[ROUTE_FS]; double a = 0;
      route_split_features(&x->s, st, &x->t, v, S, tok, phi);
      for (int k = 0; k < ROUTE_FS; k++) a += (double)w[k] * phi[k];
      const double th = tanh(a / sqrt((double)ROUTE_FS)), s1 = ROUTE_CLIP * (1 - th * th) / sqrt((double)ROUTE_FS);
      for (int t = 0; t < ROUTE_TASKS; t++) for (int k = 0; k < ROUTE_FS; k++) dw[((size_t)t * nc + c) * ROUTE_FS + k] += (float)(A[t] * (s1 * phi[k] - epsi[k]));
      *loglik += lg[tok] - lse;
    }
    free(lg);
  }
}

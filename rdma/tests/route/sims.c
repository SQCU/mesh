/* The simulators as arrays (kernels.h): n slots, each a simulator and its arena (its refreshes' decisions and tokens as
   made, its ended run's draws by refresh), every call a parallel pass over the slots it is given. */
#include "kernels.h"
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

struct drawn { struct route_draw *d; int *ctx, *first, draws, rows, room; };   /* an ended refresh's draws, by context */
struct slot {
  struct route_sim *sim;
  struct route_decision *dec;   /* [K] */
  float *tok;                   /* [K][CONTEXTS][IN] */
  struct drawn *drawn;          /* [K] */
  int made;                     /* refreshes decided */
};
struct launched { struct route_sims *x; int slot; };   /* a slot's task context, held from route_sims_new */
struct route_sims { int n, K; struct slot *s; _Atomic int *ready; int *status, *nc; struct launched *task; };
/* the one wake-up: a run at its refresh or end, a forward or a step done (net.swift's completion handlers) */
static dispatch_semaphore_t wake;
void route_wake(void) { dispatch_semaphore_signal(wake); }

struct route_sims *route_sims_new(int n, int refreshes) {
  struct route_sims *x = calloc(1, sizeof *x);
  x->n = n; x->K = refreshes; x->s = calloc((size_t)n, sizeof *x->s);
  x->ready = calloc((size_t)n, sizeof *x->ready); x->status = calloc((size_t)n, sizeof(int)); x->nc = calloc((size_t)n, sizeof(int));
  x->task = calloc((size_t)n, sizeof *x->task);
  for (int i = 0; i < n; i++) { x->task[i].x = x; x->task[i].slot = i; }
  if (!wake) wake = dispatch_semaphore_create(0);
  for (int i = 0; i < n; i++) {
    struct slot *s = &x->s[i];
    s->sim = route_sim_new();
    s->dec = calloc((size_t)refreshes, sizeof *s->dec);
    s->tok = calloc((size_t)refreshes * ROUTE_CONTEXTS * ROUTE_IN, sizeof(float));
    s->drawn = calloc((size_t)refreshes, sizeof *s->drawn);
    if (!s->sim || !s->dec || !s->tok || !s->drawn) abort();
  }
  return x;
}
void route_sims_free(struct route_sims *x) {
  for (int i = 0; i < x->n; i++) {
    struct slot *s = &x->s[i];
    for (int k = 0; k < x->K; k++) { free(s->drawn[k].d); free(s->drawn[k].ctx); free(s->drawn[k].first); }
    route_sim_free(s->sim); free(s->dec); free(s->tok); free(s->drawn);
  }
  free(x->s); free(x->ready); free(x->status); free(x->nc); free(x->task); free(x);
}
struct route_sim *route_sims_sim(struct route_sims *x, int slot) { return x->s[slot].sim; }

void route_sims_begin(struct route_sims *x, const int *slot, int m, const struct route_case *c, const int *which, const int *rollout) {
  for (int i = 0; i < m; i++) {
    struct slot *s = &x->s[slot[i]]; const struct route_case *k = &c[which[i]];
    struct route_stream st; route_case_stream(k, &st);
    route_sim_begin(s->sim, &st, &k->w, k->mode, k->key0[rollout[i]], k->key[rollout[i]]);
    s->made = 0;
  }
}

/* one parallel pass: `fn` on each of m slots */
struct pass { struct route_sims *x; const int *slot; const void *a; const void *b; void *c; void *d; void *e; void *f; void *g; int i0, K; const int *row0, *draw0; };
#define PASS(name) static void name(void *p_, size_t i)

/* one run to its next refresh (status 1, nc its contexts) or its end (0: its draws by refresh, by context) */
static void advance(struct route_sims *x, int slot, int *status, int *nc) {
  struct slot *s = &x->s[slot];
  const uint64_t t0 = route_tel_now();
  const int k = s->made % x->K;
  *status = route_sim_decide(s->sim, &s->dec[k], s->tok + (size_t)k * ROUTE_CONTEXTS * ROUTE_IN);
  *nc = *status ? (int)s->dec[k].contexts : 0;
  if (*status) s->made++;
  else for (int r = 0; r < s->made && r < x->K; r++) {   /* ended: each refresh's draws, by context */
    struct drawn *d = &s->drawn[r];
    d->draws = route_sim_draw_count(s->sim, r);
    if (d->draws + 1 > d->room) {
      d->room = 2 * (d->draws + 1);
      d->d = realloc(d->d, d->room * sizeof *d->d); d->ctx = realloc(d->ctx, d->room * sizeof(int)); d->first = realloc(d->first, d->room * sizeof(int));
      if (!d->d || !d->ctx || !d->first) abort();
    }
    d->rows = route_sim_draws(s->sim, r, d->d, d->ctx, d->first);
  }
  route_tel_add(RT_WORKER_NS, route_tel_now() - t0);   /* the runs' own time: against cores x wall, the cores' use */
}
static void run1(void *p_) {
  const struct launched *l = p_; struct route_sims *x = l->x; const int slot = l->slot;
  advance(x, slot, &x->status[slot], &x->nc[slot]);
  atomic_store_explicit(&x->ready[slot], 1, memory_order_release);
  dispatch_semaphore_signal(wake);
}
void route_sims_launch(struct route_sims *x, const int *slot, int m, const float *w, const int *wrow) {
  dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  for (int i = 0; i < m; i++) {
    if (w && wrow[i] >= 0) route_sim_answer(x->s[slot[i]].sim, w + (size_t)wrow[i] * ROUTE_FS);   /* copied now: the w buffer is the caller's again */
    dispatch_async_f(q, &x->task[slot[i]], run1);
  }
}
int route_sims_collect(struct route_sims *x, int wait, int *slot, int *status, int *nc) {
  if (wait) dispatch_semaphore_wait(wake, DISPATCH_TIME_FOREVER);
  int m = 0;
  for (int i = 0; i < x->n; i++)
    if (atomic_exchange_explicit(&x->ready[i], 0, memory_order_acquire)) { slot[m] = i; status[m] = x->status[i]; nc[m] = x->nc[i]; m++; }
  return m;
}

PASS(pack1) {
  struct pass *p = p_; struct slot *s = &p->x->s[p->slot[i]];
  const int *offset = p->a; float *lanes = p->c, *mask = p->d, *tokens = p->e; int *pos = p->f;
  const int k = (s->made - 1) % p->x->K, nc = (int)s->dec[k].contexts, nl = (int)s->dec[k].lanes;
  memcpy(lanes + i * ROUTE_L * ROUTE_IN, s->dec[k].lane, sizeof s->dec[k].lane);
  for (int q = 0; q < ROUTE_L; q++) mask[i * ROUTE_L + q] = q < nl;
  memcpy(tokens + (size_t)offset[i] * ROUTE_IN, s->tok + (size_t)k * ROUTE_CONTEXTS * ROUTE_IN, (size_t)nc * ROUTE_IN * sizeof(float));
  for (int c = 0; c < nc; c++) pos[offset[i] + c] = (int)i * ROUTE_L + ROUTE_L - 1;
}
void route_sims_pack(struct route_sims *x, const int *slot, int m, const int *offset, float *lanes, float *mask, float *tokens, int *pos) {
  struct pass p = {x, slot, offset, 0, lanes, mask, tokens, pos};
  dispatch_apply_f((size_t)m, DISPATCH_APPLY_AUTO, &p, pack1);
}

void route_sims_sizes(struct route_sims *x, const int *slot, int m, int *rows, int *draws) {
  for (int i = 0; i < m; i++) {
    const struct slot *s = &x->s[slot[i]];
    for (int k = 0; k < x->K; k++) { const int kept = k < s->made; rows[i * x->K + k] = kept ? s->drawn[k].rows : 0; draws[i * x->K + k] = kept ? s->drawn[k].draws : 0; }
  }
}
void route_sims_costs(struct route_sims *x, const int *slot, int m, struct route_costs *costs, uint64_t *counts) {
  for (int i = 0; i < m; i++) { route_sim_costs(x->s[slot[i]].sim, &costs[i]); if (counts) route_sim_counts(x->s[slot[i]].sim, counts + 4 * i); }
}

PASS(export1) {
  struct pass *p = p_; struct slot *s = &p->x->s[p->slot[i]];
  const int K = p->K, b = p->i0 + (int)i;
  float *lanes = p->c, *mask = p->d, *tokens = p->e, *w = p->g; int *pos = p->f; struct route_draw *draws = (void *)p->a; uint32_t *first = (void *)p->b;
  for (int k = 0; k < K && k < s->made; k++) {
    const struct route_decision *dec = &s->dec[k]; const struct drawn *d = &s->drawn[k];
    const int r0 = p->row0[i * K + k], d0 = p->draw0[i * K + k];
    memcpy(lanes + (size_t)(b * K + k) * ROUTE_L * ROUTE_IN, dec->lane, sizeof dec->lane);
    for (int q = 0; q < ROUTE_L; q++) mask[(b * K + k) * ROUTE_L + q] = q < (int)dec->lanes;
    const float *wk = route_sim_w(s->sim, k), *tk = s->tok + (size_t)k * ROUTE_CONTEXTS * ROUTE_IN;
    for (int r = 0; r < d->rows; r++) {
      memcpy(tokens + (size_t)(r0 + r) * ROUTE_IN, tk + (size_t)d->ctx[r] * ROUTE_IN, ROUTE_IN * sizeof(float));
      memcpy(w + (size_t)(r0 + r) * ROUTE_FS, wk + (size_t)d->ctx[r] * ROUTE_FS, ROUTE_FS * sizeof(float));
      pos[r0 + r] = (b * K + k) * ROUTE_L + ROUTE_L - 1;
      first[r0 + r] = (uint32_t)(d0 + d->first[r]);
    }
    for (int j = 0; j < d->draws; j++) { draws[d0 + j] = d->d[j]; draws[d0 + j].row += (uint32_t)r0; }
  }
}
void route_sims_export(struct route_sims *x, const int *slot, int m, int b0, int K, const int *row0, const int *draw0, float *lanes, float *mask,
                       float *tokens, int *pos, float *w, struct route_draw *draws, uint32_t *first) {
  struct pass p = {x, slot, draws, first, lanes, mask, tokens, pos, w, b0, K, row0, draw0};
  dispatch_apply_f((size_t)m, DISPATCH_APPLY_AUTO, &p, export1);
}

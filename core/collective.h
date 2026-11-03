/* collective.h: the collective library. A program declares its whole function's calls before it runs; the library
   compiles them into one plan of page writes and hands each node's bridge its page lists. Sources: design/data-model.md
   section 5 (the routing function and its plan; "A relay passes each part on as it arrives"; "It asks no peer to choose
   a place or grant permission between steps"; reuse from the work peers send back), section 6 (page lists and steps);
   the rubric's interface table: between collective and program cross "the NCCL API, the device-function header,
   memory through ncclMemAlloc".

   Header-only: define COLLECTIVE_IMPLEMENTATION in one translation unit (with rdma/mesh-graph.c linked: the routes are
   the graph module's trees, design/rubrics.md:422, C9). */
#ifndef CORE_COLLECTIVE_H
#define CORE_COLLECTIVE_H
#include "bridge.h"
#include <stddef.h>

/* Memory: every buffer a call touches is pages of the region, the same pages at every node for the same program. */
void *collective_alloc(size_t bytes);    /* ncclMemAlloc */
void collective_free(void *memory);      /* ncclMemFree */

/* A declared call (the NCCL API's collectives, declared, never eager). An all-gather's `receive` holds every rank's
   segment in rank order, this rank's own included; an all-reduce sums `receive` over the ranks in place. */
enum { COLLECTIVE_ALLGATHER, COLLECTIVE_ALLREDUCE };
struct collective_call { uint32_t what; void *receive; uint64_t bytes; };

/* The routing function's inputs (data-model.md section 5): the nodes, the cables that join them and their rates, in one
   order every node holds; this node's bridge links are the cables that end at it, in that order. */
struct collective_link { uint32_t a, b; double bytes_a_second; };
struct collective_fabric { uint32_t nodes, rank; const struct collective_link *links; uint32_t count; };

/* What the program does around the bridge's moves, a call occurrence at a time. A publication is one system fence then
   the node's publication count, incremented, stored in every link's supplied word: the program publishes once after
   producing the call's pages, then once after each fold. A fold adds a landed partial page into its own once that
   link's landed reaches `after`. The call is complete here when every link's landed reaches its `await` count. An
   all-reduce's receive buffer holds its bytes then (nodes - 1) scratch copies, where children's partials land. */
struct collective_fold { bridge_page into, from; uint32_t bytes, after_link; uint64_t after; };
struct collective_work {
  struct collective_fold *folds; uint32_t fold_count;
  uint64_t await[BRIDGE_LINKS];          /* a link's landed count at the occurrence's end (0: none) */
};

/* Compile the whole function for `occurrences` runs of its calls into this node's page lists (loaded into its bridge)
   and its work, one collective_work a call occurrence (calls x occurrences, call-major within an occurrence). The same
   function of the same inputs at every node, so both ends of every cable make the same lists. 0, or an errno. */
int collective_plan(const struct collective_fabric *fabric, const struct collective_call *calls, uint32_t count,
                    uint32_t occurrences, struct collective_work **work);
#endif

#ifdef COLLECTIVE_IMPLEMENTATION
#include "../rdma/mesh-graph.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- memory: pages of the region, a bump over the pages with whole-page leases ---- */
static bridge_page collective_next_;
void *collective_alloc(size_t bytes) {
  const bridge_page first = collective_next_;
  collective_next_ += (bridge_page)((bytes + BRIDGE_PAGE - 1) / BRIDGE_PAGE);
  return bridge_address(first);
}
void collective_free(void *memory) { (void)memory; }   /* a program's pages live for the program (its whole function) */

static bridge_page collective_page_of(const void *p) {
  return (bridge_page)(((const char *)p - (const char *)bridge_address(0)) / BRIDGE_PAGE);
}

/* ---- the plan ---- */
struct collective_plan_state {
  const struct collective_fabric *f;
  uint32_t n, trees, *root, *parent; double *share;
  uint32_t *local;                       /* a global cable's local link here, or UINT32_MAX */
  uint64_t *receives;                    /* each global cable's receives so far (its landed index) */
  struct bridge_step *steps; uint64_t step_count, step_room;
};

static int collective_rooted(const void *context, uint32_t r) { (void)context; (void)r; return 1; }

/* the cable between u and v that carries page index p: the pair's cables in fabric order, p modulo their count */
static uint32_t collective_cable(const struct collective_fabric *f, uint32_t u, uint32_t v, uint64_t p) {
  uint32_t between = 0;
  for (uint32_t c = 0; c < f->count; c++) between += (f->links[c].a == u && f->links[c].b == v) || (f->links[c].a == v && f->links[c].b == u);
  uint32_t k = (uint32_t)(p % (between ? between : 1));
  for (uint32_t c = 0; c < f->count; c++)
    if (((f->links[c].a == u && f->links[c].b == v) || (f->links[c].a == v && f->links[c].b == u)) && k-- == 0) return c;
  return UINT32_MAX;
}

static int collective_step(struct collective_plan_state *s, struct bridge_step step) {
  if (s->step_count == s->step_room) {
    s->step_room = s->step_room ? 2 * s->step_room : 4096;
    struct bridge_step *grown = realloc(s->steps, s->step_room * sizeof *grown);
    if (!grown) return ENOMEM;
    s->steps = grown;
  }
  s->steps[s->step_count++] = step;
  return 0;
}

/* one page moved u -> v on its cable: the send at u (after `after_link`'s word reaches `position`), the receive at v.
   Returns the cable's landed index of this page at v (its receive's place in the cable's list, from 1). */
static uint64_t collective_move(struct collective_plan_state *s, uint32_t u, uint32_t v, bridge_page page, bridge_page into,
                                uint32_t bytes, uint64_t p, uint32_t after_link, uint32_t after_landed, uint64_t position,
                                uint32_t *cable_out) {
  const uint32_t c = collective_cable(s->f, u, v, p);
  const uint64_t index = ++s->receives[c];
  if (cable_out) *cable_out = c;
  if (u == s->f->rank)
    collective_step(s, (struct bridge_step){ .link = s->local[c], .receive = 0, .page = page, .bytes = bytes,
                                             .after_link = after_link, .after_landed = after_landed, .position = position });
  if (v == s->f->rank)
    collective_step(s, (struct bridge_step){ .link = s->local[c], .receive = 1, .page = into, .bytes = bytes });
  return index;
}

/* the trees of root r, each with its share; a page index p goes to the tree whose cumulative share holds p's place */
static uint32_t collective_tree(const struct collective_plan_state *s, uint32_t r, uint64_t p, uint64_t pages) {
  double at = (pages ? (double)(p % pages) + 0.5 : 0) / (pages ? (double)pages : 1), sum = 0;
  uint32_t last = UINT32_MAX;
  for (uint32_t k = 0; k < s->trees; k++) if (s->root[k] == r) { last = k; sum += s->share[k]; if (at <= sum) return k; }
  return last;
}

int collective_plan(const struct collective_fabric *f, const struct collective_call *calls, uint32_t count,
                    uint32_t occurrences, struct collective_work **work_out) {
  struct collective_plan_state s = { .f = f, .n = f->nodes };
  const uint32_t n = f->nodes;
  /* the graph module's trees over the node graph, a pair's cables one hop of their summed rate */
  uint32_t m = 0, *from = malloc(n * n * sizeof *from), *to = malloc(n * n * sizeof *to), *through = malloc((n * n + 1) * sizeof *through);
  double *capacity = calloc(n * n, sizeof *capacity);
  for (uint32_t u = 0; u < n; u++) for (uint32_t v = 0; v < n; v++) {
    double rate = 0;
    for (uint32_t c = 0; c < f->count; c++) if ((f->links[c].a == u && f->links[c].b == v) || (f->links[c].a == v && f->links[c].b == u)) rate += f->links[c].bytes_a_second;
    if (u != v && rate > 0) { from[m] = u; to[m] = v; capacity[m] = rate / 1e9; through[m] = m; m++; }
  }
  through[m] = m;
  uint32_t room = n * 16, *crossing = malloc(m * sizeof *crossing);
  for (uint32_t e = 0; e < m; e++) crossing[e] = e;
  s.root = malloc(room * sizeof *s.root); s.parent = malloc(room * n * sizeof *s.parent);
  double *logw = malloc(room * sizeof *logw); s.share = malloc(room * sizeof *s.share);
  s.trees = mesh_graph_pack(n, m, from, to, through, crossing, m, capacity, collective_rooted, NULL, 0.1, 16, 1, s.root, s.parent, logw);
  for (uint32_t k = 0; k < s.trees; k++) s.share[k] = exp(logw[k]);
  free(from); free(to); free(through); free(capacity); free(crossing); free(logw);
  if (!s.trees) return EHOSTUNREACH;
  /* this node's links: the cables that end here, in fabric order */
  s.local = malloc(f->count * sizeof *s.local); s.receives = calloc(f->count, sizeof *s.receives);
  for (uint32_t c = 0, l = 0; c < f->count; c++) s.local[c] = (f->links[c].a == f->rank || f->links[c].b == f->rank) ? l++ : UINT32_MAX;
  struct collective_work *work = calloc((size_t)count * occurrences, sizeof *work);
  uint64_t published = 0;                /* the node's publication count, as the program will store it */
  for (uint32_t o = 0; o < occurrences; o++) for (uint32_t i = 0; i < count; i++) {
    struct collective_work *w = &work[(size_t)o * count + i];
    const struct collective_call *call = &calls[i];
    const bridge_page base = collective_page_of(call->receive);
    const uint64_t pages = (call->bytes + BRIDGE_PAGE - 1) / BRIDGE_PAGE, share = (pages + n - 1) / n;
    const uint64_t produced = ++published;  /* the publication after the program produced this call's pages */
    for (uint32_t r = 0; r < n; r++) {
      const uint64_t lo = (uint64_t)r * share, hi = lo + share < pages ? lo + share : pages;
      for (uint64_t p = lo; p < hi; p++) {
        const uint32_t *par = s.parent + (size_t)collective_tree(&s, r, p - lo, hi - lo) * n;
        const uint32_t bytes = (uint32_t)(p + 1 < pages || call->bytes % BRIDGE_PAGE == 0 ? BRIDGE_PAGE : call->bytes % BRIDGE_PAGE);
        uint32_t depth[64];
        for (uint32_t v = 0; v < n; v++) { depth[v] = 0; for (uint32_t u = v; u != r && par[u] != UINT32_MAX && depth[v] <= n; u = par[u]) depth[v]++; }
        /* the publication after which page p is ready at each node: produced, or its last fold's */
        uint64_t ready[64];
        for (uint32_t v = 0; v < n; v++) ready[v] = produced;
        if (call->what == COLLECTIVE_ALLREDUCE) {
          /* up, leaves first: v's partial (its page base + p, ready after its own last fold) to its parent u, landing in
             u's scratch copy for v (base + pages x (1 + v's place among u's children) + p), then u folds it in */
          uint32_t children[64] = {0};
          for (int d = (int)n; d >= 1; d--) for (uint32_t v = 0; v < n; v++) {
            if (depth[v] != (uint32_t)d || par[v] == UINT32_MAX || v == r) continue;
            const uint32_t u = par[v];
            const bridge_page scratch = base + (bridge_page)(pages * (1 + children[u]++) + p);
            const uint32_t send_link = v == f->rank ? s.local[collective_cable(f, v, u, p)] : 0;
            uint32_t cable;
            const uint64_t index = collective_move(&s, v, u, base + (bridge_page)p, scratch, bytes, p, send_link, 0, ready[v], &cable);
            if (u == f->rank) {
              struct collective_fold *g = realloc(w->folds, (w->fold_count + 1) * sizeof *g);
              if (!g) return ENOMEM;
              w->folds = g;
              w->folds[w->fold_count++] = (struct collective_fold){ .into = base + (bridge_page)p, .from = scratch, .bytes = bytes,
                                                                    .after_link = s.local[cable], .after = index };
              ready[u] = ++published;    /* this fold's publication */
            }
          }
        }
        /* down (an all-gather, and an all-reduce's second half): page p from r along its tree, the root's send once it
           is ready there, each relay forwarding it once it has landed there */
        uint64_t landed_at[64] = {0}; uint32_t in_link[64] = {0};
        for (uint32_t d = 1; d <= n; d++) for (uint32_t v = 0; v < n; v++) {
          const uint32_t u = par[v];
          if (u == UINT32_MAX || v == r || depth[v] != d) continue;
          uint32_t after_link = 0, after_landed = 0; uint64_t position = 0;
          if (u == f->rank && u == r) { after_link = s.local[collective_cable(f, u, v, p)]; position = ready[u]; }
          else if (u == f->rank) { after_link = in_link[u]; after_landed = 1; position = landed_at[u]; }
          uint32_t cable;
          landed_at[v] = collective_move(&s, u, v, base + (bridge_page)p, base + (bridge_page)p, bytes, p, after_link, after_landed, position, &cable);
          in_link[v] = s.local[cable] == UINT32_MAX ? 0 : s.local[cable];
        }
      }
    }
    for (uint32_t c = 0; c < f->count; c++) if (s.local[c] != UINT32_MAX) w->await[s.local[c]] = s.receives[c];
  }
  free(s.root); free(s.parent); free(s.share); free(s.local); free(s.receives);
  const int e = bridge_load(s.steps, (uint32_t)s.step_count);
  free(s.steps);
  *work_out = work;
  return e;
}
#endif

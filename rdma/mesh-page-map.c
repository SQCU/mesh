/* design/page-maps-2026-11-02.md: mesh-page-map.h's operations, integer arithmetic on pieces and page runs only. */
#include "mesh-page-map.h"
#include "mesh-graph.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* the run of invocation i's runs holding virtual page vp: its index, or SIZE_MAX where none does */
static size_t page_run(const struct mesh_page_map *map, uint64_t i, uint64_t vp) {
  size_t lo = (size_t)map->first[i], hi = (size_t)map->first[i + 1];
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    const struct mesh_map_run *r = map->runs + mid;
    if (vp < r->first_virtual) hi = mid;
    else if (vp >= r->first_virtual + r->count) lo = mid + 1;
    else return mid;
  }
  return SIZE_MAX;
}

struct image_out { struct mesh_page_bytes *out; size_t capacity, count; int unmapped; };
/* virtual bytes [vbyte, vbyte + bytes) of a value at invocation i, cut at page edges, each piece's physical page
   appended to `o`, joined with the last where it continues it on the same page */
static void image_bytes(const struct mesh_page_space *space, const struct mesh_page_map *map, uint64_t i, uint64_t vbyte,
                        uint64_t bytes, struct image_out *o) {
  const uint64_t P = space->page_bytes;
  while (bytes && !o->unmapped) {
    const uint64_t vp = vbyte / P, within = vbyte % P, take = bytes < P - within ? bytes : P - within;
    const size_t r = page_run(map, i, vp);
    if (r == SIZE_MAX) { o->unmapped = 1; return; }
    const uint64_t page = map->runs[r].first_physical + (vp - map->runs[r].first_virtual);
    struct mesh_page_bytes *last = o->count && o->count <= o->capacity ? o->out + o->count - 1 : NULL;
    if (last && last->page == page && last->first + last->bytes == within) last->bytes += take;
    else {
      if (o->count < o->capacity) o->out[o->count] = (struct mesh_page_bytes){page, within, take};
      o->count++;
    }
    vbyte += take; bytes -= take;
  }
}
/* every element of piece p in logical (row-major box) order, as virtual byte runs: a unit innermost stride makes one
   run of its row, a zero one a single element, any other stride an element each */
static void piece_bytes(const struct mesh_page_space *space, const struct mesh_page_map *map, uint64_t i,
                        const struct mesh_piece *p, uint64_t element_bytes, struct image_out *o) {
  if (!p->rank) { image_bytes(space, map, i, (uint64_t)p->origin * element_bytes, element_bytes, o); return; }
  int64_t c[MESH_PIECE_RANK] = {0};
  for (uint32_t d = 0; d < p->rank; d++) if (p->extent[d] <= 0) return;
  const uint32_t last = p->rank - 1;
  for (;;) {
    int64_t at = p->origin;
    for (uint32_t d = 0; d < last; d++) at += p->stride[d] * c[d];
    if (p->stride[last] == 1) image_bytes(space, map, i, (uint64_t)at * element_bytes, (uint64_t)p->extent[last] * element_bytes, o);
    else if (p->stride[last] == 0) image_bytes(space, map, i, (uint64_t)at * element_bytes, element_bytes, o);
    else for (int64_t k = 0; k < p->extent[last]; k++)
      image_bytes(space, map, i, (uint64_t)(at + p->stride[last] * k) * element_bytes, element_bytes, o);
    uint32_t d = last;
    while (d-- > 0) { if (++c[d] < p->extent[d]) break; c[d] = 0; }
    if (d == UINT32_MAX) return;
  }
}

/* mesh-page-map.h */
size_t mesh_page_image(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t invocation,
                       struct mesh_page_bytes *out, size_t capacity) {
  struct image_out o = {out, capacity, 0, 0};
  const uint64_t element_bytes = v->layout.element_bits / 8;
  for (size_t k = 0; k < v->layout.count && !o.unmapped; k++)
    piece_bytes(space, &v->map, invocation, v->layout.pieces + k, element_bytes, &o);
  return o.unmapped ? SIZE_MAX : o.count;
}

static int page_bytes_order(const void *left, const void *right) {
  const struct mesh_page_bytes *a = left, *b = right;
  if (a->page != b->page) return a->page < b->page ? -1 : 1;
  return (a->first > b->first) - (a->first < b->first);
}
/* value v's image at invocation i, sorted by page and first byte: a malloc'd array and its count, NULL where v's map
   leaves a page unmapped (*count SIZE_MAX) or memory is short (*count 0) */
static struct mesh_page_bytes *sorted_image(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t i,
                                            size_t *count) {
  *count = mesh_page_image(space, v, i, NULL, 0);
  if (*count == SIZE_MAX) return NULL;
  struct mesh_page_bytes *out = malloc((*count + 1) * sizeof *out);
  if (!out) { *count = 0; return NULL; }
  mesh_page_image(space, v, i, out, *count);
  qsort(out, *count, sizeof *out, page_bytes_order);
  return out;
}

/* mesh-page-map.h */
int mesh_page_meet(const struct mesh_page_space *space, const struct mesh_value *a, uint64_t ia,
                   const struct mesh_lifetime *la, const struct mesh_value *b, uint64_t ib, const struct mesh_lifetime *lb) {
  if (la->end <= lb->begin || lb->end <= la->begin) return 0;
  size_t na = 0, nb = 0;
  struct mesh_page_bytes *x = sorted_image(space, a, ia, &na), *y = sorted_image(space, b, ib, &nb);
  int met = na == SIZE_MAX || nb == SIZE_MAX || (!x && na) || (!y && nb) ? -1 : 0;
  for (size_t i = 0, j = 0; !met && i < na && j < nb;) {
    if (x[i].page != y[j].page) { if (x[i].page < y[j].page) i++; else j++; continue; }
    if (x[i].first + x[i].bytes <= y[j].first) i++;
    else if (y[j].first + y[j].bytes <= x[i].first) j++;
    else met = 1;
  }
  free(x); free(y);
  return met;
}

/* mesh-page-map.h */
size_t mesh_page_preimage(const struct mesh_page_space *space, const struct mesh_value *v, uint64_t invocation,
                          const struct mesh_page_bytes *bytes, size_t count, struct mesh_piece *out, size_t capacity) {
  /* each element of each piece, tested against the byte runs; consecutive elements of one row inside them become one
     piece of that row. Bounded by the value's element count, at preparation */
  size_t written = 0;
  const uint64_t element_bytes = v->layout.element_bits / 8, P = space->page_bytes;
  for (size_t k = 0; k < v->layout.count; k++) {
    const struct mesh_piece *p = v->layout.pieces + k;
    if (!p->rank) continue;
    int64_t c[MESH_PIECE_RANK] = {0};
    int empty = 0;
    for (uint32_t d = 0; d < p->rank; d++) empty |= p->extent[d] <= 0;
    const uint32_t last = p->rank - 1;
    while (!empty) {
      int64_t run = -1;
      for (int64_t e = 0; e <= p->extent[last]; e++) {
        int inside = 0;
        if (e < p->extent[last]) {
          int64_t at = p->origin + p->stride[last] * e;
          for (uint32_t d = 0; d < last; d++) at += p->stride[d] * c[d];
          const uint64_t vbyte = (uint64_t)at * element_bytes, vp = vbyte / P, within = vbyte % P;
          const size_t r = page_run(&v->map, invocation, vp);
          if (r == SIZE_MAX) return SIZE_MAX;
          const uint64_t page = v->map.runs[r].first_physical + (vp - v->map.runs[r].first_virtual);
          for (size_t b = 0; !inside && b < count; b++)
            inside = bytes[b].page == page && within >= bytes[b].first && within + element_bytes <= bytes[b].first + bytes[b].bytes;
        }
        if (inside && run < 0) run = e;
        if (!inside && run >= 0) {
          if (written < capacity) {
            struct mesh_piece q = *p;
            for (uint32_t d = 0; d < last; d++) { q.origin += p->stride[d] * c[d]; q.first[d] = p->first[d] + c[d]; q.extent[d] = 1; }
            q.origin += p->stride[last] * run; q.first[last] = p->first[last] + run; q.extent[last] = e - run;
            out[written] = q;
          }
          written++; run = -1;
        }
      }
      uint32_t d = last;
      while (d-- > 0) { if (++c[d] < p->extent[d]) break; c[d] = 0; }
      if (d == UINT32_MAX) break;
    }
  }
  return written;
}

/* the interval of view axis d's coordinates (relative, [lo, hi)) whose image on one axis stays in [first, first +
   extent), the axis walked by d alone, from `at` (the view's base and every other contribution); [0, 0) where none */
static void axis_window(int64_t at, int64_t step, int64_t extent_view, int64_t first, int64_t extent, int64_t *lo, int64_t *hi) {
  if (step == 0) { const int in = at >= first && at < first + extent; *lo = 0; *hi = in ? extent_view : 0; return; }
  /* at + step * c in [first, first + extent), c in [0, extent_view) */
  int64_t a, b;
  if (step > 0) { a = first - at; b = first + extent - at; a = a > 0 ? (a + step - 1) / step : -((-a) / step); b = b > 0 ? (b + step - 1) / step : -((-b) / step); }
  else { const int64_t s = -step; a = at - (first + extent) + 1; b = at - first + 1; a = a > 0 ? (a + s - 1) / s : -((-a) / s); b = b > 0 ? (b + s - 1) / s : -((-b) / s); }
  *lo = a < 0 ? 0 : a; *hi = b > extent_view ? extent_view : b;
  if (*hi < *lo) *hi = *lo;
}

/* mesh-page-map.h */
size_t mesh_page_compose(const struct mesh_view *view, const struct mesh_layout *of, struct mesh_piece *out,
                         size_t capacity) {
  size_t written = 0;
  for (size_t k = 0; k < of->count; k++) {
    const struct mesh_piece *p = of->pieces + k;
    if (p->rank != view->of_rank) return SIZE_MAX;
    struct mesh_piece q = {.rank = view->rank};
    int64_t lo[MESH_PIECE_RANK], hi[MESH_PIECE_RANK];
    for (uint32_t d = 0; d < view->rank; d++) { lo[d] = 0; hi[d] = view->extent[d]; }
    int empty = 0;
    for (uint32_t a = 0; a < p->rank && !empty; a++) {
      uint32_t walkers = 0, only = 0;
      int64_t span_lo = view->base[a], span_hi = view->base[a];
      for (uint32_t d = 0; d < view->rank; d++) if (view->axis[d] == a && view->step[d]) {
        walkers++; only = d;
        const int64_t reach = view->step[d] * (view->extent[d] - 1);
        if (reach > 0) span_hi += reach; else span_lo += reach;
      }
      if (walkers == 0) { empty = view->base[a] < p->first[a] || view->base[a] >= p->first[a] + p->extent[a]; continue; }
      if (span_lo >= p->first[a] && span_hi < p->first[a] + p->extent[a]) continue;  /* the whole image inside */
      if (walkers > 2) return SIZE_MAX;
      if (walkers == 2) {
        /* a split: outer axis o and inner axis i, the inner's image exactly one step of the outer's (mixed radix,
           positive steps). The box cuts only the outer axis where its edges fall on the outer's step from base */
        uint32_t o = UINT32_MAX, in = UINT32_MAX;
        for (uint32_t d = 0; d < view->rank; d++) if (view->axis[d] == a && view->step[d]) { if (o == UINT32_MAX) o = d; else in = d; }
        if (view->step[o] < view->step[in]) { const uint32_t t = o; o = in; in = t; }
        const int64_t so = view->step[o], si = view->step[in];
        if (si <= 0 || so != si * view->extent[in]) return SIZE_MAX;
        const int64_t first_rel = p->first[a] - view->base[a], end_rel = first_rel + p->extent[a];
        const int64_t span = so * view->extent[o];
        if ((first_rel > 0 && first_rel < span && first_rel % so) || (end_rel > 0 && end_rel < span && end_rel % so)) return SIZE_MAX;
        only = o;
      }
      int64_t l, h;
      axis_window(view->base[a], view->step[only], view->extent[only], p->first[a], p->extent[a], &l, &h);
      if (l > lo[only]) lo[only] = l;
      if (h < hi[only]) hi[only] = h;
      empty = hi[only] <= lo[only];
    }
    if (empty) continue;
    /* virtual = p.origin + sum_a p.stride[a] * (coord_a - p.first[a]), coord_a = base[a] + sum over view axes walking a */
    q.origin = p->origin;
    for (uint32_t a = 0; a < p->rank; a++) q.origin += p->stride[a] * (view->base[a] - p->first[a]);
    for (uint32_t d = 0; d < view->rank; d++) {
      const int64_t s = view->step[d] ? p->stride[view->axis[d]] * view->step[d] : 0;
      q.stride[d] = s; q.first[d] = view->first[d] + lo[d]; q.extent[d] = hi[d] - lo[d];
      q.origin += s * lo[d];
    }
    if (written < capacity) out[written] = q;
    written++;
  }
  return written;
}

/* mesh-page-map.h: the run's solve */
static int run_edge_order(const void *left, const void *right) {
  const struct mesh_graph_dependence *a = left, *b = right;
  if (a->to != b->to) return a->to < b->to ? -1 : 1;
  if (a->from != b->from) return a->from < b->from ? -1 : 1;
  if (a->kind != b->kind) return a->kind < b->kind ? -1 : 1;
  if (a->item != b->item) return a->item < b->item ? -1 : 1;
  if (a->first != b->first) return a->first < b->first ? -1 : 1;
  return (a->bytes > b->bytes) - (a->bytes < b->bytes);
}
static uint32_t run_edge_head(const void *context, size_t k) { return ((const uint32_t *)context)[k]; }
/* before[a * items + b]: every access of item a happens before every access of item b */
struct run_before { const unsigned char *before; size_t items; };
static int run_bit(const unsigned char *bits, size_t at) { return bits[at / 8] >> (at % 8) & 1; }
static int run_items_apart(const void *context, size_t k, size_t l) {
  const struct run_before *r = context;
  return run_bit(r->before, k * r->items + l) || run_bit(r->before, l * r->items + k);
}
int mesh_run_solve(const struct mesh_page_space *space, uint32_t events, size_t accesses,
                   const struct mesh_graph_access *access, size_t links, const uint32_t (*link)[2], size_t items,
                   const struct mesh_run_item *item, size_t bins, const struct mesh_run_bin *bin,
                   struct mesh_run_plan *plan) {
  memset(plan, 0, sizeof *plan);
  if (!bins || !space->page_bytes || (items && items > SIZE_MAX / items)) return EINVAL;
  for (size_t b = 0; b < bins; b++)
    if (bin[b].first > UINT64_MAX - bin[b].pages || (b && bin[b].first < bin[b - 1].first + bin[b - 1].pages))
      return EINVAL;
  size_t spans = 0;
  for (size_t a = 0; a < accesses; a++) {
    const struct mesh_graph_access *x = access + a;
    if (x->begin >= events || x->end >= events || x->item >= items || !(x->mode & 3u) || (x->mode & ~3u) ||
        x->first > UINT64_MAX - x->bytes || item[x->item].pages > UINT64_MAX / space->page_bytes ||
        x->first + x->bytes > item[x->item].pages * space->page_bytes)
      return EINVAL;
    spans += x->begin != x->end;
  }
  for (size_t k = 0; k < links; k++)
    if (link[k][0] >= events || link[k][1] >= events) return EINVAL;
  const size_t found = mesh_graph_dependences(accesses, access, NULL, 0);
  if (found == SIZE_MAX) return ENOMEM;
  const size_t m = found + spans + links, n = events;
  plan->edge = malloc((m + 1) * sizeof *plan->edge);
  plan->order = malloc((n + 1) * sizeof *plan->order);
  plan->place = malloc((items + 1) * sizeof *plan->place);
  size_t *first = calloc(n + 2, sizeof *first), *cursor = malloc((n + 1) * sizeof *cursor),
         *by_item = calloc(items + 2, sizeof *by_item), *of_item = malloc((accesses + 1) * sizeof *of_item);
  uint32_t *head = malloc((m + 1) * sizeof *head), *indegree = malloc((n + 1) * sizeof *indegree),
           *ordinal = malloc((n + 1) * sizeof *ordinal), *seen = calloc(n + 1, sizeof *seen),
           *reached = malloc((n + 1) * sizeof *reached), *stack = malloc((n + 1) * sizeof *stack);
  uint64_t *size = malloc((items + 1) * sizeof *size), *key = malloc((items + 1) * sizeof *key),
           (*bounds)[2] = malloc((bins + 1) * sizeof *bounds);
  unsigned char *pinned = malloc(items + 1), *before = calloc((items * items + 7) / 8 + 1, 1);
  int result = plan->edge && plan->order && plan->place && first && cursor && by_item && of_item && head && indegree &&
               ordinal && seen && reached && stack && size && key && bounds && pinned && before ? 0 : ENOMEM;
  /* the edges: the dependences, each access's own span, the transport's links; duplicates once */
  if (!result && mesh_graph_dependences(accesses, access, plan->edge, found) != found) result = ENOMEM;
  size_t e = found;
  for (size_t a = 0; !result && a < accesses; a++)
    if (access[a].begin != access[a].end)
      plan->edge[e++] = (struct mesh_graph_dependence){access[a].begin, access[a].end, MESH_GRAPH_LINK, access[a].item, 0, 0};
  for (size_t k = 0; !result && k < links; k++)
    plan->edge[e++] = (struct mesh_graph_dependence){link[k][0], link[k][1], MESH_GRAPH_LINK, 0, 0, 0};
  if (!result) {
    qsort(plan->edge, e, sizeof *plan->edge, run_edge_order);
    size_t kept = 0;
    for (size_t k = 0; k < e; k++)
      if (!kept || run_edge_order(plan->edge + kept - 1, plan->edge + k)) plan->edge[kept++] = plan->edge[k];
    plan->edges = kept;
    /* each event's edges out, as rows */
    for (size_t k = 0; k < kept; k++) first[plan->edge[k].from + 1]++;
    for (size_t v = 0; v < n; v++) { first[v + 1] += first[v]; cursor[v] = first[v]; }
    for (size_t k = 0; k < kept; k++) head[cursor[plan->edge[k].from]++] = plan->edge[k].to;
    if (mesh_graph_order(events, first, run_edge_head, head, indegree, ordinal)) result = EINVAL;
  }
  for (size_t v = 0; !result && v < n; v++) plan->order[ordinal[v]] = (uint32_t)v;
  /* each item's accesses, as rows */
  for (size_t a = 0; !result && a < accesses; a++) by_item[access[a].item + 1]++;
  for (size_t k = 0; !result && k < items; k++) by_item[k + 1] += by_item[k];
  for (size_t a = 0; !result && a < accesses; a++) of_item[by_item[access[a].item]++] = a;
  for (size_t k = items; !result && k > 0; k--) by_item[k] = by_item[k - 1];
  if (!result) by_item[0] = 0;
  /* before(a, b): every begin of b is reached from every end of a. reached[v] counts the ends of a that reach v */
  uint32_t stamp = 0;
  for (size_t a = 0; !result && a < items; a++) {
    if (item[a].lives & MESH_RUN_AFTER) continue;
    for (size_t v = 0; v < n; v++) reached[v] = 0;
    uint32_t ends = 0;
    for (size_t x = by_item[a]; x < by_item[a + 1]; x++) {
      stack[0] = access[of_item[x]].end;
      mesh_graph_mark(first, run_edge_head, head, seen, ++stamp, stack, 1);
      for (size_t v = 0; v < n; v++) reached[v] += seen[v] == stamp;
      ends++;
    }
    for (size_t b = 0; b < items; b++) {
      if (b == a || (item[b].lives & MESH_RUN_BEFORE)) continue;
      int all = 1;
      for (size_t y = by_item[b]; all && y < by_item[b + 1]; y++) all = reached[access[of_item[y]].begin] == ends;
      if (all) before[(a * items + b) / 8] |= (unsigned char)(1u << ((a * items + b) % 8));
    }
  }
  /* place: pins, then greedy by size; an item with no access, pin or life holds no page */
  for (size_t k = 0; !result && k < items; k++) {
    const int touched = by_item[k + 1] > by_item[k];
    pinned[k] = item[k].pin != MESH_RUN_FREE;
    size[k] = touched || pinned[k] || item[k].lives ? item[k].pages : 0;
    plan->place[k] = pinned[k] ? item[k].pin : 0;
    key[k] = UINT64_MAX;
    for (size_t x = by_item[k]; x < by_item[k + 1]; x++)
      if (ordinal[access[of_item[x]].begin] < key[k]) key[k] = ordinal[access[of_item[x]].begin];
  }
  for (size_t b = 0; !result && b < bins; b++) { bounds[b][0] = bin[b].first; bounds[b][1] = bin[b].pages; }
  const struct run_before relation = {before, items};
  if (!result)
    result = mesh_graph_place(items, size, key, pinned, bins, (const uint64_t (*)[2])bounds, plan->place,
                              run_items_apart, &relation);
  for (size_t k = 0; !result && k < items; k++)
    if (plan->place[k] != MESH_RUN_FREE && plan->place[k] >= bin[bins - 1].first &&
        plan->place[k] + size[k] - bin[bins - 1].first > plan->arena)
      plan->arena = plan->place[k] + size[k] - bin[bins - 1].first;
  free(first); free(cursor); free(by_item); free(of_item); free(head); free(indegree); free(ordinal); free(seen);
  free(reached); free(stack); free(size); free(key); free(bounds); free(pinned); free(before);
  if (result) mesh_run_plan_free(plan);
  return result;
}
void mesh_run_plan_free(struct mesh_run_plan *plan) {
  free(plan->edge); free(plan->order); free(plan->place);
  memset(plan, 0, sizeof *plan);
}

/* design/page-maps-2026-11-02.md: mesh-page-map.h's operations, integer arithmetic on pieces and page runs only. */
#include "mesh-page-map.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* the run of invocation i's runs holding virtual page vp: its index, or SIZE_MAX where none does */
static size_t page_run(const struct mesh_page_map *map, uint64_t i, uint64_t vp) {
  size_t lo = (size_t)map->first[i], hi = (size_t)map->first[i + 1];
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    const struct mesh_page_run *r = map->runs + mid;
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

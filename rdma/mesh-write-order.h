/* AOT ordering of streamed path fragments. No clocks or transport words are
   read here. Every task occupies one directed cable; a relay can begin after
   its parent's first page, while its last page still depends on that parent. */
#ifndef MESH_WRITE_ORDER_H
#define MESH_WRITE_ORDER_H
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct mesh_write_task {
  uint32_t arc, parent, order;
  double bytes, release, start, finish, first_bytes;
};

static int mesh_write_order(struct mesh_write_task *tasks, uint32_t count,
                            const double *capacity, uint32_t arcs, double page) {
  double *free_at = calloc(arcs ? arcs : 1, sizeof *free_at);
  if (!free_at) return ENOMEM;
  for (uint32_t e = 0; e < arcs; e++) if (!(capacity[e] > 0) || !isfinite(capacity[e])) { free(free_at); return EINVAL; }
  for (uint32_t k = 0; k < count; k++) {
    tasks[k].order = UINT32_MAX;
    if (tasks[k].arc >= arcs || (tasks[k].parent != UINT32_MAX && tasks[k].parent >= count) ||
        !(tasks[k].bytes > 0) || (tasks[k].first_bytes &&
          (!(tasks[k].first_bytes > 0) || tasks[k].first_bytes > tasks[k].bytes)) || !isfinite(tasks[k].release)) { free(free_at); return EINVAL; }
  }
  for (uint32_t order = 0; order < count; order++) {
    uint32_t best = UINT32_MAX;
    double begin = INFINITY, end = 0;
    for (uint32_t k = 0; k < count; k++) {
      struct mesh_write_task *t = tasks + k;
      if (t->order != UINT32_MAX) continue;
      const struct mesh_write_task *p = t->parent == UINT32_MAX ? NULL : tasks + t->parent;
      if (p && p->order == UINT32_MAX) continue;
      double at = fmax(free_at[t->arc], p ? p->start + (p->first_bytes ? p->first_bytes : fmin(page, p->bytes)) / capacity[p->arc] : t->release);
      double until = at + t->bytes / capacity[t->arc];
      if (p) until = fmax(until, p->finish + (t->first_bytes ? t->first_bytes : fmin(page, t->bytes)) / capacity[t->arc]);
      if (at < begin) { best = k; begin = at; end = until; }
    }
    if (best == UINT32_MAX) { free(free_at); return ELOOP; }
    tasks[best].order = order; tasks[best].start = begin; tasks[best].finish = end;
    free_at[tasks[best].arc] = end;
  }
  free(free_at);
  return 0;
}
#endif

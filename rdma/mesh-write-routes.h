/* Communicator-realization defaults for explicit physical writes. Every simple
   node path is represented, with parallel cables remaining distinct choices.
   Equal zero log weights introduce no hop penalty. A solver can replace a
   pair's weights later through ncclMeshCommWriteRoutes; execution never calls
   this enumerator. Path order matches routing.py Fabric's cable-order DFS. */
#ifndef MESH_WRITE_ROUTES_H
#define MESH_WRITE_ROUTES_H
#include "mesh-plan.h"
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The same ownership as nccl-mesh.c paths_free: three independent allocations
   per pair, and one nodes*nodes table. Empty diagonal entries own no arrays. */
static inline void mesh_write_routes_free(struct mesh_paths *routes, uint32_t nodes) {
  if (!routes) return;
  for (size_t pair = 0; pair < (size_t)nodes * nodes; pair++) {
    free((void *)routes[pair].first);
    free((void *)routes[pair].cable);
    free((void *)routes[pair].log_weight);
  }
  free(routes);
}

struct mesh_write_routes_walk {
  uint32_t *node, *next, *path;
  unsigned char *seen;
};

/* Count first, then fill exactly the same paths. The explicit DFS stack keeps
   depth off the C call stack. `next` is the next cable index at each depth;
   visiting a destination ends that path rather than forwarding through it. */
static inline int mesh_write_routes_walk(const struct mesh_fabric *f, uint32_t source, uint32_t destination,
                                         struct mesh_write_routes_walk *walk, uint32_t *first, uint32_t *cable,
                                         uint32_t *paths, uint32_t *edges) {
  uint32_t depth = 0;
  *paths = *edges = 0;
  if (first) first[0] = 0;
  memset(walk->seen, 0, f->nodes);
  walk->node[0] = source; walk->next[0] = 0; walk->seen[source] = 1;
  for (;;) {
    const uint32_t at = walk->node[depth];
    if (at == destination) {
      if (*paths == UINT32_MAX || depth > UINT32_MAX - *edges) return EOVERFLOW;
      if (cable) memcpy(cable + *edges, walk->path, (size_t)depth * sizeof *cable);
      *edges += depth;
      (*paths)++;
      if (first) first[*paths] = *edges;
    } else {
      uint32_t other = 0, edge = 0;
      int found = 0;
      while (walk->next[depth] < f->cables) {
        edge = walk->next[depth]++;
        const struct mesh_cable *c = f->cable + edge;
        if (c->node[0] != at && c->node[1] != at) continue;
        other = c->node[c->node[0] == at];
        if (!walk->seen[other]) { found = 1; break; }
      }
      if (found) {
        walk->path[depth++] = edge;
        walk->node[depth] = other; walk->next[depth] = 0; walk->seen[other] = 1;
        continue;
      }
    }
    if (!depth) break;
    walk->seen[at] = 0;
    depth--;
  }
  return *paths ? 0 : ENOLINK;
}

/* Positive errno on failure, with *out unchanged. *out must initially be NULL:
   initialization cannot accidentally overwrite a solver's existing routes.
   There is no path-count/hop cap: only mesh_paths' uint32 representation and
   allocation limits bound enumeration. No partial table is ever published. */
static inline int mesh_write_routes_default(const struct mesh_fabric *f, struct mesh_paths **out) {
  if (!out || *out) return EINVAL;
  int error = mesh_fabric_check(f);
  if (error) return error;
  const uint32_t n = f->nodes;
  if ((uint64_t)n * n > SIZE_MAX / sizeof(struct mesh_paths) ||
      (uint64_t)n > SIZE_MAX / sizeof(uint32_t)) return EOVERFLOW;
  struct mesh_paths *routes = calloc((size_t)n * n, sizeof *routes);
  if (!routes) return ENOMEM;
  if (n == 1) { *out = routes; return 0; }
  struct mesh_write_routes_walk walk = {
    .node = malloc((size_t)n * sizeof *walk.node),
    .next = malloc((size_t)n * sizeof *walk.next),
    .path = malloc((size_t)n * sizeof *walk.path),
    .seen = calloc(n, sizeof *walk.seen)
  };
  if (!walk.node || !walk.next || !walk.path || !walk.seen) { error = ENOMEM; goto done; }
  /* Reject disconnected fabrics before enumerating potentially large connected
     components. This is connectivity validation, not a subset of allowed paths. */
  uint32_t reached = 1;
  walk.node[0] = 0; walk.seen[0] = 1;
  for (uint32_t k = 0; k < reached; k++) for (uint32_t e = 0; e < f->cables; e++) {
    const struct mesh_cable *c = f->cable + e;
    if (c->node[0] != walk.node[k] && c->node[1] != walk.node[k]) continue;
    const uint32_t other = c->node[c->node[0] == walk.node[k]];
    if (!walk.seen[other]) { walk.seen[other] = 1; walk.node[reached++] = other; }
  }
  if (reached != n) { error = ENOLINK; goto done; }
  for (uint32_t source = 0; source < n; source++) for (uint32_t destination = 0; destination < n; destination++) {
    if (source == destination) continue;
    uint32_t paths, edges;
    error = mesh_write_routes_walk(f, source, destination, &walk, NULL, NULL, &paths, &edges);
    if (error) goto done;
    if ((uint64_t)paths + 1 > SIZE_MAX / sizeof(uint32_t) ||
        (uint64_t)edges > SIZE_MAX / sizeof(uint32_t) ||
        (uint64_t)paths > SIZE_MAX / sizeof(double)) { error = EOVERFLOW; goto done; }
    struct mesh_paths *pair = routes + (size_t)source * n + destination;
    uint32_t *first = malloc(((size_t)paths + 1) * sizeof *first);
    uint32_t *cable = malloc((size_t)edges * sizeof *cable);
    double *weights = calloc(paths, sizeof *weights);
    *pair = (struct mesh_paths){paths, first, cable, weights};
    if (!first || !cable || !weights) { error = ENOMEM; goto done; }
    error = mesh_write_routes_walk(f, source, destination, &walk, first, cable, &paths, &edges);
    if (error) goto done;
  }
done:
  free(walk.node); free(walk.next); free(walk.path); free(walk.seen);
  if (error) mesh_write_routes_free(routes, n);
  else *out = routes;
  return error;
}
#endif

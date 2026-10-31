#include "mesh-graph.h"
#include <errno.h>

/* mesh-graph.h */
double mesh_graph_tree_path(const uint32_t *parent, uint32_t from, uint32_t hops,
                            uint32_t (*step)(const void *context, uint32_t k, uint32_t at),
                            double (*split)(const void *context, uint32_t k, uint32_t at), const void *context,
                            uint32_t *end) {
  double part = 1;
  uint32_t at = from;
  for (uint32_t k = 0; k < hops; k++) {
    const uint32_t next = step(context, k, at);
    if (parent[at] != next) { *end = at; return 0; }
    part *= split(context, k, at);
    at = next;
  }
  *end = at;
  return part;
}

/* mesh-graph.h */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached) {
  for (uint32_t v = 0; v < vertices; v++) reached[v] = 0;
  uint32_t count = 1;
  order[0] = from; reached[from] = 1;
  for (uint32_t head = 0; head < count; head++)
    for (uint32_t e = 0; e < edges; e++)
      for (uint32_t side = 0; side < 2; side++) {
        const uint32_t other = end(context, e, 1 - side);
        if (end(context, e, side) == order[head] && !reached[other]) { reached[other] = 1; order[count++] = other; }
      }
  return count;
}

/* mesh-graph.h: node[d] the vertex at depth d, next[d] the next edge it tries, path[d] the edge it left by */
int mesh_graph_paths(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                     const void *context, uint32_t source, uint32_t destination, uint32_t *work, unsigned char *seen,
                     int (*visit)(void *consumer, const uint32_t *path, uint32_t length), void *consumer) {
  uint32_t *node = work, *next = work + vertices, *path = work + 2 * (uint64_t)vertices, depth = 0;
  int any = 0;
  for (uint32_t v = 0; v < vertices; v++) seen[v] = 0;
  node[0] = source; next[0] = 0; seen[source] = 1;
  for (;;) {
    const uint32_t at = node[depth];
    if (at == destination) {
      const int stop = visit(consumer, path, depth);
      if (stop) return stop;
      any = 1;
    } else {
      uint32_t other = 0, edge = 0;
      int found = 0;
      while (next[depth] < edges) {
        edge = next[depth]++;
        const uint32_t a = end(context, edge, 0), b = end(context, edge, 1);
        if (a != at && b != at) continue;
        other = a == at ? b : a;
        if (!seen[other]) { found = 1; break; }
      }
      if (found) {
        path[depth++] = edge;
        node[depth] = other; next[depth] = 0; seen[other] = 1;
        continue;
      }
    }
    if (!depth) break;
    seen[at] = 0;
    depth--;
  }
  return any ? 0 : ENOLINK;
}

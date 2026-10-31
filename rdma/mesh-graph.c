#include "mesh-graph.h"
#include <errno.h>
#include <stdlib.h>

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

/* mesh-graph.h */
void mesh_graph_mark(const size_t *first, uint32_t (*head)(const void *context, size_t k), const void *context,
                     uint32_t *seen, uint32_t stamp, uint32_t *stack, size_t pending) {
  while (pending) {
    const uint32_t at = stack[--pending];
    for (size_t k = first[at]; k < first[at + 1]; k++) {
      const uint32_t next = head(context, k);
      if (seen[next] != stamp) { seen[next] = stamp; stack[pending++] = next; }
    }
  }
}

/* mesh-graph.h */
int mesh_graph_order(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                     const void *context, uint32_t *indegree, uint32_t *ordinal) {
  for (uint32_t v = 0; v < vertices; v++) { indegree[v] = 0; ordinal[v] = UINT32_MAX; }
  for (size_t k = 0; k < first[vertices]; k++) indegree[head(context, k)]++;
  for (uint32_t place = 0; place < vertices; place++) {
    uint32_t next = 0;
    while (next < vertices && (indegree[next] || ordinal[next] != UINT32_MAX)) next++;
    if (next == vertices) return EDEADLK;
    ordinal[next] = place;
    for (size_t k = first[next]; k < first[next + 1]; k++) indegree[head(context, k)]--;
  }
  return 0;
}

/* mesh-graph.h */
int mesh_graph_arborescence(uint32_t N,uint32_t root,uint32_t E,const uint32_t *from,const uint32_t *to,const double *weight,int32_t *chosen){
  int32_t *best=malloc(N*sizeof *best),*id=malloc(N*sizeof *id),*seen=malloc(N*sizeof *seen);
  uint32_t *nfrom=malloc((E+1)*sizeof *nfrom),*nto=malloc((E+1)*sizeof *nto),*origin=malloc((E+1)*sizeof *origin);
  double *nweight=malloc((E+1)*sizeof *nweight);
  int32_t *nchosen=malloc(N*sizeof *nchosen);
  unsigned char *cyclic=calloc(N,1);
  int status=-1;
  if(!best || !id || !seen || !nfrom || !nto || !origin || !nweight || !nchosen || !cyclic)goto done;
  for(uint32_t v=0;v<N;v++){best[v]=-1;id[v]=-1;seen[v]=-1;}
  for(uint32_t e=0;e<E;e++)
    if(from[e]!=to[e] && to[e]!=root && (best[to[e]]<0 || weight[e]<weight[best[to[e]]]))best[to[e]]=(int32_t)e;
  for(uint32_t v=0;v<N;v++)if(v!=root && best[v]<0)goto done;
  uint32_t components=0;
  for(uint32_t v=0;v<N;v++){
    uint32_t at=v;
    while(at!=root && seen[at]<0 && id[at]<0){seen[at]=(int32_t)v;at=from[best[at]];}
    if(at!=root && id[at]<0 && seen[at]==(int32_t)v){
      uint32_t x=at;
      do{id[x]=(int32_t)components;x=from[best[x]];}while(x!=at);
      cyclic[components++]=1;
    }
  }
  if(!components){
    for(uint32_t v=0;v<N;v++)chosen[v]=v==root?-1:best[v];
    status=0;goto done;
  }
  for(uint32_t v=0;v<N;v++)if(id[v]<0)id[v]=(int32_t)components++;
  uint32_t m=0;
  for(uint32_t e=0;e<E;e++){
    const uint32_t u=(uint32_t)id[from[e]],w=(uint32_t)id[to[e]];
    if(u==w)continue;
    nfrom[m]=u;nto[m]=w;origin[m]=e;
    nweight[m]=cyclic[w]?weight[e]-weight[best[to[e]]]:weight[e];
    m++;
  }
  if(mesh_graph_arborescence(components,(uint32_t)id[root],m,nfrom,nto,nweight,nchosen))goto done;
  for(uint32_t v=0;v<N;v++)chosen[v]=v==root?-1:best[v];
  for(uint32_t x=0;x<components;x++)
    if(x!=(uint32_t)id[root]){const uint32_t e=origin[nchosen[x]];chosen[to[e]]=(int32_t)e;}
  status=0;
done:
  free(best);free(id);free(seen);free(nfrom);free(nto);free(origin);free(nweight);free(nchosen);free(cyclic);
  return status;
}

/* mesh-graph.h */
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer) {
  uint32_t at = from;
  for (uint32_t step = 0; step <= vertices; step++) {
    if (visit) visit(consumer, at, step);
    if (at == root || parent[at] >= vertices) return step;
    at = parent[at];
  }
  return UINT32_MAX;
}

/* mesh-graph.h: the climb's visits, a cycle's return, a level's height and a mark */
static void climb_return(void *consumer, uint32_t at, uint32_t step) {
  uint32_t *found = consumer;
  if (step && at == found[0] && !found[1]) found[1] = step;
}
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from) {
  uint32_t found[2] = {from, 0};
  mesh_graph_climb(parent, vertices, UINT32_MAX, from, climb_return, found);
  return found[1];
}
static void climb_height(void *consumer, uint32_t at, uint32_t step) {
  uint32_t *height = consumer;
  if (step && height[at] < step) height[at] = step;
}
void mesh_graph_tree_levels(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t *depth, uint32_t *height) {
  for (uint32_t v = 0; v < vertices; v++) height[v] = 0;
  for (uint32_t v = 0; v < vertices; v++) {
    const uint32_t steps = mesh_graph_climb(parent, vertices, root, v, climb_height, height);
    depth[v] = steps == UINT32_MAX ? vertices : steps;
  }
}
static void climb_mark(void *consumer, uint32_t at, uint32_t step) { (void)step; ((unsigned char *)consumer)[at] = 1; }
void mesh_graph_ancestors(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from, unsigned char *marked) {
  mesh_graph_climb(parent, vertices, root, from, climb_mark, marked);
}

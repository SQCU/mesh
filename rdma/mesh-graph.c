#include "mesh-graph.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

/* mesh-graph.h: the schedule's moves, their follows each way (CSR), and every unit's time (an arc unit's end, an
   event's moment; NAN until known) */
struct schedule {
  const struct mesh_graph_move *move; const struct mesh_graph_follow *follow;
  size_t *base, *in_first, *out_first; uint32_t *in, *out, *next, *height, *depth;
  double *at, *began, *done;
};
static uint32_t schedule_after(const void *context, size_t k) {
  const struct schedule *s = context;
  return s->follow[s->out[k]].after;
}
/* move k's next unit's ready time, where every unit it waits for has its time */
static int schedule_ready(const struct schedule *s, uint32_t k, double *ready) {
  double t = s->move[k].release;
  for (size_t i = s->in_first[k]; i < s->in_first[k + 1]; i++) {
    const struct mesh_graph_follow *f = s->follow + s->in[i];
    const double x = f->wait == MESH_GRAPH_STREAMED ? s->at[s->base[f->before] + f->offset + s->next[k]]
                   : f->wait == MESH_GRAPH_COMPLETE ? s->done[f->before] : s->began[f->before];
    if (isnan(x)) return 0;
    if (x > t) t = x;
  }
  *ready = t;
  return 1;
}
/* the events on `stack` given every unit time they can have, and the events below each that moved queued in turn */
static void schedule_events(struct schedule *s, uint32_t arcs, uint32_t *stack, size_t pending, unsigned char *queued) {
  while (pending) {
    const uint32_t c = stack[--pending];
    const struct mesh_graph_move *m = s->move + c;
    int moved = 0;
    double ready;
    queued[c] = 0;
    while (s->next[c] < m->units && schedule_ready(s, c, &ready)) {
      if (!s->next[c]) s->began[c] = ready;
      s->at[s->base[c] + s->next[c]++] = ready;
      moved = 1;
    }
    if (!moved) continue;
    if (s->next[c] == m->units) s->done[c] = s->at[s->base[c] + m->units - 1];
    for (size_t i = s->out_first[c]; i < s->out_first[c + 1]; i++) {
      const uint32_t child = s->follow[s->out[i]].after;
      if (s->move[child].arc == arcs && !queued[child] && s->next[child] < s->move[child].units) {
        queued[child] = 1;
        stack[pending++] = child;
      }
    }
  }
}
/* the arc's move to carry next: the higher, then the one carried last, then the lower production, then the lower */
static int schedule_better(const struct schedule *s, uint32_t k, uint32_t than, uint32_t last) {
  if (s->height[k] != s->height[than]) return s->height[k] > s->height[than];
  if ((k == last) != (than == last)) return k == last;
  if (s->move[k].production != s->move[than].production) return s->move[k].production < s->move[than].production;
  return k < than;
}
struct schedule_place { double at; uint32_t depth, k; };
static int schedule_place_order(const void *left, const void *right) {
  const struct schedule_place *a = left, *b = right;
  if (a->at != b->at) return a->at < b->at ? -1 : 1;
  if (a->depth != b->depth) return a->depth < b->depth ? -1 : 1;
  return (a->k > b->k) - (a->k < b->k);
}
/* mesh-graph.h */
int mesh_graph_schedule(const struct mesh_graph_move *move, uint32_t moves, uint32_t arcs,
                        const struct mesh_graph_follow *follow, size_t follows, uint32_t *order, double *finish) {
  if ((moves && (!move || !order)) || (follows && !follow)) return EINVAL;
  size_t units = 0, carried = 0;
  for (uint32_t k = 0; k < moves; k++) {
    const struct mesh_graph_move *m = move + k;
    if (m->arc > arcs || !m->units || !isfinite(m->release) || m->release < 0) return EINVAL;
    if (m->arc < arcs && (!(m->unit_seconds > 0) || !isfinite(m->unit_seconds) || !(m->last_seconds > 0) ||
                          !isfinite(m->last_seconds))) return EINVAL;
    units += m->units;
    if (m->arc < arcs) carried += m->units;
  }
  for (size_t e = 0; e < follows; e++) {
    const struct mesh_graph_follow *f = follow + e;
    if (f->before >= moves || f->after >= moves || f->before == f->after || f->wait > MESH_GRAPH_COMPLETE ||
        (f->wait == MESH_GRAPH_STREAMED && (uint64_t)f->offset + move[f->after].units > move[f->before].units))
      return EINVAL;
  }
  struct schedule s = {.move = move, .follow = follow,
    .base = malloc(((size_t)moves + 1) * sizeof *s.base), .in_first = calloc((size_t)moves + 2, sizeof *s.in_first),
    .out_first = calloc((size_t)moves + 2, sizeof *s.out_first), .in = malloc((follows + 1) * sizeof *s.in),
    .out = malloc((follows + 1) * sizeof *s.out), .next = calloc((size_t)moves + 1, sizeof *s.next),
    .height = calloc((size_t)moves + 1, sizeof *s.height), .depth = calloc((size_t)moves + 1, sizeof *s.depth),
    .at = malloc((units + 1) * sizeof *s.at), .began = malloc(((size_t)moves + 1) * sizeof *s.began),
    .done = malloc(((size_t)moves + 1) * sizeof *s.done)};
  uint32_t *indegree = malloc(((size_t)moves + 1) * sizeof *indegree), *ordinal = malloc(((size_t)moves + 1) * sizeof *ordinal),
           *placed = malloc(((size_t)moves + 1) * sizeof *placed), *stack = malloc(((size_t)moves + 1) * sizeof *stack),
           *on = malloc(((size_t)moves + 1) * sizeof *on), *carrying = calloc((size_t)arcs + 1, sizeof *carrying),
           *last = malloc(((size_t)arcs + 1) * sizeof *last);
  size_t *arc_first = calloc((size_t)arcs + 2, sizeof *arc_first);
  double *free_at = calloc((size_t)arcs + 1, sizeof *free_at), *ready = malloc(((size_t)moves + 1) * sizeof *ready);
  unsigned char *queued = calloc((size_t)moves + 1, 1), *known = calloc((size_t)moves + 1, 1);
  struct schedule_place *place = malloc(((size_t)moves + 1) * sizeof *place);
  int status = 0;
  if (!s.base || !s.in_first || !s.out_first || !s.in || !s.out || !s.next || !s.height || !s.depth || !s.at || !s.began ||
      !s.done || !indegree || !ordinal || !placed || !stack || !on || !carrying || !last || !arc_first || !free_at ||
      !ready || !queued || !known || !place) { status = ENOMEM; goto done; }
  for (uint32_t k = 0; k < moves; k++) s.base[k] = k ? s.base[k - 1] + move[k - 1].units : 0;
  for (size_t e = 0; e < follows; e++) { s.in_first[follow[e].after + 2]++; s.out_first[follow[e].before + 2]++; }
  for (uint32_t k = 0; k < moves; k++) { s.in_first[k + 2] += s.in_first[k + 1]; s.out_first[k + 2] += s.out_first[k + 1]; }
  for (size_t e = 0; e < follows; e++) {
    s.in[s.in_first[follow[e].after + 1]++] = (uint32_t)e;
    s.out[s.out_first[follow[e].before + 1]++] = (uint32_t)e;
  }
  if ((status = mesh_graph_order(moves, s.out_first, schedule_after, &s, indegree, ordinal))) goto done;
  for (uint32_t k = 0; k < moves; k++) placed[ordinal[k]] = k;
  for (uint32_t p = moves; p-- > 0;) {
    const uint32_t k = placed[p];
    uint32_t below = 0;
    for (size_t i = s.out_first[k]; i < s.out_first[k + 1]; i++)
      if (s.height[follow[s.out[i]].after] > below) below = s.height[follow[s.out[i]].after];
    s.height[k] = below + (move[k].arc < arcs);
  }
  for (uint32_t p = 0; p < moves; p++) {
    const uint32_t k = placed[p];
    for (size_t i = s.out_first[k]; i < s.out_first[k + 1]; i++) {
      const uint32_t child = follow[s.out[i]].after;
      if (s.depth[child] < s.depth[k] + 1) s.depth[child] = s.depth[k] + 1;
    }
  }
  for (size_t u = 0; u < units; u++) s.at[u] = NAN;
  for (uint32_t k = 0; k < moves; k++) { s.began[k] = s.done[k] = NAN; arc_first[move[k].arc + 1]++; }
  for (uint32_t a = 0; a < arcs; a++) { arc_first[a + 1] += arc_first[a]; last[a] = UINT32_MAX; }
  for (uint32_t k = 0; k < moves; k++) if (move[k].arc < arcs) on[arc_first[move[k].arc] + carrying[move[k].arc]++] = k;
  size_t pending = 0;
  for (uint32_t p = 0; p < moves; p++) if (move[placed[p]].arc == arcs) { queued[placed[p]] = 1; stack[pending++] = placed[p]; }
  schedule_events(&s, arcs, stack, pending, queued);
  for (; carried; carried--) {
    double best = INFINITY;
    uint32_t pick = UINT32_MAX, arc = arcs;
    for (uint32_t a = 0; a < arcs; a++) {
      double earliest = INFINITY;
      for (uint32_t j = 0; j < carrying[a]; j++) {
        const uint32_t k = on[arc_first[a] + j];
        known[k] = (unsigned char)schedule_ready(&s, k, ready + k);
        if (known[k] && ready[k] < earliest) earliest = ready[k];
      }
      if (!isfinite(earliest)) continue;
      const double t = free_at[a] > earliest ? free_at[a] : earliest;
      if (t >= best) continue;
      uint32_t chosen = UINT32_MAX;
      for (uint32_t j = 0; j < carrying[a]; j++) {
        const uint32_t k = on[arc_first[a] + j];
        if (known[k] && ready[k] <= t && (chosen == UINT32_MAX || schedule_better(&s, k, chosen, last[a]))) chosen = k;
      }
      best = t; pick = chosen; arc = a;
    }
    if (pick == UINT32_MAX) { status = EDEADLK; goto done; }
    const struct mesh_graph_move *m = move + pick;
    const uint32_t u = s.next[pick];
    const double end = best + (u + 1 == m->units ? m->last_seconds : m->unit_seconds);
    if (!u) s.began[pick] = best;
    s.at[s.base[pick] + u] = end;
    free_at[arc] = end; last[arc] = pick;
    if (++s.next[pick] == m->units) {
      s.done[pick] = end;
      for (uint32_t j = 0; j < carrying[arc]; j++)
        if (on[arc_first[arc] + j] == pick) { on[arc_first[arc] + j] = on[arc_first[arc] + --carrying[arc]]; break; }
    }
    pending = 0;
    for (size_t i = s.out_first[pick]; i < s.out_first[pick + 1]; i++) {
      const uint32_t child = follow[s.out[i]].after;
      if (move[child].arc == arcs && !queued[child] && s.next[child] < move[child].units) { queued[child] = 1; stack[pending++] = child; }
    }
    schedule_events(&s, arcs, stack, pending, queued);
  }
  double latest = 0;
  for (uint32_t k = 0; k < moves; k++) {
    if (s.next[k] != move[k].units) { status = EDEADLK; goto done; }
    if (s.done[k] > latest) latest = s.done[k];
    place[k] = (struct schedule_place){s.began[k], s.depth[k], k};
  }
  qsort(place, moves, sizeof *place, schedule_place_order);
  for (uint32_t p = 0; p < moves; p++) order[place[p].k] = p;
  if (finish) *finish = latest;
done:
  free(s.base); free(s.in_first); free(s.out_first); free(s.in); free(s.out); free(s.next); free(s.height); free(s.depth);
  free(s.at); free(s.began); free(s.done); free(indegree); free(ordinal); free(placed); free(stack); free(on); free(carrying);
  free(last); free(arc_first); free(free_at); free(ready); free(queued); free(known); free(place);
  return status;
}

/* mesh-graph.h mesh_graph_chain */
int mesh_graph_chain(struct mesh_graph_chain *cursor, uint64_t q, uint64_t (*follow)(const void *context, uint64_t node),
                     uint64_t (*items)(const void *context, uint64_t node),
                     uint64_t (*weight)(const void *context, uint64_t node, uint64_t item), const void *context,
                     void (*leave)(void *consumer, uint64_t node, uint64_t left), void *consumer) {
  for (;;) {
    for (const uint64_t n = cursor->node ? items(context, cursor->node) : 0; cursor->item < n; cursor->item++) {
      const uint64_t w = weight(context, cursor->node, cursor->item);
      if (q - cursor->before < w) return 1;
      cursor->before += w;
    }
    const uint64_t next = follow(context, cursor->node);
    if (!next) return 0;
    if (cursor->node && leave) leave(consumer, cursor->node, ++cursor->left);
    cursor->node = next; cursor->item = 0;
  }
}

/* mesh-graph.h mesh_graph_row */
int mesh_graph_row(const uint32_t *first, const uint32_t *head, uint32_t v, int (*visit)(void *consumer, uint32_t head),
                   void *consumer, uint32_t *count, uint32_t *last) {
  if (count) *count = first[v + 1] - first[v];
  if (last && first[v + 1] > first[v]) *last = head[first[v + 1] - 1];
  for (uint32_t k = first[v]; visit && k < first[v + 1]; k++) {
    const int stop = visit(consumer, head[k]);
    if (stop) return stop;
  }
  return 0;
}

int64_t mesh_graph_forward(const uint32_t *first, const uint32_t *head, uint32_t v, int first_edge,
                           int (*holds)(const void *consumer, uint32_t head), const void *consumer) {
  if (first[v] == first[v + 1]) return -1;
  const uint32_t r = head[first_edge ? first[v] : first[v + 1] - 1];
  return !holds || holds(consumer, r) ? (int64_t)r : -1;
}

int64_t mesh_graph_frontier(const uint32_t *first, const uint32_t *head, uint32_t v,
                            uint64_t (*key)(const void *consumer, uint32_t head),
                            uint64_t (*rank)(const void *consumer, uint32_t head), const void *consumer, uint32_t most,
                            uint32_t *kept, uint64_t *keys) {
  uint32_t n = 0;
  for (uint32_t k = first[v]; k < first[v + 1]; k++) {
    const uint32_t r = head[k];
    const uint64_t at = key(consumer, r);
    if (!at) return -1;
    uint32_t g = 0;
    while (g < n && keys[g] != at) g++;
    if (g == n) { if (n == most) return -1; keys[n] = at; kept[n++] = r; }
    else if (rank(consumer, r) > rank(consumer, kept[g])) kept[g] = r;
  }
  return n;
}

int mesh_graph_cut_at(uint64_t delta, uint64_t a, uint64_t b, uint64_t (*weight)(const void *context, int side, uint64_t k),
                      const void *context, uint64_t *at) {
  uint64_t k = 0, bytes = 0;
  for (; bytes < delta && k < a; k++) bytes += weight(context, 0, k);
  *at = k;
  int same = bytes == delta && k + b <= a;
  for (uint64_t u = 0; same && u < b; u++) same = weight(context, 1, u) == weight(context, 0, k + u);
  return same;
}

void mesh_graph_through(const uint64_t *a, size_t na, const uint64_t *b, size_t nb, uint64_t *through) {
  for (size_t j = 0; j < nb; j++) {
    size_t lo = 0, hi = na;
    while (lo < hi) { const size_t at = lo + (hi - lo) / 2; if (a[at] <= b[j]) lo = at + 1; else hi = at; }
    through[j] = lo;
  }
}

int mesh_graph_cells_through(const uint64_t *a_first, const uint64_t *a_length, size_t na, const uint64_t *b_first,
                             const uint64_t *b_length, size_t nb, int64_t delta, uint64_t cell, uint64_t *through) {
  const uint64_t theirs = mesh_graph_cells(a_length, na, cell, NULL, NULL);
  uint64_t *starts = malloc((theirs + 1) * sizeof *starts), at = 0, j = 0;
  int ordered = starts != NULL;
  for (size_t r = 0; starts && r < na; r++)
    for (uint64_t k = 0; k < (a_length[r] + cell - 1) / cell; k++, at++) {
      starts[at] = a_first[r] + k * cell;
      if (at && starts[at] < starts[at - 1]) ordered = 0;
    }
  for (size_t r = 0; r < nb; r++)
    for (uint64_t k = 0; k < (b_length[r] + cell - 1) / cell; k++, j++) {
      const uint64_t end = (k + 1) * cell < b_length[r] ? (k + 1) * cell : b_length[r];
      const int64_t last = delta + (int64_t)(b_first[r] + end) - 1;
      if (!ordered) through[j] = theirs;
      else if (last < 0) through[j] = 0;
      else { const uint64_t value = (uint64_t)last; mesh_graph_through(starts, theirs, &value, 1, through + j); }
    }
  free(starts);
  return ordered ? 0 : -1;
}

uint64_t mesh_graph_cells(const uint64_t *length, size_t n, uint64_t cell, uint64_t *segment, uint64_t *place) {
  uint64_t q = 0;
  for (size_t i = 0; i < n; i++) {
    const uint64_t cells = (length[i] + cell - 1) / cell;
    for (uint64_t k = 0; segment && k < cells; k++) { segment[q + k] = i; place[q + k] = k; }
    q += cells;
  }
  return q;
}

void mesh_graph_span_reach(const struct mesh_graph_span *a, size_t n, uint64_t *reach) {
  for (size_t i = 0; i < n; i++) reach[i] = i && reach[i - 1] > a[i].end ? reach[i - 1] : a[i].end;
}
void mesh_graph_meets(const struct mesh_graph_span *a, const uint64_t *reach, size_t n, struct mesh_graph_span b,
                      void (*meet)(void *consumer, size_t i), void *consumer) {
  size_t low = 0, high = n;
  while (low < high) { const size_t mid = low + (high - low) / 2; if (a[mid].begin < b.end) low = mid + 1; else high = mid; }
  for (size_t i = low; i-- > 0 && reach[i] > b.begin;) if (a[i].end > b.begin) meet(consumer, i);
}

/* mesh-graph.h mesh_graph_classes */
struct classes_span { struct mesh_graph_span span; uint32_t k; };
static int classes_span_order(const void *left, const void *right) {
  const struct classes_span *a = left, *b = right;
  return (a->span.begin > b->span.begin) - (a->span.begin < b->span.begin);
}
struct classes_meet { const struct classes_span *spans; const uint32_t *part; uint32_t at; int crossed; };
static void classes_met(void *consumer, size_t i) {
  struct classes_meet *m = consumer;
  if (m->part[m->spans[i].k] != m->part[m->at]) m->crossed = 1;
}
static uint32_t classes_head(const void *context, size_t k) { return ((const uint32_t *)context)[k]; }
static int classes_from_order(const void *left, const void *right) {
  const uint64_t a = *(const uint64_t *)left, b = *(const uint64_t *)right;
  return (a > b) - (a < b);
}
int mesh_graph_classes(uint32_t vertices, const size_t *first, const uint32_t *head, const uint32_t *class_of,
                       uint32_t classes, const struct mesh_graph_span *span, const uint32_t *part, uint32_t *place,
                       size_t **in_first, uint32_t **in_from) {
  size_t n = 0;
  for (uint32_t v = 0; v < vertices; v++)
    if (class_of[v] != UINT32_MAX)
      for (size_t k = first[v]; k < first[v + 1]; k++) n += class_of[head[k]] != UINT32_MAX && class_of[head[k]] != class_of[v];
  /* each edge as (to << 32 | from), sorted and each kept once: the in-edges' rows */
  uint64_t *pairs = malloc((n + 1) * sizeof *pairs);
  *in_first = calloc((size_t)classes + 2, sizeof **in_first); *in_from = malloc((n + 1) * sizeof **in_from);
  struct classes_span *sorted = malloc(((size_t)classes + 1) * sizeof *sorted);
  struct mesh_graph_span *spans = malloc(((size_t)classes + 1) * sizeof *spans);
  uint64_t *reach = malloc(((size_t)classes + 1) * sizeof *reach);
  size_t *rows = calloc((size_t)classes + 2, sizeof *rows), *at = calloc((size_t)classes + 1, sizeof *at), edges = 0;
  uint32_t *heads = malloc((n + 1) * sizeof *heads), *indegree = malloc(((size_t)classes + 1) * sizeof *indegree);
  int result = !pairs || !*in_first || !*in_from || !sorted || !spans || !reach || !rows || !at || !heads || !indegree ? ENOMEM : 0;
  for (uint32_t v = 0; !result && v < vertices; v++)
    if (class_of[v] != UINT32_MAX)
      for (size_t k = first[v]; k < first[v + 1]; k++) {
        const uint32_t a = class_of[head[k]];
        if (a != UINT32_MAX && a != class_of[v]) pairs[edges++] = (uint64_t)class_of[v] << 32 | a;
      }
  if (!result) {
    qsort(pairs, edges, sizeof *pairs, classes_from_order);
    size_t kept = 0;
    for (size_t k = 0; k < edges; k++) if (!kept || pairs[k] != pairs[kept - 1]) pairs[kept++] = pairs[k];
    edges = kept;
    for (size_t k = 0; k < edges; k++) { (*in_first)[(pairs[k] >> 32) + 1]++; (*in_from)[k] = (uint32_t)pairs[k]; }
    for (uint32_t k = 0; k < classes; k++) (*in_first)[k + 1] += (*in_first)[k];
  }
  if (!result) {
    for (uint32_t k = 0; k < classes; k++) sorted[k] = (struct classes_span){span[k], k};
    qsort(sorted, classes, sizeof *sorted, classes_span_order);
    for (uint32_t k = 0; k < classes; k++) spans[k] = sorted[k].span;
    mesh_graph_span_reach(spans, classes, reach);
    struct classes_meet meet = {sorted, part, 0, 0};
    for (uint32_t k = 0; !meet.crossed && k < classes; k++) {
      meet.at = sorted[k].k;
      mesh_graph_meets(spans, reach, classes, spans[k], classes_met, &meet);
    }
    if (meet.crossed) result = 1;
  }
  if (!result) {
    for (size_t k = 0; k < edges; k++) rows[(uint32_t)pairs[k] + 1]++;
    for (uint32_t k = 0; k < classes; k++) rows[k + 1] += rows[k];
    for (size_t k = 0; k < edges; k++) heads[rows[(uint32_t)pairs[k]] + at[(uint32_t)pairs[k]]++] = (uint32_t)(pairs[k] >> 32);
    if (mesh_graph_order(classes, rows, classes_head, heads, indegree, place)) result = 1;
  }
  free(pairs); free(sorted); free(spans); free(reach); free(rows); free(at); free(heads); free(indegree);
  return result;
}

int mesh_graph_join(uint32_t parts, const size_t *base, const uint32_t *const *first, const uint32_t *const *head,
                    const unsigned char *marked, size_t **rows, uint32_t **heads, size_t *outside) {
  const size_t vertices = base[parts];
  size_t n = 0;
  *outside = 0;
  *rows = calloc(vertices + 2, sizeof **rows);
  for (uint32_t p = 0; *rows && p < parts; p++)
    for (size_t j = 0; first[p] && j < base[p + 1] - base[p]; j++)
      if (marked[base[p] + j])
        for (uint32_t k = first[p][j]; k < first[p][j + 1]; k++) {
          if (marked[base[p] + head[p][k]]) { (*rows)[base[p] + j + 1]++; n++; }
          else (*outside)++;
        }
  *heads = malloc((n + 1) * sizeof **heads);
  if (!*rows || !*heads) return ENOMEM;
  for (size_t v = 0; v < vertices; v++) (*rows)[v + 1] += (*rows)[v];
  for (uint32_t p = 0; p < parts; p++)
    for (size_t j = 0; first[p] && j < base[p + 1] - base[p]; j++) {
      size_t at = (*rows)[base[p] + j];
      if (marked[base[p] + j])
        for (uint32_t k = first[p][j]; k < first[p][j + 1]; k++)
          if (marked[base[p] + head[p][k]]) (*heads)[at++] = (uint32_t)(base[p] + head[p][k]);
    }
  return 0;
}

/* mesh-graph.h mesh_graph_depth */
static uint32_t depth_head(const void *context, size_t k) { return ((const uint32_t *)context)[k]; }
int mesh_graph_depth(uint32_t vertices, const size_t *first, const uint32_t *head, const unsigned char *marked, uint32_t *depth) {
  size_t *out = calloc((size_t)vertices + 2, sizeof *out), *at = calloc((size_t)vertices + 1, sizeof *at);
  uint32_t *heads = malloc((first[vertices] + 1) * sizeof *heads), *indegree = malloc(((size_t)vertices + 1) * sizeof *indegree);
  uint32_t *ordinal = malloc(((size_t)vertices + 1) * sizeof *ordinal), *order = malloc(((size_t)vertices + 1) * sizeof *order);
  int result = !out || !at || !heads || !indegree || !ordinal || !order ? ENOMEM : 0;
  for (uint32_t v = 0; !result && v < vertices; v++) for (size_t k = first[v]; k < first[v + 1]; k++) out[head[k] + 1]++;
  for (uint32_t v = 0; !result && v < vertices; v++) out[v + 1] += out[v];
  for (uint32_t v = 0; !result && v < vertices; v++) for (size_t k = first[v]; k < first[v + 1]; k++) heads[out[head[k]] + at[head[k]]++] = v;
  if (!result) result = mesh_graph_order(vertices, out, depth_head, heads, indegree, ordinal);
  for (uint32_t v = 0; !result && v < vertices; v++) order[ordinal[v]] = v;
  for (uint32_t p = 0; !result && p < vertices; p++) {
    const uint32_t v = order[p];
    depth[v] = 0;
    for (size_t k = first[v]; k < first[v + 1]; k++)
      if (marked[head[k]] && depth[head[k]] + 1 > depth[v]) depth[v] = depth[head[k]] + 1;
  }
  free(out); free(at); free(heads); free(indegree); free(ordinal); free(order);
  return result;
}

/* mesh-graph.h mesh_graph_rows */
void mesh_graph_rows(uint32_t keys, size_t items, uint32_t (*key)(const void *context, size_t k), const void *context,
                     size_t *first, size_t *order) {
  for (uint32_t v = 0; v <= keys; v++) first[v] = 0;
  for (size_t k = 0; k < items; k++) first[key(context, k) + 1]++;
  for (uint32_t v = 0; v < keys; v++) first[v + 1] += first[v];
  for (size_t k = 0; k < items; k++) order[first[key(context, k)]++] = k;
  for (uint32_t v = keys; v > 0; v--) first[v] = first[v - 1];
  first[0] = 0;
}

/* mesh-graph.h mesh_graph_keyed_find */
uint32_t mesh_graph_keyed_find(const uint32_t *head, const uint32_t *next, uint32_t key,
                               int (*equal)(const void *context, uint32_t k), const void *context) {
  for (uint32_t k = head[key]; k != UINT32_MAX; k = next[k]) if (equal(context, k)) return k;
  return UINT32_MAX;
}

/* mesh-graph.h mesh_graph_keyed_add */
void mesh_graph_keyed_add(uint32_t *head, uint32_t *next, uint32_t key, uint32_t k) {
  next[k] = head[key];
  head[key] = k;
}

/* mesh-graph.h mesh_graph_exp, mesh_graph_log (moved from mesh-collective.c, unchanged) */
double mesh_graph_exp(double x){
  if(x<-745)return 0;
  if(x>709)return INFINITY;
  const double ln2_high=0.693147180369123816490,ln2_low=1.90821492927058770002e-10;
  const double k=floor(x/0.693147180559945309417+0.5),r=(x-k*ln2_high)-k*ln2_low;
  double term=1,sum=1;
  for(int i=1;i<=20;i++){term=term*r/i;sum+=term;}
  return ldexp(sum,(int)k);
}
double mesh_graph_log(double x){
  if(!(x>0))return -INFINITY;
  int e;
  const double m=frexp(x,&e),s=(m-1)/(m+1),s2=s*s;
  double term=s,sum=0;
  for(int i=1;i<=61;i+=2){sum+=term/i;term*=s2;}
  return 2*sum+e*0.693147180559945309417;
}

/* a solved tree equal to the one just found, among its demand's alone (mesh_graph_keyed_find) */
struct flow_same { const uint32_t *parent,*tree; uint32_t n; };
static int flow_same(const void *context,uint32_t k){
  const struct flow_same *s=context;
  return !memcmp(s->parent+(size_t)k*s->n,s->tree,s->n*sizeof *s->tree);
}

/* mesh-graph.h mesh_graph_flow (mesh-collective.c mesh_solve's rounds, moved) */
uint32_t mesh_graph_flow(uint32_t n, uint32_t E, const uint32_t *from, const uint32_t *to, const double *cap,
                         uint32_t count, const uint32_t *root, const double *upward, const double *downward,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share){
  double *load=calloc(2*(size_t)E+1,sizeof *load),*add=calloc(2*(size_t)E+1,sizeof *add),
         *len=calloc(2*(size_t)E+1,sizeof *len),*dist=calloc((size_t)n+1,sizeof *dist);
  uint32_t *up=calloc((size_t)n+1,sizeof *up),*upl=calloc((size_t)n+1,sizeof *upl),*tree=calloc((size_t)n+1,sizeof *tree),
           *treel=calloc((size_t)n+1,sizeof *treel),*hits=calloc((size_t)room+1,sizeof *hits),made=0;
  uint32_t *head=malloc(((size_t)count+1)*sizeof *head),*next=malloc(((size_t)room+1)*sizeof *next);
  unsigned char *done=calloc((size_t)n+1,1);
  int failed=!load || !add || !len || !dist || !up || !upl || !tree || !treel || !hits || !done || !head || !next || !(eps>0);
  for(uint32_t d=0;!failed && d<count;d++)head[d]=UINT32_MAX;
  double least=0,total=0;
  for(uint32_t e=0;!failed && e<E;e++)if(!least || cap[e]<least)least=cap[e];
  for(uint32_t d=0;!failed && d<count;d++)total+=upward[d]+downward[d];
  const double width=total>0 && least>0?total/rounds/least:1;
  for(uint32_t r=0;!failed && r<rounds;r++){
    double most=0;
    for(size_t e=0;e<2*(size_t)E;e++)if(load[e]/cap[e/2]>most)most=load[e]/cap[e/2];
    for(size_t e=0;e<2*(size_t)E;e++){len[e]=mesh_graph_exp(eps*(load[e]/cap[e/2]-most)/width)/cap[e/2];add[e]=0;}
    for(uint32_t d=0;!failed && d<count;d++){
      const double u=upward[d],w=downward[d];
      for(uint32_t v=0;v<n;v++){dist[v]=INFINITY;up[v]=UINT32_MAX;upl[v]=UINT32_MAX;done[v]=0;tree[v]=UINT32_MAX;treel[v]=UINT32_MAX;}
      if(root[d]>=n){failed=1;break;}
      dist[root[d]]=0;up[root[d]]=root[d];
      /* the least-length tree toward the root: a direction's length weighed by the bytes that cross it, up toward the
         root (from the leaf's side) and down from it */
      for(uint32_t step=0;step<n;step++){
        uint32_t at=n;
        for(uint32_t v=0;v<n;v++)if(!done[v] && dist[v]<INFINITY && (at==n || dist[v]<dist[at]))at=v;
        if(at==n)break;
        done[at]=1;
        for(uint32_t e=0;e<E;e++){
          if(from[e]!=at && to[e]!=at)continue;
          const uint32_t v=from[e]==at?to[e]:from[e],toward=from[e]==v?0:1;
          if(done[v])continue;
          const double through=dist[at]+u*len[2*(size_t)e+toward]+w*len[2*(size_t)e+(toward^1)];
          if(through<dist[v]){dist[v]=through;up[v]=at;upl[v]=e;}
        }
      }
      tree[root[d]]=root[d];
      for(uint32_t v=0;v<n && !failed;v++){
        if(v==root[d] || !leaf(context,d,v))continue;
        if(up[v]==UINT32_MAX){failed=1;break;}
        for(uint32_t at=v;tree[at]==UINT32_MAX;at=up[at]){tree[at]=up[at];treel[at]=upl[at];}
      }
      if(failed)break;
      const struct flow_same same={parent,tree,n};
      uint32_t k=mesh_graph_keyed_find(head,next,d,flow_same,&same);
      if(k==UINT32_MAX){
        if(made>=room){failed=1;break;}
        demand[made]=d;
        memcpy(parent+(size_t)made*n,tree,n*sizeof *tree);
        mesh_graph_keyed_add(head,next,d,made);
        k=made++;
      }
      hits[k]++;
      for(uint32_t v=0;v<n;v++){
        if(v==root[d] || tree[v]==UINT32_MAX)continue;
        const uint32_t e=treel[v],toward=from[e]==v?0:1;
        add[2*(size_t)e+toward]+=u/rounds;
        add[2*(size_t)e+(toward^1)]+=w/rounds;
      }
    }
    for(size_t e=0;e<2*(size_t)E;e++)load[e]+=add[e];
  }
  for(uint32_t k=0;!failed && k<made;k++)share[k]=(double)hits[k]/rounds;
  free(load);free(add);free(len);free(dist);free(up);free(upl);free(tree);free(treel);free(hits);free(done);free(head);free(next);
  return failed?UINT32_MAX:made;
}

/* a packed tree (mesh_graph_pack): its root, its flow and its parents; ordered root by root, heaviest first */
struct mesh_packed { uint32_t root; double flow; uint32_t *parent; };
static int mesh_packed_heavier(const void *a,const void *b){
  const struct mesh_packed *x=a,*y=b;
  if(x->root!=y->root)return x->root<y->root?-1:1;
  return x->flow>y->flow?-1:x->flow<y->flow;
}

/* a packed tree equal to the arborescence just chosen, among its root's alone (mesh_graph_keyed_find) */
struct pack_same { const struct mesh_packed *packed; const uint32_t *from; const int32_t *chosen; uint32_t r,n; };
static int pack_same(const void *context,uint32_t t){
  const struct pack_same *s=context;
  uint32_t v=0;
  while(v<s->n && (v==s->r || s->packed[t].parent[v]==s->from[s->chosen[v]]))v++;
  return v==s->n;
}

/* mesh-graph.h mesh_graph_pack (mesh-collective.c mesh_trees_pack's rounds, moved) */
uint32_t mesh_graph_pack(uint32_t n, uint32_t m, const uint32_t *from, const uint32_t *to, const uint32_t *through,
                         const uint32_t *crossing, uint32_t P, const double *capacity,
                         int (*rooted)(const void *context, uint32_t r), const void *context, double eps, uint32_t most,
                         int shallow, uint32_t *root, uint32_t *parent, double *log_weight){
  uint32_t *uses=calloc((size_t)P+1,sizeof *uses);
  /* shallow: root r's edges, those from layer d of its breadth-first layering to layer d + 1 (kept[r * m ..], their
     count kept_count[r]), the arborescence searched over them alone (sub_*) and its choice mapped back */
  uint32_t *kept=shallow?malloc(((size_t)n*m+1)*sizeof *kept):NULL,*kept_count=shallow?calloc((size_t)n+1,sizeof *kept_count):NULL;
  uint32_t *layer=shallow?malloc(((size_t)n+1)*sizeof *layer):NULL;
  uint32_t *sub_from=shallow?malloc(((size_t)m+1)*sizeof *sub_from):NULL,*sub_to=shallow?malloc(((size_t)m+1)*sizeof *sub_to):NULL;
  double *sub_weight=shallow?malloc(((size_t)m+1)*sizeof *sub_weight):NULL;
  double *length=malloc(((size_t)P+1)*sizeof *length),*weight=malloc(((size_t)m+1)*sizeof *weight);
  int32_t *chosen=malloc(((size_t)n+1)*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  uint32_t *head=malloc(((size_t)n+1)*sizeof *head),*next=NULL;
  size_t trees=0,room=0;
  uint32_t written=0;
  if(!uses || !length || !weight || !chosen || !head)goto done;
  if(shallow && (!kept || !kept_count || !layer || !sub_from || !sub_to || !sub_weight))goto done;
  for(uint32_t r=0;shallow && r<n;r++){
    for(uint32_t v=0;v<n;v++)layer[v]=UINT32_MAX;
    layer[r]=0;
    for(uint32_t d=0,grew=1;grew;d++){
      grew=0;
      for(uint32_t e=0;e<m;e++)if(layer[from[e]]==d && layer[to[e]]==UINT32_MAX){layer[to[e]]=d+1;grew=1;}
    }
    for(uint32_t e=0;e<m;e++)if(layer[from[e]]!=UINT32_MAX && layer[to[e]]==layer[from[e]]+1)kept[(size_t)r*m+kept_count[r]++]=e;
  }
  for(uint32_t r=0;r<n;r++)head[r]=UINT32_MAX;
  const double delta=mesh_graph_exp(-mesh_graph_log(P/(1-eps))/eps);
  double D=0;
  for(uint32_t k=0;k<P;k++){length[k]=delta/capacity[k];D+=length[k]*capacity[k];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=rooted(context,r);demand>0 && D<1;){
    for(uint32_t e=0;e<m;e++){weight[e]=0;for(uint32_t i=through[e];i<through[e+1];i++)weight[e]+=length[crossing[i]];}
    if(shallow){
      const uint32_t *own=kept+(size_t)r*m;
      for(uint32_t k=0;k<kept_count[r];k++){sub_from[k]=from[own[k]];sub_to[k]=to[own[k]];sub_weight[k]=weight[own[k]];}
      if(mesh_graph_arborescence(n,r,kept_count[r],sub_from,sub_to,sub_weight,chosen))goto done;
      for(uint32_t v=0;v<n;v++)if(v!=r)chosen[v]=(int32_t)own[chosen[v]];
    }
    else if(mesh_graph_arborescence(n,r,m,from,to,weight,chosen))goto done;
    /* the tree's flow: its demand, at most each directed hop's capacity over the tree's edges crossing it */
    for(uint32_t v=0;v<n;v++)if(v!=r)for(uint32_t i=through[chosen[v]];i<through[chosen[v]+1];i++)uses[crossing[i]]++;
    double f=demand;
    for(uint32_t v=0;v<n;v++)if(v!=r)for(uint32_t i=through[chosen[v]];i<through[chosen[v]+1];i++)
      if(capacity[crossing[i]]/uses[crossing[i]]<f)f=capacity[crossing[i]]/uses[crossing[i]];
    demand-=f;
    const struct pack_same same={packed,from,chosen,r,n};
    const uint32_t found=mesh_graph_keyed_find(head,next,r,pack_same,&same);
    size_t t=found==UINT32_MAX?trees:found;
    if(t==trees){
      if(trees==room){
        struct mesh_packed *grown=realloc(packed,(room?2*room:64)*sizeof *grown);
        if(!grown)goto done;
        packed=grown;
        uint32_t *chained=realloc(next,(room?2*room:64)*sizeof *chained);
        if(!chained)goto done;
        next=chained;room=room?2*room:64;
      }
      if(!(packed[t].parent=malloc(n*sizeof *packed[t].parent)))goto done;
      packed[t].root=r;packed[t].flow=0;
      for(uint32_t v=0;v<n;v++)packed[t].parent[v]=v==r?r:from[chosen[v]];
      mesh_graph_keyed_add(head,next,r,(uint32_t)t);
      trees++;
    }
    packed[t].flow+=f;
    for(uint32_t v=0;v<n;v++)if(v!=r)for(uint32_t i=through[chosen[v]];i<through[chosen[v]+1];i++){
      const uint32_t k=crossing[i];
      if(!uses[k])continue;
      D+=length[k]*eps*f*uses[k];
      length[k]*=1+eps*f*uses[k]/capacity[k];
      uses[k]=0;
    }
  }
  qsort(packed,trees,sizeof *packed,mesh_packed_heavier);
  for(size_t t=0;t<trees;){
    size_t end=t,keep;
    double total=0;
    while(end<trees && packed[end].root==packed[t].root)end++;
    keep=end-t<most?end-t:most;
    for(size_t u=t;u<t+keep;u++)total+=packed[u].flow;
    for(size_t u=t;u<t+keep;u++){
      root[written]=packed[u].root;
      memcpy(parent+(size_t)written*n,packed[u].parent,n*sizeof *parent);
      log_weight[written++]=mesh_graph_log(packed[u].flow/total);
    }
    t=end;
  }
done:
  for(size_t t=0;t<trees;t++)free(packed[t].parent);
  free(packed);free(uses);free(length);free(weight);free(chosen);free(head);free(next);
  free(kept);free(kept_count);free(layer);free(sub_from);free(sub_to);free(sub_weight);
  return written;
}

/* mesh-graph.h mesh_graph_tree_loads */
void mesh_graph_tree_loads(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                           const double *weight, double *load){
  for(uint32_t k=0;k<trees;k++)
    for(uint32_t v=0;v<vertices;v++){
      const uint32_t u=parent[(size_t)k*vertices+v];
      if(v!=root[k] && u!=UINT32_MAX)load[(size_t)u*vertices+v]+=weight[k];
    }
}

/* mesh-graph.h mesh_graph_settle */
size_t mesh_graph_settle(void **items, size_t count, int (*take)(void *consumer, void *item), void *consumer) {
  for (int taken = 1; taken && count;) {
    taken = 0;
    for (size_t k = 0; k < count;) {
      if (take(consumer, items[k])) { items[k] = items[--count]; taken = 1; }
      else k++;
    }
  }
  return count;
}

/* mesh-graph.h mesh_graph_pair_classes. Added by the layout algebra session (moved here from MetalTensorPairQuotient by
   C9). */
void *mesh_graph_pair_classes(const struct mesh_graph_lattice *removed, int64_t first, int64_t end,
                              const struct mesh_graph_terms *terms, void *consumer) {
  const size_t m = removed->axes;
  const int64_t *stride = removed->stride, *extent = removed->extent, origin = removed->origin;
  int64_t last = 0;
  for (size_t k = 0; k < m; k++) last += stride[k] * (extent[k] - 1);
  void *p = terms->position(consumer), *y = terms->sum(consumer, p, terms->constant(consumer, -origin));
  void *digits[m ? m : 1], *inside[m + 3];
  size_t conditions = 0;
  for (size_t k = 0; k < m; k++)
    digits[k] = terms->quotient(consumer, k + 1 < m ? terms->quotient(consumer, y, stride[k + 1], 1) : y, stride[k], 0);
  void *matched = terms->kept(consumer, digits, m);
  if (origin > first) inside[conditions++] = terms->compare(consumer, terms->position(consumer), MESH_GRAPH_AT_LEAST, origin);
  if (origin + last < end - 1)
    inside[conditions++] = terms->compare(consumer, terms->position(consumer), MESH_GRAPH_AT_MOST, origin + last);
  if (stride[0] > 1)
    inside[conditions++] = terms->compare(consumer, terms->quotient(consumer, y, stride[0], 1), MESH_GRAPH_EQUAL, 0);
  for (size_t k = 0; k + 1 < m; k++)
    if (stride[k + 1] > stride[k] * extent[k])
      inside[conditions++] = terms->compare(consumer, terms->quotient(consumer, y, stride[k + 1], 1), MESH_GRAPH_BELOW,
                                            stride[k] * extent[k]);
  if (!conditions) return matched;
  /* C(y) over the axes with contiguous neighbours joined */
  int64_t level_stride[m], level_extent[m];
  size_t depth = 0;
  for (size_t k = 0; k < m; k++) {
    if (depth && stride[k] == level_stride[depth - 1] * level_extent[depth - 1]) level_extent[depth - 1] *= extent[k];
    else { level_stride[depth] = stride[k]; level_extent[depth++] = extent[k]; }
  }
  int zero = level_stride[0] <= 1;
  void *below = zero ? terms->constant(consumer, 0)
                     : terms->minimum(consumer, terms->quotient(consumer, y, level_stride[0], 1), 1);
  int64_t points = 1;
  for (size_t k = 0; k < depth; k++) {
    void *q = terms->quotient(consumer, k + 1 < depth ? terms->quotient(consumer, y, level_stride[k + 1], 1) : y,
                              level_stride[k], 0);
    void *part = terms->scale(consumer, terms->minimum(consumer, q, level_extent[k]), points);
    below = zero ? part : terms->sum(consumer, part, terms->select(consumer,
        terms->compare(consumer, q, MESH_GRAPH_BELOW, level_extent[k]), below, terms->constant(consumer, 0)));
    zero = 0;
    points *= level_extent[k];
  }
  if (origin > first)
    below = terms->select(consumer, terms->compare(consumer, terms->position(consumer), MESH_GRAPH_BELOW, origin),
                          terms->constant(consumer, 0), below);
  void *other = terms->sum(consumer, terms->position(consumer), terms->scale(consumer, below, -1));
  return terms->select(consumer, conditions == 1 ? inside[0] : terms->all(consumer, inside, conditions), matched, other);
}

/* mesh-graph.h mesh_graph_tile_slots. Added by the layout algebra session (moved here from TensorPlacedOnTiles by C9). */
static int tile_least_order(const void *a, const void *b) {
  const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
  return x < y ? -1 : x > y;
}
int mesh_graph_tile_slots(size_t count, size_t width, const int64_t *at, int64_t *placed) {
  if (!width || count % width) return -1;
  const size_t tiles = count / width;
  int64_t *member = malloc((count ? count : 1) * sizeof *member), *least = malloc((tiles ? tiles : 1) * 2 * sizeof *least),
          *rank = malloc((tiles ? tiles : 1) * sizeof *rank);
  int result = member && least && rank ? 0 : -1;
  for (size_t p = 0; !result && p < count; p++) member[p] = -1;
  for (size_t i = 0; !result && i < count; i++) {
    if (at[i] < 0 || (uint64_t)at[i] >= count || member[at[i]] >= 0) result = -1;
    else member[at[i]] = (int64_t)i;
  }
  for (size_t k = 0; !result && k < tiles; k++) { least[2 * k] = (int64_t)count; least[2 * k + 1] = (int64_t)k; }
  for (size_t i = 0; !result && i < count; i++)
    if (at[i] < least[2 * (i / width)]) least[2 * (i / width)] = at[i];
  if (!result) qsort(least, tiles, 2 * sizeof *least, tile_least_order);
  for (size_t r = 0; !result && r < tiles; r++) rank[least[2 * r + 1]] = (int64_t)r;
  for (size_t p = 0; !result && p < count; p++)
    placed[p] = rank[(size_t)member[p] / width] * (int64_t)width + member[p] % (int64_t)width;
  free(member); free(least); free(rank);
  return result;
}

/* mesh-graph.h */
static int access_by_item(void *context, const void *left, const void *right) {
  const struct mesh_graph_access *a = context;
  const size_t x = *(const size_t *)left, y = *(const size_t *)right;
  if (a[x].item != a[y].item) return a[x].item < a[y].item ? -1 : 1;
  return (x > y) - (x < y);
}
static int dependence_order(const void *left, const void *right) {
  const struct mesh_graph_dependence *a = left, *b = right;
  if (a->to != b->to) return a->to < b->to ? -1 : 1;
  if (a->from != b->from) return a->from < b->from ? -1 : 1;
  if (a->kind != b->kind) return a->kind < b->kind ? -1 : 1;
  if (a->item != b->item) return a->item < b->item ? -1 : 1;
  return (a->first > b->first) - (a->first < b->first);
}
/* room for `need` byte runs in each of the two run lists the sweep alternates between */
struct byte_run { uint64_t first, end; };
static int byte_runs_room(struct byte_run **open, struct byte_run **next, size_t *room, size_t need) {
  if (need <= *room) return 0;
  size_t grown = *room ? *room : 8;
  while (grown < need) grown *= 2;
  struct byte_run *a = realloc(*open, grown * sizeof *a);
  if (!a) return ENOMEM;
  *open = a;
  struct byte_run *b = realloc(*next, grown * sizeof *b);
  if (!b) return ENOMEM;
  *next = b; *room = grown;
  return 0;
}
size_t mesh_graph_dependences(size_t n, const struct mesh_graph_access *access, struct mesh_graph_dependence *out,
                              size_t capacity) {
  size_t *by = malloc((n + 1) * sizeof *by), room = 0, count = 0, found_room = 0;
  struct byte_run *open = NULL, *next = NULL;
  struct mesh_graph_dependence *found = NULL;
  int failed = !by;
  for (size_t k = 0; !failed && k < n; k++) by[k] = k;
  if (!failed) qsort_r(by, n, sizeof *by, (void *)access, access_by_item);
  /* each item's accesses in the declared order: for access j, the earlier ones from the latest back, over the bytes
     of j that no write met since */
  for (size_t g = 0; !failed && g < n;) {
    size_t h = g;
    while (h < n && access[by[h]].item == access[by[g]].item) h++;
    for (size_t x = g + 1; !failed && x < h; x++) {
      const struct mesh_graph_access *j = access + by[x];
      if (!j->bytes || byte_runs_room(&open, &next, &room, 2)) { failed = !!j->bytes; continue; }
      size_t opens = 1;
      open[0] = (struct byte_run){j->first, j->first + j->bytes};
      for (size_t y = x; y-- > g && opens && !failed;) {
        const struct mesh_graph_access *i = access + by[y];
        const uint64_t lo = i->first, hi = i->first + i->bytes;
        const uint32_t kind = i->mode & MESH_GRAPH_WRITE ? (j->mode & MESH_GRAPH_READ ? MESH_GRAPH_RAW : MESH_GRAPH_WAW)
                              : j->mode & MESH_GRAPH_WRITE ? MESH_GRAPH_WAR : 0;
        if (byte_runs_room(&open, &next, &room, 2 * opens)) { failed = 1; break; }
        size_t kept = 0;
        for (size_t r = 0; r < opens; r++) {
          const uint64_t a = open[r].first > lo ? open[r].first : lo, b = open[r].end < hi ? open[r].end : hi;
          if (a >= b) { next[kept++] = open[r]; continue; }
          if (kind && i->end != j->begin) {
            if (count == found_room) {
              const size_t grown = found_room ? 2 * found_room : 64;
              struct mesh_graph_dependence *more = realloc(found, grown * sizeof *more);
              if (!more) { failed = 1; break; }
              found = more; found_room = grown;
            }
            found[count++] = (struct mesh_graph_dependence){i->end, j->begin, kind, j->item, a, b - a};
          }
          if (i->mode & MESH_GRAPH_WRITE) {
            if (open[r].first < a) next[kept++] = (struct byte_run){open[r].first, a};
            if (b < open[r].end) next[kept++] = (struct byte_run){b, open[r].end};
          } else next[kept++] = open[r];
        }
        struct byte_run *swap = open; open = next; next = swap; opens = kept;
      }
    }
    g = h;
  }
  size_t joined = 0;
  if (!failed && count) {
    qsort(found, count, sizeof *found, dependence_order);
    for (size_t k = 0; k < count; k++) {
      struct mesh_graph_dependence *last = joined ? found + joined - 1 : NULL;
      if (last && last->to == found[k].to && last->from == found[k].from && last->kind == found[k].kind &&
          last->item == found[k].item && last->first + last->bytes == found[k].first)
        last->bytes += found[k].bytes;
      else found[joined++] = found[k];
    }
  }
  for (size_t k = 0; !failed && k < joined && k < capacity; k++) out[k] = found[k];
  free(by); free(open); free(next); free(found);
  return failed ? SIZE_MAX : joined;
}

/* mesh-graph.h */
int mesh_graph_precedence(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                          const void *context, size_t sets, const size_t *source_first, const uint32_t *source,
                          const size_t *target_first, const uint32_t *target, unsigned char *precedes) {
  /* reached[v]: how many of set g's sources reach v; v follows all of them where it equals their count */
  uint32_t *seen = calloc((size_t)vertices + 1, sizeof *seen), *reached = malloc(((size_t)vertices + 1) * sizeof *reached),
           *stack = malloc(((size_t)vertices + 1) * sizeof *stack);
  int result = seen && reached && stack ? 0 : ENOMEM;
  uint32_t stamp = 0;
  for (size_t g = 0; !result && g < sets; g++) {
    for (uint32_t v = 0; v < vertices; v++) reached[v] = 0;
    for (size_t x = source_first[g]; x < source_first[g + 1]; x++) {
      stack[0] = source[x];
      mesh_graph_mark(first, head, context, seen, ++stamp, stack, 1);
      for (uint32_t v = 0; v < vertices; v++) reached[v] += seen[v] == stamp;
    }
    const uint32_t sources = (uint32_t)(source_first[g + 1] - source_first[g]);
    for (size_t h = 0; h < sets; h++) {
      if (h == g) continue;
      int all = 1;
      for (size_t y = target_first[h]; all && y < target_first[h + 1]; y++) all = reached[target[y]] == sources;
      if (all) precedes[(g * sets + h) / 8] |= (unsigned char)(1u << ((g * sets + h) % 8));
    }
  }
  free(seen); free(reached); free(stack);
  return result;
}

/* mesh-graph.h */
static int realize_by_target(const void *left, const void *right) {
  const uint32_t *a = left, *b = right;
  if (a[1] != b[1]) return a[1] < b[1] ? -1 : 1;
  return (a[0] > b[0]) - (a[0] < b[0]);
}
static int realize_wait_order(const void *left, const void *right) {
  const struct mesh_graph_crossing *a = left, *b = right;
  if (a->kind != b->kind) return a->kind < b->kind ? -1 : 1;
  if (a->waiter != b->waiter) return a->waiter < b->waiter ? -1 : 1;
  return (a->producer > b->producer) - (a->producer < b->producer);
}
int mesh_graph_realize(uint32_t n, const unsigned char *host, size_t m, const uint32_t (*edge)[2], uint32_t *part,
                       uint32_t *parts, unsigned char *barrier, unsigned char *fence, struct mesh_graph_crossing *waits,
                       size_t capacity, size_t *count, unsigned char *deferred) {
  for (size_t k = 0; k < m; k++) if (edge[k][0] >= edge[k][1] || edge[k][1] >= n) return EINVAL;
  uint32_t (*sorted)[2] = malloc((m + 1) * sizeof *sorted);
  struct mesh_graph_crossing *found = malloc((m + 1) * sizeof *found);
  uint32_t *waited = calloc((size_t)n + 1, sizeof *waited); /* the part (plus 1) that last waited on host node h */
  if (!sorted || !found || !waited) { free(sorted); free(found); free(waited); return ENOMEM; }
  memcpy(sorted, edge, m * sizeof *sorted);
  qsort(sorted, m, sizeof *sorted, realize_by_target);
  /* parts: walk the order; a GPU node with a host predecessor the current part has not waited on starts a new part */
  uint32_t current = UINT32_MAX;
  int open = 0;
  size_t k = 0;
  for (uint32_t v = 0; v < n; v++) {
    barrier[v] = 0; fence[v] = 0;
    if (host[v]) { part[v] = UINT32_MAX; while (k < m && sorted[k][1] == v) k++; continue; }
    int fresh = !open;
    for (size_t j = k; j < m && sorted[j][1] == v; j++)
      if (host[sorted[j][0]] && open && waited[sorted[j][0]] != current + 1) fresh = 1;
    if (fresh) { current = current == UINT32_MAX ? 0 : current + 1; open = 1; }
    part[v] = current;
    for (; k < m && sorted[k][1] == v; k++)
      if (host[sorted[k][0]]) waited[sorted[k][0]] = current + 1;
  }
  *parts = open ? current + 1 : 0;
  /* barriers: the edges within one part, by target; a barrier before the target of the first one no barrier crosses
     covers every later edge whose span holds that point (interval stabbing by right endpoint) */
  uint32_t last = UINT32_MAX; /* the latest barrier placed, before node `last` */
  for (size_t j = 0; j < m; j++) {
    const uint32_t u = sorted[j][0], v = sorted[j][1];
    if (host[u] || host[v] || part[u] != part[v]) continue;
    if (last != UINT32_MAX && last > u && last <= v) continue;
    barrier[v] = 1; last = v;
  }
  /* waits: an edge into a part from another part or a host node, or into a host node */
  size_t w = 0;
  for (size_t j = 0; j < m; j++) {
    const uint32_t u = sorted[j][0], v = sorted[j][1];
    if (!host[v] && !host[u] && part[u] == part[v]) continue;
    if (!host[u] && !host[v]) fence[u] = 1;
    const uint32_t kind = (host[v] ? 1u : 0u) | (host[u] ? 4u : 0u);
    found[w++] = (struct mesh_graph_crossing){host[v] ? v : part[v], kind, host[u] ? u : part[u]};
  }
  qsort(found, w, sizeof *found, realize_wait_order);
  size_t kept = 0;
  for (size_t j = 0; j < w; j++)
    if (!kept || realize_wait_order(found + kept - 1, found + j)) found[kept++] = found[j];
  for (size_t j = 0; j < kept && j < capacity; j++) waits[j] = found[j];
  *count = kept;
  /* deferral: the rows sort by kind, then waiter; a host operation's rows (kind 4) mark their parts, then the part rows
     (kind 0), whose producers precede their waiters, carry it forward */
  for (uint32_t p = 0; p < *parts; p++) deferred[p] = 0;
  for (size_t j = 0; j < kept; j++)
    if (found[j].kind == 4 && host[found[j].producer] == 1) deferred[found[j].waiter] = 1;
  for (size_t j = 0; j < kept; j++)
    if (found[j].kind == 0 && deferred[found[j].producer]) deferred[found[j].waiter] = 1;
  free(sorted); free(found); free(waited);
  return 0;
}

/* ==== Begin: greedy by size, from TensorFlow Lite (Apache-2.0). The notice below covers this section. ==== */
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* This section, to its end marker, is ported for the mesh library (design/page-maps-2026-11-02.md, "The run's
   solve", 10) from TensorFlow's
   tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc (GreedyBySizeAssignment) and
   internal.cc (CompareBySize), master as fetched 2026-11-02; the algorithm of Y. Pisarchyk and J. Lee, "Efficient
   memory management for deep neural net inference", 2020. Changes from the original:
   - C in place of C++, status codes in place of absl::Status, the records as index and size arrays;
   - the test "tensor allocated_id's usage interval does not intersect the current one's" is apart(context,
     allocated_id, current), the run's exclusion relation over a partial order rather than intervals of one;
   - alignment 1 (the units are pages), so AlignByN is the identity;
   - the "Total size is wrong" consistency check is dropped, and a total past `capacity` is ENOMEM;
   - items already placed (pins) may be given, occupying the line before any item is assigned;
   - stable_sort by CompareBySize is a sort by size descending, then by the given order. */
struct by_size { const uint64_t *size; const size_t *rank; };
/* CompareBySize: the larger first; the given order among equal sizes (stable_sort) */
static int greedy_by_size_order(void *context, const void *left, const void *right) {
  const struct by_size *s = context;
  const size_t a = *(const size_t *)left, b = *(const size_t *)right;
  if (s->size[a] != s->size[b]) return s->size[a] > s->size[b] ? -1 : 1;
  return (s->rank[a] > s->rank[b]) - (s->rank[a] < s->rank[b]);
}

/* mesh-graph.h */
int mesh_graph_greedy_by_size(size_t count, const size_t *items, size_t placed, const size_t *pinned,
                              const uint64_t *size, int (*apart)(const void *context, size_t k, size_t l),
                              const void *context, uint64_t capacity, uint64_t *offset, uint64_t *total) {
  size_t *ordered_records = malloc((count + 1) * sizeof *ordered_records),
         *ordered_allocs = malloc((count + placed + 1) * sizeof *ordered_allocs), *rank = NULL, allocs = 0;
  size_t largest = 0;
  for (size_t i = 0; i < count; i++) if (items[i] + 1 > largest) largest = items[i] + 1;
  for (size_t i = 0; i < placed; i++) if (pinned[i] + 1 > largest) largest = pinned[i] + 1;
  rank = malloc((largest + 1) * sizeof *rank);
  if (!ordered_records || !ordered_allocs || !rank) {
    free(ordered_records); free(ordered_allocs); free(rank);
    return ENOMEM;
  }
  *total = 0;
  /* the line's prior occupants, ordered by offset */
  for (size_t i = 0; i < placed; i++) {
    size_t at = allocs;
    while (at > 0 && offset[ordered_allocs[at - 1]] > offset[pinned[i]]) { ordered_allocs[at] = ordered_allocs[at - 1]; at--; }
    ordered_allocs[at] = pinned[i];
    allocs++;
    if (offset[pinned[i]] + size[pinned[i]] > *total) *total = offset[pinned[i]] + size[pinned[i]];
  }
  for (size_t i = 0; i < count; i++) { ordered_records[i] = items[i]; rank[items[i]] = i; }
  struct by_size order = {size, rank};
  qsort_r(ordered_records, count, sizeof *ordered_records, &order, greedy_by_size_order);
  int result = 0;
  for (size_t r = 0; r < count && !result; r++) {
    const size_t rec = ordered_records[r];
    uint64_t best_diff = UINT64_MAX, best_offset = UINT64_MAX, prev_offset = 0;
    for (size_t a = 0; a < allocs; a++) {
      const size_t allocated_id = ordered_allocs[a];
      if (apart(context, allocated_id, rec)) continue;
      const uint64_t cur_offset = offset[allocated_id];
      if (cur_offset >= prev_offset) {
        const uint64_t diff = cur_offset - prev_offset;
        /* the gap directly left of allocated_id fits, and is the smallest suitable gap so far */
        if (diff >= size[rec] && diff < best_diff) { best_diff = diff; best_offset = prev_offset; }
      }
      if (cur_offset + size[allocated_id] > prev_offset) prev_offset = cur_offset + size[allocated_id];
    }
    /* no suitable gap: after the rightmost item that is not apart from this one */
    if (best_offset == UINT64_MAX) best_offset = prev_offset;
    size_t at = 0;
    while (at < allocs && offset[ordered_allocs[at]] <= best_offset) at++;
    for (size_t y = allocs; y > at; y--) ordered_allocs[y] = ordered_allocs[y - 1];
    ordered_allocs[at] = rec;
    allocs++;
    offset[rec] = best_offset;
    if (best_offset + size[rec] > *total) *total = best_offset + size[rec];
    if (*total > capacity) result = ENOMEM;
  }
  free(ordered_records); free(ordered_allocs); free(rank);
  return result;
}
/* ==== End: greedy by size, from TensorFlow Lite. ==== */

/* mesh-graph.h */
struct placed_item { uint64_t first, size; size_t item; };
struct place_rank { const uint64_t *size, *key; };
/* by key, then by number */
static int place_by_key(void *context, const void *left, const void *right) {
  const struct place_rank *r = context;
  const size_t a = *(const size_t *)left, b = *(const size_t *)right;
  if (r->key[a] != r->key[b]) return r->key[a] < r->key[b] ? -1 : 1;
  return (a > b) - (a < b);
}
/* the first placed run (ascending by first position) that would share a position of [at, at + size) with `item` and
   is not apart from it, or SIZE_MAX: only the runs that can reach `at` are visited (`widest`, the largest placed) */
static size_t place_conflict(const struct placed_item *placed, size_t count, uint64_t widest, uint64_t at, uint64_t size,
                             size_t item, int (*apart)(const void *context, size_t k, size_t l), const void *context) {
  size_t lo = 0, hi = count;
  const uint64_t reach = at > widest ? at - widest : 0;
  while (lo < hi) { const size_t mid = lo + (hi - lo) / 2; if (placed[mid].first < reach) lo = mid + 1; else hi = mid; }
  for (size_t y = lo; y < count && placed[y].first < at + size; y++)
    if (placed[y].first + placed[y].size > at && !apart(context, placed[y].item, item)) return y;
  return SIZE_MAX;
}
/* the placement's state: placed runs ascending by first position, and every run taken, by item at the end */
struct item_run { size_t item; uint64_t first, pages; };
struct placing { struct placed_item *placed; size_t count, room; uint64_t widest; struct item_run *taken; size_t takes, take_room; };
static int item_run_order(const void *left, const void *right) {
  const struct item_run *a = left, *b = right;
  if (a->item != b->item) return a->item < b->item ? -1 : 1;
  return (a->first > b->first) - (a->first < b->first);
}
static int place_take(struct placing *p, size_t item, uint64_t first, uint64_t pages) {
  if (p->takes == p->take_room) {
    const size_t grown = p->take_room ? 2 * p->take_room : 64;
    struct item_run *more = realloc(p->taken, grown * sizeof *more);
    if (!more) return ENOMEM;
    p->taken = more; p->take_room = grown;
  }
  p->taken[p->takes++] = (struct item_run){item, first, pages};
  return 0;
}
static int place_insert(struct placing *p, const struct item_run *r) {
  if (p->count == p->room) {
    const size_t grown = p->room ? 2 * p->room : 64;
    struct placed_item *more = realloc(p->placed, grown * sizeof *more);
    if (!more) return ENOMEM;
    p->placed = more; p->room = grown;
  }
  size_t y = p->count;
  while (y > 0 && p->placed[y - 1].first > r->first) { p->placed[y] = p->placed[y - 1]; y--; }
  p->placed[y] = (struct placed_item){r->first, r->pages, r->item};
  p->count++;
  if (r->pages > p->widest) p->widest = r->pages;
  return 0;
}
int mesh_graph_place(size_t n, const uint64_t *size, const uint64_t *key, const unsigned char *contiguous,
                     const uint64_t *pin, size_t bins, const uint64_t (*bin)[2],
                     int (*apart)(const void *context, size_t k, size_t l), const void *context, size_t *run_first,
                     uint64_t (**runs)[2]) {
  struct placing p = {0};
  size_t *order = malloc((n + 1) * sizeof *order), whole = 0, paged = 0;
  *runs = NULL;
  int result = order ? 0 : ENOMEM;
  /* order: the contiguous items, then the others; pins are neither */
  for (size_t k = 0; !result && k < n; k++) {
    if (!size[k] || pin[k] != UINT64_MAX) continue;
    if (contiguous && contiguous[k]) order[whole++] = k;
  }
  for (size_t k = 0; !result && k < n; k++)
    if (size[k] && pin[k] == UINT64_MAX && !(contiguous && contiguous[k])) order[whole + paged++] = k;
  /* pins: one run each, inside one bin, apart from every pin it meets */
  for (size_t k = 0; !result && k < n; k++) {
    if (!size[k] || pin[k] == UINT64_MAX) continue;
    size_t b = 0;
    while (b < bins && !(bin[b][0] <= pin[k] && pin[k] - bin[b][0] <= bin[b][1] && size[k] <= bin[b][1] - (pin[k] - bin[b][0]))) b++;
    if (b == bins) { result = EINVAL; break; }
    if (place_conflict(p.placed, p.count, p.widest, pin[k], size[k], k, apart, context) != SIZE_MAX) { result = EBUSY; break; }
    result = place_take(&p, k, pin[k], size[k]);
    if (!result) result = place_insert(&p, p.taken + p.takes - 1);
  }
  struct place_rank rank = {size, key};
  if (!result) {
    qsort_r(order, whole, sizeof *order, &rank, place_by_key);
    qsort_r(order + whole, paged, sizeof *order, &rank, place_by_key);
  }
  /* contiguous items: greedy by size on the arena's line (mesh_graph_greedy_by_size, ties by key), with the pins that
     lie in the arena already on it */
  if (!result && whole) {
    const uint64_t base = bin[bins - 1][0];
    size_t *here = malloc((n + 1) * sizeof *here), pins_here = 0;
    uint64_t *line = malloc((n + 1) * sizeof *line), extent = 0;
    if (!here || !line) result = ENOMEM;
    for (size_t k = 0; !result && k < n; k++)
      if (size[k] && pin[k] != UINT64_MAX && pin[k] >= base) { here[pins_here++] = k; line[k] = pin[k] - base; }
    if (!result)
      result = mesh_graph_greedy_by_size(whole, order, pins_here, here, size, apart, context, bin[bins - 1][1], line,
                                         &extent);
    for (size_t x = 0; !result && x < whole; x++) {
      result = place_take(&p, order[x], base + line[order[x]], size[order[x]]);
      if (!result) result = place_insert(&p, p.taken + p.takes - 1);
    }
    free(here); free(line);
  }
  /* every other item: the lowest free positions, page by page, joined into runs within a bin */
  for (size_t x = whole; !result && x < whole + paged; x++) {
    const size_t l = order[x], mine = p.takes;
    uint64_t need = size[l];
    for (size_t b = 0; !result && need && b < bins; b++) {
      const uint64_t end = bin[b][0] + bin[b][1];
      for (uint64_t at = bin[b][0]; !result && need && at < end;) {
        const size_t y = place_conflict(p.placed, p.count, p.widest, at, 1, l, apart, context);
        if (y != SIZE_MAX) { at = p.placed[y].first + p.placed[y].size; continue; }
        const uint64_t start = at;
        while (need && at < end && place_conflict(p.placed, p.count, p.widest, at, 1, l, apart, context) == SIZE_MAX) {
          at++; need--;
        }
        result = place_take(&p, l, start, at - start);
      }
    }
    if (!result && need) result = ENOMEM;
    for (size_t r = mine; !result && r < p.takes; r++) result = place_insert(&p, p.taken + r);
  }
  /* each item's runs, by item and position */
  if (!result) {
    qsort(p.taken, p.takes, sizeof *p.taken, item_run_order);
    *runs = malloc((p.takes + 1) * sizeof **runs);
    if (!*runs) result = ENOMEM;
  }
  for (size_t k = 0; !result && k <= n; k++) run_first[k] = 0;
  for (size_t r = 0; !result && r < p.takes; r++) {
    run_first[p.taken[r].item + 1]++;
    (*runs)[r][0] = p.taken[r].first; (*runs)[r][1] = p.taken[r].pages;
  }
  for (size_t k = 0; !result && k < n; k++) run_first[k + 1] += run_first[k];
  if (result) { free(*runs); *runs = NULL; }
  free(p.placed); free(p.taken); free(order);
  return result;
}

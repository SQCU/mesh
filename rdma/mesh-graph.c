#include "mesh-graph.h"
#include <errno.h>
#include <math.h>
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

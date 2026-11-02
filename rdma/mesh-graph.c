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
                         uint32_t *root, uint32_t *parent, double *log_weight){
  uint32_t *uses=calloc((size_t)P+1,sizeof *uses);
  double *length=malloc(((size_t)P+1)*sizeof *length),*weight=malloc(((size_t)m+1)*sizeof *weight);
  int32_t *chosen=malloc(((size_t)n+1)*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  uint32_t *head=malloc(((size_t)n+1)*sizeof *head),*next=NULL;
  size_t trees=0,room=0;
  uint32_t written=0;
  if(!uses || !length || !weight || !chosen || !head)goto done;
  for(uint32_t r=0;r<n;r++)head[r]=UINT32_MAX;
  const double delta=mesh_graph_exp(-mesh_graph_log(P/(1-eps))/eps);
  double D=0;
  for(uint32_t k=0;k<P;k++){length[k]=delta/capacity[k];D+=length[k]*capacity[k];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=rooted(context,r);demand>0 && D<1;){
    for(uint32_t e=0;e<m;e++){weight[e]=0;for(uint32_t i=through[e];i<through[e+1];i++)weight[e]+=length[crossing[i]];}
    if(mesh_graph_arborescence(n,r,m,from,to,weight,chosen))goto done;
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
  return written;
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
struct pack_event { uint64_t at; size_t item; int begins; };
static int pack_event_order(const void *left, const void *right) {
  const struct pack_event *a = left, *b = right;
  if (a->at != b->at) return a->at < b->at ? -1 : 1;
  if (a->begins != b->begins) return a->begins - b->begins;  /* an end at t frees before a begin at t takes */
  return (a->item > b->item) - (a->item < b->item);
}
/* the first place in free run [first, first + count) for an item of `size` units over [b, e) that meets no pinned item
   live at any time in [b, e) (the pinned items' units are given, so a later pinned item is respected too); UINT64_MAX
   where none */
static uint64_t pack_fit(uint64_t first, uint64_t count, uint64_t size, uint64_t b, uint64_t e, size_t n,
                         const uint64_t *sizes, const uint64_t *begins, const uint64_t *ends, const unsigned char *pinned,
                         const uint64_t *place) {
  uint64_t at = first;
  for (int moved = 1; moved && at + size <= first + count;) {
    moved = 0;
    for (size_t p = 0; pinned && p < n; p++)
      if (pinned[p] && sizes[p] && begins[p] < e && b < ends[p] && place[p] < at + size && at < place[p] + sizes[p]) {
        at = place[p] + sizes[p]; moved = 1;
      }
  }
  return at + size <= first + count ? at : UINT64_MAX;
}
int mesh_graph_interval_pack(size_t n, const uint64_t *size, const uint64_t *begin, const uint64_t *end,
                             const unsigned char *pinned, uint64_t capacity, uint64_t *place) {
  struct pack_event *events = malloc((2 * n + 1) * sizeof *events);
  uint64_t (*free_runs)[2] = malloc((2 * n + 2) * sizeof *free_runs);  /* [first, count), ascending, disjoint, joined */
  if (!events || !free_runs) { free(events); free(free_runs); return ENOMEM; }
  size_t e = 0, runs = 1;
  for (size_t k = 0; k < n; k++) if (size[k]) {
    events[e++] = (struct pack_event){begin[k], k, 1};
    events[e++] = (struct pack_event){end[k], k, 0};
  } else place[k] = 0;
  qsort(events, e, sizeof *events, pack_event_order);
  free_runs[0][0] = 0; free_runs[0][1] = capacity;
  int result = 0;
  for (size_t x = 0; x < e && !result; x++) {
    const size_t k = events[x].item;
    if (events[x].begins) {
      size_t r = 0;
      uint64_t at = UINT64_MAX;
      if (pinned && pinned[k]) {
        /* a pinned item's units must be free now: the run holding them whole */
        while (r < runs && !(free_runs[r][0] <= place[k] && place[k] + size[k] <= free_runs[r][0] + free_runs[r][1])) r++;
        if (r == runs) { result = EBUSY; break; }
        at = place[k];
      } else {
        for (; r < runs; r++)
          if ((at = pack_fit(free_runs[r][0], free_runs[r][1], size[k], begin[k], end[k], n, size, begin, end, pinned,
                             place)) != UINT64_MAX) break;
        if (r == runs) { result = ENOMEM; break; }
        place[k] = at;
      }
      /* [at, at + size) taken out of run r: what lies before and after it stays free */
      const uint64_t run_first = free_runs[r][0], run_end = free_runs[r][0] + free_runs[r][1];
      const int keep_before = at > run_first, keep_after = at + size[k] < run_end;
      if (keep_before && keep_after) {
        memmove(free_runs + r + 2, free_runs + r + 1, (runs - r - 1) * sizeof *free_runs); runs++;
        free_runs[r][1] = at - run_first;
        free_runs[r + 1][0] = at + size[k]; free_runs[r + 1][1] = run_end - (at + size[k]);
      } else if (keep_before) free_runs[r][1] = at - run_first;
      else if (keep_after) { free_runs[r][0] = at + size[k]; free_runs[r][1] = run_end - (at + size[k]); }
      else { memmove(free_runs + r, free_runs + r + 1, (runs - r - 1) * sizeof *free_runs); runs--; }
    } else {
      /* its units back, joined with a neighbour they touch */
      const uint64_t first = place[k], count = size[k];
      size_t r = 0;
      while (r < runs && free_runs[r][0] < first) r++;
      const int before = r > 0 && free_runs[r - 1][0] + free_runs[r - 1][1] == first;
      const int after = r < runs && first + count == free_runs[r][0];
      if (before && after) {
        free_runs[r - 1][1] += count + free_runs[r][1];
        memmove(free_runs + r, free_runs + r + 1, (runs - r - 1) * sizeof *free_runs); runs--;
      } else if (before) free_runs[r - 1][1] += count;
      else if (after) { free_runs[r][0] = first; free_runs[r][1] += count; }
      else {
        memmove(free_runs + r + 1, free_runs + r, (runs - r) * sizeof *free_runs); runs++;
        free_runs[r][0] = first; free_runs[r][1] = count;
      }
    }
  }
  free(events); free(free_runs);
  return result;
}

#include "mesh-graph.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <CommonCrypto/CommonDigest.h>
/* design/plan-to-bridge-change-request.md R5, Code; HiGHS 1.15.1, highs/interfaces/highs_c_api.h */
#include <interfaces/highs_c_api.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/local_space.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>

uint32_t mesh_graph_row_head(const void *context, size_t k) { return ((const uint32_t *)context)[k]; }

uint32_t mesh_graph_relation_head(const void *context, size_t k) {
  const struct mesh_graph_relation_data *data = context;
  return data->edge[data->row_edge[k]].to;
}

uint32_t mesh_graph_relation_incoming_head(const void *context, size_t k) {
  const struct mesh_graph_relation_data *data = context;
  return data->edge[data->in_edge[k]].from;
}

void mesh_graph_search_free(struct mesh_graph_search *made) {
  if (!made) return;
  free(made->preorder); free(made->postorder); free(made->parent); free(made->place); free(made->entry);
  free(made->end); free(made);
}

static struct mesh_graph_search *search_made(uint32_t vertices, int finishing) {
  const size_t room = (size_t)vertices + 1;
  struct mesh_graph_search *made = calloc(1, sizeof *made);
  if (!made) return NULL;
  made->vertices = vertices;
  made->preorder = malloc(room * sizeof *made->preorder);
  made->postorder = finishing ? malloc(room * sizeof *made->postorder) : NULL;
  made->parent = malloc(room * sizeof *made->parent);
  made->place = malloc(room * sizeof *made->place);
  made->entry = malloc(room * sizeof *made->entry);
  made->end = finishing ? malloc(room * sizeof *made->end) : NULL;
  if (!made->preorder || (finishing && (!made->postorder || !made->end)) || !made->parent || !made->place ||
      !made->entry) {
    mesh_graph_search_free(made);
    return NULL;
  }
  for (uint32_t v = 0; v < vertices; v++) {
    made->parent[v] = made->place[v] = UINT32_MAX; made->entry[v] = SIZE_MAX;
    if (finishing) made->end[v] = UINT32_MAX;
  }
  return made;
}

int mesh_graph_search(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                      const void *context, uint32_t roots, const uint32_t *root, int reverse,
                      struct mesh_graph_search **made) {
  *made = NULL;
  struct mesh_graph_search *s = search_made(vertices, 1);
  size_t *next = malloc(((size_t)vertices + 1) * sizeof *next);
  unsigned char *finished = calloc((size_t)vertices + 1, 1);
  int status = s && next && finished ? 0 : ENOMEM;
  uint32_t done = 0;
  for (uint32_t r = 0; !status && r < roots; r++) {
    uint32_t v = root[r];
    if (v >= vertices) { status = EINVAL; break; }
    if (s->place[v] != UINT32_MAX) continue;
    s->place[v] = s->count; s->preorder[s->count++] = v; next[v] = 0;
    while (!status && v != UINT32_MAX) {
      if (next[v] == first[v + 1] - first[v]) {
        finished[v] = 1; s->postorder[done++] = v; s->end[v] = s->count; v = s->parent[v];
        continue;
      }
      const size_t k = reverse ? first[v + 1] - 1 - next[v]++ : first[v] + next[v]++;
      const uint32_t w = head(context, k);
      if (w >= vertices) status = EINVAL;
      else if (s->place[w] == UINT32_MAX) {
        s->parent[w] = v; s->entry[w] = k; s->place[w] = s->count; s->preorder[s->count++] = w; next[w] = 0; v = w;
      } else if (!finished[w]) status = EDEADLK;
    }
  }
  free(next); free(finished);
  if (status) { mesh_graph_search_free(s); return status; }
  *made = s;
  return 0;
}

int mesh_graph_ancestor(const struct mesh_graph_search *search, uint32_t ancestor, uint32_t vertex) {
  return search->end && ancestor < search->vertices && vertex < search->vertices &&
         search->place[ancestor] != UINT32_MAX && search->place[vertex] != UINT32_MAX &&
         search->place[ancestor] <= search->place[vertex] && search->place[vertex] < search->end[ancestor];
}

int mesh_graph_breadth(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                       const void *context, uint32_t starts, const uint32_t *start, struct mesh_graph_search **made) {
  return mesh_graph_breadth_masked(vertices, first, head, context, starts, start, NULL, made);
}

int mesh_graph_breadth_masked(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                              const void *context, uint32_t starts, const uint32_t *start, const unsigned char *active,
                              struct mesh_graph_search **made) {
  *made = NULL;
  struct mesh_graph_search *s = search_made(vertices, 0);
  int status = s ? 0 : ENOMEM;
  for (uint32_t r = 0; !status && r < starts; r++) {
    const uint32_t v = start[r];
    if (v >= vertices) status = EINVAL;
    else if (s->place[v] == UINT32_MAX) { s->place[v] = s->count; s->preorder[s->count++] = v; }
  }
  for (uint32_t at = 0; !status && at < s->count; at++) {
    const uint32_t v = s->preorder[at];
    for (size_t k = first[v]; !status && (!active || active[v]) && k < first[v + 1]; k++) {
      const uint32_t w = head(context, k);
      if (w >= vertices) status = EINVAL;
      else if (s->place[w] == UINT32_MAX) {
        s->parent[w] = v; s->entry[w] = k; s->place[w] = s->count; s->preorder[s->count++] = w;
      }
    }
  }
  if (status) { mesh_graph_search_free(s); return status; }
  *made = s;
  return 0;
}

void mesh_graph_closure_free(struct mesh_graph_closure *made) {
  if (!made) return;
  free(made->reached); free(made);
}

static int closure_descending(const void *a, const void *b) {
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return (x < y) - (x > y);
}

/* the closure rows of mesh_graph_closure (label NULL: column u for head u, every row followed, a head whose bit is
   already set skipped) and mesh_graph_closure_labels (column label[u], each head's row merged) */
static int closure_rows(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                        const void *context, const uint32_t *order, ptrdiff_t step, const unsigned char *active,
                        uint32_t columns, const uint32_t *label, struct mesh_graph_closure **made) {
  *made = NULL;
  const size_t words = ((size_t)columns + 63) / 64;
  size_t most = 0;
  for (uint32_t v = 0; v < vertices; v++) if (first[v + 1] - first[v] > most) most = first[v + 1] - first[v];
  struct mesh_graph_closure *c = calloc(1, sizeof *c);
  uint32_t *heads = malloc((most + 1) * sizeof *heads), *rank = malloc(((size_t)vertices + 1) * sizeof *rank);
  int status = c && heads && rank && !(words && (size_t)vertices > SIZE_MAX / sizeof(uint64_t) / words) ? 0 : ENOMEM;
  for (uint32_t v = 0; !status && v < vertices; v++) rank[v] = UINT32_MAX;
  for (uint32_t i = 0; !status && i < vertices; i++) {
    const uint32_t v = order[(ptrdiff_t)i * step];
    if (v >= vertices || rank[v] != UINT32_MAX) status = EINVAL;
    else rank[v] = i;
  }
  if (!status) {
    c->vertices = vertices; c->words = words;
    c->reached = calloc((size_t)vertices * words + 1, sizeof *c->reached);
    if (!c->reached) status = ENOMEM;
  }
  for (uint32_t i = 0; !status && i < vertices; i++) {
    const uint32_t v = order[(ptrdiff_t)i * step];
    size_t n = 0;
    for (size_t k = first[v]; !status && (!active || active[v]) && k < first[v + 1]; k++) {
      const uint32_t u = head(context, k);
      if (u >= vertices || rank[u] >= i) status = EINVAL;
      else heads[n++] = rank[u];
    }
    qsort(heads, n, sizeof *heads, closure_descending);
    uint64_t *row = c->reached + (size_t)v * words;
    for (size_t j = 0; !status && j < n; j++) {
      const uint32_t u = order[(ptrdiff_t)heads[j] * step], column = label ? label[u] : u;
      if (column != UINT32_MAX && column >= columns) status = EINVAL;
      else if (label || !(row[u / 64] >> (u % 64) & 1)) {
        const uint64_t *from = c->reached + (size_t)u * words;
        for (size_t w = 0; w < words; w++) row[w] |= from[w];
        if (column != UINT32_MAX) row[column / 64] |= UINT64_C(1) << (column % 64);
      }
    }
  }
  free(heads); free(rank);
  if (status) { mesh_graph_closure_free(c); return status; }
  *made = c;
  return 0;
}

int mesh_graph_closure(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                       const void *context, const uint32_t *order, ptrdiff_t step, struct mesh_graph_closure **made) {
  return closure_rows(vertices, first, head, context, order, step, NULL, vertices, NULL, made);
}

int mesh_graph_closure_labels(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                              const void *context, const uint32_t *order, ptrdiff_t step, const unsigned char *active,
                              uint32_t columns, const uint32_t *label, struct mesh_graph_closure **made) {
  return closure_rows(vertices, first, head, context, order, step, active, columns, label, made);
}

int mesh_graph_longest(const struct mesh_graph_relation *relation, int incoming, const uint32_t *order, ptrdiff_t step,
                       const double *weight, double *length) {
  const struct mesh_graph_relation_data *d = mesh_graph_relation_data(relation);
  const uint32_t vertices = d->vertices;
  const size_t *first = incoming ? d->in_first : d->row_first, *entry = incoming ? d->in_edge : d->row_edge;
  uint32_t *rank = malloc(((size_t)vertices + 1) * sizeof *rank);
  int status = rank ? 0 : ENOMEM;
  for (uint32_t v = 0; !status && v < vertices; v++) rank[v] = UINT32_MAX;
  for (uint32_t i = 0; !status && i < vertices; i++) {
    const uint32_t v = order[(ptrdiff_t)i * step];
    if (v >= vertices || rank[v] != UINT32_MAX) status = EINVAL;
    else rank[v] = i;
  }
  for (uint32_t i = 0; !status && i < vertices; i++) {
    const uint32_t v = order[(ptrdiff_t)i * step];
    double best = 0;
    for (size_t k = first[v]; !status && k < first[v + 1]; k++) {
      const size_t e = entry[k];
      const uint32_t u = incoming ? d->edge[e].from : d->edge[e].to;
      if (u >= vertices || rank[u] >= i) status = EINVAL;
      else if (k == first[v] || length[u] + weight[e] > best) best = length[u] + weight[e];
    }
    length[v] = best;
  }
  free(rank);
  return status;
}

int mesh_graph_boundaries(const struct mesh_graph_relation *relation, const uint32_t *class_of,
                          const unsigned char *excluded, unsigned char *source, unsigned char *sink) {
  const struct mesh_graph_relation_data *d = mesh_graph_relation_data(relation);
  for (uint32_t v = 0; v < d->vertices; v++) source[v] = sink[v] = 1;
  for (size_t e = 0; e < d->edges; e++)
    if (!excluded[e] && class_of[d->edge[e].from] == class_of[d->edge[e].to]) {
      source[d->edge[e].to] = 0;
      sink[d->edge[e].from] = 0;
    }
  return 0;
}

/* GSPMD §3.5, "Merging compatible shardings": the projection of mesh_graph_sharding through `map` into vertex w,
   seen[x] == stamp marking mesh axis x in the result; 1 where it updated a slot */
static int sharding_merge(const struct mesh_graph_sharding *s, uint32_t w, const size_t *map, uint32_t *candidate,
                          size_t *seen, size_t stamp) {
  const size_t base = s->first[w], dims = s->first[w + 1] - base;
  int updated = 0;
  if (s->frozen[w]) return 0;
  for (size_t i = 0; i < dims; i++) candidate[i] = map[i] == SIZE_MAX ? UINT32_MAX : s->state[map[i]];
  for (size_t i = 0; i < dims; i++) {
    const uint32_t x = s->state[base + i], c = candidate[i], m = c == UINT32_MAX ? x : c;
    if (x != UINT32_MAX && c != UINT32_MAX && x != c) return 0;
    if (m != UINT32_MAX && seen[s->axis[m]] == stamp) return 0;
    if (m != UINT32_MAX) seen[s->axis[m]] = stamp;
    updated |= x == UINT32_MAX && c != UINT32_MAX;
  }
  for (size_t i = 0; updated && i < dims; i++) if (candidate[i] != UINT32_MAX) s->state[base + i] = candidate[i];
  return updated;
}

int mesh_graph_sharding(const struct mesh_graph_sharding *s) {
  size_t most = 0, axes = 0, stamp = 0;
  for (uint32_t v = 0; v < s->vertices; v++) if (s->first[v + 1] - s->first[v] > most) most = s->first[v + 1] - s->first[v];
  for (uint32_t a = 0; a < s->atoms; a++) if ((size_t)s->axis[a] + 1 > axes) axes = (size_t)s->axis[a] + 1;
  uint32_t *candidate = malloc((most + 1) * sizeof *candidate);
  size_t *seen = calloc(axes + 1, sizeof *seen);
  const int status = candidate && seen ? 0 : ENOMEM;
  for (uint32_t l = 0; !status && l < s->levels; l++)
    for (int updated = 1; updated;) {
      updated = 0;
      for (int backward = 0; backward < 2; backward++)
        for (uint32_t i = 0; i < s->vertices; i++) {
          const uint32_t v = s->order[backward ? s->vertices - 1 - i : i];
          for (size_t e = s->edge_first[v]; e < s->edge_first[v + 1]; e++)
            if (s->edge[e].level <= l)
              updated |= backward ? sharding_merge(s, s->edge[e].operand, s->map + s->edge[e].backward, candidate, seen,
                                                   ++stamp)
                                  : sharding_merge(s, v, s->map + s->edge[e].forward, candidate, seen, ++stamp);
        }
    }
  free(candidate); free(seen);
  return status;
}

/* the breadth-first search of mesh_graph_breadth over both directions of each edge, row v's entries in edge order and
   side order, as the edges were scanned; parent[from] is from */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent) {
  for (uint32_t v = 0; v < vertices; v++) { reached[v] = 0; if (parent) parent[v] = UINT32_MAX; }
  size_t *first = calloc((size_t)vertices + 2, sizeof *first);
  uint32_t *head = malloc(2 * (size_t)edges * sizeof *head + sizeof *head);
  struct mesh_graph_search *s = NULL;
  uint32_t count = 0;
  if (first && head) {
    for (uint32_t e = 0; e < edges; e++) for (uint32_t side = 0; side < 2; side++) first[end(context, e, side) + 2]++;
    for (uint32_t v = 0; v < vertices; v++) first[v + 2] += first[v + 1];
    for (uint32_t e = 0; e < edges; e++)
      for (uint32_t side = 0; side < 2; side++) head[first[end(context, e, side) + 1]++] = end(context, e, 1 - side);
  }
  if (first && head && !mesh_graph_breadth(vertices, first, mesh_graph_row_head, head, 1, &from, &s)) {
    for (count = 0; count < s->count; count++) {
      order[count] = s->preorder[count]; reached[order[count]] = 1;
      if (parent) parent[order[count]] = order[count] == from ? from : s->parent[order[count]];
    }
  }
  mesh_graph_search_free(s); free(first); free(head);
  return count;
}

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

int mesh_graph_preorder(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                        const void *context, uint32_t root,
                        int (*visit)(void *consumer, uint32_t vertex, uint32_t parent), void *consumer) {
  if (root >= vertices) return EINVAL;
  struct mesh_graph_search *s;
  int status = mesh_graph_search(vertices, first, head, context, 1, &root, 0, &s);
  for (uint32_t k = 0; !status && k < s->count; k++) status = visit(consumer, s->preorder[k], s->parent[s->preorder[k]]);
  mesh_graph_search_free(s);
  return status;
}
/* T. H. Cormen et al., 3rd ed., §22.3 without the visited set; A. V. Aho, M. S. Lam, R. Sethi and J. D. Ullman,
   Compilers, 2nd ed., §7.2.1, activation trees: the path stack marks its vertices open, so a vertex recurs once per
   path and never twice on one */
int mesh_graph_unfold(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                      const void *context, uint32_t root,
                      int (*visit)(void *consumer, uint32_t vertex, size_t k, uint32_t parent), void *consumer) {
  if (root >= vertices) return EINVAL;
  uint32_t *at = malloc(((size_t)vertices + 1) * sizeof *at), *instance = malloc(((size_t)vertices + 1) * sizeof *instance);
  size_t *next = malloc(((size_t)vertices + 1) * sizeof *next);
  unsigned char *open = calloc((size_t)vertices + 1, 1);
  int status = at && instance && next && open ? 0 : ENOMEM;
  uint32_t depth = 0, made = 0;
  if (!status) status = visit(consumer, root, SIZE_MAX, UINT32_MAX);
  if (!status) { at[0] = root; next[0] = first[root]; instance[0] = made++; open[root] = 1; depth = 1; }
  while (!status && depth) {
    const uint32_t v = at[depth - 1];
    if (next[depth - 1] == first[v + 1]) { open[v] = 0; depth--; continue; }
    const size_t k = next[depth - 1]++;
    const uint32_t w = head(context, k);
    status = w >= vertices ? EINVAL : open[w] ? EDEADLK : made == UINT32_MAX ? EOVERFLOW : visit(consumer, w, k, instance[depth - 1]);
    if (status) break;
    at[depth] = w; next[depth] = first[w]; instance[depth] = made++; open[w] = 1; depth++;
  }
  free(at); free(instance); free(next); free(open);
  return status;
}

struct within { uint32_t N,root,depth; const uint32_t *from,*order,*into; const uint32_t *in; const double *weight,*rest;
                int32_t *now,*best; double cost; };
static int within_walk(const struct within *s,uint32_t v){
  for(uint32_t k=0;k<=s->depth;k++){
    if(v==s->root)return 1;
    if(s->now[v]<0)return 1;
    v=s->from[s->now[v]];
  }
  return 0;
}
static void within_search(struct within *s,uint32_t at,double cost){
  if(cost+s->rest[at]>=s->cost)return;
  if(at==s->N-1){
    for(uint32_t k=0;k+1<s->N;k++)if(!within_walk(s,s->order[k]))return;
    s->cost=cost;
    memcpy(s->best,s->now,s->N*sizeof *s->now);
    return;
  }
  const uint32_t v=s->order[at];
  for(uint32_t i=s->in[v];i<s->in[v+1];i++){
    const uint32_t e=s->into[i];
    if(cost+s->weight[e]+s->rest[at+1]>=s->cost)break;
    s->now[v]=(int32_t)e;
    int ok=within_walk(s,v);
    for(uint32_t k=0;ok && k<at;k++)ok=within_walk(s,s->order[k]);
    if(ok)within_search(s,at+1,cost+s->weight[e]);
  }
  s->now[v]=-1;
}
int mesh_graph_arborescence_within(uint32_t N,uint32_t root,uint32_t E,const uint32_t *from,const uint32_t *to,
                                   const double *weight,uint32_t depth,int32_t *chosen){
  uint32_t *in=calloc((size_t)N+2,sizeof *in),*into=malloc(((size_t)E+1)*sizeof *into),*order=malloc(((size_t)N+1)*sizeof *order);
  uint32_t *layer=malloc(((size_t)N+1)*sizeof *layer);
  double *rest=calloc((size_t)N+1,sizeof *rest);
  int32_t *now=malloc(((size_t)N+1)*sizeof *now),*best=malloc(((size_t)N+1)*sizeof *best);
  int status=-1;
  if(!in || !into || !order || !layer || !rest || !now || !best || root>=N)goto done;
  for(uint32_t e=0;e<E;e++)if(from[e]!=to[e] && to[e]!=root)in[to[e]+1]++;
  for(uint32_t v=0;v<N;v++)in[v+1]+=in[v];
  {
    uint32_t *fill=calloc((size_t)N+1,sizeof *fill);
    if(!fill)goto done;
    for(uint32_t e=0;e<E;e++)if(from[e]!=to[e] && to[e]!=root)into[in[to[e]]+fill[to[e]]++]=e;
    free(fill);
  }
  for(uint32_t v=0;v<N;v++)for(uint32_t i=in[v]+1;i<in[v+1];i++)
    for(uint32_t j=i;j>in[v] && weight[into[j]]<weight[into[j-1]];j--){const uint32_t t=into[j];into[j]=into[j-1];into[j-1]=t;}
  for(uint32_t v=0;v<N;v++){layer[v]=UINT32_MAX;now[v]=-1;}
  layer[root]=0;
  uint32_t placed=0;
  for(uint32_t d=0,grew=1;grew;d++){
    grew=0;
    for(uint32_t e=0;e<E;e++)if(layer[from[e]]==d && layer[to[e]]==UINT32_MAX){layer[to[e]]=d+1;grew=1;}
  }
  for(uint32_t d=1;placed+1<N && d<=N;d++)for(uint32_t v=0;v<N;v++)if(v!=root && layer[v]==d)order[placed++]=v;
  if(placed+1!=N)goto done;
  for(uint32_t at=N-1;at-->0;)rest[at]=rest[at+1]+weight[into[in[order[at]]]];
  struct within s={N,root,depth,from,order,into,in,weight,rest,now,best,INFINITY};
  within_search(&s,0,0);
  if(s.cost==INFINITY)goto done;
  for(uint32_t v=0;v<N;v++)chosen[v]=v==root?-1:best[v];
  status=0;
done:
  free(in);free(into);free(order);free(layer);free(rest);free(now);free(best);
  return status;
}

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

void mesh_graph_rows(uint32_t keys, size_t items, uint32_t (*key)(const void *context, size_t k), const void *context,
                     size_t *first, size_t *order) {
  for (uint32_t v = 0; v <= keys; v++) first[v] = 0;
  for (size_t k = 0; k < items; k++) first[key(context, k) + 1]++;
  for (uint32_t v = 0; v < keys; v++) first[v + 1] += first[v];
  for (size_t k = 0; k < items; k++) order[first[key(context, k)]++] = k;
  for (uint32_t v = keys; v > 0; v--) first[v] = first[v - 1];
  first[0] = 0;
}

int mesh_graph_lifetimes(uint32_t values, size_t uses, const uint32_t *value, const size_t *position,
                         size_t *first, size_t *last) {
  for (uint32_t v = 0; v < values; v++) first[v] = last[v] = SIZE_MAX;
  for (size_t k = 0; k < uses; k++) {
    uint32_t v = value[k];
    if (first[v] == SIZE_MAX || first[v] > position[k]) first[v] = position[k];
    if (last[v] == SIZE_MAX || position[k] > last[v]) last[v] = position[k];
  }
  return 0;
}

uint32_t mesh_graph_keyed_find(const uint32_t *head, const uint32_t *next, uint32_t key,
                               int (*equal)(const void *context, uint32_t k), const void *context) {
  for (uint32_t k = head[key]; k != UINT32_MAX; k = next[k]) if (equal(context, k)) return k;
  return UINT32_MAX;
}

void mesh_graph_keyed_add(uint32_t *head, uint32_t *next, uint32_t key, uint32_t k) {
  next[k] = head[key];
  head[key] = k;
}

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

struct flow_same { const uint32_t *parent,*tree; uint32_t n; };
static int flow_same(const void *context,uint32_t k){
  const struct flow_same *s=context;
  return !memcmp(s->parent+(size_t)k*s->n,s->tree,s->n*sizeof *s->tree);
}

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

struct mesh_packed { uint32_t root; double flow; uint32_t *parent; };
static int mesh_packed_heavier(const void *a,const void *b){
  const struct mesh_packed *x=a,*y=b;
  if(x->root!=y->root)return x->root<y->root?-1:1;
  return x->flow>y->flow?-1:x->flow<y->flow;
}

struct pack_same { const struct mesh_packed *packed; const uint32_t *from; const int32_t *chosen; uint32_t r,n; };
static int pack_same(const void *context,uint32_t t){
  const struct pack_same *s=context;
  uint32_t v=0;
  while(v<s->n && (v==s->r || s->packed[t].parent[v]==s->from[s->chosen[v]]))v++;
  return v==s->n;
}

uint32_t mesh_graph_pack(uint32_t n, uint32_t m, const uint32_t *from, const uint32_t *to, const uint32_t *through,
                         const uint32_t *crossing, uint32_t P, const double *capacity,
                         int (*rooted)(const void *context, uint32_t r), const void *context, double eps, uint32_t most,
                         int shallow, uint32_t *root, uint32_t *parent, double *log_weight){
  uint32_t *uses=calloc((size_t)P+1,sizeof *uses);
  uint32_t *eccentricity=shallow?calloc((size_t)n+1,sizeof *eccentricity):NULL,*layer=shallow?malloc(((size_t)n+1)*sizeof *layer):NULL;
  double *length=malloc(((size_t)P+1)*sizeof *length),*weight=malloc(((size_t)m+1)*sizeof *weight);
  int32_t *chosen=malloc(((size_t)n+1)*sizeof *chosen);
  struct mesh_packed *packed=NULL;
  uint32_t *head=malloc(((size_t)n+1)*sizeof *head),*next=NULL;
  size_t trees=0,room=0;
  uint32_t written=0;
  if(!uses || !length || !weight || !chosen || !head)goto done;
  if(shallow && (!eccentricity || !layer))goto done;
  for(uint32_t r=0;shallow && r<n;r++){
    for(uint32_t v=0;v<n;v++)layer[v]=UINT32_MAX;
    layer[r]=0;
    for(uint32_t d=0,grew=1;grew;d++){
      grew=0;
      for(uint32_t e=0;e<m;e++)if(layer[from[e]]==d && layer[to[e]]==UINT32_MAX){layer[to[e]]=d+1;grew=1;}
    }
    for(uint32_t v=0;v<n;v++)if(layer[v]!=UINT32_MAX && layer[v]>eccentricity[r])eccentricity[r]=layer[v];
  }
  for(uint32_t r=0;r<n;r++)head[r]=UINT32_MAX;
  const double delta=mesh_graph_exp(-mesh_graph_log(P/(1-eps))/eps);
  double D=0;
  for(uint32_t k=0;k<P;k++){length[k]=delta/capacity[k];D+=length[k]*capacity[k];}
  while(D<1)for(uint32_t r=0;r<n && D<1;r++)for(double demand=rooted(context,r);demand>0 && D<1;){
    for(uint32_t e=0;e<m;e++){weight[e]=0;for(uint32_t i=through[e];i<through[e+1];i++)weight[e]+=length[crossing[i]];}
    if(shallow){if(mesh_graph_arborescence_within(n,r,m,from,to,weight,eccentricity[r],chosen))goto done;}
    else if(mesh_graph_arborescence(n,r,m,from,to,weight,chosen))goto done;
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
  free(eccentricity);free(layer);
  return written;
}

uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index){
  const uint64_t g=cell?cell:1,end=first+length;
  uint32_t made=0;
  for(uint64_t at=first;at<end;){
    const uint64_t k=at/g,edge=(k+1)*g<end?(k+1)*g:end;
    start[made]=at;count[made]=edge-at;index[made++]=k;at=edge;
  }
  return made;
}
uint32_t mesh_graph_tree_depth(uint32_t vertices, uint32_t root, const uint32_t *parent){
  size_t *first=calloc((size_t)vertices+1,sizeof *first);
  uint32_t *head=malloc(((size_t)vertices+1)*sizeof *head),*depth=malloc(((size_t)vertices+1)*sizeof *depth),most=UINT32_MAX;
  unsigned char *marked=malloc((size_t)vertices+1);
  if(first && head && depth && marked){
    for(uint32_t v=0;v<vertices;v++){
      first[v+1]=first[v];marked[v]=parent[v]!=UINT32_MAX;
      if(v!=root && parent[v]!=UINT32_MAX)head[first[v+1]++]=parent[v];
    }
    if(!mesh_graph_depth(vertices,first,head,marked,depth)){
      most=0;
      for(uint32_t v=0;v<vertices;v++)if(marked[v] && depth[v]>most)most=depth[v];
    }
  }
  free(first);free(head);free(depth);free(marked);
  return most;
}
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count){
  for(uint32_t p=0;p<parts;p++)count[p]=0;
  if(!units || !parts)return 0;
  double lo=INFINITY,hi=0;
  for(uint32_t p=0;p<parts;p++)if(rate[p]>0){
    if(latency[p]<lo)lo=latency[p];
    const double alone=latency[p]+(double)units/rate[p];
    if(!(hi>0) || alone<hi)hi=alone;
  }
  if(!isfinite(lo))return INFINITY;
#define MESH_FILL_BY(T) ({ uint64_t n_=0; for(uint32_t p_=0;p_<parts;p_++) if(rate[p_]>0 && (T)>latency[p_]) \
    n_+=(uint64_t)floor(((T)-latency[p_])*rate[p_]); n_; })
  for(int i=0;i<200 && hi-lo>0;i++){
    const double mid=lo+(hi-lo)/2;
    if(mid<=lo || mid>=hi)break;
    if(MESH_FILL_BY(mid)>=units)hi=mid;else lo=mid;
  }
#undef MESH_FILL_BY
  uint64_t total=0;
  for(uint32_t p=0;p<parts;p++)
    if(rate[p]>0 && hi>latency[p]){count[p]=(uint64_t)floor((hi-latency[p])*rate[p]);total+=count[p];}
  while(total<units){
    uint32_t best=UINT32_MAX;double soon=INFINITY;
    for(uint32_t p=0;p<parts;p++)if(rate[p]>0 && latency[p]+(double)(count[p]+1)/rate[p]<soon){soon=latency[p]+(double)(count[p]+1)/rate[p];best=p;}
    count[best]++;total++;
  }
  while(total>units){
    uint32_t worst=UINT32_MAX;double late=-INFINITY;
    for(uint32_t p=0;p<parts;p++)if(count[p] && latency[p]+(double)count[p]/rate[p]>late){late=latency[p]+(double)count[p]/rate[p];worst=p;}
    count[worst]--;total--;
  }
  double end=0;
  for(uint32_t p=0;p<parts;p++)if(count[p] && latency[p]+(double)count[p]/rate[p]>end)end=latency[p]+(double)count[p]/rate[p];
  return end;
}

/* design/page-queue-change-request.md, "The rule both ends compute" */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried){
  for(uint32_t l=0;l<machines;l++)carried[l]=0;
  int lost=0;
  for(uint64_t k=0;k<units;k++){
    uint32_t best=UINT32_MAX;double soon=INFINITY;
    for(uint32_t l=0;l<machines;l++){
      if(!(rate[l]>0))continue;
      const double at=(double)(carried[l]+bytes[k])/rate[l];
      if(best==UINT32_MAX || at<soon){soon=at;best=l;}
    }
    machine[k]=best;
    if(best==UINT32_MAX)lost=1;else carried[best]+=bytes[k];
  }
  if(lost)return INFINITY;
  double end=0;
  for(uint32_t l=0;l<machines;l++)if(carried[l] && (double)carried[l]/rate[l]>end)end=(double)carried[l]/rate[l];
  return end;
}

/* OpenXLA gpu_performance_model_base.cc: AdjustBandwidth */
static double actor_bandwidth(const struct mesh_graph_machine *x, double bandwidth, uint64_t blocks){
  const double most=(double)blocks*x->block_bandwidth;
  return x->block_bandwidth>0 && most<bandwidth?most:bandwidth;
}
/* OpenXLA gpu_performance_model_base.cc: ReadTimeWithDRAMHeuristic, in seconds */
static double actor_read(const struct mesh_graph_machine *x, uint64_t blocks, const struct mesh_graph_read *r){
  double dram=x->bandwidth*r->utilization,rest=x->bandwidth;
  if((double)r->net<x->l2){
    rest*=x->l2_speedup;
    if((double)r->net<x->l1*x->cores)rest*=x->l1_speedup;
  }
  else rest*=r->utilization;
  dram=actor_bandwidth(x,dram,blocks);rest=actor_bandwidth(x,rest,blocks);
  const uint64_t from_dram=r->net<r->total?r->net:r->total,from_cache=r->total-from_dram;
  return (double)from_dram/dram+(double)from_cache/rest;
}
/* OpenXLA gpu_performance_model_base.cc: CalculateEffectiveFlopsPerNs, ComputeTime, WriteTime and
   CombineComputeAndMemoryAccessTime; gpu_performance_model.cc EstimateRunTimeForInstructionImpl and EstimateRunTimes'
   kKernelLaunchOverhead a launch */
int mesh_graph_actor_time(const struct mesh_graph_machine *x, const struct mesh_graph_actor *a, double *time){
  const uint64_t cores=a->blocks<x->cores?a->blocks:x->cores;
  int missing=a->work.unknown!=0;
  double compute=0;
  for(uint32_t resource=0;resource<MESH_GRAPH_RESOURCES;resource++)
    for(uint32_t width=0;width<MESH_GRAPH_WIDTHS;width++){
      const double work=a->work.operations[resource][width];
      if(!(work>0))continue;
      const struct mesh_graph_rate *rate=&x->rate[resource][width];
      if(!(rate->peak>0) || !rate->lanes || !x->cores || !cores || !a->threads){missing=1;continue;}
      const uint64_t lanes=a->threads<rate->lanes?a->threads:rate->lanes;
      compute=fmax(compute,work/(rate->peak*(double)cores/x->cores*(double)lanes/rate->lanes));
    }
  uint64_t moved=a->written;
  for(size_t k=0;k<a->reads;k++)moved+=a->read[k].total;
  const int banded=x->bandwidth>0,capped=x->block_bandwidth>0 || a->blocks>=x->cores;
  missing|=(moved && !(banded && capped)) || (a->launches && !(x->launch>0)) ||
           (a->direct && !(x->direct>0)) || (a->blits && !(x->blit>0));
  double memory=0;
  if(moved && banded){
    memory=(double)a->written/actor_bandwidth(x,x->bandwidth,a->blocks);
    for(size_t k=0;k<a->reads;k++){
      if(a->read[k].total && !(a->read[k].utilization>0)){missing=1;continue;}
      if(a->read[k].total)memory+=actor_read(x,a->blocks,a->read+k);
    }
    memory*=1.0e9;
  }
  const double least=compute<memory?compute:memory;
  *time=compute+memory-least*x->overlap+(double)a->launches*x->launch+(double)a->direct*x->direct+(double)a->blits*x->blit;
  return missing?ENODATA:0;
}

/* docs/provider.md#composed-resource-costs */
int mesh_graph_cost_times(const struct mesh_graph_machine *machine,size_t actors,const struct mesh_graph_actor *actor,
                          size_t nodes,const struct mesh_graph_cost *node,double *value,int *status){
  if(nodes>UINT32_MAX || (nodes && (!node || !value || !status)))return EINVAL;
  for(uint32_t i=0;i<nodes;i++){
    const struct mesh_graph_cost *n=node+i;
    value[i]=0;status[i]=0;
    if(n->kind==MESH_GRAPH_COST_ACTOR){
      status[i]=n->first<actors?mesh_graph_actor_time(machine,actor+n->first,value+i):EINVAL;
      continue;
    }
    if(n->first>=i || n->second>=i){status[i]=EINVAL;continue;}
    status[i]=status[n->first];
    if(status[n->second] && (!status[i] || status[i]==ENODATA))status[i]=status[n->second];
    if(n->kind==MESH_GRAPH_COST_SUM)value[i]=value[n->first]+value[n->second];
    else status[i]=EINVAL;
  }
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.3; design/rubrics.md:422 (C9) */
static void heap_up(uint32_t *heap, uint32_t *position, int (*before)(const void *context, uint32_t a, uint32_t b),
                    const void *context, size_t i) {
  const uint32_t v = heap[i];
  while (i && before(context, v, heap[(i - 1) / 2])) {
    heap[i] = heap[(i - 1) / 2];
    position[heap[i]] = (uint32_t)i;
    i = (i - 1) / 2;
  }
  heap[i] = v;
  position[v] = (uint32_t)i;
}
static void heap_down(uint32_t *heap, uint32_t *position, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, size_t count, size_t i) {
  const uint32_t v = heap[i];
  for (size_t c = 2 * i + 1; c < count; c = 2 * i + 1) {
    if (c + 1 < count && before(context, heap[c + 1], heap[c])) c++;
    if (!before(context, heap[c], v)) break;
    heap[i] = heap[c];
    position[heap[i]] = (uint32_t)i;
    i = c;
  }
  heap[i] = v;
  position[v] = (uint32_t)i;
}
static int lags_before(const void *context, uint32_t a, uint32_t b) {
  const uint64_t *distance = context;
  return distance[a] < distance[b] || (distance[a] == distance[b] && a < b);
}
static size_t lags_relax(uint32_t *heap, uint32_t *place, uint64_t *distance, size_t count, uint32_t v, uint64_t at,
                         uint64_t weight, size_t *via, size_t k) {
  if (weight == UINT64_MAX || at > UINT64_MAX - 1 - weight || at + weight >= distance[v]) return count;
  distance[v] = at + weight;
  if (via) via[v] = k;
  if (place[v] == UINT32_MAX) {
    heap[count] = v;
    heap_up(heap, place, lags_before, distance, count);
    return count + 1;
  }
  heap_up(heap, place, lags_before, distance, place[v]);
  return count;
}
/* E. W. Dijkstra, Numerische Mathematik 1, 1959, problem 2: lags[s * targets + x] the least sum of lags over a walk of
   at least one edge from source[s] to target[x], UINT64_MAX where there is none; an edge of lag UINT64_MAX is absent,
   and a walk's sum is kept below UINT64_MAX */
/* via[s * vertices + v], where given: the edge position k whose relaxation set v's least lag from source[s], the
   selected walk's last edge into v */
static int lags_walks(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                      uint64_t (*lag)(const void *context, size_t k), const void *context, uint32_t sources,
                      const uint32_t *source, uint32_t targets, const uint32_t *target, uint64_t *lags, size_t *via) {
  const size_t edges = first[vertices];
  if (edges >= SIZE_MAX / sizeof(uint64_t)) return ENOMEM;
  uint32_t *heads = malloc((edges + 1) * sizeof *heads);
  uint64_t *weights = malloc((edges + 1) * sizeof *weights), *distance = malloc(((size_t)vertices + 1) * sizeof *distance);
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap), *place = malloc(((size_t)vertices + 1) * sizeof *place);
  unsigned char *wanted = calloc((size_t)vertices + 1, 1);
  const int result = !heads || !weights || !distance || !heap || !place || !wanted ? ENOMEM : 0;
  uint32_t distinct = 0;
  for (size_t k = 0; !result && k < edges; k++) { heads[k] = head(context, k); weights[k] = lag(context, k); }
  for (uint32_t x = 0; !result && x < targets; x++)
    if (!wanted[target[x]]) { wanted[target[x]] = 1; distinct++; }
  for (uint32_t s = 0; !result && distinct && s < sources; s++) {
    for (uint32_t v = 0; v < vertices; v++) { distance[v] = UINT64_MAX; place[v] = UINT32_MAX; }
    const uint32_t from = source[s];
    size_t count = 0, *walk = via ? via + (size_t)s * vertices : NULL;
    for (size_t k = first[from]; k < first[from + 1]; k++)
      count = lags_relax(heap, place, distance, count, heads[k], 0, weights[k], walk, k);
    for (uint32_t left = distinct; count;) {
      const uint32_t at = heap[0];
      place[at] = UINT32_MAX;
      if (--count) { heap[0] = heap[count]; heap_down(heap, place, lags_before, distance, count, 0); }
      if (wanted[at] && !--left) break;
      for (size_t k = first[at]; k < first[at + 1]; k++)
        count = lags_relax(heap, place, distance, count, heads[k], distance[at], weights[k], walk, k);
    }
    for (uint32_t x = 0; x < targets; x++) lags[(size_t)s * targets + x] = distance[target[x]];
  }
  free(heads); free(weights); free(distance); free(heap); free(place); free(wanted);
  return result;
}
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint64_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint64_t *lags) {
  return lags_walks(vertices, first, head, lag, context, sources, source, targets, target, lags, NULL);
}

/* design/compilation-composition-change-request.md item 6.2: vertex contraction by mesh_graph_lags over the graph with
   each kept vertex v split into its tail v and head vertices + v, so a walk leaves a kept vertex and ends at the first
   kept vertex it reaches, its interior all contracted */
struct contract_graph { const struct mesh_graph_dependence *edge; const size_t *order; const unsigned char *kept;
                        uint32_t vertices; };
static uint32_t contract_tail(const void *context, size_t e) {
  return ((const struct mesh_graph_dependence *)context)[e].from;
}
static uint32_t contract_head(const void *context, size_t k) {
  const struct contract_graph *g = context;
  const uint32_t to = g->edge[g->order[k]].to;
  return g->kept[to] ? g->vertices + to : to;
}
static uint64_t contract_lag(const void *context, size_t k) {
  const struct contract_graph *g = context;
  return g->edge[g->order[k]].delay;
}
/* mesh_graph_contract's one solve: kept vertex v a source and, as vertices + v, a target; with `walks`, each source's
   selected walks (lags_walks) */
struct contract_solved { uint32_t keeps; size_t *first, *order, *via; uint32_t *source, *target; uint64_t *lags; };
static void contract_free(struct contract_solved *c) {
  free(c->first); free(c->order); free(c->via); free(c->source); free(c->target); free(c->lags);
}
static int contract_solve(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                          const unsigned char *kept, int walks, struct contract_solved *c) {
  memset(c, 0, sizeof *c);
  for (size_t e = 0; e < edges; e++) if (edge[e].from >= vertices || edge[e].to >= vertices) return EINVAL;
  if (vertices >= UINT32_MAX / 2 || edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  for (uint32_t v = 0; v < vertices; v++) c->keeps += kept[v] != 0;
  const uint32_t keeps = c->keeps;
  if ((size_t)keeps * keeps >= SIZE_MAX / sizeof(uint64_t)) return ENOMEM;
  if (walks && keeps && 2 * (size_t)vertices >= SIZE_MAX / sizeof(size_t) / keeps) return ENOMEM;
  c->first = malloc((2 * (size_t)vertices + 1) * sizeof *c->first); c->order = malloc((edges + 1) * sizeof *c->order);
  c->source = malloc(((size_t)keeps + 1) * sizeof *c->source); c->target = malloc(((size_t)keeps + 1) * sizeof *c->target);
  c->lags = malloc(((size_t)keeps * keeps + 1) * sizeof *c->lags);
  if (walks) c->via = malloc(((size_t)keeps * 2 * vertices + 1) * sizeof *c->via);
  int result = c->first && c->order && c->source && c->target && c->lags && (!walks || c->via) ? 0 : ENOMEM;
  for (uint32_t v = 0, k = 0; !result && v < vertices; v++)
    if (kept[v]) { c->source[k] = v; c->target[k++] = vertices + v; }
  const struct contract_graph g = {edge, c->order, kept, vertices};
  if (!result) {
    mesh_graph_rows(2 * vertices, edges, contract_tail, edge, c->first, c->order);
    if (keeps) result = lags_walks(2 * vertices, c->first, contract_head, contract_lag, &g, keeps, c->source, keeps,
                                   c->target, c->lags, c->via);
  }
  if (result) contract_free(c);
  return result;
}
int mesh_graph_contract(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                        const unsigned char *kept, struct mesh_graph_dependence **contracted, size_t *made) {
  *contracted = NULL; *made = 0;
  struct contract_solved c;
  int result = contract_solve(vertices, edges, edge, kept, 0, &c);
  if (result) return result;
  const uint32_t keeps = c.keeps;
  size_t count = 0, at = 0;
  for (size_t p = 0; p < (size_t)keeps * keeps; p++) count += c.lags[p] != UINT64_MAX;
  if (count >= SIZE_MAX / sizeof(struct mesh_graph_dependence)) result = ENOMEM;
  struct mesh_graph_dependence *relation = result ? NULL : malloc((count + 1) * sizeof *relation);
  if (!result && !relation) result = ENOMEM;
  for (size_t p = 0; !result && p < (size_t)keeps * keeps; p++)
    if (c.lags[p] != UINT64_MAX)
      relation[at++] = (struct mesh_graph_dependence){c.source[p / keeps], c.source[p % keeps], c.lags[p]};
  if (!result) { *contracted = relation; *made = count; relation = NULL; }
  free(relation); contract_free(&c);
  return result;
}
/* design/algorithm-sources.md#relations-over-copies-and-classes; M. Lam, PLDI 1988, §2.3 */
int mesh_graph_contract_unrolled(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                                 const unsigned char *kept, uint32_t copies, struct mesh_graph_dependence **contracted,
                                 size_t *made) {
  *contracted = NULL; *made = 0;
  for (size_t e = 0; e < edges; e++) if (edge[e].from >= vertices || edge[e].to >= vertices) return EINVAL;
  if ((uint64_t)vertices * copies >= UINT32_MAX / 2 || (copies && edges >= SIZE_MAX / sizeof *edge / copies)) return ENOMEM;
  size_t count = 0, at = 0;
  for (size_t e = 0; e < edges; e++) if (edge[e].delay < copies) count += copies - (size_t)edge[e].delay;
  struct mesh_graph_dependence *unrolled = malloc((count + 1) * sizeof *unrolled);
  unsigned char *keep = malloc((size_t)vertices * copies + 1);
  int result = unrolled && keep ? 0 : ENOMEM;
  for (size_t e = 0; !result && e < edges; e++)
    for (uint64_t c = edge[e].delay; c < copies; c++)
      unrolled[at++] = (struct mesh_graph_dependence){(uint32_t)(c - edge[e].delay) * vertices + edge[e].from,
                                                      (uint32_t)c * vertices + edge[e].to, 0};
  for (uint32_t c = 0; !result && c < copies; c++) memcpy(keep + (size_t)c * vertices, kept, vertices);
  if (!result) result = mesh_graph_contract(vertices * copies, count, unrolled, keep, contracted, made);
  free(unrolled); free(keep);
  return result;
}
/* design/algorithm-sources.md#relations-over-copies-and-classes;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §4.1 */
struct quotient_pair { uint64_t key; size_t edge; };
static int quotient_before(const void *a, const void *b) {
  const struct quotient_pair *x = a, *y = b;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  return (x->edge > y->edge) - (x->edge < y->edge);
}
/* mesh_graph_quotient's one solve: each edge between two classes, (class of from, class of to) and the edge, sorted */
static int quotient_solve(uint32_t vertices, const uint32_t *class_of, size_t edges,
                          uint32_t (*tail)(const void *context, size_t e), uint32_t (*head)(const void *context, size_t e),
                          const void *context, struct quotient_pair **made, size_t *pairs) {
  *made = NULL; *pairs = 0;
  for (size_t e = 0; e < edges; e++) if (tail(context, e) >= vertices || head(context, e) >= vertices) return EINVAL;
  if (edges >= SIZE_MAX / sizeof(struct quotient_pair)) return ENOMEM;
  struct quotient_pair *pair = malloc((edges + 1) * sizeof *pair);
  if (!pair) return ENOMEM;
  size_t n = 0;
  for (size_t e = 0; e < edges; e++) {
    const uint32_t a = class_of[tail(context, e)], b = class_of[head(context, e)];
    if (a != b) pair[n++] = (struct quotient_pair){(uint64_t)a << 32 | b, e};
  }
  if (n) qsort(pair, n, sizeof *pair, quotient_before);
  *made = pair; *pairs = n;
  return 0;
}
struct quotient_arrays { const uint32_t *from, *to; };
static uint32_t quotient_tail(const void *context, size_t e) { return ((const struct quotient_arrays *)context)->from[e]; }
static uint32_t quotient_head(const void *context, size_t e) { return ((const struct quotient_arrays *)context)->to[e]; }
int mesh_graph_quotient(uint32_t vertices, const uint32_t *class_of, size_t edges, const uint32_t *from, const uint32_t *to,
                        struct mesh_graph_dependence **made, size_t *count) {
  *made = NULL; *count = 0;
  const struct quotient_arrays arrays = {from, to};
  struct quotient_pair *pair;
  size_t pairs, kept = 0;
  int result = quotient_solve(vertices, class_of, edges, quotient_tail, quotient_head, &arrays, &pair, &pairs);
  if (result) return result;
  for (size_t k = 0; k < pairs; k++)
    if (!kept || pair[k].key != pair[kept - 1].key) pair[kept++] = pair[k];
  struct mesh_graph_dependence *relation =
      kept < SIZE_MAX / sizeof(struct mesh_graph_dependence) ? malloc((kept + 1) * sizeof *relation) : NULL;
  if (!relation) { free(pair); return ENOMEM; }
  for (size_t k = 0; k < kept; k++)
    relation[k] = (struct mesh_graph_dependence){(uint32_t)(pair[k].key >> 32), (uint32_t)pair[k].key, 0};
  free(pair);
  *made = relation; *count = kept;
  return 0;
}

static int synchronization_solve(uint32_t vertices, size_t edges, uint32_t (*tail)(const void *context, size_t e),
                                 uint32_t (*head)(const void *context, size_t e), const void *context,
                                 unsigned char *redundant, size_t *kept);
static uint32_t view_hazard(const struct mesh_graph_view *from, const struct mesh_graph_view *to);
static int view_conflicts(size_t views, const struct mesh_graph_view *view, size_t supplied,
                          const uint32_t *supplied_from, const uint32_t *supplied_to, int self,
                          int (*decide)(void *context, size_t a, size_t b, struct mesh_graph_decided *out),
                          void *context, struct mesh_graph_conflict **conflict, struct mesh_graph_origin **origin,
                          struct mesh_graph_relation ***decider, const uint32_t ***decider_map, size_t *deciders,
                          size_t *edges);
/* design/compilation-composition-change-request.md items 6, 6.4 and 7: mesh-graph.h struct mesh_graph_relation */
struct mesh_graph_relation {
  size_t references;
  struct mesh_graph_relation_data data;
  void (*release)(void *);
  void *owner;
  struct mesh_graph_relation *parent;
  struct mesh_graph_dependence *edge;
  size_t *first, *member, *row_first, *row_edge, *in_first, *in_edge, *source_first, *source;
  pthread_mutex_t lock;
  int synchronized;
  unsigned char *redundant;
  size_t kept;
  struct mesh_graph_conflict *owned;
  size_t *witness, witnesses;
  /* mesh_graph_relation_decided: each record's deciding original; every relation that answered a pair, retained, in
     the order first answered, with its map; this query's own map */
  struct mesh_graph_origin *origin;
  struct mesh_graph_relation **decider;
  size_t deciders;
  const uint32_t **decider_map;
  const uint32_t *map;
  /* the synchronization mask returned (its own `redundant`, or a loaded archive's); `loaded`: edge, first, member,
     row_first, row_edge, in_first, in_edge, source_first, source and witness are borrowed from mesh_graph_relation_load's
     archive */
  const unsigned char *mask;
  int loaded;
  /* mesh_graph_relation_newest: the iteration's vertices and the iterations it folds (newest_lag 0: no newest
     projection), retained and archived with it */
  uint32_t newest_count, newest_lag;
};
static struct mesh_graph_relation *relation_new(void) {
  struct mesh_graph_relation *r = calloc(1, sizeof *r);
  if (!r) return NULL;
  if (pthread_mutex_init(&r->lock, NULL)) { free(r); return NULL; }
  r->references = 1;
  return r;
}
static void relation_drop(struct mesh_graph_relation *r) {
  if (!r->loaded) {
    free(r->edge); free(r->first); free(r->member); free(r->row_first); free(r->row_edge); free(r->in_first);
    free(r->in_edge); free(r->source_first); free(r->source); free(r->witness); free(r->origin);
  }
  free(r->redundant); free(r->owned);
  for (size_t d = 0; d < r->deciders; d++) mesh_graph_relation_free(r->decider[d]);
  free(r->decider); free(r->decider_map);
  pthread_mutex_destroy(&r->lock);
  free(r);
}
void mesh_graph_relation_retain(struct mesh_graph_relation *r) { __atomic_fetch_add(&r->references, 1, __ATOMIC_RELAXED); }
void mesh_graph_relation_free(struct mesh_graph_relation *r) {
  while (r && __atomic_fetch_sub(&r->references, 1, __ATOMIC_ACQ_REL) == 1) {
    struct mesh_graph_relation *parent = r->parent;
    void (*release)(void *) = r->release;
    void *owner = r->owner;
    relation_drop(r);
    if (release) release(owner);
    r = parent;
  }
}
const struct mesh_graph_relation_data *mesh_graph_relation_data(const struct mesh_graph_relation *r) { return &r->data; }
static uint32_t relation_tail(const void *context, size_t e);
static uint32_t relation_head(const void *context, size_t e);
/* the edges, in any order, by tail and by head (mesh_graph_rows, each row in edge order), the same edge identities */
static int relation_rows(struct mesh_graph_relation *r) {
  const uint32_t vertices = r->data.vertices;
  const size_t edges = r->data.edges;
  r->row_first = malloc(((size_t)vertices + 1) * sizeof *r->row_first);
  r->row_edge = malloc((edges + 1) * sizeof *r->row_edge);
  r->in_first = malloc(((size_t)vertices + 1) * sizeof *r->in_first);
  r->in_edge = malloc((edges + 1) * sizeof *r->in_edge);
  if (!r->row_first || !r->row_edge || !r->in_first || !r->in_edge) return ENOMEM;
  mesh_graph_rows(vertices, edges, relation_tail, r->edge, r->row_first, r->row_edge);
  mesh_graph_rows(vertices, edges, relation_head, r->edge, r->in_first, r->in_edge);
  r->data.row_first = r->row_first; r->data.row_edge = r->row_edge;
  r->data.in_first = r->in_first; r->data.in_edge = r->in_edge;
  return 0;
}
/* a derived edge's members: its source edges' members, in source order */
static int relation_compose(struct mesh_graph_relation *r) {
  const struct mesh_graph_relation_data *up = &r->parent->data;
  const size_t edges = r->data.edges;
  size_t total = 0;
  for (size_t k = 0; k < r->source_first[edges]; k++) {
    const size_t e = r->source[k], n = up->first[e + 1] - up->first[e];
    if (n > SIZE_MAX / sizeof(size_t) - 1 - total) return ENOMEM;
    total += n;
  }
  r->first = malloc((edges + 1) * sizeof *r->first);
  r->member = malloc((total + 1) * sizeof *r->member);
  if (!r->first || !r->member) return ENOMEM;
  size_t at = 0;
  for (size_t e = 0; e < edges; e++) {
    r->first[e] = at;
    for (size_t k = r->source_first[e]; k < r->source_first[e + 1]; k++)
      for (size_t m = up->first[r->source[k]]; m < up->first[r->source[k] + 1]; m++) r->member[at++] = up->member[m];
  }
  r->first[edges] = at;
  r->data.first = r->first; r->data.member = r->member;
  return 0;
}
static void relation_derived(struct mesh_graph_relation *r, struct mesh_graph_relation *parent, uint32_t vertices,
                             size_t edges) {
  r->parent = parent;
  r->data.vertices = vertices;
  r->data.views = parent->data.views; r->data.view = parent->data.view;
  r->data.conflicts = parent->data.conflicts; r->data.conflict = parent->data.conflict;
  r->data.edges = edges; r->data.edge = r->edge;
  r->data.parent = parent;
  r->data.source_first = r->source_first; r->data.source = r->source;
  r->data.query = parent->data.query;
  r->data.origin = parent->data.origin;
}
struct relation_key { uint32_t from, to; size_t index; };
static int relation_key_before(const void *a, const void *b) {
  const struct relation_key *x = a, *y = b;
  if (x->from != y->from) return x->from < y->from ? -1 : 1;
  if (x->to != y->to) return x->to < y->to ? -1 : 1;
  return (x->index > y->index) - (x->index < y->index);
}
/* the root's original records with view witnesses (GIVEN excluded) ordered by (from_view, to_view, record): `index`
   the original record */
struct witness_key { size_t from_view, to_view, index; };
static int witness_key_before(const void *a, const void *b) {
  const struct witness_key *x = a, *y = b;
  if (x->from_view != y->from_view) return x->from_view < y->from_view ? -1 : 1;
  if (x->to_view != y->to_view) return x->to_view < y->to_view ? -1 : 1;
  return (x->index > y->index) - (x->index < y->index);
}
static int relation_root(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t conflicts,
                         const struct mesh_graph_conflict *conflict, uint32_t query, void (*release)(void *),
                         void *owner, struct mesh_graph_relation **out) {
  *out = NULL;
  for (size_t k = 0; k < conflicts; k++) if (conflict[k].from >= vertices || conflict[k].to >= vertices) return EINVAL;
  if (conflicts >= SIZE_MAX / sizeof(struct relation_key) - 1) return ENOMEM;
  struct mesh_graph_relation *r = relation_new();
  struct relation_key *key = malloc((conflicts + 1) * sizeof *key);
  int result = r && key ? 0 : ENOMEM;
  size_t n = 0, edges = 0;
  for (size_t k = 0; !result && k < conflicts; k++)
    if (conflict[k].from != conflict[k].to) key[n++] = (struct relation_key){conflict[k].from, conflict[k].to, k};
  if (!result && n) qsort(key, n, sizeof *key, relation_key_before);
  if (!result) {
    r->edge = malloc((n + 1) * sizeof *r->edge);
    r->first = malloc((n + 2) * sizeof *r->first);
    r->member = malloc((n + 1) * sizeof *r->member);
    if (!r->edge || !r->first || !r->member) result = ENOMEM;
  }
  for (size_t i = 0; !result && i < n; i++) {
    if (!i || key[i].from != key[i - 1].from || key[i].to != key[i - 1].to) {
      r->first[edges] = i;
      r->edge[edges++] = (struct mesh_graph_dependence){key[i].from, key[i].to, 0};
    }
    r->member[i] = key[i].index;
  }
  free(key);
  if (!result) {
    r->first[edges] = n;
    r->data.vertices = vertices; r->data.views = views; r->data.view = view;
    r->data.conflicts = conflicts; r->data.conflict = conflict;
    r->data.edges = edges; r->data.edge = r->edge; r->data.first = r->first; r->data.member = r->member;
    r->data.query = query;
    result = relation_rows(r);
  }
  if (!result && query != MESH_GRAPH_QUERY_NONE) {
    r->witness = malloc((conflicts + 1) * sizeof *r->witness);
    struct witness_key *order = malloc((conflicts + 1) * sizeof *order);
    if (!r->witness || !order) result = ENOMEM;
    for (size_t k = 0; !result && k < conflicts; k++)
      if (conflict[k].kind != MESH_GRAPH_GIVEN)
        order[r->witnesses++] = (struct witness_key){conflict[k].from_view, conflict[k].to_view, k};
    if (!result && r->witnesses) qsort(order, r->witnesses, sizeof *order, witness_key_before);
    for (size_t i = 0; !result && i < r->witnesses; i++) r->witness[i] = order[i].index;
    free(order);
  }
  if (result) { if (r) relation_drop(r); return result; }
  r->release = release; r->owner = owner;
  *out = r;
  return 0;
}
int mesh_graph_relation_create(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t conflicts,
                               const struct mesh_graph_conflict *conflict, void (*release)(void *), void *owner,
                               struct mesh_graph_relation **out) {
  return relation_root(vertices, views, view, conflicts, conflict, MESH_GRAPH_QUERY_NONE, release, owner, out);
}
int mesh_graph_relation_views(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t supplied,
                              const uint32_t *from, const uint32_t *to, int self, void (*release)(void *), void *owner,
                              struct mesh_graph_relation **out) {
  return mesh_graph_relation_decided(vertices, views, view, supplied, from, to, self, NULL, NULL, NULL, release, owner,
                                     out);
}
int mesh_graph_relation_decided(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t supplied,
                                const uint32_t *from, const uint32_t *to, int self, const uint32_t *map,
                                int (*decide)(void *context, size_t a, size_t b, struct mesh_graph_decided *out),
                                void *context, void (*release)(void *), void *owner, struct mesh_graph_relation **out) {
  *out = NULL;
  struct mesh_graph_conflict *conflict = NULL;
  struct mesh_graph_origin *origin = NULL;
  struct mesh_graph_relation **decider = NULL;
  const uint32_t **decider_map = NULL;
  size_t count = 0, deciders = 0;
  int result = view_conflicts(views, view, supplied, from, to, self != 0, decide, context, &conflict,
                              decide ? &origin : NULL, &decider, &decider_map, &deciders, &count);
  if (!result) result = relation_root(vertices, views, view, count, conflict,
                                      self ? MESH_GRAPH_QUERY_SELF : MESH_GRAPH_QUERY_DIRECTED, release, owner, out);
  if (result) { free(conflict); free(origin); free(decider); free(decider_map); return result; }
  (*out)->owned = conflict;
  (*out)->origin = origin;
  (*out)->data.origin = origin;
  for (size_t d = 0; d < deciders; d++) mesh_graph_relation_retain(decider[d]);
  (*out)->decider = decider;
  (*out)->decider_map = decider_map;
  (*out)->deciders = deciders;
  (*out)->map = map;
  return 0;
}
size_t mesh_graph_relation_deciders(const struct mesh_graph_relation *r, struct mesh_graph_relation *const **relation,
                                    const uint32_t *const **maps, const uint32_t **map) {
  while (r->parent) r = r->parent;
  *relation = r->decider;
  *maps = r->decider_map;
  *map = r->map;
  return r->deciders;
}
int mesh_graph_relation_convex(const struct mesh_graph_relation *r, size_t members, const uint32_t *member, size_t copies,
                               const uint32_t *copy, int *convex) {
  const struct mesh_graph_relation_data *d = &r->data;
  *convex = 1;
  for (size_t k = 0; k < members; k++) if (member[k] >= d->vertices) return EINVAL;
  for (size_t k = 0; k < copies; k++) if (copy[k] >= d->vertices) return EINVAL;
  unsigned char *role = calloc((size_t)d->vertices + 1, 1);
  uint32_t *root = malloc((d->edges + 1) * sizeof *root);
  if (!role || !root) { free(role); free(root); return ENOMEM; }
  for (size_t k = 0; k < copies; k++) role[copy[k]] = 2;
  for (size_t k = 0; k < members; k++) role[member[k]] = 1;
  uint32_t roots = 0;
  for (size_t k = 0; k < members; k++)
    for (size_t q = d->row_first[member[k]]; q < d->row_first[member[k] + 1]; q++) {
      const uint32_t head = d->edge[d->row_edge[q]].to;
      if (!role[head]) { role[head] = 3; root[roots++] = head; }
    }
  int result = 0;
  if (roots) {
    struct mesh_graph_search *s = NULL;
    result = mesh_graph_search(d->vertices, d->row_first, mesh_graph_relation_head, d, roots, root, 0, &s);
    for (size_t k = 0; !result && k < members; k++) if (s->place[member[k]] != UINT32_MAX) *convex = 0;
    mesh_graph_search_free(s);
  }
  free(role); free(root);
  return result;
}
static const struct mesh_graph_relation *relation_of_root(const struct mesh_graph_relation *r) {
  while (r->parent) r = r->parent;
  return r;
}
/* the root record witnessing views (a, b) in that orientation, SIZE_MAX for none */
static size_t witness_find(const struct mesh_graph_relation *root, size_t a, size_t b) {
  size_t lo = 0, hi = root->witnesses;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    const struct mesh_graph_conflict *x = root->data.conflict + root->witness[mid];
    if (x->from_view < a || (x->from_view == a && x->to_view < b)) lo = mid + 1;
    else hi = mid;
  }
  if (lo < root->witnesses) {
    const struct mesh_graph_conflict *x = root->data.conflict + root->witness[lo];
    if (x->from_view == a && x->to_view == b) return root->witness[lo];
  }
  return SIZE_MAX;
}
int mesh_graph_relation_witness(const struct mesh_graph_relation *r, size_t from_view, size_t to_view,
                                struct mesh_graph_witness *out) {
  const struct mesh_graph_relation *root = relation_of_root(r);
  if (from_view >= root->data.views || to_view >= root->data.views) return EINVAL;
  *out = (struct mesh_graph_witness){SIZE_MAX, 0, 0};
  const uint32_t query = root->data.query;
  if (query == MESH_GRAPH_QUERY_NONE ||
      (query == MESH_GRAPH_QUERY_DIRECTED && root->data.view[from_view].vertex == root->data.view[to_view].vertex))
    return 0;
  out->known = 1;
  if ((out->record = witness_find(root, from_view, to_view)) != SIZE_MAX) return 0;
  if ((out->record = witness_find(root, to_view, from_view)) != SIZE_MAX) out->reverse = 1;
  return 0;
}
int mesh_graph_relation_cause(const struct mesh_graph_relation *r, const struct mesh_graph_witness *witness,
                              struct mesh_graph_cause *out) {
  const struct mesh_graph_relation *root = relation_of_root(r);
  if (!witness->known) return ENODATA;
  if (witness->record == SIZE_MAX) return ENOENT;
  if (witness->record >= root->data.conflicts) return EINVAL;
  const struct mesh_graph_conflict *x = root->data.conflict + witness->record;
  const size_t from = witness->reverse ? x->to_view : x->from_view, to = witness->reverse ? x->from_view : x->to_view;
  *out = (struct mesh_graph_cause){x->key, view_hazard(root->data.view + from, root->data.view + to), x->given, from, to};
  return 0;
}
static uint32_t relation_tail(const void *context, size_t e) {
  return ((const struct mesh_graph_dependence *)context)[e].from;
}
static uint32_t relation_head(const void *context, size_t e) {
  return ((const struct mesh_graph_dependence *)context)[e].to;
}
int mesh_graph_relation_quotient(struct mesh_graph_relation *parent, uint32_t classes, const uint32_t *class_of,
                                 void (*release)(void *), void *owner, struct mesh_graph_relation **out) {
  *out = NULL;
  const struct mesh_graph_relation_data *up = &parent->data;
  for (uint32_t v = 0; v < up->vertices; v++) if (class_of[v] >= classes) return EINVAL;
  struct quotient_pair *pair;
  size_t pairs, edges = 0;
  int result = quotient_solve(up->vertices, class_of, up->edges, relation_tail, relation_head, up->edge, &pair, &pairs);
  if (result) return result;
  struct mesh_graph_relation *r = relation_new();
  if (!r) result = ENOMEM;
  if (!result) {
    r->edge = malloc((pairs + 1) * sizeof *r->edge);
    r->source_first = malloc((pairs + 2) * sizeof *r->source_first);
    r->source = malloc((pairs + 1) * sizeof *r->source);
    if (!r->edge || !r->source_first || !r->source) result = ENOMEM;
  }
  for (size_t i = 0; !result && i < pairs; i++) {
    if (!i || pair[i].key != pair[i - 1].key) {
      r->source_first[edges] = i;
      r->edge[edges++] = (struct mesh_graph_dependence){(uint32_t)(pair[i].key >> 32), (uint32_t)pair[i].key, 0};
    }
    r->source[i] = pair[i].edge;
  }
  free(pair);
  if (!result) {
    r->source_first[edges] = pairs;
    relation_derived(r, parent, classes, edges);
    r->data.class_of = class_of;
    result = relation_compose(r);
    if (!result) result = relation_rows(r);
  }
  if (result) { if (r) { r->parent = NULL; relation_drop(r); } return result; }
  mesh_graph_relation_retain(parent);
  r->release = release; r->owner = owner;
  *out = r;
  return 0;
}
int mesh_graph_relation_contract(struct mesh_graph_relation *parent, const unsigned char *kept, void (*release)(void *),
                                 void *owner, struct mesh_graph_relation **out) {
  *out = NULL;
  const struct mesh_graph_relation_data *up = &parent->data;
  struct contract_solved c;
  int result = contract_solve(up->vertices, up->edges, up->edge, kept, 1, &c);
  if (result) return result;
  const uint32_t keeps = c.keeps, vertices = 2 * up->vertices;
  size_t edges = 0, walked = 0;
  for (size_t q = 0; q < (size_t)keeps * keeps; q++) {
    if (c.lags[q] == UINT64_MAX) continue;
    edges++;
    const size_t *walk = c.via + (q / keeps) * vertices;
    for (uint32_t v = c.target[q % keeps];; walked++) {
      const uint32_t tail = up->edge[c.order[walk[v]]].from;
      if (tail == c.source[q / keeps]) { walked++; break; }
      v = tail;
    }
  }
  struct mesh_graph_relation *r = relation_new();
  if (!r) result = ENOMEM;
  if (!result) {
    r->edge = malloc((edges + 1) * sizeof *r->edge);
    r->source_first = malloc((edges + 1) * sizeof *r->source_first);
    r->source = malloc((walked + 1) * sizeof *r->source);
    if (!r->edge || !r->source_first || !r->source) result = ENOMEM;
  }
  size_t e = 0, at = 0;
  for (size_t q = 0; !result && q < (size_t)keeps * keeps; q++) {
    if (c.lags[q] == UINT64_MAX) continue;
    r->source_first[e] = at;
    r->edge[e++] = (struct mesh_graph_dependence){c.source[q / keeps], c.source[q % keeps], c.lags[q]};
    const size_t *walk = c.via + (q / keeps) * vertices, begin = at;
    for (uint32_t v = c.target[q % keeps];;) {
      const size_t parentEdge = c.order[walk[v]];
      r->source[at++] = parentEdge;
      const uint32_t tail = up->edge[parentEdge].from;
      if (tail == c.source[q / keeps]) break;
      v = tail;
    }
    for (size_t i = begin, j = at - 1; i < j; i++, j--) {
      const size_t swapped = r->source[i];
      r->source[i] = r->source[j];
      r->source[j] = swapped;
    }
  }
  contract_free(&c);
  if (!result) {
    r->source_first[edges] = at;
    relation_derived(r, parent, up->vertices, edges);
    r->data.kept = kept;
    result = relation_compose(r);
    if (!result) result = relation_rows(r);
  }
  if (result) { if (r) { r->parent = NULL; relation_drop(r); } return result; }
  mesh_graph_relation_retain(parent);
  r->release = release; r->owner = owner;
  *out = r;
  return 0;
}
/* mesh-graph.h mesh_graph_relation_newest: the folded edges in the parent's edge order, each its parent edge's */
int mesh_graph_relation_newest(struct mesh_graph_relation *parent, uint32_t count, uint32_t lag,
                               struct mesh_graph_relation **out) {
  *out = NULL;
  const struct mesh_graph_relation_data *up = &parent->data;
  if (!count || !lag || up->vertices > (uint64_t)lag * count) return EINVAL;
  const uint64_t newest = (uint64_t)(lag - 1) * count;
  size_t edges = 0;
  for (size_t e = 0; e < up->edges; e++) edges += up->edge[e].to >= newest;
  struct mesh_graph_relation *r = relation_new();
  int result = r ? 0 : ENOMEM;
  if (!result) {
    r->edge = malloc((edges + 1) * sizeof *r->edge);
    r->source_first = malloc((edges + 1) * sizeof *r->source_first);
    r->source = malloc((edges + 1) * sizeof *r->source);
    if (!r->edge || !r->source_first || !r->source) result = ENOMEM;
  }
  for (size_t e = 0, k = 0; !result && e < up->edges; e++) {
    const struct mesh_graph_dependence x = up->edge[e];
    if (x.to < newest) continue;
    r->edge[k] = (struct mesh_graph_dependence){x.from % count, (uint32_t)(x.to - newest), x.delay + lag - 1 - x.from / count};
    r->source_first[k] = k; r->source[k] = e;
    k++;
  }
  if (!result) {
    r->source_first[edges] = edges;
    relation_derived(r, parent, count, edges);
    r->newest_count = count; r->newest_lag = lag;
    result = relation_compose(r);
    if (!result) result = relation_rows(r);
  }
  if (result) { if (r) { r->parent = NULL; relation_drop(r); } return result; }
  mesh_graph_relation_retain(parent);
  *out = r;
  return 0;
}
int mesh_graph_relation_newest_operands(const struct mesh_graph_relation *r, uint32_t *count, uint32_t *lag) {
  if (!r->newest_lag) return ENOENT;
  *count = r->newest_count; *lag = r->newest_lag;
  return 0;
}
int mesh_graph_relation_synchronization(struct mesh_graph_relation *r, const unsigned char **redundant, size_t *kept) {
  pthread_mutex_lock(&r->lock);
  int result = 0;
  if (!r->synchronized) {
    unsigned char *mask = malloc(r->data.edges + 1);
    result = mask ? synchronization_solve(r->data.vertices, r->data.edges, relation_tail, relation_head, r->data.edge,
                                          mask, &r->kept)
                  : ENOMEM;
    if (!result) { r->redundant = mask; r->mask = mask; r->synchronized = 1; }
    else free(mask);
  }
  if (!result) { *redundant = r->mask; *kept = r->kept; }
  pthread_mutex_unlock(&r->lock);
  return result;
}

/* design/plan-to-bridge-change-request.md R2; design/disjoint-pages-change-request.md, A3 and C3.3 */
struct chains_net {
  uint32_t items;
  const uint64_t *count;
  const size_t *first, *into_first, *into;
  const uint32_t *head, *tail;
  uint64_t *flow, *out, *in;
};
static size_t chains_arcs(const struct chains_net *n, size_t u) {
  const size_t items = n->items;
  if (u < items) return n->first[u + 1] - n->first[u];
  if (u < 2 * items) return 1 + n->into_first[u - items + 1] - n->into_first[u - items];
  return u == 2 * items ? items : 0;
}
static uint64_t chains_arc(const struct chains_net *n, size_t u, size_t a, size_t *to) {
  const size_t items = n->items;
  if (u < items) { *to = items + n->head[n->first[u] + a]; return UINT64_MAX; }
  if (u == 2 * items) { *to = a; return n->count[a] - n->out[a]; }
  const size_t j = u - items;
  if (!a) { *to = 2 * items + 1; return n->count[j] - n->in[j]; }
  const size_t e = n->into[n->into_first[j] + a - 1];
  *to = n->tail[e];
  return n->flow[e];
}
static void chains_push(struct chains_net *n, size_t u, size_t a, uint64_t delta) {
  const size_t items = n->items;
  if (u < items) n->flow[n->first[u] + a] += delta;
  else if (u == 2 * items) n->out[a] += delta;
  else if (!a) n->in[u - items] += delta;
  else n->flow[n->into[n->into_first[u - items] + a - 1]] -= delta;
}
/* E. A. Dinic, Soviet Math. Doklady 11, 1970 */
static int chains_layer(const struct chains_net *n, size_t *level, size_t *queue) {
  const size_t vertices = 2 * (size_t)n->items + 2, sink = vertices - 1;
  for (size_t v = 0; v < vertices; v++) level[v] = SIZE_MAX;
  level[vertices - 2] = 0;
  queue[0] = vertices - 2;
  for (size_t at = 0, end = 1; at < end && level[sink] == SIZE_MAX; at++)
    for (size_t a = 0, arcs = chains_arcs(n, queue[at]); a < arcs; a++) {
      size_t v;
      if (!chains_arc(n, queue[at], a, &v) || level[v] != SIZE_MAX) continue;
      level[v] = level[queue[at]] + 1;
      queue[end++] = v;
    }
  return level[sink] != SIZE_MAX;
}
/* E. A. Dinic, Soviet Math. Doklady 11, 1970 */
static void chains_block(struct chains_net *n, const size_t *level, size_t *at, size_t *path) {
  const size_t vertices = 2 * (size_t)n->items + 2, sink = vertices - 1;
  for (size_t v = 0; v < vertices; v++) at[v] = 0;
  size_t depth = 1, v = 0;
  path[0] = vertices - 2;
  while (depth) {
    const size_t u = path[depth - 1];
    if (u == sink) {
      uint64_t least = UINT64_MAX;
      for (size_t k = 0; k + 1 < depth; k++) {
        const uint64_t residual = chains_arc(n, path[k], at[path[k]], &v);
        if (residual < least) least = residual;
      }
      size_t saturated = depth;
      for (size_t k = 0; k + 1 < depth; k++) {
        chains_push(n, path[k], at[path[k]], least);
        if (saturated == depth && !chains_arc(n, path[k], at[path[k]], &v)) saturated = k;
      }
      depth = saturated + 1;
      continue;
    }
    const size_t arcs = chains_arcs(n, u);
    for (; at[u] < arcs; at[u]++)
      if (chains_arc(n, u, at[u], &v) && level[v] == level[u] + 1 && (v == sink || level[v] < level[sink])) break;
    if (at[u] < arcs) path[depth++] = v;
    else if (--depth) at[path[depth - 1]]++;
  }
}
/* design/plan-to-bridge-change-request.md R2; R. P. Dilworth, Annals of Mathematics 51(1), 1950; D. R. Fulkerson, Proc.
   AMS 7(4), 1956; E. A. Dinic, Soviet Math. Doklady 11, 1970; J. E. Hopcroft and R. M. Karp, SIAM J. Comput. 2(4),
   1973 */
int mesh_graph_chains(uint32_t items, const uint64_t *count, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, uint64_t *chain, uint64_t *chains) {
  const size_t vertices = 2 * (size_t)items + 2;
  const size_t rows = (size_t)items + 1;
  size_t *first = malloc(rows * sizeof *first), *into_first = malloc(rows * sizeof *into_first);
  size_t *order = malloc(rows * sizeof *order), *into = NULL;
  size_t *level = malloc(vertices * sizeof *level), *queue = malloc(vertices * sizeof *queue);
  size_t *at = malloc(vertices * sizeof *at), *path = malloc(vertices * sizeof *path);
  uint32_t *indegree = malloc(rows * sizeof *indegree), *ordinal = malloc(rows * sizeof *ordinal), *head = NULL,
           *tail = NULL;
  uint64_t *out = calloc(rows, sizeof *out), *in = calloc(rows, sizeof *in), *flow = NULL;
  uint64_t *place = malloc(rows * sizeof *place), *taken = calloc(rows, sizeof *taken);
  int result = first && into_first && order && level && queue && at && path && indegree && ordinal && out && in &&
               place && taken ? 0 : ENOMEM;
  size_t edges = 0, room = 0;
  if (!result) first[0] = 0;
  for (uint32_t i = 0; !result && i < items; i++) {
    for (uint32_t j = 0; !result && j < items; j++) {
      if (i == j || !before(context, i, j)) continue;
      if (edges == room) {
        room = room ? 2 * room : 64;
        uint32_t *grown = realloc(head, room * sizeof *grown);
        if (grown) head = grown;
        grown = grown ? realloc(tail, room * sizeof *grown) : NULL;
        if (!grown) { result = ENOMEM; break; }
        tail = grown;
      }
      head[edges] = j; tail[edges++] = i;
    }
    first[i + 1] = edges;
  }
  if (!result) {
    flow = calloc(edges + 1, sizeof *flow);
    into = malloc((edges + 1) * sizeof *into);
    if (!flow || !into) result = ENOMEM;
  }
  if (!result) result = mesh_graph_order(items, first, depth_head, head, indegree, ordinal);
  if (!result) {
    mesh_graph_rows(items, edges, depth_head, head, into_first, into);
    struct chains_net net = {items, count, first, into_first, into, head, tail, flow, out, in};
    while (chains_layer(&net, level, queue)) chains_block(&net, level, at, path);
    uint64_t next = 0, o = 0;
    for (uint32_t i = 0; i < items; i++) {
      place[i] = o;
      for (uint64_t k = in[i]; k < count[i]; k++) chain[o + k] = next++;
      o += count[i];
    }
    for (uint32_t i = 0; i < items; i++) order[ordinal[i]] = i;
    for (uint32_t p = 0; p < items; p++) {
      const size_t i = order[p];
      uint64_t from = place[i];
      for (size_t e = first[i]; e < first[i + 1]; e++) {
        const uint32_t j = head[e];
        for (uint64_t k = 0; k < flow[e]; k++) chain[place[j] + taken[j] + k] = chain[from + k];
        taken[j] += flow[e];
        from += flow[e];
      }
    }
    *chains = next;
  }
  free(first); free(into_first); free(order); free(into); free(level); free(queue); free(at); free(path);
  free(indegree); free(ordinal); free(head); free(tail); free(out); free(in); free(place); free(taken); free(flow);
  return result;
}
/* mesh-graph.h mesh_graph_closure_chains: the closure and the members its items name */
struct closure_members { const struct mesh_graph_closure *closed; const uint32_t *member; };
static int closure_before(const void *context, uint32_t a, uint32_t b) {
  const struct closure_members *c = context;
  const uint32_t x = c->member[a], y = c->member[b];
  return (int)(c->closed->reached[(size_t)y * c->closed->words + x / 64] >> (x % 64) & 1);
}
int mesh_graph_closure_chains(const struct mesh_graph_closure *closed, uint32_t items, const uint32_t *member,
                              const uint64_t *count, uint64_t *chain, uint64_t *chains) {
  for (uint32_t k = 0; k < items; k++) if (member[k] >= closed->vertices) return EINVAL;
  const struct closure_members c = {closed, member};
  return mesh_graph_chains(items, count, closure_before, &c, chain, chains);
}

/* C. Mukerji, M.Sc. thesis, McGill University, 1994, §3.1 */
static uint32_t cover_sections(uint64_t length, uint64_t start, uint64_t end, uint64_t *from, uint64_t *to) {
  if (start <= end) { from[0] = start; to[0] = end; return 1; }
  from[0] = 0; to[0] = end; from[1] = start; to[1] = length - 1;
  return 2;
}
static int cover_meets(uint64_t length, uint64_t start, uint64_t end, uint64_t from, uint64_t to) {
  uint64_t f[2], t[2];
  for (uint32_t x = 0, n = cover_sections(length, start, end, f, t); x < n; x++)
    if (f[x] <= to && from <= t[x]) return 1;
  return 0;
}
/* C. Mukerji, M.Sc. thesis, McGill University, 1994, Definition 2.1.2 */
static int cover_conflict(uint64_t length, const uint64_t *start, const uint64_t *end, uint32_t i, uint32_t j) {
  uint64_t f[2], t[2];
  if (i == j) return 0;
  for (uint32_t x = 0, n = cover_sections(length, start[j], end[j], f, t); x < n; x++)
    if (cover_meets(length, start[i], end[i], f[x], t[x])) return 1;
  return 0;
}
struct cover_key { uint64_t at; uint32_t interval; };
static int cover_key_before(const void *a, const void *b) {
  const struct cover_key *x = a, *y = b;
  if (x->at != y->at) return x->at < y->at ? -1 : 1;
  return (x->interval > y->interval) - (x->interval < y->interval);
}
static size_t cover_find(const uint64_t *point, size_t points, uint64_t at) {
  size_t low = 0, high = points;
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    if (point[middle] < at) low = middle + 1; else high = middle;
  }
  return low;
}
/* C. Mukerji, M.Sc. thesis, McGill University, 1994, Definitions 2.3.1, 2.3.2 and 3.2.1, §3.2.3, §3.2.6, Theorem 2.3.1
   and Algorithm A.0.1 */
int mesh_graph_fat_cover(uint64_t length, uint32_t intervals, const uint64_t *start, const uint64_t *end,
                         const uint64_t *need, uint64_t *color, uint64_t *colors) {
  const size_t n = intervals, keys = 4 * n;
  uint64_t elements = 0;
  for (uint32_t i = 0; i < intervals; i++)
    if (__builtin_add_overflow(elements, need[i], &elements)) return ENOMEM;
  if (elements >= SIZE_MAX / sizeof(uint64_t)) return ENOMEM;
  struct cover_key *key = malloc((keys + 1) * sizeof *key), *by_end = malloc((n + 1) * sizeof *by_end);
  uint64_t *point = malloc((keys + 1) * sizeof *point), *width = malloc((keys + 1) * sizeof *width);
  uint64_t *spot = malloc((keys + 1) * sizeof *spot), *place = malloc((n + 1) * sizeof *place);
  uint64_t *idle = malloc(((size_t)elements + 1) * sizeof *idle), *taken = calloc(n + 1, sizeof *taken);
  unsigned char *colored = calloc(n + 1, 1), *chosen = calloc(n + 1, 1);
  size_t *row = NULL, *first = NULL, *parent = NULL, *prev = NULL, *next = NULL, *cur = NULL;
  uint32_t *live = NULL, *node_of = NULL, *picked = NULL;
  int result = key && by_end && point && width && spot && place && idle && taken && colored && chosen ? 0 : ENOMEM;
  size_t points = 0, fatspots = 0, cyclic = 0, most_row = 0;
  if (!result) {
    uint64_t o = 0;
    size_t k = 0;
    for (uint32_t i = 0; i < intervals; i++) {
      uint64_t f[2], t[2];
      place[i] = o; o += need[i];
      cyclic += end[i] < start[i];
      for (uint32_t x = 0, s = cover_sections(length, start[i], end[i], f, t); x < s; x++) {
        key[k++] = (struct cover_key){f[x], i};
        key[k++] = (struct cover_key){t[x] + 1, i};
      }
    }
    qsort(key, k, sizeof *key, cover_key_before);
    for (size_t q = 0; q < k; q++)
      if (!points || point[points - 1] != key[q].at) point[points++] = key[q].at;
    for (size_t q = 0; q < points; q++) width[q] = 0;
    for (uint32_t i = 0; i < intervals; i++) {
      uint64_t f[2], t[2];
      for (uint32_t x = 0, s = cover_sections(length, start[i], end[i], f, t); x < s; x++) {
        width[cover_find(point, points, f[x])]++;
        width[cover_find(point, points, t[x] + 1)]--;
      }
    }
    uint64_t most = 0;
    for (size_t q = 0; q + 1 < points; q++) {
      if (q) width[q] += width[q - 1];
      if (width[q] > most) most = width[q];
    }
    for (size_t q = 0; most && q + 1 < points; q++)
      if (width[q] == most) spot[fatspots++] = q;
  }
  if (!result && cyclic) {
    size_t total = 0, node_room = fatspots + 1, link_room = fatspots;
    row = calloc(fatspots + 2, sizeof *row);
    if (!row) result = ENOMEM;
    for (uint32_t i = 0; !result && i < intervals; i++) {
      if (end[i] < start[i]) continue;
      const size_t to = cover_find(point, points, end[i] + 1);
      for (size_t f = cover_find(spot, fatspots, cover_find(point, points, start[i])); f < fatspots && spot[f] < to; f++)
        row[f + 1]++;
    }
    for (size_t f = 0; !result && f < fatspots; f++) {
      const size_t r = row[f + 1], above = (f ? row[f] : 0) + 1;
      size_t links;
      if (r > most_row) most_row = r;
      if (__builtin_mul_overflow(r, above, &links) || __builtin_add_overflow(link_room, links, &link_room) ||
          __builtin_add_overflow(node_room, r, &node_room)) result = ENOMEM;
      total += r;
    }
    for (size_t f = 0; !result && f < fatspots; f++) row[f + 1] += row[f];
    if (!result && (total >= SIZE_MAX / sizeof(size_t) || node_room >= SIZE_MAX / sizeof(size_t) - 1 ||
                    link_room >= SIZE_MAX / sizeof(size_t) - 1)) result = ENOMEM;
    if (!result) {
      live = malloc((total + 1) * sizeof *live);
      node_of = malloc((node_room + 1) * sizeof *node_of);
      first = malloc((node_room + 2) * sizeof *first);
      parent = malloc((link_room + 1) * sizeof *parent);
      prev = malloc((most_row + 2) * sizeof *prev);
      next = malloc((most_row + 2) * sizeof *next);
      cur = malloc((most_row + 2) * sizeof *cur);
      picked = malloc((fatspots + 2) * sizeof *picked);
      if (!live || !node_of || !first || !parent || !prev || !next || !cur || !picked) result = ENOMEM;
    }
    if (!result) {
      size_t *at = calloc(fatspots + 1, sizeof *at);
      if (!at) result = ENOMEM;
      for (uint32_t i = 0; !result && i < intervals; i++) {
        if (end[i] < start[i]) continue;
        const size_t to = cover_find(point, points, end[i] + 1);
        for (size_t f = cover_find(spot, fatspots, cover_find(point, points, start[i])); f < fatspots && spot[f] < to; f++)
          live[row[f] + at[f]++] = i;
      }
      free(at);
    }
  }
  uint64_t made = 0;
  for (uint32_t c = 0; !result && c < intervals; c++) {
    if (start[c] <= end[c]) continue;
    size_t nodes = 0, links = 0, previous = 0, kept = 0;
    int front = 0;
    for (size_t f = 0; f < fatspots; f++) {
      const uint64_t t = point[spot[f]];
      if (t <= end[c]) {
        node_of[nodes] = c; first[nodes] = links;
        if (front) parent[links++] = prev[0];
        first[nodes + 1] = links; prev[0] = nodes++; previous = 1; front = 1;
        continue;
      }
      if (!front) { node_of[nodes] = c; first[nodes] = first[nodes + 1] = links; prev[0] = nodes++; previous = 1; front = 1; }
      size_t children = 0;
      for (size_t r = row[f]; r < row[f + 1]; r++) {
        const uint32_t i = live[r];
        const size_t from = links;
        if (colored[i]) continue;
        for (size_t q = 0; q < previous; q++)
          if (!cover_conflict(length, start, end, i, node_of[prev[q]])) parent[links++] = prev[q];
        if (links == from) continue;
        node_of[nodes] = i; first[nodes] = from; first[nodes + 1] = links; next[children++] = nodes++;
      }
      if (t >= start[c]) break;
      size_t *swap = prev; prev = next; next = swap;
      previous = children;
    }
    for (size_t q = 0; q < previous; q++) {
      const uint32_t i = node_of[prev[q]];
      if (i == c || !cover_meets(length, start[i], end[i], start[c], length - 1)) cur[kept++] = prev[q];
    }
    size_t count = kept, walked = 0, num = 0;
    uint64_t earliest = UINT64_MAX;
    while (count) {
      for (size_t q = 0; q < count; q++) {
        const uint32_t i = node_of[cur[q]];
        const uint64_t from = start[i] <= end[i] ? start[i] : 0;
        if (chosen[i]) num = cur[q];
        else if (from < earliest || (from == earliest && end[i] > end[node_of[num]])) { num = cur[q]; earliest = from; }
      }
      if (!chosen[node_of[num]]) { chosen[node_of[num]] = 1; picked[walked++] = node_of[num]; }
      count = first[num + 1] - first[num];
      for (size_t q = 0; q < count; q++) cur[q] = parent[first[num] + q];
    }
    for (size_t q = 0; q < walked; q++) {
      const uint32_t m = picked[q];
      for (; taken[m] < need[m] && taken[m] < need[c]; taken[m]++) color[place[m] + taken[m]] = made + taken[m];
      colored[m] = 1; chosen[m] = 0;
    }
    if (!kept) {
      for (; taken[c] < need[c]; taken[c]++) color[place[c] + taken[c]] = made + taken[c];
      colored[c] = 1;
    }
    made += need[c];
  }
  if (!result) {
    size_t rest = 0, idled = 0, released = 0;
    for (uint32_t i = 0; i < intervals; i++)
      if (taken[i] < need[i]) { key[rest] = (struct cover_key){start[i], i}; by_end[rest++] = (struct cover_key){end[i], i}; }
    qsort(key, rest, sizeof *key, cover_key_before);
    qsort(by_end, rest, sizeof *by_end, cover_key_before);
    for (size_t q = 0; q < rest; q++) {
      const uint32_t i = key[q].interval;
      for (; released < rest && by_end[released].at < start[i]; released++)
        for (uint64_t k = need[by_end[released].interval]; k-- > taken[by_end[released].interval];)
          idle[idled++] = color[place[by_end[released].interval] + k];
      for (uint64_t k = taken[i]; k < need[i]; k++) color[place[i] + k] = idled ? idle[--idled] : made++;
    }
    *colors = made;
  }
  free(key); free(by_end); free(point); free(width); free(spot); free(place); free(idle); free(taken); free(colored); free(chosen);
  free(row); free(first); free(parent); free(prev); free(next); free(cur); free(live); free(node_of); free(picked);
  return result;
}

/* design/plan-to-bridge-change-request.md R5, "How: the program" and "How: solving it" */
struct completion_key { double at; uint32_t vertex; };
static int completion_key_before(const void *a, const void *b) {
  const struct completion_key *x = a, *y = b;
  if (x->at != y->at) return x->at < y->at ? -1 : 1;
  return (x->vertex > y->vertex) - (x->vertex < y->vertex);
}
static int completion_edge_before(const void *a, const void *b) {
  const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}
/* design/plan-to-bridge-change-request.md R5, "How: the program" and "How: solving it"; C.-Y. Chen, arXiv 2205.02474,
   2022, (1), (1b) to (1e), (14) and (14a); L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3),
   1997, (1), (3) and (5); M. Queyranne, Math. Programming 58, 1993; Q. Huangfu and J. A. J. Hall, Math. Program.
   Comput. 10(1), 2018; H. Isermann, OR Spektrum 4, 1982; HiGHS 1.15.1, highs_c_api.h and HighsOptions.h */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion) {
  for (uint32_t j = 0; j < vertices; j++)
    if (!(length[j] >= 0 && length[j] <= DBL_MAX) || resource[j] >= resources) return EINVAL;
  for (size_t e = 0; e < edges; e++)
    if (from[e] >= vertices || to[e] >= vertices || from[e] == to[e]) return EINVAL;
  if (!vertices) return 0;
  const uint64_t room = (uint64_t)vertices + edges + resources + 1;
  if (room >= INT32_MAX / 2) return ENOMEM;
  const HighsInt columns = (HighsInt)vertices + 1;
  uint64_t *pair = malloc((edges + 1) * sizeof *pair);
  size_t *first = malloc(((size_t)resources + 1) * sizeof *first), *member = malloc(((size_t)vertices + 1) * sizeof *member);
  struct completion_key *key = malloc(((size_t)vertices + 1) * sizeof *key);
  double *lower = malloc(room * sizeof *lower), *upper = malloc(room * sizeof *upper);
  double *value = malloc(2 * room * sizeof *value), *column = malloc((size_t)columns * sizeof *column);
  double *dual = malloc((size_t)columns * sizeof *dual), *row_value = NULL, *row_dual = NULL;
  HighsInt *start = malloc(room * sizeof *start), *index = malloc(2 * room * sizeof *index);
  void *highs = pair && first && member && key && lower && upper && value && column && dual && start && index
                    ? Highs_create() : NULL;
  int result = highs ? 0 : ENOMEM;
  double tolerance = 0, infinity = 0;
  HighsInt total = 0;
  if (!result) {
    infinity = Highs_getInfinity(highs);
    if (Highs_setBoolOptionValue(highs, "output_flag", 0) == kHighsStatusError ||
        Highs_setStringOptionValue(highs, "solver", "simplex") == kHighsStatusError ||
        Highs_setIntOptionValue(highs, "simplex_strategy", 1) == kHighsStatusError ||
        Highs_setStringOptionValue(highs, "parallel", "off") == kHighsStatusError ||
        Highs_setIntOptionValue(highs, "threads", 1) == kHighsStatusError ||
        Highs_getDoubleOptionValue(highs, "primal_feasibility_tolerance", &tolerance) == kHighsStatusError ||
        Highs_changeObjectiveSense(highs, kHighsObjSenseMinimize) == kHighsStatusError)
      result = EDOM;
  }
  if (!result) {
    for (uint32_t j = 0; j < vertices; j++) { lower[j] = length[j]; upper[j] = infinity; }
    lower[vertices] = 0; upper[vertices] = infinity;
    if (Highs_addVars(highs, columns, lower, upper) == kHighsStatusError ||
        Highs_changeColCost(highs, (HighsInt)vertices, 1) == kHighsStatusError)
      result = EDOM;
  }
  if (!result) {
    HighsInt nonzeros = 0;
    for (uint32_t j = 0; j < vertices; j++) {
      start[total] = nonzeros; lower[total] = 0; upper[total] = infinity; total++;
      index[nonzeros] = (HighsInt)vertices; value[nonzeros++] = 1;
      index[nonzeros] = (HighsInt)j; value[nonzeros++] = -1;
    }
    for (size_t e = 0; e < edges; e++) pair[e] = (uint64_t)from[e] << 32 | to[e];
    qsort(pair, edges, sizeof *pair, completion_edge_before);
    for (size_t e = 0; e < edges; e++) {
      if (e && pair[e] == pair[e - 1]) continue;
      const uint32_t i = (uint32_t)(pair[e] >> 32), j = (uint32_t)pair[e];
      start[total] = nonzeros; lower[total] = length[j]; upper[total] = infinity; total++;
      index[nonzeros] = (HighsInt)j; value[nonzeros++] = 1;
      index[nonzeros] = (HighsInt)i; value[nonzeros++] = -1;
    }
    if (Highs_addRows(highs, total, lower, upper, nonzeros, start, index, value) == kHighsStatusError) result = EDOM;
    mesh_graph_rows(resources, vertices, depth_head, resource, first, member);
  }
  int second = 0;
  uint64_t entries = 2 * room;
  while (!result) {
    if (Highs_run(highs) == kHighsStatusError) { result = EDOM; break; }
    const HighsInt model = Highs_getModelStatus(highs);
    if (model == kHighsModelStatusInfeasible) { result = EINVAL; break; }
    if (model != kHighsModelStatusOptimal) { result = EDOM; break; }
    double *grown = realloc(row_value, ((size_t)total + 1) * sizeof *grown);
    if (grown) row_value = grown;
    grown = grown ? realloc(row_dual, ((size_t)total + 1) * sizeof *grown) : NULL;
    if (!grown) { result = ENOMEM; break; }
    row_dual = grown;
    if (Highs_getSolution(highs, column, dual, row_value, row_dual) == kHighsStatusError) { result = EDOM; break; }
    HighsInt cuts = 0, nonzeros = 0;
    for (uint32_t r = 0; r < resources; r++) {
      size_t count = 0;
      for (size_t k = first[r]; k < first[r + 1]; k++) {
        const uint32_t j = (uint32_t)member[k];
        if (length[j] > 0) key[count++] = (struct completion_key){column[j], j};
      }
      qsort(key, count, sizeof *key, completion_key_before);
      double sum = 0, squares = 0, left = 0;
      for (size_t k = 0; k < count && !result; k++) {
        const double d = length[key[k].vertex];
        sum += d; squares += d * d; left += d * key[k].at;
        const double right = (sum * sum + squares) / 2;
        if (!((right - left) / sum > tolerance)) continue;
        if ((uint64_t)nonzeros + k + 1 > entries) {
          const uint64_t more = 2 * ((uint64_t)nonzeros + k + 1);
          HighsInt *wider = more < INT32_MAX ? realloc(index, more * sizeof *index) : NULL;
          if (wider) index = wider;
          double *longer = wider ? realloc(value, more * sizeof *value) : NULL;
          if (!longer) { result = ENOMEM; break; }
          value = longer; entries = more;
        }
        start[cuts] = nonzeros; lower[cuts] = right / sum; upper[cuts] = infinity; cuts++;
        for (size_t q = 0; q <= k; q++) { index[nonzeros] = (HighsInt)key[q].vertex; value[nonzeros++] = length[key[q].vertex] / sum; }
      }
    }
    if (result) break;
    if (!cuts && second) break;
    if (!cuts) {
      second = 1;
      for (uint32_t j = 0; j < vertices; j++) lower[j] = 1;
      if (Highs_changeColBounds(highs, (HighsInt)vertices, column[vertices], column[vertices]) == kHighsStatusError ||
          Highs_changeColCost(highs, (HighsInt)vertices, 0) == kHighsStatusError ||
          Highs_changeColsCostByRange(highs, 0, (HighsInt)vertices - 1, lower) == kHighsStatusError)
        result = EDOM;
      continue;
    }
    if (Highs_addRows(highs, cuts, lower, upper, nonzeros, start, index, value) == kHighsStatusError) { result = EDOM; break; }
    total += cuts;
  }
  if (!result) for (uint32_t j = 0; j < vertices; j++) completion[j] = column[j];
  if (highs) Highs_destroy(highs);
  free(pair); free(first); free(member); free(key); free(lower); free(upper); free(value); free(column); free(dual);
  free(row_value); free(row_dual); free(start); free(index);
  return result;
}

/* design/unequal-shards-change-request.md items 6 and 14; S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3), "The
   objective function is the sum of the communication time and computation time of all stages"; X. Jia et al., Whale,
   USENIX ATC 2022, with HiGHS 1.15.1 README.md, "l <= x <= u": columns B[p * ranks + j] in [0, upper[j]], each part's
   largest share and each stage's largest computation time; rows each part's shares' sum, its largest share over each
   of its shares and each stage's time over each of its rows, a row's coefficients over the full share variables its
   terms couple; HiGHS 1.15.1, highs_c_api.h */
int mesh_graph_shares_coupled(uint32_t ranks, uint32_t parts, uint32_t stages, uint32_t rows, const double *comm,
                              const uint32_t *stage, const double *fixed, const size_t *first, const uint32_t *variable,
                              const double *coefficient, const double *upper, double *share) {
  const uint64_t shares = (uint64_t)parts * ranks, terms = first[rows];
  const uint64_t constraints = parts + shares + (uint64_t)rows, columns = shares + parts + stages;
  const uint64_t entries = 3 * shares + (uint64_t)rows + terms;
  if (constraints >= INT32_MAX / 2 || columns >= INT32_MAX / 2 || entries >= INT32_MAX / 2) return ENOMEM;
  for (uint32_t r = 0; r < rows; r++) if (stage[r] >= stages) return EINVAL;
  for (uint64_t k = 0; k < terms; k++) if (variable[k] >= shares) return EINVAL;
  const size_t room = constraints > columns ? constraints : columns;
  double *lower = malloc(room * sizeof *lower), *high = malloc(room * sizeof *high);
  double *value = malloc(entries * sizeof *value), *column = malloc(columns * sizeof *column);
  HighsInt *start = malloc(room * sizeof *start), *index = malloc(entries * sizeof *index);
  void *highs = lower && high && value && column && start && index ? Highs_create() : NULL;
  int result = highs ? 0 : ENOMEM;
  const double infinity = highs ? Highs_getInfinity(highs) : 0;
  if (!result) {
    for (uint64_t c = 0; c < shares; c++) { lower[c] = 0; high[c] = upper[c % ranks]; }
    for (uint32_t p = 0; p < parts; p++) { lower[shares + p] = 0; high[shares + p] = infinity; }
    for (uint32_t i = 0; i < stages; i++) { lower[shares + parts + i] = -infinity; high[shares + parts + i] = infinity; }
    if (Highs_setBoolOptionValue(highs, "output_flag", 0) == kHighsStatusError ||
        Highs_setStringOptionValue(highs, "solver", "simplex") == kHighsStatusError ||
        Highs_setIntOptionValue(highs, "simplex_strategy", 1) == kHighsStatusError ||
        Highs_setStringOptionValue(highs, "parallel", "off") == kHighsStatusError ||
        Highs_setIntOptionValue(highs, "threads", 1) == kHighsStatusError ||
        Highs_changeObjectiveSense(highs, kHighsObjSenseMinimize) == kHighsStatusError ||
        Highs_addVars(highs, (HighsInt)columns, lower, high) == kHighsStatusError)
      result = EDOM;
    for (uint32_t p = 0; p < parts && !result; p++) {
      double largest = 0;
      for (uint32_t i = 0; i < stages; i++) largest += comm[(size_t)i * parts + p];
      if (Highs_changeColCost(highs, (HighsInt)(shares + p), largest) == kHighsStatusError) result = EDOM;
    }
    for (uint32_t i = 0; i < stages && !result; i++)
      if (Highs_changeColCost(highs, (HighsInt)(shares + parts + i), 1) == kHighsStatusError) result = EDOM;
  }
  if (!result) {
    HighsInt total = 0, nonzeros = 0;
    for (uint32_t p = 0; p < parts; p++) {
      start[total] = nonzeros; lower[total] = 1; high[total] = 1; total++;
      for (uint32_t j = 0; j < ranks; j++) { index[nonzeros] = (HighsInt)((uint64_t)p * ranks + j); value[nonzeros++] = 1; }
    }
    for (uint64_t c = 0; c < shares; c++) {
      start[total] = nonzeros; lower[total] = 0; high[total] = infinity; total++;
      index[nonzeros] = (HighsInt)c; value[nonzeros++] = -1;
      index[nonzeros] = (HighsInt)(shares + c / ranks); value[nonzeros++] = 1;
    }
    for (uint32_t r = 0; r < rows; r++) {
      start[total] = nonzeros; lower[total] = fixed[r]; high[total] = infinity; total++;
      for (size_t k = first[r]; k < first[r + 1]; k++) {
        index[nonzeros] = (HighsInt)variable[k]; value[nonzeros++] = -coefficient[k];
      }
      index[nonzeros] = (HighsInt)(shares + parts + stage[r]); value[nonzeros++] = 1;
    }
    if (Highs_addRows(highs, total, lower, high, nonzeros, start, index, value) == kHighsStatusError) result = EDOM;
  }
  if (!result) {
    if (Highs_run(highs) == kHighsStatusError) result = EDOM;
    else if (Highs_getModelStatus(highs) == kHighsModelStatusInfeasible) result = EINVAL;
    else if (Highs_getModelStatus(highs) != kHighsModelStatusOptimal ||
             Highs_getSolution(highs, column, NULL, NULL, NULL) == kHighsStatusError) result = EDOM;
  }
  if (!result) for (uint64_t c = 0; c < shares; c++) share[c] = column[c];
  if (highs) Highs_destroy(highs);
  free(lower); free(high); free(value); free(column); free(start); free(index);
  return result;
}

int mesh_graph_shares(uint32_t ranks, uint32_t stages, const double *comm, const double *fixed, const double *slope,
                      const double *upper, double *share) {
  const uint64_t rows = (uint64_t)stages * ranks;
  if (rows >= INT32_MAX / 2) return ENOMEM;
  size_t *first = malloc((rows + 1) * sizeof *first);
  uint32_t *stage = malloc((rows + 1) * sizeof *stage), *variable = malloc((rows + 1) * sizeof *variable);
  int result = first && stage && variable ? 0 : ENOMEM;
  for (uint64_t r = 0; !result && r <= rows; r++) first[r] = r;
  for (uint64_t r = 0; !result && r < rows; r++) { stage[r] = (uint32_t)(r / ranks); variable[r] = (uint32_t)(r % ranks); }
  if (!result)
    result = mesh_graph_shares_coupled(ranks, 1, stages, (uint32_t)rows, comm, stage, fixed, first, variable, slope, upper,
                                       share);
  free(first); free(stage); free(variable);
  return result;
}

/* design/plan-to-bridge-change-request.md R5, "The order of each cable's list, by the reference algorithm" */
static int order_before(const void *context, uint32_t a, uint32_t b) {
  const double *priority = context;
  return priority[a] > priority[b] || (priority[a] == priority[b] && a < b);
}
/* M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2, step 3; design/plan-to-bridge-change-request.md R5; F. B. Schneider,
   ACM Comput. Surv. 1990 */
int mesh_graph_heap_order_view(const struct mesh_graph_rows_view *rows, const double *priority, uint32_t *place) {
  const uint32_t vertices = rows->vertices;
  size_t *indegree = malloc(((size_t)vertices + 1) * sizeof *indegree);
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap);
  uint32_t *position = malloc(((size_t)vertices + 1) * sizeof *position);
  int result = !indegree || !heap || !position ? ENOMEM : 0;
  for (uint32_t v = 0; !result && v < vertices; v++) { indegree[v] = 0; position[v] = UINT32_MAX; place[v] = UINT32_MAX; }
  for (uint32_t v = 0; !result && v < vertices; v++)
    for (size_t k = 0, n = rows->degree(rows->context, v); !result && k < n; k++) {
      const uint32_t head = rows->head(rows->context, v, k);
      if (head == UINT32_MAX) continue;
      if (head >= vertices) result = EINVAL;
      else indegree[head]++;
    }
  if (!result) {
    size_t count = 0;
    for (uint32_t v = 0; v < vertices; v++)
      if (!indegree[v]) { heap[count] = v; heap_up(heap, position, order_before, priority, count); count++; }
    uint32_t placed = 0;
    while (count) {
      const uint32_t v = heap[0];
      position[v] = UINT32_MAX;
      if (--count) { heap[0] = heap[count]; heap_down(heap, position, order_before, priority, count, 0); }
      place[v] = placed++;
      for (size_t k = 0, n = rows->degree(rows->context, v); k < n; k++) {
        const uint32_t head = rows->head(rows->context, v, k);
        if (head != UINT32_MAX && !--indegree[head]) {
          heap[count] = head; heap_up(heap, position, order_before, priority, count); count++;
        }
      }
    }
    if (placed < vertices) result = EINVAL;
  }
  free(indegree); free(heap); free(position);
  return result;
}

/* a flat edge list's rows by `from` (mesh_graph_rows): vertex v's entries the edges order[first[v] .. first[v + 1]) */
struct flat_rows { const size_t *first, *order; const uint32_t *to; };
static size_t flat_degree(const void *context, uint32_t v) {
  const struct flat_rows *r = context;
  return r->first[v + 1] - r->first[v];
}
static uint32_t flat_head(const void *context, uint32_t v, size_t k) {
  const struct flat_rows *r = context;
  return r->to[r->order[r->first[v] + k]];
}

int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place) {
  for (size_t e = 0; e < edges; e++) if (from[e] >= vertices || to[e] >= vertices) return EINVAL;
  if (edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  size_t *first = malloc(((size_t)vertices + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  int result = !first || !order ? ENOMEM : 0;
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, from, first, order);
    const struct flat_rows flat = {first, order, to};
    const struct mesh_graph_rows_view rows = {vertices, &flat, flat_degree, flat_head};
    result = mesh_graph_heap_order_view(&rows, priority, place);
  }
  free(first); free(order);
  return result;
}

/* M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2, step 3; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper.
   Res. 22(3), 1997, §2; design/integration-finishing-change-request.md item 2 */
int mesh_graph_heap_cuts(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from, const uint32_t *to,
                         const unsigned char *counted, uint32_t *listed, uint32_t *before, uint32_t *count) {
  uint32_t *place = malloc(((size_t)vertices + 1) * sizeof *place), *at = malloc(((size_t)vertices + 1) * sizeof *at);
  uint32_t *rank = malloc(((size_t)vertices + 1) * sizeof *rank);
  unsigned char *uncounted = malloc((size_t)vertices + 1);
  int result = place && at && rank && uncounted ? mesh_graph_heap_order(vertices, priority, edges, from, to, place) : ENOMEM;
  if (!result) {
    for (uint32_t v = 0; v < vertices; v++) uncounted[v] = !counted[v];
    mesh_graph_project(vertices, place, uncounted, rank, at, count);
    for (uint32_t k = 0; k < *count; k++) { listed[k] = at[k]; before[k] = place[at[k]] - k; }
  }
  free(place); free(at); free(rank); free(uncounted);
  return result;
}

/* design/compilation-composition-change-request.md item 6.3: a placement (place[v] the position of v, a permutation)
   restricted to the `kept` vertices */
void mesh_graph_project(uint32_t vertices, const uint32_t *place, const unsigned char *kept, uint32_t *rank, uint32_t *at,
                        uint32_t *count) {
  for (uint32_t v = 0; v < vertices; v++) at[place[v]] = v;
  uint32_t made = 0;
  for (uint32_t p = 0; p < vertices; p++) {
    const uint32_t v = at[p];
    rank[v] = made;
    if (kept[v]) at[made++] = v;
  }
  *count = made;
}

/* design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the self-timed
   buffer bound"; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997 (BSL) */
struct graph_scratch { size_t *first, *order, *at; uint32_t *index, *low, *stack, *frame, *component; double *distance; };
static void scratch_free(struct graph_scratch *s) {
  free(s->first); free(s->order); free(s->at); free(s->index); free(s->low); free(s->stack); free(s->frame);
  free(s->component); free(s->distance);
}
static int scratch_alloc(struct graph_scratch *s, uint32_t vertices, size_t edges) {
  s->first = s->order = s->at = NULL;
  s->index = s->low = s->stack = s->frame = s->component = NULL;
  s->distance = NULL;
  if (vertices == UINT32_MAX || edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  const size_t n = (size_t)vertices + 2;
  s->first = malloc(n * sizeof *s->first); s->order = malloc((edges + 1) * sizeof *s->order);
  s->at = malloc(n * sizeof *s->at); s->index = malloc(n * sizeof *s->index); s->low = malloc(n * sizeof *s->low);
  s->stack = malloc(n * sizeof *s->stack); s->frame = malloc(n * sizeof *s->frame);
  s->component = malloc(n * sizeof *s->component); s->distance = malloc(n * sizeof *s->distance);
  return s->first && s->order && s->at && s->index && s->low && s->stack && s->frame && s->component && s->distance
             ? 0 : ENOMEM;
}
static double time_sum(uint32_t vertices, const double *time) {
  double sum = 0;
  for (uint32_t v = 0; v < vertices; v++) sum += time[v];
  return sum;
}
static int graph_check(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to) {
  for (uint32_t v = 0; time && v < vertices; v++) if (!(time[v] >= 0 && time[v] <= DBL_MAX)) return EINVAL;
  if (time && !(time_sum(vertices, time) <= DBL_MAX)) return EINVAL;
  for (size_t e = 0; e < edges; e++) if (from[e] >= vertices || to[e] >= vertices) return EINVAL;
  return 0;
}
/* R. E. Tarjan, SIAM J. Comput. 1(2), 1972 */
static uint32_t components_search(const struct graph_scratch *s, uint32_t vertices, const uint32_t *to,
                                  uint32_t *component) {
  uint32_t count = 0, next = 0, top = 0;
  for (uint32_t v = 0; v < vertices; v++) { s->index[v] = UINT32_MAX; component[v] = UINT32_MAX; }
  for (uint32_t root = 0; root < vertices; root++) {
    if (s->index[root] != UINT32_MAX) continue;
    uint32_t depth = 0;
    s->index[root] = s->low[root] = next++; s->stack[top++] = root;
    s->frame[depth] = root; s->at[depth++] = s->first[root];
    while (depth) {
      const uint32_t v = s->frame[depth - 1];
      if (s->at[depth - 1] < s->first[v + 1]) {
        const uint32_t w = to[s->order[s->at[depth - 1]++]];
        if (s->index[w] == UINT32_MAX) {
          s->index[w] = s->low[w] = next++; s->stack[top++] = w;
          s->frame[depth] = w; s->at[depth++] = s->first[w];
        } else if (component[w] == UINT32_MAX && s->index[w] < s->low[v]) s->low[v] = s->index[w];
        continue;
      }
      if (s->low[v] == s->index[v]) {
        uint32_t w;
        do { w = s->stack[--top]; component[w] = count; } while (w != v);
        count++;
      }
      if (--depth && s->low[v] < s->low[s->frame[depth - 1]]) s->low[s->frame[depth - 1]] = s->low[v];
    }
  }
  return count;
}
struct zero_rows { const uint32_t *from; const uint64_t *delay; uint32_t vertices; };
static uint32_t zero_row(const void *context, size_t e) {
  const struct zero_rows *z = context;
  return z->delay[e] ? z->vertices : z->from[e];
}
/* BSL Lemma 1 */
static int zero_delay_cycle(const struct graph_scratch *s, uint32_t vertices, size_t edges, const uint32_t *from,
                            const uint32_t *to, const uint64_t *delay) {
  const struct zero_rows rows = {from, delay, vertices};
  mesh_graph_rows(vertices + 1, edges, zero_row, &rows, s->first, s->order);
  components_search(s, vertices, to, s->component);
  for (size_t e = 0; e < edges; e++) if (!delay[e] && s->component[from[e]] == s->component[to[e]]) return 1;
  return 0;
}
static double cycle_tolerance(uint32_t vertices, const double *time) { return 1e-9 * time_sum(vertices, time); }
/* R. Bellman, Quart. Appl. Math. 16, 1958; L. R. Ford, RAND P-923, 1956 */
static int positive_cycle(double *distance, uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                          const uint32_t *to, const uint64_t *delay, double lambda, double tolerance) {
  for (uint32_t v = 0; v < vertices; v++) distance[v] = 0;
  for (uint32_t round = 0; round < vertices; round++) {
    int moved = 0;
    for (size_t e = 0; e < edges; e++) {
      if (delay[e] == UINT64_MAX) continue;
      const double through = distance[from[e]] + (time[from[e]] - lambda * (double)delay[e]);
      if (distance[to[e]] + tolerance < through) { distance[to[e]] = through; moved = 1; }
    }
    if (!moved) return 0;
  }
  return vertices > 0;
}
/* BSL Fig. 8 */
static int delays_pass(const struct graph_scratch *s, uint32_t vertices, const double *time, size_t edges,
                       const uint32_t *from, const uint32_t *to, const uint64_t *delay, double lambda, double tolerance) {
  return !zero_delay_cycle(s, vertices, edges, from, to, delay) &&
         !positive_cycle(s->distance, vertices, time, edges, from, to, delay, lambda, tolerance);
}
/* BSL §3 */
int mesh_graph_components(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *component, uint32_t *count) {
  int result = graph_check(vertices, NULL, edges, from, to);
  if (result) return result;
  struct graph_scratch s;
  result = scratch_alloc(&s, vertices, edges);
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, from, s.first, s.order);
    *count = components_search(&s, vertices, to, component);
  }
  scratch_free(&s);
  return result;
}
/* BSL §9, BellmanFord; E. L. Lawler, Combinatorial Optimization: Networks and Matroids, 1976, pp. 94-97 */
int mesh_graph_cycle_mean(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          const uint64_t *delay, double *mean) {
  int result = graph_check(vertices, time, edges, from, to);
  if (result) return result;
  struct graph_scratch s;
  result = scratch_alloc(&s, vertices, edges);
  if (!result) {
    const double tolerance = cycle_tolerance(vertices, time);
    double lo = 0, hi = time_sum(vertices, time), found = 0;
    if (zero_delay_cycle(&s, vertices, edges, from, to, delay)) found = INFINITY;
    else if (positive_cycle(s.distance, vertices, time, edges, from, to, delay, 0, tolerance)) {
      for (int step = 0; step < 200; step++) {
        const double mid = lo + (hi - lo) / 2;
        if (mid <= lo || mid >= hi) break;
        if (positive_cycle(s.distance, vertices, time, edges, from, to, delay, mid, tolerance)) lo = mid; else hi = mid;
      }
      found = hi + vertices * tolerance;
    }
    *mean = found;
  }
  scratch_free(&s);
  return result;
}
/* BSL §9, Fig. 8, MinDelay */
int mesh_graph_min_delay(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                         const uint64_t *delay, size_t e, double lambda, uint64_t bound, uint64_t *least) {
  int result = graph_check(vertices, time, edges, from, to);
  if (!result && (e >= edges || !(lambda >= 0 && lambda <= DBL_MAX))) result = EINVAL;
  if (result) return result;
  struct graph_scratch s;
  result = scratch_alloc(&s, vertices, edges);
  uint64_t *changed = result ? NULL : malloc((edges + 1) * sizeof *changed);
  if (!result && !changed) result = ENOMEM;
  if (!result) {
    const double tolerance = cycle_tolerance(vertices, time);
    for (size_t k = 0; k < edges; k++) changed[k] = delay[k];
    changed[e] = bound;
    if (!delays_pass(&s, vertices, time, edges, from, to, changed, lambda, tolerance)) result = ERANGE;
    uint64_t lo = 0, hi = bound;
    while (!result && lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      changed[e] = mid;
      if (delays_pass(&s, vertices, time, edges, from, to, changed, lambda, tolerance)) hi = mid; else lo = mid + 1;
    }
    if (!result) *least = hi;
  }
  scratch_free(&s); free(changed);
  return result;
}
/* BSL §4.2-4.3, (6): Bff of each feedforward edge of the two-processor IPC graph */
int mesh_graph_buffer_bound(uint32_t actors, const double *host, const double *work, size_t edges,
                            const struct mesh_graph_dependence *edge, uint64_t *bound) {
  if (!actors || actors >= UINT32_MAX / 2 || edges >= SIZE_MAX / sizeof(uint64_t) - 3 * (size_t)actors - 2) return EINVAL;
  for (uint32_t j = 0; j < actors; j++)
    if (!(host[j] >= 0 && host[j] <= DBL_MAX) || !(work[j] >= 0 && work[j] <= DBL_MAX)) return EINVAL;
  const uint32_t V = 2 * actors;
  for (size_t e = 0; e < edges; e++) if (edge[e].from >= V || edge[e].to >= V) return EINVAL;
  const size_t E = 3 * (size_t)actors + edges;
  double *time = malloc(((size_t)V + 1) * sizeof *time);
  uint32_t *from = malloc((E + 2) * sizeof *from), *to = malloc((E + 2) * sizeof *to);
  uint64_t *delay = malloc((E + 2) * sizeof *delay);
  int result = time && from && to && delay ? 0 : ENOMEM;
  for (uint32_t j = 0; !result && j < actors; j++) {
    const uint32_t next = j + 1 == actors ? 0 : j + 1;
    time[j] = host[j]; time[actors + j] = work[j];
    from[3 * j] = j; to[3 * j] = next; delay[3 * j] = next == 0;
    from[3 * j + 1] = actors + j; to[3 * j + 1] = actors + next; delay[3 * j + 1] = next == 0;
    from[3 * j + 2] = j; to[3 * j + 2] = actors + j; delay[3 * j + 2] = 0;
  }
  for (size_t e = 0; !result && e < edges; e++) {
    const size_t k = 3 * (size_t)actors + e;
    from[k] = edge[e].from; to[k] = edge[e].to; delay[k] = edge[e].delay;
  }
  double lambda = 0, sum = 0;
  if (!result) result = mesh_graph_cycle_mean(V, time, E, from, to, delay, &lambda);
  if (!result && lambda == INFINITY) result = EINVAL;
  for (uint32_t v = 0; !result && v < V; v++) sum += time[v];
  for (uint32_t j = 0; !result && j < actors; j++) {
    if (!(lambda > 0)) { bound[j] = 1; continue; }
    const double most = ceil(sum / lambda);
    from[E] = actors + j; to[E] = j; delay[E] = 0;
    uint64_t least = 0;
    result = mesh_graph_min_delay(V, time, E + 1, from, to, delay, E, lambda, most >= 1 ? (uint64_t)most : 1, &least);
    bound[j] = least ? least : 1;
  }
  free(time); free(from); free(to); free(delay);
  return result;
}
/* BSL §3, §9, Fig. 6 (Convert-to-SC-graph) and Theorem 4 */
int mesh_graph_convert_sc(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *added, uint32_t *added_from, uint32_t *added_to) {
  int result = graph_check(vertices, time, edges, from, to);
  if (result) return result;
  struct graph_scratch s;
  result = scratch_alloc(&s, vertices, edges);
  const size_t n = (size_t)vertices + 1;
  uint32_t *least = malloc(n * sizeof *least), *link = malloc(n * sizeof *link);
  uint32_t *sources = malloc(n * sizeof *sources), *sinks = malloc(n * sizeof *sinks);
  unsigned char *entered = malloc(n), *left = malloc(n);
  if (!result && (!least || !link || !sources || !sinks || !entered || !left)) result = ENOMEM;
  uint32_t count = 0, m = 0, k = 0;
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, from, s.first, s.order);
    count = components_search(&s, vertices, to, s.component);
    for (uint32_t c = 0; c < count; c++) { least[c] = UINT32_MAX; entered[c] = 0; left[c] = 0; }
    for (size_t e = 0; e < edges; e++)
      if (s.component[from[e]] != s.component[to[e]]) { entered[s.component[to[e]]] = 1; left[s.component[from[e]]] = 1; }
    for (uint32_t v = 0; v < vertices; v++) {
      const uint32_t c = s.component[v];
      if (least[c] == UINT32_MAX) { least[c] = v; link[c] = v; }
      else if (time[v] < time[link[c]]) link[c] = v;
    }
    for (uint32_t v = 0; count > 1 && v < vertices; v++) {
      const uint32_t c = s.component[v];
      if (least[c] != v) continue;
      if (!entered[c]) sources[m++] = link[c];
      if (!left[c]) sinks[k++] = link[c];
    }
  }
  if (!result) {
    const uint32_t made = count > 1 ? m + k - 1 : 0;
    if (made) { added_from[0] = sinks[k - 1]; added_to[0] = sources[0]; }
    for (uint32_t i = 1; i < m; i++) { added_from[i] = sources[i - 1]; added_to[i] = sources[i]; }
    for (uint32_t j = 0; j + 1 < k; j++) { added_from[m + j] = sinks[k - 2 - j]; added_to[m + j] = sinks[k - 1 - j]; }
    *added = made;
  }
  scratch_free(&s); free(least); free(link); free(sources); free(sinks); free(entered); free(left);
  return result;
}
/* BSL §4.3 (6) and §9, Fig. 8 (DetermineDelays) */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, double lambda, uint64_t *added_delay) {
  int result = graph_check(vertices, time, edges, from, to);
  if (!result) result = graph_check(vertices, NULL, added, added_from, added_to);
  if (!result && !(lambda >= 0 && lambda <= DBL_MAX)) result = EINVAL;
  if (result) return result;
  if (edges >= SIZE_MAX / sizeof(uint64_t) - added) return ENOMEM;
  const size_t total = edges + added;
  uint32_t *all_from = malloc((total + 1) * sizeof *all_from), *all_to = malloc((total + 1) * sizeof *all_to);
  uint64_t *all_delay = malloc((total + 1) * sizeof *all_delay);
  result = all_from && all_to && all_delay ? 0 : ENOMEM;
  if (!result) {
    for (size_t e = 0; e < edges; e++) { all_from[e] = from[e]; all_to[e] = to[e]; all_delay[e] = delay[e]; }
    for (uint32_t i = 0; i < added; i++) {
      all_from[edges + i] = added_from[i]; all_to[edges + i] = added_to[i]; all_delay[edges + i] = UINT64_MAX;
    }
  }
  if (!result) {
    const double quotient = lambda > 0 ? time_sum(vertices, time) / lambda : 0;
    uint64_t bound = quotient < 0x1p64 ? (uint64_t)quotient : UINT64_MAX;
    if (bound < UINT64_MAX && (double)bound < quotient) bound++;
    if (!bound) bound = 1;
    for (uint32_t i = 0; !result && i < added; i++) {
      uint64_t least = 0;
      result = mesh_graph_min_delay(vertices, time, total, all_from, all_to, all_delay, edges + i, lambda, bound, &least);
      if (!result) all_delay[edges + i] = least;
    }
  }
  for (uint32_t i = 0; !result && i < added; i++) added_delay[i] = all_delay[edges + i];
  free(all_from); free(all_to); free(all_delay);
  return result;
}

/* S. Verdoolaege, isl: An Integer Set Library for the Polyhedral Model, ICMS 2010, LNCS 6327 (isl 0.27, its exact
   integer arithmetic imath): a view's dimensions [0, dims) at `offset` */
struct view_part { uint64_t offset, element; const uint64_t *pairs; uint32_t dims; };
/* the hull [offset, *end) of a view's bytes, 0 where it exceeds 128-bit arithmetic (no hull, the exact test decides) */
static int view_hull_end(const struct view_part *v, __int128 *end) {
  __int128 w = v->element, step;
  for (uint32_t d = 0; d < v->dims; d++) {
    if (!v->pairs[2 * d]) continue;
    if (__builtin_mul_overflow((__int128)(v->pairs[2 * d] - 1), (__int128)v->pairs[2 * d + 1], &step) ||
        __builtin_add_overflow(w, step, &w)) return 0;
  }
  return !__builtin_add_overflow((__int128)v->offset, w, end);
}
static isl_val *view_value(isl_ctx *ctx, uint64_t x, int negative) {
  isl_val *v = isl_val_int_from_ui(ctx, (unsigned long)x);
  return negative ? isl_val_neg(v) : v;
}
/* whether some byte is in both views: the integer points (i_k < size_k of a's dimensions, j_l < size_l of b's, u <
   a's element, v < b's) of a.offset + sum_k i_k stride_k + u = b.offset + sum_l j_l stride_l + v, isl_basic_set_is_empty
   over them (exact, arbitrary precision); disjoint hulls first. 1 meets, 0 not, -errno */
static int view_meets(isl_ctx *ctx, const struct view_part *a, const struct view_part *b) {
  __int128 ea, eb;
  if (view_hull_end(a, &ea) && view_hull_end(b, &eb) && ((__int128)a->offset >= eb || (__int128)b->offset >= ea)) return 0;
  const unsigned n = a->dims + b->dims + 2;
  isl_space *space = isl_space_set_alloc(ctx, 0, n);
  isl_local_space *local = isl_local_space_from_space(isl_space_copy(space));
  isl_basic_set *set = isl_basic_set_universe(space);
  isl_constraint *c = isl_constraint_alloc_equality(isl_local_space_copy(local));
  c = isl_constraint_set_constant_val(c, isl_val_sub(view_value(ctx, a->offset, 0), view_value(ctx, b->offset, 0)));
  for (uint32_t k = 0; k < a->dims; k++)
    c = isl_constraint_set_coefficient_val(c, isl_dim_set, (int)k, view_value(ctx, a->pairs[2 * k + 1], 0));
  for (uint32_t l = 0; l < b->dims; l++)
    c = isl_constraint_set_coefficient_val(c, isl_dim_set, (int)(a->dims + l), view_value(ctx, b->pairs[2 * l + 1], 1));
  c = isl_constraint_set_coefficient_si(c, isl_dim_set, (int)(n - 2), 1);
  c = isl_constraint_set_coefficient_si(c, isl_dim_set, (int)(n - 1), -1);
  set = isl_basic_set_add_constraint(set, c);
  for (unsigned x = 0; x < n; x++) {
    const uint64_t size = x < a->dims ? a->pairs[2 * x] : x < a->dims + b->dims ? b->pairs[2 * (x - a->dims)] :
                          x == n - 2 ? a->element : b->element;
    c = isl_constraint_alloc_inequality(isl_local_space_copy(local));
    set = isl_basic_set_add_constraint(set, isl_constraint_set_coefficient_si(c, isl_dim_set, (int)x, 1));
    c = isl_constraint_alloc_inequality(isl_local_space_copy(local));
    c = isl_constraint_set_coefficient_si(c, isl_dim_set, (int)x, -1);
    set = isl_basic_set_add_constraint(set, isl_constraint_set_constant_val(c, isl_val_sub_ui(view_value(ctx, size, 0), 1)));
  }
  isl_local_space_free(local);
  const isl_bool empty = isl_basic_set_is_empty(set);
  isl_basic_set_free(set);
  return empty == isl_bool_error ? -ENOMEM : empty == isl_bool_false;
}
struct view_hull { uint64_t key; __int128 start, end; size_t index; };
static int view_hull_before(const void *a, const void *b) {
  const struct view_hull *x = a, *y = b;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  return (x->start > y->start) - (x->start < y->start);
}
static int view_edge_before(const void *a, const void *b) {
  const struct mesh_graph_conflict *x = a, *y = b;
  if (x->from != y->from) return x->from < y->from ? -1 : 1;
  if (x->to != y->to) return x->to < y->to ? -1 : 1;
  if (x->kind != y->kind) return x->kind < y->kind ? -1 : 1;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  if (x->given != y->given) return x->given < y->given ? -1 : 1;
  if (x->from_view != y->from_view) return x->from_view < y->from_view ? -1 : 1;
  return (x->to_view > y->to_view) - (x->to_view < y->to_view);
}
/* the hazard of access `from` before access `to` (mesh-graph.h mesh_graph_conflict `kind`) */
static uint32_t view_hazard(const struct mesh_graph_view *from, const struct mesh_graph_view *to) {
  return from->write && to->write ? MESH_GRAPH_WRITE_AFTER_WRITE :
         from->write ? MESH_GRAPH_READ_AFTER_WRITE : MESH_GRAPH_WRITE_AFTER_READ;
}
/* a pair the decider answered: its two views as the record orients them, and the deciding original */
struct view_decided { size_t from_view, to_view; struct mesh_graph_origin origin; };
static int view_decided_before(const void *a, const void *b) {
  const struct view_decided *x = a, *y = b;
  if (x->from_view != y->from_view) return x->from_view < y->from_view ? -1 : 1;
  return (x->to_view > y->to_view) - (x->to_view < y->to_view);
}
static int view_conflicts(size_t views, const struct mesh_graph_view *view, size_t supplied,
                          const uint32_t *supplied_from, const uint32_t *supplied_to, int self,
                          int (*decide)(void *context, size_t a, size_t b, struct mesh_graph_decided *out),
                          void *context, struct mesh_graph_conflict **conflict, struct mesh_graph_origin **origin,
                          struct mesh_graph_relation ***decider, const uint32_t ***decider_map, size_t *deciders,
                          size_t *edges) {
  *conflict = NULL; *edges = 0;
  if (origin) *origin = NULL;
  if (decider) { *decider = NULL; *decider_map = NULL; *deciders = 0; }
  struct mesh_graph_relation **answered = NULL;
  const uint32_t **answered_map = NULL;
  size_t answers = 0, answer_room = 0;
  for (size_t e = 0; e < supplied; e++) if (supplied_from[e] >= supplied_to[e]) return EINVAL;
  if (supplied >= SIZE_MAX / sizeof(struct mesh_graph_conflict) - 64 || views >= SIZE_MAX / sizeof(struct view_hull)) return ENOMEM;
  struct view_hull *hull = malloc((views + 1) * sizeof *hull);
  size_t made = supplied, held = supplied + 64, count = 0, decideds = 0, room = 0;
  struct view_decided *decided = NULL;
  struct mesh_graph_conflict *edge = malloc(held * sizeof *edge);
  int result = hull && edge ? 0 : ENOMEM;
  for (size_t e = 0; !result && e < supplied; e++)
    edge[e] = (struct mesh_graph_conflict){supplied_from[e], supplied_to[e], MESH_GRAPH_GIVEN, (uint32_t)e, 0, SIZE_MAX,
                                           SIZE_MAX};
  for (size_t v = 0; !result && v < views; v++) {
    const struct mesh_strided *s = &view[v].view;
    const struct view_part p = {s->offset, s->element, s->pairs, s->dims};
    __int128 end;
    if (!view_hull_end(&p, &end)) end = ((__int128)1 << 126);
    hull[v] = (struct view_hull){view[v].key, (__int128)s->offset, end, v};
  }
  isl_ctx *ctx = result ? NULL : isl_ctx_alloc();
  if (!result && !ctx) result = ENOMEM;
  if (!result && views) qsort(hull, views, sizeof *hull, view_hull_before);
  for (size_t i = 0; !result && i < views; i++)
    for (size_t j = i + !self; !result && j < views && hull[j].key == hull[i].key && hull[j].start < hull[i].end; j++) {
      const struct mesh_graph_view *a = view + hull[i].index, *b = view + hull[j].index;
      if ((!self && a->vertex == b->vertex) || !(a->write || b->write) || (a->write == 2 && b->write == 2)) continue;
      const struct view_part pa = {a->view.offset, a->view.element, a->view.pairs, a->view.dims},
                             pb = {b->view.offset, b->view.element, b->view.pairs, b->view.dims};
      int empty = !a->view.element || !b->view.element;
      for (uint32_t d = 0; d < a->view.dims; d++) empty |= !a->view.pairs[2 * d];
      for (uint32_t d = 0; d < b->view.dims; d++) empty |= !b->view.pairs[2 * d];
      if (empty) continue;
      struct mesh_graph_decided d = {NULL, NULL, {SIZE_MAX, 0, 0}};
      if (a != b && decide) {
        const int status = decide(context, hull[i].index, hull[j].index, &d);
        if (status) { result = status; break; }
        /* every answering (relation, map) input once, in the order first answered: one relation under two maps (a
           source appended under two bindings) is two inputs */
        size_t seen = answers;
        for (size_t q = answers; d.witness.known && q-- > 0;)
          if (answered[q] == d.relation && answered_map[q] == d.map) { seen = q; break; }
        if (d.witness.known && seen == answers) {
          if (answers == answer_room) {
            const size_t more = answer_room ? 2 * answer_room : 8;
            struct mesh_graph_relation **grown = realloc(answered, more * sizeof *grown);
            if (grown) answered = grown;
            const uint32_t **maps = grown ? realloc(answered_map, more * sizeof *maps) : NULL;
            if (maps) answered_map = maps;
            if (!grown || !maps) { result = ENOMEM; break; }
            answer_room = more;
          }
          answered[answers] = d.relation;
          answered_map[answers++] = d.map;
        }
      }
      const int meets = a == b ? 1 : d.witness.known ? d.witness.record != SIZE_MAX : view_meets(ctx, &pa, &pb);
      if (meets < 0) { result = -meets; break; }
      if (!meets) continue;
      const struct mesh_graph_view *lo = a->vertex < b->vertex ? a : b, *hi = a->vertex < b->vertex ? b : a;
      uint32_t kind = view_hazard(lo, hi);
      const unsigned directions = 1 + (lo->vertex == hi->vertex && kind != MESH_GRAPH_WRITE_AFTER_WRITE);
      size_t first = (size_t)(lo - view), second = (size_t)(hi - view);
      for (unsigned direction = 0; direction < directions; direction++) {
        if (made == held) {
          if (held > SIZE_MAX / 2 / sizeof *edge) { result = ENOMEM; break; }
          struct mesh_graph_conflict *more = realloc(edge, 2 * held * sizeof *more);
          if (!more) { result = ENOMEM; break; }
          edge = more; held *= 2;
        }
        edge[made++] = (struct mesh_graph_conflict){lo->vertex, hi->vertex, kind, 0, a->key, first, second};
        if (d.witness.known) {
          if (decideds == room) {
            if (room > SIZE_MAX / 4 / sizeof *decided) { result = ENOMEM; break; }
            struct view_decided *more = realloc(decided, (room ? 2 * room : 64) * sizeof *more);
            if (!more) { result = ENOMEM; break; }
            decided = more; room = room ? 2 * room : 64;
          }
          struct mesh_graph_witness w = d.witness;
          w.reverse ^= first != hull[i].index;
          decided[decideds++] = (struct view_decided){first, second, {d.relation, w}};
        }
        kind = kind == MESH_GRAPH_READ_AFTER_WRITE ? MESH_GRAPH_WRITE_AFTER_READ : MESH_GRAPH_READ_AFTER_WRITE;
        const size_t swapped = first;
        first = second;
        second = swapped;
      }
    }
  if (!result && made) qsort(edge, made, sizeof *edge, view_edge_before);
  for (size_t k = 0; !result && k < made; k++) {
    if (k && !view_edge_before(edge + k, edge + k - 1)) continue;
    edge[count++] = edge[k];
  }
  if (!result && origin) {
    if (decideds) qsort(decided, decideds, sizeof *decided, view_decided_before);
    struct mesh_graph_origin *of = malloc((count + 1) * sizeof *of);
    if (!of) result = ENOMEM;
    for (size_t k = 0; !result && k < count; k++) {
      of[k] = (struct mesh_graph_origin){NULL, {SIZE_MAX, 0, 0}};
      if (edge[k].kind == MESH_GRAPH_GIVEN) continue;
      size_t lo = 0, hi = decideds;
      while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (decided[mid].from_view < edge[k].from_view ||
            (decided[mid].from_view == edge[k].from_view && decided[mid].to_view < edge[k].to_view)) lo = mid + 1;
        else hi = mid;
      }
      if (lo < decideds && decided[lo].from_view == edge[k].from_view && decided[lo].to_view == edge[k].to_view)
        of[k] = decided[lo].origin;
    }
    if (!result) *origin = of;
    else free(of);
  }
  if (!result) { *edges = count; *conflict = edge; edge = NULL; }
  if (!result && decider) {
    *decider = answered; *decider_map = answered_map; *deciders = answers;
    answered = NULL; answered_map = NULL;
  }
  if (ctx) isl_ctx_free(ctx);
  free(hull); free(edge); free(decided); free(answered); free(answered_map);
  return result;
}

int mesh_graph_conflicts_views(size_t views, const struct mesh_graph_view *view, size_t supplied,
                               const uint32_t *supplied_from, const uint32_t *supplied_to,
                               struct mesh_graph_conflict **conflict, size_t *count) {
  return view_conflicts(views, view, supplied, supplied_from, supplied_to, 1, NULL, NULL, conflict, NULL, NULL, NULL,
                        NULL, count);
}

int mesh_graph_dependences_views(size_t views, const struct mesh_graph_view *view, size_t supplied,
                                 const uint32_t *supplied_from, const uint32_t *supplied_to,
                                 struct mesh_graph_conflict **conflict, size_t *count) {
  return view_conflicts(views, view, supplied, supplied_from, supplied_to, 0, NULL, NULL, conflict, NULL, NULL, NULL,
                        NULL, count);
}

/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §5.2, Definition 3, §7,
   Theorem 3 and Fig. 4; E. W. Dijkstra, Numerische Mathematik 1, 1959 */
struct redundant_entry { uint64_t at; size_t first; uint32_t vertex; };
struct redundant_search { uint64_t *best; size_t *first; struct redundant_entry *heap; size_t count; };
static int redundant_before(const struct redundant_entry *x, const struct redundant_entry *y) {
  return x->at < y->at || (x->at == y->at && x->vertex < y->vertex);
}
static void redundant_push(struct redundant_search *s, uint64_t at, uint32_t vertex, size_t first) {
  const struct redundant_entry e = {at, first, vertex};
  size_t k = s->count++;
  while (k && redundant_before(&e, s->heap + (k - 1) / 2)) { s->heap[k] = s->heap[(k - 1) / 2]; k = (k - 1) / 2; }
  s->heap[k] = e;
}
static struct redundant_entry redundant_pop(struct redundant_search *s) {
  const struct redundant_entry top = s->heap[0], last = s->heap[--s->count];
  size_t k = 0;
  for (size_t c = 1; c < s->count; c = 2 * k + 1) {
    if (c + 1 < s->count && redundant_before(s->heap + c + 1, s->heap + c)) c++;
    if (!redundant_before(s->heap + c, &last)) break;
    s->heap[k] = s->heap[c]; k = c;
  }
  if (s->count) s->heap[k] = last;
  return top;
}
static void redundant_offer(struct redundant_search *s, uint32_t y, uint64_t at, size_t first) {
  uint64_t *best = s->best + 2 * (size_t)y;
  size_t *by = s->first + 2 * (size_t)y;
  if (by[0] == first) {
    if (at < best[0]) { best[0] = at; redundant_push(s, at, y, first); }
  } else if (at < best[0]) {
    best[1] = best[0]; by[1] = by[0]; best[0] = at; by[0] = first;
    redundant_push(s, at, y, first);
  } else if (at < best[1]) {
    best[1] = at; by[1] = first;
    redundant_push(s, at, y, first);
  }
}
/* mesh_graph_redundant's one solve over edges read through tail, head and delay (synchronization NULL: every edge a
   candidate), shared by its flat-array callers and the retained relation */
static int redundant_solve(uint32_t vertices, size_t edges, uint32_t (*tail)(const void *context, size_t e),
                           uint32_t (*head)(const void *context, size_t e), uint64_t (*delay)(const void *context, size_t e),
                           const void *context, const unsigned char *synchronization, unsigned char *redundant) {
  for (size_t e = 0; e < edges; e++) {
    const uint32_t a = tail(context, e), b = head(context, e);
    if (a >= vertices || b >= vertices || a == b) return EINVAL;
  }
  if (vertices == UINT32_MAX || edges >= SIZE_MAX / (4 * sizeof(struct redundant_entry))) return ENOMEM;
  const size_t n = (size_t)vertices + 1;
  size_t *first = malloc(n * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  uint64_t *pair = malloc((edges + 1) * sizeof *pair);
  struct redundant_search s = {malloc(2 * n * sizeof *s.best), malloc(2 * n * sizeof *s.first),
                               malloc((3 * edges + 1) * sizeof *s.heap), 0};
  unsigned char *done = malloc(2 * n);
  int result = first && order && pair && s.best && s.first && s.heap && done ? 0 : ENOMEM;
  size_t synchronizations = 0;
  for (size_t e = 0; !result && e < edges; e++)
    if (!synchronization || synchronization[e]) pair[synchronizations++] = (uint64_t)tail(context, e) << 32 | head(context, e);
  if (!result && synchronizations) qsort(pair, synchronizations, sizeof *pair, completion_edge_before);
  for (size_t k = 1; !result && k < synchronizations; k++) if (pair[k] == pair[k - 1]) result = EINVAL;
  if (!result) {
    mesh_graph_rows(vertices, edges, tail, context, first, order);
    for (size_t e = 0; e < edges; e++) redundant[e] = 0;
  }
  for (uint32_t u = 0; !result && u < vertices; u++) {
    int any = 0;
    for (size_t k = first[u]; k < first[u + 1]; k++) any |= !synchronization || synchronization[order[k]] != 0;
    if (!any) continue;
    for (size_t v = 0; v < 2 * (size_t)vertices; v++) { s.best[v] = UINT64_MAX; s.first[v] = SIZE_MAX; done[v] = 0; }
    s.count = 0;
    for (size_t k = first[u]; k < first[u + 1]; k++)
      redundant_offer(&s, head(context, order[k]), delay(context, order[k]), order[k]);
    while (s.count) {
      const struct redundant_entry x = redundant_pop(&s);
      const size_t at = 2 * (size_t)x.vertex;
      const int label = s.best[at] == x.at && s.first[at] == x.first && !done[at] ? 0
                      : s.best[at + 1] == x.at && s.first[at + 1] == x.first && !done[at + 1] ? 1 : -1;
      if (label < 0) continue;
      done[at + (size_t)label] = 1;
      for (size_t k = first[x.vertex]; k < first[x.vertex + 1]; k++) {
        const uint64_t lag = delay(context, order[k]);
        if (lag <= UINT64_MAX - x.at) redundant_offer(&s, head(context, order[k]), x.at + lag, x.first);
      }
    }
    for (size_t k = first[u]; k < first[u + 1]; k++) {
      const size_t e = order[k], at = 2 * (size_t)head(context, e);
      if (!synchronization || synchronization[e])
        redundant[e] = (s.first[at] == e ? s.best[at + 1] : s.best[at]) <= delay(context, e);
    }
  }
  free(first); free(order); free(pair); free(s.best); free(s.first); free(s.heap); free(done);
  return result;
}
struct redundant_arrays { const uint32_t *from, *to; const uint64_t *delay; };
static uint32_t redundant_tail(const void *context, size_t e) { return ((const struct redundant_arrays *)context)->from[e]; }
static uint32_t redundant_head(const void *context, size_t e) { return ((const struct redundant_arrays *)context)->to[e]; }
static uint64_t redundant_delay(const void *context, size_t e) {
  const struct redundant_arrays *x = context;
  return x->delay ? x->delay[e] : 0;
}
int mesh_graph_redundant(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, const uint64_t *delay,
                         const unsigned char *synchronization, unsigned char *redundant) {
  const struct redundant_arrays arrays = {from, to, delay};
  return redundant_solve(vertices, edges, redundant_tail, redundant_head, redundant_delay, &arrays, synchronization,
                         redundant);
}

/* design/torch-source-lowering-change-request.md item 3; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans.
   Signal Processing 45(6), 1997, §7, Fig. 4: every edge a candidate, at zero delays */
static uint64_t synchronization_delay(const void *context, size_t e) { (void)context; (void)e; return 0; }
static int synchronization_solve(uint32_t vertices, size_t edges, uint32_t (*tail)(const void *context, size_t e),
                                 uint32_t (*head)(const void *context, size_t e), const void *context,
                                 unsigned char *redundant, size_t *kept) {
  const int result = redundant_solve(vertices, edges, tail, head, synchronization_delay, context, NULL, redundant);
  size_t made = 0;
  for (size_t e = 0; !result && e < edges; e++) made += !redundant[e];
  if (!result) *kept = made;
  return result;
}
int mesh_graph_synchronization(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                               unsigned char *redundant, size_t *kept) {
  const struct redundant_arrays arrays = {from, to, NULL};
  return synchronization_solve(vertices, edges, redundant_tail, redundant_head, &arrays, redundant, kept);
}

/* design/torch-source-lowering-change-request.md item 3; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2 */
int mesh_graph_levels_view(const struct mesh_graph_rows_view *rows, uint32_t *level) {
  for (uint32_t v = 0; v < rows->vertices; v++) level[v] = 0;
  for (uint32_t v = 0; v < rows->vertices; v++)
    for (size_t k = 0, n = rows->degree(rows->context, v); k < n; k++) {
      const uint32_t head = rows->head(rows->context, v, k);
      if (head == UINT32_MAX) continue;
      if (head >= rows->vertices) return EINVAL;
      if (head <= v) return EDEADLK;
      if (level[v] + 1 > level[head]) level[head] = level[v] + 1;
    }
  return 0;
}

int mesh_graph_levels(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, uint32_t *level) {
  for (size_t e = 0; e < edges; e++) if (from[e] >= vertices || to[e] >= vertices) return EINVAL;
  if (vertices == UINT32_MAX || edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  size_t *first = malloc(((size_t)vertices + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  int result = first && order ? 0 : ENOMEM;
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, from, first, order);
    const struct flat_rows flat = {first, order, to};
    const struct mesh_graph_rows_view rows = {vertices, &flat, flat_degree, flat_head};
    result = mesh_graph_levels_view(&rows, level);
  }
  free(first); free(order);
  return result;
}

int mesh_graph_schedule(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, uint32_t *level,
                        uint32_t *place) {
  int result = mesh_graph_levels(vertices, edges, from, to, level);
  double *priority = malloc(((size_t)vertices + 1) * sizeof *priority);
  if (!priority) result = ENOMEM;
  for (uint32_t v = 0; !result && v < vertices; v++) priority[v] = -(double)level[v];
  if (!result) result = mesh_graph_heap_order(vertices, priority, edges, from, to, place);
  free(priority);
  return result;
}

/* design/synchronization-signature-change-request.md items 1, 4 and 6; S. S. Bhattacharyya, S. Sriram and E. A. Lee,
   IEEE Trans. Signal Processing 45(6), 1997, §4.1, §5.1 and §7, Fig. 4; D. P. Reed and R. K. Kanodia, CACM 22(2),
   1979; MTLEvent.h waitUntilSignaledValue:timeoutMS:; MTL4CommandQueue.h waitForEvent:value: */
static int signature_signal(uint32_t p) {
  return p == MESH_GRAPH_QUEUE_SIGNAL || p == MESH_GRAPH_FENCE_UPDATE || p == MESH_GRAPH_BRIDGE_START ||
         p == MESH_GRAPH_BRIDGE_RAISE || p == MESH_GRAPH_INITIAL || p == MESH_GRAPH_HOST_PUBLISH ||
         p == MESH_GRAPH_HOST_SIGNAL;
}
/* design/synchronization-signature-change-request.md item 6: a static count in the compiled kernel, `value` of them,
   UINT64_MAX where the kernel's source is not known, `object` the commands so counted */
static int signature_static(uint32_t p) {
  return p == MESH_GRAPH_THREADGROUP_BARRIER || p == MESH_GRAPH_SIMDGROUP_BARRIER || p == MESH_GRAPH_ATOMIC ||
         p == MESH_GRAPH_KERNEL_FENCE;
}
/* MESH_GRAPH_BRIDGE_READ: the bridge's SEND hop's read of a release's supplied word (rdma/mesh-collective.c
   plan_supplied), the word a publication raises: a wait of the publications' class */
static int signature_wait(uint32_t p) {
  return p == MESH_GRAPH_QUEUE_WAIT || p == MESH_GRAPH_FENCE_WAIT || p == MESH_GRAPH_HOST_WAIT || p == MESH_GRAPH_HOST_NOTIFY ||
         p == MESH_GRAPH_BRIDGE_READ;
}
static uint32_t signature_class(uint32_t p) {
  return p == MESH_GRAPH_FENCE_UPDATE || p == MESH_GRAPH_FENCE_WAIT ? 1 : p == MESH_GRAPH_INITIAL ? 2 :
         p == MESH_GRAPH_HOST_PUBLISH || p == MESH_GRAPH_BRIDGE_READ ? 3 : 0;
}
static size_t graph_lower(const void *base, size_t count, size_t size, const void *probe,
                          int (*before)(const void *, const void *)) {
  size_t low = 0, high = count;
  while (low < high) {
    const size_t mid = low + (high - low) / 2;
    if (before((const char *)base + mid * size, probe) < 0) low = mid + 1; else high = mid;
  }
  return low;
}
struct signature_key { uint32_t class, object; uint64_t value; size_t index; };
static int signature_key_before(const void *a, const void *b) {
  const struct signature_key *x = a, *y = b;
  if (x->class != y->class) return x->class < y->class ? -1 : 1;
  if (x->object != y->object) return x->object < y->object ? -1 : 1;
  if (x->value != y->value) return x->value < y->value ? -1 : 1;
  return (x->index > y->index) - (x->index < y->index);
}
static int signature_dependence_before(const void *a, const void *b) {
  const struct mesh_graph_dependence *x = a, *y = b;
  if (x->from != y->from) return x->from < y->from ? -1 : 1;
  if (x->to != y->to) return x->to < y->to ? -1 : 1;
  return (x->delay > y->delay) - (x->delay < y->delay);
}
struct signature_task { uint32_t processor, place, task; };
static int signature_task_before(const void *a, const void *b) {
  const struct signature_task *x = a, *y = b;
  if (x->processor != y->processor) return x->processor < y->processor ? -1 : 1;
  if (x->place != y->place) return x->place < y->place ? -1 : 1;
  return (x->task > y->task) - (x->task < y->task);
}
struct signature_graph { uint32_t vertices; size_t edges, held, skip; uint32_t *from, *to; uint64_t *delay; size_t *first, *order; };
static int signature_edge(struct signature_graph *g, uint32_t a, uint32_t b, uint64_t d) {
  if (g->edges == g->held) {
    const size_t grown = g->held ? 2 * g->held : 64;
    uint32_t *f = realloc(g->from, grown * sizeof *f);
    if (f) g->from = f;
    uint32_t *t = f ? realloc(g->to, grown * sizeof *t) : NULL;
    if (t) g->to = t;
    uint64_t *y = t ? realloc(g->delay, grown * sizeof *y) : NULL;
    if (!y) return ENOMEM;
    g->delay = y; g->held = grown;
  }
  g->from[g->edges] = a; g->to[g->edges] = b; g->delay[g->edges++] = d;
  return 0;
}
static uint32_t signature_tail(const void *context, size_t k) { return ((const struct signature_graph *)context)->from[k]; }
static uint32_t signature_head(const void *context, size_t k) {
  const struct signature_graph *g = context;
  return g->to[g->order[k]];
}
static uint64_t signature_lag(const void *context, size_t k) {
  const struct signature_graph *g = context;
  const size_t e = g->order[k];
  return e == g->skip ? UINT64_MAX : g->delay[e];
}
static uint32_t signature_processor(const void *context, size_t k) {
  return ((const struct mesh_graph_operation *)context)[k].processor;
}
struct signature_image { uint32_t from, to; uint64_t delay; size_t edge; };
static int signature_image_before(const void *a, const void *b) {
  const struct signature_image *x = a, *y = b;
  if (x->from != y->from) return x->from < y->from ? -1 : 1;
  if (x->to != y->to) return x->to < y->to ? -1 : 1;
  if (x->delay != y->delay) return x->delay < y->delay ? -1 : 1;
  return (x->edge > y->edge) - (x->edge < y->edge);
}
static int signature_enforces(const struct mesh_graph_operation *operation, const struct mesh_graph_synchronized *e) {
  return e->signal != MESH_GRAPH_NONE && operation[e->signal].vertex != MESH_GRAPH_NONE &&
         operation[e->wait].vertex != MESH_GRAPH_NONE && operation[e->signal].execution <= operation[e->wait].execution;
}

static void realized_free(struct mesh_graph_realized *);
/* mesh-graph.h struct mesh_graph_archived: a restored archive's bytes [bytes, bytes + length) and their adopted owner,
   reference counted as the scopes are (its signature's reference, and its restored scopes' while they are held) */
struct mesh_graph_archived { size_t references; void (*release)(void *); void *owner; const char *bytes; size_t length; };
static void archived_free(struct mesh_graph_archived *x) {
  if (__atomic_fetch_sub(&x->references, 1, __ATOMIC_ACQ_REL) != 1) return;
  if (x->release) x->release(x->owner);
  free(x);
}
static void archived_released(void *x) { archived_free(x); }
/* whether `p` is one of the signature's views of its archive (mesh_graph_archive_read: every view, an empty one at the
   archive's start, lies within its bytes), never written or freed */
static int signature_viewed(const struct mesh_graph_signature *s, const void *p) {
  if (!s->archived || !p) return 0;
  const uintptr_t at = (uintptr_t)p, base = (uintptr_t)s->archived->bytes;
  return at >= base && at - base < s->archived->length;
}
static void signature_release(const struct mesh_graph_signature *s, void *p) { if (!signature_viewed(s, p)) free(p); }
/* the signature's own copy of its pairs where they are its archive's view, before any of them is changed */
static int signature_edges_own(struct mesh_graph_signature *s) {
  if (!signature_viewed(s, s->edge)) return 0;
  struct mesh_graph_synchronized *e = malloc((s->synchronized + 1) * sizeof *e);
  if (!e) return ENOMEM;
  if (s->synchronized) memcpy(e, s->edge, s->synchronized * sizeof *e);
  s->edge = e;
  return 0;
}
void mesh_graph_signature_free(struct mesh_graph_signature *s) {
  void *const array[] = {s->processor, s->task, s->operation, s->dependence, s->edge, s->position, s->count, s->least,
                         s->previous, s->waited, s->first, s->order, s->unknown, s->alias, s->event, s->span,
                         s->repeated, s->proof, s->answer, s->projected};
  for (size_t k = 0; k < sizeof array / sizeof *array; k++) signature_release(s, array[k]);
  if(s->realized)realized_free(s->realized);
  if(s->scopes)mesh_graph_scopes_free(s->scopes);
  if(s->archived)archived_free(s->archived);
  memset(s, 0, sizeof *s);
}

static int signature_digest(struct mesh_graph_signature *s) {
  size_t made = 0;
  int result = mesh_graph_matrix_market(s, NULL, 0, &made);
  if (result != ERANGE) return result ? result : EINVAL;
  char *text = malloc(made);
  if (!text) return ENOMEM;
  result = mesh_graph_matrix_market(s, text, made, &made);
  if (!result) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    for (size_t at = 0; at + 1 < made;) {
      const size_t step = made - 1 - at < (size_t)1 << 30 ? made - 1 - at : (size_t)1 << 30;
      CC_SHA256_Update(&context, text + at, (CC_LONG)step);
      at += step;
    }
    CC_SHA256_Final(s->digest, &context);
  }
  free(text);
  return result;
}
int mesh_graph_signature_program(struct mesh_graph_signature *s,size_t events,const struct mesh_graph_event *event){
  struct mesh_graph_event *e=malloc((events+1)*sizeof *e);
  if(!e)return ENOMEM;
  if(events)memcpy(e,event,events*sizeof *e);
  signature_release(s,s->event);
  s->events=events;s->event=e;
  return signature_digest(s);
}
int mesh_graph_signature_spans(struct mesh_graph_signature *s,size_t spans,const struct mesh_graph_span *span){
  if(spans && !span)return EINVAL;
  for(size_t k=0;k<spans;k++)if(span[k].count>UINT64_MAX-span[k].first || span[k].begin>span[k].end ||
    span[k].end>s->operations || (k && (span[k].first!=span[k-1].first+span[k-1].count ||
    span[k].begin!=span[k-1].end)))return EINVAL;
  struct mesh_graph_span *p=malloc((spans+1)*sizeof *p);
  if(!p)return ENOMEM;
  if(spans)memcpy(p,span,spans*sizeof *p);
  signature_release(s,s->span);s->span=p;s->spans=spans;
  return signature_digest(s);
}
/* design/synchronization-signature-change-request.md items 1, 4 and 6: completion scopes (mesh-graph.h struct
   mesh_graph_scopes) over borrowed row descriptors, and a signature's realized occurrences, sites and published values
   (struct mesh_graph_realized), each reference counted as relations are, each releasing its adopted owner with its
   last reference; `own` the holder's own arrays (a restored or composed scopes' descriptors, a composed holder's
   occurrences and published values), `part` the composed operands' holders it retains (each original owner kept);
   a scopes' bound native sources `source` (mesh_graph_scopes_bind) kept by `bound`, a holder of the same kind (a
   composed scopes' own projection of its operands' pointers, retaining their exact holders) */
struct mesh_graph_scopes { size_t references, rows; const struct mesh_graph_scopes_data *row; void (*release)(void *);
                           void *owner; void *own; struct mesh_graph_scopes *part[2]; const void *const *source;
                           struct mesh_graph_realized *bound; };
struct mesh_graph_realized { size_t references; void (*release)(void *); void *owner; void *own;
                             struct mesh_graph_realized *part[2]; };
static int scopes_row_valid(const struct mesh_graph_scopes_data *d) {
  if ((d->completions && (!d->completion || !d->release_first || d->release_first[0])) || (d->writers && !d->writer) ||
      (d->completions && d->release_first[d->completions] != d->releases) || (d->releases && !d->release)) return 0;
  for (size_t c = 0; c < d->completions; c++) {
    const struct mesh_graph_completion *x = d->completion + c;
    if (x->writer > d->writers || x->writers > d->writers - x->writer || d->release_first[c + 1] < d->release_first[c])
      return 0;
    uint64_t total = 0;
    for (uint32_t w = 0; w < x->writers; w++) {
      if (d->writer[x->writer + w].count > UINT64_MAX - total) return 0;
      total += d->writer[x->writer + w].count;
    }
    if (total != x->total) return 0;
  }
  return 1;
}
int mesh_graph_scopes_create(size_t rows, const struct mesh_graph_scopes_data *row, void (*release_owner)(void *),
                             void *owner, struct mesh_graph_scopes **out) {
  *out = NULL;
  if (rows && !row) return EINVAL;
  for (size_t r = 0; r < rows; r++) if (!scopes_row_valid(row + r)) return EINVAL;
  struct mesh_graph_scopes *made = malloc(sizeof *made);
  if (!made) return ENOMEM;
  *made = (struct mesh_graph_scopes){.references = 1, .rows = rows, .row = row, .release = release_owner, .owner = owner};
  *out = made;
  return 0;
}
void mesh_graph_scopes_retain(struct mesh_graph_scopes *x) { __atomic_fetch_add(&x->references, 1, __ATOMIC_RELAXED); }
void mesh_graph_scopes_free(struct mesh_graph_scopes *x) {
  if (__atomic_fetch_sub(&x->references, 1, __ATOMIC_ACQ_REL) != 1) return;
  if (x->release) x->release(x->owner);
  free(x->own);
  for (int k = 0; k < 2; k++) if (x->part[k]) mesh_graph_scopes_free(x->part[k]);
  if (x->bound) realized_free(x->bound);
  free(x);
}
const struct mesh_graph_scopes_data *mesh_graph_scopes_rows(const struct mesh_graph_scopes *x, size_t *rows) {
  *rows = x->rows;
  return x->row;
}
int mesh_graph_scopes_bind(struct mesh_graph_scopes *x, size_t rows, const void *const *source,
                           void (*release_owner)(void *), void *owner) {
  if (rows != x->rows || (rows && !source)) return EINVAL;
  struct mesh_graph_realized *bound = malloc(sizeof *bound);
  if (!bound) return ENOMEM;
  *bound = (struct mesh_graph_realized){.references = 1, .release = release_owner, .owner = owner};
  struct mesh_graph_realized *prior = x->bound;
  x->source = source; x->bound = bound;
  /* the earlier binding's owner released; a composed scopes that projected it retains its holder */
  if (prior) realized_free(prior);
  return 0;
}
const void *const *mesh_graph_scopes_sources(const struct mesh_graph_scopes *x) { return x->source; }
static void realized_free(struct mesh_graph_realized *x) {
  if (__atomic_fetch_sub(&x->references, 1, __ATOMIC_ACQ_REL) != 1) return;
  if (x->release) x->release(x->owner);
  free(x->own);
  for (int k = 0; k < 2; k++) if (x->part[k]) realized_free(x->part[k]);
  free(x);
}
static void realized_retain(struct mesh_graph_realized *x) { __atomic_fetch_add(&x->references, 1, __ATOMIC_RELAXED); }
static void scopes_released(void *x) { mesh_graph_scopes_free(x); }
static void signature_states(struct mesh_graph_signature *s) {
  memset(s->states, 0, sizeof s->states);
  for (size_t k = 0; k < s->synchronized; k++) s->states[s->edge[k].state]++;
}
/* a publication event's identity: its row's source node, its release's owner call and representative transfer, and its
   published value */
struct scope_key { uint32_t node, owner, transfer, event; uint64_t value; };
static int scope_key_before(const void *a, const void *b) {
  const struct scope_key *x = a, *y = b;
  if (x->node != y->node) return x->node < y->node ? -1 : 1;
  if (x->owner != y->owner) return x->owner < y->owner ? -1 : 1;
  if (x->transfer != y->transfer) return x->transfer < y->transfer ? -1 : 1;
  return (x->value > y->value) - (x->value < y->value);
}
/* a signature's attachment: its scopes and realized holder (one reference each its own), their occurrences, sites and
   published values, its answered reads and its projected sites (its own, NULL for none) */
struct signature_attached { struct mesh_graph_scopes *scopes; struct mesh_graph_realized *realized;
                            size_t occurrences, sites, publications, answers;
                            const struct mesh_graph_occurrence *occurrence; const struct mesh_graph_site *site;
                            const uint64_t *published; struct mesh_graph_answered *answer;
                            struct mesh_graph_site *projected; };
static struct signature_attached signature_attachment(const struct mesh_graph_signature *s) {
  return (struct signature_attached){s->scopes, s->realized, s->occurrences, s->sites, s->publications, s->answers,
                                     s->occurrence, s->site, s->published, s->answer, s->projected};
}
/* attachment x set, nothing released: every pair MESH_GRAPH_SCOPED by an earlier attachment UNMATCHED again (SCOPED is
   only ever made from UNMATCHED, here), then each of x's answered reads' pairs MESH_GRAPH_SCOPED where no operation
   answered it, the states counted; no edge, least delay or task changes */
static void signature_attach_set(struct mesh_graph_signature *s, const struct signature_attached *x) {
  s->scopes = x->scopes; s->realized = x->realized; s->occurrences = x->occurrences; s->sites = x->sites;
  s->publications = x->publications; s->answers = x->answers; s->occurrence = x->occurrence; s->site = x->site;
  s->published = x->published; s->answer = x->answer; s->projected = x->projected;
  for (size_t k = 0; k < s->synchronized; k++)
    if (s->edge[k].state == MESH_GRAPH_SCOPED) s->edge[k].state = MESH_GRAPH_UNMATCHED;
  for (size_t k = 0; k < s->answers; k++) {
    const size_t e = s->waited[s->answer[k].operation];
    if (e != SIZE_MAX && s->edge[e].state == MESH_GRAPH_UNMATCHED) s->edge[e].state = MESH_GRAPH_SCOPED;
  }
  signature_states(s);
}
/* x's answered reads over the signature's operations: every MESH_GRAPH_BRIDGE_READ answered by the publication event
   of its (node: its processor's index, owner, representative transfer, value), an event of a row without source none
   (no node is assumed for it); none without scopes. The signature is not changed */
static int signature_answered(const struct mesh_graph_signature *s, struct signature_attached *x) {
  x->answer = NULL; x->answers = 0;
  if (!x->scopes) return 0;
  size_t keys = 0;
  for (size_t k = 0; k < x->occurrences; k++) {
    const struct mesh_graph_occurrence *y = x->occurrence + k;
    const struct mesh_graph_scopes_data *d = x->scopes->row + y->row;
    const size_t n = d->source ? d->release_first[y->completion + 1] - d->release_first[y->completion] : 0;
    if (n > SIZE_MAX / sizeof(struct scope_key) - 1 - keys) return ENOMEM;
    keys += n;
  }
  struct scope_key *key = malloc((keys + 1) * sizeof *key);
  struct mesh_graph_answered *answer = malloc((s->operations + 1) * sizeof *answer);
  if (!key || !answer) { free(key); free(answer); return ENOMEM; }
  size_t answers = 0;
  keys = 0;
  for (size_t k = 0; k < x->occurrences; k++) {
    const struct mesh_graph_occurrence *y = x->occurrence + k;
    const struct mesh_graph_scopes_data *d = x->scopes->row + y->row;
    const size_t f = d->release_first[y->completion], n = d->source ? d->release_first[y->completion + 1] - f : 0;
    for (size_t r = 0; r < n; r++)
      key[keys++] = (struct scope_key){d->source->node, d->release[f + r].owner, d->release[f + r].transfer,
                                       y->event + (uint32_t)r, x->published[y->event + r]};
  }
  qsort(key, keys, sizeof *key, scope_key_before);
  for (size_t o = 0; o < s->operations; o++) {
    const struct mesh_graph_operation *w = s->operation + o;
    if (w->primitive != MESH_GRAPH_BRIDGE_READ || w->vertex == MESH_GRAPH_NONE || w->object < MESH_GRAPH_PLAN_SUPPLIED)
      continue;
    const struct scope_key probe = {s->processor[w->processor].index, s->task[w->vertex].label,
                                    w->object - MESH_GRAPH_PLAN_SUPPLIED, 0, w->value};
    const size_t at = graph_lower(key, keys, sizeof *key, &probe, scope_key_before);
    if (at == keys || scope_key_before(key + at, &probe)) continue;
    answer[answers++] = (struct mesh_graph_answered){key[at].event, (uint32_t)o};
  }
  free(key);
  x->answer = answer; x->answers = answers;
  return 0;
}
/* a restored or composed signature's answered reads from the attachment it holds, its pairs' states from them */
static int signature_answers(struct mesh_graph_signature *s) {
  struct signature_attached x = signature_attachment(s);
  const int result = signature_answered(s, &x);
  if (!result) { signature_release(s, s->answer); signature_attach_set(s, &x); }
  return result;
}
/* attachment x, its scopes and holder one reference each the caller holds for it and its projected sites the caller's,
   made the signature's as one transaction: its answered reads, states and digest from x, then the earlier attachment
   released (its references, answers and projected sites). On failure the signature is as it was (its attachment,
   states and digest), x the caller's still, nothing released */
static int signature_attach(struct mesh_graph_signature *s, struct signature_attached x) {
  int result = signature_answered(s, &x);
  if (result) return result;
  /* an archive's pairs are its view: the states change in the signature's own copy */
  struct mesh_graph_synchronized *const viewed = signature_viewed(s, s->edge) ? s->edge : NULL;
  if ((result = signature_edges_own(s))) { free(x.answer); return result; }
  const struct signature_attached prior = signature_attachment(s);
  signature_attach_set(s, &x);
  result = signature_digest(s);
  if (result) {
    signature_attach_set(s, &prior); free(x.answer);
    if (viewed) { free(s->edge); s->edge = viewed; }
    return result;
  }
  signature_release(s, prior.answer); signature_release(s, prior.projected);
  if (prior.realized) realized_free(prior.realized);
  if (prior.scopes) mesh_graph_scopes_free(prior.scopes);
  return 0;
}
/* the domains of a signature's realized arrays against its scopes and tasks */
static int signature_realized_valid(const struct mesh_graph_signature *s, const struct mesh_graph_scopes *scopes,
                                    size_t occurrences, const struct mesh_graph_occurrence *occurrence, size_t sites,
                                    const struct mesh_graph_site *site, size_t publications) {
  for (size_t k = 0; k < occurrences; k++) {
    const struct mesh_graph_occurrence *x = occurrence + k;
    if (x->row >= scopes->rows || x->completion >= scopes->row[x->row].completions) return 0;
    const struct mesh_graph_scopes_data *d = scopes->row + x->row;
    const struct mesh_graph_completion *c = d->completion + x->completion;
    const size_t releases = d->release_first[x->completion + 1] - d->release_first[x->completion];
    if (x->site > sites || c->writers > sites - x->site || x->event > publications ||
        releases > publications - x->event) return 0;
  }
  for (size_t k = 0; k < sites; k++)
    if (site[k].first >= s->tasks || (site[k].later != MESH_GRAPH_NONE && site[k].later >= s->tasks)) return 0;
  return 1;
}
int mesh_graph_signature_scopes(struct mesh_graph_signature *s, struct mesh_graph_scopes *scopes, size_t occurrences,
                                const struct mesh_graph_occurrence *occurrence, size_t sites,
                                const struct mesh_graph_site *site, size_t publications, const uint64_t *published,
                                void (*release_owner)(void *), void *owner) {
  if (!scopes || (occurrences && !occurrence) || (sites && !site) || (publications && !published) ||
      !signature_realized_valid(s, scopes, occurrences, occurrence, sites, site, publications)) return EINVAL;
  struct mesh_graph_realized *realized = malloc(sizeof *realized);
  if (!realized) return ENOMEM;
  *realized = (struct mesh_graph_realized){.references = 1, .release = release_owner, .owner = owner};
  mesh_graph_scopes_retain(scopes);
  const int result = signature_attach(s, (struct signature_attached){
    .scopes = scopes, .realized = realized, .occurrences = occurrences, .sites = sites, .publications = publications,
    .occurrence = occurrence, .site = site, .published = published});
  /* on failure the caller keeps `owner` and its arrays: the holder freed unreleased, its scopes reference dropped */
  if (result) { free(realized); mesh_graph_scopes_free(scopes); }
  return result;
}
/* everything a signature holds beyond its processors, tasks, dependences, operations, pairs, least delays and aliases,
   from those: each operation's place in its processor's order, the pair it waits in, the counts, states and digest */
static int signature_derive(struct mesh_graph_signature *s) {
  const size_t cells = (size_t)s->processors * MESH_GRAPH_PRIMITIVES + 1, n = s->operations;
  s->position = malloc((n + 1) * sizeof *s->position);
  s->previous = malloc((n + 1) * sizeof *s->previous);
  s->waited = malloc((n + 1) * sizeof *s->waited);
  s->first = malloc(((size_t)s->processors + 2) * sizeof *s->first);
  s->order = malloc((n + 1) * sizeof *s->order);
  s->count = calloc(cells, sizeof *s->count);
  s->unknown = calloc(cells, sizeof *s->unknown);
  unsigned char *used = calloc(n + 1, 1);
  int result = s->position && s->previous && s->waited && s->first && s->order && s->count && s->unknown && used ? 0 : ENOMEM;
  if (!result) {
    mesh_graph_rows(s->processors, n, signature_processor, s->operation, s->first, s->order);
    for (uint32_t p = 0; p < s->processors; p++)
      for (size_t k = s->first[p]; k < s->first[p + 1]; k++) {
        const size_t o = s->order[k];
        s->position[o] = k - s->first[p];
        s->previous[o] = k > s->first[p] ? s->order[k - 1] : SIZE_MAX;
      }
    for (size_t o = 0; o < n; o++) {
      const struct mesh_graph_operation *x = s->operation + o;
      const size_t cell = (size_t)x->processor * MESH_GRAPH_PRIMITIVES + x->primitive;
      s->waited[o] = SIZE_MAX;
      if (!signature_static(x->primitive)) s->count[cell]++;
      else if (x->value == UINT64_MAX) s->unknown[cell] += x->object;
      else s->count[cell] += x->value;
    }
    memset(s->states, 0, sizeof s->states);
    s->redundant = s->unwaited = s->uncovered = 0;
    for (size_t k = 0; k < s->synchronized; k++) {
      const struct mesh_graph_synchronized *e = s->edge + k;
      s->waited[e->wait] = k;
      if (e->signal != MESH_GRAPH_NONE) used[e->signal] = 1;
      s->states[e->state]++;
      s->redundant += e->redundant;
    }
    for (size_t o = 0; o < n; o++)
      s->unwaited += signature_signal(s->operation[o].primitive) && signature_class(s->operation[o].primitive) != 2 && !used[o];
    for (size_t k = 0; k < s->dependences; k++) s->uncovered += s->least[k] > s->dependence[k].delay;
    result = signature_digest(s);
  }
  free(used);
  return result;
}

int mesh_graph_signature(uint32_t processors, const struct mesh_graph_processor *processor, uint32_t tasks,
                         const struct mesh_graph_task *task, size_t dependences,
                         const struct mesh_graph_dependence *dependence, size_t operations,
                         const struct mesh_graph_operation *operation, struct mesh_graph_signature *s) {
  memset(s, 0, sizeof *s);
  for (uint32_t p = 0; p < processors; p++) if (processor[p].kind >= MESH_GRAPH_KINDS) return EINVAL;
  for (uint32_t t = 0; t < tasks; t++) if (task[t].processor >= processors) return EINVAL;
  for (size_t k = 0; k < dependences; k++) if (dependence[k].from >= tasks || dependence[k].to >= tasks) return EINVAL;
  for (size_t o = 0; o < operations; o++)
    if (operation[o].processor >= processors || operation[o].primitive >= MESH_GRAPH_PRIMITIVES ||
        (operation[o].vertex != MESH_GRAPH_NONE && operation[o].vertex >= tasks)) return EINVAL;
  if (tasks == UINT32_MAX || operations >= SIZE_MAX / (4 * sizeof(struct signature_key)) ||
      dependences >= SIZE_MAX / (4 * sizeof *dependence) || (size_t)processors >= SIZE_MAX / (MESH_GRAPH_PRIMITIVES * 8))
    return ENOMEM;
  s->processors = processors; s->tasks = tasks; s->operations = operations; s->dependences = dependences;
  s->processor = malloc(((size_t)processors + 1) * sizeof *s->processor);
  s->task = malloc(((size_t)tasks + 1) * sizeof *s->task);
  s->operation = malloc((operations + 1) * sizeof *s->operation);
  s->dependence = malloc((dependences + 1) * sizeof *s->dependence);
  s->edge = malloc((operations + 1) * sizeof *s->edge);
  s->least = malloc((dependences + 1) * sizeof *s->least);
  s->alias = malloc((operations + 1) * sizeof *s->alias);
  struct signature_key *key = malloc((operations + 1) * sizeof *key);
  struct signature_task *row = malloc(((size_t)tasks + 1) * sizeof *row);
  struct signature_image *image = malloc((operations + 1) * sizeof *image);
  size_t *slot = malloc((operations + 1) * sizeof *slot), *pair = NULL;
  uint32_t *sinks = malloc((dependences + 1) * sizeof *sinks);
  uint64_t *lags = malloc((dependences + 1) * sizeof *lags);
  unsigned char *barred = NULL, *flag = NULL, *redundant = NULL;
  struct signature_graph g = {tasks, 0, 0, SIZE_MAX, NULL, NULL, NULL, malloc(((size_t)tasks + 2) * sizeof(size_t)), NULL};
  uint32_t *rf = NULL, *rt = NULL;
  uint64_t *rd = NULL;
  int result = s->processor && s->task && s->operation && s->dependence && s->edge && s->least && s->alias && key &&
               row && image && slot && sinks && lags && g.first ? 0 : ENOMEM;
  if (!result) {
    memcpy(s->processor, processor, (size_t)processors * sizeof *processor);
    memcpy(s->task, task, (size_t)tasks * sizeof *task);
    memcpy(s->operation, operation, operations * sizeof *operation);
    memcpy(s->dependence, dependence, dependences * sizeof *dependence);
    qsort(s->dependence, dependences, sizeof *s->dependence, signature_dependence_before);
  }
  size_t signals = 0;
  for (size_t o = 0; !result && o < operations; o++) {
    s->alias[o] = MESH_GRAPH_NONE;
    if (signature_signal(operation[o].primitive))
      key[signals++] = (struct signature_key){signature_class(operation[o].primitive), operation[o].object, operation[o].value, o};
  }
  if (!result) qsort(key, signals, sizeof *key, signature_key_before);
  for (size_t o = 0; !result && o < operations; o++) {
    const struct mesh_graph_operation *w = operation + o;
    if (!signature_wait(w->primitive)) continue;
    const uint32_t class = signature_class(w->primitive);
    struct mesh_graph_synchronized *e = s->edge + s->synchronized++;
    *e = (struct mesh_graph_synchronized){MESH_GRAPH_NONE, (uint32_t)o, MESH_GRAPH_UNMATCHED, 0, MESH_GRAPH_NONE,
                                          MESH_GRAPH_NONE, 0};
    if (!class) {
      const struct signature_key initial = {2, w->object, 0, 0};
      const size_t i = graph_lower(key, signals, sizeof *key, &initial, signature_key_before);
      if (i < signals && key[i].class == 2 && key[i].object == w->object && key[i].value >= w->value) {
        e->state = MESH_GRAPH_SATISFIED; e->signal = (uint32_t)key[i].index;
        continue;
      }
    }
    const struct signature_key probe = {class, w->object, w->value, 0};
    const size_t low = graph_lower(key, signals, sizeof *key, &probe, signature_key_before);
    size_t chosen = SIZE_MAX;
    for (size_t k = low; k < signals && key[k].class == class && key[k].object == w->object &&
                         key[k].value == key[low].value && (class || key[k].value >= w->value) &&
                         (!class || key[k].value == w->value); k++)
      if (key[k].index < o || chosen == SIZE_MAX) chosen = k;
    if (chosen == SIZE_MAX) continue;
    const struct mesh_graph_operation *x = operation + key[chosen].index;
    e->signal = (uint32_t)key[chosen].index;
    e->state = MESH_GRAPH_FALSE;
    if (!signature_enforces(operation, e)) continue;
    e->delay = w->execution - x->execution;
    const struct mesh_graph_dependence want = {x->vertex, w->vertex, e->delay};
    const size_t at = graph_lower(s->dependence, dependences, sizeof *s->dependence, &want, signature_dependence_before);
    if (at < dependences && s->dependence[at].from == want.from && s->dependence[at].to == want.to) {
      e->dependence = (uint32_t)at;
      e->state = s->dependence[at].delay == want.delay ? MESH_GRAPH_IMAGE : MESH_GRAPH_STRONGER;
    }
  }
  for (uint32_t t = 0; !result && t < tasks; t++) row[t] = (struct signature_task){task[t].processor, task[t].place, t};
  if (!result) qsort(row, tasks, sizeof *row, signature_task_before);
  for (uint32_t a = 0, b; !result && a < tasks; a = b) {
    for (b = a; b < tasks && row[b].processor == row[a].processor; b++) {}
    if (processor[row[a].processor].kind == MESH_GRAPH_HOST && b - a > 1)
      for (uint32_t k = a; !result && k < b; k++) result = signature_edge(&g, row[k].task, row[k + 1 < b ? k + 1 : a].task, k + 1 == b);
  }
  if (!result && !(barred = calloc((size_t)tasks + 1, 1))) result = ENOMEM;
  for (size_t o = 0; !result && o < operations; o++) {
    const uint32_t b = operation[o].vertex, primitive = operation[o].primitive, mark = primitive == MESH_GRAPH_QUEUE_BARRIER ? 1 : 2;
    if ((primitive != MESH_GRAPH_QUEUE_BARRIER && primitive != MESH_GRAPH_PRODUCER_BARRIER) || b == MESH_GRAPH_NONE ||
        barred[b] & mark) continue;
    barred[b] |= mark;
    for (uint32_t a = 0; !result && a < tasks; a++) {
      if (task[a].processor != task[b].processor) continue;
      if (mark == 1) result = signature_edge(&g, a, b, task[a].place >= task[b].place);
      else if (task[a].place <= task[b].place)
        for (uint32_t c = 0; !result && c < tasks; c++)
          if (task[c].processor == task[b].processor && c != a) result = signature_edge(&g, a, c, task[c].place <= task[b].place);
    }
  }
  size_t images = 0;
  for (size_t k = 0; !result && k < s->synchronized; k++) {
    const struct mesh_graph_synchronized *e = s->edge + k;
    slot[k] = SIZE_MAX;
    if (e->state == MESH_GRAPH_SATISFIED || !signature_enforces(operation, e)) continue;
    slot[k] = g.edges;
    image[images++] = (struct signature_image){operation[e->signal].vertex, operation[e->wait].vertex, e->delay, g.edges};
    result = signature_edge(&g, operation[e->signal].vertex, operation[e->wait].vertex, e->delay);
  }
  if (!result) {
    g.order = malloc((g.edges + 1) * sizeof *g.order);
    pair = malloc((g.edges + 1) * sizeof *pair);
    rf = malloc((g.edges + 1) * sizeof *rf); rt = malloc((g.edges + 1) * sizeof *rt); rd = malloc((g.edges + 1) * sizeof *rd);
    flag = malloc(g.edges + 1); redundant = calloc(g.edges + 1, 1);
    if (!g.order || !pair || !rf || !rt || !rd || !flag || !redundant) result = ENOMEM;
  }
  if (!result) mesh_graph_rows(tasks, g.edges, signature_tail, &g, g.first, g.order);
  size_t kept = 0, flagged = 0;
  for (size_t e = 0; !result && e < g.edges; e++) {
    pair[e] = SIZE_MAX;
    if (g.from[e] == g.to[e]) continue;
    rf[kept] = g.from[e]; rt[kept] = g.to[e]; rd[kept] = g.delay[e]; flag[kept] = 0;
    pair[e] = kept++;
  }
  if (!result) qsort(image, images, sizeof *image, signature_image_before);
  for (size_t k = 0; !result && k < images; k++)
    if (image[k].from != image[k].to && (!k || image[k - 1].from != image[k].from || image[k - 1].to != image[k].to)) {
      flag[pair[image[k].edge]] = 1; flagged++;
    }
  if (!result && flagged) result = mesh_graph_redundant(tasks, kept, rf, rt, rd, flag, redundant);
  for (size_t k = 0; !result && k < s->synchronized; k++) {
    struct mesh_graph_synchronized *x = s->edge + k;
    const size_t e = slot[k];
    if (e == SIZE_MAX) continue;
    if (g.from[e] == g.to[e]) {
      uint64_t least = UINT64_MAX;
      g.skip = e;
      result = mesh_graph_lags(tasks, g.first, signature_head, signature_lag, &g, 1, g.from + e, 1, g.to + e, &least);
      g.skip = SIZE_MAX;
      x->redundant = least <= g.delay[e];
    } else x->redundant = !flag[pair[e]] || redundant[pair[e]];
  }
  for (size_t k = 0, j; !result && k < dependences; k = j) {
    for (j = k; j < dependences && s->dependence[j].from == s->dependence[k].from; j++) sinks[j - k] = s->dependence[j].to;
    result = mesh_graph_lags(tasks, g.first, signature_head, signature_lag, &g, 1, &s->dependence[k].from,
                             (uint32_t)(j - k), sinks, lags);
    for (size_t q = k; !result && q < j; q++) s->least[q] = lags[q - k];
  }
  if (!result) result = signature_derive(s);
  free(key); free(row); free(image); free(slot); free(pair); free(sinks); free(lags); free(barred);
  free(g.from); free(g.to); free(g.delay); free(g.first); free(g.order); free(rf); free(rt); free(rd);
  free(flag); free(redundant);
  if (result) mesh_graph_signature_free(s);
  return result;
}

/* design/synchronization-signature-change-request.md item 1; mesh-graph.h struct mesh_graph_repeated: the one
   expansion of a producer's repeated rows over executions [0, executions) */
static int repeated_operations(size_t repeats, const struct mesh_graph_repeated *r, uint64_t executions,
                               struct mesh_graph_operation **made, size_t *count) {
  size_t once = 0;
  for (size_t k = 0; k < repeats; k++) once += r[k].primitive == MESH_GRAPH_INITIAL;
  const size_t each = repeats - once;
  if (each && executions > (SIZE_MAX / sizeof **made - 1 - once) / each) return ENOMEM;
  const size_t n = once + (size_t)executions * each;
  struct mesh_graph_operation *o = malloc((n + 1) * sizeof *o);
  if (!o) return ENOMEM;
  size_t at = 0;
  for (size_t k = 0; k < repeats; k++)
    if (r[k].primitive == MESH_GRAPH_INITIAL)
      o[at++] = (struct mesh_graph_operation){r[k].processor, r[k].primitive, r[k].vertex, r[k].object, 0, 0};
  for (uint64_t K = 0; K < executions; K++)
    for (size_t k = 0; k < repeats; k++) {
      const struct mesh_graph_repeated *x = r + k;
      if (x->primitive == MESH_GRAPH_INITIAL) continue;
      struct mesh_graph_operation *y = o + at++;
      *y = (struct mesh_graph_operation){x->processor, x->primitive, x->vertex, x->object, K, 0};
      if (x->primitive == MESH_GRAPH_HOST_WAIT) {
        const __int128 need = (__int128)K * x->every + (__int128)x->ordinal - (__int128)x->lookback * x->every;
        y->value = need > 0 ? (uint64_t)need : 0;
      } else if (x->primitive == MESH_GRAPH_BRIDGE_START || x->primitive == MESH_GRAPH_BRIDGE_READ) {
        y->value = K * x->every + x->ordinal;
      } else if (x->primitive == MESH_GRAPH_BRIDGE_RAISE) {
        const uint64_t submitted = K * x->every + x->ordinal - 1;
        const int ended = submitted >= x->lookback;
        y->vertex = ended ? x->vertex : x->startup;
        y->execution = ended ? (submitted - x->lookback) / x->every : K;
        y->value = submitted + 1;
      }
    }
  *made = o; *count = n;
  return 0;
}
int mesh_graph_signature_repeated(uint32_t processors, const struct mesh_graph_processor *processor, uint32_t tasks,
                                  const struct mesh_graph_task *task, size_t dependences,
                                  const struct mesh_graph_dependence *dependence, size_t repeats,
                                  const struct mesh_graph_repeated *repeated, uint64_t executions,
                                  struct mesh_graph_signature *s) {
  memset(s, 0, sizeof *s);
  struct mesh_graph_operation *operation = NULL;
  size_t operations = 0;
  int result = repeated_operations(repeats, repeated, executions, &operation, &operations);
  if (!result) result = mesh_graph_signature(processors, processor, tasks, task, dependences, dependence, operations,
                                             operation, s);
  free(operation);
  if (result) return result;
  s->repeated = malloc((repeats + 1) * sizeof *s->repeated);
  if (!s->repeated) { mesh_graph_signature_free(s); return ENOMEM; }
  if (repeats) memcpy(s->repeated, repeated, repeats * sizeof *repeated);
  s->repeats = repeats;
  return 0;
}
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §9; mesh-graph.h struct
   mesh_graph_proof: a false pair takes the first kept proof of its task edge at its delay */
static int proof_before(const void *a, const void *b) {
  const struct mesh_graph_proof *x = a, *y = b;
  if (x->from != y->from) return x->from < y->from ? -1 : 1;
  if (x->to != y->to) return x->to < y->to ? -1 : 1;
  return (x->witness > y->witness) - (x->witness < y->witness);
}
static int signature_proved(struct mesh_graph_signature *s) {
  for (size_t k = 0; k < s->synchronized; k++) {
    struct mesh_graph_synchronized *e = s->edge + k;
    if (e->state != MESH_GRAPH_FALSE || !signature_enforces(s->operation, e)) continue;
    const struct mesh_graph_proof probe = {s->operation[e->signal].vertex, s->operation[e->wait].vertex, 0, 0, 0};
    for (size_t low = graph_lower(s->proof, s->proofs, sizeof *s->proof, &probe, proof_before);
         low < s->proofs && s->proof[low].from == probe.from && s->proof[low].to == probe.to; low++) {
      if (s->proof[low].total_delay != e->delay) continue;
      e->witness = s->proof[low].witness;
      if (s->proof[low].proven) e->state = MESH_GRAPH_TRANSFORMED;
      break;
    }
  }
  memset(s->states, 0, sizeof s->states);
  for (size_t k = 0; k < s->synchronized; k++) s->states[s->edge[k].state]++;
  return signature_digest(s);
}
int mesh_graph_signature_span(const struct mesh_graph_signature *kept, uint64_t executions,
                              struct mesh_graph_signature *s) {
  memset(s, 0, sizeof *s);
  if (kept->operations && !kept->repeats) return ENODATA;
  int result = mesh_graph_signature_repeated(kept->processors, kept->processor, kept->tasks, kept->task,
                                             kept->dependences, kept->dependence, kept->repeats, kept->repeated,
                                             executions, s);
  if (result) return result;
  s->proof = malloc((kept->proofs + 1) * sizeof *s->proof);
  if (!s->proof) result = ENOMEM;
  if (!result && kept->proofs) memcpy(s->proof, kept->proof, kept->proofs * sizeof *s->proof);
  if (!result) { s->proofs = kept->proofs; result = signature_proved(s); }
  if (result) mesh_graph_signature_free(s);
  return result;
}

/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §3, §9, Theorem 4, Figs. 6
   and 8; design/aot-compilation-change-request.md, appendix item 2, BS-4 */
static int resync_connected(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, size_t skip,
                            uint32_t *f, uint32_t *t, uint32_t *component, uint32_t *count) {
  size_t m = 0;
  for (size_t e = 0; e < edges; e++) if (e != skip) { f[m] = from[e]; t[m++] = to[e]; }
  return mesh_graph_components(vertices, m, f, t, component, count);
}
int mesh_graph_resynchronized(struct mesh_graph_signature *s, const double *time, size_t edges, const uint32_t *from,
                              const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                              const uint32_t *added_to, const uint64_t *added_delay, double lambda,
                              const uint64_t *offset) {
  const uint32_t T = s->tasks;
  if (!T) return 0;
  int result = graph_check(T, time, edges, from, to);
  if (!result) result = graph_check(T, NULL, added, added_from, added_to);
  if (!result && !(lambda >= 0 && lambda <= DBL_MAX)) result = EINVAL;
  if (result) return result;
  const size_t total = edges + added;
  uint32_t *all_from = malloc((total + 1) * sizeof *all_from), *all_to = malloc((total + 1) * sizeof *all_to);
  uint32_t *f = malloc((total + 1) * sizeof *f), *t = malloc((total + 1) * sizeof *t);
  uint32_t *component = malloc(((size_t)T + 1) * sizeof *component), *alone = malloc(((size_t)T + 1) * sizeof *alone);
  uint64_t *all_delay = malloc((total + 1) * sizeof *all_delay);
  struct mesh_graph_proof *proof = malloc(((size_t)added + 1) * sizeof *proof);
  unsigned char *ok = malloc((size_t)added + 1);
  struct graph_scratch g;
  result = scratch_alloc(&g, T, total);
  if (!result && !(all_from && all_to && f && t && component && alone && all_delay && proof && ok)) result = ENOMEM;
  uint32_t count = 0;
  for (size_t e = 0; !result && e < edges; e++) { all_from[e] = from[e]; all_to[e] = to[e]; all_delay[e] = delay[e]; }
  for (uint32_t i = 0; !result && i < added; i++) {
    all_from[edges + i] = added_from[i]; all_to[edges + i] = added_to[i]; all_delay[edges + i] = UINT64_MAX;
  }
  if (!result) result = mesh_graph_components(T, edges, from, to, alone, &count);
  for (uint32_t i = 0; !result && i < added; i++) ok[i] = alone[added_from[i]] != alone[added_to[i]];
  if (!result) result = resync_connected(T, total, all_from, all_to, SIZE_MAX, f, t, component, &count);
  const int strong = !result && count == 1;
  for (uint32_t i = 0; !result && i < added; i++) {
    ok[i] &= strong;
    if (ok[i] && !(result = resync_connected(T, total, all_from, all_to, edges + i, f, t, component, &count)))
      ok[i] &= count > 1;
  }
  const double tolerance = cycle_tolerance(T, time);
  for (uint32_t i = 0; !result && i < added; i++) {
    all_delay[edges + i] = added_delay[i];
    ok[i] &= delays_pass(&g, T, time, total, all_from, all_to, all_delay, lambda, tolerance);
    if (ok[i] && added_delay[i]) {
      all_delay[edges + i] = added_delay[i] - 1;
      ok[i] &= !delays_pass(&g, T, time, total, all_from, all_to, all_delay, lambda, tolerance);
      all_delay[edges + i] = added_delay[i];
    }
  }
  size_t proofs = 0;
  for (uint32_t i = 0; !result && i < added; i++) {
    uint64_t total_delay;
    if (__builtin_add_overflow(added_delay[i], offset[added_from[i]], &total_delay)) continue;
    proof[proofs++] = (struct mesh_graph_proof){added_from[i], added_to[i], total_delay, i, ok[i]};
  }
  /* an archive's pairs are its view: signature_proved changes the signature's own copy */
  if (!result) result = signature_edges_own(s);
  if (!result) {
    qsort(proof, proofs, sizeof *proof, proof_before);
    signature_release(s, s->proof);
    s->proof = proof; s->proofs = proofs; proof = NULL;
    result = signature_proved(s);
  }
  scratch_free(&g);
  free(all_from); free(all_to); free(f); free(t); free(component); free(alone); free(all_delay); free(proof); free(ok);
  return result;
}

/* design/synchronization-signature-change-request.md item 1 */
static size_t record_align(size_t at) { return (at + 7) / 8 * 8; }
struct record_part { void *data; size_t size; };
static size_t record_parts(const struct mesh_graph_signature *s, struct record_part part[7]) {
  part[0] = (struct record_part){s->processor, (size_t)s->processors * sizeof *s->processor};
  part[1] = (struct record_part){s->task, (size_t)s->tasks * sizeof *s->task};
  part[2] = (struct record_part){s->dependence, s->dependences * sizeof *s->dependence};
  part[3] = (struct record_part){s->operation, s->operations * sizeof *s->operation};
  part[4] = (struct record_part){s->edge, s->synchronized * sizeof *s->edge};
  part[5] = (struct record_part){s->least, s->dependences * sizeof *s->least};
  part[6] = (struct record_part){s->alias, s->operations * sizeof *s->alias};
  size_t at = sizeof(struct mesh_graph_record);
  for (int k = 0; k < 7; k++) at = record_align(at + part[k].size);
  return at;
}
/* the record's program section: its events, spans, repeated rows and proofs; a scoped signature's (GRAPH_PROGRAM_SCOPED)
   then its rows' counts (struct scope_row_record, `sourced` 1 where the row has a source), each row's completion[],
   writer[], release_first[completions + 1], release[] and its struct mesh_graph_scope_source where sourced (the
   reference, never a pointer), its occurrence[], site[] and published[], every array 8-byte sized, so each starts
   8-aligned where the record is */
struct program_record { uint64_t magic, events, spans, repeats, proofs, rows, occurrences, sites, publications; };
struct scope_row_record { uint64_t completions, writers, releases, sourced; };
#define GRAPH_PROGRAM_RECORD UINT64_C(0x414d41524748534d)
#define GRAPH_PROGRAM_REPEATED UINT64_C(0x5045525048534d47)
#define GRAPH_PROGRAM_SCOPED UINT64_C(0x324f435348534d47)
static size_t program_head(const struct mesh_graph_signature *s){
  return s->scopes?sizeof(struct program_record):s->repeats||s->proofs?offsetof(struct program_record,rows):
         offsetof(struct program_record,repeats);
}
static size_t scope_row_bytes(const struct mesh_graph_scopes_data *d){
  return d->completions*sizeof *d->completion+d->writers*sizeof *d->writer+(d->completions+1)*sizeof *d->release_first+
         d->releases*sizeof *d->release+(d->source?sizeof *d->source:0);
}
static size_t program_scoped(const struct mesh_graph_signature *s){
  if(!s->scopes)return 0;
  size_t bytes=s->scopes->rows*sizeof(struct scope_row_record)+s->occurrences*sizeof *s->occurrence+
               s->sites*sizeof *s->site+s->publications*sizeof *s->published;
  for(size_t r=0;r<s->scopes->rows;r++)bytes+=scope_row_bytes(s->scopes->row+r);
  return bytes;
}
static size_t program_parts(const struct mesh_graph_signature *s,struct record_part part[4]){
  part[0]=(struct record_part){s->event,s->events*sizeof *s->event};
  part[1]=(struct record_part){s->span,s->spans*sizeof *s->span};
  part[2]=(struct record_part){s->repeated,s->repeats*sizeof *s->repeated};
  part[3]=(struct record_part){s->proof,s->proofs*sizeof *s->proof};
  size_t bytes=program_scoped(s);for(int k=0;k<4;k++)bytes+=part[k].size;
  return bytes || s->scopes?program_head(s)+bytes:0;
}
static size_t program_put(char *at,const void *data,size_t size){ if(size)memcpy(at,data,size); return size; }
static void program_scoped_write(const struct mesh_graph_signature *s,char *at){
  if(!s->scopes)return;
  static const size_t none=0;
  for(size_t r=0;r<s->scopes->rows;r++){
    const struct mesh_graph_scopes_data *d=s->scopes->row+r;
    const struct scope_row_record row={d->completions,d->writers,d->releases,d->source!=NULL};
    at+=program_put(at,&row,sizeof row);
  }
  for(size_t r=0;r<s->scopes->rows;r++){
    const struct mesh_graph_scopes_data *d=s->scopes->row+r;
    at+=program_put(at,d->completion,d->completions*sizeof *d->completion);
    at+=program_put(at,d->writer,d->writers*sizeof *d->writer);
    at+=program_put(at,d->completions?d->release_first:&none,(d->completions+1)*sizeof *d->release_first);
    at+=program_put(at,d->release,d->releases*sizeof *d->release);
    if(d->source)at+=program_put(at,d->source,sizeof *d->source);
  }
  at+=program_put(at,s->occurrence,s->occurrences*sizeof *s->occurrence);
  at+=program_put(at,s->site,s->sites*sizeof *s->site);
  program_put(at,s->published,s->publications*sizeof *s->published);
}
/* a scoped record's section read as views of `at` (room bytes): its rows' descriptors (the restored scopes' own), its
   scopes adopting nothing yet (the caller adopts the record's owner on success), its realized arrays held through them */
static int program_scoped_read(struct mesh_graph_signature *s,const struct program_record *body,const char *at,
                               size_t room){
  const uint64_t rows=body->rows;
  if(rows>room/sizeof(struct scope_row_record))return EINVAL;
  struct mesh_graph_scopes_data *row=malloc(((size_t)rows+1)*sizeof *row);
  if(!row)return ENOMEM;
  const struct scope_row_record *count=(const void *)at;
  size_t used=(size_t)rows*sizeof *count;
  int result=0;
  for(size_t r=0;!result && r<rows;r++){
    const struct scope_row_record x=count[r];
    if(x.completions>room/24 || x.writers>room/24 || x.releases>room/8){result=EINVAL;break;}
    struct mesh_graph_scopes_data *d=row+r;
    *d=(struct mesh_graph_scopes_data){.completions=(size_t)x.completions,.writers=(size_t)x.writers,
                                       .releases=(size_t)x.releases};
    const size_t bytes=scope_row_bytes(d)+(x.sourced?sizeof *d->source:0);
    if(bytes>room-used){result=EINVAL;break;}
    const char *y=at+used;
    d->completion=(const void *)y;y+=d->completions*sizeof *d->completion;
    d->writer=(const void *)y;y+=d->writers*sizeof *d->writer;
    d->release_first=(const void *)y;y+=(d->completions+1)*sizeof *d->release_first;
    d->release=(const void *)y;y+=d->releases*sizeof *d->release;
    d->source=x.sourced?(const void *)y:NULL;
    used+=bytes;
  }
  if(!result && (body->occurrences>room/sizeof *s->occurrence || body->sites>room/sizeof *s->site ||
                 body->publications>room/sizeof *s->published ||
                 body->occurrences*sizeof *s->occurrence+body->sites*sizeof *s->site+
                 body->publications*sizeof *s->published!=room-used))result=EINVAL;
  struct mesh_graph_scopes *scopes=NULL;
  if(!result)result=mesh_graph_scopes_create((size_t)rows,row,NULL,NULL,&scopes);
  if(result){free(row);return result;}
  scopes->own=row;
  /* an archive's scopes keep its bytes, of which they are views, while any signature holds them */
  if(s->archived){
    __atomic_fetch_add(&s->archived->references,1,__ATOMIC_RELAXED);
    scopes->release=archived_released;scopes->owner=s->archived;
  }
  const struct mesh_graph_occurrence *occurrence=(const void *)(at+used);
  const struct mesh_graph_site *site=(const void *)((const char *)occurrence+body->occurrences*sizeof *occurrence);
  const uint64_t *published=(const void *)((const char *)site+body->sites*sizeof *site);
  struct mesh_graph_realized *realized=malloc(sizeof *realized);
  if(!realized || !signature_realized_valid(s,scopes,(size_t)body->occurrences,occurrence,(size_t)body->sites,site,
                                            (size_t)body->publications)){
    free(realized);mesh_graph_scopes_free(scopes);return realized?EINVAL:ENOMEM;
  }
  /* the realized arrays are views of the same record bytes the scopes hold; a fresh signature's attachment, its
     answered reads made once its pairs are (record_restore, mesh_graph_archive_read) */
  mesh_graph_scopes_retain(scopes);
  *realized=(struct mesh_graph_realized){.references=1,.release=scopes_released,.owner=scopes};
  s->scopes=scopes;s->realized=realized;s->occurrences=(size_t)body->occurrences;s->occurrence=occurrence;
  s->sites=(size_t)body->sites;s->site=site;s->publications=(size_t)body->publications;s->published=published;
  return 0;
}
static size_t record_size(const struct mesh_graph_signature *s){
  struct record_part original[7],program[4];return record_parts(s,original)+program_parts(s,program);
}
int mesh_graph_record(const struct mesh_graph_signature *s, void *record, size_t room, size_t *made) {
  struct record_part part[7];
  struct record_part program[4];
  const size_t primary=record_parts(s,part),metadata=program_parts(s,program);
  *made = primary+metadata;
  if (*made > room) return ERANGE;
  struct mesh_graph_record *head = record;
  memset(record, 0, *made);
  *head = (struct mesh_graph_record){.bytes = *made, .processors = s->processors, .tasks = s->tasks,
                                     .operations = s->operations, .dependences = s->dependences,
                                     .synchronized = s->synchronized};
  memcpy(head->digest, s->digest, sizeof head->digest);
  size_t at = sizeof *head;
  for (int k = 0; k < 7; k++) { memcpy((char *)record + at, part[k].data, part[k].size); at = record_align(at + part[k].size); }
  if(metadata){
    const struct program_record body={s->scopes?GRAPH_PROGRAM_SCOPED:s->repeats||s->proofs?GRAPH_PROGRAM_REPEATED:
                                      GRAPH_PROGRAM_RECORD,s->events,s->spans,s->repeats,s->proofs,
                                      s->scopes?s->scopes->rows:0,s->occurrences,s->sites,s->publications};
    memcpy((char *)record+at,&body,program_head(s));at+=program_head(s);
    for(int k=0;k<4;k++){if(program[k].size)memcpy((char *)record+at,program[k].data,program[k].size);at+=program[k].size;}
    program_scoped_write(s,(char *)record+at);
  }
  return 0;
}
/* a record's array at `at`, `size` bytes: with `archived` its view (an empty one at the archive's start, so that every
   view lies within the archive's bytes), else the record's copy in a new array of `room` bytes */
static void *record_array(const void *record, size_t at, size_t size, size_t room,
                          const struct mesh_graph_archived *archived) {
  if (archived) return (void *)(size ? (const char *)record + at : archived->bytes);
  void *made = malloc(room);
  if (made && size) memcpy(made, (const char *)record + at, size);
  return made;
}
/* a record's signature: with `archived` (mesh_graph_archive_read) every array a view of the archive holding the
   record, its digest the record's, nothing derived; without (mesh_graph_record_read, the tool's conversion) the primary
   and program arrays copied and the rest derived, the scoped section's views of the record */
static int record_restore(const void *record, struct mesh_graph_signature *s, struct mesh_graph_archived *archived) {
  const struct mesh_graph_record *head = record;
  memset(s, 0, sizeof *s);
  s->archived = archived;
  s->processors = head->processors; s->tasks = head->tasks; s->operations = head->operations;
  s->dependences = head->dependences; s->synchronized = head->synchronized;
  struct record_part part[7];
  const size_t primary=record_parts(s,part);
  int result = primary > head->bytes ? EINVAL : 0;
  const size_t room[7] = {((size_t)s->processors + 1) * sizeof *s->processor, ((size_t)s->tasks + 1) * sizeof *s->task,
                          (s->dependences + 1) * sizeof *s->dependence, (s->operations + 1) * sizeof *s->operation,
                          (s->synchronized + 1) * sizeof *s->edge, (s->dependences + 1) * sizeof *s->least,
                          (s->operations + 1) * sizeof *s->alias};
  void *array[7] = {0};
  size_t at = sizeof *head;
  for (int k = 0; !result && k < 7; k++) {
    if (!(array[k] = record_array(record, at, part[k].size, room[k], archived))) result = ENOMEM;
    at = record_align(at + part[k].size);
  }
  s->processor = array[0]; s->task = array[1]; s->dependence = array[2]; s->operation = array[3]; s->edge = array[4];
  s->least = array[5]; s->alias = array[6];
  if(!result && primary<head->bytes){
    struct program_record body={0};
    if(head->bytes-primary<sizeof body.magic)result=EINVAL;
    else memcpy(&body.magic,(const char *)record+primary,sizeof body.magic);
    const size_t size=body.magic==GRAPH_PROGRAM_SCOPED?sizeof body:body.magic==GRAPH_PROGRAM_REPEATED?
                      offsetof(struct program_record,rows):offsetof(struct program_record,repeats);
    if(!result && ((body.magic!=GRAPH_PROGRAM_RECORD && body.magic!=GRAPH_PROGRAM_REPEATED &&
                    body.magic!=GRAPH_PROGRAM_SCOPED) || head->bytes-primary<size))result=EINVAL;
    if(!result){
      memcpy(&body,(const char *)record+primary,size);at=primary+size;
      const size_t space=(size_t)head->bytes-at;
      if(body.events>space/sizeof *s->event || body.spans>space/sizeof *s->span ||
         body.repeats>space/sizeof *s->repeated || body.proofs>space/sizeof *s->proof)result=EINVAL;
      if(!result){
        s->events=(size_t)body.events;s->spans=(size_t)body.spans;
        s->repeats=(size_t)body.repeats;s->proofs=(size_t)body.proofs;
        struct record_part program[4];
        const size_t metadata=program_parts(s,program),payload=metadata?metadata-program_head(s):0;
        if(body.magic!=GRAPH_PROGRAM_SCOPED && payload!=space)result=EINVAL;
        else if(payload>space)result=EINVAL;
        const size_t made[4]={(s->events+1)*sizeof *s->event,s->spans*sizeof *s->span,
                              (s->repeats+1)*sizeof *s->repeated,(s->proofs+1)*sizeof *s->proof};
        void *section[4]={0};
        /* no spans: no span array */
        for(int k=0;!result && k<4;k++){
          if(k==1 && !s->spans)continue;
          if(!(section[k]=record_array(record,at,program[k].size,made[k],archived)))result=ENOMEM;
          at+=program[k].size;
        }
        s->event=section[0];s->span=section[1];s->repeated=section[2];s->proof=section[3];
        if(!result && body.magic==GRAPH_PROGRAM_SCOPED)
          result=program_scoped_read(s,&body,(const char *)record+at,(size_t)head->bytes-at);
      }
    }
  }
  if (!result && !archived) result = signature_derive(s);
  if (!result && !archived && s->scopes && !(result = signature_answers(s))) result = signature_digest(s);
  if (!result && archived) memcpy(s->digest, head->digest, sizeof s->digest);
  if (result) mesh_graph_signature_free(s);
  return result;
}
int mesh_graph_record_read(const void *record, struct mesh_graph_signature *s) { return record_restore(record, s, NULL); }

/* design/aot-compilation-change-request.md A1; the solved signature in full: its record, then its derived indices and
   its answered reads (`answers` of them), each array 8-byte sized, so each starts 8-aligned where the archive is;
   SIGNATURE_ARCHIVE "MESHGRA2", the answered reads' version */
struct signature_archive {
  uint64_t magic, bytes, record, uncovered, redundant, unwaited, answers, states[MESH_GRAPH_STATES];
  uint32_t primitives, padding;
};
#define SIGNATURE_ARCHIVE UINT64_C(0x324152474853454d)
static size_t archive_parts(const struct mesh_graph_signature *s, struct record_part part[8]) {
  const size_t n = s->operations, cells = (size_t)s->processors * MESH_GRAPH_PRIMITIVES;
  part[0] = (struct record_part){s->position, n * sizeof *s->position};
  part[1] = (struct record_part){s->previous, n * sizeof *s->previous};
  part[2] = (struct record_part){s->waited, n * sizeof *s->waited};
  part[3] = (struct record_part){s->first, ((size_t)s->processors + 1) * sizeof *s->first};
  part[4] = (struct record_part){s->order, n * sizeof *s->order};
  part[5] = (struct record_part){s->count, cells * sizeof *s->count};
  part[6] = (struct record_part){s->unknown, cells * sizeof *s->unknown};
  part[7] = (struct record_part){s->answer, s->answers * sizeof *s->answer};
  size_t bytes = 0;
  for (int k = 0; k < 8; k++) bytes += part[k].size;
  return bytes;
}
int mesh_graph_archive(const struct mesh_graph_signature *s, void *bytes, size_t room, size_t *made) {
  struct record_part part[8];
  const size_t primary = record_size(s), derived = archive_parts(s, part);
  *made = sizeof(struct signature_archive) + primary + derived;
  if (room < *made) return ERANGE;
  struct signature_archive head = {.magic = SIGNATURE_ARCHIVE, .bytes = *made, .record = primary,
    .uncovered = s->uncovered, .redundant = s->redundant, .unwaited = s->unwaited, .answers = s->answers,
    .primitives = MESH_GRAPH_PRIMITIVES};
  for (uint32_t k = 0; k < MESH_GRAPH_STATES; k++) head.states[k] = s->states[k];
  memcpy(bytes, &head, sizeof head);
  size_t written = 0;
  int status = mesh_graph_record(s, (char *)bytes + sizeof head, primary, &written);
  if (status) return status;
  size_t at = sizeof head + primary;
  for (int k = 0; k < 8; k++) { if (part[k].size) memcpy((char *)bytes + at, part[k].data, part[k].size); at += part[k].size; }
  return 0;
}
int mesh_graph_archive_read(const void *bytes, size_t length, void (*release_owner)(void *), void *owner,
                            struct mesh_graph_signature *s) {
  memset(s, 0, sizeof *s);
  struct signature_archive head;
  if (length < sizeof head || (uintptr_t)bytes % 8) return EINVAL;
  memcpy(&head, bytes, sizeof head);
  if (head.magic != SIGNATURE_ARCHIVE || head.primitives != MESH_GRAPH_PRIMITIVES || head.bytes != length ||
      head.record > length - sizeof head || head.record < sizeof(struct mesh_graph_record) ||
      head.answers > length / sizeof *s->answer) return EINVAL;
  const struct mesh_graph_record *record = (const void *)((const char *)bytes + sizeof head);
  if (record->bytes != head.record) return EINVAL;
  /* every array a view of `bytes`, held by `archived`; its owner adopted only once the whole result is restored */
  struct mesh_graph_archived *archived = malloc(sizeof *archived);
  if (!archived) return ENOMEM;
  *archived = (struct mesh_graph_archived){.references = 1, .bytes = bytes, .length = length};
  int status = record_restore(record, s, archived);
  if (status) return status;
  s->answers = (size_t)head.answers;
  struct record_part part[8];
  if (archive_parts(s, part) != length - sizeof head - head.record) { mesh_graph_signature_free(s); return EINVAL; }
  void *view[8];
  size_t at = sizeof head + head.record;
  for (int k = 0; k < 8; k++) { view[k] = (void *)((const char *)bytes + (part[k].size ? at : 0)); at += part[k].size; }
  s->position = view[0]; s->previous = view[1]; s->waited = view[2]; s->first = view[3]; s->order = view[4];
  s->count = view[5]; s->unknown = view[6]; s->answer = view[7];
  s->uncovered = head.uncovered; s->redundant = head.redundant; s->unwaited = head.unwaited;
  for (uint32_t k = 0; k < MESH_GRAPH_STATES; k++) s->states[k] = head.states[k];
  archived->release = release_owner; archived->owner = owner;
  return 0;
}

/* design/compilation-composition-change-request.md items 3, 6.4 and 7: mesh-graph.h mesh_graph_relation_save. The
   relations a relation references (its parent; a root's deciding relations, mesh_graph_relation_decided's; theirs),
   each once, in mesh_graph_search's postorder from it (every reference to an earlier relation, the relation itself
   last): struct relation_archive, then each relation's struct relation_entry and its arrays, each at an 8-byte
   boundary (record_align): a root's views (pairs zeroed), every view's pairs, its conflicts and witness index, its
   origin records (struct relation_origin: the deciding relation's ID, UINT64_MAX for none, and its witness), its
   deciders in the order first answered (struct relation_decider: ID, and its map's presence and length), its own map
   and each decider's map (uint32_t view identities); every relation's edge, first, member, row_first, row_edge,
   in_first and in_edge; a derived relation's source_first, source and operand (a quotient's class_of, a
   contraction's kept, over its parent's vertices; a newest projection's count and lag); its synchronization mask where
   it was computed */
_Static_assert(sizeof(size_t) == 8 && sizeof(struct mesh_graph_view) == 48 && sizeof(struct mesh_graph_conflict) == 40 &&
               sizeof(struct mesh_graph_dependence) == 16 && sizeof(struct mesh_graph_origin) == 24,
               "relation archive: native arrays of 8-byte alignment");
struct relation_archive { uint64_t magic, bytes; uint32_t relations, padding; };
struct relation_entry { uint64_t views, conflicts, pairs, witnesses, edges, members, sources, operand, kept, deciders;
                        uint32_t vertices, kind, query, synchronized, parent, origins, mapped, padding; };
struct relation_origin { uint64_t relation; struct mesh_graph_witness witness; };
_Static_assert(sizeof(struct relation_origin) == sizeof(struct mesh_graph_origin), "relation_origin");
/* a root's decider: its ID, and its map's presence and length (its root's views) */
struct relation_decider { uint64_t relation, views; uint32_t mapped, padding; };
enum { RELATION_ROOT, RELATION_QUOTIENT, RELATION_CONTRACT, RELATION_NEWEST };
#define RELATION_ARCHIVE UINT64_C(0x334c45524748534d)
struct relation_state { int synchronized; const unsigned char *mask; size_t kept; };
struct relation_cursor { char *out; size_t at; };
static void *relation_put(struct relation_cursor *c, const void *data, size_t size) {
  char *at = c->out ? c->out + c->at : NULL;
  if (at && size) memcpy(at, data, size);
  c->at = record_align(c->at + size);
  return at;
}
/* x's place in `node` (count of them where it is none) */
static size_t relation_place(const struct mesh_graph_relation *const *node, size_t count,
                             const struct mesh_graph_relation *x) {
  size_t k = 0;
  while (k < count && node[k] != x) k++;
  return k;
}
static void *relation_grow(void *array, size_t *room, size_t need, size_t width) {
  if (need <= *room) return array;
  const size_t more = need > SIZE_MAX / 2 / width ? need : 2 * need;
  void *grown = realloc(array, more * width);
  if (grown) *room = more;
  return grown;
}
/* the relations `r` references, each once: the relation's own operands (its parent; a root's deciders) found from r,
   then ordered by mesh_graph_search's postorder over those operands (every operand before its user, r last) */
static int relation_graph(const struct mesh_graph_relation *r, const struct mesh_graph_relation ***out,
                          uint32_t *count) {
  *out = NULL; *count = 0;
  size_t rooms = 8, heads = 0, headroom = 8, n = 1;
  const struct mesh_graph_relation **node = malloc(rooms * sizeof *node);
  size_t *first = malloc((rooms + 1) * sizeof *first);
  uint32_t *head = malloc(headroom * sizeof *head);
  int status = node && first && head ? 0 : ENOMEM;
  if (!status) node[0] = r;
  for (size_t i = 0; !status && i < n; i++) {
    const struct mesh_graph_relation *x = node[i];
    first[i] = heads;
    const size_t operands = x->parent ? 1 : x->deciders;
    for (size_t k = 0; !status && k < operands; k++) {
      const struct mesh_graph_relation *y = x->parent ? x->parent : x->decider[k];
      size_t j = relation_place(node, n, y);
      if (j == n) {
        const struct mesh_graph_relation **grown = relation_grow(node, &rooms, n + 1, sizeof *node);
        if (grown) node = grown;
        size_t *firsts = grown ? realloc(first, (rooms + 1) * sizeof *first) : NULL;
        if (firsts) first = firsts;
        if (!grown || !firsts || n + 1 >= UINT32_MAX) { status = ENOMEM; break; }
        node[n++] = y;
      }
      uint32_t *heads_grown = relation_grow(head, &headroom, heads + 1, sizeof *head);
      if (!heads_grown) { status = ENOMEM; break; }
      head = heads_grown;
      head[heads++] = (uint32_t)j;
    }
  }
  struct mesh_graph_search *search = NULL;
  const uint32_t start = 0;
  if (!status) {
    first[n] = heads;
    status = mesh_graph_search((uint32_t)n, first, mesh_graph_row_head, head, 1, &start, 0, &search);
  }
  const struct mesh_graph_relation **ordered = NULL;
  if (!status && !(ordered = malloc(n * sizeof *ordered))) status = ENOMEM;
  for (uint32_t k = 0; !status && k < search->count; k++) ordered[k] = node[search->postorder[k]];
  if (!status) { *out = ordered; *count = search->count; }
  else free(ordered);
  mesh_graph_search_free(search);
  free(node); free(first); free(head);
  return status;
}
static void relation_emit(struct relation_cursor *c, const struct mesh_graph_relation *const *chain,
                          const struct relation_state *state, uint32_t n) {
  const struct relation_archive head = {RELATION_ARCHIVE, 0, n, 0};
  struct relation_archive *written = relation_put(c, &head, sizeof head);
  for (uint32_t k = 0; k < n; k++) {
    const struct mesh_graph_relation *r = chain[k];
    const struct mesh_graph_relation_data *d = &r->data;
    struct relation_entry e = {.edges = d->edges, .members = d->first[d->edges], .kept = state[k].kept,
                               .vertices = d->vertices, .query = d->query,
                               .synchronized = (uint32_t)!!state[k].synchronized, .parent = UINT32_MAX};
    if (r->parent) {
      e.kind = r->newest_lag ? RELATION_NEWEST : d->class_of ? RELATION_QUOTIENT : RELATION_CONTRACT;
      e.sources = d->source_first[d->edges];
      e.operand = r->parent->data.vertices;
      e.parent = (uint32_t)relation_place(chain, n, r->parent);
    } else {
      e.kind = RELATION_ROOT;
      e.views = d->views; e.conflicts = d->conflicts; e.witnesses = r->witnesses;
      e.origins = d->origin != NULL; e.deciders = r->deciders; e.mapped = r->map != NULL;
      for (size_t v = 0; v < d->views; v++) e.pairs += 2 * (uint64_t)d->view[v].view.dims;
    }
    relation_put(c, &e, sizeof e);
    if (!r->parent) {
      struct mesh_graph_view *views = relation_put(c, d->view, d->views * sizeof *d->view);
      for (size_t v = 0; views && v < d->views; v++) views[v].view.pairs = NULL;
      for (size_t v = 0; v < d->views; v++)
        relation_put(c, d->view[v].view.pairs, 2 * (size_t)d->view[v].view.dims * sizeof(uint64_t));
      relation_put(c, d->conflict, d->conflicts * sizeof *d->conflict);
      relation_put(c, r->witness, r->witnesses * sizeof *r->witness);
      for (size_t q = 0; e.origins && q < d->conflicts; q++) {
        const struct mesh_graph_origin *o = d->origin + q;
        const struct relation_origin saved = {o->relation ? relation_place(chain, n, o->relation) : UINT64_MAX,
                                              o->witness};
        relation_put(c, &saved, sizeof saved);
      }
      for (size_t q = 0; q < r->deciders; q++) {
        const uint32_t *map = r->decider_map ? r->decider_map[q] : NULL;
        const struct relation_decider saved = {relation_place(chain, n, r->decider[q]),
                                               map ? relation_of_root(r->decider[q])->data.views : 0, map != NULL, 0};
        relation_put(c, &saved, sizeof saved);
      }
      if (r->map) relation_put(c, r->map, d->views * sizeof *r->map);
      for (size_t q = 0; q < r->deciders; q++)
        if (r->decider_map && r->decider_map[q])
          relation_put(c, r->decider_map[q], relation_of_root(r->decider[q])->data.views * sizeof *r->decider_map[q]);
    }
    relation_put(c, d->edge, d->edges * sizeof *d->edge);
    relation_put(c, d->first, (d->edges + 1) * sizeof *d->first);
    relation_put(c, d->member, e.members * sizeof *d->member);
    relation_put(c, d->row_first, ((size_t)d->vertices + 1) * sizeof *d->row_first);
    relation_put(c, d->row_edge, d->edges * sizeof *d->row_edge);
    relation_put(c, d->in_first, ((size_t)d->vertices + 1) * sizeof *d->in_first);
    relation_put(c, d->in_edge, d->edges * sizeof *d->in_edge);
    if (r->parent) {
      relation_put(c, d->source_first, (d->edges + 1) * sizeof *d->source_first);
      relation_put(c, d->source, e.sources * sizeof *d->source);
      if (r->newest_lag) {
        const uint32_t scalars[2] = {r->newest_count, r->newest_lag};
        relation_put(c, scalars, sizeof scalars);
      } else if (d->class_of) relation_put(c, d->class_of, (size_t)e.operand * sizeof *d->class_of);
      else relation_put(c, d->kept, (size_t)e.operand);
    }
    if (state[k].synchronized) relation_put(c, state[k].mask, d->edges);
  }
  if (written) written->bytes = c->at;
}
int mesh_graph_relation_save(const struct mesh_graph_relation *r, void **bytes, size_t *length) {
  *bytes = NULL; *length = 0;
  const struct mesh_graph_relation **chain = NULL;
  uint32_t n = 0;
  int status = relation_graph(r, &chain, &n);
  if (status) return status;
  struct relation_state *state = malloc((size_t)n * sizeof *state);
  if (!state) { free(chain); return ENOMEM; }
  for (uint32_t k = 0; k < n; k++) {
    struct mesh_graph_relation *x = (struct mesh_graph_relation *)(uintptr_t)chain[k];
    pthread_mutex_lock(&x->lock);
    state[k] = (struct relation_state){x->synchronized, x->mask, x->kept};
    pthread_mutex_unlock(&x->lock);
  }
  struct relation_cursor c = {NULL, 0};
  relation_emit(&c, chain, state, n);
  char *out = calloc(1, c.at);
  if (out) {
    c = (struct relation_cursor){out, 0};
    relation_emit(&c, chain, state, n);
    *bytes = out; *length = c.at;
  }
  free(chain); free(state);
  return out ? 0 : ENOMEM;
}
/* the caller's archive owner, shared by every loaded root (NULL release until success), and one root's own
   relocated headers: its views and its origin records */
struct relation_loaded { size_t roots; void (*release)(void *); void *owner; };
struct relation_root {
  struct relation_loaded *shared;
  struct mesh_graph_view *view;
  struct mesh_graph_origin *origin;
};
static void relation_root_release(void *owner) {
  struct relation_root *o = owner;
  struct relation_loaded *shared = o->shared;
  free(o->view); free(o->origin); free(o);
  if (__atomic_fetch_sub(&shared->roots, 1, __ATOMIC_ACQ_REL) == 1) {
    if (shared->release) shared->release(shared->owner);
    free(shared);
  }
}
struct relation_reader { const char *bytes; size_t at, length; int status; };
static const void *relation_take(struct relation_reader *x, uint64_t count, size_t width) {
  if (x->status) return NULL;
  if (x->at > x->length || count > (x->length - x->at) / width) { x->status = EINVAL; return NULL; }
  const void *at = x->bytes + x->at;
  x->at = record_align(x->at + (size_t)count * width);
  return at;
}
static int relation_index(const size_t *x, size_t n, size_t bound) {
  for (size_t i = 0; i < n; i++) if (x[i] >= bound) return EINVAL;
  return 0;
}
static int relation_csr(const size_t *first, size_t rows, size_t total) {
  if (first[0]) return EINVAL;
  for (size_t i = 0; i < rows; i++) if (first[i + 1] < first[i]) return EINVAL;
  return first[rows] == total ? 0 : EINVAL;
}
/* one relation's arrays as borrowed spans of the archive, each index checked against what it indexes; a root's
   origins and deciders refer to the relations made before it (made[0 .. at)) */
static int relation_restore(struct relation_reader *x, struct mesh_graph_relation *r, const struct relation_entry *e,
                            struct mesh_graph_relation *const *made, uint32_t at, struct relation_root *root) {
  struct mesh_graph_relation_data *d = &r->data;
  if (!r->parent) {
    const struct mesh_graph_view *stored = relation_take(x, e->views, sizeof *stored);
    const uint64_t *pairs = relation_take(x, e->pairs, sizeof *pairs);
    d->conflict = relation_take(x, e->conflicts, sizeof *d->conflict);
    r->witness = (size_t *)(uintptr_t)relation_take(x, e->witnesses, sizeof *r->witness);
    const struct relation_origin *origin = relation_take(x, e->origins ? e->conflicts : 0, sizeof *origin);
    const struct relation_decider *decider = relation_take(x, e->deciders, sizeof *decider);
    const uint32_t *map = relation_take(x, e->mapped ? e->views : 0, sizeof *map);
    if (x->status) return x->status;
    if (e->origins > 1) return EINVAL;
    if (!(root->view = malloc(((size_t)e->views + 1) * sizeof *root->view))) return ENOMEM;
    uint64_t used = 0;
    for (size_t v = 0; v < e->views; v++) {
      const uint64_t count = 2 * (uint64_t)stored[v].view.dims;
      if (count > e->pairs - used) return EINVAL;
      root->view[v] = stored[v];
      root->view[v].view.pairs = count ? pairs + used : NULL;
      used += count;
    }
    if (used != e->pairs) return EINVAL;
    d->vertices = e->vertices; d->views = (size_t)e->views; d->view = root->view;
    d->conflicts = (size_t)e->conflicts; d->query = e->query; r->witnesses = (size_t)e->witnesses;
    for (size_t k = 0; k < d->conflicts; k++) {
      const struct mesh_graph_conflict *y = d->conflict + k;
      if (y->from >= d->vertices || y->to >= d->vertices || (y->from_view != SIZE_MAX && y->from_view >= d->views) ||
          (y->to_view != SIZE_MAX && y->to_view >= d->views)) return EINVAL;
    }
    if (relation_index(r->witness, r->witnesses, d->conflicts)) return EINVAL;
    if (e->origins) {
      if (!(root->origin = malloc((d->conflicts + 1) * sizeof *root->origin))) return ENOMEM;
      for (size_t k = 0; k < d->conflicts; k++) {
        const struct relation_origin *o = origin + k;
        const struct mesh_graph_relation *by = o->relation == UINT64_MAX ? NULL
                                                : o->relation < at ? made[o->relation] : r;
        if (by == r || o->witness.known > 1 || o->witness.reverse > 1 ||
            (by && o->witness.record != SIZE_MAX && o->witness.record >= relation_of_root(by)->data.conflicts))
          return EINVAL;
        root->origin[k] = (struct mesh_graph_origin){by, o->witness};
      }
      r->origin = root->origin; d->origin = root->origin;
    }
    if (e->mapped > 1) return EINVAL;
    if (!(r->decider = malloc(((size_t)e->deciders + 1) * sizeof *r->decider)) ||
        !(r->decider_map = malloc(((size_t)e->deciders + 1) * sizeof *r->decider_map))) return ENOMEM;
    for (; r->deciders < e->deciders; r->deciders++) {
      const struct relation_decider *g = decider + r->deciders;
      if (g->relation >= at || g->mapped > 1 ||
          (g->mapped && g->views != relation_of_root(made[g->relation])->data.views)) return EINVAL;
      r->decider[r->deciders] = made[g->relation];
      r->decider_map[r->deciders] = g->mapped ? relation_take(x, g->views, sizeof *r->decider_map[r->deciders]) : NULL;
      if (x->status) return x->status;
      mesh_graph_relation_retain(r->decider[r->deciders]);
    }
    r->map = e->mapped ? map : NULL;
  }
  r->edge = (struct mesh_graph_dependence *)(uintptr_t)relation_take(x, e->edges, sizeof *r->edge);
  r->first = (size_t *)(uintptr_t)relation_take(x, e->edges + 1, sizeof *r->first);
  r->member = (size_t *)(uintptr_t)relation_take(x, e->members, sizeof *r->member);
  r->row_first = (size_t *)(uintptr_t)relation_take(x, (uint64_t)e->vertices + 1, sizeof *r->row_first);
  r->row_edge = (size_t *)(uintptr_t)relation_take(x, e->edges, sizeof *r->row_edge);
  r->in_first = (size_t *)(uintptr_t)relation_take(x, (uint64_t)e->vertices + 1, sizeof *r->in_first);
  r->in_edge = (size_t *)(uintptr_t)relation_take(x, e->edges, sizeof *r->in_edge);
  if (r->parent) {
    r->source_first = (size_t *)(uintptr_t)relation_take(x, e->edges + 1, sizeof *r->source_first);
    r->source = (size_t *)(uintptr_t)relation_take(x, e->sources, sizeof *r->source);
    if (e->kind == RELATION_QUOTIENT) d->class_of = relation_take(x, e->operand, sizeof *d->class_of);
    else if (e->kind == RELATION_CONTRACT) d->kept = relation_take(x, e->operand, 1);
    else {
      const uint32_t *scalars = relation_take(x, 2, sizeof *scalars);
      if (scalars) { r->newest_count = scalars[0]; r->newest_lag = scalars[1]; }
    }
  }
  if (e->synchronized) { r->mask = relation_take(x, e->edges, 1); r->synchronized = 1; r->kept = (size_t)e->kept; }
  if (x->status) return x->status;
  const size_t edges = (size_t)e->edges;
  d->edges = edges; d->edge = r->edge; d->first = r->first; d->member = r->member;
  d->row_first = r->row_first; d->row_edge = r->row_edge; d->in_first = r->in_first; d->in_edge = r->in_edge;
  d->source_first = r->source_first; d->source = r->source;
  for (size_t k = 0; k < edges; k++) if (r->edge[k].from >= d->vertices || r->edge[k].to >= d->vertices) return EINVAL;
  if (relation_csr(r->first, edges, (size_t)e->members) || relation_index(r->member, (size_t)e->members, d->conflicts) ||
      relation_csr(r->row_first, d->vertices, edges) || relation_index(r->row_edge, edges, edges) ||
      relation_csr(r->in_first, d->vertices, edges) || relation_index(r->in_edge, edges, edges)) return EINVAL;
  if (r->parent) {
    if (e->operand != r->parent->data.vertices || relation_csr(r->source_first, edges, (size_t)e->sources) ||
        relation_index(r->source, (size_t)e->sources, r->parent->data.edges)) return EINVAL;
    for (size_t v = 0; d->class_of && v < e->operand; v++) if (d->class_of[v] >= d->vertices) return EINVAL;
    if (e->kind == RELATION_NEWEST && (!r->newest_lag || r->newest_count != d->vertices ||
                                       r->parent->data.vertices > (uint64_t)r->newest_lag * r->newest_count)) return EINVAL;
  }
  return 0;
}
int mesh_graph_relation_load(const void *bytes, size_t length, void (*release)(void *), void *owner,
                             struct mesh_graph_relation **out) {
  *out = NULL;
  if ((uintptr_t)bytes % 8) return EINVAL;
  struct relation_reader x = {bytes, 0, length, 0};
  const struct relation_archive *head = relation_take(&x, 1, sizeof *head);
  if (!head || head->magic != RELATION_ARCHIVE || head->bytes != length || !head->relations ||
      head->relations > length / sizeof(struct relation_entry)) return EINVAL;
  const uint32_t n = head->relations;
  struct mesh_graph_relation **made = calloc(n, sizeof *made);
  struct relation_loaded *shared = calloc(1, sizeof *shared);
  int status = made && shared ? 0 : ENOMEM;
  uint32_t count = 0;
  for (; !status && count < n; count++) {
    const struct relation_entry *e = relation_take(&x, 1, sizeof *e);
    if (!e || e->kind > RELATION_NEWEST || e->query > MESH_GRAPH_QUERY_SELF ||
        (e->kind == RELATION_ROOT) != (e->parent == UINT32_MAX) || (e->parent != UINT32_MAX && e->parent >= count)) {
      status = EINVAL;
      break;
    }
    struct mesh_graph_relation *m = relation_new();
    struct relation_root *root = m && e->parent == UINT32_MAX ? calloc(1, sizeof *root) : NULL;
    if (!m || (e->parent == UINT32_MAX && !root)) { if (m) relation_drop(m); status = ENOMEM; break; }
    m->loaded = 1;
    made[count] = m;
    if (root) { root->shared = shared; shared->roots++; m->release = relation_root_release; m->owner = root; }
    else {
      relation_derived(m, made[e->parent], e->vertices, (size_t)e->edges);
      mesh_graph_relation_retain(made[e->parent]);
    }
    status = relation_restore(&x, m, e, made, count, root);
    if (status) { count++; break; }
  }
  if (!status && x.at != length) status = EINVAL;
  if (!status) { shared->release = release; shared->owner = owner; }
  const size_t roots = shared ? shared->roots : 0;
  for (uint32_t k = 0; k < count; k++) if (status || k + 1 < n) mesh_graph_relation_free(made[k]);
  if (!roots) free(shared);
  if (!status) *out = made[n - 1];
  free(made);
  return status;
}

/* design/synchronization-signature-change-request.md item 1; L. Lamport, CACM 21(7), 1978 */
struct compose_key { uint32_t processor, object; uint64_t value; size_t index; };
static int compose_key_before(const void *a, const void *b) {
  const struct compose_key *x = a, *y = b;
  if (x->processor != y->processor) return x->processor < y->processor ? -1 : 1;
  if (x->object != y->object) return x->object < y->object ? -1 : 1;
  if (x->value != y->value) return x->value < y->value ? -1 : 1;
  return (x->index > y->index) - (x->index < y->index);
}
static uint32_t compose_processor(struct mesh_graph_signature *s, struct mesh_graph_processor p) {
  for (uint32_t q = 0; q < s->processors; q++)
    if (s->processor[q].kind == p.kind && s->processor[q].index == p.index) return q;
  s->processor[s->processors] = p;
  return s->processors++;
}
/* design/synchronization-signature-change-request.md items 1 and 6: a composed signature's attachment, both
   operands' occurrences kept (mesh-graph.h mesh_graph_compose): a's rows, sites, publication events and tasks first,
   b's after them (b's row r composed row a's rows + r, its own for the same scopes; its site k a's sites + k; its event
   k a's publications + k; its task t a's tasks + t), each operand's scopes, source binding and realized holder
   retained, so each original owner is kept, and only the composed descriptors (their sources borrowed as they are),
   bound source pointers, occurrences, sites and published values made; one operand's realized arrays (its sites'
   tasks moved for b) where the other has no scopes. The composed scopes are always the result's own, so that a later
   mesh_graph_scopes_bind of an operand's scopes leaves the binding projected here, and its holder, as they were */
static int compose_scopes(struct mesh_graph_signature *s, const struct mesh_graph_signature *a,
                          const struct mesh_graph_signature *b) {
  const int both = a->scopes && b->scopes, same = a->scopes == b->scopes;
  const struct mesh_graph_signature *g = b->scopes ? b : a;
  struct mesh_graph_scopes *const part[2] = {both && !same ? a->scopes : NULL, g->scopes};
  const size_t rows = part[0] ? part[0]->rows : 0, total = rows + g->scopes->rows, lead = both ? a->sites : 0,
               sites = lead + g->sites, occurrences = (both ? a->occurrences : 0) + g->occurrences,
               publications = (both ? a->publications : 0) + g->publications;
  const int bound = (part[0] && part[0]->bound) || g->scopes->bound;
  /* the occurrence's uint32_t row, site and event indices */
  if (both && (total > UINT32_MAX || sites > UINT32_MAX || publications > UINT32_MAX)) return EOVERFLOW;
  struct mesh_graph_site *projected = b->scopes && sites ? malloc(sites * sizeof *projected) : NULL;
  struct mesh_graph_occurrence *occurrence =
    both ? malloc(occurrences * sizeof *occurrence + publications * sizeof *s->published + 1) : NULL;
  struct mesh_graph_realized *realized = both ? malloc(sizeof *realized) : NULL;
  struct mesh_graph_scopes_data *row = malloc((total + 1) * sizeof *row);
  struct mesh_graph_scopes *scopes = malloc(sizeof *scopes);
  const void **pointer = bound ? malloc((total + 1) * sizeof *pointer) : NULL;
  struct mesh_graph_realized *binding = bound ? malloc(sizeof *binding) : NULL;
  if ((b->scopes && sites && !projected) || (both && (!occurrence || !realized)) || !row || !scopes ||
      (bound && (!pointer || !binding))) {
    free(projected); free(occurrence); free(realized); free(row); free(scopes); free(pointer); free(binding);
    return ENOMEM;
  }
  struct signature_attached x = {.realized = g->realized, .occurrences = g->occurrences, .sites = sites,
                                 .publications = g->publications, .occurrence = g->occurrence, .site = g->site,
                                 .published = g->published};
  if (projected) {
    if (lead) memcpy(projected, a->site, lead * sizeof *projected);
    for (size_t k = 0; k < b->sites; k++) {
      struct mesh_graph_site y = b->site[k];
      y.first += a->tasks;
      if (y.later != MESH_GRAPH_NONE) y.later += a->tasks;
      projected[lead + k] = y;
    }
    x.site = x.projected = projected;
  }
  if (both) {
    uint64_t *published = (uint64_t *)(void *)(occurrence + occurrences);
    if (a->occurrences) memcpy(occurrence, a->occurrence, a->occurrences * sizeof *occurrence);
    for (size_t k = 0; k < b->occurrences; k++) {
      struct mesh_graph_occurrence y = b->occurrence[k];
      y.row += (uint32_t)rows; y.site += (uint32_t)a->sites; y.event += (uint32_t)a->publications;
      occurrence[a->occurrences + k] = y;
    }
    if (a->publications) memcpy(published, a->published, a->publications * sizeof *published);
    if (b->publications) memcpy(published + a->publications, b->published, b->publications * sizeof *published);
    realized_retain(a->realized); realized_retain(b->realized);
    *realized = (struct mesh_graph_realized){.references = 1, .own = occurrence, .part = {a->realized, b->realized}};
    x.realized = realized; x.occurrences = occurrences; x.publications = publications;
    x.occurrence = occurrence; x.published = published;
  } else realized_retain(x.realized);
  /* the descriptors and bound pointers of part[0] (a's, different scopes) then part[1] (the scoped or same scopes) */
  *scopes = (struct mesh_graph_scopes){.references = 1, .rows = total, .row = row, .own = row,
                                       .part = {part[0], part[1]}};
  size_t at = 0;
  for (int k = 0; k < 2; k++) {
    if (!part[k]) continue;
    mesh_graph_scopes_retain(part[k]);
    if (part[k]->rows) memcpy(row + at, part[k]->row, part[k]->rows * sizeof *row);
    for (size_t r = 0; bound && r < part[k]->rows; r++) pointer[at + r] = part[k]->source ? part[k]->source[r] : NULL;
    if (part[k]->bound) realized_retain(part[k]->bound);
    at += part[k]->rows;
  }
  if (bound) {
    *binding = (struct mesh_graph_realized){.references = 1, .own = pointer,
                                            .part = {part[0] ? part[0]->bound : NULL, part[1]->bound}};
    scopes->source = pointer; scopes->bound = binding;
  }
  x.scopes = scopes;
  const int result = signature_attach(s, x);
  if (result) { free(projected); realized_free(x.realized); mesh_graph_scopes_free(x.scopes); }
  return result;
}
int mesh_graph_compose(const struct mesh_graph_signature *a, const struct mesh_graph_signature *b,
                       struct mesh_graph_signature *s) {
  memset(s, 0, sizeof *s);
  if ((uint64_t)a->tasks + b->tasks >= UINT32_MAX) return ENOMEM;
  const size_t operations = a->operations + b->operations, dependences = a->dependences + b->dependences;
  s->processor = malloc(((size_t)a->processors + b->processors + 1) * sizeof *s->processor);
  s->task = malloc(((size_t)a->tasks + b->tasks + 1) * sizeof *s->task);
  s->dependence = malloc((dependences + 1) * sizeof *s->dependence);
  s->least = malloc((dependences + 1) * sizeof *s->least);
  s->operation = malloc((operations + 1) * sizeof *s->operation);
  s->alias = malloc((operations + 1) * sizeof *s->alias);
  s->edge = malloc((a->synchronized + b->synchronized + 1) * sizeof *s->edge);
  uint32_t *pa = malloc(((size_t)a->processors + 1) * sizeof *pa), *pb = malloc(((size_t)b->processors + 1) * sizeof *pb);
  size_t *ia = malloc((a->operations + 1) * sizeof *ia), *ib = malloc((b->operations + 1) * sizeof *ib);
  size_t *matched = malloc((b->operations + 1) * sizeof *matched), *cursor = calloc((size_t)a->processors + 1, sizeof *cursor);
  struct compose_key *key = malloc((a->operations + 1) * sizeof *key);
  unsigned char *taken = calloc(a->operations + 1, 1);
  int result = s->processor && s->task && s->dependence && s->least && s->operation && s->alias && s->edge && pa && pb &&
               ia && ib && matched && cursor && key && taken ? 0 : ENOMEM;
  for (uint32_t p = 0; !result && p < a->processors; p++) pa[p] = compose_processor(s, a->processor[p]);
  for (uint32_t p = 0; !result && p < b->processors; p++) pb[p] = compose_processor(s, b->processor[p]);
  s->tasks = a->tasks + b->tasks; s->dependences = dependences;
  for (uint32_t t = 0; !result && t < a->tasks; t++) { s->task[t] = a->task[t]; s->task[t].processor = pa[a->task[t].processor]; }
  for (uint32_t t = 0; !result && t < b->tasks; t++) {
    s->task[a->tasks + t] = b->task[t]; s->task[a->tasks + t].processor = pb[b->task[t].processor];
  }
  for (size_t k = 0; !result && k < a->dependences; k++) { s->dependence[k] = a->dependence[k]; s->least[k] = a->least[k]; }
  for (size_t k = 0; !result && k < b->dependences; k++) {
    s->dependence[a->dependences + k] = (struct mesh_graph_dependence){b->dependence[k].from + a->tasks,
                                                                       b->dependence[k].to + a->tasks, b->dependence[k].delay};
    s->least[a->dependences + k] = b->least[k];
  }
  size_t keys = 0;
  for (size_t o = 0; !result && o < a->operations; o++)
    if (a->operation[o].primitive == MESH_GRAPH_BRIDGE_START)
      key[keys++] = (struct compose_key){pa[a->operation[o].processor], a->operation[o].object, a->operation[o].value, o};
  if (!result) qsort(key, keys, sizeof *key, compose_key_before);
  for (size_t o = 0; !result && o < b->operations; o++) {
    const struct mesh_graph_operation *x = b->operation + o;
    matched[o] = SIZE_MAX;
    if (x->primitive != MESH_GRAPH_BRIDGE_START) continue;
    const struct compose_key probe = {pb[x->processor], x->object, x->value, 0};
    for (size_t k = graph_lower(key, keys, sizeof *key, &probe, compose_key_before);
         k < keys && key[k].processor == probe.processor && key[k].object == probe.object && key[k].value == probe.value; k++)
      if (!taken[key[k].index]) { taken[key[k].index] = 1; matched[o] = key[k].index; break; }
  }
  size_t n = 0;
  for (size_t o = 0; !result && o < b->operations; o++) {
    const size_t y = matched[o];
    if (y != SIZE_MAX) {
      const uint32_t p = a->operation[y].processor;
      for (; cursor[p] < a->position[y]; cursor[p]++) {
        const size_t z = a->order[a->first[p] + cursor[p]];
        if (taken[z]) continue;
        s->operation[n] = a->operation[z]; s->operation[n].processor = pa[p]; s->alias[n] = a->alias[z]; ia[z] = n++;
      }
      if (cursor[p] == a->position[y]) cursor[p]++;
    }
    const struct mesh_graph_operation *x = b->operation + o;
    s->operation[n] = *x;
    s->operation[n].processor = pb[x->processor];
    s->operation[n].vertex = x->vertex == MESH_GRAPH_NONE ? MESH_GRAPH_NONE : x->vertex + a->tasks;
    s->alias[n] = y != SIZE_MAX ? a->operation[y].vertex : b->alias[o] == MESH_GRAPH_NONE ? MESH_GRAPH_NONE : b->alias[o] + a->tasks;
    if (y != SIZE_MAX) ia[y] = n;
    ib[o] = n++;
  }
  for (uint32_t p = 0; !result && p < a->processors; p++)
    for (; cursor[p] < a->first[p + 1] - a->first[p]; cursor[p]++) {
      const size_t z = a->order[a->first[p] + cursor[p]];
      if (taken[z]) continue;
      s->operation[n] = a->operation[z]; s->operation[n].processor = pa[p]; s->alias[n] = a->alias[z]; ia[z] = n++;
    }
  s->operations = n;
  for (size_t k = 0; !result && k < a->synchronized; k++) {
    struct mesh_graph_synchronized e = a->edge[k];
    e.signal = e.signal == MESH_GRAPH_NONE ? MESH_GRAPH_NONE : (uint32_t)ia[e.signal];
    e.wait = (uint32_t)ia[e.wait];
    s->edge[s->synchronized++] = e;
  }
  for (size_t k = 0; !result && k < b->synchronized; k++) {
    struct mesh_graph_synchronized e = b->edge[k];
    e.signal = e.signal == MESH_GRAPH_NONE ? MESH_GRAPH_NONE : (uint32_t)ib[e.signal];
    e.wait = (uint32_t)ib[e.wait];
    if (e.dependence != MESH_GRAPH_NONE) e.dependence += (uint32_t)a->dependences;
    s->edge[s->synchronized++] = e;
  }
  if(!result && (a->events || b->events)){
    s->events=a->events+b->events;
    s->event=malloc((s->events+1)*sizeof *s->event);
    if(!s->event)result=ENOMEM;
    for(int side=0;!result && side<2;side++){
      const struct mesh_graph_signature *g=side?b:a;
      if(g->events)memcpy(s->event+(side?a->events:0),g->event,g->events*sizeof *g->event);
    }
  }
  if(!result && (a->spans || b->spans)){
    const struct mesh_graph_signature *g=b->spans?b:a;const size_t *map=b->spans?ib:ia;
    s->spans=g->spans;s->span=malloc(s->spans*sizeof *s->span);
    if(!s->span)result=ENOMEM;
    if(a->spans && b->spans && a->spans!=b->spans)result=EINVAL;
    for(size_t k=0;!result && k<s->spans;k++){
      struct mesh_graph_span x=g->span[k];
      x.begin=x.begin?map[x.begin-1]+1:0;x.end=x.end?map[x.end-1]+1:0;
      if(a->spans && b->spans){
        const struct mesh_graph_span y=a->span[k];
        if(x.first!=y.first || x.count!=y.count){result=EINVAL;break;}
        const uint64_t begin=y.begin?ia[y.begin-1]+1:0,end=y.end?ia[y.end-1]+1:0;
        if(begin<x.begin)x.begin=begin;if(end>x.end)x.end=end;
      }
      s->span[k]=x;
    }
  }
  if (!result) result = signature_derive(s);
  if (!result && (a->scopes || b->scopes)) result = compose_scopes(s, a, b);
  free(pa); free(pb); free(ia); free(ib); free(matched); free(cursor); free(key); free(taken);
  if (result) mesh_graph_signature_free(s);
  return result;
}

/* design/synchronization-signature-change-request.md item 2 */
static const char *const signature_kind[MESH_GRAPH_KINDS] = {"queue", "host", "bridge", "kernel"};
static const char *const signature_primitive[MESH_GRAPH_PRIMITIVES] = {
  "submit", "queue_signal", "queue_wait", "fence_update", "fence_wait", "pass_barrier", "queue_barrier", "host_wait",
  "host_notify", "bridge_start", "threadgroup_barrier", "simdgroup_barrier", "atomic", "producer_barrier",
  "bridge_raise", "initial", "host_publish", "kernel_fence", "host_signal", "bridge_read"};
static const char *const signature_state[MESH_GRAPH_STATES] = {"image", "stronger", "false", "unmatched", "satisfied",
                                                              "transformed", "scoped"};
struct graph_sink { char *text; size_t room, at; int status; };
static int graph_put(struct graph_sink *t, const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  const int made = vsnprintf(t->at < t->room ? t->text + t->at : NULL, t->at < t->room ? t->room - t->at : 0, format,
                             arguments);
  va_end(arguments);
  if (made < 0) t->status = EINVAL; else t->at += (size_t)made;
  return t->status ? t->status : t->at >= t->room ? ERANGE : 0;
}
static int graph_end(struct graph_sink *t, size_t *made) {
  *made = t->at + 1;
  return t->status ? t->status : t->at >= t->room ? ERANGE : 0;
}
static void signature_number(struct graph_sink *t, uint32_t x) {
  if (x == MESH_GRAPH_NONE) graph_put(t, " -"); else graph_put(t, " %u", x);
}
static void signature_value(struct graph_sink *t, const struct mesh_graph_operation *x) {
  if (signature_static(x->primitive) && x->value == UINT64_MAX) graph_put(t, " value unknown");
  else graph_put(t, " value %llu", (unsigned long long)x->value);
}
static int signature_ordered(const struct mesh_graph_signature *s, size_t o) {
  const size_t k = s->waited[o];
  return s->previous[o] != SIZE_MAX && (k == SIZE_MAX || s->edge[k].signal != s->previous[o]);
}
static int signature_drawn(const struct mesh_graph_synchronized *e) {
  return e->signal != MESH_GRAPH_NONE && e->state != MESH_GRAPH_SATISFIED;
}
/* a scoped signature's completion scopes, occurrences, sites, publication events and answered reads, one line each
   (none for a signature without scopes: its text is as before) */
/* a scope row's source as printed: its original artifact row, native node and structural identity, or none */
static void signature_source(struct graph_sink *t, const struct mesh_graph_scopes_data *d) {
  if (!d->source) { graph_put(t, " source none"); return; }
  graph_put(t, " source artifact %u node %u identity ", d->source->row, d->source->node);
  for (size_t k = 0; k < sizeof d->source->identity; k++) graph_put(t, "%02x", d->source->identity[k]);
}
static void signature_scoped(struct graph_sink *t, const struct mesh_graph_signature *s) {
  if (!s->scopes) return;
  for (size_t row = 0; row < s->scopes->rows; row++) {
    const struct mesh_graph_scopes_data *d = s->scopes->row + row;
    graph_put(t, "%% scope_source row %zu", row);
    signature_source(t, d);
    graph_put(t, "\n");
    for (size_t c = 0; c < d->completions; c++) {
      const struct mesh_graph_completion *x = d->completion + c;
      graph_put(t, "%% scope row %zu %zu value %u counter %u total %llu writers %u+%u releases %zu+%zu\n", row, c,
                x->value, x->counter, (unsigned long long)x->total, x->writer, x->writers, d->release_first[c],
                d->release_first[c + 1] - d->release_first[c]);
    }
    for (size_t w = 0; w < d->writers; w++) {
      const struct mesh_graph_completion_writer *x = d->writer + w;
      graph_put(t, "%% scope_writer row %zu %zu dispatch %u completion %u later", row, w, x->dispatch, x->completion);
      signature_number(t, x->later);
      graph_put(t, " count %llu\n", (unsigned long long)x->count);
    }
    for (size_t r = 0; r < d->releases; r++)
      graph_put(t, "%% scope_release row %zu %zu owner %u transfer %u\n", row, r, d->release[r].owner,
                d->release[r].transfer);
  }
  for (size_t k = 0; k < s->occurrences; k++) {
    const struct mesh_graph_occurrence *x = s->occurrence + k;
    graph_put(t, "%% occurrence %zu row %u scope %u execution %llu generation %llu copy %llu sites %u events %u\n", k,
              x->row, x->completion, (unsigned long long)x->execution, (unsigned long long)x->generation,
              (unsigned long long)x->copy, x->site, x->event);
  }
  for (size_t k = 0; k < s->sites; k++) {
    const struct mesh_graph_site *x = s->site + k;
    graph_put(t, "%% site %zu first task %u later task", k, x->first);
    signature_number(t, x->later);
    graph_put(t, " native %u", x->first_site);
    signature_number(t, x->later_site);
    graph_put(t, "\n");
  }
  for (size_t k = 0; k < s->publications; k++)
    graph_put(t, "%% publication event %zu value %llu\n", k, (unsigned long long)s->published[k]);
  for (size_t k = 0; k < s->answers; k++)
    graph_put(t, "%% answered operation %u event %u\n", s->answer[k].operation + 1, s->answer[k].event);
  graph_put(t, "%% totals scoped %zu publications %zu\n", s->states[MESH_GRAPH_SCOPED], s->publications);
}
int mesh_graph_matrix_market(const struct mesh_graph_signature *s, char *text, size_t room, size_t *made) {
  struct graph_sink t = {text, room, 0, 0};
  size_t entries = 0;
  for (size_t o = 0; o < s->operations; o++) entries += signature_ordered(s, o);
  for (size_t k = 0; k < s->synchronized; k++) entries += signature_drawn(s->edge + k);
  graph_put(&t, "%%%%MatrixMarket matrix coordinate integer general\n"
                "%% synchronization signature: row and column i are operation i; entry (i, j) is an edge from "
                "operation i to operation j, its value the edge's delay in executions\n");
  for (uint32_t p = 0; p < s->processors; p++) {
    graph_put(&t, "%% processor %u %s %u", p, signature_kind[s->processor[p].kind], s->processor[p].index);
    for (uint32_t q = 0; q < MESH_GRAPH_PRIMITIVES; q++)
      if (s->count[(size_t)p * MESH_GRAPH_PRIMITIVES + q])
        graph_put(&t, " %s=%llu", signature_primitive[q], (unsigned long long)s->count[(size_t)p * MESH_GRAPH_PRIMITIVES + q]);
    for (uint32_t q = 0; q < MESH_GRAPH_PRIMITIVES; q++)
      if (s->unknown[(size_t)p * MESH_GRAPH_PRIMITIVES + q])
        graph_put(&t, " %s_unknown_commands=%llu", signature_primitive[q],
                  (unsigned long long)s->unknown[(size_t)p * MESH_GRAPH_PRIMITIVES + q]);
    graph_put(&t, "\n");
  }
  for (uint32_t k = 0; k < s->tasks; k++)
    graph_put(&t, "%% task %u processor %u place %u label %u copy %u\n", k, s->task[k].processor, s->task[k].place,
              s->task[k].label, s->task[k].copy);
  for (size_t o = 0; o < s->operations; o++) {
    const struct mesh_graph_operation *x = s->operation + o;
    graph_put(&t, "%% operation %zu processor %u position %llu %s task", o + 1, x->processor,
              (unsigned long long)s->position[o], signature_primitive[x->primitive]);
    signature_number(&t, x->vertex);
    graph_put(&t, " object %u execution %llu", x->object, (unsigned long long)x->execution);
    signature_value(&t, x);
    if (s->alias[o] != MESH_GRAPH_NONE) graph_put(&t, " alias %u", s->alias[o]);
    graph_put(&t, "\n");
  }
  for (size_t k = 0; k < s->synchronized; k++) {
    const struct mesh_graph_synchronized *e = s->edge + k;
    graph_put(&t, "%% synchronized wait %u signal", e->wait + 1);
    signature_number(&t, e->signal == MESH_GRAPH_NONE ? MESH_GRAPH_NONE : e->signal + 1);
    graph_put(&t, " %s %s delay %llu dependence", signature_state[e->state], e->redundant ? "redundant" : "kept",
              (unsigned long long)e->delay);
    signature_number(&t, e->dependence);
    graph_put(&t, " witness");
    signature_number(&t, e->witness);
    graph_put(&t, "\n");
  }
  for (size_t k = 0; k < s->dependences; k++) {
    const struct mesh_graph_dependence *d = s->dependence + k;
    graph_put(&t, "%% dependence %zu task %u task %u delay %llu least", k, d->from, d->to, (unsigned long long)d->delay);
    if (s->least[k] == UINT64_MAX) graph_put(&t, " -"); else graph_put(&t, " %llu", (unsigned long long)s->least[k]);
    graph_put(&t, " %s\n", s->least[k] <= d->delay ? "covered" : "uncovered");
  }
  for(size_t e=0;e<s->events;e++){
    const struct mesh_graph_event *x=s->event+e;
    graph_put(&t,"%% event object %u role %u argument %u lookback %llu\n",x->object,x->role,x->argument,
      (unsigned long long)x->lookback);
  }
  for(size_t k=0;k<s->spans;k++){
    const struct mesh_graph_span *x=s->span+k;
    graph_put(&t,"%% span %zu first %llu count %llu begin %llu end %llu\n",k,(unsigned long long)x->first,
      (unsigned long long)x->count,(unsigned long long)x->begin,(unsigned long long)x->end);
  }
  signature_scoped(&t, s);
  graph_put(&t, "%% totals operations %zu synchronized %zu image %zu stronger %zu false %zu unmatched %zu satisfied %zu "
                "transformed %zu redundant %zu uncovered %zu unwaited %zu\n", s->operations, s->synchronized,
            s->states[MESH_GRAPH_IMAGE], s->states[MESH_GRAPH_STRONGER], s->states[MESH_GRAPH_FALSE],
            s->states[MESH_GRAPH_UNMATCHED], s->states[MESH_GRAPH_SATISFIED], s->states[MESH_GRAPH_TRANSFORMED],
            s->redundant, s->uncovered, s->unwaited);
  graph_put(&t, "%zu %zu %zu\n", s->operations, s->operations, entries);
  for (size_t o = 0; o < s->operations; o++)
    if (signature_ordered(s, o)) graph_put(&t, "%zu %zu 0\n", s->previous[o] + 1, o + 1);
  for (size_t k = 0; k < s->synchronized; k++)
    if (signature_drawn(s->edge + k))
      graph_put(&t, "%u %u %llu\n", s->edge[k].signal + 1, s->edge[k].wait + 1, (unsigned long long)s->edge[k].delay);
  return graph_end(&t, made);
}

int mesh_graph_dot(const struct mesh_graph_signature *s, char *text, size_t room, size_t *made) {
  struct graph_sink t = {text, room, 0, 0};
  graph_put(&t, "digraph signature {\n  node [shape=box];\n");
  for (uint32_t p = 0; p < s->processors; p++) {
    graph_put(&t, "  subgraph cluster_%u {\n    label=\"%u %s %u\";\n", p, p, signature_kind[s->processor[p].kind],
              s->processor[p].index);
    for (size_t k = s->first[p]; k < s->first[p + 1]; k++) {
      const size_t o = s->order[k];
      const struct mesh_graph_operation *x = s->operation + o;
      graph_put(&t, "    o%zu [label=\"%zu %s\\ntask", o + 1, o + 1, signature_primitive[x->primitive]);
      signature_number(&t, x->vertex);
      graph_put(&t, " execution %llu\\nobject %u", (unsigned long long)x->execution, x->object);
      signature_value(&t, x);
      graph_put(&t, "\"];\n");
      if (s->previous[o] != SIZE_MAX) graph_put(&t, "    o%zu -> o%zu [style=dotted];\n", s->previous[o] + 1, o + 1);
    }
    graph_put(&t, "  }\n");
  }
  for (size_t k = 0; k < s->synchronized; k++) {
    const struct mesh_graph_synchronized *e = s->edge + k;
    if (e->signal == MESH_GRAPH_NONE) continue;
    graph_put(&t, "  o%u -> o%u [label=\"%llu %s%s\"%s];\n", e->signal + 1, e->wait + 1, (unsigned long long)e->delay,
              signature_state[e->state], e->redundant ? " redundant" : "", e->redundant ? " style=dashed" : "");
  }
  /* a publication event: no processor's, after all of its occurrence's completions (its sites' first and later
     tasks, alternatives, no edge of the task graph), answering the reads drawn to it */
  for (size_t k = 0; k < s->occurrences; k++) {
    const struct mesh_graph_occurrence *x = s->occurrence + k;
    const struct mesh_graph_scopes_data *d = s->scopes->row + x->row;
    const size_t f = d->release_first[x->completion], n = d->release_first[x->completion + 1] - f;
    for (size_t r = 0; r < n; r++) {
      graph_put(&t, "  e%llu [shape=diamond label=\"publication %llu\\nrow %u scope %u execution %llu value %llu\\n",
                (unsigned long long)(x->event + r), (unsigned long long)(x->event + r), x->row, x->completion,
                (unsigned long long)x->execution, (unsigned long long)s->published[x->event + r]);
      signature_source(&t, d);
      graph_put(&t, "\"];\n");
    }
  }
  for (size_t k = 0; k < s->answers; k++)
    graph_put(&t, "  e%u -> o%u [label=\"scoped\" style=bold];\n", s->answer[k].event, s->answer[k].operation + 1);
  graph_put(&t, "}\n");
  return graph_end(&t, made);
}

int mesh_graph_space_time(const struct mesh_graph_signature *s, char *text, size_t room, size_t *made) {
  struct graph_sink t = {text, room, 0, 0};
  const size_t n = s->operations, width = s->processors ? 2 * (size_t)s->processors - 1 : 0;
  uint32_t *from = malloc((2 * n + 1) * sizeof *from), *to = malloc((2 * n + 1) * sizeof *to);
  uint32_t *place = malloc((n + 1) * sizeof *place), *at = malloc((n + 1) * sizeof *at);
  double *priority = malloc((n + 1) * sizeof *priority);
  char *line = malloc(width + 1);
  size_t edges = 0;
  int result = from && to && place && at && priority && line && n < UINT32_MAX ? 0 : ENOMEM;
  for (size_t o = 0; !result && o < n; o++) {
    priority[o] = -(double)o;
    if (s->previous[o] != SIZE_MAX) { from[edges] = (uint32_t)s->previous[o]; to[edges++] = (uint32_t)o; }
  }
  for (size_t k = 0; !result && k < s->synchronized; k++)
    if (signature_drawn(s->edge + k) && !s->edge[k].delay) { from[edges] = s->edge[k].signal; to[edges++] = s->edge[k].wait; }
  if (!result && (result = mesh_graph_heap_order((uint32_t)n, priority, edges, from, to, place)) == EINVAL) result = EDEADLK;
  for (size_t o = 0; !result && o < n; o++) at[place[o]] = (uint32_t)o;
  for (uint32_t p = 0; !result && p < s->processors; p++)
    graph_put(&t, "%*s%u %s %u\n", 2 * (int)p, "", p, signature_kind[s->processor[p].kind], s->processor[p].index);
  for (size_t r = 0; !result && r < n; r++) {
    const size_t o = at[r];
    const struct mesh_graph_operation *x = s->operation + o;
    const size_t k = s->waited[o];
    if (k != SIZE_MAX && signature_drawn(s->edge + k)) {
      const struct mesh_graph_synchronized *e = s->edge + k;
      const size_t a = 2 * (size_t)s->operation[e->signal].processor, b = 2 * (size_t)x->processor;
      for (size_t c = 0; c < width; c++) line[c] = c % 2 ? ' ' : '|';
      for (size_t c = (a < b ? a : b) + 1; c < (a < b ? b : a); c++) line[c] = e->delay ? '~' : '-';
      line[a] = '+';
      line[b] = a < b ? '>' : a > b ? '<' : '*';
      line[width] = 0;
      graph_put(&t, "%s  %u -> %zu %s delay %llu%s\n", line, e->signal + 1, o + 1, signature_state[e->state],
                (unsigned long long)e->delay, e->redundant ? " redundant" : "");
    }
    for (size_t c = 0; c < width; c++) line[c] = c % 2 ? ' ' : '|';
    line[2 * (size_t)x->processor] = 'o';
    line[width] = 0;
    graph_put(&t, "%s  %zu %s task", line, o + 1, signature_primitive[x->primitive]);
    signature_number(&t, x->vertex);
    graph_put(&t, " execution %llu object %u", (unsigned long long)x->execution, x->object);
    signature_value(&t, x);
    if (k != SIZE_MAX && !signature_drawn(s->edge + k)) graph_put(&t, " %s", signature_state[s->edge[k].state]);
    graph_put(&t, "\n");
  }
  for (size_t k = 0; !result && k < s->occurrences; k++) {
    const struct mesh_graph_occurrence *x = s->occurrence + k;
    const struct mesh_graph_scopes_data *d = s->scopes->row + x->row;
    const struct mesh_graph_completion *c = d->completion + x->completion;
    graph_put(&t, "publication events %u+%zu of row %u", x->event,
              d->release_first[x->completion + 1] - d->release_first[x->completion], x->row);
    signature_source(&t, d);
    graph_put(&t, " scope %u execution %llu, once after all %llu completions of tasks", x->completion,
              (unsigned long long)x->execution, (unsigned long long)c->total);
    for (uint32_t w = 0; w < c->writers; w++) {
      graph_put(&t, " %u", s->site[x->site + w].first);
      if (s->site[x->site + w].later != MESH_GRAPH_NONE) graph_put(&t, "|%u", s->site[x->site + w].later);
    }
    graph_put(&t, "\n");
  }
  free(from); free(to); free(place); free(at); free(priority); free(line);
  return result ? result : graph_end(&t, made);
}

static int intersect_before(const void *a, const void *b) {
  const struct mesh_graph_interval *x = a, *y = b;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  return (x->start > y->start) - (x->start < y->start);
}
int mesh_graph_intersect(size_t count, struct mesh_graph_interval *interval) {
  qsort(interval, count, sizeof *interval, intersect_before);
  for (size_t k = 1; k < count; k++)
    if (interval[k].key == interval[k - 1].key && interval[k].start < interval[k - 1].end) return 1;
  return 0;
}

/* GaijinEntertainment/DagorEngine 2baf1947, prog/gameLibs/render/daFrameGraph/backend/resourceScheduling/heapLayout.cpp:13-23,
   do_arcs_overlap */
int mesh_graph_arcs_overlap(uint64_t a_start, uint64_t a_end, uint64_t b_start, uint64_t b_end) {
  const int a_wraps = a_end <= a_start;
  const int b_wraps = b_end <= b_start;
  if (a_wraps && b_wraps) return 1;
  if (a_wraps) return a_end > b_start || b_end > a_start;
  if (b_wraps) return b_end > a_start || a_end > b_start;
  return a_end > b_start && b_end > a_start;
}

/* TensorFlow Lite 6cf18d88, tensorflow/lite/delegates/gpu/common/memory_management/internal.h:35-43,
   TensorUsageWithIndex, and internal.cc:28-31, CompareBySize */
struct greedy_record { uint64_t tensor_size; size_t idx; };
static int greedy_compare_by_size(const void *a, const void *b) {
  const struct greedy_record *first = a, *second = b;
  if (first->tensor_size != second->tensor_size) return first->tensor_size > second->tensor_size ? -1 : 1;
  return (first->idx > second->idx) - (first->idx < second->idx);
}
/* TensorFlow Lite 6cf18d88, tensorflow/lite/delegates/gpu/common/util.h:26-44, DivideRoundUp and AlignByN */
static uint64_t greedy_align_by_n(uint64_t number, uint64_t n) {
  const uint64_t q = number / n;
  return (number % n == 0 ? q : q + 1) * n;
}
/* TensorFlow Lite 6cf18d88, tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc:68-143,
   GreedyBySizeAssignment, and internal.h:31, kNotAssigned; Y. Pisarchyk and J. Lee, arXiv:2001.03288, 2020, §5.2 */
/* the record `idx` assigned at `at`: ordered_allocs kept in order of offset, `total` raised to its end (GreedyBySizeAssignment's
   insertion, above) */
static void greedy_assign(size_t *ordered_allocs, size_t *allocs, uint64_t *offset, const uint64_t *size, size_t idx,
                          uint64_t at, uint64_t *total) {
  size_t it = 0;
  while (it < *allocs && offset[ordered_allocs[it]] <= at) it++;
  memmove(ordered_allocs + it + 1, ordered_allocs + it, (*allocs - it) * sizeof *ordered_allocs);
  ordered_allocs[it] = idx;
  ++*allocs;
  offset[idx] = at;
  if (at + size[idx] > *total) *total = at + size[idx];
}
static int greedy_pinned(const uint64_t *pinned, size_t r) { return pinned && pinned[r / 64] >> (r % 64) & 1; }
/* `pinned` (NULL: none), bit r % 64 of word r / 64: record r's offset[r] is given (an occupied range already
   reserved: design/aot-compilation-change-request.md, appendix item 6), assigned before the size-ordered pass, which
   places the other records around it */
int mesh_graph_greedy_by_size(size_t records, const uint64_t *size, const uint64_t *pinned,
                              int (*intersect)(const void *context, size_t a, size_t b),
                              const void *context, uint64_t align, uint64_t *offset, uint64_t *total) {
  struct greedy_record *ordered_records = malloc((records + 1) * sizeof *ordered_records);
  size_t *ordered_allocs = malloc((records + 1) * sizeof *ordered_allocs), allocs = 0;
  int result = ordered_records && ordered_allocs ? 0 : ENOMEM;
  *total = 0;
  for (size_t i = 0; !result && i < records; i++) ordered_records[i] = (struct greedy_record){size[i], i};
  if (!result) qsort(ordered_records, records, sizeof *ordered_records, greedy_compare_by_size);
  for (size_t i = 0; !result && i < records; i++)
    if (greedy_pinned(pinned, i)) greedy_assign(ordered_allocs, &allocs, offset, size, i, offset[i], total);
  for (size_t r = 0; !result && r < records; r++) {
    const size_t idx = ordered_records[r].idx;
    if (greedy_pinned(pinned, idx)) continue;
    uint64_t best_diff = UINT64_MAX, best_offset = UINT64_MAX, prev_offset = 0;
    for (size_t k = 0; k < allocs; k++) {
      const size_t allocated_id = ordered_allocs[k];
      if (!intersect(context, allocated_id, idx)) continue;
      const uint64_t cur_offset = offset[allocated_id];
      if (cur_offset >= prev_offset) {
        const uint64_t diff = cur_offset - prev_offset;
        if (diff >= size[idx] && diff < best_diff) {
          best_diff = diff;
          best_offset = prev_offset;
        }
      }
      const uint64_t aligned = greedy_align_by_n(cur_offset + size[allocated_id], align);
      if (aligned > prev_offset) prev_offset = aligned;
    }
    if (*total + allocs * (align - 1) < prev_offset) { result = EINVAL; break; }
    if (best_offset == UINT64_MAX) best_offset = prev_offset;
    greedy_assign(ordered_allocs, &allocs, offset, size, idx, best_offset, total);
  }
  free(ordered_records); free(ordered_allocs);
  return result;
}

void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context) {
  for (uint32_t j = 0; j < count; j++) visit(context, row[j]);
}

/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §5.2; B. R. Rau, MICRO 1994;
   design/torch-source-lowering-change-request.md item 3 */
size_t mesh_graph_delayed_rows(const size_t *first, const size_t *order, uint32_t (*tail)(const void *context, size_t e),
                               const void *context, const unsigned char *redundant, uint32_t row, uint32_t count,
                               uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot, size_t *edge) {
  size_t made = 0;
  for (size_t q = first[row]; q < first[row + 1]; q++) {
    const size_t e = order[q];
    if (redundant && redundant[e]) continue;
    const uint32_t from = tail(context, e), delay = lag - 1 - from / count;
    if (t < delay) continue;
    source[made] = from % count; slot[made] = (uint32_t)((t - delay) % lag);
    if (edge) edge[made] = e;
    made++;
  }
  return made;
}
size_t mesh_graph_delayed(const size_t *first, const size_t *order, const uint32_t *tail, uint32_t row, uint32_t count,
                          uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot, size_t *edge) {
  return mesh_graph_delayed_rows(first, order, mesh_graph_row_head, tail, NULL, row, count, lag, t, source, slot, edge);
}
uint32_t mesh_graph_relation_tail(const void *context, size_t e) {
  return ((const struct mesh_graph_relation_data *)context)->edge[e].from;
}
size_t mesh_graph_relation_delayed(const struct mesh_graph_relation_data *relation, const unsigned char *redundant,
                                   uint32_t offset, uint32_t row, uint32_t count, uint32_t lag, uint64_t t,
                                   uint32_t *source, uint32_t *slot, size_t *edge) {
  return mesh_graph_delayed_rows(relation->in_first + offset, relation->in_edge, mesh_graph_relation_tail, relation,
                                 redundant, row, count, lag, t, source, slot, edge);
}

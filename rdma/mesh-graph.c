#include "mesh-graph.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* HiGHS 1.15.1's C API (design/plan-to-bridge-change-request.md R5, Code: "The C API is highs/interfaces/highs_c_api.h
   of that source"), reached from the source's highs directory (where the header's own includes, "lp_data/..." and
   "util/...", resolve) and its root (where its HConfig.h is), as system includes (-isystem), since it declares a
   static function it never defines, Highs_compilationDate (rdma/Makefile HIGHS_CFLAGS; "The revisions of 46ae8fdf",
   LP8).  2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", HB): the pinned source the
   Makefile fetches, checks and builds, where it was Homebrew's install (R5: "Nothing is installed by hand, and nothing
   comes from Homebrew").  What must stay: the header included as <interfaces/highs_c_api.h> */
#include <interfaces/highs_c_api.h>

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_reach, 6379e868; `parent` added
   (design/compilation-coordination.md, "The graph module", Q3): a vertex's parent is the vertex at the queue's head
   when it is first reached, the breadth-first tree of that same walk. */
/* mesh-graph.h */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent) {
  for (uint32_t v = 0; v < vertices; v++) { reached[v] = 0; if (parent) parent[v] = UINT32_MAX; }
  uint32_t count = 1;
  order[0] = from; reached[from] = 1;
  if (parent) parent[from] = from;
  for (uint32_t head = 0; head < count; head++)
    for (uint32_t e = 0; e < edges; e++)
      for (uint32_t side = 0; side < 2; side++) {
        const uint32_t other = end(context, e, 1 - side);
        if (end(context, e, side) == order[head] && !reached[other]) {
          reached[other] = 1; order[count++] = other;
          if (parent) parent[other] = order[head];
        }
      }
  return count;
}

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_climb, 7e7f8726 */
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

static uint32_t word_at(const void *context, size_t k) { return ((const uint32_t *)context)[k]; }

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_rows, d8d861b5 */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_keyed_find and mesh_graph_keyed_add, d8d861b5 */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_exp and mesh_graph_log, 4163911c */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_flow with flow_same, 4163911c */
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
  const int failed=!load || !add || !len || !dist || !up || !upl || !tree || !treel || !hits || !done || !head || !next;
  for(uint32_t d=0;!failed && d<count;d++)head[d]=UINT32_MAX;
  double least=0,total=0;
  for(uint32_t e=0;!failed && e<E;e++)if(!least || cap[e]<least)least=cap[e];
  for(uint32_t d=0;!failed && d<count;d++)total+=upward[d]+downward[d];
  const double width=total>0 && least>0?total/rounds/least:1;
  for(uint32_t r=0;!failed && r<rounds;r++){
    double most=0;
    for(size_t e=0;e<2*(size_t)E;e++)if(load[e]/cap[e/2]>most)most=load[e]/cap[e/2];
    for(size_t e=0;e<2*(size_t)E;e++){len[e]=mesh_graph_exp(eps*(load[e]/cap[e/2]-most)/width)/cap[e/2];add[e]=0;}
    for(uint32_t d=0;d<count;d++){
      const double u=upward[d],w=downward[d];
      for(uint32_t v=0;v<n;v++){dist[v]=INFINITY;up[v]=UINT32_MAX;upl[v]=UINT32_MAX;done[v]=0;tree[v]=UINT32_MAX;treel[v]=UINT32_MAX;}
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
      for(uint32_t v=0;v<n;v++){
        if(v==root[d] || !leaf(context,d,v))continue;
        for(uint32_t at=v;tree[at]==UINT32_MAX;at=up[at]){tree[at]=up[at];treel[at]=upl[at];}
      }
      const struct flow_same same={parent,tree,n};
      uint32_t k=mesh_graph_keyed_find(head,next,d,flow_same,&same);
      if(k==UINT32_MAX){
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_grid_cut, ea247333 (the form ea247333 left) */
/* mesh-graph.h mesh_graph_grid_cut */
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index){
  const uint64_t g=cell?cell:1,end=first+length;
  uint32_t made=0;
  for(uint64_t at=first;at<end;){
    const uint64_t k=at/g,edge=(k+1)*g<end?(k+1)*g:end;
    start[made]=at;count[made]=edge-at;index[made++]=k;at=edge;
  }
  return made;
}
/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_fill, d30f02f5 */
/* mesh-graph.h mesh_graph_fill: the least time T at which the parts' units by T, floor((T - latency) rate) each,
   reach `units` (a bisection on T), then the units past `units` taken back from the parts whose last finishes
   latest */
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
  /* rounding at the bisection's end: the units short given where each would finish soonest */
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

/* New beside mesh_graph_fill (design/page-queue-change-request.md, "The rule both ends compute"): no tag source.
   mesh-graph.h mesh_graph_list_schedule: each unit in list order to the machine it would end soonest on, the machines
   scanned in order so a tie keeps the lower one */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried){
  for(uint32_t l=0;l<machines;l++)carried[l]=0;
  int lost=0;
  for(uint64_t k=0;k<units;k++){
    uint32_t best=UINT32_MAX;double soon=INFINITY;
    for(uint32_t l=0;l<machines;l++){
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

/* New (design/disjoint-pages-change-request.md, Appendix C, C3.3; design/compilation-coordination.md, U1;
   design/disjoint-pages-revision-coordination.md, "The allocator"): no tag source.
   2026-11-04, owner C2 (U1): cited to C3.3 and U1, where it cited design/disjoint-pages-revision-coordination.md, D5.
   Why: D5 appears in neither Lam, Rau nor C3.3's text (design/aot-compilation-change-request.md, appendix item 1); the
   least lags decide C3.3's tests at every larger distance by U1's reason, its consumer's (every walk into a write
   passes its range's production; the programs' dependences order each node's events of t before those of t + 1).
   What must stay: the algorithm as it is.
   The binary heap, one copy (design/rubrics.md:422, C9: the graph algorithms one copy each), mesh_graph_lags' and
   mesh_graph_heap_order's: heap[0 .. count) vertices, heap[0] the first; position[v] v's index in it, UINT32_MAX where
   v is not in it.  before(context, a, b) != 0 where a comes before b, the consumer's strict total order of the vertices
   (mesh_graph_lags' lags_before, mesh_graph_heap_order's order_before), so the vertex the heap gives first among those
   in it is the same whatever order they entered it in.
   2026-11-04, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", "The graph
   module"): lags_up and lags_down are heap_up and heap_down, taking their comparison and its context as operands, where
   they called lags_before over mesh_graph_lags' distances.  Why: mesh_graph_heap_order's heap is this one, one copy
   (C9).  What must stay: mesh_graph_lags' behavior, its heap ordered by lags_before as it was. */
/* heap[i]'s vertex moved toward the root past each parent it comes before */
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
/* heap[i]'s vertex moved away from the root past each child, the first of the two, that comes before it */
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
/* mesh_graph_lags' heap: the vertices whose distance is set but not yet known least, by (distance, vertex) over the
   distances (`context`).  a comes before b where its distance is less, a tie to the lower vertex. */
static int lags_before(const void *context, uint32_t a, uint32_t b) {
  const uint32_t *distance = context;
  return distance[a] < distance[b] || (distance[a] == distance[b] && a < b);
}
/* a walk to v of total lag `through`: v's distance lowered to it where it is less, v put into the heap or moved up in
   it; the heap's count.  distance[v] <= UINT32_MAX, so a sum at or past UINT32_MAX lowers nothing: no walk. */
static size_t lags_relax(uint32_t *heap, uint32_t *place, uint32_t *distance, size_t count, uint32_t v,
                         uint64_t through) {
  if (through >= distance[v]) return count;
  distance[v] = (uint32_t)through;
  if (place[v] == UINT32_MAX) {
    heap[count] = v;
    heap_up(heap, place, lags_before, distance, count);
    return count + 1;
  }
  heap_up(heap, place, lags_before, distance, place[v]);
  return count;
}
/* mesh-graph.h mesh_graph_lags.  From each source, Dijkstra's shortest paths (his second problem, the lags the
   lengths), every vertex's distance first unreached (UINT32_MAX) and the source's own edges relaxed first, from 0,
   with the source itself not reached: the walks are those from a vertex whose edges are the source's, so each has at
   least one edge, and the source reached again takes its least cycle.  A vertex leaves the heap least (distance,
   vertex) first, its distance then the least (no lag is negative), and never returns to it, nothing after it being
   less; the walk from a source ends once every distinct target has left the heap (Dijkstra's ends once his target is
   transferred to set A) or once the heap is empty, a target never reached keeping UINT32_MAX.  The least lags are
   unique, so the order fixes only the steps, and its ties to the lower vertex make those the same at every node too.
   The file's rules: integer + and < alone (no rounding at all, beneath "IEEE-exact operations alone"), so every node
   computes the same lags from the same operands; the lags are the consumer's operands (mesh-collective.c's template);
   one copy of the algorithm over integer lags (mesh_graph_flow's least-length tree, over undirected edges, double
   lengths and with its parents, is its own).  The edges' heads and lags are read once.  Every allocation precedes the
   first lag written, so ENOMEM writes none.
   Cost: O(sources x (E + V) log V): per source, V distances and places set, each edge relaxed at most once at its
   tail's departure (and the source's own once more, first), each relaxation and each departure one heap move of
   O(log V); memory O(V + E). */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint32_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint32_t *lags) {
  uint32_t *distance = malloc(((size_t)vertices + 1) * sizeof *distance);
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap), *place = malloc(((size_t)vertices + 1) * sizeof *place);
  unsigned char *wanted = calloc((size_t)vertices + 1, 1);
  const int result = !distance || !heap || !place || !wanted ? ENOMEM : 0;
  uint32_t distinct = 0;
  for (uint32_t x = 0; !result && x < targets; x++)
    if (!wanted[target[x]]) { wanted[target[x]] = 1; distinct++; }
  for (uint32_t s = 0; !result && s < sources; s++) {
    for (uint32_t v = 0; v < vertices; v++) { distance[v] = UINT32_MAX; place[v] = UINT32_MAX; }
    const uint32_t from = source[s];
    size_t count = 0;
    for (size_t k = first[from]; k < first[from + 1]; k++)
      count = lags_relax(heap, place, distance, count, head(context, k), lag(context, k));
    for (uint32_t left = distinct; count;) {
      const uint32_t at = heap[0];
      place[at] = UINT32_MAX;
      if (--count) { heap[0] = heap[count]; heap_down(heap, place, lags_before, distance, count, 0); }
      if (wanted[at] && !--left) break;
      for (size_t k = first[at]; k < first[at + 1]; k++)
        count = lags_relax(heap, place, distance, count, head(context, k), (uint64_t)distance[at] + lag(context, k));
    }
    for (uint32_t x = 0; x < targets; x++) lags[(size_t)s * targets + x] = distance[target[x]];
  }
  free(distance); free(heap); free(place); free(wanted);
  return result;
}

/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6: "Place every
   value's pages anywhere, by the fewest chains of the interference order" and "Greedy by size goes, with
   `mesh_graph_left_edge`'s colouring"; "The revisions of 4e78f6f4"): mesh_graph_left_edge and its placed runs (struct
   left_edge_run) are removed, the left-edge colouring of items that each took a run of colours at the lowest colour
   free of interference, the offset calculation of greedy by size (T. Hashimoto and J. Stevens, DAC 1971; Y. Pisarchyk
   and J. Lee, arXiv 2001.03288, 2020).  Why: design/plan-to-bridge-change-request.md R2, corrected 2026-11-05: "No
   value is placed as one run.  So greedy by size (Pisarchyk and Lee), and the contiguous-block problem it approximates
   (M. R. Garey and D. S. Johnson, Computers and Intractability, 1979, problem SR2), have no use here.  The chains
   replace C3.3's colouring."  What must stay: placement an operand of the caller's, by mesh_graph_chains below. */
/* New (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6; "The revisions of
   4e78f6f4"; design/plan-to-bridge-change-request.md R2, corrected 2026-11-05; design/disjoint-pages-change-request.md,
   A3 and C3.3, corrected 2026-11-05): no tag source.
   mesh_graph_chains' network: item i's left vertex L_i = i and right vertex R_i = items + i, the source 2 items and the
   sink 2 items + 1.  The middle arcs, one for each ordered pair i before j: edge e from L_tail[e] to R_head[e], flow[e]
   its flow f(i, j), the edges out of L_i first[i] .. first[i + 1] - 1 in the caller's order of j, and those into R_j
   into[into_first[j] .. into_first[j + 1]) in the caller's order of i; out[i] the flow from the source into L_i, in[j]
   the flow from R_j into the sink. */
struct chains_net {
  uint32_t items;
  const uint64_t *count;
  const size_t *first, *into_first, *into;
  const uint32_t *head, *tail;
  uint64_t *flow, *out, *in;
};
/* vertex u's arcs in the residual network, in their order: the source's to L_0 .. L_items-1; L_i's to each R_j it
   precedes, in the caller's order; R_j's to the sink first, then back to each L_i that precedes it, in the caller's
   order; the sink's none */
static size_t chains_arcs(const struct chains_net *n, size_t u) {
  const size_t items = n->items;
  if (u < items) return n->first[u + 1] - n->first[u];
  if (u < 2 * items) return 1 + n->into_first[u - items + 1] - n->into_first[u - items];
  return u == 2 * items ? items : 0;
}
/* vertex u's arc a: its head (*to) and residual capacity.  Source -> L_i: count[i] less out[i]; L_i -> R_j: unbounded
   (UINT64_MAX); R_j -> sink: count[j] less in[j]; R_j -> L_i, back along the middle arc: f(i, j) */
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
/* `delta` more along vertex u's arc a: its flow raised, or, back along a middle arc, that arc's lowered */
static void chains_push(struct chains_net *n, size_t u, size_t a, uint64_t delta) {
  const size_t items = n->items;
  if (u < items) n->flow[n->first[u] + a] += delta;
  else if (u == 2 * items) n->out[a] += delta;
  else if (!a) n->in[u - items] += delta;
  else n->flow[n->into[n->into_first[u - items] + a - 1]] -= delta;
}
/* Dinic's layered network: each vertex's level its distance from the source over the arcs of positive residual
   capacity, by a breadth-first search (queue[0 .. vertices)), ended once the sink is reached; unreached SIZE_MAX.  1
   where the sink is reached, else 0 (no augmenting path: the flow is maximum). */
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
/* Dinic's blocking flow in the layered network: depth-first searches from the source (path[0 .. depth), at most one
   vertex a level), each along an arc of positive residual capacity into the next level, the sink's level the last (a
   vertex there other than the sink leads to no augmenting path of this phase).  Each vertex's arcs are taken in their
   order from a pointer, at[u], that passes an arc once it is found saturated or leading to no path: Dinic's removal of
   blocked arcs from the layered network, each arc passed at most once a phase.  At the sink, the path's least residual
   capacity is pushed along it and the search goes back to the tail of the first arc that saturated; a vertex whose arcs
   are all passed is left, and its parent's pointer passes the arc into it.  The phase ends when the source's arcs are
   all passed, so every path to the sink in the layered network holds a saturated arc. */
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
/* mesh-graph.h mesh_graph_chains.  R2, corrected 2026-11-05: "By Dilworth's theorem that number equals the largest set
   of items no two of which are ordered (R. P. Dilworth, A decomposition theorem for partially ordered sets, Annals of
   Mathematics 51(1), 1950).  It is found by maximum bipartite matching (D. R. Fulkerson, Note on Dilworth's
   decomposition theorem for partially ordered sets, Proc. AMS 7(4), 1956), exactly and in polynomial time."  The Python
   half's form of the same method is rdma/mesh_graph.py `width`, by `matching` (Hopcroft and Karp's phases); this one
   gives the chains too, and takes multiplicities.
   - The network (mesh-graph.h: Fulkerson's bipartite graph, "with an item's elements as one vertex of capacity count[i]
     on each side"): source -> L_i of capacity count[i], R_j -> sink of capacity count[j], and L_i -> R_j wherever
     before(context, i, j) != 0, unbounded (it stands for the count[i] x count[j] element pairs, more than L_i takes
     in).  An integral flow of value F is then a matching of F element pairs, f(i, j) of them an element of item i's
     followed by one of item j's, and conversely.  `before` is called once for each ordered pair of distinct items, i
     then j in the caller's order, items x (items - 1) calls; never with a == b, the order being irreflexive.
   - Its maximum flow by Dinic's phases (E. A. Dinic, Algorithm for solution of a problem of maximum flow in a network
     with power estimation, Soviet Math. Doklady 11, 1970), which on a network of capacities are Hopcroft and Karp's (J.
     E. Hopcroft and R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite graphs, SIAM J. Comput. 2(4),
     1973; mesh-graph.h; mesh_graph.py `matching`: "a breadth-first search layers the graph ... to the length of the
     shortest augmenting paths, then depth-first searches augment along vertex-disjoint paths of that length, until
     none is left"): each phase the layered network (chains_layer), then a blocking flow in it (chains_block), until
     the sink is not reached.  The residual network's arcs are chains_arcs' and chains_arc's; the arcs back into the
     source and out of the sink lie on no path from the source to the sink and are left out.  Each phase lengthens the
     shortest augmenting path (Dinic), an odd length from 3 to 2 items + 1, so there are at most `items` phases; where
     every count is 1 they are Hopcroft and Karp's own, O(sqrt(items)) of them.
   - The chains.  Item j's first in[j] elements (in[j] = the sum of f(i, j) over i) continue chains and its other
     count[j] - in[j] begin them; item i's first out[i] elements (out[i] = the sum of f(i, j) over j) are continued.
     The elements that begin chains are numbered 0, 1, ... in the elements' order, item i's element k at o_i + k, o_i
     the sum of count[h] over h < i (mesh-graph.h).  Then the items are taken in a linear extension of `before`, and
     for each item i its edges L_i -> R_j in the caller's order of j: the next f(i, j) of item i's elements, from its
     first, give their chains to the next f(i, j) of item j's, from its first, in turn.  An item's predecessors all
     come before it in the extension, so each element's chain is set before it is given.  The extension: the items by
     their number of predecessors, ascending, a tie in the caller's order (mesh_graph_rows, a stable counting sort, with
     word_at over the counts, its rows in queue's words); a before b makes every predecessor of a one of b
     (transitivity), and a one of b but not of a (irreflexivity), so b has more.
     2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G1:
     "`mesh_graph_chains` orders its items by its own predecessor count (mesh-graph.c 875–894) where
     `mesh_graph_order` computes that order ... Both are to call the module's own functions"; "The revisions of
     55b9cbad"): the extension is mesh_graph_order's topological order of the items over the edges `before` gave
     (their rows by tail, `first` and `head`), each place the lowest-numbered item whose predecessors are all placed,
     where it was the stable sort by predecessor count; any topological order is a linear extension, so each element's
     chain is still set before it is given.  EDEADLK where those edges hold a cycle, which a strict partial order
     cannot (mesh_graph_order's own error, before any output is written).  What must stay: the chains numbered as
     above, every party computing the same from the same operands.
     - Each element then has at most one element before it in its chain and one after it, of an item it follows or
       precedes, so each chain is a path x_1 -> x_2 -> ... of mutually ordered elements (transitivity), and they number
       the elements less F, the fewest (Fulkerson).
     - An item that precedes and follows none has no flow in or out: its count[i] elements begin chains, numbered
       consecutively.
   (2026-11-05, "The revisions of 55b9cbad", B4: "Its time on the largest program is a measurement for when runs are
   authorized"; the network is Fulkerson's reduction, its phases Hopcroft and Karp's where every count is 1, as
   above.)
   Arithmetic: integer + - and comparisons alone, every loop in the caller's order or the extension's, no hashing:
   every party with the same operands finds the same flow and the same chains.  Bound: each count[i] < 2^32 (the caller's block counts) and
   items < 2^32, so the elements' sum, every flow and every chain number lie below 2^64, under the unbounded capacity
   UINT64_MAX.  The edges' arrays double from 64 entries while `before` is read; a doubling's size cannot pass SIZE_MAX,
   since half of it was already allocated.  Every allocation precedes the first output written, so ENOMEM writes none.
   Cost: items x (items - 1) calls of `before`; memory O(items + E) for E ordered pairs, 24 bytes each (head and tail's
   8 of them up to twice that while they grow); each phase O(items + E) for the layering and O(items x E) for the
   blocking flow (Dinic). */
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
  if (!result) result = mesh_graph_order(items, first, word_at, head, indegree, ordinal);
  if (!result) {
    mesh_graph_rows(items, edges, word_at, head, into_first, into);
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

/* design/problems-of-14b9a757-change-request.md, "Appendix: the stops of 36bd7cbd", 14.  Mukerji 1994, §3.1:
   "Front-end of cyclic intervals : While sweeping the graph from left to right, the first section of the interval that
   is encountered is called the front-end of the cyclic interval"; "Tail-end of cyclic intervals : The second section of
   the cyclic interval that is encountered while sweeping the graph is the tail-end of the interval" */
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
/* Definition 2.1.2: "Two intervals I1, I2 overlap if there exists a time t that is covered by both I1 and I2" */
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
/* mesh-graph.h mesh_graph_fat_cover.  design/problems-of-14b9a757-change-request.md, "Appendix: the stops of
   36bd7cbd", 14: "G is colored by Main Fatcover, Establish All Covers Of C_i and Choose A Cover For C_i as printed, with
   the left edge algorithm (A.0.1) for the non-cyclic remainder.  A cyclic interval with no fat cover gets new colors,
   as §3.2.3's second phase states."  Mukerji 1994:
   - Definitions 2.3.1, 2.3.2 and 3.2.1: "the number of intervals covering t"; "the maximum width(G,t), for all t which
     is covered by some interval in G"; "the set of all times ti where width(G, ti) = Wmax(G)".
   - §3.2.6, p. 38: "The colored intervals are removed from the graph and the resulting graph contains no cyclic
     intervals"; p. 40: "A left to right sweep of G' is performed and each fatspot is visited"; "Once all the
     children[] intervals have been found at a fatspot they're moved into parent[]"; "C_i is always a part of its
     fatcover, so the front and tail ends of C_i are always inserted as intervals of it's fatcover"; p. 42: "Our
     heuristics favor : Intervals which have earlier start times.  When two intervals have the same start time then the
     interval which is live longer is chosen"; Establish All Covers Of C_i, "Set parent of C_i"; Choose A Cover For C_i,
     "IF int was before and s in chosen_fc[] THEN int_num = int".
   - Main Fatcover: "Assign FC_{C_i}[chosen] (reg_{number}(C_i)) new colors to FC_{C_i}[chosen]"; "Assign C_i
     (reg_{number}(i)) new colors, c_i".
   - Theorem 2.3.1's proof: "For each interval, I, which ends at t, release its color back to the pool of free colors.
     For each interval I beginning at t, give I a free color which is not being used by any interval covering t";
     §3.2.3: "The coloring of G' is guaranteed to use only w new colors"; "the second phase simply colors the remaining
     cyclic intervals with new colors, and applies the simple left to right algorithm to color the remaining
     intervals"; Algorithm A.0.1: "Obtain colors c_i ...
     c_{regnum_i} that are free for the lifetime of i". */
int mesh_graph_fat_cover(uint64_t length, uint32_t intervals, const uint64_t *start, const uint64_t *end,
                         const uint64_t *need, uint64_t *color, uint64_t *colors) {
  const size_t n = intervals, keys = 4 * n;
  uint64_t elements = 0;
  for (uint32_t i = 0; i < intervals; i++) elements += need[i];
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
      if (r > most_row) most_row = r;
      link_room += r * above;
      node_room += r;
      total += r;
    }
    for (size_t f = 0; !result && f < fatspots; f++) row[f + 1] += row[f];
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

/* New (design/plan-to-bridge-change-request.md R5 at mesh 0e56d8da, "How: the program" and "How: solving it";
   design/compilation-coordination.md, "The revisions of 46ae8fdf", LP): no tag source.
   2026-11-05 (LP): levels, mesh_graph_bottom_levels (Sinnen and Sousa's eq. (7)) and mesh_graph_contention_schedule
   (their Algorithm 1 with Algorithm 2), with their helpers (levels_rows, levels_head, schedule_task, schedule_key,
   schedule_leaders), are removed.  Why: R5, "It replaces MSCCLang's depth and reverse depth, and edge scheduling's
   start times, which are no longer computed for the order.  Edge scheduling's part, one communication at a time on
   each link, is now the resources' rows"; their one caller, mesh-collective.c plan_schedule, takes the order from
   mesh_graph_completion_times.  What must stay: nothing of them.
   mesh_graph_completion_times' sort keys: a resource's vertices by their value in the current solution, the least
   first, a tie to the lower vertex (a total order, so every party's sort ends alike: "The revisions of 46ae8fdf",
   LP5) */
struct completion_key { double at; uint32_t vertex; };
static int completion_key_before(const void *a, const void *b) {
  const struct completion_key *x = a, *y = b;
  if (x->at != y->at) return x->at < y->at ? -1 : 1;
  return (x->vertex > y->vertex) - (x->vertex < y->vertex);
}
/* mesh-graph.h mesh_graph_completion_times.  R5 (mesh 0e56d8da), "How: the program" and "How: solving it", built as
   rows and handed to HiGHS through its C API (highs_c_api.h, HiGHS 1.15.1); no solver is written here (R5: "The code
   builds the rows and calls the library.  It writes no solver of its own").
   - Columns: C_j for each vertex j, its lower bound d_j (C.-Y. Chen, Scheduling coflows with precedence constraints
     for minimizing the total weighted completion time in identical parallel networks, arXiv 2205.02474, 2022, (1b):
     "Cijk >= rk + dijk", every release time zero within one execution), and C_max, of cost 1, every other cost 0:
     Chen's (14), "min Cmax".
   - Rows: C_max - C_j >= 0 for every j, Chen's (14a), "Cmax >= Ck"; C_j - C_i >= d_j for every edge i -> j, Chen's
     (1c), "the precedence constraint", and L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Scheduling to minimize
     average completion time, Math. Oper. Res. 22(3), 1997, (3), each edge once (LP4).
   - The resources' rows, R5: "For each resource, and each set S of the instructions on it: sum over j in S of d_j C_j
     >= 1/2 (d(S)^2 + d^2(S))", Hall et al.'s (5), "introduced by Queyranne (1993)", Chen's (1d) and (1e) with m = 1.
     They are exponentially many, so they enter by cutting planes (R5: "Solve it with the rows found so far.  Find each
     resource's violated rows by Queyranne's separation, and add them.  Repeat until no row is violated").  The
     separation (M. Queyranne, Structure of a simple scheduling polyhedron, Math. Programming 58, 1993, as R5 cites it,
     "O(n log n) time"): a resource's vertices sorted by the solution's C_j (completion_key_before), and the nested sets
     of that order, its prefixes, each tested in one pass with its d(S), d^2(S) and sum of d_j C_j kept as they grow; a
     vertex of length 0 adds 0 to every side and is left out.  Each round adds, for each resource that has one, the
     prefix whose right side most exceeds its left (LP2), where it exceeds it by more than HiGHS's own primal
     feasibility tolerance times 1 + the right side (LP3), and ends where no resource has one.
   - Each linear program by HiGHS's dual revised simplex (Q. Huangfu and J. A. J. Hall, Parallelizing the dual revised
     simplex method, Math. Program. Comput. 10(1), 2018): the options "solver" "simplex" and "simplex_strategy" 1 ("1 =>
     Dual (serial)", HighsOptions.h), on one thread ("parallel" "off", "threads" 1; R5: "Every rank solves the same
     program with the same library, on one thread"), "output_flag" false; every other option the library's own (LP6).
     A round's rows are added to the model the last round solved, whose basis HiGHS keeps.
   - Termination: a row is added only where the solution falls short of it by more than the tolerance times 1 + its
     right side; HiGHS reports a solution optimal only within that tolerance of every row it holds, so no row is added
     twice, and a resource has finitely many sets (LP3).
   - The second objective (R5 at mesh f3f7cf46: "Then, over (14)'s optimal solutions, minimize Σ_j C_j: Chen's (1) and
     Hall et al.'s (1) with unit weights.  The two are one lexicographic linear program (H. Isermann, Linear
     lexicographic optimization, OR Spektrum 4, 1982).  They are solved in sequence, the second with C_max held at the
     first's optimum"): where the first's rounds end, no row violated (`second` 0), C_max's bounds are both its value
     in that optimal solution, its cost 0 and every C_j's cost 1, and the rounds go on, the rows kept, the same
     separation, until no row is violated again.
     2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", LX): new.  Why: R5, "(Corrected
     2026-11-05: the text before minimized C_max alone.)"; "The second objective fixes C̄_j off the critical path,
     which (14) alone leaves to whichever optimal vertex the solver returns".  What must stay: the first objective
     solved whole, by its own rounds, before the second begins.
   completion[j] the last solution's C_j, the second objective's.  Every allocation precedes the first output written,
   and an error writes none.  Cost: the rows' building O(V + E log E); each round O(V log V) for the separation and
   HiGHS's own solve. */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion) {
  const uint64_t room = (uint64_t)vertices + edges + resources + 1;
  const HighsInt columns = (HighsInt)vertices + 1;
  size_t *first = malloc(((size_t)resources + 1) * sizeof *first), *member = malloc(((size_t)vertices + 1) * sizeof *member);
  struct completion_key *key = malloc(((size_t)vertices + 1) * sizeof *key);
  double *lower = malloc(room * sizeof *lower), *upper = malloc(room * sizeof *upper);
  double *value = malloc(2 * room * sizeof *value), *column = malloc((size_t)columns * sizeof *column);
  double *dual = malloc((size_t)columns * sizeof *dual), *row_value = NULL, *row_dual = NULL;
  HighsInt *start = malloc(room * sizeof *start), *index = malloc(2 * room * sizeof *index);
  void *highs = first && member && key && lower && upper && value && column && dual && start && index
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
    for (size_t e = 0; e < edges; e++) {
      const uint32_t i = from[e], j = to[e];
      start[total] = nonzeros; lower[total] = length[j]; upper[total] = infinity; total++;
      index[nonzeros] = (HighsInt)j; value[nonzeros++] = 1;
      index[nonzeros] = (HighsInt)i; value[nonzeros++] = -1;
    }
    if (Highs_addRows(highs, total, lower, upper, nonzeros, start, index, value) == kHighsStatusError) result = EDOM;
    mesh_graph_rows(resources, vertices, word_at, resource, first, member);
  }
  int second = 0;
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
      double sum = 0, squares = 0, left = 0, most = 0, bound = 0;
      size_t prefix = 0;
      for (size_t k = 0; k < count; k++) {
        const double d = length[key[k].vertex];
        sum += d; squares += d * d; left += d * key[k].at;
        const double right = (sum * sum + squares) / 2, short_by = right - left;
        if (short_by > tolerance * (1 + right) && short_by > most) { most = short_by; bound = right; prefix = k + 1; }
      }
      if (!prefix) continue;
      start[cuts] = nonzeros; lower[cuts] = bound; upper[cuts] = infinity; cuts++;
      for (size_t k = 0; k < prefix; k++) { index[nonzeros] = (HighsInt)key[k].vertex; value[nonzeros++] = length[key[k].vertex]; }
    }
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
  free(first); free(member); free(key); free(lower); free(upper); free(value); free(column); free(dual);
  free(row_value); free(row_dual); free(start); free(index);
  return result;
}

/* New (design/plan-to-bridge-change-request.md, R5, "The order of each cable's list, by the reference algorithm";
   design/compilation-coordination.md, "The documents' revisions of edd4b583", "The graph module" and GO3): no tag
   source.
   mesh_graph_heap_order's comparison over the priorities (`context`): a comes before b where its priority is greater,
   a tie to the lower vertex (GO3: "the greatest bottom level first ... a tie to the lower instruction"; the priority
   now -C̄, R5's "least C̄_j first", LP7); comparisons alone, a strict total order of the vertices over priorities that
   compare (none NaN). */
static int order_before(const void *context, uint32_t a, uint32_t b) {
  const double *priority = context;
  return priority[a] > priority[b] || (priority[a] == priority[b] && a < b);
}
/* mesh-graph.h mesh_graph_heap_order.  M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS
   2023, §5.2, step 3: "Sort instructions into a global topological order with respect to their dependencies with a
   heap using the priority to order instructions."
   - Each vertex's count of the edges into it (an edge given twice counted twice), and the edges out of each vertex as
     compressed rows (mesh_graph_rows keyed by from[e]; word_at reads the e-th word).
   - The vertices with no edge in enter the heap (heap_up, ordered by order_before).  While the heap holds any, its
     first leaves it (heap_down) and takes the next place, and each edge out of it lowers its head's count once (an
     edge given twice lowers it twice), the head entering the heap where its count reaches 0.  So a vertex is placed
     only after every vertex an edge into it leaves (the order is topological with respect to the dependencies), and
     at each place the vertex taken is the first of the ready ones by the priority, the greatest first, a tie to the
     lower vertex (GO3).  MSCCLang names no tie; the lower index is the one every party computes alike.
   - The priority is the consumer's operand.  Where this differs from the reference algorithm
     (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO3′): MSCCLang §5.2 step (1)'s own
     priority is "the depth (maximum hops from a root node) and reverse depth (maximum hops to a leaf node)", counted
     in hops; mesh-collective.c's is R5's, each instruction's completion time C̄ in the linear program's solution, the
     least first (mesh_graph_completion_times; R5 at mesh 0e56d8da: "It replaces MSCCLang's depth and reverse depth"),
     given here as -C̄ so the greatest is the least C̄, a tie to the lower vertex as R5 states ("Where two are equal,
     the heap takes the lower instruction"; design/compilation-coordination.md, "The revisions of 46ae8fdf", LP7).  The
     heap is step 3's whichever priority it is given.
     2026-11-05 (LP): the consumer's priority C̄, where it was the bottom level by eq. (7) (GO2), and before that
     MSCCLang's depth and reverse depth.  Why: R5 at mesh 0e56d8da.  What must stay: the comparison as it is
   - The heap's order is total, so the vertex it gives first among those in it does not depend on the order in which
     they entered: every party computes the same places from the same operands (F. B. Schneider, ACM Comput. Surv.
     1990, as R5 cites it).  Arithmetic: comparisons alone.
   - Vertices left unplaced when the heap empties lie on a cycle or after one: EINVAL, their place UINT32_MAX.  An edge
     naming no vertex is EINVAL before anything is written.  Every allocation precedes the first place written, so
     ENOMEM writes none.
   Cost: O((V + E) log V), each vertex entering and leaving the heap once at O(log V) each and each edge lowering one
   count once; memory O(V + E). */
static uint32_t heap_place(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *rows, size_t k),
                           const void *rows, int (*before)(const void *context, uint32_t a, uint32_t b),
                           const void *context, uint32_t *indegree, uint32_t *heap, uint32_t *position,
                           uint32_t *place) {
  for (uint32_t v = 0; v < vertices; v++) { indegree[v] = 0; position[v] = UINT32_MAX; place[v] = UINT32_MAX; }
  for (size_t k = 0; k < first[vertices]; k++) indegree[head(rows, k)]++;
  size_t count = 0;
  for (uint32_t v = 0; v < vertices; v++)
    if (!indegree[v]) { heap[count] = v; heap_up(heap, position, before, context, count); count++; }
  uint32_t placed = 0;
  while (count) {
    const uint32_t v = heap[0];
    position[v] = UINT32_MAX;
    if (--count) { heap[0] = heap[count]; heap_down(heap, position, before, context, count, 0); }
    place[v] = placed++;
    for (size_t k = first[v]; k < first[v + 1]; k++) {
      const uint32_t w = head(rows, k);
      if (!--indegree[w]) { heap[count] = w; heap_up(heap, position, before, context, count); count++; }
    }
  }
  return placed;
}
struct heap_rows { const uint32_t *to; const size_t *order; };
static uint32_t heap_rows_head(const void *rows, size_t k) {
  const struct heap_rows *r = rows;
  return r->to[r->order[k]];
}
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place) {
  size_t *first = malloc(((size_t)vertices + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  uint32_t *indegree = malloc(((size_t)vertices + 1) * sizeof *indegree);
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap);
  uint32_t *position = malloc(((size_t)vertices + 1) * sizeof *position);
  int result = !first || !order || !indegree || !heap || !position ? ENOMEM : 0;
  if (!result) {
    mesh_graph_rows(vertices, edges, word_at, from, first, order);
    const struct heap_rows rows = {to, order};
    if (heap_place(vertices, first, heap_rows_head, &rows, order_before, priority, indegree, heap, position, place) <
        vertices)
      result = EINVAL;
  }
  free(first); free(order); free(indegree); free(heap); free(position);
  return result;
}
static int lower_before(const void *context, uint32_t a, uint32_t b) {
  (void)context;
  return a < b;
}
int mesh_graph_order(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                     const void *context, uint32_t *indegree, uint32_t *ordinal) {
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap);
  uint32_t *position = malloc(((size_t)vertices + 1) * sizeof *position);
  int result = !heap || !position ? ENOMEM : 0;
  if (!result && heap_place(vertices, first, head, context, lower_before, NULL, indegree, heap, position, ordinal) <
                     vertices)
    result = EDEADLK;
  free(heap); free(position);
  return result;
}

/* New (design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the
   self-timed buffer bound"; design/compilation-coordination.md, "The revisions of 3f106747", BS-1): no tag source.
   2026-11-05 (design/aot-compilation-change-request.md, appendix item 2; design/compilation-coordination.md, "The
   revisions of 3f106747", BS-1): new.  What must stay: integer and IEEE-exact operations alone (+, -, x, / and
   comparisons, correctly rounded; the Makefile's -ffp-contract=off) and every loop bounded by the operands' counts, so
   every party computes the same components, means and delays from the same operands (mesh-graph.h); one copy each of
   Tarjan's search (components_search) and of the cycle tests (zero_delay_cycle, positive_cycle), which the five
   functions share; the caller's operands written only through the outputs the header names.
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP systems, IEEE Trans.
   Signal Processing 45(6), 1997, here BSL.  The scratch the searches share: compressed rows first[0 .. |V| + 1] and
   order[0 .. E) (mesh_graph_rows), Tarjan's index, low, stack, frame and at, the components, and Bellman and Ford's
   distances, |V| words each. */
struct graph_scratch { size_t *first, *order, *at; uint32_t *index, *low, *stack, *frame, *component; double *distance; };
static void scratch_free(struct graph_scratch *s) {
  free(s->first); free(s->order); free(s->at); free(s->index); free(s->low); free(s->stack); free(s->frame);
  free(s->component); free(s->distance);
}
/* every array, or ENOMEM (also where |V| + 1 rows or E positions pass their words) */
static int scratch_alloc(struct graph_scratch *s, uint32_t vertices, size_t edges) {
  const size_t n = (size_t)vertices + 2;
  s->first = malloc(n * sizeof *s->first); s->order = malloc((edges + 1) * sizeof *s->order);
  s->at = malloc(n * sizeof *s->at); s->index = malloc(n * sizeof *s->index); s->low = malloc(n * sizeof *s->low);
  s->stack = malloc(n * sizeof *s->stack); s->frame = malloc(n * sizeof *s->frame);
  s->component = malloc(n * sizeof *s->component); s->distance = malloc(n * sizeof *s->distance);
  return s->first && s->order && s->at && s->index && s->low && s->stack && s->frame && s->component && s->distance
             ? 0 : ENOMEM;
}
/* the times' sum, in vertex order */
static double time_sum(uint32_t vertices, const double *time) {
  double sum = 0;
  for (uint32_t v = 0; v < vertices; v++) sum += time[v];
  return sum;
}
/* EINVAL where an edge names no vertex, or (time not NULL) a time is negative or not finite or their sum is not finite
   (mesh-graph.h: time[v] >= 0, BSL §4.1's "execution time t(v)"; as mesh_graph_completion_times refuses its lengths);
   else 0 */
/* R. E. Tarjan's depth-first search for the strongly connected components (Depth-first search and linear graph
   algorithms, SIAM J. Comput. 1(2), 1972) over the rows s->first, s->order: vertex v's edges order[first[v]] ..
   order[first[v + 1] - 1], edge k entering to[k]; rows past |V| are not read.  Each vertex is numbered (index) as the
   search first reaches it; low[v] is the least number of a vertex still on the search's stack that v's subtree reaches
   by one edge; a vertex whose low is its own number roots a component, which takes the stack's vertices down to it.  A
   vertex is on the stack exactly while it is numbered and has no component.  The recursion is an explicit stack of
   frames (frame[d] its vertex, at[d] its next row position), at most |V| deep, so a large graph cannot overflow the C
   stack.  Roots in vertex order and each row in its order, so component[v] is numbered as the search completes the
   components, alike at every party; their count.  Cost: O(V + E), each vertex framed once and each edge read once. */
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
/* zero_delay_cycle's rows: an edge's row is its tail where its delay is 0, else row |V|, past every vertex's, which
   components_search does not read (an absent edge's delay, UINT64_MAX, is not 0) */
struct zero_rows { const uint32_t *from; const uint64_t *delay; uint32_t vertices; };
static uint32_t zero_row(const void *context, size_t e) {
  const struct zero_rows *z = context;
  return z->delay[e] ? z->vertices : z->from[e];
}
/* 1 where the edges of delay 0 hold a cycle (BSL Lemma 1: "Every cycle C in the IPC graph has a path delay of at least
   one if and only if the static schedule it is constructed from is free of deadlock"), exactly, by components alone:
   an edge of delay 0 whose ends lie in one component of those edges, which a self-loop's do, lies on a cycle with
   the component's path back to its tail; else 0 */
static int zero_delay_cycle(const struct graph_scratch *s, uint32_t vertices, size_t edges, const uint32_t *from,
                            const uint32_t *to, const uint64_t *delay) {
  const struct zero_rows rows = {from, delay, vertices};
  mesh_graph_rows(vertices + 1, edges, zero_row, &rows, s->first, s->order);
  components_search(s, vertices, to, s->component);
  for (size_t e = 0; e < edges; e++) if (!delay[e] && s->component[from[e]] == s->component[to[e]]) return 1;
  return 0;
}
/* The tolerance τ of positive_cycle, 1e-9 x Σ time: this file's choice.  Why: at λ_max the critical cycle's length is
   0 in exact arithmetic (BSL Definition 2: "A cycle in Gipc whose cycle mean is λ max is called a critical cycle of
   Gipc"), and the test must not report it.  Where no cycle is positive in exact arithmetic every distance is the
   length of a path of distinct vertices, at most Σ time, so a cycle's computed length, a sum of at most |V| terms each
   rounded once and added once, is off by a few times |V| x 2^-53 x Σ time, below τ while |V| is under about 10^6.  τ
   is relative to the times, so the test is the same whatever their unit (A3's weighting, the caller's); every time 0,
   τ is 0 and no length is positive. */
static double cycle_tolerance(uint32_t vertices, const double *time) { return 1e-9 * time_sum(vertices, time); }
/* Bellman and Ford's test (R. Bellman, On a routing problem, Quart. Appl. Math. 16, 1958; L. R. Ford, Network flow
   theory, RAND P-923, 1956) for a cycle of positive length under the lengths time[from[e]] - λ delay[e], an edge of
   delay UINT64_MAX absent: every distance 0 (a source joined to every vertex by an edge of length 0); each round
   relaxes the edges in their order, an edge only where it raises its head's distance by more than `tolerance`; 1 where
   round |V| still relaxes one, 0 where a round relaxes none.
   - Sound: a relaxation in round r reads its tail's distance as set in round r - 1 or r (had it been set before the
     edge's turn in round r - 1, that turn would have relaxed the edge), so the relaxations behind one in round |V|
     number |V| or more and repeat a vertex, between whose two settings its distance rose by more than `tolerance`: a
     cycle of that length, up to the sums' rounding.
   - Complete within the tolerance: where a round relaxes none, each edge's head lies within `tolerance` of its tail's
     distance plus its length, so a cycle C's length, the sum of those around it, is at most |C| x `tolerance`.
   Arithmetic: + - x and < alone, in the edges' order.  Cost: O(|V| E). */
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
/* MinDelay's test at λ: 1 where the edges hold no cycle of zero delay and none of positive length */
static int delays_pass(const struct graph_scratch *s, uint32_t vertices, const double *time, size_t edges,
                       const uint32_t *from, const uint32_t *to, const uint64_t *delay, double lambda, double tolerance) {
  return !zero_delay_cycle(s, vertices, edges, from, to, delay) &&
         !positive_cycle(s->distance, vertices, time, edges, from, to, delay, lambda, tolerance);
}
/* mesh-graph.h mesh_graph_cycle_mean.  BSL §9's BellmanFord, by "the Bellman-Ford algorithm discussed in pp. 94-97 of
   [15]" (E. L. Lawler, Combinatorial Optimization: Networks and Matroids, 1976): a binary search on λ, each step the
   test for a cycle of positive length under the lengths time[from[e]] - λ delay[e] (positive_cycle, at τ =
   cycle_tolerance).
   - A cycle of present edges with zero total delay: INFINITY (§9: "deadlock is equivalent to the existence of a cycle
     that has zero path delay, and is thus equivalent to an infinite maximum cycle mean"), found exactly, over the
     present edges of delay 0 alone (zero_delay_cycle), before any length is computed.
   - Otherwise every cycle's delay is at least 1 (Lemma 1), and a fundamental cycle's time is at most Σ time, so its
     mean is at most Σ time: the search's range is [0, Σ time], no cycle positive at its upper end.  Where the test
     finds none positive at 0 (no cycle, or none of positive time), *mean is 0.
   - Each step halves the range at its midpoint, the lower end moved where a cycle is positive there and the upper end
     where none is, for 200 steps or until the midpoint equals an end (the ends adjacent doubles): bounded, and the
     same steps at every party.
   - The test's resolution: where it finds no cycle positive at λ, a cycle C's length is at most |C|τ, so C's mean is
     at most λ + |C|τ / Delay(C) <= λ + |V|τ.  *mean is the search's upper end raised by |V|τ, the least value at which
     the test certifies that no cycle is positive in exact arithmetic.  Why (this file's choice): at the upper end
     itself the critical cycle's length lies within about τ of the test's threshold, so the test at that λ on this
     graph with edges added (MinDelay's X[e → r], DetermineDelays' X_i), whose distances round differently, could
     report one of this graph's own cycles, and MinDelay would fail where Fig. 8 succeeds, or take a delay more than
     the least where a new cycle's mean equals λ_max; at *mean every cycle of this graph has length at most 0, below
     any report's τ.  The cost: a mean above the critical cycle's by at most about (|V| + 1)τ, 1e-9 (|V| + 1) Σ time.
   Every allocation precedes *mean, which an error does not write.  Cost: O(|V| E) for each of at most 201 tests. */
int mesh_graph_cycle_mean(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          const uint64_t *delay, double *mean) {
  struct graph_scratch s;
  int result = scratch_alloc(&s, vertices, edges);
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
/* mesh-graph.h mesh_graph_min_delay.  BSL Fig. 8, Function MinDelay(X, e, λ, B): "Perform a binary search in the range
   [0, 1, …, B] to find the minimum value of r ∈ {0, 1, …, B} such that BellmanFord(X[e → r]) returns a value less
   than or equal to λ"; X[e → r] "is simply the DFG that results from “changing the delay”" on e (§9), here in a copy
   of the caller's delays, which are not written.
   - Each step's test (delays_pass): no cycle of zero delay among the present edges (zero_delay_cycle) and none of
     positive length under time[from[e]] - λ delay[e] (positive_cycle, at mesh_graph_cycle_mean's τ from the same
     times).  BellmanFord's value is at most λ exactly where no cycle's delay is 0 and no cycle's mean exceeds λ, so
     where no cycle's length is positive at λ: the test decides it without a search of its own.
   - r = B first: ERANGE where it fails (the input's "Assuming X[e → B] has estimated throughput no less than λ^-1").
     Then the least r of [0, B] that passes, by halving [lo, hi] (hi passes, every r below lo fails): at most 64 steps.
     A larger r lowers the length of every cycle through e and leaves the others', so in exact arithmetic the tests
     fail and then pass as r rises.
   - λ finite and not negative (the input's "a positive real number λ", with 0, DetermineDelays' λ_max where no cycle
     carries time), else EINVAL, as for e >= edges.  Every allocation precedes the first test, and an error does not
     write *least.
   Cost: O(|V| E) a test, at most 65 tests. */
int mesh_graph_min_delay(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                         const uint64_t *delay, size_t e, double lambda, uint64_t bound, uint64_t *least) {
  struct graph_scratch s;
  int result = scratch_alloc(&s, vertices, edges);
  uint64_t *changed = malloc((edges + 1) * sizeof *changed);
  if (!changed) result = ENOMEM;
  if (!result) {
    const double tolerance = cycle_tolerance(vertices, time);
    for (size_t k = 0; k < edges; k++) changed[k] = delay[k];
    uint64_t lo = 0, hi = bound;
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo) / 2;
      changed[e] = mid;
      if (delays_pass(&s, vertices, time, edges, from, to, changed, lambda, tolerance)) hi = mid; else lo = mid + 1;
    }
    *least = hi;
  }
  scratch_free(&s); free(changed);
  return result;
}
/* mesh-graph.h mesh_graph_convert_sc.  BSL §9, Fig. 6, Function Convert-to-SC-graph: "This algorithm simply “chains
   together” the source SCCs, and similarly, chains together the sink SCCs.  The construction is completed by
   connecting the first SCC of the “source chain” to the last SCC of the sink chain with an edge that we call the
   sink-source edge."
   - The SCCs by components_search over every edge; one SCC, or no vertex: strongly connected, none added.
   - An SCC is a source SCC where no edge enters it from another, a sink SCC where none leaves it for another (BSL §3);
     an SCC that no edge joins to another is both, and in both chains.
   - Each SCC's link: "From each source or sink SCC, the algorithm selects a vertex that has minimum execution time to
     be the chain “link” corresponding to that SCC"; its vertex of least time, the vertices scanned ascending and a
     later one taken only where its time is less, so a tie goes to the lower vertex.
   - The orderings (step 1, "Generate an ordering"): the SCCs ascending by their least vertex, the order in which an
     ascending scan of the vertices first meets them; Fig. 6 names none, and this one every party computes alike.
   - Out, in the order DetermineDelays takes (§9: "(e_0, e_1, …, e_m – 1) ≡ (e_s, a_1, a_2, …, a_k, b_l, b_l – 1, …,
     b_1), where e_s is the sink-source edge"): e_s from the last sink SCC's link to the first source SCC's (step 6),
     a_i from C_i's link to C_i+1's (step 3), b_i from D_i's link to D_i+1's (step 5).  Their number is "(n src + n snk
     – 1), where n src is the number of source SCCs, and n snk is the number of sink SCCs" (Theorem 4's proof), up to
     2|V| - 1 where SCCs of one vertex are both, within the header's room of 2 `vertices`.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 3f106747", G-3, the coordinating implementer's
   ruling): the room 2 `vertices`, where it was `vertices` with ENOMEM past it, so no count passes it (m + k - 1 <=
   2 x the SCCs - 1 < 2 |V|) and the test goes.  What must stay: every allocation before the first output.
   Every allocation precedes the first output written.  Cost: O(V + E). */
int mesh_graph_convert_sc(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *added, uint32_t *added_from, uint32_t *added_to) {
  struct graph_scratch s;
  int result = scratch_alloc(&s, vertices, edges);
  const size_t n = (size_t)vertices + 1;
  uint32_t *least = malloc(n * sizeof *least), *link = malloc(n * sizeof *link);
  uint32_t *sources = malloc(n * sizeof *sources), *sinks = malloc(n * sizeof *sinks);
  unsigned char *entered = malloc(n), *left = malloc(n);
  if (!result && (!least || !link || !sources || !sinks || !entered || !left)) result = ENOMEM;
  uint32_t count = 0, m = 0, k = 0;
  if (!result) {
    mesh_graph_rows(vertices, edges, word_at, from, s.first, s.order);
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
/* mesh-graph.h mesh_graph_determine_delays.  BSL §9, Fig. 8, Function DetermineDelays, over one edge array: the
   caller's edges, then the added ones in mesh_graph_convert_sc's order, edge `edges` + i being e_i.
   - X_0 = "Ĝs[e_0 → ∞, …, e_m – 1 → ∞]": the added edges at delay UINT64_MAX, absent from every cycle, so X_0's
     cycles are Gs's.
   - λ_max = mesh_graph_cycle_mean(X_0) (Fig. 8: "compute the max. cycle mean of G s"), computed even where nothing is
     added; EDEADLK where it is INFINITY (a cycle of zero delay, Lemma 1's deadlock).
   - d_ub = ⌈Σ time / λ_max⌉ (Fig. 8: "an upper bound on the delay required for any e i"; §4.3, (6): "the maximum
     execution time along any such newly introduced cycle can be at most" Σ t(x)): the quotient's integer part by a
     conversion, one more where that falls short of the quotient, an exact ceiling; 1 where λ_max is 0.
   - For i = 0, 1, …, m – 1: δ_i = mesh_graph_min_delay(X_i, e_i, λ_max, d_ub), then e_i's delay in the array δ_i,
     which makes it X_i+1.  §9: "the algorithm does guarantee that the output graph is not deadlocked, assuming that
     the input graph is not deadlocked"; here each δ_i passed MinDelay's test, which a cycle of zero delay fails.
   - added_delay[i] = δ_i, written once every δ is found; an error (ERANGE, MinDelay's; EDEADLK; EINVAL where an edge or
     an added edge names no vertex; ENOMEM) writes none.
   Cost: one cycle mean and m MinDelays, over |V| vertices and E + m edges. */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, uint64_t *added_delay) {
  const size_t total = edges + added;
  uint32_t *all_from = malloc((total + 1) * sizeof *all_from), *all_to = malloc((total + 1) * sizeof *all_to);
  uint64_t *all_delay = malloc((total + 1) * sizeof *all_delay);
  int result = all_from && all_to && all_delay ? 0 : ENOMEM;
  double lambda = 0;
  if (!result) {
    for (size_t e = 0; e < edges; e++) { all_from[e] = from[e]; all_to[e] = to[e]; all_delay[e] = delay[e]; }
    for (uint32_t i = 0; i < added; i++) {
      all_from[edges + i] = added_from[i]; all_to[edges + i] = added_to[i]; all_delay[edges + i] = UINT64_MAX;
    }
    result = mesh_graph_cycle_mean(vertices, time, total, all_from, all_to, all_delay, &lambda);
  }
  if (!result && lambda == INFINITY) result = EDEADLK;
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

/* mesh-graph.h mesh_graph_intersect: the intervals by key, then start, a tie in either order */
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

/* mesh-graph.h mesh_graph_row_visit */
void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context) {
  for (uint32_t j = 0; j < count; j++) visit(context, row[j]);
}

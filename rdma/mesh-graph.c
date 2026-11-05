#include "mesh-graph.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_order, ec2ad141 */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_arborescence_within and its search, 9bb6972a */
/* mesh_graph_arborescence_within's search: node order[at] given each of its edges in (into[v] the edge list by
   weight, its first in[v] to in[v + 1]), the rest each its lightest in rest[at + 1] */
struct within { uint32_t N,root,depth; const uint32_t *from,*order,*into; const uint32_t *in; const double *weight,*rest;
                int32_t *now,*best; double cost; };
/* where node v's walk up the parents now chosen reaches the root within `depth` edges: 1; it meets a node not yet
   given a parent: 1 (decided later); a cycle or past depth: 0 */
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
    /* every node given: each one's walk ends at the root within depth (no node left undecided) */
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
    /* this node's walk, and every chosen node's whose walk now runs through it */
    int ok=within_walk(s,v);
    for(uint32_t k=0;ok && k<at;k++)ok=within_walk(s,s->order[k]);
    if(ok)within_search(s,at+1,cost+s->weight[e]);
  }
  s->now[v]=-1;
}
/* mesh-graph.h mesh_graph_arborescence_within */
int mesh_graph_arborescence_within(uint32_t N,uint32_t root,uint32_t E,const uint32_t *from,const uint32_t *to,
                                   const double *weight,uint32_t depth,int32_t *chosen){
  uint32_t *in=calloc((size_t)N+2,sizeof *in),*into=malloc(((size_t)E+1)*sizeof *into),*order=malloc(((size_t)N+1)*sizeof *order);
  uint32_t *layer=malloc(((size_t)N+1)*sizeof *layer);
  double *rest=calloc((size_t)N+1,sizeof *rest);
  int32_t *now=malloc(((size_t)N+1)*sizeof *now),*best=malloc(((size_t)N+1)*sizeof *best);
  int status=-1;
  if(!in || !into || !order || !layer || !rest || !now || !best || root>=N)goto done;
  /* each node's edges in, lightest first */
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
  /* the nodes in breadth-first order from the root (a near node's parent decided first cuts the most) */
  for(uint32_t v=0;v<N;v++){layer[v]=UINT32_MAX;now[v]=-1;}
  layer[root]=0;
  uint32_t placed=0;
  for(uint32_t d=0,grew=1;grew;d++){
    grew=0;
    for(uint32_t e=0;e<E;e++)if(layer[from[e]]==d && layer[to[e]]==UINT32_MAX){layer[to[e]]=d+1;grew=1;}
  }
  for(uint32_t d=1;placed+1<N && d<=N;d++)for(uint32_t v=0;v<N;v++)if(v!=root && layer[v]==d)order[placed++]=v;
  if(placed+1!=N)goto done; /* a node the root reaches by no edges */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_arborescence, ab823357 */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_cycle_length with climb_return, f21a4d24 */
/* mesh-graph.h: the climb's visits, a cycle's return (the tag's comment here names the level's height and the mark
   too, whose functions are not brought) */
static void climb_return(void *consumer, uint32_t at, uint32_t step) {
  uint32_t *found = consumer;
  if (step && at == found[0] && !found[1]) found[1] = step;
}
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from) {
  uint32_t found[2] = {from, 0};
  mesh_graph_climb(parent, vertices, UINT32_MAX, from, climb_return, found);
  return found[1];
}

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_depth with depth_head, 8cd0c480 */
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

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_pack with mesh_packed_heavier and pack_same, 9bb6972a */
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
  /* shallow: root r's arborescences those within its eccentricity's depth (eccentricity[r], the last layer of its
     breadth-first layering), searched over every edge (mesh_graph_arborescence_within).
     Changed from the layering's edges alone (an edge from layer d to d + 1, so a tree its root's shortest paths),
     which left out trees as shallow that cross a layer's own edges (m5ult3 -> Mini -> m5ult2, depth 2): its packing
     carried 26.7 GB/s into a rank where trees within depth 2 carry 30, the whole capacity of the cut around the Mini
     (9bb6972a, the trees' linear program, 2026-11-03). What must stay: every tree
     as shallow as one from its root can be (a small segment's units on a tree of its root's eccentricity, run-23).
     Changed from the tag in this comment alone: the measurement is cited as 9bb6972a's message, which records it,
     where the tag cited a document deleted with everything citing it (mesh 90677543, metal-microbench 7aa6491a). */
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
  free(eccentricity);free(layer);
  return written;
}

/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_tree_loads, feef1156 */
/* mesh-graph.h mesh_graph_tree_loads */
void mesh_graph_tree_loads(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                           const double *weight, double *load){
  for(uint32_t k=0;k<trees;k++)
    for(uint32_t v=0;v<vertices;v++){
      const uint32_t u=parent[(size_t)k*vertices+v];
      if(v!=root[k] && u!=UINT32_MAX)load[(size_t)u*vertices+v]+=weight[k];
    }
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
/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_tree_depth, d30f02f5 */
/* mesh-graph.h mesh_graph_tree_depth */
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
/* Source: pre-return-20261103 rdma/mesh-graph.c mesh_graph_trees_rate, d30f02f5 */
/* mesh-graph.h mesh_graph_trees_rate */
double mesh_graph_trees_rate(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                             const double *weight, const double *capacity){
  double *load=calloc((size_t)vertices*vertices+1,sizeof *load),rate=INFINITY;
  if(!load)return 0;
  mesh_graph_tree_loads(vertices,trees,root,parent,weight,load);
  for(size_t e=0;e<(size_t)vertices*vertices;e++)if(load[e]>0){
    if(!(capacity[e]>0)){rate=0;break;}
    if(capacity[e]/load[e]<rate)rate=capacity[e]/load[e];
  }
  free(load);
  return isfinite(rate)?rate:0;
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
  const size_t edges = first[vertices];
  if (edges >= SIZE_MAX / sizeof(uint32_t)) return ENOMEM;
  uint32_t *heads = malloc((edges + 1) * sizeof *heads), *weights = malloc((edges + 1) * sizeof *weights);
  uint32_t *distance = malloc(((size_t)vertices + 1) * sizeof *distance);
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap), *place = malloc(((size_t)vertices + 1) * sizeof *place);
  unsigned char *wanted = calloc((size_t)vertices + 1, 1);
  const int result = !heads || !weights || !distance || !heap || !place || !wanted ? ENOMEM : 0;
  uint32_t distinct = 0;
  for (size_t k = 0; !result && k < edges; k++) { heads[k] = head(context, k); weights[k] = lag(context, k); }
  for (uint32_t x = 0; !result && x < targets; x++)
    if (!wanted[target[x]]) { wanted[target[x]] = 1; distinct++; }
  for (uint32_t s = 0; !result && distinct && s < sources; s++) {
    for (uint32_t v = 0; v < vertices; v++) { distance[v] = UINT32_MAX; place[v] = UINT32_MAX; }
    const uint32_t from = source[s];
    size_t count = 0;
    for (size_t k = first[from]; k < first[from + 1]; k++)
      count = lags_relax(heap, place, distance, count, heads[k], weights[k]);
    for (uint32_t left = distinct; count;) {
      const uint32_t at = heap[0];
      place[at] = UINT32_MAX;
      if (--count) { heap[0] = heap[count]; heap_down(heap, place, lags_before, distance, count, 0); }
      if (wanted[at] && !--left) break;
      for (size_t k = first[at]; k < first[at + 1]; k++)
        count = lags_relax(heap, place, distance, count, heads[k], (uint64_t)distance[at] + weights[k]);
    }
    for (uint32_t x = 0; x < targets; x++) lags[(size_t)s * targets + x] = distance[target[x]];
  }
  free(heads); free(weights); free(distance); free(heap); free(place); free(wanted);
  return result;
}

/* New (design/disjoint-pages-change-request.md, Appendix A, A3, "The assignment";
   design/disjoint-pages-coordination.md, "The allocator"): no tag source.
   mesh_graph_left_edge's placed runs, kept by first colour */
struct left_edge_run { uint64_t start,end; uint32_t item; };
/* mesh-graph.h mesh_graph_left_edge.  An item's run starts at colour 0 and walks the placed runs by first colour,
   moved to the end of each it overlaps and interferes with, until a placed run starts at or past its run's end (as
   every later one does).  After its turn a walked run does not interfere or ends at or below the start, which only
   rises, so the run reached overlaps none it interferes with; and a rise to a run's end passes only starts whose run
   overlaps that one, so the start reached is the lowest.  A start past capacity - size is ENOSPC at once (the lowest
   lies at or past it).  The run then goes in among the placed after those starting at or below it (a binary search
   and one move).  This is the offset calculation of Pisarchyk and Lee's greedy by size (the tensors placed so far
   walked in offset order), at the lowest colour free of interference, where theirs takes the smallest gap that fits.
   Cost: each item walks the placed runs once, so O(items^2) interference tests and run moves at worst (items all
   interfering: items^2 / 2 tests, 4.5 x 10^8 for 30,000), and O(items) memory.
   `interferes` NULL, every pair interfering: the walk's result without the walk.  The placed runs then lie end to
   end from colour 0 in the caller's order, so each item's lowest start is the end of the runs before it (an empty
   run, which overlaps none, colour 0 as above), and ENOSPC where that end passes capacity - size: O(items), no
   memory. */
int mesh_graph_left_edge(uint32_t items, const uint64_t *size, const uint32_t *order,
                         int (*interferes)(const void *context, uint32_t a, uint32_t b), const void *context,
                         uint64_t capacity, uint64_t *colour, uint64_t *used) {
  if (!interferes) {
    uint64_t end = 0;
    int full = 0;
    for (uint32_t n = 0; n < items; n++) {
      const uint32_t i = order[n];
      const uint64_t need = size[i];
      if (need > capacity - end) { full = 1; break; }
      colour[i] = need ? end : 0;
      end += need;
    }
    *used = end;
    return full ? ENOSPC : 0;
  }
  struct left_edge_run *placed = malloc(((size_t)items + 1) * sizeof *placed);
  size_t count = 0;
  uint64_t most = 0;
  int result = placed ? 0 : ENOMEM;
  for (uint32_t n = 0; !result && n < items; n++) {
    const uint32_t i = order[n];
    const uint64_t need = size[i];
    uint64_t c = 0;
    if (need > capacity) { result = ENOSPC; break; }
    for (size_t p = 0; p < count && placed[p].start < c + need; p++)
      if (placed[p].end > c && interferes(context, i, placed[p].item)) {
        c = placed[p].end;
        if (c > capacity - need) { result = ENOSPC; break; }
      }
    if (result) break;
    colour[i] = c;
    if (c + need > most) most = c + need;
    if (!need) continue; /* an empty run overlaps none */
    size_t lo = 0, hi = count;
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      if (placed[mid].start <= c) lo = mid + 1; else hi = mid;
    }
    memmove(placed + lo + 1, placed + lo, (count - lo) * sizeof *placed);
    placed[lo] = (struct left_edge_run){c, c + need, i};
    count++;
  }
  *used = most;
  free(placed);
  return result;
}

/* New (design/plan-to-bridge-change-request.md, R5; design/compilation-coordination.md, "The graph module" and T3):
   no tag source.
   levels' rows: compressed row entry k is edge order[k], which enters to[order[k]] */
struct levels_rows { const uint32_t *to; const size_t *order; };
static uint32_t levels_head(const void *context, size_t k) {
  const struct levels_rows *rows = context;
  return rows->to[rows->order[k]];
}
/* O. Sinnen and L. A. Sousa, IEEE TPDS 16(6) 2005, §2.2.1 eq. (7): bl(ni) = w(ni) + max over nj ∈ succ(ni)
   {c(eij) + bl(nj)}, "the length of the longest path leaving the node", each edge out of ni a term.  The edges out of
   each task as compressed rows (mesh_graph_rows keyed by from[e]; depth_head reads the k-th word), a topological order
   of them (mesh_graph_order: ordinal[n] n's place, every edge's origin placed before its end; its EDEADLK, a cycle,
   is EINVAL), then eq. (7) from the last place to the first, so every successor's level stands before a predecessor
   takes it; a task with no edge out (a sink node) is w alone.  Each term one correctly rounded +, the max by
   comparisons.  Every allocation precedes the first level written, so ENOMEM writes none. */
static int levels(uint32_t tasks, const double *weight, size_t edges, const uint32_t *from, const uint32_t *to,
                  const double *cost, double *level, uint32_t *ordinal) {
  for (size_t e = 0; e < edges; e++) if (from[e] >= tasks || to[e] >= tasks) return EINVAL;
  if (edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  size_t *first = malloc(((size_t)tasks + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  uint32_t *indegree = malloc(((size_t)tasks + 1) * sizeof *indegree), *inverse = malloc(((size_t)tasks + 1) * sizeof *inverse);
  int result = !first || !order || !indegree || !inverse ? ENOMEM : 0;
  if (!result) {
    mesh_graph_rows(tasks, edges, depth_head, from, first, order);
    const struct levels_rows rows = {to, order};
    if (mesh_graph_order(tasks, first, levels_head, &rows, indegree, ordinal)) result = EINVAL;
  }
  for (uint32_t n = 0; !result && n < tasks; n++) inverse[ordinal[n]] = n;
  for (uint32_t p = tasks; !result && p-- > 0;) {
    const uint32_t n = inverse[p];
    double most = 0;
    for (size_t k = first[n]; k < first[n + 1]; k++) {
      const double through = cost[order[k]] + level[to[order[k]]];
      if (k == first[n] || through > most) most = through;
    }
    level[n] = weight[n] + most;
  }
  free(first); free(order); free(indegree); free(inverse);
  return result;
}
/* mesh-graph.h mesh_graph_bottom_levels: eq. (7) as levels takes it, its topological order let go */
int mesh_graph_bottom_levels(uint32_t tasks, const double *weight, size_t edges, const uint32_t *from, const uint32_t *to,
                             const double *cost, double *level) {
  uint32_t *ordinal = malloc(((size_t)tasks + 1) * sizeof *ordinal);
  const int result = ordinal ? levels(tasks, weight, edges, from, to, cost, level, ordinal) : ENOMEM;
  free(ordinal);
  return result;
}

/* New (design/plan-to-bridge-change-request.md, R5; design/compilation-coordination.md, "The graph module", T3 to
   T5): no tag source.
   mesh_graph_contention_schedule's list (Algorithm 1, part 1): a task by decreasing bottom level, then increasing
   topological ordinal */
struct schedule_task { double level; uint32_t ordinal, task; };
static int schedule_task_before(const void *a, const void *b) {
  const struct schedule_task *x = a, *y = b;
  if (x->level != y->level) return x->level > y->level ? -1 : 1;
  return x->ordinal < y->ordinal ? -1 : x->ordinal > y->ordinal;
}
/* mesh_graph_contention_schedule's labels: item `item` of key `key`, sorted by (key, item) */
struct schedule_key { uint64_t key; size_t item; };
static int schedule_key_before(const void *a, const void *b) {
  const struct schedule_key *x = a, *y = b;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  return x->item < y->item ? -1 : x->item > y->item;
}
/* keys[0 .. count), each an item (from base on) and its key: leader[item - base] the least item of its key, so the
   items of one key name one leader.  The order is total (no two items alike), so every party's sort ends alike. */
static void schedule_leaders(size_t count, struct schedule_key *keys, size_t base, size_t *leader) {
  qsort(keys, count, sizeof *keys, schedule_key_before);
  for (size_t i = 0; i < count; i++)
    leader[keys[i].item - base] = i && keys[i].key == keys[i - 1].key ? leader[keys[i - 1].item - base] : keys[i].item;
}
/* mesh-graph.h mesh_graph_contention_schedule.  O. Sinnen and L. A. Sousa, IEEE TPDS 16(6) 2005, §6.1: Algorithm 1
   (list scheduling, §2.2.1) with its line 6 performed as Algorithm 2 (the scheduling of a free node in the contention
   model), each edge scheduled on its route as §5.2 states it.
   - Algorithm 1, part 1, "Sort nodes n ∈ V into list L, according to priority scheme and precedence constraints":
     decreasing bottom level, eq. (7) (levels, mesh_graph_bottom_levels' computation; design/compilation-coordination.md,
     T3), a tie by the topological ordinal levels takes them over.  No two tasks share an ordinal, so the contract's
     further tie by index never arises.  The list meets the precedence constraints: with every weight > 0 and every
     cost >= 0 (Definition 1: "The positive weight w(n)", "the nonnegative weight c(eij)") a predecessor's level is no
     less than its successor's (a correctly rounded + of a term >= 0 is no less than the term), and at an equal level
     its ordinal is the lower, so every task is free (its predecessors scheduled) at its turn.
   - Algorithm 1, line 5, "Find processor P ∈ P that allows earliest finish time of n": P is processor[n], given (the
     placement operand: design/aot-compilation-change-request.md, A2; design/plan-to-bridge-change-request.md, R5), so
     §6.1's tentative scheduling on every processor, the incoming edges removed from the links between tries, does not
     arise.  A processor is a label (the operands name no count of them; design/compilation-coordination.md, G-1):
     tasks of equal processor[] are on one, whose last finish is kept at the least such task (schedule_leaders).
   - Algorithm 2, "for each ni ∈ pred(nj) in a definite order": the edges into nj in the order of their indices (§6.1,
     "the edges might be processed in the order of their indices"; mesh_graph_rows keyed by to[e] keeps their order).
     "if proc(ni) ≠ P then determine route R = <L1, ..., Ll> from proc(ni) to P": R is the operand's
     link[route[e] .. route[e + 1]), in route order; "schedule eij on R" (§5.2, below); then "schedule nj on P".
   - An edge on its route, §5.2, nonaligned (§5.1.1; §5.1.3, "the nonaligned approach appears to better reflect the
     behavior of communication networks and its utilization in edge scheduling is proposed"): on link Lk,
     ς(e, Lk) = c(e) / b(Lk) (§5), b(L) the operand rate[]; the idle interval [A, B] the end technique's, A the link's
     last finish and B = ∞ (Condition 3 and Condition 6 admit the end or the insertion technique;
     design/compilation-coordination.md, T5), in which Condition 6, (12) B - A >= ς(e, Lk) and (13), always holds; the
     start (14): ts(e, L1) = max{A, tf(ni)}, and for k > 1 ts(e, Lk) = max{A, tf(e, Lk-1) - ς(e, Lk), ts(e, L1)},
     which meets Condition 5's causality, (10) and (11); tf(e, Lk) = ts(e, Lk) + ς(e, Lk), then the link's last finish
     (one edge on a link at a time, Condition 4, (9)).
   - Where this differs from the paper (design/compilation-coordination.md, T4): route entries of one
     communication[r] on one link are one occupation of it.  The first scheduled takes its interval by (14); every
     later one takes that start and finish as they stand and occupies nothing more (design/compilation-coordination.md,
     G-1: sound for T4's common prefixes of routes), and the entries after it on its own route follow it by (14),
     causality from its finish and its route's first start.  Sinnen and Sousa schedule
     each edge on its own; the consumer's communications cross a link once for all the edges that need them there.
     What must stay: one occupation per (communication, link), held by the first entry scheduled.
   - An edge's finish, Definition 8 (15): tf(ni) where proc(ni) = proc(nj), else tf(e, Ll).  nj's data-ready time (3):
     the max of its edges' finishes, 0 for a source node; its start (8): ts(nj, P) = max{A, tdr(nj, P)}, A the
     processor's last finish (the end technique, [tf(P), ∞), in which Condition 3's (6) always holds); its finish
     ts(nj, P) + w(nj), then the processor's last finish.
   start[n] ts(n, proc(n)); link_start[r] ts(e, Lk) of route entry r.  Arithmetic: +, -, / and comparisons alone, each
   correctly rounded (no libm; the file is compiled with -ffp-contract=off), and both sorts' orders total, so every
   party computes the same schedule from the same operands.  EINVAL: a weight not > 0 or not finite (Definition 1's
   positive weight), a cost negative or not finite (its nonnegative weight; a local edge's 0, Definition 2's free
   local communication), a rate not > 0, an edge naming no task, a route running backward (route[e] > route[e + 1])
   or naming no link, a remote edge with no route, a cycle; and, design/compilation-coordination.md, G-1, a local edge
   with a route or a time past the doubles' range (operands whose sums overflow).  Every allocation precedes the first
   output written, so ENOMEM writes none. */
int mesh_graph_contention_schedule(uint32_t tasks, const double *weight, const uint32_t *processor, size_t edges,
                                   const uint32_t *from, const uint32_t *to, const double *cost, const size_t *route,
                                   const uint32_t *link, const uint32_t *communication, uint32_t links,
                                   const double *rate, double *start, double *link_start) {
  for (uint32_t n = 0; n < tasks; n++) if (!(weight[n] > 0 && weight[n] <= DBL_MAX)) return EINVAL;
  for (uint32_t l = 0; l < links; l++) if (!(rate[l] > 0)) return EINVAL;
  for (size_t e = 0; e < edges; e++) {
    if (from[e] >= tasks || to[e] >= tasks || !(cost[e] >= 0 && cost[e] <= DBL_MAX) || route[e] > route[e + 1]) return EINVAL;
    if ((processor[from[e]] == processor[to[e]]) != (route[e] == route[e + 1])) return EINVAL;
  }
  const size_t base = route[0], entries = route[edges] - base;
  for (size_t r = base; r < route[edges]; r++) if (link[r] >= links) return EINVAL;
  if (edges >= SIZE_MAX / sizeof(struct schedule_key) || entries >= SIZE_MAX / sizeof(struct schedule_key)) return ENOMEM;
  const size_t keyed = entries > tasks ? entries : tasks;
  double *level = malloc(((size_t)tasks + 1) * sizeof *level), *finish = malloc(((size_t)tasks + 1) * sizeof *finish);
  double *busy = malloc(((size_t)tasks + 1) * sizeof *busy), *free_at = malloc(((size_t)links + 1) * sizeof *free_at);
  double *end = malloc((entries + 1) * sizeof *end);
  uint32_t *ordinal = malloc(((size_t)tasks + 1) * sizeof *ordinal);
  struct schedule_task *list = malloc(((size_t)tasks + 1) * sizeof *list);
  struct schedule_key *keys = malloc((keyed + 1) * sizeof *keys);
  size_t *home = malloc(((size_t)tasks + 1) * sizeof *home), *leader = malloc((entries + 1) * sizeof *leader);
  size_t *holder = malloc((entries + 1) * sizeof *holder);
  size_t *in = malloc(((size_t)tasks + 1) * sizeof *in), *into = malloc((edges + 1) * sizeof *into);
  int result = !level || !finish || !busy || !free_at || !end || !ordinal || !list || !keys || !home || !leader ||
               !holder || !in || !into ? ENOMEM : 0;
  if (!result) result = levels(tasks, weight, edges, from, to, cost, level, ordinal);
  if (!result) {
    for (uint32_t n = 0; n < tasks; n++) keys[n] = (struct schedule_key){processor[n], n};
    schedule_leaders(tasks, keys, 0, home);
    for (size_t r = base; r < route[edges]; r++)
      keys[r - base] = (struct schedule_key){(uint64_t)communication[r] << 32 | link[r], r};
    schedule_leaders(entries, keys, base, leader);
    for (uint32_t n = 0; n < tasks; n++) { list[n] = (struct schedule_task){level[n], ordinal[n], n}; busy[n] = 0; }
    qsort(list, tasks, sizeof *list, schedule_task_before);
    mesh_graph_rows(tasks, edges, depth_head, to, in, into);
    for (uint32_t l = 0; l < links; l++) free_at[l] = 0;
    for (size_t r = 0; r < entries; r++) holder[r] = SIZE_MAX;
  }
  for (uint32_t i = 0; !result && i < tasks; i++) {
    const uint32_t nj = list[i].task, P = processor[nj];
    double ready = 0;
    for (size_t k = in[nj]; !result && k < in[nj + 1]; k++) {
      const size_t e = into[k];
      const uint32_t ni = from[e];
      double arrive = finish[ni], first_start = 0;
      if (processor[ni] != P)
        for (size_t r = route[e]; r < route[e + 1]; r++) {
          const double occupy = cost[e] / rate[link[r]];
          const size_t g = leader[r - base] - base;
          if (holder[g] == SIZE_MAX) {
            double s = free_at[link[r]];
            if (r == route[e]) { if (finish[ni] > s) s = finish[ni]; }
            else { if (arrive - occupy > s) s = arrive - occupy; if (first_start > s) s = first_start; }
            link_start[r] = s;
            end[r - base] = s + occupy;
            if (!(end[r - base] <= DBL_MAX)) { result = EINVAL; break; }
            free_at[link[r]] = end[r - base];
            holder[g] = r;
          } else {
            link_start[r] = link_start[holder[g]];
            end[r - base] = end[holder[g] - base];
          }
          if (r == route[e]) first_start = link_start[r];
          arrive = end[r - base];
        }
      if (arrive > ready) ready = arrive;
    }
    if (result) break;
    const double A = busy[home[nj]];
    start[nj] = A > ready ? A : ready;
    finish[nj] = start[nj] + weight[nj];
    if (!(finish[nj] <= DBL_MAX)) { result = EINVAL; break; }
    busy[home[nj]] = finish[nj];
  }
  free(level); free(finish); free(busy); free(free_at); free(end); free(ordinal); free(list); free(keys); free(home);
  free(leader); free(holder); free(in); free(into);
  return result;
}

/* New (design/plan-to-bridge-change-request.md, R5, "The order of each cable's list, by the reference algorithm";
   design/compilation-coordination.md, "The documents' revisions of edd4b583", "The graph module" and GO3): no tag
   source.
   mesh_graph_heap_order's comparison over the priorities (`context`): a comes before b where its priority is greater,
   a tie to the lower vertex (GO3: "the greatest bottom level first ... a tie to the lower instruction"); comparisons
   alone, a strict total order of the vertices over priorities that compare (none NaN). */
static int order_before(const void *context, uint32_t a, uint32_t b) {
  const double *priority = context;
  return priority[a] > priority[b] || (priority[a] == priority[b] && a < b);
}
/* mesh-graph.h mesh_graph_heap_order.  M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS
   2023, §5.2, step 3: "Sort instructions into a global topological order with respect to their dependencies with a
   heap using the priority to order instructions."
   - Each vertex's count of the edges into it (an edge given twice counted twice), and the edges out of each vertex as
     compressed rows (mesh_graph_rows keyed by from[e]; depth_head reads the e-th word), as levels takes them.
   - The vertices with no edge in enter the heap (heap_up, ordered by order_before).  While the heap holds any, its
     first leaves it (heap_down) and takes the next place, and each edge out of it lowers its head's count once (an
     edge given twice lowers it twice), the head entering the heap where its count reaches 0.  So a vertex is placed
     only after every vertex an edge into it leaves (the order is topological with respect to the dependencies), and
     at each place the vertex taken is the first of the ready ones by the priority, the greatest first, a tie to the
     lower vertex (GO3).  MSCCLang names no tie; the lower index is the one every party computes alike.
   - The heap's order is total, so the vertex it gives first among those in it does not depend on the order in which
     they entered: every party computes the same places from the same operands (F. B. Schneider, ACM Comput. Surv.
     1990, as R5 cites it).  Arithmetic: comparisons alone.
   - Vertices left unplaced when the heap empties lie on a cycle or after one: EINVAL, their place UINT32_MAX.  An edge
     naming no vertex is EINVAL before anything is written.  Every allocation precedes the first place written, so
     ENOMEM writes none.
   Cost: O((V + E) log V), each vertex entering and leaving the heap once at O(log V) each and each edge lowering one
   count once; memory O(V + E). */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place) {
  for (size_t e = 0; e < edges; e++) if (from[e] >= vertices || to[e] >= vertices) return EINVAL;
  if (edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  size_t *first = malloc(((size_t)vertices + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  size_t *indegree = malloc(((size_t)vertices + 1) * sizeof *indegree);
  uint32_t *heap = malloc(((size_t)vertices + 1) * sizeof *heap);
  uint32_t *position = malloc(((size_t)vertices + 1) * sizeof *position);
  int result = !first || !order || !indegree || !heap || !position ? ENOMEM : 0;
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, from, first, order);
    for (uint32_t v = 0; v < vertices; v++) { indegree[v] = 0; position[v] = UINT32_MAX; place[v] = UINT32_MAX; }
    for (size_t e = 0; e < edges; e++) indegree[to[e]]++;
    size_t count = 0;
    for (uint32_t v = 0; v < vertices; v++)
      if (!indegree[v]) { heap[count] = v; heap_up(heap, position, order_before, priority, count); count++; }
    uint32_t placed = 0;
    while (count) {
      const uint32_t v = heap[0];
      position[v] = UINT32_MAX;
      if (--count) { heap[0] = heap[count]; heap_down(heap, position, order_before, priority, count, 0); }
      place[v] = placed++;
      for (size_t k = first[v]; k < first[v + 1]; k++) {
        const uint32_t head = to[order[k]];
        if (!--indegree[head]) { heap[count] = head; heap_up(heap, position, order_before, priority, count); count++; }
      }
    }
    if (placed < vertices) result = EINVAL;
  }
  free(first); free(order); free(indegree); free(heap); free(position);
  return result;
}

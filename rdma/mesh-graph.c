#include "mesh-graph.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CommonCrypto/CommonDigest.h>
/* design/plan-to-bridge-change-request.md R5, Code; HiGHS 1.15.1, highs/interfaces/highs_c_api.h */
#include <interfaces/highs_c_api.h>

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
  uint32_t *parent = malloc(((size_t)vertices + 1) * sizeof *parent);
  uint32_t *order = malloc(((size_t)vertices + 1) * sizeof *order);
  size_t *next = malloc(((size_t)vertices + 1) * sizeof *next);
  unsigned char *color = calloc((size_t)vertices + 1, 1);
  int status = !parent || !order || !next || !color ? ENOMEM : 0;
  uint32_t count = 0;
  if (!status) parent[root] = UINT32_MAX;
  for (uint32_t v = root; !status && v != UINT32_MAX;) {
    if (!color[v]) { color[v] = 1; next[v] = first[v]; order[count++] = v; }
    if (next[v] == first[v + 1]) { color[v] = 2; v = parent[v]; continue; }
    const uint32_t w = head(context, next[v]++);
    if (color[w] == 1) status = EDEADLK;
    else if (!color[w]) { parent[w] = v; v = w; }
  }
  for (uint32_t k = 0; !status && k < count; k++) status = visit(consumer, order[k], parent[order[k]]);
  free(parent); free(order); free(next); free(color);
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
                         uint64_t weight) {
  if (weight == UINT64_MAX || at > UINT64_MAX - 1 - weight || at + weight >= distance[v]) return count;
  distance[v] = at + weight;
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
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint64_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint64_t *lags) {
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
    size_t count = 0;
    for (size_t k = first[from]; k < first[from + 1]; k++)
      count = lags_relax(heap, place, distance, count, heads[k], 0, weights[k]);
    for (uint32_t left = distinct; count;) {
      const uint32_t at = heap[0];
      place[at] = UINT32_MAX;
      if (--count) { heap[0] = heap[count]; heap_down(heap, place, lags_before, distance, count, 0); }
      if (wanted[at] && !--left) break;
      for (size_t k = first[at]; k < first[at + 1]; k++)
        count = lags_relax(heap, place, distance, count, heads[k], distance[at], weights[k]);
    }
    for (uint32_t x = 0; x < targets; x++) lags[(size_t)s * targets + x] = distance[target[x]];
  }
  free(heads); free(weights); free(distance); free(heap); free(place); free(wanted);
  return result;
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
int mesh_graph_contract(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                        const unsigned char *kept, size_t room, struct mesh_graph_dependence *contracted, size_t *made) {
  for (size_t e = 0; e < edges; e++) if (edge[e].from >= vertices || edge[e].to >= vertices) return EINVAL;
  if (vertices >= UINT32_MAX / 2 || edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  uint32_t keeps = 0;
  for (uint32_t v = 0; v < vertices; v++) keeps += kept[v] != 0;
  if ((size_t)keeps * keeps >= SIZE_MAX / sizeof(uint64_t)) return ENOMEM;
  size_t *first = malloc((2 * (size_t)vertices + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  uint32_t *source = malloc(((size_t)keeps + 1) * sizeof *source), *target = malloc(((size_t)keeps + 1) * sizeof *target);
  uint64_t *lags = malloc(((size_t)keeps * keeps + 1) * sizeof *lags);
  int result = first && order && source && target && lags ? 0 : ENOMEM;
  for (uint32_t v = 0, k = 0; !result && v < vertices; v++)
    if (kept[v]) { source[k] = v; target[k++] = vertices + v; }
  const struct contract_graph g = {edge, order, kept, vertices};
  if (!result) {
    mesh_graph_rows(2 * vertices, edges, contract_tail, edge, first, order);
    if (keeps) result = mesh_graph_lags(2 * vertices, first, contract_head, contract_lag, &g, keeps, source, keeps, target, lags);
  }
  size_t count = 0;
  for (size_t p = 0; !result && p < (size_t)keeps * keeps; p++) {
    if (lags[p] == UINT64_MAX) continue;
    if (count < room) contracted[count] = (struct mesh_graph_dependence){source[p / keeps], source[p % keeps], lags[p]};
    count++;
  }
  if (!result) { *made = count; if (count > room) result = ERANGE; }
  free(first); free(order); free(source); free(target); free(lags);
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
  free(pair); free(first); free(member); free(key); free(lower); free(upper); free(value); free(column); free(dual);
  free(row_value); free(row_dual); free(start); free(index);
  return result;
}

/* design/plan-to-bridge-change-request.md R5, "The order of each cable's list, by the reference algorithm" */
static int order_before(const void *context, uint32_t a, uint32_t b) {
  const double *priority = context;
  return priority[a] > priority[b] || (priority[a] == priority[b] && a < b);
}
/* M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2, step 3; design/plan-to-bridge-change-request.md R5; F. B. Schneider,
   ACM Comput. Surv. 1990 */
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

/* design/torch-source-lowering-change-request.md item 3 */
static int access_conflict(const struct mesh_graph_access *a, const struct mesh_graph_access *b) {
  return (a->write || b->write) && !(a->write == 2 && b->write == 2);
}
static int access_before(const void *a, const void *b) {
  const struct mesh_graph_access *x = a, *y = b;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  if (x->start != y->start) return x->start < y->start ? -1 : 1;
  return (x->vertex > y->vertex) - (x->vertex < y->vertex);
}
int mesh_graph_dependences(size_t accesses, struct mesh_graph_access *access, size_t supplied,
                           const uint32_t *supplied_from, const uint32_t *supplied_to, size_t room, uint32_t *from,
                           uint32_t *to, size_t *edges) {
  for (size_t e = 0; e < supplied; e++) if (supplied_from[e] >= supplied_to[e]) return EINVAL;
  if (supplied >= SIZE_MAX / sizeof(uint64_t) - 64) return ENOMEM;
  if (accesses) qsort(access, accesses, sizeof *access, access_before);
  size_t made = 0, held = supplied + 64, count = 0;
  uint64_t *pair = malloc(held * sizeof *pair);
  int result = pair ? 0 : ENOMEM;
  for (size_t e = 0; !result && e < supplied; e++) pair[made++] = (uint64_t)supplied_from[e] << 32 | supplied_to[e];
  for (size_t i = 0; !result && i < accesses; i++)
    for (size_t j = i + 1; j < accesses && access[j].key == access[i].key && access[j].start < access[i].end; j++) {
      if (access[j].vertex == access[i].vertex || !access_conflict(access + i, access + j) || access[j].end <= access[j].start)
        continue;
      if (made == held) {
        const size_t grown = held ? 2 * held : 64;
        uint64_t *more = grown < SIZE_MAX / sizeof *more ? realloc(pair, grown * sizeof *more) : NULL;
        if (!more) { result = ENOMEM; break; }
        pair = more; held = grown;
      }
      const uint32_t a = access[i].vertex, b = access[j].vertex;
      pair[made++] = a < b ? (uint64_t)a << 32 | b : (uint64_t)b << 32 | a;
    }
  if (!result && made) qsort(pair, made, sizeof *pair, completion_edge_before);
  for (size_t k = 0; !result && k < made; k++) {
    if (k && pair[k] == pair[k - 1]) continue;
    if (count < room) { from[count] = (uint32_t)(pair[k] >> 32); to[count] = (uint32_t)pair[k]; }
    count++;
  }
  if (!result) { *edges = count; if (count > room) result = ERANGE; }
  free(pair);
  return result;
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
int mesh_graph_redundant(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, const uint64_t *delay,
                         const unsigned char *synchronization, unsigned char *redundant) {
  for (size_t e = 0; e < edges; e++) if (from[e] >= vertices || to[e] >= vertices || from[e] == to[e]) return EINVAL;
  if (vertices == UINT32_MAX || edges >= SIZE_MAX / (4 * sizeof(struct redundant_entry))) return ENOMEM;
  const size_t n = (size_t)vertices + 1;
  size_t *first = malloc(n * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  uint64_t *pair = malloc((edges + 1) * sizeof *pair);
  struct redundant_search s = {malloc(2 * n * sizeof *s.best), malloc(2 * n * sizeof *s.first),
                               malloc((3 * edges + 1) * sizeof *s.heap), 0};
  unsigned char *done = malloc(2 * n);
  int result = first && order && pair && s.best && s.first && s.heap && done ? 0 : ENOMEM;
  size_t synchronizations = 0;
  for (size_t e = 0; !result && e < edges; e++) if (synchronization[e]) pair[synchronizations++] = (uint64_t)from[e] << 32 | to[e];
  if (!result && synchronizations) qsort(pair, synchronizations, sizeof *pair, completion_edge_before);
  for (size_t k = 1; !result && k < synchronizations; k++) if (pair[k] == pair[k - 1]) result = EINVAL;
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, from, first, order);
    for (size_t e = 0; e < edges; e++) redundant[e] = 0;
  }
  for (uint32_t u = 0; !result && u < vertices; u++) {
    int any = 0;
    for (size_t k = first[u]; k < first[u + 1]; k++) any |= synchronization[order[k]] != 0;
    if (!any) continue;
    for (size_t v = 0; v < 2 * (size_t)vertices; v++) { s.best[v] = UINT64_MAX; s.first[v] = SIZE_MAX; done[v] = 0; }
    s.count = 0;
    for (size_t k = first[u]; k < first[u + 1]; k++) redundant_offer(&s, to[order[k]], delay[order[k]], order[k]);
    while (s.count) {
      const struct redundant_entry x = redundant_pop(&s);
      const size_t at = 2 * (size_t)x.vertex;
      const int label = s.best[at] == x.at && s.first[at] == x.first && !done[at] ? 0
                      : s.best[at + 1] == x.at && s.first[at + 1] == x.first && !done[at + 1] ? 1 : -1;
      if (label < 0) continue;
      done[at + (size_t)label] = 1;
      for (size_t k = first[x.vertex]; k < first[x.vertex + 1]; k++)
        if (delay[order[k]] <= UINT64_MAX - x.at) redundant_offer(&s, to[order[k]], x.at + delay[order[k]], x.first);
    }
    for (size_t k = first[u]; k < first[u + 1]; k++) {
      const size_t e = order[k], at = 2 * (size_t)to[e];
      if (synchronization[e]) redundant[e] = (s.first[at] == e ? s.best[at + 1] : s.best[at]) <= delay[e];
    }
  }
  free(first); free(order); free(pair); free(s.best); free(s.first); free(s.heap); free(done);
  return result;
}

/* design/torch-source-lowering-change-request.md item 3; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans.
   Signal Processing 45(6), 1997, §7, Fig. 4 */
int mesh_graph_synchronization(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                               unsigned char *redundant, size_t *kept) {
  if (edges >= SIZE_MAX / sizeof(uint64_t)) return ENOMEM;
  uint64_t *delay = calloc(edges + 1, sizeof *delay);
  unsigned char *synchronization = malloc(edges + 1);
  int result = delay && synchronization ? 0 : ENOMEM;
  if (!result) {
    memset(synchronization, 1, edges + 1);
    result = mesh_graph_redundant(vertices, edges, from, to, delay, synchronization, redundant);
  }
  size_t made = 0;
  for (size_t e = 0; !result && e < edges; e++) made += !redundant[e];
  if (!result) *kept = made;
  free(delay); free(synchronization);
  return result;
}

/* design/torch-source-lowering-change-request.md item 3; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2 */
int mesh_graph_schedule(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                        const unsigned char *after, int self_timed, uint32_t *level, uint32_t *place) {
  for (size_t e = 0; e < edges; e++) if (from[e] >= vertices || to[e] >= vertices) return EINVAL;
  if (vertices == UINT32_MAX || edges >= SIZE_MAX / sizeof(size_t)) return ENOMEM;
  size_t *first = malloc(((size_t)vertices + 1) * sizeof *first), *order = malloc((edges + 1) * sizeof *order);
  double *priority = malloc(((size_t)vertices + 1) * sizeof *priority);
  int result = first && order && priority ? 0 : ENOMEM;
  if (!result) mesh_graph_rows(vertices, edges, depth_head, to, first, order);
  uint32_t chained = UINT32_MAX;
  for (uint32_t v = 0; !result && v < vertices; v++) {
    const unsigned char a = after ? after[v] : 0;
    const int ordered = !self_timed && a != MESH_GRAPH_UNORDERED;
    level[v] = ordered && chained != UINT32_MAX ? chained + (a == 1) : 0;
    for (size_t k = first[v]; !result && k < first[v + 1]; k++) {
      const uint32_t u = from[order[k]];
      if (u >= v) result = EDEADLK;
      else if (level[u] + 1 > level[v]) level[v] = level[u] + 1;
    }
    if (ordered) chained = level[v];
    priority[v] = -(double)level[v];
  }
  if (!result && self_timed) result = mesh_graph_heap_order(vertices, priority, edges, from, to, place);
  for (uint32_t v = 0; !result && !self_timed && v < vertices; v++) place[v] = v;
  free(first); free(order); free(priority);
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
static int signature_wait(uint32_t p) {
  return p == MESH_GRAPH_QUEUE_WAIT || p == MESH_GRAPH_FENCE_WAIT || p == MESH_GRAPH_HOST_WAIT || p == MESH_GRAPH_HOST_NOTIFY;
}
static uint32_t signature_class(uint32_t p) {
  return p == MESH_GRAPH_FENCE_UPDATE || p == MESH_GRAPH_FENCE_WAIT ? 1 : p == MESH_GRAPH_INITIAL ? 2 :
         p == MESH_GRAPH_HOST_PUBLISH ? 3 : 0;
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

void mesh_graph_signature_free(struct mesh_graph_signature *s) {
  free(s->processor); free(s->task); free(s->operation); free(s->dependence); free(s->edge); free(s->position);
  free(s->count); free(s->least); free(s->previous); free(s->waited); free(s->first); free(s->order); free(s->unknown);
  free(s->alias);
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

/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §3, §9, Theorem 4, Figs. 6
   and 8; design/aot-compilation-change-request.md, appendix item 2, BS-4 */
static int resync_connected(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, size_t skip,
                            uint32_t *f, uint32_t *t, uint32_t *component, uint32_t *count) {
  size_t m = 0;
  for (size_t e = 0; e < edges; e++) if (e != skip) { f[m] = from[e]; t[m++] = to[e]; }
  return mesh_graph_components(vertices, m, f, t, component, count);
}
struct resync_end { uint32_t from, to, edge; };
static int resync_end_before(const void *a, const void *b) {
  const struct resync_end *x = a, *y = b;
  if (x->from != y->from) return x->from < y->from ? -1 : 1;
  if (x->to != y->to) return x->to < y->to ? -1 : 1;
  return (x->edge > y->edge) - (x->edge < y->edge);
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
  struct resync_end *end = malloc(((size_t)added + 1) * sizeof *end);
  unsigned char *ok = malloc((size_t)added + 1);
  struct graph_scratch g;
  result = scratch_alloc(&g, T, total);
  if (!result && !(all_from && all_to && f && t && component && alone && all_delay && end && ok)) result = ENOMEM;
  uint32_t count = 0;
  for (size_t e = 0; !result && e < edges; e++) { all_from[e] = from[e]; all_to[e] = to[e]; all_delay[e] = delay[e]; }
  for (uint32_t i = 0; !result && i < added; i++) {
    all_from[edges + i] = added_from[i]; all_to[edges + i] = added_to[i]; all_delay[edges + i] = UINT64_MAX;
    end[i] = (struct resync_end){added_from[i], added_to[i], i};
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
  if (!result) qsort(end, added, sizeof *end, resync_end_before);
  for (size_t k = 0; !result && k < s->synchronized; k++) {
    struct mesh_graph_synchronized *e = s->edge + k;
    if (e->state != MESH_GRAPH_FALSE || !signature_enforces(s->operation, e)) continue;
    const struct resync_end probe = {s->operation[e->signal].vertex, s->operation[e->wait].vertex, 0};
    for (size_t low = graph_lower(end, added, sizeof *end, &probe, resync_end_before);
         low < added && end[low].from == probe.from && end[low].to == probe.to; low++) {
      const uint32_t i = end[low].edge;
      if (e->delay < offset[probe.from] || e->delay - offset[probe.from] != added_delay[i]) continue;
      e->witness = i;
      if (ok[i]) e->state = MESH_GRAPH_TRANSFORMED;
      break;
    }
  }
  if (!result) {
    memset(s->states, 0, sizeof s->states);
    for (size_t k = 0; k < s->synchronized; k++) s->states[s->edge[k].state]++;
    result = signature_digest(s);
  }
  scratch_free(&g);
  free(all_from); free(all_to); free(f); free(t); free(component); free(alone); free(all_delay); free(end); free(ok);
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
int mesh_graph_record(const struct mesh_graph_signature *s, void *record, size_t room, size_t *made) {
  struct record_part part[7];
  *made = record_parts(s, part);
  if (*made > room) return ERANGE;
  struct mesh_graph_record *head = record;
  memset(record, 0, *made);
  *head = (struct mesh_graph_record){.bytes = *made, .processors = s->processors, .tasks = s->tasks,
                                     .operations = s->operations, .dependences = s->dependences,
                                     .synchronized = s->synchronized};
  memcpy(head->digest, s->digest, sizeof head->digest);
  size_t at = sizeof *head;
  for (int k = 0; k < 7; k++) { memcpy((char *)record + at, part[k].data, part[k].size); at = record_align(at + part[k].size); }
  return 0;
}
int mesh_graph_record_read(const void *record, struct mesh_graph_signature *s) {
  const struct mesh_graph_record *head = record;
  memset(s, 0, sizeof *s);
  s->processors = head->processors; s->tasks = head->tasks; s->operations = head->operations;
  s->dependences = head->dependences; s->synchronized = head->synchronized;
  s->processor = malloc(((size_t)s->processors + 1) * sizeof *s->processor);
  s->task = malloc(((size_t)s->tasks + 1) * sizeof *s->task);
  s->dependence = malloc((s->dependences + 1) * sizeof *s->dependence);
  s->operation = malloc((s->operations + 1) * sizeof *s->operation);
  s->edge = malloc((s->synchronized + 1) * sizeof *s->edge);
  s->least = malloc((s->dependences + 1) * sizeof *s->least);
  s->alias = malloc((s->operations + 1) * sizeof *s->alias);
  int result = s->processor && s->task && s->dependence && s->operation && s->edge && s->least && s->alias ? 0 : ENOMEM;
  struct record_part part[7];
  if (!result && record_parts(s, part) != head->bytes) result = EINVAL;
  size_t at = sizeof *head;
  for (int k = 0; !result && k < 7; k++) { memcpy(part[k].data, (const char *)record + at, part[k].size); at = record_align(at + part[k].size); }
  if (!result) result = signature_derive(s);
  if (result) mesh_graph_signature_free(s);
  return result;
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
  if (!result) result = signature_derive(s);
  free(pa); free(pb); free(ia); free(ib); free(matched); free(cursor); free(key); free(taken);
  if (result) mesh_graph_signature_free(s);
  return result;
}

/* design/synchronization-signature-change-request.md item 2 */
static const char *const signature_kind[MESH_GRAPH_KINDS] = {"queue", "host", "bridge", "kernel"};
static const char *const signature_primitive[MESH_GRAPH_PRIMITIVES] = {
  "submit", "queue_signal", "queue_wait", "fence_update", "fence_wait", "pass_barrier", "queue_barrier", "host_wait",
  "host_notify", "bridge_start", "threadgroup_barrier", "simdgroup_barrier", "atomic", "producer_barrier",
  "bridge_raise", "initial", "host_publish", "kernel_fence", "host_signal"};
static const char *const signature_state[MESH_GRAPH_STATES] = {"image", "stronger", "false", "unmatched", "satisfied",
                                                              "transformed"};
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
int mesh_graph_greedy_by_size(size_t records, const uint64_t *size, int (*intersect)(const void *context, size_t a, size_t b),
                              const void *context, uint64_t align, uint64_t *offset, uint64_t *total) {
  struct greedy_record *ordered_records = malloc((records + 1) * sizeof *ordered_records);
  size_t *ordered_allocs = malloc((records + 1) * sizeof *ordered_allocs), allocs = 0;
  int result = ordered_records && ordered_allocs ? 0 : ENOMEM;
  *total = 0;
  for (size_t i = 0; !result && i < records; i++) ordered_records[i] = (struct greedy_record){size[i], i};
  if (!result) qsort(ordered_records, records, sizeof *ordered_records, greedy_compare_by_size);
  for (size_t r = 0; !result && r < records; r++) {
    const size_t idx = ordered_records[r].idx;
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
    size_t it = 0;
    while (it < allocs && offset[ordered_allocs[it]] <= best_offset) it++;
    memmove(ordered_allocs + it + 1, ordered_allocs + it, (allocs - it) * sizeof *ordered_allocs);
    ordered_allocs[it] = idx;
    allocs++;
    offset[idx] = best_offset;
    if (best_offset + size[idx] > *total) *total = best_offset + size[idx];
  }
  free(ordered_records); free(ordered_allocs);
  return result;
}

void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context) {
  for (uint32_t j = 0; j < count; j++) visit(context, row[j]);
}

/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §5.2; B. R. Rau, MICRO 1994;
   design/torch-source-lowering-change-request.md item 3 */
size_t mesh_graph_delayed(const size_t *first, const size_t *order, const uint32_t *tail, uint32_t row, uint32_t count,
                          uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot) {
  size_t made = 0;
  for (size_t q = first[row]; q < first[row + 1]; q++) {
    const uint32_t from = tail[order[q]], delay = lag - 1 - from / count;
    if (t < delay) continue;
    source[made] = from % count; slot[made++] = (uint32_t)((t - delay) % lag);
  }
  return made;
}

/* G. E. Blelloch, Prefix sums and their applications, 1990; M. Harris, S. Sengupta and J. D. Owens, GPU Gems 3, ch. 39,
   2007; R. Nasre, M. Burtscher and K. Pingali, IPDPS 2013; Metal Shading Language Specification 4.1, §6.10.2;
   MTL4ComputeCommandEncoder.h:90-112 */
int mesh_graph_compaction_source(uint32_t items, uint32_t simds, char *text, size_t room) {
  const int made = snprintf(text, room, "#include <metal_stdlib>\n"
    "kernel void mesh_compact(device const uint *mesh_flags [[buffer(0)]],device uint *mesh_list [[buffer(1)]],"
    "device uint *mesh_arguments [[buffer(2)]],uint mesh_tl [[thread_index_in_threadgroup]],uint mesh_lanes [[threads_per_threadgroup]],"
    "uint mesh_sl [[thread_index_in_simdgroup]],uint mesh_sg [[simdgroup_index_in_threadgroup]]){"
    "threadgroup uint mesh_counts[%u];uint mesh_left=0u;"
    "for(uint mesh_at=0u;mesh_at<%uu;mesh_at+=mesh_lanes){uint mesh_g=mesh_at+mesh_tl;"
    "bool mesh_off=metal::select(false,mesh_flags[metal::min(mesh_g,%uu)]==0u,mesh_g<%uu);"
    "uint mesh_own=metal::simd_prefix_exclusive_sum(metal::select(0u,1u,mesh_off));"
    "uint mesh_simd=metal::popcount(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(mesh_off)));"
    "if(mesh_sl==0u)mesh_counts[mesh_sg]=mesh_simd;metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
    "if(mesh_sg==0u){uint mesh_count=metal::select(0u,mesh_counts[metal::min(mesh_sl,%uu)],mesh_sl<%uu);"
    "uint mesh_before=metal::simd_prefix_exclusive_sum(mesh_count);uint mesh_total=metal::simd_sum(mesh_count);"
    "if(mesh_sl<%uu)mesh_counts[mesh_sl]=mesh_before;if(mesh_sl==0u)mesh_counts[%u]=mesh_total;}"
    "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
    "if(mesh_off)mesh_list[mesh_left+mesh_counts[mesh_sg]+mesh_own]=mesh_g;mesh_left+=mesh_counts[%u];"
    "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);}"
    "if(mesh_tl==0u){mesh_arguments[0]=mesh_left;mesh_arguments[1]=1u;mesh_arguments[2]=1u;}}\n",
    simds + 1, items, items - 1, items, simds - 1, simds, simds, simds, simds);
  return made < 0 ? EINVAL : (size_t)made >= room ? ERANGE : 0;
}

/* R. Nasre, M. Burtscher and K. Pingali, IPDPS 2013; Arvind, R. S. Nikhil and K. K. Pingali, ACM TOPLAS 11(4), 1989;
   RISC-V V 1.0, the mask register v0; R. M. Russell, CACM 21(1), 1978, the vector mask register; Metal Shading Language
   Specification 4.1, §6.10.1 and §6.10.2 simd_ballot; MTL4ComputeCommandEncoder.h:90-112; MPI 4.1 §4.2.2
   MPI_PARRIVED; design/disjoint-pages-change-request.md C3.1, "Landings and cancellations both arrive on the count";
   rdma/mesh-flow.c count_raise */
int mesh_graph_worklist_source(uint32_t later, const uint32_t groups[3], const uint32_t block[3], const uint32_t *extent,
                               uint32_t simds, uint32_t operands, uint32_t classes, const uint32_t *slots,
                               const struct mesh_graph_generation *generation, const char *const builtin[5],
                               char *parameters, size_t parameters_room, char *code, size_t code_room) {
  for (uint32_t d = 0; later && d < operands; d++) if (!generation[d].every) return EINVAL;
  struct graph_sink p = {parameters, parameters_room, 0, 0}, c = {code, code_room, 0, 0};
  graph_put(&c, "threadgroup uint mesh_mask_votes[%u];uint mesh_mask_tl=%s;uint3 mesh_mask_tg=%s;", simds, builtin[0],
            builtin[1]);
  if (later) graph_put(&c, "uint mesh_mask_group=mesh_mask_list[mesh_mask_tg.x];"
    "uint3 mesh_mask_origin=uint3(mesh_mask_group%%%uu,(mesh_mask_group/%uu)%%%uu,mesh_mask_group/%uu);",
    groups[0], groups[0], groups[1], groups[0] * groups[1]);
  else graph_put(&c, "uint3 mesh_mask_origin=mesh_mask_tg;");
  if (classes > 1) graph_put(&c, "uint mesh_mask_class=mesh_mask_select[%uu+mesh_mask_origin.x];", classes * operands * 2);
  else graph_put(&c, "uint mesh_mask_class=0u;");
  if (extent) graph_put(&c, "uint3 mesh_mask_extent=metal::min(uint3(%uu,%uu,%uu),"
    "uint3(%uu,%uu,%uu)-mesh_mask_origin*uint3(%uu,%uu,%uu));", block[0], block[1], block[2], extent[0], extent[1], extent[2],
    block[0], block[1], block[2]);
  if (extent && !later) graph_put(&c, "uint mesh_mask_lanes=mesh_mask_extent.x*mesh_mask_extent.y*mesh_mask_extent.z;");
  else graph_put(&c, "uint mesh_mask_lanes=%uu;", block[0] * block[1] * block[2]);
  for (uint32_t d = 0; later && d < operands; d++)
    graph_put(&c, "ulong mesh_mask_generation_%u=(mesh_mask_repetition/%uul)*%uul+%uul;", d, generation[d].every,
              generation[d].producer, generation[d].ordinal);
  if (later) for (uint32_t d = 0; d < operands; d++) graph_put(&c,
    "uint4 mesh_mask_row_%u=mesh_mask_expect_%u[0];"
    "ulong2 mesh_mask_offset_%u=as_type<ulong2>(mesh_mask_expect_%u[1]);"
    "volatile coherent(system) device const ulong *mesh_mask_count_%u="
    "reinterpret_cast<volatile coherent(system) device const ulong *>("
    "reinterpret_cast<volatile coherent(system) device const uchar *>(mesh_mask_phases_%u)+"
    "as_type<long>(mesh_mask_offset_%u.x)-long(mesh_mask_generation_%u%%mesh_mask_row_%u.w)*long(mesh_mask_offset_%u.y));",
    d, d, d, d, d, d, d, d, d, d);
  graph_put(&c, "uint mesh_mask_vote=0u;%s", later ? "for(;;){mesh_mask_vote=0u;" : "{");
  if (later) {
    for (uint32_t d = 0; d < operands; d++)
      graph_put(&c, "bool mesh_mask_due_%u="
                    "mesh_mask_count_%u[mesh_mask_generation_%u%%mesh_mask_row_%u.w]>="
                    "(ulong(mesh_mask_generation_%u)/mesh_mask_row_%u.w+1ul)*ulong(mesh_mask_row_%u.z);", d, d, d, d, d, d, d);
    graph_put(&c,
      "metal::atomic_thread_fence(metal::mem_flags::mem_device,metal::memory_order_seq_cst,static_cast<metal::thread_scope>(3));");
  }
  for (uint32_t d = 0; d < operands; d++) {
    graph_put(&c,
    "{uint4 mesh_mask_row=mesh_mask_expect_%u[0];"
    "uint2 mesh_mask_span=uint2(mesh_mask_select[(mesh_mask_class*%uu+%uu)*2u],"
    "mesh_mask_select[(mesh_mask_class*%uu+%uu)*2u+1u]);"
    "for(uint mesh_mask_at=mesh_mask_span.x/64u+mesh_mask_tl;mesh_mask_at*64u<mesh_mask_span.y;"
    "mesh_mask_at+=mesh_mask_lanes){"
    "uint4 mesh_mask_word=mesh_mask_phases_%u[mesh_mask_at];"
    "uint4 mesh_mask_field=uint4(mesh_mask_at*64u)+uint4(0u,16u,32u,48u);"
    "uint4 mesh_mask_low=metal::select(metal::min(uint4(mesh_mask_span.x)-mesh_mask_field,uint4(16u)),uint4(0u),"
    "uint4(mesh_mask_span.x)<=mesh_mask_field);"
    "uint4 mesh_mask_high=metal::select(metal::min(uint4(mesh_mask_span.y)-mesh_mask_field,uint4(16u)),uint4(0u),"
    "uint4(mesh_mask_span.y)<=mesh_mask_field);"
    "uint4 mesh_mask_fields=(metal::select((uint4(1u)<<(metal::min(mesh_mask_high,uint4(15u))*2u))-1u,uint4(0xffffffffu),"
    "mesh_mask_high>=16u)&~metal::select((uint4(1u)<<(metal::min(mesh_mask_low,uint4(15u))*2u))-1u,uint4(0xffffffffu),"
    "mesh_mask_low>=16u))&uint4(0x55555555u);"
    "uint4 mesh_mask_unequal=mesh_mask_word^uint4(mesh_mask_row.x);"
    "mesh_mask_unequal=(mesh_mask_unequal|(mesh_mask_unequal>>1u))&mesh_mask_fields;"
    "uint4 mesh_mask_ahead=mesh_mask_word^uint4(mesh_mask_row.y);"
    "mesh_mask_ahead=(mesh_mask_ahead|(mesh_mask_ahead>>1u))&mesh_mask_unequal;"
    "mesh_mask_vote|=metal::select(0u,1u,metal::any(mesh_mask_unequal!=0u))|metal::select(0u,2u,metal::any(mesh_mask_ahead!=0u));",
    d, operands, d, operands, d, d);
    if (later) graph_put(&c, "mesh_mask_vote|=metal::select(0u,4u,mesh_mask_due_%u&&metal::any(mesh_mask_unequal!=0u));", d);
    graph_put(&c, "}}");
  }
  graph_put(&c,
    "mesh_mask_vote=metal::select(0u,1u,metal::simd_ballot((mesh_mask_vote&1u)!=0u).any())|"
    "metal::select(0u,2u,metal::simd_ballot((mesh_mask_vote&2u)!=0u).any())|"
    "metal::select(0u,4u,metal::simd_ballot((mesh_mask_vote&4u)!=0u).any());"
    "if(%s==0u)mesh_mask_votes[%s]=mesh_mask_vote;metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
    "mesh_mask_vote=0u;for(uint mesh_mask_s=0u;mesh_mask_s<%s;mesh_mask_s++)mesh_mask_vote|=mesh_mask_votes[mesh_mask_s];",
    builtin[2], builtin[3], builtin[4]);
  if (later) graph_put(&c, "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
    "if((mesh_mask_vote&1u)==0u)break;"
    "if((mesh_mask_vote&6u)!=0u){if(mesh_mask_tl==0u){"
    "metal::atomic_fetch_add_explicit(mesh_mask_status+metal::select(0u,1u,(mesh_mask_vote&2u)!=0u),1u,metal::memory_order_relaxed);"
    "metal::atomic_max_explicit(reinterpret_cast<device metal::atomic_ulong *>(mesh_mask_status+2),~mesh_mask_repetition,"
    "metal::memory_order_relaxed);}return;}}");
  else graph_put(&c, "}if(mesh_mask_tl==0u)mesh_mask_flags[mesh_mask_tg.x+%uu*(mesh_mask_tg.y+%uu*mesh_mask_tg.z)]="
    "metal::select(0u,1u,(mesh_mask_vote&1u)==0u);if((mesh_mask_vote&1u)!=0u)return;", groups[0], groups[1]);
  graph_put(&c,
    "metal::atomic_thread_fence(metal::mem_flags::mem_device,metal::memory_order_seq_cst,static_cast<metal::thread_scope>(3));");
  for (uint32_t d = 0; d < operands; d++) graph_put(&p,
    "%svolatile coherent(system) device const uint4 *mesh_mask_phases_%u [[buffer(%u)]], "
    "device const uint4 *mesh_mask_expect_%u [[buffer(%u)]]", d ? ", " : "", d, slots[2 * d], d, slots[2 * d + 1]);
  graph_put(&p, "%sdevice const uint *mesh_mask_select [[buffer(%u)]]", operands ? ", " : "", slots[2 * operands + 3]);
  if (later) {
    graph_put(&p, ", device const uint *mesh_mask_list [[buffer(%u)]], "
      "device metal::atomic_uint *mesh_mask_status [[buffer(%u)]], constant ulong &mesh_mask_repetition [[buffer(%u)]]",
      slots[2 * operands], slots[2 * operands + 1], slots[2 * operands + 2]);
  } else graph_put(&p, ", device uint *mesh_mask_flags [[buffer(%u)]]", slots[2 * operands]);
  size_t made;
  const int status = graph_end(&p, &made);
  return status ? status : graph_end(&c, &made);
}

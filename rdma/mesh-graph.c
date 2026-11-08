#include "mesh-graph.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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

int mesh_graph_tree_visit(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                          const void *context, uint32_t root,
                          int (*visit)(void *consumer, uint32_t vertex, uint32_t parent), void *consumer) {
  if (root >= vertices) return EINVAL;
  uint32_t *parent = malloc(((size_t)vertices + 1) * sizeof *parent);
  uint32_t *order = malloc(((size_t)vertices + 1) * sizeof *order);
  size_t *next = malloc(((size_t)vertices + 1) * sizeof *next);
  unsigned char *color = calloc((size_t)vertices + 1, 1);
  int status = !parent || !order || !next || !color ? ENOMEM : 0, crossed = 0;
  uint32_t count = 0;
  if (!status) parent[root] = UINT32_MAX;
  for (uint32_t v = root; !status && v != UINT32_MAX;) {
    if (!color[v]) { color[v] = 1; next[v] = first[v]; order[count++] = v; }
    if (next[v] == first[v + 1]) { color[v] = 2; v = parent[v]; continue; }
    const uint32_t w = head(context, next[v]++);
    if (color[w] == 1) status = EDEADLK;
    else if (color[w] == 2) crossed = 1;
    else { parent[w] = v; v = w; }
  }
  if (!status && crossed) status = EINVAL;
  for (uint32_t k = 0; !status && k < count; k++) status = visit(consumer, order[k], parent[order[k]]);
  free(parent); free(order); free(next); free(color);
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

static void climb_return(void *consumer, uint32_t at, uint32_t step) {
  uint32_t *found = consumer;
  if (step && at == found[0] && !found[1]) found[1] = step;
}
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from) {
  uint32_t found[2] = {from, 0};
  mesh_graph_climb(parent, vertices, UINT32_MAX, from, climb_return, found);
  return found[1];
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

void mesh_graph_tree_loads(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                           const double *weight, double *load){
  for(uint32_t k=0;k<trees;k++)
    for(uint32_t v=0;v<vertices;v++){
      const uint32_t u=parent[(size_t)k*vertices+v];
      if(v!=root[k] && u!=UINT32_MAX)load[(size_t)u*vertices+v]+=weight[k];
    }
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
  const uint32_t *distance = context;
  return distance[a] < distance[b] || (distance[a] == distance[b] && a < b);
}
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
/* E. W. Dijkstra, Numerische Mathematik 1, 1959, problem 2 */
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
                                const uint32_t *added_to, uint64_t *added_delay) {
  int result = graph_check(vertices, time, edges, from, to);
  if (!result) result = graph_check(vertices, NULL, added, added_from, added_to);
  if (result) return result;
  if (edges >= SIZE_MAX / sizeof(uint64_t) - added) return ENOMEM;
  const size_t total = edges + added;
  uint32_t *all_from = malloc((total + 1) * sizeof *all_from), *all_to = malloc((total + 1) * sizeof *all_to);
  uint64_t *all_delay = malloc((total + 1) * sizeof *all_delay);
  result = all_from && all_to && all_delay ? 0 : ENOMEM;
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
int mesh_graph_dependences(size_t accesses, struct mesh_graph_access *access, size_t room, uint32_t *from, uint32_t *to,
                           size_t *edges) {
  if (accesses) qsort(access, accesses, sizeof *access, access_before);
  size_t made = 0, held = 0, count = 0;
  uint64_t *pair = NULL;
  int result = 0;
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
int mesh_graph_synchronization(uint32_t vertices, size_t accesses, struct mesh_graph_access *access, size_t room,
                               uint32_t *from, uint32_t *to, size_t *edges, size_t *dependences) {
  for (size_t a = 0; a < accesses; a++) if (access[a].vertex >= vertices) return EINVAL;
  size_t held = accesses + 1, made = 0;
  uint32_t *tail = NULL, *head = NULL;
  int result = ERANGE;
  while (result == ERANGE) {
    free(tail); free(head);
    tail = malloc((held + 1) * sizeof *tail); head = malloc((held + 1) * sizeof *head);
    result = tail && head ? mesh_graph_dependences(accesses, access, held, tail, head, &made) : ENOMEM;
    held = made;
  }
  uint64_t *delay = result ? NULL : calloc(made + 1, sizeof *delay);
  unsigned char *synchronization = result ? NULL : malloc(made + 1), *redundant = result ? NULL : malloc(made + 1);
  if (!result && (!delay || !synchronization || !redundant)) result = ENOMEM;
  if (!result) {
    memset(synchronization, 1, made + 1);
    result = mesh_graph_redundant(vertices, made, tail, head, delay, synchronization, redundant);
  }
  size_t kept = 0;
  for (size_t e = 0; !result && e < made; e++) {
    if (redundant[e]) continue;
    if (kept < room) { from[kept] = tail[e]; to[kept] = head[e]; }
    kept++;
  }
  if (!result) { *edges = kept; *dependences = made; if (kept > room) result = ERANGE; }
  free(tail); free(head); free(delay); free(synchronization); free(redundant);
  return result;
}

/* design/torch-source-lowering-change-request.md item 3; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2 */
int mesh_graph_schedule(uint32_t vertices, size_t accesses, struct mesh_graph_access *access, const unsigned char *after,
                        int self_timed, uint32_t *level, uint32_t *place, size_t *dependences, size_t *synchronizations) {
  if (vertices == UINT32_MAX || accesses >= SIZE_MAX / 4) return ENOMEM;
  const size_t reserve = 2 * (size_t)vertices + 1;
  size_t room = accesses + reserve, edges = 0, made = 0;
  uint32_t *from = malloc((room + 1) * sizeof *from), *to = malloc((room + 1) * sizeof *to);
  uint32_t *weight = malloc((room + 1) * sizeof *weight);
  size_t *first = malloc(((size_t)vertices + 1) * sizeof *first), *order = malloc((room + 1) * sizeof *order);
  double *priority = malloc(((size_t)vertices + 1) * sizeof *priority);
  int result = from && to && weight && first && order && priority ? 0 : ENOMEM;
  while (!result) {
    result = self_timed ? mesh_graph_synchronization(vertices, accesses, access, room - reserve, from, to, &edges, &made)
                        : mesh_graph_dependences(accesses, access, room - reserve, from, to, &edges);
    if (result != ERANGE) break;
    room = edges + reserve;
    uint32_t *f = realloc(from, (room + 1) * sizeof *f), *t = f ? realloc(to, (room + 1) * sizeof *t) : NULL;
    uint32_t *w = t ? realloc(weight, (room + 1) * sizeof *w) : NULL;
    size_t *o = w ? realloc(order, (room + 1) * sizeof *o) : NULL;
    if (f) from = f;
    if (t) to = t;
    if (w) weight = w;
    if (o) order = o;
    result = o ? 0 : ENOMEM;
  }
  if (!result && !self_timed) made = edges;
  for (size_t e = 0; !result && e < edges; e++) weight[e] = 1;
  const size_t dependent = edges;
  for (uint32_t v = 1; !result && !self_timed && v < vertices; v++) {
    from[edges] = v - 1; to[edges] = v; weight[edges++] = 0;
    if (after && after[v]) { from[edges] = v - 1; to[edges] = v; weight[edges++] = 1; }
  }
  if (!result) {
    mesh_graph_rows(vertices, edges, depth_head, to, first, order);
    for (uint32_t v = 0; v < vertices; v++) {
      level[v] = 0;
      for (size_t k = first[v]; k < first[v + 1]; k++) {
        const size_t e = order[k];
        if (from[e] >= v) { result = EDEADLK; break; }
        if (level[from[e]] + weight[e] > level[v]) level[v] = level[from[e]] + weight[e];
      }
      if (result) break;
      priority[v] = -(double)level[v];
    }
  }
  if (!result && self_timed) result = mesh_graph_heap_order(vertices, priority, dependent, from, to, place);
  for (uint32_t v = 0; !result && !self_timed && v < vertices; v++) place[v] = v;
  if (!result) { *dependences = made; *synchronizations = dependent; }
  free(from); free(to); free(weight); free(first); free(order); free(priority);
  return result;
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

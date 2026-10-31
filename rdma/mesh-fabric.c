#include "mesh-plan.h"
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <stdatomic.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* design/whole-function.md#fabric */
int mesh_fabric_check(const struct mesh_fabric *f){
  if(!f || !f->nodes || (f->cables && !f->cable))return EINVAL;
  for(uint32_t e=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(!c->wire || c->node[0]>=f->nodes || c->node[1]>=f->nodes || c->node[0]==c->node[1])return EINVAL;
    for(uint32_t j=0;j<e;j++)if(f->cable[j].wire==c->wire)return EINVAL;
  }
  return 0;
}

static int mesh_cut_order(const void *a,const void *b){
  const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
  return (x>y)-(x<y);
}

int mesh_fabric_refine(const struct mesh_transfer *in,uint32_t count,uint64_t page,
                       struct mesh_transfer *out,uint32_t room){
  if((count && !in) || !page || (out && out==in))return -EINVAL;
  if(count>INT_MAX)return -EOVERFLOW;
  if((uint64_t)count*2+2>SIZE_MAX/sizeof(uint64_t))return -EOVERFLOW;
  for(uint32_t k=0;k<count;k++){
    const struct mesh_step *s=&in[k].step;
    if(s->op>MESH_STEP_COPY || !s->piece.element_bytes)return -EINVAL;
    if(s->piece.elements>UINT64_MAX-s->first ||
       s->first+s->piece.elements>UINT64_MAX/s->piece.element_bytes)return -EOVERFLOW;
  }
  uint64_t *cuts=malloc(((size_t)count*2+2)*sizeof *cuts);
  if(!cuts)return -ENOMEM;
  uint64_t written=0;
  int error=0;
  for(uint32_t k=0;k<count && !error;k++){
    if(written>=INT_MAX){error=-EOVERFLOW;break;}
    const struct mesh_step *s=&in[k].step;
    if(s->op!=MESH_STEP_SEND || !s->piece.elements){
      if(out && written<room)out[written]=in[k];
      written++;
      continue;
    }
    const uint64_t end=s->first+s->piece.elements;
    size_t n=0;
    cuts[n++]=s->first;cuts[n++]=end;
    for(uint32_t j=0;j<k;j++){
      const struct mesh_step *a=&in[j].step;
      if(a->op==MESH_STEP_SEND || a->buffer!=s->buffer || a->piece.element_bytes!=s->piece.element_bytes ||
         !a->piece.elements || a->first>=end || a->first+a->piece.elements<=s->first)continue;
      if(a->first>s->first)cuts[n++]=a->first;
      if(a->first+a->piece.elements<end)cuts[n++]=a->first+a->piece.elements;
    }
    qsort(cuts,n,sizeof *cuts,mesh_cut_order);
    const uint64_t quantum=mesh_page_quantum(page,s->piece.element_bytes);
    uint64_t first=s->first;
    const uint64_t begin=written;
    int whole=0;
    int64_t owner=-2;
    for(size_t i=0;i+1<n;i++){
      if(cuts[i]==cuts[i+1])continue;
      int64_t next=-1;
      for(uint32_t j=k;j>0;){
        const struct mesh_step *prior=&in[--j].step;
        if(prior->op==MESH_STEP_SEND || prior->buffer!=s->buffer ||
           prior->piece.element_bytes!=s->piece.element_bytes || prior->first>cuts[i] ||
           prior->first+prior->piece.elements<=cuts[i])continue;
        next=j;
        break;
      }
      if(next>=0 && next!=owner && (cuts[i]-in[next].step.first)%quantum){whole=1;break;}
      if(owner!=-2 && next!=owner){
        if((cuts[i]-s->first)%quantum){whole=1;break;}
        if(out && written<room){
          out[written]=in[k];out[written].step.first=first;
          out[written].step.piece.elements=cuts[i]-first;
        }
        written++;first=cuts[i];
      }
      owner=next;
    }
    /* The packet operand determines representable streaming cuts. An indivisible
       dependency keeps this SEND whole, without suppressing other SEND cuts. */
    if(whole){written=begin;first=s->first;}
    if(!error){
      if(out && written<room){
        out[written]=in[k];out[written].step.first=first;
        out[written].step.piece.elements=end-first;
      }
      written++;
    }
    if(written>INT_MAX)error=-EOVERFLOW;
  }
  free(cuts);
  return error?error:out && written>room?-ENOSPC:(int)written;
}

/* design/whole-function.md#one-way-writes */
int mesh_paths_check(const struct mesh_fabric *f,uint32_t source,uint32_t destination,const struct mesh_paths *paths){
  if(mesh_fabric_check(f) || source>=f->nodes || destination>=f->nodes || source==destination || !paths || !paths->count ||
     !paths->first || !paths->cable || !paths->weight || paths->first[0])return EINVAL;
  unsigned char *seen=calloc(f->nodes,1);
  if(!seen)return ENOMEM;
  int status=0;
  uint64_t total=0;
  for(uint32_t p=0;!status && p<paths->count;p++){
    memset(seen,0,f->nodes);
    uint32_t at=source;
    seen[at]=1;
    if(paths->first[p+1]<=paths->first[p] || paths->first[p+1]-paths->first[p]>=f->nodes){
      status=EINVAL;break;
    }
    if(paths->weight[p]>UINT64_MAX-total){status=EOVERFLOW;break;}
    total+=paths->weight[p];
    for(uint32_t k=paths->first[p];k<paths->first[p+1];k++){
      const uint32_t e=paths->cable[k];
      if(e>=f->cables){status=EINVAL;break;}
      const struct mesh_cable *c=f->cable+e;
      if(c->node[0]!=at && c->node[1]!=at){status=EINVAL;break;}
      at=c->node[c->node[0]==at];
      if(seen[at]){status=EINVAL;break;}
      seen[at]=1;
    }
    if(at!=destination)status=EINVAL;
  }
  free(seen);
  return status?status:total?0:EINVAL;
}

int mesh_write_compile(const struct mesh_fabric *f,uint32_t rank,uint32_t source,uint32_t destination,
                        const struct mesh_paths *paths,struct mesh_operand operand,uint64_t page,uint64_t tile,
                        struct mesh_transfer *out,uint32_t room){
  const int check=mesh_paths_check(f,source,destination,paths);
  if(check || rank>=f->nodes || !operand.element_bytes || !page)return -(check?check:EINVAL);
  if(!operand.elements)return 0;
  const uint32_t P=paths->count;
  const uint64_t q=mesh_page_quantum(page,operand.element_bytes),units=(operand.elements-1)/q+1;
  uint64_t *count=calloc(P,sizeof *count),*base=calloc(P,sizeof *base);
  uint64_t *remainder=calloc(P,sizeof *remainder);
  if(!count || !base || !remainder){free(count);free(base);free(remainder);return -ENOMEM;}
  uint64_t total=0;
  for(uint32_t p=0;p<P;p++)total+=paths->weight[p];
  uint64_t assigned=0;
  for(uint32_t p=0;p<P;p++){
    /* Weights describe total flow, not surplus after a page on every path.
       A sub-page path can round to zero; cable coverage belongs to the joint
       flow allocation, not to an extra relay page in every commodity. */
    const __uint128_t product=(__uint128_t)units*paths->weight[p];
    count[p]=(uint64_t)(product/total);remainder[p]=(uint64_t)(product%total);assigned+=count[p];
  }
  while(assigned<units){
    uint32_t best=0;
    for(uint32_t p=1;p<P;p++)if(remainder[p]>remainder[best])best=p;
    count[best]++;remainder[best]=0;assigned++;
  }
  for(uint32_t p=1;p<P;p++)base[p]=base[p-1]+count[p-1];
  uint32_t written=0;
  for(uint32_t p=0;p<P;p++){
    if(!count[p])continue;
    const uint64_t lo=base[p]*q;
    const uint64_t amount=count[p]>(operand.elements-lo)/q?operand.elements-lo:count[p]*q;
    uint32_t at=source;
    for(uint32_t k=paths->first[p];k<paths->first[p+1];k++){
      const uint32_t e=paths->cable[k],next=f->cable[e].node[f->cable[e].node[0]==at];
      if(rank==at || rank==next){
        const uint64_t fragments=tile?(lo+amount-1)/tile-lo/tile+1:1;
        if(fragments>INT_MAX-written){free(count);free(base);free(remainder);return -EOVERFLOW;}
        if(!out)written+=(uint32_t)fragments;
        else for(uint64_t first=lo,left=amount;left;){
          uint64_t n=tile?tile-first%tile:left;
          if(n>left)n=left;
          if(out && written<room)out[written]=(struct mesh_transfer){
            .step={rank==at?MESH_STEP_SEND:MESH_STEP_COPY,rank==at?next:at,
                   k-paths->first[p],0,first,{operand.type,operand.element_bytes,n}},.cable=e};
          written++;first+=n;left-=n;
        }
      }
      at=next;
    }
  }
  free(count);free(base);free(remainder);
  return out && written>room?-ENOSPC:(int)written;
}

/* Communicator-realization defaults for explicit physical writes. Every simple
   node path is represented, with parallel cables remaining distinct choices.
   Equal unit weights introduce no hop penalty. A solver can replace a
   pair's weights later through ncclMeshCommWriteRoutes; execution never calls
   this enumerator. Path order matches routing.py Fabric's cable-order DFS. */

/* The same ownership as nccl-mesh.c paths_free: three independent allocations
   per pair, and one nodes*nodes table. Empty diagonal entries own no arrays. */
void mesh_write_routes_free(struct mesh_paths *routes, uint32_t nodes) {
  if (!routes) return;
  for (size_t pair = 0; pair < (size_t)nodes * nodes; pair++) {
    free((void *)routes[pair].first);
    free((void *)routes[pair].cable);
    free((void *)routes[pair].weight);
  }
  free(routes);
}

struct mesh_write_routes_walk {
  uint32_t *node, *next, *path;
  unsigned char *seen;
};

/* Count first, then fill exactly the same paths. The explicit DFS stack keeps
   depth off the C call stack. `next` is the next cable index at each depth;
   visiting a destination ends that path rather than forwarding through it. */
static int mesh_write_routes_walk(const struct mesh_fabric *f, uint32_t source, uint32_t destination,
                                         struct mesh_write_routes_walk *walk, uint32_t *first, uint32_t *cable,
                                         uint32_t *paths, uint32_t *edges) {
  uint32_t depth = 0;
  *paths = *edges = 0;
  if (first) first[0] = 0;
  memset(walk->seen, 0, f->nodes);
  walk->node[0] = source; walk->next[0] = 0; walk->seen[source] = 1;
  for (;;) {
    const uint32_t at = walk->node[depth];
    if (at == destination) {
      if (*paths == UINT32_MAX || depth > UINT32_MAX - *edges) return EOVERFLOW;
      if (cable) memcpy(cable + *edges, walk->path, (size_t)depth * sizeof *cable);
      *edges += depth;
      (*paths)++;
      if (first) first[*paths] = *edges;
    } else {
      uint32_t other = 0, edge = 0;
      int found = 0;
      while (walk->next[depth] < f->cables) {
        edge = walk->next[depth]++;
        const struct mesh_cable *c = f->cable + edge;
        if (c->node[0] != at && c->node[1] != at) continue;
        other = c->node[c->node[0] == at];
        if (!walk->seen[other]) { found = 1; break; }
      }
      if (found) {
        walk->path[depth++] = edge;
        walk->node[depth] = other; walk->next[depth] = 0; walk->seen[other] = 1;
        continue;
      }
    }
    if (!depth) break;
    walk->seen[at] = 0;
    depth--;
  }
  return *paths ? 0 : ENOLINK;
}

/* Positive errno on failure, with *out unchanged. *out must initially be NULL:
   initialization cannot accidentally overwrite a solver's existing routes.
   There is no path-count/hop cap: only mesh_paths' uint32 representation and
   allocation limits bound enumeration. No partial table is ever published. */
int mesh_write_routes_default(const struct mesh_fabric *f, struct mesh_paths **out) {
  if (!out || *out) return EINVAL;
  int error = mesh_fabric_check(f);
  if (error) return error;
  const uint32_t n = f->nodes;
  if ((uint64_t)n * n > SIZE_MAX / sizeof(struct mesh_paths) ||
      (uint64_t)n > SIZE_MAX / sizeof(uint32_t)) return EOVERFLOW;
  struct mesh_paths *routes = calloc((size_t)n * n, sizeof *routes);
  if (!routes) return ENOMEM;
  if (n == 1) { *out = routes; return 0; }
  struct mesh_write_routes_walk walk = {
    .node = malloc((size_t)n * sizeof *walk.node),
    .next = malloc((size_t)n * sizeof *walk.next),
    .path = malloc((size_t)n * sizeof *walk.path),
    .seen = calloc(n, sizeof *walk.seen)
  };
  if (!walk.node || !walk.next || !walk.path || !walk.seen) { error = ENOMEM; goto done; }
  /* Reject disconnected fabrics before enumerating potentially large connected
     components. This is connectivity validation, not a subset of allowed paths. */
  uint32_t reached = 1;
  walk.node[0] = 0; walk.seen[0] = 1;
  for (uint32_t k = 0; k < reached; k++) for (uint32_t e = 0; e < f->cables; e++) {
    const struct mesh_cable *c = f->cable + e;
    if (c->node[0] != walk.node[k] && c->node[1] != walk.node[k]) continue;
    const uint32_t other = c->node[c->node[0] == walk.node[k]];
    if (!walk.seen[other]) { walk.seen[other] = 1; walk.node[reached++] = other; }
  }
  if (reached != n) { error = ENOLINK; goto done; }
  for (uint32_t source = 0; source < n; source++) for (uint32_t destination = 0; destination < n; destination++) {
    if (source == destination) continue;
    uint32_t paths, edges;
    error = mesh_write_routes_walk(f, source, destination, &walk, NULL, NULL, &paths, &edges);
    if (error) goto done;
    if ((uint64_t)paths + 1 > SIZE_MAX / sizeof(uint32_t) ||
        (uint64_t)edges > SIZE_MAX / sizeof(uint32_t) ||
        (uint64_t)paths > SIZE_MAX / sizeof(uint64_t)) { error = EOVERFLOW; goto done; }
    struct mesh_paths *pair = routes + (size_t)source * n + destination;
    uint32_t *first = malloc(((size_t)paths + 1) * sizeof *first);
    uint32_t *cable = malloc((size_t)edges * sizeof *cable);
    uint64_t *weights = calloc(paths, sizeof *weights);
    *pair = (struct mesh_paths){paths, first, cable, weights};
    if (!first || !cable || !weights) { error = ENOMEM; goto done; }
    for (uint32_t p = 0; p < paths; p++) weights[p] = 1;
    error = mesh_write_routes_walk(f, source, destination, &walk, first, cable, &paths, &edges);
    if (error) goto done;
  }
done:
  free(walk.node); free(walk.next); free(walk.path); free(walk.seen);
  if (error) mesh_write_routes_free(routes, n);
  else *out = routes;
  return error;
}

/* A tree program already allocates traffic and relay work to graph edges.
   Realize those edges on all their physical cables, rather than routing each
   edge through a second independent set of relays. Explicit pair routes are
   authoritative, including nonadjacent endpoints and arbitrary relay paths.
   This is configuration for mesh_write_compile, not another transport path. */
int mesh_write_routes_compose(const struct mesh_fabric *f,
    const struct mesh_paths *routes, const unsigned char *overrides, struct mesh_paths **out) {
  if (!routes || !out || *out) return EINVAL;
  int checked = mesh_fabric_check(f);
  if (checked) return checked;
  if ((uint64_t)f->nodes * f->nodes > SIZE_MAX / sizeof(struct mesh_paths)) return EOVERFLOW;
  const size_t pairs = (size_t)f->nodes * f->nodes;
  struct mesh_paths *made = calloc(pairs, sizeof *made);
  if (!made) return ENOMEM;
  int status = 0;
  for (uint32_t a = 0; a < f->nodes && !status; a++) for (uint32_t b = 0; b < f->nodes && !status; b++) {
    if (a == b) continue;
    const size_t pair = (size_t)a * f->nodes + b;
    uint32_t count = 0;
    if (!overrides || !overrides[pair]) for (uint32_t e = 0; e < f->cables; e++) {
      const struct mesh_cable *c = f->cable + e;
      count += (c->node[0] == a && c->node[1] == b) || (c->node[0] == b && c->node[1] == a);
    }
    const struct mesh_paths *given = routes + pair;
    if (!count && (status = mesh_paths_check(f, a, b, given))) break;
    const uint32_t paths = count ? count : given->count, edges = count ? count : given->first[given->count];
    uint32_t *first = malloc(((size_t)paths + 1) * sizeof *first), *cable = malloc((size_t)edges * sizeof *cable);
    uint64_t *weight = malloc((size_t)paths * sizeof *weight);
    made[pair] = (struct mesh_paths){paths, first, cable, weight};
    if (!first || !cable || !weight) { status = ENOMEM; break; }
    if (!count) {
      memcpy(first, given->first, ((size_t)paths + 1) * sizeof *first);
      memcpy(cable, given->cable, (size_t)edges * sizeof *cable);
      memcpy(weight, given->weight, (size_t)paths * sizeof *weight);
    } else {
      uint32_t k = 0;
      for (uint32_t e = 0; e < f->cables; e++) {
        const struct mesh_cable *c = f->cable + e;
        if (!((c->node[0] == a && c->node[1] == b) || (c->node[0] == b && c->node[1] == a))) continue;
        first[k] = k; cable[k] = e; weight[k++] = c->bandwidth ? c->bandwidth : 1;
      }
      first[count] = count;
    }
  }
  if (status) mesh_write_routes_free(made, f->nodes); else *out = made;
  return status;
}

/* the node's GPU cores (IORegistry AGXAccelerator gpu-core-count), read once a process: the one reading, which the
   tracks pool's workers (nccl-mesh-metal.m), the node's facts mesh-stat publishes and torch_mesh's compute capability
   (mesh_compute.py) all take; 0 where the registry states none */
uint32_t mesh_gpu_cores(void) {
  static _Atomic uint32_t read;
  uint32_t cores = atomic_load_explicit(&read, memory_order_relaxed);
  if (cores) return cores;
  io_service_t gpu = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
  CFTypeRef counted = gpu ? IORegistryEntryCreateCFProperty(gpu, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0) : NULL;
  long long value = 0;
  if (counted && CFGetTypeID(counted) == CFNumberGetTypeID()) CFNumberGetValue(counted, kCFNumberLongLongType, &value);
  if (counted) CFRelease(counted);
  if (gpu) IOObjectRelease(gpu);
  cores = value > 0 && value <= UINT32_MAX ? (uint32_t)value : 0;
  atomic_store_explicit(&read, cores, memory_order_relaxed);
  return cores;
}

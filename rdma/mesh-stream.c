#include "mesh-stream.h"
#include <errno.h>
#include <string.h>
/* design/pages-and-functions.md#streaming-tiles */

uint32_t mesh_stream_tile(uint32_t block_pages,uint32_t page_bytes,uint32_t columns,uint32_t element_bytes,uint32_t granularity){
  if(block_pages<2 || !page_bytes || !columns || !element_bytes || !granularity) return 0;
  uint64_t rows=(uint64_t)(block_pages-1)*page_bytes/((uint64_t)columns*element_bytes);
  return (uint32_t)(rows/granularity*granularity);
}

uint32_t mesh_stream_spans(uint32_t rows,uint32_t tile,double rho,uint32_t *spans,uint32_t capacity){
  if(!rows || !tile || rows%tile || !(rho>0)) return 0;
  uint32_t count=0,remaining=rows,span=tile;
  while(remaining){
    if(count==capacity) return 0;
    if(span>remaining) span=remaining;
    spans[count++]=span; remaining-=span;
    uint32_t next=tile*(uint32_t)(rho*span/tile);
    span=next<tile?tile:next;
  }
  return count;
}

#define INF (1.0/0.0)
static int owner_of(const struct mesh_stream *s,uint32_t t){ return t>=s->owner_tiles[0]; }
static _Thread_local uint32_t launch_sizes[2][MESH_STREAM_TILES],launch_count[2];
uint32_t mesh_stream_launch_tiles(int node,uint32_t i){ return i<launch_count[node]?launch_sizes[node][i]:0; }

/* rho: link rows per second over the slowest producer's rows per second, per tile. */
static double stream_rho(const struct mesh_stream *s){
  double slowest=s->node[0].produce_ns>s->node[1].produce_ns?s->node[0].produce_ns:s->node[1].produce_ns;
  return slowest/s->transfer_ns;
}

/* Blocks a node holds landed and unconsumed at `now`: partials until their span's reduce completes, returns until the invocation ends if held. */
static uint32_t landed_now(const struct mesh_stream *s,int n,double now,const double *land,const double *red,const double *gland){
  uint32_t count=0;
  for(uint32_t t=0;t<s->tiles;t++){
    if(owner_of(s,t)==n){ if(land[t]<=now && !(red[t]<=now)) count++; }
    else if(s->gathers_held && gland[t]<=now) count++;
  }
  return count;
}

/* Launches of node n: the order's maximal runs inside one owner region, each partitioned into spans; a peer-owned run ascends so the wire
   fills from the first tile, an own run descends so the last reduce and its returns are one tile. launch[i] = first index in the order, size[i] tiles. */
static uint32_t partition(const struct mesh_stream *s,int node,const uint32_t *order,uint32_t *first,uint32_t *size){
  uint32_t T=s->tiles,count=0,spans[MESH_STREAM_TILES]; double rho=stream_rho(s);
  for(uint32_t i=0;i<T;){
    uint32_t k=1; while(i+k<T && order[i+k]==order[i]+k && owner_of(s,order[i+k])==owner_of(s,order[i])) k++;
    uint32_t n=mesh_stream_spans(k,1,rho,spans,MESH_STREAM_TILES);
    if(!n) return 0;
    int own=owner_of(s,order[i])==node;
    for(uint32_t j=0,at=i;j<n;j++){ uint32_t span=spans[own?n-1-j:j]; first[count]=at; size[count++]=span; at+=span; }
    i+=k;
  }
  return count;
}

int mesh_stream_simulate(const struct mesh_stream *s,const uint32_t order[2][MESH_STREAM_TILES],struct mesh_stream_plan *out){
  uint32_t T=s->tiles;
  double prod[2][MESH_STREAM_TILES],input[2][MESH_STREAM_TILES],land[MESH_STREAM_TILES],red[MESH_STREAM_TILES],gland[MESH_STREAM_TILES];
  uint32_t blk[2][MESH_STREAM_TILES]; double ready[2][MESH_STREAM_TILES];
  uint32_t first[2][MESH_STREAM_TILES],size[2][MESH_STREAM_TILES],launches[2],next_launch[2]={0,0};
  uint32_t count[2]={0,0},head[2]={0,0},peak[2]={0,0};
  int in_flight[2]={-1,-1};
  double land_at[2]={0,0},wire_free[2]={0,0},gpu_free[2]={0,0},busy[2]={0,0},last_busy[2]={0,0},first_launch[2]={INF,INF};
  double parts[2][4]={{0,0,0,0},{0,0,0,0}};
  for(uint32_t t=0;t<T;t++){ prod[0][t]=prod[1][t]=land[t]=red[t]=gland[t]=INF; }
  for(int n=0;n<2;n++){
    launches[n]=partition(s,n,order[n],first[n],size[n]); if(!launches[n]) return EINVAL;
    launch_count[n]=launches[n]; for(uint32_t i=0;i<launches[n];i++) launch_sizes[n][i]=size[n][i];
    for(uint32_t i=0;i<T;i++) input[n][order[n][i]]=s->input_ns*i;
  }
  double now=0;
  for(;;){
    int progressed=0;
    for(int n=0;n<2;n++){
      if(gpu_free[n]>now) continue;
      const struct mesh_stream_node *d=&s->node[n];
      /* A ready reduce first: an own-region span whose own partial is present and whose peer partial has wholly landed. */
      int best=-1; double best_at=INF; uint32_t best_k=0;
      for(uint32_t i=0;i<launches[n];i++){
        uint32_t t0=order[n][first[n][i]],k=size[n][i];
        if(owner_of(s,t0)!=n || red[t0]<INF) continue;
        double at=0; for(uint32_t x=0;x<k;x++){ uint32_t t=order[n][first[n][i]+x]; double a=prod[n][t]>land[t]?prod[n][t]:land[t]; if(a>at) at=a; }
        if(at<=now && at<best_at){ best_at=at; best=(int)i; best_k=k; }
      }
      if(best>=0){
        double cost=d->overhead_ns+best_k*d->reduce_ns;
        gpu_free[n]=now+cost; busy[n]+=cost; last_busy[n]=gpu_free[n];
        for(uint32_t x=0;x<best_k;x++){ uint32_t t=order[n][first[n][best]+x]; red[t]=gpu_free[n]; blk[n][count[n]]=t|0x80000000u; ready[n][count[n]++]=gpu_free[n]; }
        progressed=1; continue;
      }
      if(next_launch[n]<launches[n]){
        uint32_t i=next_launch[n],k=size[n][i]; int present=1;
        for(uint32_t x=0;x<k && present;x++) present=input[n][order[n][first[n][i]+x]]<=now;
        if(present){
          double cost=d->overhead_ns+k*d->produce_ns;
          if(first_launch[n]==INF) first_launch[n]=now;
          gpu_free[n]=now+cost; busy[n]+=cost; last_busy[n]=gpu_free[n]; next_launch[n]++;
          for(uint32_t x=0;x<k;x++){ uint32_t t=order[n][first[n][i]+x]; prod[n][t]=gpu_free[n]; if(owner_of(s,t)!=n){ blk[n][count[n]]=t; ready[n][count[n]++]=gpu_free[n]; } }
          progressed=1;
        }
      }
    }
    for(int d=0;d<2;d++){
      int peer=1-d;
      if(in_flight[d]>=0 && land_at[d]<=now){
        uint32_t b=blk[d][in_flight[d]],t=b&0x7fffffffu;
        if(b&0x80000000u) gland[t]=now; else land[t]=now;
        in_flight[d]=-1; wire_free[d]=now;
        uint32_t held=landed_now(s,peer,now,land,red,gland); if(held>peak[peer]) peak[peer]=held;
        progressed=1;
      }
      if(in_flight[d]<0 && head[d]<count[d]){
        double at=ready[d][head[d]]>wire_free[d]?ready[d][head[d]]:wire_free[d];
        if(at<=now && landed_now(s,peer,now,land,red,gland)<s->window){
          in_flight[d]=(int)head[d]++; land_at[d]=now+s->transfer_ns; progressed=1;
        }
      }
    }
    if(progressed) continue;
    double next=INF;
    for(int n=0;n<2;n++){
      if(gpu_free[n]>now && gpu_free[n]<next) next=gpu_free[n];
      if(next_launch[n]<launches[n]) for(uint32_t x=0;x<size[n][next_launch[n]];x++){ double at=input[n][order[n][first[n][next_launch[n]]+x]]; if(at>now && at<next) next=at; }
    }
    for(int d=0;d<2;d++){
      if(in_flight[d]>=0){ if(land_at[d]<next) next=land_at[d]; }
      else if(head[d]<count[d]){ double at=ready[d][head[d]]>wire_free[d]?ready[d][head[d]]:wire_free[d]; if(at>now && at<next) next=at; }
    }
    if(next==INF) break;
    /* Idle nodes over [now,next): fill before the first launch; window when a block bound here is ready but has no credit; else the peer. */
    for(int n=0;n<2;n++){
      if(gpu_free[n]>now) continue;
      int d=1-n, blocked=in_flight[d]<0 && head[d]<count[d] && ready[d][head[d]]<=now && landed_now(s,n,now,land,red,gland)>=s->window;
      parts[n][first_launch[n]==INF?0:blocked?1:2]+=next-now;
    }
    now=next;
  }
  double makespan=0;
  for(uint32_t t=0;t<T;t++){
    if(!(red[t]<INF) || !(gland[t]<INF)) return EDEADLK;
    if(red[t]>makespan) makespan=red[t];
    if(gland[t]>makespan) makespan=gland[t];
  }
  out->makespan_ns=makespan;
  for(int n=0;n<2;n++){
    out->launches[n]=launches[n]; out->peak_landed[n]=peak[n]; out->idle[n]=makespan>0?1-busy[n]/makespan:0;
    /* Idle after the last launch is drain, whatever the simulation attributed it to in flight. */
    double drain=makespan-last_busy[n]; parts[n][3]=drain;
    double tail=0; for(int p=0;p<3;p++) tail+=parts[n][p];
    if(tail>makespan-busy[n]-drain){ double scale=(makespan-busy[n]-drain)/tail; for(int p=0;p<3;p++) parts[n][p]*=scale>0?scale:0; }
    for(int p=0;p<4;p++) out->idle_parts[n][p]=parts[n][p];
  }
  return 0;
}

/* Production order: the peer-owned region (spans ascending), then the own region (spans descending); tiles ascend within each. */
static void canonical(const struct mesh_stream *s,int n,uint32_t *order){
  uint32_t count=0;
  for(uint32_t t=0;t<s->tiles;t++) if(owner_of(s,t)!=n) order[count++]=t;
  for(uint32_t t=0;t<s->tiles;t++) if(owner_of(s,t)==n) order[count++]=t;
}

int mesh_stream_plan(const struct mesh_stream *s,struct mesh_stream_plan *plan){
  if(!s->tiles || s->tiles>MESH_STREAM_TILES || !s->window || !(s->transfer_ns>0) || !(s->input_ns>=0)) return EINVAL;
  if(s->owner_tiles[0]+s->owner_tiles[1]!=s->tiles) return EINVAL;
  for(int n=0;n<2;n++){
    const struct mesh_stream_node *d=&s->node[n];
    if(!d->launch_tiles_min || d->launch_tiles_min>s->tiles || !(d->produce_ns>0) || !(d->reduce_ns>0) || !(d->overhead_ns>=0)) return EINVAL;
  }
  memset(plan,0,sizeof *plan);
  for(int n=0;n<2;n++) canonical(s,n,plan->order[n]);
  int e=mesh_stream_simulate(s,(const uint32_t(*)[MESH_STREAM_TILES])plan->order,plan);
  if(e) return e==EDEADLK?ENOSPC:e;
  return plan->peak_landed[0]>s->window || plan->peak_landed[1]>s->window?ENOSPC:0;
}

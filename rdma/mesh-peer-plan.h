#ifndef MESH_PEER_PLAN_H
#define MESH_PEER_PLAN_H
#include "mesh-peer-cables.h"
#include "mesh-peer.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

struct peer_offer { uint32_t endpoints,records,units,pad; };
struct peer_record { uint64_t match,bytes,gate,threshold;uint32_t first,units,source,pad; };
struct peer_layout {
  struct peer_offer head;
  struct mesh_endpoint *links;
  struct peer_record *row;
  struct mesh_dma_span *spans;
};
static inline int peer_layout_size(struct peer_offer h,size_t *size){
  uint64_t n=sizeof h+(uint64_t)h.endpoints*sizeof(struct mesh_endpoint)+(uint64_t)h.records*sizeof(struct peer_record)+
    (uint64_t)h.units*sizeof(struct mesh_dma_span);
  if(h.endpoints>MESH_STREAMS || !h.records || n>UINT32_MAX)return EINVAL;
  *size=(size_t)n;return 0;
}
static inline int peer_layout(void *data,size_t bytes,struct peer_layout *l){
  if(bytes<sizeof(struct peer_offer))return EPROTO;
  memcpy(&l->head,data,sizeof l->head);
  size_t n=0;
  if(peer_layout_size(l->head,&n) || n!=bytes)return EPROTO;
  l->links=(void *)((char *)data+sizeof l->head);
  l->row=(void *)(l->links+l->head.endpoints);
  l->spans=(void *)(l->row+l->head.records);
  return 0;
}
struct peer_piece { struct mesh_dma_span span;uint64_t prefix;uint32_t channel; };
struct peer_row {
  struct mesh_peer_transfer logical;
  size_t first,end,next;
  size_t retired;uint32_t remote_open,announced,issued,done,published,prev,succ;
};
struct peer_plan {
  struct mesh_cable cable[MESH_STREAMS]; uint32_t links,dynamic[2];
  struct peer_row *row;uint32_t rows;
  struct peer_piece *piece;size_t count,room;
};
static inline int peer_wire_sort(const void *a,const void *b){
  const struct mesh_cable *x=a,*y=b;return (x->wire>y->wire)-(x->wire<y->wire);
}
static inline int peer_record_order(const void *a,const void *b){
  const struct peer_record *x=a,*y=b;
  if(x->source!=y->source)return (x->source>y->source)-(x->source<y->source);
  return (x->match>y->match)-(x->match<y->match);
}
static inline int peer_cut(struct peer_plan *p,uint32_t channel,uint64_t offset,uint64_t bytes){
  if(p->count==p->room){
    if(p->room>(SIZE_MAX/2)/sizeof *p->piece)return EOVERFLOW;
    size_t cap=p->room?2*p->room:16;void *next=realloc(p->piece,cap*sizeof *p->piece);
    if(!next)return ENOMEM;
    p->piece=next;p->room=cap;
  }
  p->piece[p->count++]=(struct peer_piece){.channel=channel,.span={offset,bytes}};
  return 0;
}
static inline int peer_plan_compile(struct peer_plan *p,uint32_t node,uint32_t peer,
    struct peer_layout a,struct peer_layout b,const struct mesh_peer_transfer *records,uint64_t data_off,uint32_t packet){
  if(!packet || a.head.records!=b.head.records)return EPROTO;
  uint32_t rank=node>peer;
  struct peer_layout x=rank?b:a,y=rank?a:b;
  int status=mesh_peer_cables(0,rank?peer:node,x.links,x.head.endpoints,1,rank?node:peer,y.links,y.head.endpoints,p->cable,&p->links);
  if(status || !p->links)return status?status:ENOLINK;
  qsort(p->cable,p->links,sizeof *p->cable,peer_wire_sort);
  qsort(a.row,a.head.records,sizeof *a.row,peer_record_order);
  qsort(b.row,b.head.records,sizeof *b.row,peer_record_order);
  struct mesh_fabric fabric={2,p->links,p->cable};
  p->rows=a.head.records;p->row=calloc(p->rows,sizeof *p->row);
  if(!p->row)return ENOMEM;
  long double load[2][MESH_STREAMS]={0};
  struct peer_record prior[2][2][MESH_STREAMS]={0};
  for(uint32_t k=0;k<p->rows;k++){
    struct peer_record u=a.row[k],v=b.row[k];
    if(peer_record_order(&u,&v) || (k && !peer_record_order(a.row+k-1,&u)) ||
       (u.source!=node && u.source!=peer) || u.bytes!=v.bytes || !u.units || !v.units ||
       u.first>a.head.units || u.units>a.head.units-u.first || v.first>b.head.units || v.units>b.head.units-v.first ||
       u.pad>=a.head.records)return EPROTO;
    struct peer_row *r=p->row+k;
    r->logical=records[u.pad];r->first=r->next=p->count;
    unsigned way=r->logical.land;
    struct mesh_transfer stripes[MESH_STREAMS]={0};
    int n;
    if(u.bytes<=packet){
      uint32_t best=0;
      for(uint32_t e=1;e<p->links;e++)
        if((load[way][e]+u.bytes)/p->cable[e].bandwidth < (load[way][best]+u.bytes)/p->cable[best].bandwidth)best=e;
      stripes[0]=(struct mesh_transfer){.step.piece={1,1,u.bytes},.cable=best};n=1;
    }else{
      struct mesh_step step={.op=MESH_STEP_SEND,.peer=1,.piece={1,1,u.bytes}};
      n=mesh_fabric_stripe(&fabric,0,&step,1,NULL,stripes,p->links);
      if(n<=0)return n?-n:EPROTO;
    }
    r->prev=r->succ=UINT32_MAX;
    uint32_t ai=0,bi=0;uint64_t ax=0,bx=0,logical=0;
    for(int i=0;i<n;i++){
      uint32_t e=stripes[i].cable;uint64_t left=stripes[i].step.piece.elements;
      if(stripes[i].step.first!=logical || !p->cable[e].bandwidth)return EPROTO;
      load[way][e]+=left;
      struct peer_record gates[]={u,v};
      for(uint32_t side=0;side<2;side++){
        struct peer_record *last=&prior[side][way][e],gate=gates[side];
        if(last->units && last->gate &&
          (gate.gate!=last->gate || gate.threshold<last->threshold))p->dynamic[way]=1;
        *last=gate;
      }
      while(left){
        if(ai>=u.units || bi>=v.units)return EPROTO;
        struct mesh_dma_span as=a.spans[u.first+ai],bs=b.spans[v.first+bi];
        if(as.bytes<=ax || bs.bytes<=bx || as.offset>UINT64_MAX-ax || as.offset+ax<data_off)return EPROTO;
        uint64_t at=as.offset+ax,length=left;
        uint64_t bounds[]={as.bytes-ax,bs.bytes-bx,packet,MESH_DMA_EXTENT-(at-data_off)%MESH_DMA_EXTENT,
                           MESH_DMA_EXTENT-(bs.offset%MESH_DMA_EXTENT+bx%MESH_DMA_EXTENT)%MESH_DMA_EXTENT};
        for(size_t t=0;t<sizeof bounds/sizeof *bounds;t++)if(length>bounds[t])length=bounds[t];
        status=peer_cut(p,e,at,length);if(status)return status;
        ax+=length;bx+=length;logical+=length;left-=length;
        if(ax==as.bytes){ax=0;ai++;}if(bx==bs.bytes){bx=0;bi++;}
      }
    }
    if(logical!=u.bytes || ai!=u.units || bi!=v.units)return EPROTO;
    r->remote_open=!v.gate;r->end=p->count;
  }
  return 0;
}
#endif

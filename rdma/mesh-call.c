#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* ---- The lane pairs' bindings (mesh-call.h; design/transport.md section 14) ------------------------- */
int mesh_lanes_free(struct hdr *m,struct mesh_lane_report *report){
  if(!m || !report)return EINVAL;
  *report=(struct mesh_lane_report){.node=m->node};
  for(uint32_t p=0;p<m->links && p<MESH_PORTS;p++){
    const struct mesh_link_info *link=mesh_links(m)+p;
    const uint32_t peer=atomic_load_explicit(&link->peer,memory_order_relaxed);
    if(!peer || atomic_load_explicit(&link->lanes,memory_order_acquire)!=MESH_LINK_LANES)continue;
    const uint32_t free=(uint32_t)atomic_load_explicit(&m->free[p],memory_order_acquire)&((UINT32_C(1)<<MESH_LANE_PAIRS)-1);
    report->link[report->count++]=(struct mesh_lane_link){.peer=peer,.port=p,.free=free};
  }
  return 0;
}
int mesh_lanes_choose(const struct mesh_lane_report *reports,uint32_t ranks,uint32_t *chosen){
  for(uint32_t r=0;r<ranks;r++){
    if(reports[r].count>MESH_PORTS)return EINVAL;
    for(uint32_t k=0;k<MESH_PORTS;k++){
      chosen[(size_t)r*MESH_PORTS+k]=UINT32_MAX;
      if(k<reports[r].count && reports[r].link[k].port>=MESH_PORTS)return EINVAL;
    }
  }
  int status=0;
  for(uint32_t r=0;r<ranks;r++){
    const struct mesh_lane_report *own=reports+r;
    uint32_t *row=chosen+(size_t)r*MESH_PORTS;
    for(uint32_t k=0;k<own->count;k++){
      const uint32_t node=own->node,peer=own->link[k].peer;
      if(row[k]!=UINT32_MAX || peer==node)continue;
      uint32_t free=(UINT32_C(1)<<MESH_LANE_PAIRS)-1,occupied=0,heard=0,ports=0;
      for(uint32_t q=0;q<ranks;q++){
        occupied|=reports[q].node==peer;
        for(uint32_t j=0;j<reports[q].count;j++){
          const struct mesh_lane_link *l=reports[q].link+j;
          if(reports[q].node==node && l->peer==peer)free&=l->free;
          if(reports[q].node==peer && l->peer==node){free&=l->free;heard=1;}
        }
      }
      if(!occupied)continue;
      for(uint32_t j=0;j<own->count;j++)if(own->link[j].peer==peer)ports|=UINT32_C(1)<<own->link[j].port;
      for(uint32_t q=0;q<r;q++)if(reports[q].node==node)for(uint32_t j=0;j<reports[q].count;j++){
        const uint32_t taken=chosen[(size_t)q*MESH_PORTS+j];
        if(ports>>reports[q].link[j].port&1 && taken!=UINT32_MAX)free&=~(UINT32_C(1)<<taken);
      }
      if(!heard || !free){status=ENOSPC;continue;}
      for(uint32_t j=0;j<own->count;j++)if(own->link[j].peer==peer)row[j]=(uint32_t)__builtin_ctz(free);
    }
  }
  return status;
}
static struct mesh_bind *bind_entry(struct mesh_ctx *context,uint32_t port,uint32_t pair){
  return mesh_client_at(context->M,context->client)->bind[port]+pair;
}
void mesh_lanes_unbind(struct mesh_ctx *context,const struct mesh_lane_report *mine,const uint32_t *chosen){
  if(!context || !context->client)return;
  for(uint32_t k=0;k<mine->count && k<MESH_PORTS;k++)if(chosen[k]<MESH_LANE_PAIRS && mine->link[k].port<MESH_PORTS)
    atomic_store_explicit(&bind_entry(context,mine->link[k].port,chosen[k])->want,0,memory_order_release);
  mesh_notify(&context->M->control);
}
int mesh_lanes_bind(struct mesh_ctx *context,const struct mesh_lane_report *mine,const uint32_t *chosen,uint64_t (*start)[2]){
  if(!context || !context->client || mine->count>MESH_PORTS)return EINVAL;
  for(uint32_t k=0;k<mine->count;k++)
    if(chosen[k]!=UINT32_MAX && (chosen[k]>=MESH_LANE_PAIRS || mine->link[k].port>=MESH_PORTS))return EINVAL;
  uint64_t serial[MESH_PORTS]={0};
  for(uint32_t k=0;k<mine->count;k++)if(chosen[k]!=UINT32_MAX){
    struct mesh_bind *b=bind_entry(context,mine->link[k].port,chosen[k]);
    const uint64_t answered=mesh_bind_serial(atomic_load_explicit(&b->answer,memory_order_acquire)),
      wanted=atomic_load_explicit(&b->want,memory_order_relaxed);
    serial[k]=(answered>wanted?answered:wanted)+1;
    atomic_store_explicit(&b->declared[0],0,memory_order_relaxed);atomic_store_explicit(&b->declared[1],0,memory_order_relaxed);
    atomic_store_explicit(&b->want,serial[k],memory_order_release);
  }
  mesh_notify(&context->M->control);
  const uint64_t until=clock_gettime_nsec_np(CLOCK_UPTIME_RAW)+MESH_PLATFORM_NS;
  int status=0;
  for(uint32_t k=0;k<mine->count && !status;k++)if(chosen[k]!=UINT32_MAX){
    struct mesh_bind *b=bind_entry(context,mine->link[k].port,chosen[k]);
    uint64_t answer;
    while(mesh_bind_serial(answer=atomic_load_explicit(&b->answer,memory_order_acquire))!=serial[k] &&
          clock_gettime_nsec_np(CLOCK_UPTIME_RAW)<until)usleep(20);
    if(answer==serial[k]){
      start[k][0]=atomic_load_explicit(&b->start[0],memory_order_relaxed);
      start[k][1]=atomic_load_explicit(&b->start[1],memory_order_relaxed);
    }
    else status=mesh_bind_serial(answer)==serial[k]?(int)(answer&0xffff):ETIMEDOUT;
  }
  if(status)mesh_lanes_unbind(context,mine,chosen);
  return status;
}
/* ---- A bound lane's rings as its client maps them (mesh-call.h; design/transport.md section 14) -------- */
int mesh_lane_view(struct mesh_ctx *context,uint32_t port,uint32_t lane,struct mesh_lane_view *view){
  if(!context || !context->client || port>=MESH_PORTS || lane<MESH_LANE_FIRST || lane>=MESH_LINK_LANES || !view)return EINVAL;
  const struct mesh_bind *b=bind_entry(context,port,(lane-MESH_LANE_FIRST)/2);
  const uint64_t want=atomic_load_explicit(&b->want,memory_order_relaxed);
  if(!want || atomic_load_explicit(&b->answer,memory_order_acquire)!=want)return EPERM;
  struct hdr *m=context->M;
  const struct mesh_lane *x=mesh_lane_at(m,port,lane);
  if(!x->units)return ENOLCK;
  *view=(struct mesh_lane_view){.in=(unsigned char *)m+x->in,.out=(unsigned char *)m+x->out,.ready=(_Atomic uint64_t *)((char *)m+x->ready),
    .landed=(_Atomic uint64_t *)((char *)m+x->landed),.unit=x->unit,.units=x->units,.port=port,.lane=lane};
  return 0;
}
int mesh_lane_declare(struct mesh_ctx *context,uint32_t port,uint32_t lane,uint64_t declared,uint64_t deadline){
  if(!context || !context->client || port>=MESH_PORTS || lane<MESH_LANE_FIRST || lane>=MESH_LINK_LANES)return EINVAL;
  struct mesh_bind *b=bind_entry(context,port,(lane-MESH_LANE_FIRST)/2);
  const uint32_t j=(lane-MESH_LANE_FIRST)&1;
  atomic_store_explicit(&b->deadline,deadline,memory_order_relaxed);
  atomic_store_explicit(&b->declared[j],declared,memory_order_release);
  mesh_notify(&context->M->bell[port]);
  return 0;
}
void mesh_lane_wake(struct hdr *m,uint32_t port){ if(port<MESH_PORTS)mesh_notify(&m->bell[port]); }

/* design/algorithm-sources.md#programtensor: a lease's rows and arena pages, the pages anywhere the window has them */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz,capacity=mesh_arena_pages(m);
  if(bytes>capacity*(size_t)m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>capacity/count)return ENOMEM;
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    buffer->pages=(uint32_t)span;
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(mesh_plane(m,MESH_ROW_HOT),first,rows);
  /* design/transport.md */
  /* design/transport.md */
  uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block);
  if(page==MESH_ABSENT){
    int error=errno;
    mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),first,rows);
    for(uint32_t r=first;r<first+rows;r++){mesh_buffers(m)[r].pages=0;atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_release);}
    mesh_rows_release(context,first,rows);return error;
  }
  for(uint32_t slot=0;slot<count;slot++)
    mesh_backing_bind(context,first+slot*stride,(uint32_t)span,page+slot*(uint32_t)span,slot);
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride};
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return (char *)context->M+atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)].address,memory_order_relaxed);}


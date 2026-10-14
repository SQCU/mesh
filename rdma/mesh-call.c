#include "mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/prepared-machine.md#M09 */
/* A bound row is an SGE target, so it must lie in the registered window.  Refusing it here is the
   whole enforcement of the split: an operand allocated out of the unregistered arena fails at bind
   with EINVAL instead of reaching the wire with a lkey that does not cover it. */
static uint64_t mesh_section_rows(struct mesh_section section){
  return section.stride?(uint64_t)section.stride*section.count:1;
}
static int mesh_section_wired(struct hdr *m,struct mesh_section section){
  uint64_t block=(uint64_t)m->block*m->pgsz,rows=mesh_section_rows(section);
  for(uint64_t i=0;i<rows;i++){
    struct mesh_page_entry *entry=&mesh_page(m)[section.first+i];
    if(atomic_load_explicit(&entry->mapping,memory_order_relaxed)==MESH_ABSENT)continue;
    uint64_t offset=atomic_load_explicit(&entry->address,memory_order_relaxed)-m->data_off;
    if(!mesh_wired(m,offset,block-offset%block))return 0;
  }
  return 1;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *context,uint32_t queue,int receive,uint32_t identity,struct mesh_section section,uint32_t invocation_pages,
  uint32_t begin,uint32_t end,uint32_t depth){
  struct hdr *m=context->M;
  if(queue==MESH_ABSENT)return EHOSTUNREACH;
  if(queue>=m->links*MESH_QUEUES || end<=begin)return EINVAL;
  if(!section.count || (uint64_t)section.first+mesh_section_rows(section)>mesh_rows(m))return EINVAL;
  /* design/prepared-machine.md#M09 */
  if(!mesh_section_wired(m,section))return EINVAL;
  /* design/bridge-and-ipc.md#clients: the link's lists, allocated at its first transfer, and the peer the link
     reaches, which every transfer on it is bound for */
  uint32_t link=queue/MESH_QUEUES,peer=atomic_load_explicit(&mesh_links(m)[link].peer,memory_order_relaxed);
  struct mesh_tx *tx=&mesh_client_at(m,context->client)->tx[link];
  if(tx->peer && tx->peer!=peer)return EHOSTUNREACH;
  tx->peer=peer;
  if(!tx->orders){
    struct mesh_section lists;
    int status=mesh_section_create(context,(size_t)MESH_QUEUES*2*MESH_ORDERS*sizeof(struct mesh_transfer),1,0,&lists);
    if(status)return status;
    tx->orders=(uint64_t)((char *)mesh_section_address(context,lists,0)-(char *)m);
  }
  _Atomic uint32_t *length=mesh_order_length(m,context->client,queue,receive);
  uint32_t index=atomic_load_explicit(length,memory_order_relaxed);
  if(index==MESH_ORDERS)return ENOSPC;
  if(!receive)for(uint32_t value=0;value<section.count;value++)mesh_publish_bind(context,mesh_section_row(section,value),link)->count++;
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){.local_row=section.first,.binding=identity,
    .count=section.count,.stride=section.stride,.invocation_pages=invocation_pages,.begin=begin,.end=end,.depth=depth,
    .bytes=section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
static int mesh_binding_order(const void *a,const void *b){
  const struct mesh_transfer *left=a,*right=b;
  int varying=(left->stride!=0)-(right->stride!=0);
  return varying?varying:(left->binding>right->binding)-(left->binding<right->binding);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M07 */
/* design/algorithm-sources.md#programcopy */
static int mesh_transfer_compare(const void *a,const void *b){
  return mesh_binding_order(*(const struct mesh_transfer *const *)a,*(const struct mesh_transfer *const *)b);
}

/* design/algorithm-sources.md#programcopy */
/* Each link's SEND cells: every publication (a send transfer's slot) a column of one cell an
   invocation of its range and a terminal one no producer writes, in (class, binding) order; the
   transfer's `first` is its slot 0 column's first cell, slot s's `s` columns on.  Each cell's
   request carries the producer's argument (1: a nonzero ready word); the bridge prepares the chains,
   each cell's successor and the queue gates (mesh-flow.c link_configure).  Each link's completion
   words: every receive transfer's slots and chunks an invocation of its range, transfer by transfer
   in the same order; the transfer's `first` is its first word (mesh-metal.m, mesh_host_inputs). */
int mesh_transfers_prepare(struct mesh_ctx *context,uint32_t slots,uint32_t invocations,uint32_t depth){
  struct hdr *m=context->M;
  if(!slots || slots>mesh_rows(m) || !invocations)return EINVAL;
  /* design/prepared-machine.md#M01 */
  /* The ring depth is one run-wide default, published in the header before mesh_transfers_start
     hands the client to the bridge; a transfer may state its own.  depth==invocations is the unrung
     machine. */
  if(!depth || depth>invocations)depth=invocations;
  struct mesh_client *client=mesh_client_at(m,context->client);
  atomic_store_explicit(&client->depth,depth,memory_order_relaxed);
  atomic_store_explicit(&client->cyclic,0,memory_order_relaxed);
  uint64_t cells=0;
  for(uint32_t q=0;q<m->links*MESH_QUEUES;q++){
    /* design/prepared-machine.md#M08 */
    for(int d=0;d<2;d++)qsort(mesh_transfers(m,context->client,q,d),
      atomic_load(mesh_order_length(m,context->client,q,d)),sizeof(struct mesh_transfer),mesh_binding_order);
    struct mesh_transfer *out=mesh_transfers(m,context->client,q,MESH_SEND);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_SEND));i++)
      cells+=(uint64_t)(out[i].stride?slots:1)*(mesh_transfer_active(out+i,invocations)+1);
  }
  struct mesh_section storage;
  /* design/prepared-machine.md#M04 */
  /* The SEND cell array is read by the provider and written by the GPU; it never appears in an
     SGE, so it is allocated outside the registered window. */
  int status=mesh_section_create(context,(cells?cells:1)*sizeof(struct mesh_send),1,0,&storage);
  if(status)return status;
  void *address=mesh_section_address(context,storage,0);
  memset(address,0,storage.bytes);
  context->send_off=(uintptr_t)address-(uintptr_t)m;
  context->send_bytes=(uint64_t)storage.pages*m->pgsz;
  uint64_t first=context->send_off;
  /* design/prepared-machine.md#M04 */
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=&client->tx[p];
    uint64_t base=(uint64_t)((char *)tx-(char *)m);
    uint32_t count=0,once=0,length=0,incoming=0;
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      length+=atomic_load_explicit(mesh_order_length(m,context->client,p*MESH_QUEUES+q,MESH_SEND),memory_order_relaxed);
      incoming+=atomic_load_explicit(mesh_order_length(m,context->client,p*MESH_QUEUES+q,MESH_RECEIVE),memory_order_relaxed);
    }
    uint32_t capacity=length>incoming?length:incoming;
    struct mesh_transfer **ordered=malloc((capacity?capacity:1)*sizeof *ordered);
    if(!ordered)return ENOMEM;
    uint32_t position=0;
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      struct mesh_transfer *out=mesh_transfers(m,context->client,p*MESH_QUEUES+q,MESH_SEND);
      uint32_t length=atomic_load_explicit(mesh_order_length(m,context->client,p*MESH_QUEUES+q,MESH_SEND),memory_order_relaxed);
      for(uint32_t i=0;i<length;i++){
        ordered[position++]=out+i;
        if(out[i].stride)count++;else once++;
      }
    }
    tx->count=count;tx->slots=slots;tx->once=once;tx->invocations=invocations;tx->cells=first;
    qsort(ordered,length,sizeof *ordered,mesh_transfer_compare);
    uint64_t cell=0;
    for(uint32_t i=0;i<length;i++){
      struct mesh_transfer *out=ordered[i];
      uint32_t column=mesh_transfer_active(out,invocations)+1;
      out->first=(uint32_t)cell;
      for(uint32_t slot=0;slot<out->count;slot++){
        uint64_t stream=tx->cells+sizeof(struct mesh_send)*(cell+(uint64_t)slot*column);
        struct mesh_send *cells=(void *)((char *)m+stream);
        for(uint32_t u=0;u<column;u++)cells[u].request.wr_id=1;
        struct mesh_publication *publication=mesh_publication_at(m,out->local_row+slot*out->stride);
        for(uint32_t j=0;j<publication->sends;j++)if(publication->targets[j].stream==base){
          publication->targets[j].stream=stream;
          publication->targets[j].stride=column;
        }
      }
      cell+=(uint64_t)(out->stride?slots:1)*column;
    }
    first+=cell*sizeof(struct mesh_send);
    /* design/prepared-machine.md#M07 */
    position=0;
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,p*MESH_QUEUES+q,MESH_RECEIVE);
      uint32_t count=atomic_load_explicit(mesh_order_length(m,context->client,p*MESH_QUEUES+q,MESH_RECEIVE),memory_order_relaxed);
      for(uint32_t i=0;i<count;i++)ordered[position++]=in+i;
    }
    qsort(ordered,position,sizeof *ordered,mesh_transfer_compare);
    uint64_t word=0;
    for(uint32_t i=0;i<position;i++){
      ordered[i]->first=(uint32_t)word;
      word+=(uint64_t)mesh_row_chunks(m,ordered[i]->local_row,ordered[i]->bytes)*ordered[i]->count*mesh_transfer_active(ordered[i],invocations);
    }
    free(ordered);
  }

  return 0;
}

/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
/* design/bridge-and-ipc.md#clients: the program handed to the node's bridge, which pairs it with the members' on each
   link it binds, or fails it (the link's status names the failure: its peer's program never started, a link lost, a
   lease the node cannot give): 0 once every link is paired, else the failure; ESRCH where the bridge is gone */
int mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  /* design/prepared-machine.md#M12 */
  uint64_t idle=0;
  if(!atomic_compare_exchange_strong_explicit(&client->configured,&idle,context->client,memory_order_release,memory_order_relaxed))return ECANCELED;
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
  for(uint32_t p=0;p<m->links;p++){
    if(!mesh_link_transfers(m,context->client,p))continue;
    struct mesh_port_info *port=&client->port[p];
    while(atomic_load_explicit(&port->prepared,memory_order_acquire)!=context->client){
      pid_t bridge=(pid_t)atomic_load_explicit(&m->bridge_pid,memory_order_relaxed);
      if(!bridge || (kill(bridge,0) && errno==ESRCH))return ESRCH;
      usleep(1000);
    }
    if(atomic_load_explicit(&port->phase,memory_order_relaxed)==MESH_STOPPED)
      return (int)atomic_load_explicit(&port->code,memory_order_relaxed);
  }
  return 0;
}

/* design/prepared-machine.md#M30 */
/* The prepared program runs without end: running invocation T is prepared invocation T mod N (N the
   program's invocations) in cycle T / N.  A producer releases T by storing its argument plus the cycle
   (mesh_host_publish), the bridge stores 1 plus the cycle in a landing's completion word, so a reader of T
   waits for a word of at least 1 + T / N (mesh_host_arrived returns the word).  Every transfer's ring depth
   divides N, so T's slot is (T - begin) mod depth in every cycle.  Called after mesh_transfers_prepare,
   before mesh_transfers_start. */
int mesh_transfers_cyclic(struct mesh_ctx *context){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  uint32_t invocations=client->tx[0].invocations;
  uint32_t fallback=atomic_load_explicit(&client->depth,memory_order_relaxed);
  if(!invocations)return EINVAL;
  for(uint32_t q=0;q<m->links*MESH_QUEUES;q++)for(int d=0;d<2;d++){
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,d);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,d));i++){
      uint32_t depth=transfers[i].depth?transfers[i].depth:fallback;
      if(!depth || invocations%depth || transfers[i].begin || transfers[i].end<invocations)return EINVAL;
    }
  }
  atomic_store_explicit(&client->cyclic,1,memory_order_release);
  return 0;
}

/* design/prepared-machine.md#M12 */
/* The client's program ends and the client stays attached: its allocations remain, its transfers are
   cleared for another mesh_transfers_prepare and mesh_transfers_start.  The bridge takes the program off its
   ports once `configured` no longer names the client and then clears `released`.  ETIMEDOUT where the bridge
   has not let it go within `seconds`. */
int mesh_transfers_stop(struct mesh_ctx *context,double seconds){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  uint64_t owner=context->client;
  if(atomic_compare_exchange_strong_explicit(&client->configured,&owner,0,memory_order_seq_cst,memory_order_relaxed)){
    mesh_control_notify(m);
    for(double waited=0;atomic_load_explicit(&client->released,memory_order_seq_cst)==context->client;waited+=1e-4){
      if(waited>seconds)return ETIMEDOUT;
      usleep(100);
    }
  }
  for(uint32_t q=0;q<m->links*MESH_QUEUES*2;q++)atomic_store_explicit(client->length+q,0,memory_order_release);
  for(uint32_t p=0;p<m->links;p++){atomic_store_explicit(&client->port[p].prepared,0,memory_order_release);client->tx[p].peer=0;}
  return 0;
}

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,int wire,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz;
  /* design/prepared-machine.md#M09 */
  size_t capacity=mesh_arena_range(m,wire).count;
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
  /* design/prepared-machine.md#M01 */
  /* design/prepared-machine.md#M03 */
  uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block,wire);
  /* design/prepared-machine.md#M01 */
  /* Registration wires its pages eagerly, so until now every operand was resident by the time the
     bridge came up.  Unregistered pages are not, and their first fault would land in the measured
     region; touch the span here, at preparation, where it costs nothing that is being timed. */
  if(!wire && page!=MESH_ABSENT)
    for(uint32_t p=0;p<(uint32_t)span*count;p++)*(volatile unsigned char *)mesh_at(m,page+p)=0;
  if(page==MESH_ABSENT){
    int error=errno;
    mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),first,rows);
    for(uint32_t r=first;r<first+rows;r+=stride)mesh_buffers(m)[r].pages=0;
    mesh_rows_release(context,first,rows);return error;
  }
  for(uint32_t slot=0;slot<count;slot++)
    mesh_backing_bind(context,first+slot*stride,(uint32_t)span,page+slot*(uint32_t)span,slot);
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
int mesh_section_slice(struct mesh_ctx *context,struct mesh_section source,size_t offset,size_t bytes,
  uint32_t invocations,uint32_t invocation_pages,struct mesh_section *section){
  struct hdr *m=context->M;
  uint64_t block=(uint64_t)m->block*m->pgsz,span=(uint64_t)invocation_pages*m->pgsz;
  if(!bytes || !invocations || !invocation_pages || invocation_pages%m->block ||
     offset>span || bytes>span-offset || (uint64_t)invocations*invocation_pages>source.pages)return EINVAL;
  uint32_t chunks=(uint32_t)((offset%block+bytes+block-1)/block);
  uint64_t stride=(uint64_t)chunks*invocations;
  if(stride*source.count>mesh_rows(m))return ENOMEM;
  uint32_t first=mesh_rows_alloc(context,(uint32_t)stride*source.count);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t s=0;s<source.count;s++)for(uint32_t t=0;t<invocations;t++)for(uint32_t k=0;k<chunks;k++){
    struct mesh_page_entry *from=&mesh_page(m)[source.first+s*source.stride+(uint64_t)t*invocation_pages/m->block+offset/block+k];
    struct mesh_page_entry *to=&mesh_page(m)[first+s*stride+(uint64_t)t*chunks+k];
    atomic_store_explicit(&to->mapping,atomic_load_explicit(&from->mapping,memory_order_relaxed),memory_order_relaxed);
    atomic_store_explicit(&to->address,atomic_load_explicit(&from->address,memory_order_relaxed)+(k?0:offset%block),memory_order_relaxed);
  }
  *section=(struct mesh_section){first,chunks*m->block,bytes,source.count,(uint32_t)stride};
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return (char *)context->M+atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)].address,memory_order_relaxed);}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffers(context->M)[section.first].constant=1;
}

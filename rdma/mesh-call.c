#include "mesh-call.h"
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

static uint64_t mesh_section_rows(struct mesh_section section){
  return section.stride?(uint64_t)section.stride*section.count:1;
}

/* design/bridge-and-ipc.md#region: a destination's lists leased with room for twice the entries a list holds now (at
   first the entries a block holds: the lists are sized by their use, a relay's and a session's a block), the entries
   copied and the lease they leave returned (mesh_lease_return); 0, or the refused lease */
static int mesh_lists_grow(struct mesh_ctx *context,struct mesh_tx *tx){
  struct hdr *m=context->M;
  const uint32_t lists=MESH_QUEUES*2,first=(uint32_t)((uint64_t)m->block*m->pgsz/(lists*sizeof(struct mesh_transfer)));
  uint32_t room=tx->orders?2*tx->room:first;
  if(room>MESH_ORDERS)room=MESH_ORDERS;
  if(room<1)room=1;
  struct mesh_section section;
  int status=mesh_section_create(context,(size_t)lists*room*sizeof(struct mesh_transfer),1,&section);
  if(status)return status;
  struct mesh_transfer *grown=mesh_section_address(context,section,0);
  if(tx->orders){
    const struct mesh_transfer *held=(const struct mesh_transfer *)((char *)m+tx->orders);
    for(uint32_t list=0;list<lists;list++)
      memcpy(grown+(size_t)list*room,held+(size_t)list*tx->room,
             atomic_load_explicit(tx->length+list,memory_order_relaxed)*sizeof(struct mesh_transfer));
    mesh_lease_return(context,tx->lists);
  }
  tx->orders=(uint64_t)((char *)grown-(char *)m);tx->room=room;tx->lists=section.first;
  return 0;
}

/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *context,uint32_t node,uint32_t queue,int receive,uint32_t identity,struct mesh_section section,
  uint32_t invocation_pages,uint32_t begin,uint32_t end,uint32_t depth){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  if(!node || node==m->node || queue>=MESH_QUEUES || end<=begin)return EINVAL;
  if(!section.count || (uint64_t)section.first+mesh_section_rows(section)>mesh_rows(m))return EINVAL;
  /* design/bridge-and-ipc.md#clients: the client's destination slot for the node, the one bound for it or the next
     vacant, and its lists, leased at its first transfer */
  uint32_t destination=mesh_destination(client,node);
  if(destination==MESH_DESTINATIONS)return ENOSPC;
  struct mesh_tx *tx=&client->tx[destination];
  tx->node=node;
  uint32_t channel=destination*MESH_QUEUES+queue;
  _Atomic uint32_t *length=mesh_order_length(m,context->client,channel,receive);
  uint32_t index=atomic_load_explicit(length,memory_order_relaxed);
  if(index==MESH_ORDERS)return ENOSPC;
  if(!tx->orders || index==tx->room){
    int status=mesh_lists_grow(context,tx);
    if(status)return status;
  }
  mesh_transfers(m,context->client,channel,receive)[index]=(struct mesh_transfer){.local_row=section.first,.binding=identity,
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

/* design/prepared-machine.md#M10: a program's targets by row, a row's by its cells */
static int mesh_target_order(const void *a,const void *b){
  const struct mesh_target *left=a,*right=b;
  return left->row!=right->row?(left->row>right->row)-(left->row<right->row):(left->cells>right->cells)-(left->cells<right->cells);
}

/* design/algorithm-sources.md#programcopy */
/* Each destination's SEND cells: every publication (a send transfer's slot) a column of one cell an
   invocation of its range and a terminal one no producer writes, in (class, binding) order; the
   transfer's `first` is its slot 0 column's first cell, slot s's `s` columns on.  Each column is a
   target of the row its slot sends (mesh.h mesh_row_targets: the program's targets, after its SEND
   cells in their lease, sorted by row).  Each cell's
   request carries the producer's argument (1: a nonzero ready word); the bridge prepares the chains,
   each cell's successor and the queue gates (mesh-flow.c sends_configure).  Each destination's completion
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
  uint64_t cells=0,targets=0;
  const uint32_t destinations=mesh_destination(client,0);
  for(uint32_t q=0;q<destinations*MESH_QUEUES;q++){
    /* design/prepared-machine.md#M08: a transfer's slots are among the program's (a slot a queue pair of its queue) */
    for(int d=0;d<2;d++){
      struct mesh_transfer *list=mesh_transfers(m,context->client,q,d);
      const uint32_t length=atomic_load(mesh_order_length(m,context->client,q,d));
      for(uint32_t i=0;i<length;i++)if(list[i].count>slots)return EINVAL;
      qsort(list,length,sizeof(struct mesh_transfer),mesh_binding_order);
    }
    struct mesh_transfer *out=mesh_transfers(m,context->client,q,MESH_SEND);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_SEND));i++){
      cells+=mesh_transfer_cells(out+i,invocations);targets+=out[i].count;
    }
  }
  /* design/prepared-machine.md#M04, #M10: one lease, the SEND cells and after them the targets */
  const uint64_t cell_bytes=(cells?cells:1)*sizeof(struct mesh_send);
  struct mesh_section storage;
  int status=mesh_section_create(context,cell_bytes+targets*sizeof(struct mesh_target),1,&storage);
  if(status)return status;
  void *address=mesh_section_address(context,storage,0);
  memset(address,0,storage.bytes);
  struct mesh_target *target=targets?(struct mesh_target *)((char *)address+cell_bytes):NULL;
  uint32_t target_count=0;
  context->send_off=(uintptr_t)address-(uintptr_t)m;
  context->send_bytes=(uint64_t)storage.pages*m->pgsz;
  uint64_t first=context->send_off;
  /* design/prepared-machine.md#M04 */
  for(uint32_t d=0;d<destinations;d++){
    struct mesh_tx *tx=&client->tx[d];
    uint32_t count=0,once=0,length=0,incoming=0;
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      length+=atomic_load_explicit(mesh_order_length(m,context->client,d*MESH_QUEUES+q,MESH_SEND),memory_order_relaxed);
      incoming+=atomic_load_explicit(mesh_order_length(m,context->client,d*MESH_QUEUES+q,MESH_RECEIVE),memory_order_relaxed);
    }
    uint32_t capacity=length>incoming?length:incoming;
    struct mesh_transfer **ordered=malloc((capacity?capacity:1)*sizeof *ordered);
    if(!ordered)return ENOMEM;
    uint32_t position=0;
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      struct mesh_transfer *out=mesh_transfers(m,context->client,d*MESH_QUEUES+q,MESH_SEND);
      uint32_t length=atomic_load_explicit(mesh_order_length(m,context->client,d*MESH_QUEUES+q,MESH_SEND),memory_order_relaxed);
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
        target[target_count++]=(struct mesh_target){.cells=stream,.row=out->local_row+slot*out->stride};
      }
      cell+=mesh_transfer_cells(out,invocations);
    }
    first+=cell*sizeof(struct mesh_send);
    /* design/prepared-machine.md#M07 */
    position=0;
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,d*MESH_QUEUES+q,MESH_RECEIVE);
      uint32_t count=atomic_load_explicit(mesh_order_length(m,context->client,d*MESH_QUEUES+q,MESH_RECEIVE),memory_order_relaxed);
      for(uint32_t i=0;i<count;i++)ordered[position++]=in+i;
    }
    qsort(ordered,position,sizeof *ordered,mesh_transfer_compare);
    uint64_t word=0;
    for(uint32_t i=0;i<position;i++){ordered[i]->first=(uint32_t)word;word+=mesh_transfer_frames(m,ordered[i],invocations);}
    free(ordered);
  }
  /* design/prepared-machine.md#M10 */
  if(target)qsort(target,target_count,sizeof *target,mesh_target_order);
  client->targets=target?(uint64_t)((char *)target-(char *)m):0;
  client->target_count=target_count;
  return 0;
}

/* design/prepared-machine.md#M06; design/bridge-and-ipc.md#programs: the program handed to the node's bridge, which
   compiles its chains toward each destination on the leases its slot names and posts its first receives, or fails it
   (that destination's status names the code and the node): 0 once every destination started, else the first failed
   destination's code; ESRCH where the bridge is gone (its exit wakes it: mesh-dataflow.c mesh_watch), and ECANCELED once
   the program's own slot is retired while it waits (mesh_retire wakes it: a revoke); it waits on nothing else, and on
   no peer */
int mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  uint64_t idle=0;
  if(!atomic_compare_exchange_strong_explicit(&client->configured,&idle,context->client,memory_order_release,memory_order_relaxed))return ECANCELED;
  mesh_notify(&m->control);
  const uint32_t destinations=mesh_destination(client,0);
  int failed=0;
  for(uint32_t d=0;d<destinations;d++){
    if(!mesh_destination_transfers(m,context->client,d))continue;
    struct mesh_port_info *status=&client->status[d];
    for(uint64_t seen;(seen=atomic_load_explicit(&status->prepared,memory_order_acquire))!=context->client;){
      if(atomic_load_explicit(&client->owner,memory_order_acquire)!=context->client)return ECANCELED;
      if(mesh_gone((pid_t)atomic_load_explicit(&m->bridge_pid,memory_order_acquire)))return ESRCH;
      os_sync_wait_on_address(&status->prepared,seen,sizeof seen,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
    }
    if(!failed && atomic_load_explicit(&status->phase,memory_order_relaxed)==MESH_STOPPED)
      failed=(int)atomic_load_explicit(&status->code,memory_order_relaxed);
  }
  return failed;
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
  for(uint32_t q=0;q<mesh_destination(client,0)*MESH_QUEUES;q++)for(int d=0;d<2;d++){
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,d);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,d));i++){
      uint32_t depth=transfers[i].depth?transfers[i].depth:fallback;
      if(!depth || invocations%depth || transfers[i].begin || transfers[i].end<invocations)return EINVAL;
    }
  }
  atomic_store_explicit(&client->cyclic,1,memory_order_release);
  return 0;
}

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
  /* design/prepared-machine.md#M01 */
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

/* ---- The members' agreement on leases and relays (design/bridge-and-ipc.md#leases, #meet, #relays) --------------- */

/* a member's report at the meeting: this node, its paired ports (each its index, its peer, the index the peer gave the
   cable, its link's leases and the mask of them this program claimed), and each destination the program binds (its
   node, its slots, invocations, depth and cycle, and its transfers each way queue by queue, which a relay mirrors);
   the root's second report the assignment (a code and, for each rank's destination, the port it takes and each queue's
   lease) */
struct agree_port { uint32_t index,peer,peer_port,leases,mask; };
struct agree_destination { uint32_t node,slots,invocations,depth,cyclic,sends[MESH_QUEUES],receives[MESH_QUEUES]; };
struct agree_head { uint32_t node,ports,destinations,reserved; };
struct agree_lease { uint32_t rank,node,port; uint8_t lease[MESH_QUEUES]; };
struct agree_assignment { int32_t code; uint32_t count; };
#define AGREE_SECOND UINT64_C(0x6d656574206c6561)

static int agree_cancelled(struct mesh_ctx *c){ return atomic_load(&mesh_client_at(c->M,c->client)->owner)!=c->client; }
static int whole_io(struct mesh_ctx *c,int fd,void *bytes,size_t size,int sending){
  __atomic_store_n(&c->agreeing,fd,__ATOMIC_RELEASE);
  if(agree_cancelled(c)){__atomic_store_n(&c->agreeing,-1,__ATOMIC_RELEASE);return ECANCELED;}
  for(size_t done=0;done<size;){
    const ssize_t moved=sending?send(fd,(char *)bytes+done,size-done,0):recv(fd,(char *)bytes+done,size-done,0);
    if(moved<0 && errno==EINTR)continue;
    if(moved<=0){__atomic_store_n(&c->agreeing,-1,__ATOMIC_RELEASE);return moved?errno:ECONNRESET;}
    done+=(size_t)moved;
  }
  __atomic_store_n(&c->agreeing,-1,__ATOMIC_RELEASE);
  return 0;
}
/* the meeting of `key` at node `root`: this member's report sent, and every member's read back (*table, the caller's
   to free: a uint32 n, n uint32 byte counts, the reports); 0, or the errno */
static int agree_meet(struct mesh_ctx *c,uint32_t root,uint64_t key,uint32_t n,uint32_t rank,const void *report,uint32_t bytes,unsigned char **table,size_t *length){
  const int fd=mesh_request(root,"meet",0);
  if(fd<0)return -fd;
  const struct mesh_meet head={key,n,rank,bytes,0};
  int error=whole_io(c,fd,(void *)&head,sizeof head,1);
  if(!error && bytes)error=whole_io(c,fd,(void *)report,bytes,1);
  if(!error && shutdown(fd,SHUT_WR))error=errno;
  size_t room=4096,got=0;
  unsigned char *made=error?NULL:malloc(room);
  if(!error && !made)error=ENOMEM;
  while(!error){
    if(got==room){unsigned char *grown=realloc(made,room*2);if(!grown){error=ENOMEM;break;}made=grown;room*=2;}
    __atomic_store_n(&c->agreeing,fd,__ATOMIC_RELEASE);
    if(agree_cancelled(c)){__atomic_store_n(&c->agreeing,-1,__ATOMIC_RELEASE);error=ECANCELED;break;}
    const ssize_t moved=recv(fd,made+got,room-got,0);
    __atomic_store_n(&c->agreeing,-1,__ATOMIC_RELEASE);
    if(moved<0 && errno==EINTR)continue;
    if(moved<0)error=errno;
    else if(!moved)break;
    else got+=(size_t)moved;
  }
  close(fd);
  uint32_t count=0;
  if(!error && got>=sizeof count)memcpy(&count,made,sizeof count);
  if(!error && (count!=n || got<sizeof(uint32_t)*(1+(size_t)n)))error=ECONNRESET;
  if(error){free(made);return error;}
  *table=made;*length=got;
  return 0;
}
/* rank r's report in a meeting's table (its bytes in *bytes), NULL past it */
static const unsigned char *agree_report(const unsigned char *table,size_t length,uint32_t r,uint32_t *bytes){
  uint32_t n,at=0;
  memcpy(&n,table,sizeof n);
  if(r>=n)return NULL;
  size_t offset=sizeof(uint32_t)*(1+(size_t)n);
  for(uint32_t i=0;i<=r;i++){memcpy(&at,table+sizeof(uint32_t)*(1+i),sizeof at);if(i<r)offset+=at;}
  if(offset+at>length)return NULL;
  *bytes=at;
  return table+offset;
}
/* a report's port toward neighbour `node` over the cable both ends take (the lowest index the pair's lower node gave
   one of its cables to the other): its entry, NULL where none */
static const struct agree_port *agree_port_to(const unsigned char *report,uint32_t own,uint32_t node){
  struct agree_head head;memcpy(&head,report,sizeof head);
  const struct agree_port *best=NULL;
  uint32_t rank=UINT32_MAX;
  for(uint32_t p=0;p<head.ports;p++){
    const struct agree_port *at=(const struct agree_port *)(report+sizeof head)+p;
    if(at->peer!=node)continue;
    const uint32_t r=own<node?at->index:at->peer_port;
    if(r<rank){rank=r;best=at;}
  }
  return best;
}
/* a report's destination record for `node` and its transfers, NULL where it binds none */
static const struct agree_destination *agree_destination(const unsigned char *report,uint32_t bytes,uint32_t node,const struct mesh_transfer **list){
  struct agree_head head;memcpy(&head,report,sizeof head);
  size_t at=sizeof head+head.ports*sizeof(struct agree_port);
  for(uint32_t d=0;d<head.destinations && at+sizeof(struct agree_destination)<=bytes;d++){
    const struct agree_destination *x=(const void *)(report+at);
    uint64_t transfers=0;
    for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=(uint64_t)x->sends[k]+x->receives[k];
    at+=sizeof *x;
    if(x->node==node){*list=(const void *)(report+at);return at+transfers*sizeof(struct mesh_transfer)<=bytes?x:NULL;}
    at+=transfers*sizeof(struct mesh_transfer);
  }
  return NULL;
}
/* the root's assignment: every member pair either binds, routed on this node's map (mesh_route), each hop a lease of
   each queue the pair uses, the lowest that the hop's member ends claimed and no other pair of this assignment holds on
   that link; each relay of a route asked to forward the pair's two directions (its channel kept in the program's
   context); *out the assignment, its code 0 or the refusal (EHOSTUNREACH no route, EBUSY no lease free, a relay's) */
struct agree_used { uint32_t a,b,mask; };
static int agree_plan(struct mesh_ctx *c,const uint32_t *nodes,uint32_t n,const unsigned char *table,size_t length,unsigned char **out,uint32_t *bytes){
  struct hdr *m=c->M;
  const size_t room=(size_t)n*n+1;
  struct agree_lease *lease=calloc(room,sizeof *lease);
  struct agree_used *used=calloc(room*MESH_MEET_RANKS+1,sizeof *used);
  unsigned char **request=calloc(MESH_MEET_RANKS,sizeof *request);
  size_t *request_bytes=calloc(MESH_MEET_RANKS,sizeof *request_bytes);
  uint32_t *relay_node=calloc(MESH_MEET_RANKS,sizeof *relay_node),relays=0,leases=0,links=0;
  int error=lease && used && request && request_bytes && relay_node?0:ENOMEM;
  for(uint32_t i=0;i<n && !error;i++){
    uint32_t bytes_i=0;
    const unsigned char *report=agree_report(table,length,i,&bytes_i);
    if(!report || bytes_i<sizeof(struct agree_head)){error=EPROTO;break;}
    for(uint32_t j=i+1;j<n && !error;j++){
      uint32_t bytes_j=0;
      const unsigned char *other=agree_report(table,length,j,&bytes_j);
      const struct mesh_transfer *list_i=NULL,*list_j=NULL;
      if(!other || bytes_j<sizeof(struct agree_head)){error=EPROTO;break;}
      const struct agree_destination *to_j=agree_destination(report,bytes_i,nodes[j],&list_i),*to_i=agree_destination(other,bytes_j,nodes[i],&list_j);
      if(!to_j && !to_i)continue;
      uint32_t rings=0;
      for(uint32_t k=0;k<MESH_QUEUES;k++)
        rings|=(uint32_t)((to_j && (to_j->sends[k] || to_j->receives[k])) || (to_i && (to_i->sends[k] || to_i->receives[k])))<<k;
      if((to_j && to_j->slots>1) || (to_i && to_i->slots>1)){error=EINVAL;break;}
      uint32_t *path=NULL;
      const uint32_t hops=mesh_route(m,nodes[i],nodes[j],&path);
      if(hops<2 || hops>MESH_MEET_RANKS){free(path);error=EHOSTUNREACH;break;}
      uint8_t (*chosen)[MESH_QUEUES]=calloc(hops,sizeof *chosen);
      if(!chosen){free(path);error=ENOMEM;break;}
      for(uint32_t h=0;h+1<hops && !error;h++){
        const uint32_t x=path[h],y=path[h+1],lo=x<y?x:y,hi=x<y?y:x;
        uint32_t mask=(UINT32_C(1)<<MESH_LEASES)-1,most=MESH_LEASES;
        for(uint32_t r=0;r<n && !error;r++)if(nodes[r]==x || nodes[r]==y){
          uint32_t b=0;
          const unsigned char *at=agree_report(table,length,r,&b);
          const struct agree_port *port=at?agree_port_to(at,nodes[r],nodes[r]==x?y:x):NULL;
          if(!port){error=EHOSTUNREACH;break;}
          mask&=port->mask;
          if(port->leases<most)most=port->leases;
        }
        uint32_t u=0;
        while(u<links && !(used[u].a==lo && used[u].b==hi))u++;
        if(u==links)used[links++]=(struct agree_used){lo,hi,0};
        for(uint32_t k=0;k<MESH_QUEUES && !error;k++){
          if(!(rings>>k&1))continue;
          uint32_t l=0;
          while(l<most && (!(mask>>l&1) || (used[u].mask>>l&1)))l++;
          if(l==most){error=EBUSY;break;}
          used[u].mask|=UINT32_C(1)<<l;chosen[h][k]=(uint8_t)(l+1);
        }
      }
      for(uint32_t end=0;end<2 && !error;end++){
        const uint32_t r=end?j:i,far=end?nodes[i]:nodes[j],next=end?path[hops-2]:path[1],h=end?hops-2:0;
        uint32_t b=0;
        const unsigned char *at=agree_report(table,length,r,&b);
        const struct agree_port *port=agree_port_to(at,nodes[r],next);
        lease[leases]=(struct agree_lease){r,far,port->index,{0}};
        memcpy(lease[leases++].lease,chosen[h],MESH_QUEUES);
      }
      for(uint32_t h=1;h+1<hops && !error;h++){
        const struct agree_destination *shape=to_j?to_j:to_i;
        const struct mesh_transfer *list=to_j?list_i:list_j;
        const int flipped=!to_j;
        struct mesh_relay_entry entry={.a=flipped?path[h+1]:path[h-1],.b=flipped?path[h-1]:path[h+1],.slots=shape->slots?shape->slots:1,
          .invocations=shape->invocations,.depth=shape->depth,.cyclic=shape->cyclic};
        memcpy(flipped?entry.lease_b:entry.lease_a,chosen[h-1],MESH_QUEUES);
        memcpy(flipped?entry.lease_a:entry.lease_b,chosen[h],MESH_QUEUES);
        uint64_t transfers=0;
        for(uint32_t k=0;k<MESH_QUEUES;k++){entry.sends[k]=shape->sends[k];entry.receives[k]=shape->receives[k];transfers+=(uint64_t)shape->sends[k]+shape->receives[k];}
        uint32_t v=0;
        while(v<relays && relay_node[v]!=path[h])v++;
        if(v==MESH_MEET_RANKS){error=E2BIG;break;}
        if(v==relays){relay_node[relays++]=path[h];request_bytes[v]=sizeof(struct mesh_relay_request);}
        const size_t add=sizeof entry+transfers*sizeof(struct mesh_transfer);
        unsigned char *grown=realloc(request[v],request_bytes[v]+add);
        if(!grown){error=ENOMEM;break;}
        request[v]=grown;
        memcpy(grown+request_bytes[v],&entry,sizeof entry);
        memcpy(grown+request_bytes[v]+sizeof entry,list,transfers*sizeof(struct mesh_transfer));
        request_bytes[v]+=add;
      }
      free(chosen);free(path);
    }
  }
  if(!error && relays && !(c->relays=calloc(relays,sizeof *c->relays)))error=ENOMEM;
  for(uint32_t v=0;v<relays && !error;v++){
    struct mesh_relay_request head={mesh_client_at(m,c->client)->key,0,(uint32_t)(request_bytes[v]-sizeof head)};
    for(size_t at=sizeof head;at<request_bytes[v];){
      struct mesh_relay_entry e;memcpy(&e,request[v]+at,sizeof e);
      uint64_t transfers=0;
      for(uint32_t k=0;k<MESH_QUEUES;k++)transfers+=(uint64_t)e.sends[k]+e.receives[k];
      at+=sizeof e+transfers*sizeof(struct mesh_transfer);head.count++;
    }
    memcpy(request[v],&head,sizeof head);
    const int fd=mesh_request(relay_node[v],"relay",0);
    if(fd<0){error=-fd;break;}
    c->relays[c->relay_count++]=fd;
    error=whole_io(c,fd,request[v],request_bytes[v],1);
  }
  for(uint32_t v=0;v<c->relay_count && !error;v++){
    int32_t answer=0;
    error=whole_io(c,c->relays[v],&answer,sizeof answer,0);
    if(!error)error=answer;
  }
  for(uint32_t v=0;v<MESH_MEET_RANKS && request;v++)free(request[v]);
  free(request);free(request_bytes);free(relay_node);free(used);
  const size_t made=sizeof(struct agree_assignment)+(error?0:leases)*sizeof *lease;
  unsigned char *assignment=malloc(made);
  if(!assignment){free(lease);return ENOMEM;}
  const struct agree_assignment head={error,error?0:leases};
  memcpy(assignment,&head,sizeof head);
  if(!error)memcpy(assignment+sizeof head,lease,leases*sizeof *lease);
  free(lease);
  *out=assignment;*bytes=(uint32_t)made;
  return error;
}
void mesh_transfers_interrupt(struct mesh_ctx *context){
  const int fd=__atomic_load_n(&context->agreeing,__ATOMIC_ACQUIRE);
  if(fd>=0)shutdown(fd,SHUT_RDWR);
}
int mesh_transfers_agree(struct mesh_ctx *context,const uint32_t *nodes,uint32_t n,uint32_t rank){
  struct hdr *m=context->M;
  struct mesh_client *client=mesh_client_at(m,context->client);
  if(!n || rank>=n || n>MESH_MEET_RANKS || nodes[rank]!=m->node)return EINVAL;
  const uint32_t destinations=mesh_destination(client,0);
  int bound=0;
  for(uint32_t d=0;d<destinations;d++)bound|=mesh_destination_transfers(m,context->client,d)!=0;
  if(n==1 && !bound)return 0;
  size_t bytes=sizeof(struct agree_head)+MESH_PORTS*sizeof(struct agree_port);
  for(uint32_t d=0;d<destinations;d++){
    bytes+=sizeof(struct agree_destination);
    for(uint32_t q=0;q<MESH_QUEUES*2;q++)bytes+=atomic_load(client->tx[d].length+q)*sizeof(struct mesh_transfer);
  }
  unsigned char *report=calloc(1,bytes);
  if(!report)return ENOMEM;
  struct agree_head head={m->node,0,0,0};
  struct agree_port port[MESH_PORTS];
  for(uint32_t p=0;p<m->links && p<MESH_PORTS;p++){
    const struct mesh_link_info *link=&mesh_links(m)[p];
    const uint32_t leases=atomic_load(&link->leases);
    if(atomic_load(&link->phase)!=MESH_PAIRED || !leases)continue;
    uint32_t mask=0;
    for(uint32_t l=0;l<leases && l<MESH_LEASES;l++){
      uint64_t vacant=0;
      if(atomic_compare_exchange_strong(mesh_leases(m,p)+l,&vacant,context->client))mask|=UINT32_C(1)<<l;
    }
    port[head.ports++]=(struct agree_port){p,atomic_load(&link->peer),atomic_load(&link->peer_port),leases,mask};
  }
  memcpy(report+sizeof head,port,head.ports*sizeof *port);
  size_t at=sizeof head+head.ports*sizeof(struct agree_port);
  for(uint32_t d=0;d<destinations;d++){
    if(!mesh_destination_transfers(m,context->client,d))continue;
    const struct mesh_tx *tx=&client->tx[d];
    struct agree_destination x={tx->node,tx->slots,tx->invocations,atomic_load(&client->depth),atomic_load(&client->cyclic),{0},{0}};
    unsigned char *list=report+at+sizeof x;
    for(int way=0;way<2;way++)for(uint32_t k=0;k<MESH_QUEUES;k++){
      const uint32_t count=atomic_load(mesh_order_length(m,context->client,d*MESH_QUEUES+k,way?MESH_RECEIVE:MESH_SEND));
      if(way)x.receives[k]=count;else x.sends[k]=count;
      memcpy(list,mesh_transfers(m,context->client,d*MESH_QUEUES+k,way?MESH_RECEIVE:MESH_SEND),count*sizeof(struct mesh_transfer));
      list+=count*sizeof(struct mesh_transfer);
    }
    memcpy(report+at,&x,sizeof x);
    at=(size_t)(list-report);head.destinations++;
  }
  memcpy(report,&head,sizeof head);
  const uint64_t key=client->key;
  unsigned char *table=NULL,*second=NULL,*plan=NULL;
  size_t length=0,second_length=0;
  uint32_t plan_bytes=0;
  int error=agree_meet(context,nodes[0],key,n,rank,report,(uint32_t)at,&table,&length);
  if(!error && !rank){
    const int planned=agree_plan(context,nodes,n,table,length,&plan,&plan_bytes);
    if(planned && !plan)error=planned;
  }
  if(!error)error=agree_meet(context,nodes[0],key^AGREE_SECOND,n,rank,plan,rank?0:plan_bytes,&second,&second_length);
  uint32_t got=0;
  const unsigned char *assigned=error?NULL:agree_report(second,second_length,0,&got);
  struct agree_assignment result={EPROTO,0};
  if(assigned && got>=sizeof result)memcpy(&result,assigned,sizeof result);
  if(!error && (result.code || got<sizeof result+(size_t)result.count*sizeof(struct agree_lease)))error=result.code?result.code:EPROTO;
  uint32_t kept[MESH_PORTS]={0};
  for(uint32_t e=0;!error && e<result.count;e++){
    struct agree_lease x;memcpy(&x,assigned+sizeof result+e*sizeof x,sizeof x);
    if(x.rank!=rank)continue;
    const uint32_t d=mesh_destination(client,x.node);
    if(d==MESH_DESTINATIONS || client->tx[d].node!=x.node || x.port>=MESH_PORTS){error=EPROTO;break;}
    client->tx[d].port=(uint8_t)(x.port+1);
    memcpy(client->tx[d].lease,x.lease,MESH_QUEUES);
    for(uint32_t k=0;k<MESH_QUEUES;k++)if(x.lease[k])kept[x.port]|=UINT32_C(1)<<(x.lease[k]-1);
  }
  for(uint32_t p=0;p<head.ports;p++)for(uint32_t l=0;l<MESH_LEASES;l++){
    if(!(port[p].mask>>l&1) || (!error && (kept[port[p].index]>>l&1)))continue;
    uint64_t held=context->client;
    atomic_compare_exchange_strong(mesh_leases(m,port[p].index)+l,&held,0);
  }
  if(error && context->relays){
    for(uint32_t v=0;v<context->relay_count;v++)close(context->relays[v]);
    free(context->relays);context->relays=NULL;context->relay_count=0;
  }
  free(report);free(table);free(second);free(plan);
  return error;
}

/* ---- The numerical executor (mesh 2b49f3c6^; mesh-call.h) ------------------------------------------------------ */

/* hot-path-loads M23 */
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs;
  unsigned char *values;
  struct mesh_call_operand *operands;
  size_t input_count,output_count,dependency_count;
  uint32_t worker,identity,pending,initial;
  mesh_invoke submit;
  void *argument;
  struct mesh_function *next;
  struct mesh_completion *completions;
};
_Static_assert(sizeof(struct mesh_function)==104,"M23");
/* hot-path-loads M18 on prepared-machine.md M07: a received row's landing of one invocation slot, which a worker polls
   (`word`, 1 plus the cycle once landed, UINT64_MAX once its destination was cancelled; `expected` the word of its next
   cycle): its uses (call records uses[first, first + count)) each take one arrival; the slot's one counting landing
   also releases a forwarded row's SEND cells (`publications`: the row sent on from where it landed) and completes
   one event of `instances` instances from `slot` (1, or a constant's every slot); `identity` its destination (+
   MESH_DESTINATIONS in the constants program) */
struct mesh_call_landing {
  _Atomic uint64_t *word;
  uint64_t expected;
  const struct prepared_publication *publications;
  uint32_t first,count,publication_count,slot,instances,counted,identity;
};
/* hot-path-loads M22 */
struct mesh_call_worker {
  _Alignas(128) struct mesh_calls *calls;
  uint32_t index;
  _Atomic uint32_t completed;
  struct mesh_arrival *arrivals;
  struct mesh_completion *returns;
  uint32_t return_count,arrival_count;
  _Atomic uintptr_t frame;
  struct mesh_call_landing *landings;
  uint32_t *uses;
  uint32_t landing_count,use_count;
};
_Static_assert(sizeof(struct mesh_call_worker)==128 && _Alignof(struct mesh_call_worker)==128,"mesh_call_worker");
/* a resident consumer's input word of slot `index` of a produced section (mesh_section_input) */
struct mesh_watch { uint32_t first,index; _Atomic uint64_t *word; };
/* a produced section's slot `index` arrival: one consumer call's cell, by the section's first row */
struct mesh_produced { uint32_t first,index; struct mesh_arrival *cell; };
/* design/bridge-dataflow.md#routed-readiness (mesh 45902d8e^ struct mesh_route, mesh_route_use and mesh_active, on this
   executor).  A route's state of one generation's index and cycle parity (a cycle's events may come while the one
   before it finishes): the directory's sections arrived (a bit each), whether the directory was read (`prepared`), the
   candidates arrived (each 1 produced, 2 omitted, and their count), each consumer's routed candidates still awaited,
   and the consumers fired */
struct mesh_route_state { uint32_t metas,candidates,delivered,prepared; uint8_t *arrived; uint32_t *remaining; };
/* a route: its worker, its three vectors (owners, ordinals, offsets) and the distinct sections they lie in (`meta`, a
   vector's at `vector_meta`), its candidates, consumers and producers, each index's vectors' addresses once laid out,
   and its states (two an index) */
struct mesh_routing {
  struct mesh_calls *calls;
  struct mesh_routing *next;
  uint32_t worker,candidates,consumers,metas,vector_meta[3];
  struct mesh_route_vector vector[3];
  struct mesh_section meta[3];
  struct mesh_section *candidate;
  struct mesh_function **consumer,**producer;
  const uint32_t **vectors;
  struct mesh_route_state *state;
  uint8_t *arrived;
  uint32_t *remaining;
};
/* a function's active binding (mesh_call_active): the count's vector and the function's slot */
struct mesh_active { struct mesh_active *next; struct mesh_function *function; struct mesh_route_vector count; uint32_t slot; };
/* an arrival a route takes: a directory section's (`which` its meta) or a candidate's (`which` the candidate), of the
   generation's index `index`; its arrival cell's or landing use's call is the program's call records' count plus its
   number (mesh_calls `route_base`) */
enum { MESH_ROUTE_META, MESH_ROUTE_CANDIDATE };
struct mesh_route_event { struct mesh_routing *route; uint32_t kind,which,index; };
/* an omitted call's publication, its arrival's value marked */
#define MESH_OMITTED (UINT64_C(1)<<63)
/* hot-path-loads M24 (2b49f3c6^) */
struct mesh_calls {
  struct mesh_ctx *context,*constants;
  struct mesh_function *functions;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,function_count;
  unsigned char *values;
  size_t stride;
  struct mesh_instance *instances;
  struct mesh_submission *submissions;
  _Atomic uint32_t references;
  _Atomic int running;
  void *owner;
  mesh_dispose dispose;
  struct mesh_watch *watches;
  uint32_t watch_count,watch_room,words_left,produced_count;
  _Atomic uint64_t *words;
  struct mesh_produced *produced;
  struct prepared_publication *forwards;
  /* design/bridge-dataflow.md#routed-readiness: its routes and active bindings, and the route events (`route_base`
     the first call index that names one; UINT32_MAX where it has no route) */
  struct mesh_routing *routes;
  struct mesh_active *actives;
  struct mesh_route_event *route_events;
  uint32_t route_base,route_event_count;
  /* design/bridge-and-ipc.md#programs: the destinations a landing found cancelled (a bit an identity), whose every other
     landing is lost with it: its bridge cancels the positions past a channel's landed count alone */
  _Atomic uint64_t lost[2*MESH_DESTINATIONS/64];
};

/* design/algorithm-sources.md#programtensor: the arrival cells, completion records and input words are leases of the
   client's slot, returned with it (mesh_detach) */
static void mesh_calls_release(struct mesh_calls *calls,uint32_t references){
  if(atomic_fetch_sub_explicit(&calls->references,references,memory_order_acq_rel)!=references)return;
  for(struct mesh_function *function=calls->functions;function;){
    struct mesh_function *next=function->next;
    free(function->operands);free(function->inputs);free(function);
    function=next;
  }
  for(uint32_t i=0;i<MESH_COMPUTE_THREADS;i++){free(calls->workers[i].landings);free(calls->workers[i].uses);}
  for(struct mesh_routing *route=calls->routes;route;){
    struct mesh_routing *next=route->next;
    free(route->candidate);free(route->consumer);free(route->producer);free(route->vectors);free(route->state);
    free(route->arrived);free(route->remaining);free(route);
    route=next;
  }
  for(struct mesh_active *active=calls->actives;active;){struct mesh_active *next=active->next;free(active);active=next;}
  free(calls->route_events);
  free(calls->watches);free(calls->produced);free(calls->forwards);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls->submissions);free(calls->instances);free(calls->values);free(calls);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,struct mesh_ctx *constants,uint32_t workers,uint32_t count,
                                     void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS || !count){errno=EINVAL;return NULL;}
  /* hot-path-loads M24 */
  struct mesh_calls *calls=aligned_alloc(_Alignof(struct mesh_calls),sizeof *calls);
  if(!calls)return NULL;
  memset(calls,0,sizeof *calls);
  calls->context=context;calls->constants=constants && constants->M?constants:NULL;
  calls->count=workers;calls->extent=count;atomic_init(&calls->references,1);calls->route_base=UINT32_MAX;
  /* hot-path-loads M22 */
  for(uint32_t i=0;i<workers;i++)calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i};
  /* hot-path-loads M41, M42: the client's own memory */
  calls->instances=aligned_alloc(_Alignof(struct mesh_instance),(size_t)count*sizeof *calls->instances);
  calls->submissions=aligned_alloc(_Alignof(struct mesh_submission),(size_t)count*sizeof *calls->submissions);
  if(!calls->instances || !calls->submissions){mesh_calls_release(calls,1);errno=ENOMEM;return NULL;}
  memset(calls->instances,0,(size_t)count*sizeof *calls->instances);
  for(uint32_t i=0;i<count;i++){
    calls->submissions[i]=(struct mesh_submission){.generation=i,.stride=count};
    atomic_store_explicit(&calls->instances[i].status,((struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}),memory_order_relaxed);
  }
  calls->owner=owner;calls->dispose=dispose;
  return calls;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_function *mesh_call_bind(struct mesh_calls *calls,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,size_t dependency_count,
  const struct mesh_section *outputs,size_t output_count,const void *submit,void *argument){
  if(worker>=calls->count || dependency_count>input_count || input_count>UINT32_MAX || output_count>UINT32_MAX-input_count || atomic_load(&calls->running)){errno=EINVAL;return NULL;}
  /* hot-path-loads M23 */
  struct mesh_function *function=calloc(1,sizeof *function);
  if(!function)return NULL;
  size_t count=input_count+output_count;
  function->inputs=calloc(count?count:1,sizeof *inputs);
  if(!function->inputs){free(function);return NULL;}
  memcpy(function->inputs,inputs,input_count*sizeof *inputs);
  memcpy(function->inputs+input_count,outputs,output_count*sizeof *outputs);
  function->calls=calls;function->worker=worker;function->input_count=input_count;function->output_count=output_count;
  function->dependency_count=dependency_count;
  function->submit=(mesh_invoke)submit;function->argument=argument;
  function->identity=calls->function_count++;
  function->next=calls->functions;calls->functions=function;
  return function;
}

/* design/bridge-dataflow.md#routed-readiness (mesh 45902d8e^ mesh_algebra_route_create): a vector of `length` uint32
   lies in its section's bytes */
static int mesh_route_vector_fits(struct mesh_route_vector v,uint64_t length){
  return v.section.count && v.offset<=v.section.bytes && length<=(v.section.bytes-v.offset)/sizeof(uint32_t) && !(v.offset%sizeof(uint32_t));
}
struct mesh_routing *mesh_route_create(struct mesh_calls *calls,uint32_t worker,struct mesh_route_vector owners,struct mesh_route_vector ordinals,
  struct mesh_route_vector offsets,const struct mesh_section *candidates,size_t candidate_count,size_t consumers){
  if(!calls || atomic_load(&calls->running) || worker>=calls->count || !candidates || !candidate_count || !consumers ||
     candidate_count>=UINT32_MAX || consumers>=UINT32_MAX-1 || !mesh_route_vector_fits(owners,candidate_count) ||
     !mesh_route_vector_fits(ordinals,candidate_count) || !mesh_route_vector_fits(offsets,(uint64_t)consumers+1)){errno=EINVAL;return NULL;}
  const struct mesh_route_vector vectors[3]={owners,ordinals,offsets};
  /* each generation writes its own directory and candidates: a section of a slot each index */
  for(int v=0;v<3;v++)if(!vectors[v].section.stride || vectors[v].section.count<calls->extent){errno=EINVAL;return NULL;}
  for(size_t k=0;k<candidate_count;k++)if(!candidates[k].stride || candidates[k].count<calls->extent){errno=EINVAL;return NULL;}
  struct mesh_routing *d=calloc(1,sizeof *d);
  if(!d)return NULL;
  *d=(struct mesh_routing){.calls=calls,.worker=worker,.candidates=(uint32_t)candidate_count,.consumers=(uint32_t)consumers};
  d->candidate=malloc(candidate_count*sizeof *d->candidate);
  d->consumer=calloc(consumers,sizeof *d->consumer);d->producer=calloc(candidate_count,sizeof *d->producer);
  if(!d->candidate || !d->consumer || !d->producer){free(d->candidate);free(d->consumer);free(d->producer);free(d);errno=ENOMEM;return NULL;}
  memcpy(d->candidate,candidates,candidate_count*sizeof *candidates);
  for(int v=0;v<3;v++){
    d->vector[v]=vectors[v];
    uint32_t j=0;
    while(j<d->metas && d->meta[j].first!=vectors[v].section.first)j++;
    if(j==d->metas)d->meta[d->metas++]=vectors[v].section;
    d->vector_meta[v]=j;
  }
  d->next=calls->routes;calls->routes=d;
  return d;
}
int mesh_route_attach(struct mesh_function *function,struct mesh_routing *d,size_t consumer){
  if(!function || !d || function->calls!=d->calls || !function->submit || function->worker!=d->worker || consumer>=d->consumers ||
     d->consumer[consumer] || atomic_load(&d->calls->running))return EINVAL;
  d->consumer[consumer]=function;
  return 0;
}
int mesh_route_producers(struct mesh_routing *d,struct mesh_function *const *producers,size_t count){
  if(!d || !producers || count!=d->candidates || atomic_load(&d->calls->running))return EINVAL;
  for(size_t k=0;k<count;k++)if(producers[k] && producers[k]->calls!=d->calls)return EINVAL;
  memcpy(d->producer,producers,count*sizeof *producers);
  return 0;
}
/* mesh 45902d8e^ mesh_algebra_active, on the executor: the count a section the function waits on */
int mesh_call_active(struct mesh_function *function,struct mesh_route_vector count,uint32_t slot){
  if(!function || !function->submit || atomic_load(&function->calls->running) || !mesh_route_vector_fits(count,1))return EINVAL;
  int waited=0;
  for(size_t i=0;i<function->dependency_count;i++)waited|=function->inputs[i].first==count.section.first;
  for(const struct mesh_active *a=function->calls->actives;a;a=a->next)if(a->function==function)return EINVAL;
  if(!waited)return EINVAL;
  struct mesh_active *active=calloc(1,sizeof *active);
  if(!active)return ENOMEM;
  *active=(struct mesh_active){.next=function->calls->actives,.function=function,.count=count,.slot=slot};
  function->calls->actives=active;
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* hot-path-loads M37 */
uint64_t *mesh_function_completion(struct mesh_function *function,uint32_t frame){
  return (uint64_t *)&function->completions[frame].event;
}

/* a lease of the client's slot of `bytes`, zeroed; NULL (errno) where the window refuses it */
static void *mesh_calls_lease(struct mesh_calls *calls,size_t bytes){
  struct mesh_section section;
  int error=mesh_section_create(calls->context,bytes?bytes:1,1,&section);
  if(error){errno=error;return NULL;}
  void *address=mesh_section_address(calls->context,section,0);
  memset(address,0,(size_t)section.pages*calls->context->M->pgsz);
  return address;
}

/* prepared-machine.md M07: the landing word of invocation `t` of the section received at row `first` (its last chunk's,
   as mesh_host_arrived reads it); NULL where its program has no completion words (mesh_host_inputs) */
static _Atomic uint64_t *mesh_landing_word(struct hdr *m,uint32_t first,uint64_t bytes,uint32_t t){
  const struct mesh_delivery *delivery=mesh_delivery_at(m,first+mesh_row_chunks(m,first,bytes)-1);
  return delivery->device_input?(_Atomic uint64_t *)((char *)m+delivery->device_input+(uint64_t)t*delivery->device_stride):NULL;
}

/* design/algorithm-sources.md#device-operands */
uint64_t *mesh_section_input(struct mesh_calls *calls,struct mesh_section section,uint32_t index){
  struct hdr *m=calls->context->M;
  const uint32_t t=section.stride?index:0;
  _Atomic uint64_t *landed=mesh_landing_word(m,section.first,section.bytes,t);
  if(landed)return (uint64_t *)landed;
  for(uint32_t i=0;i<calls->watch_count;i++)
    if(calls->watches[i].first==section.first && calls->watches[i].index==t)return (uint64_t *)calls->watches[i].word;
  if(calls->watch_count==calls->watch_room){
    uint32_t room=calls->watch_room?2*calls->watch_room:16;
    struct mesh_watch *grown=realloc(calls->watches,room*sizeof *grown);
    if(!grown)return NULL;
    calls->watches=grown;calls->watch_room=room;
  }
  if(!calls->words_left){
    const size_t bytes=(size_t)m->block*m->pgsz;
    if(!(calls->words=mesh_calls_lease(calls,bytes)))return NULL;
    calls->words_left=(uint32_t)(bytes/sizeof(uint64_t));
  }
  _Atomic uint64_t *word=calls->words++;
  calls->words_left--;
  calls->watches[calls->watch_count++]=(struct mesh_watch){section.first,t,word};
  return (uint64_t *)word;
}

/* hot-path-loads M13 (2b49f3c6^): the publications of slot `index` of the section produced at row `first`: its
   consumers' arrival cells, its SEND cells of invocation `index` in either program (prepared-machine.md M10), its
   resident consumers' input words; each stored its argument plus the cycle at run time */
static uint32_t mesh_row_records(struct mesh_calls *calls,uint32_t first,uint32_t index,struct prepared_publication *out){
  uint32_t n=0,low=0,high=calls->produced_count;
  while(low<high){
    uint32_t middle=low+(high-low)/2;
    const struct mesh_produced *p=calls->produced+middle;
    if(p->first<first || (p->first==first && p->index<index))low=middle+1;else high=middle;
  }
  for(;low<calls->produced_count && calls->produced[low].first==first && calls->produced[low].index==index;low++,n++)
    if(out)out[n]=(struct prepared_publication){.destination=(uintptr_t)&calls->produced[low].cell->argument,.argument=1};
  struct mesh_ctx *programs[2]={calls->context,calls->constants};
  for(int p=0;p<2;p++){
    if(!programs[p])continue;
    uint32_t count=mesh_publication_prepare(programs[p],first,out?out+n:NULL);
    for(uint32_t j=0;out && j<count;j++)out[n+j].destination+=(uint64_t)(p?0:index)*sizeof(struct mesh_send);
    n+=count;
  }
  for(uint32_t i=0;i<calls->watch_count;i++)if(calls->watches[i].first==first && calls->watches[i].index==index){
    if(out)out[n]=(struct prepared_publication){.destination=(uintptr_t)calls->watches[i].word,.argument=1};
    n++;
  }
  return n;
}

/* design/algorithm-sources.md#resident-metal */
uint32_t mesh_function_publications(struct mesh_function *function,struct mesh_section section,uint32_t index,struct prepared_publication *records){
  return mesh_row_records(function->calls,section.first,index,records);
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_cleanup(struct mesh_call *,int);

/* design/algorithm-sources.md#meshresult: a landing read UINT64_MAX, its destination cancelled (prepared-machine.md
   M12), or another landing of its destination did: its instances concluded with the link's failure, the destination's
   code where the bridge gave one, and the destination marked lost (`lost`) */
static void mesh_landing_lost(struct mesh_calls *calls,const struct mesh_call_landing *landing){
  atomic_fetch_or_explicit(calls->lost+landing->identity/64,UINT64_C(1)<<(landing->identity%64),memory_order_release);
  struct mesh_ctx *program=landing->identity<MESH_DESTINATIONS?calls->context:calls->constants;
  const int64_t code=atomic_load_explicit(&mesh_client_at(program->M,program->client)->status[landing->identity%MESH_DESTINATIONS].code,memory_order_relaxed);
  for(uint32_t i=0;i<landing->instances;i++)
    mesh_result_conclude(&calls->instances[landing->slot+i].status,MESH_RESULT(MESH_RESULT_LINK,landing->identity,code?code:ECANCELED));
}

/* design/bridge-dataflow.md#routed-readiness: an active call's generation past its count, omitted (mesh 45902d8e^
   mesh_active_issue): nothing submitted, its publications (route candidates' arrivals alone) stored marked omitted, its
   generation's event completed */
static __attribute__((noinline)) void mesh_call_omit(struct mesh_call *call){
  const uint64_t cycle=call->cycle;
  for(uint32_t i=0;i<call->publication_count;i++)
    atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)call->publications[i].destination,(call->publications[i].argument+cycle)|MESH_OMITTED,
                          memory_order_release);
  mesh_instance_complete(call->instance,1);
}
/* hot-path-loads M18: a call whose arrivals all came, fired for `cycle`: submitted, or omitted where it is active and its
   slot is past its count */
static inline __attribute__((always_inline)) void mesh_call_fire(struct mesh_call *call,uint64_t cycle,uint32_t extent,uint32_t *submitted){
  call->pending=call->recurring;
  call->cycle=(uint32_t)cycle;
  call->generation=call->cycle*extent+call->index;
  if(call->count && atomic_load_explicit(call->count,memory_order_acquire)<=call->slot){mesh_call_omit(call);return;}
  call->submit(call,call->argument);
  ++*submitted;
}
/* a route's state laid out afresh for the next cycle of its index and parity */
static void mesh_route_reset(const struct mesh_routing *d,struct mesh_route_state *s){
  s->metas=s->candidates=s->delivered=s->prepared=0;
  memset(s->arrived,0,d->candidates);
}
/* consumer c's routed candidates all came: its call of the index fired where its other arrivals came too */
static void mesh_route_deliver(struct mesh_calls *calls,struct mesh_routing *d,struct mesh_route_state *s,uint32_t c,uint32_t index,
                               uint64_t cycle,uint32_t *submitted){
  struct mesh_call *call=(void *)(calls->values+((size_t)d->consumer[c]->identity*calls->extent+index)*calls->stride);
  s->delivered++;
  if(!--call->pending)mesh_call_fire(call,cycle,calls->extent,submitted);
}
/* a generation the route cannot serve as its directory states (a span past the candidates, a candidate in a span its
   owner is not, a routed candidate omitted): failed as a value naming the consumer's function */
static void mesh_route_refuse(struct mesh_calls *calls,const struct mesh_routing *d,uint32_t index){
  mesh_result_conclude(&calls->instances[index].status,MESH_RESULT(MESH_RESULT_FUNCTION,d->consumer[0]->identity,EPROTO));
}
/* mesh 45902d8e^ mesh_route_event (prepared): the index's directory read once its sections all came, checked (each
   consumer's span within the candidates, each ordinal of it owned by it, and each owned candidate in its owner's span),
   each consumer's routed candidates not yet come counted, and a consumer with none fired; a directory that breaks the
   rules fails the generation and fires every consumer */
static void mesh_route_prepare(struct mesh_calls *calls,struct mesh_routing *d,struct mesh_route_state *s,uint32_t index,uint64_t cycle,
                               uint32_t *submitted){
  const uint32_t *owners=d->vectors[(size_t)index*3],*ordinals=d->vectors[(size_t)index*3+1],*offsets=d->vectors[(size_t)index*3+2];
  int valid=1;
  uint32_t owned=0;
  for(uint32_t k=0;k<d->candidates;k++)owned+=owners[k]<d->consumers;
  uint32_t spanned=0;
  for(uint32_t c=0;c<d->consumers && valid;c++){
    const uint32_t first=offsets[c],end=offsets[c+1];
    if(first>end || end>d->candidates){valid=0;break;}
    spanned+=end-first;
    for(uint32_t j=first;j<end && valid;j++)valid=ordinals[j]<d->candidates && owners[ordinals[j]]==c;
  }
  if(valid && spanned!=owned)valid=0;
  s->prepared=1;
  if(!valid){
    mesh_route_refuse(calls,d,index);
    for(uint32_t c=0;c<d->consumers;c++)mesh_route_deliver(calls,d,s,c,index,cycle,submitted);
    return;
  }
  for(uint32_t c=0;c<d->consumers;c++){
    uint32_t remaining=0;
    for(uint32_t j=offsets[c];j<offsets[c+1];j++){
      const uint8_t arrived=s->arrived[ordinals[j]];
      if(!arrived)remaining++;
      else if(arrived==2)mesh_route_refuse(calls,d,index);
    }
    s->remaining[c]=remaining;
    if(!remaining)mesh_route_deliver(calls,d,s,c,index,cycle,submitted);
  }
}
/* mesh 45902d8e^ mesh_route_ready, _retire, _finish: an arrival a route takes, of a directory section or a candidate
   (marked omitted where its producer was), into its index's state of the arrival's cycle parity; a candidate that
   comes once its directory was read counts toward its owner alone (an unowned one is discharged as it comes); the state
   laid out afresh once the directory, every candidate and every consumer's firing are done */
static void mesh_route_arrival(struct mesh_calls *calls,uint32_t event,uint64_t value,uint32_t *submitted){
  const struct mesh_route_event *e=calls->route_events+event;
  struct mesh_routing *d=e->route;
  const uint64_t cycle=(value&~MESH_OMITTED)-1;
  struct mesh_route_state *s=d->state+(size_t)e->index*2+(cycle&1);
  if(e->kind==MESH_ROUTE_META){
    s->metas|=UINT32_C(1)<<e->which;
    if(s->metas==(UINT32_C(1)<<d->metas)-1)mesh_route_prepare(calls,d,s,e->index,cycle,submitted);
  }else{
    s->arrived[e->which]=value&MESH_OMITTED?2:1;
    s->candidates++;
    if(s->prepared){
      const uint32_t owner=d->vectors[(size_t)e->index*3][e->which];
      if(owner<d->consumers && s->delivered<d->consumers){
        if(value&MESH_OMITTED)mesh_route_refuse(calls,d,e->index);
        if(s->remaining[owner] && !--s->remaining[owner])mesh_route_deliver(calls,d,s,owner,e->index,cycle,submitted);
      }
    }
  }
  if(s->prepared && s->candidates==d->candidates && s->delivered==d->consumers)mesh_route_reset(d,s);
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  unsigned char *values=calls->values;
  const size_t stride=calls->stride;
  const uint32_t extent=calls->extent,route_base=calls->route_base;
  struct mesh_arrival *arrivals=worker->arrivals;
  uint32_t count=worker->arrival_count,cursor=0,submitted=0;
  struct mesh_call_landing *landings=worker->landings;
  const uint32_t landing_count=worker->landing_count,*uses=worker->uses;
  struct mesh_completion *returns=worker->returns;
  uint32_t return_count=worker->return_count;

  pthread_setname_np("mesh.numerical");
  /* hot-path-loads M25 */
  atomic_store_explicit(&worker->frame,(uintptr_t)__builtin_frame_address(0),memory_order_release);
  while(atomic_load_explicit(&calls->running,memory_order_acquire) || atomic_load_explicit(&worker->completed,memory_order_acquire)!=submitted){
    /* hot-path-loads M18: FIRE where a pending count reaches 0 */
    for(uint32_t visited=0;visited<count;visited++){
      struct mesh_arrival *cell=arrivals+cursor;
      if(++cursor==count)cursor=0;
      uint64_t value=atomic_load_explicit(&cell->argument,memory_order_acquire);
      if(!value)continue;
      atomic_store_explicit(&cell->argument,0,memory_order_relaxed);
      /* design/bridge-dataflow.md#routed-readiness */
      if(cell->call>=route_base){mesh_route_arrival(calls,cell->call-route_base,value,&submitted);continue;}
      /* hot-path-loads M20 */
      struct mesh_call *call=(void *)(values+(size_t)cell->call*stride);
      if(--call->pending)continue;
      mesh_call_fire(call,value-1,extent,&submitted);
    }
    /* prepared-machine.md M07, M30: PUBLISH(row) of a landed row, its use range walked; a pending one of a destination
       found lost, lost (mesh_landing_lost) */
    const uint64_t lost=atomic_load_explicit(calls->lost,memory_order_acquire)|atomic_load_explicit(calls->lost+1,memory_order_acquire);
    for(uint32_t i=0;i<landing_count;i++){
      struct mesh_call_landing *landing=landings+i;
      if(!landing->word)continue;
      const uint64_t value=atomic_load_explicit(landing->word,memory_order_acquire);
      if(value<landing->expected){
        if(lost && atomic_load_explicit(calls->lost+landing->identity/64,memory_order_relaxed)>>(landing->identity%64)&1){
          mesh_landing_lost(calls,landing);landing->word=NULL;
        }
        continue;
      }
      if(value==UINT64_MAX){mesh_landing_lost(calls,landing);landing->word=NULL;continue;}
      const uint64_t cycle=landing->expected++-1;
      /* prepared-machine.md M04, M30 */
      for(uint32_t j=0;j<landing->publication_count;j++)
        atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)landing->publications[j].destination,
                              landing->publications[j].argument+cycle,memory_order_release);
      for(uint32_t j=0;j<landing->count;j++){
        const uint32_t use=uses[landing->first+j];
        /* design/bridge-dataflow.md#routed-readiness */
        if(use>=route_base){mesh_route_arrival(calls,use-route_base,cycle+1,&submitted);continue;}
        struct mesh_call *call=(void *)(values+(size_t)use*stride);
        if(--call->pending)continue;
        mesh_call_fire(call,cycle,extent,&submitted);
      }
      if(landing->counted)mesh_instance_complete(calls->instances+landing->slot,landing->instances);
    }
    /* hot-path-loads M37 */
    for(uint32_t i=0;i<return_count;i++){
      struct mesh_completion *record=returns+i;
      if(!atomic_load_explicit(&record->event,memory_order_acquire))continue;
      atomic_store_explicit(&record->event,0,memory_order_relaxed);
      mesh_instance_complete(record->result,1);
    }
  }
  mesh_calls_release(calls,1);
  return NULL;
}

/* a program's receive transfers, each a source of landings: its rows, its stride (0: a constant's), its bytes and its
   destination (`identity`: + MESH_DESTINATIONS in the constants program) */
struct mesh_received { struct mesh_ctx *program; uint32_t first,stride,identity; uint64_t bytes; };
static int mesh_received_order(const void *a,const void *b){
  const struct mesh_received *x=a,*y=b;
  return (x->first>y->first)-(x->first<y->first);
}
static uint32_t mesh_receives(struct mesh_ctx *program,uint32_t base,struct mesh_received *out){
  uint32_t n=0;
  if(!program)return 0;
  struct hdr *m=program->M;
  const uint32_t destinations=mesh_destination(mesh_client_at(m,program->client),0);
  for(uint32_t d=0;d<destinations;d++)for(uint32_t q=0;q<MESH_QUEUES;q++){
    const uint32_t channel=d*MESH_QUEUES+q,length=atomic_load_explicit(mesh_order_length(m,program->client,channel,MESH_RECEIVE),memory_order_acquire);
    const struct mesh_transfer *in=length?mesh_transfers(m,program->client,channel,MESH_RECEIVE):NULL;
    for(uint32_t i=0;i<length;i++,n++)
      if(out)out[n]=(struct mesh_received){.program=program,.first=in[i].local_row,.stride=in[i].stride,.identity=base+d,.bytes=in[i].bytes};
  }
  return n;
}
static uint32_t mesh_received_find(const struct mesh_received *received,uint32_t n,uint32_t first){
  uint32_t low=0,high=n;
  while(low<high){uint32_t middle=low+(high-low)/2;if(received[middle].first<first)low=middle+1;else high=middle;}
  return low<n && received[low].first==first?low:n;
}

/* hot-path-loads M18: an input's arrival of one slot, by its source: the slot's submission (ROOT), a section this node's
   functions produce (PRODUCED, by its first row) or a received one (LANDING, by its receive transfer and slot) */
enum { MESH_USE_ROOT, MESH_USE_PRODUCED, MESH_USE_LANDING };
struct mesh_use { uint32_t worker,kind,key,index,call; };
static int mesh_use_group(const struct mesh_use *x,const struct mesh_use *y){
  if(x->worker!=y->worker)return (x->worker>y->worker)-(x->worker<y->worker);
  if(x->kind!=y->kind)return (x->kind>y->kind)-(x->kind<y->kind);
  if(x->key!=y->key)return (x->key>y->key)-(x->key<y->key);
  return (x->index>y->index)-(x->index<y->index);
}
static int mesh_use_order(const void *a,const void *b){
  const struct mesh_use *x=a,*y=b;
  int group=mesh_use_group(x,y);
  return group?group:(x->call>y->call)-(x->call<y->call);
}
static int mesh_produced_order(const void *a,const void *b){
  const struct mesh_produced *x=a,*y=b;
  if(x->first!=y->first)return (x->first>y->first)-(x->first<y->first);
  return (x->index>y->index)-(x->index<y->index);
}

/* hot-path-loads M22: each worker's arrival cells, landings and their uses, the call records and their publications,
   the resident completions, the instances; the constants sent */
static int mesh_calls_layout(struct mesh_calls *calls,struct mesh_received *received,uint32_t received_count,
                             uint32_t *landing_base,struct mesh_use *uses,size_t use_count,struct mesh_call_landing **designated){
  struct hdr *m=calls->context->M;
  const uint32_t extent=calls->extent,landing_total=landing_base[received_count];
  uint32_t arrivals[MESH_COMPUTE_THREADS]={0},landing_uses[MESH_COMPUTE_THREADS]={0},landing_runs[MESH_COMPUTE_THREADS]={0};
  size_t produced=0;
  for(size_t i=0;i<use_count;i++){
    const uint32_t w=uses[i].worker;
    if(uses[i].kind!=MESH_USE_LANDING){arrivals[w]++;produced+=uses[i].kind==MESH_USE_PRODUCED;continue;}
    landing_uses[w]++;
    if(!i || mesh_use_group(uses+i,uses+i-1))landing_runs[w]++;
  }
  if(produced>UINT32_MAX)return EOVERFLOW;
  calls->produced=malloc((produced?produced:1)*sizeof *calls->produced);
  if(!calls->produced)return ENOMEM;
  for(uint32_t w=0;w<calls->count;w++){
    struct mesh_call_worker *worker=calls->workers+w;
    if(arrivals[w] && !(worker->arrivals=mesh_calls_lease(calls,(size_t)arrivals[w]*sizeof *worker->arrivals)))return errno;
    worker->landings=calloc((size_t)landing_runs[w]+(w?0:landing_total)+1,sizeof *worker->landings);
    worker->uses=malloc(((size_t)landing_uses[w]+1)*sizeof *worker->uses);
    if(!worker->landings || !worker->uses)return ENOMEM;
  }
  for(size_t i=0,end;i<use_count;i=end){
    const struct mesh_use u=uses[i];
    for(end=i+1;end<use_count && !mesh_use_group(uses+end,&u);end++);
    struct mesh_call_worker *worker=calls->workers+u.worker;
    if(u.kind==MESH_USE_LANDING){
      const struct mesh_received *r=received+u.key;
      struct mesh_call_landing *landing=worker->landings+worker->landing_count++;
      *landing=(struct mesh_call_landing){.word=mesh_landing_word(r->program->M,r->first,r->bytes,u.index),.expected=1,
        .first=worker->use_count,.count=(uint32_t)(end-i),.slot=u.index,.instances=r->stride?1:extent,.identity=r->identity};
      for(size_t j=i;j<end;j++)worker->uses[worker->use_count++]=uses[j].call;
      if(!designated[landing_base[u.key]+u.index])designated[landing_base[u.key]+u.index]=landing;
      continue;
    }
    struct mesh_arrival *cells=worker->arrivals+worker->arrival_count;
    for(size_t j=i;j<end;j++){
      cells[j-i]=(struct mesh_arrival){.call=uses[j].call};
      if(u.kind==MESH_USE_PRODUCED)calls->produced[calls->produced_count++]=(struct mesh_produced){u.key,u.index,cells+(j-i)};
    }
    /* hot-path-loads M41: a slot's root arrivals, a range a worker */
    if(u.kind==MESH_USE_ROOT){
      struct mesh_submission *submission=calls->submissions+u.index;
      submission->destinations[submission->count]=cells;submission->counts[submission->count++]=(uint32_t)(end-i);
    }
    worker->arrival_count+=(uint32_t)(end-i);
  }
  qsort(calls->produced,calls->produced_count,sizeof *calls->produced,mesh_produced_order);

  /* each landing slot's one counting landing: its first worker's, else worker 0's alone; it forwards a relayed row */
  size_t forwards=0;
  for(uint32_t r=0;r<received_count;r++)
    forwards+=(size_t)mesh_publication_prepare(received[r].program,received[r].first,NULL)*(landing_base[r+1]-landing_base[r]);
  calls->forwards=malloc((forwards?forwards:1)*sizeof *calls->forwards);
  if(!calls->forwards)return ENOMEM;
  uint32_t varying=0,constant=0;
  size_t forward=0;
  for(uint32_t r=0;r<received_count;r++){
    for(uint32_t t=0;t<landing_base[r+1]-landing_base[r];t++){
      struct mesh_call_landing *landing=designated[landing_base[r]+t];
      if(!landing){
        struct mesh_call_worker *worker=calls->workers;
        landing=worker->landings+worker->landing_count++;
        *landing=(struct mesh_call_landing){.word=mesh_landing_word(received[r].program->M,received[r].first,received[r].bytes,t),
          .expected=1,.first=worker->use_count,.slot=t,.instances=received[r].stride?1:extent,.identity=received[r].identity};
      }
      landing->counted=1;
      landing->publications=calls->forwards+forward;
      landing->publication_count=mesh_publication_prepare(received[r].program,received[r].first,calls->forwards+forward);
      for(uint32_t j=0;received[r].stride && j<landing->publication_count;j++)
        calls->forwards[forward+j].destination+=(uint64_t)t*sizeof(struct mesh_send);
      forward+=landing->publication_count;
    }
    if(received[r].stride)varying++;else constant++;
  }

  /* hot-path-loads M13 (2b49f3c6^ mesh_events_prepare): the constants this node sends, published once; a relayed one
     is published as it lands */
  for(uint32_t d=0;calls->constants && d<mesh_destination(mesh_client_at(m,calls->constants->client),0);d++)
    for(uint32_t q=0;q<MESH_QUEUES;q++){
      const uint32_t channel=d*MESH_QUEUES+q,length=atomic_load(mesh_order_length(m,calls->constants->client,channel,MESH_SEND));
      const struct mesh_transfer *out=length?mesh_transfers(m,calls->constants->client,channel,MESH_SEND):NULL;
      for(uint32_t i=0;i<length;i++){
        if(mesh_received_find(received,received_count,out[i].local_row)<received_count)continue;
        uint32_t count=mesh_publication_prepare(calls->constants,out[i].local_row,NULL);
        struct prepared_publication stores[count?count:1];
        mesh_publication_prepare(calls->constants,out[i].local_row,stores);
        for(uint32_t j=0;j<count;j++)atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)stores[j].destination,stores[j].argument,memory_order_release);
      }
    }

  /* hot-path-loads M20 */
  size_t publications=0,records=0;
  for(struct mesh_function *function=calls->functions;function;function=function->next)if(function->submit){
    if(records<=function->identity)records=(size_t)function->identity+1;
    for(uint32_t frame=0;frame<extent;frame++){
      size_t count=0;
      for(size_t i=0;i<function->output_count;i++)count+=mesh_row_records(calls,function->inputs[function->input_count+i].first,frame,NULL);
      if(count>publications)publications=count;
    }
  }
  calls->stride=(sizeof(struct mesh_call)+publications*sizeof(struct prepared_publication)+127)&~(size_t)127;
  calls->values=records?aligned_alloc(128,records*extent*calls->stride):NULL;
  if(records && !calls->values)return ENOMEM;
  uint32_t returns[MESH_COMPUTE_THREADS]={0};
  for(struct mesh_function *function=calls->functions;function;function=function->next)if(!function->submit)returns[function->worker]+=extent;
  for(uint32_t w=0;w<calls->count;w++)
    if(returns[w] && !(calls->workers[w].returns=mesh_calls_lease(calls,(size_t)returns[w]*sizeof(struct mesh_completion))))return errno;
  for(struct mesh_function *function=calls->functions;function;function=function->next){
    /* hot-path-loads M37 */
    if(!function->submit){
      struct mesh_call_worker *worker=calls->workers+function->worker;
      function->completions=worker->returns+worker->return_count;
      worker->return_count+=extent;
      for(uint32_t slot=0;slot<extent;slot++)function->completions[slot]=(struct mesh_completion){.result=calls->instances+slot};
      continue;
    }
    size_t count=function->input_count+function->output_count;
    function->values=calls->values+(size_t)function->identity*extent*calls->stride;
    /* hot-path-loads M43 */
    function->operands=aligned_alloc(32,extent*(count?count:1)*sizeof *function->operands);
    if(!function->operands)return ENOMEM;
    for(uint32_t index=0;index<extent;index++){
      struct mesh_call *call=(void *)(function->values+index*calls->stride);
      *call=(struct mesh_call){.operands=function->operands+index*count,.index=index,
        .pending=function->initial,.recurring=function->pending,.identity=function->identity,
        .instance=calls->instances+index,.completed=&calls->workers[function->worker].completed,
        .submit=function->submit,.argument=function->argument,.generation=index};
      for(size_t i=0;i<count;i++){
        struct mesh_section section=function->inputs[i];
        uint32_t row=mesh_section_row(section,index);
        call->operands[i]=(struct mesh_call_operand){.data=(char *)m+atomic_load_explicit(&mesh_page(m)[row].address,memory_order_relaxed),
          .bytes=section.bytes,.index=section.stride?index:0,.row=row,.sequence=&call->generation};
        if(i<function->input_count)continue;
        /* hot-path-loads M13 */
        call->publication_count+=mesh_row_records(calls,section.first,index,call->publications+call->publication_count);
      }
    }
  }

  /* hot-path-loads M42: a generation's events, its calls and its landings (a SEND's completion is the bridge's alone);
     the first's also each constant's one landing */
  const uint32_t dynamic=calls->function_count+varying;
  for(uint32_t frame=0;frame<extent;frame++){
    calls->instances[frame].count=dynamic;calls->instances[frame].stride=extent;
    atomic_store_explicit(&calls->instances[frame].invocation,frame,memory_order_relaxed);
    atomic_store_explicit(&calls->instances[frame].remaining,dynamic+constant,memory_order_relaxed);
    if(!dynamic && !constant)atomic_store_explicit(&calls->instances[frame].status,((struct mesh_status){0,UINT64_MAX}),memory_order_relaxed);
  }
  atomic_thread_fence(memory_order_release);
  return 0;
}

/* design/bridge-dataflow.md#routed-readiness: whether a publication's destination is a route's arrival cell (a cell of a
   worker's arrivals whose call names a route event) */
static int mesh_route_cell(const struct mesh_calls *calls,uint64_t destination){
  for(uint32_t w=0;w<calls->count;w++){
    const struct mesh_call_worker *worker=calls->workers+w;
    if(!worker->arrival_count)continue;
    const uint64_t first=(uint64_t)(uintptr_t)&worker->arrivals[0].argument;
    if(destination<first || (destination-first)%sizeof(struct mesh_arrival) || (destination-first)/sizeof(struct mesh_arrival)>=worker->arrival_count)
      continue;
    return worker->arrivals[(destination-first)/sizeof(struct mesh_arrival)].call>=calls->route_base;
  }
  return 0;
}
/* design/bridge-dataflow.md#routed-readiness, once the call records are laid out: each route's vectors' addresses an
   index and its states (two an index, laid out afresh); each active function's call records given their count's word
   and slot, every publication of theirs a route's arrival (else EINVAL: an omitted call publishes those alone) */
static int mesh_routes_layout(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  const uint32_t extent=calls->extent;
  for(struct mesh_routing *d=calls->routes;d;d=d->next){
    d->vectors=malloc((size_t)extent*3*sizeof *d->vectors);
    d->state=calloc((size_t)extent*2,sizeof *d->state);
    d->arrived=calloc((size_t)extent*2*d->candidates,1);
    d->remaining=calloc((size_t)extent*2*d->consumers,sizeof *d->remaining);
    if(!d->vectors || !d->state || !d->arrived || !d->remaining)return ENOMEM;
    for(uint32_t index=0;index<extent;index++){
      for(int v=0;v<3;v++){
        const struct mesh_route_vector vector=d->vector[v];
        const uint32_t row=vector.section.first+index*vector.section.stride;
        d->vectors[(size_t)index*3+v]=(const uint32_t *)((char *)m+atomic_load_explicit(&mesh_page(m)[row].address,memory_order_relaxed)+vector.offset);
      }
      for(uint32_t parity=0;parity<2;parity++){
        struct mesh_route_state *s=d->state+(size_t)index*2+parity;
        s->arrived=d->arrived+((size_t)index*2+parity)*d->candidates;
        s->remaining=d->remaining+((size_t)index*2+parity)*d->consumers;
        mesh_route_reset(d,s);
      }
    }
  }
  for(const struct mesh_active *a=calls->actives;a;a=a->next){
    const struct mesh_function *function=a->function;
    for(uint32_t index=0;index<extent;index++){
      struct mesh_call *call=(void *)(function->values+(size_t)index*calls->stride);
      const uint32_t row=a->count.section.first+(a->count.section.stride?index:0)*a->count.section.stride;
      call->count=(const _Atomic uint32_t *)((char *)m+atomic_load_explicit(&mesh_page(m)[row].address,memory_order_relaxed)+a->count.offset);
      call->slot=a->slot;
      for(uint32_t i=0;i<call->publication_count;i++)if(!mesh_route_cell(calls,call->publications[i].destination))return EINVAL;
    }
  }
  return 0;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_prepare(struct mesh_calls *calls){
  if(atomic_load(&calls->running))return EINVAL;
  const uint32_t extent=calls->extent;
  /* prepared-machine.md M07: the landings' sources, every receive transfer of the two programs, by row */
  const uint32_t received_count=mesh_receives(calls->context,0,NULL)+mesh_receives(calls->constants,MESH_DESTINATIONS,NULL);
  struct mesh_received *received=malloc((received_count+(size_t)1)*sizeof *received);
  uint32_t *landing_base=malloc((received_count+(size_t)1)*sizeof *landing_base);
  if(!received || !landing_base){free(received);free(landing_base);return ENOMEM;}
  mesh_receives(calls->constants,MESH_DESTINATIONS,received+mesh_receives(calls->context,0,received));
  qsort(received,received_count,sizeof *received,mesh_received_order);
  int error=0;
  landing_base[0]=0;
  for(uint32_t r=0;r<received_count;r++){
    landing_base[r+1]=landing_base[r]+(received[r].stride?extent:1);
    if(!mesh_landing_word(received[r].program->M,received[r].first,received[r].bytes,0))error=EINVAL;
  }
  /* design/bridge-dataflow.md#routed-readiness: every route's consumers attached, and its events (each directory section
     and each candidate an arrival a slot) numbered past the call records */
  uint64_t route_events=0;
  for(struct mesh_routing *d=calls->routes;d;d=d->next){
    for(uint32_t c=0;c<d->consumers;c++)if(!d->consumer[c])error=EINVAL;
    /* a candidate's producer (mesh_route_producers) is active and writes it, as mesh_algebra_route_producers asked */
    for(uint32_t k=0;k<d->candidates;k++)if(d->producer[k]){
      const struct mesh_function *producer=d->producer[k];
      int active=0,covers=0;
      for(const struct mesh_active *a=calls->actives;a;a=a->next)active|=a->function==producer;
      for(size_t o=0;o<producer->output_count;o++)covers|=producer->inputs[producer->input_count+o].first==d->candidate[k].first;
      if(!active || !covers)error=EINVAL;
    }
    route_events+=(uint64_t)(d->metas+d->candidates)*extent;
  }
  free(calls->route_events);calls->route_events=NULL;calls->route_event_count=0;
  calls->route_base=calls->routes?calls->function_count*extent:UINT32_MAX;
  if(calls->routes && (uint64_t)calls->function_count*extent+route_events>=UINT32_MAX)error=EOVERFLOW;
  if(!error && route_events && !(calls->route_events=malloc((size_t)route_events*sizeof *calls->route_events)))error=ENOMEM;
  /* hot-path-loads M18: each waited input of each function an arrival a slot, and a function with no varying input the
     slot's submission's; a constant of this node's (stride 0, received by none) is ready at start */
  size_t room=(size_t)route_events,use_count=0;
  for(struct mesh_function *function=calls->functions;function;function=function->next)
    if(function->submit)room+=(function->dependency_count+1)*(size_t)extent;
  struct mesh_use *uses=malloc((room?room:1)*sizeof *uses);
  struct mesh_call_landing **designated=calloc(landing_base[received_count]+(size_t)1,sizeof *designated);
  if(!uses || !designated)error=ENOMEM;
  for(struct mesh_function *function=calls->functions;!error && function;function=function->next){
    if(!function->submit)continue;
    uint32_t pending=0,initial=0;int varying=0;
    for(size_t i=0;i<function->dependency_count;i++){
      struct mesh_section section=function->inputs[i];
      varying|=section.stride!=0;
      const uint32_t r=mesh_received_find(received,received_count,section.first);
      if(r==received_count && !section.stride)continue;
      initial++;pending+=section.stride!=0;
      for(uint32_t index=0;index<extent;index++){
        const uint32_t call=function->identity*extent+index;
        uses[use_count++]=r<received_count?(struct mesh_use){function->worker,MESH_USE_LANDING,r,section.stride?index:0,call}
                                          :(struct mesh_use){function->worker,MESH_USE_PRODUCED,section.first,index,call};
      }
    }
    /* design/bridge-dataflow.md#routed-readiness: each route it owns a part of, one arrival a generation */
    uint32_t routed=0;
    for(const struct mesh_routing *d=calls->routes;d;d=d->next)for(uint32_t c=0;c<d->consumers;c++)routed+=d->consumer[c]==function;
    initial+=routed;pending+=routed;varying|=routed!=0;
    if(!varying){
      initial++;pending++;
      for(uint32_t index=0;index<extent;index++)
        uses[use_count++]=(struct mesh_use){function->worker,MESH_USE_ROOT,0,index,function->identity*extent+index};
    }
    function->pending=pending;function->initial=initial;
  }
  /* design/bridge-dataflow.md#routed-readiness: a route's arrivals, on its worker: each directory section's and each
     candidate's, of every index, received (a landing) or produced here (a cell its producer's publications store) */
  uint32_t event=0;
  for(struct mesh_routing *d=calls->routes;!error && d;d=d->next)
    for(uint32_t j=0;j<d->metas+d->candidates;j++){
      const struct mesh_section section=j<d->metas?d->meta[j]:d->candidate[j-d->metas];
      const uint32_t r=mesh_received_find(received,received_count,section.first);
      for(uint32_t index=0;index<extent;index++,event++){
        calls->route_events[event]=(struct mesh_route_event){d,j<d->metas?MESH_ROUTE_META:MESH_ROUTE_CANDIDATE,j<d->metas?j:j-d->metas,index};
        const uint32_t call=calls->route_base+event;
        uses[use_count++]=r<received_count?(struct mesh_use){d->worker,MESH_USE_LANDING,r,index,call}
                                          :(struct mesh_use){d->worker,MESH_USE_PRODUCED,section.first,index,call};
      }
    }
  calls->route_event_count=event;
  if(!error){
    qsort(uses,use_count,sizeof *uses,mesh_use_order);
    error=mesh_calls_layout(calls,received,received_count,landing_base,uses,use_count,designated);
  }
  if(!error)error=mesh_routes_layout(calls);
  free(received);free(landing_base);free(uses);free(designated);
  return error;
}

/* hot-path-loads M17 */
/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls){
  uint32_t workers=0;
  for(uint32_t i=0;i<calls->count;i++)
    if(calls->workers[i].arrival_count || calls->workers[i].return_count || calls->workers[i].landing_count)workers|=UINT32_C(1)<<i;
  atomic_store_explicit(&calls->running,1,memory_order_release);
  atomic_fetch_add_explicit(&calls->references,(uint32_t)__builtin_popcount(workers),memory_order_relaxed);
  while(workers){
    uint32_t i=(uint32_t)__builtin_ctz(workers);
    pthread_t thread;
    int error=pthread_create(&thread,NULL,mesh_call_progress,&calls->workers[i]);
    if(error){
      atomic_store(&calls->running,0);
      mesh_calls_release(calls,(uint32_t)__builtin_popcount(workers));return error;
    }
    /* hot-path-loads M25 */
    while(!atomic_load_explicit(&calls->workers[i].frame,memory_order_acquire)){}
    workers&=workers-1;
    pthread_detach(thread);
  }
  return 0;
}

/* hot-path-loads M41 */
/* design/algorithm-sources.md#program */
struct mesh_submission *mesh_submission_at(struct mesh_calls *calls,uint32_t slot,struct mesh_instance **instance){
  *instance=calls->instances+slot;
  return calls->submissions+slot;
}

/* design/algorithm-sources.md#meshresult */
uint32_t mesh_calls_node(struct mesh_calls *calls,uint32_t identity){
  struct mesh_ctx *program=identity<MESH_DESTINATIONS?calls->context:calls->constants;
  return program && identity<2*MESH_DESTINATIONS?mesh_client_at(program->M,program->client)->tx[identity%MESH_DESTINATIONS].node:0;
}

/* design/algorithm-sources.md#programkernel_call */
static __attribute__((noinline)) void mesh_call_cleanup(struct mesh_call *call,int error){
  if(error)mesh_result_conclude(&call->instance->status,MESH_RESULT(MESH_RESULT_FUNCTION,call->identity,error));
  else mesh_instance_complete(call->instance,1);
  atomic_fetch_add_explicit(call->completed,1,memory_order_release);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call,int error){
  if(!error){
    /* hot-path-loads M13; prepared-machine.md M30: the prepared argument plus the generation's cycle */
    const uint64_t cycle=call->cycle;
    #pragma clang loop unroll(disable)
    for(uint32_t i=0,count=call->publication_count;i<count;i++){
      struct prepared_publication store=call->publications[i];
      atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)store.destination,store.argument+cycle,memory_order_release);
    }
  }
  __attribute__((musttail)) return mesh_call_cleanup(call,error);
}

/* design/algorithm-sources.md#meshresult */
void mesh_call_fail(struct mesh_call *call,int error){mesh_call_complete(call,error?error:EIO);}

/* design/algorithm-sources.md#programkernel_call */
void mesh_calls_destroy(struct mesh_calls *calls){
  if(!calls)return;
  atomic_store_explicit(&calls->running,0,memory_order_release);
  mesh_calls_release(calls,1);
}

#include <signal.h>
#include "mesh-dataflow.h"
/* mesh_transfers_start's declaration: defined here, beside the map reader it reads */
#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
/* tag pre-return-20261103 etc/bridge/layout.h MESH_PLATFORM_NS, which the tag's libmesh reached through its mesh.h: the
   bound of libmesh's waits (the map read, the client's start).  Here until mesh.h carries it: mesh-bridge.h holds it
   for the bridge, and no client includes mesh-bridge.h.  The same tokens, so either definition stands beside the other */
#ifndef MESH_PLATFORM_NS
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#endif
/* design/pages-and-functions.md#what-the-page-table-is */
static void mesh_reclaim(struct hdr *);

/* design/algorithm-sources.md#meshobserve */
int mesh_observe(const char *name,struct mesh_link_view *out,uint32_t capacity,uint32_t *node){
  int file=shm_open(name?name:MESH_NAME,O_RDONLY,0);
  if(file<0)return -errno;
  struct stat info;
  if(fstat(file,&info)){int error=errno;close(file);return -error;}
  if(info.st_size<(off_t)sizeof(struct hdr)){close(file);return -EINVAL;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ,MAP_SHARED,file,0);
  int error=errno;close(file);
  if(m==MAP_FAILED)return -error;
  int result=-EINVAL;
  if(m->magic==MESH_MAGIC && m->version==MESH_VERSION && m->length<=(uint64_t)info.st_size &&
     m->link_off>=sizeof *m && m->link_off<=m->length && m->links<=INT_MAX && m->links<=(m->length-m->link_off)/sizeof(struct mesh_link_info)){
    *node=m->node;result=(int)m->links;
    for(uint32_t i=0;i<m->links && i<capacity;i++){
      struct mesh_link_info *link=&mesh_links(m)[i];
      out[i]=(struct mesh_link_view){.peer=link->peer,.phase=atomic_load_explicit(&link->port.phase,memory_order_acquire)};
      out[i].bandwidth=__atomic_load_n(&link->bandwidth,__ATOMIC_RELAXED);
      memcpy(out[i].device,link->device,sizeof out[i].device);out[i].device[sizeof out[i].device-1]=0;
    }
  }
  munmap(m,(size_t)info.st_size);return result;
}

/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *m,uint64_t client){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff);
  uint64_t retiring=(client&(UINT64_C(1)<<63))|(generation<<32)|(uint32_t)getpid();
  if(!atomic_compare_exchange_strong_explicit(&m->client,&client,retiring,memory_order_seq_cst,memory_order_acquire))return;
  atomic_store_explicit(&m->configured,0,memory_order_release);
  for(uint32_t q=0;q<m->links*m->qps;q++)for(int d=0;d<2;d++)atomic_store_explicit(mesh_order_length(m,client,q,d),0,memory_order_release);
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==client){
      if(buffer->pages)atomic_store_explicit(&buffer->closed,1,memory_order_release);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  atomic_store_explicit(&m->client,0,memory_order_release);
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
}

int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  if(!name) name=getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int file=shm_open(name,O_RDWR,MESH_MODE);
  if(file<0) return errno;
  struct stat info;
  if(fstat(file,&info)){ int error=errno; close(file); return error; }
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  if(memory==MAP_FAILED){ close(file); return error; }
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION || memory->length>(uint64_t)info.st_size){
    munmap(memory,(size_t)info.st_size); close(file); return EINVAL;
  }
  uint64_t client=(((atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff))<<32)|(uint32_t)getpid(),vacant=0;
  while(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,client,memory_order_seq_cst,memory_order_acquire)){
    /* an earlier client whose process is gone (mesh-holders.h mesh_gone, main's kill/ESRCH test) is retired */
    if(!vacant || !mesh_gone((pid_t)(uint32_t)vacant)){ munmap(memory,(size_t)info.st_size); close(file); return EADDRINUSE; }
    mesh_retire(memory,vacant);
    vacant=0;
  }
  /* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_slot_take:284, the slot's start as hdr `client_start`: with
     `client`'s pid, this client's identity as a holder of the region (mesh-holders.h mesh_holds) */
  memory->client_start=mesh_started(getpid());
  uint64_t device=atomic_load_explicit(&memory->device_client,memory_order_seq_cst);
  client|=(~device)&(UINT64_C(1)<<63);
  for(uint32_t q=0;q<memory->links*memory->qps;q++)for(int d=0;d<2;d++)atomic_store_explicit(mesh_order_length(memory,client,q,d),0,memory_order_relaxed);
  atomic_store_explicit(&memory->client,client,memory_order_release);
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file};
  /* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_attach: the expiry (a store, seq_cst) read again after the
     client's compare-exchange (seq_cst): a client the heartbeat's read of the holders missed reads it expired here,
     and is let go */
  if(__atomic_load_n(&memory->version,__ATOMIC_SEQ_CST)!=MESH_VERSION){mesh_detach(c);return EINVAL;}
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire(c->M,c->client);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programkernel_call */
/* One first-fit scan over an explicit pair of bitmaps and an explicit range.  The caller names the
   bitmaps because the arena bitmap is indexed by arena page and the row planes by row: they are two
   index spaces and sharing one sizing is what forced rows >= pages. */
static uint32_t mesh_scan(uint32_t count,uint32_t align,uint32_t begin,uint32_t end,
                          _Atomic uint64_t *own,_Atomic uint64_t *hot){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    uint32_t next=first;
    for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
      uint64_t occupied=(atomic_load_explicit(&own[w],memory_order_acquire)|atomic_load_explicit(&hot[w],memory_order_acquire))&mesh_word_mask(first,count,w);
      if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
    }
    if(next==first){ mesh_bits_set(own,first,count); return first; }
    first=(next+align-1)/align*align;
  }
  return MESH_ABSENT;
}

/* design/algorithm-sources.md#programkernel_call */
/* The cursor resumes where the last allocation ended, so preparing N operands over P pages costs
   O(P) in total rather than O(N*P); a full scan and then a reclaim still back it, so the result is
   the same page the unresumed scan would have returned whenever nothing has been released. */
static uint32_t mesh_allocate(struct mesh_ctx *c,uint32_t count,uint32_t align,uint32_t begin,uint32_t end,
                              _Atomic uint64_t *own,_Atomic uint64_t *hot,uint32_t *cursor){
  uint32_t resume=*cursor<begin||*cursor>end?begin:*cursor;
  uint32_t first=mesh_scan(count,align,resume,end,own,hot);
  if(first==MESH_ABSENT && resume!=begin)first=mesh_scan(count,align,begin,end,own,hot);
  if(first==MESH_ABSENT){ mesh_reclaim(c->M); first=mesh_scan(count,align,begin,end,own,hot); }
  if(first==MESH_ABSENT){ errno=ENOMEM; return MESH_ABSENT; }
  *cursor=first+count;
  return first;
}

uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,count,1,0,mesh_rows(c->M),mesh_plane(c->M,MESH_ROW_OWN),mesh_plane(c->M,MESH_ROW_HOT),&c->row_cursor);
  if(first==MESH_ABSENT) return first;
  for(uint32_t r=first;r<first+count;r++){
    atomic_store_explicit(&mesh_page(c->M)[r].device,0,memory_order_relaxed);
    atomic_store_explicit(&mesh_page(c->M)[r].mapping,MESH_ABSENT,memory_order_release);
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[r];
    atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
    buffer->pages=buffer->constant=0;
    struct mesh_publication *publication=mesh_publication_at(c->M,r);
    publication->sends=publication->device_input=publication->device_stride=0;
    atomic_store_explicit(&publication->argument,0,memory_order_relaxed);
    atomic_store_explicit(&buffer->owner,c->client,memory_order_release);
  }
  c->rows+=count;
  return first;
}

/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
/* Two ranges over one arena and one bitmap: [0,wire_pages) is registered and is the only memory an
   SGE may name; [wire_pages,arena) is addressable by every client and by the GPU and is never
   registered, so it costs virtual memory and neither wired pages nor an MR slot. */
uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align,int wire){
  struct hdr *m=c->M;
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  struct mesh_range range=mesh_arena_range(m,wire);
  _Atomic uint64_t *bits=mesh_arena_bits(m);
  uint32_t first=mesh_allocate(c,pages,align,range.first,range.first+range.count,bits,bits,
    wire?&c->wire_cursor:&c->bulk_cursor);
  if(first!=MESH_ABSENT){ c->arena+=pages; if(wire)c->wire+=pages; }
  return first;
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page,uint32_t index){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    struct mesh_page_entry *entry=&mesh_page(c->M)[first+offset/c->M->block];
    atomic_store_explicit(&entry->mapping,((uint64_t)index<<32)|(page+offset),memory_order_relaxed);
    atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(c->M,page+offset)-(uintptr_t)c->M,memory_order_relaxed);
  }
}

/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *c,uint32_t row,uint64_t address){
  if(row!=MESH_ABSENT)atomic_store_explicit(&mesh_page(c->M)[row].device,address,memory_order_relaxed);
}

/* design/algorithm-sources.md#programtensor */
static void mesh_buffer_reclaim(struct hdr *m,uint32_t row){
  struct mesh_buffer *buffer=&mesh_buffers(m)[row];
  uint64_t owner=atomic_load_explicit(&buffer->owner,memory_order_acquire);
  if(!owner || !atomic_compare_exchange_strong_explicit(&buffer->owner,&owner,0,memory_order_acq_rel,memory_order_relaxed))return;
  atomic_fetch_and_explicit(&mesh_plane(m,MESH_FREE)[row/64],~(UINT64_C(1)<<(row%64)),memory_order_relaxed);
  uint32_t pages=buffer->pages;
  for(uint32_t offset=0;offset<pages;offset+=m->block){
    uint32_t page=atomic_load_explicit(&mesh_page(m)[row+offset/m->block].mapping,memory_order_acquire);
    if(page!=MESH_ABSENT)mesh_bits_clear(mesh_arena_bits(m),page,m->block);
  }
  atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
  mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),row,pages/m->block);
}

/* design/algorithm-sources.md#programtensor */
static void mesh_reclaim(struct hdr *m){
  for(uint32_t word=0;word<mesh_words(m);word++){
    uint64_t available=atomic_load_explicit(&mesh_plane(m,MESH_FREE)[word],memory_order_acquire);
    while(available){
      uint32_t row=word*64+(uint32_t)__builtin_ctzll(available);available&=available-1;
      mesh_buffer_reclaim(m,row);
    }
  }
}

/* design/algorithm-sources.md#programtensor */
void mesh_retired_release(struct hdr *m){
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    uint64_t owner=atomic_load_explicit(&buffer->owner,memory_order_acquire);
    if(owner && atomic_load_explicit(&buffer->closed,memory_order_acquire) &&
       atomic_compare_exchange_strong_explicit(&buffer->owner,&owner,0,memory_order_acq_rel,memory_order_relaxed)){
      atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
      mesh_bits_set(mesh_plane(m,MESH_FREE),row,1);
      atomic_store_explicit(&buffer->owner,owner,memory_order_release);
    }
  }

}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(mesh_plane(c->M,MESH_ROW_OWN),first,count);
}

/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M18 */
/* its queue argument replaced by mesh_plan (mesh-collective.h); removed at sub-assignment 16 */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *c,uint32_t row,uint32_t queue){
  struct hdr *m=c->M;
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t *count=&publication->sends;
  struct mesh_target *targets=publication->targets;
  uint32_t destination=mesh_notice_queue(m,c->client,queue);
  uint64_t first=m->notice_off+(uint64_t)destination*m->notice_bytes;
  for(uint32_t i=0;i<*count;i++)if(targets[i].stream>=first && targets[i].stream<first+m->notice_bytes)return &targets[i];
  struct mesh_target *target=&targets[(*count)++];
  *target=(struct mesh_target){.stream=first};
  return target;
}

/* design/prepared-machine.md#M13 */
/* design/prepared-machine.md#M04 */
/* design/algorithm-sources.md#programkernel_call */
/* A row's release words as mesh_plan records them (mesh-collective.h): `targets[0].stream` the region offset of a send's
   release word of execution 0 (count 1, stride 1) and the publication's `argument` the `position` that releases it
   (struct mesh_hop), stored there at execution t's word, + t x hop_stride (sizeof(struct mesh_send)), by the GPU or
   mesh_host_publish.  The SEND cells' wr_id the replaced form stored is not read: a release word is one word, no cell */
uint32_t mesh_publication_prepare(struct hdr *m,uint32_t row,struct prepared_publication *records){
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t count=0;
  for(uint32_t i=0;i<publication->sends;i++){
    struct mesh_target target=publication->targets[i];
    for(uint32_t k=0;k<target.count;k++){
      if(records){
        char *word=(char *)m+target.stream+sizeof(struct mesh_send)*target.stride*k;
        records[count]=(struct prepared_publication){.destination=(uintptr_t)word,
          .argument=atomic_load_explicit(&publication->argument,memory_order_relaxed)};
      }
      count++;
    }
  }
  return count;
}

/* The map reader and the client's start (mesh-dataflow.h mesh_nodes_read, mesh-call.h mesh_transfers_start).  The
   map's records, a cable's both ends and its ports stay in this file: the collective is given node pairs and summed
   capacity (struct mesh_nodes), and the start a status. */

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_records_read, adapted: the map's records as one serial left them,
   the seqlock's reader (mesh.h mesh_map_whole), copied into r (MESH_NODES_MOST of room) with their count and the map's
   serial (hdr `map_serial`, moved only while `map` is odd); the read ended by `deadline` (the monotonic clock's ns),
   which its caller sets MESH_PLATFORM_NS past the start of its wait.  It differs from the tag's: no copy is allocated
   (the room is the reader's MESH_NODES_MOST, E2BIG past it); main has no expiry to register (mesh_expiry_set), so
   past the deadline it returns ETIMEDOUT, or ESRCH where the bridge's process is gone (its write left open); and an
   open write is waited on until the deadline, where the tag looked again each MESH_PAUSE_NS */
static int mesh_map_copy(struct mesh_ctx *c,uint64_t deadline,struct mesh_record *r,uint32_t *count,uint64_t *serial){
  struct hdr *m=c->M;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->map,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(now>=deadline)return mesh_gone((pid_t)atomic_load_explicit(&m->bridge_pid,memory_order_acquire))?ESRCH:ETIMEDOUT;
    if(before&1){
      os_sync_wait_on_address_with_timeout(&m->map,before,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
      continue;
    }
    const uint64_t nodes=atomic_load_explicit(&m->map_nodes,memory_order_relaxed),at=atomic_load_explicit(&m->map_off,memory_order_relaxed);
    *serial=atomic_load_explicit(&m->map_serial,memory_order_relaxed);
    const int fits=at>=sizeof *m && at<=c->len && nodes<=(c->len-at)/sizeof *r;
    if(fits && nodes<=MESH_NODES_MOST)memcpy(r,(char *)m+at,nodes*sizeof *r);
    if(!mesh_map_whole(m,before))continue;
    if(!fits)return EINVAL;
    if(nodes>MESH_NODES_MOST)return E2BIG;
    *count=(uint32_t)nodes;return 0;
  }
}

/* the records' indices in ascending node-id order: order[i] the record of map index i */
static void mesh_map_order(const struct mesh_record *r,uint32_t count,uint32_t *order){
  for(uint32_t i=0;i<count;i++){
    uint32_t k=i;
    for(;k && r[order[k-1]].node>r[i].node;k--)order[k]=order[k-1];
    order[k]=i;
  }
}

/* the cables between a's node and b's that both records list (mesh.h struct mesh_reach: each record lists a cable from
   its own side, so b's entry names a's node with the two ports swapped), each at the lesser of its two entries'
   bandwidths: their summed bandwidth (bits a second), and a's port of each in port[] where given (a's record lists at
   most MESH_PORTS).  A cable one record lists alone (the other's record not yet heard) is not counted: both bridges'
   rule reads a cable from both records, so neither end assigns it a page.  The tag's join (mesh_join) took the fewer
   of the two ends' counts at the least bandwidth of all; a cable here is matched by its ports, at its own bandwidth */
static uint64_t mesh_cables(const struct mesh_record *a,const struct mesh_record *b,uint32_t *port,uint32_t *cables){
  uint64_t sum=0;
  uint32_t n=0;
  for(uint32_t i=0;i<a->reached && i<MESH_PORTS;i++){
    const struct mesh_reach *e=&a->reach[i];
    if(e->node!=b->node)continue;
    for(uint32_t k=0;k<b->reached && k<MESH_PORTS;k++){
      const struct mesh_reach *f=&b->reach[k];
      if(f->node!=a->node || f->port!=e->peer_port || f->peer_port!=e->port)continue;
      sum+=e->bandwidth<f->bandwidth?e->bandwidth:f->bandwidth;
      if(port)port[n]=e->port;
      n++;
      break;
    }
  }
  if(cables)*cables=n;
  return sum;
}

int mesh_nodes_read(struct mesh_ctx *c,struct mesh_nodes *out){
  struct mesh_record r[MESH_NODES_MOST];
  uint32_t count,order[MESH_NODES_MOST];
  uint64_t serial;
  int error=mesh_map_copy(c,clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS,r,&count,&serial);
  if(error)return error;
  mesh_map_order(r,count,order);
  *out=(struct mesh_nodes){.serial=serial,.count=count,.self=MESH_ABSENT};
  for(uint32_t i=0;i<count;i++){
    out->node[i]=r[order[i]].node;
    if(out->node[i]==c->M->node)out->self=i;
    for(uint32_t j=0;j<i;j++)out->capacity[i][j]=out->capacity[j][i]=mesh_cables(r+order[i],r+order[j],NULL,NULL);
  }
  return out->self==MESH_ABSENT?ENOENT:0;
}

/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
/* The configured latch (hdr `configured`'s compare-exchange) and the unbounded spin on each link's `prepared` are gone:
   the plan is the bridge's to find (hdr `plan`, `hops`), and each port whose cable carries it stores `plan` in its
   `prepared` once its data path has loaded it, or sets MESH_STOPPED and `code` where it refuses it.  The ports waited
   on are this node's cables, in the map at the plan's serial, to the nodes the hops name (map indices, struct
   mesh_hop `node`), as mesh_cables matches them.  One deadline, MESH_PLATFORM_NS from the start, bounds the map's read
   and the ports' wait together.  A spin, bounded: `prepared` is stored with no wake.  A port stopped with no code
   returns ENOTCONN, so that a stop never reads as a start */
int mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  const uint64_t deadline=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
  const uint64_t hops=atomic_load_explicit(&m->hops,memory_order_acquire),plan=atomic_load_explicit(&m->plan,memory_order_relaxed);
  if(!hops)return 0;
  if(hops>m->hop_capacity)return EINVAL;
  struct mesh_record r[MESH_NODES_MOST];
  uint32_t count,order[MESH_NODES_MOST];
  uint64_t serial;
  int error=mesh_map_copy(context,deadline,r,&count,&serial);
  if(error)return error;
  if(serial!=plan)return ESTALE;
  mesh_map_order(r,count,order);
  uint32_t self=count;
  for(uint32_t i=0;i<count;i++)if(r[i].node==m->node)self=i;
  if(self==count)return ENOENT;
  uint64_t named=0;
  for(uint64_t k=0;k<hops;k++){
    const uint32_t node=mesh_hops(m)[k].node;
    if(node>=count)return EINVAL;
    named|=UINT64_C(1)<<node;
  }
  /* a call names at most MESH_PORTS of this node's cables, so this holds them even where two records named one node */
  uint32_t port[MESH_PORTS*MESH_NODES_MOST],ports=0;
  for(uint32_t j=0;j<count;j++)if(named>>j&1){
    uint32_t cables;
    mesh_cables(r+self,r+order[j],port+ports,&cables);
    ports+=cables;
  }
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t i=0;i<ports;i++){
    if(port[i]>=links)return EINVAL;
    struct mesh_port_info *info=&mesh_links(m)[port[i]].port;
    while(atomic_load_explicit(&info->prepared,memory_order_acquire)!=plan){
      if(atomic_load_explicit(&info->phase,memory_order_acquire)==MESH_STOPPED){
        const int64_t code=atomic_load_explicit(&info->code,memory_order_relaxed);
        return code?(int)code:ENOTCONN;
      }
      if(clock_gettime_nsec_np(CLOCK_MONOTONIC)>=deadline)return ETIMEDOUT;
    }
  }
  return 0;
}

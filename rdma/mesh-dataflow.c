#include <signal.h>
#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
/* design/pages-and-functions.md#what-the-page-table-is */
static void mesh_reclaim(struct hdr *);

/* design/algorithm-sources.md#meshobserve */
/* design/bridge-and-ipc.md#ports: each port of the node's bridge: its peer node while paired (0 else), its phase, its
   device and bandwidth */
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
     m->link_off>=sizeof *m && m->link_off<=m->length && m->links<=(m->length-m->link_off)/sizeof(struct mesh_link_info)){
    *node=m->node;result=0;
    for(uint32_t i=0;i<m->links;i++){
      struct mesh_link_info *link=&mesh_links(m)[i];
      if(!link->device[0])continue;
      if((uint32_t)result<capacity){
        out[result]=(struct mesh_link_view){.peer=atomic_load_explicit(&link->peer,memory_order_relaxed),
          .phase=atomic_load_explicit(&link->phase,memory_order_acquire),.bandwidth=atomic_load_explicit(&link->bandwidth,memory_order_relaxed)};
        memcpy(out[result].device,link->device,sizeof out[result].device);out[result].device[sizeof out[result].device-1]=0;
      }
      result++;
    }
  }
  munmap(m,(size_t)info.st_size);return result;
}

/* design/algorithm-sources.md#programtensor */
/* design/bridge-and-ipc.md#clients: a client's slot given up (by its detach, by an attach or the bridge where its
   process is gone): its program stopped, its rows closed for the bridge to release once its programs left the ports */
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_client *slot=mesh_client_at(m,client);
  if(!client || (client&MESH_RETIRING) || !atomic_compare_exchange_strong(&slot->owner,&client,client|MESH_RETIRING))return;
  atomic_store_explicit(&slot->configured,0,memory_order_release);
  for(uint32_t q=0;q<MESH_PORTS*MESH_QUEUES*2;q++)atomic_store_explicit(slot->length+q,0,memory_order_release);
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==client){
      if(buffer->pages)atomic_store_explicit(&buffer->closed,1,memory_order_release);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  atomic_store_explicit(&slot->owner,0,memory_order_release);
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
}

static uint64_t mesh_key(const char *name){
  uint64_t key=UINT64_C(0xcbf29ce484222325);
  for(const char *p=name;p && *p;p++)key=(key^(unsigned char)*p)*UINT64_C(0x100000001b3);
  return name && *name?key:0;
}

int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  int file=shm_open(MESH_NAME,O_RDWR,MESH_MODE);
  if(file<0) return errno;
  struct stat info;
  if(fstat(file,&info)){ int error=errno; close(file); return error; }
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  if(memory==MAP_FAILED){ close(file); return error; }
  pid_t bridge=(size_t)info.st_size<sizeof *memory?0:(pid_t)atomic_load(&memory->bridge_pid);
  error=(size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION ||
    memory->length>(uint64_t)info.st_size?EINVAL:!bridge || (kill(bridge,0) && errno==ESRCH)?ESRCH:EBUSY;
  uint64_t generation=error==EBUSY?(atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<25)-1):0;
  for(uint32_t s=0;error==EBUSY && s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)memory+memory->client_off)+s;
    uint64_t owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid(),vacant=atomic_load(&slot->owner);
    if(vacant && !(vacant&MESH_RETIRING) && kill((pid_t)(uint32_t)vacant,0) && errno==ESRCH){mesh_retire(memory,vacant);vacant=atomic_load(&slot->owner);}
    if(vacant || !atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
    slot->key=mesh_key(name);
    atomic_store(&slot->depth,0);atomic_store(&slot->cyclic,0);
    memset(slot->tx,0,sizeof slot->tx);
    for(uint32_t p=0;p<MESH_PORTS;p++){
      atomic_store(&slot->port[p].phase,0);atomic_store(&slot->port[p].domain,0);atomic_store(&slot->port[p].code,0);
      atomic_store(&slot->port[p].prepared,0);
    }
    *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=owner,.fd=file};
    return 0;
  }
  munmap(memory,(size_t)info.st_size); close(file);
  return error;
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
/* a row's publication target on link `link`: until mesh_transfers_prepare lays its cells, the client's program on that
   link names it */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *c,uint32_t row,uint32_t link){
  struct hdr *m=c->M;
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t *count=&publication->sends;
  struct mesh_target *targets=publication->targets;
  uint64_t first=(uint64_t)((char *)&mesh_client_at(m,c->client)->tx[link]-(char *)m);
  for(uint32_t i=0;i<*count;i++)if(targets[i].stream==first)return &targets[i];
  struct mesh_target *target=&targets[(*count)++];
  *target=(struct mesh_target){.stream=first};
  return target;
}

/* design/prepared-machine.md#M13 */
/* design/prepared-machine.md#M04 */
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *m,uint32_t row,struct prepared_publication *records){
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t count=0;
  for(uint32_t i=0;i<publication->sends;i++){
    struct mesh_target target=publication->targets[i];
    for(uint32_t k=0;k<target.count;k++){
      if(records){
        struct mesh_send *cell=(void *)((char *)m+target.stream+sizeof(struct mesh_send)*target.stride*k);
        records[count]=(struct prepared_publication){.destination=(uintptr_t)cell,.argument=cell->request.wr_id};
      }
      count++;
    }
  }
  return count;
}

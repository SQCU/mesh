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
      mesh_bits_clear(m,MESH_ROW_OWN,row,1);
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
    if(!vacant || !kill((pid_t)(uint32_t)vacant,0) || errno!=ESRCH){ munmap(memory,(size_t)info.st_size); close(file); return EADDRINUSE; }
    mesh_retire(memory,vacant);
    vacant=0;
  }
  uint64_t device=atomic_load_explicit(&memory->device_client,memory_order_seq_cst);
  client|=(~device)&(UINT64_C(1)<<63);
  for(uint32_t q=0;q<memory->links*memory->qps;q++)for(int d=0;d<2;d++)atomic_store_explicit(mesh_order_length(memory,client,q,d),0,memory_order_relaxed);
  atomic_store_explicit(&memory->client,client,memory_order_release);
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file};
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
static uint32_t mesh_allocate(struct mesh_ctx *c,uint32_t count,uint32_t align,uint32_t begin,uint32_t end,int own,int hot){
  for(uint32_t pass=0;pass<2;pass++){
    if(pass)mesh_reclaim(c->M);
    for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
      uint32_t next=first;
      for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
        uint64_t occupied=(atomic_load_explicit(&mesh_plane(c->M,own)[w],memory_order_acquire)|atomic_load_explicit(&mesh_plane(c->M,hot)[w],memory_order_acquire))&mesh_word_mask(first,count,w);
        if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
      }
      if(next==first){ mesh_bits_set(c->M,own,first,count); return first; }
      first=(next+align-1)/align*align;
    }
  }
  errno=ENOMEM; return MESH_ABSENT;
}

uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,count,1,0,mesh_rows(c->M),MESH_ROW_OWN,MESH_ROW_HOT);
  if(first==MESH_ABSENT) return first;
  for(uint32_t r=first;r<first+count;r++){
    atomic_store_explicit(&mesh_page(c->M)[r].device,0,memory_order_relaxed);
    atomic_store_explicit(&mesh_page(c->M)[r].mapping,MESH_ABSENT,memory_order_release);
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[r];
    atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
    buffer->pages=buffer->constant=0;buffer->channel=MESH_ABSENT;
    struct mesh_publication *publication=mesh_publication_at(c->M,r);
    publication->sends=publication->device_input=0;
    atomic_store_explicit(&publication->argument,0,memory_order_relaxed);
    atomic_store_explicit(&buffer->owner,c->client,memory_order_release);
  }
  c->rows+=count;
  return first;
}

uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,pages,align,0,mesh_rows(c->M),MESH_PAGE_OWN,MESH_PAGE_OWN);
  if(first!=MESH_ABSENT) c->arena+=pages;
  return first;
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page,uint32_t index){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    struct mesh_page_entry *entry=&mesh_page(c->M)[first+offset/c->M->block];
    atomic_store_explicit(&entry->mapping,((uint64_t)index<<32)|(page+offset),memory_order_relaxed);
    atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(c->M,page+offset),memory_order_relaxed);
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
    if(page!=MESH_ABSENT)mesh_bits_clear(m,MESH_PAGE_OWN,page,m->block);
  }
  atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
  mesh_bits_clear(m,MESH_ROW_HOT,row,pages/m->block);
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
      if(buffer->channel<m->links*m->qps)for(uint32_t offset=0;offset<buffer->pages;offset+=m->block)
        atomic_store_explicit(&mesh_page(m)[row+offset/m->block].mapping,MESH_ABSENT,memory_order_relaxed);
      buffer->channel=MESH_ABSENT;
      atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
      mesh_bits_set(m,MESH_FREE,row,1);
      atomic_store_explicit(&buffer->owner,owner,memory_order_release);
    }
  }
  for(uint32_t index=0;index<mesh_blocks(m);index++){
    struct mesh_pool *pool=&mesh_pools(m)[index];
    uint64_t owner=atomic_load_explicit(&pool->owner,memory_order_acquire);
    if(owner && owner!=atomic_load_explicit(&m->client,memory_order_acquire)){
      uint32_t pages=pool->pages;pool->pages=0;
      atomic_store_explicit(&pool->owner,0,memory_order_relaxed);
      mesh_bits_clear(m,MESH_PAGE_OWN,index*m->block,pages);
    }
  }
}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(c->M,MESH_ROW_OWN,first,count);
}

/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M18 */
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
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *m,uint32_t row,struct prepared_publication *records){
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t count=0;
  for(uint32_t i=0;i<publication->sends;i++){
    struct mesh_target target=publication->targets[i];
    for(uint32_t k=0;k<target.count;k++){
      if(records)records[count]=(struct prepared_publication){.destination=(uintptr_t)m+target.stream+sizeof(struct mesh_send)*target.stride*k,.argument=1};
      count++;
    }
  }
  return count;
}

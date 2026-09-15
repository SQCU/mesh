#include <signal.h>
#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
/* design/pages-and-functions.md#what-the-page-table-is */

/* design/collective-dependency-ledger.md#d14-teardown-retains-outstanding-device-storage */
static void mesh_retire(struct hdr *m,uint64_t client){
  atomic_store_explicit(&m->configured,0,memory_order_release);
  for(uint32_t i=0;i<2*MESH_QPS;i++) atomic_store_explicit(&m->order_length[i],0,memory_order_release);
  struct mesh_ctx context={.M=m};
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(buffer->owner==client && buffer->first==row && buffer->pages){
      mesh_backing_release(&context,row,buffer->pages);
      mesh_bits_clear(m,MESH_ROW_OWN,row,buffer->pages);
    }
  }
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
    uint64_t previous=vacant;
    if(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,client,memory_order_seq_cst,memory_order_acquire)) continue;
    mesh_retire(memory,previous);
    break;
  }
  uint64_t device=atomic_load_explicit(&memory->device_client,memory_order_seq_cst);
  client|=(~device)&(UINT64_C(1)<<63);
  for(uint32_t queue=0;queue<MESH_NOTICE_QUEUES;queue++)
    atomic_store_explicit(&memory->notice_head[mesh_notice_queue(client,queue)],MESH_ABSENT,memory_order_relaxed);
  atomic_store_explicit(&memory->client,client,memory_order_release);
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire(c->M,c->client);
  atomic_store_explicit(&c->M->client,0,memory_order_release);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programkernel_call */
static uint32_t mesh_allocate(struct mesh_ctx *c,uint32_t count,uint32_t align,uint32_t begin,uint32_t end,int own,int hot){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    uint32_t next=first;
    for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
      uint64_t occupied=(atomic_load_explicit(&mesh_plane(c->M,own)[w],memory_order_acquire)|atomic_load_explicit(&mesh_plane(c->M,hot)[w],memory_order_acquire))&mesh_word_mask(first,count,w);
      if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
    }
    if(next==first){ mesh_bits_set(c->M,own,first,count); return first; }
    first=(next+align-1)/align*align;
  }
  errno=ENOMEM; return MESH_ABSENT;
}

uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,count,1,0,mesh_rows(c->M),MESH_ROW_OWN,MESH_ROW_HOT);
  if(first==MESH_ABSENT) return first;
  for(int plane=0;plane<MESH_ROW_OWN;plane++) mesh_bits_clear(c->M,plane,first,count);
  for(uint32_t r=first;r<first+count;r++){
    atomic_store_explicit(&mesh_page(c->M)[r],MESH_ABSENT,memory_order_release);
    mesh_buffers(c->M)[r]=(struct mesh_buffer){.first=r};
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
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    atomic_store_explicit(&mesh_page(c->M)[first+offset],page+offset,memory_order_relaxed);
    mesh_backing(c->M)[page+offset]=first+offset;
    *mesh_tag(c->M,page+offset)=first+offset;
  }
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[row];
    if(!buffer->pages){row++;continue;}
    row=buffer->first+buffer->pages;
    uint64_t ownership=atomic_fetch_or_explicit(&buffer->ownership,MESH_BUFFER_FLAG(MESH_BUFFER_CLOSED|MESH_BUFFER_SEALED),memory_order_acq_rel);
    if(!(ownership&MESH_BUFFER_FLAG(MESH_BUFFER_RECLAIMED)))mesh_buffer_seal(c->M,buffer->first,buffer->pages);
  }
}

/* design/algorithm-sources.md#programtensor */
static void mesh_buffer_enqueue(struct hdr *m,struct mesh_buffer *buffer){
  if(atomic_fetch_or_explicit(&buffer->ownership,MESH_BUFFER_FLAG(MESH_BUFFER_QUEUED),memory_order_acq_rel)&MESH_BUFFER_FLAG(MESH_BUFFER_QUEUED|MESH_BUFFER_RECLAIMED))return;
  uint32_t head=atomic_load_explicit(&m->reclaim_head,memory_order_relaxed);
  do {buffer->next=head;}
  while(!atomic_compare_exchange_weak_explicit(&m->reclaim_head,&head,buffer->first,memory_order_release,memory_order_relaxed));
}

/* design/algorithm-sources.md#programtensor */
int mesh_buffer_retain(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    uint64_t ownership=atomic_load_explicit(&buffer->ownership,memory_order_acquire);
    do {
      if((!(uint32_t)ownership && (ownership&MESH_BUFFER_FLAG(MESH_BUFFER_SEALED))) || (uint32_t)ownership==UINT32_MAX){
        mesh_buffer_release(m,first,row-first);return ESTALE;
      }
    } while(!atomic_compare_exchange_weak_explicit(&buffer->ownership,&ownership,ownership+1,memory_order_acq_rel,memory_order_acquire));
    row=buffer->first+buffer->pages;
  }
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void mesh_buffer_release(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    row=buffer->first+buffer->pages;
    uint64_t ownership=atomic_fetch_sub_explicit(&buffer->ownership,1,memory_order_acq_rel);
    if((uint32_t)ownership==1 && (ownership&MESH_BUFFER_FLAG(MESH_BUFFER_SEALED)))mesh_buffer_enqueue(m,buffer);
  }
}

/* design/algorithm-sources.md#programtensor */
void mesh_buffer_seal(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    row=buffer->first+buffer->pages;
    uint64_t ownership=atomic_fetch_or_explicit(&buffer->ownership,MESH_BUFFER_FLAG(MESH_BUFFER_SEALED),memory_order_acq_rel);
    if((ownership&MESH_BUFFER_FLAG(MESH_BUFFER_CLOSED)) || !(uint32_t)ownership)mesh_buffer_enqueue(m,buffer);
  }
}

/* design/algorithm-sources.md#programtensor */
uint32_t mesh_collect(struct hdr *m,uint32_t pending){
  uint32_t lists[]={atomic_exchange_explicit(&m->reclaim_head,MESH_ABSENT,memory_order_acquire),pending},deferred=MESH_ABSENT;
  for(uint32_t list=0;list<2;list++)for(uint32_t row=lists[list];row!=MESH_ABSENT;){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];row=buffer->next;
    uint64_t ownership=atomic_load_explicit(&buffer->ownership,memory_order_acquire);
    if((ownership&MESH_BUFFER_FLAG(MESH_BUFFER_CLOSED)) && atomic_load_explicit(&m->device_client,memory_order_seq_cst)==buffer->owner){
      buffer->next=deferred;deferred=buffer->first;continue;
    }
    uint32_t first=buffer->first,pages=buffer->pages;
    for(uint32_t offset=0;offset<pages;offset+=m->block){
      uint32_t page=atomic_load_explicit(&mesh_page(m)[first+offset],memory_order_acquire);
      if(page!=MESH_ABSENT)mesh_bits_clear(m,MESH_PAGE_OWN,page,m->block);
    }
    atomic_store_explicit(&buffer->ownership,MESH_BUFFER_FLAG(MESH_BUFFER_RECLAIMED|MESH_BUFFER_SEALED),memory_order_release);
    mesh_bits_clear(m,MESH_ROW_HOT,first,pages);
  }
  return deferred;
}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(c->M,MESH_ROW_OWN,first,count);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_publish(struct hdr *m,uint32_t row){
  struct mesh_buffer *buffer=&mesh_buffers(m)[row];
  atomic_fetch_or_explicit(&mesh_plane(m,MESH_PRESENT)[row/64],UINT64_C(1)<<(row%64),memory_order_release);
  uint32_t uses=atomic_load_explicit(&buffer->uses,memory_order_relaxed);
  if(uses>>MESH_COMPUTE_THREADS)
    mesh_notice_push(m,mesh_notice_queue(buffer->owner,MESH_NOTICE_SEND),row);
  uses&=MESH_COMPUTE_MASK;
  while(uses){
    uint32_t worker=(uint32_t)__builtin_ctz(uses);uses&=uses-1;
    mesh_notice_push(m,mesh_notice_queue(buffer->owner,MESH_NOTICE_COMPUTE+worker),row);
  }
  mesh_buffer_release(m,row,buffer->pages);
}

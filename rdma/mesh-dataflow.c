#include <signal.h>
#include "mesh-memory.h"
#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
/* design/pages-and-functions.md#what-the-page-table-is */

static struct mesh_ctx CTX0;
struct mesh_ctx *mesh_context(void){ return &CTX0; }
struct hdr *mesh_region(struct mesh_ctx *c){ return c->M; }

/* ledger D14: a leaving client's queue orders, unsent productions and ownership end with it. Work requests
   the bridge already posted keep their occupancy until the bridge destroys their queue pairs. */
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
  mesh_memory_warning((uint64_t)info.st_size,0);
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  if(memory==MAP_FAILED){ close(file); return error; }
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION || memory->length>(uint64_t)info.st_size){
    munmap(memory,(size_t)info.st_size); close(file); return EINVAL;
  }
  uint64_t client=((atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)<<32)|(uint32_t)getpid(),vacant=0;
  while(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,client,memory_order_acq_rel,memory_order_acquire)){
    if(!vacant || !kill((pid_t)(uint32_t)vacant,0) || errno!=ESRCH){ munmap(memory,(size_t)info.st_size); close(file); return EADDRINUSE; }
    uint64_t previous=vacant;
    if(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,client,memory_order_acq_rel,memory_order_acquire)) continue;
    mesh_retire(memory,previous);
    break;
  }
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

/* design/algorithm-sources.md#programtensor */
void *mesh_view_create(struct mesh_ctx *c,uint32_t row,size_t count,_Atomic uint32_t *mapped){
  size_t page_bytes=c->M->pgsz;
  if(!count || count>SIZE_MAX/page_bytes || row>mesh_rows(c->M) || count>mesh_rows(c->M)-row){ errno=EINVAL; return NULL; }
  _Atomic uint32_t *pages=mesh_page(c->M)+row;
  for(size_t i=0;i<count;i++) if(atomic_load_explicit(&pages[i],memory_order_acquire)>=mesh_rows(c->M)){ errno=EINVAL; return NULL; }
  size_t length=count*page_bytes;
  unsigned char *address=mmap(NULL,length,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(address==MAP_FAILED) return NULL;
  for(size_t i=0;i<count;i++)atomic_init(&mapped[i],MESH_ABSENT);
  int error=mesh_view_bind(c,row,count,address,mapped);
  if(error){munmap(address,length);errno=error;return NULL;}
  return address;
}

/* design/algorithm-sources.md#programtensor */
int mesh_view_bind(struct mesh_ctx *c,uint32_t row,size_t count,void *address,_Atomic uint32_t *mapped){
  size_t page_bytes=c->M->pgsz;
  _Atomic uint32_t *pages=mesh_page(c->M)+row;
  for(size_t first=0;first<count;){
    uint32_t page=atomic_load_explicit(&pages[first],memory_order_acquire);
    if(atomic_load_explicit(&mapped[first],memory_order_acquire)==page){first++;continue;}
    size_t end=first+1;
    while(end<count && atomic_load_explicit(&mapped[end],memory_order_acquire)!=page+end-first && atomic_load_explicit(&pages[end],memory_order_acquire)==page+end-first) end++;
    off_t offset=(off_t)(c->M->data_off+(uint64_t)page*page_bytes);
    if(mmap((char *)address+first*page_bytes,(end-first)*page_bytes,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,c->fd,offset)==MAP_FAILED)return errno;
    for(size_t i=first;i<end;i++)atomic_store_explicit(&mapped[i],page+(uint32_t)(i-first),memory_order_release);
    first=end;
  }
  return 0;
}

/* design/algorithm-sources.md#programtensor */
int mesh_view_destroy(void *address,size_t length){ return munmap(address,length)?errno:0; }

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
int mesh_backing_alloc(struct mesh_ctx *c,uint32_t first,uint32_t count,uint32_t quantum,int contiguous){
  if(!quantum || !count || count%quantum || first>mesh_rows(c->M) || count>mesh_rows(c->M)-first)return EINVAL;
  uint32_t span=contiguous?count:quantum;
  for(uint32_t offset=0;offset<count;offset+=span){
    uint32_t page=mesh_arena_alloc(c,span,quantum);
    if(page==MESH_ABSENT)return errno;
    mesh_map(c,first+offset,span,page);
    for(uint32_t index=0;index<span;index+=quantum){
      uint32_t row=first+offset+index;
      for(uint32_t i=0;i<quantum;i++)mesh_buffers(c->M)[row+i]=(struct mesh_buffer){.first=row};
      mesh_buffers(c->M)[row]=(struct mesh_buffer){.ownership=1|MESH_BUFFER_FLAG(MESH_BUFFER_PRODUCER),.first=row,.pages=quantum,.owner=c->client};
      for(uint32_t i=0;i<quantum;i++)mesh_backing(c->M)[page+index+i]=row;
      mesh_bits_set(c->M,MESH_ROW_HOT,row,quantum);
    }
  }
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[mesh_buffers(c->M)[row].first];
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
    struct mesh_buffer *buffer=&mesh_buffers(m)[mesh_buffers(m)[row].first];
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
    struct mesh_buffer *buffer=&mesh_buffers(m)[mesh_buffers(m)[row].first];
    row=buffer->first+buffer->pages;
    uint64_t ownership=atomic_fetch_sub_explicit(&buffer->ownership,1,memory_order_acq_rel);
    if((uint32_t)ownership==1 && (ownership&MESH_BUFFER_FLAG(MESH_BUFFER_SEALED)))mesh_buffer_enqueue(m,buffer);
  }
}

/* design/algorithm-sources.md#programtensor */
void mesh_buffer_produced(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(m)[mesh_buffers(m)[row].first];
    row=buffer->first+buffer->pages;
    if(atomic_fetch_and_explicit(&buffer->ownership,~MESH_BUFFER_FLAG(MESH_BUFFER_PRODUCER),memory_order_acq_rel)&MESH_BUFFER_FLAG(MESH_BUFFER_PRODUCER))
      mesh_buffer_release(m,buffer->first,buffer->pages);
  }
}

/* design/algorithm-sources.md#programtensor */
void mesh_buffer_seal(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;){
    struct mesh_buffer *buffer=&mesh_buffers(m)[mesh_buffers(m)[row].first];
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
    uint32_t page=atomic_load_explicit(&mesh_page(m)[first],memory_order_acquire);
    mesh_bits_clear(m,MESH_PAGE_OWN,page,pages);
    atomic_store_explicit(&buffer->ownership,MESH_BUFFER_FLAG(MESH_BUFFER_RECLAIMED|MESH_BUFFER_SEALED),memory_order_release);
    mesh_bits_clear(m,MESH_ROW_HOT,first,pages);
  }
  return deferred;
}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(c->M,MESH_ROW_OWN,first,count);
}

void mesh_map(struct mesh_ctx *c,uint32_t first,uint32_t count,uint32_t page){
  _Atomic uint32_t *table=mesh_page(c->M);
  for(uint32_t i=0;i<count;i++) atomic_store_explicit(&table[first+i],page+i,memory_order_release);
}

void mesh_constant(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_set(c->M,MESH_CONSTANT,first,count);
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  mesh_notify(c->M,first,count);
  mesh_buffer_produced(c->M,first,count);
}

void *mesh_row_data(struct mesh_ctx *c,uint32_t row){
  uint32_t page=atomic_load_explicit(&mesh_page(c->M)[row],memory_order_acquire);
  return page==MESH_ABSENT?NULL:mesh_at(c->M,page);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_publish_partial(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  mesh_notify(c->M,first,count);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_notify(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t word=first/64;count && word<=(first+count-1)/64;word++){
    uint64_t sources=atomic_load_explicit(&mesh_plane(m,MESH_SEND_SOURCE)[word],memory_order_acquire)&mesh_word_mask(first,count,word);
    while(sources){
      uint32_t row=word*64+(uint32_t)__builtin_ctzll(sources);sources&=sources-1;
      mesh_notice_push(m,MESH_NOTICE_SEND,row);
    }
  }
}

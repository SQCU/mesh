#include <signal.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include "mesh-memory.h"
#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
/* design/pages-and-functions.md#what-the-page-table-is */

static void mesh_execution_destroy(struct mesh_ctx *);
static int mesh_execution_create(struct mesh_ctx *);
static void mesh_reader_destroy(struct mesh_ctx *);
static int mesh_reader_unbind_serial(struct mesh_ctx *,struct mesh_row_map *);
static struct mesh_ctx CTX0;
struct mesh_ctx *mesh_context(void){ return &CTX0; }
struct hdr *mesh_region(struct mesh_ctx *c){ return c->M; }

/* ledger D14: a leaving client's queue orders, unsent productions and ownership end with it. Work requests
   the bridge already posted keep their occupancy until the bridge destroys their queue pairs. */
static void mesh_retire(struct hdr *m){
  atomic_store_explicit(&m->configured,0,memory_order_release);
  for(uint32_t i=0;i<2*MESH_QPS;i++) atomic_store_explicit(&m->order_length[i],0,memory_order_release);
  for(uint32_t w=0;w<mesh_words(m);w++) atomic_store_explicit(&mesh_plane(m,MESH_ROW_OWN)[w],0,memory_order_release);
  mesh_bits_clear(m,MESH_PAGE_OWN,0,mesh_rows(m));
  mesh_bits_clear(m,MESH_SEND_SOURCE,0,mesh_rows(m));
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
  uint64_t vacant=0;
  while(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)){
    if(!vacant || !kill((pid_t)vacant,0) || errno!=ESRCH){ munmap(memory,(size_t)info.st_size); close(file); return EBUSY; }
    if(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)) continue;
    mesh_retire(memory);
    break;
  }
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.fd=file};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_execution_destroy(c);
  mesh_reader_destroy(c);
  mesh_retire(c->M);
  atomic_store_explicit(&c->M->client,0,memory_order_release);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_view_create(struct mesh_ctx *c,uint32_t row,size_t count){
  size_t page_bytes=c->M->pgsz;
  if(!count || count>SIZE_MAX/page_bytes || row>mesh_rows(c->M) || count>mesh_rows(c->M)-row){ errno=EINVAL; return NULL; }
  _Atomic uint32_t *pages=mesh_page(c->M)+row;
  for(size_t i=0;i<count;i++) if(atomic_load_explicit(&pages[i],memory_order_acquire)>=mesh_rows(c->M)){ errno=EINVAL; return NULL; }
  size_t length=count*page_bytes;
  unsigned char *address=mmap(NULL,length,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(address==MAP_FAILED) return NULL;
  for(size_t first=0;first<count;){
    uint32_t page=atomic_load_explicit(&pages[first],memory_order_acquire);
    size_t end=first+1;
    while(end<count && atomic_load_explicit(&pages[end],memory_order_acquire)==page+end-first) end++;
    off_t offset=(off_t)(c->M->data_off+(uint64_t)page*page_bytes);
    if(mmap(address+first*page_bytes,(end-first)*page_bytes,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,c->fd,offset)==MAP_FAILED){
      int error=errno; munmap(address,length); errno=error; return NULL;
    }
    first=end;
  }
  return address;
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
  for(int plane=MESH_READ;plane<MESH_PLANES;plane++) mesh_bits_clear(c->M,plane,first,count);
  for(uint32_t r=first;r<first+count;r++){
    atomic_store_explicit(&mesh_page(c->M)[r],MESH_ABSENT,memory_order_release);
    mesh_mask(c->M)[r]=0;
    struct mesh_reader_state *state=&mesh_reader_states(c->M)[r];
    atomic_store(&state->generation,0);atomic_store(&state->expected,0);atomic_store(&state->completed,0);state->plane=MESH_ABSENT;
  }
  c->rows+=count;
  return first;
}

uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,pages,align,0,mesh_rows(c->M),MESH_PAGE_OWN,MESH_PAGE_HOT);
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
  }
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  _Atomic uint32_t *pages=mesh_page(c->M);
  for(uint32_t offset=0;offset<count;offset++){
    uint32_t page=atomic_exchange_explicit(&pages[first+offset],MESH_ABSENT,memory_order_acq_rel);
    if(page!=MESH_ABSENT)mesh_arena_release(c,page,1);
  }
}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(c->M,MESH_ROW_OWN,first,count);
}

/* design/algorithm-sources.md#program */
void mesh_arena_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(c->M,MESH_PAGE_OWN,first,count);
}

void mesh_map(struct mesh_ctx *c,uint32_t first,uint32_t count,uint32_t page){
  _Atomic uint32_t *table=mesh_page(c->M);
  for(uint32_t i=0;i<count;i++) atomic_store_explicit(&table[first+i],page+i,memory_order_release);
}

void mesh_constant(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_set(c->M,MESH_CONSTANT,first,count);
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  mesh_notify(c->M,first,count);
}

static int mesh_free_plane(uint64_t busy,int *plane){
  for(int p=0;p<MESH_READERS;p++) if(!(busy&(UINT64_C(1)<<p))){ *plane=p; return 0; }
  return ENOSPC;
}
struct mesh_reader_storage {size_t count;struct mesh_reader_storage *next;struct mesh_reader_member members[];};
/* design/algorithm-sources.md#programkernel_call */
static int mesh_reader_bind(struct mesh_ctx *c,uint64_t *used,struct mesh_row_map *map){
  if(!map->count || (uint64_t)map->first+map->count>mesh_rows(c->M))return EINVAL;
  int mutable=0;
  for(uint32_t row=map->first;row<map->first+map->count;row++)mutable|=!mesh_bit(c->M,MESH_CONSTANT,row);
  if(!mutable)return 0;
  if(map->count>(SIZE_MAX-sizeof(struct mesh_reader_storage))/sizeof(struct mesh_reader_member))return EOVERFLOW;
  struct mesh_reader_storage *storage=calloc(1,sizeof *storage+map->count*sizeof *storage->members);
  if(!storage)return ENOMEM;
  storage->count=map->count;
  for(size_t i=0;i<storage->count;i++)storage->members[i].row=MESH_ABSENT;
  storage->next=c->readers;c->readers=storage;
  map->members=storage->members;
  for(uint32_t i=0;i<map->count;i++){
    uint32_t row=map->first+i;if(mesh_bit(c->M,MESH_CONSTANT,row))continue;
    struct mesh_reader_state *state=&mesh_reader_states(c->M)[row];
    if(atomic_load(&state->expected)==UINT32_MAX)return EOVERFLOW;
    if(state->plane==MESH_ABSENT){
      int plane,error=mesh_free_plane(used[row],&plane);if(error)return error;
      state->plane=(uint32_t)plane;
    }
    used[row]|=UINT64_C(1)<<state->plane;
    struct mesh_reader_member *member=&storage->members[i];
    member->row=row;atomic_store(&member->generation,atomic_load(&state->generation)-1);
    atomic_fetch_add(&state->expected,1);
    mesh_bits_clear(c->M,MESH_READ+(int)state->plane,row,1);
  }
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
static void mesh_reader_destroy(struct mesh_ctx *c){
  while(c->readers){
    struct mesh_reader_storage *storage=c->readers;c->readers=storage->next;
    free(storage);
  }
}
/* design/algorithm-sources.md#programkernel_call */
void mesh_reader_unbind(struct mesh_ctx *c,struct mesh_row_map *map){
  if(!map->members || mesh_reader_unbind_serial(c,map))return;
  struct mesh_reader_storage **at=(struct mesh_reader_storage **)&c->readers;
  while(*at && (*at)->members!=map->members)at=&(*at)->next;
  if(!*at)return;
  struct mesh_reader_storage *storage=*at;*at=storage->next;
  for(size_t i=0;i<storage->count;i++){
    struct mesh_reader_member *member=&storage->members[i];if(member->row==MESH_ABSENT)continue;
    struct mesh_reader_state *state=&mesh_reader_states(c->M)[member->row];
    if(atomic_load(&member->generation)==atomic_load(&state->generation))atomic_fetch_sub(&state->completed,1);
    uint32_t expected=atomic_fetch_sub(&state->expected,1)-1;
    if(atomic_load(&state->completed)==expected){
      mesh_bits_set(c->M,MESH_READ+(int)state->plane,member->row,1);
      mesh_notify(c->M,member->row,1);
    }
  }
  free(storage);
  map->members=NULL;
}

int mesh_realize(struct mesh_ctx *c,struct mesh_row_function *const *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count){
  struct hdr *m=c->M;
  uint32_t rows=mesh_rows(m),block=m->block;
  uint64_t *used=malloc(rows*sizeof *used);
  size_t *order=malloc((binding_count?binding_count:1)*sizeof *order);
  if(!used || !order){ free(used); free(order); return ENOMEM; }
  memcpy(used,mesh_mask(m),rows*sizeof *used);
  int error=0;
  for(size_t i=0;i<count && !error;i++){
    struct mesh_row_function *f=functions[i];
    if(!f->outputs || !f->output || (f->inputs && !f->input)){ error=EINVAL; break; }
    for(uint32_t j=0;j<f->inputs && !error;j++)error=mesh_reader_bind(c,used,&f->input[j]);
    for(uint32_t j=0;j<f->outputs && !error;j++){
      struct mesh_row_map r=f->output[j];
      if(!r.count || (uint64_t)r.first+r.count>rows){ error=EINVAL; break; }
      for(uint32_t x=0;x<r.count;x++) if(mesh_bit(m,MESH_CONSTANT,r.first+x)) error=EINVAL;
    }
  }
  for(size_t i=0;i<return_count && !error;i++)error=mesh_reader_bind(c,used,&returns[i]);
  if(!error && c->readers && !c->execution)error=mesh_execution_create(c);
  _Atomic uint32_t *table=mesh_page(m);
  for(size_t i=0;i<binding_count && !error;i++){
    struct mesh_row_binding *b=&bindings[i];
    if(!m->qps || !b->count || b->count%block || (uint64_t)b->first+b->count>rows){ error=EINVAL; break; }
    if(b->bytes>(uint64_t)b->count*m->pgsz || b->bytes<=(uint64_t)(b->count-block)*m->pgsz){error=EINVAL;break;}
    /* ledger D4, D6: each block is one message on one contiguous, block-aligned page run inside one region */
    for(uint32_t k=0;k<b->count && !error;k+=block){
      uint32_t page=atomic_load_explicit(&table[b->first+k],memory_order_acquire);
      if(page==MESH_ABSENT || page%block || (uint64_t)page+block>rows){ error=EINVAL; break; }
      for(uint32_t x=1;x<block;x++) if(atomic_load_explicit(&table[b->first+k+x],memory_order_acquire)!=page+x){ error=EINVAL; break; }
    }
    if(error || b->receive) continue;
    uint64_t busy=0; int plane=0;
    for(uint32_t r=b->first;r<b->first+b->count;r++)busy|=used[r];
    error=mesh_free_plane(busy,&plane);
    if(error) break;
    for(uint32_t r=b->first;r<b->first+b->count;r++)used[r]|=UINT64_C(1)<<plane;
    b->plane=(uint32_t)plane;
  }
  if(!error){
    for(uint32_t r=0;r<rows;r++) if(mesh_bit(m,MESH_ROW_OWN,r)) mesh_mask(m)[r]=used[r];
    for(size_t i=0;i<binding_count;i++)if(!bindings[i].receive)mesh_bits_set(m,MESH_SEND_SOURCE,bindings[i].first,bindings[i].count);
    /* ledger D5: both participants append each queue's blocks in (identity, block index) order */
    for(size_t i=0;i<binding_count;i++){
      size_t j=i;
      while(j && (bindings[order[j-1]].queue>bindings[i].queue ||
                  (bindings[order[j-1]].queue==bindings[i].queue && bindings[order[j-1]].binding>bindings[i].binding))){ order[j]=order[j-1]; j--; }
      order[j]=i;
    }
    for(size_t i=0;i<binding_count && !error;i++){
      struct mesh_row_binding *b=&bindings[order[i]];
      uint32_t queue=b->queue%m->qps;
      int direction=b->receive?MESH_RECEIVE:MESH_SEND;
      _Atomic uint32_t *length=mesh_order_length(m,queue,direction);
      for(uint32_t k=0;k<b->count;k+=block){
        uint32_t at=atomic_load_explicit(length,memory_order_acquire);
        if(at>=mesh_blocks(m)){ error=ENOSPC; break; }
        uint64_t remaining=b->bytes-(uint64_t)k*m->pgsz,maximum=(uint64_t)block*m->pgsz;
        uint32_t bytes=(uint32_t)(((remaining<maximum?remaining:maximum)+4095)/4096*4096);
        mesh_transfers(m,queue,direction)[at]=(struct mesh_transfer){.local_row=b->first+k,.binding=b->binding,.offset=k,.plane=b->plane,.bytes=bytes};
        atomic_store_explicit(length,at+1,memory_order_release);
      }
    }
  }
  free(order); free(used);
  if(!error && binding_count){
    if(!atomic_load_explicit(&m->configured,memory_order_acquire)){
      atomic_store(&m->port.code,0);atomic_store(&m->port.domain,0);
    }
    atomic_store_explicit(&m->configured,(uint32_t)getpid(),memory_order_release);
  }
  return error;
}

void *mesh_row_data(struct mesh_ctx *c,uint32_t row){
  uint32_t page=atomic_load_explicit(&mesh_page(c->M)[row],memory_order_acquire);
  return page==MESH_ABSENT?NULL:mesh_at(c->M,page);
}

static int mesh_claimable(struct hdr *m,uint32_t first,uint32_t count){
  _Atomic uint64_t *present=mesh_plane(m,MESH_PRESENT);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t bits=mesh_word_mask(first,count,w);
    if(atomic_load_explicit(&mesh_plane(m,MESH_PRODUCING)[w],memory_order_acquire)&bits) return 0;
    uint64_t pending=atomic_load_explicit(&present[w],memory_order_acquire)&bits;
    while(pending){
      uint32_t row=w*64+(uint32_t)__builtin_ctzll(pending); pending&=pending-1;
      uint64_t need=mesh_mask(m)[row];
      for(int p=0;need;p++,need>>=1) if((need&1) && !mesh_bit(m,MESH_READ+p,row)) return 0;
    }
  }
  return 1;
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_reset(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct hdr *m=c->M;
  mesh_bits_clear(m,MESH_PRESENT,first,count);
  mesh_reads_reset(m,first,count);
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_issue(struct mesh_ctx *c,const struct mesh_row_function *f){
  struct hdr *m=c->M;
  for(uint32_t j=0;j<f->inputs;j++){
    if(!mesh_available(c,f->input[j]))return 0;
  }
  for(uint32_t j=0;j<f->outputs;j++){
    struct mesh_row_map r=f->output[j];
    if(!mesh_claimable(m,r.first,r.count))return 0;
  }
  for(uint32_t j=0;j<f->outputs;j++){
    struct mesh_row_map r=f->output[j];
    mesh_reset(c,r.first,r.count);mesh_bits_set(m,MESH_PRODUCING,r.first,r.count);
  }
  return 1;
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_publish_partial(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  mesh_notify(c->M,first,count);
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_publish(struct hdr *m,uint32_t first,uint32_t count){
  mesh_bits_set(m,MESH_PRESENT,first,count);
  mesh_bits_clear(m,MESH_PRODUCING,first,count);
  mesh_notify(m,first,count);
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_available(struct mesh_ctx *c,struct mesh_row_map map){
  if(!map.members)return mesh_bits_all(c->M,MESH_PRESENT,map.first,map.count);
  for(uint32_t i=0;i<map.count;i++){
    uint32_t row=map.first+i;
    const struct mesh_reader_state *state=&mesh_reader_states(c->M)[row];
    uint64_t generation=atomic_load_explicit(&state->generation,memory_order_acquire);
    if(!mesh_bit(c->M,MESH_PRESENT,row))return 0;
    if(map.members[i].row!=MESH_ABSENT && atomic_load_explicit(&map.members[i].generation,memory_order_acquire)==generation)return 0;
    if(atomic_load_explicit(&state->generation,memory_order_acquire)!=generation)return 0;
  }
  return 1;
}
/* design/algorithm-sources.md#programkernel_call */
void mesh_consume(struct mesh_ctx *c,struct mesh_row_map map){
  if(!map.members)return;
  for(uint32_t i=0;i<map.count;i++){
    struct mesh_reader_member *member=&map.members[i];if(member->row==MESH_ABSENT)continue;
    struct mesh_reader_state *state=&mesh_reader_states(c->M)[member->row];
    uint64_t generation=atomic_load_explicit(&state->generation,memory_order_acquire);
    if(atomic_exchange_explicit(&member->generation,generation,memory_order_acq_rel)==generation)continue;
    uint32_t completed=atomic_fetch_add_explicit(&state->completed,1,memory_order_acq_rel)+1;
    if(completed==atomic_load_explicit(&state->expected,memory_order_acquire)){
      mesh_bits_set(c->M,MESH_READ+(int)state->plane,member->row,1);
      mesh_notify(c->M,member->row,1);
    }
  }
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_complete(struct mesh_ctx *c,const struct mesh_row_function *f){
  for(uint32_t j=0;j<f->inputs;j++)mesh_consume(c,f->input[j]);
  for(uint32_t j=0;j<f->outputs;j++){struct mesh_row_map r=f->output[j];mesh_publish(c->M,r.first,r.count);}
}

struct mesh_edge { struct mesh_row_function *function; struct mesh_edge *next,**previous,*owned_next; uint32_t row; };
struct mesh_execution {
  struct mesh_ctx *context;
  dispatch_queue_t queue;
  pthread_t thread;
  _Atomic int stop;
  struct mesh_edge **readers;
  struct mesh_row_function *functions;
};
/* design/algorithm-sources.md#programkernel_call */
static void mesh_edge_bind(struct mesh_execution *e,struct mesh_edge *edge){
  edge->previous=&e->readers[edge->row];edge->next=*edge->previous;
  if(edge->next)edge->next->previous=&edge->next;
  *edge->previous=edge;
  struct mesh_edge **owned=&edge->function->edges;
  edge->owned_next=*owned;*owned=edge;
}
/* design/algorithm-sources.md#programkernel_call */
static void mesh_edge_remove(struct mesh_edge *edge){
  *edge->previous=edge->next;
  if(edge->next)edge->next->previous=edge->previous;
  free(edge);
}
/* design/algorithm-sources.md#programkernel_call */
static int mesh_reader_unbind_serial(struct mesh_ctx *c,struct mesh_row_map *map){
  struct mesh_execution *e=c->execution;if(!e || dispatch_get_specific(e)==e)return 0;
  dispatch_sync(e->queue,^{mesh_reader_unbind(c,map);});return 1;
}
/* design/algorithm-sources.md#programkernel_call */
void mesh_notify(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;row++)mesh_notice_push(m,MESH_NOTICE_COMPUTE,row);
  for(uint32_t word=first/64;count && word<=(first+count-1)/64;word++){
    uint64_t sources=atomic_load_explicit(&mesh_plane(m,MESH_SEND_SOURCE)[word],memory_order_acquire)&mesh_word_mask(first,count,word);
    while(sources){
      uint32_t row=word*64+(uint32_t)__builtin_ctzll(sources);sources&=sources-1;
      mesh_notice_push(m,MESH_NOTICE_SEND,row);
    }
  }
}
/* design/algorithm-sources.md#programkernel_call */
static void mesh_fire(struct mesh_execution *e,struct mesh_row_function *function){
  if(mesh_issue(e->context,function))function->submit(function->argument);
}
/* design/algorithm-sources.md#programkernel_call */
static void mesh_events(struct mesh_execution *e){
  struct hdr *m=e->context->M;
  uint32_t row=mesh_notice_take(m,MESH_NOTICE_COMPUTE);
  while(row!=MESH_ABSENT){
    uint32_t next=mesh_notice_next(m,MESH_NOTICE_COMPUTE,row);
    struct mesh_row_function *pending=NULL;
    for(struct mesh_edge *edge=e->readers[row];edge;edge=edge->next){
      struct mesh_row_function *function=edge->function;
      if(!function->pending){function->pending=1;function->pending_next=pending;pending=function;}
    }
    while(pending){
      struct mesh_row_function *function=pending;pending=function->pending_next;
      function->pending=0;mesh_fire(e,function);
    }
    row=next;
  }
}
/* design/algorithm-sources.md#programkernel_call */
static void *mesh_execution_progress(void *argument){
  struct mesh_execution *e=argument;
  pthread_setname_np("mesh.presence");
  while(!atomic_load_explicit(&e->stop,memory_order_acquire))
    if(atomic_load_explicit(&e->context->M->notice_head[MESH_NOTICE_COMPUTE],memory_order_acquire)!=MESH_ABSENT)
      dispatch_sync(e->queue,^{mesh_events(e);});
  return NULL;
}
/* design/algorithm-sources.md#programkernel_call */
static int mesh_execution_create(struct mesh_ctx *c){
  struct mesh_execution *e=calloc(1,sizeof *e);if(!e)return ENOMEM;
  e->context=c;e->readers=calloc(mesh_rows(c->M),sizeof *e->readers);
  if(!e->readers){free(e);return ENOMEM;}
  e->queue=dispatch_queue_create("mesh.presence",DISPATCH_QUEUE_SERIAL);
  dispatch_queue_set_specific(e->queue,e,e,NULL);
  c->execution=e;
  int error=pthread_create(&e->thread,NULL,mesh_execution_progress,e);
  if(error){c->execution=NULL;dispatch_release(e->queue);free(e->readers);free(e);}
  return error;
}
/* design/algorithm-sources.md#programkernel_call */
int mesh_execution_add(struct mesh_ctx *c,struct mesh_row_function *function,void *owner,void *argument){
  if(!c->execution){int error=mesh_execution_create(c);if(error)return error;}
  struct mesh_execution *e=c->execution;
  struct mesh_edge *edges=NULL;
  int error=0;
  for(uint32_t i=0;i<function->inputs+function->outputs && !error;i++){
    struct mesh_row_map map=i<function->inputs?function->input[i]:function->output[i-function->inputs];
    for(uint32_t row=map.first;row<map.first+map.count;row++){
      struct mesh_edge *edge=malloc(sizeof *edge);if(!edge){error=ENOMEM;break;}
      *edge=(struct mesh_edge){.function=function,.next=edges,.row=row};edges=edge;
    }
  }
  if(error){
    while(edges){struct mesh_edge *next=edges->next;free(edges);edges=next;}
    return error;
  }
  function->owner=owner;function->argument=argument;
  dispatch_sync(e->queue,^{
    struct mesh_edge *edge=edges;
    while(edge){struct mesh_edge *next=edge->next;mesh_edge_bind(e,edge);edge=next;}
    function->next=e->functions;e->functions=function;mesh_fire(e,function);
  });
  return 0;
}
/* design/algorithm-sources.md#programkernel_call */
void mesh_execution_remove(struct mesh_ctx *c,void *owner){
  struct mesh_execution *e=c->execution;if(!e)return;
  dispatch_sync(e->queue,^{
    struct mesh_row_function **at=&e->functions;
    while(*at){
      struct mesh_row_function *function=*at;
      if(function->owner==owner){
        *at=function->next;
        while(function->edges){struct mesh_edge *edge=function->edges;function->edges=edge->owned_next;mesh_edge_remove(edge);}
      }else at=&function->next;
    }
  });
}
/* design/algorithm-sources.md#programkernel_call */
static void mesh_execution_destroy(struct mesh_ctx *c){
  struct mesh_execution *e=c->execution;if(!e)return;
  atomic_store_explicit(&e->stop,1,memory_order_release);
  pthread_join(e->thread,NULL);
  while(e->functions){
    struct mesh_row_function *function=e->functions;e->functions=function->next;
    while(function->edges){struct mesh_edge *edge=function->edges;function->edges=edge->owned_next;mesh_edge_remove(edge);}
  }
  dispatch_release(e->queue);free(e->readers);free(e);c->execution=NULL;
}

#include <signal.h>
#include <dispatch/dispatch.h>
#include <sys/socket.h>
#include <sys/un.h>
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
static void mesh_index_reset(struct mesh_ctx *,uint32_t,uint32_t);
static int mesh_index_ready(struct mesh_ctx *,const struct mesh_indexed_read *);
static void mesh_index_complete(struct mesh_ctx *,const struct mesh_indexed_read *);
static struct mesh_ctx CTX0;
struct mesh_ctx *mesh_context(void){ return &CTX0; }
struct hdr *mesh_region(struct mesh_ctx *c){ return c->M; }

static int mesh_is(struct hdr *m,int plane,uint32_t row){
  return (atomic_load_explicit(&mesh_plane(m,plane)[row/64],memory_order_acquire)>>(row%64))&1;
}

/* ledger D14: a leaving client's queue orders, unsent productions and ownership end with it. Work requests
   the bridge already posted keep their occupancy until the bridge destroys their queue pairs. */
static void mesh_retire(struct hdr *m){
  atomic_store_explicit(&m->configured,0,memory_order_release);
  for(uint32_t i=0;i<2*MESH_QPS;i++) atomic_store_explicit(&m->order_length[i],0,memory_order_release);
  for(uint32_t w=0;w<mesh_words(m);w++) atomic_store_explicit(&mesh_plane(m,MESH_ROW_OWN)[w],0,memory_order_release);
  mesh_bits_clear(m,MESH_PAGE_OWN,0,mesh_rows(m));
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
  mesh_signal_init();
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_execution_destroy(c);
  mesh_retire(c->M);
  atomic_store_explicit(&c->M->client,0,memory_order_release);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#registered-memory-views */
void *mesh_view_create(struct mesh_ctx *c,const uint32_t *pages,size_t count){
  size_t page_bytes=c->M->pgsz;
  if(!count || count>SIZE_MAX/page_bytes){ errno=EINVAL; return NULL; }
  for(size_t i=0;i<count;i++) if(pages[i]>=mesh_rows(c->M)){ errno=EINVAL; return NULL; }
  size_t length=count*page_bytes;
  unsigned char *address=mmap(NULL,length,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(address==MAP_FAILED) return NULL;
  for(size_t first=0;first<count;){
    size_t end=first+1;
    while(end<count && pages[end]==pages[end-1]+1) end++;
    off_t offset=(off_t)(c->M->data_off+(uint64_t)pages[first]*page_bytes);
    if(mmap(address+first*page_bytes,(end-first)*page_bytes,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,c->fd,offset)==MAP_FAILED){
      int error=errno; munmap(address,length); errno=error; return NULL;
    }
    first=end;
  }
  return address;
}

/* design/algorithm-sources.md#registered-memory-views */
int mesh_view_destroy(void *address,size_t length){ return munmap(address,length)?errno:0; }

/* ledger D12: work-completion status is read out of band */
struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *c,size_t index){
  const struct mesh_port_info *port=&c->M->port;
  return (struct mesh_row_metadata){.when=port->when,.function=UINT32_MAX,.index=(uint32_t)index,
    .peer=port->peer,.code=port->code,.domain=port->domain};
}

/* design/algorithm-sources.md#nonblocking-table-ownership */
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

/* design/algorithm-sources.md#independent-configured-programs */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(c->M,MESH_ROW_OWN,first,count);
}

/* design/algorithm-sources.md#independent-configured-programs */
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

static int mesh_survey(struct mesh_ctx *c,const uint64_t *used,uint32_t first,uint32_t count,uint64_t *busy){
  if(!count || (uint64_t)first+count>mesh_rows(c->M)) return EINVAL;
  for(uint32_t r=first;r<first+count;r++) if(!mesh_is(c->M,MESH_CONSTANT,r)) *busy|=used[r];
  return 0;
}
static int mesh_free_plane(uint64_t busy,int *plane){
  for(int p=0;p<MESH_READERS;p++) if(!(busy&(UINT64_C(1)<<p))){ *plane=p; return 0; }
  return ENOSPC;
}
static void mesh_take(struct mesh_ctx *c,uint64_t *used,uint32_t first,uint32_t count,int plane){
  for(uint32_t r=first;r<first+count;r++) if(!mesh_is(c->M,MESH_CONSTANT,r)) used[r]|=UINT64_C(1)<<plane;
}

int mesh_realize(struct mesh_ctx *c,struct mesh_row_function *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count){
  struct hdr *m=c->M;
  uint32_t rows=mesh_rows(m),block=m->block;
  uint64_t *used=malloc(rows*sizeof *used);
  size_t *order=malloc((binding_count?binding_count:1)*sizeof *order);
  if(!used || !order){ free(used); free(order); return ENOMEM; }
  memcpy(used,mesh_mask(m),rows*sizeof *used);
  int error=0;
  for(size_t i=0;i<count && !error;i++){
    struct mesh_row_function *f=&functions[i];
    if(!f->rows || !f->outputs || !f->output || (f->inputs && !f->input)){ error=EINVAL; break; }
    for(uint32_t j=0;j<f->inputs && !error;j++){
      uint64_t busy=0; int plane=0;
      for(uint32_t k=0;k<f->rows && !error;k++){ struct mesh_row_range r=mesh_range(f->input[j],k); error=mesh_survey(c,used,r.first,r.count,&busy); }
      if(!error) error=mesh_free_plane(busy,&plane);
      for(uint32_t k=0;k<f->rows && !error;k++){ struct mesh_row_range r=mesh_range(f->input[j],k); mesh_take(c,used,r.first,r.count,plane); }
      f->input[j].plane=(uint32_t)plane;
    }
    for(struct mesh_indexed_read *d=f->indexed;d && !error;d=d->next){
      for(uint32_t j=0;j<d->selectors;j++)for(uint32_t r=d->selector[j].first;r<d->selector[j].first+d->selector[j].count;r++)if(mesh_is(m,MESH_CONSTANT,r))error=EINVAL;
      for(uint32_t k=0;k<=d->candidates && !error;k++){
        struct mesh_row_map *maps=k==d->candidates?d->selector:d->candidate[k].maps;
        uint32_t n=k==d->candidates?d->selectors:d->candidate[k].count;
        uint64_t busy=0;int plane=0;
        for(uint32_t j=0;j<n && !error;j++)error=mesh_survey(c,used,maps[j].first,maps[j].count,&busy);
        if(!error)error=mesh_free_plane(busy,&plane);
        for(uint32_t j=0;j<n && !error;j++){mesh_take(c,used,maps[j].first,maps[j].count,plane);maps[j].plane=(uint32_t)plane;}
      }
    }
    for(uint32_t j=0;j<f->outputs && !error;j++) for(uint32_t k=0;k<f->rows && !error;k++){
      struct mesh_row_range r=mesh_range(f->output[j],k);
      if(!r.count || (uint64_t)r.first+r.count>rows){ error=EINVAL; break; }
      for(uint32_t x=0;x<r.count;x++) if(mesh_is(m,MESH_CONSTANT,r.first+x)) error=EINVAL;
    }
  }
  for(size_t i=0;i<return_count && !error;i++){
    uint64_t busy=0; int plane=0;
    struct mesh_row_range r=mesh_range(returns[i],0);
    error=mesh_survey(c,used,r.first,r.count,&busy);
    if(!error) error=mesh_free_plane(busy,&plane);
    if(!error) mesh_take(c,used,r.first,r.count,plane);
    returns[i].plane=(uint32_t)plane;
  }
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
    for(uint32_t r=0;r<rows;r++) if(mesh_is(m,MESH_ROW_OWN,r)) mesh_mask(m)[r]=used[r];
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
        mesh_transfers(m,queue,direction)[at]=(struct mesh_transfer){.local_row=b->first+k,.local_page=atomic_load_explicit(&table[b->first+k],memory_order_acquire),.peer_row=MESH_ABSENT,.peer_page=MESH_ABSENT,.peer_index=MESH_ABSENT,.binding=b->binding,.offset=k,.plane=b->plane,.index=at,.bytes=bytes};
        atomic_store_explicit(length,at+1,memory_order_release);
      }
    }
  }
  free(order); free(used);
  if(!error && binding_count)atomic_store_explicit(&m->configured,(uint32_t)getpid(),memory_order_release);
  return error;
}

void *mesh_row_data(struct mesh_ctx *c,uint32_t row){
  uint32_t page=atomic_load_explicit(&mesh_page(c->M)[row],memory_order_acquire);
  return page==MESH_ABSENT?NULL:mesh_at(c->M,page);
}

static int mesh_ready(struct hdr *m,uint32_t first,uint32_t count,uint32_t plane){
  _Atomic uint64_t *present=mesh_plane(m,MESH_PRESENT),*constant=mesh_plane(m,MESH_CONSTANT),*read=mesh_plane(m,MESH_READ+plane);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t k=mesh_word_mask(first,count,w);
    if((atomic_load_explicit(&present[w],memory_order_acquire)&k)!=k) return 0;
    uint64_t c=atomic_load_explicit(&constant[w],memory_order_acquire);
    if(atomic_load_explicit(&read[w],memory_order_acquire)&k&~c) return 0;
  }
  return 1;
}

int mesh_present(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range r=mesh_range(map,index);
  return mesh_bits_all(c->M,MESH_PRESENT,r.first,r.count);
}

int mesh_available(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range r=mesh_range(map,index);
  return mesh_ready(c->M,r.first,r.count,map.plane);
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
      for(int p=0;need;p++,need>>=1) if((need&1) && !mesh_is(m,MESH_READ+p,row)) return 0;
    }
  }
  return 1;
}

/* design/algorithm-sources.md#xonotic-frame-migration */
int mesh_writable(struct mesh_ctx *c,uint32_t first,uint32_t count){return mesh_claimable(c->M,first,count);}

static void mesh_reset(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct hdr *m=c->M;
  mesh_bits_clear(m,MESH_PRESENT,first,count);
  for(int p=0;p<MESH_READERS;p++) mesh_bits_clear(m,MESH_READ+p,first,count);
  mesh_index_reset(c,first,count);
}

int mesh_republish(struct mesh_ctx *c,uint32_t first,uint32_t count){
  if(!mesh_claimable(c->M,first,count)) return 0;
  mesh_reset(c,first,count);
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  mesh_notify(c->M,first,count);
  return 1;
}

/* design/algorithm-sources.md#presence-driven-execution */
static int mesh_issue_index(struct mesh_ctx *c,const struct mesh_row_function *f,uint32_t index){
  struct hdr *m=c->M;
  for(uint32_t j=0;j<f->inputs;j++){
    struct mesh_row_range r=mesh_range(f->input[j],index);
    if(!mesh_ready(m,r.first,r.count,f->input[j].plane))return 0;
  }
  for(const struct mesh_indexed_read *d=f->indexed;d;d=d->next)if(!mesh_index_ready(c,d))return 0;
  for(uint32_t j=0;j<f->outputs;j++){
    struct mesh_row_range r=mesh_range(f->output[j],index);
    if(!mesh_claimable(m,r.first,r.count))return 0;
  }
  for(uint32_t j=0;j<f->outputs;j++){
    struct mesh_row_range r=mesh_range(f->output[j],index);
    mesh_reset(c,r.first,r.count);mesh_bits_set(m,MESH_PRODUCING,r.first,r.count);
  }
  return 1;
}

size_t mesh_issue(struct mesh_ctx *c,const struct mesh_row_function *f,uint32_t *indices,size_t capacity){
  size_t selected=0;
  for(uint32_t i=0;i<f->rows && selected<capacity;i++)
    if(mesh_issue_index(c,f,i))indices[selected++]=i;
  return selected;
}

/* ledger D7: publication marks produced send blocks for the bridge and returns */
static void mesh_publish(struct hdr *m,uint32_t first,uint32_t count){
  mesh_bits_set(m,MESH_PRESENT,first,count);
  mesh_bits_clear(m,MESH_PRODUCING,first,count);
  mesh_notify(m,first,count);
}

static void mesh_read(struct hdr *m,uint32_t first,uint32_t count,uint32_t plane){
  _Atomic uint64_t *read=mesh_plane(m,MESH_READ+plane),*constant=mesh_plane(m,MESH_CONSTANT);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t k=mesh_word_mask(first,count,w)&~atomic_load_explicit(&constant[w],memory_order_acquire);
    if(k) atomic_fetch_or_explicit(&read[w],k,memory_order_acq_rel);
  }
  mesh_notify(m,first,count);
}

void mesh_complete(struct mesh_ctx *c,const struct mesh_row_function *f,const uint32_t *indices,size_t count){
  for(size_t n=0;n<count;n++){
    uint32_t i=indices[n];
    for(const struct mesh_indexed_read *d=f->indexed;d;d=d->next)mesh_index_complete(c,d);
    for(uint32_t j=0;j<f->inputs;j++){ struct mesh_row_range r=mesh_range(f->input[j],i); mesh_read(c->M,r.first,r.count,f->input[j].plane); }
    for(uint32_t j=0;j<f->outputs;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); mesh_publish(c->M,r.first,r.count); }
  }
}

void mesh_consume(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range r=mesh_range(map,index);
  mesh_read(c->M,r.first,r.count,map.plane);
}

struct mesh_watch {
  struct mesh_row_function *function;
  void *owner,*argument;
  void (*submit)(void *,uint32_t);
  uint32_t index;
  struct mesh_watch *next,*pending_next;
  int pending;
};
struct mesh_edge { struct mesh_watch *watch; struct mesh_edge *next; uint32_t row,candidate; struct mesh_indexed_read *indexed; void *owner; };
struct mesh_execution {
  struct mesh_ctx *context;
  dispatch_queue_t queue;
  dispatch_source_t source;
  struct mesh_edge **readers;
  struct mesh_watch *watches;
  int socket;
};
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static int mesh_index_active(struct mesh_ctx *c,const struct mesh_indexed_read *d){
  for(uint32_t i=0;i<d->selectors;i++)if(!mesh_ready(c->M,d->selector[i].first,d->selector[i].count,d->selector[i].plane))return 0;
  return 1;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static int mesh_index_span(const struct mesh_indexed_read *d,size_t *first,size_t *end){
  *first=d->bounds?d->bounds[0]:0;*end=d->bounds?d->bounds[d->bounds_stride]:d->rows*d->columns;
  return *first<=*end && *end<=d->rows*d->columns;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static void mesh_index_prepare(struct mesh_ctx *c,const struct mesh_indexed_read *d){
  if(mesh_is(c->M,MESH_PRESENT,d->mapped))return;
  size_t first,end;if(!mesh_index_span(d,&first,&end))return;
  for(size_t i=first;i<end;i++){
    uint32_t index=d->indices[(i/d->columns)*d->row_stride+(i%d->columns)*d->column_stride];
    if(index<d->candidates)mesh_bits_set(c->M,MESH_PRESENT,d->selected+index,1);
  }
  mesh_bits_set(c->M,MESH_PRESENT,d->mapped,1);
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static void mesh_index_finish(struct mesh_ctx *c,const struct mesh_indexed_read *d){
  if(!mesh_bits_all(c->M,MESH_PRESENT,d->retired,d->candidates))return;
  for(uint32_t i=0;i<d->selectors;i++)mesh_read(c->M,d->selector[i].first,d->selector[i].count,d->selector[i].plane);
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static int mesh_index_retire(struct mesh_ctx *c,const struct mesh_indexed_read *d,uint32_t index){
  uint32_t row=d->retired+index,word=row/64;uint64_t bit=UINT64_C(1)<<(row%64);
  uint64_t before=atomic_fetch_or_explicit(&mesh_plane(c->M,MESH_PRESENT)[word],bit,memory_order_acq_rel);
  if(before&bit)return 0;
  struct mesh_index_candidate candidate=d->candidate[index];
  for(uint32_t j=0;j<candidate.count;j++)mesh_read(c->M,candidate.maps[j].first,candidate.maps[j].count,candidate.maps[j].plane);
  uint64_t mask=mesh_word_mask(d->retired,d->candidates,word);return ((before|bit)&mask)==mask;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static void mesh_index_event(struct mesh_ctx *c,const struct mesh_indexed_read *d,uint32_t index){
  if(!mesh_index_active(c,d))return;
  mesh_index_prepare(c,d);
  if(!mesh_is(c->M,MESH_PRESENT,d->mapped))return;
  uint32_t first=index==MESH_ABSENT?0:index,end=index==MESH_ABSENT?d->candidates:index+1;int finish=0;
  for(uint32_t i=first;i<end;i++){
    if(mesh_is(c->M,MESH_PRESENT,d->retired+i))continue;
    if(mesh_is(c->M,MESH_PRESENT,d->selected+i) && !mesh_is(c->M,MESH_PRESENT,d->completed))continue;
    struct mesh_index_candidate candidate=d->candidate[i];int present=1;
    for(uint32_t j=0;j<candidate.count;j++)present&=mesh_ready(c->M,candidate.maps[j].first,candidate.maps[j].count,candidate.maps[j].plane);
    if(present)finish|=mesh_index_retire(c,d,i);
  }
  if(finish)mesh_index_finish(c,d);
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static int mesh_index_ready(struct mesh_ctx *c,const struct mesh_indexed_read *d){
  for(uint32_t i=0;i<d->selectors;i++)if(!mesh_bits_all(c->M,MESH_PRESENT,d->selector[i].first,d->selector[i].count))return 0;
  size_t first,end;if(!mesh_index_span(d,&first,&end))return 0;
  for(size_t i=first;i<end;i++){
    uint32_t index=d->indices[(i/d->columns)*d->row_stride+(i%d->columns)*d->column_stride];
    if(index==MESH_ABSENT)continue;
    if(index>=d->candidates)return 0;
    struct mesh_index_candidate candidate=d->candidate[index];
    for(uint32_t k=0;k<candidate.count;k++)if(!mesh_ready(c->M,candidate.maps[k].first,candidate.maps[k].count,candidate.maps[k].plane))return 0;
  }
  return 1;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static void mesh_index_complete(struct mesh_ctx *c,const struct mesh_indexed_read *d){
  mesh_bits_set(c->M,MESH_PRESENT,d->completed,1);
  for(uint32_t i=0;i<d->selectors;i++)mesh_notify(c->M,d->selector[i].first,d->selector[i].count);
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static void mesh_index_reset(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct mesh_execution *e=c->execution;if(!e)return;
  for(uint32_t r=first;r<first+count;r++)for(struct mesh_edge *edge=e->readers[r];edge;edge=edge->next)
    if(edge->indexed && edge->candidate==MESH_ABSENT)mesh_bits_clear(c->M,MESH_PRESENT,edge->indexed->retired,2*edge->indexed->candidates+2);
}

static int mesh_signal_socket=-1;

/* design/algorithm-sources.md#presence-driven-execution */
int mesh_signal_init(void){
  if(mesh_signal_socket>=0)return 0;
  mesh_signal_socket=socket(AF_UNIX,SOCK_DGRAM,0);
  if(mesh_signal_socket<0)return errno;
  fcntl(mesh_signal_socket,F_SETFL,O_NONBLOCK);
  fcntl(mesh_signal_socket,F_SETFD,FD_CLOEXEC);
  return 0;
}
/* design/algorithm-sources.md#presence-driven-execution */
void mesh_notify(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t row=first;row<first+count;row++){
    mesh_notice_push(m,MESH_NOTICE_COMPUTE,row);
    mesh_notice_push(m,MESH_NOTICE_SEND,row);
  }
  struct sockaddr_un address={.sun_family=AF_UNIX};
  memcpy(address.sun_path,m->event_path,sizeof address.sun_path);
  unsigned char wake=0;
  if(address.sun_path[0])sendto(mesh_signal_socket,&wake,1,MSG_DONTWAIT,(struct sockaddr *)&address,sizeof address);
}
/* design/algorithm-sources.md#presence-driven-execution */
static void mesh_fire(struct mesh_execution *e,struct mesh_watch *watch){
  if(mesh_issue_index(e->context,watch->function,watch->index))watch->submit(watch->argument,watch->index);
}
/* design/algorithm-sources.md#presence-driven-execution */
static void mesh_events(struct mesh_execution *e){
  unsigned char bytes[256];
  while(recv(e->socket,bytes,sizeof bytes,MSG_DONTWAIT)>0){}
  struct hdr *m=e->context->M;
  struct mesh_watch *pending=NULL;
  uint32_t row=mesh_notice_take(m,MESH_NOTICE_COMPUTE);
  while(row!=MESH_ABSENT){
    uint32_t next=mesh_notice_next(m,MESH_NOTICE_COMPUTE,row);
    for(struct mesh_edge *edge=e->readers[row];edge;edge=edge->next){
      if(edge->indexed){mesh_index_event(e->context,edge->indexed,edge->candidate);continue;}
      struct mesh_watch *watch=edge->watch;
      if(!watch->pending){watch->pending=1;watch->pending_next=pending;pending=watch;}
    }
    row=next;
  }
  while(pending){
    struct mesh_watch *watch=pending;pending=watch->pending_next;
    watch->pending=0;mesh_fire(e,watch);
  }
}
/* design/algorithm-sources.md#presence-driven-execution */
static int mesh_execution_create(struct mesh_ctx *c){
  struct mesh_execution *e=calloc(1,sizeof *e);if(!e)return ENOMEM;
  e->context=c;e->socket=socket(AF_UNIX,SOCK_DGRAM,0);
  e->readers=calloc(mesh_rows(c->M),sizeof *e->readers);
  if(e->socket<0 || !e->readers){int error=errno?errno:ENOMEM;if(e->socket>=0)close(e->socket);free(e->readers);free(e);return error;}
  fcntl(e->socket,F_SETFL,O_NONBLOCK);fcntl(e->socket,F_SETFD,FD_CLOEXEC);
  struct sockaddr_un address={.sun_family=AF_UNIX};
  snprintf(address.sun_path,sizeof address.sun_path,"/tmp/mesh-events.%d",getpid());
  unlink(address.sun_path);
  if(bind(e->socket,(struct sockaddr *)&address,sizeof address)){int error=errno;close(e->socket);free(e->readers);free(e);return error;}
  memcpy(c->M->event_path,address.sun_path,sizeof address.sun_path);
  e->queue=dispatch_queue_create("mesh.presence",DISPATCH_QUEUE_SERIAL);
  e->source=dispatch_source_create(DISPATCH_SOURCE_TYPE_READ,(uintptr_t)e->socket,0,e->queue);
  dispatch_source_set_event_handler(e->source,^{mesh_events(e);});
  c->execution=e;dispatch_resume(e->source);return 0;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
int mesh_execution_indexed(struct mesh_ctx *c,struct mesh_indexed_read *d,void *owner){
  if(!c->execution){int error=mesh_execution_create(c);if(error)return error;}
  struct mesh_execution *e=c->execution;struct mesh_edge *edges=NULL;
  for(uint32_t i=0;i<=d->candidates;i++){
    struct mesh_row_map *maps=i==d->candidates?d->selector:d->candidate[i].maps;
    uint32_t count=i==d->candidates?d->selectors:d->candidate[i].count;
    for(uint32_t j=0;j<count;j++)for(uint32_t r=maps[j].first;r<maps[j].first+maps[j].count;r++){
      struct mesh_edge *edge=calloc(1,sizeof *edge);
      if(!edge){while(edges){struct mesh_edge *next=edges->next;free(edges);edges=next;}return ENOMEM;}
      *edge=(struct mesh_edge){.next=edges,.row=r,.candidate=i==d->candidates?MESH_ABSENT:i,.indexed=d,.owner=owner};edges=edge;
    }
  }
  dispatch_sync(e->queue,^{struct mesh_edge *edge=edges;while(edge){struct mesh_edge *next=edge->next;edge->next=e->readers[edge->row];e->readers[edge->row]=edge;edge=next;}mesh_index_event(c,d,MESH_ABSENT);});
  return 0;
}

/* design/algorithm-sources.md#presence-driven-execution */
int mesh_execution_add(struct mesh_ctx *c,struct mesh_row_function *function,void *owner,void (*submit)(void *,uint32_t),void *argument){
  if(!c->execution){int error=mesh_execution_create(c);if(error)return error;}
  struct mesh_execution *e=c->execution;
  struct mesh_watch *watches=NULL;
  struct mesh_edge *edges=NULL;
  int error=0;
  for(uint32_t index=0;index<function->rows && !error;index++){
    struct mesh_watch *watch=calloc(1,sizeof *watch);if(!watch){error=ENOMEM;break;}
    *watch=(struct mesh_watch){.function=function,.owner=owner,.argument=argument,.submit=submit,.index=index,.next=watches};
    watches=watch;
    for(uint32_t i=0;i<function->inputs+function->outputs && !error;i++){
      struct mesh_row_map map=i<function->inputs?function->input[i]:function->output[i-function->inputs];
      struct mesh_row_range range=mesh_range(map,index);
      for(uint32_t row=range.first;row<range.first+range.count;row++){
        struct mesh_edge *edge=malloc(sizeof *edge);if(!edge){error=ENOMEM;break;}
        *edge=(struct mesh_edge){.watch=watch,.next=edges,.row=row};edges=edge;
      }
    }
  }
  for(struct mesh_watch *watch=watches;watch && !error;watch=watch->next)
    for(struct mesh_indexed_read *d=function->indexed;d && !error;d=d->next)
      for(uint32_t i=0;i<d->candidates && !error;i++)for(uint32_t j=0;j<d->candidate[i].count && !error;j++){
        struct mesh_row_map map=d->candidate[i].maps[j];
        for(uint32_t row=map.first;row<map.first+map.count;row++){
          struct mesh_edge *edge=calloc(1,sizeof *edge);if(!edge){error=ENOMEM;break;}
          *edge=(struct mesh_edge){.watch=watch,.next=edges,.row=row};edges=edge;
        }
      }
  if(error){
    while(edges){struct mesh_edge *next=edges->next;free(edges);edges=next;}
    while(watches){struct mesh_watch *next=watches->next;free(watches);watches=next;}
    return error;
  }
  dispatch_sync(e->queue,^{
    struct mesh_edge *edge=edges;
    while(edge){struct mesh_edge *next=edge->next;edge->next=e->readers[edge->row];e->readers[edge->row]=edge;edge=next;}
    struct mesh_watch *watch=watches;
    while(watch){struct mesh_watch *next=watch->next;watch->next=e->watches;e->watches=watch;mesh_fire(e,watch);watch=next;}
  });
  return 0;
}
/* design/algorithm-sources.md#presence-driven-execution */
void mesh_execution_remove(struct mesh_ctx *c,void *owner){
  struct mesh_execution *e=c->execution;if(!e)return;
  dispatch_sync(e->queue,^{
    for(uint32_t row=0;row<mesh_rows(c->M);row++){
      struct mesh_edge **at=&e->readers[row];
      while(*at){struct mesh_edge *edge=*at;if((edge->indexed?edge->owner:edge->watch->owner)==owner){*at=edge->next;free(edge);}else at=&edge->next;}
    }
    struct mesh_watch **at=&e->watches;
    while(*at){struct mesh_watch *watch=*at;if(watch->owner==owner){*at=watch->next;free(watch);}else at=&watch->next;}
  });
}
/* design/algorithm-sources.md#presence-driven-execution */
static void mesh_execution_destroy(struct mesh_ctx *c){
  struct mesh_execution *e=c->execution;if(!e)return;
  dispatch_semaphore_t stopped=dispatch_semaphore_create(0);
  dispatch_source_set_cancel_handler(e->source,^{dispatch_semaphore_signal(stopped);});
  dispatch_source_cancel(e->source);
  dispatch_semaphore_wait(stopped,DISPATCH_TIME_FOREVER);
  dispatch_release(stopped);
  close(e->socket);unlink(c->M->event_path);c->M->event_path[0]=0;
  for(uint32_t row=0;row<mesh_rows(c->M);row++){
    struct mesh_edge *edge=e->readers[row];while(edge){struct mesh_edge *next=edge->next;free(edge);edge=next;}
  }
  struct mesh_watch *watch=e->watches;while(watch){struct mesh_watch *next=watch->next;free(watch);watch=next;}
  dispatch_release(e->source);dispatch_release(e->queue);free(e->readers);free(e);c->execution=NULL;
}

/* design/algorithm-sources.md#performance-evidence */
size_t mesh_transfer_trace_count(struct mesh_ctx *c){
  size_t count=0;
  for(uint32_t q=0;q<c->M->qps;q++)for(int d=0;d<2;d++)count+=atomic_load_explicit(mesh_order_length(c->M,q,d),memory_order_acquire);
  return count;
}
/* design/algorithm-sources.md#performance-evidence */
struct mesh_transfer_event mesh_transfer_trace(struct mesh_ctx *c,size_t index){
  for(uint32_t q=0;q<c->M->qps;q++)for(int d=0;d<2;d++){
    size_t count=atomic_load_explicit(mesh_order_length(c->M,q,d),memory_order_acquire);
    if(index>=count){index-=count;continue;}
    struct mesh_transfer_times *t=mesh_transfer_times(c->M,q,d,(uint32_t)index);
    return (struct mesh_transfer_event){.queue=q,.direction=(uint32_t)d,.transfer=mesh_transfers(c->M,q,d)[index],
      .ready_ns=atomic_load(&t->ready_ns),.post_ns=atomic_load(&t->post_ns),.cq_ns=atomic_load(&t->cq_ns),.occurrences=atomic_load(&t->occurrences)};
  }
  return (struct mesh_transfer_event){0};
}

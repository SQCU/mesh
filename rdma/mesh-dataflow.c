#include <signal.h>
#include "mesh-dataflow.h"
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
/* design/pages-and-functions.md#what-the-page-table-is */

static struct mesh_ctx CTX0;
struct mesh_ctx *mesh_context(void){ return &CTX0; }
struct hdr *mesh_region(struct mesh_ctx *c){ return c->M; }

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
  uint64_t vacant=0;
  while(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)){
    if(!vacant || !kill((pid_t)vacant,0) || errno!=ESRCH){ munmap(memory,(size_t)info.st_size); close(file); return EBUSY; }
    if(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)) continue;
    for(uint32_t b=0;b<MESH_BINDINGS;b++) atomic_store_explicit(&mesh_base(memory)[b],MESH_ABSENT,memory_order_release);
    mesh_retire_rows(memory);
    mesh_bits_clear(memory,MESH_PAGE_OWN,0,mesh_rows(memory));
    break;
  }
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.fd=file};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  for(uint32_t b=0;b<MESH_BINDINGS;b++) atomic_store_explicit(&mesh_base(c->M)[b],MESH_ABSENT,memory_order_release);
  mesh_retire_rows(c->M);
  mesh_bits_clear(c->M,MESH_PAGE_OWN,0,mesh_rows(c->M));
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
      uint64_t occupied=(atomic_load_explicit(&mesh_plane(c->M,own)[w],memory_order_acquire)|atomic_load_explicit(&mesh_plane(c->M,hot)[w],memory_order_acquire)|(own==MESH_ROW_OWN?(atomic_load_explicit(&mesh_plane(c->M,MESH_ROW_LANDED)[w],memory_order_acquire)|atomic_load_explicit(&mesh_plane(c->M,MESH_ROW_BOUND)[w],memory_order_acquire)):0))&mesh_word_mask(first,count,w);
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
    mesh_mask(c->M)[r]=0; mesh_send(c->M)[r]=0;
  }
  c->rows+=count;
  return first;
}

uint32_t mesh_landing_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count || count%c->M->block || c->landing+count>c->M->pool){ errno=ENOMEM; return MESH_ABSENT; }
  uint32_t first=mesh_rows_alloc(c,count);
  if(first!=MESH_ABSENT) c->landing+=count;
  return first;
}

uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align || (align&(align-1))){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,pages,align,c->M->pool,mesh_rows(c->M),MESH_PAGE_OWN,MESH_PAGE_HOT);
  if(first!=MESH_ABSENT) c->arena+=pages;
  return first;
}

void mesh_map(struct mesh_ctx *c,uint32_t first,uint32_t count,uint32_t page){
  _Atomic uint32_t *table=mesh_page(c->M);
  for(uint32_t i=0;i<count;i++) atomic_store_explicit(&table[first+i],page+i,memory_order_release);
}

void mesh_constant(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_set(c->M,MESH_CONSTANT,first,count);
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
}

static int mesh_is(struct hdr *m,int plane,uint32_t row){
  return (atomic_load_explicit(&mesh_plane(m,plane)[row/64],memory_order_acquire)>>(row%64))&1;
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
  uint64_t *used=calloc(rows,sizeof *used);
  if(!used) return ENOMEM;
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
  for(size_t i=0;i<binding_count && !error;i++){
    struct mesh_row_binding *b=&bindings[i];
    if(!b->count || b->count%block || (uint64_t)b->first+b->count>rows || b->binding>=MESH_BINDINGS){ error=EINVAL; break; }
    if(b->receive){
      for(uint32_t k=0;k<b->count;k+=block) mesh_send(m)[b->first+k]|=0x80;
      continue;
    }
    uint64_t busy=0; int plane=0;
    error=mesh_survey(c,used,b->first,b->count,&busy);
    if(!error) error=mesh_free_plane(busy,&plane);
    if(error) break;
    mesh_take(c,used,b->first,b->count,plane);
    b->plane=(uint32_t)plane;
    for(uint32_t k=0;k<b->count/block;k++){
      uint32_t row=b->first+k*block;
      uint32_t page=atomic_load_explicit(&mesh_page(m)[row+block-1],memory_order_acquire);
      if(page==MESH_ABSENT){ error=EINVAL; break; }
      mesh_send(m)[row]=(uint8_t)(0x40|plane);
      *(struct mesh_tag*)mesh_at(m,page)=(struct mesh_tag){MESH_TAG,b->binding,k,0};
    }
  }
  if(!error){
    for(uint32_t r=0;r<rows;r++) if(mesh_is(m,MESH_ROW_OWN,r)) mesh_mask(m)[r]=used[r];
    for(size_t i=0;i<binding_count;i++) if(bindings[i].receive){
      struct mesh_row_binding *b=&bindings[i];
      mesh_bits_set(m,MESH_ROW_BOUND,b->first,b->count);
      atomic_store_explicit(&mesh_base(m)[b->binding],b->first,memory_order_release);
    }
  }
  free(used);
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

static void mesh_reset(struct hdr *m,uint32_t first,uint32_t count){
  mesh_bits_clear(m,MESH_PRESENT,first,count);
  for(int p=0;p<MESH_READERS;p++) mesh_bits_clear(m,MESH_READ+p,first,count);
}

int mesh_republish(struct mesh_ctx *c,uint32_t first,uint32_t count){
  if(!mesh_claimable(c->M,first,count)) return 0;
  mesh_reset(c->M,first,count);
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  return 1;
}

size_t mesh_issue(struct mesh_ctx *c,const struct mesh_row_function *f,uint32_t *indices,size_t capacity){
  struct hdr *m=c->M;
  size_t selected=0;
  for(uint32_t i=0;i<f->rows && selected<capacity;i++){
    int ready=1;
    for(uint32_t j=0;j<f->inputs && ready;j++){ struct mesh_row_range r=mesh_range(f->input[j],i); ready=mesh_ready(m,r.first,r.count,f->input[j].plane); }
    for(uint32_t j=0;j<f->outputs && ready;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); ready=mesh_claimable(m,r.first,r.count); }
    if(!ready) continue;
    for(uint32_t j=0;j<f->outputs;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); mesh_reset(m,r.first,r.count); mesh_bits_set(m,MESH_PRODUCING,r.first,r.count); }
    indices[selected++]=i;
  }
  return selected;
}

static void mesh_publish(struct hdr *m,uint32_t first,uint32_t count){
  mesh_bits_set(m,MESH_PRESENT,first,count);
  uint8_t *send=mesh_send(m);
  for(uint32_t r=first;r<first+count;r++) if(send[r]&0x40){
    uint32_t page=atomic_load_explicit(&mesh_page(m)[r],memory_order_acquire);
    mesh_bits_set(m,MESH_PAGE_HOT,page,m->block);
    mesh_bits_set(m,MESH_ROW_HOT,r,m->block);
    uint64_t entry=mesh_submission(page,r,send[r]&63);
    mesh_push(m,SUB,entry);
  }
  mesh_bits_clear(m,MESH_PRODUCING,first,count);
}

static void mesh_read(struct hdr *m,uint32_t first,uint32_t count,uint32_t plane){
  _Atomic uint64_t *read=mesh_plane(m,MESH_READ+plane),*constant=mesh_plane(m,MESH_CONSTANT);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t k=mesh_word_mask(first,count,w)&~atomic_load_explicit(&constant[w],memory_order_acquire);
    if(k) atomic_fetch_or_explicit(&read[w],k,memory_order_acq_rel);
  }
}

void mesh_complete(struct mesh_ctx *c,const struct mesh_row_function *f,const uint32_t *indices,size_t count){
  for(size_t n=0;n<count;n++){
    uint32_t i=indices[n];
    for(uint32_t j=0;j<f->inputs;j++){ struct mesh_row_range r=mesh_range(f->input[j],i); mesh_read(c->M,r.first,r.count,f->input[j].plane); }
    for(uint32_t j=0;j<f->outputs;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); mesh_publish(c->M,r.first,r.count); }
  }
}

void mesh_consume(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range r=mesh_range(map,index);
  mesh_read(c->M,r.first,r.count,map.plane);
}

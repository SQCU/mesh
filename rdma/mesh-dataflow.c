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

/* A client's table state lives only while it is attached. Attach and detach both leave the table as the
   bridge expects it: no rows mapped, no bits set, no bindings, every delivered landing block back on the
   free index. The bridge itself never changes across clients. */
static void mesh_clear(struct hdr *m){
  uint32_t rows=mesh_rows(m),words=mesh_words(m);
  for(int plane=0;plane<MESH_PLANES;plane++) for(uint32_t w=0;w<words;w++) atomic_store_explicit(&mesh_plane(m,plane)[w],0,memory_order_relaxed);
  for(uint32_t r=0;r<rows;r++) atomic_store_explicit(&mesh_page(m)[r],MESH_ABSENT,memory_order_relaxed);
  for(uint32_t r=0;r<rows;r++){ mesh_mask(m)[r]=0; mesh_send(m)[r]=0; }
  for(uint32_t b=0;b<MESH_BINDINGS;b++) mesh_base(m)[b]=MESH_ABSENT;
  mesh_reclaim_landed(m);
  atomic_thread_fence(memory_order_release);
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
  close(file);
  if(memory==MAP_FAILED) return error;
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION || memory->length>(uint64_t)info.st_size){
    munmap(memory,(size_t)info.st_size); return EINVAL;
  }
  uint64_t vacant=0;
  if(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)){
    munmap(memory,(size_t)info.st_size); return EBUSY;
  }
  /* The previous client's committed sends clear the link before this client may touch the arena. */
  while(atomic_load_explicit(&memory->r[SUB].head,memory_order_acquire)!=atomic_load_explicit(&memory->r[SUB].tail,memory_order_acquire) ||
        atomic_load_explicit(&memory->sending,memory_order_acquire)) usleep(100);
  atomic_fetch_add_explicit(&memory->generation,1,memory_order_acq_rel);
  mesh_clear(memory);
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_clear(c->M);
  atomic_store_explicit(&c->M->client,0,memory_order_release);
  int status=munmap(c->M,c->len);
  if(status) return errno;
  *c=(struct mesh_ctx){0};
  return 0;
}

struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *c,size_t index){
  const struct mesh_port_info *port=&c->M->port;
  return (struct mesh_row_metadata){.when=port->when,.function=UINT32_MAX,.index=(uint32_t)index,
    .peer=port->peer,.code=port->code,.domain=port->domain};
}

uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count || c->rows+count>mesh_rows(c->M)){ errno=ENOMEM; return MESH_ABSENT; }
  uint32_t first=c->rows; c->rows+=count; return first;
}

uint32_t mesh_landing_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count || count%c->M->block || c->landing+count>c->M->pool){ errno=ENOMEM; return MESH_ABSENT; }
  uint32_t first=mesh_rows_alloc(c,count);
  if(first!=MESH_ABSENT) c->landing+=count;
  return first;
}

uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align || (align&(align-1))){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=(c->arena+align-1)/align*align;
  if(first>c->M->arena || pages>c->M->arena-first){ errno=ENOMEM; return MESH_ABSENT; }
  c->arena=first+pages;
  return c->M->pool+first;
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

/* Reader planes: each reader of a range takes the lowest plane free on every row of the range;
   masks record the planes each row must see. Readers of overlapping sub-ranges therefore never collide. */
static int mesh_survey(struct mesh_ctx *c,const uint8_t *used,uint32_t first,uint32_t count,uint8_t *busy){
  if(!count || (uint64_t)first+count>mesh_rows(c->M)) return EINVAL;
  for(uint32_t r=first;r<first+count;r++) if(!mesh_is(c->M,MESH_CONSTANT,r)) *busy|=used[r];
  return 0;
}
static int mesh_free_plane(uint8_t busy,int *plane){
  for(int p=0;p<MESH_READERS;p++) if(!(busy&(1u<<p))){ *plane=p; return 0; }
  return ENOSPC;
}
static void mesh_take(struct mesh_ctx *c,uint8_t *used,uint32_t first,uint32_t count,int plane){
  for(uint32_t r=first;r<first+count;r++) if(!mesh_is(c->M,MESH_CONSTANT,r)) used[r]|=(uint8_t)(1u<<plane);
}

int mesh_realize(struct mesh_ctx *c,struct mesh_row_function *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count){
  struct hdr *m=c->M;
  uint32_t rows=mesh_rows(m),block=m->block;
  uint8_t *used=calloc(rows,1);
  if(!used) return ENOMEM;
  int error=0;
  for(size_t i=0;i<count && !error;i++){
    struct mesh_row_function *f=&functions[i];
    if(!f->rows || !f->outputs || !f->output || (f->inputs && !f->input)){ error=EINVAL; break; }
    for(uint32_t j=0;j<f->inputs && !error;j++){
      uint8_t busy=0; int plane=0;
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
    uint8_t busy=0; int plane=0;
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
      mesh_base(m)[b->binding]=b->first;
      for(uint32_t k=0;k<b->count;k+=block) mesh_send(m)[b->first+k]|=0x80;
      continue;
    }
    uint8_t busy=0; int plane=0;
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
  if(!error) for(uint32_t r=0;r<rows;r++) mesh_mask(m)[r]=used[r];
  free(used);
  return error;
}

void *mesh_row_data(struct mesh_ctx *c,uint32_t row){
  uint32_t page=atomic_load_explicit(&mesh_page(c->M)[row],memory_order_acquire);
  return page==MESH_ABSENT?NULL:mesh_at(c->M,page);
}

int mesh_present(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range r=mesh_range(map,index);
  return mesh_bits_all(c->M,MESH_PRESENT,r.first,r.count);
}

/* An input range is ready when every row is present and this reader has not yet read it. */
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

/* An output range may be rewritten when it was never produced or every configured reader has read it. */
static int mesh_claimable(struct hdr *m,uint32_t first,uint32_t count){
  _Atomic uint64_t *present=mesh_plane(m,MESH_PRESENT);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t pending=atomic_load_explicit(&present[w],memory_order_acquire)&mesh_word_mask(first,count,w);
    while(pending){
      uint32_t row=w*64+(uint32_t)__builtin_ctzll(pending); pending&=pending-1;
      uint8_t need=mesh_mask(m)[row];
      for(int p=0;need;p++,need>>=1) if((need&1) && !mesh_is(m,MESH_READ+p,row)) return 0;
    }
  }
  return 1;
}

static void mesh_reset(struct hdr *m,uint32_t first,uint32_t count){
  mesh_bits_clear(m,MESH_PRESENT,first,count);
  mesh_bits_clear(m,MESH_FREED,first,count);
  for(int p=0;p<MESH_READERS;p++) mesh_bits_clear(m,MESH_READ+p,first,count);
}

size_t mesh_issue(struct mesh_ctx *c,const struct mesh_row_function *f,uint32_t *indices,size_t capacity){
  struct hdr *m=c->M;
  size_t selected=0;
  for(uint32_t i=0;i<f->rows && selected<capacity;i++){
    int ready=1;
    for(uint32_t j=0;j<f->inputs && ready;j++){ struct mesh_row_range r=mesh_range(f->input[j],i); ready=mesh_ready(m,r.first,r.count,f->input[j].plane); }
    for(uint32_t j=0;j<f->outputs && ready;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); ready=mesh_claimable(m,r.first,r.count); }
    if(!ready) continue;
    for(uint32_t j=0;j<f->outputs;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); mesh_reset(m,r.first,r.count); }
    indices[selected++]=i;
  }
  return selected;
}

static void mesh_publish(struct hdr *m,uint32_t first,uint32_t count){
  mesh_bits_set(m,MESH_PRESENT,first,count);
  uint8_t *send=mesh_send(m);
  uint64_t generation=atomic_load_explicit(&m->generation,memory_order_relaxed);
  for(uint32_t r=first;r<first+count;r++) if(send[r]&0x40)
    if(mesh_push(m,SUB,mesh_submission(atomic_load_explicit(&mesh_page(m)[r],memory_order_acquire),r,generation,send[r]&7))){ m->port.code=ENOBUFS; m->port.domain=1; }
}

/* Reading is an OR. A landing block whose every reader has read it returns its pages to the free index. */
static void mesh_read(struct hdr *m,uint32_t first,uint32_t count,uint32_t plane){
  _Atomic uint64_t *read=mesh_plane(m,MESH_READ+plane),*constant=mesh_plane(m,MESH_CONSTANT);
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t k=mesh_word_mask(first,count,w)&~atomic_load_explicit(&constant[w],memory_order_acquire);
    if(k) atomic_fetch_or_explicit(&read[w],k,memory_order_acq_rel);
  }
  uint8_t *send=mesh_send(m),*mask=mesh_mask(m);
  uint32_t block=m->block;
  for(uint32_t r=first;r<first+count;r++){
    if(!(send[r]&0x80)) continue;
    int done=1;
    for(uint32_t x=0;x<block && done;x++){
      uint8_t need=mask[r+x];
      for(int p=0;need && done;p++,need>>=1) if((need&1) && !mesh_is(m,MESH_READ+p,r+x)) done=0;
    }
    if(!done) continue;
    uint64_t bit=UINT64_C(1)<<(r%64);
    if(atomic_fetch_or_explicit(&mesh_plane(m,MESH_FREED)[r/64],bit,memory_order_acq_rel)&bit) continue;
    uint32_t page=atomic_load_explicit(&mesh_page(m)[r],memory_order_acquire);
    mesh_reset(m,r,block);
    if(mesh_push(m,FREE,page)){ m->port.code=ENOBUFS; m->port.domain=1; }
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

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
static int mesh_execution_create(struct mesh_ctx *);
static void mesh_reader_destroy(struct mesh_ctx *);
static void mesh_reader_release(struct mesh_ctx *,uint32_t,uint32_t);
static int mesh_reader_release_serial(struct mesh_ctx *,uint32_t,uint32_t);
static int mesh_reader_unbind_serial(struct mesh_ctx *,struct mesh_row_map *);
static void mesh_reader_reset(struct mesh_ctx *,uint32_t,uint32_t);
static void mesh_reader_event(struct mesh_ctx *,uint32_t);
static int mesh_map_ready(struct mesh_ctx *,struct mesh_row_map,uint32_t);
static void mesh_map_read(struct mesh_ctx *,struct mesh_row_map,uint32_t);
static int mesh_index_ready(struct mesh_ctx *,const struct mesh_indexed_read *);
static void mesh_index_complete(struct mesh_ctx *,const struct mesh_indexed_read *);
static int mesh_route_ready(struct mesh_ctx *,const struct mesh_route_use *);
static void mesh_route_complete(struct mesh_ctx *,const struct mesh_route_use *);
static int mesh_active_issue(struct mesh_ctx *,struct mesh_active *);
static void mesh_active_event(struct mesh_ctx *,struct mesh_active *);
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
  mesh_reader_destroy(c);
  mesh_retire(c->M);
  atomic_store_explicit(&c->M->client,0,memory_order_release);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#page-table-backing-assignment */
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

/* design/algorithm-sources.md#page-table-backing-assignment */
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

/* design/algorithm-sources.md#page-table-backing-assignment */
void mesh_backing_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  _Atomic uint32_t *pages=mesh_page(c->M);
  for(uint32_t offset=0;offset<count;offset++){
    uint32_t page=atomic_exchange_explicit(&pages[first+offset],MESH_ABSENT,memory_order_acq_rel);
    if(page!=MESH_ABSENT)mesh_arena_release(c,page,1);
  }
}

/* design/algorithm-sources.md#independent-configured-programs */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_reader_release(c,first,count);
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

static int mesh_free_plane(uint64_t busy,int *plane){
  for(int p=0;p<MESH_READERS;p++) if(!(busy&(UINT64_C(1)<<p))){ *plane=p; return 0; }
  return ENOSPC;
}
struct mesh_reader_pending {uint32_t *member;struct mesh_reader_pending *next;};
struct mesh_reader_chunk {uint32_t first,count,owners;struct mesh_reader_chunk *next;};
struct mesh_reader_group {uint32_t plane,completed,pending_count;struct mesh_reader_pending *pending;struct mesh_reader_chunk *chunks;};
struct mesh_reader_storage {uint32_t *members,*sources;size_t *offsets,count;struct mesh_reader_storage *next;};
struct mesh_readers {struct mesh_reader_group **groups;struct mesh_reader_storage *storage;};
/* design/algorithm-sources.md#canonical-reader-groups */
static int mesh_reader_bind(struct mesh_ctx *c,uint64_t *used,const uint32_t *fanout,struct mesh_row_map *map,uint32_t occurrences){
  uint64_t busy=0;int grouped=0,plane=0;
  for(uint32_t i=0;i<occurrences;i++){
    struct mesh_row_range range=mesh_range(*map,i);
    for(uint32_t row=range.first;row<range.first+range.count;row++)if(!mesh_is(c->M,MESH_CONSTANT,row)){busy|=used[row];grouped|=fanout[row]>MESH_READERS;}
  }
  for(uint32_t i=0;i<occurrences && !grouped;i++)for(uint32_t j=0;j<i && !grouped;j++){
    struct mesh_row_range a=mesh_range(*map,i),b=mesh_range(*map,j);
    uint32_t first=a.first>b.first?a.first:b.first,end=a.first+a.count<b.first+b.count?a.first+a.count:b.first+b.count;
    for(uint32_t row=first;row<end;row++)if(!mesh_is(c->M,MESH_CONSTANT,row)){grouped=1;break;}
  }
  if(!grouped && !mesh_free_plane(busy,&plane)){
    map->plane=(uint32_t)plane;
    for(uint32_t i=0;i<occurrences;i++){
      struct mesh_row_range range=mesh_range(*map,i);
      for(uint32_t row=range.first;row<range.first+range.count;row++)if(!mesh_is(c->M,MESH_CONSTANT,row))used[row]|=UINT64_C(1)<<plane;
    }
    return 0;
  }
  struct mesh_readers *readers=c->readers;
  if(!readers){
    readers=calloc(1,sizeof *readers);if(!readers)return ENOMEM;
    readers->groups=calloc(mesh_rows(c->M),sizeof *readers->groups);
    if(!readers->groups){free(readers);return ENOMEM;}c->readers=readers;
  }
  struct mesh_reader_storage *storage=calloc(1,sizeof *storage);if(!storage)return ENOMEM;
  storage->offsets=calloc(occurrences,sizeof *storage->offsets);
  if(!storage->offsets){free(storage);return ENOMEM;}
  size_t total=0;
  for(uint32_t i=0;i<occurrences;i++){
    struct mesh_row_range range=mesh_range(*map,i);storage->offsets[i]=total;
    if(!range.count || (uint64_t)range.first+range.count>mesh_rows(c->M)){free(storage->offsets);free(storage);return EINVAL;}
    total+=range.count;
  }
  storage->members=malloc(total*sizeof *storage->members);storage->sources=malloc(total*sizeof *storage->sources);storage->count=total;
  if(!storage->members || !storage->sources){free(storage->members);free(storage->sources);free(storage->offsets);free(storage);return ENOMEM;}
  for(size_t i=0;i<total;i++){storage->members[i]=MESH_ABSENT;storage->sources[i]=MESH_ABSENT;}
  storage->next=readers->storage;readers->storage=storage;
  map->members=storage->members;map->member_offsets=storage->offsets;map->plane=MESH_ABSENT;
  for(uint32_t i=0;i<occurrences;i++){
    struct mesh_row_range range=mesh_range(*map,i);
    for(uint32_t j=0;j<range.count;j++){
      uint32_t row=range.first+j,*member=&storage->members[storage->offsets[i]+j];storage->sources[storage->offsets[i]+j]=row;
      if(mesh_is(c->M,MESH_CONSTANT,row))continue;
      struct mesh_reader_group *group=readers->groups[row];
      if(!group){
        int plane,error=mesh_free_plane(used[row],&plane);if(error)return error;
        group=calloc(1,sizeof *group);if(!group)return ENOMEM;group->plane=(uint32_t)plane;
        group->completed=mesh_rows_alloc(c,1);if(group->completed==MESH_ABSENT){free(group);return errno;}
        readers->groups[row]=group;used[row]|=UINT64_C(1)<<plane;
      }
      struct mesh_reader_pending *pending=malloc(sizeof *pending);if(!pending)return ENOMEM;
      *pending=(struct mesh_reader_pending){.member=member,.next=group->pending};group->pending=pending;group->pending_count++;
    }
  }
  return 0;
}
/* design/algorithm-sources.md#canonical-reader-groups */
static int mesh_reader_realize(struct mesh_ctx *c){
  struct mesh_readers *readers=c->readers;if(!readers)return 0;
  for(uint32_t row=0;row<mesh_rows(c->M);row++){
    struct mesh_reader_group *group=readers->groups[row];if(!group || !group->pending_count)continue;
    struct mesh_reader_chunk *chunk=malloc(sizeof *chunk);if(!chunk)return ENOMEM;
    chunk->first=mesh_rows_alloc(c,group->pending_count);if(chunk->first==MESH_ABSENT){free(chunk);return errno;}
    chunk->count=chunk->owners=group->pending_count;chunk->next=group->chunks;group->chunks=chunk;
    uint32_t index=chunk->first;
    while(group->pending){struct mesh_reader_pending *pending=group->pending;*pending->member=index++;group->pending=pending->next;free(pending);}
    group->pending_count=0;
  }
  return c->execution?0:mesh_execution_create(c);
}
/* design/algorithm-sources.md#canonical-reader-groups */
static void mesh_reader_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  if(mesh_reader_release_serial(c,first,count))return;
  struct mesh_readers *readers=c->readers;if(!readers)return;
  for(uint32_t row=first;row<first+count;row++){
    struct mesh_reader_group *group=readers->groups[row];if(!group)continue;readers->groups[row]=NULL;
    while(group->pending){struct mesh_reader_pending *next=group->pending->next;free(group->pending);group->pending=next;}
    while(group->chunks){struct mesh_reader_chunk *next=group->chunks->next;mesh_rows_release(c,group->chunks->first,group->chunks->count);free(group->chunks);group->chunks=next;}
    mesh_rows_release(c,group->completed,1);free(group);
  }
}
/* design/algorithm-sources.md#canonical-reader-groups */
static void mesh_reader_destroy(struct mesh_ctx *c){
  struct mesh_readers *readers=c->readers;if(!readers)return;
  mesh_reader_release(c,0,mesh_rows(c->M));
  while(readers->storage){struct mesh_reader_storage *next=readers->storage->next;free(readers->storage->members);free(readers->storage->sources);free(readers->storage->offsets);free(readers->storage);readers->storage=next;}
  free(readers->groups);free(readers);c->readers=NULL;
}

/* design/algorithm-sources.md#canonical-reader-groups */
void mesh_reader_unbind(struct mesh_ctx *c,struct mesh_row_map *map){
  if(!map->members || mesh_reader_unbind_serial(c,map))return;
  struct mesh_readers *readers=c->readers;struct mesh_reader_storage **at=&readers->storage;
  while(*at && (*at)->members!=map->members)at=&(*at)->next;
  if(!*at)return;
  struct mesh_reader_storage *storage=*at;*at=storage->next;
  for(size_t i=0;i<storage->count;i++){
    uint32_t row=storage->sources[i];if(row==MESH_ABSENT)continue;
    struct mesh_reader_group *group=readers->groups[row];if(!group)continue;
    struct mesh_reader_pending **pending=&group->pending;
    while(*pending){
      if((*pending)->member==&storage->members[i]){struct mesh_reader_pending *old=*pending;*pending=old->next;free(old);group->pending_count--;}
      else pending=&(*pending)->next;
    }
    uint32_t member=storage->members[i];if(member==MESH_ABSENT)continue;
    struct mesh_reader_chunk **chunk=&group->chunks;
    while(*chunk && !(member>=(*chunk)->first && member<(*chunk)->first+(*chunk)->count))chunk=&(*chunk)->next;
    if(!*chunk)continue;
    mesh_bits_set(c->M,MESH_PRESENT,member,1);
    if(!--(*chunk)->owners){struct mesh_reader_chunk *old=*chunk;*chunk=old->next;mesh_rows_release(c,old->first,old->count);free(old);}
    mesh_reader_event(c,row);mesh_notify(c->M,row,1);
  }
  free(storage->members);free(storage->sources);free(storage->offsets);free(storage);
  map->members=NULL;map->member_offsets=NULL;
}

/* design/algorithm-sources.md#canonical-reader-groups */
static int mesh_reader_survey(struct mesh_ctx *c,uint32_t *fanout,struct mesh_row_map map,uint32_t occurrences){
  for(uint32_t i=0;i<occurrences;i++){
    struct mesh_row_range range=mesh_range(map,i);
    if(!range.count || (uint64_t)range.first+range.count>mesh_rows(c->M))return EINVAL;
    for(uint32_t row=range.first;row<range.first+range.count;row++)if(!mesh_is(c->M,MESH_CONSTANT,row) && fanout[row]<=MESH_READERS)fanout[row]++;
  }
  return 0;
}

int mesh_realize(struct mesh_ctx *c,struct mesh_row_function *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count){
  struct hdr *m=c->M;
  uint32_t rows=mesh_rows(m),block=m->block;
  uint64_t *used=malloc(rows*sizeof *used);
  uint32_t *fanout=calloc(rows,sizeof *fanout);
  size_t *order=malloc((binding_count?binding_count:1)*sizeof *order);
  if(!used || !order || !fanout){ free(used); free(order);free(fanout); return ENOMEM; }
  memcpy(used,mesh_mask(m),rows*sizeof *used);
  int error=0;
  for(uint32_t row=0;row<rows;row++)fanout[row]=(uint32_t)__builtin_popcountll(used[row]);
  for(size_t i=0;i<count && !error;i++){
    struct mesh_row_function *f=&functions[i];
    if(!f->rows || (f->inputs && !f->input)){error=EINVAL;break;}
    for(uint32_t j=0;j<f->inputs && !error;j++)error=mesh_reader_survey(c,fanout,f->input[j],f->rows);
    if(f->active)for(uint32_t j=0;j<f->active->maps && !error;j++)error=mesh_reader_survey(c,fanout,f->active->count_maps[j],1);
    for(struct mesh_route_use *u=f->routes;u && !error;u=u->next)if(!u->consumer){
      struct mesh_route *d=u->domain;
      for(uint32_t j=0;j<d->metadata_count && !error;j++)error=mesh_reader_survey(c,fanout,d->metadata[j],1);
      for(uint32_t j=0;j<d->candidates && !error;j++){for(uint32_t k=0;k<d->candidate[j].count && !error;k++)error=mesh_reader_survey(c,fanout,d->candidate[j].maps[k],1);if(d->candidate[j].producer && !error)error=mesh_reader_survey(c,fanout,d->candidate[j].disposition,1);}
    }
    for(struct mesh_indexed_read *d=f->indexed;d && !error;d=d->next){
      for(uint32_t j=0;j<d->selectors && !error;j++)error=mesh_reader_survey(c,fanout,d->selector[j],1);
      for(uint32_t j=0;j<d->candidates && !error;j++)for(uint32_t k=0;k<d->candidate[j].count && !error;k++)error=mesh_reader_survey(c,fanout,d->candidate[j].maps[k],1);
    }
  }
  for(size_t i=0;i<return_count && !error;i++)error=mesh_reader_survey(c,fanout,returns[i],1);
  for(size_t i=0;i<binding_count && !error;i++)if(!bindings[i].receive)error=mesh_reader_survey(c,fanout,(struct mesh_row_map){.first=bindings[i].first,.count=bindings[i].count},1);
  for(size_t i=0;i<count && !error;i++){
    struct mesh_row_function *f=&functions[i];
    if(!f->rows || !f->outputs || !f->output || (f->inputs && !f->input)){ error=EINVAL; break; }
    for(uint32_t j=0;j<f->inputs && !error;j++)error=mesh_reader_bind(c,used,fanout,&f->input[j],f->rows);
    if(f->active)for(uint32_t j=0;j<f->active->maps && !error;j++)error=mesh_reader_bind(c,used,fanout,&f->active->count_maps[j],1);
    for(struct mesh_route_use *u=f->routes;u && !error;u=u->next)if(!u->consumer){
      struct mesh_route *d=u->domain;
      for(uint32_t j=0;j<d->metadata_count && !error;j++)error=mesh_reader_bind(c,used,fanout,&d->metadata[j],1);
      for(uint32_t j=0;j<d->candidates && !error;j++){for(uint32_t k=0;k<d->candidate[j].count && !error;k++)error=mesh_reader_bind(c,used,fanout,&d->candidate[j].maps[k],1);if(d->candidate[j].producer && !error)error=mesh_reader_bind(c,used,fanout,&d->candidate[j].disposition,1);}
    }
    for(struct mesh_indexed_read *d=f->indexed;d && !error;d=d->next){
      for(uint32_t j=0;j<d->selectors;j++)for(uint32_t r=d->selector[j].first;r<d->selector[j].first+d->selector[j].count;r++)if(mesh_is(m,MESH_CONSTANT,r))error=EINVAL;
      for(uint32_t k=0;k<=d->candidates && !error;k++){
        struct mesh_row_map *maps=k==d->candidates?d->selector:d->candidate[k].maps;
        uint32_t n=k==d->candidates?d->selectors:d->candidate[k].count;
        for(uint32_t j=0;j<n && !error;j++)error=mesh_reader_bind(c,used,fanout,&maps[j],1);
      }
    }
    for(uint32_t j=0;j<f->outputs && !error;j++) for(uint32_t k=0;k<f->rows && !error;k++){
      struct mesh_row_range r=mesh_range(f->output[j],k);
      if(!r.count || (uint64_t)r.first+r.count>rows){ error=EINVAL; break; }
      for(uint32_t x=0;x<r.count;x++) if(mesh_is(m,MESH_CONSTANT,r.first+x)) error=EINVAL;
    }
  }
  for(size_t i=0;i<return_count && !error;i++)error=mesh_reader_bind(c,used,fanout,&returns[i],1);
  if(!error)error=mesh_reader_realize(c);
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
  free(order); free(used);free(fanout);
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
  return mesh_map_ready(c,map,index);
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
  if(f->active && !mesh_active_issue(c,f->active))return 0;
  for(uint32_t j=0;j<f->inputs;j++){
    if(!mesh_map_ready(c,f->input[j],index))return 0;
  }
  for(const struct mesh_indexed_read *d=f->indexed;d;d=d->next)if(!mesh_index_ready(c,d))return 0;
  for(const struct mesh_route_use *u=f->routes;u;u=u->next)if(!mesh_route_ready(c,u))return 0;
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

/* design/algorithm-sources.md#in-operation-publication */
void mesh_publish_partial(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_set(c->M,MESH_PRESENT,first,count);
  mesh_notify(c->M,first,count);
}

/* design/algorithm-sources.md#in-operation-publication */
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

/* design/algorithm-sources.md#canonical-reader-groups */
static int mesh_map_ready(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range range=mesh_range(map,index);
  if(!map.members)return mesh_ready(c->M,range.first,range.count,map.plane);
  if(!mesh_bits_all(c->M,MESH_PRESENT,range.first,range.count))return 0;
  const uint32_t *members=map.members+map.member_offsets[index];
  struct mesh_readers *readers=c->readers;
  for(uint32_t i=0;i<range.count;i++){
    if(members[i]==MESH_ABSENT)continue;
    struct mesh_reader_group *group=readers->groups[range.first+i];
    if(mesh_is(c->M,MESH_PRESENT,group->completed) || mesh_is(c->M,MESH_PRESENT,members[i]))return 0;
  }
  return 1;
}
/* design/algorithm-sources.md#canonical-reader-groups */
static void mesh_map_read(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  struct mesh_row_range range=mesh_range(map,index);
  if(!map.members){mesh_read(c->M,range.first,range.count,map.plane);return;}
  const uint32_t *members=map.members+map.member_offsets[index];
  for(uint32_t i=0;i<range.count;i++)if(members[i]!=MESH_ABSENT)mesh_bits_set(c->M,MESH_PRESENT,members[i],1);
  mesh_notify(c->M,range.first,range.count);
}
/* design/algorithm-sources.md#canonical-reader-groups */
static void mesh_reader_reset(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct mesh_readers *readers=c->readers;if(!readers)return;
  for(uint32_t row=first;row<first+count;row++){
    struct mesh_reader_group *group=readers->groups[row];if(!group)continue;
    for(struct mesh_reader_chunk *chunk=group->chunks;chunk;chunk=chunk->next)mesh_bits_clear(c->M,MESH_PRESENT,chunk->first,chunk->count);
    mesh_bits_clear(c->M,MESH_PRESENT,group->completed,1);
  }
}
/* design/algorithm-sources.md#canonical-reader-groups */
static void mesh_reader_event(struct mesh_ctx *c,uint32_t row){
  struct mesh_readers *readers=c->readers;if(!readers)return;
  struct mesh_reader_group *group=readers->groups[row];if(!group || !mesh_is(c->M,MESH_PRESENT,row))return;
  if(mesh_is(c->M,MESH_PRESENT,group->completed)){
    if(!mesh_is(c->M,MESH_READ+group->plane,row))mesh_reader_reset(c,row,1);
    return;
  }
  for(struct mesh_reader_chunk *chunk=group->chunks;chunk;chunk=chunk->next)if(!mesh_bits_all(c->M,MESH_PRESENT,chunk->first,chunk->count))return;
  mesh_bits_set(c->M,MESH_PRESENT,group->completed,1);
  mesh_read(c->M,row,1,group->plane);
}

/* design/algorithm-sources.md#in-operation-publication */
void mesh_complete(struct mesh_ctx *c,const struct mesh_row_function *f,const uint32_t *indices,size_t count){
  for(size_t n=0;n<count;n++){
    uint32_t i=indices[n];
    for(const struct mesh_indexed_read *d=f->indexed;d;d=d->next)mesh_index_complete(c,d);
    for(const struct mesh_route_use *u=f->routes;u;u=u->next)mesh_route_complete(c,u);
    for(uint32_t j=0;j<f->inputs;j++){if(f->active)mesh_bits_set(c->M,MESH_PRESENT,f->active->retired+j,1);mesh_map_read(c,f->input[j],i);}
    for(uint32_t j=0;j<f->outputs;j++){ struct mesh_row_range r=mesh_range(f->output[j],i); mesh_publish(c->M,r.first,r.count); }
    if(f->active)mesh_publish(c->M,f->active->disposition,1);
  }
}

/* design/algorithm-sources.md#active-segment-domains */
static void mesh_active_event(struct mesh_ctx *c,struct mesh_active *a){
  if(!mesh_is(c->M,MESH_PRESENT,a->disposition))return;
  for(uint32_t j=0;j<a->maps;j++)if(!mesh_map_ready(c,a->count_maps[j],0))return;
  if(mesh_is(c->M,MESH_PRESENT,a->omitted))for(uint32_t j=0;j<a->inputs;j++){
    if(mesh_is(c->M,MESH_PRESENT,a->retired+j) || !mesh_map_ready(c,a->function->input[j],0))continue;
    mesh_bits_set(c->M,MESH_PRESENT,a->retired+j,1);mesh_map_read(c,a->function->input[j],0);
  }
  if((a->inputs && !mesh_bits_all(c->M,MESH_PRESENT,a->retired,a->inputs)) || !mesh_claimable(c->M,a->disposition,1))return;
  for(uint32_t j=0;j<a->maps;j++)mesh_map_read(c,a->count_maps[j],0);
}
/* design/algorithm-sources.md#active-segment-domains */
static int mesh_active_issue(struct mesh_ctx *c,struct mesh_active *a){
  if(mesh_is(c->M,MESH_PRESENT,a->disposition)){mesh_active_event(c,a);return 0;}
  for(uint32_t j=0;j<a->maps;j++)if(!mesh_map_ready(c,a->count_maps[j],0))return 0;
  if(a->slot<*a->count)return 1;
  mesh_bits_set(c->M,MESH_PRESENT,a->omitted,1);atomic_fetch_add_explicit(&a->omissions,1,memory_order_relaxed);
  for(struct mesh_indexed_read *d=a->function->indexed;d;d=d->next)mesh_index_complete(c,d);
  mesh_publish(c->M,a->disposition,1);mesh_active_event(c,a);return 0;
}

void mesh_consume(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index){
  mesh_map_read(c,map,index);
}

struct mesh_watch {
  struct mesh_row_function *function;
  void *owner,*argument;
  void (*submit)(void *,uint32_t);
  uint32_t index;
  struct mesh_watch *next,*pending_next;
  int pending;
};
struct mesh_edge { struct mesh_watch *watch; struct mesh_edge *next; uint32_t row,candidate; struct mesh_indexed_read *indexed; struct mesh_route *route; struct mesh_active *active; uint32_t consumer,reset; void *owner; };
struct mesh_execution {
  struct mesh_ctx *context;
  dispatch_queue_t queue;
  dispatch_source_t source;
  struct mesh_edge **readers;
  struct mesh_watch *watches;
  int socket;
};
/* design/algorithm-sources.md#canonical-reader-groups */
static int mesh_reader_release_serial(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct mesh_execution *e=c->execution;if(!e || !c->readers || dispatch_get_specific(e)==e)return 0;
  dispatch_sync(e->queue,^{mesh_reader_release(c,first,count);});return 1;
}
/* design/algorithm-sources.md#canonical-reader-groups */
static int mesh_reader_unbind_serial(struct mesh_ctx *c,struct mesh_row_map *map){
  struct mesh_execution *e=c->execution;if(!e || dispatch_get_specific(e)==e)return 0;
  dispatch_sync(e->queue,^{mesh_reader_unbind(c,map);});return 1;
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static int mesh_index_active(struct mesh_ctx *c,const struct mesh_indexed_read *d){
  for(uint32_t i=0;i<d->selectors;i++)if(!mesh_map_ready(c,d->selector[i],0))return 0;
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
  for(uint32_t i=0;i<d->selectors;i++)mesh_map_read(c,d->selector[i],0);
}
/* design/algorithm-sources.md#dynamic-reader-lifetimes */
static int mesh_index_retire(struct mesh_ctx *c,const struct mesh_indexed_read *d,uint32_t index){
  uint32_t row=d->retired+index,word=row/64;uint64_t bit=UINT64_C(1)<<(row%64);
  uint64_t before=atomic_fetch_or_explicit(&mesh_plane(c->M,MESH_PRESENT)[word],bit,memory_order_acq_rel);
  if(before&bit)return 0;
  struct mesh_index_candidate candidate=d->candidate[index];
  for(uint32_t j=0;j<candidate.count;j++)mesh_map_read(c,candidate.maps[j],0);
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
    if(mesh_is(c->M,MESH_PRESENT,d->selected+i) && !mesh_is(c->M,MESH_PRESENT,d->completed) &&
       !(d->domain && mesh_is(c->M,MESH_PRESENT,d->domain->disposition) && mesh_is(c->M,MESH_PRESENT,d->domain->omitted)))continue;
    struct mesh_index_candidate candidate=d->candidate[i];int present=1;
    for(uint32_t j=0;j<candidate.count;j++)present&=mesh_map_ready(c,candidate.maps[j],0);
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
    for(uint32_t k=0;k<candidate.count;k++)if(!mesh_map_ready(c,candidate.maps[k],0))return 0;
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
    if(edge->active && edge->reset){
      struct mesh_active *a=edge->active;mesh_reset(c,a->disposition,1);mesh_bits_clear(c->M,MESH_PRESENT,a->omitted,1);if(a->inputs)mesh_bits_clear(c->M,MESH_PRESENT,a->retired,a->inputs);
    }else if(edge->indexed && edge->candidate==MESH_ABSENT)mesh_bits_clear(c->M,MESH_PRESENT,edge->indexed->retired,2*edge->indexed->candidates+2);
    else if(edge->route && edge->candidate==MESH_ABSENT && edge->consumer==MESH_ABSENT)mesh_bits_clear(c->M,MESH_PRESENT,edge->route->retired,edge->route->candidates+edge->route->consumers+1);
}

/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static uint32_t mesh_route_value(struct mesh_route_vector v,size_t index){
  return v.values[(index/v.columns)*v.row_stride+(index%v.columns)*v.column_stride];
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static int mesh_route_active(struct mesh_ctx *c,const struct mesh_route *d){
  for(uint32_t i=0;i<d->metadata_count;i++)if(!mesh_map_ready(c,d->metadata[i],0))return 0;
  return 1;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static int mesh_route_span(const struct mesh_route *d,uint32_t consumer,uint32_t *first,uint32_t *end){
  *first=mesh_route_value(d->offsets,consumer);*end=mesh_route_value(d->offsets,consumer+1);
  return *first<=*end && *end<=d->ordinals.length;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static int mesh_route_ready(struct mesh_ctx *c,const struct mesh_route_use *u){
  struct mesh_route *d=u->domain;
  if(!mesh_is(c->M,MESH_PRESENT,d->prepared) || mesh_is(c->M,MESH_PRESENT,d->completed+u->consumer))return 0;
  uint32_t first,end;if(!mesh_route_span(d,u->consumer,&first,&end))return 0;
  for(uint32_t i=first;i<end;i++){
    uint32_t index=mesh_route_value(d->ordinals,i);
    if(index>=d->candidates || mesh_route_value(d->owners,index)!=u->consumer || mesh_is(c->M,MESH_PRESENT,d->retired+index))return 0;
    struct mesh_index_candidate v=d->candidate[index];
    if(v.producer && (!mesh_map_ready(c,v.disposition,0) || mesh_is(c->M,MESH_PRESENT,v.producer->omitted)))return 0;
    for(uint32_t j=0;j<v.count;j++)if(!mesh_map_ready(c,v.maps[j],0))return 0;
  }
  return 1;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static void mesh_route_complete(struct mesh_ctx *c,const struct mesh_route_use *u){
  mesh_bits_set(c->M,MESH_PRESENT,u->domain->completed+u->consumer,1);
  mesh_notify(c->M,u->domain->completed+u->consumer,1);
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static int mesh_route_retire(struct mesh_ctx *c,const struct mesh_route *d,uint32_t index){
  if(mesh_is(c->M,MESH_PRESENT,d->retired+index))return 0;
  struct mesh_index_candidate v=d->candidate[index];
  if(v.producer && !mesh_map_ready(c,v.disposition,0))return 0;
  int omitted=v.producer && mesh_is(c->M,MESH_PRESENT,v.producer->omitted);
  if(!omitted)for(uint32_t j=0;j<v.count;j++)if(!mesh_map_ready(c,v.maps[j],0))return 0;
  mesh_bits_set(c->M,MESH_PRESENT,d->retired+index,1);
  if(!omitted)for(uint32_t j=0;j<v.count;j++)mesh_map_read(c,v.maps[j],0);
  if(v.producer)mesh_map_read(c,v.disposition,0);
  uint32_t word=(d->retired+index)/64;
  uint64_t mask=mesh_word_mask(d->retired,d->candidates,word);
  return (atomic_load_explicit(&mesh_plane(c->M,MESH_PRESENT)[word],memory_order_acquire)&mask)==mask;
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static void mesh_route_finish(struct mesh_ctx *c,const struct mesh_route *d){
  if(!mesh_bits_all(c->M,MESH_PRESENT,d->retired,d->candidates) || !mesh_bits_all(c->M,MESH_PRESENT,d->completed,d->consumers))return;
  mesh_bits_clear(c->M,MESH_PRESENT,d->prepared,1);
  for(uint32_t i=0;i<d->metadata_count;i++)mesh_map_read(c,d->metadata[i],0);
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static void mesh_route_pending(struct mesh_watch *watch,struct mesh_watch **pending){
  if(watch && !watch->pending){watch->pending=1;watch->pending_next=*pending;*pending=watch;}
}
/* design/algorithm-sources.md#shared-sparse-routing-lowering */
static void mesh_route_event(struct mesh_ctx *c,struct mesh_route *d,uint32_t index,uint32_t consumer,struct mesh_watch **pending){
  int finish=0;
  if(!mesh_is(c->M,MESH_PRESENT,d->prepared)){
    if(index!=MESH_ABSENT || consumer!=MESH_ABSENT || !mesh_route_active(c,d))return;
    for(uint32_t i=0;i<d->consumers;i++){
      uint32_t first,end;if(!mesh_route_span(d,i,&first,&end))return;
      for(uint32_t j=first;j<end;j++){
        uint32_t candidate=mesh_route_value(d->ordinals,j);
        if(candidate>=d->candidates || mesh_route_value(d->owners,candidate)!=i)return;
      }
    }
    mesh_bits_set(c->M,MESH_PRESENT,d->prepared,1);
    for(uint32_t i=0;i<d->candidates;i++)if(mesh_route_value(d->owners,i)==MESH_ABSENT)finish|=mesh_route_retire(c,d,i);
    for(uint32_t i=0;i<d->consumers;i++)mesh_route_pending(d->watches[i],pending);
  }
  if(consumer!=MESH_ABSENT){
    if(!mesh_is(c->M,MESH_PRESENT,d->completed+consumer))return;
    uint32_t first,end;if(!mesh_route_span(d,consumer,&first,&end))return;
    for(uint32_t j=first;j<end;j++)finish|=mesh_route_retire(c,d,mesh_route_value(d->ordinals,j));
    uint32_t word=(d->completed+consumer)/64;uint64_t mask=mesh_word_mask(d->completed,d->consumers,word);
    finish|=(atomic_load_explicit(&mesh_plane(c->M,MESH_PRESENT)[word],memory_order_acquire)&mask)==mask;
  }else if(index!=MESH_ABSENT && !mesh_is(c->M,MESH_PRESENT,d->retired+index)){
    uint32_t target=mesh_route_value(d->owners,index);
    if(target==MESH_ABSENT)finish|=mesh_route_retire(c,d,index);
    else if(target<d->consumers)mesh_route_pending(d->watches[target],pending);
  }
  if(finish)mesh_route_finish(c,d);
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
    mesh_reader_event(e->context,row);
    for(struct mesh_edge *edge=e->readers[row];edge;edge=edge->next){
      if(edge->active){mesh_active_event(e->context,edge->active);continue;}
      if(edge->route){mesh_route_event(e->context,edge->route,edge->candidate,edge->consumer,&pending);continue;}
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
  dispatch_queue_set_specific(e->queue,e,e,NULL);
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

/* design/algorithm-sources.md#shared-sparse-routing-lowering */
int mesh_execution_route(struct mesh_ctx *c,struct mesh_route *d,void *owner){
  if(!c->execution){int error=mesh_execution_create(c);if(error)return error;}
  struct mesh_execution *e=c->execution;struct mesh_edge *edges=NULL;
  for(uint32_t i=0;i<d->candidates+d->consumers+1;i++){
    struct mesh_row_map completion={.first=d->completed+i-d->candidates,.count=1};
    struct mesh_row_map *maps=i<d->candidates?d->candidate[i].maps:i<d->candidates+d->consumers?&completion:d->metadata;
    uint32_t count=i<d->candidates?d->candidate[i].count:i<d->candidates+d->consumers?1:d->metadata_count;
    uint32_t extra=i<d->candidates && d->candidate[i].producer?1:0;
    for(uint32_t j=0;j<count+extra;j++){
      struct mesh_row_map map=j<count?maps[j]:d->candidate[i].disposition;
      for(uint32_t r=map.first;r<map.first+map.count;r++){
      struct mesh_edge *edge=calloc(1,sizeof *edge);
      if(!edge){while(edges){struct mesh_edge *next=edges->next;free(edges);edges=next;}return ENOMEM;}
      *edge=(struct mesh_edge){.next=edges,.row=r,.candidate=i<d->candidates?i:MESH_ABSENT,.consumer=i>=d->candidates && i<d->candidates+d->consumers?i-d->candidates:MESH_ABSENT,.route=d,.owner=owner};edges=edge;
      }
    }
  }
  dispatch_sync(e->queue,^{struct mesh_edge *edge=edges;while(edge){struct mesh_edge *next=edge->next;edge->next=e->readers[edge->row];e->readers[edge->row]=edge;edge=next;}struct mesh_watch *pending=NULL;mesh_route_event(c,d,MESH_ABSENT,MESH_ABSENT,&pending);while(pending){struct mesh_watch *watch=pending;pending=watch->pending_next;watch->pending=0;mesh_fire(e,watch);}});
  return 0;
}

/* design/algorithm-sources.md#active-segment-domains */
int mesh_execution_active(struct mesh_ctx *c,struct mesh_active *a,void *owner){
  if(!c->execution){int error=mesh_execution_create(c);if(error)return error;}
  struct mesh_execution *e=c->execution;struct mesh_edge *edges=NULL;
  for(uint32_t i=0;i<=a->maps;i++){
    struct mesh_row_map map=i<a->maps?a->count_maps[i]:(struct mesh_row_map){.first=a->disposition,.count=1};
    for(uint32_t row=map.first;row<map.first+map.count;row++){
      struct mesh_edge *edge=calloc(1,sizeof *edge);
      if(!edge){while(edges){struct mesh_edge *next=edges->next;free(edges);edges=next;}return ENOMEM;}
      *edge=(struct mesh_edge){.next=edges,.row=row,.active=a,.reset=i<a->maps,.owner=owner};edges=edge;
    }
  }
  dispatch_sync(e->queue,^{struct mesh_edge *edge=edges;while(edge){struct mesh_edge *next=edge->next;edge->next=e->readers[edge->row];e->readers[edge->row]=edge;edge=next;}});return 0;
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
    if(function->active)for(uint32_t i=0;i<function->active->maps && !error;i++){
      struct mesh_row_map map=function->active->count_maps[i];
      for(uint32_t row=map.first;row<map.first+map.count;row++){
        struct mesh_edge *edge=calloc(1,sizeof *edge);if(!edge){error=ENOMEM;break;}
        *edge=(struct mesh_edge){.watch=watch,.next=edges,.row=row};edges=edge;
      }
    }
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
    while(watch){struct mesh_watch *next=watch->next;watch->next=e->watches;e->watches=watch;for(struct mesh_route_use *u=function->routes;u;u=u->next)u->domain->watches[u->consumer]=watch;mesh_fire(e,watch);watch=next;}
  });
  return 0;
}
/* design/algorithm-sources.md#presence-driven-execution */
void mesh_execution_remove(struct mesh_ctx *c,void *owner){
  struct mesh_execution *e=c->execution;if(!e)return;
  dispatch_sync(e->queue,^{
    for(uint32_t row=0;row<mesh_rows(c->M);row++){
      struct mesh_edge **at=&e->readers[row];
      while(*at){struct mesh_edge *edge=*at;if(((edge->indexed || edge->route || edge->active)?edge->owner:edge->watch->owner)==owner){*at=edge->next;free(edge);}else at=&edge->next;}
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

/* design/algorithm-sources.md#canonical-reader-groups */
struct mesh_reader_event mesh_reader_trace(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index,uint32_t row){
  struct mesh_row_range range=mesh_range(map,index);
  if(row>=range.count)return (struct mesh_reader_event){.source=MESH_ABSENT,.member=MESH_ABSENT,.plane=MESH_ABSENT,.completed=MESH_ABSENT};
  uint32_t source=range.first+row,member=map.members?map.members[map.member_offsets[index]+row]:MESH_ABSENT;
  struct mesh_readers *readers=c->readers;
  struct mesh_reader_group *group=member!=MESH_ABSENT && readers?readers->groups[source]:NULL;
  uint32_t plane=group?group->plane:map.plane,flags=mesh_is(c->M,MESH_PRESENT,source)?1:0;
  if(member!=MESH_ABSENT && mesh_is(c->M,MESH_PRESENT,member))flags|=2;
  if(plane<MESH_READERS && mesh_is(c->M,MESH_READ+plane,source))flags|=group?4:6;
  if(group && mesh_is(c->M,MESH_PRESENT,group->completed))flags|=8;
  return (struct mesh_reader_event){.source=source,.member=member,.plane=plane,.completed=group?group->completed:MESH_ABSENT,.flags=flags};
}

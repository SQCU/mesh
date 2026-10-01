#include <signal.h>
#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>

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

/* What a process allocated (rows and arena pages, each a run) and the request rings it registered, freed at its
   detach.  Shared by the contexts made from one attach (mesh_transfers_bank). */
static int mesh_owned_add(struct mesh_range **list,uint32_t *n,uint32_t *capacity,uint32_t first,uint32_t count){
  if(*n==*capacity){
    uint32_t grown=*capacity?2**capacity:64;
    struct mesh_range *bigger=realloc(*list,grown*sizeof *bigger);
    if(!bigger)return ENOMEM;
    *list=bigger;*capacity=grown;
  }
  (*list)[(*n)++]=(struct mesh_range){first,count};
  return 0;
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
  struct mesh_owned *owned=calloc(1,sizeof *owned);
  if(!owned){ munmap(memory,(size_t)info.st_size); close(file); return ENOMEM; }
  const char *key=getenv("MESH_SESSION");
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.key=key?strtoull(key,NULL,0):0,.session=MESH_ABSENT,.fd=file,.owned=owned};
  return 0;
}

int mesh_session_claim(struct mesh_ctx *c){
  if(c->session!=MESH_ABSENT)return 0;
  struct hdr *m=c->M;
  for(uint32_t s=0;s<MESH_SESSIONS;s++){
    struct mesh_session *session=&mesh_sessions(m)[s];
    uint64_t vacant=0;
    if(atomic_load_explicit(&session->served,memory_order_acquire) ||
       !atomic_compare_exchange_strong_explicit(&session->pid,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_relaxed))continue;
    atomic_store_explicit(&session->request,MESH_REQUEST_NONE,memory_order_relaxed);
    atomic_store_explicit(&session->depth,0,memory_order_relaxed);
    for(uint32_t q=0;q<m->links*m->qps;q++)for(int d=0;d<2;d++)atomic_store_explicit(mesh_order_length(m,s,q,d),0,memory_order_relaxed);
    for(uint32_t p=0;p<m->links;p++){
      memset(mesh_events(m,mesh_notice_queue(m,s,p)),0,m->notice_bytes);
      struct mesh_port_info *port=mesh_session_port(m,s,p);
      atomic_store_explicit(&port->phase,MESH_UNKNOWN,memory_order_relaxed);atomic_store_explicit(&port->code,0,memory_order_relaxed);
      atomic_store_explicit(&port->prepared,0,memory_order_release);
    }
    c->session=s;
    c->owned->sessions[c->owned->nsessions++]=s;
    return 0;
  }
  return EBUSY;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  struct hdr *m=c->M;
  struct mesh_owned *owned=c->owned;
  /* its sessions stopped (the bridge closes them: their receives, posted into its pages, gone) and given back, then
     its rings deregistered and its rows and pages freed */
  for(uint32_t i=0;i<owned->nsessions;i++){
    struct mesh_ctx session=*c;session.session=owned->sessions[i];
    (void)mesh_transfers_stop(&session);
    atomic_store_explicit(&mesh_sessions(m)[owned->sessions[i]].pid,0,memory_order_release);
  }
  for(uint32_t i=0;i<owned->nrings;i++){
    uint64_t at=owned->rings[i];
    for(uint32_t r=0;r<MESH_RINGS;r++)
      if(atomic_compare_exchange_strong_explicit(&mesh_rings(m)[r],&at,0,memory_order_acq_rel,memory_order_relaxed))break;
      else at=owned->rings[i];
  }
  for(uint32_t i=0;i<owned->nrows;i++)mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),owned->rows[i].first,owned->rows[i].count);
  for(uint32_t i=0;i<owned->npages;i++)mesh_bits_clear(mesh_arena_bits(m),owned->pages[i].first,owned->pages[i].count);
  free(owned->rows);free(owned->pages);free(owned->rings);free(owned);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* `count` bits from `first` set in one claim, or none: a bit another set first undoes this claim's (another process
   allocating at once). */
static int mesh_bits_claim(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    uint64_t mask=mesh_word_mask(first,count,w),old=atomic_fetch_or_explicit(&p[w],mask,memory_order_acq_rel);
    if(old&mask){
      atomic_fetch_and_explicit(&p[w],~(mask&~old),memory_order_acq_rel);
      for(uint32_t v=first/64;v<w;v++)atomic_fetch_and_explicit(&p[v],~mesh_word_mask(first,count,v),memory_order_acq_rel);
      return 0;
    }
  }
  return 1;
}

/* design/algorithm-sources.md#programkernel_call */
/* One first-fit scan over an explicit pair of bitmaps and an explicit range.  The caller names the
   bitmaps because the arena bitmap is indexed by arena page and the row planes by row: they are two
   index spaces and sharing one sizing is what forced rows >= pages. */
static uint32_t mesh_scan(uint32_t count,uint32_t align,uint32_t begin,uint32_t end,_Atomic uint64_t *own){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    uint32_t next=first;
    for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
      uint64_t occupied=atomic_load_explicit(&own[w],memory_order_acquire)&mesh_word_mask(first,count,w);
      if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
    }
    if(next==first){
      if(mesh_bits_claim(own,first,count))return first;
      next=first+1;
    }
    first=(next+align-1)/align*align;
  }
  return MESH_ABSENT;
}

/* design/algorithm-sources.md#programkernel_call */
/* The cursor resumes where the last allocation ended, so preparing N operands over P pages costs
   O(P) in total rather than O(N*P); a full scan and then a reclaim still back it, so the result is
   the same page the unresumed scan would have returned whenever nothing has been released. */
static uint32_t mesh_allocate(uint32_t count,uint32_t align,uint32_t begin,uint32_t end,_Atomic uint64_t *own,uint32_t *cursor){
  uint32_t resume=*cursor<begin||*cursor>end?begin:*cursor;
  uint32_t first=mesh_scan(count,align,resume,end,own);
  if(first==MESH_ABSENT && resume!=begin)first=mesh_scan(count,align,begin,end,own);
  if(first==MESH_ABSENT){ errno=ENOMEM; return MESH_ABSENT; }
  *cursor=first+count;
  return first;
}

uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(count,1,0,mesh_rows(c->M),mesh_plane(c->M,MESH_ROW_OWN),&c->row_cursor);
  if(first==MESH_ABSENT) return first;
  if(mesh_owned_add(&c->owned->rows,&c->owned->nrows,&c->owned->crows,first,count)){
    mesh_bits_clear(mesh_plane(c->M,MESH_ROW_OWN),first,count);errno=ENOMEM;return MESH_ABSENT;
  }
  for(uint32_t r=first;r<first+count;r++){
    atomic_store_explicit(&mesh_page(c->M)[r].device,0,memory_order_relaxed);
    atomic_store_explicit(&mesh_page(c->M)[r].mapping,MESH_ABSENT,memory_order_release);
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[r];
    buffer->pages=buffer->constant=0;
    struct mesh_publication *publication=mesh_publication_at(c->M,r);
    publication->sends=publication->device_input=publication->device_stride=0;
    atomic_store_explicit(&publication->argument,0,memory_order_release);
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
  uint32_t first=mesh_allocate(pages,align,range.first,range.first+range.count,bits,wire?&c->wire_cursor:&c->bulk_cursor);
  if(first==MESH_ABSENT)return first;
  if(mesh_owned_add(&c->owned->pages,&c->owned->npages,&c->owned->cpages,first,pages)){mesh_bits_clear(bits,first,pages);errno=ENOMEM;return MESH_ABSENT;}
  c->arena+=pages; if(wire)c->wire+=pages;
  return first;
}

/* exactly pages [first, first + pages) of the arena's `wire` range, or MESH_ABSENT (ENOMEM: one is taken) */
uint32_t mesh_arena_claim(struct mesh_ctx *c,uint32_t first,uint32_t pages,int wire){
  struct mesh_range range=mesh_arena_range(c->M,wire);
  if(!pages || first<range.first || first>range.first+range.count || pages>range.first+range.count-first){errno=EINVAL;return MESH_ABSENT;}
  _Atomic uint64_t *bits=mesh_arena_bits(c->M);
  if(!mesh_bits_claim(bits,first,pages)){errno=ENOMEM;return MESH_ABSENT;}
  if(mesh_owned_add(&c->owned->pages,&c->owned->npages,&c->owned->cpages,first,pages)){mesh_bits_clear(bits,first,pages);errno=ENOMEM;return MESH_ABSENT;}
  c->arena+=pages; if(wire)c->wire+=pages;
  return first;
}

void mesh_arena_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct mesh_owned *owned=c->owned;
  for(uint32_t i=0;i<owned->npages;i++)if(owned->pages[i].first==first && owned->pages[i].count==count){
    owned->pages[i]=owned->pages[--owned->npages];
    mesh_bits_clear(mesh_arena_bits(c->M),first,count);
    return;
  }
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

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  struct mesh_owned *owned=c->owned;
  for(uint32_t i=0;i<owned->nrows;i++)if(owned->rows[i].first==first && owned->rows[i].count==count){owned->rows[i]=owned->rows[--owned->nrows];break;}
  mesh_bits_clear(mesh_plane(c->M,MESH_ROW_OWN),first,count);
}

/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M18 */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *c,uint32_t row,uint32_t queue){
  struct hdr *m=c->M;
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t *count=&publication->sends;
  struct mesh_target *targets=publication->targets;
  uint32_t destination=mesh_notice_queue(m,c->session,queue);
  uint64_t first=m->notice_off+(uint64_t)destination*m->notice_bytes;
  for(uint32_t i=0;i<*count;i++)if(targets[i].stream>=first && targets[i].stream<first+m->notice_bytes)return &targets[i];
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

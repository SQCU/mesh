#include <signal.h>
#include <errno.h>
#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
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

/* The link table (mesh-dataflow.h): the latch sequence counter of Linux include/linux/seqlock.h
   (raw_write_seqcount_latch: two copies, the counter's parity naming the one readers take), in shared
   memory named after the region. */
struct mesh_link_contents *mesh_link_contents_new(uint32_t nodes){
  struct mesh_link_contents *c=calloc(1,mesh_link_contents_bytes(nodes));
  if(c)c->nodes=nodes;
  return c;
}
int mesh_link_table_open(const char *region,uint32_t nodes,uint32_t node,struct mesh_link_table **out){
  char name[64];
  snprintf(name,sizeof name,"%s.links",region?region:MESH_NAME);
  if(nodes && node>=nodes)return EINVAL;
  if(nodes)shm_unlink(name);
  int file=shm_open(name,nodes?O_CREAT|O_EXCL|O_RDWR:O_RDWR,MESH_MODE);
  if(file<0)return errno;
  struct stat info;
  if(nodes)fchmod(file,MESH_MODE);  /* as the region's: past the umask where the system allows it */
  size_t bytes=mesh_link_table_bytes(nodes);
  int error=nodes && ftruncate(file,(off_t)bytes)?errno:fstat(file,&info)?errno:0;
  if(!error && !nodes){
    /* the bridge's table: its size from its header (the system rounds the object up to a page) */
    struct mesh_link_table *head=(size_t)info.st_size<sizeof *head?MAP_FAILED:mmap(NULL,sizeof *head,PROT_READ,MAP_SHARED,file,0);
    if(head==MAP_FAILED)error=EINVAL;
    else{
      if(head->magic!=MESH_LINK_MAGIC || (size_t)info.st_size<(bytes=mesh_link_table_bytes(head->nodes)))error=EINVAL;
      munmap(head,sizeof *head);
    }
  }
  struct mesh_link_table *t=error?MAP_FAILED:mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  if(!error && t==MAP_FAILED)error=errno;
  close(file);
  if(error)return error;
  if(nodes){
    t->nodes=nodes;t->node=node;
    mesh_link_copy(t,0)->nodes=mesh_link_copy(t,1)->nodes=nodes;
    atomic_thread_fence(memory_order_release);
    t->magic=MESH_LINK_MAGIC;
  }
  *out=t;
  return 0;
}
void mesh_link_table_close(struct mesh_link_table *t){if(t)munmap(t,mesh_link_table_bytes(t->nodes));}
uint64_t mesh_link_table_read(const struct mesh_link_table *t,struct mesh_link_contents *out){
  for(;;){
    uint64_t epoch=atomic_load_explicit(&t->epoch,memory_order_acquire);
    memcpy(out,mesh_link_copy(t,epoch),mesh_link_contents_bytes(t->nodes));
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&t->epoch,memory_order_relaxed)==epoch)return epoch;
  }
}
/* The writer word holds its holder's pid: a holder whose process is gone (it can be a crashed bridge, the table kept)
   is taken over, as an edit is published only by the epoch's bump, so a dead holder's half edit is never seen. */
static void mesh_link_table_lock(struct mesh_link_table *t){
  const uint32_t me=(uint32_t)getpid();
  for(;;){
    uint32_t holder=0;
    if(atomic_compare_exchange_weak_explicit(&t->writer,&holder,me,memory_order_acquire,memory_order_relaxed))return;
    if(holder && holder!=me && kill((pid_t)holder,0) && errno==ESRCH &&
       atomic_compare_exchange_strong_explicit(&t->writer,&holder,me,memory_order_acquire,memory_order_relaxed))return;
    usleep(10);
  }
}
static void mesh_link_table_unlock(struct mesh_link_table *t){atomic_store_explicit(&t->writer,0,memory_order_release);}
/* The write, the writer held. */
static int mesh_link_table_edit(struct mesh_link_table *t,void (*edit)(struct mesh_link_contents *,const void *),const void *argument){
  uint64_t epoch=atomic_load_explicit(&t->epoch,memory_order_relaxed);
  struct mesh_link_contents *now=mesh_link_copy(t,epoch),*next=mesh_link_copy(t,epoch+1);
  const size_t bytes=mesh_link_contents_bytes(t->nodes);
  memcpy(next,now,bytes);
  edit(next,argument);
  int changed=memcmp(next,now,bytes)!=0;
  if(changed)atomic_store_explicit(&t->epoch,epoch+1,memory_order_release);
  return changed;
}
int mesh_link_table_write(struct mesh_link_table *t,void (*edit)(struct mesh_link_contents *,const void *),const void *argument){
  mesh_link_table_lock(t);
  int changed=mesh_link_table_edit(t,edit,argument);
  mesh_link_table_unlock(t);
  return changed;
}
/* Node `node`'s links' up set to the bits of `up`. */
struct mesh_link_row { uint32_t node; const uint64_t *up; };
static void mesh_link_row_set(struct mesh_link_contents *c,const void *argument){
  const struct mesh_link_row *r=argument;
  for(uint32_t b=0;b<c->nodes;b++)mesh_link_at(c,r->node,b)->up=b!=r->node && (r->up[b/64]>>(b%64)&1);
}
uint64_t mesh_link_table_row(struct mesh_link_table *t,uint32_t v,uint64_t *up){
  if(v>=t->nodes)return 0;
  memset(up,0,(t->nodes+63)/64*sizeof *up);
  mesh_link_table_lock(t);
  const struct mesh_link_contents *c=mesh_link_copy(t,atomic_load_explicit(&t->epoch,memory_order_relaxed));
  for(uint32_t b=0;b<t->nodes;b++)if(mesh_link_at(c,v,b)->up)up[b/64]|=UINT64_C(1)<<(b%64);
  uint64_t sequence=atomic_load_explicit(&mesh_link_reported(t)[v],memory_order_relaxed);
  mesh_link_table_unlock(t);
  return sequence;
}
/* One link of this node set up or down, inside one edit (each session observes its own link). */
struct mesh_link_one { uint32_t node,peer,up; };
static void mesh_link_one_set(struct mesh_link_contents *c,const void *argument){
  const struct mesh_link_one *o=argument;
  mesh_link_at(c,o->node,o->peer)->up=o->up!=0;
}
int mesh_link_table_observe(struct mesh_link_table *t,uint32_t peer,uint32_t up){
  if(peer>=t->nodes || peer==t->node)return -EINVAL;
  mesh_link_table_lock(t);
  int changed=mesh_link_table_edit(t,mesh_link_one_set,&(struct mesh_link_one){t->node,peer,up});
  if(changed)atomic_fetch_add_explicit(&mesh_link_reported(t)[t->node],1,memory_order_release);
  mesh_link_table_unlock(t);
  return changed;
}
void mesh_link_table_forget(struct mesh_link_table *t,uint32_t node){
  if(node>=t->nodes || node==t->node)return;
  mesh_link_table_lock(t);
  atomic_store_explicit(&mesh_link_reported(t)[node],0,memory_order_release);
  mesh_link_table_unlock(t);
}
int mesh_link_table_report(struct mesh_link_table *t,uint32_t origin,uint64_t sequence,const uint64_t *up){
  if(origin>=t->nodes || origin==t->node)return 0;
  mesh_link_table_lock(t);
  int newer=sequence>atomic_load_explicit(&mesh_link_reported(t)[origin],memory_order_relaxed);
  if(newer){
    mesh_link_table_edit(t,mesh_link_row_set,&(struct mesh_link_row){origin,up});
    atomic_store_explicit(&mesh_link_reported(t)[origin],sequence,memory_order_release);
  }
  mesh_link_table_unlock(t);
  return newer;
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
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
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
/* One first-fit scan over an explicit pair of bitmaps and an explicit range.  The caller names the
   bitmaps because the arena bitmap is indexed by arena page and the row planes by row: they are two
   index spaces and sharing one sizing is what forced rows >= pages. */
/* The claim is atomic word by word: several processes allocate from one arena (the prepared program's
   client and the communicator clients, mesh-net.c), so a bit another claimant set since the scan read
   it undoes this claim and the scan goes on past it. */
static uint32_t mesh_scan(uint32_t count,uint32_t align,uint32_t begin,uint32_t end,
                          _Atomic uint64_t *own,_Atomic uint64_t *hot){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    uint32_t next=first;
    for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
      uint64_t occupied=(atomic_load_explicit(&own[w],memory_order_acquire)|atomic_load_explicit(&hot[w],memory_order_acquire))&mesh_word_mask(first,count,w);
      if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
    }
    if(next==first){
      uint32_t w=first/64,last=(first+count-1)/64;
      for(;w<=last;w++){
        uint64_t mask=mesh_word_mask(first,count,w),before=atomic_fetch_or_explicit(&own[w],mask,memory_order_acq_rel);
        if(before&mask){atomic_fetch_and_explicit(&own[w],~(mask&~before),memory_order_acq_rel);break;}
      }
      if(w>last) return first;
      for(uint32_t u=first/64;u<w;u++)atomic_fetch_and_explicit(&own[u],~mesh_word_mask(first,count,u),memory_order_acq_rel);
      next=first+1;
    }
    first=(next+align-1)/align*align;
  }
  return MESH_ABSENT;
}

/* Arena pages of the registered window (wire) or the rest, for a process that is not the region's
   program client (mesh_net_mem_alloc): the same atomic claim, after a reclaim when none is free. */
uint32_t mesh_arena_claim(struct hdr *m,uint32_t pages,uint32_t align,int wire){
  struct mesh_range range=mesh_arena_range(m,wire);
  _Atomic uint64_t *bits=mesh_arena_bits(m);
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_scan(pages,align,range.first,range.first+range.count,bits,bits);
  if(first==MESH_ABSENT){ mesh_reclaim(m); first=mesh_scan(pages,align,range.first,range.first+range.count,bits,bits); }
  if(first==MESH_ABSENT) errno=ENOMEM;
  return first;
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

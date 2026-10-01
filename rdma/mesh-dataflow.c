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
     m->link_off>=sizeof *m && m->link_off<=m->length && m->links<=INT_MAX && m->links<=(m->length-m->link_off)/sizeof(struct mesh_link_info) &&
     m->net_off>=sizeof *m && m->net_off<=m->length && m->links<=(m->length-m->net_off)/sizeof(struct mesh_net_link)){
    *node=m->node;result=(int)m->links;
    for(uint32_t i=0;i<m->links && i<capacity;i++){
      struct mesh_link_info *link=&mesh_links(m)[i];
      out[i]=(struct mesh_link_view){.peer=link->peer,.phase=atomic_load_explicit(&mesh_net_links(m)[i].phase,memory_order_acquire)};
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

/* design/algorithm-sources.md#programkernel_call */
/* One first-fit scan of the arena bitmap over a range.  The claim is atomic word by word: several processes
   allocate from one arena (the communicator clients, mesh-net.c), so a bit another claimant set since the scan
   read it undoes this claim and the scan goes on past it. */
static uint32_t mesh_scan(uint32_t count,uint32_t align,uint32_t begin,uint32_t end,_Atomic uint64_t *bits){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    uint32_t next=first;
    for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
      uint64_t occupied=atomic_load_explicit(&bits[w],memory_order_acquire)&mesh_word_mask(first,count,w);
      if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
    }
    if(next==first){
      uint32_t w=first/64,last=(first+count-1)/64;
      for(;w<=last;w++){
        uint64_t mask=mesh_word_mask(first,count,w),before=atomic_fetch_or_explicit(&bits[w],mask,memory_order_acq_rel);
        if(before&mask){atomic_fetch_and_explicit(&bits[w],~(mask&~before),memory_order_acq_rel);break;}
      }
      if(w>last) return first;
      for(uint32_t u=first/64;u<w;u++)atomic_fetch_and_explicit(&bits[u],~mesh_word_mask(first,count,u),memory_order_acq_rel);
      next=first+1;
    }
    first=(next+align-1)/align*align;
  }
  return MESH_ABSENT;
}

uint32_t mesh_arena_claim(struct hdr *m,uint32_t pages,uint32_t align,int wire){
  struct mesh_range range=mesh_arena_range(m,wire);
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_scan(pages,align,range.first,range.first+range.count,mesh_arena_bits(m));
  if(first==MESH_ABSENT) errno=ENOMEM;
  return first;
}

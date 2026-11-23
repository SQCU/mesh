#include "mesh.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* The node's region mapped, `writable` (a client's) or read-only (a reader's, which attaches no client and changes
   nothing of the bridge's), its descriptor left open in *fd where one is asked for, whether or not a bridge runs on it
   now (it outlives a restart of its bridge): 0, EINVAL where it is no region of this layout, or one expired, else the
   errno */
static int mesh_open(int writable,struct hdr **out,size_t *length,int *fd){
  int file=shm_open(MESH_NAME,writable?O_RDWR:O_RDONLY,0);
  if(file<0)return errno;
  struct stat info;
  if(fstat(file,&info)){int error=errno;close(file);return error;}
  if(info.st_size<(off_t)sizeof(struct hdr)){close(file);return EINVAL;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ|(writable?PROT_WRITE:0),MAP_SHARED,file,0);
  int error=errno;
  if(m==MAP_FAILED){close(file);return error;}
  error=m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT || m->length>(uint64_t)info.st_size?EINVAL:0;
  if(error || !fd)close(file);
  if(error){munmap(m,(size_t)info.st_size);return error;}
  *out=m;*length=(size_t)info.st_size;
  if(fd)*fd=file;
  return 0;
}

/* ---- The fabric view (mesh-view.h) ---------------------------------------------------------------------------------- */
int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  struct mesh_reach all[MESH_NODES];
  const uint32_t count=mesh_nodes_reached(m,all,MESH_NODES);
  for(uint32_t k=0;out && k<count && k<capacity;k++)out[k]=all[k];
  if(node)*node=m->node;
  munmap(m,length);return (int)count;
}

/* The bridge stores idle_since before busy (mesh-records.c busy_tell), and idle_since is never 0 while its lanes have
   nothing (the region's boot from its start), so a read that meets it as it becomes busy (busy 0 and idle_since 0: the
   old count, the new time) is a busy one and is told as one */
int mesh_busy(uint64_t *busy,uint64_t *idle_since){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *busy=atomic_load_explicit(&m->busy,memory_order_acquire);*idle_since=atomic_load_explicit(&m->idle_since,memory_order_acquire);
  if(!*busy && !*idle_since)*busy=1;
  munmap(m,length);return 0;
}

int mesh_region(uint64_t *bytes,uint64_t *unleased){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *bytes=(uint64_t)mesh_arena_pages(m)*m->pgsz;*unleased=(uint64_t)mesh_arena_free(m)*m->pgsz;
  munmap(m,length);return 0;
}

/* The map's records as its published copy has them (mesh.h mesh_map_published), never waited for: a copy in *out the
   caller frees and its count, or -ENOMEM */
static int mesh_records_read(struct hdr *m,struct mesh_record **out){
  const struct mesh_map_copy *map=mesh_map_published(m);
  const uint32_t count=mesh_map_nodes(map);
  struct mesh_record *copy=malloc((count?count:1)*sizeof *copy);
  if(!copy)return -ENOMEM;
  memcpy(copy,map->record,count*sizeof *copy);
  *out=copy;return (int)count;
}

/* The one join of two nodes' records: the cables each one's bridge reaches the other over (its record names the other
   once a cable, mesh-link.c record_reach), the fewer of the two ends' counts, *slowest the least bandwidth either
   reports of them (0: neither reports one); 0 where either names the other none */
static uint32_t mesh_paired(struct mesh_record *record,uint32_t node,uint64_t *slowest){
  uint32_t cables=0;
  for(uint32_t i=0;i<record->reached && i<MESH_STREAMS;i++)
    if(record->reach[i].node==node){
      const uint64_t bandwidth=record->reach[i].bandwidth;
      if(bandwidth && (!*slowest || bandwidth<*slowest))*slowest=bandwidth;
      cables++;
    }
  return cables;
}
static uint32_t mesh_join(struct mesh_record *a,struct mesh_record *b,uint64_t *bandwidth){
  uint64_t slowest=0;
  const uint32_t ends[2]={mesh_paired(a,b->node,&slowest),mesh_paired(b,a->node,&slowest)};
  if(!ends[0] || !ends[1])return 0;
  *bandwidth=slowest;
  return ends[0]<ends[1]?ends[0]:ends[1];
}
int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  uint64_t bandwidth=0;
  const uint32_t joined=mesh_join(a,b,&bandwidth);
  if(joined)*beta=bandwidth?8e9/(double)bandwidth:0;
  return (int)joined;
}

static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
/* The records' map (mesh-view.h mesh_map) in the caller's storage: the records sorted by node, a link a cable where
   two join (mesh_join), pairs in ascending order */
static int mesh_records_map(struct mesh_record *r,uint32_t n,uint32_t *node,struct mesh_link_map *map){
  qsort(r,n,sizeof *r,mesh_node_order);
  uint64_t bandwidth=0;
  uint32_t links=0,pairs=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++){
    const uint32_t cables=mesh_join(r+a,r+b,&bandwidth);
    links+=cables;pairs+=cables>0;
  }
  if(map->nodes<n || map->links<links){map->nodes=n;map->links=links;return -ENOSPC;}
  uint32_t l=0;
  for(uint32_t a=0;a<n;a++){
    if(node)node[a]=r[a].node;
    for(uint32_t b=a+1;b<n;b++)
      for(uint32_t k=mesh_join(r+a,r+b,&bandwidth);k;k--){
        map->link[l][0]=a;map->link[l][1]=b;
        if(map->bandwidth)map->bandwidth[l]=bandwidth;
        l++;
      }
  }
  map->nodes=n;map->links=links;
  map->kind=(uint64_t)pairs*2==(uint64_t)n*(n?n-1:0)?MESH_LINKS_MESH:MESH_LINKS_GRAPH;
  return (int)n;
}
/* a record carries no host name (mesh.h struct mesh_record): a name a person reads is the name table's by node number
   (rdma/peers.py) */
int mesh_map(uint32_t *node,struct mesh_link_map *map){
  if(!map || (map->links && !map->link))return -EINVAL;
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  const int count=mesh_records_read(m,&r);
  munmap(m,length);
  if(count<0)return count;
  const int result=mesh_records_map(r,(uint32_t)count,node,map);
  free(r);
  return result;
}

/* The records with their health, as the map's published copy has them (a health sample is a publish of its own:
   mesh-records.c health_put) */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity){
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  const int count=mesh_records_read(m,&r);
  munmap(m,length);
  if(count<0)return count;
  for(int i=0;i<count && (uint32_t)i<capacity;i++){
    out[i].node=r[i].node;
    out[i].region=r[i].region;out[i].memory=r[i].memory;
    memcpy(out[i].model,r[i].model,MESH_MODEL_BYTES);out[i].model[MESH_MODEL_BYTES-1]=0;
    out[i].health=r[i].health;
  }
  free(r);
  return count;
}

/* The map's serial waited on, with no clock, until it differs from *serial: each wake (the bridge's publish that
   changes the mesh, its successor's first, or the region's expiry, each of which moves the serial: mesh.h
   mesh_map_moved) reads the serial and the region's layout again; ESRCH where the region expired (the heartbeat's
   release, or a bridge that could not adopt it: no bridge serves it again) */
int mesh_map_wait(uint64_t *serial){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return error;
  uint64_t seen;
  while((seen=atomic_load_explicit(&m->map,memory_order_acquire))==*serial && __atomic_load_n(&m->version,__ATOMIC_ACQUIRE)==MESH_LAYOUT)
    os_sync_wait_on_address(&m->map,seen,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  if(__atomic_load_n(&m->version,__ATOMIC_ACQUIRE)!=MESH_LAYOUT)error=ESRCH;
  else *serial=seen;
  munmap(m,length);
  return error;
}

/* ---- Client slots ------------------------------------------------------------------------------------------------- */
int mesh_slot_take(struct mesh_ctx *c,struct hdr *m){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<25)-1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)m+m->client_off)+s;
    uint64_t vacant=0,owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid();
    if(!atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
    slot->start=mesh_started(getpid());
    /* the slot's write ring, whose one producer this process is now: its put moved past every position the slot's
       last client wrote, to its take where the take is past it, then past the cell at put where that cell holds a
       write (its sequence put + 1: a client that exited between its cell's store and put's), so this process never
       writes a cell a taker may be copying; a write so left on the ring flagged for the bridge's port children (mesh.h
       mesh_ring_flag), which carry it */
    struct mesh_ring *ring=mesh_ring_at(m,s);
    uint64_t put=atomic_load_explicit(&ring->put,memory_order_relaxed);
    const uint64_t take=atomic_load_explicit(&ring->take,memory_order_acquire);
    if(take>put)put=take;
    if(atomic_load_explicit(&ring->cell[put%MESH_RING].sequence,memory_order_acquire)==put+1)put++;
    atomic_store(&ring->put,put);
    if(put>take)mesh_ring_flag(m,s);
    *c=(struct mesh_ctx){.M=m,.len=c->len,.client=owner,.fd=c->fd};
    return 0;
  }
  return EBUSY;
}

int mesh_attach(struct mesh_ctx *c){
  if(c->M)return 0;
  struct hdr *m;size_t length;int file;
  int error=mesh_open(1,&m,&length,&file);
  if(error)return error;
  *c=(struct mesh_ctx){.len=length,.fd=file};
  error=mesh_slot_take(c,m);
  if(error){munmap(m,length);close(file);*c=(struct mesh_ctx){0};return error;}
  /* the expiry (a store, seq_cst) read again after the slot's compare-exchange (seq_cst): a slot the heartbeat's read
     of the region's processes missed reads it expired here, and is let go */
  if(__atomic_load_n(&m->version,__ATOMIC_SEQ_CST)!=MESH_LAYOUT){mesh_detach(c);return EINVAL;}
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M)return 0;
  mesh_leave(c->M,c->client);
  int error=(c->fd>=0 && munmap(c->M,c->len))?errno:0;
  if(c->fd>=0 && close(c->fd) && !error)error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* ---- The arena's leases ------------------------------------------------------------------------------------------- */
/* the bits of [first, first + count) in bitmap word `word` */
static uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a)return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
/* [first, first + count) taken word by word, each by a compare-exchange that expects its bits clear; where another
   took one first, the words taken are given back: 1 where taken, else 0 */
static int arena_mark(_Atomic uint64_t *bits,uint32_t first,uint32_t count){
  const uint32_t last=(first+count-1)/64;
  uint32_t w=first/64;
  for(;w<=last;w++){
    const uint64_t mask=mesh_word_mask(first,count,w);
    uint64_t word=atomic_load_explicit(&bits[w],memory_order_acquire);
    while(!(word&mask) && !atomic_compare_exchange_weak_explicit(&bits[w],&word,word|mask,memory_order_acq_rel,memory_order_acquire));
    if(word&mask)break;
  }
  if(w>last)return 1;
  for(uint32_t v=first/64;v<w;v++)atomic_fetch_and_explicit(&bits[v],~mesh_word_mask(first,count,v),memory_order_acq_rel);
  return 0;
}
/* the first `count` consecutive clear pages in [begin, end), its first a multiple of `align` and all within one extent
   of `extent` pages, taken (arena_mark); MESH_ABSENT where none is */
static uint32_t mesh_scan(_Atomic uint64_t *bits,uint32_t count,uint32_t align,uint32_t extent,uint32_t begin,uint32_t end){
  for(uint64_t first=((uint64_t)begin+align-1)/align*align;first<end && count<=end-first;){
    if(first/extent!=(first+count-1)/extent){first=((first/extent+1)*extent+align-1)/align*align;continue;}
    uint64_t next=first;
    for(uint32_t w=(uint32_t)first/64;w<=(uint32_t)(first+count-1)/64;w++){
      const uint64_t occupied=atomic_load_explicit(&bits[w],memory_order_acquire)&mesh_word_mask((uint32_t)first,count,w);
      if(occupied)next=(uint64_t)w*64+64-(uint64_t)__builtin_clzll(occupied);
    }
    if(next==first && arena_mark(bits,(uint32_t)first,count))return (uint32_t)first;
    if(next>first)first=(next+align-1)/align*align;
  }
  return MESH_ABSENT;
}

/* The scan resumes at the context's rover, where its last lease ended, then from page 0 */
uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  struct hdr *m=c?c->M:NULL;
  if(!m || !pages || !align){errno=EINVAL;return MESH_ABSENT;}
  const uint32_t arena=mesh_arena_pages(m),extent=(uint32_t)(MESH_DMA_EXTENT/m->pgsz),resume=c->arena_rover<arena?c->arena_rover:0;
  _Atomic uint64_t *bits=mesh_arena_bits(m);
  uint32_t first=mesh_scan(bits,pages,align,extent,resume,arena);
  if(first==MESH_ABSENT && resume)first=mesh_scan(bits,pages,align,extent,0,arena);
  if(first==MESH_ABSENT){errno=ENOMEM;return MESH_ABSENT;}
  c->arena_rover=first+pages;
  return first;
}

uint32_t mesh_arena_free(struct hdr *m){
  const uint32_t pages=mesh_arena_pages(m),words=(uint32_t)(((uint64_t)pages+63)/64);
  uint32_t used=0;
  for(uint32_t w=0;w<words;w++)
    used+=(uint32_t)__builtin_popcountll(atomic_load_explicit(&mesh_arena_bits(m)[w],memory_order_relaxed)&mesh_word_mask(0,pages,w));
  return pages-used;
}

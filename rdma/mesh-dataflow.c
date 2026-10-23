#include "mesh-dataflow.h"
#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
/* design/transport.md */
static void mesh_reclaim(struct hdr *);

/* design/transport.md: the node's region mapped, `writable` (a client's) or read-only (a reader's, which attaches no
   client and changes nothing of the bridge's), its descriptor kept in *fd where one is asked for, whether or not a bridge
   runs on it now (it outlives a restart of its bridge, design/transport.md section 2): 0, EINVAL where it is no region
   of this layout, or one expired,
   else the errno */
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

/* design/transport.md section 14: this node's bridge's node and the nodes it reaches (mesh-plan.h mesh_reached) */
int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  struct mesh_reach all[MESH_STREAMS];
  const uint32_t count=mesh_nodes_reached(m,all,MESH_STREAMS);
  for(uint32_t k=0;out && k<count && k<capacity;k++)out[k]=all[k];
  if(node)*node=m->node;
  munmap(m,length);return (int)count;
}

/* design/transport.md: the flows and awaiting programs the bridge holds and the wall clock (ns) at which it
   last held none (0 while it holds any).  The bridge stores idle_since before busy (mesh-program.c busy_tell), and
   idle_since is never 0 while it holds none (the region's boot from its start), so a read that meets it as it becomes
   busy (busy 0 and idle_since 0: the old count, the new time) is a busy one and is told as one */
int mesh_busy(uint64_t *busy,uint64_t *idle_since){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *busy=atomic_load_explicit(&m->busy,memory_order_acquire);*idle_since=atomic_load_explicit(&m->idle_since,memory_order_acquire);
  if(!*busy && !*idle_since)*busy=1;
  munmap(m,length);return 0;
}

/* design/transport.md */
int mesh_region(uint64_t *bytes,uint64_t *unleased){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *bytes=(uint64_t)mesh_arena_pages(m)*m->pgsz;*unleased=(uint64_t)mesh_arena_free(m)*m->pgsz;
  munmap(m,length);return 0;
}

/* design/transport.md: the map's records as one serial left them, the seqlock's reader (mesh_map_whole): a
   copy in *out the caller frees and its count, or -errno (ESRCH: the bridge went while its write was open) */
static int mesh_records_read(struct hdr *m,size_t length,struct mesh_record **out){
  for(;;){
    uint64_t before=atomic_load_explicit(&m->map,memory_order_acquire);
    if(before&1){
      if(os_sync_wait_on_address_with_timeout(&m->map,before,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
           OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS)<0 && errno==ETIMEDOUT && mesh_gone((pid_t)atomic_load(&m->bridge_pid)))return -ESRCH;
      continue;
    }
    uint64_t count=atomic_load_explicit(&m->map_nodes,memory_order_relaxed),at=atomic_load_explicit(&m->map_off,memory_order_relaxed);
    int fits=at<=length && count<=(length-at)/sizeof(struct mesh_record);
    struct mesh_record *copy=fits?malloc((count?count:1)*sizeof *copy):NULL;
    if(copy)memcpy(copy,(char *)m+at,count*sizeof *copy);
    if(mesh_map_whole(m,before)){
      if(!copy)return fits?-ENOMEM:-EINVAL;
      *out=copy;return (int)count;
    }
    free(copy);
  }
}

/* design/transport.md section 14: the one join of two nodes' records: 1 where each one's bridge reaches the other
   (its record names the other among the nodes it reaches), *slowest the lesser bandwidth either reports (0: neither
   reports one); else 0 */
static uint32_t mesh_paired(struct mesh_record *record,uint32_t node,uint64_t *slowest){
  for(uint32_t i=0;i<record->reached && i<MESH_STREAMS;i++)
    if(record->reach[i].node==node){
      const uint64_t bandwidth=record->reach[i].bandwidth;
      if(bandwidth && (!*slowest || bandwidth<*slowest))*slowest=bandwidth;
      return 1;
    }
  return 0;
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

/* The records' whole map, its nodes ascending (the records sorted by node), one link where two join (mesh_join), at
   the join's bandwidth, and its routing table (mesh-collective.c mesh_routes, each node marked by its bridge's number,
   so every member that holds the same records makes the same routes) */
struct mesh_whole { uint32_t n,(*link)[2],*mark,*next; uint64_t *bandwidth; struct mesh_link_map map; };
static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
static void mesh_whole_free(struct mesh_whole *w){free(w->link);free(w->bandwidth);free(w->mark);free(w->next);}
static int mesh_whole_make(struct mesh_record *r,uint32_t n,struct mesh_whole *w){
  qsort(r,n,sizeof *r,mesh_node_order);
  size_t joins=1;
  uint64_t bandwidth;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++)joins+=(size_t)mesh_join(r+a,r+b,&bandwidth);
  *w=(struct mesh_whole){.n=n,.link=malloc(joins*sizeof *w->link),.bandwidth=malloc(joins*sizeof *w->bandwidth),
    .mark=malloc(((size_t)n+1)*sizeof *w->mark),.next=malloc(((size_t)n*n+1)*sizeof *w->next)};
  if(!w->link || !w->bandwidth || !w->mark || !w->next)return ENOMEM;
  uint32_t links=0;
  for(uint32_t a=0;a<n;a++){
    w->mark[a]=r[a].node;
    for(uint32_t b=a+1;b<n;b++)
      if(mesh_join(r+a,r+b,&bandwidth)){
        w->link[links][0]=a;w->link[links][1]=b;w->bandwidth[links]=bandwidth;links++;
      }
  }
  w->map=(struct mesh_link_map){.kind=MESH_LINKS_GRAPH,.nodes=n,.links=links,.link=w->link,.bandwidth=w->bandwidth,.relays=0,.next=NULL,.mark=w->mark};
  const int error=n?mesh_routes(&w->map,w->next):0;
  if(!error)w->map.next=w->next;
  return error;
}

/* design/algorithm-sources.md#meshobserve: mesh-plan.h mesh_map over a copy of the records: the whole map's
   mesh_map_members over the members' records (count: a member no record holds) */
static int mesh_records_map(struct mesh_record *r,uint32_t count,const uint32_t *members,uint32_t n,uint32_t *node,
                            struct mesh_link_map *map){
  struct mesh_whole w;
  int error=mesh_whole_make(r,count,&w);
  uint32_t *index=malloc(((size_t)n+1)*sizeof *index),*whole=malloc(((size_t)map->nodes+1)*sizeof *whole);
  if(!error && (!index || !whole))error=ENOMEM;
  for(uint32_t i=0;!error && i<n;i++){
    index[i]=count;
    for(uint32_t k=0;k<count;k++)if(r[k].node==members[i])index[i]=k;
  }
  int made=error?-error:mesh_map_members(&w.map,n?index:NULL,n,whole,map);
  for(uint32_t i=0;node && made>=0 && i<(uint32_t)made+map->relays;i++)node[i]=i<(uint32_t)made && n?members[i]:r[whole[i]].node;
  free(index);free(whole);mesh_whole_free(&w);
  return made;
}

/* a record carries no host name (mesh.h struct mesh_record; AGENTS.md "A node is its hardware; names are virtual"): a
   name a person reads is the name table's by node number (rdma/peers.py) */
int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,struct mesh_link_map *map){
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  int count=mesh_records_read(m,length,&r);
  munmap(m,length);
  if(count<0)return count;
  int result=mesh_records_map(r,(uint32_t)count,members,n,node,map);
  free(r);
  return result;
}

/* design/transport.md */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity){
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  /* the records under both serials: the map's (mesh_records_read) and the health's, which a health-only change moves
     (mesh-records.c health_put); a read that a health write overlapped is read again, and an odd serial (a write open)
     waited on as mesh_records_read waits on the map's, ESRCH where the bridge went while it was open */
  int count;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->health,memory_order_acquire);
    if(before&1){
      if(os_sync_wait_on_address_with_timeout(&m->health,before,sizeof m->health,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
           OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS)<0 && errno==ETIMEDOUT && mesh_gone((pid_t)atomic_load(&m->bridge_pid))){
        count=-ESRCH;break;
      }
      continue;
    }
    count=mesh_records_read(m,length,&r);
    if(count<0 || atomic_load_explicit(&m->health,memory_order_acquire)==before)break;
    free(r);r=NULL;
  }
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

/* design/transport.md C5 (8e716999's mesh.wait(serial), on this bound): the map's serial waited on until
   it differs from *serial, the bridge's exit (ESRCH: its pid in the region 0 or another, or its process gone, read at
   every wake and every MESH_PAUSE_NS re-look, as mesh_records_read reads it) or bound_ns (UINT64_MAX: none) */
int mesh_map_wait(uint64_t *serial,uint64_t bound_ns){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return error;
  const uint64_t bridge=atomic_load_explicit(&m->bridge_pid,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const uint64_t until=bound_ns>UINT64_MAX-now?UINT64_MAX:now+bound_ns;
  uint64_t seen=atomic_load_explicit(&m->map,memory_order_acquire);
  for(uint64_t at=now;!error && seen==*serial && at<until;at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW)){
    os_sync_wait_on_address_with_timeout(&m->map,seen,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,
                                         until-at<MESH_PAUSE_NS?until-at:MESH_PAUSE_NS);
    seen=atomic_load_explicit(&m->map,memory_order_acquire);
    if(seen==*serial && (atomic_load_explicit(&m->bridge_pid,memory_order_acquire)!=bridge || mesh_gone((pid_t)bridge)))error=ESRCH;
  }
  if(!error)error=seen!=*serial?0:ETIMEDOUT;
  if(!error)*serial=seen;
  munmap(m,length);
  return error;
}

/* design/algorithm-sources.md#programtensor */
/* design/transport.md: a client's slot given up (by its detach, or by the bridge where its process is
   gone): its rows closed for the bridge to release (the streams it held are let go on the bridge's turn:
   mesh-link.c streams_let_go) */
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_client *slot=mesh_client_at(m,client);
  if(!client || (client&MESH_RETIRING) || !atomic_compare_exchange_strong(&slot->owner,&client,client|MESH_RETIRING))return;
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==client){
      if(buffer->pages)atomic_store_explicit(&buffer->closed,1,memory_order_release);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  atomic_store_explicit(&slot->owner,0,memory_order_release);
  /* design/transport.md */
  mesh_notify(&m->control);
}

int mesh_slot_take(struct mesh_ctx *c,struct hdr *m){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<25)-1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)m+m->client_off)+s;
    uint64_t vacant=0,owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid();
    if(!atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
    slot->start=mesh_started(getpid());
    *c=(struct mesh_ctx){.M=m,.len=c->len,.client=owner,.fd=c->fd};
    /* design/transport.md section 2: the bridge woken (as mesh_retire wakes it), so it watches this process's exit
       from its next turn, at once: a holder taken between two of its turns, which come MESH_PAUSE_NS apart where
       nothing wakes it, and ended before the second kept its streams to that turn (a next program's EBUSY, 10-23) */
    mesh_notify(&m->control);
    return 0;
  }
  return EBUSY;
}

int mesh_attach(struct mesh_ctx *c){
  if(c->M) return 0;
  struct hdr *m;size_t length;int file;
  int error=mesh_open(1,&m,&length,&file);
  if(error)return error;
  *c=(struct mesh_ctx){.len=length,.fd=file};
  error=mesh_slot_take(c,m);
  if(error){munmap(m,length);close(file);*c=(struct mesh_ctx){0};return error;}
  /* design/transport.md section 2: the expiry (a store, seq_cst) read again after the slot's compare-exchange
     (seq_cst): a slot the heartbeat's read of the holders missed reads it expired here, and is let go */
  if(__atomic_load_n(&m->version,__ATOMIC_SEQ_CST)!=MESH_LAYOUT){mesh_detach(c);return EINVAL;}
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire(c->M,c->client);
  int error=(c->fd>=0 && munmap(c->M,c->len))?errno:0;
  if(c->fd>=0 && close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programkernel_call */
/* One first-fit scan over an explicit pair of bitmaps and an explicit range.  The caller names the
   bitmaps because the arena bitmap is indexed by arena page and the row planes by row: they are two
   index spaces and sharing one sizing is what forced rows >= pages.  A range found clear is claimed
   word by word with a compare-exchange that expects its bits clear; where another client claimed one
   first, the words taken are given back and the scan goes on from there, so two clients never hold
   one range.  A claimed range still `hot` (released, not yet reclaimed) is given back likewise. */
static int mesh_claim(_Atomic uint64_t *own,_Atomic uint64_t *hot,uint32_t first,uint32_t count){
  uint32_t last=(first+count-1)/64,w=first/64;
  for(;w<=last;w++){
    uint64_t mask=mesh_word_mask(first,count,w),word=atomic_load_explicit(&own[w],memory_order_acquire);
    while(!(word&mask) && !atomic_compare_exchange_weak_explicit(&own[w],&word,word|mask,memory_order_acq_rel,memory_order_acquire));
    if(word&mask)break;
  }
  int held=w>last;
  for(uint32_t v=first/64;held && hot!=own && v<=last;v++)held=!(atomic_load_explicit(&hot[v],memory_order_acquire)&mesh_word_mask(first,count,v));
  if(held)return 1;
  for(uint32_t v=first/64;v<w;v++)atomic_fetch_and_explicit(&own[v],~mesh_word_mask(first,count,v),memory_order_acq_rel);
  return 0;
}
static uint32_t mesh_scan(uint32_t count,uint32_t align,uint32_t begin,uint32_t end,
                          _Atomic uint64_t *own,_Atomic uint64_t *hot){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    uint32_t next=first;
    for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
      uint64_t occupied=(atomic_load_explicit(&own[w],memory_order_acquire)|atomic_load_explicit(&hot[w],memory_order_acquire))&mesh_word_mask(first,count,w);
      if(occupied) next=w*64+64-(uint32_t)__builtin_clzll(occupied);
    }
    if(next==first && mesh_claim(own,hot,first,count)) return first;
    if(next>first) first=(next+align-1)/align*align;
  }
  return MESH_ABSENT;
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
    atomic_store_explicit(&mesh_page(c->M)[r].mapping,MESH_ABSENT,memory_order_release);
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[r];
    atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
    buffer->pages=0;
    atomic_store_explicit(&buffer->owner,c->client,memory_order_release);
  }
  return first;
}

/* design/transport.md */
/* design/transport.md */
/* The arena's pages: every one addressable by every client and by the GPU, and registered on every port's device
   since that device came up (mesh-verbs.h device_up); a lease the region cannot meet is refused (ENOMEM).  The bits
   name no owner: a client's lease takes its pages with rows its slot owns (mesh-call.h mesh_section_create), which
   carry them back at its retirement (mesh_buffer_reclaim) or its return (mesh_lease_return). */
uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  _Atomic uint64_t *bits=mesh_arena_bits(c->M);
  return mesh_allocate(c,pages,align,0,mesh_arena_pages(c->M),bits,bits,&c->arena_cursor);
}

/* design/transport.md */
uint32_t mesh_arena_free(struct hdr *m){
  const uint32_t pages=mesh_arena_pages(m),words=(uint32_t)(((uint64_t)pages+63)/64);
  uint32_t held=0;
  for(uint32_t w=0;w<words;w++)
    held+=(uint32_t)__builtin_popcountll(atomic_load_explicit(&mesh_arena_bits(m)[w],memory_order_relaxed)&mesh_word_mask(0,pages,w));
  return pages-held;
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page,uint32_t index){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    struct mesh_page_entry *entry=&mesh_page(c->M)[first+offset/c->M->block];
    atomic_store_explicit(&entry->mapping,((uint64_t)index<<32)|(page+offset),memory_order_relaxed);
    atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(c->M,page+offset)-(uintptr_t)c->M,memory_order_relaxed);
  }
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

/* design/transport.md section 14; design/extents-and-liveness.md, d: whether a row of `first` to `first` + `count` is in
   W(t), a request outstanding on some stream's lane naming its page: a unit posted and not completed, a receive not
   landed or a SEND not done (mesh.h struct mesh_posts: the plan's window in flight, at most a lane's queue each way).
   A row is held for nothing else: a holder that gave or died has no future write targeting any other of its rows (the
   bridge posts a unit only while its holding is its holder's, and tells it first: mesh-link.c unit_told), so they are
   released at once, and a row in W(t) once its unit completes or the port child that posted it is gone (its pid and
   start, struct mesh_stream `child`, no longer a live process: its queue pairs, and the requests on them, went with
   it), whichever launch of the bridge spawned it.  Each lane's units completed are read before those posted, every
   read sequentially consistent after the caller's give or retirement */
static int mesh_pinned(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t s=0;s<m->streams && s<MESH_STREAMS;s++){
    const struct mesh_stream *x=mesh_stream_at(m,s);
    int live=-1;
    for(int way=0;x->rows && way<2;way++)
      for(int lane=0;lane<2;lane++){
        const uint64_t done=atomic_load(&x->posts[way].done[lane]),posted=atomic_load(&x->posts[way].posted[lane]);
        if(done<posted && live<0){const pid_t child=(pid_t)atomic_load(&x->child);live=mesh_holds(child,x->child_start);}
        for(uint64_t u=done;live>0 && u<posted && u-done<MESH_LANE_UNITS;u++)
          if(atomic_load(mesh_unit_row(m,x->rows,way,lane,u))-first<count)return 1;
      }
  }
  return 0;
}
/* design/algorithm-sources.md#programtensor; design/transport.md section 14: a retired client's closed rows released,
   and a lease returned while a request still named a page of it, but those a request outstanding on a lane still
   names (mesh_pinned); their count */
uint32_t mesh_retired_release(struct hdr *m){
  uint32_t kept=0;
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    uint64_t owner=atomic_load_explicit(&buffer->owner,memory_order_acquire);
    if(owner && atomic_load_explicit(&buffer->closed,memory_order_acquire) && mesh_pinned(m,row,buffer->pages/m->block)){kept++;continue;}
    if(owner && atomic_load_explicit(&buffer->closed,memory_order_acquire) &&
       atomic_compare_exchange_strong_explicit(&buffer->owner,&owner,0,memory_order_acq_rel,memory_order_relaxed)){
      atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
      mesh_bits_set(mesh_plane(m,MESH_FREE),row,1);
      atomic_store_explicit(&buffer->owner,owner,memory_order_release);
    }
  }
  mesh_reclaim(m);
  return kept;
}

/* design/transport.md: the lease's rows other than its head disowned and every one of its rows released
   while they stay hot, then the head reclaimed (mesh_buffer_reclaim: its pages' arena bits, then its rows' hot bits),
   so no client claims a row or page of it before it is whole again; while a request outstanding on a lane names a page
   of it (mesh_pinned), the head closed instead, which the bridge releases once none does (mesh_retired_release, each
   liveness turn) */
int mesh_lease_return(struct mesh_ctx *c,uint32_t row){
  struct hdr *m=c->M;
  struct mesh_buffer *buffer=row<mesh_rows(m)?&mesh_buffers(m)[row]:NULL;
  const uint32_t rows=buffer?buffer->pages/m->block:0;
  if(!rows || rows>mesh_rows(m)-row || atomic_load_explicit(&buffer->owner,memory_order_acquire)!=c->client ||
     atomic_load_explicit(&buffer->closed,memory_order_acquire))return EINVAL;
  for(uint32_t r=row+1;r<row+rows;r++)atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_relaxed);
  mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,rows);
  if(mesh_pinned(m,row,rows)){atomic_store_explicit(&buffer->closed,1,memory_order_release);return 0;}
  mesh_buffer_reclaim(m,row);
  return 0;
}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(mesh_plane(c->M,MESH_ROW_OWN),first,count);
}

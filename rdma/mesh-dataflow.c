#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
/* design/pages-and-functions.md#what-the-page-table-is */
static void mesh_reclaim(struct hdr *);
#ifdef MESH_LOOP_FABRIC
/* design/bridge-dataflow.md#the-loopback-fabric: a loop client's bridge, its region and its requests' loopback port (the
   host build of libmesh alone: make host); the region MESH_NAME and MESH_PORT until bound */
static char mesh_loop_region[32]=MESH_NAME;
static uint16_t mesh_loop_service;
int mesh_loop_bind(const char *region,uint16_t service){
  if(!region || region[0]!='/' || strlen(region)>=sizeof mesh_loop_region || !service)return EINVAL;
  strcpy(mesh_loop_region,region);mesh_loop_service=service;
  return 0;
}
#define MESH_REGION_NAME mesh_loop_region
#define MESH_SERVICE_PORT (mesh_loop_service?mesh_loop_service:(uint16_t)atoi(MESH_PORT))
#else
#define MESH_REGION_NAME MESH_NAME
#define MESH_SERVICE_PORT ((uint16_t)atoi(MESH_PORT))
#endif

/* design/bridge-and-ipc.md#region: the node's region mapped, `writable` (a client's) or read-only (a reader's, which
   attaches no client and changes nothing of the bridge's), its descriptor kept in *fd where one is asked for: 0, EINVAL
   where it is no region of this version, ESRCH where its bridge is gone, else the errno */
static int mesh_open(int writable,struct hdr **out,size_t *length,int *fd){
  int file=shm_open(MESH_REGION_NAME,writable?O_RDWR:O_RDONLY,0);
  if(file<0)return errno;
  struct stat info;
  if(fstat(file,&info)){int error=errno;close(file);return error;}
  if(info.st_size<(off_t)sizeof(struct hdr)){close(file);return EINVAL;}
  struct hdr *m=mmap(NULL,(size_t)info.st_size,PROT_READ|(writable?PROT_WRITE:0),MAP_SHARED,file,0);
  int error=errno;
  if(m==MAP_FAILED){close(file);return error;}
  error=m->magic!=MESH_MAGIC || m->version!=MESH_VERSION || m->length>(uint64_t)info.st_size?EINVAL:
    mesh_gone((pid_t)atomic_load(&m->bridge_pid))?ESRCH:0;
  if(error || !fd)close(file);
  if(error){munmap(m,(size_t)info.st_size);return error;}
  *out=m;*length=(size_t)info.st_size;
  if(fd)*fd=file;
  return 0;
}

/* design/algorithm-sources.md#meshobserve */
/* design/bridge-and-ipc.md#ports: each port of the node's bridge, by its index: its peer node while paired (0 else), its
   phase and bandwidth; a port is one the bridge found (its phase not MESH_UNKNOWN), and the bridge finds them in index
   order */
int mesh_observe(struct mesh_link_view *out,uint32_t capacity,uint32_t *node){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  int result=-EINVAL;
  if(m->link_off>=sizeof *m && m->link_off<=m->length && m->links<=(m->length-m->link_off)/sizeof(struct mesh_link_info)){
    *node=m->node;result=0;
    for(uint32_t i=0;i<m->links;i++){
      struct mesh_link_info *link=&mesh_links(m)[i];
      uint32_t phase=atomic_load_explicit(&link->phase,memory_order_acquire);
      if(phase==MESH_UNKNOWN)break;
      if((uint32_t)result<capacity)out[result]=(struct mesh_link_view){.peer=atomic_load_explicit(&link->peer,memory_order_relaxed),
        .phase=phase,.bandwidth=atomic_load_explicit(&link->bandwidth,memory_order_relaxed)};
      result++;
    }
  }
  munmap(m,length);return result;
}

/* design/bridge-and-ipc.md#region: the flows and awaiting programs the bridge holds and the wall clock (ns) at which it
   last held none (0 while it holds any).  The bridge stores idle_since before busy (mesh-flow.c busy_tell), and
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

/* design/bridge-and-ipc.md#region */
int mesh_window(uint64_t *bytes,uint64_t *unleased){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *bytes=(uint64_t)mesh_arena_pages(m)*m->pgsz;*unleased=(uint64_t)mesh_arena_free(m)*m->pgsz;
  munmap(m,length);return 0;
}

/* design/bridge-and-ipc.md#region: the map's records as one serial left them, the seqlock's reader (mesh_map_whole): a
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

/* design/bridge-and-ipc.md#ports, #cables: the one join of two nodes' records: the cables between them, the ports each
   end's record holds paired with the other (the fewer of the two ends' counts), and *slowest the least bandwidth any
   of those ports at either end reports (0: none reports one): the bridge stripes a program between the two over every
   cable, a routed one too (mesh-flow.c route_stripes), so they carry it at their count times the slowest */
static uint32_t mesh_paired(struct mesh_record *record,uint32_t node,uint64_t *slowest){
  uint32_t cables=0;
  for(uint32_t i=0;i<record->ports && i<MESH_PORTS;i++)
    if(atomic_load_explicit(&record->port[i].phase,memory_order_relaxed)==MESH_PAIRED &&
       atomic_load_explicit(&record->port[i].peer,memory_order_relaxed)==node){
      const uint64_t bandwidth=atomic_load_explicit(&record->port[i].bandwidth,memory_order_relaxed);
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
  const uint32_t cables=mesh_join(a,b,&bandwidth);
  if(cables)*beta=bandwidth?8e9/(double)bandwidth:0;
  return (int)cables;
}

/* The records' whole map, its nodes ascending (the records sorted by node), a link a cable where two join, each at the
   join's bandwidth, and its routing table (mesh-collective.c mesh_routes, each node marked by its bridge's number, so
   every bridge and member that holds the same records makes the same routes) */
struct mesh_whole { uint32_t n,(*link)[2],*mark,*next; uint64_t *bandwidth; struct mesh_link_map map; };
static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
static void mesh_whole_free(struct mesh_whole *w){free(w->link);free(w->bandwidth);free(w->mark);free(w->next);}
static int mesh_whole_make(struct mesh_record *r,uint32_t n,struct mesh_whole *w){
  qsort(r,n,sizeof *r,mesh_node_order);
  size_t cables=1;
  uint64_t bandwidth;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++)cables+=(size_t)mesh_join(r+a,r+b,&bandwidth);
  *w=(struct mesh_whole){.n=n,.link=malloc(cables*sizeof *w->link),.bandwidth=malloc(cables*sizeof *w->bandwidth),
    .mark=malloc(((size_t)n+1)*sizeof *w->mark),.next=malloc(((size_t)n*n+1)*sizeof *w->next)};
  if(!w->link || !w->bandwidth || !w->mark || !w->next)return ENOMEM;
  uint32_t links=0;
  for(uint32_t a=0;a<n;a++){
    w->mark[a]=r[a].node;
    for(uint32_t b=a+1;b<n;b++)
      for(uint32_t k=0,joined=mesh_join(r+a,r+b,&bandwidth);k<joined;k++){
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
                            char (*name)[MESH_NAME_BYTES],struct mesh_link_map *map){
  struct mesh_whole w;
  int error=mesh_whole_make(r,count,&w);
  uint32_t *index=malloc(((size_t)n+1)*sizeof *index),*whole=malloc(((size_t)map->nodes+1)*sizeof *whole);
  if(!error && (!index || !whole))error=ENOMEM;
  for(uint32_t i=0;!error && i<n;i++){
    index[i]=count;
    for(uint32_t k=0;k<count;k++)if(r[k].node==members[i])index[i]=k;
  }
  int made=error?-error:mesh_map_members(&w.map,n?index:NULL,n,whole,map);
  for(uint32_t i=0;made>=0 && i<(uint32_t)made+map->relays;i++){
    if(node)node[i]=i<(uint32_t)made && n?members[i]:r[whole[i]].node;
    if(name && whole[i]<count)memcpy(name[i],r[whole[i]].name,MESH_NAME_BYTES);
    else if(name)name[i][0]=0;
  }
  free(index);free(whole);mesh_whole_free(&w);
  return made;
}

int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,char (*name)[MESH_NAME_BYTES],struct mesh_link_map *map){
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  int count=mesh_records_read(m,length,&r);
  munmap(m,length);
  if(count<0)return count;
  int result=mesh_records_map(r,(uint32_t)count,members,n,node,name,map);
  free(r);
  return result;
}

/* design/recovery.md#diagnosis */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity){
  struct hdr *m;size_t length;struct mesh_record *r=NULL;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  /* the records under both serials: the map's (mesh_records_read) and the health's, which a health-only change moves
     (mesh-flow.c health_put); a read that a health write overlapped is read again, at most 1024 times */
  int count=-EAGAIN;
  for(int tries=0;tries<1024;tries++){
    const uint64_t before=atomic_load_explicit(&m->health,memory_order_acquire);
    if(before&1)continue;
    count=mesh_records_read(m,length,&r);
    if(count<0 || atomic_load_explicit(&m->health,memory_order_acquire)==before)break;
    free(r);r=NULL;count=-EAGAIN;
  }
  munmap(m,length);
  if(count<0)return count;
  for(int i=0;i<count && (uint32_t)i<capacity;i++){
    out[i].node=r[i].node;memcpy(out[i].name,r[i].name,MESH_NAME_BYTES);out[i].name[MESH_NAME_BYTES-1]=0;
    out[i].window=r[i].window;out[i].memory=r[i].memory;
    memcpy(out[i].model,r[i].model,MESH_MODEL_BYTES);out[i].model[MESH_MODEL_BYTES-1]=0;
    out[i].health=r[i].health;
    out[i].health.session[sizeof out[i].health.session-1]=0;out[i].health.shutdown[sizeof out[i].health.shutdown-1]=0;
  }
  free(r);
  return count;
}

int mesh_map_wait(uint64_t *serial,uint64_t bound_ns){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return error;
  const uint64_t until=clock_gettime_nsec_np(CLOCK_UPTIME_RAW)+bound_ns;
  uint64_t seen=atomic_load_explicit(&m->map,memory_order_acquire);
  for(uint64_t at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);seen==*serial && at<until;at=clock_gettime_nsec_np(CLOCK_UPTIME_RAW)){
    os_sync_wait_on_address_with_timeout(&m->map,seen,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,until-at);
    seen=atomic_load_explicit(&m->map,memory_order_acquire);
  }
  error=seen!=*serial?0:ETIMEDOUT;
  *serial=seen;
  munmap(m,length);
  return error;
}

uint32_t mesh_route(struct hdr *m,uint32_t from,uint32_t to,uint32_t **path){
  struct mesh_record *r=NULL;
  int count=mesh_records_read(m,m->length,&r);
  uint32_t hops=0,a=UINT32_MAX,b=UINT32_MAX,*nodes=count>0?malloc(((size_t)count+1)*sizeof *nodes):NULL;
  struct mesh_whole w={0};
  if(nodes && !mesh_whole_make(r,(uint32_t)count,&w)){
    const uint32_t n=(uint32_t)count;
    for(uint32_t k=0;k<n;k++){if(r[k].node==from)a=k;if(r[k].node==to)b=k;}
    /* the table's route between a and b (mesh-plan.h mesh_route_path: the one from b to a reversed) */
    hops=a<n && b<n?mesh_route_path(&w.map,a,b,nodes):0;
    for(uint32_t i=0;i<hops;i++)nodes[i]=r[nodes[i]].node;
    if(hops && nodes[hops-1]!=to)hops=0;
  }
  mesh_whole_free(&w);free(r);
  if(!hops){free(nodes);nodes=NULL;}
  *path=nodes;
  return hops;
}

/* design/algorithm-sources.md#programtensor */
/* design/bridge-and-ipc.md#clients, #control: a client's slot given up, its program stopped, its rows closed for the
   bridge to release once its programs left the ports: by an abort (mesh_retire: a revoke, or the bridge where its
   process is gone), the slot vacant at once and its flows failed (FAIL), or in order (mesh_detach), the slot left
   MESH_ORDERLY for the bridge, which closes its flows (their released cells posted, then END) and vacates it */
static void mesh_retire_as(struct hdr *m,uint64_t client,int orderly){
  struct mesh_client *slot=mesh_client_at(m,client);
  if(!client || (client&(MESH_RETIRING|MESH_ORDERLY)) || !atomic_compare_exchange_strong(&slot->owner,&client,client|MESH_RETIRING))return;
  atomic_store_explicit(&slot->configured,0,memory_order_release);
  for(uint32_t d=0;d<MESH_DESTINATIONS;d++)for(uint32_t q=0;q<MESH_QUEUES*2;q++)atomic_store_explicit(slot->tx[d].length+q,0,memory_order_release);
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==client){
      if(buffer->pages)atomic_store_explicit(&buffer->closed,1,memory_order_release);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  /* design/bridge-dataflow.md#stripes: its request rings deregistered (the bridge takes none of them past here) */
  _Atomic uint64_t *rings=mesh_rings(m);
  for(uint32_t r=0;r<MESH_RINGS;r++){
    uint64_t at=atomic_load_explicit(rings+r,memory_order_acquire);
    if(at && at>=m->data_off && at<=m->length-sizeof(struct mesh_requests) && ((const struct mesh_requests *)((char *)m+at))->owner==client)
      atomic_compare_exchange_strong(rings+r,&at,0);
  }
  atomic_store_explicit(&slot->owner,orderly?client|MESH_RETIRING|MESH_ORDERLY:0,memory_order_release);
  /* a start waiting on its destinations' status (mesh-call.c mesh_transfers_start) reads the slot gone at once: a
     revoke from another thread of its process retires the program it waits for (nccl-mesh.c ncclCommRevoke) */
  for(uint32_t d=0;d<MESH_DESTINATIONS && slot->tx[d].node;d++)
    os_sync_wake_by_address_all(&slot->status[d].prepared,sizeof slot->status[d].prepared,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  /* design/prepared-machine.md#M26 */
  mesh_notify(&m->control);
}
void mesh_retire(struct hdr *m,uint64_t client){ mesh_retire_as(m,client,0); }

static uint64_t mesh_key(const char *name){
  uint64_t key=UINT64_C(0xcbf29ce484222325);
  for(const char *p=name;p && *p;p++)key=(key^(unsigned char)*p)*UINT64_C(0x100000001b3);
  return name && *name?key:0;
}

int mesh_slot_take(struct mesh_ctx *c,struct hdr *m,uint64_t key){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<24)-1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)m+m->client_off)+s;
    uint64_t vacant=0,owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid();
    if(!atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
    slot->key=key;
    atomic_store(&slot->depth,0);atomic_store(&slot->cyclic,0);
    slot->targets=0;slot->target_count=0;
    memset(slot->tx,0,sizeof slot->tx);memset(slot->status,0,sizeof slot->status);
    for(uint32_t k=0;k<MESH_DESTINATIONS*MESH_QUEUES;k++)atomic_store_explicit(slot->delivered+k,0,memory_order_relaxed);
    atomic_store_explicit(&slot->ending,0,memory_order_relaxed);
    *c=(struct mesh_ctx){.M=m,.len=c->len,.client=owner,.fd=c->fd};
    return 0;
  }
  return EBUSY;
}

int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  struct hdr *m;size_t length;int file;
  int error=mesh_open(1,&m,&length,&file);
  if(error)return error;
  *c=(struct mesh_ctx){.len=length,.fd=file};
  error=mesh_slot_take(c,m,mesh_key(name));
  if(error){munmap(m,length);close(file);*c=(struct mesh_ctx){0};}
  return error;
}

int mesh_program_attach(struct mesh_ctx *program,const struct mesh_ctx *client,const char *name){
  *program=(struct mesh_ctx){.fd=-1};
  int error=mesh_gone((pid_t)atomic_load(&client->M->bridge_pid))?ESRCH:mesh_slot_take(program,client->M,mesh_key(name));
  if(error)*program=(struct mesh_ctx){0};
  return error;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire_as(c->M,c->client,1);
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
    *mesh_delivery_at(c->M,r)=(struct mesh_delivery){0};
    atomic_store_explicit(&buffer->owner,c->client,memory_order_release);
  }
  return first;
}

/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
/* The arena's pages: every one addressable by every client and by the GPU, and registered on every port's device
   since that device came up (mesh-verbs.h device_up); a lease the window cannot meet is refused (ENOMEM).  The bits
   name no owner: a client's lease takes its pages with rows its slot owns (mesh-call.h mesh_section_create), which
   carry them back at its retirement (mesh_buffer_reclaim) or its return (mesh_lease_return). */
uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  _Atomic uint64_t *bits=mesh_arena_bits(c->M);
  return mesh_allocate(c,pages,align,0,mesh_arena_pages(c->M),bits,bits,&c->arena_cursor);
}

/* design/bridge-dataflow.md#stripes: the exact pages a stripe's two ends place alike, claimed as a scan's are; a range a
   returned lease still holds hot is reclaimed first, as mesh_allocate's last scan does */
uint32_t mesh_arena_claim(struct mesh_ctx *c,uint32_t first,uint32_t pages){
  const uint32_t arena=mesh_arena_pages(c->M);
  if(!pages || first>arena || pages>arena-first){ errno=EINVAL; return MESH_ABSENT; }
  _Atomic uint64_t *bits=mesh_arena_bits(c->M);
  if(mesh_claim(bits,bits,first,pages))return first;
  mesh_reclaim(c->M);
  if(mesh_claim(bits,bits,first,pages))return first;
  errno=ENOMEM;
  return MESH_ABSENT;
}

/* design/prepared-machine.md#M09 */
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
  mesh_reclaim(m);
}

/* design/prepared-machine.md#M09: the lease's rows other than its head disowned and every one of its rows released
   while they stay hot, then the head reclaimed (mesh_buffer_reclaim: its pages' arena bits, then its rows' hot bits),
   so no client claims a row or page of it before it is whole again */
int mesh_lease_return(struct mesh_ctx *c,uint32_t row){
  struct hdr *m=c->M;
  struct mesh_buffer *buffer=row<mesh_rows(m)?&mesh_buffers(m)[row]:NULL;
  const uint32_t rows=buffer?buffer->pages/m->block:0;
  if(!rows || rows>mesh_rows(m)-row || atomic_load_explicit(&buffer->owner,memory_order_acquire)!=c->client ||
     atomic_load_explicit(&buffer->closed,memory_order_acquire))return EINVAL;
  for(uint32_t r=row+1;r<row+rows;r++)atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_relaxed);
  mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,rows);
  mesh_buffer_reclaim(m,row);
  return 0;
}

/* design/algorithm-sources.md#program */
void mesh_rows_release(struct mesh_ctx *c,uint32_t first,uint32_t count){
  mesh_bits_clear(mesh_plane(c->M,MESH_ROW_OWN),first,count);
}

/* design/prepared-machine.md#M13 */
/* design/prepared-machine.md#M04 */
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct mesh_ctx *c,uint32_t row,struct prepared_publication *records){
  uint32_t count;
  const struct mesh_target *targets=mesh_row_targets(c->M,c->client,row,&count);
  for(uint32_t i=0;records && i<count;i++){
    struct mesh_send *cell=(void *)((char *)c->M+targets[i].cells);
    records[i]=(struct prepared_publication){.destination=(uintptr_t)cell,.argument=cell->request.wr_id};
  }
  return count;
}

/* design/bridge-and-ipc.md#requests: a connection to this node's bridge that has sent `request` and read the bridge's
   answer: its descriptor where the answer is 0, else -errno */
static int bridge_ask(const struct mesh_request *request){
  int fd=socket(AF_INET6,SOCK_STREAM,0),on=1;
  if(fd<0)return -errno;
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons(MESH_SERVICE_PORT),.sin6_addr=IN6ADDR_LOOPBACK_INIT};
  int32_t answer=0;
  size_t got=0;
  if(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) || setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&on,sizeof on) ||
     connect(fd,(struct sockaddr *)&to,sizeof to) || send(fd,request,sizeof *request,0)!=(ssize_t)sizeof *request){
    const int error=errno==ECONNREFUSED?ESRCH:errno;close(fd);return -error;
  }
  while(got<sizeof answer){
    ssize_t n=recv(fd,(char *)&answer+got,sizeof answer-got,0);
    if(n<0 && errno==EINTR)continue;
    if(n<=0){const int error=n?errno:ECONNRESET;close(fd);return -error;}
    got+=(size_t)n;
  }
  if(answer){close(fd);return -answer;}
  return fd;
}
int mesh_request(uint32_t node,const char *service,uint64_t bytes){
  struct mesh_request request={.magic=MESH_REQUEST_MAGIC,.kind=MESH_REQUEST_OPEN,.node=node,.bytes=bytes};
  if(!service || strlen(service)>=sizeof request.service)return -EINVAL;
  strcpy(request.service,service);
  return bridge_ask(&request);
}
int mesh_ledger(const char *name,const char *path){
  struct mesh_request request={.magic=MESH_REQUEST_MAGIC,.kind=MESH_REQUEST_LEDGER,.bytes=mesh_key(name)};
  if(!name || !*name || !path || strlen(path)>=sizeof request.path || path[0]!='/')return -EINVAL;
  strcpy(request.service,"ledger");strcpy(request.path,path);
  const int fd=bridge_ask(&request);
  if(fd<0)return fd;
  close(fd);
  return 0;
}
int mesh_stream(uint32_t node,uint64_t bytes,char *result,size_t room){
  struct mesh_request request={.magic=MESH_REQUEST_MAGIC,.kind=MESH_REQUEST_STREAM,.node=node,.bytes=bytes};
  if(!result || !room)return -EINVAL;
  strcpy(request.service,"stream");
  const int fd=bridge_ask(&request);
  if(fd<0)return fd;
  size_t got=0;
  for(ssize_t n;got+1<room && (n=recv(fd,result+got,room-1-got,0))!=0;){
    if(n<0 && errno==EINTR)continue;
    if(n<0){const int error=errno;close(fd);return -error;}
    got+=(size_t)n;
  }
  result[got]=0;
  close(fd);
  return 0;
}
int mesh_serve(const char *service,const char *path){
  struct mesh_request request={.magic=MESH_REQUEST_MAGIC,.kind=MESH_REQUEST_SERVE};
  if(!service || !path || strlen(service)>=sizeof request.service || strlen(path)>=sizeof request.path || path[0]!='/')return -EINVAL;
  strcpy(request.service,service);strcpy(request.path,path);
  return bridge_ask(&request);
}

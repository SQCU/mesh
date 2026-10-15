#include "mesh-dataflow.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
/* design/pages-and-functions.md#what-the-page-table-is */
static void mesh_reclaim(struct hdr *);

/* design/bridge-and-ipc.md#region: the node's region mapped, `writable` (a client's) or read-only (a reader's, which
   attaches no client and changes nothing of the bridge's), its descriptor kept in *fd where one is asked for: 0, EINVAL
   where it is no region of this version, ESRCH where its bridge is gone, else the errno */
static int mesh_open(int writable,struct hdr **out,size_t *length,int *fd){
  int file=shm_open(MESH_NAME,writable?O_RDWR:O_RDONLY,0);
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

/* design/bridge-and-ipc.md#ports: the one join of two nodes' records */
static uint64_t mesh_paired(struct mesh_record *record,uint32_t node){
  uint64_t widest=0;
  for(uint32_t i=0;i<record->ports && i<MESH_PORTS;i++)
    if(atomic_load_explicit(&record->port[i].phase,memory_order_relaxed)==MESH_PAIRED &&
       atomic_load_explicit(&record->port[i].peer,memory_order_relaxed)==node){
      uint64_t bandwidth=atomic_load_explicit(&record->port[i].bandwidth,memory_order_relaxed)+1;
      if(bandwidth>widest)widest=bandwidth;
    }
  return widest;
}
int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  uint64_t one=mesh_paired(a,b->node),other=mesh_paired(b,a->node),widest=(one>other?one:other)-1;
  if(!one || !other)return 0;
  *beta=widest?8e9/(double)widest:MESH_BETA_NS;
  return 1;
}

/* The records' whole map, its nodes ascending (the records sorted by node), a link where two join, at MESH_ALPHA_US and
   the join's beta, and its shortest-path trees (mesh-collective.c mesh_trees_shortest: tree r rooted at node r) */
struct mesh_whole { uint32_t n,(*link)[2],*root,*parent; double (*cost)[2],*weight; struct mesh_link_map map; uint32_t trees; };
static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
static void mesh_whole_free(struct mesh_whole *w){free(w->link);free(w->cost);free(w->root);free(w->parent);free(w->weight);}
static int mesh_whole_make(struct mesh_record *r,uint32_t n,struct mesh_whole *w){
  qsort(r,n,sizeof *r,mesh_node_order);
  size_t pairs=(size_t)n*(n?n-1:0)/2+1;
  *w=(struct mesh_whole){.n=n,.link=malloc(pairs*sizeof *w->link),.cost=malloc(pairs*sizeof *w->cost),
    .root=malloc(((size_t)n+1)*sizeof *w->root),.parent=malloc(((size_t)n*n+1)*sizeof *w->parent),.weight=malloc(((size_t)n+1)*sizeof *w->weight)};
  if(!w->link || !w->cost || !w->root || !w->parent || !w->weight)return ENOMEM;
  uint32_t links=0;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++)if(mesh_joined(r+a,r+b,&w->cost[links][1])){
    w->link[links][0]=a;w->link[links][1]=b;w->cost[links][0]=MESH_ALPHA_US;links++;
  }
  w->map=(struct mesh_link_map){.kind=MESH_LINKS_GRAPH,.nodes=n,.links=links,.link=w->link,.cost=w->cost};
  w->trees=n?mesh_trees_shortest(&w->map,w->root,w->parent,w->weight):0;
  return 0;
}
/* The route between whole-map nodes a and b on fewest links: the lower's path up the higher's tree (mesh_trees_path),
   from a to b; its node count (0: none, the map not connected) */
static uint32_t mesh_whole_route(const struct mesh_whole *w,uint32_t a,uint32_t b,uint32_t *path){
  if(a>=w->n || b>=w->n || a==b || w->trees!=w->n)return 0;
  const struct mesh_trees trees={.nodes=w->n,.count=w->trees,.root=w->root,.parent=w->parent,.log_weight=w->weight};
  uint32_t count=mesh_trees_path(&trees,a<b?a:b,a<b?b:a,path);
  for(uint32_t i=0;a>b && i<count/2;i++){uint32_t t=path[i];path[i]=path[count-1-i];path[count-1-i]=t;}
  return count;
}
static uint32_t mesh_part(uint32_t *part,uint32_t v){
  while(part[v]!=v)v=part[v]=part[part[v]];
  return v;
}

/* design/algorithm-sources.md#meshobserve: mesh-plan.h mesh_map over a copy of the records */
static int mesh_records_map(struct mesh_record *r,uint32_t count,const uint32_t *members,uint32_t n,uint32_t *node,
                            char (*name)[MESH_NAME_BYTES],struct mesh_link_map *map){
  struct mesh_whole w;
  int error=mesh_whole_make(r,count,&w);
  const uint32_t N=n?n:count,room=map->links;
  uint32_t *at=malloc(((size_t)N+1)*sizeof *at),*place=malloc(((size_t)count+1)*sizeof *place),*part=malloc(((size_t)N+1)*sizeof *part),
    *path=malloc(((size_t)count+1)*sizeof *path),links=0;
  if(!error && (!at || !place || !part || !path))error=ENOMEM;
  /* map node i is record at[i] (count: none), record k map node place[k] (N: none) */
  for(uint32_t k=0;!error && k<count;k++)place[k]=n?N:k;
  for(uint32_t i=0;!error && i<N;i++){
    at[i]=n?count:i;part[i]=i;
    for(uint32_t k=0;n && k<count;k++)if(r[k].node==members[i]){at[i]=k;place[k]=i;}
  }
  /* the map nodes' own links, as the whole map holds them */
  for(uint32_t l=0;!error && l<w.map.links;l++){
    const uint32_t a=place[w.link[l][0]],b=place[w.link[l][1]];
    if(a==N || b==N)continue;
    if(links<room){map->link[links][0]=a;map->link[links][1]=b;map->cost[links][0]=w.cost[l][0];map->cost[links][1]=w.cost[l][1];}
    links++;part[mesh_part(part,a)]=mesh_part(part,b);
  }
  /* members left in parts: the routed pair of least cost across two parts linked at its route's cost (its links'
     alphas summed, and their betas), until one part remains or no route joins two [Kruskal 1956] */
  for(uint32_t best[2]={0,0};!error && n && best[0]<N;){
    double least[2]={0,0};
    best[0]=N;
    for(uint32_t a=0;a<N;a++)for(uint32_t b=a+1;b<N;b++){
      uint32_t hops=mesh_part(part,a)==mesh_part(part,b)?0:mesh_whole_route(&w,at[a],at[b],path);
      double alpha=0,beta=0,link[2];
      if(hops<2)continue;
      for(uint32_t h=0;h+1<hops;h++){mesh_link_cost(&w.map,path[h],path[h+1],MESH_ALPHA_US,MESH_BETA_NS,link,link+1);alpha+=link[0];beta+=link[1];}
      if(best[0]==N || alpha<least[0] || (alpha==least[0] && beta<least[1])){best[0]=a;best[1]=b;least[0]=alpha;least[1]=beta;}
    }
    if(best[0]==N)break;
    if(links<room){map->link[links][0]=best[0];map->link[links][1]=best[1];map->cost[links][0]=least[0];map->cost[links][1]=least[1];}
    links++;part[mesh_part(part,best[0])]=mesh_part(part,best[1]);
  }
  if(!error && (map->nodes<N || room<links)){map->nodes=N;map->links=links;error=ENOSPC;}
  else if(!error){
    for(uint32_t i=0;i<N;i++){
      if(node)node[i]=n?members[i]:r[i].node;
      if(name && at[i]<count)memcpy(name[i],r[at[i]].name,MESH_NAME_BYTES);
      else if(name)name[i][0]=0;
    }
    map->nodes=N;map->links=links;
    map->kind=(uint64_t)links*2==(uint64_t)N*(N-1)?MESH_LINKS_MESH:MESH_LINKS_GRAPH;
  }
  free(at);free(place);free(part);free(path);mesh_whole_free(&w);
  return error?-error:(int)N;
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
    for(uint32_t k=0;k<(uint32_t)count;k++){if(r[k].node==from)a=k;if(r[k].node==to)b=k;}
    hops=mesh_whole_route(&w,a,b,nodes);
    for(uint32_t i=0;i<hops;i++)nodes[i]=r[nodes[i]].node;
  }
  mesh_whole_free(&w);free(r);
  if(!hops){free(nodes);nodes=NULL;}
  *path=nodes;
  return hops;
}

/* design/algorithm-sources.md#programtensor */
/* design/bridge-and-ipc.md#clients: a client's slot given up (by its detach, or by the bridge where its process is
   gone): its program stopped, its rows closed for the bridge to release once its programs left the ports */
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_client *slot=mesh_client_at(m,client);
  if(!client || (client&MESH_RETIRING) || !atomic_compare_exchange_strong(&slot->owner,&client,client|MESH_RETIRING))return;
  atomic_store_explicit(&slot->configured,0,memory_order_release);
  for(uint32_t d=0;d<MESH_DESTINATIONS;d++)for(uint32_t q=0;q<MESH_QUEUES*2;q++)atomic_store_explicit(slot->tx[d].length+q,0,memory_order_release);
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==client){
      if(buffer->pages)atomic_store_explicit(&buffer->closed,1,memory_order_release);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  atomic_store_explicit(&slot->owner,0,memory_order_release);
  /* design/prepared-machine.md#M26 */
  mesh_notify(&m->control);
}

static uint64_t mesh_key(const char *name){
  uint64_t key=UINT64_C(0xcbf29ce484222325);
  for(const char *p=name;p && *p;p++)key=(key^(unsigned char)*p)*UINT64_C(0x100000001b3);
  return name && *name?key:0;
}

int mesh_slot_take(struct mesh_ctx *c,struct hdr *m,uint64_t key){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<25)-1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)m+m->client_off)+s;
    uint64_t vacant=0,owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid();
    if(!atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
    slot->key=key;
    atomic_store(&slot->depth,0);atomic_store(&slot->cyclic,0);
    memset(slot->tx,0,sizeof slot->tx);memset(slot->status,0,sizeof slot->status);
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
    struct mesh_publication *publication=mesh_publication_at(c->M,r);
    publication->sends=publication->device_input=publication->device_stride=0;
    atomic_store_explicit(&publication->argument,0,memory_order_relaxed);
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

/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M18 */
/* a row's publication target toward destination `destination`: until mesh_transfers_prepare lays its cells, the
   client's program toward that destination names it */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *c,uint32_t row,uint32_t destination){
  struct hdr *m=c->M;
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t *count=&publication->sends;
  struct mesh_target *targets=publication->targets;
  uint64_t first=(uint64_t)((char *)&mesh_client_at(m,c->client)->tx[destination]-(char *)m);
  for(uint32_t i=0;i<*count;i++)if(targets[i].stream==first)return &targets[i];
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

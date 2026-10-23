#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
/* design/transport.md section 15, "The library" */
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
int mesh_region(uint64_t *bytes,uint64_t *unleased){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *bytes=(uint64_t)mesh_arena_pages(m)*m->pgsz;*unleased=(uint64_t)mesh_arena_free(m)*m->pgsz;
  munmap(m,length);return 0;
}
static int serial_even(struct hdr *m,_Atomic uint64_t *word,uint64_t *before){
  while((*before=atomic_load_explicit(word,memory_order_acquire))&1)
    if(os_sync_wait_on_address_with_timeout(word,*before,sizeof *word,OS_SYNC_WAIT_ON_ADDRESS_SHARED,OS_CLOCK_MACH_ABSOLUTE_TIME,
         MESH_PAUSE_NS)<0 && errno==ETIMEDOUT && mesh_gone((pid_t)atomic_load(&m->bridge_pid)))return ESRCH;
  return 0;
}
static int mesh_records_read(int health,struct mesh_record **out){
  struct hdr *m;size_t length;
  const int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  int count;
  for(uint64_t before,held=0;;){
    if((health && serial_even(m,&m->health,&held)) || serial_even(m,&m->map,&before)){count=-ESRCH;break;}
    const uint64_t n=atomic_load_explicit(&m->map_nodes,memory_order_relaxed),at=atomic_load_explicit(&m->map_off,memory_order_relaxed);
    const int fits=at<=length && n<=(length-at)/sizeof(struct mesh_record);
    struct mesh_record *copy=fits?malloc((n?n:1)*sizeof *copy):NULL;
    if(copy)memcpy(copy,(char *)m+at,n*sizeof *copy);
    if(mesh_map_whole(m,before) && (!health || atomic_load_explicit(&m->health,memory_order_acquire)==held)){
      count=copy?(int)n:fits?-ENOMEM:-EINVAL;*out=copy;break;
    }
    free(copy);
  }
  munmap(m,length);
  return count;
}
/* design/transport.md section 3, RECORD; section 15, "The library" */
uint32_t mesh_joined(const struct mesh_record *a,const struct mesh_record *b,uint64_t *bandwidth,uint32_t room){
  uint32_t count=0;
  for(uint32_t i=0;i<a->reached && i<MESH_STREAMS;i++)
    for(uint32_t j=0;a->reach[i].node==b->node && j<b->reached && j<MESH_STREAMS;j++){
      const struct mesh_reach *x=a->reach+i,*y=b->reach+j;
      if(y->node!=a->node || (x->cable!=y->cable && x->cable && y->cable))continue;
      if(count<room)bandwidth[count]=!x->bandwidth || (y->bandwidth && y->bandwidth<x->bandwidth)?y->bandwidth:x->bandwidth;
      count++;
    }
  return count;
}
struct mesh_whole { struct mesh_link_map map; uint32_t *next; };
static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
static void mesh_whole_free(struct mesh_whole *w){free(w->map.link);free(w->map.bandwidth);free(w->map.mark);free(w->next);}
static int mesh_whole_make(struct mesh_record *r,uint32_t n,struct mesh_whole *w){
  qsort(r,n,sizeof *r,mesh_node_order);
  size_t joins=1;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++)joins+=mesh_joined(r+a,r+b,NULL,0);
  *w=(struct mesh_whole){.map={.kind=MESH_LINKS_GRAPH,.nodes=n,.link=malloc(joins*sizeof *w->map.link),.bandwidth=malloc(joins*sizeof *w->map.bandwidth),
    .mark=malloc(((size_t)n+1)*sizeof *w->map.mark)},.next=malloc(((size_t)n*n+1)*sizeof *w->next)};
  if(!w->map.link || !w->map.bandwidth || !w->map.mark || !w->next)return ENOMEM;
  for(uint32_t a=0;a<n;a++){
    w->map.mark[a]=r[a].node;
    for(uint32_t b=a+1;b<n;b++)
      for(uint32_t k=mesh_joined(r+a,r+b,w->map.bandwidth+w->map.links,(uint32_t)(joins-w->map.links));k--;w->map.links++){
        w->map.link[w->map.links][0]=a;w->map.link[w->map.links][1]=b;
      }
  }
  const int error=n?mesh_routes(&w->map,w->next):0;
  if(!error)w->map.next=w->next;
  return error;
}
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
int mesh_map(const uint32_t *members,uint32_t n,uint32_t *node,struct mesh_link_map *map){
  struct mesh_record *r=NULL;
  const int count=mesh_records_read(0,&r);
  const int result=count<0?count:mesh_records_map(r,(uint32_t)count,members,n,node,map);
  free(r);
  return result;
}
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity){
  struct mesh_record *r=NULL;
  const int count=mesh_records_read(1,&r);
  for(int i=0;i<count && (uint32_t)i<capacity;i++){
    out[i].node=r[i].node;
    out[i].region=r[i].region;out[i].memory=r[i].memory;
    memcpy(out[i].model,r[i].model,MESH_MODEL_BYTES);out[i].model[MESH_MODEL_BYTES-1]=0;
    out[i].health=r[i].health;
  }
  free(r);
  return count;
}
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
/* design/transport.md section 2, the client slots */
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_client *slot=mesh_client_at(m,client);
  uint64_t own=client&~MESH_RETIRING;
  if(!own || (!(client&MESH_RETIRING) && !atomic_compare_exchange_strong(&slot->owner,&own,own|MESH_RETIRING)))return;
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==own){
      if(buffer->pages)mesh_lease_close(buffer,own);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  atomic_store_explicit(&slot->owner,0,memory_order_release);
  mesh_retired_release(m,own);
}
static int mesh_slot_take(struct mesh_ctx *c,struct hdr *m){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<25)-1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)m+m->client_off)+s;
    uint64_t vacant=0,owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid();
    if(!atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
    slot->start=mesh_started(getpid());
    *c=(struct mesh_ctx){.M=m,.len=c->len,.client=owner,.fd=c->fd};
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
/* design/transport.md section 15, "The library" */
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
static uint32_t mesh_allocate(struct mesh_ctx *c,uint32_t count,uint32_t align,uint32_t begin,uint32_t end,
                              _Atomic uint64_t *own,_Atomic uint64_t *hot,uint32_t *cursor){
  uint32_t resume=*cursor<begin||*cursor>end?begin:*cursor;
  uint32_t first=mesh_scan(count,align,resume,end,own,hot);
  if(first==MESH_ABSENT && resume!=begin)first=mesh_scan(count,align,begin,end,own,hot);
  if(first==MESH_ABSENT){ mesh_retired_release(c->M,c->client); first=mesh_scan(count,align,begin,end,own,hot); }
  if(first==MESH_ABSENT){ errno=ENOMEM; return MESH_ABSENT; }
  *cursor=first+count;
  return first;
}
static uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(c,count,1,0,mesh_rows(c->M),mesh_plane(c->M,MESH_ROW_OWN),mesh_plane(c->M,MESH_ROW_HOT),&c->row_cursor);
  if(first==MESH_ABSENT) return first;
  for(uint32_t r=first;r<first+count;r++){
    atomic_store_explicit(&mesh_page(c->M)[r].mapping,MESH_ABSENT,memory_order_release);
    struct mesh_buffer *buffer=&mesh_buffers(c->M)[r];
    atomic_store_explicit(&buffer->closed,(atomic_load_explicit(&buffer->closed,memory_order_relaxed)|MESH_LEASE_STATE)+1u,memory_order_release);
    buffer->pages=0;
    atomic_store_explicit(&buffer->owner,c->client,memory_order_release);
  }
  return first;
}
uint32_t mesh_arena_alloc(struct mesh_ctx *c,uint32_t pages,uint32_t align){
  if(!pages || !align){ errno=EINVAL; return MESH_ABSENT; }
  _Atomic uint64_t *bits=mesh_arena_bits(c->M);
  return mesh_allocate(c,pages,align,0,mesh_arena_pages(c->M),bits,bits,&c->arena_cursor);
}
uint32_t mesh_arena_free(struct hdr *m){
  const uint32_t pages=mesh_arena_pages(m),words=(uint32_t)(((uint64_t)pages+63)/64);
  uint32_t held=0;
  for(uint32_t w=0;w<words;w++)
    held+=(uint32_t)__builtin_popcountll(atomic_load_explicit(&mesh_arena_bits(m)[w],memory_order_relaxed)&mesh_word_mask(0,pages,w));
  return pages-held;
}
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz,capacity=mesh_arena_pages(m);
  if(bytes>capacity*(size_t)m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>capacity/count)return ENOMEM;
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    buffer->pages=(uint32_t)span;
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(mesh_plane(m,MESH_ROW_HOT),first,rows);
  uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block);
  if(page==MESH_ABSENT){
    int error=errno;
    mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),first,rows);
    for(uint32_t r=first;r<first+rows;r++){mesh_buffers(m)[r].pages=0;atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_release);}
    mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),first,rows);return error;
  }
  for(uint32_t slot=0;slot<count;slot++)
    for(uint32_t offset=0;offset<(uint32_t)span;offset+=m->block){
      struct mesh_page_entry *entry=&mesh_page(m)[first+slot*stride+offset/m->block];
      atomic_store_explicit(&entry->mapping,((uint64_t)slot<<32)|(page+slot*(uint32_t)span+offset),memory_order_relaxed);
      atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(m,page+slot*(uint32_t)span+offset)-(uintptr_t)m,memory_order_relaxed);
    }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride};
  return 0;
}
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){
  return (char *)context->M+atomic_load_explicit(&mesh_page(context->M)[section.first+index*section.stride].address,memory_order_relaxed);
}
static void mesh_lease_free(struct hdr *m,uint32_t row,uint32_t from){
  struct mesh_buffer *buffer=&mesh_buffers(m)[row];
  uint32_t was=atomic_load_explicit(&buffer->closed,memory_order_acquire);
  if(!mesh_lease_move(buffer,&was,from,MESH_LEASE_FREE))return;
  const uint32_t pages=buffer->pages;
  for(uint32_t r=row;r<row+pages/m->block;r++)atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_release);
  for(uint32_t offset=0;offset<pages;offset+=m->block){
    uint32_t page=atomic_load_explicit(&mesh_page(m)[row+offset/m->block].mapping,memory_order_acquire);
    if(page!=MESH_ABSENT)mesh_bits_clear(mesh_arena_bits(m),page,m->block);
  }
  mesh_bits_clear(mesh_plane(m,MESH_ROW_HOT),row,pages/m->block);
}
/* design/transport.md section 14, "A holding's end"; design/extents-and-liveness.md, d */
static int mesh_pinned(struct hdr *m,uint32_t first,uint32_t count,uint64_t client){
  const int gone=atomic_load_explicit(&mesh_client_at(m,client)->owner,memory_order_acquire)!=client;
  for(uint32_t s=0;s<m->streams && s<MESH_STREAMS;s++){
    const struct mesh_stream *x=mesh_stream_at(m,s);
    const uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
    int live=-1;
    if((owner&~MESH_STREAM_GIVEN)==client && ((owner&MESH_STREAM_GIVEN) || gone)){
      live=mesh_holds((pid_t)atomic_load_explicit(&x->child,memory_order_acquire),x->child_start);
      if(live)return 1;
    }
    for(int way=0;x->rows && way<2;way++)
      for(int lane=0;lane<2;lane++){
        const uint64_t done=atomic_load_explicit(&x->posts[way].done[lane],memory_order_acquire);
        const uint64_t posted=atomic_load_explicit(&x->posts[way].posted[lane],memory_order_acquire);
        if(done<posted && live<0)live=mesh_holds((pid_t)atomic_load_explicit(&x->child,memory_order_acquire),x->child_start);
        for(uint64_t u=done;live>0 && u<posted && u-done<MESH_LANE_UNITS;u++)
          if(atomic_load_explicit(mesh_unit_row(m,x->rows,way,lane,u),memory_order_relaxed)-first<count)return 1;
      }
  }
  return 0;
}
static int mesh_release(struct hdr *m,uint32_t row,uint64_t client){
  struct mesh_buffer *buffer=&mesh_buffers(m)[row];
  uint32_t was=atomic_load_explicit(&buffer->closed,memory_order_acquire);
  if(!client && (was&MESH_LEASE_STATE)==MESH_LEASE_RELEASED)mesh_bits_set(mesh_plane(m,MESH_FREE),row,1);
  if((was&MESH_LEASE_STATE)!=MESH_LEASE_CLOSED)return 0;
  const uint64_t owner=atomic_load_explicit(&buffer->owner,memory_order_acquire);
  if(!owner || (client && owner!=client))return 0;
  if(mesh_pinned(m,row,buffer->pages/m->block,owner))return 1;
  if(mesh_lease_move(buffer,&was,MESH_LEASE_CLOSED,MESH_LEASE_RELEASED))mesh_bits_set(mesh_plane(m,MESH_FREE),row,1);
  return 0;
}
uint32_t mesh_retired_release(struct hdr *m,uint64_t client){
  uint32_t kept=0;
  for(uint32_t row=0;row<mesh_rows(m);row++)kept+=(uint32_t)mesh_release(m,row,client);
  for(uint32_t word=0;word<mesh_words(m);word++){
    _Atomic uint64_t *plane=&mesh_plane(m,MESH_FREE)[word];
    uint64_t available=atomic_load_explicit(plane,memory_order_acquire);
    while(available){
      const uint32_t row=word*64+(uint32_t)__builtin_ctzll(available);available&=available-1;
      const uint64_t bit=UINT64_C(1)<<(row%64);
      if(atomic_fetch_and_explicit(plane,~bit,memory_order_acq_rel)&bit)mesh_lease_free(m,row,MESH_LEASE_RELEASED);
    }
  }
  return kept;
}
int mesh_lease_return(struct mesh_ctx *c,uint32_t row){
  struct hdr *m=c->M;
  struct mesh_buffer *buffer=row<mesh_rows(m)?&mesh_buffers(m)[row]:NULL;
  const uint32_t rows=buffer?buffer->pages/m->block:0;
  if(!rows || rows>mesh_rows(m)-row || atomic_load_explicit(&buffer->owner,memory_order_acquire)!=c->client ||
     (atomic_load_explicit(&buffer->closed,memory_order_acquire)&MESH_LEASE_STATE)!=MESH_LEASE_LIVE)return EINVAL;
  for(uint32_t r=row+1;r<row+rows;r++)atomic_store_explicit(&mesh_buffers(m)[r].owner,0,memory_order_relaxed);
  mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,rows);
  if(mesh_pinned(m,row,rows,c->client)){mesh_lease_close(buffer,c->client);return 0;}
  mesh_lease_free(m,row,MESH_LEASE_LIVE);
  return 0;
}
/* design/transport.md section 14, "Holding it" */
uint32_t mesh_nodes_reached(struct hdr *m,struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;m && p<m->streams && p<MESH_STREAMS && count<room;p++){
    const struct mesh_stream *x=mesh_stream_at(m,p);
    const int live=atomic_load_explicit(&x->live,memory_order_acquire);
    const uint32_t node=atomic_load_explicit(&x->peer,memory_order_relaxed),cable=atomic_load_explicit(&x->cable,memory_order_relaxed);
    if(!node || !live)continue;
    int seen=0;
    for(uint32_t k=0;k<count;k++)seen|=reach[k].node==node && reach[k].cable==cable;
    if(!seen)reach[count++]=(struct mesh_reach){.node=node,.cable=cable,.bandwidth=atomic_load_explicit(&x->bandwidth,memory_order_relaxed)};
  }
  return count;
}
uint32_t mesh_streams_free(struct hdr *m,uint32_t *node,uint64_t *wire,uint32_t *cable,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;m && p<m->streams && p<MESH_STREAMS && count<room;p++){
    const struct mesh_stream *x=mesh_stream_at(m,p);
    if(!atomic_load_explicit(&x->live,memory_order_acquire) || atomic_load_explicit(&x->owner,memory_order_relaxed))continue;
    node[count]=atomic_load_explicit(&x->peer,memory_order_relaxed);wire[count]=atomic_load_explicit(&x->wire,memory_order_relaxed);
    cable[count++]=atomic_load_explicit(&x->cable,memory_order_relaxed);
  }
  return count;
}
int mesh_stream_take(struct mesh_ctx *context,uint64_t wire,uint32_t *stream){
  if(!context || !context->client || !stream || !wire)return EINVAL;
  struct hdr *m=context->M;
  for(uint32_t p=0;p<m->streams && p<MESH_STREAMS;p++){
    struct mesh_stream *x=mesh_stream_at(m,p);
    if(!atomic_load_explicit(&x->live,memory_order_acquire) || atomic_load_explicit(&x->wire,memory_order_relaxed)!=wire)continue;
    uint64_t vacant=0;
    if(!atomic_compare_exchange_strong_explicit(&x->owner,&vacant,context->client,memory_order_acq_rel,memory_order_acquire))return EBUSY;
    *stream=p;
    return 0;
  }
  return ENOLINK;
}
static struct mesh_stream *stream_held(struct hdr *m,uint32_t stream){
  return stream<m->streams && stream<MESH_STREAMS?mesh_stream_at(m,stream):NULL;
}
void mesh_stream_give(struct mesh_ctx *context,uint32_t stream){
  if(!context || !context->client)return;
  struct mesh_stream *x=stream_held(context->M,stream);
  uint64_t mine=context->client;
  if(x)atomic_compare_exchange_strong(&x->owner,&mine,mine|MESH_STREAM_GIVEN);
}
int mesh_stream_held(struct mesh_ctx *context,uint32_t stream,uint64_t wire){
  const struct mesh_stream *x=context?stream_held(context->M,stream):NULL;
  return x && context->client && atomic_load_explicit(&x->live,memory_order_acquire) &&
         atomic_load_explicit(&x->owner,memory_order_acquire)==context->client && atomic_load_explicit(&x->wire,memory_order_relaxed)==wire;
}
static struct mesh_stream *stream_owned(struct mesh_ctx *context,uint32_t stream,int *error){
  struct mesh_stream *x=context && context->client?stream_held(context->M,stream):NULL;
  *error=!context || !context->client?EINVAL:!x || !atomic_load_explicit(&x->live,memory_order_acquire)?ENOLINK:
         atomic_load_explicit(&x->owner,memory_order_acquire)!=context->client?EPERM:0;
  return *error?NULL:x;
}
int mesh_stream_view(struct mesh_ctx *context,uint32_t stream,struct mesh_stream_view *view){
  int error;
  struct mesh_stream *x=view?stream_owned(context,stream,&error):NULL;
  if(!view)return EINVAL;
  if(!x)return error;
  char *m=(char *)context->M;
  struct mesh_entry *lists=(struct mesh_entry *)(m+x->lists);
  *view=(struct mesh_stream_view){.list={lists,lists+MESH_STREAM_ENTRIES},.presented={&x->sends,&x->lands},.taken={&x->taken[0],&x->taken[1]},
    .ready=&x->ready,.sent=&x->sent,.landed=&x->landed,
    .posted_at=(_Atomic uint64_t *)(m+x->posted_at),.landed_at=(_Atomic uint64_t *)(m+x->landed_at),.posts=x->posts};
  return 0;
}
int mesh_stream_present(struct mesh_ctx *context,uint32_t stream,int land,struct mesh_entry entry){
  int error;
  struct mesh_stream *x=stream_owned(context,stream,&error);
  if(!x)return error;
  const struct hdr *m=context->M;
  if(land<0 || land>1 || !mesh_rows_hold(m,mesh_entry_row(&entry),mesh_entry_pages(&entry)))return EINVAL;
  _Atomic uint64_t *presented=land?&x->lands:&x->sends;
  const uint64_t e=atomic_load_explicit(presented,memory_order_relaxed);
  if(e-atomic_load_explicit(&x->taken[land],memory_order_acquire)>=MESH_STREAM_ENTRIES)return EAGAIN;
  ((struct mesh_entry *)((char *)m+x->lists))[(uint64_t)land*MESH_STREAM_ENTRIES+e%MESH_STREAM_ENTRIES]=entry;
  atomic_store_explicit(presented,e+1,memory_order_release);
  return 0;
}

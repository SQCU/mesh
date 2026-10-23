#include "mesh-call.h"
#include "mesh-observe.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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
int mesh_region(uint64_t *bytes,uint64_t *unleased){
  struct hdr *m;size_t length;
  int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  *bytes=(uint64_t)mesh_arena_pages(m)*m->pgsz;*unleased=(uint64_t)mesh_arena_free(m)*m->pgsz;
  munmap(m,length);return 0;
}
int mesh_map_copy(struct hdr *m,struct mesh_record **out,uint64_t (**names)[2]){
  struct mesh_record *copy=NULL;
  uint64_t (*named)[2]=NULL;
  uint32_t n=0;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->map,memory_order_acquire),at=atomic_load_explicit(&m->map_at,memory_order_acquire);
    n=(uint32_t)at;
    free(copy);free(named);named=NULL;
    if(!(copy=malloc((n?n:1)*sizeof *copy)))break;
    memcpy(copy,mesh_records(m,at),n*sizeof *copy);
    uint64_t total=0;
    for(uint32_t i=0;i<n;i++)total+=copy[i].names;
    const uint64_t room=m->length-((uint64_t)((char *)mesh_map_names(m,at)-(char *)m));
    if(names && total*sizeof *named<=room && (named=malloc((total+1)*sizeof *named)))memcpy(named,mesh_map_names(m,at),total*sizeof *named);
    atomic_thread_fence(memory_order_acquire);
    if(atomic_load_explicit(&m->map,memory_order_relaxed)==before)break;
  }
  *out=copy;
  if(names)*names=named;
  else free(named);
  return copy?(int)n:-ENOMEM;
}
static int mesh_records_read(struct mesh_record **out,uint32_t *node){
  struct hdr *m;size_t length;
  const int error=mesh_open(0,&m,&length,NULL);
  if(error)return -error;
  if(node)*node=m->node;
  const int count=mesh_map_copy(m,out,NULL);
  munmap(m,length);
  return count;
}
int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node){
  struct mesh_record *r=NULL;
  uint32_t own=0;
  const int count=mesh_records_read(&r,&own);
  int reached=count<0?count:0;
  for(int i=0;i<count;i++)if(r[i].node==own){
    reached=(int)(r[i].reached<MESH_REACH?r[i].reached:MESH_REACH);
    for(int k=0;out && k<reached && (uint32_t)k<capacity;k++)out[k]=r[i].reach[k];
  }
  if(node && count>=0)*node=own;
  free(r);
  return reached;
}
/* design/transport.md section 14, "A registration" */
uint32_t mesh_names_find(struct hdr *m,const uint64_t (*name)[2],uint32_t count,uint32_t *node){
  struct mesh_record *r=NULL;
  uint64_t (*named)[2]=NULL;
  const int records=mesh_map_copy(m,&r,&named);
  uint32_t found=0;
  for(uint32_t i=0;i<count;i++){
    node[i]=0;
    for(int k=0,at=0;!node[i] && named && k<records;at+=(int)r[k].names,k++)
      for(uint32_t j=0;j<r[k].names;j++)
        if(named[at+(int)j][0]==name[i][0] && named[at+(int)j][1]==name[i][1]){node[i]=r[k].node;break;}
    found+=node[i]!=0;
  }
  free(r);free(named);
  return found;
}
/* design/transport.md section 3, RECORD; section 15, "The library" */
uint32_t mesh_joined(const struct mesh_record *a,const struct mesh_record *b,uint32_t *cable,uint64_t *bandwidth,uint32_t room){
  uint32_t count=0;
  for(uint32_t i=0;i<a->reached && i<MESH_REACH;i++)
    for(uint32_t j=0;a->reach[i].node==b->node && j<b->reached && j<MESH_REACH;j++){
      const struct mesh_reach *x=a->reach+i,*y=b->reach+j;
      if(y->node!=a->node || x->cable!=y->cable)continue;
      if(count<room){cable[count]=x->cable;bandwidth[count]=x->bandwidth && y->bandwidth?(y->bandwidth<x->bandwidth?y->bandwidth:x->bandwidth):0;}
      count++;
    }
  return count;
}
static int mesh_node_order(const void *a,const void *b){
  uint32_t x=((const struct mesh_record *)a)->node,y=((const struct mesh_record *)b)->node;
  return (x>y)-(x<y);
}
void mesh_whole_free(struct mesh_whole *w){free(w->map.edge);free(w->map.mark);free(w->next);}
int mesh_whole_make(struct mesh_record *r,uint32_t n,struct mesh_whole *w){
  qsort(r,n,sizeof *r,mesh_node_order);
  size_t joins=1;
  for(uint32_t a=0;a<n;a++)for(uint32_t b=a+1;b<n;b++)joins+=mesh_joined(r+a,r+b,NULL,NULL,0);
  *w=(struct mesh_whole){.map={.nodes=n,.edge=malloc(joins*sizeof *w->map.edge),.mark=malloc(((size_t)n+1)*sizeof *w->map.mark)},
    .next=malloc(((size_t)n*n+1)*sizeof *w->next)};
  if(!w->map.edge || !w->map.mark || !w->next)return ENOMEM;
  for(uint32_t a=0;a<n;a++){
    w->map.mark[a]=r[a].node;
    for(uint32_t b=a+1;b<n;b++){
      uint32_t cable[MESH_REACH];
      uint64_t bandwidth[MESH_REACH];
      for(uint32_t c=0,cables=mesh_joined(r+a,r+b,cable,bandwidth,MESH_REACH);c<cables && c<MESH_REACH;c++)
        w->map.edge[w->map.edges++]=(struct mesh_edge){a,b,cable[c],bandwidth[c]/8};
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
  const int count=mesh_records_read(&r,NULL);
  const int result=count<0?count:mesh_records_map(r,(uint32_t)count,members,n,node,map);
  free(r);
  return result;
}
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity){
  struct mesh_record *r=NULL;
  const int count=mesh_records_read(&r,NULL);
  for(int i=0;i<count && (uint32_t)i<capacity;i++){
    out[i].node=r[i].node;
    out[i].region=r[i].region;out[i].memory=r[i].memory;
    memcpy(out[i].model,r[i].model,MESH_MODEL_BYTES);out[i].model[MESH_MODEL_BYTES-1]=0;
    out[i].health=r[i].health;
  }
  free(r);
  return count;
}
/* design/transport.md section 2, the client slots; section 14, "The queue" */
static void mesh_queue_lost(struct hdr *m,struct mesh_client *slot){
  uint32_t tables=mesh_rows(m);
  for(struct mesh_plan *t=mesh_plan_at(m,atomic_load_explicit(&slot->queue,memory_order_acquire));t && tables--;
      t=mesh_plan_at(m,atomic_load_explicit(&t->next,memory_order_acquire)))
    mesh_plan_lost(t);
}
int mesh_plan_balanced(const struct mesh_plan *t){
  for(uint32_t e=0;e<t->sends+t->lands;e++)if(!mesh_entry_balanced(t->entry+e))return 0;
  return 1;
}
void mesh_tuples_lost(struct hdr *m,uint64_t client){
  for(uint32_t s=0;s<MESH_CLIENTS;s++)if(!client || mesh_slot(m,s)==mesh_client_at(m,client))mesh_queue_lost(m,mesh_slot(m,s));
}
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_client *slot=mesh_client_at(m,client);
  uint64_t own=client&~MESH_RETIRING;
  if(!own || (!(client&MESH_RETIRING) && !atomic_compare_exchange_strong(&slot->owner,&own,own|MESH_RETIRING)))return;
  mesh_queue_lost(m,slot);
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==own){
      if(buffer->pages)mesh_lease_close(buffer,own);
      mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,1);
    }
  }
  atomic_store_explicit(&slot->owner,0,memory_order_seq_cst);
  mesh_retired_release(m,own);
}
static int mesh_slot_take(struct mesh_ctx *c,struct hdr *m){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&((UINT64_C(1)<<25)-1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)m+m->client_off)+s;
    uint64_t vacant=0,owner=(generation<<38)|((uint64_t)s<<32)|(uint32_t)getpid();
    if(atomic_load_explicit(&slot->queue,memory_order_acquire) || !atomic_compare_exchange_strong(&slot->owner,&vacant,owner))continue;
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
static uint32_t mesh_scan(uint32_t count,uint32_t align,uint32_t extent,uint32_t begin,uint32_t end,
                          _Atomic uint64_t *own,_Atomic uint64_t *hot){
  for(uint32_t first=(begin+align-1)/align*align;first<=end && count<=end-first;){
    if(extent && (count>extent?first%extent!=0:first/extent!=(first+count-1)/extent)){first=((first/extent+1)*extent+align-1)/align*align;continue;}
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
static uint32_t mesh_allocate(uint32_t count,uint32_t align,uint32_t extent,uint32_t end,_Atomic uint64_t *own,_Atomic uint64_t *hot){
  const uint32_t first=mesh_scan(count,align,extent,0,end,own,hot);
  if(first==MESH_ABSENT)errno=ENOMEM;
  return first;
}
static uint32_t mesh_rows_alloc(struct mesh_ctx *c,uint32_t count){
  if(!count){ errno=EINVAL; return MESH_ABSENT; }
  uint32_t first=mesh_allocate(count,1,0,mesh_rows(c->M),mesh_plane(c->M,MESH_ROW_OWN),mesh_plane(c->M,MESH_ROW_HOT));
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
  return mesh_allocate(pages,align,mesh_extent_pages(c->M),mesh_arena_pages(c->M),bits,bits);
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
/* design/transport.md section 14, "Rows"; design/extents-and-liveness.md, d */
static int mesh_names_rows(const struct hdr *m,const struct mesh_plan *t,uint32_t first,uint32_t count){
  const uint64_t k=atomic_load_explicit(&t->start,memory_order_acquire);
  for(uint32_t e=0;e<t->sends+t->lands;e++){
    const struct mesh_entry *x=t->entry+e;
    if(x->row>=(uint64_t)first+count || first>=x->row+mesh_entry_rows(m,x))continue;
    const uint64_t done=atomic_load_explicit(&x->done,memory_order_seq_cst);
    if((k && done<k) || !mesh_entry_balanced(x))return 1;
  }
  return 0;
}
static int mesh_pinned(struct hdr *m,uint32_t first,uint32_t count,uint64_t client){
  const uint64_t lo=atomic_load_explicit(&mesh_page(m)[first].address,memory_order_relaxed),hi=lo+(uint64_t)count*m->block*m->pgsz;
  uint32_t tables=mesh_rows(m);
  for(const struct mesh_plan *t=mesh_plan_at(m,atomic_load_explicit(&mesh_client_at(m,client)->queue,memory_order_acquire));t && tables--;
      t=mesh_plan_at(m,atomic_load_explicit(&t->next,memory_order_acquire))){
    const uint64_t at=(uint64_t)((const char *)t-(const char *)m);
    if((at>=lo && at<hi) || mesh_names_rows(m,t,first,count))return 1;
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
/* design/transport.md section 14, "The queue" and "Walks" */
void mesh_queue_link(struct mesh_ctx *context,struct mesh_plan *table){
  struct mesh_client *slot=mesh_client_at(context->M,context->client);
  atomic_store_explicit(&table->next,atomic_load_explicit(&slot->queue,memory_order_relaxed),memory_order_relaxed);
  atomic_store_explicit(&slot->queue,(uint64_t)((char *)table-(char *)context->M),memory_order_release);
}
int mesh_queue_unlink(struct mesh_ctx *context,struct mesh_plan *table){
  struct hdr *m=context->M;
  const uint64_t at=(uint64_t)((char *)table-(char *)m);
  _Atomic uint64_t *link=&mesh_client_at(m,context->client)->queue;
  for(uint32_t tables=mesh_rows(m);tables-- && atomic_load_explicit(link,memory_order_relaxed)!=at;){
    struct mesh_plan *t=mesh_plan_at(m,atomic_load_explicit(link,memory_order_relaxed));
    if(!t)return ENOENT;
    link=&t->next;
  }
  if(atomic_load_explicit(link,memory_order_relaxed)!=at)return ENOENT;
  atomic_store_explicit(link,atomic_load_explicit(&table->next,memory_order_relaxed),memory_order_seq_cst);
  return 0;
}
void mesh_walks(struct hdr *m,struct mesh_walk *walk){
  const struct mesh_bridge_pages *own=mesh_bridge_pages(m);
  for(uint32_t p=0;p<=MESH_PORTS;p++)for(int way=0;way<2;way++)walk->word[p][way]=atomic_load_explicit(&own->walking[p][way],memory_order_seq_cst);
}
int mesh_walked(struct hdr *m,const struct mesh_walk *walk){
  const struct mesh_bridge_pages *own=mesh_bridge_pages(m);
  for(uint32_t p=0;p<=MESH_PORTS;p++)for(int way=0;way<2;way++)
    if((walk->word[p][way]&1) && atomic_load_explicit(&own->walking[p][way],memory_order_acquire)==walk->word[p][way])return 0;
  return 1;
}

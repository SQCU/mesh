
#include "mesh.h"
#include "mesh-wire.h"
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <signal.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

#ifndef MESH_ATTACH_ATTEMPTS
#define MESH_ATTACH_ATTEMPTS 3000
#endif

static struct mesh_ctx CTX0={.last=-1};
struct mesh_ctx *mesh_context(void){ return &CTX0; }
void mesh_receiver(struct mesh_ctx *context,mesh_receive_fn receiver,void *capture){
  context->receiver=receiver; context->receiver_capture=capture;
}

static const char *rname(const char *name){
  if(!name) name=getenv("MESH_REGION");
  return name?name:MESH_NAME; }

static uint64_t identity(void *p, struct stat *s){
#ifdef __APPLE__
  (void)s;
  mach_vm_address_t address=(mach_vm_address_t)p; mach_vm_size_t size=0;
  natural_t depth=0; vm_region_submap_info_data_64_t info={0};
  mach_msg_type_number_t count=VM_REGION_SUBMAP_INFO_COUNT_64;
  kern_return_t rc=mach_vm_region_recurse(mach_task_self(),&address,&size,&depth,
                                         (vm_region_recurse_info_t)&info,&count);
  if(rc) fprintf(stderr,"mesh region identity: %s\n",mach_error_string(rc));
  return rc?0:info.object_id_full;
#else
  (void)p; return (uint64_t)s->st_ino;
#endif
}

static int mesh_attach_attempts(struct mesh_ctx *c, const char *name, int attempts){
  if(c->M){ if(c->mapping_pinned<0 || c->detaching){ errno=ESTALE; return -1; } return 0; }
  name=name?name:(c->name?c->name:rname(0));
  for(int t=0;t<attempts && !c->M;t++){
    int f=shm_open(name,O_RDWR,MESH_MODE);
    if(f>=0){
      struct stat s;
      if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
        struct hdr *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
        if(b!=MAP_FAILED){
          if(b->magic==MESH_MAGIC && b->version==MESH_VERSION && atomic_load(&b->phase)<MESH_STOPPING){
            uint64_t vacant=0;
            if(!atomic_compare_exchange_strong_explicit(&b->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)){
              munmap(b,(size_t)s.st_size); errno=EBUSY;
            } else {
              c->M=b; c->len=(size_t)s.st_size; c->ino=identity(b,&s);
              if(!c->name){ c->name=strdup(name); c->name_owned=1; } } }
          else munmap(b,(size_t)s.st_size); } }
      close(f); }
    if(!c->M && t+1<attempts) usleep(10000); }
  if(!c->M) return -1;
  c->arena=mesh_data(c->M,c->M->pool); c->busy=calloc(c->M->arena,1);
  if(!c->busy){ atomic_store(&c->M->client,0); munmap(c->M,c->len); c->M=0; c->arena=0; c->len=0; return -1; }
  c->cursor=0; c->inflight=0; c->last=-1; c->sub=c->ack=0; c->idle=0;
  return 0; }

int mesh_attach(struct mesh_ctx *c, const char *name){ return mesh_attach_attempts(c,name,MESH_ATTACH_ATTEMPTS); }
int mesh_try_attach(struct mesh_ctx *c, const char *name){ return mesh_attach_attempts(c,name,1); }
int mesh_link_reset(struct mesh_ctx *c,size_t port){
  if(!c->M || port>=atomic_load_explicit(&c->M->port_count,memory_order_acquire)) return EINVAL;
  atomic_store_explicit(&mesh_ports(c->M)[port].reset_request,1,memory_order_release); return 0;
}

static int stale(struct mesh_ctx *c){
  uint64_t pid=atomic_load(&c->M->bridge_pid);
  if(atomic_load(&c->M->phase)>=MESH_STOPPING || (pid && kill((pid_t)pid,0)<0 && errno==ESRCH)) return 1;
  if(c->mapping_pinned) return 0;
  struct stat s; int f=shm_open(c->name?c->name:rname(0),O_RDWR,MESH_MODE);
  if(f<0) return errno==ENOENT;
  int gone=0;
  if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
    void *p=mmap(NULL,sizeof(struct hdr),PROT_READ,MAP_SHARED,f,0);
    if(p!=MAP_FAILED){
      uint64_t id=identity(p,&s);
      uint64_t current=c->ino?c->ino:identity(c->M,&s);
      gone=(id && current && id!=current) || atomic_load(&((struct hdr*)p)->bridge_pid)!=pid;
      if(!gone && current) c->ino=current;
      munmap(p,sizeof(struct hdr)); } }
  close(f); return gone; }

static void mesh_retire(struct mesh_ctx *c, struct mstream **v, int k){
  for(int i=0;i<k;i++){ struct mstream *s=v[i];
    if(s->st==MS_RUN) s->st=MS_FAIL;
    if(!s->retain_seen){ free(s->seen); s->seen=0; } }
  free(c->busy); c->busy=0; munmap(c->M,c->len); c->M=0; c->arena=0; }

static void reattach(struct mesh_ctx *c, struct mstream **v, int k){
  mesh_retire(c,v,k);
  if(!mesh_try_attach(c,0)) fprintf(stderr,"mesh reattached %s object %llu\n",c->name,(unsigned long long)c->ino); }

static int refresh(struct mesh_ctx *c, struct mstream **v, int k){
  if(c->mapping_pinned && atomic_load(&c->M->bridge_pid) && atomic_load(&c->M->phase)>=MESH_STOPPING) c->mapping_pinned=-1;
  if(c->mapping_pinned<0){
    for(int i=0;i<k;i++){ v[i]->st=MS_FAIL; mesh_stream_changed(v[i]); }
    errno=ESTALE;
    return 1; }
  if(++c->idle<600) return 0;
  c->idle=0;
  if(!stale(c)) return 0;
  if(c->mapping_pinned){
    c->mapping_pinned=-1;
    for(int i=0;i<k;i++){ v[i]->st=MS_FAIL; mesh_stream_changed(v[i]); }
    errno=ESTALE;
    return 1; }
  reattach(c,v,k); return 1; }

static size_t cwrite(struct mesh_ctx *c, const void *p, size_t nbytes, int node){
  struct hdr *M=c->M;
  uint32_t s=(uint32_t)(((const unsigned char*)p-c->arena)/M->pgsz); size_t done=0;
  while(done<nbytes && s<M->arena){
    uint32_t u=mesh_pay(M);
    if(c->busy && (c->busy[s]&1)) break;
    struct desc d={.page=M->pool+s,.bytes=nbytes-done<u?(uint32_t)(nbytes-done):u,.node=(uint16_t)node};
    if(push(M,SUB,&d)) break;
    if(c->busy){ c->busy[s]|=1; c->inflight++; }
    done+=d.bytes; s++; }
  return done; }

static size_t cread(struct mesh_ctx *c, void **p, int *from){
  struct hdr *M=c->M;
  if(c->last>=0){ struct desc r={.page=(uint32_t)c->last};
    if(push(M,REL,&r)) return 0;
    c->last=-1; }
  struct desc d; if(pop(M,CMP,&d)) return 0;
  c->last=(int)d.page;
  if(p) *p=mesh_data(M,d.page);
  if(from) *from=d.node;
  return d.bytes; }

static void reclaim(struct mesh_ctx *c){
  struct desc d;
  while(!pop(c->M,ACK,&d)){
    if(c->busy && d.page>=c->M->pool && d.page<c->M->pool+c->M->arena && (c->busy[d.page-c->M->pool]&1)){
      c->busy[d.page-c->M->pool]&=2; c->inflight--; }
    c->ack++; } }

static unsigned char *credit(struct mesh_ctx *c){
  if(c->inflight>=c->M->arena) return 0;
  for(size_t i=0;i<c->M->arena;i++){
    size_t s=(c->cursor+i)%c->M->arena;
    if(!c->busy || !c->busy[s]){ c->cursor=(s+1)%c->M->arena;
      return c->arena+s*c->M->pgsz; } }
  return 0; }

static int pending_grow(struct mesh_ctx *c, size_t want){
  size_t cap=c->pending_capacity?c->pending_capacity:1;
  while(cap<want){
    if(cap>SIZE_MAX/2){ cap=want; break; }
    cap*=2; }
  size_t u=mesh_pay(c->M);
  if(!u || cap>SIZE_MAX/u || cap>SIZE_MAX/sizeof *c->pending_nodes || cap>SIZE_MAX/sizeof *c->pending_bytes) return -1;
  unsigned char *p=realloc(c->pending,cap*u);
  if(!p) return -1;
  c->pending=p;
  int *nodes=realloc(c->pending_nodes,cap*sizeof *nodes);
  if(!nodes) return -1;
  c->pending_nodes=nodes;
  uint32_t *bytes=realloc(c->pending_bytes,cap*sizeof *bytes);
  if(!bytes) return -1;
  c->pending_bytes=bytes; c->pending_capacity=cap; return 0; }

static size_t pending_flush(struct mesh_ctx *c){
  reclaim(c); size_t before=c->pending_count;
  while(c->pending_count){
    unsigned char *q=credit(c); if(!q) break;
    size_t at=c->pending_head, u=mesh_pay(c->M); uint32_t bytes=c->pending_bytes[at];
    memcpy(q,c->pending+at*u,bytes);
    if(cwrite(c,q,bytes,c->pending_nodes[at])!=bytes) break;
    c->sub++; c->pending_head++; c->pending_count--; }
  if(!c->pending_count) c->pending_head=0;
  return before-c->pending_count; }

static size_t pending_push(struct mesh_ctx *c, const void *p, size_t stride,
                           size_t bytes, size_t nslots, int node){
  size_t u=mesh_pay(c->M);
  if(nslots>SIZE_MAX-c->pending_head-c->pending_count) return 0;
  if(c->pending_head && c->pending_head+c->pending_count+nslots>c->pending_capacity){
    memmove(c->pending,c->pending+c->pending_head*u,c->pending_count*u);
    memmove(c->pending_nodes,c->pending_nodes+c->pending_head,c->pending_count*sizeof *c->pending_nodes);
    memmove(c->pending_bytes,c->pending_bytes+c->pending_head,c->pending_count*sizeof *c->pending_bytes);
    c->pending_head=0; }
  size_t want=c->pending_head+c->pending_count+nslots;
  if(want>c->pending_capacity && pending_grow(c,want)) return 0;
  for(size_t i=0;i<nslots;i++){
    size_t at=c->pending_head+c->pending_count+i;
    memset(c->pending+at*u,0,u); memcpy(c->pending+at*u,(const char*)p+i*stride,bytes);
    c->pending_nodes[at]=node; c->pending_bytes[at]=(uint32_t)bytes; }
  c->pending_count+=nslots; pending_flush(c); return nslots; }

static int stream_ctl(struct mesh_ctx *c, const struct mstream *s, uint32_t kind, uint64_t offset, const void *body, size_t bytes){
  unsigned char *q=credit(c); if(!q) return -1;
  size_t header=mesh_stream_header(s);
  if(bytes) memcpy(q+header,body,bytes);
  struct mesh_frame frame={.h={offset,s->sid,kind},.epoch=s->scope.epoch,.source=(uint16_t)c->M->node,.target=(uint16_t)s->node};
  size_t n=mesh_frame_encode(q,&frame,bytes);
  if(cwrite(c,q,n,s->node)!=n) return -1;
  c->sub++; return 0;
}
static void stream_retry(struct mstream *s,uint64_t now_ns,uint64_t floor_ns){
  s->retry_ns=s->retry_ns?s->retry_ns<500000000?s->retry_ns*2:1000000000:floor_ns?floor_ns:1000000;
  s->fin_after_ns=now_ns+s->retry_ns;
}
static size_t stream_page(struct mesh_ctx *c,struct mstream *s,size_t offset){
  size_t header=mesh_stream_header(s), u=mesh_stream_payload(c->M,s);
  uint32_t len=s->n-offset<u?(uint32_t)(s->n-offset):(uint32_t)u;
  unsigned char *q=s->stride?(unsigned char*)s->src+(offset/u)*s->stride-header:credit(c);
  if(!q || (s->stride && (c->busy[(q-c->arena)/c->M->pgsz]&1))) return 0;
  if(!s->stride) memcpy(q+header,s->src+offset,len);
  struct mesh_frame frame={.h={offset,s->sid,K_DATA},.epoch=s->scope.epoch,.source=(uint16_t)c->M->node,.target=(uint16_t)s->node};
  size_t bytes=mesh_frame_encode(q,&frame,len);
  if(cwrite(c,q,bytes,s->node)!=bytes) return 0;
  c->sub++; return len;
}
int mesh_stream_push_page(struct mesh_ctx *c,struct mstream *s,size_t page,size_t window){
  struct mesh_page_bits *word=&s->work[page/64]; uint64_t bit=UINT64_C(1)<<(page%64);
  if(s->st!=MS_RUN || (mesh_epoch_set(s->scope.epoch) && !s->agreed) || c->inflight>=window || !(__atomic_load_n(&word->pending,__ATOMIC_ACQUIRE)&bit)) return 0;
  size_t bytes=stream_page(c,s,page*mesh_stream_payload(c->M,s));
  if(!bytes) return 0;
  __atomic_fetch_and(&word->pending,~bit,__ATOMIC_RELEASE);
  s->off+=bytes; s->fin_after_ns=0; s->retry_ns=0; return 1;
}
ptrdiff_t mesh_stream_push_pages(struct mesh_ctx *c,struct mstream *s,size_t window){
  size_t before=s->off;
  size_t words=mesh_page_words(s->nb);
  for(size_t examined=0;s->off<atomic_load_explicit(&s->available,memory_order_acquire) && c->inflight<window && examined<words;examined++){
    size_t word=s->scan_word;
    uint64_t pending=__atomic_load_n(&s->work[word].pending,__ATOMIC_ACQUIRE);
    for(;pending && c->inflight<window;pending&=pending-1)
      mesh_stream_push_page(c,s,word*64+(size_t)__builtin_ctzll(pending),window);
    s->scan_word=word+1==words?0:word+1;
  }
  size_t bytes=s->off-before, payload=mesh_stream_payload(c->M,s);
  return (ptrdiff_t)(bytes/payload+(bytes%payload!=0));
}

void mesh_yell_start(struct mesh_ctx *c, struct mstream *s,
                     const void *p, size_t n, int node, uint32_t sid){
  *s=(struct mstream){.src=p,.n=n,.node=node,.sid=sid,.st=n?MS_RUN:MS_DONE,.available=n}; (void)c; }

int mesh_stream_receive(struct mesh_ctx *c, struct mstream *s, uint32_t *pages){
  if(mesh_attach(c,0)) return -1;
  size_t u=mesh_stream_payload(c->M,s);
  s->nb=s->n/u+(s->n%u!=0); s->pages=pages;
  s->seen=pages?NULL:calloc((s->nb+7)/8,1);
  if(pages) for(size_t i=0;i<s->nb;i++) pages[i]=UINT32_MAX;
  return pages||s->seen?0:-1;
}
static int stream_missing(const struct mstream *s,size_t page){
  return s->pages?s->pages[page]==UINT32_MAX:s->seen && !(s->seen[page>>3]>>(page&7)&1);
}
int mesh_lissen_start(struct mesh_ctx *c, struct mstream *s, void *p, size_t n, uint32_t sid){
  *s=(struct mstream){.buf=p,.n=n,.sid=sid,.rx=1,.st=n?MS_RUN:MS_DONE,.node=-1};
  return mesh_stream_receive(c,s,NULL);
}
void *mesh_stream_lease(struct mesh_ctx *c, struct mstream *s){
  if(mesh_attach(c,0)) return NULL;
  reclaim(c);
  size_t header=mesh_stream_header(s), u=mesh_stream_payload(c->M,s), pages=s->n/u+(s->n%u!=0), run=0;
  for(size_t i=0;pages && i<c->M->arena;i++){
    run=c->busy[i]?0:run+1;
    if(run<pages) continue;
    size_t first=i+1-pages;
    memset(c->busy+first,2,pages);
    void *base=c->arena+first*c->M->pgsz-sizeof(struct wire);
    s->src=(char*)base+sizeof(struct wire)+header;
    s->stride=c->M->pgsz; s->nb=pages; atomic_store(&s->available,0);
    return base;
  }
  return NULL;
}
void *mesh_yell_view(struct mesh_ctx *c, struct mstream *s, size_t n, int node, uint32_t sid){
  mesh_yell_start(c,s,NULL,n,node,sid); return mesh_stream_lease(c,s);
}

int mesh_stream_idle(struct mesh_ctx *c, const struct mstream *s){
  reclaim(c);
  if(!c->inflight || !s->stride) return 1;
  size_t first=((const unsigned char*)s->src-mesh_stream_header(s)-c->arena)/c->M->pgsz;
  for(size_t i=0;i<s->nb;i++) if(c->busy[first+i]&1) return 0;
  return 1;
}
int mesh_lissen_view(struct mesh_ctx *c, struct mstream *s, uint32_t *pages, size_t n, uint32_t sid){
  *s=(struct mstream){.n=n,.sid=sid,.rx=1,.st=n?MS_RUN:MS_DONE,.node=-1};
  return mesh_stream_receive(c,s,pages);
}

size_t mesh_release_view(struct mesh_ctx *c, struct mstream *s){
  size_t pending=0;
  if(s->stride){
    size_t first=((const unsigned char*)s->src-mesh_stream_header(s)-c->arena)/c->M->pgsz;
    for(size_t i=0;i<s->nb;i++) c->busy[first+i]&=1;
    s->stride=0; }
  if(s->pages){
    struct ring *ring=&c->M->r[REL];
    uint64_t head=atomic_load_explicit(&ring->head,memory_order_relaxed);
    uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_acquire);
    size_t count=s->arrivals?s->done/mesh_stream_payload(c->M,s)+(s->done%mesh_stream_payload(c->M,s)!=0):s->nb;
    for(size_t i=0;i<count;i++){
      size_t page=s->arrivals?s->arrivals[i]:i;
      if(s->pages[page]==UINT32_MAX) continue;
      if(c->mapping_pinned>=0 && head-tail>=MESH_RING){ pending++; continue; }
      if(c->mapping_pinned>=0) *slot(c->M,REL,head++)=(struct desc){.page=s->pages[page]};
      s->pages[page]=UINT32_MAX;
    }
    atomic_store_explicit(&ring->head,head,memory_order_release);
  }
  if(!pending) s->pages=NULL;
  return pending; }

static size_t stream_bucket(uint32_t sid,int receive,size_t mask){
  uint64_t key=((uint64_t)sid<<1)^(unsigned)receive;
  key^=key>>33; key*=0xff51afd7ed558ccdULL; key^=key>>33;
  return (size_t)key&mask;
}
static void stream_index(struct mesh_ctx *c,struct mstream **v,size_t count){
  memset(c->stream_index,0,c->stream_index_capacity*sizeof *c->stream_index);
  for(size_t i=0;i<count;i++){
    size_t bucket=stream_bucket(v[i]->sid,v[i]->rx,c->stream_index_capacity-1);
    while(c->stream_index[bucket]) bucket=(bucket+1)&(c->stream_index_capacity-1);
    c->stream_index[bucket]=(uint32_t)i+1;
  }
}

int mesh_stream_reserve(struct mesh_ctx *c, size_t count){
  if(count>SIZE_MAX/4/sizeof(uint32_t)) return EOVERFLOW;
  size_t capacity=1;
  while(capacity<count*2) capacity*=2;
  if(capacity>c->stream_index_capacity){
    uint32_t *index=realloc(c->stream_index,capacity*sizeof *index);
    if(!index) return ENOMEM;
    c->stream_index=index; c->stream_index_capacity=capacity;
    stream_index(c,c->streams,c->stream_count);
  }
  return 0;
}
int mesh_stream_register(struct mesh_ctx *c,struct mstream **v,size_t count){
  int error=mesh_stream_reserve(c,count); if(error) return error;
  stream_index(c,v,count);
  c->streams=v; c->stream_count=count; c->stream_cursor=0;
  return 0;
}
int mesh_stream_conflict(const struct mesh_ctx *c,const struct mstream *stream,struct mesh_epoch epoch){
  size_t mask=c->stream_index_capacity-1;
  size_t bucket=stream_bucket(stream->sid,stream->rx,mask);
  for(size_t probe=0;probe<c->stream_index_capacity;probe++){
    uint32_t index=c->stream_index[(bucket+probe)&mask]; if(!index) break;
    const struct mstream *other=c->streams[index-1];
    if(other!=stream && other->sid==stream->sid && other->rx==stream->rx && other->node==stream->node &&
       mesh_epoch_equal(other->scope.epoch,epoch)) return 1;
  }
  return 0;
}
int mesh_poll_streams(struct mesh_ctx *c, struct mstream **v, int k, uint64_t *now_ns){
  if(mesh_try_attach(c,0)){
    if(errno==ESTALE) for(int i=0;i<k;i++){ v[i]->st=MS_FAIL; v[i]->error=ESTALE; mesh_stream_changed(v[i]); }
    return 0; }
  reclaim(c);
  if(refresh(c,v,k)) return 0;
  size_t capacity=c->stream_index_capacity;
  uint32_t *index=v==c->streams && (size_t)k==c->stream_count?c->stream_index:NULL;
  struct timespec clock; clock_gettime(CLOCK_MONOTONIC,&clock);
  *now_ns=(uint64_t)clock.tv_sec*1000000000u+clock.tv_nsec;
  for(int turn=0;turn<256;turn++){
    void *q; int from; size_t b=cread(c,&q,&from);
    if(!b) break;
    if(c->receiver && c->receiver(c->receiver_capture,q,b,from)) continue;
    if(b<MESH_OFF){ c->integrity_failures++; continue; }
    struct mesh_frame frame; size_t header=mesh_frame_decode(q,b,&frame);
    if(!header){ c->integrity_failures++; continue; }
    if(header>MESH_OFF && (frame.source!=from || frame.target!=c->M->node)){ c->integrity_failures++; continue; }
    struct shdr sh=frame.h; struct mstream *s=NULL;
    int receive=mesh_frame_receive(sh.k);
    if(header>MESH_OFF && sh.k!=K_DATA) for(int i=0;i<k;i++)
      if(!v[i]->rx && !v[i]->parked && !v[i]->agreed && v[i]->st==MS_RUN && v[i]->node==from){ v[i]->fin_after_ns=0; v[i]->retry_ns=0; }
    size_t bucket=stream_bucket(sh.sid,receive,capacity-1);
    for(size_t probe=0;probe<(index?capacity:(size_t)k);probe++){
      int i=index?(int)index[(bucket+probe)&(capacity-1)]-1:(int)probe;
      if(i<0) break;
      if(!v[i]->parked && v[i]->sid==sh.sid && v[i]->rx==receive &&
         (v[i]->node<0 || v[i]->node==from) && mesh_epoch_equal(v[i]->scope.epoch,frame.epoch)) s=v[i];
    }
    if(!s){
      int kind=sh.k==K_FIN&&header>MESH_OFF?K_OK:mesh_frame_reply(sh.k);
      if(kind==K_CLOSED || kind==K_OK){
        struct mstream reply={.sid=sh.sid,.node=from,.scope.epoch=frame.epoch};
        stream_ctl(c,&reply,(uint32_t)kind,0,NULL,0);
      } else if(header>MESH_OFF) c->stale_frames++;
      continue;
    }
    if(sh.k==K_ABORT_RX || sh.k==K_ABORT_TX){
      if(!s->error) s->error=sh.off>0&&sh.off<=INT32_MAX?(int)sh.off:ECANCELED;
      s->st=MS_FAIL; s->abort_ack=1;
      stream_ctl(c,s,(uint32_t)mesh_frame_reply(sh.k),0,NULL,0); mesh_stream_changed(s); continue;
    }
    if(sh.k==K_ABORTED_RX || sh.k==K_ABORTED_TX){ if(s->error){ s->abort_ack=1; mesh_stream_changed(s); } continue; }
    if(s->st==MS_FAIL) continue;
    size_t u=mesh_stream_payload(c->M,s);
    if(sh.k==K_OPEN && header>MESH_OFF){
      uint64_t logical_offset=0,chunk=0;
      size_t agreement=sizeof s->scope.plan+sizeof logical_offset;
      if(b==header+agreement || b==header+agreement+sizeof chunk){
        memcpy(&logical_offset,(char*)q+header+sizeof s->scope.plan,sizeof logical_offset);
        if(b==header+agreement+sizeof chunk) memcpy(&chunk,(char*)q+header+agreement,sizeof chunk);
      }
      uint64_t expected_chunk=s->chunk|((uint64_t)s->resident<<63);
      if(b!=header+agreement+(expected_chunk?sizeof chunk:0) || chunk!=expected_chunk || sh.off!=s->n || logical_offset!=s->logical_offset ||
         memcmp((char*)q+header,s->scope.plan,sizeof s->scope.plan)){
        s->error=EPROTO; s->st=MS_FAIL; s->fin_after_ns=0; s->retry_ns=0;
      } else { s->agreed=1; stream_ctl(c,s,K_READY,0,NULL,0); }
    }
    else if(sh.k==K_READY && header>MESH_OFF){ s->agreed=1; s->fin_after_ns=0; s->retry_ns=0; }
    else if(header>MESH_OFF && !s->agreed) continue;
    else if(sh.k==K_DATA){
      size_t len=b-header, ci=sh.off/u;
      if(s->st==MS_RUN && sh.off<s->n && sh.off%u==0 && len==(s->n-sh.off<u?s->n-sh.off:u) &&
         ci<s->nb && stream_missing(s,ci)){
        if(s->pages){ __atomic_store_n(&s->pages[ci],(uint32_t)c->last,__ATOMIC_RELEASE); c->last=-1; }
        else { s->seen[ci>>3]|=(unsigned char)(1u<<(ci&7)); memcpy(s->buf+sh.off,(char*)q+header,len); }
        if(s->arrivals) s->arrivals[s->done/u+(s->done%u!=0)]=(uint32_t)ci;
        __atomic_store_n(&s->done,s->done+len,__ATOMIC_RELEASE);
        if(s->resident && s->done==s->n){ s->st=MS_DONE; s->closed=1; }
      }
    }
    else if(sh.k==K_FIN && sh.off==s->n){
      s->node=from;
      if(s->done>=s->n){ stream_ctl(c,s,K_OK,0,NULL,0); s->st=MS_DONE; }
      else { while(s->hole<s->nb && !stream_missing(s,s->hole)) s->hole++;
             stream_ctl(c,s,K_REQ,s->hole*(uint64_t)u,NULL,0); }
    }
    else if(sh.k==K_REQ){ if(s->st==MS_RUN && sh.off<s->n && sh.off%u==0){
      if(s->paged){
        size_t page=sh.off/u; uint64_t bit=UINT64_C(1)<<(page%64);
        if(mesh_stream_published(s,page) && !(__atomic_fetch_or(&s->work[page/64].pending,bit,__ATOMIC_RELEASE)&bit))
          s->off-=s->n-sh.off<u?s->n-sh.off:u;
      } else s->off=sh.off;
      s->fin_after_ns=0; s->retry_ns=0;
    } }
    else if(sh.k==K_OK && s->st==MS_RUN && s->off==s->n){
      __atomic_store_n(&s->done,s->n,__ATOMIC_RELEASE); s->st=MS_DONE; s->fin_after_ns=0; s->retry_ns=0;
      if(s->resident) s->closed=1;
    }
    else if(sh.k==K_CLOSE && s->st==MS_DONE){ stream_ctl(c,s,K_CLOSED,0,NULL,0); s->closed=1; }
    else if(sh.k==K_CLOSED && s->st==MS_DONE) s->closed=1;
    if(sh.k!=K_DATA || s->done==s->n || s->st!=MS_RUN) mesh_stream_changed(s);
    if(s->st!=MS_RUN && s->seen && !s->retain_seen){ free(s->seen); s->seen=NULL; }
  }
  return 1;
}

int mesh_progress_stream(struct mesh_ctx *c, struct mstream *s, size_t window, uint64_t now_ns){
  if(s->parked) return 0;
  size_t header=mesh_stream_header(s), u=mesh_stream_payload(c->M,s);
  if(s->error && header>MESH_OFF && !s->abort_ack){
    if(now_ns>=s->fin_after_ns && !stream_ctl(c,s,s->rx?K_ABORT_TX:K_ABORT_RX,(uint64_t)s->error,NULL,0)) stream_retry(s,now_ns,0);
    return 1;
  }
  if(!s->rx && s->st==MS_DONE && s->closing && !s->closed){
    if(now_ns>=s->fin_after_ns && !stream_ctl(c,s,K_CLOSE,0,NULL,0)) stream_retry(s,now_ns,0);
    return 1;
  }
  if(s->rx || s->st!=MS_RUN) return 0;
  if(header>MESH_OFF && !s->agreed){
    if(now_ns>=s->fin_after_ns){
      unsigned char body[48]; uint64_t logical_offset=s->logical_offset,chunk=s->chunk|((uint64_t)s->resident<<63);
      memcpy(body,s->scope.plan,32); memcpy(body+32,&logical_offset,8); memcpy(body+40,&chunk,8);
      if(!stream_ctl(c,s,K_OPEN,s->n,body,chunk?48:40)) stream_retry(s,now_ns,s->open_retry_ns);
    }
    return 1;
  }
  size_t was=s->off;
  if(s->paged) mesh_stream_push_pages(c,s,window);
  else while(s->off<s->n && c->inflight<window){
    uint32_t len=s->n-s->off<u?(uint32_t)(s->n-s->off):(uint32_t)u;
    if(s->off+len>atomic_load_explicit(&s->available,memory_order_acquire)) break;
    if(!stream_page(c,s,s->off)) break;
    s->off+=len;
  }
  if(s->off>was){ s->fin_after_ns=0; s->retry_ns=0; }
  else if(s->off>=s->n && now_ns>=s->fin_after_ns){
    if(!stream_ctl(c,s,K_FIN,s->n,NULL,0)) stream_retry(s,now_ns,0);
  }
  return s->off==s->n || s->off<atomic_load_explicit(&s->available,memory_order_acquire);
}

int mesh_turn_window(struct mesh_ctx *c, struct mstream **v, int k, size_t window){
  uint64_t now_ns;
  if(!mesh_poll_streams(c,v,k,&now_ns)) return 0;
  int ndone=0;
  size_t start=k?c->stream_cursor%(size_t)k:0;
  for(int i=0;i<k;i++){
    struct mstream *s=v[(start+(size_t)i)%(size_t)k];
    ndone+=!s->parked && s->st==MS_DONE;
    mesh_progress_stream(c,s,window,now_ns);
  }
  c->stream_cursor=start+1;
  return ndone;
}

int mesh_turn(struct mesh_ctx *c, struct mstream **v, int k){
  return mesh_turn_window(c,v,k,SIZE_MAX); }

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  if(atomic_load(&c->executor_attached)) return EBUSY;
  if(c->mapping_pinned<0 || stale(c)) goto detached;
  if(c->detaching){
    if(atomic_load_explicit(&c->M->client,memory_order_acquire)==MESH_CLIENT_DETACH) return EBUSY;
    goto detached;
  }
  reclaim(c);
  if(c->inflight || c->pending_count) return EBUSY;
  for(size_t i=0;i<c->M->arena;i++) if(c->busy[i]) return EBUSY;
  for(int i=0;i<MESH_RING;i++){
    void *payload; int from;
    if(!cread(c,&payload,&from)) break;
  }
  if(c->last>=0 || atomic_load(&c->M->r[CMP].head)!=atomic_load(&c->M->r[CMP].tail)) return EBUSY;
  uint64_t owner=(uint64_t)getpid();
  if(!atomic_compare_exchange_strong_explicit(&c->M->client,&owner,MESH_CLIENT_DETACH,memory_order_acq_rel,memory_order_acquire)) return EOWNERDEAD;
  c->detaching=1; return EBUSY;
detached:
  munmap(c->M,c->len); free(c->busy); free(c->pending); free(c->pending_nodes); free(c->pending_bytes); free(c->stream_index);
  if(c->name_owned) free(c->name);
  *c=(struct mesh_ctx){.last=-1}; return 0;
}

int mesh_scatter(struct mesh_ctx *c, struct mstream *ss, const void *p, size_t n,
                 const int *nodes, int k, uint32_t sid0){
  size_t sh=(n+k-1)/k;
  for(int i=0;i<k;i++){ size_t o=(size_t)i*sh, l=o<n?(n-o<sh?n-o:sh):0;
    mesh_yell_start(c,&ss[i],(const char*)p+o,l,nodes[i],sid0+i); }
  return k; }

int mesh_gather(struct mesh_ctx *c, struct mstream *ss, void *p, size_t n,
                int k, uint32_t sid0){
  size_t sh=(n+k-1)/k;
  for(int i=0;i<k;i++){ size_t o=(size_t)i*sh, l=o<n?(n-o<sh?n-o:sh):0;
    if(mesh_lissen_start(c,&ss[i],(char*)p+o,l,sid0+i)) return -1; }
  return k; }

void *mesh_open(size_t *ns, size_t *sp, size_t *up){
  if(mesh_attach(&CTX0,0)) return NULL;
  return mesh_try_open(ns,sp,up); }

void *mesh_try_open(size_t *ns, size_t *sp, size_t *up){
  if(mesh_try_attach(&CTX0,0)) return NULL;
  if(ns) *ns=CTX0.M->arena;
  if(sp) *sp=CTX0.M->pgsz; if(up) *up=mesh_pay(CTX0.M);
  return CTX0.arena; }

int mesh_close(void){ return mesh_detach(&CTX0); }

size_t mesh_write(const void *p, size_t nbytes, int node){
  if(mesh_try_attach(&CTX0,0)) return 0;
  reclaim(&CTX0);
  return cwrite(&CTX0,p,nbytes,node); }

size_t mesh_write_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node){
  if(mesh_try_attach(&CTX0,0) || !bytes || bytes>mesh_pay(CTX0.M)) return 0;
  reclaim(&CTX0); size_t done=0;
  while(done<nslots){
    unsigned char *q=credit(&CTX0); if(!q) break;
    memcpy(q,(const unsigned char*)p+done*stride,bytes);
    if(cwrite(&CTX0,q,bytes,node)!=bytes) break;
    CTX0.sub++; done++; }
  return done; }

size_t mesh_queue_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node){
  if(mesh_try_attach(&CTX0,0) || !bytes || bytes>mesh_pay(CTX0.M)) return 0;
  return pending_push(&CTX0,p,stride,bytes,nslots,node); }

size_t mesh_pump(void){
  if(mesh_try_attach(&CTX0,0)) return 0;
  if(refresh(&CTX0,0,0)) return CTX0.pending_count;
  pending_flush(&CTX0); return CTX0.pending_count+CTX0.inflight; }

size_t mesh_queued(void){ return CTX0.pending_count; }
size_t mesh_inflight(void){ return CTX0.inflight; }

size_t mesh_read(void **p, int *from){
  if(mesh_try_attach(&CTX0,0)) return 0;
  if(refresh(&CTX0,0,0)) return 0;
  pending_flush(&CTX0);
  return cread(&CTX0,p,from); }

size_t mesh_readv(void *p, size_t stride, uint32_t *sizes, int *from, size_t count){
  if(mesh_try_attach(&CTX0,0)) return 0;
  if(refresh(&CTX0,0,0)) return 0;
  pending_flush(&CTX0);
  size_t got=0;
  while(got<count){
    struct ring *ring=&CTX0.M->r[CMP];
    uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_relaxed);
    if(tail!=atomic_load_explicit(&ring->head,memory_order_acquire) && slot(CTX0.M,CMP,tail)->bytes>stride){ errno=EMSGSIZE; break; }
    void *q=0; int src=0; size_t b=cread(&CTX0,&q,&src);
    if(!b) break;
    memcpy((char*)p+got*stride,q,b);
    sizes[got]=(uint32_t)b; from[got]=src; got++; }
  return got; }

size_t mesh_yell(const void *p, size_t n, int node){
  struct mstream s, *v=&s; mesh_yell_start(&CTX0,&s,p,n,node,0);
  while(s.st==MS_RUN) mesh_turn(&CTX0,&v,1);
  return s.st==MS_DONE?n:0; }

size_t mesh_lissen(void *p, size_t n){
  struct mstream s, *v=&s;
  if(mesh_lissen_start(&CTX0,&s,p,n,0)) return 0;
  while(s.st==MS_RUN) mesh_turn(&CTX0,&v,1);
  size_t g=s.done; free(s.seen); return g; }

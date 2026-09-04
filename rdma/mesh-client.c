
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static struct mesh_ctx CTX0={.last=-1};

static const char *rname(const char *name){
  if(!name) name=getenv("MESH_REGION");
  return name?name:MESH_NAME; }

int mesh_attach(struct mesh_ctx *c, const char *name){
  if(c->M) return 0;
  name=rname(name);
  for(int t=0;t<3000 && !c->M;t++){
    int f=shm_open(name,O_RDWR,MESH_MODE);
    if(f>=0){
      struct stat s;
      if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
        struct hdr *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
        if(b!=MAP_FAILED){
          if(b->magic==MESH_MAGIC && b->version==MESH_VERSION){
            c->M=b; c->len=(size_t)s.st_size; c->ino=(uint64_t)s.st_ino; }
          else munmap(b,(size_t)s.st_size); } }
      close(f); }
    if(!c->M) usleep(10000); }
  if(!c->M) return -1;
  c->arena=mesh_data(c->M,c->M->pool); c->busy=calloc(c->M->arena,1);
  if(!c->busy){ munmap(c->M,c->len); c->M=0; c->arena=0; c->len=0; return -1; }
  c->cursor=0; c->inflight=0; c->last=-1; c->sub=c->ack=0; c->idle=0;
  atomic_store_explicit(&c->M->client,(uint64_t)getpid(),memory_order_release);
  return 0; }

static int stale(struct mesh_ctx *c){
  struct stat s; int f=shm_open(rname(0),O_RDWR,MESH_MODE);
  if(f<0) return 0;
  int gone = !fstat(f,&s) && (uint64_t)s.st_ino!=c->ino;
  close(f); return gone; }

static void mesh_retire(struct mesh_ctx *c, struct mstream **v, int k){
  for(int i=0;i<k;i++){ struct mstream *s=v[i];
    if(s->st==MS_RUN) s->st=MS_FAIL;
    free(s->seen); s->seen=0; }
  free(c->busy); c->busy=0; munmap(c->M,c->len); c->M=0; c->arena=0; }

static void reattach(struct mesh_ctx *c, struct mstream **v, int k){
  struct stat s; int f=shm_open(rname(0),O_RDWR,MESH_MODE);
  if(f<0) return;
  if(fstat(f,&s) || (size_t)s.st_size!=c->len){ close(f); mesh_retire(c,v,k); return; }
  struct hdr *t=mmap(NULL,c->len,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
  if(t==MAP_FAILED){ close(f); return; }
  int ready = t->magic==MESH_MAGIC && t->version==MESH_VERSION;
  size_t arena=t->arena;
  munmap(t,c->len);
  if(!ready){ close(f); return; }
  unsigned char *busy=calloc(arena,1);
  if(!busy){ close(f); return; }
  struct hdr *b=mmap(c->M,c->len,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,f,0);
  close(f);
  if(b==MAP_FAILED){ free(busy); c->M=0; c->arena=0; return; }
  for(int i=0;i<k;i++){ struct mstream *s=v[i];
    if(s->st==MS_RUN) s->st=MS_FAIL;
    free(s->seen); s->seen=0; }
  free(c->busy); c->M=b; c->arena=mesh_data(b,b->pool); c->busy=busy;
  c->cursor=0; c->inflight=0;
  c->ino=(uint64_t)s.st_ino; c->sub=c->ack=0; c->last=-1;
  atomic_store_explicit(&b->client,(uint64_t)getpid(),memory_order_release); }

static size_t cwrite(struct mesh_ctx *c, const void *p, size_t nbytes, int node){
  struct hdr *M=c->M;
  uint32_t s=(uint32_t)(((const unsigned char*)p-c->arena)/M->pgsz); size_t done=0;
  while(done<nbytes && s<M->arena){
    uint32_t u=mesh_pay(M);
    if(c->busy && c->busy[s]) break;
    struct desc d={.page=M->pool+s,.bytes=nbytes-done<u?(uint32_t)(nbytes-done):u,.node=(uint16_t)node};
    if(push(M,SUB,&d)) break;
    if(c->busy){ c->busy[s]=1; c->inflight++; }
    done+=d.bytes; s++; }
  return done; }

static size_t cread(struct mesh_ctx *c, void **p, int *from){
  struct hdr *M=c->M;
  if(c->last>=0){ struct desc r={.page=(uint32_t)c->last};
    while(push(M,REL,&r)){}
    c->last=-1; }
  struct desc d; if(pop(M,CMP,&d)) return 0;
  c->last=(int)d.page;
  if(p) *p=mesh_data(M,d.page);
  if(from) *from=d.node;
  return d.bytes; }

static void reclaim(struct mesh_ctx *c){
  struct desc d;
  while(!pop(c->M,ACK,&d)){
    if(c->busy && d.page>=c->M->pool && d.page<c->M->pool+c->M->arena && c->busy[d.page-c->M->pool]){
      c->busy[d.page-c->M->pool]=0; c->inflight--; }
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

static int ctl(struct mesh_ctx *c, int node, uint32_t sid, uint32_t k, uint64_t a){
  unsigned char *q=credit(c); if(!q) return -1;
  struct shdr sh={a,sid,k}; memcpy(q,&sh,MESH_OFF);
  if(cwrite(c,q,MESH_OFF,node)!=MESH_OFF) return -1;
  c->sub++; return 0; }

void mesh_yell_start(struct mesh_ctx *c, struct mstream *s,
                     const void *p, size_t n, int node, uint32_t sid){
  *s=(struct mstream){.src=p,.n=n,.node=node,.sid=sid,.st=n?MS_RUN:MS_DONE}; (void)c; }

int mesh_lissen_start(struct mesh_ctx *c, struct mstream *s,
                      void *p, size_t n, uint32_t sid){
  if(mesh_attach(c,0)) return -1;
  uint32_t u=mesh_pay(c->M)-MESH_OFF;
  *s=(struct mstream){.buf=p,.n=n,.sid=sid,.rx=1,.st=n?MS_RUN:MS_DONE,
                      .nb=(n+u-1)/u,.node=-1};
  s->seen=calloc((s->nb+7)/8,1);
  return s->seen?0:-1; }

int mesh_turn(struct mesh_ctx *c, struct mstream **v, int k){
  if(mesh_attach(c,0)) return 0;
  if(++c->idle >= 600){
    c->idle=0;
    if(stale(c)){ reattach(c,v,k); return 0; } }
  uint32_t u=mesh_pay(c->M)-MESH_OFF;
  uint64_t oldack=c->ack; reclaim(c); if(c->ack!=oldack) c->idle=0;
  for(;;){
    void *q; int from; size_t b=cread(c,&q,&from);
    if(b<MESH_OFF) break;
    c->idle=0;
    struct shdr sh; memcpy(&sh,q,MESH_OFF);
    struct mstream *s=0;
    for(int i=0;i<k;i++)
      if(v[i]->sid==sh.sid && v[i]->rx==(sh.k==K_DATA||sh.k==K_FIN)) s=v[i];
    if(!s) continue;
    if(sh.k==K_DATA){
      size_t len=b-MESH_OFF, ci=sh.off/u;
      if(sh.off+len<=s->n && ci<s->nb && !(s->seen[ci>>3]>>(ci&7)&1)){
        s->seen[ci>>3]|=(unsigned char)(1u<<(ci&7));
        memcpy(s->buf+sh.off,(char*)q+MESH_OFF,len); s->done+=len; } }
    else if(sh.k==K_FIN){
      s->node=from;
      if(s->done>=s->n){ ctl(c,from,s->sid,K_OK,0); s->st=MS_DONE; }
      else { while(s->hole<s->nb && (s->seen[s->hole>>3]>>(s->hole&7)&1)) s->hole++;
             ctl(c,from,s->sid,K_REQ,s->hole*(uint64_t)u); } }
    else if(sh.k==K_REQ){ if(sh.off<s->n){ s->off=sh.off; s->st=MS_RUN; s->fin_ack=0; } }
    else if(sh.k==K_OK){ s->done=s->n; s->st=MS_DONE; }
    if(s->st!=MS_RUN && s->seen){ free(s->seen); s->seen=0; } }
  int ndone=0;
  for(int i=0;i<k;i++){
    struct mstream *s=v[i];
    if(s->rx || s->st!=MS_RUN){ ndone += s->st==MS_DONE; continue; }
    size_t was=s->off;
    unsigned char *q;
    while(s->off<s->n && (q=credit(c))){
      uint32_t len = s->n-s->off<u ? (uint32_t)(s->n-s->off) : u;
      struct shdr sh={s->off,s->sid,K_DATA}; memcpy(q,&sh,MESH_OFF);
      memcpy(q+MESH_OFF,s->src+s->off,len);
      if(cwrite(c,q,MESH_OFF+len,s->node)!=MESH_OFF+len) break;
      c->sub++; s->off+=len; c->idle=0; }
    if(s->off>was) s->fin_ack=0;
    else if(s->off>=s->n && (!s->fin_ack || c->ack>=s->fin_ack)){
      if(!ctl(c,s->node,s->sid,K_FIN,s->n)) s->fin_ack=c->sub; } }
  return ndone; }

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
  if(ns) *ns=CTX0.M->arena;
  if(sp) *sp=CTX0.M->pgsz; if(up) *up=mesh_pay(CTX0.M);
  return CTX0.arena; }

size_t mesh_write(const void *p, size_t nbytes, int node){
  if(mesh_attach(&CTX0,0)) return 0;
  reclaim(&CTX0);
  return cwrite(&CTX0,p,nbytes,node); }

size_t mesh_write_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node){
  if(mesh_attach(&CTX0,0) || !bytes || bytes>mesh_pay(CTX0.M)) return 0;
  reclaim(&CTX0); size_t done=0;
  while(done<nslots){
    unsigned char *q=credit(&CTX0); if(!q) break;
    memcpy(q,(const unsigned char*)p+done*stride,bytes);
    if(cwrite(&CTX0,q,bytes,node)!=bytes) break;
    CTX0.sub++; done++; }
  return done; }

size_t mesh_queue_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node){
  if(mesh_attach(&CTX0,0) || !bytes || bytes>mesh_pay(CTX0.M)) return 0;
  return pending_push(&CTX0,p,stride,bytes,nslots,node); }

size_t mesh_pump(void){
  if(mesh_attach(&CTX0,0)) return 0;
  pending_flush(&CTX0); return CTX0.pending_count+CTX0.inflight; }

size_t mesh_queued(void){ return CTX0.pending_count; }
size_t mesh_inflight(void){ return CTX0.inflight; }

size_t mesh_read(void **p, int *from){
  if(mesh_attach(&CTX0,0)) return 0;
  pending_flush(&CTX0);
  size_t b=cread(&CTX0,p,from);
  if(b) CTX0.idle=0;
  else if(++CTX0.idle >= 600){
    CTX0.idle=0;
    if(stale(&CTX0)) reattach(&CTX0,0,0); }
  return b; }

size_t mesh_readv(void *p, size_t stride, uint32_t *sizes, int *from, size_t count){
  if(mesh_attach(&CTX0,0)) return 0;
  pending_flush(&CTX0);
  size_t got=0;
  while(got<count){
    void *q=0; int src=0; size_t b=cread(&CTX0,&q,&src);
    if(!b) break;
    memcpy((char*)p+got*stride,q,b);
    sizes[got]=(uint32_t)b; from[got]=src; got++; CTX0.idle=0; }
  if(!got && ++CTX0.idle>=600){
    CTX0.idle=0;
    if(stale(&CTX0)) reattach(&CTX0,0,0); }
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

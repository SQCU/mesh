// see RDMA-FIRST.md
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define FINQ 200000
#define FAILN 64

static struct mesh_ctx CTX0={.last=-1};

int mesh_attach(struct mesh_ctx *c, const char *name){
  if(c->M) return 0;
  if(!name) name=getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  for(int t=0;t<3000 && !c->M;t++){
    int f=shm_open(name,O_RDWR,MESH_MODE);
    if(f>=0){
      struct stat s;
      if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
        struct hdr *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
        if(b!=MAP_FAILED){
          if(b->magic==MESH_MAGIC && b->version==MESH_VERSION) c->M=b;
          else munmap(b,(size_t)s.st_size); } }
      if(!c->M) close(f); }
    if(!c->M) usleep(10000); }
  if(!c->M) return -1;
  c->arena=mesh_data(c->M,c->M->pool); c->last=-1; c->sub=c->ack=0;
  atomic_store_explicit(&c->M->client,(uint64_t)getpid(),memory_order_release);
  return 0; }

static size_t cwrite(struct mesh_ctx *c, const void *p, size_t nbytes, int node){
  struct hdr *M=c->M;
  uint32_t s=(uint32_t)(((const unsigned char*)p-c->arena)/M->pgsz); size_t done=0;
  while(done<nbytes && s<M->arena){
    uint32_t u=mesh_pay(M);
    struct desc d={.page=M->pool+s,.bytes=nbytes-done<u?(uint32_t)(nbytes-done):u,.node=(uint16_t)node};
    if(push(M,SUB,&d)) break;
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

static unsigned char *credit(struct mesh_ctx *c){
  if(c->sub - c->ack >= c->M->arena) return 0;
  return c->arena+(size_t)(c->sub%c->M->arena)*c->M->pgsz; }

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
  uint32_t u=mesh_pay(c->M)-MESH_OFF;
  struct desc da;
  while(!pop(c->M,ACK,&da)) c->ack++;
  for(;;){
    void *q; int from; size_t b=cread(c,&q,&from);
    if(b<MESH_OFF) break;
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
    else if(sh.k==K_REQ){ if(sh.off<s->n){ s->off=sh.off; s->st=MS_RUN; s->quiet=0; } }
    else if(sh.k==K_OK){ s->done=s->n; s->st=MS_DONE; } }
  int ndone=0;
  for(int i=0;i<k;i++){
    struct mstream *s=v[i];
    if(s->rx || s->st!=MS_RUN){ ndone += s->st==MS_DONE; continue; }
    unsigned char *q;
    while(s->off<s->n && (q=credit(c))){
      uint32_t len = s->n-s->off<u ? (uint32_t)(s->n-s->off) : u;
      struct shdr sh={s->off,s->sid,K_DATA}; memcpy(q,&sh,MESH_OFF);
      memcpy(q+MESH_OFF,s->src+s->off,len);
      if(cwrite(c,q,MESH_OFF+len,s->node)!=MESH_OFF+len) break;
      c->sub++; s->off+=len; }
    if(s->off>=s->n && ++s->quiet%FINQ==1){
      if(s->quiet/FINQ>FAILN) s->st=MS_FAIL;
      else ctl(c,s->node,s->sid,K_FIN,s->n); } }
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
  return cwrite(&CTX0,p,nbytes,node); }

size_t mesh_read(void **p, int *from){
  if(mesh_attach(&CTX0,0)) return 0;
  return cread(&CTX0,p,from); }

size_t mesh_yell(const void *p, size_t n, int node){
  struct mstream s, *v=&s; mesh_yell_start(&CTX0,&s,p,n,node,0);
  while(s.st==MS_RUN) mesh_turn(&CTX0,&v,1);
  return s.st==MS_DONE?n:0; }

size_t mesh_lissen(void *p, size_t n){
  struct mstream s, *v=&s;
  if(mesh_lissen_start(&CTX0,&s,p,n,0)) return 0;
  while(s.st==MS_RUN) mesh_turn(&CTX0,&v,1);
  size_t g=s.done; free(s.seen); return g; }

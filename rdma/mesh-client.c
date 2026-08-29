// see RDMA-FIRST.md
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static struct hdr *M; static unsigned char *arena; static int last=-1;

void *mesh_open(size_t *ns, size_t *sp, size_t *up){
  if(!M){
    const char *n=getenv("MESH_REGION"); if(!n) n=MESH_NAME;
    for(int t=0;t<3000 && !M;t++){
      int f=shm_open(n,O_RDWR,MESH_MODE);
      if(f>=0){
        struct stat s;
        if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
          struct hdr *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
          if(b!=MAP_FAILED){
            if(b->magic==MESH_MAGIC && b->version==MESH_VERSION) M=b;
            else munmap(b,(size_t)s.st_size); } }
        if(!M) close(f); }
      if(!M) usleep(10000); }
    if(!M) return NULL;
    arena=mesh_data(M,M->pool);
    atomic_store_explicit(&M->client,(uint64_t)getpid(),memory_order_release); }
  if(ns) *ns=M->arena;
  if(sp) *sp=M->pgsz; if(up) *up=mesh_pay(M);
  return arena; }

size_t mesh_write(const void *p, size_t nbytes, int node){
  if(!M) return 0;
  uint32_t s=(uint32_t)(((const unsigned char*)p-arena)/M->pgsz); size_t done=0;
  while(done<nbytes && s<M->arena){
    uint32_t u=mesh_pay(M);
    struct desc d={.page=M->pool+s,.bytes=nbytes-done<u?(uint32_t)(nbytes-done):u,.node=(uint16_t)node};
    if(push(M,SUB,&d)) break;
    done+=d.bytes; s++; }
  return done; }

size_t mesh_read(void **p, int *from){
  if(!M) return 0;
  if(last>=0){ struct desc r={.page=(uint32_t)last};
    while(push(M,REL,&r)){}
    last=-1; }
  struct desc d; if(pop(M,CMP,&d)) return 0;
  last=(int)d.page;
  if(p) *p=mesh_data(M,d.page);
  if(from) *from=d.node;
  return d.bytes; }

#define FINQ 200000
#define FAILN 64

static uint64_t sub_c, ack_c;

static void harvest(void){ struct desc a; while(!pop(M,ACK,&a)) ack_c++; }

static unsigned char *try_take(void){
  if(sub_c-ack_c >= M->arena) return 0;
  return arena+(size_t)(sub_c%M->arena)*M->pgsz; }

static int ctl(int node, uint32_t sid, uint32_t k, uint64_t a){
  unsigned char *q=try_take(); if(!q) return -1;
  struct shdr sh={a,sid,k}; memcpy(q,&sh,MESH_OFF);
  if(mesh_write(q,MESH_OFF,node)!=MESH_OFF) return -1;
  sub_c++; return 0; }

void mesh_yell_start(struct mstream *s, const void *p, size_t n, int node, uint32_t sid){
  mesh_open(0,0,0);
  *s=(struct mstream){.src=p,.n=n,.node=node,.sid=sid,.st=n?MS_RUN:MS_DONE,.done=0}; }

int mesh_lissen_start(struct mstream *s, void *p, size_t n, uint32_t sid){
  if(!mesh_open(0,0,0)) return -1;
  uint32_t u=mesh_pay(M)-MESH_OFF;
  *s=(struct mstream){.buf=p,.n=n,.sid=sid,.rx=1,.st=n?MS_RUN:MS_DONE,
                      .nb=(n+u-1)/u,.node=-1};
  s->seen=calloc((s->nb+7)/8,1);
  return s->seen?0:-1; }

int mesh_poll(struct mstream **v, int k){
  if(!M) return 0;
  uint32_t u=mesh_pay(M)-MESH_OFF;
  harvest();
  for(;;){
    void *q; int from; size_t b=mesh_read(&q,&from);
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
      if(s->done>=s->n){ ctl(from,s->sid,K_OK,0); s->st=MS_DONE; }
      else { while(s->hole<s->nb && (s->seen[s->hole>>3]>>(s->hole&7)&1)) s->hole++;
             ctl(from,s->sid,K_REQ,s->hole*(uint64_t)u); } }
    else if(sh.k==K_REQ){ if(sh.off<s->n){ s->off=sh.off; s->st=MS_RUN; s->quiet=0; } }
    else if(sh.k==K_OK){ s->done=s->n; s->st=MS_DONE; } }
  int ndone=0;
  for(int i=0;i<k;i++){
    struct mstream *s=v[i];
    if(s->rx){ ndone += s->st==MS_DONE; continue; }
    if(s->st!=MS_RUN){ ndone += s->st==MS_DONE; continue; }
    unsigned char *q;
    while(s->off<s->n && (q=try_take())){
      uint32_t len = s->n-s->off<u ? (uint32_t)(s->n-s->off) : u;
      struct shdr sh={s->off,s->sid,K_DATA}; memcpy(q,&sh,MESH_OFF);
      memcpy(q+MESH_OFF,s->src+s->off,len);
      if(mesh_write(q,MESH_OFF+len,s->node)!=MESH_OFF+len) break;
      sub_c++; s->off+=len; }
    if(s->off>=s->n && ++s->quiet%FINQ==1){
      if(s->quiet/FINQ>FAILN) s->st=MS_FAIL;
      else ctl(s->node,s->sid,K_FIN,s->n); } }
  return ndone; }

int mesh_scatter(struct mstream *ss, const void *p, size_t n,
                 const int *nodes, int k, uint32_t sid0){
  size_t sh=(n+k-1)/k;
  for(int i=0;i<k;i++){ size_t o=(size_t)i*sh, l=o<n?(n-o<sh?n-o:sh):0;
    mesh_yell_start(&ss[i],(const char*)p+o,l,nodes[i],sid0+i); }
  return k; }

int mesh_gather(struct mstream *ss, void *p, size_t n, int k, uint32_t sid0){
  size_t sh=(n+k-1)/k;
  for(int i=0;i<k;i++){ size_t o=(size_t)i*sh, l=o<n?(n-o<sh?n-o:sh):0;
    if(mesh_lissen_start(&ss[i],(char*)p+o,l,sid0+i)) return -1; }
  return k; }

size_t mesh_yell(const void *p, size_t n, int node){
  struct mstream s, *v=&s; mesh_yell_start(&s,p,n,node,0);
  while(s.st==MS_RUN) mesh_poll(&v,1);
  return s.st==MS_DONE?n:0; }

size_t mesh_lissen(void *p, size_t n){
  struct mstream s, *v=&s;
  if(mesh_lissen_start(&s,p,n,0)) return 0;
  while(s.st==MS_RUN) mesh_poll(&v,1);
  size_t g=s.done; free(s.seen); return g; }

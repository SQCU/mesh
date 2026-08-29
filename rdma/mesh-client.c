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
    struct desc d={.page=M->pool+s,.bytes=mesh_clamp(M,nbytes-done),.node=(uint16_t)node};
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

#define CTL  (1ull<<63)
#define FINK 1ull
#define REQK 2ull
#define OKK  3ull

static uint64_t sub_c, ack_c;

static void harvest(void){ struct desc a; while(!pop(M,ACK,&a)) ack_c++; }

static unsigned char *take(size_t ns){
  for(;;){ harvest(); if(sub_c-ack_c<ns) break; }
  return arena+(size_t)(sub_c%ns)*M->pgsz; }

static void ctl(int node, uint64_t k, uint64_t a){
  size_t ns=M->arena; unsigned char *q=take(ns);
  uint64_t w[2]={CTL|k,a}; memcpy(q,w,16);
  if(mesh_write(q,16,node)==16) sub_c++; }

static void range(const void *p, size_t from, size_t n, int node){
  size_t ns=M->arena; uint32_t u=mesh_pay(M)-MESH_OFF;
  for(size_t off=from; off<n; ){
    unsigned char *q=take(ns);
    uint32_t len = n-off<u ? (uint32_t)(n-off) : u;
    memcpy(q,&off,MESH_OFF); memcpy(q+MESH_OFF,(const char*)p+off,len);
    if(mesh_write(q,MESH_OFF+len,node)!=MESH_OFF+len) continue;
    sub_c++; off+=len; } }

size_t mesh_yell(const void *p, size_t n, int node){
  if(!mesh_open(0,0,0)) return 0;
  range(p,0,n,node);
  for(int round=0; round<8; round++){
    ctl(node,FINK,n);
    for(long s=0; s<4000000; s++){
      void *q; int from; size_t b=mesh_read(&q,&from);
      if(b<16 || !(*(uint64_t*)q & CTL)) continue;
      uint64_t k=*(uint64_t*)q & ~CTL, a=((uint64_t*)q)[1];
      if(k==OKK) return n;
      if(k==REQK){ range(p,a,n,node); round=-1; break; } } }
  return 0; }

size_t mesh_lissen(void *dst, size_t n){
  if(!mesh_open(0,0,0)) return 0;
  uint32_t u=mesh_pay(M)-MESH_OFF;
  size_t nb=(n+u-1)/u, got=0;
  unsigned char *seen=calloc((nb+7)/8,1); if(!seen) return 0;
  size_t hole=0;
  for(;;){
    void *q; int from; size_t b=mesh_read(&q,&from);
    if(!b) continue;
    uint64_t w0; memcpy(&w0,q,MESH_OFF);
    if(w0 & CTL){
      if((w0&~CTL)!=FINK) continue;
      if(got>=n){
        for(long s=0; s<2000000; s++){
          ctl(from,OKK,0);
          size_t b2=mesh_read(&q,&from);
          if(b2>=16){ uint64_t v; memcpy(&v,q,8); if((v&CTL)&&(v&~CTL)==FINK) s=0; } }
        free(seen); return got; }
      while(hole<nb && (seen[hole>>3]>>(hole&7)&1)) hole++;
      ctl(from,REQK,hole*(uint64_t)u);
      continue; }
    size_t off=w0, len=b-MESH_OFF, ci=off/u;
    if(off+len<=n && ci<nb && !(seen[ci>>3]>>(ci&7)&1)){
      seen[ci>>3]|=(unsigned char)(1u<<(ci&7));
      memcpy((char*)dst+off,(char*)q+MESH_OFF,len); got+=len; } }
}

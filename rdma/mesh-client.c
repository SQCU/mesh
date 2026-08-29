// see RDMA-FIRST.md
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static struct hdr *M; static unsigned char *arena; static int held=-1;

void *mesh_open(size_t *ns, size_t *sp, size_t *up){
  if(!M){
    const char *n=getenv("MESH_REGION"); if(!n) n=MESH_NAME;
    int f=-1; for(int t=0;t<200 && f<0;t++){ f=shm_open(n,O_RDWR,MESH_MODE); if(f<0) usleep(10000); }
    if(f<0) return NULL;
    struct stat s; if(fstat(f,&s)||(size_t)s.st_size<sizeof(struct hdr)){ close(f); return NULL; }
    void *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
    if(b==MAP_FAILED){ close(f); return NULL; }
    M=b;
    if(M->magic!=MESH_MAGIC||M->version!=MESH_VERSION){ M=NULL; return NULL; }
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
  if(held>=0){ struct desc r={.page=(uint32_t)held};
    while(push(M,REL,&r)){}
    held=-1; }
  struct desc d; if(pop(M,CMP,&d)) return 0;
  held=(int)d.page;
  if(p) *p=mesh_data(M,d.page);
  if(from) *from=d.node;
  return d.bytes; }

size_t mesh_yell(const void *p, size_t n, int node){
  size_t ns; if(!mesh_open(&ns,0,0)) return 0;
  uint32_t u=mesh_pay(M)-MESH_OFF;
  uint64_t sub=0, ack=0; size_t off=0; struct desc a;
  while(off<n){
    while(!pop(M,ACK,&a)) ack++;
    if(sub-ack>=ns) continue;
    unsigned char *q=arena+(size_t)(sub%ns)*M->pgsz;
    uint32_t len = n-off<u ? (uint32_t)(n-off) : u;
    memcpy(q,&off,MESH_OFF); memcpy(q+MESH_OFF,(const char*)p+off,len);
    if(mesh_write(q,MESH_OFF+len,node)!=MESH_OFF+len) continue;
    sub++; off+=len; }
  while(ack<sub) if(!pop(M,ACK,&a)) ack++;
  return off; }

size_t mesh_lissen(void *dst, size_t n){
  if(!mesh_open(0,0,0)) return 0;
  size_t got=0;
  while(got<n){
    void *q; size_t b=mesh_read(&q,0); if(b<=MESH_OFF) continue;
    uint64_t off; memcpy(&off,q,MESH_OFF); size_t len=b-MESH_OFF;
    if(off+len<=n){ memcpy((char*)dst+off,(char*)q+MESH_OFF,len); got+=len; } }
  return got; }

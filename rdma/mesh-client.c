// see RDMA-FIRST.md
#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static struct mesh M; static unsigned char *arena;
static uint32_t stride, use, first; static int held=-1;

void *mesh_open(size_t bytes, size_t *sp, size_t *up){
  if(!M.h){
    const char *n=getenv("MESH_REGION"); if(!n) n=MESH_NAME;
    int f=-1; for(int t=0;t<200 && f<0;t++){ f=shm_open(n,O_RDWR,MESH_MODE); if(f<0) usleep(10000); }
    if(f<0) return NULL;
    struct stat s; if(fstat(f,&s)||(size_t)s.st_size<sizeof(struct hdr)){ close(f); return NULL; }
    void *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
    if(b==MAP_FAILED){ close(f); return NULL; }
    M.h=b; M.b=b;
    if(M.h->magic!=MESH_MAGIC||M.h->version!=MESH_VERSION){ M.h=NULL; return NULL; }
    stride=M.h->pgsz; use=mesh_pay(&M); first=M.h->pool;
    arena=mesh_at(&M,first)+sizeof(struct wire);
    atomic_store_explicit(&M.h->client,(uint64_t)getpid(),memory_order_release); }
  if(bytes > (size_t)M.h->arena*use) return NULL;
  if(sp) *sp=stride; if(up) *up=use;
  return arena; }

size_t mesh_write(const void *p, size_t nbytes, int node){
  if(!M.h) return 0;
  uint32_t s=(uint32_t)(((const unsigned char*)p-arena)/stride); size_t done=0;
  while(done<nbytes && s<M.h->arena){
    struct desc d={.page=first+s,.bytes=(uint32_t)(nbytes-done<use?nbytes-done:use),.node=(uint16_t)node};
    if(push(&M,SUB,&d)) break;
    done+=d.bytes; s++; }
  return done; }

size_t mesh_read(void **p, int *from){
  if(!M.h) return 0;
  if(held>=0){ struct desc r={.page=(uint32_t)held};
    while(push(&M,REL,&r)){}
    held=-1; }
  struct desc d; if(pop(&M,CMP,&d)) return 0;
  held=(int)d.page;
  if(p) *p=mesh_at(&M,d.page)+sizeof(struct wire);
  if(from) *from=d.node;
  return d.bytes; }

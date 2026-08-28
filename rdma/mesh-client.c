#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int mesh_attach(struct mesh *M, const char *name){
  memset(M,0,sizeof *M);
  M->fd = shm_open(name, O_RDWR, 0600);
  if(M->fd < 0) return -1;
  struct stat st;
  if(fstat(M->fd, &st) || (size_t)st.st_size < sizeof(struct mesh_hdr)){ close(M->fd); return -1; }
  M->len = (size_t)st.st_size;
  M->base = mmap(NULL, M->len, PROT_READ|PROT_WRITE, MAP_SHARED, M->fd, 0);
  if(M->base == MAP_FAILED){ close(M->fd); return -1; }
  M->h = (struct mesh_hdr*)M->base;
  if(M->h->magic != MESH_MAGIC || M->h->version != MESH_VERSION){
    munmap(M->base, M->len); close(M->fd); return -1; }
  atomic_store_explicit(&M->h->client_pid,(uint64_t)getpid(),memory_order_release);
  return 0;
}

void mesh_detach(struct mesh *M){
  if(!M->base) return;
  atomic_store_explicit(&M->h->client_pid,0,memory_order_release);
  munmap(M->base, M->len); close(M->fd);
  memset(M,0,sizeof *M);
}

int mesh_acquire(struct mesh *M, uint32_t *page){
  struct mesh_desc d;
  if(mesh_pop(M,&M->h->rfree,M->h->free_off,&d)) return -1;
  *page = d.page; return 0;
}

int mesh_send(struct mesh *M, uint32_t page, uint32_t bytes, uint16_t node){
  struct mesh_desc d = {.page=page,.bytes=bytes,.seq=0,.node=node,.flags=0};
  return mesh_push(M,&M->h->rsub,M->h->sub_off,&d);
}

int mesh_poll(struct mesh *M, uint32_t *page, uint32_t *bytes, uint16_t *from){
  struct mesh_desc d;
  if(mesh_pop(M,&M->h->rcmp,M->h->cmp_off,&d)) return -1;
  *page=d.page; *bytes=d.bytes; if(from) *from=d.node; return 0;
}

void mesh_release(struct mesh *M, uint32_t page){
  struct mesh_desc d = {.page=page};
  while(mesh_push(M,&M->h->rrel,M->h->rel_off,&d)) ;   // bridge always drains
}

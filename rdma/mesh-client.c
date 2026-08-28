#include "mesh.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

static struct mesh G;
static unsigned char *g_arena;      // first slot's payload
static uint32_t g_stride, g_usable, g_hdr, g_first;
static int g_held = -1;             // slot handed out by the last mesh_read

void *mesh_open(size_t bytes, size_t *stride, size_t *usable){
  if(!G.h){
    const char *name = getenv("MESH_REGION"); if(!name) name = "/mesh0";
    for(int t=0;t<500;t++){
      int fd = shm_open(name, O_RDWR, 0600);
      if(fd >= 0){
        struct stat st;
        if(!fstat(fd,&st) && (size_t)st.st_size > sizeof(struct mesh_hdr)){
          void *b = mmap(NULL,(size_t)st.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
          if(b != MAP_FAILED){
            struct mesh_hdr *h = (struct mesh_hdr*)b;
            if(h->magic==MESH_MAGIC && h->version==MESH_VERSION && h->arena_pages){
              G.h=h; G.base=(unsigned char*)b; G.len=(size_t)st.st_size; G.fd=fd;
              break;
            }
            munmap(b,(size_t)st.st_size);
          }
        }
        close(fd);
      }
      usleep(20000);
    }
    if(!G.h) return NULL;
    g_hdr    = G.h->hdr_bytes;
    g_stride = G.h->pgsz;
    g_usable = G.h->pgsz - g_hdr;
    g_arena  = G.base + G.h->arena_off + g_hdr;
    g_first  = (uint32_t)((G.h->arena_off - G.h->data_off) / G.h->pgsz);
    atomic_store_explicit(&G.h->client_pid,(uint64_t)getpid(),memory_order_release);
  }
  if(bytes > (size_t)G.h->arena_pages * g_usable) return NULL;
  if(stride) *stride = g_stride;
  if(usable) *usable = g_usable;
  return g_arena;
}

size_t mesh_write(const void *p, size_t nbytes, int node){
  if(!G.h) return 0;
  size_t off = (size_t)((const unsigned char*)p - g_arena);
  uint32_t slot = (uint32_t)(off / g_stride);
  size_t done = 0;
  while(done < nbytes && slot < G.h->arena_pages){
    uint32_t n = (uint32_t)(nbytes - done < g_usable ? nbytes - done : g_usable);
    struct mesh_desc d = {.page = g_first + slot,
                          .bytes = n, .seq = 0, .node = (uint16_t)node, .flags = 0};
    if(mesh_push(&G,&G.h->rsub,G.h->sub_off,&d)) break;   // bridge is behind; come back
    done += n; slot++;
  }
  return done;
}

size_t mesh_read(void **p, int *from){
  if(!G.h) return 0;
  if(g_held >= 0){                                   // return the previous slot
    struct mesh_desc r = {.page=(uint32_t)g_held};
    while(mesh_push(&G,&G.h->rrel,G.h->rel_off,&r)) ;
    g_held = -1;
  }
  struct mesh_desc d;
  if(mesh_pop(&G,&G.h->rcmp,G.h->cmp_off,&d)) return 0;
  g_held = (int)d.page;
  if(p) *p = G.base + G.h->data_off + (size_t)d.page * G.h->pgsz + g_hdr;
  if(from) *from = d.node;
  return d.bytes;
}

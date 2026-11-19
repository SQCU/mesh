/* mesh-order-state: a read-only view of the region's orders and lanes (design/node-addressed-transfers.md), for a
   stalled transfer: each live order's taken and retired transfers each way, every transfer between them with its
   pieces, completions, done word and gates, and each lane's units posted and done each way.  It maps the region
   read-only and writes nothing. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mesh.h"
static uint64_t ld(const _Atomic uint64_t *w){ return atomic_load_explicit(w,memory_order_acquire); }
int main(int argc,char **argv){
  const unsigned show=argc>1?(unsigned)atoi(argv[1]):8;   /* transfers listed per direction */
  int fd=shm_open(MESH_NAME,O_RDONLY,0);
  if(fd<0){perror("shm_open");return 1;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){close(fd);return 2;}
  struct hdr *m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0);close(fd);
  if(m==MAP_FAILED){perror("mmap");return 3;}
  for(uint32_t n=0;n<MESH_NODES;n++){
    const struct mesh_node *x=mesh_node_at(m,n);
    const uint32_t node=atomic_load(&x->node);
    if(!node)continue;
    printf("order %u toward node %u: live %u generation %llu freeze %llu epochs %u\n",n,node,atomic_load(&x->live),
           (unsigned long long)ld(&x->generation),(unsigned long long)ld(&x->freeze),atomic_load(&x->epochs));
    for(int way=0;way<2;way++){
      const uint64_t tail=ld(&x->tail[way]),retired=ld(&x->retired[way]);
      printf("  %s: taken %llu retired %llu\n",way?"receive":"send",(unsigned long long)tail,(unsigned long long)retired);
      unsigned listed=0;
      for(uint64_t i=retired;i<tail && listed<show;i++,listed++){
        const struct mesh_slot *s=mesh_ring(m,x,way,i);
        const struct mesh_crossing *t=&s->t;
        printf("    #%llu seq %llu owner %llu pieces %llu completed %llu state %llu | units %llu gates %u gate %#llx base %llu done %#llx value %llu gen %u",
               (unsigned long long)i,(unsigned long long)ld(&s->sequence),(unsigned long long)ld(&s->owner),
               (unsigned long long)ld(&s->pieces),(unsigned long long)ld(&s->completed),(unsigned long long)ld(&s->state),
               (unsigned long long)t->units,t->gates,(unsigned long long)t->gate,(unsigned long long)t->base,
               (unsigned long long)t->done,(unsigned long long)t->value,t->generation);
        if(t->gate && t->gate<(uint64_t)st.st_size && !t->gates)
          printf(" gate word now %llu",(unsigned long long)ld((const _Atomic uint64_t *)((const char *)m+t->gate)));
        if(t->done && t->done<(uint64_t)st.st_size)
          printf(" done word now %llu",(unsigned long long)ld((const _Atomic uint64_t *)((const char *)m+t->done)));
        putchar('\n');
      }
    }
  }
  for(uint32_t p=0;p<MESH_STREAMS;p++){
    const struct mesh_lanes *l=mesh_lanes_at(m,p);
    if(!atomic_load(&l->live))continue;
    printf("lanes %u: peer %u child %u",p,atomic_load(&l->peer),atomic_load(&l->child));
    for(int way=0;way<2;way++)
      for(int j=0;j<2;j++)
        printf(" | %s%d posted %llu done %llu",way?"recv":"send",j,(unsigned long long)ld(&l->posts[way].posted[j]),
               (unsigned long long)ld((const _Atomic uint64_t *)&l->posts[way].done[j]));
    putchar('\n');
  }
  return 0;
}

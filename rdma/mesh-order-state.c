/* mesh-order-state: a read-only view of the region's switch and its programs (design/switched-multipath-dma.md), for a
   stalled transfer: the pool (slots free, by state), each live lane (its peer, bytes queued, its outbox, SENDs
   completed and their mean service and wait, landings), the view (each link's line and stamp), the weights (each
   destination's distribution over the neighbours, the refreshes made), and each program toward each node (its key,
   steps issued, closing, the next send piece staged, and per direction each transfer's executions complete, with its
   gate and done words as they stand).  It maps the region read-only and writes nothing.

   usage: mesh-order-state [TRANSFERS]   (TRANSFERS: the transfers listed per program table, default 8) */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mesh-switched.h"
static uint64_t ld(const _Atomic uint64_t *w){ return atomic_load_explicit(w,memory_order_acquire); }
static unsigned long long u(uint64_t v){ return (unsigned long long)v; }
static void word(const struct hdr *m,const char *name,uint64_t at){
  if(at && at%8==0 && at+8<=m->length)printf(" %s@%#llx=%llu",name,u(at),u(ld((const _Atomic uint64_t *)((const char *)m+at))));
}
int main(int argc,char **argv){
  const unsigned show=argc>1?(unsigned)atoi(argv[1]):8;
  int fd=shm_open(mesh_name(),O_RDONLY,0);
  if(fd<0){perror("shm_open");return 1;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){close(fd);return 2;}
  struct hdr *m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0);close(fd);
  if(m==MAP_FAILED){perror("mmap");return 3;}
  if(m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT){fprintf(stderr,"mesh-order-state: not a region of layout %u\n",MESH_LAYOUT);return 4;}
  const struct mesh_switch *sw=mesh_switch_of(m);
  if(sw->magic!=MESH_SWITCH_MAGIC){printf("node %u: no switch made\n",m->node);return 0;}
  uint32_t state[4]={0};
  for(uint32_t s=0;s<sw->slots;s++){const uint32_t k=atomic_load(&sw->slot[s].state);if(k<4)state[k]++;}
  printf("node %u: pool %u slots of %llu bytes: %llu free (stack), %u receiving, %u held, %u sending\n",m->node,sw->slots,u(sw->slot_bytes),
         u(ld(&sw->free_count)),state[SLOT_RECEIVE],state[SLOT_HELD],state[SLOT_SENDING]);
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){
    const struct mesh_switch_lane *x=sw->lane+l;
    if(!atomic_load(&x->live) && !ld(&x->sends))continue;
    const uint64_t sends=ld(&x->sends);float rate,wait;mesh_switch_unpack(ld(&x->estimate),&rate,&wait);
    printf("  lane %u (port %u, lane %u): %s toward node %u, %llu bytes queued, outbox %llu, %llu SENDs (%.1f us service, %.1f us wait mean), "
           "%llu landed, estimate %.2f GB/s\n",l,l/(2*MESH_LINK_STREAMS),l%(2*MESH_LINK_STREAMS),atomic_load(&x->live)?"live":"not live",
           atomic_load(&x->peer),u(ld(&x->queued)),u(ld(&x->tail)-ld(&x->head)),u(sends),sends?(double)ld(&x->service)/sends/1e3:0,
           sends?(double)ld(&x->wait)/sends/1e3:0,u(ld(&x->landed)),rate);
  }
  const struct mesh_switch_view *v=&sw->view;
  printf("  view: %u nodes, %u links\n",v->nodes,v->links);
  for(uint32_t l=0;l<v->links;l++){
    float wait,spread;mesh_switch_unpack(ld(&v->link[l].line),&wait,&spread);
    printf("    %u -> %u: %u cables, %.2f GB/s, %.1f us; line %.1f +- %.1f us, stamp %llu\n",v->link[l].from,v->link[l].to,v->link[l].cables,
           v->link[l].rate,v->link[l].latency/1e3,wait/1e3,spread/1e3,u(ld(&v->link[l].stamp)));
  }
  const struct mesh_switch_weights *w=&sw->weights;
  printf("  weights: %llu refreshes, %u neighbours\n",u(w->refreshes),w->deg);
  for(uint32_t d=0;d<w->nodes && d<v->nodes;d++){
    printf("    toward node %u:",v->node[d]);
    for(uint32_t k=0;k<w->deg;k++)printf(" %u:%.2f",w->neighbour[k]<v->nodes?v->node[w->neighbour[k]]:0,w->w[d][k]);
    printf("\n");
  }
  for(uint32_t g=0;g<MESH_SWITCH_LOGS;g++)if(ld(&sw->log[g].tail))
    printf("  log %u: %llu records, %llu taken, %llu dropped\n",g,u(ld(&sw->log[g].tail)),u(ld(&sw->log[g].head)),u(ld(&sw->log[g].dropped)));
  for(uint32_t n=0;n<MESH_NODES;n++){
    const struct mesh_node *x=mesh_node_at(m,n);
    const uint32_t node=atomic_load(&x->node);
    if(!node)continue;
    printf("toward node %u: %s, bandwidth %llu\n",node,atomic_load(&x->live)?"live":"not live",u(ld(&x->bandwidth)));
    for(uint32_t p=0;p<MESH_PROGRAMS;p++){
      const struct mesh_declared *d=x->program+p;
      const uint64_t owner=ld(&d->owner);
      if(!owner)continue;
      printf("  program %u: owner %#llx key %#llx period %u issued %llu closed %lld claimed %llu%s\n",p,u(owner),u(d->program.key),d->program.period,
             u(ld(&d->issued)),ld(&d->closed)==UINT64_MAX?-1LL:(long long)ld(&d->closed),u(ld(&d->claimed)),mesh_switch_done(m,d)?" (done)":"");
      for(int way=0;way<2;way++){
        const struct mesh_crossing *t=mesh_switch_table(m,d,way);
        if(!t || !d->progress[way])continue;
        const struct mesh_switch_transfer *pr=(const struct mesh_switch_transfer *)((const char *)m+d->progress[way]);
        printf("    %s: %llu transfers, %llu pieces an execution\n",way?"receive":"send",u(d->program.transfers[way]),u(d->program.pieces[way]));
        for(uint64_t i=0;i<d->program.transfers[way] && i<show;i++){
          printf("      transfer %llu: %llu units, step %u, complete %llu",u(i),u(t[i].units),t[i].step,u(ld(&pr[i].complete)));
          word(m,"gate",t[i].gates?0:t[i].gate);word(m,"done",t[i].done);
          printf("\n");
        }
      }
    }
  }
  munmap(m,(size_t)st.st_size);
  return 0;
}

/* mesh-order-state: a read-only view of the region's orders and lanes (design/declared-programs.md), for a stalled
   transfer: each live order's epochs, segments and declared programs (each one's steps issued, its tables' first
   entries with their gate and done words as they stand), each direction's posting bound and pieces raised; and each
   lane's units posted and done each way, with the piece its next unit to post and its next unit to complete are.  It
   maps the region read-only and writes nothing.

   usage: mesh-order-state [TRANSFERS]   (TRANSFERS: the entries listed per program table, default 8) */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mesh-order.h"
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
  for(uint32_t n=0;n<MESH_NODES;n++){
    const struct mesh_node *x=mesh_node_at(m,n);
    const uint32_t node=atomic_load(&x->node);
    if(!node)continue;
    const uint32_t epochs=atomic_load(&x->epochs);
    const uint64_t segments=ld(&x->segments);
    printf("order %u toward node %u: live %u generation %llu epochs %u segments %llu agreed %llu declaring %llu restart %llu\n",n,node,
           atomic_load(&x->live),u(ld(&x->generation)),epochs,u(segments),u(ld(&x->agreed)),u(ld(&x->declaring)),u(ld(&x->restart)));
    for(int d=0;d<2;d++)printf("  %s: limit %llu raised %llu\n",d?"receive":"send",u(ld(&x->limit[d])),u(ld(&x->raised[d])));
    for(uint32_t e=0;e<epochs && e<MESH_EPOCHS;e++){
      const struct mesh_epoch *at=x->epoch+e;
      printf("  epoch %u from %llu/%llu:",e,u(at->first[0]),u(at->first[1]));
      for(uint32_t c=0;c<at->cables && c<MESH_PORTS;c++)printf(" port %u lanes %u wire %#llx;",at->cable[c].port,at->cable[c].lanes,u(at->cable[c].wire));
      putchar('\n');
    }
    for(uint64_t s=segments>MESH_SEGMENTS?segments-MESH_SEGMENTS:0;s<segments;s++){
      const struct mesh_segment *g=order_segment(x,s);
      printf("  segment %llu from %llu/%llu round %llu/%llu rounds %lld:",u(s),u(g->start[0]),u(g->start[1]),u(g->round[0]),u(g->round[1]),
             (long long)ld(&g->rounds));
      for(uint32_t k=0;k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++)printf(" program %u from execution %llu;",g->slot[k],u(g->base[k]));
      putchar('\n');
    }
    for(uint32_t p=0;p<MESH_PROGRAMS;p++){
      const struct mesh_declared *d=x->program+p;
      if(!ld(&d->owner))continue;
      printf("  program %u: owner %#llx key %llu period %u count %u issued %llu closed %lld end %lld/%lld\n",p,u(ld(&d->owner)),
             u(d->program.key),d->program.period,d->program.count,u(ld(&d->issued)),(long long)ld(&d->closed),(long long)ld(&d->end[0]),
             (long long)ld(&d->end[1]));
      for(int way=0;way<2;way++){
        const struct mesh_crossing *t=order_table(m,d,way);
        printf("    %s: %llu transfers, %llu pieces an execution\n",way?"receive":"send",u(d->program.transfers[way]),u(d->program.pieces[way]));
        for(uint64_t i=0;t && i<d->program.transfers[way] && i<show;i++){
          printf("      #%llu step %u first %llu units %llu gate %llu+%llu/execution done %llu+%llu/execution",u(i),t[i].step,u(t[i].first),
                 u(t[i].units),u(t[i].gate_base),u(t[i].gate_step),u(t[i].done_value),u(t[i].done_step));
          word(m,"gate",t[i].gate);word(m,"done",t[i].done);
          putchar('\n');
        }
      }
    }
  }
  for(uint32_t p=0;p<MESH_STREAMS;p++){
    const struct mesh_lanes *l=mesh_lanes_at(m,p);
    if(!atomic_load(&l->live))continue;
    const uint32_t peer=atomic_load(&l->peer);
    const struct mesh_node *x=NULL;
    for(uint32_t n=0;n<MESH_NODES;n++)if(atomic_load(&mesh_node_at(m,n)->node)==peer)x=mesh_node_at(m,n);
    const uint32_t epochs=x?atomic_load(&x->epochs):0;
    printf("lanes %u (port %u): peer %u child %u passes %llu",p,p/MESH_LINK_STREAMS,peer,atomic_load(&l->child),u(ld(&l->passes)));
    for(int way=0;way<2;way++)
      for(int j=0;j<2;j++){
        const uint32_t lane=(p%MESH_LINK_STREAMS)*2+(uint32_t)j;
        const uint64_t posted=ld(&l->posts[way].posted[j]),done=ld(&l->posts[way].done[j]);
        printf(" | %s lane %u posted %llu done %llu",way?"recv":"send",lane,u(posted),u(done));
        if(x)printf(" (next piece %lld, first undone %lld)",(long long)order_unit_piece(x,epochs,way,p/MESH_LINK_STREAMS,lane,posted),
                    (long long)order_unit_piece(x,epochs,way,p/MESH_LINK_STREAMS,lane,done));
      }
    putchar('\n');
  }
  return 0;
}

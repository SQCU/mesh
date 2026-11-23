/* mesh-switch-stat: the switch as the bridge keeps it in its region (mesh-switched.h): its pool's pages and their free
   counts, the counters (struct mesh_switch_counters: where a pass's time goes and what could not be carried), each
   port's line (its child's state, its link's peer and plan) and cable's queue and measured terms (its rate, bytes a ns,
   and a short and a long page's time alone, ns) and the pages on its inbound rings, its serving or used lanes'
   in-flight positions and feedback, and its published view, one JSON line to stdout.  It reads the region and writes nothing: two lines around
   a workload are its counts (their difference).

   usage: mesh-switch-stat */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "mesh-switched.h"

#define U(x) ((unsigned long long)atomic_load_explicit(&(x),memory_order_relaxed))
int main(void){
  const int fd=shm_open(MESH_NAME,O_RDONLY,0);
  if(fd<0){printf("{\"up\":false,\"errno\":%d}\n",errno);return 0;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){printf("{\"up\":false,\"errno\":%d}\n",EINVAL);return 0;}
  struct hdr *m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0);
  close(fd);
  if(m==MAP_FAILED){printf("{\"up\":false,\"errno\":%d}\n",errno);return 0;}
  if(m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT){printf("{\"up\":false,\"layout\":%u}\n",m->version);return 0;}
  const struct mesh_switch *sw=(const struct mesh_switch *)((const char *)m+m->switch_off);
  if(sw->magic!=MESH_SWITCH_MAGIC){printf("{\"up\":true,\"node\":%u,\"switch\":false}\n",m->node);return 0;}
  const struct mesh_pool *pool=(const struct mesh_pool *)((const char *)m+m->pool_off);
  printf("{\"up\":true,\"node\":%u,\"clock\":%llu,\"pool\":{\"pages\":%u,\"classes\":[",m->node,
         (unsigned long long)clock_gettime_nsec_np(CLOCK_UPTIME_RAW),pool->pages);
  for(uint32_t c=0;c<MESH_CLASSES;c++)
    printf("%s{\"frames\":%u,\"pages\":%u,\"free\":%llu}",c?",":"",pool->cls[c].frames,pool->cls[c].pages,U(pool->cls[c].free));
  printf("]},\"counters\":[");
  for(uint32_t r=0;r<MESH_SWITCH_LOGS;r++){
    const struct mesh_switch_counters *c=sw->counters+r;
    printf("%s{\"ring\":%u,\"passes\":%llu,\"pass_ns\":[%llu,%llu],\"part_ns\":[%llu,%llu,%llu,%llu,%llu],",r?",":"",r,U(c->passes),
           U(c->pass_ns[0]),U(c->pass_ns[1]),U(c->part_ns[0]),U(c->part_ns[1]),U(c->part_ns[2]),U(c->part_ns[3]),U(c->part_ns[4]));
    printf("\"taken\":%llu,\"landed\":%llu,\"set\":%llu,\"forwards\":%llu,\"groups\":%llu,",U(c->taken),U(c->landed),U(c->set),
           U(c->forwards),U(c->groups));
    printf("\"posts\":%llu,\"post_wait_ns\":%llu,\"post_call_ns\":%llu,\"sent\":%llu,\"send_ns\":%llu,",U(c->posts),U(c->post_wait_ns),
           U(c->post_call_ns),U(c->sent),U(c->send_ns));
    printf("\"starved\":%llu,\"refreshes\":%llu,",U(c->starved),U(c->refreshes));
    printf("\"unreachable\":%llu,\"lost\":%llu}",U(c->unreachable),U(c->lost));
  }
  printf("],\"ports\":[");
  uint64_t line[MESH_PORTS];
  for(uint32_t p=0;p<MESH_PORTS;p++){
    line[p]=atomic_load_explicit(&sw->port[p],memory_order_acquire);
    const struct mesh_switch_cable *x=sw->cable+p;
    uint64_t inbound[MESH_CLASSES]={0};
    for(uint32_t q=0;q<MESH_PORTS;q++)for(uint32_t c=0;c<MESH_CLASSES;c++){
      const struct mesh_switch_inbound *r=&sw->inbound[q][p][c];
      const uint64_t head=U(r->head);   /* the head first: the tail read after it is never below it */
      inbound[c]+=U(r->tail)-head;
    }
    printf("%s{\"port\":%u,\"state\":%u,\"peer\":%u,\"plan\":%u,\"queued\":%llu,\"inbound\":[%llu,%llu],\"rate\":%g,\"page\":[%g,%g]}",
           p?",":"",p,mesh_port_state(line[p]),mesh_port_peer(line[p]),mesh_port_plan(line[p]),U(x->queued),
           (unsigned long long)inbound[0],(unsigned long long)inbound[1],
           (double)atomic_load_explicit(&x->rate,memory_order_relaxed),(double)atomic_load_explicit(&x->page[0],memory_order_relaxed),
           (double)atomic_load_explicit(&x->page[1],memory_order_relaxed));
  }
  printf("],\"lanes\":[");
  int first=1;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){
    const struct mesh_switch_lane *x=sw->lane+l;
    const uint32_t p=mesh_switch_port_of(l),j=l%(2*MESH_LINK_STREAMS);
    const int serving=mesh_port_state(line[p])==CHILD_SERVING && mesh_port_plan(line[p])>>j&1;
    if(!U(x->sends) && !U(x->landed) && !serving)continue;
    float rate,wait;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&rate,&wait);
    printf("%s{\"lane\":%u,\"serving\":%d,\"frames\":%u,\"posted\":%llu,\"completed\":%llu,"
           "\"sends\":%llu,\"bytes\":%llu,\"service\":%llu,\"wait\":%llu,\"landed\":%llu,\"rate\":%g,\"est_wait\":%g}",
           first?"":",",l,serving,mesh_switch_lane_frames(j),U(x->posted),U(x->completed),U(x->sends),U(x->bytes),
           U(x->service),U(x->wait),U(x->landed),rate,wait);
    first=0;
  }
  /* the view as published (its copy as its version names it) */
  const struct mesh_switch_view *v=sw->view+mesh_switch_published(&sw->view_version);
  printf("],\"view\":[");
  for(uint32_t i=0;i<v->nodes && i<MESH_SWITCH_N;i++)printf("%s%u",i?",":"",v->node[i]);
  printf("]}\n");
  return 0;
}

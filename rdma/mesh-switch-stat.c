/* mesh-switch-stat: the switch's counters as the bridge keeps them in its region (mesh-switched.h struct
   mesh_switch_counters: where a crossing's time goes), its live lanes' feedback and its weights, one JSON line to stdout.
   It reads the region and writes nothing: two lines around a run are the run's counts (their difference).

   usage: mesh-switch-stat   (MESH_REGION names another region, as the clients' attach) */
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
  const int fd=shm_open(mesh_name(),O_RDONLY,0);
  if(fd<0){printf("{\"up\":false,\"errno\":%d}\n",errno);return 0;}
  struct stat st;
  if(fstat(fd,&st) || st.st_size<(off_t)sizeof(struct hdr)){printf("{\"up\":false,\"errno\":%d}\n",EINVAL);return 0;}
  struct hdr *m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0);
  close(fd);
  if(m==MAP_FAILED){printf("{\"up\":false,\"errno\":%d}\n",errno);return 0;}
  if(m->magic!=MESH_MAGIC || m->version!=MESH_LAYOUT){printf("{\"up\":false,\"layout\":%u}\n",m->version);return 0;}
  const struct mesh_switch *sw=(const struct mesh_switch *)((const char *)m+m->switch_off);
  printf("{\"up\":true,\"node\":%u,\"clock\":%llu,\"slots\":%u,\"slot_bytes\":%llu,\"free\":%llu,\"counters\":[",m->node,
         (unsigned long long)clock_gettime_nsec_np(CLOCK_MONOTONIC),sw->slots,(unsigned long long)sw->slot_bytes,U(sw->free_count));
  for(uint32_t r=0;r<MESH_SWITCH_LOGS;r++){
    const struct mesh_switch_counters *c=sw->counters+r;
    printf("%s{\"ring\":%u,\"passes\":%llu,\"pass_ns\":%llu,\"part_ns\":[%llu,%llu,%llu,%llu,%llu],",r?",":"",r,U(c->passes),U(c->pass_ns),
           U(c->part_ns[0]),U(c->part_ns[1]),U(c->part_ns[2]),U(c->part_ns[3]),U(c->part_ns[4]));
    printf("\"staged\":%llu,\"staged_dests\":%llu,\"stage_ns\":%llu,\"credit_waits\":%llu,\"window_waits\":%llu,\"slot_waits\":%llu,",
           U(c->staged),U(c->staged_dests),U(c->stage_ns),U(c->credit_waits),U(c->window_waits),U(c->slot_waits));
    printf("\"forwards\":%llu,\"forward_ns\":%llu,\"groups\":%llu,\"copies\":%llu,\"kept\":%llu,",U(c->forwards),U(c->forward_ns),U(c->groups),U(c->copies),U(c->kept));
    printf("\"posts\":%llu,\"post_wait_ns\":%llu,\"sent\":%llu,\"send_ns\":%llu,",U(c->posts),U(c->post_wait_ns),U(c->sent),U(c->send_ns));
    printf("\"landed\":%llu,\"delivered\":%llu,\"deliver_wait_ns\":%llu,\"hops\":[%llu,%llu,%llu,%llu],",U(c->landed),U(c->delivered),U(c->deliver_wait_ns),
           U(c->hops[0]),U(c->hops[1]),U(c->hops[2]),U(c->hops[3]));
    printf("\"credits\":%llu,\"credits_taken\":%llu,\"refreshes\":%llu,\"refresh_ns\":%llu,\"unplaced\":[",U(c->credits),U(c->credits_taken),U(c->refreshes),U(c->refresh_ns));
    for(int k=0;k<8;k++)printf("%s%llu",k?",":"",U(c->unplaced[k]));
    printf("],\"post_call_ns\":%llu,\"orphans\":%llu}",U(c->post_call_ns),U(c->orphans));
  }
  printf("],\"lanes\":[");
  int first=1;
  for(uint32_t l=0;l<MESH_SWITCH_LANES;l++){
    const struct mesh_switch_lane *x=sw->lane+l;
    if(!U(x->sends) && !U(x->landed) && !U(x->live))continue;
    float rate,wait;mesh_switch_unpack(atomic_load_explicit(&x->estimate,memory_order_relaxed),&rate,&wait);
    printf("%s{\"lane\":%u,\"live\":%llu,\"peer\":%llu,\"queued\":%llu,\"sends\":%llu,\"bytes\":%llu,\"service\":%llu,\"wait\":%llu,\"landed\":%llu,\"rate\":%g,\"est_wait\":%g}",
           first?"":",",l,U(x->live),U(x->peer),U(x->queued),U(x->sends),U(x->bytes),U(x->service),U(x->wait),U(x->landed),rate,wait);
    first=0;
  }
  const struct mesh_switch_weights *w=&sw->weights;
  printf("],\"view\":[");
  for(uint32_t i=0;i<sw->view.nodes && i<MESH_SWITCH_N;i++)printf("%s%u",i?",":"",sw->view.node[i]);
  printf("],\"refreshes\":%llu,\"neighbour\":[",(unsigned long long)w->refreshes);
  for(uint32_t k=0;k<w->deg && k<MESH_SWITCH_N;k++)printf("%s%u",k?",":"",w->neighbour[k]);
  printf("],\"w\":[");
  for(uint32_t d=0;d<w->nodes && d<MESH_SWITCH_N;d++){
    printf("%s[",d?",":"");
    for(uint32_t k=0;k<w->deg && k<MESH_SWITCH_N;k++)printf("%s%.3f",k?",":"",w->w[d][k]);
    printf("]");
  }
  printf("]}\n");
  return 0;
}

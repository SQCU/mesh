/* design/declared-programs.md "Testing": one end of a declared program that leaves a piece never sent.  A program of one
   step, a 4 KB SEND each way an execution, its SEND gated by a word this client raises; it issues EXECUTIONS
   executions and opens every one's gate but the last (as a fused crossing whose last frame's producer never ran: its
   publication never made), awaits every landing but the last's, closes its program at the executions it issued (a
   whole number: the close succeeds) and ends.  Its last SEND is then never posted, and its peer's last receive never
   filled: the queue pairs they are on wait at them for ever, and so would every program declared after them.

   With MODE "inflight" it opens every execution's gate, awaits nothing and ends at once: its last pieces are posted
   and still crossing as it ends, and complete through their own completions (the next program follows them).

   usage: sim-dangling PEER_NODE [EXECUTIONS [MODE]]   (environment: MESH_REGION; the two ends' clients the same, each
   naming the other's node) */
#include "../mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct lease { struct mesh_crossing send,receive; struct mesh_dma_span spans[2]; _Atomic uint64_t gate,sent,landed; unsigned char bytes[2][4096]; };
int main(int argc,char **argv){
  if(argc<2)return 2;
  const uint32_t peer=(uint32_t)strtoul(argv[1],NULL,0);
  const uint64_t executions=argc>2?strtoull(argv[2],NULL,0):16;
  const int inflight=argc>3 && !strcmp(argv[3],"inflight");
  struct mesh_ctx ctx={.fd=-1};
  int status=mesh_attach(&ctx);
  if(status){fprintf(stderr,"sim-dangling: mesh_attach: %d\n",status);return 1;}
  struct mesh_order order;
  /* the order live (the bridges may be agreeing its epochs) */
  for(const uint64_t until=mesh_clock_ns()+30*MESH_PLATFORM_NS;(status=mesh_order_open(&ctx,peer,&order))==ENOLINK && mesh_clock_ns()<until;)usleep(1000);
  if(status){fprintf(stderr,"sim-dangling: mesh_order_open: %d\n",status);return 1;}
  struct mesh_section section;
  if((status=mesh_section_create(&ctx,sizeof(struct lease),1,&section))){fprintf(stderr,"sim-dangling: lease: %d\n",status);return 1;}
  struct lease *l=mesh_section_address(&ctx,section,0);
  memset(l,0,sizeof *l);
  const uint64_t at=(uint64_t)((char *)l-(char *)ctx.M);
  for(int d=0;d<2;d++)l->spans[d]=(struct mesh_dma_span){at+offsetof(struct lease,bytes)+(uint64_t)d*4096,4096};
  l->send=(struct mesh_crossing){.spans=at+offsetof(struct lease,spans),.units=1,.first=0,.step=0,
    .gate=at+offsetof(struct lease,gate),.gate_base=0,.gate_step=1,.done=at+offsetof(struct lease,sent),.done_value=1,.done_step=1};
  l->receive=(struct mesh_crossing){.spans=at+offsetof(struct lease,spans)+sizeof(struct mesh_dma_span),.units=1,.first=0,.step=0,
    .done=at+offsetof(struct lease,landed),.done_value=1,.done_step=1};
  const struct mesh_declaration program={.key=0x5161dd4e,.period=1,.count=1,
    .table={at+offsetof(struct lease,send),at+offsetof(struct lease,receive)},.transfers={1,1},.pieces={1,1}};
  struct mesh_run run;
  if((status=mesh_declare(&ctx,&order,&program,&run))){fprintf(stderr,"sim-dangling: mesh_declare: %d\n",status);return 1;}
  mesh_issue(&run,executions);
  /* every execution's SEND gate but the last's opened */
  atomic_store_explicit(&l->gate,inflight?executions:executions-1,memory_order_release);
  for(const uint64_t until=mesh_clock_ns()+MESH_PLATFORM_NS;!inflight && atomic_load(&l->landed)<executions-1;){
    if(mesh_clock_ns()>=until){fprintf(stderr,"sim-dangling: landed %llu of %llu\n",(unsigned long long)atomic_load(&l->landed),(unsigned long long)executions-1);return 1;}
    usleep(100);
  }
  if((status=mesh_close(&ctx,&run))){fprintf(stderr,"sim-dangling: mesh_close: %d\n",status);return 1;}
  printf("{\"mode\":\"%s\",\"dangling\":%s,",inflight?"inflight":"dangling",inflight?"false":"true");
  printf("\"executions\":%llu,\"landed\":%llu}\n",(unsigned long long)executions,(unsigned long long)atomic_load(&l->landed));
  return 0;
}

/* design/transport-one-gib-2026-09-10.md: the one-way stream reading of a link, a client's program of one transfer on
   a lane pair of the link, every payload word checked at the receiver (mesh
   554a3793^ mesh-verbs-stream.c and cea7fce7^ mesh-page-stream.c; the bridge's own raw stream, its own queue pair and
   its STREAM messages, is gone with the per-program queue pairs: design/bridge-and-ipc.md#lanes).

     mesh-stream send NODE BYTES PAIR
     mesh-stream receive NODE BYTES PAIR
       a program of one transfer of BYTES toward NODE, a neighbour, bound, prepared, started and published as any
       client's (mesh-call.h, mesh-collective.h mesh_host_publish), received into a lease with a landing word a block
       (mesh_host_inputs), which the receiver follows in order, and of the receiver's answer, one block back on the
       pair's small lane once it has checked every word, at whose landing the sender ends (the 10-17 lifecycle rule,
       delivery a peer's count: design/recovery.md#five-seconds F8); riding the link lanes' pair PAIR (0 to
       MESH_LANE_PAIRS - 1, the same at both ends: its large lane MESH_LANE_FIRST + 2 PAIR), which each end leases at its
       own bridge on every cable to NODE whose link has its lanes made, all or none (mesh-call.h mesh_lanes_claim), and
       lets go at its end (mesh_lanes_release); run the receiver first, then the sender within MESH_PLATFORM_NS, on the
       two nodes; each end's JSON line

   Each end's wait concludes at its last landing, at ~0 or its destination's failure, or MESH_PLATFORM_NS after its
   last landing (its start's, where none) with nothing landed (design/recovery.md#five-seconds).  The receiver's line
   names its mode, the bytes, the blocks, the mismatched words, its first-to-last time and its interior: the bytes
   landed between a tenth and nine tenths of the blocks and their time, the rate the 2026-09-10 comparison takes (raw
   9.383 GB/s, the bridge's page path 9.338). */
#include "mesh-collective.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STREAM_MAGIC UINT64_C(0x6d65736870616765)

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static int usage(void){
  fprintf(stderr,"usage: mesh-stream send|receive NODE BYTES PAIR\n");
  return 2;
}

/* the bridge's path: one program of the stream's transfer and the answer's, the receiver following its landing words
   in order */
static int stream_flow(int receive,uint32_t node,uint64_t bytes,uint32_t pair){
  struct mesh_ctx context={0};
  char name[64];
  snprintf(name,sizeof name,"mesh-stream-%llu",(unsigned long long)bytes);
  int status=mesh_attach(&context,name);
  if(status){fprintf(stderr,"mesh-stream: attach: %s\n",strerror(status));return 1;}
  struct hdr *m=context.M;
  const uint64_t block=(uint64_t)m->block*m->pgsz,blocks=bytes/block;
  struct mesh_section section,answer;
  if(!blocks || bytes%block)status=EINVAL;
  if(!status)status=mesh_section_create(&context,bytes,1,&section);
  if(!status)status=mesh_section_create(&context,block,1,&answer);
  if(!status)status=mesh_transfer_bind(&context,node,0,receive?MESH_RECEIVE:MESH_SEND,1,section,section.pages,0,1,1);
  if(!status)status=mesh_transfer_bind(&context,node,1,receive?MESH_SEND:MESH_RECEIVE,2,answer,answer.pages,0,1,1);
  if(!status)status=mesh_transfers_prepare(&context,1,1,1);
  if(!status)status=mesh_host_inputs(&context);
  if(status){fprintf(stderr,"mesh-stream: %s: %s\n",receive?"receive":"send",strerror(status));mesh_detach(&context);return 1;}
  struct mesh_client *client=mesh_client_at(m,context.client);
  const uint64_t owner=atomic_load(&client->owner);
  struct mesh_lane_report mine;
  uint32_t chosen[MESH_PORTS],cables=0;
  mesh_lanes_free(m,&mine);
  for(uint32_t k=0;k<mine.count;k++){chosen[k]=mine.link[k].peer==node?pair:UINT32_MAX;cables+=chosen[k]!=UINT32_MAX;}
  status=cables?mesh_lanes_claim(m,owner,&mine,chosen):EHOSTUNREACH;
  if(status){
    fprintf(stderr,"mesh-stream: lane pair %u toward node %u on its %u cables: %s\n",pair,node,cables,strerror(status));
    mesh_detach(&context);return 1;
  }
  struct mesh_tx *tx=&client->tx[mesh_destination(client,node)];
  tx->lane[0]=MESH_LANE_FIRST+2*pair;tx->lane[1]=MESH_LANE_FIRST+2*pair+1;
  unsigned char *data=mesh_section_address(&context,section,0);
  const uint64_t words=block/sizeof(uint64_t);
  for(uint64_t p=0;p<blocks;p++){
    uint64_t *page=(uint64_t *)(data+p*block);
    if(receive)memset(page,0,block);
    else for(uint64_t w=0;w<words;w++)page[w]=(p<<32)^w^STREAM_MAGIC;
  }
  printf("{\"stream\":\"flow\",\"role\":\"%s\",\"ready\":true,\"bytes\":%llu,\"blocks\":%llu}\n",receive?"receive":"send",
         (unsigned long long)bytes,(unsigned long long)blocks);
  fflush(stdout);
  status=mesh_transfers_start(&context);
  if(status){fprintf(stderr,"mesh-stream: start: %s\n",strerror(status));mesh_lanes_release(m,owner);mesh_detach(&context);return 1;}
  struct mesh_port_info *peer=&client->status[mesh_destination(client,node)];
  if(!receive){
    const uint64_t begin=clock_ns();
    mesh_host_publish(&context,section,0);
    mesh_transfers_wake(&context);
    uint64_t answered=0;
    while(!(answered=mesh_host_arrived(&context,answer,0)) && atomic_load_explicit(&peer->phase,memory_order_acquire)!=MESH_STOPPED &&
          clock_ns()-begin<MESH_PLATFORM_NS){}
    printf("{\"stream\":\"flow\",\"role\":\"send\",\"bytes\":%llu,\"blocks\":%llu,\"answered\":%d,\"publish_to_peer_end_s\":%.9f,\"peer_code\":%lld}\n",
           (unsigned long long)bytes,(unsigned long long)blocks,answered==1,(double)(clock_ns()-begin)/1e9,
           (long long)atomic_load_explicit(&peer->code,memory_order_relaxed));
    mesh_lanes_release(m,owner);
    mesh_detach(&context);
    return answered!=1;
  }
  uint64_t completed=0,first_count=0,low_count=0,high_count=0,polls=0,first=0,low=0,high=0,last=0,cancelled=0;
  const uint64_t begin=clock_ns();
  for(uint64_t heard=begin;completed<blocks && !cancelled && atomic_load_explicit(&peer->phase,memory_order_acquire)!=MESH_STOPPED &&
      clock_ns()-heard<MESH_PLATFORM_NS;){
    polls++;
    const uint64_t before=completed;
    for(;completed<blocks;completed++){
      const struct mesh_delivery *delivery=mesh_delivery_at(m,section.first+(uint32_t)completed);
      const uint64_t word=atomic_load_explicit((_Atomic uint64_t *)((char *)m+delivery->device_input),memory_order_acquire);
      if(word==UINT64_MAX){cancelled=1;break;}
      if(!word)break;
    }
    if(completed==before)continue;
    heard=last=clock_ns();
    if(!first){first=last;first_count=completed;}
    if(!low && completed>=blocks/10){low=last;low_count=completed;}
    if(!high && completed>=9*blocks/10){high=last;high_count=completed;}
  }
  uint64_t mismatches=0,correct=0;
  for(uint64_t p=0;p<completed;p++){
    const uint64_t *page=(const uint64_t *)(data+p*block);
    uint64_t bad=0;
    for(uint64_t w=0;w<words;w++)bad+=page[w]!=((p<<32)^w^STREAM_MAGIC);
    mismatches+=bad;correct+=!bad;
  }
  if(completed==blocks){mesh_host_publish(&context,answer,0);mesh_transfers_wake(&context);}
  const double interior=high>low?(double)(high-low)/1e9:0;
  const uint64_t interior_bytes=(high_count-low_count)*block;
  printf("{\"stream\":\"flow\",\"role\":\"receive\",\"bytes\":%llu,\"block\":%llu,\"blocks\":%llu,\"completed\":%llu,\"cancelled\":%llu,"
         "\"mismatches\":%llu,\"correct_blocks\":%llu,\"first_to_last_s\":%.9f,\"first_to_last_bytes\":%llu,\"interior_s\":%.9f,"
         "\"interior_bytes\":%llu,\"interior_GBps\":%.6f,\"polls\":%llu,\"peer_code\":%lld}\n",(unsigned long long)bytes,
         (unsigned long long)block,(unsigned long long)blocks,(unsigned long long)completed,(unsigned long long)cancelled,
         (unsigned long long)mismatches,(unsigned long long)correct,last>first?(double)(last-first)/1e9:0,
         (unsigned long long)((completed-first_count)*block),interior,(unsigned long long)interior_bytes,
         interior>0?(double)interior_bytes/interior/1e9:0,(unsigned long long)polls,
         (long long)atomic_load_explicit(&peer->code,memory_order_relaxed));
  mesh_lanes_release(m,owner);
  mesh_detach(&context);
  return completed!=blocks || mismatches;
}

int main(int argc,char **argv){
  if(argc!=5)return usage();
  char *end;
  const unsigned long node=strtoul(argv[2],&end,0);
  if(*end || !node || node>UINT32_MAX)return usage();
  const unsigned long long bytes=strtoull(argv[3],&end,0);
  if(*end || !bytes)return usage();
  const unsigned long pair=strtoul(argv[4],&end,0);
  if(*end || pair>=MESH_LANE_PAIRS)return usage();
  if(!strcmp(argv[1],"send") || !strcmp(argv[1],"receive"))return stream_flow(!strcmp(argv[1],"receive"),(uint32_t)node,bytes,(uint32_t)pair);
  return usage();
}

#include "mesh-dataflow.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static volatile sig_atomic_t stopped;

// ../design/algorithm-sources.md#literal-transport-primitives
static void stop(int number){ stopped=number; }

// ../design/algorithm-sources.md#literal-transport-primitives
static double seconds(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec+t.tv_nsec*1e-9;
}

// ../design/algorithm-sources.md#literal-transport-primitives
int main(int argc,char **argv){
  if(argc<4){ fprintf(stderr,"usage: %s send|echo peer pages [timeout_seconds]\n",argv[0]); return 2; }
  int echo=!strcmp(argv[1],"echo"),status=0;
  uint16_t peer=(uint16_t)strtoul(argv[2],NULL,10);
  uint32_t n=(uint32_t)strtoul(argv[3],NULL,10);
  double timeout=argc>4?strtod(argv[4],NULL):30;
  if((!echo && strcmp(argv[1],"send")) || !n || n>UINT32_MAX/2 || !(timeout>0)) return 2;
  signal(SIGTERM,stop); signal(SIGINT,stop);
  struct mesh_ctx *context=mesh_context();
  status=mesh_attach(context,NULL);
  if(status){ fprintf(stderr,"attach=%d\n",status); return 1; }
  struct hdr *memory=mesh_region(context);
  struct mesh_rows *pages=mesh_rows_create(context,(size_t)n*2,0);
  if(!pages){ fprintf(stderr,"create=%d\n",errno); return 1; }
  uint32_t physical=mesh_rows_allocate(pages,n,pages->bytes);
  if(physical==MESH_ROW_ABSENT){ fprintf(stderr,"allocate=%d\n",errno); return 1; }
  struct mesh_row_map source={.first=0,.count=1,.stride=1,.physical=physical,.physical_stride=1};
  struct mesh_row_function function={.output=&source,.outputs=1,.rows=n};
  struct mesh_row_binding bindings[]={
    {.first=echo?n:0,.count=n,.remote=1,.peer=peer,.remote_table=0},
    {.first=n,.count=n,.peer=peer,.receive=1}
  };
  struct mesh_row_map held={.first=n,.count=n};
  status=mesh_rows_realize(pages,&function,1,bindings,2,&held,1);
  if(status){ fprintf(stderr,"realize=%d\n",status); return 1; }
  double *sent=calloc(n,sizeof *sent),*arrived=calloc(n,sizeof *arrived);
  if(!sent || !arrived){ fprintf(stderr,"timestamps=%d\n",ENOMEM); return 1; }
  if(!echo) for(uint32_t i=0;i<n;i++){
    uint64_t *data=(void*)mesh_at(memory,physical+i);
    for(uint32_t j=0;j<pages->bytes/sizeof *data;j++) data[j]=((uint64_t)i<<32)^j^UINT64_C(0x6d65736870616765);
    mesh_rows_map(pages,i,physical+i,1,source.uses[i],0);
  }
  uint64_t bad=atomic_load_explicit(&memory->bad,memory_order_relaxed);
  printf("page_echo_ready role=%s pages=%u page_bytes=%u arena_pages=%zu receive_pages_required=%u\n",
    argv[1],n,pages->bytes,context->allocation,n); fflush(stdout);
  double begin=seconds(),deadline=begin+timeout,end=begin;
  if(!echo) for(uint32_t i=0;i<n;i++){
    sent[i]=seconds(); mesh_rows_publish(pages,&function,i,1);
  }
  uint32_t received=0;
  while(received<n && !stopped && seconds()<deadline){
    mesh_rows_poll(context);
    for(uint32_t i=0;i<n;i++) if(!arrived[i] &&
      __atomic_load_n(&pages->table[n+i].stamp,__ATOMIC_ACQUIRE)==1){
      arrived[i]=seconds(); received++; end=arrived[i];
    }
  }
  uint64_t mismatches=0;
  for(uint32_t i=0;i<n;i++) if(arrived[i]){
    const uint64_t *data=mesh_row_data(pages,n+i);
    for(uint32_t j=0;j<pages->bytes/sizeof *data;j++)
      mismatches+=data[j]!=(((uint64_t)i<<32)^j^UINT64_C(0x6d65736870616765));
  }
  double rtt=0;
  if(!echo) for(uint32_t i=0;i<n;i++) if(arrived[i]) rtt+=arrived[i]-sent[i];
  double elapsed=end-begin;
  printf("page_echo_result role=%s pages=%u received=%u mismatched_words=%llu cohort_seconds=%.9f "
    "mean_observed_rtt_us=%.3f half_rtt_estimate_us=%.3f roundtrip_pages_per_second=%.3f "
    "roundtrip_payload_bytes_per_second=%.3f aggregate_link_payload_bytes_per_second=%.3f\n",
    argv[1],n,received,(unsigned long long)mismatches,elapsed,
    echo || !received?0:rtt/received*1e6,echo || !received?0:rtt/received*5e5,
    echo || !received?0:received/elapsed,echo || !received?0:(double)received*pages->bytes/elapsed,
    echo || !received?0:(double)received*pages->bytes*2/elapsed); fflush(stdout);
  status=received!=n || mismatches || stopped;
  if(atomic_load_explicit(&memory->bad,memory_order_relaxed)!=bad) status=1;
  for(size_t i=0;i<atomic_load_explicit(&memory->port_count,memory_order_acquire);i++){
    struct mesh_row_metadata meta=mesh_link_metadata(context,i);
    if(meta.code){ fprintf(stderr,"port=%zu domain=%u code=%lld\n",i,meta.domain,(long long)meta.code); status=1; }
  }
  int detached=EBUSY;
  deadline=seconds()+timeout;
  while(detached==EBUSY && !stopped && seconds()<deadline) detached=mesh_detach(context);
  printf("page_echo_close status=%d detached=%d\n",status,detached);
  free(arrived); free(sent);
  return status || detached;
}

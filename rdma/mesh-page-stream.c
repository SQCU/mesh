#include "mesh-dataflow.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static volatile sig_atomic_t stopped;

// ../design/algorithm-sources.md#one-gib-stream-measurement
static void stop(int number){ stopped=number; }

// ../design/algorithm-sources.md#one-gib-stream-measurement
static double seconds(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec+t.tv_nsec*1e-9;
}

// ../design/algorithm-sources.md#one-gib-stream-measurement
int main(int argc,char **argv){
  if(argc<3){ fprintf(stderr,"usage: %s send|receive peer [timeout_seconds]\n",argv[0]); return 2; }
  int receive=!strcmp(argv[1],"receive"),status=0;
  uint16_t peer=(uint16_t)strtoul(argv[2],NULL,10);
  double timeout=argc>3?strtod(argv[3],NULL):60;
  if((!receive && strcmp(argv[1],"send")) || !(timeout>0)) return 2;
  signal(SIGTERM,stop); signal(SIGINT,stop);
  struct mesh_ctx *context=mesh_context();
  status=mesh_attach(context,NULL);
  if(status){ fprintf(stderr,"attach=%d\n",status); return 1; }
  struct hdr *memory=mesh_region(context);
  const uint64_t bytes=UINT64_C(1)<<30;
  uint32_t n=(uint32_t)(bytes/memory->pgsz),local=receive?1:n;
  struct mesh_rows *pages=mesh_rows_create(context,(size_t)n+1,0);
  if(!pages){ fprintf(stderr,"create=%d\n",errno); return 1; }
  uint32_t physical=mesh_rows_allocate(pages,local,pages->bytes);
  if(physical==MESH_ROW_ABSENT){ fprintf(stderr,"allocate=%d\n",errno); return 1; }
  struct mesh_row_map source={.first=receive?n:0,.count=1,.stride=1,.physical=physical,.physical_stride=1};
  struct mesh_row_function function={.output=&source,.outputs=1,.rows=local};
  struct mesh_row_binding binding={.first=0,.count=n,.remote=0,.peer=peer,.receive=receive,.remote_table=0};
  struct mesh_row_map held={.first=0,.count=n};
  status=mesh_rows_realize(pages,&function,1,&binding,1,receive?&held:NULL,receive?1:0);
  if(status){ fprintf(stderr,"realize=%d\n",status); return 1; }
  if(!receive) for(uint32_t i=0;i<n;i++){
    uint64_t *data=(void*)mesh_at(memory,physical+i);
    for(uint32_t j=0;j<pages->bytes/sizeof *data;j++) data[j]=((uint64_t)i<<32)^j^UINT64_C(0x6d65736870616765);
  }
  uint32_t *indices=receive?NULL:malloc((size_t)n*sizeof *indices);
  printf("page_stream_ready role=%s bytes=%llu pages=%u page_bytes=%u arena_pages=%zu\n",
    argv[1],(unsigned long long)bytes,n,pages->bytes,context->allocation); fflush(stdout);
  uint64_t bad=atomic_load_explicit(&memory->bad,memory_order_relaxed);
  double begin=seconds(),deadline=begin+timeout,first=0,low=0,high=0,end=0;
  if(!receive){
    size_t count=mesh_rows_issue(pages,&function,1,indices,n);
    if(count!=n) status=1;
    for(size_t i=0;i<count/2;i++){ uint32_t x=indices[i]; indices[i]=indices[count-i-1]; indices[count-i-1]=x; }
    mesh_rows_complete(pages,&function,1,indices,count);
  }
  double submission_end=seconds();
  uint32_t completed=0,first_count=0,low_count=0,high_count=0;
  uint64_t polls=0;
  while(completed<n && !stopped && seconds()<deadline){
    polls++;
    uint32_t previous=completed;
    while(completed<n && (receive?
      __atomic_load_n(&pages->table[completed].stamp,__ATOMIC_ACQUIRE)==1:
      __atomic_load_n(&pages->table[completed].uses,__ATOMIC_ACQUIRE)==0)) completed++;
    if(completed==previous) continue;
    end=seconds();
    if(!first){ first=end; first_count=completed; }
    if(!low && completed>=n/10){ low=end; low_count=completed; }
    if(!high && completed>=n*9/10){ high=end; high_count=completed; }
  }
  uint64_t mismatches=0;
  if(receive) for(uint32_t i=0;i<completed;i++){
    const uint64_t *data=mesh_row_data(pages,i);
    for(uint32_t j=0;j<pages->bytes/sizeof *data;j++)
      mismatches+=data[j]!=(((uint64_t)i<<32)^j^UINT64_C(0x6d65736870616765));
  }
  printf("page_stream_result role=%s bytes=%llu completed_pages=%u mismatched_words=%llu polls=%llu "
    "submission_seconds=%.9f first_to_last_seconds=%.9f first_to_last_bytes=%llu "
    "interior_seconds=%.9f interior_bytes=%llu interior_bytes_per_second=%.3f\n",
    argv[1],(unsigned long long)bytes,completed,(unsigned long long)mismatches,(unsigned long long)polls,
    receive?0:submission_end-begin,end-first,(unsigned long long)(completed-first_count)*pages->bytes,
    high-low,(unsigned long long)(high_count-low_count)*pages->bytes,
    high>low?(double)(high_count-low_count)*pages->bytes/(high-low):0); fflush(stdout);
  status=completed!=n || mismatches || stopped || atomic_load_explicit(&memory->bad,memory_order_relaxed)!=bad;
  for(size_t i=0;i<atomic_load_explicit(&memory->port_count,memory_order_acquire);i++){
    struct mesh_row_metadata meta=mesh_link_metadata(context,i);
    if(meta.code){ fprintf(stderr,"port=%zu domain=%u code=%lld\n",i,meta.domain,(long long)meta.code); status=1; }
  }
  free(indices);
  int detached=mesh_detach(context);
  printf("page_stream_close status=%d detached=%d\n",status,detached);
  return status || detached;
}

#include "mesh-dataflow.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* One GiB of literal blocks, one way, through the page table and bridge. No numerical function. */
static volatile sig_atomic_t stopped;
static void stop(int number){ stopped=number; }
static double seconds(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

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
  uint32_t block=memory->block,data=block-1;
  uint32_t blocks=(uint32_t)(bytes/((uint64_t)data*memory->pgsz)),n=blocks*block;
  uint32_t first,page=MESH_ABSENT;
  if(receive) first=mesh_landing_alloc(context,n);
  else { first=mesh_rows_alloc(context,n); page=mesh_arena_alloc(context,n,1); if(page!=MESH_ABSENT) mesh_map(context,first,n,page); }
  if(first==MESH_ABSENT || (!receive && page==MESH_ABSENT)){ fprintf(stderr,"allocate=%d\n",errno); return 1; }
  struct mesh_row_map map={.first=first,.count=n,.stride=n};
  struct mesh_row_function function={.output=&map,.outputs=1,.rows=1};
  uint32_t identity=mesh_bindings_reserve(context,MESH_ABSENT,1);
  if(identity==MESH_ABSENT){ fprintf(stderr,"binding=%d\n",errno); return 1; }
  struct mesh_row_binding binding={.first=first,.count=n,.binding=identity,.peer=peer,.receive=(uint16_t)receive};
  struct mesh_row_map held={.first=first,.count=n};
  status=mesh_realize(context,&function,receive?0:1,&binding,1,&held,receive?1:0);
  if(status){ fprintf(stderr,"realize=%d\n",status); return 1; }
  if(!receive) for(uint32_t b=0;b<blocks;b++) for(uint32_t i=0;i<data;i++){
    uint64_t *words=(void*)mesh_at(memory,page+b*block+i);
    for(uint32_t j=0;j<memory->pgsz/sizeof *words;j++) words[j]=((uint64_t)(b*data+i)<<32)^j^UINT64_C(0x6d65736870616765);
  }
  printf("page_stream_ready role=%s bytes=%llu blocks=%u block_pages=%u page_bytes=%u\n",argv[1],(unsigned long long)bytes,blocks,block,memory->pgsz); fflush(stdout);
  uint64_t bad=atomic_load_explicit(&memory->bad,memory_order_relaxed);
  double begin=seconds(),deadline=begin+timeout,firstTime=0,low=0,high=0,end=0;
  if(!receive){ uint32_t index; if(mesh_issue(context,&function,&index,1)!=1) status=1; else mesh_complete(context,&function,&index,1); }
  double submission_end=seconds();
  uint32_t completed=0,first_count=0,low_count=0,high_count=0;
  uint64_t polls=0;
  while(completed<blocks && !stopped && seconds()<deadline){
    polls++;
    uint32_t previous=completed;
    while(completed<blocks && (receive?mesh_bits_all(memory,MESH_PRESENT,first+completed*block,block)
      :mesh_bits_all(memory,MESH_READ+(int)binding.plane,first+completed*block,block))) completed++;
    if(completed==previous) continue;
    end=seconds();
    if(!firstTime){ firstTime=end; first_count=completed; }
    if(!low && completed>=blocks/10){ low=end; low_count=completed; }
    if(!high && completed>=blocks*9/10){ high=end; high_count=completed; }
  }
  uint64_t mismatches=0;
  if(receive) for(uint32_t b=0;b<completed;b++) for(uint32_t i=0;i<data;i++){
    const uint64_t *words=mesh_row_data(context,first+b*block+i);
    for(uint32_t j=0;j<memory->pgsz/sizeof *words;j++) mismatches+=words[j]!=(((uint64_t)(b*data+i)<<32)^j^UINT64_C(0x6d65736870616765));
  }
  uint64_t blockBytes=(uint64_t)data*memory->pgsz;
  printf("page_stream_result role=%s bytes=%llu completed_blocks=%u mismatched_words=%llu polls=%llu "
    "submission_seconds=%.9f first_to_last_seconds=%.9f first_to_last_bytes=%llu "
    "interior_seconds=%.9f interior_bytes=%llu interior_bytes_per_second=%.3f\n",
    argv[1],(unsigned long long)bytes,completed,(unsigned long long)mismatches,(unsigned long long)polls,
    receive?0:submission_end-begin,end-firstTime,(unsigned long long)(completed-first_count)*blockBytes,
    high-low,(unsigned long long)(high_count-low_count)*blockBytes,
    high>low?(double)(high_count-low_count)*blockBytes/(high-low):0); fflush(stdout);
  if(receive) mesh_consume(context,held,0);
  status=completed!=blocks || mismatches || stopped || atomic_load_explicit(&memory->bad,memory_order_relaxed)!=bad;
  struct mesh_row_metadata meta=mesh_link_metadata(context,0);
  if(meta.code){ fprintf(stderr,"port domain=%u code=%lld\n",meta.domain,(long long)meta.code); status=1; }
  int detached=mesh_detach(context);
  printf("page_stream_close status=%d detached=%d\n",status,detached);
  return status || detached;
}

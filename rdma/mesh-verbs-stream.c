#include "mesh-verbs.h"

#define STREAM_PAGES 65536u
#define STREAM_BYTES 16384u

// ../design/algorithm-sources.md#one-gibibyte-provider-stream
int main(int argc,char **argv){
  if(argc<3){ fprintf(stderr,"usage: %s send|receive peer [device] [timeout_seconds] [window_pages]\n",argv[0]); return 2; }
  int sending=!strcmp(argv[1],"send"),status=0;
  if(!sending && strcmp(argv[1],"receive")) return 2;
  double timeout=argc>4?strtod(argv[4],NULL):60;
  selected_device=argc>3?argv[3]:NULL;
  signal(SIGTERM,onsig); signal(SIGINT,onsig); signal(SIGPIPE,SIG_IGN);
  struct mesh_verbs verbs={0}; provider=&verbs;
  mynonce=((uint64_t)arc4random()<<32)|arc4random();
  size_t bytes=(size_t)STREAM_PAGES*STREAM_BYTES;
  char *memory=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(memory==MAP_FAILED){ perror("mmap"); return 1; }
  memset(memory,0,bytes);
  if(sending) for(uint32_t page=0;page<STREAM_PAGES;page++){
    uint64_t *words=(void*)(memory+(size_t)page*STREAM_BYTES);
    for(uint32_t word=0;word<STREAM_BYTES/sizeof *words;word++)
      words[word]=((uint64_t)page<<32)^word^UINT64_C(0x6d65736870616765);
  }
  double deadline=monotime()+timeout;
  while(!stop && monotime()<deadline &&
        ((lsock<0 && listener_up()) || verbs_up(sending?NULL:argv[2],memory,bytes,sending?0:1,STREAM_BYTES,0))){
    while(!down_pair()){}
    if(retire_device){ while(!down_verbs()){} retire_device=0; }
    usleep(100000);
  }
  if(stop || !provider->pair || monotime()>=deadline){ status=1; goto finish; }
  uint32_t capacity=(uint32_t)(sending?provider->send_capacity:provider->receive_capacity);
  uint32_t requested=argc>5?(uint32_t)strtoul(argv[5],NULL,10):capacity;
  if(requested && requested<capacity) capacity=requested;
  uint64_t *completion_ids=calloc(STREAM_PAGES,sizeof *completion_ids);
  uint8_t *valid_pages=calloc(STREAM_PAGES,sizeof *valid_pages);
  struct ibv_sge *sges=calloc(capacity,sizeof *sges);
  struct ibv_send_wr *sends=calloc(capacity,sizeof *sends);
  struct ibv_recv_wr *receives=calloc(capacity,sizeof *receives);
  struct ibv_wc *completions=calloc(capacity,sizeof *completions);
  if(!completion_ids || !valid_pages || !sges || !sends || !receives || !completions){ status=1; goto storage; }
  printf("provider_stream_ready role=%s bytes=%zu page_bytes=%u capacity=%u active_width=%u active_speed=%u pid=%d\n",
    argv[1],bytes,STREAM_BYTES,capacity,pa.active_width,pa.active_speed,getpid()); fflush(stdout);
  uint32_t posted=0,completed=0,first_count=0,low_count=0,high_count=0;
  uint64_t polls=0,posts=0,empty_polls=0,mismatches=0;
  double begin=monotime(),first=0,last=0,low_time=0,high_time=0,poll_seconds=0,post_seconds=0;
  deadline=begin+timeout;
  while(completed<STREAM_PAGES && !stop){
    uint32_t count=capacity-(posted-completed);
    if(count>STREAM_PAGES-posted) count=STREAM_PAGES-posted;
    if(count){
      for(uint32_t i=0;i<count;i++){
        uint32_t page=posted+i;
        sges[i]=region_sge(memory,(size_t)page*STREAM_BYTES,STREAM_BYTES);
        if(sending) sends[i]=(struct ibv_send_wr){.wr_id=page,.next=i+1<count?&sends[i+1]:NULL,
          .sg_list=&sges[i],.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
        else receives[i]=(struct ibv_recv_wr){.wr_id=page,.next=i+1<count?&receives[i+1]:NULL,
          .sg_list=&sges[i],.num_sge=1};
      }
      struct ibv_send_wr *bad_send=NULL; struct ibv_recv_wr *bad_receive=NULL;
#ifdef MESH_TRANSPORT_TIMING
      double post_begin=monotime();
#endif
      int error=sending?ibv_post_send(provider->pair,sends,&bad_send):ibv_post_recv(provider->pair,receives,&bad_receive);
#ifdef MESH_TRANSPORT_TIMING
      post_seconds+=monotime()-post_begin;
#endif
      posts++;
      if(error){ fprintf(stderr,"post=%d submitted=%u completed=%u\n",error,posted,completed); status=1; break; }
      posted+=count;
    }
#ifdef MESH_TRANSPORT_TIMING
    double poll_begin=monotime();
#endif
    int n=ibv_poll_cq(provider->completion_queue,(int)capacity,completions);
#ifdef MESH_TRANSPORT_TIMING
    poll_seconds+=monotime()-poll_begin;
#endif
    polls++;
    if(n<0){ fprintf(stderr,"poll=%d\n",n); status=1; break; }
    if(!n) empty_polls++;
    for(int i=0;i<n;i++){
      completion_ids[completed+(uint32_t)i]=completions[i].wr_id;
      if(completions[i].status!=IBV_WC_SUCCESS ||
         (!sending && completions[i].byte_len!=STREAM_BYTES)){
        fprintf(stderr,"completion index=%llu status=%u bytes=%u\n",
          (unsigned long long)completions[i].wr_id,completions[i].status,completions[i].byte_len);
        status=1;
      }
    }
    if(status) break;
    if(n){
      double observed=monotime();
      completed+=(uint32_t)n;
      if(!first){ first=observed; first_count=completed; }
      last=observed;
      if(!low_count && completed>=STREAM_PAGES/10){ low_count=completed; low_time=observed; }
      if(!high_count && completed>=9*STREAM_PAGES/10){ high_count=completed; high_time=observed; }
    }
    if(!(polls%4096) && monotime()>=deadline){ status=1; break; }
  }
  if(completed!=STREAM_PAGES) status=1;
  uint32_t correct_pages=0,correct_completed_pages=0,first_incorrect_page=STREAM_PAGES;
  uint32_t first_identity_mismatch=STREAM_PAGES;
  for(uint32_t i=0;i<completed;i++) if(completion_ids[i]!=i){ first_identity_mismatch=i; break; }
  if(first_identity_mismatch!=STREAM_PAGES) status=1;
  if(!sending) for(uint32_t page=0;page<STREAM_PAGES;page++){
    const uint64_t *words=(void*)(memory+(size_t)page*STREAM_BYTES);
    uint64_t incorrect=0;
    for(uint32_t word=0;word<STREAM_BYTES/sizeof *words;word++)
      incorrect+=words[word]!=(((uint64_t)page<<32)^word^UINT64_C(0x6d65736870616765));
    valid_pages[page]=!incorrect;
    correct_pages+=!incorrect;
    if(incorrect && first_incorrect_page==STREAM_PAGES) first_incorrect_page=page;
    mismatches+=incorrect;
  }
  if(!sending) for(uint32_t i=0;i<completed;i++)
    correct_completed_pages+=completion_ids[i]<STREAM_PAGES && valid_pages[completion_ids[i]];
  if(mismatches) status=1;
  printf("{\"mode\":\"provider_stream\",\"role\":\"%s\",\"bytes\":%zu,\"page_bytes\":%u,\"window_pages\":%u,\"posted\":%u,\"completed\":%u,\"status\":%d,\"mismatches\":%llu,\"correct_pages\":%u,\"correct_completed_pages\":%u,\"first_incorrect_page\":%u,\"first_identity_mismatch\":%u,\"first_mismatched_id\":%llu,\"submit_to_last_seconds\":%.9f,\"first_to_last_seconds\":%.9f,\"first_count\":%u,\"low_count\":%u,\"high_count\":%u,\"first_to_last_bytes\":%llu,\"interior_bytes\":%llu,\"interior_seconds\":%.9f,\"polls\":%llu,\"empty_polls\":%llu,\"posts\":%llu,\"poll_seconds\":%.9f,\"post_seconds\":%.9f}\n",
    argv[1],bytes,STREAM_BYTES,capacity,posted,completed,status,(unsigned long long)mismatches,
    correct_pages,correct_completed_pages,first_incorrect_page,first_identity_mismatch,
    (unsigned long long)(first_identity_mismatch<completed?completion_ids[first_identity_mismatch]:UINT64_MAX),
    last?last-begin:0,last?last-first:0,first_count,low_count,high_count,
    (unsigned long long)(completed-first_count)*STREAM_BYTES,(unsigned long long)(high_count-low_count)*STREAM_BYTES,
    high_count?high_time-low_time:0,(unsigned long long)polls,(unsigned long long)empty_polls,
    (unsigned long long)posts,poll_seconds,post_seconds); fflush(stdout);
  sigset_t blocked,previous; sigemptyset(&blocked); sigaddset(&blocked,SIGTERM); sigaddset(&blocked,SIGINT);
  sigprocmask(SIG_BLOCK,&blocked,&previous);
  while(!stop) sigsuspend(&previous);
  sigprocmask(SIG_SETMASK,&previous,NULL);
 storage:
  free(valid_pages); free(completion_ids); free(sges); free(sends); free(receives); free(completions);
 finish:
  while(!down_verbs()){}
  if(lsock>=0) close(lsock);
  munmap(memory,bytes);
  return status;
}

#import <Metal/Metal.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mesh-metal.h"

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7 */
static int state_area(uint64_t from,uint64_t at,uint64_t count,uint64_t size,uint64_t align,uint64_t len,uint64_t *end){
  if(at<from || at%align || at>len || count>(len-at)/size)return EINVAL;
  if(at+count*size>*end)*end=at+count*size;
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7 */
/* design/aot-compilation-change-request.md, appendix item 6 */
int mesh_metal_transport_create(struct mesh_ctx *context,uint64_t plan,void *device,struct mesh_metal_transport *transport){
  *transport=(struct mesh_metal_transport){0};
  struct hdr *m=context->M;
  const uint64_t at=plan,page=m->pgsz,len=context->len;
  if(!at || at%page || at<m->data_off || at>len || len-at<sizeof(struct mesh_state))return EINVAL;
  struct mesh_state *words=(struct mesh_state *)((char *)m+at);
  const uint64_t from=at+sizeof *words,copies=words->copies,ranges=words->ranges,word=sizeof(uint64_t);
  uint64_t end=from;
  if(!copies || state_area(from,words->status,1,sizeof(struct mesh_status),sizeof(uint32_t),len,&end) ||
     (ranges && state_area(from,words->phases,0,1,MESH_LANE_BYTES,len,&end)) ||
     (ranges && state_area(from,words->counts,ranges*copies,word,word,len,&end)) ||
     (words->releases && state_area(from,words->supplied,words->releases,word,word,len,&end)))
    return EINVAL;
  const uint64_t bytes=(end-at+page-1)/page*page;
  if(bytes>len-at)return EINVAL;
  id<MTLBuffer> state=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)m+at length:bytes
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!state)return ENOMEM;
  *transport=(struct mesh_metal_transport){.state=state,.words=words,.offset=at,.bytes=bytes};
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.7 */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint32_t transfer,
  struct mesh_metal_input *input){
  const struct mesh_publication *entry=mesh_transfer_at(context->M,transport->words,transfer);
  const uint64_t at=transport->offset;
  *input=(struct mesh_metal_input){.phases=entry->phases-at,.group=mesh_group_bytes(entry->pages),.count=entry->count-at,
    .pages=entry->pages,.copies=transport->words->copies,
    .time=((const double *)(const void *)((const char *)context->M+transport->words->times))[transfer]};
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7 */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint32_t transfer,
  struct mesh_metal_publication *publication){
  publication->supplied=mesh_transfer_at(context->M,transport->words,transfer)->targets[0].word-transport->offset;
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.6 */
#define MESH_METAL_SLICE_NS UINT64_C(10000000)
struct mesh_metal_waiter {
  pthread_t thread;
  void *event;
  char *words,*region;
  uint32_t passes;
  _Atomic uint32_t stop,freed;
  int cause;
  _Atomic uint64_t *_Atomic waiting;
  size_t waits;
  struct mesh_metal_wait wait[];
};

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.6 */
/* os_sync_wait_on_address.h os_sync_wait_on_address_with_timeout */
/* MPI 4.1 §3.9 and p. 110; rdma/mesh-dataflow.c mesh_transfers_start; D. P. Reed and R. K. Kanodia, CACM 22(2), 1979,
   p. 116; Linux v6.12 kernel/sched/completion.c:52-78, complete_all, and Documentation/scheduler/completion.rst:252;
   MPI Forum ULFM chapter draft, 2022, §16.3.1; W. Bland et al., IJHPCA 27(3), 2013, §5.4 */
static void *waits_run(void *argument){
  struct mesh_metal_waiter *waiter=argument;
  @autoreleasepool{
    id<MTLSharedEvent> event=(id<MTLSharedEvent>)waiter->event;
    const struct mesh_state *state=(const struct mesh_state *)(const void *)waiter->words;
    const struct mesh_submission *queue=(const struct mesh_submission *)(const void *)(waiter->region+state->submits);
    _Atomic uint64_t *plans=&((struct hdr *)(void *)waiter->region)->plans;
    const uint64_t passes=waiter->passes,room=state->submit_room;
    uint64_t since=0,set=0,last=0;
    const _Atomic uint64_t *read=NULL;
    int cause=0,freed=0;
    for(uint64_t t=0;!cause && !freed;t++){
      while(!cause){
        const uint32_t ended=atomic_load_explicit(&waiter->freed,memory_order_acquire);
        const uint64_t notice=atomic_load_explicit(plans,memory_order_acquire),
                       k=atomic_load_explicit(&state->submissions,memory_order_acquire);
        if(k && queue[(k-1)%room].end>t)break;
        atomic_store_explicit(&waiter->waiting,plans,memory_order_release);
        if(atomic_load_explicit(&waiter->stop,memory_order_acquire))cause=ECANCELED;
        else if(ended){freed=1;break;}
        else if(os_sync_wait_on_address_with_timeout(plans,notice,sizeof *plans,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                  OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_METAL_SLICE_NS)<0 && errno==EINVAL)
          cause=EINVAL;
      }
      for(size_t i=0;i<waiter->waits && !cause && !freed;){
        const struct mesh_metal_wait *wait=waiter->wait+i;
        const uint64_t value=t*passes+wait->pass+1,target=(t/wait->copies+1)*wait->pages;
        if(value-1>set){[event setSignaledValue:value-1];set=value-1;}
        _Atomic uint64_t *word=(_Atomic uint64_t *)(waiter->words+wait->count+sizeof(uint64_t)*(t%wait->copies));
        const uint64_t seen=atomic_load_explicit(word,memory_order_acquire);
        if(seen>=target){i++;continue;}
        const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
        if(word!=read || seen!=last){read=word;last=seen;since=now;}
        const uint64_t spent=now-since;
        const uint64_t left=spent<MESH_PLATFORM_NS?MESH_PLATFORM_NS-spent:0;
        atomic_store_explicit(&waiter->waiting,word,memory_order_release);
        if(atomic_load_explicit(&waiter->stop,memory_order_acquire))cause=ECANCELED;
        else if(!left)cause=ETIMEDOUT;
        else if(os_sync_wait_on_address_with_timeout(word,seen,sizeof *word,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                  OS_CLOCK_MACH_ABSOLUTE_TIME,left<MESH_METAL_SLICE_NS?left:MESH_METAL_SLICE_NS)<0 && errno==EINVAL)
          cause=EINVAL;
      }
      if(!cause && !freed && (t+1)*passes>set){[event setSignaledValue:(t+1)*passes];set=(t+1)*passes;}
    }
    if(cause)[event setSignaledValue:UINT64_MAX];
    waiter->cause=cause;
  }
  return NULL;
}

int mesh_metal_waits_start(struct mesh_metal_transport *transport,void *event,const struct mesh_metal_wait *waits,
  size_t count,uint32_t passes,struct mesh_metal_waiter **out){
  *out=NULL;
  if(!transport->state || !event || transport->bytes<sizeof(uint64_t))return EINVAL;
  const uint64_t last=transport->bytes-sizeof(uint64_t);
  for(size_t i=0;i<count;i++){
    const struct mesh_metal_wait *wait=waits+i;
    if(!wait->copies || !wait->pages || wait->pass>=passes || (i && wait->pass<waits[i-1].pass) ||
       wait->count%sizeof(uint64_t) || wait->count>last || wait->copies-1>(last-wait->count)/sizeof(uint64_t))
      return EINVAL;
  }
  if(count>(SIZE_MAX-sizeof(struct mesh_metal_waiter))/sizeof *waits)return ENOMEM;
  struct mesh_metal_waiter *waiter=malloc(sizeof *waiter+count*sizeof *waits);
  if(!waiter)return ENOMEM;
  waiter->event=[(id)event retain];
  waiter->words=(char *)transport->words;
  waiter->region=(char *)transport->words-transport->offset;
  waiter->passes=passes;
  waiter->cause=0;
  waiter->waits=count;
  atomic_init(&waiter->stop,0);
  atomic_init(&waiter->freed,0);
  atomic_init(&waiter->waiting,NULL);
  if(count)memcpy(waiter->wait,waits,count*sizeof *waits);
  const int status=pthread_create(&waiter->thread,NULL,waits_run,waiter);
  if(status){[(id)waiter->event release];free(waiter);return status;}
  *out=waiter;
  return 0;
}

void mesh_metal_waits_stop(struct mesh_metal_waiter *waiter){
  const int saved=errno;
  atomic_store_explicit(&waiter->stop,1,memory_order_release);
  _Atomic uint64_t *word=atomic_load_explicit(&waiter->waiting,memory_order_acquire);
  if(word)os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  errno=saved;
}

int mesh_metal_waits_end(struct mesh_metal_waiter *waiter){
  if(!waiter)return 0;
  atomic_store_explicit(&waiter->freed,1,memory_order_release);
  _Atomic uint64_t *word=atomic_load_explicit(&waiter->waiting,memory_order_acquire);
  if(word)os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  pthread_join(waiter->thread,NULL);
  const int cause=waiter->cause;
  [(id)waiter->event release];
  free(waiter);
  return cause;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 */
int mesh_metal_status(struct mesh_metal_transport *transport,int cause,uint64_t *first){
  if(first)*first=UINT64_MAX;
  struct mesh_state *words=transport->words;
  if(!words)return EINVAL;
  struct mesh_status *status=(struct mesh_status *)((char *)words+(words->status-transport->offset));
  const uint32_t ahead=atomic_load_explicit(&status->ahead,memory_order_acquire);
  const uint32_t behind=atomic_load_explicit(&status->behind,memory_order_acquire);
  const uint32_t failed=atomic_load_explicit(&status->first,memory_order_acquire);
  if(failed && first)*first=(uint32_t)~failed;
  if(behind)return cause?cause:ENOLINK;
  if(ahead)return EPROTO;
  return 0;
}

void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->state release];
  *transport=(struct mesh_metal_transport){0};
}

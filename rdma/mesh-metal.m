#import <Metal/Metal.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mesh-metal.h"

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7; design/disjoint-pages-revision-coordination.md,
   "The region" and "The GPU's side of the plan's words" */
/* One area of the state section: `count` items of `size` bytes at region offset `at`, after the section's struct
   (`from`), aligned to `align` and inside the region (`len`); the greatest end so far in *end.  EINVAL otherwise. */
static int state_area(uint64_t from,uint64_t at,uint64_t count,uint64_t size,uint64_t align,uint64_t len,uint64_t *end){
  if(at<from || at%align || at>len || count>(len-at)/size)return EINVAL;
  if(at+count*size>*end)*end=at+count*size;
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7; design/disjoint-pages-revision-coordination.md,
   "The region" and "The GPU's side of the plan's words" */
/* The plan's state section (mesh.h struct mesh_state at region offset `plan`, mesh-collective.h mesh_plan's `*plan`)
   wrapped once for the GPU, by address, with no copy: mesh_plan places it in the arena with the plan's other values,
   in whole blocks, its first byte block-aligned (mesh-collective.h, "The state section"), so the wrap starts at a page
   of the region and covers the plan's own pages.  No section is created and nothing is written.  Its length is taken
   as the greatest end of its struct, its counts (ranges x copies words), its supplied words (releases words) and its
   status (struct mesh_status), rounded up to the region's page: the phase list lies before the counts
   (mesh-collective.h, "The state section": the struct, the phase list, the counts, the supplied words, the status),
   and each received range's groups are checked to lie inside the wrap where the range is prepared
   (mesh_metal_receive_prepare).  An area with no items (no received range, no source range) is not read.  EINVAL
   where `plan` is 0, where the section's first byte is not at a page of the arena,
   where it holds no copy, or where an area lies before the struct's end, is misaligned for its loads
   (the phases at MESH_LANE_BYTES, the counts and supplied words at 8, the status at 4) or ends past the region (the
   context's `len`); ENOMEM where Metal wraps nothing.
   2026-11-04, sub-assignment 8: reversing sub-assignment 9's 2026-11-03 form, two buffers over the layout's supplied
   words (hdr word_off, word_capacity) and landed list (hdr landed_off, landed_capacity), and the program's
   cancellation, a wire-0 section mesh_section_create laid out at this call, one range of every execution's cancel word.
   Why: the layout's areas are gone (mesh.h 109); the state section is the plan's own pages (C3.2, C3.4), so nothing is
   laid out at this call; the cancel word is gone, a cancellation arrives on the counts and a stop is
   mesh_metal_waits_stop (C3.1).  What must stay: one wrap, by address, with no copy, its length whole pages.
   2026-11-04, owner L (design/compilation-coordination.md, "The region": "every function takes the plan from
   `ctx->state`, never from hdr"): the section the context's standing plan's, where it was hdr `state`, the region's
   one plan.  What must stay: as above.
   2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 6; design/compilation-coordination.md,
   "The appendix's revisions", N2, N5): the section the caller's `plan`, where it was the context's `state`, gone.
   Why: a program holds any number of compiled functions at once, each its own plan, and the transport wraps the one
   it names.  What must stay: as above. */
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

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.7; design/disjoint-pages-revision-coordination.md,
   "The region" (struct mesh_publication), D1 and "The GPU's side of the plan's words" */
/* A received range's words (a value mesh_plan placed with `count` u copies), from its first row's publication, which
   mesh_plan writes: `pages` P, `phases` the region offset of its copy-0 phase group (copy c's at phases + c x
   mesh_group_bytes(P)), `count` the region offset of its copy-0 count word (copy c's at count + 8 c, D1), and `state`
   the plan's state section.  The offsets are made relative to the transport's (its buffer's byte 0), and every copy's
   group and count word must lie inside the buffer, the groups at MESH_LANE_BYTES and the count words at 8, as the first
   pass's lane loads and the waiter's waits take them; `copies` is the plan's u (struct mesh_state).  EINVAL where the
   row names no receive (`pages` 0), where its `state` is not the transport's `offset` (another plan's row), or where a
   copy's words lie outside the buffer or are misaligned.  (2026-11-04, owner L: the context's `state`, where it was hdr
   `state`; design/compilation-coordination.md, "The region".)
   2026-11-04, owner L2 (design/compilation-coordination.md, "The appendix's revisions", N5): the row's `state` tested
   against the transport's `offset` alone, where it was that and the context's `state`, gone.  Why: the transport names
   its plan (mesh_metal_transport_create's `plan`), so a row of another plan is one whose `state` is not its offset.
   What must stay: EINVAL for a row no receive of the transport's plan names.
   2026-11-04, sub-assignment 8: reversing sub-assignment 9's 2026-11-03 form, the landed list's buffer retained with
   execution 0's landed record at `offset` and the records at hdr landed_stride, and the range's bits one run [first,
   first + count) of every page's row's publication (`device_input`, `device_stride`, `landed`).  Why: the landed
   records an execution are gone; a receive's state is a 2-bit phase a copy in the phase list, and its range's arrivals
   a count a copy (C3.7, D1); the first row names the range's words.  What must stay: one input a received range, as
   byte offsets in the one buffer the recorder binds; EINVAL for a row no receive of the plan names.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
   revisions of 14b9a757", R-1; mesh-metal.h and mesh.h struct mesh_state `transfers`, mesh_transfer_at at mesh
   31ee5d4a), reversing the range named by its first row's publication: the range's words are entry `transfer` of the
   transport's plan's `transfers`, the declared transfer's index among the plan's, which mesh_plan writes once ("where
   it is the transfer's destination, `pages` the range's instructions it receives a copy (P), `phases` the region
   offset of its copy-0 phase group ..., `count` the region offset of its copy-0 count word"), made relative to the
   transport's offset as before.  Why: "ranges may share a block where their bytes are disjoint, so a row names no one
   range" (mesh-metal.h).  The EINVALs go: the row past the region's rows, `pages` 0, the row's `state` against the
   transport's `offset`, a word outside the buffer or misaligned, and the transport not made.  The entry is the plan's
   own, at the place the plan fixed, its offsets the plan's inside its state section, and the caller names a transfer
   its IR makes this node the destination of (K2: "Run-time bound and null tests of facts the plan fixes go").  Nothing
   here is a device's or the system's to refuse, so it returns 0.  What must stay: one input a received range, as byte
   offsets in the one buffer the recorder binds, `copies` the plan's u */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint32_t transfer,
  struct mesh_metal_input *input){
  const struct mesh_publication *entry=mesh_transfer_at(context->M,transport->words,transfer);
  const uint64_t at=transport->offset;
  *input=(struct mesh_metal_input){.phases=entry->phases-at,.group=mesh_group_bytes(entry->pages),.count=entry->count-at,
    .pages=entry->pages,.copies=transport->words->copies};
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7; design/disjoint-pages-revision-coordination.md,
   "The region" (struct mesh_publication), D9 and "Publications" */
/* A source range's supplied word, from its first row's publication, which mesh_plan writes: `sends` 1, `targets[0]`
   {word: the region offset of its supplied word, count 1, stride 0} and `state` the plan's state section.  The
   offset is made relative to the transport's; the word lies inside the buffer, at 8.  Its one writer is the range's
   producer, which raises it by one after its last write of each execution (D9), through the recorder's binding of the
   buffer at this offset.  EINVAL where the row names no source range (`sends` not 1), where its `state` is not the
   transport's `offset`, or where the word lies outside the buffer or is misaligned.  (2026-11-04, owner L: the
   context's `state`, where it was hdr `state`; design/compilation-coordination.md, "The region".)
   2026-11-04, owner L2 (design/compilation-coordination.md, "The appendix's revisions", N5): the row's `state` tested
   against the transport's `offset` alone, where it was that and the context's `state`, gone.  What must stay: EINVAL
   for a row no source range of the transport's plan names.
   2026-11-04, sub-assignment 8: reversing sub-assignment 9's 2026-11-03 form, `ranges` sections' records (struct
   prepared_publication from mesh_publication_prepare, relative to the supplied words' area, at MESH_WORD_STRIDE an
   execution) in a section mesh_section_create laid out at this call and wrapped as `records`, and with `invocations`
   the statuses (struct mesh_output_status, one a range and invocation, seeded with the range's bytes) in a second such
   section wrapped as `completion`.  Why: the allocator places every value and nothing is laid out at its call, the
   GPU's publication records among them (C3.2); one supplied word serves every execution and only rises (C3.7); the
   recorder's restoring counter, a value the plan places, replaces the statuses (the contract, "Publications").  What
   must stay: one store of a range's word releases every send of the range's pages.
   2026-11-06 (R-1; "The revisions of 14b9a757", R-1; mesh-metal.h, mesh.h struct mesh_state `transfers` at mesh
   31ee5d4a), reversing the range named by its first row's publication: the supplied word is `targets[0]`'s of entry
   `transfer` of the transport's plan's `transfers` ("where this node is the transfer's source, `sends` 1, `targets[0]`
   {word: the region offset of its range's supplied word, count 1, stride 0}"), made relative to the transport's offset
   as before.  Why: as mesh_metal_receive_prepare's.  The EINVALs go, as there: the row past the region's rows, `sends`
   not 1, the row's `state`, the word outside the buffer or misaligned, the transport not made; the entry and its word
   are the plan's, and the caller names a transfer its IR makes this node the source of (K2).  It returns 0.  What
   must stay: one store of a range's word releases every send of the range's pages */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint32_t transfer,
  struct mesh_metal_publication *publication){
  publication->supplied=mesh_transfer_at(context->M,transport->words,transfer)->targets[0].word-transport->offset;
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.6; design/disjoint-pages-revision-coordination.md,
   D1, D7 and "The GPU's side of the plan's words" */
/* The longest one wait on a count word lasts: a stop whose wake comes between the thread's reading of `stop` and its
   wait is seen within it. */
#define MESH_METAL_SLICE_NS UINT64_C(10000000)
/* the waiter: one repetition's waits in nondecreasing pass, `passes` a repetition's, `repetitions` the run's, the state
   section's words (the transport's `words`), the event retained for the thread's life; `stop` the stop's one store,
   `waiting` the count word the thread waits on (the stop's wake) */
struct mesh_metal_waiter {
  pthread_t thread;
  void *event;
  char *words;
  uint64_t repetitions;
  uint32_t passes;
  _Atomic uint32_t stop;
  int cause;
  _Atomic uint64_t *_Atomic waiting;
  size_t waits;
  struct mesh_metal_wait wait[];
};

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.6; design/disjoint-pages-revision-coordination.md,
   D1 and D7 */
/* The client's signalling of the queue's event (C3.1: "The client alone signals the event: when the count reaches the
   step's target, or at the step's deadline. Its host thread waits on the count's address with a timeout"), counted
   completion as Portals 4's counting events and MTLSharedEvent's rising value.  The waits are one repetition's
   (mesh-metal.h struct mesh_metal_wait), each affine in the repetition t: the thread walks t = 0 .. repetitions - 1
   and, within t, the waits in their order (nondecreasing pass), wait i of t its count word count + 8 (t mod copies),
   its target (t / copies + 1) x pages and its value t x passes + pass + 1, so the values it meets do not decrease.  A
   wait is met once its count word (an _Atomic uint64_t that only rises, the bridge's arrivals: a landing adds one, a
   cancellation raises it to its final value) reads (acquire) at least its target.  Before a wait of value v, every
   value below v is met (each of its waits, or none has a wait: a pass that reads nothing), and the event is set to
   v - 1 where it stands lower, so every later pass held on it or less goes; after the last wait of the run, the event
   is set to the final value, repetitions x passes, the values past it having no wait.  While a count reads below its
   target, the thread waits on its
   address for the value it read (os_sync_wait_on_address_with_timeout, OS_SYNC_WAIT_ON_ADDRESS_SHARED, 8 bytes, as the
   bridge wakes it with OS_SYNC_WAKE_BY_ADDRESS_SHARED), for the least of MESH_METAL_SLICE_NS and the time left to the
   value's deadline, MESH_PLATFORM_NS after the previous value was set (after the start for the first).  Every return
   of the wait reads the count again (os_sync_wait_on_address.h: a wake or a changed value returns at once; ETIMEDOUT,
   the slice spent; EINTR, a spurious wake; EFAULT and ENOMEM, transient, to retry), so the loop ends at the target,
   the deadline or a stop; EINVAL (a word the system's wait refuses: misaligned, or its kernel state inconsistent with
   the flags or size) ends it at once rather than spin to the deadline.  The slice, 10 ms, is short enough that a stop
   whose wake is lost is seen promptly, and long enough that a deadline costs at most 500 returns.  At the deadline
   (ETIMEDOUT), a stop seen while a count reads below its target (ECANCELED) or EINVAL, the event is set to the final
   value at once, so every pass the command buffer holds goes and its later passes find their fields behind
   (mesh_metal_status), and the thread ends.  No value is set twice and none lowers the event: the values set rise,
   and the final value is the greatest.  Nothing the thread holds grows with the run: one repetition's waits and t.
   2026-11-04, sub-assignment 2 (review 10's finding 1, the coordinator's ruling): reversing sub-assignment 8's walk
   over every repetition's waits as given, sorted by value, each its own count word, target and value.  Why: that
   table was the run's, steps x passes x reads, though each wait is affine in t (C1, C3.3; design/disjoint-pages-
   change-request.md, D.3.7).  What must stay: each value set once every wait of that value or less is met, in order;
   the deadline MESH_PLATFORM_NS after the last value set (after the start for the first); a stop or a deadline setting
   the final value at once.
   2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-7:
   "Every wait ends at MESH_PLATFORM_NS without progress, and the first ETIMEDOUT ends the run ... e2b's step waits and
   mesh-metal.m's waiter take one bound each, not one per unit.  The operator's bound is judged by time without
   progress"; W10: "every wait ends at MESH_PLATFORM_NS without progress"; "The revisions of 14b9a757", R-7), reversing
   the deadline above, MESH_PLATFORM_NS after the last value set: a wait's bound restarts at each rise of the count it
   waits on, its progress, as mesh_transfers_start's waits end "at MESH_PLATFORM_NS after the last rise of its count it
   read" (mesh-call.h at mesh 31ee5d4a).  `read` and `last` are the count word last read and what it read; a read of
   another word (a wait's first) or a greater value (counts only rise) restarts `since`, and a wait ends ETIMEDOUT once
   MESH_PLATFORM_NS has passed since with no rise.  A value set restarts nothing: setting the event is the waiter's own
   write, not the count's progress.  The waiter's `start` goes with the first value's deadline counted from it.  The
   first ETIMEDOUT ends the walk, as every cause does, and the event goes to the final value at once.  mesh-metal.h's
   note on mesh_metal_waits_start ("a value not met within MESH_PLATFORM_NS of the last value set", "The first value's
   deadline counts from this call") is the form reversed here; the header is the coordinator's.  What must stay: each
   value set once every wait of that value or less is met, in order; a stop or a timeout setting the final value at
   once; one bound a wait, never a multiple */
static void *waits_run(void *argument){
  struct mesh_metal_waiter *waiter=argument;
  @autoreleasepool{
    id<MTLSharedEvent> event=(id<MTLSharedEvent>)waiter->event;
    const uint64_t passes=waiter->passes,final=waiter->repetitions*passes;
    uint64_t since=0,set=0,last=0;
    const _Atomic uint64_t *read=NULL;
    int cause=0;
    for(uint64_t t=0;t<waiter->repetitions && waiter->waits && !cause;t++)
      for(size_t i=0;i<waiter->waits && !cause;){
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
    if(set<final)[event setSignaledValue:final];
    waiter->cause=cause;
  }
  return NULL;
}

/* design/disjoint-pages-revision-coordination.md, "The GPU's side of the plan's words" */
/* One repetition's waits copied, once, and one thread started (waits_run).  Every copy's count word of a wait lies
   inside the transport's buffer, at 8 (count + 8 c for each c < copies); `copies` and `pages` are not 0; `pass` is
   below `passes` and does not decrease; the final value repetitions x passes is a uint64_t, and so is the last
   repetition's target, ((repetitions - 1) / copies + 1) x pages.  The event, an id<MTLSharedEvent>, stays the
   caller's: it is retained for the thread's life and released by mesh_metal_waits_end; the transport outlives the
   waiter.  Each wait's bound runs from its first read of its count, restarted at each rise (waits_run, 2026-11-06,
   R-7), where the first value's deadline counted from this call.  EINVAL where a wait is not of that form (`passes` 0
   with waits given among them), the event is NULL or the transport wraps nothing; ENOMEM; or pthread_create's
   error. */
int mesh_metal_waits_start(struct mesh_metal_transport *transport,void *event,const struct mesh_metal_wait *waits,
  size_t count,uint32_t passes,uint64_t repetitions,struct mesh_metal_waiter **out){
  *out=NULL;
  uint64_t final;
  if(!transport->state || !event || transport->bytes<sizeof(uint64_t) ||
     __builtin_mul_overflow(repetitions,(uint64_t)passes,&final))return EINVAL;
  const uint64_t last=transport->bytes-sizeof(uint64_t);
  for(size_t i=0;i<count;i++){
    const struct mesh_metal_wait *wait=waits+i;
    if(!wait->copies || !wait->pages || wait->pass>=passes || (i && wait->pass<waits[i-1].pass) ||
       wait->count%sizeof(uint64_t) || wait->count>last || wait->copies-1>(last-wait->count)/sizeof(uint64_t) ||
       (repetitions && (repetitions-1)/wait->copies+1>UINT64_MAX/wait->pages))
      return EINVAL;
  }
  if(count>(SIZE_MAX-sizeof(struct mesh_metal_waiter))/sizeof *waits)return ENOMEM;
  struct mesh_metal_waiter *waiter=malloc(sizeof *waiter+count*sizeof *waits);
  if(!waiter)return ENOMEM;
  waiter->event=[(id)event retain];
  waiter->words=(char *)transport->words;
  waiter->repetitions=repetitions;
  waiter->passes=passes;
  waiter->cause=0;
  waiter->waits=count;
  atomic_init(&waiter->stop,0);
  atomic_init(&waiter->waiting,NULL);
  if(count)memcpy(waiter->wait,waits,count*sizeof *waits);
  const int status=pthread_create(&waiter->thread,NULL,waits_run,waiter);
  if(status){[(id)waiter->event release];free(waiter);return status;}
  *out=waiter;
  return 0;
}

/* design/disjoint-pages-revision-coordination.md, "The GPU's side of the plan's words": the program's stop (in place of
   mesh_cancel), from a signal handler or any thread between mesh_metal_waits_start and mesh_metal_waits_end.
   Async-signal-safe: one atomic store, one atomic load, and one wake of the count word the thread last named, which is
   libsystem_platform's os_sync_wake_by_address_all, a wrapper of the __ulock_wake system call that takes no lock and
   allocates nothing (its disassembly, macOS 26.3); errno is kept.  A wake that comes before the thread's wait, or
   names a word it has left, is lost, and the thread sees the stop within MESH_METAL_SLICE_NS. */
void mesh_metal_waits_stop(struct mesh_metal_waiter *waiter){
  const int saved=errno;
  atomic_store_explicit(&waiter->stop,1,memory_order_release);
  _Atomic uint64_t *word=atomic_load_explicit(&waiter->waiting,memory_order_acquire);
  if(word)os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  errno=saved;
}

/* design/disjoint-pages-revision-coordination.md, "The GPU's side of the plan's words" */
/* The thread joined, which ends in bounded time (every value's wait ends at its deadline, a stop within a slice), the
   event released and the waiter freed: 0 where every value was set, ETIMEDOUT (a deadline), ECANCELED (a stop) or
   EINVAL (a count word the system's wait refused). */
int mesh_metal_waits_end(struct mesh_metal_waiter *waiter){
  if(!waiter)return 0;
  pthread_join(waiter->thread,NULL);
  const int cause=waiter->cause;
  [(id)waiter->event release];
  free(waiter);
  return cause;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1; design/disjoint-pages-revision-coordination.md, "The
   region" (struct mesh_status) */
/* The GPU's status after the run (struct mesh_status at the state section's `status`, the GPU's alone, read acquire):
   a later pass that found a field behind (a receive the bridge cancelled: its count reached its target with the field
   one phase behind) is `cause` where the caller gives one (the waiter's ETIMEDOUT or ECANCELED), else ENOLINK, whatever
   `ahead` holds; with none behind, one that found a field ahead of the allocator's order is EPROTO; none, 0.  *first is
   the first failed execution, ~first where the status's `first` (the greatest ~t over the failed passes) is not 0,
   else UINT64_MAX.
   2026-11-04, sub-assignment 8, the coordinator's ruling on owner 6's finding (mesh-flow.c 04b5e561): `behind` is read
   before `ahead`, where EPROTO came first.  Why: a cancelled receive's field is never written, so a reader a round
   later finds it two behind and counts it `ahead`, and three rounds later reads it as landed; after a cancellation
   `ahead` is the cancellation's.  What must stay: EPROTO where no pass found a field behind and one found it ahead.
   2026-11-04, sub-assignment 8: in place of the host's read of each execution's cancel word after the run (the GPU
   lane's UINT64_MAX past its bound among its writers).  Why: no lane waits, so none has a bound to reach; the later
   pass writes a failure into the status, a word only the GPU writes (C3.1).  What must stay: a run's failure read after
   the run as an error value, with its first execution. */
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

/* design/disjoint-pages-revision-coordination.md, "The GPU's side of the plan's words" */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->state release];
  *transport=(struct mesh_metal_transport){0};
}

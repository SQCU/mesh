#include "nccl.h"
#include "mesh-net.h"
#include "mesh-collective.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

/* libnccl-mesh (nccl.h): NCCL's calls as calls of the mesh's one collective planner, run on the bridge's
   communicator sessions (mesh-net.h).  A communicator is a clique key (the unique id's), this rank,
   the link table and each rank's node in it (ncclMeshConfig_t), the planner's map of the table's last
   snapshot over its ranks, and two connections each way with every rank that map links it to:
   channel 0 carries the collectives' steps, channel 1 the point-to-point calls, so that each keeps
   NCCL's matching rule (the k-th send on a connection is the peer's k-th receive on it) whatever the
   other does.  A collective is mesh_collective_choose's algorithm and this rank's steps from
   mesh_collective_plan, its operand and each REDUCE step's received piece in the bridge's registered
   window.
     The window's memory is handed out as allocations, each a record owned by the allocator (`heap`):
   its pages, the one Metal buffer over them (the caller's kernels and the library's programs bind the
   same object), and the completion points of its uses, the shared-event values after which its last
   writer and each reader since are done.  A group waits for the points of the allocations it touches
   (a buffer it reads for the writer; one it writes for the writer and the readers), and leaves its own
   completion there; the caller's GPU work declares its own (ncclMeshMemUse).  Freeing an allocation
   (ncclMemFree, or the closure that releases it, ncclMeshMemRelease) marks its record; its pages are
   given out again only once every recorded point's event has reached its value.
     A group ends by placing its operands, pieces, staged transfers and completion words in an
   allocation of its own and committing one GPU program (nccl-mesh-metal.m) to its stream's queue (the
   NULL stream's: the communicator's), or keeping it for the caller's command buffer (a deferred stream:
   nccl.h ncclMeshStreamDefer, the workers then waiting for the recorded points themselves, no gate): the
   recorded points waited for; the send buffers copied in by the copy kernel (memory outside the window
   read in place through a Metal buffer over its pages), or premultiplied; the gate word the workers
   start at; for each REDUCE chunk a leaky spin on its receive's completion word (and the earlier sends'
   reading its range), the combine kernel reading the piece where it landed and writing the operand in
   place where the spin saw them, and, where a later send reads it, a published word; the gates (struct
   gate: a command buffer beginning with a wait for an event the worker signals once the words are set);
   the combines the spins gave up on, the post-division and the results copied out; the group's completion.
   Every value the program reads is in mapped memory and every word it waits on is the bridge's or the
   worker's, so a late network makes it late, never different, and no command buffer runs on waiting for it.
   The bridge stores each isend's and irecv's end into its completion word (mesh.h), so the GPU waits on
   the network itself, no library thread between.  Each communicator's worker thread starts its part of a
   group in issue order once the gate word is set (or, without a program, the recorded points are reached):
   a SEND is an isend of the operand's piece (once the GPU has published its combine), a COPY an irecv into
   the operand, a REDUCE an irecv into its piece.  Its point-to-point transfers stay in flight while the
   worker goes on (a send completes only once its peer has posted the receive), and the part is done once
   both are; a part fails only where its bridge observed a failure (a peer that exited), and then sets every
   completion word of its calls the bridge has not, so its program goes on.  No library thread copies bytes
   on the CPU; what the library copied, sent, received and waited for is counted, a call each
   (ncclMeshGetCounts, ncclMeshGroupCounts). */

enum { CH_COLL, CH_P2P, CHANNELS };
enum { K_SEND=MESH_ALLGATHER+1, K_RECV, K_COUNTED };
#define P2P(k) ((k)->kind==K_SEND || (k)->kind==K_RECV)
/* a call's tallies: counts (also summed for the process), then its part's start and end times, when the
   worker saw the last of its pieces land and when it saw its send buffer's bytes ready (a persistent
   call's cut, or its first published range) */
enum { CPU_COPY, GPU_COPY, GPU_KERNELS, HOST_WAITS, INPUT_WAITS, SENT, RECEIVED, GPU_EVENT_WAITS, GPU_WORD_WAITS, HOST_WORD_WAITS, COMMITS,
       WAKEUPS, BUFFERS, SENDS, GRANT_WAITS, COUNTS, STARTED=COUNTS, ENDED, ARRIVED, READIED, DONE, TALLIES };
enum { KERNEL_COMBINE, KERNEL_PREMULTIPLY, KERNEL_POSTDIVIDE };
#define UID_MAGIC 0x4d4e434cu
#define FLIGHTS 32
struct uid { uint32_t magic,version; uint64_t key; };
_Static_assert(sizeof(struct uid)<=NCCL_UNIQUE_ID_BYTES,"unique id");

struct peer { int dev; void *send[CHANNELS],*recv[CHANNELS]; };
struct op_entry { int used; ncclDataType_t type; unsigned char scalar[8]; };
/* A REDUCE step's received piece in the window. */
struct piece { unsigned char *at; };
/* Where the program reads or writes a buffer: a Metal buffer and the address at its offset 0; `span`
   the window allocation's record (its first byte) it lies in, NULL for memory outside the window (the
   buffer then over the host pages holding it). */
struct region { void *buffer; const unsigned char *base; unsigned char *span; };
static uint64_t off(struct region r,const void *p){return (uint64_t)((const unsigned char *)p-r.base);}
struct call {
  int kind; const void *send; void *recv; size_t count; ncclDataType_t type; int op,root,peer;
  struct ncclComm *comm; struct ncclMeshStream *stream; uint32_t how; int force; uint64_t epoch;
  /* its place among the collective calls issued on the communicator since its last agreement, from 1 (a
     point-to-point call takes the next collective's: ranks issue different numbers of them), and that agreement */
  uint64_t index,agreement;
  /* a reduce-scatter's or all-gather's count a rank, the call's own copy (NULL: `count` each); a counted
     all-to-all's (K_COUNTED, ncclMeshAlltoAllCounted) segment lengths, the sent then the received, a rank
     each, its counts, where the received ones go on the host and the word set as they have, its rows of
     `row` elements sent and the most received, and its counts' place */
  uint64_t *segments;
  const int64_t *counted; int64_t *received,*landed; uint64_t *arrived; uint64_t row,rows,capacity; struct region cnt,lnd;
  /* resolved when the group ends */
  int combine,premultiply,postdivide; unsigned char scalar[8];
  uint64_t elements; struct mesh_collective chosen; struct mesh_step *steps; struct piece *pieces; uint32_t nsteps;
  /* placed when the group ends: send's and recv's regions; the operand in the window (recv itself where
     it is an allocation holding the whole result, else the group's) and its region; a point-to-point
     call's bytes in the window (its buffer, or the group's allocation where the buffer lies outside);
     a receive from this rank itself that the program copies; its completion words in the group's
     allocation, one each isend and irecv the worker posts for it in plan order (chunk by chunk), which
     the bridge sets, and, where a later send of the call reads a combine, one each REDUCE chunk, which
     the GPU publishes; after them its spins' flags (one a request, by the request's place: the spin before a
     REDUCE chunk's combine is the chunk's receive's), its leaked word (a spin gave up: its later spins give up at
     once) and, where its combines are gated chunk by chunk (a later send reads one), each chunk's landing word */
  struct region in,out,at; unsigned char *operand,*wire;
  /* fresh: the send buffer is read in place (nccl.h: no copy into the operand) */
  int copied,fresh;
  _Atomic uint64_t *words,*combined,*flags,*landings; uint32_t nwords,ncombined;
  uint64_t *tally;
  /* a persistent call's (ncclMeshPersistentBegin): its requests in plan order as posted this iteration (NULL:
     not yet), the next of its isends and of its irecvs to post ahead (plan order: `spost`, `rpost`), and
     whether its own part has posted its irecvs (`posted`); its isends are all posted ahead (persistent_post),
     held until their bytes are ready; `ahead` where every receive of its plan lands where nothing reads or
     writes before the call in its iteration (a REDUCE step's piece, or a COPY step's range of an operand that
     is the group's own or a receive buffer the caller declared free, which none of its sends reads), so its
     irecvs are posted ahead too (a receive buffer free only after a cut of the iteration, `post_after`, once
     that cut is passed), else by its part; `ready`,
     where the caller's GPU work publishes its send buffer in ranges of `range` bytes (`ranges` of them,
     ncclMeshPersistentNext): each range's word, counted up once each iteration, and its count when the run
     started.  A persistent point-to-point call has one request, an isend held until its one range's word has
     counted up (`ready`), or an irecv whose end the bridge stores into the caller's landed word `word`
     (ncclMeshPersistentLanded), its `words` */
  void **handles; uint32_t spost,rpost; int posted,ahead,post_after;
  const _Atomic uint64_t *ready; uint64_t range,*ready_base; uint32_t ranges;
  uint64_t *word;
  /* a persistent point-to-point call in slots (ncclMeshPersistentSlots): its buffer, ready or landed word and tally
     at slot 0 (`wire`, `ready`, `word` and `tally` then the open iteration's: persistent_open), its slots' buffers
     `slot_stride` bytes apart, `slot_depth` of them, its words `slot_words` bytes apart (one a slot, written once),
     and a tally a buffer slot */
  uint64_t slot_depth,slot_stride,slot_words;
  unsigned char *wire0; uint64_t *word0; const _Atomic uint64_t *ready0; uint64_t *tallies;
};
/* An event's value: the point after which a use of an allocation is done. */
struct point { void *event; uint64_t value; };
struct mark { struct ncclMeshStream *stream; uint64_t done; };
struct tally { _Atomic int refs; int n; uint64_t counts[][TALLIES]; };
struct persistent;
/* A gate of a group's program (the GPU's work that waits on the network, after its leaky spins): the
   communicator whose worker opens it, the completion words it waits for (the work after it needs them set: every
   word of the group's calls on that communicator, or a chunk's receive and the earlier sends of its range), the
   event and value the gate's command buffer waits for (a persistent call's: its run's, value + i G + index + 1 at
   iteration i, G the recording's gates, `index` its place among them, given when the recording plays it), and the
   landing word the spin before it waits for too, set once the event is signalled (so a GPU that saw the words finds
   the gate open).  The worker signals a communicator's gates in the order of their values, each once every word
   of it and of every gate before it is set (set 2: failed, so the program goes on). */
struct gate { struct ncclComm *comm; _Atomic uint64_t **words; uint32_t nwords; _Atomic uint64_t *landing; void *event; uint64_t value;
  struct persistent *run; int index,ready,armed; };
/* A group: its streams' marks; its program's event and last value, and the gate word the workers start at
   (NULL: none); without a program, the recorded points the workers wait for; its window allocation; its
   program's gates (a persistent call's the launch's; another's its own once armed, freed once opened: its part,
   and the launch, may end first). */
struct launch { pthread_mutex_t lock; pthread_cond_t cond; int items,sync,nmarks,nwaits,persistent,ngates; ncclResult_t result; struct mark *marks;
  void *event; uint64_t end; _Atomic uint64_t *gate,*landing; struct tally *tally; struct point *waits; unsigned char *stage,*placed; struct region own;
  struct gate **gates; uint64_t zero_at,zero_words; };
struct flight;
/* A communicator's part of a group: `run` a persistent call's (nccl.h ncclMeshPersistentBegin), which the
   worker starts again each iteration and never frees, `cut` the place of its cut among the recording's
   (0: none, its send buffer published in ranges), `fresh` its receive buffer untouched before it in its
   iteration (ncclMeshPersistentNext).  Nothing bounds it in time: it ends once its transfers have, or fails
   where the bridge observed the failure (a peer's exit, a comm closed). */
struct item { struct call *calls; int n; struct launch *launch; struct item *next;
  int collectives_done,networked; struct flight *flight; ncclResult_t result; struct persistent *run; int cut,fresh; };
/* A plan kept for (epoch, call signature): the algorithm chosen, this rank's steps and the whole plan's
   hash; its segments (a rank each) and steps (MESH_COLLECTIVE_STEPS of the ranks) sized with the
   communicator. */
#define PLANS 64
struct plan { uint64_t epoch,elements,*segments; int kind,type,root,force,uneven; uint32_t how,nsteps;
  struct mesh_collective chosen; struct mesh_step *steps; };
struct ncclComm {
  uint64_t key; int rank,nranks;
  /* the link table and rank r's node in it; the snapshot the calls plan on (`epoch` its epoch), the planner's
     map of it over the ranks (the links stated and up) and the stated map (the links stated, up or not: a call
     no up link carries is planned there, and waits for its links' sessions to resume); the plans */
  struct mesh_link_table *table; uint32_t *nodes; uint64_t epoch; struct mesh_link_contents *seen;
  struct mesh_link_map map,stated; uint32_t (*pairs)[2],(*stated_pairs)[2]; float (*cost)[2]; struct plan *plans;
  /* the worker's: the receive connections a collective used (a rank each), flushed at its end */
  void **used;
  /* alive: its connections found alive since the epoch last moved */
  void *net; struct peer *peers; int alive;
  uint64_t splits;
  struct op_entry *ops; int nops;
  /* its own queue and that queue's event, the value reserved on it; the programs not yet run; its gates' event
     (landed), the value the next gate takes, and the gates armed (by value; gates_lock) */
  void *queue,*event; uint64_t event_value;
  _Atomic int programs;
  void *landed; uint64_t landed_next;
  pthread_mutex_t gates_lock; struct gate **armed; int narmed,carmed;
  /* woke: the worker came out of its condition variable (parked) with a part to start; agreeing: an
     agreement is making the connections again (the worker leaves them alone) */
  pthread_t worker; int started,stopping,finalized,parked,woke,agreeing;
  /* the persistent calls the worker runs (ncclMeshPersistentStart), or NULL */
  struct persistent *run;
  pthread_mutex_t lock; pthread_cond_t cond;
  struct item *head,*tail; int busy;
  struct flight *flights; int nflights,*blocked;
  /* gpu_failed: a GPU program of this communicator failed and could not be run again (its inputs not retained):
     its call failed, the communicator revoked until the next agreement */
  _Atomic int aborting,broken,async,gpu_failed;
  char error[512];
  /* recovery (ncclMeshCommAgree): the calls issued since the last agreement, the index of the first
     that failed (0: none) and its cause; the agreements made; the members of the last one (`view`: a hash of the
     ranks that voted) and the agreements made in a row with that view (`in_view`), alike on the ranks of one view,
     which key an agreement's connections, so ranks whose histories differ (a rank that agreed alone while its node
     was gone) meet once their views are one; the slots of the connections closed since, and the event a failed
     part's memory is held on until the bridge has vacated them (`quiet`: the value it is held for); the ranks that
     voted in the last agreement (`voted`: 1 a rank, 0 one departed); each rank's process as its connections named
     it (`owner`, mesh-net.h mesh_net_peer_owner: a rank whose process its node's bridge no longer reports departed) */
  _Atomic uint64_t issued; uint64_t failed,agreements,view,in_view; char cause[256]; unsigned char *voted;
  uint64_t *closed,*owner; int nclosed,open; void *quiet; uint64_t quiet_value;
};

#define HIDDEN __attribute__((visibility("hidden")))
HIDDEN int nccl_mesh_gpu_attach(char *error,size_t size);
HIDDEN void *nccl_mesh_buffer(const void *pointer,size_t bytes,uint64_t *offset);
HIDDEN void *nccl_mesh_buffer_owned(const void *pointer,size_t bytes,uint64_t options,void (*gone)(void *),void *argument);
HIDDEN void *nccl_mesh_queue_create(void);
HIDDEN void nccl_mesh_retain(void *object);
HIDDEN void nccl_mesh_release(void *object);
HIDDEN void *nccl_mesh_event_create(void *queue);
HIDDEN uint64_t nccl_mesh_event_value(void *event);
HIDDEN void nccl_mesh_event_signal(void *event,uint64_t value);
HIDDEN void *nccl_mesh_program_begin(void *queue);
HIDDEN void nccl_mesh_program_wait(void *program,void *event,uint64_t value);
HIDDEN void nccl_mesh_program_signal(void *program,void *event,uint64_t value);
HIDDEN void nccl_mesh_program_kernel(void *program,int kernel,void *to,uint64_t dst,void *from,uint64_t src,uint64_t n,int type,int op,int nranks,uint64_t scalar,
  int published,void *other,uint64_t at,void *pred,uint64_t pred_at,int want,uint64_t grid);
HIDDEN void nccl_mesh_program_copy(void *program,void *to,uint64_t dst,void *from,uint64_t src,uint64_t bytes,int received,int published,
  void *pred,uint64_t pred_at,int want,uint64_t grid);
HIDDEN int nccl_mesh_program_spin(void *program,void *buffer,const uint64_t *at,uint32_t n,uint64_t flag,uint64_t leaked,void *given,
  void *const *touch,int ntouch,void **held);
HIDDEN void nccl_mesh_program_publish(void *program,void *buffer,uint64_t at,uint64_t value,void *pred,uint64_t pred_at,int want);
HIDDEN void nccl_mesh_program_zero(void *program,void *buffer,uint64_t at,uint64_t n);
HIDDEN void nccl_mesh_program_counted(void *program,void *to,void *from,void *counts,void *got,void *landed,const uint64_t *a,uint32_t na,uint64_t units,
  void *pred,uint64_t grid);
HIDDEN void nccl_mesh_program_end(void *program);
HIDDEN void nccl_mesh_program_handler(void *program,void (*done)(void *,int),void *argument);
HIDDEN void nccl_mesh_program_commit(void *program,void (*done)(void *,int),void *argument);
HIDDEN void nccl_mesh_program_drop(void *program);

/* ---- errors ---- */
static _Thread_local char last_error[512];
static void fail_message(struct ncclComm *c,const char *format,...){
  char text[512];va_list list;va_start(list,format);vsnprintf(text,sizeof text,format,list);va_end(list);
  snprintf(last_error,sizeof last_error,"%s",text);
  if(c){pthread_mutex_lock(&c->lock);snprintf(c->error,sizeof c->error,"%s",text);pthread_mutex_unlock(&c->lock);}
  if(getenv("MESH_NCCL_DEBUG"))fprintf(stderr,"nccl-mesh: %s\n",text);
}
#define FAIL(c,result,...) (fail_message((c),__VA_ARGS__),(ncclResult_t)(result))
/* The observed event behind a transport error, in words (mesh-flow.c: a peer of another instance ESTALE, a peer that
   left EHOSTDOWN, a peer client that exited ESRCH). */
static const char *event_of(int error){
  return error==ESTALE?"its peer's bridge paired again as another instance":error==EHOSTDOWN?"its peer's bridge left the mesh":
         error==ESRCH?"its peer's process exited":strerror(error);
}
static ncclResult_t net_failure(struct ncclComm *c,int result,const char *what){
  int error=mesh_net_error();
  return FAIL(c,result?result:ncclSystemError,"%s: %s (%s)",what,ncclGetErrorString((ncclResult_t)result),event_of(error));
}

static uint64_t now_ns(void){return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);}
static uint64_t mix(uint64_t x){
  x+=UINT64_C(0x9e3779b97f4a7c15);
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}

/* ---- counts: the process's, and each call's in its group's tally ---- */
static _Atomic uint64_t totals[COUNTS];
static void count(struct call *k,int what,uint64_t amount){
  if(!amount)return;
  atomic_fetch_add_explicit(&totals[what],amount,memory_order_relaxed);
  if(k && k->tally)k->tally[what]+=amount;
}
static void tally_release(struct tally *t){if(t && atomic_fetch_sub(&t->refs,1)==1)free(t);}
/* Gates opened (struct gate), and a word of the window each spin that gives up counts itself in (the process's
   first program makes it), counts of the process's as the others. */
static _Atomic uint64_t opened;
static struct { void *buffer; _Atomic uint32_t *word; } gave_up;
/* An evaluation's end (a part of a group, finish): the bridge's ring (mesh.h) takes every link's counts and the
   process's (totals, then the gates opened), which only a reader K evaluations on sees (ncclMeshStats). */
static void stats_record(void){
  uint64_t client[MESH_STATS_CLIENT]={0};
  for(int i=0;i<COUNTS;i++)client[i]=atomic_load_explicit(&totals[i],memory_order_relaxed);
  client[COUNTS]=atomic_load_explicit(&opened,memory_order_relaxed);
  client[COUNTS+1]=gave_up.word?atomic_load_explicit(gave_up.word,memory_order_relaxed):0;
  mesh_net_stats_record(client,MESH_STATS_CLIENT);
}

/* ---- datatypes and operators ---- */
static const size_t type_bytes[ncclNumTypes]={1,1,4,4,8,8,2,4,8,2,1,1};
static int integral(ncclDataType_t t){return t<=ncclUint64;}

/* A double rounded to nearest even into bfloat16 or an fp8 format (their scalars: ncclAvg's 1/nranks);
   fp8 saturates to its largest finite value as NCCL's __NV_SATFINITE does. */
struct small { int e,m,fn,saturate; };
static const struct small BF16={8,7,0,0},E4M3={4,3,1,1},E5M2={5,2,0,1};
static uint32_t small_encode(double x,struct small f){
  uint32_t sign=signbit(x)?1u<<(f.e+f.m):0,top=(1u<<f.e)-1;
  int bias=(1<<(f.e-1))-1;
  uint32_t largest=f.fn?(top<<f.m)|((1u<<f.m)-2):((top-1)<<f.m)|((1u<<f.m)-1);
  if(isnan(x))return f.fn?sign|(top<<f.m)|((1u<<f.m)-1):sign|(top<<f.m)|(1u<<(f.m-1));
  double a=fabs(x);
  if(isinf(a))return f.saturate?sign|largest:sign|(top<<f.m);
  if(a==0)return sign;
  int exponent;frexp(a,&exponent);exponent-=1;
  if(exponent<1-bias)exponent=1-bias;
  double q=nearbyint(ldexp(a,f.m-exponent));
  if(q>=ldexp(1.0,f.m+1)){q=ldexp(1.0,f.m);exponent++;}
  uint32_t field=q<ldexp(1.0,f.m)?0:(uint32_t)(exponent+bias),mantissa=(uint32_t)q&((1u<<f.m)-1);
  uint32_t bits=(field<<f.m)|mantissa;
  if(field>top || (!f.fn && field==top) || (f.fn && field==top && mantissa==(1u<<f.m)-1) || bits>largest)
    return f.saturate?sign|largest:sign|(top<<f.m);
  return sign|bits;
}
/* 1/nranks rounded to the type (ncclAvg's premultiplier on floating types: 1/n as a double, then
   the type); the GPU's premultiply kernel applies it. */
static void reciprocal(unsigned char *scalar,ncclDataType_t t,int nranks){
  double v=1.0/nranks;
  memset(scalar,0,8);
  if(t==ncclFloat32){float k=(float)v;memcpy(scalar,&k,4);}
  else if(t==ncclFloat64)memcpy(scalar,&v,8);
  else if(t==ncclFloat16){_Float16 k=(_Float16)v;memcpy(scalar,&k,2);}
  else{uint32_t bits=small_encode(v,t==ncclBfloat16?BF16:t==ncclFloat8e4m3?E4M3:E5M2);memcpy(scalar,&bits,type_bytes[t]);}
}

/* ---- the link map over a communicator's ranks ---- */
static int linked(const struct mesh_link_map *map,int a,int b){
  if(a==b || a<0 || b<0 || (uint32_t)a>=map->nodes || (uint32_t)b>=map->nodes)return 0;
  if(map->kind==MESH_LINKS_MESH)return 1;
  for(uint32_t l=0;l<map->links;l++)
    if(((int)map->link[l][0]==a && (int)map->link[l][1]==b) || ((int)map->link[l][0]==b && (int)map->link[l][1]==a))return 1;
  return 0;
}
/* The algorithm a collective takes on `map`: its selection, else (not forced) any; MESH_UNAVAILABLE where
   none carries it. */
static struct mesh_collective choose(const struct mesh_link_map *map,const struct call *k,struct mesh_operand operand){
  struct mesh_collective wanted={.what=(uint32_t)k->kind,.how=k->how,.root=(uint32_t)k->root,.segments=k->segments};
  struct mesh_collective chosen=mesh_collective_choose(map,wanted,operand,0,0);
  if(chosen.how==MESH_UNAVAILABLE && k->how && !k->force){wanted.how=0;chosen=mesh_collective_choose(map,wanted,operand,0,0);}
  chosen.segments=k->segments;
  return chosen;
}
/* The planner's maps of the snapshot c->seen: the links stated and up, and the links stated (every one taken as
   up: a link its bridge has not observed up yet, or lost, whose session resumes). */
static void maps(struct ncclComm *c){
  mesh_link_table_map(c->seen,c->nodes,(uint32_t)c->nranks,&c->map,c->pairs,c->cost);
  const size_t n=(size_t)c->table->nodes;
  struct mesh_link_contents *all=mesh_link_contents_new(c->table->nodes);
  if(!all){c->stated=c->map;return;}
  memcpy(all,c->seen,mesh_link_contents_bytes(c->table->nodes));
  for(size_t i=0;i<n*n;i++)all->link[i].up=all->link[i].stated;
  mesh_link_table_map(all,c->nodes,(uint32_t)c->nranks,&c->stated,c->stated_pairs,c->cost);
  free(all);
}
/* The communicator's link table and nodes from its config (ncclMeshConfig_t, recognized by its size),
   each node distinct and present in the table: no table, no communicator, and every rank refuses alike
   (the refusal reads only the config and the table's stated contents).  Its arrays are sized here, by
   its ranks and the table's nodes (the bridge's configuration), the only bounds. */
static ncclResult_t links_for(struct ncclComm *c,const ncclConfig_t *config){
  const ncclMeshConfig_t *mesh=config && config->size==sizeof(ncclMeshConfig_t)?(const ncclMeshConfig_t *)config:NULL;
  if(!mesh || !mesh->links || !mesh->nodes)
    return FAIL(c,ncclInvalidUsage,"no link map: the communicator takes an ncclMeshConfig_t (base.size sizeof(ncclMeshConfig_t)) naming the link "
                "table (ncclMeshLinksAttach) and each rank's node in it%s",mesh?(mesh->links?"; its nodes are NULL":"; its links are NULL"):"");
  c->table=mesh->links;
  const size_t n=(size_t)c->nranks,steps=MESH_COLLECTIVE_STEPS(c->nranks);
  c->nodes=calloc(n,sizeof *c->nodes);
  c->voted=calloc(n,1);
  c->owner=calloc(n,sizeof *c->owner);
  c->seen=mesh_link_contents_new(c->table->nodes);
  c->pairs=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->pairs);
  c->stated_pairs=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->stated_pairs);
  c->cost=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->cost);
  c->plans=calloc(PLANS,sizeof *c->plans);
  for(int p=0;c->plans && p<PLANS;p++){c->plans[p].segments=calloc(n,sizeof *c->plans[p].segments);c->plans[p].steps=calloc(steps,sizeof *c->plans[p].steps);}
  for(int p=0;c->plans && p<PLANS;p++)if(!c->plans[p].segments || !c->plans[p].steps)return FAIL(c,ncclSystemError,"allocation");
  c->used=calloc(n,sizeof *c->used);
  if(!c->nodes || !c->voted || !c->owner || !c->seen || !c->pairs || !c->stated_pairs || !c->cost || !c->plans || !c->used)
    return FAIL(c,ncclSystemError,"allocation");
  c->epoch=mesh_link_table_read(c->table,c->seen);
  for(int r=0;r<c->nranks;r++){
    int v=mesh->nodes[r];
    for(int q=0;q<r;q++)if(mesh->nodes[q]==v)return FAIL(c,ncclInvalidUsage,"no link map for rank %d: node %d is rank %d's too",r,v,q);
    if(v<0 || (uint32_t)v>=c->table->nodes || !mesh_link_present(c->seen)[v])
      return FAIL(c,ncclInvalidUsage,"no link map for rank %d: its node %d is not stated in the link table of %u nodes (ncclMeshLinksState)",
                  r,v,c->table->nodes);
    c->nodes[r]=(uint32_t)v;
  }
  maps(c);
  return ncclSuccess;
}

/* ---- connections ---- */
static uint64_t listen_key(uint64_t key,int channel,int from,int to){
  return mix(key^mix(((uint64_t)channel<<48)|((uint64_t)(uint32_t)from<<24)|(uint32_t)to))|1;
}
/* Every connection of `peers` closed; with `keep`, its slot kept (c->closed) until the bridge has
   vacated it (the agreement waits for that). */
static void close_peers(struct ncclComm *c,struct peer *peers,int keep){
  for(int p=0;peers && p<c->nranks;p++)for(int ch=0;ch<CHANNELS;ch++){
    void **end[2]={&peers[p].send[ch],&peers[p].recv[ch]};
    for(int e=0;e<2;e++){
      if(!*end[e])continue;
      if(keep && c->nclosed<c->nranks*CHANNELS*4)c->closed[c->nclosed++]=mesh_net_slot(*end[e]);
      if(e)mesh_net_close_recv(*end[e]);else mesh_net_close_send(*end[e]);
      *end[e]=NULL;
    }
  }
  if(peers==c->peers)c->open=0;
}
/* Whether rank p departed as this rank's bridge observes it: its node's bridge left the mesh (or this node's did, or
   this process's region was replaced), or its process exited (absent from its node's client table as that node's
   bridge last reported it: mesh-net.h mesh_net_exited).  Observed events only; a node merely silent is late. */
static int departed(struct ncclComm *c,int p){
  return p!=c->rank && (mesh_net_departed(c->nodes[p]) || mesh_net_exited(c->nodes[p],c->owner[p]));
}
/* Each rank's process as the connections of `peers` name it. */
static void owners(struct ncclComm *c,const struct peer *peers){
  for(int p=0;p<c->nranks;p++){
    const uint64_t owner=peers[p].recv[CH_P2P]?mesh_net_peer_owner(peers[p].recv[CH_P2P]):0;
    if(owner)c->owner[p]=owner;
  }
}
/* Both channels each way with every rank `want` marks, into `peers`, on whichever of the bridge's
   links reaches it: a listen on every link for each such rank's connections (keys both ends derive
   from the clique key `key`), a connect on every link to each such rank's; the one its bridge accepts
   is the link.  It waits until each is connected or departed (marked in `gone`, where given, and no longer
   wanted): no clock ends it.  The probing connects that met no listen are withdrawn with their context. */
static ncclResult_t connect_ranks(struct ncclComm *c,uint64_t key,unsigned char *want,struct peer *peers,unsigned char *gone){
  for(int p=0;p<c->nranks;p++)peers[p].dev=-1;
  int wanted=0;
  for(int p=0;p<c->nranks;p++)wanted+=want[p];
  if(!wanted)return ncclSuccess;
  struct mesh_link_view views[64];uint32_t node;
  int links=mesh_observe(getenv("MESH_REGION"),views,64,&node);
  if(links<0)return FAIL(c,ncclSystemError,"the bridge region %s: %s",getenv("MESH_REGION")?getenv("MESH_REGION"):"/mesh0",strerror(-links));
  if(!links)return FAIL(c,ncclInvalidUsage,"this node's bridge has no links, and rank %d is linked to %d ranks",c->rank,wanted);
  if(links>64)links=64;
  void *probe=NULL;
  int result=mesh_net_init(&probe,c->key,NULL,NULL,NULL);
  if(result)return net_failure(c,result,"mesh_net_init");
  size_t count=(size_t)c->nranks*CHANNELS*(size_t)links;
  void **listens=calloc(count,sizeof *listens);
  struct mesh_net_handle *targets=calloc(count,sizeof *targets);
  ncclResult_t status=listens && targets?ncclSuccess:FAIL(c,ncclSystemError,"connection table allocation");
  for(int p=0;p<c->nranks && !status;p++)if(want[p])for(int ch=0;ch<CHANNELS && !status;ch++)for(int l=0;l<links && !status;l++){
    size_t at=((size_t)p*CHANNELS+(size_t)ch)*(size_t)links+(size_t)l;
    unsigned char handle[MESH_NET_HANDLE_BYTES]={0};
    struct mesh_net_handle named={MESH_NET_HANDLE_MAGIC,0,listen_key(key,ch,p,c->rank)};
    memcpy(handle,&named,sizeof named);
    if((result=mesh_net_listen(c->net,l,handle,listens+at)))status=net_failure(c,result,"mesh_net_listen");
    targets[at]=(struct mesh_net_handle){MESH_NET_HANDLE_MAGIC,views[l].peer,listen_key(key,ch,c->rank,p)};
  }
  for(int done=0;!status && !done;){
    done=1;
    for(int p=0;p<c->nranks && !status;p++)if(want[p] && departed(c,p)){
      /* departed while waiting: no longer wanted, its connections made so far closed */
      want[p]=0;if(gone)gone[p]=1;
      for(int ch=0;ch<CHANNELS;ch++){
        if(peers[p].send[ch]){mesh_net_close_send(peers[p].send[ch]);peers[p].send[ch]=NULL;}
        if(peers[p].recv[ch]){mesh_net_close_recv(peers[p].recv[ch]);peers[p].recv[ch]=NULL;}
      }
    }
    for(int p=0;p<c->nranks && !status;p++)if(want[p])for(int ch=0;ch<CHANNELS && !status;ch++){
      struct peer *e=peers+p;
      /* one made in a session that has since ended (a bridge resuming from a stop takes the connects its
         socket still holds, then finds the session over) is withdrawn and made again on the next */
      if(e->send[ch] && !mesh_net_alive(e->send[ch])){mesh_net_close_send(e->send[ch]);e->send[ch]=NULL;}
      if(e->recv[ch] && !mesh_net_alive(e->recv[ch])){mesh_net_close_recv(e->recv[ch]);e->recv[ch]=NULL;}
      for(int l=0;l<links && !e->send[ch] && !status;l++){
        size_t at=((size_t)p*CHANNELS+(size_t)ch)*(size_t)links+(size_t)l;
        unsigned char handle[MESH_NET_HANDLE_BYTES]={0};memcpy(handle,targets+at,sizeof targets[at]);
        void *send=NULL;
        if((result=mesh_net_connect(probe,l,handle,&send,NULL)))status=net_failure(c,result,"mesh_net_connect");
        else if(send){e->send[ch]=send;if(e->dev<0)e->dev=l;}
      }
      for(int l=0;l<links && !e->recv[ch] && !status;l++){
        size_t at=((size_t)p*CHANNELS+(size_t)ch)*(size_t)links+(size_t)l;
        void *recv=NULL;
        if((result=mesh_net_accept(listens[at],&recv,NULL)))status=net_failure(c,result,"mesh_net_accept");
        else if(recv)e->recv[ch]=recv;
      }
      done&=e->send[ch] && e->recv[ch];
    }
    if(!done && !status)usleep(200);
  }
  for(size_t i=0;listens && i<count;i++)if(listens[i])mesh_net_close_listen(listens[i]);
  free(listens);free(targets);
  mesh_net_finalize(probe);
  return status;
}
/* The communicator's connections: with every rank the stated map links this one to (a link not yet up waits for
   its session), each rank's process as they name it. */
static ncclResult_t connect_peers(struct ncclComm *c){
  unsigned char *want=calloc((size_t)c->nranks,1);
  if(!want)return FAIL(c,ncclSystemError,"allocation");
  for(int p=0;p<c->nranks;p++)want[p]=(unsigned char)linked(&c->stated,c->rank,p);
  ncclResult_t status=connect_ranks(c,c->key,want,c->peers,NULL);
  free(want);
  if(!status)owners(c,c->peers);
  c->open=!status;
  return status;
}

/* ---- window memory ---- */
static pthread_mutex_t global_lock=PTHREAD_MUTEX_INITIALIZER;
static void *global_net;
/* The bridge's registered window, named once for every request. */
static struct { unsigned char *base; size_t bytes; void *mh; } window;
static ncclResult_t global_attach(void){
  ncclResult_t status=ncclSuccess;
  pthread_mutex_lock(&global_lock);
  if(!window.base){
    void *base=NULL,*mh=NULL;size_t bytes=0;char error[256];
    int result=global_net?0:mesh_net_init(&global_net,0,NULL,NULL,NULL);
    if(!result)result=mesh_net_window(&base,&bytes);
    if(!result)result=mesh_net_reg_mr(NULL,base,bytes,MESH_NET_PTR_HOST,&mh);
    if(result)status=net_failure(NULL,result,"mesh_net_init (the bridge region MESH_REGION)");
    else if(nccl_mesh_gpu_attach(error,sizeof error)){mesh_net_dereg_mr(NULL,mh);status=FAIL(NULL,ncclUnhandledCudaError,"the GPU: %s",error);}
    else{window.base=base;window.bytes=bytes;window.mh=mh;}
  }
  pthread_mutex_unlock(&global_lock);
  return status;
}
static size_t slice(size_t bytes){return (bytes+255)&~(size_t)255;}

/* The allocations (sorted by address) in slabs of the window this process claimed.  An allocation's
   record: its pages and Metal buffer (`owned`: the caller's, ncclMeshMemAllocBuffer, which the library
   holds no reference to); freed or not; its last writer's point and its readers' since; `kept`: retired,
   its pages and buffer kept for the next allocation of its size (at most KEPT of them, given back where
   the window has no other room), so an allocation made again and again (a group's own) makes no Metal
   buffer. */
#define READS 8
#define SLAB ((size_t)64<<20)
#define KEPT 16
struct span { unsigned char *at; size_t bytes; void *buffer; int freed,nread,owned,kept; struct point wrote,read[READS]; };
struct slab { unsigned char *at; size_t bytes; };
/* Memory outside the window the library orders by the same records (`outside`: a range, no buffer,
   no lifetime), made where the caller declares a use of it or a call on a stream touches it, and
   dropped once every point on it is reached. */
static struct { pthread_mutex_t lock; struct span *spans; int n,capacity; struct slab *slabs; int nslabs; struct span *outside; int nout,outcap,kept;
  size_t owned; } heap={.lock=PTHREAD_MUTEX_INITIALIZER};

static int reached(struct point p){return !p.event || nccl_mesh_event_value(p.event)>=p.value;}
static void point_drop(struct point *p){if(p->event)nccl_mesh_release(p->event);p->event=NULL;p->value=0;}
static void point_set(struct point *p,void *event,uint64_t value){
  if(event)nccl_mesh_retain(event);
  point_drop(p);
  p->event=event;p->value=value;
}
/* A reader's point joins the record's: the same event's later value replaces its earlier one, the
   points already reached make room; with none, the host waits for the oldest (counted), for as long as it
   takes (the GPU's work is late, not wrong; a program that failed for good has its points given by the host). */
static void read_add(struct span *s,void *event,uint64_t value,uint64_t *tally){
  for(int i=0;i<s->nread;i++)if(s->read[i].event==event){if(value>s->read[i].value)s->read[i].value=value;return;}
  for(int i=0;i<s->nread;)if(reached(s->read[i])){point_drop(s->read+i);s->read[i]=s->read[--s->nread];}else i++;
  if(s->nread==READS){
    if(tally)tally[HOST_WAITS]++;
    atomic_fetch_add_explicit(&totals[HOST_WAITS],1,memory_order_relaxed);
    while(!reached(s->read[0]))sched_yield();
    point_drop(s->read);s->read[0]=s->read[--s->nread];
  }
  s->read[s->nread].event=NULL;
  point_set(s->read+s->nread++,event,value);
}
static int done(const struct span *s){
  if(!reached(s->wrote))return 0;
  for(int i=0;i<s->nread;i++)if(!reached(s->read[i]))return 0;
  return 1;
}
static int retired(const struct span *s){return s->freed && done(s);}
static void points_drop(struct span *s){
  point_drop(&s->wrote);
  for(int r=0;r<s->nread;r++)point_drop(s->read+r);
  s->nread=0;
}
/* The record of [p, p+bytes) outside the window, made if absent; heap.lock held. */
static struct span *outside_of(const void *p,size_t bytes){
  for(int i=0;i<heap.nout;i++)if(heap.outside[i].at==(const unsigned char *)p && heap.outside[i].bytes==bytes)return heap.outside+i;
  if(heap.nout==heap.outcap){
    int capacity=heap.outcap?2*heap.outcap:64;
    struct span *grown=realloc(heap.outside,(size_t)capacity*sizeof *grown);
    if(!grown)return NULL;
    heap.outside=grown;heap.outcap=capacity;
  }
  struct span *s=heap.outside+heap.nout++;
  *s=(struct span){.at=(unsigned char *)p,.bytes=bytes};
  return s;
}
static int overlap(const struct span *s,const void *p,size_t bytes){
  const unsigned char *q=p;
  return q<s->at+s->bytes && s->at<q+bytes;
}
/* The first span at or after address p (binary search); heap.lock held. */
static int span_index(const void *p){
  int lo=0,hi=heap.n;
  while(lo<hi){int mid=(lo+hi)/2;if(heap.spans[mid].at<(const unsigned char *)p)lo=mid+1;else hi=mid;}
  return lo;
}
/* The live allocation holding [p, p+bytes), or NULL; heap.lock held. */
static struct span *span_of(const void *p,size_t bytes){
  int i=span_index((const unsigned char *)p+1)-1;
  if(i<0 || !bytes)return NULL;
  struct span *s=heap.spans+i;
  const unsigned char *q=p;
  return !s->freed && q>=s->at && (size_t)(q-s->at)<=s->bytes && bytes<=s->bytes-(size_t)(q-s->at)?s:NULL;
}
/* Every freed allocation whose points are all reached is kept (KEPT, above) or leaves the table (its
   buffer released, unless the caller owns it; its pages free for the next), with `drop` every kept one
   too; then every empty slab but one goes back to the region; heap.lock held. */
static void sweep_(int drop){
  int kept=0;
  for(int i=0;i<heap.n;i++){
    struct span *s=heap.spans+i;
    if(s->kept && !drop){heap.spans[kept++]=*s;continue;}
    if(s->kept){nccl_mesh_release(s->buffer);heap.kept--;continue;}
    if(retired(s)){
      points_drop(s);
      if(s->owned){heap.owned-=s->bytes;continue;}
      if(!drop && heap.kept<KEPT){s->kept=1;heap.kept++;heap.spans[kept++]=*s;continue;}
      nccl_mesh_release(s->buffer);
      continue;
    }
    heap.spans[kept++]=*s;
  }
  heap.n=kept;
  kept=0;
  for(int i=0;i<heap.nout;i++){
    struct span *s=heap.outside+i;
    if(done(s)){points_drop(s);continue;}
    heap.outside[kept++]=*s;
  }
  heap.nout=kept;
  int empty=0;
  for(int j=0;j<heap.nslabs;){
    struct slab *b=heap.slabs+j;int i=span_index(b->at);
    if((i<heap.n && heap.spans[i].at<b->at+b->bytes) || !empty++){j++;continue;}
    mesh_net_mem_free(b->at);
    heap.slabs[j]=heap.slabs[--heap.nslabs];
  }
}
static void sweep(void){sweep_(0);}
/* `bytes` at page p of a slab where no allocation lies, first fit; heap.lock held. */
static unsigned char *gap(size_t bytes){
  for(int j=0;j<heap.nslabs;j++){
    struct slab *b=heap.slabs+j;unsigned char *free=b->at;
    for(int i=span_index(b->at);i<heap.n && heap.spans[i].at<b->at+b->bytes;i++){
      if((size_t)(heap.spans[i].at-free)>=bytes)return free;
      free=heap.spans[i].at+heap.spans[i].bytes;
    }
    if((size_t)(b->at+b->bytes-free)>=bytes)return free;
  }
  return NULL;
}
/* A window allocation of `size` bytes (whole pages) and its record and Metal buffer: a kept one of its
   size (its buffer again), else in a free gap of a slab (after the retired allocations have left, and
   then the kept ones), else a new slab from the bridge's window; while none, the host waits for a freed
   allocation's points (counted) for as long as they take, or, `owned` (ncclMeshMemAllocBuffer: its buffer
   made with `options`, the caller's), fails at once. */
static void owned_gone(void *at);
static ncclResult_t span_alloc_(size_t size,unsigned char **out,uint64_t *tally,int owned,uint64_t options,void **made){
  const size_t page=(size_t)getpagesize(),bytes=(size?size+page-1:page)&~(page-1);
  ncclResult_t status=ncclSuccess;
  int waited=0;
  pthread_mutex_lock(&heap.lock);
  if(owned && (heap.owned+bytes)/3>(window.bytes-heap.owned-bytes)){
    pthread_mutex_unlock(&heap.lock);
    return FAIL(NULL,ncclSystemError,"%zu bytes of the bridge's registered window for a caller's buffer: its buffers would hold more than three quarters",bytes);
  }
  for(;;){
    sweep();
    for(int i=0;!owned && i<heap.n;i++)if(heap.spans[i].kept && heap.spans[i].bytes==bytes && done(heap.spans+i)){
      struct span *s=heap.spans+i;
      s->kept=0;s->freed=0;heap.kept--;
      *out=s->at;
      pthread_mutex_unlock(&heap.lock);
      return ncclSuccess;
    }
    unsigned char *at=gap(bytes);
    if(!at && heap.kept){sweep_(1);at=gap(bytes);}
    if(!at){
      /* a new slab, the allocation its first */
      void *claimed=NULL;
      struct slab *grown=realloc(heap.slabs,(size_t)(heap.nslabs+1)*sizeof *grown);
      if(!grown){status=FAIL(NULL,ncclSystemError,"allocation");break;}
      heap.slabs=grown;
      size_t want=bytes>SLAB?bytes:SLAB;
      if(!mesh_net_mem_alloc(&claimed,want) || (want>bytes && !mesh_net_mem_alloc(&claimed,want=bytes))){
        heap.slabs[heap.nslabs++]=(struct slab){claimed,want};
        at=claimed;
      }
    }
    if(at){
      uint64_t offset;
      void *buffer=owned?nccl_mesh_buffer_owned(at,bytes,options,owned_gone,at):nccl_mesh_buffer(at,bytes,&offset);
      if(!buffer){status=FAIL(NULL,ncclUnhandledCudaError,"window memory as a Metal buffer (newBufferWithBytesNoCopy)");break;}
      if(tally)tally[BUFFERS]++;
      atomic_fetch_add_explicit(&totals[BUFFERS],1,memory_order_relaxed);
      if(heap.n==heap.capacity){
        int capacity=heap.capacity?2*heap.capacity:256;
        struct span *grown=realloc(heap.spans,(size_t)capacity*sizeof *grown);
        if(!grown){nccl_mesh_release(buffer);status=FAIL(NULL,ncclSystemError,"allocation");break;}
        heap.spans=grown;heap.capacity=capacity;
      }
      int i=span_index(at);
      memmove(heap.spans+i+1,heap.spans+i,(size_t)(heap.n-i)*sizeof *heap.spans);
      heap.spans[i]=(struct span){.at=at,.bytes=bytes,.buffer=buffer,.owned=owned};heap.n++;
      if(owned){heap.owned+=bytes;*made=buffer;}
      *out=at;
      break;
    }
    int freed=0;
    for(int i=0;i<heap.n && !freed;i++)freed=heap.spans[i].freed;
    if(owned || !freed){
      status=FAIL(NULL,ncclSystemError,"%zu bytes of the bridge's registered window: no room (the bridge's -A/-W window)",bytes);
      break;
    }
    if(!waited++){if(tally)tally[HOST_WAITS]++;atomic_fetch_add_explicit(&totals[HOST_WAITS],1,memory_order_relaxed);}
    pthread_mutex_unlock(&heap.lock);
    sched_yield();
    pthread_mutex_lock(&heap.lock);
  }
  pthread_mutex_unlock(&heap.lock);
  return status;
}
static ncclResult_t span_alloc(size_t size,unsigned char **out,uint64_t *tally){return span_alloc_(size,out,tally,0,0,NULL);}
/* An allocation freed once `event` (if any) reaches `value` and every use recorded on it is done. */
static ncclResult_t span_release(void *p,void *event,uint64_t value){
  pthread_mutex_lock(&heap.lock);
  int i=span_index(p);
  struct span *s=i<heap.n && heap.spans[i].at==(unsigned char *)p && !heap.spans[i].freed?heap.spans+i:NULL;
  if(s){
    if(event)read_add(s,event,value,NULL);
    s->freed=1;
    sweep();
  }
  pthread_mutex_unlock(&heap.lock);
  return s?ncclSuccess:FAIL(NULL,ncclInvalidArgument,"%p is not a live allocation of the window (ncclMemAlloc)",p);
}

/* ---- requests ---- */
/* Nothing here fails in time: a request is done once its bridge says so, however late (a peer slowed by other
   work, a bridge stopped for seconds, a link whose session resumes), and fails only where the bridge observed a
   failure (a peer that exited, a comm its peer closed) or the communicator was aborted or revoked. */
struct pending { size_t lo,hi; void *request; };
static void progress(struct ncclComm *c);
static int stopped(struct ncclComm *c){return atomic_load(&c->aborting) || atomic_load(&c->broken);}
/* Why a part stopped; a revoked communicator's parts fail without a message of their own, so the
   communicator's error stays its first failure's. */
static ncclResult_t stop_reason(struct ncclComm *c,const char *what){
  if(atomic_load(&c->aborting))return FAIL(c,ncclInvalidUsage,"%s: the communicator was aborted",what);
  return ncclRemoteError;
}
/* Reaps every pending isend that is done (its ring slot free again). */
static ncclResult_t reap(struct ncclComm *c,struct pending *sends,int *count){
  for(int i=0;i<*count;){
    int done=0,size=0,result=mesh_net_test(sends[i].request,&done,&size);
    if(result)return net_failure(c,result,"an isend");
    if(done){sends[i]=sends[--*count];continue;}
    i++;
  }
  return ncclSuccess;
}
/* A request waited for: `bytes` expected (SIZE_MAX: any, those that arrived in *arrived where it is not
   NULL). */
static ncclResult_t await_(struct ncclComm *c,void *request,size_t bytes,size_t *arrived,struct pending *sends,int *count,const char *what){
  for(;;){
    int done=0,size=0,result=mesh_net_test(request,&done,&size);
    if(result)return net_failure(c,result,what);
    if(done){
      if(arrived)*arrived=(size_t)(unsigned)size;
      if(bytes<=INT32_MAX && (size_t)size!=bytes)return FAIL(c,ncclInvalidUsage,"%s: %d bytes arrived, %zu expected (the peer's count or datatype differs)",what,size,bytes);
      return ncclSuccess;
    }
    if(sends){ncclResult_t r=reap(c,sends,count);if(r)return r;}
    if(stopped(c))return stop_reason(c,what);
    progress(c);
    sched_yield();
  }
}
static ncclResult_t await(struct ncclComm *c,void *request,size_t bytes,struct pending *sends,int *count,const char *what){
  return await_(c,request,bytes,NULL,sends,count,what);
}
/* An isend or irecv posted, its end stored into `word` by the bridge, waiting while its connection's
   request ring is full. */
static ncclResult_t post(struct ncclComm *c,int send,void *comm,void *data,size_t bytes,_Atomic uint64_t *word,
  struct pending *sends,int *count,void **request){
  for(*request=NULL;;){
    int result=send?mesh_net_isend_word(comm,data,bytes,window.mh,(uint64_t *)word,request):
      mesh_net_irecv_word(comm,data,bytes,window.mh,(uint64_t *)word,request);
    if(result)return net_failure(c,result,send?"mesh_net_isend":"mesh_net_irecv");
    if(*request)return ncclSuccess;
    if(sends){ncclResult_t r=reap(c,sends,count);if(r)return r;}
    if(stopped(c))return stop_reason(c,send?"an isend":"an irecv");
    progress(c);
    sched_yield();
  }
}

/* The worker waits for a word the program publishes (a combine a later send reads), the point-to-point
   transfers moving meanwhile. */
static ncclResult_t word_wait(struct ncclComm *c,struct call *k,_Atomic uint64_t *word,const char *what){
  count(k,HOST_WORD_WAITS,1);
  while(!atomic_load_explicit(word,memory_order_acquire)){
    if(atomic_load(&c->gpu_failed))return FAIL(c,ncclUnhandledCudaError,"%s: a GPU program of the communicator failed",what);
    if(stopped(c))return stop_reason(c,what);
    progress(c);
    sched_yield();
  }
  return ncclSuccess;
}
/* A failed part's completion words the bridge has not set, set (2: failed), so the program waiting on
   them goes on; collectives, or point-to-point calls. */
static void words_fail(struct item *it,int p2p){
  for(int i=0;i<it->n;i++){
    struct call *k=it->calls+i;
    if(P2P(k)!=p2p)continue;
    for(uint32_t w=0;w<k->nwords;w++){uint64_t none=0;atomic_compare_exchange_strong(k->words+w,&none,2);}
    if(k->arrived){uint64_t none=0;atomic_compare_exchange_strong((_Atomic uint64_t *)k->arrived,&none,2);}
  }
}

/* ---- one collective ---- */
/* A step's piece moves in chunks, each its own isend or irecv and, for a REDUCE, its own arrival and
   combine, so the GPU combines a chunk while the chunks after it are on the wire and after the last byte
   lands only the last chunk's combine remains [the pipelined chunks of Patarasuk & Yuan 2009, §4; NCCL's
   slices].  That overlap is all they are for (the bridge cuts every request into its own RECV chunks,
   MESH_DISCARD, and the GPU waits on each request's completion word), so a piece is cut as its combines
   need: from its end, TAIL, TAIL, 2 TAIL, 4 TAIL, ... bytes, as many as the piece holds whole, the first
   chunk taking the rest too.  The last chunk's combine is TAIL's, and every other chunk's is done before
   the chunks after it have landed: the GPU combines several times faster than the link delivers (16 MB in
   340 us on the M4 Pro, 1.8 ms on the wire), and the chunks after one are at least a third of it.  A piece
   under 2 TAIL is one chunk.  Both ranks cut a piece alike, from its elements alone. */
#define TAIL ((size_t)1<<20)
/* The doublings of the piece's cut: the largest m with TAIL << m bytes within it (0: one chunk). */
static uint32_t chunk_doublings(uint64_t elements,size_t e){
  const uint64_t t=TAIL/e;
  uint32_t m=0;
  while(m<40 && (t<<(m+1))<=elements)m++;
  return m;
}
static uint32_t chunks_of(uint64_t elements,size_t e){return chunk_doublings(elements,e)+1;}
static uint64_t chunk_first(uint64_t elements,size_t e,uint32_t j){
  const uint32_t m=chunk_doublings(elements,e);
  return j?elements-((TAIL/e)<<(m-j)):0;
}
static uint64_t chunk_count(uint64_t elements,size_t e,uint32_t j){
  const uint32_t m=chunk_doublings(elements,e);
  const uint64_t t=TAIL/e;
  if(!j)return m?elements-(t<<(m-1)):elements;
  return j==m?t:t<<(m-j-1);
}
/* The receives a REDUCE or COPY step keeps posted ahead of the one it waits for (a connection's request
   ring holds MESH_NET_REQUESTS). */
#define AHEAD 8
/* The worker's side of a collective the group's end placed: this rank's plan, each isend and irecv with
   its completion word (the next of the call's, in plan order).  A piece the GPU combines is sent only once
   the program has published its combine (the call's combined words); the GPU waits for a REDUCE chunk's
   receive and the earlier sends of its range on their words itself.  A persistent call's requests may have
   been posted ahead (struct call `ahead`, persistent_post): its isends, held, are released here once their
   bytes are ready (its cut passed, or the ranges they read published), and whether the receiver had granted
   one's first chunk by then is counted (a send not granted: its first byte waits for a request and credit
   exchanged over the bridges' control sockets, as every send posted here does).  A persistent call flushes
   no connection: a later call's receives may be posted on it already, and an iflush waits for every earlier
   request of its connection (its GPU work waits on the completion words, and loads what the NIC wrote
   system-coherent). */
struct combined { size_t lo,hi; _Atomic uint64_t *word; };
static int written(const struct call *k,uint32_t s,size_t lo,size_t hi);
static const unsigned char *given(const struct call *k,size_t lo);
static size_t in_bytes(const struct call *k);
static void persistent_post(struct ncclComm *c);
static uint64_t persistent_iteration(const struct item *it);
static ncclResult_t persistent_failure(struct ncclComm *c);
/* A persistent call's send buffer's bytes [lo, hi) published (ncclMeshPersistentNext): each range they
   touch counted up this iteration (`iteration` of the run); the first seen, READIED. */
static ncclResult_t ranges_wait(struct ncclComm *c,struct call *k,size_t lo,size_t hi,uint64_t iteration){
  for(uint64_t r=lo/k->range;hi>lo && r<=(hi-1)/k->range && r<k->ranges;r++){
    const uint64_t want=k->ready_base[r]+iteration+1;
    if(atomic_load_explicit(k->ready+r,memory_order_acquire)>=want)continue;
    count(k,HOST_WORD_WAITS,1);
    while(atomic_load_explicit(k->ready+r,memory_order_acquire)<want){
      if(atomic_load(&c->gpu_failed))return FAIL(c,ncclUnhandledCudaError,"a published range of a persistent call's send buffer: a GPU program of the communicator failed");
      if(stopped(c))return stop_reason(c,"a published range of a persistent call's send buffer");
      progress(c);
      sched_yield();
    }
  }
  if(k->tally && !k->tally[READIED])k->tally[READIED]=now_ns();
  return ncclSuccess;
}
/* A persistent call's request q, posted ahead by persistent_post (run from progress as the rings take it). */
static ncclResult_t posted_wait(struct ncclComm *c,struct call *k,uint32_t q){
  while(!k->handles[q]){
    ncclResult_t status=persistent_failure(c);
    if(status)return status;
    if(stopped(c))return stop_reason(c,"a persistent call's request posted ahead");
    progress(c);
    if(!k->handles[q])sched_yield();
  }
  return ncclSuccess;
}
/* A persistent part of point-to-point calls (ncclMeshPersistentBegin): each call's one request was posted as its
   iteration opened (persistent_post: an isend held, an irecv into the caller's buffer whose end the bridge stores
   into the caller's landed word, which the caller's GPU work waits on); each isend is released once its ready word
   has counted up this iteration (READIED, and whether its receiver had granted its first byte by then), in whatever
   order the words count up: a send and a receive of one exchange fly together, and no request waits for another's.
   A pass reads each unreleased send's ready word and each receive's landed word (ARRIVED, as first seen), one load
   each, so a release follows its word within one short pass; the requests' ends are tested in issue order, which
   frees their ring slots in the order they were posted, and the part ends once every one has. */
/* A failed persistent part of point-to-point calls: every landed word not yet set set (2), its receive posted or not
   (the rings take an iteration's requests as earlier ones end, so some may never have reached the bridge, which
   stores the posted ones' ends), so the caller's GPU work waiting on one ends and the caller reads the failure. */
static void held_failed(struct item *it){
  for(int i=0;i<it->n;i++){
    struct call *k=it->calls+i;
    uint64_t unset=0;
    if(k->kind==K_RECV && k->words)atomic_compare_exchange_strong_explicit(k->words,&unset,2,memory_order_acq_rel,memory_order_relaxed);
  }
}
static ncclResult_t run_held_p2p(struct ncclComm *c,struct item *it){
  const uint64_t iteration=persistent_iteration(it);
  unsigned char *state=calloc((size_t)it->n,1);  /* a send: 1 released; a receive: 1 its landed word seen */
  ncclResult_t status=state?ncclSuccess:FAIL(c,ncclSystemError,"allocation");
  int tested=0;
  while(tested<it->n && !status){
    int moved=0;
    progress(c);
    if((status=persistent_failure(c)))break;
    for(int i=tested;i<it->n && !status;i++){
      struct call *k=it->calls+i;
      if(state[i] || !k->handles[0])continue;
      if(k->kind==K_RECV){
        if(!atomic_load_explicit(k->words,memory_order_acquire))continue;
        if(k->tally)k->tally[ARRIVED]=now_ns();
        state[i]=1;moved=1;
        continue;
      }
      if(atomic_load_explicit(k->ready,memory_order_acquire)<(k->slot_depth?1:k->ready_base[0]+iteration+1))continue;
      uint64_t granted=0;
      int result=mesh_net_release(k->handles[0],&granted);
      if(result){status=net_failure(c,result,"mesh_net_release");break;}
      if(k->tally)k->tally[READIED]=now_ns();
      count(k,SENDS,1);
      if(!granted)count(k,GRANT_WAITS,1);
      state[i]=1;moved=1;
    }
    for(;tested<it->n && !status;tested++){
      struct call *k=it->calls+tested;
      if(!k->handles[0] || (k->kind==K_SEND && !state[tested]))break;
      int done=0,size=0,result=mesh_net_test(k->handles[0],&done,&size);
      if(result){status=net_failure(c,result,k->kind==K_SEND?"a persistent send":"a persistent receive");break;}
      if(!done)break;
      moved=1;
      const size_t bytes=k->count*type_bytes[k->type];
      count(k,k->kind==K_SEND?SENT:RECEIVED,bytes);
      if(k->kind==K_RECV && k->tally && !state[tested])k->tally[ARRIVED]=now_ns();
      if(k->kind==K_RECV && bytes<=INT32_MAX && (size_t)size!=bytes)
        status=FAIL(c,ncclInvalidUsage,"a persistent receive: %d bytes arrived, %zu expected (the peer's count or datatype differs)",size,bytes);
    }
    if(!status && tested<it->n && stopped(c))status=stop_reason(c,"a persistent point-to-point call");
    if(!moved && !status)sched_yield();
  }
  if(status)held_failed(it);
  free(state);
  return status;
}
static ncclResult_t run_collective(struct ncclComm *c,struct call *k,struct item *it){
  const size_t e=type_bytes[k->type];
  const int held=it->run!=NULL,ahead=held && k->ahead;
  const uint64_t iteration=it->run?persistent_iteration(it):0;
  ncclResult_t status=ncclSuccess;
  size_t total=1;
  for(uint32_t i=0;i<k->nsteps;i++)total+=chunks_of(k->steps[i].piece.elements,e);
  struct pending *sends=calloc(total,sizeof *sends);int count_=0;
  struct combined *done=calloc(total,sizeof *done);int ndone=0;
  void **requests=calloc(total,sizeof *requests);
  void **used=c->used;int nused=0;
  uint32_t w=0,r=0;
  if(!sends || !done || !requests)status=FAIL(c,ncclSystemError,"allocation");
  for(uint32_t i=0;i<k->nsteps && !status;i++){
    const struct mesh_step *s=k->steps+i;
    const uint32_t n=chunks_of(s->piece.elements,e);
    struct peer *p=c->peers+s->peer;
    if(!s->piece.elements)continue;
    if(s->op==MESH_STEP_SEND){
      for(uint32_t j=0;j<n && !status;j++){
        const size_t lo=(s->first+chunk_first(s->piece.elements,e,j))*e,length=chunk_count(s->piece.elements,e,j)*e;
        for(int d=0;d<ndone && !status;d++)
          if(done[d].lo<lo+length && lo<done[d].hi && !atomic_load_explicit(done[d].word,memory_order_acquire))
            status=word_wait(c,k,done[d].word,"a combine");
        void *request=NULL;uint64_t granted=0;
        unsigned char *from=k->fresh && !written(k,i,lo,lo+length)?(unsigned char *)given(k,lo):k->operand+lo;
        const size_t in=(size_t)(from-(const unsigned char *)k->send);
        if(!status && k->ready && from>=(const unsigned char *)k->send && in+length<=in_bytes(k))status=ranges_wait(c,k,in,in+length,iteration);
        if(status)break;
        if(held && !(status=posted_wait(c,k,w+j))){
          request=k->handles[w+j];
          int result=mesh_net_release(request,&granted);
          if(result)status=net_failure(c,result,"mesh_net_release");
        }
        else if(!held)status=post(c,1,p->send[CH_COLL],from,length,k->words+w+j,sends,&count_,&request);
        if(!status){
          sends[count_++]=(struct pending){lo,lo+length,request};
          count(k,SENT,length);count(k,SENDS,1);
          if(!granted)count(k,GRANT_WAITS,1);
        }
      }
      w+=n;
      continue;
    }
    int known=it->run!=NULL;
    for(int u=0;u<nused;u++)known|=used[u]==p->recv[CH_COLL];
    if(!known)used[nused++]=p->recv[CH_COLL];
    const size_t lo=s->first*e,length=s->piece.elements*e;
    /* a piece of the operand is written only once no isend still reads it */
    for(int j=0;j<count_ && !status && s->op==MESH_STEP_COPY;)
      if(sends[j].lo<lo+length && lo<sends[j].hi){status=await(c,sends[j].request,SIZE_MAX,NULL,NULL,"an isend");sends[j]=sends[--count_];}
      else j++;
    if(status)break;
    uint32_t posted=0;
    for(uint32_t j=0;j<n && !status;j++){
      for(;posted<n && posted<j+AHEAD && !status;posted++){
        const size_t at=chunk_first(s->piece.elements,e,posted)*e,bytes=chunk_count(s->piece.elements,e,posted)*e;
        unsigned char *into=s->op==MESH_STEP_COPY?k->operand+lo+at:k->pieces[i].at+at;
        if(!ahead)status=post(c,0,p->recv[CH_COLL],into,bytes,k->words+w+posted,sends,&count_,requests+posted);
        else if(!(status=posted_wait(c,k,w+posted)))requests[posted]=k->handles[w+posted];
      }
      if(status)break;
      const size_t at=lo+chunk_first(s->piece.elements,e,j)*e,bytes=chunk_count(s->piece.elements,e,j)*e;
      if((status=await(c,requests[j],bytes,sends,&count_,s->op==MESH_STEP_COPY?"a COPY step's irecv":"a REDUCE step's irecv")))break;
      count(k,RECEIVED,bytes);
      if(s->op!=MESH_STEP_REDUCE)continue;
      if(k->tally)k->tally[ARRIVED]=now_ns();
      if(k->combined)done[ndone++]=(struct combined){at,at+bytes,k->combined+r+j};
    }
    w+=n;
    if(s->op==MESH_STEP_REDUCE && k->combined)r+=n;
  }
  /* a persistent call whose irecvs are not posted ahead has posted them all: the posting passes it */
  if(held && !ahead && !status)k->posted=1;
  while(count_ && !status){status=await(c,sends[count_-1].request,SIZE_MAX,NULL,NULL,"an isend");count_--;}
  for(int u=0;u<nused && !status;u++){
    void *flush=NULL;int result=mesh_net_iflush(used[u],1,NULL,NULL,NULL,&flush);
    if(result)status=net_failure(c,result,"mesh_net_iflush");
    else if(flush)status=await(c,flush,SIZE_MAX,NULL,NULL,"an iflush");
  }
  free(sends);free(done);free(requests);
  return status;
}

/* ---- a counted all-to-all (nccl.h ncclMeshAlltoAllCounted) ----
   The worker's side, one exchange: to each rank its segment of the counts and then its rows (as many as
   that segment sums to; none, a message of no bytes), both posted at once, and from each rank the same,
   so no rows wait for the counts to cross first.  Each rank's rows are received packed in rank order,
   where the rows of the ranks before it end: their receive is posted once the counts of those ranks have
   landed (at once for the first other rank, before which lie at most this rank's own rows, counted
   here), sized to the capacity left and checked against its counts once they land.  Once every count has
   landed each rank's segment for this rank is written to the caller's host array in rank order (this
   rank's own from its counts) and the caller's word set, the rows maybe still on the wire; the GPU copies
   this rank's own rows (the program).  Completion words: the counts' sends, their receives, the rows'
   sends, their receives, n - 1 each; a segment of no counts moves nothing (its word set here). */
static ncclResult_t run_counted(struct ncclComm *c,struct call *k){
  const int n=c->nranks,me=c->rank,peers=n-1;
  const size_t row=k->row*type_bytes[k->type];
  const uint64_t *sseg=k->segments,*rseg=k->segments+n;
  int64_t *got=(int64_t *)k->wire;
  void **requests=calloc((size_t)(4*n),sizeof *requests);
  /* per rank: the rows sent to it and received from it, where its counts lie among this rank's and in
     `got`; its place among the peers (-1: this rank) and the rank at each place; whether its counts have
     landed */
  uint64_t *rows=calloc((size_t)(4*n),sizeof *rows),*so=rows+2*n,*ro=rows+3*n;
  int *index=calloc((size_t)(3*n),sizeof *index),*rank_at=index+n,*landed=index+2*n;
  ncclResult_t status=requests && rows && index?ncclSuccess:FAIL(c,ncclSystemError,"allocation");
  uint64_t sent=0;
  for(int q=0,i=0;q<n && !status;q++){
    if(q){so[q]=so[q-1]+sseg[q-1];ro[q]=ro[q-1]+rseg[q-1];}
    for(uint64_t x=0;x<sseg[q];x++)rows[q]+=(uint64_t)k->counted[so[q]+x];
    sent+=rows[q];
    index[q]=q==me?-1:i;
    if(q!=me)rank_at[i++]=q;
    landed[q]=q==me || !rseg[q];
  }
  if(!status)rows[n+me]=rows[me];
  if(!status && sent>k->rows)
    status=FAIL(c,ncclInvalidUsage,"a counted all-to-all: its counts send %llu rows of %llu",(unsigned long long)sent,(unsigned long long)k->rows);
  /* to each rank: its counts, then its rows; from each: its counts */
  uint64_t before=0;
  for(int q=0;q<n && !status;before+=rows[q],q++){
    if(q==me)continue;
    struct peer *p=c->peers+q;
    const int i=index[q];
    if(rseg[q])status=post(c,0,p->recv[CH_COLL],got+ro[q],rseg[q]*8,k->words+peers+i,NULL,NULL,requests+peers+i);
    else atomic_store_explicit(k->words+peers+i,1,memory_order_release);
    if(!status && sseg[q])status=post(c,1,p->send[CH_COLL],(void *)(k->counted+so[q]),sseg[q]*8,k->words+i,NULL,NULL,requests+i);
    else if(!status)atomic_store_explicit(k->words+i,1,memory_order_release);
    if(!status)status=post(c,1,p->send[CH_COLL],(unsigned char *)k->send+before*row,rows[q]*row,k->words+2*peers+i,NULL,NULL,requests+2*peers+i);
  }
  /* the rows from each rank received where the rows of the ranks before it end: posted once that is known,
     then the next rank's counts waited for */
  int next=0;
  uint64_t at=0;
  while(!status){
    for(;next<n && !status;next++){
      const int q=next,i=index[q];
      if(q!=me && !requests[3*peers+i])
        status=post(c,0,c->peers[q].recv[CH_COLL],(unsigned char *)k->recv+at*row,(k->capacity-at)*row,k->words+3*peers+i,NULL,NULL,
                    requests+3*peers+i);
      if(status || !landed[q])break;
      if(rows[n+q]>k->capacity-at)
        status=FAIL(c,ncclInvalidUsage,"a counted all-to-all: its counts receive more than its capacity of %llu rows",(unsigned long long)k->capacity);
      else at+=rows[n+q];
    }
    if(status || next==n)break;
    status=await(c,requests[peers+index[next]],rseg[next]*8,NULL,NULL,"a counted all-to-all's counts");
    count(k,RECEIVED,rseg[next]*8);
    for(uint64_t x=0;!status && x<rseg[next];x++)rows[n+next]+=(uint64_t)got[ro[next]+x];
    landed[next]=1;
  }
  if(!status){
    /* the counts on the host */
    for(int q=0;q<n;q++)memcpy(k->received+ro[q],q==me?k->counted+so[q]:got+ro[q],rseg[q]*8);
    atomic_store_explicit((_Atomic uint64_t *)k->arrived,1,memory_order_release);
    count(k,GPU_COPY,rows[me]*row);  /* this rank's own rows: the GPU's counted copy */
  }
  /* the counts sent, the rows sent, the rows received (as many as their counts say) */
  for(int j=0;j<4*peers && !status;j++){
    if((j>=peers && j<2*peers) || !requests[j])continue;
    const int q=rank_at[j%peers];
    const size_t bytes=j<peers?sseg[q]*8:j<3*peers?rows[q]*row:rows[n+q]*row;
    status=await(c,requests[j],j<3*peers?SIZE_MAX:bytes,NULL,NULL,j<peers?"a counted all-to-all's counts":"a counted all-to-all's rows");
    count(k,j<3*peers?SENT:RECEIVED,bytes);
  }
  if(status && k->arrived){uint64_t none=0;atomic_compare_exchange_strong((_Atomic uint64_t *)k->arrived,&none,2);}
  free(requests);free(rows);free(index);
  return status;
}

/* ---- one communicator's part of a group ---- */
struct transfer { int send; struct peer *peer; unsigned char *window; size_t bytes; void *request; int state; struct call *call; };
/* A part's point-to-point transfers, in flight: posted in order on each connection (across every
   flight, oldest first) as far as its request ring takes them, tested, and retired together. */
struct flight { struct item *it; struct transfer *t; int n,remaining; ncclResult_t status; struct flight *next; };

static void finish(struct ncclComm *c,struct item *it,ncclResult_t result);
static void dispatch_pump(void);
/* ---- gates (struct gate) ---- */
/* `g` armed on its communicator: in the armed list by value (where it is armed already, its entry taken out
   first: a persistent call's gate is armed again each iteration with its value then). */
static void gate_arm(struct gate *g){
  struct ncclComm *c=g->comm;
  pthread_mutex_lock(&c->gates_lock);
  if(g->armed){
    int kept=0;
    for(int i=0;i<c->narmed;i++)if(c->armed[i]!=g)c->armed[kept++]=c->armed[i];
    __atomic_store_n(&c->narmed,kept,__ATOMIC_RELEASE);
  }
  g->ready=0;g->armed=1;
  if(c->narmed==c->carmed){
    /* a gate is never left unarmed (it would never open): the table's growth waited for */
    int capacity=c->carmed?2*c->carmed:64;
    struct gate **grown;
    while(!(grown=realloc(c->armed,(size_t)capacity*sizeof *grown)))sched_yield();
    c->armed=grown;c->carmed=capacity;
  }
  {
    int at=c->narmed;
    while(at && c->armed[at-1]->value>g->value){c->armed[at]=c->armed[at-1];at--;}
    c->armed[at]=g;__atomic_store_n(&c->narmed,c->narmed+1,__ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&c->gates_lock);
}
/* The communicator's armed gates whose words are all set marked ready, and each ready one opened where every
   armed gate of its event before it (by value: an event's value only grows) is: its event signalled at its value,
   then its landing word set; a gate that is not a persistent call's then freed. */
#define GATE_EVENTS 8
static void gates_check(struct ncclComm *c){
  if(!__atomic_load_n(&c->narmed,__ATOMIC_ACQUIRE))return;
  pthread_mutex_lock(&c->gates_lock);
  void *blocked[GATE_EVENTS];int nblocked=0,kept=0,opened_now=0;
  for(int i=0;i<c->narmed;i++){
    struct gate *g=c->armed[i];
    if(!g->ready){
      g->ready=1;
      for(uint32_t w=0;w<g->nwords && g->ready;w++)g->ready=atomic_load_explicit(g->words[w],memory_order_acquire)!=0;
    }
    int held=nblocked==GATE_EVENTS;
    for(int b=0;b<nblocked && !held;b++)held=blocked[b]==g->event;
    if(!g->ready || held){
      if(!held)blocked[nblocked++]=g->event;
      c->armed[kept++]=g;
      continue;
    }
    if(g->landing)atomic_store_explicit(g->landing,1,memory_order_release);
    if(g->event)nccl_mesh_event_signal(g->event,g->value);
    atomic_fetch_add_explicit(&opened,1,memory_order_relaxed);
    g->armed=0;
    if(!g->run){free(g->words);free(g);}
    opened_now++;
  }
  __atomic_store_n(&c->narmed,kept,__ATOMIC_RELEASE);
  pthread_mutex_unlock(&c->gates_lock);
  if(opened_now)dispatch_pump();
}
/* A flight done: its part finished if its collectives are. */
static void retire(struct ncclComm *c,struct flight *f){
  struct item *it=f->it;
  it->flight=NULL;
  if(f->status)words_fail(it,1);
  if(f->status && !it->result)it->result=f->status;
  it->networked=1;
  free(f->t);free(f);
  pthread_mutex_lock(&c->lock);c->nflights--;pthread_cond_broadcast(&c->cond);pthread_mutex_unlock(&c->lock);
  if(it->collectives_done)finish(c,it,it->result);
}
static void progress(struct ncclComm *c){
  gates_check(c);
  if(__atomic_load_n(&c->run,__ATOMIC_ACQUIRE))persistent_post(c);
  if(!c->flights)return;
  memset(c->blocked,0,sizeof *c->blocked*(size_t)c->nranks*2);
  for(struct flight **link=&c->flights;*link;){
    struct flight *f=*link;
    if(!f->status && atomic_load(&c->broken))f->status=ncclRemoteError;  /* revoked: its connections are closed */
    for(int j=0;j<f->n && !f->status;j++){
      struct transfer *x=f->t+j;int slot=(int)(x->peer-c->peers)*2+x->send;
      if(x->state==0 && !c->blocked[slot]){
        void *comm=x->send?x->peer->send[CH_P2P]:x->peer->recv[CH_P2P];
        uint64_t *word=x->call->nwords?(uint64_t *)x->call->words:NULL;
        int result=x->send?mesh_net_isend_word(comm,x->window,x->bytes,window.mh,word,&x->request):
          mesh_net_irecv_word(comm,x->window,x->bytes,window.mh,word,&x->request);
        if(result)f->status=net_failure(c,result,x->send?"a send's isend":"a receive's irecv");
        else if(x->request)x->state=1;
      }
      if(x->state==0)c->blocked[slot]=1;
      if(x->state==1){
        int done=0,size=0,result=mesh_net_test(x->request,&done,&size);
        if(result)f->status=net_failure(c,result,x->send?"a send":"a receive");
        else if(done){
          x->state=2;f->remaining--;
          count(x->call,x->send?SENT:RECEIVED,x->bytes);
          if(!x->send && x->bytes<=INT32_MAX && (size_t)size!=x->bytes)
            f->status=FAIL(c,ncclInvalidUsage,"a receive: %d bytes arrived, %zu expected (the peer's count or datatype differs)",size,x->bytes);
        }
      }
    }
    if(!f->status && f->remaining && stopped(c))f->status=stop_reason(c,"a point-to-point call");
    if(f->status || !f->remaining){*link=f->next;retire(c,f);}
    else link=&f->next;
  }
}

/* A part started: its point-to-point calls put in flight (each from and into its bytes in the window:
   a receive from this rank itself the program copied), then the collectives run in issue order; the
   part is finished here, or when its flight retires. */
static void start(struct ncclComm *c,struct item *it){
  ncclResult_t status=atomic_load(&c->broken)?ncclRemoteError:ncclSuccess;
  const uint64_t now=now_ns();
  if(c->woke && it->n){count(it->calls,WAKEUPS,1);c->woke=0;}
  for(int i=0;i<it->n;i++)if(it->calls[i].tally && !it->calls[i].tally[STARTED])it->calls[i].tally[STARTED]=now;
  /* what it started at: the gate word, or the recorded points (events, words) */
  const struct launch *l=it->launch;
  if(it->run){if(it->cut)count(it->calls,INPUT_WAITS,1);}  /* its cut's event (a published range: ranges_wait) */
  else if(it->n && l->gate)count(it->calls,HOST_WORD_WAITS,1);
  for(int w=0;it->n && !it->run && !l->gate && w<l->nwaits;w++)count(it->calls,INPUT_WAITS,1);
  /* a persistent part of point-to-point calls: its requests posted ahead, run here to their ends */
  const int held=it->run && it->n && P2P(it->calls);
  if(held && !status){status=run_held_p2p(c,it);it->networked=1;}
  else if(held)held_failed(it);
  int transfers=0;
  for(int i=0;i<it->n && !status && !held;i++){
    struct call *k=it->calls+i;
    if(!P2P(k))continue;
    if(k->peer!=c->rank)transfers++;
    else if(k->kind==K_RECV && !k->copied)status=FAIL(c,ncclInvalidUsage,"a receive from this rank itself has no matching send to itself of its size in the group");
  }
  if(!status && transfers){
    struct flight *f=calloc(1,sizeof *f);
    if(f)f->t=calloc((size_t)transfers,sizeof *f->t);
    if(!f || !f->t){if(f)free(f->t);free(f);status=FAIL(c,ncclSystemError,"allocation");}
    else{
      for(int i=0;i<it->n;i++){
        struct call *k=it->calls+i;
        if(!P2P(k) || k->peer==c->rank)continue;
        f->t[f->n++]=(struct transfer){.send=k->kind==K_SEND,.peer=c->peers+k->peer,.window=k->wire,.bytes=k->count*type_bytes[k->type],.call=k};
      }
      f->remaining=f->n;f->it=it;it->flight=f;
      struct flight **end=&c->flights;
      while(*end)end=&(*end)->next;
      *end=f;
      pthread_mutex_lock(&c->lock);c->nflights++;pthread_mutex_unlock(&c->lock);
      progress(c);
    }
  }
  for(int i=0;i<it->n && !status;i++)if(!P2P(it->calls+i)){
    struct call *k=it->calls+i;
    status=k->kind==K_COUNTED?run_counted(c,k):run_collective(c,k,it);
    it->networked=1;
  }
  for(int i=0;status && i<it->n;i++)if(it->calls[i].arrived){uint64_t none=0;atomic_compare_exchange_strong((_Atomic uint64_t *)it->calls[i].arrived,&none,2);}
  /* every completion word the program (or a persistent receive's caller) waits on, whatever happened (a flight's once
     it retires) */
  if(status){words_fail(it,0);if(!it->flight)words_fail(it,1);}
  if(status && !it->result)it->result=status;
  it->collectives_done=1;
  if(!it->flight)finish(c,it,it->result);
}

/* ---- the worker: a communicator's parts of groups, started in issue order ----
   With nothing to do it spins for WORKER_SPIN_NS after its last part before it sleeps on the
   communicator's condition variable, so a group issued while it spins costs no wakeup (counted where one
   did: WAKEUPS). */
#define WORKER_SPIN_NS UINT64_C(1000000000)
static void launch_free(struct launch *l){
  pthread_mutex_destroy(&l->lock);pthread_cond_destroy(&l->cond);
  /* the group's allocation given back once its last part has ended (the workers are done with it) and its program
     has run (its end), never before */
  if(l->stage)span_release(l->stage,l->persistent?NULL:l->event,l->persistent?0:l->end);
  for(int w=0;w<l->nwaits;w++)point_drop(l->waits+w);
  for(int g=0;g<l->ngates;g++)if(l->gates[g]->run){free(l->gates[g]->words);free(l->gates[g]);}
  tally_release(l->tally);free(l->waits);free(l->marks);free(l->gates);free(l);
}
/* The first failure since the last agreement: its call's index (the least) and its cause. */
/* (a call issued before the last agreement counts no more: its index was of the calls before it) */
static void failed_at(struct ncclComm *c,uint64_t index,const char *cause,uint64_t agreement){
  pthread_mutex_lock(&c->lock);
  if(agreement==c->agreements){
    if(index && (!c->failed || index<c->failed))c->failed=index;
    if(!c->cause[0])snprintf(c->cause,sizeof c->cause,"%s",cause);
  }
  pthread_mutex_unlock(&c->lock);
}
/* A failure revokes the communicator (ULFM's MPI_Comm_revoke [Bland et al., "Post-failure recovery of
   MPI communication capability", IJHPCA 27(3) 2013]): every part in flight stops, the worker closes
   every connection (so each peer's pending and later transfers with this rank fail at once), and the
   communicator makes no call until an agreement (ncclMeshCommAgree) has made its connections again. */
static void comm_revoke(struct ncclComm *c){
  pthread_mutex_lock(&c->lock);
  if(!atomic_exchange(&c->broken,1))c->quiet_value++;
  pthread_cond_broadcast(&c->cond);
  pthread_mutex_unlock(&c->lock);
}
/* The span that starts at p, freed or not; heap.lock held. */
static struct span *span_at(const void *p){
  int i=span_index(p);
  return i<heap.n && heap.spans[i].at==(const unsigned char *)p?heap.spans+i:NULL;
}
/* The memory a failed part received into (its group's allocation, each window allocation a call of it
   wrote in place) held until the bridge has vacated its connections: a transfer already on the wire may
   still land there; the agreement signals `quiet` once they are vacated. */
static void hold(struct ncclComm *c,struct item *it){
  pthread_mutex_lock(&heap.lock);
  struct span *s;
  if(it->launch->placed && (s=span_at(it->launch->placed)))read_add(s,c->quiet,c->quiet_value,NULL);
  for(int i=0;i<it->n;i++)if(it->calls[i].out.span && (s=span_at(it->calls[i].out.span)))read_add(s,c->quiet,c->quiet_value,NULL);
  pthread_mutex_unlock(&heap.lock);
}
/* ---- persistent calls (nccl.h ncclMeshPersistentBegin) ----
   [MPI-4's persistent collectives, MPI_Allreduce_init and MPI_Start: MPI Forum, MPI 4.0, 2021, §6.12] Their
   parts in issue order, one communicator's; started, iteration i opens once `event` reaches value + i *
   stride + 1 (the recording's leading cut: every command buffer before the iteration done, so the last
   iteration's GPU work has read its pieces and waited for, and zeroed, its completion words), and its
   requests are posted ahead (persistent_post); part k of it starts once `event` reaches value + i * stride +
   cut + 1 (its cut, `cut` its place among the recording's cuts: the command buffer ending with its inputs
   complete, as plain stores reach the NIC then), or, without a cut, once the part before it is done (its
   isends released as the ranges of its send buffer they read are published: ranges_wait).  `next` is the
   next iteration x n + part to start, `result` the first failure (the communicator is then revoked, so every
   later part fails at its start and sets its words: the programs waiting on them go on).  The posting: the
   iteration open (`posting`: its index + 1), the first part and call whose isends, and whose irecvs, are not
   all posted, and the first failure of a post.  The gates: `gates` of them in the recording (numbered as the
   recording plays them, struct gate index), each iteration's opened at gate_value + i gates + index + 1 of
   `gate_event` (ncclMeshPersistentGate), which the recording's replay waits for before the work after each. */
struct persistent { struct item **items; int n,capacity; struct ncclComm *comm; void *event; uint64_t value,stride,total,origin;
  _Atomic uint64_t next; ncclResult_t result; uint64_t posting; int spart,scall,rpart,rcall; ncclResult_t post_failed;
  void *gate_event; uint64_t gate_value; int gates; };
/* (process-wide: a recorded step's calls come from every thread that encodes it, PyTorch's autograd engine's too; a
   process records one step at a time, between ncclMeshPersistentBegin and End); `landed` the next receive's word
   (ncclMeshPersistentLanded).  A point-to-point call takes `next` (a send) or `landed` (a receive) as it is issued,
   since one group holds many of them; a collective takes `next` as its group ends. */
static struct { pthread_mutex_t lock; struct persistent *building; struct { const uint64_t *ready; uint64_t range,bytes; int fresh,given; } next;
  uint64_t *landed; int cut,groups; struct { uint64_t depth,stride,words; int given; } slots; } persisting={PTHREAD_MUTEX_INITIALIZER,NULL,{0},NULL,0,0,{0}};
static uint64_t persistent_iteration(const struct item *it){return atomic_load(&it->run->next)/(uint64_t)it->run->n;}
static ncclResult_t persistent_failure(struct ncclComm *c){struct persistent *p=__atomic_load_n(&c->run,__ATOMIC_ACQUIRE);return p?p->post_failed:ncclSuccess;}
static void persistent_ended(struct ncclComm *c,struct item *it,ncclResult_t result){
  struct persistent *p=it->run;
  if(result && !p->result)p->result=result;
  pthread_mutex_lock(&c->lock);
  if(atomic_fetch_add(&p->next,1)+1>=p->total)__atomic_store_n(&c->run,NULL,__ATOMIC_RELEASE);
  pthread_cond_broadcast(&c->cond);
  pthread_mutex_unlock(&c->lock);
}
/* Request q of persistent call k (plan order, as run_collective takes them) posted: an isend held, or an
   irecv; whether it was (not: its connection's request ring is full, or the post failed: post_failed). */
static int persistent_request(struct ncclComm *c,struct persistent *p,struct call *k,uint32_t q){
  const size_t e=type_bytes[k->type];
  if(P2P(k)){
    /* a point-to-point call's one request: its isend held, or its irecv into the caller's buffer, its end stored into
       the caller's word */
    struct peer *peer=c->peers+k->peer;
    void *request=NULL;
    int result=k->kind==K_SEND?mesh_net_isend_held(peer->send[CH_P2P],k->wire,k->count*e,window.mh,NULL,&request):
      mesh_net_irecv_word(peer->recv[CH_P2P],k->wire,k->count*e,window.mh,k->word,&request);
    if(result){if(!p->post_failed)p->post_failed=net_failure(c,result,k->kind==K_SEND?"a held isend posted ahead":"an irecv posted ahead");return 0;}
    if(!request)return 0;
    k->handles[0]=request;
    return 1;
  }
  for(uint32_t i=0,w=0;i<k->nsteps;i++){
    const struct mesh_step *s=k->steps+i;
    if(!s->piece.elements)continue;
    const uint32_t n=chunks_of(s->piece.elements,e);
    if(q>=w+n){w+=n;continue;}
    const uint32_t j=q-w;
    struct peer *peer=c->peers+s->peer;
    const size_t lo=(s->first+chunk_first(s->piece.elements,e,j))*e,bytes=chunk_count(s->piece.elements,e,j)*e;
    void *request=NULL;int result;
    if(s->op==MESH_STEP_SEND){
      unsigned char *from=k->fresh && !written(k,i,lo,lo+bytes)?(unsigned char *)given(k,lo):k->operand+lo;
      result=mesh_net_isend_held(peer->send[CH_COLL],from,bytes,window.mh,(uint64_t *)(k->words+q),&request);
    } else {
      unsigned char *into=s->op==MESH_STEP_COPY?k->operand+lo:k->pieces[i].at+(lo-s->first*e);
      result=mesh_net_irecv_word(peer->recv[CH_COLL],into,bytes,window.mh,(uint64_t *)(k->words+q),&request);
    }
    if(result){if(!p->post_failed)p->post_failed=net_failure(c,result,s->op==MESH_STEP_SEND?"a held isend posted ahead":"an irecv posted ahead");return 0;}
    if(!request)return 0;
    k->handles[q]=request;
    return 1;
  }
  return 0;
}
/* Whether request q of call k (plan order) is an isend. */
static int request_sends(const struct call *k,uint32_t q){
  const size_t e=type_bytes[k->type];
  if(P2P(k))return k->kind==K_SEND;
  for(uint32_t i=0,w=0;i<k->nsteps;i++){
    if(!k->steps[i].piece.elements)continue;
    w+=chunks_of(k->steps[i].piece.elements,e);
    if(q<w)return k->steps[i].op==MESH_STEP_SEND;
  }
  return 0;
}
/* The open iteration's requests posted in order as far as the request rings take them, each connection's in
   the order of its peer's: the irecvs of a call posted ahead (struct call `ahead`, after its `post_after` cut), and
   a call's that is not passed once its own part has posted them; a point-to-point call's irecv as its iteration
   opens (the caller opens it once the iteration before it is done with the buffer); every call's isends, held.  The
   irecvs first: a peer's first send of the iteration waits for its receive's grant, while this rank's isends are
   held until the caller's GPU work publishes their bytes (b0-litert-heads-p1024 and p1-pipelined, metal-microbench
   output_data/stall-20261001: the irecvs posted after some 64 isends, each step's first crossing waited 51 and 86
   us for its grant, against 8 us for the others). */
static void persistent_post(struct ncclComm *c){
  struct persistent *p=c->run;
  if(!p || !p->posting || p->post_failed || atomic_load(&c->broken))return;
  for(;p->rpart<p->n;p->rcall=0,p->rpart++)
    for(;p->rcall<p->items[p->rpart]->n;p->rcall++){
      struct call *k=p->items[p->rpart]->calls+p->rcall;
      if(P2P(k)){
        if(k->kind==K_RECV && !k->rpost){if(!persistent_request(c,p,k,0))goto sends;k->rpost=1;}
        continue;
      }
      if(!k->ahead){if(!k->posted)goto sends;continue;}
      /* a receive buffer free only after a cut of the iteration: once that cut is passed */
      if(k->post_after && k->rpost<k->nwords &&
         nccl_mesh_event_value(p->event)<p->value+(p->posting-1)*p->stride+(uint64_t)k->post_after+1)goto sends;
      for(;k->rpost<k->nwords;k->rpost++)
        if(!request_sends(k,k->rpost) && !persistent_request(c,p,k,k->rpost))goto sends;
    }
  sends:
  for(;p->spart<p->n;p->scall=0,p->spart++)
    for(;p->scall<p->items[p->spart]->n;p->scall++){
      struct call *k=p->items[p->spart]->calls+p->scall;
      if(P2P(k)){
        if(k->kind==K_SEND && !k->spost){if(!persistent_request(c,p,k,0))return;k->spost=1;}
        continue;
      }
      for(;k->spost<k->nwords;k->spost++)
        if(request_sends(k,k->spost) && !persistent_request(c,p,k,k->spost))return;
    }
}
static void persistent_open(struct persistent *p,uint64_t i){
  for(int x=0;x<p->n;x++)for(int j=0;j<p->items[x]->n;j++){
    struct call *k=p->items[x]->calls+j;
    memset(k->handles,0,(k->nwords?k->nwords:1)*sizeof *k->handles);k->spost=k->rpost=0;k->posted=0;
    /* a call in slots: the iteration's slot's buffer, word and tally */
    if(k->slot_depth){
      const uint64_t slot=p->origin+i,ring=slot%k->slot_depth;
      k->wire=k->wire0+ring*k->slot_stride;
      if(k->kind==K_SEND)k->ready=(const _Atomic uint64_t *)((const unsigned char *)k->ready0+slot*k->slot_words);
      else{k->word=(uint64_t *)((unsigned char *)k->word0+slot*k->slot_words);k->words=(_Atomic uint64_t *)k->word;}
      k->tally=k->tallies+ring*TALLIES;
    }
  }
  /* the iteration's gates (its words and landing words zeroed by the iteration before it, as its leading cut
     is passed) */
  for(int x=0;x<p->n;x++){
    struct launch *l=p->items[x]->launch;
    for(int g=0;g<l->ngates;g++){
      struct gate *gate=l->gates[g];
      if(gate->index<0)continue;
      gate->event=p->gate_event;gate->value=p->gate_value+i*(uint64_t)p->gates+(uint64_t)gate->index+1;
      gate_arm(gate);
    }
  }
  p->spart=p->scall=p->rpart=p->rcall=0;p->posting=i+1;
}
static void start(struct ncclComm *c,struct item *it);
/* The next part of the communicator's persistent calls, started once its iteration is open and its cut is
   reached (none: at once); whether it was. */
static int persistent_step(struct ncclComm *c,struct persistent *p){
  const uint64_t next=atomic_load(&p->next);
  if(next>=p->total)return 0;
  const uint64_t i=next/(uint64_t)p->n,k=next%(uint64_t)p->n,base=p->value+i*p->stride,reached=nccl_mesh_event_value(p->event);
  if(!k && p->posting!=i+1){
    if(reached<base+1)return 0;
    persistent_open(p,i);
  }
  persistent_post(c);
  struct item *it=p->items[k];
  if(it->cut && reached<base+(uint64_t)it->cut+1)return 0;
  const uint64_t now=now_ns();
  for(int j=0;j<it->n;j++)if(it->calls[j].tally){
    memset(it->calls[j].tally,0,TALLIES*sizeof *it->calls[j].tally);
    if(it->cut)it->calls[j].tally[READIED]=now;
  }
  it->result=ncclSuccess;it->collectives_done=0;it->networked=0;
  pthread_mutex_lock(&c->lock);c->busy=1;pthread_mutex_unlock(&c->lock);
  start(c,it);
  pthread_mutex_lock(&c->lock);c->busy=0;pthread_cond_broadcast(&c->cond);pthread_mutex_unlock(&c->lock);
  return 1;
}

/* A part ended: its words are all set now (each by the bridge, or 2 where the part failed), so its gates open;
   a failure (one the bridge observed, or a call refused) revokes the communicator. */
static void finish(struct ncclComm *c,struct item *it,ncclResult_t result){
  struct launch *l=it->launch;
  /* a failed part's memory held before its gates open (its program may then end, and its allocations go) */
  if(result){
    uint64_t first=0;
    for(int i=0;i<it->n;i++)if(!first || it->calls[i].index<first)first=it->calls[i].index;
    failed_at(c,first,c->error,it->calls[0].agreement);
    if(it->networked || (result!=ncclInvalidArgument && result!=ncclInvalidUsage)){comm_revoke(c);hold(c,it);}
  }
  gates_check(c);
  stats_record();
  const uint64_t now=now_ns();
  for(int i=0;i<it->n;i++)if(it->calls[i].tally){it->calls[i].tally[ENDED]=now;__atomic_store_n(it->calls[i].tally+DONE,1,__ATOMIC_RELEASE);}
  if(it->run){persistent_ended(c,it,result);return;}
  pthread_mutex_lock(&l->lock);
  if(result && !l->result)l->result=result;
  int last=!--l->items;
  if(result && !l->sync){int none=0;atomic_compare_exchange_strong(&c->async,&none,(int)result);}
  int sync=l->sync;
  if(last && sync)pthread_cond_broadcast(&l->cond);
  pthread_mutex_unlock(&l->lock);
  if(last && !sync)launch_free(l);
  for(int i=0;i<it->n;i++){free(it->calls[i].steps);free(it->calls[i].pieces);free(it->calls[i].segments);}
  free(it->calls);free(it);
}
/* Whether the part's program has published its gate word (the regions' recorded points reached and its
   operands copied in), or, without a gate, whether the recorded points are reached. */
static int ready(struct item *it){
  struct launch *l=it->launch;
  for(int w=0;w<l->nwaits;w++)if(!reached(l->waits[w]))return 0;
  return !l->gate || atomic_load_explicit(l->gate,memory_order_acquire);
}
static void *worker(void *argument){
  struct ncclComm *c=argument;
  pthread_setname_np("nccl-mesh.comm");
  for(uint64_t busy=now_ns();;){
    pthread_mutex_lock(&c->lock);
    const int closing=atomic_load(&c->broken) && c->open && !c->agreeing;
    if(!c->head && !c->flights && !c->run && !c->stopping && !closing && now_ns()-busy>WORKER_SPIN_NS){
      c->parked=1;
      while(!c->head && !c->flights && !c->run && !c->stopping && !(atomic_load(&c->broken) && c->open && !c->agreeing))pthread_cond_wait(&c->cond,&c->lock);
      c->parked=0;c->woke=c->head!=NULL || c->run!=NULL;
    }
    if(atomic_load(&c->broken) && c->open && !c->agreeing)close_peers(c,c->peers,1);
    struct item *it=c->head;
    struct persistent *run=c->run;
    int flying=c->flights!=NULL,room=c->nflights<FLIGHTS;
    if(!it && !flying && !run && c->stopping){pthread_mutex_unlock(&c->lock);break;}
    pthread_mutex_unlock(&c->lock);
    if(run){
      if(persistent_step(c,run))busy=now_ns();
      else{gates_check(c);sched_yield();}
      continue;
    }
    if(!it && !flying){gates_check(c);sched_yield();continue;}
    busy=now_ns();
    progress(c);
    if(it && room){
      int go=ready(it);
      if(!go && atomic_load(&c->gpu_failed)){go=1;it->result=FAIL(c,ncclUnhandledCudaError,"waiting for the group's gate: a GPU program of the communicator failed");}
      if(!go && stopped(c)){go=1;it->result=stop_reason(c,"waiting for the group's gate");}
      if(go){
        pthread_mutex_lock(&c->lock);
        c->head=it->next;if(!c->head)c->tail=NULL;
        c->busy=1;
        pthread_mutex_unlock(&c->lock);
        if(it->result){
          words_fail(it,0);words_fail(it,1);
          it->collectives_done=1;finish(c,it,it->result);
        }
        else start(c,it);
        pthread_mutex_lock(&c->lock);
        c->busy=0;pthread_cond_broadcast(&c->cond);
        pthread_mutex_unlock(&c->lock);
        busy=now_ns();
        continue;
      }
    }
    /* a part waiting for its gate or its recorded points: the word or value is polled, not slept on */
    sched_yield();
  }
  return NULL;
}
static void drain(struct ncclComm *c){
  pthread_mutex_lock(&c->lock);
  while(c->head || c->busy || c->nflights)pthread_cond_wait(&c->cond,&c->lock);
  pthread_mutex_unlock(&c->lock);
}

/* ---- communicators ---- */
static struct ncclComm *comm_new(uint64_t key,int rank,int nranks){
  struct ncclComm *c=calloc(1,sizeof *c);
  if(!c)return NULL;
  c->key=key;c->rank=rank;c->nranks=nranks;
  pthread_mutex_init(&c->lock,NULL);pthread_cond_init(&c->cond,NULL);pthread_mutex_init(&c->gates_lock,NULL);
  return c;
}
static void comm_free(struct ncclComm *c){
  if(c->started){
    pthread_mutex_lock(&c->lock);c->stopping=1;pthread_cond_broadcast(&c->cond);pthread_mutex_unlock(&c->lock);
    pthread_join(c->worker,NULL);
  }
  /* the programs that use it end first (each ends: run, or failed and released), however late */
  while(atomic_load(&c->programs))usleep(100);
  close_peers(c,c->peers,0);
  if(c->net)mesh_net_finalize(c->net);
  for(int p=0;c->plans && p<PLANS;p++){free(c->plans[p].segments);free(c->plans[p].steps);}
  free(c->nodes);free(c->voted);free(c->owner);free(c->seen);free(c->pairs);free(c->stated_pairs);free(c->cost);free(c->plans);free(c->closed);free(c->used);
  free(c->armed);
  if(c->landed)nccl_mesh_release(c->landed);
  if(c->quiet)nccl_mesh_release(c->quiet);
  if(c->event)nccl_mesh_release(c->event);
  if(c->queue)nccl_mesh_release(c->queue);
  pthread_mutex_destroy(&c->lock);pthread_cond_destroy(&c->cond);pthread_mutex_destroy(&c->gates_lock);
  free(c->peers);free(c->blocked);free(c->ops);free(c);
}
/* Attached to the bridge, its GPU queue and events made, connected to every linked rank, its worker started. */
static ncclResult_t comm_start(struct ncclComm *c){
  ncclResult_t status=global_attach();
  if(!status){
    c->queue=nccl_mesh_queue_create();
    if(c->queue){c->event=nccl_mesh_event_create(c->queue);c->quiet=nccl_mesh_event_create(c->queue);c->landed=nccl_mesh_event_create(c->queue);}
    if(!c->event || !c->quiet || !c->landed)status=FAIL(c,ncclUnhandledCudaError,"the communicator's Metal queue and events");
  }
  c->peers=calloc((size_t)c->nranks,sizeof *c->peers);
  c->blocked=calloc((size_t)c->nranks*2,sizeof *c->blocked);
  /* an agreement closes the old connections and its own: twice a rank's */
  c->closed=calloc((size_t)c->nranks*CHANNELS*4,sizeof *c->closed);
  if(!status && (!c->peers || !c->blocked || !c->closed))status=FAIL(c,ncclSystemError,"peer table allocation");
  int result=status?0:mesh_net_init(&c->net,c->key,NULL,NULL,NULL);
  if(!status && result)status=net_failure(c,result,"mesh_net_init (the bridge region MESH_REGION)");
  if(!status)status=connect_peers(c);
  if(!status){
    if(pthread_create(&c->worker,NULL,worker,c))status=FAIL(c,ncclSystemError,"worker thread");
    else c->started=1;
  }
  return status;
}

ncclResult_t ncclGetVersion(int *version){
  if(!version)return FAIL(NULL,ncclInvalidArgument,"ncclGetVersion: version is NULL");
  *version=NCCL_VERSION_CODE;
  return ncclSuccess;
}
ncclResult_t ncclMeshUniqueIdOf(uint64_t key,ncclUniqueId *id){
  if(!id || !key)return FAIL(NULL,ncclInvalidArgument,"ncclMeshUniqueIdOf: %s",id?"key 0 names no clique":"uniqueId is NULL");
  struct uid made={UID_MAGIC,NCCL_VERSION_CODE,key};
  memset(id,0,sizeof *id);memcpy(id->internal,&made,sizeof made);
  return ncclSuccess;
}
ncclResult_t ncclGetUniqueId(ncclUniqueId *id){
  uint64_t key=0;
  while(!key)key=((uint64_t)arc4random()<<32)|arc4random();
  return ncclMeshUniqueIdOf(key,id);
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm,int nranks,ncclUniqueId commId,int rank,ncclConfig_t *config){
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRank: comm is NULL");
  *comm=NULL;
  struct uid id;memcpy(&id,commId.internal,sizeof id);
  if(id.magic!=UID_MAGIC || !id.key)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRank: not a unique id from ncclGetUniqueId");
  if(nranks<1 || rank<0 || rank>=nranks)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRank: rank %d of %d",rank,nranks);
  if(config && config->magic!=NCCL_API_MAGIC)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRankConfig: config not initialized with NCCL_CONFIG_INITIALIZER");
  struct ncclComm *c=comm_new(id.key,rank,nranks);
  if(!c)return FAIL(NULL,ncclSystemError,"allocation");
  ncclResult_t status=links_for(c,config);
  if(!status)status=comm_start(c);
  if(status){snprintf(last_error,sizeof last_error,"%s",c->error);comm_free(c);return status;}
  *comm=c;
  return ncclSuccess;
}
ncclResult_t ncclCommInitRank(ncclComm_t *comm,int nranks,ncclUniqueId commId,int rank){
  return ncclCommInitRankConfig(comm,nranks,commId,rank,NULL);
}
/* One process holds one rank of a node: a node has one Metal device. */
ncclResult_t ncclCommInitAll(ncclComm_t *comm,int ndev,const int *devlist){
  if(!comm || ndev<1)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitAll: %d devices",ndev);
  if(ndev!=1 || (devlist && devlist[0]!=0))
    return FAIL(NULL,ncclInvalidArgument,"ncclCommInitAll: a node has one Metal device (device 0); ranks on other nodes join with ncclCommInitRank");
  ncclUniqueId id;ncclGetUniqueId(&id);
  return ncclCommInitRank(comm,1,id,0);
}
ncclResult_t ncclCommFinalize(ncclComm_t comm){
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"ncclCommFinalize: comm is NULL");
  drain(comm);
  comm->finalized=1;
  return (ncclResult_t)atomic_load(&comm->async);
}
ncclResult_t ncclCommDestroy(ncclComm_t comm){
  if(!comm)return ncclSuccess;
  drain(comm);
  comm_free(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommAbort(ncclComm_t comm){
  if(!comm)return ncclSuccess;
  atomic_store(&comm->aborting,1);
  drain(comm);
  comm_free(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommCount(const ncclComm_t comm,int *count){
  if(!comm || !count)return FAIL(NULL,ncclInvalidArgument,"ncclCommCount: NULL argument");
  *count=comm->nranks;return ncclSuccess;
}
ncclResult_t ncclCommUserRank(const ncclComm_t comm,int *rank){
  if(!comm || !rank)return FAIL(NULL,ncclInvalidArgument,"ncclCommUserRank: NULL argument");
  *rank=comm->rank;return ncclSuccess;
}
ncclResult_t ncclCommCuDevice(const ncclComm_t comm,int *device){
  if(!comm || !device)return FAIL(NULL,ncclInvalidArgument,"ncclCommCuDevice: NULL argument");
  *device=0;return ncclSuccess;
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,ncclResult_t *asyncError){
  if(!comm || !asyncError)return FAIL(NULL,ncclInvalidArgument,"ncclCommGetAsyncError: NULL argument");
  *asyncError=(ncclResult_t)atomic_load(&comm->async);return ncclSuccess;
}
const char *ncclGetErrorString(ncclResult_t result){
  switch(result){
  case ncclSuccess: return "no error";
  case ncclUnhandledCudaError: return "unhandled Metal error";
  case ncclSystemError: return "unhandled system error (set MESH_NCCL_DEBUG=1 for details)";
  case ncclInternalError: return "internal error";
  case ncclInvalidArgument: return "invalid argument (set MESH_NCCL_DEBUG=1 for details)";
  case ncclInvalidUsage: return "invalid usage (set MESH_NCCL_DEBUG=1 for details)";
  case ncclRemoteError: return "remote process exited or there was a network error";
  case ncclInProgress: return "NCCL operation in progress";
  case ncclTimeout: return "operation timed out";
  default: return "unknown result code";
  }
}
/* The communicator's last failure (copied under its lock into this thread's text), else this thread's. */
const char *ncclGetLastError(ncclComm_t comm){
  if(comm){
    pthread_mutex_lock(&comm->lock);
    if(comm->error[0])snprintf(last_error,sizeof last_error,"%s",comm->error);
    pthread_mutex_unlock(&comm->lock);
  }
  return last_error;
}
struct registration { void *buffer; size_t size; };
ncclResult_t ncclCommRegister(const ncclComm_t comm,void *buff,size_t size,void **handle){
  if(!comm || !handle)return FAIL(comm,ncclInvalidArgument,"ncclCommRegister: NULL argument");
  struct registration *r=malloc(sizeof *r);
  if(!r)return FAIL(comm,ncclSystemError,"allocation");
  *r=(struct registration){buff,size};*handle=r;
  return ncclSuccess;
}
ncclResult_t ncclCommDeregister(const ncclComm_t comm,void *handle){
  (void)comm;free(handle);return ncclSuccess;
}
ncclResult_t ncclMemAlloc(void **ptr,size_t size){
  if(!ptr)return FAIL(NULL,ncclInvalidArgument,"ncclMemAlloc: ptr is NULL");
  ncclResult_t status=global_attach();
  return status?status:span_alloc(size,(unsigned char **)ptr,NULL);
}
ncclResult_t ncclMemFree(void *ptr){
  return ptr?span_release(ptr,NULL,0):ncclSuccess;
}
/* A caller's buffer deallocated (its last reference dropped, by whatever thread): its allocation freed. */
static void owned_gone(void *at){span_release(at,NULL,0);}
ncclResult_t ncclMeshMemAllocBuffer(void **ptr,size_t size,uint64_t options,void **buffer){
  if(!ptr || !buffer)return FAIL(NULL,ncclInvalidArgument,"ncclMeshMemAllocBuffer: NULL argument");
  ncclResult_t status=global_attach();
  return status?status:span_alloc_(size,(unsigned char **)ptr,NULL,1,options,buffer);
}
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op,void *scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm){
  if(!op || !scalar || !comm || (unsigned)datatype>=ncclNumTypes)return FAIL(comm,ncclInvalidArgument,"ncclRedOpCreatePreMulSum: invalid argument");
  if(residence!=ncclScalarHostImmediate)return FAIL(comm,ncclInvalidArgument,"ncclRedOpCreatePreMulSum: the host path reads the scalar now (ncclScalarHostImmediate)");
  int slot=0;
  while(slot<comm->nops && comm->ops[slot].used)slot++;
  if(slot==comm->nops){
    struct op_entry *grown=realloc(comm->ops,(size_t)(comm->nops+16)*sizeof *grown);
    if(!grown)return FAIL(comm,ncclSystemError,"allocation");
    memset(grown+comm->nops,0,16*sizeof *grown);comm->ops=grown;comm->nops+=16;
  }
  comm->ops[slot].used=1;comm->ops[slot].type=datatype;
  memset(comm->ops[slot].scalar,0,8);memcpy(comm->ops[slot].scalar,scalar,type_bytes[datatype]);
  *op=(ncclRedOp_t)(ncclNumOps+slot);
  return ncclSuccess;
}
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm){
  int slot=(int)op-ncclNumOps;
  if(!comm || slot<0 || slot>=comm->nops || !comm->ops[slot].used)return FAIL(comm,ncclInvalidArgument,"ncclRedOpDestroy: no such operator");
  comm->ops[slot].used=0;
  return ncclSuccess;
}

/* ---- groups ---- */
static _Thread_local struct { int depth,n,capacity; struct call *calls; ncclResult_t error; } group;
static _Thread_local struct { int n; int how[4096],root[4096]; uint64_t epoch[4096]; struct tally *tally; } plans;
static pthread_mutex_t stream_lock=PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclGroupStart(void){group.depth++;return ncclSuccess;}

/* A reduce-scatter's or all-gather's elements in the segments of the ranks before `rank`. */
static uint64_t before(const struct call *k,int rank){
  uint64_t at=0;
  for(int r=0;r<rank;r++)at+=k->segments?k->segments[r]:k->count;
  return at;
}
static void drain(struct ncclComm *c);
/* The snapshot a call plans on: the table read again once its epoch has moved, and the maps made again.  A
   call in flight keeps the plan it was made with whatever the table does meanwhile; the next plans on the new
   contents.  A revoked communicator makes no call; a connection its bridge failed (the peer's process exited, or
   its bridge left the mesh: a link lost or a bridge restarted only suspends it, and its session resumes), or a
   rank no stated link reaches, fails the
   call (and revokes the communicator: ncclGroupEnd), and ncclMeshCommAgree makes the connections again.  The
   connections are checked once the epoch moves and after an agreement, not every call: a call's hot path is the
   epoch word; a connection failed with no move fails its call on the worker (revoking) instead. */
static ncclResult_t refresh(struct ncclComm *c){
  pthread_mutex_lock(&c->lock);const int agreeing=c->agreeing;pthread_mutex_unlock(&c->lock);
  if(agreeing)return FAIL(c,ncclInvalidUsage,"a call while the communicator agrees (ncclMeshCommAgree)");
  if(atomic_load(&c->broken)){
    char cause[256];uint64_t failed;
    pthread_mutex_lock(&c->lock);snprintf(cause,sizeof cause,"%s",c->cause);failed=c->failed;pthread_mutex_unlock(&c->lock);
    return FAIL(c,ncclRemoteError,"revoked: call %llu since the last agreement failed (%s); the communicator makes no call before "
                "ncclMeshCommAgree",(unsigned long long)failed,cause);
  }
  if(atomic_load_explicit(&c->table->epoch,memory_order_acquire)!=c->epoch){
    c->epoch=mesh_link_table_read(c->table,c->seen);
    maps(c);
    c->alive=0;
  }
  /* a connection its bridge failed (its peer's process exited, or its bridge left the mesh: a session that resumes
     keeps it) */
  int lost=-1;
  pthread_mutex_lock(&c->lock);
  for(int p=0;!c->alive && lost<0 && p<c->nranks;p++)
    for(int ch=0;ch<CHANNELS;ch++)if((c->peers[p].send[ch] && !mesh_net_alive(c->peers[p].send[ch])) ||
                                     (c->peers[p].recv[ch] && !mesh_net_alive(c->peers[p].recv[ch])))lost=p;
  pthread_mutex_unlock(&c->lock);
  if(lost>=0)
    return FAIL(c,ncclRemoteError,"the connection with rank %d is lost: its peer closed it, its process exited, its bridge left the mesh or paired "
                "again as another instance (ncclMeshCommAgree makes it again)",lost);
  c->alive=1;
  return ncclSuccess;
}
/* Whether this rank reaches rank q: a link the stated map has and a connection made on it. */
static int reaches(struct ncclComm *c,int q,int channel){
  pthread_mutex_lock(&c->lock);
  const int connected=c->peers[q].send[channel] && c->peers[q].recv[channel];
  pthread_mutex_unlock(&c->lock);
  return linked(&c->stated,c->rank,q) && connected;
}
/* The ranks whose link is stated but down, for a failure's message. */
static void down_links(struct ncclComm *c,char *text,size_t size){
  size_t at=0;text[0]=0;
  for(int a=0;a<c->nranks;a++)for(int b=a+1;b<c->nranks;b++){
    const struct mesh_link_state *x=mesh_link_at(c->seen,c->nodes[a],c->nodes[b]),*y=mesh_link_at(c->seen,c->nodes[b],c->nodes[a]);
    if(x->stated && y->stated && !(x->up && y->up) && at<size)
      at+=(size_t)snprintf(text+at,size-at,"%sranks %d-%d (nodes %u-%u)",at?", ":"",a,b,c->nodes[a],c->nodes[b]);
  }
}
/* The operator, the algorithm and this rank's plan of a collective (the one kept for this epoch and
   call, else the planner's, then kept); a point-to-point call's peer checked. */
static ncclResult_t resolve(struct call *k){
  struct ncclComm *c=k->comm;
  ncclResult_t status=refresh(c);
  if(status)return status;
  k->epoch=c->epoch;
  for(int q=0;k->kind==K_COUNTED && q<c->nranks;q++)if(q!=c->rank && !reaches(c,q,CH_COLL)){
    char down[256];down_links(c,down,sizeof down);
    return FAIL(c,*down?ncclRemoteError:ncclInvalidUsage,"a counted all-to-all: no link joins rank %d to rank %d at the link map's epoch %llu%s%s",q,
                c->rank,(unsigned long long)c->epoch,*down?"; down: ":"",down);
  }
  if(k->kind==K_COUNTED)return ncclSuccess;
  if(k->kind>=K_SEND){
    if(k->peer!=c->rank && !reaches(c,k->peer,CH_P2P)){
      char down[256];down_links(c,down,sizeof down);
      return FAIL(c,*down?ncclRemoteError:ncclInvalidUsage,"%s rank %d: no link joins rank %d to it at the link map's epoch %llu%s%s",
                  k->kind==K_SEND?"ncclSend to":"ncclRecv from",k->peer,c->rank,(unsigned long long)c->epoch,*down?"; down: ":
                  " (the mesh does not forward point-to-point)",down);
    }
    return ncclSuccess;
  }
  k->elements=k->kind==MESH_REDUCE_SCATTER || k->kind==MESH_ALLGATHER?before(k,c->nranks):k->count;
  k->combine=ncclSum;
  if(k->kind==MESH_ALLREDUCE || k->kind==MESH_REDUCE || k->kind==MESH_REDUCE_SCATTER){
    if(k->op<ncclAvg)k->combine=k->op;
    else if(k->op==ncclAvg){
      if(integral(k->type))k->postdivide=1;
      else{k->premultiply=1;reciprocal(k->scalar,k->type,c->nranks);}
    } else {
      int slot=k->op-ncclNumOps;
      if(slot<0 || slot>=c->nops || !c->ops[slot].used)return FAIL(c,ncclInvalidArgument,"operator %d was not created on this communicator",k->op);
      if(c->ops[slot].type!=k->type)return FAIL(c,ncclInvalidArgument,"a PreMulSum operator reduces the datatype it was created for");
      k->premultiply=1;memcpy(k->scalar,c->ops[slot].scalar,8);
    }
  }
  if(c->nranks==1)return ncclSuccess;
  struct mesh_operand operand={(uint32_t)k->type,(uint32_t)type_bytes[k->type],k->elements};
  uint64_t hash=mix(c->epoch^mix(((uint64_t)k->kind<<56)^((uint64_t)k->how<<40)^((uint64_t)k->root<<24)^((uint64_t)k->type<<16)^
    (uint64_t)k->force)^k->elements);
  for(int r=0;k->segments && r<c->nranks;r++)hash=mix(hash^k->segments[r]);
  struct plan *p=c->plans+hash%PLANS;
  int kept=p->nsteps && p->epoch==c->epoch && p->kind==k->kind && p->how==k->how && p->root==k->root && p->type==(int)k->type &&
    p->force==k->force && p->elements==k->elements && p->uneven==(k->segments!=NULL) &&
    (!k->segments || !memcmp(p->segments,k->segments,(size_t)c->nranks*sizeof *k->segments));
  if(!kept){
    /* on the links up, else on the links stated (their sessions resume: the call waits for them) */
    const struct mesh_link_map *map=&c->map;
    struct mesh_collective chosen=choose(map,k,operand);
    if(chosen.how==MESH_UNAVAILABLE)chosen=choose(map=&c->stated,k,operand);
    if(chosen.how==MESH_UNAVAILABLE)
      return FAIL(c,ncclInvalidUsage,"no algorithm of the selection %#x carries this collective (%d) of %llu elements on the "
        "link map's epoch %llu",k->how,k->kind,(unsigned long long)k->elements,(unsigned long long)c->epoch);
    uint32_t nsteps=mesh_collective_plan(map,(uint32_t)c->rank,chosen,operand,p->steps);
    if(!nsteps)return FAIL(c,ncclInternalError,"the planner gave rank %d no steps",c->rank);
    p->nsteps=nsteps;p->epoch=c->epoch;p->kind=k->kind;p->how=k->how;p->root=k->root;p->type=(int)k->type;p->force=k->force;
    p->elements=k->elements;p->uneven=k->segments!=NULL;p->chosen=chosen;
    if(k->segments)memcpy(p->segments,k->segments,(size_t)c->nranks*sizeof *k->segments);
  }
  k->chosen=p->chosen;k->chosen.segments=k->segments;
  k->steps=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *k->steps);
  k->pieces=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *k->pieces);
  if(!k->steps || !k->pieces)return FAIL(c,ncclSystemError,"allocation");
  k->nsteps=p->nsteps;memcpy(k->steps,p->steps,p->nsteps*sizeof *k->steps);
  pthread_mutex_lock(&c->lock);
  uint32_t unconnected=UINT32_MAX;
  for(uint32_t i=0;i<k->nsteps && unconnected==UINT32_MAX;i++)if(!c->peers[k->steps[i].peer].send[CH_COLL])unconnected=k->steps[i].peer;
  pthread_mutex_unlock(&c->lock);
  /* a rank the last agreement left out (departed then): the communicator agrees again, connecting it if it is back */
  if(unconnected!=UINT32_MAX)
    return FAIL(c,ncclRemoteError,"the plan uses rank %u, which no connection reaches (it %s; ncclMeshCommAgree connects the ranks that stay)",
                unconnected,departed(c,(int)unconnected)?"departed":"was left out of the last agreement");
  return ncclSuccess;
}

static void calls_free(struct call *calls,int n){
  for(int i=0;i<n;i++){free(calls[i].steps);free(calls[i].pieces);free(calls[i].segments);free(calls[i].handles);free(calls[i].ready_base);free(calls[i].tallies);}
  free(calls);
}

/* ---- a group's placement in the window and its GPU program ---- */
/* The bytes of send a call reads (a collective's contribution, the root's data, this rank's segment; a
   send's) and of recv it writes (its result; a receive's). */
static size_t in_bytes(const struct call *k){
  const size_t e=type_bytes[k->type],part=k->count*e;
  if(k->kind==K_COUNTED)return k->rows*k->row*e;
  if(k->kind>=K_SEND)return k->kind==K_SEND?part:0;
  return k->kind==MESH_ALLGATHER?part:k->kind==MESH_BROADCAST && k->comm->rank!=k->root?0:(size_t)k->elements*e;
}
static size_t out_bytes(const struct call *k){
  const size_t e=type_bytes[k->type],part=k->count*e;
  if(k->kind==K_COUNTED)return k->capacity*k->row*e;
  if(k->kind>=K_SEND)return k->kind==K_RECV?part:0;
  return k->kind==MESH_REDUCE_SCATTER?part:k->kind==MESH_REDUCE && k->comm->rank!=k->root?0:(size_t)k->elements*e;
}
/* Where a collective's contribution goes in its operand (an all-gather's: this rank's segment), and
   where its result is (a reduce-scatter's: this rank's segment). */
static unsigned char *input_at(const struct call *k){
  return k->kind==MESH_ALLGATHER?k->operand+before(k,k->comm->rank)*type_bytes[k->type]:k->operand;
}
static unsigned char *output_at(const struct call *k){
  return k->kind==MESH_REDUCE_SCATTER?k->operand+before(k,k->comm->rank)*type_bytes[k->type]:k->operand;
}
static int reduces(const struct call *k){
  for(uint32_t s=0;s<k->nsteps;s++)if(k->steps[s].op==MESH_STEP_REDUCE)return 1;
  return 0;
}
/* ---- reading the send buffer in place (nccl.h) ----
   The operand's byte `lo` holds, until a step writes it, the send buffer's byte lo - own_at(k): a
   contribution the rank's own segment of an all-gather's operand, the whole operand otherwise. */
static size_t own_at(const struct call *k){return k->kind==MESH_ALLGATHER?(size_t)before(k,k->comm->rank)*type_bytes[k->type]:0;}
static const unsigned char *given(const struct call *k,size_t lo){return (const unsigned char *)k->send+(lo-own_at(k));}
/* Whether steps of k's plan before step s write the operand's bytes [lo, hi) (a REDUCE or a COPY): 0 none,
   1 all (one step's piece holds them), 2 some. */
static int written(const struct call *k,uint32_t s,size_t lo,size_t hi){
  const size_t e=type_bytes[k->type];int some=0;
  for(uint32_t q=0;q<s;q++){
    const struct mesh_step *t=k->steps+q;
    if(t->op==MESH_STEP_SEND || !t->piece.elements)continue;
    const size_t a=t->first*e,b=a+t->piece.elements*e;
    if(a<=lo && hi<=b)return 1;
    if(a<hi && lo<b)some=1;
  }
  return some?2:0;
}
/* Whether k can read its send buffer in place: it is an allocation, not premultiplied, and every SEND and
   REDUCE chunk of the plan is either written whole by an earlier step or not at all (then inside the
   send buffer's bytes); with `own`, every byte of the operand the plan writes or sends once written lies
   in this rank's segment (a reduce-scatter's result: its operand can be recvbuff alone). */
static int reads_in_place(const struct call *k,int own){
  const size_t e=type_bytes[k->type],in=in_bytes(k),at=own_at(k),lo_own=(size_t)before(k,k->comm->rank)*e,hi_own=lo_own+k->count*e;
  if(k->kind>=K_SEND || !in || !k->in.span || k->premultiply)return 0;
  for(uint32_t s=0;s<k->nsteps;s++){
    const struct mesh_step *t=k->steps+s;
    if(!t->piece.elements)continue;
    const size_t a=t->first*e,b=a+t->piece.elements*e;
    if(t->op!=MESH_STEP_SEND && own && (a<lo_own || b>hi_own))return 0;
    if(t->op==MESH_STEP_COPY)continue;
    for(uint32_t j=0,n=chunks_of(t->piece.elements,e);j<n;j++){
      const size_t lo=(t->first+chunk_first(t->piece.elements,e,j))*e,hi=lo+chunk_count(t->piece.elements,e,j)*e;
      const int w=written(k,s,lo,hi);
      if(w==2 || (!w && (lo<at || hi>at+in)) || (w && own && t->op==MESH_STEP_SEND && (lo<lo_own || hi>hi_own)))return 0;
    }
  }
  return 1;
}

/* A recorded point the group waits for (the same event's once, at its latest value). */
static void wait_add(struct launch *l,struct point p){
  if(reached(p))return;
  for(int w=0;w<l->nwaits;w++)
    if(l->waits[w].event==p.event){if(p.value>l->waits[w].value)l->waits[w].value=p.value;return;}
  l->waits[l->nwaits]=(struct point){0};
  point_set(l->waits+l->nwaits++,p.event,p.value);
}
/* A buffer where the program reads or writes it: its window allocation's Metal buffer, else one over
   the host pages holding it; retained until the program is committed. */
static ncclResult_t region_of(struct region *r,const void *p,size_t bytes){
  struct span *s=span_of(p,bytes);
  if(s){nccl_mesh_retain(s->buffer);*r=(struct region){s->buffer,s->at,s->at};return ncclSuccess;}
  uint64_t offset;void *buffer=nccl_mesh_buffer(p,bytes,&offset);
  if(!buffer)return FAIL(NULL,ncclUnhandledCudaError,"the host pages of %p (%zu bytes) as a Metal buffer (newBufferWithBytesNoCopy)",p,bytes);
  *r=(struct region){buffer,(const unsigned char *)p-offset,NULL};
  return ncclSuccess;
}
/* Each call's send and recv as regions, and the recorded points the group waits for: of a buffer it
   reads, its allocation's writer; of one it writes, the writer and every reader. */
static ncclResult_t regions(struct call *calls,int n,struct launch *l){
  ncclResult_t status=ncclSuccess;
  pthread_mutex_lock(&heap.lock);
  for(int i=0;i<n && !status;i++){
    struct call *k=calls+i;struct span *s;
    const size_t in=in_bytes(k),out=out_bytes(k);
    if(in && !(status=region_of(&k->in,k->send,in)))
      for(int o=-1;o<(k->in.span?0:heap.nout);o++)
        if((s=o<0?span_of(k->send,in):overlap(heap.outside+o,k->send,in)?heap.outside+o:NULL))wait_add(l,s->wrote);
    if(out && !status && !(status=region_of(&k->out,k->recv,out)))
      for(int o=-1;o<(k->out.span?0:heap.nout);o++)
        if((s=o<0?span_of(k->recv,out):overlap(heap.outside+o,k->recv,out)?heap.outside+o:NULL)){
          wait_add(l,s->wrote);
          for(int r=0;r<s->nread;r++)wait_add(l,s->read[r]);
        }
    const size_t counts=k->kind==K_COUNTED?8*(size_t)before(k,k->comm->nranks):0;
    if(counts && !status && !(status=region_of(&k->cnt,k->counted,counts)))
      for(int o=-1;o<(k->cnt.span?0:heap.nout);o++)
        if((s=o<0?span_of(k->counted,counts):overlap(heap.outside+o,k->counted,counts)?heap.outside+o:NULL))wait_add(l,s->wrote);
    const size_t landed=k->landed?8*(size_t)(before(k,2*k->comm->nranks)-before(k,k->comm->nranks)):0;
    if(landed && !status && !(status=region_of(&k->lnd,k->landed,landed)))
      for(int o=-1;o<(k->lnd.span?0:heap.nout);o++)
        if((s=o<0?span_of(k->landed,landed):overlap(heap.outside+o,k->landed,landed)?heap.outside+o:NULL)){
          wait_add(l,s->wrote);
          for(int r=0;r<s->nread;r++)wait_add(l,s->read[r]);
        }
  }
  pthread_mutex_unlock(&heap.lock);
  return status;
}
static void regions_release(struct call *calls,int n,struct launch *l){
  for(int i=0;i<n;i++){
    if(calls[i].in.buffer)nccl_mesh_release(calls[i].in.buffer);
    if(calls[i].out.buffer)nccl_mesh_release(calls[i].out.buffer);
    if(calls[i].cnt.buffer)nccl_mesh_release(calls[i].cnt.buffer);
    if(calls[i].lnd.buffer)nccl_mesh_release(calls[i].lnd.buffer);
    calls[i].in.buffer=calls[i].out.buffer=calls[i].cnt.buffer=calls[i].lnd.buffer=NULL;
  }
  if(l->own.buffer)nccl_mesh_release(l->own.buffer);
  l->own.buffer=NULL;
}
/* A writer's point replaces the record's; the one it replaces, if not yet reached, stays as a reader's
   (the allocation is not given out again before it). */
static void wrote_set(struct span *s,void *event,uint64_t value,uint64_t *tally){
  if(!reached(s->wrote))read_add(s,s->wrote.event,s->wrote.value,tally);
  point_set(&s->wrote,event,value);
}
/* The group's completion recorded on the allocations it touched (the writer of those it wrote, a
   reader of those it read), and on its stream calls' memory outside the window; its own allocation
   freed at it. */
static void record(struct launch *l,struct call *calls,int n){
  pthread_mutex_lock(&heap.lock);
  for(int i=0;i<n;i++){
    struct call *k=calls+i;struct span *s;
    const size_t in=in_bytes(k),out=out_bytes(k);
    if(in && (s=k->in.span?span_of(k->send,in):l->sync?NULL:outside_of(k->send,in)))read_add(s,l->event,l->end,k->tally);
    if(out && (s=k->out.span?span_of(k->recv,out):l->sync?NULL:outside_of(k->recv,out)))wrote_set(s,l->event,l->end,k->tally);
    const size_t counts=k->kind==K_COUNTED?8*(size_t)before(k,k->comm->nranks):0;
    if(counts && (s=k->cnt.span?span_of(k->counted,counts):l->sync?NULL:outside_of(k->counted,counts)))read_add(s,l->event,l->end,k->tally);
    const size_t landed=k->landed?8*(size_t)(before(k,2*k->comm->nranks)-before(k,k->comm->nranks)):0;
    if(landed && (s=k->lnd.span?span_of(k->landed,landed):l->sync?NULL:outside_of(k->landed,landed)))wrote_set(s,l->event,l->end,k->tally);
    /* the host counts and word the worker writes: an allocation holding them is not given out again before the group's end */
    if(landed && (s=span_of(k->received,landed)))wrote_set(s,l->event,l->end,k->tally);
    if(k->arrived && (s=span_of(k->arrived,8)))wrote_set(s,l->event,l->end,k->tally);
  }
  pthread_mutex_unlock(&heap.lock);
}

/* The operand lies in recv itself where recv is a window allocation and holds the whole result
   (all-reduce, broadcast, all-gather, a reduce's root: 1), in send itself for a reduce-scatter in place
   (recv is rank's segment of send, a window allocation: NCCL's in-place convention, send's other
   segments then undefined: 2), for a reduce-scatter out of place whose send buffer is read in place and
   whose plan writes only this rank's segment, around recv (recv its segment, no other byte of it
   touched: 3), else in the group's own allocation with the pieces and the bytes of the point-to-point
   calls whose buffers lie outside the window (0). */
static int in_place(const struct call *k){
  int whole=k->kind==MESH_ALLREDUCE || k->kind==MESH_BROADCAST || k->kind==MESH_ALLGATHER || (k->kind==MESH_REDUCE && k->comm->rank==k->root);
  if(k->kind<K_SEND && whole && k->out.span)return 1;
  if(k->kind==MESH_REDUCE_SCATTER && k->in.span &&
     (const unsigned char *)k->recv==(const unsigned char *)k->send+before(k,k->comm->rank)*type_bytes[k->type])return 2;
  if(k->kind==MESH_REDUCE_SCATTER && k->out.span && reads_in_place(k,1))return 3;
  return 0;
}
/* A call's isends and irecvs, as the worker posts them (a collective's chunk by chunk in plan order; a
   point-to-point call's one, none to this rank itself), and its REDUCE chunks where a later SEND of the
   call reads a combine (a ring's or tree's partial sent on; else none). */
static uint32_t requests(const struct call *k){
  if(k->kind==K_COUNTED)return 4*(uint32_t)(k->comm->nranks-1);
  if(k->kind>=K_SEND)return k->peer!=k->comm->rank;
  uint32_t n=0;
  for(uint32_t s=0;s<k->nsteps;s++)if(k->steps[s].piece.elements)n+=chunks_of(k->steps[s].piece.elements,type_bytes[k->type]);
  return n;
}
static uint32_t combines(const struct call *k){
  uint32_t n=0;int reduced=0,resent=0;
  for(uint32_t s=0;k->kind<K_SEND && s<k->nsteps;s++){
    if(k->steps[s].op==MESH_STEP_REDUCE){reduced=1;n+=k->steps[s].piece.elements?chunks_of(k->steps[s].piece.elements,type_bytes[k->type]):0;}
    else if(k->steps[s].op==MESH_STEP_SEND && reduced)resent=1;
  }
  return resent?n:0;
}
/* Operands, pieces and staged bytes in the group's allocation (in_place above), then, where the group has
   a program (`words`), its control words, zeroed: each call's requests' and combines', its spins' flags and its
   leaked word and its chunks' landing words (struct call); the gate the workers start at; and the group's landing
   words and final spins' flags, one each of its `ncomms` communicators. */
static ncclResult_t place(struct call *calls,int n,struct launch *l,int words,int ncomms){
  size_t first=0,end=0;
  for(int pass=0;pass<2;pass++){
    size_t used=0;
    for(int i=0;i<n;i++){
      struct call *k=calls+i;
      const size_t e=type_bytes[k->type];
      if(k->kind==K_COUNTED){
        if((in_bytes(k) && !k->in.span) || (out_bytes(k) && !k->out.span) || !k->cnt.span)
          return FAIL(k->comm,ncclInvalidArgument,"a counted all-to-all's send buffer, receive buffer and counts are window allocations (ncclMemAlloc)");
        uint64_t got=0;
        for(int q=0;q<k->comm->nranks;q++)got+=k->segments[k->comm->nranks+q];
        if(pass)k->wire=l->stage+used;
        used+=slice(got*8);
        continue;
      }
      if(k->kind>=K_SEND){
        if(k->peer==k->comm->rank)continue;
        struct region r=k->kind==K_SEND?k->in:k->out;
        if(r.span)k->wire=(unsigned char *)(k->kind==K_SEND?k->send:k->recv);
        else{if(pass)k->wire=l->stage+used;used+=slice(k->count*e);}
        continue;
      }
      const int placed=in_place(k);
      if(placed==1){k->operand=k->recv;k->at=k->out;}
      else if(placed==2){k->operand=(unsigned char *)k->send;k->at=k->in;}
      else if(placed==3){k->operand=(unsigned char *)k->recv-before(k,k->comm->rank)*e;k->at=k->out;}
      else{if(pass){k->operand=l->stage+used;k->at=l->own;}used+=slice((size_t)k->elements*e);}
      if(pass)k->fresh=input_at(k)!=(const unsigned char *)k->send && reads_in_place(k,0);
      for(uint32_t s=0;s<k->nsteps;s++)if(k->steps[s].op==MESH_STEP_REDUCE){
        if(pass)k->pieces[s].at=l->stage+used;
        used+=slice(k->steps[s].piece.elements*e);
      }
    }
    first=used;
    for(int i=0;words && i<n;i++){
      struct call *k=calls+i;
      const uint32_t w=requests(k),c=combines(k);
      if(pass){
        k->words=(_Atomic uint64_t *)(l->stage+used);k->nwords=w;k->combined=c?(_Atomic uint64_t *)(l->stage+used+8*w):NULL;k->ncombined=c;
        k->flags=(_Atomic uint64_t *)(l->stage+used+8*((size_t)w+c));k->landings=k->flags+w+1;
      }
      used+=8*(3*(size_t)w+c+1);
    }
    if(words){if(pass)l->gate=(_Atomic uint64_t *)(l->stage+used);used+=8;}
    if(words){if(pass)l->landing=(_Atomic uint64_t *)(l->stage+used);used+=16*(size_t)ncomms;}
    end=used;
    if(!pass){
      if(!used)return ncclSuccess;
      ncclResult_t status=span_alloc(used,&l->stage,calls[0].tally);
      if(status)return status;
      l->placed=l->stage;
      pthread_mutex_lock(&heap.lock);
      struct span *s=span_of(l->stage,used);
      nccl_mesh_retain(s->buffer);l->own=(struct region){s->buffer,s->at,s->at};
      pthread_mutex_unlock(&heap.lock);
    }
  }
  if(words){memset(l->stage+first,0,end-first);l->zero_at=first;l->zero_words=(end-first)/8;}
  return ncclSuccess;
}
/* Whether a persistent call's requests are posted ahead (struct call `ahead`): every receive of its plan lands
   in a REDUCE step's piece (the group's own), or a COPY step's range of an operand that is the group's own or
   a receive buffer free once cut fresh - 1 of its iteration is passed (`fresh`, then its `post_after`), which
   none of its sends reads; and its requests fit a connection's request ring with room to spare. */
static int ahead_of(struct call *k,int fresh,const struct launch *l){
  if(k->kind>=K_SEND || k->kind==K_COUNTED || !k->nwords || k->nwords>MESH_NET_REQUESTS/2)return 0;
  const size_t e=type_bytes[k->type];
  const int own=k->at.buffer && k->at.buffer==l->own.buffer;
  for(uint32_t s=0;s<k->nsteps;s++){
    const struct mesh_step *t=k->steps+s;
    if(!t->piece.elements || t->op!=MESH_STEP_COPY)continue;
    if(!own && !fresh)return 0;
    if(!own)k->post_after=fresh-1;
    const size_t a=t->first*e,b=a+t->piece.elements*e;
    for(uint32_t q=0;q<k->nsteps;q++){
      const struct mesh_step *u=k->steps+q;
      if(u->op==MESH_STEP_SEND && u->piece.elements && u->first*e<b && a<(u->first+u->piece.elements)*e)return 0;
    }
  }
  return 1;
}
/* A receive from this rank itself: its send (the k-th with the k-th) where the sizes agree, which the
   program copies. */
static const struct call *self_copy(const struct call *calls,int n,int i){
  const struct call *k=calls+i;
  if(k->kind!=K_RECV || k->peer!=k->comm->rank)return NULL;
  int sends=0,receives=0;
  for(int j=0;j<i;j++)receives+=calls[j].kind==K_RECV && calls[j].comm==k->comm && calls[j].peer==k->comm->rank;
  size_t bytes=k->count*type_bytes[k->type];
  for(int j=0;j<n;j++)
    if(calls[j].kind==K_SEND && calls[j].comm==k->comm && calls[j].peer==k->comm->rank && sends++==receives)
      return calls[j].count*type_bytes[calls[j].type]==bytes?calls+j:NULL;
  return NULL;
}
/* Whether call i gives the GPU work: a copy in or out (a buffer outside the window, the operand not
   the buffer itself, a receive from this rank itself), a premultiplication, a combine, a post-division. */
static int gpu_work(const struct call *calls,int n,int i){
  const struct call *k=calls+i;
  (void)n;
  if(k->kind==K_COUNTED)return 1;
  if(k->kind==K_RECV)return k->peer==k->comm->rank || !k->out.span;
  if(k->kind==K_SEND)return k->peer!=k->comm->rank && !k->in.span;
  const size_t in=in_bytes(k),out=out_bytes(k);
  if(in && (k->premultiply || input_at(k)!=(const unsigned char *)k->send))return 1;
  if(reduces(k) || (out && k->postdivide))return 1;
  return out && output_at(k)!=(unsigned char *)k->recv;
}
/* What a program holds until the GPU has run it: the streams' events it signals, and a count on each
   communicator whose words it waits on; its commands (below) until then too. */
struct recording;
/* What a program holds until it has run: its communicators, the events it signals, its commands, and its stream
   (NULL: a communicator's own), which it marks failed where it fails for good. */
struct ran { int ncomms,nevents; struct ncclComm **comms; void **events; struct recording *kept; struct ncclMeshStream *stream; };
/* A group's program: its commands, each object they name retained, encoded once its group has ended (into the
   library's own command buffers, committed) or later into a command buffer the caller hands over (a deferred
   stream's, nccl.h ncclMeshStreamDefer, and so a recording's).  A predicated command runs where the flag word at
   `pred` - 1 of buffer `q` (0: none) is `want`; a gate (struct gate) ends a command buffer, the next beginning with
   its wait. */
enum { R_WAIT, R_SIGNAL, R_KERNEL, R_COPY, R_SPIN, R_PUBLISH, R_COUNTED, R_ZERO, R_GATE };
struct recorded { int kind,which,type,op,nranks,published,want; void *a,*b,*c,*d,*e,*q; uint64_t x,y,z,n,scalar,pred,grid; uint64_t *list;
  void *touch[4]; struct gate *gate; void *held; };
struct recording { struct recording *next; int n,capacity,failed; struct recorded *ops; struct ran *ran; };
/* The threads of a kernel that runs where a spin gave up, after its gate (the network was late, so the program
   was too): a few, so that where the spin saw its words the kernel costs next to nothing. */
#define LATE_GRID 4096
static void add(struct recording *k,struct recorded r){
  if(k->n==k->capacity){
    int capacity=k->capacity?2*k->capacity:32;
    struct recorded *grown=realloc(k->ops,(size_t)capacity*sizeof *grown);
    if(!grown){k->failed=1;free(r.list);return;}
    k->ops=grown;k->capacity=capacity;
  }
  void *const held[6]={r.a,r.b,r.c,r.d,r.e,r.q};
  for(int i=0;i<6;i++)if(held[i])nccl_mesh_retain(held[i]);
  for(int t=0;t<4;t++)if(r.touch[t])nccl_mesh_retain(r.touch[t]);
  k->ops[k->n++]=r;
}
static void program_wait(struct recording *k,struct call *call,void *event,uint64_t value){
  add(k,(struct recorded){.kind=R_WAIT,.a=event,.x=value});
  count(call,GPU_EVENT_WAITS,1);
}
static void program_signal(struct recording *k,void *event,uint64_t value){add(k,(struct recorded){.kind=R_SIGNAL,.a=event,.x=value});}
/* A leaky spin for words of the group's allocation `r` to reach their values (`list`: n pairs of an offset and
   a value): `flag` set once it has seen them all, else `leaked` (a word of k's); the buffers k's requests read and
   write declared used (with `r`'s), so later work on them follows it. */
static void spin(struct recording *k,struct call *call,struct region r,const uint64_t *list,uint32_t n,const _Atomic uint64_t *flag,
  const _Atomic uint64_t *leaked){
  if(!n)return;
  void *touch[4]={0};int m=0;
  void *const each[4]={r.buffer,call->at.buffer,call->in.buffer,call->out.buffer};
  for(int i=0;i<4;i++){int seen=!each[i];for(int j=0;j<m;j++)seen|=touch[j]==each[i];if(!seen)touch[m++]=each[i];}
  uint64_t *copy=malloc(2*n*sizeof *copy);
  if(!copy){k->failed=1;return;}
  memcpy(copy,list,2*n*sizeof *copy);
  if(!gave_up.buffer){
    unsigned char *word=NULL;
    if(!span_alloc(64,&word,NULL)){
      memset(word,0,64);
      pthread_mutex_lock(&heap.lock);
      gave_up.buffer=span_of(word,64)->buffer;nccl_mesh_retain(gave_up.buffer);
      pthread_mutex_unlock(&heap.lock);
      gave_up.word=(_Atomic uint32_t *)word;
    }
  }
  /* without the counter's word the spin would count into the group's own bytes: the program is not made */
  if(!gave_up.buffer){free(copy);k->failed=1;return;}
  struct recorded rec={.kind=R_SPIN,.a=r.buffer,.b=gave_up.buffer,.n=n,.list=copy,.x=off(r,(const void *)flag),.y=off(r,(const void *)leaked)};
  memcpy(rec.touch,touch,sizeof touch);
  add(k,rec);
  count(call,GPU_WORD_WAITS,n);
}
/* A word of the window allocation `r` set to `value` once the program's dispatches before it are done, where the
   group's flag `pred` (NULL: always) is `want`. */
static void publish(struct recording *k,struct region r,const void *word,uint64_t value,struct region own,const _Atomic uint64_t *pred,int want){
  add(k,(struct recorded){.kind=R_PUBLISH,.a=r.buffer,.x=off(r,word),.y=value,.q=pred?own.buffer:NULL,.pred=pred?off(own,(const void *)pred)+1:0,.want=want});
}
/* A kernel over n elements; `published`: its stores system-coherent and fenced, as a word published after
   them lets a SEND read them (nccl-mesh-metal.m); `first` (a combine's, not NULL): its first operand the
   send buffer's bytes there, read in place, not dst's; predicated on the group's flag `pred` (in `own`); at most
   `grid` threads (0: one an element). */
static void kernel(struct recording *s,struct call *k,int which,struct region to,const void *dst,struct region from,const void *src,uint64_t n,int op,
  int published,const void *first,struct region own,const _Atomic uint64_t *pred,int want,uint64_t grid){
  uint64_t scalar;memcpy(&scalar,k->scalar,8);
  void *other=first?k->in.buffer:NULL;const uint64_t at=first?off(k->in,first):0;
  add(s,(struct recorded){.kind=R_KERNEL,.which=which,.a=to.buffer,.x=off(to,dst),.b=from.buffer,.y=off(from,src),.c=other,.z=at,.n=n,
                          .type=(int)k->type,.op=op,.nranks=k->comm->nranks,.scalar=scalar,.published=published,
                          .q=pred?own.buffer:NULL,.pred=pred?off(own,(const void *)pred)+1:0,.want=want,.grid=grid});
  count(k,GPU_KERNELS,1);
}
/* `bytes` copied by the copy kernel (loaded system-coherent where the NIC wrote them: `received`; stored
   system-coherent where a SEND reads them: `published`). */
static void copy_into(struct recording *s,struct call *k,struct region to,const void *dst,struct region from,const void *src,size_t bytes,int received,
  int published){
  add(s,(struct recorded){.kind=R_COPY,.which=received,.published=published,.a=to.buffer,.x=off(to,dst),.b=from.buffer,.y=off(from,src),.n=bytes});
  count(k,GPU_COPY,bytes);
}
/* A counted all-to-all's own rows copied from its send buffer into its receive buffer, their offsets and number
   from the counts (its own; the other ranks' segments for this rank where they landed in the group's
   allocation), and each rank's segment for this rank written to `landed` (if any): the counted kernel's
   arguments (nccl-mesh-metal.m), predicated on the group's flag `pred`. */
static void counted_copy(struct recording *s,struct call *k,struct launch *l,const _Atomic uint64_t *pred,int want,uint64_t grid){
  const int n=k->comm->nranks;
  const uint64_t row=k->row*type_bytes[k->type];
  const uint32_t width=!(row&15)?16:!(row&3)?4:1,na=11+4*(uint32_t)n;
  uint64_t *a=malloc(na*sizeof *a),landed=0;
  if(!a){s->failed=1;return;}
  a[0]=off(k->out,k->recv);a[1]=off(k->in,k->send);a[2]=off(k->cnt,k->counted);a[3]=off(l->own,k->wire);a[4]=row;a[5]=width;a[6]=(uint64_t)n;
  a[7]=(uint64_t)k->comm->rank;a[8]=k->landed?off(k->lnd,k->landed):UINT64_MAX;
  for(int q=0,so=0,ro=0;q<n;q++){
    a[9+4*q]=(uint64_t)so;a[10+4*q]=k->segments[q];a[11+4*q]=(uint64_t)ro;a[12+4*q]=k->segments[n+q];
    so+=(int)k->segments[q];ro+=(int)k->segments[n+q];landed+=k->segments[n+q];
  }
  a[9+4*n]=pred?off(l->own,(const void *)pred)+1:0;a[10+4*n]=(uint64_t)want;
  uint64_t units=k->rows*row/width;
  if(k->landed && units<landed)units=landed;
  void *to=k->lnd.buffer?k->lnd.buffer:k->out.buffer;
  add(s,(struct recorded){.kind=R_COUNTED,.a=k->out.buffer,.b=k->in.buffer,.c=k->cnt.buffer,.d=l->own.buffer,.e=to,.x=na,.n=units,.list=a,
                          .q=pred?l->own.buffer:NULL,.grid=grid});
  count(k,GPU_KERNELS,1);
}
/* A gate: its event and value kept with the command (the group's launch, and its gates, may be gone by the time a
   kept program is encoded: its part ended), a persistent call's the gate itself (its launch lives with the
   persistent calls), whose place among the recording's gates is given as it is played. */
static void program_gate(struct recording *s,struct gate *g){add(s,(struct recorded){.kind=R_GATE,.a=g->event,.x=g->value,.gate=g->run?g:NULL});}
/* A gate's command buffer (nccl.h ncclMeshStreamEncodeWait): `argument`'s. */
typedef void *(*gate_fn)(void *argument,void *program,void *event,uint64_t value);
/* ---- the dispatcher: the library's own command buffers, each committed once its gates are reached ----
   Metal ends a command buffer that waits on an event at its start past its watchdog (about 5 s: a lone one waiting
   8 s on the M5, metal-microbench output_data/maybe-20260930/probe), so the library encodes no such wait: a command
   buffer of its own carries its gates (events and values: a group's gate, a recorded point, another stream's
   end) and the dispatcher commits it once they are reached, each queue's in the order given (a thread of its
   own polls them, and the workers as they open gates).
     A program's command buffers (a run) are committed in order, each once the one before it has run, so one that
   fails is the last of its program the GPU ran.  One that failed by no fault of its own (nccl-mesh-metal.m
   nccl_mesh_program_handler, status 1) runs again: where the program's inputs are retained (`redoable`: no call of
   it in place, no send of it reading a combine, nothing of a caller's kept in it), its recording is played again
   from its start (a new generation; the command buffers of the last one not yet committed are dropped), the same
   commands on the same inputs giving the same result (S3); otherwise, or where it faulted itself, it fails for good
   (program_ran), the rest of it run so that every handler and signal of it happens. */
struct run { _Atomic int refs; struct recording *rec; struct ran *ran; void *queue; int redoable,failed;
  _Atomic uint32_t generation,done; struct run *redo; };
struct ticket { struct run *run; uint32_t generation,index; int last; };
static void program_ran(void *argument,int status);
struct entry { struct entry *next; void *program,*queue; int n,capacity; struct point *gates; struct run *run; uint32_t generation,index; int last;
  struct ticket *ticket; };
static struct { pthread_mutex_t lock; pthread_cond_t cond; struct entry *head,*tail; struct run *redo; int started; } dispatcher=
  {PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,NULL,NULL,NULL,0};
static void run_release(struct run *r){if(atomic_fetch_sub(&r->refs,1)==1)free(r);}
/* The run ended: its program released (failed where any of it failed for good). */
static void run_finish(struct run *r){
  if(r->ran)program_ran(r->ran,r->failed?2:0);
  r->ran=NULL;r->rec=NULL;
}
static void *dispatch_run(void *argument);
/* The dispatcher's thread started (once) and woken; dispatcher.lock held. */
static void dispatch_wake_locked(void){
  if(!dispatcher.started){pthread_t t;if(!pthread_create(&t,NULL,dispatch_run,NULL)){pthread_detach(t);dispatcher.started=1;}}
  pthread_cond_signal(&dispatcher.cond);
}
static void dispatch_wake(void){pthread_mutex_lock(&dispatcher.lock);dispatch_wake_locked();pthread_mutex_unlock(&dispatcher.lock);}
/* A command buffer of a run has run (Metal's completion thread). */
static void entry_ran(void *argument,int status){
  struct ticket *t=argument;struct run *r=t->run;
  if(t->generation==atomic_load(&r->generation)){
    if(status==1 && r->redoable){
      /* run again from its start, by the dispatcher's thread (it encodes) */
      pthread_mutex_lock(&dispatcher.lock);
      atomic_fetch_add(&r->refs,1);r->redo=dispatcher.redo;dispatcher.redo=r;
      dispatch_wake_locked();
      pthread_mutex_unlock(&dispatcher.lock);
    } else {
      if(status)r->failed=1;
      atomic_store(&r->done,t->index+1);
      if(t->last)run_finish(r);
      else dispatch_wake();
    }
  }
  run_release(r);free(t);
}
/* The entries whose gates are reached, whose run's earlier command buffers have run, and whose queue's earlier ones
   are all committed, committed in order; an older play's entries dropped; dispatcher.lock held (the one committer). */
static int dispatch_pump_locked(void){
  int committed=0;
  for(struct entry **at=&dispatcher.head,*e;(e=*at);){
    const int stale=e->run && e->generation!=atomic_load(&e->run->generation);
    int go=!stale;
    for(struct entry *f=dispatcher.head;f!=e && go;f=f->next)go=f->queue!=e->queue;
    if(go && e->run)go=e->index==atomic_load(&e->run->done);
    for(int g=0;g<e->n && go;g++)go=reached(e->gates[g]);
    if(!go && !stale){at=&e->next;continue;}
    *at=e->next;
    if(dispatcher.tail==e){dispatcher.tail=NULL;for(struct entry *f=dispatcher.head;f;f=f->next)dispatcher.tail=f;}
    if(stale){nccl_mesh_program_drop(e->program);free(e->ticket);}
    else{
      *e->ticket=(struct ticket){e->run,e->generation,e->index,e->last};atomic_fetch_add(&e->run->refs,1);
      nccl_mesh_program_commit(e->program,entry_ran,e->ticket);
      committed++;
    }
    for(int g=0;g<e->n;g++)point_drop(e->gates+g);
    if(e->run)run_release(e->run);
    free(e->gates);free(e);
  }
  return committed;
}
static void dispatch_pump(void){
  if(!atomic_load_explicit((_Atomic(struct entry *) *)&dispatcher.head,memory_order_relaxed) || pthread_mutex_trylock(&dispatcher.lock))return;
  dispatch_pump_locked();
  pthread_mutex_unlock(&dispatcher.lock);
}
static void run_play(struct run *r);
static void *dispatch_run(void *argument){
  (void)argument;
  pthread_setname_np("nccl-mesh.dispatch");
  pthread_mutex_lock(&dispatcher.lock);
  for(;;){
    while(!dispatcher.head && !dispatcher.redo)pthread_cond_wait(&dispatcher.cond,&dispatcher.lock);
    while(dispatcher.redo){
      struct run *r=dispatcher.redo;dispatcher.redo=r->redo;
      pthread_mutex_unlock(&dispatcher.lock);
      run_play(r);run_release(r);
      pthread_mutex_lock(&dispatcher.lock);
    }
    dispatch_pump_locked();
    if(!dispatcher.head)continue;
    pthread_mutex_unlock(&dispatcher.lock);
    usleep(20);
    pthread_mutex_lock(&dispatcher.lock);
  }
  return NULL;
}
static void dispatch_submit(struct entry *e){
  pthread_mutex_lock(&dispatcher.lock);
  if(dispatcher.tail)dispatcher.tail->next=e;else dispatcher.head=e;
  dispatcher.tail=e;
  dispatch_pump_locked();
  if(dispatcher.head)dispatch_wake_locked();
  pthread_mutex_unlock(&dispatcher.lock);
}
static struct entry *entry_new(void *queue){
  struct entry *e=calloc(1,sizeof *e);
  if(e && (!(e->ticket=malloc(sizeof *e->ticket)) || !(e->program=nccl_mesh_program_begin(queue)))){free(e->ticket);free(e);e=NULL;}
  if(e)e->queue=queue;
  return e;
}
static void entry_gate(struct entry *e,void *event,uint64_t value){
  if(!event || nccl_mesh_event_value(event)>=value)return;
  if(e->n==e->capacity){
    int capacity=e->capacity?2*e->capacity:4;
    struct point *grown=realloc(e->gates,(size_t)capacity*sizeof *grown);
    if(!grown){
      /* a gate that cannot be kept is waited for here, so the command buffer never runs before it */
      while(nccl_mesh_event_value(event)<value)sched_yield();
      return;
    }
    e->gates=grown;e->capacity=capacity;
  }
  e->gates[e->n]=(struct point){0};
  point_set(e->gates+e->n++,event,value);
}
/* The library's own program on `queue`, in command buffers each committed by the dispatcher: `current` the one
   being encoded, the `index`th of its run.  A gate (or a recorded point to wait for) submits it and begins the next,
   gated. */
struct owned { void *queue; struct entry *current; struct run *run; uint32_t index; };
static struct entry *owned_entry(struct owned *o){
  struct entry *e=entry_new(o->queue);
  if(e){e->run=o->run;e->generation=atomic_load(&o->run->generation);e->index=o->index++;atomic_fetch_add(&o->run->refs,1);}
  return e;
}
static void *owned_gate(void *argument,void *program,void *event,uint64_t value){
  struct owned *o=argument;
  nccl_mesh_program_end(program);
  dispatch_submit(o->current);
  o->current=owned_entry(o);
  if(!o->current)return NULL;
  entry_gate(o->current,event,value);
  return o->current->program;
}
/* A run of `queue` for a program (`ran`, its recording `rec`; both NULL: the commands of streams' kept programs, which
   carry their own), held by its maker until run_release. */
static struct run *run_new(void *queue,struct recording *rec,struct ran *ran,int redoable){
  struct run *r=calloc(1,sizeof *r);
  if(r){atomic_store(&r->refs,1);r->queue=queue;r->rec=rec;r->ran=ran;r->redoable=redoable && rec;}
  return r;
}
/* The program's last command buffer submitted; its run ends once it has run. */
static void owned_finish(struct owned *o){
  if(!o->current)return;
  nccl_mesh_program_end(o->current->program);
  o->current->last=1;
  dispatch_submit(o->current);
  o->current=NULL;
}
static void *play(struct recording *k,void *program,gate_fn gate,void *argument);
/* A run played again from its start (the dispatcher's thread): its recording into command buffers of a new
   generation; where it cannot be encoded, it fails for good. */
static void run_play(struct run *r){
  pthread_mutex_lock(&stream_lock);
  atomic_fetch_add(&r->generation,1);atomic_store(&r->done,0);
  struct owned o={r->queue,NULL,r,0};
  o.current=owned_entry(&o);
  void *program=o.current?o.current->program:NULL;
  if(program)program=play(r->rec,program,owned_gate,&o);
  if(program && !r->rec->failed)owned_finish(&o);
  else{
    if(o.current){nccl_mesh_program_drop(o.current->program);run_release(o.current->run);free(o.current->gates);free(o.current->ticket);free(o.current);}
    atomic_fetch_add(&r->generation,1);r->failed=1;run_finish(r);
  }
  pthread_mutex_unlock(&stream_lock);
}
/* A caller's that names no gate of its own: the wait inside its command buffer (the caller's to keep short). */
static void *inline_gate(void *argument,void *program,void *event,uint64_t value){
  (void)argument;
  if(event)nccl_mesh_program_wait(program,event,value);
  return program;
}
/* The kept commands encoded into `program`, each gate by `gate` (a persistent call's with no event and its place
   among the recording's gates, given as it is played; for the library's own programs a recorded point to wait for
   too); the command buffer the last of them went into. */
static void *play(struct recording *k,void *program,gate_fn gate,void *argument){
  for(int i=0;i<k->n && program;i++){
    const struct recorded *r=k->ops+i;
    if(r->kind==R_GATE){
      struct gate *g=r->gate;
      nccl_mesh_program_end(program);
      if(g){if(g->index<0)g->index=g->run->gates++;program=gate(argument,program,NULL,(uint64_t)g->index);}
      else program=gate(argument,program,r->a,r->x);
      continue;
    }
    if(r->kind==R_WAIT){
      if(gate==owned_gate)program=gate(argument,program,r->a,r->x);
      else nccl_mesh_program_wait(program,r->a,r->x);
    }
    else if(r->kind==R_SIGNAL)nccl_mesh_program_signal(program,r->a,r->x);
    else if(r->kind==R_KERNEL)nccl_mesh_program_kernel(program,r->which,r->a,r->x,r->b,r->y,r->n,r->type,r->op,r->nranks,r->scalar,r->published,r->c,r->z,
                                                      r->q,r->pred?r->pred-1:0,r->want,r->grid);
    else if(r->kind==R_COPY)nccl_mesh_program_copy(program,r->a,r->x,r->b,r->y,r->n,r->which,r->published,NULL,0,0,0);
    else if(r->kind==R_SPIN){
      struct recorded *op=k->ops+i;
      if(op->held){nccl_mesh_release(op->held);op->held=NULL;}
      if(nccl_mesh_program_spin(program,r->a,r->list,(uint32_t)r->n,r->x,r->y,r->b,r->touch,4,&op->held))k->failed=1;
    }
    else if(r->kind==R_COUNTED)nccl_mesh_program_counted(program,r->a,r->b,r->c,r->d,r->e,r->list,(uint32_t)r->x,r->n,r->q,r->grid);
    else if(r->kind==R_ZERO)nccl_mesh_program_zero(program,r->a,r->x,r->n);
    else nccl_mesh_program_publish(program,r->a,r->x,r->y,r->q,r->pred?r->pred-1:0,r->want);
  }
  if(program)nccl_mesh_program_end(program);
  return program;
}
static void recording_free(struct recording *k){
  for(int i=0;k && i<k->n;i++){
    void *const held[6]={k->ops[i].a,k->ops[i].b,k->ops[i].c,k->ops[i].d,k->ops[i].e,k->ops[i].q};
    for(int h=0;h<6;h++)if(held[h])nccl_mesh_release(held[h]);
    for(int t=0;t<4;t++)if(k->ops[i].touch[t])nccl_mesh_release(k->ops[i].touch[t]);
    if(k->ops[i].held)nccl_mesh_release(k->ops[i].held);
    free(k->ops[i].list);
  }
  if(k)free(k->ops);
  free(k);
}
static void stream_release(struct ncclMeshStream *s);
/* A program that failed for good (nccl-mesh-metal.m nccl_mesh_program_handler: its own fault, or not to be run
   again): its communicators' calls fail and they are revoked until their next agreement, its stream fails, and the
   values it signals are given by the host, so the waits on them end (on calls that end failed). */
static void program_failed(struct ran *r){
  for(int i=0;i<r->ncomms;i++){atomic_store(&r->comms[i]->gpu_failed,1);comm_revoke(r->comms[i]);}
  if(r->stream)__atomic_store_n(&r->stream->failed,1,__ATOMIC_RELEASE);
  for(int i=0;r->kept && i<r->kept->n;i++)if(r->kept->ops[i].kind==R_SIGNAL)nccl_mesh_event_signal(r->kept->ops[i].a,r->kept->ops[i].x);
}
/* Once the GPU has run a program (status: nccl_mesh_program_handler's, a failure for good where not 0): its objects
   released. */
static void program_ran(void *argument,int status){
  struct ran *r=argument;
  if(status)program_failed(r);
  for(int i=0;i<r->nevents;i++)nccl_mesh_release(r->events[i]);
  for(int i=0;i<r->ncomms;i++)atomic_fetch_sub(&r->comms[i]->programs,1);
  if(r->stream)stream_release(r->stream);
  recording_free(r->kept);
  free(r);
}
/* The stream's kept programs taken off it (oldest first) and encoded into `program` (gates by `gate`), each
   released once the command buffer its last command went into has run; that command buffer. stream_lock held. */
static void *take_kept(struct ncclMeshStream *s,void *program,gate_fn gate,void *argument){
  struct recording *k=s->pending;
  s->pending=NULL;
  while(k){
    struct recording *next=k->next;
    program=play(k,program,gate,argument);
    k->ran->kept=k;k->next=NULL;
    if(program)nccl_mesh_program_handler(program,program_ran,k->ran);
    k=next;
  }
  return program;
}
/* Whether a call writes what it reads (its send buffer's bytes and its result's overlap): its input is not
   retained, so its program cannot run again from its start. */
static int overwrites_input(const struct call *k){
  const unsigned char *in=k->send,*out=k->recv;const size_t ni=in_bytes(k),no=out_bytes(k);
  return ni && no && in && out && in<out+no && out<in+ni;
}
/* Whether a group's program has GPU work before its transfers start (so it needs a gate): a receive
   from this rank itself, a send's bytes staged into the group's allocation, a contribution copied or
   premultiplied into its operand. */
static int gated(const struct call *calls,int n){
  for(int i=0;i<n;i++){
    const struct call *k=calls+i;
    if(self_copy(calls,n,i))return 1;
    if(k->kind==K_SEND && k->peer!=k->comm->rank && k->wire!=(const unsigned char *)k->send)return 1;
    if(k->kind>=K_SEND)continue;
    if(in_bytes(k) && (k->premultiply || (input_at(k)!=(const unsigned char *)k->send && !k->fresh)))return 1;
  }
  return 0;
}
/* Whether a call's worker waits for a combine (a SEND after a REDUCE: a ring's or tree's partial sent
   on), which the library's own program must then run as the pieces arrive. */
static int sends_combined(const struct call *calls,int n){
  for(int i=0;i<n;i++)if(combines(calls+i))return 1;
  return 0;
}
/* A word's wait: its (offset, value) pair appended to `list` (grown). */
static uint64_t *pair(uint64_t *list,uint32_t *n,size_t *capacity,struct region r,const void *word,uint64_t value){
  if(2*(size_t)(*n+1)>*capacity){
    /* a word is never left out of a spin's list: its growth waited for */
    size_t grown=*capacity?2**capacity:64;
    uint64_t *more;
    while(!(more=realloc(list,grown*sizeof *more)))sched_yield();
    list=more;*capacity=grown;
  }
  list[2**n]=off(r,word);list[2**n+1]=value;++*n;
  return list;
}
/* A gate of the group, `words` its words (taken), on communicator c: its value the next of c's (a persistent
   call's given per iteration), landing word `landing`. */
static struct gate *gate_add(struct launch *l,struct ncclComm *c,_Atomic uint64_t **words,uint32_t nwords,_Atomic uint64_t *landing,
  struct persistent *run){
  struct gate *g=malloc(sizeof *g);
  if(!g){free(words);return NULL;}
  *g=(struct gate){.comm=c,.words=words,.nwords=nwords,.landing=landing,.event=run?NULL:c->landed,.value=run?0:++c->landed_next,.run=run,.index=-1};
  l->gates[l->ngates++]=g;
  return g;
}

/* The group's program: the other streams' prior work and the recorded points waited for; the operands copied in
   (a receive from this rank itself, a send's bytes into the group's allocation, a contribution into its operand,
   or premultiplied there) and, after them, the gate word the workers start at (a kept program, `kept`, has none:
   the workers wait for the recorded points themselves).  Then the network's part, which no command buffer waits
   on for long: each REDUCE chunk's leaky spin on its receive's completion word and those of the earlier sends
   reading its range, and its combine predicated on the spin having seen them (and, where a later send of the call
   reads the combine, its published word); a spin on every word of each communicator's calls and the group's
   landing word; the group's gates; then every combine again predicated on its spin having given up, the
   post-division and copies out, the received bytes of point-to-point calls copied out, a counted all-to-all's own
   rows (their first try after its counts' spin); a persistent call's control words zeroed, so its next iteration's
   spins see only that iteration's (ncclMeshPersistentStart); the end and each stream's completion value.  A
   call whose combine a later send reads is gated chunk by chunk instead: each chunk's gate after its first try (the
   send waits for the combine, and the call's gate for the send).  Values: the program's event's, under
   stream_lock. */
static void encode(struct launch *l,struct call *calls,int n,struct ncclComm **comms,int ncomms,struct recording *rec,struct ran *ran,int kept,
  struct persistent *run){
  struct ncclMeshStream *primary=l->nmarks?l->marks[0].stream:NULL;
  void *event=primary?primary->event:comms[0]->event;
  uint64_t *value=primary?&primary->value:&comms[0]->event_value;
  uint64_t *list=NULL;size_t capacity=0;uint32_t m=0;
  l->event=event;
  _Atomic uint64_t *gate=l->gate;
  l->gate=NULL;
  if(!kept){
    int before=0;
    for(int k=1;k<l->nmarks;k++){struct ncclMeshStream *s=l->marks[k].stream;if(s->value){program_wait(rec,calls,s->event,s->value);before=1;}}
    for(int w=0;w<l->nwaits;w++,before=1)program_wait(rec,calls,l->waits[w].event,l->waits[w].value);
    for(int i=0;i<n;i++){
      struct call *k=calls+i;
      const struct call *mine=self_copy(calls,n,i);
      const size_t bytes=k->count*type_bytes[k->type];
      if(mine){copy_into(rec,k,k->out,k->recv,mine->in,mine->send,bytes,0,0);k->copied=1;before=1;continue;}
      if(k->kind==K_SEND && k->peer!=k->comm->rank && k->wire!=(const unsigned char *)k->send){copy_into(rec,k,l->own,k->wire,k->in,k->send,bytes,0,1);before=1;}
      if(k->kind>=K_SEND)continue;
      const size_t in=in_bytes(k);unsigned char *to=input_at(k);
      if(!in)continue;
      if(k->premultiply){kernel(rec,k,KERNEL_PREMULTIPLY,k->at,to,k->in,k->send,in/type_bytes[k->type],0,1,NULL,l->own,NULL,0,0);before=1;}
      else if(to!=(const unsigned char *)k->send && !k->fresh){copy_into(rec,k,k->at,to,k->in,k->send,in,0,1);before=1;}
    }
    if(before && gate){publish(rec,l->own,gate,1,l->own,NULL,0);l->gate=gate;}
  }
  /* a call reading its send buffer in place: the ranges of its result no step writes, copied from it
     (plain stores: no SEND reads them, a SEND of such a range reads the send buffer) */
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(!k->fresh)continue;
    const size_t e=type_bytes[k->type],lo=(size_t)(output_at(k)-k->operand),hi=lo+out_bytes(k),own=own_at(k),in=in_bytes(k);
    for(size_t at=lo;at<hi;){
      size_t end=hi;int covered=0;
      for(uint32_t q=0;q<k->nsteps && !covered;q++){
        const struct mesh_step *t=k->steps+q;
        if(t->op==MESH_STEP_SEND || !t->piece.elements)continue;
        const size_t a=t->first*e,b=a+t->piece.elements*e;
        if(a<=at && at<b){covered=1;end=b;}
        else if(a>at && a<end)end=a;
      }
      if(end>hi)end=hi;
      if(!covered && at>=own && end<=own+in)copy_into(rec,k,k->at,k->operand+at,k->in,given(k,at),end-at,0,0);
      at=end;
    }
  }
  /* the network's part, first try: each REDUCE chunk's spin and combine */
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind>=K_SEND)continue;
    const size_t e=type_bytes[k->type];
    const int chunked=k->combined!=NULL;
    uint32_t w=0,r=0;
    for(uint32_t s=0;s<k->nsteps && k->nwords;s++){
      const struct mesh_step *step=k->steps+s;
      if(!step->piece.elements)continue;
      const uint32_t chunks=chunks_of(step->piece.elements,e);
      for(uint32_t j=0;step->op==MESH_STEP_REDUCE && j<chunks;j++){
        const uint64_t first=chunk_first(step->piece.elements,e,j),elements=chunk_count(step->piece.elements,e,j);
        const size_t lo=(step->first+first)*e,hi=lo+elements*e;
        _Atomic uint64_t **needs=calloc(k->nwords,sizeof *needs);uint32_t nneeds=0;
        if(!needs){rec->failed=1;continue;}
        m=0;list=pair(list,&m,&capacity,l->own,k->words+w+j,1);needs[nneeds++]=k->words+w+j;
        /* the earlier steps' sends that read the range the combine writes */
        for(uint32_t q=0,v=0;q<s;q++){
          const struct mesh_step *earlier=k->steps+q;
          if(!earlier->piece.elements)continue;
          const uint32_t sent=chunks_of(earlier->piece.elements,e);
          for(uint32_t t=0;earlier->op==MESH_STEP_SEND && t<sent;t++){
            const size_t a=(earlier->first+chunk_first(earlier->piece.elements,e,t))*e,b=a+chunk_count(earlier->piece.elements,e,t)*e;
            if(a<hi && lo<b){list=pair(list,&m,&capacity,l->own,k->words+v+t,1);needs[nneeds++]=k->words+v+t;}
          }
          v+=sent;
        }
        struct gate *g=NULL;
        if(chunked && !(g=gate_add(l,k->comm,needs,nneeds,k->landings+w+j,run))){rec->failed=1;continue;}
        if(chunked)list=pair(list,&m,&capacity,l->own,k->landings+w+j,1);
        else free(needs);
        const _Atomic uint64_t *flag=k->flags+w+j;
        const void *given_first=k->fresh && !written(k,s,lo,hi)?given(k,lo):NULL;
        spin(rec,k,l->own,list,m,flag,k->flags+k->nwords);
        kernel(rec,k,KERNEL_COMBINE,k->at,k->operand+lo,l->own,k->pieces[s].at+first*e,elements,k->combine,chunked,given_first,l->own,flag,1,0);
        if(chunked){
          publish(rec,l->own,k->combined+r+j,1,l->own,flag,1);
          program_gate(rec,g);
          kernel(rec,k,KERNEL_COMBINE,k->at,k->operand+lo,l->own,k->pieces[s].at+first*e,elements,k->combine,1,given_first,l->own,flag,0,LATE_GRID);
          publish(rec,l->own,k->combined+r+j,1,l->own,flag,0);
        }
      }
      if(step->op==MESH_STEP_REDUCE && chunked)r+=chunks;
      w+=chunks;
    }
  }
  /* a counted all-to-all's own rows, first try: once the other ranks' count segments have landed */
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind!=K_COUNTED || !k->nwords)continue;
    const uint32_t peers=(uint32_t)k->comm->nranks-1;
    m=0;
    for(uint32_t x=peers;x<2*peers;x++)list=pair(list,&m,&capacity,l->own,k->words+x,1);
    spin(rec,k,l->own,list,m,k->flags+peers,k->flags+k->nwords);
    if(k->rows || k->landed)counted_copy(rec,k,l,k->flags+peers,1,0);
  }
  /* each communicator's gate: every word of its calls (the last spin also on its landing word) */
  for(int j=0;j<ncomms;j++){
    uint32_t total=0;struct call *any=NULL;
    for(int i=0;i<n;i++)if(calls[i].comm==comms[j]){total+=calls[i].nwords;if(!any)any=calls+i;}
    if(!total)continue;
    _Atomic uint64_t **needs=calloc(total,sizeof *needs);uint32_t nneeds=0;
    if(!needs){rec->failed=1;continue;}
    m=0;
    for(int i=0;i<n;i++)if(calls[i].comm==comms[j])
      for(uint32_t x=0;x<calls[i].nwords;x++){list=pair(list,&m,&capacity,l->own,calls[i].words+x,1);needs[nneeds++]=calls[i].words+x;}
    struct gate *g=gate_add(l,comms[j],needs,nneeds,l->landing+2*j,run);
    if(!g){rec->failed=1;continue;}
    list=pair(list,&m,&capacity,l->own,g->landing,1);
    spin(rec,any,l->own,list,m,l->landing+2*j+1,any->flags+any->nwords);
  }
  for(int g=0;g<l->ngates;g++)if(l->gates[g]->landing>=l->landing && l->gates[g]->landing<l->landing+2*ncomms)program_gate(rec,l->gates[g]);
  /* after the gates: what a spin gave up on, then what needs every word */
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind>=K_SEND)continue;
    const size_t e=type_bytes[k->type];
    uint32_t w=0;
    for(uint32_t s=0;s<k->nsteps && k->nwords && !k->combined;s++){
      const struct mesh_step *step=k->steps+s;
      if(!step->piece.elements)continue;
      const uint32_t chunks=chunks_of(step->piece.elements,e);
      for(uint32_t j=0;step->op==MESH_STEP_REDUCE && j<chunks;j++){
        const uint64_t first=chunk_first(step->piece.elements,e,j),elements=chunk_count(step->piece.elements,e,j);
        const size_t lo=(step->first+first)*e,hi=lo+elements*e;
        kernel(rec,k,KERNEL_COMBINE,k->at,k->operand+lo,l->own,k->pieces[s].at+first*e,elements,k->combine,0,
               k->fresh && !written(k,s,lo,hi)?given(k,lo):NULL,l->own,k->flags+w+j,0,LATE_GRID);
      }
      w+=chunks;
    }
    const size_t out=out_bytes(k);unsigned char *from=output_at(k);
    const int divide=out && k->postdivide,copy=out && from!=(unsigned char *)k->recv;
    if(divide)kernel(rec,k,KERNEL_POSTDIVIDE,k->at,from,k->at,from,out/e,0,0,NULL,l->own,NULL,0,0);
    if(copy)copy_into(rec,k,k->out,k->recv,k->at,from,out,1,0);
  }
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind==K_RECV && k->peer!=k->comm->rank && k->wire!=(unsigned char *)k->recv)
      copy_into(rec,k,k->out,k->recv,l->own,k->wire,k->count*type_bytes[k->type],1,0);
    if(k->kind==K_COUNTED && k->nwords && (k->rows || k->landed))counted_copy(rec,k,l,k->flags+(uint32_t)k->comm->nranks-1,0,LATE_GRID);
  }
  if(l->persistent && l->zero_words)add(rec,(struct recorded){.kind=R_ZERO,.a=l->own.buffer,.x=l->zero_at,.n=l->zero_words});
  free(list);
  for(int j=0;j<ncomms;j++){ran->comms[ran->ncomms++]=comms[j];atomic_fetch_add(&comms[j]->programs,1);}
  l->end=++*value;program_signal(rec,event,l->end);
  for(int k=0;k<l->nmarks;k++){
    struct mark *mark=l->marks+k;struct ncclMeshStream *s=mark->stream;
    if(s==primary){mark->done=l->end;continue;}
    mark->done=++s->value;program_signal(rec,s->event,mark->done);
    nccl_mesh_retain(s->event);ran->events[ran->nevents++]=s->event;
  }
  rec->ran=ran;
}

ncclResult_t ncclGroupEnd(void){
  if(group.depth<=0)return FAIL(NULL,ncclInvalidUsage,"ncclGroupEnd without ncclGroupStart");
  if(--group.depth)return ncclSuccess;
  struct call *calls=group.calls;int n=group.n;ncclResult_t status=group.error;
  group.calls=NULL;group.n=group.capacity=0;group.error=ncclSuccess;
  for(int i=0;i<n && !status;i++)
    if((status=resolve(calls+i)) && status!=ncclInvalidArgument && status!=ncclInvalidUsage){
      /* the communicator failed (a link down, a connection lost, or revoked earlier): revoked */
      struct ncclComm *c=calls[i].comm;
      failed_at(c,calls[i].index,c->error,calls[i].agreement);
      comm_revoke(c);
    }
  plans.n=0;
  tally_release(plans.tally);plans.tally=NULL;
  for(int i=0;i<n && !status && plans.n<4096;i++,plans.n++){
    int local=calls[i].kind>=K_SEND || calls[i].comm->nranks==1;
    plans.how[plans.n]=local?-1:(int)calls[i].chosen.how;plans.root[plans.n]=local?0:(int)calls[i].chosen.root;
    plans.epoch[plans.n]=calls[i].kind>=K_SEND?0:calls[i].epoch;
  }
  if(status || !n){calls_free(calls,n);return status;}
  struct launch *l=calloc(1,sizeof *l);
  struct tally *tally=calloc(1,sizeof *tally+(size_t)n*sizeof tally->counts[0]);
  struct ncclComm **comms=calloc((size_t)n,sizeof *comms);int ncomms=0;
  struct ran *ran=calloc(1,sizeof *ran+(size_t)n*(sizeof(struct ncclComm *)+sizeof(void *)));
  if(l){l->marks=calloc((size_t)n,sizeof *l->marks);l->waits=calloc((size_t)n*(2*(READS+1)+1),sizeof *l->waits);}
  if(!l || !l->marks || !l->waits || !tally || !comms || !ran){
    if(l){free(l->marks);free(l->waits);}
    free(l);free(tally);free(comms);free(ran);calls_free(calls,n);
    return FAIL(NULL,ncclSystemError,"allocation");
  }
  ran->comms=(struct ncclComm **)(ran+1);ran->events=(void **)(ran->comms+n);
  pthread_mutex_init(&l->lock,NULL);pthread_cond_init(&l->cond,NULL);
  tally->refs=2;tally->n=n;l->tally=plans.tally=tally;
  for(int i=0;i<n;i++){
    calls[i].tally=tally->counts[i];
    int seen=0;
    for(int j=0;j<ncomms;j++)seen|=comms[j]==calls[i].comm;
    if(!seen)comms[ncomms++]=calls[i].comm;
    if(!calls[i].stream)l->sync=1;
    else{
      seen=0;
      for(int m=0;m<l->nmarks;m++)seen|=l->marks[m].stream==calls[i].stream;
      if(!seen)l->marks[l->nmarks++].stream=calls[i].stream;
    }
  }
  pthread_mutex_lock(&stream_lock);
  status=regions(calls,n,l);
  /* a persistent call (ncclMeshPersistentBegin): kept and never started here, on the persistent calls' one
     communicator; a group of persistent point-to-point calls (`held`) has no GPU work of the library's: its
     buffers are window allocations, sent and received in place, and its caller's GPU work publishes each send's
     bytes (its ready word) and waits on each receive's landed word itself */
  pthread_mutex_lock(&persisting.lock);
  struct persistent *persist=persisting.building;
  pthread_mutex_unlock(&persisting.lock);
  const int held=persist && P2P(calls);
  for(int i=0;held && i<n && !status;i++)
    if(!P2P(calls+i) || calls[i].stream || calls[i].peer==calls[i].comm->rank || (calls[i].kind==K_SEND?!calls[i].in.span:!calls[i].out.span))
      status=FAIL(calls[i].comm,ncclInvalidUsage,"a persistent point-to-point call takes no stream, has a peer other than its rank, and "
                  "sends or receives a window allocation (ncclMemAlloc) in place, as every call of its group does");
    else if(calls[i].kind==K_SEND?!calls[i].ready:!calls[i].word)
      status=FAIL(calls[i].comm,ncclInvalidUsage,"a persistent %s names its word before it is issued (%s)",calls[i].kind==K_SEND?"ncclSend":"ncclRecv",
                  calls[i].kind==K_SEND?"ncclMeshPersistentNext: its bytes' ready word":"ncclMeshPersistentLanded");
  /* completion words where a program may wait on them: a stream's call, or any isend or irecv */
  int words=l->nmarks>0;
  for(int i=0;i<n && !words;i++)words=requests(calls+i)>0;
  if(!status)status=place(calls,n,l,words && !held,ncomms);
  /* a program where a stream's completion is its, or the GPU has work; kept where its stream defers and
     it has no GPU work before its transfers (ncclMeshStreamDefer) */
  int work=l->nmarks>0;
  for(int i=0;i<n && !work && !status;i++)work=gpu_work(calls,n,i);
  const int defer=!status && l->nmarks==1 && l->marks[0].stream->deferred && ncomms==1 && !gated(calls,n) && !sends_combined(calls,n);
  /* a persistent collective must defer, and be a collective of the persistent calls' one communicator */
  for(int i=0;persist && !held && i<n && !status;i++)
    if(P2P(calls+i) || calls[i].kind==K_COUNTED || (persist->comm && persist->comm!=calls[i].comm))
      status=FAIL(calls[i].comm,ncclInvalidUsage,"a persistent call is a collective on the persistent calls' one communicator");
  for(int i=0;held && i<n && !status;i++)
    if(persist->comm && persist->comm!=calls[i].comm)
      status=FAIL(calls[i].comm,ncclInvalidUsage,"a persistent call is a call on the persistent calls' one communicator");
  if(!status && persist && !held && !defer)
    status=FAIL(calls[0].comm,ncclInvalidUsage,"a persistent call defers (a deferred stream, no GPU work before its transfers: nccl.h ncclMeshStreamDefer)");
  if(!status && persist){l->persistent=1;persist->comm=calls[0].comm;}
  /* a persistent point-to-point call's one request each iteration: a send's ready word one range of its bytes, a
     receive's end stored into its landed word (its one completion word); no cut */
  for(int i=0;held && i<n && !status;i++){
    struct call *k=calls+i;
    if(k->kind==K_SEND){
      k->range=k->count*type_bytes[k->type];k->ranges=1;
      if(!(k->ready_base=calloc(1,sizeof *k->ready_base)))status=FAIL(NULL,ncclSystemError,"allocation");
    }
    else{k->words=(_Atomic uint64_t *)k->word;k->nwords=1;}
    k->ahead=1;
    if(!status && !(k->handles=calloc(1,sizeof *k->handles)))status=FAIL(NULL,ncclSystemError,"allocation");
    /* in slots: each slot's buffer in the same window allocation as the one issued */
    if(!status && k->slot_depth){
      k->wire0=k->wire;k->word0=k->word;k->ready0=k->ready;
      pthread_mutex_lock(&heap.lock);
      const int fits=span_of(k->wire,(k->slot_depth-1)*k->slot_stride+k->count*type_bytes[k->type])!=NULL;
      pthread_mutex_unlock(&heap.lock);
      if(!fits)
        status=FAIL(k->comm,ncclInvalidArgument,"a persistent call's %llu slots of %llu bytes reach past its window allocation",
                    (unsigned long long)k->slot_depth,(unsigned long long)k->slot_stride);
      else if(!(k->tallies=calloc(k->slot_depth*TALLIES,sizeof *k->tallies)))status=FAIL(NULL,ncclSystemError,"allocation");
    }
  }
  /* a persistent collective's receives posted ahead where they may be, and its send buffer published in ranges
     where the caller said so (ncclMeshPersistentNext): then it needs no cut */
  int fresh=0,cut=!held;
  if(!status && persist && !held){
    pthread_mutex_lock(&persisting.lock);
    const __typeof__(persisting.next) next=persisting.next;
    persisting.next.given=0;
    pthread_mutex_unlock(&persisting.lock);
    fresh=next.given?next.fresh:0;
    for(int i=0;i<n;i++)calls[i].ahead=ahead_of(calls+i,fresh,l);
    if(next.given && next.ready && n==1 && next.range && in_bytes(calls) && in_bytes(calls)==next.bytes){
      calls[0].ready=(const _Atomic uint64_t *)next.ready;calls[0].range=next.range;
      calls[0].ranges=(uint32_t)((next.bytes+next.range-1)/next.range);
      calls[0].ready_base=calloc(calls[0].ranges,sizeof *calls[0].ready_base);
      if(!calls[0].ready_base)status=FAIL(NULL,ncclSystemError,"allocation");
      cut=!calls[0].ahead;
    }
    for(int i=0;i<n && !status;i++)if(!(calls[i].handles=calloc(calls[i].nwords?calls[i].nwords:1,sizeof *calls[i].handles)))
      status=FAIL(NULL,ncclSystemError,"allocation");
  }
  struct recording *rec=NULL;
  uint32_t most=(uint32_t)ncomms;
  for(int i=0;i<n;i++)most+=calls[i].nwords;
  if(!status && (work || defer) && (!(rec=calloc(1,sizeof *rec)) || !(l->gates=calloc(most,sizeof *l->gates))))
    status=FAIL(NULL,ncclSystemError,"allocation");
  if(status){
    pthread_mutex_unlock(&stream_lock);
    regions_release(calls,n,l);
    free(rec);launch_free(l);free(comms);free(ran);calls_free(calls,n);
    return status;
  }
  if(rec){
    encode(l,calls,n,comms,ncomms,rec,ran,defer,persist);
    record(l,calls,n);
    if(l->nmarks){ran->stream=l->marks[0].stream;__atomic_add_fetch(&ran->stream->refs,1,__ATOMIC_RELAXED);}
    /* a gate opens once its words are set (a persistent call's each iteration, ncclMeshPersistentStart) */
    for(int g=0;!persist && g<l->ngates;g++)gate_arm(l->gates[g]);
    if(defer){
      struct recording **at=(struct recording **)&l->marks[0].stream->pending;
      while(*at)at=&(*at)->next;
      *at=rec;
    }
    else{
      /* run again from its start after a failure that is not its own where its inputs are retained (struct run) */
      void *queue=l->nmarks?l->marks[0].stream->queue:comms[0]->queue;
      int redoable=!sends_combined(calls,n);
      for(int i=0;i<n;i++)redoable&=!overwrites_input(calls+i);
      for(int k=0;k<l->nmarks;k++)redoable&=!l->marks[k].stream->pending;
      ran->kept=rec;
      struct run *r=run_new(queue,rec,ran,redoable);
      struct owned o={queue,NULL,r,0};
      o.current=r?owned_entry(&o):NULL;
      void *program=o.current?o.current->program:NULL;
      for(int k=0;program && k<l->nmarks;k++)if(l->marks[k].stream->pending)program=take_kept(l->marks[k].stream,program,owned_gate,&o);
      if(program)program=play(rec,program,owned_gate,&o);
      for(int k=0;k<l->nmarks;k++)l->marks[k].stream->committed=l->marks[k].stream->value;
      count(calls,COMMITS,1);
      if(program && !rec->failed)owned_finish(&o);
      else{
        /* not encoded (an allocation): it fails for good, its calls with it */
        FAIL(NULL,ncclUnhandledCudaError,"a command buffer on the queue");
        if(o.current){nccl_mesh_program_drop(o.current->program);run_release(o.current->run);free(o.current->gates);free(o.current->ticket);free(o.current);}
        if(r){atomic_fetch_add(&r->generation,1);r->failed=1;run_finish(r);}
        else program_ran(ran,2);
      }
      if(r)run_release(r);
      for(int w=0;w<l->nwaits;w++)point_drop(l->waits+w);
      l->nwaits=0;
    }
  }
  else{free(ran);l->gate=NULL;}
  regions_release(calls,n,l);
  l->items=ncomms;
  for(int j=0;j<ncomms;j++){
    struct item *it=calloc(1,sizeof *it);
    it->calls=calloc((size_t)n,sizeof *it->calls);it->launch=l;
    for(int i=0;i<n;i++)if(calls[i].comm==comms[j]){
      it->calls[it->n++]=calls[i];calls[i].steps=NULL;calls[i].pieces=NULL;calls[i].segments=NULL;calls[i].handles=NULL;calls[i].ready_base=NULL;
    }
    struct ncclComm *c=comms[j];
    if(persist){
      /* kept for ncclMeshPersistentStart, not started */
      pthread_mutex_lock(&persisting.lock);
      if(persist->n==persist->capacity){
        int capacity=persist->capacity?2*persist->capacity:32;
        struct item **grown=realloc(persist->items,(size_t)capacity*sizeof *grown);
        if(grown){persist->items=grown;persist->capacity=capacity;}
      }
      if(persist->n<persist->capacity){it->run=persist;it->cut=cut;it->fresh=fresh;persist->items[persist->n++]=it;}
      persisting.cut=cut;persisting.groups++;
      pthread_mutex_unlock(&persisting.lock);
      continue;
    }
    pthread_mutex_lock(&c->lock);
    if(c->tail)c->tail->next=it;else c->head=it;
    c->tail=it;
    if(c->parked)pthread_cond_broadcast(&c->cond);
    pthread_mutex_unlock(&c->lock);
  }
  pthread_mutex_unlock(&stream_lock);
  free(calls);
  /* a persistent call is kept for its starts, with no stream or a deferred one */
  if(persist || !l->sync){free(comms);return ncclSuccess;}
  pthread_mutex_lock(&l->lock);
  while(l->items)pthread_cond_wait(&l->cond,&l->lock);
  status=l->result;
  pthread_mutex_unlock(&l->lock);
  if(l->event){
    /* the NULL stream: the program's end, however late (one run again included; one that failed for good has its end
       given by the host, and its communicators marked) */
    count(&(struct call){.tally=tally->counts[n-1]},HOST_WAITS,1);
    while(!status && nccl_mesh_event_value(l->event)<l->end)sched_yield();
    for(int j=0;!status && j<ncomms;j++)if(atomic_load(&comms[j]->gpu_failed))status=FAIL(comms[j],ncclUnhandledCudaError,"the group's GPU program failed");
  }
  free(comms);
  launch_free(l);
  return status;
}

/* A call joins the open group, or is a group of one; its segments (if any) are the group's then, else
   freed. */
static ncclResult_t enqueue(struct call k,const ncclCollConfig_t *config){
  struct ncclComm *c=k.comm;
  ncclResult_t status=ncclSuccess;
  if(!c)status=FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  else if((unsigned)k.type>=ncclNumTypes)status=FAIL(c,ncclInvalidArgument,"datatype %d",k.type);
  else if(P2P(&k) && (k.peer<0 || k.peer>=c->nranks))status=FAIL(c,ncclInvalidArgument,"peer %d of %d ranks",k.peer,c->nranks);
  else if((k.kind==MESH_BROADCAST || k.kind==MESH_REDUCE) && (k.root<0 || k.root>=c->nranks))status=FAIL(c,ncclInvalidArgument,"root %d of %d ranks",k.root,c->nranks);
  else if(k.count && k.kind!=K_RECV && !k.send && !(k.kind==MESH_BROADCAST && c->rank!=k.root))status=FAIL(c,ncclInvalidArgument,"sendbuff is NULL");
  else if(k.count && k.kind!=K_SEND && !k.recv && !(k.kind==MESH_REDUCE && c->rank!=k.root))status=FAIL(c,ncclInvalidArgument,"recvbuff is NULL");
  else if(k.stream && !k.stream->event)status=FAIL(c,ncclInvalidArgument,"the stream has no event (ncclMeshStreamCreate)");
  else if(atomic_load(&c->aborting))status=FAIL(c,ncclInvalidUsage,"the communicator was aborted");
  else if(k.op<0 || (k.op>=ncclNumOps && k.op-ncclNumOps>=c->nops))status=FAIL(c,ncclInvalidArgument,"operator %d",k.op);
  if(!status && config){
    if(config->magic!=NCCL_API_MAGIC)status=FAIL(c,ncclInvalidArgument,"ncclCollConfig_t not initialized with NCCL_COLLCONFIG_INITIALIZER");
    else if(config->launchCompletionEvent)status=FAIL(c,ncclInvalidUsage,"launchCompletionEvent is not supported (use the stream's event)");
    else if(config->algSelection && *config->algSelection){
      static const char *const names[]={"direct","ring","tree","binomial"};
      char text[256];snprintf(text,sizeof text,"%s",config->algSelection);
      k.force=config->forceAlgSelection!=0;
      for(char *save=NULL,*name=strtok_r(text,", \t",&save);name && !status;name=strtok_r(NULL,", \t",&save)){
        int found=-1;
        for(int a=0;a<4;a++)if(!strcasecmp(name,names[a]))found=a;
        if(found>=0)k.how|=1u<<found;
        else if(k.force)status=FAIL(c,ncclInvalidArgument,"algSelection \"%s\": the mesh's algorithms are direct, ring, tree, binomial",name);
      }
    }
  }
  for(int r=0;!status && k.segments && k.kind!=K_COUNTED && r<c->nranks;r++)
    if(!k.segments[r])status=FAIL(c,ncclInvalidArgument,"counts[%d] is 0 (every rank's segment holds an element)",r);
  /* a persistent point-to-point call takes the word named for it as it is issued (one group holds many): a send's
     ready word (ncclMeshPersistentNext, its whole bytes one range), a receive's landed word (ncclMeshPersistentLanded) */
  if(!status && P2P(&k)){
    pthread_mutex_lock(&persisting.lock);
    if(persisting.building && k.kind==K_SEND && persisting.next.given){
      if(persisting.next.bytes!=k.count*type_bytes[k.type])
        status=FAIL(c,ncclInvalidArgument,"a persistent ncclSend of %zu bytes: its ready word names %llu",k.count*type_bytes[k.type],
                    (unsigned long long)persisting.next.bytes);
      k.ready=(const _Atomic uint64_t *)persisting.next.ready;persisting.next.given=0;
    }
    if(persisting.building && k.kind==K_RECV){k.word=persisting.landed;persisting.landed=NULL;}
    if(persisting.building && persisting.slots.given){
      k.slot_depth=persisting.slots.depth;k.slot_stride=persisting.slots.stride;k.slot_words=persisting.slots.words;persisting.slots.given=0;
    }
    pthread_mutex_unlock(&persisting.lock);
  }
  if(!status && !k.count){free(k.segments);return ncclSuccess;}
  ncclGroupStart();
  if(status){if(!group.error)group.error=status;}
  else {
    if(group.n==group.capacity){
      int capacity=group.capacity?2*group.capacity:16;
      struct call *grown=realloc(group.calls,(size_t)capacity*sizeof *grown);
      if(!grown){if(!group.error)group.error=FAIL(c,ncclSystemError,"allocation");}
      else{group.calls=grown;group.capacity=capacity;}
    }
    if(group.n<group.capacity){
      k.index=P2P(&k)?atomic_load(&c->issued)+1:atomic_fetch_add(&c->issued,1)+1;
      pthread_mutex_lock(&c->lock);k.agreement=c->agreements;pthread_mutex_unlock(&c->lock);
      group.calls[group.n++]=k;k.segments=NULL;
    }
  }
  free(k.segments);
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
#define CALL(...) ((struct call){__VA_ARGS__})

ncclResult_t ncclAllReduceConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_ALLREDUCE,.send=sendbuff,.recv=recvbuff,.count=count,.type=datatype,.op=op,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclAllReduce(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,cudaStream_t stream){
  return ncclAllReduceConfig(sendbuff,recvbuff,count,datatype,op,comm,stream,NULL);
}
ncclResult_t ncclReduceConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_REDUCE,.send=sendbuff,.recv=recvbuff,.count=count,.type=datatype,.op=op,.root=root,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclReduce(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclReduceConfig(sendbuff,recvbuff,count,datatype,op,root,comm,stream,NULL);
}
ncclResult_t ncclBroadcastConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_BROADCAST,.send=sendbuff,.recv=recvbuff,.count=count,.type=datatype,.root=root,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclBroadcast(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclBroadcastConfig(sendbuff,recvbuff,count,datatype,root,comm,stream,NULL);
}
ncclResult_t ncclBcast(void *buff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclBroadcast(buff,buff,count,datatype,root,comm,stream);
}
ncclResult_t ncclReduceScatterConfig(const void *sendbuff,void *recvbuff,size_t recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_REDUCE_SCATTER,.send=sendbuff,.recv=recvbuff,.count=recvcount,.type=datatype,.op=op,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclReduceScatter(const void *sendbuff,void *recvbuff,size_t recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,cudaStream_t stream){
  return ncclReduceScatterConfig(sendbuff,recvbuff,recvcount,datatype,op,comm,stream,NULL);
}
ncclResult_t ncclAllGatherConfig(const void *sendbuff,void *recvbuff,size_t sendcount,ncclDataType_t datatype,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_ALLGATHER,.send=sendbuff,.recv=recvbuff,.count=sendcount,.type=datatype,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclAllGather(const void *sendbuff,void *recvbuff,size_t sendcount,ncclDataType_t datatype,ncclComm_t comm,cudaStream_t stream){
  return ncclAllGatherConfig(sendbuff,recvbuff,sendcount,datatype,comm,stream,NULL);
}
/* MPI_Allgatherv and MPI_Reduce_scatter: equal counts are NCCL's own call; others the planner's cut
   into them (mesh_collective.segments), a copy of them the call's. */
static ncclResult_t enqueue_counts(struct call k,const size_t *counts,const ncclCollConfig_t *config){
  struct ncclComm *c=k.comm;
  if(!c || !counts)return FAIL(c,ncclInvalidArgument,"%s is NULL",c?"counts":"comm");
  int equal=1;
  for(int r=0;r<c->nranks;r++)equal&=counts[r]==counts[0];
  k.count=counts[c->rank];
  if(!equal && !(k.segments=malloc((size_t)c->nranks*sizeof *k.segments)))return FAIL(c,ncclSystemError,"allocation");
  for(int r=0;!equal && r<c->nranks;r++)k.segments[r]=counts[r];
  return enqueue(k,config);
}
ncclResult_t ncclMeshAllGatherVConfig(const void *sendbuff,void *recvbuff,const size_t *counts,ncclDataType_t datatype,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue_counts(CALL(.kind=MESH_ALLGATHER,.send=sendbuff,.recv=recvbuff,.type=datatype,.comm=comm,.stream=stream),counts,config);
}
ncclResult_t ncclMeshAllGatherV(const void *sendbuff,void *recvbuff,const size_t *counts,ncclDataType_t datatype,ncclComm_t comm,cudaStream_t stream){
  return ncclMeshAllGatherVConfig(sendbuff,recvbuff,counts,datatype,comm,stream,NULL);
}
ncclResult_t ncclMeshReduceScatterVConfig(const void *sendbuff,void *recvbuff,const size_t *counts,ncclDataType_t datatype,ncclRedOp_t op,
  ncclComm_t comm,cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue_counts(CALL(.kind=MESH_REDUCE_SCATTER,.send=sendbuff,.recv=recvbuff,.type=datatype,.op=op,.comm=comm,.stream=stream),counts,config);
}
ncclResult_t ncclMeshReduceScatterV(const void *sendbuff,void *recvbuff,const size_t *counts,ncclDataType_t datatype,ncclRedOp_t op,
  ncclComm_t comm,cudaStream_t stream){
  return ncclMeshReduceScatterVConfig(sendbuff,recvbuff,counts,datatype,op,comm,stream,NULL);
}
ncclResult_t ncclMeshAlltoAllCounted(const void *sendbuff,size_t sendrows,const int64_t *counts,const size_t *sendsegs,void *recvbuff,size_t capacity,
  int64_t *received,int64_t *landed,const size_t *recvsegs,uint64_t *arrived,size_t row,ncclDataType_t datatype,ncclComm_t comm,cudaStream_t stream){
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if(!counts || !sendsegs || !recvsegs || !received || !arrived || !row || !sendbuff || !recvbuff)
    return FAIL(comm,ncclInvalidArgument,"ncclMeshAlltoAllCounted: NULL argument or a row of no elements");
  if(sendsegs[comm->rank]!=recvsegs[comm->rank])
    return FAIL(comm,ncclInvalidArgument,"ncclMeshAlltoAllCounted: this rank's segment for itself sent (%zu) and received (%zu) differ",
                sendsegs[comm->rank],recvsegs[comm->rank]);
  uint64_t *segs=malloc(2*(size_t)comm->nranks*sizeof *segs);
  if(!segs)return FAIL(comm,ncclSystemError,"allocation");
  for(int r=0;r<comm->nranks;r++){segs[r]=sendsegs[r];segs[comm->nranks+r]=recvsegs[r];}
  *arrived=0;
  return enqueue(CALL(.kind=K_COUNTED,.send=sendbuff,.recv=recvbuff,.count=1,.type=datatype,.comm=comm,.stream=stream,.segments=segs,.counted=counts,
                      .received=received,.landed=landed,.arrived=arrived,.row=row,.rows=sendrows,.capacity=capacity),NULL);
}
ncclResult_t ncclSend(const void *sendbuff,size_t count,ncclDataType_t datatype,int peer,ncclComm_t comm,cudaStream_t stream){
  return enqueue(CALL(.kind=K_SEND,.send=sendbuff,.count=count,.type=datatype,.peer=peer,.comm=comm,.stream=stream),NULL);
}
ncclResult_t ncclRecv(void *recvbuff,size_t count,ncclDataType_t datatype,int peer,ncclComm_t comm,cudaStream_t stream){
  return enqueue(CALL(.kind=K_RECV,.recv=recvbuff,.count=count,.type=datatype,.peer=peer,.comm=comm,.stream=stream),NULL);
}
/* NCCL 2.32's own lowering of these three (src/enqueue/task_prep/task_classify.cc
   classifyCollToP2pTasks): grouped sends and receives. */
ncclResult_t ncclAlltoAllConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  (void)config;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if((unsigned)datatype>=ncclNumTypes)return FAIL(comm,ncclInvalidArgument,"datatype %d",datatype);
  size_t bytes=count*type_bytes[datatype];
  ncclResult_t status=ncclSuccess;
  ncclGroupStart();
  for(int r=0;r<comm->nranks;r++){
    ncclResult_t a=ncclSend((const char *)sendbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    ncclResult_t b=ncclRecv((char *)recvbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    if(!status)status=a?a:b;
  }
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
ncclResult_t ncclAlltoAll(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclComm_t comm,cudaStream_t stream){
  return ncclAlltoAllConfig(sendbuff,recvbuff,count,datatype,comm,stream,NULL);
}
ncclResult_t ncclGatherConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  (void)config;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if((unsigned)datatype>=ncclNumTypes || root<0 || root>=comm->nranks)return FAIL(comm,ncclInvalidArgument,"datatype %d, root %d",datatype,root);
  size_t bytes=count*type_bytes[datatype];
  ncclGroupStart();
  ncclResult_t status=ncclSend(sendbuff,count,datatype,root,comm,stream);
  for(int r=0;comm->rank==root && r<comm->nranks;r++){
    ncclResult_t b=ncclRecv((char *)recvbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    if(!status)status=b;
  }
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
ncclResult_t ncclGather(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclGatherConfig(sendbuff,recvbuff,count,datatype,root,comm,stream,NULL);
}
ncclResult_t ncclScatterConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  (void)config;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if((unsigned)datatype>=ncclNumTypes || root<0 || root>=comm->nranks)return FAIL(comm,ncclInvalidArgument,"datatype %d, root %d",datatype,root);
  size_t bytes=count*type_bytes[datatype];
  ncclResult_t status=ncclSuccess;
  ncclGroupStart();
  for(int r=0;comm->rank==root && r<comm->nranks;r++){
    ncclResult_t a=ncclSend((const char *)sendbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    if(!status)status=a;
  }
  ncclResult_t b=ncclRecv(recvbuff,count,datatype,root,comm,stream);
  if(!status)status=b;
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
ncclResult_t ncclScatter(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclScatterConfig(sendbuff,recvbuff,count,datatype,root,comm,stream,NULL);
}

/* NCCL's split: every rank's (color, key) all-gathered on `comm`, the ranks of this color ordered by
   key, then rank; the new clique's key derived from the parent's and its count of splits (every rank
   splits in the same order).  Its link map: the config's (ncclMeshConfig_t), else the parent's table with
   the parent's nodes of its ranks. */
ncclResult_t ncclCommSplit(ncclComm_t comm,int color,int key,ncclComm_t *newcomm,ncclConfig_t *config){
  if(!comm || !newcomm)return FAIL(comm,ncclInvalidArgument,"ncclCommSplit: NULL argument");
  if(config && config->magic!=NCCL_API_MAGIC)return FAIL(comm,ncclInvalidArgument,"ncclCommSplit: config not initialized");
  *newcomm=NULL;
  const int n=comm->nranks;
  int64_t mine[2]={color,key},*every=calloc((size_t)n*2,sizeof *every);
  if(!every)return FAIL(comm,ncclSystemError,"allocation");
  /* the all-gather is its own group: the caller's open group (if any) is set aside meanwhile */
  __typeof__(group) open=group;
  memset(&group,0,sizeof group);
  ncclResult_t status=ncclAllGather(mine,every,2,ncclInt64,comm,NULL);
  group=open;
  uint64_t split=comm->splits++;
  if(status || color==NCCL_SPLIT_NOCOLOR){free(every);return status;}
  int *ranks=calloc((size_t)n,sizeof *ranks),*nodes=calloc((size_t)n,sizeof *nodes),size=0;
  if(!ranks || !nodes){free(every);free(ranks);free(nodes);return FAIL(comm,ncclSystemError,"allocation");}
  for(int r=0;r<n;r++)if(every[2*r]==color)ranks[size++]=r;
  for(int i=1;i<size;i++)for(int j=i;j>0;j--){
    int a=ranks[j-1],b=ranks[j];
    if(every[2*a+1]>every[2*b+1] || (every[2*a+1]==every[2*b+1] && a>b)){ranks[j-1]=b;ranks[j]=a;}
  }
  int me=0;
  for(int i=0;i<size;i++){nodes[i]=(int)comm->nodes[ranks[i]];if(ranks[i]==comm->rank)me=i;}
  struct ncclComm *c=comm_new(mix(comm->key^mix(split))^mix((uint64_t)(uint32_t)color<<1)^1,me,size);
  if(!c){free(every);free(ranks);free(nodes);return FAIL(comm,ncclSystemError,"allocation");}
  ncclMeshConfig_t inherited=NCCL_MESH_CONFIG_INITIALIZER;
  inherited.links=comm->table;inherited.nodes=nodes;
  status=links_for(c,config && config->size==sizeof(ncclMeshConfig_t)?config:&inherited.base);
  free(every);free(ranks);free(nodes);
  if(!status)status=comm_start(c);
  if(status){snprintf(last_error,sizeof last_error,"%s",c->error[0]?c->error:comm->error);comm_free(c);return status;}
  *newcomm=c;
  return ncclSuccess;
}

/* ---- the stream ---- */
ncclResult_t ncclMeshStreamCreate(cudaStream_t *stream,void *queue){
  if(!stream)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamCreate: stream is NULL");
  /* never the caller's queue (nccl.h): the dispatcher commits the work after a gate late, and a caller's command
     buffer waiting on the call there would hold it */
  (void)queue;
  char error[256];
  pthread_mutex_lock(&global_lock);
  int failed=nccl_mesh_gpu_attach(error,sizeof error);
  pthread_mutex_unlock(&global_lock);
  if(failed)return FAIL(NULL,ncclUnhandledCudaError,"the GPU: %s",error);
  struct ncclMeshStream *s=calloc(1,sizeof *s);
  if(!s)return FAIL(NULL,ncclSystemError,"allocation");
  s->made=1;s->refs=1;s->queue=nccl_mesh_queue_create();
  s->event=s->queue?nccl_mesh_event_create(s->queue):NULL;
  if(!s->event){if(s->made && s->queue)nccl_mesh_release(s->queue);free(s);return FAIL(NULL,ncclUnhandledCudaError,"the stream's Metal queue and event");}
  *stream=s;
  return ncclSuccess;
}
/* A stream's holder gone: the stream freed with its last (the caller's, and its programs' until they have run). */
static void stream_release(struct ncclMeshStream *s){
  if(__atomic_sub_fetch(&s->refs,1,__ATOMIC_ACQ_REL))return;
  nccl_mesh_release(s->event);
  if(s->made)nccl_mesh_release(s->queue);
  free(s);
}
ncclResult_t ncclMeshStreamDestroy(cudaStream_t stream){
  if(!stream)return ncclSuccess;
  if(stream->pending)ncclMeshStreamSynchronize(stream);
  stream_release(stream);
  return ncclSuccess;
}
/* The stream's kept programs committed to its queue, after its committed ones (the dispatcher's gate on their
   completion); stream_lock held. */
static ncclResult_t flush_kept(struct ncclMeshStream *s){
  if(!s->pending)return ncclSuccess;
  struct run *r=run_new(s->queue,NULL,NULL,0);
  struct owned o={s->queue,NULL,r,0};
  o.current=r?owned_entry(&o):NULL;
  if(!o.current){if(r)run_release(r);return FAIL(NULL,ncclUnhandledCudaError,"a command buffer on the stream's queue");}
  if(s->committed)entry_gate(o.current,s->event,s->committed);
  void *program=take_kept(s,o.current->program,owned_gate,&o);
  s->committed=s->value;
  count(NULL,COMMITS,1);
  if(program)owned_finish(&o);
  run_release(r);
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamSynchronize(cudaStream_t stream){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamSynchronize: no stream");
  uint64_t value;
  pthread_mutex_lock(&stream_lock);
  ncclResult_t status=flush_kept(stream);
  value=stream->value;
  pthread_mutex_unlock(&stream_lock);
  if(status)return status;
  /* however late (a program run again included; one failed for good has its values given by the host) */
  while(nccl_mesh_event_value(stream->event)<value)usleep(20);
  if(__atomic_load_n(&stream->failed,__ATOMIC_ACQUIRE))return FAIL(NULL,ncclUnhandledCudaError,"ncclMeshStreamSynchronize: a GPU program of the stream failed");
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamQuery(cudaStream_t stream){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamQuery: no stream");
  pthread_mutex_lock(&stream_lock);uint64_t value=stream->value;int kept=stream->pending!=NULL;pthread_mutex_unlock(&stream_lock);
  if(!kept && nccl_mesh_event_value(stream->event)>=value)
    return __atomic_load_n(&stream->failed,__ATOMIC_ACQUIRE)?FAIL(NULL,ncclUnhandledCudaError,"ncclMeshStreamQuery: a GPU program of the stream failed"):ncclSuccess;
  return ncclInProgress;
}
ncclResult_t ncclMeshStreamDefer(cudaStream_t stream,int defer){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamDefer: no stream");
  pthread_mutex_lock(&stream_lock);
  stream->deferred=defer!=0;
  ncclResult_t status=defer?ncclSuccess:flush_kept(stream);
  pthread_mutex_unlock(&stream_lock);
  return status;
}
ncclResult_t ncclMeshEncodeCopies(void *commandBuffer,int n,void *const *dst,void *const *src,const size_t *offset,const size_t *bytes,
  uint64_t *word,uint64_t value){
  if(!commandBuffer || n<0 || (n && (!dst || !src || !offset || !bytes)) || !word)return FAIL(NULL,ncclInvalidArgument,"ncclMeshEncodeCopies: NULL argument");
  void **buffers=calloc((size_t)n+1,sizeof *buffers);uint64_t *at=calloc((size_t)n+1,sizeof *at);
  ncclResult_t status=buffers && at?ncclSuccess:FAIL(NULL,ncclSystemError,"allocation");
  pthread_mutex_lock(&heap.lock);
  for(int i=0;i<=n && !status;i++){
    const void *p=i<n?dst[i]:(const void *)word;
    struct span *s=span_of(p,i<n?(bytes[i]?bytes[i]:1):8);
    if(!s)status=FAIL(NULL,ncclInvalidArgument,"ncclMeshEncodeCopies: %p is not in a live allocation of the window (ncclMemAlloc)",p);
    else{buffers[i]=s->buffer;nccl_mesh_retain(s->buffer);at[i]=(uint64_t)((const unsigned char *)p-s->at);}
  }
  pthread_mutex_unlock(&heap.lock);
  if(!status){
    pthread_mutex_lock(&stream_lock);
    for(int i=0;i<n;i++)nccl_mesh_program_copy(commandBuffer,buffers[i],at[i],src[i],offset[i],bytes[i],0,1,NULL,0,0,0);
    nccl_mesh_program_publish(commandBuffer,buffers[n],at[n],value,NULL,0,0);
    nccl_mesh_program_end(commandBuffer);
    pthread_mutex_unlock(&stream_lock);
  }
  for(int i=0;buffers && i<=n;i++)if(buffers[i])nccl_mesh_release(buffers[i]);
  free(buffers);free(at);
  return status;
}
ncclResult_t ncclMeshStreamEncodeWait(cudaStream_t stream,void *commandBuffer,ncclMeshGate_t gate,void *argument){
  if(!stream || !stream->event || !commandBuffer)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamEncodeWait: NULL argument");
  if(!gate)gate=inline_gate;
  pthread_mutex_lock(&stream_lock);
  void *program=commandBuffer;
  /* the stream's committed programs first: a gate of their completion */
  if(stream->committed && nccl_mesh_event_value(stream->event)<stream->committed)program=gate(argument,program,stream->event,stream->committed);
  if(program)program=take_kept(stream,program,gate,argument);
  pthread_mutex_unlock(&stream_lock);
  return program?ncclSuccess:FAIL(NULL,ncclUnhandledCudaError,"ncclMeshStreamEncodeWait: the caller's gate gave no command buffer");
}
/* ---- agreement: ULFM's MPI_Comm_agree [Bland et al. 2013; MPI 5.0 draft, chapter "Process Fault
   Tolerance"] after a failure ---- */
/* A rank's vote: whether the flood has reached it, the index of its first call that failed since the
   last agreement (0: none), whether its connections are whole, and the hash of its link table's map of
   the communicator (what its next call plans on). */
ncclResult_t ncclMeshPersistentBegin(void){
  pthread_mutex_lock(&persisting.lock);
  struct persistent *p=persisting.building?NULL:calloc(1,sizeof *p);
  if(p){persisting.building=p;persisting.next.given=0;persisting.landed=NULL;persisting.slots.given=0;}
  pthread_mutex_unlock(&persisting.lock);
  return p?ncclSuccess:FAIL(NULL,ncclInvalidUsage,"ncclMeshPersistentBegin: persistent calls are being made already (or no memory)");
}
ncclResult_t ncclMeshPersistentEnd(void **handle,int *calls){
  pthread_mutex_lock(&persisting.lock);
  struct persistent *p=persisting.building;
  persisting.building=NULL;
  pthread_mutex_unlock(&persisting.lock);
  if(!p)return FAIL(NULL,ncclInvalidUsage,"ncclMeshPersistentEnd without ncclMeshPersistentBegin");
  *handle=p;*calls=p->n;
  /* each cut's place among the recording's, after its leading one */
  for(int i=0,cuts=0;i<p->n;i++)if(p->items[i]->cut)p->items[i]->cut=++cuts;
  /* each call's GPU work taken into a command buffer (ncclMeshStreamEncodeWait), which a recording holds */
  int kept=0;
  pthread_mutex_lock(&stream_lock);
  for(int i=0;i<p->n;i++)kept|=p->items[i]->launch->nmarks && p->items[i]->launch->marks[0].stream->pending!=NULL;
  pthread_mutex_unlock(&stream_lock);
  return kept?FAIL(p->comm,ncclInvalidUsage,"ncclMeshPersistentEnd: a persistent call's GPU work was never encoded (ncclMeshStreamEncodeWait)"):ncclSuccess;
}
ncclResult_t ncclMeshPersistentGate(void *handle,void *event,uint64_t value){
  struct persistent *p=handle;
  if(!p || !event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentGate: no handle or no event");
  if(p->gate_event)nccl_mesh_release(p->gate_event);
  nccl_mesh_retain(event);
  p->gate_event=event;p->gate_value=value;
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentGates(void *handle,int *gates){
  struct persistent *p=handle;
  if(!p || !gates)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentGates: NULL argument");
  *gates=p->gates;
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentStart(void *handle,void *event,uint64_t value,uint64_t stride,uint64_t count){
  struct persistent *p=handle;
  if(!p || !event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentStart: no handle or no event");
  if(!p->n || !count)return ncclSuccess;
  if(p->gates && !p->gate_event)return FAIL(p->comm,ncclInvalidUsage,"ncclMeshPersistentStart: the recording's gates have no event (ncclMeshPersistentGate)");
  struct ncclComm *c=p->comm;
  pthread_mutex_lock(&c->lock);
  if(c->run || c->head){pthread_mutex_unlock(&c->lock);return FAIL(c,ncclInvalidUsage,"ncclMeshPersistentStart: the communicator has calls in flight");}
  /* not revoked, and every rank of the calls' plans connected (one the last agreement left out, its node since back,
     is not: the communicator agrees first, a remote error as an eager call's) */
  int unconnected=-1;
  for(int i=0;i<p->n && unconnected<0;i++)for(int j=0;j<p->items[i]->n && unconnected<0;j++){
    const struct call *k=p->items[i]->calls+j;
    if(P2P(k) && !(k->kind==K_SEND?c->peers[k->peer].send[CH_P2P]:c->peers[k->peer].recv[CH_P2P]))unconnected=k->peer;
    for(uint32_t t=0;t<k->nsteps && unconnected<0;t++){
      const struct peer *e=c->peers+k->steps[t].peer;
      if(!e->send[CH_COLL] || !e->recv[CH_COLL])unconnected=(int)k->steps[t].peer;
    }
  }
  const int revoked=atomic_load(&c->broken)!=0;
  if(revoked || unconnected>=0){
    pthread_mutex_unlock(&c->lock);
    if(revoked)return FAIL(c,ncclRemoteError,"ncclMeshPersistentStart: the communicator is revoked (ncclMeshCommAgree first)");
    return FAIL(c,ncclRemoteError,"ncclMeshPersistentStart: no connection reaches rank %d (ncclMeshCommAgree connects the ranks that stay)",unconnected);
  }
  if(p->event)nccl_mesh_release(p->event);
  nccl_mesh_retain(event);
  p->event=event;p->value=value;p->stride=stride;p->total=count*(uint64_t)p->n;p->result=ncclSuccess;
  p->posting=0;p->post_failed=ncclSuccess;
  /* each published range's count before the run (the last run's publications done) */
  for(int i=0;i<p->n;i++)for(int j=0;j<p->items[i]->n;j++){
    struct call *k=p->items[i]->calls+j;
    for(uint32_t r=0;k->ready && r<k->ranges;r++)k->ready_base[r]=atomic_load_explicit(k->ready+r,memory_order_acquire);
  }
  atomic_store(&p->next,0);
  __atomic_store_n(&c->run,p,__ATOMIC_RELEASE);
  pthread_cond_broadcast(&c->cond);
  pthread_mutex_unlock(&c->lock);
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentWait(void *handle){
  struct persistent *p=handle;
  if(!p || !p->n)return ncclSuccess;
  struct ncclComm *c=p->comm;
  pthread_mutex_lock(&c->lock);
  while(c->run==p)pthread_cond_wait(&c->cond,&c->lock);
  pthread_mutex_unlock(&c->lock);
  return p->result;
}
/* The persistent calls' communicator. */
ncclResult_t ncclMeshPersistentComm(void *handle,ncclComm_t *comm){
  struct persistent *p=handle;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentComm: comm is NULL");
  *comm=p?p->comm:NULL;
  return ncclSuccess;
}
/* The persistent calls' communicator revoked (their replay failed): every part not yet started fails at its start,
   sending nothing; the caller may then let the replay's cuts pass so that every part starts and ends. */
ncclResult_t ncclMeshPersistentAbort(void *handle){
  struct persistent *p=handle;
  if(p && p->comm)comm_revoke(p->comm);
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentFree(void *handle){
  struct persistent *p=handle;
  if(!p)return ncclSuccess;
  if(p->comm){
    pthread_mutex_lock(&p->comm->lock);
    const int running=p->comm->run==p;
    pthread_mutex_unlock(&p->comm->lock);
    if(running)return FAIL(p->comm,ncclInvalidUsage,"ncclMeshPersistentFree: its calls are running (ncclMeshPersistentWait first)");
  }
  for(int i=0;i<p->n;i++){
    struct item *it=p->items[i];
    calls_free(it->calls,it->n);launch_free(it->launch);free(it);
  }
  if(p->event)nccl_mesh_release(p->event);
  if(p->gate_event)nccl_mesh_release(p->gate_event);
  free(p->items);free(p);
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentNext(const uint64_t *ready,uint64_t range,uint64_t bytes,int fresh){
  pthread_mutex_lock(&persisting.lock);
  const int building=persisting.building!=NULL;
  if(building)persisting.next=(__typeof__(persisting.next)){ready,range,bytes,fresh,1};
  pthread_mutex_unlock(&persisting.lock);
  return building?ncclSuccess:FAIL(NULL,ncclInvalidUsage,"ncclMeshPersistentNext outside ncclMeshPersistentBegin and End");
}
ncclResult_t ncclMeshPersistentLanded(uint64_t *landed){
  if(!landed || ((uintptr_t)landed&7))return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentLanded: an 8-byte word of a window allocation");
  pthread_mutex_lock(&persisting.lock);
  const int building=persisting.building!=NULL;
  if(building)persisting.landed=landed;
  pthread_mutex_unlock(&persisting.lock);
  return building?ncclSuccess:FAIL(NULL,ncclInvalidUsage,"ncclMeshPersistentLanded outside ncclMeshPersistentBegin and End");
}
ncclResult_t ncclMeshPersistentSlots(uint64_t depth,uint64_t stride,uint64_t words){
  if(!depth || (depth>1 && !stride) || (words&7))return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentSlots: a depth, a stride between buffers and whole words between words");
  pthread_mutex_lock(&persisting.lock);
  const int building=persisting.building!=NULL;
  if(building){persisting.slots.depth=depth;persisting.slots.stride=stride;persisting.slots.words=words;persisting.slots.given=1;}
  pthread_mutex_unlock(&persisting.lock);
  return building?ncclSuccess:FAIL(NULL,ncclInvalidUsage,"ncclMeshPersistentSlots outside ncclMeshPersistentBegin and End");
}
ncclResult_t ncclMeshPersistentOrigin(void *handle,uint64_t origin){
  struct persistent *p=handle;
  if(!p)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentOrigin: no handle");
  if(p->comm){
    pthread_mutex_lock(&p->comm->lock);
    const int running=p->comm->run==p;
    pthread_mutex_unlock(&p->comm->lock);
    if(running)return FAIL(p->comm,ncclInvalidUsage,"ncclMeshPersistentOrigin: its calls are running");
  }
  p->origin=origin;
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentCut(int *cut,int *groups){
  if(!cut || !groups)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentCut: cut or groups is NULL");
  pthread_mutex_lock(&persisting.lock);
  *cut=persisting.cut;*groups=persisting.groups;persisting.groups=0;
  pthread_mutex_unlock(&persisting.lock);
  return ncclSuccess;
}
static ncclMeshCounts_t counts_of(const uint64_t *c,int timed);
/* A call's tally once its part has ended (its DONE word, written last), else none yet (zeros). */
static ncclMeshCounts_t tally_of(const uint64_t *c){
  if(__atomic_load_n(c+DONE,__ATOMIC_ACQUIRE))return counts_of(c,1);
  return (ncclMeshCounts_t){0};
}
ncclResult_t ncclMeshPersistentSlotCounts(void *handle,uint64_t slot,ncclMeshCounts_t *counts,int capacity,int *count){
  struct persistent *p=handle;
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentSlotCounts: count is NULL");
  *count=0;
  for(int i=0;p && i<p->n;i++)for(int j=0;j<p->items[i]->n;j++,++*count){
    const struct call *k=p->items[i]->calls+j;
    if(counts && *count<capacity)counts[*count]=k->tallies?tally_of(k->tallies+(slot%k->slot_depth)*TALLIES):(ncclMeshCounts_t){0};
  }
  return ncclSuccess;
}
ncclResult_t ncclMeshPersistentCounts(void *handle,ncclMeshCounts_t *counts,int capacity,int *count){
  struct persistent *p=handle;
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshPersistentCounts: count is NULL");
  *count=0;
  for(int i=0;p && i<p->n;i++)for(int j=0;j<p->items[i]->n;j++,++*count)
    if(counts && *count<capacity && p->items[i]->calls[j].tally)counts[*count]=tally_of(p->items[i]->calls[j].tally);
  return ncclSuccess;
}

struct vote { uint64_t known,failed,whole,map; };
static uint64_t map_hash(const struct ncclComm *c){
  uint64_t h=mix(((uint64_t)c->map.kind<<32)^c->map.links);
  for(uint32_t l=0;l<c->map.links;l++)h=mix(h^((uint64_t)c->map.link[l][0]<<32)^c->map.link[l][1]);
  for(int i=0;i<c->nranks*c->nranks;i++){
    uint32_t a,b;memcpy(&a,&c->cost[i][0],4);memcpy(&b,&c->cost[i][1],4);
    h=mix(h^((uint64_t)a<<32)^b);
  }
  return h;
}
/* Whether this rank's connections are whole: not revoked, and alive with every rank the map links. */
static int whole(struct ncclComm *c){
  if(atomic_load(&c->broken) || !c->open)return 0;
  for(int p=0;p<c->nranks;p++)if(linked(&c->map,c->rank,p))
    for(int ch=0;ch<CHANNELS;ch++)if(!mesh_net_alive(c->peers[p].send[ch]) || !mesh_net_alive(c->peers[p].recv[ch]))return 0;
  return 1;
}
/* A request posted (retried while its ring is full, the bridge consuming it), or none where rank p departed meanwhile. */
static ncclResult_t host_post(struct ncclComm *c,int send,void *comm,void *data,size_t bytes,int p,void **request){
  for(*request=NULL;;){
    int result=send?mesh_net_isend(comm,data,bytes,0,window.mh,NULL,request):
      mesh_net_irecv(comm,1,&data,&bytes,(int[]){0},(void *[]){window.mh},NULL,request);
    if(result)return departed(c,p)?ncclSuccess:net_failure(c,result,send?"ncclMeshCommAgree: an isend":"ncclMeshCommAgree: an irecv");
    if(*request || departed(c,p))return ncclSuccess;
    sched_yield();
  }
}
/* One round of FloodSet [Lynch, "Distributed Algorithms" 1996, §6.2.1]: the votes this rank knows sent
   to each neighbour, each neighbour's received first, and theirs merged in.  A neighbour that departs meanwhile
   leaves the round (`gone`): its requests are abandoned on connections closed after the agreement. */
static ncclResult_t flood(struct ncclComm *c,struct peer *peers,unsigned char *want,unsigned char *gone,struct vote *votes,unsigned char *buffers,
  size_t slot){
  const int n=c->nranks;
  void **requests=calloc((size_t)n*2,sizeof *requests);
  if(!requests)return FAIL(c,ncclSystemError,"allocation");
  memcpy(buffers,votes,(size_t)n*sizeof *votes);
  ncclResult_t status=ncclSuccess;
  for(int p=0;p<n && !status;p++)if(want[p]){
    status=host_post(c,0,peers[p].recv[CH_P2P],buffers+slot*(size_t)(p+1),(size_t)n*sizeof *votes,p,requests+2*p);
    if(!status)status=host_post(c,1,peers[p].send[CH_P2P],buffers,(size_t)n*sizeof *votes,p,requests+2*p+1);
  }
  for(int i=0;i<2*n && !status;i++)while(requests[i] && want[i/2]){
    int done=0,size=0,result=mesh_net_test(requests[i],&done,&size);
    if(departed(c,i/2)){want[i/2]=0;gone[i/2]=1;break;}
    if(result)status=net_failure(c,result,"ncclMeshCommAgree: a vote");
    else if(done)requests[i]=NULL;
    else sched_yield();
  }
  for(int p=0;p<n && !status;p++)if(want[p]){
    const struct vote *theirs=(const struct vote *)(buffers+slot*(size_t)(p+1));
    for(int r=0;r<n;r++)if(theirs[r].known && !votes[r].known)votes[r]=theirs[r];
  }
  free(requests);
  return status;
}
/* Every connection closed since the last agreement vacated by the bridge (nothing of it on the wire: its peer
   closed it too, or its link was left), however late; then the memory held for the failed parts is given back. */
static ncclResult_t vacate(struct ncclComm *c){
  while(c->nclosed){
    if(mesh_net_vacated(c->closed[c->nclosed-1])){c->nclosed--;continue;}
    usleep(200);
  }
  nccl_mesh_event_signal(c->quiet,c->quiet_value);
  return ncclSuccess;
}
/* The view an agreement starts from: a hash of the ranks not departed. */
static uint64_t view_of(const struct ncclComm *c,const unsigned char *gone){
  uint64_t h=mix((uint64_t)c->nranks);
  for(int r=0;r<c->nranks;r++)if(!gone[r])h=mix(h^(uint64_t)(r+1));
  return h;
}
ncclResult_t ncclMeshCommAgree(ncclComm_t comm,uint64_t *failed,uint64_t *epoch){
  if(!comm || !failed || !epoch)return FAIL(comm,ncclInvalidArgument,"ncclMeshCommAgree: NULL argument");
  struct ncclComm *c=comm;
  const int n=c->nranks;
  drain(c);
  pthread_mutex_lock(&c->lock);
  if(c->run){pthread_mutex_unlock(&c->lock);return FAIL(c,ncclInvalidUsage,"ncclMeshCommAgree while persistent calls run (ncclMeshPersistentWait first)");}
  if(atomic_load(&c->broken) && c->open)close_peers(c,c->peers,1);
  c->agreeing=1;
  pthread_mutex_unlock(&c->lock);
  const size_t slot=slice((size_t)n*sizeof(struct vote));
  unsigned char *want=calloc((size_t)n,1),*gone=calloc((size_t)n,1),*buffers=NULL;
  struct peer *fresh=calloc((size_t)n,sizeof *fresh);
  struct vote *votes=calloc((size_t)n,sizeof *votes);
  ncclResult_t status=want && gone && fresh && votes?ncclSuccess:FAIL(c,ncclSystemError,"allocation");
  /* the flood's neighbours: the ranks the stated map links to this one, up or not, but those departed as this node
     observes it (departed: its bridge left, this node's region was replaced, or its process exited; a membership
     change): the agreement is among the ranks that stay, and a rank departing while it runs leaves it then */
  c->epoch=mesh_link_table_read(c->table,c->seen);
  for(int p=0;!status && p<n;p++){
    gone[p]=departed(c,p);
    want[p]=p!=c->rank && !gone[p] && mesh_link_at(c->seen,c->nodes[c->rank],c->nodes[p])->stated &&
      mesh_link_at(c->seen,c->nodes[p],c->nodes[c->rank])->stated;
  }
  /* the connections' key: the view and the agreements made in a row with it (a returning rank's view and the ranks
     that stayed are new alike, so both count from 0) */
  const uint64_t view=view_of(c,gone);
  if(view!=c->view){c->view=view;c->in_view=0;}
  const uint64_t key=mix(c->key^mix(view^mix(++c->in_view)));
  pthread_mutex_lock(&c->lock);c->agreements++;pthread_mutex_unlock(&c->lock);
  if(!status && n>1)status=connect_ranks(c,key,want,fresh,gone);
  if(!status)status=span_alloc(slot*(size_t)(n+1),&buffers,NULL);
  /* the votes flooded n - 1 rounds; again, once the tables move, until every rank's map is the same */
  for(int same=0;!status && !same;){
    c->epoch=mesh_link_table_read(c->table,c->seen);
    maps(c);
    memset(votes,0,(size_t)n*sizeof *votes);
    pthread_mutex_lock(&c->lock);
    votes[c->rank]=(struct vote){1,c->failed,(uint64_t)whole(c),map_hash(c)};
    pthread_mutex_unlock(&c->lock);
    for(int round=1;round<n && !status;round++)status=flood(c,fresh,want,gone,votes,buffers,slot);
    same=1;
    for(int r=0;!status && r<n;r++){
      if(gone[r])continue;
      if(!votes[r].known)status=FAIL(c,ncclRemoteError,"ncclMeshCommAgree: rank %d's vote reached this rank by no stated link",r);
      same&=votes[r].map==votes[c->rank].map;
    }
    /* not yet one map: the tables are waited for (paced, never failed by time) and the votes flooded again */
    for(uint64_t e=c->epoch,until=now_ns()+20000000;!status && !same && atomic_load(&c->table->epoch)==e && now_ns()<until;)usleep(200);
  }
  /* the first failed call among the ranks; after one, or where a rank's connections are not whole, every
     rank takes the agreement's connections as its own */
  uint64_t first=0;int kept=1;
  for(int r=0;!status && r<n;r++){
    if(gone[r])continue;
    if(votes[r].failed && (!first || votes[r].failed<first))first=votes[r].failed;
    kept&=votes[r].whole!=0;
  }
  for(int r=0;!status && r<n;r++)kept&=!gone[r] || !c->voted[r] || r==c->rank;
  pthread_mutex_lock(&c->lock);
  if(!status && (first || !kept)){
    close_peers(c,c->peers,1);
    memcpy(c->peers,fresh,(size_t)n*sizeof *fresh);memset(fresh,0,(size_t)n*sizeof *fresh);
    owners(c,c->peers);
    c->open=1;
  }
  close_peers(c,fresh,1);
  pthread_mutex_unlock(&c->lock);
  if(!status)status=vacate(c);
  /* the vote buffers given back once the connections that received into them are vacated (the quiet point) */
  if(buffers)span_release(buffers,c->quiet,c->quiet_value);
  pthread_mutex_lock(&c->lock);
  if(!status){atomic_store(&c->issued,0);c->failed=0;c->cause[0]=0;atomic_store(&c->broken,0);atomic_store(&c->async,0);atomic_store(&c->gpu_failed,0);c->alive=0;}
  c->agreeing=0;
  pthread_mutex_unlock(&c->lock);
  if(!status){*failed=first;*epoch=c->epoch;for(int r=0;r<n;r++)c->voted[r]=!gone[r];}
  free(want);free(gone);free(fresh);free(votes);
  return status;
}
/* The ranks this rank's bridge observes as members now: 1 each but those departed (departed: its node's bridge left
   the mesh, this node's region was replaced, or its process exited); a rank the last agreement left out counts once
   its node is back (the next agreement connects it). */
ncclResult_t ncclMeshCommMembers(ncclComm_t comm,int *members){
  if(!comm || !members)return FAIL(comm,ncclInvalidArgument,"ncclMeshCommMembers: NULL argument");
  for(int r=0;r<comm->nranks;r++)members[r]=r==comm->rank || !departed(comm,r);
  return ncclSuccess;
}
ncclResult_t ncclMeshCommVoters(ncclComm_t comm,int *voters){
  if(!comm || !voters)return FAIL(comm,ncclInvalidArgument,"ncclMeshCommVoters: NULL argument");
  for(int r=0;r<comm->nranks;r++)voters[r]=comm->voted[r];
  return ncclSuccess;
}
ncclResult_t ncclMeshGroupEpochs(uint64_t *epochs,int capacity,int *count){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshGroupEpochs: count is NULL");
  *count=plans.n;
  for(int i=0;epochs && i<plans.n && i<capacity;i++)epochs[i]=plans.epoch[i];
  return ncclSuccess;
}
_Static_assert(sizeof(ncclMeshLink_t)==sizeof(struct mesh_link_state),"the link table");
ncclResult_t ncclMeshLinksAttach(const char *region,void **links){
  if(!links)return FAIL(NULL,ncclInvalidArgument,"ncclMeshLinksAttach: links is NULL");
  if(!region)region=getenv("MESH_REGION");
  int error=mesh_link_table_open(region,0,0,(struct mesh_link_table **)links);
  return error?FAIL(NULL,ncclSystemError,"the link table of the bridge region %s: %s",region?region:"/mesh0",strerror(error)):ncclSuccess;
}
ncclResult_t ncclMeshLinksDetach(void *links){mesh_link_table_close(links);return ncclSuccess;}
ncclResult_t ncclMeshLinksState(void *links,const char *path){
  if(!links || !path)return FAIL(NULL,ncclInvalidArgument,"ncclMeshLinksState: NULL argument");
  int error=mesh_link_table_state(links,path);
  return error?FAIL(NULL,error==EINVAL?ncclInvalidArgument:ncclSystemError,"the link map %s: %s%s",path,strerror(error),
                    error==EINVAL?" (a link line without its alpha and beta, or a node past the table's)":""):ncclSuccess;
}
ncclResult_t ncclMeshLinksRead(void *links,uint32_t *nodes,ncclMeshLink_t *snapshot,uint32_t *present,uint64_t *reported,uint32_t *node,
  uint64_t *epoch){
  if(!links)return FAIL(NULL,ncclInvalidArgument,"ncclMeshLinksRead: links is NULL");
  struct mesh_link_table *t=links;
  struct mesh_link_contents *c=mesh_link_contents_new(t->nodes);
  if(!c)return FAIL(NULL,ncclSystemError,"allocation");
  const size_t n=t->nodes;
  uint64_t at=mesh_link_table_read(t,c);
  if(nodes)*nodes=t->nodes;
  if(snapshot)memcpy(snapshot,c->link,n*n*sizeof *snapshot);
  if(present)memcpy(present,mesh_link_present(c),n*sizeof *present);
  for(size_t v=0;reported && v<n;v++)reported[v]=atomic_load_explicit(&mesh_link_reported(t)[v],memory_order_acquire);
  free(c);
  if(node)*node=((struct mesh_link_table *)links)->node;
  if(epoch)*epoch=at;
  return ncclSuccess;
}
ncclResult_t ncclMeshGroupPlans(int *algorithms,int *roots,int capacity,int *count){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshGroupPlans: count is NULL");
  *count=plans.n;
  for(int i=0;i<plans.n && i<capacity;i++){if(algorithms)algorithms[i]=plans.how[i];if(roots)roots[i]=plans.root[i];}
  return ncclSuccess;
}
static ncclMeshCounts_t counts_of(const uint64_t *c,int timed){
  return (ncclMeshCounts_t){.cpuCopyBytes=c[CPU_COPY],.gpuCopyBytes=c[GPU_COPY],.gpuKernels=c[GPU_KERNELS],.hostWaits=c[HOST_WAITS],
    .inputWaits=c[INPUT_WAITS],.sentBytes=c[SENT],.receivedBytes=c[RECEIVED],.startNs=timed?c[STARTED]:0,.endNs=timed?c[ENDED]:0,
    .arrivedNs=timed?c[ARRIVED]:0,.gpuEventWaits=c[GPU_EVENT_WAITS],.gpuWordWaits=c[GPU_WORD_WAITS],.hostWordWaits=c[HOST_WORD_WAITS],
    .commits=c[COMMITS],.wakeups=c[WAKEUPS],.buffers=c[BUFFERS],.sends=c[SENDS],.grantWaits=c[GRANT_WAITS],.readyNs=timed?c[READIED]:0};
}
_Static_assert(sizeof(ncclMeshStatsLink_t)==sizeof(struct mesh_stats_link) && MESH_STATS_CLIENT==24 && COUNTS+2<=MESH_STATS_CLIENT,"ncclMeshStats_t");
/* The ring's entries (mesh.h) read as ncclMeshStats_t; the last of this process's among them, its counts. */
static void stats_copy(const struct mesh_stats_entry *e,ncclMeshStats_t *out){
  memset(out,0,sizeof *out);
  out->evaluation=atomic_load_explicit(&e->evaluation,memory_order_relaxed);out->ns=e->ns;out->pid=e->pid;
  out->links=e->links<8?e->links:8;
  memcpy(out->client,e->client,sizeof out->client);
  for(uint32_t l=0;l<out->links;l++)memcpy(out->link+l,e->link+l,sizeof out->link[l]);
}
ncclResult_t ncclMeshStats(uint64_t evaluation,uint64_t first,ncclMeshStats_t *out,int capacity,int *count,uint64_t *evaluations,uint32_t *lag){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStats: count is NULL");
  ncclResult_t status=global_attach();
  if(status)return status;
  uint64_t ended=0;uint32_t k=0;size_t bytes=0;
  mesh_net_stats_shape(&ended,&k,&bytes);
  if(evaluations)*evaluations=ended;
  if(lag)*lag=k;
  *count=0;
  if(!evaluation || evaluation>ended)evaluation=ended;
  unsigned char *entries=capacity>0?malloc((size_t)capacity*bytes):NULL;
  if(capacity>0 && !entries)return FAIL(NULL,ncclSystemError,"allocation");
  const size_t got=capacity>0?mesh_net_stats_read(evaluation,first,entries,(size_t)capacity):0;
  for(size_t i=0;i<got;i++)stats_copy((const struct mesh_stats_entry *)(entries+i*bytes),out+i);
  *count=(int)got;
  free(entries);
  return ncclSuccess;
}
ncclResult_t ncclMeshGetCounts(ncclMeshCounts_t *counts){
  if(!counts)return FAIL(NULL,ncclInvalidArgument,"ncclMeshGetCounts: counts is NULL");
  memset(counts,0,sizeof *counts);
  /* this process's counts as its last evaluation K or more before the region's last recorded them */
  ncclMeshStats_t last[64];int n=0;uint64_t ended=0;uint32_t k=0;
  ncclResult_t status=ncclMeshStats(0,0,NULL,0,&n,&ended,&k);
  if(status)return status;
  const uint64_t top=ended>k?ended-k:0;
  const uint32_t pid=(uint32_t)getpid();
  for(uint64_t first=top>=64?top-63:1;top && first>=1;first=first>64?first-64:0){
    if((status=ncclMeshStats(ended,first,last,64,&n,NULL,NULL)))return status;
    for(int i=n-1;i>=0;i--)if(last[i].pid==pid){
      uint64_t now[COUNTS];
      for(int j=0;j<COUNTS;j++)now[j]=last[i].client[j];
      *counts=counts_of(now,0);
      return ncclSuccess;
    }
    if(first==1)break;
  }
  return ncclSuccess;
}
ncclResult_t ncclMeshTallyCounts(void *tally,ncclMeshCounts_t *counts,int capacity,int *count){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshTallyCounts: count is NULL");
  struct tally *t=tally;
  *count=t?t->n:0;
  for(int i=0;t && counts && i<t->n && i<capacity;i++)counts[i]=tally_of(t->counts[i]);
  return ncclSuccess;
}
ncclResult_t ncclMeshGroupCounts(ncclMeshCounts_t *counts,int capacity,int *count){return ncclMeshTallyCounts(plans.tally,counts,capacity,count);}
ncclResult_t ncclMeshGroupTally(void **tally){
  if(!tally)return FAIL(NULL,ncclInvalidArgument,"ncclMeshGroupTally: tally is NULL");
  if((*tally=plans.tally))atomic_fetch_add(&plans.tally->refs,1);
  return ncclSuccess;
}
ncclResult_t ncclMeshTallyRelease(void *tally){tally_release(tally);return ncclSuccess;}

/* ---- the window's allocations, for the caller ---- */
ncclResult_t ncclMeshMemBuffer(const void *ptr,void **buffer,size_t *offset){
  if(!ptr || !buffer || !offset)return FAIL(NULL,ncclInvalidArgument,"ncclMeshMemBuffer: NULL argument");
  pthread_mutex_lock(&heap.lock);
  struct span *s=span_of(ptr,1);
  if(s){*buffer=s->buffer;*offset=(size_t)((const unsigned char *)ptr-s->at);}
  pthread_mutex_unlock(&heap.lock);
  return s?ncclSuccess:FAIL(NULL,ncclInvalidArgument,"ncclMeshMemBuffer: %p is not in a live allocation of the window (ncclMemAlloc)",ptr);
}
ncclResult_t ncclMeshMemUse(const void *ptr,size_t bytes,void *event,uint64_t value,int write){
  if(!ptr || !event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshMemUse: NULL argument");
  pthread_mutex_lock(&heap.lock);
  struct span *s=span_of(ptr,1);
  if(!s)s=outside_of(ptr,bytes?bytes:1);
  if(s){if(write)wrote_set(s,event,value,NULL);else read_add(s,event,value,NULL);}
  pthread_mutex_unlock(&heap.lock);
  return s?ncclSuccess:FAIL(NULL,ncclSystemError,"allocation");
}
ncclResult_t ncclMeshMemWaits(const void *ptr,size_t bytes,int write,void **events,uint64_t *values,int capacity,int *count){
  if(!ptr || !count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshMemWaits: NULL argument");
  pthread_mutex_lock(&heap.lock);
  struct span *in=span_of(ptr,1);
  *count=0;
  for(int o=-1;o<(in?0:heap.nout);o++){
    struct span *s=o<0?in:overlap(heap.outside+o,ptr,bytes?bytes:1)?heap.outside+o:NULL;
    for(int i=-1;s && i<(write?s->nread:0);i++){
      struct point p=i<0?s->wrote:s->read[i];
      if(reached(p))continue;
      if(*count<capacity){events[*count]=p.event;values[*count]=p.value;}
      ++*count;
    }
  }
  pthread_mutex_unlock(&heap.lock);
  return ncclSuccess;
}
ncclResult_t ncclMeshMemRelease(void *ptr,void *event,uint64_t value){
  return ptr?span_release(ptr,event,value):ncclSuccess;
}
ncclResult_t ncclMeshMemRecords(ncclMeshMemRecord_t *records,int capacity,int *count){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshMemRecords: count is NULL");
  pthread_mutex_lock(&heap.lock);
  *count=heap.n;
  for(int i=0;records && i<heap.n && i<capacity;i++){
    const struct span *s=heap.spans+i;
    ncclMeshMemRecord_t *r=records+i;
    memset(r,0,sizeof *r);
    r->address=(uint64_t)(uintptr_t)s->at;r->bytes=s->bytes;r->freed=s->freed;
    for(int j=-1;j<s->nread;j++){
      struct point p=j<0?s->wrote:s->read[j];
      if(!p.event)continue;
      r->value[r->points]=p.value;r->reached[r->points]=nccl_mesh_event_value(p.event);r->points++;
    }
  }
  pthread_mutex_unlock(&heap.lock);
  return ncclSuccess;
}

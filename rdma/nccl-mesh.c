#include "nccl.h"
#include "mesh-net.h"
#include "mesh-collective.h"
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
   start at; for each REDUCE chunk a wait on its receive's completion word (and the earlier sends'
   reading its range), the combine kernel reading the piece where it landed and writing the operand in
   place, and, where a later send reads it, a published word; the post-division and the results copied
   out once every completion word is set; the group's completion.  The bridge stores each isend's and
   irecv's end into its completion word (mesh.h), so the GPU waits on the network itself, no library
   thread between.  Each communicator's worker thread starts its part of a group in issue order once the
   gate word is set (or, without a program, the recorded points are reached): a SEND is an isend of the
   operand's piece (once the GPU has published its combine), a COPY an irecv into the operand, a REDUCE an
   irecv into its piece.  Its point-to-point transfers stay in flight while the worker goes on (a send
   completes only once its peer has posted the receive), and the part is done once both are; a part that
   fails sets every completion word of its calls the bridge has not, so its program goes on.  No library
   thread copies bytes on the CPU; what the library copied, sent, received and waited for is counted, a
   call each (ncclMeshGetCounts, ncclMeshGroupCounts). */

enum { CH_COLL, CH_P2P, CHANNELS };
enum { K_SEND=MESH_ALLGATHER+1, K_RECV, K_COUNTED };
#define P2P(k) ((k)->kind==K_SEND || (k)->kind==K_RECV)
/* a call's tallies: counts (also summed for the process), then its part's start and end times and when
   the worker saw the last of its pieces land */
enum { CPU_COPY, GPU_COPY, GPU_KERNELS, HOST_WAITS, INPUT_WAITS, SENT, RECEIVED, GPU_EVENT_WAITS, GPU_WORD_WAITS, HOST_WORD_WAITS, COMMITS,
       WAKEUPS, BUFFERS, COUNTS, STARTED=COUNTS, ENDED, ARRIVED, TALLIES };
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
  /* its place among the calls issued on the communicator since its last agreement, from 1; the hash of
     its whole plan (every rank's steps: what an epoch's move must leave alone for it to stand) */
  uint64_t index,whole;
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
     the GPU publishes */
  struct region in,out,at; unsigned char *operand,*wire;
  /* fresh: the send buffer is read in place (nccl.h: no copy into the operand) */
  int copied,fresh;
  _Atomic uint64_t *words,*combined; uint32_t nwords,ncombined;
  uint64_t *tally;
};
/* An event's value, or a word's in the window (a GPU kernel publishes it, system-coherent): the point after
   which a use of an allocation is done. */
struct point { void *event; uint64_t value; _Atomic uint64_t *word; };
struct mark { struct ncclMeshStream *stream; uint64_t done; };
struct tally { _Atomic int refs; int n; uint64_t counts[][TALLIES]; };
/* A group: its streams' marks; its program's event and last value, and the gate word the workers start at
   (NULL: none); without a program, the recorded points the workers wait for; its window allocation. */
struct launch { pthread_mutex_t lock; pthread_cond_t cond; int items,sync,nmarks,nwaits; ncclResult_t result; struct mark *marks;
  void *event; uint64_t end; _Atomic uint64_t *gate; struct tally *tally; struct point *waits; unsigned char *stage,*placed; struct region own; };
struct flight;
/* A communicator's part of a group: `gpu` where a program waits on its completion words; `bound`, while it
   does, when that wait fails it. */
struct item { struct call *calls; int n; struct launch *launch; struct item *next; uint64_t deadline,bound;
  int gpu,collectives_done,networked; struct flight *flight; ncclResult_t result; };
/* A plan kept for (epoch, call signature): the algorithm chosen, this rank's steps and the whole plan's
   hash; its segments (a rank each) and steps (MESH_COLLECTIVE_STEPS of the ranks) sized with the
   communicator. */
#define PLANS 64
struct plan { uint64_t epoch,elements,whole,*segments; int kind,type,root,force,uneven; uint32_t how,nsteps;
  struct mesh_collective chosen; struct mesh_step *steps; };
struct ncclComm {
  uint64_t key; int rank,nranks;
  /* the link table and rank r's node in it; the snapshot the calls plan on (`epoch` its epoch) and the
     planner's map of it over the ranks; the plans */
  struct mesh_link_table *table; uint32_t *nodes; uint64_t epoch; struct mesh_link_contents *seen;
  struct mesh_link_map map; uint32_t (*pairs)[2]; float (*cost)[2]; struct plan *plans; struct mesh_step *planning;
  /* the worker's: the receive connections a collective used (a rank each), flushed at its end */
  void **used;
  /* the worker's own: a later snapshot, its map, and steps, to judge whether a move revokes a call */
  struct mesh_link_contents *later; struct mesh_link_map moved; uint32_t (*moved_pairs)[2]; float (*moved_cost)[2]; struct mesh_step *scratch;
  /* alive: its connections found alive since the epoch last moved (a session a bridge loses moves it) */
  void *net; struct peer *peers; int alive;
  uint64_t splits;
  struct op_entry *ops; int nops;
  /* its own queue and that queue's event, the value reserved on it; the programs not yet run; its control
     allocation (its first word the failure word a timed-out GPU wait of its calls sets, its second the
     progress word: advance) and Metal buffer */
  void *queue,*event; uint64_t event_value;
  _Atomic int programs;
  _Atomic uint64_t *control; void *control_buffer; uint64_t ticked;
  /* woke: the worker came out of its condition variable (parked) with a part to start; agreeing: an
     agreement is making the connections again (the worker leaves them alone) */
  pthread_t worker; int started,stopping,finalized,parked,woke,agreeing;
  pthread_mutex_t lock; pthread_cond_t cond;
  struct item *head,*tail; int busy;
  struct flight *flights; int nflights,*blocked;
  _Atomic int aborting,broken,async;
  char error[512];
  /* recovery (ncclMeshCommAgree): the calls issued since the last agreement, the index of the first
     that failed (0: none) and its cause; the agreements made (alike on every rank: their connections'
     keys); the slots of the connections closed since, and the event a failed part's memory is held on
     until the bridge has vacated them (`quiet`: the value it is held for) */
  _Atomic uint64_t issued; uint64_t failed,agreements; char cause[256];
  uint64_t *closed; int nclosed,open; void *quiet; uint64_t quiet_value;
};

#define HIDDEN __attribute__((visibility("hidden")))
HIDDEN int nccl_mesh_gpu_attach(char *error,size_t size);
HIDDEN int nccl_mesh_gpu_failed(void);
HIDDEN uint64_t nccl_mesh_wait_bound(void);
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
  int published,void *other,uint64_t at);
HIDDEN void nccl_mesh_program_copy(void *program,void *to,uint64_t dst,void *from,uint64_t src,uint64_t bytes,int received,int published);
HIDDEN void nccl_mesh_program_words(void *program,void *buffer,const uint64_t *at,uint32_t n,void *failures,uint64_t failure,uint64_t ns,
  void *const *touch,int ntouch);
HIDDEN void nccl_mesh_program_publish(void *program,void *buffer,uint64_t at,uint64_t value);
HIDDEN void nccl_mesh_program_counted(void *program,void *to,void *from,void *counts,void *got,void *landed,const uint64_t *a,uint32_t na,uint64_t units);
HIDDEN void nccl_mesh_program_end(void *program);
HIDDEN void nccl_mesh_program_handler(void *program,void (*done)(void *,int),void *argument);
HIDDEN void nccl_mesh_program_commit(void *program,void (*done)(void *,int),void *argument);

/* ---- errors ---- */
static _Thread_local char last_error[512];
static void fail_message(struct ncclComm *c,const char *format,...){
  char text[512];va_list list;va_start(list,format);vsnprintf(text,sizeof text,format,list);va_end(list);
  snprintf(last_error,sizeof last_error,"%s",text);
  if(c){pthread_mutex_lock(&c->lock);snprintf(c->error,sizeof c->error,"%s",text);pthread_mutex_unlock(&c->lock);}
  if(getenv("MESH_NCCL_DEBUG"))fprintf(stderr,"nccl-mesh: %s\n",text);
}
#define FAIL(c,result,...) (fail_message((c),__VA_ARGS__),(ncclResult_t)(result))
static ncclResult_t net_failure(struct ncclComm *c,int result,const char *what){
  int error=mesh_net_error();
  return FAIL(c,result?result:ncclSystemError,"%s: %s (%s)",what,ncclGetErrorString((ncclResult_t)result),strerror(error));
}

static uint64_t now_ns(void){return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);}
static uint64_t deadline_after(void){
  const char *text=getenv("MESH_NCCL_TIMEOUT");double seconds=text?atof(text):0;
  return now_ns()+(uint64_t)((seconds>0?seconds:300)*1e9);
}
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
/* The hash of a collective's whole plan on `map`: the algorithm and every rank's steps. */
static uint64_t whole_plan(const struct mesh_link_map *map,int n,struct mesh_collective chosen,struct mesh_operand operand,struct mesh_step *steps){
  uint64_t h=mix(((uint64_t)chosen.how<<32)^chosen.root);
  for(int r=0;chosen.how!=MESH_UNAVAILABLE && r<n;r++){
    uint32_t count=mesh_collective_plan(map,(uint32_t)r,chosen,operand,steps);
    h=mix(h^count^((uint64_t)r<<32));
    for(uint32_t i=0;i<count;i++)
      h=mix(h^((uint64_t)steps[i].op<<56)^((uint64_t)steps[i].peer<<40)^((uint64_t)steps[i].round<<24)^mix(steps[i].first^(steps[i].piece.elements<<20)));
  }
  return h;
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
  c->seen=mesh_link_contents_new(c->table->nodes);
  c->pairs=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->pairs);
  c->cost=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->cost);
  c->plans=calloc(PLANS,sizeof *c->plans);
  for(int p=0;c->plans && p<PLANS;p++){c->plans[p].segments=calloc(n,sizeof *c->plans[p].segments);c->plans[p].steps=calloc(steps,sizeof *c->plans[p].steps);}
  for(int p=0;c->plans && p<PLANS;p++)if(!c->plans[p].segments || !c->plans[p].steps)return FAIL(c,ncclSystemError,"allocation");
  c->used=calloc(n,sizeof *c->used);
  c->later=mesh_link_contents_new(c->table->nodes);
  c->moved_pairs=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->moved_pairs);
  c->moved_cost=calloc((size_t)c->nranks*(size_t)c->nranks,sizeof *c->moved_cost);
  c->scratch=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *c->scratch);
  c->planning=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *c->planning);
  if(!c->nodes || !c->seen || !c->pairs || !c->cost || !c->plans || !c->later || !c->moved_pairs || !c->moved_cost || !c->scratch || !c->planning ||
     !c->used)
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
  mesh_link_table_map(c->seen,c->nodes,(uint32_t)c->nranks,&c->map,c->pairs,c->cost);
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
      if(keep && c->nclosed<c->nranks*CHANNELS*2)c->closed[c->nclosed++]=mesh_net_slot(*end[e]);
      if(e)mesh_net_close_recv(*end[e]);else mesh_net_close_send(*end[e]);
      *end[e]=NULL;
    }
  }
  if(peers==c->peers)c->open=0;
}
/* Both channels each way with every rank `want` marks, into `peers`, on whichever of the bridge's
   links reaches it: a listen on every link for each such rank's connections (keys both ends derive
   from the clique key `key`), a connect on every link to each such rank's; the one its bridge accepts
   is the link.  The probing connects that met no listen are withdrawn with their context. */
static ncclResult_t connect_ranks(struct ncclComm *c,uint64_t key,const unsigned char *want,struct peer *peers,uint64_t deadline){
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
    if(!done && !status){
      if(now_ns()>deadline){
        int missing=-1;
        for(int p=0;p<c->nranks && missing<0;p++)if(want[p])
          for(int ch=0;ch<CHANNELS;ch++)if(!peers[p].send[ch] || !peers[p].recv[ch])missing=p;
        status=FAIL(c,ncclTimeout,"rank %d: no connection with rank %d by the deadline (MESH_NCCL_TIMEOUT): no bridge link reaches it, or it has not joined",c->rank,missing);
      } else usleep(200);
    }
  }
  for(size_t i=0;listens && i<count;i++)if(listens[i])mesh_net_close_listen(listens[i]);
  free(listens);free(targets);
  mesh_net_finalize(probe);
  return status;
}
/* The communicator's connections: with every rank the map links this one to. */
static ncclResult_t connect_peers(struct ncclComm *c){
  unsigned char *want=calloc((size_t)c->nranks,1);
  if(!want)return FAIL(c,ncclSystemError,"allocation");
  for(int p=0;p<c->nranks;p++)want[p]=(unsigned char)linked(&c->map,c->rank,p);
  ncclResult_t status=connect_ranks(c,c->key,want,c->peers,deadline_after());
  free(want);
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

static int reached(struct point p){
  if(p.word)return atomic_load_explicit(p.word,memory_order_acquire)>=p.value;
  return !p.event || nccl_mesh_event_value(p.event)>=p.value;
}
static void point_drop(struct point *p){if(p->event)nccl_mesh_release(p->event);p->event=NULL;p->value=0;p->word=NULL;}
static void point_set(struct point *p,void *event,uint64_t value){
  if(event)nccl_mesh_retain(event);
  point_drop(p);
  p->event=event;p->value=value;
}
/* A reader's point joins the record's: the same event's later value replaces its earlier one, the
   points already reached make room; with none, the host waits for the oldest (counted). */
static void read_add(struct span *s,void *event,uint64_t value,uint64_t *tally){
  for(int i=0;i<s->nread;i++)if(s->read[i].event==event){if(value>s->read[i].value)s->read[i].value=value;return;}
  for(int i=0;i<s->nread;)if(reached(s->read[i])){point_drop(s->read+i);s->read[i]=s->read[--s->nread];}else i++;
  if(s->nread==READS){
    if(tally)tally[HOST_WAITS]++;
    atomic_fetch_add_explicit(&totals[HOST_WAITS],1,memory_order_relaxed);
    for(uint64_t deadline=deadline_after();!reached(s->read[0]) && !nccl_mesh_gpu_failed() && now_ns()<deadline;)sched_yield();
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
   allocation's points (counted), or, `owned` (ncclMeshMemAllocBuffer: its buffer made with `options`,
   the caller's), fails at once. */
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
  for(uint64_t deadline=deadline_after();;){
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
    if(owned || !freed || nccl_mesh_gpu_failed() || now_ns()>deadline){
      status=FAIL(NULL,ncclSystemError,"%zu bytes of the bridge's registered window: %s",bytes,
                  freed?"its freed allocations' uses not done by the deadline (MESH_NCCL_TIMEOUT)":"no room (the bridge's -A/-W window)");
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
/* The longest the GPU waits on the network: a part whose program waits on its completion words fails as
   a value once the network has given the GPU nothing it waits for in this long, and the worker sets
   every word of the part the bridge has not, so the program and its stream's later work go on; the
   GPU's own wait is bounded too, by this long of the host's clock without progress (advance, tick) and
   a count of polls (nccl_mesh_wait_bound), and a wait that runs out sets the communicator's failure word
   (a failed call, program_ran).  Metal ends a command
   buffer that waits past its watchdog, and then refuses the process's later submissions
   (kIOGPUCommandBufferCallbackErrorTimeout, then ...SubmissionsIgnored: the M5's own bridge stopped for
   5 s, metal-microbench output_data/epoch-20260929/replay-local-stop). */
#define GPU_WAIT_NS UINT64_C(1000000000)
struct pending { size_t lo,hi; void *request; };
static void progress(struct ncclComm *c);
static int stopped(struct ncclComm *c,const struct item *it){
  const uint64_t now=now_ns();
  return atomic_load(&c->aborting) || atomic_load(&c->broken) || now>it->deadline || (it->bound && now>it->bound);
}
/* Why a part stopped; a revoked communicator's parts fail without a message of their own, so the
   communicator's error stays its first failure's. */
static ncclResult_t stop_reason(struct ncclComm *c,const struct item *it,const char *what){
  if(atomic_load(&c->aborting))return FAIL(c,ncclInvalidUsage,"%s: the communicator was aborted",what);
  if(atomic_load(&c->broken))return ncclRemoteError;
  if(it->bound && now_ns()>it->bound)
    return FAIL(c,ncclRemoteError,"%s: the network gave the GPU nothing it waits for in %.0f s (a bridge stalled, or a peer has not "
                "joined the call); the call fails before Metal's command-buffer watchdog",what,GPU_WAIT_NS/1e9);
  return FAIL(c,ncclTimeout,"%s: not done by the deadline (MESH_NCCL_TIMEOUT)",what);
}
/* The network gave the GPU what it waits for (a piece landed, a combine published): the bound starts again. */
static void rebound(struct item *it){if(it->gpu)it->bound=now_ns()+GPU_WAIT_NS;}
/* The network moved for the communicator (a part started, a request done): its progress word advances, which
   starts the bound of every GPU wait on its words again (its programs may wait on parts the worker has not
   reached yet: it takes them in issue order).  The bound is the host's clock, which the worker writes as it
   runs (tick): a GPU wait fails once GPU_WAIT_NS of it pass with no progress. */
static void advance(struct ncclComm *c){atomic_fetch_add_explicit(c->control+1,1,memory_order_release);}
static void tick(struct ncclComm *c){
  const uint64_t now=now_ns();
  if(now-c->ticked<1000000)return;  /* a millisecond's resolution: the bound is a second */
  c->ticked=now;atomic_store_explicit(c->control+2,now,memory_order_relaxed);
}
/* Reaps every pending isend that is done (its ring slot free again). */
static ncclResult_t reap(struct ncclComm *c,struct pending *sends,int *count){
  for(int i=0;i<*count;){
    int done=0,size=0,result=mesh_net_test(sends[i].request,&done,&size);
    if(result)return net_failure(c,result,"an isend");
    if(done){sends[i]=sends[--*count];advance(c);continue;}
    i++;
  }
  return ncclSuccess;
}
static ncclResult_t await(struct ncclComm *c,void *request,size_t bytes,struct pending *sends,int *count,struct item *it,const char *what){
  for(;;){
    int done=0,size=0,result=mesh_net_test(request,&done,&size);
    if(result)return net_failure(c,result,what);
    if(done){
      advance(c);
      if(bytes<=INT32_MAX && (size_t)size!=bytes)return FAIL(c,ncclInvalidUsage,"%s: %d bytes arrived, %zu expected (the peer's count or datatype differs)",what,size,bytes);
      return ncclSuccess;
    }
    if(sends){ncclResult_t r=reap(c,sends,count);if(r)return r;}
    if(stopped(c,it))return stop_reason(c,it,what);
    progress(c);
    sched_yield();
  }
}
/* An isend or irecv posted, its end stored into `word` by the bridge, waiting while its connection's
   request ring is full. */
static ncclResult_t post(struct ncclComm *c,int send,void *comm,void *data,size_t bytes,_Atomic uint64_t *word,
  struct pending *sends,int *count,struct item *it,void **request){
  for(*request=NULL;;){
    int result=send?mesh_net_isend_word(comm,data,bytes,window.mh,(uint64_t *)word,request):
      mesh_net_irecv_word(comm,data,bytes,window.mh,(uint64_t *)word,request);
    if(result)return net_failure(c,result,send?"mesh_net_isend":"mesh_net_irecv");
    if(*request)return ncclSuccess;
    if(sends){ncclResult_t r=reap(c,sends,count);if(r)return r;}
    if(stopped(c,it))return stop_reason(c,it,send?"an isend":"an irecv");
    progress(c);
    sched_yield();
  }
}

/* The worker waits for a word the program publishes (the library's own GPU work, which waits on no network
   meanwhile: the bound starts again once it is done), the point-to-point transfers moving meanwhile. */
static ncclResult_t word_wait(struct ncclComm *c,struct call *k,_Atomic uint64_t *word,struct item *it,const char *what){
  count(k,HOST_WORD_WAITS,1);
  it->bound=0;
  while(!atomic_load_explicit(word,memory_order_acquire)){
    if(nccl_mesh_gpu_failed())return FAIL(c,ncclUnhandledCudaError,"%s: a GPU program failed",what);
    if(stopped(c,it))return stop_reason(c,it,what);
    progress(c);
    sched_yield();
  }
  rebound(it);
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
   receive and the earlier sends of its range on their words itself. */
struct combined { size_t lo,hi; _Atomic uint64_t *word; };
static int written(const struct call *k,uint32_t s,size_t lo,size_t hi);
static const unsigned char *given(const struct call *k,size_t lo);
static ncclResult_t run_collective(struct ncclComm *c,struct call *k,struct item *it){
  const size_t e=type_bytes[k->type];
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
            status=word_wait(c,k,done[d].word,it,"a combine");
        void *request=NULL;
        unsigned char *from=k->fresh && !written(k,i,lo,lo+length)?(unsigned char *)given(k,lo):k->operand+lo;
        if(!status && !(status=post(c,1,p->send[CH_COLL],from,length,k->words+w+j,sends,&count_,it,&request))){
          sends[count_++]=(struct pending){lo,lo+length,request};
          count(k,SENT,length);
        }
      }
      w+=n;
      continue;
    }
    int known=0;
    for(int u=0;u<nused;u++)known|=used[u]==p->recv[CH_COLL];
    if(!known)used[nused++]=p->recv[CH_COLL];
    const size_t lo=s->first*e,length=s->piece.elements*e;
    /* a piece of the operand is written only once no isend still reads it */
    for(int j=0;j<count_ && !status && s->op==MESH_STEP_COPY;)
      if(sends[j].lo<lo+length && lo<sends[j].hi){status=await(c,sends[j].request,SIZE_MAX,NULL,NULL,it,"an isend");sends[j]=sends[--count_];}
      else j++;
    if(status)break;
    uint32_t posted=0;
    for(uint32_t j=0;j<n && !status;j++){
      for(;posted<n && posted<j+AHEAD && !status;posted++){
        const size_t at=chunk_first(s->piece.elements,e,posted)*e,bytes=chunk_count(s->piece.elements,e,posted)*e;
        unsigned char *into=s->op==MESH_STEP_COPY?k->operand+lo+at:k->pieces[i].at+at;
        status=post(c,0,p->recv[CH_COLL],into,bytes,k->words+w+posted,sends,&count_,it,requests+posted);
      }
      if(status)break;
      const size_t at=lo+chunk_first(s->piece.elements,e,j)*e,bytes=chunk_count(s->piece.elements,e,j)*e;
      if((status=await(c,requests[j],bytes,sends,&count_,it,s->op==MESH_STEP_COPY?"a COPY step's irecv":"a REDUCE step's irecv")))break;
      count(k,RECEIVED,bytes);
      if(s->op!=MESH_STEP_REDUCE)continue;
      if(k->tally)k->tally[ARRIVED]=now_ns();
      rebound(it);
      if(k->combined)done[ndone++]=(struct combined){at,at+bytes,k->combined+r+j};
    }
    w+=n;
    if(s->op==MESH_STEP_REDUCE && k->combined)r+=n;
  }
  while(count_ && !status){status=await(c,sends[count_-1].request,SIZE_MAX,NULL,NULL,it,"an isend");count_--;}
  for(int u=0;u<nused && !status;u++){
    void *flush=NULL;int result=mesh_net_iflush(used[u],1,NULL,NULL,NULL,&flush);
    if(result)status=net_failure(c,result,"mesh_net_iflush");
    else if(flush)status=await(c,flush,SIZE_MAX,NULL,NULL,it,"an iflush");
  }
  free(sends);free(done);free(requests);
  return status;
}

/* ---- a counted all-to-all (nccl.h ncclMeshAlltoAllCounted) ----
   The worker's side: each rank's segment of the counts exchanged with it first (its completion words the
   call's first 2(n - 1): sends, then receives), each rank's segment for this rank written to the caller's
   host array in rank order (this rank's own from its counts) and the caller's word set; then the rows,
   as many to each rank as the sum of its segment and from each as the sum of the one it sent, packed in
   rank order (the next 2(n - 1) words: sends, then receives); the GPU copies this rank's own rows (the
   program).  A segment or a row count of 0 moves nothing: its word is set here. */
static ncclResult_t run_counted(struct ncclComm *c,struct call *k,struct item *it){
  const int n=c->nranks,me=c->rank,peers=n-1;
  const size_t row=k->row*type_bytes[k->type];
  const uint64_t *sseg=k->segments,*rseg=k->segments+n;
  int64_t *got=(int64_t *)k->wire;
  void **requests=calloc((size_t)(4*n),sizeof *requests);
  size_t *bytes=calloc((size_t)(4*n),sizeof *bytes);
  uint64_t *rows=calloc((size_t)(2*n),sizeof *rows);
  ncclResult_t status=requests && bytes && rows?ncclSuccess:FAIL(c,ncclSystemError,"allocation");
  for(int phase=0;phase<2 && !status;phase++){
    uint64_t so=0,ro=0;
    for(int q=0,i=0;q<n && !status;q++){
      const uint64_t out=phase?rows[q]:sseg[q],in=phase?rows[n+q]:rseg[q];
      if(q==me){so+=out;ro+=in;continue;}
      struct peer *p=c->peers+q;
      const int w=2*phase*peers+i;
      unsigned char *into=phase?(unsigned char *)k->recv+ro*row:(unsigned char *)(got+ro);
      const unsigned char *from=phase?(const unsigned char *)k->send+so*row:(const unsigned char *)(k->counted+so);
      bytes[w]=out*(phase?row:8);bytes[w+peers]=in*(phase?row:8);
      if(bytes[w+peers])status=post(c,0,p->recv[CH_COLL],into,bytes[w+peers],k->words+w+peers,NULL,NULL,it,requests+w+peers);
      else atomic_store_explicit(k->words+w+peers,1,memory_order_release);
      if(!status && bytes[w])status=post(c,1,p->send[CH_COLL],(void *)from,bytes[w],k->words+w,NULL,NULL,it,requests+w);
      else if(!status)atomic_store_explicit(k->words+w,1,memory_order_release);
      so+=out;ro+=in;i++;
    }
    for(int j=2*phase*peers;j<2*(phase+1)*peers && !status;j++)if(requests[j]){
      status=await(c,requests[j],j<(2*phase+1)*peers?SIZE_MAX:bytes[j],NULL,NULL,it,phase?"a counted all-to-all's rows":"a counted all-to-all's counts");
      count(k,j<(2*phase+1)*peers?SENT:RECEIVED,bytes[j]);
      rebound(it);
    }
    if(status || phase)break;
    /* the counts on the host, and each rank's rows: sent (its segment's sum), received (the sum of its segment for this rank) */
    so=0;ro=0;
    uint64_t total=0;
    for(int q=0;q<n;q++){
      const int64_t *mine=q==me?k->counted+so:got+ro;
      memcpy(k->received+ro,mine,rseg[q]*8);
      for(uint64_t x=0;x<sseg[q];x++)rows[q]+=(uint64_t)k->counted[so+x];
      for(uint64_t x=0;x<rseg[q];x++)rows[n+q]+=(uint64_t)mine[x];
      total+=rows[n+q];so+=sseg[q];ro+=rseg[q];
    }
    atomic_store_explicit((_Atomic uint64_t *)k->arrived,1,memory_order_release);
    uint64_t sent=0;
    for(int q=0;q<n;q++)sent+=rows[q];
    count(k,GPU_COPY,rows[me]*row);  /* this rank's own rows: the GPU's counted copy */
    if(total>k->capacity || sent>k->rows)
      status=FAIL(c,ncclInvalidUsage,"a counted all-to-all: its counts send %llu rows of %llu and receive %llu, capacity %llu",(unsigned long long)sent,
                  (unsigned long long)k->rows,(unsigned long long)total,(unsigned long long)k->capacity);
  }
  if(status && k->arrived){uint64_t none=0;atomic_compare_exchange_strong((_Atomic uint64_t *)k->arrived,&none,2);}
  free(requests);free(bytes);free(rows);
  return status;
}

/* ---- one communicator's part of a group ---- */
struct transfer { int send; struct peer *peer; unsigned char *window; size_t bytes; void *request; int state; struct call *call; };
/* A part's point-to-point transfers, in flight: posted in order on each connection (across every
   flight, oldest first) as far as its request ring takes them, tested, and retired together. */
struct flight { struct item *it; struct transfer *t; int n,remaining; ncclResult_t status; struct flight *next; };

static void finish(struct ncclComm *c,struct item *it,ncclResult_t result);
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
  tick(c);
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
          x->state=2;f->remaining--;advance(c);
          count(x->call,x->send?SENT:RECEIVED,x->bytes);
          if(!x->send && x->bytes<=INT32_MAX && (size_t)size!=x->bytes)
            f->status=FAIL(c,ncclInvalidUsage,"a receive: %d bytes arrived, %zu expected (the peer's count or datatype differs)",size,x->bytes);
        }
      }
    }
    if(!f->status && f->remaining && stopped(c,f->it))f->status=stop_reason(c,f->it,"a point-to-point call");
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
  rebound(it);advance(c);
  if(c->woke && it->n){count(it->calls,WAKEUPS,1);c->woke=0;}
  for(int i=0;i<it->n;i++)if(it->calls[i].tally && !it->calls[i].tally[STARTED])it->calls[i].tally[STARTED]=now;
  /* what it started at: the gate word, or the recorded points (events, words) */
  const struct launch *l=it->launch;
  if(it->n && l->gate)count(it->calls,HOST_WORD_WAITS,1);
  for(int w=0;it->n && !l->gate && w<l->nwaits;w++)count(it->calls,l->waits[w].word?HOST_WORD_WAITS:INPUT_WAITS,1);
  int transfers=0;
  for(int i=0;i<it->n && !status;i++){
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
    status=k->kind==K_COUNTED?run_counted(c,k,it):run_collective(c,k,it);
    it->networked=1;
  }
  for(int i=0;status && i<it->n;i++)if(it->calls[i].arrived){uint64_t none=0;atomic_compare_exchange_strong((_Atomic uint64_t *)it->calls[i].arrived,&none,2);}
  /* every completion word the program waits on, whatever happened (a flight's once it retires) */
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
  if(l->stage)span_release(l->stage,NULL,0);
  for(int w=0;w<l->nwaits;w++)point_drop(l->waits+w);
  tally_release(l->tally);free(l->waits);free(l->marks);free(l);
}
/* A part whose calls planned on an epoch the link map has since left fails as a value, whatever its
   transfers did, where the move touches a call's plan: a collective whose whole plan (the algorithm the
   selection takes on the new contents and every rank's steps: the links and nodes it uses, and the
   costs that choose it) is not the one it ran, a point-to-point call whose link is no longer stated and
   up.  A move that leaves every plan as it was (a node or link the calls do not use, a cost that does not
   change their choice) revokes nothing.  Every rank judges from its own table, so alike where the
   tables are.  The next call plans on the new contents. */
static ncclResult_t revoked(struct ncclComm *c,struct item *it,ncclResult_t result){
  const uint64_t now=atomic_load_explicit(&c->table->epoch,memory_order_acquire);
  int moved=0;
  for(int i=0;i<it->n;i++)moved|=it->calls[i].epoch!=now;
  if(result || !moved)return result;
  const uint64_t later=mesh_link_table_read(c->table,c->later);
  mesh_link_table_map(c->later,c->nodes,(uint32_t)c->nranks,&c->moved,c->moved_pairs,c->moved_cost);
  for(int i=0;i<it->n;i++){
    const struct call *k=it->calls+i;
    int touched;
    if(k->epoch==later)continue;
    if(k->kind==K_COUNTED){touched=0;for(int q=0;q<c->nranks;q++)touched|=q!=c->rank && !linked(&c->moved,c->rank,q);}
    else if(k->kind>=K_SEND)touched=k->peer!=c->rank && !linked(&c->moved,c->rank,k->peer);
    else if(c->nranks==1)touched=0;
    else{
      struct mesh_operand operand={(uint32_t)k->type,(uint32_t)type_bytes[k->type],k->elements};
      touched=whole_plan(&c->moved,c->nranks,choose(&c->moved,k,operand),operand,c->scratch)!=k->whole;
    }
    if(touched)
      return FAIL(c,ncclRemoteError,"revoked: the link map moved from epoch %llu to %llu during call %llu, and the move touches its plan "
                  "(a link or node it uses, or costs that choose another)",(unsigned long long)k->epoch,(unsigned long long)later,
                  (unsigned long long)k->index);
  }
  return result;
}
/* The first failure since the last agreement: its call's index (the least) and its cause. */
static void failed_at(struct ncclComm *c,uint64_t index,const char *cause){
  pthread_mutex_lock(&c->lock);
  if(index && (!c->failed || index<c->failed))c->failed=index;
  if(!c->cause[0])snprintf(c->cause,sizeof c->cause,"%s",cause);
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
static void finish(struct ncclComm *c,struct item *it,ncclResult_t result){
  struct launch *l=it->launch;
  result=revoked(c,it,result);
  if(result){
    uint64_t first=0;
    for(int i=0;i<it->n;i++)if(!first || it->calls[i].index<first)first=it->calls[i].index;
    failed_at(c,first,c->error);
    if(it->networked || (result!=ncclInvalidArgument && result!=ncclInvalidUsage)){comm_revoke(c);hold(c,it);}
  }
  const uint64_t now=now_ns();
  for(int i=0;i<it->n;i++)if(it->calls[i].tally)it->calls[i].tally[ENDED]=now;
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
    if(!c->head && !c->flights && !c->stopping && !closing && now_ns()-busy>WORKER_SPIN_NS){
      c->parked=1;
      while(!c->head && !c->flights && !c->stopping && !(atomic_load(&c->broken) && c->open && !c->agreeing))pthread_cond_wait(&c->cond,&c->lock);
      c->parked=0;c->woke=c->head!=NULL;
    }
    if(atomic_load(&c->broken) && c->open && !c->agreeing)close_peers(c,c->peers,1);
    struct item *it=c->head;
    int flying=c->flights!=NULL,room=c->nflights<FLIGHTS;
    if(!it && !flying && c->stopping){pthread_mutex_unlock(&c->lock);break;}
    pthread_mutex_unlock(&c->lock);
    if(!it && !flying){tick(c);sched_yield();continue;}
    busy=now_ns();
    progress(c);
    if(it && room){
      int go=ready(it);
      if(!go && nccl_mesh_gpu_failed()){go=1;it->result=FAIL(c,ncclUnhandledCudaError,"waiting for the group's gate: a GPU program failed");}
      if(!go && stopped(c,it)){go=1;it->result=stop_reason(c,it,"waiting for the group's gate");}
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
    tick(c);
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
  pthread_mutex_init(&c->lock,NULL);pthread_cond_init(&c->cond,NULL);
  return c;
}
static void comm_free(struct ncclComm *c){
  if(c->started){
    pthread_mutex_lock(&c->lock);c->stopping=1;pthread_cond_broadcast(&c->cond);pthread_mutex_unlock(&c->lock);
    pthread_join(c->worker,NULL);
  }
  /* the programs that wait for its events run first */
  for(uint64_t deadline=deadline_after();atomic_load(&c->programs) && !nccl_mesh_gpu_failed() && now_ns()<deadline;)usleep(100);
  close_peers(c,c->peers,0);
  if(c->net)mesh_net_finalize(c->net);
  for(int p=0;c->plans && p<PLANS;p++){free(c->plans[p].segments);free(c->plans[p].steps);}
  free(c->nodes);free(c->seen);free(c->pairs);free(c->cost);free(c->plans);free(c->closed);free(c->used);
  free(c->later);free(c->moved_pairs);free(c->moved_cost);free(c->scratch);free(c->planning);
  if(c->control_buffer)nccl_mesh_release(c->control_buffer);
  if(c->control)span_release((void *)c->control,NULL,0);
  if(c->quiet)nccl_mesh_release(c->quiet);
  if(c->event)nccl_mesh_release(c->event);
  if(c->queue)nccl_mesh_release(c->queue);
  pthread_mutex_destroy(&c->lock);pthread_cond_destroy(&c->cond);
  free(c->peers);free(c->blocked);free(c->ops);free(c);
}
/* Attached to the bridge, its GPU queue, events and control allocation made, connected to every linked
   rank, its worker started. */
static ncclResult_t comm_start(struct ncclComm *c){
  ncclResult_t status=global_attach();
  if(!status){
    c->queue=nccl_mesh_queue_create();
    if(c->queue){c->event=nccl_mesh_event_create(c->queue);c->quiet=nccl_mesh_event_create(c->queue);}
    if(!c->event || !c->quiet)status=FAIL(c,ncclUnhandledCudaError,"the communicator's Metal queue and events");
  }
  unsigned char *control=NULL;
  if(!status && !(status=span_alloc(64,&control,NULL))){
    memset(control,0,64);
    c->control=(_Atomic uint64_t *)control;
    tick(c);
    pthread_mutex_lock(&heap.lock);
    c->control_buffer=span_of(control,64)->buffer;nccl_mesh_retain(c->control_buffer);
    pthread_mutex_unlock(&heap.lock);
  }
  c->peers=calloc((size_t)c->nranks,sizeof *c->peers);
  c->blocked=calloc((size_t)c->nranks*2,sizeof *c->blocked);
  c->closed=calloc((size_t)c->nranks*CHANNELS*2,sizeof *c->closed);
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
ncclResult_t ncclGetUniqueId(ncclUniqueId *id){
  if(!id)return FAIL(NULL,ncclInvalidArgument,"ncclGetUniqueId: uniqueId is NULL");
  struct uid made={UID_MAGIC,NCCL_VERSION_CODE,0};
  while(!made.key)made.key=((uint64_t)arc4random()<<32)|arc4random();
  memset(id,0,sizeof *id);memcpy(id->internal,&made,sizeof made);
  return ncclSuccess;
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
  case ncclTimeout: return "operation timed out (MESH_NCCL_TIMEOUT)";
  default: return "unknown result code";
  }
}
const char *ncclGetLastError(ncclComm_t comm){
  return comm && comm->error[0]?comm->error:last_error;
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
/* The snapshot a call plans on: the table read again once its epoch has moved, and the map made again.
   A revoked communicator makes no call; a connection the bridge lost (its link's session ended, which
   moves the epoch: the bridge marks the link down), or a rank the map links with none, fails the call
   (and revokes the communicator: ncclGroupEnd), and ncclMeshCommAgree makes the connections again.  The
   connections are checked once the epoch moves and after an agreement, not every call: a call's hot path
   is the epoch word; a connection lost with no move fails its call on the worker (revoking) instead. */
static ncclResult_t refresh(struct ncclComm *c){
  if(atomic_load(&c->broken)){
    char cause[256];uint64_t failed;
    pthread_mutex_lock(&c->lock);snprintf(cause,sizeof cause,"%s",c->cause);failed=c->failed;pthread_mutex_unlock(&c->lock);
    return FAIL(c,ncclRemoteError,"revoked: call %llu since the last agreement failed (%s); the communicator makes no call before "
                "ncclMeshCommAgree",(unsigned long long)failed,cause);
  }
  if(atomic_load_explicit(&c->table->epoch,memory_order_acquire)!=c->epoch){
    c->epoch=mesh_link_table_read(c->table,c->seen);
    mesh_link_table_map(c->seen,c->nodes,(uint32_t)c->nranks,&c->map,c->pairs,c->cost);
    c->alive=0;
  }
  for(int p=0;!c->alive && p<c->nranks;p++)if(linked(&c->map,c->rank,p))
    for(int ch=0;ch<CHANNELS;ch++)if(!mesh_net_alive(c->peers[p].send[ch]) || !mesh_net_alive(c->peers[p].recv[ch]))
      return FAIL(c,ncclRemoteError,"the connection with rank %d is %s (ncclMeshCommAgree makes it again)",p,
                  c->peers[p].send[ch] && c->peers[p].recv[ch]?"lost: its link's session ended":"not made: the map did not link it then");
  c->alive=1;
  return ncclSuccess;
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
  for(int q=0;k->kind==K_COUNTED && q<c->nranks;q++)if(q!=c->rank && (!linked(&c->map,c->rank,q) || !c->peers[q].send[CH_COLL])){
    char down[256];down_links(c,down,sizeof down);
    return FAIL(c,*down?ncclRemoteError:ncclInvalidUsage,"a counted all-to-all: no link joins rank %d to rank %d at the link map's epoch %llu%s%s",q,
                c->rank,(unsigned long long)c->epoch,*down?"; down: ":"",down);
  }
  if(k->kind==K_COUNTED)return ncclSuccess;
  if(k->kind>=K_SEND){
    if(k->peer!=c->rank && (!linked(&c->map,c->rank,k->peer) || !c->peers[k->peer].send[CH_P2P])){
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
    struct mesh_collective chosen=choose(&c->map,k,operand);
    if(chosen.how==MESH_UNAVAILABLE){
      char down[256];down_links(c,down,sizeof down);
      return FAIL(c,*down?ncclRemoteError:ncclInvalidUsage,"no algorithm of the selection %#x carries this collective (%d) of %llu elements on the "
        "link map's epoch %llu%s%s",k->how,k->kind,(unsigned long long)k->elements,(unsigned long long)c->epoch,*down?"; down: ":"",down);
    }
    uint32_t nsteps=mesh_collective_plan(&c->map,(uint32_t)c->rank,chosen,operand,p->steps);
    if(!nsteps)return FAIL(c,ncclInternalError,"the planner gave rank %d no steps",c->rank);
    p->whole=whole_plan(&c->map,c->nranks,chosen,operand,c->planning);
    p->nsteps=nsteps;p->epoch=c->epoch;p->kind=k->kind;p->how=k->how;p->root=k->root;p->type=(int)k->type;p->force=k->force;
    p->elements=k->elements;p->uneven=k->segments!=NULL;p->chosen=chosen;
    if(k->segments)memcpy(p->segments,k->segments,(size_t)c->nranks*sizeof *k->segments);
  }
  k->chosen=p->chosen;k->chosen.segments=k->segments;k->whole=p->whole;
  k->steps=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *k->steps);
  k->pieces=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *k->pieces);
  if(!k->steps || !k->pieces)return FAIL(c,ncclSystemError,"allocation");
  k->nsteps=p->nsteps;memcpy(k->steps,p->steps,p->nsteps*sizeof *k->steps);
  for(uint32_t i=0;i<k->nsteps;i++)if(!c->peers[k->steps[i].peer].send[CH_COLL])
    return FAIL(c,ncclInvalidUsage,"the plan uses rank %u, which no connection reaches",k->steps[i].peer);
  return ncclSuccess;
}

static void calls_free(struct call *calls,int n){
  for(int i=0;i<n;i++){free(calls[i].steps);free(calls[i].pieces);free(calls[i].segments);}
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

/* A recorded point the group waits for (the same event's or word's once, at its latest value). */
static void wait_add(struct launch *l,struct point p){
  if(reached(p))return;
  for(int w=0;w<l->nwaits;w++)
    if(p.word?l->waits[w].word==p.word:!l->waits[w].word && l->waits[w].event==p.event){if(p.value>l->waits[w].value)l->waits[w].value=p.value;return;}
  if(p.word){l->waits[l->nwaits++]=(struct point){.word=p.word,.value=p.value};return;}
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
  if(l->stage)span_release(l->stage,l->event,l->end);
  l->stage=NULL;
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
   a program (`words`), the completion words: each call's requests' and combines' and the gate's, zeroed. */
static ncclResult_t place(struct call *calls,int n,struct launch *l,int words){
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
      if(pass){k->words=(_Atomic uint64_t *)(l->stage+used);k->nwords=w;k->combined=c?(_Atomic uint64_t *)(l->stage+used+8*w):NULL;k->ncombined=c;}
      used+=8*((size_t)w+c);
    }
    if(words){if(pass)l->gate=(_Atomic uint64_t *)(l->stage+used);used+=8;}
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
  if(words)memset(l->stage+first,0,end-first);
  return ncclSuccess;
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
   communicator whose words it waits on (with its first call's index in the group, for a timed-out wait's
   failure); a kept program's commands (below) until then too. */
struct recording;
struct ran { int ncomms,nevents; struct ncclComm **comms; uint64_t *first; void **events; struct recording *kept; };
/* A kept program (a deferred stream's, nccl.h ncclMeshStreamDefer): its commands, each object they name
   retained, encoded later into a command buffer the caller hands over. */
enum { R_WAIT, R_SIGNAL, R_KERNEL, R_COPY, R_WORDS, R_PUBLISH, R_COUNTED };
struct recorded { int kind,which,type,op,nranks,published; void *a,*b,*c,*d,*e; uint64_t x,y,z,n,scalar; uint64_t *list; void *touch[4]; };
struct recording { struct recording *next; int n,capacity,failed; struct recorded *ops; struct ran *ran; };
/* Where a program's commands go: a command buffer, or a recording. */
struct sink { void *program; struct recording *kept; };
static void keep(struct sink *s,struct recorded r){
  struct recording *k=s->kept;
  if(k->n==k->capacity){
    int capacity=k->capacity?2*k->capacity:32;
    struct recorded *grown=realloc(k->ops,(size_t)capacity*sizeof *grown);
    if(!grown){k->failed=1;free(r.list);return;}
    k->ops=grown;k->capacity=capacity;
  }
  if(r.a)nccl_mesh_retain(r.a);
  if(r.b)nccl_mesh_retain(r.b);
  if(r.c)nccl_mesh_retain(r.c);
  if(r.d)nccl_mesh_retain(r.d);
  if(r.e)nccl_mesh_retain(r.e);
  for(int t=0;t<4;t++)if(r.touch[t])nccl_mesh_retain(r.touch[t]);
  k->ops[k->n++]=r;
}
static void sink_wait(struct sink *s,struct call *k,void *event,uint64_t value){
  if(s->kept)keep(s,(struct recorded){.kind=R_WAIT,.a=event,.x=value});else nccl_mesh_program_wait(s->program,event,value);
  count(k,GPU_EVENT_WAITS,1);
}
static void sink_signal(struct sink *s,void *event,uint64_t value){
  if(s->kept)keep(s,(struct recorded){.kind=R_SIGNAL,.a=event,.x=value});else nccl_mesh_program_signal(s->program,event,value);
}
/* A wait for words of the window allocation `r` to reach their values (`list`: n pairs of an offset and a
   value), a timed-out wait setting the failure word of k's communicator; the buffers k's requests read and
   write declared used (with `r`'s), so later work on them follows the wait. */
static void sink_words(struct sink *s,struct call *k,struct region r,const uint64_t *list,uint32_t n){
  if(!n)return;
  struct ncclComm *c=k->comm;
  void *touch[4]={0};int m=0;
  void *const each[4]={r.buffer,k->at.buffer,k->in.buffer,k->out.buffer};
  for(int i=0;i<4;i++){int seen=!each[i];for(int j=0;j<m;j++)seen|=touch[j]==each[i];if(!seen)touch[m++]=each[i];}
  if(s->kept){
    uint64_t *copy=malloc(2*n*sizeof *copy);
    if(!copy){s->kept->failed=1;return;}
    memcpy(copy,list,2*n*sizeof *copy);
    struct recorded rec={.kind=R_WORDS,.a=r.buffer,.b=c->control_buffer,.n=n,.list=copy};
    memcpy(rec.touch,touch,sizeof touch);
    keep(s,rec);
  } else nccl_mesh_program_words(s->program,r.buffer,list,n,c->control_buffer,0,GPU_WAIT_NS,touch,m);
  count(k,GPU_WORD_WAITS,n);
}
/* A word of the window allocation `r` set to `value` once the program's dispatches before it are done. */
static void sink_publish(struct sink *s,struct region r,const void *word,uint64_t value){
  if(s->kept)keep(s,(struct recorded){.kind=R_PUBLISH,.a=r.buffer,.x=off(r,word),.y=value});
  else nccl_mesh_program_publish(s->program,r.buffer,off(r,word),value);
}
/* The kept commands encoded into `program`, their objects released once it has run (program_ran). */
static void play(struct recording *k,void *program){
  for(int i=0;i<k->n;i++){
    const struct recorded *r=k->ops+i;
    if(r->kind==R_WAIT)nccl_mesh_program_wait(program,r->a,r->x);
    else if(r->kind==R_SIGNAL)nccl_mesh_program_signal(program,r->a,r->x);
    else if(r->kind==R_KERNEL)nccl_mesh_program_kernel(program,r->which,r->a,r->x,r->b,r->y,r->n,r->type,r->op,r->nranks,r->scalar,r->published,r->c,r->z);
    else if(r->kind==R_COPY)nccl_mesh_program_copy(program,r->a,r->x,r->b,r->y,r->n,r->which,r->published);
    else if(r->kind==R_WORDS)nccl_mesh_program_words(program,r->a,r->list,(uint32_t)r->n,r->b,r->y,GPU_WAIT_NS,r->touch,4);
    else if(r->kind==R_COUNTED)nccl_mesh_program_counted(program,r->a,r->b,r->c,r->d,r->e,r->list,(uint32_t)r->x,r->n);
    else nccl_mesh_program_publish(program,r->a,r->x,r->y);
  }
  nccl_mesh_program_end(program);
}
static void recording_free(struct recording *k){
  for(int i=0;k && i<k->n;i++){
    if(k->ops[i].a)nccl_mesh_release(k->ops[i].a);
    if(k->ops[i].b)nccl_mesh_release(k->ops[i].b);
    if(k->ops[i].c)nccl_mesh_release(k->ops[i].c);
    if(k->ops[i].d)nccl_mesh_release(k->ops[i].d);
    if(k->ops[i].e)nccl_mesh_release(k->ops[i].e);
    for(int t=0;t<4;t++)if(k->ops[i].touch[t])nccl_mesh_release(k->ops[i].touch[t]);
    free(k->ops[i].list);
  }
  if(k)free(k->ops);
  free(k);
}
/* A kernel over n elements; `published`: its stores system-coherent and fenced, as a word published after
   them lets a SEND read them (nccl-mesh-metal.m); `first` (a combine's, not NULL): its first operand the
   send buffer's bytes there, read in place, not dst's. */
static void kernel(struct sink *s,struct call *k,int which,struct region to,const void *dst,struct region from,const void *src,uint64_t n,int op,
  int published,const void *first){
  uint64_t scalar;memcpy(&scalar,k->scalar,8);
  void *other=first?k->in.buffer:NULL;const uint64_t at=first?off(k->in,first):0;
  if(s->kept)keep(s,(struct recorded){.kind=R_KERNEL,.which=which,.a=to.buffer,.x=off(to,dst),.b=from.buffer,.y=off(from,src),.c=other,.z=at,.n=n,
                                      .type=(int)k->type,.op=op,.nranks=k->comm->nranks,.scalar=scalar,.published=published});
  else nccl_mesh_program_kernel(s->program,which,to.buffer,off(to,dst),from.buffer,off(from,src),n,(int)k->type,op,k->comm->nranks,scalar,published,
                                other,at);
  count(k,GPU_KERNELS,1);
}
/* `bytes` copied by the copy kernel (loaded system-coherent where the NIC wrote them: `received`; stored
   system-coherent where a SEND reads them: `published`). */
static void copy_into(struct sink *s,struct call *k,struct region to,const void *dst,struct region from,const void *src,size_t bytes,int received,
  int published){
  if(s->kept)keep(s,(struct recorded){.kind=R_COPY,.which=received,.published=published,.a=to.buffer,.x=off(to,dst),.b=from.buffer,.y=off(from,src),.n=bytes});
  else nccl_mesh_program_copy(s->program,to.buffer,off(to,dst),from.buffer,off(from,src),bytes,received,published);
  count(k,GPU_COPY,bytes);
}
/* A counted all-to-all's own rows copied from its send buffer into its receive buffer, their offsets and number
   from the counts (its own; the other ranks' segments for this rank where they landed in the group's
   allocation), and each rank's segment for this rank written to `landed` (if any): the counted kernel's
   arguments (nccl-mesh-metal.m). */
static void counted_copy(struct sink *s,struct call *k,struct launch *l){
  const int n=k->comm->nranks;
  const uint64_t row=k->row*type_bytes[k->type];
  const uint32_t width=!(row&15)?16:!(row&3)?4:1,na=9+4*(uint32_t)n;
  uint64_t *a=malloc(na*sizeof *a),landed=0;
  if(!a){if(s->kept)s->kept->failed=1;return;}
  a[0]=off(k->out,k->recv);a[1]=off(k->in,k->send);a[2]=off(k->cnt,k->counted);a[3]=off(l->own,k->wire);a[4]=row;a[5]=width;a[6]=(uint64_t)n;
  a[7]=(uint64_t)k->comm->rank;a[8]=k->landed?off(k->lnd,k->landed):UINT64_MAX;
  for(int q=0,so=0,ro=0;q<n;q++){
    a[9+4*q]=(uint64_t)so;a[10+4*q]=k->segments[q];a[11+4*q]=(uint64_t)ro;a[12+4*q]=k->segments[n+q];
    so+=(int)k->segments[q];ro+=(int)k->segments[n+q];landed+=k->segments[n+q];
  }
  uint64_t units=k->rows*row/width;
  if(k->landed && units<landed)units=landed;
  void *to=k->lnd.buffer?k->lnd.buffer:k->out.buffer;
  if(s->kept)keep(s,(struct recorded){.kind=R_COUNTED,.a=k->out.buffer,.b=k->in.buffer,.c=k->cnt.buffer,.d=l->own.buffer,.e=to,.x=na,.n=units,.list=a});
  else{nccl_mesh_program_counted(s->program,k->out.buffer,k->in.buffer,k->cnt.buffer,l->own.buffer,to,a,na,units);free(a);}
  count(k,GPU_KERNELS,1);
}
/* Once the GPU has run a program: its objects released; a communicator whose failure word a timed-out wait
   set fails (the call is its group's first there; the communicator revoked, the error its async error). */
static void program_ran(void *argument,int failed){
  (void)failed;
  struct ran *r=argument;
  for(int i=0;i<r->nevents;i++)nccl_mesh_release(r->events[i]);
  for(int i=0;i<r->ncomms;i++){
    struct ncclComm *c=r->comms[i];
    if(atomic_exchange_explicit(c->control,0,memory_order_acq_rel)){
      FAIL(c,ncclRemoteError,"a GPU wait on the network ran out (%.0f s of the host's clock without progress, or %llu polls): a bridge "
           "stalled, or a peer has not joined the call",GPU_WAIT_NS/1e9,(unsigned long long)nccl_mesh_wait_bound());
      failed_at(c,r->first[i],c->error);
      comm_revoke(c);
      int none=0;atomic_compare_exchange_strong(&c->async,&none,(int)ncclRemoteError);
    }
    atomic_fetch_sub(&c->programs,1);
  }
  recording_free(r->kept);
  free(r);
}
/* The stream's kept programs taken off it (oldest first) and encoded into `program`, each released once
   it has run; stream_lock held. */
static void take_kept(struct ncclMeshStream *s,void *program){
  struct recording *k=s->pending;
  s->pending=NULL;
  while(k){
    struct recording *next=k->next;
    play(k,program);
    k->ran->kept=k;k->next=NULL;
    nccl_mesh_program_handler(program,program_ran,k->ran);
    k=next;
  }
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
    size_t grown=*capacity?2**capacity:64;
    uint64_t *more=realloc(list,grown*sizeof *more);
    if(!more)return list;
    list=more;*capacity=grown;
  }
  list[2**n]=off(r,word);list[2**n+1]=value;++*n;
  return list;
}

/* The group's program, on its first stream's queue (else its first communicator's): the other streams'
   prior work and the recorded points waited for; the operands copied in (a receive from this rank
   itself, a send's bytes into the group's allocation, a contribution into its operand, or
   premultiplied there) and, after them, the gate word the workers start at; each REDUCE chunk's
   combine once its receive's completion word and those of the earlier sends reading its range are set,
   and, where a later send reads it, its combined word published; the post-division and copy out once
   every completion word of the call is set; the received bytes of point-to-point calls copied out once
   theirs are; every word not yet waited for; the end and each stream's completion value.  Values: the
   program's event's, under stream_lock.  A kept program (`kept`: a deferred stream's,
   ncclMeshStreamDefer) has no gate: the workers wait for the recorded points themselves. */
static void encode(struct launch *l,struct call *calls,int n,struct ncclComm **comms,int ncomms,struct sink *sink,struct ran *ran){
  struct ncclMeshStream *primary=l->nmarks?l->marks[0].stream:NULL;
  void *event=primary?primary->event:comms[0]->event;
  uint64_t *value=primary?&primary->value:&comms[0]->event_value;
  const int kept=sink->kept!=NULL;
  uint64_t *list=NULL;size_t capacity=0;uint32_t m=0;
  l->event=event;
  _Atomic uint64_t *gate=l->gate;
  l->gate=NULL;
  if(!kept){
    int before=0;
    for(int k=0;k<l->nmarks;k++)if(l->marks[k].stream->pending)take_kept(l->marks[k].stream,sink->program);
    for(int k=1;k<l->nmarks;k++){struct ncclMeshStream *s=l->marks[k].stream;if(s->value){sink_wait(sink,calls,s->event,s->value);before=1;}}
    for(int w=0;w<l->nwaits;w++,before=1){
      if(!l->waits[w].word){sink_wait(sink,calls,l->waits[w].event,l->waits[w].value);continue;}
      pthread_mutex_lock(&heap.lock);
      struct span *s=span_of((const void *)l->waits[w].word,8);
      struct region r=s?(struct region){s->buffer,s->at,s->at}:(struct region){0};
      pthread_mutex_unlock(&heap.lock);
      m=0;list=pair(list,&m,&capacity,r,(const void *)l->waits[w].word,l->waits[w].value);
      if(r.buffer)sink_words(sink,calls,r,list,m);
    }
    for(int i=0;i<n;i++){
      struct call *k=calls+i;
      const struct call *mine=self_copy(calls,n,i);
      const size_t bytes=k->count*type_bytes[k->type];
      if(mine){copy_into(sink,k,k->out,k->recv,mine->in,mine->send,bytes,0,0);k->copied=1;before=1;continue;}
      if(k->kind==K_SEND && k->peer!=k->comm->rank && k->wire!=(const unsigned char *)k->send){copy_into(sink,k,l->own,k->wire,k->in,k->send,bytes,0,1);before=1;}
      if(k->kind>=K_SEND)continue;
      const size_t in=in_bytes(k);unsigned char *to=input_at(k);
      if(!in)continue;
      if(k->premultiply){kernel(sink,k,KERNEL_PREMULTIPLY,k->at,to,k->in,k->send,in/type_bytes[k->type],0,1,NULL);before=1;}
      else if(to!=(const unsigned char *)k->send && !k->fresh){copy_into(sink,k,k->at,to,k->in,k->send,in,0,1);before=1;}
    }
    if(before && gate){sink_publish(sink,l->own,gate,1);l->gate=gate;}
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
      if(!covered && at>=own && end<=own+in)copy_into(sink,k,k->at,k->operand+at,k->in,given(k,at),end-at,0,0);
      at=end;
    }
  }
  /* the words each call's program has waited for so far */
  unsigned char **waited=calloc((size_t)n,sizeof *waited);
  for(int i=0;waited && i<n;i++)waited[i]=calloc(calls[i].nwords?calls[i].nwords:1,1);
  #define WAIT(k,i,w) do{ if(waited && waited[i] && !waited[i][w]){waited[i][w]=1;list=pair(list,&m,&capacity,l->own,(k)->words+(w),1);} }while(0)
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind>=K_SEND)continue;
    const size_t e=type_bytes[k->type];
    uint32_t w=0,r=0;
    for(uint32_t s=0;s<k->nsteps && k->nwords;s++){
      const struct mesh_step *step=k->steps+s;
      if(!step->piece.elements)continue;
      const uint32_t chunks=chunks_of(step->piece.elements,e);
      for(uint32_t j=0;step->op==MESH_STEP_REDUCE && j<chunks;j++){
        const uint64_t first=chunk_first(step->piece.elements,e,j),elements=chunk_count(step->piece.elements,e,j);
        const size_t lo=(step->first+first)*e,hi=lo+elements*e;
        m=0;WAIT(k,i,w+j);
        /* the earlier steps' sends that read the range the combine writes */
        for(uint32_t q=0,v=0;q<s;q++){
          const struct mesh_step *earlier=k->steps+q;
          if(!earlier->piece.elements)continue;
          const uint32_t sent=chunks_of(earlier->piece.elements,e);
          for(uint32_t t=0;earlier->op==MESH_STEP_SEND && t<sent;t++){
            const size_t a=(earlier->first+chunk_first(earlier->piece.elements,e,t))*e,b=a+chunk_count(earlier->piece.elements,e,t)*e;
            if(a<hi && lo<b)WAIT(k,i,v+t);
          }
          v+=sent;
        }
        sink_words(sink,k,l->own,list,m);
        kernel(sink,k,KERNEL_COMBINE,k->at,k->operand+lo,l->own,k->pieces[s].at+first*e,elements,k->combine,k->combined!=NULL,
               k->fresh && !written(k,s,lo,hi)?given(k,lo):NULL);
        if(k->combined)sink_publish(sink,l->own,k->combined+r+j,1);
      }
      if(step->op==MESH_STEP_REDUCE && k->combined)r+=chunks;
      w+=chunks;
    }
    const size_t out=out_bytes(k);unsigned char *from=output_at(k);
    const int divide=out && k->postdivide,copy=out && from!=(unsigned char *)k->recv;
    if(divide || copy){
      m=0;
      for(uint32_t x=0;x<k->nwords;x++)WAIT(k,i,x);
      sink_words(sink,k,l->own,list,m);
      if(divide)kernel(sink,k,KERNEL_POSTDIVIDE,k->at,from,k->at,from,out/e,0,0,NULL);
      if(copy)copy_into(sink,k,k->out,k->recv,k->at,from,out,1,0);
    }
  }
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind==K_RECV && k->peer!=k->comm->rank && k->wire!=(unsigned char *)k->recv){
      m=0;WAIT(k,i,0);
      sink_words(sink,k,l->own,list,m);
      copy_into(sink,k,k->out,k->recv,l->own,k->wire,k->count*type_bytes[k->type],1,0);
    }
  }
  /* a counted all-to-all's own rows, once the other ranks' count segments have landed */
  for(int i=0;i<n;i++){
    struct call *k=calls+i;
    if(k->kind!=K_COUNTED)continue;
    const uint32_t peers=(uint32_t)k->comm->nranks-1;
    m=0;
    for(uint32_t x=peers;x<2*peers;x++)WAIT(k,i,x);
    sink_words(sink,k,l->own,list,m);
    if(k->rows || k->landed)counted_copy(sink,k,l);
  }
  /* the network done before the program's end: every word not yet waited for (a call's sends, COPY
     steps' receives, point-to-point transfers) */
  for(int i=0;i<n;i++){
    m=0;
    for(uint32_t x=0;x<calls[i].nwords;x++)WAIT(calls+i,i,x);
    sink_words(sink,calls+i,l->own,list,m);
  }
  #undef WAIT
  for(int i=0;waited && i<n;i++)free(waited[i]);
  free(waited);free(list);
  for(int j=0;j<ncomms;j++){
    uint64_t first=0;
    for(int i=0;i<n;i++)if(calls[i].comm==comms[j] && (!first || calls[i].index<first))first=calls[i].index;
    ran->first[ran->ncomms]=first;
    ran->comms[ran->ncomms++]=comms[j];atomic_fetch_add(&comms[j]->programs,1);
  }
  l->end=++*value;sink_signal(sink,event,l->end);
  for(int k=0;k<l->nmarks;k++){
    struct mark *mark=l->marks+k;struct ncclMeshStream *s=mark->stream;
    if(s==primary){mark->done=l->end;continue;}
    mark->done=++s->value;sink_signal(sink,s->event,mark->done);
    nccl_mesh_retain(s->event);ran->events[ran->nevents++]=s->event;
  }
  if(kept){
    struct recording *k=sink->kept,**at=(struct recording **)&primary->pending;
    k->ran=ran;
    while(*at)at=&(*at)->next;
    *at=k;
    return;
  }
  if(primary)primary->committed=primary->value;
  for(int k=1;k<l->nmarks;k++)l->marks[k].stream->committed=l->marks[k].stream->value;
  count(calls,COMMITS,1);
  nccl_mesh_program_commit(sink->program,program_ran,ran);
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
      failed_at(c,calls[i].index,c->error);
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
  struct ran *ran=calloc(1,sizeof *ran+(size_t)n*(sizeof(struct ncclComm *)+sizeof(uint64_t)+sizeof(void *)));
  if(l){l->marks=calloc((size_t)n,sizeof *l->marks);l->waits=calloc((size_t)n*(2*(READS+1)+1),sizeof *l->waits);}
  if(!l || !l->marks || !l->waits || !tally || !comms || !ran){
    if(l){free(l->marks);free(l->waits);}
    free(l);free(tally);free(comms);free(ran);calls_free(calls,n);
    return FAIL(NULL,ncclSystemError,"allocation");
  }
  ran->comms=(struct ncclComm **)(ran+1);ran->first=(uint64_t *)(ran->comms+n);ran->events=(void **)(ran->first+n);
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
  /* a stream's word (ncclMeshStreamWaitWord): a recorded point of its groups from then on (reached, it
     costs a later group one load) */
  for(int m=0;m<l->nmarks;m++){
    struct ncclMeshStream *s=l->marks[m].stream;
    if(s->word)wait_add(l,(struct point){.word=(_Atomic uint64_t *)s->word,.value=s->word_value});
  }
  /* completion words where a program may wait on them: a stream's call, or any isend or irecv */
  int words=l->nmarks>0;
  for(int i=0;i<n && !words;i++)words=requests(calls+i)>0;
  if(!status)status=place(calls,n,l,words);
  /* a program where a stream's completion is its, or the GPU has work; kept where its stream defers and
     it has no GPU work before its transfers (ncclMeshStreamDefer) */
  int work=l->nmarks>0;
  for(int i=0;i<n && !work && !status;i++)work=gpu_work(calls,n,i);
  const int defer=!status && l->nmarks==1 && l->marks[0].stream->deferred && ncomms==1 && !gated(calls,n) && !sends_combined(calls,n);
  struct sink sink={0};
  if(!status && defer && !(sink.kept=calloc(1,sizeof *sink.kept)))status=FAIL(NULL,ncclSystemError,"allocation");
  if(!status && work && !defer && !(sink.program=nccl_mesh_program_begin(l->nmarks?l->marks[0].stream->queue:comms[0]->queue)))
    status=FAIL(NULL,ncclUnhandledCudaError,"a command buffer on the queue");
  if(status){
    pthread_mutex_unlock(&stream_lock);
    regions_release(calls,n,l);
    launch_free(l);free(comms);free(ran);calls_free(calls,n);
    return status;
  }
  const int program=sink.program || sink.kept;
  if(program){
    encode(l,calls,n,comms,ncomms,&sink,ran);
    record(l,calls,n);
    if(!defer){
      for(int w=0;w<l->nwaits;w++)point_drop(l->waits+w);
      l->nwaits=0;
    }
  }
  else{free(ran);l->gate=NULL;}
  regions_release(calls,n,l);
  l->items=ncomms;
  for(int j=0;j<ncomms;j++){
    struct item *it=calloc(1,sizeof *it);
    it->calls=calloc((size_t)n,sizeof *it->calls);it->launch=l;it->deadline=deadline_after();it->gpu=program;
    for(int i=0;i<n;i++)if(calls[i].comm==comms[j]){it->calls[it->n++]=calls[i];calls[i].steps=NULL;calls[i].pieces=NULL;calls[i].segments=NULL;}
    struct ncclComm *c=comms[j];
    pthread_mutex_lock(&c->lock);
    if(c->tail)c->tail->next=it;else c->head=it;
    c->tail=it;
    if(c->parked)pthread_cond_broadcast(&c->cond);
    pthread_mutex_unlock(&c->lock);
  }
  pthread_mutex_unlock(&stream_lock);
  free(comms);free(calls);
  if(!l->sync)return ncclSuccess;
  pthread_mutex_lock(&l->lock);
  while(l->items)pthread_cond_wait(&l->cond,&l->lock);
  status=l->result;
  pthread_mutex_unlock(&l->lock);
  if(l->event){
    /* the NULL stream: the program's end */
    count(&(struct call){.tally=tally->counts[n-1]},HOST_WAITS,1);
    for(uint64_t deadline=deadline_after();!status && nccl_mesh_event_value(l->event)<l->end;){
      if(nccl_mesh_gpu_failed())status=FAIL(NULL,ncclUnhandledCudaError,"the group's GPU program failed");
      else if(now_ns()>deadline)status=FAIL(NULL,ncclTimeout,"the group's GPU program: not done by the deadline (MESH_NCCL_TIMEOUT)");
      else sched_yield();
    }
  }
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
    if(group.n<group.capacity){k.index=atomic_fetch_add(&c->issued,1)+1;group.calls[group.n++]=k;k.segments=NULL;}
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
  char error[256];
  pthread_mutex_lock(&global_lock);
  int failed=!queue && nccl_mesh_gpu_attach(error,sizeof error);
  pthread_mutex_unlock(&global_lock);
  if(failed)return FAIL(NULL,ncclUnhandledCudaError,"the GPU: %s",error);
  struct ncclMeshStream *s=calloc(1,sizeof *s);
  if(!s)return FAIL(NULL,ncclSystemError,"allocation");
  s->made=!queue;s->queue=queue?queue:nccl_mesh_queue_create();
  s->event=s->queue?nccl_mesh_event_create(s->queue):NULL;
  if(!s->event){if(s->made && s->queue)nccl_mesh_release(s->queue);free(s);return FAIL(NULL,ncclUnhandledCudaError,"the stream's Metal queue and event");}
  *stream=s;
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamDestroy(cudaStream_t stream){
  if(!stream)return ncclSuccess;
  if(stream->pending)ncclMeshStreamSynchronize(stream);
  nccl_mesh_release(stream->event);
  if(stream->made)nccl_mesh_release(stream->queue);
  free(stream);
  return ncclSuccess;
}
/* The stream's kept programs committed to its queue, after its committed ones; stream_lock held. */
static void no_ran(void *argument,int failed){(void)argument;(void)failed;}
static ncclResult_t flush_kept(struct ncclMeshStream *s){
  if(!s->pending)return ncclSuccess;
  void *program=nccl_mesh_program_begin(s->queue);
  if(!program)return FAIL(NULL,ncclUnhandledCudaError,"a command buffer on the stream's queue");
  if(s->committed)nccl_mesh_program_wait(program,s->event,s->committed);
  take_kept(s,program);
  s->committed=s->value;
  count(NULL,COMMITS,1);
  nccl_mesh_program_commit(program,no_ran,NULL);
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamSynchronize(cudaStream_t stream){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamSynchronize: no stream");
  uint64_t deadline=deadline_after(),value;
  pthread_mutex_lock(&stream_lock);
  ncclResult_t status=flush_kept(stream);
  value=stream->value;
  pthread_mutex_unlock(&stream_lock);
  if(status)return status;
  while(nccl_mesh_event_value(stream->event)<value){
    if(nccl_mesh_gpu_failed())return FAIL(NULL,ncclUnhandledCudaError,"ncclMeshStreamSynchronize: a GPU program failed");
    if(now_ns()>deadline)return FAIL(NULL,ncclTimeout,"ncclMeshStreamSynchronize: the stream did not reach %llu",(unsigned long long)value);
    usleep(20);
  }
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamQuery(cudaStream_t stream){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamQuery: no stream");
  pthread_mutex_lock(&stream_lock);uint64_t value=stream->value;int kept=stream->pending!=NULL;pthread_mutex_unlock(&stream_lock);
  return !kept && nccl_mesh_event_value(stream->event)>=value?ncclSuccess:ncclInProgress;
}
ncclResult_t ncclMeshStreamDefer(cudaStream_t stream,int defer){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamDefer: no stream");
  pthread_mutex_lock(&stream_lock);
  stream->deferred=defer!=0;
  ncclResult_t status=defer?ncclSuccess:flush_kept(stream);
  pthread_mutex_unlock(&stream_lock);
  return status;
}
ncclResult_t ncclMeshStreamWaitWord(cudaStream_t stream,const uint64_t *word,uint64_t value){
  if(!stream || !word)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamWaitWord: NULL argument");
  pthread_mutex_lock(&stream_lock);
  stream->word=word;stream->word_value=value;
  pthread_mutex_unlock(&stream_lock);
  return ncclSuccess;
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
    for(int i=0;i<n;i++)nccl_mesh_program_copy(commandBuffer,buffers[i],at[i],src[i],offset[i],bytes[i],0,1);
    nccl_mesh_program_publish(commandBuffer,buffers[n],at[n],value);
    nccl_mesh_program_end(commandBuffer);
    pthread_mutex_unlock(&stream_lock);
  }
  for(int i=0;buffers && i<=n;i++)if(buffers[i])nccl_mesh_release(buffers[i]);
  free(buffers);free(at);
  return status;
}
ncclResult_t ncclMeshStreamEncodeWait(cudaStream_t stream,void *commandBuffer){
  if(!stream || !stream->event || !commandBuffer)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamEncodeWait: NULL argument");
  pthread_mutex_lock(&stream_lock);
  if(stream->committed && nccl_mesh_event_value(stream->event)<stream->committed)nccl_mesh_program_wait(commandBuffer,stream->event,stream->committed);
  take_kept(stream,commandBuffer);
  pthread_mutex_unlock(&stream_lock);
  return ncclSuccess;
}
/* ---- agreement: ULFM's MPI_Comm_agree [Bland et al. 2013; MPI 5.0 draft, chapter "Process Fault
   Tolerance"] after a failure ---- */
/* A rank's vote: whether the flood has reached it, the index of its first call that failed since the
   last agreement (0: none), whether its connections are whole, and the hash of its link table's map of
   the communicator (what its next call plans on). */
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
/* A request posted (retried while its ring is full) and waited for, by the deadline. */
static ncclResult_t host_post(struct ncclComm *c,int send,void *comm,void *data,size_t bytes,uint64_t deadline,void **request){
  for(*request=NULL;;){
    int result=send?mesh_net_isend(comm,data,bytes,0,window.mh,NULL,request):
      mesh_net_irecv(comm,1,&data,&bytes,(int[]){0},(void *[]){window.mh},NULL,request);
    if(result)return net_failure(c,result,send?"ncclMeshCommAgree: an isend":"ncclMeshCommAgree: an irecv");
    if(*request)return ncclSuccess;
    if(now_ns()>deadline)return FAIL(c,ncclTimeout,"ncclMeshCommAgree: a request ring stayed full past the deadline");
    sched_yield();
  }
}
/* One round of FloodSet [Lynch, "Distributed Algorithms" 1996, §6.2.1]: the votes this rank knows sent
   to each neighbour, each neighbour's received first, and theirs merged in. */
static ncclResult_t flood(struct ncclComm *c,struct peer *peers,const unsigned char *want,struct vote *votes,unsigned char *buffers,
  size_t slot,uint64_t deadline){
  const int n=c->nranks;
  void **requests=calloc((size_t)n*2,sizeof *requests);
  if(!requests)return FAIL(c,ncclSystemError,"allocation");
  memcpy(buffers,votes,(size_t)n*sizeof *votes);
  ncclResult_t status=ncclSuccess;
  for(int p=0;p<n && !status;p++)if(want[p]){
    status=host_post(c,0,peers[p].recv[CH_P2P],buffers+slot*(size_t)(p+1),(size_t)n*sizeof *votes,deadline,requests+2*p);
    if(!status)status=host_post(c,1,peers[p].send[CH_P2P],buffers,(size_t)n*sizeof *votes,deadline,requests+2*p+1);
  }
  for(int i=0;i<2*n && !status;i++)while(requests[i]){
    int done=0,size=0,result=mesh_net_test(requests[i],&done,&size);
    if(result)status=net_failure(c,result,"ncclMeshCommAgree: a vote");
    else if(done)requests[i]=NULL;
    else if(now_ns()>deadline)status=FAIL(c,ncclTimeout,"ncclMeshCommAgree: rank %d's vote did not arrive by the deadline (MESH_NCCL_TIMEOUT)",i/2);
    else sched_yield();
    if(status)break;
  }
  for(int p=0;p<n && !status;p++)if(want[p]){
    const struct vote *theirs=(const struct vote *)(buffers+slot*(size_t)(p+1));
    for(int r=0;r<n;r++)if(theirs[r].known && !votes[r].known)votes[r]=theirs[r];
  }
  free(requests);
  return status;
}
/* Every connection closed since the last agreement vacated by the bridge (nothing of it on the wire),
   by the deadline; then the memory held for the failed parts is given back. */
static ncclResult_t vacate(struct ncclComm *c,uint64_t deadline){
  while(c->nclosed){
    if(mesh_net_vacated(c->closed[c->nclosed-1])){c->nclosed--;continue;}
    if(now_ns()>deadline)return FAIL(c,ncclTimeout,"ncclMeshCommAgree: the bridge did not vacate a closed connection by the deadline");
    usleep(200);
  }
  nccl_mesh_event_signal(c->quiet,c->quiet_value);
  return ncclSuccess;
}
ncclResult_t ncclMeshCommAgree(ncclComm_t comm,uint64_t *failed,uint64_t *epoch){
  if(!comm || !failed || !epoch)return FAIL(comm,ncclInvalidArgument,"ncclMeshCommAgree: NULL argument");
  struct ncclComm *c=comm;
  const int n=c->nranks;
  drain(c);
  pthread_mutex_lock(&c->lock);
  if(atomic_load(&c->broken) && c->open)close_peers(c,c->peers,1);
  c->agreeing=1;
  pthread_mutex_unlock(&c->lock);
  const uint64_t deadline=deadline_after(),key=mix(c->key^mix(++c->agreements));
  const size_t slot=slice((size_t)n*sizeof(struct vote));
  unsigned char *want=calloc((size_t)n,1),*buffers=NULL;
  struct peer *fresh=calloc((size_t)n,sizeof *fresh);
  struct vote *votes=calloc((size_t)n,sizeof *votes);
  ncclResult_t status=want && fresh && votes?ncclSuccess:FAIL(c,ncclSystemError,"allocation");
  /* the flood's neighbours: the ranks the stated map links to this one, up or not */
  c->epoch=mesh_link_table_read(c->table,c->seen);
  for(int p=0;!status && p<n;p++)
    want[p]=p!=c->rank && mesh_link_at(c->seen,c->nodes[c->rank],c->nodes[p])->stated && mesh_link_at(c->seen,c->nodes[p],c->nodes[c->rank])->stated;
  if(!status && n>1)status=connect_ranks(c,key,want,fresh,deadline);
  if(!status)status=span_alloc(slot*(size_t)(n+1),&buffers,NULL);
  /* the votes flooded n - 1 rounds; again, once the tables move, until every rank's map is the same */
  for(int same=0;!status && !same;){
    c->epoch=mesh_link_table_read(c->table,c->seen);
    mesh_link_table_map(c->seen,c->nodes,(uint32_t)n,&c->map,c->pairs,c->cost);
    memset(votes,0,(size_t)n*sizeof *votes);
    pthread_mutex_lock(&c->lock);
    votes[c->rank]=(struct vote){1,c->failed,(uint64_t)whole(c),map_hash(c)};
    pthread_mutex_unlock(&c->lock);
    for(int round=1;round<n && !status;round++)status=flood(c,fresh,want,votes,buffers,slot,deadline);
    same=1;
    for(int r=0;!status && r<n;r++){
      if(!votes[r].known)status=FAIL(c,ncclRemoteError,"ncclMeshCommAgree: rank %d's vote reached this rank by no stated link",r);
      same&=votes[r].map==votes[c->rank].map;
    }
    if(!status && !same){
      if(now_ns()>deadline)status=FAIL(c,ncclTimeout,"ncclMeshCommAgree: the ranks' link tables did not hold one map of the communicator by the deadline");
      for(uint64_t e=c->epoch,until=now_ns()+20000000;!status && atomic_load(&c->table->epoch)==e && now_ns()<until;)usleep(200);
    }
  }
  /* the first failed call among the ranks; after one, or where a rank's connections are not whole, every
     rank takes the agreement's connections as its own */
  uint64_t first=0;int kept=1;
  for(int r=0;!status && r<n;r++){
    if(votes[r].failed && (!first || votes[r].failed<first))first=votes[r].failed;
    kept&=votes[r].whole!=0;
  }
  pthread_mutex_lock(&c->lock);
  if(!status && (first || !kept)){
    close_peers(c,c->peers,1);
    memcpy(c->peers,fresh,(size_t)n*sizeof *fresh);memset(fresh,0,(size_t)n*sizeof *fresh);
    c->open=1;
  }
  close_peers(c,fresh,0);
  pthread_mutex_unlock(&c->lock);
  if(buffers)span_release(buffers,NULL,0);
  if(!status)status=vacate(c,deadline);
  pthread_mutex_lock(&c->lock);
  if(!status){atomic_store(&c->issued,0);c->failed=0;c->cause[0]=0;atomic_store(&c->broken,0);atomic_store(&c->async,0);c->alive=0;}
  c->agreeing=0;
  pthread_mutex_unlock(&c->lock);
  if(!status){*failed=first;*epoch=c->epoch;}
  free(want);free(fresh);free(votes);
  return status;
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
    .commits=c[COMMITS],.wakeups=c[WAKEUPS],.buffers=c[BUFFERS]};
}
ncclResult_t ncclMeshGetCounts(ncclMeshCounts_t *counts){
  if(!counts)return FAIL(NULL,ncclInvalidArgument,"ncclMeshGetCounts: counts is NULL");
  uint64_t now[COUNTS];
  for(int i=0;i<COUNTS;i++)now[i]=atomic_load_explicit(&totals[i],memory_order_relaxed);
  *counts=counts_of(now,0);
  return ncclSuccess;
}
ncclResult_t ncclMeshTallyCounts(void *tally,ncclMeshCounts_t *counts,int capacity,int *count){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshTallyCounts: count is NULL");
  struct tally *t=tally;
  *count=t?t->n:0;
  for(int i=0;t && counts && i<t->n && i<capacity;i++)counts[i]=counts_of(t->counts[i],1);
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

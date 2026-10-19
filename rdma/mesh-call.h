#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim); 0, or the refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe): a lease of one slot of `bytes` at exactly arena page `page` (a
   multiple of the block), so the two ends of a stripe place what they exchange at the same window offsets; ENOMEM where
   a page of it is held, EINVAL where `page` is not a block's first */
int mesh_section_at(struct mesh_ctx *,uint32_t page,size_t bytes,struct mesh_section *);
/* design/bridge-dataflow.md#stripes (mesh 8fc27cbe, on the resident bridge): a ring of `entries` (a power of two, 4 at
   least) stripe requests (mesh.h struct mesh_requests), a lease of the client's slot registered with its node's bridge
   (MESH_RINGS of every client's at once), whose requests the bridge takes while the slot is the client's (its
   retirement deregisters the ring); a GPU writes requests into a ring of its own by the protocol mesh.h states.  0,
   EINVAL, EBUSY where MESH_RINGS rings are registered, or the refused lease */
int mesh_requests_create(struct mesh_ctx *,uint32_t entries,struct mesh_requests **);
/* a host thread's request into a ring (any number of threads at once; mesh.h struct mesh_stripe's operands, `offset` a
   byte offset of the window, `word` a region offset), then mesh_requests_wake: 0, EAGAIN where the ring holds `entries`
   requests the bridge has not taken (not yet: write it again), EINVAL where it is not one the window holds */
int mesh_write(struct mesh_requests *,uint32_t node,uint64_t offset,uint64_t bytes,uint64_t word,uint64_t value);
/* the bridge told a ring's requests toward `node` are written: the bells of the ports paired with `node` rung (each
   port's send thread takes them; one parked looks again within 1 ms anyway while a stripe flow is up on it) and the
   control word (the bridge makes the ports' stripe flows where none is up); a GPU writer's host calls it once its
   kernel wrote its requests */
void mesh_requests_wake(struct mesh_requests *,uint32_t node);
/* design/algorithm-sources.md#programcopy: `section` sent to or received from (`direction`) the member `node`, never a
   port, on its `queue` (< MESH_QUEUES), for the invocations [begin, end) alone, invocation t at ring slot (t - begin)
   mod `depth` (0: the program's), its cells, words and slots indexed from `begin`; 0, EINVAL or a refused lease */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t node,uint32_t queue,int direction,uint32_t identity,struct mesh_section,
  uint32_t invocation_pages,uint32_t begin,uint32_t end,uint32_t depth);
/* design/prepared-machine.md#M04, #M08: the program's cells, words and targets laid out for `slots` queue pairs a
   queue, `invocations` invocations at ring `depth`; EINVAL where a transfer has more slots than the program */
int mesh_transfers_prepare(struct mesh_ctx *,uint32_t slots,uint32_t invocations,uint32_t depth);
/* design/prepared-machine.md#M30 */
int mesh_transfers_cyclic(struct mesh_ctx *);
/* design/bridge-and-ipc.md#protocol: the digest of what the program is made from (0: none), named after
   mesh_transfers_prepare and before mesh_transfers_start; a peer's program of the same execution that names another
   fails both as values (ESTALE); EBUSY once started */
int mesh_transfers_digest(struct mesh_ctx *,uint64_t digest);
/* design/bridge-and-ipc.md#clients, design/recovery.md#first: 0 once the node's bridge has paired the program toward
   every node it binds, on its current map, directly or along its route through other nodes' bridges; else the failure,
   and that node's status in the client's slot names the node and the code: ESRCH the bridge gone, ETIMEDOUT past
   MESH_PEER_NS+MESH_PAIR_NS+MESH_PAUSE_NS, EHOSTUNREACH no cable or route reaches it, ENETDOWN, ECONNRESET,
   ECANCELED a link or the peer's program lost, or this program's own slot retired while it waits (a revoke), ESTALE a
   member's program of the execution made from another digest (mesh_transfers_digest; every destination fails, naming
   that member), ENOSPC, ENOMEM a grant refused. A later loss fails the program alike; the next start rereads the map */
int mesh_transfers_start(struct mesh_ctx *);
/* design/bridge-and-ipc.md#progress: the progress threads of the ports a started program's flows ride woken (each port's
   bell), a group of it about to publish or to wait on its landings (a thread parked after MESH_IDLE_NS of nothing looks
   again within 1 ms anyway while a flow is its port's) */
void mesh_transfers_wake(struct mesh_ctx *);
/* design/bridge-and-ipc.md#files, the link lanes: a rank's report, gathered at a communicator's init through NCCL's
   bootstrap, of its bridge's node and of each port whose link has its MESH_LINK_LANES lanes made and a peer (`peer`,
   `port`), the lane pairs free there (`free`, a bit a pair: pair p is lanes MESH_LANE_FIRST + 2 p and + 2 p + 1, free
   where both lease words are 0, a DIRTY one not), read wait-free from the region's lease words (mesh_lanes_free; 0, or
   EINVAL).  The rule every rank applies alike to the gathered reports (mesh_lanes_choose, a pure function), in rank
   order: for rank r's links toward a node another rank occupies, the lowest pair free in every report of either end
   for every cable between the two nodes (the reports do not match the cables port to port) that no earlier rank of r's
   node chose on one of those ports, into chosen[r * MESH_PORTS + k] for each such link k (UINT32_MAX for every other
   k); 0, EINVAL where a report holds more than MESH_PORTS links or a port past them, or ENOSPC where such a link has
   none (its init fails as a value; the next init reads the words as they then stand).  A rank's claim of its chosen
   pairs (its row of `chosen`) at its own bridge, each lane's word 0 to `owner` by compare-exchange: 0, EINVAL, or EAGAIN
   where a word was taken first (its claims released and the init failed as a value).  The release of every word
   `owner` holds, by compare-exchange on the word observed: `owner` to 0, `owner` with DIRTY to DIRTY alone, the
   bridge's mark kept until the link pairs afresh (design/recovery.md#five-seconds W3b) */
struct mesh_lane_link { uint32_t peer,port,free,padding; };
struct mesh_lane_report { uint32_t node,count; struct mesh_lane_link link[MESH_PORTS]; };
int mesh_lanes_free(struct hdr *,struct mesh_lane_report *);
int mesh_lanes_choose(const struct mesh_lane_report *reports,uint32_t ranks,uint32_t *chosen);
int mesh_lanes_claim(struct hdr *,uint64_t owner,const struct mesh_lane_report *mine,const uint32_t *chosen);
void mesh_lanes_release(struct hdr *,uint64_t owner);

/* ---- The numerical executor (mesh 2b49f3c6^, restored on the resident bridge's substrate) -----------------------
   Worker threads fire a function when its inputs' arrivals reach its pending count, then publish its outputs by
   prepared publications; an instance's status word counts its calls and its landings (deliverables.md, The machine:
   FIRE, PUBLISH, SUBMIT, RESULT).  On this substrate (mesh 800fe6bc): the program is a client slot of the node's
   bridge (mesh-dataflow.h mesh_attach), its N invocation slots one cyclic program of N invocations (prepared-machine.md
   M30: running invocation T is slot T mod N in cycle T / N, which is the old generation T = slot + k N), every
   section's slots a ring of one stream (M08); a received row's arrival is the program's landing word (M07,
   mesh-collective.h mesh_host_inputs: 1 plus the cycle once landed), which the workers poll as libnccl-mesh's
   progress thread does; a publication stores its prepared argument plus the cycle (M04, M10, mesh_host_publish);
   constants, sent once, are a one-pass program of their own slot (`constants`: a cyclic program sends every transfer
   every cycle); the arrival cells, completion records and input words are leases of the client's slot, and the call
   records, instances and submissions the process's memory.  The records' tags M18, M20, M22, M23, M25, M37, M40 to
   M43 are those of mesh 2b49f3c6^'s design/hot-path-loads.md and prepared-machine.md, not today's rows. */
#define MESH_COMPUTE_THREADS 8
/* design/algorithm-sources.md#meshresult */
enum { MESH_RESULT_SUCCESS, MESH_RESULT_LINK, MESH_RESULT_FUNCTION, MESH_RESULT_BUSY };
#define MESH_RESULT(kind,id,code) ((uint64_t)(kind)<<62|(uint64_t)(id)<<32|(uint32_t)(code))
struct mesh_status { uint64_t value,completed; };
_Static_assert(sizeof(struct mesh_status)==16 && __atomic_always_lock_free(sizeof(struct mesh_status),0),"mesh_status lock-free snapshot");
/* hot-path-loads M42: an invocation slot's status and the events (`remaining`) its generation (`invocation`) waits for,
   `count` of them a generation after its first */
struct mesh_instance { _Alignas(32) _Atomic(struct mesh_status) status; _Atomic uint32_t remaining,invocation; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_instance)==32 && _Alignof(struct mesh_instance)==32,"mesh_instance");
/* design/algorithm-sources.md#meshresult */
static inline void mesh_result_conclude(_Atomic(struct mesh_status) *destination,uint64_t status){
  struct mesh_status previous=atomic_load_explicit(destination,memory_order_relaxed);
  do {
    if(previous.value>>62!=MESH_RESULT_SUCCESS && previous.value>>62!=MESH_RESULT_BUSY)return;
    struct mesh_status next={status,previous.completed};
    if(status>>62==MESH_RESULT_SUCCESS)next.completed=status>>32?UINT64_MAX:(uint64_t)(uint32_t)status+1;
    if(atomic_compare_exchange_strong_explicit(destination,&previous,next,memory_order_release,memory_order_relaxed))return;
  } while(status>>62!=MESH_RESULT_SUCCESS);
}
/* design/algorithm-sources.md#meshresult; hot-path-loads M42 */
static __attribute__((noinline,unused)) void mesh_instance_complete(struct mesh_instance *instances,uint32_t count){
  for(uint32_t i=0;i<count;i++){
    struct mesh_instance *instance=&instances[i];
    if(atomic_fetch_sub_explicit(&instance->remaining,1,memory_order_acq_rel)!=1)continue;
    uint32_t generation=atomic_load_explicit(&instance->invocation,memory_order_relaxed);
    atomic_store_explicit(&instance->invocation,generation+instance->stride,memory_order_relaxed);
    atomic_store_explicit(&instance->remaining,instance->count,memory_order_relaxed);
    mesh_result_conclude(&instance->status,MESH_RESULT(MESH_RESULT_SUCCESS,!instance->count,generation));
  }
}
/* hot-path-loads M18: a call's arrival cell, which its producer (or the submission) stores 1 plus the cycle */
struct mesh_arrival { _Alignas(16) uint32_t call; _Atomic uint64_t argument; };
_Static_assert(sizeof(struct mesh_arrival)==16 && _Alignof(struct mesh_arrival)==16 && offsetof(struct mesh_arrival,argument)==8,"M18");
/* hot-path-loads M37: a resident function's completion of one frame, which its GPU work stores nonzero */
struct mesh_completion { _Alignas(32) _Atomic uint64_t event; struct mesh_instance *result; };
_Static_assert(sizeof(struct mesh_completion)==32 && _Alignof(struct mesh_completion)==32 && offsetof(struct mesh_completion,result)==8,"M37");

struct mesh_calls;
struct mesh_call;
struct mesh_function;
/* design/algorithm-sources.md#programkernel_call */
typedef __attribute__((swiftcall)) void (*mesh_invoke)(struct mesh_call *,void * __attribute__((swift_context)));
/* hot-path-loads M41: a slot's root arrival ranges, a range a worker; `cycle` the cycle its next generation runs in */
struct mesh_submission {
  _Alignas(128) struct mesh_arrival *destinations[MESH_COMPUTE_THREADS];
  uint32_t counts[MESH_COMPUTE_THREADS];
  uint32_t generation,stride,count,cycle;
};
_Static_assert(sizeof(struct mesh_submission)==128 && _Alignof(struct mesh_submission)==128,"M41");
/* hot-path-loads M41; design/algorithm-sources.md#program: the slot's next generation submitted, its root arrivals
   stored 1 plus its cycle (prepared-machine.md M30); the generation */
static inline __attribute__((always_inline)) uint32_t mesh_submit(struct mesh_submission *submission){
  uint32_t generation=submission->generation,count=submission->count;
  const uint64_t argument=1+(uint64_t)submission->cycle++;
  submission->generation=generation+submission->stride;
  for(uint32_t i=0;i<count;i++){
    struct mesh_arrival *cells=submission->destinations[i];
    uint32_t extent=submission->counts[i];
    for(uint32_t k=0;k<extent;k++)atomic_store_explicit(&cells[k].argument,argument,memory_order_release);
  }
  return generation;
}
/* hot-path-loads M42; design/algorithm-sources.md#meshresult */
static inline __attribute__((always_inline)) uint64_t mesh_result(struct mesh_instance *instance,uint32_t generation){
  struct mesh_status status=atomic_load_explicit(&instance->status,memory_order_acquire);
  if(status.completed==(uint64_t)generation+1 || status.completed==UINT64_MAX)return 0;
  return status.value>>62==MESH_RESULT_SUCCESS?MESH_RESULT(MESH_RESULT_BUSY,0,0):status.value;
}
/* hot-path-loads M43: a call's operand, its slot's bytes, the slot index, its row, and the call's running generation
   (`sequence`, the invocation T: mesh-plan.h holds the name mesh_operand, a collective's typed operand) */
struct mesh_call_operand {
  _Alignas(16) void *data;
  size_t bytes;
  uint32_t index,row;
  const uint32_t *sequence;
};
_Static_assert(sizeof(struct mesh_call_operand)==32 && _Alignof(struct mesh_call_operand)==16 &&
  offsetof(struct mesh_call_operand,sequence)==24,"M43");
/* hot-path-loads M20: a function's call record of one slot: its arrivals pending, the cycle and generation it was
   last fired for (the arrival that fired it: 1 plus the cycle), and its outputs' prepared publications; where its
   function is active (mesh_call_active, design/bridge-dataflow.md#routed-readiness), the word of the count its slot
   is checked against as it fires (`count`, NULL: not active) and its slot */
struct mesh_call {
  _Alignas(128) struct mesh_call_operand *operands;
  struct mesh_instance *instance;
  _Atomic uint32_t *completed;
  _Alignas(16) uint32_t index;
  uint32_t pending,identity;
  mesh_invoke submit;
  void *argument;
  uint32_t publication_count,recurring;
  uint32_t cycle,generation;
  const _Atomic uint32_t *count;
  uint32_t slot;
  _Alignas(128) struct prepared_publication publications[];
};
_Static_assert(sizeof(struct mesh_call)==128 && _Alignof(struct mesh_call)==128,"mesh_call record");
_Static_assert(offsetof(struct mesh_call,pending)==36 &&
  offsetof(struct mesh_call,submit)==48 && offsetof(struct mesh_call,argument)==56 &&
  offsetof(struct mesh_call,publication_count)==64 && offsetof(struct mesh_call,recurring)==68 &&
  offsetof(struct mesh_call,cycle)==72 && offsetof(struct mesh_call,generation)==76 && offsetof(struct mesh_call,count)==80 &&
  offsetof(struct mesh_call,publications)==128,"M20");
/* hot-path-loads M26 (2b49f3c6^) */
struct prepared_residency {
  _Alignas(32) uint64_t completed;
  uint32_t reserved[6];
};
_Static_assert(sizeof(struct prepared_residency)==32 && offsetof(struct prepared_residency,completed)==0,"M26 (2b49f3c6^)");
/* hot-path-loads M40 */
struct mesh_resident_control {
  _Alignas(32) _Atomic uint32_t shutdown;
  uint32_t reserved[7];
};
_Static_assert(sizeof(struct mesh_resident_control)==32 && offsetof(struct mesh_resident_control,shutdown)==0,"M40");
/* hot-path-loads M40; design/algorithm-sources.md#resident-metal */
static inline void mesh_residency_stop(struct mesh_resident_control *control){
  atomic_store_explicit(&control->shutdown,1,memory_order_release);
}
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline)) uint32_t mesh_call_index(const struct mesh_call *call){return call->index;}
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline,returns_nonnull)) struct mesh_call_operand *mesh_call_operands(const struct mesh_call *call){return call->operands;}
typedef void (*mesh_dispose)(void *);

/* design/algorithm-sources.md#programkernel_call: `count` invocation slots of `workers` workers over the program
   `program` (cyclic, its transfers' sections rings of `count` slots) and the one-pass program `constants` (NULL: none),
   both slots of one attach; `dispose(owner)` once the last worker has ended after mesh_calls_destroy */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *program,struct mesh_ctx *constants,uint32_t workers,uint32_t count,
  void *owner,mesh_dispose);
/* a function on `worker` of its inputs (the first `dependency_count` waited on) and outputs; `submit` NULL: a resident
   function, which reports each frame's completion (mesh_function_completion) and publishes its outputs itself */
struct mesh_function *mesh_call_bind(struct mesh_calls *,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,size_t dependency_count,
  const struct mesh_section *outputs,size_t output_count,const void *submit,void *context);
/* design/algorithm-sources.md#resident-metal: a resident function's completion word of slot `frame` (after prepare) */
uint64_t *mesh_function_completion(struct mesh_function *,uint32_t frame);
/* design/algorithm-sources.md#resident-metal: the publications of slot `index` of an output section (after prepare):
   its consumers' arrival cells, its SEND cells of that invocation (prepared-machine.md M10) and resident consumers'
   input words, each to be stored its argument plus the invocation's cycle (M30); their count, written where given */
uint32_t mesh_function_publications(struct mesh_function *,struct mesh_section,uint32_t index,struct prepared_publication *);
/* design/algorithm-sources.md#device-operands: the word a resident consumer of slot `index` of `section` waits on, 1 plus
   the cycle once the slot is filled: a received section's landing word (mesh_host_inputs, before), else a word its
   producer's publications store (asked before mesh_calls_prepare); NULL where no word could be leased */
uint64_t *mesh_section_input(struct mesh_calls *,struct mesh_section,uint32_t index);
/* the arrivals, landings, call records, completions and instances laid out, the constants sent; after both programs'
   mesh_transfers_prepare and mesh_host_inputs (mesh-collective.h), before their mesh_transfers_start */
int mesh_calls_prepare(struct mesh_calls *);
/* hot-path-loads M17: the workers started */
int mesh_calls_start(struct mesh_calls *);
/* hot-path-loads M41 */
struct mesh_submission *mesh_submission_at(struct mesh_calls *,uint32_t slot,struct mesh_instance **);
/* design/algorithm-sources.md#meshresult: the node a MESH_RESULT_LINK status's identity names (0: none) */
uint32_t mesh_calls_node(struct mesh_calls *,uint32_t identity);

/* ---- Routed readiness (mesh 45902d8e^ mesh_algebra_route_create, _attach, _producers and mesh_algebra_active, on
   this executor; design/bridge-dataflow.md#routed-readiness) ------------------------------------------------------
   A route is a directory a function of the program writes at run time (the caller's sort: its owners, ordinals and
   offsets, each a vector of uint32 at `offset` bytes into a slot of `section`, the slot the generation's index where
   the section has one a slot), over `candidate_count` candidate sections (each a partial a producer writes, or a
   received one) and `consumers` owners: candidate k is owned by owners[k] (MESH_ABSENT or >= consumers: none, its
   arrival discharged as it comes), and consumer c's candidates are ordinals[offsets[c] .. offsets[c + 1]), each owned by
   c.  An attached consumer function fires once the directory's sections and the candidates routed to it have arrived,
   and waits on no other candidate: the route is one of its arrivals, beside its own waited inputs (it binds the
   candidates it reads as inputs past its dependency count, unwaited).  A directory that breaks those rules fails the
   generation as a value (MESH_RESULT_FUNCTION, EPROTO) and fires its consumers so the records stay in step.  The sort
   and the segmented sums stay the caller's functions; this is their readiness alone.  Every event of a route runs on
   `worker`, every consumer's too. */
struct mesh_route_vector { struct mesh_section section; size_t offset; };
struct mesh_routing;
/* NULL (errno EINVAL) where a vector lies past its section's bytes, a count is 0 or past UINT32_MAX, or the program is
   running */
struct mesh_routing *mesh_route_create(struct mesh_calls *,uint32_t worker,struct mesh_route_vector owners,struct mesh_route_vector ordinals,
  struct mesh_route_vector offsets,const struct mesh_section *candidates,size_t candidate_count,size_t consumers);
/* function `consumer`-th owner of the route: 0, EINVAL where the function is resident, on another worker, or the
   consumer is attached already */
int mesh_route_attach(struct mesh_function *,struct mesh_routing *,size_t consumer);
/* each candidate's producer (NULL: none, a received candidate's), so a producer omitted past its count (mesh_call_active)
   discharges its candidate without a payload; a routed candidate whose producer was omitted fails the generation
   (EPROTO).  0, EINVAL where `count` is not the route's candidates */
int mesh_route_producers(struct mesh_routing *,struct mesh_function *const *producers,size_t count);
/* Megablox's bounded grid (design/active-segments.md at mesh e1acaf7e^): the function's call of a generation runs only
   where `slot` is below the count (a uint32 at `count.offset` of `count.section`, a section among the function's waited
   inputs, which a function of the program writes at run time); past it the call is omitted: nothing is submitted, its
   generation's event completes, and its publications (which must all be route candidates' arrivals: mesh_calls_prepare
   refuses others, EINVAL) are stored marked omitted.  0, EINVAL */
int mesh_call_active(struct mesh_function *,struct mesh_route_vector count,uint32_t slot);
void mesh_call_complete(struct mesh_call *,int error);
void mesh_call_fail(struct mesh_call *,int error);
void mesh_calls_destroy(struct mesh_calls *);
#endif

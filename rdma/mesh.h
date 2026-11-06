#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>
#include <os/os_sync_wait_on_address.h>
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
#define MESH_VERSION 121u
#define MESH_ABSENT UINT32_MAX
#ifndef MESH_PLATFORM_NS
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#endif
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
};
_Static_assert(sizeof(struct mesh_page_entry)==16 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8,"mesh_page_entry");
enum { MESH_SEND, MESH_RECEIVE };
struct mesh_target { uint64_t word; };
_Static_assert(sizeof(struct mesh_target)==8,"mesh_target");
/* Copy c: phase group at phases + c × mesh_group_bytes(pages); count word at count + 8c; send-completion counter at
   completion + 16c, its error count at completion + 16c + 8.
   rdma-core, libibverbs/man/ibv_post_send.3, NOTES: "The buffers used by a WR can only be safely reused after WR the
   request is fully executed and a work completion has been retrieved from the corresponding completion queue (CQ)."
   libfabric, man/fi_cntr.3.md, DESCRIPTION: "Counters record the number of requested operations that have completed.";
   fi_cntr_readerr: "Reads the number of operations which have completed in error." */
struct mesh_publication { _Alignas(64) uint32_t sends; uint32_t pages; uint64_t phases,count;
  struct mesh_target targets[1]; uint64_t completion; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 &&
  offsetof(struct mesh_publication,sends)==0 && offsetof(struct mesh_publication,pages)==4 &&
  offsetof(struct mesh_publication,phases)==8 && offsetof(struct mesh_publication,count)==16 &&
  offsetof(struct mesh_publication,targets)==24 && offsetof(struct mesh_publication,completion)==32,"mesh_publication");
/* `prepared`: M. M. Michael, Hazard pointers: safe memory reclamation for lock-free objects, IEEE TPDS 15(6), 2004 */
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };

#define MESH_PORTS 8u
struct mesh_holder { uint32_t pid,unit; uint64_t start; };
struct mesh_bridge_pages { uint32_t link[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS]; };
_Static_assert(sizeof(struct mesh_holder)==16 && offsetof(struct mesh_bridge_pages,bridge_start)==(MESH_PORTS+4)*sizeof(uint32_t),
               "mesh_bridge_pages");

/* A plan's entries at a node, struct mesh_state `entries` = u × n of them at its `hops`, copy c's n at [c n, c n + n),
   every copy the same hops in the same order; execution t runs copy t mod u (M. Lam, Software pipelining: an effective
   scheduling technique for VLIW machines, PLDI 1988, modulo variable expansion; B. R. Rau, Iterative modulo
   scheduling, MICRO 1994).  A hop moves bytes [page × pgsz + offset, page × pgsz + offset + bytes) of the arena, with
   offset + bytes ≤ block × pgsz.
   A receive of copy c lands in executions c, c + u, c + 2u, ...; after execution t's landing its field holds
   mesh_phase(⌊t / u⌋ + 1).  Execution t of received range g is complete when the count of copy t mod u reaches
   (⌊t / u⌋ + 1) × P_g.
   Both ends of a node pair hold the same hops in the same order, a subsequence of one global order: M. Cowan et al.,
   MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, §5.2, step 3: "Sort instructions into a global
   topological order with respect to their dependencies with a heap using the priority to order instructions"; "all
   implicit dependencies introduced by thread block sequential execution cannot produce cycles"; the priority each
   instruction's C̄_j (C.-Y. Chen, arXiv 2205.02474, 2022, LP (14); L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein,
   Math. Oper. Res. 22(3), 1997, "Schedule-by-C̄j"); F. B. Schneider, Implementing fault-tolerant services using the
   state machine approach, ACM Comput. Surv., 1990.
   A send's work completion adds 1 to its `count`, or 1 to `count` + 8 where it completed in error (libfabric,
   man/fi_cntr.3.md, fi_cntr_readerr; man/fi_endpoint.3.md, fi_cancel: "Canceling an operation causes the fabric
   provider to search for the operation and, if it is still pending, complete it as having been canceled."). */
struct mesh_hop { uint32_t page,node,direction,field; uint64_t word,count; uint32_t pages,bytes; uint32_t set,offset; };
_Static_assert(sizeof(struct mesh_hop)==48 && offsetof(struct mesh_hop,word)==16 && offsetof(struct mesh_hop,count)==24 &&
  offsetof(struct mesh_hop,pages)==32 && offsetof(struct mesh_hop,bytes)==36 && offsetof(struct mesh_hop,set)==40 &&
  offsetof(struct mesh_hop,offset)==44,"mesh_hop");
#define MESH_LANE_BYTES 16u
/* mesh_phase(k): k mod 4 in the Gray code 00 01 11 10 */
static inline uint32_t mesh_phase(uint64_t k){ k&=3; return (uint32_t)(k^(k>>1)); }
/* page k of a phase group: bits [2 (k mod 32), 2 (k mod 32) + 2) of word ⌊k / 32⌋ */
static inline uint64_t mesh_group_bytes(uint32_t pages){ return ((uint64_t)pages+63)/64*MESH_LANE_BYTES; }
struct mesh_status { _Atomic uint32_t behind,ahead,first,padding; };
_Static_assert(sizeof(struct mesh_status)==16 && offsetof(struct mesh_status,ahead)==4 &&
  offsetof(struct mesh_status,first)==8,"mesh_status");
/* JGroups ViewId (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/ViewId.java): `id` "Lamport
   time of the view"; `sequencer` and `boot` the "Address of the creator of this view", as JGroups' UUID
   (src/org/jgroups/util/UUID.java): "Logical address which is unique over space and time" */
struct mesh_view_id { uint64_t id; uint32_t sequencer,padding; uint64_t boot; };
_Static_assert(sizeof(struct mesh_view_id)==24 && offsetof(struct mesh_view_id,sequencer)==8 &&
  offsetof(struct mesh_view_id,boot)==16,"mesh_view_id");
/* ViewId.compareTo: "The comparison is done on the IDs, if they are equal, we use the creator", `id > other.id ? 1 :
   id < other.id ? -1 : creator.compareTo(other.creator)` */
static inline int mesh_view_compare(struct mesh_view_id a,struct mesh_view_id b){
  return a.id>b.id?1:a.id<b.id?-1:a.sequencer>b.sequencer?1:a.sequencer<b.sequencer?-1:
    a.boot>b.boot?1:a.boot<b.boot?-1:0;
}
/* - `counts`: received range g's copy c at index g × copies + c.
   - `completions`: range g's copy c the pair at index g × copies + c, its count then its error count.
   - `executions`: MPI 4.1 §3.9 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node83.htm), MPI_START: "The
     request becomes active once the call is made."
   - `sender`, `submitted`: X. Défago, A. Schiper and P. Urbán, Total order broadcast and multicast algorithms: taxonomy
     and survey, ACM Comput. Surv. 36(4), 2004, §7.1: "a sender sends m to the sequencer".
   - `submissions`, the queue at `submits` (entry k at k mod `submit_room`): L. Lamport, Time, clocks, and the ordering
     of events in a distributed system, CACM 21(7), 1978, IR1: "Each process Pi increments Ci between any two
     successive events".
   - `room`, `submit_room`: S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor
     DSP systems, IEEE Trans. Signal Processing 45(6), 1997, Theorem 2: "Bfb(e) = ρ_Gs(snk(e), src(e)) + delay(e)".
   - `arrived`: J. M. Mellor-Crummey and M. L. Scott, ACM TOCS 9(1), 1991, §3.1.
   - `waits`, `tells`, `syncs`, `rows`: the waits into the call at place a are entries rows[a] .. rows[a + 1] − 1.
   - (`serial`, `sequencer`, `boot`) is a struct mesh_view_id (mesh_plan_view).
   - `removed`: Documentation/RCU/whatisRCU.rst: "The removal phase removes references to data items within a data
     structure". */
struct mesh_state { _Alignas(64) uint32_t copies; uint32_t receives,ranges,releases;
  uint64_t phases,counts,supplied,status; _Atomic uint64_t executions; uint64_t completions;
  uint64_t group,epoch,serial,hops,entries; _Atomic uint64_t held; _Atomic int32_t refused[MESH_PORTS];
  uint64_t nodes; uint32_t sender,outstanding; uint64_t submitted; _Atomic uint64_t delivered;
  _Atomic uint64_t submissions; uint64_t broadcast,arrived,room,deliveries,submits,announced;
  uint32_t waits,tells; uint64_t syncs; uint32_t calls; _Atomic uint32_t closed; uint64_t rows,submit_room,transfers;
  uint32_t sequencer; _Atomic uint32_t removed; uint64_t boot,padding3[3]; };
_Static_assert(sizeof(struct mesh_state)==320 && _Alignof(struct mesh_state)==64 &&
  offsetof(struct mesh_state,receives)==4 && offsetof(struct mesh_state,ranges)==8 && offsetof(struct mesh_state,releases)==12 &&
  offsetof(struct mesh_state,phases)==16 && offsetof(struct mesh_state,counts)==24 &&
  offsetof(struct mesh_state,supplied)==32 && offsetof(struct mesh_state,status)==40 &&
  offsetof(struct mesh_state,executions)==48 && offsetof(struct mesh_state,completions)==56 &&
  offsetof(struct mesh_state,group)==64 && offsetof(struct mesh_state,epoch)==72 && offsetof(struct mesh_state,serial)==80 &&
  offsetof(struct mesh_state,hops)==88 && offsetof(struct mesh_state,entries)==96 && offsetof(struct mesh_state,held)==104 &&
  offsetof(struct mesh_state,refused)==112 && offsetof(struct mesh_state,nodes)==144 && offsetof(struct mesh_state,sender)==152 &&
  offsetof(struct mesh_state,outstanding)==156 && offsetof(struct mesh_state,submitted)==160 &&
  offsetof(struct mesh_state,delivered)==168 && offsetof(struct mesh_state,submissions)==176 &&
  offsetof(struct mesh_state,broadcast)==184 && offsetof(struct mesh_state,arrived)==192 &&
  offsetof(struct mesh_state,room)==200 && offsetof(struct mesh_state,deliveries)==208 &&
  offsetof(struct mesh_state,submits)==216 && offsetof(struct mesh_state,announced)==224 &&
  offsetof(struct mesh_state,waits)==232 && offsetof(struct mesh_state,tells)==236 &&
  offsetof(struct mesh_state,syncs)==240 && offsetof(struct mesh_state,calls)==248 &&
  offsetof(struct mesh_state,closed)==252 && offsetof(struct mesh_state,rows)==256 &&
  offsetof(struct mesh_state,submit_room)==264 && offsetof(struct mesh_state,transfers)==272 &&
  offsetof(struct mesh_state,sequencer)==280 && offsetof(struct mesh_state,removed)==284 &&
  offsetof(struct mesh_state,boot)==288,"mesh_state");
static inline struct mesh_view_id mesh_plan_view(const struct mesh_state *s){
  return (struct mesh_view_id){.id=s->serial,.sequencer=s->sequencer,.boot=s->boot};
}
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal
   Processing 45(6), 1997, §9, Fig. 6 (Convert-to-SC-graph), Fig. 8 (DetermineDelays), (7): "start(vi, k) ≥ end(vj, k −
   delay((vj, vi)))".  A wait: base = position + 1 − delay × every, and start k of the plan, K = ⌊k / calls⌋, waits on
   each entry of its row k mod calls until count ≥ K × every + base over ℤ.  A tell: `entry` its edge's index among the
   sink plan's waits; every = base = 0. */
struct mesh_sync { uint32_t node,function,entry,every; int64_t base; _Atomic uint64_t count; };
_Static_assert(sizeof(struct mesh_sync)==32 && offsetof(struct mesh_sync,entry)==8 && offsetof(struct mesh_sync,every)==12 &&
  offsetof(struct mesh_sync,base)==16 && offsetof(struct mesh_sync,count)==24,"mesh_sync");
/* M. Cowan et al., MSCCLang, ASPLOS 2023, §5: "a tree data structure that
   divides a collective into individual GPU programs".
   - `published`: P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998 (rcu_assign_pointer).
   - `published`, `retired`, `released`: each one writer's (L. Lamport, On interprocess communication, Distributed
     Computing 1(2), 1986).
   - `table`: word k the region offset of function k + 1's state section.
   - `adjacent`: n + 1 words first[0 .. n], then the places; map index i's places are words [first[i], first[i + 1])
     after them. */
struct mesh_algo { _Alignas(64) _Atomic uint64_t published; _Atomic uint64_t retired,released; uint32_t plans,nodes;
  uint64_t group,table,adjacent,padding1; };
_Static_assert(sizeof(struct mesh_algo)==64 && offsetof(struct mesh_algo,retired)==8 &&
  offsetof(struct mesh_algo,released)==16 && offsetof(struct mesh_algo,plans)==24 && offsetof(struct mesh_algo,nodes)==28 &&
  offsetof(struct mesh_algo,group)==32 && offsetof(struct mesh_algo,table)==40 &&
  offsetof(struct mesh_algo,adjacent)==48,"mesh_algo");
/* E. A. Lee and D. G. Messerschmitt, Static scheduling of synchronous data flow programs for digital signal processing,
   IEEE Trans. Computers C-36(1), 1987: "We can replace each arc with a FIFO queue (buffer) to pass data from one block
   to another".  The plan's k-th delivery at a node is entry k mod `room` of its queue at `deliveries`. */
struct mesh_delivery { uint64_t sequence,first,end,padding; };
_Static_assert(sizeof(struct mesh_delivery)==32,"mesh_delivery");
struct mesh_submission { uint64_t end,clock; };
_Static_assert(sizeof(struct mesh_submission)==16,"mesh_submission");
/* `delivered`: X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4), 2004, Fig. 5: "while ∃(m′, seqnum′) ∈
   pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver := nextdeliver + 1"; `delivered` = nextdeliver − 1.
   (`view`, `view_sequencer`, `view_boot`) is a struct mesh_view_id (mesh_view_installed). */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,links,wire_pages;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,target_off,data_off,length,target_stride;
  _Atomic uint64_t client,bridge_pid,serial,control;
  _Atomic uint64_t view_map,view;
  uint64_t view_off;
  _Atomic uint64_t view_nodes;
  _Atomic uint64_t map,map_off,map_nodes;
  _Atomic uint64_t map_serial;
  uint64_t bridge_off;
  _Atomic uint64_t client_start;
  uint64_t boot;
  _Atomic uint64_t health,busy,idle_since,stopping;
  uint64_t map_room;
  uint64_t algo_off;
  _Atomic uint64_t plans;
  uint64_t deliveries_off,deliveries_room;
  _Atomic uint64_t delivered;
  _Atomic uint32_t view_sequencer;
  _Atomic uint64_t view_boot;
};
static inline struct mesh_view_id mesh_view_installed(struct hdr *m,memory_order order){
  return (struct mesh_view_id){.id=atomic_load_explicit(&m->view,order),
                               .sequencer=atomic_load_explicit(&m->view_sequencer,order),
                               .boot=atomic_load_explicit(&m->view_boot,order)};
}
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_blocks(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz)); }
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return mesh_blocks(m)*m->block; }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
static inline struct mesh_hop *mesh_hops(struct hdr *m,const struct mesh_state *state){
  return (struct mesh_hop *)((char *)m+state->hops);
}
static inline struct mesh_state *mesh_state_at(struct hdr *m,uint64_t at){
  return (struct mesh_state *)((char *)m+at);
}
static inline struct mesh_algo *mesh_algo(struct hdr *m){ return (struct mesh_algo *)((char *)m+m->algo_off); }
static inline struct mesh_state *mesh_algo_state(struct hdr *m,const struct mesh_algo *algo,uint32_t k){
  return mesh_state_at(m,((const uint64_t *)(const void *)((const char *)m+algo->table))[k]);
}
/* P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998: the publication read (acquire) before every other
   word of the operand (rcu_dereference) */
static inline uint64_t mesh_algo_unreleased(struct hdr *m){
  const struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=atomic_load_explicit(&algo->published,memory_order_acquire);
  return published && atomic_load_explicit(&algo->released,memory_order_acquire)!=published?published:0;
}
static inline uint64_t mesh_algo_open(struct hdr *m){
  const uint64_t published=mesh_algo_unreleased(m);
  return published && atomic_load_explicit(&mesh_algo(m)->retired,memory_order_acquire)!=published?published:0;
}
static inline struct mesh_state *mesh_algo_standing(struct hdr *m,uint32_t k){
  struct mesh_state *state=mesh_algo_state(m,mesh_algo(m),k);
  return !atomic_load_explicit(&state->closed,memory_order_acquire)?state:NULL;
}
static inline const uint32_t *mesh_algo_adjacent(struct hdr *m,uint32_t i,uint32_t *count){
  const struct mesh_algo *algo=mesh_algo(m);
  const uint32_t *first=(const uint32_t *)(const void *)((const char *)m+algo->adjacent);
  *count=first[i+1]-first[i];
  return first+algo->nodes+1+first[i];
}
static inline const struct mesh_publication *mesh_transfer_at(struct hdr *m,const struct mesh_state *state,
                                                              uint32_t transfer){
  return (const struct mesh_publication *)(const void *)((const char *)m+state->transfers+(size_t)transfer*64);
}
static inline struct mesh_sync *mesh_syncs(struct hdr *m,const struct mesh_state *state){
  return (struct mesh_sync *)(void *)((char *)m+state->syncs);
}
static inline int mesh_seqlock_whole(const _Atomic uint64_t *lock,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(lock,memory_order_relaxed)==before;
}
static inline int mesh_map_whole(struct hdr *m,uint64_t before){ return mesh_seqlock_whole(&m->map,before); }
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off);
}
#endif

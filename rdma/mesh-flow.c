/* The bridge's data path, in a port's child (mesh-child.c): one cable and its queue pairs (the tag's lanes, the link's
   `plan`), serving this node's share of the region's page lists (mesh.h struct mesh_hop) by the rule both ends compute
   (design/page-queue-change-request.md, "Claude's proposal": "The page lists (the queue)", "What the bridge decides",
   "The rule both ends compute", "The functions"; and the operator's "The page queue" and "Interleaving every
   transfer": every inbound link used, a ready page waiting only behind earlier pages of its own queue pair), as
   design/disjoint-pages-change-request.md, Appendix C, C3, changes it (design/disjoint-pages-revision-coordination.md,
   "The bridge"): a plan is u copies of one execution's hops, its template, execution t running copy t mod u, every
   page placed once by the plan's one allocator (C3.3, modulo variable expansion: M. Lam, Software pipelining, PLDI
   1988), so it runs any number of executions, the client's start saying how many (the plan's state word `executions`,
   which only rises: the contract's D2); and its state lives apart from its pages, in the plan's state section (mesh.h
   struct mesh_state; C3.7), no word of it written to an earlier value: a landing flips its receive's 2-bit Gray-coded
   phase (PTX's mbarrier phase bit, widened to two) and adds one to its range's count of its copy (counted completion,
   C3.1: Portals 4's counting events; the contract's D1), and a cancellation arrives on that count with no phase written
   (C3.1).  A send of execution t is released by its range's supplied word reading at least t + 1, or, forwarding, by
   its receive's phase of execution t.  Nothing here writes a data page.
   Its native forms are the d6dceac transport's (MESH_VERSION 104, restored 2026-10-01: the source of the fastest
   recorded E2B decode, metal-microbench docs/measurement.md): SENDs prepared once a plan, each posted in its queue
   pair's order as its word releases it; receives prepared once a plan, posted in their queue pair's order as it has
   room, a receive record again only once its last posting has landed; a record of either posted once for each
   execution of its copy (link_prepare, link_receive_drain); completions polled through the
   provider's own functions (mesh-verbs.h struct mesh_verbs `poll`, struct mesh_queue `send`, `receive`); traced
   variants under MESH_LEDGER.  Their inputs are the page lists, which replaced d6dceac's per-link, per-queue transfer
   lists and its --link form's link_configure (both since removed).  The progress thread, the guard around its device's
   verbs calls and a port's halts are tag pre-return-20261103 rdma/mesh-link.c's, verbatim but at interfaces.  No main
   (the binary's is mesh-link.c's, the child's mesh-child.c's) and no verbs teardown: a halt clears `served` and moves
   `pairing`, and the child's exit is the teardown (mesh-child.c child_leave, lanes_flush; down_pair stays deleted).
   2026-11-04, sub-assignment 6: reversing this note's 2026-11-03 form (every execution's entries unrolled into the
   list, a landing's bit ORed into its execution's landed record, a cancellation UINT64_MAX in that record's cancel
   word), which unrolled by the run's length and gave the landed record five writers (C1; C4).  What must stay: the
   k-th SEND on a queue pair fills the k-th receive posted on it; nothing is written in or beside a page; no request is
   taken back but by the child's exit.  mesh-dataflow.h is no longer included: mesh_cables, the one thing this file
   took from it, is mesh-map.h's (through mesh-bridge.h).
   2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.5 and D.4; design/disjoint-pages-rulings-
   coordination.md, R1): reversing this note's receives "posted ahead as their queue pair has room", under which a
   receive record was posted again while an earlier posting of it might be outstanding: a received page is now named
   once per landing.  What must stay: as above */
#include "mesh-bridge.h"
#include "mesh-graph.h"

/* design/prepared-machine.md#M29: d6dceac's prepared SEND (its span and request, one cache line's worth of what the
   post reads), its release its entry's `word` and `field` (mesh.h struct mesh_hop; design/disjoint-pages-change-
   request.md, C3.7): from its range's source (`field` MESH_ABSENT), execution t's send released where the range's
   supplied word reads (acquire) at least t + 1, its producer's release of t; forwarding (a relay's, or a destination's
   that sends the page on), where its receive's phase field, bits [field, field + 2) of `word`, reads (acquire)
   mesh_phase(t / u + 1), that receive's landing of t; t the execution its lane's cursor serves (struct cursor).  The
   word is the plan's state section's, the record the child's.  One record a template entry, posted once for each
   execution of its copy (link_prepare).
   2026-11-04, sub-assignment 6: `mask` (2026-11-03: the bits of its word its release needed, MESH_SUPPLIED or its
   receive's bit of its execution's landed record) is `field`, the landed records and MESH_SUPPLIED gone (C4).  What
   must stay: the word and field in the line the post reads */
struct prepared_send {
  _Alignas(128) _Atomic uint64_t *word;
  uint32_t field,padding;
  struct ibv_sge span;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct prepared_send)==256 && _Alignof(struct prepared_send)==128 &&
  offsetof(struct prepared_send,field)==8 && offsetof(struct prepared_send,span)==16 &&
  offsetof(struct prepared_send,request)==32 &&
  offsetof(struct prepared_send,request.send_flags)+sizeof(unsigned int)<=128,"M29");
/* design/prepared-machine.md#M08: d6dceac's prepared receive, its request and span verbatim, and what its landing
   writes (design/disjoint-pages-change-request.md, C3.1 and C3.7; design/disjoint-pages-revision-coordination.md, "The
   bridge"): its phase field, bits [field, field + 2) of `word`, and `count`, its range's count word of its copy, which
   takes the range's `pages` arrivals an execution, NULL where its entry's `count` is 0 (a relay's own receive: its
   forwards are released by its phase and nobody waits on a count of it, so it adds, wakes and is raised for nothing;
   2026-11-04, the coordinator's ruling on review 15's finding 1); `landings` the record's completions since the load, the progress
   thread's alone (its j-th is execution copy + j u: link_prepare), which also holds back its next posting: its round
   j is posted only once `landings` reads j, so at most one posting of it is outstanding and its page is named by at
   most one posted receive (link_receive_drain; 2026-11-04, sub-assignment 1: design/disjoint-pages-change-request.md,
   D.3.5, design/disjoint-pages-rulings-coordination.md, R1, reversing its posting again while an earlier posting
   might be outstanding; what must stay: the record's request never changed once written); and `copy` its copy.  Every
   field its landing reads is in its second 64 bytes, the 128-byte line its request starts.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's `input`, `mask` and `cancel` (its bit of its execution's landed
   record, ORed in, and that record's cancel word, stored UINT64_MAX by link_close: C4); `next` (its queue pair's chain
   of receives each posted once) and `pair` gone, the lane's cursor walking its records (struct cursor) and the queue
   pair the lane's.  What must stay: the record one line, its request never changed once link_prepare has written it */
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Alignas(64) _Atomic uint64_t *word;
  _Atomic uint64_t *count;
  uint64_t landings;
  uint32_t field,pages,copy,padding;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,word)==64 &&
  offsetof(struct prepared_receive,count)==72 && offsetof(struct prepared_receive,landings)==80 &&
  offsetof(struct prepared_receive,field)==88 && offsetof(struct prepared_receive,pages)==92 &&
  offsetof(struct prepared_receive,copy)==96,"M08 M09");
/* design/prepared-machine.md#M27 */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
_Static_assert(sizeof(struct mesh_trace)==32 && _Alignof(struct mesh_trace)==32,"M27");
/* A lane's place in the executions it serves one way (design/disjoint-pages-revision-coordination.md, "The bridge":
   each lane in execution order, execution t's records on it copy t mod u's, in assignment order): execution t
   (`execution`), its copy t mod u (`copy`) and t / u (`round`), and the records of that copy on the lane still to post
   in t, indexes [at, end) of its direction's records.  Kept in registers through a pass (link_receive_drain,
   link_send_drain), so a post costs no division */
struct cursor { uint64_t execution,round,at,end; uint32_t copy,padding; };
/* A port's plan as its child serves it, the progress thread's alone: none loaded; refused (the view `hops`, `plan` it
   refused, until the plan call writes again); served; done (every record served every execution below the plan's
   `executions` as link_run last read it: every SEND completed, every receive landed; served again where that word
   rises); or ended with work outstanding on the link's queue pairs, which only the child's exit releases
   (LINK_STREAMS_ENDED), so nothing more is loaded in the child's life.  The plan's state (`words`, its `copies` u), its
   records by lane, copy and assignment order (lane q's copy c at [first[q (u + 1) + c], first[q (u + 1) + c + 1]) of
   its direction's), each lane's cursor each way, the lanes that carry each way (`carrying`, `lanes` either), the
   executions last read and the completions they ask (`send_total`, `receive_total`) against those taken (`sent`,
   `landed`), and the traces (MESH_LEDGER) */
enum { FLOW_NONE, FLOW_REFUSED, FLOW_SERVING, FLOW_DONE, FLOW_ENDED };
struct flow {
  int state,ledger,refused;
  uint32_t lanes,carrying[2],copies;
  uint64_t hops,plan,client,executions,send_count,receive_count,send_total,receive_total,sent,landed;
  struct mesh_state *words;
  struct prepared_send *sends;
  struct prepared_receive *receives;
  uint64_t *first[2];
  struct cursor cursor[2][MESH_LINK_LANES];
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
};
static struct flow flows[MESH_PORTS];

/* tag pre-return-20261103 rdma/mesh-link.c:56-65 port_halt (3f72cbde, 306b22be), verbatim but not static */
/* design/transport.md: a port halted from any thread (a peer interface's termination heard, a provider's fault
   absorbed; atomics alone): its pairing lost, so no job of it makes a further call (B10), then no lane of its link
   served, both sequentially consistent (link_turn's connect stores served and reads the pairing again), and its bit set
   in `which` for the event loop's conclusion (terminations_turn, ports_stalled) */
_Static_assert(MESH_PORTS<=32,"a bit a port");
void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_fetch_or(which,UINT32_C(1)<<port->index);
}
/* tag pre-return-20261103 rdma/mesh-link.c:71-81 lane_lost (d556370e), verbatim */
/* a lane's failure, found by a progress thread: served no more, and the link lost by the event loop on reading
   `broken` (link_turn) */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    mesh_say("bridge","link %u (%s): lane %u failed (code %lld): the link toward node %u lost",port->index,port->name,lane,
             (long long)code,port->peer);
  atomic_store_explicit(&l->lost,code,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}
/* tag pre-return-20261103 rdma/mesh-link.c:275-286 link_failed (d556370e), verbatim */
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: no lane of the
   link served from here, so nothing is posted on it again, and the link lost by the event loop on reading `broken`
   (link_turn), the stream's words ~0 (port_lost) */
static void link_failed(struct port *port,int64_t code){
  struct link *l=&port->link;
  const uint32_t served=atomic_exchange_explicit(&l->served,0,memory_order_acq_rel);
  if(served)mesh_say("bridge","link %u (%s): a completion or its poll failed (code %lld): no lane served, the link toward node %u lost",
                     port->index,port->name,(long long)code,port->peer);
  atomic_store_explicit(&l->lost,code,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c:79-103 mesh_records_read (e3c4c986), adapted: the region the bridge's
   own (M), the map's serial read inside the same reading, an error value in place of -errno, and its deadline the
   bridge's clock (main's mesh.h has no struct mesh_deadline); the bridge that writes the map is this child's parent, so
   a write that stays open past MESH_PLATFORM_NS is ETIMEDOUT, not the tag's ESRCH */
/* the map's records as one serial left them, the seqlock's reader (mesh_map_whole): a copy in *out the caller frees,
   its count and the map's serial with it, or an error value; the read within its deadline, MESH_PLATFORM_NS from its
   start */
static int map_read(struct mesh_record **out,uint64_t *count,uint64_t *serial){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(;;){
    const uint64_t before=atomic_load_explicit(&M->map,memory_order_acquire),at=clock_ns();
    if(at>=until)return ETIMEDOUT;
    if(before&1){
      const uint64_t left=until-at;
      os_sync_wait_on_address_with_timeout(&M->map,before,sizeof M->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,left<MESH_PAUSE_NS?left:MESH_PAUSE_NS);
      continue;
    }
    const uint64_t n=atomic_load_explicit(&M->map_nodes,memory_order_relaxed),off=atomic_load_explicit(&M->map_off,memory_order_relaxed);
    const int fits=off<=M->length && n<=(M->length-off)/sizeof(struct mesh_record);
    struct mesh_record *copy=fits?malloc((n?n:1)*sizeof *copy):NULL;
    if(copy)memcpy(copy,(char *)M+off,n*sizeof *copy);
    const uint64_t s=atomic_load_explicit(&M->map_serial,memory_order_relaxed);
    if(mesh_map_whole(M,before)){
      if(!copy)return fits?ENOMEM:EINVAL;
      *out=copy;*count=n;*serial=s;return 0;
    }
    free(copy);
  }
}
/* a node's index in the map: its place in ascending node-id order among the map's records (mesh.h hdr `map_serial`);
   UINT32_MAX where the map holds no record of it */
static uint32_t map_index(const struct mesh_record *records,uint64_t count,uint32_t node){
  uint32_t index=0,held=0;
  for(uint64_t r=0;r<count;r++){index+=records[r].node<node;held|=records[r].node==node;}
  return held?index:UINT32_MAX;
}
static inline _Atomic uint64_t *region_word(uint64_t offset){ return (_Atomic uint64_t *)(void *)((char *)M+offset); }
/* 2026-11-04, sub-assignment 6: landed_cancel (2026-11-03: an execution's cancel word, its landed record's word 0,
   found from a word of the record and hdr `landed_stride`) is gone with the landed records and the cancel word (C4);
   a cancellation is count_raise's */
/* the executions t < `executions` with t mod u == `copy` (u `copies`): those a record of that copy serves */
static inline uint64_t copy_executions(uint64_t executions,uint32_t copies,uint32_t copy){
  return executions>copy?(executions-copy-1)/copies+1:0;
}
/* A cancellation's arrival (design/disjoint-pages-change-request.md, C3.1: "Landings and cancellations both arrive on
   the count, as arrivals do on an mbarrier"; design/disjoint-pages-revision-coordination.md, "The bridge"): the count
   word raised to `final` and never lowered, by a compare-and-swap (release) that stores only above what it read and is
   tried again only where another arrival moved the word between (so it ends within the arrivals still to come, a
   range's P an execution), then the word's waiters woken (os_sync_wake_by_address_all, 8 bytes, region-shared: the
   client's waiter's os_sync_wait_on_address_with_timeout, mesh-metal.m) where this call raised it.  A count at or past
   `final` is left as it is, so several records or entries of one range raise its word once.  No phase is written:
   the first execution the receive did not land reads its field one phase behind (mesh.h, "Phases") */
static void count_raise(_Atomic uint64_t *count,uint64_t final){
  uint64_t seen=atomic_load_explicit(count,memory_order_relaxed);
  while(seen<final && !atomic_compare_exchange_weak_explicit(count,&seen,final,memory_order_release,memory_order_relaxed)){}
  if(seen<final)os_sync_wake_by_address_all(count,sizeof *count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* a load's refusal of a plan none writes (link_prepare's EINVAL), or EAGAIN where `hops` or `plan` moved since the load
   read them (the plan read whole, as at link_prepare's last check: an acquire fence after what it read, then both
   again), so a read torn by a plan written again is never refused */
static int plan_refused(uint64_t hops,uint64_t plan,int error){
  atomic_thread_fence(memory_order_acquire);
  return atomic_load_explicit(&M->hops,memory_order_acquire)!=hops || atomic_load_explicit(&M->plan,memory_order_relaxed)!=plan?
    EAGAIN:error;
}
/* a lane's cursor at the next execution, on that execution's copy's records on the lane (`first` the lane's u + 1
   bounds, struct flow) */
static inline void cursor_step(struct cursor *at,const uint64_t *first,uint32_t copies){
  at->execution++;
  if(++at->copy==copies){at->copy=0;at->round++;}
  at->at=first[at->copy];at->end=first[at->copy+1];
}
/* the executions to serve, the plan's word as link_run read it, and the completions they ask each way: every record of
   copy c once for each execution t < executions with t mod u == c (copy_executions) */
static void flow_executions(struct flow *f,uint64_t executions){
  uint64_t total[2]={0,0};
  const uint64_t width=(uint64_t)f->copies+1;
  for(int d=0;d<2;d++)for(uint32_t q=0;q<MESH_LINK_LANES;q++){
    const uint64_t *first=f->first[d]+q*width;
    for(uint32_t c=0;c<f->copies;c++)total[d]+=(first[c+1]-first[c])*copy_executions(executions,f->copies,c);
  }
  f->executions=executions;f->send_total=total[MESH_SEND];f->receive_total=total[MESH_RECEIVE];
}

/* The rule both ends compute (design/page-queue-change-request.md; mesh.h struct mesh_hop, "Both ends"), and this
   port's share of the plan built from it.  The plan written (hdr `plan` its generation) is refused (ESTALE) where its
   entries' `serial` is not the map's serial as one whole reading of the map left it.  The cables between this node and
   the port's peer are the reach entries both records list for each other (each from its own side, the ports crossed),
   one cable an entry of each (mesh-map.h mesh_cables, the one per-cable join); a cable's key is (the lower node id's
   port index, the higher's), cables in key order; its bandwidth the lesser of the two records' and its queue pairs the
   lesser of the two (the pairing's, the same at both ends).  The pair's links are each cable's queue pairs
   0 .. qps - 1 in (key, queue pair) order, each of rate the cable's bandwidth over its queue pairs.  The plan's list
   is its template, u copies of one execution's n hops (`hops` = u n, copy c's at [c n, c n + n); u the plan's state's
   `copies`, mesh.h struct mesh_state, read after `hops`: design/disjoint-pages-change-request.md, C3.3).  Each
   direction's hops of the pair (the template's entries naming the peer's index, in list order: copy 0's, then copy
   1's, ...) are its units in that order, every one a page (block x pgsz bytes), assigned by mesh_graph_list_schedule
   (Graham's list scheduling, mesh-graph.h): this node's SENDs to the peer are the peer's receives from this node, and
   u is the collective's (the contract's D5: every rank computes the same), so both ends assign one list alike.  Kept:
   the units on this port's cable, each on its queue pair (lane q, link_plan), one record each, by lane, copy and
   assignment order (struct flow `first`).  A lane serves the executions in order, execution t's records on it copy
   t mod u's in assignment order (struct cursor), so in every execution the k-th SEND on a queue pair fills the k-th
   receive posted on it.  A record is posted once for each execution of its copy, and a receive record again only once
   its last posting has landed (link_receive_drain: its round j, j = t / u, only once its `landings` reads j, the
   lane's posting stopping at the first record that has not), so a page is named by at most one posted receive at any
   moment and enters its list again only once its last landing has landed (design/disjoint-pages-change-request.md,
   D.2, "How many pages are in flight at once", and D.3.5; design/disjoint-pages-rulings-coordination.md, R1).  A queue
   pair's receives complete in the order posted (Hilland et al., RDMA Protocol Verbs Specification, section 8.2.2:
   design/prepared-machine.md#M08), so a receive record's j-th completion since the load is execution copy + j u
   (`landings`).  A posting after the last landing is in time: the hardware holds a SEND until its receive is posted
   (the data model, section 3), and the peer's SEND of execution t + u follows this node's reads of t, which follow t's
   landing, by the allocator's order (C3.3), not by anything here.  A SEND sends its page once released
   (link_send_drain); a relay is no special case, its SEND naming its receive's word and field.  A receive lands in its
   page and writes its phase and its count alone (link_receive_drain).  No verbs call: the progress thread posts as it
   serves (link_receive_progress, link_send_progress).  0, or an error value: ESTALE, ETIMEDOUT or ENOMEM reading the
   map, EINVAL where the plan has no state, its copies do not divide the list, or an entry's `field` or `pages` is none
   a plan writes (mesh.h struct mesh_hop: a field's low bit even and below 64, the pages of a receive with a count at
   least 1), and
   `hops` and `plan` still read as loaded (else EAGAIN: plan_refused), EALREADY where this port carries units of a
   plan whose run has begun (its `executions` above 0 at the load: the note at the refusal, below),
   EHOSTUNREACH where the pair has units but no link of rate, EPROTO where the map gives this cable a queue pair its
   link does not hold, EMSGSIZE where a page is more frames than a lane it is assigned can hold; or EAGAIN, no refusal,
   where the plan was written again while it was read.  On 0 the port's `prepared` is the plan's generation, stored
   before the last check (below); on an error it is as it was, 0.
   2026-11-04, sub-assignment 6 (design/disjoint-pages-change-request.md, C1, C3.3, C4): reversing 2026-11-03's list of
   every execution's entries (executions x n, hdr `executions`), one record an entry, each posted once, so the records
   and the load's work grew with the run's length; now one record a template entry, posted once for each execution of
   its copy, so they are the template's whatever the run's length.  What must stay: both ends pick and assign the same
   u x n list in list order; each lane's executions in order; a record's request unchanged once written here.
   2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.5 and D.4; design/disjoint-pages-rulings-
   coordination.md, R1): reversing sub-assignment 6's note here, that a record was posted again while an earlier
   posting of it might be outstanding, relying on the provider taking a request at its post (its tbt_post_recv read
   from the disassembly, never documented), so that two postings could name one page.  Why: nothing required it; it
   gained nothing over the u rounds the copies give, and named one page in two transfers at once (D.2).  What must
   stay: the k-th SEND on a queue pair fills the k-th receive posted on it (both ends' one assignment, in list order);
   each lane's executions in order; no request taken back but by the child's exit.
   2026-11-03, sub-assignment 6 (design/disjoint-pages-change-request.md, "Claude's proposal" 1 and 3; Appendix A, A4):
   the ring went from the bridge: plan_shape (hdr `depth`, a slot's hops, execution t's page the slot t mod depth's
   and its word the cell at word + t hop_stride), the receive window (a receive of t posted once its hop's executions
   0 .. t - window had landed: struct flow_hop, landed_through, `need`) and the coordinator's forwards gate of the same
   day (a relay page's receive of t posted, from t = depth, once the page's forwarding SENDs of t - depth had completed,
   each completion adding 1 at offset 8 of its landed word's cell: `sent`, `forwarding`), with the per-page cells both
   read.  Why: they reused `depth` copies of a page under a wait before each execution, which nothing asked for; the
   plan's one allocator gives each execution's values their own pages, sharing a page only where the plan's dependence
   graph orders the last read (a consumption, or a relay's forward) before the next write (a landing), so a receive
   needs no landing or forward of another execution before it is posted.  What must stay: the k-th SEND on a queue pair
   fills the k-th receive posted on it (both ends' one assignment, in list order); a landing is never overwritten while
   it is read, by the allocator's order, not a count here; and no request is taken back but by the child's exit
   (link_close) */
static int link_prepare(struct port *port){
  struct link *l=&port->link;
  struct flow *f=flows+port->index;
  const uint64_t hops=atomic_load_explicit(&M->hops,memory_order_acquire);
  const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_relaxed);
  f->hops=hops;f->plan=plan;f->client=atomic_load_explicit(&M->client,memory_order_acquire);
  /* `hops` 0 since link_run read it: the plan call's first store, so the plan is being written again (EAGAIN) */
  if(!hops)return EAGAIN;
  /* the plan's state, written before `hops` (mesh.h, "Writing a plan", step 5): its copies divide the list */
  struct mesh_state *words=mesh_plan_state(M);
  const uint32_t copies=words?words->copies:0;
  if(!copies || hops%copies)return plan_refused(hops,plan,EINVAL);
  const uint64_t n=hops/copies,width=(uint64_t)copies+1;
  const struct mesh_hop *list=mesh_hops(M);
  struct mesh_record *records=NULL;uint64_t count=0,serial=0;
  int error=map_read(&records,&count,&serial);
  if(error)return error;
  if(list[0].serial!=serial){free(records);return ESTALE;}
  const uint32_t self=M->node,peer=port->peer,index=map_index(records,count,peer);
  const struct mesh_record *mine=NULL,*theirs=NULL;
  for(uint64_t r=0;r<count;r++){if(records[r].node==self)mine=records+r;if(records[r].node==peer)theirs=records+r;}
  /* the pair's cables, the one per-cable join (mesh-map.h mesh_cables, which mesh_joined and mesh_nodes_read read too:
     C9), in key order; this port's the one whose key names it at this node's end */
  struct mesh_cable cables[MESH_PORTS];
  const uint32_t cable_count=mine && theirs?mesh_cables(mine,theirs,cables):0,end=self<peer?0:1;
  free(records);
  double rate[MESH_PORTS*MESH_LINK_LANES];
  uint32_t on[MESH_PORTS*MESH_LINK_LANES],queue[MESH_PORTS*MESH_LINK_LANES],links=0;
  for(uint32_t c=0;c<cable_count;c++){
    if(cables[c].qps>MESH_LINK_LANES)return EPROTO;
    for(uint32_t q=0;q<cables[c].qps;q++){
      rate[links]=(double)cables[c].bandwidth/(double)cables[c].qps;on[links]=c;queue[links]=q;links++;
    }
  }
  /* each direction's hops of the pair, its units: the template's entries naming the peer, picked once over the whole
     list, in list order (copy order) */
  uint32_t *pick[2]={NULL,NULL};uint64_t units[2]={0,0};
  uint64_t *bytes=NULL,*carried=NULL;uint32_t *assigned[2]={NULL,NULL};
  for(int d=0;d<2 && index!=UINT32_MAX;d++){
    if(!(pick[d]=malloc(hops*sizeof *pick[d]))){error=ENOMEM;goto failed;}
    for(uint64_t j=0;j<hops;j++)
      if(list[j].node==index && list[j].direction==(uint32_t)d)pick[d][units[d]++]=(uint32_t)j;
  }
  const uint64_t most=units[0]>units[1]?units[0]:units[1];
  if(most){
    bytes=malloc(most*sizeof *bytes);carried=malloc((links?links:1)*sizeof *carried);
    if(!bytes || !carried){error=ENOMEM;goto failed;}
    for(uint64_t k=0;k<most;k++)bytes[k]=page_bytes();
  }
  /* the assignment, and this cable's units on each of its queue pairs */
  uint64_t lane_units[2][MESH_LINK_LANES]={{0}};
  for(int d=0;d<2;d++){
    if(!units[d])continue;
    if(!(assigned[d]=malloc(units[d]*sizeof *assigned[d]))){error=ENOMEM;goto failed;}
    if(isinf(mesh_graph_list_schedule(links,rate,units[d],bytes,assigned[d],carried))){error=EHOSTUNREACH;goto failed;}
    for(uint64_t k=0;k<units[d];k++){
      const uint32_t a=assigned[d][k];
      if(cables[on[a]].key[end]!=port->index){assigned[d][k]=UINT32_MAX;continue;}
      const uint32_t q=queue[a];
      if(!(l->plan>>q&1) || !l->provider.queues || (int)q>=l->provider.count || !l->provider.queues[q].pair){error=EPROTO;goto failed;}
      assigned[d][k]=q;lane_units[d][q]++;
    }
  }
  /* a page a lane's queue cannot hold refused, not left unposted: its frames past the lane's receive room (lane_reset's
     frame_capacity) or past its SEND gate's (link_connect) */
  const uint64_t frames=frames_of(page_bytes()).n;
  uint64_t carried_units=0;
  for(uint32_t q=0;q<MESH_LINK_LANES;q++){
    if((lane_units[MESH_RECEIVE][q] && frames>l->lane[q].frame_capacity) || (lane_units[MESH_SEND][q] && frames>l->gate[q].capacity.n)){
      error=EMSGSIZE;goto failed;
    }
    carried_units+=lane_units[MESH_SEND][q]+lane_units[MESH_RECEIVE][q];
  }
  /* 2026-11-04, the coordinator's ruling on sub-assignment 6's finding 2 (a re-paired child loading the plan again):
     a plan whose run has begun, its state's `executions` (acquire) above 0 at this load, is refused (EALREADY, by
     link_run's refusal: `code`, phase MESH_STOPPED, `prepared` the refused generation), where this port carries units
     of it.  Why: a record's executions are counted from the load (`landings`, the cursors at execution 0), and a load
     after the run began cannot know which executions landed or were sent before it; mesh-dataflow.c
     mesh_transfers_start raises `executions` only once every port its hops name has loaded the plan, so a load that
     reads above 0 is a late one (a child re-paired on the same plan, or a port loading after the start).  Such a port
     posts nothing of the plan, so the next plan finds no request of it outstanding on its queue pairs, and its
     `prepared` is cleared by link_close once `hops`, `plan` or the client moves, so the release of the plan's claim
     is held no longer than the plan stands.  Where the port carries no unit of the plan it is loaded as before
     (differs from the ruling's text, which refuses every such load): it posts nothing either way, and a refused port
     whose cable reaches a node a hop names would fail the next start's wait (STOPPED) with nothing to serve.  A read
     torn by a plan written again is EAGAIN, not a refusal (plan_refused).  What must stay: no record served from a
     load after its run began */
  if(carried_units && atomic_load_explicit(&words->executions,memory_order_acquire)){
    error=plan_refused(hops,plan,EALREADY);goto failed;
  }
  /* the records, by lane, copy and assignment order: first[q (u + 1) + c] lane q's copy c's first, each lane's units
     counted by copy and summed in that order (copy pick / n; a lane's units come in list order, so in copy order) */
  f->send_count=f->receive_count=0;
  for(uint32_t q=0;q<MESH_LINK_LANES;q++){f->send_count+=lane_units[MESH_SEND][q];f->receive_count+=lane_units[MESH_RECEIVE][q];}
  f->sends=f->send_count?aligned_alloc(128,f->send_count*sizeof *f->sends):NULL;
  f->receives=f->receive_count?aligned_alloc(128,f->receive_count*sizeof *f->receives):NULL;
  for(int d=0;d<2;d++)f->first[d]=calloc(MESH_LINK_LANES*width,sizeof *f->first[d]);
  if((f->send_count && !f->sends) || (f->receive_count && !f->receives) || !f->first[0] || !f->first[1]){error=ENOMEM;goto failed;}
  uint64_t fill[2][MESH_LINK_LANES];
  f->carrying[0]=f->carrying[1]=0;
  for(int d=0;d<2;d++){
    uint64_t *first=f->first[d],at=0;
    for(uint64_t k=0;k<units[d];k++)if(assigned[d][k]!=UINT32_MAX)first[assigned[d][k]*width+pick[d][k]/n]++;
    for(uint32_t q=0;q<MESH_LINK_LANES;q++){
      for(uint32_t c=0;c<copies;c++){const uint64_t m=first[q*width+c];first[q*width+c]=at;at+=m;}
      first[q*width+copies]=at;
      fill[d][q]=first[q*width];
      f->cursor[d][q]=(struct cursor){.at=first[q*width],.end=first[q*width+1]};
      if(lane_units[d][q])f->carrying[d]|=UINT32_C(1)<<q;
    }
  }
  f->lanes=f->carrying[MESH_SEND]|f->carrying[MESH_RECEIVE];
  for(int d=0;d<2;d++)for(uint64_t k=0;k<units[d];k++){
    const uint32_t q=assigned[d][k];
    if(q==UINT32_MAX)continue;
    const struct mesh_hop *hop=list+pick[d][k];
    /* a field the shifts below cannot hold, or a receive of no pages, is none a plan writes: refused (EINVAL) */
    if(((d==MESH_RECEIVE || hop->field!=MESH_ABSENT) && (hop->field>62 || hop->field&1)) ||
       (d==MESH_RECEIVE && hop->count && !hop->pages)){
      error=plan_refused(hops,plan,EINVAL);goto failed;
    }
    const struct ibv_sge span=wire_span(&port->device,(uint64_t)hop->page*M->pgsz,page_bytes());
    if(d==MESH_SEND){
      struct prepared_send *record=f->sends+fill[d][q]++;
      *record=(struct prepared_send){.word=region_word(hop->word),.field=hop->field,.span=span};
      record->request=(struct ibv_send_wr){.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1,.opcode=IBV_WR_SEND,
        .send_flags=IBV_SEND_SIGNALED};
    } else {
      struct prepared_receive *record=f->receives+fill[d][q]++;
      *record=(struct prepared_receive){.span=span,.word=region_word(hop->word),.count=hop->count?region_word(hop->count):NULL,
        .field=hop->field,.pages=hop->pages,.copy=(uint32_t)(pick[d][k]/n)};
      record->request=(struct ibv_recv_wr){.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1};
    }
  }
  /* design/prepared-machine.md#M27 (its traces' room taken before the plan is told loaded, below, so the only failure
     after that store is a plan written again: no refusal follows it).  Room for one event a record each way, its first
     u executions' worth (each record's first posting, or the first landings), sized here once and written while room
     lasts (link_receive_drain, link_send_drain), so a trace is bounded by the template whatever the run's length */
  const char *ledger=getenv("MESH_LEDGER");
  f->ledger=ledger && ledger[0] && !(ledger[0]=='0' && !ledger[1]);
  const uint64_t traces[2]={f->send_count,f->receive_count};
  for(int d=0;d<2;d++){
    f->traced[d]=0;f->trace_capacity[d]=f->ledger?traces[d]:0;f->trace[d]=NULL;
    if(!f->trace_capacity[d])continue;
    if(!(f->trace[d]=aligned_alloc(32,(f->trace_capacity[d]+1)*sizeof(struct mesh_trace)))){error=ENOMEM;goto failed;}
  }
  /* The plan told loaded, then read whole.  The port's `prepared` (the generation it loads, release) is this port's
     announcement for the deferred reclamation of a retired plan's pages (design/disjoint-pages-change-request.md, C3.4:
     RCU's call_rcu, P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004,
     epoch-based reclamation; design/disjoint-pages-revision-coordination.md, D10): the bridge parent's turn releases a
     closed claim once no port's `prepared` holds its generation (mesh-dataflow.c mesh_retired_release: a seq_cst
     fence, then the announcements read), and a claim is closed only once its plan is no longer the one written (hdr
     `hops` 0 or a later plan's: mesh-collective.c mesh_plan, mesh-dataflow.c mesh_retire).  So `prepared` is stored,
     then a seq_cst fence, then `hops` and `plan` read again (Fraser's announced epoch and its re-read): of the two
     fences, where the release's comes first this re-read sees `hops` or `plan` moved and the load withdraws
     (`prepared` 0, EAGAIN) before anything is posted; where this one comes first the release reads this generation and
     keeps the claim.  Nobody waits.  The re-read is the plan read whole too (review R1, round 2): the entries and the
     plan's state were read after `hops` (acquire); were the plan written again meanwhile (its `hops` 0 first, then a
     new generation `plan`, the plan call's release fence after the 0), a read of the new writes is seen here, after
     the fence (seq_cst, an acquire fence as well), as `hops` or `plan` moved, and what was read is dropped (EAGAIN:
     link_run tries again on its next pass).  The refusal's order stays: no refusal follows this store, and a refusal
     stores `code` and phase before `prepared` (link_run); and so does a load's: a refusal of this child's undone
     (`code` 0, phase MESH_PAIRED) before `prepared` is stored.
     2026-11-04, sub-assignment 6: reversing this note's 2026-11-03 form (reviewer 12's defect 1 on 5d919bd7), in which
     the announcement paired with mesh_plan_erased, the plan call's bounded wait (a Dekker store-fence-load pair: E. W.
     Dijkstra, Cooperating sequential processes, 1965) for no port to hold a plan whose fixed areas it would zero and
     rewrite in place.  That wait went with the rewriting in place (C4; C3.4: a new plan's state is in new pages), and
     the announcement is now the reclamation's.  What must stay: `prepared` stored before the fence and the re-read,
     nothing posted before both, and the store moved here from link_run with the undoing of a refusal */
  struct mesh_port_info *info=&mesh_links(M)[port->index].port;
  if(f->refused){
    f->refused=0;
    atomic_store_explicit(&info->code,0,memory_order_relaxed);
    atomic_store_explicit(&info->phase,MESH_PAIRED,memory_order_release);
  }
  atomic_store_explicit(&info->prepared,plan,memory_order_release);
  atomic_thread_fence(memory_order_seq_cst);
  /* `hops` acquire: where it reads a next plan's count equal to this one's (a re-plan of the same size), the `plan`
     load after it reads that plan's generation (written before its `hops`, release), so the check fails */
  if(atomic_load_explicit(&M->hops,memory_order_acquire)!=hops ||
     atomic_load_explicit(&M->plan,memory_order_relaxed)!=plan){
    atomic_store_explicit(&info->prepared,0,memory_order_release);
    error=EAGAIN;goto failed;
  }
  /* the trace layout, the plan's copies and executions read from its state (the executions as the load finds them;
     link_run reads them again each pass) */
  if(f->ledger)for(uint32_t q=0;q<MESH_LINK_LANES;q++)if(f->lanes>>q&1)
    fprintf(stderr,"{\"trace_layout\":%u,\"rank\":%u,\"lane\":%u,\"send_base\":%llu,\"sends\":%llu,\"receive_base\":%llu,\"receives\":%llu,"
      "\"send_record_bytes\":%zu,\"receive_record_bytes\":%zu,\"copies\":%u,\"executions\":%llu}\n",port->index,M->node,q,
      (unsigned long long)(uintptr_t)(f->sends?f->sends+f->first[MESH_SEND][q*width]:NULL),
      (unsigned long long)lane_units[MESH_SEND][q],
      (unsigned long long)(uintptr_t)(f->receives?f->receives+f->first[MESH_RECEIVE][q*width]:NULL),
      (unsigned long long)lane_units[MESH_RECEIVE][q],sizeof(struct prepared_send),sizeof(struct prepared_receive),copies,
      (unsigned long long)atomic_load_explicit(&words->executions,memory_order_acquire));
  f->words=words;f->copies=copies;
  f->executions=f->send_total=f->receive_total=f->sent=f->landed=0;
  for(int d=0;d<2;d++){free(pick[d]);free(assigned[d]);}
  free(bytes);free(carried);
  return 0;
failed:
  for(int d=0;d<2;d++){free(pick[d]);free(assigned[d]);free(f->trace[d]);f->trace[d]=NULL;free(f->first[d]);f->first[d]=NULL;}
  free(bytes);free(carried);
  free(f->sends);free(f->receives);
  f->sends=NULL;f->receives=NULL;f->send_count=f->receive_count=0;f->lanes=f->carrying[0]=f->carrying[1]=0;
  return error;
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* d6dceac's receive drain (its unordered form: a completion names its record), now one pass of the port's progress
   thread over the link's receive completions (the queue pairs' shared queue, `completion`), in the tag's batches
   (rdma/mesh-link.c link_receive): each landing of a lane served, after the acquire the poll's DMA asks, is its
   record's j-th since the load (`landings`), execution t = copy + j u (link_prepare), and writes the plan's state alone
   (design/disjoint-pages-change-request.md, C3.1 and C3.7; design/disjoint-pages-revision-coordination.md, "The
   bridge"; mesh.h struct mesh_hop, a receive): its phase field flipped by one atomic XOR (release) of bit field +
   ((t / u) & 1), t / u being j, so the field reads mesh_phase(t / u + 1); then its range's count of its copy one more
   (an atomic add, release); and, where the count it made is (j + 1) x the entry's `pages` (the pages this node
   receives of the range a copy), the count word's waiters woken (os_sync_wake_by_address_all, 8 bytes, region-shared,
   as the client's waiter waits: mesh-metal.m).  That add is the one that completes round j: by the allocator's order
   every landing of a copy's round j comes before any of its round j + 1, so the arrival that brings the count to
   (j + 1) P is a landing of round j, and an atomic add returns each value once, so exactly one landing wakes for each
   round (a cancellation's raise wakes for itself: count_raise).  A receive with no count (a relay's own, `count`
   NULL) flips its phase alone.  2026-11-04, the coordinator's ruling on review 15's finding 1: in place of this note's
   first 2026-11-04 test, the count a multiple of `pages` (a 64-bit division a landing), a multiply and a compare, and
   no add for a relay's own receive.  It is counted
   (the lane's `got` and frames, which lanes_flush reads); a failed one, or one of other than a whole page, fails the
   link (link_failed), never retried.  Then, after this pass's completions, each served lane's receives posted, its
   cursor's (struct cursor: the executions in order, execution t's records copy t mod u's), through the executions
   below the plan's `executions` as link_run last read it, while its queue has room (the tag's requests,
   MESH_LANE_UNITS, and its frames, frame_capacity), a record of round j (execution t, j = t / u, the cursor's `round`)
   only once its `landings` reads at least j: its last posting has landed (one load, from the line its request
   starts).  The lane's posting stops at the first record that has not, so its order stays, and a page is named by at
   most one posted receive at any moment (design/disjoint-pages-change-request.md, D.2, "How many pages are in flight
   at once", and D.3.5; design/disjoint-pages-rulings-coordination.md, R1).  A receive waits on its own record's last
   landing alone, and behind the earlier receives of its lane, whose order the peer's SENDs fill; at most
   MESH_POST_QUANTUM a pass.  0 where the link failed.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's landing, its record's `mask` ORed into its execution's landed
   record (C4: the landed record's ORed bits), and its chain of receives each posted once (`next`).  What must stay: the
   XOR before the add, both after the poll's acquire; no load, fence or call a landing more but the wake at a range's
   end; each lane's executions in order.
   2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.5 and D.4; design/disjoint-pages-rulings-
   coordination.md, R1): reversing sub-assignment 6's posting "ahead", on nothing but the queue's room, which posted a
   record again while an earlier posting of it might be outstanding, so two postings named one page.  Why: it gained
   nothing over the u rounds the plan's copies already give, and rested on the provider copying a request at its post,
   which nothing documents (D.2).  What must stay: the completions taken before the posting in the same pass, so a
   landing frees its record's next round at once; each lane's executions in order, no record skipped */
static __attribute__((always_inline)) inline int link_receive_drain(struct port *port,struct flow *f,uint32_t served,int traced){
  struct link *l=&port->link;
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *)=l->provider.poll;
  const uint64_t page=page_bytes(),each=frames_of(page).n;
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?f->trace[MESH_RECEIVE]:NULL;
  const size_t capacity=traced?f->trace_capacity[MESH_RECEIVE]:0;
  for(int count=MESH_COMPLETION_BATCH;count==MESH_COMPLETION_BATCH;){
    count=poll(l->provider.completion,MESH_COMPLETION_BATCH,done);
    if(count<0){link_failed(port,EIO);return 0;}
    if(count)atomic_thread_fence(memory_order_acquire);
    for(int i=0;i<count;i++){
      const uint32_t lane=link_lane(l,done[i].qp_num);
      if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
      uint64_t status_opcode;
      memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
      if((uint32_t)status_opcode){link_failed(port,(uint32_t)status_opcode);return 0;}
      if(!((status_opcode>>32)&IBV_WC_RECV))continue;
      /* a landing of other than a whole page (a peer of another block, or a plan the two ends do not share) is the
         link's failure by error, EMSGSIZE, not a page told landed with bytes of it unwritten (review R1, round 2) */
      if(done[i].byte_len!=page){link_failed(port,EMSGSIZE);return 0;}
      const uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
      const uint64_t round=record->landings++;
      atomic_fetch_xor_explicit(record->word,UINT64_C(1)<<(record->field+(round&1)),memory_order_release);
      if(record->count && atomic_fetch_add_explicit(record->count,1,memory_order_release)+1==(round+1)*record->pages)
        os_sync_wake_by_address_all(record->count,sizeof *record->count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
      struct lane_state *state=l->lane+lane;
      state->got++;state->frames_retired+=each;f->landed++;
      if(traced && f->traced[MESH_RECEIVE]<capacity){
        const uint64_t published=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        trace[f->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)record,polled,published,published};
      }
    }
  }
  const uint64_t executions=f->executions;
  const uint32_t copies=f->copies;
  for(uint32_t lanes=f->carrying[MESH_RECEIVE]&served;lanes;lanes&=lanes-1){
    const uint32_t k=(uint32_t)__builtin_ctz(lanes);
    struct lane_state *state=l->lane+k;
    const struct mesh_queue *queue=l->provider.queues+k;
    const uint64_t *first=f->first[MESH_RECEIVE]+k*((uint64_t)copies+1);
    struct cursor at=f->cursor[MESH_RECEIVE][k];
    for(uint32_t budget=MESH_POST_QUANTUM;budget;budget--){
      while(at.at==at.end && at.execution<executions)cursor_step(&at,first,copies);
      if(at.execution>=executions)break;
      if(f->receives[at.at].landings<at.round)break;
      if(state->posted-state->got>=MESH_LANE_UNITS || state->frames_posted-state->frames_retired+each>state->frame_capacity)break;
      struct ibv_recv_wr *bad;
      const int error=queue->receive(queue->pair,&f->receives[at.at].request,&bad);
      if(error){lane_lost(port,k,error<0?-error:error);break;}
      state->frames_posted+=each;state->posted++;
      at.at++;
    }
    f->cursor[MESH_RECEIVE][k]=at;
  }
  return 1;
}
/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
/* d6dceac link_receive_progress, now a pass and not a thread's entry: its traced variant under MESH_LEDGER */
static int link_receive_progress(struct port *port,uint32_t served){
  struct flow *f=flows+port->index;
  /* design/prepared-machine.md#M27 */
  if(f->trace[MESH_RECEIVE])return link_receive_drain(port,f,served,1);
  return link_receive_drain(port,f,served,0);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* d6dceac's send drain (each queue pair's records from its cursor, a record posted as its release word holds, through
   the provider's own post), now one pass of the port's progress thread: the SEND completions first, on their own queue
   (`sent`, mesh-verbs.h pair_up), in the tag's batches (rdma/mesh-link.c link_send), each of a lane served retiring its
   frames through the lane's gate and counted (`done`), a failed one failing the link (no completion counts a page's
   forwards: link_prepare's note of 2026-11-03); then each served lane's SENDs, its cursor's (struct cursor: the
   executions in order, execution t's records copy t mod u's), through the executions below the plan's `executions` as
   link_run last read it, each of execution t once released, one acquire load of its word
   (design/disjoint-pages-change-request.md, C3.7; design/disjoint-pages-revision-coordination.md, "The bridge"): from
   its range's source (`field` MESH_ABSENT) where the range's supplied word reads at least t + 1, its producer's
   release of t; forwarding, where its receive's phase field reads mesh_phase(t / u + 1), that receive's landing of t
   (a cancellation writes no phase, so it releases nothing); while the lane's gate holds its frames and its requests are
   under MESH_LANE_UNITS, at most MESH_POST_QUANTUM a pass.  A ready page waits only behind earlier pages of its own
   queue pair.  0 where the link failed.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's release, a word holding every bit of the record's `mask`
   (MESH_SUPPLIED, or its receive's bit of its execution's landed record: C4).  What must stay: one acquire load a send,
   and each lane's executions in order */
static __attribute__((always_inline)) inline int link_send_drain(struct port *port,struct flow *f,uint32_t served,int traced){
  struct link *l=&port->link;
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  const struct frames each=frames_of(page_bytes());
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?f->trace[MESH_SEND]:NULL;
  const size_t capacity=traced?f->trace_capacity[MESH_SEND]:0;
  for(int count=MESH_COMPLETION_BATCH;count==MESH_COMPLETION_BATCH;){
    count=l->provider.poll(l->provider.sent,MESH_COMPLETION_BATCH,done);
    if(count<0){link_failed(port,EIO);return 0;}
    for(int i=0;i<count;i++){
      const uint32_t lane=link_lane(l,done[i].qp_num);
      if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
      if(done[i].status){link_failed(port,(int64_t)done[i].status);return 0;}
      gate_retire(l->gate+lane,each);l->lane[lane].done++;f->sent++;
    }
  }
  const uint64_t executions=f->executions;
  const uint32_t copies=f->copies;
  for(uint32_t lanes=f->carrying[MESH_SEND]&served;lanes;lanes&=lanes-1){
    const uint32_t k=(uint32_t)__builtin_ctz(lanes);
    struct lane_state *state=l->lane+k;
    struct send_gate *gate=l->gate+k;
    const struct mesh_queue *queue=l->provider.queues+k;
    int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=queue->send;
    const uint64_t *first=f->first[MESH_SEND]+k*((uint64_t)copies+1);
    struct cursor at=f->cursor[MESH_SEND][k];
    for(uint32_t budget=MESH_POST_QUANTUM;budget;budget--){
      while(at.at==at.end && at.execution<executions)cursor_step(&at,first,copies);
      if(at.execution>=executions)break;
      struct prepared_send *record=f->sends+at.at;
      const uint64_t word=atomic_load_explicit(record->word,memory_order_acquire);
      if(record->field==MESH_ABSENT?word<=at.execution:((word>>record->field)&3)!=mesh_phase(at.round+1))break;
      if(state->sent-state->done>=MESH_LANE_UNITS || !gate_holds(gate,each))break;
      const uint64_t observed=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      struct ibv_send_wr *bad;
      const int error=post(queue->pair,&record->request,&bad);
      if(traced && f->traced[MESH_SEND]<capacity){
        const uint64_t sent=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        trace[f->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)record,observed,observed,sent};
      }
      if(error){lane_lost(port,k,error<0?-error:error);break;}
      gate_post(gate,each);
      __atomic_store_n(&state->sent,state->sent+1,__ATOMIC_RELEASE);gate->requests++;
      at.at++;
    }
    f->cursor[MESH_SEND][k]=at;
  }
  return 1;
}
/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* d6dceac link_send_progress, now a pass and not a thread's entry: its traced variant under MESH_LEDGER */
static int link_send_progress(struct port *port,uint32_t served){
  struct flow *f=flows+port->index;
  /* design/prepared-machine.md#M27 */
  if(f->trace[MESH_SEND])return link_send_drain(port,f,served,1);
  return link_send_drain(port,f,served,0);
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
/* The end of the plan a port's child serves, on its progress thread: where the plan was written again (hdr `hops` 0,
   the plan call's first store, or another plan's), where the link serves no lane (a halt, a loss), and as the child
   closes.  Each of its receives with a count (none for a relay's own, which nobody waits on) not landed for every
   execution it serves (its `landings` below the executions t <
   `executions` of its copy, the plan's word read now, so a rise since the last pass is cancelled too) is cancelled, the
   cancellation an arrival (design/disjoint-pages-change-request.md, C3.1: "Landings and cancellations both arrive on
   the count ... A link loss is such a cancellation"; design/disjoint-pages-revision-coordination.md, "The bridge"):
   its range's count of its copy raised to its final value, the entry's `pages` x those executions, never lowered, and
   woken (count_raise); no phase is written.  These stores land in the plan's own pages while this port's `prepared`
   still holds the plan (stored 0 below, after them, release), so the bridge's release of a retired plan's claim (D10:
   mesh-dataflow.c mesh_retired_release, its fence, then the announcements read) comes after them.  Its traces told
   (d6dceac link_close's lines); its records freed; and the port's `prepared` 0, no plan loaded.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's cancellation, UINT64_MAX stored into the cancel word of the
   execution whose landed record lacked a receive's bit (C4: the cancel word and the landed record's several writers);
   a cancellation now has the one form a landing has, an arrival on its count.  What must stay: a count only raised,
   never a phase written for a cancellation, and `prepared` cleared after every store into the plan's pages.
   Where a SEND or receive of the link's queue pairs is outstanding (the lanes' counts),
   nothing more is loaded in the child's life, and where the link still serves, it is lost as the tag loses a link whose
   holdings ended with units outstanding (rdma/mesh-link.c stream_end: `lost` LINK_STREAMS_ENDED, every lane `broken`,
   so the child tells CHILD_BROKEN and its parent loses the link): a posted receive is never taken back but by the
   child's exit (lanes_flush), so no later plan's page is posted beside it, and `prepared` stays the plan's until the
   parent reaps the child.  No verbs call: d6dceac's down_pair and listener close are the child's exit's */
static void link_close(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct flow *f=flows+port->index;
  if(f->state==FLOW_NONE || f->state==FLOW_ENDED)return;
  /* a refused plan's end: its generation in `prepared` (the refusal's tell, link_run) cleared; the phase and code stay
     the refusal's until a plan loads */
  if(f->state==FLOW_REFUSED){
    f->state=FLOW_NONE;
    atomic_store_explicit(&mesh_links(M)[port->index].port.prepared,0,memory_order_release);
    return;
  }
  /* a record's `landings` is this thread's alone (its landings, link_receive_drain) */
  const uint64_t executions=atomic_load_explicit(&f->words->executions,memory_order_acquire);
  for(uint64_t r=0;r<f->receive_count;r++){
    const struct prepared_receive *record=f->receives+r;
    const uint64_t serves=copy_executions(executions,f->copies,record->copy);
    if(record->count && record->landings<serves)count_raise(record->count,(uint64_t)record->pages*serves);
  }
  /* design/prepared-machine.md#M27 */
  for(int d=0;d<2;d++){
    for(size_t i=0;i<f->traced[d];i++){
      const struct mesh_trace t=f->trace[d][i];
      fprintf(stderr,"{\"native_trace\":%u,\"direction\":%d,\"event\":%zu,\"identity\":%llu,\"ns\":[%llu,%llu,%llu]}\n",
        port->index,d,i,(unsigned long long)t.identity,(unsigned long long)t.begin,(unsigned long long)t.middle,(unsigned long long)t.end);
    }
    free(f->trace[d]);f->trace[d]=NULL;f->traced[d]=0;f->trace_capacity[d]=0;
  }
  uint32_t outstanding=0;
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)
    if(l->plan>>k&1 && (l->lane[k].sent!=l->lane[k].done || l->lane[k].posted!=l->lane[k].got))outstanding|=UINT32_C(1)<<k;
  free(f->sends);free(f->receives);
  f->sends=NULL;f->receives=NULL;f->send_count=f->receive_count=0;f->lanes=f->carrying[0]=f->carrying[1]=0;
  for(int d=0;d<2;d++){free(f->first[d]);f->first[d]=NULL;}
  memset(f->cursor,0,sizeof f->cursor);
  f->words=NULL;f->copies=0;f->executions=f->send_total=f->receive_total=0;
  f->state=outstanding?FLOW_ENDED:FLOW_NONE;
  /* `prepared` 0 at once where nothing is outstanding; else it stays the plan's until the parent reaps the child
     (mesh-children.c child_reaped, streams_cancel), so a retired client's rows its requests name are kept
     (mesh-dataflow.c mesh_retired_release reads every port's `prepared`: the coordinator, 2026-11-03) */
  if(!outstanding)atomic_store_explicit(&mesh_links(M)[port->index].port.prepared,0,memory_order_release);
  if(outstanding && served){
    mesh_say("bridge","link %u (%s): its plan ended with work outstanding on lanes %#x: the link toward node %u lost",
             port->index,port->name,outstanding,port->peer);
    atomic_store_explicit(&l->lost,LINK_STREAMS_ENDED,memory_order_relaxed);
    atomic_fetch_or_explicit(&l->broken,l->plan,memory_order_release);
    wake();
  }
}

/* The plan's turn on the port's progress thread, each pass while its link serves: the plan served ended where the
   region's was written again (link_close: hdr `hops` 0, or another generation `plan`) or the client it was loaded for
   holds the region no more (hdr `client` another: mesh-dataflow.c mesh_retire); a plan written while a client holds
   the region and not yet loaded loaded (link_prepare), its generation stored in the port's `prepared` by link_prepare
   before its last check (release, then a seq_cst fence: its note of 2026-11-03; the client's start waits on it at
   every port whose cable reaches a node a hop names, so it is stored where the rule gave this cable no page too), or
   refused, the port's phase MESH_STOPPED, its `code` the refusal's and its `prepared` the refused
   generation (CONTRACT "A plan loaded", review R1 round 2), until the plan call writes again, or read torn and tried
   again on the next pass; a plan loaded served, its receives, then its SENDs (a landed relay page's SEND goes in the
   same pass), through the executions below the plan's `executions`, read (acquire) each pass the plan is served or
   done (one load a pass: the client's start raises it, design/disjoint-pages-revision-coordination.md, D2, and wakes
   `plan`, on which the idle thread waits: port_progress), until every record has served every one of them (every
   SEND completed, every receive landed: done), and served again where it rises (done to served).  1 while it serves a
   plan with work left, 0 idle.
   2026-11-04, sub-assignment 6: reversing 2026-11-03's done, every record's one posting completed with hdr
   `executions` fixed at the plan (C3.3: no input of the plan); what must stay: done only where every execution the
   word asks is served */
static int link_run(struct port *port,uint32_t served){
  struct flow *f=flows+port->index;
  const uint64_t hops=atomic_load_explicit(&M->hops,memory_order_acquire);
  const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_relaxed);
  const uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
  if(f->state!=FLOW_NONE && f->state!=FLOW_ENDED && (!hops || hops!=f->hops || plan!=f->plan || client!=f->client))
    link_close(port,served);
  if(f->state==FLOW_NONE && hops && client){
    struct mesh_port_info *info=&mesh_links(M)[port->index].port;
    const int error=link_prepare(port);
    /* a load that read the plan torn (re-planned while it read, link_prepare's last check) is dropped and tried again
       on the next pass, neither loaded nor refused */
    if(error==EAGAIN)return 1;
    if(error){
      f->state=FLOW_REFUSED;f->refused=1;
      mesh_say("bridge","link %u (%s): plan %llu refused (code %d)",port->index,port->name,
               (unsigned long long)f->plan,error);
      /* the refusal told with the generation it refuses (review R1, round 2): `code` and phase MESH_STOPPED, then
         `prepared` the refused generation (release, last), so a start that reads `prepared` == its plan reads this
         refusal's code and phase, never an earlier plan's; a STOPPED port holds no request of the plan (no retention) */
      atomic_store_explicit(&info->code,error,memory_order_relaxed);
      atomic_store_explicit(&info->phase,MESH_STOPPED,memory_order_release);
      atomic_store_explicit(&info->prepared,f->plan,memory_order_release);
      return 0;
    }
    /* loaded: a refusal of this child's undone and `prepared` the plan's, both stored by link_prepare before its last
       check (its note of 2026-11-03, reviewer 12's defect 1) */
    f->state=FLOW_SERVING;
  }
  if(f->state==FLOW_SERVING || f->state==FLOW_DONE){
    const uint64_t executions=atomic_load_explicit(&f->words->executions,memory_order_acquire);
    if(executions>f->executions){flow_executions(f,executions);f->state=FLOW_SERVING;}
  }
  if(f->state!=FLOW_SERVING)return 0;
  if(!link_receive_progress(port,served) || !link_send_progress(port,served))return 0;
  if(f->sent==f->send_total && f->landed==f->receive_total){f->state=FLOW_DONE;return 0;}
  return 1;
}

/* tag pre-return-20261103 rdma/mesh-link.c:367-371 port_tell (60b00774), verbatim */
/* the progress thread's tell (the epoch it has seen, the guard it holds at), its raiser woken (port_wait) */
static void port_tell(_Atomic uint64_t *word,uint64_t value){
  atomic_store_explicit(word,value,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* tag pre-return-20261103 rdma/mesh-link.c:372-386 port_pass (8b757084), verbatim but its work: the plan's turn
   (link_run) in place of each stream's holding, SENDs, receives and completions, and whether it has work left told */
/* design/transport.md section 14: one pass of a port's progress thread: the lanes the link serves (`served` read once a
   pass: none posts on a lane its link no longer serves), the plan's turn on them (its receives first: a peer's SEND is
   held until its receive is posted, TN3205, so ours let its SENDs go).  It tells the port's epoch it began at (a
   conclusion around a stuck call moves it: ports_stalled).  1 where the plan has work left */
static int port_pass(struct port *port,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  const int busy=served?link_run(port,served):0;
  if(epoch!=*told){port_tell(&port->seen,epoch);*told=epoch;}
  return busy;
}
/* tag pre-return-20261103 rdma/mesh-link.c:387-424 port_progress (3a82ede9), verbatim but the plan: while the link
   serves and its plan has no work left (none written, or done), the thread waits on the plan's word (hdr `plan`, its
   generation, region-shared) as it waits on the guard, looking again every MESH_PAUSE_NS (so a client's retirement,
   which wakes `control` and not `plan`, is seen within it); a link that serves no lane ends the plan it served
   (link_close: a halt); and the thread's end ends it too */
/* design/transport.md section 14: a port's progress thread, its SENDs' and its receives': it passes without pause while
   its link serves a lane (rubric M8: the queues read as fast as a hardware thread allows), and waits on the port's
   guard word while it serves none, looking again every MESH_PAUSE_NS (the stop's `closing`) */
static void *port_progress(void *argument){
  struct port *port=argument;
  pthread_setname_np("mesh.rdma");
  /* fault_heard: this thread its port's, and its handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  atomic_store_explicit(&port->thread[0],pthread_self(),memory_order_release);
  uint64_t told=atomic_load_explicit(&port->seen,memory_order_relaxed);
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* while the port's guard stands, or its link serves no lane, no pass (no post on any queue pair of its device) and
       no lane read: the thread waits on the guard word (design/transport.md B13: a guarded call in the driver costs no
       core), woken by its raising and lowering, by a settle and by the link's coming up (guard_wake), looking again
       every MESH_PAUSE_NS (the stop's `closing`), the guard told and the epoch at each waking, each tell waking its
       raiser (port_tell) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    const uint32_t served=atomic_load_explicit(&port->link.served,memory_order_acquire);
    if(guard&1 || !served){
      /* a fault absorbed on the port (fault_heard: the handler, which may make no call but mmap, munmap and sigaction,
         halted the port) wakes its child's loop, which tells its parent at once, not at its next tick */
      if(atomic_load_explicit(&faulted,memory_order_relaxed))wake();
      if(guard&1)port_tell(&port->guarded,guard);
      /* a link that serves no lane ends the plan it served (no verbs call: link_close) */
      if(!served)link_close(port,0);
      const uint64_t epoch=atomic_load_explicit(&port->epoch,memory_order_acquire);
      if(epoch!=told){port_tell(&port->seen,epoch);told=epoch;}
      /* the wait on the word that holds the thread: the guard while it stands, else the lanes served while they are
         none, so a change of that word between the read above and the wait ends the wait at once (a wait on the guard
         for a link's coming up slept MESH_PAUSE_NS where the lanes were served between the two) */
      if(guard&1)os_sync_wait_on_address_with_timeout(&port->guard,guard,sizeof port->guard,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                      OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      else os_sync_wait_on_address_with_timeout(&port->link.served,0,sizeof port->link.served,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      continue;
    }
    /* the plan's generation read before the pass, so a plan written during it ends the wait below at once (the plan
       call wakes it, region-shared, after its last store: mesh-collective.c mesh_plan) */
    const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_acquire);
    if(port_pass(port,&told))continue;
    os_sync_wait_on_address_with_timeout(&M->plan,plan,sizeof M->plan,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
  }
  link_close(port,0);
  return NULL;
}
/* tag pre-return-20261103 rdma/mesh-link.c:425-429 guard_wake (60b00774), verbatim but not static, and the plan's word
   woken too (region-shared: the progress thread's third wait, port_progress), so a raise holds at once */
/* design/transport.md B13: the progress thread parked on the port's guard looks again */
void guard_wake(struct port *port){
  os_sync_wake_by_address_all(&port->guard,sizeof port->guard,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  os_sync_wake_by_address_all(&port->link.served,sizeof port->link.served,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  if(M)os_sync_wake_by_address_all(&M->plan,sizeof M->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* tag pre-return-20261103 rdma/mesh-link.c:430-442 port_threads (3a82ede9), verbatim but not static */
/* design/transport.md: the port's progress thread started, once; 0, or pthread_create's error */
int port_threads(struct port *port){
  if(port->threads)return 0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  atomic_store(&port->seen,atomic_load(&port->epoch));
  pthread_t created;
  const int error=pthread_create(&created,&attributes,port_progress,port);
  if(!error)port->threads=1;
  pthread_attr_destroy(&attributes);
  return error;
}
/* tag pre-return-20261103 rdma/mesh-link.c:443-456 port_wait (e3c4c986), verbatim but its deadline: main's mesh.h has
   no struct mesh_deadline, so the tag's mesh_deadline, mesh_deadline_left and mesh_expired are written out (the same
   MESH_PLATFORM_NS from the wait's start, the same end: the wait named on the log and the child's exit, status 1) */
/* design/transport.md: the raiser of a settle or a guard parked, never spinning, until the port's progress thread,
   where started, has its word (`told`: the epoch it has seen, or the guard it holds at) at `value`, woken by the
   thread's tell (port_tell), its guard's waiters woken as it begins and at each waking, within its deadline,
   MESH_PLATFORM_NS from its start: a thread that has not told by then is in a call that has not returned, and the
   port's child, the waiting process, ends at once naming the wait (design/eager-execution.md, assumption A4), its exit the
   kernel's release of its port's objects and its parent's loss of the port (child_reaped) */
static void port_wait(struct port *port,_Atomic uint64_t *told,uint64_t value){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(uint64_t seen;port->threads && (seen=atomic_load_explicit(told,memory_order_acquire))<value;){
    const uint64_t at=clock_ns();
    if(at>=until){
      mesh_say("bridge","mesh-flow child: its port's progress thread holding a settle or the guard (port_wait): past its deadline "
               "of %.3f s; the program ends (design/eager-execution.md, assumption A4)",(double)MESH_PLATFORM_NS/1e9);
      _exit(1);
    }
    guard_wake(port);
    os_sync_wait_on_address_with_timeout(told,seen,sizeof *told,OS_SYNC_WAIT_ON_ADDRESS_NONE,OS_CLOCK_MACH_ABSOLUTE_TIME,until-at);
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:457-473 guard_lower and guard_raise (306b22be), verbatim but not static */
/* design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a port's device, on the port's queue,
   around a change of its verbs objects (queue pairs and completion queues made, connected or destroyed): raised, it
   returns once each started progress thread holds (it told the guard's value before its next pass, and passes no more
   while the guard stands), so no queue pair of the device is posted on while another is made, changed or destroyed
   (Apple's provider: a post on a long-lived queue pair across another's teardown stores to the device's shared doorbell
   mapping after it moved, tbt_post_recv); lowered, the threads parked on it woken and going on
   (design/transport.md B13).  It stands for every change once the port's threads run, the link lanes'
   long-lived queue pairs being posted on while the link is up.  Raised: 1; 0 where no progress thread runs (a thread
   that does not hold within its deadline ends the child: port_wait) */
void guard_lower(struct port *port,int raised){
  if(raised>0){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);guard_wake(port);}
}
int guard_raise(struct port *port){
  if(!port->threads)return 0;
  port_wait(port,&port->guarded,atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1);
  return 1;
}

/* this node's receives of the plan written from the node of map index `from` (UINT32_MAX: every receive) cancelled in
   the bridge parent (streams_cancel, streams_let_go), as link_close cancels its child's: for each such entry of the
   list (copy c's at [c n, c n + n)), its range's count of copy c raised to its final value, the entry's `pages` x the
   executions t < `executions` (the plan's word, read once) with t mod u == c, never lowered, and woken (count_raise;
   design/disjoint-pages-change-request.md, C3.1; design/disjoint-pages-revision-coordination.md, "The bridge"); no
   phase is written.  The parent keeps no record of which receives landed (the children's `landings`), so it raises
   every such count: one already at its final value (every arrival in, or an earlier cancellation) is left as it is,
   and the entries of one range raise its word once.  A plan with no state, or whose copies do not divide its list,
   which every port refuses (link_prepare, EINVAL), names no count to raise; nor does a receive entry whose `count` is
   0 (a relay's own receive, which nobody waits on: 2026-11-04, the coordinator's ruling on review 15's finding 1).
   2026-11-04, sub-assignment 6: reversing 2026-11-03's cancellation, UINT64_MAX stored into the cancel word of each
   not-landed entry's execution (C4: the cancel word); what must stay: a count only raised, and no phase written.
   The plan read whole, as link_prepare reads it (b3cc9990's re-read; 2026-11-03, sub-assignment 6, on the
   coordinator's routing), as before: the plan call stores `hops` 0 (release) and a release fence before it writes any
   entry or its state, and moves `plan` (mesh.h, "Writing a plan"; mesh_retire stores `hops` 0 the same way before it
   closes a retiring client's claims).  So `hops` and `plan` are read (acquire) once before the walk, then the plan's
   state (its `copies` u and `executions`); each receive entry's fields are read, then an acquire fence, then `hops`
   and `plan` again, and the walk stops where either moved: an entry read that saw a rewrite is never acted on, and no
   count word is computed from one or from that state.
   Where its stores land (design/disjoint-pages-change-request.md, C3.6): in the plan's pages while its claim still
   holds them.  Every caller of this walk and of the release of a retired plan's claim (mesh-dataflow.c
   mesh_retired_release) runs on the bridge parent's one event-loop thread, mesh-link.c main's: streams_cancel from
   mesh-ports.c port_lost (called from the loop's turns and events alone: link_turn, ports_stalled, control_heard,
   hello_heard, port_flush, port_say, port_greet, port_watch, the message handlers, mesh-events.c network_heard and
   terminations_turn, mesh-children.c child_spawn and child_reaped), from mesh-children.c child_heard and child_reaped
   (the loop's events), from mesh-link.c region_adopt (main, before the loop) and from the stop (main, after it);
   streams_let_go from mesh-records.c clients_turn (the loop's turn, before its release); and the release from
   clients_turn and region_adopt (after its streams_cancel) alone.  The parent's other threads (control_watch, the
   finder's queue's block in ports_find, the terminations' queue's peer_terminated and peer_arrived) call none of
   them.  So a release never runs inside this walk, and a claim is closed only once its plan is no longer the one
   written (`hops` 0 or a later plan's, before the close: mesh_plan, mesh_retire): a count word computed from an entry
   the re-check found current is stored while that plan's claim holds its pages, closed or not, and the release that
   frees them runs later on this thread.  What must stay: no store from an entry read before its check; this walk and
   the release on one thread; and no claim closed while its plan is still the one written */
static void plan_cancel(uint32_t from){
  const uint64_t hops=atomic_load_explicit(&M->hops,memory_order_acquire);
  const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_acquire);
  if(!hops)return;
  struct mesh_state *words=mesh_plan_state(M);
  const uint32_t copies=words?words->copies:0;
  if(!copies || hops%copies)return;
  const uint64_t executions=atomic_load_explicit(&words->executions,memory_order_acquire),n=hops/copies;
  const struct mesh_hop *list=mesh_hops(M);
  for(uint32_t c=0;c<copies;c++){
    const uint64_t serves=copy_executions(executions,copies,c);
    if(!serves)continue;
    for(uint64_t j=c*n;j<c*n+n;j++){
      const uint32_t direction=list[j].direction,node=list[j].node;
      if(direction!=MESH_RECEIVE || (from!=UINT32_MAX && node!=from))continue;
      const uint64_t count=list[j].count,pages=list[j].pages;
      if(!count)continue;
      atomic_thread_fence(memory_order_acquire);
      /* `hops` acquire, as link_prepare's check: a re-plan of the same size is seen as `plan` moved */
      if(atomic_load_explicit(&M->hops,memory_order_acquire)!=hops ||
         atomic_load_explicit(&M->plan,memory_order_relaxed)!=plan)return;
      count_raise(region_word(count),pages*serves);
    }
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:619-654 streams_open (11807bec), reduced to what it is under the page lists:
   the tag's streams made new and live for a new pairing; here the port's serving opened, as its lanes connect and
   before any is served (mesh-child.c child_jobs): no plan loaded on the new pairing (`prepared` 0: its progress thread
   loads the plan written as the lanes are served, link_run), and the link's bandwidth told in the port's table, as
   d6dceac link_run told it at its pairing.  `node` is the port's peer, which the child already holds (`port->peer`,
   CHILD_CONNECT): the tag stored it in each stream */
void streams_open(struct port *port,uint32_t node,uint64_t bandwidth){
  (void)node;
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  __atomic_store_n(&link->bandwidth,bandwidth,__ATOMIC_RELAXED);
  atomic_store_explicit(&link->port.prepared,0,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:699-711 streams_cancel (11807bec), reduced to what it is under the page
   lists: the tag's streams of a port live no more and their words ~0; here, as its link is lost or the bridge stops
   (mesh-ports.c port_lost, mesh-children.c child_reaped, the stop), this node's receives not landed from the port's
   peer (its index in the map; none where the map holds no record of it), every one where `port` is NULL, cancelled
   (plan_cancel: each one's count raised to its final value, no phase written; design/disjoint-pages-change-request.md,
   C3.1: "A link loss is such a cancellation"), and the port's (every port's) plan no longer loaded (`prepared` 0) where
   no child of it lives: a live child's requests may still name the plan's pages, so its `prepared` is its own to clear
   (link_close) or its reap's (child_reaped, after `port->child` is 0).  A receive the port's child lands after the
   cancellation flips its phase and adds its arrival beyond the final value, which a reader's test (the count at least
   its target) reads as the raise; the tag's cancel again (child_heard's connect of a lost pairing, child_reaped)
   raises the counts of the executions the plan's `executions` asks by then (2026-11-04, sub-assignment 6: the cancel
   word, UINT64_MAX and never cleared, gone: C4).  A cancel again comes after the loss has zeroed `port->peer` (mesh-ports.c
   port_lost, which cancels first), so each port's peer at its last cancel with one is kept (`cancelled`, the parent's,
   its node, its map index then, or every receive where the map was not read) and a cancel with no peer cancels by it,
   as at the tag (review R2, C1).  The parent never sees its child load a plan, so the peer is kept at that first cancel,
   where the loss's own cancel reads it */
static struct cancelled { uint32_t node,index,every; } cancelled[MESH_PORTS];
void streams_cancel(const struct port *port){
  if(!port){
    plan_cancel(UINT32_MAX);
    for(uint32_t p=0;p<port_count;p++)
      if(!ports[p].child)atomic_store_explicit(&mesh_links(M)[ports[p].index].port.prepared,0,memory_order_release);
    return;
  }
  struct cancelled *last=cancelled+port->index;
  if(port->peer){
    struct mesh_record *records=NULL;uint64_t count=0,serial=0;
    const int error=map_read(&records,&count,&serial);
    *last=(struct cancelled){.node=port->peer,.index=error?UINT32_MAX:map_index(records,count,port->peer),.every=error!=0};
    free(records);
  }
  if(last->node){
    if(last->every)plan_cancel(UINT32_MAX);
    else if(last->index!=UINT32_MAX)plan_cancel(last->index);
  }
  if(!port->child)atomic_store_explicit(&mesh_links(M)[port->index].port.prepared,0,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:712-722 streams_let_go (60b00774), reduced to what it is under the page
   lists: the tag let a gone client's streams go; here, on the bridge's turn over its clients (mesh-records.c
   clients_turn, before its release of retired claims), where no client holds the region (hdr `client` 0) and a plan
   stays written, this node's receives of it cancelled (plan_cancel: each one's count raised to its final value, no
   phase written), so no consumer reads an execution of a plan whose client is gone as whole; a relay's forward of a
   page not landed is never released (its phase unchanged).  A client's retirement (mesh-dataflow.c mesh_retire)
   stores `hops` 0 before it closes the client's claims, so its own plan is no longer written here and this finds none
   (2026-11-04, sub-assignment 6: the cancel word gone, C4) */
void streams_let_go(void){
  if(atomic_load_explicit(&M->client,memory_order_acquire))return;
  plan_cancel(UINT32_MAX);
}

#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>
#include <os/os_sync_wait_on_address.h>
/* design/pages-and-functions.md#block-addressing */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
/* 105: the page lists (struct mesh_hop) and the bridge's pages (struct mesh_bridge_pages) in the layout, the map's
   fields (struct mesh_record) in the header, `executions` in place of its padding
   (design/page-queue-change-request.md).  106: the tag's header fields `boot`, `health`, `busy`, `idle_since`,
   `stopping` appended.  107: the --link form's and the per-link, per-queue transfer lists' header fields removed,
   their bytes left vacant so every other field keeps its place (struct hdr), and their areas gone from the layout.
   108: the plan's ring and its per-page cells gone (design/disjoint-pages-change-request.md, "Claude's proposal: the
   canonical path", items 1-3): hdr `depth` and `hop_stride` left vacant (`vacant7`, `vacant8`); the supplied words, the
   landed list and the map's records laid out as their own areas (`word_off`, `landed_off` and their room appended,
   `map_off` the layout's with `map_room`); struct mesh_hop's `forwards` gone and `position` its `mask`; struct
   mesh_publication's `landed`; struct mesh_cancel_range's `stride` */
#define MESH_VERSION 108u
#define MESH_ABSENT UINT32_MAX
/* tag pre-return-20261103 etc/bridge/layout.h MESH_PLATFORM_NS, verbatim: a wait's deadline where its caller gives
   none; the operator's deadline (2026-10-18): no DMA or network send/receive channel of the mesh is used longer than this
   without something happening on it or its scope concluding */
#ifndef MESH_PLATFORM_NS
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#endif
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
/* 2026-11-03, sub-assignment 1 (design/disjoint-pages-change-request.md, Appendix A, A4): the row planes MESH_ROW_HOT
   and MESH_FREE went with the call-site allocators and their reclaim, which alone read and wrote them; the layout sizes
   the one plane left (MESH_PLANES).  What must stay: MESH_ROW_OWN holds the rows of each program's one claim */
enum { MESH_ROW_OWN, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages,constant;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M10 */
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
  _Atomic uint64_t device;
};
_Static_assert(sizeof(struct mesh_page_entry)==32 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8 && offsetof(struct mesh_page_entry,device)==16,"mesh_page_entry");
/* design/collective-dependency-ledger.md#d5-receive-consumption-has-per-queue-fifo-order */
enum { MESH_SEND, MESH_RECEIVE };
/* design/prepared-machine.md#M04 */
struct mesh_send {
  _Alignas(128) _Atomic uint64_t ready;
  uintptr_t pair;
  _Alignas(32) struct ibv_sge span;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 &&
  offsetof(struct mesh_send,request.send_flags)+sizeof(unsigned int)<=128,"M04/M29");
struct mesh_target { uint64_t stream; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M10 */
/* a row's publication, which the plan call writes on instance 0's rows of a declared section (108,
   design/disjoint-pages-coordination.md, "The plan's words", its rows' publications): a destination page's `landed`
   its bit b in its execution's landed record, `device_input` hdr `landed_off` (execution 0's record) and
   `device_stride` hdr `landed_stride`; a source range's first row's `sends` its targets, `targets[i]` {the region
   offset of mesh_supplied(m, r, 0), count 1, stride 1} and `argument` MESH_SUPPLIED, execution t's word at `stream` +
   t x MESH_WORD_STRIDE (struct mesh_hop, "Supplied words" and "The landed list").  `landed` takes the padding at
   offset 4 (2026-11-03, sub-assignment 1), so no field moves */
struct mesh_publication { _Alignas(64) uint32_t sends; uint32_t landed; _Atomic uint64_t argument; uint64_t device_input,device_stride; struct mesh_target targets[]; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 && offsetof(struct mesh_publication,landed)==4 && offsetof(struct mesh_publication,argument)==8 && offsetof(struct mesh_publication,device_input)==16 && offsetof(struct mesh_publication,device_stride)==24 && offsetof(struct mesh_publication,targets)==32,"mesh_publication");
/* design/prepared-machine.md#M13 */
struct prepared_publication { _Alignas(16) uint64_t destination; uint64_t argument,padding[2]; };
_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==16 && offsetof(struct prepared_publication,argument)==8,"M13");
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };
/* the bridge-side region types (the map's records and cables, the link table, the layout) are mesh-map.h's, which no
   client includes (reviews R1, C1 and R2, B2; design/page-queue-change-request.md, "Queue pairs and links belong to the
   bridge"); struct mesh_port_info stays here, struct hdr `port` being one */

/* tag pre-return-20261103 rdma/mesh-observe.h MESH_PORTS: the most ports (there MESH_STREAMS / MESH_LINK_STREAMS) */
#define MESH_PORTS 8u
/* tag pre-return-20261103 rdma/mesh-observe.h struct mesh_holder, verbatim */
/* a holder of the region (a port's child): its pid, its device (`unit`: its interface's unit + 1; 0, any, as builds
   before 8ac5c9fb's form wrote) and its start (mesh-holders.h mesh_started) */
struct mesh_holder { uint32_t pid,unit; uint64_t start; };
/* tag pre-return-20261103 rdma/mesh-observe.h struct mesh_bridge_pages, verbatim, at the layout's bridge area
   (`bridge_off`) rather than after the tag's MESH_PORTS link infos */
/* the bridge's own pages of the region, recorded in the bridge area so the next bridge adopts them with the region
   instead of leasing them again: each port's link pages and its streams' lists, the duds' block and the map's pages
   (MESH_ABSENT where none; 2026-11-03: the bridge leases no arena page, so `link`, `ones` and `map` are recorded
   MESH_ABSENT and `map_pages` 0 and none is read for a page, the map's records being the layout's area at hdr
   `map_off` from 108; the fields stay so none moves); and the processes of the bridge that hold the region, each by
   its pid and start (mesh_started): the bridge's own start (beside the header's `bridge_pid`) and each child, which
   registers the region and holds it until its exit, in an entry no live child holds (twice the ports: an earlier
   launch's child that has not ended keeps its entry), so the heartbeat unlinks an expired region only once none of
   them lives.  Fields are only ever appended */
struct mesh_bridge_pages { uint32_t link[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS]; };
_Static_assert(sizeof(struct mesh_holder)==16 && offsetof(struct mesh_bridge_pages,bridge_start)==(MESH_PORTS+4)*sizeof(uint32_t),
               "mesh_bridge_pages");

/* 2026-11-03, sub-assignment 1 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", items
   1-3; Appendix A, A4): reversing this comment's 105-107 form (17612857, 90962999, 52aeae89), the plan's ring and its
   cells are gone: hdr `depth`'s slots (execution t on slot t mod depth's page, a receive held on earlier executions'
   landings and on its page's forwards, struct mesh_hop's `forwards`) and the 256-byte cell a word an execution (hdr
   `hop_stride`, a landed word a page, its sent count at offset 8).  Why: the ring reused pages under waits between
   executions, which nothing asked for, and the per-page words had every reader fence one word a page for a landing
   the bridge already learns from its completion (the change request, "The turns").  What must stay: the list
   unrolled, an entry an execution's own hop; no hop of one execution waits on another's; a landing sets one bit of its
   execution's landed record, in the landed list's own area, and writes nothing beside its page. */
/* The page lists (design/page-queue-change-request.md, "The page lists (the queue)", superseded in part by
   design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path"): this node's hops, which the plan
   call writes once a plan and the bridge alone reads.  An entry is one hop of one page in one execution:
   - `page`: an arena page of this region (mesh_at), a multiple of `block`, its block within the registered window
     (mesh_wired): the page sent, the page a receive lands in, or a relay's page, a page like any other.  A hop moves
     that whole page, `block` arena pages (block x pgsz bytes), from it or into it; nothing is copied.
   - `node`: the adjacent node by its index in the map (hdr `map_serial`): a send's next hop, a receive's previous hop.
   - `direction`: MESH_SEND or MESH_RECEIVE; `padding` unused.
   - `word`: the region offset of the _Atomic uint64_t the hop names in its execution, a supplied word or a word of a
     landed record (below), never a cancel word; `mask`: the bits of it the hop names.  No SGE names either area.
   - `serial`: the map serial the plan was made against, every entry's the same.
   Executions (main's invocations).  A plan runs hdr `executions` times, t = 0 .. executions - 1, unrolled
   (Appendix A, A3, "Repeated executions are unrolled, not ringed"): the list is `hops` = executions x n entries,
   execution t's n at [t n, t n + n).  Every execution's entries are the same hops in the same order, with the same
   `node`, `direction` and `serial`; `page`, `word` and `mask` are execution t's own.  The plan's one allocator assigns
   every execution's pages before the first runs, two value instances sharing a page only where the plan's dependence
   graph orders every read of the one before the other's first write (Appendix A, A3, and Appendix B: as CUDA graph
   memory nodes reuse memory), so nothing live at once shares a page and no execution waits on another's pages.
   - What releases a send: `(atomic_load(word, acquire) & mask) == mask`.  A send from a range's source names its
     range's supplied word of its execution (`mask` MESH_SUPPLIED); a forwarding send (a relay's, or a destination's
     that forwards the page on) names the `word` and `mask` of its node's receive of the same page in the same
     execution, so that landing releases it.  Several sends may name one word (a range's pages released by one store, a
     page sent to several nodes).
   - What a receive's landing does: `atomic_fetch_or(word, mask, release)` once the page's bytes are in its page, its
     bit of its execution's landed record (below) set; nothing else is written for it.
   Supplied words (the change request, proposal item 3: a supplied word stays what releases a send, and one store
   still releases a range): `word_capacity` bytes at hdr `word_off`.  Release r (this node's r-th range it is the
   source of, in the plan's demand order) of execution t is mesh_supplied(m, r, t), a release's executions
   MESH_WORD_STRIDE apart.  The plan call zeroes those it uses; the producer stores MESH_SUPPLIED (release) after its
   last write to the range in execution t (the GPU's publication, prepared_publication `argument`, or
   mesh_host_publish).  Nothing cancels a supplied word: 0 releases nothing.
   The landed list (proposal item 2): `landed_capacity` bytes at hdr `landed_off`, a contiguous array in its own area
   of the region beside the page table and the page lists, not beside the pages it tells of.  Execution t's landed
   record is mesh_landed_record(m, t), hdr `landed_stride` bytes, landed_stride = 32 x ceil((1 + ceil(R / 64)) / 4)
   where R is this node's receive hops in one execution (the same in every execution; R = 0 gives 32), so a record is
   whole 32-byte vectors.  Its word 0 is the execution's cancel word: 0, or UINT64_MAX once cancelled.  Bit b (b < R,
   the execution's b-th receive hop in list order) is bit b % 64 of word 1 + b / 64: that receive's `word` is the region
   offset of that word and its `mask` 1 << (b % 64).  A destination's pages of one range are consecutive receive hops at
   that node, so a range's bits are one run [first, first + P) (P its pages), and whether a range has landed is a few
   words of one record (its run's words and the cancel word), which a reader takes with vector loads, not a word a page
   (proposal items 2 and 4).  Read-after-write is honoured by that mask, not by waiting: work whose input pages have
   not landed is predicated off and taken on a later pass, and nothing spins on a page (proposal item 2).  The plan
   call zeroes records 0 .. executions - 1; nothing clears a bit while the plan stands.
   Cancellation: UINT64_MAX in a cancel word, which releases no send (no hop names a cancel word).  The bridge cancels
   a receive not landed (link_close, plan_cancel) by storing it into its execution's cancel word, the record's start
   landed_off + ((word - landed_off) / landed_stride) x landed_stride; the program cancels its executions by one
   mesh_cancel range {offset landed_off, count executions, stride landed_stride}; and a GPU lane whose reads of a
   step's mask reach their bound stores it into that step's cancel word (metal-microbench metal_recording.m, a remote
   read), so the step is cancelled where the host reads its record after the run.
   Both ends: for adjacent nodes a and b, execution t's entries at a naming b with MESH_SEND and those at b naming a
   with MESH_RECEIVE are the same hops in the same order.  Both bridges assign the pair's hops in list order (execution
   0's, then execution 1's, ...) to the pair's cables and queue pairs by one rule from the map (struct mesh_reach;
   design/page-queue-change-request.md, "The rule both ends compute"), and each queue pair carries its hops in that
   order: the k-th SEND on a queue pair fills the k-th receive posted on it.
   Writing a plan (design/disjoint-pages-coordination.md, "Writing a plan", and the coordinator's rulings on its zeroing
   and its erase; mesh-collective.c mesh_plan): the plan call computes everything first (its demands, routes, values,
   dependence graph and allocation), and refuses a plan past `hop_capacity`, `word_capacity` or `landed_capacity`
   (ENOSPC) or one the allocator finds no room for in the program's window (ENOMEM), so a failure there leaves the
   previous plan standing.  Then, in this order, it
   1. stores `hops` 0 (release) and a release fence, and wakes `plan`;
   2. awaits no port's `prepared` holding the previous generation (mesh_plan_erased: a seq_cst fence, then a wait
      bounded by MESH_PLATFORM_NS, ETIMEDOUT past it), so the areas and pages the previous plan named are the next
      plan's to reuse;
   3. writes `executions` and `landed_stride`, so mesh_supplied and mesh_landed_record name this plan's words wherever
      it zeroes and writes them;
   4. binds the declared sections' rows to their pages and zeroes every page it binds, since a peer's SEND held in
      credit lands as soon as this node's bridge loads the plan (a program zeroes no routed operand after mesh_plan);
   5. zeroes the supplied words and landed records it uses;
   6. writes the rows' publications (struct mesh_publication) and the entries, then `plan`, the plan's generation (the
      last plan's + 1, never 0);
   7. reads the map again: ESTALE where its serial moved, or the map reader's own error, either leaving `hops` 0;
   8. stores `hops` (release), and wakes `plan`.
   The bridge reads none of it while `hops` (acquire) is 0.  A port loading a plan stores its generation in its
   mesh_links(m)[p].port.prepared before its last reading of `hops` and `plan`, with a seq_cst fence between, and
   clears `prepared` where either moved, so no port begins serving a plan the plan call has found erased.  A plan whose entries' `serial` is not `map_serial` when it is loaded is refused: the call fails by error and its
   caller plans again on the map as it is. */
struct mesh_hop { uint32_t page,node,direction,padding; uint64_t word,mask,serial; };
_Static_assert(sizeof(struct mesh_hop)==40 && offsetof(struct mesh_hop,word)==16 && offsetof(struct mesh_hop,mask)==24,"mesh_hop");
/* the value a producer stores into a supplied word (struct mesh_hop, "Supplied words") */
#define MESH_SUPPLIED UINT64_C(1)
/* the bytes between one supplied word's consecutive executions (mesh_supplied) */
#define MESH_WORD_STRIDE 8u

/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
/* 2026-11-04, sub-assignment 16 (design/page-queue-change-request.md, "Order, and what judges it"): the header fields
   only the --link form and its per-link, per-queue transfer lists read were removed at 107, and their bytes are left
   vacant (`vacant0` .. `vacant6`, never read or written) rather than closed up, so every other field keeps the offset
   it had at 106.  Why: a region is read at these offsets after its layout is expired (version 0, so no layout check
   applies): mesh-link.c region_open takes its `bridge_off` and the children recorded there, and signals the running
   ones (a recorded start of 0 matches the pid alone), and mesh-heartbeat.c region_own waits on those holders.  Closing
   the gap would move `bridge_off` from 296 to 248, where a 106 region keeps `hops`, and its pages would be read as
   pids.  What must stay: no field moves (the assert below); a field is only ever appended. */
/* 2026-11-03, sub-assignment 1 (design/disjoint-pages-change-request.md, Appendix A, A4, "the hdr fields `depth` and
   `hop_stride` that serve them"): at 108 the plan's ring depth `depth` and word stride `hop_stride` went with the ring
   and its cells (struct mesh_hop's note), their bytes left vacant the same way (`vacant7`, `vacant8`), and the layout's
   areas of the supplied words, the landed list and the map's records appended after `stopping`.  What must stay: as
   above. */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,vacant0,links;
  uint32_t vacant1,wire_pages;
  /* the ring depth's bytes until 108 (the note above), never read or written */
  uint32_t vacant7;
  /* the page lists: the executions of the plan written (struct mesh_hop), main's invocations */
  uint32_t executions;
  uint64_t vacant2;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,vacant3,target_off,vacant4,vacant5,data_off,length;
  uint64_t vacant6,target_stride;
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
  /* the page lists (struct mesh_hop, mesh_hops): room for `hop_capacity` entries at `hop_off` (mesh_geometry `hops`),
     `hops` the entries of the plan written (0 none) and `plan` its generation (the last plan's + 1, never 0; each
     port's mesh_links(m)[p].port.prepared the generation the bridge loaded); `vacant8` the word stride's bytes until
     108 (the note above), never read or written */
  uint64_t hop_off,hop_capacity,vacant8;
  _Atomic uint64_t hops,plan;
  /* tag pre-return-20261103 rdma/mesh.h struct hdr `map`, `map_off`, `map_nodes`, verbatim: the map's seqlock
     (mesh_notify, mesh_map_whole) and its `map_nodes` records at `map_off` (mesh_records), at most `map_room`.  From
     108 `map_off` is the layout's (mesh-map.h mesh_layout), set once with the region, where it was arena pages the
     bridge leased and leased again larger (design/disjoint-pages-change-request.md, Appendix A, A3, "The bridge's own
     pages (its records) are laid out once with the region") */
  _Atomic uint64_t map,map_off,map_nodes;
  /* the map's serial: moved by one (mesh-records.c, while `map` is odd) at every change of the map's nodes or cables
     (a record added or dropped, any field of a reach entry); a plan is made against it (struct mesh_hop `serial`).
     A node's index is its place in ascending node-id order among the map's records. */
  _Atomic uint64_t map_serial;
  /* the bridge's pages (struct mesh_bridge_pages) at `bridge_off`; the attached client's start (mesh-holders.h
     mesh_started), with `client`'s low 32 bits its identity as a holder of the region */
  uint64_t bridge_off;
  _Atomic uint64_t client_start;
  /* tag pre-return-20261103 rdma/mesh.h struct hdr `boot`, `health`, `busy`, `idle_since`, `stopping`, verbatim (their
     tag types; read by the tag's mesh-records.c and mesh-link.c region_open and main): `boot` the bridge's start on the
     wall clock (ns, CLOCK_REALTIME), its records' boot; `health` the serial a health-only change of a record moves,
     never the map's (mesh-records.c health_put); `busy`: the streams held (0 between calls) and since when none has
     been (`idle_since`, the wall clock in ns; 0 while any is); `stopping` stored 0 as the region opens */
  uint64_t boot;
  _Atomic uint64_t health,busy,idle_since,stopping;
  /* 108 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", items 2 and 3; Appendix A,
     A3): the layout's (mesh-map.h mesh_layout) supplied words, `word_capacity` bytes at `word_off` (mesh_supplied),
     its landed list, `landed_capacity` bytes at `landed_off` (mesh_landed_record), and the map's room, `map_room`
     records at `map_off`; and the plan's `landed_stride`, the bytes of each of its landed records, which the plan call
     writes with `executions` once the previous plan is erased, before it binds its rows and zeroes and writes its words
     through them, and before `hops` (struct mesh_hop, "Writing a plan") */
  uint64_t word_off,word_capacity,landed_off,landed_capacity,landed_stride,map_room;
};
/* every field at its 106 offset, the 108 fields after them (the vacant bytes' notes above struct hdr) */
_Static_assert(offsetof(struct hdr,links)==28 && offsetof(struct hdr,wire_pages)==36 && offsetof(struct hdr,planes_off)==56 &&
  offsetof(struct hdr,target_off)==104 && offsetof(struct hdr,data_off)==128 && offsetof(struct hdr,target_stride)==152 &&
  offsetof(struct hdr,client)==160 && offsetof(struct hdr,hop_off)==224 && offsetof(struct hdr,map_serial)==288 &&
  offsetof(struct hdr,bridge_off)==296 && offsetof(struct hdr,client_start)==304 && offsetof(struct hdr,stopping)==344 &&
  offsetof(struct hdr,vacant7)==40 && offsetof(struct hdr,executions)==44 && offsetof(struct hdr,vacant8)==240 &&
  offsetof(struct hdr,hops)==248 && offsetof(struct hdr,plan)==256 && offsetof(struct hdr,map_off)==272 &&
  offsetof(struct hdr,word_off)==352 && offsetof(struct hdr,landed_stride)==384 && offsetof(struct hdr,map_room)==392 &&
  sizeof(struct hdr)==400,"struct hdr: no field moves");
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/prepared-machine.md#M12 */
struct mesh_cancel_range { _Alignas(32) uint64_t offset; uint64_t count,stride,padding; };
struct mesh_cancellation { _Alignas(32) _Atomic uint32_t requested; uint32_t padding[7]; struct mesh_cancel_range ranges[]; };
_Static_assert(sizeof(struct mesh_cancel_range)==32 && offsetof(struct mesh_cancel_range,count)==8 && offsetof(struct mesh_cancel_range,stride)==16 &&
  sizeof(struct mesh_cancellation)==32 && offsetof(struct mesh_cancellation,ranges)==32,"M12");
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M12 */
/* range `link` of a cancellation: for j < count, the word at region offset `offset` + j x `stride` stored UINT64_MAX
   (release) where it reads 0, and `requested` set.  `stride` from 108 (2026-11-03, sub-assignment 1), where the words
   were consecutive: the program's one range is every execution's cancel word of the landed list, {landed_off,
   executions, landed_stride} (struct mesh_hop, "Cancellation") */
static inline void mesh_cancel(struct hdr *m,struct mesh_cancellation *cancel,uint32_t link){
  const struct mesh_cancel_range range=cancel->ranges[link];
  for(uint64_t j=0;j<range.count;j++){
    _Atomic uint64_t *word=(_Atomic uint64_t *)((char *)m+range.offset+j*range.stride);
    if(!atomic_load_explicit(word,memory_order_relaxed)){
      atomic_store_explicit(&cancel->requested,1,memory_order_release);
      atomic_store_explicit(word,UINT64_MAX,memory_order_release);
    }
  }
}
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M26 */
static inline void mesh_control_notify(struct hdr *m){
  atomic_fetch_add_explicit(&m->control,1,memory_order_release);
  os_sync_wake_by_address_all(&m->control,sizeof m->control,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* tag pre-return-20261103 rdma/mesh.h mesh_notify, verbatim */
/* design/algorithm-sources.md#meshresult: a shared word (`control`, `map`) bumped and its waiters woken, the writes
   after it ordered behind it */
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/algorithm-sources.md#index-hand-off */
static inline struct mesh_publication *mesh_publication_at(struct hdr *m,uint32_t row){return (struct mesh_publication *)((char *)m+m->target_off+(size_t)row*m->target_stride);}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
/* design/prepared-machine.md#M09 */
static inline uint32_t mesh_blocks(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz)); }
/* design/pages-and-functions.md#what-the-page-table-is */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return mesh_blocks(m)*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/prepared-machine.md#M01 */
/* the arena occupancy bitmap is indexed by arena page and sized by arena pages: it is not a row plane */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* design/prepared-machine.md#M09 */
/* the registered window is the first wire_pages of the arena; every other page is addressable and unregistered */
static inline uint64_t mesh_wire_bytes(const struct hdr *m){ return (uint64_t)m->wire_pages*m->pgsz; }
static inline int mesh_wired(const struct hdr *m,uint64_t offset,uint64_t bytes){ return offset<=mesh_wire_bytes(m) && bytes<=mesh_wire_bytes(m)-offset; }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* the page lists: the plan's entries (struct mesh_hop), hdr `hops` of them at `hop_off` */
static inline struct mesh_hop *mesh_hops(struct hdr *m){ return (struct mesh_hop *)((char *)m+m->hop_off); }
/* the supplied word of release `release` (this node's release-th range it is the source of, in the plan's demand order)
   in execution t: a release's words of every execution consecutive, MESH_WORD_STRIDE apart, at `word_off` (struct
   mesh_hop, "Supplied words") */
static inline _Atomic uint64_t *mesh_supplied(struct hdr *m,uint64_t release,uint64_t t){
  return (_Atomic uint64_t *)((char *)m+m->word_off+(release*m->executions+t)*MESH_WORD_STRIDE);
}
/* execution t's landed record, `landed_stride` bytes at `landed_off`: its cancel word, then its receive hops' bits
   (struct mesh_hop, "The landed list") */
static inline _Atomic uint64_t *mesh_landed_record(struct hdr *m,uint64_t t){
  return (_Atomic uint64_t *)((char *)m+m->landed_off+t*m->landed_stride);
}
/* tag pre-return-20261103 rdma/mesh.h mesh_records and mesh_map_whole, verbatim; mesh_records moved to mesh-map.h
   (review R1, C1) */
/* the map, `map_nodes` records at `map_off`, at most `map_room` (the layout's area of the map from 108, where it was
   arena pages the bridge leased for itself, and leased again larger as the mesh grew); the bridge notifies `map`, its
   serial, before and after each change (odd while it writes; its own record is in before the region opens, so a
   client never reads 0), and a reader's copy is whole where `map` read (acquired) before it was even and is unchanged
   after it */
static inline int mesh_map_whole(struct hdr *m,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(&m->map,memory_order_relaxed)==before;
}
/* tag pre-return-20261103 rdma/mesh-observe.h mesh_bridge_pages, at the layout's bridge area (`bridge_off`), no link
   infos before it */
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off);
}
/* 2026-11-03, sub-assignment 1 (review 12, second round, finding 4): mesh_row_chunks (design/algorithm-sources.md
   #programcopy, design/prepared-machine.md#M08), the blocks a row's bytes span, is removed: its last callers, the
   page-cut forms of mesh_host_arrived and mesh_metal_receive_prepare, are gone with the per-page words
   (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", items 3 and 4), and nothing in
   mesh or metal-microbench calls it.  What must stay: a range is read by its bit run in a landed record, never by its
   chunks */

static inline uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a) return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
static inline void mesh_bits_set(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_or_explicit(&p[w],mesh_word_mask(first,count,w),memory_order_acq_rel);
}
static inline void mesh_bits_clear(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_and_explicit(&p[w],~mesh_word_mask(first,count,w),memory_order_acq_rel);
}
#endif

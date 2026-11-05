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
   mesh_publication's `landed`; struct mesh_cancel_range's `stride`.  109: the plan's state section (struct
   mesh_state at hdr `state`, appended) in place of the layout's supplied words and landed list (hdr `word_off` ..
   `landed_stride` left vacant, `vacant9` .. `vacant13`, and `executions` `vacant14`), the receives' 2-bit phases and
   counts, struct mesh_hop's `field`, `count` and `pages` (48 bytes), struct mesh_publication's `pages`, `phases`,
   `count` and `state`; the cancel word, mesh_cancel and the GPU's publication records gone
   (design/disjoint-pages-change-request.md, Appendix C, C3 and C4; design/disjoint-pages-revision-coordination.md,
   "The region").  110: plans in their claims, any number standing (R8, A7); hdr's one-plan fields vacant; `plans`
   appended; struct mesh_hop `bytes`; struct mesh_state's name, entries, holders and sequence words; struct mesh_buffer
   `start` (design/plan-to-bridge-change-request.md R5 and R8; design/aot-compilation-change-request.md A7;
   design/compilation-coordination.md, "The region").  111: struct mesh_hop `set` and `padding1` appended (56 bytes): a
   hop's set, 0 a sequence (design/aot-compilation-change-request.md, appendix item 5); struct mesh_state `epoch` the
   plan's function (item 6) */
#define MESH_VERSION 111u
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
   the one plane left (MESH_PLANES).  What must stay: MESH_ROW_OWN holds the rows of every claim (each plan's, any number
   at once: 2026-11-04, owner L, where it said each program's one claim) */
enum { MESH_ROW_OWN, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
/* A claim's head row, its first, which no value's rows include (mesh-dataflow.h mesh_claim_take), is the claim's
   record (design/compilation-coordination.md, "The region"; design/plan-to-bridge-change-request.md R8):
   - its struct mesh_buffer: `owner` the client word (stored last, release; 0 once released), `closed` 1 once closed,
     `pages` the claim's rows, `constant` 0, `start` mesh_started(getpid()) of the claiming process (mesh-holders.h),
     with `owner`'s low 32 bits its identity as a holder of the region;
   - its struct mesh_page_entry: `mapping` the window run, `address` the local run, `device` the claim's generation;
   - its struct mesh_publication: `state` the region offset of the plan's state section (struct mesh_state), stored
     with release once the plan stands and 0 before, never cleared while the claim lives; every other field 0.
   A plan stands where its head's `owner` != 0, `closed` == 0 and publication `state` != 0 (mesh-dataflow.h
   mesh_head_state).  No row but a head holds an owner word.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): `start` in the 8 bytes that were padding,
   where hdr `client_start` held the one client's start; any number of claims, each its claimer's.  What must stay:
   32 bytes, no field moves (the assert below) */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages,constant;
  uint64_t start;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32 && offsetof(struct mesh_buffer,closed)==8 &&
  offsetof(struct mesh_buffer,pages)==12 && offsetof(struct mesh_buffer,constant)==16 &&
  offsetof(struct mesh_buffer,start)==24,"mesh_buffer");
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
/* a publication's target: `word` the region offset of a source range's supplied word, `count` 1, `stride` 0 (struct
   mesh_publication below).  2026-11-04, sub-assignment 2 (the operator's ruling of 2026-11-04 on transport names):
   `stream` renamed `word`; the size and every offset unchanged (the assert below) */
struct mesh_target { uint64_t word; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M10 */
/* a row's publication: where a client finds a section's words in the plan's state section (struct mesh_state; struct
   mesh_hop's note, "The state section"), by the section's first row.  mesh_plan writes every row it claims,
   once, with the plan's other rows (design/disjoint-pages-change-request.md, Appendix C, C3.2;
   design/disjoint-pages-revision-coordination.md, "The region"), and nothing else writes it.  Every field is 0 but:
   - a received range's first row (a range this node consumes; copy 0's first row): `pages` its pages P; `phases` the
     region offset of its copy-0 phase group, copy c's at phases + c x mesh_group_bytes(P); `count` the region offset
     of its copy-0 count word, copy c's at count + 8 c; `state` the region offset of its plan's state section;
   - a source range's first row: `sends` 1; `targets[0]` {word: the region offset of its supplied word, count 1,
     stride 0}; `state` the region offset of its plan's state section;
   - a claim's head row: `state` as struct mesh_buffer's note states it (stored with release, read with acquire), every
     other field 0.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): a row's `state` its own plan's state
   section, where it was hdr `state`, the one plan's; any number of plans stand at once.  What must stay: a row of a
   plan names that plan's state section.
   2026-11-04, sub-assignment 1 (C3.1, C3.7 and C4): at 109 `landed` (a page's bit in its execution's landed record),
   `argument` (MESH_SUPPLIED), `device_input` and `device_stride` (hdr `landed_off`, `landed_stride`) became `pages`,
   `phases`, `count` and `state` at the same offsets, since the landed records and per-execution supplied words are
   gone.  What must stay: no field moves (the assert below) */
struct mesh_publication { _Alignas(64) uint32_t sends; uint32_t pages; uint64_t phases,count,state;
  struct mesh_target targets[]; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 &&
  offsetof(struct mesh_publication,sends)==0 && offsetof(struct mesh_publication,pages)==4 &&
  offsetof(struct mesh_publication,phases)==8 && offsetof(struct mesh_publication,count)==16 &&
  offsetof(struct mesh_publication,state)==24 && offsetof(struct mesh_publication,targets)==32,"mesh_publication");
/* a port's words: `prepared` the generation of the claim the port is loading now, 0 none, set and cleared within one
   load, the load's transient announcement, which mesh_retired_release reads before it releases a claim: an
   announcement lives in memory that reclamation cannot reuse, as a hazard pointer does (M. M. Michael, Hazard
   pointers: safe memory reclamation for lock-free objects, IEEE TPDS 15(6), 2004; design/compilation-coordination.md,
   ruling 1 and Q5).  A port that holds a plan says so in the plan's state section (struct mesh_state `held`).
   2026-11-04, owner L: reversing the 109 form, where `prepared` held the one plan's generation while the port served
   it.  Why: any number of plans stand, so a port's holding is per plan, in each plan's words.  What must stay: the
   announcement, a seq_cst fence, then the head read again (design/disjoint-pages-revision-coordination.md, D10). */
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
   the bridge already learns from its completion (the change request, "The turns").  What must stay: no slot reused
   under a run-time window; no hop waits on another execution's landing or forward; a landing writes nothing in or
   beside its page. */
/* 2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, Appendix C, C3.1-C3.4, C3.7 and C4;
   design/disjoint-pages-revision-coordination.md, "The region"): reversing this comment's 108 form (67631dbe,
   e3c2442c, 430c3cda): the list unrolled over the run's length (hdr `executions`, executions x n entries); the supplied
   words and the landed list as fixed areas of the layout (hdr `word_off`, `landed_off`, their room and
   `landed_stride`; mesh_supplied, mesh_landed_record), which every plan zeroed in place behind mesh_plan_erased's
   fence pair and wait; a supplied word an execution, stored MESH_SUPPLIED (MESH_WORD_STRIDE apart), and the GPU's
   publication records (struct prepared_publication); the landed record's ORed bits (struct mesh_hop's `mask`) and its
   cancel word, stored UINT64_MAX by link_close, plan_cancel, a GPU lane at its poll bound and the program's
   mesh_cancel (struct mesh_cancel_range, struct mesh_cancellation), five writers of one word.  Why: the operator's
   rulings (C1): the zeroing of every declared page and word is a class of waste to remove, the run's length entered
   compile time squared, and the erase, its wait and the cancel race it opened came from neither the specification
   nor the literature; so the template is unrolled into u copies by modulo variable expansion (C3.3), the plan's state
   is placed by its allocator in new pages for every plan (C3.2, C3.4), each state word has one writer and is never
   reset (C3.7), and a cancellation arrives on a count (C3.1).  What must stay: no data page written but by its
   producer or its landing; every state word one writer (counts take arrivals), set once where placed and never
   written to an earlier value while its plan stands; a plan's state in its own pages, apart from the data; the run's
   length no input of the plan. */
/* The page lists (design/page-queue-change-request.md, "The page lists (the queue)", superseded in part by
   design/disjoint-pages-change-request.md, Appendix C, C3, design/plan-to-bridge-change-request.md R5 and R8, and
   design/compilation-coordination.md, "The region"): a plan's hops at this node, which mesh_plan writes once, in the
   plan's state section (struct mesh_state `hops` and `entries`; mesh_hops), and the bridge serves (the library's
   mesh_transfers_start reads their `node` alone, for the ports it waits on).  An entry is one hop of one page in one
   copy:
   - `page`: an arena page of this region (mesh_at), a multiple of `block`, its block within the registered window
     (mesh_wired): the page sent, the page a receive lands in, or a relay's page, a page like any other.  A hop moves
     its `bytes` from the start of that page or into it; nothing is copied.
   - `node`: the adjacent node by its index in the map (hdr `map_serial`): a send's next hop, a receive's previous hop.
   - `direction`: MESH_SEND or MESH_RECEIVE.
   - `word`, `field`, `count`, `pages`: the state words the hop names, region offsets into the plan's state section
     (below): what releases a send, and what a receive's landing writes.  No SGE names any of them.
   - `bytes`: the bytes the hop moves: page k of a range of B bytes moves min(Q, B - k Q), Q = block x pgsz; every
     hop's, a send's and a receive's alike (design/plan-to-bridge-change-request.md R5; design/aot-compilation-change-
     request.md A9; design/compilation-coordination.md, T7).
   - `serial`: the map serial the plan was made against, every entry's the same and the state section's `serial`.
   - `set`: 0 for a hop of a sequence; else j + 1 for a page of tree j of a set range, a range every declared transfer
     of which is MESH_PAGES, j the tree's index in the solve's trees grouped by demand (the same at every rank), the
     same in both ends' entries of the hop (design/aot-compilation-change-request.md, appendix item 5;
     design/compilation-coordination.md, "The appendix's revisions", S1 to S3).  A set's entries are contiguous on its
     node pair's list within each copy, in walk order, both ends' alike (S3): the plan's view, the bridge's cables its
     own.  The bridge permutes a set's members per cable, each execution, at both ends (S4, S5, S7′: on each cable of
     the pair, the members whose positions the deal gave that cable, among those positions).  The bridge sees a set or
     a sequence and no kind (design/plan-to-bridge-change-request.md R4).  `padding1` 0.
   2026-11-04, owner L2 (the appendix, item 5): `set` and `padding1` appended, where every hop was one of a sequence in
   the schedule's order.  Why: the pages of function B's operands are written as a set, not a sequence (R4, item 5).
   What must stay: offsets 0-47 as at 110; both ends' entries of a hop the same `set`.
   2026-11-04, owner L2 (review W3's finding 12, the coordinator's ruling; S7′): a set contiguous on its node pair's
   list and permuted per cable, where this note said contiguous on its cable and permuted whole.  Why: the deal spreads
   a pair's positions over the pair's cables, and each cable's records are its own (S7′).  What must stay: as above.
   Copies (C3.3: modulo variable expansion, M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo
   scheduling, MICRO 1994).  mesh_plan unrolls the plan's one execution, its template, into u copies: L of a routed
   value is the least lag at which every read of it in execution t happens before its write in t + L, and u the
   largest L over the collective's routed values at every node (C3.3; design/compilation-coordination.md, U1), so
   every rank computes the same u; a value with no such L fails the plan with ENOMEM (C3.3).  The list is struct
   mesh_state `entries` = u x n entries at its `hops`, copy c's n at [c n, c n + n).  Every copy's entries are the same
   hops in the same order, with the same `node`, `direction`, `pages`, `bytes`, `serial` and `set`; `page`, `word`,
   `field` and `count` are copy c's own.  Execution t runs copy t mod u.  A plan runs any number of executions: the
   run's length is no input of mesh_plan, and the bridge serves the executions t < struct mesh_state `executions` (D2)
   and t < its `released` (below).  The plan's one allocator places every copy's pages before the first execution, two
   copies sharing a page only where the plan's happens-before orders every read of the one before the other's write at
   the distances at which their executions meet, so nothing live at once shares a page and no execution waits on
   another's pages.
   2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 1; design/compilation-coordination.md,
   "The appendix's revisions", U1): u as C3.3 states it, cited to C3.3 where it cited the coordination's D5, whose
   refusal of a write that does not reach its own next instance at lag 1 is gone.  Why: D5 is in neither Lam, Rau nor
   C3.3's text (item 1).  What must stay: u the largest least lag, the same at every rank; ENOMEM only for a value with
   no such L.
   - A send from its range's source: `word` the region offset of the range's supplied word; `field` MESH_ABSENT;
     `count` 0; `pages` 0.  Execution t's send is released once the word reads (acquire) at least t + 1.
   - A forwarding send (a relay's, or a destination's that sends the page on): `word` and `field` those of its node's
     receive of the same page in the same copy; `count` 0; `pages` 0.  Execution t's send is released once that field
     reads (acquire) mesh_phase(t / u + 1), its receive's landing in execution t.
   - A receive: `page` its page; `word` the region offset of the uint64 holding its phase field and `field` the
     field's low bit (even, < 64); `count` the region offset of its range's count word for this copy, or 0 where this
     node does not consume the range (a relay's own receive: its forwards are released by its phase, nobody waits on a
     count, and the state section has no count word for it); `pages` its range's pages P.  Its landing in execution t,
     once its `bytes` are in its page (after the poll's acquire), is one atomic XOR of bit `field` + ((t / u) & 1) of
     `word` (release), then, where `count` is not 0, the count + 1 (release), then, where the count it made is a
     multiple of `pages`, the count word's waiters woken (os_sync_wake_by_address_all, OS_SYNC_WAKE_BY_ADDRESS_SHARED).
     Nothing else is written for it, and nothing in or beside its page; for a receive whose `count` is 0 the bridge
     adds, wakes and raises nothing.
   Phases (C3.7: PTX's mbarrier phase bit, widened to two).  A receive of copy c lands in executions c, c + u, c + 2u,
   ...; its field starts at 00, and after execution t's landing holds mesh_phase(t / u + 1), Gray 00 01 11 10: a
   landing flips one bit, which never carries into the next field.  A reader of execution t compares the field with
   mesh_phase(t / u + 1): equal, landed; equal to mesh_phase(t / u), one behind (not yet, or cancelled); anything else,
   a write ran ahead of the allocator's order: an error.  A field is reused only with its page, so the allocator's
   order for the page covers it.
   The state section (C3.2, C3.7; struct mesh_state at its claim head's publication `state`, mesh_head_state, and at
   mesh_plan's `*plan`, the plan's handle, mesh_state_at; design/compilation-coordination.md, "The appendix's
   revisions", N2): the plan's state, the landed list's own section, which mesh_plan
   places with the plan's values in its claim (an invariant of the plan, no page-table row naming it), in new pages for
   every plan, apart from the data pages it tells of.  Its layout: the struct; the phases, counts, supplied words and
   status; the entries; each 64-aligned.  Each word has one writer:
   - `phases`: the phase list, a dense array of 2-bit fields.  For each range this node receives (one it consumes,
     or a relay's own, whose forwards its phases release), in the plan's demand order, and copy c, a group of
     mesh_group_bytes(P) bytes, groups in (range, c) order, range major; page k of the range (k < P) has its field at
     bits [2 (k % 32), 2 (k % 32) + 2) of uint64 word k / 32 of its group.  A group is whole GPU lane loads
     (MESH_LANE_BYTES, a compile-time 16: 64 fields), so a lane's load never spans two groups and a reader takes a
     range's copy with ceil(P / 64) loads.  A field's one writer is the bridge's data path that takes its receive.
   - `counts`: `ranges` x `copies` uint64 words, `ranges` the received ranges this node consumes (a relay's own
     receives have none), range g (in the plan's demand order) copy c's at index g x copies + c.  A count takes
     arrivals (+1 a landing; a cancellation raises it to its final value, below) and only rises; execution t of range
     g is complete at (t / u + 1) x P_g on the count of copy t mod u.  Per received range and copy, where C3.1 counts
     once a node (D1, a departure): a decode's exchanges are read in sequence, each computed from the read of the one
     before, so a pass held for every receive of its execution would wait on receives its peers send only after their
     own reads, and every pass would be held to its deadline; Portals 4's counting events (one counter an operation set)
     and PTX's mbarrier (one a tile) count once a unit of work.  Per copy, since with u >= 2 execution t + 1's
     landings into the other copy can arrive before t's are all in.
   - `supplied`: `releases` uint64 words, release r this node's r-th source range in the plan's demand order.  Its
     one writer is the range's producer, which raises it to t + 1 after its last write in execution t (release): a
     host producer stores t + 1 (mesh_host_publish), a GPU producer adds 1, its executions running in order (D9).  One
     word serves every execution; nothing cancels it.
   - `status`: one struct mesh_status, the GPU's alone: `behind` the later passes that found a field one behind,
     `ahead` those that found anything else, `first` the greatest ~t over them (0 none: the first failed execution is
     ~first; 32 bits, so exact below 2^32 executions).  The host reads it after the run.
   - `executions`: the executions the program has submitted, raised (never lowered) by its start
     (mesh_transfers_start, D2): the program's submission of those function executions to its node's bridge
     (design/aot-compilation-change-request.md A7; design/compilation-coordination.md, Q1), so the bridge posts exactly
     the receives its peers send and compile time never sees the run's length.
   - `copies` u; `receives` this node's receives a copy (the sum of the pages of every range it receives); `ranges`
     and `releases` as above.
   - `group` and `epoch`: the plan's name, (group, function), the same at every rank: `group` the launcher's
     (MESH_GROUP), `epoch` the program's ordinal of the compiled function, mesh_plan's `function`, kept by every replan
     of that function (mesh-collective.h mesh_plan; design/compilation-coordination.md, "The appendix's revisions",
     N1).  mesh_plan's alone.
     2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 6; N1, N5): `epoch` the function's
     ordinal, where it was the planning context's count of its mesh_plan calls (C-epoch, gone).  Why: compiled
     functions coexist, each its own plan (item 6), so a plan is named by its function, not by the order of the calls.
     What must stay: the name the same at every rank; no field moves.
   - `serial`: the map serial the plan was made on, every entry's `serial`.  mesh_plan's alone.
   - `hops`: the region offset of the plan's entries, `entries` (u x n) struct mesh_hop of them (above).  mesh_plan's.
   - `held`: a bit a port, bit p "port p holds the plan" (set by port p once it has loaded the plan, cleared by it as
     its last access once it has ended it; e6bd7db2's section 6, as design/compilation-coordination.md ruling 1 and Q5
     take it).  `refused[p]`: port p's refusal code of the plan, 0 none; port p its one writer.
   - `nodes`: the plan's nodes, a set of map indices (bit i, map index i): every source and destination of a declared
     transfer and every node a tree of it reaches (this node's own where it has none), the same at every rank.
     `sender`: the least of `nodes`, the bridge that forwards the plan's executions to the island's sequencer (Q2: the
     fixed sequencer's one sender a message, X. Défago, A. Schiper and P. Urbán, Total order broadcast and multicast
     algorithms: taxonomy and survey, ACM Comput. Surv. 36(4), 2004, section 7.1).  mesh_plan's alone.
   - `submitted`: the executions the bridge at `sender` has submitted to the sequencer, [submitted, executions) the
     next submission (Q2); the bridge parent's alone, at that node.
   - `released`: the executions this node's bridge parent has released to its ports, in sequence order: it delivers
     an ORDER of the plan once the plan stands here and every port holding the previously released plan has served it,
     then stores `released` its `end` (release; Q4: Défago et al., Fig. 5, the destinations' `pending` delivered in
     sequence order); the bridge parent's alone.  A port serves execution t only where t < `executions` and t <
     `released`, so this node's cables carry one plan's batch at a time, in sequence order.  The hold stays
     (design/compilation-coordination.md, "The appendix's revisions", O2): both ends of a cable must post the released
     executions of several plans in one order, the sequence order both ends' parents deliver, and what a delivery
     leaves in the region for the ports, the parent's child processes, is each plan's `released`, a count, which
     records no order between two plans' deliveries; what would carry that order is open for the operator (O2).
     2026-11-04, owner L2 (O2): the hold's reason O2's, where it was that a cable never interleaves two executions
     (Claude's words, which design/aot-compilation-change-request.md, appendix item 2, withdraws).  What must stay: a
     port serves t only below `executions` and `released`.
   - `served[p]`: the executions port p has served whole, every record of them done (every send completed and every
     receive landed), and `released` for a plan it carries nothing of (Q4); port p's alone.
   Every word is set to its starting value once, where mesh_plan places the section (D3), as mbarrier.init runs once
   a kernel: the phases, counts, supplied words, status, `executions`, `held`, `refused`, `submitted`, `released` and
   `served` 0.  None is reset, and none written to an earlier value, while the plan stands, but a port's bit of `held`,
   which it clears as its last access.  `submitted`, `released` and `served` only rise; they, each bridge parent's
   ORDERs not yet delivered and the sequencer's own number are what A7 adds to C3.3's run time (Q6).
   Cancellation (C3.1: cancellations arrive on the count, as arrivals do on an mbarrier).  Where a plan ends with
   receives not landed, and where the bridge cancels (mesh-flow.c plan_cancel), each such receive's count word is
   raised (a compare-and-swap loop that never lowers it; release) to its final value, P x the executions t <
   `executions` with t mod u its copy, then woken (a receive whose `count` is 0 has none to raise).  The bridge cancels
   a standing plan's receives not landed from an adjacent node it loses, and every one of them where this node's
   `map_serial` moves while the plan stands, its `serial` no longer the map's (mesh-flow.c plan_stale;
   design/disjoint-pages-change-request.md, D.3.6; design/disjoint-pages-rulings-coordination.md, R2), over every
   standing plan: a plan stands only while its map does, so every node of a plan whose map changed reads the
   cancellation within one flooding of the map.  The client's start raises `executions`, fences (seq_cst) and reads
   `map_serial`, and the bridge moves the serial, fences and reads `executions`, so either that cancellation covers the
   raised executions or mesh_transfers_start returns ESTALE after its raise and the program runs none of them (R3).  No
   phase is written for a cancellation, so a reader finds the receive one phase behind, and the GPU's later pass counts
   it in `status`.  No other word is a cancellation's.
   Both ends: for adjacent nodes a and b, copy c's entries at a naming b with MESH_SEND and those at b naming a with
   MESH_RECEIVE are the same hops in the same order, and u is the collective's, so both ends hold the same copies.  The
   order is each cable's sends in the schedule's order (design/plan-to-bridge-change-request.md R5: list scheduling by
   decreasing bottom level with edge scheduling under the contention model, O. Sinnen and L. A. Sousa, Communication
   contention in task scheduling, IEEE TPDS 16(6), 2005; design/compilation-coordination.md, T6): every rank computes
   the same schedule from the same inputs (F. B. Schneider, Implementing fault-tolerant services using the state
   machine approach, ACM Comput. Surv., 1990), and each cable carries execution t's sends in that order, then
   execution t + 1's.  R5 states that order is a topological order of the unrolled executions, so the cables'
   sequences cannot wait on each other in a cycle (M. Cowan et al., MSCCLang, ASPLOS 2023, section 5.2); for the
   bridge's store-and-forward waits it is not shown (below).  How a pair's sequence is spread over what its devices
   give it is the bridge's own (R5, R7).
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R5; design/compilation-coordination.md, T6, T7): the
   order of the entries is the schedule's, where it was the walk's under a hand priority, and each entry carries its
   `bytes` in place of `padding`, where every hop moved its whole page; the entries are in the plan's state section,
   where they were the layout's area at hdr `hop_off` (`hops` of them, at most `hop_capacity`).  What must stay: both
   ends' lists the same hops in the same order; every copy the same order.
   2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 5; design/compilation-coordination.md,
   "The appendix's revisions", S2, S3: T6 amended): each entry's group is its set's entries of the same `node` and
   `direction` where `set` is not 0, else the entry alone, and the entries are sorted by their group's least start in
   the schedule, then their group's least walk, then their own walk; so a set's entries are contiguous on their node
   pair's list within each copy, in walk order, both ends' alike, and every other entry keeps the schedule's order.
   What must stay: both ends' lists the same hops in the same order; every copy the same order.
   2026-11-04, owner L2 (review W3's finding 6, the coordinator's ruling): withdrawn, the claim of aa112f5b that S2's
   grouping keeps the per-pair order a topological order of the unrolled executions (a set's position i following its
   receive at position i upstream, whose group start is earlier).  The contention schedule starts a page on a route's
   links by Sinnen and Sousa's eq. (14), nonaligned (cut-through; mesh-graph.c: ts(e, Lk) = max{A, tf(e, Lk-1) -
   ς(e, Lk), ts(e, L1)}), so a page's start on a link can precede its finish on the link before, and the start order
   is not shown to be a topological order of the bridge's store-and-forward waits (a forward released by its receive's
   landing), for the T6 order itself as for the groups.  Open for the operator (design/compilation-coordination.md,
   the appendix's revisions, record).  What must stay: both ends' lists the same hops in the same order.
   Several plans (design/plan-to-bridge-change-request.md R8; design/compilation-coordination.md, Q5): each plan is in
   its own claim, and any number stand at once; a program holds nothing but its plans' claims (C3.4).  Writing a plan
   (mesh-collective.c mesh_plan; e6bd7db2's section 6):
   1. mesh_plan computes everything first: its demands, routes, the task graph and its schedule, the template with the
      program's data edges, the least lags and u, the items and both colourings, this node's entries of one copy and
      every room; EINVAL, EEXIST (a plan of the same (group, function) standing here on the same map; N1) or ENOMEM
      found here leave every standing plan standing.
   2. The plan's claim of exactly the rows and pages it uses, with its generation from hdr `serial`, taken whole or not
      at all (mesh-dataflow.h mesh_claim_take: the head's `start`, its owner word last; ENOMEM leaves every standing
      plan standing).  Nothing is claimed at attach (C3.4).
   3. The state section written to its starting values and every state value's bytes to 0, once (D3), its entries
      and its name, `serial`, `hops`, `entries`, `nodes` and `sender` written; every value's rows bound to its copies'
      pages; every claimed row's publication (struct mesh_publication) written.  No data page is written.
   4. The map read again: where it moved, or on the map reader's own error, this claim closed, its own alone
      (mesh_claim_close), ESTALE, `*plan` untouched.
   5. The head's publication `state` stored (release), `*plan` set to it, `plans` raised and woken: the plan stands.
   No earlier plan is closed: a program holds any number of compiled functions at once, and a function's claim is
   closed only by mesh_plan_retire, by mesh_detach or the bridge's retirement of a gone pid, or by step 4
   (design/aot-compilation-change-request.md, appendix item 6; design/compilation-coordination.md, "The appendix's
   revisions", N3).
   2026-11-04, owner L2 (item 6; N5): step 4 closes this claim alone, where it closed every earlier claim of the
   client too, and step 6 (every earlier claim of the client closed once the plan stood) is gone, with the context's
   `state`.  Why: compiled functions coexist, each launched by its calls until it is retired (item 6: JAX's
   compiled-function cache, CUDA Graphs instantiated once and launched many times, Pathways §3, §4.4).  What must
   stay: a plan's words written before it stands, its claim closed on its own failure after its writes.
   Nothing is erased and nothing waits: a plan's state and pages are new pages, never another plan's (C3.4).  The
   bridge reads none of it before its head's publication `state` (acquire) is stored.  Each port reads `plans` once a
   pass and, where it moved, walks the heads (mesh-dataflow.h mesh_head_after, mesh_head_state): it loads each
   standing plan it has neither loaded nor refused, announcing the claim's generation in its `prepared`, then a
   seq_cst fence, then the head read again and the load withdrawn where it changed (D3: the hazard pointer's
   announce-and-recheck, M. M. Michael, IEEE TPDS 15(6), 2004), ESTALE or EALREADY as before, then its `held` bit or
   its `refused` code, then `prepared` 0; and it ends each loaded plan whose claim closed, clearing its `held` bit as
   its last access.  The bridge's own turn (mesh_retired_release) reads a claim closed, fences (seq_cst), then
   releases it only where no port's `prepared` is its generation and its plan's `held` is 0, so a port that read a
   retired plan cannot begin serving it once its pages are released.  That is the announced epoch and its re-read of
   deferred reclamation (D10; C3.4: RCU's call_rcu, P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K.
   Fraser, Practical lock-freedom, 2004, epoch-based reclamation), and nobody waits for the release.  A plan whose
   `serial` is not `map_serial` when it is loaded is refused, and one that stands is cancelled where `map_serial` moves
   (Cancellation, above; D.3.6, R2): the start fails by error, ESTALE, before its raise or after it (R3), and its
   caller plans again on the map as it is.
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R8; design/compilation-coordination.md, Q5): reversing
   the 109 form, one plan a region published in hdr `hops`, `plan` and `state`, its generation hdr `plan` + 1, and
   withdrawn (`hops` 0, `state` 0) before every new plan and at retirement.  Why: "nothing happens when a program joins
   or leaves because a program is a sequence of like, function calls" (R8); several programs' plans stand at once,
   each in its claim, ordered by the island's sequencer (A7).  What must stay: a plan's words written before it stands
   and never rewritten while it stands; the announcement and its re-read before the release. */
struct mesh_hop { uint32_t page,node,direction,field; uint64_t word,count; uint32_t pages,bytes; uint64_t serial;
  uint32_t set,padding1; };
_Static_assert(sizeof(struct mesh_hop)==56 && offsetof(struct mesh_hop,word)==16 && offsetof(struct mesh_hop,count)==24 &&
  offsetof(struct mesh_hop,pages)==32 && offsetof(struct mesh_hop,bytes)==36 && offsetof(struct mesh_hop,serial)==40 &&
  offsetof(struct mesh_hop,set)==48 && offsetof(struct mesh_hop,padding1)==52,"mesh_hop");
/* the bytes one lane loads of the phase list: 64 receives' phases (struct mesh_state `phases`) */
#define MESH_LANE_BYTES 16u
/* the phase code, Gray 00 01 11 10 (struct mesh_hop's note, "Phases") */
static inline uint32_t mesh_phase(uint64_t k){ k&=3; return (uint32_t)(k^(k>>1)); }
/* the bytes of one phase group: a received range of `pages` pages in one copy, whole lanes */
static inline uint64_t mesh_group_bytes(uint32_t pages){ return ((uint64_t)pages+63)/64*MESH_LANE_BYTES; }
/* the GPU's alone (struct mesh_hop's note, "The state section"): `first` is 32 bits, the greatest ~t (t's low 32
   bits) over the failed passes, so the first failed execution it names is exact below 2^32 executions */
struct mesh_status { _Atomic uint32_t behind,ahead,first,padding; };
_Static_assert(sizeof(struct mesh_status)==16 && offsetof(struct mesh_status,ahead)==4 &&
  offsetof(struct mesh_status,first)==8,"mesh_status");
/* the start of a plan's state section, at its claim head's publication `state` (struct mesh_hop's note, "The state
   section"); `phases`, `counts`, `supplied`, `status` and `hops` region offsets; `nodes` a set of map indices.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): bytes 64-255 appended, the plan's name,
   entries, holders and sequence words, where the region held one plan's in hdr `hops`, `plan` and `state`.  What must
   stay: bytes 0-63 as at 109 */
struct mesh_state { _Alignas(64) uint32_t copies; uint32_t receives,ranges,releases;
  uint64_t phases,counts,supplied,status; _Atomic uint64_t executions; uint64_t padding;
  uint64_t group,epoch,serial,hops,entries; _Atomic uint64_t held; _Atomic int32_t refused[MESH_PORTS];
  uint64_t nodes; uint32_t sender,padding1; uint64_t submitted; _Atomic uint64_t released;
  _Atomic uint64_t served[MESH_PORTS]; };
_Static_assert(sizeof(struct mesh_state)==256 && _Alignof(struct mesh_state)==64 &&
  offsetof(struct mesh_state,receives)==4 && offsetof(struct mesh_state,ranges)==8 && offsetof(struct mesh_state,releases)==12 &&
  offsetof(struct mesh_state,phases)==16 && offsetof(struct mesh_state,counts)==24 &&
  offsetof(struct mesh_state,supplied)==32 && offsetof(struct mesh_state,status)==40 &&
  offsetof(struct mesh_state,executions)==48 && offsetof(struct mesh_state,padding)==56 &&
  offsetof(struct mesh_state,group)==64 && offsetof(struct mesh_state,epoch)==72 && offsetof(struct mesh_state,serial)==80 &&
  offsetof(struct mesh_state,hops)==88 && offsetof(struct mesh_state,entries)==96 && offsetof(struct mesh_state,held)==104 &&
  offsetof(struct mesh_state,refused)==112 && offsetof(struct mesh_state,nodes)==144 && offsetof(struct mesh_state,sender)==152 &&
  offsetof(struct mesh_state,padding1)==156 && offsetof(struct mesh_state,submitted)==160 &&
  offsetof(struct mesh_state,released)==168 && offsetof(struct mesh_state,served)==176,"mesh_state");

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
/* 2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, Appendix C, C3.2, C3.3 and C4;
   design/disjoint-pages-revision-coordination.md, "The region"): at 109 the plan's `executions` (C3.3: the run's
   length no input of the plan; the bridge reads struct mesh_state `executions`, D2) and the 108 areas' fields
   `word_off`, `word_capacity`, `landed_off`, `landed_capacity` and `landed_stride` (C4: the supplied words and landed
   list are the plan's state section, which mesh_plan places, C3.2) are left vacant the same way (`vacant14`, `vacant9`
   .. `vacant13`), and `state` appended after `map_room`.  What must stay: as above. */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,vacant0,links;
  uint32_t vacant1,wire_pages;
  /* the ring depth's bytes until 108 (the note above), never read or written */
  uint32_t vacant7;
  /* the plan's executions until 109 (struct mesh_hop's note), never read or written */
  uint32_t vacant14;
  uint64_t vacant2;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,vacant3,target_off,vacant4,vacant5,data_off,length;
  uint64_t vacant6,target_stride;
  /* `client`: never written from 110, kept by name (design/compilation-coordination.md, L3: mesh-heartbeat.c
     replace_turn, a lifecycle file, reads it with `client_start` as its one client holder, and so finds none; open
     for the operator); each claim's holder is its head's owner and start (struct mesh_buffer's note).  `serial`: the
     count that mints client words (mesh-dataflow.h mesh_attach, masked to 31 bits) and claim generations
     (atomic_fetch_add + 1, unmasked, never 0: design/compilation-coordination.md, C-gen); it only rises */
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
  /* the page lists' area and room until 110 (`hop_off`, `hop_capacity`; struct mesh_hop's note), never read or
     written; `vacant8` the word stride's bytes until 108 (the note above), never read or written */
  uint64_t vacant16,vacant17,vacant8;
  /* the one plan's entries and generation until 110 (`hops`, `plan`; struct mesh_hop's note, "Several plans"), never
     read or written */
  uint64_t vacant18,vacant19;
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
  /* the bridge's pages (struct mesh_bridge_pages) at `bridge_off`; `client_start` the one attached client's start
     until 110 (mesh-holders.h mesh_started), never written from 110 and kept by name, as `client` (L3) */
  uint64_t bridge_off;
  _Atomic uint64_t client_start;
  /* tag pre-return-20261103 rdma/mesh.h struct hdr `boot`, `health`, `busy`, `idle_since`, `stopping`, verbatim (their
     tag types; read by the tag's mesh-records.c and mesh-link.c region_open and main): `boot` the bridge's start on the
     wall clock (ns, CLOCK_REALTIME), its records' boot; `health` the serial a health-only change of a record moves,
     never the map's (mesh-records.c health_put); `busy`: the streams held (0 between calls) and since when none has
     been (`idle_since`, the wall clock in ns; 0 while any is); `stopping` stored 0 as the region opens */
  uint64_t boot;
  _Atomic uint64_t health,busy,idle_since,stopping;
  /* the supplied words' and landed list's area and room and the landed records' stride until 109 (struct mesh_hop's
     note), never read or written; the map's room, `map_room` records at `map_off` (mesh-map.h mesh_layout) */
  uint64_t vacant9,vacant10,vacant11,vacant12,vacant13,map_room;
  /* the one plan's state section's region offset until 110 (`state`; struct mesh_hop's note, "Several plans"), never
     read or written: each plan's is its claim head's publication `state` */
  uint64_t vacant21;
  /* 110 (design/compilation-coordination.md, "The region"): a count that only rises.  Every publication of a plan
     (mesh_plan), every close of a claim (mesh-dataflow.h mesh_claim_close, mesh_plan_retire, mesh_retire) and every
     start (mesh-call.h mesh_transfers_start) adds 1 (release), then wakes it (mesh_notify, OS_SYNC_WAKE_BY_ADDRESS_SHARED);
     the ports read it once a pass and walk the claims' heads where it moved.  Many writers, as a count has; never
     reset */
  _Atomic uint64_t plans;
};
/* every field at its 106 offset, the 108 fields after them, the 109 field after those, the 110 field after that (the
   vacant bytes' notes above struct hdr) */
_Static_assert(offsetof(struct hdr,links)==28 && offsetof(struct hdr,wire_pages)==36 && offsetof(struct hdr,planes_off)==56 &&
  offsetof(struct hdr,target_off)==104 && offsetof(struct hdr,data_off)==128 && offsetof(struct hdr,target_stride)==152 &&
  offsetof(struct hdr,client)==160 && offsetof(struct hdr,vacant16)==224 && offsetof(struct hdr,vacant17)==232 &&
  offsetof(struct hdr,map_serial)==288 &&
  offsetof(struct hdr,bridge_off)==296 && offsetof(struct hdr,client_start)==304 && offsetof(struct hdr,stopping)==344 &&
  offsetof(struct hdr,vacant7)==40 && offsetof(struct hdr,vacant14)==44 && offsetof(struct hdr,vacant8)==240 &&
  offsetof(struct hdr,vacant18)==248 && offsetof(struct hdr,vacant19)==256 && offsetof(struct hdr,map_off)==272 &&
  offsetof(struct hdr,vacant9)==352 && offsetof(struct hdr,vacant13)==384 && offsetof(struct hdr,map_room)==392 &&
  offsetof(struct hdr,vacant21)==400 && offsetof(struct hdr,plans)==408 && sizeof(struct hdr)==416,
  "struct hdr: no field moves");
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
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
/* a plan's entries (struct mesh_hop): its state section's `entries` of them at its `hops` (struct mesh_hop's note).
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): a plan's, from its state section, where
   mesh_hops(m) read the layout's one area at hdr `hop_off`.  What must stay: the entries read where the plan wrote
   them */
static inline struct mesh_hop *mesh_hops(struct hdr *m,const struct mesh_state *state){
  return (struct mesh_hop *)((char *)m+state->hops);
}
/* a plan's state section at region offset `at` (a claim head's publication `state`, mesh-dataflow.h mesh_head_state;
   mesh_plan's `*plan`, design/compilation-coordination.md, "The appendix's revisions", N2), NULL where `at` is 0.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): in place of mesh_plan_state, which read hdr
   `state`, the one plan's.  2026-11-04, owner L2 (N2, N5): the program's handle `*plan`, where it was the planning
   context's `state`, gone.  What must stay: a state section read at the offset its plan published */
static inline struct mesh_state *mesh_state_at(struct hdr *m,uint64_t at){
  return at?(struct mesh_state *)((char *)m+at):NULL;
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
   mesh or metal-microbench calls it.  What must stay: a range is read by its phase group in the plan's state section
   (struct mesh_hop's note, "The state section"; 2026-11-04, from 109, where it was its bit run in a landed record),
   never by its chunks */

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

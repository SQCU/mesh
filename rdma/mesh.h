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
   plan's function (item 6).  112: struct mesh_state `served` read as `taken` and `sequences` appended in the struct's
   tail (bytes 240-255; still 256 bytes): the plan's sequence words, one a copy, which the bridge parent writes and each
   port merges (design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node";
   design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1).  113: the deliveries' channel
   (struct mesh_delivery, MESH_DELIVERY_ROOM of them at hdr `deliveries_off`), hdr `deliveries_off`,
   `deliveries_room`, `delivered` and `delivery_read` appended, the area laid out after the map's records; struct
   mesh_state `released`, `taken` and `sequences` read and written no more (design/aot-compilation-change-request.md at
   mesh 0e56d8da, appendix item 2, "Delivery at a node": "publish them to each port through the channel above, in
   place of the sequence words (`sequences`), `taken[p]` and the parent's wait while they are full";
   design/compilation-coordination.md, "The revisions of 46ae8fdf", CH).  114: each plan's own queues in its state
   section, its delivered calls' (struct mesh_delivery, room `room` at `deliveries`, `delivered` of them) and its
   program's submissions' (struct mesh_submission, room `outstanding` at `submits`, `submissions` of them); struct
   mesh_state's words from byte 156 renamed in place (`outstanding`, `delivered`, `submissions`, `broadcast`,
   `arrived`, `room`, `deliveries`, `submits`, `announced`), `sender` the program's; the deliveries' channel's area gone
   from the layout, hdr `deliveries_off`, `deliveries_room` and `delivery_read` kept by name, hdr `delivered` this
   node's delivered count (Fig. 5's nextdeliver - 1); every received range a count word, a relay's among them; the
   wire's ARRIVE (mesh-bridge.h) (design/aot-compilation-change-request.md at mesh abecd389, A7 and appendix items 2
   and 5; design/compilation-coordination.md, "The revisions of abecd389").  115: struct mesh_record `view` appended
   (the bridge's counter, mesh-map.h); struct mesh_state 320 bytes, its words from byte 232 the plan's synchronization
   edge's (`waits`, `tells`, `wait_delay`, `wait_count`, `told`) and its submissions' queue's room (`submit_room`);
   the wire's VIEW and COUNT, and struct mesh_order's `view` where its `serial` was (mesh-bridge.h); hdr `map_serial`
   moved at each view's installation too (design/aot-compilation-change-request.md at mesh 3f106747, A7 and appendix
   item 2; design/compilation-coordination.md, "The revisions of 3f106747").  116: struct mesh_record `view_boot`,
   `view_sequencer` and `padding` appended, the process that created the record's bridge's installed view (mesh-map.h)
   (design/aot-compilation-change-request.md at mesh b5a3c6e6, A7, "The membership service is the sequencer";
   design/compilation-coordination.md, "The revisions of b5a3c6e6").  117: the program's serialized operand found at the
   layout's one place (hdr `program_off`, where `vacant21` was; struct mesh_program, laid out after the bridge's pages),
   no claim's head row (struct mesh_buffer's note); struct mesh_state's words from byte 232 the plan's synchronization
   edges (`waits` and `tells` their numbers, `syncs` their struct mesh_sync entries), `calls` and `closed`, where they
   were one edge's (`wait_delay`, `wait_count`, `told`) (design/aot-compilation-change-request.md at mesh f715c249, A9,
   "What the bridge runs: the self-timed schedule", and appendix item 2; design/compilation-coordination.md, "The
   documents' corrections of 2026-11-05", points 4 and 5, and "The revisions of 4e78f6f4").  118: the program's
   serialized operand's top level struct mesh_algo at hdr `algo_off` (`program_off`'s place), its words `published`,
   `retired` and `released`, each one writer's, its holder hdr `client` and `client_start`; struct mesh_sync 32 bytes,
   the waits in rows by call place (struct mesh_state `rows`, where `padding5` was), each delay folded into its `base`,
   a tell naming its sink's wait (`entry`); the wire's COUNT carrying that entry (mesh-bridge.h)
   (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4, B2 and K1; "The
   revisions of 55b9cbad").  119: struct mesh_hop `offset` where `padding1` was, an instruction's bytes at their
   offset in its page; struct mesh_state `transfers` where `padding3[0]` was, each declared transfer's range's words;
   struct mesh_algo `nodes` and `adjacent`, the plans with entries toward each map index; hdr `view_map`, `view`,
   `view_off` and `view_nodes`, the installed view's identifier and the sequencer's map records its first message
   carried, which plans are made on and stand by; the wire's VIEW carrying those records (mesh-bridge.h struct
   mesh_group_view) (design/compilation-coordination.md, "The root cause of the copies, and the revised work list",
   R-1 and R-6; design/aot-compilation-change-request.md A7, corrected 2026-11-06; "The revisions of 14b9a757").  120:
   hdr `view_sequencer` and `view_boot` appended and struct mesh_state `sequencer` and `boot` where `padding3[0]` and
   `padding3[1]` were, the installed view's creator and a plan's view's; struct mesh_view_id, the view's identifier,
   in the wire's ORDER, SUBMIT, ARRIVE and COUNT (mesh-bridge.h) (JGroups ViewId.java) */
#define MESH_VERSION 120u
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
   at once: 2026-11-04, owner L, where it said each program's one claim).
   2026-11-05 (117; design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): MESH_ROW_OWN and the arena's bits (hdr `arena_off`) are read and written no more, and kept
   in the layout by name, as every vacant field is.  Why: point 5, "place the whole program's values in one offline
   allocation at compile time.  These go: `mesh_claim_take`'s first fit and the claim's head record"; they were that
   first fit's bitmaps (Knuth vol. 1 §2.5).  The program's rows and pages are numbered by its one allocation
   (mesh-collective.h mesh_plan).  What must stay: the layout's areas where they are */
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
   32 bytes, no field moves (the assert below).
   2026-11-05 (117; design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
   revisions of 4e78f6f4"): no row is a claim's head, and the note above is the 116 form's.  A row's struct
   mesh_buffer holds `constant` alone (mesh-call.h mesh_section_constant); `owner`, `closed`, `pages` and `start` are
   read and written no more, kept by name.  The program's record is its serialized operand's root at the layout's one
   place (struct mesh_program, hdr `program_off`).  Why: point 5: "These go: `mesh_claim_take`'s first fit and the
   claim's head record; `mesh_plan_head`'s walk of every claim head"; the bridge reads the serialized operand "at a
   place its compilation fixed, and searches for nothing" (design/aot-compilation-change-request.md A9, "What the
   bridge runs").  What must stay: 32 bytes, no field moves */
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
     other field 0 (until 117: no row is a head now, struct mesh_buffer's note).
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
   announcement, a seq_cst fence, then the head read again (design/disjoint-pages-revision-coordination.md, D10).
   2026-11-05 (117; design/compilation-coordination.md, "The revisions of 4e78f6f4"): `prepared` the generation of the
   program's serialized operand the port is loading (struct mesh_program `generation`), where it was a claim's, and
   the root read again after the fence, where the head was.  What must stay: the announcement, a seq_cst fence, then
   the record read again.
   2026-11-05 (118; "The revisions of 55b9cbad", G4): the generation is struct mesh_algo `published`.  What must stay:
   as above */
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
     design/compilation-coordination.md, "The appendix's revisions", S1 and S2).  A set's entries keep their places in
     the plan's one global order, as every entry does, both ends' alike (design/plan-to-bridge-change-request.md R5:
     "A set's pages ... keep their places in that order.  The set's random permutation is drawn over those places";
     "Both ends", below): the plan's view, the bridge's cables its own.  The bridge permutes a set's members per cable,
     each execution, at both ends (S4, S5, S7′: on each cable of the pair, the members whose positions the deal gave
     that cable, among those positions, wherever they lie in the list).  The bridge sees a set or a sequence and no
     kind (design/plan-to-bridge-change-request.md R4).  `padding1` 0.
   2026-11-06 (119; design/compilation-coordination.md, "The root cause of the copies, and the revised work list",
   R-1: "The bridge's instructions are those blocks: aot A9, MSCCL-IR's steps with offset and count"; PB R2, corrected
   2026-11-06: "A page instruction names its bytes at their offset in its page, as MSCCL-IR's steps name a destination
   offset and a count"; "The revisions of 14b9a757"), reversing "A hop moves its `bytes` from the start of that page":
   `offset` where `padding1` was, the byte of block `page` at which the hop's `bytes` begin, so a hop moves bytes
   [page x pgsz + offset, + bytes) of the region's arena, `offset` + `bytes` <= block x pgsz (mesh_plan's: each
   instruction one extent within one block at each end, mesh-collective.h); every copy's entries of a hop the same
   `offset` where their blocks hold the same bytes, both ends' a hop's own (a send's in its source's block, a receive's
   in its destination's or its relay's).  `bytes` is the instruction's: the refinement's extent, at most block x pgsz,
   where it was min(Q, B - k Q) of page k.  A receive's `pages` is its range's instructions this node receives in one
   copy (its range's P), each of them one field of its range's group and one arrival on its count.  What must stay:
   offsets 0-51 where they are, 56 bytes
   2026-11-04, owner L2 (the appendix, item 5): `set` and `padding1` appended, where every hop was one of a sequence in
   the schedule's order.  Why: the pages of function B's operands are written as a set, not a sequence (R4, item 5).
   What must stay: offsets 0-47 as at 110; both ends' entries of a hop the same `set`.
   2026-11-04, owner L2 (review W3's finding 12, the coordinator's ruling; S7′): a set contiguous on its node pair's
   list and permuted per cable, where this note said contiguous on its cable and permuted whole.  Why: the deal spreads
   a pair's positions over the pair's cables, and each cable's records are its own (S7′).  What must stay: as above.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", OP): the bridge draws one permutation
   of a set's every position on the pair each execution, counted by (execution, set), and each cable carries the
   members the deal gave it in that permutation's order (mesh-flow.c set_draw, position_record), where each cable drew
   a permutation of its own members (S7′).  Why: design/aot-compilation-change-request.md, appendix item 5, "One
   permutation of the whole set per execution"; "A draw for each cable departs from the counter."  What must stay:
   both ends' entries of a hop the same `set`, at the same place.
   2026-11-04, owner C3 (GO4): a set's entries at the place of its first page in R5's global order, each by its own
   place, where this note said contiguous in walk order (S3, at the group's least scheduled start).  Why: R5 orders
   each cable's list by MSCCLang §5.2 step 3, and "Scheduled start times do not order the lists".  What must stay: a
   set's entries contiguous on its node pair and direction within each copy, both ends' alike.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): a set's entries at their own
   places, where this note said consecutive at the place of the set's first page (GO4), and the bridge's set its
   positions on the pair within a copy wherever they lie, where it was a contiguous run.  Why: R5, "(Corrected
   2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"  What must stay:
   both ends' entries of a hop the same `set`, at the same place.
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", PM): position j of a set's M positions
   on a pair carries member σ(j) of that execution's one permutation, on whichever cable the deal gave position j,
   each port holding a record of every member of each set it carries a position of (mesh-flow.c link_prepare,
   position_record), where each cable carried the members the deal gave it (M1).  Why:
   design/aot-compilation-change-request.md, appendix item 5, "So the deal fixes positions, not members."  What must
   stay: both ends' entries of a hop the same `set`, at the same place.
   Copies (C3.3: modulo variable expansion, M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo
   scheduling, MICRO 1994).  mesh_plan unrolls the plan's one execution, its template, into u copies: L of a routed
   value is the least lag at which every read of it in execution t happens before its write in t + L, and u the
   largest L over the collective's routed values at every node (C3.3; design/compilation-coordination.md, U1), so
   every rank computes the same u; a value with no such L fails the plan with ENOMEM (C3.3).  The list is struct
   mesh_state `entries` = u x n entries at its `hops`, copy c's n at [c n, c n + n).  Every copy's entries are the same
   hops in the same order, with the same `node`, `direction`, `pages`, `bytes`, `serial` and `set`; `page`, `word`,
   `field` and `count` are copy c's own.  Execution t runs copy t mod u.  A plan runs any number of executions: the
   run's length is no input of mesh_plan, and the bridge serves the executions t < struct mesh_state `executions` (D2)
   that the deliveries' channel has delivered (struct mesh_delivery; 2026-11-05, CH: where it said t < its `released`).
   The plan's one allocator places every copy's pages before the first execution, two
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
     2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", RC): `count` is never 0 from 114: a
     relay's own receive names its range's count word too (`counts`, below), which nobody but the bridge reads.  Why:
     design/aot-compilation-change-request.md, appendix item 5, "A page's round rule reads a count that only rises": a
     page is named for round j where its range's count of its copy reads at least j x `pages` (mesh-flow.c
     link_receive_drain), at a relay as at a destination.  What must stay: the XOR before the add.
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
   status; the delivered calls' queue, `room` struct mesh_delivery at `deliveries`; the submissions' queue,
   `submit_room` struct mesh_submission at `submits` (`outstanding` of them at 114); the entries; each 64-aligned
   (2026-11-05, "The revisions of abecd389", PQ: the two queues where the sequence words were before 113; "The
   revisions of 3f106747", BS: `submit_room`) (mesh-collective.c plan_layout; design/compilation-coordination.md, "The documents' revisions of
   edd4b583", "The region").  Each word has one writer:
   - `phases`: the phase list, a dense array of 2-bit fields.  For each range this node receives (one it consumes,
     or a relay's own, whose forwards its phases release), in the plan's demand order, and copy c, a group of
     mesh_group_bytes(P) bytes, groups in (range, c) order, range major; page k of the range (k < P) has its field at
     bits [2 (k % 32), 2 (k % 32) + 2) of uint64 word k / 32 of its group.  A group is whole GPU lane loads
     (MESH_LANE_BYTES, a compile-time 16: 64 fields), so a lane's load never spans two groups and a reader takes a
     range's copy with ceil(P / 64) loads.  A field's one writer is the bridge's data path that takes its receive.
   - `counts`: R x `copies` uint64 words, R every range this node receives, `ranges` (the struct's word) of them those
     it consumes, first, in the plan's demand order, then a relay's own, in that order; range g's copy c at index
     g x copies + c.  2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", RC): a relay's own
     received ranges have their counts too, where they had none (review 15's finding 1), so every receive's `count` is
     its range's count word.  Why: design/aot-compilation-change-request.md, appendix item 5: "The code must name a page
     for round j by that count, in place of PM2's test of the phase field"; "The plan keeps such a count already: each
     range's landed count of each copy, which every landing raises", which a relay's receive lacked.  What must stay: a
     consumed range's count where its publication names it; one writer, the landings, and the cancellation's raise.  A count takes
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
   - `released`: the executions of the plan whose sequence numbers this node's bridge parent has published, each in
     its copy's sequence word, in sequence order: L. Lamport's ms, "the total number of messages sent" (Concurrent
     reading and writing, CACM 20(11), 1977, "Message Buffer"; design/compilation-coordination.md, "The documents'
     revisions of edd4b583", DL1).  The bridge parent's alone; it only rises.  The parent delivers its pending ORDERs
     in arrival order, which is sequence order (Défago et al., Fig. 5), the head first, with no timeout and no skip,
     the head matched to a standing plan of its (group, epoch) on this node's map serial (DL2).  The plan's next
     execution is t = `released` (DL2′): where `first` <= t < `end`, it is numbered `sequence` + t - `first`; where
     `end` <= t, the ORDER leaves the head, its executions all published; where `first` > t, the head stays, said on
     the bridge's log once a head (Défago et al., Fig. 5: the message numbered `nextdeliver` has not arrived).  For t
     numbered: where t >= u and some port of the plan's port set that has not refused the plan has `taken[p]` < t - u
     + 1, Lamport's sender waits ("wait until ms < mr + P"), the head staying until a later turn of the parent's event
     loop, nobody blocking; else the number is stored in sequence word t mod u, then `released` t + 1 (release): "put
     message in B[ms mod P]; ms := ms + 1".  Before a plan's first number (its `released` 0) the parent adds 1 to hdr
     `plans` (release, then wakes it: DL7, hdr `plans`' note), and after each turn that published it raises hdr
     `plans` (mesh_notify: add 1, release, then wake; DL9).  The head's wait holds every later ORDER behind it, since
     the deliveries are in sequence order: it is Lamport's sender's wait for the buffer the appendix item names, not a
     hold of a plan's batch (DL2).  A port takes execution t of the plan only where t < `executions` (D2) and its
     number is published (t < `released`, acquire).
     2026-11-04, owner C3 (DL2 to DL4; design/aot-compilation-change-request.md, appendix item 2, "Delivery at a
     node"): reversing Q4's hold and O2's reason for it (an ORDER delivered only once every port holding the plan
     released before had served it, `released` then its `end`, so this node's cables carried one plan's batch at a
     time; what carried the delivery order to the ports was open).  Why: the parent publishes each released
     execution's sequence number in one word per copy, each word one writer (Lamport 1977), and each port merges the
     plans' sequences (D. E. Knuth, The Art of Computer Programming vol. 3, §5.4.1), so released executions of every
     plan interleave on each cable and nothing holds.  What must stay: a port takes t only below `executions` and
     `released`; `released` only rises.
     2026-11-05, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", the amendments
     on the reviews, DL2′ and DL9; reviews X1 #2, #7 and X3 #4): the next execution `released`, where it was the
     greater of the ORDER's `first` and `released`, and hdr `plans` raised after a turn that published, where it was
     woken without a raise.  Why: the greater of the two let `released` pass executions whose words were never
     written (an earlier ORDER of the plan that never reached this node), so a port would read them as published with
     a stale number, and Lamport's ms advances one message at a time; a wake with no raise that lands between a
     waiter's read of `plans` and its wait is lost until MESH_PAUSE_NS.  What must stay: delivery in sequence order,
     no timeout and no skip; `released` only rises.
   - `taken[p]`: Lamport's mr, port p's (DL1): the least execution of the plan whose number port p may still read,
     port p reading no number below it again.  For a plan port p carries nothing of, or has cancelled, it is
     `released` as that rises (taken_tell, each pass); for a plan port p ends, it is raised to `released` once, at the
     end, and the port then holds the plan no more (plan_end).  Port p's alone; it only rises (release), and the port
     raises hdr `plans` (mesh_notify: add 1, release, then wake; DL9) where it rose.  Port p serves the executions of
     every plan it carries in sequence-number order, a k-way merge (Knuth vol. 3, §5.4.1, as the appendix item states
     it: of each plan its next execution there, the least number among them next; each of the port's own readers, each
     direction, merging on its own, the bridge's: DL3): execution n of a plan a candidate where n < its `released`
     (acquire), its number its copy's word n mod u ("wait until mr < ms; read message in B[mr mod P]"), every
     candidate read, then every plan's `released` and hdr `plans` again (Lamport's "writing data elements in one order
     and reading them in the opposite order"), so the least number over the second reading is next.  Both ends of a
     cable hold the same plans and the same records and read the same numbers, so they post the same order (the item:
     "they merge alike"; DL3).
     2026-11-04, owner C3 (DL1, DL4): `taken[p]` where `served[p]` was (the executions port p had served whole, every
     record of them done, which Q4's hold read).  Why: the hold is gone; a port's word is now the receiver's count of
     the buffer it no longer reads.  What must stay: port p its one writer; the word only rises; `released` for a plan
     the port carries nothing of.
     2026-11-05, owner C3 (reviews X1 #4 and #7; DL9): an ended plan's `taken[p]` raised to `released` once, at its
     end, where this note said it follows `released` as it rises (which a plan the port has freed cannot); and hdr
     `plans` raised where `taken[p]` rose, where it was woken without a raise (Q4′'s form, overturned with Q4).  What
     must stay: port p its one writer; the word only rises.
   - The sequence words (`sequences`, copy c's at `sequences` + 8 c): Lamport's B[c], P = u (DL1; R2, C3.3: "execution t
     runs copy t mod u"): word c the sequence number of the execution t, t mod u = c, the parent published last in
     copy c, struct mesh_order `sequence` + t - `first` of the ORDER that holds t.  The bridge parent's alone.  Where
     this differs from the paper (DL1): every port is a receiver of every message (each port serves every standing
     plan), so the sender's wait is over every receiver's mr; and a port's mr is the least of its own readers' next
     reads (the bridge's), each reading the messages in order on its own.
   2026-11-05 (114; design/aot-compilation-change-request.md at mesh abecd389, A7 and appendix item 2;
   design/compilation-coordination.md, "The revisions of abecd389", SQ2, PS, PB and PQ): the words from byte 156, each
   in the place of a word gone (the 112 and 113 forms' notes above on `released`, `taken` and the sequence words are
   history), the struct still 256 bytes:
   - `sender`: the program's one sender, "the least of the program's nodes, the program being its launcher's `group`"
     (A7, "What each term names here", sender(m)): the least of mesh_plan's `program`, the program's nodes, so every
     plan of one program names one sender, where it was the least of this plan's `nodes` (Q2; A7's "Corrections").
     mesh_plan's alone.  `nodes` stays the plan's own, Dest(m) (A7).
   - `outstanding` (where `padding1` was): A10's bound on the program's calls of this function outstanding at once,
     mesh_plan's operand, the room of the submissions' queue.  mesh_plan's alone.
   - `submitted`: the executions of the plan the sender has TO-broadcast, [submitted, end) its next message; the
     bridge parent's alone, at the sender.
   - `delivered` (where `released` was): the messages Fig. 5 has delivered into this plan's queue at this node (A7,
     deliver(m): "appends m to its plan's queue there"); the bridge parent's alone, only rising (release after the
     entry's stores).  The queue: `room` struct mesh_delivery at `deliveries` (region offsets), the k-th delivery in
     entry k mod `room`, each one message with its sequence number (appendix item 2: "Each entry is one delivered
     message, with its sequence number"; "The plan's k-th delivery at the node is entry k mod the queue's room"), laid
     out in the plan's own state section, claimed and released with its pages (C3.2, C3.4), read in place by every port
     (mesh-flow.c lane_take) and copied by nothing.  `room` is fixed at compile time: `outstanding` plus u times the
     hops of the plan's routes (appendix item 2: "That is at most A10's bound on the plan's submissions outstanding at
     its sender, plus C3.3's u copies on each hop of the plan's routes"; mesh-collective.c mesh_plan).  Nothing waits to
     write and nothing checks for a full queue (item 2).
   - `submissions` (where `taken[0]` was): the program's submissions of the plan, each one start that raised
     `executions` (mesh-call.h mesh_transfers_start), in its queue `outstanding` struct mesh_submission at `submits`,
     the k-th in entry k mod `outstanding`: the raise's `executions` and the program's Lamport clock at it (L. Lamport,
     Time, clocks, and the ordering of events in a distributed system, CACM 21(7), 1978, IR1: "Each process Pi
     increments Ci between any two successive events"; mesh-dataflow.h struct mesh_ctx `clock`).  The program its one
     writer (release after the entry's stores); read by the bridge parent at `sender` alone.  `broadcast` (where
     `taken[1]` was): the submissions that bridge has TO-broadcast, its alone.
   - `arrived` (where `taken[2]` was): at the sender, the plan's nodes counted as holding the plan (A7, "A plan's
     destinations exist before its first message": "The program's sender counts the plan's nodes as each holds the
     plan, as a centralized barrier counts arrivals"; J. M. Mellor-Crummey and M. L. Scott, ACM TOCS 9(1), 1991, §3.1),
     a set of map indices; the bridge parent's alone at the sender; no submission of the plan is TO-broadcast before it
     holds `nodes`.  `announced` (where `taken[6]` was): 1 once this node's bridge parent has told the sender that this
     node holds the plan (mesh-records.c orders_announce), its alone.
   2026-11-05 (115; design/aot-compilation-change-request.md at mesh 3f106747, appendix item 2, "Its room is fixed at
   compile time, by the self-timed buffer bound"; design/compilation-coordination.md, "The revisions of 3f106747", BS),
   reversing 114's rooms above (`room` `outstanding` plus u times the hops of the plan's routes, and the submissions'
   queue's room `outstanding`): `room` and `submit_room` (the struct's note) are the queues' self-timed buffer bounds in
   the plan's part of the program's IPC graph, and the submissions' queue is `submit_room` struct mesh_submission at
   `submits`, the k-th in entry k mod `submit_room`; `outstanding` is A10's bound alone.  Why: "the text before
   bounded the room by "A10's bound on the plan's submissions outstanding at its sender, plus C3.3's u copies on each
   hop ..." ... It holds only where the queue lies on a cycle through the sender" (item 2's correction).  What must
   stay: nothing waits to write and nothing checks for a full queue.
   Every word is set to its starting value once, where mesh_plan places the section (D3), as mbarrier.init runs once
   a kernel: the phases, counts, supplied words, status, `executions`, `held`, `refused`, `submitted`, `released`,
   every `taken` and every sequence word 0.  None is reset, and none written to an earlier value, while the plan
   stands, but a port's bit of `held`, which it clears as its last access; a sequence word holds the number its
   copy's last publication stored (Lamport's B[ms mod P]).  `submitted`, `released` and every `taken` only rise.  What
   A7 adds to C3.3's run time is the sequencer's `seqnum`, each bridge parent's pending ORDERs (the head's `first` and
   `sequence` advanced as its executions are published), and each plan's `submitted`, `released`, `taken` and
   sequence words (Q6, as design/compilation-coordination.md, "The documents' revisions of edd4b583", DL6 leaves it).
   2026-11-04, owner C3 (DL1, DL6): the sequence words and `taken` in this list and in Q6's, where `served` was, and
   each parent's pending ORDERs with their heads advanced as they are published, where it named the ORDERs not yet
   delivered.  Why: delivery publishes each execution's number (appendix item 2).  What must stay: every word set
   once where placed, none reset while the plan stands.
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
   order is R5's one global topological order of the plan's instructions (design/plan-to-bridge-change-request.md R5,
   "The order of each cable's list, by the reference algorithm"): its tasks and its communications, a communication a
   page over a tree edge, whose send at one end and receive at the other are one instruction at one place, sorted as
   MSCCLang's step 3 states it, "Sort instructions into a global topological order with respect to their dependencies
   with a heap using the priority to order instructions" (M. Cowan et al., MSCCLang: Microsoft Collective
   Communication Language, ASPLOS 2023, section 5.2), the priority each instruction's completion time C̄_j in one
   linear program's optimal solution, the least first (R5 at mesh 0e56d8da: C.-Y. Chen, arXiv 2205.02474, 2022, LP
   (14); L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, "Schedule-by-C̄j"), a tie
   to the lower instruction (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO1 to GO3;
   "The revisions of 46ae8fdf", LP; mesh-collective.c plan_schedule; 2026-11-05: the priority was the bottom level,
   Sinnen and Sousa's eq. (7)).  Each node pair's list at each end is a subsequence of it, every entry at its own place, a set's
   among them (R5: "A set's pages ... keep their places in that order"), so both ends' lists agree; every rank
   computes the same order from the same inputs (F. B. Schneider, Implementing fault-tolerant services using the state
   machine approach, ACM Comput. Surv., 1990); and each cable carries execution t's sends in that order, then
   execution t + 1's (R5), the cables' sequences subsequences of it.  Its dependencies are the bridge's
   store-and-forward waits (a forward released by its receive's landing, a source's send by its supplied word: GO1),
   the order is a topological order of them, and every lag points forward, so in that order the cables' sequences
   cannot wait on each other in a cycle (R5; MSCCLang, section 5.2: "all implicit dependencies introduced by thread
   block sequential execution cannot produce cycles"; mesh-collective.c plan_schedule).  The bridge draws one
   permutation of a set's every position on a pair each execution, and each cable carries the members the deal gave it
   in its order (design/aot-compilation-change-request.md, appendix item 5).  The item's argument for the sequences it
   makes ("each cable's sequence is the same renaming of them.  So MSCCLang §5.2's argument against cycles holds")
   holds where one cable carries every position of the set on each pair of its tree, and is not shown where the deal
   splits a set's positions over a pair's cables (design/compilation-coordination.md, "The revisions of 46ae8fdf",
   stop 3; "The revisions of 67ca86f8", stop 3, before it).  2026-11-05 ("The revisions of f3f7cf46", PM): each
   position now carries σ's member on whichever cable the deal gave it, so σ is one renaming over the set's positions
   on every lane it crosses, the item's revised argument ("Why σ keeps a topological order", "So the deal fixes
   positions, not members").  Edge scheduling's start times are no longer
   computed for the order (R5 at 0e56d8da; "The revisions of 46ae8fdf", LP; 2026-11-05: they were computed and read by
   nothing, GO5).  How a pair's sequence is spread over what its devices give it is the bridge's own (R5, R7).
   2026-11-05 (design/compilation-coordination.md, "The revisions of 67ca86f8", SP): every entry at its own place,
   where a set's entries were at its first page's place (GO4), and the topological order R5's own, where it was GO4's
   groups in order of their least places with each set whole.  Why: R5, "(Corrected 2026-11-05: the first text moved
   a set's pages to consecutive places, which nothing requires.)"  What must stay: both ends' lists the same hops in the
   same order; every copy the same order.
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R5; design/compilation-coordination.md, T6, T7): the
   order of the entries is the schedule's, where it was the walk's under a hand priority, and each entry carries its
   `bytes` in place of `padding`, where every hop moved its whole page; the entries are in the plan's state section,
   where they were the layout's area at hdr `hop_off` (`hops` of them, at most `hop_capacity`).  What must stay: both
   ends' lists the same hops in the same order; every copy the same order.
   2026-11-04, owner C3 (GO1 to GO5; R5, "The order of each cable's list, by the reference algorithm"): reversing the
   schedule's order (T6: each list in its link's scheduled starts) and S3's groups at their least scheduled start, then
   their least walk, then each entry's walk (owner L2's notes of aa112f5b).  Why: R5 orders each cable's list as a
   subsequence of MSCCLang section 5.2 step 3's global topological order by bottom level, a set's pages at the place
   of its first page, and "Scheduled start times do not order the lists"; this overturns T6 and S3.  The claim owner L2
   withdrew (review W3's finding 6: that the start order is a topological order of the bridge's store-and-forward
   waits, unshown since Sinnen and Sousa's eq. (14) starts are cut-through) is now GO4's, over the instruction DAG,
   whose dependencies are those waits.  What must stay: both ends' lists the same hops in the same order; every copy
   the same order; a set's entries contiguous on their node pair and direction within each copy.
   Several plans (design/plan-to-bridge-change-request.md R8; design/compilation-coordination.md, Q5): each plan is in
   its own claim, and any number stand at once; a program holds nothing but its plans' claims (C3.4).  Writing a plan
   (mesh-collective.c mesh_plan; e6bd7db2's section 6):
   1. mesh_plan computes everything first: its demands, routes, the task graph, its schedule and the order of its
      lists (GO1 to GO5), the template with the program's data edges, the least lags and u, the items and both
      colourings, this node's entries of one copy and every room; EINVAL, EEXIST (a plan of the same (group,
      function) standing here on the same map; N1) or ENOMEM found here leave every standing plan standing.
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
   announce-and-recheck, M. M. Michael, IEEE TPDS 15(6), 2004), ESTALE as before (EALREADY no longer: "The revisions
   of f3f7cf46", SW), then its `held` bit or
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
  uint32_t set,offset; };
_Static_assert(sizeof(struct mesh_hop)==56 && offsetof(struct mesh_hop,word)==16 && offsetof(struct mesh_hop,count)==24 &&
  offsetof(struct mesh_hop,pages)==32 && offsetof(struct mesh_hop,bytes)==36 && offsetof(struct mesh_hop,serial)==40 &&
  offsetof(struct mesh_hop,set)==48 && offsetof(struct mesh_hop,offset)==52,"mesh_hop");
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
/* JGroups ViewId (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/ViewId.java): `id` "Lamport
   time of the view"; `sequencer` and `boot` the "Address of the creator of this view", a sequencer's node id and its
   bridge's start (hdr `boot`), as JGroups' UUID (src/org/jgroups/util/UUID.java): "Logical address which is unique
   over space and time" */
struct mesh_view_id { uint64_t id; uint32_t sequencer,padding; uint64_t boot; };
_Static_assert(sizeof(struct mesh_view_id)==24 && offsetof(struct mesh_view_id,sequencer)==8 &&
  offsetof(struct mesh_view_id,boot)==16,"mesh_view_id");
/* ViewId.compareTo: "The comparison is done on the IDs, if they are equal, we use the creator", `id > other.id ? 1 :
   id < other.id ? -1 : creator.compareTo(other.creator)`; the creator's as UUID.compareTo, its first word, then its
   second.  ViewId.equals: compareTo == 0 */
static inline int mesh_view_compare(struct mesh_view_id a,struct mesh_view_id b){
  return a.id>b.id?1:a.id<b.id?-1:a.sequencer>b.sequencer?1:a.sequencer<b.sequencer?-1:
    a.boot>b.boot?1:a.boot<b.boot?-1:0;
}
/* the start of a plan's state section, at its claim head's publication `state` (struct mesh_hop's note, "The state
   section"); `phases`, `counts`, `supplied`, `status` and `hops` region offsets; `nodes` a set of map indices.
   2026-11-04, owner L (design/compilation-coordination.md, "The region"): bytes 64-255 appended, the plan's name,
   entries, holders and sequence words, where the region held one plan's in hdr `hops`, `plan` and `state`.  What must
   stay: bytes 0-63 as at 109.
   2026-11-04, the coordinator (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1;
   design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"): `served` (the executions port p
   had served whole, which Q4's hold read) is `taken`, port p's count of the plan's executions whose sequence numbers it
   reads no more (L. Lamport, Concurrent reading and writing, CACM 20(11), 1977, "Message Buffer": the receiver's mr),
   and `sequences`, the region offset of the plan's sequence words (Lamport's B[0 .. P - 1], P = u), is appended in the
   struct's tail.  What must stay: bytes 0-239 where they were, the struct 256 bytes.
   2026-11-05 (113; design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): `released`, `taken` and
   `sequences` are read and written no more, but for their starting value 0 where mesh_plan places the section, and
   kept by name, every field in its place: the delivered calls reach the ports through the deliveries' channel (hdr
   `deliveries_off`, struct mesh_delivery), where each execution's number was published in its plan's sequence word of
   its copy, `released` raised past it, and the parent waited on every port's `taken[p]` while the words were full
   (Lamport's Message Buffer).  Why: design/aot-compilation-change-request.md at mesh 0e56d8da, appendix item 2, "The
   delivered calls reach the ports, in delivery order, through a first-in, first-out channel whose writer never
   waits"; "in place of the sequence words (`sequences`), `taken[p]` and the parent's wait while they are full".  The
   notes above on those three words are the 112 form's.  What must stay: bytes 0-255 where they are.
   2026-11-05 (114; design/compilation-coordination.md, "The revisions of abecd389", PQ): `padding1`, `released`,
   `taken` and `sequences` gone, their bytes the plan's queues' and the barrier's words (the note "2026-11-05 (114"
   above).  Why: design/aot-compilation-change-request.md, appendix item 2: "This replaces the sequence words
   (`sequences`), `taken[p]` and the parent's wait while they are full ..., and then the one channel for every plan".
   What must stay: bytes 0-155 where they are, the struct 256 bytes.
   2026-11-05 (115; design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by
   the self-timed buffer bound"; design/compilation-coordination.md, "The revisions of 3f106747", BS), the struct 320
   bytes, `padding3`, `padding4` and `padding2` gone, their bytes and those appended after them:
   - `outstanding` is A10's bound as mesh_plan took it, the delay on the edge from a call's completion to the
     program's later submission in the program's IPC graph ("At each node, A10's bound is the delay on the edge from a
     call's completion to the program's later submission"), and no queue's room;
   - `room` the delivered calls' queue's room here and `submit_room` the submissions' queue's, each its edge's
     self-timed buffer bound (S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997,
     Theorem 2: "Bfb(e) = ρ_Gs(snk(e), src(e)) + delay(e)"), mesh_plan's;
   - the plan's synchronization edge into this node, added by BSL's Convert-to-SC-graph (§9, Fig. 6) with its delay
     from DetermineDelays (Fig. 8): `waits` its source's map index (MESH_ABSENT none), `wait_delay` its delay, and
     `wait_count` the count its source raises as the bridge carries it here (the bridge parent its one writer, only
     rising; UINT64_MAX once the plan is cancelled); the program's k-th start of the plan reads it first (mesh-call.h
     mesh_transfers_start: BSL (7), "start(vi, k) ≥ end(vj, k − delay((vj, vi)))");
   - the plan's synchronization edge out of this node: `tells` its sink's map index (MESH_ABSENT none), and `told` the
     count the bridge parent here last carried toward it (mesh-records.c counts_tell).
   Why: "An added edge (vj, vi) holds "start(vi, k) ≥ end(vj, k − delay((vj, vi)))" (BSL (7)).  Its source raises a
   count that only rises (C3's counted completion: Portals 4 counting events), and its sink reads that count before
   its k-th start.  The bridge carries the count from source to sink, as it carries an arrival."  What must stay:
   bytes 0-231 where they are.
   2026-11-05 (117; design/aot-compilation-change-request.md, appendix item 2, "Each plan builds its own part of that
   graph" withdrawn: "the compilation builds the program's whole IPC graph ... and each queue's room is (5) in that
   graph"; design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 4, and "The revisions
   of 4e78f6f4"), the plan's synchronization edges at this node, where it had one into it and one out of it:
   - the program's IPC graph has an actor for each node's part of each of the program's calls ("The actors are each
     node's parts of the program's calls, in program order"), so a plan whose function the program calls `calls`
     times in one execution of its traced block (A10, "The calls' graph") has `calls` actors at a node, and
     Convert-to-SC-graph adds at most one edge into each actor and one out of it (BSL §9, Fig. 6: each SCC's chosen
     vertex);
   - `waits` the edges into this plan's actors here and `tells` those out of them, `syncs` the region offset of their
     struct mesh_sync entries, the `waits` first, in the state section (mesh-collective.c plan_layout);
   - `calls` the program's calls of the plan's function in one execution of the block (A10), at least 1: the plan's
     k-th start here (its `submissions` before it) is its function's call at place k mod `calls` among them, in the
     block's k / `calls`-th execution;
   - `closed` 1 (release) once the program retires the plan alone (mesh-dataflow.h mesh_plan_retire), 0 else.
   `wait_delay`, `wait_count` and `told` go, their bytes these.  What must stay: bytes 0-231 where they are.
   2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1: "the
   start's scan of every wait entry ...: the waits are stored in rows keyed by call place, and each edge's delay is
   folded into its threshold", as MSCCL-IR's steps name their dependences by index (`depid`, `deps`); "The revisions of
   55b9cbad"): `rows`, where `padding5` was, the region offset of `calls` + 1 uint32_t words, word a the index among
   the waits at `syncs` of the first wait into the call at place a, so the waits into the call at place a are entries
   rows[a] .. rows[a + 1] - 1 (the waits sorted by call place: mesh-graph.h mesh_graph_rows' compressed rows, keyed by
   place).  Why: K1; design/aot-compilation-change-request.md A9, "What the bridge runs: the self-timed schedule":
   "Every address in it is fixed at compile time ... and searches for nothing".  What must stay: every other field
   where it is, the struct 320 bytes
   2026-11-06 (119; design/compilation-coordination.md, "The root cause of the copies, and the revised work list",
   R-1; "The revisions of 14b9a757"):
   - `transfers`, where `padding3[0]` was: the region offset of one struct mesh_publication (64 bytes) for each of the
     function's declared transfers, in their order (mesh_transfer_at), each written once by mesh_plan, read by index:
     where this node is the transfer's source, `sends` 1, `targets[0]` {word: the region offset of its range's supplied
     word, count 1, stride 0}; where it is the transfer's destination, `pages` the range's instructions it receives a
     copy (P), `phases` the region offset of its copy-0 phase group (copy c's at phases + c x mesh_group_bytes(P)),
     `count` the region offset of its copy-0 count word (copy c's at count + 8 c); `state` this section's region offset
     where either holds; every other field 0.  Why: ranges may share a block where their bytes are disjoint (PB R2,
     corrected 2026-11-06), so several begin in one row and a row's publication names none of them alone; the
     declared transfer's index is the IR's own name of it.  Every row's publication (mesh_publication_at) is written by
     nothing now and read by nothing.
   - `serial` is the Lamport time of the installed view whose map the plan was made on (hdr `view`), every entry's
     `serial` the same, where it was the map serial; the plan stands while that view is installed (A7, corrected
     2026-11-06: "A plan then stands while its view does, and its view while the sequencer's map does").
   - `sequencer` and `boot`, where `padding3[0]` and `padding3[1]` were: that view's creator (hdr `view_sequencer`,
     `view_boot`), so the plan's view is struct mesh_view_id (`serial`, `sequencer`, `boot`), mesh_plan_view.
   What must stay: every other field where it is, the struct 320 bytes.
   `removed`, where `padding4` was, the bridge parent its one writer (design/problems-of-14b9a757-change-request.md,
   "Sources for the stops", 11; Documentation/RCU/whatisRCU.rst: "The removal phase removes references to data items
   within a data structure") */
struct mesh_state { _Alignas(64) uint32_t copies; uint32_t receives,ranges,releases;
  uint64_t phases,counts,supplied,status; _Atomic uint64_t executions; uint64_t padding;
  uint64_t group,epoch,serial,hops,entries; _Atomic uint64_t held; _Atomic int32_t refused[MESH_PORTS];
  uint64_t nodes; uint32_t sender,outstanding; uint64_t submitted; _Atomic uint64_t delivered;
  _Atomic uint64_t submissions; uint64_t broadcast,arrived,room,deliveries,submits,announced;
  uint32_t waits,tells; uint64_t syncs; uint32_t calls; _Atomic uint32_t closed; uint64_t rows,submit_room,transfers;
  uint32_t sequencer; _Atomic uint32_t removed; uint64_t boot,padding3[3]; };
_Static_assert(sizeof(struct mesh_state)==320 && _Alignof(struct mesh_state)==64 &&
  offsetof(struct mesh_state,receives)==4 && offsetof(struct mesh_state,ranges)==8 && offsetof(struct mesh_state,releases)==12 &&
  offsetof(struct mesh_state,phases)==16 && offsetof(struct mesh_state,counts)==24 &&
  offsetof(struct mesh_state,supplied)==32 && offsetof(struct mesh_state,status)==40 &&
  offsetof(struct mesh_state,executions)==48 && offsetof(struct mesh_state,padding)==56 &&
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
/* the view a plan was made in (struct mesh_view_id) */
static inline struct mesh_view_id mesh_plan_view(const struct mesh_state *s){
  return (struct mesh_view_id){.id=s->serial,.sequencer=s->sequencer,.boot=s->boot};
}
/* One synchronization edge of the program's IPC graph at a plan's actor at this node (117; design/aot-compilation-
   change-request.md, appendix item 2: "The compilation adds synchronization edges by BSL's Convert-to-SC-graph (§9,
   Fig. 6) and sets their delays by DetermineDelays (Fig. 8)"; "An added edge (vj, vi) holds "start(vi, k) ≥ end(vj, k −
   delay((vj, vi)))" (BSL (7)).  Its source raises a count that only rises (C3's counted completion: Portals 4 counting
   events), and its sink reads that count before its k-th start.  The bridge carries the count from source to sink, as
   it carries an arrival"; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, here
   BSL).  The count is the source plan's starts at the source's node (its `submissions`), "the program's starts of the
   plan at that node" (item 2), which only rise.  Every delay counts executions of the program's traced block, BSL's
   iterations (one firing of every actor).
   - An edge into this plan here (struct mesh_state `waits` of them, first at `syncs`): `node` its source's map index,
     `function` the source plan's (its struct mesh_state `epoch`), `every` the source's `calls`, `position` the place
     among them of the call whose start the count must reach, `at` the place among this plan's `calls` of the call it
     enters, `delay` DetermineDelays' delay; `count` the source's starts the bridge parent here has carried (its one
     writer, only rising; UINT64_MAX once the plan is cancelled).  This plan's k-th start here, where k mod `calls` is
     `at` and K = k / `calls` >= `delay`, waits until `count` >= (K - `delay`) x `every` + `position` + 1 (mesh-call.h
     mesh_transfers_start).
   - An edge out of this plan here (`tells` of them, after the waits): `node` its sink's map index, `function` the sink
     plan's, `count` the starts here the bridge parent last carried toward it (mesh-records.c counts_tell); `every`,
     `position`, `at` and `delay` 0.
   What must stay: 40 bytes; written once where mesh_plan places the plan, `count` the bridge's after.
   2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", K1: "Each
   is to be read by index, as MSCCL-IR's steps name their dependences by index (`depid`, `deps`): the start's scan of
   every wait entry: the waits are stored in rows keyed by call place, and each edge's delay is folded into its
   threshold; `count_take`'s search: the COUNT carries the sink entry's index"; "The revisions of 55b9cbad"), 32 bytes,
   `position`, `at` and `delay` gone:
   - A wait (into this plan here, `waits` of them at `syncs`, sorted by the place among this plan's `calls` of the call
     it enters, struct mesh_state `rows` their compressed rows): `node` its source's map index and `function` the
     source plan's epoch, as before; `every` the source's `calls`; `base` = `position` + 1 - `delay` x `every`, the
     edge's position and delay as above folded into one constant, so this plan's k-th start here, K = k / `calls`,
     waits on each of its row k mod `calls` until `count` >= K x `every` + `base`, a bound of 0 or below holding at
     once (BSL (7), "start(vi, k) ≥ end(vj, k − delay((vj, vi)))": for K below `delay` the source's start it needs is
     of an execution before the first); `entry` 0.  Its source is a node's submission of a call (σ, design/aot-
     compilation-change-request.md appendix item 2 as corrected for IG-5: "the submissions σ_1 … σ_n of the block's
     calls are one processor's actors ... each of time 0"), whose end is that start, or a node's part of a call (x),
     whose end the start of its A10 successor follows (IG-3); mesh-collective.c plan_graph gives each its position and
     delay.
   - A tell (out of this plan here, `tells` of them after the waits): `node` its sink's map index, `function` the sink
     plan's epoch, `entry` the index of its edge among the sink plan's waits at the sink (its place at the sink's
     `syncs`), which the COUNT carries (mesh-bridge.h struct mesh_count), `every` and `base` 0; `count` the starts here
     the bridge parent last carried toward it.
   Why: K1.  What must stay: written once where mesh_plan places the plan, `count` the bridge's after */
struct mesh_sync { uint32_t node,function,entry,every; int64_t base; _Atomic uint64_t count; };
_Static_assert(sizeof(struct mesh_sync)==32 && offsetof(struct mesh_sync,entry)==8 && offsetof(struct mesh_sync,every)==12 &&
  offsetof(struct mesh_sync,base)==16 && offsetof(struct mesh_sync,count)==24,"mesh_sync");
/* The root of the program's serialized operand, at the layout's one place, hdr `program_off` (117; mesh-map.h
   mesh_layout).  design/aot-compilation-change-request.md A9, "What the bridge runs: the self-timed schedule": "Every
   address in it is fixed at compile time, as memory planning fixes them ... So the bridge reads the serialized operand
   at a place its compilation fixed, and searches for nothing"; design/compilation-coordination.md, "The documents'
   corrections of 2026-11-05", point 5: "R1's method is U-Net's and VIA's queues of descriptors over registered memory,
   which the interface knows from the endpoint's creation; it does not search for them.  So the serialized operand's
   place is a constant the region's layout names once, as the bridge's own pages are "laid out once with the region"
   (disjoint-pages A3).  Nothing walks heads to find it."  The run is one program on the whole collective (A7,
   corrected 2026-11-05), so the region holds one program's operand at a time.
   - `owner`: the client word of the program whose operand it is (mesh-dataflow.h mesh_attach), 0 none.  mesh_plan
     stores it last (release), once every plan of the program is written, and the bridge parent stores 0 (release)
     where it releases a closed program no port holds (mesh-dataflow.h mesh_retired_release, deferred reclamation).
     mesh_plan is EBUSY where it reads it not 0 (mesh-collective.h);
   - `closed`: 1 (release) once the program retires (mesh-dataflow.h mesh_retire: its detach, or the bridge's
     retirement of a gone process), 0 where mesh_plan writes the root;
   - `plans`: the program's plans, one a compiled function, in the plan call's order (mesh-collective.h struct
     mesh_function); `table` the region offset of `plans` uint64_t words, word k the region offset of function k + 1's
     state section (struct mesh_state, its `epoch` k + 1), placed by the program's one allocation;
   - `start`: mesh_started(getpid()) of the planning process (mesh-holders.h), with `owner`'s low 32 bits its identity
     as a holder of the region; `group` the program's (MESH_GROUP); `generation` minted from hdr `serial`
     (atomic_fetch_add + 1, never 0), the ports' announcement while they load a plan of it (struct mesh_port_info
     `prepared`).
   A plan of the program stands where `owner` is not 0, `closed` is 0 and its state section's `closed` is 0
   (mesh_program_plan).  What must stay: 64 bytes, every field in its place.
   2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4 and
   B2, and "Answered in the documents", EB-1; "The revisions of 55b9cbad"), reversing the note above: struct
   mesh_algo, where it was struct mesh_program, "the root".
   - Its name is A9's: MSCCL-IR's top level (M. Cowan et al., MSCCLang, ASPLOS 2023, §5: "a tree data structure that
     divides a collective into individual GPU programs"), the `<algo>` element with one `<gpu>` per rank; here this
     node's, its `table` naming each compiled function's plan (G4: "Its name is A9's: the serialized operand in
     MSCCL-IR's shape, whose top level is the `<algo>` element with one `<gpu>` per rank"; "Root" names the trees'
     roots, and the record is not called so).
   - Its words are what A9 and the failport's lifecycle need (G4), each with one writer (C3's rule 2, "Each state word
     has one writer"; L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986: a single-writer
     register):
     - `published`: the operand's generation, minted from hdr `serial` (atomic_fetch_add + 1, never 0), 0 before the
       region's first.  Its one writer is the program's process (B2: "The record has one writer by construction, the
       program's process, since a run is one program on the whole collective (A7, corrected), so it needs no exchange
       and no lock"): mesh_plan stores it last, release, once every word of the operand is written (RCU's publication,
       P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998: a structure published only once it is
       initialized; rcu_assign_pointer).  The ports announce it in their `prepared` while they load a plan of it
       (struct mesh_port_info), as the generation was.
     - `retired`: the generation retired, 0 none (mesh-dataflow.h mesh_retire: the program's detach, or the bridge's
       retirement of a gone holder, the failport's lifecycle, as the tag's mesh_retire retires a gone client); release.
     - `released`: the generation the bridge parent released once no port holds a plan of it (mesh-dataflow.h
       mesh_retired_release: deferred reclamation's end of the grace period), its one writer, release, then woken
       (os_sync_wake_by_address_all, region-shared); mesh_plan waits for it, bounded by MESH_PLATFORM_NS, as RCU's
       writer waits out a grace period (`synchronize_rcu`; B2, EB-1), where it was EBUSY.
   - The holder, the process whose operand it is, is the one the heartbeat already knows: hdr `client`'s low 32 bits
     its pid and hdr `client_start` its start (mesh-heartbeat.c replace_turn, a failport lifecycle file, reads them as
     its one client holder; mesh-holders.h mesh_holds), which mesh_plan writes from its context (struct mesh_ctx
     `client`, `start`) before it publishes (G4: "the holder the heartbeat already knows, reused as the failport's code
     is").  `owner` and `start` go, as `closed` and `generation` go into `retired` and `published`: G4, "keeps the
     claim head record's `owner`, `closed` and `start` under a new name, which point 5 removes".
   - `plans`, `group` and `table` as above.
   A plan of the program stands where `published` is not 0, `retired` is not `published` (each acquire) and its state
   section's `closed` is 0 (mesh_algo_plan).  What must stay: 64 bytes, `plans`, `group` and `table` written before
   `published`
   2026-11-06 (119; design/compilation-coordination.md, "The root cause of the copies, and the revised work list",
   R-3, R-6, R-8 and R-10; "The revisions of 14b9a757"):
   - `nodes`, where `padding` was: the node count n of the map the operand was planned on (the installed view's,
     mesh-dataflow.h mesh_nodes_read); `adjacent`, where `padding1[0]` was: the region offset of n + 1 uint32_t words
     `first` and then the places, map index i's places [first[i], first[i + 1]) of the words after `first` (mesh-graph.h
     mesh_graph_rows' compressed rows, keyed by map index), each a place k below `plans` whose plan has an entry naming
     node i (struct mesh_hop `node`, either direction), ascending; mesh_plan's, placed with the table and written before
     `published` (mesh_algo_adjacent).  Why: R-6, "Each cable's processor executes its list of instructions in the
     compile-time order, firing each on its dependency word ... It scans no table of plans"; W8, "The places with flows
     toward each adjacent node are fixed when the plan is made, so they are a list the plan writes".
   - Every field is read through the acquired publication, never before it (R-3: "every field of the published operand
     is read through the acquired publication, never before it (McKenney and Slingwine)"; RCU's rcu_dereference):
     mesh_algo_open or mesh_algo_unreleased first, then `plans`, `group`, `table`, `nodes` and `adjacent`.
   - One writer a word (R-8: "the program writes `retired`, the bridge writes only `released`, and a gone holder is
     read from the heartbeat's words"; L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986):
     `published` and `retired` are the program's process's, `released` the bridge parent's.  The operand is open
     where `published` is not 0 and neither `retired` nor `released` is it (mesh_algo_open); a gone holder's operand is
     open until the bridge, which reads the holder from the heartbeat's words (hdr `client`, `client_start`;
     mesh-holders.h mesh_algo_holder), releases it.
   What must stay: 64 bytes, `plans`, `group`, `table`, `nodes` and `adjacent` written before `published` */
struct mesh_algo { _Alignas(64) _Atomic uint64_t published; _Atomic uint64_t retired,released; uint32_t plans,nodes;
  uint64_t group,table,adjacent,padding1; };
_Static_assert(sizeof(struct mesh_algo)==64 && offsetof(struct mesh_algo,retired)==8 &&
  offsetof(struct mesh_algo,released)==16 && offsetof(struct mesh_algo,plans)==24 && offsetof(struct mesh_algo,nodes)==28 &&
  offsetof(struct mesh_algo,group)==32 && offsetof(struct mesh_algo,table)==40 &&
  offsetof(struct mesh_algo,adjacent)==48,"mesh_algo");

/* The deliveries' channel (design/aot-compilation-change-request.md at mesh 0e56d8da, appendix item 2, "Delivery at a
   node"; design/compilation-coordination.md, "The revisions of 46ae8fdf", CH and M3 to M6): the calls this node's bridge
   parent has delivered, in delivery order, each one ORDER's (mesh-bridge.h struct mesh_order: the fixed sequencer's
   relayed message, "m is the submitted call, its plan and its execution") plan name (`group`, `epoch`) and executions
   [`first`, `end`), "a first-in, first-out channel whose writer never waits" (G. Kahn, The semantics of a simple
   language for parallel programming, IFIP Congress 1974: an unbounded FIFO queue between processes).  The bridge's
   own (item 2: "How the bridge holds the channel is the bridge's own"; R7): an area of the region laid out with it, as
   the map's records are (design/disjoint-pages-change-request.md, Appendix A, A3, "The bridge's own pages (its
   records) are laid out once with the region"), of hdr `deliveries_room` entries at `deliveries_off`; the k-th
   delivery (k from 0, hdr `delivered` the count, the bridge parent its one writer, only rising) in entry k mod
   `deliveries_room`.  Each port's child copies every delivery into its own memory as it reads it, in order, and tells
   the count it has read in hdr `delivery_read[p]` (port p's alone, only rising); an entry is reused for delivery k only
   once every port that reads (its link up with a queue pair in its plan, as mesh-ports.c record_reach lists a cable)
   has read past delivery k - `deliveries_room`, by deferred reclamation ("An entry is reclaimed once every port has read
   past it"; P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004): the
   readers' announcements read before the reuse (mesh-records.c orders_deliver; mesh-flow.c deliveries_copy).
   2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", CH2), reversing the copy and the reuse's
   test above (the implementation's M5 and M4): each lane of a port reads the channel in place, by view, entry k mod
   `deliveries_room` for delivery k below `delivered` (mesh-flow.c lane_take), and the bridge parent writes entry k mod
   `deliveries_room` for delivery k with nothing read before it (mesh-records.c orders_deliver); `delivery_read[p]` is
   read and written no more, kept by name.  Why: design/aot-compilation-change-request.md, appendix item 2: "The
   delivered calls reach the ports, in delivery order, through a channel laid out once with the region ... Each port
   reads it in place, by view, and nothing copies it.  Only the driver moves bytes"; "The writer never waits and never
   checks for a full channel"; "(M4 and M5 go)".  The item's room, "fixed at compile time" by each plan's claim of
   "that many entries", is not built ("The revisions of f3f7cf46", stop 1): the room is MESH_DELIVERY_ROOM, as M3 laid it
   out.  What must stay: the area laid out once with the region; delivery k in entry k mod `deliveries_room`.
   2026-11-05 (114; design/compilation-coordination.md, "The revisions of abecd389", PQ), reversing the one channel for
   every plan and MESH_DELIVERY_ROOM (CH2-2, M3's room): each plan's delivered calls are its own queue in its state
   section (struct mesh_state `deliveries`, `room`, `delivered`), an entry one delivered message, its sequence number
   (`sequence`, the sequencer's sn(m)) and its executions [`first`, `end`), `padding` 0; hdr `deliveries_off`,
   `deliveries_room` and `delivery_read` are read and written no more, kept by name.  Why:
   design/aot-compilation-change-request.md, appendix item 2: "The data structure is synchronous dataflow's. "We can
   replace each arc with a FIFO queue (buffer) to pass data from one block to another ..." (E. A. Lee and D. G.
   Messerschmitt, Static scheduling of synchronous data flow programs for digital signal processing, IEEE Trans.
   Computers C-36(1), 1987).  The arc here runs from a node's delivery of a plan's calls to the bridge's taking of them
   at that node, so there is one queue for each plan at each node of it"; "The overflow came from the shared channel,
   which no cited work has".  What must stay: each entry one message with its sequence number, read in place.
   The submissions' queue (struct mesh_state `submits`, `outstanding`, `submissions`): one entry a start that raised
   the plan's `executions`, `end` the executions after the raise and `clock` the program's Lamport clock at it, the
   arc from the program's submission to its sender's TO-broadcast (Lee and Messerschmitt, as above); its room
   A10's bound on the calls of the function outstanding at the program (design/compilation-coordination.md, "The
   revisions of abecd389", PS) */
struct mesh_delivery { uint64_t sequence,first,end,padding; };
_Static_assert(sizeof(struct mesh_delivery)==32,"mesh_delivery");
struct mesh_submission { uint64_t end,clock; };
_Static_assert(sizeof(struct mesh_submission)==16,"mesh_submission");
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
     (atomic_fetch_add + 1, unmasked, never 0: design/compilation-coordination.md, C-gen); it only rises.
     2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4; "The
     revisions of 55b9cbad"): `client` and `client_start` the holder of the program's operand (struct mesh_algo's
     note), the client word of the program whose operand it is (low 32 bits its pid) and its start (mesh-holders.h
     mesh_started), written by mesh_plan from its context (mesh-dataflow.h struct mesh_ctx `client`, `start`) before it
     publishes (release), where they were written by no one; the heartbeat reads them as before.  Why: G4, "the holder
     the heartbeat already knows, reused as the failport's code is".  What must stay: every field in its place; nothing
     written at attach (design/plan-to-bridge-change-request.md R8) */
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
  /* the page lists' area and room until 110 (`hop_off`, `hop_capacity`; struct mesh_hop's note), never read or
     written; `vacant8` the word stride's bytes until 108 (the note above), never read or written.
     2026-11-06 (119; design/aot-compilation-change-request.md A7, corrected 2026-11-06: "The map a view is created from
     is the sequencer's, and the view's first message carries it: the records the sequencer's map holds when it creates
     the view.  Every member plans on those records"; "The revisions of 14b9a757"): `view_map` and `view`, where
     `vacant16` and `vacant17` were, and `view_off` and `view_nodes`, where `vacant18` and `vacant19` were.  `view` the
     Lamport time of the view this node's bridge installed (mesh-records.c view_install), 0 before its first, and its
     `view_nodes` records (mesh-map.h struct mesh_record), the ones the view's first message carried, at `view_off`
     (mesh-map.h mesh_layout: room `map_room`, laid out once with the region), written under `view_map`, a seqlock as
     `map` is (odd while the bridge parent writes, mesh_notify before and after; read whole as mesh_map_whole reads `map`:
     even before the read and unchanged after it, mesh-dataflow.c mesh_nodes_read).  The
     bridge parent their one writer.  `view_sequencer` and `view_boot` (appended) the view's creator, written with
     `view` under `view_map`: the view's identifier is struct mesh_view_id (mesh_view_installed).  A plan is made on
     those records (mesh-dataflow.h mesh_nodes_read), its view that identifier (struct mesh_state `serial`,
     `sequencer`, `boot`), and stands while the installed view is it.  What must stay: every other field in its
     place */
  _Atomic uint64_t view_map,view;
  uint64_t vacant8;
  uint64_t view_off;
  _Atomic uint64_t view_nodes;
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
     read or written: each plan's is its claim head's publication `state`.
     2026-11-05 (117; design/compilation-coordination.md, "The revisions of 4e78f6f4"): `program_off`, where `vacant21`
     was: the region offset of the program's serialized operand's root (struct mesh_program), the layout's (mesh-map.h
     mesh_layout), set once with the region.  Why: design/compilation-coordination.md, "The documents' corrections of
     2026-11-05", point 5: "the serialized operand's place is a constant the region's layout names once".  What must
     stay: every field in its place.
     2026-11-05 (118; "Review of 4e78f6f4 against the requirement lines", G4; "The revisions of 55b9cbad"): `algo_off`,
     where it was `program_off`: the region offset of the program's serialized operand's top level, struct mesh_algo
     (MSCCL-IR's `<algo>`).  What must stay: its place, set once with the region */
  uint64_t algo_off;
  /* 110 (design/compilation-coordination.md, "The region"): a count that only rises.  Every publication of a plan
     (mesh_plan), every close of a claim (mesh-dataflow.h mesh_claim_close, mesh_plan_retire, mesh_retire), every
     start (mesh-call.h mesh_transfers_start) and the bridge parent's first publication of a plan's sequence numbers
     (mesh-records.c orders_deliver, before it stores the plan's first number, its `released` 0), the bridge parent
     after each turn of orders_deliver that published, and a port where a plan's `taken[p]` rose (mesh-flow.c
     taken_tell, plan_end) add 1 (release), then wake it (mesh_notify, OS_SYNC_WAKE_BY_ADDRESS_SHARED); the ports read
     it once a pass and walk the claims' heads where it moved.  Many writers, as a count has; never reset.
     2026-11-05, owner C3 (design/compilation-coordination.md, "The documents' revisions of edd4b583", "Amendments
     during the work (2026-11-05)", DL7): the bridge parent's first publication of a plan's sequence numbers named
     among the raisers.  Why: a plan stands at mesh_plan's store of its head's publication `state`, before mesh_plan
     raises this count, so a parent can publish the plan's first number, and a later plan's, before that raise; a
     port's merge (DL3: every candidate read, then every plan's `released` and this count again, in Lamport's opposite
     order) must see every plan that can hold a smaller number before it takes a later one, and a walk that reads the
     parent's raise reads the plan standing.  What must stay: a count that only rises, many writers, never reset.
     2026-11-05, owner C3 (the amendments on the reviews, DL9; reviews X1 #7, V1 #9): the parent's turn that
     published and a port's rise of `taken[p]` named among the raisers, where each woke the count without raising it.
     Why: a wake with no raise that lands between a waiter's read of `plans` and its wait is lost until MESH_PAUSE_NS,
     and the delivery now hands off between the parent and its ports once per execution past the u-th
     (os_sync_wait_on_address returns at once where the word has moved).  What must stay: as above.
     2026-11-05 (113; design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): the parent raises it after
     each turn that delivered a call into the deliveries' channel, where it raised it before a plan's first sequence
     number and after a turn that published; no port raises it for a `taken[p]` (gone).  What must stay: as above */
  _Atomic uint64_t plans;
  /* 113 (struct mesh_delivery's note; design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): the
     deliveries' channel, `deliveries_room` entries at `deliveries_off` (mesh-map.h mesh_layout); `delivered` the calls
     the bridge parent has delivered into it, only rising (release after the entry's stores); `delivery_read[p]` the
     deliveries port p's child has read, only rising (release after its copies).  2026-11-05 ("The revisions of
     f3f7cf46", CH2): `delivery_read[p]` read and written no more, kept by name, every field in its place (struct
     mesh_delivery's note).
     2026-11-05 (114; design/compilation-coordination.md, "The revisions of abecd389", SQ2 and PQ): `deliveries_off`,
     `deliveries_room` and `delivery_read` read and written no more, kept by name, the channel's area gone from the
     layout; `delivered` is this node's count of the sequencer's messages delivered or discarded here, Fig. 5's
     nextdeliver - 1 (X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4), 2004, Fig. 5: "while ∃(m′, seqnum′)
     ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver := nextdeliver + 1"), so every message numbered at
     most `delivered` is in its plan's queue here or was not this node's (§5.2.4, "processes not in Dest(m) discard
     m"); the bridge parent its one writer, stored 0 at its start (mesh-records.c map_adopt) and only rising after,
     release after the entries and hdr `plans`' raise (mesh-records.c orders_deliver).  The ports' merge reads it first
     (mesh-flow.c lane_take: appendix item 2, "every entry below the node's delivery count is in its queue already, and
     no smaller one comes later").  What must stay: every field in its place */
  uint64_t deliveries_off,deliveries_room;
  _Atomic uint64_t delivered;
  _Atomic uint64_t delivery_read[MESH_PORTS];
  _Atomic uint32_t view_sequencer;
  uint32_t vacant22;
  _Atomic uint64_t view_boot;
};
/* every field at its 106 offset, the 108 fields after them, the 109 field after those, the 110 field after that (the
   vacant bytes' notes above struct hdr) */
_Static_assert(offsetof(struct hdr,links)==28 && offsetof(struct hdr,wire_pages)==36 && offsetof(struct hdr,planes_off)==56 &&
  offsetof(struct hdr,target_off)==104 && offsetof(struct hdr,data_off)==128 && offsetof(struct hdr,target_stride)==152 &&
  offsetof(struct hdr,client)==160 && offsetof(struct hdr,view_map)==224 && offsetof(struct hdr,view)==232 &&
  offsetof(struct hdr,map_serial)==288 &&
  offsetof(struct hdr,bridge_off)==296 && offsetof(struct hdr,client_start)==304 && offsetof(struct hdr,stopping)==344 &&
  offsetof(struct hdr,vacant7)==40 && offsetof(struct hdr,vacant14)==44 && offsetof(struct hdr,vacant8)==240 &&
  offsetof(struct hdr,view_off)==248 && offsetof(struct hdr,view_nodes)==256 && offsetof(struct hdr,map_off)==272 &&
  offsetof(struct hdr,vacant9)==352 && offsetof(struct hdr,vacant13)==384 && offsetof(struct hdr,map_room)==392 &&
  offsetof(struct hdr,algo_off)==400 && offsetof(struct hdr,plans)==408 && offsetof(struct hdr,deliveries_off)==416 &&
  offsetof(struct hdr,delivered)==432 && offsetof(struct hdr,delivery_read)==440 &&
  offsetof(struct hdr,view_sequencer)==504 && offsetof(struct hdr,view_boot)==512 && sizeof(struct hdr)==520,
  "struct hdr: no field moves");
/* the installed view's identifier (struct mesh_view_id: hdr `view`, `view_sequencer`, `view_boot`) */
static inline struct mesh_view_id mesh_view_installed(struct hdr *m,memory_order order){
  return (struct mesh_view_id){.id=atomic_load_explicit(&m->view,order),
                               .sequencer=atomic_load_explicit(&m->view_sequencer,order),
                               .boot=atomic_load_explicit(&m->view_boot,order)};
}
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
/* the program's serialized operand's top level, at the layout's one place (struct mesh_algo's note) */
static inline struct mesh_algo *mesh_algo(struct hdr *m){ return (struct mesh_algo *)((char *)m+m->algo_off); }
/* function k + 1's state section, the operand's `table` word k (struct mesh_algo's note), closed or standing; the
   caller's k below the operand's `plans` (2026-11-05, "The revisions of 4e78f6f4": the one reading of a table word,
   where mesh_program_plan, mesh-dataflow.c mesh_holding_retained and mesh-flow.c held_clear each read it, sub-agent
   D's finding; the release and a dead child's `held` read closed plans too).  2026-11-05 (118; "The revisions of
   55b9cbad", G4): mesh_algo_state, where it was mesh_program_state.  What must stay: the one reading of a table
   word */
static inline struct mesh_state *mesh_algo_state(struct hdr *m,const struct mesh_algo *algo,uint32_t k){
  return mesh_state_at(m,((const uint64_t *)(const void *)((const char *)m+algo->table))[k]);
}
/* function k + 1's plan (struct mesh_algo `table` word k), NULL where it does not stand: the root's `owner` 0
   (acquire) or `closed` 1 (acquire), k not below its `plans`, or the plan's `closed` 1 (acquire).  The bridge's one
   reading of the program's plans, each at its compile-time place (A9, "What the bridge runs").
   2026-11-05 (118; design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G4 and
   K2: "Run-time bound and null tests of facts the plan fixes go: mesh.h 1017 and 1019"; "The revisions of
   55b9cbad"): mesh_algo_plan, NULL where `published` (acquire) is 0 or `retired` (acquire) is `published`, or the
   plan's `closed` (acquire) is 1; the test of k against `plans` and of the table word against 0 go.  Why: K2; the
   caller's k is a place the plan fixed (below `plans`, mesh-bridge.h plan_from, or a plan's own epoch less 1 at every
   rank: mesh-collective.h struct mesh_function), and every table word of a published operand names its plan's state
   section (mesh-collective.c mesh_plan writes it before `published`).  What must stay: the publication and the
   retirements read, each acquire */
/* the program's operand open: its `published` generation where it is not 0 and `retired` is not it (each acquire),
   else 0 (2026-11-05, "The revisions of 55b9cbad", G2 and G4: the one form of the test that mesh_algo_plan,
   mesh-records.c clients_turn and its pid's retirement, mesh-link.c region_adopt and mesh-stat.c each wrote; "merge
   duplicated patterns found by analysis").  What must stay: the publication read before the retirement */
/* 2026-11-06 (119; design/compilation-coordination.md, "The root cause of the copies, and the revised work list",
   R-3, R-8 and R-10; "The revisions of 14b9a757"):
   - mesh_algo_unreleased: the operand published and not yet released, its `published` generation where it is not 0
     and `released` (acquire, after it) is not it, else 0.  R-10: ""published and not yet released"" written three
     times (W13: mesh-collective.c, mesh-dataflow.c, mesh-flow.c), "to be beside `mesh_algo_open`": its one form.
   - mesh_algo_open: unreleased, and `retired` (acquire) not it either.  The `released` test is R-8's: the bridge
     writes no `retired` for a gone holder, so the operand it released stays published and unretired, and is open no
     more once released.
   - mesh_algo_standing: function k + 1's plan where its `closed` (acquire) is 0, else NULL, read by a caller that has
     acquired the publication (mesh_algo_open) and read `plans` after it, k below them.  mesh_algo_plan: the two in
     that order.  R-3: "every field of the published operand is read through the acquired publication, never before it"
     (McKenney and Slingwine; rcu_dereference before the fields it guards).
   What must stay: the publication read before every other word of the operand */
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
static inline struct mesh_state *mesh_algo_plan(struct hdr *m,uint32_t k){
  return mesh_algo_open(m)?mesh_algo_standing(m,k):NULL;
}
/* the places of the plans with an entry toward map index i (struct mesh_algo `adjacent`): *count of them at the
   result, ascending; read after the publication's acquire (R-3, R-6) */
static inline const uint32_t *mesh_algo_adjacent(struct hdr *m,uint32_t i,uint32_t *count){
  const struct mesh_algo *algo=mesh_algo(m);
  const uint32_t *first=(const uint32_t *)(const void *)((const char *)m+algo->adjacent);
  *count=first[i+1]-first[i];
  return first+algo->nodes+1+first[i];
}
/* declared transfer `transfer`'s range's words at this node (struct mesh_state `transfers`; R-1) */
static inline const struct mesh_publication *mesh_transfer_at(struct hdr *m,const struct mesh_state *state,
                                                              uint32_t transfer){
  return (const struct mesh_publication *)(const void *)((const char *)m+state->transfers+(size_t)transfer*64);
}
/* the synchronization edges of plan `state` here, `waits` into it then `tells` out of it (struct mesh_sync's note) */
static inline struct mesh_sync *mesh_syncs(struct hdr *m,const struct mesh_state *state){
  return (struct mesh_sync *)(void *)((char *)m+state->syncs);
}
/* tag pre-return-20261103 rdma/mesh.h mesh_records and mesh_map_whole, verbatim; mesh_records moved to mesh-map.h
   (review R1, C1) */
/* the map, `map_nodes` records at `map_off`, at most `map_room` (the layout's area of the map from 108, where it was
   arena pages the bridge leased for itself, and leased again larger as the mesh grew); the bridge notifies `map`, its
   serial, before and after each change (odd while it writes; its own record is in before the region opens, so a
   client never reads 0), and a reader's copy is whole where `map` read (acquired) before it was even and is unchanged
   after it */
/* 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-10 and
   W13: "Duplicated patterns, each to be merged"; "The revisions of 14b9a757", sub-agent D1's finding): the seqlock
   reader's test over its sequence word `lock` (an acquire fence, then `before` even and `lock` unchanged), the one form
   mesh_map_whole reads `map` by and the installed view's readers read `view_map` by (mesh-flow.c records_read,
   mesh-dataflow.c mesh_view_copy), where each wrote the two steps itself.  What must stay: the fence before the second
   read */
static inline int mesh_seqlock_whole(const _Atomic uint64_t *lock,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(lock,memory_order_relaxed)==before;
}
static inline int mesh_map_whole(struct hdr *m,uint64_t before){ return mesh_seqlock_whole(&m->map,before); }
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

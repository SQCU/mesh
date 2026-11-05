#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
/* mesh_started, mesh_holds, mesh_gone (sub-assignment 4's header; tag pre-return-20261103 rdma/mesh-dataflow.h) */
#include "mesh-holders.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* The program's context: the region mapped (`M`, `len`, `fd`) and the client word it minted (`client`, each of its
   claims' head owner).  It holds no rows and no pages: a program's rows and pages are its plans' claims (struct
   mesh_claim, mesh_claim_take), each taken by mesh_plan for exactly what that plan places
   (design/disjoint-pages-change-request.md, Appendix C, C3.4; design/disjoint-pages-revision-coordination.md, "The
   program's claim and its sections").  It holds no plan: a plan's handle is its state section's region offset,
   mesh_plan's `*plan`, which mesh_transfers_start, mesh-metal.h and mesh_plan_retire take
   (design/compilation-coordination.md, "The appendix's revisions", N2, N5).
   2026-11-04, owner L (design/plan-to-bridge-change-request.md R8): `padding`, `epoch` and `state` appended, where the
   region's one plan was hdr `state` and the client word hdr `client`.  What must stay: no rows or pages in the
   context.
   2026-11-04, owner L2 (design/aot-compilation-change-request.md, appendix item 6; N5): `epoch` and `state` gone
   (32 bytes).  Why: compiled functions coexist, each its own plan (item 6), so the context holds no standing plan and
   no count of plans; a plan is named by its function (mesh-collective.h mesh_plan, N1).  What must stay: no rows,
   pages or plans in the context.
   2026-11-04, sub-assignment 5: reversing sub-assignment 7's 2026-11-03 form, where the context held the program's one
   claim, taken at attach (`rows`, `window`, `local`: every free row and the widest free run of pages in the window and
   outside it) and laid out by cursors (`row_cursor`, `local_cursor`).  Why: C3.4, "Nothing is claimed at attach", and
   C4 (mesh_claim's claim at attach of everything free; the claim's local run, where values were laid out when
   created).  What must stay: no call site but mesh_plan's claim takes rows or pages from the region.
   2026-11-05 (design/compilation-coordination.md, "The revisions of abecd389", PS): `clock` appended (40 bytes), the
   program's logical clock, which each of its starts that raises a plan's `executions` increments and stamps on the
   submission (mesh-call.h mesh_transfers_start): L. Lamport, Time, clocks, and the ordering of events in a distributed
   system, CACM 21(7), 1978, IR1, "Each process Pi increments Ci between any two successive events"; 0 at attach.  Why:
   design/aot-compilation-change-request.md A7: sender(m) "TO-broadcasts the program's submissions in the order the
   program makes them", and the order among one program's plans reaches its sender's bridge only so (mesh-records.c
   orders_submit).  What must stay: no rows, pages or plans in the context */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; uint64_t clock; };
_Static_assert(sizeof(struct mesh_ctx)==40 && offsetof(struct mesh_ctx,fd)==24 && offsetof(struct mesh_ctx,padding)==28 &&
  offsetof(struct mesh_ctx,clock)==32,"mesh_ctx: rdma/mesh.py Context");
/* design/prepared-machine.md#M09 */
/* The two arena ranges.  [0,wire_pages) is the registered window and the only memory an SGE may
   name; [wire_pages,arena) is addressable and never registered.  An undeclared window is the whole
   arena, and then there is one range over it, exactly as before the split. */
static inline struct mesh_range mesh_arena_range(const struct hdr *m,int wire){
  uint32_t arena=mesh_arena_pages(m),window=m->wire_pages>arena?arena:m->wire_pages;
  uint32_t begin=wire||window==arena?0:window,end=wire?window:arena;
  return (struct mesh_range){begin,end-begin};
}
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
/* the region attached (design/plan-to-bridge-change-request.md R8; design/compilation-coordination.md, "The region"):
   the region mapped and its magic, version and length checked; the client word minted, ((hdr `serial` + 1) &
   0x7fffffff) << 32 | pid, bit 63 set where hdr `device_client`'s is clear, as before; the version read again.  It
   holds nothing and writes nothing in the region but hdr `serial`: no client word, no row, no page.  0, EINVAL where
   the region is not of this layout or expired as it was attached (it is let go), else the errno.
   2026-11-04, owner L (R8): reversing the 109 form, the compare-and-swap of hdr `client` (EADDRINUSE where another live
   client held it, an earlier gone client retired first) and hdr `client_start` written.  Why: "nothing happens when a
   program joins or leaves because a program is a sequence of like, function calls"; a program holds its plans'
   claims and nothing else.  What must stay: the version read again after the client word is minted.
   2026-11-04, owner L2 (design/compilation-coordination.md, "The appendix's revisions", N5): no `epoch` or `state`
   set, the context holding neither.  What must stay: nothing held at attach */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
/* the client retired (design/plan-to-bridge-change-request.md R8; e6bd7db2's section 6, as design/compilation-
   coordination.md, Q5 takes it): every claim whose head's `owner` is `client` closed (struct mesh_buffer `closed`,
   release), then hdr `plans` raised and woken; nothing is let go here, the bridge releasing each closed claim once no
   port holds it (mesh_retired_release).
   2026-11-04, owner L: reversing the 109 form, the retirer's compare-and-swap of hdr `client`, hdr `hops` and `state`
   stored 0 before the claims closed, and `client` cleared after.  Why: those words are vacant (mesh.h 110); a plan's
   unpublishing is its claim's close, which the ports read through `plans`.  What must stay: nothing let go here */
void mesh_retire(struct hdr *,uint64_t client);
/* the program's retirement of one compiled function (design/aot-compilation-change-request.md, appendix item 6: a
   function's claim released only when the program retires that function or itself; design/compilation-
   coordination.md, "The appendix's revisions", N3, as NVIDIA, CUDA Programming Guide §4.2.3.1, cudaGraphExecDestroy(),
   destroys one instantiated graph): the claim of this client whose head's publication `state` (acquire) is `plan`
   closed (struct mesh_buffer `closed`, release), then hdr `plans` raised and woken (mesh.h mesh_notify); nothing is let
   go here, the bridge releasing the claim once no port holds it (mesh_retired_release; C3.4).  0, or ENOENT where no
   claim of this client holds that plan standing (mesh_head_state: a closed claim holds none, so a second retire of
   one plan is ENOENT; 2026-11-04, owner L2, the coordinator's ruling on review W1's finding 4) */
int mesh_plan_retire(struct mesh_ctx *,uint64_t plan);
/* the first claim head row after `row` (row UINT32_MAX: the first in the region), UINT32_MAX where none: a row of
   MESH_ROW_OWN whose struct mesh_buffer `owner` reads (acquire) not 0, past `row`'s own claim where `row` is a head (its
   `pages`, stored before its owner word).  A walk of the heads (e6bd7db2's section 6, "Discovery"; design/compilation-
   coordination.md, Q5) reads a moving set: a claim taken or released during the walk may be met or missed, and the
   walk ends within one pass over the rows */
uint32_t mesh_head_after(const struct hdr *,uint32_t row);
/* the state section of the plan standing at claim head `head` (struct mesh_buffer's note: `owner` != 0, `closed` 0 and
   the head's publication `state` != 0, each read with acquire), NULL where none stands (design/compilation-
   coordination.md, "The region") */
struct mesh_state *mesh_head_state(struct hdr *,uint32_t head);

/* A plan's claim (design/disjoint-pages-change-request.md, Appendix C, C3.4; design/disjoint-pages-revision-
   coordination.md, "The program's claim and its sections"): the only place rows and pages are taken, mesh_plan's one
   call, for exactly the rows and pages that plan places.  Any number of claims coexist, each its own: a retired plan's
   with the next plan's, until the bridge releases it. */
struct mesh_claim { struct mesh_range rows,window,local; };
/* rows: `rows` free page-table rows (MESH_ROW_OWN), one run; window: `window` free arena pages inside the registered
   window, local: `local` outside it (inside it where the window is the whole arena), each one block-aligned run, first
   fit; recorded in the rows' first row (its head, mesh.h struct mesh_buffer's note: the three runs, `generation`, the
   head's publication 0, `start` mesh_started(getpid()), then the client its owner, last, release) for the bridge's
   release.  Taken whole or not at all: 0, or ENOMEM with nothing taken.  rows >= 1 (the head)
   The head row is the claim's record alone (mesh-dataflow.c, "The claim's record"): its struct mesh_buffer and struct
   mesh_page_entry are written by the claim, its close and its release, and no value's rows include it, so a plan's
   values' rows are [rows.first + 1, rows.first + rows.count) */
int mesh_claim_take(struct mesh_ctx *,uint64_t generation,uint32_t rows,uint32_t window,uint32_t local,
                    struct mesh_claim *);
/* this client's claim of `generation` closed (release); hdr `plans` raised and woken (mesh.h mesh_notify), so the
   bridge's ports look again (2026-11-04, owner L: where hdr `plan` was woken, vacant from 110).
   2026-11-04, owner L2 (design/compilation-coordination.md, "The appendix's revisions", N3, N5): the `others` form
   (every claim of this client but `generation`'s closed) gone, with mesh_plan's step 6.  Why: compiled functions
   coexist (design/aot-compilation-change-request.md, appendix item 6), a function's claim closed only by its
   retirement, the program's, or its own plan call's failure.  What must stay: the close a release, `plans` raised and
   woken after it */
void mesh_claim_close(struct mesh_ctx *,uint64_t generation);

/* rows [first, first + pages / block) of a claim bound to arena pages [page, page + pages), block by block, with
   `index` the copy (an entry's `mapping` high half): mesh_plan's binding of the rows and pages it placed; nothing is
   taken here, only named */
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
/* 2026-11-04, sub-assignment 5: mesh_publication_prepare is removed (the GPU's publication records went with layout
   109, mesh.h; its one caller, mesh-metal.m, reads a range's first row's publication itself).  What must stay: a
   range's release is its supplied word, which its first row's publication names */
/* the bridge's release of closed claims (deferred reclamation: RCU's call_rcu, P. E. McKenney and J. D. Slingwine,
   Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004; the contract's D10; the hazard pointer's announcement
   read before reclamation, M. M. Michael, IEEE TPDS 15(6), 2004): for each closed claim, a seq_cst fence, then the claim
   kept where any port's `prepared` is its generation or its plan's struct mesh_state `held` is not 0, and released
   otherwise, its pages' bits, then its rows' bits, by the winner of the head's owner exchange; the closed claims kept.
   The bridge parent's (mesh-records.c clients_turn, mesh-link.c region_adopt); nobody waits for it
   (design/compilation-coordination.md, "The region") */
uint32_t mesh_retired_release(struct hdr *);

/* The map reader (design/page-queue-change-request.md, "What the collective decides, and what the bridge decides"): the
   collective sees node-level adjacency and each node pair's capacity, nothing else of how the bridge carries pages.
   `serial` is the map's (hdr `map_serial`, the one a plan is made against: struct mesh_hop `serial`); `count` the map's
   nodes, at most MESH_NODES_MOST; `node[i]` the node id of index i, a node's index its place in ascending node-id order
   among the map's records; `self` this node's index; `capacity[i][j]` == `capacity[j][i]` the bits a second of the
   cables between i and j, summed, each a cable both records list (mesh_cables) at the lesser of its two bandwidths;
   0 where none joins them. */
#define MESH_NODES_MOST 32
struct mesh_nodes { uint64_t serial; uint32_t count,self; uint32_t node[MESH_NODES_MOST];
                    uint64_t capacity[MESH_NODES_MOST][MESH_NODES_MOST]; /* bits/s, a pair's cables summed; 0 none */ };
/* 0, or an error value: ETIMEDOUT where no whole copy was read within MESH_PLATFORM_NS of its start, ESRCH where the
   bridge left its write open (its process gone), E2BIG past MESH_NODES_MOST records, EINVAL where the records lie
   outside the region, ENOENT where the map holds no record of this node (hdr `node`); read under the map seqlock
   (mesh.h mesh_map_whole) */
int mesh_nodes_read(struct mesh_ctx *,struct mesh_nodes *);
/* mesh_joined, struct mesh_cable and mesh_cables, which name records and cables, are mesh-map.h's, bridge-side (review
   R1, C1) */
#endif

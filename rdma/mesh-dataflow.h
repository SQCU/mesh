#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
/* mesh_started, mesh_holds, mesh_gone (sub-assignment 4's header; tag pre-return-20261103 rdma/mesh-dataflow.h) */
#include "mesh-holders.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* The program's context: the region mapped (`M`, `len`, `fd`) and the client word it took (`client`, hdr `client`).  It
   holds no rows and no pages: a program's rows and pages are its plans' claims (struct mesh_claim, mesh_claim_take),
   each taken by mesh_plan for exactly what that plan places (design/disjoint-pages-change-request.md, Appendix C, C3.4;
   design/disjoint-pages-revision-coordination.md, "The program's claim and its sections").
   2026-11-04, sub-assignment 5: reversing sub-assignment 7's 2026-11-03 form, where the context held the program's one
   claim, taken at attach (`rows`, `window`, `local`: every free row and the widest free run of pages in the window and
   outside it) and laid out by cursors (`row_cursor`, `local_cursor`).  Why: C3.4, "Nothing is claimed at attach", and
   C4 (mesh_claim's claim at attach of everything free; the claim's local run, where values were laid out when
   created).  What must stay: no call site but mesh_plan's claim takes rows or pages from the region */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; };
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
/* the region attached: the client's compare-and-swap on hdr `client` (an earlier client whose process is gone retired,
   mesh_retire), its start beside it (hdr `client_start`) and the region's version read again after the swap.  It claims
   nothing (design/disjoint-pages-change-request.md, Appendix C, C3.4): no row, no page, and no wait for a retired
   client's claim, which the bridge releases on its own turn (mesh_retired_release).  0, EINVAL where the region is not
   of this layout or expired as the client was taken (it is let go), EADDRINUSE where another live client holds it,
   else the errno */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
/* the client retired: once the client word is the retirer's, hdr `hops` 0 and `state` 0 (its plan unpublished, release,
   `plan` woken: the coordinator's rulings on owner 7's finding and review 15's finding 2, mesh-dataflow.c), every claim
   it owns closed (struct
   mesh_buffer `closed` on each head whose `owner` it is), then hdr `client` cleared; nothing is let go here, the bridge
   releasing each closed claim once no port holds it (mesh_retired_release) */
void mesh_retire(struct hdr *,uint64_t client);

/* A plan's claim (design/disjoint-pages-change-request.md, Appendix C, C3.4; design/disjoint-pages-revision-
   coordination.md, "The program's claim and its sections"): the only place rows and pages are taken, mesh_plan's one
   call, for exactly the rows and pages that plan places.  Any number of claims coexist, each its own: a retired plan's
   with the next plan's, until the bridge releases it. */
struct mesh_claim { struct mesh_range rows,window,local; };
/* rows: `rows` free page-table rows (MESH_ROW_OWN), one run; window: `window` free arena pages inside the registered
   window, local: `local` outside it (inside it where the window is the whole arena), each one block-aligned run, first
   fit; recorded in the rows' first row (its head: the client, `generation`, the three runs) for the bridge's release.
   Taken whole or not at all: 0, or ENOMEM with nothing taken.  rows >= 1 (the head)
   The head row is the claim's record alone (mesh-dataflow.c, "The claim's record"): its struct mesh_buffer and struct
   mesh_page_entry are written by the claim, its close and its release, and no value's rows include it, so a plan's
   values' rows are [rows.first + 1, rows.first + rows.count) */
int mesh_claim_take(struct mesh_ctx *,uint64_t generation,uint32_t rows,uint32_t window,uint32_t local,
                    struct mesh_claim *);
/* this client's claim of `generation` closed (others 0), or every claim of this client but that one (others 1); hdr
   `plan` woken, so the bridge's ports look again */
void mesh_claim_close(struct mesh_ctx *,uint64_t generation,int others);

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
   Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004; the contract's D10): a seq_cst fence, then each
   closed claim whose generation no port's `prepared` holds released, its pages' bits, then its rows' bits, by the
   winner of the head's owner exchange; the closed claims kept.  The bridge parent's (mesh-records.c clients_turn,
   mesh-link.c region_adopt); nobody waits for it */
uint32_t mesh_retired_release(struct hdr *);

/* The map reader (design/page-queue-change-request.md, "What the collective decides, and what the bridge decides"): the
   collective sees node-level adjacency and each node pair's capacity, nothing else; no cable, port or queue pair.
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

#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
/* mesh_started, mesh_holds, mesh_gone (sub-assignment 4's header; tag pre-return-20261103 rdma/mesh-dataflow.h) */
#include "mesh-holders.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* The program's section of the region (design/disjoint-pages-change-request.md, Appendix A, A3, "Sharing the region
   between programs", and A4: "One claim of a program's section at attach remains"), taken once by mesh_attach and held
   until the client retires; nothing else takes rows or pages from the region:
   - `rows`: the claim's rows of the page table, [first, first + count).  Its first two are the claim's heads, which
     record its page runs for the bridge's release (mesh_retired_release): row `first` the window run's, row `first`
     + 1 the local run's, each its run's first page in its entry's `mapping` (MESH_ABSENT where the run is empty) and
     its pages in its buffer's `pages`.  The sections' rows follow, from `first` + 2.
   - `window`: arena pages [first, first + count), block-aligned, inside the registered window: the pages the plan's
     allocator assigns (mesh-collective.h mesh_plan), and no one else.
   - `local`: arena pages [first, first + count), block-aligned, outside the registered window (where the window is the
     whole arena, the upper half of the claim's run): the pages of the values no plan names (mesh-call.h
     mesh_section_create, wire 0), laid out in creation order.
   - `row_cursor`: the claim's first row no section names; `local_cursor`: `local`'s first page no section holds.
   2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A4 (sub-assignment 7): `send_off` and `send_bytes`,
   the plan's words section that mesh_plan allocated at its call, are gone: the supplied words and the landed list are
   areas of the layout (hdr `word_off`, `landed_off`).  The counts `arena`, `wire`, `shared_pages` and the cursors of
   the call-site allocators (`wire_cursor`, `bulk_cursor`) went with those allocators.  What must stay: the claim
   taken once, at attach, and every row and page a program names inside it */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; struct mesh_range rows,window,local;
                  uint32_t row_cursor,local_cursor; int fd; };
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
/* the region attached: the client's compare-and-swap on hdr `client`, its start beside it (hdr `client_start`), the
   region's version read again after the swap, and the program's one claim (struct mesh_ctx `rows`, `window`, `local`):
   the widest run of free rows and, of the arena's free pages, the widest block-aligned run in the registered window and
   the widest outside it, none of them a page or row a retired client's claim still holds; 0, EINVAL where the region
   is not of this layout or expired as the client was taken (it is let go), EADDRINUSE where another live client holds
   it, ENOMEM where fewer than the claim's two head rows are free (it is let go), else the errno */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);

/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A4 (sub-assignment 7): mesh_rows_alloc and
   mesh_arena_alloc, the call-site allocators (first fit at the call that needs memory, A2), and mesh_rows_release
   are gone; a program's rows and pages are its claim's (struct mesh_ctx), and its plan's pages are the plan's
   allocator's.  What must stay: no call site takes rows or pages from the region */
/* rows [first, first + pages / block) of the claim bound to arena pages [page, page + pages), block by block, with
   `index` the instance (an entry's `mapping` high half) */
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *,uint32_t row,struct prepared_publication *);
/* a retired client's claim released, its rows and its head rows' page runs, none while a port's requests may name
   their pages; the rows kept */
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

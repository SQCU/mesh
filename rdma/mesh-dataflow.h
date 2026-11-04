#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
/* mesh_started, mesh_holds, mesh_gone (sub-assignment 4's header; tag pre-return-20261103 rdma/mesh-dataflow.h) */
#include "mesh-holders.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
/* `send_off`, `send_bytes`: the region offset and the bytes of the words this client's plan allocated (mesh-collective.h
   mesh_plan: a section outside the window, zeroed), which mesh-metal.m wraps for the GPU and its cancellation covers */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client,send_off,send_bytes; uint32_t rows,arena,wire,shared_pages,row_cursor,wire_cursor,bulk_cursor; int fd; };
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
/* the region attached: the client's compare-and-swap on hdr `client`, its start beside it (hdr `client_start`), and the
   region's version read again after the swap; 0, EINVAL where the region is not of this layout or expired as the client
   was taken (it is let go), EADDRINUSE where another live client holds it, else the errno */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* design/prepared-machine.md#M09 */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align,int wire);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *,uint32_t row,struct prepared_publication *);
/* a retired client's closed rows released, none while a port's requests may name their pages; the count kept */
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
/* tag pre-return-20261103 rdma/mesh-dataflow.h mesh_joined, verbatim (the bridge's map reads it: mesh-records.c; the
   collective reads mesh_nodes_read, which sums a pair's capacity through the same join) */
/* the one join of two nodes' records: the cables each one's bridge reaches the other over (both records list them,
   mesh_cables), 0 where none; *beta 8e9 over the least bandwidth either reports of them (0 where neither reports one) */
int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);
/* The bridge's per-cable join (mesh-flow.c link_prepare, "The rule both ends compute"), the one matching of two
   records' cables that mesh_joined and mesh_nodes_read read too: the cables a and b both list for each other, ports
   matching both ways, each `key` (the lower node id's port index, the higher's), sorted by key, `bandwidth` and `qps`
   each the lesser of the two records'; `out` holds MESH_PORTS; their count.  No client API gives a cable: hidden, so
   libmesh.dylib does not export it, and the bridge's image, built from mesh-dataflow.c, calls it */
struct mesh_cable { uint32_t key[2],qps; uint64_t bandwidth; };
__attribute__((visibility("hidden")))
uint32_t mesh_cables(const struct mesh_record *a,const struct mesh_record *b,struct mesh_cable *out /* room MESH_PORTS */);
#endif

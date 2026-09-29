#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_range { uint32_t first,count; };
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client,send_off,send_bytes; uint32_t rows,arena,wire,shared_pages,row_cursor,wire_cursor,bulk_cursor; int fd; };
struct mesh_link_view { uint32_t peer,phase; char device[32]; uint64_t bandwidth; };
/* design/algorithm-sources.md#meshobserve */
int mesh_observe(const char *name,struct mesh_link_view *out,uint32_t capacity,uint32_t *node);
/* The link map as an operand: its shape fixed (MESH_LINK_NODES nodes), its contents varying.  Per node:
   present, as stated.  Per directed link a->b: alpha (us a message) and beta (ns a byte), as stated, and
   up, as node a's bridge observes it (its communicator session with b paired, or lost): the table's own
   node's links from its own sessions, every other node's from that node's last report.  A report is a
   node's links' up with a sequence that grows with each (reported[v]: the sequence of node v's last one
   applied here, 0: none); the bridges pass each report they take on to their peers, a newer one
   replacing an older, so every node's table carries the links it is not on as their nodes observe
   them, and a link no report has named is down, whatever is stated.  The bridge makes the table beside
   its region (<region>.links, `node` its own), writes its links' up, and unlinks it at exit.  Two
   copies of the contents: a writer copies the current one (copy[epoch & 1]) into the other, changes it
   there and publishes it by bumping epoch; a reader copies the current one and keeps it only if epoch
   has not moved meanwhile, so no reader sees a half-written table.  Writers (the bridge, a stated
   configuration) take `writer` one at a time; a write that changes nothing leaves epoch as it is. */
#define MESH_LINK_NODES 16
#define MESH_LINK_MAGIC 0x4d4c4e4bu
struct mesh_link_state { float alpha,beta; uint32_t stated,up; };
struct mesh_link_contents { uint32_t present[MESH_LINK_NODES]; struct mesh_link_state link[MESH_LINK_NODES][MESH_LINK_NODES]; };
struct mesh_link_table { uint32_t magic,node; _Atomic uint32_t writer,padding; _Atomic uint64_t epoch,reported[MESH_LINK_NODES];
  struct mesh_link_contents copy[2]; };
/* The table of the bridge of `region` (NULL: MESH_NAME), mapped; create (the bridge's): made afresh. */
int mesh_link_table_open(const char *region,int create,struct mesh_link_table **);
void mesh_link_table_close(struct mesh_link_table *);
/* One consistent snapshot of the contents; its epoch. */
uint64_t mesh_link_table_read(const struct mesh_link_table *,struct mesh_link_contents *);
/* `edit` applied to a copy of the contents and published; 1 where it changed them (epoch bumped). */
int mesh_link_table_write(struct mesh_link_table *,void (*edit)(struct mesh_link_contents *,const void *),const void *);
/* The bridge's observation: its link to `peer` up or down; 1 where it changed its node's links (its
   report's sequence then moves on), 0 where not, -EINVAL for a node past the table's. */
int mesh_link_table_observe(struct mesh_link_table *,uint32_t peer,uint32_t up);
/* Node `origin`'s report (its links' up, a bit a node, 64 a word) applied where `sequence` is newer than
   the last applied from it; 1 where applied, 0 where not (older, or the table's own node). */
int mesh_link_table_report(struct mesh_link_table *,uint32_t origin,uint64_t sequence,const uint64_t *up);
/* Node v's report as this table holds it: its links' up (a bit a node) and its sequence (0: none). */
uint64_t mesh_link_table_row(struct mesh_link_table *,uint32_t v,uint64_t *up);
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
/* design/algorithm-sources.md#programcopy */
static inline uint32_t mesh_peer_channel(struct mesh_ctx *c,uint32_t peer,uint32_t queue){
  for(uint32_t i=0;i<c->M->links;i++)if(mesh_links(c->M)[i].peer==peer){
    if(queue<c->M->qps)return i*c->M->qps+queue;
    queue-=c->M->qps;
  }
  return MESH_ABSENT;
}
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* design/prepared-machine.md#M09 */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align,int wire);
uint32_t mesh_arena_claim(struct hdr *,uint32_t pages,uint32_t align,int wire);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#index-hand-off */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *,uint32_t row,uint32_t queue);
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *,uint32_t row,struct prepared_publication *);
void mesh_retired_release(struct hdr *);
#endif

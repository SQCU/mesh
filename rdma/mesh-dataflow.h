#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
struct mesh_range { uint32_t first,count; };
/* A link as its region holds it: its peer, its communicator session's phase, its device and bandwidth. */
struct mesh_link_view { uint32_t peer,phase; char device[32]; uint64_t bandwidth; };
/* design/algorithm-sources.md#meshobserve */
int mesh_observe(const char *name,struct mesh_link_view *out,uint32_t capacity,uint32_t *node);
/* The link map as an operand: its shape fixed when the bridge makes it (`nodes` nodes, from the
   bridge's configuration: mesh-flow -N), its contents varying.  Per node: present, as stated.  Per
   directed link a->b: alpha (us a message) and beta (ns a byte), as stated, and up, as node a's bridge
   observes it (its communicator session with b paired, or lost): the table's own node's links from its
   own sessions, every other node's from that node's last report.  A report is a node's links' up with a
   sequence that grows with each (mesh_link_reported(t)[v]: the sequence of node v's last one applied
   here, 0: none); the bridges pass each report they take on to their peers, a newer one replacing an
   older, so every node's table carries the links it is not on as their nodes observe them, and a link
   no report has named is down, whatever is stated.  The bridge makes the table beside its region
   (<region>.links, `node` its own), writes its links' up, and unlinks it at exit.  Two copies of the
   contents: a writer copies the current one (mesh_link_copy(t, epoch)) into the other, changes it there
   and publishes it by bumping epoch; a reader copies the current one and keeps it only if epoch has not
   moved meanwhile, so no reader sees a half-written table.  Writers take `writer` one at a time: the
   bridge (the links' up), a stated configuration (presence, alpha and beta: mesh_link_table_state), and
   the bridges' estimator (mesh-flow -E: a stated link's beta as the node it leads into estimates it from
   its receives, passed to every bridge); a write that changes nothing leaves epoch as it is. */
#define MESH_LINK_MAGIC 0x4d4c4e4cu
struct mesh_link_state { float alpha,beta; uint32_t stated,up; };
/* The contents over `nodes` nodes: link a->b at mesh_link_at(c, a, b), then present[v] (mesh_link_present);
   mesh_link_contents_new makes one of a table's size for a snapshot. */
struct mesh_link_contents { uint32_t nodes,padding[3]; struct mesh_link_state link[]; };
static inline struct mesh_link_state *mesh_link_at(const struct mesh_link_contents *c,uint32_t a,uint32_t b){
  return (struct mesh_link_state *)&c->link[(size_t)a*c->nodes+b];
}
static inline uint32_t *mesh_link_present(const struct mesh_link_contents *c){return (uint32_t *)&c->link[(size_t)c->nodes*c->nodes];}
static inline size_t mesh_link_contents_bytes(uint32_t nodes){
  return (sizeof(struct mesh_link_contents)+(size_t)nodes*nodes*sizeof(struct mesh_link_state)+(size_t)nodes*sizeof(uint32_t)+15)&~(size_t)15;
}
struct mesh_link_contents *mesh_link_contents_new(uint32_t nodes);
/* The table: this header, then reported[nodes], then the two copies. */
struct mesh_link_table { uint32_t magic,node,nodes,padding; _Atomic uint32_t writer,padding2; _Atomic uint64_t epoch; };
static inline _Atomic uint64_t *mesh_link_reported(const struct mesh_link_table *t){return (_Atomic uint64_t *)(t+1);}
static inline struct mesh_link_contents *mesh_link_copy(const struct mesh_link_table *t,uint64_t which){
  return (struct mesh_link_contents *)((char *)(t+1)+(((size_t)t->nodes*8+15)&~(size_t)15)+(size_t)(which&1)*mesh_link_contents_bytes(t->nodes));
}
static inline size_t mesh_link_table_bytes(uint32_t nodes){
  return sizeof(struct mesh_link_table)+(((size_t)nodes*8+15)&~(size_t)15)+2*mesh_link_contents_bytes(nodes);
}
/* The table of the bridge of `region` (NULL: MESH_NAME), mapped: with `nodes` (the bridge's) made afresh
   over that many nodes as node `node`'s, without (0) the one the bridge made. */
int mesh_link_table_open(const char *region,uint32_t nodes,uint32_t node,struct mesh_link_table **);
void mesh_link_table_close(struct mesh_link_table *);
/* One consistent snapshot of the contents (into one of the table's size); its epoch. */
uint64_t mesh_link_table_read(const struct mesh_link_table *,struct mesh_link_contents *);
/* `edit` applied to a copy of the contents and published; 1 where it changed them (epoch bumped). */
int mesh_link_table_write(struct mesh_link_table *,void (*edit)(struct mesh_link_contents *,const void *),const void *);
/* The bridge's observation: its link to `peer` up or down; 1 where it changed its node's links (its
   report's sequence then moves on), 0 where not, -EINVAL for a node past the table's. */
int mesh_link_table_observe(struct mesh_link_table *,uint32_t peer,uint32_t up);
/* Node `origin`'s report (its links' up, a bit a node, 64 a word) applied where `sequence` is newer than
   the last applied from it; 1 where applied, 0 where not (older, the table's own node, past its nodes). */
int mesh_link_table_report(struct mesh_link_table *,uint32_t origin,uint64_t sequence,const uint64_t *up);
/* Node `node`'s reports taken afresh: the sequence of its last applied forgotten (a new instance of it, whose
   sequence may start below its predecessor's). */
void mesh_link_table_forget(struct mesh_link_table *,uint32_t node);
/* Node v's report as this table holds it: its links' up (a bit a node, (nodes + 63) / 64 words) and its
   sequence (0: none). */
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
/* Arena pages of the registered window (wire) or the rest, claimed atomically by any process of the region. */
uint32_t mesh_arena_claim(struct hdr *,uint32_t pages,uint32_t align,int wire);
#endif

#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim): a slot `pages` arena pages on `stride` rows from row `first`; 0, or the
   refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/switched-multipath-dma.md: the nodes of the map this node's bridge reaches, through cables or relays, each with
   the bandwidth of its own cables toward it (0: through relays alone; mesh-plan.h struct mesh_reach), at most `room`;
   their count */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);
/* the programs toward node `node` as a client uses them: its slot and the generation it was opened in.  A client
   holds nothing by opening it */
struct mesh_order { uint32_t slot,node; uint32_t generation,padding; };
/* the programs toward `node`: 0, EINVAL, ENOLINK where this node's bridge's map does not reach it */
int mesh_order_open(struct mesh_ctx *,uint32_t node,struct mesh_order *);
/* whether the slot still names the node it was opened toward */
int mesh_order_live(struct mesh_ctx *,const struct mesh_order *);
/* A program declared toward an order's node (mesh.h struct mesh_declaration, its tables in the client's own leases, which
   stay unchanged and held while it runs): its program slot and the counter of steps issued, the one word its client
   raises at run time (mesh_issue) */
struct mesh_run { struct mesh_order order; uint32_t program,padding; _Atomic uint64_t *issued; };
/* `program` declared toward the order's node, linked to this one or not: this client's program there before it (if
   any) ends at the executions it issued.  Both ends' clients declare the same program, its directions swapped, under
   one key; nothing is agreed and nothing is sent: the bridges switch every piece as it moves (mesh-switched.h).  0;
   EINVAL (a table, span, gate or done word the region does not hold, a span longer than a piece, a step past the
   period); ENOLINK where the slot no longer names the node; ENOSPC where the node's programs are all in use; ENOMEM
   where the arena holds no room for the program's progress */
int mesh_declare(struct mesh_ctx *,const struct mesh_order *,const struct mesh_declaration *,struct mesh_run *);
/* the client's program ended at the executions it issued.  0 */
int mesh_close(struct mesh_ctx *,struct mesh_run *);
/* `steps` steps of the program issued (monotone): the whole of a client's publication */
static inline void mesh_issue(const struct mesh_run *run,uint64_t steps){ atomic_store_explicit(run->issued,steps,memory_order_release); }
/* whether a closed program is wholly done (every transfer of it complete through its last execution) */
int mesh_run_done(struct mesh_ctx *,const struct mesh_run *);
#endif

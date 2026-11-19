#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim): a slot `pages` arena pages on `stride` rows from row `first`; 0, or the
   refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/node-addressed-transfers.md: the nodes this node's bridge holds a live order toward, each with the bandwidth of
   the cables under it (mesh-plan.h struct mesh_reach), at most `room`; their count */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);
/* design/declared-programs.md: the order toward node `node` as a client uses it: its slot and the generation it was
   opened in.  A client holds nothing by opening it */
struct mesh_order { uint32_t slot,node; uint32_t generation,padding; };
/* the order toward `node`: 0, EINVAL, ENOLINK where this node's bridge holds no live order toward it */
int mesh_order_open(struct mesh_ctx *,uint32_t node,struct mesh_order *);
/* whether the order is still the generation it was opened in, and live (a restart of it loses every program of the
   generation: their done words ~0) */
int mesh_order_live(struct mesh_ctx *,const struct mesh_order *);
/* A program declared toward an order's node (mesh.h struct mesh_declaration, its tables in the client's own leases, which
   stay unchanged and held while it runs): its program slot and the counter of steps issued, the one word its client
   raises at run time (mesh_issue) */
struct mesh_run { struct mesh_order order; uint32_t program,padding; _Atomic uint64_t *issued; };
/* `program` declared toward the order's node: this client's program there before it (if any) ends at the steps it
   issued, which are a whole number of executions, and the node's next segment holds the programs going on and this one,
   in key order (mesh-order.h order_compose).  Both ends' clients declare the same program, its directions swapped, at
   the same point of their code; nothing is sent.  0; EINVAL (a table, span, gate or done word the region does not hold,
   a span longer than a piece, a step past the period); ENOLINK where the order is not the one opened (or was restarted
   here: a program of a client gone held pieces outstanding); EBUSY where another client's declaration toward the node
   held it past MESH_PLATFORM_NS or the programs going on are not at one round boundary; ENOSPC where the node's programs or segments are all in
   use */
int mesh_declare(struct mesh_ctx *,const struct mesh_order *,const struct mesh_declaration *,struct mesh_run *);
/* the client's program ended at the steps it issued (a whole number of executions): the node's next segment without it
   (nothing where it ended already).  0, or mesh_declare's refusals */
int mesh_close(struct mesh_ctx *,struct mesh_run *);
/* `steps` steps of the program issued (monotone): the whole of a client's publication */
static inline void mesh_issue(const struct mesh_run *run,uint64_t steps){ atomic_store_explicit(run->issued,steps,memory_order_release); }
/* whether a closed program is wholly done (every done word of its pieces raised): its tables and bytes free of the bridge */
int mesh_run_done(struct mesh_ctx *,const struct mesh_run *);
#endif

#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* An explicit link map, one small text file:
     <kind> <nodes>             kind is mesh, ring or tree
     <a> <b>                    one link per line; '#' starts a comment line
   ring: b follows a around the ring.  tree: a is b's parent; the root is nobody's child.  mesh: every pair is linked.
   The kind picks the all-reduce: mesh -> direct exchange, ring -> reduce-scatter + all-gather,
   tree (a star is a tree) -> reduce to the root + broadcast.  Nothing is inferred from the links. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE };
#define MESH_LINK_MAP_NODES 32
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t link[MESH_LINK_MAP_NODES*(MESH_LINK_MAP_NODES-1)/2][2]; };
int mesh_link_map_read(const char *path,struct mesh_link_map *);

/* A typed operand: `type` is the caller's own element-type code, carried through untouched. */
struct mesh_operand { uint32_t type,element_bytes; uint64_t elements; };
/* One step of one rank's all-reduce, in that rank's dependency order.
   SEND   publishes operand elements [first, first+piece.elements) to peer.
   REDUCE receives that typed piece from peer into the next caller-supplied received section;
          the caller's supplied reduction adds it into the operand at `first` before any later
          step that reads those elements.  In place is safe on a ring or tree: whatever a REDUCE
          or COPY brings in was caused by the arrival of every earlier SEND of those elements.
          On a mesh (direct exchange) it is not, so the sum goes to the caller's result.
   COPY   receives that typed piece from peer directly into the operand at `first`.
   round  is the step's position in the algorithm; sender and receiver agree on it. */
enum { MESH_STEP_SEND, MESH_STEP_REDUCE, MESH_STEP_COPY };
struct mesh_step { uint32_t op,peer,round,padding; uint64_t first; struct mesh_operand piece; };
#define MESH_ALLREDUCE_STEPS(nodes) (4*(nodes))

/* The rank's steps for one all-reduce of `operand` over `map`; returns the step count. */
uint32_t mesh_allreduce_plan(const struct mesh_link_map *,uint32_t rank,struct mesh_operand,struct mesh_step *steps);
/* Binds every step onto the existing SEND/RECV transport (mesh_transfer_bind).  `operand` is the
   whole operand's section; `received` holds one section per REDUCE step, in step order, each
   sized for that step's piece.  identity+round is the transfer identity on both ends; a ring
   uses identities [identity, identity+2*(nodes-1)), a tree two, a mesh one. */
int mesh_allreduce_bind(struct mesh_ctx *,const struct mesh_step *steps,uint32_t count,uint32_t identity,
  struct mesh_section operand,const struct mesh_section *received,uint32_t invocations,uint32_t invocation_pages);
#endif

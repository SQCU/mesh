#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

#include "mesh-plan.h"

/* Binds a plan's steps onto the SEND/RECV transport (mesh_transfer_bind).  `operand` is the
   whole operand's section; `received` holds one section per REDUCE step, in step order, each
   sized for that step's piece.  identity+round is the transfer identity on both ends.  `pieces`,
   when given, receives every step's bound section in step order: what a SEND publishes and where a
   REDUCE or COPY arrives. */
int mesh_collective_bind(struct mesh_ctx *,const struct mesh_step *steps,uint32_t count,uint32_t identity,
  struct mesh_section operand,const struct mesh_section *received,uint32_t invocations,uint32_t invocation_pages,
  struct mesh_section *pieces);

/* A host (CPU) program's side of the prepared transfers, as mesh-metal.m is a GPU program's.
   mesh_host_inputs gives every received chunk one completion word per invocation (M07) and the
   link's cancellation range (M12); call it between mesh_transfers_prepare and mesh_transfers_start.
   mesh_host_publish releases invocation t of a bound SEND section.  mesh_host_arrived acquires
   invocation t of a bound receive section: below 1 + c until it lands, then 1 + c, or UINT64_MAX once
   its link was cancelled, c the cycle t / N of a cyclic program (mesh_transfers_cyclic; 0 otherwise). */
int mesh_host_inputs(struct mesh_ctx *);
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
#endif

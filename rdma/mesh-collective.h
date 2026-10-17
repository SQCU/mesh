#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

#include "mesh-plan.h"

/* A host (CPU) program's side of the prepared transfers, as mesh-metal.m is a GPU program's.
   mesh_host_inputs gives every received chunk one completion word per invocation (M07) and each destination's
   cancellation range (M12); call it between mesh_transfers_prepare and mesh_transfers_start.
   mesh_host_publish releases invocation t of a bound SEND section.  mesh_host_arrived acquires
   invocation t of a bound receive section: below 1 + c until it lands, then 1 + c, or UINT64_MAX once
   its destination was cancelled, c the cycle t / N of a cyclic program (mesh_transfers_cyclic; 0 otherwise). */
int mesh_host_inputs(struct mesh_ctx *);
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint64_t invocation);
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint32_t invocation);
#endif

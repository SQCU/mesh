#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* the client's start, once mesh_plan has written this node's hops (design/page-queue-change-request.md, in place of
   the configured latch and its unbounded spin; defined in mesh-dataflow.c beside the map reader it reads): for every port whose cable joins this node to a node some hop names, at the plan's
   map serial, a wait, bounded by MESH_PLATFORM_NS from its start, until the port's data path has loaded the plan
   (struct mesh_port_info `prepared` == hdr `plan`); 0 (also where no hop is written), that port's `code` where its
   data path refused the plan (phase MESH_STOPPED), ETIMEDOUT past the bound, ESTALE where the map moved since the plan
   was made (plan again), or the map reader's error value (mesh-dataflow.h mesh_nodes_read) */
int mesh_transfers_start(struct mesh_ctx *);

/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M09 */
/* wire selects the registered window; everything a peer never reads belongs outside it */
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,int wire,struct mesh_section *);
/* design/algorithm-sources.md#programtensor */
int mesh_section_slice(struct mesh_ctx *,struct mesh_section,size_t offset,size_t bytes,uint32_t invocations,uint32_t invocation_pages,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);

#endif

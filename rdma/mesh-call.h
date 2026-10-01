#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t queue,int receive,uint32_t identity,struct mesh_section,uint32_t invocation_pages);
int mesh_transfers_prepare(struct mesh_ctx *,uint32_t slots,uint32_t invocations,uint32_t depth);
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

#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride,channel; };

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t queue,int receive,uint32_t identity,struct mesh_section,uint32_t invocation_pages);
int mesh_transfers_prepare(struct mesh_ctx *,uint32_t slots,uint32_t invocations);
int mesh_transfers_start(struct mesh_ctx *);

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,uint32_t channel,struct mesh_section *);
uint32_t mesh_row_page(struct mesh_ctx *,uint32_t row,uint32_t chunk);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);

#endif

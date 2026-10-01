#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t queue,int receive,uint32_t identity,struct mesh_section,uint32_t invocation_pages);
int mesh_transfers_prepare(struct mesh_ctx *,uint32_t slots,uint32_t invocations,uint32_t depth);
int mesh_transfers_start(struct mesh_ctx *);
/* design/prepared-machine.md#M01 */
int mesh_transfers_bank(struct mesh_ctx *,struct mesh_ctx *other);
/* design/prepared-machine.md#M12 */
int mesh_transfers_stop(struct mesh_ctx *);

/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M09 */
/* wire selects the registered window; everything a peer never reads belongs outside it */
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,int wire,struct mesh_section *);
int mesh_section_aligned(struct mesh_ctx *,size_t bytes,uint32_t align,int wire,struct mesh_section *);
int mesh_section_at(struct mesh_ctx *,uint32_t page,size_t bytes,int wire,struct mesh_section *);
/* design/algorithm-sources.md#programtensor */
int mesh_section_slice(struct mesh_ctx *,struct mesh_section,size_t offset,size_t bytes,uint32_t invocations,uint32_t invocation_pages,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* mesh.h mesh_requests: a ring of `entries` (a power of two) stripe requests in the process's memory, registered with
   the bridge (deregistered at its detach); a GPU writes requests into a ring of its own by the protocol mesh.h states.
   mesh_write: a host thread's request into a ring (any number of threads at once): 0, or EAGAIN where the ring holds
   `entries` requests the bridge has not taken (not yet: write it again). */
int mesh_requests_create(struct mesh_ctx *,uint32_t entries,struct mesh_requests **);
int mesh_write(struct mesh_requests *,uint32_t link,uint64_t offset,uint64_t bytes,uint64_t word,uint64_t value);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);

#endif

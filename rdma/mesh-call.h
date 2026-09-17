#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_calls;
struct mesh_call;
struct mesh_function;
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride,channel; };
struct mesh_operand {
  size_t bytes;
  struct mesh_page_entry *pages; struct mesh_publication *publication; uint32_t index;
  uint32_t *sequence; size_t quantum;
};
_Static_assert(sizeof(struct mesh_operand)==48,"mesh_operand");
/* design/algorithm-sources.md#programtensor */
static inline __attribute__((always_inline)) void *mesh_operand_data(const struct mesh_page_entry *pages){return (void *)atomic_load_explicit(&pages->address,memory_order_relaxed);}
/* design/algorithm-sources.md#programtensor */
static inline void *mesh_operand_address(struct mesh_operand operand,size_t offset){
  size_t block=offset/operand.quantum;
  return (char *)mesh_operand_data(operand.pages+block)+(offset-block*operand.quantum);
}
/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_operand_page(struct mesh_operand operand){return atomic_load_explicit(&operand.pages->mapping,memory_order_relaxed);}
/* design/algorithm-sources.md#programtensor */
static inline __attribute__((always_inline)) intptr_t mesh_operand_view(const struct mesh_page_entry *pages){return (intptr_t)(atomic_load_explicit(&pages->mapping,memory_order_relaxed)>>32);}
typedef void (*mesh_dispose)(void *);

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *,uint32_t workers,uint32_t count,void *owner,mesh_dispose);
struct mesh_function *mesh_call_bind(struct mesh_calls *,uint32_t worker,
  const struct mesh_section *inputs,const struct mesh_section *views,size_t input_count,
  const struct mesh_section *outputs,size_t output_count,const void *submit,void *context,const void *rearm,void *rearm_context);
int mesh_calls_start(struct mesh_calls *);
uint64_t mesh_calls_submit(struct mesh_calls *,uint32_t index);
/* design/algorithm-sources.md#meshresult */
uint64_t mesh_calls_result(struct mesh_calls *,uint32_t index);
void mesh_call_complete(struct mesh_call *,int error);
void mesh_call_fail(struct mesh_call *,int error);
void mesh_calls_destroy(struct mesh_calls *);

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t queue,int receive,uint32_t identity,struct mesh_section);
int mesh_transfers_prepare(struct mesh_ctx *);
void mesh_transfers_start(struct mesh_ctx *);

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,uint32_t channel,struct mesh_section *);
uint32_t mesh_row_page(struct mesh_ctx *,uint32_t row,uint32_t chunk);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);
void mesh_section_release(struct mesh_ctx *,struct mesh_section);

/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *,const struct mesh_section *,size_t count,uint32_t index);
#endif

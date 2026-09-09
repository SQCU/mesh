#ifndef MESH_METAL_EXECUTOR_H
#define MESH_METAL_EXECUTOR_H
#include "mesh-metal.h"
#include "mesh-functions.h"
struct mesh_metal_dependency { uint64_t value; size_t argument; int publish; };
size_t mesh_metal_row_chunk(struct mesh_ctx *context, struct mesh_scope scope, size_t row_bytes, size_t alignment);
int mesh_metal_bind_rows(const struct mesh_view *view, struct mesh_metal_layout layout,
  size_t rows, size_t row_bytes, size_t alignment, struct mesh_metal_rows *result);
id<MTLBuffer> mesh_metal_alias(id<MTLDevice> device, const struct mesh_view *view, struct mesh_metal_layout *layout);
id<MTLBuffer> mesh_metal_indices(id<MTLDevice> device, const mesh_call *call);
NSString *mesh_metal_source(void);
void mesh_metal_publish_cpu(uint32_t *generation, uint32_t value);
void mesh_metal_completion(id<MTLCommandBuffer> command, void (^complete)(int));
ptrdiff_t mesh_metal_advance(id<MTLSharedEvent> event, mesh_call *call, uint64_t base,
  const struct mesh_metal_dependency *dependencies, size_t count);
int mesh_metal_acquire(id<MTLSharedEvent> event, mesh_call *call, size_t argument, uint64_t value);

struct mesh_metal_reduction {
  mesh_call *call;
  size_t argument, source, elements, padding;
  struct mesh_selection selection;
  __unsafe_unretained id<MTLSharedEvent> event;
  uint64_t value;
};
struct mesh_metal_span { size_t edge; const uint32_t *indices; size_t count; };
struct mesh_metal_reduce_policy { uint32_t element_bytes, accumulator_bytes; int cancel_on_error; };
ptrdiff_t mesh_metal_select(struct mesh_metal_reduction *edges, size_t count,
  struct mesh_metal_reduce_policy policy, struct mesh_metal_span *work, size_t *work_count);
ptrdiff_t mesh_metal_reduce(struct mesh_metal_reduction *edges, size_t count,
  struct mesh_metal_reduce_policy policy, struct mesh_metal_span *work, size_t *work_count);
int mesh_metal_gather(mesh_executor *executor, struct mesh_metal_reduction *edges, size_t count,
  struct mesh_metal_reduce_policy policy, struct mesh_metal_span *work);
#endif

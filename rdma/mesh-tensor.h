#ifndef MESH_TENSOR_H
#define MESH_TENSOR_H
#include "mesh-dataflow.h"
struct mesh_tensor_view {
  uint64_t offset, size, shape[8], stride[8];
  uint32_t dtype, rank, first, physical, pages, page_bytes;
};
struct mesh_tensor_command {
  uint32_t kernel, argument_offset;
  uint32_t grid[3], group[3];
};
void *mesh_tensor_create(const char *source);
const char *mesh_tensor_error(void *program);
int mesh_tensor_kernel(void *program, const char *name);
int mesh_tensor_reserve(void *program, struct mesh_rows *pages,
  const struct mesh_tensor_view *views, size_t count,
  const uint64_t *dimensions, size_t rank,
  const uint32_t *arguments, size_t argument_count);
void *mesh_tensor_data(void *program, uint32_t tensor);
int mesh_tensor_function(void *program, uint32_t index, uint32_t identifier,
  const struct mesh_row_function *function, struct mesh_row_map metadata,
  const struct mesh_tensor_command *commands, size_t count);
void mesh_tensor_submit(void *program, uint32_t function, uint64_t stamp, uint32_t indices);
void mesh_tensor_free(void *program);
#endif

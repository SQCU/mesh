#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"

#define MESH_ROW_ABSENT UINT32_MAX
#define MESH_ROW_WRITING (UINT64_C(1)<<63)

struct mesh_row { uint32_t page, uses; uint64_t stamp; };
struct mesh_rows {
  struct hdr *memory; struct mesh_row *table; size_t count; uint32_t offset, bytes;
  struct mesh_ctx *context; uint64_t identity;
  const struct mesh_row_binding *bindings; size_t binding_count;
  const struct mesh_row_function *functions; size_t function_count;
  const struct mesh_row_map *returns; size_t return_count;
};
struct mesh_row_range { uint32_t first, count; };
struct mesh_row_map { uint32_t first, count, stride, physical, physical_stride, immutable; const uint32_t *uses; const struct mesh_row_range *ranges; };
struct mesh_row_function {
  struct mesh_row_map *input, *output;
  uint32_t inputs, outputs, rows;
  struct mesh_row_map indices;
};
struct mesh_row_binding {
  uint32_t first, count, remote, headers;
  const uint32_t *uses;
  uint16_t peer, receive;
  uint64_t remote_table;
  struct mesh_row_map *input;
  uint32_t inputs;
};
struct mesh_row_metadata {
  uint64_t stamp, when;
  uint32_t function, index, peer;
  int64_t code;
  uint32_t domain, reserved;
};

struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *context, size_t index);

struct mesh_rows *mesh_rows_create(struct mesh_ctx *context, size_t rows, uint64_t identity);
uint32_t mesh_rows_allocate(struct mesh_rows *pages, size_t count, size_t alignment);
void mesh_rows_map(struct mesh_rows *pages, uint32_t first, uint32_t physical,
  uint32_t count, uint32_t uses, uint64_t stamp);
int mesh_rows_invalidate(struct mesh_rows **pages, const struct mesh_row_map *held, size_t count);
int mesh_rows_close(struct mesh_ctx *context);
size_t mesh_rows_poll(struct mesh_ctx *context);

uint32_t mesh_rows_issue(const struct mesh_rows *pages, const struct mesh_row_function *function,
  uint64_t stamp);
void mesh_rows_complete(const struct mesh_rows *pages, const struct mesh_row_function *function,
  uint64_t stamp, uint32_t indices);

int mesh_rows_realize(const struct mesh_rows *pages, const struct mesh_row_function *functions,
  size_t count, struct mesh_row_binding *bindings, size_t binding_count,
  struct mesh_row_map *returns, size_t return_count);
void *mesh_row_data(const struct mesh_rows *pages, uint32_t row);
int mesh_rows_present(const struct mesh_rows *pages, struct mesh_row_map map, uint32_t index, uint64_t stamp);
void mesh_rows_publish(const struct mesh_rows *pages, const struct mesh_row_function *function,
  uint32_t index, uint64_t stamp);
void mesh_rows_report(const struct mesh_rows *pages, struct mesh_row_map output,
  uint32_t occurrence, struct mesh_row_metadata metadata);
void mesh_row_release(const struct mesh_rows *pages, uint32_t row);
#endif

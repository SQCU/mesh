#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"

#define MESH_ROW_ABSENT UINT32_MAX
#define MESH_ROW_WRITING (UINT64_C(1)<<63)

struct mesh_row { uint32_t page, uses; uint64_t stamp; };
struct mesh_rows { struct hdr *memory; struct mesh_row *table; size_t count; uint32_t offset, bytes; };
struct mesh_row_map { uint32_t first, count, stride; };
struct mesh_row_function {
  const struct mesh_row_map *input, *output;
  uint32_t inputs, outputs, rows;
};
struct mesh_row_binding { uint32_t first, count, remote; uint16_t peer, receive; };
struct mesh_row_address { uint64_t epoch, stamp; uint32_t source, target; };

int mesh_rows_validate(const struct mesh_rows *pages, const struct mesh_row_function *function);
void *mesh_row_data(const struct mesh_rows *pages, uint32_t row);
int mesh_rows_present(const struct mesh_rows *pages, struct mesh_row_map map, uint32_t index, uint64_t stamp);
size_t mesh_rows_select(const struct mesh_rows *pages, const struct mesh_row_function *function,
  uint64_t stamp, uint32_t *indices, size_t capacity);
int mesh_rows_publish(const struct mesh_rows *pages, const struct mesh_row_function *function,
  uint32_t index, uint64_t stamp);
int mesh_row_release(const struct mesh_rows *pages, uint32_t row, uint64_t stamp);
int mesh_row_zero(const struct mesh_rows *pages, uint32_t row, uint64_t stamp);
int mesh_rows_add_f16(const struct mesh_rows *pages, const uint32_t *inputs, size_t count,
  const uint32_t *accumulators, size_t elements, uint32_t index_row, uint32_t index,
  uint64_t stamp);
int mesh_rows_indexed(const struct mesh_rows *pages, uint32_t index_row, uint32_t first,
  uint32_t count, uint64_t stamp);
int mesh_rows_normalize_f32(const struct mesh_rows *pages, const uint32_t *accumulators,
  const uint32_t *gamma, const uint32_t *residual, const uint32_t *outputs,
  size_t elements, float epsilon, float scale);
size_t mesh_rows_send(const struct mesh_rows *pages, uint64_t epoch,
  const struct mesh_row_binding *bindings, size_t count);
int mesh_rows_receive(const struct mesh_rows *pages, uint64_t epoch,
  const struct mesh_row_binding *bindings, size_t count);
size_t mesh_rows_acknowledge(const struct mesh_rows *pages, uint64_t epoch);
int mesh_rows_return(const struct mesh_rows *pages, uint32_t row, uint64_t stamp);
int mesh_rows_digest(const struct mesh_rows *pages, uint32_t input, uint32_t output,
  uint32_t index, uint64_t seed);
int mesh_rows_equal(const struct mesh_rows *pages, uint32_t first, uint32_t second,
  uint64_t stamp);
#endif

#ifndef MESH_TENSOR_H
#define MESH_TENSOR_H
#include <stddef.h>
#include <stdint.h>
struct mesh_tensor_view {
  uint64_t offset, size, shape[8], stride[8];
  uint32_t dtype, rank, pool, reserved;
  uint64_t page_start, origin;
  uint32_t page_stride, payload;
};
struct mesh_tensor_command {
  uint32_t kernel, argument_offset;
  uint32_t grid[3], group[3];
};
struct mesh_tensor_span { uint64_t stream, offset, bytes, target; };
struct mesh_tensor_binding { uint64_t stream, offset; uint32_t tensor, padding; };
void *mesh_tensor_create(const char *source);
const char *mesh_tensor_error(void *program);
int mesh_tensor_kernel(void *program, const char *name);
int mesh_tensor_reserve(void *program, size_t bytes, const struct mesh_tensor_view *views,
                        size_t count, const uint64_t *dimensions, size_t rank,
                        const uint32_t *arguments, size_t argument_count);
void *mesh_tensor_data(void *program);
void *mesh_tensor_memory(void *program);
void mesh_tensor_memory_free(void *memory);
int mesh_tensor_phase(void *program, uint32_t phase, const struct mesh_tensor_command *commands, size_t count);
int mesh_tensor_submit(void *program, uint32_t phase);
int mesh_tensor_status(void *program);
void mesh_tensor_free(void *program);
void *mesh_tensor_transport(void *context);
size_t mesh_tensor_transport_bytes(void *transport);
int mesh_tensor_transport_progress(void *transport);
int mesh_tensor_transport_free(void *transport);
void *mesh_tensor_channel(void *transport, void *program, int peer, int receive,
                          uint32_t channel, uint64_t epoch, const unsigned char plan[32], size_t bytes);
int mesh_tensor_transfer(void *channel, uint64_t epoch, uint64_t offset, uint64_t bytes);
int mesh_tensor_transfer_plan(void *channel, const struct mesh_tensor_span *spans, size_t count,
                              const struct mesh_tensor_binding *bindings, size_t binding_count);
int mesh_tensor_transfer_release(void *channel);
int mesh_tensor_transfer_commit(void *channel);
int mesh_tensor_transfer_status(void *channel);
void mesh_tensor_transfer_cancel(void *channel, int error);
int mesh_tensor_channel_free(void *channel);
#endif

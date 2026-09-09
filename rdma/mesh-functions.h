#ifndef MESH_FUNCTIONS_H
#define MESH_FUNCTIONS_H
#include "mesh.h"

struct mesh_extent { size_t offset, bytes; uint32_t channel; int peer, receive; uint32_t chunk; };
struct mesh_view {
  unsigned char *base;
  const uint32_t *pages;
  const struct mesh_epoch *epoch;
  size_t offset, bytes;
  uint32_t stride, payload, capacity;
};
typedef int (*mesh_index_fn)(void *, size_t, struct mesh_extent *);
typedef struct mesh_function mesh_function;
typedef struct mesh_executor mesh_executor;
typedef struct mesh_call mesh_call;

mesh_function *mesh_compile(size_t count, mesh_index_fn index, void *capture);
const struct mesh_extent *mesh_function_extent(const mesh_function *f, size_t index);
size_t mesh_function_count(const mesh_function *f);
void mesh_function_free(mesh_function *f);
struct mesh_executor_policy { uint64_t open_retry_ns; };
mesh_executor *mesh_executor_create(struct mesh_ctx *context, size_t window_pages);
mesh_executor *mesh_executor_create_with(struct mesh_ctx *context, size_t window_pages, struct mesh_executor_policy policy);
size_t mesh_call_unsent(const mesh_call *call);
mesh_call *mesh_bind(mesh_executor *e, mesh_function *f, uint32_t channel_base);
mesh_call *mesh_bind_scoped(mesh_executor *e, mesh_function *f, uint32_t channel_base, struct mesh_scope scope);
int mesh_call_rearm(mesh_call *call, struct mesh_epoch epoch);
void mesh_call_retain_transmit(mesh_call *call);
const struct mesh_view *mesh_call_view(const mesh_call *call, size_t index);
const uint32_t *mesh_call_indices(const mesh_call *call, size_t *bytes);
int mesh_progress(mesh_executor *e);
void mesh_request(mesh_call *call, size_t index);
int mesh_acquire(mesh_call *call, size_t index, struct mesh_view *view);
int mesh_acquire_page(mesh_call *call, size_t index, size_t page);
ptrdiff_t mesh_ready_pages(const mesh_call *call, size_t index);
const uint32_t *mesh_arrived_pages(const mesh_call *call, size_t index, size_t *count);
struct mesh_selection { size_t consumed, arrivals, published; };
struct mesh_join_state { uint64_t generation, coverage; };
size_t mesh_join_indices(struct mesh_join_state *state, uint64_t generation, uint64_t required,
  uint64_t contribution, uint32_t offset, const uint32_t *indices, size_t count, uint32_t *ready);
ptrdiff_t mesh_select_arrivals(mesh_call *call, size_t received, size_t produced, struct mesh_selection *selection,
  const uint32_t **selected);
int mesh_call_cycle(mesh_call *call, uint64_t period, uint64_t last);
int mesh_publish_page(mesh_call *call, size_t index, size_t page);
int mesh_page_published(const mesh_call *call, size_t index, size_t page);
ptrdiff_t mesh_flush(mesh_call *call, size_t index);
int mesh_publish(mesh_call *call, size_t index);
int mesh_complete(mesh_call *call, size_t index, int error);
int mesh_call_status(const mesh_call *call);
void mesh_call_cancel(mesh_call *call, int error);
int mesh_call_retire(mesh_call *call);
int mesh_call_abandon(mesh_call *call);
int mesh_executor_free(mesh_executor *e);
void mesh_view_read(const struct mesh_view *v, size_t offset, void *out, size_t bytes);
void mesh_view_write(const struct mesh_view *v, size_t offset, const void *in, size_t bytes);
#endif

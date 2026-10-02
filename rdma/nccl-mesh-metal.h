/* libnccl-mesh's Metal half (nccl-mesh-metal.m): a group's work on the GPU, encoded into the caller's command
   buffer.  Buffers are id<MTLBuffer> handles with byte offsets.  A wait for a landing is a spin on its
   completion word (the engine's mesh_remote_wait), ended by the landing or by the progress thread cancelling
   a silent link (the word ~0, mesh_cancel), as metal-microbench mesh_rank.m's await_window does. */
#ifndef NCCL_MESH_METAL_H
#define NCCL_MESH_METAL_H
#include <stddef.h>
#include <stdint.h>

enum { METAL_PLAIN, METAL_LAND, METAL_SEND };
struct metal_program;

void *metal_device(void);
void *metal_wrap(void *address, size_t bytes);
void *metal_scratch(size_t bytes);
void *metal_event(void);
void metal_signal(void *event, uint64_t value);
uint64_t metal_signaled(void *event);
void metal_release(void *object);

struct metal_program *metal_begin(void *command_buffer);
int metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);
int metal_combine(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n);
int metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar);
int metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor);
int metal_publish(struct metal_program *, void *cells, size_t offset, uint64_t argument);
int metal_spin(struct metal_program *, void *words, size_t offset);
void metal_wait(struct metal_program *, void *event, uint64_t value);
void metal_keep(struct metal_program *, void *object);
void metal_end(struct metal_program *, void *event, uint64_t value);
void metal_collect(void *event);
const char *metal_error(void);
#endif

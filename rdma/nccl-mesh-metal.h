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

struct metal_program *metal_begin(void *command_buffer, void *encoder);
int metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);
int metal_combine(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n);
int metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar);
int metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor);
/* positions [first, first + count) of a channel whose landing a wait covers, checked once it has landed (unless its
   word was cancelled): bit c of `mask` set where this rank receives a piece at first + c, whose slot must not hold
   that position's filler stamp, clear where it receives nothing, whose slot must; a slot is ring + (t % depth) x slot.
   The first position found otherwise is written to violation[0] (1 + it) and violation[1] (channel x 2 + its bit),
   where violation[0] is still 0. */
struct metal_check { void *ring, *violation; size_t slot; uint64_t first; uint32_t count, mask, channel, depth; };
uint64_t metal_stamp(uint64_t position);
int metal_publish(struct metal_program *, void *cells, size_t offset, uint64_t argument);
/* an empty position's publication: its slot's first eight bytes the position's stamp, then its cell released */
int metal_publish_filler(struct metal_program *, void *cells, size_t offset, uint64_t argument, void *slot, size_t slot_offset,
                         uint64_t position);
int metal_spin(struct metal_program *, void *words, size_t offset, uint64_t expected, const struct metal_check *check);
/* a block's piece in one kernel: copied into its slot and its cell released (send), or awaited and combined (an
   ncclDataType_t, op as combine_into) or copied (type -1, n bytes) where it landed (land) */
int metal_send_small(struct metal_program *, void *slot, size_t slot_offset, void *src, size_t src_offset, size_t bytes,
                     void *cells, size_t cell_offset, uint64_t value);
int metal_land(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
               void *words, size_t word_offset, uint64_t expected, const struct metal_check *check);
void metal_wait(struct metal_program *, void *event, uint64_t value);
void metal_keep(struct metal_program *, void *object);
void metal_end(struct metal_program *, void *event, uint64_t value);
void metal_collect(void *event);
const char *metal_error(void);
#endif

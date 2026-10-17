/* libnccl-mesh's Metal half (nccl-mesh-metal.m): a group's work on the GPU, encoded into the caller's command
   buffer.  Buffers are id<MTLBuffer> handles with byte offsets.  A wait for a landing is rdma/mesh.metal's
   mesh_remote_wait on its completion word at the position's cycle, ended by the landing or by the session's canceller
   (every word a dispatch waits on is the session's, in its cancellation ranges: the progress thread cancels a link
   that lands nothing for mesh.h MESH_SILENT_NS, a lost link's bridge and a revoke cancel theirs; the word ~0,
   mesh_cancel), and, where its program is bounded (metal_begin's polls), by the bound. */
#ifndef NCCL_MESH_METAL_H
#define NCCL_MESH_METAL_H
#include <stddef.h>
#include <stdint.h>

enum { METAL_PLAIN, METAL_LAND, METAL_SEND };
struct metal_program;

void *metal_device(void);
/* the default device's argument records encoded and the ring's capacity (nccl.h ncclMeshMetal4Records) */
void metal_records(uint64_t *issued, uint64_t *capacity);
void *metal_wrap(void *address, size_t bytes);
void *metal_scratch(size_t bytes);
void *metal_event(void);
uint64_t metal_signaled(void *event);
/* 1 once `event` reaches `value`, 0 where `ns` pass first */
int metal_wait(void *event, uint64_t value, uint64_t ns);
void metal_release(void *object);

/* A program's waits bounded where `polls` is set and a dispatch has violation words to record in (the session's,
   `violation`, or those a wait's metal_check names): at most `polls` loads of a completion word (rdma/mesh.metal mesh_wait_bounded), past which the dispatch stores the position, its
   channel and the bound in violation words [2], [3] and [4] (where [2] is still 0) and every later dispatch of the
   program reads them and does nothing, so the host finds the failure there after the command buffer (nccl-mesh.c
   violated) and the call fails as a value.  The bound is the caller's, carried per crossing (nccl.h ncclMeshStream
   polls), not a constant of the library's.  A load's period, the time a bound stands for, is the node's GPU's: on the
   M4 Pro (the Mini) 251 to 265 ns once the GPU's clock has risen, about 1.0 to 1.1 us while it rises (an idle GPU's
   first ~4 ms of work), in either of the library's wait shapes (rdma/NCCL.md "Bounded waits"; metal-microbench
   output_data/library-20261015/round3, 2026-10-16); the M5 Max's is not yet read.  polls 0: unbounded, as the classic
   form's waits have always been. */
struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls);
/* The Metal 4 form: the kernels encoded into the caller's id<MTL4ComputeCommandEncoder> (left open, nothing committed,
   nothing signalled), on pipelines of the residency set's device, every binding through the caller's id<MTL4ArgumentTable>
   (each dispatch's arguments a record of the library's argument ring, never setBytes), each dispatch after the one
   before it (an intra-encoder barrier, and one before the first and after the last, so the caller's commands on either
   side are ordered with them), and every buffer of the library's the program touches made resident through the
   caller's id<MTLResidencySet> (added, and the set committed, where it was not).  Its waits bounded by `polls` as
   metal_begin's (nccl-mesh.c gives it at least 1 here).  NULL where memory runs out. */
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls);
int metal_metal4(const struct metal_program *);
/* A buffer of the program's made resident (the Metal 4 form; the classic form needs none). */
void metal_resident(struct metal_program *, void *buffer);
int metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);
int metal_combine(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n);
int metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar);
int metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor);
/* positions [first, first + count) of a channel whose landing a wait covers, checked once it has landed (unless its
   word was cancelled): bit c of `mask` set where this rank receives a piece at first + c, whose slot must not hold
   that position's filler stamp, clear where it receives nothing, whose slot must; a slot is ring + (t % depth) x slot.
   The first position found otherwise is written to violation[0] (1 + it) and violation[1] (channel x 2 + its bit),
   where violation[0] is still 0. */
struct metal_check { void *ring, *violation; size_t slot; uint64_t first; uint32_t count, mask, channel, depth; uint64_t at; };
/* a filler's stamp at position t, the host's and the kernels' one definition */
#define METAL_STAMP(t) (0x6d6573682d66696cUL ^ ((t) * 0x9e3779b97f4a7c15UL))
int metal_publish(struct metal_program *, void *cells, size_t offset, uint64_t argument);
/* an empty position's publication: its slot's first eight bytes the position's stamp, its cell's first SEND `length`
   bytes (mesh.h MESH_SEND_LENGTH), then its cell released */
int metal_publish_filler(struct metal_program *, void *cells, size_t offset, uint64_t argument, void *slot, size_t slot_offset,
                         uint64_t position, size_t length);
int metal_spin(struct metal_program *, void *words, size_t offset, uint64_t cycle, const struct metal_check *check);
/* a block's piece in one kernel: copied into its slot and its cell released (send), or awaited and combined (an
   ncclDataType_t, op as combine_into) or copied (type -1, n bytes) where it landed (land) */
int metal_send_small(struct metal_program *, void *slot, size_t slot_offset, void *src, size_t src_offset, size_t bytes,
                     void *cells, size_t cell_offset, uint64_t value);
int metal_land(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
               void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check);
/* the same two steps run by the caller's kernel (nccl.h ncclMeshFused, the Metal 4 form): each one's record written as
   its dispatch's would be and its six bindings' addresses given in `bound` (0 d, 1 s, 2 the record, 3 a, 4 v,
   5 progress), nothing dispatched; the kernel calls metal_steps' functions (the send: its piece stored into
   d + dst, n bytes, then mesh_step_release on a; the land: mesh_step_await on a, then the piece read from s + src) */
int metal_send_deferred(struct metal_program *, void *slot, size_t slot_offset, size_t bytes, void *cells, size_t cell_offset,
                        uint64_t value, uint64_t bound[6]);
int metal_land_deferred(struct metal_program *, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
                        void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check, uint64_t bound[6]);
/* the steps' Metal text (after rdma/mesh.metal): the argument record and mesh_step_* */
const char *metal_steps(void);
void metal_keep(struct metal_program *, void *object);
void metal_end(struct metal_program *, void *event, uint64_t value);
void metal_collect(void *event);
const char *metal_error(void);
#endif

/* libnccl-mesh's Metal half (nccl-mesh-metal.m): a group's work on the GPU, encoded into the caller's command
   buffer.  Buffers are id<MTLBuffer> handles with byte offsets.  A wait for a landing is rdma/mesh.metal's
   mesh_remote_wait_bounded (nccl-mesh-metal.m mesh_step_wait) on its completion word at the position's cycle, ended by
   the landing, by the session's canceller (every word a dispatch waits on is the session's, in its cancellation
   ranges: a lost or silent link's bridge, the progress thread where the session's bridge is gone or refused its
   program, and a revoke cancel theirs; the word ~0, mesh_cancel), or by its program's bound (metal_begin's polls). */
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
void *metal_retain(void *object);

/* A program's waits bounded by `polls` wherever a dispatch has violation words to record in (the program's,
   `violation` or metal_gate's, or those a wait's metal_check names): at most `polls` loads of a completion word
   (rdma/mesh.metal mesh_wait_bounded), past which the dispatch stores the position, its channel and the bound in
   violation words [2], [3] and [4] (where [2] is still 0) and every later dispatch gated on those words reads them and
   does nothing, so the host finds the failure there after the command buffer (nccl-mesh.c violated) and the call fails
   as a value.  The bound is the caller's, carried per crossing (nccl.h ncclMeshStream polls); 0, or more than nccl.h
   NCCL_MESH_POLLS, is NCCL_MESH_POLLS (design/transport.md; nccl-mesh-metal.m bound_of).  A load's period,
   the time a bound stands for, is the node's GPU's: on the M4 Pro (the Mini) 251 to 265 ns once the GPU's clock has
   risen, about 1.0 to 1.1 us while it rises (an idle GPU's first ~4 ms of work), in either of the library's wait shapes
   (rdma/NCCL.md "Bounded waits"; metal-microbench output_data/library-20261015/round3, 2026-10-16); the M5 Max's is not
   yet read.  A dispatch with no violation words halts on nothing, and its wait makes one load. */
struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls);
/* The Metal 4 form: the kernels encoded into the caller's id<MTL4ComputeCommandEncoder> (left open, nothing committed,
   nothing signalled), on pipelines of the residency set's device, every binding through the caller's id<MTL4ArgumentTable>
   (each dispatch's arguments a record of the library's argument ring, never setBytes), each dispatch after the one
   before it (an intra-encoder barrier, and one before the first and after the last, so the caller's commands on either
   side are ordered with them), and every buffer of the library's the program touches made resident through the
   caller's id<MTLResidencySet> (added, and the set committed, where it was not).  Its waits bounded by `polls` as
   metal_begin's (0: NCCL_MESH_POLLS).  NULL where memory runs out. */
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls);
int metal_metal4(const struct metal_program *);
/* A buffer of the program's made resident (the Metal 4 form; the classic form needs none). */
void metal_resident(struct metal_program *, void *buffer);
/* The program's violation words from here on, `violation` (made resident): those of the session whose group is encoded
   next, which gate every dispatch naming none of its own (nccl-mesh.c complete_group, issue_group, retire), so one
   session's reached bound halts that session's dispatches and no other's. */
void metal_gate(struct metal_program *, void *violation);
int metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);
/* dst = dst op src, n elements of an ncclDataType_t (op as nccl-mesh.c combine_into: 0 sum, 1 product, 2 max, 3 min), or,
   `type` METAL_MERGE, the partials merge (nccl.h ncclMeshRedOpCreatePartials) of n rows of `op` float32 [m, l, O]: a
   row with l > 0 rescaled to the greater m and summed into dst's, one with none counting for none (two such the row
   [the least float, 0, 0]); metal_land's landed piece the same way */
#define METAL_MERGE 12
int metal_combine(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n);
/* dst = with op src, n elements of an ncclDataType_t: a landed slot (src, loaded system-coherent) combined with an operand
   read in place (`with`: a lease's send buffer, nccl.h ncclMeshMemBuffer), the first combine of a range no step wrote */
int metal_combine3(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, void *with,
                   size_t with_offset, size_t n);
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
/* an empty position's publication: its slot's first eight bytes the position's stamp, then its cell released */
int metal_publish_filler(struct metal_program *, void *cells, size_t offset, uint64_t argument, void *slot, size_t slot_offset,
                         uint64_t position);
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
/* A counted all-to-all's steps (nccl.h ncclMeshAlltoAllCounted), every offset of its rows read on the GPU from int64
   counts (a row a count's unit, `row` bytes; a count below 0 none), so no host reads them:
     START   (d the landed counts, s the rank's counts) its own segment's counts, `bytes` of them from entry `first`,
             copied to d + dst, and at d + aux its tail: the rows before that segment, the rows of its `scalar` counts,
             and an overflow word 0;
     CHECK   (d the landed counts) the tail at d + dst's overflow word set where the rows of entries [first, first +
             entries) pass `scalar` (a pair's bound);
     ARRIVED (d the host's word, s the landed counts, a the session's violation words or NULL) the word at d + dst,
             where it still holds 0, released (system scope) to 1 once the landed counts at s + src (`bytes` entries,
             its tail at entry `first`) hold at most `aux` rows, its own at most `scalar` and no overflow, else 2, and
             then violation word [5] set (nccl-mesh.c violated: the session fails as a value);
     SEND    (d a slot, s the rows, a the counts at aux) bytes [scalar, scalar + bytes) of the rows the counts' entries
             [first, first + entries) send, after the rows of the entries before them, stored system-coherent at
             d + dst, within `limit` bytes of s + src;
     LAND    (d the receive buffer, s a landed slot, a the landed counts at aux) the slot's bytes the same way into the
             rows of entries [first, first + entries) after those before them, within `limit` bytes of d + dst;
     OWN     (d the receive buffer, s the rows, a the landed counts at aux) the rank's own rows (entries [first, first
             + entries), the rows before them in s the tail's at entry `scalar`), within `limit` bytes of d + dst and
             `bytes` of s + src.
   SEND, LAND and OWN are one threadgroup (its thread 0 sums the counts). metal_contents: a shared buffer's host address
   (NULL: none). */
enum { METAL_COUNTED_START, METAL_COUNTED_CHECK, METAL_COUNTED_ARRIVED, METAL_COUNTED_SEND, METAL_COUNTED_LAND, METAL_COUNTED_OWN };
struct metal_counted { void *d, *s, *a; size_t dst, src, aux; uint64_t bytes, scalar, first, row, limit; uint32_t entries; };
int metal_counted(struct metal_program *, int kind, const struct metal_counted *);
void *metal_contents(void *buffer);
/* the steps' Metal text (after rdma/mesh.metal): the argument record and mesh_step_* */
const char *metal_steps(void);
void metal_keep(struct metal_program *, void *object);
void metal_end(struct metal_program *, void *event, uint64_t value);
void metal_collect(void *event);
const char *metal_error(void);
#endif

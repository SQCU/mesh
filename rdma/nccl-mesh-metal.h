/* libnccl-mesh's Metal half (nccl-mesh-metal.m): a group's work on the GPU, encoded into the caller's command
   buffer.  Buffers are id<MTLBuffer> handles with byte offsets.  A channel's stream class (mesh-call.h
   mesh_stream_view: its out ring and ready words, its in ring and landed words) is one buffer over the region's pages
   the bridge took for it.  A wait for a landing is rdma/mesh.metal's mesh_wait_bounded (nccl-mesh-metal.m
   mesh_step_wait) on the landed word of its run's last unit at that unit's cycle (the bridge releases a stream's units
   in order), ended by the landing, by a dud's landing (the word's MESH_DUD, mesh.h), by the link's loss (the
   bridge's ~0) or by its program's bound (metal_begin's polls), reading no other word. */
#ifndef NCCL_MESH_METAL_H
#define NCCL_MESH_METAL_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
/* the library's own: none is an ABI target of the libraries it is built into (design/rubrics.md, What a client cannot reach) */
#pragma GCC visibility push(hidden)

enum { METAL_PLAIN, METAL_LAND, METAL_SEND };
struct metal_program;
/* A step's result and its message beside it (design/rubrics.md, Strings are messages): `status` 0, or an errno, and
   with an errno `message`, what failed (a literal, or text the same thread's next failure here writes over), which its
   caller carries (nccl-mesh.c gpu) and reads for no decision; with 0, NULL.  METAL_DONE the step done; metal_failed an
   errno named by its own text. */
struct metal_result { int status; const char *message; };
#define METAL_DONE ((struct metal_result){0, NULL})
static inline struct metal_result metal_failed(int status) { return (struct metal_result){status, strerror(status)}; }

void *metal_device(void);
/* the default device's argument records encoded and the ring's capacity (nccl.h ncclMeshMetal4Records) */
void metal_records(uint64_t *issued, uint64_t *capacity);
void *metal_wrap(void *address, size_t bytes);
void *metal_scratch(size_t bytes);
/* a buffer of `bytes` the host and the GPU both read and write, zeroed (its address metal_contents') */
void *metal_shared(size_t bytes);
void *metal_event(void);
uint64_t metal_signaled(void *event);
/* 1 once `event` reaches `value`, 0 where `ns` pass first */
int metal_wait(void *event, uint64_t value, uint64_t ns);
void metal_release(void *object);
void *metal_retain(void *object);

/* A program's waits bounded by `polls` wherever a dispatch has failure words to record in (the program's, `violation`
   or metal_gate's, or those a wait names): at most `polls` loads of a landed word (rdma/mesh.metal mesh_wait_bounded),
   past which the dispatch records a failure, which the host finds there after the command buffer (nccl-mesh.c
   comm_failure) and the call fails as a value; no kernel halts on the record (a counted all-to-all's end reads it to report its outcome), every later dispatch running as encoded,
   a later wait ending at its own landing or bound.  The failure words: [0] the cause (METAL_BOUND the bound reached,
   METAL_LOST the link's ~0, METAL_REVOKED the bound reached once a revoke set [6], METAL_DUD a dud's landing, its word's
   MESH_DUD: its producer's group failed, design/transport.md section 10; a landing is none, whatever its bytes), [1] the
   claim (its low 32 bits exchanged for 1 by every wait that fails, the one that found 0 recording [0],
   [3], [4] and [2]: nccl-mesh-kernels.h mesh_step_claim), [2] 1 + the position (stored last by the claimant, by the GPU
   alone, which the host reads, as METAL_COUNTED_ARRIVED does), [3] the channel, [4] the bound, [5] a counted all-to-all's overflow,
   and [6] the host's stop (a revoke's), which a failed wait reads to name its cause.  The bound is the caller's, carried per crossing (nccl.h
   ncclMeshStream polls); 0, or more than nccl.h NCCL_MESH_POLLS, is NCCL_MESH_POLLS (design/transport.md;
   nccl-mesh-metal.m bound_of).  A load's period, the time a bound stands for, is the node's GPU's: on the M4 Pro (the
   Mini) 251 to 265 ns once the GPU's clock has risen, about 1.0 to 1.1 us while it rises (an idle GPU's first ~4 ms of
   work), in either of the library's wait shapes (rdma/NCCL.md "Bounded waits"; metal-microbench
   output_data/library-20261015/round3, 2026-10-16); the M5 Max's is not yet read.  A dispatch with no failure words
   records nothing, and its wait makes one load. */
#define METAL_BOUND 1
#define METAL_LOST 2
#define METAL_REVOKED 4
#define METAL_DUD 8
/* The classic form: the kernels encoded into the caller's id<MTLCommandBuffer>, in its compute encoder (`encoder`, left
   open) or one made here, each dispatch's arguments its setBytes.  With `residency` (an id<MTLResidencySet>; NULL: none),
   a fused group's: its deferred steps' records are the device's argument ring's, as the Metal 4 form's are, and the
   ring and every buffer their bindings name are made resident through it (added, and the set committed, where they
   were not: nccl.h ncclMeshStream residencySet), since the caller's kernels reach them by address.  NULL where it cannot
   be made (memory, or that device's kernels or argument ring), *message then what failed (struct metal_result's
   message) */
struct metal_program *metal_begin(void *command_buffer, void *encoder, void *residency, void *violation, uint32_t polls, const char **message);
/* The Metal 4 form: the kernels encoded into the caller's id<MTL4ComputeCommandEncoder> (left open, nothing committed,
   nothing signalled), on pipelines of the residency set's device, every binding through the caller's id<MTL4ArgumentTable>
   (each dispatch's arguments a record of the library's argument ring, never setBytes), each dispatch after the one
   before it (an intra-encoder barrier, and one before the first and after the last, so the caller's commands on either
   side are ordered with them), and every buffer of the library's the program touches made resident through the
   caller's id<MTLResidencySet> (added, and the set committed, where it was not).  Its waits bounded by `polls` as
   metal_begin's (0: NCCL_MESH_POLLS).  NULL where memory runs out or the device's kernels or argument ring cannot be
   made, *message then what failed. */
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls, const char **message);
int metal_metal4(const struct metal_program *);
/* A buffer of the program's made resident through its residency set (the Metal 4 form's, or a classic program's begun with
   one; a classic program without one needs none, its dispatches binding their buffers). */
void metal_resident(struct metal_program *, void *buffer);
/* The program's failure words from here on, `violation` (made resident): those of the communicator whose group is
   encoded next, in which every dispatch naming none of its own records a failed wait (nccl-mesh.c complete,
   issue_group, retire), so one communicator's failure is recorded as its own and no other's. */
void metal_gate(struct metal_program *, void *violation);
struct metal_result metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);
/* dst = dst op src, n elements of an ncclDataType_t (op as nccl-mesh.c combine_into: 0 sum, 1 product, 2 max, 3 min), or,
   `type` METAL_MERGE, the partials merge (nccl.h ncclMeshRedOpCreatePartials) of n rows of `op` float32 [m, l, O]: a
   row with l > 0 rescaled to the greater m and summed into dst's, one with none counting for none (two such the row
   [the least float, 0, 0]); metal_land's landed piece the same way */
#define METAL_MERGE 12
struct metal_result metal_combine(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n);
/* dst = with op src, n elements of an ncclDataType_t: a landed slot (src, loaded system-coherent) combined with an operand
   read in place (`with`: a lease's send buffer, nccl.h ncclMeshMemBuffer), the first combine of a range no step wrote */
struct metal_result metal_combine3(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                   void *with, size_t with_offset, size_t n);
struct metal_result metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      uint64_t scalar);
struct metal_result metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   uint64_t divisor);
/* A wait on a run's landing (design/transport.md section 14): in `buffer` (a channel's stream class), the landed word of the
   run's last unit at `word`, landed once it holds at least 1 + `cycle`, and the run's first unit at `slot` (the landed
   piece); its failure recorded in `violation`'s words at position `at` of channel `channel`. */
struct metal_wait { void *buffer, *violation; size_t word, slot; uint64_t cycle, at; uint32_t channel; };
/* A run's release: in `buffer` (a channel's stream class), `count` ready words from `word`, each stored `value` (1 plus the
   units' cycle) after every store before it, with MESH_DUD where a wait of the communicator whose failure words the
   release names (the program's: metal_gate) has failed (their claim, [1]), so its units go as duds, never as data a
   failed wait's piece made (design/transport.md section 10); its first unit's slot of the out ring at `slot`. */
struct metal_release { void *buffer; size_t word, slot; uint64_t value; uint32_t count; };
/* a run's release, its slots as they are (an empty position's too) */
struct metal_result metal_publish(struct metal_program *, const struct metal_release *);
struct metal_result metal_spin(struct metal_program *, const struct metal_wait *);
/* a piece in one kernel: copied into its run's slot and its words released (send), or awaited and combined (an ncclDataType_t, op as combine_into) or copied (type -1, n bytes)
   where it landed, the wait's slot (land) */
struct metal_result metal_send_small(struct metal_program *, void *src, size_t src_offset, size_t bytes, const struct metal_release *);
struct metal_result metal_land(struct metal_program *, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *);
/* the same two steps run by the caller's kernel (nccl.h ncclMeshFused: the Metal 4 form, or a classic program begun with
   a residency set): each one's record written to the device's argument ring as a Metal 4 dispatch's would be, the
   buffers it names made resident, and its six bindings' addresses given in `bound` (0 d, 1 s, 2 the record, 3 a, 4 v,
   5 progress), nothing dispatched; EINVAL on a program with no ring; the kernel calls metal_steps' functions (the
   send: its piece stored into d + dst, n bytes, then mesh_step_release on a; the land: mesh_step_await on a with s,
   then the piece read from s + src) */
struct metal_result metal_send_deferred(struct metal_program *, size_t bytes, const struct metal_release *, uint64_t bound[6]);
struct metal_result metal_land_deferred(struct metal_program *, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *,
                                        uint64_t bound[6]);
/* A counted all-to-all's steps (nccl.h ncclMeshAlltoAllCounted), every offset of its rows read on the GPU from int64
   counts (a row a count's unit, `row` bytes; a count below 0 none), so no host reads them:
     START   (d the landed counts, s the rank's counts) its own segment's counts, `bytes` of them from entry `first`,
             copied to d + dst, and at d + aux its tail: the rows before that segment, the rows of its `scalar` counts,
             and an overflow word 0;
     CHECK   (d the landed counts) the tail at d + dst's overflow word set where the rows of entries [first, first +
             entries) pass `scalar` (a pair's bound);
     ARRIVED (d the host's word, s the landed counts, a the communicator's failure words or NULL) the word at d + dst,
             where it still holds 0, released (system scope) to 1 once the landed counts at s + src (`bytes` entries,
             its tail at entry `first`) hold at most `aux` rows, its own at most `scalar` and no overflow, else 2, and
             then failure word [5] set (nccl-mesh.c comm_failure: the communicator fails as a value); 2 also where
             the failure words already record a failure ([2]: a wait of the call reached its bound, its counts not
             landed), the call's outcome a value its caller reads, since no kernel halts after a failure;
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
struct metal_result metal_counted(struct metal_program *, int kind, const struct metal_counted *);
void *metal_contents(void *buffer);
/* Tracks (rdma/NCCL.md "Per-channel progress"): a
   completion's work on two or more channels, recorded and then run with every channel progressing on its own, one
   thread group serving every track in turn on the GPU (no track waits on another thread group, which Metal may not
   run at once) and one cooperative loop on the host.  metal_record(program, track, cursor, position, need): the
   steps encoded after it (metal_copy .. metal_counted) become item `position` of cursor `cursor` (METAL_PUBLISH or
   METAL_CONSUME) of track `track`, in place of their dispatches, until the next metal_record; track METAL_NONE ends the
   recording, and the steps after it are dispatched as ever.  An item runs once every metal_depend of it holds (track
   `other` has consumed `count`) and its own `need` does: a publication's, its track has consumed `need` positions (the
   credit); a consumption's, its track's publications have passed position `need` - 1 (0: none; a position whose
   publication reads in place what the consumption writes, nccl-mesh.c sends_what_lands), and the landing its steps
   await came (or its bound passed: the link's ~0 is a landing word too); a consumption's end raises its track's count to
   `position` + 1.  Each cursor runs its items in order, and neither waits on the other but by `need`.  metal_tracks:
   the tracks recorded since the last, track t's count from consumed[t], encoded (one dispatch) or run; its result
   (struct metal_result).  The deferred steps (metal_send_deferred, metal_land_deferred) are never recorded. */
#define METAL_NONE UINT32_MAX
enum { METAL_PUBLISH, METAL_CONSUME };
struct metal_result metal_record(struct metal_program *, uint32_t track, int cursor, uint64_t position, uint64_t need);
struct metal_result metal_depend(struct metal_program *, uint32_t other, uint64_t count);
struct metal_result metal_tracks(struct metal_program *, uint32_t tracks, const uint64_t *consumed);
/* the steps' Metal text (after rdma/mesh.metal): the argument record, a deferred step's bindings as one record
   (mesh_step_binding) and mesh_step_* */
const char *metal_steps(void);
/* rdma/mesh.metal as this library was built with it, then metal_steps' text: what a caller's kernel compiles after */
const char *metal_prelude(void);
void metal_keep(struct metal_program *, void *object);
void metal_end(struct metal_program *, void *event, uint64_t value);
void metal_collect(void *event);
#pragma GCC visibility pop
#endif

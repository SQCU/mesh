/* libnccl-mesh's Metal half (nccl-mesh-metal.m): a group's work on the GPU, encoded into the caller's command
   buffer.  Buffers are id<MTLBuffer> handles with byte offsets.  A channel's holding (mesh-call.h mesh_stream_view: its
   three page counts) is one buffer over the region's page that holds them, and the bytes it sends and lands are the
   client's own pages.  A wait for a landing is rdma/mesh.metal's mesh_wait_bounded (nccl-mesh-kernels.h
   mesh_step_wait) on the holding's `landed` until it covers the entry, and on no other count; it ends at the count, at the link's loss (the bridge's ~0) or at its program's bound (metal_begin's
   polls), and reads no other word.  A publication is one system fence and one store of the holding's `ready`. */
#ifndef NCCL_MESH_METAL_H
#define NCCL_MESH_METAL_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "mesh.h"
#include "nccl.h"
/* the library's own: none is an ABI target of the libraries it is built into (design/rubrics.md, Three kinds of source: the interface table) */
#pragma GCC visibility push(hidden)

/* A count of a holding's pages (mesh.h struct mesh_stream `ready`, `sent` and `landed`; an entry's end): a type of its
   own, so a byte count, an entry or an element count is never added to it, taken from it or compared with it but
   through a named conversion.  pages_of: a piece's bytes as the pages it fills, rounded up, the one conversion from
   bytes.  pages_to_word and pages_from_word: the count as a region word or a kernel's argument record holds it, used
   at those boundaries alone (and in a sum that only tells whether counts moved, nccl-mesh.c run).  pages_lost: the
   bridge's ~0, its link lost. */
typedef struct mesh_pages mesh_pages;
static inline mesh_pages pages_of(uint64_t bytes, uint64_t page) { return (mesh_pages){(bytes + page - 1) / page}; }
static inline mesh_pages pages_add(mesh_pages a, mesh_pages b) { return (mesh_pages){a.n + b.n}; }
static inline mesh_pages pages_sub(mesh_pages a, mesh_pages b) { return (mesh_pages){a.n - b.n}; }
static inline int pages_below(mesh_pages a, mesh_pages b) { return a.n < b.n; }
static inline int pages_lost(mesh_pages a) { return a.n == UINT64_MAX; }
static inline uint64_t pages_to_word(mesh_pages a) { return a.n; }
static inline mesh_pages pages_from_word(uint64_t word) { return (mesh_pages){word}; }

/* a copy's loads: plain, or coherent (METAL_LAND: bytes the NIC may have landed in the same command buffer) */
enum { METAL_PLAIN, METAL_LAND };
/* Fixed encoding operands live in the caller's frame. Objective-C sees the
   Metal handle types; the C half only passes the same pointer fields back. */
#ifdef __OBJC__
#import <Metal/Metal.h>
#define METAL_HANDLE(type) type
#else
#define METAL_HANDLE(type) void *
#endif
struct recording;
struct kernels;
struct metal_program {
  METAL_HANDLE(id<MTL4ComputeCommandEncoder>) encoder;
  METAL_HANDLE(id<MTL4ArgumentTable>) table;
  METAL_HANDLE(id<MTLResidencySet>) residency;
  METAL_HANDLE(id<MTLBuffer>) violation;
  int added;
  uint32_t polls;
  struct recording *recording;
  struct kernels *kernels;
  /* metal_order's state: a present being dispatched, and the presents dispatched so far */
  int presenting, presented;
  /* its group's operands declared to the caller's schedule (metal_boundary): no barrier after its pool */
  int declared;
};
#undef METAL_HANDLE
/* A step's result and its message beside it (design/rubrics.md, Strings are messages): `status` 0, or an errno, and
   with an errno `message`, what failed (a literal, or text the same thread's next failure here writes over), which its
   caller carries (nccl-mesh.c gpu) and reads for no decision; with 0, NULL.  METAL_DONE the step done; metal_failed an
   errno named by its own text. */
struct metal_result { int status; const char *message; };
#define METAL_DONE ((struct metal_result){0, NULL})
static inline struct metal_result metal_failed(int status) { return (struct metal_result){status, strerror(status)}; }

void *metal_device(void);
void *metal_wrap(void *address, size_t bytes);
/* bytes of the region the GPU reads once a program has encoded them (nccl-mesh.c, struct piece): metal_carve cuts
   `bytes` (256-aligned) for one use, whose end metal_done names next (each of `n` words at `words` at least `value`;
   a use never named is never done); metal_piece_hold takes a whole piece (*bytes of it) until metal_piece_give.  Under
   the groups' lock but metal_piece_give; 0 or a piece, ENOMEM or NULL where the region cannot meet it */
struct metal_bytes { void *buffer; size_t offset; void *contents; };
int metal_carve(size_t bytes, struct metal_bytes *made);
void metal_done(const _Atomic uint64_t *words, uint32_t n, uint64_t value);
void *metal_piece_hold(struct metal_bytes *made, size_t *bytes);
void metal_piece_give(void *piece);
void *metal_event(void);
uint64_t metal_signaled(void *event);
/* 1 once `event` reaches `value`, 0 where `ns` pass first */
int metal_wait(void *event, uint64_t value, uint64_t ns);
void metal_release(void *object);
void *metal_retain(void *object);
/* A program's waits bounded by `polls` wherever a dispatch has failure words to record in (the program's, `violation`
   or metal_gate's, or those a wait names): at most `polls` loads of a count (rdma/mesh.metal mesh_wait_bounded), past
   which the dispatch records a failure, which the host finds there after the command buffer (nccl-mesh.c
   comm_failure): the bound is the wait's deadline, and its expiry ends the program there (design/data-model.md section
   6), a revoke's or a lost link's being a value; no kernel halts on the record, every later dispatch running as encoded,
   a later wait on those failure words ending at once, and one under way within 2^10 loads (nccl-mesh-kernels.h
   mesh_step_failed), so a dispatch's waits spin about one bound in all, however many waves its threadgroups run in.  The failure words: [0] the cause (METAL_BOUND the bound reached,
   METAL_LOST the link's ~0, METAL_REVOKED the bound reached once a revoke set [6]; a landing is none, whatever its
   bytes), [1] the claim (its low 32 bits exchanged for 1 by every wait that fails, the one that found 0 recording [0],
   [3], [4] and [2]: nccl-mesh-kernels.h mesh_step_claim), [2] 1 + the entry (stored last by the claimant, by the GPU
   alone, which the host reads), [3] the channel, [4] the bound, and [6] the host's stop (a revoke's), which a failed
   wait reads to name its cause.  The bound is the caller's, carried per crossing (nccl.h ncclMeshStream polls); 0, or
   more than nccl.h NCCL_MESH_POLLS, is NCCL_MESH_POLLS (design/transport.md; nccl-mesh-metal.m bound_of).  A load's
   period, the time a bound stands for, is the node's GPU's: on the M4 Pro (the Mini) 251 to 265 ns once the GPU's
   clock has risen, about 1.0 to 1.1 us while it rises (an idle GPU's first ~4 ms of work), in either of the library's
   wait shapes (rdma/NCCL.md "Bounded waits"; metal-microbench output_data/library-20261015/round3, 2026-10-16); the M5
   Max's is not yet read.  A dispatch with no failure words records nothing, and its wait makes one load. */
#define METAL_BOUND 1
#define METAL_LOST 2
#define METAL_REVOKED 4
/* The kernels encoded into the caller's id<MTL4ComputeCommandEncoder, left open for its recorder,
   on pipelines of the residency set's device, every binding through the
   id<MTL4ArgumentTable> (each dispatch's arguments a record of the function's immutable argument pages). Exact payload
   and progress-word accesses enter the caller's prepared dependency frontier; it emits visibility barriers only for
   overlapping dependent uses. Every buffer of the library's the program touches is made resident
   through the id<MTLResidencySet> (added, and the set committed, where it was not).  Its waits bounded by `polls`
   (0: NCCL_MESH_POLLS).  NULL where memory runs out or the device's kernels cannot be made, *message then what
   failed. */
struct metal_program *metal_begin(struct metal_program *, void *encoder, void *table, void *residency, void *violation, uint32_t polls,
                                  const char **message);
/* A buffer of the program's made resident through its residency set. */
void metal_resident(struct metal_program *, void *buffer);
/* The program's failure words from here on, `violation` (made resident): those of the communicator whose group is
   encoded next, in which every dispatch naming none of its own records a failed wait (nccl-mesh.c complete,
   issue_group, retire), so one communicator's failure is recorded as its own and no other's. */
void metal_gate(struct metal_program *, void *violation);
struct metal_result metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);
/* `type` METAL_MERGE: the partials merge (nccl.h ncclMeshRedOpCreatePartials) of rows of `op` float32 [m, l, O], a row
   with l > 0 rescaled to the greater m and summed, one with none counting for none (two such the row [the least float,
   0, 0]); else `type` an ncclDataType_t and op as nccl-mesh.c combine_into (0 sum, 1 product, 2 max, 3 min) */
#define METAL_MERGE 12
/* dst = with op src, n elements of an ncclDataType_t: landed bytes (src, loaded system-coherent) combined with an
   operand read in place (`with`: a lease's send buffer, nccl.h ncclMeshMemBuffer), the first combine of a range no step
   wrote, after a wait (metal_spin) */
struct metal_result metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      size_t element_bytes, uint64_t scalar);
struct metal_result metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   size_t element_bytes, uint64_t divisor);
/* A wait (design/transport.md section 14): in `words` (the Metal buffer over a holding's counts), the count at `word`
   (its `landed`) until it reaches `target` (0: none), in pages; `data` at `data_offset` the landed bytes a landing combines (the group's
   landing area), `bytes` its exact extent (zero for a counter-only observation); its failure recorded in `violation`'s words at entry `at` of channel `channel`; `stride` the pages
   each later repeat of a fused crossing's adds to its target (nccl.h ncclMeshFused repeats; 0 none). */
struct metal_wait { void *words, *violation, *data; size_t word, data_offset; mesh_pages target; uint64_t at; uint32_t channel; mesh_pages stride; size_t bytes; };
/* A publication: in `words`, the count at `word` (a holding's `ready`) stored `value` after one system fence, each
   later repeat of a fused crossing's `stride` pages past it */
struct metal_release { void *words; size_t word; mesh_pages value, stride; };
struct metal_result metal_publish(struct metal_program *, const struct metal_release *);
/* design/zorvek.md zelloth: the sum `out` of `type` and reduction `op` (as metal_land's), each element once, by one
   writer, from `own` and every input's landed element for it, once all have landed, whatever their order; each input
   one landing (its landed data, its channel's `landed` word, its target and packets as a metal_wait's, its landing runs,
   every input's runs over the same elements) and the summed word its forwards name (none: summed NULL), stored past
   each block in order */
struct metal_zelloth_input { void *data; size_t data_offset; void *words; size_t word; void *summed; size_t summed_offset;
                             mesh_pages target, packets; const struct mesh_landing_run *runs; size_t count; };
/* a sum: its result `out`, the rank's own operand `own`, its `inputs` landings */
struct metal_zelloth_sum { void *out; size_t out_offset; void *own; size_t own_offset; uint32_t inputs; const struct metal_zelloth_input *in; };
/* every sum of a group of one `type` and `op` in one dispatch, one threadgroup a core for all of them, after a barrier
   (no present still spinning holds a core one of them needs) */
struct metal_result metal_zelloth(struct metal_program *, int type, int op, size_t element_bytes, uint32_t sums,
                                  const struct metal_zelloth_sum *sum, uint32_t channel, uint64_t at);
/* The same publication as part of a group's presenting (design/zorvek.md zook): after its presents' barrier and with
   none of its own, as a later present takes none */
struct metal_result metal_publish_presented(struct metal_program *, const struct metal_release *);
/* The next dispatch (or recorded tracks pool) reads/writes these prepared page
   ranges. Residency and diagnostic atomic words do not imply data ordering. */
void metal_accesses(struct metal_program *, const ncclMeshAccess *, size_t n);
/* a group's operands, read and written, declared once to its caller's schedule in place of a barrier after its pool */
void metal_boundary(struct metal_program *, const ncclMeshAccess *, size_t n);
/* A communicator's gate word (at `word` in `words`) stored `value` after one system fence, as metal_publish stores: the
   number of the group whose landings it opens (nccl-mesh.c gate_of), a count of groups and no holding's. */
struct metal_result metal_open_gate(struct metal_program *, void *words, size_t word, uint64_t value);
/* the wait alone (a COPY's landing in place), one threadgroup */
struct metal_result metal_spin(struct metal_program *, const struct metal_wait *);
/* Encode the numerical groups as their last required packet arrives. The
   prepared span lengths lower to arithmetic readiness runs (mesh-landing.h),
   including rows split across packets. `with` supplies the other operand,
   whether it is the original input or the existing result. No packet list is
   read by the kernel. `summed` (NULL none), at summed_offset, is the landing channel's count of units summed: each
   prefix of the landing's chunks summed stores there the landed count its units reach (nccl-mesh-kernels.h
   track_summed), so a SEND of the sum names it and goes unit by unit as the sum is made. */
struct mesh_landing_run;
struct metal_result metal_land(struct metal_program *, int type, int op, void *dst, size_t dst_offset,
                               void *with, size_t with_offset,
                               size_t element_bytes, mesh_pages packets, const struct mesh_landing_run *, size_t runs, const struct metal_wait *,
                               void *summed, size_t summed_offset);
/* the same two steps run by the caller's kernel (nccl.h ncclMeshFused): each one's record written to the function's immutable argument pages as a Metal 4 dispatch's would be, the
   buffers it names made resident, and its six bindings' addresses given in `bound` (0 d, 1 s, 2 the record, 3 a, 4 v,
   5 progress), nothing dispatched.  The publication: d its operand (at dst), a the
   holding's counts (mesh_step_release stores `ready`).  The landing: d and s its landing
   buffer (at dst), a the holding's counts (mesh_step_await), v the communicator's failure words. */
struct metal_result metal_send_deferred(struct metal_program *, void *operand, size_t offset, const struct metal_release *, uint64_t bound[6]);
struct metal_result metal_land_deferred(struct metal_program *, void *landing, size_t offset, const struct metal_wait *, uint64_t bound[6]);
/* metal_contents: a shared buffer's host address (NULL: none, a private buffer's) */
void *metal_contents(void *buffer);
/* Tracks (rdma/NCCL.md "Per-channel progress"): a completion's work on its channels, recorded and then run by one
   pool of workers, each channel progressing on its own.
   metal_record(program, track, cursor, position, need): the steps encoded after it (metal_copy .. metal_land) become
   item `position` of cursor `cursor` (METAL_PUBLISH, its send entries, or METAL_CONSUME, its landing entries) of track
   `track`, in place of their dispatches, until the next metal_record; track METAL_NONE ends the recording, and the
   steps after it are dispatched as ever.  An item runs once every metal_depend of it holds (track `other` has consumed
   `count`), and a consumption's once its track's publications have passed send entry `need` - 1 (0: none; a combine
   into bytes a SEND of its channel reads, nccl-mesh.c schedule_laid, unordered with the NIC's read of it); each step's
   chunks once the word it awaits has come (a spin's count; each landing chunk's own pages; the link's ~0 is a count
   too); a consumption's end raises its track's count to `position` + 1.  A publication's `need` is 0: nothing credits
   it.  Each cursor runs its items in order, and neither waits on the other but by `need`.  metal_tracks: the tracks
   recorded since the last, `tracks` of them, track t's count from consumed[t], encoded as one dispatch of workers;
   its result (struct metal_result).  The deferred steps (metal_send_deferred, metal_land_deferred) are never
   recorded. */
#define METAL_NONE UINT32_MAX
enum { METAL_PUBLISH, METAL_CONSUME };
struct metal_result metal_record(struct metal_program *, uint32_t track, int cursor, uint64_t position, uint64_t need);
struct metal_result metal_depend(struct metal_program *, uint32_t other, uint64_t count);
/* the item being recorded, a combine's consumption, follows track `other`'s consumption at `position` chunk by chunk:
   its chunk c once that one has finished its first c + 1 (the two the same elements in the same runs), in place of a
   count of that consumption whole (metal_depend) */
struct metal_result metal_after(struct metal_program *, uint32_t other, uint64_t position);
struct metal_result metal_tracks(struct metal_program *, uint32_t tracks, const uint64_t *consumed);
/* a tracks pool's records as metal_tracks laid them (`records`, its workers' end words at `ended`, `workers` of them),
   read on a failure alone: where some worker has not ended, each open cursor's track and kind, its task, that task's
   chunks claimed and done, its item's position, need and dependencies (each count awaited and the count now) and its
   step's kind and awaited count, into `out`; the characters written, 0 where every worker has ended, none has begun,
   or the words are not that pool's */
size_t metal_tracks_open(const void *records, const void *ended, uint32_t workers, char *out, size_t room);
/* rdma/mesh.metal as this library was built with it, then the steps' Metal text: the argument record, a deferred step's
   bindings as one record (mesh_step_binding) and mesh_step_*; what a caller's kernel compiles after */
const char *metal_prelude(void);
/* Publish a compiled descriptor list. Its only dependency is reuse of slots
   already read by the local bridge; no peer acknowledgement is introduced. */
struct metal_result metal_present(struct metal_program *, void *lists, size_t list_offset,
    void *records, size_t record_offset, size_t count, uint64_t first,
    void *words, size_t taken, size_t presented);
/* Release preparation state and commit residency additions. Submission remains
   owned by the enclosing recorded function. */
void metal_end(struct metal_program *);
/* design/transport.md section 5: an expiry's end of the program, its GPU work released first and awaited within
   MESH_PLATFORM_NS (nccl-mesh.c); and the process's watch, every MESH_PLATFORM_NS / 20, for GPU work that makes no progress, which
   it ends so, started by the first communicator realized or by a caller's first prepared function */
__attribute__((noreturn)) void mesh_library_expired(const char *wait, uint64_t ns);
void mesh_library_watch(void);
#pragma GCC visibility pop
#endif

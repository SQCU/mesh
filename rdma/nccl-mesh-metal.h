/* libnccl-mesh's Metal half (nccl-mesh-metal.m): a group's work on the GPU, encoded into the caller's command
   buffer.  Buffers are id<MTLBuffer> handles with byte offsets.  A channel's holding (mesh-call.h mesh_stream_view: its
   three page counts) is one buffer over the region's page that holds them, and the bytes it sends and lands are the
   client's own pages.  A wait for a landing is rdma/mesh.metal's mesh_wait_bounded (nccl-mesh-kernels.h
   mesh_step_wait) on the holding's `landed` until it covers the entry, and where a write meets bytes a SEND reads, on
   its `sent`; it ends at the count, at the link's loss (the bridge's ~0) or at its program's bound (metal_begin's
   polls), and reads no other word.  A publication is one system fence and one store of the holding's `ready`. */
#ifndef NCCL_MESH_METAL_H
#define NCCL_MESH_METAL_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "mesh.h"
/* the library's own: none is an ABI target of the libraries it is built into (design/rubrics.md, What a client cannot reach) */
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
/* the command buffer a program is encoded into, retained (NULL where it has none: a Metal 4 encoder's) */
void *metal_command(const struct metal_program *);
/* a command buffer's end as its status tells it: 1 completed, -1 ended in an error (the platform's watchdog ends one
   that overruns so), 0 committed and not ended, 2 not committed (nothing runs it until its committer commits it) */
int metal_ended(void *command);

/* A program's waits bounded by `polls` wherever a dispatch has failure words to record in (the program's, `violation`
   or metal_gate's, or those a wait names): at most `polls` loads of a count (rdma/mesh.metal mesh_wait_bounded), past
   which the dispatch records a failure, which the host finds there after the command buffer (nccl-mesh.c
   comm_failure) and the call fails as a value; no kernel halts on the record, every later dispatch running as encoded,
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
/* `type` METAL_MERGE: the partials merge (nccl.h ncclMeshRedOpCreatePartials) of rows of `op` float32 [m, l, O], a row
   with l > 0 rescaled to the greater m and summed, one with none counting for none (two such the row [the least float,
   0, 0]); else `type` an ncclDataType_t and op as nccl-mesh.c combine_into (0 sum, 1 product, 2 max, 3 min) */
#define METAL_MERGE 12
/* dst = with op src, n elements of an ncclDataType_t: landed bytes (src, loaded system-coherent) combined with an
   operand read in place (`with`: a lease's send buffer, nccl.h ncclMeshMemBuffer), the first combine of a range no step
   wrote, after a wait (metal_spin) */
struct metal_result metal_combine3(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                   void *with, size_t with_offset, size_t n);
struct metal_result metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      uint64_t scalar);
struct metal_result metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   uint64_t divisor);
/* A wait (design/transport.md section 14): in `words` (the Metal buffer over a holding's counts), the count at `word`
   (its `landed`) until it reaches `target` (0: none), and the count at `sent_word` (its `sent`) until it reaches
   `sent_target` (0: none), each in pages; `data` at `data_offset` the landed bytes a landing combines (the group's
   landing area); its failure recorded in `violation`'s words at entry `at` of channel `channel`. */
struct metal_wait { void *words, *violation, *data; size_t word, sent_word, data_offset; mesh_pages target, sent_target; uint64_t at; uint32_t channel; };
/* A publication: in `words`, the count at `word` (a holding's `ready`) stored `value` after one system fence; the count
   at `sent_word` (the holding's `sent`) reaches `sent_target` once the NIC is done reading the published pages (a
   deferred publication's record, read by mesh_step_sent; 0 none). */
struct metal_release { void *words; size_t word; mesh_pages value; size_t sent_word; mesh_pages sent_target; };
struct metal_result metal_publish(struct metal_program *, const struct metal_release *);
/* Buffers a dispatch reaches through no binding of its own (a publication's operands, a landing's buffers), declared on
   the classic form's encoder before it, with write use where `write`: Metal orders a pass after the passes that write
   (or, for a write, read) a tracked buffer only where both name it, and the library's counts
   dispatches otherwise name the counts page alone (a publish could run before its operand's bytes were written, or a
   caller's kernel that stores `ready` by address after a later publish).  The Metal 4 form orders its passes by its
   barriers; a recorded program by its replay. */
void metal_uses(struct metal_program *, void *const *buffers, size_t n, int write);
/* The stamp words of a communicator's pool (nccl-mesh.c spare_take, stamp_end): the words of its failure words' page
   from METAL_STAMP on, past their cache line, each holding the number of the last program end stored there, written by
   metal_stamp alone and read by the host.  metal_stamp: `value` stored at `word` in `words` once every dispatch the
   program encoded before it has ended: on the classic form by its command buffer's completed handler (the GPU's end
   of that buffer; nothing waits, and a buffer never committed stamps nothing), on the Metal 4 form by one dispatch
   after the barrier that ends the program's dispatches before it (each dispatch's barrier, device visibility), one
   system-scope fence and one system-coherent store, as metal_publish stores.  Each program's end its own word, so
   ends are seen in whatever order the GPU reaches them.  A release, never a wait. */
#define METAL_STAMP 16
struct metal_result metal_stamp(struct metal_program *, void *words, size_t word, uint64_t value);
/* the wait alone (a COPY's landing in place, a group's end on `sent`), one threadgroup */
struct metal_result metal_spin(struct metal_program *, const struct metal_wait *);
/* the wait, then every threadgroup's piece of the landed bytes combined into dst (an ncclDataType_t or METAL_MERGE, op as
   above), n elements (the merge's rows), each threadgroup awaiting the counts on its own (nccl-mesh-kernels.h
   mesh_step_await) */
struct metal_result metal_land(struct metal_program *, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *);
/* the same two steps run by the caller's kernel (nccl.h ncclMeshFused: the Metal 4 form, or a classic program begun with
   a residency set): each one's record written to the device's argument ring as a Metal 4 dispatch's would be, the
   buffers it names made resident, and its six bindings' addresses given in `bound` (0 d, 1 s, 2 the record, 3 a, 4 v,
   5 progress), nothing dispatched; EINVAL on a program with no ring.  The publication: d its operand (at dst), a the
   holding's counts (mesh_step_release stores `ready`; mesh_step_sent awaits `sent`).  The landing: d and s its landing
   buffer (at dst), a the holding's counts (mesh_step_await), v the communicator's failure words. */
struct metal_result metal_send_deferred(struct metal_program *, void *operand, size_t offset, const struct metal_release *, uint64_t bound[6]);
struct metal_result metal_land_deferred(struct metal_program *, void *landing, size_t offset, const struct metal_wait *, uint64_t bound[6]);
/* metal_contents: a shared buffer's host address (NULL: none, a private buffer's) */
void *metal_contents(void *buffer);
/* Tracks (rdma/NCCL.md "Per-channel progress"): a completion's work on two or more channels, recorded and then run with
   every channel progressing on its own, one thread group a track on the GPU and one cooperative loop on the host.
   metal_record(program, track, cursor, position, need): the steps encoded after it (metal_copy .. metal_land) become
   item `position` of cursor `cursor` (METAL_PUBLISH, its send entries, or METAL_CONSUME, its landing entries) of track
   `track`, in place of their dispatches, until the next metal_record; track METAL_NONE ends the recording, and the
   steps after it are dispatched as ever.  An item runs once every metal_depend of it holds (track `other` has consumed
   `count`), and a consumption's once its track's publications have passed send entry `need` - 1 (0: none; a combine
   into bytes a SEND of its channel reads, nccl-mesh.c meeting) and the landing its first wait awaits came (or its
   bound passed: the link's ~0 is a count too); a consumption's end raises its track's count to `position` + 1.  A
   publication's `need` is 0: nothing credits it.  Each cursor runs its items in order, and neither waits on the other
   but by `need`.  metal_tracks: the tracks recorded since the last, track t's count from consumed[t], encoded (one
   dispatch) or run; its result (struct metal_result).  The deferred steps (metal_send_deferred, metal_land_deferred)
   are never recorded. */
#define METAL_NONE UINT32_MAX
enum { METAL_PUBLISH, METAL_CONSUME };
struct metal_result metal_record(struct metal_program *, uint32_t track, int cursor, uint64_t position, uint64_t need);
struct metal_result metal_depend(struct metal_program *, uint32_t other, uint64_t count);
struct metal_result metal_tracks(struct metal_program *, uint32_t tracks, const uint64_t *consumed);
/* rdma/mesh.metal as this library was built with it, then the steps' Metal text: the argument record, a deferred step's
   bindings as one record (mesh_step_binding) and mesh_step_*; what a caller's kernel compiles after */
const char *metal_prelude(void);
void metal_keep(struct metal_program *, void *object);
void metal_end(struct metal_program *, void *event, uint64_t value);
void metal_collect(void *event);
#pragma GCC visibility pop
#endif

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
#include "mesh-write-compiler.h"
#include "metal-capabilities.h"
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

/* Fixed encoding operands live in the caller's frame. Objective-C sees the
   Metal handle types; the C half only passes the same pointer fields back. */
#ifdef __OBJC__
#import <Metal/Metal.h>
#define METAL_HANDLE(type) type
#else
#define METAL_HANDLE(type) void *
#endif
struct kernels;
struct metal_program {
  METAL_HANDLE(id<MTL4ComputeCommandEncoder>) encoder;
  METAL_HANDLE(id<MTL4ArgumentTable>) table;
  METAL_HANDLE(id<MTLResidencySet>) residency;
  METAL_HANDLE(id<MTLBuffer>) violation;
  int added;
  uint32_t polls, frame;
  struct kernels *kernels;
  /* Preparation-owned exact accesses for the next typed command. */
  void *accesses;
  /* Retained numerical declarations. The shared page-effect pass completes
     their sources before any command is bound to the recorder. */
  void *commands, *transfers, *group_accesses, *loop_accesses, *relation_context;
  void *arguments,*resources,*sources;
  void *coverage_values; /* interned preparation types; released after source specialization */
  ncclMeshSources declaration;
  int administrative;
  const char *cpu_compiler,*cpu_resources;
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
/* Preparation allocates argument pages and retains them with the function. */
struct metal_bytes { void *buffer; size_t offset; void *contents; };
void *metal_piece_hold(struct metal_bytes *made,size_t *bytes,size_t minimum);
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
   comm_failure). The effective load budget is fixed during lowering and starts
   at the first executed observation; exhausting it ends the program, while a
   revoke or lost link is a value; no kernel halts on the record, every later dispatch running as encoded,
   a later wait on those failure words ending at once, and one under way within 2^10 loads (nccl-mesh-kernels.h
   mesh_wait_bounded), so a dispatch's waits spin about one bound in all, however many waves its threadgroups run in.  The failure words: [0] the cause (METAL_BOUND the bound reached,
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
#define METAL_BOUND MESH_OBSERVATION_BOUND
#define METAL_LOST MESH_OBSERVATION_LOST
#define METAL_REVOKED MESH_OBSERVATION_REVOKED
/* The kernels encoded into the caller's id<MTL4ComputeCommandEncoder, left open for its recorder,
   on pipelines of the residency set's device, every binding through the
   id<MTL4ArgumentTable> (each dispatch's arguments a record of the function's immutable argument pages). Exact payload
   and progress-word accesses enter the caller's prepared dependency frontier; it emits visibility barriers only for
   overlapping dependent uses. Every buffer of the library's the program touches is made resident
   through the id<MTLResidencySet> (added, and the set committed, where it was not).  Its waits bounded by `polls`
   (0: NCCL_MESH_POLLS).  NULL where memory runs out or the device's kernels cannot be made, *message then what
   failed. */
struct metal_program *metal_prepare(struct metal_program *,struct isl_ctx *,void *violation,uint32_t polls);
size_t metal_command_count(const struct metal_program *);
const ncclMeshSources *metal_sources(struct metal_program *);
void metal_command_accesses(struct metal_program *,size_t command,const ncclMeshAccess *,size_t count);
struct metal_result metal_specialize(struct metal_program *,void *sources);
struct metal_result metal_record(struct metal_program *,const size_t *,size_t,const ncclMeshStream *);
/* A buffer of the program's made resident through its residency set. */
void metal_resident(struct metal_program *, void *buffer);
/* Local numerical leaves and their selected DMA values enter the same source
   dependency solve as caller numerical leaves. These are preparation operands. */
void metal_page_effect(struct metal_program *, const ncclMeshPageEffect *,
                       void *payload, size_t offset, size_t bytes);
void *metal_page_selection(struct metal_program *,const ncclMeshPageEffect *,
                          void *payload,const ncclMeshSpan *,size_t count);
/* The program's failure words from here on, `violation` (made resident): those of the communicator whose group is
   encoded next, in which every dispatch naming none of its own records a failed wait (nccl-mesh.c complete,
   issue_group, retire), so one communicator's failure is recorded as its own and no other's. */
void metal_gate(struct metal_program *, void *violation);
struct metal_wait;
/* The source compiler attaches observations and publications to this retained
   numerical copy, before it becomes an execution record. */
struct metal_result metal_copy(struct metal_program *,void *dst,size_t dst_offset,void *src,size_t src_offset,size_t bytes);
/* Specialize prepared span lengths from one ordinary coverage scalar. The
   original extents are copied into preparation-owned argument pages; offsets
   stay fixed. Exact table writes and active-page reads join the shared solve. */
struct metal_result metal_mask_spans(struct metal_program *,ncclMeshBuffer table,
                                    const struct mesh_dma_span *,size_t count,ncclMeshBuffer active);
struct metal_coverage_projection {
  size_t transport,first,bytes,units,cut;
  uint64_t base;
  const uint64_t *ends;
  ncclMeshPageCoverage values;
};
struct metal_result metal_coverage(struct metal_program *,const struct metal_coverage_projection *,size_t count);
void metal_coverage_announcement(struct metal_program *,ncclMeshPageCoverage,size_t cut);
void metal_place_sources(struct metal_program *,size_t previous,const size_t *positions,
                        const void *const *effects,size_t count,size_t *relocation);
/* Repeat a declared byte pattern: the first element's scalar bits, followed by
   zero elements through unit_bytes. The command has only an exact output reach. */
struct metal_result metal_fill_pattern(struct metal_program *, void *dst, size_t dst_offset,
    size_t bytes, uint64_t scalar, size_t element_bytes, size_t unit_bytes);
/* `type` METAL_MERGE: the partials merge (nccl.h ncclMeshRedOpCreatePartials) of rows of `op` float32 [m, l, O], a row
   with l > 0 rescaled to the greater m and summed, one with none counting for none (two such the row [the least float,
   0, 0]); else `type` an ncclDataType_t and op as nccl-mesh.c combine_into (0 sum, 1 product, 2 max, 3 min) */
#define METAL_MERGE 12
/* dst = with op src, n elements of an ncclDataType_t: landed bytes (src, loaded system-coherent) combined with an
   operand read in place (`with`: a lease's send buffer, nccl.h ncclMeshMemBuffer), the first combine of a range no step
   wrote, after its declared input observation */
struct metal_scalar { void *buffer; size_t offset; uint64_t bits; };
struct metal_result metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      size_t element_bytes, struct metal_scalar scalar);
struct metal_result metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   size_t element_bytes, uint64_t divisor);
/* A wait (design/transport.md section 14): in `words` (the Metal buffer over a holding's counts), the count at `word`
   (its `landed`) until it reaches `target` (0: none), in pages; `data` at `data_offset` the landed bytes a landing combines (the group's
   landing area), `bytes` its exact extent (zero for a counter-only observation); its failure recorded in `violation`'s words at entry `at` of channel `channel`; `stride` the pages
   each later repeat of a fused crossing's adds to its target (ncclMeshCall.frames; 0 none). */
struct metal_wait { void *words, *violation, *data; size_t word, data_offset; mesh_pages target; uint64_t at; uint32_t channel; mesh_pages stride; size_t bytes; };
/* A publication: in `words`, the count at `word` (a holding's `ready`) stored `value` after one system fence, each
   later repeat of a fused crossing's `stride` pages past it */
struct metal_release { void *words; size_t word; mesh_pages value, stride; };
struct metal_result metal_publish(struct metal_program *, const struct metal_release *);
/* A numerical sum and its preparation relations. Landing maps name each
   input's page availability; summed names its downstream publication. Shared
   page lowering consumes both. The numerical body retains only cell ranges,
   operand addresses, the combine equation and ordinary activity values. */
struct metal_zelloth_input { void *data; size_t data_offset; void *words; size_t word; void *summed; size_t summed_offset;
                             mesh_pages target, packets; const struct mesh_landing_run *runs; size_t count;
                             size_t transport; /* retained incoming transport identity, preparation only */
                             uint32_t channel; uint64_t entry; };
/* Each sum's exact accesses enter the common command graph. */
struct metal_zelloth_sum { void *out; size_t out_offset; void *own; size_t own_offset;
  uint32_t inputs; const struct metal_zelloth_input *in; uint32_t own_access;
  size_t call;                         /* retained declared source occurrence */
  /* Exact output ranges which overwrite an outstanding SEND source. These
     observations constrain arithmetic stores, never the scratch DMA landing. */
  size_t waits; const struct metal_wait *wait;
  /* Initial accumulator's declared COPY producer, when there is one. Its
     activity replaces an absent accumulator with the declared identity. */
  struct metal_zelloth_input initial; size_t initial_first; uint64_t identity; };
/* Emit each resolved value version as a command with its exact page accesses.
   The grid assigns one common landing cell to each threadgroup. */
struct metal_result metal_zelloth(struct metal_program *, int type, int op, size_t element_bytes, uint32_t sums,
                                  const struct metal_zelloth_sum *sum);
/* The next dispatch reads/writes these prepared page
   ranges. Residency and diagnostic atomic words do not imply data ordering. */
void metal_accesses(struct metal_program *, const ncclMeshAccess *, size_t n);
/* A communicator's gate word (at `word` in `words`) stored `value` after one system fence, as metal_publish stores: the
   number of the group whose landings it opens (nccl-mesh.c gate_of), a count of groups and no holding's. */
struct metal_result metal_open_gate(struct metal_program *, void *words, size_t word, uint64_t value);
/* the wait alone (a COPY's landing in place), one threadgroup */
struct mesh_landing_run;
/* metal_contents: a shared buffer's host address (NULL: none, a private buffer's) */
void *metal_contents(void *buffer);
/* Shared payload and page-effect primitives from this library's source build. */
const char *metal_prelude(void);
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

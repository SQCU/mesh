/* libnccl-mesh's kernels: their Metal text, after rdma/mesh.metal's, and their kinds, which nccl-mesh-metal.m
   dispatches.  The text is compiled when the library is built, never in a process that runs it: the build tool
   (nccl-mesh-kernels.m) writes it out, the Makefile compiles it to libnccl-mesh.metallib (vendor/PROVENANCE.md), and
   each node's install makes its binary archive of every kind's pipeline; nccl-mesh-metal.m loads the two (kernels_of)
   and hands STEPS to callers that compile their own kernels after it (nccl.h ncclMeshMetalPrelude). */
#ifndef NCCL_MESH_KERNELS_H
#define NCCL_MESH_KERNELS_H
#import <Foundation/Foundation.h>
#include <stdint.h>
#include "nccl-mesh-metal.h"
#include "nccl.h"

#define STR(...) #__VA_ARGS__
#define XSTR(...) STR(__VA_ARGS__)
/* a kernel's arguments, one layout for the host (struct args) and the kernels (struct mesh_args): a wait's count at
   aux until scalar, a publication's count at aux stored scalar; a landing's count is its first page's, of count pages of
   mask bytes (nccl-mesh-metal.m metal_land).
   A holding's counts are pages, as words here (nccl-mesh-metal.m wait_args, release_args); aux and first are byte
   offsets, at an entry */
#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 at, sequence; U32 count, mask, channel; }

/* a step of a compiled program, the library's one implementation of it (metal-microbench docs/acceptance-20261014.md
   A39), after rdma/mesh.metal: libnccl-mesh's kernels call these, and so does a caller's kernel that runs a step its
   group handed it (ncclMeshFused: a product's finish that publishes its own operand, the kernel after the call that
   reads its landing buffer), its six bindings one mesh_step_binding (nccl.h ncclMeshFused's order) where its encoder
   is classic and takes them as bytes (setBytes), its argument table's six where it is Metal 4.  Each guards nothing
   the caller owns: under NCCL's contract the caller touches no buffer of a collective in flight and issues the same
   collectives in the same order on every rank, so the order of a program's reads and writes is known from the programs
   both ranks compiled and is encoded in the stream's order (rdma/NCCL.md "One track"); the host issues no group on a
   communicator whose failure it has read (nccl-mesh.c comm_failure), a group already encoded runs as encoded, and each
   wait on a peer that never publishes ends at its own bound, or sooner once any wait on the same failure words has
   failed (mesh_step_failed).  mesh_step_enter: a kernel whose record is the argument
   ring's (every Metal 4 dispatch's, and a deferred step's on either form) first stores the record's sequence in
   `progress` (`first` in one thread of the dispatch; every dispatch before it has finished: a barrier stands between
   each two, or a classic encoder's serial order), the argument ring's own record of what the GPU has begun
   (nccl-mesh-metal.m record4: a record is written again only once its dispatch began); a classic dispatch's record is
   its setBytes, its sequence 0, and it stores none.  It reads nothing, and its result is true (kept so that a caller's
   kernel that tests it compiles as it is).  mesh_step_wait: a wait on a holding's count at `word` until it reaches
   `target`, at most p.polls loads (mesh_wait_bounded: the program's bound, at most NCCL_MESH_POLLS, metal_begin),
   ended by the count, by the link's loss (~0) or by the bound, reading nothing but that word; given the failure words
   vb, ended also once their claim is taken, read every 2^10 loads (mesh.metal mesh_wait_bounded's stop).
   mesh_step_claim: failure word [1]'s low half exchanged for 1, true in the one caller that found it 0, so of every wait
   that fails (a dispatch's threadgroups', the tracks') one records.  mesh_step_failed: whether that claim is taken,
   where p.polls names failure words (a dispatch with none has no claim).  mesh_step_check: the wait's failure recorded by its claimant in
   failure words [0] (its cause, nccl-mesh-metal.h METAL_*), [3] and [4] and then [2], stored last by the claimant; a
   loser records nothing and waits for nothing: a revoke, the link lost or the bound reached; a landing is no failure,
   whatever its bytes (a dud's among them: placed where a send must happen and its producer is gone, never a value a
   consumer reads as a signal, design/rubrics.md M4).  The host reads the record after the command buffer and fails the
   group as a value (nccl-mesh.c comm_failure); no kernel halts on it.  mesh_step_counts: the record's waits, `landed`
   at p.aux until p.scalar, where its target is not 0 and no wait on vb has
   failed (mesh_step_failed), each ended once one has: a dispatch's later waits, a wide landing's later threadgroups'
   among them, end at once, so a dispatch spins about one bound however many waves its threadgroups run in (a resident
   grid past the GPU's co-residency reset the Mini, 2026-10-21).  mesh_step_await: the
   counts awaited by thread 0 and checked, then the threadgroup past a device barrier and one system-scope fence, after
   which its landed bytes are read (a running kernel sees the NIC's stores late without a fence: mesh.metal
   mesh_wait_bounded); every threadgroup of a dispatch may await them, each its own bounded wait, ended at once after any
   of them failed (mesh_step_counts), calling it from every thread alike; its result is true, kept as mesh_step_enter's.  mesh_step_write: element k of the operand a publication sends, stored system-coherent (a plain store
   reaches the NIC only once its command buffer has ended, and the bridge sends as soon as `ready` passes: 2,041,216 of
   4,194,304 plain-stored words read stale on the M5, none stored so, metal-microbench docs/measurement.md, 2026-09-29);
   the caller's kernel stores every element it publishes through it, once.  mesh_step_release: a publication, the
   operand already written through mesh_step_write by the caller's kernel (its last threadgroup to arrive calls it,
   after every other threadgroup's stores): a device barrier, then thread 0's one system-scope fence (mesh.metal
   mesh_publish: every store of the operand visible before the count the bridge acts on) and its one store of the
   holding's `ready` (p.scalar at p.aux).  mesh_step_read: element k of a landing buffer the
   NIC wrote, loaded system-coherent (mesh.metal mesh_payload; half4 where it names no type: float16 fourths), so the
   caller names no transport qualifier of its own (they are compiled here, under mesh.metal's pragma: a header after
   it, MetalPerformancePrimitives', may end that pragma). */
static NSString *const STEPS =
  @"\nusing namespace metal;\n"
  XSTR(ARGS(mesh_args, ulong, uint)) ";\n"
  "struct mesh_step_binding { device uchar *d; device uchar *s; constant mesh_args *p; device uchar *a; device uchar *v; device ulong *progress; };\n"
  "static inline bool mesh_step_enter(constant mesh_args &p, device uchar *vb, device ulong *progress, bool first) { "
  "(void)vb; if (p.sequence && first) *progress = p.sequence; return true; }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, ulong target) { "
  "return mesh_wait_bounded((volatile coherent(system) device const ulong *)word, target, p.polls); }\n"
  "template <typename W> inline W mesh_wait(volatile coherent(system) device const W *word, ulong target) { "
  "return mesh_wait_bounded(word, target, " XSTR(NCCL_MESH_POLLS) "); }\n"
  "static inline bool mesh_step_claim(device uchar *vb) { "
  "return atomic_exchange_explicit((device atomic_uint *)(vb + 8), 1u, memory_order_relaxed) == 0u; }\n"
  "static inline bool mesh_step_failed(constant mesh_args &p, device uchar *vb) { "
  "return p.polls && atomic_load_explicit((device atomic_uint *)(vb + 8), memory_order_relaxed) != 0u; }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, ulong target, device uchar *vb) { "
  "return mesh_wait_bounded((volatile coherent(system) device const ulong *)word, target, p.polls, p.polls ? (device atomic_uint *)(vb + 8) : nullptr); }\n"
  "static inline void mesh_step_check(constant mesh_args &p, ulong status, ulong target, device uchar *vb) { "
  "if (!p.polls) return; "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; "
  "ulong cause; "
  "if (status != ~0UL && status >= target) return; "
  "cause = status == ~0UL ? " XSTR(METAL_LOST) "UL : v[6] ? " XSTR(METAL_REVOKED) "UL : " XSTR(METAL_BOUND) "UL; "
  "if (!mesh_step_claim(vb)) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "v[0] = cause; v[3] = ulong(p.channel); v[4] = ulong(p.polls); mesh_publish(v + 2, p.at + 1); }\n"
  "static inline void mesh_step_counts(constant mesh_args &p, device uchar *words, device uchar *vb) { "
  "if (p.scalar && !mesh_step_failed(p, vb)) mesh_step_check(p, mesh_step_wait(p, words + p.aux, p.scalar, vb), p.scalar, vb); }\n"
  "static inline bool mesh_step_await(constant mesh_args &p, device uchar *words, device uchar *vb, uint i) { "
  "if (i == 0) mesh_step_counts(p, words, vb); "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); return true; }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *words, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(words + p.aux), p.scalar); }\n"
  "template <typename T> static inline void mesh_step_write(device uchar *operand, ulong k, T value) { "
  "((volatile coherent(system) device T *)operand)[k] = value; }\n"
  "template <typename T = half4> static inline T mesh_step_read(device const uchar *landing, ulong k) { "
  "return mesh_payload((volatile coherent(system) device const T *)landing + k); }\n";

/* Payload-producing stores are system-coherent: a later SEND may read them before this command buffer ends.
   The accessor contract above applies to the collective's own producers as well as user kernels.
   The kernels, after STEPS: each operator's arithmetic is written once (op_int, op_float, op_f8, merge_row: the partials
   merge, a thread a row), a landing being the wait and then the combine's body; it is the host's (nccl-mesh.c
   combine_into, merge_rows, premultiply, truncdiv): the small floating types in float32 rounded to nearest even,
   float8 decoded and encoded by mesh.metal's f8_to and f8_from, integers wrapping.  A landing's combine loads both its
   operand and the landed bytes system-coherent: the operand may itself have landed in place (a COPY earlier in the
   group, mesh-plan.h mesh_compile's node outside the contributors), and the landed bytes are the NIC's.  Every kernel but
   publish and spin is wide (its threads over the elements); a landing's threadgroup combines a block of four elements a
   thread (a vector where the operand and the landed bytes are both aligned to one, else one by one), its thread 0
   awaiting the count of the block's last page first where the count it last read is short of it (LANDS: the sum page
   by page as they land), and ends its waits at once after any has failed
   (mesh_step_failed); a merge's every threadgroup awaits the
   whole landing (l, its thread in its threadgroup: mesh_step_counts), and a track runs its body in one threadgroup (l its thread there) */
static NSString *const KERNELS =
  @"\n"
  "#define PARAMS device uchar *d, device uchar *s, constant mesh_args &p, device uchar *a, device uchar *vb, device ulong *progress, uint i, uint g, uint l, uint t, threadgroup ulong *tg\n"
  "#define ENTER(first) mesh_step_enter(p, vb, progress, first);\n"
  "#define ENTER_G if (i == 0) mesh_step_enter(p, vb, progress, true);\n"
  "#define HEAD(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]], uint l [[thread_position_in_threadgroup]], uint t [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, l, t, tg); } static void body_##name(PARAMS)\n"
  "#define GROUP(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, i, g, tg); } static void body_##name(PARAMS)\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define AWAIT mesh_step_await(p, a, vb, l);\n"
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { ENTER(i == 0) volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { ENTER(i == 0) volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "static inline void mesh_pages_await(constant mesh_args &p, device uchar *a, device uchar *vb, threadgroup ulong *tg, uint l, ulong want, thread ulong &seen, thread uint &looked) { "
  "if (want <= seen) return; "
  "if (l == 0) { ulong v = ~0UL; if (!mesh_step_failed(p, vb)) { v = mesh_step_wait(p, a + p.aux, want, vb); mesh_step_check(p, v, want, vb); if (v < want) v = ~0UL; } tg[looked & 1] = v; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); seen = tg[looked & 1]; looked++; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); }\n"
  "#define LANDS(T, OP) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *xs = (volatile coherent(system) device const T *)(d + p.dst); "
  "volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); "
  "const ulong block = 4 * ulong(t), step = 4 * ulong(g); const uint shift = ctz(uint(p.mask / sizeof(T))); ulong seen = 0; uint looked = 0; "
  "if ((p.dst | p.src) % (4 * sizeof(T)) == 0) for (ulong b = 4 * ulong(i - l); b < p.n; b += step) { "
  "mesh_pages_await(p, a, vb, tg, l, p.scalar + ((min(b + block, p.n) - 1) >> shift), seen, looked); const ulong k = b + 4 * l; "
  "if (k + 4 <= p.n) { const vec<T, 4> u = mesh_payload((volatile coherent(system) device const vec<T, 4> *)(xs + k)), w = mesh_payload((volatile coherent(system) device const vec<T, 4> *)(y + k)); "
  "vec<T, 4> r; for (int c = 0; c < 4; c++) r[c] = OP(u[c], w[c], p.op); *(volatile coherent(system) device vec<T, 4> *)(x + k) = r; } "
  "else for (ulong e = k; e < p.n; e++) x[e] = OP(mesh_payload(xs + e), mesh_payload(y + e), p.op); } "
  "else for (ulong b = 4 * ulong(i - l); b < p.n; b += step) { const ulong end = min(b + block, p.n); "
  "mesh_pages_await(p, a, vb, tg, l, p.scalar + ((end - 1) >> shift), seen, looked); "
  "for (ulong e = b + l; e < end; e += t) x[e] = OP(mesh_payload(xs + e), mesh_payload(y + e), p.op); }\n"
  "#define COMBINE(name, T, OP) HEAD(land_##name) { ENTER(i == 0) LANDS(T, OP) } "
  "HEAD(combine3_##name) { "
  "ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); device const T *z = (device const T *)(a + p.aux); "
  "volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP x[k] = OP(z[k], mesh_payload(y + k), p.op); }\n"
  "#define INTEGER(name, T, SIGNED) COMBINE(name, T, op_int<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); ulong c = ulong(T(p.scalar)); LOOP x[k] = T(ulong(y[k]) * c); } "
  "HEAD(truncdiv_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP { T v = mesh_payload(y + k); bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) COMBINE(name, T, op_float<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "#define FLOAT8(name, E5) COMBINE(name, uchar, op_f8<E5>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device uchar *x = (volatile coherent(system) device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = f8_from(f8_to(y[k], E5) * c, E5); }\n"
  "INTEGER(i8, char, true)\n"
  "INTEGER(u8, uchar, false)\n"
  "INTEGER(i32, int, true)\n"
  "INTEGER(u32, uint, false)\n"
  "INTEGER(i64, long, true)\n"
  "INTEGER(u64, ulong, false)\n"
  "FLOATING(f16, half)\n"
  "FLOATING(f32, float)\n"
  "FLOATING(bf16, bfloat)\n"
  "FLOAT8(e4m3, false)\n"
  "FLOAT8(e5m2, true)\n"
  "GROUP(publish) { ENTER_G mesh_step_release(p, d, i); }\n"
  "GROUP(spin) { ENTER_G if (i == 0) mesh_step_counts(p, d, vb); }\n"
  "constant float PARTIAL_NONE = -3.4028234663852886e38f;\n"
  "static inline void merge_row(volatile coherent(system) device float *x, volatile coherent(system) device const float *y, uint u) { "
  "volatile coherent(system) device const float *xs = (volatile coherent(system) device const float *)x; "
  "const float m = mesh_payload(y), ly = mesh_payload(y + 1), mx = mesh_payload(xs), lx = mesh_payload(xs + 1); const bool mine = lx > 0.0f, theirs = ly > 0.0f; "
  "if (mine && !theirs) return; "
  "if (!mine || !theirs) { for (uint j = 0; j < u; j++) x[j] = theirs ? mesh_payload(y + j) : j ? 0.0f : PARTIAL_NONE; return; } "
  "const float top = max(mx, m), a = exp(mx - top), b = exp(m - top); x[0] = top; "
  "for (uint j = 1; j < u; j++) x[j] = (a > 0.0f ? mesh_payload(xs + j) * a : 0.0f) + (b > 0.0f ? mesh_payload(y + j) * b : 0.0f); }\n"
  "#define MERGE volatile coherent(system) device float *x = (volatile coherent(system) device float *)(d + p.dst); volatile coherent(system) device const float *y = (volatile coherent(system) device const float *)(s + p.src); "
  "for (ulong r = i; r < p.n; r += g) merge_row(x + r * p.op, y + r * p.op, p.op);\n"
  "HEAD(land_merge) { ENTER(i == 0) AWAIT MERGE }\n";

/* every kernel a track may record, by its kind: one list, the one numbering of the host's steps (struct host_op kind,
   a device's pipelines' index: nccl-mesh-metal.m kernels_of) and of the track kernel's switch (TRACKS' enum and TRACK_KERNEL's cases, made
   from this list by the preprocessor, so neither names a kind the other lacks); each kernel's name, which the Metal
   library's lookup takes, once a process (nccl-mesh-metal.m kernels_of) and once an install (nccl-mesh-kernels.m)
   (newFunctionWithName, an interface the library does not implement: design/rubrics.md, Strings are messages) and whose body the switch calls, and the word a consumption's item awaits in it (record_step:
   WORD_D a spin's, in its d; WORD_A a merge's, in its a; WORD_P a landing's first pages', in its a; 0 none) */
enum { WORD_D = 1, WORD_A, WORD_P };
#define KINDS_OF(X) \
  X(copy_plain_4, 0) X(copy_land_4, 0) X(copy_plain_1, 0) X(copy_land_1, 0) \
  X(land_i8, WORD_P) X(combine3_i8, 0) X(premul_i8, 0) X(truncdiv_i8, 0) \
  X(land_u8, WORD_P) X(combine3_u8, 0) X(premul_u8, 0) X(truncdiv_u8, 0) \
  X(land_i32, WORD_P) X(combine3_i32, 0) X(premul_i32, 0) X(truncdiv_i32, 0) \
  X(land_u32, WORD_P) X(combine3_u32, 0) X(premul_u32, 0) X(truncdiv_u32, 0) \
  X(land_i64, WORD_P) X(combine3_i64, 0) X(premul_i64, 0) X(truncdiv_i64, 0) \
  X(land_u64, WORD_P) X(combine3_u64, 0) X(premul_u64, 0) X(truncdiv_u64, 0) \
  X(land_f16, WORD_P) X(combine3_f16, 0) X(premul_f16, 0) X(land_f32, WORD_P) X(combine3_f32, 0) X(premul_f32, 0) \
  X(land_bf16, WORD_P) X(combine3_bf16, 0) X(premul_bf16, 0) X(land_e4m3, WORD_P) X(combine3_e4m3, 0) X(premul_e4m3, 0) \
  X(land_e5m2, WORD_P) X(combine3_e5m2, 0) X(premul_e5m2, 0) \
  X(publish, 0) X(spin, WORD_D) X(land_merge, WORD_A)
#define KIND_ENUM(name, word) KIND_##name,
#define KIND_NAME(name, word) #name,
#define KIND_WORD(name, word) word,
#define KIND_CASE(name, word) case KIND_##name: body_##name(o.d, o.s, o.p, o.a, o.v, progress, i, g, i, g, tg); break;
/* the kinds, then the track kernel's own pipeline (KIND_tracks: no track records it) */
enum { KINDS_OF(KIND_ENUM) KINDS, KIND_tracks = KINDS, PIPELINES };
static const char *const NAME[PIPELINES] = {KINDS_OF(KIND_NAME) "tracks"};
static const uint8_t WORD[KINDS] = {KINDS_OF(KIND_WORD)};

/* the track kernel (nccl-mesh-metal.h metal_tracks; rdma/NCCL.md "Per-channel progress"), after KERNELS, for a
   completion on two or more tracks (a group on one track is plain dispatches: nccl-mesh.c complete_track): one thread
   group a track, its thread 0 choosing the next item of either cursor whose own conditions hold (a consumption first):
   a publication's dependencies (nothing credits it); a consumption's dependencies, its landing and, where it combines
   into bytes a SEND of its channel reads (`need` that send entry + 1; 0 none), that publication run (the publication
   cursor past it), its own wait; every thread running the item's steps by their bodies,
   and a consumption's end raising the track's count (system scope, release).  A group that chooses nothing for the
   program's polls of passes records its blocked items' failures (mesh_step_check for a landing, else the bound at its
   position, by its claimant: mesh_step_claim) and ends once nothing more can run; a track ends at its own bound, as a
   wait does, and reads no other track's failure.  The records' layouts are struct track_head, track, track_item,
   track_op and track_dep below, the host's alike; struct track_op's kind one of KINDS_OF's, the enum the host's */
static NSString *const TRACKS =
  @"\n"
  "enum { " XSTR(KINDS_OF(KIND_ENUM)) " };\n"
  "struct track_op { device uchar *d; device uchar *s; device uchar *a; device uchar *v; mesh_args p; uint kind; uint pad; };\n"
  "struct track_item { ulong position, need, target; device ulong *word; device uchar *v; uint first_op, ops, first_dep, deps, wait_op, cursor; };\n"
  "struct track_dep { ulong count; uint other, pad; };\n"
  "struct track { uint first[2], count[2]; };\n"
  "struct track_head { ulong tracks_off, items_off, ops_off, deps_off, count_off, sequence, progress_off; uint tracks, polls; };\n"
  "static inline bool track_ready(constant track_item &it, constant track_dep *deps, volatile coherent(system) device ulong *count) { "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
    "for (uint k = 0; k < it.deps; k++) if (count[deps[it.first_dep + k].other] < deps[it.first_dep + k].count) return false; "
  "if (it.cursor != 1 || it.wait_op == ~0u) return true; "
  "return *(volatile coherent(system) device const ulong *)it.word >= it.target; }\n"
  "static inline void track_expired(constant track_item &it, constant track_op *ops, uint polls) { "
  "if (it.wait_op != ~0u) { constant track_op &w = ops[it.wait_op]; mesh_step_check(w.p, *(volatile coherent(system) device const ulong *)it.word, it.target, w.v); return; } "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)it.v; "
  "if (!v || !mesh_step_claim(it.v)) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "v[0] = " XSTR(METAL_BOUND) "UL; v[4] = ulong(polls); mesh_publish(v + 2, it.position + 1); }\n";
static NSString *const TRACK_KERNEL =
  @"kernel void tracks(constant uchar *base [[buffer(0)]], device uchar *ring [[buffer(1)]], uint t [[threadgroup_position_in_grid]], "
  "uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { "
  "constant track_head &h = *(constant track_head *)base; "
  "if (t >= h.tracks) return; "
  "constant track &tr = ((constant track *)(base + h.tracks_off))[t]; "
  "constant track_item *items = (constant track_item *)(base + h.items_off); "
  "constant track_op *ops = (constant track_op *)(base + h.ops_off); "
  "constant track_dep *deps = (constant track_dep *)(base + h.deps_off); "
  "volatile coherent(system) device ulong *count = (volatile coherent(system) device ulong *)(ring + h.count_off); "
  "threadgroup ulong tg[4]; threadgroup uint pick[2]; "
  "uint next0 = 0, next1 = 0; ulong looks = 0; bool expired = false; "
  "if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(ring + h.progress_off), h.sequence); "
  "for (;;) { "
  "if (i == 0) { uint chosen = ~0u, left = 0; "
  "if (next1 < tr.count[1]) { left = 1; constant track_item &c = items[tr.first[1] + next1]; "
  "if ((next0 >= tr.count[0] || items[tr.first[0] + next0].position >= c.need) && track_ready(c, deps, count)) { chosen = tr.first[1] + next1; next1++; } } "
  "if (chosen == ~0u && next0 < tr.count[0]) { left = 1; if (track_ready(items[tr.first[0] + next0], deps, count)) { chosen = tr.first[0] + next0; next0++; } } "
  "if (chosen != ~0u) { looks = 0; expired = false; } "
  "else if (left && expired) left = 0; "
  "else if (left && ++looks >= h.polls) { "
  "if (next1 < tr.count[1]) track_expired(items[tr.first[1] + next1], ops, h.polls); "
  "if (next0 < tr.count[0]) track_expired(items[tr.first[0] + next0], ops, h.polls); expired = true; } "
  "pick[0] = chosen; pick[1] = left; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); "
  "const uint chosen = pick[0]; "
  "if (!pick[1]) return; "
  "if (chosen != ~0u) { constant track_item &it = items[chosen]; "
  "for (uint k = 0; k < it.ops; k++) { constant track_op &o = ops[it.first_op + k]; device ulong *progress = (device ulong *)o.d; "
  "switch (o.kind) { " XSTR(KINDS_OF(KIND_CASE)) " default: break; } "
  "threadgroup_barrier(mem_flags::mem_device); } "
  "if (it.cursor == 1 && i == 0) mesh_publish(count + t, it.position + 1); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); } }\n";

#endif

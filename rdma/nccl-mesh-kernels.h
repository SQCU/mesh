/* libnccl-mesh's kernels: their Metal text, after rdma/mesh.metal's, and their kinds, which nccl-mesh-metal.m
   dispatches.  The text is compiled when the library is built, never in a process that runs it: the build tool
   (nccl-mesh-kernels.m) writes it out, the Makefile compiles it to libnccl-mesh.metallib (vendor/PROVENANCE.md), and
   each node's install makes its binary archive of every kind's pipeline; nccl-mesh-metal.m loads the two (kernels_of)
   and hands STEPS to callers that compile their own kernels after it (nccl.h ncclMeshMetalSteps). */
#ifndef NCCL_MESH_KERNELS_H
#define NCCL_MESH_KERNELS_H
#import <Foundation/Foundation.h>
#include <stdint.h>
#include "nccl-mesh-metal.h"

#define STR(...) #__VA_ARGS__
#define XSTR(...) STR(__VA_ARGS__)
/* a kernel's arguments, one layout for the host (struct args) and the kernels (struct mesh_args) */
#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 first, slot, at, sequence; U32 count, mask, channel; }

/* a step of a compiled program, the library's one implementation of it (metal-microbench docs/acceptance-20261014.md
   A39), after rdma/mesh.metal: libnccl-mesh's kernels call these, and so does a caller's kernel that runs a step its
   group handed it (ncclMeshFused: a product's finish that publishes its piece, the kernel after the call that lands its
   peer's), its six bindings one mesh_step_binding (nccl.h ncclMeshFused's order) where its encoder is classic and takes
   them as bytes (setBytes), its argument table's six where it is Metal 4.  Each guards nothing the caller owns: under
   NCCL's contract the caller touches no buffer of a collective in flight and issues the same collectives in the same order on every rank, so the order of a program's reads and writes
   is known from the programs both ranks compiled and is encoded in the stream's order (rdma/NCCL.md "One track"); the
   host issues no group on a communicator whose failure it has read (nccl-mesh.c comm_failure), a group already encoded
   runs as encoded, and each wait on a peer that never publishes ends at its own bound.  mesh_step_enter: a kernel whose
   record is the argument ring's (every Metal 4 dispatch's, and a deferred step's on either form) first stores the
   record's sequence in `progress` (`first` in one thread of the dispatch; every dispatch before it has finished: a
   barrier stands between each two, or a classic encoder's serial order), the argument ring's own record of what the GPU
   has begun (nccl-mesh-metal.m record4: a record is written again only once its dispatch began); a classic dispatch's
   record is its setBytes, its sequence 0, and it stores none.  It reads nothing, and its result is true (kept
   so that a caller's kernel that tests it, metal-microbench metal/layers.metal, compiles as it is).
   mesh_step_wait: a wait on its run's last unit's landed word at p.aux for 1 + p.scalar in its bits but MESH_DUD, at
   most p.polls loads (mesh_remote_wait_bounded: the program's bound, at most NCCL_MESH_POLLS, metal_begin), ended by the
   landing, a dud's among them, by the link's loss (~0) or by the bound, reading nothing but that word.
   mesh_step_claim: failure word [1]'s low half exchanged for
   1, true in the one caller that found it 0, so of every wait that fails (a dispatch's threadgroups', the tracks') one
   records.  mesh_step_check: the wait's failure recorded by its claimant in failure words [0] (its cause, nccl-mesh-metal.h
   METAL_*), [3] and [4] and then [2], stored last by the claimant; a loser records nothing and waits for nothing: a
   revoke, the link lost, the bound reached, or a dud's landing (its word's MESH_DUD, the bridge's: design/transport.md
   section 10, its producer's group failed, so a member past a failed one never completes on filler); any other landing
   is no failure, whatever its bytes (no byte is read as a signal, design/rubrics.md M4), and it reads the failure words,
   claims and fences only where it failed.  The host reads the record after the command buffer and fails the group as a value
   (nccl-mesh.c comm_failure); no kernel halts on it (METAL_COUNTED_ARRIVED reads it to report its call's outcome).  mesh_step_await: a landing awaited by thread 0 and checked, then the
   threadgroup past a device barrier and one system-scope fence, its payload then read with mesh_payload (a running
   kernel sees the NIC's stores late without a fence: mesh.metal mesh_wait_bounded); every threadgroup of
   a dispatch may await one landing, each its own bounded wait, calling it from every thread alike; its result is true,
   kept as mesh_step_enter's.  mesh_step_release: a piece the threadgroup stored into its run's slots, then, after a
   device barrier, thread 0's one system-scope fence (mesh.metal mesh_publish: every thread's stores of the piece
   visible before any ready word the bridge acts on) and its units' ready words (p.count from p.aux, each p.scalar): a
   publication is one threadgroup's, one fence; with the failure words (the library's own kernels'), each p.scalar with
   MESH_DUD where a wait of the communicator has failed (their claim, read after the fence, which follows every
   observation that ordered this publication after that wait: a dispatch boundary, a track's count), so its bridge sends
   the units as duds and no member past this one completes on a piece a failed wait made (design/transport.md section
   10); a caller's kernel's release (no failure words: a fused crossing, two ranks, its publication before its landing)
   stores p.scalar.  A caller's kernel stores its piece's element k of type T with
   mesh_step_store and reads its peer's with mesh_step_read<T> (half4 where it names none: float16 fourths), so it names
   no transport qualifier of its own (they are compiled here, under mesh.metal's pragma: a header after it,
   MetalPerformancePrimitives', may end that pragma). */
static NSString *const STEPS =
  @"\nusing namespace metal;\n"
  XSTR(ARGS(mesh_args, ulong, uint)) ";\n"
  "struct mesh_step_binding { device uchar *d; device uchar *s; constant mesh_args *p; device uchar *a; device uchar *v; device ulong *progress; };\n"
  "static inline bool mesh_step_enter(constant mesh_args &p, device uchar *vb, device ulong *progress, bool first) { "
  "(void)vb; if (p.sequence && first) *progress = p.sequence; return true; }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, device uchar *vb) { "
  "(void)vb; return mesh_remote_wait_bounded((volatile coherent(system) device const ulong *)word, p.scalar, p.polls); }\n"
  "static inline bool mesh_step_claim(device uchar *vb) { "
  "return atomic_exchange_explicit((device atomic_uint *)(vb + 8), 1u, memory_order_relaxed) == 0u; }\n"
  "static inline void mesh_step_check(constant mesh_args &p, ulong status, device uchar *ring, device uchar *vb) { "
  "if (!p.polls) return; "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; "
  "const bool landed = (status & ~MESH_DUD) >= 1 + p.scalar; "
  "ulong cause; "
  "(void)ring; if (landed && !(status & MESH_DUD)) return; "
  "cause = status == ~0UL ? " XSTR(METAL_LOST) "UL : landed ? " XSTR(METAL_DUD) "UL : v[6] ? " XSTR(METAL_REVOKED) "UL : " XSTR(METAL_BOUND) "UL; "
  "if (!mesh_step_claim(vb)) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "v[0] = cause; v[3] = ulong(p.channel); v[4] = ulong(p.polls); mesh_publish(v + 2, p.at + 1); }\n"
  "static inline bool mesh_step_await(constant mesh_args &p, device uchar *words, device uchar *ring, device uchar *vb, uint i) { "
  "if (i == 0) mesh_step_check(p, mesh_step_wait(p, words + p.aux, vb), ring, vb); "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); return true; }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *words, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0 && p.count) { volatile coherent(system) device ulong *w = (volatile coherent(system) device ulong *)(words + p.aux); "
  "mesh_publish(w, p.scalar); for (uint u = 1; u < p.count; u++) w[u] = p.scalar; } }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *words, device uchar *vb, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0 && p.count) { volatile coherent(system) device ulong *w = (volatile coherent(system) device ulong *)(words + p.aux); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "const ulong value = p.polls && ((volatile coherent(system) device const uint *)vb)[2] != 0u ? p.scalar | MESH_DUD : p.scalar; "
  "for (uint u = 0; u < p.count; u++) w[u] = value; } }\n"
  "template <typename T> static inline void mesh_step_store(constant mesh_args &p, device uchar *slot, ulong k, T value) { "
  "((volatile coherent(system) device T *)(slot + p.dst))[k] = value; }\n"
  "template <typename T = half4> static inline T mesh_step_read(constant mesh_args &p, device uchar *slot, ulong k) { "
  "return mesh_payload((volatile coherent(system) device const T *)(slot + p.src) + k); }\n";

/* the kernels, after STEPS: each operator's arithmetic is written once (op_int, op_float, op_f8, merge_row: the partials
   merge, a thread a row), a landing being the wait and then the combine's body; it is the host's (nccl-mesh.c
   combine_into, merge_rows, premultiply, truncdiv): the small floating types in float32 rounded to
   nearest even, float8 decoded and encoded by mesh.metal's f8_to and f8_from, integers wrapping */
static NSString *const KERNELS =
  @"\n"
  "#define PARAMS device uchar *d, device uchar *s, constant mesh_args &p, device uchar *a, device uchar *vb, device ulong *progress, uint i, uint g, threadgroup ulong *tg\n"
  "#define ENTER(first) mesh_step_enter(p, vb, progress, first);\n"
  "#define ENTER_G if (i == 0) mesh_step_enter(p, vb, progress, true);\n"
  "#define HEAD(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) { body_##name(d, s, p, d, vb, progress, i, g, nullptr); } static void body_##name(PARAMS)\n"
  "#define HEAD_A(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) { body_##name(d, s, p, a, vb, progress, i, g, nullptr); } static void body_##name(PARAMS)\n"
  "#define GROUP(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, tg); } static void body_##name(PARAMS)\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define AWAIT mesh_step_await(p, a, s, vb, i);\n"
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { ENTER(i == 0) device W *x = (device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { ENTER(i == 0) device W *x = (device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); } "
  "HEAD(copy_send_##Z) { ENTER(i == 0) volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "#define BODY(T, OP) device T *x = (device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); "
  "LOOP x[k] = OP(x[k], mesh_payload(y + k), p.op);\n"
  "#define COMBINE(name, T, OP) HEAD(combine_##name) { ENTER(i == 0) BODY(T, OP) } GROUP(land_##name) { ENTER_G AWAIT BODY(T, OP) } "
  "HEAD_A(combine3_##name) { "
  "ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *z = (device const T *)(a + p.aux); "
  "volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP x[k] = OP(z[k], mesh_payload(y + k), p.op); }\n"
  "#define INTEGER(name, T, SIGNED) COMBINE(name, T, op_int<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); ulong c = ulong(T(p.scalar)); LOOP x[k] = T(ulong(y[k]) * c); } "
  "HEAD(truncdiv_##name) { ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); LOOP { T v = y[k]; bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) COMBINE(name, T, op_float<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "#define FLOAT8(name, E5) COMBINE(name, uchar, op_f8<E5>) "
  "HEAD(premul_##name) { ENTER(i == 0) device uchar *x = (device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = f8_from(f8_to(y[k], E5) * c, E5); }\n"
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
  "GROUP(publish) { ENTER_G mesh_step_release(p, d, vb, i); }\n"
  "GROUP(send_small) { ENTER_G volatile coherent(system) device uchar *x = (volatile coherent(system) device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); "
  "LOOP x[k] = y[k]; "
  "mesh_step_release(p, a, vb, i); }\n"
  "GROUP(land_copy) { ENTER_G AWAIT device uchar *x = d + p.dst; volatile coherent(system) device const uchar *y = (volatile coherent(system) device const uchar *)(s + p.src); LOOP x[k] = mesh_payload(y + k); }\n"
  "GROUP(spin) { ENTER_G if (i == 0) mesh_step_check(p, mesh_step_wait(p, d + p.aux, vb), s, vb); }\n"
  "static inline ulong counted_rows(device const long *c, ulong first, ulong count) { ulong rows = 0; for (ulong k = 0; k < count; ++k) rows += c[first + k] > 0 ? ulong(c[first + k]) : 0UL; return rows; }\n"
  "HEAD(counted_start) { ENTER(i == 0) device long *got = (device long *)(d + p.dst); device const long *c = (device const long *)(s + p.src); "
  "LOOP got[k] = c[p.first + k]; "
  "if (i == 0) { device ulong *tail = (device ulong *)(d + p.aux); tail[0] = counted_rows(c, 0, p.first); tail[1] = counted_rows(c, 0, p.scalar); tail[2] = 0; } }\n"
  "HEAD(counted_check) { ENTER(i == 0) if (i == 0 && counted_rows((device const long *)d, p.first, p.count) > p.scalar) ((device ulong *)(d + p.dst))[2] = 1; }\n"
  "HEAD(counted_arrived) { ENTER(i == 0) if (i == 0) { device const ulong *tail = (device const ulong *)(s + p.src) + p.first; "
  "bool over = counted_rows((device const long *)(s + p.src), 0, p.n) > p.aux || tail[1] > p.scalar || tail[2] != 0; "
  "volatile coherent(system) device ulong *word = (volatile coherent(system) device ulong *)(d + p.dst); "
  "const bool failed = p.mask && ((volatile coherent(system) device const ulong *)vb)[2] != 0UL; "
  "if (over && p.mask) mesh_publish((volatile coherent(system) device ulong *)vb + 5, 1UL); "
  "if (*word == 0) mesh_publish(word, over || failed ? 2UL : 1UL); } }\n"
  "#define PLACE(BEFORE, ROWS, LIMIT) threadgroup ulong *place = tg; if (i == 0) { ulong at = (BEFORE) * p.slot + p.scalar, rows = (ROWS) * p.slot; "
  "ulong bytes = rows > p.scalar ? min(rows - p.scalar, p.n) : 0UL; place[0] = at; place[1] = at >= (LIMIT) ? 0UL : min(bytes, (LIMIT) - at); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "#define COUNTED(Z, W) "
  "GROUP(counted_send_##Z) { ENTER_G device const long *c = (device const long *)(a + p.aux); PLACE(counted_rows(c, 0, p.first), counted_rows(c, p.first, p.count), p.at) "
  "volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src + place[0]); "
  "for (ulong k = i; k < place[1] / sizeof(W); k += g) x[k] = y[k]; } "
  "GROUP(counted_land_##Z) { ENTER_G device const long *c = (device const long *)(a + p.aux); PLACE(counted_rows(c, 0, p.first), counted_rows(c, p.first, p.count), p.at) "
  "device W *x = (device W *)(d + p.dst + place[0]); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); "
  "for (ulong k = i; k < place[1] / sizeof(W); k += g) x[k] = mesh_payload(y + k); } "
  "GROUP(counted_own_##Z) { ENTER_G device const long *c = (device const long *)(a + p.aux); threadgroup ulong *place = tg; "
  "if (i == 0) { ulong to = counted_rows(c, 0, p.first) * p.slot, from = ((device const ulong *)(a + p.aux))[p.scalar] * p.slot, bytes = counted_rows(c, p.first, p.count) * p.slot; "
  "bytes = to >= p.at || from >= p.n ? 0UL : min(bytes, min(p.at - to, p.n - from)); place[0] = to; place[1] = from; place[2] = bytes; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); "
  "device W *x = (device W *)(d + p.dst + place[0]); device const W *y = (device const W *)(s + p.src + place[1]); "
  "for (ulong k = i; k < place[2] / sizeof(W); k += g) x[k] = y[k]; }\n"
  "COUNTED(4, uint)\n"
  "COUNTED(1, uchar)\n"
  "constant float PARTIAL_NONE = -3.4028234663852886e38f;\n"
  "static inline void merge_row(device float *x, volatile coherent(system) device const float *y, uint u) { "
  "const float m = mesh_payload(y), l = mesh_payload(y + 1); const bool mine = x[1] > 0.0f, theirs = l > 0.0f; "
  "if (mine && !theirs) return; "
  "if (!mine || !theirs) { for (uint j = 0; j < u; j++) x[j] = theirs ? mesh_payload(y + j) : j ? 0.0f : PARTIAL_NONE; return; } "
  "const float top = max(x[0], m), a = exp(x[0] - top), b = exp(m - top); x[0] = top; "
  "for (uint j = 1; j < u; j++) x[j] = (a > 0.0f ? x[j] * a : 0.0f) + (b > 0.0f ? mesh_payload(y + j) * b : 0.0f); }\n"
  "#define MERGE device float *x = (device float *)(d + p.dst); volatile coherent(system) device const float *y = (volatile coherent(system) device const float *)(s + p.src); "
  "for (ulong r = i; r < p.n; r += g) merge_row(x + r * p.op, y + r * p.op, p.op);\n"
  "HEAD(combine_merge) { ENTER(i == 0) MERGE } GROUP(land_merge) { ENTER_G AWAIT MERGE }\n";

/* every kernel a track may record, by its kind: one list, the one numbering of the host's steps (struct host_op kind,
   a device's pipelines' index: nccl-mesh-metal.m kernels_of) and of the track kernel's switch (TRACKS' enum and TRACK_KERNEL's cases, made
   from this list by the preprocessor, so neither names a kind the other lacks); each kernel's name, which the Metal
   library's lookup takes, once a process (nccl-mesh-metal.m kernels_of) and once an install (nccl-mesh-kernels.m)
   (newFunctionWithName, an interface the library does not implement: design/rubrics.md, Strings are messages) and whose body the switch calls, and the word a consumption's item awaits in it (record_step:
   WORD_D a spin's, in its d; WORD_A a landing's, in its a; 0 none) */
enum { WORD_D = 1, WORD_A };
#define KINDS_OF(X) \
  X(copy_plain_4, 0) X(copy_land_4, 0) X(copy_send_4, 0) X(copy_plain_1, 0) X(copy_land_1, 0) X(copy_send_1, 0) \
  X(combine_i8, 0) X(land_i8, WORD_A) X(combine3_i8, 0) X(premul_i8, 0) X(truncdiv_i8, 0) \
  X(combine_u8, 0) X(land_u8, WORD_A) X(combine3_u8, 0) X(premul_u8, 0) X(truncdiv_u8, 0) \
  X(combine_i32, 0) X(land_i32, WORD_A) X(combine3_i32, 0) X(premul_i32, 0) X(truncdiv_i32, 0) \
  X(combine_u32, 0) X(land_u32, WORD_A) X(combine3_u32, 0) X(premul_u32, 0) X(truncdiv_u32, 0) \
  X(combine_i64, 0) X(land_i64, WORD_A) X(combine3_i64, 0) X(premul_i64, 0) X(truncdiv_i64, 0) \
  X(combine_u64, 0) X(land_u64, WORD_A) X(combine3_u64, 0) X(premul_u64, 0) X(truncdiv_u64, 0) \
  X(combine_f16, 0) X(land_f16, WORD_A) X(combine3_f16, 0) X(premul_f16, 0) X(combine_f32, 0) X(land_f32, WORD_A) X(combine3_f32, 0) X(premul_f32, 0) \
  X(combine_bf16, 0) X(land_bf16, WORD_A) X(combine3_bf16, 0) X(premul_bf16, 0) X(combine_e4m3, 0) X(land_e4m3, WORD_A) X(combine3_e4m3, 0) X(premul_e4m3, 0) \
  X(combine_e5m2, 0) X(land_e5m2, WORD_A) X(combine3_e5m2, 0) X(premul_e5m2, 0) \
  X(publish, 0) X(send_small, 0) X(land_copy, WORD_A) X(spin, WORD_D) \
  X(counted_start, 0) X(counted_check, 0) X(counted_arrived, 0) X(counted_send_4, 0) X(counted_land_4, 0) X(counted_own_4, 0) \
  X(counted_send_1, 0) X(counted_land_1, 0) X(counted_own_1, 0) X(combine_merge, 0) X(land_merge, WORD_A)
#define KIND_ENUM(name, word) KIND_##name,
#define KIND_NAME(name, word) #name,
#define KIND_WORD(name, word) word,
#define KIND_CASE(name, word) case KIND_##name: body_##name(o.d, o.s, o.p, o.a, o.v, progress, i, g, tg); break;
/* the kinds, then the track kernel's own pipeline (KIND_tracks: no track records it) */
enum { KINDS_OF(KIND_ENUM) KINDS, KIND_tracks = KINDS, PIPELINES };
static const char *const NAME[PIPELINES] = {KINDS_OF(KIND_NAME) "tracks"};
static const uint8_t WORD[KINDS] = {KINDS_OF(KIND_WORD)};

/* the track kernel (nccl-mesh-metal.h metal_tracks; rdma/NCCL.md "Per-channel progress"), after KERNELS, for a
   completion on two or more tracks (a group on one track is plain dispatches: nccl-mesh.c complete_track): one thread
   group serving every track in turn, as the host executor's one loop does (nccl-mesh-host.c metal_tracks), so no
   track waits on a thread group Metal has not made resident: a dispatch's thread groups are not promised to run at
   once, and a track whose progress needed another group's (a dependency on its count, or a landing whose peer awaits
   this node's publication on it) would wait on one that may not run until it ends.  Its thread 0 looks at each track
   from the one after the last served, choosing the first next item of either cursor whose own conditions hold (a
   consumption first): a publication's credit (its track's count at `need`) and dependencies; a consumption's
   dependencies, its landing (its word's bits but MESH_DUD at its target: a dud's landing is one, which its
    wait's check records as a failure) and, where its position's publication reads in place what it writes (`need` the position
   + 1; 0 none), that publication run (the publication cursor past its position).  Every thread runs the item's steps by
   their bodies, and a consumption's end raises its track's count (system scope, release).  A track's cursors and its
   phase (running; its blocked items' failures recorded; ended) are its struct track_state, in the ring.  Each look at a
   blocked track is one load of what it awaits; once the dispatch has looked the program's polls of times since an item
   last ran (a wait's bound: at most that many loads without progress, however many tracks are blocked), every running
   track records its blocked items' failures (mesh_step_check for a landing, else the bound at its position, by its
   claimant: mesh_step_claim), and a track that runs nothing at its next look ends; the dispatch ends once every track
   has, and no track reads another's failure.  The records' layouts are struct track_head, track, track_item, track_op,
   track_dep and track_state below, the host's alike; struct track_op's kind one of KINDS_OF's, the enum the host's */
static NSString *const TRACKS =
  @"\n"
  "enum { " XSTR(KINDS_OF(KIND_ENUM)) " };\n"
  "struct track_op { device uchar *d; device uchar *s; device uchar *a; device uchar *v; mesh_args p; uint kind; uint pad; };\n"
  "struct track_item { ulong position, need, target; device ulong *word; device uchar *v; uint first_op, ops, first_dep, deps, wait_op, cursor; };\n"
  "struct track_dep { ulong count; uint other, pad; };\n"
  "struct track { uint first[2], count[2]; };\n"
  "struct track_state { uint next[2], phase, pad; };\n"
  "struct track_head { ulong tracks_off, items_off, ops_off, deps_off, count_off, state_off, sequence, progress_off; uint tracks, polls; };\n"
  "static inline bool track_ready(constant track_item &it, constant track_dep *deps, volatile coherent(system) device ulong *count, uint t) { "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "if (it.cursor == 0 && count[t] < it.need) return false; "
  "for (uint k = 0; k < it.deps; k++) if (count[deps[it.first_dep + k].other] < deps[it.first_dep + k].count) return false; "
  "if (it.cursor != 1 || it.wait_op == ~0u) return true; "
  "return (*(volatile coherent(system) device const ulong *)it.word & ~MESH_DUD) >= it.target; }\n"
  "static inline void track_expired(constant track_item &it, constant track_op *ops, uint polls) { "
  "if (it.wait_op != ~0u) { constant track_op &w = ops[it.wait_op]; mesh_step_check(w.p, *(volatile coherent(system) device const ulong *)it.word, w.s, w.v); return; } "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)it.v; "
  "if (!v || !mesh_step_claim(it.v)) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "v[0] = " XSTR(METAL_BOUND) "UL; v[4] = ulong(polls); mesh_publish(v + 2, it.position + 1); }\n";
static NSString *const TRACK_KERNEL =
  @"kernel void tracks(constant uchar *base [[buffer(0)]], device uchar *ring [[buffer(1)]], "
  "uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { "
  "constant track_head &h = *(constant track_head *)base; "
  "constant track *trs = (constant track *)(base + h.tracks_off); "
  "constant track_item *items = (constant track_item *)(base + h.items_off); "
  "constant track_op *ops = (constant track_op *)(base + h.ops_off); "
  "constant track_dep *deps = (constant track_dep *)(base + h.deps_off); "
  "volatile coherent(system) device ulong *count = (volatile coherent(system) device ulong *)(ring + h.count_off); "
  "device track_state *state = (device track_state *)(ring + h.state_off); "
  "threadgroup ulong tg[4]; threadgroup uint pick[3]; "
  "uint start = 0, looks = 0; "
  "if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(ring + h.progress_off), h.sequence); "
  "for (;;) { "
  "if (i == 0) { uint chosen = ~0u, served = 0, left = 0; "
  "for (uint k = 0; k < h.tracks && chosen == ~0u; k++) { "
  "const uint t = start + k < h.tracks ? start + k : start + k - h.tracks; "
  "device track_state &s = state[t]; constant track &tr = trs[t]; "
  "if (s.phase == 2) continue; "
  "bool more = false; "
  "if (s.next[1] < tr.count[1]) { more = true; constant track_item &c = items[tr.first[1] + s.next[1]]; "
  "if ((s.next[0] >= tr.count[0] || items[tr.first[0] + s.next[0]].position >= c.need) && track_ready(c, deps, count, t)) chosen = tr.first[1] + s.next[1]++; } "
  "if (chosen == ~0u && s.next[0] < tr.count[0]) { more = true; if (track_ready(items[tr.first[0] + s.next[0]], deps, count, t)) chosen = tr.first[0] + s.next[0]++; } "
  "if (chosen != ~0u) { s.phase = 0; served = t; } "
  "else if (!more || s.phase == 1) s.phase = 2; "
  "else { left = 1; looks++; } } "
  "if (chosen != ~0u) { looks = 0; start = served + 1 < h.tracks ? served + 1 : 0; } "
  "else if (left && looks >= h.polls) { looks = 0; "
  "for (uint t = 0; t < h.tracks; t++) { device track_state &s = state[t]; constant track &tr = trs[t]; "
  "if (s.phase != 0) continue; "
  "if (s.next[1] < tr.count[1]) track_expired(items[tr.first[1] + s.next[1]], ops, h.polls); "
  "if (s.next[0] < tr.count[0]) track_expired(items[tr.first[0] + s.next[0]], ops, h.polls); s.phase = 1; } } "
  "pick[0] = chosen; pick[1] = served; pick[2] = left || chosen != ~0u; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); "
  "const uint chosen = pick[0], t = pick[1]; "
  "if (!pick[2]) return; "
  "if (chosen != ~0u) { constant track_item &it = items[chosen]; "
  "for (uint k = 0; k < it.ops; k++) { constant track_op &o = ops[it.first_op + k]; device ulong *progress = (device ulong *)o.d; "
  "switch (o.kind) { " XSTR(KINDS_OF(KIND_CASE)) " default: break; } "
  "threadgroup_barrier(mem_flags::mem_device); } "
  "if (it.cursor == 1 && i == 0) mesh_publish(count + t, it.position + 1); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); } }\n";

#endif

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
#include "mesh-graph.h"
#include "nccl.h"

#define STR(...) #__VA_ARGS__
#define XSTR(...) STR(__VA_ARGS__)
/* a kernel's arguments, one layout for the host (struct args) and the kernels (struct mesh_args): a wait's count at
   aux until scalar, a publication's count at aux stored scalar, each repeat j of a fused crossing's at scalar + j x
   stride (the packets a repeat adds to the count: nccl.h ncclMeshFused repeats).
   A landing element j needs scalar + ((phase + j * step) >> mask), compiled
   from its packet lengths before execution (mesh-landing.h). Its input is the
   prepared address of the other numerical operand, also for an in-place merge.
   A holding's counts are pages, as words here (nccl-mesh-metal.m wait_args, release_args); aux and first are byte
   offsets, at an entry.
   op is the kernel kind's operand, fixed with the kind at compile time (nccl-mesh.c ktype, kop): a reduction's
   operator (COMBINE_SUM, _PROD, _MAX, _MIN) for a typed kind, and a partials merge's row floats for zelloth_merge
   (METAL_MERGE). */
#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 at, sequence; U32 count, mask, channel, stride; U64 phase, step, input; }

/* Page visibility helpers shared by collective arithmetic and caller kernels.
   Argument records belong to the prepared function; execution never borrows or
   recycles them. Source stores are system-coherent. A source publication orders
   those stores before its ready value; a consumer observes the declared remote
   value before reading the pages. These are data edges, not dispatch ordering.
   Failure words report a lost link or an expired declared observation bound.
   The enclosing function fails; it never selects a replacement executor. */
static NSString *const STEPS =
  @"\nusing namespace metal;\n"
  XSTR(ARGS(mesh_args, ulong, uint)) ";\n"
  "struct mesh_step_binding { device uchar *d; device uchar *s; constant mesh_args *p; device uchar *a; device uchar *v; device ulong *progress; };\n"
  "static inline bool mesh_step_enter(constant mesh_args &p, device uchar *vb, device ulong *progress, bool first) { "
  "(void)p; (void)vb; (void)progress; (void)first; return true; }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, ulong target) { "
  "return mesh_wait_bounded((volatile coherent(system) device const ulong *)word, target, p.polls); }\n"
  "static inline bool mesh_step_claim(device uchar *vb) { "
  "return atomic_exchange_explicit((device atomic_uint *)(vb + 8), 1u, memory_order_relaxed) == 0u; }\n"
  "static inline bool mesh_step_failed(constant mesh_args &p, device uchar *vb) { "
  "return p.polls && atomic_load_explicit((device atomic_uint *)(vb + 8), memory_order_relaxed) != 0u; }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, ulong target, device uchar *vb) { "
  "return mesh_wait_bounded((volatile coherent(system) device const ulong *)word, target, p.polls, p.polls ? (device atomic_uint *)(vb + 8) : nullptr); }\n"
  "static inline void mesh_step_check(uint polls, uint channel, ulong at, ulong status, ulong target, device uchar *vb) { "
  "if (!polls) return; "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; "
  "ulong cause; "
  "if (status != ~0UL && status >= target) return; "
  "cause = status == ~0UL ? ulong(MESH_OBSERVATION_LOST) : v[6] ? ulong(MESH_OBSERVATION_REVOKED) : ulong(MESH_OBSERVATION_BOUND); "
  "if (!mesh_step_claim(vb)) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "v[0] = cause; v[3] = ulong(channel); v[4] = ulong(polls); v[5] = status; mesh_publish(v + 2, at + 1); }\n"
  "static inline void mesh_step_check(constant mesh_args &p, ulong status, ulong target, device uchar *vb) { "
  "mesh_step_check(p.polls, p.channel, p.at, status, target, vb); }\n"
  "static inline void mesh_step_observe(device uchar *word, ulong target, device uchar *vb, uint polls, uint channel, ulong at) { "
  "const ulong status = mesh_wait_bounded((volatile coherent(system) device const ulong *)word, target, polls, "
  "polls ? (device atomic_uint *)(vb + 8) : nullptr); mesh_step_check(polls, channel, at, status, target, vb); }\n"
  "static inline void mesh_step_counts(constant mesh_args &p, device uchar *words, device uchar *vb, uint j = 0) { "
  "const ulong target = p.scalar + ulong(j) * p.stride; "
  "if (p.scalar && !mesh_step_failed(p, vb)) mesh_step_check(p, mesh_step_wait(p, words + p.aux, target, vb), target, vb); }\n"
  "static inline bool mesh_step_await(constant mesh_args &p, device uchar *words, device uchar *vb, uint i, uint j = 0) { "
  "if (i == 0) mesh_step_counts(p, words, vb, j); "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); return true; }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *words, uint i, uint j = 0) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(words + p.aux), p.scalar + ulong(j) * p.stride); }\n"
  "static inline void mesh_source_release(device ulong *words, uint slot, ulong value, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(words + slot), value); }\n"
  /* one tile's landing record as the communicator declaration lays it (nccl.h: { uint64_t count; ncclMeshBinding
     bindings[capacity]; }), at the address the caller's own ncclMeshTileLayout gives that tile: the record's one
     statement is nccl.h's, so this reads no stride or header of its own */
  "static inline void mesh_landing_counts(device const ulong *record) { "
  "device const mesh_step_binding *land = (device const mesh_step_binding *)(record + 1); "
  "for (uint k = 0; k < record[0]; k++) { auto b = land[k]; "
  "mesh_step_enter(*b.p, b.v, b.progress, true); mesh_step_counts(*b.p, b.a, b.v); } }\n"
  "static inline void mesh_landing_await(device const ulong *record, uint i) { "
  "if (!record) return; if (i == 0) mesh_landing_counts(record); "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); }\n"
  "template <typename T> static inline void mesh_step_write(device uchar *operand, ulong k, T value) { "
  "((volatile coherent(system) device T *)operand)[k] = value; }\n"
  "template <typename T = half4> static inline T mesh_step_read(device const uchar *landing, ulong k) { "
  "return mesh_payload((volatile coherent(system) device const T *)landing + k); }\n"
  MESH_GRAPH_PREFIX_METAL;

/* Payload-producing stores are system-coherent: a later SEND may read them before this command buffer ends.
   The accessor contract above applies to the collective's own producers as well as user kernels.
   The kernels, after STEPS: each operator's arithmetic is written once (op_int, op_float, op_f8, merge_row: the partials
   merge, a thread a row), zelloth's fold of a sum; it is the host's (nccl-mesh.c combine_into, merge_rows,
   premultiply, truncdiv): the small floating types in float32 rounded to nearest even, float8 decoded and encoded by
   mesh.metal's f8_to and f8_from, integers wrapping.  zelloth loads both its operand and the landed bytes
   system-coherent: the operand may itself have landed in place (a COPY earlier in the group, mesh-plan.h mesh_compile's
   node outside the contributors), and the landed bytes are the NIC's.  Every kernel but publish and spin is wide (its
   threads over the elements). */
static NSString *const KERNELS =
  @"\n"
  "#define PARAMS device uchar *d, device uchar *s, constant mesh_args &p, device uchar *a, device uchar *vb, device ulong *progress, uint i, uint g, uint l, uint t, threadgroup ulong *tg\n"
  "#define ENTER(first) mesh_step_enter(p, vb, progress, first);\n"
  "#define ENTER_G if (i == 0) mesh_step_enter(p, vb, progress, true);\n"
  "#define HEAD(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]], uint l [[thread_position_in_threadgroup]], uint t [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, l, t, tg); } static void body_##name(PARAMS)\n"
  "#define GROUP(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, i, g, tg); } static void body_##name(PARAMS)\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "template<typename W> static void copy_body(PARAMS) { ENTER(i == 0) "
  "volatile coherent(system) device W *x=(volatile coherent(system) device W *)(d+p.dst); "
  "LOOP x[k]=mesh_step_read<W>(s+p.src,k); }\n"
  "#define COPY(Z, W) HEAD(copy_##Z) { copy_body<W>(d,s,p,a,vb,progress,i,g,l,t,tg); }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "HEAD(fill_pattern) { volatile coherent(system) device uchar *out=(volatile coherent(system) device uchar *)(d+p.dst); "
  "LOOP { const ulong at=k%p.step; out[k]=at<p.input?uchar(p.scalar>>(8*at)):uchar(0); } }\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  /* The cell number is the threadgroup number. Preparation flattened every
     landing partition and threshold; execution reads only those operands. */
  "#define ZFOLD_OP(T, OP, C) T v = mesh_payload(own + j); "
  "for (uint k = 0; k < K; k++) v = OP(v, mesh_payload((volatile coherent(system) device const T *)in[3 * k] + j), C); out[j] = v;\n"
  "#define ZFOLD_MERGE(T, OP, C) const ulong row = j * ulong(p.op); "
  "merge_row<false>(out + row, own + row, (volatile coherent(system) device const float *)in[0] + row, p.op); "
  "for (uint k = 1; k < K; k++) merge_row<true>(out + row, out + row, (volatile coherent(system) device const float *)in[3 * k] + row, p.op);\n"
  "#define ZELLOTH(name, T, OP) ZELLOTH_OF(name##_sum, T, OP, ZFOLD_OP, 0) ZELLOTH_OF(name##_prod, T, OP, ZFOLD_OP, 1) ZELLOTH_OF(name##_max, T, OP, ZFOLD_OP, 2) ZELLOTH_OF(name##_min, T, OP, ZFOLD_OP, 3)\n"
  "struct zelloth_publish { device const ulong *z; uint K; "
  "void at(uint p) { device const ulong *cell = z + z[4] + (2 + K + z[7]) * ulong(p - 1); "
  "for (uint k = 0; k < K; k++) if (z[10 + 3 * k]) "
  "mesh_publish((volatile coherent(system) device ulong *)z[10 + 3 * k], cell[2 + k]); } };\n"
  "#define ZELLOTH_OF(name, T, OP, FOLD, C) HEAD(zelloth_##name) { ENTER(i == 0) "
  "device const ulong *z = (device const ulong *)(s + p.src); const uint K = uint(z[0]), W = uint(z[7]), q = i / t; "
  "device const ulong *cell = z + z[4] + ulong(2 + K + W) * q, *in = z + 8; "
  "if (l == 0) { for (uint k = 0; k < K; k++) "
  "mesh_step_check(p, mesh_step_wait(p, (device uchar *)in[3 * k + 1], cell[2 + k], vb), cell[2 + k], vb); "
  "for (uint k = 0; k < W; k++) if (cell[2 + K + k]) "
  "mesh_step_check(p.polls, uint(in[3 * (K + k) + 1]), in[3 * (K + k) + 2], "
  "mesh_step_wait(p, (device uchar *)in[3 * (K + k)], cell[2 + K + k], vb), cell[2 + K + k], vb); } "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "volatile coherent(system) device const T *own = (volatile coherent(system) device const T *)z[2]; "
  "volatile coherent(system) device T *out = (volatile coherent(system) device T *)z[3]; "
  "for (ulong e = l; e < cell[1]; e += t) { const ulong j = cell[0] + e; FOLD(T, OP, C) } "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (l == 0) { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "device atomic_uint *state = (device atomic_uint *)(z + z[5]), *prefix = (device atomic_uint *)(z + z[6]); "
  "atomic_store_explicit(state + q, 2u, memory_order_relaxed); "
  "zelloth_publish publish = {z, K}; mesh_graph_settle(state, 0, uint(z[1]), prefix, publish); "
  "} }\n"
  "#define INTEGER(name, T, SIGNED) ZELLOTH(name, T, op_int<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); ulong c = ulong(mesh_payload((volatile coherent(system) device const T *)(a + p.aux))); LOOP x[k] = T(ulong(mesh_payload(y + k)) * c); } "
  "HEAD(truncdiv_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP { T v = mesh_payload(y + k); bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) ZELLOTH(name, T, op_float<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); float c = float(mesh_payload((volatile coherent(system) device const T *)(a + p.aux))); LOOP x[k] = T(float(mesh_payload(y + k)) * c); }\n"
  "#define FLOAT8(name, E5) ZELLOTH(name, uchar, op_f8<E5>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device uchar *x = (volatile coherent(system) device uchar *)(d + p.dst); volatile coherent(system) device const uchar *y = (volatile coherent(system) device const uchar *)(s + p.src); float c = f8_to(mesh_payload((volatile coherent(system) device const uchar *)(a + p.aux)), E5); LOOP x[k] = f8_from(f8_to(mesh_payload(y + k), E5) * c, E5); }\n"
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
  "constant float PARTIAL_NONE = -3.4028234663852886e38f;\n"
  "template <bool Same> static inline void merge_row(volatile coherent(system) device float *x, volatile coherent(system) device const float *xs, volatile coherent(system) device const float *y, uint u) { "
  "const float m = mesh_payload(y), ly = mesh_payload(y + 1), mx = mesh_payload(xs), lx = mesh_payload(xs + 1); const bool mine = lx > 0.0f, theirs = ly > 0.0f; "
  "if (mine && !theirs) { if (!Same) for (uint j = 0; j < u; j++) x[j] = mesh_payload(xs + j); return; } "
  "if (!mine || !theirs) { for (uint j = 0; j < u; j++) x[j] = theirs ? mesh_payload(y + j) : j ? 0.0f : PARTIAL_NONE; return; } "
  "const float top = max(mx, m), a = exp(mx - top), b = exp(m - top); x[0] = top; "
  "for (uint j = 1; j < u; j++) x[j] = (a > 0.0f ? mesh_payload(xs + j) * a : 0.0f) + (b > 0.0f ? mesh_payload(y + j) * b : 0.0f); }\n"
  "ZELLOTH_OF(merge, float, 0, ZFOLD_MERGE, 0)\n";

/* every kernel by its kind: one list, the one numbering of a device's pipelines (nccl-mesh-metal.m kernels_of); each
   kernel's name, which the Metal library's lookup takes, once a process (nccl-mesh-metal.m kernels_of) and once an
   install (nccl-mesh-kernels.m) (newFunctionWithName, an interface the library does not implement: design/rubrics.md,
   Strings are messages).  The tracks pool's own kernel, the combine landings it ran and the words and lanes it read
   them by are deleted with it (design/zorvek.md "The pool's last users") */
#define REDUCTION_KINDS(X, T) X(zelloth_##T##_sum) X(zelloth_##T##_prod) X(zelloth_##T##_max) X(zelloth_##T##_min)
#define KINDS_OF(X) \
  X(fill_pattern) X(copy_4) X(copy_1) \
  X(premul_i8) X(truncdiv_i8) X(premul_u8) X(truncdiv_u8) X(premul_i32) X(truncdiv_i32) X(premul_u32) X(truncdiv_u32) \
  X(premul_i64) X(truncdiv_i64) X(premul_u64) X(truncdiv_u64) \
  X(premul_f16) X(premul_f32) X(premul_bf16) X(premul_e4m3) X(premul_e5m2) \
  X(publish) \
  REDUCTION_KINDS(X,i8) REDUCTION_KINDS(X,u8) REDUCTION_KINDS(X,i32) REDUCTION_KINDS(X,u32) REDUCTION_KINDS(X,i64) REDUCTION_KINDS(X,u64) \
  REDUCTION_KINDS(X,f16) REDUCTION_KINDS(X,f32) REDUCTION_KINDS(X,bf16) REDUCTION_KINDS(X,e4m3) REDUCTION_KINDS(X,e5m2) \
  X(zelloth_merge)
#define KIND_ENUM(name) KIND_##name,
#define KIND_NAME(name) #name,
enum { KINDS_OF(KIND_ENUM) KINDS };
static const char *const NAME[KINDS] = {KINDS_OF(KIND_NAME)};


#endif

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
   aux until scalar, a publication's count at aux stored scalar, each repeat j of a fused crossing's at scalar + j x
   stride (the packets a repeat adds to the count: nccl.h ncclMeshFused repeats).
   A landing element j needs scalar + ((phase + j * step) >> mask), compiled
   from its packet lengths before execution (mesh-landing.h). Its input is the
   prepared address of the other numerical operand, also for an in-place merge.
   A holding's counts are pages, as words here (nccl-mesh-metal.m wait_args, release_args); aux and first are byte
   offsets, at an entry.
   op is the kernel kind's operand, fixed with the kind at compile time (nccl-mesh.c ktype, kop): a reduction's
   operator (COMBINE_SUM, _PROD, _MAX, _MIN) for a typed kind, and a partials merge's row floats for land_merge and
   land_merge_from (METAL_MERGE). */
#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 at, sequence; U32 count, mask, channel, stride; U64 phase, step, input; }

/* a step of a compiled program, the library's one implementation of it (metal-microbench docs/acceptance-20261014.md
   A39), after rdma/mesh.metal: libnccl-mesh's kernels call these, and so does a caller's kernel that runs a step its
   group handed it (ncclMeshFused: a product's finish that publishes its own operand, the kernel after the call that
   reads its landing buffer), its six bindings one mesh_step_binding (nccl.h ncclMeshFused's order) where its encoder
   is classic and takes them as bytes (setBytes), its argument table's six where it is Metal 4.  Each guards nothing
   the caller owns: under NCCL's contract the caller touches no buffer of a collective in flight and issues the same
   collectives in the same order on every rank, so the order of a program's reads and writes is known from the programs
   both ranks compiled and is encoded in its tracks' order (rdma/NCCL.md "Per-channel progress"); the host issues no group on a
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
   failure words [0] (its cause, nccl-mesh-metal.h METAL_*), [3], [4] and [5] (the count it last read), and then [2], stored last by the claimant; a
   loser records nothing and waits for nothing: a revoke, the link lost or the bound reached; a landing is no failure,
   whatever its bytes (a dud's among them: placed where a send must happen and its producer is gone, never a value a
   consumer reads as a signal, design/rubrics.md M4).  The host reads the record after the command buffer and fails the
   group as a value (nccl-mesh.c comm_failure); no kernel halts on it.  mesh_step_counts: the record's waits, `landed`
   at p.aux until p.scalar + j x p.stride (repeat j of a fused crossing's records: nccl.h ncclMeshFused repeats; 0, the
   library's own kernels, and a crossing of one repeat), where its target is not 0 and no wait on vb has
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
   holding's `ready` (p.scalar + j x p.stride at p.aux).  mesh_step_read: element k of a landing buffer the
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
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "v[0] = cause; v[3] = ulong(p.channel); v[4] = ulong(p.polls); v[5] = status; mesh_publish(v + 2, p.at + 1); }\n"
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
  /* one tile's landing record as ncclMeshWriteLandings lays it (nccl.h: { uint64_t count; ncclMeshBinding
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
   (mesh_step_failed). A merge uses the same arrival relation for each group of
   complete rows. A track's worker runs a body as one threadgroup of its step's grid, its chunk (TRACK_KERNEL). */
static NSString *const KERNELS =
  @"\n"
  "#define PARAMS device uchar *d, device uchar *s, constant mesh_args &p, device uchar *a, device uchar *vb, device ulong *progress, uint i, uint g, uint l, uint t, threadgroup ulong *tg\n"
  "#define ENTER(first) mesh_step_enter(p, vb, progress, first);\n"
  "#define ENTER_G if (i == 0) mesh_step_enter(p, vb, progress, true);\n"
  "#define HEAD(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]], uint l [[thread_position_in_threadgroup]], uint t [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, l, t, tg); } static void body_##name(PARAMS)\n"
  "#define GROUP(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, i, g, tg); } static void body_##name(PARAMS)\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
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
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); }\n"
  "#define LANDS(T, OP) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *xs = (volatile coherent(system) device const T *)p.input; "
  "volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); "
  "const ulong block = 4 * ulong(t), step = 4 * ulong(g); ulong seen = 0; uint looked = 0; "
  "if ((p.dst | p.src | p.input) % (4 * sizeof(T)) == 0) for (ulong b = 4 * ulong(i - l); b < p.n; b += step) { "
  "mesh_pages_await(p, a, vb, tg, l, p.scalar + ((p.phase + (min(b + block, p.n) - 1) * p.step) >> p.mask), seen, looked); const ulong k = b + 4 * l; "
  "if (k + 4 <= p.n) { const vec<T, 4> u = mesh_payload((volatile coherent(system) device const vec<T, 4> *)(xs + k)), w = mesh_payload((volatile coherent(system) device const vec<T, 4> *)(y + k)); "
  "vec<T, 4> r; for (int c = 0; c < 4; c++) r[c] = OP(u[c], w[c], p.op); *(volatile coherent(system) device vec<T, 4> *)(x + k) = r; } "
  "else for (ulong e = k; e < p.n; e++) x[e] = OP(mesh_payload(xs + e), mesh_payload(y + e), p.op); } "
  "else for (ulong b = 4 * ulong(i - l); b < p.n; b += step) { const ulong end = min(b + block, p.n); "
  "mesh_pages_await(p, a, vb, tg, l, p.scalar + ((p.phase + (end - 1) * p.step) >> p.mask), seen, looked); "
  "for (ulong e = b + l; e < end; e += t) x[e] = OP(mesh_payload(xs + e), mesh_payload(y + e), p.op); }\n"
  "#define COMBINE(name, T, OP) HEAD(land_##name) { ENTER(i == 0) LANDS(T, OP) }\n"
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
  "template <bool Same> static inline void merge_row(volatile coherent(system) device float *x, volatile coherent(system) device const float *xs, volatile coherent(system) device const float *y, uint u) { "
  "const float m = mesh_payload(y), ly = mesh_payload(y + 1), mx = mesh_payload(xs), lx = mesh_payload(xs + 1); const bool mine = lx > 0.0f, theirs = ly > 0.0f; "
  "if (mine && !theirs) { if (!Same) for (uint j = 0; j < u; j++) x[j] = mesh_payload(xs + j); return; } "
  "if (!mine || !theirs) { for (uint j = 0; j < u; j++) x[j] = theirs ? mesh_payload(y + j) : j ? 0.0f : PARTIAL_NONE; return; } "
  "const float top = max(mx, m), a = exp(mx - top), b = exp(m - top); x[0] = top; "
  "for (uint j = 1; j < u; j++) x[j] = (a > 0.0f ? mesh_payload(xs + j) * a : 0.0f) + (b > 0.0f ? mesh_payload(y + j) * b : 0.0f); }\n"
  "template <bool Same> static void land_merge_body(PARAMS) { ENTER(i == 0) volatile coherent(system) device float *x = (volatile coherent(system) device float *)(d + p.dst); "
  "volatile coherent(system) device const float *xs = (volatile coherent(system) device const float *)p.input; "
  "volatile coherent(system) device const float *y = (volatile coherent(system) device const float *)(s + p.src); ulong seen = 0; uint looked = 0; "
  "for (ulong b = i - l; b < p.n; b += g) { const ulong end = min(b + t, p.n); "
  "mesh_pages_await(p, a, vb, tg, l, p.scalar + ((p.phase + (end - 1) * p.step) >> p.mask), seen, looked); "
  "const ulong r = b + l; const uint u = p.op; /* the merge's row floats (ARGS op) */ "
  "if (r < end) merge_row<Same>(x + r * u, xs + r * u, y + r * u, u); } }\n"
  "HEAD(land_merge) { land_merge_body<true>(d, s, p, a, vb, progress, i, g, l, t, tg); }\n"
  "HEAD(land_merge_from) { land_merge_body<false>(d, s, p, a, vb, progress, i, g, l, t, tg); }\n"
  "GROUP(present) { ENTER_G const ulong end = p.scalar + p.n; "
  "if (i == 0) { const ulong need = end > p.count ? end - p.count : 0; "
  "const ulong got = mesh_step_wait(p, a + p.aux, need, vb); mesh_step_check(p, got, need, vb); "
  /* the list's count at this group's first entry, its predecessor's end, before this group's end is stored: the list
     presented in issue order, as the host's mesh_stream_present presented it at the count it read (whatever encoder or
     command buffer each group's present runs in) */
  "const ulong prior = got >= need ? mesh_step_wait(p, a + p.at, p.scalar, vb) : got; mesh_step_check(p, prior, p.scalar, vb); "
  "tg[0] = got >= need && prior >= p.scalar; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); if (!tg[0]) return; "
  "volatile coherent(system) device ulong *dst = (volatile coherent(system) device ulong *)(d + p.dst); "
  "device const ulong *src = (device const ulong *)(s + p.src); "
  "for (ulong k = i; k < p.n * 4; k += g) dst[((p.scalar + k / 4) % p.count) * 4 + k % 4] = src[k]; "
  "threadgroup_barrier(mem_flags::mem_device); if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(a + p.at), end); }\n";

/* every kernel a track may record, by its kind: one list, the one numbering of the host's steps (struct host_op kind,
   a device's pipelines' index: nccl-mesh-metal.m kernels_of) and of the track kernel's switch (TRACKS' enum and TRACK_KERNEL's cases, made
   from this list by the preprocessor, so neither names a kind the other lacks); each kernel's name, which the Metal
   library's lookup takes, once a process (nccl-mesh-metal.m kernels_of) and once an install (nccl-mesh-kernels.m)
   (newFunctionWithName, an interface the library does not implement: design/rubrics.md, Strings are messages) and whose body the switch calls; the word
   its step awaits (metal_tracks: WORD_D a spin's count, in its d; WORD_A a landing's pages, in its a; 0 none); and a
   landing's elements a thread of its block (LANES: 4 a combine's, 1 a merge's rows; 0 a step that awaits no pages), so
   a landing's chunk is one block and the scheduler reads the pages that block awaits (TRACKS track_want) */
enum { WORD_D = 1, WORD_A };
#define KINDS_OF(X) \
  X(copy_plain_4, 0, 0) X(copy_land_4, 0, 0) X(copy_plain_1, 0, 0) X(copy_land_1, 0, 0) \
  X(land_i8, WORD_A, 4) X(premul_i8, 0, 0) X(truncdiv_i8, 0, 0) \
  X(land_u8, WORD_A, 4) X(premul_u8, 0, 0) X(truncdiv_u8, 0, 0) \
  X(land_i32, WORD_A, 4) X(premul_i32, 0, 0) X(truncdiv_i32, 0, 0) \
  X(land_u32, WORD_A, 4) X(premul_u32, 0, 0) X(truncdiv_u32, 0, 0) \
  X(land_i64, WORD_A, 4) X(premul_i64, 0, 0) X(truncdiv_i64, 0, 0) \
  X(land_u64, WORD_A, 4) X(premul_u64, 0, 0) X(truncdiv_u64, 0, 0) \
  X(land_f16, WORD_A, 4) X(premul_f16, 0, 0) X(land_f32, WORD_A, 4) X(premul_f32, 0, 0) \
  X(land_bf16, WORD_A, 4) X(premul_bf16, 0, 0) X(land_e4m3, WORD_A, 4) X(premul_e4m3, 0, 0) \
  X(land_e5m2, WORD_A, 4) X(premul_e5m2, 0, 0) \
  X(publish, 0, 0) X(spin, WORD_D, 0) X(land_merge, WORD_A, 1) X(land_merge_from, WORD_A, 1) X(present, 0, 0)
#define KIND_ENUM(name, word, lanes) KIND_##name,
#define KIND_NAME(name, word, lanes) #name,
#define KIND_WORD(name, word, lanes) word,
#define KIND_LANES(name, word, lanes) lanes,
#define KIND_CASE(name, word, lanes) case KIND_##name: body_##name(o.d, o.s, o.p, o.a, o.v, progress, i, g, l, t, tg); break;
/* the kinds, then the track kernel's own pipeline (KIND_tracks: no track records it) */
enum { KINDS_OF(KIND_ENUM) KINDS, KIND_tracks = KINDS, PIPELINES };
static const char *const NAME[PIPELINES] = {KINDS_OF(KIND_NAME) "tracks"};
static const uint8_t WORD[KINDS] = {KINDS_OF(KIND_WORD)};
static const uint8_t LANES[KINDS] = {KINDS_OF(KIND_LANES)};

/* the track kernel (nccl-mesh-metal.h metal_tracks; rdma/NCCL.md "Per-channel progress"): every completion's work, on
   any number of tracks, run by a pool of workers that no track owns, at most one a GPU core (metal_tracks).  A worker
   is a threadgroup; its thread 0, after one system-scope fence a pass (so the pass's loads see the NIC's and the other
   workers' stores), first looks at the task it last ran (the next task of its cursor where it moved that cursor), and
   else scans the cursors (each track's publications, then its consumptions, from a track that differs worker by
   worker) for the current task of one whose conditions hold, and claims a chunk of it: an atomic count of the task's chunks claimed,
   compared and exchanged where the chunk's own pages decide (a landing's, TASK_PACED: no worker claims a chunk whose
   pages have not landed, so none ever waits inside a step for data), else added to.  Every thread then runs the
   chunk as threadgroup c of its step's own grid (i = c t + l, g = the step's chunks x t: the body a dispatch of that
   grid runs, unchanged), and the chunk's end is added to the task's finished count; the worker that finishes its last
   chunk raises its track's count where it ends a consumption and moves the cursor to its next task, a fence between
   the two: the next task's worker stores the same count, and its store follows this one only where this one is
   ordered before the cursor's move (without it the count's store, system-coherent, reached memory after the next
   consumption's, and a track's count stood one short of its last consumption with every cursor of it ended: m5ult1's
   all-gather relay awaiting count 530 at 529, 2026-10-30).  A task is a run
   of an item's steps that may run at once (a landing's runs), its first a consumption's or publication's conditions:
   its dependencies (another track's count), a consumption's `need` (its track's publications past that send entry),
   and every chunk's awaited word (a spin's count; a landing chunk's pages, track_want).  A worker that claims nothing
   for the program's polls of readiness checks (each a load of an awaited word or count, as a wait's bound counts its
   loads) since any chunk last finished records each blocked task's failure (mesh_step_check
   for its awaited word, else the bound at its position, by its claimant: mesh_step_claim) and ends that cursor, as a
   wait ends at its bound; read every 2^10 idle passes, a cursor whose item's failure claim is taken (a wait that
   failed, or the host's stop: nccl-mesh.c mesh_library_expired) ends then, its task's claimed count raised past every
   chunk (2^31) so no worker claims one.  A worker ends once every cursor has.  The
   records' layouts are struct track_head, track_cursor, track_task, track_item, track_op and track_dep below, the
   host's alike; struct track_op's kind one of KINDS_OF's, the enum the host's */
static NSString *const TRACKS =
  @"\n"
  "enum { " XSTR(KINDS_OF(KIND_ENUM)) " };\n"
  "struct track_op { device uchar *d; device uchar *s; device uchar *a; device uchar *v; mesh_args p; device ulong *word; ulong block; uint kind, chunks; };\n"
  "struct track_task { uint first_op, ops, chunks, item, cursor, flags, pad0, pad1; };\n"
  "struct track_item { ulong position, need; device uchar *v; uint first_dep, deps; };\n"
  "struct track_dep { ulong count; uint other, pad; };\n"
  "struct track_cursor { uint first, end, track, kind; };\n"
  "struct track_head { ulong cursors_off, tasks_off, items_off, ops_off, deps_off, count_off, state_off, claim_off, done_off, progress_off, ended_off; uint cursors, polls; };\n"
  "static inline uint track_op_of(constant track_task &x, constant track_op *ops, thread uint &c) { "
  "uint k = x.first_op; while (c >= ops[k].chunks) { c -= ops[k].chunks; k++; } return k; }\n"
  "static inline ulong track_want(constant track_op &o, uint c) { "
  "return o.p.scalar + (o.block ? ((o.p.phase + (min(ulong(c + 1) * o.block, o.p.n) - 1) * o.p.step) >> o.p.mask) : 0UL); }\n"
  "static inline bool track_ready(constant track_task &x, uint c, constant track_cursor *cursors, constant track_task *tasks, "
  "constant track_item *items, constant track_op *ops, constant track_dep *deps, volatile coherent(system) device ulong *count, "
  "device atomic_uint *state) { "
  "if (x.flags & 1u) { constant track_item &it = items[x.item]; "
  "for (uint k = 0; k < it.deps; k++) if (count[deps[it.first_dep + k].other] < deps[it.first_dep + k].count) return false; "
  "if (cursors[x.cursor].kind == 1u) { const uint q = x.cursor - 1, j = atomic_load_explicit(state + q, memory_order_relaxed); "
  "if (j < cursors[q].end && items[tasks[j].item].position < it.need) return false; } } "
  "if (!x.ops) return true; "
  "constant track_op &o = ops[track_op_of(x, ops, c)]; "
  "return !o.word || *(volatile coherent(system) device const ulong *)o.word >= track_want(o, c); }\n"
  "static inline void track_expired(constant track_task &x, uint c, constant track_item &it, constant track_op *ops, uint polls) { "
  "if (x.ops) { constant track_op &o = ops[track_op_of(x, ops, c)]; "
  "if (o.word) mesh_step_check(o.p, *(volatile coherent(system) device const ulong *)o.word, track_want(o, c), o.v); } "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)it.v; "
  "if (!v || !mesh_step_claim(it.v)) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "v[0] = " XSTR(METAL_BOUND) "UL; v[4] = ulong(polls); mesh_publish(v + 2, it.position + 1); }\n";
static NSString *const TRACK_KERNEL =
  @"kernel void tracks(constant uchar *base [[buffer(0)]], device uchar *ring [[buffer(1)]], uint w [[threadgroup_position_in_grid]], "
  "uint l [[thread_position_in_threadgroup]], uint t [[threads_per_threadgroup]]) { "
  "constant track_head &h = *(constant track_head *)base; "
  "constant track_cursor *cursors = (constant track_cursor *)(base + h.cursors_off); "
  "constant track_task *tasks = (constant track_task *)(base + h.tasks_off); "
  "constant track_item *items = (constant track_item *)(base + h.items_off); "
  "constant track_op *ops = (constant track_op *)(base + h.ops_off); "
  "constant track_dep *deps = (constant track_dep *)(base + h.deps_off); "
  "volatile coherent(system) device ulong *count = (volatile coherent(system) device ulong *)(ring + h.count_off); "
  "device atomic_uint *state = (device atomic_uint *)(ring + h.state_off), *claim = (device atomic_uint *)(ring + h.claim_off), "
  "*done = (device atomic_uint *)(ring + h.done_off), *advanced = (device atomic_uint *)(ring + h.progress_off); "
  "threadgroup ulong tg[4]; threadgroup uint pick[3]; "
  "const uint tracks = h.cursors / 2; ulong looks = 0; uint seen = 0, last = ~0u; "
  "for (;;) { "
  "if (l == 0) { uint chosen = ~0u, chunk = 0, open = 0, checked = 0; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "if (last != ~0u) { constant track_task &x = tasks[last]; uint c = atomic_load_explicit(claim + last, memory_order_relaxed); "
  "if (c < x.chunks && track_ready(x, c, cursors, tasks, items, ops, deps, count, state) && ((x.flags & 4u) "
  "? atomic_compare_exchange_weak_explicit(claim + last, &c, c + 1, memory_order_relaxed, memory_order_relaxed) "
  ": (c = atomic_fetch_add_explicit(claim + last, 1u, memory_order_relaxed)) < x.chunks)) { chosen = last; chunk = c; open = 1; } } "
  "for (uint kind = 0; kind < 2 && chosen == ~0u; kind++) for (uint k = 0; k < tracks && chosen == ~0u; k++) { "
  "const uint cu = 2 * ((k + w) % tracks) + kind, j = atomic_load_explicit(state + cu, memory_order_relaxed); "
  "if (j >= cursors[cu].end) continue; "
  "open = 1; checked++; constant track_task &x = tasks[j]; uint c = atomic_load_explicit(claim + j, memory_order_relaxed); "
  "if (c >= x.chunks || !track_ready(x, c, cursors, tasks, items, ops, deps, count, state)) continue; "
  "if ((x.flags & 4u) ? atomic_compare_exchange_weak_explicit(claim + j, &c, c + 1, memory_order_relaxed, memory_order_relaxed) "
  ": (c = atomic_fetch_add_explicit(claim + j, 1u, memory_order_relaxed)) < x.chunks) { chosen = j; chunk = c; } } "
  "if (chosen != ~0u) looks = 0; "
  "else if (open) { const uint now = atomic_load_explicit(advanced, memory_order_relaxed); "
  "if (now != seen) { seen = now; looks = 0; } "
  "const ulong before = looks; looks += checked; "
  "const bool bound = looks >= h.polls, stop = before / MESH_STOP_POLL_PERIOD != looks / MESH_STOP_POLL_PERIOD; "
  "for (uint cu = 0; (bound || stop) && cu < h.cursors; cu++) { "
  "const uint j = atomic_load_explicit(state + cu, memory_order_relaxed); "
  "if (j >= cursors[cu].end || atomic_load_explicit(claim + j, memory_order_relaxed) >= tasks[j].chunks) continue; "
  "constant track_task &x = tasks[j]; constant track_item &it = items[x.item]; "
  "if (!bound && !(it.v && atomic_load_explicit((device atomic_uint *)(it.v + 8), memory_order_relaxed) != 0u)) continue; "
  "const uint c = atomic_fetch_add_explicit(claim + j, 0x80000000u, memory_order_relaxed); "
  "if (c >= x.chunks) continue; "
  "track_expired(x, c, it, ops, h.polls); atomic_store_explicit(state + cu, cursors[cu].end, memory_order_relaxed); } } "
  "pick[0] = chosen; pick[1] = chunk; pick[2] = open; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); "
  "const uint j = pick[0]; uint c = pick[1]; "
  "if (j == ~0u) { if (!pick[2]) { if (l == 0) mesh_publish((volatile coherent(system) device ulong *)(ring + h.ended_off) + w, 1); return; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); continue; } "
  "constant track_task &x = tasks[j]; "
  "if (x.ops) { constant track_op &o = ops[track_op_of(x, ops, c)]; device ulong *progress = (device ulong *)o.d; "
  "const uint i = c * t + l, g = o.chunks * t; "
  "switch (o.kind) { " XSTR(KINDS_OF(KIND_CASE)) " default: break; } } "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (l == 0) { last = j; atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "atomic_fetch_add_explicit(advanced, 1u, memory_order_relaxed); "
  "if (atomic_fetch_add_explicit(done + j, 1u, memory_order_relaxed) + 1u == x.chunks) { "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "constant track_cursor &cr = cursors[x.cursor]; "
  "if ((x.flags & 2u) && cr.kind == 1u) { mesh_publish(count + cr.track, items[x.item].position + 1); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); } "
  "atomic_store_explicit(state + x.cursor, j + 1, memory_order_relaxed); last = j + 1 < cr.end ? j + 1 : ~0u; } } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); } }\n";

#endif

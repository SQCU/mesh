#import <Metal/Metal.h>
#include "nccl-mesh-metal.h"
#include "nccl.h"
#pragma GCC visibility push(hidden)
#include "mesh_metal.h"
#pragma GCC visibility pop
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define STR(...) #__VA_ARGS__
#define XSTR(...) STR(__VA_ARGS__)
/* a kernel's arguments, one layout for the host (struct args) and the kernels (struct mesh_args) */
#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 first, slot, at, sequence; U32 count, mask, channel; }
/* the loads a wait makes between its reads of its communicator's stop word (failure word [6], a revoke's) */
#define STEP_LOOK 4096u

/* a step of a compiled program, the library's one implementation of it (metal-microbench docs/acceptance-20261014.md
   A39), after rdma/mesh.metal: libnccl-mesh's kernels call these, and so does a caller's kernel that runs a step its
   group handed it (ncclMeshFused: a product's finish that publishes its piece, a layer's finish that lands its peer's).
   mesh_step_enter: in the Metal 4 form a kernel first stores its record's sequence in `progress` (every dispatch before
   it has finished: a barrier stands between each two), whether or not its program has failure words (the classic
   form's sequence is 0, and it stores none), and, where it has them (p.polls, set wherever they are), goes on only
   while failure word [2], the gate, holds none (only the GPU's waits write it, so every thread of a dispatch reads it
   alike).  mesh_step_wait: a wait on its run's last unit's landed word at p.aux for 1 + p.scalar, at most p.polls
   loads (mesh_wait_bounded, STEP_LOOK at a time with the stop word [6] read between them; the program's bound, at most
   NCCL_MESH_POLLS: metal_begin), ended sooner by the landing, the link's loss (~0) or a revoke.  mesh_step_check: the
   wait's failure recorded in failure words [0] (its cause, nccl-mesh-metal.h METAL_*), [3] and [4] and then the gate
   [2], where the gate is still clear: a revoke, the link lost, the bound reached, or a dud, its run's first eight bytes
   (at p.src) all ones (the peer supplied nothing there: design/transport.md section 10); a landing reads its first
   eight bytes alone, as HEAD's stamp check read them, and the failure words and the fence only where it failed.  mesh_step_await: a landing
   awaited by thread 0 and checked, then the threadgroup past a device barrier and a system-scope fence (its payload
   then read with mesh_payload).  mesh_step_release: a piece the threadgroup stored into its run's slots, then its
   units' ready words (p.count from p.aux, each p.scalar) released by thread 0 after a device barrier.  One threadgroup
   a step.  A caller's kernel stores its piece's element k with mesh_step_store and reads its peer's fourth k with
   mesh_step_read (float16), so it names no transport qualifier of its own (they are compiled here, under mesh.metal's
   pragma: a header after it, MetalPerformancePrimitives', may end that pragma).  The library's own kernels that write a
   stream class's out ring or ready words (a send, a release) do nothing once the stop word is set (STOP, GROUP_STOP: read by
   thread 0 alone where the threadgroup meets a barrier), so none writes them after a revoke. */
static NSString *const STEPS =
  @"\nusing namespace metal;\n"
  XSTR(ARGS(mesh_args, ulong, uint)) ";\n"
  "static inline bool mesh_step_enter(constant mesh_args &p, device uchar *vb, device ulong *progress, bool first) { "
  "if (p.sequence && first) *progress = p.sequence; return !(p.polls && ((volatile device const ulong *)vb)[2]); }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, device uchar *vb) { "
  "volatile coherent(system) device const ulong *w = (volatile coherent(system) device const ulong *)word, *v = (volatile coherent(system) device const ulong *)vb; "
  "ulong seen = mesh_wait_bounded(w, 1 + p.scalar, min(p.polls, " XSTR(STEP_LOOK) ")); "
  "for (uint k = " XSTR(STEP_LOOK) "; seen < 1 + p.scalar && k < p.polls && !v[6]; k += " XSTR(STEP_LOOK) ") "
  "seen = mesh_wait_bounded(w, 1 + p.scalar, min(p.polls - k, " XSTR(STEP_LOOK) ")); return seen; }\n"
  "static inline void mesh_step_check(constant mesh_args &p, ulong status, device uchar *ring, device uchar *vb) { "
  "if (!p.polls) return; "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; "
  "ulong cause; "
  "if (status != ~0UL && status >= 1 + p.scalar) { "
  "if (mesh_payload((volatile coherent(system) device const ulong *)(ring + p.src)) != ~0UL) return; cause = " XSTR(METAL_DUD) "UL; } "
  "else cause = status == ~0UL ? " XSTR(METAL_LOST) "UL : v[6] ? " XSTR(METAL_REVOKED) "UL : " XSTR(METAL_BOUND) "UL; "
  "if (v[2]) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "v[0] = cause; v[3] = ulong(p.channel); v[4] = ulong(p.polls); mesh_publish(v + 2, p.at + 1); }\n"
  "static inline bool mesh_step_await(constant mesh_args &p, device uchar *words, device uchar *ring, device uchar *vb, uint i) { "
  "if (i == 0) mesh_step_check(p, mesh_step_wait(p, words + p.aux, vb), ring, vb); "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "return !(p.polls && ((volatile device const ulong *)vb)[2]); }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *words, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0) for (uint u = 0; u < p.count; u++) mesh_publish((volatile coherent(system) device ulong *)(words + p.aux) + u, p.scalar); }\n"
  "static inline void mesh_step_store(constant mesh_args &p, device uchar *slot, ulong k, half value) { "
  "((volatile coherent(system) device half *)(slot + p.dst))[k] = value; }\n"
  "static inline half4 mesh_step_read(constant mesh_args &p, device uchar *slot, ulong k) { "
  "return mesh_payload((volatile coherent(system) device const half4 *)(slot + p.src) + k); }\n";

/* the kernels, after STEPS: each operator's arithmetic is written once (op_int, op_float, op_f8, merge_row: the partials
   merge, a thread a row), a landing being the wait and then the combine's body; it is the host's (nccl-mesh.c
   combine_into, merge_rows, premultiply, truncdiv): the small floating types in float32 rounded to
   nearest even, float8 decoded and encoded by mesh.metal's f8_to and f8_from, integers wrapping */
static NSString *const KERNELS =
  @"\n"
  "#define PARAMS device uchar *d, device uchar *s, constant mesh_args &p, device uchar *a, device uchar *vb, device ulong *progress, uint i, uint g, threadgroup ulong *tg\n"
  "#define ENTER(first) if (!mesh_step_enter(p, vb, progress, first)) return;\n"
  "#define ENTER_G if (i == 0) tg[3] = mesh_step_enter(p, vb, progress, true) ? 0UL : 1UL; threadgroup_barrier(mem_flags::mem_threadgroup); if (tg[3]) return;\n"
  "#define STOPPED (p.polls && ((volatile coherent(system) device const ulong *)vb)[6] != 0UL)\n"
  "#define STOP if (STOPPED) return;\n"
  "#define GROUP_STOP if (i == 0) tg[2] = STOPPED ? 1UL : 0UL; threadgroup_barrier(mem_flags::mem_threadgroup); if (tg[2]) return;\n"
  "#define HEAD(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) { body_##name(d, s, p, d, vb, progress, i, g, nullptr); } static void body_##name(PARAMS)\n"
  "#define HEAD_A(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) { body_##name(d, s, p, a, vb, progress, i, g, nullptr); } static void body_##name(PARAMS)\n"
  "#define GROUP(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, vb, progress, i, g, tg); } static void body_##name(PARAMS)\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define AWAIT if (!mesh_step_await(p, a, s, vb, i)) return;\n"
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { ENTER(i == 0) device W *x = (device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { ENTER(i == 0) device W *x = (device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); } "
  "HEAD(copy_send_##Z) { ENTER(i == 0) STOP volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; }\n"
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
  "GROUP(publish) { ENTER_G GROUP_STOP mesh_step_release(p, d, i); }\n"
  "GROUP(publish_filler) { ENTER_G GROUP_STOP if (i == 0) *(volatile coherent(system) device ulong *)(d + p.dst) = 0UL; mesh_step_release(p, d, i); }\n"
  "GROUP(send_small) { ENTER_G GROUP_STOP volatile coherent(system) device uchar *x = (volatile coherent(system) device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); "
  "if (i == 0 && p.n < 8) *(volatile coherent(system) device ulong *)x = 0UL; threadgroup_barrier(mem_flags::mem_device); LOOP x[k] = y[k]; "
  "mesh_step_release(p, a, i); }\n"
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
  "if (over && p.mask) mesh_publish((volatile coherent(system) device ulong *)vb + 5, 1UL); "
  "if (*word == 0) mesh_publish(word, over ? 2UL : 1UL); } }\n"
  "#define PLACE(BEFORE, ROWS, LIMIT) threadgroup ulong *place = tg; if (i == 0) { ulong at = (BEFORE) * p.slot + p.scalar, rows = (ROWS) * p.slot; "
  "ulong bytes = rows > p.scalar ? min(rows - p.scalar, p.n) : 0UL; place[0] = at; place[1] = at >= (LIMIT) ? 0UL : min(bytes, (LIMIT) - at); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "#define COUNTED(Z, W) "
  "GROUP(counted_send_##Z) { ENTER_G GROUP_STOP device const long *c = (device const long *)(a + p.aux); PLACE(counted_rows(c, 0, p.first), counted_rows(c, p.first, p.count), p.at) "
  "volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src + place[0]); "
  "if (i == 0 && place[1] < 8) *(volatile coherent(system) device ulong *)x = 0UL; threadgroup_barrier(mem_flags::mem_device); "
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

/* every kernel a track may record, by its index (struct op kind): the track kernel's switch calls each one's body */
static const char *const KIND[] = {
  "copy_plain_4", "copy_land_4", "copy_send_4", "copy_plain_1", "copy_land_1", "copy_send_1",
  "combine_i8", "land_i8", "combine3_i8", "premul_i8", "truncdiv_i8", "combine_u8", "land_u8", "combine3_u8", "premul_u8", "truncdiv_u8",
  "combine_i32", "land_i32", "combine3_i32", "premul_i32", "truncdiv_i32", "combine_u32", "land_u32", "combine3_u32", "premul_u32", "truncdiv_u32",
  "combine_i64", "land_i64", "combine3_i64", "premul_i64", "truncdiv_i64", "combine_u64", "land_u64", "combine3_u64", "premul_u64", "truncdiv_u64",
  "combine_f16", "land_f16", "combine3_f16", "premul_f16", "combine_f32", "land_f32", "combine3_f32", "premul_f32",
  "combine_bf16", "land_bf16", "combine3_bf16", "premul_bf16", "combine_e4m3", "land_e4m3", "combine3_e4m3", "premul_e4m3",
  "combine_e5m2", "land_e5m2", "combine3_e5m2", "premul_e5m2",
  "publish", "publish_filler", "send_small", "land_copy", "spin",
  "counted_start", "counted_check", "counted_arrived", "counted_send_4", "counted_land_4", "counted_own_4", "counted_send_1", "counted_land_1",
  "counted_own_1", "combine_merge", "land_merge"};
enum { KINDS = sizeof KIND / sizeof *KIND };

/* the track kernel (nccl-mesh-metal.h metal_tracks; rdma/NCCL.md "Per-channel progress"), after KERNELS: one thread
   group a track, its thread 0 choosing the next item of either cursor whose credit, dependencies and (a consumption)
   landing hold (a consumption first), every thread running the item's steps by their bodies, and a consumption's end
   raising the track's count (system scope, release); an item whose failure words hold a failure runs as its bodies do
   (nothing); a group that chooses nothing for the program's polls of passes records its blocked items' failures
   (mesh_step_check for a landing, else the bound at its position) and ends once nothing more can run.  The records'
   layouts are struct track_head, track, track_item, track_op and track_dep below, the host's alike */
static NSString *const TRACKS =
  @"\n"
  "struct track_op { device uchar *d; device uchar *s; device uchar *a; device uchar *v; mesh_args p; uint kind; uint pad; };\n"
  "struct track_item { ulong position, need, target; device ulong *word; device uchar *v; uint first_op, ops, first_dep, deps, wait_op, cursor; };\n"
  "struct track_dep { ulong count; uint other, pad; };\n"
  "struct track { uint first[2], count[2]; };\n"
  "struct track_head { ulong tracks_off, items_off, ops_off, deps_off, count_off, sequence, progress_off; uint tracks, polls; };\n"
  "static inline bool track_ready(constant track_item &it, constant track_dep *deps, volatile coherent(system) device ulong *count, uint t) { "
  "volatile coherent(system) device const ulong *v = (volatile coherent(system) device const ulong *)it.v; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "if (v && v[2]) return true; "
  "if (count[t] < it.need) return false; "
  "for (uint k = 0; k < it.deps; k++) if (count[deps[it.first_dep + k].other] < deps[it.first_dep + k].count) return false; "
  "if (it.cursor != 1 || it.wait_op == ~0u) return true; "
  "if (v && v[6]) return true; "
  "return *(volatile coherent(system) device const ulong *)it.word >= it.target; }\n"
  "static inline void track_expired(constant track_item &it, constant track_op *ops, uint polls) { "
  "if (it.wait_op != ~0u) { constant track_op &w = ops[it.wait_op]; mesh_step_check(w.p, *(volatile coherent(system) device const ulong *)it.word, w.s, w.v); return; } "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)it.v; "
  "if (!v || v[2]) return; "
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
  "if (next1 < tr.count[1]) { left = 1; if (track_ready(items[tr.first[1] + next1], deps, count, t)) { chosen = tr.first[1] + next1; next1++; } } "
  "if (chosen == ~0u && next0 < tr.count[0]) { left = 1; if (track_ready(items[tr.first[0] + next0], deps, count, t)) { chosen = tr.first[0] + next0; next0++; } } "
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
  "switch (o.kind) { %@ default: break; } "
  "threadgroup_barrier(mem_flags::mem_device); } "
  "if (it.cursor == 1 && i == 0) mesh_publish(count + t, it.position + 1); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); } }\n";

/* ncclDataType_t's order: int8 uint8 int32 uint32 int64 uint64 float16 float32 float64 bfloat16 e4m3 e5m2, then the
   partials merge's rows (METAL_MERGE) */
static const char *const TYPE[METAL_MERGE + 1] = {"i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", NULL, "bf16", "e4m3", "e5m2", "merge"};
static const char *const MODE[3] = {"plain", "land", "send"};

ARGS(args, uint64_t, uint32_t);
/* a program: the classic form's command buffer and compute encoder (made here, or the caller's: `borrowed`) and what it
   keeps until the program's end; or the Metal 4 form's caller's encoder, argument table and residency set, the
   device's argument ring, and whether a buffer was made resident since the set's last commit; and either form's
   failure words, those of the communicator whose group it encodes (metal_gate; none: nothing recorded or halted), and
   its waits' bound (`polls`, never 0) */
struct ring;
struct recording;
struct metal_program { id<MTLCommandBuffer> buffer; id<MTLComputeCommandEncoder> encoder; NSMutableArray *kept; int borrowed;
                       id<MTL4ComputeCommandEncoder> encoder4; id<MTL4ArgumentTable> table; id<MTLResidencySet> residency;
                       id<MTLBuffer> violation; struct ring *ring; int added; uint32_t polls; struct recording *recording; };

/* a device's kernels: its library and pipelines by name; and its argument ring (the Metal 4 form's arguments: RECORDS
   records of RECORD bytes in `records`, the `next` one written, which the GPU's dispatch of sequence q reads from record
   (q - 1) mod RECORDS, `progress` the sequence of the last dispatch the GPU began, every one before it finished) */
#define RECORDS 65536u
#define RECORD 128u
struct ring { id<MTLBuffer> records, progress; uint64_t next; };
/* a device's track ring (metal_tracks): TRACK_RING bytes, its first 64 the progress word the track kernel stores its
   dispatch's sequence into as it begins (every dispatch before it finished), then each dispatch's records in turn,
   `held` the spans still read: a span of sequence q is free once the progress word reads more than q */
#define TRACK_RING (64u << 20)
struct span { uint64_t sequence, first, end; };
struct track_ring { id<MTLBuffer> buffer; uint64_t sequence, head; struct span *held; size_t helds, room; };
struct kernels { id<MTLLibrary> library; NSMutableDictionary *pipelines; struct ring ring; struct track_ring tracks; };
_Static_assert(sizeof(struct args) <= RECORD, "an argument record");

static id<MTLDevice> device_;
static NSMutableDictionary *kernels_;
static NSMutableArray *retiring_;
static char error_[512];

void *metal_device(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ device_ = MTLCreateSystemDefaultDevice(); });
  return device_;
}

const char *metal_error(void) { return error_; }

/* `device`'s kernels, compiled at its first use */
static struct kernels *kernels_of(id<MTLDevice> device) {
  @autoreleasepool {
    if (!kernels_) { kernels_ = [NSMutableDictionary new]; retiring_ = [NSMutableArray new]; }
    NSValue *key = [NSValue valueWithPointer:(__bridge const void *)device];
    NSValue *held = kernels_[key];
    if (held) return held.pointerValue;
    MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
    options.mathMode = MTLMathModeSafe;
    NSError *error = nil;
    NSString *primitives = [[[NSString alloc] initWithBytes:mesh_metal length:mesh_metal_len encoding:NSUTF8StringEncoding] autorelease];
    NSMutableString *cases = [NSMutableString string];
    for (int k = 0; k < KINDS; k++) [cases appendFormat:@"case %d: body_%s(o.d, o.s, o.p, o.a, o.v, progress, i, g, tg); break; ", k, KIND[k]];
    NSString *source = [[[[primitives stringByAppendingString:STEPS] stringByAppendingString:KERNELS] stringByAppendingString:TRACKS]
                        stringByAppendingString:[NSString stringWithFormat:TRACK_KERNEL, cases]];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { snprintf(error_, sizeof error_, "the kernels: %s", error.localizedDescription.UTF8String); return NULL; }
    struct kernels *made = calloc(1, sizeof *made);
    if (!made) { [library release]; return NULL; }
    made->library = library;
    made->pipelines = [NSMutableDictionary new];
    [device retain];
    kernels_[key] = [NSValue valueWithPointer:made];
    return made;
  }
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, NSString *name) {
  @autoreleasepool {
    struct kernels *k = kernels_of(device);
    if (!k) return nil;
    id<MTLComputePipelineState> made = k->pipelines[name];
    if (made) return made;
    id<MTLFunction> function = [[k->library newFunctionWithName:name] autorelease];
    NSError *error = nil;
    made = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!made) { snprintf(error_, sizeof error_, "kernel %s: %s", name.UTF8String, error ? error.localizedDescription.UTF8String : "absent"); return nil; }
    k->pipelines[name] = made;
    [made release];
    return made;
  }
}

/* `device`'s argument ring, made at its first Metal 4 program */
static struct ring *ring_of(id<MTLDevice> device) {
  struct kernels *k = kernels_of(device);
  if (!k) return NULL;
  if (!k->ring.records) {
    k->ring.records = [device newBufferWithLength:(NSUInteger)RECORDS * RECORD options:MTLResourceStorageModeShared];
    k->ring.progress = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!k->ring.records || !k->ring.progress) {
      [k->ring.records release]; [k->ring.progress release]; k->ring.records = k->ring.progress = nil;
      snprintf(error_, sizeof error_, "the argument ring of %u records", RECORDS);
      return NULL;
    }
    memset(k->ring.progress.contents, 0, 64);
  }
  return &k->ring;
}

void metal_records(uint64_t *issued, uint64_t *capacity) {
  struct ring *ring = ring_of(metal_device());
  *issued = ring ? ring->next : 0;
  *capacity = RECORDS - 1;
}

void *metal_wrap(void *address, size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithBytesNoCopy:address length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

void *metal_scratch(size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModePrivate];
}

void *metal_shared(size_t bytes) {
  id<MTLBuffer> made = [(id<MTLDevice>)metal_device() newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
  if (made) memset(made.contents, 0, made.length);
  return made;
}

void *metal_event(void) { return [(id<MTLDevice>)metal_device() newSharedEvent]; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
int metal_wait(void *event, uint64_t value, uint64_t ns) { return [(id<MTLSharedEvent>)event waitUntilSignaledValue:value timeoutMS:ns / 1000000]; }
void metal_release(void *object) { [(id)object release]; }
void *metal_retain(void *object) { return [(id)object retain]; }

/* a stream's polls as a program's bound: 0, or more than NCCL_MESH_POLLS, is NCCL_MESH_POLLS
   (design/transport.md) */
static uint32_t bound_of(uint32_t polls) { return polls && polls < NCCL_MESH_POLLS ? polls : NCCL_MESH_POLLS; }

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls) {
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) return NULL;
  program->buffer = [(id<MTLCommandBuffer>)command_buffer retain];
  if (encoder) { program->encoder = [(id<MTLComputeCommandEncoder>)encoder retain]; program->borrowed = 1; }
  program->kept = [NSMutableArray new];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  return program;
}

struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls) {
  struct ring *ring = ring_of(((id<MTLResidencySet>)residency).device);
  struct metal_program *program = ring ? calloc(1, sizeof *program) : NULL;
  if (!program) return NULL;
  program->encoder4 = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  program->ring = ring;
  program->borrowed = 1;
  metal_resident(program, ring->records);
  metal_resident(program, ring->progress);
  metal_resident(program, violation);
  [program->encoder4 barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
  return program;
}

int metal_metal4(const struct metal_program *program) { return program->encoder4 != nil; }

void metal_resident(struct metal_program *program, void *buffer) {
  if (!program->residency || !buffer || [program->residency containsAllocation:(id<MTLBuffer>)buffer]) return;
  [program->residency addAllocation:(id<MTLBuffer>)buffer];
  program->added = 1;
}

void metal_gate(struct metal_program *program, void *violation) {
  if ((id<MTLBuffer>)violation == program->violation) return;
  [(id<MTLBuffer>)violation retain];
  [program->violation release];
  program->violation = (id<MTLBuffer>)violation;
  metal_resident(program, violation);
}

static void close_encoder(struct metal_program *program) {
  if (!program->encoder) return;
  if (!program->borrowed) [program->encoder endEncoding];
  [program->encoder release];
  program->encoder = nil;
}

/* the Metal 4 form's record of a step: its arguments the ring's next record (refused where the GPU has not begun the
   dispatch that read that record RECORDS dispatches ago: its command buffers not committed, or that far behind), its
   buffers made resident and their addresses in binding order (0 d, 1 s, 2 the record, 3 a, 4 v: the failure words a
   wait names, else the program's (metal_gate), the waits bounded by the program's polls and failures recorded where
   there are such words; 5 progress) */
static int record4(struct metal_program *program, id<MTLBuffer> d, id<MTLBuffer> s, id<MTLBuffer> a, id<MTLBuffer> words, struct args p,
                   uint64_t bound[6]) {
  struct ring *ring = program->ring;
  const uint64_t begun = *(volatile uint64_t *)ring->progress.contents;
  if (ring->next - begun >= RECORDS - 1) {
    snprintf(error_, sizeof error_, "the GPU began none of the library's last %u dispatches (dispatch %llu written, %llu begun): their command "
             "buffers are not committed, or that far behind", RECORDS - 1, (unsigned long long)ring->next, (unsigned long long)begun);
    return EOVERFLOW;
  }
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence = ++ring->next;
  memcpy((char *)ring->records.contents + ((p.sequence - 1) % RECORDS) * RECORD, &p, sizeof p);
  for (id<MTLBuffer> b in @[d, s ? s : d, a ? a : d, v]) metal_resident(program, b);
  const uint64_t made[6] = {d.gpuAddress, (s ? s : d).gpuAddress, ring->records.gpuAddress + ((p.sequence - 1) % RECORDS) * RECORD,
                            (a ? a : d).gpuAddress, v.gpuAddress, ring->progress.gpuAddress};
  memcpy(bound, made, sizeof made);
  return 0;
}

/* the Metal 4 form's dispatch: its record (record4), every buffer bound by address through the caller's table, then a
   barrier before the next */
static int dispatch4(struct metal_program *program, id<MTLComputePipelineState> state, id<MTLBuffer> d, id<MTLBuffer> s, id<MTLBuffer> a,
                     id<MTLBuffer> words, struct args p, uint64_t units, int group) {
  uint64_t bound[6];
  const int recorded = record4(program, d, s, a, words, p, bound);
  if (recorded) return recorded;
  for (NSUInteger k = 0; k < 6; k++) [program->table setAddress:bound[k] atIndex:k];
  id<MTL4ComputeCommandEncoder> encoder = program->encoder4;
  [encoder setComputePipelineState:state];
  [encoder setArgumentTable:program->table];
  const NSUInteger most = group ? 1024 : 256, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
  if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  else [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
  return 0;
}

/* kernel `name` on buffers d, s, a and v (0, 1, 3 and 4; an absent one d, and v, where none is given, the program's
   failure words: metal_gate) and the arguments p (2, its polls the program's where v is such words), the progress
   word (5; the classic form binds d there, which no kernel then writes): a GROUP kernel (`group`) as one threadgroup of
   up to 1024 threads over a block's piece, any other over `units` threads */
static int dispatch_as(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p, uint64_t units,
                       int group);
/* a GROUP kernel where `a` is given (a stream class's ready or landed words) */
static int dispatch(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p, uint64_t units) {
  return dispatch_as(program, name, d, s, a, v, p, units, a != NULL);
}
static int record_step(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p);
static int recording_on(const struct metal_program *program);
static int dispatch_as(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p, uint64_t units,
                       int group) {
  if (!units) return 0;
  if (recording_on(program)) return record_step(program, name, d, s, a, v, p);
  id<MTLComputePipelineState> state = pipeline(program->encoder4 ? program->residency.device : (id<MTLDevice>)metal_device(), name);
  if (!state) return EINVAL;
  @autoreleasepool {
    if (program->encoder4) return dispatch4(program, state, (id<MTLBuffer>)d, (id<MTLBuffer>)s, (id<MTLBuffer>)a, (id<MTLBuffer>)v, p, units, group);
    if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
    id<MTLComputeCommandEncoder> encoder = program->encoder;
    if (!v) v = (void *)program->violation;
    p.polls = v && v != d ? program->polls : 0;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)(s ? s : d) offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof p atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)(a ? a : d) offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)(v ? v : d) offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:5];
    const NSUInteger most = group ? 1024 : 256, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
    if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    else [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  return 0;
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  NSString *name = [NSString stringWithFormat:@"copy_%s_%d", MODE[mode], words ? 4 : 1];
  return dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes}, words ? bytes / 4 : bytes);
}

static NSString *typed(const char *what, int type) {
  return (unsigned)type <= METAL_MERGE && TYPE[type] ? [NSString stringWithFormat:@"%s_%s", what, TYPE[type]] : nil;
}

int metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n) {
  NSString *name = typed("combine", type);
  return name ? dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op}, n) : EINVAL;
}

int metal_combine3(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, void *with,
                   size_t with_offset, size_t n) {
  NSString *name = typed("combine3", type);
  return name ? dispatch_as(program, name, dst, src, with, NULL,
                            (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op, .aux = with_offset}, n, 0) : EINVAL;
}

int metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar) {
  NSString *name = typed("premul", type);
  return name ? dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = scalar}, n) : EINVAL;
}

int metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor) {
  NSString *name = type <= 5 ? typed("truncdiv", type) : nil;
  return name ? dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = divisor}, n) : EINVAL;
}

/* a wait's arguments (nccl-mesh-metal.h struct metal_wait): its word at aux for 1 + scalar, its run's first unit at src
   (a dud's check, and a landing's piece), its position and channel for its failure's record */
static struct args wait_args(const struct metal_wait *w, size_t dst_offset, size_t n, int op) {
  return (struct args){.dst = dst_offset, .src = w->slot, .n = n, .scalar = w->cycle, .aux = w->word, .op = (uint32_t)op, .at = w->at,
                       .channel = w->channel};
}

/* a release's arguments (struct metal_release): its slot at dst, its count words at aux each scalar */
static struct args release_args(const struct metal_release *r, size_t src_offset, size_t bytes) {
  return (struct args){.dst = r->slot, .src = src_offset, .n = bytes, .scalar = r->value, .aux = r->word, .count = r->count};
}

int metal_publish(struct metal_program *program, const struct metal_release *r) {
  return dispatch(program, @"publish", r->buffer, NULL, NULL, NULL, release_args(r, 0, 1), 1);
}

int metal_publish_filler(struct metal_program *program, const struct metal_release *r) {
  return dispatch(program, @"publish_filler", r->buffer, NULL, NULL, NULL, release_args(r, 0, 1), 1);
}

int metal_spin(struct metal_program *program, const struct metal_wait *w) {
  return dispatch(program, @"spin", w->buffer, w->buffer, NULL, w->violation, wait_args(w, 0, 1, 0), 1);
}

int metal_send_small(struct metal_program *program, void *src, size_t src_offset, size_t bytes, const struct metal_release *r) {
  return dispatch(program, @"send_small", r->buffer, src, r->buffer, NULL, release_args(r, src_offset, bytes), 1);
}

int metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w) {
  NSString *name = type < 0 ? @"land_copy" : typed("land", type);
  return name ? dispatch(program, name, dst, w->buffer, w->buffer, w->violation, wait_args(w, dst_offset, n, op), 1) : EINVAL;
}

int metal_send_deferred(struct metal_program *program, size_t bytes, const struct metal_release *r, uint64_t bound[6]) {
  if (!program->encoder4) { snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's"); return EINVAL; }
  return record4(program, (id<MTLBuffer>)r->buffer, nil, (id<MTLBuffer>)r->buffer, nil, release_args(r, 0, bytes), bound);
}

int metal_land_deferred(struct metal_program *program, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w,
                        uint64_t bound[6]) {
  if (!program->encoder4) { snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's"); return EINVAL; }
  return record4(program, (id<MTLBuffer>)dst, (id<MTLBuffer>)w->buffer, (id<MTLBuffer>)w->buffer, (id<MTLBuffer>)w->violation,
                 wait_args(w, dst_offset, n, op), bound);
}

/* a counted all-to-all's dispatch (nccl-mesh-metal.h metal_counted): its kernel by kind, its copies' width the widest
   word that divides every offset, row and piece it moves; each one's arguments as the kernel above reads them */
int metal_counted(struct metal_program *program, int kind, const struct metal_counted *k) {
  static const char *const NAME[] = {"counted_start", "counted_check", "counted_arrived", "counted_send", "counted_land", "counted_own"};
  if (kind < METAL_COUNTED_START || kind > METAL_COUNTED_OWN) return EINVAL;
  const int words = !((k->dst | k->src | k->row | k->bytes | k->aux) & 3);
  NSString *name = kind >= METAL_COUNTED_SEND ? [NSString stringWithFormat:@"%s_%d", NAME[kind], words ? 4 : 1] : @(NAME[kind]);
  const struct args p = {.dst = k->dst, .src = k->src, .n = k->bytes, .scalar = k->scalar, .aux = k->aux, .first = k->first, .slot = k->row,
                         .at = k->limit, .count = k->entries, .mask = kind == METAL_COUNTED_ARRIVED && k->a};
  return dispatch(program, name, k->d, k->s, kind >= METAL_COUNTED_SEND ? k->a : NULL, kind == METAL_COUNTED_ARRIVED ? k->a : NULL, p,
                  kind == METAL_COUNTED_START ? (k->bytes ? k->bytes : 1) : 1);
}

/* the tracks being recorded (nccl-mesh-metal.h metal_record): each step's kernel and bindings as its dispatch would bind
   them (`v` its failure words, else the program's; its polls the program's where it has such words), each item's
   position, credit, steps, dependencies and the landing a consumption awaits (its first wait: spin's word in d, a
   landing's in a), each track's items by cursor; the buffers named, made resident at metal_tracks */
struct host_op { void *d, *s, *a, *v; struct args p; uint32_t kind; };
struct host_item { uint64_t position, need, target; void *word_buffer, *v; size_t word; uint32_t first_op, ops, first_dep, deps, wait_op, cursor; };
struct host_dep { uint64_t count; uint32_t other; };
struct host_track { uint32_t *item[2]; size_t count[2], room[2]; };
struct recording { struct host_op *op; size_t ops, op_room; struct host_item *item; size_t items, item_room; struct host_dep *dep; size_t deps, dep_room;
                   struct host_track *track; uint32_t tracks; uint32_t current; int on; NSMutableSet *named; };

static int recording_on(const struct metal_program *program) { return program->recording && program->recording->on; }

static void *more(void *items, size_t *room, size_t count, size_t size) {
  if (count < *room) return items;
  const size_t bigger = *room ? 2 * *room : 16;
  void *made = realloc(items, bigger * size);
  if (made) *room = bigger;
  return made;
}

static void recording_free(struct recording *r) {
  if (!r) return;
  for (uint32_t t = 0; t < r->tracks; t++) { free(r->track[t].item[0]); free(r->track[t].item[1]); }
  free(r->track); free(r->op); free(r->item); free(r->dep);
  [r->named release];
  free(r);
}

static int record_step(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p) {
  struct recording *r = program->recording;
  uint32_t kind = 0;
  while (kind < KINDS && strcmp(KIND[kind], name.UTF8String)) kind++;
  if (kind == KINDS) { snprintf(error_, sizeof error_, "kernel %s on a track", name.UTF8String); return EINVAL; }
  if (!v) v = (void *)program->violation;
  p.polls = v && v != d ? program->polls : 0;
  struct host_op *ops = more(r->op, &r->op_room, r->ops, sizeof *ops);
  if (!ops) return ENOMEM;
  r->op = ops;
  const struct host_op made = {d, s ? s : d, a ? a : d, v ? v : d, p, kind};
  r->op[r->ops++] = made;
  for (id b in @[(id)made.d, (id)made.s, (id)made.a, (id)made.v]) [r->named addObject:b];
  struct host_item *it = r->item + r->current;
  it->ops++;
  const int spin = !strcmp(KIND[kind], "spin"), lands = !strncmp(KIND[kind], "land_", 5);
  if (it->cursor == METAL_CONSUME && it->wait_op == UINT32_MAX && (spin || lands)) {
    it->wait_op = (uint32_t)(r->ops - 1);
    it->word_buffer = spin ? made.d : made.a;
    it->word = p.aux;
    it->target = 1 + p.scalar;
  }
  if (!it->v && p.polls) it->v = made.v;
  return 0;
}

int metal_record(struct metal_program *program, uint32_t track, int cursor, uint64_t position, uint64_t need) {
  if (track == METAL_NONE) { if (program->recording) program->recording->on = 0; return 0; }
  if (!program->recording) {
    if (!(program->recording = calloc(1, sizeof *program->recording))) return ENOMEM;
    program->recording->named = [NSMutableSet new];
  }
  struct recording *r = program->recording;
  if (track >= r->tracks) {
    struct host_track *grown = realloc(r->track, ((size_t)track + 1) * sizeof *grown);
    if (!grown) return ENOMEM;
    memset(grown + r->tracks, 0, ((size_t)track + 1 - r->tracks) * sizeof *grown);
    r->track = grown; r->tracks = track + 1;
  }
  struct host_track *t = r->track + track;
  uint32_t *listed = more(t->item[cursor], &t->room[cursor], t->count[cursor], sizeof *listed);
  struct host_item *items = listed ? more(r->item, &r->item_room, r->items, sizeof *items) : NULL;
  if (listed) t->item[cursor] = listed;
  if (!items) return ENOMEM;
  r->item = items;
  r->item[r->items] = (struct host_item){.position = position, .need = need, .first_op = (uint32_t)r->ops, .first_dep = (uint32_t)r->deps,
                                         .wait_op = UINT32_MAX, .cursor = (uint32_t)cursor};
  t->item[cursor][t->count[cursor]++] = (uint32_t)r->items;
  r->current = (uint32_t)r->items++;
  r->on = 1;
  return 0;
}

int metal_depend(struct metal_program *program, uint32_t other, uint64_t count) {
  struct recording *r = program->recording;
  if (!r || !r->on) return EINVAL;
  struct host_dep *deps = more(r->dep, &r->dep_room, r->deps, sizeof *deps);
  if (!deps) return ENOMEM;
  r->dep = deps;
  r->dep[r->deps++] = (struct host_dep){count, other};
  r->item[r->current].deps++;
  return 0;
}

/* the records as the track kernel reads them (TRACKS: track_head, track, track_item, track_op, track_dep) */
struct track_head { uint64_t tracks_off, items_off, ops_off, deps_off, count_off, sequence, progress_off; uint32_t tracks, polls; };
struct track_rec { uint32_t first[2], count[2]; };
struct track_item { uint64_t position, need, target, word, v; uint32_t first_op, ops, first_dep, deps, wait_op, cursor; };
struct track_op { uint64_t d, s, a, v; struct args p; uint32_t kind, pad; };
struct track_dep { uint64_t count; uint32_t other, pad; };
_Static_assert(sizeof(struct track_head) == 64 && sizeof(struct track_rec) == 16 && sizeof(struct track_item) == 64 &&
               sizeof(struct track_op) == 136 && sizeof(struct track_dep) == 16, "the track records");

static uint64_t up(uint64_t x) { return (x + 255) & ~(uint64_t)255; }

/* `bytes` of the device's track ring for the dispatch of sequence `sequence`: its offset, or UINT64_MAX where the GPU
   still reads every span that place would take (its dispatches not begun: their command buffers not committed, or that
   far behind) */
static uint64_t ring_take(struct track_ring *t, uint64_t sequence, uint64_t bytes) {
  const uint64_t begun = *(volatile uint64_t *)t->buffer.contents;
  size_t kept = 0;
  for (size_t k = 0; k < t->helds; k++) if (t->held[k].sequence >= begun) t->held[kept++] = t->held[k];
  t->helds = kept;
  uint64_t first = t->head + bytes <= TRACK_RING ? t->head : 64;
  if (first < 64) first = 64;
  if (first + bytes > TRACK_RING) return UINT64_MAX;
  for (size_t k = 0; k < t->helds; k++)
    if (first < t->held[k].end && t->held[k].first < first + bytes) return UINT64_MAX;
  struct span *held = more(t->held, &t->room, t->helds, sizeof *held);
  if (!held) return UINT64_MAX;
  t->held = held;
  t->held[t->helds++] = (struct span){sequence, first, first + bytes};
  t->head = first + bytes;
  return first;
}

int metal_tracks(struct metal_program *program, uint32_t tracks, const uint64_t *consumed) {
  struct recording *r = program->recording;
  program->recording = NULL;
  if (!r || !r->tracks) { recording_free(r); return 0; }
  @autoreleasepool {
    id<MTLDevice> device = program->encoder4 ? program->residency.device : (id<MTLDevice>)metal_device();
    struct kernels *k = kernels_of(device);
    id<MTLComputePipelineState> state = pipeline(device, @"tracks");
    if (!k || !state) { recording_free(r); return EINVAL; }
    struct track_ring *ring = &k->tracks;
    if (!ring->buffer) {
      ring->buffer = [device newBufferWithLength:TRACK_RING options:MTLResourceStorageModeShared];
      if (!ring->buffer) { recording_free(r); snprintf(error_, sizeof error_, "the track ring of %u bytes", TRACK_RING); return ENOMEM; }
      memset(ring->buffer.contents, 0, 64);
      ring->head = 64;
    }
    const uint64_t T = r->tracks, at_tracks = up(sizeof(struct track_head)), at_items = up(at_tracks + T * sizeof(struct track_rec));
    const uint64_t at_ops = up(at_items + r->items * sizeof(struct track_item)), at_deps = up(at_ops + r->ops * sizeof(struct track_op));
    const uint64_t at_count = up(at_deps + r->deps * sizeof(struct track_dep)), bytes = up(at_count + T * sizeof(uint64_t));
    const uint64_t sequence = ++ring->sequence, first = ring_take(ring, sequence, bytes);
    if (first == UINT64_MAX) {
      recording_free(r);
      snprintf(error_, sizeof error_, "the track ring holds no %llu bytes free: the GPU has not begun the tracks that read it (their command buffers "
               "not committed, or that far behind)", (unsigned long long)bytes);
      return EOVERFLOW;
    }
    unsigned char *base = (unsigned char *)ring->buffer.contents + first;
    const uint64_t address = ring->buffer.gpuAddress;
    *(struct track_head *)base = (struct track_head){at_tracks, at_items, at_ops, at_deps, first + at_count, sequence, 0, (uint32_t)T, program->polls};
    struct track_rec *rec = (struct track_rec *)(base + at_tracks);
    struct track_item *item = (struct track_item *)(base + at_items);
    uint32_t placed = 0;
    for (uint32_t t = 0; t < T; t++)
      for (int c = 0; c < 2; c++) {
        rec[t].first[c] = placed; rec[t].count[c] = (uint32_t)r->track[t].count[c];
        for (size_t x = 0; x < r->track[t].count[c]; x++) {
          const struct host_item *it = r->item + r->track[t].item[c][x];
          item[placed++] = (struct track_item){it->position, it->need, it->target,
                                               it->word_buffer ? ((id<MTLBuffer>)it->word_buffer).gpuAddress + it->word : 0,
                                               it->v ? ((id<MTLBuffer>)it->v).gpuAddress : 0, it->first_op, it->ops, it->first_dep, it->deps, it->wait_op,
                                               it->cursor};
        }
      }
    struct track_op *op = (struct track_op *)(base + at_ops);
    for (size_t x = 0; x < r->ops; x++) {
      const struct host_op *o = r->op + x;
      op[x] = (struct track_op){((id<MTLBuffer>)o->d).gpuAddress, ((id<MTLBuffer>)o->s).gpuAddress, ((id<MTLBuffer>)o->a).gpuAddress,
                                ((id<MTLBuffer>)o->v).gpuAddress, o->p, o->kind, 0};
    }
    struct track_dep *dep = (struct track_dep *)(base + at_deps);
    for (size_t x = 0; x < r->deps; x++) dep[x] = (struct track_dep){r->dep[x].count, r->dep[x].other, 0};
    uint64_t *count = (uint64_t *)(base + at_count);
    for (uint32_t t = 0; t < T; t++) count[t] = t < tracks ? consumed[t] : 0;
    const NSUInteger width = state.maxTotalThreadsPerThreadgroup < 1024 ? state.maxTotalThreadsPerThreadgroup : 1024;
    if (program->encoder4) {
      metal_resident(program, ring->buffer);
      for (id b in r->named) metal_resident(program, (void *)b);
      [program->table setAddress:address + first atIndex:0];
      [program->table setAddress:address atIndex:1];
      id<MTL4ComputeCommandEncoder> encoder = program->encoder4;
      [encoder setComputePipelineState:state];
      [encoder setArgumentTable:program->table];
      [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    } else {
      if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
      id<MTLComputeCommandEncoder> encoder = program->encoder;
      for (id b in r->named) [encoder useResource:(id<MTLBuffer>)b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      [encoder setComputePipelineState:state];
      [encoder setBuffer:ring->buffer offset:first atIndex:0];
      [encoder setBuffer:ring->buffer offset:0 atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
  }
  recording_free(r);
  return 0;
}

void *metal_contents(void *buffer) { return buffer ? ((id<MTLBuffer>)buffer).contents : NULL; }

const char *metal_steps(void) { return STEPS.UTF8String; }

void metal_keep(struct metal_program *program, void *object) { [program->kept addObject:(id)object]; }

/* the program's end signals `event` to `value`; what it kept is released once the event has passed it
   (metal_collect); one with no event (a failed group's, nccl-mesh.c program_end) signals nothing, and what its own
   encoder's commands kept is held by its command buffer (a completion handler's capture) until that buffer completes
   or is freed; a program in a borrowed encoder keeps nothing; the Metal 4 form's residency set committed where the
   program made a buffer resident */
void metal_end(struct metal_program *program, void *event, uint64_t value) {
  recording_free(program->recording);
  program->recording = NULL;
  close_encoder(program);
  if (event) {
    [program->buffer encodeSignalEvent:(id<MTLEvent>)event value:value];
    @autoreleasepool { [retiring_ addObject:@[@(value), program->kept]]; }
  } else if (!program->borrowed && program->buffer) {
    NSMutableArray *kept = program->kept;
    [program->buffer addCompletedHandler:^(id<MTLCommandBuffer> done) { (void)done; (void)kept; }];
  }
  if (program->added) [program->residency commit];
  [program->kept release];
  [program->buffer release];
  [program->encoder4 release]; [program->table release]; [program->residency release]; [program->violation release];
  free(program);
}

void metal_collect(void *event) {
  const uint64_t signaled = metal_signaled(event);
  @autoreleasepool {
    NSIndexSet *done = [retiring_ indexesOfObjectsPassingTest:^BOOL(NSArray *entry, NSUInteger index, BOOL *stop) {
      (void)index; (void)stop; return [entry[0] unsignedLongLongValue] <= signaled; }];
    [retiring_ removeObjectsAtIndexes:done];
  }
}

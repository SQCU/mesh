#ifndef NCCL_MESH_KERNELS_H
#define NCCL_MESH_KERNELS_H
#import <Foundation/Foundation.h>
#include <stdint.h>
#include "nccl-mesh-metal.h"

#define STR(...) #__VA_ARGS__
#define XSTR(...) STR(__VA_ARGS__)

#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 first, sent; }

static NSString *const STEPS =
  @"\nusing namespace metal;\n"
  XSTR(ARGS(mesh_args, ulong, uint)) ";\n"
  XSTR(MESH_PEER(device uchar *)) ";\n"
  XSTR(MESH_WAIT(device uchar *, ulong)) ";\n"
  XSTR(MESH_RELAY(device uchar *, ulong)) ";\n"
  XSTR(MESH_TABLE(device uchar *, ulong, uint, atomic_uint)) ";\n"
  "template <typename T> struct mesh_view { mesh_view(device uchar *at = nullptr) : at(at) {} "
  "T operator[](ulong k) const { return mesh_payload((volatile coherent(system) device const T *)at + k); } "
  "void store(ulong k, T v) const { ((volatile coherent(system) device T *)at)[k] = v; } "
  "mesh_view<T> operator+(ulong k) const { return mesh_view<T>(at + k * sizeof(T)); } "
  "template <typename U> mesh_view<U> as() const { return mesh_view<U>(at); } "
  "private: device uchar *at; };\n"
  XSTR(MESH_CROSSING(device mesh_table *, uint, ulong,
                     friend void mesh_enter(constant mesh_crossing &, uint); friend void mesh_leave(constant mesh_crossing &, uint);
                     friend uint mesh_landings(constant mesh_crossing &);
                     template <typename T> friend mesh_view<T> mesh_landing(constant mesh_crossing &, uint);
                     template <typename T> friend mesh_view<T> mesh_own(constant mesh_crossing &);
                     template <typename T> friend mesh_view<T> mesh_send(constant mesh_crossing &); private:)) ";\n"
  "#define MESH_SYSTEM static_cast<thread_scope>(3)\n"
  "static inline void mesh_step_until(uint polls, device uchar *word, ulong target) { "
  "if (target) mesh_wait_bounded((volatile coherent(system) device const ulong *)word, target, polls); }\n"
  "static inline void mesh_step_counts(constant mesh_args &p, device uchar *words) { "
  "mesh_step_until(p.polls, words + p.aux, p.scalar); mesh_step_until(p.polls, words + p.first, p.sent); }\n"
  "static inline void mesh_step_await(constant mesh_args &p, device uchar *words, uint i) { "
  "if (i == 0) mesh_step_counts(p, words); "
  "threadgroup_barrier(mem_flags::mem_device); atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, MESH_SYSTEM); }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *words, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(words + p.aux), p.scalar); }\n"
  "static inline device mesh_peer *mesh_peers(device mesh_table *t) { return (device mesh_peer *)(t + 1); }\n"
  "static inline device mesh_wait *mesh_waits(device mesh_table *t) { return (device mesh_wait *)(mesh_peers(t) + t->peers); }\n"
  "static inline void mesh_relay_all(device mesh_table *t, uint polls) { "
  "device mesh_relay *r = (device mesh_relay *)(mesh_waits(t) + t->waits); "
  "uint looks = 0u, left = 0u; "
  "do { left = 0u; atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, MESH_SYSTEM); "
  "for (uint q = 0; q < t->relays; q++) { volatile coherent(system) device ulong *p = (volatile coherent(system) device ulong *)&r[q].published; "
  "if (*p >= r[q].ready) continue; "
  "if (*(volatile coherent(system) device const ulong *)r[q].landed >= r[q].landed_target) *p = r[q].ready; else left++; } "
  "} while (left && ++looks < polls); }\n"
  "inline void mesh_enter(constant mesh_crossing &c, uint i) { "
  "for (uint q = 0; i == 0 && c.land && q < c.land->waits; q++) { device mesh_wait &e = mesh_waits(c.land)[q]; "
  "mesh_step_until(c.polls, e.landed, c.target); } "
  "threadgroup_barrier(mem_flags::mem_device); atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, MESH_SYSTEM); }\n"
  "inline void mesh_leave(constant mesh_crossing &c, uint i) { "
  "device mesh_table *t = c.publish ? c.publish : c.land; "
  "if (!t) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, MESH_SYSTEM); threadgroup_barrier(mem_flags::mem_device); "
  "if (i != 0 || atomic_fetch_add_explicit(&t->arrivals, 1u, memory_order_relaxed) != c.groups - 1) return; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, MESH_SYSTEM); "
  "if (c.publish && c.publish->ready) *(volatile coherent(system) device ulong *)&c.publish->published = c.publish->ready; "
  "mesh_relay_all(t, c.polls); "
  "if (c.land) *(volatile coherent(system) device uint *)&c.land->ended += 1u; "
  "if (c.publish && c.publish != c.land) *(volatile coherent(system) device uint *)&c.publish->ended += 1u; "
  "atomic_store_explicit(&t->arrivals, 0u, memory_order_relaxed); }\n"
  "template <typename W> inline W mesh_wait(volatile coherent(system) device const W *word, ulong target) { "
  "return mesh_wait_bounded(word, target, " XSTR(MESH_POLLS) "); }\n"
  "inline uint mesh_landings(constant mesh_crossing &c) { return c.land ? c.land->peers : 0u; }\n"
  "template <typename T> mesh_view<T> mesh_landing(constant mesh_crossing &c, uint q) { return mesh_view<T>(mesh_peers(c.land)[q].landing); }\n"
  "template <typename T> mesh_view<T> mesh_own(constant mesh_crossing &c) { return mesh_view<T>(c.land->part); }\n"
  "template <typename T> mesh_view<T> mesh_send(constant mesh_crossing &c) { return mesh_view<T>(c.publish->part); }\n";

static NSString *const KERNELS =
  @"\n"
  "#define PARAMS device uchar *d, device uchar *s, constant mesh_args &p, device uchar *a, uint i, uint g, uint l, threadgroup ulong *tg\n"
  "#define HEAD(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]], uint l [[thread_position_in_threadgroup]]) { body_##name(d, s, p, a, i, g, l, nullptr); } static void body_##name(PARAMS)\n"
  "#define GROUP(name) static void body_##name(PARAMS); kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_##name(d, s, p, a, i, g, i, tg); } static void body_##name(PARAMS)\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define AWAIT mesh_step_await(p, a, l);\n"
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { device W *x = (device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { device W *x = (device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "#define BODY(T, OP) device T *x = (device T *)(d + p.dst); volatile coherent(system) device const T *xs = (volatile coherent(system) device const T *)(d + p.dst); "
  "volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); "
  "LOOP x[k] = OP(mesh_payload(xs + k), mesh_payload(y + k), p.op);\n"
  "#define COMBINE(name, T, OP) HEAD(land_##name) { AWAIT BODY(T, OP) } "
  "HEAD(combine3_##name) { "
  "device T *x = (device T *)(d + p.dst); device const T *z = (device const T *)(a + p.aux); "
  "volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP x[k] = OP(z[k], mesh_payload(y + k), p.op); }\n"
  "#define INTEGER(name, T, SIGNED) COMBINE(name, T, op_int<T>) "
  "HEAD(premul_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); ulong c = ulong(T(p.scalar)); LOOP x[k] = T(ulong(y[k]) * c); } "
  "HEAD(truncdiv_##name) { device T *x = (device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP { T v = mesh_payload(y + k); bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) COMBINE(name, T, op_float<T>) "
  "HEAD(premul_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "#define FLOAT8(name, E5) COMBINE(name, uchar, op_f8<E5>) "
  "HEAD(premul_##name) { device uchar *x = (device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = f8_from(f8_to(y[k], E5) * c, E5); }\n"
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
  "GROUP(publish) { mesh_step_release(p, d, i); }\n"
  "GROUP(stamp) { threadgroup_barrier(mem_flags::mem_device); if (i == 0) { "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, MESH_SYSTEM); atomic_fetch_add_explicit((device atomic_uint *)(d + p.aux), 1u, memory_order_relaxed); } }\n"
  "GROUP(spin) { if (i == 0) mesh_step_counts(p, d); }\n"
  "constant float PARTIAL_NONE = -3.4028234663852886e38f;\n"
  "static inline void merge_row(device float *x, volatile coherent(system) device const float *xs, volatile coherent(system) device const float *y, uint u) { "
  "const float m = mesh_payload(y), ly = mesh_payload(y + 1), mx = mesh_payload(xs), lx = mesh_payload(xs + 1); const bool mine = lx > 0.0f, theirs = ly > 0.0f; "
  "if (mine && !theirs) { for (uint j = 0; j < u; j++) x[j] = mesh_payload(xs + j); return; } "
  "if (!mine || !theirs) { for (uint j = 0; j < u; j++) x[j] = theirs ? mesh_payload(y + j) : j ? 0.0f : PARTIAL_NONE; return; } "
  "const float top = max(mx, m), a = exp(mx - top), b = exp(m - top); x[0] = top; "
  "for (uint j = 1; j < u; j++) x[j] = (a > 0.0f ? mesh_payload(xs + j) * a : 0.0f) + (b > 0.0f ? mesh_payload(y + j) * b : 0.0f); }\n"
  "#define MERGE device float *x = (device float *)(d + p.dst); volatile coherent(system) device const float *y = (volatile coherent(system) device const float *)(s + p.src); "
  "for (ulong r = i; r < p.n; r += g) merge_row(x + r * p.op, (volatile coherent(system) device const float *)x + r * p.op, y + r * p.op, p.op);\n"
  "HEAD(land_merge) { AWAIT MERGE }\n"
  "HEAD(combine3_merge) { device float *x = (device float *)(d + p.dst); "
  "volatile coherent(system) device const float *z = (volatile coherent(system) device const float *)(a + p.aux); "
  "volatile coherent(system) device const float *y = (volatile coherent(system) device const float *)(s + p.src); "
  "for (ulong r = i; r < p.n; r += g) merge_row(x + r * p.op, z + r * p.op, y + r * p.op, p.op); }\n";

enum { WORD_D = 1, WORD_A };
#define KINDS_OF(X) \
  X(copy_plain_4, 0) X(copy_land_4, 0) X(copy_plain_1, 0) X(copy_land_1, 0) \
  X(land_i8, WORD_A) X(combine3_i8, 0) X(premul_i8, 0) X(truncdiv_i8, 0) \
  X(land_u8, WORD_A) X(combine3_u8, 0) X(premul_u8, 0) X(truncdiv_u8, 0) \
  X(land_i32, WORD_A) X(combine3_i32, 0) X(premul_i32, 0) X(truncdiv_i32, 0) \
  X(land_u32, WORD_A) X(combine3_u32, 0) X(premul_u32, 0) X(truncdiv_u32, 0) \
  X(land_i64, WORD_A) X(combine3_i64, 0) X(premul_i64, 0) X(truncdiv_i64, 0) \
  X(land_u64, WORD_A) X(combine3_u64, 0) X(premul_u64, 0) X(truncdiv_u64, 0) \
  X(land_f16, WORD_A) X(combine3_f16, 0) X(premul_f16, 0) X(land_f32, WORD_A) X(combine3_f32, 0) X(premul_f32, 0) \
  X(land_bf16, WORD_A) X(combine3_bf16, 0) X(premul_bf16, 0) X(land_e4m3, WORD_A) X(combine3_e4m3, 0) X(premul_e4m3, 0) \
  X(land_e5m2, WORD_A) X(combine3_e5m2, 0) X(premul_e5m2, 0) \
  X(publish, 0) X(stamp, 0) X(spin, WORD_D) X(land_merge, WORD_A) X(combine3_merge, 0)
#define KIND_ENUM(name, word) KIND_##name,
#define KIND_NAME(name, word) #name,
#define KIND_WORD(name, word) word,
#define KIND_CASE(name, word) case KIND_##name: body_##name(o.d, o.s, o.p, o.a, i, g, i, tg); break;

enum { KINDS_OF(KIND_ENUM) KINDS, KIND_tracks = KINDS, PIPELINES };
static const char *const NAME[PIPELINES] = {KINDS_OF(KIND_NAME) "tracks"};
static const uint8_t WORD[KINDS] = {KINDS_OF(KIND_WORD)};

static NSString *const TRACKS =
  @"\n"
  "enum { " XSTR(KINDS_OF(KIND_ENUM)) " };\n"
  "struct track_op { device uchar *d; device uchar *s; device uchar *a; mesh_args p; uint kind; uint pad; };\n"
  "struct track_item { ulong position, need, target; device ulong *word; uint first_op, ops, first_dep, deps, wait_op, cursor; };\n"
  "struct track_dep { ulong count; uint other, pad; };\n"
  "struct track { uint first[2], count[2]; };\n"
  "struct track_head { ulong tracks_off, items_off, ops_off, deps_off, count_off; uint tracks, polls; };\n"
  "static inline bool track_ready(constant track_item &it, constant track_dep *deps, volatile coherent(system) device ulong *count) { "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
    "for (uint k = 0; k < it.deps; k++) if (count[deps[it.first_dep + k].other] < deps[it.first_dep + k].count) return false; "
  "if (it.cursor != 1 || it.wait_op == ~0u) return true; "
  "return *(volatile coherent(system) device const ulong *)it.word >= it.target; }\n";
static NSString *const TRACK_KERNEL =
  @"kernel void tracks(constant uchar *base [[buffer(0)]], device uchar *rows [[buffer(1)]], uint t [[threadgroup_position_in_grid]], "
  "uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]]) { "
  "constant track_head &h = *(constant track_head *)base; "
  "if (t >= h.tracks) return; "
  "constant track &tr = ((constant track *)(base + h.tracks_off))[t]; "
  "constant track_item *items = (constant track_item *)(base + h.items_off); "
  "constant track_op *ops = (constant track_op *)(base + h.ops_off); "
  "constant track_dep *deps = (constant track_dep *)(base + h.deps_off); "
  "volatile coherent(system) device ulong *count = (volatile coherent(system) device ulong *)(rows + h.count_off); "
  "threadgroup ulong tg[4]; threadgroup uint pick[2]; "
  "uint next0 = 0, next1 = 0; ulong looks = 0; "
  "for (;;) { "
  "if (i == 0) { uint chosen = ~0u, left = 0; "
  "if (next1 < tr.count[1]) { left = 1; constant track_item &c = items[tr.first[1] + next1]; "
  "if ((next0 >= tr.count[0] || items[tr.first[0] + next0].position >= c.need) && track_ready(c, deps, count)) { chosen = tr.first[1] + next1; next1++; } } "
  "if (chosen == ~0u && next0 < tr.count[0]) { left = 1; if (track_ready(items[tr.first[0] + next0], deps, count)) { chosen = tr.first[0] + next0; next0++; } } "
  "if (chosen != ~0u) looks = 0; "
  "else if (left && ++looks >= h.polls) left = 0; "
  "pick[0] = chosen; pick[1] = left; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); "
  "const uint chosen = pick[0]; "
  "if (!pick[1]) return; "
  "if (chosen != ~0u) { constant track_item &it = items[chosen]; "
  "for (uint k = 0; k < it.ops; k++) { constant track_op &o = ops[it.first_op + k]; "
  "switch (o.kind) { " XSTR(KINDS_OF(KIND_CASE)) " default: break; } "
  "threadgroup_barrier(mem_flags::mem_device); } "
  "if (it.cursor == 1 && i == 0) mesh_publish(count + t, it.position + 1); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); } }\n";

#endif

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
/* a combine's elements a thread of zelloth's block (nccl-mesh-metal.m metal_zelloth) */
#define MESH_COMBINE_LANES 16
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
  "return mesh_payload((volatile coherent(system) device const T *)landing + k); }\n";

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
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { ENTER(i == 0) volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { ENTER(i == 0) volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  /* design/zorvek.md zelloth: a group's sums, each element written once, by one thread, from the rank's own element
     and every contribution's for it, once all of them have landed, in whatever order they landed. The table at
     s + p.src (GPU addresses): [0] S sums, [1] a block's elements, [2] blocks of every sum, [3] the word offset of
     the blocks' cells; then S sums of 9 words {K inputs, R runs, own, out, its done word (its blocks published), its
     first block, its blocks, the word offset of its payload, the word offset of its waits (0: none)}; its waits, the
     earlier sums whose results its own operand holds, each laid before it: a count, then {its done word, the word
     offset of its thresholds} each, then each one's thresholds, a word for each of this sum's blocks: the done prefix
     of that sum the block waits for (metal_zelloth: its blocks starting at or before this block's last element),
     as the pool's combine followed an earlier one chunk by chunk (zelloth_needs); a payload: K inputs {landed data,
     landed word, summed
     word (0: none)}, then R runs {first, elements, K x {scalar, phase, step, shift}}: element j of a run needs
     scalar + ((phase + j * step) >> shift) of its input's word; then each block's cell {its sum's word offset, its
     run's, its place in that run in blocks}, in their global order (mesh-graph.h mesh_graph_cells, laid by
     metal_zelloth); then each sum's done word, the threadgroups' `ended`, the first sum with open blocks `low`, each
     block's state (0 open, 1 claimed, 2 computed), each sum's computed prefix and its first unclaimed block, all zero at
     the start.  One threadgroup a core for the whole group (all resident), each claiming whichever sum's first open
     block is ready (mesh-graph.h mesh_graph_claim, a sum a run: its inputs landed past the block and each earlier sum
     it waits on done, zelloth_ready; a later block of a sum is ready only once its first open one is, its inputs'
     units landing in order), not the next block in a fixed order, which stood bandwidth once a sum waited on another
     (test-ov), nor any block a scan over all of them finds, whose loads took 64 MiB all-reduce from 4.8 ms to 11
     (test-cz); the threadgroup folds it, marks it computed and moves its sum's
     prefix past every computed block, then stores each input's summed count and the done word for the prefix it saw,
     again while the prefix moves (a count published late is one the prefix passed, so the bridge acts on none too
     high: mesh-graph.h mesh_graph_settle).  A thread 0 that finds nothing ready ends
     once it has looked at p.polls blocks, each at least a load, as a wait at its bound does (mesh_step_check), every threadgroup once one has failed; each counts
     itself `ended` as it leaves, the table's end (metal_done).  FOLD writes element j: ZFOLD_OP an operator's, ZFOLD_MERGE a partials merge's row of p.op floats
     (merge_row, after its definition below), the own row merged with each input's in turn, any order a valid one */
  "#define ZFOLD_OP(T, OP) T v = mesh_payload(own + j); "
  "for (uint k = 0; k < K; k++) v = OP(v, mesh_payload((volatile coherent(system) device const T *)in[3 * k] + j), p.op); out[j] = v;\n"
  "#define ZFOLD_MERGE(T, OP) const ulong row = j * ulong(p.op); "
  "merge_row<false>(out + row, own + row, (volatile coherent(system) device const float *)in[0] + row, p.op); "
  "for (uint k = 1; k < K; k++) merge_row<true>(out + row, out + row, (volatile coherent(system) device const float *)in[3 * k] + row, p.op);\n"
  "#define ZELLOTH(name, T, OP) ZELLOTH_OF(name, T, OP, ZFOLD_OP)\n"
  MESH_GRAPH_METAL
  "struct zelloth_runs { device const ulong *z; uint head(uint r) const { return uint(z[4 + 9 * ulong(r) + 5]); } "
  "uint count(uint r) const { return uint(z[4 + 9 * ulong(r) + 6]); } };\n"
  "struct zelloth_needs { device const ulong *z; ulong block, seen, want; bool lost; "
  "device const ulong *sum, *run, *in; ulong end, local; uint K; "
  "uint count(ulong q, uint r) { device const ulong *cell = z + z[3] + 3 * q; sum = z + 4 + 9 * ulong(r); run = z + cell[1]; "
  "in = z + sum[7]; end = min((cell[2] + 1) * block, run[1]); local = q - sum[5]; K = uint(sum[0]); "
  "return K + (sum[8] ? uint(z[sum[8]]) : 0u); } "
  "ulong at(ulong q, uint r, uint k, thread ulong &need) const { (void)q; (void)r; "
  "if (k < K) { device const ulong *x = run + 2 + 4 * ulong(k); "
  "need = x[0] + ((x[1] + (end - 1) * x[2]) >> x[3]); return *(volatile coherent(system) device const ulong *)in[3 * k + 1]; } "
  "device const ulong *w = z + sum[8] + 1 + 2 * ulong(k - K); "
  "need = z[w[1] + local]; return *(volatile coherent(system) device const ulong *)w[0]; } "
  "void short_of(ulong v, ulong w) { seen = v; want = w; if (v == ~0UL) lost = true; } };\n"
  "struct zelloth_publish { device const ulong *z, *sum, *in; ulong block; uint K, head; "
  "void at(uint p) { device const ulong *c = z + z[3] + 3 * ulong(head + p - 1), *r = z + c[1]; "
  "const ulong e2 = min((c[2] + 1) * block, r[1]); "
  "for (uint k = 0; k < K; k++) if (in[3 * k + 2]) { device const ulong *x = r + 2 + 4 * ulong(k); "
  "mesh_publish((volatile coherent(system) device ulong *)in[3 * k + 2], x[0] + ((x[1] + (e2 - 1) * x[2]) >> x[3])); } "
  "mesh_publish((volatile coherent(system) device ulong *)sum[4], ulong(p)); } };\n"
  "struct zelloth_stop { constant mesh_args *p; device uchar *vb; thread zelloth_needs *needs; "
  "bool now() { return mesh_step_failed(*p, vb) || needs->lost; } bool bound(ulong looked) const { return looked >= ulong(p->polls); } };\n"
  "#define ZELLOTH_OF(name, T, OP, FOLD) HEAD(zelloth_##name) { ENTER(i == 0) "
  "device const ulong *z = (device const ulong *)(s + p.src); const ulong S = z[0], block = z[1], blocks = z[2]; "
  "device const ulong *done = z + z[3] + 3 * blocks; device atomic_uint *ended = (device atomic_uint *)(done + S); "
  "device atomic_uint *low = (device atomic_uint *)(done + S + 1), *state = (device atomic_uint *)(done + S + 2), "
  "*prefix = state + blocks, *next = prefix + S; "
  "const zelloth_runs shape = {z}; "
  "for (;;) { "
  "if (l == 0) { zelloth_needs needs = {z, block, 0, 0, false, nullptr, nullptr, nullptr, 0, 0, 0u}; zelloth_stop stop = {&p, vb, &needs}; "
  "const ulong got = mesh_graph_take(low, next, state, uint(S), shape, needs, stop); "
  "if (got == ~0UL && atomic_load_explicit(low, memory_order_relaxed) < uint(S) && !mesh_step_failed(p, vb)) "
  "mesh_step_check(p, needs.lost ? ~0UL : needs.seen, needs.want, vb); "
  "tg[0] = got; } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); const ulong q = tg[0]; threadgroup_barrier(mem_flags::mem_threadgroup); "
  "if (q == ~0UL) break; "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "device const ulong *cell = z + z[3] + 3 * q, *sum = z + cell[0], *run = z + cell[1], *in = z + sum[7]; "
  "const uint K = uint(sum[0]); "
  "volatile coherent(system) device const T *own = (volatile coherent(system) device const T *)sum[2]; "
  "volatile coherent(system) device T *out = (volatile coherent(system) device T *)sum[3]; "
  "const ulong first = run[0], b = cell[2] * block, end = min(b + block, run[1]); "
  "for (ulong e = b + l; e < end; e += t) { const ulong j = first + e; FOLD(T, OP) } "
  "threadgroup_barrier(mem_flags::mem_device); "
  "if (l == 0) { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "atomic_store_explicit(state + q, 2u, memory_order_relaxed); "
  "zelloth_publish publish = {z, sum, in, block, K, uint(sum[5])}; "
  "mesh_graph_settle(state, uint(sum[5]), uint(sum[6]), prefix + (cell[0] - 4) / 9, publish); } "
  "threadgroup_barrier(mem_flags::mem_threadgroup); } "
  "if (l == 0) { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE)); "
  "atomic_fetch_add_explicit(ended, 1u, memory_order_relaxed); } }\n"
  "#define INTEGER(name, T, SIGNED) ZELLOTH(name, T, op_int<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); ulong c = ulong(T(p.scalar)); LOOP x[k] = T(ulong(y[k]) * c); } "
  "HEAD(truncdiv_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); LOOP { T v = mesh_payload(y + k); bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) ZELLOTH(name, T, op_float<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) volatile coherent(system) device T *x = (volatile coherent(system) device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "#define FLOAT8(name, E5) ZELLOTH(name, uchar, op_f8<E5>) "
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
  "ZELLOTH_OF(merge, float, 0, ZFOLD_MERGE)\n";

/* every kernel by its kind: one list, the one numbering of a device's pipelines (nccl-mesh-metal.m kernels_of); each
   kernel's name, which the Metal library's lookup takes, once a process (nccl-mesh-metal.m kernels_of) and once an
   install (nccl-mesh-kernels.m) (newFunctionWithName, an interface the library does not implement: design/rubrics.md,
   Strings are messages).  The tracks pool's own kernel, the combine landings it ran and the words and lanes it read
   them by are deleted with it (design/zorvek.md "The pool's last users") */
#define KINDS_OF(X) \
  X(copy_plain_4) X(copy_land_4) X(copy_plain_1) X(copy_land_1) \
  X(premul_i8) X(truncdiv_i8) X(premul_u8) X(truncdiv_u8) X(premul_i32) X(truncdiv_i32) X(premul_u32) X(truncdiv_u32) \
  X(premul_i64) X(truncdiv_i64) X(premul_u64) X(truncdiv_u64) \
  X(premul_f16) X(premul_f32) X(premul_bf16) X(premul_e4m3) X(premul_e5m2) \
  X(publish) X(spin) \
  X(zelloth_i8) X(zelloth_u8) X(zelloth_i32) X(zelloth_u32) X(zelloth_i64) X(zelloth_u64) \
  X(zelloth_f16) X(zelloth_f32) X(zelloth_bf16) X(zelloth_e4m3) X(zelloth_e5m2) \
  X(zelloth_merge)
#define KIND_ENUM(name) KIND_##name,
#define KIND_NAME(name) #name,
enum { KINDS_OF(KIND_ENUM) KINDS };
static const char *const NAME[KINDS] = {KINDS_OF(KIND_NAME)};


#endif

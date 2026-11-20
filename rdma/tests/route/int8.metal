// int8 x int8 -> int32 products on the GPU's matrix units (Metal 4, MetalPerformancePrimitives' matmul2d), and the
// quantizers feeding them: every 2D product of the policy's forward and backward (net.swift).
//
// Exact integer coverage: a product's int32 sum is exact while (terms) x 127^2 < 2^31, so for at most 133,000 terms.
//   forward and dgrad reduce over K <= 192 (a layer's width): one scale a row of each operand over its whole K, one
//     matmul2d run a tile, the scales applied once in the epilogue: y = fma(float(sum) sa[m], sb[n], bias[n]).
//   wgrad reduces over the rows (up to ~10^5): blocks of 256 rows, one scale a column a block, each block's exact sum
//     scaled into an fp32 accumulator (NVFP4's structure, its 16-element blocks here 256); the blocks split across
//     threadgroups, their partial sums reduced in a fixed order (deterministic).
// The numerics (codes, scales, the epilogue's fma) are int8mm.h's, shared by every user, so they agree bit for bit.
//
// No outlier-dominated operand: before its absmax scale, each operand is rotated along the dimension the product reduces,
// by a normalized Walsh-Hadamard transform within 64-blocks (y = H x / 8, H's entries +-1; both operands rotated, so
// (H a / 8) . (H b / 8) = a . b and the product is unchanged), which spreads any one large entry over its block.
// Activations round to nearest; gradients (seed != 0) stochastically, unbiased.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// a logical row r's physical row: (r / seg) stride + off + r % seg (tasks stacked: seg rows a task, stride apart)
struct RowMap { uint seg, stride, off, pad; };
static inline uint rmap(RowMap m, uint r) { return m.seg == 0xffffffffu ? m.off + r : (r / m.seg) * m.stride + m.off + r % m.seg; }
// the products' epilogue flags, a function constant: each pipeline compiled for its own (net.swift's pipe(name, flags))
constant uint FLAGS [[function_constant(0)]];

#include "int8mm.h"   // the numerics: codes, scales, the epilogue (shared with forward.c and int8check.swift)
static inline int8_t quant(float v, float inv, uint seed, uint at) { return i8_code(v, inv, seed, at); }
static inline float inverse(float m) { return i8_inverse(m); }

// rows: logical row r of x (map, ld), its first `valid` columns (the rest 0), padded to `cols` (64, 128, 192 or 256)
// -> q [rows][cols], s [rows] (one scale the whole row). Eight lanes a 64-block, 8 contiguous columns a lane (column
// 8 li + k of a row, li the lane's place in its row's lanes): rows of 64 columns 4 a SIMD group, of 128 2, else 1.
// rotate: the transform within each 64-block, its strides 1, 2, 4 inside a lane, 8, 16, 32 across lanes (the CPU
// mirror's order). Dispatched (rows / rows a SIMD group, rounded up) x 32 threads.
struct QR { uint rows, cols, ld, valid, rotate, seed, p0, p1; RowMap map; };
template <typename T> static inline void quant_rows_t(device const T *x, device int8_t *q, device float *s, constant QR &p, uint gid) {
  const uint lane = gid % 32, lpr = p.cols <= 64 ? 8 : p.cols <= 128 ? 16 : 32, row = gid / 32 * (32 / lpr) + lane / lpr, li = lane % lpr, lb = li % 8;
  const bool live = row < p.rows && li * 8 < p.cols;
  float v[8];
  const device T *src = x + size_t(live ? rmap(p.map, row) : 0) * p.ld;
  for (uint k = 0; k < 8; k++) { const uint j = li * 8 + k; v[k] = live && j < p.valid ? float(src[j]) : 0.0f; }
  if (p.rotate) {
    for (uint st = 1; st < 8; st <<= 1) for (uint k = 0; k < 8; k++) if (!(k & st)) { const float a = v[k], c = v[k | st]; v[k] = a + c; v[k | st] = a - c; }
    for (ushort st = 1; st < 4; st <<= 1) for (uint k = 0; k < 8; k++) { const float o = simd_shuffle_xor(v[k], st); v[k] = (lb & st) ? o - v[k] : v[k] + o; }
    for (uint k = 0; k < 8; k++) { const float o = simd_shuffle_xor(v[k], ushort(4)); v[k] = (lb & 4) ? (o - v[k]) * 0.125f : (v[k] + o) * 0.125f; }
  }
  float m = 0;
  for (uint k = 0; k < 8; k++) m = max(m, abs(v[k]));
  for (ushort st = 1; st < lpr; st <<= 1) m = max(m, simd_shuffle_xor(m, st));
  if (!live) return;
  const float inv = inverse(m);
  if (li == 0) s[row] = i8_scale(m);
  uint lo = 0, hi = 0;
  for (uint k = 0; k < 8; k++) {
    const uint code = uint(uchar(quant(v[k], inv, p.seed, row * p.cols + li * 8 + k)));
    if (k < 4) lo |= code << (8 * k); else hi |= code << (8 * (k - 4));
  }
  *(device uint2 *)(q + size_t(row) * p.cols + li * 8) = uint2(lo, hi);
}

// fp32, fp16 and bf16 inputs
#define QROWS(name, T) kernel void name(device const T *x [[buffer(0)]], device int8_t *q [[buffer(1)]], device float *s [[buffer(2)]], \
  constant QR &p [[buffer(3)]], uint gid [[thread_position_in_grid]]) { quant_rows_t<T>(x, q, s, p, gid); }
QROWS(quant_rows, float)
QROWS(quant_rows_h, half)
QROWS(quant_rows_b, bfloat)

// columns: rows [0, rows) of x from row `off` (ld), the first `valid` of them data (the rest 0); columns [0, xcols) of x,
// column `ones` (if < cols) all ones on the valid rows (a bias's gradient), the rest of [0, cols) 0 -> q [rows][cols],
// s [cols][rows / block]: one scale a column a block of `block` rows (64, 128, 192 or 256); rotate: the transform along
// the rows within each 64-block. A threadgroup a (block, 32 columns), eight SIMD groups of four columns each: each 64-row
// sub-block loaded coalesced into threadgroup memory, a column's 64 rows two a lane (rows l, l + 32, as quant_rows'
// columns), rotated by shuffles and kept in registers; the block's maximum; the codes back through threadgroup memory.
struct QC { uint rows, cols, ld, valid, block, rotate, seed, ones, off, xcols, p0, p1; };
template <typename T> static inline void quant_cols_t(device const T *x, device int8_t *q, device float *s, constant QC &p, uint2 tg, uint tid, uint sg, uint lane,
                                                      threadgroup float (&tile)[64][33], threadgroup int8_t (&codes)[256][32]) {
  const uint nb = p.rows / p.block, b = tg.y, c0 = tg.x * 32, nsb = p.block / 64;
  float v[4][4][2];   // [sub-block][column][half]
  for (uint sb = 0; sb < nsb; sb++) {
    const uint r0 = b * p.block + sb * 64;
    for (uint k = tid; k < 64 * 8; k += 256) {   // four consecutive columns a thread
      const uint r = k / 8, cl = k % 8 * 4, row = r0 + r;
      const device T *src = x + size_t(p.off + row) * p.ld;
      for (uint j = 0; j < 4; j++) {
        const uint col = c0 + cl + j;
        tile[r][cl + j] = row >= p.valid || col >= p.cols ? 0.0f : col < p.xcols ? float(src[col]) : col == p.ones ? 1.0f : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 4; j++) {
      const uint cl = sg * 4 + j;
      float a = tile[lane][cl], c = tile[lane + 32][cl];
      if (p.rotate) {
        for (ushort st = 1; st < 32; st <<= 1) {
          const float oa = simd_shuffle_xor(a, st), oc = simd_shuffle_xor(c, st);
          a = (lane & st) ? oa - a : a + oa; c = (lane & st) ? oc - c : c + oc;
        }
        const float t = a; a = (t + c) * 0.125f; c = (t - c) * 0.125f;
      }
      v[sb][j][0] = a; v[sb][j][1] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < 4; j++) {
    const uint cl = sg * 4 + j, col = c0 + cl;
    float m = 0;
    for (uint sb = 0; sb < nsb; sb++) m = max(m, max(abs(v[sb][j][0]), abs(v[sb][j][1])));
    m = simd_max(m);
    const float inv = inverse(m);
    if (lane == 0 && col < p.cols) s[size_t(col) * nb + b] = i8_scale(m);
    for (uint sb = 0; sb < nsb; sb++) for (uint h = 0; h < 2; h++) {
      const uint r = sb * 64 + h * 32 + lane, row = b * p.block + r;
      codes[r][cl] = quant(v[sb][j][h], inv, p.seed, row * p.cols + col);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = tid; k < p.block * 32; k += 256) {
    const uint r = k / 32, cl = k % 32, col = c0 + cl;
    if (col < p.cols) q[size_t(b * p.block + r) * p.cols + col] = codes[r][cl];
  }
}

#define QCOLS(name, T) kernel void name(device const T *x [[buffer(0)]], device int8_t *q [[buffer(1)]], device float *s [[buffer(2)]], \
  constant QC &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float tile[64][33]; threadgroup int8_t codes[256][32]; quant_cols_t<T>(x, q, s, p, tg, tid, sg, lane, tile, codes); }
QCOLS(quant_cols, float)
QCOLS(quant_cols_h, half)
QCOLS(quant_cols_b, bfloat)

// forward (NT: a [M][K] against b [N][K]) and dgrad (NN: a [M][K] against b [K][N]): the whole K one run, a threadgroup a
// 64 x 32 tile of the output, four SIMD groups. The epilogue, in order: y = fma(float(sum) sa[m], sb[n], bias[n]);
// relu; gates (sigmoid on columns < N / 2, tanh past: a minGRU's z and h~); mask (y where aux > 0, else 0: relu's
// derivative); + res; stored at out's row map.
enum { BIAS = 1, RELU = 2, GATES = 4, MASK = 8, RES = 16 };
struct GP { uint M, N, K, flags, ldo, ldr, lda, auxSeg; RowMap out, res; uint auxOff, p0, p1, p2; };
// Stores: where out's map is an offset alone (seg NONE) and its stride N, the results go back into the int32 cooperative
// tensor's slots (bit-cast) and out by its coalesced store (the fp32 buffer viewed as int32); likewise the residual is
// read by a coalesced load; else element by element.
#define FWD(name, TR, BEXT, BSL) \
kernel void name(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], device const float *sa [[buffer(2)]], device const float *sb [[buffer(3)]], \
                 device float *out [[buffer(4)]], device const float *bias [[buffer(5)]], device const float *res [[buffer(6)]], device const float *aux [[buffer(7)]], \
                 constant GP &p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]]) { \
  const int M = p.M, N = p.N, K = p.K, m0 = tg.y * 64, n0 = tg.x * 32; \
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, M)); \
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2> BEXT); \
  constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, TR); \
  matmul2d<d, execution_simdgroups<4>> op; \
  auto tA = A.slice(0, m0); auto tB = B.slice BSL; \
  auto c = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>(); \
  for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0; \
  op.run(tA, tB, c); \
  const bool tiled = p.out.seg == 0xffffffffu && p.ldo == uint(N); \
  const bool resTiled = (FLAGS & RES) && p.res.seg == 0xffffffffu && p.ldr == uint(N); \
  auto r = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>(); \
  if (resTiled) { \
    auto Rt = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>((device int32_t *)(res + size_t(p.res.off) * N), dextents<int32_t, 2>(N, M)); \
    r.load(Rt.slice(n0, m0)); \
  } \
  for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) { \
    auto at = c.get_multidimensional_index(i); const int n = n0 + at[0], m = m0 + at[1]; \
    if (m >= M || n >= N) continue; \
    float y = i8_out(c[i], sa[m], sb[n], (FLAGS & BIAS) ? bias[n] : 0.0f); \
    if (FLAGS & RELU) y = max(y, 0.0f); \
    if (FLAGS & GATES) y = n < N / 2 ? 1.0f / (1.0f + exp(-y)) : tanh(clamp(y, -20.0f, 20.0f)); \
    if ((FLAGS & MASK) && !(aux[size_t(p.auxOff + uint(m) % p.auxSeg) * p.lda + n] > 0)) y = 0; \
    if (FLAGS & RES) y += resTiled ? as_type<float>(r[i]) : res[size_t(rmap(p.res, m)) * p.ldr + n]; \
    if (tiled) c[i] = as_type<int32_t>(y); else out[size_t(rmap(p.out, m)) * p.ldo + n] = y; \
  } \
  if (tiled) { \
    auto Ot = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>((device int32_t *)(out + size_t(p.out.off) * N), dextents<int32_t, 2>(N, M)); \
    c.store(Ot.slice(n0, m0)); \
  } \
}
FWD(gemm_nt, true, (K, N), (0, n0))
FWD(gemm_nn, false, (N, K), (n0, 0))

// The fused product: a threadgroup 32 whole output rows (64-column tiles in turn, each its own matmul2d run, into
// threadgroup memory), then a row pass: mask, + res, rms (forward: y / rms(y), the 1 / rms into inv) or rms's backward
// (y = inv (y - n mean(y n)), n = aux's row, inv given), stored (fp32, out's map) if STORE, and the next product's
// int8 codes and scale (logical row m, cols = N padded to 64) by quant_rows' own lanes and butterfly, so the same codes a
// separate quant_rows of the stored values would give. Flags beyond FWD's: STORE, CODES, RMS, RMSB.
enum { STORE = 32, CODES = 64, RMS = 128, RMSB = 256, STOREH = 512, STOREB = 1024, AUXH = 2048 };   // STOREH/B: out is fp16 / bf16; AUXH: aux is fp16
struct GQ { uint M, N, K, flags, ldo, ldr, lda, auxSeg; RowMap out, res; uint auxOff, seed, p1, p2; };
#define QFWD(name, TR, BEXT, BSL) \
kernel void name(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], device const float *sa [[buffer(2)]], device const float *sb [[buffer(3)]], \
                 device float *out [[buffer(4)]], device const float *bias [[buffer(5)]], device const float *res [[buffer(6)]], device const float *aux [[buffer(7)]], \
                 constant GQ &p [[buffer(8)]], device int8_t *q [[buffer(9)]], device float *qs [[buffer(10)]], device float *inv [[buffer(11)]], \
                 threadgroup int32_t *tile [[threadgroup(0)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  const int M = p.M, N = p.N, K = p.K, cols = (N + 63) / 64 * 64, m0 = tg.x * 32; \
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, M)); \
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2> BEXT); \
  const int ldt = cols + 4;   /* rows 4 words apart: a lane's 8 sums, its neighbours' rows, on different banks */ \
  auto Tt = tensor<threadgroup int32_t, dextents<int32_t, 2>, tensor_inline>(tile, dextents<int32_t, 2>(cols, 32), array<int32_t, 2>{1, ldt}); \
  constexpr auto d = matmul2d_descriptor(32, 64, static_cast<int>(dynamic_extent), false, TR); \
  matmul2d<d, execution_simdgroups<4>> op; \
  for (int n0 = 0; n0 < cols; n0 += 64) { \
    auto tA = A.slice(0, m0); auto tB = B.slice BSL; \
    auto c = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>(); \
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0; \
    op.run(tA, tB, c); \
    c.store(Tt.slice(n0, 0)); \
  } \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  const uint lpr = cols <= 64 ? 8 : cols <= 128 ? 16 : 32, per = 32 / lpr, li = lane % lpr, lb = li % 8; \
  for (uint base = 0; base < 32; base += 4 * per) { \
    const uint r = base + sg * per + lane / lpr; const int m = m0 + int(r); \
    const bool live = m < M && li * 8 < uint(cols); \
    float v[8]; int sums[8]; \
    if (live) { const threadgroup int4 *t4 = (const threadgroup int4 *)(tile + r * ldt + li * 8); const int4 x0 = t4[0], x1 = t4[1]; \
      sums[0] = x0.x; sums[1] = x0.y; sums[2] = x0.z; sums[3] = x0.w; sums[4] = x1.x; sums[5] = x1.y; sums[6] = x1.z; sums[7] = x1.w; } \
    const float sam = live ? sa[m] : 0.0f; \
    const size_t orow = live ? size_t(rmap(p.out, m)) * p.ldo : 0, rrow = (FLAGS & RES) && live ? size_t(rmap(p.res, m)) * p.ldr : 0; \
    const size_t arow = (FLAGS & (MASK | RMSB)) && live ? size_t(p.auxOff + uint(m) % p.auxSeg) * p.lda : 0; \
    /* a lane's 8 columns: whole 64-blocks (N % 64 == 0) and 16-byte aligned rows, so four-wide loads and stores */ \
    const uint n8 = li * 8; \
    float4 sb4[2] = {float4(0), float4(0)}, b4[2] = {float4(0), float4(0)}, r4[2] = {float4(0), float4(0)}, a4[2] = {float4(1), float4(1)}; \
    if (live) { \
      sb4[0] = *(device const float4 *)(sb + n8); sb4[1] = *(device const float4 *)(sb + n8 + 4); \
      if (FLAGS & BIAS) { b4[0] = *(device const float4 *)(bias + n8); b4[1] = *(device const float4 *)(bias + n8 + 4); } \
      if (FLAGS & RES) { r4[0] = *(device const float4 *)(res + rrow + n8); r4[1] = *(device const float4 *)(res + rrow + n8 + 4); } \
      if (FLAGS & MASK) { if (FLAGS & AUXH) { const device half4 *h4 = (const device half4 *)((const device half *)aux + arow + n8); a4[0] = float4(h4[0]); a4[1] = float4(h4[1]); } \
                          else { a4[0] = *(device const float4 *)(aux + arow + n8); a4[1] = *(device const float4 *)(aux + arow + n8 + 4); } } \
    } \
    for (uint k = 0; k < 8; k++) { \
      const uint n = n8 + k; float y = 0; \
      if (live) { \
        y = i8_out(sums[k], sam, sb4[k / 4][k % 4], (FLAGS & BIAS) ? b4[k / 4][k % 4] : 0.0f); \
        if (FLAGS & RELU) y = max(y, 0.0f); \
        if (FLAGS & GATES) y = n < uint(N) / 2 ? 1.0f / (1.0f + exp(-y)) : tanh(clamp(y, -20.0f, 20.0f)); \
        if ((FLAGS & MASK) && !(a4[k / 4][k % 4] > 0)) y = 0; \
        if (FLAGS & RES) y += r4[k / 4][k % 4]; \
      } \
      v[k] = y; \
    } \
    if (FLAGS & (RMS | RMSB)) { \
      float nn[8], t = 0; \
      for (uint k = 0; k < 8; k++) { \
        const uint n = li * 8 + k; \
        nn[k] = (FLAGS & RMSB) && live && n < uint(N) ? aux[arow + n] : 0.0f; \
        t += (FLAGS & RMS) ? v[k] * v[k] : v[k] * nn[k]; \
      } \
      for (ushort st = 1; st < lpr; st <<= 1) t += simd_shuffle_xor(t, st); \
      if (FLAGS & RMS) { const float sc = rsqrt(t / float(N) + 1e-6f); for (uint k = 0; k < 8; k++) v[k] *= sc; if (live && li == 0) inv[m] = sc; } \
      else { const float sc = live ? inv[uint(m) % p.auxSeg] : 0.0f, mn = t / float(N); for (uint k = 0; k < 8; k++) v[k] = sc * (v[k] - nn[k] * mn); } \
    } \
    if ((FLAGS & STORE) && live) { \
      if (FLAGS & STOREH) { device half4 *o = (device half4 *)((device half *)out + orow + n8); o[0] = half4(v[0], v[1], v[2], v[3]); o[1] = half4(v[4], v[5], v[6], v[7]); } \
      else if (FLAGS & STOREB) { device bfloat4 *o = (device bfloat4 *)((device bfloat *)out + orow + n8); o[0] = bfloat4(float4(v[0], v[1], v[2], v[3])); o[1] = bfloat4(float4(v[4], v[5], v[6], v[7])); } \
      else { *(device float4 *)(out + orow + n8) = float4(v[0], v[1], v[2], v[3]); *(device float4 *)(out + orow + n8 + 4) = float4(v[4], v[5], v[6], v[7]); } \
    } \
    if (FLAGS & CODES) { \
      for (uint st = 1; st < 8; st <<= 1) for (uint k = 0; k < 8; k++) if (!(k & st)) { const float x = v[k], z = v[k | st]; v[k] = x + z; v[k | st] = x - z; } \
      for (ushort st = 1; st < 4; st <<= 1) for (uint k = 0; k < 8; k++) { const float o = simd_shuffle_xor(v[k], st); v[k] = (lb & st) ? o - v[k] : v[k] + o; } \
      for (uint k = 0; k < 8; k++) { const float o = simd_shuffle_xor(v[k], ushort(4)); v[k] = (lb & 4) ? (o - v[k]) * 0.125f : (v[k] + o) * 0.125f; } \
      float mx = 0; for (uint k = 0; k < 8; k++) mx = max(mx, abs(v[k])); \
      for (ushort st = 1; st < lpr; st <<= 1) mx = max(mx, simd_shuffle_xor(mx, st)); \
      if (live) { \
        const float iv = inverse(mx); \
        if (li == 0) qs[m] = i8_scale(mx); \
        uint lo = 0, hi = 0; \
        for (uint k = 0; k < 8; k++) { const uint code = uint(uchar(quant(v[k], iv, p.seed, uint(m) * cols + li * 8 + k))); if (k < 4) lo |= code << (8 * k); else hi |= code << (8 * (k - 4)); } \
        *(device uint2 *)(q + size_t(m) * cols + li * 8) = uint2(lo, hi); \
      } \
    } \
  } \
}
QFWD(gemmq_nt, true, (K, N), (0, n0))
QFWD(gemmq_nn, false, (N, K), (n0, 0))

// wgrad (TN: a [K][M] against b [K][N], K the rows): 256-row blocks, one scale a column a block; the blocks [z per,
// (z + 1) per) of split z summed into partial [z][M][N]; reduce_splits adds the splits in order
struct WP { uint M, N, K, per; };
kernel void gemm_tn(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], device const float *sa [[buffer(2)]], device const float *sb [[buffer(3)]],
                    device float *partial [[buffer(4)]], constant WP &p [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = p.M, N = p.N, K = p.K, nb = K / 256, m0 = tg.y * 64, n0 = tg.x * 32;
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(M, K));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(N, K));
  constexpr auto d = matmul2d_descriptor(64, 32, 256, true, false);
  matmul2d<d, execution_simdgroups<4>> op;
  float acc[32];
  int mi[32], ni[32];
  for (int i = 0; i < 32; i++) { acc[i] = 0; mi[i] = -1; ni[i] = 0; }
  { auto tA = A.slice(m0, 0); auto tB = B.slice(n0, 0);
    auto c = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
    for (uint16_t i = 0; i < c.get_capacity() && i < 32; ++i) if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i); ni[i] = n0 + at[0]; mi[i] = m0 + at[1]; if (mi[i] >= M || ni[i] >= N) mi[i] = -1;
    } }
  const int kb0 = tg.z * p.per, kb1 = min(nb, kb0 + int(p.per));
  for (int kb = kb0; kb < kb1; kb++) {
    auto tA = A.slice(m0, kb * 256); auto tB = B.slice(n0, kb * 256);
    auto c = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    op.run(tA, tB, c);
    for (uint16_t i = 0; i < c.get_capacity() && i < 32; ++i) if (mi[i] >= 0) acc[i] = i8_acc(c[i], sa[size_t(mi[i]) * nb + kb], sb[size_t(ni[i]) * nb + kb], acc[i]);
  }
  for (int i = 0; i < 32; ++i) if (mi[i] >= 0) partial[(size_t(tg.z) * M + mi[i]) * N + ni[i]] = acc[i];
}
kernel void reduce_splits(device const float *partial [[buffer(0)]], device float *out [[buffer(1)]], constant uint2 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x) return;   // p: (M N, splits)
  float s = 0;
  for (uint z = 0; z < p.y; z++) s += partial[size_t(z) * p.x + i];
  out[i] = s;
}

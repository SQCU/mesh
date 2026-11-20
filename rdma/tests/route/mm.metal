// The int8 x int8 -> int32 products on the GPU's matrix units (Metal 4, MetalPerformancePrimitives' matmul2d): the
// part of the earlier backend kept, measured against roofline.swift's ceiling (stages.swift) and retiled from it.
//   gemm_nt: C [M][N] = i8_out(A [M][K] . B [N][K]^T, sa[m], sb[n], bias[n]) (forward)
//   gemm_nn: the same against B [K][N] (dgrad)
//   gemm_tn: wgrad, A [K][M] . B [K][N] over K rows in blocks of I8_WBLOCK, one scale a column a block, each block's
//            exact sum into an fp32 accumulator (i8_acc); its threadgroups' partial sums [z][M][N]
//   gemm_wg: the same product over operands stored transposed (A [M][K], B [N][K]: each column's rows contiguous), as
//            the network's tiles write them; the one the trainer runs (measured 37-40 TOP/s at the network's shapes
//            against gemm_tn's layout at 18-21, the same tiles)
//   roof:    roofline.swift's kernel (int32 out, no epilogue), the ceiling stages.swift compares them with
// Each is a template over its tile (TM x TN outputs, SG SIMD groups); the instances are the roofline sweep's
// candidates: 64 x 32 (the earlier backend's), 64 x 64 and 64 x 128 with 4 SIMD groups, 128 x 64 with 8.
// The cooperative tensors' loops are unrolled and the destinations not cleared before a multiply (it writes them
// whole): with either, the same products measured 2x slower (stages).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#include "int8mm.h"

struct GP { uint M, N, K, bias; };
template <int TM, int TN, int SG, bool NN>
static inline void gemm(device int8_t *a, device int8_t *b, device const float *sa, device const float *sb, device float *out, device const float *bias, constant GP &p, uint2 tg) {
  const int M = p.M, N = p.N, K = p.K, m0 = tg.y * TM, n0 = tg.x * TN;
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, M));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, NN ? dextents<int32_t, 2>(N, K) : dextents<int32_t, 2>(K, N));
  constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, !NN);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto tA = A.slice(0, m0); auto tB = NN ? B.slice(n0, 0) : B.slice(0, n0);
  auto c = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  op.run(tA, tB, c);
  auto f = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  #pragma unroll
  for (uint16_t i = 0; i < c.get_capacity(); ++i) {
    auto at = c.get_multidimensional_index(i); const int n = n0 + at[0], m = m0 + at[1];
    f[i] = m < M && n < N ? i8_out(c[i], sa[m], sb[n], p.bias ? bias[n] : 0.0f) : 0.0f;
  }
  auto O = tensor<device float, dextents<int32_t, 2>, tensor_inline>(out, dextents<int32_t, 2>(N, M));
  f.store(O.slice(n0, m0));
}
#define GEMM(name, TM, TN, SG, NN) kernel void name##_##TM##x##TN##_##SG(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], \
  device const float *sa [[buffer(2)]], device const float *sb [[buffer(3)]], device float *out [[buffer(4)]], device const float *bias [[buffer(5)]], \
  constant GP &p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) { gemm<TM, TN, SG, NN>(a, b, sa, sb, out, bias, p, tg); }
GEMM(gemm_nt, 64, 32, 4, false) GEMM(gemm_nt, 64, 64, 4, false) GEMM(gemm_nt, 64, 128, 4, false) GEMM(gemm_nt, 128, 64, 8, false)
GEMM(gemm_nn, 64, 32, 4, true) GEMM(gemm_nn, 64, 64, 4, true) GEMM(gemm_nn, 64, 128, 4, true) GEMM(gemm_nn, 128, 64, 8, true)

// wgrad: blocks b0 + [z per, (z + 1) per) of I8_WBLOCK rows (within [b0, b1)); scales sa [K / blk][M], sb [K / blk][N]; partial [z][M][N].
// TRANSPOSED: operands [M][K], [N][K] (gemm_wg), else [K][M], [K][N] (gemm_tn)
struct WP { uint M, N, K, per, b0, b1, p0, p1; };   // K: the operands' row length (their stride); blocks [b0, b1)
template <int TM, int TN, int SG, bool TRANSPOSED>
static inline void wgrad(device int8_t *a, device int8_t *b, device const float *sa, device const float *sb, device float *partial, constant WP &p, uint3 tg) {
  const int M = p.M, N = p.N, K = p.K, m0 = tg.y * TM, n0 = tg.x * TN, z = tg.z;
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, TRANSPOSED ? dextents<int32_t, 2>(K, M) : dextents<int32_t, 2>(M, K));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, TRANSPOSED ? dextents<int32_t, 2>(K, N) : dextents<int32_t, 2>(N, K));
  constexpr auto d = matmul2d_descriptor(TM, TN, I8_WBLOCK, !TRANSPOSED, TRANSPOSED);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto tA0 = TRANSPOSED ? A.slice(0, m0) : A.slice(m0, 0); auto tB0 = TRANSPOSED ? B.slice(0, n0) : B.slice(n0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
  auto c = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), int32_t>();
  #pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  const int kb1 = min(int(p.b1), int(p.b0) + (z + 1) * int(p.per));
  for (int kb = p.b0 + z * p.per; kb < kb1; kb++) {
    auto tA = TRANSPOSED ? A.slice(kb * I8_WBLOCK, m0) : A.slice(m0, kb * I8_WBLOCK);
    auto tB = TRANSPOSED ? B.slice(kb * I8_WBLOCK, n0) : B.slice(n0, kb * I8_WBLOCK);
    op.run(tA, tB, c);
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
      auto at = c.get_multidimensional_index(i); const int n = min(n0 + at[0], N - 1), m = min(m0 + at[1], M - 1);
      acc[i] = i8_acc(c[i], sa[size_t(kb) * M + m], sb[size_t(kb) * N + n], acc[i]);
    }
  }
  auto O = tensor<device float, dextents<int32_t, 2>, tensor_inline>(partial + size_t(z) * M * N, dextents<int32_t, 2>(N, M));
  acc.store(O.slice(n0, m0));
}
#define WGRAD(name, TM, TN, SG, T) kernel void name##_##TM##x##TN##_##SG(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], \
  device const float *sa [[buffer(2)]], device const float *sb [[buffer(3)]], device float *partial [[buffer(4)]], constant WP &p [[buffer(5)]], \
  uint3 tg [[threadgroup_position_in_grid]]) { wgrad<TM, TN, SG, T>(a, b, sa, sb, partial, p, tg); }
WGRAD(gemm_tn, 64, 32, 4, false) WGRAD(gemm_tn, 64, 64, 4, false) WGRAD(gemm_tn, 128, 64, 8, false)
WGRAD(gemm_wg, 64, 32, 4, true) WGRAD(gemm_wg, 64, 64, 4, true) WGRAD(gemm_wg, 128, 64, 8, true)

// roofline.swift's kernel: C [M][N] int32 = A [M][K] . B [N][K]^T
template <int TM, int TN, int SG>
static inline void roof(device int8_t *a, device int8_t *b, device int32_t *out, constant GP &p, uint2 tg) {
  const int M = p.M, N = p.N, K = p.K, m0 = tg.y * TM, n0 = tg.x * TN;
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(K, M));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(K, N));
  auto C = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>(out, dextents<int32_t, 2>(N, M));
  constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto tA = A.slice(0, m0); auto tB = B.slice(0, n0);
  auto r = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  for (uint16_t i = 0; i < r.get_capacity(); ++i) if (r.is_valid_element(i)) r[i] = 0;
  op.run(tA, tB, r);
  r.store(C.slice(n0, m0));
}
#define ROOF(TM, TN, SG) kernel void roof_##TM##x##TN##_##SG(device int8_t *a [[buffer(0)]], device int8_t *b [[buffer(1)]], device int32_t *out [[buffer(4)]], \
  constant GP &p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) { roof<TM, TN, SG>(a, b, out, p, tg); }
ROOF(64, 32, 4) ROOF(64, 64, 4) ROOF(64, 128, 4) ROOF(128, 64, 8)

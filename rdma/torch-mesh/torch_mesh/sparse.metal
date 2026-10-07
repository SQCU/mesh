// Tile-sparse products on the GPU's tensor operations (Metal 4, MetalPerformancePrimitives matmul2d), the kept-tile
// predicate KEPT(k, n) a closure compiled into every kernel (torch_mesh/sparse.py): k a tile of W's rows, n of its
// columns, KB and NB their counts, `bits` the closure's captured bitmap where it reads one.  A simdgroup computes a
// 32 x 32 block, four a 64 x 64 threadgroup tile; fp16 operands, fp32 accumulation, outputs OUT (half or float).  Rows
// M need not fill the last block: the operands' extents bound the tensor operations' reads, and a store past row M is
// skipped.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant constexpr int RM = 32, CN = 32, KT = 32;
#define KEPT(k, n) (TILE_MOD)
#define OUT (OUT_TYPE)
struct dims { uint M, K, N, T; };

// Y[M, N] = X[M, K] . W_M[K, N]: each block's K-loop visits the row tiles k its column tile keeps
kernel void forward(device half* X [[buffer(0)]], device half* W [[buffer(1)]], device OUT* Y [[buffer(2)]],
                    constant dims& s [[buffer(3)]], device const uint* bits [[buffer(4)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const int M = s.M, K = s.K, N = s.N, T = s.T, KB = K / T, NB = N / T;
  const int row = int(tg.x) * 64 + (sg / 2) * RM, col = int(tg.y) * 64 + (sg % 2) * CN, n = col / T;
  auto A = tensor(X, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  auto B = tensor(W, dextents<int, 2>{N, K}, array<int, 2>{1, N});
  constexpr auto desc = matmul2d_descriptor(RM, CN, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto acc = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  for (int k = 0; k < KB; k++) {
    if (!KEPT(k, n)) continue;
    for (int at = k * T; at < (k + 1) * T; at += KT) { auto a = A.slice<KT, RM>(at, row); auto b = B.slice<CN, KT>(col, at); op.run(a, b, acc); }
  }
  for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
    auto e = acc.get_multidimensional_index(i);
    if (row + e[1] < M) Y[(row + e[1]) * N + col + e[0]] = OUT(acc[i]);
  }
}

// dX[M, K] = dY[M, N] . W_M^T: the same closure with its arguments' roles exchanged, W read transposed
kernel void input_grad(device half* dY [[buffer(0)]], device half* W [[buffer(1)]], device OUT* dX [[buffer(2)]],
                       constant dims& s [[buffer(3)]], device const uint* bits [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const int M = s.M, K = s.K, N = s.N, T = s.T, KB = K / T, NB = N / T;
  const int row = int(tg.x) * 64 + (sg / 2) * RM, kc = int(tg.y) * 64 + (sg % 2) * CN, k = kc / T;
  auto A = tensor(dY, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto B = tensor(W, dextents<int, 2>{N, K}, array<int, 2>{1, N});
  constexpr auto desc = matmul2d_descriptor(RM, CN, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto acc = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  for (int n = 0; n < NB; n++) {
    if (!KEPT(k, n)) continue;
    for (int at = n * T; at < (n + 1) * T; at += KT) { auto a = A.slice<KT, RM>(at, row); auto b = B.slice<KT, CN>(at, kc); op.run(a, b, acc); }
  }
  for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
    auto e = acc.get_multidimensional_index(i);
    if (row + e[1] < M) dX[(row + e[1]) * K + kc + e[0]] = OUT(acc[i]);
  }
}

// dW[K, N] = M (.) (X^T . dY): a block of a kept tile over every row, X read transposed; a dropped tile's blocks return
kernel void weight_grad(device half* X [[buffer(0)]], device half* dY [[buffer(1)]], device OUT* dW [[buffer(2)]],
                        constant dims& s [[buffer(3)]], device const uint* bits [[buffer(4)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const int M = s.M, K = s.K, N = s.N, T = s.T, KB = K / T, NB = N / T;
  const int kr = int(tg.x) * 64 + (sg / 2) * RM, col = int(tg.y) * 64 + (sg % 2) * CN, k = kr / T, n = col / T;
  if (!KEPT(k, n)) return;
  auto A = tensor(X, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  auto B = tensor(dY, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  constexpr auto desc = matmul2d_descriptor(RM, CN, KT, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto acc = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  for (int at = 0; at < M; at += KT) { auto a = A.slice<RM, KT>(kr, at); auto b = B.slice<CN, KT>(col, at); op.run(a, b, acc); }
  for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
    auto e = acc.get_multidimensional_index(i);
    dW[(kr + e[1]) * N + col + e[0]] = OUT(acc[i]);
  }
}

// the closure over the tile grid as a bitmap (bit n * KB + k), one 32-bit word a thread: FlexAttention's
// create_block_mask, the closure evaluated once at its granule
kernel void evaluate(device uint* out [[buffer(2)]], constant dims& s [[buffer(3)]], device const uint* bits [[buffer(4)]],
                     uint word [[thread_position_in_grid]]) {
  const int K = s.K, N = s.N, T = s.T, KB = K / T, NB = N / T;
  uint value = 0;
  for (int b = 0; b < 32; b++) {
    const int i = int(word) * 32 + b;
    if (i >= KB * NB) break;
    const int k = i % KB, n = i / KB;
    if (KEPT(k, n)) value |= 1u << b;
  }
  out[word] = value;
}

// dW at the kept tiles only, one grid entry a block of a kept tile: the ordinal'th set bit of the closure's evaluated
// bitmap `bits` (evaluate) is the tile, found by a popcount scan of its words in each simdgroup
kernel void weight_grad_kept(device half* X [[buffer(0)]], device half* dY [[buffer(1)]], device OUT* dW [[buffer(2)]],
                             constant dims& s [[buffer(3)]], device const uint* bits [[buffer(4)]],
                             uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  const int M = s.M, K = s.K, N = s.N, T = s.T, KB = K / T, NB = N / T, side = T / 64, words = (KB * NB + 31) / 32;
  const uint ordinal = tg.x / uint(side * side), block = tg.x % uint(side * side);
  uint before = 0, tile = 0;
  for (int base = 0; base < words; base += 32) {
    const uint word = base + lane < words ? bits[base + lane] : 0u, count = popcount(word);
    const uint prefix = simd_prefix_exclusive_sum(count), total = simd_sum(count);
    if (ordinal < before + total) {
      const bool mine = ordinal >= before + prefix && ordinal < before + prefix + count;
      uint at = 0;
      if (mine) {
        uint left = ordinal - before - prefix, w = word;
        for (; left; left--) w &= w - 1;
        at = (base + lane) * 32 + ctz(w);
      }
      tile = simd_max(mine ? at : 0u);
      break;
    }
    before += total;
  }
  const int k = int(tile) % KB, n = int(tile) / KB;
  const int kr = k * T + int(block / side) * 64 + (sg / 2) * RM, col = n * T + int(block % side) * 64 + (sg % 2) * CN;
  auto A = tensor(X, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  auto B = tensor(dY, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  constexpr auto desc = matmul2d_descriptor(RM, CN, KT, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto acc = op.get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  for (int at = 0; at < M; at += KT) { auto a = A.slice<RM, KT>(kr, at); auto b = B.slice<CN, KT>(col, at); op.run(a, b, acc); }
  for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
    auto e = acc.get_multidimensional_index(i);
    dW[(kr + e[1]) * N + col + e[0]] = OUT(acc[i]);
  }
}

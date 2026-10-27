// Random orthogonal rotations by fast Walsh-Hadamard transforms (torch_mesh/structured.py): each threadgroup one
// block of N (a power of two, at most 4096) of one row, in threadgroup memory: y = (H_N / sqrt N) diag(sign) x, the
// log2 N butterfly stages between barriers.  fp32 in and out.
#include <metal_stdlib>
using namespace metal;
kernel void fwht(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], device const float *sign [[buffer(2)]],
                 constant uint *dims [[buffer(3)]], uint block [[threadgroup_position_in_grid]],
                 uint t [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float v[4096];
  const uint N = dims[0], blocks_per_row = dims[1], row = block / blocks_per_row;
  const uint offset = (block % blocks_per_row) * N;
  for (uint i = t; i < N; i += threads)
    v[i] = x[ulong(row) * dims[2] + ulong(offset + i) * dims[3]] * sign[ulong(offset + i) * dims[6]];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < N; h <<= 1) {
    for (uint i = t; i < N / 2; i += threads) {
      const uint j = (i / h) * 2 * h + (i % h);
      const float a = v[j], b = v[j + h];
      v[j] = a + b;
      v[j + h] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float scale = rsqrt(float(N));
  for (uint i = t; i < N; i += threads) y[ulong(row) * dims[4] + ulong(offset + i) * dims[5]] = v[i] * scale;
}

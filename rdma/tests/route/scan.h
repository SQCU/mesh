/* The minGRU recurrence's prefix, defined once. h_t = a_t h_{t-1} + b_t (a = 1 - m z, b = m z h~) is an affine map a
   position; composing them, (a2, b2) o (a1, b1) = (a2 a1, fma(a2, b1, b2)), is associative, so every position's state
   is a prefix scan, h_t = fma(A_t, h_0, B_t). The order: inclusive within each group of 32 positions (a SIMD group: the
   earlier element at offsets 1, 2, 4, 8, 16), then the groups' totals scanned alike, and each later group's elements
   composed with the total of the groups before it. net.metal's scans run it on SIMD groups (scan_affine_simd);
   forward.c, the deployed forward, runs it on arrays (scan_affine): the same operations in the same order. */
#ifndef ROUTE_SCAN_H
#define ROUTE_SCAN_H
#ifdef __METAL_VERSION__
static inline void scan_affine_simd(thread float &A, thread float &B, uint lane) {
  for (ushort d = 1; d < 32; d <<= 1) {
    const float pa = metal::simd_shuffle_up(A, d), pb = metal::simd_shuffle_up(B, d);
    if (lane >= d) { B = metal::fma(A, pb, B); A = A * pa; }
  }
}
#else
#include <math.h>
static inline void scan_affine(float *A, float *B, int n) {   /* in place, inclusive */
  for (int g = 0; g < n; g += 32) {
    const int m = n - g < 32 ? n - g : 32;
    for (int d = 1; d < 32; d <<= 1) for (int i = m - 1; i >= d; i--) { B[g + i] = fmaf(A[g + i], B[g + i - d], B[g + i]); A[g + i] *= A[g + i - d]; }
  }
  const int ng = (n + 31) / 32;
  if (ng < 2) return;
  float TA[32], TB[32];
  for (int k = 0; k < ng; k++) { const int last = k * 32 + 31 < n ? k * 32 + 31 : n - 1; TA[k] = A[last]; TB[k] = B[last]; }
  for (int d = 1; d < 32; d <<= 1) for (int i = ng - 1; i >= d; i--) { TB[i] = fmaf(TA[i], TB[i - d], TB[i]); TA[i] *= TA[i - d]; }
  for (int i = 32; i < n; i++) { const int k = i / 32; B[i] = fmaf(A[i], TB[k - 1], B[i]); A[i] *= TA[k - 1]; }
}
#endif
#endif

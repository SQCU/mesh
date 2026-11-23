/* The minGRU recurrence's prefix, defined once. h_t = a_t h_{t-1} + b_t (a = 1 - m z, b = m z h~) is an affine map a
   position; composing them, (a2, b2) o (a1, b1) = (a2 a1, fma(a2, b1, b2)), is associative, so a group's states are a
   prefix scan from the state before it, h_t = fma(A_t, h_in, B_t). The order: inclusive within each group of 8
   positions (the earlier element at offsets 1, 2, 4; on the GPU eight consecutive lanes, a lane a position), then each
   group from the state its previous group left (h_in: the run's state before the group), across a run's refreshes in
   order. The trainer's GPU runs it on SIMD groups (scan_affine_lanes8), forward.c, the deployed forward, on
   arrays (scan_group, scan_run): the same operations in the same order.
   The backward runs the same composition over the positions reversed: with u_t the gradient arriving at h_t, the
   gradient e_t = a_t (u_t + e_{t+1}) passed to h_{t-1} is the affine map e -> a_t e + a_t u_t, scanned from the latest
   position back (scan_affine_lanes8 on reversed lanes, the groups in turn), and h_t's own gradient is u_t + e_{t+1}. */
#ifndef ROUTE_SCAN_H
#define ROUTE_SCAN_H
enum { SCAN_GROUP = 8 };
#ifdef __METAL_VERSION__
/* a group's inclusive scan over eight consecutive lanes (a lane a position, four groups a SIMD group) */
static inline void scan_affine_lanes8(thread float &A, thread float &B, uint lane) {
  for (ushort d = 1; d < 8; d <<= 1) {
    const float pa = metal::simd_shuffle_up(A, d), pb = metal::simd_shuffle_up(B, d);
    if ((lane & 7) >= d) { B = metal::fma(A, pb, B); A = A * pa; }
  }
}
#else
#include <math.h>
/* one group of m <= 8 positions, inclusive, in place */
static inline void scan_group(float *A, float *B, int m) {
  for (int d = 1; d < 8; d <<= 1) for (int i = m - 1; i >= d; i--) { B[i] = fmaf(A[i], B[i - d], B[i]); A[i] *= A[i - d]; }
}
/* n positions' maps (a, b) from state h: every position's state into out (a, b consumed); returns the last state */
static inline float scan_run(float *a, float *b, int n, float h, float *out) {
  for (int g = 0; g < n; g += SCAN_GROUP) {
    const int m = n - g < SCAN_GROUP ? n - g : SCAN_GROUP;
    scan_group(a + g, b + g, m);
    for (int i = 0; i < m; i++) out[g + i] = fmaf(a[g + i], h, b[g + i]);
    h = out[g + m - 1];
  }
  return h;
}
#endif
#endif

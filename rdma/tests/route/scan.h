/* The minGRU recurrence's prefix, defined once. h_t = a_t h_{t-1} + b_t (a = 1 - m z, b = m z h~) is an affine map a
   position; composing them, (a2, b2) o (a1, b1) = (a2 a1, fma(a2, b1, b2)), is associative, so a group's states are a
   prefix scan from the state before it, h_t = fma(A_t, h_in, B_t). The order: inclusive within each group of 8
   positions (the earlier element at offsets 1, 2, 4; on the GPU a group is eight lanes of a SIMD group's tile of 32
   rows, sg.h's layout), then each group from the state its previous group left (h_in: the run's state before the
   group). Across a run's refreshes (tiles of 32 positions) the GPU's learning forward composes each tile's total map
   (its four groups' totals composed in order) from the run's start: the tiles then scan in parallel. route.metal's lanes kernels
   run it on SIMD groups (scan_affine_simd), forward.c, the deployed forward, on arrays (scan_group, scan_run): the same
   operations in the same order.
   The backward runs the same composition over the positions reversed: with u_t the gradient arriving at h_t, the
   gradient e_t = a_t (u_t + e_{t+1}) passed to h_{t-1} is the affine map e -> a_t e + a_t u_t, scanned from the latest
   position back (scan_affine_simd on reversed lanes), and h_t's own gradient is u_t + e_{t+1}. */
#ifndef ROUTE_SCAN_H
#define ROUTE_SCAN_H
enum { SCAN_GROUP = 8 };
#ifdef __METAL_VERSION__
/* a group's inclusive scan over its eight positions r = 0..7, held by lanes (lane & 9) | 2 (r & 3) | 16 (r >> 2) (sg.h's
   layout) */
static inline ushort scan_lane(uint lane, uint r) { return ushort((lane & 0x9) | ((r & 3) << 1) | ((r >> 2) << 4)); }
static inline void scan_affine_group(thread float &A, thread float &B, uint r, uint lane) {
  for (ushort d = 1; d < 8; d <<= 1) {
    const ushort src = scan_lane(lane, r >= d ? r - d : 0);
    const float pa = metal::simd_shuffle(A, src), pb = metal::simd_shuffle(B, src);
    if (r >= d) { B = metal::fma(A, pb, B); A = A * pa; }
  }
}
/* the reverse: position r's map composed with the later ones' (r + 1 .. 7) */
static inline void scan_affine_group_reverse(thread float &A, thread float &B, uint r, uint lane) {
  for (ushort d = 1; d < 8; d <<= 1) {
    const ushort src = scan_lane(lane, r + d < 8 ? r + d : 7);
    const float pa = metal::simd_shuffle(A, src), pb = metal::simd_shuffle(B, src);
    if (r + d < 8) { B = metal::fma(A, pb, B); A = A * pa; }
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

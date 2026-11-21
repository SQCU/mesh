/* UPGrad's task weights, defined once (opt.metal's optimizer kernel computes them where the Gram matrix is made, so the
   optimizer's step never reads anything back): from G = J J^T of the three tasks' gradients, for each task i
   u_i = argmin_(u >= e_i) u^T G u (the dual of projecting g_i onto {x : J x >= 0}, whose projection is J^T u_i); the
   aggregate J^T (sum_i u_i) / 3. A 3-variable box QP each, by projected gradient (step 1 / (3 max |G|), a bound on
   G's largest eigenvalue), 500 sweeps. */
#ifndef ROUTE_UPGRAD_H
#define ROUTE_UPGRAD_H
#ifdef __METAL_VERSION__
#define UPGRAD_T thread
#define UPGRAD_ABS(x) metal::abs(x)
#else
#include <math.h>
#define UPGRAD_T
#define UPGRAD_ABS(x) fabsf(x)
#endif
static inline void upgrad3(UPGRAD_T const float *G, UPGRAD_T float *alpha) {
  float top = 0;
  for (int i = 0; i < 9; i++) top = UPGRAD_ABS(G[i]) > top ? UPGRAD_ABS(G[i]) : top;
  const float step = 1.0f / (3.0f * (top > 1e-30f ? top : 1e-30f));
  for (int t = 0; t < 3; t++) alpha[t] = 0;
  for (int i = 0; i < 3; i++) {
    float u[3] = {0, 0, 0}; u[i] = 1;
    for (int sweep = 0; sweep < 500; sweep++)
      for (int t = 0; t < 3; t++) {
        const float g = G[3 * t] * u[0] + G[3 * t + 1] * u[1] + G[3 * t + 2] * u[2], lo = t == i ? 1.0f : 0.0f, next = u[t] - step * 2 * g;
        u[t] = next > lo ? next : lo;
      }
    for (int t = 0; t < 3; t++) alpha[t] += u[t] / 3;
  }
}
#endif

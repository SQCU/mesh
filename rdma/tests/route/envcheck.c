/* The environment's own check (sim.c over the bridges, route.h's API as the trainer calls it): one all-gather and one
   all-reduce of each size on the measured cluster (tests/fabric-20261119.txt), in a training window of 4 refreshes,
   every context's w random; every refresh's tokens finite, every draw in its table (a valid, allowed token, its
   context one of the stream's, its log-probability under its w at most 0), the collectives all ending; and the
   runs' rate: many runs at once, a thread each (dispatch), runs and draws a second.  usage: envcheck [fabric] */
#include "route.h"
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static struct route_fabric fab;
static void load(const char *path) {
  FILE *f = fopen(path, "r"); if (!f) { perror(path); exit(2); }
  char line[256], names[ROUTE_N][16]; memset(&fab, 0, sizeof fab); fab.duplex = 1;
  while (fgets(line, sizeof line, f)) {
    char a[16], b[16], kind[16]; double rate, us; unsigned n = 1;
    if (line[0] == '#') continue;
    if (sscanf(line, "node %15s %15s", a, kind) == 2) { snprintf(names[fab.nodes], 16, "%s", a); if (!strcmp(kind, "compute")) fab.compute[fab.ncompute++] = fab.nodes; fab.nodes++; }
    else if (sscanf(line, "duplex %lf", &rate) == 1) fab.duplex = rate;
    else if (sscanf(line, "link %15s %15s %lf %lf %u", a, b, &rate, &us, &n) >= 4) {
      int x = -1, y = -1; for (int i = 0; i < fab.nodes; i++) { if (!strcmp(names[i], a)) x = i; if (!strcmp(names[i], b)) y = i; }
      const int l = fab.lanes++; fab.from[l] = x; fab.to[l] = y; fab.cables[l] = (int)n; fab.rate[l] = rate; fab.latency[l] = us * 1e3;
    }
  }
  fclose(f);
}
struct result { uint64_t draws, refreshes, bad, ops; struct route_costs c; };
static void one(int kind, double size, uint32_t key, struct result *out) {
  struct route_sim *x = route_sim_new();
  struct route_stream s; route_stream_one(&fab, kind, size, fab.compute[0], &s);
  s.op[1] = s.op[0]; s.op[1].arrival = 40e3; s.op[1].kind = kind ^ 1; s.ops = 2;   /* a second collective arriving midway */
  s.op[1].floor = route_op_floor(&fab, s.op[1].kind, size, fab.compute[0]);
  const struct route_window w = {0, 30e3, 10e3, 4, 2, 1};
  route_sim_begin(x, &s, &w, 0, key, key * 7 + 1);
  struct route_decision d; float *tok = malloc(sizeof(float) * ROUTE_CONTEXTS * ROUTE_IN), *wt = malloc(sizeof(float) * ROUTE_CONTEXTS * ROUTE_FS);
  uint64_t g = key;
  while (route_sim_decide(x, &d, tok)) {
    out->refreshes++;
    for (unsigned c = 0; c < d.contexts * ROUTE_IN; c++) if (!isfinite(tok[c])) out->bad++;
    for (unsigned l = 0; l < d.lanes; l++) for (int i = 0; i < ROUTE_IN; i++) if (!isfinite(d.lane[l][i])) out->bad++;
    for (unsigned c = 0; c < d.contexts * ROUTE_FS; c++) { g = g * 6364136223846793005ull + 1442695040888963407ull; wt[c] = (float)((double)(g >> 11) / 9007199254740992.0 * 4 - 2); }
    route_sim_answer(x, wt);
  }
  route_sim_costs(x, &out->c);
  const float gap = route_sim_logp_gap(x);
  if (gap > 1e-4f) { fprintf(stderr, "behaviour log-probability gap %g\n", gap); out->bad++; }
  for (int k = 0; k < route_sim_points(x); k++) {
    const int n = route_sim_draw_count(x, k);
    struct route_draw *r = malloc(sizeof *r * (size_t)(n + 1)); int *ctx = malloc(sizeof(int) * (size_t)(n + 1)), *first = malloc(sizeof(int) * (size_t)(n + 2));
    const int rows = route_sim_draws(x, k, r, ctx, first);
    for (int row = 0; row < rows; row++) for (int i = first[row]; i < first[row + 1]; i++) {
      float score[ROUTE_FS];
      const float lp = route_draw_score(&r[i], route_sim_w(x, k) + (size_t)ctx[row] * ROUTE_FS, score);
      if (!(lp <= 1e-5f) || ctx[row] < 0 || ctx[row] >= route_sim_contexts(x) || !split_of(&r[i], r[i].token).ok) out->bad++;
      out->draws++;
    }
    free(r); free(ctx); free(first);
  }
  out->ops = (uint64_t)out->c.ops;
  free(tok); free(wt); route_sim_free(x);
}
int main(int argc, char **argv) {
  load(argc > 1 ? argv[1] : "../fabric-20261119.txt");
  static const double sizes[] = {6144, 98304, 3145728};
  struct result total = {0};
  for (int k = 0; k < 2; k++) for (int i = 0; i < 3; i++) {
    struct result r = {0}; one(k, sizes[i], (uint32_t)(k * 3 + i + 1), &r);
    printf("%s %8.0f: %llu refreshes, %llu draws, %d of 2 collectives ended, completion over floor %.2f, piece latency %.2f, bad %llu\n",
           k ? "all-reduce" : "all-gather", sizes[i], (unsigned long long)r.refreshes, (unsigned long long)r.draws, r.c.ops, r.c.throughput, r.c.latency, (unsigned long long)r.bad);
    total.bad += r.bad; total.ops += r.ops;
  }
  /* the rate: N runs at once */
  const int N = argc > 2 ? atoi(argv[2]) : 64;
  struct result *rs = calloc((size_t)N, sizeof *rs);
  struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
  dispatch_apply((size_t)N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) { one((int)(i & 1), sizes[i % 3], (uint32_t)(100 + i), &rs[i]); });
  clock_gettime(CLOCK_MONOTONIC, &b);
  const double sec = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
  uint64_t draws = 0; for (int i = 0; i < N; i++) { draws += rs[i].draws; total.bad += rs[i].bad; }
  printf("%d runs at once: %.2f s, %.1f runs/s, %.0f draws/s\n", N, sec, N / sec, (double)draws / sec);
  if (total.bad || total.ops != 12) { printf("envcheck: failed (%llu bad, %llu of 12 collectives ended)\n", (unsigned long long)total.bad, (unsigned long long)total.ops); return 1; }
  printf("envcheck: passed\n");
  return 0;
}

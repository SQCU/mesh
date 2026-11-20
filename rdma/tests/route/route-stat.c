/* route-stat: reads trainers' telemetry tables (telemetry.h) where they lie (mmap, read only) and prints, each interval,
   one JSON object a table: the rates of its counters over the interval it sampled, its gauges, and its phases' times
   since its start (s). usage: route-stat [-i seconds] [-n samples] FILE...   (defaults: 1 s, until interrupted) */
#include "telemetry.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

struct route_tel_file *route_tel = 0;   /* (telemetry.h's declaration; this reader writes none) */
uint64_t route_tel_now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

int main(int argc, char **argv) {
  double interval = 1; long samples = -1; int a = 1;
  for (; a < argc && argv[a][0] == '-'; a++) {
    if (!strcmp(argv[a], "-i") && a + 1 < argc) interval = atof(argv[++a]);
    else if (!strcmp(argv[a], "-n") && a + 1 < argc) samples = atol(argv[++a]);
  }
  const int n = argc - a;
  if (n < 1) { fprintf(stderr, "usage: route-stat [-i seconds] [-n samples] FILE...\n"); return 2; }
  const struct route_tel_file **t = calloc(n, sizeof *t);
  uint64_t (*before)[RT_COUNT] = calloc(n, sizeof *before), *at = calloc(n, sizeof *at);
  for (int k = 0; k < n; k++) {
    const int fd = open(argv[a + k], O_RDONLY);
    if (fd < 0) { perror(argv[a + k]); return 1; }
    void *m = mmap(0, sizeof(struct route_tel_file), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    t[k] = m;
  }
  for (long s = 0; samples < 0 || s <= samples; s++) {
    const uint64_t now = route_tel_now();
    for (int k = 0; k < n; k++) {
      if (__atomic_load_n(&t[k]->magic, __ATOMIC_ACQUIRE) != ROUTE_TEL_MAGIC) continue;
      uint64_t v[RT_COUNT];
      for (int i = 0; i < RT_COUNT; i++) v[i] = __atomic_load_n(&t[k]->v[i], __ATOMIC_RELAXED);
      if (s > 0 && at[k]) {
        const double dt = (now - at[k]) / 1e9, since = v[RT_PHASE_START] ? (now - v[RT_PHASE_START]) / 1e9 : 0;
        printf("{\"rank\":%u,\"t\":%.1f", t[k]->rank, since);
        for (int i = 0; i < RT_FIRST_GAUGE; i++) printf(",\"%s\":%.4g", route_tel_names[i], (double)(v[i] - before[k][i]) / dt);
        for (int i = RT_FIRST_GAUGE; i < RT_FIRST_PHASE; i++) printf(",\"%s\":%llu", route_tel_names[i], (unsigned long long)v[i]);
        for (int i = RT_FIRST_PHASE + 1; i < RT_COUNT; i++) if (v[i]) printf(",\"%s\":%.2f", route_tel_names[i], (v[i] - v[RT_PHASE_START]) / 1e9);
        printf(",\"totals\":{\"steps\":%llu,\"optimizer_steps\":%llu,\"transactions\":%llu}}\n", (unsigned long long)v[RT_STEPS], (unsigned long long)v[RT_OPTIMIZER_STEPS], (unsigned long long)v[RT_TRANSACTIONS]);
      }
      memcpy(before[k], v, sizeof v); at[k] = now;
    }
    fflush(stdout);
    if (samples < 0 || s < samples) usleep((useconds_t)(interval * 1e6));
  }
  return 0;
}

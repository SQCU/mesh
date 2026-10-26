#include "mesh-statistics.h"
#include <time.h>

double mesh_statistics_now(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

struct mesh_statistics mesh_statistics_measure(double started, double observed,
                                                double useful_bytes, double units) {
  const double seconds = observed - started;
  return (struct mesh_statistics){seconds, useful_bytes / seconds * 1e-9, units / seconds};
}

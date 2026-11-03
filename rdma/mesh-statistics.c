#include "mesh-statistics.h"
#include "mesh.h"
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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

/* mesh-statistics.h mesh_statistics_streams (mesh-stat.c streams_watch's spin, kept in memory) */
long mesh_statistics_streams(unsigned long long *moves, unsigned long capacity, const volatile int *stop) {
  const int file = shm_open(MESH_NAME, O_RDONLY, 0);
  if (file < 0) return -errno;
  struct stat info;
  if (fstat(file, &info) || info.st_size < (off_t)sizeof(struct hdr)) { close(file); return -EINVAL; }
  struct hdr *m = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_SHARED, file, 0);
  close(file);
  if (m == MAP_FAILED) return -errno;
  if (m->magic != MESH_MAGIC || m->version != MESH_LAYOUT) { munmap(m, (size_t)info.st_size); return -EINVAL; }
  static uint64_t seen[MESH_STREAMS][4];
  for (uint32_t t = 0; t < MESH_STREAMS; t++) for (int w = 0; w < 4; w++) seen[t][w] = UINT64_MAX;
  unsigned long n = 0;
  while (!*stop && n + 4 * MESH_STREAMS <= capacity) {
    const uint64_t ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    for (uint32_t t = 0; t < MESH_STREAMS; t++) {
      const struct mesh_stream *x = mesh_stream_at(m, t);
      const uint64_t words[4] = {atomic_load(&x->owner), atomic_load(&x->ready), atomic_load(&x->sent), atomic_load(&x->landed)};
      for (uint32_t w = 0; w < 4; w++) if (words[w] != seen[t][w]) {
        moves[4 * n] = ns; moves[4 * n + 1] = t; moves[4 * n + 2] = w; moves[4 * n + 3] = words[w];
        seen[t][w] = words[w]; n++;
      }
    }
  }
  munmap(m, (size_t)info.st_size);
  return (long)n;
}

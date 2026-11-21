/* telemetry.h's table: a POSIX shared memory object (shm_open: memory, never storage), mapped MAP_SHARED, so route-stat
   reads the counters where they lie */
#include "telemetry.h"
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static struct route_tel_file private_table;
struct route_tel_file *route_tel = &private_table;

uint64_t route_tel_now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
int route_tel_open(const char *path, int rank, int world) {
  const int fd = shm_open(path, O_RDWR | O_CREAT, 0600);   /* path: the object's name, "/route-telemetry-0" */
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st) != 0 || (st.st_size < (off_t)sizeof(struct route_tel_file) && ftruncate(fd, sizeof(struct route_tel_file)) != 0)) { close(fd); return -1; }
  void *m = mmap(0, sizeof(struct route_tel_file), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (m == MAP_FAILED) return -1;
  struct route_tel_file *t = m;
  memset(t, 0, sizeof *t);
  t->version = ROUTE_TEL_VERSION; t->rank = (uint32_t)rank; t->world = (uint32_t)world; t->pid = (uint64_t)getpid(); t->clock_hz = 1000000000ull;
  memcpy(t->v, private_table.v, sizeof t->v);   /* what was counted before the table was mapped */
  __atomic_store_n(&t->magic, ROUTE_TEL_MAGIC, __ATOMIC_RELEASE);   /* last: a reader sees a whole table or none */
  route_tel = t;
  return 0;
}
void route_tel_inc(int k, uint64_t n) { route_tel_add(k, n); }
void route_tel_put(int k, uint64_t n) { route_tel_set(k, n); }
void route_tel_top(int k, uint64_t n) { route_tel_max(k, n); }
void route_tel_reach(int k) { route_tel_phase(k); }

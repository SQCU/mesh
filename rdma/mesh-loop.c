/* design/transport.md: N resident bridges on one host over loopverbs (mesh 800fe6bc^'s loop
   fabric, its driver metal-microbench b9dbc43^ tools/nccl_demo.py `loop`, on the resident bridge), each its own
   configuration (a canonical property list this tool writes: its node, region and cables), and the
   readings that exercise what two nodes cannot show: collectives over 3 to 8 node maps.

     mesh-loop up FAMILY N REGION_MIB DIRECTORY
       FAMILY one of line, ring, star, complete, torus (two rows of N/2), double (a line whose neighbours two cables
       join, parallel links); N bridges (.build/loop/mesh-flow-loop) of a
       region of REGION_MIB each, their configurations, logs and the map in DIRECTORY, every cable two loopverbs devices;
       returns once every bridge's map holds the N nodes and its own cables paired (a JSON line a node)
     mesh-loop nccl DIRECTORY COUNT host|walk
       N ranks, one a node: all-reduce, all-gather, reduce-scatter, broadcast and reduce from each root, and a send and
       receive around the ring, of COUNT float32 each; then the exchanges libnccl-mesh forwards along its routing table
       wherever two ranks no cable joins (the members between them relaying, piece by piece): the all-to-all, each root's
       gather and scatter, ncclMeshAlltoAllV of ragged counts (its first stage gathering them) and its whole form,
       ncclMeshAlltoAllPattern, sends to the rank half the ring on, one rank's send to another with every other rank in
       the group with nothing (ncclMeshGroupPattern, NULL) and its whole form, two sends to one peer in one group, the
       counted all-to-all, the same exchanges along the table ncclMeshCommGetRouting gives set again
       (ncclMeshCommSetRouting), and a send around the ring on each half of a split; every value exact; `host` the host
       path (a NULL stream), `walk` the Metal path's walk with its kernels on the CPU (nccl-mesh-host.c), never a GPU
     mesh-loop down DIRECTORY
       every bridge stopped (SIGTERM, a release: design/transport.md section 2), the fabric and every node's region
       unlinked

   MESH_LOOP_NAME names the loop's shared memory (loop_name), so loops on one host run apart.  A JSON line a reading,
   with its status; the exit status is nonzero where any failed. */
#include "nccl.h"
#include "mesh-collective.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define LOOP_NODES 8
#define LOOP_CABLES 32
extern char **environ;

/* the loop's shared-memory names: MESH_LOOP_NAME (a POSIX name of at most 20 bytes, its leading slash among them; unset:
   /meshloop), then -fabric for the fabric and the node's index for each node's region, so loops that run at once on one
   host name none of each other's (the same MESH_LOOP_NAME for its up, nccl and down) */
static const char *loop_name(void) {
  const char *name = getenv("MESH_LOOP_NAME");
  return name && *name == '/' && strlen(name) <= 20 ? name : "/meshloop";
}
static const char *loop_fabric(void) {
  static char fabric[32];
  snprintf(fabric, sizeof fabric, "%s-fabric", loop_name());
  return fabric;
}

struct loop_node { uint32_t node; char region[32]; pid_t pid; };
struct loop_cable { int a, b; uint16_t listen[2]; };
struct loop_map { char family[16]; int n, cables; uint64_t region_bytes; struct loop_node node[LOOP_NODES]; struct loop_cable cable[LOOP_CABLES]; };

static uint64_t clock_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int usage(void) {
  fprintf(stderr, "usage: mesh-loop up FAMILY N REGION_MIB DIRECTORY | nccl DIRECTORY COUNT host|walk | down DIRECTORY\n");
  return 2;
}

/* a loopback port no socket holds now */
static uint16_t free_port(void) {
  struct sockaddr_in6 at = {.sin6_len = sizeof at, .sin6_family = AF_INET6, .sin6_addr = IN6ADDR_LOOPBACK_INIT};
  socklen_t length = sizeof at;
  const int fd = socket(AF_INET6, SOCK_STREAM, 0);
  uint16_t port = 0;
  if (fd >= 0 && !bind(fd, (struct sockaddr *)&at, sizeof at) && !getsockname(fd, (struct sockaddr *)&at, &length)) port = ntohs(at.sin6_port);
  if (fd >= 0) close(fd);
  return port;
}

/* the family's cables over n nodes */
static int family_cables(const char *family, int n, struct loop_map *map) {
  map->cables = 0;
#define CABLE(x, y) do { if (map->cables == LOOP_CABLES) return -1; map->cable[map->cables++] = (struct loop_cable){x, y, {0, 0}}; } while (0)
  if (!strcmp(family, "line") || !strcmp(family, "ring")) {
    for (int i = 0; i + 1 < n; i++) CABLE(i, i + 1);
    if (!strcmp(family, "ring") && n > 2) CABLE(n - 1, 0);
  } else if (!strcmp(family, "double")) for (int i = 0; i + 1 < n; i++) { CABLE(i, i + 1); CABLE(i, i + 1); } else if (!strcmp(family, "star")) for (int i = 1; i < n; i++) CABLE(0, i);
  else if (!strcmp(family, "complete")) { for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) CABLE(i, j); }
  else if (!strcmp(family, "torus")) {
    if (n % 2 || n < 4) return -1;
    const int columns = n / 2;
    for (int r = 0; r < 2; r++) for (int c = 0; c < columns && columns > 1; c++) {
      if (columns == 2 && c == 1) break;
      CABLE(r * columns + c, r * columns + (c + 1) % columns);
    }
    for (int c = 0; c < columns; c++) CABLE(c, columns + c);
  } else return -1;
#undef CABLE
  return 0;
}

/* the map's file in DIRECTORY, read or written */
static int map_path(const char *directory, char *path, size_t room) { return snprintf(path, room, "%s/loop.map", directory) >= (int)room; }
static int map_write(const char *directory, const struct loop_map *map) {
  char path[PATH_MAX];
  if (map_path(directory, path, sizeof path)) return ENAMETOOLONG;
  FILE *f = fopen(path, "w");
  if (!f) return errno;
  fprintf(f, "%s %d %d %llu\n", map->family, map->n, map->cables, (unsigned long long)map->region_bytes);
  for (int i = 0; i < map->n; i++)
    fprintf(f, "%u %s %d\n", map->node[i].node, map->node[i].region, (int)map->node[i].pid);
  for (int c = 0; c < map->cables; c++)
    fprintf(f, "%d %d %u %u\n", map->cable[c].a, map->cable[c].b, map->cable[c].listen[0], map->cable[c].listen[1]);
  return fclose(f) ? errno : 0;
}
static int map_read(const char *directory, struct loop_map *map) {
  char path[PATH_MAX];
  if (map_path(directory, path, sizeof path)) return ENAMETOOLONG;
  FILE *f = fopen(path, "r");
  if (!f) return errno;
  unsigned long long region_bytes = 0;
  int ok = fscanf(f, "%15s %d %d %llu", map->family, &map->n, &map->cables, &region_bytes) == 4 && map->n > 0 && map->n <= LOOP_NODES &&
           map->cables >= 0 && map->cables <= LOOP_CABLES;
  map->region_bytes = region_bytes;
  for (int i = 0; ok && i < map->n; i++) {
    unsigned node; int pid;
    ok = fscanf(f, "%u %31s %d", &node, map->node[i].region, &pid) == 3;
    map->node[i].node = node; map->node[i].pid = (pid_t)pid;
  }
  for (int c = 0; ok && c < map->cables; c++) {
    unsigned x, y;
    ok = fscanf(f, "%d %d %u %u", &map->cable[c].a, &map->cable[c].b, &x, &y) == 4;
    map->cable[c].listen[0] = (uint16_t)x; map->cable[c].listen[1] = (uint16_t)y;
  }
  fclose(f);
  return ok ? 0 : EINVAL;
}

/* node i's canonical configuration (bridge_configuration.py's form, and its `loop`: mesh-link.c loop_read) */
static int plist_write(const char *directory, const struct loop_map *map, int i) {
  char path[PATH_MAX];
  if (snprintf(path, sizeof path, "%s/node%d.plist", directory, i) >= (int)sizeof path) return ENAMETOOLONG;
  FILE *f = fopen(path, "w");
  if (!f) return errno;
  fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
             "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict>\n"
             "<key>name</key><string>loop%d</string>\n<key>purpose</key><string>node %d of a %s of %d on the loopback fabric "
             "(mesh-loop)</string>\n<key>region_bytes</key><integer>%llu</integer>\n"
             "<key>loop</key><dict>\n<key>region</key><string>%s</string>\n<key>fabric</key><string>%s</string>\n"
             "<key>node</key><integer>%u</integer>\n<key>ports</key><array>\n",
          i, i, map->family, map->n, (unsigned long long)map->region_bytes, map->node[i].region, loop_fabric(), map->node[i].node);
  for (int c = 0; c < map->cables; c++) {
    const struct loop_cable *k = map->cable + c;
    if (k->a != i && k->b != i) continue;
    const int end = k->a == i ? 0 : 1;
    fprintf(f, "<dict><key>device</key><string>rdma_loop%d</string><key>listen</key><integer>%u</integer>"
               "<key>peer</key><integer>%u</integer></dict>\n", 2 * c + end, k->listen[end], k->listen[1 - end]);
  }
  fprintf(f, "</array>\n</dict>\n</dict></plist>\n");
  return fclose(f) ? errno : 0;
}

/* the loop bridge's binary, beside this tool's build: ../loop/mesh-flow-loop */
static int bridge_path(char *path, size_t room) {
  char self[PATH_MAX];
  uint32_t size = sizeof self;
  if (_NSGetExecutablePath(self, &size)) return ENAMETOOLONG;
  char *slash = strrchr(self, '/');
  if (!slash) return ENOENT;
  *slash = 0;
  return snprintf(path, room, "%s/../loop/mesh-flow-loop", self) >= (int)room ? ENAMETOOLONG : 0;
}

static int degree(const struct loop_map *map, int i) {
  int d = 0;
  for (int c = 0; c < map->cables; c++) d += map->cable[c].a == i || map->cable[c].b == i;
  return d;
}
/* every bridge's map holding the n nodes and its own cables paired (mesh_reached: a link an entry, so node i's bridge
   lists its degree once every cable has paired, two cables toward one node two), within `seconds` */
static int converged(const struct loop_map *map, double seconds) {
  const uint64_t deadline = clock_ns() + (uint64_t)(seconds * 1e9);
  int done = 0;
  while (!done && clock_ns() < deadline) {
    done = 1;
    for (int i = 0; i < map->n && done; i++) {
      struct mesh_node_health health[LOOP_NODES + 1];
      struct mesh_reach reached[MESH_STREAMS];
      uint32_t node = 0;
      if (mesh_loop_bind(map->node[i].region)) return EINVAL;
      const int records = mesh_nodes_health(health, LOOP_NODES + 1), count = mesh_reached(reached, MESH_STREAMS, &node);
      done = records == map->n && count == degree(map, i);
    }
    if (!done) usleep(100000);
  }
  return done ? 0 : ETIMEDOUT;
}

static int loop_up(const char *family, int n, uint64_t region_mib, const char *directory) {
  struct loop_map map = {.n = n, .region_bytes = region_mib << 20};
  snprintf(map.family, sizeof map.family, "%s", family);
  if (n < 2 || n > LOOP_NODES || !region_mib || family_cables(family, n, &map)) return usage();
  char binary[PATH_MAX];
  if (bridge_path(binary, sizeof binary)) return 1;
  mkdir(directory, 0755);
  shm_unlink(loop_fabric());
  for (int i = 0; i < n; i++) {
    map.node[i] = (struct loop_node){.node = (uint32_t)(i + 1)};
    snprintf(map.node[i].region, sizeof map.node[i].region, "%s%d", loop_name(), i);
    shm_unlink(map.node[i].region);
  }
  for (int c = 0; c < map.cables; c++) { map.cable[c].listen[0] = free_port(); map.cable[c].listen[1] = free_port(); }
  for (int i = 0; i < n; i++) {
    int error = plist_write(directory, &map, i);
    char plist[PATH_MAX], log[PATH_MAX];
    snprintf(plist, sizeof plist, "%s/node%d.plist", directory, i);
    snprintf(log, sizeof log, "%s/node%d.log", directory, i);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 1, log, O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&actions, 1, 2);
    char *argv[] = {binary, plist, NULL};
    if (!error) error = posix_spawn(&map.node[i].pid, binary, &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (error) { fprintf(stderr, "mesh-loop: node %d's bridge: %s\n", i, strerror(error)); return 1; }
  }
  if (map_write(directory, &map)) return 1;
  const int status = converged(&map, 60);
  for (int i = 0; i < n; i++)
    printf("{\"loop\":\"up\",\"family\":\"%s\",\"nodes\":%d,\"index\":%d,\"node\":%u,\"region\":\"%s\",\"pid\":%d,"
           "\"cables\":%d,\"converged\":%s}\n", family, n, i, map.node[i].node, map.node[i].region, (int)map.node[i].pid,
           degree(&map, i), status ? "false" : "true");
  return status ? 1 : 0;
}

static int loop_down(const char *directory) {
  struct loop_map map;
  if (map_read(directory, &map)) return usage();
  for (int i = 0; i < map.n; i++) if (map.node[i].pid > 0) kill(map.node[i].pid, SIGTERM);
  const uint64_t deadline = clock_ns() + UINT64_C(30000000000);
  int left = 1;
  while (left && clock_ns() < deadline) {
    left = 0;
    for (int i = 0; i < map.n; i++) if (map.node[i].pid > 0 && !kill(map.node[i].pid, 0)) left = 1;
    if (left) usleep(100000);
  }
  shm_unlink(loop_fabric());
  for (int i = 0; i < map.n; i++) shm_unlink(map.node[i].region);
  printf("{\"loop\":\"down\",\"nodes\":%d,\"stopped\":%s}\n", map.n, left ? "false" : "true");
  return left;
}

/* run `count` children, child i running `run(i)`, and wait for all; how many exited nonzero */
static int children(int count, int (^run)(int)) {
  pid_t pid[2 * LOOP_NODES];
  for (int i = 0; i < count; i++) {
    pid[i] = fork();
    if (!pid[i]) _exit(run(i) & 0xff);
  }
  int failed = 0;
  for (int i = 0; i < count; i++) {
    int status = 0;
    if (pid[i] < 0 || waitpid(pid[i], &status, 0) != pid[i] || !WIFEXITED(status) || WEXITSTATUS(status)) failed++;
  }
  return failed;
}

/* a reading's line, and whether it failed */
static int reading(const char *call, int root, int walk, int rank, int n, uint64_t count, ncclResult_t result, uint64_t wrong, ncclComm_t comm,
                   const char *on) {
  printf("{\"loop\":\"nccl\",\"path\":\"%s\",\"rank\":%d,\"ranks\":%d,\"call\":\"%s%s\",\"root\":%d,\"count\":%llu,\"result\":\"%s\","
         "\"wrong\":%llu,\"why\":\"%s\"}\n", walk ? "walk" : "host", rank, n, on, call, root, (unsigned long long)count,
         result ? ncclGetErrorString(result) : "ok", (unsigned long long)wrong, result ? ncclGetLastError(comm) : "");
  fflush(stdout);
  return result || wrong;
}

/* an exchange's values (every one an integer a float32 holds): rank `from`'s element k for rank `to`; ragged counts every
   rank computes alike; a counted all-to-all's rows from rank r for rank q, of ROW elements, at most 3 */
static float sent_of(int from, int to, uint64_t k) { return (float)(from * 3 + to * 5 + (int)(k % 89)); }
static uint64_t ragged_of(uint64_t count, int a, int b) { return count + (uint64_t)((a * 7 + b * 3) % 5); }
static uint64_t rows_of(int r, int q) { return (uint64_t)((r + 2 * q) % 3) + 1; }
#define ROW 4

/* the exchanges at rank `rank` of comm's n (the members between two ranks no cable joins forwarding their bytes), each
   checked exactly: the all-to-all, each root's gather and scatter, the all-to-all of ragged counts and its whole form,
   a send to the rank half the ring on, one rank's send to another with every other rank's part nothing and that
   group's whole pattern, two sends to one peer in one group, the counted all-to-all and its whole form; `on` a reading's
   prefix */
static int rank_exchanges(ncclComm_t comm, int rank, int n, uint64_t count, int walk, const char *on) {
  const size_t room = (size_t)(count + 8) * n * 2 + (size_t)ROW * 3 * n + 64;
  float *send = calloc(room, sizeof *send), *receive = calloc(room, sizeof *receive);
  int64_t counts[LOOP_NODES], *received = NULL;
  uint64_t *arrived = NULL;
  size_t sc[LOOP_NODES], rc[LOOP_NODES], full[LOOP_NODES * LOOP_NODES], ones[LOOP_NODES], all_ones[LOOP_NODES * LOOP_NODES];
  if (!send || !receive || ncclMemAlloc((void **)&received, LOOP_NODES * sizeof *received) || ncclMemAlloc((void **)&arrived, sizeof *arrived))
    return 1;
  ncclMeshStream stream = {.commandBuffer = (void *)1};
  cudaStream_t at = walk ? (cudaStream_t)&stream : NULL;
  int failed = 0;
#define B(p, offset) (walk ? (void *)&(ncclMeshBuffer){(p), (offset)} : (void *)((unsigned char *)(p) + (offset)))
#define CLEAR() do { for (size_t k = 0; k < room; k++) { send[k] = 0; receive[k] = -1.0f; } } while (0)
  /* the all-to-all, every count `count` */
  CLEAR();
  for (int b = 0; b < n; b++) for (uint64_t k = 0; k < count; k++) send[(size_t)b * count + k] = sent_of(rank, b, k);
  ncclResult_t result = ncclAlltoAll(B(send, 0), B(receive, 0), count, ncclFloat32, comm, at);
  uint64_t wrong = 0;
  for (int a = 0; !result && a < n; a++) for (uint64_t k = 0; k < count; k++) wrong += receive[(size_t)a * count + k] != sent_of(a, rank, k);
  failed += reading("alltoall", -1, walk, rank, n, count, result, wrong, comm, on);
  for (int root = 0; root < n; root++) {
    CLEAR();
    for (uint64_t k = 0; k < count; k++) send[k] = sent_of(rank, root, k);
    result = ncclGather(B(send, 0), B(receive, 0), count, ncclFloat32, root, comm, at);
    wrong = 0;
    for (int a = 0; !result && rank == root && a < n; a++) for (uint64_t k = 0; k < count; k++) wrong += receive[(size_t)a * count + k] != sent_of(a, root, k);
    failed += reading("gather", root, walk, rank, n, count, result, wrong, comm, on);
    CLEAR();
    for (int b = 0; rank == root && b < n; b++) for (uint64_t k = 0; k < count; k++) send[(size_t)b * count + k] = sent_of(root, b, k);
    result = ncclScatter(B(send, 0), B(receive, 0), count, ncclFloat32, root, comm, at);
    wrong = 0;
    for (uint64_t k = 0; !result && k < count; k++) wrong += receive[k] != sent_of(root, rank, k);
    failed += reading("scatter", root, walk, rank, n, count, result, wrong, comm, on);
  }
  /* ragged counts, each rank its own row and column (the first stage gathers them where a member relays), then whole */
  for (int whole = 0; whole < 2; whole++) {
    CLEAR();
    size_t k = 0;
    for (int b = 0; b < n; b++) { sc[b] = ragged_of(count, rank, b); rc[b] = ragged_of(count, b, rank); }
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) full[a * n + b] = ragged_of(count, a, b);
    for (int b = 0; b < n; b++) for (uint64_t j = 0; j < sc[b]; j++) send[k++] = sent_of(rank, b, j);
    result = whole ? ncclMeshAlltoAllPattern(B(send, 0), full, B(receive, 0), ncclFloat32, comm, at)
                   : ncclMeshAlltoAllV(B(send, 0), sc, B(receive, 0), rc, ncclFloat32, comm, at);
    wrong = 0; k = 0;
    for (int a = 0; !result && a < n; a++) for (uint64_t j = 0; j < rc[a]; j++) wrong += receive[k++] != sent_of(a, rank, j);
    failed += reading(whole ? "alltoall_pattern" : "alltoallv", -1, walk, rank, n, count, result, wrong, comm, on);
  }
  /* a send to the rank half the ring on, every rank's group a send and a receive */
  const int to = (rank + n / 2) % n, from = (rank + n - n / 2) % n;
  CLEAR();
  for (uint64_t k = 0; k < count; k++) send[k] = sent_of(rank, to, k);
  ncclGroupStart();
  ncclSend(B(send, 0), count, ncclFloat32, to, comm, at);
  ncclRecv(B(receive, 0), count, ncclFloat32, from, comm, at);
  result = ncclGroupEnd();
  wrong = 0;
  for (uint64_t k = 0; !result && k < count; k++) wrong += receive[k] != sent_of(from, rank, k);
  failed += reading("sendrecv_half", -1, walk, rank, n, count, result, wrong, comm, on);
  /* rank 0's send to rank n - 1, every other rank's part in the group nothing (its pattern alone), then its whole pattern */
  for (int whole = 0; whole < 2; whole++) {
    CLEAR();
    for (size_t k = 0; k < (size_t)n * n; k++) full[k] = 0;
    full[n - 1] = count * sizeof(float);
    for (uint64_t k = 0; k < count; k++) send[k] = sent_of(0, n - 1, k);
    ncclGroupStart();
    if (rank == 0) ncclSend(B(send, 0), count, ncclFloat32, n - 1, comm, at);
    if (rank == n - 1) ncclRecv(B(receive, 0), count, ncclFloat32, 0, comm, at);
    if (whole || (rank != 0 && rank != n - 1)) ncclMeshGroupPattern(whole ? full : NULL, comm, at);
    result = ncclGroupEnd();
    wrong = 0;
    for (uint64_t k = 0; !result && rank == n - 1 && k < count; k++) wrong += receive[k] != sent_of(0, n - 1, k);
    failed += reading(whole ? "send_one_pattern" : "send_one", -1, walk, rank, n, count, result, wrong, comm, on);
  }
  /* two sends to the next rank in one group, received in their order from the one before */
  const uint64_t first = count, second = count / 2 + 1;
  CLEAR();
  for (uint64_t k = 0; k < first + second; k++) send[k] = sent_of(rank, (rank + 1) % n, k);
  ncclGroupStart();
  ncclSend(B(send, 0), first, ncclFloat32, (rank + 1) % n, comm, at);
  ncclSend(B(send, first * sizeof(float)), second, ncclFloat32, (rank + 1) % n, comm, at);
  ncclRecv(B(receive, 0), first, ncclFloat32, (rank + n - 1) % n, comm, at);
  ncclRecv(B(receive, first * sizeof(float)), second, ncclFloat32, (rank + n - 1) % n, comm, at);
  result = ncclGroupEnd();
  wrong = 0;
  for (uint64_t k = 0; !result && k < first + second; k++) wrong += receive[k] != sent_of((rank + n - 1) % n, rank, k);
  failed += reading("send_twice", -1, walk, rank, n, count, result, wrong, comm, on);
  /* the counted all-to-all: rows_of(r, q) rows for each rank, one count entry each, then its whole form */
  for (int whole = 0; whole < 2; whole++) {
    CLEAR();
    size_t k = 0, rows = 3 * (size_t)n;
    for (int q = 0; q < n; q++) { counts[q] = (int64_t)rows_of(rank, q); ones[q] = 1; received[q] = -1; }
    for (size_t x = 0; x < (size_t)n * n; x++) all_ones[x] = 1;
    for (int q = 0; q < n; q++) for (uint64_t i = 0; i < rows_of(rank, q); i++) for (int j = 0; j < ROW; j++) send[k++] = sent_of(rank, q, i * ROW + j);
    void *c = walk ? (void *)&(ncclMeshBuffer){counts, 0} : (void *)counts;
    *arrived = 7;
    result = whole ? ncclMeshAlltoAllCountedPattern(B(send, 0), rows, c, all_ones, B(receive, 0), rows, received, NULL, arrived, ROW, ncclFloat32, comm, at)
                   : ncclMeshAlltoAllCounted(B(send, 0), rows, c, ones, B(receive, 0), rows, received, NULL, ones, arrived, ROW, ncclFloat32, comm, at);
    wrong = !result && *arrived != 1;
    k = 0;
    for (int q = 0; !result && q < n; q++) {
      wrong += received[q] != (int64_t)rows_of(q, rank);
      for (uint64_t i = 0; i < rows_of(q, rank); i++) for (int j = 0; j < ROW; j++) wrong += receive[k++] != sent_of(q, rank, i * ROW + j);
    }
    failed += reading(whole ? "counted_pattern" : "counted", -1, walk, rank, n, count, result, wrong, comm, on);
  }
#undef CLEAR
#undef B
  ncclMemFree(received); ncclMemFree(arrived);
  free(send); free(receive);
  return failed;
}

/* a rank's NCCL calls, each checked exactly (every value an integer a float32 holds): rank r's element k is r + k mod 97 */
static float value_of(int rank, uint64_t k) { return (float)(rank + (int)(k % 97)); }
static int rank_calls(const struct loop_map *map, int rank, uint64_t count, int walk, uint16_t root_port) {
  const int n = map->n;
  if (mesh_loop_bind(map->node[rank].region)) return 1;
  ncclUniqueId id = {{0}};
  snprintf(id.internal, sizeof id.internal, "::1:%u", root_port);
  ncclComm_t comm;
  if (ncclCommInitRank(&comm, n, id, rank)) { fprintf(stderr, "rank %d: %s\n", rank, ncclGetLastError(NULL)); return 1; }
  float *send = calloc((size_t)count * n, sizeof *send), *receive = calloc((size_t)count * n, sizeof *receive);
  if (!send || !receive) return 1;
  ncclMeshStream stream = {.commandBuffer = (void *)1};
  ncclMeshBuffer sent = {send, 0}, got = {receive, 0};
  cudaStream_t on = walk ? (cudaStream_t)&stream : NULL;
  const void *s = walk ? (const void *)&sent : send;
  void *d = walk ? (void *)&got : receive;
  int failed = 0;
  const char *names[] = {"allreduce", "allgather", "reduce_scatter", "broadcast", "reduce", "sendrecv"};
  for (int call = 0; call < 6; call++) for (int root = 0; root < (call == 3 || call == 4 ? n : 1); root++) {
    for (uint64_t k = 0; k < count * (uint64_t)n; k++) { send[k] = value_of(rank, k); receive[k] = -1.0f; }
    ncclResult_t result;
    if (call == 0) result = ncclAllReduce(s, d, count, ncclFloat32, ncclSum, comm, on);
    else if (call == 1) result = ncclAllGather(s, d, count, ncclFloat32, comm, on);
    else if (call == 2) result = ncclReduceScatter(s, d, count, ncclFloat32, ncclSum, comm, on);
    else if (call == 3) result = ncclBroadcast(s, d, count, ncclFloat32, root, comm, on);
    else if (call == 4) result = ncclReduce(s, d, count, ncclFloat32, ncclSum, root, comm, on);
    else {
      ncclGroupStart();
      ncclSend(s, count, ncclFloat32, (rank + 1) % n, comm, on);
      ncclRecv(d, count, ncclFloat32, (rank + n - 1) % n, comm, on);
      result = ncclGroupEnd();
    }
    uint64_t wrong = 0;
    const uint64_t checked = call == 1 ? count * (uint64_t)n : call == 4 && rank != root ? 0 : count;
    for (uint64_t k = 0; !result && k < checked; k++) {
      float want = 0;
      if (call == 0 || call == 4) for (int r = 0; r < n; r++) want += value_of(r, k);
      else if (call == 1) want = value_of((int)(k / count), k % count);
      else if (call == 2) for (int r = 0; r < n; r++) want += value_of(r, (uint64_t)rank * count + k);
      else if (call == 3) want = value_of(root, k);
      else want = value_of((rank + n - 1) % n, k);
      wrong += receive[k] != want;
    }
    failed += result || wrong;
    printf("{\"loop\":\"nccl\",\"path\":\"%s\",\"rank\":%d,\"ranks\":%d,\"call\":\"%s\",\"root\":%d,\"count\":%llu,\"result\":\"%s\","
           "\"wrong\":%llu,\"why\":\"%s\"}\n", walk ? "walk" : "host", rank, n, names[call], root, (unsigned long long)count,
           result ? ncclGetErrorString(result) : "ok", (unsigned long long)wrong, result ? ncclGetLastError(comm) : "");
    fflush(stdout);
  }
  failed += rank_exchanges(comm, rank, n, count, walk, "");
  struct mesh_routing routing;
  const ncclResult_t got_routing = ncclMeshCommGetRouting(comm, &routing, NULL), set_routing = got_routing ? got_routing : ncclMeshCommSetRouting(comm, &routing);
  failed += reading("set_routing", -1, walk, rank, n, routing.routes, set_routing, 0, comm, "");
  if (!set_routing) failed += rank_exchanges(comm, rank, n, count, walk, "set ");
  ncclComm_t child = NULL;
  const ncclResult_t split = ncclCommSplit(comm, 0, n - rank, &child, NULL);
  failed += reading("split", -1, walk, rank, n, 0, split, 0, comm, "");
  int at = -1;
  if (!split) ncclCommUserRank(child, &at);
  if (!split) failed += rank_exchanges(child, at, n, count, walk, "split ");
  if (child) ncclCommDestroy(child);
  ncclCommDestroy(comm);
  free(send); free(receive);
  return failed != 0;
}

static int loop_nccl(const char *directory, uint64_t count, int walk) {
  struct loop_map map;
  if (map_read(directory, &map) || !count) return usage();
  const uint16_t root = free_port();
  const int failed = children(map.n, ^(int i) { return rank_calls(&map, i, count, walk, root); });
  printf("{\"loop\":\"nccl\",\"path\":\"%s\",\"family\":\"%s\",\"ranks\":%d,\"count\":%llu,\"ranks_failed\":%d}\n", walk ? "walk" : "host",
         map.family, map.n, (unsigned long long)count, failed);
  return failed != 0;
}

int main(int argc, char **argv) {
  if (argc < 3) return usage();
  if (!strcmp(argv[1], "up") && argc == 6) return loop_up(argv[2], atoi(argv[3]), strtoull(argv[4], NULL, 0), argv[5]);
  if (!strcmp(argv[1], "down") && argc == 3) return loop_down(argv[2]);
  if (!strcmp(argv[1], "nccl") && argc == 5 && (!strcmp(argv[4], "host") || !strcmp(argv[4], "walk")))
    return loop_nccl(argv[2], strtoull(argv[3], NULL, 0), !strcmp(argv[4], "walk"));
  return usage();
}

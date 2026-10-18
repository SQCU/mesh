/* design/bridge-dataflow.md#the-loopback-fabric: N resident bridges on one host over loopverbs (mesh 800fe6bc^'s loop
   fabric, its driver metal-microbench b9dbc43^ tools/nccl_demo.py `loop`, on the resident bridge), each its own
   configuration (a canonical property list this tool writes: its node, region, loopback service and cables), and the
   readings that exercise what two nodes cannot show: relays and forwarding on 3 to 8 node maps.

     mesh-loop up FAMILY N WINDOW_MIB DIRECTORY
       FAMILY one of line, ring, star, complete, torus (two rows of N/2); N bridges (.build/loop/mesh-flow-loop) of a
       window of WINDOW_MIB each, their configurations, logs and the map in DIRECTORY, every cable two loopverbs devices;
       returns once every bridge's map holds the N nodes and its own cables paired (a JSON line a node)
     mesh-loop relay DIRECTORY BYTES
       every ordered pair of nodes, one after another: a program of one transfer of BYTES from the one to the other, the
       bridges routing it through the nodes between where the two are not cabled (relays, lanes); every word checked
     mesh-loop stripes DIRECTORY BYTES
       every cable, each way: a stripe of BYTES written at the same offset of the far window (mesh-call.h mesh_write) with
       no program at the far end but a reader of its word; every word checked
     mesh-loop nccl DIRECTORY COUNT host|walk
       N ranks, one a node: all-reduce, all-gather, reduce-scatter, broadcast and reduce from each root, and a send and
       receive around the ring, of COUNT float32 each, every value exact; `host` the host path (a NULL stream), `walk`
       the Metal path's walk with its kernels on the CPU (nccl-mesh-host.c), never a GPU
     mesh-loop destroy DIRECTORY COUNT RUNS host|walk
       RUNS times, N ranks: an all-reduce, then a last group (a send and receive around the ring) of COUNT float32 each,
       every value exact, and every rank's ncclCommDestroy right after its last group (design/bridge-and-ipc.md#control:
       each bridge posts what its rank released and releases its flows once those SENDs completed, telling no peer, so
       a landing is never cancelled); then every bridge holding no flow, and the bridges' logs read: the flows ended in
       order, the relays released by both their ends, the flows failed, and each exclusive flow's flush (every work
       request it held drained)
     mesh-loop kill DIRECTORY COUNT host|walk [bridge]
       N ranks: an all-reduce, then the last rank waits while the others begin a second (of COUNT float32 each) and is
       killed (SIGKILL): each other rank's call fails as a value, and the time from the kill to its return is read; with
       `bridge`, every rank begins the second and the last rank's node's bridge is stopped (SIGTERM, as launchd stops
       one) in its place: every rank's call fails as a value, the last rank's by its client's watch of its bridge's exit
       (mesh-dataflow.c mesh_watch: its waits' words cancelled), the others' by the stopped bridge's FAREWELL; run it
       last, the stopped bridge stays stopped
     mesh-loop down DIRECTORY
       every bridge stopped (SIGTERM, as launchd stops one) and its exit awaited, the fabric unlinked

   Every wait is on an event (a map's serial, a process's exit, a landing word, a pipe), none bounded by a clock.  A
   JSON line a reading, with its status; the exit status is nonzero where any failed. */
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
#include <sys/event.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#define LOOP_NODES 8
#define LOOP_CABLES 32
#define LOOP_FABRIC "/meshloop-fabric"
#define LOOP_MAGIC UINT64_C(0x6c6f6f702d6d6573)
extern char **environ;

struct loop_node { uint32_t node; char region[32]; uint16_t service; pid_t pid; };
struct loop_cable { int a, b; uint16_t listen[2]; };
struct loop_map { char family[16]; int n, cables; uint64_t window; struct loop_node node[LOOP_NODES]; struct loop_cable cable[LOOP_CABLES]; };

static uint64_t clock_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int usage(void) {
  fprintf(stderr, "usage: mesh-loop up FAMILY N WINDOW_MIB DIRECTORY | relay DIRECTORY BYTES | stripes DIRECTORY BYTES | "
                  "nccl DIRECTORY COUNT host|walk | destroy DIRECTORY COUNT RUNS host|walk | kill DIRECTORY COUNT host|walk [bridge] | down DIRECTORY\n");
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
  } else if (!strcmp(family, "star")) for (int i = 1; i < n; i++) CABLE(0, i);
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
  fprintf(f, "%s %d %d %llu\n", map->family, map->n, map->cables, (unsigned long long)map->window);
  for (int i = 0; i < map->n; i++)
    fprintf(f, "%u %s %u %d\n", map->node[i].node, map->node[i].region, map->node[i].service, (int)map->node[i].pid);
  for (int c = 0; c < map->cables; c++)
    fprintf(f, "%d %d %u %u\n", map->cable[c].a, map->cable[c].b, map->cable[c].listen[0], map->cable[c].listen[1]);
  return fclose(f) ? errno : 0;
}
static int map_read(const char *directory, struct loop_map *map) {
  char path[PATH_MAX];
  if (map_path(directory, path, sizeof path)) return ENAMETOOLONG;
  FILE *f = fopen(path, "r");
  if (!f) return errno;
  unsigned long long window = 0;
  int ok = fscanf(f, "%15s %d %d %llu", map->family, &map->n, &map->cables, &window) == 4 && map->n > 0 && map->n <= LOOP_NODES &&
           map->cables >= 0 && map->cables <= LOOP_CABLES;
  map->window = window;
  for (int i = 0; ok && i < map->n; i++) {
    unsigned node, service; int pid;
    ok = fscanf(f, "%u %31s %u %d", &node, map->node[i].region, &service, &pid) == 4;
    map->node[i].node = node; map->node[i].service = (uint16_t)service; map->node[i].pid = (pid_t)pid;
  }
  for (int c = 0; ok && c < map->cables; c++) {
    unsigned x, y;
    ok = fscanf(f, "%d %d %u %u", &map->cable[c].a, &map->cable[c].b, &x, &y) == 4;
    map->cable[c].listen[0] = (uint16_t)x; map->cable[c].listen[1] = (uint16_t)y;
  }
  fclose(f);
  return ok ? 0 : EINVAL;
}

/* node i's canonical configuration (bridge_configuration.py's form, and its `loop`: mesh-flow.c loop_read), its keepalive
   and HELLO intervals the deployed bridges' (etc/bridge: keepalive_us 5, hello_us 1000000; a loop port's lower end dials
   a listener that may not listen yet, and dials again at the next HELLO) */
static int plist_write(const char *directory, const struct loop_map *map, int i) {
  char path[PATH_MAX];
  if (snprintf(path, sizeof path, "%s/node%d.plist", directory, i) >= (int)sizeof path) return ENAMETOOLONG;
  FILE *f = fopen(path, "w");
  if (!f) return errno;
  fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
             "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict>\n"
             "<key>name</key><string>loop%d</string>\n<key>purpose</key><string>node %d of a %s of %d on the loopback fabric "
             "(mesh-loop)</string>\n<key>window</key><integer>%llu</integer>\n<key>exec</key><false/>\n<key>keepalive</key><integer>5000</integer>\n"
             "<key>hello</key><integer>1000000000</integer>\n"
             "<key>loop</key><dict>\n<key>region</key><string>%s</string>\n<key>fabric</key><string>%s</string>\n"
             "<key>node</key><integer>%u</integer>\n<key>service</key><integer>%u</integer>\n<key>ports</key><array>\n",
          i, i, map->family, map->n, (unsigned long long)map->window, map->node[i].region, LOOP_FABRIC, map->node[i].node,
          map->node[i].service);
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
/* the hops between a and b on the map (breadth first), 0 where none */
static int hops(const struct loop_map *map, int a, int b) {
  int distance[LOOP_NODES], queue[LOOP_NODES], head = 0, tail = 0;
  for (int i = 0; i < map->n; i++) distance[i] = -1;
  distance[a] = 0; queue[tail++] = a;
  while (head < tail) {
    const int x = queue[head++];
    for (int c = 0; c < map->cables; c++) {
      const int y = map->cable[c].a == x ? map->cable[c].b : map->cable[c].b == x ? map->cable[c].a : -1;
      if (y >= 0 && distance[y] < 0) { distance[y] = distance[x] + 1; queue[tail++] = y; }
    }
  }
  return distance[b] > 0 ? distance[b] : 0;
}

/* every bridge's map holding the n nodes and its own cables paired, each bridge's awaited on its map's serial (its
   ports' pairings move its record, so its map); 0, or the bridge's error */
static int converged(const struct loop_map *map) {
  for (int i = 0; i < map->n; i++) {
    if (mesh_loop_bind(map->node[i].region, map->node[i].service)) return EINVAL;
    for (;;) {
      uint64_t serial = mesh_map_serial();
      struct mesh_node_health health[LOOP_NODES + 1];
      struct mesh_link_view links[MESH_PORTS];
      uint32_t node = 0;
      int paired = 0;
      const int records = mesh_nodes_health(health, LOOP_NODES + 1), ports = mesh_observe(links, MESH_PORTS, &node);
      for (int p = 0; p < ports && p < MESH_PORTS; p++) paired += links[p].phase == MESH_PAIRED;
      if (records == map->n && paired == degree(map, i)) break;
      const int error = serial ? mesh_map_wait(&serial) : mesh_bridge_wait();
      if (error) return error;
    }
  }
  return 0;
}

static int loop_up(const char *family, int n, uint64_t window_mib, const char *directory) {
  struct loop_map map = {.n = n, .window = window_mib << 20};
  snprintf(map.family, sizeof map.family, "%s", family);
  if (n < 2 || n > LOOP_NODES || !window_mib || family_cables(family, n, &map)) return usage();
  char binary[PATH_MAX];
  if (bridge_path(binary, sizeof binary)) return 1;
  mkdir(directory, 0755);
  shm_unlink(LOOP_FABRIC);
  for (int i = 0; i < n; i++) {
    map.node[i] = (struct loop_node){.node = (uint32_t)(i + 1), .service = free_port()};
    snprintf(map.node[i].region, sizeof map.node[i].region, "/meshloop%d", i);
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
  const int status = converged(&map);
  for (int i = 0; i < n; i++)
    printf("{\"loop\":\"up\",\"family\":\"%s\",\"nodes\":%d,\"index\":%d,\"node\":%u,\"region\":\"%s\",\"service\":%u,\"pid\":%d,"
           "\"cables\":%d,\"converged\":%s}\n", family, n, i, map.node[i].node, map.node[i].region, map.node[i].service,
           (int)map.node[i].pid, degree(&map, i), status ? "false" : "true");
  return status ? 1 : 0;
}

/* every bridge stopped (SIGTERM) and each exit awaited (EVFILT_PROC: a bridge another invocation started) */
static int loop_down(const char *directory) {
  struct loop_map map;
  if (map_read(directory, &map)) return usage();
  const int events = kqueue();
  int left = 0;
  for (int i = 0; i < map.n && events >= 0; i++) {
    struct kevent64_s watch;
    EV_SET64(&watch, (uint64_t)map.node[i].pid, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0, 0, 0, 0);
    if (map.node[i].pid > 0 && !kevent64(events, &watch, 1, NULL, 0, KEVENT_FLAG_IMMEDIATE, NULL)) left++;
    if (map.node[i].pid > 0) kill(map.node[i].pid, SIGTERM);
  }
  for (struct kevent64_s exited; left > 0 && kevent64(events, NULL, 0, &exited, 1, 0, NULL) == 1;) left--;
  if (events >= 0) close(events);
  shm_unlink(LOOP_FABRIC);
  printf("{\"loop\":\"down\",\"nodes\":%d,\"stopped\":%s}\n", map.n, left || events < 0 ? "false" : "true");
  return left || events < 0;
}

/* the pattern word w of block p of a reading (mesh-stream.c's, the loop's magic) */
static uint64_t pattern(uint64_t p, uint64_t w) { return (p << 32) ^ w ^ LOOP_MAGIC; }

/* one end of a relayed program: one transfer of `bytes` toward or from node `peer`, the sender detaching once it
   published (its bridge posts it and releases its flows in order), the receiver following its landing words and
   checking every word (mesh-stream.c's flow path); 0, or what failed */
static int relay_end(const struct loop_map *map, int self, int peer, int receive, uint64_t bytes, int ready) {
  struct mesh_ctx context = {0};
  char name[64];
  snprintf(name, sizeof name, "loop-relay-%d-%d", receive ? peer : self, receive ? self : peer);
  int status = mesh_loop_bind(map->node[self].region, map->node[self].service);
  if (!status) status = mesh_attach(&context, name);
  if (status) { if (ready >= 0) close(ready); return status; }
  struct hdr *m = context.M;
  const uint64_t block = (uint64_t)m->block * m->pgsz, blocks = bytes / block, words = block / sizeof(uint64_t);
  struct mesh_section section;
  if (!blocks || bytes % block) status = EINVAL;
  if (!status) status = mesh_section_create(&context, bytes, 1, &section);
  if (!status) status = mesh_transfer_bind(&context, map->node[peer].node, 0, receive ? MESH_RECEIVE : MESH_SEND, 1, section, section.pages, 0, 1, 1);
  if (!status) status = mesh_transfers_prepare(&context, 1, 1, 1);
  if (!status && receive) status = mesh_host_inputs(&context);
  unsigned char *data = status ? NULL : mesh_section_address(&context, section, 0);
  for (uint64_t p = 0; data && p < blocks; p++)
    for (uint64_t w = 0; w < words; w++) ((uint64_t *)(data + p * block))[w] = receive ? 0 : pattern(p, w);
  if (ready >= 0) { if (write(ready, "r", 1) != 1) status = status ? status : EPIPE; close(ready); }
  if (!status) status = mesh_transfers_start(&context);
  if (!status && !receive) {
    mesh_host_publish(&context, section, 0);
    mesh_transfers_wake(&context);
  }
  uint64_t completed = 0, mismatches = 0;
  if (!status && receive) {
    while (completed < blocks) {
      const struct mesh_delivery *delivery = mesh_delivery_at(m, section.first + (uint32_t)completed);
      const uint64_t word = atomic_load_explicit((_Atomic uint64_t *)((char *)m + delivery->device_input), memory_order_acquire);
      if (word == UINT64_MAX) { status = ECANCELED; break; }
      if (word) completed++;
      else sched_yield();
    }
    for (uint64_t p = 0; p < completed; p++)
      for (uint64_t w = 0; w < words; w++) mismatches += ((const uint64_t *)(data + p * block))[w] != pattern(p, w);
    if (!status && mismatches) status = EIO;
  }
  mesh_detach(&context);
  return status;
}

/* run `count` children, child i running `run(i)`, then `forked` here (the parent's ends of their pipes closed, so a
   child reads end of file where another exited without writing), and wait for all; how many exited nonzero */
static int children(int count, int (^run)(int), void (^forked)(void)) {
  pid_t pid[2 * LOOP_NODES];
  for (int i = 0; i < count; i++) {
    pid[i] = fork();
    if (!pid[i]) _exit(run(i) & 0xff);
  }
  if (forked) forked();
  int failed = 0;
  for (int i = 0; i < count; i++) {
    int status = 0;
    if (pid[i] < 0 || waitpid(pid[i], &status, 0) != pid[i] || !WIFEXITED(status) || WEXITSTATUS(status)) failed++;
  }
  return failed;
}

static int loop_relay(const char *directory, uint64_t bytes) {
  struct loop_map map;
  if (map_read(directory, &map) || !bytes) return usage();
  int failed = 0;
  for (int a = 0; a < map.n; a++) for (int b = 0; b < map.n; b++) {
    if (a == b) continue;
    int ready[2];
    if (pipe(ready)) return 1;
    const int heard = ready[0], told = ready[1];
    const uint64_t begin = clock_ns();
    const int bad = children(2, ^(int i) {
      if (i == 0) { close(heard); return relay_end(&map, b, a, 1, bytes, told); }
      close(told);
      char r;
      if (read(heard, &r, 1) != 1) return EPIPE;
      close(heard);
      return relay_end(&map, a, b, 0, bytes, -1);
    }, ^{ close(heard); close(told); });
    failed += bad != 0;
    printf("{\"loop\":\"relay\",\"family\":\"%s\",\"nodes\":%d,\"from\":%d,\"to\":%d,\"hops\":%d,\"bytes\":%llu,\"seconds\":%.6f,"
           "\"status\":\"%s\"}\n", map.family, map.n, a, b, hops(&map, a, b), (unsigned long long)bytes, (double)(clock_ns() - begin) / 1e9,
           bad ? "failed" : "ok");
    fflush(stdout);
  }
  return failed != 0;
}

/* one end of a stripe across cable a-b: the writer at `self` toward `peer`, or the reader at `self`; both lease the same
   exact pages near their windows' tops (mesh_section_at), the stripe's bytes then a block holding the word */
static int stripe_end(const struct loop_map *map, int self, int peer, int writer, uint64_t bytes, int ready) {
  struct mesh_ctx context = {0};
  char name[64];
  snprintf(name, sizeof name, "loop-stripe-%d-%d", writer ? self : peer, writer ? peer : self);
  int status = mesh_loop_bind(map->node[self].region, map->node[self].service);
  if (!status) status = mesh_attach(&context, name);
  if (status) { if (ready >= 0) close(ready); return status; }
  struct hdr *m = context.M;
  const uint64_t block = (uint64_t)m->block * m->pgsz, blocks = bytes / block, words = block / sizeof(uint64_t);
  const uint32_t pages = (uint32_t)((bytes + block) / m->pgsz), arena = mesh_arena_pages(m);
  const uint32_t page = arena > pages + m->block ? (arena - pages - m->block) / m->block * m->block : 0;
  struct mesh_section section;
  if (!blocks || bytes % block || !page) status = EINVAL;
  if (!status) status = mesh_section_at(&context, page, bytes + block, &section);
  unsigned char *data = status ? NULL : mesh_section_address(&context, section, 0);
  _Atomic uint64_t *word = data ? (_Atomic uint64_t *)(data + bytes) : NULL;
  if (word) atomic_store_explicit(word, 0, memory_order_release);
  for (uint64_t p = 0; data && p < blocks; p++)
    for (uint64_t w = 0; w < words; w++) ((uint64_t *)(data + p * block))[w] = writer ? pattern(p, w) : 0;
  if (ready >= 0) { if (write(ready, "r", 1) != 1) status = status ? status : EPIPE; close(ready); }
  const uint64_t offset = (uint64_t)page * m->pgsz, at = (uint64_t)((char *)word - (char *)m);
  if (!status && writer) {
    struct mesh_requests *ring = NULL;
    status = mesh_requests_create(&context, 16, &ring);
    if (!status) status = mesh_write(ring, map->node[peer].node, offset, bytes, at, LOOP_MAGIC);
  }
  while (!status && atomic_load_explicit(word, memory_order_acquire) != LOOP_MAGIC) {
    if (atomic_load_explicit(word, memory_order_acquire) == UINT64_MAX) { status = ECANCELED; break; }
    sched_yield();
  }
  uint64_t mismatches = 0;
  for (uint64_t p = 0; !status && !writer && p < blocks; p++)
    for (uint64_t w = 0; w < words; w++) mismatches += ((const uint64_t *)(data + p * block))[w] != pattern(p, w);
  if (mismatches) status = EIO;
  mesh_detach(&context);
  return status;
}

static int loop_stripes(const char *directory, uint64_t bytes) {
  struct loop_map map;
  if (map_read(directory, &map) || !bytes) return usage();
  int failed = 0;
  for (int c = 0; c < map.cables; c++) for (int way = 0; way < 2; way++) {
    const int a = way ? map.cable[c].b : map.cable[c].a, b = way ? map.cable[c].a : map.cable[c].b;
    int ready[2];
    if (pipe(ready)) return 1;
    const int heard = ready[0], told = ready[1];
    const uint64_t begin = clock_ns();
    const int bad = children(2, ^(int i) {
      if (i == 0) { close(heard); return stripe_end(&map, b, a, 0, bytes, told); }
      close(told);
      char r;
      if (read(heard, &r, 1) != 1) return EPIPE;
      close(heard);
      return stripe_end(&map, a, b, 1, bytes, -1);
    }, ^{ close(heard); close(told); });
    failed += bad != 0;
    printf("{\"loop\":\"stripes\",\"family\":\"%s\",\"from\":%d,\"to\":%d,\"bytes\":%llu,\"seconds\":%.6f,\"status\":\"%s\"}\n", map.family, a, b,
           (unsigned long long)bytes, (double)(clock_ns() - begin) / 1e9, bad ? "failed" : "ok");
    fflush(stdout);
  }
  return failed != 0;
}

/* a rank's NCCL calls, each checked exactly (every value an integer a float32 holds): rank r's element k is r + k mod 97 */
static float value_of(int rank, uint64_t k) { return (float)(rank + (int)(k % 97)); }
static int rank_calls(const struct loop_map *map, int rank, uint64_t count, int walk, ncclUniqueId id) {
  const int n = map->n;
  if (mesh_loop_bind(map->node[rank].region, map->node[rank].service)) return 1;
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
           "\"wrong\":%llu}\n", walk ? "walk" : "host", rank, n, names[call], root, (unsigned long long)count,
           result ? ncclGetErrorString(result) : "ok", (unsigned long long)wrong);
    fflush(stdout);
  }
  ncclCommDestroy(comm);
  free(send); free(receive);
  return failed != 0;
}

/* the ranks' unique id, its root's listener opened here before the ranks are forked (nccl-mesh.c ncclGetUniqueId), so
   every rank's connect meets it listening */
static int unique_id(ncclUniqueId *id) {
  if (ncclGetUniqueId(id) == ncclSuccess) return 0;
  fprintf(stderr, "mesh-loop: the unique id: %s\n", ncclGetLastError(NULL));
  return 1;
}

static int loop_nccl(const char *directory, uint64_t count, int walk) {
  struct loop_map map;
  ncclUniqueId root;
  if (map_read(directory, &map) || !count) return usage();
  if (unique_id(&root)) return 1;
  const int failed = children(map.n, ^(int i) { return rank_calls(&map, i, count, walk, root); }, NULL);
  printf("{\"loop\":\"nccl\",\"path\":\"%s\",\"family\":\"%s\",\"ranks\":%d,\"count\":%llu,\"ranks_failed\":%d}\n", walk ? "walk" : "host",
         map.family, map.n, (unsigned long long)count, failed);
  return failed != 0;
}

/* a rank's communicator over the map's nodes, met at the unique id `id` */
static ncclResult_t rank_comm(const struct loop_map *map, int rank, ncclUniqueId id, ncclComm_t *comm) {
  if (mesh_loop_bind(map->node[rank].region, map->node[rank].service)) return ncclInvalidArgument;
  return ncclCommInitRank(comm, map->n, id, rank);
}
/* an all-reduce of `count` (send) or a send and receive around the ring (last), every value exact; its result where
   it failed, else the values wrong in *wrong */
static ncclResult_t rank_group(ncclComm_t comm, int rank, int n, uint64_t count, int walk, int ring, uint64_t *wrong) {
  float *send = calloc((size_t)count, sizeof *send), *receive = calloc((size_t)count, sizeof *receive);
  if (!send || !receive) { free(send); free(receive); return ncclSystemError; }
  for (uint64_t k = 0; k < count; k++) { send[k] = value_of(rank, k); receive[k] = -1.0f; }
  ncclMeshStream stream = {.commandBuffer = (void *)1};
  ncclMeshBuffer sent = {send, 0}, got = {receive, 0};
  cudaStream_t on = walk ? (cudaStream_t)&stream : NULL;
  const void *s = walk ? (const void *)&sent : send;
  void *d = walk ? (void *)&got : receive;
  ncclResult_t result;
  if (ring) {
    ncclGroupStart();
    ncclSend(s, count, ncclFloat32, (rank + 1) % n, comm, on);
    ncclRecv(d, count, ncclFloat32, (rank + n - 1) % n, comm, on);
    result = ncclGroupEnd();
  } else result = ncclAllReduce(s, d, count, ncclFloat32, ncclSum, comm, on);
  *wrong = 0;
  for (uint64_t k = 0; !result && k < count; k++) {
    float want = 0;
    if (ring) want = value_of((rank + n - 1) % n, k);
    else for (int r = 0; r < n; r++) want += value_of(r, k);
    *wrong += receive[k] != want;
  }
  /* a walk's failure is its session's, recorded as its steps ran (the host executor runs them as they are encoded):
     read where its values are wrong (a peer that closed in order since is no failure of this group) */
  ncclResult_t async = ncclSuccess;
  if (!result && walk && *wrong && !ncclCommGetAsyncError(comm, &async)) result = async;
  free(send); free(receive);
  return result;
}

/* the bytes each node's log holds now (where the reading's lines begin) */
static void logs_mark(const char *directory, int n, off_t *at) {
  for (int i = 0; i < n; i++) {
    char path[PATH_MAX];
    struct stat info;
    snprintf(path, sizeof path, "%s/node%d.log", directory, i);
    at[i] = stat(path, &info) ? 0 : info.st_size;
  }
}
/* the bridges' lines past `at` (mesh-flow.c flow_released, relay_released, flow_fail, flow_release): flows ended in
   order, relays released by both their ends, flows failed, exclusive flows flushed, and those of the flushes that
   drained fewer completions than work requests held */
struct log_counts { uint64_t ended, relays, failed, flushed, short_of; };
static void logs_read(const char *directory, int n, const off_t *at, struct log_counts *counts) {
  *counts = (struct log_counts){0};
  for (int i = 0; i < n; i++) {
    char path[PATH_MAX], line[2048];
    snprintf(path, sizeof path, "%s/node%d.log", directory, i);
    FILE *f = fopen(path, "r");
    if (!f) continue;
    fseeko(f, at[i], SEEK_SET);
    while (fgets(line, sizeof line, f)) {
      const char *c = strstr(line, " flushed ");
      unsigned long long drained, held;
      if (strstr(line, ") ended in order")) counts->ended++;
      else if (strstr(line, ") released by both its ends")) counts->relays++;
      else if (strstr(line, ") failed at node ")) counts->failed++;
      else if (c && sscanf(c, " flushed %llu of %llu work requests", &drained, &held) == 2) { counts->flushed++; counts->short_of += drained != held; }
    }
    fclose(f);
  }
}
/* every bridge holding no flow and no awaiting program (mesh_busy), each looked at until it holds none; 0, or EINVAL */
static int bridges_idle(const struct loop_map *map) {
  for (int i = 0; i < map->n;) {
    uint64_t busy = 1, since = 0;
    if (mesh_loop_bind(map->node[i].region, map->node[i].service) || mesh_busy(&busy, &since) < 0) return EINVAL;
    if (!busy) i++;
    else sched_yield();
  }
  return 0;
}

static int loop_destroy(const char *directory, uint64_t count, int runs, int walk) {
  struct loop_map map;
  if (map_read(directory, &map) || !count || runs < 1) return usage();
  off_t marks[LOOP_NODES];
  logs_mark(directory, map.n, marks);
  int failed_runs = 0;
  uint64_t cancelled = 0;
  for (int run = 0; run < runs; run++) {
    ncclUniqueId root;
    if (unique_id(&root)) return 1;
    int held[LOOP_NODES][2], (*pipes)[2] = held;
    for (int i = 0; i < map.n; i++) if (pipe(pipes[i])) return 1;
    const uint64_t begin = clock_ns();
    const int bad = children(map.n, ^(int i) {
      for (int j = 0; j < map.n; j++) { close(pipes[j][0]); if (j != i) close(pipes[j][1]); }
      ncclComm_t comm = NULL;
      ncclResult_t result = rank_comm(&map, i, root, &comm);
      uint64_t wrong = 0;
      if (!result) result = rank_group(comm, i, map.n, count, walk, 0, &wrong);
      if (!result && !wrong) result = rank_group(comm, i, map.n, count, walk, 1, &wrong);
      char said[256];
      snprintf(said, sizeof said, "%s", result ? ncclGetLastError(NULL) : "");
      ncclCommDestroy(comm);
      const char mark = result ? (strstr(said, "cancelled") ? 'c' : 'f') : wrong ? 'w' : 'o';
      if (write(pipes[i][1], &mark, 1) != 1) return 1;
      if (result) fprintf(stderr, "destroy run %d rank %d: %s\n", run, i, said);
      return result || wrong;
    }, ^{ for (int j = 0; j < map.n; j++) close(pipes[j][1]); });
    int cancels = 0, others = 0;
    for (int i = 0; i < map.n; i++) {
      char mark = 'f';
      if (read(pipes[i][0], &mark, 1) != 1) mark = 'f';
      close(pipes[i][0]);
      cancels += mark == 'c';
      others += mark == 'f' || mark == 'w';
    }
    cancelled += (uint64_t)cancels;
    failed_runs += bad != 0;
    printf("{\"loop\":\"destroy\",\"path\":\"%s\",\"family\":\"%s\",\"ranks\":%d,\"count\":%llu,\"run\":%d,\"seconds\":%.6f,"
           "\"ranks_failed\":%d,\"cancelled\":%d,\"other\":%d}\n", walk ? "walk" : "host", map.family, map.n, (unsigned long long)count, run,
           (double)(clock_ns() - begin) / 1e9, bad, cancels, others);
    fflush(stdout);
  }
  const int idle = bridges_idle(&map);
  struct log_counts counts;
  logs_read(directory, map.n, marks, &counts);
  printf("{\"loop\":\"destroy\",\"path\":\"%s\",\"family\":\"%s\",\"runs\":%d,\"runs_failed\":%d,\"cancelled\":%llu,\"bridges_idle\":%s,"
         "\"flows_ended\":%llu,\"relays_released\":%llu,\"flows_failed\":%llu,\"flushed\":%llu,\"flushed_short\":%llu}\n",
         walk ? "walk" : "host", map.family, runs, failed_runs, (unsigned long long)cancelled, idle ? "false" : "true",
         (unsigned long long)counts.ended, (unsigned long long)counts.relays, (unsigned long long)counts.failed,
         (unsigned long long)counts.flushed, (unsigned long long)counts.short_of);
  return failed_runs || idle || counts.short_of || counts.failed;
}

/* the kill reading's shared page: when the victim was killed, and each rank's return (its time and result) */
struct kill_page { _Atomic uint64_t killed, returned[LOOP_NODES], wrong[LOOP_NODES]; _Atomic int result[LOOP_NODES]; char said[LOOP_NODES][160]; };
static int loop_kill(const char *directory, uint64_t count, int walk, int bridge) {
  struct loop_map map;
  ncclUniqueId root;
  if (map_read(directory, &map) || !count || map.n < 2) return usage();
  struct kill_page *page = mmap(NULL, sizeof *page, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
  int told[2];
  if (page == MAP_FAILED || pipe(told) || unique_id(&root)) return 1;
  memset(page, 0, sizeof *page);
  const int victim = map.n - 1;
  pid_t pid[LOOP_NODES];
  for (int i = 0; i < map.n; i++) {
    pid[i] = fork();
    if (pid[i]) continue;
    close(told[0]);
    ncclComm_t comm = NULL;
    ncclResult_t result = rank_comm(&map, i, root, &comm);
    uint64_t wrong = 0;
    if (!result) result = rank_group(comm, i, map.n, count, walk, 0, &wrong);
    const char mark = result || wrong ? 'f' : 'r';
    if (write(told[1], &mark, 1) != 1 || mark == 'f') _exit(2);
    close(told[1]);
    if (i == victim && !bridge) for (;;) pause();
    result = rank_group(comm, i, map.n, count, walk, 0, &wrong);
    atomic_store(&page->returned[i], clock_ns());
    atomic_store(&page->result[i], (int)result);
    atomic_store(&page->wrong[i], wrong);
    snprintf(page->said[i], sizeof page->said[i], "%s", result ? ncclGetLastError(NULL) : "");
    ncclCommDestroy(comm);
    _exit(0);
  }
  close(told[1]);
  /* every rank through its first all-reduce (a byte each), the others then entering the second, which awaits the
     victim's contribution */
  int ready = 0, failed = 0;
  for (char mark; ready + failed < map.n && read(told[0], &mark, 1) == 1;) mark == 'r' ? ready++ : failed++;
  close(told[0]);
  atomic_store(&page->killed, clock_ns());
  kill(bridge ? map.node[victim].pid : pid[victim], bridge ? SIGTERM : SIGKILL);
  for (int i = 0; i < map.n; i++) {
    int status = 0;
    const int counted = i != victim || bridge;
    if (waitpid(pid[i], &status, 0) != pid[i] || (counted && (!WIFEXITED(status) || WEXITSTATUS(status)))) failed += counted;
  }
  for (int i = 0; i < map.n - !bridge; i++) {
    const uint64_t at = atomic_load(&page->returned[i]), killed = atomic_load(&page->killed);
    const int result = atomic_load(&page->result[i]);
    failed += !result;
    printf("{\"loop\":\"kill\",\"path\":\"%s\",\"family\":\"%s\",\"ranks\":%d,\"victim\":%d,\"stopped\":\"%s\",\"rank\":%d,"
           "\"failed_as_value\":%s,\"wrong\":%llu,\"latency_us\":%.1f,\"result\":\"%s\"}\n", walk ? "walk" : "host", map.family, map.n, victim,
           bridge ? "bridge" : "client", i,
           result ? "true" : "false", (unsigned long long)atomic_load(&page->wrong[i]), at > killed ? (double)(at - killed) / 1e3 : -1.0, page->said[i]);
  }
  fflush(stdout);
  munmap(page, sizeof *page);
  return failed != 0;
}

int main(int argc, char **argv) {
  if (argc < 3) return usage();
  if (!strcmp(argv[1], "up") && argc == 6) return loop_up(argv[2], atoi(argv[3]), strtoull(argv[4], NULL, 0), argv[5]);
  if (!strcmp(argv[1], "down") && argc == 3) return loop_down(argv[2]);
  if (!strcmp(argv[1], "relay") && argc == 4) return loop_relay(argv[2], strtoull(argv[3], NULL, 0));
  if (!strcmp(argv[1], "stripes") && argc == 4) return loop_stripes(argv[2], strtoull(argv[3], NULL, 0));
  if (!strcmp(argv[1], "nccl") && argc == 5 && (!strcmp(argv[4], "host") || !strcmp(argv[4], "walk")))
    return loop_nccl(argv[2], strtoull(argv[3], NULL, 0), !strcmp(argv[4], "walk"));
  if (!strcmp(argv[1], "destroy") && argc == 6 && (!strcmp(argv[5], "host") || !strcmp(argv[5], "walk")))
    return loop_destroy(argv[2], strtoull(argv[3], NULL, 0), atoi(argv[4]), !strcmp(argv[5], "walk"));
  if (!strcmp(argv[1], "kill") && (argc == 5 || (argc == 6 && !strcmp(argv[5], "bridge"))) && (!strcmp(argv[4], "host") || !strcmp(argv[4], "walk")))
    return loop_kill(argv[2], strtoull(argv[3], NULL, 0), !strcmp(argv[4], "walk"), argc == 6);
  return usage();
}

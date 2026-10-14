/* libnccl-mesh (nccl.h): NCCL's API on the bridges' prepared transfers.  Each communicator holds a session on its
   node's bridge (its own attach under its name, a ring of slots each way to every other member, which the bridges
   carry along the map's route where no link joins the two), opened at its first group that crosses a link, on the map
   current then, and opened again after a failure; a group's calls are planned by the one collective planner and
   streamed through the rings, the receives combined on the host or the GPU.  The calls' mapping is metal-microbench
   tools/nccl_demo.py's as of dec7676. */
#include "nccl.h"
#include "mesh-collective.h"
#include "mesh.h"
#include "mesh-metal.h"
#include "nccl-mesh-metal.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/in.h>

enum { WHAT_SEND = 16, WHAT_RECV };
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN };
#define DEADLINE_S 120.0
#define LINGER_S 1.0
#define SILENT_S (MESH_SILENT_NS * 1e-9)
#define OPS 64

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
static _Thread_local char last[512];

static ncclResult_t fail(ncclResult_t result, const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(last, sizeof last, format, arguments);
  va_end(arguments);
  return result;
}

/* -- the element types: host arithmetic as numpy and ml_dtypes do it (the small floating types in float32,
   rounded to nearest even) -------------------------------------------------------------------------- */

static float bf16_to(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t bf16_from(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

/* float8_e4m3fn (no infinity; NaN S.1111.111) and float8_e5m2 (IEEE-like) */
static float f8_to(uint8_t v, int e5m2) {
  const int mbits = e5m2 ? 2 : 3, bias = e5m2 ? 15 : 7;
  const int expf = (v >> mbits) & (e5m2 ? 31 : 15), mant = v & ((1 << mbits) - 1);
  double r;
  if (e5m2 && expf == 31) r = mant ? NAN : INFINITY;
  else if (!e5m2 && expf == 15 && mant == 7) r = NAN;
  else if (!expf) r = ldexp((double)mant, 1 - bias - mbits);
  else r = ldexp(1.0 + (double)mant / (1 << mbits), expf - bias);
  return (float)((v & 0x80) ? -r : r);
}
static uint8_t f8_from(float x, int e5m2) {
  const int mbits = e5m2 ? 2 : 3, bias = e5m2 ? 15 : 7;
  const uint8_t s = signbit(x) ? 0x80 : 0;
  if (isnan(x)) return s | 0x7f;
  if (isinf(x)) return e5m2 ? (s | 0x7c) : (s | 0x7f);
  const double a = fabs((double)x);
  if (a == 0) return s;
  int binade;
  frexp(a, &binade);
  int e = binade - 1;
  if (e < 1 - bias) e = 1 - bias;
  double n = rint(a / ldexp(1.0, e - mbits));
  if (n >= ldexp(1.0, mbits + 1)) { n /= 2; e += 1; }
  int expf, mant;
  if (n < ldexp(1.0, mbits)) { expf = 0; mant = (int)n; }
  else { expf = e + bias; mant = (int)n - (1 << mbits); }
  if (e5m2 ? expf >= 31 : (expf > 15 || (expf == 15 && mant == 7))) return e5m2 ? (s | 0x7c) : (s | 0x7f);
  return s | (uint8_t)(expf << mbits) | (uint8_t)mant;
}

static int floating(ncclDataType_t t) { return t >= ncclFloat16; }

static double load(ncclDataType_t t, const unsigned char *p) {
  switch (t) {
  case ncclFloat16: { _Float16 h; memcpy(&h, p, 2); return (double)(float)h; }
  case ncclFloat32: { float f; memcpy(&f, p, 4); return f; }
  case ncclFloat64: { double d; memcpy(&d, p, 8); return d; }
  case ncclBfloat16: { uint16_t h; memcpy(&h, p, 2); return bf16_to(h); }
  case ncclFloat8e4m3: return f8_to(*p, 0);
  case ncclFloat8e5m2: return f8_to(*p, 1);
  default: return 0;
  }
}

/* x is a float32 result for the 16- and 8-bit types (computed in float32, as numpy and ml_dtypes do) */
static void store(ncclDataType_t t, unsigned char *p, double x) {
  switch (t) {
  case ncclFloat16: { _Float16 h = (_Float16)(float)x; memcpy(p, &h, 2); break; }
  case ncclFloat32: { float f = (float)x; memcpy(p, &f, 4); break; }
  case ncclFloat64: memcpy(p, &x, 8); break;
  case ncclBfloat16: { uint16_t h = bf16_from((float)x); memcpy(p, &h, 2); break; }
  case ncclFloat8e4m3: *p = f8_from((float)x, 0); break;
  case ncclFloat8e5m2: *p = f8_from((float)x, 1); break;
  default: break;
  }
}

static double arithmetic(ncclDataType_t t, double a, double b, int combine) {
  if (t == ncclFloat64) return combine == COMBINE_PROD ? a * b : a + b;
  return combine == COMBINE_PROD ? (double)((float)a * (float)b) : (double)((float)a + (float)b);
}

#define INTEGER_LOOP(S, U) do { \
    S *d = (S *)dst; const S *x = (const S *)src; \
    for (size_t i = 0; i < n; i++) switch (combine) { \
      case COMBINE_SUM: d[i] = (S)((U)d[i] + (U)x[i]); break; \
      case COMBINE_PROD: d[i] = (S)((U)d[i] * (U)x[i]); break; \
      case COMBINE_MAX: if (x[i] > d[i]) d[i] = x[i]; break; \
      default: if (x[i] < d[i]) d[i] = x[i]; break; } } while (0)

/* dst[i] = dst[i] combine src[i] (Steps.__call__: op(flat[span], received, out=flat[span])) */
static void combine_into(ncclDataType_t t, int combine, void *dst, const void *src, size_t n) {
  switch (t) {
  case ncclInt8: INTEGER_LOOP(int8_t, uint8_t); return;
  case ncclUint8: INTEGER_LOOP(uint8_t, uint8_t); return;
  case ncclInt32: INTEGER_LOOP(int32_t, uint32_t); return;
  case ncclUint32: INTEGER_LOOP(uint32_t, uint32_t); return;
  case ncclInt64: INTEGER_LOOP(int64_t, uint64_t); return;
  case ncclUint64: INTEGER_LOOP(uint64_t, uint64_t); return;
  default: break;
  }
  const size_t z = SIZE[t];
  unsigned char *d = dst;
  const unsigned char *x = src;
  for (size_t i = 0; i < n; i++, d += z, x += z) {
    const double a = load(t, d), b = load(t, x);
    if (combine == COMBINE_MAX) { if (b > a) memcpy(d, x, z); }
    else if (combine == COMBINE_MIN) { if (b < a) memcpy(d, x, z); }
    else store(t, d, arithmetic(t, a, b, combine));
  }
}

/* dst[i] = src[i] * scalar in the type (np.multiply(sendbuff, premultiplier)); dst may be src */
static void premultiply(ncclDataType_t t, void *dst, const void *src, const unsigned char *scalar, size_t n) {
  if (dst != src) memmove(dst, src, n * SIZE[t]);
  switch (t) {
#define SCALE(S, U) { S s; memcpy(&s, scalar, sizeof s); S *d = dst; for (size_t i = 0; i < n; i++) d[i] = (S)((U)d[i] * (U)s); return; }
  case ncclInt8: SCALE(int8_t, uint8_t)
  case ncclUint8: SCALE(uint8_t, uint8_t)
  case ncclInt32: SCALE(int32_t, uint32_t)
  case ncclUint32: SCALE(uint32_t, uint32_t)
  case ncclInt64: SCALE(int64_t, uint64_t)
  case ncclUint64: SCALE(uint64_t, uint64_t)
#undef SCALE
  default: break;
  }
  const size_t z = SIZE[t];
  const double s = load(t, scalar);
  unsigned char *d = dst;
  for (size_t i = 0; i < n; i++, d += z) store(t, d, arithmetic(t, load(t, d), s, COMBINE_PROD));
}

/* FuncSumPostDiv::divide (src/device/reduce_kernel.h): the wrapped sum's magnitude over n, its sign kept */
static void truncdiv(ncclDataType_t t, void *dst, const void *src, size_t count, uint64_t n) {
  switch (t) {
#define DIVIDE(S, U, SIGNED) { const S *x = src; S *d = dst; for (size_t i = 0; i < count; i++) { \
      const int negative = SIGNED && (((U)x[i] >> (8 * sizeof(U) - 1)) & 1); const U magnitude = negative ? (U)(0 - (U)x[i]) : (U)x[i]; \
      const U q = (U)(magnitude / (U)n); d[i] = negative ? (S)(0 - q) : (S)q; } return; }
  case ncclInt8: DIVIDE(int8_t, uint8_t, 1)
  case ncclUint8: DIVIDE(uint8_t, uint8_t, 0)
  case ncclInt32: DIVIDE(int32_t, uint32_t, 1)
  case ncclUint32: DIVIDE(uint32_t, uint32_t, 0)
  case ncclInt64: DIVIDE(int64_t, uint64_t, 1)
  case ncclUint64: DIVIDE(uint64_t, uint64_t, 0)
#undef DIVIDE
  default: memmove(dst, src, count * SIZE[t]);
  }
}


/* -- communicators -------------------------------------------------------------------------------- */

struct premul { int used; ncclDataType_t type; unsigned char value[8]; };
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; };
struct session;

/* a communicator: rank r on the bridge of node members[r]; its ranks' link map (each link's cost its own), message paths
   and programs, the caller's (`given`) or else made from the bridges' map at each opening of its session; its name, and
   the count of splits and shrinks made of it, which name theirs; its session (NULL until a group crosses a link, and
   after a failure) */
struct ncclComm {
  struct mesh_link_map map;
  struct mesh_trees paths;
  struct entry *entries;
  uint32_t programs, splits, *members;
  int nranks, rank, given;
  char name[NCCL_UNIQUE_ID_BYTES + 1];
  struct premul ops[OPS];
  ncclResult_t async;
  struct session *session;
};

/* the sessions' one lock: a group, a completion, a session's opening and closing */
static pthread_mutex_t session_lock = PTHREAD_MUTEX_INITIALIZER;

static void program_free(struct mesh_program *program) {
  free((void *)program->segment); free((void *)program->log_weight); free((void *)program->first); free((void *)program->move);
}

static void routes_free(struct ncclComm *comm) {
  free(comm->map.link); free(comm->map.cost);
  free((void *)comm->paths.root); free((void *)comm->paths.parent); free((void *)comm->paths.log_weight);
  for (uint32_t e = 0; e < comm->programs; e++) program_free(&comm->entries[e].program);
  free(comm->entries);
  comm->map.link = NULL; comm->map.cost = NULL; comm->paths = (struct mesh_trees){0}; comm->entries = NULL; comm->programs = 0;
}

static void comm_free(struct ncclComm *comm) {
  routes_free(comm);
  free(comm->members);
  free(comm);
}

/* a program's own copy */
static int program_copy(struct mesh_program *made, const struct mesh_program *from) {
  const uint32_t moves = from->first[from->nodes];
  uint32_t *segment = malloc((from->parts + 1) * sizeof *segment), *first = malloc((from->nodes + 1) * sizeof *first);
  double *weight = malloc((from->parts + 1) * sizeof *weight);
  struct mesh_move *move = malloc((moves + 1) * sizeof *move);
  if (!segment || !first || !weight || !move) { free(segment); free(first); free(weight); free(move); return ENOMEM; }
  memcpy(segment, from->segment, from->parts * sizeof *segment); memcpy(weight, from->log_weight, from->parts * sizeof *weight);
  memcpy(first, from->first, (from->nodes + 1) * sizeof *first); memcpy(move, from->move, moves * sizeof *move);
  *made = *from;
  made->segment = segment; made->log_weight = weight; made->first = first; made->move = move;
  return 0;
}

/* `what` along `trees` compiled into the table's next entry, for calls below `below` bytes (0: any) */
static int compile_entry(struct ncclComm *comm, const struct mesh_trees *trees, uint32_t what, uint32_t root, int whole, uint64_t below) {
  uint32_t parts, moves;
  mesh_compile_room(trees, &parts, &moves);
  uint32_t *segment = malloc((parts + 1) * sizeof *segment), *first = malloc((trees->nodes + 1) * sizeof *first);
  double *weight = malloc((parts + 1) * sizeof *weight);
  struct mesh_move *move = malloc((moves + 1) * sizeof *move);
  struct entry *e = comm->entries + comm->programs;
  if (!segment || !first || !weight || !move || mesh_compile(trees, what, root, whole, NULL, &e->program, segment, weight, first, move)) {
    free(segment); free(first); free(weight); free(move);
    return EINVAL;
  }
  e->below = below;
  comm->programs++;
  return 0;
}


/* The communicator's message paths (the caller's, else the map's shortest-path trees) and its programs (the caller's
   table, else the declared default, compiled here once): the single-phase all-reduce along the shortest-path trees for
   calls below the first size (1 KiB to 16 GiB by powers of two) at which the reduce-scatter and all-gather along the
   map's packed trees is faster in the alpha-beta model (any size where none is: two nodes), then that all-reduce, the
   reduce-scatter, the all-gather, and each root's reduce and broadcast along the packed trees. */
static ncclResult_t comm_routes(struct ncclComm *comm, const struct mesh_trees *paths, const ncclMeshProgram *programs, int count) {
  const uint32_t n = (uint32_t)comm->nranks, most = 4;
  const uint32_t trees = paths ? paths->count : n;
  uint32_t *root = calloc(trees + 1, sizeof *root), *parent = calloc((size_t)trees * n + 1, sizeof *parent);
  double *weight = calloc(trees + 1, sizeof *weight);
  comm->paths = (struct mesh_trees){n, 0, root, parent, weight};
  if (!root || !parent || !weight) return fail(ncclSystemError, "out of memory");
  if (paths) {
    if (paths->nodes != n || mesh_trees_check(&comm->map, paths))
      return fail(ncclInvalidArgument, "the paths are not trees of the topology's links (mesh_trees_check)");
    memcpy(root, paths->root, trees * sizeof *root); memcpy(parent, paths->parent, (size_t)trees * n * sizeof *parent);
    memcpy(weight, paths->log_weight, trees * sizeof *weight);
    comm->paths.count = trees;
  } else if (!(comm->paths.count = mesh_trees_shortest(&comm->map, root, parent, weight)))
    return fail(ncclInvalidArgument, "the map's links do not join every rank (mesh_trees_shortest)");
  if (programs) {
    if (!(comm->entries = calloc((size_t)count + 1, sizeof *comm->entries))) return fail(ncclSystemError, "out of memory");
    for (int e = 0; e < count; e++) {
      if (!programs[e].program || programs[e].program->nodes != n) return fail(ncclInvalidArgument, "program %d is not of %u ranks", e, n);
      if (program_copy(&comm->entries[e].program, programs[e].program)) return fail(ncclSystemError, "out of memory");
      comm->entries[e].below = programs[e].below;
      comm->programs++;
    }
    return ncclSuccess;
  }
  if (n < 2) return ncclSuccess;
  uint32_t *proot = calloc((size_t)n * most + 1, sizeof *proot), *pparent = calloc((size_t)n * n * most + 1, sizeof *pparent);
  double *pweight = calloc((size_t)n * most + 1, sizeof *pweight);
  ncclResult_t result = ncclSuccess;
  uint32_t packed = 0;
  if (!proot || !pparent || !pweight || !(comm->entries = calloc(4 + 2 * (size_t)n, sizeof *comm->entries)))
    result = fail(ncclSystemError, "out of memory");
  else if (!(packed = mesh_trees_pack(&comm->map, 0.1, most, proot, pparent, pweight)))
    result = fail(ncclInvalidArgument, "the topology's links pack no trees");
  const struct mesh_trees tree = {n, packed, proot, pparent, pweight};
  if (!result && (compile_entry(comm, &comm->paths, MESH_ALLREDUCE, 0, 1, 0) || compile_entry(comm, &tree, MESH_ALLREDUCE, 0, 0, 0)))
    result = fail(ncclInternalError, "the default all-reduce does not compile");
  for (int k = 10; !result && k <= 34; k++) {
    const struct mesh_operand operand = {0, 4, ((uint64_t)1 << k) / 4};
    if (mesh_program_time(&comm->map, &comm->entries[1].program, operand, MESH_ALPHA_US, MESH_BETA_NS) <
        mesh_program_time(&comm->map, &comm->entries[0].program, operand, MESH_ALPHA_US, MESH_BETA_NS)) {
      comm->entries[0].below = (uint64_t)1 << k;
      break;
    }
  }
  if (!result && (compile_entry(comm, &tree, MESH_REDUCE_SCATTER, 0, 0, 0) || compile_entry(comm, &tree, MESH_ALLGATHER, 0, 0, 0)))
    result = fail(ncclInternalError, "the default reduce-scatter or all-gather does not compile");
  for (uint32_t r = 0; !result && r < n; r++)
    if (compile_entry(comm, &tree, MESH_REDUCE, r, 0, 0) || compile_entry(comm, &tree, MESH_BROADCAST, r, 0, 0))
      result = fail(ncclInternalError, "the default reduce or broadcast of root %u does not compile", r);
  free(proot); free(pparent); free(pweight);
  return result;
}

/* the program a call runs: the table's first entry of its collective (and root, a rooted one's) whose bound it is below */
static const struct mesh_program *program_of(ncclComm_t comm, int what, int root, uint64_t bytes) {
  for (uint32_t e = 0; e < comm->programs; e++) {
    const struct entry *x = comm->entries + e;
    if ((int)x->program.what != what || ((what == MESH_REDUCE || what == MESH_BROADCAST) && (int)x->program.root != root)) continue;
    if (!x->below || bytes < x->below) return &x->program;
  }
  return NULL;
}

/* a communicator of `nranks` ranks, rank r on the bridge of node members[r]: over `map`, the caller's (its links and
   costs copied, its messages along `paths`, its collectives the table `programs`, else comm_routes's defaults), or with
   `map` NULL over the bridges' map, read with its routes and programs at each opening of its session (comm_read) */
static ncclResult_t comm_make(ncclComm_t *made, const struct mesh_link_map *map, const struct mesh_trees *paths, const ncclMeshProgram *programs,
                              int listed, int nranks, const uint32_t *members, int rank, const char *name) {
  struct ncclComm *comm = calloc(1, sizeof *comm);
  if (!comm) return fail(ncclSystemError, "out of memory");
  comm->members = calloc((size_t)nranks, sizeof *comm->members);
  if (map)
    comm->map = (struct mesh_link_map){map->kind, (uint32_t)nranks, map->links, calloc(map->links + 1, sizeof *comm->map.link),
                                       map->cost ? calloc(map->links + 1, sizeof *comm->map.cost) : NULL};
  if (!comm->members || (map && (!comm->map.link || (map->cost && !comm->map.cost)))) { comm_free(comm); return fail(ncclSystemError, "out of memory"); }
  if (map && map->links) memcpy(comm->map.link, map->link, map->links * sizeof *map->link);
  if (map && map->links && map->cost) memcpy(comm->map.cost, map->cost, map->links * sizeof *map->cost);
  memcpy(comm->members, members, (size_t)nranks * sizeof *members);
  comm->nranks = nranks; comm->rank = rank; comm->given = map != NULL;
  snprintf(comm->name, sizeof comm->name, "%s", name);
  const ncclResult_t result = map ? comm_routes(comm, paths, programs, listed) : ncclSuccess;
  if (result) { comm_free(comm); return result; }
  *made = comm;
  return ncclSuccess;
}

/* the bridges' map over the communicator's members as it is now (mesh-plan.h mesh_map: members the cables leave apart
   joined along the route the bridges forward on), and its paths and default programs compiled on it */
static ncclResult_t comm_read(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks, room = n * (n - 1) / 2 + 1;
  routes_free(comm);
  comm->map = (struct mesh_link_map){MESH_LINKS_GRAPH, n, room, calloc(room, sizeof *comm->map.link), calloc(room, sizeof *comm->map.cost)};
  if (!comm->map.link || !comm->map.cost) return fail(ncclSystemError, "out of memory");
  const int read = mesh_map(comm->members, n, NULL, &comm->map);
  if (read < 0) return fail(ncclRemoteError, "the map of this node's bridge over the %u members of %s: %s", n, comm->name, strerror(-read));
  for (uint32_t r = 0; n > 1 && r < n; r++) {
    int linked = 0;
    for (uint32_t l = 0; l < comm->map.links && !linked; l++) linked = comm->map.link[l][0] == r || comm->map.link[l][1] == r;
    if (!linked) return fail(ncclRemoteError, "node %u, rank %u of %s, has no route on the map of this node's bridge", comm->members[r], r, comm->name);
  }
  return comm_routes(comm, NULL, NULL, 0);
}

ncclResult_t ncclGetVersion(int *version) {
  if (!version) return fail(ncclInvalidArgument, "version is NULL");
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t result) {
  static const char *const names[ncclNumResults] = {"no error", "unhandled cuda error", "unhandled system error",
    "internal error", "invalid argument", "invalid usage", "remote process exited or there was a network error",
    "NCCL operation in progress", "timeout"};
  return (unsigned)result < ncclNumResults ? names[result] : "unknown result code";
}

const char *ncclGetLastError(ncclComm_t comm) { (void)comm; return last; }

/* NCCL's bootstrap (NVIDIA/nccl src/bootstrap.cc): the unique id is the root's address, host:port (NCCL_COMM_ID where
   set, else a listener this process opens); rank 0 takes every rank's report there and sends each the whole table.  A
   rank's report is its node (mesh_observe); the links are the bridges', read at each session (comm_read).  The id is
   also the communicator's execution name, which its sessions take on every member (mesh_attach). */
static double now_s(void);
struct report { int32_t rank; uint32_t node; };
static struct { int fd; char id[NCCL_UNIQUE_ID_BYTES]; } root = {-1, {0}};

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  memset(id->internal, 0, sizeof id->internal);
  const char *given = getenv("NCCL_COMM_ID");
  if (given && *given) { snprintf(id->internal, sizeof id->internal, "%s", given); return ncclSuccess; }
  struct sockaddr_in6 any = {.sin6_len = sizeof any, .sin6_family = AF_INET6, .sin6_addr = IN6ADDR_ANY_INIT};
  socklen_t length = sizeof any;
  char host[256] = {0};
  if (root.fd >= 0) close(root.fd);
  root.fd = socket(AF_INET6, SOCK_STREAM, 0);
  if (root.fd < 0 || bind(root.fd, (struct sockaddr *)&any, sizeof any) || listen(root.fd, 64) ||
      getsockname(root.fd, (struct sockaddr *)&any, &length) || gethostname(host, sizeof host - 1))
    return fail(ncclSystemError, "the bootstrap root's listener: %s", strerror(errno));
  snprintf(id->internal, sizeof id->internal, "%s:%u", host, ntohs(any.sin6_port));
  memcpy(root.id, id->internal, sizeof root.id);
  return ncclSuccess;
}

static int whole(int fd, void *bytes, size_t size, int sending) {
  for (size_t done = 0; done < size;) {
    const ssize_t moved = sending ? send(fd, (char *)bytes + done, size - done, 0) : recv(fd, (char *)bytes + done, size - done, 0);
    if (moved <= 0) return moved ? errno : ECONNRESET;
    done += (size_t)moved;
  }
  return 0;
}

static ncclResult_t bootstrap(const char *id, int nranks, int rank, struct report *reports) {
  struct report *mine = reports + rank;
  uint32_t node = 0;
  const int ports = mesh_observe(NULL, 0, &node);
  *mine = (struct report){rank, node};
  if (ports < 0) return fail(ncclSystemError, "this node's bridge: %s", strerror(-ports));
  if (nranks == 1) return ncclSuccess;
  char host[NCCL_UNIQUE_ID_BYTES];
  snprintf(host, sizeof host, "%s", id);
  char *colon = strrchr(host, ':');
  if (!colon) return fail(ncclInvalidArgument, "the unique id %s is no host:port", id);
  *colon = 0;
  struct timeval bound = {(time_t)DEADLINE_S, 0};
  const double deadline = now_s() + DEADLINE_S;
  int status = 0;
  if (rank == 0) {
    int listener = !strcmp(root.id, id) ? root.fd : -1;
    if (listener < 0) {
      struct sockaddr_in6 any = {.sin6_len = sizeof any, .sin6_family = AF_INET6, .sin6_addr = IN6ADDR_ANY_INIT,
                                 .sin6_port = htons((uint16_t)atoi(colon + 1))};
      int on = 1;
      listener = socket(AF_INET6, SOCK_STREAM, 0);
      if (listener < 0 || setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on) ||
          bind(listener, (struct sockaddr *)&any, sizeof any) || listen(listener, 64))
        return fail(ncclSystemError, "the bootstrap root at %s: %s", id, strerror(errno));
    }
    int *peers = calloc((size_t)nranks, sizeof *peers);
    for (int r = 1; peers && r < nranks && !status; r++) {
      struct pollfd ready = {listener, POLLIN, 0};
      const int left = (int)((deadline - now_s()) * 1000);
      int fd = left > 0 && poll(&ready, 1, left) == 1 ? accept(listener, NULL, NULL) : -1;
      struct report heard;
      if (fd < 0 || setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &bound, sizeof bound) || setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &bound, sizeof bound) ||
          (status = whole(fd, &heard, sizeof heard, 0)) || heard.rank <= 0 || heard.rank >= nranks || peers[heard.rank]) {
        status = status ? status : ETIMEDOUT;
        if (fd >= 0) close(fd);
        break;
      }
      peers[heard.rank] = fd;
      reports[heard.rank] = heard;
    }
    for (int r = 1; peers && r < nranks; r++) {
      if (!status && peers[r]) status = whole(peers[r], reports, (size_t)nranks * sizeof *reports, 1);
      if (peers[r]) close(peers[r]);
    }
    close(listener);
    if (listener == root.fd) root.fd = -1;
    free(peers);
    if (!peers) return fail(ncclSystemError, "out of memory");
  } else {
    struct addrinfo hint = {.ai_socktype = SOCK_STREAM}, *addresses = NULL;
    int fd = -1;
    while (fd < 0 && now_s() < deadline) {
      if (!getaddrinfo(host, colon + 1, &hint, &addresses)) {
        for (struct addrinfo *a = addresses; a && fd < 0; a = a->ai_next) {
          fd = socket(a->ai_family, SOCK_STREAM, 0);
          if (fd >= 0 && connect(fd, a->ai_addr, a->ai_addrlen)) { close(fd); fd = -1; }
        }
        freeaddrinfo(addresses);
      }
      if (fd < 0) usleep(100000);
    }
    if (fd < 0) status = ETIMEDOUT;
    if (!status && (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &bound, sizeof bound) || setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &bound, sizeof bound)))
      status = errno;
    struct report kept = *mine;
    if (!status) status = whole(fd, mine, sizeof *mine, 1);
    if (!status) status = whole(fd, reports, (size_t)nranks * sizeof *reports, 0);
    if (fd >= 0) close(fd);
    reports[rank] = kept;
  }
  return status ? fail(ncclRemoteError, "the bootstrap at %s: %s", id, strerror(status)) : ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return fail(ncclInvalidArgument, "rank %d of %d", rank, nranks);
  char id[sizeof commId.internal + 1];
  memcpy(id, commId.internal, sizeof commId.internal);
  id[sizeof commId.internal] = 0;
  if (!*id) return fail(ncclInvalidArgument, "the unique id is empty (ncclGetUniqueId)");
  struct report *reports = calloc((size_t)nranks, sizeof *reports);
  uint32_t *members = calloc((size_t)nranks, sizeof *members);
  ncclResult_t result = reports && members ? bootstrap(id, nranks, rank, reports) : fail(ncclSystemError, "out of memory");
  for (int r = 0; !result && r < nranks; r++) members[r] = reports[r].node;
  if (!result) result = comm_make(comm, NULL, NULL, NULL, 0, nranks, members, rank, id);
  free(reports); free(members);
  return result;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config) {
  (void)config;
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  (void)devlist;
  if (ndev != 1) return fail(ncclInvalidArgument, "a node has one Metal device: ndev %d", ndev);
  ncclUniqueId id;
  const ncclResult_t result = ncclGetUniqueId(&id);
  return result ? result : ncclCommInitRank(comm, 1, id, 0);
}

/* a communicator over the caller's topology operand; the bridges carry a message between ranks it does not link along
   their own map's route */
ncclResult_t ncclMeshCommInitRank(ncclComm_t *comm, int rank, const struct mesh_link_map *topology, const struct mesh_trees *paths,
                                  const ncclMeshProgram *programs, int count, const int *node, const char *name) {
  static const char *const kinds[] = {"mesh", "ring", "tree", "graph"};
  if (!comm || !topology) return fail(ncclInvalidArgument, "comm or topology is NULL");
  if (rank < 0 || (uint32_t)rank >= topology->nodes) return fail(ncclInvalidArgument, "rank %d of %u", rank, topology->nodes);
  if (mesh_link_map_check(topology))
    return fail(ncclInvalidArgument, "the topology (%s of %u ranks, %u links) is not one its algorithms can run on (mesh-plan.h)",
                topology->kind <= MESH_LINKS_GRAPH ? kinds[topology->kind] : "unknown kind", topology->nodes, topology->links);
  if (count < 0 || (count && !programs)) return fail(ncclInvalidArgument, "programs is NULL with count %d", count);
  uint32_t *members = calloc(topology->nodes + 1, sizeof *members);
  if (!members) return fail(ncclSystemError, "out of memory");
  for (uint32_t r = 0; r < topology->nodes; r++) members[r] = node ? (uint32_t)node[r] : r;
  const ncclResult_t result = comm_make(comm, topology, paths, count ? programs : NULL, count, (int)topology->nodes, members, rank,
                                        name && *name ? name : "nccl");
  free(members);
  return result;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL"); }

static void session_close(struct session *s, double linger);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  pthread_mutex_lock(&session_lock);
  session_close(comm->session, LINGER_S);
  pthread_mutex_unlock(&session_lock);
  comm_free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

static ncclResult_t session_async(ncclComm_t comm);
/* a launch's failure, else the failure of the communicator's session (the progress thread's: a link lost or silent),
   which a group that had already returned, its work enqueued, could not report */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  *asyncError = comm->async ? comm->async : session_async(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  if (!comm || !count) return fail(ncclInvalidArgument, "comm or count is NULL");
  *count = comm->nranks;
  return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  if (!comm || !device) return fail(ncclInvalidArgument, "comm or device is NULL");
  *device = 0;
  return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  if (!comm || !rank) return fail(ncclInvalidArgument, "comm or rank is NULL");
  *rank = comm->rank;
  return ncclSuccess;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
  (void)size;
  if (!comm || !handle) return fail(ncclInvalidArgument, "comm or handle is NULL");
  *handle = buff;
  return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  (void)handle;
  return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL");
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  if (!ptr) return fail(ncclInvalidArgument, "ptr is NULL");
  if (posix_memalign(ptr, (size_t)getpagesize(), size ? size : 1)) return fail(ncclSystemError, "%zu bytes: out of memory", size);
  return ncclSuccess;
}

ncclResult_t ncclMemFree(void *ptr) { free(ptr); return ncclSuccess; }

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  if (!op || !scalar || !comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "PreMulSum's arguments");
  if (residence != ncclScalarHostImmediate) return fail(ncclInvalidArgument, "the host path reads the scalar when the operator is made (ncclScalarHostImmediate)");
  for (int k = 0; k < OPS; k++)
    if (!comm->ops[k].used) {
      comm->ops[k].used = 1; comm->ops[k].type = datatype;
      memcpy(comm->ops[k].value, scalar, SIZE[datatype]);
      *op = (ncclRedOp_t)(ncclNumOps + k);
      return ncclSuccess;
    }
  return fail(ncclInvalidUsage, "%d PreMulSum operators on one communicator", OPS);
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  const int k = (int)op - ncclNumOps;
  if (!comm || k < 0 || k >= OPS || !comm->ops[k].used) return fail(ncclInvalidArgument, "operator %d is none of this communicator's", (int)op);
  comm->ops[k].used = 0;
  return ncclSuccess;
}

/* -- the session: a communicator's prepared program on its node's bridge, attached under the communicator's name at
   its first group that crosses a link and kept until a failure or the communicator's end.  A channel to each other
   member: a ring of DEPTH slots each way, slot t % DEPTH carrying invocation t (a position), which the bridges carry
   along their map's route where no link joins the two nodes.  Both ends of a channel compute the same positions for a
   group (each collective's rounds in call order, then the point-to-point calls paired in order, a segment as long as
   its longer direction), each publishes every position (filler where it has nothing to send), and each publishes
   position t once it has consumed the peer's t - LAG: the peer then has consumed this end's t - 2 LAG = t - DEPTH, the
   slot t reuses, so no slot is overwritten before it is read (NCCL's Simple protocol's slots, its head credit carried
   by the reverse direction's positions). ------------------------------------------------------------------------ */

#define LAG 8                       /* NCCL_STEPS */
#define DEPTH (2 * LAG)
#define BUFFSIZE ((size_t)1 << 22)  /* NCCL_BUFFSIZE's default */
#define POSITIONS 8192
#define IDENTITY 1                  /* a channel's transfers: IDENTITY + 2 * its sender's rank + large */

/* a channel to a member: one class of its pieces on a queue of its own (a queue's receives are posted invocation-major
   across its transfers, so two rings on one would advance together): `large`, NCCL_BUFFSIZE/NCCL_STEPS slots on queue
   0, else one-block slots on queue 1, so a small piece pays one block on the wire (NCCL picks its protocol by size).
   Its rings of `slot` bytes; its Metal side: its rings as buffers, its out ring's invocation-0 SEND cell in the
   transport's publication buffer and the producer argument a publication stores there, its in ring's invocation-0
   completion word and the words' stride in the transport's inputs; `waited`, the arrivals the encoded GPU work already
   waits for; `checked`, the positions checked against their roles; `signaled`, the positions landed that the progress
   thread has seen, up to the `target` the groups reach */
struct channel {
  uint32_t node;
  int large;
  size_t slot;
  struct mesh_section out, in;
  unsigned char *sending, *receiving;
  uint64_t sent, consumed, end, waited, checked;
  void *ring_out, *ring_in;
  size_t cell, word, word_stride;
  uint64_t argument;
  _Atomic uint64_t target, signaled;
};

/* a communicator's session: its client of the node's bridge, its channels, `positions` positions a cycle, its progress
   thread and its failure once one is kept (`failed`, `failure`); `next`, the process's other sessions, closed at its
   exit */
struct session {
  struct ncclComm *comm;
  struct mesh_ctx context;
  struct hdr *header;
  struct channel *channels;
  uint32_t count, positions;
  size_t slot, small;
  int running;
  struct mesh_metal_transport transport;
  void *violation_buffer;
  _Atomic uint64_t *violation;
  pthread_t progress;
  pthread_mutex_t lock;
  pthread_cond_t wake;
  _Atomic int stop, failed;
  char failure[512];
  struct session *next;
};
static struct session *sessions;
static int registered;
static void *finished;
static uint64_t issued;

/* a position's completion word as M30 reads it: landed once at least 1 plus its cycle, ~0 its link cancelled */
static uint64_t cycle_of(const struct session *s, uint64_t at) { return at / s->positions; }
static int landed(const struct session *s, uint64_t word, uint64_t at) { return word != UINT64_MAX && word >= 1 + cycle_of(s, at); }

static double now_s(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

/* every destination of the session with pending words cancelled (mesh_cancel: each pending word ~0, so every spin on
   it ends) and the failure kept for its next group */
static void release_all(struct session *s, int result, const char *format, ...) {
  int none = ncclSuccess;
  if (atomic_compare_exchange_strong(&s->failed, &none, result)) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(s->failure, sizeof s->failure, format, arguments);
    va_end(arguments);
  }
  struct mesh_client *client = mesh_client_at(s->header, s->context.client);
  for (uint32_t d = 0; s->transport.cancel && d < MESH_DESTINATIONS && client->tx[d].node; d++) mesh_cancel(s->header, s->transport.cancel, d);
  for (uint32_t h = 0; h < s->count; h++) atomic_store(&s->channels[h].signaled, atomic_load(&s->channels[h].target));
}

/* the communicator's session's failure, where no group holds the sessions' lock (a group in progress returns its own) */
static void violated(struct session *s);
static ncclResult_t session_async(ncclComm_t comm) {
  if (pthread_mutex_trylock(&session_lock)) return ncclSuccess;
  struct session *s = comm->session;
  if (s) violated(s);
  const ncclResult_t failed = s ? (ncclResult_t)atomic_load(&s->failed) : ncclSuccess;
  if (failed) fail(failed, "%s", s->failure);
  pthread_mutex_unlock(&session_lock);
  return failed;
}

static int outstanding(struct session *s) {
  for (uint32_t h = 0; h < s->count; h++)
    if (atomic_load(&s->channels[h].signaled) < atomic_load(&s->channels[h].target)) return 1;
  return 0;
}

/* a position whose role the two ends disagree on: a filler's slot carries its position's stamp (metal_stamp), which a
   receiver checks where it receives a piece (absent) and where it receives nothing (present) */
#define MISMATCH "position %llu on the channel to node %u: this rank receives %s there and its peer sent %s (send and " \
                 "receive calls that do not pair: an ncclSend and an ncclRecv issued separately in the same order on both " \
                 "ranks, or different counts or peers; or different collectives on the two ranks)"

/* a violation the Metal path's walk recorded (struct metal_check), made the session's failure */
static void violated(struct session *s) {
  const uint64_t at = s->violation ? atomic_load_explicit(s->violation, memory_order_acquire) : 0;
  if (!at || atomic_load(&s->failed)) return;
  const uint64_t where = atomic_load_explicit(s->violation + 1, memory_order_relaxed);
  const uint32_t h = (uint32_t)(where >> 1);
  release_all(s, ncclInvalidUsage, MISMATCH, (unsigned long long)(at - 1), h < s->count ? s->channels[h].node : UINT32_MAX,
              where & 1 ? "a piece" : "nothing", where & 1 ? "nothing" : "a piece");
}

/* the progress thread, the session's one silent-link watcher: it follows each channel's landings up to its target, and
   where a channel has positions pending and none lands for mesh.h MESH_SILENT_NS, or a landing reads ~0 (the bridge
   cancelled it), every destination of the session is cancelled */
static void *progress_run(void *argument) {
  struct session *s = argument;
  double idle = now_s() + SILENT_S;
  while (!atomic_load(&s->stop)) {
    violated(s);
    int waiting = 0, moved = 0;
    uint32_t silent = 0;
    for (uint32_t h = 0; h < s->count && !atomic_load(&s->failed); h++) {
      struct channel *ch = s->channels + h;
      const uint64_t from = atomic_load(&ch->signaled), target = atomic_load(&ch->target);
      uint64_t at = from;
      for (; at < target; at++) {
        const uint64_t arrived = mesh_host_arrived(&s->context, ch->in, (uint32_t)at);
        if (arrived == UINT64_MAX) { release_all(s, ncclRemoteError, "the link to node %u was lost (its landings cancelled)", ch->node); break; }
        if (!landed(s, arrived, at)) { if (!waiting) silent = ch->node; waiting = 1; break; }
      }
      if (at > from && !atomic_load(&s->failed)) { atomic_store(&ch->signaled, at); moved = 1; }
    }
    if (moved) idle = now_s() + SILENT_S;
    else if (waiting && now_s() > idle)
      release_all(s, ncclRemoteError, "node %u landed nothing in %.0f s (mesh.h MESH_SILENT_NS): its links cancelled", silent, SILENT_S);
    if (!waiting || atomic_load(&s->failed)) {
      pthread_mutex_lock(&s->lock);
      while (!atomic_load(&s->stop) && (!outstanding(s) || atomic_load(&s->failed))) pthread_cond_wait(&s->wake, &s->lock);
      pthread_mutex_unlock(&s->lock);
      idle = now_s() + SILENT_S;
    }
  }
  return NULL;
}

static void progress_targets(struct session *s) {
  pthread_mutex_lock(&s->lock);
  for (uint32_t h = 0; h < s->count; h++) atomic_store(&s->channels[h].target, s->channels[h].end);
  pthread_cond_signal(&s->wake);
  pthread_mutex_unlock(&s->lock);
}

/* every encoded GPU program finished (its end signals `finished`); MESH_SILENT_NS with no program finishing and no
   position of session s (NULL: none) landing releases s's waits, and a second ends the drain */
static ncclResult_t drain(struct session *s) {
  if (!finished) return ncclSuccess;
  double idle = now_s() + SILENT_S;
  int released = 0;
  uint64_t last = metal_signaled(finished), landed = 0;
  while (metal_signaled(finished) < issued) {
    uint64_t now = 0;
    for (uint32_t h = 0; s && h < s->count; h++) now += atomic_load(&s->channels[h].signaled);
    if (metal_signaled(finished) != last || now != landed) { last = metal_signaled(finished); landed = now; idle = now_s() + SILENT_S; }
    else if (now_s() > idle) {
      if (released) return fail(ncclInternalError, "a GPU program of the session did not finish (its command buffer committed?)");
      if (s) release_all(s, ncclTimeout, "the session's GPU programs ran %.0f s with nothing landing", SILENT_S);
      released = 1;
      idle = now_s() + SILENT_S;
    }
    usleep(50);
  }
  metal_collect(finished);
  return ncclSuccess;
}

static void inflight_lose(struct session *s);
/* the session ended: its groups in flight lost, every GPU program finished, its progress thread joined, `linger`
   seconds for the bridge to post its last publications, its client detached */
static void session_close(struct session *s, double linger) {
  if (!s) return;
  inflight_lose(s);
  drain(s);
  if (s->running) {
    pthread_mutex_lock(&s->lock);
    atomic_store(&s->stop, 1);
    pthread_cond_signal(&s->wake);
    pthread_mutex_unlock(&s->lock);
    pthread_join(s->progress, NULL);
  }
  if (linger > 0) usleep((useconds_t)(linger * 1e6));
  for (uint32_t h = 0; h < s->count; h++) { metal_release(s->channels[h].ring_out); metal_release(s->channels[h].ring_in); }
  if (s->violation_buffer) metal_release(s->violation_buffer);
  free((void *)s->violation);
  mesh_metal_transport_destroy(&s->transport);
  mesh_detach(&s->context);
  for (struct session **at = &sessions; *at; at = &(*at)->next)
    if (*at == s) { *at = s->next; break; }
  s->comm->session = NULL;
  pthread_mutex_destroy(&s->lock);
  pthread_cond_destroy(&s->wake);
  free(s->channels);
  free(s);
}

/* the process's exit: every GPU program finished, one linger for every session's last publications, each closed */
static void session_exit(void) {
  pthread_mutex_lock(&session_lock);
  if (sessions) { drain(NULL); usleep((useconds_t)(LINGER_S * 1e6)); }
  while (sessions) session_close(sessions, 0);
  pthread_mutex_unlock(&session_lock);
}

static size_t setting(const char *name, size_t otherwise) {
  const char *value = getenv(name);
  const long long x = value ? atoll(value) : 0;
  return x > 0 ? (size_t)x : otherwise;
}

static int ring_make(struct session *s, size_t slot, struct mesh_section *section, unsigned char **memory) {
  const int status = mesh_section_create(&s->context, DEPTH * slot, 1, section);
  if (status) return status;
  *memory = mesh_section_address(&s->context, *section, 0);
  section->bytes = slot;
  return 0;
}

/* a ring's transfer toward the member `node` on its queue `queue` (a queue's SENDs, and the receives its peer posts,
   run invocation by invocation through every transfer on it, so rings that advance apart need queues apart) */
static int ring_bind(struct session *s, uint32_t node, uint32_t queue, int receive, uint32_t identity, struct mesh_section section) {
  return mesh_transfer_bind(&s->context, node, queue, receive ? MESH_RECEIVE : MESH_SEND, identity, section,
                            (uint32_t)(section.bytes / s->header->pgsz), 0, UINT32_MAX, DEPTH);
}

/* a channel's Metal side once its transfers are prepared: the rings wrapped, its SEND cell, its completion words */
static int channel_metal(struct session *s, struct channel *ch) {
  ch->ring_out = metal_wrap(ch->sending, DEPTH * ch->slot);
  ch->ring_in = metal_wrap(ch->receiving, DEPTH * ch->slot);
  struct prepared_publication record;
  if (mesh_publication_prepare(s->header, ch->out.first, NULL) != 1) return EINVAL;
  mesh_publication_prepare(s->header, ch->out.first, &record);
  ch->cell = (size_t)(record.destination - ((uintptr_t)s->header + s->context.send_off));
  ch->argument = record.argument;
  struct mesh_metal_input input;
  const int status = mesh_metal_receive_prepare(&s->context, &s->transport, ch->in, &input);
  if (status) return status;
  metal_release(input.completion);
  ch->word = input.offset; ch->word_stride = input.stride;
  return ch->ring_out && ch->ring_in ? 0 : ENOMEM;
}

/* a start's failure as its destination's status names it: the node it was bound for, the node where it failed (that
   node's, or a relay's on its route) and the code (mesh-call.h mesh_transfers_start) */
static ncclResult_t start_failure(struct session *s, int status) {
  struct mesh_client *client = mesh_client_at(s->header, s->context.client);
  for (uint32_t d = 0; d < MESH_DESTINATIONS && client->tx[d].node; d++) {
    const int64_t code = atomic_load(&client->status[d].code);
    if (code)
      return fail(ncclRemoteError, "the session of %s toward node %u failed at node %u: %s", s->comm->name, client->tx[d].node,
                  atomic_load(&client->status[d].failed), strerror((int)code));
  }
  return fail(status == ESRCH ? ncclRemoteError : ncclSystemError, "the session of %s did not start on this node's bridge: %s",
              s->comm->name, strerror(status));
}

/* the communicator's session opened: on the bridges' map as it is now where the communicator's map is not the caller's
   (comm_read), a channel of each class to every other member, its transfers prepared and started, its progress thread */
static ncclResult_t session_open(ncclComm_t comm) {
  ncclResult_t result = comm->given ? ncclSuccess : comm_read(comm);
  if (result) return result;
  struct session *s = calloc(1, sizeof *s);
  if (!s) return fail(ncclSystemError, "out of memory");
  int status = mesh_attach(&s->context, comm->name);
  if (status) {
    free(s);
    return fail(status == ESRCH ? ncclRemoteError : ncclSystemError, "this node's bridge, attaching %s: %s", comm->name, strerror(status));
  }
  pthread_mutex_init(&s->lock, NULL);
  pthread_cond_init(&s->wake, NULL);
  s->comm = comm;
  comm->session = s;
  const struct hdr *m = s->header = s->context.M;
  if (m->node != comm->members[comm->rank]) {
    result = fail(ncclInvalidUsage, "rank %d of %s runs on the bridge of node %u, not of node %u", comm->rank, comm->name, m->node,
                  comm->members[comm->rank]);
    session_close(s, 0);
    return result;
  }
  const size_t block = (size_t)m->pgsz * m->block;
  s->slot = (setting("NCCL_BUFFSIZE", BUFFSIZE) / LAG + block - 1) / block * block;
  s->small = block;
  s->positions = (uint32_t)((setting("MESH_POSITIONS", POSITIONS) + DEPTH - 1) / DEPTH * DEPTH);
  const uint32_t N = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  status = (s->channels = calloc(2 * (size_t)N, sizeof *s->channels)) ? 0 : ENOMEM;
  for (uint32_t r = 0; r < N && !status; r++)
    for (uint32_t large = 0; r != me && large < 2 && !status; large++) {
      struct channel *ch = s->channels + s->count++;
      ch->node = comm->members[r];
      ch->large = (int)large;
      ch->slot = large ? s->slot : s->small;
      status = ring_make(s, ch->slot, &ch->out, &ch->sending);
      if (!status) status = ring_make(s, ch->slot, &ch->in, &ch->receiving);
      if (!status) status = ring_bind(s, ch->node, !large, 0, IDENTITY + 2 * me + large, ch->out);
      if (!status) status = ring_bind(s, ch->node, !large, 1, IDENTITY + 2 * r + large, ch->in);
    }
  if (!status) status = mesh_transfers_prepare(&s->context, 1, s->positions, DEPTH);
  if (!status) status = mesh_transfers_cyclic(&s->context);
  if (!status) status = mesh_metal_transport_create(&s->context, metal_device(), &s->transport);
  for (uint32_t h = 0; h < s->count && !status; h++) status = channel_metal(s, s->channels + h);
  void *violation = NULL;
  if (!status && posix_memalign(&violation, (size_t)getpagesize(), (size_t)getpagesize())) status = ENOMEM;
  if (!status) {
    memset(violation, 0, (size_t)getpagesize());
    s->violation = violation;
    if (!(s->violation_buffer = metal_wrap(violation, (size_t)getpagesize()))) status = ENOMEM;
  }
  if (!status && !finished && !(finished = metal_event())) status = ENOMEM;
  if (status) result = fail(ncclSystemError, "the session of %s on this node's bridge: %s", comm->name, strerror(status));
  else if ((status = mesh_transfers_start(&s->context))) result = start_failure(s, status);
  else if ((status = pthread_create(&s->progress, NULL, progress_run, s))) result = fail(ncclSystemError, "the progress thread: %s", strerror(status));
  if (result) { session_close(s, 0); return result; }
  s->running = 1;
  s->next = sessions;
  sessions = s;
  if (!registered) registered = !atexit(session_exit);
  return ncclSuccess;
}

/* -- a group ------------------------------------------------------------------------------------------- */

/* where a step's operand is on the Metal path: a buffer and a byte offset into it */
struct where { void *buffer; size_t offset; };

struct steps {
  struct mesh_step *steps;
  uint32_t count;
  unsigned char *done;
  /* the buffers the steps name: 0 the operand, b >= 1 scratch b (its copy); on the host (`host`, scratch owned) or on
     the GPU (`gpu`) */
  uint32_t buffers;
  unsigned char **host;
  struct where *gpu;
  int root;
  const struct mesh_program *program;
  struct mesh_operand operand;
  /* each step's positions as schedule_group places them (its channel, first position, pieces), and the receive before
     it (by round, then plan order) that has any, or -1 */
  uint32_t *channel;
  uint64_t *first, *pieces;
  int32_t *before;
};

struct call {
  int what, root, peer;
  const void *send;
  void *recv;
  size_t count;
  ncclDataType_t type;
  ncclRedOp_t op;
  ncclComm_t comm;
  int combine, post;
  unsigned char pre[8];
  int premultiplied;
  struct steps *bound;
  unsigned char *result, *copy;
  int owned;
  cudaStream_t stream;
  void *command, *encoder, *(*workspace)(size_t, void *), *context;
  ncclMeshIssue *issue;
  struct where send_at, recv_at, sum;
};

static _Thread_local int depth;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;
static _Thread_local int *plan_steps, *plan_root;
static _Thread_local size_t plan_count;

static int reducing(int what) { return what == MESH_ALLREDUCE || what == MESH_REDUCE || what == MESH_REDUCE_SCATTER; }
static int p2p(const struct call *c) { return c->what == WHAT_SEND || c->what == WHAT_RECV; }
static size_t elements(const struct call *c) {
  return c->count * ((c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER) ? (size_t)c->comm->nranks : 1);
}
/* whether a call crosses a link: elements, more than one rank, and a point-to-point call's peer another rank */
static int crosses(const struct call *c) { return c->count && c->comm->nranks > 1 && !(p2p(c) && c->peer == c->comm->rank); }

static void steps_free(struct steps *s) {
  if (!s) return;
  for (uint32_t b = 1; s->host && b < s->buffers; b++) free(s->host[b]);
  free(s->host); free(s->gpu);
  free(s->steps); free(s->done); free(s->channel); free(s->first); free(s->pieces); free(s->before); free(s);
}

/* every call that crosses a link planned: a collective along the communicator's trees, a point-to-point call one SEND or
   COPY step */
static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
    const struct mesh_program *program = p2p(c) ? NULL : program_of(comm, c->what, c->root, elements(c) * SIZE[c->type]);
    if (!p2p(c) && !program) return fail(ncclInvalidUsage, "the communicator has no program for collective %d at root %d", c->what, c->root);
    struct steps *s = calloc(1, sizeof *s);
    const size_t most = (program ? program->first[program->nodes] : 1) + 1;
    if (s) s->buffers = 1 + (program ? program->scratch : 0);
    if (!s || !(s->steps = calloc(most, sizeof *s->steps)) || !(s->done = calloc(most, 1)) || !(s->channel = calloc(most, sizeof *s->channel)) ||
        !(s->first = calloc(most, sizeof *s->first)) || !(s->pieces = calloc(most, sizeof *s->pieces)) || !(s->before = calloc(most, sizeof *s->before)) ||
        !(s->host = calloc(s->buffers, sizeof *s->host)) || !(s->gpu = calloc(s->buffers, sizeof *s->gpu))) {
      steps_free(s);
      return fail(ncclSystemError, "out of memory");
    }
    c->bound = s;
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], elements(c)};
    s->operand = operand;
    s->program = program;
    s->root = c->root;
    if (p2p(c)) {
      s->steps[0] = (struct mesh_step){c->what == WHAT_SEND ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)c->peer, 0, 0, 0, operand};
      s->count = 1;
      continue;
    }
    s->count = mesh_program_steps(program, (uint32_t)comm->rank, operand, s->steps);
  }
  return ncclSuccess;
}

/* a step's piece on one channel: its positions [first, first + pieces), at byte offset of its operand */
struct message { struct call *c; uint32_t step; uint64_t first, pieces; size_t offset, bytes; };
struct schedule { struct message *out, *in; size_t outs, ins, out_at, in_at; };

static size_t piece_bytes(const struct mesh_step *step) { return step->piece.elements * step->piece.element_bytes; }
static int in_class(const struct session *s, const struct mesh_step *step, int large) { return (piece_bytes(step) > LAG * s->small) == large; }

/* the comm's rank on a channel's node (-1: none) */
static int rank_of(ncclComm_t comm, uint32_t node) {
  for (int r = 0; r < comm->nranks; r++)
    if (r != comm->rank && comm->members[r] == node) return r;
  return -1;
}

/* a message as one end of a channel has it: its call and step, whether it is that end's SEND, its first position from
   the channel's start for the group, and its pieces */
struct placing { uint32_t call, step; int send; uint64_t first, pieces; };

/* the positions a rank's messages take on its channel to rank `peer` of class `large` for the group, from 0: each
   collective's rounds in call order (a round's SENDs and receives side by side from where the last round ended), then
   its point-to-point calls, its j-th SEND to the peer beside its j-th receive from it; steps[i] and counts[i] are its
   steps of call i (a point-to-point call's only for this rank).  The count of positions; `made` gets the messages (no
   zero-piece step), *count of them */
static uint64_t lay(const struct session *ses, const struct call *list, size_t n, struct mesh_step *const *steps, const uint32_t *counts,
                    int peer, int large, size_t slot, struct placing *made, size_t *count) {
  uint64_t at = 0;
  *count = 0;
  for (size_t i = 0; i < n; i++) {
    if (p2p(list + i)) continue;
    uint32_t rounds = 0;
    for (uint32_t k = 0; k < counts[i]; k++)
      if ((int)steps[i][k].peer == peer && in_class(ses, steps[i] + k, large) && steps[i][k].round + 1 > rounds) rounds = steps[i][k].round + 1;
    for (uint32_t round = 0; round < rounds; round++) {
      uint64_t out = at, in = at;
      for (uint32_t k = 0; k < counts[i]; k++) {
        const struct mesh_step *step = steps[i] + k;
        const uint64_t pieces = (piece_bytes(step) + slot - 1) / slot;
        if ((int)step->peer != peer || !in_class(ses, step, large) || step->round != round || !pieces) continue;
        uint64_t *cursor = step->op == MESH_STEP_SEND ? &out : &in;
        made[(*count)++] = (struct placing){(uint32_t)i, k, step->op == MESH_STEP_SEND, *cursor, pieces};
        *cursor += pieces;
      }
      at = out > in ? out : in;
    }
  }
  for (size_t s = 0, r = 0;; s++, r++) {
    while (s < n && !(p2p(list + s) && counts[s] && list[s].what == WHAT_SEND && (int)steps[s][0].peer == peer && in_class(ses, steps[s], large))) s++;
    while (r < n && !(p2p(list + r) && counts[r] && list[r].what == WHAT_RECV && (int)steps[r][0].peer == peer && in_class(ses, steps[r], large))) r++;
    if (s >= n && r >= n) break;
    uint64_t out = at, in = at;
    if (s < n && piece_bytes(steps[s])) { made[(*count)++] = (struct placing){(uint32_t)s, 0, 1, out, (piece_bytes(steps[s]) + slot - 1) / slot}; out += made[*count - 1].pieces; }
    if (r < n && piece_bytes(steps[r])) { made[(*count)++] = (struct placing){(uint32_t)r, 0, 0, in, (piece_bytes(steps[r]) + slot - 1) / slot}; in += made[*count - 1].pieces; }
    at = out > in ? out : in;
  }
  return at;
}

/* the receive before step k (by round, then plan order) with positions (pieces[j] > 0), or -1 */
static int32_t before_of(const struct mesh_step *steps, uint32_t count, const uint64_t *pieces, uint32_t k) {
  int32_t best = -1;
  for (uint32_t j = 0; j < count; j++)
    if (steps[j].op != MESH_STEP_SEND && pieces[j] && (steps[j].round < steps[k].round || (steps[j].round == steps[k].round && j < k)) &&
        (best < 0 || steps[j].round > steps[best].round || (steps[j].round == steps[best].round && (int32_t)j > best)))
      best = (int32_t)j;
  return best;
}

/* each channel's positions for the group, from the last scheduled one's end (lay); each step's positions and the receive
   before it kept, a receive with none done */
static ncclResult_t schedule_group(struct session *s, struct call *list, size_t n, struct schedule *plans) {
  const ncclComm_t comm = list[0].comm;
  size_t total = 1;
  for (size_t i = 0; i < n; i++) total += list[i].bound ? list[i].bound->count : 0;
  struct mesh_step **steps = calloc(n + 1, sizeof *steps);
  uint32_t *counts = calloc(n + 1, sizeof *counts);
  struct placing *made = calloc(total, sizeof *made);
  ncclResult_t result = steps && counts && made ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (size_t i = 0; i < n && !result; i++)
    if (list[i].bound) {
      steps[i] = list[i].bound->steps; counts[i] = list[i].bound->count;
      memset(list[i].bound->pieces, 0, counts[i] * sizeof *list[i].bound->pieces);
    }
  for (uint32_t h = 0; h < s->count && !result; h++) {
    struct channel *ch = s->channels + h;
    struct schedule *p = plans + h;
    *p = (struct schedule){calloc(total, sizeof *p->out), calloc(total, sizeof *p->in), 0, 0, 0, 0};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    size_t count;
    const uint64_t positions = lay(s, list, n, steps, counts, rank_of(comm, ch->node), ch->large, ch->slot, made, &count);
    for (size_t k = 0; k < count; k++) {
      struct call *c = list + made[k].call;
      const struct mesh_step *step = c->bound->steps + made[k].step;
      const struct message m = {c, made[k].step, ch->end + made[k].first, made[k].pieces, step->first * step->piece.element_bytes, piece_bytes(step)};
      if (made[k].send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
      if (getenv("MESH_CHECK_TRACE"))
        fprintf(stderr, "schedule channel %u (node %u) %s positions %llu..%llu step %u\n", h, ch->node, made[k].send ? "send" : "receive",
                (unsigned long long)m.first, (unsigned long long)(m.first + m.pieces), m.step);
      c->bound->channel[m.step] = h; c->bound->first[m.step] = m.first; c->bound->pieces[m.step] = m.pieces;
    }
    ch->end += positions;
  }
  for (size_t i = 0; i < n && !result; i++)
    for (uint32_t k = 0; list[i].bound && k < list[i].bound->count; k++) {
      struct steps *st = list[i].bound;
      st->before[k] = before_of(st->steps, st->count, st->pieces, k);
      if (st->steps[k].op != MESH_STEP_SEND && !st->pieces[k]) st->done[k] = 1;
    }
  free(steps); free(counts); free(made);
  return result;
}

static void schedules_free(const struct session *s, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < s->count; h++) { free(plans[h].out); free(plans[h].in); }
  free(plans);
}

/* a step may run once every receive before it (by round, then plan order) has been consumed */
static int ready(const struct steps *s, uint32_t k) {
  for (uint32_t j = 0; j < s->count; j++)
    if (s->steps[j].op != MESH_STEP_SEND && !s->done[j] &&
        (s->steps[j].round < s->steps[k].round || (s->steps[j].round == s->steps[k].round && j < k)))
      return 0;
  return 1;
}

/* every channel's positions published and consumed: a SEND's piece copied into its slot, a REDUCE's
   combined into the result, a COPY's copied into the operand (and the result, where they differ: a direct
   exchange's sums go to a copy, as its SENDs read the operand); the progress thread watches the landings */
static ncclResult_t run(struct session *ses, struct schedule *plans) {
  for (;;) {
    if (atomic_load(&ses->failed)) return fail((ncclResult_t)atomic_load(&ses->failed), "%s", ses->failure);
    int open = 0;
    for (uint32_t h = 0; h < ses->count; h++) {
      struct channel *ch = ses->channels + h;
      struct schedule *p = plans + h;
      while (ch->consumed < ch->end) {
        const uint64_t at = ch->consumed, arrived = mesh_host_arrived(&ses->context, ch->in, (uint32_t)at);
        if (arrived == UINT64_MAX) return fail(ncclRemoteError, "the link to node %u was lost (its landings cancelled)", ch->node);
        if (!landed(ses, arrived, at)) break;
        struct message *m = p->in_at < p->ins ? p->in + p->in_at : NULL;
        const int data = m && at >= m->first;
        uint64_t seen;
        memcpy(&seen, ch->receiving + (at % DEPTH) * ch->slot, sizeof seen);
        if (data == (seen == metal_stamp(at)))
          return fail(ncclInvalidUsage, MISMATCH, (unsigned long long)at, ch->node, data ? "a piece" : "nothing", data ? "nothing" : "a piece");
        if (at + 1 > ch->checked) ch->checked = at + 1;
        if (data) {
          struct steps *s = m->c->bound;
          if (at == m->first && !ready(s, m->step)) break;
          const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
          const unsigned char *from = ch->receiving + (at % DEPTH) * ch->slot;
          unsigned char *to = s->host[s->steps[m->step].buffer] + m->offset + offset;
          if (s->steps[m->step].op == MESH_STEP_REDUCE) combine_into(m->c->type, m->c->combine, to, from, length / SIZE[m->c->type]);
          else memcpy(to, from, length);
          if (at + 1 == m->first + m->pieces) { s->done[m->step] = 1; p->in_at++; }
        }
        ch->consumed++;
      }
      while (ch->sent < ch->end && ch->sent < ch->consumed + LAG) {
        const uint64_t at = ch->sent;
        struct message *m = p->out_at < p->outs ? p->out + p->out_at : NULL;
        if (m && at >= m->first) {
          if (at == m->first && !ready(m->c->bound, m->step)) break;
          const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
          memcpy(ch->sending + (at % DEPTH) * ch->slot, m->c->bound->host[m->c->bound->steps[m->step].buffer] + m->offset + offset, length);
          if (at + 1 == m->first + m->pieces) p->out_at++;
        } else {
          const uint64_t stamp = metal_stamp(at);
          memcpy(ch->sending + (at % DEPTH) * ch->slot, &stamp, sizeof stamp);
        }
        mesh_host_publish(&ses->context, ch->out, (uint32_t)at);
        ch->sent++;
      }
      open |= ch->sent < ch->end || ch->consumed < ch->end;
    }
    if (!open) return ncclSuccess;
  }
}

/* the communicator's session, opened where it has none on the map current now; a failure its progress thread kept fails
   this group and closes the session, and the caller's next group opens another (no group is tried again here) */
static struct inflight *inflight;
static ncclResult_t session_ensure(ncclComm_t comm) {
  struct session *s = comm->session;
  if (s) violated(s);
  if (s && atomic_load(&s->failed)) {
    const ncclResult_t result = fail((ncclResult_t)atomic_load(&s->failed), "%s", s->failure);
    session_close(s, 0);
    return result;
  }
  return s ? ncclSuccess : session_open(comm);
}

static ncclResult_t session_schedule(struct session *s, struct call *list, size_t n, struct schedule **made) {
  struct schedule *plans = calloc(s->count + 1, sizeof *plans);
  ncclResult_t result;
  *made = NULL;
  if (!plans) return fail(ncclSystemError, "out of memory");
  if ((result = schedule_group(s, list, n, plans))) { schedules_free(s, plans); return result; }
  *made = plans;
  return ncclSuccess;
}

/* the group on the host, the progress thread watching its landings; its positions then count as landed */
static ncclResult_t session_run(struct session *s, struct call *list, size_t n) {
  struct schedule *plans;
  ncclResult_t result = session_schedule(s, list, n, &plans);
  if (result) return result;
  progress_targets(s);
  result = run(s, plans);
  schedules_free(s, plans);
  if (result) { session_close(s, 0); return result; }
  for (uint32_t h = 0; h < s->count; h++) {
    struct channel *ch = s->channels + h;
    atomic_store(&ch->target, ch->end); atomic_store(&ch->signaled, ch->end);
    ch->waited = ch->end;
  }
  return ncclSuccess;
}

/* _stage: the call's contribution into its operand */
static void stage(const struct call *c, unsigned char *slot) {
  const size_t z = SIZE[c->type], n = c->count, r = (size_t)c->comm->rank;
  if (reducing(c->what)) {
    const size_t all = elements(c);
    if (c->premultiplied) premultiply(c->type, slot, c->send, c->pre, all);
    else memmove(slot, c->send, all * z);
  } else if (c->what == MESH_BROADCAST && c->root == c->comm->rank) memmove(slot, c->send, n * z);
  else if (c->what == MESH_ALLGATHER) memmove(slot + r * n * z, c->send, n * z);
}

/* _finish: the result into recvbuff */
static void finish(const struct call *c) {
  const size_t z = SIZE[c->type], n = c->count, r = (size_t)c->comm->rank;
  const int nranks = c->comm->nranks;
  if (!c->result) return;
  if (c->what == MESH_ALLREDUCE || (c->what == MESH_REDUCE && c->root == c->comm->rank)) {
    if (c->post) truncdiv(c->type, c->recv, c->result, n, (uint64_t)nranks);
    else memmove(c->recv, c->result, n * z);
  } else if (c->what == MESH_REDUCE_SCATTER) {
    if (c->post) truncdiv(c->type, c->recv, c->result + r * n * z, n, (uint64_t)nranks);
    else memmove(c->recv, c->result + r * n * z, n * z);
  } else if (c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER || c->what == WHAT_RECV)
    memmove(c->recv, c->result, elements(c) * z);
}

/* _reduction: an operator as the host runs it (premultiplier, combine, postoperation) */
static ncclResult_t reduction(struct call *c) {
  const ncclComm_t comm = c->comm;
  c->premultiplied = 0; c->post = 0;
  if ((int)c->op < ncclAvg) { c->combine = (int)c->op; return ncclSuccess; }
  c->combine = COMBINE_SUM;
  if (c->op == ncclAvg) {
    if (!floating(c->type)) c->post = 1;
    else { store(c->type, c->pre, 1.0 / comm->nranks); c->premultiplied = 1; }
    return ncclSuccess;
  }
  const struct premul *made = comm->ops + ((int)c->op - ncclNumOps);
  if (made->type != c->type) return fail(ncclInvalidArgument, "a PreMulSum operator reduces the datatype it was made for");
  memcpy(c->pre, made->value, SIZE[c->type]);
  c->premultiplied = 1;
  return ncclSuccess;
}

/* _launch: one group: the communicator's session ensured (its programs on the current map), its calls planned and
   staged (a SEND read and a receive written in place), run on the session after every GPU program, finished */
static ncclResult_t launch(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  for (size_t i = 0; i < n; i++) {
    if (list[i].comm != comm) return fail(ncclInvalidUsage, "a group on several communicators");
    if (list[i].stream) return fail(ncclInvalidUsage, "a group on several streams");
    if (reducing(list[i].what) && (result = reduction(list + i))) return result;
    remote |= crosses(list + i);
  }
  pthread_mutex_lock(&session_lock);
  if (remote && inflight) result = fail(ncclInvalidUsage, "a host-path group while GPU groups are issued and not complete (ncclMeshComplete)");
  if (!result && remote) result = drain(comm->session);
  if (!result && remote) result = session_ensure(comm);
  if (!result) result = plan_calls(comm, list, n);
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *s = c->bound;
    const size_t bytes = elements(c) * SIZE[c->type];
    if (c->what == WHAT_SEND && c->peer == comm->rank) {
      if (!(c->copy = malloc(bytes ? bytes : 1))) result = fail(ncclSystemError, "out of memory");
      else memcpy(c->copy, c->send, bytes);
    } else if (c->what == WHAT_SEND) { if (s) s->host[0] = (unsigned char *)(uintptr_t)c->send; }
    else if (c->what == WHAT_RECV) { if (s) s->host[0] = c->recv; }
    else if (!(c->result = calloc(bytes ? bytes : 1, 1))) result = fail(ncclSystemError, "out of memory");
    else {
      c->owned = 1;
      stage(c, c->result);
      if (s) s->host[0] = c->result;
      for (uint32_t b = 1; s && b < s->buffers && !result; b++)
        if (!(s->host[b] = malloc(bytes ? bytes : 1))) result = fail(ncclSystemError, "out of memory");
        else memcpy(s->host[b], c->result, bytes);
    }
  }
  if (!result && remote) result = session_run(comm->session, list, n);
  pthread_mutex_unlock(&session_lock);
  if (!result) {
    size_t self = 0;
    for (size_t i = 0; i < n; i++) {
      struct call *c = list + i;
      if (c->what == WHAT_RECV && c->peer == comm->rank) {
        while (self < n && !(list[self].what == WHAT_SEND && list[self].peer == comm->rank)) self++;
        if (self < n) { c->result = list[self].copy; self++; }
      }
      finish(c);
    }
  }
  return result;
}

/* -- the Metal path ------------------------------------------------------------------------------------ */

static ncclResult_t gpu(int status, const char *what) {
  return status ? fail(ncclSystemError, "%s on the GPU: %s", what, *metal_error() ? metal_error() : strerror(status)) : ncclSuccess;
}

static int receives_at(struct session *s, uint32_t h, uint64_t at);

/* the positions of channel `ch` up to `count` not yet checked, each slot against its role once a wait covers it (its
   slot stays until this rank publishes the position LAG past it, which waits on the position after it first) */
static struct metal_check check_of(struct session *s, struct channel *ch, uint64_t count) {
  const uint32_t h = (uint32_t)(ch - s->channels);
  uint64_t first = ch->checked < count ? ch->checked : count;
  if (count - first > DEPTH) first = count - DEPTH;
  struct metal_check check = {ch->ring_in, s->violation_buffer, ch->slot, first, (uint32_t)(count - first), 0, h, DEPTH};
  for (uint64_t t = first; t < count; t++)
    if (receives_at(s, h, t)) check.mask |= 1u << (t - first);
  if (count > ch->checked) ch->checked = count;
  if (getenv("MESH_CHECK_TRACE"))
    fprintf(stderr, "check channel %u (node %u, %s) positions %llu..%llu mask %x\n", h, ch->node, ch->large ? "large" : "small",
            (unsigned long long)first, (unsigned long long)count, check.mask);
  return check;
}

/* the GPU waits until `count` of the channel's positions have landed: a spin on the last one's completion word */
static ncclResult_t arrive(struct metal_program *program, struct session *s, struct channel *ch, uint64_t count) {
  if (count <= ch->waited) return ncclSuccess;
  ch->waited = count;
  const uint64_t at = count - 1;
  const struct metal_check check = check_of(s, ch, count);
  return gpu(metal_spin(program, s->transport.inputs, ch->word + (at % s->positions) * ch->word_stride, cycle_of(s, at), &check), "a spin");
}

/* one of this rank's publications in its group's order (order_group): its level, channel and position */
struct event { uint64_t level; uint32_t h; uint64_t at; };

/* a group issued and not complete: its session, its calls, its schedule and its positions [start, end) on each of the
   session's channels, its publications in order (`next` the first the walk has not completed); the groups in flight
   are in issue order, a session's holding consecutive positions on every channel of it */
struct inflight {
  uint64_t ticket;
  struct session *session;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t *start, *end;
  struct event *order;
  size_t events, next;
  struct inflight *next_group;
};
static struct inflight *inflight_last;
static uint64_t tickets;
static _Atomic uint64_t retired;
/* the groups lost with their sessions, each with its session's failure, until a completion through it reports it */
static struct loss { uint64_t ticket; ncclResult_t result; char reason[256]; } *losses;
static size_t lost;

static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    steps_free(list[i].bound);
    if (list[i].owned) free(list[i].result);
    free(list[i].copy);
  }
  free(list);
}

static void inflight_free(struct inflight *g) {
  schedules_free(g->session, g->plans);
  free(g->start);
  free(g->order);
  calls_free(g->list, g->n);
  free(g);
}

/* every group in flight on session s lost with it: a later completion through one returns the session's failure (or
   its closing); every earlier group is complete or lost */
static void inflight_lose(struct session *s) {
  const ncclResult_t failed = (ncclResult_t)atomic_load(&s->failed);
  inflight_last = NULL;
  for (struct inflight **at = &inflight; *at;) {
    struct inflight *g = *at;
    if (g->session != s) { inflight_last = g; at = &g->next_group; continue; }
    struct loss *grown = realloc(losses, (lost + 1) * sizeof *grown);
    if (grown) {
      losses = grown;
      losses[lost] = (struct loss){g->ticket, failed ? failed : ncclRemoteError, {0}};
      snprintf(losses[lost++].reason, sizeof grown->reason, "%s", failed ? s->failure : "its communicator's session closed");
    }
    *at = g->next_group;
    inflight_free(g);
  }
  atomic_store(&retired, inflight ? inflight->ticket - 1 : tickets);
}

static struct inflight *inflight_at(struct session *s, uint32_t h, uint64_t at) {
  for (struct inflight *g = inflight; g; g = g->next_group)
    if (g->session == s && at < g->end[h]) return at >= g->start[h] ? g : NULL;
  return NULL;
}

/* whether this rank receives a piece at position `at` of channel h: a position of no group in flight is an empty one
   (each group's receives are awaited, and checked, before it completes) */
static int receives_at(struct session *s, uint32_t h, uint64_t at) {
  const struct inflight *g = inflight_at(s, h, at);
  if (!g) return 0;
  const struct schedule *p = g->plans + h;
  for (size_t k = 0; k < p->ins; k++)
    if (at >= p->in[k].first && at < p->in[k].first + p->in[k].pieces) return 1;
  return 0;
}

static int event_order(const void *x, const void *y) {
  const struct event *a = x, *b = y;
  if (a->level != b->level) return a->level < b->level ? -1 : 1;
  if (a->h != b->h) return a->h < b->h ? -1 : 1;
  return (a->at > b->at) - (a->at < b->at);
}

/* the group's publications in the one order every rank's walk keeps (MSCCLang's: a topological order of the dependency
   graph over every rank, so no rank's wait is on what another encodes after a wait of its own).  Every rank's plan of
   each collective laid out on every channel (lay), each end's publication of each position and its consumption are
   the nodes: a publication after its channel's previous one, the peer's publication LAG positions back (whose landing
   frees the slot it fills) and its own consumption of that position, a SEND's first piece after the consumption of the
   receive before it; a consumption after its channel's previous one, the peer's publication where a piece lands, a
   receive's first piece after the receive before it.  A publication's level is one more than its predecessors'
   largest, a consumption's that largest.  This rank's publications go to g->order by level; the walk encodes a
   consumption only where a publication (or the group's end) needs it, so a wait is on a publication of a level below
   the next one encoded, and the lowest level not yet published always can be */
static ncclResult_t order_group(ncclComm_t comm, struct inflight *g) {
  const struct session *s = g->session;
  const int N = comm->nranks, me = comm->rank;
  struct call *list = g->list;
  size_t S = 2;
  for (size_t i = 0; i < g->n; i++)
    if (list[i].bound && list[i].bound->program && list[i].bound->program->first[N] + 1 > S) S = list[i].bound->program->first[N] + 1;
  const size_t n = g->n, E = (size_t)N * (N - 1);
  struct mesh_step **steps = calloc((size_t)N * n + 1, sizeof *steps), *planned = calloc((size_t)N * n * S + 1, sizeof *planned);
  uint32_t *counts = calloc((size_t)N * n + 1, sizeof *counts), *lane = calloc((size_t)N * n * S + 1, sizeof *lane);
  uint64_t *first = calloc((size_t)N * n * S + 1, sizeof *first), *pieces = calloc((size_t)N * n * S + 1, sizeof *pieces);
  uint64_t *length = calloc(2 * E + 1, sizeof *length), *offset = calloc(E + 1, sizeof *offset);
  struct placing *made = calloc(n * S + 1, sizeof *made);
  ncclResult_t result = steps && planned && counts && lane && first && pieces && length && offset && made ? ncclSuccess : fail(ncclSystemError, "out of memory");
  /* a channel: the pair a < b (its index among the pairs) and the class; an end's side 0 for a, 1 for b */
#define CHANNEL(a, b, large) ((((size_t)(a) * (2 * N - (a) - 1) / 2 + ((b) - (a) - 1))) * 2 + (size_t)(large))
#define AT(r, i, k) (((size_t)(r) * n + (i)) * S + (k))
  for (int r = 0; r < N && !result; r++)
    for (size_t i = 0; i < n && !result; i++) {
      const struct steps *own = list[i].bound;
      if (!own) continue;
      if (r == me) { steps[r * n + i] = own->steps; counts[r * n + i] = own->count; }
      else if (!p2p(list + i)) {
        steps[r * n + i] = planned + AT(r, i, 0);
        counts[r * n + i] = mesh_program_steps(own->program, (uint32_t)r, own->operand, steps[r * n + i]);
      }
    }
  for (int r = 0; r < N && !result; r++)
    for (int q = 0; q < N; q++)
      for (int large = 0; q != r && large < 2; large++) {
        size_t count;
        const size_t e = r < q ? CHANNEL(r, q, large) : CHANNEL(q, r, large);
        length[2 * e + (r > q)] = lay(s, list, n, steps + (size_t)r * n, counts + (size_t)r * n, q, large, large ? s->slot : s->small, made, &count);
        for (size_t m = 0; m < count; m++) {
          lane[AT(r, made[m].call, made[m].step)] = (uint32_t)e;
          first[AT(r, made[m].call, made[m].step)] = made[m].first;
          pieces[AT(r, made[m].call, made[m].step)] = made[m].pieces;
        }
      }
  size_t T = 0;
  for (size_t e = 0; e < E && !result; e++) {
    offset[e] = T;
    T += length[2 * e] > length[2 * e + 1] ? length[2 * e] : length[2 * e + 1];
  }
  /* per end and position: the consumption a publication's first piece and a receive's first piece wait on (node + 1, 0
     none), whether a piece leaves (sends) and arrives (lands) there */
  uint64_t *after_send = calloc(2 * T + 1, sizeof *after_send), *after_receive = calloc(2 * T + 1, sizeof *after_receive);
  unsigned char *sends = calloc(2 * T + 1, 1), *lands = calloc(2 * T + 1, 1);
  if (!result && (!after_send || !after_receive || !sends || !lands)) result = fail(ncclSystemError, "out of memory");
#define NODE(e, t, kind) (4 * (offset[e] + (t)) + (kind))
  for (int r = 0; r < N && !result; r++)
    for (size_t i = 0; i < n; i++)
      for (uint32_t k = 0; k < counts[r * n + i]; k++) {
        if (!pieces[AT(r, i, k)]) continue;
        const size_t e = lane[AT(r, i, k)], side = (size_t)(r > (int)steps[r * n + i][k].peer), at = offset[e] + first[AT(r, i, k)];
        const int send = steps[r * n + i][k].op == MESH_STEP_SEND;
        uint64_t *pieces_of = pieces + AT(r, i, 0);
        const int32_t j = before_of(steps[r * n + i], counts[r * n + i], pieces_of, k);
        uint64_t wait = 0;
        if (j >= 0) {
          const size_t ej = lane[AT(r, i, j)], sj = (size_t)(r > (int)steps[r * n + i][j].peer);
          wait = 1 + 4 * (offset[ej] + first[AT(r, i, j)] + pieces[AT(r, i, j)] - 1) + 2 + sj;
        }
        (send ? after_send : after_receive)[2 * at + side] = wait;
        for (uint64_t t = 0; t < pieces[AT(r, i, k)]; t++) (send ? sends : lands)[2 * (at + t) + side] = 1;
      }
  /* the nodes, their predecessors (at most four), Kahn's order */
  uint64_t *pred = calloc(16 * T + 1, sizeof *pred), *level = calloc(4 * T + 1, sizeof *level), *queue = calloc(4 * T + 1, sizeof *queue);
  uint32_t *degree = calloc(4 * T + 1, sizeof *degree), *successors = calloc(4 * T + 2, sizeof *successors);
  uint64_t *edge = calloc(16 * T + 1, sizeof *edge);
  if (!result && (!pred || !level || !queue || !degree || !successors || !edge)) result = fail(ncclSystemError, "out of memory");
  for (size_t e = 0; e < E && !result; e++) {
    const uint64_t positions = (e + 1 < E ? offset[e + 1] : T) - offset[e];
    for (uint64_t t = 0; t < positions; t++)
      for (size_t side = 0; side < 2; side++) {
        const size_t o = offset[e] + t;
        uint64_t *pp = pred + 4 * NODE(e, t, side), *pc = pred + 4 * NODE(e, t, 2 + side);
        if (t) { pp[0] = 1 + NODE(e, t - 1, side); pc[0] = 1 + NODE(e, t - 1, 2 + side); }
        if (t >= LAG) { pp[1] = 1 + NODE(e, t - LAG, 1 - side); pp[2] = 1 + NODE(e, t - LAG, 2 + side); }
        pp[3] = after_send[2 * o + side];
        if (lands[2 * o + side] || sends[2 * o + 1 - side]) pc[1] = 1 + NODE(e, t, 1 - side);
        pc[2] = after_receive[2 * o + side];
      }
  }
  for (uint64_t v = 0; v < 4 * T && !result; v++)
    for (int k = 0; k < 4; k++)
      if (pred[4 * v + k]) { degree[v]++; successors[pred[4 * v + k] - 1 + 1]++; }
  for (uint64_t v = 0; v < 4 * T && !result; v++) successors[v + 1] += successors[v];
  for (uint64_t v = 0; v < 4 * T && !result; v++)
    for (int k = 0; k < 4; k++)
      if (pred[4 * v + k]) edge[successors[pred[4 * v + k] - 1]++] = v;
  for (uint64_t v = 4 * T; v > 0 && !result; v--) successors[v] = successors[v - 1];
  if (!result) successors[0] = 0;
  size_t head = 0, tail = 0;
  for (uint64_t v = 0; v < 4 * T && !result; v++)
    if (!degree[v]) queue[tail++] = v;
  while (head < tail && !result) {
    const uint64_t v = queue[head++];
    uint64_t most = 0;
    for (int k = 0; k < 4; k++)
      if (pred[4 * v + k] && level[pred[4 * v + k] - 1] > most) most = level[pred[4 * v + k] - 1];
    level[v] = most + ((v & 3) < 2);
    for (uint32_t x = successors[v]; x < successors[v + 1]; x++)
      if (!--degree[edge[x]]) queue[tail++] = edge[x];
  }
  if (!result && tail != 4 * T) result = fail(ncclInternalError, "the group's positions admit no order (%zu of %llu nodes)", tail, (unsigned long long)(4 * T));
  size_t events = 0;
  for (uint32_t h = 0; h < s->count; h++) events += g->end[h] - g->start[h];
  if (!result && !(g->order = calloc(events + 1, sizeof *g->order))) result = fail(ncclSystemError, "out of memory");
  for (uint32_t h = 0; h < s->count && !result; h++) {
    const struct channel *ch = s->channels + h;
    const int q = rank_of(comm, ch->node);
    if (g->end[h] == g->start[h]) continue;
    if (q < 0) { result = fail(ncclInternalError, "channel %u has positions and no rank", h); break; }
    const size_t e = me < q ? CHANNEL(me, q, ch->large) : CHANNEL(q, me, ch->large), side = (size_t)(me > q);
    if (length[2 * e + side] != g->end[h] - g->start[h]) { result = fail(ncclInternalError, "the group's positions on channel %u disagree", h); break; }
    for (uint64_t t = 0; t < g->end[h] - g->start[h]; t++) g->order[g->events++] = (struct event){level[NODE(e, t, side)], h, g->start[h] + t};
  }
#undef NODE
#undef AT
#undef CHANNEL
  if (!result) qsort(g->order, g->events, sizeof *g->order, event_order);
  free(steps); free(planned); free(counts); free(lane); free(first); free(pieces); free(length); free(offset); free(made);
  free(after_send); free(after_receive); free(sends); free(lands); free(pred); free(level); free(queue); free(degree); free(successors); free(edge);
  return result;
}

/* channel h's positions consumed up to `end`, in order: a piece's landing awaited and combined or copied where it
   belongs (an empty position needs nothing), a receive's first piece after the receive before it */
static ncclResult_t consume(struct metal_program *program, struct session *s, uint32_t h, uint64_t end) {
  struct channel *ch = s->channels + h;
  ncclResult_t result = ncclSuccess;
  while (ch->consumed < end && !result) {
    const uint64_t at = ch->consumed;
    struct inflight *g = inflight_at(s, h, at);
    if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
    struct schedule *p = g->plans + h;
    struct message *m = p->in_at < p->ins && at >= p->in[p->in_at].first ? p->in + p->in_at : NULL;
    if (m && at == m->first && m->c->bound->before[m->step] >= 0) {
      const struct steps *prior = m->c->bound;
      const int32_t j = prior->before[m->step];
      result = consume(program, s, prior->channel[j], prior->first[j] + prior->pieces[j]);
      if (result) break;
    }
    if (m) {
      struct steps *st = m->c->bound;
      const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
      const size_t from = (at % DEPTH) * ch->slot;
      const int reduce = st->steps[m->step].op == MESH_STEP_REDUCE;
      const struct where to = st->gpu[st->steps[m->step].buffer];
      if (!ch->large) {
        const struct metal_check check = check_of(s, ch, at + 1);
        result = gpu(metal_land(program, reduce ? (int)m->c->type : -1, m->c->combine, to.buffer, to.offset + m->offset + offset, ch->ring_in, from,
                                reduce ? length / SIZE[m->c->type] : length, s->transport.inputs,
                                ch->word + (at % s->positions) * ch->word_stride, cycle_of(s, at), &check), "a landing");
        if (at + 1 > ch->waited) ch->waited = at + 1;
      } else {
        result = arrive(program, s, ch, at + 1);
        if (!result && reduce)
          result = gpu(metal_combine(program, m->c->type, m->c->combine, to.buffer, to.offset + m->offset + offset,
                                     ch->ring_in, from, length / SIZE[m->c->type]), "a combine");
        else if (!result)
          result = gpu(metal_copy(program, METAL_LAND, to.buffer, to.offset + m->offset + offset, ch->ring_in, from, length), "a landing");
      }
      if (result) break;
      if (at + 1 == m->first + m->pieces) { st->done[m->step] = 1; p->in_at++; }
    }
    ch->consumed++;
  }
  return result;
}

/* channel h's next position published, after the landing of the peer's position LAG back, whose slot it fills: its
   piece copied into its slot and its cell released, or the cell alone (an empty position) */
static ncclResult_t publish(struct metal_program *program, struct session *s, uint32_t h, int *published) {
  struct channel *ch = s->channels + h;
  const uint64_t at = ch->sent, need = at >= LAG ? at - LAG + 1 : 0;
  struct inflight *g = inflight_at(s, h, at);
  if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
  struct schedule *p = g->plans + h;
  struct message *m = p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
  ncclResult_t result = arrive(program, s, ch, need);
  const size_t cell = ch->cell + (at % s->positions) * sizeof(struct mesh_send);
  const uint64_t value = ch->argument + cycle_of(s, at);
  int released = 0;
  if (m && !result) {
    const struct steps *st = m->c->bound;
    const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
    if (!ch->large) {
      const struct where source = st->gpu[st->steps[m->step].buffer];
      result = gpu(metal_send_small(program, ch->ring_out, (at % DEPTH) * ch->slot, source.buffer, source.offset + m->offset + offset,
                                    length, s->transport.publication, cell, value), "a piece into its slot and its release");
      released = 1;
    } else
      result = gpu(metal_copy(program, METAL_SEND, ch->ring_out, (at % DEPTH) * ch->slot, st->gpu[st->steps[m->step].buffer].buffer,
                              st->gpu[st->steps[m->step].buffer].offset + m->offset + offset, length), "a piece into its slot");
    if (at + 1 == m->first + m->pieces) p->out_at++;
  }
  if (!result && !released && m) result = gpu(metal_publish(program, s->transport.publication, cell, value), "a publication");
  else if (!result && !released)
    result = gpu(metal_publish_filler(program, s->transport.publication, cell, value, ch->ring_out, (at % DEPTH) * ch->slot, at),
                 "an empty publication");
  if (!result) { ch->sent++; *published = 1; }
  return result;
}

static struct message *sending(struct inflight *g, uint32_t h, uint64_t at) {
  struct schedule *p = g->plans + h;
  return p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
}

/* a group completed: its publications in its order, each after the consumptions it needs (the positions LAG back on
   its channel; a SEND's first piece, the receive before it), then every position of it consumed */
static ncclResult_t complete_group(struct metal_program *program, struct inflight *g, int *published) {
  struct session *s = g->session;
  ncclResult_t result = ncclSuccess;
  for (; g->next < g->events && !result; g->next++) {
    const struct event v = g->order[g->next];
    const struct channel *ch = s->channels + v.h;
    if (v.at < ch->sent) continue;
    if (v.at != ch->sent) return fail(ncclInternalError, "the group's positions admit no order");
    result = consume(program, s, v.h, v.at >= LAG ? v.at - LAG + 1 : 0);
    const struct message *m = result ? NULL : sending(g, v.h, v.at);
    if (m && v.at == m->first && m->c->bound->before[m->step] >= 0) {
      const struct steps *prior = m->c->bound;
      const int32_t j = prior->before[m->step];
      result = consume(program, s, prior->channel[j], prior->first[j] + prior->pieces[j]);
    }
    if (!result) result = publish(program, s, v.h, published);
  }
  for (uint32_t h = 0; h < s->count && !result; h++) result = consume(program, s, h, g->end[h]);
  return result;
}

/* a group's publications that need no wait, in its order: its channel's earlier positions published, the positions LAG
   back consumed (empty ones freely) and the peer's landing there awaited before, a SEND's receives before it consumed */
static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  struct session *s = g->session;
  ncclResult_t result = ncclSuccess;
  for (size_t e = g->next; e < g->events && !result; e++) {
    const struct event v = g->order[e];
    struct channel *ch = s->channels + v.h;
    const uint64_t need = v.at >= LAG ? v.at - LAG + 1 : 0;
    if (v.at != ch->sent || need > ch->waited) continue;
    while (ch->consumed < need) {
      struct inflight *at = inflight_at(s, v.h, ch->consumed);
      struct schedule *p = at ? at->plans + v.h : NULL;
      if (!p || (p->in_at < p->ins && ch->consumed >= p->in[p->in_at].first)) break;
      ch->consumed++;
    }
    const struct message *m = sending(g, v.h, v.at);
    if (ch->consumed < need || (m && v.at == m->first && !ready(m->c->bound, m->step))) continue;
    result = publish(program, s, v.h, published);
  }
  return result;
}

/* the groups in flight, in the order every rank's walk keeps (order_group): completing through a group, each group up
   to it completed, then any later group's publications that need no wait; issuing a group, its publications that need
   no wait, so a group issued encodes no wait at all */
static ncclResult_t walk(struct metal_program *program, struct inflight *issuing, struct inflight *through, int *published) {
  ncclResult_t result = ncclSuccess;
  struct inflight *g = issuing;
  if (through) {
    for (g = inflight; g && !result; g = g->next_group) {
      result = complete_group(program, g, published);
      if (g == through) break;
    }
    g = g ? g->next_group : NULL;
  }
  for (; g && !result; g = g->next_group) result = issue_group(program, g, published);
  return result;
}

static int on_gpu(ncclDataType_t t) { return t != ncclFloat64; }
static int same(struct where a, struct where b) { return a.buffer == b.buffer && a.offset == b.offset; }
static struct where where_of(const void *argument) {
  const ncclMeshBuffer *b = argument;
  return b ? (struct where){b->buffer, b->offset} : (struct where){NULL, 0};
}

/* a group's scratch: the stream's workspace (the caller's allocator, ordered with its work), else a buffer the
   program keeps until its end */
static void *workspace(struct metal_program *program, const struct call *c, size_t bytes) {
  if (c->workspace) return c->workspace(bytes ? bytes : 16, c->context);
  void *made = metal_scratch(bytes);
  if (made) { metal_keep(program, made); metal_release(made); }
  return made;
}

/* a premultiplier as the kernels read it: a floating type's value as float32 bits, an integer's bits */
static uint64_t scalar_bits(const struct call *c) {
  uint64_t bits = 0;
  if (floating(c->type)) { const float f = (float)load(c->type, c->pre); uint32_t u; memcpy(&u, &f, 4); return u; }
  memcpy(&bits, c->pre, SIZE[c->type]);
  return bits;
}

static void plans_record(const struct call *list, size_t n) {
  free(plan_steps); free(plan_root);
  plan_steps = calloc(n + 1, sizeof *plan_steps); plan_root = calloc(n + 1, sizeof *plan_root);
  plan_count = plan_steps && plan_root ? n : 0;
  for (size_t i = 0; i < plan_count; i++) {
    plan_steps[i] = list[i].bound ? (int)list[i].bound->count : -1;
    plan_root[i] = list[i].bound ? list[i].bound->root : 0;
  }
}

/* a group's results into recvbuff, once its positions are consumed (a self-addressed SEND's copy among them) */
static ncclResult_t epilogue(struct metal_program *program, const struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  const size_t r = (size_t)comm->rank;
  ncclResult_t result = ncclSuccess;
  size_t self = 0;
  for (size_t i = 0; i < n && !result; i++) {
    const struct call *c = list + i;
    const size_t z = SIZE[c->type], count = c->count;
    const struct where recv = c->recv_at, sum = c->sum;
    if (c->what == WHAT_RECV && c->peer == comm->rank) {
      while (self < n && !(list[self].what == WHAT_SEND && list[self].peer == comm->rank)) self++;
      if (self < n) {
        const struct where from = list[self].send_at;
        result = gpu(metal_copy(program, METAL_PLAIN, recv.buffer, recv.offset, from.buffer, from.offset, count * z), "a copy to itself");
        self++;
      }
    } else if (p2p(c)) continue;
    else if (c->what == MESH_ALLREDUCE || (c->what == MESH_REDUCE && c->root == comm->rank) || c->what == MESH_REDUCE_SCATTER) {
      const struct where from = {sum.buffer, sum.offset + (c->what == MESH_REDUCE_SCATTER ? r * count * z : 0)};
      if (c->post) result = gpu(metal_truncdiv(program, c->type, recv.buffer, recv.offset, from.buffer, from.offset, count, (uint64_t)comm->nranks), "the result");
      else if (!same(from, recv)) result = gpu(metal_copy(program, METAL_PLAIN, recv.buffer, recv.offset, from.buffer, from.offset, count * z), "the result");
    } else if ((c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER) && !same(sum, recv))
      result = gpu(metal_copy(program, METAL_PLAIN, recv.buffer, recv.offset, sum.buffer, sum.offset, elements(c) * z), "the result");
  }
  return result;
}

/* the groups in flight whose positions are all consumed, in issue order: their results, and they are complete */
static ncclResult_t retire(struct metal_program *program) {
  ncclResult_t result = ncclSuccess;
  while (inflight && !result) {
    struct inflight *g = inflight;
    for (uint32_t h = 0; h < g->session->count; h++)
      if (g->session->channels[h].consumed < g->end[h]) return ncclSuccess;
    result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g);
  }
  return result;
}

static void program_end(struct metal_program *program, int borrowed) {
  if (borrowed) metal_end(program, NULL, 0);
  else metal_end(program, finished ? finished : (finished = metal_event()), ++issued);
}

/* a walk's failure: session s (the walk's own group's; NULL: none) and every session with a group in flight through
   `ticket` failed with it, their groups lost */
static void walk_failed(struct session *s, uint64_t ticket, ncclResult_t result) {
  for (; s || (inflight && inflight->ticket <= ticket); s = NULL) {
    if (!s) s = inflight->session;
    release_all(s, result, "%s", last);
    inflight_lose(s);
  }
}

/* the groups through `ticket` whose sessions failed lost; the first loss through it, now reported, and every loss
   through it forgotten */
static ncclResult_t losses_through(uint64_t ticket) {
  for (struct inflight *g = inflight; g && g->ticket <= ticket;)
    if (atomic_load(&g->session->failed)) { inflight_lose(g->session); g = inflight; }
    else g = g->next_group;
  ncclResult_t result = ncclSuccess;
  size_t kept = 0;
  for (size_t i = 0; i < lost; i++)
    if (losses[i].ticket > ticket) losses[kept++] = losses[i];
    else if (!result) result = fail(losses[i].result, "the group was lost with its session: %s", losses[i].reason);
  lost = kept;
  return result;
}

/* _launch on the stream's command buffer, taking the group's calls: each collective's operand recvbuff itself where
   it holds the whole operand (an all-reduce's, a root's reduce, a broadcast's, an all-gather's), else a scratch one
   (a reduce-scatter's, a non-root's reduce); a direct exchange's sums apart; then the group in flight after the
   others, issued (with issue) or completed with every group before it; its results into recvbuff as it completes;
   the program's end signals `finished` where the library made its encoder */
static ncclResult_t launch_metal(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclMeshIssue *const issue = list[0].issue;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  if (issue) *issue = (ncclMeshIssue){0, 0};
  for (size_t i = 0; i < n && !result; i++) {
    if (list[i].comm != comm) result = fail(ncclInvalidUsage, "a group on several communicators");
    else if (list[i].command != list[0].command) result = fail(ncclInvalidUsage, "a group on several command buffers");
    else if (!on_gpu(list[i].type)) result = fail(ncclInvalidArgument, "datatype %d on the Metal path (no float64 there)", (int)list[i].type);
    else if (reducing(list[i].what)) result = reduction(list + i);
    remote |= crosses(list + i);
  }
  if (!result && !list[0].command) result = fail(ncclInvalidArgument, "an ncclMeshStream without a command buffer");
  if (!result && list[0].encoder && !list[0].workspace) result = fail(ncclInvalidArgument, "an ncclMeshStream with an encoder and no workspace");
  pthread_mutex_lock(&session_lock);
  if (!result && remote) result = session_ensure(comm);
  if (!result) result = plan_calls(comm, list, n);
  plans_record(list, n);
  if (result) { pthread_mutex_unlock(&session_lock); calls_free(list, n); return result; }
  if (finished) metal_collect(finished);
  struct metal_program *program = metal_begin(list[0].command, list[0].encoder);
  if (!program) { pthread_mutex_unlock(&session_lock); calls_free(list, n); return fail(ncclSystemError, "out of memory"); }
  struct session *const s = comm->session;
  const size_t r = (size_t)comm->rank;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *st = c->bound;
    const size_t z = SIZE[c->type], bytes = elements(c) * z, all = elements(c);
    const struct where send = c->send_at;
    if (p2p(c)) {
      if (st) st->gpu[0] = c->what == WHAT_SEND ? send : c->recv_at;
      continue;
    }
    const struct where recv = c->recv_at;
    const int in_place = c->what == MESH_ALLREDUCE || c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER ||
                         (c->what == MESH_REDUCE && c->root == comm->rank);
    struct where own = recv;
    if (!in_place) {
      void *scratch = workspace(program, c, bytes);
      if (!scratch) { result = fail(ncclSystemError, "out of GPU memory"); break; }
      own = (struct where){scratch, 0};
    }
    const struct where segment = {own.buffer, own.offset + (c->what == MESH_ALLGATHER ? r * c->count * z : 0)};
    if (reducing(c->what) && c->premultiplied)
      result = gpu(metal_premultiply(program, c->type, own.buffer, own.offset, send.buffer, send.offset, all, scalar_bits(c)), "the operand");
    else if ((reducing(c->what) || (c->what == MESH_BROADCAST && c->root == comm->rank) || c->what == MESH_ALLGATHER) && !same(segment, send))
      result = gpu(metal_copy(program, METAL_PLAIN, segment.buffer, segment.offset, send.buffer, send.offset,
                              reducing(c->what) ? bytes : c->count * z), "the operand");
    c->sum = own;
    if (st) st->gpu[0] = own;
    for (uint32_t b = 1; st && b < st->buffers && !result; b++) {
      void *scratch = workspace(program, c, bytes);
      if (!scratch) { result = fail(ncclSystemError, "out of GPU memory"); break; }
      st->gpu[b] = (struct where){scratch, 0};
      result = gpu(metal_copy(program, METAL_PLAIN, scratch, 0, own.buffer, own.offset, bytes), "a scratch copy");
    }
  }
  int scheduled = 0, published = 0, owned = 1;
  uint64_t ticket = 0;
  if (!result && remote) {
    struct inflight *g = calloc(1, sizeof *g);
    if (!g || !(g->start = calloc(2 * (size_t)s->count + 1, sizeof *g->start))) { free(g); result = fail(ncclSystemError, "out of memory"); }
    else {
      g->session = s;
      g->end = g->start + s->count;
      for (uint32_t h = 0; h < s->count; h++) g->start[h] = s->channels[h].end;
      result = session_schedule(s, list, n, &g->plans);
      if (result) { free(g->start); free(g); }
      else {
        for (uint32_t h = 0; h < s->count; h++) g->end[h] = s->channels[h].end;
        g->list = list; g->n = n; g->ticket = ticket = ++tickets; owned = 0;
        if (inflight_last) inflight_last->next_group = g; else inflight = g;
        inflight_last = g;
        scheduled = 1;
        result = order_group(comm, g);
        progress_targets(s);
        if (!result) result = walk(program, issue ? g : NULL, issue ? NULL : g, &published);
        if (!result) result = retire(program);
      }
    }
  } else if (!result)
    result = epilogue(program, list, n);
  program_end(program, list[0].encoder != NULL);
  if (result && scheduled) walk_failed(s, issue ? 0 : ticket, result);
  if (issue && !result) { issue->ticket = ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  pthread_mutex_unlock(&session_lock);
  if (owned) calls_free(list, n);
  if (result) comm->async = result;
  return result;
}

ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream *stream, int *published) {
  int any = 0;
  if (published) *published = 0;
  if (!stream || !stream->commandBuffer) return fail(ncclInvalidArgument, "ncclMeshComplete without a command buffer");
  pthread_mutex_lock(&session_lock);
  const ncclResult_t loss = losses_through(ticket);
  ncclResult_t result = ncclSuccess;
  if (inflight && ticket >= inflight->ticket) {
    struct inflight *through = inflight;
    while (through->next_group && through->next_group->ticket <= ticket) through = through->next_group;
    if (finished) metal_collect(finished);
    struct metal_program *program = metal_begin(stream->commandBuffer, stream->commandEncoder);
    if (!program) result = fail(ncclSystemError, "out of memory");
    else {
      result = walk(program, NULL, through, &any);
      if (!result) result = retire(program);
      program_end(program, stream->commandEncoder != NULL);
      if (result) walk_failed(NULL, ticket, result);
    }
  }
  pthread_mutex_unlock(&session_lock);
  if (published) *published = any;
  return result ? result : loss;
}

uint64_t ncclMeshRetired(void) { return atomic_load(&retired); }

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

ncclResult_t ncclGroupEnd(void) {
  if (depth <= 0) return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart");
  if (--depth) return ncclSuccess;
  struct call *list = calls;
  const size_t n = call_count;
  calls = NULL; call_count = call_capacity = 0;
  if (!n) { free(list); return ncclSuccess; }
  if (list[0].stream) return launch_metal(list, n);
  const ncclResult_t result = launch(list, n);
  plans_record(list, n);
  if (result) list[0].comm->async = result;
  calls_free(list, n);
  return result;
}

ncclResult_t ncclMeshGroupPlans(int *steps, int *roots, int capacity, int *count) {
  if (!steps || !roots || !count || capacity < 0) return fail(ncclInvalidArgument, "ncclMeshGroupPlans' arguments");
  const size_t k = plan_count < (size_t)capacity ? plan_count : (size_t)capacity;
  for (size_t i = 0; i < k; i++) { steps[i] = plan_steps[i]; roots[i] = plan_root[i]; }
  *count = (int)k;
  return ncclSuccess;
}

/* a call on the Metal path keeps its command buffer and its buffers' MTLBuffers and offsets, so neither the
   stream nor the ncclMeshBuffers need outlive the call (a lowering's are its own, gone before an enclosing
   group ends) */
static ncclResult_t enqueue(struct call c) {
  if (!c.comm) return fail(ncclInvalidArgument, "comm is NULL");
  if (c.stream) {
    const ncclMeshStream *stream = c.stream;
    c.command = stream->commandBuffer; c.encoder = stream->commandEncoder; c.workspace = stream->workspace; c.context = stream->context;
    c.issue = stream->issue;
    c.send_at = where_of(c.send); c.recv_at = where_of(c.recv);
  }
  if ((unsigned)c.type >= ncclNumTypes) return fail(ncclInvalidArgument, "datatype %d", (int)c.type);
  const ncclComm_t comm = c.comm;
  if (reducing(c.what)) {
    const int k = (int)c.op - ncclNumOps;
    if ((int)c.op < 0 || (k >= 0 && (k >= OPS || !comm->ops[k].used))) return fail(ncclInvalidArgument, "operator %d", (int)c.op);
  }
  if ((c.what == MESH_REDUCE || c.what == MESH_BROADCAST) && (c.root < 0 || c.root >= comm->nranks))
    return fail(ncclInvalidArgument, "root %d of %d ranks", c.root, comm->nranks);
  if ((c.what == WHAT_SEND || c.what == WHAT_RECV) && (c.peer < 0 || c.peer >= comm->nranks))
    return fail(ncclInvalidArgument, "peer %d of %d ranks", c.peer, comm->nranks);
  ncclGroupStart();
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) { depth--; return fail(ncclSystemError, "out of memory"); }
    calls = grown; call_capacity = capacity;
  }
  calls[call_count++] = c;
  return ncclGroupEnd();
}

static ncclResult_t collective(int what, const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                               int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  struct call c = {.what = what, .root = root, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype, .op = op, .comm = comm,
                   .stream = stream};
  (void)config;
  return enqueue(c);
}

ncclResult_t ncclAllReduceConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  return collective(MESH_ALLREDUCE, sendbuff, recvbuff, count, datatype, op, 0, comm, stream, config);
}
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  return ncclAllReduceConfig(sendbuff, recvbuff, count, datatype, op, comm, stream, NULL);
}

ncclResult_t ncclReduceConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
                              ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  return collective(MESH_REDUCE, sendbuff, recvbuff, count, datatype, op, root, comm, stream, config);
}
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  return ncclReduceConfig(sendbuff, recvbuff, count, datatype, op, root, comm, stream, NULL);
}

ncclResult_t ncclBroadcastConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  return collective(MESH_BROADCAST, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, config);
}
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcastConfig(sendbuff, recvbuff, count, datatype, root, comm, stream, NULL);
}
ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduceScatterConfig(const void *sendbuff, void *recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
                                     ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  return collective(MESH_REDUCE_SCATTER, sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, config);
}
ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  return ncclReduceScatterConfig(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, NULL);
}

ncclResult_t ncclAllGatherConfig(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  return collective(MESH_ALLGATHER, sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, config);
}
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  return ncclAllGatherConfig(sendbuff, recvbuff, sendcount, datatype, comm, stream, NULL);
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_SEND, .peer = peer, .send = sendbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_RECV, .peer = peer, .recv = recvbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}

/* a buffer `bytes` on: a host pointer's, or (on the Metal path) an ncclMeshBuffer's made in `made` */
#define MAX_RANKS 64
static const void *piece(cudaStream_t stream, const void *buffer, size_t bytes, ncclMeshBuffer *made) {
  if (!stream) return (const unsigned char *)buffer + bytes;
  const ncclMeshBuffer *b = buffer;
  *made = (ncclMeshBuffer){b->buffer, b->offset + bytes};
  return made;
}

/* NCCL 2.32's own lowering of these three (src/enqueue/task_prep/task_classify.cc
   classifyCollToP2pTasks): each rank's sends and receives in one group */
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (!comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "comm or datatype");
  const size_t z = count * SIZE[datatype];
  ncclMeshBuffer at[2 * MAX_RANKS];
  if (stream && comm->nranks > MAX_RANKS) return fail(ncclInvalidUsage, "%d ranks on the Metal path's lowering", comm->nranks);
  ncclResult_t result = ncclGroupStart();
  for (int r = 0; r < comm->nranks && !result; r++) {
    result = ncclSend(piece(stream, sendbuff, r * z, at + 2 * r), count, datatype, r, comm, stream);
    if (!result) result = ncclRecv((void *)piece(stream, recvbuff, r * z, at + 2 * r + 1), count, datatype, r, comm, stream);
  }
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}
ncclResult_t ncclAlltoAllConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  if (!comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "comm or datatype");
  const size_t z = count * SIZE[datatype];
  ncclMeshBuffer at[MAX_RANKS];
  if (stream && comm->nranks > MAX_RANKS) return fail(ncclInvalidUsage, "%d ranks on the Metal path's lowering", comm->nranks);
  ncclResult_t result = ncclGroupStart();
  if (!result) result = ncclSend(sendbuff, count, datatype, root, comm, stream);
  for (int r = 0; comm->rank == root && r < comm->nranks && !result; r++)
    result = ncclRecv((void *)piece(stream, recvbuff, r * z, at + r), count, datatype, r, comm, stream);
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}
ncclResult_t ncclGatherConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                              cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  if (!comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "comm or datatype");
  const size_t z = count * SIZE[datatype];
  ncclMeshBuffer at[MAX_RANKS];
  if (stream && comm->nranks > MAX_RANKS) return fail(ncclInvalidUsage, "%d ranks on the Metal path's lowering", comm->nranks);
  ncclResult_t result = ncclGroupStart();
  for (int r = 0; comm->rank == root && r < comm->nranks && !result; r++)
    result = ncclSend(piece(stream, sendbuff, r * z, at + r), count, datatype, r, comm, stream);
  if (!result) result = ncclRecv(recvbuff, count, datatype, root, comm, stream);
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}
ncclResult_t ncclScatterConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                               cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

/* a communicator over comm's ranks `ranks` (count of them, in that order), on the bridges' map read at each of its
   sessions, named after comm and its `split`-th split or shrink and `color`, the same on every member */
static ncclResult_t comm_sub(ncclComm_t comm, const int *ranks, int count, uint32_t split, int color, ncclComm_t *made) {
  uint32_t *members = calloc((size_t)count + 1, sizeof *members);
  char name[NCCL_UNIQUE_ID_BYTES + 1];
  int rank = 0;
  if (!members) return fail(ncclSystemError, "out of memory");
  for (int i = 0; i < count; i++) {
    members[i] = comm->members[ranks[i]];
    if (ranks[i] == comm->rank) rank = i;
  }
  snprintf(name, sizeof name, "%s/%u.%d", comm->name, split, color);
  const ncclResult_t result = comm_make(made, NULL, NULL, NULL, 0, count, members, rank, name);
  free(members);
  return result;
}

/* ncclCommSplit: every rank's (color, key) all-gathered on comm, the ranks of this color ordered by key, then rank */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  if (depth) return fail(ncclInvalidUsage, "ncclCommSplit inside a group");
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  int64_t mine[2] = {color, key}, *every = calloc(2 * (size_t)n, sizeof *every);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  ncclResult_t result = every && ranks ? ncclAllGather(mine, every, 2, ncclInt64, comm, NULL) : fail(ncclSystemError, "out of memory");
  *newcomm = NULL;
  for (int r = 0; !result && color != NCCL_SPLIT_NOCOLOR && r < n; r++)
    if (every[2 * r] == color) ranks[count++] = r;
  for (int a = 1; a < count; a++)
    for (int b = a; b > 0 && every[2 * ranks[b] + 1] < every[2 * ranks[b - 1] + 1]; b--) { const int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
  if (count) result = comm_sub(comm, ranks, count, split, color, newcomm);
  free(every); free(ranks);
  return result;
}

/* ncclCommShrink: comm's ranks but the excluded ones, in rank order, with no exchange among them (a departed member
   answers none); with NCCL_SHRINK_ABORT, comm's session closed first, its groups in flight lost */
ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t *config,
                            int shrinkFlags) {
  (void)config;
  if (!comm || !newcomm || excludeRanksCount < 0 || (excludeRanksCount && !excludeRanksList))
    return fail(ncclInvalidArgument, "ncclCommShrink's arguments");
  if (depth) return fail(ncclInvalidUsage, "ncclCommShrink inside a group");
  *newcomm = NULL;
  int *ranks = calloc((size_t)comm->nranks, sizeof *ranks), count = 0, kept = 0;
  if (!ranks) return fail(ncclSystemError, "out of memory");
  for (int r = 0; r < comm->nranks; r++) {
    int excluded = 0;
    for (int e = 0; e < excludeRanksCount; e++) excluded |= excludeRanksList[e] == r;
    if (!excluded) ranks[count++] = r;
    kept |= !excluded && r == comm->rank;
  }
  if (shrinkFlags & NCCL_SHRINK_ABORT) {
    pthread_mutex_lock(&session_lock);
    session_close(comm->session, 0);
    pthread_mutex_unlock(&session_lock);
  }
  const ncclResult_t result = kept ? comm_sub(comm, ranks, count, comm->splits++, -1, newcomm)
                                   : fail(ncclInvalidArgument, "rank %d is among the ranks it excludes", comm->rank);
  free(ranks);
  return result;
}

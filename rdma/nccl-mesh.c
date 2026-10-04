/* libnccl-mesh (nccl.h): NCCL's API on the bridges' prepared transfers.  A process holds one session on
   its bridge (one attach, one pairing, a ring of slots each way to every peer) for its life; a group's
   calls are planned by the one collective planner and streamed through the rings, the receives combined
   on the host.  The calls' mapping is metal-microbench tools/nccl_demo.py's as of dec7676 (1,724 of
   1,724 calls checked on the pair, output_data/nccl-20260928). */
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

enum { WHAT_SEND = 16, WHAT_RECV };
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN };
#define DEADLINE_S 120.0
#define LINGER_S 1.0
#define ATTEMPTS 4
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

#ifdef MESH_HOST_EXECUTOR
/* the host's arithmetic, for nccl-mesh-host.c's executor of the walk's kernels on the CPU (a premultiplier as the
   kernels take it: a floating type's value as float32 bits, an integer's bits) */
__attribute__((visibility("hidden"))) void nccl_mesh_combine(int t, int combine, void *dst, const void *src, size_t n) {
  combine_into((ncclDataType_t)t, combine, dst, src, n);
}
__attribute__((visibility("hidden"))) void nccl_mesh_premultiply(int t, void *dst, const void *src, uint64_t bits, size_t n) {
  unsigned char scalar[8] = {0};
  if (floating((ncclDataType_t)t)) { uint32_t u = (uint32_t)bits; float f; memcpy(&f, &u, 4); store((ncclDataType_t)t, scalar, f); }
  else memcpy(scalar, &bits, SIZE[t]);
  premultiply((ncclDataType_t)t, dst, src, scalar, n);
}
__attribute__((visibility("hidden"))) void nccl_mesh_truncdiv(int t, void *dst, const void *src, size_t n, uint64_t divisor) {
  truncdiv((ncclDataType_t)t, dst, src, n, divisor);
}
#endif

/* -- communicators -------------------------------------------------------------------------------- */

struct premul { int used; ncclDataType_t type; unsigned char value[8]; };
/* a communicator: its ranks' link map (mesh-plan.h; each link's cost its own), rank r on the bridge of node
   members[r], this rank's bridge region */
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; };

struct ncclComm {
  struct mesh_link_map map;
  struct mesh_trees paths;
  struct entry *entries;
  uint32_t programs;
  int nranks, rank, *members;
  char region[64];
  struct premul ops[OPS];
  ncclResult_t async;
};

static void program_free(struct mesh_program *program) {
  free((void *)program->segment); free((void *)program->log_weight); free((void *)program->first); free((void *)program->move);
}

/* the communicators alive in this process (comm_make to comm_free): a session opens over the widest one among whose
   ranks' nodes a group's are, so a subgroup's collectives find their channels open (torch makes its world group first) */
static struct { pthread_mutex_t lock; ncclComm_t comm[64]; int count; } alive = {PTHREAD_MUTEX_INITIALIZER, {0}, 0};

static void alive_add(ncclComm_t comm) {
  pthread_mutex_lock(&alive.lock);
  if (alive.count < 64) alive.comm[alive.count++] = comm;
  pthread_mutex_unlock(&alive.lock);
}

static void alive_remove(ncclComm_t comm) {
  pthread_mutex_lock(&alive.lock);
  for (int i = 0; i < alive.count; i++)
    if (alive.comm[i] == comm) { alive.comm[i] = alive.comm[--alive.count]; break; }
  pthread_mutex_unlock(&alive.lock);
}

static int among(ncclComm_t wide, ncclComm_t comm) {
  for (int r = 0; r < comm->nranks; r++) {
    int found = 0;
    for (int w = 0; w < wide->nranks && !found; w++) found = wide->members[w] == comm->members[r];
    if (!found) return 0;
  }
  return 1;
}

static ncclComm_t widest(ncclComm_t comm) {
  ncclComm_t best = comm;
  pthread_mutex_lock(&alive.lock);
  for (int i = 0; i < alive.count; i++) {
    const ncclComm_t c = alive.comm[i];
    if (c->nranks > best->nranks && !strcmp(c->region, comm->region) && c->members[c->rank] == comm->members[comm->rank] &&
        among(c, comm))
      best = c;
  }
  pthread_mutex_unlock(&alive.lock);
  return best;
}

static void comm_free(struct ncclComm *comm) {
  alive_remove(comm);
  free(comm->map.link); free(comm->map.cost); free(comm->members);
  free((void *)comm->paths.root); free((void *)comm->paths.parent); free((void *)comm->paths.log_weight);
  for (uint32_t e = 0; e < comm->programs; e++) program_free(&comm->entries[e].program);
  free(comm->entries);
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

#define DEFAULT_ALPHA 5.0   /* us a message, ns a byte: a link's cost where the map gives none (torch-mesh observe's) */
#define DEFAULT_BETA 0.1

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
    return fail(ncclInvalidArgument, "the topology's links do not join every rank");
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
    if (mesh_program_time(&comm->map, &comm->entries[1].program, operand, DEFAULT_ALPHA, DEFAULT_BETA) <
        mesh_program_time(&comm->map, &comm->entries[0].program, operand, DEFAULT_ALPHA, DEFAULT_BETA)) {
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

/* a communicator over `map` (NULL: every pair linked, no cost), its links and costs copied, its messages along `paths`
   and its collectives the table `programs` (NULL: comm_routes's defaults) */
static ncclResult_t comm_make(ncclComm_t *made, const struct mesh_link_map *map, const struct mesh_trees *paths, const ncclMeshProgram *programs,
                              int listed, int nranks, const int *members, int rank, const char *region) {
  struct ncclComm *comm = calloc(1, sizeof *comm);
  if (!comm) return fail(ncclSystemError, "out of memory");
  const uint32_t count = map ? map->links : 0;
  const int costed = map && map->cost;
  comm->map = (struct mesh_link_map){map ? map->kind : MESH_LINKS_MESH, (uint32_t)nranks, count, NULL, NULL};
  comm->map.link = calloc(count ? count : 1, sizeof *comm->map.link);
  if (costed) comm->map.cost = calloc(count ? count : 1, sizeof *comm->map.cost);
  comm->members = calloc((size_t)nranks, sizeof *comm->members);
  if (!comm->map.link || (costed && !comm->map.cost) || !comm->members) {
    free(comm->map.link); free(comm->map.cost); free(comm->members); free(comm);
    return fail(ncclSystemError, "out of memory");
  }
  if (count) memcpy(comm->map.link, map->link, count * sizeof *map->link);
  if (count && costed) memcpy(comm->map.cost, map->cost, count * sizeof *map->cost);
  for (int r = 0; r < nranks; r++) comm->members[r] = members ? members[r] : r;
  comm->nranks = nranks; comm->rank = rank;
  snprintf(comm->region, sizeof comm->region, "%s", region);
  const ncclResult_t result = comm_routes(comm, paths, programs, listed);
  if (result) { comm_free(comm); return result; }
  alive_add(comm);
  *made = comm;
  return ncclSuccess;
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

/* An NCCL program's communicator: every pair of its ranks linked, no cost (ncclMeshCommInitRank takes the
   topology), rank r on the bridge of node r, its region the process's MESH_REGION (default /mesh0), which the
   unique id carries, as NCCL's carries its bootstrap address. */
static const char *region_of_process(void) {
  const char *region = getenv("MESH_REGION");
  return region && *region ? region : "/mesh0";
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  const char *region = region_of_process();
  if (strlen(region) + 1 > sizeof id->internal) return fail(ncclInvalidArgument, "MESH_REGION exceeds the unique id's %zu bytes", sizeof id->internal);
  memset(id->internal, 0, sizeof id->internal);
  strcpy(id->internal, region);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return fail(ncclInvalidArgument, "rank %d of %d", rank, nranks);
  char region[sizeof commId.internal + 1];
  memcpy(region, commId.internal, sizeof commId.internal);
  region[sizeof commId.internal] = 0;
  if (!*region) return fail(ncclInvalidArgument, "the unique id names no region (ncclGetUniqueId)");
  return comm_make(comm, NULL, NULL, NULL, 0, nranks, NULL, rank, region);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config) {
  (void)config;
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  (void)devlist;
  if (ndev != 1) return fail(ncclInvalidArgument, "a node has one Metal device: ndev %d", ndev);
  return comm_make(comm, NULL, NULL, NULL, 0, 1, NULL, 0, region_of_process());
}

static ncclResult_t session_relaying(ncclComm_t comm);
/* a communicator over `topology`; where it leaves a pair of ranks unlinked, its session opened now (every rank calls this
   together), so the rank forwards routed messages for others whether or not it calls anything */
ncclResult_t ncclMeshCommInitRank(ncclComm_t *comm, int rank, const struct mesh_link_map *topology, const struct mesh_trees *paths,
                                  const ncclMeshProgram *programs, int count, const int *node, const char *region) {
  static const char *const kinds[] = {"mesh", "ring", "tree", "graph"};
  if (!comm || !topology) return fail(ncclInvalidArgument, "comm or topology is NULL");
  if (rank < 0 || (uint32_t)rank >= topology->nodes) return fail(ncclInvalidArgument, "rank %d of %u", rank, topology->nodes);
  if (mesh_link_map_check(topology))
    return fail(ncclInvalidArgument, "the topology (%s of %u ranks, %u links) is not one its algorithms can run on (mesh-plan.h)",
                topology->kind <= MESH_LINKS_GRAPH ? kinds[topology->kind] : "unknown kind", topology->nodes, topology->links);
  if (count < 0 || (count && !programs)) return fail(ncclInvalidArgument, "programs is NULL with count %d", count);
  ncclResult_t result = comm_make(comm, topology, paths, count ? programs : NULL, count, (int)topology->nodes, node, rank,
                                  region && *region ? region : "/mesh0");
  if (!result) result = session_relaying(*comm);
  if (result) { ncclCommDestroy(*comm); *comm = NULL; }
  return result;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL"); }

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  comm_free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

static ncclResult_t session_async(ncclComm_t comm);
/* a launch's failure, else the failure of the session the communicator's groups run on (the progress thread's: a
   link cancelled or silent), which a group that had already returned, its work enqueued, could not report */
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

/* -- the session: one prepared program a process, attached at its first group that crosses a link and
   kept for its life.  A channel to each peer node of the communicator that opened it: a ring of DEPTH
   slots each way, slot t % DEPTH carrying invocation t (a position).  Both ends of a channel compute the
   same positions for a group (each collective's rounds in call order, then the point-to-point calls
   paired in order, a segment as long as its longer direction), each publishes every position (filler
   where it has nothing to send), and each publishes position t once it has consumed the peer's t - LAG:
   the peer then has consumed this end's t - 2 LAG = t - DEPTH, the slot t reuses, so no slot is
   overwritten before it is read (NCCL's Simple protocol's slots, its head credit carried by the reverse
   direction's positions). ----------------------------------------------------------------------------- */

#define LAG 8                       /* NCCL_STEPS */
#define DEPTH (2 * LAG)
#define BUFFSIZE ((size_t)1 << 22)  /* NCCL_BUFFSIZE's default */
#define POSITIONS 8192
#define IDENTITY 1
#define RELAY (1u << 20)            /* a routed pair's transfers: RELAY + 2 * (from * NODES + to) + large */
#define NODES 1024
#define BOUND 10.0                  /* metal-microbench configs/e2b-programs.json remote_bound_seconds (MESH_REMOTE_BOUND) */

/* a channel: one class of a peer's pieces on a queue pair of its own (a queue pair's receives are posted
   invocation-major across its transfers, so two rings on one would advance together): `large`, past a small
   ring's flight of LAG one-block slots, NCCL_BUFFSIZE/NCCL_STEPS slots on queue pair 0, else one-block slots on
   queue pair 1, so a small piece pays one block on the wire (NCCL picks its protocol by size); a bridge of one
   queue pair a link has the large class alone.  Its rings of `slot` bytes; its Metal side: its rings as buffers, its out ring's invocation-0 SEND cell in the publication buffer
   and the producer argument a publication stores there, its in ring's invocation-0 completion word and the
   words' stride in the transport's inputs; `waited`, the arrivals the encoded GPU work already waits for;
   `signaled`, the positions landed that the progress thread has seen, up to the `target` the groups reach */
struct channel {
  uint32_t node, hop;
  int large;
  size_t slot;
  struct mesh_section out, in;
  unsigned char *sending, *receiving;
  uint64_t sent, consumed, end, waited;
  void *ring_out, *ring_in;
  size_t cell, word, word_stride;
  uint64_t argument;
  _Atomic uint64_t target, signaled;
};

/* a routed pair's messages through this node (NCCL's PXN proxy): what lands from the route's previous node is published,
   from the slot it landed in, to the next, by the progress thread, with no call of this rank's */
struct relay { uint32_t from, to; struct mesh_section ring; uint64_t forwarded; };

static struct {
  struct mesh_ctx context;
  struct hdr *header;
  char region[64];
  struct channel *channels;
  uint32_t count, positions;
  size_t slot, small;
  int classes;
  double bound;
  int registered, running;
  struct mesh_metal_transport transport;
  struct relay *relays;
  uint32_t relaying;
  pthread_t progress;
  _Atomic int stop, failed;
} session;
static pthread_mutex_t session_lock = PTHREAD_MUTEX_INITIALIZER, progress_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t progress_wake = PTHREAD_COND_INITIALIZER;
static void *finished;
static uint64_t issued;

/* a position's completion word as M30 reads it: landed once at least 1 plus its cycle, ~0 its link cancelled */
static uint64_t cycle_of(uint64_t at) { return at / session.positions; }
static int landed(uint64_t word, uint64_t at) { return word != UINT64_MAX && word >= 1 + cycle_of(at); }

static double now_s(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

/* every link with pending words cancelled (mesh_cancel: each pending word ~0, so every spin on it ends) and the
   failure kept for the next group */
static char failure[512];
static void release_all(int result, const char *format, ...) {
  int none = ncclSuccess;
  if (atomic_compare_exchange_strong(&session.failed, &none, result)) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(failure, sizeof failure, format, arguments);
    va_end(arguments);
  }
  for (uint32_t l = 0; session.transport.cancel && l < session.header->links; l++) mesh_cancel(session.header, session.transport.cancel, l);
  for (uint32_t h = 0; h < session.count; h++) atomic_store(&session.channels[h].signaled, atomic_load(&session.channels[h].target));
}

static ncclResult_t session_async(ncclComm_t comm) {
  const ncclResult_t failed = (ncclResult_t)atomic_load(&session.failed);
  if (!failed || !session.header || strcmp(session.region, comm->region)) return ncclSuccess;
  return fail(failed, "%s", failure);
}

static int outstanding(void) {
  for (uint32_t h = 0; h < session.count; h++)
    if (atomic_load(&session.channels[h].signaled) < atomic_load(&session.channels[h].target)) return 1;
  return 0;
}

/* the progress thread, mesh_rank.m's silent-link detector: it follows each channel's landings up to its target,
   and a channel with positions pending that sees none land for the bound (MESH_REMOTE_BOUND seconds) has its
   link cancelled, as does a landing the bridge cancelled (~0) */
/* every routed piece that has landed here published to the route's next node, from where it landed */
static int forward(void) {
  int moved = 0;
  for (uint32_t i = 0; i < session.relaying && !atomic_load(&session.failed); i++) {
    struct relay *rl = session.relays + i;
    for (;;) {
      const uint64_t arrived = mesh_host_arrived(&session.context, rl->ring, (uint32_t)rl->forwarded);
      if (arrived == UINT64_MAX) { release_all(ncclRemoteError, "a link of the route from node %u to node %u was cancelled", rl->from, rl->to); break; }
      if (!landed(arrived, rl->forwarded)) break;
      mesh_host_publish(&session.context, rl->ring, (uint32_t)rl->forwarded++);
      moved = 1;
    }
  }
  return moved;
}

static void *progress_run(void *unused) {
  (void)unused;
  double idle = now_s() + session.bound;
  while (!atomic_load(&session.stop)) {
    int waiting = 0, moved = forward();
    for (uint32_t h = 0; h < session.count && !atomic_load(&session.failed); h++) {
      struct channel *ch = session.channels + h;
      const uint64_t from = atomic_load(&ch->signaled), target = atomic_load(&ch->target);
      uint64_t at = from;
      for (; at < target; at++) {
        const uint64_t arrived = mesh_host_arrived(&session.context, ch->in, (uint32_t)at);
        if (arrived == UINT64_MAX) { release_all(ncclRemoteError, "the link to node %u was cancelled", ch->node); break; }
        if (!landed(arrived, at)) { waiting = 1; break; }
      }
      if (at > from && !atomic_load(&session.failed)) { atomic_store(&ch->signaled, at); moved = 1; }
    }
    if (moved) idle = now_s() + session.bound;
    else if (waiting && now_s() > idle) release_all(ncclRemoteError, "no landing from the peers in %.1f s: the links cancelled as silent", session.bound);
    if (!waiting && session.relaying && !atomic_load(&session.failed)) {
      if (!moved) usleep(20);
      idle = now_s() + session.bound;
    } else if (!waiting || atomic_load(&session.failed)) {
      pthread_mutex_lock(&progress_lock);
      while (!atomic_load(&session.stop) && (!outstanding() || atomic_load(&session.failed))) pthread_cond_wait(&progress_wake, &progress_lock);
      pthread_mutex_unlock(&progress_lock);
      idle = now_s() + session.bound;
    }
  }
  return NULL;
}

static void progress_targets(void) {
  pthread_mutex_lock(&progress_lock);
  for (uint32_t h = 0; h < session.count; h++) atomic_store(&session.channels[h].target, session.channels[h].end);
  pthread_cond_signal(&progress_wake);
  pthread_mutex_unlock(&progress_lock);
}

/* every encoded GPU program finished (its end signals `finished`); DEADLINE_S with no program finishing and
   no position landing releases the waits, a second ends the drain */
static ncclResult_t drain(void) {
  if (!finished) return ncclSuccess;
  double idle = now_s() + DEADLINE_S;
  int released = 0;
  uint64_t last = metal_signaled(finished), landed = 0;
  while (metal_signaled(finished) < issued) {
    uint64_t now = 0;
    for (uint32_t h = 0; h < session.count; h++) now += atomic_load(&session.channels[h].signaled);
    if (metal_signaled(finished) != last || now != landed) { last = metal_signaled(finished); landed = now; idle = now_s() + DEADLINE_S; }
    else if (now_s() > idle) {
      if (released) return fail(ncclInternalError, "a GPU program of the session did not finish (its command buffer committed?)");
      release_all(ncclTimeout, "the session's GPU programs ran %.0f s with nothing landing", DEADLINE_S);
      released = 1;
      idle = now_s() + DEADLINE_S;
    }
    usleep(50);
  }
  metal_collect(finished);
  return ncclSuccess;
}

static void inflight_lose(void);
static void session_close(double linger) {
  if (!session.header) return;
  inflight_lose();
  drain();
  if (session.running) {
    pthread_mutex_lock(&progress_lock);
    atomic_store(&session.stop, 1);
    pthread_cond_signal(&progress_wake);
    pthread_mutex_unlock(&progress_lock);
    pthread_join(session.progress, NULL);
    session.running = 0;
  }
  if (linger > 0) usleep((useconds_t)(linger * 1e6));
  for (uint32_t h = 0; h < session.count; h++) {
    metal_release(session.channels[h].ring_out); metal_release(session.channels[h].ring_in);
  }
  mesh_metal_transport_destroy(&session.transport);
  mesh_detach(&session.context);
  free(session.channels); free(session.relays);
  session.channels = NULL; session.header = NULL; session.count = 0; session.relays = NULL; session.relaying = 0;
  atomic_store(&session.stop, 0); atomic_store(&session.failed, ncclSuccess);
}

static void session_exit(void) {
  pthread_mutex_lock(&session_lock);
  session_close(LINGER_S);
  pthread_mutex_unlock(&session_lock);
}

static struct channel *channel_of(uint32_t node) {
  for (uint32_t i = 0; i < session.count; i++)
    if (session.channels[i].node == node) return session.channels + i;
  return NULL;
}

static int session_covers(ncclComm_t comm) {
  if (!session.header || strcmp(session.region, comm->region)) return 0;
  for (int r = 0; r < comm->nranks; r++)
    if (r != comm->rank && !channel_of((uint32_t)comm->members[r])) return 0;
  return 1;
}

static size_t setting(const char *name, size_t otherwise) {
  const char *value = getenv(name);
  const long long x = value ? atoll(value) : 0;
  return x > 0 ? (size_t)x : otherwise;
}

static int ring_make(size_t slot, struct mesh_section *section, unsigned char **memory) {
  const int status = mesh_section_create(&session.context, DEPTH * slot, 1, 1, section);
  if (status) return status;
  if (memory) *memory = mesh_section_address(&session.context, *section, 0);
  section->bytes = slot;
  return 0;
}

/* a ring's transfer on the link to `node`, its queue pair `queue` (a queue pair's SENDs, and the receives its peer
   posts, run invocation by invocation through every transfer on it, so rings that advance apart need queue pairs apart) */
static int ring_bind(uint32_t node, uint32_t queue, int receive, uint32_t identity, struct mesh_section section) {
  if (queue >= session.header->qps) return ERANGE;
  return mesh_transfer_bind(&session.context, mesh_peer_channel(&session.context, node, queue), receive ? MESH_RECEIVE : MESH_SEND,
                            identity, section, (uint32_t)(section.bytes / session.header->pgsz), 0, UINT32_MAX, DEPTH);
}

/* the queue pair a pair of ranks' ring of a class takes on the link from u to v: a linked pair's 0 (large) and 1 (small);
   the k-th of the unlinked pairs (from, to), in order, whose path (mesh_trees_path) crosses from u to v, 2 + 2k and
   3 + 2k (NCCL's PXN: a connection of the pair's own on each link its proxy carries it over; the pairs crossing the
   other way take the same queue pairs' other direction) */
static uint32_t queue_of(ncclComm_t comm, uint32_t u, uint32_t v, uint32_t from, uint32_t to, int large, uint32_t *path) {
  const uint32_t class = session.classes == 2 && !large;
  if ((from == u && to == v) || (from == v && to == u)) return class;
  uint32_t k = 0;
  for (uint32_t a = 0; a < (uint32_t)comm->nranks; a++)
    for (uint32_t b = 0; b < (uint32_t)comm->nranks; b++) {
      if (a == b || mesh_link_between(&comm->map, a, b)) continue;
      const uint32_t hops = mesh_trees_path(&comm->paths, a, b, path);
      int crosses = 0;
      for (uint32_t i = 0; i + 1 < hops; i++) crosses |= path[i] == u && path[i + 1] == v;
      if (!crosses) continue;
      if (a == from && b == to) return 2 + 2 * k + class;
      k++;
    }
  return UINT32_MAX;
}

static uint32_t relayed(uint32_t from, uint32_t to, int large) { return RELAY + 2 * (from * NODES + to) + (uint32_t)large; }

/* a channel's Metal side once its transfers are prepared: the rings wrapped, its SEND cell, its completion words */
static int channel_metal(struct channel *ch) {
  struct prepared_publication record;
  if (mesh_publication_prepare(session.header, ch->out.first, NULL) != 1) return EINVAL;
  mesh_publication_prepare(session.header, ch->out.first, &record);
  ch->cell = (size_t)(record.destination - ((uintptr_t)session.header + session.context.send_off));
  ch->argument = record.argument;
  struct mesh_metal_input input;
  const int status = mesh_metal_receive_prepare(&session.context, &session.transport, ch->in, &input);
  if (status) return status;
  metal_release(input.completion);
  ch->word = input.offset; ch->word_stride = input.stride;
  ch->ring_out = metal_wrap(ch->sending, DEPTH * ch->slot);
  ch->ring_in = metal_wrap(ch->receiving, DEPTH * ch->slot);
  return ch->ring_out && ch->ring_in ? 0 : ENOMEM;
}

static int session_open(ncclComm_t comm, int *other) {
  int status = mesh_attach(&session.context, comm->region);
  if (status) return status;
  const struct hdr *m = session.header = session.context.M;
  snprintf(session.region, sizeof session.region, "%s", comm->region);
  if ((uint32_t)comm->members[comm->rank] != m->node) { *other = (int)m->node; session_close(0); return EINVAL; }
  const size_t block = (size_t)m->pgsz * m->block;
  session.slot = (setting("NCCL_BUFFSIZE", BUFFSIZE) / LAG + block - 1) / block * block;
  session.small = block;
  session.classes = m->qps >= 2 ? 2 : 1;
  session.positions = (uint32_t)((setting("MESH_POSITIONS", POSITIONS) + DEPTH - 1) / DEPTH * DEPTH);
  session.bound = setting("MESH_REMOTE_BOUND", 0) ? (double)setting("MESH_REMOTE_BOUND", 0) : BOUND;
  const uint32_t N = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  uint32_t *path = calloc(N + 1, sizeof *path);
  if (!path || !(session.channels = calloc(2 * (size_t)N, sizeof *session.channels))) { free(path); session_close(0); return ENOMEM; }
  /* a channel to each rank: over the link to it, else its out ring over the first link of the path to it and its in ring
     over the last link of the path from it (mesh_trees_path), the routed pair's own transfers */
  for (uint32_t r = 0; r < N && !status; r++)
    for (int large = 2 - session.classes; large < 2 && r != me && !status; large++) {
      struct channel *ch = session.channels + session.count++;
      ch->node = (uint32_t)comm->members[r];
      ch->large = large;
      ch->slot = large ? session.slot : session.small;
      uint32_t out = IDENTITY + 2 * m->node + (uint32_t)large, in = IDENTITY + 2 * ch->node + (uint32_t)large, queue = session.classes == 2 && !large;
      ch->hop = ch->node;
      uint32_t back = ch->node, queue_in = queue;
      if (!mesh_link_between(&comm->map, me, r)) {
        const uint32_t to = mesh_trees_path(&comm->paths, me, r, path), next = to >= 3 ? path[1] : 0;
        const uint32_t from = mesh_trees_path(&comm->paths, r, me, path), previous = from >= 3 ? path[from - 2] : 0;
        if (to < 3 || from < 3 || m->node >= NODES || ch->node >= NODES) { status = EHOSTUNREACH; break; }
        ch->hop = (uint32_t)comm->members[next]; back = (uint32_t)comm->members[previous];
        out = relayed(m->node, ch->node, large); in = relayed(ch->node, m->node, large);
        queue = queue_of(comm, me, next, me, r, large, path);
        queue_in = queue_of(comm, previous, me, r, me, large, path);
      }
      status = ring_make(ch->slot, &ch->out, &ch->sending);
      if (!status) status = ring_bind(ch->hop, queue, 0, out, ch->out);
      if (!status) status = ring_make(ch->slot, &ch->in, &ch->receiving);
      if (!status) status = ring_bind(back, queue_in, 1, in, ch->in);
    }
  /* the routes through this node: each a ring both received from the route's previous node and sent from to its next */
  for (uint32_t from = 0; from < N && !status; from++)
    for (uint32_t to = 0; to < N && !status; to++) {
      const uint32_t hops = from == to || mesh_link_between(&comm->map, from, to) ? 0 : mesh_trees_path(&comm->paths, from, to, path);
      for (uint32_t i = 1; i + 1 < hops && !status; i++)
        for (int large = 2 - session.classes; path[i] == me && large < 2 && !status; large++) {
          struct relay *grown = realloc(session.relays, (session.relaying + 1) * sizeof *grown);
          if (!grown) { status = ENOMEM; break; }
          session.relays = grown;
          struct relay *rl = session.relays + session.relaying++;
          *rl = (struct relay){(uint32_t)comm->members[from], (uint32_t)comm->members[to], {0}, 0};
          const uint32_t identity = relayed(rl->from, rl->to, large), previous = path[i - 1], next = path[i + 1];
          uint32_t *scratch = calloc(N + 1, sizeof *scratch);
          const uint32_t in = scratch ? queue_of(comm, previous, me, from, to, large, scratch) : UINT32_MAX;
          const uint32_t out = scratch ? queue_of(comm, me, next, from, to, large, scratch) : UINT32_MAX;
          free(scratch);
          status = ring_make(large ? session.slot : session.small, &rl->ring, NULL);
          if (!status) status = ring_bind((uint32_t)comm->members[previous], in, 1, identity, rl->ring);
          if (!status) status = ring_bind((uint32_t)comm->members[next], out, 0, identity, rl->ring);
        }
    }
  free(path);
  if (!status) status = mesh_transfers_prepare(&session.context, 1, session.positions, DEPTH);
  if (!status) status = mesh_transfers_cyclic(&session.context);
  if (!status) status = mesh_metal_transport_create(&session.context, metal_device(), &session.transport);
  for (uint32_t h = 0; h < session.count && !status; h++) status = channel_metal(session.channels + h);
  if (!status && !finished && !(finished = metal_event())) status = ENOMEM;
  if (!status) status = mesh_transfers_start(&session.context);
  if (!status) status = pthread_create(&session.progress, NULL, progress_run, NULL);
  if (status) { session_close(0); return status; }
  session.running = 1;
  if (!session.registered) session.registered = !atexit(session_exit);
  return 0;
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
    if ((p2p(c) && c->peer == comm->rank) || !c->count || comm->nranks < 2) continue;
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
static int in_class(const struct mesh_step *step, int large) {
  return session.classes == 1 || (piece_bytes(step) > LAG * session.small) == large;
}

/* the comm's rank on a channel's node (-1: none) */
static int rank_of(ncclComm_t comm, uint32_t node) {
  for (int r = 0; r < comm->nranks; r++)
    if (r != comm->rank && (uint32_t)comm->members[r] == node) return r;
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
static uint64_t lay(const struct call *list, size_t n, struct mesh_step *const *steps, const uint32_t *counts, int peer, int large,
                    size_t slot, struct placing *made, size_t *count) {
  uint64_t at = 0;
  *count = 0;
  for (size_t i = 0; i < n; i++) {
    if (p2p(list + i)) continue;
    uint32_t rounds = 0;
    for (uint32_t k = 0; k < counts[i]; k++)
      if ((int)steps[i][k].peer == peer && in_class(steps[i] + k, large) && steps[i][k].round + 1 > rounds) rounds = steps[i][k].round + 1;
    for (uint32_t round = 0; round < rounds; round++) {
      uint64_t out = at, in = at;
      for (uint32_t k = 0; k < counts[i]; k++) {
        const struct mesh_step *step = steps[i] + k;
        const uint64_t pieces = (piece_bytes(step) + slot - 1) / slot;
        if ((int)step->peer != peer || !in_class(step, large) || step->round != round || !pieces) continue;
        uint64_t *cursor = step->op == MESH_STEP_SEND ? &out : &in;
        made[(*count)++] = (struct placing){(uint32_t)i, k, step->op == MESH_STEP_SEND, *cursor, pieces};
        *cursor += pieces;
      }
      at = out > in ? out : in;
    }
  }
  for (size_t s = 0, r = 0;; s++, r++) {
    while (s < n && !(p2p(list + s) && counts[s] && list[s].what == WHAT_SEND && (int)steps[s][0].peer == peer && in_class(steps[s], large))) s++;
    while (r < n && !(p2p(list + r) && counts[r] && list[r].what == WHAT_RECV && (int)steps[r][0].peer == peer && in_class(steps[r], large))) r++;
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
static ncclResult_t schedule_group(struct call *list, size_t n, struct schedule *plans) {
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
  for (uint32_t h = 0; h < session.count && !result; h++) {
    struct channel *ch = session.channels + h;
    struct schedule *p = plans + h;
    *p = (struct schedule){calloc(total, sizeof *p->out), calloc(total, sizeof *p->in), 0, 0, 0, 0};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    size_t count;
    const uint64_t positions = lay(list, n, steps, counts, rank_of(comm, ch->node), ch->large, ch->slot, made, &count);
    for (size_t k = 0; k < count; k++) {
      struct call *c = list + made[k].call;
      const struct mesh_step *step = c->bound->steps + made[k].step;
      const struct message m = {c, made[k].step, ch->end + made[k].first, made[k].pieces, step->first * step->piece.element_bytes, piece_bytes(step)};
      if (made[k].send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
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

static void schedules_free(struct schedule *plans) {
  for (uint32_t h = 0; plans && h < session.count; h++) { free(plans[h].out); free(plans[h].in); }
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
   exchange's sums go to a copy, as its SENDs read the operand) */
static ncclResult_t run(struct schedule *plans) {
  double idle = now_s() + DEADLINE_S;
  for (;;) {
    int busy = 0, open = 0;
    for (uint32_t h = 0; h < session.count; h++) {
      struct channel *ch = session.channels + h;
      struct schedule *p = plans + h;
      while (ch->consumed < ch->end) {
        const uint64_t at = ch->consumed, arrived = mesh_host_arrived(&session.context, ch->in, (uint32_t)at);
        if (arrived == UINT64_MAX) return fail(ncclRemoteError, "the link to node %u was cancelled", ch->node);
        if (!landed(arrived, at)) break;
        struct message *m = p->in_at < p->ins ? p->in + p->in_at : NULL;
        if (m && at >= m->first) {
          struct steps *s = m->c->bound;
          if (at == m->first && !ready(s, m->step)) break;
          const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
          const unsigned char *from = ch->receiving + (at % DEPTH) * ch->slot;
          unsigned char *to = s->host[s->steps[m->step].buffer] + m->offset + offset;
          if (s->steps[m->step].op == MESH_STEP_REDUCE) combine_into(m->c->type, m->c->combine, to, from, length / SIZE[m->c->type]);
          else memcpy(to, from, length);
          if (at + 1 == m->first + m->pieces) { s->done[m->step] = 1; p->in_at++; }
        }
        ch->consumed++; busy = 1;
      }
      while (ch->sent < ch->end && ch->sent < ch->consumed + LAG) {
        const uint64_t at = ch->sent;
        struct message *m = p->out_at < p->outs ? p->out + p->out_at : NULL;
        if (m && at >= m->first) {
          if (at == m->first && !ready(m->c->bound, m->step)) break;
          const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
          memcpy(ch->sending + (at % DEPTH) * ch->slot, m->c->bound->host[m->c->bound->steps[m->step].buffer] + m->offset + offset, length);
          if (at + 1 == m->first + m->pieces) p->out_at++;
        }
        mesh_host_publish(&session.context, ch->out, (uint32_t)at);
        ch->sent++; busy = 1;
      }
      open |= ch->sent < ch->end || ch->consumed < ch->end;
    }
    if (!open) return ncclSuccess;
    if (busy) idle = now_s() + DEADLINE_S;
    else if (now_s() > idle) return fail(ncclTimeout, "nothing landed from the peers in %.0f s", DEADLINE_S);
  }
}

/* the session made to cover the communicator: opened (or renewed: other peers or another region, never while
   groups are in flight); a failure the progress thread kept fails this group and retires the session, and the
   next group opens another */
static struct inflight *inflight;
static ncclResult_t session_ensure(ncclComm_t comm) {
  if (session.header && atomic_load(&session.failed)) {
    const ncclResult_t result = fail((ncclResult_t)atomic_load(&session.failed), "%s", failure);
    session_close(0);
    return result;
  }
  for (int attempt = 0;;) {
    if (!session_covers(comm)) {
      if (inflight) return fail(ncclInvalidUsage, "a group on other peers while groups are issued and not complete");
      session_close(LINGER_S);
      int other = -1;
      const int status = session_open(widest(comm), &other);
      if (other >= 0) return fail(ncclInvalidUsage, "rank %d runs on the bridge of node %d", comm->rank, other);
      if (status == ERANGE)
        return fail(ncclInvalidUsage, "the bridges' links have too few queue pairs for the routes across them (MESH_QPS: 2, and 2 for each "
                                      "routed pair crossing the busiest link)");
      if (status == EHOSTUNREACH) return fail(ncclInvalidUsage, "a pair of ranks has no route on the map, or a node number is %d or more", NODES);
      if (status) {
        const ncclResult_t result = fail(ncclSystemError, "the session's transfers did not start: %s", strerror(status));
        if (++attempt == ATTEMPTS) return result;
        fprintf(stderr, "%s; again\n", last);
        continue;
      }
    }
    return ncclSuccess;
  }
}

static ncclResult_t session_relaying(ncclComm_t comm) {
  int partial = 0;
  for (int a = 0; a < comm->nranks; a++)
    for (int b = a + 1; b < comm->nranks; b++) partial |= !mesh_link_between(&comm->map, (uint32_t)a, (uint32_t)b);
  if (!partial) return ncclSuccess;
  pthread_mutex_lock(&session_lock);
  const ncclResult_t result = session_ensure(comm);
  pthread_mutex_unlock(&session_lock);
  return result;
}

static ncclResult_t session_schedule(struct call *list, size_t n, struct schedule **made) {
  struct schedule *plans = calloc(session.count + 1, sizeof *plans);
  ncclResult_t result;
  *made = NULL;
  if (!plans) return fail(ncclSystemError, "out of memory");
  if ((result = schedule_group(list, n, plans))) { schedules_free(plans); return result; }
  *made = plans;
  return ncclSuccess;
}

/* the group on the host, after every GPU program; its positions count as landed for the progress thread */
static ncclResult_t session_run(ncclComm_t comm, struct call *list, size_t n) {
  struct schedule *plans;
  if (inflight) return fail(ncclInvalidUsage, "a host-path group while GPU groups are issued and not complete (ncclMeshComplete)");
  ncclResult_t result = drain();
  if (!result) result = session_ensure(comm);
  if (!result) result = session_schedule(list, n, &plans);
  if (result) return result;
  result = run(plans);
  schedules_free(plans);
  if (result) { session_close(0); return result; }
  for (uint32_t h = 0; h < session.count; h++) {
    struct channel *ch = session.channels + h;
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

/* _launch: one group: its calls planned and staged (a SEND read and a receive written in place), run on
   the session, finished */
static ncclResult_t launch(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclResult_t result = ncclSuccess;
  for (size_t i = 0; i < n; i++) {
    if (list[i].comm != comm) return fail(ncclInvalidUsage, "a group on several communicators");
    if (list[i].stream) return fail(ncclInvalidUsage, "a group on several streams");
    if (reducing(list[i].what) && (result = reduction(list + i))) return result;
  }
  if ((result = plan_calls(comm, list, n))) return result;
  int remote = 0;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *s = c->bound;
    const size_t bytes = elements(c) * SIZE[c->type];
    remote |= s != NULL;
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
  if (!result && remote) {
    pthread_mutex_lock(&session_lock);
    result = session_run(comm, list, n);
    pthread_mutex_unlock(&session_lock);
  }
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

/* the GPU waits until `count` of the channel's positions have landed: a spin on the last one's completion word */
static ncclResult_t arrive(struct metal_program *program, struct channel *ch, uint64_t count) {
  if (count <= ch->waited) return ncclSuccess;
  ch->waited = count;
  const uint64_t at = count - 1;
  return gpu(metal_spin(program, session.transport.inputs, ch->word + (at % session.positions) * ch->word_stride, 1 + cycle_of(at)), "a spin");
}

/* a group issued and not complete: its calls, its schedule and its positions [start, end) on each channel; the
   groups in flight in issue order hold consecutive positions on every channel */
/* one of this rank's publications in its group's order (order_group): its level, channel and position */
struct event { uint64_t level; uint32_t h; uint64_t at; };

/* a group issued and not complete: its calls, its schedule and its positions [start, end) on each channel, its
   publications in order (`next` the first the walk has not completed); the groups in flight in issue order hold
   consecutive positions on every channel */
struct inflight {
  uint64_t ticket;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t *start, *end;
  struct event *order;
  size_t events, next;
  struct inflight *next_group;
};
static struct inflight *inflight_last;
static uint64_t tickets, lost_from = 1, lost_to;
static _Atomic uint64_t retired;

static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    steps_free(list[i].bound);
    if (list[i].owned) free(list[i].result);
    free(list[i].copy);
  }
  free(list);
}

static void inflight_free(struct inflight *g) {
  schedules_free(g->plans);
  free(g->start);
  free(g->order);
  calls_free(g->list, g->n);
  free(g);
}

/* every group in flight lost with its session: a later completion of one returns the failure */
static void inflight_lose(void) {
  if (!inflight) return;
  lost_from = inflight->ticket; lost_to = inflight_last->ticket;
  while (inflight) { struct inflight *g = inflight; inflight = g->next_group; inflight_free(g); }
  inflight_last = NULL;
  atomic_store(&retired, lost_to);
}

static struct inflight *inflight_at(uint32_t h, uint64_t at) {
  for (struct inflight *g = inflight; g; g = g->next_group)
    if (at < g->end[h]) return at >= g->start[h] ? g : NULL;
  return NULL;
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
      for (int large = 2 - session.classes; q != r && large < 2; large++) {
        size_t count;
        const size_t e = r < q ? CHANNEL(r, q, large) : CHANNEL(q, r, large);
        length[2 * e + (r > q)] = lay(list, n, steps + (size_t)r * n, counts + (size_t)r * n, q, large, large ? session.slot : session.small, made, &count);
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
  for (uint32_t h = 0; h < session.count; h++) events += g->end[h] - g->start[h];
  if (!result && !(g->order = calloc(events + 1, sizeof *g->order))) result = fail(ncclSystemError, "out of memory");
  for (uint32_t h = 0; h < session.count && !result; h++) {
    const struct channel *ch = session.channels + h;
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
static ncclResult_t consume(struct metal_program *program, uint32_t h, uint64_t end) {
  struct channel *ch = session.channels + h;
  ncclResult_t result = ncclSuccess;
  while (ch->consumed < end && !result) {
    const uint64_t at = ch->consumed;
    struct inflight *g = inflight_at(h, at);
    if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
    struct schedule *p = g->plans + h;
    struct message *m = p->in_at < p->ins && at >= p->in[p->in_at].first ? p->in + p->in_at : NULL;
    if (m && at == m->first && m->c->bound->before[m->step] >= 0) {
      const struct steps *prior = m->c->bound;
      const int32_t j = prior->before[m->step];
      result = consume(program, prior->channel[j], prior->first[j] + prior->pieces[j]);
      if (result) break;
    }
    if (m) {
      struct steps *st = m->c->bound;
      const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
      const size_t from = (at % DEPTH) * ch->slot;
      const int reduce = st->steps[m->step].op == MESH_STEP_REDUCE;
      const struct where to = st->gpu[st->steps[m->step].buffer];
      if (!ch->large) {
        result = gpu(metal_land(program, reduce ? (int)m->c->type : -1, m->c->combine, to.buffer, to.offset + m->offset + offset, ch->ring_in, from,
                                reduce ? length / SIZE[m->c->type] : length, session.transport.inputs,
                                ch->word + (at % session.positions) * ch->word_stride, 1 + cycle_of(at)), "a landing");
        if (at + 1 > ch->waited) ch->waited = at + 1;
      } else {
        result = arrive(program, ch, at + 1);
        if (!result && reduce)
          result = gpu(metal_combine(program, m->c->type, m->c->combine, to.buffer, to.offset + m->offset + offset,
                                     ch->ring_in, from, length / SIZE[m->c->type]), "a combine");
        else if (!result) {
          result = gpu(metal_copy(program, METAL_LAND, to.buffer, to.offset + m->offset + offset, ch->ring_in, from, length), "a landing");
        }
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
static ncclResult_t publish(struct metal_program *program, uint32_t h, int *published) {
  struct channel *ch = session.channels + h;
  const uint64_t at = ch->sent, need = at >= LAG ? at - LAG + 1 : 0;
  struct inflight *g = inflight_at(h, at);
  if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
  struct schedule *p = g->plans + h;
  struct message *m = p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
  ncclResult_t result;
  result = arrive(program, ch, need);
  const size_t cell = ch->cell + (at % session.positions) * sizeof(struct mesh_send);
  const uint64_t value = ch->argument + cycle_of(at);
  int released = 0;
  if (m && !result) {
    const struct steps *st = m->c->bound;
    const size_t offset = (at - m->first) * ch->slot, length = m->bytes - offset < ch->slot ? m->bytes - offset : ch->slot;
    if (!ch->large) {
      const struct where source = st->gpu[st->steps[m->step].buffer];
      result = gpu(metal_send_small(program, ch->ring_out, (at % DEPTH) * ch->slot, source.buffer, source.offset + m->offset + offset,
                                    length, session.transport.publication, cell, value), "a piece into its slot and its release");
      released = 1;
    } else
      result = gpu(metal_copy(program, METAL_SEND, ch->ring_out, (at % DEPTH) * ch->slot, st->gpu[st->steps[m->step].buffer].buffer,
                              st->gpu[st->steps[m->step].buffer].offset + m->offset + offset, length), "a piece into its slot");
    if (at + 1 == m->first + m->pieces) p->out_at++;
  }
  if (!result && !released) result = gpu(metal_publish(program, session.transport.publication, cell, value), "a publication");
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
  ncclResult_t result = ncclSuccess;
  for (; g->next < g->events && !result; g->next++) {
    const struct event v = g->order[g->next];
    const struct channel *ch = session.channels + v.h;
    if (v.at < ch->sent) continue;
    if (v.at != ch->sent) return fail(ncclInternalError, "the group's positions admit no order");
    result = consume(program, v.h, v.at >= LAG ? v.at - LAG + 1 : 0);
    const struct message *m = result ? NULL : sending(g, v.h, v.at);
    if (m && v.at == m->first && m->c->bound->before[m->step] >= 0) {
      const struct steps *prior = m->c->bound;
      const int32_t j = prior->before[m->step];
      result = consume(program, prior->channel[j], prior->first[j] + prior->pieces[j]);
    }
    if (!result) result = publish(program, v.h, published);
  }
  for (uint32_t h = 0; h < session.count && !result; h++) result = consume(program, h, g->end[h]);
  return result;
}

/* a group's publications that need no wait, in its order: its channel's earlier positions published, the positions LAG
   back consumed (empty ones freely) and the peer's landing there awaited before, a SEND's receives before it consumed */
static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  ncclResult_t result = ncclSuccess;
  for (size_t e = g->next; e < g->events && !result; e++) {
    const struct event v = g->order[e];
    struct channel *ch = session.channels + v.h;
    const uint64_t need = v.at >= LAG ? v.at - LAG + 1 : 0;
    if (v.at != ch->sent || need > ch->waited) continue;
    while (ch->consumed < need) {
      struct inflight *at = inflight_at(v.h, ch->consumed);
      struct schedule *p = at ? at->plans + v.h : NULL;
      if (!p || (p->in_at < p->ins && ch->consumed >= p->in[p->in_at].first)) break;
      ch->consumed++;
    }
    const struct message *m = sending(g, v.h, v.at);
    if (ch->consumed < need || (m && v.at == m->first && !ready(m->c->bound, m->step))) continue;
    result = publish(program, v.h, published);
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

static int on_gpu(ncclDataType_t t) { return t != ncclFloat64 && t != ncclFloat8e4m3 && t != ncclFloat8e5m2; }
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
    for (uint32_t h = 0; h < session.count; h++)
      if (session.channels[h].consumed < g->end[h]) return ncclSuccess;
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

/* _launch on the stream's command buffer, taking the group's calls: each collective's operand recvbuff itself where
   it holds the whole operand (an all-reduce's, a root's reduce, a broadcast's, an all-gather's), else a scratch one
   (a reduce-scatter's, a non-root's reduce); a direct exchange's sums apart; then the group in flight after the
   others, issued (with issue) or completed with every group before it; its results into recvbuff as it completes;
   the program's end signals `finished` where the library made its encoder */
static ncclResult_t launch_metal(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclMeshIssue *const issue = list[0].issue;
  ncclResult_t result = ncclSuccess;
  if (issue) *issue = (ncclMeshIssue){0, 0};
  for (size_t i = 0; i < n && !result; i++) {
    if (list[i].comm != comm) result = fail(ncclInvalidUsage, "a group on several communicators");
    else if (list[i].command != list[0].command) result = fail(ncclInvalidUsage, "a group on several command buffers");
    else if (!on_gpu(list[i].type)) result = fail(ncclInvalidArgument, "datatype %d on the Metal path (no float64 or float8 there)", (int)list[i].type);
    else if (reducing(list[i].what)) result = reduction(list + i);
  }
  if (!result && !list[0].command) result = fail(ncclInvalidArgument, "an ncclMeshStream without a command buffer");
  if (!result && list[0].encoder && !list[0].workspace) result = fail(ncclInvalidArgument, "an ncclMeshStream with an encoder and no workspace");
  if (!result) result = plan_calls(comm, list, n);
  plans_record(list, n);
  if (result) { calls_free(list, n); return result; }
  pthread_mutex_lock(&session_lock);
  if (finished) metal_collect(finished);
  struct metal_program *program = metal_begin(list[0].command, list[0].encoder);
  if (!program) { pthread_mutex_unlock(&session_lock); calls_free(list, n); return fail(ncclSystemError, "out of memory"); }
  const size_t r = (size_t)comm->rank;
  int remote = 0;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *st = c->bound;
    const size_t z = SIZE[c->type], bytes = elements(c) * z, all = elements(c);
    const struct where send = c->send_at;
    remote |= st != NULL;
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
  if (!result && remote) result = session_ensure(comm);
  if (!result && remote) {
    struct inflight *g = calloc(1, sizeof *g);
    if (!g || !(g->start = calloc(2 * (size_t)session.count + 1, sizeof *g->start))) { free(g); result = fail(ncclSystemError, "out of memory"); }
    else {
      g->end = g->start + session.count;
      for (uint32_t h = 0; h < session.count; h++) g->start[h] = session.channels[h].end;
      result = session_schedule(list, n, &g->plans);
      if (result) { free(g->start); free(g); }
      else {
        for (uint32_t h = 0; h < session.count; h++) g->end[h] = session.channels[h].end;
        g->list = list; g->n = n; g->ticket = ticket = ++tickets; owned = 0;
        if (inflight_last) inflight_last->next_group = g; else inflight = g;
        inflight_last = g;
        scheduled = 1;
        result = order_group(comm, g);
        progress_targets();
        if (!result) result = walk(program, issue ? g : NULL, issue ? NULL : g, &published);
        if (!result) result = retire(program);
      }
    }
  } else if (!result)
    result = epilogue(program, list, n);
  program_end(program, list[0].encoder != NULL);
  if (result && scheduled) { release_all(result, "%s", last); inflight_lose(); }
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
  ncclResult_t result = ncclSuccess;
  if (ticket >= lost_from && ticket <= lost_to) result = fail(ncclRemoteError, "the group was lost with its session: %s", failure);
  else if (inflight && ticket >= inflight->ticket && atomic_load(&session.failed)) {
    result = fail((ncclResult_t)atomic_load(&session.failed), "%s", failure);
    inflight_lose();
  } else if (inflight && ticket >= inflight->ticket) {
    struct inflight *through = inflight;
    while (through->next_group && through->next_group->ticket <= ticket) through = through->next_group;
    if (finished) metal_collect(finished);
    struct metal_program *program = metal_begin(stream->commandBuffer, stream->commandEncoder);
    if (!program) result = fail(ncclSystemError, "out of memory");
    else {
      result = walk(program, NULL, through, &any);
      if (!result) result = retire(program);
      program_end(program, stream->commandEncoder != NULL);
      if (result) { release_all(result, "%s", last); inflight_lose(); }
    }
  }
  pthread_mutex_unlock(&session_lock);
  if (published) *published = any;
  return result;
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

/* ncclCommSplit: every rank's (color, key) all-gathered on comm, the ranks of this color ordered by key,
   then rank; the new map is comm's links among them, relabelled, each with its cost: the same kind where every
   rank stays, a mesh where every pair of them is linked, else a graph (refused where those links do not connect
   them) */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  if (depth) return fail(ncclInvalidUsage, "ncclCommSplit inside a group");
  const int n = comm->nranks;
  int64_t mine[2] = {color, key}, *every = calloc(2 * (size_t)n, sizeof *every);
  int *ranks = calloc((size_t)n, sizeof *ranks), *index = calloc((size_t)n, sizeof *index), *members = calloc((size_t)n, sizeof *members);
  uint32_t (*linked)[2] = calloc(comm->map.links + 1, sizeof *linked);
  double (*priced)[2] = calloc(comm->map.links + 1, sizeof *priced);
  ncclResult_t result = !every || !ranks || !index || !members || !linked || !priced ? fail(ncclSystemError, "out of memory")
                                                                                     : ncclAllGather(mine, every, 2, ncclInt64, comm, NULL);
  *newcomm = NULL;
  if (!result && color != NCCL_SPLIT_NOCOLOR) {
    int count = 0;
    for (int r = 0; r < n; r++) if (every[2 * r] == color) ranks[count++] = r;
    for (int a = 1; a < count; a++)
      for (int b = a; b > 0 && (every[2 * ranks[b] + 1] < every[2 * ranks[b - 1] + 1]); b--) { int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
    for (int r = 0; r < n; r++) index[r] = -1;
    for (int i = 0; i < count; i++) { index[ranks[i]] = i; members[i] = comm->members[ranks[i]]; }
    uint32_t links = 0;
    for (uint32_t l = 0; l < comm->map.links; l++) {
      const int a = index[comm->map.link[l][0]], b = index[comm->map.link[l][1]];
      if (a >= 0 && b >= 0) {
        linked[links][0] = (uint32_t)a; linked[links][1] = (uint32_t)b;
        if (comm->map.cost) { priced[links][0] = comm->map.cost[l][0]; priced[links][1] = comm->map.cost[l][1]; }
        links++;
      }
    }
    int every_pair = comm->map.kind == MESH_LINKS_MESH;
    for (int a = 0; !every_pair && a < count; a++)
      for (int b = a + 1; b < count; b++) {
        int found = 0;
        for (uint32_t l = 0; l < links; l++)
          found |= ((int)linked[l][0] == a && (int)linked[l][1] == b) || ((int)linked[l][0] == b && (int)linked[l][1] == a);
        if (!found) goto checked;
      }
    every_pair = 1;
  checked:;
    const struct mesh_link_map restricted = {count == n ? comm->map.kind : every_pair ? MESH_LINKS_MESH : MESH_LINKS_GRAPH, (uint32_t)count, links,
                                             linked, comm->map.cost ? priced : NULL};
    if (count > 1 && mesh_link_map_check(&restricted)) result = fail(ncclInvalidUsage, "a split to ranks whose links do not connect them");
    else result = comm_make(newcomm, &restricted, NULL, NULL, 0, count, members, index[comm->rank], comm->region);
  }
  free(every); free(ranks); free(index); free(members); free(linked); free(priced);
  return result;
}

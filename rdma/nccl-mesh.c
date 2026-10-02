/* libnccl-mesh (nccl.h): NCCL's API on the bridges' prepared transfers.  A group is one Mesh of
   rdma/mesh.py, transcribed: one attach, every call planned by the one collective planner and bound
   (mesh_collective_bind), one pairing, the SENDs published and the receives awaited and combined on the
   host (Steps.__call__), the attach retired.  The calls' mapping is metal-microbench tools/nccl_demo.py's
   as of dec7676 (1,724 of 1,724 calls checked on the pair, output_data/nccl-20260928). */
#include "nccl.h"
#include "mesh-collective.h"
#include "mesh.h"
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { WHAT_SEND = 16, WHAT_RECV };
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN };
#define P2P (1u << 24)
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

/* -- communicators -------------------------------------------------------------------------------- */

struct premul { int used; ncclDataType_t type; unsigned char value[8]; };
struct ncclComm {
  struct mesh_link_map map;
  int nranks, rank, *members;
  char region[64];
  double alpha, beta;
  struct premul ops[OPS];
  ncclResult_t async;
};

static ncclResult_t comm_make(ncclComm_t *made, uint32_t kind, int nranks, const uint32_t (*links)[2], uint32_t count,
                              const int *members, int rank, const char *region, double alpha, double beta) {
  struct ncclComm *comm = calloc(1, sizeof *comm);
  if (!comm) return fail(ncclSystemError, "out of memory");
  comm->map.kind = kind; comm->map.nodes = (uint32_t)nranks; comm->map.links = count;
  comm->map.link = calloc(count ? count : 1, sizeof *comm->map.link);
  comm->members = calloc((size_t)nranks, sizeof *comm->members);
  if (!comm->map.link || !comm->members) { free(comm->map.link); free(comm->members); free(comm); return fail(ncclSystemError, "out of memory"); }
  if (count) memcpy(comm->map.link, links, count * sizeof *links);
  for (int r = 0; r < nranks; r++) comm->members[r] = members ? members[r] : r;
  comm->nranks = nranks; comm->rank = rank; comm->alpha = alpha; comm->beta = beta;
  snprintf(comm->region, sizeof comm->region, "%s", region);
  *made = comm;
  return ncclSuccess;
}

/* the first link line's cost "a b alpha beta" (mesh_link_map_read reads its first two fields) */
static void map_cost(const char *path, double *alpha, double *beta) {
  *alpha = *beta = 0;
  FILE *file = fopen(path, "r");
  if (!file) return;
  char line[512];
  while (fgets(line, sizeof line, file)) {
    unsigned a, b;
    double x, y;
    char rest[2];
    if (sscanf(line, " %u %u %lf %lf %1s", &a, &b, &x, &y, rest) == 4) { *alpha = x; *beta = y; break; }
  }
  fclose(file);
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

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  const char *links = getenv("MESH_LINKS"), *region = getenv("MESH_REGION");
  links = links ? links : "";
  region = region && *region ? region : "/mesh0";
  if (strlen(links) + strlen(region) + 2 > sizeof id->internal)
    return fail(ncclInvalidArgument, "MESH_LINKS and MESH_REGION exceed the unique id's %zu bytes", sizeof id->internal);
  memset(id->internal, 0, sizeof id->internal);
  strcpy(id->internal, links);
  strcpy(id->internal + strlen(links) + 1, region);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return fail(ncclInvalidArgument, "rank %d of %d", rank, nranks);
  char id[sizeof commId.internal + 1];
  memcpy(id, commId.internal, sizeof commId.internal);
  id[sizeof commId.internal] = 0;
  const char *path = id, *region = id + strlen(id) + 1;
  if (!*region) return fail(ncclInvalidArgument, "the unique id names no region (ncclGetUniqueId)");
  if (!*path) return comm_make(comm, MESH_LINKS_MESH, nranks, NULL, 0, NULL, rank, region, 0, 0);
  struct mesh_link_map read;
  if (mesh_link_map_read(path, &read)) return fail(ncclInvalidArgument, "%s: not a link map (mesh-collective.h)", path);
  if ((int)read.nodes != nranks) {
    mesh_link_map_free(&read);
    return fail(ncclInvalidArgument, "%s has %u nodes, not %d", path, read.nodes, nranks);
  }
  double alpha, beta;
  map_cost(path, &alpha, &beta);
  ncclResult_t result = comm_make(comm, read.kind, nranks, (const uint32_t (*)[2])read.link, read.links, NULL, rank, region, alpha, beta);
  mesh_link_map_free(&read);
  return result;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config) {
  (void)config;
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  (void)devlist;
  if (ndev != 1) return fail(ncclInvalidArgument, "a node has one Metal device: ndev %d", ndev);
  const char *region = getenv("MESH_REGION");
  return comm_make(comm, MESH_LINKS_MESH, 1, NULL, 0, NULL, 0, region && *region ? region : "/mesh0", 0, 0);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL"); }

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  free(comm->map.link); free(comm->members); free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  *asyncError = comm->async;
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

/* -- a group: one Mesh (rdma/mesh.py) ------------------------------------------------------------- */

struct group {
  struct mesh_ctx context;
  struct hdr *header;
  const struct mesh_link_map *map;
  const int *members;
  int rank;
  uint32_t identity;
  size_t block;
};

struct received { unsigned char *slots; size_t offset, size; };
struct steps {
  struct mesh_step *steps;
  uint32_t count, received_count;
  struct mesh_section operand, *pieces;
  struct received *received;
  unsigned char *slots;
  int direct, how, root;
};

struct call {
  int what, root, peer;
  const void *send;
  void *recv;
  size_t count;
  ncclDataType_t type;
  ncclRedOp_t op;
  ncclComm_t comm;
  uint32_t how;
  int force;
  int combine, post;
  unsigned char pre[8];
  int premultiplied;
  struct steps *bound;
  unsigned char *result, *copy;
  int owned;
};

static _Thread_local int depth;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;
static _Thread_local int *plan_how, *plan_root;
static _Thread_local size_t plan_count;

static int reducing(int what) { return what == MESH_ALLREDUCE || what == MESH_REDUCE || what == MESH_REDUCE_SCATTER; }
static size_t elements(const struct call *c) {
  return c->count * ((c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER) ? (size_t)c->comm->nranks : 1);
}

static void group_close(struct group *g, double linger) {
  if (!g->header) return;
  if (linger > 0) usleep((useconds_t)(linger * 1e6));
  mesh_detach(&g->context);
  g->header = NULL;
}

static int group_open(struct group *g, ncclComm_t comm) {
  memset(g, 0, sizeof *g);
  const int status = mesh_attach(&g->context, comm->region);
  if (status) return status;
  g->header = g->context.M;
  g->map = &comm->map;
  g->members = comm->members;
  g->rank = -1;
  for (int r = 0; r < comm->nranks; r++)
    if ((uint32_t)comm->members[r] == g->header->node) g->rank = r;
  g->identity = 1;
  g->block = (size_t)g->header->pgsz * g->header->block;
  return 0;
}

/* Mesh.ring: a registered section of one slot of offset + nbytes in whole blocks */
static int ring(struct group *g, size_t nbytes, size_t offset, struct mesh_section *section, uint32_t *pages, unsigned char **memory) {
  *pages = (uint32_t)(((offset + nbytes + g->block - 1) / g->block) * g->header->block);
  const int status = mesh_section_create(&g->context, (size_t)*pages * g->header->pgsz, 1, 1, section);
  if (status) return status;
  *memory = mesh_section_address(&g->context, *section, 0);
  return *memory ? 0 : ENOMEM;
}

static void steps_free(struct steps *s) {
  if (!s) return;
  free(s->steps); free(s->pieces); free(s->received); free(s);
}

/* Steps.__init__: the operand's slot, a received section a REDUCE (at its SEND piece's offset within a
   block, so both ends cut the piece in the same chunks), bound at transfer identity `identity` */
static int steps_bind(struct group *g, struct steps *s, ncclDataType_t type, size_t count, uint32_t identity) {
  uint32_t pages;
  int status = ring(g, count * SIZE[type], 0, &s->operand, &pages, &s->slots);
  if (status) return status;
  struct mesh_section *received = calloc(s->count + 1, sizeof *received);
  struct mesh_step *bound = calloc(s->count + 1, sizeof *bound);
  s->received = calloc(s->count + 1, sizeof *s->received);
  s->pieces = calloc(s->count + 1, sizeof *s->pieces);
  if (!received || !bound || !s->received || !s->pieces) { free(received); free(bound); return ENOMEM; }
  for (uint32_t k = 0; k < s->count && !status; k++) {
    const struct mesh_step *step = s->steps + k;
    bound[k] = *step;
    bound[k].peer = (uint32_t)g->members[step->peer];
    if (step->op != MESH_STEP_REDUCE) continue;
    const size_t size = step->piece.elements * step->piece.element_bytes, offset = step->first * step->piece.element_bytes % g->block;
    struct mesh_section section;
    uint32_t step_pages;
    unsigned char *memory;
    if ((status = ring(g, size, offset, &section, &step_pages, &memory))) break;
    status = mesh_section_slice(&g->context, section, offset, size, 1, step_pages, received + s->received_count);
    s->received[s->received_count++] = (struct received){memory, offset, size};
  }
  if (!status)
    status = mesh_collective_bind(&g->context, bound, s->count, identity, s->operand, received, 1, pages, s->pieces);
  free(received); free(bound);
  return status;
}

/* Steps.__call__ at invocation 0: each SEND published, each receive awaited, a REDUCE's piece combined
   into the result; a direct exchange combines into a copy (its SEND may still be reading the slot) */
static ncclResult_t steps_run(struct group *g, struct steps *s, struct call *c, double deadline, unsigned char **total, int *owned) {
  const size_t z = SIZE[c->type], bytes = elements(c) * z;
  unsigned char *own = s->slots;
  *owned = s->direct;
  *total = own;
  if (s->direct) {
    if (!(*total = malloc(bytes ? bytes : 1))) return fail(ncclSystemError, "out of memory");
    memcpy(*total, own, bytes);
  }
  uint32_t r = 0;
  for (uint32_t k = 0; k < s->count; k++) {
    const struct mesh_step *step = s->steps + k;
    if (step->op == MESH_STEP_SEND) { mesh_host_publish(&g->context, s->pieces[k], 0); continue; }
    uint64_t arrived;
    struct timespec now;
    while (!(arrived = mesh_host_arrived(&g->context, s->pieces[k], 0))) {
      clock_gettime(CLOCK_MONOTONIC, &now);
      if (now.tv_sec + now.tv_nsec * 1e-9 > deadline)
        return fail(ncclTimeout, "step %u (op %u) from rank %u: nothing landed in %.0f s", k, step->op, step->peer, DEADLINE_S);
    }
    if (arrived == UINT64_MAX) return fail(ncclRemoteError, "the link to rank %u was cancelled", step->peer);
    unsigned char *span = *total + step->first * z;
    if (step->op == MESH_STEP_REDUCE) {
      const struct received *in = s->received + r++;
      combine_into(c->type, c->combine, span, in->slots + in->offset, step->piece.elements);
    } else if (*total != own) memcpy(span, own + step->first * z, step->piece.elements * z);
  }
  return ncclSuccess;
}

static int selection(const ncclCollConfig_t *config, uint32_t *how, int *force) {
  *how = 0; *force = 1;
  if (!config) return 0;
  if (config->forceAlgSelection != NCCL_CONFIG_UNDEF_INT) *force = config->forceAlgSelection;
  const char *names = config->algSelection;
  if (!names) return 0;
  static const char *const algorithms[] = {"direct", "ring", "tree", "binomial"};
  for (const char *at = names; *at;) {
    const size_t length = strcspn(at, ",");
    int found = 0;
    for (uint32_t a = 0; a < 4; a++)
      if (length == strlen(algorithms[a]) && !strncmp(at, algorithms[a], length)) { *how |= 1u << a; found = 1; }
    if (length && !found) { fail(ncclInvalidArgument, "algSelection %s: the planner's algorithms are direct, ring, tree, binomial", names); return -1; }
    at += length + (at[length] == ',');
  }
  return 0;
}

/* _bind: every call that crosses a link bound in call order, a collective as the planner plans it, a
   point-to-point call at identity P2P + its ordinal among this rank's calls of its kind with its peer */
static ncclResult_t bind_calls(struct group *g, ncclComm_t comm, struct call *list, size_t n, int *retry) {
  const uint32_t nodes = comm->map.nodes;
  uint32_t *ordinal = calloc(2 * (size_t)comm->nranks, sizeof *ordinal);
  if (!ordinal) return fail(ncclSystemError, "out of memory");
  ncclResult_t result = ncclSuccess;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    if ((c->what == WHAT_SEND || c->what == WHAT_RECV) && c->peer == comm->rank) continue;
    if (!c->count) continue;
    struct steps *s = calloc(1, sizeof *s);
    if (!s || !(s->steps = calloc(MESH_COLLECTIVE_STEPS(nodes) + 1, sizeof *s->steps))) { free(s); result = fail(ncclSystemError, "out of memory"); break; }
    c->bound = s;
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], elements(c)};
    uint32_t identity;
    if (c->what == WHAT_SEND || c->what == WHAT_RECV) {
      uint32_t *k = ordinal + 2 * (size_t)c->peer + (c->what == WHAT_RECV);
      identity = P2P + (*k)++;
      s->steps[0] = (struct mesh_step){c->what == WHAT_SEND ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)c->peer, 0, 0, 0, operand};
      s->count = 1; s->how = -1; s->root = 0;
    } else {
      struct mesh_collective chosen = mesh_collective_choose(&comm->map, (struct mesh_collective){(uint32_t)c->what, c->how, (uint32_t)c->root, 0, NULL},
                                                             operand, comm->alpha, comm->beta);
      if (chosen.how >= MESH_UNAVAILABLE && c->how && !c->force)
        chosen = mesh_collective_choose(&comm->map, (struct mesh_collective){(uint32_t)c->what, 0, (uint32_t)c->root, 0, NULL},
                                        operand, comm->alpha, comm->beta);
      if (chosen.how >= MESH_UNAVAILABLE) {
        result = fail(ncclInvalidArgument, "no algorithm of %#x carries collective %d of %llu elements on this map", c->how, c->what,
                      (unsigned long long)operand.elements);
        break;
      }
      s->count = mesh_collective_plan(&comm->map, (uint32_t)g->rank, chosen, operand, s->steps);
      s->direct = chosen.how == MESH_DIRECT;
      s->how = (int)chosen.how; s->root = (int)chosen.root;
      identity = g->identity;
      g->identity += MESH_COLLECTIVE_STEPS(nodes);
    }
    const int status = steps_bind(g, s, c->type, elements(c), identity);
    if (status) { *retry = 1; result = fail(ncclSystemError, "binding call %zu: %s", i, strerror(status)); }
  }
  free(ordinal);
  return result;
}

/* _stage: the call's contribution into its operand in registered pages */
static void stage(const struct call *c, unsigned char *slot) {
  const size_t z = SIZE[c->type], n = c->count, r = (size_t)c->comm->rank;
  if (reducing(c->what)) {
    const size_t all = elements(c);
    if (c->premultiplied) premultiply(c->type, slot, c->send, c->pre, all);
    else memmove(slot, c->send, all * z);
  } else if (c->what == MESH_BROADCAST && c->root == c->comm->rank) memmove(slot, c->send, n * z);
  else if (c->what == MESH_ALLGATHER) memmove(slot + r * n * z, c->send, n * z);
  else if (c->what == WHAT_SEND) memmove(slot, c->send, n * z);
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

static int order(const struct call *c) { return c->what == WHAT_SEND ? 0 : c->what == WHAT_RECV ? 2 : 1; }

/* _launch: one group */
static ncclResult_t launch(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclResult_t result = ncclSuccess;
  for (size_t i = 0; i < n; i++) {
    if (list[i].comm != comm) return fail(ncclInvalidUsage, "a group on several communicators");
    if (reducing(list[i].what) && (result = reduction(list + i))) return result;
  }
  size_t remote = 0;
  for (size_t i = 0; i < n; i++)
    remote += list[i].count && !((list[i].what == WHAT_SEND || list[i].what == WHAT_RECV) && list[i].peer == comm->rank);
  struct group g = {0};
  for (int attempt = 0; comm->nranks > 1 && remote && attempt < ATTEMPTS; attempt++) {
    int status = group_open(&g, comm), retry = 0;
    if (status) {
      result = fail(ncclSystemError, "attach %s: %s", comm->region, strerror(status));
      if (attempt + 1 == ATTEMPTS) return result;
      continue;
    }
    if (g.rank != comm->rank) {
      group_close(&g, 0);
      return fail(ncclInvalidUsage, "rank %d runs on the bridge of rank %d", comm->rank, g.rank);
    }
    result = bind_calls(&g, comm, list, n, &retry);
    if (!result) {
      if ((status = mesh_transfers_prepare(&g.context, 1, 1, 1)) || (status = mesh_host_inputs(&g.context)) ||
          (status = mesh_transfers_start(&g.context))) {
        retry = 1;
        result = fail(ncclSystemError, "the group's transfers did not start: %s", strerror(status));
      }
    }
    if (!result) break;
    for (size_t i = 0; i < n; i++) { steps_free(list[i].bound); list[i].bound = NULL; }
    group_close(&g, 0);
    if (!retry || attempt + 1 == ATTEMPTS) return result;
    fprintf(stderr, "a group of %zu calls did not start (%s); again\n", n, last);
    result = ncclSuccess;
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    const size_t bytes = elements(c) * SIZE[c->type];
    if (c->what == WHAT_SEND && c->peer == comm->rank) {
      if (!(c->copy = malloc(bytes ? bytes : 1))) result = fail(ncclSystemError, "out of memory");
      else memcpy(c->copy, c->send, bytes);
    } else if (c->bound) stage(c, c->bound->slots);
    else if (c->what != WHAT_SEND && c->what != WHAT_RECV) {
      if (!(c->result = calloc(bytes ? bytes : 1, 1))) result = fail(ncclSystemError, "out of memory");
      else { c->owned = 1; stage(c, c->result); }
    }
  }
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  const double deadline = now.tv_sec + now.tv_nsec * 1e-9 + DEADLINE_S;
  for (int pass = 0; pass < 3 && !result; pass++)
    for (size_t i = 0; i < n && !result; i++)
      if (order(list + i) == pass && list[i].bound) result = steps_run(&g, list[i].bound, list + i, deadline, &list[i].result, &list[i].owned);
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
  if (g.header) group_close(&g, LINGER_S);
  return result;
}

static void plans_record(const struct call *list, size_t n) {
  free(plan_how); free(plan_root);
  plan_how = calloc(n + 1, sizeof *plan_how); plan_root = calloc(n + 1, sizeof *plan_root);
  plan_count = plan_how && plan_root ? n : 0;
  for (size_t i = 0; i < plan_count; i++) {
    plan_how[i] = list[i].bound ? list[i].bound->how : -1;
    plan_root[i] = list[i].bound ? list[i].bound->root : 0;
  }
}

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

ncclResult_t ncclGroupEnd(void) {
  if (depth <= 0) return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart");
  if (--depth) return ncclSuccess;
  struct call *list = calls;
  const size_t n = call_count;
  calls = NULL; call_count = call_capacity = 0;
  ncclResult_t result = n ? launch(list, n) : ncclSuccess;
  plans_record(list, n);
  for (size_t i = 0; i < n; i++) {
    steps_free(list[i].bound);
    if (list[i].owned) free(list[i].result);
    free(list[i].copy);
  }
  if (result && n) list[0].comm->async = result;
  free(list);
  return result;
}

ncclResult_t ncclMeshGroupPlans(int *algorithms, int *roots, int capacity, int *count) {
  if (!algorithms || !roots || !count || capacity < 0) return fail(ncclInvalidArgument, "ncclMeshGroupPlans' arguments");
  const size_t k = plan_count < (size_t)capacity ? plan_count : (size_t)capacity;
  for (size_t i = 0; i < k; i++) { algorithms[i] = plan_how[i]; roots[i] = plan_root[i]; }
  *count = (int)k;
  return ncclSuccess;
}

static ncclResult_t enqueue(struct call c) {
  if (!c.comm) return fail(ncclInvalidArgument, "comm is NULL");
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
                               int root, ncclComm_t comm, const ncclCollConfig_t *config) {
  struct call c = {.what = what, .root = root, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype, .op = op, .comm = comm};
  if (selection(config, &c.how, &c.force)) return ncclInvalidArgument;
  return enqueue(c);
}

ncclResult_t ncclAllReduceConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)stream;
  return collective(MESH_ALLREDUCE, sendbuff, recvbuff, count, datatype, op, 0, comm, config);
}
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  return ncclAllReduceConfig(sendbuff, recvbuff, count, datatype, op, comm, stream, NULL);
}

ncclResult_t ncclReduceConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
                              ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)stream;
  return collective(MESH_REDUCE, sendbuff, recvbuff, count, datatype, op, root, comm, config);
}
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  return ncclReduceConfig(sendbuff, recvbuff, count, datatype, op, root, comm, stream, NULL);
}

ncclResult_t ncclBroadcastConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)stream;
  return collective(MESH_BROADCAST, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, config);
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
  (void)stream;
  return collective(MESH_REDUCE_SCATTER, sendbuff, recvbuff, recvcount, datatype, op, 0, comm, config);
}
ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  return ncclReduceScatterConfig(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, NULL);
}

ncclResult_t ncclAllGatherConfig(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)stream;
  return collective(MESH_ALLGATHER, sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, config);
}
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  return ncclAllGatherConfig(sendbuff, recvbuff, sendcount, datatype, comm, stream, NULL);
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  (void)stream;
  return enqueue((struct call){.what = WHAT_SEND, .peer = peer, .send = sendbuff, .count = count, .type = datatype, .comm = comm});
}
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  (void)stream;
  return enqueue((struct call){.what = WHAT_RECV, .peer = peer, .recv = recvbuff, .count = count, .type = datatype, .comm = comm});
}

/* NCCL 2.32's own lowering of these three (src/enqueue/task_prep/task_classify.cc
   classifyCollToP2pTasks): each rank's sends and receives in one group */
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (!comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "comm or datatype");
  const size_t z = count * SIZE[datatype];
  ncclResult_t result = ncclGroupStart();
  for (int r = 0; r < comm->nranks && !result; r++) {
    result = ncclSend((const unsigned char *)sendbuff + r * z, count, datatype, r, comm, stream);
    if (!result) result = ncclRecv((unsigned char *)recvbuff + r * z, count, datatype, r, comm, stream);
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
  ncclResult_t result = ncclGroupStart();
  if (!result) result = ncclSend(sendbuff, count, datatype, root, comm, stream);
  for (int r = 0; comm->rank == root && r < comm->nranks && !result; r++)
    result = ncclRecv((unsigned char *)recvbuff + r * z, count, datatype, r, comm, stream);
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
  ncclResult_t result = ncclGroupStart();
  for (int r = 0; comm->rank == root && r < comm->nranks && !result; r++)
    result = ncclSend((const unsigned char *)sendbuff + r * z, count, datatype, r, comm, stream);
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
   then rank; the new map is comm's links among them, relabelled: the same kind where every rank stays, a
   mesh where every pair is linked (one rank alone) */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  if (depth) return fail(ncclInvalidUsage, "ncclCommSplit inside a group");
  const int n = comm->nranks;
  int64_t mine[2] = {color, key}, *every = calloc(2 * (size_t)n, sizeof *every);
  int *ranks = calloc((size_t)n, sizeof *ranks), *index = calloc((size_t)n, sizeof *index), *members = calloc((size_t)n, sizeof *members);
  uint32_t (*linked)[2] = calloc(comm->map.links + 1, sizeof *linked);
  ncclResult_t result = !every || !ranks || !index || !members || !linked ? fail(ncclSystemError, "out of memory")
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
      if (a >= 0 && b >= 0) { linked[links][0] = (uint32_t)a; linked[links][1] = (uint32_t)b; links++; }
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
  checked:
    if (count == n && comm->map.kind != MESH_LINKS_MESH)
      result = comm_make(newcomm, comm->map.kind, count, (const uint32_t (*)[2])linked, links, members, index[comm->rank], comm->region,
                         comm->alpha, comm->beta);
    else if (every_pair)
      result = comm_make(newcomm, MESH_LINKS_MESH, count, NULL, 0, members, index[comm->rank], comm->region, comm->alpha, comm->beta);
    else result = fail(ncclInvalidUsage, "a split to some ranks that are not every pair linked");
  }
  free(every); free(ranks); free(index); free(members); free(linked);
  return result;
}

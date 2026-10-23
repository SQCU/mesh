#include "nccl.h"
#include "mesh-call.h"
#include "mesh-plan.h"
#include "mesh.h"
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
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <poll.h>
#include <fcntl.h>
#include <mach/mach_time.h>

enum { WHAT_SEND = 16, WHAT_RECV, WHAT_COUNTED };
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN, COMBINE_MERGE };

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};

struct said { uint64_t id; char text[512]; };
static _Atomic uint64_t messages;

static _Thread_local struct said last;

static ncclResult_t fail(ncclResult_t result, const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(last.text, sizeof last.text, format, arguments);
  va_end(arguments);
  last.id = 0;
  return result;
}

static ncclResult_t told(ncclResult_t result, const struct said *said) {
  if (last.id != said->id) last = *said;
  return result;
}

static uint64_t keep(void) { return atomic_fetch_add_explicit(&messages, 1, memory_order_relaxed) + 1; }

static float bf16_to(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t bf16_from(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

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

static const float PARTIAL_NONE = -3.4028234663852886e38f;
static void merge_rows(float *x, const float *y, size_t rows, size_t unit) {
  for (size_t r = 0; r < rows; r++, x += unit, y += unit) {
    const int mine = x[1] > 0, theirs = y[1] > 0;
    if (mine && !theirs) continue;
    if (!mine || !theirs) {
      for (size_t j = 0; j < unit; j++) x[j] = theirs ? y[j] : j ? 0.0f : PARTIAL_NONE;
      continue;
    }
    const float top = x[0] > y[0] ? x[0] : y[0], a = expf(x[0] - top), b = expf(y[0] - top);
    x[0] = top;
    for (size_t j = 1; j < unit; j++) x[j] = (a > 0 ? x[j] * a : 0.0f) + (b > 0 ? y[j] * b : 0.0f);
  }
}

static void combine_into(ncclDataType_t t, int combine, void *dst, const void *src, size_t n) {
  if ((int)t == METAL_MERGE) { merge_rows(dst, src, n, (size_t)combine); return; }
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

struct redop { int used; ncclDataType_t type; unsigned char value[8]; uint32_t unit; const void *at; };

struct entry { uint64_t below; struct mesh_program program; };

struct channel {
  int peer;
  uint32_t node, stream;
  const uint64_t *wire;
  struct mesh_stream_view view;
  void *buffer;
  size_t ready, sent, landed;
  uint64_t sends, lands, published, consumed, waited;
  mesh_pages out, in, sent_waited;
  int64_t differ[2];
};

struct attach;
struct plan;

#define GROUP_TIMES 65536u
struct ncclComm {
  struct mesh_link_map map;
  struct mesh_routing routing, table;
  int routed, taken;
  uint64_t page;
  struct mesh_trees paths, trees;
  struct entry *entries, *given;
  ncclMeshForm *forms;
  uint32_t programs, splits, *members, form_count, count, nops;
  int nranks, rank, listed, compiled, binding, touched, sweeps, unpublished;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  struct redop *ops;
  ncclResult_t async, failed;
  pthread_mutex_t guard;
  int revoked, *peers, initing, *toward;
  double until;
  pthread_t init;
  struct said why, failure;
  struct attach *attach;
  uint32_t held;
  struct channel *channels;
  struct plan *plans;
  struct ncclMeshRequest *freed;
  _Atomic uint64_t *words;
  void *violation;
  uint64_t last, groups, (*timed)[2];
  void *command;
  struct ncclComm *next_touched;
  _Atomic int runs;
};

struct where { void *buffer; size_t offset; };
struct row_chunk { void *pages, *buffer; size_t offset, bytes; };
struct metal_rows { struct ncclComm *comm; struct row_chunk *chunk; size_t chunks, at, used; };
struct plan {
  struct call *list;
  size_t n;
  struct schedule *plans;
  struct metal_rows rows;
  void *pages, *buffer;
  struct where base;
  size_t word;
  uint32_t reader;
  int flight, leaving, stale;
  struct plan *next;
};

static pthread_mutex_t group_lock = PTHREAD_MUTEX_INITIALIZER;
static void *finished;
static uint64_t issued;

static void program_free(struct mesh_program *program) {
  free((void *)program->segment); free((void *)program->log_weight); free((void *)program->first); free((void *)program->move);
}

static void trees_free(struct mesh_trees *trees) {
  free((void *)trees->root); free((void *)trees->parent); free((void *)trees->log_weight);
  *trees = (struct mesh_trees){0};
}

static int trees_copy(struct mesh_trees *made, const struct mesh_trees *from) {
  uint32_t *root = calloc(from->count + 1, sizeof *root), *parent = calloc((size_t)from->count * from->nodes + 1, sizeof *parent);
  double *weight = calloc(from->count + 1, sizeof *weight);
  *made = (struct mesh_trees){from->nodes, from->count, root, parent, weight};
  if (!root || !parent || !weight) return ENOMEM;
  if (from->count) {
    memcpy(root, from->root, from->count * sizeof *root); memcpy(weight, from->log_weight, from->count * sizeof *weight);
    memcpy(parent, from->parent, (size_t)from->count * from->nodes * sizeof *parent);
  }
  return 0;
}

static void map_free(struct mesh_link_map *map) {
  free(map->link); free(map->bandwidth); free(map->next); free(map->mark);
  *map = (struct mesh_link_map){0};
}

static void routing_free(struct mesh_routing *routing) {
  free((void *)routing->pair); free((void *)routing->start); free((void *)routing->node); free((void *)routing->share);
  *routing = (struct mesh_routing){0};
}

static int routing_room(struct mesh_routing *made, uint32_t nodes, uint32_t routes, uint32_t hops) {
  uint32_t *pair = calloc((size_t)nodes * nodes + 1, sizeof *pair), *start = calloc((size_t)routes + 1, sizeof *start),
           *node = calloc((size_t)hops + 1, sizeof *node);
  double *share = calloc((size_t)routes + 1, sizeof *share);
  *made = (struct mesh_routing){nodes, routes, pair, start, node, share};
  if (pair && start && node && share) return 0;
  routing_free(made);
  return ENOMEM;
}

static int routing_copy(struct mesh_routing *made, const struct mesh_routing *from) {
  const uint32_t n = from->nodes, hops = from->routes ? from->start[from->routes] : 0;
  if (routing_room(made, n, from->routes, hops)) return ENOMEM;
  memcpy((void *)made->pair, from->pair, ((size_t)n * n + 1) * sizeof *from->pair);
  memcpy((void *)made->start, from->start, ((size_t)from->routes + 1) * sizeof *from->start);
  if (hops) memcpy((void *)made->node, from->node, hops * sizeof *from->node);
  if (from->routes) memcpy((void *)made->share, from->share, from->routes * sizeof *from->share);
  return 0;
}

static int comm_routed(struct ncclComm *comm) {
  if (comm->routed) return 0;
  const uint32_t n = (uint32_t)comm->nranks;
  const int mapped = comm->map.nodes == n && comm->map.mark;
  uint32_t routes, hops;
  mesh_routing_room(n, n, &routes, &hops);
  const int fits = mapped && comm->table.pair ? mesh_routing_check(&comm->map, &comm->table) : EINVAL;
  comm->taken = !fits;
  int status = fits == ENOMEM ? ENOMEM : comm->taken ? routing_copy(&comm->routing, &comm->table)
                                                      : routing_room(&comm->routing, n, mapped ? routes : 0, mapped ? hops : 0);
  if (!status && !comm->taken && mapped &&
      mesh_routing_pack(&comm->map, 0, n, &comm->routing, (uint32_t *)comm->routing.pair, (uint32_t *)comm->routing.start,
                        (uint32_t *)comm->routing.node, (double *)comm->routing.share) == UINT32_MAX)
    status = ENOMEM;
  if (!status && !comm->taken && mapped) {
    void *start = realloc((void *)comm->routing.start, ((size_t)comm->routing.routes + 1) * sizeof *comm->routing.start);
    void *share = realloc((void *)comm->routing.share, ((size_t)comm->routing.routes + 1) * sizeof *comm->routing.share);
    if (start) comm->routing.start = start;
    if (share) comm->routing.share = share;
    void *node = realloc((void *)comm->routing.node, ((size_t)comm->routing.start[comm->routing.routes] + 1) * sizeof *comm->routing.node);
    if (node) comm->routing.node = node;
  }
  if (status) routing_free(&comm->routing);
  comm->routed = !status;
  return status;
}

static void routes_free(struct ncclComm *comm, int all) {
  trees_free(&comm->trees);
  comm->compiled = 0;
  for (uint32_t e = 0; e < comm->programs; e++) program_free(&comm->entries[e].program);
  free(comm->entries);
  comm->entries = NULL; comm->programs = 0;
  if (!all) return;
  routing_free(&comm->routing); routing_free(&comm->table);
  comm->routed = 0;
  trees_free(&comm->paths); map_free(&comm->map);
  for (int e = 0; e < comm->listed; e++) program_free(&comm->given[e].program);
  free(comm->given);
  comm->given = NULL; comm->listed = 0;
}

static void bootstrap_close(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) close(comm->peers[r]);
  free(comm->peers);
  comm->peers = NULL;
  pthread_mutex_unlock(&comm->guard);
}

static void comm_leave(struct ncclComm *comm);
static void sweep(struct ncclComm *comm, int all, int end);

static void comm_free(struct ncclComm *comm) {
  bootstrap_close(comm);
  routes_free(comm, 1);
  for (uint32_t h = 0; h < comm->count; h++) metal_release(comm->channels[h].buffer);
  if (comm->violation) metal_release(comm->violation);
  metal_release(comm->command);
  comm_leave(comm);
  free(comm->channels); free(comm->toward); free(comm->timed);
  free(comm->ops);
  pthread_mutex_destroy(&comm->guard);
  free(comm->members); free(comm->forms);
  free(comm);
}

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

static int compile_entry(struct ncclComm *comm, const struct mesh_trees *trees, uint32_t what, uint32_t root, int whole, uint32_t chunks,
                         uint64_t below) {
  uint32_t parts, moves;
  mesh_compile_room(trees, chunks, &parts, &moves);
  uint32_t *segment = malloc((parts + 1) * sizeof *segment), *first = malloc((trees->nodes + 1) * sizeof *first);
  double *weight = malloc((parts + 1) * sizeof *weight);
  struct mesh_move *move = malloc((moves + 1) * sizeof *move);
  struct entry *e = comm->entries + comm->programs;
  if (!segment || !first || !weight || !move || mesh_compile(trees, what, root, whole, chunks, NULL, &e->program, segment, weight, first, move)) {
    free(segment); free(first); free(weight); free(move);
    return EINVAL;
  }
  e->below = below;
  comm->programs++;
  return 0;
}

static int has_root(int what) { return what == MESH_REDUCE || what == MESH_BROADCAST; }

static int root_carried(const struct mesh_trees *trees, uint32_t r) {
  for (uint32_t t = 0; t < trees->count; t++)
    if (trees->root[t] == r && trees->log_weight[t] > -INFINITY) return 1;
  return 0;
}

static ncclResult_t compile_form(struct ncclComm *comm, const struct mesh_trees *along, const ncclMeshForm *x) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_trees packed = {n, 0, NULL, NULL, NULL}, weighed = {n, 0, NULL, NULL, NULL};
  const struct mesh_trees *trees = along;
  ncclResult_t result = ncclSuccess;

  if (x->trees != ncclMeshTreesPacked && x->shares) {
    if (trees_copy(&weighed, along)) result = fail(ncclSystemError, "out of memory");
    for (uint32_t t = 0; !result && t < weighed.count; t++) {
      uint32_t k = 0;
      for (uint32_t u = 0; u < t; u++) k += weighed.root[u] == weighed.root[t];
      ((double *)weighed.log_weight)[t] = k < x->most ? log(x->shares[(size_t)weighed.root[t] * x->most + k]) : -INFINITY;
    }
    trees = &weighed;
  }
  if (x->trees == ncclMeshTreesPacked) {
    uint32_t *root = calloc((size_t)n * x->most + 1, sizeof *root), *parent = calloc((size_t)n * n * x->most + 1, sizeof *parent);
    double *weight = calloc((size_t)n * x->most + 1, sizeof *weight);
    packed = (struct mesh_trees){n, 0, root, parent, weight};
    if (!root || !parent || !weight) result = fail(ncclSystemError, "out of memory");
    else if (!(packed.count = mesh_trees_pack(&comm->map, x->eps, x->most, root, parent, weight)))
      result = fail(ncclInvalidArgument, "the map's links pack no trees (mesh_trees_pack)");
    for (uint32_t t = 0, k = 0; t < packed.count; t++) {
      k = t && root[t] == root[t - 1] ? k + 1 : 0;
      weight[t] = x->shares ? log(x->shares[(size_t)root[t] * x->most + k]) : 0;
    }
    trees = &packed;
  }
  for (uint32_t r = 0; !result && r < (has_root(x->collective) && x->root < 0 ? n : 1); r++) {
    const uint32_t root = has_root(x->collective) ? (x->root < 0 ? r : (uint32_t)x->root) : 0;
    if (root >= n) continue;

    const int weighs = x->shares && !(x->collective == MESH_ALLREDUCE && x->whole);
    for (uint32_t q = has_root(x->collective) ? root : 0; !result && weighs && q < (has_root(x->collective) ? root + 1 : n); q++)
      if (!root_carried(trees, q))
        result = fail(ncclInvalidArgument, "the configured form of collective %d: every tree of root %u takes share 0", x->collective, q);
    if (!result && compile_entry(comm, trees, (uint32_t)x->collective, root, x->whole, x->chunks, x->below))
      result = fail(ncclInternalError, "the configured form of collective %d at root %u does not compile", x->collective, root);
  }
  trees_free(&packed);
  trees_free(&weighed);
  return result;
}

static ncclResult_t comm_routes(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  if (n < 2) return ncclSuccess;
  const struct mesh_trees *along = &comm->paths;
  if (!comm->listed || comm->form_count) {
    uint32_t apart[2] = {0, 0};
    const int joined = mesh_link_map_joined(&comm->map, apart);
    if (joined == ENOTCONN)
      return fail(ncclInvalidUsage, "rank %u (node %u) and rank %u (node %u) of %s are joined by no path of links between its members: the "
                  "bridges forward nothing, so a route through a node outside the communicator is refused", apart[0], comm->members[apart[0]],
                  apart[1], comm->members[apart[1]], comm->name);
    if (joined) return fail(ncclInternalError, "the members' map of %s: %s", comm->name, strerror(joined));
    if (comm->paths.count && mesh_trees_check(&comm->map, &comm->paths))
      return fail(ncclInvalidArgument, "the paths are not trees of the links of the members' map (mesh_trees_check)");
    if (!comm->paths.count) {
      uint32_t *root = calloc(n + 1, sizeof *root), *parent = calloc((size_t)n * n + 1, sizeof *parent);
      double *weight = calloc(n + 1, sizeof *weight);
      comm->trees = (struct mesh_trees){n, 0, root, parent, weight};
      if (!root || !parent || !weight) return fail(ncclSystemError, "out of memory");
      if (!(comm->trees.count = mesh_trees_shortest(&comm->map, root, parent, weight)))
        return fail(ncclInvalidArgument, "the map's routes do not join every rank (mesh_trees_shortest)");
      along = &comm->trees;
    }
  }
  size_t room = comm->listed ? (size_t)comm->listed : 3 + 2 * (size_t)n;
  for (uint32_t f = 0; f < comm->form_count; f++) room += has_root(comm->forms[f].collective) && comm->forms[f].root < 0 ? n : 1;
  if (!(comm->entries = calloc(room + 1, sizeof *comm->entries))) return fail(ncclSystemError, "out of memory");
  for (uint32_t f = 0; f < comm->form_count; f++) {
    const ncclResult_t result = compile_form(comm, along, comm->forms + f);
    if (result) return result;
  }
  for (int e = 0; e < comm->listed; e++) {
    if (program_copy(&comm->entries[comm->programs].program, &comm->given[e].program)) return fail(ncclSystemError, "out of memory");
    comm->entries[comm->programs++].below = comm->given[e].below;
  }
  if (comm->listed) return ncclSuccess;
  if (compile_entry(comm, along, MESH_ALLREDUCE, 0, 1, 1, 0) || compile_entry(comm, along, MESH_REDUCE_SCATTER, 0, 0, 1, 0) ||
      compile_entry(comm, along, MESH_ALLGATHER, 0, 0, 1, 0))
    return fail(ncclInternalError, "the all-reduce, reduce-scatter or all-gather along the communicator's trees does not compile");
  for (uint32_t r = 0; r < n; r++)
    if (compile_entry(comm, along, MESH_REDUCE, r, 0, 1, 0) || compile_entry(comm, along, MESH_BROADCAST, r, 0, 1, 0))
      return fail(ncclInternalError, "the reduce or broadcast of root %u along the communicator's trees does not compile", r);
  return ncclSuccess;
}

static ncclResult_t comm_compile(struct ncclComm *comm) {
  if (comm->compiled) return ncclSuccess;
  ncclResult_t result = comm_routes(comm);
  for (uint32_t e = 0; !result && e < comm->programs; e++) {
    const struct mesh_program *x = &comm->entries[e].program;
    for (uint32_t r = 0; !result && r < x->nodes; r++)
      for (uint32_t m = x->first[r]; !result && m < x->first[r + 1]; m++)
        if (!mesh_link_between(&comm->map, r, x->move[m].peer))
          result = fail(ncclInvalidUsage, "program %u of %s moves between ranks %u and %u, which no link joins: a program's move crosses one "
                        "link", e, comm->name, r, x->move[m].peer);
  }
  if (result) { routes_free(comm, 0); return result; }
  comm->compiled = 1;
  return ncclSuccess;
}

static ncclMeshForm *forms_copy(const ncclMeshForm *forms, uint32_t count, uint32_t nranks) {
  size_t shares = 0;
  for (uint32_t f = 0; f < count; f++) shares += forms[f].shares ? (size_t)nranks * forms[f].most : 0;
  ncclMeshForm *made = malloc(count * sizeof *made + shares * sizeof(double) + 1);
  double *at = made ? (double *)(made + count) : NULL;
  for (uint32_t f = 0; made && f < count; f++) {
    made[f] = forms[f];
    if (!forms[f].shares) continue;
    memcpy(at, forms[f].shares, (size_t)nranks * forms[f].most * sizeof *at);
    made[f].shares = at;
    at += (size_t)nranks * forms[f].most;
  }
  return made;
}

ncclResult_t ncclMeshCommSetRouting(ncclComm_t comm, const struct mesh_routing *table) {
  if (!comm || (table && (table->nodes != (uint32_t)comm->nranks || !table->pair || !table->start || (table->routes && (!table->node || !table->share)))))
    return fail(ncclInvalidArgument, "ncclMeshCommSetRouting: a table of the communicator's %d ranks, its pairs and its routes' starts, and their "
                "nodes and shares where it has routes", comm ? comm->nranks : 0);
  struct mesh_routing made = {0};
  if (table && routing_copy(&made, table)) return fail(ncclSystemError, "out of memory");
  pthread_mutex_lock(&group_lock);
  routing_free(&comm->table);
  routing_free(&comm->routing);
  comm->table = made;
  comm->routed = comm->taken = 0;
  for (struct plan *p = comm->plans; p; p = p->next) p->stale = comm->sweeps = 1;
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

ncclResult_t ncclMeshCommGetRouting(ncclComm_t comm, struct mesh_routing *table, int *given) {
  if (!comm || !table) return fail(ncclInvalidArgument, "comm or table is NULL");
  pthread_mutex_lock(&group_lock);
  int status = comm_routed(comm);
  if (!status) status = routing_copy(table, &comm->routing);
  if (given) *given = comm->taken;
  pthread_mutex_unlock(&group_lock);
  return status ? fail(ncclSystemError, "out of memory") : ncclSuccess;
}

ncclResult_t ncclMeshRoutingFree(struct mesh_routing *table) {
  if (table) routing_free(table);
  return ncclSuccess;
}

ncclResult_t ncclMeshCommForms(ncclComm_t comm, const ncclMeshForm *forms, int count) {
  if (!comm || count < 0 || (count && !forms)) return fail(ncclInvalidArgument, "ncclMeshCommForms's arguments");
  for (int f = 0; f < count; f++) {
    const ncclMeshForm *x = forms + f;
    int shared = !x->shares || x->most;
    for (size_t k = 0; x->shares && shared && k < (size_t)comm->nranks * x->most; k++) shared = x->shares[k] >= 0 && isfinite(x->shares[k]);
    if (x->collective < MESH_ALLREDUCE || x->collective > MESH_ALLGATHER || (has_root(x->collective) && x->root < -1) || !x->chunks || !shared ||
        (x->trees != ncclMeshTreesShortest && x->trees != ncclMeshTreesPacked) || (x->trees == ncclMeshTreesPacked && (!x->most || !(x->eps > 0))))
      return fail(ncclInvalidArgument, "form %d: a collective (mesh-plan.h MESH_*), a rooted one's root (-1 every root), its trees "
                  "(shortest, or packed with eps > 0 and most > 0), shares of 0 or more (most of them a root) or none, and chunks > 0", f);
  }
  ncclMeshForm *made = count ? forms_copy(forms, (uint32_t)count, (uint32_t)comm->nranks) : NULL;
  if (count && !made) return fail(ncclSystemError, "out of memory");
  pthread_mutex_lock(&group_lock);
  free(comm->forms);
  comm->forms = made; comm->form_count = (uint32_t)count;
  routes_free(comm, 0);
  for (struct plan *p = comm->plans; p; p = p->next) p->stale = comm->sweeps = 1;
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

static const struct mesh_program *program_of(ncclComm_t comm, int what, int root, uint64_t bytes) {
  for (uint32_t e = 0; e < comm->programs; e++) {
    const struct entry *x = comm->entries + e;
    if ((int)x->program.what != what || ((what == MESH_REDUCE || what == MESH_BROADCAST) && (int)x->program.root != root)) continue;
    if (!x->below || bytes < x->below) return &x->program;
  }
  return NULL;
}

struct lease { void *address; uint32_t row; size_t bytes; void *buffer; };

struct hold { uint32_t node, stream, users, padding; struct mesh_stream_view view; uint64_t wire; };
struct attach {
  struct mesh_ctx context;
  struct lease *leases;
  size_t count, room;
  uint64_t comms;

  _Atomic int wrapped;
  struct hold holds[MESH_STREAMS];
  struct attach *next;
};

static struct attach *attaches;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;

static int attach_expired(const struct attach *a) {
  return __atomic_load_n(&a->context.M->version, __ATOMIC_ACQUIRE) != MESH_LAYOUT;
}

static void attach_settle(void) {
  struct attach *a = attaches;
  if (!a || a->count || a->comms || atomic_load(&a->wrapped) || !attach_expired(a)) return;
  mesh_detach(&a->context);
  free(a->leases);
  free(a);
  attaches = NULL;
}

static int attach_current(struct attach **out) {
  attach_settle();
  if (attaches) { *out = attaches; return attach_expired(attaches) ? ESTALE : 0; }
  struct attach *a = calloc(1, sizeof *a);
  if (!a) return ENOMEM;
  const int status = mesh_attach(&a->context);
  if (status) { free(a); return status; }
  a->next = attaches;
  attaches = a;
  *out = a;
  return 0;
}

static int lease_room(struct attach *a) {
  const size_t room = a->room ? 2 * a->room : 64;
  struct lease *grown = realloc(a->leases, room * sizeof *grown);
  if (!grown) return ENOMEM;
  a->leases = grown;
  a->room = room;
  return 0;
}

static ncclResult_t attach_fail(int status, const char *format, ...) {
  char what[256];
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(what, sizeof what, format, arguments);
  va_end(arguments);
  if (status == EBUSY) return fail(ncclSystemError, "%s: every client slot of this node's bridge is held (mesh.h MESH_CLIENTS)", what);
  if (status == ESTALE)
    return fail(ncclRemoteError, "%s: the region of this node's bridge was released or replaced while this process holds leases "
                "or communicators of it; a process that held a communicator or a lease's Metal buffer ends, any other frees its "
                "leases (ncclMemFree), and its next lease or communicator then attaches the bridge's next region "
                "(ncclMeshRegionExpired; design/transport.md section 2)", what);
  return fail(ncclSystemError, "%s: this process attaches no region of this node's bridge "
              "(%s); the node's bridge must run, with a configuration whose region holds this client's footprint "
              "(etc/bridge/README.md)", what, strerror(status));
}

static struct ncclComm *comm_new(int nranks, int rank, const char *name) {
  struct ncclComm *comm = calloc(1, sizeof *comm);
  if (comm && (!(comm->members = calloc((size_t)nranks, sizeof *comm->members)) || !(comm->toward = malloc((size_t)nranks * sizeof *comm->toward)))) {
    free(comm->members); free(comm); comm = NULL;
  }
  if (!comm) return NULL;
  comm->nranks = nranks; comm->rank = rank;
  for (int r = 0; r < nranks; r++) comm->toward[r] = -1;
  pthread_mutex_init(&comm->guard, NULL);
  snprintf(comm->name, sizeof comm->name, "%s", name);
  return comm;
}

static int comm_attach(struct ncclComm *comm, const uint32_t *members) {
  pthread_mutex_lock(&attach_lock);
  const int status = attach_current(&comm->attach);
  if (!status) comm->attach->comms++;
  else comm->attach = NULL;
  pthread_mutex_unlock(&attach_lock);
  if (!status && members) memcpy(comm->members, members, (size_t)comm->nranks * sizeof *members);
  return status;
}

static ncclResult_t comm_refused(int status, const char *name) {
  return status == ENOMEM ? fail(ncclSystemError, "out of memory") : attach_fail(status, "communicator %s", name);
}

static void comm_let_go(struct ncclComm *comm) {
  if (!comm->binding) return;
  comm->binding = 0;
  for (uint32_t k = 0; k < MESH_STREAMS; k++) {
    struct hold *h = comm->attach->holds + k;
    if (!(comm->held >> k & 1) || !h->users || --h->users) continue;
    if (mesh_stream_held(&comm->attach->context, h->stream, h->wire)) mesh_stream_give(&comm->attach->context, h->stream);
    *h = (struct hold){0};
  }
  comm->held = 0;
}

static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) comm_let_go(comm);
  pthread_mutex_unlock(&attach_lock);
  sweep(comm, 1, 1);
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) {
    comm->attach->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
}

struct report {
  int32_t rank, status;
  uint32_t node, count, frees, padding;
  struct mesh_reach reach[MESH_STREAMS];
  uint64_t posted[MESH_STREAMS][2], free_wire[MESH_STREAMS];
  uint32_t free_node[MESH_STREAMS];
};

static uint32_t reach_of(const struct report *x, uint32_t node) {
  uint32_t k = 0;
  while (k < x->count && k < MESH_STREAMS && x->reach[k].node != node) k++;
  return k;
}

static uint32_t joined(const struct report *reports, uint32_t a, uint32_t b, uint64_t *bandwidth) {
  const uint32_t j = reach_of(reports + a, reports[b].node), k = reach_of(reports + b, reports[a].node);
  *bandwidth = 0;
  if (j >= reports[a].count || k >= reports[b].count) return 0;
  const uint64_t one = reports[a].reach[j].bandwidth, other = reports[b].reach[k].bandwidth;
  *bandwidth = one && other ? (one < other ? one : other) : 0;
  return 1;
}

static int map_of(struct ncclComm *comm, const struct report *reports) {
  const uint32_t n = (uint32_t)comm->nranks;
  uint32_t links = 0;
  uint64_t bandwidth;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++) links += joined(reports, a, b, &bandwidth);
  struct mesh_link_map *map = &comm->map;
  *map = (struct mesh_link_map){MESH_LINKS_GRAPH, n, links, calloc(links + 1, sizeof *map->link), calloc(links + 1, sizeof *map->bandwidth), 0,
                                calloc((size_t)n * n + 1, sizeof *map->next), calloc(n + 1, sizeof *map->mark)};
  if (!map->link || !map->bandwidth || !map->next || !map->mark) { map_free(map); return ENOMEM; }
  uint32_t l = 0;
  for (uint32_t a = 0; a < n; a++) {
    map->mark[a] = reports[a].node;
    for (uint32_t b = a + 1; b < n; b++)
      if (joined(reports, a, b, &bandwidth)) {
        map->link[l][0] = a; map->link[l][1] = b; map->bandwidth[l] = bandwidth; l++;
      }
  }
  const int status = mesh_routes(map, map->next);
  if (status) map_free(map);
  return status;
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

const char *ncclGetLastError(ncclComm_t comm) { (void)comm; return last.text; }

static struct { int fd; char id[NCCL_UNIQUE_ID_BYTES]; } root = {-1, {0}};

static int bootstrap_listen(const char *host, uint16_t *port) {
  struct sockaddr_in6 at = {.sin6_len = sizeof at, .sin6_family = AF_INET6, .sin6_port = htons(*port)};
  socklen_t length = sizeof at;
  int on = 1, fd = socket(AF_INET6, SOCK_STREAM, 0);
  if (inet_pton(AF_INET6, host, &at.sin6_addr) != 1) at.sin6_addr = in6addr_any;
  if (fd >= 0 && !setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on) && !bind(fd, (struct sockaddr *)&at, sizeof at) &&
      !listen(fd, 64) && !getsockname(fd, (struct sockaddr *)&at, &length) && !fcntl(fd, F_SETFL, O_NONBLOCK)) {
    *port = ntohs(at.sin6_port);
    return fd;
  }
  const int error = errno;
  if (fd >= 0) close(fd);
  errno = error;
  return -1;
}

static ncclResult_t fabric_address(char *address, size_t room) {
  if (!address || !room) return fail(ncclInvalidArgument, "the fabric address: no room");
  *address = 0;
  struct ifaddrs *interfaces = NULL;
  if (getifaddrs(&interfaces)) return fail(ncclSystemError, "the node's interfaces (getifaddrs): %s", strerror(errno));
  int short_room = 0;
  for (struct ifaddrs *i = interfaces; i && !*address && !short_room; i = i->ifa_next) {
    char text[INET6_ADDRSTRLEN];
    if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET6 || (i->ifa_flags & IFF_LOOPBACK) ||
        !inet_ntop(AF_INET6, &((struct sockaddr_in6 *)i->ifa_addr)->sin6_addr, text, sizeof text) ||
        strncmp(text, MESH_ULA_PREFIX, strlen(MESH_ULA_PREFIX)))
      continue;
    if (strlen(text) < room) snprintf(address, room, "%s", text);
    else short_room = 1;
  }
  freeifaddrs(interfaces);
  if (short_room) return fail(ncclInvalidArgument, "the fabric address: %zu bytes hold no address of the fabric's", room);
  return *address ? ncclSuccess
                  : fail(ncclSystemError, "this node holds no fabric address (no interface's address under %s: bin/mesh-router-init.sh)", MESH_ULA_PREFIX);
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  memset(id->internal, 0, sizeof id->internal);
  const char *given = getenv("NCCL_COMM_ID");
  if (given && *given) { snprintf(id->internal, sizeof id->internal, "%s", given); return ncclSuccess; }
  char host[256] = {0};
  if (fabric_address(host, sizeof host)) *host = 0;
  uint16_t port = 0;
  if (root.fd >= 0) close(root.fd);
  if ((root.fd = *host || !gethostname(host, sizeof host - 1) ? bootstrap_listen(host, &port) : -1) < 0)
    return fail(ncclSystemError, "the bootstrap root's listener: %s", strerror(errno));
  snprintf(id->internal, sizeof id->internal, "%s:%u", host, port);
  memcpy(root.id, id->internal, sizeof root.id);
  return ncclSuccess;
}

static int revoked(ncclComm_t comm) { return __atomic_load_n(&comm->revoked, __ATOMIC_SEQ_CST); }
static double now_s(void);

#define BOOTSTRAP_SLICE_MS 100

static int whole(struct ncclComm *comm, int fd, void *bytes, size_t size, int sending) {
  for (size_t done = 0; done < size;) {
    const ssize_t moved = sending ? send(fd, (char *)bytes + done, size - done, 0) : recv(fd, (char *)bytes + done, size - done, 0);
    if (moved < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      if (revoked(comm)) return ECANCELED;
      if (now_s() >= comm->until) return ETIMEDOUT;
      continue;
    }
    if (moved <= 0) return moved ? errno : ECONNRESET;
    done += (size_t)moved;
  }
  return 0;
}
static ncclResult_t revoked_fail(ncclComm_t comm) {
  return fail(ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
}

static void bootstrap_put(struct ncclComm *comm, int r, int fd) {
  const struct timeval slice = {0, BOOTSTRAP_SLICE_MS * 1000};
  if (fd >= 0) {
    fcntl(fd, F_SETFL, 0);
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &(int){1}, sizeof(int));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &slice, sizeof slice);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &slice, sizeof slice);
  }
  pthread_mutex_lock(&comm->guard);
  if (fd < 0 && comm->peers[r] >= 0) close(comm->peers[r]);
  comm->peers[r] = fd;
  pthread_mutex_unlock(&comm->guard);
}

static int bootstrap_open(struct ncclComm *comm, const char *id, struct report *reports) {
  int *peers = comm->peers, status = 0;
  char host[NCCL_UNIQUE_ID_BYTES];
  snprintf(host, sizeof host, "%s", id);
  char *colon = strrchr(host, ':');
  if (!colon) return EINVAL;
  *colon = 0;
  const double bound = (double)MESH_PLATFORM_NS / 1e9;
  comm->until = now_s() + bound;
  if (comm->rank) {
    struct addrinfo hint = {.ai_socktype = SOCK_STREAM}, *addresses = NULL;
    while (peers[0] < 0 && !revoked(comm) && now_s() < comm->until) {
      if (!getaddrinfo(host, colon + 1, &hint, &addresses)) {
        for (struct addrinfo *a = addresses; a && peers[0] < 0 && !revoked(comm); a = a->ai_next) {
          const int fd = socket(a->ai_family, SOCK_STREAM, 0);
          if (fd < 0) continue;
          int error = fcntl(fd, F_SETFL, O_NONBLOCK) || connect(fd, a->ai_addr, a->ai_addrlen) ? errno : 0;
          struct pollfd look = {fd, POLLOUT, 0};
          socklen_t length = sizeof error;
          while (error == EINPROGRESS || error == EINTR)
            if (revoked(comm)) error = ECANCELED;
            else if (now_s() >= comm->until) error = ETIMEDOUT;
            else if (poll(&look, 1, BOOTSTRAP_SLICE_MS) < 0) error = errno == EINTR ? EINPROGRESS : errno;
            else if (look.revents && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length)) error = errno;
          if (error) close(fd);
          else bootstrap_put(comm, 0, fd);
        }
        freeaddrinfo(addresses);
      }
      if (peers[0] < 0) usleep(BOOTSTRAP_SLICE_MS * 1000);
    }
    comm->until = now_s() + bound;
    return revoked(comm) ? ECANCELED : peers[0] < 0 ? ETIMEDOUT : 0;
  }
  uint16_t port = (uint16_t)atoi(colon + 1);
  int listener = !strcmp(root.id, id) ? root.fd : -1;
  if (listener < 0 && (listener = bootstrap_listen(host, &port)) < 0) return errno;
  int met = 0;
  for (int r = 1; r < comm->nranks && !status;) {
    struct report heard;
    struct pollfd look = {listener, POLLIN, 0};
    int fd = -1;
    while (fd < 0 && !status)
      if (revoked(comm)) status = ECANCELED;
      else if (now_s() >= comm->until) status = ETIMEDOUT;
      else if (poll(&look, 1, BOOTSTRAP_SLICE_MS) < 0 && errno != EINTR) status = errno;
      else if ((fd = accept(listener, NULL, NULL)) < 0 && errno != EWOULDBLOCK && errno != ECONNABORTED && errno != EINTR) status = errno;
    if (!status && !met) { comm->until = now_s() + bound; met = 1; }
    if (!status) bootstrap_put(comm, 0, fd);
    if (!status && revoked(comm)) status = ECANCELED;
    const int read = status ? status : whole(comm, fd, &heard, sizeof heard, 0);
    if (read == ECANCELED || read == ETIMEDOUT) status = read;
    if (status) { bootstrap_put(comm, 0, -1); break; }

    if (read || heard.rank <= 0 || heard.rank >= comm->nranks || peers[heard.rank] >= 0) { bootstrap_put(comm, 0, -1); continue; }
    reports[heard.rank] = heard;
    pthread_mutex_lock(&comm->guard);
    peers[heard.rank] = fd; peers[0] = -1;
    pthread_mutex_unlock(&comm->guard);
    r++;
  }
  close(listener);
  if (listener == root.fd) root.fd = -1;
  return status;
}

static int gather(struct ncclComm *comm, int nranks, int rank, struct report *reports, int sent) {
  const int *peers = comm->peers;
  const size_t table = (size_t)nranks * sizeof *reports;
  int status = 0;
  if (rank) {
    const struct report kept = reports[rank];
    if (!sent) status = whole(comm, peers[0], reports + rank, sizeof *reports, 1);
    if (!status) status = whole(comm, peers[0], reports, table, 0);
    reports[rank] = kept;
    return status;
  }
  for (int r = 1; r < nranks && !status && !sent; r++)
    if (!(status = whole(comm, peers[r], reports + r, sizeof *reports, 0)) && reports[r].rank != r) status = EPROTO;
  for (int r = 1; r < nranks && !status; r++) status = whole(comm, peers[r], reports, table, 1);
  return status;
}

static void reach_read(struct attach *a, struct report *report) {
  report->node = a->context.M->node;
  report->count = mesh_nodes_reached(a->context.M, report->reach, MESH_STREAMS);
  report->frees = mesh_streams_free(a->context.M, report->free_node, report->free_wire, MESH_STREAMS);
}

static uint64_t wire_between(const struct report *mine, const struct report *theirs) {
  uint64_t least = 0;
  for (uint32_t i = 0; i < mine->frees && i < MESH_STREAMS; i++) {
    if (mine->free_node[i] != theirs->node) continue;
    for (uint32_t j = 0; j < theirs->frees && j < MESH_STREAMS; j++)
      if (theirs->free_node[j] == mine->node && theirs->free_wire[j] == mine->free_wire[i] && (!least || mine->free_wire[i] < least))
        least = mine->free_wire[i];
  }
  return least;
}

static int comm_apart(const struct report *reports, uint32_t n) {
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++)
      if (reports[a].node == reports[b].node) return EINVAL;
  return 0;
}
static ncclResult_t apart_fail(const char *name) {
  return fail(ncclInvalidUsage, "communicator %s: two of its ranks are on one node (its holdings and their pages posted are kept a node)", name);
}
static ncclResult_t hold_fail(const char *name, int rank, int status) {
  return fail(ncclSystemError, "communicator %s: rank %d could not hold its streams at its bridge (%s: EBUSY, a stream another communicator holds, "
              "or whose last holding its bridge is still ending; ENOLINK, no stream toward a member's node); a communicator made next "
              "reads the streams as they then stand", name, rank, strerror(status));
}

static int comm_table(struct ncclComm *made, const struct mesh_trees *paths, const ncclMeshProgram *programs, int count) {
  if ((paths && trees_copy(&made->paths, paths)) || (count && !(made->given = calloc((size_t)count, sizeof *made->given)))) return ENOMEM;
  for (int e = 0; e < count; e++) {
    if (program_copy(&made->given[e].program, programs[e].program)) return ENOMEM;
    made->given[made->listed++].below = programs[e].below;
  }
  return 0;
}

static struct hold *comm_hold_toward(struct ncclComm *comm, uint32_t node) {
  for (uint32_t k = 0; k < MESH_STREAMS; k++)
    if ((comm->held >> k & 1) && comm->attach->holds[k].node == node) return comm->attach->holds + k;
  return NULL;
}

static int comm_hold(struct ncclComm *comm, const struct report *reports, struct report *mine) {
  struct attach *a = comm->attach;
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  uint64_t bandwidth;
  int status = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t q = 0; !status && q < n; q++) {
    if (q == me || !joined(reports, me, q, &bandwidth)) continue;
    const uint32_t node = reports[q].node, k = reach_of(mine, node);
    const uint64_t wire = wire_between(reports + me, reports + q);
    struct hold *h = NULL;
    uint32_t stream = 0;
    for (uint32_t s = 0; !h && s < MESH_STREAMS; s++)
      if (!a->holds[s].users) h = a->holds + s;
    if (!wire) status = EBUSY;
    else if (!h) status = ENOSPC;
    else if (!(status = mesh_stream_take(&a->context, wire, &stream))) {
      for (uint32_t j = 0; j < MESH_STREAMS; j++)
        if (a->holds + j != h && a->holds[j].users && a->holds[j].stream == stream) a->holds[j].wire = 0;
      *h = (struct hold){.node = node, .stream = stream, .wire = wire};
    }
    if (status) break;
    const uint32_t index = (uint32_t)(h - a->holds);
    h->users++; comm->held |= UINT32_C(1) << index; comm->binding = 1;
    if (!(status = mesh_stream_view(&a->context, stream, &h->view)))
      for (int way = 0; way < 2; way++)
        mine->posted[k][way] = atomic_load_explicit(&h->view.posts[way].posted[0], memory_order_acquire) +
                               atomic_load_explicit(&h->view.posts[way].posted[1], memory_order_acquire);
  }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

static int channel_buffer(struct channel *ch) {
  const uintptr_t page = (uintptr_t)getpagesize();
  const uintptr_t at[3] = {(uintptr_t)ch->view.ready, (uintptr_t)ch->view.sent, (uintptr_t)ch->view.landed};
  uintptr_t lo = at[0], hi = at[0] + sizeof(uint64_t);
  for (int k = 1; k < 3; k++) { lo = at[k] < lo ? at[k] : lo; hi = at[k] + sizeof(uint64_t) > hi ? at[k] + sizeof(uint64_t) : hi; }
  lo &= ~(page - 1);
  hi = (hi + page - 1) & ~(page - 1);
  ch->ready = at[0] - lo; ch->sent = at[1] - lo; ch->landed = at[2] - lo;
  return (ch->buffer = metal_wrap((void *)lo, hi - lo)) ? 0 : ENOMEM;
}

static int comm_join(struct ncclComm *comm, const struct report *reports) {
  struct mesh_ctx *context = &comm->attach->context;
  const struct hdr *m = context->M;
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank, node = reports[me].node;
  uint64_t bandwidth;
  comm->page = (uint64_t)m->block * m->pgsz;
  int status = map_of(comm, reports);
  if (!status && !(comm->violation = metal_shared((size_t)getpagesize()))) status = ENOMEM;
  if (!status) comm->words = metal_contents(comm->violation);
  if (!status && !(comm->channels = calloc((size_t)n + 1, sizeof *comm->channels))) status = ENOMEM;
  for (uint32_t q = 0; !status && q < n; q++) {
    if (q == me || !joined(reports, me, q, &bandwidth)) continue;
    pthread_mutex_lock(&attach_lock);
    struct hold *h = comm_hold_toward(comm, reports[q].node);
    pthread_mutex_unlock(&attach_lock);
    if (!h) { status = EPROTO; break; }
    comm->toward[q] = (int)comm->count;
    struct channel *ch = comm->channels + comm->count++;
    *ch = (struct channel){.peer = (int)q, .node = reports[q].node, .stream = h->stream, .wire = &h->wire, .view = h->view};
    if (!(status = channel_buffer(ch))) atomic_store(&comm->attach->wrapped, 1);
    const uint64_t *own = reports[me].posted[reach_of(reports + me, reports[q].node)], *theirs = reports[q].posted[reach_of(reports + q, node)];
    ch->differ[0] = (int64_t)(theirs[1] - own[0]);
    ch->differ[1] = (int64_t)(theirs[0] - own[1]);
  }
  if (!status && !(comm->timed = calloc((size_t)GROUP_TIMES * comm->count + 1, sizeof *comm->timed))) status = ENOMEM;
  return status;
}

static ncclResult_t comm_init(struct ncclComm *made) {
  const int nranks = made->nranks, rank = made->rank;
  struct report *reports = calloc((size_t)nranks, sizeof *reports);
  int *peers = calloc((size_t)nranks, sizeof *peers);
  if (!reports || !peers) { free(reports); free(peers); return fail(ncclSystemError, "out of memory"); }
  for (int r = 0; r < nranks; r++) peers[r] = -1;
  pthread_mutex_lock(&made->guard);
  made->peers = peers;
  pthread_mutex_unlock(&made->guard);
  struct report *mine = reports + rank;
  int status = nranks > 1 ? bootstrap_open(made, made->id, reports) : 0;
  const int making = status ? status : comm_attach(made, NULL);
  *mine = (struct report){.rank = rank, .status = making};
  if (!making) reach_read(made->attach, mine);
  if (!status && nranks > 1) status = gather(made, nranks, rank, reports, !rank);
  ncclResult_t result = status ? fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status)) : making ? comm_refused(making, made->name) : ncclSuccess;
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status && r != rank)
      result = fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  for (int r = 0; !result && r < nranks; r++) made->members[r] = reports[r].node;
  if (!result && nranks > 1 && comm_apart(reports, (uint32_t)nranks)) result = apart_fail(made->name);
  if (!result && nranks > 1) {
    mine->status = comm_hold(made, reports, mine);
    made->until = now_s() + (double)MESH_PLATFORM_NS / 1e9;
    if ((status = gather(made, nranks, rank, reports, 0))) result = fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status));
    for (int r = 0; !result && r < nranks; r++)
      if (reports[r].status) result = hold_fail(made->name, r, reports[r].status);
    if (!result && (status = comm_join(made, reports)))
      result = fail(ncclSystemError, "communicator %s: its channels on the streams it held: %s", made->name, strerror(status));
  }
  free(reports);
  if (revoked(made)) result = revoked_fail(made);
  if (result) { bootstrap_close(made); comm_leave(made); }
  return result;
}

static void *init_run(void *argument) {
  struct ncclComm *comm = argument;
  const ncclResult_t result = comm_init(comm);
  if (result) { comm->why = last; comm->why.id = keep(); }
  __atomic_store_n(&comm->async, result, __ATOMIC_RELEASE);
  return NULL;
}

ncclResult_t ncclMeshCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, const struct mesh_trees *paths,
                                  const ncclMeshProgram *programs, int count, uint64_t small, const char *name, ncclConfig_t *config) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks || count < 0 || (count && !programs) || (paths && paths->nodes != (uint32_t)nranks))
    return fail(ncclInvalidArgument, "the init's arguments: rank %d of %d, %d programs", rank, nranks, count);
  for (int e = 0; e < count; e++)
    if (!programs[e].program || programs[e].program->nodes != (uint32_t)nranks) return fail(ncclInvalidArgument, "program %d is not of %d ranks", e, nranks);
  char id[sizeof commId.internal + 1];
  memcpy(id, commId.internal, sizeof commId.internal);
  id[sizeof commId.internal] = 0;
  struct ncclComm *made = comm_new(nranks, rank, name && *name ? name : id);
  (void)small;
  if (!made || comm_table(made, paths, programs, count)) {
    if (made) comm_free(made);
    return fail(ncclSystemError, "out of memory");
  }
  memcpy(made->id, id, sizeof made->id);
  if (config && !config->blocking) {
    made->async = ncclInProgress;
    *comm = made;
    const int error = pthread_create(&made->init, NULL, init_run, made);
    made->initing = !error;
    return error ? (made->async = fail(ncclSystemError, "the thread of communicator %s's init: %s", made->name, strerror(error))) : ncclInProgress;
  }
  const ncclResult_t result = comm_init(made);
  if (result) comm_free(made);
  else *comm = made;
  return result;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  return ncclMeshCommInitRank(comm, nranks, commId, rank, NULL, NULL, 0, 0, NULL, NULL);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config) {
  return ncclMeshCommInitRank(comm, nranks, commId, rank, NULL, NULL, 0, 0, NULL, config);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  (void)devlist;
  if (ndev != 1) return fail(ncclInvalidArgument, "a node has one Metal device: ndev %d", ndev);
  ncclUniqueId id;
  const ncclResult_t result = ncclGetUniqueId(&id);
  return result ? result : ncclCommInitRank(comm, 1, id, 0);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL"); }

static void inflight_lose(struct ncclComm *comm);
static ncclResult_t comm_failure(struct ncclComm *comm);

static void comm_stop(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  if (comm->words) atomic_store_explicit(comm->words + 6, 1, memory_order_release);
  pthread_mutex_unlock(&comm->guard);
}

ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  (void)revokeFlags;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  comm_stop(comm);
  return ncclSuccess;
}

static int gpu_wait(struct ncclComm *comm, int group);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  comm_stop(comm);
  if (comm->initing) pthread_join(comm->init, NULL);
  pthread_mutex_lock(&group_lock);
  while (atomic_load(&comm->runs)) {
    pthread_mutex_unlock(&group_lock);
    usleep(20);
    pthread_mutex_lock(&group_lock);
  }
  gpu_wait(comm, 0);
  inflight_lose(comm);
  pthread_mutex_unlock(&group_lock);
  comm_free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  const ncclResult_t async = __atomic_load_n(&comm->async, __ATOMIC_ACQUIRE);
  if (async != ncclInProgress && comm->why.id) told(async, &comm->why);
  if (async) { *asyncError = async; return ncclSuccess; }
  if (revoked(comm)) { *asyncError = ncclSuccess; return ncclSuccess; }
  pthread_mutex_lock(&group_lock);
  *asyncError = comm_failure(comm);
  pthread_mutex_unlock(&group_lock);
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
  if (!comm || !handle) return fail(ncclInvalidArgument, "comm or handle is NULL");
  pthread_mutex_lock(&attach_lock);
  const struct hdr *m = attaches ? attaches->context.M : NULL;
  const uintptr_t at = (uintptr_t)buff, first = m ? (uintptr_t)m + m->data_off : 0, end = m ? (uintptr_t)m + m->length : 0;
  pthread_mutex_unlock(&attach_lock);
  if (!m || at < first || at > end || size > end - at)
    return fail(ncclInvalidArgument, "ncclCommRegister of %zu bytes at %p: not in the region of this node's bridge; a buffer "
                "collectives move is leased from the region (ncclMemAlloc)", size, buff);
  *handle = buff;
  return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  (void)handle;
  return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL");
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  if (!ptr) return fail(ncclInvalidArgument, "ptr is NULL");
  pthread_mutex_lock(&attach_lock);
  struct attach *a = NULL;
  struct mesh_section lease = {0};
  const size_t bytes = size ? size : 1;
  ncclResult_t result = ncclSuccess;
  int status = attach_current(&a);
  if (status)
    result = attach_fail(status, "ncclMemAlloc of %zu bytes", size);
  else if (a->count == a->room && lease_room(a))
    result = fail(ncclSystemError, "out of memory");
  else if ((status = mesh_section_create(&a->context, bytes, 1, &lease))) {
    const struct hdr *m = a->context.M;
    const size_t quantum = (size_t)m->block * m->pgsz;
    result = fail(ncclSystemError, "ncclMemAlloc of %zu bytes (%zu pages of %u bytes): the region of this node's bridge has %u of "
                  "its %u pages free (%s); a client sizes its leases to the region its node's bridge configuration states "
                  "(etc/bridge/README.md)", size, (bytes + quantum - 1) / quantum * m->block, m->pgsz, mesh_arena_free(a->context.M),
                  mesh_arena_pages(m), strerror(status));
  } else {
    *ptr = mesh_section_address(&a->context, lease, 0);
    a->leases[a->count++] = (struct lease){*ptr, lease.first, (size_t)lease.pages * a->context.M->pgsz, NULL};
  }
  pthread_mutex_unlock(&attach_lock);
  return result;
}

ncclResult_t ncclMeshRegionExpired(int *expired) {
  if (!expired) return fail(ncclInvalidArgument, "ncclMeshRegionExpired: expired is NULL");
  pthread_mutex_lock(&attach_lock);
  attach_settle();
  *expired = attaches && attach_expired(attaches);
  pthread_mutex_unlock(&attach_lock);
  return ncclSuccess;
}

ncclResult_t ncclMemFree(void *ptr) {
  if (!ptr) return ncclSuccess;
  pthread_mutex_lock(&attach_lock);
  int found = 0, status = 0;
  uint32_t row = 0;
  for (struct attach *a = attaches; a && !found; a = a->next)
    for (size_t i = 0; i < a->count && !found; i++)
      if (a->leases[i].address == ptr) {
        found = 1;
        row = a->leases[i].row;
        if (a->leases[i].buffer) metal_release(a->leases[i].buffer);
        status = mesh_lease_return(&a->context, row);
        a->leases[i] = a->leases[--a->count];
      }
  attach_settle();
  pthread_mutex_unlock(&attach_lock);
  if (!found) return fail(ncclInvalidArgument, "ncclMemFree of %p: no lease of this process (ncclMemAlloc)", ptr);
  if (status) return fail(ncclInternalError, "ncclMemFree of %p: row %u of this node's region heads no lease of this process", ptr, row);
  return ncclSuccess;
}

static int lease_buffer(const void *ptr, void **buffer, size_t *offset) {
  int status = ENOENT;
  pthread_mutex_lock(&attach_lock);
  for (struct attach *a = attaches; a && status == ENOENT; a = a->next)
    for (size_t i = 0; i < a->count && status == ENOENT; i++) {
      struct lease *l = a->leases + i;
      const uintptr_t at = (uintptr_t)ptr, first = (uintptr_t)l->address;
      if (at < first || at >= first + l->bytes) continue;
      if (!l->buffer && (l->buffer = metal_wrap(l->address, l->bytes))) atomic_store(&a->wrapped, 1);
      status = l->buffer ? 0 : ENOMEM;
      if (!status) { *buffer = l->buffer; *offset = (size_t)(at - first); }
    }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

static struct lease *lease_at(const void *ptr) {
  for (struct attach *a = attaches; a; a = a->next)
    for (size_t i = 0; i < a->count; i++) {
      const uintptr_t at = (uintptr_t)ptr, first = (uintptr_t)a->leases[i].address;
      if (at >= first && at < first + a->leases[i].bytes) return a->leases + i;
    }
  return NULL;
}

ncclResult_t ncclMeshMemBuffer(const void *ptr, void **buffer, size_t *offset) {
  if (!ptr || !buffer || !offset) return fail(ncclInvalidArgument, "ncclMeshMemBuffer: a NULL argument");
  const int status = lease_buffer(ptr, buffer, offset);
  if (status == ENOMEM) return fail(ncclSystemError, "ncclMeshMemBuffer of %p: no Metal buffer over its lease", ptr);
  return status ? fail(ncclInvalidArgument, "ncclMeshMemBuffer: %p is in no lease of this process (ncclMemAlloc)", ptr) : ncclSuccess;
}

static ncclResult_t op_make(ncclComm_t comm, struct redop made, ncclRedOp_t *op) {
  uint32_t k = 0;
  while (k < comm->nops && comm->ops[k].used) k++;
  if (k == comm->nops) {
    if (comm->nops >= (uint32_t)INT_MAX - ncclNumOps) return fail(ncclInvalidUsage, "operator handles past INT_MAX (ncclRedOp_t)");
    struct redop *grown = realloc(comm->ops, (comm->nops + 1) * sizeof *grown);
    if (!grown) return fail(ncclSystemError, "out of memory");
    comm->ops = grown;
    comm->nops++;
  }
  comm->ops[k] = made;
  *op = (ncclRedOp_t)(ncclNumOps + (int)k);
  return ncclSuccess;
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  if (!op || !scalar || !comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "PreMulSum's arguments");
  const int immediate = residence == ncclScalarHostImmediate;
  struct redop made = {1, datatype, {0}, 0, immediate ? NULL : scalar};
  if (immediate) memcpy(made.value, scalar, SIZE[datatype]);
  return op_make(comm, made, op);
}

ncclResult_t ncclMeshRedOpCreatePartials(ncclRedOp_t *op, size_t dim, ncclDataType_t datatype, ncclComm_t comm) {
  if (!op || !comm || datatype != ncclFloat32 || dim > UINT32_MAX - 2)
    return fail(ncclInvalidArgument, "the partials merge's arguments: its operator, its communicator, float32 and a row's dim");
  return op_make(comm, (struct redop){1, datatype, {0}, (uint32_t)(2 + dim), NULL}, op);
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  const int k = (int)op - ncclNumOps;
  if (!comm || k < 0 || k >= (int)comm->nops || !comm->ops[k].used) return fail(ncclInvalidArgument, "operator %d is none of this communicator's", (int)op);
  comm->ops[k].used = 0;
  return ncclSuccess;
}

static double now_s(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

static ncclResult_t comm_told(struct ncclComm *comm) {
  inflight_lose(comm);
  return told(comm->failed, &comm->failure);
}

static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) __attribute__((format(printf, 3, 4)));
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) {
  if (!comm->failed) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(comm->failure.text, sizeof comm->failure.text, format, arguments);
    va_end(arguments);
    comm->failure.id = keep();
    comm->failed = result;
  }
  return comm_told(comm);
}

static ncclResult_t comm_failure(struct ncclComm *comm) {
  if (comm->failed) return comm_told(comm);
  if (revoked(comm))
    return comm_fail(comm, ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
  _Atomic uint64_t *v = comm->words;
  const uint64_t at = v ? atomic_load_explicit(v + 2, memory_order_acquire) : 0;
  if (at) {
    const uint64_t cause = atomic_load_explicit(v, memory_order_relaxed), h = atomic_load_explicit(v + 3, memory_order_relaxed);
    const uint32_t node = h < comm->count ? comm->channels[h].node : UINT32_MAX;
    if (cause == METAL_LOST)
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost (its holding's counts ~0) at entry %llu", node,
                       (unsigned long long)(at - 1));
    return comm_fail(comm, ncclRemoteError, "entry %llu on the channel to node %u was not done within %llu polls of a GPU wait (the "
                     "library's bound): its bytes had not landed or been sent, the peer's work there beginning that "
                     "much later, or its link was lost", (unsigned long long)(at - 1), node,
                     (unsigned long long)atomic_load_explicit(v + 4, memory_order_relaxed));
  }
  struct attach *a = comm->attach;
  for (uint32_t k = 0; comm->held && k < MESH_STREAMS; k++)
    if ((comm->held >> k & 1) && !mesh_stream_held(&a->context, a->holds[k].stream, a->holds[k].wire))
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost and paired again: communicator %s's stream there is no "
                       "longer its own (design/transport.md section 14); it is made again (ncclCommDestroy, then an init)",
                       a->holds[k].node, comm->name);
  return ncclSuccess;
}

static int same(struct where a, struct where b) { return a.buffer == b.buffer && a.offset == b.offset; }
static struct where shifted(struct where w, size_t by) { w.offset += by; return w; }

static unsigned char *address_of(struct where w) {
  if (!w.buffer) return (unsigned char *)w.offset;
  unsigned char *base = metal_contents(w.buffer);
  return base ? base + w.offset : NULL;
}

struct steps {
  struct mesh_step *steps;
  uint32_t count, buffers;
  uint64_t relayed;
  unsigned char *done, *reads, *fresh, *needed;
  struct where *buffer;
  size_t *at;
  uint64_t lo;
  struct mesh_operand operand;

  uint64_t *cut;

  uint32_t *channel;
  uint64_t *entry;
  int32_t *before;
};

struct call {
  int what, root, peer;
  const void *send;
  void *recv;
  size_t count, total;
  uint64_t *segments;

  uint64_t *complement;
  uint32_t complements;
  ncclDataType_t type;
  ncclRedOp_t op;
  ncclComm_t comm;
  int combine, post, every;
  uint32_t unit;
  unsigned char pre[8];

  const void *scalar_at;
  int premultiplied, direct, direct_ok;
  struct steps *bound;
  unsigned char *copy;
  size_t at;
  cudaStream_t stream;
  void *command, *encoder, *table, *residency;
  ncclMeshIssue *issue;
  struct where send_at, recv_at, sum;
};

static _Thread_local int depth;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;

static int reducing(int what) { return what == MESH_ALLREDUCE || what == MESH_REDUCE || what == MESH_REDUCE_SCATTER; }

static uint32_t unit_of(const struct call *c) { return c->combine == COMBINE_MERGE ? c->unit : 1; }
static int ktype(const struct call *c) { return c->combine == COMBINE_MERGE ? METAL_MERGE : (int)c->type; }
static int kop(const struct call *c) { return c->combine == COMBINE_MERGE ? (int)c->unit : c->combine; }
static size_t kcount(const struct call *c, size_t bytes) { return bytes / (SIZE[c->type] * unit_of(c)); }
static int p2p(const struct call *c) { return c->what == WHAT_SEND || c->what == WHAT_RECV; }
static int counted(const struct call *c) { return c->what == WHAT_COUNTED; }
static size_t elements(const struct call *c) {
  if (counted(c)) return c->count;
  if (c->segments) return c->total;
  return c->count * ((c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER) ? (size_t)c->comm->nranks : 1);
}

static size_t segment_at(const struct call *c, int r, size_t *length) {
  size_t before = 0;
  for (int q = 0; c->segments && q < r; q++) before += c->segments[q];
  *length = c->segments ? c->segments[r] : c->count;
  return c->segments ? before : (size_t)r * c->count;
}

static uint64_t seg_first(const uint64_t *seg, int q) {
  uint64_t at = 0;
  for (int r = 0; r < q; r++) at += seg[r];
  return at;
}

static const char *call_name(const struct call *c) {
  static const char *const names[] = {"ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclReduceScatter", "ncclAllGather"};
  return c->what == WHAT_SEND ? "ncclSend" : c->what == WHAT_RECV ? "ncclRecv" : !counted(c) ? names[c->what] : !c->every ? "ncclMeshAlltoAllV"
         : c->root >= 0 ? "ncclScatter" : c->peer >= 0 ? "ncclGather" : "ncclAlltoAll";
}

static int crosses(const struct call *c) { return elements(c) && c->comm->nranks > 1 && !(p2p(c) && c->peer == c->comm->rank); }

static int in_region(const struct ncclComm *comm, struct where w, size_t bytes) {
  const struct hdr *m = comm->attach->context.M;
  const unsigned char *at = address_of(w), *first = (const unsigned char *)m + m->data_off, *end = (const unsigned char *)m + m->length;
  return at && at >= first && at <= end && bytes <= (size_t)(end - at);
}

static size_t sent_bytes(const struct call *c) {
  const size_t z = SIZE[c->type];
  size_t length = 0;
  if (c->what == WHAT_RECV || (c->what == MESH_BROADCAST && c->root != c->comm->rank)) return 0;
  if (c->what == MESH_ALLGATHER) { segment_at(c, c->comm->rank, &length); return length * z; }
  if (counted(c)) return seg_first(c->segments, c->comm->nranks) * z;
  if (c->what == WHAT_SEND || c->what == MESH_BROADCAST) return c->count * z;
  return elements(c) * z;
}

static ncclResult_t operands_check(const struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const struct call *c = list + i;
    const size_t bytes = crosses(c) ? sent_bytes(c) : 0;
    if (bytes && !in_region(c->comm, c->send_at, bytes))
      return fail(ncclInvalidUsage, "%s on communicator %s: its send buffer (%zu bytes) is not in this node's region: an operand that crosses a "
                  "link is a lease's (ncclMemAlloc, ncclMeshMemBuffer), and none is copied into the region (nccl.h, Memory)", call_name(c),
                  c->comm->name, bytes);
  }
  return ncclSuccess;
}

static void steps_free(struct steps *s) {
  if (!s) return;
  free(s->buffer); free(s->steps); free(s->done); free(s->channel); free(s->entry); free(s->before); free(s->reads); free(s->fresh);
  free(s->needed); free(s->cut); free(s->at);
  free(s);
}

static size_t piece_bytes(const struct mesh_step *step) { return step->piece.elements * step->piece.element_bytes; }

static uint64_t hosted_count(const struct call *c, int a, int b) {
  return (c->root < 0 || c->root == a) && (c->peer < 0 || c->peer == b) ? c->total : 0;
}

static ncclResult_t hosted_steps(ncclComm_t comm, const struct call *c, struct mesh_operand operand, struct mesh_step **steps, uint32_t *count,
                                 uint64_t *relayed) {
  const int n = comm->nranks, me = comm->rank;
  const struct mesh_routing *g = &comm->routing;
  struct mesh_block *blocks = calloc((size_t)n * n + 1, sizeof *blocks);
  uint32_t made = 0;
  if (!blocks || !(*steps = calloc(c->every ? 1 : 2 * (size_t)n + 1, sizeof **steps))) { free(blocks); return fail(ncclSystemError, "out of memory"); }
  if (!c->every) {
    *count = mesh_counted_steps((uint32_t)n, (uint32_t)me, c->segments, (struct mesh_operand){operand.type, operand.element_bytes, 0}, *steps);
    free(blocks);
    return ncclSuccess;
  }
  for (int a = 0; a < n; a++)
    for (int b = 0; b < n; b++) {
      uint64_t first = 0, at = 0;
      for (int q = 0; q < b; q++) first += hosted_count(c, a, q);
      for (int q = 0; q < a; q++) at += hosted_count(c, q, b);
      if (a != b && hosted_count(c, a, b))
        blocks[made++] = (struct mesh_block){(uint32_t)a, (uint32_t)b, 0, 1, operand.type, operand.element_bytes, first, at, hosted_count(c, a, b)};
    }
  const uint32_t room = mesh_exchange_room(g, blocks, made, NULL);
  ncclResult_t result = ncclSuccess;
  free(*steps);
  if (!(*steps = calloc((size_t)room + 1, sizeof **steps))) result = fail(ncclSystemError, "out of memory");
  else if ((*count = mesh_exchange_steps(g, (uint32_t)me, blocks, made, 0, 2, comm->page, *steps, NULL, relayed)) == UINT32_MAX)
    result = fail(ncclInvalidUsage, "%s on %s: a block between ranks the communicator's routing table joins by no route", call_name(c), comm->name);
  free(blocks);
  return result;
}

static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
    const ncclResult_t compiled = p2p(c) ? ncclSuccess : !counted(c) ? comm_compile(comm) : c->every && comm_routed(comm)
                                                                    ? fail(ncclSystemError, "out of memory") : ncclSuccess;
    if (compiled) return compiled;
    const struct mesh_program *program = p2p(c) || counted(c) ? NULL : program_of(comm, c->what, c->root, elements(c) * SIZE[c->type]);
    if (!p2p(c) && !counted(c) && !program)
      return fail(ncclInvalidUsage, "the communicator has no program for collective %d at root %d", c->what, c->root);
    const int nccl = c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER;
    if (nccl) {
      uint32_t segments = 1;
      for (uint32_t p = 0; p < program->parts; p++) if (program->segment[p] + 1 > segments) segments = program->segment[p] + 1;
      if (segments != (uint32_t)comm->nranks)
        return fail(ncclInvalidUsage, "a reduce-scatter's or all-gather's program cuts %u segments, not one a rank of %d", segments, comm->nranks);
    }
    struct steps *s = calloc(1, sizeof *s);
    if (!s) return fail(ncclSystemError, "out of memory");
    c->bound = s;
    const uint32_t u = unit_of(c);
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)(SIZE[c->type] * u), elements(c) / u};
    const ncclResult_t hosted = counted(c) ? hosted_steps(comm, c, operand, &s->steps, &s->count, &s->relayed) : ncclSuccess;
    if (hosted) return hosted;
    const size_t most = (program ? program->first[program->nodes] : counted(c) ? s->count : 1) + 1;
    s->buffers = counted(c) ? 2 + (s->relayed != 0) : 1 + (program ? program->scratch : 0);
    if ((!s->steps && !(s->steps = calloc(most, sizeof *s->steps))) || !(s->done = calloc(most, 1)) || !(s->channel = calloc(most, sizeof *s->channel)) ||
        !(s->entry = calloc(most, sizeof *s->entry)) || !(s->before = calloc(most, sizeof *s->before)) ||
        !(s->buffer = calloc(s->buffers, sizeof *s->buffer)) || !(s->at = calloc(s->buffers, sizeof *s->at)) || (nccl && !(s->cut = malloc((size_t)comm->nranks * sizeof *s->cut))))
      return fail(ncclSystemError, "out of memory");
    s->operand = operand;
    for (int r = 0; nccl && r < comm->nranks; r++) s->cut[r] = (c->segments ? c->segments[r] : c->count) / u;
    if (p2p(c)) {
      s->steps[0] = (struct mesh_step){c->what == WHAT_SEND ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)c->peer, 0, 0, 0, operand};
      s->count = 1;
    } else if (!counted(c)) s->count = mesh_program_steps_cut(program, (uint32_t)comm->rank, operand, s->cut, comm->page, s->steps);
    for (uint32_t k = 0; k < s->count; k++)
      if (piece_bytes(s->steps + k) && comm->toward[s->steps[k].peer] < 0)
        return fail(ncclInvalidUsage, "%s on %s moves bytes between rank %d and rank %u (node %u), which no link of this node joins (rdma/NCCL.md, "
                    "Routes between unlinked ranks)", call_name(c), comm->name, comm->rank, s->steps[k].peer, comm->members[s->steps[k].peer]);
  }
  return ncclSuccess;
}

struct message { struct call *c; uint32_t step; uint64_t entry, need, rel_entry, rel_need; mesh_pages pages, end, sent, rel_end, rel_sent;
                 size_t offset, bytes, area; struct where data; uint64_t row; };

struct schedule { struct message *out, *in; size_t outs, ins; uint64_t first[2]; };

static uint32_t group_kind(const struct call *c) { return p2p(c) ? MESH_GROUP_POINT : counted(c) ? MESH_GROUP_HOSTED : MESH_GROUP_COLLECTIVE; }
static void group_of(const struct call *list, size_t n, uint32_t N, const struct mesh_step **steps, uint32_t *count,
                     struct mesh_group_call *calls, uint64_t page, struct mesh_group *group) {
  for (size_t i = 0; i < n; i++) calls[i] = (struct mesh_group_call){group_kind(list + i), 0, count + i * N, steps + i * N};
  *group = (struct mesh_group){N, (uint32_t)n, calls, page};
}

static size_t page_up(size_t bytes, uint64_t page) { return page ? (size_t)((bytes + page - 1) / page * page) : bytes; }

static void group_numbered(struct ncclComm *comm) {
  if (comm->timed) memset(comm->timed + (comm->groups % GROUP_TIMES) * comm->count, 0, comm->count * sizeof *comm->timed);
  comm->groups++;
}

static int meet(const struct mesh_step *x, const struct mesh_step *y) {
  return x->buffer == y->buffer && x->first < y->first + y->piece.elements && y->first < x->first + x->piece.elements;
}

static ncclResult_t schedule_group(struct ncclComm *comm, struct call *list, size_t n, struct schedule *plans, size_t *area) {
  const uint32_t N = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  size_t total = 1;
  for (size_t i = 0; i < n; i++) total += list[i].bound ? list[i].bound->count : 0;
  const struct mesh_step **steps = calloc(n * N + 1, sizeof *steps);
  uint32_t *counts = calloc(n * N + 1, sizeof *counts);
  struct mesh_group_call *group_calls = calloc(n + 1, sizeof *group_calls);
  struct mesh_group_place *made = calloc(total, sizeof *made);
  struct mesh_place *scratch = calloc(total, sizeof *scratch);
  struct mesh_group group;
  ncclResult_t result = steps && counts && group_calls && made && scratch ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (size_t i = 0; i < n && !result; i++)
    if (list[i].bound) { steps[i * N + me] = list[i].bound->steps; counts[i * N + me] = list[i].bound->count; }
  if (!result) group_of(list, n, N, steps, counts, group_calls, comm->page, &group);
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    const struct channel *ch = comm->channels + h;
    struct schedule *p = plans + h;
    *p = (struct schedule){calloc(total, sizeof *p->out), calloc(total, sizeof *p->in), 0, 0, {0, 0}};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    for (int send = 1; send >= 0 && !result; send--) {
      uint32_t placed = 0;
      mesh_group_lay(&group, me, (uint32_t)ch->peer, send, 0, scratch, made, &placed);
      mesh_pages end = {0};
      for (uint32_t k = 0; k < placed; k++) {
        const struct mesh_place *x = &made[k].place;
        struct call *c = list + made[k].call;
        const struct mesh_step *step = c->bound->steps + x->step;
        const size_t bytes = piece_bytes(step);
        const mesh_pages pages = pages_of(bytes, comm->page);
        end = pages_add(end, pages);
        struct message m = {.c = c, .step = x->step, .rel_entry = x->entry, .pages = pages, .rel_end = end,
                            .offset = step->first * step->piece.element_bytes, .bytes = bytes};
        if (!send && step->op == MESH_STEP_REDUCE) { m.area = *area; *area += page_up(bytes, comm->page); }
        if (send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
        c->bound->channel[x->step] = h;
      }
    }
    for (size_t k = 0; k < p->ins; k++) {
      struct message *m = p->in + k;
      const struct steps *st = m->c->bound;
      if (st->steps[m->step].op != MESH_STEP_REDUCE) continue;
      for (size_t j = 0; j < p->outs; j++) {
        const struct message *o = p->out + j;
        if (o->c != m->c || o->step >= m->step || (st->reads && st->reads[o->step]) || !meet(st->steps + o->step, st->steps + m->step)) continue;
        if (o->rel_entry + 1 > m->rel_need) m->rel_need = o->rel_entry + 1;
        if (pages_below(m->rel_sent, o->rel_end)) m->rel_sent = o->rel_end;
      }
    }
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct steps *st = list[i].bound;
    if (st && mesh_group_before(group_kind(list + i), st->steps, st->count, st->before)) result = fail(ncclSystemError, "out of memory");
    for (uint32_t k = 0; st && st->relayed && !result && k < st->count; k++) {
      const struct mesh_step *x = st->steps + k;
      uint32_t j = 0;
      while (x->op == MESH_STEP_SEND && x->buffer == 2 && j < st->count &&
             !(st->steps[j].op == MESH_STEP_COPY && st->steps[j].buffer == 2 && st->steps[j].first == x->first && st->steps[j].piece.elements == x->piece.elements))
        j++;
      if (x->op == MESH_STEP_SEND && x->buffer == 2 && j == st->count) result = fail(ncclInternalError, "a relayed part of %s with no receive", call_name(list + i));
      else if (x->op == MESH_STEP_SEND && x->buffer == 2) st->before[k] = (int32_t)j;
    }
  }
  free(steps); free(counts); free(group_calls); free(made); free(scratch);
  return result;
}

static void schedules_free(const struct ncclComm *comm, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < comm->count; h++) { free(plans[h].out); free(plans[h].in); }
  free(plans);
}







static struct where read_at(const struct call *c) { return (struct where){c->send_at.buffer, c->send_at.offset - c->bound->lo * SIZE[c->type]}; }

static ncclResult_t place_group(struct ncclComm *comm, struct plan *plan) {
  const unsigned char *region = (const unsigned char *)comm->attach->context.M;
  const uint64_t page = comm->page;
  struct schedule *plans = plan->plans;
  for (uint32_t h = 0; h < comm->count; h++)
    for (int send = 1; send >= 0; send--)
      for (size_t k = 0; k < (send ? plans[h].outs : plans[h].ins); k++) {
        struct message *x = (send ? plans[h].out : plans[h].in) + k;
        const struct steps *st = x->c->bound;
        const struct mesh_step *step = st->steps + x->step;
        if (send) x->data = shifted(st->reads && st->reads[x->step] ? read_at(x->c) : st->buffer[step->buffer], x->offset);
        else if (step->op == MESH_STEP_REDUCE) x->data = shifted(plan->base, x->area);
        else x->data = shifted(st->buffer[step->buffer], x->offset);
        const unsigned char *address = address_of(x->data);
        const int inside = in_region(comm, x->data, x->bytes);
        pthread_mutex_lock(&attach_lock);
        const struct lease *held = inside ? lease_at(address) : NULL;
        const uint64_t row = held ? held->row + (uint64_t)(address - (const unsigned char *)held->address) / page : 0;
        pthread_mutex_unlock(&attach_lock);
        if (held && !((size_t)(address - region) % page)) { x->row = row; continue; }
        return fail(ncclInvalidUsage, "%s on communicator %s: the %zu bytes its %s rank %u %s (buffer %u, byte %zu) %s", call_name(x->c), comm->name,
                    x->bytes, send ? "SEND to" : "receive from", step->peer, send ? "reads" : "lands in", step->buffer, x->offset,
                    held ? "do not start on a page: an entry is a range of rows, each a page (design/pages-and-functions.md), and a cut NCCL's "
                           "call fixes is never moved by a copy"
                         : "are in no lease of this process: a crossing's pages are a lease's rows (ncclMemAlloc), never copied");
      }
  return ncclSuccess;
}

static void calls_free(struct call *list, size_t n);
static int plan_direct(struct call *c);
static int plan_idle(const struct plan *p) {
  return !p->flight && (!p->pages || __atomic_load_n((const uint32_t *)((const char *)p->pages + p->word), __ATOMIC_ACQUIRE) == p->reader);
}
static void rows_free(struct metal_rows *rows, int idle) {
  for (size_t k = 0; idle && k < rows->chunks; k++) ncclMemFree(rows->chunk[k].pages);
  free(rows->chunk);
  *rows = (struct metal_rows){.comm = rows->comm};
}

static void plan_free(struct ncclComm *comm, struct plan *p) {
  const int idle = plan_idle(p);
  calls_free(p->list, p->n);
  schedules_free(comm, p->plans);
  if (p->pages && idle) ncclMemFree(p->pages);
  rows_free(&p->rows, idle);
  free(p);
}

static int in_place(const struct call *c) {
  return c->what == MESH_ALLREDUCE || c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER || (c->what == MESH_REDUCE && c->root == c->comm->rank);
}
static int direct_ok(const struct call *c) {
  if (p2p(c) || counted(c) || !crosses(c) || c->premultiplied || unit_of(c) > 1 || (c->what == MESH_BROADCAST && c->root != c->comm->rank)) return 0;
  size_t length = elements(c);
  if (c->what == MESH_ALLGATHER) segment_at(c, c->comm->rank, &length);
  const size_t z = SIZE[c->type], sent = length * z, made = elements(c) * z;
  const unsigned char *a = address_of(c->send_at), *b = address_of(c->recv_at);
  return in_region(c->comm, c->send_at, sent) && (!in_place(c) || (b && !(a < b + made && b < a + sent)));
}
static int call_same(const struct call *a, const struct call *b) {
  const size_t segs = (counted(a) ? 2 : 1) * (size_t)a->comm->nranks;
  return a->what == b->what && a->root == b->root && a->peer == b->peer && a->count == b->count && a->total == b->total && a->type == b->type &&
         a->combine == b->combine && a->post == b->post && a->premultiplied == b->premultiplied && a->unit == b->unit &&
         a->direct_ok == b->direct_ok && a->every == b->every && !a->stream == !b->stream && !a->segments == !b->segments &&
         (!a->segments || !memcmp(a->segments, b->segments, segs * sizeof *a->segments));
}

static ncclResult_t lease_take(struct ncclComm *comm, size_t bytes, void **pages, void **buffer, size_t *offset) {
  ncclResult_t result = ncclMemAlloc(pages, bytes);
  if (result) { sweep(comm, 1, 0); result = ncclMemAlloc(pages, bytes); }
  if (!result && buffer && lease_buffer(*pages, buffer, offset)) result = fail(ncclSystemError, "no Metal buffer over a lease of %zu bytes", bytes);
  return result;
}

int metal_rows_take(struct metal_rows *rows, size_t bytes, void **buffer, size_t *offset) {
  bytes = (bytes + 127) & ~(size_t)127;
  while (rows->at < rows->chunks && rows->used + bytes > rows->chunk[rows->at].bytes) { rows->at++; rows->used = 0; }
  if (rows->at == rows->chunks) {
    struct row_chunk made = {.bytes = page_up(bytes > (64u << 10) ? bytes : (64u << 10), rows->comm->page)};
    struct row_chunk *grown = realloc(rows->chunk, (rows->chunks + 1) * sizeof *grown);
    if (grown) rows->chunk = grown;
    if (!grown || lease_take(rows->comm, made.bytes, &made.pages, &made.buffer, &made.offset)) {
      if (made.pages) ncclMemFree(made.pages);
      return ENOMEM;
    }
    rows->chunk[rows->chunks++] = made;
    rows->used = 0;
  }
  *buffer = rows->chunk[rows->at].buffer;
  *offset = rows->chunk[rows->at].offset + rows->used;
  rows->used += bytes;
  return 0;
}

static ncclResult_t plan_make(struct ncclComm *comm, const struct call *list, size_t n, struct plan **out) {
  struct plan *p = calloc(1, sizeof *p);
  if (!p) return fail(ncclSystemError, "out of memory");
  ncclResult_t result = (p->list = calloc(n, sizeof *p->list)) && (p->plans = calloc(comm->count + 1, sizeof *p->plans)) ? ncclSuccess
                                                                                                                       : fail(ncclSystemError, "out of memory");
  p->rows.comm = comm;
  size_t bytes = 0;
  for (size_t i = 0; i < n && !result; i++, p->n++) {
    struct call *c = p->list + i;
    const size_t segs = (counted(list + i) ? 2 : 1) * (size_t)comm->nranks;
    *c = list[i];
    c->bound = NULL; c->complement = NULL; c->complements = 0; c->copy = NULL; c->segments = NULL; c->at = SIZE_MAX;
    if (list[i].segments && !(c->segments = malloc(segs * sizeof *c->segments))) result = fail(ncclSystemError, "out of memory");
    else if (list[i].segments) memcpy(c->segments, list[i].segments, segs * sizeof *c->segments);
  }
  if (!result) result = plan_calls(comm, p->list, n);
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = p->list + i;
    struct steps *st = c->bound;
    if (counted(c) && st && st->relayed) { const size_t each = page_up(st->relayed, comm->page); st->at[2] = (bytes += each) - each; }
    if (p2p(c) || counted(c)) continue;
    const size_t each = page_up(elements(c) * SIZE[c->type], comm->page);
    c->direct = st && plan_direct(c);
    c->at = in_place(c) ? SIZE_MAX : (bytes += each) - each;
    for (uint32_t b = 1; st && b < st->buffers; b++) st->at[b] = c->direct && !st->needed[b] ? SIZE_MAX : (bytes += each) - each;
  }
  if (!result) result = schedule_group(comm, p->list, n, p->plans, &bytes);
  p->word = bytes;
  size_t offset = 0;
  if (!result && (bytes || list[0].stream)) result = lease_take(comm, bytes + sizeof(uint32_t), &p->pages, list[0].stream ? &p->buffer : NULL, &offset);
  if (p->pages) memset((char *)p->pages + p->word, 0, sizeof(uint32_t));
  p->base = p->buffer ? (struct where){p->buffer, offset} : (struct where){NULL, (size_t)p->pages};
  if (result) { plan_free(comm, p); return result; }
  p->next = comm->plans;
  comm->plans = p;
  *out = p;
  return ncclSuccess;
}

static ncclResult_t plan_take(struct ncclComm *comm, struct call *list, size_t n, struct plan **out) {
  for (size_t i = 0; i < n; i++) list[i].direct_ok = direct_ok(list + i);
  sweep(comm, 0, 0);
  struct plan *p = comm->plans;
  for (int same = 0; p && !same; p = same ? p : p->next) {
    same = !p->stale && p->n == n && plan_idle(p);
    for (size_t i = 0; same && i < n; i++) same = call_same(p->list + i, list + i);
  }
  const ncclResult_t result = p ? ncclSuccess : plan_make(comm, list, n, &p);
  if (result) return result;
  p->rows.at = p->rows.used = 0;
  for (size_t i = 0; i < n; i++) {
    struct call *c = p->list + i, made = list[i];
    made.bound = c->bound; made.segments = c->segments; made.complement = c->complement; made.complements = c->complements;
    made.direct = c->direct; made.at = c->at; made.copy = NULL;
    *c = made;
  }
  p->flight = 1;
  *out = p;
  return ncclSuccess;
}

static ncclResult_t present_group(struct ncclComm *comm, struct schedule *plans);
static void commit_group(struct ncclComm *comm, const struct schedule *plans);
static ncclResult_t plan_start(struct ncclComm *comm, struct plan *plan) {
  for (size_t i = 0; i < plan->n; i++) {
    struct steps *st = plan->list[i].bound;
    for (uint32_t k = 0; st && k < st->count; k++) st->done[k] = st->steps[k].op != MESH_STEP_SEND && !piece_bytes(st->steps + k);
  }
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct channel *ch = comm->channels + h;
    struct schedule *p = plan->plans + h;
    p->first[0] = ch->sends; p->first[1] = ch->lands;
    for (size_t k = 0; k < p->outs; k++) {
      struct message *m = p->out + k;
      m->entry = ch->sends + m->rel_entry;
      m->end = pages_add(ch->out, m->rel_end);
      m->c->bound->entry[m->step] = m->entry;
    }
    for (size_t k = 0; k < p->ins; k++) {
      struct message *m = p->in + k;
      m->entry = ch->lands + m->rel_entry;
      m->end = pages_add(ch->in, m->rel_end);
      m->need = m->rel_need ? ch->sends + m->rel_need : 0;
      m->sent = m->rel_sent.n ? pages_add(ch->out, m->rel_sent) : (mesh_pages){0};
      m->c->bound->entry[m->step] = m->entry;
    }
  }
  ncclResult_t result = place_group(comm, plan);
  if (!result) result = present_group(comm, plan->plans);
  if (!result) commit_group(comm, plan->plans);
  return result;
}

static struct ncclComm *touching;
static void touch(struct ncclComm *comm) {
  if (comm->touched) return;
  comm->touched = 1;
  comm->next_touched = touching;
  touching = comm;
}

static int room_for(const struct channel *ch, int list, uint64_t need) {
  return atomic_load_explicit(ch->view.presented[list], memory_order_relaxed) - atomic_load_explicit(ch->view.taken[list], memory_order_acquire) +
         need <= MESH_STREAM_ENTRIES;
}

static ncclResult_t room_wait(struct ncclComm *comm, const struct channel *ch, int list, uint64_t need) {
  const int mine = touching == comm;
  if (mine) { touching = comm->next_touched; comm->touched = 0; }
  atomic_fetch_add(&comm->runs, 1);
  pthread_mutex_unlock(&group_lock);
  ncclResult_t result = ncclSuccess;
  while (!result && !room_for(ch, list, need))
    if (revoked(comm)) result = revoked_fail(comm);
    else if (!mesh_stream_held(&comm->attach->context, ch->stream, *ch->wire))
      result = fail(ncclRemoteError, "communicator %s: its holding of the stream toward node %u ended before the group's entries there were "
                    "presented (design/transport.md section 14); none was presented", comm->name, ch->node);
  pthread_mutex_lock(&group_lock);
  atomic_fetch_sub(&comm->runs, 1);
  if (mine) touch(comm);
  return result ? result : comm_failure(comm);
}

static ncclResult_t present_group(struct ncclComm *comm, struct schedule *plans) {
  struct mesh_ctx *context = &comm->attach->context;
  for (uint32_t h = 0; h < comm->count; h++)
    for (int list = 0; list < 2; list++) {
      const struct channel *ch = comm->channels + h;
      const uint64_t need = list ? plans[h].ins : plans[h].outs;
      if (need > MESH_STREAM_ENTRIES)
        return fail(ncclSystemError, "communicator %s: the group lays %llu entries on its %s list toward node %u, and its bridge takes at most %u "
                    "not yet done (mesh.h MESH_STREAM_ENTRIES); none was presented", comm->name, (unsigned long long)need, list ? "landing" : "send",
                    ch->node, MESH_STREAM_ENTRIES);
      if (need && !room_for(ch, list, need)) {
        const ncclResult_t result = room_wait(comm, ch, list, need);
        if (result) return result;
      }
    }
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct channel *ch = comm->channels + h;
    const struct schedule *p = plans + h;
    int status = 0;
    for (size_t k = 0; !status && k < p->outs; k++) {
      const struct message *m = p->out + k;
      status = mesh_stream_present(context, ch->stream, 0, (struct mesh_entry){m->row, pages_to_word(m->pages), 0, pages_to_word(pages_sub(m->end, m->pages))});
    }
    for (size_t k = 0; !status && k < p->ins; k++)
      status = mesh_stream_present(context, ch->stream, 1, (struct mesh_entry){p->in[k].row, pages_to_word(p->in[k].pages), 0, 0});
    if (status)
      return comm_fail(comm, status == ENOLINK || status == EPERM ? ncclRemoteError : ncclSystemError, "communicator %s: an entry toward node %u "
                       "presented to its bridge: %s", comm->name, ch->node, strerror(status));
  }
  return ncclSuccess;
}

static void commit_group(struct ncclComm *comm, const struct schedule *plans) {
  for (uint32_t h = 0; h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    const struct schedule *p = plans + h;
    uint64_t *timed = comm->timed[((comm->groups - 1) % GROUP_TIMES) * comm->count + h];
    timed[0] = p->outs ? p->first[0] + 1 : 0;
    timed[1] = p->ins ? p->first[1] + p->ins : 0;
    if (p->outs) { ch->sends = p->first[0] + p->outs; ch->out = p->out[p->outs - 1].end; }
    if (p->ins) { ch->lands = p->first[1] + p->ins; ch->in = p->in[p->ins - 1].end; }
  }
}


static int ready(const struct steps *s, uint32_t k) { return s->before[k] < 0 || s->done[s->before[k]]; }

static mesh_pages count_of(_Atomic uint64_t *word) { return pages_from_word(atomic_load_explicit(word, memory_order_acquire)); }

static ncclResult_t held_fail(const struct ncclComm *comm, const struct channel *ch) {
  return fail(ncclRemoteError, "communicator %s: its holding of the stream toward node %u ended with the group's entries there not done (the "
              "stream is no longer its own: its link lost and paired again, or its bridge restarted; design/transport.md section 14)",
              comm->name, ch->node);
}
static ncclResult_t run(struct ncclComm *comm, struct schedule *plans) {
  const mesh_pages none = {0};
  for (;;) {
    if (revoked(comm)) return revoked_fail(comm);
    int open = 0;
    for (uint32_t h = 0; h < comm->count; h++) {
      struct channel *ch = comm->channels + h;
      struct schedule *p = plans + h;
      const uint64_t lands = p->first[1] + p->ins;
      if (ch->consumed < lands && pages_lost(count_of(ch->view.landed)))
        return fail(ncclRemoteError, "the link to node %u was lost at landing entry %llu (its holding's counts ~0)", ch->node,
                    (unsigned long long)ch->consumed);
      if ((ch->consumed < lands || ch->published < p->first[0] + p->outs) && !mesh_stream_held(&comm->attach->context, ch->stream, *ch->wire))
        return held_fail(comm, ch);
      while (ch->consumed < lands) {
        const struct message *m = p->in + (ch->consumed - p->first[1]);
        struct steps *st = m->c->bound;
        const struct mesh_step *step = st->steps + m->step;
        if (!ready(st, m->step) || ch->published < m->need) break;
        const mesh_pages landed = count_of(ch->view.landed), sent = pages_below(none, m->sent) ? count_of(ch->view.sent) : none;
        if (pages_lost(landed) || pages_lost(sent))
          return fail(ncclRemoteError, "the link to node %u was lost at landing entry %llu (its holding's counts ~0)", ch->node,
                      (unsigned long long)ch->consumed);
        if (pages_below(landed, m->end) || pages_below(sent, m->sent)) break;
        if (step->op == MESH_STEP_REDUCE) {
          unsigned char *to = address_of(st->buffer[step->buffer]) + m->offset;
          if (st->fresh && st->fresh[m->step]) memmove(to, address_of(read_at(m->c)) + m->offset, m->bytes);
          combine_into(ktype(m->c), kop(m->c), to, address_of(m->data), kcount(m->c, m->bytes));
        }
        st->done[m->step] = 1;
        ch->consumed++;
      }
      while (ch->published < p->first[0] + p->outs) {
        const struct message *m = p->out + (ch->published - p->first[0]);
        if (!ready(m->c->bound, m->step)) break;
        atomic_thread_fence(memory_order_release);
        atomic_store_explicit(ch->view.ready, pages_to_word(m->end), memory_order_release);
        ch->published++;
      }
      open |= ch->published < p->first[0] + p->outs || ch->consumed < lands;
    }
    if (!open) break;
  }
  for (int all = 0; !all;) {
    if (revoked(comm)) return revoked_fail(comm);
    all = 1;
    for (uint32_t h = 0; h < comm->count; h++) {
      const struct channel *ch = comm->channels + h;
      const mesh_pages sent = plans[h].outs ? count_of(ch->view.sent) : none;
      if (pages_lost(sent)) return fail(ncclRemoteError, "the link to node %u was lost before its SENDs completed (its holding's counts ~0)", ch->node);
      if (plans[h].outs && pages_below(sent, ch->out) && !mesh_stream_held(&comm->attach->context, ch->stream, *ch->wire))
        return held_fail(comm, ch);
      all &= !plans[h].outs || !pages_below(sent, ch->out);
    }
  }
  for (uint32_t h = 0; h < comm->count; h++) {
    comm->channels[h].waited = comm->channels[h].consumed;
    if (plans[h].outs) comm->channels[h].sent_waited = comm->channels[h].out;
  }
  return ncclSuccess;
}

static void stage(const struct call *c, unsigned char *own) {
  const size_t z = SIZE[c->type], n = c->count;
  if (reducing(c->what)) {
    const size_t all = elements(c);
    if (c->premultiplied) premultiply(c->type, own, c->send, c->pre, all);
    else memmove(own, c->send, all * z);
  } else if (c->what == MESH_BROADCAST && c->root == c->comm->rank) memmove(own, c->send, n * z);
  else if (c->what == MESH_ALLGATHER) {
    size_t length;
    const size_t before = segment_at(c, c->comm->rank, &length);
    memmove(own + before * z, c->send, length * z);
  }
}

static void finish(const struct call *c) {
  const size_t z = SIZE[c->type], n = c->count;
  const int nranks = c->comm->nranks;
  unsigned char *sum = address_of(c->sum), *recv = c->recv;
  for (uint32_t k = 0; k < c->complements; k++) {
    const uint64_t *x = c->complement + 3 * k;
    memmove(address_of(shifted(c->sum, (size_t)x[0] * z)), address_of(shifted(c->send_at, (size_t)x[1] * z)), (size_t)x[2] * z);
  }
  if (!sum) return;
  if (c->what == MESH_ALLREDUCE || (c->what == MESH_REDUCE && c->root == c->comm->rank)) {
    if (c->post) truncdiv(c->type, recv, sum, n, (uint64_t)nranks);
    else if (sum != recv) memmove(recv, sum, n * z);
  } else if (c->what == MESH_REDUCE_SCATTER) {
    size_t length;
    const size_t before = segment_at(c, c->comm->rank, &length);
    if (c->post) truncdiv(c->type, recv, sum + before * z, length, (uint64_t)nranks);
    else memmove(recv, sum + before * z, length * z);
  } else if ((c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER) && sum != recv) memmove(recv, sum, elements(c) * z);
}

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
  const struct redop *made = comm->ops + ((int)c->op - ncclNumOps);

  if (made->type != c->type) return fail(ncclInvalidArgument, "an operator made on the communicator reduces the datatype it was made for");
  if (made->unit) { c->combine = COMBINE_MERGE; c->unit = made->unit; return ncclSuccess; }

  if (made->at && c->stream)
    return fail(ncclInvalidArgument, "a PreMulSum scalar in device memory (ncclScalarDevice) on the Metal path: its kernels take the scalar as an "
                "argument written when the group is encoded, so none reads it where the GPU runs");
  if (made->at) c->scalar_at = made->at;
  else memcpy(c->pre, made->value, SIZE[c->type]);
  c->premultiplied = 1;
  return ncclSuccess;
}

static size_t own_segment(const struct call *c, size_t *sent, size_t *landed) {
  const int n = c->comm->nranks, me = c->comm->rank;
  *sent = seg_first(c->segments, me) * SIZE[c->type]; *landed = seg_first(c->segments + n, me) * SIZE[c->type];
  const uint64_t out = c->segments[me], in = c->segments[n + me];
  return (out < in ? out : in) * SIZE[c->type];
}


enum { GPU_ENDED, GPU_ERROR, GPU_UNCOMMITTED, GPU_REVOKED, GPU_HOLDING };

static int gpu_wait(struct ncclComm *comm, int group) {
  int ended = GPU_ENDED;
  while (ended == GPU_ENDED && finished && comm->last && metal_signaled(finished) < comm->last) {
    const uint64_t last = comm->last;
    void *command = metal_retain(comm->command);
    if (group) atomic_fetch_add(&comm->runs, 1);
    pthread_mutex_unlock(&group_lock);
    while (metal_signaled(finished) < last) {
      const int status = metal_ended(command);
      if (status) { ended = status < 0 ? GPU_ERROR : status == 2 ? GPU_UNCOMMITTED : GPU_ENDED; break; }
      if (group && revoked(comm)) { ended = GPU_REVOKED; break; }
      for (uint32_t h = 0; group && h < comm->count && ended == GPU_ENDED; h++)
        if (!mesh_stream_held(&comm->attach->context, comm->channels[h].stream, *comm->channels[h].wire)) ended = GPU_HOLDING;
      if (ended != GPU_ENDED) break;
      metal_wait(finished, last, MESH_PAUSE_NS);
    }
    metal_release(command);
    pthread_mutex_lock(&group_lock);
    if (group) atomic_fetch_sub(&comm->runs, 1);
    if (comm->last == last) break;
  }
  return ended;
}

static ncclResult_t gpu_finished(struct ncclComm *comm) {
  switch (gpu_wait(comm, 1)) {
  case GPU_ERROR:
    return fail(ncclSystemError, "communicator %s's last GPU program's command buffer ended in an error before its host-path group", comm->name);
  case GPU_UNCOMMITTED:
    return fail(ncclInvalidUsage, "a host-path group on communicator %s before the command buffer of its last GPU program is committed", comm->name);
  case GPU_REVOKED: return revoked_fail(comm);
  case GPU_HOLDING:
    return fail(ncclRemoteError, "communicator %s: a holding of its streams ended before its host-path group (design/transport.md section 14)",
                comm->name);
  default: return comm_failure(comm);
  }
}

static int comm_issued(const struct ncclComm *comm);
static void calls_free(struct call *list, size_t n);
static void operands_at(const struct plan *plan, struct call *c) {
  struct steps *st = c->bound;
  if (counted(c) && st) { st->buffer[0] = c->send_at; st->buffer[1] = c->recv_at; }
  if (counted(c) && st && st->relayed) st->buffer[2] = shifted(plan->base, st->at[2]);
  if (p2p(c) && st) st->buffer[0] = c->what == WHAT_SEND ? c->send_at : c->recv_at;
  if (p2p(c) || counted(c)) return;
  c->sum = c->at == SIZE_MAX ? c->recv_at : shifted(plan->base, c->at);
  for (uint32_t b = 0; st && b < st->buffers; b++) st->buffer[b] = !b ? c->sum : st->at[b] == SIZE_MAX ? (struct where){NULL, 0} : shifted(plan->base, st->at[b]);
}

static ncclResult_t launch(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  for (size_t i = 0; i < n; i++) {
    if (list[i].stream) return fail(ncclInvalidUsage, "a group on several streams");
    if (reducing(list[i].what) && (result = reduction(list + i))) return result;
    remote |= crosses(list + i);
  }
  if (revoked(comm)) return revoked_fail(comm);
  for (size_t i = 0; i < n; i++)
    if (list[i].scalar_at) memcpy(list[i].pre, list[i].scalar_at, SIZE[list[i].type]);
  pthread_mutex_lock(&group_lock);
  group_numbered(comm);
  if (remote) result = comm_failure(comm);
  if (!result && remote && comm_issued(comm))
    result = fail(ncclInvalidUsage, "a host-path group while this communicator's GPU groups are issued and not complete (ncclMeshComplete), or "
                  "an exchange's start is not bound to the kernel that publishes it (ncclMeshBind)");
  if (!result && remote && finished && comm->last) result = gpu_finished(comm);
  if (!result) result = operands_check(list, n);
  struct plan *plan = NULL;
  if (!result) result = plan_take(comm, list, n, &plan);
  if (result) { pthread_mutex_unlock(&group_lock); return result; }
  struct call *const calls = plan->list;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = calls + i;
    const size_t bytes = elements(c) * SIZE[c->type];
    operands_at(plan, c);
    if (c->what == WHAT_SEND && c->peer == comm->rank) {
      if (!(c->copy = malloc(bytes ? bytes : 1))) result = fail(ncclSystemError, "out of memory");
      else memcpy(c->copy, c->send, bytes);
    }
    if (p2p(c) || counted(c) || c->direct) continue;
    stage(c, address_of(c->sum));
    for (uint32_t b = 1; c->bound && b < c->bound->buffers; b++)
      if (c->bound->at[b] != SIZE_MAX) memcpy(address_of(c->bound->buffer[b]), address_of(c->sum), bytes);
  }
  if (!result && remote && !(result = plan_start(comm, plan))) atomic_fetch_add(&comm->runs, 1);
  const int running = remote && !result;
  pthread_mutex_unlock(&group_lock);
  if (running && (result = run(comm, plan->plans))) {
    pthread_mutex_lock(&group_lock);
    plan->stale = comm->sweeps = 1;
    comm_fail(comm, result, "%s", last.text);
    pthread_mutex_unlock(&group_lock);
  }
  if (running) atomic_fetch_sub(&comm->runs, 1);
  for (size_t i = 0, self = 0; i < n && !result; i++) {
    struct call *c = calls + i;
    if (c->what == WHAT_RECV && c->peer == comm->rank) {
      while (self < n && !(calls[self].what == WHAT_SEND && calls[self].peer == comm->rank)) self++;
      if (self < n) { memmove(c->recv, calls[self].copy, c->count * SIZE[c->type]); self++; }
    } else if (counted(c)) {
      size_t sent, landed, own = own_segment(c, &sent, &landed);
      if (own) memmove((unsigned char *)c->recv + landed, (const unsigned char *)c->send + sent, own);
    } else if (!p2p(c)) finish(c);
  }
  for (size_t i = 0; i < n; i++) {
    free(calls[i].copy);
    calls[i].copy = NULL;
  }
  pthread_mutex_lock(&group_lock);
  plan->flight = 0;
  pthread_mutex_unlock(&group_lock);
  return result;
}

static ncclResult_t gpu(struct metal_result step, const char *what) {
  return step.status ? fail(ncclSystemError, "%s on the GPU: %s", what, step.message) : ncclSuccess;
}

struct inflight {
  uint64_t ticket;
  struct ncclComm *comm;
  struct plan *plan;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t (*start)[2], (*end)[2];
  uint32_t lone;
  int composed;
  struct inflight *next_group;
};

static uint32_t lone_of(const struct inflight *g) {
  uint32_t lone = g->comm->count;
  for (uint32_t h = 0; h < g->comm->count; h++)
    if (g->end[h][0] > g->start[h][0] || g->end[h][1] > g->start[h][1]) {
      if (lone != g->comm->count) return METAL_NONE;
      lone = h;
    }
  return lone;
}
static struct inflight *inflight, *inflight_last;
static uint64_t tickets;
static _Atomic uint64_t retired;

static struct loss { uint64_t ticket; ncclResult_t result; char reason[256]; } *losses;
static size_t lost;

static void call_parts_free(struct call *c) {
  steps_free(c->bound);
  free(c->copy);
  free(c->segments);
  free(c->complement);
}
static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) call_parts_free(list + i);
  free(list);
}

static void inflight_free(struct inflight *g, int retired) {
  if (g->plan && retired) g->plan->leaving = 1;
  else if (g->plan) { g->plan->stale = g->comm->sweeps = 1; g->plan->flight = 0; }
  free(g->start);
  free(g);
}

static int comm_issued(const struct ncclComm *comm) {
  for (const struct inflight *g = inflight; g; g = g->next_group)
    if (g->comm == comm) return 1;
  return comm->unpublished > 0;
}

static void inflight_lose(struct ncclComm *comm) {
  inflight_last = NULL;
  for (struct inflight **at = &inflight; *at;) {
    struct inflight *g = *at;
    if (g->comm != comm) { inflight_last = g; at = &g->next_group; continue; }
    struct loss *grown = realloc(losses, (lost + 1) * sizeof *grown);
    if (grown) {
      losses = grown;
      losses[lost] = (struct loss){g->ticket, comm->failed ? comm->failed : ncclRemoteError, {0}};
      snprintf(losses[lost++].reason, sizeof grown->reason, "%s", comm->failed ? comm->failure.text : "its communicator was destroyed");
    }
    *at = g->next_group;
    inflight_free(g, 0);
  }
  atomic_store(&retired, inflight ? inflight->ticket - 1 : tickets);
}

static struct inflight *inflight_at(struct ncclComm *comm, uint32_t h, int list, uint64_t at) {
  for (struct inflight *g = inflight; g; g = g->next_group)
    if (g->comm == comm && at < g->end[h][list]) return at >= g->start[h][list] ? g : NULL;
  return NULL;
}
static struct message *out_at(struct inflight *g, uint32_t h, uint64_t at) { return g->plans[h].out + (at - g->start[h][0]); }
static struct message *in_at(struct inflight *g, uint32_t h, uint64_t at) { return g->plans[h].in + (at - g->start[h][1]); }

static ncclResult_t outside_groups(struct ncclComm *comm, uint32_t h, int list, uint64_t at) {
  return fail(ncclInternalError, "%s entry %llu of channel %u of %s is in no group in flight", list ? "landing" : "send", (unsigned long long)at, h,
              comm->name);
}

static int publishable(struct ncclComm *comm, uint32_t h, uint64_t at) {
  struct inflight *g = inflight_at(comm, h, 0, at);
  const struct message *m = g ? out_at(g, h, at) : NULL;
  return m && ready(m->c->bound, m->step);
}

static ncclResult_t publish(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t upto, int *published) {
  struct channel *ch = comm->channels + h;
  struct inflight *g = inflight_at(comm, h, 0, upto - 1);
  if (!g) return outside_groups(comm, h, 0, upto - 1);
  const struct message *m = out_at(g, h, upto - 1);
  const struct metal_release release = {ch->buffer, ch->ready, m->end};
  void *operands[64];
  size_t n = 0;
  for (uint64_t at = ch->published; at < upto; at++) {
    if (n == sizeof operands / sizeof *operands) { metal_uses(program, operands, n, 0); n = 0; }
    operands[n++] = out_at(g, h, at)->data.buffer;
  }
  metal_uses(program, operands, n, 0);
  const ncclResult_t result = gpu(metal_publish(program, &release), "a publication");
  if (!result) { ch->published = upto; *published = 1; }
  return result;
}

static uint64_t consumed_past(struct ncclComm *comm, uint32_t h, uint64_t at) {
  for (struct inflight *g; at < comm->channels[h].lands && (g = inflight_at(comm, h, 1, at)) && g->composed;) at++;
  return at;
}

static ncclResult_t consume(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t at, uint32_t base, int fold) {
  struct channel *ch = comm->channels + h;
  struct inflight *g = inflight_at(comm, h, 1, at);
  if (!g) return outside_groups(comm, h, 1, at);
  const struct message *m = in_at(g, h, at);
  struct steps *st = m->c->bound;
  const struct mesh_step *step = st->steps + m->step;
  ncclResult_t result = ncclSuccess;
  if (base != METAL_NONE && st->before[m->step] >= 0) {
    const int32_t j = st->before[m->step];
    result = gpu(metal_depend(program, base + st->channel[j], st->entry[j] + 1), "a receive's dependency");
  }
  mesh_pages sent = m->sent;
  const mesh_pages last = g->end[h][0] > g->start[h][0] ? out_at(g, h, g->end[h][0] - 1)->end : (mesh_pages){0};
  if (fold && at + 1 == g->end[h][1] && ch->published >= g->end[h][0] && pages_below(sent, last)) sent = last;
  const struct metal_wait wait = {ch->buffer, comm->violation, m->data.buffer, ch->landed, ch->sent, m->data.offset, m->end, sent, at, h};
  if (!result && step->op == MESH_STEP_REDUCE) {
    const struct where to = shifted(st->buffer[step->buffer], m->offset);
    if (st->fresh && st->fresh[m->step]) {
      const struct where with = shifted(read_at(m->c), m->offset);
      result = gpu(metal_spin(program, &wait), "a wait");
      if (!result)
        result = gpu(metal_combine3(program, m->c->type, m->c->combine, to.buffer, to.offset, m->data.buffer, m->data.offset, with.buffer, with.offset,
                                    m->bytes / SIZE[m->c->type]), "a first combine");
    } else result = gpu(metal_land(program, ktype(m->c), kop(m->c), to.buffer, to.offset, kcount(m->c, m->bytes), &wait), "a landing");
  } else if (!result) result = gpu(metal_spin(program, &wait), "a wait");
  if (result) return result;
  if (pages_below(ch->sent_waited, sent)) ch->sent_waited = sent;
  st->done[m->step] = 1;
  if (at == ch->consumed) ch->consumed = consumed_past(comm, h, at + 1);
  if (ch->waited < at + 1) ch->waited = at + 1;
  return ncclSuccess;
}

static uint32_t one_track(const struct inflight *through) {
  uint32_t h = METAL_NONE;
  for (const struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group) {
    if (g->composed) continue;
    if (g->comm != through->comm || g->lone == METAL_NONE) return METAL_NONE;
    if (g->lone == g->comm->count) continue;
    if (h != METAL_NONE && h != g->lone) return METAL_NONE;
    h = g->lone;
  }
  return h;
}

static ncclResult_t complete_track(struct metal_program *program, struct inflight *through, uint32_t h, int *published) {
  struct ncclComm *const comm = through->comm;
  struct channel *ch = comm->channels + h;
  const uint64_t out = through->end[h][0], in = through->end[h][1];
  ncclResult_t result = ncclSuccess;
  touch(comm);
  metal_rows(program, &through->plan->rows);
  metal_gate(program, comm->violation);
  while (!result && (ch->published < out || ch->consumed < in)) {
    uint64_t upto = ch->published;
    while (upto < out && publishable(comm, h, upto)) upto++;
    if (upto > ch->published) { result = publish(program, comm, h, upto, published); continue; }
    struct inflight *x = ch->consumed < in ? inflight_at(comm, h, 1, ch->consumed) : NULL;
    const struct message *r = x ? in_at(x, h, ch->consumed) : NULL;
    if (r && ready(r->c->bound, r->step) && ch->published >= r->need) result = consume(program, comm, h, ch->consumed, METAL_NONE, 1);
    else
      result = fail(ncclInternalError, "channel %u's entries admit no order on one track (published %llu of %llu, consumed %llu of %llu)", h,
                    (unsigned long long)ch->published, (unsigned long long)out, (unsigned long long)ch->consumed, (unsigned long long)in);
  }
  return result;
}

static ncclResult_t complete(struct metal_program *program, struct inflight *through, int *published) {
  struct inflight *last = NULL;
  for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group)
    if (!g->composed) last = g;
  if (!last) return ncclSuccess;
  through = last;
  const uint32_t lone = one_track(through);
  if (lone != METAL_NONE) return complete_track(program, through, lone, published);
  size_t comms = 0, room = 0;
  struct ncclComm **comm = NULL;
  uint32_t *base = NULL, tracks = 0;
  ncclResult_t result = ncclSuccess;
  for (struct inflight *g = inflight; g && !result; g = g == through ? NULL : g->next_group) {
    size_t k = 0;
    if (g->composed) continue;
    while (k < comms && comm[k] != g->comm) k++;
    if (k < comms) continue;
    if (comms == room) {
      room = room ? 2 * room : 4;
      struct ncclComm **c = realloc(comm, room * sizeof *c);
      uint32_t *b = c ? realloc(base, room * sizeof *b) : NULL;
      if (c) comm = c;
      if (!b) { result = fail(ncclSystemError, "out of memory"); break; }
      base = b;
    }
    comm[comms] = g->comm; base[comms++] = tracks; tracks += g->comm->count;
  }
  uint64_t *consumed = result ? NULL : calloc((size_t)tracks + 1, sizeof *consumed), (*end)[2] = NULL;
  if (!result && !consumed) result = fail(ncclSystemError, "out of memory");
  for (size_t k = 0; k < comms && !result; k++) {
    struct ncclComm *const c = comm[k];
    touch(c);
    metal_gate(program, c->violation);
    free(end);
    if (!(end = calloc((size_t)c->count + 1, sizeof *end))) { result = fail(ncclSystemError, "out of memory"); break; }
    for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group)
      if (g->comm == c) {
        memcpy(end, g->end, c->count * sizeof *end);
        if (g->plan) metal_rows(program, &g->plan->rows);
      }
    for (uint32_t h = 0; h < c->count && !result; h++) {
      struct channel *ch = c->channels + h;
      const uint32_t track = base[k] + h;
      consumed[track] = ch->consumed;
      while (!result && ch->published < end[h][0]) {
        const uint64_t at = ch->published;
        struct inflight *g = inflight_at(c, h, 0, at);
        if (!g || g->composed) { result = g ? fail(ncclInternalError, "a composed crossing's publication on two tracks") : outside_groups(c, h, 0, at); break; }
        const struct message *m = out_at(g, h, at);
        result = gpu(metal_record(program, track, METAL_PUBLISH, at, 0), "a publication's record");
        if (!result && m->c->bound->before[m->step] >= 0) {
          const struct steps *prior = m->c->bound;
          const int32_t j = prior->before[m->step];
          result = gpu(metal_depend(program, base[k] + prior->channel[j], prior->entry[j] + 1), "a SEND's dependency");
        }
        if (!result) result = publish(program, c, h, at + 1, published);
      }
      while (!result && ch->consumed < end[h][1]) {
        const uint64_t at = ch->consumed;
        struct inflight *g = inflight_at(c, h, 1, at);
        if (!g || g->composed) { result = g ? fail(ncclInternalError, "a composed crossing's landing on two tracks") : outside_groups(c, h, 1, at); break; }
        result = gpu(metal_record(program, track, METAL_CONSUME, at, in_at(g, h, at)->need), "a consumption's record");
        if (!result) result = consume(program, c, h, at, base[k], 0);
      }
    }
  }
  if (!result) result = gpu(metal_record(program, METAL_NONE, 0, 0, 0), "the tracks' end");
  if (!result) result = gpu(metal_tracks(program, tracks, consumed), "the tracks");
  else metal_record(program, METAL_NONE, 0, 0, 0);
  free(end); free(consumed); free(comm); free(base);
  return result;
}


static int meets(const struct inflight *x, const struct inflight *g) {
  for (uint32_t h = 0; !x->composed && x->comm == g->comm && h < g->comm->count; h++)
    if (g->end[h][0] > g->start[h][0] && x->end[h][0] > x->start[h][0] && g->comm->channels[h].published < x->end[h][0]) return 1;
  return 0;
}

static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
  if (g->plan) metal_rows(program, &g->plan->rows);
  metal_gate(program, comm->violation);
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    const struct channel *ch = comm->channels + h;
    if (ch->published < g->start[h][0] || ch->published >= g->end[h][0]) continue;
    uint64_t upto = ch->published;
    while (upto < g->end[h][0] && publishable(comm, h, upto)) upto++;
    if (upto > ch->published) result = publish(program, comm, h, upto, published);
  }
  return result;
}

static ncclResult_t walk(struct metal_program *program, struct inflight *issuing, struct inflight *through, int *published) {
  ncclResult_t result = ncclSuccess;
  struct inflight *g = issuing;
  if (through) { result = complete(program, through, published); g = through->next_group; }
  for (; g && !result; g = g->next_group) result = issue_group(program, g, published);
  return result;
}

static int on_gpu(ncclDataType_t t) { return t != ncclFloat64; }
static struct where where_of(const void *argument) {
  const ncclMeshBuffer *b = argument;
  return b ? (struct where){b->buffer, b->offset} : (struct where){NULL, 0};
}


struct spans { uint64_t (*span)[2]; size_t count, room; };

static int spans_meet(const struct spans *w, uint64_t lo, uint64_t hi) {
  uint64_t covered = 0;
  for (size_t k = 0; k < w->count; k++) {
    const uint64_t a = w->span[k][0] > lo ? w->span[k][0] : lo, b = w->span[k][1] < hi ? w->span[k][1] : hi;
    if (a < b) covered += b - a;
  }
  return !covered ? 0 : covered == hi - lo ? 1 : -1;
}

static int spans_add(struct spans *w, uint64_t lo, uint64_t hi) {
  if (lo >= hi) return 0;
  if (w->count == w->room) {
    const size_t room = w->room ? 2 * w->room : 8;
    uint64_t (*grown)[2] = realloc(w->span, room * sizeof *grown);
    if (!grown) return ENOMEM;
    w->span = grown; w->room = room;
  }
  size_t at = 0;
  while (at < w->count && w->span[at][0] < lo) at++;
  memmove(w->span + at + 1, w->span + at, (w->count - at) * sizeof *w->span);
  w->span[at][0] = lo; w->span[at][1] = hi; w->count++;
  size_t kept = 0;
  for (size_t k = 0; k < w->count; k++)
    if (kept && w->span[k][0] <= w->span[kept - 1][1]) { if (w->span[k][1] > w->span[kept - 1][1]) w->span[kept - 1][1] = w->span[k][1]; }
    else { w->span[kept][0] = w->span[k][0]; w->span[kept][1] = w->span[k][1]; kept++; }
  w->count = kept;
  return 0;
}

static int plan_direct(struct call *c) {
  struct steps *st = c->bound;
  uint64_t lo = 0, hi = elements(c);
  if (!c->direct_ok || !st->count) return 0;
  if (c->what == MESH_ALLGATHER) { size_t length; lo = segment_at(c, c->comm->rank, &length); hi = lo + length; }
  const uint32_t B = st->buffers;
  struct spans *w = calloc(B, sizeof *w);
  st->reads = calloc(st->count, 1); st->fresh = calloc(st->count, 1); st->needed = calloc(B, 1);
  int direct = w && st->reads && st->fresh && st->needed;
  st->lo = lo;
  for (uint32_t k = 0; direct && k < st->count; k++) {
    const struct mesh_step *step = st->steps + k;
    const uint64_t from = step->first, to = step->first + step->piece.elements;
    const int meet = spans_meet(w + step->buffer, from, to), inside = from >= lo && to <= hi;
    if (meet < 0 || (!meet && step->op != MESH_STEP_COPY && !inside)) direct = 0;
    else if (step->op == MESH_STEP_SEND) st->reads[k] = !meet;
    else {
      if (step->op == MESH_STEP_REDUCE && !meet) st->fresh[k] = st->reads[k] = 1;
      st->needed[step->buffer] = 1;
      if (spans_add(w + step->buffer, from, to)) direct = 0;
    }
  }
  uint64_t first = lo, last = hi;
  if (c->what == MESH_REDUCE_SCATTER) { size_t length; first = segment_at(c, c->comm->rank, &length); last = first + length; }
  if (c->what == MESH_REDUCE && c->root != c->comm->rank) last = first;
  if (direct) {
    c->complement = calloc(3 * (w[0].count + 1), sizeof *c->complement);
    direct = c->complement != NULL;
    for (size_t k = 0, at = first; direct && at < last; k++) {
      const uint64_t end = k < w[0].count ? (w[0].span[k][0] < last ? w[0].span[k][0] : last) : last;
      if (end > at) {
        uint64_t *x = c->complement + 3 * c->complements++;
        x[0] = at; x[1] = at - lo; x[2] = end - at;
      }
      if (k >= w[0].count) break;
      if (w[0].span[k][1] > at) at = w[0].span[k][1];
    }
  }
  for (uint32_t x = 0; w && x < B; x++) free(w[x].span);
  free(w);
  if (!direct) {
    free(st->reads); free(st->fresh); free(st->needed); free(c->complement);
    st->reads = st->fresh = st->needed = NULL; c->complement = NULL; c->complements = 0;
  }
  return direct;
}

static ncclResult_t complement_copy(struct metal_program *program, const struct call *c) {
  const size_t z = SIZE[c->type];
  ncclResult_t result = ncclSuccess;
  for (uint32_t k = 0; k < c->complements && !result; k++) {
    const uint64_t *x = c->complement + 3 * k;
    result = gpu(metal_copy(program, METAL_PLAIN, c->sum.buffer, c->sum.offset + x[0] * z, c->send_at.buffer, c->send_at.offset + x[1] * z, x[2] * z),
                 "the operand's own ranges");
  }
  return result;
}

static uint64_t scalar_bits(const struct call *c) {
  uint64_t bits = 0;
  if (floating(c->type)) { const float f = (float)load(c->type, c->pre); uint32_t u; memcpy(&u, &f, 4); return u; }
  memcpy(&bits, c->pre, SIZE[c->type]);
  return bits;
}

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
    } else if (p2p(c) || counted(c)) continue;
    else if (c->what == MESH_ALLREDUCE || (c->what == MESH_REDUCE && c->root == comm->rank) || c->what == MESH_REDUCE_SCATTER) {
      size_t length = count;
      const size_t before = c->what == MESH_REDUCE_SCATTER ? segment_at(c, (int)r, &length) : 0;
      const struct where from = {sum.buffer, sum.offset + before * z};
      if (c->post) result = gpu(metal_truncdiv(program, c->type, recv.buffer, recv.offset, from.buffer, from.offset, length, (uint64_t)comm->nranks), "the result");
      else if (!same(from, recv)) result = gpu(metal_copy(program, METAL_LAND, recv.buffer, recv.offset, from.buffer, from.offset, length * z), "the result");
    } else if ((c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER) && !same(sum, recv))
      result = gpu(metal_copy(program, METAL_LAND, recv.buffer, recv.offset, sum.buffer, sum.offset, elements(c) * z), "the result");
  }
  return result;
}

static ncclResult_t retire(struct metal_program *program) {
  ncclResult_t result = ncclSuccess;
  while (inflight && !result) {
    struct inflight *g = inflight;
    struct ncclComm *comm = g->comm;
    for (uint32_t h = 0; h < comm->count; h++)
      if (comm->channels[h].consumed < g->end[h][1] || comm->channels[h].published < g->end[h][0]) return ncclSuccess;
    if (!program && !g->composed) return ncclSuccess;
    if (program) touch(comm);
    if (program && g->plan) metal_rows(program, &g->plan->rows);
    if (program) metal_gate(program, comm->violation);
    for (uint32_t h = 0; h < comm->count && !result && !g->composed; h++) {
      struct channel *ch = comm->channels + h;
      if (g->end[h][0] == g->start[h][0]) continue;
      const mesh_pages last = out_at(g, h, g->end[h][0] - 1)->end;
      if (!pages_below(ch->sent_waited, last)) continue;
      const struct metal_wait wait = {ch->buffer, comm->violation, NULL, 0, ch->sent, 0, (mesh_pages){0}, last, g->end[h][0] - 1, h};
      if (!(result = gpu(metal_spin(program, &wait), "a wait for the group's SENDs"))) ch->sent_waited = last;
    }
    if (!result && g->n) result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g, 1);
  }
  return result;
}

static void stamp(struct metal_program *program, struct ncclComm *comm, struct plan *p) {
  if (p->buffer && metal_stamp(program, p->buffer, p->word).status) p->stale = comm->sweeps = 1;
  p->reader += p->buffer != NULL;
  p->flight = !p->leaving;
  p->leaving = 0;
}

static ncclResult_t program_end(struct metal_program *program, int borrowed, ncclResult_t result) {
  struct ncclComm *held_comm = NULL;
  struct plan *held = NULL;
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched)
    for (struct plan *p = comm->plans; p; p = p->next)
      if (p->flight && p->buffer && metal_rows(program, &p->rows) == &p->rows) { held = p; held_comm = comm; }
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched)
    for (struct plan *p = comm->plans; p; p = p->next)
      if (p->flight && p != held) stamp(program, comm, p);
  if (held) stamp(program, held_comm, held);
  const uint64_t count = borrowed || result ? 0 : ++issued;
  void *command = count ? metal_command(program) : NULL;
  metal_end(program, count ? (finished ? finished : (finished = metal_event())) : NULL, count);
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched) {
    comm->touched = 0;
    if (!count) continue;
    comm->last = count;
    metal_release(comm->command);
    comm->command = metal_retain(command);
  }
  metal_release(command);
  touching = NULL;
  return result;
}

static void walk_failed(struct ncclComm *comm, uint64_t ticket, ncclResult_t result) {
  char why[sizeof last.text];
  snprintf(why, sizeof why, "%s", last.text);
  for (; comm || (inflight && inflight->ticket <= ticket); comm = NULL) comm_fail(comm ? comm : inflight->comm, result, "%s", why);
  fail(result, "%s", why);
}

static ncclResult_t losses_through(uint64_t ticket) {
  for (struct inflight *g = inflight; g && g->ticket <= ticket;)
    if (comm_failure(g->comm)) g = inflight;
    else g = g->next_group;
  ncclResult_t result = ncclSuccess;
  size_t kept = 0;
  for (size_t i = 0; i < lost; i++)
    if (losses[i].ticket > ticket) losses[kept++] = losses[i];
    else if (!result) result = fail(losses[i].result, "the group was lost with its communicator: %s", losses[i].reason);
  lost = kept;
  return result;
}


static ncclResult_t launch_metal(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclMeshIssue *const issue = list[0].issue;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  if (issue) *issue = (ncclMeshIssue){0, 0};
  for (size_t i = 0; i < n && !result; i++) {
    if (list[i].command != list[0].command || list[i].encoder != list[0].encoder)
      result = fail(ncclInvalidUsage, "a group on several command buffers or encoders");
    else if (reducing(list[i].what) && !on_gpu(list[i].type))
      result = fail(ncclInvalidArgument, "a float64 reduction on the Metal path (Apple GPUs have no float64 arithmetic)");
    else if (reducing(list[i].what)) result = reduction(list + i);
    remote |= crosses(list + i);
  }
  const int metal4 = list[0].table != NULL;
  if (!result && metal4 && (!list[0].encoder || !list[0].residency))
    result = fail(ncclInvalidArgument, "a Metal 4 ncclMeshStream (argumentTable) without its encoder or residency set");
  else if (!result && !metal4 && !list[0].command) result = fail(ncclInvalidArgument, "an ncclMeshStream without a command buffer");
  if (!result && revoked(comm)) result = revoked_fail(comm);
  pthread_mutex_lock(&group_lock);
  if (!result) group_numbered(comm);
  if (!result && remote) result = comm_failure(comm);
  if (!result) result = operands_check(list, n);
  struct plan *plan = NULL;
  if (!result) result = plan_take(comm, list, n, &plan);
  if (result) { pthread_mutex_unlock(&group_lock); calls_free(list, n); return result; }
  struct call *const calls = plan->list;
  void *const violation = remote ? comm->violation : NULL;
  const char *message = NULL;
  struct metal_program *program = metal4 ? metal_begin4(list[0].encoder, list[0].table, list[0].residency, violation, &message)
                                         : metal_begin(list[0].command, list[0].encoder, violation, &message);
  if (!program) {
    plan->flight = 0;
    pthread_mutex_unlock(&group_lock); calls_free(list, n);
    return fail(ncclSystemError, "%s", message);
  }
  touch(comm);
  metal_rows(program, &plan->rows);
  for (uint32_t h = 0; remote && h < comm->count; h++) metal_resident(program, comm->channels[h].buffer);
  if (plan->buffer) metal_resident(program, plan->buffer);
  const size_t r = (size_t)comm->rank;
  for (size_t i = 0; i < n; i++) operands_at(plan, calls + i);
  struct inflight *g = NULL;
  int published = 0;
  uint64_t ticket = 0;
  if (!result && remote) {
    if (!(g = calloc(1, sizeof *g)) || !(g->start = calloc(2 * (size_t)comm->count + 1, sizeof *g->start))) result = fail(ncclSystemError, "out of memory");
    else {
      *g = (struct inflight){.comm = comm, .plan = plan, .list = plan->list, .n = n, .plans = plan->plans, .start = g->start, .end = g->start + comm->count};
      for (uint32_t h = 0; h < comm->count; h++) { g->start[h][0] = comm->channels[h].sends; g->start[h][1] = comm->channels[h].lands; }
      result = plan_start(comm, plan);
    }
    if (result && g) { free(g->start); free(g); g = NULL; }
    if (!result) {
      for (uint32_t h = 0; h < comm->count; h++) { g->end[h][0] = comm->channels[h].sends; g->end[h][1] = comm->channels[h].lands; }
      g->lone = lone_of(g);
      g->ticket = ticket = ++tickets;
      if (inflight_last) inflight_last->next_group = g; else inflight = g;
      inflight_last = g;
    }
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = calls + i;
    struct steps *st = c->bound;
    const size_t z = SIZE[c->type], bytes = elements(c) * z, all = elements(c);
    const struct where send = c->send_at, own = c->sum;
    if (counted(c)) {
      size_t sent, landed, part = own_segment(c, &sent, &landed);
      if (part && !same((struct where){send.buffer, send.offset + sent}, (struct where){c->recv_at.buffer, c->recv_at.offset + landed}))
        result = gpu(metal_copy(program, METAL_PLAIN, c->recv_at.buffer, c->recv_at.offset + landed, send.buffer, send.offset + sent, part),
                     "an all-to-all's own segment");
      continue;
    }
    if (p2p(c)) continue;
    size_t length = c->count;
    const size_t before = c->what == MESH_ALLGATHER ? segment_at(c, (int)r, &length) : 0;
    const struct where segment = {own.buffer, own.offset + before * z};
    if (!c->direct && reducing(c->what) && c->premultiplied)
      result = gpu(metal_premultiply(program, c->type, own.buffer, own.offset, send.buffer, send.offset, all, scalar_bits(c)), "the operand");
    else if (!c->direct && (reducing(c->what) || (c->what == MESH_BROADCAST && c->root == comm->rank) || c->what == MESH_ALLGATHER) &&
             !same(segment, send))
      result = gpu(metal_copy(program, METAL_PLAIN, segment.buffer, segment.offset, send.buffer, send.offset, reducing(c->what) ? bytes : length * z),
                   "the operand");
    for (uint32_t b = 1; st && b < st->buffers && !result && !c->direct; b++)
      if (st->at[b] != SIZE_MAX)
        result = gpu(metal_copy(program, METAL_PLAIN, st->buffer[b].buffer, st->buffer[b].offset, own.buffer, own.offset, bytes), "a scratch copy");
  }
  if (!result && g) {
    result = walk(program, issue ? g : NULL, issue ? NULL : g, &published);
    metal_gate(program, violation);
    for (size_t i = 0; i < n && !result; i++) result = complement_copy(program, calls + i);
    if (!result) result = retire(program);
  } else if (!result) result = epilogue(program, calls, n);
  plan->leaving = !g;
  result = program_end(program, list[0].encoder != NULL, result);
  if (result && g) walk_failed(comm, issue ? 0 : ticket, result);
  if (issue && !result) { issue->ticket = ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  pthread_mutex_unlock(&group_lock);
  calls_free(list, n);
  if (result) comm->async = result;
  return result;
}

ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream *stream, int *published) {
  int any = 0;
  if (published) *published = 0;
  if (!stream || (!stream->commandBuffer && !(stream->argumentTable && stream->commandEncoder && stream->residencySet)))
    return fail(ncclInvalidArgument, "ncclMeshComplete without a command buffer or a Metal 4 encoder, argument table and residency set");
  pthread_mutex_lock(&group_lock);
  const ncclResult_t loss = losses_through(ticket);
  ncclResult_t result = ncclSuccess;
  if (inflight && ticket >= inflight->ticket) {
    struct inflight *through = inflight;
    while (through->next_group && through->next_group->ticket <= ticket) through = through->next_group;
      const char *message = NULL;
    struct metal_program *program = stream->argumentTable
      ? metal_begin4(stream->commandEncoder, stream->argumentTable, stream->residencySet, inflight->comm->violation, &message)
      : metal_begin(stream->commandBuffer, stream->commandEncoder, inflight->comm->violation, &message);
    if (!program) result = fail(ncclSystemError, "%s", message);
    else {
      for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group) {
        for (uint32_t h = 0; h < g->comm->count; h++) metal_resident(program, g->comm->channels[h].buffer);
        if (g->plan && g->plan->buffer) metal_resident(program, g->plan->buffer);
      }
      result = walk(program, NULL, through, &any);
      if (!result) result = retire(program);
      result = program_end(program, stream->commandEncoder != NULL, result);
      if (result) walk_failed(NULL, ticket, result);
    }
  }
  pthread_mutex_unlock(&group_lock);
  if (published) *published = any;
  return result ? result : loss;
}

uint64_t ncclMeshRetired(void) { return atomic_load(&retired); }

struct slot { void *pages, *buffer; struct mesh_table *table; struct metal_rows rows; uint64_t row, start, *ends; uint32_t bound; int unbound; };
struct peer_part { int rank, sends; uint32_t channel; size_t count, landing; };
struct ncclMeshRequest {
  struct ncclComm *comm;
  ncclDataType_t type;
  size_t count, table, bytes;
  struct peer_part *part;
  uint32_t peers, nslots, turn;
  struct slot *slots;
  uint64_t starts;
  struct ncclMeshRequest *next;
};

static int entry_done(struct ncclComm *comm, const struct channel *ch, _Atomic uint64_t *word, uint64_t end) {
  const mesh_pages count = count_of(word);
  return !end || pages_lost(count) || !pages_below(count, pages_from_word(end)) || !mesh_stream_held(&comm->attach->context, ch->stream, *ch->wire);
}

static int plan_quiet(struct ncclComm *comm, const struct plan *p) {
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct schedule *x = p->plans + h;
    const struct channel *ch = comm->channels + h;
    if ((x->ins && !entry_done(comm, ch, ch->view.landed, pages_to_word(x->in[x->ins - 1].end))) ||
        (x->outs && !entry_done(comm, ch, ch->view.sent, pages_to_word(x->out[x->outs - 1].end))))
      return 0;
  }
  return 1;
}

static int slot_free(const struct ncclMeshRequest *r, const struct slot *s) {
  if (__atomic_load_n(&s->table->ended, __ATOMIC_ACQUIRE) < s->bound) return 0;
  for (uint32_t e = 0; e < r->peers; e++) {
    const struct channel *ch = r->comm->channels + r->part[e].channel;
    if (!entry_done(r->comm, ch, ch->view.sent, s->ends[2 * e]) || !entry_done(r->comm, ch, ch->view.landed, s->ends[2 * e + 1])) return 0;
  }
  return 1;
}

static struct slot *slot_of(struct ncclMeshRequest *r, uint64_t start) {
  if (!start || start > r->starts) return NULL;
  struct slot *s = r->starts - start < r->nslots ? r->slots + (r->turn + r->nslots - 1 - (uint32_t)(r->starts - start)) % r->nslots : NULL;
  for (uint32_t k = 0; (!s || s->start != start) && k < r->nslots; k++) s = r->slots + k;
  return s && s->start == start ? s : NULL;
}

static ncclResult_t slot_make(struct ncclMeshRequest *r, uint32_t at) {
  struct slot made = {.rows = {.comm = r->comm}};
  size_t offset = 0;
  ncclResult_t result = (made.ends = calloc(2 * (size_t)r->peers + 1, sizeof *made.ends)) ? lease_take(r->comm, r->bytes, &made.pages, &made.buffer, &offset)
                                                                                          : fail(ncclSystemError, "out of memory");
  struct slot *grown = result ? NULL : realloc(r->slots, (r->nslots + 1) * sizeof *grown);
  if (!result && !grown) result = fail(ncclSystemError, "out of memory");
  if (result) {
    if (made.pages) ncclMemFree(made.pages);
    free(made.ends);
    return result;
  }
  r->slots = grown;
  memmove(grown + at + 1, grown + at, (r->nslots - at) * sizeof *grown);
  struct slot *s = grown + at;
  *s = made;
  s->table = memset((char *)s->pages + r->table, 0, r->bytes - r->table);
  pthread_mutex_lock(&attach_lock);
  s->row = lease_at(s->pages)->row;
  pthread_mutex_unlock(&attach_lock);
  r->nslots++;
  return ncclSuccess;
}

static int request_sweep(struct ncclMeshRequest *r, int end) {
  uint32_t kept = 0;
  for (uint32_t k = 0; k < r->nslots; k++) {
    const int idle = slot_free(r, r->slots + k);
    if (r->slots[k].unbound && (idle || end)) { r->slots[k].unbound = 0; r->comm->unpublished--; }
    if (idle) ncclMemFree(r->slots[k].pages);
    if (idle || end) { free(r->slots[k].ends); rows_free(&r->slots[k].rows, idle); }
    else r->slots[kept++] = r->slots[k];
  }
  if ((r->nslots = kept)) return 0;
  free(r->slots); free(r->part); free(r);
  return 1;
}

static void sweep(struct ncclComm *comm, int all, int end) {
  if (!all && !end && !comm->sweeps) return;
  int left = 0;
  for (struct plan **at = &comm->plans; *at;) {
    struct plan *p = *at;
    if (end || ((p->stale || all) && plan_idle(p) && plan_quiet(comm, p))) { *at = p->next; plan_free(comm, p); }
    else { left |= p->stale; at = &p->next; }
  }
  for (struct ncclMeshRequest **at = &comm->freed; (all || end) && *at;) {
    struct ncclMeshRequest *r = *at, *next = r->next;
    if (request_sweep(r, end)) *at = next;
    else at = &r->next;
  }
  comm->sweeps = left;
}

ncclResult_t ncclMeshRequestFree(ncclMeshRequest_t r) {
  if (!r) return ncclSuccess;
  pthread_mutex_lock(&group_lock);
  if (!request_sweep(r, 0)) { r->next = r->comm->freed; r->comm->freed = r; }
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

ncclResult_t ncclMeshExchangeInit(size_t sendcount, const int *sendto, const size_t *recvcounts, ncclDataType_t datatype, ncclComm_t comm,
                                  ncclMeshRequest_t *request) {
  if (!comm || !sendto || !recvcounts || !request || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "ncclMeshExchangeInit's arguments");
  const size_t z = SIZE[datatype];
  struct ncclMeshRequest *r = calloc(1, sizeof *r);
  ncclResult_t result = r && (r->part = calloc((size_t)comm->nranks, sizeof *r->part)) ? ncclSuccess : fail(ncclSystemError, "out of memory");
  if (r) *r = (struct ncclMeshRequest){.comm = comm, .type = datatype, .count = sendcount, .bytes = page_up(sendcount * z, comm->page), .part = r->part};
  for (int q = 0; !result && q < comm->nranks; q++) {
    if (q == comm->rank || (!sendto[q] && !recvcounts[q])) continue;
    const struct channel *ch = comm->toward[q] < 0 ? NULL : comm->channels + comm->toward[q];
    if (!ch) { result = fail(ncclInvalidUsage, "an exchange of %s with rank %d, which no link of this node joins", comm->name, q); break; }
    r->part[r->peers++] = (struct peer_part){q, sendto[q] != 0, (uint32_t)comm->toward[q], recvcounts[q], r->bytes};
    r->bytes += page_up(recvcounts[q] * z, comm->page);
  }
  if (result) { ncclMeshRequestFree(r); return result; }
  r->table = r->bytes;
  r->bytes += sizeof(struct mesh_table) + r->peers * sizeof(struct mesh_peer);
  *request = r;
  return ncclSuccess;
}


ncclResult_t ncclMeshStart(ncclMeshRequest_t r, const size_t *counts, cudaStream_t stream, uint64_t *start) {
  const ncclMeshStream *on = stream;
  if (!r || !start || !on || (!on->commandBuffer && !(on->argumentTable && on->commandEncoder && on->residencySet)))
    return fail(ncclInvalidArgument, "ncclMeshStart without its request, its start, or a command buffer or a Metal 4 encoder, argument table and residency set");
  struct ncclComm *const comm = r->comm;
  const size_t z = SIZE[r->type], sent = (counts ? counts[comm->rank] : r->count) * z;
  for (uint32_t e = 0; counts && e <= r->peers; e++) {
    const int q = e < r->peers ? r->part[e].rank : comm->rank;
    const size_t most = e < r->peers ? r->part[e].count : r->count;
    if (counts[q] > most) return fail(ncclInvalidArgument, "ncclMeshStart: %zu elements of rank %d, past the %zu its exchange declared", counts[q], q, most);
  }
  if (revoked(comm)) return revoked_fail(comm);
  struct inflight *g = calloc(1, sizeof *g);
  if (!g || !(g->start = calloc(2 * (size_t)comm->count + 1, sizeof *g->start))) { free(g); return fail(ncclSystemError, "out of memory"); }
  *g = (struct inflight){.comm = comm, .start = g->start, .end = g->start + comm->count, .composed = 1};
  pthread_mutex_lock(&group_lock);
  for (uint32_t h = 0; h < comm->count; h++) {
    g->start[h][0] = g->end[h][0] = comm->channels[h].sends;
    g->start[h][1] = g->end[h][1] = comm->channels[h].lands;
  }
  for (const struct peer_part *p = r->part; p < r->part + r->peers; p++) {
    g->end[p->channel][0] += p->sends && sent;
    g->end[p->channel][1] += (counts ? counts[p->rank] : p->count) != 0;
  }
  uint64_t met = 0;
  for (struct inflight *x = inflight; x; x = x->next_group) met = meets(x, g) ? x->ticket : met;
  pthread_mutex_unlock(&group_lock);
  ncclResult_t result = met ? ncclMeshComplete(met, on, NULL) : ncclSuccess;
  pthread_mutex_lock(&group_lock);
  if (!result) result = comm_failure(comm);
  sweep(comm, 0, 0);
  uint32_t k = r->turn, looked = 0;
  while (!result && looked < r->nslots && ((r->starts && r->slots[k].start == r->starts) || !slot_free(r, r->slots + k))) { k = (k + 1) % r->nslots; looked++; }
  if (!result && looked == r->nslots) result = slot_make(r, k = r->turn);
  for (uint32_t e = 0; e < r->peers && !result; e++)
    for (int list = 0; list < 2 && !result; list++) {
      const uint32_t h = r->part[e].channel;
      if (g->end[h][list] > g->start[h][list] && !room_for(comm->channels + h, list, 1)) result = room_wait(comm, comm->channels + h, list, 1);
    }
  struct slot *s = result ? NULL : r->slots + k;
  struct mesh_table *t = s ? s->table : NULL;
  struct mesh_peer *peer = (struct mesh_peer *)(t + 1);
  if (s) { comm->unpublished -= s->unbound; s->start = 0; s->unbound = 0; s->rows.at = s->rows.used = 0; }
  if (!result) group_numbered(comm);
  for (uint32_t e = 0; e < r->peers && !result; e++) {
    const uint32_t h = r->part[e].channel;
    struct channel *ch = comm->channels + h;
    const size_t landing = (counts ? counts[ch->peer] : r->part[e].count) * z;
    const mesh_pages out = pages_of(sent, comm->page), in = pages_of(landing, comm->page);
    uint64_t *timed = comm->timed[((comm->groups - 1) % GROUP_TIMES) * comm->count + h];
    const uint64_t word = (uint64_t)((char *)&peer[e].published - (char *)comm->attach->context.M);
    int status = 0;
    peer[e] = (struct mesh_peer){.channel = h};
    s->ends[2 * e] = s->ends[2 * e + 1] = 0;
    if (g->end[h][0] > g->start[h][0]) {
      peer[e].ready_value = pages_to_word(out);
      if (!(status = mesh_stream_present(&comm->attach->context, ch->stream, 0, (struct mesh_entry){s->row, pages_to_word(out), word, 0}))) {
        ch->out = pages_add(ch->out, out);
        s->ends[2 * e] = pages_to_word(ch->out);
        timed[0] = ch->sends + 1;
        ch->published = ++ch->sends;
      }
    }
    if (!status && landing &&
        !(status = mesh_stream_present(&comm->attach->context, ch->stream, 1, (struct mesh_entry){s->row + r->part[e].landing / comm->page, pages_to_word(in), 0, 0}))) {
      ch->in = pages_add(ch->in, in);
      peer[e].landing = metal_address(s->buffer) + r->part[e].landing;
      peer[e].landed = metal_address(ch->buffer) + ch->landed;
      peer[e].landed_target = s->ends[2 * e + 1] = pages_to_word(ch->in);
      peer[e].landed_at = ch->lands;
      timed[1] = ++ch->lands;
    }
    if (status)
      result = comm_fail(comm, status == ENOLINK || status == EPERM ? ncclRemoteError : ncclSystemError, "communicator %s: an exchange's entry toward "
                         "node %u presented to its bridge: %s", comm->name, ch->node, strerror(status));
  }
  if (result) {
    free(g->start); free(g);
    pthread_mutex_unlock(&group_lock);
    return result;
  }
  *t = (struct mesh_table){metal_address(s->buffer), 0, r->peers, 0, 0};
  s->start = *start = ++r->starts;
  s->bound = 0;
  for (uint32_t e = 0; e < r->peers; e++) s->unbound |= peer[e].ready_value != 0;
  comm->unpublished += s->unbound;
  if (k == r->turn) r->turn = (k + 1) % r->nslots;
  g->ticket = ++tickets;
  if (inflight_last) inflight_last->next_group = g; else inflight = g;
  inflight_last = g;
  for (uint32_t h = 0; h < comm->count; h++)
    if (comm->channels[h].consumed == g->start[h][1]) comm->channels[h].consumed = consumed_past(comm, h, comm->channels[h].consumed);
  retire(NULL);
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

ncclResult_t ncclMeshBind(ncclMeshRequest_t r, uint64_t start, int lands, int publishes, cudaStream_t stream, uint32_t index, uint32_t groups) {
  const ncclMeshStream *on = stream;
  if (!r || !on || !on->commandEncoder || (on->argumentTable && !on->residencySet) || !groups)
    return fail(ncclInvalidArgument, "ncclMeshBind without its request, a compute encoder (and a Metal 4 one's residency set) or its threadgroups");
  pthread_mutex_lock(&group_lock);
  struct slot *land = lands ? slot_of(r, start - 1) : NULL, *publish = publishes ? slot_of(r, start) : NULL;
  if (publish && publish->unbound) { publish->unbound = 0; r->comm->unpublished--; }
  void *buffers[2 + MESH_STREAMS];
  size_t n = 0;
  if (land) buffers[n++] = land->buffer;
  if (publish) buffers[n++] = publish->buffer;
  for (uint32_t e = 0; e < r->peers && e < MESH_STREAMS; e++) buffers[n++] = r->comm->channels[r->part[e].channel].buffer;
  ncclResult_t result = (lands && !land) || (publishes && !publish)
                          ? fail(ncclInvalidUsage, "ncclMeshBind: start %llu, or the one before it, is not one of this exchange's held starts",
                                 (unsigned long long)start)
                          : gpu(metal_bind(on->commandEncoder, on->argumentTable, on->residencySet, publish ? &publish->rows : land ? &land->rows : NULL,
                                           land ? metal_address(land->buffer) + r->table : 0,
                                           publish ? metal_address(publish->buffer) + r->table : 0, r->comm->violation, groups, index, buffers, n),
                                "a crossing's binding");
  if (!result && land) land->bound++;
  if (!result && publish && publish != land) publish->bound++;
  pthread_mutex_unlock(&group_lock);
  return result;
}


static uint64_t time_of(const _Atomic uint64_t *ring, uint64_t e) {
  const _Atomic uint64_t *entry = ring + 2 * (e % MESH_STREAM_TIMES);
  const uint64_t tag = atomic_load_explicit(entry, memory_order_acquire), at = atomic_load_explicit(entry + 1, memory_order_acquire);
  return tag == 1 + e && atomic_load_explicit(entry, memory_order_relaxed) == 1 + e ? at : 0;
}

ncclResult_t ncclMeshPosition(ncclComm_t comm, uint64_t *group) {
  if (!comm || !group) return fail(ncclInvalidArgument, "comm or group is NULL");
  pthread_mutex_lock(&group_lock);
  *group = comm->groups;
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

ncclResult_t ncclMeshTimes(ncclComm_t comm, uint64_t first, uint64_t count, uint64_t *issued, uint64_t *ended) {
  if (!comm || (count && (!issued || !ended))) return fail(ncclInvalidArgument, "comm, issued or ended is NULL");

  static mach_timebase_info_data_t base;
  pthread_mutex_lock(&group_lock);
  if (!base.denom) mach_timebase_info(&base);
  for (uint64_t i = 0; i < count; i++) {
    const uint64_t g = first + i;
    uint64_t begun = UINT64_MAX, done = 0;
    int timed = g < comm->groups && comm->groups - g <= GROUP_TIMES, any = 0;
    for (uint32_t h = 0; timed && h < comm->count; h++) {
      const uint64_t *entries = comm->timed[(g % GROUP_TIMES) * comm->count + h];
      if (entries[0]) {
        const uint64_t posted = time_of(comm->channels[h].view.posted_at, entries[0] - 1);
        timed &= posted != 0;
        begun = posted < begun ? posted : begun;
      }
      if (entries[1]) {
        const uint64_t landed = time_of(comm->channels[h].view.landed_at, entries[1] - 1);
        timed &= landed != 0;
        done = landed > done ? landed : done;
      }
      any |= entries[0] || entries[1];
    }
    timed = timed && any;
    if (begun == UINT64_MAX) begun = done;
    issued[i] = timed ? (uint64_t)((unsigned __int128)begun * base.numer / base.denom) : 0;
    ended[i] = timed ? (uint64_t)((unsigned __int128)(done > begun ? done : begun) * base.numer / base.denom) : 0;
  }
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}



const char *ncclMeshMetalPrelude(void) { return metal_prelude(); }

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

static ncclResult_t launch_one(struct call *list, size_t n) {
  if (list[0].stream) return launch_metal(list, n);
  const ncclResult_t result = launch(list, n);
  if (result) list[0].comm->async = result;
  calls_free(list, n);
  return result;
}

struct part { struct call *list; size_t n; ncclResult_t result; struct said said; pthread_t thread; int threaded, metal; };
static void *part_run(void *argument) {
  struct part *p = argument;
  p->result = launch_one(p->list, p->n);
  p->said = last;
  return NULL;
}


ncclResult_t ncclGroupEnd(void) {
  if (depth <= 0) return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart");
  if (--depth) return ncclSuccess;
  struct call *list = calls;
  const size_t n = call_count;
  calls = NULL; call_count = call_capacity = 0;
  if (!n) { free(list); return ncclSuccess; }
  int several = 0;
  for (size_t i = 1; i < n; i++) several |= list[i].comm != list[0].comm;
  if (!several) return launch_one(list, n);
  struct part *part = calloc(n, sizeof *part);
  unsigned char *taken = calloc(n, 1);
  if (!part || !taken) { free(part); free(taken); calls_free(list, n); return fail(ncclSystemError, "out of memory"); }
  size_t parts = 0;
  for (size_t i = 0; i < n; i++) {
    if (taken[i]) continue;
    const ncclComm_t comm = list[i].comm;
    struct part *p = part + parts++;
    p->metal = list[i].stream != NULL;
    for (size_t j = i; j < n; j++) p->n += list[j].comm == comm;
    p->list = malloc(p->n * sizeof *p->list);
    size_t k = 0;
    for (size_t j = i; j < n; j++)
      if (list[j].comm == comm) {
        taken[j] = 1;
        if (p->list) p->list[k++] = list[j];
        else call_parts_free(list + j);
      }
    if (!p->list) { p->result = fail(ncclSystemError, "out of memory"); p->said = last; }
  }
  free(taken); free(list);
  ncclMeshIssue *issue = NULL, internal = {0, 0};
  ncclMeshStream on = {0};
  int metals = 0;
  for (size_t k = 0; k < parts; k++) {
    const struct call *c = part[k].list;
    if (!c || !part[k].metal) continue;
    if (!metals++) issue = c->issue;
    on = (ncclMeshStream){c->command, c->encoder, NULL, c->table, c->residency};
  }
  uint64_t ticket = 0;
  int published = 0;
  for (size_t k = 0; k < parts; k++) {
    struct part *p = part + k;
    if (!p->list || !p->metal) continue;
    ncclMeshIssue *const used = issue ? issue : &internal;
    for (size_t i = 0; i < p->n; i++) p->list[i].issue = used;
    p->result = launch_one(p->list, p->n);
    p->said = last;
    p->list = NULL;
    if (!p->result) { published |= used->published; ticket = used->ticket > ticket ? used->ticket : ticket; }
  }
  ncclResult_t completed = ncclSuccess;
  struct said completing = {0, ""};
  if (!issue && ticket) {
    int any = 0;
    if ((completed = ncclMeshComplete(ticket, &on, &any))) completing = last;
  }
  for (size_t k = 0; k < parts; k++) {
    struct part *p = part + k;
    if (!p->list || p->metal) continue;
    const int error = pthread_create(&p->thread, NULL, part_run, p);
    p->threaded = !error;
    if (error) {
      p->result = fail(ncclSystemError, "a thread for communicator %s's calls: %s", p->list[0].comm->name, strerror(error));
      p->said = last;
      calls_free(p->list, p->n);
    }
  }
  ncclResult_t result = ncclSuccess;
  struct said first = {0, ""};
  for (size_t k = 0; k < parts; k++) {
    if (part[k].threaded) pthread_join(part[k].thread, NULL);
    if (part[k].result && !result) { result = part[k].result; first = part[k].said; }
  }
  free(part);
  if (completed && !result) { result = completed; first = completing; }
  if (issue) *issue = (ncclMeshIssue){ticket, published};
  if (result) last = first;
  return result;
}

static ncclResult_t admit(struct call *made) {
  struct call c = *made;
  if (!c.comm) return fail(ncclInvalidArgument, "comm is NULL");
  if (c.stream) {
    const ncclMeshStream *stream = c.stream;
    c.command = stream->commandBuffer; c.encoder = stream->commandEncoder;
    c.issue = stream->issue; c.table = stream->argumentTable; c.residency = stream->residencySet;
    c.send_at = where_of(c.send); c.recv_at = where_of(c.recv);
  } else {
    c.send_at = (struct where){NULL, (size_t)c.send}; c.recv_at = (struct where){NULL, (size_t)c.recv};
  }
  *made = c;
  if ((unsigned)c.type >= ncclNumTypes) return fail(ncclInvalidArgument, "datatype %d", (int)c.type);
  const ncclComm_t comm = c.comm;
  if (reducing(c.what)) {
    const int k = (int)c.op - ncclNumOps;
    if ((int)c.op < 0 || (k >= 0 && (k >= (int)comm->nops || !comm->ops[k].used))) return fail(ncclInvalidArgument, "operator %d", (int)c.op);
    const uint32_t u = k >= 0 ? comm->ops[k].unit : 0;
    int rows = !u || !(c.count % u);
    for (int r = 0; u && c.segments && r < comm->nranks; r++) rows &= !(c.segments[r] % u);
    if (!rows) return fail(ncclInvalidArgument, "the partials merge's counts are whole rows of %u elements", u);
  }
  if ((c.what == MESH_REDUCE || c.what == MESH_BROADCAST) && (c.root < 0 || c.root >= comm->nranks))
    return fail(ncclInvalidArgument, "root %d of %d ranks", c.root, comm->nranks);
  if ((c.what == WHAT_SEND || c.what == WHAT_RECV) && (c.peer < 0 || c.peer >= comm->nranks))
    return fail(ncclInvalidArgument, "peer %d of %d ranks", c.peer, comm->nranks);
  return ncclSuccess;
}

static ncclResult_t enqueue(struct call c) {
  const ncclResult_t refused = admit(&c);
  if (refused) { free(c.segments); return refused; }
  ncclGroupStart();
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) { depth--; free(c.segments); return fail(ncclSystemError, "out of memory"); }
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

static ncclResult_t enqueue_counts(struct call c, const size_t *counts) {
  if (!c.comm || !counts) return fail(ncclInvalidArgument, "%s is NULL", c.comm ? "counts" : "comm");
  const int n = c.comm->nranks;
  int equal = 1;
  for (int r = 0; r < n; r++) equal &= counts[r] == counts[0];
  c.count = counts[c.comm->rank];
  if (equal) return enqueue(c);
  if (!(c.segments = malloc((size_t)n * sizeof *c.segments))) return fail(ncclSystemError, "out of memory");
  for (int r = 0; r < n; r++) { c.segments[r] = counts[r]; c.total += counts[r]; }
  return enqueue(c);
}
ncclResult_t ncclMeshAllGatherV(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream) {
  return enqueue_counts((struct call){.what = MESH_ALLGATHER, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm,
                                      .stream = stream}, counts);
}
ncclResult_t ncclMeshReduceScatterV(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm, cudaStream_t stream) {
  return enqueue_counts((struct call){.what = MESH_REDUCE_SCATTER, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = op, .comm = comm,
                                      .stream = stream}, counts);
}

ncclResult_t ncclMeshAlltoAllCounted(const void *sendbuff, size_t sendrows, const int64_t *counts, const size_t *sendsegs, void *recvbuff,
                                     size_t capacity, int64_t *received, int64_t *landed, const size_t *recvsegs, uint64_t *arrived, size_t row,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  (void)sendbuff; (void)sendrows; (void)counts; (void)sendsegs; (void)recvbuff; (void)capacity; (void)received; (void)landed; (void)recvsegs;
  (void)row; (void)datatype; (void)stream;
  if (arrived) __atomic_store_n(arrived, 2, __ATOMIC_RELEASE);
  return fail(ncclInvalidUsage, "ncclMeshAlltoAllCounted%s%s: its rows' places are computed on the GPU from its counts, so none of them can be an "
              "entry the host presents (a page's first byte and its bytes): they would cross only through a copy, which this library never "
              "makes; ncclMeshAlltoAllV moves rows whose counts the host holds, each pair's segment from a page", comm ? " on communicator " : "",
              comm ? comm->name : "");
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_SEND, .peer = peer, .send = sendbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_RECV, .peer = peer, .recv = recvbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}

static ncclResult_t routed(const void *sendbuff, void *recvbuff, const size_t *sendcounts, const size_t *recvcounts, size_t count, int from, int to,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int n = comm->nranks, me = comm->rank, root = from < 0 ? to : from;
  if (root < -1 || root >= n) return fail(ncclInvalidArgument, "root %d of %d ranks", root, n);
  struct call c = {.what = WHAT_COUNTED, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm, .stream = stream,
                   .every = !sendcounts, .root = from, .peer = to, .total = count};
  if (!(c.segments = malloc(2 * (size_t)n * sizeof *c.segments))) return fail(ncclSystemError, "out of memory");
  for (int q = 0; q < n; q++) {
    c.segments[q] = sendcounts ? sendcounts[q] : (from < 0 || from == me) && (to < 0 || to == q) ? count : 0;
    c.segments[n + q] = recvcounts ? recvcounts[q] : (from < 0 || from == q) && (to < 0 || to == me) ? count : 0;
    if (q != me) c.count += c.segments[q] + c.segments[n + q];
  }
  if (c.every) c.count = count;
  return enqueue(c);
}
ncclResult_t ncclMeshAlltoAllV(const void *sendbuff, const size_t *sendcounts, void *recvbuff, const size_t *recvcounts, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
  if (!sendcounts || !recvcounts) return fail(ncclInvalidArgument, "ncclMeshAlltoAllV: sendcounts or recvcounts is NULL");
  return routed(sendbuff, recvbuff, sendcounts, recvcounts, 0, -1, -1, datatype, comm, stream);
}
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, NULL, count, -1, -1, datatype, comm, stream);
}
ncclResult_t ncclAlltoAllConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}
ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, NULL, count, -1, root == -1 ? -2 : root, datatype, comm, stream);
}
ncclResult_t ncclGatherConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                              cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
}
ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, NULL, count, root == -1 ? -2 : root, -1, datatype, comm, stream);
}
ncclResult_t ncclScatterConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                               cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

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
  const int making = (*made = comm_new(count, rank, name)) ? comm_attach(*made, members) : ENOMEM;
  ncclResult_t result = making ? comm_refused(making, name) : ncclSuccess;
  if (making && *made) { comm_free(*made); *made = NULL; }
  if (!result && comm->form_count && !((*made)->forms = forms_copy(comm->forms, comm->form_count, (uint32_t)comm->nranks))) {
    comm_free(*made); *made = NULL;
    result = fail(ncclSystemError, "out of memory");
  } else if (!result) (*made)->form_count = comm->form_count;
  free(members);
  return result;
}

static ncclResult_t split(ncclComm_t comm, int color, int key, ncclComm_t *newcomm) {
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; struct report report; } mine = {color, key, {.rank = comm->rank}}, *every = calloc((size_t)n, sizeof *every);
  struct binding { int32_t status, padding; uint64_t posted[MESH_STREAMS][2]; } bound = {0, 0, {{0}}}, *binds = calloc((size_t)n, sizeof *binds);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  struct report *reports = calloc((size_t)n, sizeof *reports);

  const uint64_t page = comm->page ? comm->page : 1;
  const size_t stride = page_up(sizeof mine > sizeof bound ? sizeof mine : sizeof bound, page);
  unsigned char *rows = NULL;
  *newcomm = NULL;
  if (color != NCCL_SPLIT_NOCOLOR) reach_read(comm->attach, &mine.report);
  ncclResult_t result = !every || !binds || !ranks || !reports ? fail(ncclSystemError, "out of memory") : ncclMemAlloc((void **)&rows, (size_t)n * stride);
  if (!result) {
    memcpy(rows + (size_t)comm->rank * stride, &mine, sizeof mine);
    result = ncclAllGather(rows + (size_t)comm->rank * stride, rows, stride, ncclUint8, comm, NULL);
  }
  for (int r = 0; !result && r < n; r++) memcpy(every + r, rows + (size_t)r * stride, sizeof *every);
  const int gathered = !result;
  for (int r = 0; !result && color != NCCL_SPLIT_NOCOLOR && r < n; r++)
    if (every[r].color == color) ranks[count++] = r;
  for (int a = 1; a < count; a++)
    for (int b = a; b > 0 && every[ranks[b]].key < every[ranks[b - 1]].key; b--) { const int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
  if (!result && count) result = comm_sub(comm, ranks, count, split, color, newcomm);
  for (int i = 0; !result && i < count; i++) reports[i] = every[ranks[i]].report;
  if (!result && count > 1 && comm_apart(reports, (uint32_t)count)) result = apart_fail((*newcomm)->name);
  if (!result && count > 1) {
    struct report *own = reports + (*newcomm)->rank;
    bound.status = comm_hold(*newcomm, reports, own);
    memcpy(bound.posted, own->posted, sizeof bound.posted);
  }
  else if (result && count) bound.status = ECANCELED;
  if (gathered) {
    memcpy(rows + (size_t)comm->rank * stride, &bound, sizeof bound);
    const ncclResult_t confirmed = ncclAllGather(rows + (size_t)comm->rank * stride, rows, stride, ncclUint8, comm, NULL);
    for (int r = 0; !confirmed && r < n; r++) memcpy(binds + r, rows + (size_t)r * stride, sizeof *binds);
    if (!result) result = confirmed;
  }
  for (int i = 0; !result && i < count; i++)
    if (binds[ranks[i]].status) result = hold_fail((*newcomm)->name, i, binds[ranks[i]].status);
  for (int i = 0; !result && i < count; i++) memcpy(reports[i].posted, binds[ranks[i]].posted, sizeof reports[i].posted);
  int status = 0;
  if (!result && count > 1 && (status = comm_join(*newcomm, reports)))
    result = fail(ncclSystemError, "communicator %s: its channels on the streams it held: %s", (*newcomm)->name, strerror(status));
  if (result && *newcomm) { comm_free(*newcomm); *newcomm = NULL; }
  if (rows) ncclMemFree(rows);
  free(every); free(binds); free(ranks); free(reports);
  return result;
}

static void aside(int to) {
  static _Thread_local struct { int depth; struct call *calls; size_t count, capacity; } kept;
  if (to) {
    kept.depth = depth; kept.calls = calls; kept.count = call_count; kept.capacity = call_capacity;
    depth = 0; calls = NULL; call_count = call_capacity = 0;
    return;
  }
  free(calls);
  depth = kept.depth; calls = kept.calls; call_count = kept.count; call_capacity = kept.capacity;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  aside(1);
  const ncclResult_t result = split(comm, color, key, newcomm);
  aside(0);
  return result;
}

ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t *config,
                            int shrinkFlags) {
  (void)config;
  if (!comm || !newcomm || excludeRanksCount < 0 || (excludeRanksCount && !excludeRanksList))
    return fail(ncclInvalidArgument, "ncclCommShrink's arguments");
  *newcomm = NULL;
  int *ranks = calloc((size_t)comm->nranks, sizeof *ranks), count = 0, kept = 0;
  if (!ranks) return fail(ncclSystemError, "out of memory");
  for (int r = 0; r < comm->nranks; r++) {
    int excluded = 0;
    for (int e = 0; e < excludeRanksCount; e++) excluded |= excludeRanksList[e] == r;
    if (!excluded) ranks[count++] = r;
    kept |= !excluded && r == comm->rank;
  }
  if (shrinkFlags & NCCL_SHRINK_ABORT) ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
  const ncclResult_t result = kept ? comm_sub(comm, ranks, count, comm->splits++, -1, newcomm) : ncclSuccess;
  free(ranks);
  return result;
}

ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
  return comm ? ncclGetUniqueId(uniqueId) : fail(ncclInvalidArgument, "comm is NULL");
}

ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!newcomm || (comm ? nRanks < comm->nranks || (!comm->rank && !uniqueId) : !uniqueId))
    return fail(ncclInvalidArgument, "ncclCommGrow's arguments");
  if (!comm) return ncclCommInitRank(newcomm, nRanks, *uniqueId, rank);
  ncclUniqueId *id = NULL, given;
  aside(1);
  ncclResult_t result = ncclMemAlloc((void **)&id, sizeof *id);
  if (!result) {
    *id = comm->rank ? (ncclUniqueId){{0}} : *uniqueId;
    result = ncclBroadcast(id, id, sizeof *id, ncclUint8, 0, comm, NULL);
    given = *id;
    ncclMemFree(id);
  }
  aside(0);
  return result ? result : ncclCommInitRank(newcomm, nRanks, given, comm->rank);
}

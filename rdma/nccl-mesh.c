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
    S *d = (S *)dst; const S *w = (const S *)with, *x = (const S *)src; \
    for (size_t i = 0; i < n; i++) switch (combine) { \
      case COMBINE_SUM: d[i] = (S)((U)w[i] + (U)x[i]); break; \
      case COMBINE_PROD: d[i] = (S)((U)w[i] * (U)x[i]); break; \
      case COMBINE_MAX: d[i] = x[i] > w[i] ? x[i] : w[i]; break; \
      default: d[i] = x[i] < w[i] ? x[i] : w[i]; break; } } while (0)

static const float PARTIAL_NONE = -3.4028234663852886e38f;
static void merge_rows(float *x, const float *z, const float *y, size_t rows, size_t unit) {
  for (size_t r = 0; r < rows; r++, x += unit, z += unit, y += unit) {
    const int mine = z[1] > 0, theirs = y[1] > 0;
    if (!theirs || !mine) {
      for (size_t j = 0; j < unit; j++) x[j] = mine ? z[j] : theirs ? y[j] : j ? 0.0f : PARTIAL_NONE;
      continue;
    }
    const float top = z[0] > y[0] ? z[0] : y[0], a = expf(z[0] - top), b = expf(y[0] - top);
    x[0] = top;
    for (size_t j = 1; j < unit; j++) x[j] = (a > 0 ? z[j] * a : 0.0f) + (b > 0 ? y[j] * b : 0.0f);
  }
}

static void combine_from(ncclDataType_t t, int combine, void *dst, const void *with, const void *src, size_t n) {
  if ((int)t == METAL_MERGE) { merge_rows(dst, with, src, n, (size_t)combine); return; }
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
  const unsigned char *w = with, *x = src;
  for (size_t i = 0; i < n; i++, d += z, w += z, x += z) {
    const double a = load(t, w), b = load(t, x);
    if (combine == COMBINE_MAX || combine == COMBINE_MIN) memmove(d, (combine == COMBINE_MAX ? b > a : b < a) ? x : w, z);
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

struct channel {
  int peer;
  uint32_t node;
  uint64_t sends, lands, published, consumed;
};

struct attach;
struct plan;

struct ncclComm {
  struct mesh_link_map map;
  struct mesh_uses uses;
  uint64_t page, id;
  uint64_t exchanges;
  uint32_t splits, *members, count, nops;
  int nranks, rank, touched, unpublished;
  char name[NCCL_UNIQUE_ID_BYTES + 1];
  struct redop *ops;
  ncclResult_t async;
  int revoked, initing, *toward;
  pthread_t init;
  struct said why;
  struct attach *attach;
  struct mesh_plan *named;
  struct channel *channels;
  struct plan *plans;
  struct ncclMeshRequest *freed;
  uint64_t last;
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
  void *pages, *buffer, *stage, *stage_buffer, *command;
  struct where base;
  size_t word, stage_bytes, stage_offset;
  uint64_t stage_row;
  uint32_t reader;
  int flight, leaving;
  struct plan *next;
};

static pthread_mutex_t group_lock = PTHREAD_MUTEX_INITIALIZER;
static void *finished;
static uint64_t issued;

static void program_free(struct mesh_program *program) {
  free((void *)program->segment); free((void *)program->log_weight); free((void *)program->first); free((void *)program->move);
}

static void uses_free(struct mesh_uses *uses) {
  free((void *)uses->capacity); free((void *)uses->first); free((void *)uses->use); free((void *)uses->weight);
  *uses = (struct mesh_uses){0};
}

static void map_free(struct mesh_link_map *map) {
  free(map->edge); free(map->next); free(map->mark);
  *map = (struct mesh_link_map){0};
}


static void comm_leave(struct ncclComm *comm);
static void sweep(struct ncclComm *comm);
static void release(struct ncclComm *comm);

static void comm_free(struct ncclComm *comm) {
  map_free(&comm->map);
  uses_free(&comm->uses);
  metal_release(comm->command);
  comm_leave(comm);
  free(comm->channels); free(comm->toward);
  free(comm->ops);
  free(comm->members);
  free(comm);
}

static int has_root(int what) { return what == MESH_REDUCE || what == MESH_BROADCAST; }

#define MESH_SOLVE_EPS 0.1

struct flow { uint32_t count, *demand, *parent; double *share; };
static void flow_free(struct flow *f) {
  free(f->demand); free(f->parent); free(f->share);
  *f = (struct flow){0, NULL, NULL, NULL};
}
static ncclResult_t flow_solve(ncclComm_t comm, const struct mesh_demand *demands, uint32_t count, struct flow *f) {
  const uint32_t n = (uint32_t)comm->nranks, room = count ? mesh_solve_room(&comm->map, &comm->uses, MESH_SOLVE_EPS, count) : 0;
  *f = (struct flow){0, calloc((size_t)room + 1, sizeof *f->demand), calloc((size_t)room * n + 1, sizeof *f->parent),
                     calloc((size_t)room + 1, sizeof *f->share)};
  if (!f->demand || !f->parent || !f->share) { flow_free(f); return fail(ncclSystemError, "out of memory"); }
  const uint32_t made = count ? mesh_solve(&comm->map, &comm->uses, MESH_SOLVE_EPS, 0, demands, count, room, f->demand, f->parent, f->share) : 0;
  if (made == UINT32_MAX) {
    flow_free(f);
    return fail(ncclInvalidUsage, "communicator %s: its map joins no route between two members its calls cross between", comm->name);
  }
  f->count = made;
  return ncclSuccess;
}

struct lease { void *address; uint32_t row; size_t bytes; void *buffer; };

struct retired { void *pages; struct mesh_walk walk; uint32_t tables, settled; struct retired *next; struct mesh_plan *table[]; };
struct attach {
  struct mesh_ctx context;
  struct lease *leases;
  size_t count, room;
  uint64_t comms, calls;
  _Atomic int wrapped;
  struct retired *retired;
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

static void retire(struct attach *a, void *pages, struct mesh_plan *const *table, uint32_t tables) {
  for (uint32_t i = 0; i < tables; i++)
    if (table[i]) mesh_queue_unlink(&a->context, table[i]);
  struct retired *r = malloc(sizeof *r + (tables + 1) * sizeof *r->table);
  if (!r) return;
  *r = (struct retired){pages, {{{0}}}, tables, 0, a->retired};
  memcpy(r->table, table, tables * sizeof *table);
  mesh_walks(a->context.M, &r->walk);
  a->retired = r;
}

static void retire_locked(struct ncclComm *comm, void *pages, struct mesh_plan *const *table, uint32_t tables) {
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) retire(comm->attach, pages, table, tables);
  pthread_mutex_unlock(&attach_lock);
}

static void retired_return(void) {
  struct retired *done = NULL;
  pthread_mutex_lock(&attach_lock);
  for (struct attach *a = attaches; a; a = a->next)
    for (struct retired **at = &a->retired; *at;) {
      struct retired *r = *at;
      const int expired = attach_expired(a);
      int ready = expired || mesh_walked(a->context.M, &r->walk);
      if (ready && !expired && !r->settled) {
        for (uint32_t i = 0; ready && i < r->tables; i++) ready = !r->table[i] || mesh_plan_balanced(r->table[i]);
        if (ready) { r->settled = 1; mesh_walks(a->context.M, &r->walk); }
        ready = 0;
      }
      if (ready) { *at = r->next; r->next = done; done = r; }
      else at = &r->next;
    }
  pthread_mutex_unlock(&attach_lock);
  for (struct retired *r = done, *next; r; r = next) { next = r->next; if (r->pages) ncclMemFree(r->pages); free(r); }
}

static void comm_leave(struct ncclComm *comm) {
  sweep(comm);
  if (comm->named) { retire_locked(comm, comm->named, &comm->named, 1); comm->named = NULL; }
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) {
    comm->attach->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
}

static int hop_joins(const struct mesh_edge *e, uint32_t u, uint32_t v) { return (e->a == u && e->b == v) || (e->a == v && e->b == u); }

static int map_of(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  uint32_t room = 64, edges = 64, *node = NULL, *next = NULL, *mark = NULL, *place = NULL, *path = NULL;
  struct mesh_edge *edge = NULL;
  struct mesh_link_map whole = {0};
  int count = -ENOSPC;
  while (count == -ENOSPC) {
    free(node); free(edge); free(next); free(mark);
    node = malloc((room + 1) * sizeof *node); edge = malloc((edges + 1) * sizeof *edge);
    next = malloc(((size_t)room * room + 1) * sizeof *next); mark = malloc((room + 1) * sizeof *mark);
    if (!node || !edge || !next || !mark) { count = -ENOMEM; break; }
    whole = (struct mesh_link_map){.nodes = room, .edges = edges, .edge = edge, .next = next, .mark = mark};
    count = mesh_map(NULL, 0, node, &whole);
    room = whole.nodes; edges = whole.edges;
  }
  struct mesh_link_map *map = &comm->map;
  const size_t pairs = (size_t)n * (n - 1) / 2 + 1, cables = count > 0 ? whole.edges : 0;
  *map = (struct mesh_link_map){n, 0, calloc(pairs, sizeof *map->edge), 0, NULL, calloc(n + 1, sizeof *map->mark)};
  uint64_t *capacity = calloc(cables + 1, sizeof *capacity);
  uint32_t *first = calloc(pairs + 1, sizeof *first), *use = calloc(pairs * (cables + 1) + 1, sizeof *use);
  double *weight = calloc(pairs * (cables + 1) + 1, sizeof *weight);
  place = malloc((n + 1) * sizeof *place); path = malloc(((size_t)(count > 0 ? count : 0) + 1) * sizeof *path);
  int status = count < 0 ? -count : !map->edge || !map->mark || !place || !path || !capacity || !first || !use || !weight ? ENOMEM : 0;
  uint32_t uses = 0;
  for (size_t c = 0; !status && c < cables; c++) capacity[c] = whole.edge[c].capacity;
  for (uint32_t a = 0; !status && a < n; a++) {
    map->mark[a] = comm->members[a];
    place[a] = UINT32_MAX;
    for (int i = 0; i < count; i++) if (node[i] == comm->members[a]) place[a] = (uint32_t)i;
  }
  for (uint32_t a = 0; !status && a < n; a++)
    for (uint32_t b = a + 1; b < n; b++) {
      const uint32_t hops = place[a] == UINT32_MAX || place[b] == UINT32_MAX || place[a] == place[b] ? 0
                            : mesh_route_path(&whole, place[a], place[b], path);
      uint64_t bottleneck = 0;
      const uint32_t at = uses;
      for (uint32_t h = 1; h < hops; h++) {
        uint64_t sum = 0;
        for (uint32_t c = 0; c < whole.edges; c++) sum += hop_joins(whole.edge + c, path[h - 1], path[h]) ? whole.edge[c].capacity : 0;
        if (h == 1 || sum < bottleneck) bottleneck = sum;
        for (uint32_t c = 0; c < whole.edges; c++)
          if (whole.edge[c].capacity && hop_joins(whole.edge + c, path[h - 1], path[h])) {
            use[uses] = c << 1 | (whole.edge[c].a != path[h - 1]); weight[uses++] = (double)whole.edge[c].capacity / (double)sum;
          }
      }
      if (hops < 2 || !bottleneck) { uses = at; continue; }
      first[map->edges] = at;
      map->edge[map->edges++] = (struct mesh_edge){a, b, 0, bottleneck};
    }
  if (!status) first[map->edges] = uses;
  free(node); free(edge); free(next); free(mark); free(place); free(path);
  if (status) { map_free(map); free(capacity); free(first); free(use); free(weight); }
  else comm->uses = (struct mesh_uses){(uint32_t)cables, 0, capacity, first, use, weight};
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

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  arc4random_buf(id->internal, sizeof id->internal);
  return ncclSuccess;
}

static uint64_t fnv(uint64_t h, const void *bytes, size_t n) {
  for (const unsigned char *b = bytes; n--; b++) { h ^= *b; h *= UINT64_C(1099511628211); }
  return h;
}
static uint64_t named(uint64_t h) { return h ? h : 1; }

static int revoked(ncclComm_t comm) { return __atomic_load_n(&comm->revoked, __ATOMIC_SEQ_CST); }
static double now_s(void);

static ncclResult_t revoked_fail(ncclComm_t comm) {
  return fail(ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
}

static ncclResult_t comm_register(struct ncclComm *comm) {
  void *pages = NULL;
  const ncclResult_t result = ncclMemAlloc(&pages, mesh_plan_bytes(0, 0));
  if (result) return result;
  struct mesh_plan *t = memset(pages, 0, mesh_plan_bytes(0, 0));
  t->name[0] = comm->id; t->name[1] = (uint64_t)comm->rank;
  pthread_mutex_lock(&attach_lock);
  mesh_queue_link(&comm->attach->context, t);
  pthread_mutex_unlock(&attach_lock);
  comm->named = t;
  return ncclSuccess;
}

static ncclResult_t comm_members(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  uint64_t (*name)[2] = malloc((n + 1) * sizeof *name);
  if (!name) return fail(ncclSystemError, "out of memory");
  for (uint32_t r = 0; r < n; r++) { name[r][0] = comm->id; name[r][1] = r; }
  ncclResult_t result = ncclSuccess;
  double since = now_s();
  for (uint32_t found = 0, most = 0; (found = mesh_names_find(comm->attach->context.M, (const uint64_t (*)[2])name, n, comm->members)) < n;) {
    if (found > most) { most = found; since = now_s(); }
    if (revoked(comm)) { result = revoked_fail(comm); break; }
    if (now_s() - since >= (double)MESH_PLATFORM_NS / 1e9) {
      uint32_t r = 0;
      while (r < n && comm->members[r]) r++;
      result = fail(ncclRemoteError, "communicator %s: no bridge's record holds rank %u's registration, %.0f s after the last rank found "
                    "(%u of %u found; design/transport.md section 14, \"A registration\")", comm->name, r, (double)MESH_PLATFORM_NS / 1e9, found, n);
      break;
    }
    usleep(1000);
  }
  free(name);
  return result;
}

static ncclResult_t comm_channels(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  comm->page = (uint64_t)comm->attach->context.M->block * comm->attach->context.M->pgsz;
  if (!(comm->channels = calloc((size_t)n + 1, sizeof *comm->channels))) return fail(ncclSystemError, "out of memory");
  for (uint32_t q = 0; q < n; q++) {
    if (q == me || comm->members[q] == comm->members[me]) continue;
    comm->toward[q] = (int)comm->count;
    comm->channels[comm->count++] = (struct channel){.peer = (int)q, .node = comm->members[q]};
  }
  const int status = map_of(comm);
  return status ? fail(ncclSystemError, "communicator %s: its map: %s", comm->name, strerror(status)) : ncclSuccess;
}

static ncclResult_t comm_init(struct ncclComm *made) {
  const int making = comm_attach(made, NULL);
  ncclResult_t result = making ? comm_refused(making, made->name) : ncclSuccess;
  if (!result && made->nranks > 1 && !(result = comm_register(made))) result = comm_members(made);
  if (!result && made->nranks == 1) made->members[0] = made->attach->context.M->node;
  if (!result) result = comm_channels(made);
  if (!result && revoked(made)) result = revoked_fail(made);
  if (result) comm_leave(made);
  return result;
}

static void *init_run(void *argument) {
  struct ncclComm *comm = argument;
  const ncclResult_t result = comm_init(comm);
  if (result) { comm->why = last; comm->why.id = keep(); }
  __atomic_store_n(&comm->async, result, __ATOMIC_RELEASE);
  return NULL;
}

ncclResult_t ncclMeshCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, const char *name, ncclConfig_t *config) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return fail(ncclInvalidArgument, "the init's arguments: rank %d of %d", rank, nranks);
  const uint64_t id = named(fnv(UINT64_C(14695981039346656037), commId.internal, sizeof commId.internal));
  char hex[17];
  snprintf(hex, sizeof hex, "%016llx", (unsigned long long)id);
  struct ncclComm *made = comm_new(nranks, rank, name && *name ? name : hex);
  if (!made) return fail(ncclSystemError, "out of memory");
  made->id = id;
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
  return ncclMeshCommInitRank(comm, nranks, commId, rank, NULL, NULL);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config) {
  return ncclMeshCommInitRank(comm, nranks, commId, rank, NULL, config);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  (void)devlist;
  if (ndev != 1) return fail(ncclInvalidArgument, "a node has one Metal device: ndev %d", ndev);
  ncclUniqueId id;
  const ncclResult_t result = ncclGetUniqueId(&id);
  return result ? result : ncclCommInitRank(comm, 1, id, 0);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL"); }

static void inflight_drop(struct ncclComm *comm);

static void comm_stop(struct ncclComm *comm) { __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST); }

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
  inflight_drop(comm);
  pthread_mutex_unlock(&group_lock);
  comm_free(comm);
  retired_return();
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  const ncclResult_t async = __atomic_load_n(&comm->async, __ATOMIC_ACQUIRE);
  if (async != ncclInProgress && comm->why.id) told(async, &comm->why);
  *asyncError = async;
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

struct after { void *ptr; const _Atomic uint32_t *word; struct after *next; };
static struct after *afters;

static void afters_return(void) {
  pthread_mutex_lock(&attach_lock);
  struct after *done = NULL;
  for (struct after **at = &afters; *at;) {
    struct after *a = *at;
    if (atomic_load_explicit(a->word, memory_order_acquire)) { *at = a->next; a->next = done; done = a; }
    else at = &a->next;
  }
  pthread_mutex_unlock(&attach_lock);
  for (struct after *a = done, *next; a; a = next) { next = a->next; ncclMemFree(a->ptr); free(a); }
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  if (!ptr) return fail(ncclInvalidArgument, "ptr is NULL");
  afters_return();
  retired_return();
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

static int lease_buffer(const void *ptr, void **buffer, size_t *offset);
static ncclResult_t gpu(struct metal_result step, const char *what);

ncclResult_t ncclMeshMemFreeAfter(void *ptr, size_t word, cudaStream_t stream) {
  const ncclMeshStream *on = stream;
  void *buffer = NULL;
  size_t offset = 0;
  if (!ptr || !on || !on->commandBuffer || !on->commandEncoder || word % sizeof(uint32_t))
    return fail(ncclInvalidArgument, "ncclMeshMemFreeAfter without its lease, a 4-byte aligned word, or a command buffer and its encoder");
  if (lease_buffer(ptr, &buffer, &offset)) return fail(ncclInvalidArgument, "ncclMeshMemFreeAfter of %p: no lease of this process", ptr);
  afters_return();
  struct after *made = malloc(sizeof *made);
  const char *message = NULL;
  struct metal_program *program = made ? metal_begin(on->commandBuffer, on->commandEncoder, &message) : NULL;
  if (!program) { free(made); return fail(ncclSystemError, "ncclMeshMemFreeAfter: %s", made ? message : "out of memory"); }
  const ncclResult_t result = gpu(metal_stamp(program, buffer, offset + word), "a lease's stamp");
  metal_end(program, NULL, 0);
  if (result) { free(made); return result; }
  pthread_mutex_lock(&attach_lock);
  *made = (struct after){ptr, (const _Atomic uint32_t *)((const char *)ptr + word), afters};
  afters = made;
  pthread_mutex_unlock(&attach_lock);
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
  struct mesh_program program;
  int compiled;
  uint32_t *pair_call;
};

struct call {
  int what, root, peer;
  const void *send;
  void *recv;
  size_t count, total;
  uint64_t *segments, *matrix;
  struct pair { uint32_t from, to; uint64_t bytes; } *pairs;
  uint32_t npairs;

  uint64_t *complement;
  uint32_t complements;
  ncclDataType_t type;
  ncclRedOp_t op;
  ncclComm_t comm;
  int combine, post;
  uint32_t unit;
  unsigned char pre[8];

  const void *scalar_at;
  int premultiplied, direct, direct_ok;
  struct steps *bound;
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
  return c->what == WHAT_SEND ? "ncclSend" : c->what == WHAT_RECV ? "ncclRecv" : !counted(c) ? names[c->what] : c->pairs ? "ncclMeshGroupPairs"
         : c->matrix ? "ncclMeshAlltoAllV"
         : c->root >= 0 ? "ncclScatter" : c->peer >= 0 ? "ncclGather" : "ncclAlltoAll";
}

static int crosses(const struct call *c) { return elements(c) && c->comm->nranks > 1 && !(p2p(c) && c->peer == c->comm->rank); }

static int in_region(const struct ncclComm *comm, struct where w, size_t bytes) {
  const struct hdr *m = comm->attach->context.M;
  const unsigned char *at = address_of(w), *first = (const unsigned char *)m + m->data_off, *end = (const unsigned char *)m + m->length;
  return at && at >= first && at <= end && bytes <= (size_t)(end - at);
}

static void steps_free(struct steps *s) {
  if (!s) return;
  free(s->buffer); free(s->steps); free(s->done); free(s->channel); free(s->entry); free(s->before); free(s->reads); free(s->fresh);
  free(s->needed); free(s->cut); free(s->at);
  if (s->compiled) program_free(&s->program);
  free(s->pair_call);
  free(s);
}

static size_t piece_bytes(const struct mesh_step *step) { return step->piece.elements * step->piece.element_bytes; }

static uint64_t hosted_count(const struct call *c, int a, int b) {
  if (c->matrix) return c->matrix[(size_t)a * c->comm->nranks + b];
  return (c->root < 0 || c->root == a) && (c->peer < 0 || c->peer == b) ? c->total : 0;
}

static ncclResult_t hosted_steps(ncclComm_t comm, const struct call *c, const struct mesh_routing *g, struct mesh_operand operand,
                                 struct mesh_step **steps, uint32_t *count, uint64_t *relayed) {
  const int n = comm->nranks, me = comm->rank;
  struct mesh_block *blocks = calloc((size_t)n * n + c->npairs + 1, sizeof *blocks);
  uint32_t made = 0;
  if (!blocks) return fail(ncclSystemError, "out of memory");
  for (uint32_t i = 0; c->pairs && i < c->npairs; i++) {
    const struct pair *x = c->pairs + i;
    if (x->from != x->to && x->bytes && x->from < (uint32_t)n && x->to < (uint32_t)n)
      blocks[made++] = (struct mesh_block){x->from, x->to, 3 + i, 3 + i, operand.type, 1, 0, 0, x->bytes};
  }
  for (int a = 0; !c->pairs && a < n; a++)
    for (int b = 0; b < n; b++) {
      uint64_t first = 0, at = 0;
      for (int q = 0; q < b; q++) first += hosted_count(c, a, q);
      for (int q = 0; q < a; q++) at += hosted_count(c, q, b);
      if (a != b && hosted_count(c, a, b))
        blocks[made++] = (struct mesh_block){(uint32_t)a, (uint32_t)b, 0, 1, operand.type, operand.element_bytes, first, at, hosted_count(c, a, b)};
    }
  const uint32_t room = mesh_exchange_room(g, blocks, made, NULL);
  ncclResult_t result = ncclSuccess;
  if (!(*steps = calloc((size_t)room + 1, sizeof **steps))) result = fail(ncclSystemError, "out of memory");
  else if ((*count = mesh_exchange_steps(g, (uint32_t)me, blocks, made, 0, 2, comm->page, *steps, NULL, relayed)) == UINT32_MAX)
    result = fail(ncclSystemError, "out of memory");
  free(blocks);
  return result;
}

static uint32_t demands_of(const struct ncclComm *comm, const struct call *c, struct mesh_demand *d, uint64_t *leaves) {
  const uint32_t n = (uint32_t)comm->nranks, words = (n + 63) / 64;
  const size_t z = SIZE[c->type];
  uint32_t made = 0;
  if (counted(c) && c->pairs) {
    for (uint32_t i = 0; i < c->npairs; i++) {
      const struct pair *x = c->pairs + i;
      if (x->from == x->to || !x->bytes || x->from >= n || x->to >= n) continue;
      uint64_t *set = leaves + (size_t)made * words;
      set[x->from / 64] |= UINT64_C(1) << (x->from % 64);
      d[made++] = (struct mesh_demand){x->to, 0, x->bytes, 0, set};
    }
    return made;
  }
  if (counted(c)) {
    for (uint32_t a = 0; a < n; a++)
      for (uint32_t b = 0; b < n; b++) {
        const uint64_t count = a == b ? 0 : hosted_count(c, (int)a, (int)b);
        if (!count) continue;
        uint64_t *set = leaves + (size_t)made * words;
        set[a / 64] |= UINT64_C(1) << (a % 64);
        d[made++] = (struct mesh_demand){b, 0, count * z, 0, set};
      }
    return made;
  }
  const uint64_t bytes = elements(c) * z;
  if (has_root(c->what)) {
    d[made++] = (struct mesh_demand){(uint32_t)c->root, 0, c->what == MESH_REDUCE ? bytes : 0, c->what == MESH_BROADCAST ? bytes : 0, NULL};
    return made;
  }
  for (uint32_t r = 0; r < n; r++) {
    const uint64_t part = (c->segments ? c->segments[r] : c->what == MESH_ALLREDUCE ? (c->count + n - 1) / n : c->count) * z;
    const uint64_t up = c->what == MESH_ALLGATHER ? 0 : c->what == MESH_ALLREDUCE ? bytes : part;
    const uint64_t down = c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLREDUCE ? 0 : part;
    d[made++] = (struct mesh_demand){r, 0, up, down, NULL};
  }
  return made;
}

static ncclResult_t compiled_of(ncclComm_t comm, const struct call *c, const struct mesh_demand *d, const struct flow *f, uint32_t first,
                                uint32_t last, struct steps *s) {
  const uint32_t n = (uint32_t)comm->nranks;
  uint32_t *root = calloc((size_t)f->count + 1, sizeof *root), *parent = calloc((size_t)f->count * n + 1, sizeof *parent);
  double *weight = calloc((size_t)f->count + 1, sizeof *weight);
  ncclResult_t result = root && parent && weight ? ncclSuccess : fail(ncclSystemError, "out of memory");
  const struct mesh_trees trees = {n, result ? 0 : mesh_flow_trees(n, d, f->demand, f->parent, f->share, f->count, first, last, root, parent, weight),
                                   root, parent, weight};
  uint32_t parts = 0, moves = 0;
  if (!result) mesh_compile_room(&trees, &parts, &moves);
  uint32_t *segment = result ? NULL : malloc((parts + 1) * sizeof *segment), *at = result ? NULL : malloc((n + 1) * sizeof *at);
  double *log_weight = result ? NULL : malloc((parts + 1) * sizeof *log_weight);
  struct mesh_move *move = result ? NULL : malloc((moves + 1) * sizeof *move);
  if (!result && (!segment || !at || !log_weight || !move)) result = fail(ncclSystemError, "out of memory");
  if (!result && mesh_compile(&trees, (uint32_t)c->what, has_root(c->what) ? (uint32_t)c->root : 0, &s->program, segment, log_weight, at, move))
    result = fail(ncclInternalError, "%s on %s: the solver's trees do not compile", call_name(c), comm->name);
  if (result) { free(segment); free(at); free(log_weight); free(move); }
  else s->compiled = 1;
  free(root); free(parent); free(weight);
  return result;
}

static ncclResult_t routed_of(ncclComm_t comm, const struct call *c, const struct mesh_demand *d, const struct flow *f, uint32_t first,
                              uint32_t last, struct mesh_operand operand, struct steps *s) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_routing g = {0};
  uint32_t *pair = calloc((size_t)n * n + 1, sizeof *pair), *start = calloc((size_t)f->count + 2, sizeof *start),
           *node = calloc((size_t)f->count * n + 1, sizeof *node);
  double *weight = calloc((size_t)f->count + 1, sizeof *weight);
  ncclResult_t result = pair && start && node && weight ? ncclSuccess : fail(ncclSystemError, "out of memory");
  if (!result) mesh_flow_routing(n, d, f->demand, f->parent, f->share, f->count, first, last, &g, pair, start, node, weight);
  if (!result) result = hosted_steps(comm, c, &g, operand, &s->steps, &s->count, &s->relayed);
  free(pair); free(start); free(node); free(weight);
  return result;
}

static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  const uint32_t N = (uint32_t)comm->nranks, words = (N + 63) / 64;
  uint32_t *lo = calloc(n + 1, sizeof *lo), *hi = calloc(n + 1, sizeof *hi), room = 0, demands = 0;
  for (size_t i = 0; i < n; i++)
    room += !crosses(list + i) || p2p(list + i) ? 0 : list[i].pairs ? list[i].npairs : counted(list + i) ? N * N : N;
  struct mesh_demand *d = calloc((size_t)room + 1, sizeof *d);
  uint64_t *leaves = calloc((size_t)room * words + 1, sizeof *leaves);
  struct flow f = {0, NULL, NULL, NULL};
  ncclResult_t result = lo && hi && d && leaves ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (size_t i = 0; !result && i < n; i++) {
    struct call *c = list + i;
    lo[i] = hi[i] = demands;
    if (!crosses(c) || p2p(c)) continue;
    hi[i] = demands += demands_of(comm, c, d + demands, leaves + (size_t)demands * words);
  }
  if (!result) result = flow_solve(comm, d, demands, &f);
  int paired = 0;
  for (size_t i = 0; i < n; i++) paired |= list[i].pairs != NULL;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
    struct steps *s = calloc(1, sizeof *s);
    if (!s) { result = fail(ncclSystemError, "out of memory"); break; }
    c->bound = s;
    if (!p2p(c) && !counted(c) && (result = compiled_of(comm, c, d, &f, lo[i], hi[i], s))) break;
    const struct mesh_program *program = p2p(c) || counted(c) ? NULL : &s->program;
    const int nccl = c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER;
    if (nccl) {
      uint32_t segments = 1;
      for (uint32_t p = 0; p < program->parts; p++) if (program->segment[p] + 1 > segments) segments = program->segment[p] + 1;
      if (segments != N) { result = fail(ncclInvalidUsage, "a reduce-scatter's or all-gather's program cuts %u segments, not one a rank of %u", segments, N); break; }
    }
    const uint32_t u = unit_of(c);
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)(SIZE[c->type] * u), elements(c) / u};
    if (counted(c) && (result = routed_of(comm, c, d, &f, lo[i], hi[i], operand, s))) break;
    const size_t most = (program ? program->first[program->nodes] : counted(c) ? s->count : 1) + 1;
    s->buffers = c->pairs ? 3 + c->npairs : counted(c) ? 2 + (s->relayed != 0) : 1 + (program ? program->scratch : 0);
    if ((!s->steps && !(s->steps = calloc(most, sizeof *s->steps))) || !(s->done = calloc(most, 1)) || !(s->channel = calloc(most, sizeof *s->channel)) ||
        !(s->entry = calloc(most, sizeof *s->entry)) || !(s->before = calloc(most, sizeof *s->before)) ||
        !(s->buffer = calloc(s->buffers, sizeof *s->buffer)) || !(s->at = calloc(s->buffers, sizeof *s->at)) || (nccl && !(s->cut = malloc((size_t)N * sizeof *s->cut)))) {
      result = fail(ncclSystemError, "out of memory");
      break;
    }
    s->operand = operand;
    for (uint32_t r = 0; nccl && r < N; r++) s->cut[r] = (c->segments ? c->segments[r] : c->count) / u;
    if (c->pairs && !(s->pair_call = malloc((c->npairs + 1) * sizeof *s->pair_call))) { result = fail(ncclSystemError, "out of memory"); break; }
    for (uint32_t m = 0; c->pairs && m < c->npairs; m++) {
      const struct pair *x = c->pairs + m;
      const int sends = x->from == (uint32_t)comm->rank, lands = x->to == (uint32_t)comm->rank;
      uint32_t before = 0;
      for (uint32_t e = 0; e < m; e++) before += c->pairs[e].from == x->from && c->pairs[e].to == x->to;
      s->pair_call[m] = UINT32_MAX;
      for (size_t j = 0, seen = 0; (sends || lands) && x->from != x->to && j < n; j++) {
        const struct call *y = list + j;
        if (y->comm != c->comm || !((sends && y->what == WHAT_SEND && y->peer == (int)x->to) || (lands && y->what == WHAT_RECV && y->peer == (int)x->from))) continue;
        if (seen++ == before) { s->pair_call[m] = (uint32_t)j; break; }
      }
    }
    if (p2p(c) && paired) s->count = 0;
    else if (p2p(c)) {
      if (c->peer < 0 || c->peer >= comm->nranks || comm->toward[c->peer] < 0) {
        result = fail(ncclInvalidUsage, "%s on %s: rank %d and rank %d are on one node, which no queue pair joins", call_name(c), comm->name,
                      comm->rank, c->peer);
        break;
      }
      s->count = 1;
      s->steps[0] = (struct mesh_step){c->what == WHAT_SEND ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)c->peer, 0, 0, 0, operand};
    } else if (!counted(c)) s->count = mesh_program_steps_cut(program, (uint32_t)comm->rank, operand, s->cut, comm->page, s->steps);
  }
  flow_free(&f);
  free(lo); free(hi); free(d); free(leaves);
  return result;
}

struct message { struct call *c; uint32_t step; int32_t war; uint64_t entry, need, rel_entry, rel_need; mesh_pages pages;
                 size_t offset, bytes, area, skew, stage; struct where data; uint64_t row; int staged; };

struct schedule { struct message *out, *in; size_t outs, ins, table; uint64_t first[2], k; struct mesh_plan *t; };

static uint32_t group_kind(const struct call *c) { return p2p(c) ? MESH_GROUP_POINT : counted(c) ? MESH_GROUP_HOSTED : MESH_GROUP_COLLECTIVE; }
static void group_of(const struct call *list, size_t n, uint32_t N, const struct mesh_step **steps, uint32_t *count,
                     struct mesh_group_call *calls, uint64_t page, struct mesh_group *group) {
  for (size_t i = 0; i < n; i++) calls[i] = (struct mesh_group_call){group_kind(list + i), 0, count + i * N, steps + i * N};
  *group = (struct mesh_group){N, (uint32_t)n, calls, page};
}

static size_t page_up(size_t bytes, uint64_t page) { return page ? (size_t)((bytes + page - 1) / page * page) : bytes; }

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
    *p = (struct schedule){.out = calloc(total, sizeof *p->out), .in = calloc(total, sizeof *p->in)};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    for (int send = 1; send >= 0 && !result; send--) {
      uint32_t placed = 0;
      mesh_group_lay(&group, me, (uint32_t)ch->peer, send, 0, scratch, made, &placed);
      for (uint32_t k = 0; k < placed; k++) {
        const struct mesh_place *x = &made[k].place;
        struct call *c = list + made[k].call;
        const struct mesh_step *step = c->bound->steps + x->step;
        const size_t bytes = piece_bytes(step);
        struct message m = {.c = c, .step = x->step, .war = -1, .rel_entry = x->entry, .pages = pages_of(bytes, comm->page),
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
        if (o->c != m->c || o->step >= m->step || (st->reads && st->reads[o->step] && m->c->direct_ok != 2) || !meet(st->steps + o->step, st->steps + m->step))
          continue;
        if (o->rel_entry + 1 > m->rel_need) { m->rel_need = o->rel_entry + 1; m->war = (int32_t)o->rel_entry; }
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
      if (x->op == MESH_STEP_SEND && x->buffer == 2) st->before[k] = (int32_t)j;
    }
  }
  free(steps); free(counts); free(group_calls); free(made); free(scratch);
  return result;
}

static void schedules_free(const struct ncclComm *comm, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < comm->count; h++) { free(plans[h].out); free(plans[h].in); }
  free(plans);
}

static struct where read_at(const struct call *c) { return (struct where){c->send_at.buffer, c->send_at.offset - c->bound->lo * c->bound->operand.element_bytes}; }

static struct where buffer_at(const struct call *c, uint32_t b, size_t offset) {
  size_t length = 0;
  const size_t z = SIZE[c->type], first = !b && c->direct && c->what == MESH_REDUCE_SCATTER ? segment_at(c, c->comm->rank, &length) * z : 0;
  return offset >= first && offset < first + length * z ? shifted(c->recv_at, offset - first) : shifted(c->bound->buffer[b], offset);
}

static ncclResult_t lease_take(size_t bytes, void **pages, void **buffer, size_t *offset);
static int bank_crossed(struct hdr *m, uint32_t row, uint64_t into, uint64_t bytes, uint64_t page) {
  const uint64_t base = atomic_load_explicit(&mesh_page(m)[row].address, memory_order_relaxed) - m->data_off + into;
  for (uint64_t line = (base / MESH_BANK + 1) * MESH_BANK; line < base + bytes; line += MESH_BANK)
    if ((line - base) % page) return 1;
  return 0;
}

static ncclResult_t place_group(struct ncclComm *comm, struct plan *plan) {
  const uint64_t page = comm->page;
  struct schedule *plans = plan->plans;
  for (uint32_t h = 0; h < comm->count; h++)
    for (int send = 1; send >= 0; send--)
      for (size_t k = 0; k < (send ? plans[h].outs : plans[h].ins); k++) {
        struct message *x = (send ? plans[h].out : plans[h].in) + k;
        const struct steps *st = x->c->bound;
        const struct mesh_step *step = st->steps + x->step;
        if (send && st->reads && st->reads[x->step]) x->data = shifted(read_at(x->c), x->offset);
        else if (!send && step->op == MESH_STEP_REDUCE) x->data = shifted(plan->base, x->area);
        else x->data = buffer_at(x->c, step->buffer, x->offset);
        const unsigned char *address = address_of(x->data);
        const int inside = in_region(comm, x->data, x->bytes);
        pthread_mutex_lock(&attach_lock);
        const struct lease *held = inside ? lease_at(address) : NULL;
        const uint64_t into = held ? (uint64_t)(address - (const unsigned char *)held->address) : 0;
        pthread_mutex_unlock(&attach_lock);
        if (held && bank_crossed(comm->attach->context.M, held->row, into, x->bytes, page))
          return fail(ncclInvalidUsage, "%s on communicator %s: the %zu bytes its %s rank %u %s (buffer %u, byte %zu) lie at byte %llu of their lease, so "
                      "a page of them crosses a 4 GiB bank's line, which no SGE may cross (max_sge 1, and the provider keeps an address's low 32 bits)",
                      call_name(x->c), comm->name, x->bytes, send ? "SEND to" : "receive from", step->peer, send ? "reads" : "lands in", step->buffer,
                      x->offset, (unsigned long long)into);
        x->staged = !held;
        if (held) { x->row = held->row + into / page; x->skew = into % page; continue; }
        if (!plan->stage) {
          size_t at = 0;
          const ncclResult_t made = lease_take(plan->stage_bytes, &plan->stage, plan->buffer ? &plan->stage_buffer : NULL, &at);
          if (made) return made;
          plan->stage_offset = at;
          pthread_mutex_lock(&attach_lock);
          plan->stage_row = lease_at(plan->stage)->row;
          pthread_mutex_unlock(&attach_lock);
        }
        x->row = plan->stage_row + x->stage / page; x->skew = 0;
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
  struct mesh_plan *table[comm->count + 1];
  for (uint32_t h = 0; h < comm->count; h++) table[h] = p->plans ? p->plans[h].t : NULL;
  if (p->pages) retire_locked(comm, p->pages, table, comm->count);
  if (p->stage) retire_locked(comm, p->stage, NULL, 0);
  if (p->command) metal_release(p->command);
  calls_free(p->list, p->n);
  schedules_free(comm, p->plans);
  rows_free(&p->rows, 1);
  free(p);
}

static int in_place(const struct call *c) {
  return c->what == MESH_ALLREDUCE || c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER || (c->what == MESH_REDUCE && c->root == c->comm->rank);
}
static int direct_ok(const struct call *c) {
  if (p2p(c) || counted(c) || !crosses(c) || c->premultiplied) return 0;
  size_t length = elements(c), before = 0;
  if (c->what == MESH_ALLGATHER) before = segment_at(c, c->comm->rank, &length);
  const unsigned char *a = address_of(c->send_at), *b = address_of(c->recv_at);
  return in_place(c) && b && a == b + before * SIZE[c->type] ? 2 : 1;
}
static int call_same(const struct call *a, const struct call *b) {
  const size_t segs = (counted(a) ? 2 : 1) * (size_t)a->comm->nranks;
  return a->what == b->what && a->root == b->root && a->peer == b->peer && a->count == b->count && a->total == b->total && a->type == b->type &&
         a->combine == b->combine && a->post == b->post && a->premultiplied == b->premultiplied && a->unit == b->unit &&
         a->direct_ok == b->direct_ok && !a->matrix == !b->matrix && !a->stream == !b->stream && !a->segments == !b->segments &&
         (!a->segments || !memcmp(a->segments, b->segments, segs * sizeof *a->segments)) &&
         (!a->matrix || !memcmp(a->matrix, b->matrix, (size_t)a->comm->nranks * a->comm->nranks * sizeof *a->matrix)) &&
         a->npairs == b->npairs && (!a->npairs || !memcmp(a->pairs, b->pairs, a->npairs * sizeof *a->pairs));
}

static ncclResult_t lease_take(size_t bytes, void **pages, void **buffer, size_t *offset) {
  ncclResult_t result = ncclMemAlloc(pages, bytes);
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
    if (!grown || lease_take(made.bytes, &made.pages, &made.buffer, &made.offset)) {
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
    const size_t cells = (size_t)comm->nranks * comm->nranks;
    c->bound = NULL; c->complement = NULL; c->complements = 0; c->segments = NULL; c->matrix = NULL; c->pairs = NULL; c->at = SIZE_MAX;
    if ((list[i].segments && !(c->segments = malloc(segs * sizeof *c->segments))) || (list[i].matrix && !(c->matrix = malloc(cells * sizeof *c->matrix))) ||
        (list[i].npairs && !(c->pairs = malloc(list[i].npairs * sizeof *c->pairs))))
      result = fail(ncclSystemError, "out of memory");
    if (c->pairs) memcpy(c->pairs, list[i].pairs, list[i].npairs * sizeof *c->pairs);
    if (c->segments) memcpy(c->segments, list[i].segments, segs * sizeof *c->segments);
    if (c->matrix) memcpy(c->matrix, list[i].matrix, cells * sizeof *c->matrix);
  }
  if (!result) result = plan_calls(comm, p->list, n);
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = p->list + i;
    struct steps *st = c->bound;
    if (counted(c) && st && st->relayed) { const size_t each = page_up(st->relayed, comm->page); st->at[2] = (bytes += each) - each; }
    if (p2p(c) || counted(c)) continue;
    const size_t each = page_up(elements(c) * SIZE[c->type], comm->page);
    c->direct = st && plan_direct(c);
    if (st && c->direct_ok && !c->direct)
      result = fail(ncclInternalError, "%s on %s: its program reads a range it has partly written", call_name(c), comm->name);
    c->at = in_place(c) ? SIZE_MAX : (bytes += each) - each;
    for (uint32_t b = 1; st && b < st->buffers; b++) st->at[b] = c->direct && !st->needed[b] ? SIZE_MAX : (bytes += each) - each;
  }
  if (!result) result = schedule_group(comm, p->list, n, p->plans, &bytes);
  for (uint32_t h = 0; h < comm->count && !result; h++)
    for (size_t k = 0; k < p->plans[h].outs + p->plans[h].ins; k++) {
      struct message *m = k < p->plans[h].outs ? p->plans[h].out + k : p->plans[h].in + (k - p->plans[h].outs);
      if (k >= p->plans[h].outs && m->c->bound->steps[m->step].op == MESH_STEP_REDUCE) continue;
      m->stage = p->stage_bytes;
      p->stage_bytes += page_up(m->bytes, comm->page);
    }
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    struct schedule *s = p->plans + h;
    if (!s->outs && !s->ins) continue;
    bytes = (bytes + MESH_LINE - 1) / MESH_LINE * MESH_LINE;
    s->table = bytes;
    bytes += mesh_plan_bytes((uint32_t)s->outs, (uint32_t)s->ins);
  }
  p->word = bytes;
  size_t offset = 0;
  if (!result && (bytes || list[0].stream)) result = lease_take(bytes + sizeof(uint32_t), &p->pages, list[0].stream ? &p->buffer : NULL, &offset);
  if (p->pages) memset(p->pages, 0, p->word + sizeof(uint32_t));
  p->base = p->buffer ? (struct where){p->buffer, offset} : (struct where){NULL, (size_t)p->pages};
  if (result) { plan_free(comm, p); return result; }
  for (uint32_t h = 0; h < comm->count; h++) {
    struct schedule *s = p->plans + h;
    if (!s->outs && !s->ins) continue;
    s->t = (struct mesh_plan *)((char *)p->pages + s->table);
    s->t->sends = (uint32_t)s->outs; s->t->lands = (uint32_t)s->ins;
    pthread_mutex_lock(&attach_lock);
    mesh_queue_link(&comm->attach->context, s->t);
    pthread_mutex_unlock(&attach_lock);
  }
  p->next = comm->plans;
  comm->plans = p;
  *out = p;
  return ncclSuccess;
}

static int plan_quiet(struct ncclComm *comm, const struct plan *p);
enum { SETS = 4 };
static ncclResult_t plan_take(struct ncclComm *comm, struct call *list, size_t n, struct plan **out) {
  release(comm);
  for (size_t i = 0; i < n; i++) list[i].direct_ok = direct_ok(list + i);
  struct plan *p = NULL;
  for (;;) {
    struct plan *held = NULL;
    uint32_t shaped = 0;
    for (p = comm->plans; p; p = p->next) {
      int same = p->n == n;
      for (size_t i = 0; same && i < n; i++) same = call_same(p->list + i, list + i);
      if (same && plan_idle(p) && plan_quiet(comm, p)) break;
      shaped += same;
      if (same && !p->flight) held = p;
    }
    const int status = held && held->command ? metal_ended(held->command) : 0;
    if (p || shaped < SETS || !held || status == 2) break;
    if (revoked(comm)) return revoked_fail(comm);
    if (status < 0)
      return fail(ncclSystemError, "%s on communicator %s: the %u plans of its shape are held by GPU work whose command buffer ended in an error",
                  call_name(list), comm->name, (unsigned)SETS);
    void *command = status ? NULL : metal_retain(held->command);
    pthread_mutex_unlock(&group_lock);
    if (command) { metal_completed(command, MESH_PAUSE_NS); metal_release(command); }
    else sched_yield();
    pthread_mutex_lock(&group_lock);
    release(comm);
  }
  const ncclResult_t result = p ? ncclSuccess : plan_make(comm, list, n, &p);
  if (result) return result;
  p->rows.at = p->rows.used = 0;
  for (size_t i = 0; i < n; i++) {
    struct call *c = p->list + i, made = list[i];
    made.bound = c->bound; made.segments = c->segments; made.matrix = c->matrix; made.pairs = c->pairs; made.complement = c->complement; made.complements = c->complements;
    made.direct = c->direct; made.at = c->at;
    *c = made;
  }
  p->flight = 1;
  *out = p;
  return ncclSuccess;
}

static void tables_start(struct ncclComm *comm, struct schedule *plans);
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
      m->c->bound->entry[m->step] = m->entry;
    }
    for (size_t k = 0; k < p->ins; k++) {
      struct message *m = p->in + k;
      m->entry = ch->lands + m->rel_entry;
      m->need = m->rel_need ? ch->sends + m->rel_need : 0;
      m->c->bound->entry[m->step] = m->entry;
    }
  }
  const ncclResult_t result = place_group(comm, plan);
  if (result) return result;
  tables_start(comm, plan->plans);
  commit_group(comm, plan->plans);
  return ncclSuccess;
}

static struct ncclComm *touching;
static void touch(struct ncclComm *comm) {
  if (comm->touched) return;
  comm->touched = 1;
  comm->next_touched = touching;
  touching = comm;
}

static uint64_t region_offset(const struct ncclComm *comm, const void *at) { return (uint64_t)((const char *)at - (const char *)comm->attach->context.M); }
static uint64_t ready_need(uint64_t k, uint64_t index) { return (k << 32) + index + 1; }
static void key_of(struct mesh_entry *e, uint64_t id, uint32_t to, uint32_t from, uint64_t space, uint64_t position) {
  e->key[0] = id; e->key[1] = to; e->key[2] = from; e->key[3] = space; e->key[4] = position;
}
static void table_begin(struct ncclComm *comm, struct mesh_plan *t, uint64_t k) {
  uint32_t empty[2] = {0, 0};
  for (uint32_t e = 0; e < t->sends + t->lands; e++) {
    struct mesh_entry *x = t->entry + e;
    atomic_store_explicit(&x->count, mesh_call(k), memory_order_relaxed);
    atomic_store_explicit(&x->posted, mesh_call(k), memory_order_relaxed);
    if (!x->pages) { atomic_store_explicit(&x->done, k, memory_order_relaxed); empty[e >= t->sends]++; }
  }
  for (int way = 0; way < 2; way++) atomic_store_explicit(&t->completed[way], mesh_call(k) | empty[way], memory_order_relaxed);
  if (empty[0] == t->sends) atomic_store_explicit(&t->sent, k, memory_order_relaxed);
  if (empty[1] == t->lands) atomic_store_explicit(&t->landed, k, memory_order_relaxed);
  atomic_store_explicit(&t->start, k, memory_order_release);
  mesh_queue_begun(comm->attach->context.M, comm->attach->context.client);
}
static void tables_start(struct ncclComm *comm, struct schedule *plans) {
  const uint32_t me = (uint32_t)comm->rank;
  for (uint32_t h = 0; h < comm->count; h++) {
    struct schedule *p = plans + h;
    if (!p->t) continue;
    const uint32_t peer = (uint32_t)comm->channels[h].peer;
    p->k = ++comm->attach->calls;
    for (size_t k = 0; k < p->outs; k++) {
      const struct message *m = p->out + k;
      struct mesh_entry *e = p->t->entry + m->rel_entry;
      e->row = m->row; e->offset = m->skew; e->bytes = m->bytes; e->pages = pages_count(m->pages);
      e->word = region_offset(comm, &p->t->ready); e->need = ready_need(p->k, m->rel_entry);
      key_of(e, comm->id, peer, me, 0, m->entry);
    }
    for (size_t k = 0; k < p->ins; k++) {
      const struct message *m = p->in + k;
      struct mesh_entry *e = p->t->entry + p->outs + m->rel_entry;
      e->row = m->row; e->offset = m->skew; e->bytes = m->bytes; e->pages = pages_count(m->pages); e->word = 0; e->need = 0;
      key_of(e, comm->id, me, peer, 0, m->entry);
    }
    table_begin(comm, p->t, p->k);
  }
}

static void commit_group(struct ncclComm *comm, const struct schedule *plans) {
  for (uint32_t h = 0; h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    const struct schedule *p = plans + h;
    ch->sends = p->first[0] + p->outs;
    ch->lands = p->first[1] + p->ins;
  }
}

static int ready(const struct steps *s, uint32_t k) { return s->before[k] < 0 || s->done[s->before[k]]; }

static uint64_t stamp_of(const _Atomic uint64_t *word) { return atomic_load_explicit(word, memory_order_acquire); }
static ncclResult_t lost_fail(const struct channel *ch) {
  return fail(ncclRemoteError, "the link to node %u was lost (its plan's stamps ~0)", ch->node);
}

static ncclResult_t run_group(struct ncclComm *comm, struct plan *plan) {
  struct schedule *const plans = plan->plans;
  for (;;) {
    if (revoked(comm)) return revoked_fail(comm);
    int open = 0;
    for (uint32_t h = 0; h < comm->count; h++) {
      struct channel *ch = comm->channels + h;
      struct schedule *p = plans + h;
      const uint64_t lands = p->first[1] + p->ins;
      while (ch->consumed < lands) {
        const struct message *m = p->in + (ch->consumed - p->first[1]);
        struct steps *st = m->c->bound;
        const struct mesh_step *step = st->steps + m->step;
        if (!ready(st, m->step) || ch->published < m->need) break;
        const uint64_t landed = stamp_of(&p->t->entry[p->outs + m->rel_entry].done), sent = m->war < 0 ? p->k : stamp_of(&p->t->entry[m->war].done);
        if (stamp_lost(landed) || stamp_lost(sent)) return lost_fail(ch);
        if (landed < p->k || sent < p->k) break;
        if (step->op == MESH_STEP_REDUCE) {
          unsigned char *to = address_of(buffer_at(m->c, step->buffer, m->offset));
          const unsigned char *with = st->fresh && st->fresh[m->step] ? address_of(read_at(m->c)) + m->offset : to;
          combine_from(ktype(m->c), kop(m->c), to, with, address_of(m->data), kcount(m->c, m->bytes));
        } else if (m->staged) memcpy(address_of(m->data), (const unsigned char *)plan->stage + m->stage, m->bytes);
        st->done[m->step] = 1;
        ch->consumed++;
      }
      while (ch->published < p->first[0] + p->outs) {
        const struct message *m = p->out + (ch->published - p->first[0]);
        if (!ready(m->c->bound, m->step)) break;
        if (m->staged) memcpy((unsigned char *)plan->stage + m->stage, address_of(m->data), m->bytes);
        atomic_thread_fence(memory_order_release);
        atomic_store_explicit(&p->t->ready, ready_need(p->k, m->rel_entry), memory_order_release);
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
      const uint64_t sent = plans[h].outs ? stamp_of(&plans[h].t->sent) : plans[h].k;
      if (stamp_lost(sent)) return lost_fail(comm->channels + h);
      all &= sent >= plans[h].k;
    }
  }
  return ncclSuccess;
}

static ncclResult_t run(struct ncclComm *comm, struct plan *plan) {
  struct schedule *const plans = plan->plans;
  const ncclResult_t result = run_group(comm, plan);
  for (uint32_t h = 0; result && h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    const struct schedule *p = plans + h;
    if (p->outs) ch->published = p->first[0] + p->outs;
    if (p->ins) ch->consumed = p->first[1] + p->ins;
  }
  return result;
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
  const size_t z = SIZE[c->type], n = c->count, e = c->bound ? c->bound->operand.element_bytes : z;
  const int nranks = c->comm->nranks;
  unsigned char *sum = address_of(c->sum), *recv = c->recv;
  for (uint32_t k = 0; k < c->complements; k++) {
    const uint64_t *x = c->complement + 3 * k;
    memmove(address_of(buffer_at(c, 0, (size_t)x[0] * e)), address_of(shifted(c->send_at, (size_t)x[1] * e)), (size_t)x[2] * e);
  }
  if (!sum) return;
  if (c->what == MESH_ALLREDUCE || (c->what == MESH_REDUCE && c->root == c->comm->rank)) {
    if (c->post) truncdiv(c->type, recv, sum, n, (uint64_t)nranks);
    else if (sum != recv) memmove(recv, sum, n * z);
  } else if (c->what == MESH_REDUCE_SCATTER) {
    size_t length;
    const unsigned char *from = c->direct ? recv : sum + segment_at(c, c->comm->rank, &length) * z;
    segment_at(c, c->comm->rank, &length);
    if (c->post) truncdiv(c->type, recv, from, length, (uint64_t)nranks);
    else if (from != recv) memmove(recv, from, length * z);
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

enum { GPU_ENDED, GPU_ERROR, GPU_UNCOMMITTED, GPU_REVOKED };

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
  default: return ncclSuccess;
  }
}

static int comm_issued(const struct ncclComm *comm);
static void calls_free(struct call *list, size_t n);
static void operands_at(const struct plan *plan, struct call *c) {
  struct steps *st = c->bound;
  if (counted(c) && st) { st->buffer[0] = c->send_at; st->buffer[1] = c->recv_at; }
  if (counted(c) && st && st->relayed) st->buffer[2] = shifted(plan->base, st->at[2]);
  for (uint32_t m = 0; c->pairs && st && m < c->npairs; m++)
    if (st->pair_call[m] != UINT32_MAX) {
      const struct call *y = plan->list + st->pair_call[m];
      st->buffer[3 + m] = y->what == WHAT_SEND ? y->send_at : y->recv_at;
    }
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
  if (remote && comm_issued(comm))
    result = fail(ncclInvalidUsage, "a host-path group while this communicator's GPU groups are issued and not complete (ncclMeshComplete), or "
                  "an exchange's start is not bound to the kernel that publishes it (ncclMeshBind)");
  if (!result && remote && finished && comm->last) result = gpu_finished(comm);
  struct plan *plan = NULL;
  if (!result) result = plan_take(comm, list, n, &plan);
  if (result) { pthread_mutex_unlock(&group_lock); return result; }
  struct call *const calls = plan->list;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = calls + i;
    const size_t bytes = elements(c) * SIZE[c->type];
    operands_at(plan, c);
    if (p2p(c) || counted(c) || c->direct) continue;
    stage(c, address_of(c->sum));
    for (uint32_t b = 1; c->bound && b < c->bound->buffers; b++)
      if (c->bound->at[b] != SIZE_MAX) memcpy(address_of(c->bound->buffer[b]), address_of(c->sum), bytes);
  }
  if (!result && remote && !(result = plan_start(comm, plan))) atomic_fetch_add(&comm->runs, 1);
  const int running = remote && !result;
  pthread_mutex_unlock(&group_lock);
  if (running) result = run(comm, plan);
  if (running) atomic_fetch_sub(&comm->runs, 1);
  for (size_t i = 0, self = 0; i < n && !result; i++) {
    struct call *c = calls + i;
    if (c->what == WHAT_RECV && c->peer == comm->rank) {
      while (self < n && !(calls[self].what == WHAT_SEND && calls[self].peer == comm->rank)) self++;
      if (self < n) { memmove(c->recv, calls[self].send, c->count * SIZE[c->type]); self++; }
    } else if (counted(c)) {
      size_t sent, landed, own = own_segment(c, &sent, &landed);
      if (own) memmove((unsigned char *)c->recv + landed, (const unsigned char *)c->send + sent, own);
    } else if (!p2p(c)) finish(c);
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
  struct inflight *next_group;
};

static struct inflight *inflight, *inflight_last;
static uint64_t tickets;
static _Atomic uint64_t retired;

static void call_parts_free(struct call *c) {
  steps_free(c->bound);
  free(c->segments);
  free(c->matrix);
  free(c->pairs);
  free(c->complement);
}
static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) call_parts_free(list + i);
  free(list);
}

static void inflight_free(struct inflight *g, int retired) {
  if (g->plan && retired) g->plan->leaving = 1;
  else if (g->plan) g->plan->flight = 0;
  free(g->start);
  free(g);
}

static int comm_issued(const struct ncclComm *comm) {
  for (const struct inflight *g = inflight; g; g = g->next_group)
    if (g->comm == comm) return 1;
  return comm->unpublished > 0;
}

static void inflight_drop(struct ncclComm *comm) {
  inflight_last = NULL;
  for (struct inflight **at = &inflight; *at;) {
    struct inflight *g = *at;
    if (g->comm != comm) { inflight_last = g; at = &g->next_group; continue; }
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

static int publishable(struct ncclComm *comm, uint32_t h, uint64_t at) {
  struct inflight *g = inflight_at(comm, h, 0, at);
  const struct message *m = g ? out_at(g, h, at) : NULL;
  return m && ready(m->c->bound, m->step);
}

static size_t table_word(const struct inflight *g, const _Atomic uint64_t *word) {
  return g->plan->base.offset + (size_t)((const char *)word - (const char *)g->plan->pages);
}

static ncclResult_t publish(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t upto, int *published) {
  struct channel *ch = comm->channels + h;
  ncclResult_t result = ncclSuccess;
  while (!result && ch->published < upto) {
    struct inflight *g = inflight_at(comm, h, 0, ch->published);
    const struct schedule *s = g->plans + h;
    const uint64_t last = upto < g->end[h][0] ? upto : g->end[h][0];
    const struct metal_release release = {g->plan->buffer, table_word(g, &s->t->ready), ready_need(s->k, out_at(g, h, last - 1)->rel_entry)};
    void *operands[64];
    size_t n = 0;
    for (uint64_t at = ch->published; !result && at < last; at++) {
      const struct message *m = out_at(g, h, at);
      if (m->staged) metal_resident(program, g->plan->stage_buffer);
      if (m->staged)
        result = gpu(metal_copy(program, METAL_PLAIN, g->plan->stage_buffer, g->plan->stage_offset + m->stage, m->data.buffer, m->data.offset, m->bytes),
                     "a SEND's staging copy");
      if (n == sizeof operands / sizeof *operands) { metal_uses(program, operands, n, 0); n = 0; }
      operands[n++] = m->staged ? g->plan->stage_buffer : m->data.buffer;
    }
    if (result) break;
    metal_uses(program, operands, n, 0);
    if (!(result = gpu(metal_publish(program, &release), "a publication"))) { ch->published = last; *published = 1; }
  }
  return result;
}

static ncclResult_t consume(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t at, uint32_t base) {
  struct channel *ch = comm->channels + h;
  struct inflight *g = inflight_at(comm, h, 1, at);
  const struct message *m = in_at(g, h, at);
  struct steps *st = m->c->bound;
  const struct mesh_step *step = st->steps + m->step;
  ncclResult_t result = ncclSuccess;
  if (st->before[m->step] >= 0) {
    const int32_t j = st->before[m->step];
    result = gpu(metal_depend(program, base + st->channel[j], st->entry[j] + 1), "a receive's dependency");
  }
  struct schedule *s = g->plans + h;
  const _Atomic uint64_t *sent = m->war >= 0 ? &s->t->entry[m->war].done : NULL;
  const struct metal_wait wait = {g->plan->buffer, m->staged ? g->plan->stage_buffer : m->data.buffer, table_word(g, &s->t->entry[s->outs + m->rel_entry].done),
                                  sent ? table_word(g, sent) : 0, m->staged ? g->plan->stage_offset + m->stage : m->data.offset, s->k, sent ? s->k : 0};
  if (!result && step->op == MESH_STEP_REDUCE) {
    const struct where to = buffer_at(m->c, step->buffer, m->offset);
    if (st->fresh && st->fresh[m->step]) {
      const struct where with = shifted(read_at(m->c), m->offset);
      result = gpu(metal_spin(program, &wait), "a wait");
      if (!result)
        result = gpu(metal_combine3(program, ktype(m->c), kop(m->c), to.buffer, to.offset, m->data.buffer, m->data.offset, with.buffer, with.offset,
                                    kcount(m->c, m->bytes)), "a first combine");
    } else result = gpu(metal_land(program, ktype(m->c), kop(m->c), to.buffer, to.offset, kcount(m->c, m->bytes), &wait), "a landing");
  } else if (!result) result = gpu(metal_spin(program, &wait), "a wait");
  if (m->staged) metal_resident(program, g->plan->stage_buffer);
  if (!result && m->staged && step->op != MESH_STEP_REDUCE)
    result = gpu(metal_copy(program, METAL_PLAIN, m->data.buffer, m->data.offset, g->plan->stage_buffer, g->plan->stage_offset + m->stage, m->bytes),
                 "a landing's staging copy");
  if (result) return result;
  st->done[m->step] = 1;
  if (at == ch->consumed) ch->consumed = at + 1;
  return ncclSuccess;
}

static ncclResult_t complete(struct metal_program *program, struct inflight *through, int *published) {
  if (!inflight) return ncclSuccess;
  size_t comms = 0, room = 0;
  struct ncclComm **comm = NULL;
  uint32_t *base = NULL, tracks = 0;
  ncclResult_t result = ncclSuccess;
  for (struct inflight *g = inflight; g && !result; g = g == through ? NULL : g->next_group) {
    size_t k = 0;
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
        const struct message *m = out_at(inflight_at(c, h, 0, at), h, at);
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
        result = gpu(metal_record(program, track, METAL_CONSUME, at, in_at(inflight_at(c, h, 1, at), h, at)->need), "a consumption's record");
        if (!result) result = consume(program, c, h, at, base[k]);
      }
    }
  }
  if (!result) result = gpu(metal_record(program, METAL_NONE, 0, 0, 0), "the tracks' end");
  if (!result) result = gpu(metal_tracks(program, tracks, consumed), "the tracks");
  else metal_record(program, METAL_NONE, 0, 0, 0);
  free(end); free(consumed); free(comm); free(base);
  return result;
}

static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
  if (g->plan) metal_rows(program, &g->plan->rows);
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
  const uint64_t u = unit_of(c);
  uint64_t lo = 0, hi = elements(c) / u;
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
  if (c->what == MESH_REDUCE_SCATTER) { size_t length; first = segment_at(c, c->comm->rank, &length) / u; last = first + length / u; }
  if (c->what == MESH_REDUCE && c->root != c->comm->rank) last = first;
  if (direct && c->direct_ok != 2) {
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
  const size_t z = c->bound ? c->bound->operand.element_bytes : SIZE[c->type];
  ncclResult_t result = ncclSuccess;
  for (uint32_t k = 0; k < c->complements && !result; k++) {
    const uint64_t *x = c->complement + 3 * k;
    const struct where to = buffer_at(c, 0, x[0] * z);
    result = gpu(metal_copy(program, METAL_PLAIN, to.buffer, to.offset, c->send_at.buffer, c->send_at.offset + x[1] * z, x[2] * z),
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
      const struct where from = c->direct && c->what == MESH_REDUCE_SCATTER ? recv : (struct where){sum.buffer, sum.offset + before * z};
      if (c->post) result = gpu(metal_truncdiv(program, c->type, recv.buffer, recv.offset, from.buffer, from.offset, length, (uint64_t)comm->nranks), "the result");
      else if (!same(from, recv)) result = gpu(metal_copy(program, METAL_LAND, recv.buffer, recv.offset, from.buffer, from.offset, length * z), "the result");
    } else if ((c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER) && !same(sum, recv))
      result = gpu(metal_copy(program, METAL_LAND, recv.buffer, recv.offset, sum.buffer, sum.offset, elements(c) * z), "the result");
  }
  return result;
}

static ncclResult_t retire_groups(struct metal_program *program) {
  ncclResult_t result = ncclSuccess;
  while (inflight && !result) {
    struct inflight *g = inflight;
    struct ncclComm *comm = g->comm;
    for (uint32_t h = 0; h < comm->count; h++)
      if (comm->channels[h].consumed < g->end[h][1] || comm->channels[h].published < g->end[h][0]) return ncclSuccess;
    if (!program) return ncclSuccess;
    if (program) touch(comm);
    if (program && g->plan) metal_rows(program, &g->plan->rows);
    for (uint32_t h = 0; h < comm->count && !result; h++) {
      struct schedule *s = g->plans + h;
      if (!s->outs) continue;
      const struct metal_wait wait = {g->plan->buffer, NULL, table_word(g, &s->t->sent), 0, 0, s->k, 0};
      result = gpu(metal_spin(program, &wait), "a wait for the group's SENDs");
    }
    if (!result && g->n) result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g, 1);
  }
  return result;
}

static ncclResult_t stamp(struct metal_program *program, struct plan *p) {
  const ncclResult_t result = p->buffer ? gpu(metal_stamp(program, p->buffer, p->word), "a plan's stamp") : ncclSuccess;
  void *command = p->buffer ? metal_command(program) : NULL;
  if (command) { metal_release(p->command); p->command = command; }
  p->reader += p->buffer != NULL;
  p->flight = !p->leaving;
  p->leaving = 0;
  return result;
}

static ncclResult_t program_end(struct metal_program *program, int borrowed, ncclResult_t result) {
  struct plan *held = NULL;
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched)
    for (struct plan *p = comm->plans; p; p = p->next)
      if (p->flight && p->buffer && metal_rows(program, &p->rows) == &p->rows) held = p;
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched)
    for (struct plan *p = comm->plans; p; p = p->next)
      if (p->flight && p != held) { const ncclResult_t stamped = stamp(program, p); if (!result) result = stamped; }
  if (held) { const ncclResult_t stamped = stamp(program, held); if (!result) result = stamped; }
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
  struct plan *plan = NULL;
  if (!result) result = plan_take(comm, list, n, &plan);
  if (result) { pthread_mutex_unlock(&group_lock); calls_free(list, n); return result; }
  struct call *const calls = plan->list;
  const char *message = NULL;
  struct metal_program *program = metal4 ? metal_begin4(list[0].encoder, list[0].table, list[0].residency, &message)
                                         : metal_begin(list[0].command, list[0].encoder, &message);
  if (!program) {
    plan->flight = 0;
    pthread_mutex_unlock(&group_lock); calls_free(list, n);
    return fail(ncclSystemError, "%s", message);
  }
  touch(comm);
  metal_rows(program, &plan->rows);
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
    for (size_t i = 0; i < n && !result; i++) result = complement_copy(program, calls + i);
    if (!result) result = retire_groups(program);
  } else if (!result) result = epilogue(program, calls, n);
  plan->leaving = !g;
  result = program_end(program, list[0].encoder != NULL, result);
  if (issue && !result) { issue->ticket = ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  pthread_mutex_unlock(&group_lock);
  calls_free(list, n);
  return result;
}

ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream *stream, int *published) {
  int any = 0;
  if (published) *published = 0;
  if (!stream || (!stream->commandBuffer && !(stream->argumentTable && stream->commandEncoder && stream->residencySet)))
    return fail(ncclInvalidArgument, "ncclMeshComplete without a command buffer or a Metal 4 encoder, argument table and residency set");
  pthread_mutex_lock(&group_lock);
  ncclResult_t result = ncclSuccess;
  if (inflight && ticket >= inflight->ticket) {
    struct inflight *through = inflight;
    while (through->next_group && through->next_group->ticket <= ticket) through = through->next_group;
      const char *message = NULL;
    struct metal_program *program = stream->argumentTable
      ? metal_begin4(stream->commandEncoder, stream->argumentTable, stream->residencySet, &message)
      : metal_begin(stream->commandBuffer, stream->commandEncoder, &message);
    if (!program) result = fail(ncclSystemError, "%s", message);
    else {
      for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group) {
        if (g->plan && g->plan->buffer) metal_resident(program, g->plan->buffer);
        if (g->plan && g->plan->stage_buffer) metal_resident(program, g->plan->stage_buffer);
      }
      result = walk(program, NULL, through, &any);
      if (!result) result = retire_groups(program);
      result = program_end(program, stream->commandEncoder != NULL, result);
    }
  }
  pthread_mutex_unlock(&group_lock);
  if (published) *published = any;
  return result;
}

uint64_t ncclMeshRetired(void) { return atomic_load(&retired); }

struct slot { void *pages, *command; size_t offset; struct mesh_table *table; struct metal_rows rows; uint64_t row, start, landing, landing_k, *k, *kl; uint32_t bound; int unbound; };
struct leg { uint32_t x, piece, channel, index; int32_t from; };
struct ncclMeshRequest {
  struct ncclComm *comm;
  ncclDataType_t type;
  size_t *parts, *area, *tab, *ltab, table, bytes;
  uint32_t *sends, *lands;
  struct leg *in, *out;
  double *share;
  uint32_t ins, outs, peers, waits, relays, pieces, *own, *awaited, *first, *count;
  void *pages, *buffer;
  uint64_t starts, seq, *cut;
  struct slot slots[SETS];
  struct ncclMeshRequest *next;
};

static int plan_quiet(struct ncclComm *comm, const struct plan *p) {
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct mesh_plan *t = p->plans[h].t;
    const uint64_t k = p->plans[h].k;
    if (t && k && (stamp_of(&t->sent) < k || stamp_of(&t->landed) < k || !mesh_plan_balanced(t))) return 0;
  }
  return 1;
}

static struct mesh_plan *slot_table(const struct ncclMeshRequest *r, const struct slot *s, uint32_t h, int land) {
  const size_t at = land ? r->ltab[h] : r->tab[h];
  return at ? (struct mesh_plan *)((char *)s->pages + at) : NULL;
}
static struct mesh_entry *slot_entry(const struct ncclMeshRequest *r, const struct slot *s, uint32_t h, int land, uint32_t index) {
  return slot_table(r, s, h, land)->entry + index;
}
static uint64_t slot_address(const struct ncclMeshRequest *r, const struct slot *s, const void *at) {
  return metal_address(r->buffer) + s->offset + (uint64_t)((const char *)at - (const char *)s->pages);
}

static int tables_held(const struct ncclMeshRequest *r, const struct slot *s, int land) {
  for (uint32_t h = 0; h < r->comm->count; h++) {
    const struct mesh_plan *t = slot_table(r, s, h, land);
    const uint64_t k = land ? s->kl[h] : s->k[h];
    if (t && k && (stamp_of(&t->sent) < k || stamp_of(&t->landed) < k || !mesh_plan_balanced(t))) return 1;
  }
  return 0;
}
static int kernels_held(const struct slot *s) { return __atomic_load_n(&s->table->ended, __ATOMIC_ACQUIRE) < s->bound; }
static int slot_held(const struct ncclMeshRequest *r, const struct slot *s, int tables) {
  return kernels_held(s) || (tables && (tables_held(r, s, 0) || tables_held(r, s, 1)));
}

static int request_held(const struct ncclMeshRequest *r, int tables) {
  int held = 0;
  for (uint32_t i = 0; i < SETS; i++) held |= slot_held(r, r->slots + i, tables);
  return held;
}

static struct slot *slot_of(struct ncclMeshRequest *r, uint64_t start) { return start && r->slots[start % SETS].start == start ? r->slots + start % SETS : NULL; }

static void request_parts_free(struct ncclMeshRequest *r) {
  free(r->parts); free(r->area); free(r->in); free(r->out); free(r->share); free(r->own); free(r->awaited); free(r->first);
  free(r->count); free(r->tab); free(r->ltab); free(r->sends); free(r->lands); free(r->cut);
  for (uint32_t i = 0; i < SETS; i++) { free(r->slots[i].k); free(r->slots[i].kl); }
  free(r);
}

static void request_end(struct ncclMeshRequest *r, int idle) {
  struct ncclComm *const comm = r->comm;
  const uint32_t H = comm->count, tables = 2 * H * SETS;
  struct mesh_plan *table[tables + 1];
  for (uint32_t h = 0; h < tables; h++) table[h] = r->pages && r->tab && r->ltab ? slot_table(r, r->slots + h / (2 * H), h % H, (int)(h / H % 2)) : NULL;
  if (idle && r->pages) retire_locked(comm, r->pages, table, tables);
  else {
    pthread_mutex_lock(&attach_lock);
    for (uint32_t h = 0; comm->attach && h < tables; h++) if (table[h]) mesh_queue_unlink(&comm->attach->context, table[h]);
    pthread_mutex_unlock(&attach_lock);
  }
  for (uint32_t i = 0; i < SETS; i++) {
    struct slot *s = r->slots + i;
    if (s->unbound) { s->unbound = 0; comm->unpublished--; }
    if (s->command) metal_release(s->command);
    rows_free(&s->rows, idle);
  }
  request_parts_free(r);
}

static void sweep(struct ncclComm *comm) {
  for (struct plan *p = comm->plans, *next; p; p = next) { next = p->next; plan_free(comm, p); }
  comm->plans = NULL;
  for (struct ncclMeshRequest *r = comm->freed, *next; r; r = next) { next = r->next; request_end(r, !request_held(r, 0)); }
  comm->freed = NULL;
}

static void release(struct ncclComm *comm) {
  for (struct ncclMeshRequest **at = &comm->freed; *at;) {
    struct ncclMeshRequest *r = *at;
    if (request_held(r, 1)) { at = &r->next; continue; }
    *at = r->next;
    request_end(r, 1);
  }
}

ncclResult_t ncclMeshRequestFree(ncclMeshRequest_t r) {
  if (!r) return ncclSuccess;
  pthread_mutex_lock(&group_lock);
  const uint32_t H = r->comm->count;
  for (uint32_t i = 0; i < SETS; i++) {
    struct slot *s = r->slots + i;
    for (uint32_t h = 0; s->landing > r->starts && h < H; h++) {
      struct mesh_plan *t = slot_table(r, s, h, 1);
      for (uint32_t e = 0; t && s->kl[h] && e < t->lands; e++) mesh_stamp_lost(&t->entry[e].done, s->kl[h]);
      if (t && s->kl[h]) mesh_stamp_lost(&t->landed, s->kl[h]);
    }
  }
  r->next = r->comm->freed;
  r->comm->freed = r;
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

static int32_t leg_in(const struct ncclMeshRequest *r, uint32_t x, uint32_t piece) {
  for (uint32_t k = 0; k < r->ins; k++)
    if (r->in[k].x == x && r->in[k].piece == piece) return (int32_t)k;
  return -1;
}

static uint64_t part_bytes(const struct ncclMeshRequest *r, const size_t *counts, uint32_t x, uint32_t p) {
  const uint64_t first = r->cut[p], length = r->cut[r->pieces + p], count = counts ? counts[x] : r->parts[x];
  return (count <= first ? 0 : count - first < length ? count - first : length) * SIZE[r->type];
}
static uint32_t piece_pages(const struct ncclMeshRequest *r, const size_t *counts, uint32_t x, uint32_t p) {
  return pages_count(pages_of(part_bytes(r, counts, x, p), r->comm->page));
}

static void landings_post(struct ncclMeshRequest *r, struct slot *s, uint64_t m) {
  struct ncclComm *const comm = r->comm;
  const size_t z = SIZE[r->type];
  const uint32_t me = (uint32_t)comm->rank;
  s->landing = m; s->landing_k = ++comm->attach->calls;
  for (uint32_t h = 0; h < comm->count; h++) s->kl[h] = r->ltab[h] ? s->landing_k : 0;
  for (uint32_t e = 0; e < r->ins; e++) {
    const struct leg *l = r->in + e;
    const uint32_t p = r->first[l->x] + l->piece;
    struct mesh_entry *x = slot_entry(r, s, l->channel, 1, l->index);
    x->row = s->row + (r->area[l->x] + r->cut[p] * z) / comm->page; x->pages = piece_pages(r, NULL, l->x, p); x->word = 0;
    x->offset = 0; x->bytes = part_bytes(r, NULL, l->x, p);
    x->need = MESH_NEED_AHEAD;
    key_of(x, comm->id, me, (uint32_t)comm->channels[l->channel].peer, r->seq, m * r->lands[l->channel] + l->index);
  }
  for (uint32_t h = 0; h < comm->count; h++)
    if (r->ltab[h]) table_begin(comm, slot_table(r, s, h, 1), s->kl[h]);
}

static void landing_settle(struct mesh_plan *t, struct mesh_entry *e, uint64_t k) {
  uint64_t done = atomic_load_explicit(&e->done, memory_order_seq_cst);
  if (done == MESH_STAMP_LOST || done >= k || !atomic_compare_exchange_strong_explicit(&e->done, &done, k, memory_order_seq_cst, memory_order_seq_cst)) return;
  if (mesh_called(atomic_fetch_add_explicit(&t->completed[1], 1, memory_order_seq_cst) + 1) != t->lands) return;
  uint64_t was = atomic_load_explicit(&t->landed, memory_order_seq_cst);
  if (was != MESH_STAMP_LOST && was < k) atomic_compare_exchange_strong_explicit(&t->landed, &was, k, memory_order_seq_cst, memory_order_seq_cst);
}

static void landings_counted(struct ncclMeshRequest *r, struct slot *s, const size_t *counts) {
  for (uint32_t e = 0; e < r->ins; e++) {
    const struct leg *l = r->in + e;
    struct mesh_entry *x = slot_entry(r, s, l->channel, 1, l->index);
    const uint32_t pages = piece_pages(r, counts, l->x, r->first[l->x] + l->piece);
    uint64_t ahead = MESH_NEED_AHEAD;
    if (__atomic_compare_exchange_n(&x->need, &ahead, (uint64_t)pages + 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) && !pages)
      landing_settle(slot_table(r, s, l->channel, 1), x, s->kl[l->channel]);
  }
}

static void sends_write(struct ncclMeshRequest *r, struct slot *s, uint64_t m, const size_t *counts) {
  struct ncclComm *const comm = r->comm;
  const size_t z = SIZE[r->type];
  const uint32_t me = (uint32_t)comm->rank;
  struct mesh_table *t = s->table;
  struct mesh_relay *relay = (struct mesh_relay *)((struct mesh_wait *)((struct mesh_peer *)(t + 1) + r->peers) + r->waits);
  uint64_t ready = 0;
  comm->unpublished -= s->unbound;
  s->unbound = 0; s->rows.at = s->rows.used = 0;
  for (uint32_t h = 0; h < comm->count; h++) s->k[h] = r->tab[h] ? ++comm->attach->calls : 0;
  for (uint32_t e = 0, j = 0; e < r->outs; e++) {
    const struct leg *l = r->out + e;
    const uint32_t p = r->first[l->x] + l->piece, pages = piece_pages(r, counts, l->x, p);
    uint64_t *word = &t->published;
    if (l->from >= 0) {
      const struct leg *x = r->in + l->from;
      relay[j] = (struct mesh_relay){slot_address(r, s, &slot_entry(r, s, x->channel, 1, x->index)->done), s->kl[x->channel], 0, m};
      word = &relay[j++].published;
    } else if (pages) ready = m;
    struct mesh_entry *y = slot_entry(r, s, l->channel, 0, l->index);
    y->row = s->row + ((l->from < 0 ? 0 : r->area[l->x]) + r->cut[p] * z) / comm->page; y->pages = pages;
    y->offset = 0; y->bytes = part_bytes(r, counts, l->x, p);
    y->word = region_offset(comm, word); y->need = m;
    key_of(y, comm->id, (uint32_t)comm->channels[l->channel].peer, me, r->seq, m * r->sends[l->channel] + l->index);
  }
  t->ready = ready;
  for (uint32_t h = 0; h < comm->count; h++)
    if (r->tab[h]) table_begin(comm, slot_table(r, s, h, 0), s->k[h]);
  s->start = m;
  s->unbound = ready != 0 || r->relays != 0;
  comm->unpublished += s->unbound;
}

ncclResult_t ncclMeshExchangeInit(const size_t *parts, const int *lands, ncclDataType_t datatype, ncclComm_t comm, ncclMeshRequest_t *request) {
  if (!comm || !parts || !lands || !request || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "ncclMeshExchangeInit's arguments");
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank, H = comm->count, words = (n + 63) / 64;
  const size_t z = SIZE[datatype];
  struct mesh_demand *d = calloc((size_t)n + 1, sizeof *d);
  uint64_t *leaves = calloc((size_t)n * words + 1, sizeof *leaves);
  uint32_t *source = calloc((size_t)n + 1, sizeof *source), demands = 0;
  struct flow f = {0, NULL, NULL, NULL};
  struct ncclMeshRequest *r = calloc(1, sizeof *r);
  ncclResult_t result = d && leaves && source && r ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (uint32_t x = 0; !result && x < n; x++) {
    uint64_t *set = leaves + (size_t)demands * words;
    for (uint32_t b = 0; parts[x] && b < n; b++)
      if (b != x && lands[(size_t)x * n + b]) set[b / 64] |= UINT64_C(1) << (b % 64);
    int any = 0;
    for (uint32_t w = 0; w < words; w++) any |= set[w] != 0;
    if (!any) continue;
    source[demands] = x;
    d[demands++] = (struct mesh_demand){x, 0, 0, parts[x] * z, set};
  }
  if (!result) {
    pthread_mutex_lock(&group_lock);
    result = flow_solve(comm, d, demands, &f);
    pthread_mutex_unlock(&group_lock);
  }
  if (r) { r->comm = comm; r->type = datatype; }
  if (!result && (!(r->parts = malloc((size_t)n * sizeof *r->parts)) || !(r->area = malloc((size_t)n * sizeof *r->area)) ||
                  !(r->first = calloc((size_t)n + 1, sizeof *r->first)) || !(r->count = calloc((size_t)n + 1, sizeof *r->count)) ||
                  !(r->share = calloc((size_t)f.count + 1, sizeof *r->share)) || !(r->own = calloc((size_t)n + 1, sizeof *r->own)) ||
                  !(r->in = calloc((size_t)f.count + 1, sizeof *r->in)) || !(r->out = calloc((size_t)f.count * H + 1, sizeof *r->out)) ||
                  !(r->awaited = calloc((size_t)f.count + 1, sizeof *r->awaited))))
    result = fail(ncclSystemError, "out of memory");
  uint32_t *tree = result ? NULL : calloc((size_t)f.count * n + 1, sizeof *tree);
  if (!result && !tree) result = fail(ncclSystemError, "out of memory");
  for (uint32_t x = 0; !result && x < n; x++) { r->parts[x] = parts[x]; r->area[x] = SIZE_MAX; }
  for (uint32_t k = 0; !result && k < demands; k++) {
    const uint32_t x = source[k];
    r->first[x] = r->pieces;
    for (uint32_t t = 0; t < f.count; t++)
      if (f.demand[t] == k) {
        memcpy(tree + (size_t)r->pieces * n, f.parent + (size_t)t * n, n * sizeof *tree);
        r->share[r->pieces++] = f.share[t];
      }
    r->count[x] = r->pieces - r->first[x];
  }
  if (!result) r->bytes = page_up(parts[me] * z, comm->page);
  for (uint32_t x = 0; !result && x < n; x++)
    for (uint32_t j = 0; x != me && j < r->count[x]; j++)
      if (tree[(size_t)(r->first[x] + j) * n + me] != UINT32_MAX && r->area[x] == SIZE_MAX) {
        r->area[x] = r->bytes;
        r->bytes += page_up(parts[x] * z, comm->page);
      }
  for (uint32_t h = 0; !result && h < H; h++)
    for (uint32_t x = 0; x < n; x++)
      for (uint32_t j = 0; x != me && j < r->count[x]; j++)
        if (tree[(size_t)(r->first[x] + j) * n + me] == (uint32_t)comm->channels[h].peer &&
            (uint32_t)comm->toward[comm->channels[h].peer] == h)
          r->in[r->ins++] = (struct leg){x, j, h, 0, -1};
  for (uint32_t x = 0; !result && x < n; x++)
    if (x != me && lands[(size_t)x * n + me] && parts[x] && r->area[x] != SIZE_MAX) r->own[r->peers++] = x;
  for (uint32_t k = 0; !result && k < r->ins; k++)
    if (lands[(size_t)r->in[k].x * n + me]) r->awaited[r->waits++] = k;
  for (uint32_t h = 0; !result && h < H; h++) {
    const uint32_t c = (uint32_t)comm->channels[h].peer;
    for (uint32_t x = 0; x < n; x++)
      for (uint32_t j = 0; j < r->count[x]; j++) {
        if (c == x || tree[(size_t)(r->first[x] + j) * n + c] != me || (uint32_t)comm->toward[c] != h) continue;
        const int32_t from = x == me ? -1 : leg_in(r, x, j);
        r->out[r->outs++] = (struct leg){x, j, h, 0, from};
        r->relays += from >= 0;
      }
  }
  free(tree); free(d); free(leaves); free(source); flow_free(&f);
  if (result) { if (r) request_parts_free(r); return result; }
  if (!(r->tab = calloc((size_t)H + 1, sizeof *r->tab)) || !(r->ltab = calloc((size_t)H + 1, sizeof *r->ltab)) ||
      !(r->sends = calloc((size_t)H + 1, sizeof *r->sends)) || !(r->lands = calloc((size_t)H + 1, sizeof *r->lands)) ||
      !(r->cut = calloc(2 * (size_t)r->pieces + 1, sizeof *r->cut))) {
    request_parts_free(r);
    return fail(ncclSystemError, "out of memory");
  }
  for (uint32_t x = 0; x < n; x++)
    if (r->count[x])
      mesh_cut_shares(r->parts[x], (uint32_t)z, comm->page, r->share + r->first[x], r->count[x], r->cut + r->first[x], r->cut + r->pieces + r->first[x]);
  for (uint32_t k = 0; k < r->ins; k++) r->in[k].index = r->lands[r->in[k].channel]++;
  for (uint32_t k = 0; k < r->outs; k++) r->out[k].index = r->sends[r->out[k].channel]++;
  r->table = r->bytes;
  r->bytes += sizeof(struct mesh_table) + r->peers * sizeof(struct mesh_peer) + r->waits * sizeof(struct mesh_wait) + r->relays * sizeof(struct mesh_relay);
  for (uint32_t h = 0; h < 2 * H; h++) {
    const uint32_t legs = h < H ? r->sends[h] : r->lands[h - H];
    if (!legs) continue;
    r->bytes = (r->bytes + MESH_LINE - 1) / MESH_LINE * MESH_LINE;
    if (h < H) r->tab[h] = r->bytes; else r->ltab[h - H] = r->bytes;
    r->bytes += h < H ? mesh_plan_bytes(legs, 0) : mesh_plan_bytes(0, legs);
  }
  const size_t stride = page_up(r->bytes, comm->page);
  size_t offset = 0;
  ncclResult_t made = ncclSuccess;
  for (uint32_t i = 0; !made && i < SETS; i++)
    if (!(r->slots[i].k = calloc((size_t)H + 1, sizeof *r->slots[i].k)) || !(r->slots[i].kl = calloc((size_t)H + 1, sizeof *r->slots[i].kl)))
      made = fail(ncclSystemError, "out of memory");
  if (!made) made = lease_take(SETS * stride, &r->pages, &r->buffer, &offset);
  pthread_mutex_lock(&group_lock);
  r->seq = ++comm->exchanges;
  pthread_mutex_unlock(&group_lock);
  pthread_mutex_lock(&attach_lock);
  const uint64_t row = made ? 0 : lease_at(r->pages)->row;
  pthread_mutex_unlock(&attach_lock);
  for (uint32_t i = 0; !made && i < SETS; i++) {
    struct slot *s = r->slots + i;
    void *rows;
    size_t at;
    s->pages = (char *)r->pages + i * stride; s->offset = offset + i * stride; s->row = row + i * stride / comm->page;
    s->table = memset((char *)s->pages + r->table, 0, r->bytes - r->table);
    *s->table = (struct mesh_table){metal_address(r->buffer) + s->offset, 0, r->peers, 0, r->relays, r->waits, 0, 0, 0};
    struct mesh_peer *peer = (struct mesh_peer *)(s->table + 1);
    for (uint32_t e = 0; e < r->peers; e++) peer[e] = (struct mesh_peer){metal_address(r->buffer) + s->offset + r->area[r->own[e]]};
    s->rows.comm = comm;
    if (metal_rows_take(&s->rows, 1, &rows, &at)) made = fail(ncclSystemError, "out of memory: an exchange's rows");
    s->rows.at = s->rows.used = 0;
    for (uint32_t h = 0; !made && h < 2 * H; h++) {
      struct mesh_plan *t = slot_table(r, s, h % H, (int)(h / H));
      if (!t) continue;
      t->sends = h < H ? r->sends[h] : 0; t->lands = h < H ? 0 : r->lands[h - H];
      pthread_mutex_lock(&attach_lock);
      mesh_queue_link(&comm->attach->context, t);
      pthread_mutex_unlock(&attach_lock);
    }
  }
  for (uint32_t i = 0; !made && i < SETS; i++) {
    struct slot *s = r->slots + i;
    struct mesh_wait *wait = (struct mesh_wait *)((struct mesh_peer *)(s->table + 1) + r->peers);
    for (uint32_t e = 0; e < r->waits; e++) {
      const struct leg *l = r->in + r->awaited[e];
      wait[e] = (struct mesh_wait){slot_address(r, s, &slot_entry(r, s, l->channel, 1, l->index)->done), 0};
    }
  }
  if (made) { request_end(r, 1); return made; }
  *request = r;
  return ncclSuccess;
}

ncclResult_t ncclMeshStart(ncclMeshRequest_t r, const size_t *counts, cudaStream_t stream, uint64_t *start) {
  const ncclMeshStream *on = stream;
  if (!r || !start || !on || (!on->commandBuffer && !(on->argumentTable && on->commandEncoder && on->residencySet)))
    return fail(ncclInvalidArgument, "ncclMeshStart without its request, its start, or a command buffer or a Metal 4 encoder, argument table and residency set");
  struct ncclComm *const comm = r->comm;
  for (int q = 0; counts && q < comm->nranks; q++)
    if (counts[q] > r->parts[q]) return fail(ncclInvalidArgument, "ncclMeshStart: %zu elements of rank %d, past the %zu its exchange declared", counts[q], q, r->parts[q]);
  if (revoked(comm)) return revoked_fail(comm);
  const uint64_t number = r->starts + 1;
  uint64_t met = 0;
  pthread_mutex_lock(&group_lock);
  for (struct inflight *x = inflight; x; x = x->next_group)
    for (uint32_t h = 0; x->comm == comm && h < comm->count; h++)
      if (r->sends[h] && x->end[h][0] > x->start[h][0] && comm->channels[h].published < x->end[h][0]) met = x->ticket;
  pthread_mutex_unlock(&group_lock);
  ncclResult_t result = met ? ncclMeshComplete(met, on, NULL) : ncclSuccess;
  pthread_mutex_lock(&group_lock);
  release(comm);
  struct slot *s = r->slots + number % SETS, *ahead = r->slots + (number + 1) % SETS;
  if (!result && s->landing != number) landings_post(r, s, number);
  while (!result && slot_held(r, ahead, 1)) {
    const int status = kernels_held(ahead) && ahead->command ? metal_ended(ahead->command) : 1;
    if (revoked(comm)) result = revoked_fail(comm);
    else if (status == 2 || status < 0)
      result = fail(status < 0 ? ncclSystemError : ncclInvalidUsage, "ncclMeshStart %llu on %s: its pages are start %llu's, whose kernels' command buffer %s",
                    (unsigned long long)number, comm->name, (unsigned long long)(number - 3), status < 0 ? "ended in an error" : "is not committed");
    else {
      void *command = status ? NULL : metal_retain(ahead->command);
      pthread_mutex_unlock(&group_lock);
      if (command) { metal_completed(command, MESH_PAUSE_NS); metal_release(command); }
      else sched_yield();
      pthread_mutex_lock(&group_lock);
    }
  }
  if (!result) {
    landings_post(r, ahead, number + 1);
    sends_write(r, s, number, counts);
    landings_counted(r, s, counts);
    *start = r->starts = number;
  }
  pthread_mutex_unlock(&group_lock);
  return result;
}

ncclResult_t ncclMeshBind(ncclMeshRequest_t r, uint64_t start, int lands, int publishes, cudaStream_t stream, uint32_t index, uint32_t groups) {
  const ncclMeshStream *on = stream;
  if (!r || !on || !on->commandEncoder || (on->argumentTable && !on->residencySet) || !groups)
    return fail(ncclInvalidArgument, "ncclMeshBind without its request, a compute encoder (and a Metal 4 one's residency set) or its threadgroups");
  pthread_mutex_lock(&group_lock);
  struct slot *land = lands ? slot_of(r, start - 1) : NULL, *publish = publishes ? slot_of(r, start) : NULL, *held = publish ? publish : land;
  if (held && held->unbound) { held->unbound = 0; r->comm->unpublished--; }
  void *buffers[1] = {r->buffer};
  ncclResult_t result = (lands && !land) || (publishes && !publish)
                          ? fail(ncclInvalidUsage, "ncclMeshBind: start %llu, or the one before it, is not one of this exchange's held starts",
                                 (unsigned long long)start)
                          : gpu(metal_bind(on->commandEncoder, on->argumentTable, on->residencySet, held ? &held->rows : NULL,
                                           land ? metal_address(r->buffer) + land->offset + r->table : 0,
                                           publish ? metal_address(r->buffer) + publish->offset + r->table : 0, land ? land->landing_k : 0, groups, index,
                                           buffers, held != NULL),
                                "a crossing's binding");
  for (int i = 0; !result && i < 2; i++) {
    struct slot *s = i ? publish : land;
    if (!s) continue;
    s->bound++;
    if (on->commandBuffer && s->command != on->commandBuffer) {
      if (s->command) metal_release(s->command);
      s->command = metal_retain(on->commandBuffer);
    }
  }
  pthread_mutex_unlock(&group_lock);
  return result;
}

const char *ncclMeshMetalPrelude(void) { return metal_prelude(); }

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

static ncclResult_t launch_one(struct call *list, size_t n) {
  if (list[0].stream) return launch_metal(list, n);
  const ncclResult_t result = launch(list, n);
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

static void locate(struct call *made) {
  struct call c = *made;
  if (c.stream) {
    const ncclMeshStream *stream = c.stream;
    c.command = stream->commandBuffer; c.encoder = stream->commandEncoder;
    c.issue = stream->issue; c.table = stream->argumentTable; c.residency = stream->residencySet;
    c.send_at = where_of(c.send); c.recv_at = where_of(c.recv);
  } else {
    c.send_at = (struct where){NULL, (size_t)c.send}; c.recv_at = (struct where){NULL, (size_t)c.recv};
  }
  *made = c;
}

static ncclResult_t enqueue(struct call c) {
  locate(&c);
  ncclGroupStart();
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) { depth--; free(c.segments); free(c.matrix); free(c.pairs); return fail(ncclSystemError, "out of memory"); }
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

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_SEND, .peer = peer, .send = sendbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_RECV, .peer = peer, .recv = recvbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}

static ncclResult_t routed(const void *sendbuff, void *recvbuff, const size_t *counts, size_t count, int from, int to, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int n = comm->nranks, me = comm->rank, root = from < 0 ? to : from;
  if (root < -1 || root >= n) return fail(ncclInvalidArgument, "root %d of %d ranks", root, n);
  struct call c = {.what = WHAT_COUNTED, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm, .stream = stream,
                   .root = from, .peer = to, .total = count, .count = count};
  if (!(c.segments = malloc(2 * (size_t)n * sizeof *c.segments)) || (counts && !(c.matrix = malloc((size_t)n * n * sizeof *c.matrix)))) {
    free(c.segments);
    return fail(ncclSystemError, "out of memory");
  }
  for (size_t k = 0; counts && k < (size_t)n * n; k++) {
    c.matrix[k] = counts[k];
    if (k / (size_t)n != k % (size_t)n) c.count += counts[k];
  }
  for (int q = 0; q < n; q++) {
    c.segments[q] = hosted_count(&c, me, q);
    c.segments[n + q] = hosted_count(&c, q, me);
  }
  return enqueue(c);
}
ncclResult_t ncclMeshGroupPairs(const int *from, const int *to, const size_t *bytes, int count, ncclComm_t comm, cudaStream_t stream) {
  if (!comm || count < 0 || (count && (!from || !to || !bytes))) return fail(ncclInvalidArgument, "ncclMeshGroupPairs's arguments");
  struct call c = {.what = WHAT_COUNTED, .root = -1, .peer = -1, .type = ncclUint8, .op = ncclSum, .comm = comm, .stream = stream};
  if (!(c.segments = calloc(2 * (size_t)comm->nranks + 1, sizeof *c.segments)) || (count && !(c.pairs = malloc((size_t)count * sizeof *c.pairs)))) {
    free(c.segments);
    return fail(ncclSystemError, "out of memory");
  }
  for (int i = 0; i < count; i++) {
    c.pairs[i] = (struct pair){(uint32_t)from[i], (uint32_t)to[i], bytes[i]};
    if (from[i] != to[i]) c.count += bytes[i];
  }
  c.npairs = (uint32_t)count;
  return enqueue(c);
}

ncclResult_t ncclMeshAlltoAllV(const void *sendbuff, const size_t *counts, void *recvbuff, ncclDataType_t datatype, ncclComm_t comm,
                               cudaStream_t stream) {
  if (!counts) return fail(ncclInvalidArgument, "ncclMeshAlltoAllV: counts is NULL");
  return routed(sendbuff, recvbuff, counts, 0, -1, -1, datatype, comm, stream);
}
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, count, -1, -1, datatype, comm, stream);
}
ncclResult_t ncclAlltoAllConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}
ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, count, -1, root == -1 ? -2 : root, datatype, comm, stream);
}
ncclResult_t ncclGatherConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                              cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
}
ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, count, root == -1 ? -2 : root, -1, datatype, comm, stream);
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
  if (!result) {
    const int64_t colored = color;
    (*made)->id = named(fnv(fnv(fnv(UINT64_C(14695981039346656037), &comm->id, sizeof comm->id), &split, sizeof split), &colored, sizeof colored));
    if (count > 1) result = comm_register(*made);
  }
  if (!result) result = comm_channels(*made);
  if (result && *made) { comm_free(*made); *made = NULL; }
  free(members);
  return result;
}

static ncclResult_t split(ncclComm_t comm, int color, int key, ncclComm_t *newcomm) {
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; } mine = {color, key}, *every = calloc((size_t)n, sizeof *every);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  const size_t stride = page_up(sizeof mine, comm->page ? comm->page : 1);
  unsigned char *rows = NULL;
  *newcomm = NULL;
  ncclResult_t result = !every || !ranks ? fail(ncclSystemError, "out of memory") : ncclMemAlloc((void **)&rows, (size_t)n * stride);
  if (!result) {
    memcpy(rows + (size_t)comm->rank * stride, &mine, sizeof mine);
    result = ncclAllGather(rows + (size_t)comm->rank * stride, rows, stride, ncclUint8, comm, NULL);
  }
  for (int r = 0; !result && r < n; r++) memcpy(every + r, rows + (size_t)r * stride, sizeof *every);
  for (int r = 0; !result && color != NCCL_SPLIT_NOCOLOR && r < n; r++)
    if (every[r].color == color) ranks[count++] = r;
  for (int a = 1; a < count; a++)
    for (int b = a; b > 0 && every[ranks[b]].key < every[ranks[b - 1]].key; b--) { const int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
  if (!result && count) result = comm_sub(comm, ranks, count, split, color, newcomm);
  if (rows) ncclMemFree(rows);
  free(every); free(ranks);
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

/* libnccl-mesh (nccl.h): NCCL's API on the lanes of its node's bridge (design/transport.md section 14).  The process is
   one client of its node's bridge (an attach, whose slot owns the process's leases of the window, ncclMemAlloc, and
   asks the bridge for its communicators' lane pairs).  A communicator's init binds, on the cable toward each other
   member, a lane pair its members chose alike (mesh-call.h mesh_lanes_choose, mesh_lanes_bind) and exchanges each
   lane's first unit through NCCL's bootstrap; its channels are those lanes and its map the cables its members' reports
   name.  A group's calls are planned by the one collective planner, laid on the channels as runs of the lanes' units,
   declared to the bridge, supplied and awaited, the receives combined on the host or the GPU. */
#include "nccl.h"
#include "mesh-collective.h"
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


enum { WHAT_SEND = 16, WHAT_RECV, WHAT_COUNTED };
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN, COMBINE_MERGE };
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

/* the partials merge (nccl.h ncclMeshRedOpCreatePartials) of `rows` rows of `unit` float32 [m, l, O], y into x: a row
   with l > 0 rescaled to the greater m and summed, one with none counting for none, two such the row [PARTIAL_NONE, 0,
   0] (nccl-mesh-metal.m merge_row) */
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

/* dst[i] = dst[i] combine src[i]; the partials merge's n rows of `combine` floats where t is METAL_MERGE */
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

/* dst[i] = src[i] * scalar in the type (FuncPreMulSum, src/device/reduce_kernel.h); dst may be src */
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

/* an operator made on the communicator: PreMulSum's scalar (`value`), or the partials merge's row of `unit` elements */
struct redop { int used; ncclDataType_t type; unsigned char value[8]; uint32_t unit; };
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; };

/* a channel (design/transport.md section 14): toward rank `peer` on node `node`, of class `large` (mesh-plan.h
   mesh_wire_large), on the lane of that class of the pair its communicator's init bound on the cable toward that node
   (port `port`, lane `lane`; mesh-call.h mesh_lane_view: its rings and words), and the Metal buffer over them
   (`buffer`: its in ring, out ring, ready words and landed words at `in`, `out`, `ready` and `landed` in it).  Its
   positions: `sent` published, `consumed` consumed, `end` laid by the groups scheduled, `waited` awaited by the GPU work
   encoded; its units: `laid` the first of the next out run (from this end's start), `taken` the first of the next in
   run (from the peer's start: the bridge sends a lane's units in order), `declared` those declared to the bridge
   (mesh_lane_declare) */
struct channel {
  int peer, large;
  uint32_t node, port, lane;
  struct mesh_lane_view view;
  void *buffer;
  size_t in, out, ready, landed;
  uint64_t sent, consumed, end, waited, laid, taken, declared;
};

/* a communicator: rank r on the bridge of node members[r]; its map (map_of: its members joined by the cables their
   reports name, the routing table mesh_routes'), made at its init alike by every member and kept; the wire its lanes
   make (mesh-plan.h mesh_wire_of at its bridge's block); its message paths, the caller's (count 0: none), else the map's
   shortest-path trees (`trees`); its table of programs (`entries`), the caller's (`listed`, kept for its life), else the
   structural one compiled along its paths or trees where a call or a query first needs it (`compiled`: comm_compile),
   its configured forms first (`forms`, ncclMeshCommForms, which its splits and shrinks keep), fixed once a group of it
   crossed a link (`crossed`); its name, and the count of splits and shrinks made of it, which name theirs; the
   process's attach it holds (`attach`, until ncclCommDestroy); its own lane report and the pair it bound on each of its
   links (`bound`, `chosen`, UINT32_MAX none; `binding` while they are bound); its channels (`count`), two toward each
   member a cable joins, toward[r] the first of rank r's (-1: none); its failure words (`words`, nccl-mesh-metal.h: its
   GPU work's failure records and the gate every later dispatch reads, and the stop a revoke sets; `violation` the Metal
   buffer over them) and its failure once kept (`failed`, its text `failure`: comm_fail); `last`, the count of the last
   GPU program signalling `finished` that touched it (`touched`, `next_touched` while one is encoded: program_end), and
   `refreshed` its declarations' last refresh (ncclMeshComplete); `runs`, its host-path groups running; `revoked`, set by
   ncclCommRevoke and ncclCommDestroy, failing every group after it; `peers`, its bootstrap's sockets, kept for its life
   once its init succeeds (NCCL's bootstrapState), which a revoke shuts down and comm_free closes, under `guard`.  A
   non-blocking init (nccl.h ncclMeshCommInitRank) runs on `init` (`initing`: ncclCommDestroy joins it), `async`
   ncclInProgress until it ends and `why` its failure's text */
struct attach;
struct ncclComm {
  struct mesh_link_map map;
  struct mesh_wire wire;
  struct mesh_trees paths, trees;
  struct entry *entries;
  ncclMeshForm *forms;
  uint32_t programs, splits, *members, form_count, count;
  int nranks, rank, listed, compiled, crossed, binding, touched;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  struct redop ops[OPS];
  ncclResult_t async, failed;
  pthread_mutex_t guard;
  int revoked, *peers, initing, *toward;
  pthread_t init;
  char why[sizeof last], failure[sizeof last];
  void **scratch, **spent;
  size_t *scratch_bytes;
  uint32_t scratches, scratch_used, spents;
  struct attach *attach;
  struct mesh_lane_report bound;
  uint32_t chosen[MESH_PORTS];
  struct channel *channels;
  _Atomic uint64_t *words;
  void *violation;
  uint64_t last, refreshed;
  struct ncclComm *next_touched;
  _Atomic int runs;
};

/* the groups' one lock: a group's planning, scheduling and encoding, a completion, a communicator's failure kept and its
   end; `finished`, the event each GPU program the library made the encoder of signals at its count (`issued`) */
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

/* trees' own copy into `made` (its arrays allocated even for none); 0, or ENOMEM */
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

/* the trees and the programs compiled on the map; with `all`, the map, the caller's paths and programs too */
static void routes_free(struct ncclComm *comm, int all) {
  trees_free(&comm->trees);
  comm->compiled = 0;
  if (all) { trees_free(&comm->paths); map_free(&comm->map); }
  if (comm->listed && !all) return;
  for (uint32_t e = 0; e < comm->programs; e++) program_free(&comm->entries[e].program);
  free(comm->entries);
  comm->entries = NULL; comm->programs = 0;
}

/* comm's bootstrap's sockets closed (NCCL's bootstrapClose), under its guard */
static void bootstrap_close(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) close(comm->peers[r]);
  free(comm->peers);
  comm->peers = NULL;
  pthread_mutex_unlock(&comm->guard);
}

static void comm_leave(struct ncclComm *comm);
static void comm_free(struct ncclComm *comm) {
  bootstrap_close(comm);
  comm_leave(comm);
  routes_free(comm, 1);
  for (uint32_t k = 0; k < comm->scratches; k++) metal_release(comm->scratch[k]);
  for (uint32_t k = 0; k < comm->spents; k++) metal_release(comm->spent[k]);
  for (uint32_t h = 0; h < comm->count; h++) metal_release(comm->channels[h].buffer);
  if (comm->violation) metal_release(comm->violation);
  free(comm->scratch); free(comm->scratch_bytes); free(comm->spent); free(comm->channels); free(comm->toward);
  pthread_mutex_destroy(&comm->guard);
  free(comm->members); free(comm->forms);
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

/* `what` (at `root`) along `trees` (mesh-plan.h mesh_compile), each tree's part cut into `chunks` (1: uncut, the wire
   cutting each step's piece into slots of its class, mesh_wire_pieces), an all-reduce single-phase where `whole`,
   compiled into the table's next entry, for calls of fewer bytes than `below` (0: any size); 0, or EINVAL */
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

/* a configured form's entries (ncclMeshCommForms): its collective along the communicator's trees (`along`) or the map's
   packed trees (mesh_trees_pack at the form's eps and most, made here: mesh cc8a6f8e^ comm_routes' sets[1]; a segment
   cut among a root's trees by the form's shares, else equally, never by the flows the packing infers from the map's
   links), whole or two-phase, at its chunks, below its bound; a rooted one at its root, or at each root (-1), none
   where its root is past the communicator's ranks (a split's) */
static ncclResult_t compile_form(struct ncclComm *comm, const struct mesh_trees *along, const ncclMeshForm *x) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_trees packed = {n, 0, NULL, NULL, NULL};
  const struct mesh_trees *trees = along;
  ncclResult_t result = ncclSuccess;
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
    if (compile_entry(comm, trees, (uint32_t)x->collective, root, x->whole, x->chunks, x->below))
      result = fail(ncclInternalError, "the configured form of collective %d at root %u does not compile", x->collective, root);
  }
  trees_free(&packed);
  return result;
}

/* the communicator's structural table on its map, where the caller listed no programs: its configured forms first, in
   their order (compile_form), then, along the caller's paths (checked against the map) or else the map's shortest-path
   trees (mesh_trees_shortest: each member's in-tree of the routes over the cables between members), the single-phase
   all-reduce (each member's sum up its own tree; the direct exchange where every two members are linked), the
   reduce-scatter and the all-gather (segment s up and down s's trees), and each root's reduce and broadcast, uncut and
   of any size (mesh_compile; mesh 2bf5a50d^ comm_routes's table, its entries along the routes): a call takes the first
   entry that fits it (program_of), so a collective no form names, or a size past every form's bound, runs the default.
   A map whose cables do not join its members fails it, naming two: the bridges forward nothing, so no route crosses a
   node outside the communicator */
static ncclResult_t comm_routes(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  if (comm->listed || n < 2) return ncclSuccess;
  uint32_t apart[2] = {0, 0};
  const int joined = mesh_link_map_joined(&comm->map, apart);
  if (joined == ENOTCONN)
    return fail(ncclInvalidUsage, "rank %u (node %u) and rank %u (node %u) of %s are joined by no path of cables between its members: the "
                "bridges forward nothing, so a route through a node outside the communicator is refused", apart[0], comm->members[apart[0]],
                apart[1], comm->members[apart[1]], comm->name);
  if (joined) return fail(ncclInternalError, "the members' map of %s: %s", comm->name, strerror(joined));
  const struct mesh_trees *along = &comm->paths;
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
  size_t room = 3 + 2 * (size_t)n;
  for (uint32_t f = 0; f < comm->form_count; f++) room += has_root(comm->forms[f].collective) && comm->forms[f].root < 0 ? n : 1;
  if (!(comm->entries = calloc(room, sizeof *comm->entries))) return fail(ncclSystemError, "out of memory");
  for (uint32_t f = 0; f < comm->form_count; f++) {
    const ncclResult_t result = compile_form(comm, along, comm->forms + f);
    if (result) return result;
  }
  if (compile_entry(comm, along, MESH_ALLREDUCE, 0, 1, 1, 0) || compile_entry(comm, along, MESH_REDUCE_SCATTER, 0, 0, 1, 0) ||
      compile_entry(comm, along, MESH_ALLGATHER, 0, 0, 1, 0))
    return fail(ncclInternalError, "the all-reduce, reduce-scatter or all-gather along the communicator's trees does not compile");
  for (uint32_t r = 0; r < n; r++)
    if (compile_entry(comm, along, MESH_REDUCE, r, 0, 1, 0) || compile_entry(comm, along, MESH_BROADCAST, r, 0, 1, 0))
      return fail(ncclInternalError, "the reduce or broadcast of root %u along the communicator's trees does not compile", r);
  return ncclSuccess;
}

/* the communicator's table made where it is not (comm_routes), each move of each program between two members a cable
   joins (the caller's programs too: the bridges forward nothing, so a route through another node is refused as a
   value), and the checker's thread on it (comm_advise); under group_lock */
static void comm_advise(const struct ncclComm *comm);
static ncclResult_t comm_compile(struct ncclComm *comm) {
  if (comm->compiled) return ncclSuccess;
  ncclResult_t result = comm_routes(comm);
  for (uint32_t e = 0; !result && e < comm->programs; e++) {
    const struct mesh_program *x = &comm->entries[e].program;
    for (uint32_t r = 0; !result && r < x->nodes; r++)
      for (uint32_t m = x->first[r]; !result && m < x->first[r + 1]; m++)
        if (!mesh_link_between(&comm->map, r, x->move[m].peer))
          result = fail(ncclInvalidUsage, "program %u of %s moves between ranks %u and %u, which no cable joins: the bridges forward nothing, "
                        "so a route through another node is refused", e, comm->name, r, x->move[m].peer);
  }
  if (result) { routes_free(comm, 0); return result; }
  comm->compiled = 1;
  comm_advise(comm);
  return ncclSuccess;
}

/* forms copied with their shares (each `nranks` roots' most) into one allocation of the communicator's */
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

/* nccl.h ncclMeshCommForms: the forms checked and copied, taken at the communicator's next compile of its table (one
   compiled before, by ncclMeshFusableRooted, dropped); refused once a group of it crossed a link (its members compile
   its table alike from them), or where its table is the caller's programs */
ncclResult_t ncclMeshCommForms(ncclComm_t comm, const ncclMeshForm *forms, int count) {
  if (!comm || count < 0 || (count && !forms)) return fail(ncclInvalidArgument, "ncclMeshCommForms's arguments");
  for (int f = 0; f < count; f++) {
    const ncclMeshForm *x = forms + f;
    int shared = !x->shares || x->trees == ncclMeshTreesPacked;
    for (size_t k = 0; x->shares && shared && k < (size_t)comm->nranks * x->most; k++) shared = x->shares[k] > 0 && isfinite(x->shares[k]);
    if (x->collective < MESH_ALLREDUCE || x->collective > MESH_ALLGATHER || (has_root(x->collective) && x->root < -1) || !x->chunks || !shared ||
        (x->trees != ncclMeshTreesShortest && x->trees != ncclMeshTreesPacked) || (x->trees == ncclMeshTreesPacked && (!x->most || !(x->eps > 0))))
      return fail(ncclInvalidArgument, "form %d: a collective (mesh-plan.h MESH_*), a rooted one's root (-1 every root), its trees "
                  "(shortest, or packed with eps > 0, most > 0 and shares above 0 or none) and chunks > 0", f);
  }
  ncclMeshForm *made = count ? forms_copy(forms, (uint32_t)count, (uint32_t)comm->nranks) : NULL;
  if (count && !made) return fail(ncclSystemError, "out of memory");
  pthread_mutex_lock(&group_lock);
  ncclResult_t result = ncclSuccess;
  if (comm->listed) result = fail(ncclInvalidUsage, "communicator %s runs the caller's programs (ncclMeshCommInitRank): they are its table", comm->name);
  else if (comm->crossed)
    result = fail(ncclInvalidUsage, "communicator %s: its forms are set before its first group that crosses a link (its members compile its "
                  "table alike from them)", comm->name);
  else {
    free(comm->forms);
    comm->forms = made; comm->form_count = (uint32_t)count; made = NULL;
    routes_free(comm, 0);
  }
  pthread_mutex_unlock(&group_lock);
  free(made);
  return result;
}

/* the program a call of `bytes` runs: its communicator's table's first entry of its collective (and root, a rooted
   one's) whose bound it is below (a configured form's, else the default's); NULL where none is (mesh 2bf5a50d^
   program_of; cc8a6f8e^'s by size class, the classes now the forms' bounds) */
static const struct mesh_program *program_of(ncclComm_t comm, int what, int root, uint64_t bytes) {
  for (uint32_t e = 0; e < comm->programs; e++) {
    const struct entry *x = comm->entries + e;
    if ((int)x->program.what != what || ((what == MESH_REDUCE || what == MESH_BROADCAST) && (int)x->program.root != root)) continue;
    if (!x->below || bytes < x->below) return &x->program;
  }
  return NULL;
}

/* -- the process's client of its node's bridge (design/transport.md):
   one attach of the node's region (mesh_attach), made at the process's first lease or first communicator, whichever
   comes first.  Its slot owns every lease the process takes (ncclMemAlloc: a section of the window, mesh-call.h
   mesh_section_create, whose rows' owner is the slot), and each communicator binds its lane pairs through it (mesh-call.h
   mesh_lanes_bind: the slot's binding entries, one a pair of a port, `held` the pairs its communicators hold).  The leases'
   bookkeeping is the region's, where the bridge sees it: the bridge retires the slot of a process gone and returns its
   leases' pages as it returns every retired client's rows; `leases` is this process's index of them by address.  A
   lease is an operand of the window it was taken from, and a bridge that stops takes its region with it: while the
   process holds a lease of a bridge that stopped (no longer registered on any device), its next lease, communicator and
   ncclCommRegister fail naming it (ESTALE, attach_fail), and ncclMeshMemLive says so, until the process has freed every
   such lease; the next attaches to the bridge that runs then, so a process never holds leases of two regions (a group of
   a communicator made on the bridge that stopped fails: comm_ready).
   The attach of a bridge gone stays mapped while a lease of it is held or a communicator holds it (until
   ncclCommDestroy), then is detached. --- */

/* a lease: its address, its first row, its bytes, the Metal buffer over it (lease_buffer, made at its first use; NULL
   until then), and the points its caller's GPU work on it declared (ncclMeshMemUse: an event, the value it reaches once
   that work is done, whether the work writes the lease), which a host-path group waits for (lease_wait) */
struct point { void *event; uint64_t value; int write; };
struct lease { void *address; uint32_t row; size_t bytes; void *buffer; struct point *points; size_t npoints; };
struct attach {
  struct mesh_ctx context;
  struct lease *leases;
  size_t count, room;
  uint64_t comms;
  uint32_t held[MESH_PORTS];
  struct attach *next;
};
/* the attaches, the newest (the one a lease or communicator takes while its bridge lives) first, under attach_lock */
static struct attach *attaches;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;

static int attach_live(const struct attach *a) { return !mesh_gone((pid_t)atomic_load(&a->context.M->bridge_pid)); }

/* every attach whose bridge is gone, with no lease held and no communicator holding it, detached */
static void attaches_prune(void) {
  for (struct attach **at = &attaches; *at;) {
    struct attach *a = *at;
    if (a->count || a->comms || attach_live(a)) { at = &a->next; continue; }
    *at = a->next;
    mesh_detach(&a->context);
    free(a->leases);
    free(a);
  }
}

/* ESTALE where the process holds a lease of a bridge that stopped, else 0 */
static int attaches_stale(void) {
  for (const struct attach *a = attaches; a; a = a->next)
    if (a->count && !attach_live(a)) return ESTALE;
  return 0;
}

/* the attach whose bridge lives, made where there is none: 0; ESTALE where a lease of a bridge that stopped is still
   held; or mesh_attach's refusal */
static int attach_current(struct attach **out) {
  attaches_prune();
  if (attaches && attach_live(attaches)) { *out = attaches; return 0; }
  const int stale = attaches_stale();
  if (stale) return stale;
  struct attach *a = calloc(1, sizeof *a);
  if (!a) return ENOMEM;
  const int status = mesh_attach(&a->context, NULL);
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

/* the process's client refused (attach_current's status) for what `format` names: a lease of a
   bridge that stopped still held (ESTALE), every slot of the bridge held (EBUSY), or no region of a bridge that runs */
static ncclResult_t attach_fail(int status, const char *format, ...) {
  char what[256];
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(what, sizeof what, format, arguments);
  va_end(arguments);
  if (status == ESTALE)
    return fail(ncclRemoteError, "%s: this node's bridge stopped while this process holds leases of its window, which are no longer "
                "registered memory; free every one (ncclMemFree) and lease again from the bridge that runs now", what);
  if (status == EBUSY) return fail(ncclSystemError, "%s: every client slot of this node's bridge is held (mesh.h MESH_CLIENTS)", what);
  return fail(status == ESRCH ? ncclRemoteError : ncclSystemError, "%s: this process attaches no region of this node's bridge "
              "(%s); the node's bridge must run, with a configuration whose window holds this client's footprint "
              "(etc/bridge/README.md)", what, strerror(status));
}

/* a communicator of `nranks` ranks named `name`, this process rank `rank`, holding nothing of its bridge's yet
   (comm_attach); NULL where out of memory */
static struct ncclComm *comm_new(int nranks, int rank, const char *name) {
  struct ncclComm *comm = calloc(1, sizeof *comm);
  if (comm && (!(comm->members = calloc((size_t)nranks, sizeof *comm->members)) || !(comm->toward = malloc((size_t)nranks * sizeof *comm->toward)))) {
    free(comm->members); free(comm); comm = NULL;
  }
  if (!comm) return NULL;
  comm->nranks = nranks; comm->rank = rank;
  for (int r = 0; r < nranks; r++) comm->toward[r] = -1;
  for (uint32_t k = 0; k < MESH_PORTS; k++) comm->chosen[k] = UINT32_MAX;
  pthread_mutex_init(&comm->guard, NULL);
  snprintf(comm->name, sizeof comm->name, "%s", name);
  return comm;
}
/* comm's rank r on the bridge of node members[r] (NULL: its init's gather fills them); the process's client attached
   where it has none and held; 0, or the attach's refusal (comm_refused), comm then holding nothing of the bridge's */
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

/* a communicator's lane pairs unbound where it bound them (mesh-call.h mesh_lanes_unbind: the bridge sends the units
   they declared, those no group supplied as duds, and frees them) and taken out of its attach's held pairs; under
   attach_lock */
static void comm_unbind(struct ncclComm *comm) {
  if (!comm->binding) return;
  comm->binding = 0;
  mesh_lanes_unbind(&comm->attach->context, &comm->bound, comm->chosen);
  for (uint32_t k = 0; k < comm->bound.count && k < MESH_PORTS; k++)
    if (comm->chosen[k] < MESH_LANE_PAIRS) comm->attach->held[comm->bound.link[k].port] &= ~(UINT32_C(1) << comm->chosen[k]);
}
/* a communicator's lanes unbound (comm_unbind) and its hold of its attach let go (detached where its bridge is gone and
   nothing else holds it) */
static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) {
    comm_unbind(comm);
    comm->attach->comms--;
    comm->attach = NULL;
    attaches_prune();
  }
  pthread_mutex_unlock(&attach_lock);
}

/* a rank's report at a communicator's init (NCCL's bootstrap, below): its status (0, else why it cannot go on, so every
   rank fails alike), its bridge's node and the lane pairs free on each of its links (lanes_read) and each link's
   bandwidth, and, at the second gather, the first unit of each lane of each pair it bound (start[k][0] link k's large
   lane's, start[k][1] its small lane's: the bridge's count of units sent on it, mesh-call.h mesh_lanes_bind) */
struct report { int32_t rank, status; struct mesh_lane_report lanes; uint64_t bandwidth[MESH_PORTS], start[MESH_PORTS][2]; };

/* the cables between ranks a and b as their reports name them (the fewer of the two ends' links toward each other), and
   *bandwidth the least any of those links reports at either end (0 where none reports one: mesh-plan.h mesh_map's rule) */
static uint32_t cables_between(const struct report *reports, uint32_t a, uint32_t b, uint64_t *bandwidth) {
  uint32_t ends[2] = {0, 0};
  *bandwidth = 0;
  for (int side = 0; side < 2; side++) {
    const struct report *x = reports + (side ? b : a);
    const uint32_t toward = reports[side ? a : b].lanes.node;
    for (uint32_t k = 0; k < x->lanes.count && k < MESH_PORTS; k++) {
      if (x->lanes.link[k].peer != toward) continue;
      ends[side]++;
      if (x->bandwidth[k] && (!*bandwidth || x->bandwidth[k] < *bandwidth)) *bandwidth = x->bandwidth[k];
    }
  }
  return ends[0] < ends[1] ? ends[0] : ends[1];
}

/* design/transport.md section 14: the communicator's map, made alike by every member from the same gathered reports:
   its ranks its nodes, each marked by its bridge's node number, two linked by a link a cable between them
   (cables_between), its routing table mesh_routes'.  No node outside the communicator is in it: the bridges forward
   nothing.  0, or ENOMEM, EINVAL */
static int map_of(struct ncclComm *comm, const struct report *reports) {
  const uint32_t n = (uint32_t)comm->nranks;
  uint32_t links = 0;
  uint64_t bandwidth;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++) links += cables_between(reports, a, b, &bandwidth);
  struct mesh_link_map *map = &comm->map;
  *map = (struct mesh_link_map){MESH_LINKS_GRAPH, n, links, calloc(links + 1, sizeof *map->link), calloc(links + 1, sizeof *map->bandwidth), 0,
                                calloc((size_t)n * n + 1, sizeof *map->next), calloc(n + 1, sizeof *map->mark)};
  if (!map->link || !map->bandwidth || !map->next || !map->mark) { map_free(map); return ENOMEM; }
  uint32_t l = 0;
  for (uint32_t a = 0; a < n; a++) {
    map->mark[a] = reports[a].lanes.node;
    for (uint32_t b = a + 1; b < n; b++)
      for (uint32_t c = cables_between(reports, a, b, &bandwidth); c; c--, l++) {
        map->link[l][0] = a; map->link[l][1] = b; map->bandwidth[l] = bandwidth;
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

const char *ncclGetLastError(ncclComm_t comm) { (void)comm; return last; }

/* NCCL's bootstrap (NVIDIA/nccl src/bootstrap.cc): the unique id is the root's address, host:port (NCCL_COMM_ID where
   set, else a listener this process opens), and the root listens on that host alone where it is an IPv6 address (the
   fabric's ULA; bootstrapCreateRoot listens on the id's address), else on every address; rank 0 takes every rank's
   report there and sends each the whole table, at each gather of the init, on sockets the communicator keeps once its
   init succeeds (NCCL's bootstrapState), closed where it fails.  A non-root's connect does not block: it is looked at
   every 100 ms and tried again 100 ms after it fails, and a non-blocking init's root looks for a ringer every 100 ms (a
   blocking root waits in its accept); each reads a revoke of the communicator (ncclCommRevoke, ncclCommAbort,
   ncclCommDestroy: a non-blocking init's) at each look (NCCL's socketPollConnect reads abortFlag so), and the revoke
   shuts the init's sockets down (`peers`), so its gathers end.  The connect and the accept hold nothing of the
   bridge's: a rank attaches once its bootstrap has connected, or accepted every rank (comm_init).  The first gather has
   no deadline, that revoke and the caller's process its cancellers (these sockets carry no DMA and enter no driver),
   and it holds no binding of the bridge's.  The second follows this rank's binding of its lane pairs, on which its
   peers' lanes wait, so each of its sends and receives is bounded by mesh.h MESH_PLATFORM_NS (ETIMEDOUT:
   design/transport.md).  A rank's connect gives up MESH_PLATFORM_NS after its first try (ETIMEDOUT), so an init toward
   a root that does not listen fails as a value rather than trying for good.  A rank's report is struct report.  The id
   is also the communicator's name where its init names none. */
static struct { int fd; char id[NCCL_UNIQUE_ID_BYTES]; } root = {-1, {0}};

/* a listener on `host` alone where it is an IPv6 address, else on every address, at *port (0: any, *port then the
   one bound), non-blocking (bootstrap_open looks before it accepts): its descriptor, or -1 and errno */
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

/* nccl.h ncclMeshFabricAddress: a cabled peer is reached on the fabric, never by a name that resolves to Wi-Fi or a LAN:
   the node's mesh identity, the first address under mesh.h MESH_ULA_PREFIX of an interface other than loopback
   (bin/mesh-router-init.sh puts one on each fabric port) */
ncclResult_t ncclMeshFabricAddress(char *address, size_t room) {
  if (!address || !room) return fail(ncclInvalidArgument, "ncclMeshFabricAddress: no room");
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
  if (short_room) return fail(ncclInvalidArgument, "ncclMeshFabricAddress: %zu bytes hold no address of the fabric's", room);
  return *address ? ncclSuccess
                  : fail(ncclSystemError, "this node holds no fabric address (no interface's address under %s: bin/mesh-router-init.sh)", MESH_ULA_PREFIX);
}

/* the id: NCCL_COMM_ID where set, else a listener on the node's fabric address (ncclMeshFabricAddress) where it holds
   one, else on its host name */
ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  memset(id->internal, 0, sizeof id->internal);
  const char *given = getenv("NCCL_COMM_ID");
  if (given && *given) { snprintf(id->internal, sizeof id->internal, "%s", given); return ncclSuccess; }
  char host[256] = {0};
  if (ncclMeshFabricAddress(host, sizeof host)) *host = 0;
  uint16_t port = 0;
  if (root.fd >= 0) close(root.fd);
  if ((root.fd = *host || !gethostname(host, sizeof host - 1) ? bootstrap_listen(host, &port) : -1) < 0)
    return fail(ncclSystemError, "the bootstrap root's listener: %s", strerror(errno));
  snprintf(id->internal, sizeof id->internal, "%s:%u", host, port);
  memcpy(root.id, id->internal, sizeof root.id);
  return ncclSuccess;
}

static int whole(int fd, void *bytes, size_t size, int sending) {
  for (size_t done = 0; done < size;) {
    const ssize_t moved = sending ? send(fd, (char *)bytes + done, size - done, 0) : recv(fd, (char *)bytes + done, size - done, 0);
    if (moved <= 0) return moved ? (errno == EAGAIN || errno == EWOULDBLOCK ? ETIMEDOUT : errno) : ECONNRESET;
    done += (size_t)moved;
  }
  return 0;
}

/* whether comm was revoked; a group on it failed naming it */
static int revoked(ncclComm_t comm) { return __atomic_load_n(&comm->revoked, __ATOMIC_SEQ_CST); }
static ncclResult_t revoked_fail(ncclComm_t comm) {
  return fail(ncclRemoteError, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
}

/* socket `fd` of comm's init placed at peers[r] (-1: the one there closed), blocking (an accepted one takes its
   listener's O_NONBLOCK), under comm's guard, where a revoke shuts it down (ncclCommRevoke), a send on it after that or
   a peer's reset an error, never SIGPIPE */
static void bootstrap_put(struct ncclComm *comm, int r, int fd) {
  if (fd >= 0) { fcntl(fd, F_SETFL, 0); setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &(int){1}, sizeof(int)); }
  pthread_mutex_lock(&comm->guard);
  if (fd < 0 && comm->peers[r] >= 0) close(comm->peers[r]);
  comm->peers[r] = fd;
  pthread_mutex_unlock(&comm->guard);
}

/* the bootstrap's sockets of comm's init at `id`, the root's to each rank r (peers[r], placed by the rank its first
   report names, that report kept in `reports`; a socket not yet placed waits at peers[0]) or a non-root's to the root
   (peers[0], placed once connected), each placed before the revoke is read, so a revoke either finds it or is read: 0,
   ECANCELED (comm revoked), or the errno */
static double now_s(void);
static int bootstrap_open(struct ncclComm *comm, const char *id, struct report *reports) {
  int *peers = comm->peers, status = 0;
  char host[NCCL_UNIQUE_ID_BYTES];
  snprintf(host, sizeof host, "%s", id);
  char *colon = strrchr(host, ':');
  if (!colon) return EINVAL;
  *colon = 0;
  if (comm->rank) {
    struct addrinfo hint = {.ai_socktype = SOCK_STREAM}, *addresses = NULL;
    const double until = now_s() + (double)MESH_PLATFORM_NS / 1e9;
    while (peers[0] < 0 && !revoked(comm) && now_s() < until) {
      if (!getaddrinfo(host, colon + 1, &hint, &addresses)) {
        for (struct addrinfo *a = addresses; a && peers[0] < 0 && !revoked(comm); a = a->ai_next) {
          const int fd = socket(a->ai_family, SOCK_STREAM, 0);
          if (fd < 0) continue;
          int error = fcntl(fd, F_SETFL, O_NONBLOCK) || connect(fd, a->ai_addr, a->ai_addrlen) ? errno : 0;
          struct pollfd look = {fd, POLLOUT, 0};
          socklen_t length = sizeof error;
          while (error == EINPROGRESS || error == EINTR)
            if (revoked(comm)) error = ECANCELED;
            else if (poll(&look, 1, 100) < 0) error = errno == EINTR ? EINPROGRESS : errno;
            else if (look.revents && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length)) error = errno;
          if (error) close(fd);
          else bootstrap_put(comm, 0, fd);
        }
        freeaddrinfo(addresses);
      }
      if (peers[0] < 0) usleep(100000);
    }
    return revoked(comm) ? ECANCELED : peers[0] < 0 ? ETIMEDOUT : 0;
  }
  uint16_t port = (uint16_t)atoi(colon + 1);
  int listener = !strcmp(root.id, id) ? root.fd : -1;
  if (listener < 0 && (listener = bootstrap_listen(host, &port)) < 0) return errno;
  for (int r = 1; r < comm->nranks && !status; r++) {
    struct report heard;
    struct pollfd look = {listener, POLLIN, 0};
    int fd = -1;
    while (fd < 0 && !status)
      if (revoked(comm)) status = ECANCELED;
      else if (poll(&look, 1, __atomic_load_n(&comm->async, __ATOMIC_ACQUIRE) == ncclInProgress ? 100 : -1) < 0 && errno != EINTR) status = errno;
      else if ((fd = accept(listener, NULL, NULL)) < 0 && errno != EWOULDBLOCK && errno != ECONNABORTED && errno != EINTR) status = errno;
    if (!status) bootstrap_put(comm, 0, fd);
    if (!status && revoked(comm)) status = ECANCELED;
    if (!status && !(status = whole(fd, &heard, sizeof heard, 0)) && (heard.rank <= 0 || heard.rank >= comm->nranks || peers[heard.rank] >= 0))
      status = EPROTO;
    if (status) { bootstrap_put(comm, 0, -1); break; }
    reports[heard.rank] = heard;
    pthread_mutex_lock(&comm->guard);
    peers[heard.rank] = fd; peers[0] = -1;
    pthread_mutex_unlock(&comm->guard);
  }
  close(listener);
  if (listener == root.fd) root.fd = -1;
  return status;
}

/* a gather on the bootstrap's sockets: each other rank's report to the root (`sent`: there already, as at the first
   gather's root, whose bootstrap_open read each), then the whole table to each; 0, or the errno */
static int gather(const int *peers, int nranks, int rank, struct report *reports, int sent) {
  const size_t table = (size_t)nranks * sizeof *reports;
  int status = 0;
  if (rank) {
    const struct report kept = reports[rank];
    if (!sent) status = whole(peers[0], reports + rank, sizeof *reports, 1);
    if (!status) status = whole(peers[0], reports, table, 0);
    reports[rank] = kept;
    return status;
  }
  for (int r = 1; r < nranks && !status && !sent; r++)
    if (!(status = whole(peers[r], reports + r, sizeof *reports, 0)) && reports[r].rank != r) status = EPROTO;
  for (int r = 1; r < nranks && !status; r++) status = whole(peers[r], reports, table, 1);
  return status;
}

/* design/transport.md: the second gather's sockets, each send and receive bounded by mesh.h
   MESH_PLATFORM_NS: 0, or the errno */
static int bootstrap_bound(const int *peers, int nranks, int rank) {
  const struct timeval bound = {(time_t)(MESH_PLATFORM_NS / 1000000000u), 0};
  for (int r = rank ? 0 : 1; r < (rank ? 1 : nranks); r++)
    if (setsockopt(peers[r], SOL_SOCKET, SO_RCVTIMEO, &bound, sizeof bound) || setsockopt(peers[r], SOL_SOCKET, SO_SNDTIMEO, &bound, sizeof bound))
      return errno;
  return 0;
}

/* this rank's report of its bridge's lanes (mesh-call.h mesh_lanes_free), less the pairs this process's communicators
   hold (the bridge publishes a pair bound only at its next turn, and a second binding of one entry of the slot would end
   the first), and each link's bandwidth; 0, or mesh_lanes_free's status; under attach_lock */
static int lanes_read(struct attach *a, struct report *report) {
  const int status = mesh_lanes_free(a->context.M, &report->lanes);
  for (uint32_t k = 0; !status && k < report->lanes.count && k < MESH_PORTS; k++) {
    struct mesh_lane_link *link = report->lanes.link + k;
    link->free &= ~a->held[link->port];
    report->bandwidth[k] = atomic_load_explicit(&mesh_links(a->context.M)[link->port].bandwidth, memory_order_relaxed);
  }
  return status;
}

/* the one rule every rank applies alike to the `n` gathered reports of its communicator (mesh-call.h
   mesh_lanes_choose), into `chosen` (a row a rank) by way of `lanes`: 0; EINVAL where two ranks are on one node (a
   bridge's lanes join nodes); ENOSPC where a link toward another rank's node has no pair free at both ends; EMLINK where
   two ranks' nodes are joined by more than one cable (no bridge publishes a link's far port, so the two ends of several
   cables could not take a position's units on one cable alike) */
static int comm_choose(const struct report *reports, uint32_t n, struct mesh_lane_report *lanes, uint32_t *chosen) {
  for (uint32_t r = 0; r < n; r++) lanes[r] = reports[r].lanes;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++)
      if (lanes[a].node == lanes[b].node) return EINVAL;
  const int status = mesh_lanes_choose(lanes, n, chosen);
  for (uint32_t r = 0; !status && r < n; r++)
    for (uint32_t k = 0; k < lanes[r].count && k < MESH_PORTS; k++)
      for (uint32_t j = k + 1; chosen[(size_t)r * MESH_PORTS + k] != UINT32_MAX && j < lanes[r].count && j < MESH_PORTS; j++)
        if (lanes[r].link[j].peer == lanes[r].link[k].peer && chosen[(size_t)r * MESH_PORTS + j] != UINT32_MAX) return EMLINK;
  return status;
}
static ncclResult_t choose_fail(const char *name, int status) {
  if (status == EINVAL) return fail(ncclInvalidUsage, "communicator %s: two of its ranks are on one node (a rank a node: a bridge's lanes join nodes)", name);
  if (status == EMLINK)
    return fail(ncclInvalidUsage, "communicator %s: two of its ranks' nodes are joined by more than one cable, and no bridge publishes a link's far "
                "port (mesh.h struct mesh_link_info), so the two ends cannot take a position on one cable alike", name);
  return fail(ncclSystemError, "communicator %s: a link toward another rank's node has no lane pair free at both ends (mesh-call.h "
              "mesh_lanes_choose: %s); a communicator made next reads the pairs as they then stand", name, strerror(status));
}
static ncclResult_t bind_fail(const char *name, int rank, int status) {
  return fail(ncclSystemError, "communicator %s: rank %d could not bind its lane pairs at its bridge (%s: EBUSY, a pair another client binds, or "
              "whose last binding's units are still going; ETIMEDOUT, no answer within the platform's deadline); a communicator made next "
              "reads the pairs as they then stand", name, rank, strerror(status));
}

/* a communicator's table, the caller's operands (nccl.h ncclMeshCommInitRank): its paths (NULL: none, the map's
   shortest-path trees) and its programs copied (none: the structural table, comm_routes); 0, or ENOMEM */
static int comm_table(struct ncclComm *made, const struct mesh_trees *paths, const ncclMeshProgram *programs, int count) {
  made->listed = count > 0;
  if ((paths && trees_copy(&made->paths, paths)) || (count && !(made->entries = calloc((size_t)count, sizeof *made->entries)))) return ENOMEM;
  for (int e = 0; e < count; e++) {
    if (program_copy(&made->entries[e].program, programs[e].program)) return ENOMEM;
    made->entries[made->programs++].below = programs[e].below;
  }
  return 0;
}

/* nccl.h ncclMeshInitWaits */
static _Atomic uint64_t init_waits;

/* this rank's row of `chosen` (its report's links, `mine`) bound at its bridge through the process's slot
   (mesh-call.h mesh_lanes_bind), each link's lanes' first units into start; its pairs held by its attach first, so no
   other communicator of the process chooses them (lanes_read); 0, EBUSY where one is held already, or the binding's
   status (nothing then bound) */
static int comm_bind(struct ncclComm *comm, const struct mesh_lane_report *mine, const uint32_t *chosen, uint64_t (*start)[2]) {
  const uint64_t began = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  struct attach *a = comm->attach;
  int status = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t k = 0; k < mine->count && k < MESH_PORTS; k++)
    if (chosen[k] < MESH_LANE_PAIRS && a->held[mine->link[k].port] >> chosen[k] & 1) status = EBUSY;
  const int held = !status;
  for (uint32_t k = 0; held && k < mine->count && k < MESH_PORTS; k++)
    if (chosen[k] < MESH_LANE_PAIRS) a->held[mine->link[k].port] |= UINT32_C(1) << chosen[k];
  comm->bound = *mine;
  memcpy(comm->chosen, chosen, sizeof comm->chosen);
  pthread_mutex_unlock(&attach_lock);
  if (held) status = mesh_lanes_bind(&a->context, mine, chosen, start);
  pthread_mutex_lock(&attach_lock);
  comm->binding = !status;
  for (uint32_t k = 0; held && status && k < mine->count && k < MESH_PORTS; k++)
    if (chosen[k] < MESH_LANE_PAIRS) a->held[mine->link[k].port] &= ~(UINT32_C(1) << chosen[k]);
  pthread_mutex_unlock(&attach_lock);
  atomic_fetch_add(&init_waits, clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - began);
  return status;
}

/* the Metal buffer over a channel's lane, its rings and words (the region's pages the bridge took for them, whole
   pages), and each one's offset in it */
static int channel_buffer(struct channel *ch) {
  const struct mesh_lane_view *v = &ch->view;
  const uintptr_t page = (uintptr_t)getpagesize(), ring = (uintptr_t)v->units * v->unit, words = (uintptr_t)v->units * sizeof(uint64_t);
  const uintptr_t at[4] = {(uintptr_t)v->in, (uintptr_t)v->out, (uintptr_t)v->ready, (uintptr_t)v->landed};
  const uintptr_t end[4] = {at[0] + ring, at[1] + ring, at[2] + words, at[3] + words};
  uintptr_t lo = at[0], hi = end[0];
  for (int k = 1; k < 4; k++) { lo = at[k] < lo ? at[k] : lo; hi = end[k] > hi ? end[k] : hi; }
  lo &= ~(page - 1);
  hi = (hi + page - 1) & ~(page - 1);
  ch->in = at[0] - lo; ch->out = at[1] - lo; ch->ready = at[2] - lo; ch->landed = at[3] - lo;
  return (ch->buffer = metal_wrap((void *)lo, hi - lo)) ? 0 : ENOMEM;
}

/* design/transport.md section 14: the communicator's channels on the lanes its init bound (comm_bind), from every
   rank's report and first units as the second gather gave them: toward each rank a cable of this node joins, a channel
   of each class on the lane of that class of the pair both ends bound on it (the same pair, by the one rule), its out
   units from this end's first and its in units from the peer's (the bridge sends a lane's units in order, so the peer's
   first out is this end's first in); its map (map_of), its wire (its bridge's block: every paired bridge's is one) and
   its failure words.  0, or ENOMEM; EPROTO where a lane is not the wire's or the two ends' pairs differ; mesh_lane_view's
   refusal */
static int comm_join(struct ncclComm *comm, const struct report *reports, const uint32_t *chosen) {
  const struct hdr *m = comm->attach->context.M;
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank, node = reports[me].lanes.node;
  mesh_wire_of((uint64_t)m->pgsz * m->block, &comm->wire);
  int status = map_of(comm, reports);
  if (!status && !(comm->violation = metal_shared((size_t)getpagesize()))) status = ENOMEM;
  if (!status) comm->words = metal_contents(comm->violation);
  if (!status && !(comm->channels = calloc(2 * (size_t)n, sizeof *comm->channels))) status = ENOMEM;
  for (uint32_t q = 0; !status && q < n; q++) {
    if (q == me) continue;
    const struct mesh_lane_report *peer = &reports[q].lanes;
    uint32_t k = 0, j = 0;
    while (k < comm->bound.count && (comm->bound.link[k].peer != peer->node || chosen[(size_t)me * MESH_PORTS + k] == UINT32_MAX)) k++;
    while (j < peer->count && (peer->link[j].peer != node || chosen[(size_t)q * MESH_PORTS + j] == UINT32_MAX)) j++;
    if (k >= comm->bound.count || j >= peer->count) continue;
    const uint32_t pair = chosen[(size_t)me * MESH_PORTS + k];
    if (pair != chosen[(size_t)q * MESH_PORTS + j]) { status = EPROTO; break; }
    comm->toward[q] = (int)comm->count;
    for (int large = 1; !status && large >= 0; large--) {
      struct channel *ch = comm->channels + comm->count++;
      *ch = (struct channel){.peer = (int)q, .large = large, .node = peer->node, .port = comm->bound.link[k].port,
                             .lane = MESH_LANE_FIRST + 2 * pair + !large, .laid = reports[me].start[k][!large], .taken = reports[q].start[j][!large]};
      ch->declared = ch->laid;
      status = mesh_lane_view(&comm->attach->context, ch->port, ch->lane, &ch->view);
      if (!status && (ch->view.unit != (large ? comm->wire.block : MESH_FRAME) ||
                      ch->view.units != (large ? MESH_LANE_LARGE_UNITS : MESH_LANE_SMALL_UNITS)))
        status = EPROTO;
      if (!status) status = channel_buffer(ch);
    }
  }
  return status;
}

/* design/transport.md section 14, a communicator's init on NCCL's bootstrap at its id, the one path of every init
   (ncclMeshCommInitRank, blocking or not): its bootstrap's sockets opened (bootstrap_open), then made attached
   (comm_attach) and its lanes read (lanes_read), then every rank's report gathered; every rank applying the one rule to
   them alike (comm_choose: two ranks on one node, a link with no pair free at both ends, or several cables between two
   ranks' nodes fail every rank alike) and binding its row at its own bridge (comm_bind); a second gather carrying each
   rank's binding's status and its lanes' first units (NCCL's own init exchanges each peer's connection through its
   bootstrap: ncclTransportP2pSetup), from which each rank makes its channels (comm_join).  A rank that cannot go on, a
   pair refused (EBUSY) or unanswered (ETIMEDOUT), a peer silent past the second gather's bound (bootstrap_bound) or a
   revoke fails the init as a value, its pairs unbound, made holding nothing of its bridge's (comm_leave) and its
   sockets closed; the next init reads the pairs as they then stand.  Made, it keeps its sockets (closed at comm_free) */
static ncclResult_t comm_init(struct ncclComm *made) {
  const int nranks = made->nranks, rank = made->rank;
  struct report *reports = calloc((size_t)nranks, sizeof *reports);
  struct mesh_lane_report *lanes = calloc((size_t)nranks, sizeof *lanes);
  uint32_t *chosen = calloc((size_t)nranks * MESH_PORTS, sizeof *chosen);
  int *peers = calloc((size_t)nranks, sizeof *peers);
  if (!reports || !lanes || !chosen || !peers) { free(reports); free(lanes); free(chosen); free(peers); return fail(ncclSystemError, "out of memory"); }
  for (int r = 0; r < nranks; r++) peers[r] = -1;
  pthread_mutex_lock(&made->guard);
  made->peers = peers;
  pthread_mutex_unlock(&made->guard);
  struct report *mine = reports + rank;
  int status = nranks > 1 ? bootstrap_open(made, made->id, reports) : 0;
  const int making = status ? status : comm_attach(made, NULL);
  *mine = (struct report){.rank = rank, .status = making};
  if (!making) {
    pthread_mutex_lock(&attach_lock);
    mine->status = lanes_read(made->attach, mine);
    pthread_mutex_unlock(&attach_lock);
  }
  if (!status && nranks > 1) status = gather(peers, nranks, rank, reports, !rank);
  ncclResult_t result = status ? fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status)) : making ? comm_refused(making, made->name) : ncclSuccess;
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status)
      result = r == rank ? fail(ncclSystemError, "communicator %s: the lane pairs free at this node's bridge (mesh_lanes_free): %s", made->name,
                                strerror(reports[r].status))
                         : fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  for (int r = 0; !result && r < nranks; r++) made->members[r] = reports[r].lanes.node;
  if (!result && nranks > 1 && (status = comm_choose(reports, (uint32_t)nranks, lanes, chosen))) result = choose_fail(made->name, status);
  if (!result && nranks > 1) {
    mine->status = comm_bind(made, &mine->lanes, chosen + (size_t)rank * MESH_PORTS, mine->start);
    if ((status = bootstrap_bound(peers, nranks, rank)) && !mine->status) mine->status = status;
    if ((status = gather(peers, nranks, rank, reports, 0))) result = fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status));
    for (int r = 0; !result && r < nranks; r++)
      if (reports[r].status) result = bind_fail(made->name, r, reports[r].status);
    if (!result && (status = comm_join(made, reports, chosen)))
      result = fail(ncclSystemError, "communicator %s: its channels on the lanes it bound: %s", made->name, strerror(status));
  }
  free(reports); free(lanes); free(chosen);
  if (revoked(made)) result = revoked_fail(made);
  if (result) { bootstrap_close(made); comm_leave(made); }
  return result;
}

/* a non-blocking init's thread (ncclMeshCommInitRank): comm_init, its failure's text kept in `why` before its result
   is stored in `async` */
static void *init_run(void *argument) {
  struct ncclComm *comm = argument;
  const ncclResult_t result = comm_init(comm);
  if (result) memcpy(comm->why, last, sizeof comm->why);
  __atomic_store_n(&comm->async, result, __ATOMIC_RELEASE);
  return NULL;
}

/* nccl.h ncclMeshCommInitRank: the communicator made over the caller's operands (comm_table) and its init run
   (comm_init), here, or with config->blocking 0 on a thread of its own (init_run), *comm set at once and ncclInProgress
   returned: NCCL's non-blocking init (ncclConfig_t.blocking; ncclCommGetAsyncError ncclInProgress until it ends;
   ncclCommRevoke, ncclCommAbort and ncclCommDestroy end it as a value, metal-microbench docs/elastic.md#call-delivery) */
ncclResult_t ncclMeshCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, const struct mesh_trees *paths,
                                  const ncclMeshProgram *programs, int count, uint64_t small, const char *name, ncclConfig_t *config) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks || count < 0 || (count && !programs) || (paths && paths->nodes != (uint32_t)nranks))
    return fail(ncclInvalidArgument, "the init's arguments: rank %d of %d, %d programs", rank, nranks, count);
  for (int e = 0; e < count; e++)
    if (!programs[e].program || programs[e].program->nodes != (uint32_t)nranks) return fail(ncclInvalidArgument, "program %d is not of %d ranks", e, nranks);
  char id[sizeof commId.internal + 1];
  memcpy(id, commId.internal, sizeof commId.internal);
  id[sizeof commId.internal] = 0;
  if (!*id) return fail(ncclInvalidArgument, "the unique id is empty (ncclGetUniqueId)");
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
/* a communicator stopped for good (a revoke's and a destroy's): `revoked` set, failing every group after it; every
   socket of its bootstrap shut down, so an init still running (a non-blocking one's), its connect and accept reading the
   revoke at their next look (bootstrap_open), fails as a value; and its failure words' stop (word [6]) set, which every
   GPU wait of its reads and records as its failure, every later dispatch of it then doing nothing; under its guard */
static void comm_stop(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  if (comm->words) atomic_store_explicit(comm->words + 6, 1, memory_order_release);
  pthread_mutex_unlock(&comm->guard);
}

/* ncclCommRevoke: comm stopped (comm_stop) and its lane pairs unbound at once (comm_unbind: the bridge sends every unit
   they declared that no group supplied as a dud, so its peers' groups fail as values rather than wait), a host-path
   group of its returning the revoke at its next look and its GPU work ending at its next wait; from any thread, under
   comm's own lock and the attaches' alone, never one a group holds while it waits.  ncclCommDestroy then frees it */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  if (!comm || revokeFlags != NCCL_REVOKE_DEFAULT) return fail(ncclInvalidArgument, "comm is NULL or revokeFlags is %d", revokeFlags);
  comm_stop(comm);
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) comm_unbind(comm);
  pthread_mutex_unlock(&attach_lock);
  return ncclSuccess;
}

/* ncclCommDestroy: comm stopped (comm_stop), its init's thread joined (a non-blocking init's, which the stop ends), its
   host-path groups returned (each at its next look) and its GPU work awaited, at most MESH_PLATFORM_NS (each of its
   waits ends at the stop), its groups in flight lost; then its lane pairs unbound (the bridge sends what they declared)
   and comm freed */
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
  if (finished && comm->last) metal_wait(finished, comm->last, MESH_PLATFORM_NS);
  if (finished) metal_collect(finished);
  inflight_lose(comm);
  pthread_mutex_unlock(&group_lock);
  comm_free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

/* a non-blocking init's state (ncclInProgress until it ends, then its failure, named again here) or a launch's failure,
   else the communicator's failure (comm_failure: a revoke, a host-path group's, or one its GPU work recorded in its
   failure words), which a group that had already returned, its work enqueued, could not report */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  const ncclResult_t async = __atomic_load_n(&comm->async, __ATOMIC_ACQUIRE);
  if (async != ncclInProgress && *comm->why) fail(async, "%s", comm->why);
  if (async) { *asyncError = async; return ncclSuccess; }
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

/* a buffer in the window of this node's bridge (a lease's): its handle the buffer, the window registered on every
   device of the bridge at device up; any other buffer is refused, and every buffer while a lease of a bridge that
   stopped is held */
ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
  if (!comm || !handle) return fail(ncclInvalidArgument, "comm or handle is NULL");
  pthread_mutex_lock(&attach_lock);
  const int stale = attaches_stale();
  const struct hdr *m = attaches && attach_live(attaches) ? attaches->context.M : NULL;
  const uintptr_t at = (uintptr_t)buff, first = m ? (uintptr_t)m + m->data_off : 0, end = m ? (uintptr_t)m + m->length : 0;
  pthread_mutex_unlock(&attach_lock);
  if (stale) return attach_fail(stale, "ncclCommRegister of %zu bytes at %p", size, buff);
  if (!m || at < first || at > end || size > end - at)
    return fail(ncclInvalidArgument, "ncclCommRegister of %zu bytes at %p: not in the window of this node's bridge; a buffer "
                "collectives move is leased from the window (ncclMemAlloc)", size, buff);
  *handle = buff;
  return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  (void)handle;
  return comm ? ncclSuccess : fail(ncclInvalidArgument, "comm is NULL");
}

/* a lease of the window: its whole pages covering `size` (one page for 0), already registered on every device of the
   node's bridge, owned by the process's client; never other memory */
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
    result = fail(ncclSystemError, "ncclMemAlloc of %zu bytes (%zu pages of %u bytes): the window of this node's bridge has %u of "
                  "its %u pages free (%s); a client sizes its leases to the window its node's bridge configuration states "
                  "(etc/bridge/README.md)", size, (bytes + quantum - 1) / quantum * m->block, m->pgsz, mesh_arena_free(a->context.M),
                  mesh_arena_pages(m), strerror(status));
  } else {
    *ptr = mesh_section_address(&a->context, lease, 0);
    a->leases[a->count++] = (struct lease){*ptr, lease.first, (size_t)lease.pages * a->context.M->pgsz, NULL, NULL, 0};
  }
  pthread_mutex_unlock(&attach_lock);
  return result;
}

/* the lease at `ptr` returned to the window (mesh_lease_return): its pages free for any client's next lease */
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
        for (size_t k = 0; k < a->leases[i].npoints; k++) metal_release(a->leases[i].points[k].event);
        free(a->leases[i].points);
        status = mesh_lease_return(&a->context, row);
        a->leases[i] = a->leases[--a->count];
      }
  attaches_prune();
  pthread_mutex_unlock(&attach_lock);
  if (!found) return fail(ncclInvalidArgument, "ncclMemFree of %p: no lease of this process (ncclMemAlloc)", ptr);
  if (status) return fail(ncclInternalError, "ncclMemFree of %p: row %u of this node's region heads no lease of this process", ptr, row);
  return ncclSuccess;
}

/* the lease holding `ptr` (any byte of it) of a bridge that runs: the Metal buffer over the whole lease (made at its first
   use, kept until ncclMemFree) and ptr's offset in it; 0, ENOENT where no lease of this process holds ptr, ESTALE where
   its bridge stopped, ENOMEM */
static int lease_buffer(const void *ptr, void **buffer, size_t *offset) {
  int status = ENOENT;
  pthread_mutex_lock(&attach_lock);
  for (struct attach *a = attaches; a && status == ENOENT; a = a->next)
    for (size_t i = 0; i < a->count && status == ENOENT; i++) {
      struct lease *l = a->leases + i;
      const uintptr_t at = (uintptr_t)ptr, first = (uintptr_t)l->address;
      if (at < first || at >= first + l->bytes) continue;
      if (!attach_live(a)) { status = ESTALE; break; }
      if (!l->buffer) l->buffer = metal_wrap(l->address, l->bytes);
      status = l->buffer ? 0 : ENOMEM;
      if (!status) { *buffer = l->buffer; *offset = (size_t)(at - first); }
    }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* the lease of this process whose Metal buffer is `buffer` (lease_buffer's, nccl.h ncclMeshMemBuffer's): whether there
   is one, of a bridge that runs */
static int lease_of_buffer(const void *buffer) {
  int found = 0;
  pthread_mutex_lock(&attach_lock);
  for (struct attach *a = attaches; a && buffer && !found; a = a->next)
    for (size_t i = 0; i < a->count && !found; i++) found = a->leases[i].buffer == buffer && attach_live(a);
  pthread_mutex_unlock(&attach_lock);
  return found;
}

/* a lease's points that are reached dropped */
static void points_prune(struct lease *l) {
  size_t kept = 0;
  for (size_t k = 0; k < l->npoints; k++)
    if (metal_signaled(l->points[k].event) >= l->points[k].value) metal_release(l->points[k].event);
    else l->points[kept++] = l->points[k];
  l->npoints = kept;
}

/* the lease holding ptr, under attach_lock (NULL: none) */
static struct lease *lease_at(const void *ptr) {
  for (struct attach *a = attaches; a; a = a->next)
    for (size_t i = 0; i < a->count; i++) {
      const uintptr_t at = (uintptr_t)ptr, first = (uintptr_t)a->leases[i].address;
      if (at >= first && at < first + a->leases[i].bytes) return a->leases + i;
    }
  return NULL;
}

/* nccl.h ncclMeshMemBuffer */
ncclResult_t ncclMeshMemBuffer(const void *ptr, void **buffer, size_t *offset) {
  if (!ptr || !buffer || !offset) return fail(ncclInvalidArgument, "ncclMeshMemBuffer: a NULL argument");
  const int status = lease_buffer(ptr, buffer, offset);
  if (status == ESTALE) return attach_fail(status, "ncclMeshMemBuffer of %p", ptr);
  if (status == ENOMEM) return fail(ncclSystemError, "ncclMeshMemBuffer of %p: no Metal buffer over its lease", ptr);
  return status ? fail(ncclInvalidArgument, "ncclMeshMemBuffer: %p is in no lease of this process (ncclMemAlloc)", ptr) : ncclSuccess;
}

/* nccl.h ncclMeshMemUse: the point recorded on the lease holding ptr (its event retained until it is reached and
   dropped, or the lease freed) */
ncclResult_t ncclMeshMemUse(const void *ptr, size_t bytes, void *event, uint64_t value, int write) {
  (void)bytes;
  if (!ptr || !event) return fail(ncclInvalidArgument, "ncclMeshMemUse: a NULL argument");
  pthread_mutex_lock(&attach_lock);
  struct lease *l = lease_at(ptr);
  struct point *grown = l ? realloc(l->points, (l->npoints + 1) * sizeof *grown) : NULL;
  if (grown) {
    l->points = grown;
    points_prune(l);
    l->points[l->npoints++] = (struct point){metal_retain(event), value, write != 0};
  }
  pthread_mutex_unlock(&attach_lock);
  if (!l) return fail(ncclInvalidArgument, "ncclMeshMemUse: %p is in no lease of this process (ncclMemAlloc)", ptr);
  return grown ? ncclSuccess : fail(ncclSystemError, "out of memory");
}

/* design/transport.md: `event` reached `value` by `until` (uptime ns): 1, else 0 (a host wait ends by the platform's
   deadline, and decides nothing but its own group's failure) */
static int gpu_reached(void *event, uint64_t value, uint64_t until) {
  const uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  return metal_signaled(event) >= value || (now < until && metal_wait(event, value, until - now));
}

/* a host-path group's wait for the points recorded on the lease holding ptr (none where it is no lease): its writers'
   where the group reads it, every one where it writes it, each until the caller's GPU work reaches it (gpu_reached), all
   by `until`: 0, or ETIMEDOUT */
static int lease_wait(const void *ptr, int write, uint64_t until) {
  if (!ptr) return 0;
  for (;;) {
    pthread_mutex_lock(&attach_lock);
    struct lease *l = lease_at(ptr);
    struct point wait = {NULL, 0, 0};
    if (l) {
      points_prune(l);
      for (size_t k = 0; k < l->npoints && !wait.event; k++)
        if (write || l->points[k].write) wait = (struct point){metal_retain(l->points[k].event), l->points[k].value, l->points[k].write};
    }
    pthread_mutex_unlock(&attach_lock);
    if (!wait.event) return 0;
    const int reached = gpu_reached(wait.event, wait.value, until);
    metal_release(wait.event);
    if (!reached) return ETIMEDOUT;
  }
}

/* every lease the process holds of a bridge that runs: ncclSuccess, or ncclRemoteError where one is of a bridge that
   stopped (attach_fail's ESTALE) */
ncclResult_t ncclMeshMemLive(void) {
  pthread_mutex_lock(&attach_lock);
  const int stale = attaches_stale();
  pthread_mutex_unlock(&attach_lock);
  return stale ? attach_fail(stale, "ncclMeshMemLive") : ncclSuccess;
}

/* an operator made into the communicator's table, its handle past ncclNumOps */
static ncclResult_t op_make(ncclComm_t comm, struct redop made, ncclRedOp_t *op) {
  for (int k = 0; k < OPS; k++)
    if (!comm->ops[k].used) {
      comm->ops[k] = made;
      *op = (ncclRedOp_t)(ncclNumOps + k);
      return ncclSuccess;
    }
  return fail(ncclInvalidUsage, "%d operators made on one communicator", OPS);
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  if (!op || !scalar || !comm || (unsigned)datatype >= ncclNumTypes) return fail(ncclInvalidArgument, "PreMulSum's arguments");
  if (residence != ncclScalarHostImmediate) return fail(ncclInvalidArgument, "the host path reads the scalar when the operator is made (ncclScalarHostImmediate)");
  struct redop made = {1, datatype, {0}, 0};
  memcpy(made.value, scalar, SIZE[datatype]);
  return op_make(comm, made, op);
}

/* nccl.h ncclMeshRedOpCreatePartials: an operator of the communicator's, as PreMulSum's, its row 2 + dim floats */
ncclResult_t ncclMeshRedOpCreatePartials(ncclRedOp_t *op, size_t dim, ncclDataType_t datatype, ncclComm_t comm) {
  if (!op || !comm || datatype != ncclFloat32 || dim > UINT32_MAX - 2)
    return fail(ncclInvalidArgument, "the partials merge's arguments: its operator, its communicator, float32 and a row's dim");
  return op_make(comm, (struct redop){1, datatype, {0}, (uint32_t)(2 + dim)}, op);
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  const int k = (int)op - ncclNumOps;
  if (!comm || k < 0 || k >= OPS || !comm->ops[k].used) return fail(ncclInvalidArgument, "operator %d is none of this communicator's", (int)op);
  comm->ops[k].used = 0;
  return ncclSuccess;
}

/* -- a communicator's channels and its failures (design/transport.md section 14).  Toward each member a cable joins, a
   channel of each class on the lane of that class of the pair its init bound there.  Both ends of a channel compute the
   same positions for a group (each collective's rounds in call order, then the point-to-point calls paired in order:
   mesh-plan.h mesh_group_lay), each publishes every position (a filler where it has nothing to send), and each
   publishes position t once it has consumed the peer's t - LAG: the peer then has consumed this end's t - 2 LAG, so no
   unit of either ring is written again before it is sent and read (NCCL's Simple protocol's slots, its head credit
   carried by the reverse direction's positions; mesh-plan.h struct mesh_wire).  A position is a run of its lane's
   units, laid alike at both ends from the runs before it (this end's out units from its first, its in units from the
   peer's): one unit a filler's, a piece's bytes in whole units, and a run that would pass the ring's end starting at its
   next lap, the units it skips supplied as fillers, since the bridge sends a lane's units strictly in order.  A
   group declares its units to the bridge before it supplies any (declare), so a member that fails mid-group is
   replaced at its peers by duds, and a run's landing is its last unit's (a lane's units land in order). ---------- */

#define LAG MESH_LAG

static double now_s(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

/* the communicator failed for good, made again by its caller (ncclCommDestroy, then an init): its first failure's
   result and text kept (comm_failure reads them), its groups in flight lost (inflight_lose); that result, its failure
   named (ncclGetLastError).  Under group_lock */
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) __attribute__((format(printf, 3, 4)));
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) {
  if (!comm->failed) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(comm->failure, sizeof comm->failure, format, arguments);
    va_end(arguments);
    comm->failed = result;
  }
  inflight_lose(comm);
  return fail(comm->failed, "%s", comm->failure);
}

/* a dud's landing, a client's failure (design/transport.md section 10) */
#define DUD "node %u supplied nothing at position %llu on its channel: a dud landed there (all ones: its call failed, was " \
            "revoked or ended, or missed its deadline)"

/* the communicator's failure: the one it kept; a revoke; else one its GPU work recorded in its failure words (a wait's
   bound reached, its link lost, a dud landed: nccl-mesh-metal.h METAL_*; a counted all-to-all past its layout), kept
   now (comm_fail); ncclSuccess where none.  Under group_lock */
static ncclResult_t comm_failure(struct ncclComm *comm) {
  if (comm->failed) return comm_fail(comm, comm->failed, "%s", comm->failure);
  if (revoked(comm))
    return comm_fail(comm, ncclRemoteError, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
  _Atomic uint64_t *v = comm->words;
  const uint64_t at = v ? atomic_load_explicit(v + 2, memory_order_acquire) : 0, over = v ? atomic_load_explicit(v + 5, memory_order_acquire) : 0;
  if (at) {
    const uint64_t cause = atomic_load_explicit(v, memory_order_relaxed), h = atomic_load_explicit(v + 3, memory_order_relaxed);
    const uint32_t node = h < comm->count ? comm->channels[h].node : UINT32_MAX;
    if (cause == METAL_LOST)
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost (its lane's landed words ~0) at position %llu", node,
                       (unsigned long long)(at - 1));
    if (cause == METAL_DUD) return comm_fail(comm, ncclRemoteError, DUD, node, (unsigned long long)(at - 1));
    return comm_fail(comm, ncclRemoteError, "position %llu on the channel to node %u landed nothing within %llu polls of a GPU wait (its "
                     "stream's bound, ncclMeshStream polls): the peer's work there began that much later, or its link was lost",
                     (unsigned long long)(at - 1), node, (unsigned long long)atomic_load_explicit(v + 4, memory_order_relaxed));
  }
  if (over)
    return comm_fail(comm, ncclInvalidUsage, "a counted all-to-all's counts passed what its layout carries (ncclMeshAlltoAllCounted: a rank's "
                     "rows past sendrows, a pair's past min(sendrows, capacity), or a rank's received rows past capacity); its rows past them "
                     "were not moved and its arrived word is 2");
  return ncclSuccess;
}

/* a group's communicator ready to cross a link: not failed (comm_failure), on the bridge it bound its lanes on */
static ncclResult_t comm_ready(struct ncclComm *comm) {
  const ncclResult_t result = comm_failure(comm);
  if (result || (comm->attach && attach_live(comm->attach))) return result;
  return comm_fail(comm, ncclRemoteError, "this node's bridge, on which communicator %s bound its lanes, stopped: it is made again "
                   "(ncclCommDestroy, then an init)", comm->name);
}

/* design/transport.md, the checker of a communicator's table once it is compiled (comm_compile): its programs on its
   wire, read by mesh_program_advise at mesh-advise.c's operands, each below its entry's bound, every warning logged as
   NCCL logs one, on a thread of its own over a copy: nothing joins it, and it refuses nothing */
struct advice { const struct ncclComm *comm; uint32_t entry; struct mesh_operand operand; };
static void advise_warn(void *argument, const char *line) {
  const struct advice *a = argument;
  const struct mesh_program *x = &a->comm->entries[a->entry].program;
  fprintf(stderr, "NCCL WARN communicator %s rank %d: program %u (collective %u, root %u), %llu elements of %u bytes: %s\n", a->comm->name,
          a->comm->rank, a->entry, x->what, x->root, (unsigned long long)a->operand.elements, a->operand.element_bytes, line);
}
static void *advise_run(void *argument) {
  struct ncclComm *comm = argument;
  static const struct mesh_operand operands[] = {{ncclFloat32 + 1, 4, 1}, {ncclFloat32 + 1, 4, 4096}, {ncclFloat32 + 1, 4, 1000003},
                                                 {ncclFloat16 + 1, 2, 1u << 20}};
  for (uint32_t e = 0; e < comm->programs; e++)
    for (size_t o = 0; o < sizeof operands / sizeof *operands; o++) {
      struct advice a = {comm, e, operands[o]};
      if (!comm->entries[e].below || operands[o].elements * operands[o].element_bytes < comm->entries[e].below)
        mesh_program_advise(&comm->entries[e].program, &comm->wire, operands[o], advise_warn, &a);
    }
  comm_free(comm);
  return NULL;
}
static void comm_advise(const struct ncclComm *comm) {
  struct ncclComm *copy = comm->nranks > 1 ? calloc(1, sizeof *copy) : NULL;
  if (!copy) return;
  pthread_mutex_init(&copy->guard, NULL);
  copy->nranks = comm->nranks; copy->rank = comm->rank; copy->wire = comm->wire;
  memcpy(copy->name, comm->name, sizeof copy->name);
  int made = (copy->entries = calloc((size_t)comm->programs + 1, sizeof *copy->entries)) != NULL;
  for (uint32_t e = 0; made && e < comm->programs; e++)
    if ((made = !program_copy(&copy->entries[e].program, &comm->entries[e].program))) copy->entries[copy->programs++].below = comm->entries[e].below;
  pthread_t thread;
  if (made && !pthread_create(&thread, NULL, advise_run, copy)) pthread_detach(thread);
  else comm_free(copy);
}

/* -- a group ------------------------------------------------------------------------------------------- */

/* where a step's operand is on the Metal path: a buffer and a byte offset into it */
struct where { void *buffer; size_t offset; };
static int same(struct where a, struct where b) { return a.buffer == b.buffer && a.offset == b.offset; }

struct steps {
  struct mesh_step *steps;
  uint32_t count;
  unsigned char *done;
  /* the buffers the steps name: 0 the operand, b >= 1 scratch b (its copy); on the host (`host`, scratch owned unless
     `borrowed`: a counted all-to-all's are the call's) or on the GPU (`gpu`) */
  uint32_t buffers;
  int borrowed;
  /* a call reading its send buffer in place (plan_direct): each step's source where it reads the send buffer (`read`,
     buffer NULL: its own buffer), whether a receive's combine is its range's first (`fresh`: buffer = send op piece), and
     which buffers any step writes (`needed`, a buffer each; one no step writes is never made) */
  struct where *read;
  unsigned char *fresh, *needed;
  unsigned char **host;
  struct where *gpu;
  const struct mesh_program *program;
  struct mesh_operand operand;
  /* an uneven call's segments in operand elements: its counts, or (`cut`, owned) its counts over a merge's row */
  const uint64_t *segments;
  uint64_t *cut;
  /* each step's positions as schedule_group places them (its channel, first position, pieces), and the receive before
     it (by round, then plan order) that has any, or -1 */
  uint32_t *channel;
  uint64_t *first, *pieces;
  int32_t *before;
};

/* a call: its collective (or point-to-point direction), root, peer, buffers, count, type, operator and communicator;
   `segments`, an uneven reduce-scatter's or all-gather's count a rank (ncclMeshReduceScatterV, ncclMeshAllGatherV: the
   call's own copy, `total` their sum; NULL: `count` each), a counted all-to-all's sendsegs then recvsegs (one whose
   counts are the host's, hosted: the elements to each rank then from each, `count` those crossing); a counted
   all-to-all's (WHAT_COUNTED, ncclMeshAlltoAllCounted) counts, the host's copy of the landed counts and its word, the
   GPU's copy, its rows, the most it receives and a row's elements, its landed counts (`got`: the host path's own, the
   Metal path's a scratch, `got_at`), the counts receives not yet consumed (`counting`) and whether its end is encoded;
   a reduction's operator as the host runs it (reduction: `combine`, `post`, `pre`; the partials merge's row of `unit`
   elements) */
struct call {
  int what, root, peer;
  const void *send;
  void *recv;
  size_t count, total;
  uint64_t *segments;
  const void *counts;
  void *received, *landed;
  uint64_t *arrived;
  size_t rows, capacity, row;
  int64_t *got;
  struct where counts_at, received_at, landed_at, arrived_at, got_at;
  uint32_t counting;
  int ended;
  /* a call reading its send buffer in place (plan_direct): the ranges of its result no step writes, copied from it after
     its publications are encoded (`complement`, triples: the result's byte offset, the send buffer's, the bytes) */
  uint64_t *complement;
  uint32_t complements;
  ncclDataType_t type;
  ncclRedOp_t op;
  ncclComm_t comm;
  int combine, post;
  uint32_t unit;
  unsigned char pre[8];
  int premultiplied;
  struct steps *bound;
  unsigned char *result, *copy;
  int owned;
  cudaStream_t stream;
  void *command, *encoder, *(*workspace)(size_t, void *), *context, *table, *residency;
  uint32_t polls;
  ncclMeshIssue *issue;
  ncclMeshFused *fused;
  struct where send_at, recv_at, sum;
};

static _Thread_local int depth;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;

static int reducing(int what) { return what == MESH_ALLREDUCE || what == MESH_REDUCE || what == MESH_REDUCE_SCATTER; }
/* a reduction's operand element in the call's elements (the partials merge's row, else one), and a piece of `bytes` as
   combine_into and the kernels take it: its type and operator (the merge's METAL_MERGE and its row's floats) and its
   elements (the merge's rows) */
static uint32_t unit_of(const struct call *c) { return c->combine == COMBINE_MERGE ? c->unit : 1; }
static int ktype(const struct call *c) { return c->combine == COMBINE_MERGE ? METAL_MERGE : (int)c->type; }
static int kop(const struct call *c) { return c->combine == COMBINE_MERGE ? (int)c->unit : c->combine; }
static size_t kcount(const struct call *c, size_t bytes) { return bytes / (SIZE[c->type] * unit_of(c)); }
static int p2p(const struct call *c) { return c->what == WHAT_SEND || c->what == WHAT_RECV; }
static int counted(const struct call *c) { return c->what == WHAT_COUNTED; }
/* a counted all-to-all whose counts are the host's, both ends' (nccl.h ncclMeshAlltoAllV): no count crosses */
static int hosted(const struct call *c) { return counted(c) && !c->counts; }
static size_t elements(const struct call *c) {
  if (counted(c)) return c->count;
  if (c->segments) return c->total;
  return c->count * ((c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER) ? (size_t)c->comm->nranks : 1);
}
/* rank r's segment of a reduce-scatter's or all-gather's operand: the elements before it, and *length its own (r x
   count and count; an uneven call's, its counts') */
static size_t segment_at(const struct call *c, int r, size_t *length) {
  size_t before = 0;
  for (int q = 0; c->segments && q < r; q++) before += c->segments[q];
  *length = c->segments ? c->segments[r] : c->count;
  return c->segments ? before : (size_t)r * c->count;
}
/* a counted all-to-all's (ncclMeshAlltoAllCounted) layout: the entries before rank q's segment of a count a rank (its
   sendsegs or recvsegs), a row's bytes, and the rows each pair's message carries room for, min(sendrows, capacity),
   every rank's alike, so both ends of a pair lay its positions without its counts */
static uint64_t seg_first(const uint64_t *seg, int q) {
  uint64_t at = 0;
  for (int r = 0; r < q; r++) at += seg[r];
  return at;
}
static size_t row_bytes(const struct call *c) { return c->row * SIZE[c->type]; }
static size_t counted_bound(const struct call *c) { return c->rows < c->capacity ? c->rows : c->capacity; }
/* the rows of int64 counts [first, first + entries), a negative count none */
static uint64_t counted_rows(const int64_t *v, uint64_t first, uint64_t entries) {
  uint64_t rows = 0;
  for (uint64_t k = 0; k < entries; k++) rows += v[first + k] > 0 ? (uint64_t)v[first + k] : 0;
  return rows;
}
/* where the piece at byte `offset` (`length` bytes) of a counted message lies among the rows of the counts [first, first
   + entries), after the rows of every count before them: its byte place in the rows buffer, and *valid the bytes of it the
   counts carry, within `limit` bytes of the buffer (the kernels' rule, nccl-mesh-metal.m counted_send/counted_land) */
static size_t counted_span(const int64_t *v, uint64_t first, uint64_t entries, size_t row, size_t offset, size_t length, size_t limit,
                           size_t *valid) {
  const size_t at = (size_t)counted_rows(v, 0, first) * row + offset, rows = (size_t)counted_rows(v, first, entries) * row;
  const size_t bytes = rows > offset ? (rows - offset < length ? rows - offset : length) : 0;
  *valid = at >= limit ? 0 : (bytes < limit - at ? bytes : limit - at);
  return at;
}
/* a counted message's piece toward or from rank q, on the host: its place in the send rows (by the call's counts) or the
   receive buffer (by the landed counts) */
static size_t counted_place(const struct call *c, int q, int receive, size_t offset, size_t length, size_t *valid) {
  const int n = c->comm->nranks;
  const uint64_t *seg = c->segments + (receive ? n : 0);
  return counted_span(receive ? c->got : (const int64_t *)c->counts, seg_first(seg, q), seg[q], row_bytes(c), offset, length,
                      (receive ? c->capacity : c->rows) * row_bytes(c), valid);
}

/* whether a call crosses a link: elements, more than one rank, and a point-to-point call's peer another rank */
static int crosses(const struct call *c) { return elements(c) && c->comm->nranks > 1 && !(p2p(c) && c->peer == c->comm->rank); }

static void steps_free(struct steps *s) {
  if (!s) return;
  for (uint32_t b = 1; s->host && !s->borrowed && b < s->buffers; b++) free(s->host[b]);
  free(s->host); free(s->gpu);
  free(s->steps); free(s->done); free(s->channel); free(s->first); free(s->pieces); free(s->before); free(s->read); free(s->fresh); free(s->needed);
  free(s->cut);
  free(s);
}

/* a counted all-to-all's steps, all of round 0: one whose counts are the host's, to each other rank q its elements for q
   (buffer 0, the send buffer at q's segment) and from each its elements (buffer 1, the receive buffer at q's segment),
   the sends and receives of NCCL's lowering (classifyCollToP2pTasks) as one call; else (mesh 3981ca80^ run_counted's one
   exchange: to each rank its segment of the counts and then its rows, both at once, so no rows wait for the counts to
   cross first) to each other rank q it has counts for, its counts (buffer 0, from the call's counts at q's first entry)
   and then its rows (buffer 1, the send rows; room for the pair's bound, the place read from the counts where it
   runs); from each, its counts (buffer 2, the
   landed counts at q's place in rank order), and then from each its rows (buffer 3, the receive buffer, at the place the
   landed counts give), each after every counts receive (mesh_steps_before: a receive's first piece after the one
   before it), so a rank's k-th SEND to a peer is that peer's k-th receive from it; *counting the counts receives */
static uint32_t counted_steps(ncclComm_t comm, struct call *c, struct mesh_step *steps) {
  const int n = comm->nranks, me = comm->rank;
  const uint64_t *sseg = c->segments, *rseg = c->segments + n, bound = counted_bound(c);
  const struct mesh_operand counts = {(uint32_t)ncclInt64 + 1, 8, 0}, rows = {(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], bound * c->row};
  uint32_t k = 0;
  c->counting = 0;
  for (int q = 0; hosted(c) && q < 2 * n; q++)
    if (q % n != me && c->segments[q])
      steps[k++] = (struct mesh_step){q < n ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)(q % n), 0, q >= n, seg_first(c->segments + (q < n ? 0 : n), q % n),
                                      {(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], c->segments[q]}};
  if (hosted(c)) return k;
  for (int q = 0; q < n; q++) {
    if (q == me || !sseg[q]) continue;
    steps[k++] = (struct mesh_step){MESH_STEP_SEND, (uint32_t)q, 0, 0, seg_first(sseg, q), {counts.type, counts.element_bytes, sseg[q]}};
    if (bound) steps[k++] = (struct mesh_step){MESH_STEP_SEND, (uint32_t)q, 0, 1, 0, rows};
  }
  for (int q = 0; q < n; q++)
    if (q != me && rseg[q]) {
      steps[k++] = (struct mesh_step){MESH_STEP_COPY, (uint32_t)q, 0, 2, seg_first(rseg, q), {counts.type, counts.element_bytes, rseg[q]}};
      c->counting++;
    }
  for (int q = 0; bound && q < n; q++)
    if (q != me && rseg[q]) steps[k++] = (struct mesh_step){MESH_STEP_COPY, (uint32_t)q, 0, 3, 0, rows};
  return k;
}

static size_t piece_bytes(const struct mesh_step *step) { return step->piece.elements * step->piece.element_bytes; }

/* every call that crosses a link planned: a collective along the communicator's table (comm_compile; an uneven one's
   segments its counts, mesh_program_steps_cut: its program's segments must be the ranks), its operand's element a
   merge's row where it merges, a point-to-point call one SEND or COPY step; a step that moves bytes toward or from a
   rank no cable of this node joins refused (the bridges forward nothing) */
static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
    const ncclResult_t compiled = p2p(c) || counted(c) ? ncclSuccess : comm_compile(comm);
    if (compiled) return compiled;
    const struct mesh_program *program = p2p(c) || counted(c) ? NULL : program_of(comm, c->what, c->root, elements(c) * SIZE[c->type]);
    if (!p2p(c) && !counted(c) && !program)
      return fail(ncclInvalidUsage, "the communicator has no program for collective %d at root %d", c->what, c->root);
    if (c->segments && program) {
      uint32_t segments = 1;
      for (uint32_t p = 0; p < program->parts; p++) if (program->segment[p] + 1 > segments) segments = program->segment[p] + 1;
      if (segments != (uint32_t)comm->nranks)
        return fail(ncclInvalidUsage, "an uneven collective's program cuts %u segments, not one a rank of %d", segments, comm->nranks);
    }
    struct steps *s = calloc(1, sizeof *s);
    const size_t most = (program ? program->first[program->nodes] : counted(c) ? 4 * (size_t)comm->nranks : 1) + 1;
    if (s) s->buffers = counted(c) ? 4 : 1 + (program ? program->scratch : 0);
    if (!s || !(s->steps = calloc(most, sizeof *s->steps)) || !(s->done = calloc(most, 1)) || !(s->channel = calloc(most, sizeof *s->channel)) ||
        !(s->first = calloc(most, sizeof *s->first)) || !(s->pieces = calloc(most, sizeof *s->pieces)) || !(s->before = calloc(most, sizeof *s->before)) ||
        !(s->host = calloc(s->buffers, sizeof *s->host)) || !(s->gpu = calloc(s->buffers, sizeof *s->gpu))) {
      steps_free(s);
      return fail(ncclSystemError, "out of memory");
    }
    c->bound = s;
    const uint32_t u = unit_of(c);
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)(SIZE[c->type] * u), elements(c) / u};
    s->operand = operand;
    s->program = program;
    s->segments = c->segments;
    if (operand.element_bytes > comm->wire.slot)
      return fail(ncclInvalidUsage, "a row of the partials merge of %u bytes, past a large slot's %llu", operand.element_bytes,
                  (unsigned long long)comm->wire.slot);
    if (c->segments && u > 1 && !(s->segments = s->cut = malloc((size_t)comm->nranks * sizeof *s->cut))) return fail(ncclSystemError, "out of memory");
    for (int r = 0; s->cut && r < comm->nranks; r++) s->cut[r] = c->segments[r] / u;
    if (p2p(c)) {
      s->steps[0] = (struct mesh_step){c->what == WHAT_SEND ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)c->peer, 0, 0, 0, operand};
      s->count = 1;
    } else if (counted(c)) {
      s->count = counted_steps(comm, c, s->steps);
      s->borrowed = 1;
    } else s->count = mesh_program_steps_cut(program, (uint32_t)comm->rank, operand, s->segments, s->steps);
    for (uint32_t k = 0; k < s->count; k++)
      if (piece_bytes(s->steps + k) && comm->toward[s->steps[k].peer] < 0)
        return fail(ncclInvalidUsage, "a call of %s moves bytes between rank %d and rank %u (node %u), which no cable of this node joins: the "
                    "bridges forward nothing, so a route through another node is refused", comm->name, comm->rank, s->steps[k].peer,
                    comm->members[s->steps[k].peer]);
  }
  return ncclSuccess;
}

/* a step's piece on one channel: its positions [first, first + pieces), at byte offset of its operand, a position
   carrying `carried` bytes of it (its slot's whole elements, mesh-plan.h mesh_wire_pieces) */
struct message { struct call *c; uint32_t step; uint64_t first, pieces; size_t offset, bytes, carried; };
/* message m's piece at position `at` of its channel: its byte offset in the message, and *length its bytes */
static size_t piece_at(const struct message *m, uint64_t at, size_t *length) {
  const size_t offset = (at - m->first) * m->carried;
  *length = m->bytes - offset < m->carried ? m->bytes - offset : m->carried;
  return offset;
}
/* a run of a lane's units: its first, its count, and the units before it that it skips to start its lap (a run that
   would pass the ring's end starts at its next lap: mesh-plan.h mesh_wire), supplied as fillers */
struct run { uint64_t first; uint32_t units, skip; };
/* a group's schedule on one channel: its messages each way in position order (`out_at`, `in_at` the next not yet
   published or consumed) and, from its first position `start`, each position's runs, out (runs[2 (t - start)]) and in
   (runs[2 (t - start) + 1]) */
struct schedule { struct message *out, *in; size_t outs, ins, out_at, in_at; uint64_t start; struct run *runs; };

/* the run of `units` units from a lane's next unit *at on a ring of `ring` units, *at past it */
static struct run run_from(uint64_t *at, uint64_t units, uint64_t ring) {
  const uint64_t first = *at % ring + units > ring ? (*at / ring + 1) * ring : *at;
  const struct run made = {first, (uint32_t)units, (uint32_t)(first - *at)};
  *at = first + units;
  return made;
}

/* a group's calls as mesh-plan.h's group, which mesh_group_lay lays (the one layout both ends make): call i's kind and
   its steps at rank r, steps[i * N + r] and count[i * N + r] (none where the rank's are not known) */
static void group_of(const struct call *list, size_t n, uint32_t N, const struct mesh_step **steps, uint32_t *count,
                     struct mesh_group_call *calls, struct mesh_group *group) {
  for (size_t i = 0; i < n; i++)
    calls[i] = (struct mesh_group_call){p2p(list + i) ? MESH_GROUP_POINT : counted(list + i) ? MESH_GROUP_COUNTED : MESH_GROUP_COLLECTIVE, 0,
                                        count + i * N, steps + i * N};
  *group = (struct mesh_group){N, (uint32_t)n, 0, 0, calls};
}

/* the bytes position `at` carries of a channel's messages one way (0: none, a filler's), *k moved past those before it */
static size_t bytes_at(const struct message *m, size_t count, size_t *k, uint64_t at) {
  while (*k < count && at >= m[*k].first + m[*k].pieces) ++*k;
  size_t length = 0;
  if (*k < count && at >= m[*k].first) piece_at(m + *k, at, &length);
  return length;
}

/* each channel's positions for the group, from the last scheduled one's end (mesh_group_lay), each step's positions
   and the receive before it (mesh-plan.h mesh_steps_before) kept, a receive with none done; and each position's runs,
   out from this end's next unit and in from the peer's, a run as many units as its piece's bytes (mesh_wire_units: a
   filler's one) */
static ncclResult_t schedule_group(struct ncclComm *comm, struct call *list, size_t n, struct schedule *plans) {
  const uint32_t N = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  size_t total = 1;
  for (size_t i = 0; i < n; i++) total += list[i].bound ? list[i].bound->count : 0;
  const struct mesh_step **steps = calloc(n * N + 1, sizeof *steps);
  uint32_t *counts = calloc(n * N + 1, sizeof *counts);
  struct mesh_group_call *calls = calloc(n + 1, sizeof *calls);
  struct mesh_group_place *made = calloc(total, sizeof *made);
  struct mesh_place *scratch = calloc(total, sizeof *scratch);
  struct mesh_group group;
  ncclResult_t result = steps && counts && calls && made && scratch ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (size_t i = 0; i < n && !result; i++)
    if (list[i].bound) {
      steps[i * N + me] = list[i].bound->steps; counts[i * N + me] = list[i].bound->count;
      memset(list[i].bound->pieces, 0, list[i].bound->count * sizeof *list[i].bound->pieces);
    }
  if (!result) group_of(list, n, N, steps, counts, calls, &group);
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    struct channel *ch = comm->channels + h;
    struct schedule *p = plans + h;
    const uint64_t slot = ch->large ? comm->wire.slot : comm->wire.small;
    *p = (struct schedule){calloc(total, sizeof *p->out), calloc(total, sizeof *p->in), 0, 0, 0, 0, ch->end, NULL};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    uint32_t placed = 0;
    const uint64_t positions = mesh_group_lay(&group, me, (uint32_t)ch->peer, ch->large, &comm->wire, 0, scratch, made, &placed);
    for (uint32_t k = 0; k < placed; k++) {
      const struct mesh_place *x = &made[k].place;
      struct call *c = list + made[k].call;
      const struct mesh_step *step = c->bound->steps + x->step;
      const struct message m = {c, x->step, ch->end + x->first, x->pieces, step->first * step->piece.element_bytes, piece_bytes(step),
                                slot - slot % step->piece.element_bytes};
      if (x->send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
      c->bound->channel[m.step] = h; c->bound->first[m.step] = m.first; c->bound->pieces[m.step] = m.pieces;
    }
    if (!(p->runs = calloc(2 * positions + 1, sizeof *p->runs))) { result = fail(ncclSystemError, "out of memory"); break; }
    size_t o = 0, i = 0;
    for (uint64_t t = 0; t < positions; t++) {
      const size_t sent = bytes_at(p->out, p->outs, &o, ch->end + t), got = bytes_at(p->in, p->ins, &i, ch->end + t);
      p->runs[2 * t] = run_from(&ch->laid, mesh_wire_units(&comm->wire, ch->large, sent), ch->view.units);
      p->runs[2 * t + 1] = run_from(&ch->taken, mesh_wire_units(&comm->wire, ch->large, got), ch->view.units);
    }
    ch->end += positions;
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct steps *st = list[i].bound;
    if (st && mesh_steps_before(st->steps, st->count, st->before)) result = fail(ncclSystemError, "out of memory");
    for (uint32_t k = 0; st && !result && k < st->count; k++)
      if (st->steps[k].op != MESH_STEP_SEND && !st->pieces[k]) st->done[k] = 1;
  }
  free(steps); free(counts); free(calls); free(made); free(scratch);
  return result;
}

static void schedules_free(const struct ncclComm *comm, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < comm->count; h++) { free(plans[h].out); free(plans[h].in); free(plans[h].runs); }
  free(plans);
}

/* design/transport.md section 14: each channel's lane declared through the units its groups laid (mesh_lane_declare),
   to be supplied within the platform's deadline from now, before any of them is supplied, so a member that fails, is
   revoked or ends mid-group is replaced at its peers by duds; with `again`, each lane with positions still to publish
   declared anew (a completion's, ncclMeshComplete: its deadline from then) */
static void declare(struct ncclComm *comm, int again) {
  const uint64_t deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + MESH_PLATFORM_NS;
  for (uint32_t h = 0; h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    if (ch->laid <= ch->declared && !(again && ch->sent < ch->end)) continue;
    mesh_lane_declare(&comm->attach->context, ch->port, ch->lane, ch->laid, deadline);
    ch->declared = ch->laid;
  }
}

/* the group's schedule (schedule_group), its lanes declared (declare); a failure the communicator's (comm_fail: its
   channels stand mid-schedule); *made NULL where it failed */
static ncclResult_t schedule(struct ncclComm *comm, struct call *list, size_t n, struct schedule **made) {
  struct schedule *plans = calloc(comm->count + 1, sizeof *plans);
  *made = NULL;
  if (!plans) return fail(ncclSystemError, "out of memory");
  const ncclResult_t result = schedule_group(comm, list, n, plans);
  if (result) { schedules_free(comm, plans); return comm_fail(comm, result, "%s", last); }
  comm->crossed = 1;
  declare(comm, 0);
  *made = plans;
  return ncclSuccess;
}

/* a step may run once the receive before it is consumed (mesh_steps_before: every earlier receive then is, by induction, as
   each receive's first piece waits on the one before it) */
static int ready(const struct steps *s, uint32_t k) { return s->before[k] < 0 || s->done[s->before[k]]; }

/* a channel's out run supplied on the host (design/transport.md section 14): `length` bytes into its run's slots of the
   out ring (fewer than eight after eight bytes of 0, and a filler's, `bytes` NULL, eight bytes of 0, so neither reads
   as a dud), then, after a release fence, each unit's ready word 1 + its cycle, the units the run skips first, in order */
static void supply(const struct channel *ch, const struct run *r, const void *bytes, size_t length) {
  const uint64_t units = ch->view.units;
  unsigned char *slot = ch->view.out + (r->first % units) * ch->view.unit;
  if (!bytes || length < sizeof(uint64_t)) memset(slot, 0, sizeof(uint64_t));
  if (bytes) memcpy(slot, bytes, length);
  atomic_thread_fence(memory_order_release);
  for (uint64_t u = r->first - r->skip; u < r->first + r->units; u++)
    atomic_store_explicit(ch->view.ready + u % units, 1 + u / units, memory_order_release);
}

/* a channel's in run on the host: 1 once it has landed (its last unit's landed word at least 1 + its cycle: a lane's
   units land in order), 0 before; -1 where its link was lost (the word ~0), -2 where the peer supplied nothing (a dud:
   its first eight bytes all ones, design/transport.md section 10) */
static int landing(const struct channel *ch, const struct run *r) {
  const uint64_t units = ch->view.units, last = r->first + r->units - 1;
  const uint64_t word = atomic_load_explicit(ch->view.landed + last % units, memory_order_acquire);
  if (word == UINT64_MAX) return -1;
  if (word < 1 + last / units) return 0;
  uint64_t head;
  memcpy(&head, ch->view.in + (r->first % units) * ch->view.unit, sizeof head);
  return head == UINT64_MAX ? -2 : 1;
}

/* every channel's positions published and consumed on the host: a SEND's piece supplied into its run, a REDUCE's
   combined into the result, a COPY's copied into the operand (and the result, where they differ: a direct exchange's
   sums go to a copy, as its SENDs read the operand); a landing that reads the link's loss or a dud fails it, and so does
   a revoke, at its next look, and MESH_PLATFORM_NS with nothing landed and nothing to publish (design/transport.md: the
   platform's deadline, a host wait's bound).  Every look spins: nothing parks.  Its positions then count as awaited
   for the GPU work after it (`waited`) */
static ncclResult_t run(struct ncclComm *comm, struct schedule *plans) {
  uint64_t since = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for (;;) {
    if (revoked(comm)) return revoked_fail(comm);
    int open = 0, moved = 0;
    const struct channel *waiting = NULL;
    for (uint32_t h = 0; h < comm->count; h++) {
      struct channel *ch = comm->channels + h;
      struct schedule *p = plans + h;
      while (ch->consumed < ch->end) {
        const uint64_t at = ch->consumed;
        const struct run *r = p->runs + 2 * (at - p->start) + 1;
        const int landed = landing(ch, r);
        if (!landed) break;
        if (landed < 0)
          return landed == -1 ? fail(ncclRemoteError, "the link to node %u was lost at position %llu (its lane's landed words ~0)", ch->node,
                                     (unsigned long long)at)
                              : fail(ncclRemoteError, DUD, ch->node, (unsigned long long)at);
        struct message *m = p->in_at < p->ins && at >= p->in[p->in_at].first ? p->in + p->in_at : NULL;
        if (m) {
          struct steps *s = m->c->bound;
          if (at == m->first && !ready(s, m->step)) break;
          size_t length, offset = piece_at(m, at, &length);
          const unsigned char *from = ch->view.in + (r->first % ch->view.units) * ch->view.unit;
          unsigned char *to = s->host[s->steps[m->step].buffer] + m->offset + offset;
          if (counted(m->c) && s->steps[m->step].buffer == 3) {
            size_t valid;
            const size_t place = counted_place(m->c, (int)s->steps[m->step].peer, 1, offset, length, &valid);
            memcpy((unsigned char *)m->c->recv + place, from, valid);
          } else if (s->steps[m->step].op == MESH_STEP_REDUCE) combine_into(ktype(m->c), kop(m->c), to, from, kcount(m->c, length));
          else memcpy(to, from, length);
          if (at + 1 == m->first + m->pieces) { s->done[m->step] = 1; p->in_at++; }
        }
        ch->consumed++;
        moved = 1;
      }
      while (ch->sent < ch->end && ch->sent < ch->consumed + LAG) {
        const uint64_t at = ch->sent;
        const struct run *r = p->runs + 2 * (at - p->start);
        struct message *m = p->out_at < p->outs ? p->out + p->out_at : NULL;
        if (m && at >= m->first) {
          if (at == m->first && !ready(m->c->bound, m->step)) break;
          size_t length, offset = piece_at(m, at, &length);
          if (counted(m->c) && m->c->bound->steps[m->step].buffer == 1) {
            size_t valid;
            const size_t place = counted_place(m->c, (int)m->c->bound->steps[m->step].peer, 0, offset, length, &valid);
            supply(ch, r, (const unsigned char *)m->c->send + place, valid);
          } else supply(ch, r, m->c->bound->host[m->c->bound->steps[m->step].buffer] + m->offset + offset, length);
          if (at + 1 == m->first + m->pieces) p->out_at++;
        } else supply(ch, r, NULL, 0);
        ch->sent++;
        moved = 1;
      }
      if (!waiting && (ch->sent < ch->end || ch->consumed < ch->end)) waiting = ch;
      open |= ch->sent < ch->end || ch->consumed < ch->end;
    }
    for (uint32_t h = 0; !open && h < comm->count; h++) comm->channels[h].waited = comm->channels[h].end;
    if (!open) return ncclSuccess;
    const uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if (moved) since = now;
    else if (now - since > MESH_PLATFORM_NS)
      return fail(ncclRemoteError, "nothing landed from node %u at position %llu, and nothing could be published, for the platform's deadline "
                  "(MESH_PLATFORM_NS): its peer has not reached the group", waiting->node, (unsigned long long)waiting->consumed);
  }
}

/* the call's contribution into its operand */
static void stage(const struct call *c, unsigned char *slot) {
  const size_t z = SIZE[c->type], n = c->count;
  if (reducing(c->what)) {
    const size_t all = elements(c);
    if (c->premultiplied) premultiply(c->type, slot, c->send, c->pre, all);
    else memmove(slot, c->send, all * z);
  } else if (c->what == MESH_BROADCAST && c->root == c->comm->rank) memmove(slot, c->send, n * z);
  else if (c->what == MESH_ALLGATHER) {
    size_t length;
    const size_t before = segment_at(c, c->comm->rank, &length);
    memmove(slot + before * z, c->send, length * z);
  }
}

/* the result into recvbuff */
static void finish(const struct call *c) {
  const size_t z = SIZE[c->type], n = c->count;
  const int nranks = c->comm->nranks;
  if (!c->result) return;
  if (c->what == MESH_ALLREDUCE || (c->what == MESH_REDUCE && c->root == c->comm->rank)) {
    if (c->post) truncdiv(c->type, c->recv, c->result, n, (uint64_t)nranks);
    else memmove(c->recv, c->result, n * z);
  } else if (c->what == MESH_REDUCE_SCATTER) {
    size_t length;
    const size_t before = segment_at(c, c->comm->rank, &length);
    if (c->post) truncdiv(c->type, c->recv, c->result + before * z, length, (uint64_t)nranks);
    else memmove(c->recv, c->result + before * z, length * z);
  } else if (c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER || c->what == WHAT_RECV)
    memmove(c->recv, c->result, elements(c) * z);
}

/* an operator as the host runs it (premultiplier, combine, postoperation; the partials merge's row) */
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
  memcpy(c->pre, made->value, SIZE[c->type]);
  c->premultiplied = 1;
  return ncclSuccess;
}

/* a counted all-to-all's landed counts on the host path: every rank's segment for this rank in rank order (recvsegs),
   this rank's own copied from its counts, then the tail: the rows before its own segment in its counts, the rows of all
   its counts, an overflow word (nccl-mesh-metal.h METAL_COUNTED_START's on the GPU) */
static int64_t *counted_got(const struct call *c) {
  const int n = c->comm->nranks, me = c->comm->rank;
  const uint64_t *sseg = c->segments, *rseg = c->segments + n, R = seg_first(rseg, n), sent = seg_first(sseg, n);
  const int64_t *counts = c->counts;
  int64_t *got = calloc(R + 3, sizeof *got);
  if (!got) return NULL;
  memcpy(got + seg_first(rseg, me), counts + seg_first(sseg, me), sseg[me] * sizeof *got);
  got[R] = (int64_t)counted_rows(counts, 0, seg_first(sseg, me));
  got[R + 1] = (int64_t)counted_rows(counts, 0, sent);
  return got;
}

/* a hosted all-to-all's own segment: its byte offsets in the send and the receive buffer, and its bytes */
static size_t own_segment(const struct call *c, size_t *sent, size_t *landed) {
  const int n = c->comm->nranks, me = c->comm->rank;
  *sent = seg_first(c->segments, me) * SIZE[c->type]; *landed = seg_first(c->segments + n, me) * SIZE[c->type];
  return c->segments[me] * SIZE[c->type];
}

/* a counted all-to-all's end on the host path, once its counts landed: its own rows into their place, the landed counts
   into `received` and `landed`, then its word, 1, or 2 and ncclInvalidUsage where its counts pass its layout (a rank's
   rows past sendrows, a pair's past the bound, the received rows past capacity: those past it not moved); a hosted
   one's own segment copied */
static ncclResult_t counted_host_end(struct call *c) {
  size_t sent, landed, own = hosted(c) ? own_segment(c, &sent, &landed) : 0;
  if (own) memmove((unsigned char *)c->recv + landed, (const unsigned char *)c->send + sent, own);
  if (hosted(c)) return ncclSuccess;
  const int n = c->comm->nranks, me = c->comm->rank;
  const uint64_t *rseg = c->segments + n, R = seg_first(rseg, n);
  const size_t row = row_bytes(c), limit = c->capacity * row, from = (size_t)c->got[R] * row;
  int over = (uint64_t)c->got[R + 1] > c->rows || counted_rows(c->got, 0, R) > c->capacity;
  for (int q = 0; q < n; q++) over |= q != me && counted_rows(c->got, seg_first(rseg, q), rseg[q]) > counted_bound(c);
  size_t valid;
  const size_t to = counted_span(c->got, seg_first(rseg, me), rseg[me], row, 0, (size_t)-1, limit, &valid);
  if (from < c->rows * row) memmove((unsigned char *)c->recv + to, (const unsigned char *)c->send + from, valid < c->rows * row - from ? valid : c->rows * row - from);
  memcpy(c->received, c->got, R * sizeof *c->got);
  if (c->landed) memcpy(c->landed, c->got, R * sizeof *c->got);
  uint64_t none = 0;
  __atomic_compare_exchange_n(c->arrived, &none, over ? 2 : 1, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  return over ? fail(ncclInvalidUsage, "a counted all-to-all's counts passed its layout (a rank's rows past sendrows %zu, a pair's past %zu, "
                     "or the received rows past capacity %zu): those past it were not moved", c->rows, counted_bound(c), c->capacity)
              : ncclSuccess;
}

/* the counted all-to-alls of a group that failed or was lost: each word still 0 set to 2 (the host's wait on it ends) */
static void counted_failed(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (counted(list + i) && list[i].arrived) {
      uint64_t none = 0;
      __atomic_compare_exchange_n(list[i].arrived, &none, 2, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
}

/* one group on the host: its leases' GPU points reached, its communicator ready (comm_ready) and its GPU work finished,
   each within the platform's deadline; its calls planned and staged (a SEND read and a receive written in place), its
   lanes declared (schedule), run with the groups' lock released (run), then finished; a run's failure the
   communicator's (comm_fail: its channels stand mid-group) */
static int comm_issued(const struct ncclComm *comm);
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
  if (revoked(comm)) return revoked_fail(comm);
  const uint64_t until = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + MESH_PLATFORM_NS;
  int late = 0;
  for (size_t i = 0; i < n && !late; i++) {
    const struct call *c = list + i;
    late = lease_wait(c->send, 0, until) || lease_wait(c->recv, 1, until) ||
           (counted(c) && (lease_wait(c->counts, 0, until) || lease_wait(c->received, 1, until) || lease_wait(c->landed, 1, until)));
  }
  if (late) return fail(ncclSystemError, "a lease's GPU work (ncclMeshMemUse) did not reach its point within the platform's deadline");
  pthread_mutex_lock(&group_lock);
  if (remote) result = comm_ready(comm);
  if (!result && remote && comm_issued(comm))
    result = fail(ncclInvalidUsage, "a host-path group while this communicator's GPU groups are issued and not complete (ncclMeshComplete)");
  if (!result && remote && finished && comm->last && !gpu_reached(finished, comm->last, until))
    result = fail(ncclSystemError, "communicator %s's GPU work did not finish within the platform's deadline before its host-path group", comm->name);
  if (!result && remote && finished) metal_collect(finished);
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
    else if (hosted(c)) { if (s) { s->host[0] = (unsigned char *)(uintptr_t)c->send; s->host[1] = c->recv; } }
    else if (counted(c)) {
      if (!(c->got = counted_got(c))) result = fail(ncclSystemError, "out of memory");
      else if (s) {
        s->host[0] = (unsigned char *)(uintptr_t)c->counts; s->host[1] = (unsigned char *)(uintptr_t)c->send;
        s->host[2] = (unsigned char *)c->got; s->host[3] = c->recv;
      }
    } else if (!(c->result = calloc(bytes ? bytes : 1, 1))) result = fail(ncclSystemError, "out of memory");
    else {
      c->owned = 1;
      stage(c, c->result);
      if (s) s->host[0] = c->result;
      for (uint32_t b = 1; s && b < s->buffers && !result; b++)
        if (!(s->host[b] = malloc(bytes ? bytes : 1))) result = fail(ncclSystemError, "out of memory");
        else memcpy(s->host[b], c->result, bytes);
    }
  }
  struct schedule *plans = NULL;
  if (!result && remote && !(result = schedule(comm, list, n, &plans))) atomic_fetch_add(&comm->runs, 1);
  pthread_mutex_unlock(&group_lock);
  if (plans) {
    if ((result = run(comm, plans))) {
      pthread_mutex_lock(&group_lock);
      comm_fail(comm, result, "%s", last);
      pthread_mutex_unlock(&group_lock);
    }
    schedules_free(comm, plans);
    atomic_fetch_sub(&comm->runs, 1);
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
      const ncclResult_t ended = counted(c) ? counted_host_end(c) : ncclSuccess;
      if (!result) result = ended;
    }
  }
  if (result) counted_failed(list, n);
  return result;
}

/* -- the Metal path ------------------------------------------------------------------------------------ */

static ncclResult_t gpu(int status, const char *what) {
  return status ? fail(ncclSystemError, "%s on the GPU: %s", what, *metal_error() ? metal_error() : strerror(status)) : ncclSuccess;
}

/* one of this rank's publications in its group's order (order_group): its level, channel and position */
struct event { uint64_t level; uint32_t h; uint64_t at; };

/* a group issued and not complete: its communicator, its calls, its schedule and its positions [start, end) on each of
   its communicator's channels, its publications in order (`next` the first the walk has not completed); the groups in
   flight are in issue order, a communicator's holding consecutive positions on every channel of it */
struct inflight {
  uint64_t ticket;
  struct ncclComm *comm;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t *start, *end;
  struct event *order;
  size_t events, next;
  struct inflight *next_group;
};
static struct inflight *inflight, *inflight_last;
static uint64_t tickets;
static _Atomic uint64_t retired;
/* the groups lost with their communicators, each with its communicator's failure, until a completion through it
   reports it */
static struct loss { uint64_t ticket; ncclResult_t result; char reason[256]; } *losses;
static size_t lost;

static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    steps_free(list[i].bound);
    if (list[i].owned) free(list[i].result);
    free(list[i].copy);
    free(list[i].segments);
    free(list[i].got);
    free(list[i].complement);
  }
  free(list);
}

static void inflight_free(struct inflight *g) {
  schedules_free(g->comm, g->plans);
  free(g->start);
  free(g->order);
  calls_free(g->list, g->n);
  free(g);
}

/* whether a group of comm is in flight */
static int comm_issued(const struct ncclComm *comm) {
  for (const struct inflight *g = inflight; g; g = g->next_group)
    if (g->comm == comm) return 1;
  return 0;
}

/* every group in flight on comm lost: a later completion through one returns comm's failure (or its destroy); every
   earlier group is complete or lost */
static void inflight_lose(struct ncclComm *comm) {
  inflight_last = NULL;
  for (struct inflight **at = &inflight; *at;) {
    struct inflight *g = *at;
    if (g->comm != comm) { inflight_last = g; at = &g->next_group; continue; }
    struct loss *grown = realloc(losses, (lost + 1) * sizeof *grown);
    if (grown) {
      losses = grown;
      losses[lost] = (struct loss){g->ticket, comm->failed ? comm->failed : ncclRemoteError, {0}};
      snprintf(losses[lost++].reason, sizeof grown->reason, "%s", comm->failed ? comm->failure : "its communicator was destroyed");
    }
    *at = g->next_group;
    counted_failed(g->list, g->n);
    inflight_free(g);
  }
  atomic_store(&retired, inflight ? inflight->ticket - 1 : tickets);
}

static struct inflight *inflight_at(struct ncclComm *comm, uint32_t h, uint64_t at) {
  for (struct inflight *g = inflight; g; g = g->next_group)
    if (g->comm == comm && at < g->end[h]) return at >= g->start[h] ? g : NULL;
  return NULL;
}

/* channel h's position `at`'s in run as a GPU wait awaits it (nccl-mesh-metal.h struct metal_wait): its last unit's
   landed word, its first unit's slot (a dud's check, and a landing's piece), its communicator's failure words */
static struct metal_wait wait_of(const struct ncclComm *comm, uint32_t h, const struct run *r, uint64_t at) {
  const struct channel *ch = comm->channels + h;
  const uint64_t units = ch->view.units, last = r->first + r->units - 1;
  return (struct metal_wait){ch->buffer, comm->violation, ch->landed + (last % units) * sizeof(uint64_t),
                             ch->in + (r->first % units) * ch->view.unit, last / units, at, h};
}
/* a channel's out run as a release stores it (struct metal_release): its first unit's slot and its units' ready words,
   or, `skipped`, the units it skips before its lap */
static struct metal_release release_of(const struct channel *ch, const struct run *r, int skipped) {
  const uint64_t units = ch->view.units, first = skipped ? r->first - r->skip : r->first;
  return (struct metal_release){ch->buffer, ch->ready + (first % units) * sizeof(uint64_t), ch->out + (r->first % units) * ch->view.unit,
                                1 + first / units, skipped ? r->skip : r->units};
}

/* the GPU waits until `count` of the channel's positions have landed: a wait on each one's in run not yet awaited, in
   order (its landing checked for a dud) */
static ncclResult_t arrive(struct metal_program *program, struct ncclComm *comm, struct channel *ch, uint64_t count) {
  const uint32_t h = (uint32_t)(ch - comm->channels);
  ncclResult_t result = ncclSuccess;
  for (uint64_t at = ch->waited; at < count && !result; at++) {
    const struct inflight *g = inflight_at(comm, h, at);
    if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
    ch->waited = at + 1;
    const struct metal_wait wait = wait_of(comm, h, g->plans[h].runs + 2 * (at - g->start[h]) + 1, at);
    result = gpu(metal_spin(program, &wait), "a wait");
  }
  return result;
}

static int event_order(const void *x, const void *y) {
  const struct event *a = x, *b = y;
  if (a->level != b->level) return a->level < b->level ? -1 : 1;
  if (a->h != b->h) return a->h < b->h ? -1 : 1;
  return (a->at > b->at) - (a->at < b->at);
}

/* the group's publications in the one order every rank's walk keeps (MSCCLang's: a topological order of the dependency
   graph over every rank, so no rank's wait is on what another encodes after a wait of its own).  Every rank's plan of
   each collective laid out on every channel (mesh_group_lay), each end's publication of each position and its
   consumption are the nodes: a publication after its channel's previous one, the peer's publication LAG positions back
   (whose landing frees the units it fills) and its own consumption of that position, a SEND's first piece after the
   consumption of the receive before it; a consumption after its channel's previous one, the peer's publication where a
   piece lands, a receive's first piece after the receive before it.  A publication's level is one more than its
   predecessors' largest, a consumption's that largest.  This rank's publications go to g->order by level; the walk
   encodes a consumption only where a publication (or the group's end) needs it, so a wait is on a publication of a level
   below the next one encoded, and the lowest level not yet published always can be */
static ncclResult_t order_group(ncclComm_t comm, struct inflight *g) {
  const int N = comm->nranks, me = comm->rank;
  struct call *list = g->list;
  size_t S = 2;
  for (size_t i = 0; i < g->n; i++) {
    if (list[i].bound && list[i].bound->program && list[i].bound->program->first[N] + 1 > S) S = list[i].bound->program->first[N] + 1;
    if (list[i].bound && list[i].bound->count + 1 > S) S = list[i].bound->count + 1;
  }
  const size_t n = g->n, E = (size_t)N * (N - 1);
  const struct mesh_step **steps = calloc((size_t)N * n + 1, sizeof *steps);
  struct mesh_step *planned = calloc((size_t)N * n * S + 1, sizeof *planned);
  uint32_t *counts = calloc((size_t)N * n + 1, sizeof *counts), *lane = calloc((size_t)N * n * S + 1, sizeof *lane);
  uint64_t *first = calloc((size_t)N * n * S + 1, sizeof *first), *pieces = calloc((size_t)N * n * S + 1, sizeof *pieces);
  uint64_t *length = calloc(2 * E + 1, sizeof *length), *offset = calloc(E + 1, sizeof *offset);
  struct mesh_group_call *calls = calloc(n + 1, sizeof *calls);
  struct mesh_group_place *made = calloc(n * S + 1, sizeof *made);
  struct mesh_place *scratch = calloc(S + 1, sizeof *scratch);
  int32_t *before = calloc((size_t)N * n * S + 1, sizeof *before);
  struct mesh_group group;
  ncclResult_t result = steps && planned && counts && lane && first && pieces && length && offset && calls && made && scratch && before
                          ? ncclSuccess : fail(ncclSystemError, "out of memory");
  /* a channel: the pair a < b (its index among the pairs) and the class; an end's side 0 for a, 1 for b; rank r's steps
     of call i at steps[i * N + r] */
#define CHANNEL(a, b, large) ((((size_t)(a) * (2 * N - (a) - 1) / 2 + ((b) - (a) - 1))) * 2 + (size_t)(large))
#define AT(r, i, k) (((size_t)(r) * n + (i)) * S + (k))
#define STEPS(r, i) ((size_t)(i) * N + (size_t)(r))
  for (int r = 0; r < N && !result; r++)
    for (size_t i = 0; i < n && !result; i++) {
      const struct steps *own = list[i].bound;
      if (!own) continue;
      if (r == me) { steps[STEPS(r, i)] = own->steps; counts[STEPS(r, i)] = own->count; }
      else if (!p2p(list + i) && !counted(list + i)) {
        steps[STEPS(r, i)] = planned + AT(r, i, 0);
        counts[STEPS(r, i)] = mesh_program_steps_cut(own->program, (uint32_t)r, own->operand, own->segments, planned + AT(r, i, 0));
      }
    }
  if (!result) group_of(list, n, (uint32_t)N, steps, counts, calls, &group);
  for (int r = 0; r < N && !result; r++)
    for (int q = 0; q < N; q++)
      for (int large = 0; q != r && large < 2; large++) {
        uint32_t placed = 0;
        const size_t e = r < q ? CHANNEL(r, q, large) : CHANNEL(q, r, large);
        length[2 * e + (r > q)] = mesh_group_lay(&group, (uint32_t)r, (uint32_t)q, large, &comm->wire, 0, scratch, made, &placed);
        for (uint32_t m = 0; m < placed; m++) {
          const size_t x = AT(r, made[m].call, made[m].place.step);
          lane[x] = (uint32_t)e; first[x] = made[m].place.first; pieces[x] = made[m].place.pieces;
        }
      }
  for (int r = 0; r < N && !result; r++)
    for (size_t i = 0; i < n && !result; i++)
      if (counts[STEPS(r, i)] && mesh_steps_before(steps[STEPS(r, i)], counts[STEPS(r, i)], before + AT(r, i, 0)))
        result = fail(ncclSystemError, "out of memory");
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
      for (uint32_t k = 0; k < counts[STEPS(r, i)]; k++) {
        if (!pieces[AT(r, i, k)]) continue;
        const struct mesh_step *own = steps[STEPS(r, i)];
        const size_t e = lane[AT(r, i, k)], side = (size_t)(r > (int)own[k].peer), at = offset[e] + first[AT(r, i, k)];
        const int send = own[k].op == MESH_STEP_SEND;
        const int32_t j = before[AT(r, i, k)];
        uint64_t wait = 0;
        if (j >= 0) {
          const size_t ej = lane[AT(r, i, j)], sj = (size_t)(r > (int)own[j].peer);
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
  for (uint32_t h = 0; h < comm->count; h++) events += g->end[h] - g->start[h];
  if (!result && !(g->order = calloc(events + 1, sizeof *g->order))) result = fail(ncclSystemError, "out of memory");
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    const struct channel *ch = comm->channels + h;
    const int q = ch->peer;
    if (g->end[h] == g->start[h]) continue;
    const size_t e = me < q ? CHANNEL(me, q, ch->large) : CHANNEL(q, me, ch->large), side = (size_t)(me > q);
    if (length[2 * e + side] != g->end[h] - g->start[h]) { result = fail(ncclInternalError, "the group's positions on channel %u disagree", h); break; }
    for (uint64_t t = 0; t < g->end[h] - g->start[h]; t++) g->order[g->events++] = (struct event){level[NODE(e, t, side)], h, g->start[h] + t};
  }
#undef NODE
#undef STEPS
#undef AT
#undef CHANNEL
  if (!result) qsort(g->order, g->events, sizeof *g->order, event_order);
  free(steps); free(planned); free(counts); free(lane); free(first); free(pieces); free(length); free(offset); free(calls); free(made); free(scratch);
  free(before); free(after_send); free(after_receive); free(sends); free(lands); free(pred); free(level); free(queue); free(degree); free(successors);
  free(edge);
  return result;
}

/* a counted all-to-all's end on the GPU, once every count landed (its counts receives consumed; at once where it has
   none): its own rows into place, the landed counts into `received` (stored system-coherent: the host reads them as they
   are, mesh 3981ca80^'s `received`) and `landed`, then its word released (1, or 2 where the counts pass its layout,
   which also fails the communicator through failure word [5]; nccl-mesh-metal.h METAL_COUNTED_ARRIVED), the rows maybe
   still on the wire; `violation` the communicator's failure words (NULL: none) */
static ncclResult_t counted_end(struct metal_program *program, struct call *c, void *violation) {
  if (c->ended) return ncclSuccess;
  c->ended = 1;
  const int n = c->comm->nranks, me = c->comm->rank;
  const uint64_t *rseg = c->segments + n, R = seg_first(rseg, n);
  const size_t row = row_bytes(c);
  ncclResult_t result = gpu(metal_counted(program, METAL_COUNTED_OWN, &(struct metal_counted){
    .d = c->recv_at.buffer, .dst = c->recv_at.offset, .s = c->send_at.buffer, .src = c->send_at.offset, .a = c->got_at.buffer,
    .aux = c->got_at.offset, .first = seg_first(rseg, me), .entries = (uint32_t)rseg[me], .scalar = R, .row = row, .limit = c->capacity * row,
    .bytes = c->rows * row}), "a counted all-to-all's own rows");
  if (!result) result = gpu(metal_copy(program, METAL_SEND, c->received_at.buffer, c->received_at.offset, c->got_at.buffer, c->got_at.offset,
                                       R * sizeof(int64_t)), "a counted all-to-all's received counts");
  if (!result && c->landed_at.buffer)
    result = gpu(metal_copy(program, METAL_PLAIN, c->landed_at.buffer, c->landed_at.offset, c->got_at.buffer, c->got_at.offset, R * sizeof(int64_t)),
                 "a counted all-to-all's landed counts");
  if (!result)
    result = gpu(metal_counted(program, METAL_COUNTED_ARRIVED, &(struct metal_counted){
      .d = c->arrived_at.buffer, .dst = c->arrived_at.offset, .s = c->got_at.buffer, .src = c->got_at.offset, .a = violation, .bytes = R,
      .first = R, .aux = c->capacity, .scalar = c->rows}), "a counted all-to-all's word");
  return result;
}

/* a counted all-to-all's staging on the GPU: its landed counts a scratch of the group's (every rank's segment for this
   rank in rank order, then the tail), its own segment and tail written there (METAL_COUNTED_START), its buffers the steps'
   (0 its counts, 1 its rows, 2 the landed counts, 3 the receive buffer); its end at once where no counts land.  A hosted
   one's buffers are its send and receive buffers, and its own segment is copied */
static void *workspace(struct metal_program *program, const struct call *c, size_t bytes);
static ncclResult_t counted_stage(struct metal_program *program, struct call *c, void *violation) {
  size_t sent, landed, own = hosted(c) ? own_segment(c, &sent, &landed) : 0;
  if (hosted(c) && c->bound) { c->bound->gpu[0] = c->send_at; c->bound->gpu[1] = c->recv_at; }
  if (own && !same((struct where){c->send_at.buffer, c->send_at.offset + sent}, (struct where){c->recv_at.buffer, c->recv_at.offset + landed}))
    return gpu(metal_copy(program, METAL_PLAIN, c->recv_at.buffer, c->recv_at.offset + landed, c->send_at.buffer, c->send_at.offset + sent, own),
               "an all-to-all's own segment");
  if (hosted(c)) return ncclSuccess;
  const int n = c->comm->nranks, me = c->comm->rank;
  const uint64_t *sseg = c->segments, *rseg = c->segments + n, R = seg_first(rseg, n);
  void *got = workspace(program, c, (R + 3) * sizeof(int64_t));
  if (!got) return fail(ncclSystemError, "out of GPU memory");
  c->got_at = (struct where){got, 0};
  struct steps *st = c->bound;
  if (st) { st->gpu[0] = c->counts_at; st->gpu[1] = c->send_at; st->gpu[2] = c->got_at; st->gpu[3] = c->recv_at; }
  const ncclResult_t result = gpu(metal_counted(program, METAL_COUNTED_START, &(struct metal_counted){
    .d = got, .dst = seg_first(rseg, me) * sizeof(int64_t), .s = c->counts_at.buffer, .src = c->counts_at.offset, .aux = R * sizeof(int64_t),
    .bytes = sseg[me], .first = seg_first(sseg, me), .scalar = seg_first(sseg, n)}), "a counted all-to-all's counts");
  return result || (st && c->counting) ? result : counted_end(program, c, violation);
}

/* a counted all-to-all's counts from rank q consumed: their rows checked against the pair's bound (METAL_COUNTED_CHECK),
   and its end once every count landed */
static ncclResult_t counted_landed(struct metal_program *program, struct call *c, int q, void *violation) {
  const int n = c->comm->nranks;
  const uint64_t *rseg = c->segments + n, R = seg_first(rseg, n);
  ncclResult_t result = gpu(metal_counted(program, METAL_COUNTED_CHECK, &(struct metal_counted){
    .d = c->got_at.buffer, .dst = c->got_at.offset + R * sizeof(int64_t), .first = seg_first(rseg, q), .entries = (uint32_t)rseg[q],
    .scalar = counted_bound(c)}), "a counted all-to-all's bound");
  if (!result && c->counting && !--c->counting) result = counted_end(program, c, violation);
  return result;
}

/* the fused call of the group being encoded (nccl.h ncclMeshFusable): its publication's and its landing's records,
   written in place of their dispatches, and whether each was; under group_lock, for one launch_metal */
static struct { ncclMeshFused *out; const struct call *call; int published, landed, publishes, lands; } fusing;

/* whether message m's piece at its channel's position (`offset`, `length`) is the fused call's whole operand */
static int fused_piece(const struct message *m, size_t offset, size_t length) {
  return fusing.out && m->c == fusing.call && !m->offset && !offset && length == m->bytes && m->bytes == elements(m->c) * SIZE[m->c->type];
}

/* channel h's positions consumed up to `end`, in order: a piece's landing awaited (its run's last unit's, a dud
   checked) and combined or copied where it belongs (an empty position needs nothing), a receive's first piece after the
   receive before it; the fused call's landing of its whole operand recorded for its caller's kernel
   (metal_land_deferred) */
static ncclResult_t consume(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t end) {
  struct channel *ch = comm->channels + h;
  ncclResult_t result = ncclSuccess;
  while (ch->consumed < end && !result) {
    const uint64_t at = ch->consumed;
    struct inflight *g = inflight_at(comm, h, at);
    if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
    struct schedule *p = g->plans + h;
    struct message *m = p->in_at < p->ins && at >= p->in[p->in_at].first ? p->in + p->in_at : NULL;
    if (m && at == m->first && m->c->bound->before[m->step] >= 0) {
      const struct steps *prior = m->c->bound;
      const int32_t j = prior->before[m->step];
      result = consume(program, comm, prior->channel[j], prior->first[j] + prior->pieces[j]);
      if (result) break;
    }
    if (m) {
      struct steps *st = m->c->bound;
      size_t length, offset = piece_at(m, at, &length);
      const struct metal_wait wait = wait_of(comm, h, p->runs + 2 * (at - p->start) + 1, at);
      const size_t from = wait.slot;
      const int reduce = st->steps[m->step].op == MESH_STEP_REDUCE;
      const struct where to = st->gpu[st->steps[m->step].buffer];
      const int q = (int)st->steps[m->step].peer, n = m->c->comm->nranks;
      if (counted(m->c) && st->steps[m->step].buffer == 3) {
        const uint64_t *rseg = m->c->segments + n;
        result = arrive(program, comm, ch, at + 1);
        if (!result)
          result = gpu(metal_counted(program, METAL_COUNTED_LAND, &(struct metal_counted){
            .d = to.buffer, .dst = to.offset, .s = ch->buffer, .src = from, .bytes = length, .scalar = offset, .a = m->c->got_at.buffer,
            .aux = m->c->got_at.offset, .first = seg_first(rseg, q), .entries = (uint32_t)rseg[q], .row = row_bytes(m->c),
            .limit = m->c->capacity * row_bytes(m->c)}), "a counted landing");
      } else if (st->fresh && st->fresh[m->step]) {
        const struct where with = st->read[m->step];
        result = arrive(program, comm, ch, at + 1);
        if (!result)
          result = gpu(metal_combine3(program, m->c->type, m->c->combine, to.buffer, to.offset + m->offset + offset, ch->buffer, from, with.buffer,
                                      with.offset + m->offset + offset, length / SIZE[m->c->type]), "a first combine");
      } else if (!ch->large) {
        if (fusing.lands && !fusing.landed && fused_piece(m, offset, length)) {
          result = gpu(metal_land_deferred(program, m->c->combine, to.buffer, to.offset, reduce ? length / SIZE[m->c->type] : length, &wait,
                                           fusing.out->land), "a fused landing's record");
          fusing.landed = !result;
        } else
          result = gpu(metal_land(program, reduce ? ktype(m->c) : -1, kop(m->c), to.buffer, to.offset + m->offset + offset,
                                  reduce ? kcount(m->c, length) : length, &wait), "a landing");
        if (at + 1 > ch->waited) ch->waited = at + 1;
      } else {
        result = arrive(program, comm, ch, at + 1);
        if (!result && reduce)
          result = gpu(metal_combine(program, ktype(m->c), kop(m->c), to.buffer, to.offset + m->offset + offset, ch->buffer, from,
                                     kcount(m->c, length)), "a combine");
        else if (!result)
          result = gpu(metal_copy(program, METAL_LAND, to.buffer, to.offset + m->offset + offset, ch->buffer, from, length), "a landing");
      }
      if (result) break;
      if (at + 1 == m->first + m->pieces) {
        st->done[m->step] = 1; p->in_at++;
        if (counted(m->c) && st->steps[m->step].buffer == 2) result = counted_landed(program, m->c, q, comm->violation);
      }
    }
    ch->consumed++;
  }
  return result;
}

/* channel h's next position published, after the landing of the peer's position LAG back, which frees the units its run
   takes: the units its run skips released as fillers, then its piece copied into its run's slots and its units
   released, or a filler's (an empty position); the fused call's whole operand recorded for its caller's kernel, which
   stores it and releases its units (metal_send_deferred) */
static ncclResult_t publish(struct metal_program *program, struct ncclComm *comm, uint32_t h, int *published) {
  struct channel *ch = comm->channels + h;
  const uint64_t at = ch->sent, need = at >= LAG ? at - LAG + 1 : 0;
  struct inflight *g = inflight_at(comm, h, at);
  if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
  struct schedule *p = g->plans + h;
  struct message *m = p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
  const struct run *r = p->runs + 2 * (at - p->start);
  const struct metal_release out = release_of(ch, r, 0), skipped = release_of(ch, r, 1);
  ncclResult_t result = arrive(program, comm, ch, need);
  if (!result && r->skip) result = gpu(metal_publish(program, &skipped), "a lap's skipped units");
  int released = 0;
  if (m && !result) {
    const struct steps *st = m->c->bound;
    size_t length, offset = piece_at(m, at, &length);
    if (counted(m->c) && st->steps[m->step].buffer == 1) {
      const uint64_t *sseg = m->c->segments;
      const int q = (int)st->steps[m->step].peer;
      result = gpu(metal_counted(program, METAL_COUNTED_SEND, &(struct metal_counted){
        .d = ch->buffer, .dst = out.slot, .s = m->c->send_at.buffer, .src = m->c->send_at.offset, .bytes = length, .scalar = offset,
        .a = m->c->counts_at.buffer, .aux = m->c->counts_at.offset, .first = seg_first(sseg, q), .entries = (uint32_t)sseg[q],
        .row = row_bytes(m->c), .limit = m->c->rows * row_bytes(m->c)}), "a counted piece into its slots");
    } else if (!ch->large && fusing.publishes && !fusing.published && fused_piece(m, offset, length)) {
      result = gpu(metal_send_deferred(program, length, &out, fusing.out->publish), "a fused publication's record");
      fusing.published = !result;
      released = 1;
    } else if (!ch->large) {
      const struct where source = st->read && st->read[m->step].buffer ? st->read[m->step] : st->gpu[st->steps[m->step].buffer];
      result = gpu(metal_send_small(program, source.buffer, source.offset + m->offset + offset, length, &out), "a piece into its slots and its release");
      released = 1;
    } else {
      const struct where source = st->read && st->read[m->step].buffer ? st->read[m->step] : st->gpu[st->steps[m->step].buffer];
      result = gpu(metal_copy(program, METAL_SEND, ch->buffer, out.slot, source.buffer, source.offset + m->offset + offset, length),
                   "a piece into its slots");
    }
    if (at + 1 == m->first + m->pieces) p->out_at++;
  }
  if (!result && !released && m) result = gpu(metal_publish(program, &out), "a publication");
  else if (!result && !released) result = gpu(metal_publish_filler(program, &out), "an empty publication");
  if (!result) { ch->sent++; *published = 1; }
  return result;
}

static struct message *sending(struct inflight *g, uint32_t h, uint64_t at) {
  struct schedule *p = g->plans + h;
  return p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
}

/* the communicators a program touched, each once, until its end (program_end) */
static struct ncclComm *touching;
static void touch(struct ncclComm *comm) {
  if (comm->touched) return;
  comm->touched = 1;
  comm->next_touched = touching;
  touching = comm;
}

/* a group completed: its publications in its order, each after the consumptions it needs (the positions LAG back on
   its channel; a SEND's first piece, the receive before it), then every position of it consumed */
static ncclResult_t complete_group(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
  metal_gate(program, comm->violation);
  for (; g->next < g->events && !result; g->next++) {
    const struct event v = g->order[g->next];
    const struct channel *ch = comm->channels + v.h;
    if (v.at < ch->sent) continue;
    if (v.at != ch->sent) return fail(ncclInternalError, "the group's positions admit no order");
    result = consume(program, comm, v.h, v.at >= LAG ? v.at - LAG + 1 : 0);
    const struct message *m = result ? NULL : sending(g, v.h, v.at);
    if (m && v.at == m->first && m->c->bound->before[m->step] >= 0) {
      const struct steps *prior = m->c->bound;
      const int32_t j = prior->before[m->step];
      result = consume(program, comm, prior->channel[j], prior->first[j] + prior->pieces[j]);
    }
    if (!result) result = publish(program, comm, v.h, published);
  }
  for (uint32_t h = 0; h < comm->count && !result; h++) result = consume(program, comm, h, g->end[h]);
  return result;
}

/* a group's publications that need no wait, in its order: its channel's earlier positions published, the positions LAG
   back consumed (empty ones freely) and the peer's landing there awaited before, a SEND's receives before it consumed */
static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
  metal_gate(program, comm->violation);
  for (size_t e = g->next; e < g->events && !result; e++) {
    const struct event v = g->order[e];
    struct channel *ch = comm->channels + v.h;
    const uint64_t need = v.at >= LAG ? v.at - LAG + 1 : 0;
    if (v.at != ch->sent || need > ch->waited) continue;
    while (ch->consumed < need) {
      struct inflight *at = inflight_at(comm, v.h, ch->consumed);
      struct schedule *p = at ? at->plans + v.h : NULL;
      if (!p || (p->in_at < p->ins && ch->consumed >= p->in[p->in_at].first)) break;
      ch->consumed++;
    }
    const struct message *m = sending(g, v.h, v.at);
    if (ch->consumed < need || (m && v.at == m->first && !ready(m->c->bound, m->step))) continue;
    result = publish(program, comm, v.h, published);
  }
  return result;
}

/* the groups in flight, in the order every rank's walk keeps (order_group): completing through a group, each group up
   to it completed, then any later group's publications that need no wait; issuing a group, its publications that need
   no wait, so a group issued encodes no wait at all; each group's dispatches gated on its own communicator's failure
   words (metal_gate) */
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
static struct where where_of(const void *argument) {
  const ncclMeshBuffer *b = argument;
  return b ? (struct where){b->buffer, b->offset} : (struct where){NULL, 0};
}

/* a group's scratch: the stream's workspace (the caller's allocator, ordered with its work), else a buffer the
   program keeps until its end */
static void *workspace(struct metal_program *program, const struct call *c, size_t bytes) {
  if (c->workspace) return c->workspace(bytes ? bytes : 16, c->context);
  if (metal_metal4(program)) {
    /* the Metal 4 form, which signals nothing at its end: the communicator's scratch, the group's k-th request its k-th
       buffer, grown where short (the one it replaces kept until the communicator's end, as the GPU may yet read it);
       its groups run in the encoder's order, each complete at its end, so one group's scratch is free for the next */
    struct ncclComm *comm = c->comm;
    const uint32_t k = comm->scratch_used++;
    if (k == comm->scratches) {
      void **grown = realloc(comm->scratch, (k + 1) * sizeof *grown);
      size_t *sizes = grown ? realloc(comm->scratch_bytes, (k + 1) * sizeof *sizes) : NULL;
      if (grown) comm->scratch = grown;
      if (!sizes) return NULL;
      comm->scratch_bytes = sizes;
      comm->scratch[k] = NULL; comm->scratch_bytes[k] = 0; comm->scratches++;
    }
    if (comm->scratch_bytes[k] < (bytes ? bytes : 16)) {
      void **spent = comm->scratch[k] ? realloc(comm->spent, (comm->spents + 1) * sizeof *spent) : comm->spent;
      void *made = (comm->scratch[k] && !spent) ? NULL : metal_scratch(bytes);
      if (!made) return NULL;
      if (comm->scratch[k]) { comm->spent = spent; comm->spent[comm->spents++] = comm->scratch[k]; }
      comm->scratch[k] = made; comm->scratch_bytes[k] = bytes ? bytes : 16;
    }
    metal_resident(program, comm->scratch[k]);
    return comm->scratch[k];
  }
  void *made = metal_scratch(bytes);
  if (made) { metal_keep(program, made); metal_release(made); }
  return made;
}

/* one buffer's written ranges, in elements, disjoint and in order */
struct spans { uint64_t (*span)[2]; size_t count, room; };
/* how [lo, hi) meets them: 0 none of it written, 1 all of it, -1 part of it */
static int spans_meet(const struct spans *w, uint64_t lo, uint64_t hi) {
  uint64_t covered = 0;
  for (size_t k = 0; k < w->count; k++) {
    const uint64_t a = w->span[k][0] > lo ? w->span[k][0] : lo, b = w->span[k][1] < hi ? w->span[k][1] : hi;
    if (a < b) covered += b - a;
  }
  return !covered ? 0 : covered == hi - lo ? 1 : -1;
}
/* [lo, hi) written: merged into the ranges; 0, or ENOMEM */
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

/* whether a call reads its send buffer in place on the Metal path (mesh 95967e6; rdma/NCCL.md Memory), planned into its
   steps (struct steps read, fresh, needed) and its complement: a collective that crosses a link, not issued (every read
   of the send buffer then in its own command buffer or encoder, so once that completes nothing reads it: the caller may
   free its lease), fused, premultiplied or merging rows, whose send buffer is a lease's Metal buffer (nccl.h
   ncclMeshMemBuffer) apart from its result `own`, every step's range in its buffer wholly written or wholly unwritten by
   the rank's steps before it (by round, then order: the order its walk encodes a receive before the SEND after it,
   mesh_steps_before).  Then a SEND
   of a range no earlier step wrote reads the send buffer, a receive's first combine of such a range writes buffer = send
   op piece, a scratch buffer no step writes is never made and none is copied, and the ranges of buffer 0 no step writes
   are copied from the send buffer after the publications (`complement`); else (0) the operand is staged as ever */
static int plan_direct(struct call *c, struct where own, int issued) {
  struct steps *st = c->bound;
  const size_t z = SIZE[c->type];
  uint64_t lo = 0, hi = elements(c);
  if (!st || p2p(c) || counted(c) || c->fused || c->premultiplied || unit_of(c) > 1 || issued || !st->count || !lease_of_buffer(c->send_at.buffer))
    return 0;
  if (c->what == MESH_BROADCAST && c->root != c->comm->rank) return 0;
  if (c->what == MESH_ALLGATHER) { size_t length; lo = segment_at(c, c->comm->rank, &length); hi = lo + length; }
  const struct where send = c->send_at;
  const unsigned char *a = metal_contents(send.buffer), *b = metal_contents(own.buffer);
  const size_t sent = (size_t)(hi - lo) * z, made = elements(c) * z;
  if (send.buffer == own.buffer ? send.offset < own.offset + made && own.offset < send.offset + sent
                                : a && b && a + send.offset < b + own.offset + made && b + own.offset < a + send.offset + sent)
    return 0;
  const uint32_t B = st->buffers;
  struct spans *w = calloc(B, sizeof *w);
  st->read = calloc(st->count, sizeof *st->read); st->fresh = calloc(st->count, 1); st->needed = calloc(B, 1);
  int direct = w && st->read && st->fresh && st->needed;
  const struct where map = {send.buffer, send.offset - (size_t)lo * z};
  for (uint32_t k = 0; direct && k < st->count; k++) {
    const struct mesh_step *step = st->steps + k;
    const uint64_t from = step->first, to = step->first + step->piece.elements;
    const int meet = spans_meet(w + step->buffer, from, to), inside = from >= lo && to <= hi;
    if (meet < 0 || (!meet && step->op != MESH_STEP_COPY && !inside)) direct = 0;
    else if (step->op == MESH_STEP_SEND) { if (!meet) st->read[k] = map; }
    else {
      if (step->op == MESH_STEP_REDUCE && !meet) { st->fresh[k] = 1; st->read[k] = map; }
      st->needed[step->buffer] = 1;
      if (spans_add(w + step->buffer, from, to)) direct = 0;
    }
  }
  /* the result's own ranges: a reduce-scatter's segment, nothing of a reduce off its root, else the whole */
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
        x[0] = own.offset + at * z; x[1] = send.offset + (at - lo) * z; x[2] = (end - at) * z;
      }
      if (k >= w[0].count) break;
      if (w[0].span[k][1] > at) at = w[0].span[k][1];
    }
  }
  for (uint32_t x = 0; w && x < B; x++) free(w[x].span);
  free(w);
  if (!direct) {
    free(st->read); free(st->fresh); free(st->needed); free(c->complement);
    st->read = NULL; st->fresh = st->needed = NULL; c->complement = NULL; c->complements = 0;
  }
  return direct;
}

/* a call reading its send buffer in place: its result's ranges no step wrote, copied from the send buffer */
static ncclResult_t complement_copy(struct metal_program *program, const struct call *c) {
  ncclResult_t result = ncclSuccess;
  for (uint32_t k = 0; k < c->complements && !result; k++) {
    const uint64_t *x = c->complement + 3 * k;
    result = gpu(metal_copy(program, METAL_PLAIN, c->sum.buffer, x[0], c->send_at.buffer, x[1], x[2]), "the operand's own ranges");
  }
  return result;
}

/* a premultiplier as the kernels read it: a floating type's value as float32 bits, an integer's bits */
static uint64_t scalar_bits(const struct call *c) {
  uint64_t bits = 0;
  if (floating(c->type)) { const float f = (float)load(c->type, c->pre); uint32_t u; memcpy(&u, &f, 4); return u; }
  memcpy(&bits, c->pre, SIZE[c->type]);
  return bits;
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
      size_t length = count;
      const size_t before = c->what == MESH_REDUCE_SCATTER ? segment_at(c, (int)r, &length) : 0;
      const struct where from = {sum.buffer, sum.offset + before * z};
      if (c->post) result = gpu(metal_truncdiv(program, c->type, recv.buffer, recv.offset, from.buffer, from.offset, length, (uint64_t)comm->nranks), "the result");
      else if (!same(from, recv)) result = gpu(metal_copy(program, METAL_PLAIN, recv.buffer, recv.offset, from.buffer, from.offset, length * z), "the result");
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
    for (uint32_t h = 0; h < g->comm->count; h++)
      if (g->comm->channels[h].consumed < g->end[h]) return ncclSuccess;
    metal_gate(program, g->comm->violation);
    result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g);
  }
  return result;
}

/* a program's end: where the library made its encoder and the group succeeded, its end signals `finished` at its count
   (`issued`, which a host-path group and a destroy await: a program the caller was told to commit), each communicator it
   touched keeping that count as its `last`; a borrowed encoder's, or a failed group's, signals nothing, so no wait of the
   library's is on a command buffer the caller may never commit.  Under group_lock */
static void program_end(struct metal_program *program, int borrowed, ncclResult_t result) {
  const uint64_t count = borrowed || result ? 0 : ++issued;
  metal_end(program, count ? (finished ? finished : (finished = metal_event())) : NULL, count);
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched) {
    comm->touched = 0;
    if (count) comm->last = count;
  }
  touching = NULL;
}

/* a walk's failure: comm (the walk's own group's; NULL: none) and every communicator with a group in flight through
   `ticket` failed with it (comm_fail: its channels stand mid-group), their groups lost */
static void walk_failed(struct ncclComm *comm, uint64_t ticket, ncclResult_t result) {
  char why[sizeof last];
  snprintf(why, sizeof why, "%s", last);
  for (; comm || (inflight && inflight->ticket <= ticket); comm = NULL) comm_fail(comm ? comm : inflight->comm, result, "%s", why);
  fail(result, "%s", why);
}

/* the groups through `ticket` whose communicators failed lost; the first loss through it, now reported, and every loss
   through it forgotten */
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

/* one group on the stream's command buffer (or, the Metal 4 form, its Metal 4 encoder: nccl-mesh-metal.h metal_begin4),
   taking its calls: each collective's operand recvbuff itself where it holds the whole operand (an all-reduce's, a
   root's reduce, a broadcast's, an all-gather's), else a scratch one (a reduce-scatter's, a non-root's reduce); a
   direct exchange's sums apart; then the group scheduled and its lanes declared (schedule), in flight after the others,
   issued (with issue) or completed with every group before it; its results into recvbuff as it completes; the
   program's end signals `finished` where the library made its encoder.  The Metal 4 form makes its channels' lanes
   resident through the stream's residency set; without a workspace its scratch is the communicator's (workspace), so it
   is not issued. */
static ncclResult_t launch_metal(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclMeshIssue *const issue = list[0].issue;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  if (issue) *issue = (ncclMeshIssue){0, 0};
  for (size_t i = 0; i < n && !result; i++) {
    if (list[i].comm != comm) result = fail(ncclInvalidUsage, "a group on several communicators");
    else if (list[i].command != list[0].command || list[i].encoder != list[0].encoder)
      result = fail(ncclInvalidUsage, "a group on several command buffers or encoders");
    else if (list[i].polls != list[0].polls) result = fail(ncclInvalidUsage, "a group of several bounds (ncclMeshStream polls)");
    else if (!on_gpu(list[i].type)) result = fail(ncclInvalidArgument, "datatype %d on the Metal path (no float64 there)", (int)list[i].type);
    else if (reducing(list[i].what)) result = reduction(list + i);
    remote |= crosses(list + i);
  }
  const int metal4 = list[0].table != NULL;
  if (!result && metal4 && (!list[0].encoder || !list[0].residency))
    result = fail(ncclInvalidArgument, "a Metal 4 ncclMeshStream (argumentTable) without its encoder or residency set");
  else if (!result && metal4 && issue && !list[0].workspace)
    result = fail(ncclInvalidUsage, "a Metal 4 group issued without a workspace (its scratch is the communicator's, free at its end)");
  else if (!result && !metal4 && !list[0].command) result = fail(ncclInvalidArgument, "an ncclMeshStream without a command buffer");
  else if (!result && !metal4 && list[0].encoder && !list[0].workspace)
    result = fail(ncclInvalidArgument, "an ncclMeshStream with an encoder and no workspace");
  else if (!result && list[0].fused && (!metal4 || issue || n != 1 || !same(list[0].send_at, list[0].recv_at) ||
                                        (list[0].what != MESH_ALLREDUCE && list[0].what != MESH_BROADCAST && list[0].what != MESH_REDUCE)))
    result = fail(ncclInvalidUsage, "a fused crossing is one in-place ncclAllReduce, ncclBroadcast or ncclReduce on a Metal 4 stream, not issued "
                                    "(ncclMeshFusable, ncclMeshFusableRooted)");
  if (!result && revoked(comm)) result = revoked_fail(comm);
  pthread_mutex_lock(&group_lock);
  if (!result && remote) result = comm_ready(comm);
  if (!result) result = plan_calls(comm, list, n);
  if (!result && list[0].fused && (!remote || !list[0].bound ||
                                   list[0].bound->buffers != 1 + (list[0].what == MESH_ALLREDUCE && comm->nranks == 2)))
    result = fail(ncclInvalidUsage, "a fused crossing whose program copies its operand before publishing it (ncclMeshFusable)");
  if (result) { pthread_mutex_unlock(&group_lock); counted_failed(list, n); calls_free(list, n); return result; }
  if (finished) metal_collect(finished);
  void *const violation = remote ? comm->violation : NULL;
  struct metal_program *program = metal4 ? metal_begin4(list[0].encoder, list[0].table, list[0].residency, violation, list[0].polls)
                                         : metal_begin(list[0].command, list[0].encoder, violation, list[0].polls);
  if (!program) {
    pthread_mutex_unlock(&group_lock); counted_failed(list, n); calls_free(list, n);
    return fail(ncclSystemError, "%s", *metal_error() ? metal_error() : "out of memory");
  }
  comm->scratch_used = 0;
  if (remote) touch(comm);
  for (uint32_t h = 0; metal4 && remote && h < comm->count; h++) metal_resident(program, comm->channels[h].buffer);
  const size_t r = (size_t)comm->rank;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *st = c->bound;
    const size_t z = SIZE[c->type], bytes = elements(c) * z, all = elements(c);
    const struct where send = c->send_at;
    if (counted(c)) { result = counted_stage(program, c, violation); continue; }
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
    size_t length = c->count;
    const size_t before = c->what == MESH_ALLGATHER ? segment_at(c, (int)r, &length) : 0;
    const struct where segment = {own.buffer, own.offset + before * z};
    const int direct = plan_direct(c, own, issue != NULL);
    if (!direct && reducing(c->what) && c->premultiplied)
      result = gpu(metal_premultiply(program, c->type, own.buffer, own.offset, send.buffer, send.offset, all, scalar_bits(c)), "the operand");
    else if (!direct && (reducing(c->what) || (c->what == MESH_BROADCAST && c->root == comm->rank) || c->what == MESH_ALLGATHER) &&
             !same(segment, send))
      result = gpu(metal_copy(program, METAL_PLAIN, segment.buffer, segment.offset, send.buffer, send.offset,
                              reducing(c->what) ? bytes : length * z), "the operand");
    c->sum = own;
    if (st) st->gpu[0] = own;
    for (uint32_t b = 1; st && b < st->buffers && !result; b++) {
      /* a fused two-rank all-reduce's copy feeds only its SEND, which the caller's kernel publishes from its own store:
         no copy is made, and nothing reads the scratch */
      if (c->fused && c->what == MESH_ALLREDUCE && comm->nranks == 2) { st->gpu[b] = own; continue; }
      /* read in place: a scratch no step writes is never read (its SENDs read the send buffer), one a step writes is
         written before it is read (its first combine from the send buffer) */
      if (direct && !st->needed[b]) { st->gpu[b] = (struct where){NULL, 0}; continue; }
      void *scratch = workspace(program, c, bytes);
      if (!scratch) { result = fail(ncclSystemError, "out of GPU memory"); break; }
      st->gpu[b] = (struct where){scratch, 0};
      if (!direct) result = gpu(metal_copy(program, METAL_PLAIN, scratch, 0, own.buffer, own.offset, bytes), "a scratch copy");
    }
  }
  int scheduled = 0, published = 0, owned = 1;
  uint64_t ticket = 0;
  if (!result && remote) {
    struct inflight *g = calloc(1, sizeof *g);
    if (!g || !(g->start = calloc(2 * (size_t)comm->count + 1, sizeof *g->start))) { free(g); result = fail(ncclSystemError, "out of memory"); }
    else {
      g->comm = comm;
      g->end = g->start + comm->count;
      for (uint32_t h = 0; h < comm->count; h++) g->start[h] = comm->channels[h].end;
      result = schedule(comm, list, n, &g->plans);
      if (result) { free(g->start); free(g); }
      else {
        for (uint32_t h = 0; h < comm->count; h++) g->end[h] = comm->channels[h].end;
        g->list = list; g->n = n; g->ticket = ticket = ++tickets; owned = 0;
        if (inflight_last) inflight_last->next_group = g; else inflight = g;
        inflight_last = g;
        scheduled = 1;
        result = order_group(comm, g);
        if (list[0].fused) {
          const int what = list[0].what, rooted = comm->rank == list[0].root;
          fusing = (typeof(fusing)){list[0].fused, list, 0, 0, what == MESH_ALLREDUCE || (what == MESH_BROADCAST) == rooted,
                                    what == MESH_ALLREDUCE || (what == MESH_REDUCE) == rooted};
        }
        if (!result) result = walk(program, issue ? g : NULL, issue ? NULL : g, &published);
        if (!result && list[0].fused && (fusing.published != fusing.publishes || fusing.landed != fusing.lands))
          result = fail(ncclInvalidUsage, "a fused crossing whose program is not this rank's publication and landing of its whole operand "
                                          "(ncclMeshFusable, ncclMeshFusableRooted)");
        fusing.out = NULL; fusing.call = NULL;
        metal_gate(program, violation);
        for (size_t i = 0; i < n && !result; i++) result = complement_copy(program, list + i);
        if (!result) result = retire(program);
      }
    }
  } else if (!result)
    result = epilogue(program, list, n);
  program_end(program, list[0].encoder != NULL, result);
  if (result && scheduled) walk_failed(comm, issue ? 0 : ticket, result);
  if (issue && !result) { issue->ticket = ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  pthread_mutex_unlock(&group_lock);
  if (owned && result) counted_failed(list, n);
  if (owned) calls_free(list, n);
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
    if (finished) metal_collect(finished);
    struct metal_program *program = stream->argumentTable
      ? metal_begin4(stream->commandEncoder, stream->argumentTable, stream->residencySet, inflight->comm->violation, stream->polls)
      : metal_begin(stream->commandBuffer, stream->commandEncoder, inflight->comm->violation, stream->polls);
    if (!program) result = fail(ncclSystemError, "%s", *metal_error() ? metal_error() : "out of memory");
    else {
      /* the units still to publish declared anew, their deadline from now (declare): they are supplied as this
         completion's work runs */
      for (struct inflight *g = inflight; g; g = g->next_group)
        if (!g->comm->touched) { touch(g->comm); declare(g->comm, 1); }
      result = walk(program, NULL, through, &any);
      if (!result) result = retire(program);
      program_end(program, stream->commandEncoder != NULL, result);
      if (result) walk_failed(NULL, ticket, result);
    }
  }
  pthread_mutex_unlock(&group_lock);
  if (published) *published = any;
  return result ? result : loss;
}

uint64_t ncclMeshRetired(void) { return atomic_load(&retired); }

void ncclMeshMetal4Records(uint64_t *issued, uint64_t *capacity) { metal_records(issued, capacity); }

void ncclMeshInitWaits(uint64_t *ns) { *ns = atomic_load(&init_waits); }

/* whether a rank's moves of a program are at most one SEND and one receive, each of every part: the whole operand in one
   piece each way */
static int whole_moves(const struct mesh_program *program, uint32_t rank) {
  uint32_t sends = 0, receives = 0;
  for (uint32_t i = program->first[rank]; i < program->first[rank + 1]; i++) {
    const struct mesh_move *m = program->move + i;
    if (m->part || m->parts != program->parts || (m->op == MESH_STEP_SEND ? sends++ : receives++)) return 0;
  }
  return 1;
}

ncclResult_t ncclMeshFusableRooted(ncclComm_t comm, ncclMeshFuseKind kind, int root, size_t count, ncclDataType_t type, ncclRedOp_t op,
                                   int *publishes, int *lands) {
  if (!publishes || !lands) return fail(ncclInvalidArgument, "publishes or lands is NULL");
  *publishes = *lands = 0;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int what = kind == ncclMeshFuseBroadcast ? MESH_BROADCAST : kind == ncclMeshFuseReduce ? MESH_REDUCE : MESH_ALLREDUCE;
  if (comm->nranks != 2 || !count || root < 0 || root > 1 || (what == MESH_ALLREDUCE && root) || (int)type < 0 || (int)type >= ncclNumTypes ||
      (what != MESH_BROADCAST && (op != ncclSum || type != ncclFloat16)))
    return ncclSuccess;
  const uint64_t bytes = count * SIZE[type];
  pthread_mutex_lock(&group_lock);
  /* the answer is the communicator's program, its table's (the caller's, or the structural one compiled on its map, here
     where no group compiled it yet: a call records its crossings before its first group) */
  if (comm_compile(comm) != ncclSuccess) { pthread_mutex_unlock(&group_lock); return ncclSuccess; }
  const struct mesh_program *program = program_of(comm, what, root, bytes);
  /* a two-rank all-reduce's one scratch is its copy of the operand, which its program sends from while the sum is made
     in place; its fused steps send from the producer's own store and land into the consumer's, so neither reads it; a
     configured form that cuts the operand (two-phase, chunked) moves it in several pieces and is not fusable */
  const int fusable = program && program->scratch <= (what == MESH_ALLREDUCE ? 1u : 0u) && bytes <= comm->wire.small &&
                     !mesh_wire_large(&comm->wire, bytes, SIZE[type]) && whole_moves(program, (uint32_t)comm->rank);
  pthread_mutex_unlock(&group_lock);
  if (!fusable) return ncclSuccess;
  const int rooted = comm->rank == root;
  *publishes = what == MESH_ALLREDUCE || (what == MESH_BROADCAST) == rooted;
  *lands = what == MESH_ALLREDUCE || (what == MESH_REDUCE) == rooted;
  return ncclSuccess;
}

ncclResult_t ncclMeshFusable(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int *fusable) {
  if (!fusable) return fail(ncclInvalidArgument, "fusable is NULL");
  int publishes = 0, lands = 0;
  const ncclResult_t result = ncclMeshFusableRooted(comm, ncclMeshFuseAllReduce, 0, count, type, op, &publishes, &lands);
  *fusable = publishes && lands;
  return result;
}

const char *ncclMeshMetalSteps(void) { return metal_steps(); }

ncclResult_t ncclMeshEncodeMetal4(ncclComm_t comm, ncclRedOp_t op, const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type,
                                  void *encoder, void *table, void *residency, uint32_t polls) {
  ncclMeshStream stream = {.commandEncoder = encoder, .argumentTable = table, .residencySet = residency, .polls = polls};
  return ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, &stream);
}

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
  if (result) list[0].comm->async = result;
  calls_free(list, n);
  return result;
}

/* a call refused before it joins its group: its arguments, the stream's form taken into it */
static ncclResult_t admit(struct call *made) {
  struct call c = *made;
  if (!c.comm) return fail(ncclInvalidArgument, "comm is NULL");
  if (c.stream) {
    const ncclMeshStream *stream = c.stream;
    c.command = stream->commandBuffer; c.encoder = stream->commandEncoder; c.workspace = stream->workspace; c.context = stream->context;
    c.issue = stream->issue; c.table = stream->argumentTable; c.residency = stream->residencySet; c.polls = stream->polls;
    c.fused = stream->fused;
    c.send_at = where_of(c.send); c.recv_at = where_of(c.recv);
  }
  *made = c;
  if ((unsigned)c.type >= ncclNumTypes) return fail(ncclInvalidArgument, "datatype %d", (int)c.type);
  const ncclComm_t comm = c.comm;
  if (reducing(c.what)) {
    const int k = (int)c.op - ncclNumOps;
    if ((int)c.op < 0 || (k >= 0 && (k >= OPS || !comm->ops[k].used))) return fail(ncclInvalidArgument, "operator %d", (int)c.op);
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

/* a call on the Metal path keeps its command buffer and its buffers' MTLBuffers and offsets, so neither the
   stream nor the ncclMeshBuffers need outlive the call (a lowering's are its own, gone before an enclosing
   group ends); what the call owns (an uneven call's counts) is freed with it, or here where it is refused */
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

/* MPI_Allgatherv and MPI_Reduce_scatter (nccl.h ncclMeshAllGatherV, ncclMeshReduceScatterV): equal counts are NCCL's own
   call; others the call's own copy of them, its program's segments cut by them (mesh 3981ca80^ enqueue_counts) */
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
ncclResult_t ncclMeshAllGatherVConfig(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclComm_t comm,
                                      cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return enqueue_counts((struct call){.what = MESH_ALLGATHER, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm,
                                      .stream = stream}, counts);
}
ncclResult_t ncclMeshAllGatherV(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream) {
  return ncclMeshAllGatherVConfig(sendbuff, recvbuff, counts, datatype, comm, stream, NULL);
}
ncclResult_t ncclMeshReduceScatterVConfig(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclRedOp_t op,
                                          ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return enqueue_counts((struct call){.what = MESH_REDUCE_SCATTER, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = op, .comm = comm,
                                      .stream = stream}, counts);
}
ncclResult_t ncclMeshReduceScatterV(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm, cudaStream_t stream) {
  return ncclMeshReduceScatterVConfig(sendbuff, recvbuff, counts, datatype, op, comm, stream, NULL);
}

/* nccl.h ncclMeshAlltoAllCounted (mesh 3981ca80^'s, its signature): the call's own copy of the segments, its word 0 now
   (the GPU, or the host path's end, sets it once the counts landed; a failure 2), and on the Metal path the Metal
   buffers of the leases holding `received` and `arrived` (lease_buffer), which the GPU writes */
ncclResult_t ncclMeshAlltoAllCounted(const void *sendbuff, size_t sendrows, const int64_t *counts, const size_t *sendsegs, void *recvbuff,
                                     size_t capacity, int64_t *received, int64_t *landed, const size_t *recvsegs, uint64_t *arrived, size_t row,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  if (!counts || !sendsegs || !recvsegs || !received || !arrived || !row || !sendbuff || !recvbuff)
    return fail(ncclInvalidArgument, "ncclMeshAlltoAllCounted: a NULL argument or a row of no elements");
  if (sendsegs[comm->rank] != recvsegs[comm->rank])
    return fail(ncclInvalidArgument, "ncclMeshAlltoAllCounted: this rank's segment for itself sent (%zu) and received (%zu) differ",
                sendsegs[comm->rank], recvsegs[comm->rank]);
  struct call c = {.what = WHAT_COUNTED, .send = sendbuff, .recv = recvbuff, .count = 1, .type = datatype, .op = ncclSum, .comm = comm,
                   .stream = stream, .counts = counts, .received = received, .landed = landed, .arrived = arrived, .rows = sendrows,
                   .capacity = capacity, .row = row};
  if (stream) {
    c.counts_at = where_of(counts); c.landed_at = where_of(landed);
    const int at_received = lease_buffer(received, &c.received_at.buffer, &c.received_at.offset);
    const int at_arrived = at_received ? at_received : lease_buffer(arrived, &c.arrived_at.buffer, &c.arrived_at.offset);
    if (at_received || at_arrived)
      return at_arrived == ESTALE ? attach_fail(ESTALE, "ncclMeshAlltoAllCounted's received or arrived")
                                  : fail(ncclInvalidArgument, "ncclMeshAlltoAllCounted on the Metal path: received and arrived are host memory in "
                                         "this process's leases (ncclMemAlloc), which the GPU writes as the counts land");
  }
  const int n = comm->nranks;
  if (!(c.segments = malloc(2 * (size_t)n * sizeof *c.segments))) return fail(ncclSystemError, "out of memory");
  for (int r = 0; r < n; r++) { c.segments[r] = sendsegs[r]; c.segments[n + r] = recvsegs[r]; }
  __atomic_store_n(arrived, 0, __ATOMIC_RELEASE);
  return enqueue(c);
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_SEND, .peer = peer, .send = sendbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = WHAT_RECV, .peer = peer, .recv = recvbuff, .count = count, .type = datatype, .comm = comm, .stream = stream});
}

/* nccl.h ncclMeshAlltoAllV, and NCCL's ncclAlltoAll, ncclGather and ncclScatter as its forms: a counted all-to-all whose
   counts are the host's (hosted), the call's own copy of them (the elements to each rank, then from each: sendcounts
   and recvcounts where given, else `count` from rank `from` to rank `to`, -1 every rank), its `count` those crossing */
static ncclResult_t routed(const void *sendbuff, void *recvbuff, const size_t *sendcounts, const size_t *recvcounts, size_t count, int from, int to,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int n = comm->nranks, me = comm->rank, root = from < 0 ? to : from;
  if (root < -1 || root >= n) return fail(ncclInvalidArgument, "root %d of %d ranks", root, n);
  struct call c = {.what = WHAT_COUNTED, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm, .stream = stream};
  if (!(c.segments = malloc(2 * (size_t)n * sizeof *c.segments))) return fail(ncclSystemError, "out of memory");
  for (int q = 0; q < n; q++) {
    c.segments[q] = sendcounts ? sendcounts[q] : (from < 0 || from == me) && (to < 0 || to == q) ? count : 0;
    c.segments[n + q] = recvcounts ? recvcounts[q] : (from < 0 || from == q) && (to < 0 || to == me) ? count : 0;
    if (q != me) c.count += c.segments[q] + c.segments[n + q];
  }
  const unsigned long long sent = c.segments[me], received = c.segments[n + me];
  if (sent == received) return enqueue(c);
  free(c.segments);
  return fail(ncclInvalidArgument, "an all-to-all's own segment, sent (%llu) and received (%llu), differ", sent, received);
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

/* a communicator over comm's ranks `ranks` (count of them, in that order), named after comm and its `split`-th split or
   shrink and `color`, the same on every member, attached (comm_attach), its forms comm's; its lanes bound and its
   channels made by its caller (ncclCommSplit) */
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

/* ncclCommSplit: every rank's (color, key) and its report (lanes_read; none for NCCL_SPLIT_NOCOLOR) all-gathered on comm,
   the ranks of this color ordered by key, then rank; the one rule applied to their reports (comm_choose) and each rank's
   row bound at its bridge (comm_bind), and a second all-gather on comm carrying each rank's binding's status and its
   lanes' first units, from which the child's channels are made (comm_join): NCCL's bootstrapSplit (the child's exchange
   over its parent's), with the init's two gathers.  A failure at any rank of a color fails that color's split on each of
   its ranks as a value, its pairs unbound (comm_free) */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  if (depth) return fail(ncclInvalidUsage, "ncclCommSplit inside a group");
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; struct report report; } mine = {color, key, {.rank = comm->rank}}, *every = calloc((size_t)n, sizeof *every);
  struct binding { int32_t status, padding; uint64_t start[MESH_PORTS][2]; } bound = {0, 0, {{0}}}, *binds = calloc((size_t)n, sizeof *binds);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0, chose = 0;
  struct report *reports = calloc((size_t)n, sizeof *reports);
  struct mesh_lane_report *lanes = calloc((size_t)n, sizeof *lanes);
  uint32_t *chosen = calloc((size_t)n * MESH_PORTS, sizeof *chosen);
  *newcomm = NULL;
  if (color != NCCL_SPLIT_NOCOLOR) {
    pthread_mutex_lock(&attach_lock);
    mine.report.status = lanes_read(comm->attach, &mine.report);
    pthread_mutex_unlock(&attach_lock);
  }
  ncclResult_t result = every && binds && ranks && reports && lanes && chosen ? ncclAllGather(&mine, every, sizeof mine, ncclUint8, comm, NULL)
                                                                             : fail(ncclSystemError, "out of memory");
  const int gathered = !result;
  for (int r = 0; !result && color != NCCL_SPLIT_NOCOLOR && r < n; r++)
    if (every[r].color == color) ranks[count++] = r;
  for (int a = 1; a < count; a++)
    for (int b = a; b > 0 && every[ranks[b]].key < every[ranks[b - 1]].key; b--) { const int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
  for (int i = 0; !result && i < count; i++)
    if (every[ranks[i]].report.status)
      result = fail(ncclSystemError, "a split of %s: the lane pairs free at rank %d's bridge (mesh_lanes_free): %s", comm->name, ranks[i],
                    strerror(every[ranks[i]].report.status));
  if (!result && count) result = comm_sub(comm, ranks, count, split, color, newcomm);
  for (int i = 0; !result && i < count; i++) reports[i] = every[ranks[i]].report;
  if (!result && count > 1 && (chose = comm_choose(reports, (uint32_t)count, lanes, chosen))) result = choose_fail((*newcomm)->name, chose);
  if (!result && count > 1)
    bound.status = comm_bind(*newcomm, &reports[(*newcomm)->rank].lanes, chosen + (size_t)(*newcomm)->rank * MESH_PORTS, bound.start);
  else if (result && count) bound.status = ECANCELED;
  if (gathered) {
    const ncclResult_t confirmed = ncclAllGather(&bound, binds, sizeof bound, ncclUint8, comm, NULL);
    if (!result) result = confirmed;
  }
  for (int i = 0; !result && i < count; i++)
    if (binds[ranks[i]].status) result = bind_fail((*newcomm)->name, i, binds[ranks[i]].status);
  for (int i = 0; !result && i < count; i++) memcpy(reports[i].start, binds[ranks[i]].start, sizeof reports[i].start);
  int status = 0;
  if (!result && count > 1 && (status = comm_join(*newcomm, reports, chosen)))
    result = fail(ncclSystemError, "communicator %s: its channels on the lanes it bound: %s", (*newcomm)->name, strerror(status));
  if (result && *newcomm) { comm_free(*newcomm); *newcomm = NULL; }
  free(every); free(binds); free(ranks); free(reports); free(lanes); free(chosen);
  return result;
}

/* ncclCommShrink: comm's ranks but the excluded ones, in rank order, with no exchange among them (a departed member
   answers none), so binding no lanes: a shrink to two ranks or more, whose channels would be none, is refused (ENOLCK);
   with NCCL_SHRINK_ABORT, comm revoked first (ncclCommRevoke) */
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
  if (shrinkFlags & NCCL_SHRINK_ABORT) ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
  const ncclResult_t result =
    !kept ? fail(ncclInvalidArgument, "rank %d is among the ranks it excludes", comm->rank)
    : count > 1 ? fail(ncclInvalidUsage, "a shrink of %s to %d ranks would hold no lane pair (ENOLCK): it makes no exchange among them, and only "
                       "an exchange binds a communicator's lanes (an init's bootstrap, ncclCommSplit over its parent)", comm->name, count)
    : comm_sub(comm, ranks, count, comm->splits++, -1, newcomm);
  free(ranks);
  return result;
}

/* ncclCommGetUniqueId: the id a grow of comm meets at, a listener of comm's root (ncclGetUniqueId) */
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
  return comm ? ncclGetUniqueId(uniqueId) : fail(ncclInvalidArgument, "comm is NULL");
}

/* ncclCommGrow: NCCL's bootstrap at the id of comm's root (ncclCommGetUniqueId), which the root broadcasts on comm to
   comm's other ranks; comm's ranks (rank -1) keep their ranks and a new one (comm NULL) takes `rank`, every rank
   reporting its node and lanes; the communicator, named by the id, binds its lanes as every init does */
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!newcomm || (comm ? rank != -1 || nRanks < comm->nranks || (!comm->rank && !uniqueId) : !uniqueId))
    return fail(ncclInvalidArgument, "ncclCommGrow's arguments");
  if (depth) return fail(ncclInvalidUsage, "ncclCommGrow inside a group");
  if (!comm) return ncclCommInitRank(newcomm, nRanks, *uniqueId, rank);
  ncclUniqueId id = comm->rank ? (ncclUniqueId){{0}} : *uniqueId;
  const ncclResult_t result = ncclBroadcast(&id, &id, sizeof id, ncclUint8, 0, comm, NULL);
  return result ? result : ncclCommInitRank(newcomm, nRanks, id, comm->rank);
}

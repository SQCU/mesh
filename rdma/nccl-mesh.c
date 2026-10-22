/* libnccl-mesh (nccl.h): NCCL's API on the streams of its node's bridge (design/transport.md section 14).  The process
   is one client of its node's bridge (an attach, whose slot owns the process's leases of the region, ncclMemAlloc, and
   which holds the bridge's stream toward each node its communicators reach, as it holds rows: mesh-call.h
   mesh_stream_take).  A communicator's init holds the stream toward each other member's node and exchanges each
   stream's first units through NCCL's bootstrap; its channels are those streams, one communicator a stream (comm_hold),
   and its map the nodes its members' bridges reach.  It names nodes,
   never a link, port, lane or queue pair.  A group's calls are planned by the one collective planner, laid on the
   channels as runs of the streams' units, declared to the bridge, supplied and awaited, the receives combined on the
   host or the GPU. */
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
#include <mach/mach_time.h>


enum { WHAT_SEND = 16, WHAT_RECV, WHAT_COUNTED };
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN, COMBINE_MERGE };
#define OPS 64

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
/* a failure's message: its text, and `id`, where the message is kept with its failure (a communicator's, comm_fail; a
   non-blocking init's, init_run), the count of messages the process had kept when it was (`messages`), else 0 */
struct said { uint64_t id; char text[512]; };
static _Atomic uint64_t messages;
/* this thread's last failure's message (ncclGetLastError): formatted by fail (id 0), or a copy of a kept one (told) */
static _Thread_local struct said last;

static ncclResult_t fail(ncclResult_t result, const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(last.text, sizeof last.text, format, arguments);
  va_end(arguments);
  last.id = 0;
  return result;
}

/* `result` beside the kept message `said` (design/rubrics.md, Strings are messages): `last` its copy, copied only where
   `last` holds another (its id the match), so a failure polled again formats and copies nothing */
static ncclResult_t told(ncclResult_t result, const struct said *said) {
  if (last.id != said->id) last = *said;
  return result;
}

/* a kept message's identity, a new one */
static uint64_t keep(void) { return atomic_fetch_add_explicit(&messages, 1, memory_order_relaxed) + 1; }

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
   0] (nccl-mesh-kernels.h merge_row) */
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
   mesh_wire_large), on that class of the stream toward that node its process holds (mesh-call.h mesh_stream_view: its
   rings and words), and the Metal buffer over them (`buffer`: its in ring, out ring, ready words and landed words at
   `in`, `out`, `ready` and `landed` in it).  Its positions: `sent` published, `consumed` consumed, `end` laid by the
   groups scheduled, `waited` awaited by the GPU work encoded; its units, the stream's (struct hold): `laid` the first of
   the next out run, `taken` the first of the next in run (the
   bridge sends a stream's units in order), `declared` those declared to the bridge (mesh_stream_declare) */
struct channel {
  int peer, large;
  uint32_t node, stream;
  const uint64_t *wire;
  struct mesh_stream_view view;
  void *buffer;
  size_t in, out, ready, landed;
  uint64_t sent, consumed, end, waited, *laid, *taken, *declared;
};

/* a communicator: rank r on the bridge of node members[r]; its map (map_of: its members joined where their bridges
   reach each other's nodes, the routing table mesh_routes'), made at its init alike by every member and kept; the wire
   its streams make (mesh-plan.h mesh_wire_of at its bridge's block); its message paths, the caller's (count 0: none), else the map's
   shortest-path trees (`trees`); its table of programs (`entries`), the caller's (`listed`, kept for its life), else the
   structural one compiled along its paths or trees where a call or a query first needs it (`compiled`: comm_compile),
   its configured forms first (`forms`, ncclMeshCommForms, which its splits and shrinks keep), fixed once a group of it
   crossed a link (`crossed`); its name, and the count of splits and shrinks made of it, which name theirs; the
   process's attach it holds (`attach`, until ncclCommDestroy); the attach's holds of streams it uses (`held`, a bit a
   hold; `binding` while it counts among their users); its channels (`count`), two toward each member whose node its
   bridge reaches, toward[r] the first of rank r's (-1: none); its failure words (`words`, nccl-mesh-metal.h: its
   GPU work's failure records, which the host reads and no kernel halts on, and the stop a revoke sets; `violation` the Metal
   buffer over them) and its failure once kept (`failed`, its message `failure`: comm_fail); `last`, the count of the last
   GPU program signalling `finished` that touched it (`touched`, `next_touched` while one is encoded: program_end), and
   `refreshed` its declarations' last refresh (ncclMeshComplete); the number its next group issued takes (`groups`:
   group_numbered) and, of its last GROUP_TIMES groups, group g's at g mod GROUP_TIMES, each channel's first unit of the
   group's first out run and last unit of its last in run, each plus one (`timed`, a channel's 0 where the group has no
   position on it: schedule_group, ncclMeshTimes); `runs`, its host-path groups running; `revoked`, set by
   ncclCommRevoke and ncclCommDestroy, failing every group after it; `peers`, its bootstrap's sockets, kept for its life
   once its init succeeds (NCCL's bootstrapState), which a revoke shuts down and comm_free closes, under `guard`, and the
   end of its bootstrap's present phase (`until`, bootstrap_open, gather).  A
   non-blocking init (nccl.h ncclMeshCommInitRank) runs on `init` (`initing`: ncclCommDestroy joins it), `async`
   ncclInProgress until it ends and `why` its failure's message (its id 0 while none is kept) */
struct attach;
/* the groups a communicator's record of units holds (`timed`): as many as the Metal 4 argument ring's records
   (nccl-mesh-metal.m RECORDS, a group's dispatches one at least), within which a caller bounds the groups it encodes ahead
   (ncclMeshMetal4Records) */
#define GROUP_TIMES 65536u
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
  double until;
  pthread_t init;
  struct said why, failure;
  void **scratch, **spent;
  size_t *scratch_bytes;
  uint32_t scratches, scratch_used, spents;
  struct attach *attach;
  uint32_t held;
  struct channel *channels;
  _Atomic uint64_t *words;
  void *violation;
  uint64_t last, refreshed, groups, (*timed)[2];
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
  routes_free(comm, 1);
  for (uint32_t k = 0; k < comm->scratches; k++) metal_release(comm->scratch[k]);
  for (uint32_t k = 0; k < comm->spents; k++) metal_release(comm->spent[k]);
  for (uint32_t h = 0; h < comm->count; h++) metal_release(comm->channels[h].buffer);
  if (comm->violation) metal_release(comm->violation);
  comm_leave(comm);
  free(comm->scratch); free(comm->scratch_bytes); free(comm->spent); free(comm->channels); free(comm->toward); free(comm->timed);
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
   trees (mesh_trees_shortest: each member's in-tree of the routes over the links between members), the single-phase
   all-reduce (each member's sum up its own tree; the direct exchange where every two members are linked), the
   reduce-scatter and the all-gather (segment s up and down s's trees), and each root's reduce and broadcast, uncut and
   of any size (mesh_compile; mesh 2bf5a50d^ comm_routes's table, its entries along the routes): a call takes the first
   entry that fits it (program_of), so a collective no form names, or a size past every form's bound, runs the default.
   A map whose links do not join its members fails it, naming two: the bridges forward nothing, so no route crosses a
   node outside the communicator */
static ncclResult_t comm_routes(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  if (comm->listed || n < 2) return ncclSuccess;
  uint32_t apart[2] = {0, 0};
  const int joined = mesh_link_map_joined(&comm->map, apart);
  if (joined == ENOTCONN)
    return fail(ncclInvalidUsage, "rank %u (node %u) and rank %u (node %u) of %s are joined by no path of links between its members: the "
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

/* the communicator's table made where it is not (comm_routes), each move of each program between two members a link
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
          result = fail(ncclInvalidUsage, "program %u of %s moves between ranks %u and %u, which no link joins: the bridges forward nothing, "
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
   comes first.  Its slot owns every lease the process takes (ncclMemAlloc: a section of the region, mesh-call.h
   mesh_section_create, whose rows' owner is the slot), and it holds the stream toward each node its communicators reach
   (`holds`, mesh-call.h mesh_stream_take), each for one communicator at a time.  The leases'
   bookkeeping is the region's, where the bridge sees it: the bridge retires the slot of a process gone and returns its
   leases' pages as it returns every retired client's rows; `leases` is this process's index of them by address.  A
   lease is an operand of the node's region, which outlives its bridge (design/transport.md section 2: a bridge that
   restarts adopts it), so a lease, and every buffer over it, stands across a bridge's restart.  A region a stop released
   (or a bridge of another layout replaced) is expired (its layout version 0): each new lease or communicator then
   returns the expiry (ncclRemoteError) while the process holds a lease or communicator of it, and the attach is let go
   (unmapped, its descriptor closed: attach_settle) once it holds none, so the next lease or communicator attaches the
   bridge's next region; what the process holds of the expired one stays mapped and valid until it frees it.  No
   process ever holds two regions (design/transport.md section 2, the heartbeat). --- */

/* a lease: its address, its first row, its bytes, the Metal buffer over it (lease_buffer, made at its first use; NULL
   until then), and the points its caller's GPU work on it declared (ncclMeshMemUse: an event, the value it reaches once
   that work is done, whether the work writes the lease), which a host-path group waits for (lease_wait) */
struct point { void *event; uint64_t value; int write; };
struct lease { void *address; uint32_t row; size_t bytes; void *buffer; struct point *points; size_t npoints; };
/* design/transport.md section 14: the stream toward node `node` the process holds (`users` the communicator on it, 0
   none: the hold vacant), each class's view (0 large, 1 small), and each class's units: `laid` the first of the next out
   run, `taken` the first of the next in run, `declared` those declared to the bridge, and the identity it was taken by
   (`wire`: mesh-call.h mesh_stream_held, comm_failure) */
struct hold { uint32_t node, stream, users, padding; struct mesh_stream_view view[2]; uint64_t laid[2], taken[2], declared[2], wire; };
struct attach {
  struct mesh_ctx context;
  struct lease *leases;
  size_t count, room;
  uint64_t comms;
  /* whether the library made a Metal buffer over the region (a lease's, lease_buffer; a channel's, channel_buffer):
     the caller's GPU work, and its residency set, may hold those pages after the buffer is released, so such an attach
     is never let go and its process ends to leave the region (attach_settle) */
  _Atomic int wrapped;
  struct hold holds[MESH_STREAMS];
  struct attach *next;
};
/* the process's one attach of the node's region (`attaches`, made at its first lease or communicator), under attach_lock */
static struct attach *attaches;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;

/* whether attach `a`'s region is expired (released or replaced: its layout version 0) */
static int attach_expired(const struct attach *a) {
  return __atomic_load_n(&a->context.M->version, __ATOMIC_ACQUIRE) != MESH_LAYOUT;
}
/* the process's attach let go where its region is expired, it holds no lease and no communicator of it, and the library
   made no Metal buffer over it (`wrapped`: a process that did ends to leave it): its slot retired, its mapping and
   descriptor released (mesh_detach), so the heartbeat unlinks that region once no holder lives; under attach_lock */
static void attach_settle(void) {
  struct attach *a = attaches;
  if (!a || a->count || a->comms || atomic_load(&a->wrapped) || !attach_expired(a)) return;
  mesh_detach(&a->context);
  free(a->leases);
  free(a);
  attaches = NULL;
}
/* the process's attach, made at its first use and again after an expired one is let go: 0, ESTALE where the process
   holds leases or communicators of an expired region, or mesh_attach's error (no region: none is named, or the named
   one is expired) */
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

/* the process's attach not made (attach_current's error) for what `format` names: every client slot of the region held
   (EBUSY), or no region (no bridge has made it) */
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
/* a communicator's init without its attach: ENOMEM, else attach_fail's */
static ncclResult_t comm_refused(int status, const char *name) {
  return status == ENOMEM ? fail(ncclSystemError, "out of memory") : attach_fail(status, "communicator %s", name);
}

/* a communicator no longer a user of its attach's holds (`held`); a hold none uses let go (mesh-call.h mesh_stream_give:
   the bridge sends every unit declared on it, those no group supplied as duds, and frees the stream); under
   attach_lock */
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
/* a communicator's holds let go (comm_let_go) and its hold of its attach let go */
static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) {
    comm_let_go(comm);
    comm->attach->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
}

/* a rank's report at a communicator's init (NCCL's bootstrap, below): its status (0, else why it cannot go on, so every
   rank fails alike), its bridge's node and the nodes that bridge reaches with each one's bandwidth, and the streams free
   to take, each one's node and identity (`frees` of them: reach_read, mesh-call.h mesh_streams_free), and, at the
   second gather, the first unit of each class of its stream toward each of them (start[k][0] the large class's,
   start[k][1] the small's: its hold's `laid`, comm_hold) and whether it took that stream at this init (`fresh`, a bit
   for each) */
struct report {
  int32_t rank, status;
  uint32_t node, count, fresh, frees;
  struct mesh_reach reach[MESH_STREAMS];
  uint64_t start[MESH_STREAMS][2], free_wire[MESH_STREAMS];
  uint32_t free_node[MESH_STREAMS];
};

/* the index in report x of node `node` among the nodes its bridge reaches, x->count where it reaches none such */
static uint32_t reach_of(const struct report *x, uint32_t node) {
  uint32_t k = 0;
  while (k < x->count && k < MESH_STREAMS && x->reach[k].node != node) k++;
  return k;
}
/* whether ranks a and b are joined (each one's bridge reaches the other's node), and *bandwidth the less of the two
   ends' (0 where either reports none: mesh-plan.h mesh_map's rule) */
static uint32_t joined(const struct report *reports, uint32_t a, uint32_t b, uint64_t *bandwidth) {
  const uint32_t j = reach_of(reports + a, reports[b].node), k = reach_of(reports + b, reports[a].node);
  *bandwidth = 0;
  if (j >= reports[a].count || k >= reports[b].count) return 0;
  const uint64_t one = reports[a].reach[j].bandwidth, other = reports[b].reach[k].bandwidth;
  *bandwidth = one && other ? (one < other ? one : other) : 0;
  return 1;
}

/* design/transport.md section 14: the communicator's map, made alike by every member from the same gathered reports:
   its ranks its nodes, each marked by its bridge's node number, two linked where they are joined (joined: each one's
   bridge reaches the other's node), its routing table mesh_routes'.  No node outside the communicator is in it: the
   bridges forward nothing.  0, or ENOMEM, EINVAL */
static int map_of(struct ncclComm *comm, const struct report *reports) {
  const uint32_t n = (uint32_t)comm->nranks;
  uint32_t links = 0;
  uint64_t bandwidth;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++) links += joined(reports, a, b, &bandwidth);
  struct mesh_link_map *map = &comm->map;
  *map = (struct mesh_link_map){MESH_LINKS_GRAPH, n, links, calloc(links + 1, sizeof *map->link), calloc(links + 1, sizeof *map->bandwidth), 0,
                                calloc((size_t)n * n + 1, sizeof *map->next), calloc(n + 1, sizeof *map->mark), NULL};
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
   and it holds no stream of the bridge's.  The second follows this rank's holds of its streams, on which its peers'
   wait, so each of its sends and receives is bounded by mesh.h MESH_PLATFORM_NS (ETIMEDOUT:
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

/* a bootstrap root's address: a peer is reached on the fabric, never by a name that resolves to Wi-Fi or a LAN:
   the first address under mesh.h MESH_ULA_PREFIX of an interface other than loopback (bin/mesh-router-init.sh puts one on
   each fabric port) */
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

/* the id: NCCL_COMM_ID where set, else a listener on the node's fabric address (fabric_address) where it holds one, else
   on its host name */
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

/* whether comm was revoked; a group on it failed naming it */
static int revoked(ncclComm_t comm) { return __atomic_load_n(&comm->revoked, __ATOMIC_SEQ_CST); }
static double now_s(void);
/* design/rubrics.md M3, C5: each wait of a bootstrap a slice of BOOTSTRAP_SLICE_MS (a poll's timeout, a placed socket's
   SO_RCVTIMEO and SO_SNDTIMEO), after which the revoke and the phase's end (`until`) are read again: no wait of an init
   lasts past MESH_PLATFORM_NS of its phase, blocking or not, and a revoke ends a non-blocking one within a slice */
#define BOOTSTRAP_SLICE_MS 100
/* `size` bytes sent or received whole on comm's bootstrap socket `fd`: 0, ECANCELED (revoked), ETIMEDOUT (the phase's
   end passed), ECONNRESET (the peer closed), or the errno */
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

/* socket `fd` of comm's init placed at peers[r] (-1: the one there closed), blocking in slices (an accepted one takes
   its listener's O_NONBLOCK; each send and receive returns within BOOTSTRAP_SLICE_MS: whole), under comm's guard, where
   a revoke shuts it down (ncclCommRevoke), a send on it after that or a peer's reset an error, never SIGPIPE */
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

/* the bootstrap's sockets of comm's init at `id`, the root's to each rank r (peers[r], placed by the rank its first
   report names, that report kept in `reports`; a socket not yet placed waits at peers[0]) or a non-root's to the root
   (peers[0], placed once connected), each placed before the revoke is read, so a revoke either finds it or is read.  Its
   phase ends MESH_PLATFORM_NS after the init began where no rank was met, and MESH_PLATFORM_NS after the first rank met
   (the root's first accept, a non-root's connect) for the rest of the phase, the first gather's (`until`): 0, ECANCELED
   (comm revoked), ETIMEDOUT, or the errno */
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
  for (int r = 1; r < comm->nranks && !status; r++) {
    struct report heard;
    struct pollfd look = {listener, POLLIN, 0};
    int fd = -1;
    while (fd < 0 && !status)
      if (revoked(comm)) status = ECANCELED;
      else if (now_s() >= comm->until) status = ETIMEDOUT;
      else if (poll(&look, 1, BOOTSTRAP_SLICE_MS) < 0 && errno != EINTR) status = errno;
      else if ((fd = accept(listener, NULL, NULL)) < 0 && errno != EWOULDBLOCK && errno != ECONNABORTED && errno != EINTR) status = errno;
    if (!status && r == 1) comm->until = now_s() + bound;
    if (!status) bootstrap_put(comm, 0, fd);
    if (!status && revoked(comm)) status = ECANCELED;
    if (!status && !(status = whole(comm, fd, &heard, sizeof heard, 0)) && (heard.rank <= 0 || heard.rank >= comm->nranks || peers[heard.rank] >= 0))
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

/* a gather on comm's bootstrap sockets, within its phase (`until`): each other rank's report to the root (`sent`: there
   already, as at the first gather's root, whose bootstrap_open read each), then the whole table to each; 0, or the
   errno (whole) */
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

/* this rank's report of its bridge's node, the nodes it reaches (mesh-call.h mesh_nodes_reached) and the streams free to
   take (mesh_streams_free) */
static void reach_read(struct attach *a, struct report *report) {
  report->node = a->context.M->node;
  report->count = mesh_nodes_reached(a->context.M, report->reach, MESH_STREAMS);
  report->frees = mesh_streams_free(a->context.M, report->free_node, report->free_wire, MESH_STREAMS);
}

/* the stream two ranks' ends of a communicator take between their nodes: the least identity free toward the other's node
   in both reports (each end computes the same one from the same two reports), 0 where none is */
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

/* 0, or EINVAL where two of the `n` gathered reports are of one node (a bridge's streams join nodes, so a rank a node) */
static int comm_apart(const struct report *reports, uint32_t n) {
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++)
      if (reports[a].node == reports[b].node) return EINVAL;
  return 0;
}
static ncclResult_t apart_fail(const char *name) {
  return fail(ncclInvalidUsage, "communicator %s: two of its ranks are on one node (a rank a node: a bridge's streams join nodes)", name);
}
static ncclResult_t hold_fail(const char *name, int rank, int status) {
  return fail(ncclSystemError, "communicator %s: rank %d could not hold its streams at its bridge (%s: EBUSY, a stream another communicator holds, "
              "or whose last holder's units are still going; ENOLINK, no stream toward a member's node); a communicator made next "
              "reads the streams as they then stand", name, rank, strerror(status));
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


/* the attach's hold of the stream toward `node`, NULL where it holds none; under attach_lock */
/* comm's hold of its stream toward node `node`, NULL where it holds none */
static struct hold *comm_hold_toward(struct ncclComm *comm, uint32_t node) {
  for (uint32_t k = 0; k < MESH_STREAMS; k++)
    if ((comm->held >> k & 1) && comm->attach->holds[k].node == node) return comm->attach->holds + k;
  return NULL;
}
/* design/transport.md section 14: a stream toward each member's node this rank is joined to (joined) held for comm, the
   one of the identity both ranks' reports give free (wire_between: the two ends of a communicator hold the two ends of
   one stream), taken from the bridge as rows are (mesh-call.h mesh_stream_take: its classes' first units the stream's
   `next`, and `fresh` its bit in this rank's report), comm its one user (`held`, comm_let_go lets it go); each class's
   first unit of the next out run into this rank's report (`start`, by the index of the node in its reach).  One
   communicator a stream: the head credit that keeps a ring's units from being written again before they are read is a
   channel's own (a position published once the peer's LAG back is consumed, its ring sized for 2 LAG positions), so
   two communicators' channels on one ring could overrun it (metal-microbench output_data/bridge-20261019/FLOWS.md
   7.4e); a link has several streams, so communicators that share a node pair each hold their own.  0, or EBUSY (no
   stream toward a member's node free at both ends), ENOSPC (no hold vacant), or mesh_stream_take's refusal; what it
   took is let go with comm (comm_leave) */
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
    uint64_t next[2];
    for (uint32_t s = 0; !h && s < MESH_STREAMS; s++)
      if (!a->holds[s].users) h = a->holds + s;
    if (!wire) status = EBUSY;
    else if (!h) status = ENOSPC;
    else if (!(status = mesh_stream_take(&a->context, wire, &stream, next))) {
      for (uint32_t j = 0; j < MESH_STREAMS; j++)
        if (a->holds + j != h && a->holds[j].users && a->holds[j].stream == stream) a->holds[j].wire = 0;
      *h = (struct hold){.node = node, .stream = stream, .wire = wire};
      for (uint32_t c = 0; c < 2; c++) h->laid[c] = h->taken[c] = h->declared[c] = next[c];
    }
    if (status) break;
    const uint32_t index = (uint32_t)(h - a->holds);
    h->users++; comm->held |= UINT32_C(1) << index; comm->binding = 1;
    for (uint32_t c = 0; !status && c < 2; c++) status = mesh_stream_view(&a->context, stream, !c, &h->view[c]);
    mine->start[k][0] = h->laid[0]; mine->start[k][1] = h->laid[1];
    mine->fresh |= UINT32_C(1) << k;
  }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* the Metal buffer over a channel's stream class, its rings and words (the region's pages the bridge took for them,
   whole pages), and each one's offset in it */
static int channel_buffer(struct channel *ch) {
  const struct mesh_stream_view *v = &ch->view;
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

/* design/transport.md section 14: the communicator's channels on the streams its init held (comm_hold), from every
   rank's report and first units as the second gather gave them: toward each rank it is joined to, a channel of each
   class on that class of the stream toward the rank's node, its out units from this end's hold and its in units from
   the peer's first out (the bridge sends a stream's units in order): taken from the peer's report where either end took
   the stream at this init (comm_hold takes every stream it holds); its map
   (map_of), its wire (its bridge's block: every paired bridge's is one), its failure words and its record of its groups'
   units (`timed`).  0, or ENOMEM; EPROTO where a stream is not the wire's or no hold is of a joined node */
static int comm_join(struct ncclComm *comm, const struct report *reports) {
  const struct hdr *m = comm->attach->context.M;
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank, node = reports[me].node;
  uint64_t bandwidth;
  mesh_wire_of((uint64_t)m->pgsz * m->block, &comm->wire);
  int status = map_of(comm, reports);
  if (!status && !(comm->violation = metal_shared((size_t)getpagesize()))) status = ENOMEM;
  if (!status) comm->words = metal_contents(comm->violation);
  if (!status && !(comm->channels = calloc(2 * (size_t)n, sizeof *comm->channels))) status = ENOMEM;
  for (uint32_t q = 0; !status && q < n; q++) {
    if (q == me || !joined(reports, me, q, &bandwidth)) continue;
    pthread_mutex_lock(&attach_lock);
    struct hold *h = comm_hold_toward(comm, reports[q].node);
    pthread_mutex_unlock(&attach_lock);
    if (!h) { status = EPROTO; break; }
    const uint32_t j = reach_of(reports + q, node), k = reach_of(reports + me, reports[q].node);
    if ((reports[me].fresh >> k & 1) || (reports[q].fresh >> j & 1))
      for (uint32_t c = 0; c < 2; c++) h->taken[c] = reports[q].start[j][c];
    comm->toward[q] = (int)comm->count;
    for (int large = 1; !status && large >= 0; large--) {
      const uint32_t c = !large;
      struct channel *ch = comm->channels + comm->count++;
      *ch = (struct channel){.peer = (int)q, .large = large, .node = reports[q].node, .stream = h->stream, .wire = &h->wire, .view = h->view[c],
                             .laid = h->laid + c, .taken = h->taken + c, .declared = h->declared + c};
      if (ch->view.unit != (large ? comm->wire.block : MESH_FRAME) ||
          ch->view.units != (large ? MESH_STREAM_LARGE_UNITS : MESH_STREAM_SMALL_UNITS))
        status = EPROTO;
      if (!status) status = channel_buffer(ch);
      if (!status) atomic_store(&comm->attach->wrapped, 1);
    }
  }
  if (!status && !(comm->timed = calloc((size_t)GROUP_TIMES * comm->count + 1, sizeof *comm->timed))) status = ENOMEM;
  return status;
}

/* design/transport.md section 14, a communicator's init on NCCL's bootstrap at its id, the one path of every init
   (ncclMeshCommInitRank, blocking or not): its bootstrap's sockets opened (bootstrap_open), then made attached
   (comm_attach) and the nodes its bridge reaches read (reach_read), then every rank's report gathered (two ranks on one
   node fail every rank alike: comm_apart); each rank holding the stream toward each member's node it is joined to
   (comm_hold); a second gather carrying each rank's holding's status and its streams' first units (NCCL's own init
   exchanges each peer's connection through its bootstrap: ncclTransportP2pSetup), from which each rank makes its
   channels (comm_join).  A rank that cannot go on, a stream held by another (EBUSY) or gone (ENOLINK), a peer silent past
   its phase's end (bootstrap_open, gather: MESH_PLATFORM_NS) or a revoke fails the init as a value, made holding nothing of its
   bridge's (comm_leave) and its sockets closed; the next init reads the streams as they then stand.  Made, it keeps its
   sockets (closed at comm_free) */
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

/* a non-blocking init's thread (ncclMeshCommInitRank): comm_init, its failure's message kept in `why` (keep) before its
   result is stored in `async` */
static void *init_run(void *argument) {
  struct ncclComm *comm = argument;
  const ncclResult_t result = comm_init(comm);
  if (result) { comm->why = last; comm->why.id = keep(); }
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

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  return revoked(comm) ? fail(ncclInvalidUsage, "ncclCommFinalize after ncclCommRevoke (NCCL: not supported)") : ncclSuccess;
}

static void inflight_lose(struct ncclComm *comm);
static ncclResult_t comm_failure(struct ncclComm *comm);
/* a communicator stopped for good (a revoke's and a destroy's): `revoked` set, failing every group after it; every
   socket of its bootstrap shut down, so an init still running (a non-blocking one's), its connect and accept reading the
   revoke at their next look (bootstrap_open), fails as a value; and its failure words' stop (word [6]) set, which a
   GPU wait of its that reaches its bound records as its cause (METAL_REVOKED); no kernel stops at it, so its GPU work
   already encoded runs as encoded, each wait ending at its landing or its bound; under its guard */
static void nil_send(struct ncclComm *comm);
static void comm_stop(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST);
  nil_send(comm);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  if (comm->words) atomic_store_explicit(comm->words + 6, 1, memory_order_release);
  pthread_mutex_unlock(&comm->guard);
}

/* ncclCommRevoke, NCCL's: comm's operations in flight revoked and its resources kept (comm_stop alone: its streams stay
   held, the units its groups declared and supplied nothing for going at once as duds, nil_send, so its peers' positions
   land rather than wait), a host-path group of its returning at its next look and each of its GPU waits ending at its landing or its bound, every later
   group refused (ncclInvalidUsage), ncclCommGetAsyncError reading ncclSuccess once the revoke is done; from any thread,
   under comm's own lock alone, never one a group holds while it waits.  ncclCommDestroy then gives its streams back
   (comm_free) and frees it */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  if (!comm || revokeFlags != NCCL_REVOKE_DEFAULT) return fail(ncclInvalidArgument, "comm is NULL or revokeFlags is %d", revokeFlags);
  comm_stop(comm);
  return ncclSuccess;
}

/* ncclCommDestroy: comm stopped (comm_stop), its init's thread joined (a non-blocking init's, which the stop ends), its
   host-path groups returned (each at its next look) and the GPU work the library committed for it awaited, at most
   MESH_PLATFORM_NS (each of its waits ends at its landing or its bound), its groups in flight lost; then its streams
   given back (the bridge sends what it declared on them) and comm freed.  Work encoded in a caller's encoder is the
   caller's to finish first (NCCL's: no communicator is destroyed under its own operations in flight): a kernel of it
   still queued would write the stream given back (rdma/NCCL.md "Failures") */
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

/* a non-blocking init's state (ncclInProgress until it ends, then its failure, named again here: told, its kept message
   `why`, which init_run keeps where the init fails) or a launch's failure, else the communicator's failure
   (comm_failure: a revoke, a host-path group's, or one its GPU work recorded in its failure words), which a group that
   had already returned, its work enqueued, could not report */
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

/* a buffer in the node's region (a lease's): its handle the buffer, the region registered on every device of the bridge
   at device up; any other buffer is refused */
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

/* a lease of the region: its whole pages covering `size` (one page for 0), already registered on every device of the
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
    result = fail(ncclSystemError, "ncclMemAlloc of %zu bytes (%zu pages of %u bytes): the region of this node's bridge has %u of "
                  "its %u pages free (%s); a client sizes its leases to the region its node's bridge configuration states "
                  "(etc/bridge/README.md)", size, (bytes + quantum - 1) / quantum * m->block, m->pgsz, mesh_arena_free(a->context.M),
                  mesh_arena_pages(m), strerror(status));
  } else {
    *ptr = mesh_section_address(&a->context, lease, 0);
    a->leases[a->count++] = (struct lease){*ptr, lease.first, (size_t)lease.pages * a->context.M->pgsz, NULL, NULL, 0};
  }
  pthread_mutex_unlock(&attach_lock);
  return result;
}

/* nccl.h ncclMeshRegionExpired */
ncclResult_t ncclMeshRegionExpired(int *expired) {
  if (!expired) return fail(ncclInvalidArgument, "ncclMeshRegionExpired: expired is NULL");
  pthread_mutex_lock(&attach_lock);
  attach_settle();
  *expired = attaches && attach_expired(attaches);
  pthread_mutex_unlock(&attach_lock);
  return ncclSuccess;
}

/* the lease at `ptr` returned to the region (mesh_lease_return): its pages free for any client's next lease */
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
  attach_settle();
  pthread_mutex_unlock(&attach_lock);
  if (!found) return fail(ncclInvalidArgument, "ncclMemFree of %p: no lease of this process (ncclMemAlloc)", ptr);
  if (status) return fail(ncclInternalError, "ncclMemFree of %p: row %u of this node's region heads no lease of this process", ptr, row);
  return ncclSuccess;
}

/* the lease holding `ptr` (any byte of it): the Metal buffer over the whole lease (made at its first use, kept until
   ncclMemFree) and ptr's offset in it; 0, ENOENT where no lease of this process holds ptr, ENOMEM */
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

/* the lease of this process whose Metal buffer is `buffer` (lease_buffer's, nccl.h ncclMeshMemBuffer's): whether there
   is one */
static int lease_of_buffer(const void *buffer) {
  int found = 0;
  pthread_mutex_lock(&attach_lock);
  for (struct attach *a = attaches; a && buffer && !found; a = a->next)
    for (size_t i = 0; i < a->count && !found; i++) found = a->leases[i].buffer == buffer;
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

/* -- a communicator's channels and its failures (design/transport.md section 14).  Toward each member it is joined to, a
   channel of each class on that class of the stream its init held toward the member's node.  Both ends of a channel compute the
   same positions for a group (each collective's rounds in call order, then the point-to-point calls paired in order:
   mesh-plan.h mesh_group_lay), each publishes every position (a filler where it has nothing to send), and each
   publishes position t once it has consumed the peer's t - LAG: the peer then has consumed this end's t - 2 LAG, so no
   unit of either ring is written again before it is sent and read (NCCL's Simple protocol's slots, its head credit
   carried by the reverse direction's positions; mesh-plan.h struct mesh_wire).  A position is a run of its stream class's
   units, laid alike at both ends from the runs before it (this end's out units from its first, its in units from the
   peer's): one unit a filler's, a piece's bytes in whole units, and a run that would pass the ring's end starting at its
   next lap, the units it skips supplied as fillers, since the bridge sends a stream's units strictly in order.  A
   group declares its units to the bridge before it supplies any (declare), so a member that fails mid-group is
   replaced at its peers by duds, whose landings fail their groups (design/transport.md section 10), and a run's landing
   is its last unit's (the bridge releases a stream's units in order, and sends every unit of a holding after a dud as
   one). ---------- */

#define LAG MESH_LAG

static double now_s(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

/* a failed communicator's groups in flight lost (inflight_lose); its kept result, its kept message named
   (ncclGetLastError: told, nothing formatted again).  Under group_lock */
static ncclResult_t comm_told(struct ncclComm *comm) {
  inflight_lose(comm);
  return told(comm->failed, &comm->failure);
}

/* the communicator failed for good, made again by its caller (ncclCommDestroy, then an init): its first failure's
   result and message kept, the message formatted here once (comm_failure reads them), then comm_told.  Under
   group_lock */
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) __attribute__((format(printf, 3, 4)));
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) {
  if (!comm->failed) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(comm->failure.text, sizeof comm->failure.text, format, arguments);
    va_end(arguments);
    comm->failure.id = keep();
    comm->failed = result;
    nil_send(comm);
  }
  return comm_told(comm);
}

/* the communicator's failure: the one it kept; a revoke; one its GPU work recorded in its failure words (a wait's
   bound reached, its link lost, a dud landed: nccl-mesh-metal.h METAL_*; a counted all-to-all past its layout); else a
   stream it holds
   no longer its own (a bridge's restart, or a pairing lost and made again, opens every stream new: mesh-call.h
   mesh_stream_held), kept now (comm_fail), so ncclCommGetAsyncError reads it before any group runs on it; ncclSuccess
   where none.  Under group_lock */
static ncclResult_t comm_failure(struct ncclComm *comm) {
  if (comm->failed) return comm_told(comm);
  if (revoked(comm))
    return comm_fail(comm, ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
  _Atomic uint64_t *v = comm->words;
  const uint64_t at = v ? atomic_load_explicit(v + 2, memory_order_acquire) : 0, over = v ? atomic_load_explicit(v + 5, memory_order_acquire) : 0;
  if (at) {
    const uint64_t cause = atomic_load_explicit(v, memory_order_relaxed), h = atomic_load_explicit(v + 3, memory_order_relaxed);
    const uint32_t node = h < comm->count ? comm->channels[h].node : UINT32_MAX;
    if (cause == METAL_LOST)
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost (its stream's landed words ~0) at position %llu", node,
                       (unsigned long long)(at - 1));
    if (cause == METAL_DUD)
      return comm_fail(comm, ncclRemoteError, "position %llu on the channel to node %u landed as a dud: the group of the member there, or "
                       "of one it relays, failed, was revoked or ended (design/transport.md section 10)", (unsigned long long)(at - 1), node);
    return comm_fail(comm, ncclRemoteError, "position %llu on the channel to node %u landed nothing within %llu polls of a GPU wait (its "
                     "stream's bound, ncclMeshStream polls): the peer's work there began that much later, or its link was lost",
                     (unsigned long long)(at - 1), node, (unsigned long long)atomic_load_explicit(v + 4, memory_order_relaxed));
  }
  if (over)
    return comm_fail(comm, ncclInvalidUsage, "a counted all-to-all's counts passed what its layout carries (ncclMeshAlltoAllCounted: a rank's "
                     "rows past sendrows, a pair's past min(sendrows, capacity), or a rank's received rows past capacity); its rows past them "
                     "were not moved and its arrived word is 2");
  struct attach *a = comm->attach;
  for (uint32_t k = 0; comm->held && k < MESH_STREAMS; k++)
    if ((comm->held >> k & 1) && !mesh_stream_held(&a->context, a->holds[k].stream, a->holds[k].wire))
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost and paired again: communicator %s's stream there is no "
                       "longer its own (design/transport.md section 14); it is made again (ncclCommDestroy, then an init)",
                       a->holds[k].node, comm->name);
  return ncclSuccess;
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
   counts carry, within `limit` bytes of the buffer (the kernels' rule, nccl-mesh-kernels.h counted_send/counted_land) */
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
   rank no link of this node joins refused (the bridges forward nothing) */
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
        return fail(ncclInvalidUsage, "a call of %s moves bytes between rank %d and rank %u (node %u), which no link of this node joins: the "
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
/* a run of a stream class's units: its first, its count, and the units before it that it skips to start its lap (a run that
   would pass the ring's end starts at its next lap: mesh-plan.h mesh_wire), supplied as fillers */
struct run { uint64_t first; uint32_t units, skip; };
/* a group's schedule on one channel: its messages each way in position order (`out_at`, `in_at` the next not yet
   published or consumed) and, from its first position `start`, each position's runs, out (runs[2 (t - start)]) and in
   (runs[2 (t - start) + 1]) */
struct schedule { struct message *out, *in; size_t outs, ins, out_at, in_at; uint64_t start; struct run *runs; };

/* the run of `units` units from a stream class's next unit *at on a ring of `ring` units, *at past it */
static struct run run_from(uint64_t *at, uint64_t units, uint64_t ring) {
  const uint64_t first = *at % ring + units > ring ? (*at / ring + 1) * ring : *at;
  const struct run made = {first, (uint32_t)units, (uint32_t)(first - *at)};
  *at = first + units;
  return made;
}

/* a group's calls as mesh-plan.h's group, which mesh_group_lay lays (the one layout both ends make): call i's kind and
   its steps at rank r, steps[i * N + r] and count[i * N + r] (none where the rank's are not known) */
static uint32_t group_kind(const struct call *c) {
  return p2p(c) ? MESH_GROUP_POINT : hosted(c) ? MESH_GROUP_HOSTED : counted(c) ? MESH_GROUP_COUNTED : MESH_GROUP_COLLECTIVE;
}
static void group_of(const struct call *list, size_t n, uint32_t N, const struct mesh_step **steps, uint32_t *count,
                     struct mesh_group_call *calls, struct mesh_group *group) {
  for (size_t i = 0; i < n; i++) calls[i] = (struct mesh_group_call){group_kind(list + i), 0, count + i * N, steps + i * N};
  *group = (struct mesh_group){N, (uint32_t)n, 0, 0, calls};
}

/* the bytes position `at` carries of a channel's messages one way (0: none, a filler's), *k moved past those before it */
static size_t bytes_at(const struct message *m, size_t count, size_t *k, uint64_t at) {
  while (*k < count && at >= m[*k].first + m[*k].pieces) ++*k;
  size_t length = 0;
  if (*k < count && at >= m[*k].first) piece_at(m + *k, at, &length);
  return length;
}

/* the next number of comm's groups taken by a group it issues (`groups`), as the group takes group_lock, and its slot of
   the record cleared: a group that crosses no link records no unit (ncclMeshTimes gives it 0); schedule_group records
   a crossing group's in it.  Under group_lock */
static void group_numbered(struct ncclComm *comm) {
  if (comm->timed) memset(comm->timed + (comm->groups % GROUP_TIMES) * comm->count, 0, comm->count * sizeof *comm->timed);
  comm->groups++;
}

/* each channel's positions for the group, from the last scheduled one's end (mesh_group_lay), each step's positions
   and the receive before it (mesh-plan.h mesh_steps_before) kept, a receive with none done; and each position's runs,
   out from this end's next unit and in from the peer's, a run as many units as its piece's bytes (mesh_wire_units: a
   filler's one); the group's first out unit and last in unit on each channel recorded under the number it took
   (`timed`, group_numbered: ncclMeshTimes) */
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
      p->runs[2 * t] = run_from(ch->laid, mesh_wire_units(&comm->wire, ch->large, sent), ch->view.units);
      p->runs[2 * t + 1] = run_from(ch->taken, mesh_wire_units(&comm->wire, ch->large, got), ch->view.units);
    }
    uint64_t *timed = comm->timed[((comm->groups - 1) % GROUP_TIMES) * comm->count + h];
    timed[0] = positions ? 1 + p->runs[0].first : 0;
    timed[1] = positions ? p->runs[2 * positions - 1].first + p->runs[2 * positions - 1].units : 0;
    ch->end += positions;
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct steps *st = list[i].bound;
    if (st && mesh_group_before(group_kind(list + i), st->steps, st->count, st->before)) result = fail(ncclSystemError, "out of memory");
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

/* design/transport.md section 14: each channel's stream class declared through the units laid on it
   (mesh_stream_declare) before any of them is supplied, with no deadline (UINT64_MAX: a live producer's unit is never
   a dud), so a member that fails, is revoked or ends mid-group is replaced at its peers by duds when it says so
   (nil_send) or its holding ends; with `again`, each with positions still to publish declared anew (a completion's,
   ncclMeshComplete) */
static void declare(struct ncclComm *comm, int again) {
  for (uint32_t h = 0; h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    if (*ch->laid <= *ch->declared && !(again && ch->sent < ch->end)) continue;
    mesh_stream_declare(&comm->attach->context, ch->stream, ch->large, *ch->laid, UINT64_MAX);
    *ch->declared = *ch->laid;
  }
}

/* design/transport.md section 14: a communicator that can no longer supply what it declared (its group failed, or it
   was revoked or destroyed): every channel's declared units given the deadline now, so the bridge sends at once, as
   duds, each one no group supplied, and its peers' positions land rather than wait, their groups failing there (a dud
   is what is placed where a send must happen and its producer is gone, its bytes no value any consumer reads as a
   signal, design/rubrics.md M4; its peer's bridge tells it by its SEND and marks its landed word, section 10, so a
   member past a failed one fails rather than complete on filler, at any depth of a tree); a live holder's units are
   never given a deadline (declare: UINT64_MAX), so no dud reaches a peer of a producer still at work (design/rubrics.md
   C5, Quantities, the operator 10-20).  Only a holding still the communicator's own is written (mesh_stream_held): a
   stream opened new since (a restart or a pairing made again) may be another communicator's, which this one must not
   touch (design/rubrics.md M4); a give likewise (comm_let_go) */
static void nil_send(struct ncclComm *comm) {
  if (!comm->attach || !comm->channels) return;
  const uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct channel *ch = comm->channels + h;
    if (ch->declared && mesh_stream_held(&comm->attach->context, ch->stream, *ch->wire))
      mesh_stream_declare(&comm->attach->context, ch->stream, ch->large, *ch->declared, now);
  }
}

/* the group's schedule (schedule_group), its units declared (declare); a failure the communicator's (comm_fail: its
   channels stand mid-schedule); *made NULL where it failed */
static ncclResult_t schedule(struct ncclComm *comm, struct call *list, size_t n, struct schedule **made) {
  struct schedule *plans = calloc(comm->count + 1, sizeof *plans);
  *made = NULL;
  if (!plans) return fail(ncclSystemError, "out of memory");
  const ncclResult_t result = schedule_group(comm, list, n, plans);
  if (result) { schedules_free(comm, plans); return comm_fail(comm, result, "%s", last.text); }
  comm->crossed = 1;
  declare(comm, 0);
  *made = plans;
  return ncclSuccess;
}

/* a step may run once the receive before it is consumed (mesh_group_before: every earlier receive then is, by induction, as
   each receive's first piece waits on the one before it; a hosted all-to-all's steps wait on none) */
static int ready(const struct steps *s, uint32_t k) { return s->before[k] < 0 || s->done[s->before[k]]; }

/* the message of `m` (`count` of them, one way of a channel, in position order) whose positions hold `at`, or NULL */
static const struct message *holding(const struct message *m, size_t count, uint64_t at) {
  size_t lo = 0, hi = count;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (m[mid].first <= at) lo = mid + 1; else hi = mid;
  }
  return lo && at < m[lo - 1].first + m[lo - 1].pieces ? m + lo - 1 : NULL;
}

/* whether a channel's position `at` (its schedule p) publishes a piece that reads, in place, bytes its own consumption
   writes: one call's SEND and receive on one of its buffers over meeting bytes (mesh-plan.h mesh_compile's two-node
   direct exchange, whose SEND reads the operand its REDUCE combines into, both at the same positions of one channel).
   Such a consumption follows its position's publication: a completion on one track by its order (complete_track),
   the tracks by its item's `need` (nccl-mesh-metal.h metal_record), the host path by its loop (run) */
static int sends_what_lands(const struct schedule *p, uint64_t at) {
  const struct message *out = holding(p->out, p->outs, at), *in = holding(p->in, p->ins, at);
  if (!out || !in || out->c != in->c) return 0;
  const struct steps *st = out->c->bound;
  if (st->steps[out->step].buffer != st->steps[in->step].buffer || (st->read && st->read[out->step].buffer)) return 0;
  size_t sent, landed;
  const size_t from = out->offset + piece_at(out, at, &sent), to = in->offset + piece_at(in, at, &landed);
  return from < to + landed && to < from + sent;
}

/* a channel's out run supplied on the host (design/transport.md section 14): `length` bytes into its run's slots of the
   out ring (a filler's, `bytes` NULL, none), then, after a release fence, each unit's ready word 1 + its cycle, the units
   the run skips first, in order */
static void supply(const struct channel *ch, const struct run *r, const void *bytes, size_t length) {
  const uint64_t units = ch->view.units;
  unsigned char *slot = ch->view.out + (r->first % units) * ch->view.unit;
  if (bytes) memcpy(slot, bytes, length);
  atomic_thread_fence(memory_order_release);
  for (uint64_t u = r->first - r->skip; u < r->first + r->units; u++)
    atomic_store_explicit(ch->view.ready + u % units, 1 + u / units, memory_order_release);
}

/* a channel's in run on the host: 1 once it has landed (its last unit's landed word, its bits but MESH_DUD, at least 1 +
   its cycle: the bridge releases a stream's units in order), 0 before; -1 where its link was lost (a word ~0); -2 where
   it landed as a dud (MESH_DUD, the bridge's: design/transport.md section 10; its bridge sends every unit of a holding
   after a dud as one, so a run with a dud among its units has one as its last), which fails the group: its producer's
   failed, and no byte of it is read as a signal (design/rubrics.md M4) */
static int landing(const struct channel *ch, const struct run *r) {
  const uint64_t units = ch->view.units, last = r->first + r->units - 1;
  const uint64_t word = atomic_load_explicit(ch->view.landed + last % units, memory_order_acquire);
  if ((word & ~MESH_DUD) < 1 + last / units) return 0;
  return word == UINT64_MAX ? -1 : word & MESH_DUD ? -2 : 1;
}

/* every channel's positions published and consumed on the host: a SEND's piece supplied into its run, a REDUCE's
   combined into the result, a COPY's copied into the operand (and the result, where they differ: a direct exchange's
   sums go to a copy, as its SENDs read the operand); a landing that reads the link's loss or a dud fails it, and so
   does a revoke, at its next look: a peer's process gone or its communicator revoked ends its bindings, whose duds land
   here, and so does a member's past it, which sends its own units as duds once its group failed (comm_fail, nil_send)
   (design/transport.md sections 5, 10 and 14: a client's wait is a loop that stops on a word, never on a clock; a peer that
   computes for long is waited for).  Every look spins: nothing parks.  A position whose publication reads in place
   what its consumption writes (sends_what_lands) is consumed once it is published.  Its positions then count as
   awaited for the GPU work after it (`waited`) */
static ncclResult_t run(struct ncclComm *comm, struct schedule *plans) {
  for (;;) {
    if (revoked(comm)) return revoked_fail(comm);
    int open = 0;
    for (uint32_t h = 0; h < comm->count; h++) {
      struct channel *ch = comm->channels + h;
      struct schedule *p = plans + h;
      while (ch->consumed < ch->end) {
        const uint64_t at = ch->consumed;
        if (at >= ch->sent && sends_what_lands(p, at)) break;
        const struct run *r = p->runs + 2 * (at - p->start) + 1;
        const int landed = landing(ch, r);
        if (!landed) break;
        if (landed == -1)
          return fail(ncclRemoteError, "the link to node %u was lost at position %llu (its stream's landed words ~0)", ch->node, (unsigned long long)at);
        if (landed < 0)
          return fail(ncclRemoteError, "position %llu on the channel to node %u landed as a dud: the group of the member there, or of one "
                      "it relays, failed, was revoked or ended (design/transport.md section 10)", (unsigned long long)at, ch->node);
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
      }
      open |= ch->sent < ch->end || ch->consumed < ch->end;
    }
    for (uint32_t h = 0; !open && h < comm->count; h++) comm->channels[h].waited = comm->channels[h].end;
    if (!open) return ncclSuccess;
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

/* one group on the host: its leases' GPU points reached, its communicator not failed (comm_failure) and its GPU work finished,
   each within the platform's deadline; its calls planned and staged (a SEND read and a receive written in place), its
   units declared (schedule), run with the groups' lock released (run), then finished; a run's failure the
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
  group_numbered(comm);
  if (remote) result = comm_failure(comm);
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
      comm_fail(comm, result, "%s", last.text);
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

/* a step's result (nccl-mesh-metal.h struct metal_result): its errno a failure, named by `what` and the step's own message */
static ncclResult_t gpu(struct metal_result step, const char *what) {
  return step.status ? fail(ncclSystemError, "%s on the GPU: %s", what, step.message) : ncclSuccess;
}

/* a group issued and not complete: its communicator, its calls, its schedule and its positions [start, end) on each of
   its communicator's channels, and the one channel they lie on (`lone`: lone_of, read where the group is planned;
   comm->count where it has none, METAL_NONE where they lie on two or more); the groups in flight are in issue order, a
   communicator's holding consecutive positions on every channel of it */
struct inflight {
  uint64_t ticket;
  struct ncclComm *comm;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t *start, *end;
  uint32_t lone;
  struct inflight *next_group;
};

/* the channel all of a group's positions lie on (struct inflight `lone`): a function of its schedule alone, the
   positions both ends of each channel lay alike (schedule_group, mesh-plan.h mesh_group_lay) */
static uint32_t lone_of(const struct inflight *g) {
  uint32_t lone = g->comm->count;
  for (uint32_t h = 0; h < g->comm->count; h++)
    if (g->end[h] > g->start[h]) {
      if (lone != g->comm->count) return METAL_NONE;
      lone = h;
    }
  return lone;
}
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
      snprintf(losses[lost++].reason, sizeof grown->reason, "%s", comm->failed ? comm->failure.text : "its communicator was destroyed");
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
/* channel h's position `at` in none of comm's groups in flight, failed naming the channel's counts and each group's
   positions on it */
static ncclResult_t outside_groups(struct ncclComm *comm, uint32_t h, uint64_t at) {
  const struct channel *ch = comm->channels + h;
  char spans[256] = "";
  size_t used = 0;
  for (struct inflight *g = inflight; g && used < sizeof spans - 48; g = g->next_group)
    if (g->comm == comm)
      used += (size_t)snprintf(spans + used, sizeof spans - used, " [%llu, %llu)", (unsigned long long)g->start[h], (unsigned long long)g->end[h]);
  return fail(ncclInternalError, "position %llu of channel %u is in no group (sent %llu, consumed %llu, waited %llu; groups in flight on it:%s)",
              (unsigned long long)at, h, (unsigned long long)ch->sent, (unsigned long long)ch->consumed, (unsigned long long)ch->waited,
              *spans ? spans : " none");
}

/* channel h's position `at`'s in run as a GPU wait awaits it (nccl-mesh-metal.h struct metal_wait): its last unit's
   landed word, its first unit's slot (a landing's piece), its communicator's failure words */
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
   order */
static ncclResult_t arrive(struct metal_program *program, struct ncclComm *comm, struct channel *ch, uint64_t count) {
  const uint32_t h = (uint32_t)(ch - comm->channels);
  ncclResult_t result = ncclSuccess;
  for (uint64_t at = ch->waited; at < count && !result; at++) {
    const struct inflight *g = inflight_at(comm, h, at);
    if (!g) return outside_groups(comm, h, at);
    ch->waited = at + 1;
    const struct metal_wait wait = wait_of(comm, h, g->plans[h].runs + 2 * (at - g->start[h]) + 1, at);
    result = gpu(metal_spin(program, &wait), "a wait");
  }
  return result;
}

/* a counted all-to-all's end on the GPU, once every count landed (its counts receives consumed; at once where it has
   none): its own rows into place, the landed counts into `received` (stored system-coherent: the host reads them as they
   are, mesh 3981ca80^'s `received`) and `landed`, then its word released (1, or 2 where the counts pass its layout,
   which also fails the communicator through failure word [5], or where a wait before it recorded a failure;
   nccl-mesh-metal.h METAL_COUNTED_ARRIVED), the rows maybe
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

/* a counted all-to-all's counts from rank q consumed (its step k): their rows checked against the pair's bound
   (METAL_COUNTED_CHECK), and its end after its last counts receive in step order, which every other counts receive
   precedes (mesh_group_before: each after the one before it) */
static ncclResult_t counted_landed(struct metal_program *program, struct call *c, int q, uint32_t k, void *violation) {
  const int n = c->comm->nranks;
  const uint64_t *rseg = c->segments + n, R = seg_first(rseg, n);
  const struct steps *st = c->bound;
  ncclResult_t result = gpu(metal_counted(program, METAL_COUNTED_CHECK, &(struct metal_counted){
    .d = c->got_at.buffer, .dst = c->got_at.offset + R * sizeof(int64_t), .first = seg_first(rseg, q), .entries = (uint32_t)rseg[q],
    .scalar = counted_bound(c)}), "a counted all-to-all's bound");
  uint32_t last = k;
  for (uint32_t j = k + 1; j < st->count; j++)
    if (st->steps[j].op != MESH_STEP_SEND && st->steps[j].buffer == 2) last = j;
  if (!result && last == k) result = counted_end(program, c, violation);
  return result;
}

/* the fused call of the group being encoded (nccl.h ncclMeshFusableRooted): its publication's and its landing's records,
   written in place of their dispatches, and whether each was; under group_lock, for one launch_metal */
static struct { ncclMeshFused *out; const struct call *call; int published, landed, publishes, lands; } fusing;

/* whether a call's send buffer is its operand's own place in its receive buffer: the receive buffer itself, an
   all-gather's own segment of it (NCCL's in-place all-gather, recvbuff + rank x sendcount) */
static int sent_in_place(const struct call *c) {
  size_t length;
  const size_t before = c->what == MESH_ALLGATHER ? segment_at(c, c->comm->rank, &length) : 0;
  return same(c->send_at, (struct where){c->recv_at.buffer, c->recv_at.offset + before * SIZE[c->type]});
}

/* whether message m's piece at its channel's position (`offset`, `length`) is the fused call's whole operand, an
   all-gather's rank q's whole segment (its own published, its peer's landed) */
static int fused_piece(const struct message *m, int q, size_t offset, size_t length) {
  size_t count = elements(m->c);
  const size_t before = m->c->what == MESH_ALLGATHER ? segment_at(m->c, q, &count) : 0, z = SIZE[m->c->type];
  return fusing.out && m->c == fusing.call && !offset && length == m->bytes && m->offset == before * z && m->bytes == count * z;
}

/* channel h's next position consumed, as its track's item (metal_record; rdma/NCCL.md "Per-channel progress") or, `base`
   METAL_NONE, as dispatches in the stream's order (complete_track): a piece's landing awaited (its run's last unit's)
   and combined or copied where it belongs, an empty position's landing awaited alone, a receive's first piece after
   the receive before it (as a track's item, metal_depend: that receive's channel, track `base` + its index, consumed to
   its last; in the stream, consumed before it); the fused call's landing of its peer's whole operand (an all-gather's
   segment, its place in rank order the record's dst) recorded for its caller's kernel (metal_land_deferred) */
static ncclResult_t consume(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint32_t base) {
  struct channel *ch = comm->channels + h;
  const uint64_t at = ch->consumed;
  struct inflight *g = inflight_at(comm, h, at);
  if (!g) return outside_groups(comm, h, at);
  struct schedule *p = g->plans + h;
  struct message *m = p->in_at < p->ins && at >= p->in[p->in_at].first ? p->in + p->in_at : NULL;
  ncclResult_t result = ncclSuccess;
  if (base != METAL_NONE && m && at == m->first && m->c->bound->before[m->step] >= 0) {
    const struct steps *prior = m->c->bound;
    const int32_t j = prior->before[m->step];
    result = gpu(metal_depend(program, base + prior->channel[j], prior->first[j] + prior->pieces[j]), "a receive's dependency");
  }
  if (m && !result) {
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
      if (fusing.lands && !fusing.landed && fused_piece(m, q, offset, length)) {
        result = gpu(metal_land_deferred(program, m->c->combine, to.buffer, to.offset + m->offset, reduce ? length / SIZE[m->c->type] : length,
                                         &wait, fusing.out->land), "a fused landing's record");
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
    if (!result && at + 1 == m->first + m->pieces) {
      st->done[m->step] = 1; p->in_at++;
      if (counted(m->c) && st->steps[m->step].buffer == 2) result = counted_landed(program, m->c, q, m->step, comm->violation);
    }
  }
  if (!result && !m && ch->waited <= at) result = arrive(program, comm, ch, at + 1);
  if (!result) ch->consumed++;
  return result;
}

/* channel h's next position published, once the consumption of the peer's position LAG back has freed the units its
   run takes (its track item's credit, or, issued, already done): the units its run skips released as fillers, then its
   piece copied into its run's slots and its units released, or a filler's (an empty position); the fused call's whole
   operand (an all-gather's own segment) recorded for its caller's kernel, which stores it and releases its units
   (metal_send_deferred) */
static ncclResult_t publish(struct metal_program *program, struct ncclComm *comm, uint32_t h, int *published) {
  struct channel *ch = comm->channels + h;
  const uint64_t at = ch->sent;
  struct inflight *g = inflight_at(comm, h, at);
  if (!g) return outside_groups(comm, h, at);
  struct schedule *p = g->plans + h;
  struct message *m = p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
  const struct run *r = p->runs + 2 * (at - p->start);
  const struct metal_release out = release_of(ch, r, 0), skipped = release_of(ch, r, 1);
  ncclResult_t result = ncclSuccess;
  if (r->skip) result = gpu(metal_publish(program, &skipped), "a lap's skipped units");
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
    } else if (!ch->large && fusing.publishes && !fusing.published && fused_piece(m, comm->rank, offset, length)) {
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
  if (!result && !released) result = gpu(metal_publish(program, &out), m ? "a publication" : "an empty publication");
  if (!result) { ch->sent++; *published = 1; }
  return result;
}

static struct message *sending(struct inflight *g, uint32_t h, uint64_t at) {
  struct schedule *p = g->plans + h;
  return p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
}
static struct message *receiving(struct inflight *g, uint32_t h, uint64_t at) {
  struct schedule *p = g->plans + h;
  return p->in_at < p->ins && at >= p->in[p->in_at].first ? p->in + p->in_at : NULL;
}

/* the communicators a program touched, each once, until its end (program_end) */
static struct ncclComm *touching;
static void touch(struct ncclComm *comm) {
  if (comm->touched) return;
  comm->touched = 1;
  comm->next_touched = touching;
  touching = comm;
}

/* the channel every group in flight through `through` lies on alone, all of one communicator (each group's `lone`,
   declared where it was planned: launch_metal), or METAL_NONE where they lie on two or more channels or are of two
   communicators; a group with no position lies on none and joins any */
static uint32_t one_track(const struct inflight *through) {
  uint32_t h = METAL_NONE;
  for (const struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group) {
    if (g->comm != inflight->comm || g->lone == METAL_NONE) return METAL_NONE;
    if (g->lone == g->comm->count) continue;
    if (h != METAL_NONE && h != g->lone) return METAL_NONE;
    h = g->lone;
  }
  return h;
}

/* every group in flight through `through` completed where all of them lie on channel h of one communicator
   (one_track): the GPU command stream's own order is the dependency (rdma/NCCL.md "One track").  Each position is
   plain dispatches in position order: the next publication (publish: send_small, or a copy and a release, as
   issue_group encodes it) as soon as the host's order allows it, its credit the consumption LAG back encoded before it
   and, a SEND's first piece, the receive before it consumed; else the next consumption (consume: the bounded landing
   wait and the combine or take in one small dispatch, or a wait and then a combine), a receive's first piece after the
   receive before it, and a position whose publication reads in place what it writes only once published (a program
   whose SEND waits on a receive at or past its own first position, which mesh_group_advise names, fails here rather
   than read what was combined into).  So this rank's publication of position p runs after it consumed its peer's p - LAG, which the
   peer published after consuming this rank's p - 2 LAG: at most 2 LAG of its positions are unread at the peer, which
   its in ring holds (mesh-plan.h mesh_wire: room for 2 LAG + 1 runs).  A position is published before it is consumed,
   so a SEND that reads in place what its position's receive writes (sends_what_lands) reads it first.  No track is
   recorded, no thread picks, and no tracks dispatch runs: on one track the stream's order is the one serial order
   every rank builds alike, and no wait on it can hold another channel's */
static ncclResult_t complete_track(struct metal_program *program, struct inflight *through, uint32_t h, int *published) {
  struct ncclComm *const comm = through->comm;
  struct channel *ch = comm->channels + h;
  const uint64_t end = through->end[h];
  ncclResult_t result = ncclSuccess;
  touch(comm);
  metal_gate(program, comm->violation);
  while (!result && (ch->sent < end || ch->consumed < end)) {
    struct inflight *g = ch->sent < end ? inflight_at(comm, h, ch->sent) : NULL, *x = ch->consumed < end ? inflight_at(comm, h, ch->consumed) : NULL;
    const struct message *m = g ? sending(g, h, ch->sent) : NULL, *r = x ? receiving(x, h, ch->consumed) : NULL;
    if (ch->sent < end && ch->sent < ch->consumed + LAG && !(m && ch->sent == m->first && !ready(m->c->bound, m->step)))
      result = publish(program, comm, h, published);
    else if (ch->consumed < end && !(r && ch->consumed == r->first && !ready(r->c->bound, r->step)) &&
             (!x || ch->consumed < ch->sent || !sends_what_lands(x->plans + h, ch->consumed)))
      result = consume(program, comm, h, METAL_NONE);
    else
      result = fail(ncclInternalError, "channel %u's positions admit no order on one track (published %llu, consumed %llu, end %llu)", h,
                    (unsigned long long)ch->sent, (unsigned long long)ch->consumed, (unsigned long long)end);
  }
  return result;
}

/* every group in flight through `through` completed, on one track as plain dispatches (complete_track), else each
   channel progressing on its own (nccl-mesh-metal.h metal_record; rdma/NCCL.md "Per-channel progress"): channel h of
   the k-th communicator met is track base[k] + h, its publications to its communicator's last group through `through`
   each an item of its own (its credit the consumption LAG back; a SEND's first piece after the consumption of the
   receive before it), then its consumptions (a position whose publication reads in place what it writes, after that
   publication: sends_what_lands); the tracks run as one (metal_tracks).  No rank orders anything another rank must
   share */
static ncclResult_t complete(struct metal_program *program, struct inflight *through, int *published) {
  const uint32_t lone = one_track(through);
  if (lone != METAL_NONE) return complete_track(program, through, lone, published);
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
  uint64_t *consumed = result ? NULL : calloc((size_t)tracks + 1, sizeof *consumed), *end = NULL;
  if (!result && !consumed) result = fail(ncclSystemError, "out of memory");
  for (size_t k = 0; k < comms && !result; k++) {
    struct ncclComm *const c = comm[k];
    touch(c);
    metal_gate(program, c->violation);
    free(end);
    if (!(end = calloc((size_t)c->count + 1, sizeof *end))) { result = fail(ncclSystemError, "out of memory"); break; }
    for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group)
      if (g->comm == c) memcpy(end, g->end, c->count * sizeof *end);
    for (uint32_t h = 0; h < c->count && !result; h++) {
      struct channel *ch = c->channels + h;
      const uint32_t track = base[k] + h;
      consumed[track] = ch->consumed;
      while (!result && ch->sent < end[h]) {
        const uint64_t at = ch->sent;
        struct inflight *g = inflight_at(c, h, at);
        const struct message *m = g ? sending(g, h, at) : NULL;
        result = gpu(metal_record(program, track, METAL_PUBLISH, at, at >= LAG ? at - LAG + 1 : 0), "a publication's record");
        if (!result && m && at == m->first && m->c->bound->before[m->step] >= 0) {
          const struct steps *prior = m->c->bound;
          const int32_t j = prior->before[m->step];
          result = gpu(metal_depend(program, base[k] + prior->channel[j], prior->first[j] + prior->pieces[j]), "a SEND's dependency");
        }
        if (!result) result = publish(program, c, h, published);
      }
      while (!result && ch->consumed < end[h]) {
        const uint64_t at = ch->consumed;
        const struct inflight *g = inflight_at(c, h, at);
        result = gpu(metal_record(program, track, METAL_CONSUME, at, g && sends_what_lands(g->plans + h, at) ? at + 1 : 0),
                     "a consumption's record");
        if (!result) result = consume(program, c, h, base[k]);
      }
    }
  }
  if (!result) result = gpu(metal_record(program, METAL_NONE, 0, 0, 0), "the tracks' end");
  if (!result) result = gpu(metal_tracks(program, tracks, consumed), "the tracks");
  else metal_record(program, METAL_NONE, 0, 0, 0);
  free(end); free(consumed); free(comm); free(base);
  return result;
}

/* a group's publications that need no wait, each channel's in order: its earlier positions published, the positions
   LAG back consumed (empty ones freely) and the peer's landing there awaited before, a SEND's receives before it
   consumed */
static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
  metal_gate(program, comm->violation);
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    struct channel *ch = comm->channels + h;
    while (!result && ch->sent >= g->start[h] && ch->sent < g->end[h]) {
      const uint64_t at = ch->sent, need = at >= LAG ? at - LAG + 1 : 0;
      if (need > ch->waited) break;
      while (ch->consumed < need) {
        struct inflight *x = inflight_at(comm, h, ch->consumed);
        struct schedule *p = x ? x->plans + h : NULL;
        if (!p || (p->in_at < p->ins && ch->consumed >= p->in[p->in_at].first)) break;
        ch->consumed++;
      }
      const struct message *m = sending(g, h, at);
      if (ch->consumed < need || (m && at == m->first && !ready(m->c->bound, m->step))) break;
      result = publish(program, comm, h, published);
    }
  }
  return result;
}

/* the groups in flight: completing through a group, every group up to it completed (complete), then any later group's
   publications that need no wait; issuing a group, its publications that need no wait, so a group issued encodes no
   wait at all; each group's dispatches recording a failed wait in its own communicator's failure words (metal_gate) */
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
  char why[sizeof last.text];
  snprintf(why, sizeof why, "%s", last.text);
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
   direct exchange's sums apart; then the group scheduled and its units declared (schedule), in flight after the others,
   issued (with issue) or completed with every group before it; its results into recvbuff as it completes; the
   program's end signals `finished` where the library made its encoder.  The Metal 4 form makes its channels' stream buffers
   resident through the stream's residency set; without a workspace its scratch is the communicator's (workspace), so it
   is not issued.  A fused group on the classic form takes the stream's residency set too (metal_begin), through which
   the buffers its deferred steps' bindings name are made resident for the caller's kernels. */
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
  else if (!result && !metal4 && list[0].fused && !list[0].residency)
    result = fail(ncclInvalidArgument, "a fused crossing on a classic ncclMeshStream without its residency set (the caller's kernels reach "
                                       "the library's buffers by address)");
  else if (!result && list[0].fused && (issue || n != 1 || !sent_in_place(list) ||
                                        (list[0].what != MESH_ALLREDUCE && list[0].what != MESH_BROADCAST && list[0].what != MESH_REDUCE &&
                                         list[0].what != MESH_ALLGATHER)))
    result = fail(ncclInvalidUsage, "a fused crossing is one in-place ncclAllReduce, ncclBroadcast, ncclReduce or ncclAllGather, not issued "
                                    "(ncclMeshFusableRooted)");
  if (!result && revoked(comm)) result = revoked_fail(comm);
  pthread_mutex_lock(&group_lock);
  if (!result) group_numbered(comm);
  if (!result && remote) result = comm_failure(comm);
  if (!result) result = plan_calls(comm, list, n);
  if (!result && list[0].fused && (!remote || !list[0].bound || list[0].bound->buffers != 1))
    result = fail(ncclInvalidUsage, "a fused crossing whose program copies its operand before publishing it (ncclMeshFusableRooted)");
  if (result) { pthread_mutex_unlock(&group_lock); counted_failed(list, n); calls_free(list, n); return result; }
  if (finished) metal_collect(finished);
  void *const violation = remote ? comm->violation : NULL;
  const char *message = NULL;
  struct metal_program *program = metal4 ? metal_begin4(list[0].encoder, list[0].table, list[0].residency, violation, list[0].polls, &message)
                                         : metal_begin(list[0].command, list[0].encoder, list[0].fused ? list[0].residency : NULL, violation,
                                                       list[0].polls, &message);
  if (!program) {
    pthread_mutex_unlock(&group_lock); counted_failed(list, n); calls_free(list, n);
    return fail(ncclSystemError, "%s", message);
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
    /* a program's scratch, a tree's or a ring's (mesh-plan.h mesh_compile: a forwarding node's partials, and its own
       operand where it sends that to two nodes or more); the two-node direct exchange has none, its SEND reading the
       operand its REDUCE combines into, each position published before it is consumed (sends_what_lands) */
    for (uint32_t b = 1; st && b < st->buffers && !result; b++) {
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
        g->lone = lone_of(g);
        g->list = list; g->n = n; g->ticket = ticket = ++tickets; owned = 0;
        if (inflight_last) inflight_last->next_group = g; else inflight = g;
        inflight_last = g;
        scheduled = 1;
        if (list[0].fused) {
          const int what = list[0].what, rooted = comm->rank == list[0].root, both = what == MESH_ALLREDUCE || what == MESH_ALLGATHER;
          fusing = (typeof(fusing)){list[0].fused, list, 0, 0, both || (what == MESH_BROADCAST) == rooted, both || (what == MESH_REDUCE) == rooted};
        }
        if (!result) result = walk(program, issue ? g : NULL, issue ? NULL : g, &published);
        if (!result && list[0].fused && (fusing.published != fusing.publishes || fusing.landed != fusing.lands))
          result = fail(ncclInvalidUsage, "a fused crossing whose program is not this rank's publication and landing of its whole operand "
                                          "(ncclMeshFusableRooted)");
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
    const char *message = NULL;
    struct metal_program *program = stream->argumentTable
      ? metal_begin4(stream->commandEncoder, stream->argumentTable, stream->residencySet, inflight->comm->violation, stream->polls, &message)
      : metal_begin(stream->commandBuffer, stream->commandEncoder, NULL, inflight->comm->violation, stream->polls, &message);
    if (!program) result = fail(ncclSystemError, "%s", message);
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

/* unit u's time in a stream class's time ring (mesh-call.h struct mesh_stream_view `posted_at`, `landed_at`), read as
   ready and landed words are: its bridge's mach_absolute_time ticks; 0 where it has none */
static uint64_t time_of(const _Atomic uint64_t *ring, uint64_t u) {
  const _Atomic uint64_t *entry = ring + 2 * (u % MESH_STREAM_TIMES);
  const uint64_t tag = atomic_load_explicit(entry, memory_order_acquire), at = atomic_load_explicit(entry + 1, memory_order_acquire);
  return tag == 1 + u && atomic_load_explicit(entry, memory_order_relaxed) == 1 + u ? at : 0;
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
  if (!comm->timed && comm->nranks > 1) return fail(ncclInvalidUsage, "communicator %s holds no stream", comm->name);
  /* the bridges' ticks (mach_absolute_time) in ns of the same clock, CLOCK_UPTIME_RAW's */
  static mach_timebase_info_data_t base;
  pthread_mutex_lock(&group_lock);
  if (!base.denom) mach_timebase_info(&base);
  for (uint64_t i = 0; i < count; i++) {
    const uint64_t g = first + i;
    uint64_t begun = UINT64_MAX, done = 0;
    int timed = g < comm->groups && comm->groups - g <= GROUP_TIMES;
    for (uint32_t h = 0; timed && h < comm->count; h++) {
      const uint64_t *units = comm->timed[(g % GROUP_TIMES) * comm->count + h];
      if (!units[0]) continue;
      const uint64_t posted = time_of(comm->channels[h].view.posted_at, units[0] - 1),
                     landed = time_of(comm->channels[h].view.landed_at, units[1] - 1);
      timed = posted && landed;
      begun = posted < begun ? posted : begun;
      done = landed > done ? landed : done;
    }
    timed = timed && begun != UINT64_MAX;
    issued[i] = timed ? (uint64_t)((unsigned __int128)begun * base.numer / base.denom) : 0;
    ended[i] = timed ? (uint64_t)((unsigned __int128)(done > begun ? done : begun) * base.numer / base.denom) : 0;
  }
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}


/* whether a rank's moves of a program are at most one SEND and one receive, each of every part of what it moves: the
   whole operand in one piece each way, or (`gather`, an all-gather's) one segment's every part, its own sent and another's
   received */
static int whole_moves(const struct mesh_program *program, uint32_t rank, int gather) {
  uint32_t sends = 0, receives = 0;
  for (uint32_t i = program->first[rank]; i < program->first[rank + 1]; i++) {
    const struct mesh_move *m = program->move + i;
    const uint32_t s = program->segment[m->part];
    uint32_t parts = 0, within = 0;
    for (uint32_t p = 0; p < program->parts; p++) {
      const int of = !gather || program->segment[p] == s;
      parts += of;
      within += of && p >= m->part && p < m->part + m->parts;
    }
    const int send = m->op == MESH_STEP_SEND;
    if (within != m->parts || m->parts != parts || (gather && send != (s == rank)) || (send ? sends++ : receives++)) return 0;
  }
  return 1;
}

ncclResult_t ncclMeshFusableRooted(ncclComm_t comm, ncclMeshFuseKind kind, int root, size_t count, ncclDataType_t type, ncclRedOp_t op,
                                   int *publishes, int *lands) {
  if (!publishes || !lands) return fail(ncclInvalidArgument, "publishes or lands is NULL");
  *publishes = *lands = 0;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int what = kind == ncclMeshFuseBroadcast ? MESH_BROADCAST : kind == ncclMeshFuseReduce ? MESH_REDUCE
                 : kind == ncclMeshFuseAllGather ? MESH_ALLGATHER : MESH_ALLREDUCE;
  if (comm->nranks != 2 || !count || root < 0 || root > 1 || (!has_root(what) && root) || (int)type < 0 || (int)type >= ncclNumTypes ||
      (reducing(what) && (op != ncclSum || type != ncclFloat16)))
    return ncclSuccess;
  const uint64_t bytes = count * SIZE[type];
  pthread_mutex_lock(&group_lock);
  /* the answer is the communicator's program, its table's (the caller's, or the structural one compiled on its map, here
     where no group compiled it yet: a call records its crossings before its first group), taken at the call's operand
     (an all-gather's every rank's piece, elements) */
  if (comm_compile(comm) != ncclSuccess) { pthread_mutex_unlock(&group_lock); return ncclSuccess; }
  const struct mesh_program *program = program_of(comm, what, root, (what == MESH_ALLGATHER ? (uint64_t)comm->nranks : 1) * bytes);
  /* a two-rank all-reduce sends its operand from the operand its sum is made in (mesh-plan.h mesh_compile: the two-node
     direct exchange), so a fusable program has no scratch; its fused steps send from the producer's own store and land
     into the consumer's; a configured form that cuts the operand (two-phase, chunked) moves it in several pieces and is
     not fusable; an all-gather's piece is its own segment, sent in one piece, and its peer's, received in one */
  const int fusable = program && !program->scratch && bytes <= comm->wire.small &&
                     !mesh_wire_large(&comm->wire, bytes, SIZE[type]) && whole_moves(program, (uint32_t)comm->rank, what == MESH_ALLGATHER);
  pthread_mutex_unlock(&group_lock);
  if (!fusable) return ncclSuccess;
  const int rooted = comm->rank == root, both = what == MESH_ALLREDUCE || what == MESH_ALLGATHER;
  *publishes = both || (what == MESH_BROADCAST) == rooted;
  *lands = both || (what == MESH_REDUCE) == rooted;
  return ncclSuccess;
}

const char *ncclMeshMetalSteps(void) { return metal_steps(); }
const char *ncclMeshMetalPrelude(void) { return metal_prelude(); }

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
      return fail(ncclInvalidArgument, "ncclMeshAlltoAllCounted on the Metal path: received and arrived are host memory in "
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
   shrink and `color`, the same on every member, attached (comm_attach), its forms comm's; its streams held and its
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

/* ncclCommSplit: every rank's (color, key) and its report (reach_read; none for NCCL_SPLIT_NOCOLOR) all-gathered on comm,
   the ranks of this color ordered by key, then rank; two ranks of a color on one node refused alike (comm_apart), each
   rank's streams toward its color's nodes held (comm_hold: a stream its parent holds refuses it, EBUSY),
   and a second all-gather on comm carrying each rank's holding's status and its streams' first units, from which the
   child's channels are made (comm_join): NCCL's bootstrapSplit (the child's exchange over its parent's), with the init's
   two gathers.  A failure at any rank of a color fails that color's split on each of its ranks as a value, what it held
   let go (comm_free) */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  if (depth) return fail(ncclInvalidUsage, "ncclCommSplit inside a group");
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; struct report report; } mine = {color, key, {.rank = comm->rank}}, *every = calloc((size_t)n, sizeof *every);
  struct binding { int32_t status; uint32_t fresh; uint64_t start[MESH_STREAMS][2]; } bound = {0, 0, {{0}}}, *binds = calloc((size_t)n, sizeof *binds);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  struct report *reports = calloc((size_t)n, sizeof *reports);
  *newcomm = NULL;
  if (color != NCCL_SPLIT_NOCOLOR) reach_read(comm->attach, &mine.report);
  ncclResult_t result = every && binds && ranks && reports ? ncclAllGather(&mine, every, sizeof mine, ncclUint8, comm, NULL)
                                                           : fail(ncclSystemError, "out of memory");
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
    bound.fresh = own->fresh;
    memcpy(bound.start, own->start, sizeof bound.start);
  }
  else if (result && count) bound.status = ECANCELED;
  if (gathered) {
    const ncclResult_t confirmed = ncclAllGather(&bound, binds, sizeof bound, ncclUint8, comm, NULL);
    if (!result) result = confirmed;
  }
  for (int i = 0; !result && i < count; i++)
    if (binds[ranks[i]].status) result = hold_fail((*newcomm)->name, i, binds[ranks[i]].status);
  for (int i = 0; !result && i < count; i++) {
    memcpy(reports[i].start, binds[ranks[i]].start, sizeof reports[i].start);
    reports[i].fresh = binds[ranks[i]].fresh;
  }
  int status = 0;
  if (!result && count > 1 && (status = comm_join(*newcomm, reports)))
    result = fail(ncclSystemError, "communicator %s: its channels on the streams it held: %s", (*newcomm)->name, strerror(status));
  if (result && *newcomm) { comm_free(*newcomm); *newcomm = NULL; }
  free(every); free(binds); free(ranks); free(reports);
  return result;
}

/* ncclCommShrink: comm's ranks but the excluded ones, in rank order, with no exchange among them (a departed member
   answers none), so holding no stream: a shrink to two ranks or more, whose channels would be none, is refused (ENOLCK);
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
    : count > 1 ? fail(ncclInvalidUsage, "a shrink of %s to %d ranks would hold no stream (ENOLCK): it makes no exchange among them, and only "
                       "an exchange holds a communicator's streams (an init's bootstrap, ncclCommSplit over its parent)", comm->name, count)
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
   reporting its node and the nodes its bridge reaches; the communicator, named by the id, holds its streams as every init does */
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

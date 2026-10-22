/* libnccl-mesh (nccl.h): NCCL's API on the streams of its node's bridge (design/transport.md section 14).  The process
   is one client of its node's bridge (an attach, whose slot owns the process's leases of the region, ncclMemAlloc, and
   which holds the bridge's stream toward each node its communicators reach, as it holds rows: mesh-call.h
   mesh_stream_take).  A communicator's init holds the stream toward each other member's node and hands each holding's
   pages posted so far (the bridge's facts) to its peer through NCCL's bootstrap, keeping the two ends' difference as a
   statistic (on a live stream they are equal); its channels are those holdings, one communicator a stream
   (comm_hold), and its map the nodes its members' bridges reach.  It names nodes and its own pages, never a link, port,
   lane, frame, queue pair or SEND size.  A group's calls are planned by the one collective planner, each step one entry
   of its holding's send list (the pages it reads) or landing list (the pages it lands in), presented to the bridge,
   published by one count and awaited on another, the receives combined on the host or the GPU: zero copy, an operand
   or a piece that is not a lease's page failing as a value (nccl.h, Memory). */
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



/* -- communicators -------------------------------------------------------------------------------- */

/* an operator made on the communicator: PreMulSum's scalar (`value`, or where it lies, `at`, read at each host-path
   call: ncclScalarDevice), or the partials merge's row of `unit` elements */
struct redop { int used; ncclDataType_t type; unsigned char value[8]; uint32_t unit; const void *at; };
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; };

/* a channel (design/transport.md section 14): toward rank `peer` on node `node`, the holding of the stream toward that
   node its process holds (mesh-call.h mesh_stream_view: its two lists and its counts), and the Metal buffer over the
   page that holds the counts (`buffer`: `ready`, `sent` and `landed` at those offsets in it).  Its entries, each list
   in its own order: `sends` and `lands` laid by the groups scheduled, `out` and `in` their pages (the value each count
   reaches once they are whole), `published` send entries published, `consumed` landing entries consumed, `waited`
   landing entries awaited by the GPU work encoded, and `sent_waited` the most of `sent` a wait encoded has awaited; the
   two ends' difference as its init took the stream (`differ`: [0] the peer's receives posted less this end's SENDs
   posted, [1] the peer's SENDs posted less this end's receives posted, each 0 on a live stream: comm_join), a statistic
   nothing reads to decide anything */
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
/* a lease of a communicator's pool (struct ncclComm `spares`): its address and bytes, the group holding it (its
   number + 1: group_numbered; 0 none), the ends of the programs that named it not yet seen (`uses`, its use count:
   design/pages-and-functions.md, Freeing 6; stamp_end, ends_seen), and whether it was given back since its
   communicator's last program end (`given`) */
struct spare { void *address; size_t bytes; uint64_t holder; uint32_t uses; int given; };
/* a program's end not yet seen (struct ncclComm `ends`, by its stamp word, nccl-mesh-metal.h METAL_STAMP + its index):
   the number it stores there (0: the word idle), and the leases of the pool it named (indices into `spares`) */
struct end { uint64_t number; uint32_t count, *leases; };

/* a communicator: rank r on the bridge of node members[r]; its map (map_of: its members joined where their bridges
   reach each other's nodes, the routing table mesh_routes'), made at its init alike by every member and kept; the page
   its entries start on (`page`, its bridge's region's); its message paths, the caller's (count 0: none), else the map's
   shortest-path trees (`trees`); its table of programs (`entries`), compiled where a call or a query first needs it
   (`compiled`: comm_compile): its configured forms first (`forms`, ncclMeshCommForms, which its splits and shrinks
   keep, taken again at the compile after a change), then the caller's programs (`given`, `listed` of them, kept for its
   life), else the structural ones along its paths or trees; its operators made on it (`ops`, `nops` slots, a freed one
   reused); its name, and the count of splits and shrinks made of it, which name theirs; the process's attach it holds
   (`attach`, until ncclCommDestroy); the attach's holds of streams it uses (`held`, a bit a hold; `binding` while it
   counts among their users); its channels (`count`), one toward each member whose node its bridge reaches, toward[r]
   rank r's (-1: none); its pool of leases (`spares`, `nspares`: the pages its groups send from and land in that are not
   the caller's, spare_take), its programs' ends not yet seen by stamp word (`ends`, `nstamps` words: those of `pending`
   in use, those of `idle` free) and the numbers given them (`ended`: stamp_end, ends_seen); its failure words (`words`,
   nccl-mesh-metal.h: its GPU work's failure records, which the host reads and no kernel halts on, and the stop a revoke
   sets; `violation` the Metal buffer over them) and its failure once kept (`failed`, its message `failure`: comm_fail);
   `last`, the count of the last GPU program signalling `finished` that touched it, and that program's command buffer
   (`command`, retained), (`touched`, `next_touched` while one is encoded: program_end), and the number its next group
   issued takes (`groups`: group_numbered) and, of its last GROUP_TIMES groups, group g's at g mod GROUP_TIMES, each
   channel's first send entry of the group and its last landing entry, each plus one (`timed`, 0 where the group has
   none on it: schedule_group, ncclMeshTimes); `runs`, its host-path groups running; `revoked`, set by ncclCommRevoke
   and ncclCommDestroy, failing every group after it; `peers`, its bootstrap's sockets, kept for its life once its init
   succeeds (NCCL's bootstrapState), which a revoke shuts down and comm_free closes, under `guard`, and the end of its
   bootstrap's present phase (`until`, bootstrap_open, gather).  A non-blocking init (nccl.h ncclMeshCommInitRank) runs
   on `init` (`initing`: ncclCommDestroy joins it), `async` ncclInProgress until it ends and `why` its failure's message
   (its id 0 while none is kept) */
struct attach;
/* the groups a communicator's record of units holds (`timed`): as many as the Metal 4 argument ring's records
   (nccl-mesh-metal.m RECORDS, a group's dispatches one at least), within which a caller bounds the groups it encodes ahead
   (ncclMeshMetal4Records) */
#define GROUP_TIMES 65536u
struct ncclComm {
  struct mesh_link_map map;
  uint64_t page;
  struct mesh_trees paths, trees;
  struct entry *entries, *given;
  ncclMeshForm *forms;
  uint32_t programs, splits, *members, form_count, count, nops;
  int nranks, rank, listed, compiled, binding, touched;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  struct redop *ops;
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
  struct spare *spares;
  uint32_t nspares, nstamps, *pending, npending, *idle, nidle;
  struct end *ends;
  uint64_t ended;
  _Atomic uint64_t *words;
  void *violation;
  uint64_t last, groups, (*timed)[2];
  void *command;
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

/* the trees and the table compiled on the map; with `all`, the map, the caller's paths and programs too */
static void routes_free(struct ncclComm *comm, int all) {
  trees_free(&comm->trees);
  comm->compiled = 0;
  for (uint32_t e = 0; e < comm->programs; e++) program_free(&comm->entries[e].program);
  free(comm->entries);
  comm->entries = NULL; comm->programs = 0;
  if (!all) return;
  trees_free(&comm->paths); map_free(&comm->map);
  for (int e = 0; e < comm->listed; e++) program_free(&comm->given[e].program);
  free(comm->given);
  comm->given = NULL; comm->listed = 0;
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
/* comm's pool returned to the region (ncclMemFree), after its holdings are given back (comm_leave: the
   bridge posts nothing more of a given holding, so only rows a unit already posted names stay held, mesh-dataflow.c
   mesh_pinned) */
static void comm_spares_free(struct ncclComm *comm) {
  for (uint32_t k = 0; k < comm->nspares; k++) ncclMemFree(comm->spares[k].address);
  free(comm->spares);
  comm->spares = NULL; comm->nspares = 0;
}
static void comm_free(struct ncclComm *comm) {
  bootstrap_close(comm);
  routes_free(comm, 1);
  for (uint32_t k = 0; k < comm->scratches; k++) metal_release(comm->scratch[k]);
  for (uint32_t k = 0; k < comm->spents; k++) metal_release(comm->spent[k]);
  for (uint32_t h = 0; h < comm->count; h++) metal_release(comm->channels[h].buffer);
  if (comm->violation) metal_release(comm->violation);
  metal_release(comm->command);
  comm_leave(comm);
  free(comm->scratch); free(comm->scratch_bytes); free(comm->spent); free(comm->channels); free(comm->toward); free(comm->timed);
  for (uint32_t w = 0; comm->ends && w < comm->nstamps; w++) free(comm->ends[w].leases);
  free(comm->ops); free(comm->ends); free(comm->pending); free(comm->idle);
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

/* `what` (at `root`) along `trees` (mesh-plan.h mesh_compile), each tree's part cut into `chunks` (1: uncut, each
   step's piece one entry, which the bridge cuts into SENDs), an all-reduce single-phase where `whole`,
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

/* whether a tree rooted at r takes a share above 0 (a log weight above -inf): one that carries r's segment */
static int root_carried(const struct mesh_trees *trees, uint32_t r) {
  for (uint32_t t = 0; t < trees->count; t++)
    if (trees->root[t] == r && trees->log_weight[t] > -INFINITY) return 1;
  return 0;
}

/* a configured form's entries (ncclMeshCommForms): its collective along the communicator's trees (`along`) or the map's
   packed trees (mesh_trees_pack at the form's eps and most, made here: mesh cc8a6f8e^ comm_routes' sets[1]; a segment
   cut among a root's trees by the form's shares, else equally, never by the flows the packing infers from the map's
   links; a tree of share 0 carries none of it; the shortest trees cut alike where the form gives shares), whole or two-phase, at its chunks, below its bound; a rooted one at
   its root, or at each root (-1), none where its root is past the communicator's ranks (a split's).  A root every one
   of whose trees takes share 0 has no tree to carry its segment: the form fails there as a value */
static ncclResult_t compile_form(struct ncclComm *comm, const struct mesh_trees *along, const ncclMeshForm *x) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_trees packed = {n, 0, NULL, NULL, NULL}, weighed = {n, 0, NULL, NULL, NULL};
  const struct mesh_trees *trees = along;
  ncclResult_t result = ncclSuccess;
  /* the shortest trees (the caller's paths, else one a root) cut by the form's shares: root r's k-th tree in their
     order shares[r x most + k], a tree past `most` none */
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
    /* the single-phase all-reduce sums up each node's heaviest tree whatever its weight (mesh_compile) */
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

/* the communicator's table on its map: its configured forms first, in their order (compile_form), then the caller's
   programs where it listed some (ncclMeshCommInitRank), else, along the caller's paths (checked against the map) or
   else the map's shortest-path trees (mesh_trees_shortest: each member's in-tree of the routes over the links between
   members), the single-phase all-reduce (each member's sum up its own tree; the direct exchange where every two members
   are linked), the reduce-scatter and the all-gather (segment s up and down s's trees), and each root's reduce and
   broadcast, uncut and of any size (mesh_compile; mesh 2bf5a50d^ comm_routes's table, its entries along the routes): a
   call takes the first entry that fits it (program_of), so a collective no form names, or a size past every form's
   bound, runs the caller's program or the default.  Where trees are compiled (a form, or no programs listed), a map
   whose links do not join its members fails it, naming two: the bridges forward nothing, so no route crosses a node
   outside the communicator */
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

/* the communicator's table made where it is not (comm_routes), each move of each program between two members a link
   joins (the caller's programs too: the bridges forward nothing, so a route through another node is refused as a
   value); under group_lock */
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
   compiled before, by an earlier group or ncclMeshFusableRooted, dropped), ahead of the caller's programs where it
   listed some: its members compile its table alike from them, so every member sets them at the same point between its
   groups (a group in flight keeps its own steps, never the table) */
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
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
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

/* a lease: its address, its first row, its bytes, and the Metal buffer over it (lease_buffer, made at its first use;
   NULL until then) */
struct lease { void *address; uint32_t row; size_t bytes; void *buffer; };
/* design/transport.md section 14: the stream toward node `node` the process holds (`users` the communicator on it, 0
   none: the hold vacant), its holding's view, and the identity it was taken by (`wire`: mesh-call.h mesh_stream_held,
   comm_failure) */
struct hold { uint32_t node, stream, users, padding; struct mesh_stream_view view; uint64_t wire; };
struct attach {
  struct mesh_ctx context;
  struct lease *leases;
  size_t count, room;
  uint64_t comms;
  /* whether the library made a Metal buffer over the region (a lease's, lease_buffer; a holding's counts', channel_buffer):
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
   the bridge posts nothing more of the holding and frees the stream); under attach_lock */
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
/* a communicator's holds let go (comm_let_go), its pool's leases returned (comm_spares_free), and its hold of its
   attach let go */
static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) comm_let_go(comm);
  pthread_mutex_unlock(&attach_lock);
  comm_spares_free(comm);
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) {
    comm->attach->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
}

/* a rank's report at a communicator's init (NCCL's bootstrap, below): its status (0, else why it cannot go on, so every
   rank fails alike), its bridge's node and the nodes that bridge reaches with each one's bandwidth, and the streams free
   to take, each one's node and identity (`frees` of them: reach_read, mesh-call.h mesh_streams_free), and, at the
   second gather, the pages its stream toward each of them has posted over its pairing as this rank took it
   (posted[k][0] its SENDs, posted[k][1] its receives: the bridge's facts, mesh.h struct mesh_stream `posts`, its two
   lanes' units together, constant until this rank presents, whose difference from the peer's the channel keeps as a
   statistic, comm_join) */
struct report {
  int32_t rank, status;
  uint32_t node, count, frees, padding;
  struct mesh_reach reach[MESH_STREAMS];
  uint64_t posted[MESH_STREAMS][2], free_wire[MESH_STREAMS];
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
   report names, that report kept in `reports`; a socket not yet placed waits at peers[0]; one whose report fails (its
   peer closed it) or names no rank still to meet closed alone, the root accepting the next, though one that sends
   nothing holds the root's read until the phase ends) or a non-root's to the root
   (peers[0], placed once connected), each placed before the revoke is read, so a revoke either finds it or is read.  Its
   phase ends MESH_PLATFORM_NS after the init began where no rank was met, and MESH_PLATFORM_NS after the first rank met
   (the root's first accept, a non-root's connect) for the rest of the phase, the first gather's (`until`): 0,
   ECANCELED (comm revoked), ETIMEDOUT, or the errno */
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
    /* a connection whose report does not come, or names no rank still to meet, is closed alone */
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

/* 0, or EINVAL where two of the `n` gathered reports are of one node: a communicator keeps its holdings and the pages
   posted its ranks exchange a node (posted[k] by the node's index in a report's reach, comm_hold_toward by node), so
   two ranks of one node would share one slot, laying both ranks' bytes on one stream; and no stream joins two ranks of one node (a bridge reaches other nodes) */
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

/* a communicator's table, the caller's operands (nccl.h ncclMeshCommInitRank): its paths (NULL: none, the map's
   shortest-path trees) and its programs copied (`given`; none: the structural table, comm_routes); 0, or ENOMEM */
static int comm_table(struct ncclComm *made, const struct mesh_trees *paths, const ncclMeshProgram *programs, int count) {
  if ((paths && trees_copy(&made->paths, paths)) || (count && !(made->given = calloc((size_t)count, sizeof *made->given)))) return ENOMEM;
  for (int e = 0; e < count; e++) {
    if (program_copy(&made->given[e].program, programs[e].program)) return ENOMEM;
    made->given[made->listed++].below = programs[e].below;
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
   one stream), taken from the bridge as rows are (mesh-call.h mesh_stream_take), comm its one user (`held`, comm_let_go
   lets it go), its pages posted each way as taken into this rank's report (`posted`, by the index of the node in its
   reach), which the second gather hands its peer.  One communicator a stream: a holding's lists and counts are one order each, its
   holder's alone; a link has several streams, so communicators that share a node pair each hold their own.  0, or
   EBUSY (no stream toward a member's node free at both ends), ENOSPC (no hold vacant), or mesh_stream_take's refusal;
   what it took is let go with comm (comm_leave) */
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

/* the Metal buffer over the region's page (or pages) that holds a channel's counts, and each count's offset in it */
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

/* design/transport.md section 14: the communicator's channels on the holdings its init took (comm_hold), from every
   rank's report as the second gather gave them: toward each rank it is joined to, a channel on the holding toward the
   rank's node, the two ends' difference in pages posted kept as a statistic (`differ`, read by nothing: a holding ends
   with nothing outstanding or its stream not live, so a live stream's ends are equal); its map (map_of), its page (its
   bridge's region's), its failure words and its record of its groups' entries
   (`timed`).  0, or ENOMEM; EPROTO where no hold is of a joined node */
static int comm_join(struct ncclComm *comm, const struct report *reports) {
  struct mesh_ctx *context = &comm->attach->context;
  const struct hdr *m = context->M;
  const uint32_t n = (uint32_t)comm->nranks, me = (uint32_t)comm->rank, node = reports[me].node;
  uint64_t bandwidth;
  comm->page = (uint64_t)m->block * m->pgsz;
  int status = map_of(comm, reports);
  if (!status && !(comm->violation = metal_shared((size_t)getpagesize()))) status = ENOMEM;
  if (!status) comm->words = metal_contents(comm->violation);
  comm->nstamps = (uint32_t)((size_t)getpagesize() / sizeof(uint64_t) - METAL_STAMP);
  if (!status && (!(comm->ends = calloc(comm->nstamps, sizeof *comm->ends)) || !(comm->pending = calloc(comm->nstamps, sizeof *comm->pending)) ||
                  !(comm->idle = calloc(comm->nstamps, sizeof *comm->idle))))
    status = ENOMEM;
  for (uint32_t w = 0; !status && w < comm->nstamps; w++) comm->idle[comm->nidle++] = comm->nstamps - 1 - w;
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

/* design/transport.md section 14, a communicator's init on NCCL's bootstrap at its id, the one path of every init
   (ncclMeshCommInitRank, blocking or not): its bootstrap's sockets opened (bootstrap_open), then made attached
   (comm_attach) and the nodes its bridge reaches read (reach_read), then every rank's report gathered (two ranks on one
   node fail every rank alike: comm_apart, a bound); each rank holding the stream toward each member's node it is
   joined to (comm_hold); a second gather carrying each rank's holding's status and its holdings' pages posted
   (NCCL's own init exchanges each peer's connection through its bootstrap: ncclTransportP2pSetup), whose difference
   each rank keeps as a statistic as it makes its channels (comm_join).  A rank that cannot go on, a stream held by another (EBUSY) or gone (ENOLINK), a peer silent past
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
   revoke at their next look (bootstrap_open), fails as a value; and its failure words' stop (word [6]) set, which a
   GPU wait of its that reaches its bound records as its cause (METAL_REVOKED); no kernel stops at it, so its GPU work
   already encoded runs as encoded, each wait ending at its landing or its bound; under its guard */
static void comm_stop(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  if (comm->words) atomic_store_explicit(comm->words + 6, 1, memory_order_release);
  pthread_mutex_unlock(&comm->guard);
}

/* ncclCommRevoke, NCCL's: comm's operations in flight revoked and its resources kept (comm_stop alone: its streams stay
   held, and its bridge sends what its groups published and nothing else), a host-path group of its returning at its
   next look and each of its GPU waits ending at its landing or its bound, every later
   group refused (ncclInvalidUsage), ncclCommGetAsyncError reading ncclSuccess once the revoke is done; from any thread,
   under comm's own lock alone, never one a group holds while it waits.  ncclCommDestroy then gives its streams back
   (comm_free) and frees it */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  (void)revokeFlags;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  comm_stop(comm);
  return ncclSuccess;
}

/* ncclCommDestroy: comm stopped (comm_stop), its init's thread joined (a non-blocking init's, which the stop ends), its
   host-path groups returned (each at its next look) and the GPU work the library committed for it awaited (gpu_wait,
   group_lock released for it: its command buffer's Completed or Error status, which the platform's watchdog gives
   one that overruns, and no clock; the stop that is the destroy's own does not end it, and a buffer not committed
   is not awaited: nothing would run it while its committer waits), its groups in flight lost; then its streams
   given back (its bridge posts nothing more of their holdings), its pool's leases returned, and comm freed.  Work encoded in a caller's encoder is the
   caller's to finish first (NCCL's: no communicator is destroyed under its own operations in flight): a kernel of it
   still queued would write the stream given back (rdma/NCCL.md "Failures") */
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
    a->leases[a->count++] = (struct lease){*ptr, lease.first, (size_t)lease.pages * a->context.M->pgsz, NULL};
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




/* an operator made into the communicator's table (a freed slot, else one more), its handle past ncclNumOps */
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

/* nccl.h ncclMeshRedOpCreatePartials: an operator of the communicator's, as PreMulSum's, its row 2 + dim floats */
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

/* -- a communicator's channels and its failures (design/transport.md section 14).  Toward each member it is joined to,
   a channel on the holding its init took of the stream toward the member's node.  Each direction is its own list: this
   rank's sends to the member are its holding's send list and the member's landing list from it, entry for entry in the
   order both ends lay alike (each collective's steps by round in call order, then the point-to-point calls in call
   order: mesh-plan.h mesh_group_lay), a step one entry whatever its bytes.  A send entry names the pages its bytes lie
   in, a landing entry the pages they land in (the destination itself for a COPY, the group's landing area for a
   combine), each a page's first byte; this end publishes its sends by raising its holding's `ready` past them, in
   order, and consumes its landings once `landed` covers them.  Nothing credits a publication: the hardware's credit
   holds a SEND until its receive is posted (design/transport.md).  A write over bytes an earlier SEND reads waits
   for `sent` (the same channel's combine, meeting; the group's end, retire).  A landing entry names no word: the
   bridge posts its receives as the entry is presented (the operator, 2026-10-22: "data procesisng code gets to write
   their own guards in their own source"), so a program that names pages again orders its own reads and writes of them,
   and the library orders its own pool's (spare_take: a lease comes back once its readers' stamp has passed).  A group
   presents its entries before it publishes any (schedule); a member that fails mid-group sends nothing more, and its
   holding's end with units outstanding takes its stream out of the pairing (design/transport.md section 14). ---------- */

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
  }
  return comm_told(comm);
}

/* the communicator's failure: the one it kept; a revoke; one its GPU work recorded in its failure words (a wait's
   bound reached, its link lost: nccl-mesh-metal.h METAL_*); else a stream it holds
   no longer its own (a bridge's restart, or a pairing lost and made again, opens every stream new: mesh-call.h
   mesh_stream_held), kept now (comm_fail), so ncclCommGetAsyncError reads it before any group runs on it; ncclSuccess
   where none.  Under group_lock */
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
    return comm_fail(comm, ncclRemoteError, "entry %llu on the channel to node %u was not done within %llu polls of a GPU wait (its "
                     "stream's bound, ncclMeshStream polls): its bytes had not landed or been sent, the peer's work there beginning that "
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

/* -- a group ------------------------------------------------------------------------------------------- */

/* where a step's bytes are: on the Metal path a buffer and a byte offset into it; on the host path an address (buffer
   NULL, offset the address); {NULL, 0} none */
struct where { void *buffer; size_t offset; };
static int same(struct where a, struct where b) { return a.buffer == b.buffer && a.offset == b.offset; }
static int named(struct where w) { return w.buffer || w.offset; }
static struct where shifted(struct where w, size_t by) { w.offset += by; return w; }
/* a where's host address: NULL where it names none, or a buffer the host cannot read (a private one, never a lease's) */
static unsigned char *address_of(struct where w) {
  if (!w.buffer) return (unsigned char *)w.offset;
  unsigned char *base = metal_contents(w.buffer);
  return base ? base + w.offset : NULL;
}

struct steps {
  struct mesh_step *steps;
  uint32_t count;
  unsigned char *done;
  /* the buffers the steps name: 0 the operand, made in place; b >= 1 the rank's scratch b (`buffers` of them) */
  uint32_t buffers;
  struct where *buffer;
  /* a call reading its send buffer in place (plan_direct): each step's source where it reads the send buffer (`read`,
     none: its own buffer), whether a receive's combine is its range's first (`fresh`: buffer = send op piece), and
     which buffers any step writes (`needed`, a buffer each; one no step writes is never made) */
  struct where *read;
  unsigned char *fresh, *needed;
  const struct mesh_program *program;
  struct mesh_operand operand;
  /* the operand's segments in operand elements, owned (`cut`): a reduce-scatter's and an all-gather's NCCL's (a rank's
     count each, or an uneven call's counts), NULL the program's own cut */
  const uint64_t *segments;
  uint64_t *cut;
  /* each step's channel and entry as schedule_group places them, and the receive before it (by round, then plan
     order) that has any bytes, or -1 */
  uint32_t *channel;
  uint64_t *entry;
  int32_t *before;
};

/* a call: its collective (or point-to-point direction, or WHAT_COUNTED: an all-to-all whose counts the host holds,
   nccl.h ncclMeshAlltoAllV), root, peer, buffers, count, type, operator and communicator; `segments`, an uneven
   reduce-scatter's or all-gather's count a rank (ncclMeshReduceScatterV, ncclMeshAllGatherV: the call's own copy,
   `total` their sum; NULL: `count` each), an all-to-all's elements to each rank then from each (`count` those
   crossing); a reduction's operator as the host runs it (reduction: `combine`, `post`, `pre`; the partials merge's row
   of `unit` elements); its buffers where they are (`send_at`, `recv_at`: struct where), its result's (`sum`, the
   operand made in place) and whether it reads its send buffer in place (`direct`: plan_direct); on the host path a
   buffer of its own where nothing crosses (`result`) and a send to itself's copy (`copy`) */
struct call {
  int what, root, peer;
  const void *send;
  void *recv;
  size_t count, total;
  uint64_t *segments;
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
  /* a premultiplier in device memory (ncclScalarDevice), read into `pre` when its group runs on the host */
  const void *scalar_at;
  int premultiplied, direct;
  struct steps *bound;
  unsigned char *result, *copy;
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
/* the entries before rank q's segment of a count a rank (an all-to-all's sendcounts or recvcounts) */
static uint64_t seg_first(const uint64_t *seg, int q) {
  uint64_t at = 0;
  for (int r = 0; r < q; r++) at += seg[r];
  return at;
}
/* the name of a call's kind, its failures' */
static const char *call_name(const struct call *c) {
  static const char *const names[] = {"ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclReduceScatter", "ncclAllGather"};
  return c->what == WHAT_SEND ? "ncclSend" : c->what == WHAT_RECV ? "ncclRecv" : counted(c) ? "ncclMeshAlltoAllV" : names[c->what];
}

/* whether a call crosses a link: elements, more than one rank, and a point-to-point call's peer another rank */
static int crosses(const struct call *c) { return elements(c) && c->comm->nranks > 1 && !(p2p(c) && c->peer == c->comm->rank); }

/* whether `bytes` at w lie in this node's region (comm's attach's), a lease's pages */
static int in_region(const struct ncclComm *comm, struct where w, size_t bytes) {
  const struct hdr *m = comm->attach->context.M;
  const unsigned char *at = address_of(w), *first = (const unsigned char *)m + m->data_off, *end = (const unsigned char *)m + m->length;
  return at && at >= first && at <= end && bytes <= (size_t)(end - at);
}
/* the bytes a crossing call reads from its send buffer: all of them, a broadcast's at its root, an all-gather's own
   segment, an all-to-all's segments to every rank; none a receive's */
static size_t sent_bytes(const struct call *c) {
  const size_t z = SIZE[c->type];
  size_t length = 0;
  if (c->what == WHAT_RECV || (c->what == MESH_BROADCAST && c->root != c->comm->rank)) return 0;
  if (c->what == MESH_ALLGATHER) { segment_at(c, c->comm->rank, &length); return length * z; }
  if (counted(c)) return seg_first(c->segments, c->comm->nranks) * z;
  if (c->what == WHAT_SEND || c->what == MESH_BROADCAST) return c->count * z;
  return elements(c) * z;
}
/* every crossing call's send buffer in this node's region: an operand that crosses a link is a lease, read in place or
   staged in the region, and nothing is copied into the region from outside it (nccl.h, Memory); else the call fails as
   a value, naming it */
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
  free(s->buffer); free(s->steps); free(s->done); free(s->channel); free(s->entry); free(s->before); free(s->read); free(s->fresh);
  free(s->needed); free(s->cut);
  free(s);
}

static size_t piece_bytes(const struct mesh_step *step) { return step->piece.elements * step->piece.element_bytes; }

/* every call that crosses a link planned: a collective along the communicator's table (comm_compile), its steps cut on
   the communicator's page (mesh-plan.h mesh_program_steps_cut: the program's own cuts on pages; a reduce-scatter's and
   an all-gather's segments NCCL's, a rank's count each or its counts, so its program's segments must be the ranks), its
   operand's element a merge's row where it merges; a point-to-point call one SEND or COPY step; an all-to-all to and
   from each other rank (mesh_counted_steps); a step that moves bytes toward or from a rank no link of this node joins
   refused (the bridges forward nothing) */
static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
    const ncclResult_t compiled = p2p(c) || counted(c) ? ncclSuccess : comm_compile(comm);
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
    const size_t most = (program ? program->first[program->nodes] : counted(c) ? 2 * (size_t)comm->nranks : 1) + 1;
    if (s) s->buffers = counted(c) ? 2 : 1 + (program ? program->scratch : 0);
    if (!s || !(s->steps = calloc(most, sizeof *s->steps)) || !(s->done = calloc(most, 1)) || !(s->channel = calloc(most, sizeof *s->channel)) ||
        !(s->entry = calloc(most, sizeof *s->entry)) || !(s->before = calloc(most, sizeof *s->before)) ||
        !(s->buffer = calloc(s->buffers, sizeof *s->buffer)) || (nccl && !(s->cut = malloc((size_t)comm->nranks * sizeof *s->cut)))) {
      steps_free(s);
      return fail(ncclSystemError, "out of memory");
    }
    c->bound = s;
    const uint32_t u = unit_of(c);
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)(SIZE[c->type] * u), elements(c) / u};
    s->operand = operand;
    s->program = program;
    for (int r = 0; nccl && r < comm->nranks; r++) s->cut[r] = (c->segments ? c->segments[r] : c->count) / u;
    s->segments = s->cut;
    if (p2p(c)) {
      s->steps[0] = (struct mesh_step){c->what == WHAT_SEND ? MESH_STEP_SEND : MESH_STEP_COPY, (uint32_t)c->peer, 0, 0, 0, operand};
      s->count = 1;
    } else if (counted(c))
      s->count = mesh_counted_steps((uint32_t)comm->nranks, (uint32_t)comm->rank, c->segments, 1, (struct mesh_operand){operand.type, operand.element_bytes, 0},
                                    (struct mesh_operand){0, 0, 0}, s->steps);
    else s->count = mesh_program_steps_cut(program, (uint32_t)comm->rank, operand, s->segments, comm->page, s->steps);
    for (uint32_t k = 0; k < s->count; k++)
      if (piece_bytes(s->steps + k) && comm->toward[s->steps[k].peer] < 0)
        return fail(ncclInvalidUsage, "a call of %s moves bytes between rank %d and rank %u (node %u), which no link of this node joins: the "
                    "bridges forward nothing, so a route through another node is refused", comm->name, comm->rank, s->steps[k].peer,
                    comm->members[s->steps[k].peer]);
  }
  return ncclSuccess;
}

/* a step's entry on its channel: its call and step, its entry in its list, its pages (pages_of its bytes) and the
   value its list's count reaches once it is whole (`end`), its piece's byte offset in its buffer and its bytes, where
   its bytes lie (a SEND's source, a COPY's destination, a combine's landing area: `data`) and the first of the rows
   they lie in (`row`); a landing's `need` (the send entries a combine into bytes an earlier SEND of its call on its
   channel reads waits to be published: that entry + 1, 0 none) and `sent` (that SEND's end, the `sent` the combine
   awaits) */
struct message { struct call *c; uint32_t step; uint64_t entry; mesh_pages pages, end; uint64_t need; mesh_pages sent; size_t offset, bytes;
                 struct where data; uint64_t row; };
/* a group's entries on one channel, each list in its order from its first (`first`: [0] the send list's, [1] the
   landing list's) */
struct schedule { struct message *out, *in; size_t outs, ins; uint64_t first[2]; };

/* a group's calls as mesh-plan.h's group, which mesh_group_lay lays (the one layout both ends make): call i's kind and
   its steps at rank r, steps[i * N + r] and count[i * N + r] (none where the rank's are not known) */
static uint32_t group_kind(const struct call *c) { return p2p(c) ? MESH_GROUP_POINT : counted(c) ? MESH_GROUP_HOSTED : MESH_GROUP_COLLECTIVE; }
static void group_of(const struct call *list, size_t n, uint32_t N, const struct mesh_step **steps, uint32_t *count,
                     struct mesh_group_call *calls, uint64_t page, struct mesh_group *group) {
  for (size_t i = 0; i < n; i++) calls[i] = (struct mesh_group_call){group_kind(list + i), 0, count + i * N, steps + i * N};
  *group = (struct mesh_group){N, (uint32_t)n, calls, page};
}

/* the next number of comm's groups taken by a group it issues (`groups`), as the group takes group_lock, and its slot of
   the record cleared: a group that crosses no link records no entry (ncclMeshTimes gives it 0); commit_group records
   a crossing group's in it.  Under group_lock */
static void group_numbered(struct ncclComm *comm) {
  if (comm->timed) memset(comm->timed + (comm->groups % GROUP_TIMES) * comm->count, 0, comm->count * sizeof *comm->timed);
  comm->groups++;
}

/* whether two steps' pieces meet in one buffer */
static int meet(const struct mesh_step *x, const struct mesh_step *y) {
  return x->buffer == y->buffer && x->first < y->first + y->piece.elements && y->first < x->first + x->piece.elements;
}

/* each channel's entries for the group, each list from the last scheduled one's end (mesh_group_lay), each step's
   channel and entry kept, the receive before it (mesh-plan.h mesh_group_before) kept, a receive with no bytes done;
   each landing that combines into bytes an earlier SEND of its call on its channel reads (mesh_compile's two-node
   direct exchange) with that SEND's entry and end (struct message `need`, `sent`) */
static ncclResult_t schedule_group(struct ncclComm *comm, struct call *list, size_t n, struct schedule *plans) {
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
    *p = (struct schedule){calloc(total, sizeof *p->out), calloc(total, sizeof *p->in), 0, 0, {ch->sends, ch->lands}};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    for (int send = 1; send >= 0; send--) {
      uint32_t placed = 0;
      mesh_group_lay(&group, me, (uint32_t)ch->peer, send, send ? ch->sends : ch->lands, scratch, made, &placed);
      mesh_pages end = send ? ch->out : ch->in;
      for (uint32_t k = 0; k < placed; k++) {
        const struct mesh_place *x = &made[k].place;
        struct call *c = list + made[k].call;
        const struct mesh_step *step = c->bound->steps + x->step;
        const mesh_pages pages = pages_of(piece_bytes(step), comm->page);
        end = pages_add(end, pages);
        const struct message m = {.c = c, .step = x->step, .entry = x->entry, .pages = pages, .end = end,
                                  .offset = step->first * step->piece.element_bytes, .bytes = piece_bytes(step)};
        if (send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
        c->bound->channel[x->step] = h; c->bound->entry[x->step] = x->entry;
      }
    }
    for (size_t k = 0; k < p->ins; k++) {
      struct message *m = p->in + k;
      const struct steps *st = m->c->bound;
      if (st->steps[m->step].op != MESH_STEP_REDUCE) continue;
      for (size_t j = 0; j < p->outs; j++) {
        const struct message *o = p->out + j;
        if (o->c != m->c || o->step >= m->step || (st->read && named(st->read[o->step])) || !meet(st->steps + o->step, st->steps + m->step)) continue;
        if (o->entry + 1 > m->need) m->need = o->entry + 1;
        if (pages_below(m->sent, o->end)) m->sent = o->end;
      }
    }
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct steps *st = list[i].bound;
    if (st && mesh_group_before(group_kind(list + i), st->steps, st->count, st->before)) result = fail(ncclSystemError, "out of memory");
    for (uint32_t k = 0; st && !result && k < st->count; k++)
      if (st->steps[k].op != MESH_STEP_SEND && !piece_bytes(st->steps + k)) st->done[k] = 1;
  }
  free(steps); free(counts); free(group_calls); free(made); free(scratch);
  return result;
}

static void schedules_free(const struct ncclComm *comm, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < comm->count; h++) { free(plans[h].out); free(plans[h].in); }
  free(plans);
}

/* the program ends of comm that the GPU has reached seen (each stamp word reading its end's number): every lease the
   end named has one use less, and the word is idle again.  Under group_lock */
static void ends_seen(struct ncclComm *comm) {
  for (uint32_t k = 0; k < comm->npending;) {
    const uint32_t w = comm->pending[k];
    struct end *e = comm->ends + w;
    if (atomic_load_explicit(comm->words + METAL_STAMP + w, memory_order_acquire) != e->number) { k++; continue; }
    for (uint32_t j = 0; j < e->count; j++) comm->spares[e->leases[j]].uses--;
    free(e->leases);
    *e = (struct end){0, 0, NULL};
    comm->idle[comm->nidle++] = w;
    comm->pending[k] = comm->pending[--comm->npending];
  }
}

/* a lease of comm's pool of at least `bytes` (its smallest free one that holds them, else a new one: ncclMemAlloc),
   held by the group numbered `holder` until that group ends (spares_give); NULL where the region cannot meet it, the
   failure formatted (ncclMemAlloc's).  A lease is free once no group holds it, it was not given back in the program
   being encoded, and the end of every program that named it has been seen (its use count 0: ends_seen): the bridge
   posts a landing's receives as it is presented, so a lease a command the GPU has not ended may read or write is never
   landed in again, in whatever order the caller commits its programs and on whatever queues.  Nothing waits: the pool
   grows by the leases named by programs the GPU has not ended.  Under group_lock */
static void *spare_take(struct ncclComm *comm, size_t bytes, uint64_t holder) {
  struct spare *best = NULL;
  ends_seen(comm);
  for (uint32_t k = 0; k < comm->nspares; k++) {
    struct spare *x = comm->spares + k;
    if (!x->holder && !x->uses && !x->given && x->bytes >= bytes && (!best || x->bytes < best->bytes)) best = x;
  }
  if (!best) {
    struct spare *grown = realloc(comm->spares, (comm->nspares + 1) * sizeof *grown);
    void *address = NULL;
    if (!grown) { fail(ncclSystemError, "out of memory"); return NULL; }
    comm->spares = grown;
    if (ncclMemAlloc(&address, bytes)) return NULL;
    best = comm->spares + comm->nspares++;
    *best = (struct spare){address, bytes, 0, 0, 0};
  }
  best->holder = holder;
  return best->address;
}
/* every lease of comm's pool the group numbered `holder` held given back: free at once where `stamped` is 0 (the host
   path, whose work has ended), else named by the end of the program being encoded (the Metal path: stamp_end).  Under
   group_lock */
static void spares_give(struct ncclComm *comm, uint64_t holder, int stamped) {
  for (uint32_t k = 0; holder && k < comm->nspares; k++)
    if (comm->spares[k].holder == holder) { comm->spares[k].holder = 0; comm->spares[k].given |= stamped; }
}
/* the end of the program being encoded, for comm: every lease of its pool that a group holds or that was given back
   while the program ran (the leases the program may name: its operands, landing areas and scratch, the SENDs and
   combines of groups it completes, the results it copies out as it retires them) has one use more, freed once the GPU
   has ended the program (ends_seen).  An idle stamp word takes the program's end number, which metal_stamp stores
   there once every dispatch the program encoded has ended.  0; ncclSystemError where every word is in use (`nstamps`
   ends named leases and are not yet seen: the GPU ends programs before more are encoded) or the stamp is not encoded,
   the named leases then kept until ncclCommDestroy.  Under group_lock */
static ncclResult_t stamp_end(struct metal_program *program, struct ncclComm *comm) {
  uint32_t named = 0;
  for (uint32_t k = 0; k < comm->nspares; k++) named += comm->spares[k].holder || comm->spares[k].given;
  if (!named) return ncclSuccess;
  ends_seen(comm);
  uint32_t *leases = comm->nidle ? malloc(named * sizeof *leases) : NULL;
  const uint32_t w = comm->nidle ? comm->idle[comm->nidle - 1] : 0;
  const uint64_t number = comm->ended + 1;
  const struct metal_result stamped = leases ? metal_stamp(program, comm->violation, (METAL_STAMP + w) * sizeof(uint64_t), number) : METAL_DONE;
  uint32_t n = 0;
  for (uint32_t k = 0; k < comm->nspares; k++)
    if (comm->spares[k].holder || comm->spares[k].given) {
      comm->spares[k].uses++;
      comm->spares[k].given = 0;
      if (leases) leases[n++] = k;
    }
  if (!comm->nidle)
    return fail(ncclSystemError, "communicator %s: %u program ends that named leases of its pool are not yet seen ended (its stamp words, "
                "nccl-mesh-metal.h METAL_STAMP): the GPU ends programs before more are encoded; the leases this one named are kept "
                "until ncclCommDestroy", comm->name, comm->nstamps);
  if (!leases) return fail(ncclSystemError, "out of memory");
  if (stamped.status) {
    free(leases);
    return fail(ncclSystemError, "communicator %s: a program's end stamp: %s; the leases of its pool it named are kept until ncclCommDestroy",
                comm->name, stamped.message);
  }
  comm->ended = number;
  comm->nidle--;
  comm->ends[w] = (struct end){number, n, leases};
  comm->pending[comm->npending++] = w;
  return ncclSuccess;
}
/* a lease of comm's pool (spare_take) as a where: on the Metal path its Metal buffer (lease_buffer, made resident
   through the program's residency set where it has one), on the host path its address; {NULL, 0} where none was had,
   the failure formatted */
static struct where spare_where(struct metal_program *program, struct ncclComm *comm, size_t bytes, uint64_t holder) {
  void *address = spare_take(comm, bytes ? bytes : 1, holder), *buffer = NULL;
  size_t offset = 0;
  if (!address) return (struct where){NULL, 0};
  if (!program) return (struct where){NULL, (size_t)address};
  if (lease_buffer(address, &buffer, &offset)) { fail(ncclSystemError, "no Metal buffer over a lease of the communicator's"); return (struct where){NULL, 0}; }
  metal_resident(program, buffer);
  return (struct where){buffer, offset};
}

static size_t page_up(size_t bytes, uint64_t page) { return (size_t)((bytes + page - 1) / page * page); }

/* where each entry's pages lie (struct message `data`, `row`): a SEND's in the buffer its step reads (the send buffer,
   where it reads that in place), a COPY's in its step's buffer, a combine's in the group's landing area (a lease of
   comm's pool, each entry from a page of it); each a range of a lease's rows (design/pages-and-functions.md: the page
   table; the bridge reads each row's page in the range's order).  0, else the
   call fails as a value, naming the piece (ncclInvalidUsage: the piece's buffer is no lease, or its cut, NCCL's, does
   not lie on a page; nothing is copied to move it), nothing presented */
static ncclResult_t place_group(struct metal_program *program, struct ncclComm *comm, struct schedule *plans, uint64_t holder) {
  const struct hdr *m = comm->attach->context.M;
  const unsigned char *region = (const unsigned char *)m;
  const uint64_t page = comm->page;
  size_t landing = 0, used = 0;
  for (uint32_t h = 0; h < comm->count; h++)
    for (size_t k = 0; k < plans[h].ins; k++)
      if (plans[h].in[k].c->bound->steps[plans[h].in[k].step].op == MESH_STEP_REDUCE) landing += page_up(plans[h].in[k].bytes, page);
  struct where area = {NULL, 0};
  if (landing && !named(area = spare_where(program, comm, landing, holder))) return fail(ncclSystemError, "the group's landing area: %s", last.text);
  for (uint32_t h = 0; h < comm->count; h++)
    for (int send = 1; send >= 0; send--)
      for (size_t k = 0; k < (send ? plans[h].outs : plans[h].ins); k++) {
        struct message *x = (send ? plans[h].out : plans[h].in) + k;
        const struct steps *st = x->c->bound;
        const struct mesh_step *step = st->steps + x->step;
        if (send) x->data = shifted(st->read && named(st->read[x->step]) ? st->read[x->step] : st->buffer[step->buffer], x->offset);
        else if (step->op == MESH_STEP_REDUCE) { x->data = shifted(area, used); used += page_up(x->bytes, page); }
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

/* the communicators a program touched, each once, until its end (program_end); a program holds group_lock from its
   begin to its end but across a wait (room_wait), which sets its list aside, so the list is empty at every taking of
   the lock */
static struct ncclComm *touching;
static void touch(struct ncclComm *comm) {
  if (comm->touched) return;
  comm->touched = 1;
  comm->next_touched = touching;
  touching = comm;
}
/* whether channel ch's list (0 send, 1 landing) has room for `need` entries more: its entries presented and not taken
   by the bridge, with them, at most MESH_STREAM_ENTRIES */
static int room_for(const struct channel *ch, int list, uint64_t need) {
  return atomic_load_explicit(ch->view.presented[list], memory_order_relaxed) - atomic_load_explicit(ch->view.taken[list], memory_order_acquire) +
         need <= MESH_STREAM_ENTRIES;
}
/* the wait for room on channel ch's list with group_lock released, as a condition variable's wait releases its mutex
   (design/rubrics.md M3, M4: other communicators' groups and revokes go on meanwhile): comm kept from its destroy
   (`runs`), and the list of communicators the program being encoded touched set aside (it is comm alone where a group
   presents, launch_metal), so a program another thread encodes meanwhile ends with its own; each look spins, ending at
   the bridge's `taken` advancing, at a revoke or at the holding's end, never at a clock; then the lock taken again, the
   list restored and comm's failure read again (comm_failure: a group in flight of it may have failed meanwhile).
   Under group_lock */
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
/* each channel's entries presented (mesh-call.h mesh_stream_present), its send list's and then its landing list's, once
   the bridge has room for all of them (each list's entries presented and not taken fewer than MESH_STREAM_ENTRIES with
   them: the bridge takes an entry as it reads it while fewer than that of the list are taken and not done), so a group
   is presented whole or not at all; a send entry's bytes ready once `ready` passes them (word 0, its producer's
   declaration that they are written), a landing entry with no word, its receives posted as presented.  The wait for
   room (room_wait) holds no lock and no clock ends it (design/rubrics.md M3, 10-18: every wait killable).  0;
   ncclSystemError where a list's entries are more than the bridge ever takes, the wait's failure (a revoke, the
   holding's end, comm's failure: nothing presented, the call's failure alone); a refused presentation fails comm (its
   lists stand mid-group).  Under group_lock */
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

/* the group's entries counted on each channel and recorded under the number it took (`timed`: its first send entry and
   its last landing entry there, each + 1; group_numbered, ncclMeshTimes) */
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

/* the group's schedule (schedule_group), its bytes placed and checked (place_group), presented (present_group) and
   counted (commit_group); a failure before any entry is presented is the call's alone, one after it the communicator's
   (its lists stand mid-group); *made NULL where it failed */
static ncclResult_t schedule(struct metal_program *program, struct ncclComm *comm, struct call *list, size_t n, uint64_t holder,
                             struct schedule **made) {
  struct schedule *plans = calloc(comm->count + 1, sizeof *plans);
  *made = NULL;
  if (!plans) return fail(ncclSystemError, "out of memory");
  ncclResult_t result = schedule_group(comm, list, n, plans);
  if (!result) result = place_group(program, comm, plans, holder);
  if (!result) result = present_group(comm, plans);
  if (result) { schedules_free(comm, plans); return result; }
  commit_group(comm, plans);
  *made = plans;
  return ncclSuccess;
}

/* a step may run once the receive before it is consumed (mesh_group_before: every earlier receive then is, by induction, as
   each receive's first event waits on the one before it; an all-to-all's steps wait on none) */
static int ready(const struct steps *s, uint32_t k) { return s->before[k] < 0 || s->done[s->before[k]]; }

/* a holding's count as the host reads it (acquire): ~0 once its link is lost (pages_lost) */
static mesh_pages count_of(_Atomic uint64_t *word) { return pages_from_word(atomic_load_explicit(word, memory_order_acquire)); }

/* every channel's entries published and consumed on the host: a SEND's published by `ready` (a release store past it,
   its step's receive before it consumed), a combine's landing combined into its buffer (its first one, a call reading
   its send buffer in place, writing send op landed), a COPY's consumed once landed (it landed in place); a combine
   into bytes an earlier SEND of its call on its channel reads waits for `sent` past that SEND (struct message `sent`);
   then every send entry's bytes sent, so the caller may write its send buffer again.  Every wait ends at an event: its
   count, a count's ~0 (the link lost, the bridge stopped), the end of its holding with work left (its stream no
   longer the communicator's own, mesh_stream_held: a pairing lost and made again, a bridge restarted), or a revoke,
   each read at every look.  No clock ends it (the operator, 2026-10-22: "we don't introduce race conditions or use
   timing anywhere ... because using timing is always slower than not using timing"): a peer that has not yet
   presented is waited for, and the caller decides whether to revoke and run again
   (design/pages-and-functions.md).  Every look spins: nothing parks */
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
          if (st->fresh && st->fresh[m->step]) memmove(to, address_of(st->read[m->step]) + m->offset, m->bytes);
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

/* the call's contribution into its operand, made in place at `own` (a call that reads no send buffer in place) */
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

/* the result into recvbuff on the host: a call reading its send buffer in place, its ranges no step wrote first */
static void finish(const struct call *c) {
  const size_t z = SIZE[c->type], n = c->count;
  const int nranks = c->comm->nranks;
  unsigned char *sum = address_of(c->sum), *recv = c->recv;
  for (uint32_t k = 0; k < c->complements; k++) {
    const uint64_t *x = c->complement + 3 * k;
    memmove(address_of((struct where){c->sum.buffer, (size_t)x[0]}), address_of((struct where){c->send_at.buffer, (size_t)x[1]}), x[2]);
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
  /* NCCL's hostToDevRedOp refuses it alike ("Data type supplied to user-created ncclRedOp_t does not match"): a
     scalar of one type read as another's is a value no one gave (design/rubrics.md D2, the operator 10-19: "functions
     should throw ... before they try to guess parametric values"); and the merge's rows are float32 [m, l, O]
     (merge_rows, nccl-mesh-kernels.h merge_row), which another type's elements would be read and written past */
  if (made->type != c->type) return fail(ncclInvalidArgument, "an operator made on the communicator reduces the datatype it was made for");
  if (made->unit) { c->combine = COMBINE_MERGE; c->unit = made->unit; return ncclSuccess; }
  /* a scalar in device memory is read where the reduction runs: the host path's call; the Metal path's kernels take it
     as an argument written when the group is encoded, before the work that may write it runs */
  if (made->at && c->stream)
    return fail(ncclInvalidArgument, "a PreMulSum scalar in device memory (ncclScalarDevice) on the Metal path: its kernels take the scalar as an "
                "argument written when the group is encoded, so none reads it where the GPU runs");
  if (made->at) c->scalar_at = made->at;
  else memcpy(c->pre, made->value, SIZE[c->type]);
  c->premultiplied = 1;
  return ncclSuccess;
}

/* an all-to-all's own segment: its byte offsets in the send and the receive buffer, and its bytes, the lesser of
   what it sends itself and what it receives from itself (each in its own segment's bytes) */
static size_t own_segment(const struct call *c, size_t *sent, size_t *landed) {
  const int n = c->comm->nranks, me = c->comm->rank;
  *sent = seg_first(c->segments, me) * SIZE[c->type]; *landed = seg_first(c->segments + n, me) * SIZE[c->type];
  const uint64_t out = c->segments[me], in = c->segments[n + me];
  return (out < in ? out : in) * SIZE[c->type];
}

static int plan_direct(struct call *c, struct where own);
/* what a wait for a communicator's GPU work (gpu_wait) ended at: its last program's end (GPU_ENDED), its command buffer's
   error status (GPU_ERROR), that buffer not committed (GPU_UNCOMMITTED: nothing runs it while its committer waits), a
   revoke (GPU_REVOKED) or a holding's end (GPU_HOLDING) */
enum { GPU_ENDED, GPU_ERROR, GPU_UNCOMMITTED, GPU_REVOKED, GPU_HOLDING };
/* design/transport.md section 5: the wait for comm's last GPU program (its count `last` on `finished`, signalled at the
   end of its command buffer, `command`) with group_lock released for it, as a condition variable's wait releases its
   mutex (design/rubrics.md M3, M4: other communicators' groups and revokes go on meanwhile): the count and the buffer
   read under the lock; the wait in MESH_PAUSE_NS slices, ending at the signal or at the buffer's end as its status tells
   it (Completed, or Error, which the platform's watchdog gives a buffer that overruns: nccl-mesh-metal.m metal_ended),
   never at a clock, and for a group (`group`: comm kept from its destroy meanwhile by `runs`) at a revoke or a holding's
   end too; then the lock taken again, and the wait made again where a later program touched comm meanwhile.  Under
   group_lock */
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
/* a host-path group's wait for its communicator's GPU work (gpu_wait), its end as the call's failure alone: the
   command buffer's error, one not committed, a revoke, a holding's end; else comm's failure read again (comm_failure: a
   group in flight of it may have failed meanwhile).  Under group_lock */
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
/* one group on the host: its communicator not failed (comm_failure) and its GPU work finished (gpu_finished); its calls
   planned, each crossing call's send buffer a lease
   (operands_check), its operand made in place (in recvbuff where the call holds the whole operand, else a lease of the
   communicator's pool, plan_direct reading the send buffer in place where it may) and its scratch leases of the pool;
   its entries presented (schedule), run with the groups' lock released (run), then finished; a run's failure the
   communicator's (comm_fail: its lists stand mid-group) */
static int comm_issued(const struct ncclComm *comm);
static void calls_free(struct call *list, size_t n);
static ncclResult_t launch(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  for (size_t i = 0; i < n; i++) {
    /* a call on a stream is ordered by that stream's GPU work, which the host path cannot follow */
    if (list[i].stream) return fail(ncclInvalidUsage, "a group on several streams");
    if (reducing(list[i].what) && (result = reduction(list + i))) return result;
    remote |= crosses(list + i);
  }
  if (revoked(comm)) return revoked_fail(comm);
  for (size_t i = 0; i < n; i++)
    if (list[i].scalar_at) memcpy(list[i].pre, list[i].scalar_at, SIZE[list[i].type]);
  pthread_mutex_lock(&group_lock);
  group_numbered(comm);
  const uint64_t holder = comm->groups;
  if (remote) result = comm_failure(comm);
  if (!result && remote && comm_issued(comm))
    result = fail(ncclInvalidUsage, "a host-path group while this communicator's GPU groups are issued and not complete (ncclMeshComplete)");
  if (!result && remote && finished && comm->last) result = gpu_finished(comm);
  if (!result && remote && finished) metal_collect(finished);
  if (!result) result = plan_calls(comm, list, n);
  if (!result) result = operands_check(list, n);
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *s = c->bound;
    const size_t bytes = elements(c) * SIZE[c->type];
    if (c->what == WHAT_SEND && c->peer == comm->rank) {
      if (!(c->copy = malloc(bytes ? bytes : 1))) result = fail(ncclSystemError, "out of memory");
      else memcpy(c->copy, c->send, bytes);
    } else if (c->what == WHAT_SEND) { if (s) s->buffer[0] = c->send_at; }
    else if (c->what == WHAT_RECV) { if (s) s->buffer[0] = c->recv_at; }
    else if (counted(c)) { if (s) { s->buffer[0] = c->send_at; s->buffer[1] = c->recv_at; } }
    else {
      const int in_place = c->what == MESH_ALLREDUCE || c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER ||
                           (c->what == MESH_REDUCE && c->root == comm->rank);
      struct where own = c->recv_at;
      if (!in_place && s && !named(own = spare_where(NULL, comm, bytes, holder))) { result = fail(ncclSystemError, "its operand: %s", last.text); break; }
      if (!in_place && !s) {
        if (!(c->result = malloc(bytes ? bytes : 1))) { result = fail(ncclSystemError, "out of memory"); break; }
        own = (struct where){NULL, (size_t)c->result};
      }
      c->sum = own;
      c->direct = s && plan_direct(c, own);
      if (!c->direct) stage(c, address_of(own));
      if (s) s->buffer[0] = own;
      for (uint32_t b = 1; s && b < s->buffers && !result; b++) {
        if (c->direct && !s->needed[b]) continue;
        if (!named(s->buffer[b] = spare_where(NULL, comm, bytes, holder))) result = fail(ncclSystemError, "a scratch buffer: %s", last.text);
        else if (!c->direct) memcpy(address_of(s->buffer[b]), address_of(own), bytes);
      }
    }
  }
  struct schedule *plans = NULL;
  if (!result && remote && !(result = schedule(NULL, comm, list, n, holder, &plans))) atomic_fetch_add(&comm->runs, 1);
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
        if (self < n) { memmove(c->recv, list[self].copy, c->count * SIZE[c->type]); self++; }
      } else if (counted(c)) {
        size_t sent, landed, own = own_segment(c, &sent, &landed);
        if (own) memmove((unsigned char *)c->recv + landed, (const unsigned char *)c->send + sent, own);
      } else if (!p2p(c)) finish(c);
    }
  }
  pthread_mutex_lock(&group_lock);
  spares_give(comm, holder, 0);
  pthread_mutex_unlock(&group_lock);
  return result;
}

/* -- the Metal path ------------------------------------------------------------------------------------ */

/* a step's result (nccl-mesh-metal.h struct metal_result): its errno a failure, named by `what` and the step's own message */
static ncclResult_t gpu(struct metal_result step, const char *what) {
  return step.status ? fail(ncclSystemError, "%s on the GPU: %s", what, step.message) : ncclSuccess;
}

/* a group issued and not complete: its communicator, its calls, its schedule, its entries [start, end) of each list on
   each of its communicator's channels ([h][0] the send list's, [h][1] the landing list's), the one channel they lie on
   (`lone`: lone_of, read where the group is planned; comm->count where it has none, METAL_NONE where they lie on two
   or more), its number (`holder`: the leases of its communicator's pool it holds) and whether it is a fused crossing;
   the groups in flight are in issue order, a communicator's holding consecutive entries on every list */
struct inflight {
  uint64_t ticket, holder;
  struct ncclComm *comm;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t (*start)[2], (*end)[2];
  uint32_t lone;
  int fused;
  struct inflight *next_group;
};

/* the channel all of a group's entries lie on (struct inflight `lone`): a function of its schedule alone, the entries
   both ends of each channel lay alike (schedule_group, mesh-plan.h mesh_group_lay) */
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
/* the groups lost with their communicators, each with its communicator's failure, until a completion through it
   reports it */
static struct loss { uint64_t ticket; ncclResult_t result; char reason[256]; } *losses;
static size_t lost;

static void call_parts_free(struct call *c) {
  steps_free(c->bound);
  free(c->result);
  free(c->copy);
  free(c->segments);
  free(c->complement);
}
static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) call_parts_free(list + i);
  free(list);
}

/* a group done with: its schedule, its calls, and the leases of its communicator's pool it held given back where
   `retired` (retire: free once every program that named them has ended, stamp_end), else kept (a group lost with its
   communicator, whose programs' ends nothing will see) */
static void inflight_free(struct inflight *g, int retired) {
  if (retired) spares_give(g->comm, g->holder, 1);
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
    inflight_free(g, 0);
  }
  atomic_store(&retired, inflight ? inflight->ticket - 1 : tickets);
}

/* the group in flight on comm whose entries of list `list` on channel h hold `at` */
static struct inflight *inflight_at(struct ncclComm *comm, uint32_t h, int list, uint64_t at) {
  for (struct inflight *g = inflight; g; g = g->next_group)
    if (g->comm == comm && at < g->end[h][list]) return at >= g->start[h][list] ? g : NULL;
  return NULL;
}
static struct message *out_at(struct inflight *g, uint32_t h, uint64_t at) { return g->plans[h].out + (at - g->start[h][0]); }
static struct message *in_at(struct inflight *g, uint32_t h, uint64_t at) { return g->plans[h].in + (at - g->start[h][1]); }
/* channel h's entry `at` of list `list` in none of comm's groups in flight, failed naming the channel's counts and each
   group's entries on it */
static ncclResult_t outside_groups(struct ncclComm *comm, uint32_t h, int list, uint64_t at) {
  const struct channel *ch = comm->channels + h;
  char spans[256] = "";
  size_t used = 0;
  for (struct inflight *g = inflight; g && used < sizeof spans - 48; g = g->next_group)
    if (g->comm == comm)
      used += (size_t)snprintf(spans + used, sizeof spans - used, " [%llu, %llu)", (unsigned long long)g->start[h][list], (unsigned long long)g->end[h][list]);
  return fail(ncclInternalError, "%s entry %llu of channel %u is in no group (published %llu, consumed %llu, waited %llu; groups in flight on it:%s)",
              list ? "landing" : "send", (unsigned long long)at, h, (unsigned long long)ch->published, (unsigned long long)ch->consumed,
              (unsigned long long)ch->waited, *spans ? spans : " none");
}

/* whether channel h's send entry `at` may be published: its step's receive before it consumed */
static int publishable(struct ncclComm *comm, uint32_t h, uint64_t at) {
  struct inflight *g = inflight_at(comm, h, 0, at);
  const struct message *m = g ? out_at(g, h, at) : NULL;
  return m && ready(m->c->bound, m->step);
}

/* the fused crossing of the group being encoded (nccl.h ncclMeshFusableRooted): its records' lists, the room the
   caller gave each (`room`: publications, landings), and the publications and landings written in place of their
   dispatches; under group_lock, for one launch_metal */
static struct { ncclMeshFused *out; uint32_t room[2], published, landed; } fusing;

/* channel h's send entries [published, upto) published, their steps' bytes written by the stream's work before:
   one release storing the holding's `ready` past the last (one system fence, nccl-mesh-kernels.h mesh_step_release),
   or, the fused crossing's, its record for its caller's kernel (metal_send_deferred), whose last threadgroup to write
   its operand stores it */
static ncclResult_t publish(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t upto, int *published) {
  struct channel *ch = comm->channels + h;
  struct inflight *g = inflight_at(comm, h, 0, upto - 1);
  if (!g) return outside_groups(comm, h, 0, upto - 1);
  const struct message *m = out_at(g, h, upto - 1);
  const struct metal_release release = {ch->buffer, ch->ready, m->end, ch->sent, m->end};
  ncclResult_t result;
  if (g->fused) {
    if (fusing.published >= fusing.room[0])
      return fail(ncclInvalidUsage, "a fused crossing's publications past the %u its ncclMeshFused has room for (ncclMeshFusableRooted)", fusing.room[0]);
    result = gpu(metal_send_deferred(program, m->data.buffer, m->data.offset, &release, fusing.out->publish[fusing.published]),
                 "a fused publication's record");
    fusing.published += !result;
  } else {
    void *operands[64];
    size_t n = 0;
    for (uint64_t at = ch->published; at < upto; at++) {
      if (n == sizeof operands / sizeof *operands) { metal_uses(program, operands, n, 0); n = 0; }
      operands[n++] = out_at(g, h, at)->data.buffer;
    }
    metal_uses(program, operands, n, 0);
    result = gpu(metal_publish(program, &release), "a publication");
  }
  if (!result) { ch->published = upto; *published = 1; }
  return result;
}

/* channel h's landing entries from `at` on that fused groups recorded ahead of the channel's count passed
   (complete_fused): the entry the channel consumes next */
static uint64_t consumed_past(struct ncclComm *comm, uint32_t h, uint64_t at) {
  for (struct inflight *g; at < comm->channels[h].lands && (g = inflight_at(comm, h, 1, at)) && g->fused &&
                           in_at(g, h, at)->c->bound->done[in_at(g, h, at)->step];)
    at++;
  return at;
}

/* channel h's landing entry `at` consumed (the channel's next, but a fused group's, complete_fused), as its track's
   item (metal_record; rdma/NCCL.md "Per-channel progress") or, `base` METAL_NONE, as dispatches in the stream's order
   (complete_track): its first wait on the holding's `landed` past it, and on its `sent` past the SEND whose bytes it
   combines into (struct message `sent`), and, where `fold` (one track, the stream's own order) at the group's last
   landing on the channel with every send of the group there already published, past those (the group's end needs no
   wait of its own: retire); a combine's landed bytes combined into its buffer (its first one, a call reading its send
   buffer in place, after a wait: with op landed), a COPY's wait alone (it landed in place), the fused crossing's
   landing recorded for its caller's kernel (metal_land_deferred: its wait for `landed` alone; a kernel that writes a
   send buffer again awaits that publication's `sent` itself, mesh_step_sent); a receive's first event after the receive
   before it (as a track's item, metal_depend: that receive's channel, track `base` + its index, consumed past it; in
   the stream, consumed before it) */
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
      const struct where with = shifted(st->read[m->step], m->offset);
      result = gpu(metal_spin(program, &wait), "a wait");
      if (!result)
        result = gpu(metal_combine3(program, m->c->type, m->c->combine, to.buffer, to.offset, m->data.buffer, m->data.offset, with.buffer, with.offset,
                                    m->bytes / SIZE[m->c->type]), "a first combine");
    } else result = gpu(metal_land(program, ktype(m->c), kop(m->c), to.buffer, to.offset, kcount(m->c, m->bytes), &wait), "a landing");
  } else if (!result && g->fused) {
    if (fusing.landed >= fusing.room[1])
      return fail(ncclInvalidUsage, "a fused crossing's landings past the %u its ncclMeshFused has room for (ncclMeshFusableRooted)", fusing.room[1]);
    result = gpu(metal_land_deferred(program, m->data.buffer, m->data.offset, &wait, fusing.out->land[fusing.landed]), "a fused landing's record");
    fusing.landed += !result;
  } else if (!result) result = gpu(metal_spin(program, &wait), "a wait");
  if (result) return result;
  if (!g->fused && pages_below(ch->sent_waited, sent)) ch->sent_waited = sent;
  st->done[m->step] = 1;
  if (at == ch->consumed) ch->consumed = consumed_past(comm, h, at + 1);
  if (ch->waited < at + 1) ch->waited = at + 1;
  return ncclSuccess;
}

/* the channel every group in flight through `through` (not fused) lies on alone, all of its communicator (each group's
   `lone`, read where it was planned: launch_metal), or METAL_NONE where they lie on two or more channels or are of two
   communicators; a group with no entry lies on none and joins any, and a fused one, recorded, is passed */
static uint32_t one_track(const struct inflight *through) {
  uint32_t h = METAL_NONE;
  for (const struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group) {
    if (g->fused) continue;
    if (g->comm != through->comm || g->lone == METAL_NONE) return METAL_NONE;
    if (g->lone == g->comm->count) continue;
    if (h != METAL_NONE && h != g->lone) return METAL_NONE;
    h = g->lone;
  }
  return h;
}

/* every group in flight through `through` completed where all of them lie on channel h of one communicator
   (one_track): the GPU command stream's own order is the dependency (rdma/NCCL.md "One track").  Plain dispatches in
   one serial order every rank builds alike: the next send entries published as soon as their steps' receives before
   them are consumed (a run of them one release; the fused crossing's alone), else the next landing consumed (its
   step's receive before it consumed, and a combine into bytes an earlier SEND of its call reads only once that SEND is
   published: its wait then awaits `sent` past it); a program whose SEND waits on a receive that waits on that SEND
   fails here.  No track is recorded, no thread picks, and no tracks dispatch runs: no wait on one track can hold
   another channel's */
static ncclResult_t complete_track(struct metal_program *program, struct inflight *through, uint32_t h, int *published) {
  struct ncclComm *const comm = through->comm;
  struct channel *ch = comm->channels + h;
  const uint64_t out = through->end[h][0], in = through->end[h][1];
  ncclResult_t result = ncclSuccess;
  touch(comm);
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

/* every group in flight through `through` completed: the fused ones are records already (complete_fused), so through
   the last that is not, on one track as plain dispatches (complete_track), else each channel progressing on its own
   (nccl-mesh-metal.h metal_record; rdma/NCCL.md "Per-channel progress"): channel h of the k-th communicator met is
   track base[k] + h, its publications to its communicator's last group through `through` each an item of its own
   (nothing credits it; a SEND's step after the consumption of the receive before it), then its consumptions (a combine
   into bytes an earlier SEND of its channel reads after that publication: its `need`); the tracks run as one
   (metal_tracks).  No rank orders anything another rank must share */
static ncclResult_t complete(struct metal_program *program, struct inflight *through, int *published) {
  struct inflight *last = NULL;
  for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group)
    if (!g->fused) last = g;
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
    if (g->fused) continue;
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
      if (g->comm == c) memcpy(end, g->end, c->count * sizeof *end);
    for (uint32_t h = 0; h < c->count && !result; h++) {
      struct channel *ch = c->channels + h;
      const uint32_t track = base[k] + h;
      consumed[track] = ch->consumed;
      while (!result && ch->published < end[h][0]) {
        const uint64_t at = ch->published;
        struct inflight *g = inflight_at(c, h, 0, at);
        if (!g || g->fused) { result = g ? fail(ncclInternalError, "a fused crossing's publication on two tracks") : outside_groups(c, h, 0, at); break; }
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
        if (!g || g->fused) { result = g ? fail(ncclInternalError, "a fused crossing's landing on two tracks") : outside_groups(c, h, 1, at); break; }
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

/* a fused group's steps recorded for its caller's kernels (nccl.h ncclMeshFused), no dispatch: each channel in turn,
   its own publications in order once their steps' receives before them are recorded (on a channel where it publishes
   no group before it has a send entry not yet published: launch_metal completes such a group first, meets; on any
   other an earlier group's sends are that group's), and each landing at its own entry, also where landings of groups
   before it on the channel are not consumed yet (the counts are cumulative, so `landed` past its entry is past theirs;
   the channel's count passes it once theirs are consumed: consumed_past) */
static ncclResult_t complete_fused(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *const comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
  metal_gate(program, comm->violation);
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    struct channel *ch = comm->channels + h;
    const int sends = g->end[h][0] > g->start[h][0];
    for (uint64_t at = g->start[h][1]; !result && ((sends && ch->published < g->end[h][0]) || at < g->end[h][1]);) {
      const struct message *r = at < g->end[h][1] ? in_at(g, h, at) : NULL;
      if (sends && ch->published < g->end[h][0] && publishable(comm, h, ch->published)) result = publish(program, comm, h, ch->published + 1, published);
      else if (r && ready(r->c->bound, r->step)) result = consume(program, comm, h, at++, METAL_NONE, 0);
      else
        result = fail(ncclInternalError, "a fused crossing's entries on channel %u admit no order (published %llu of %llu, recorded %llu of %llu)", h,
                      (unsigned long long)ch->published, (unsigned long long)g->end[h][0], (unsigned long long)at, (unsigned long long)g->end[h][1]);
    }
  }
  return result;
}

/* whether fused group g's records follow the completion of group x, in flight before it (launch_metal): x is the
   library's to encode (not fused), of g's communicator, and on a channel where g publishes x has a send entry not yet
   published (g's release raises `ready`, a count, past it).  Nothing else of x orders g: the bridge lands g's bytes as
   g presents them, and pages both name are their caller's to guard */
static int meets(const struct inflight *x, const struct inflight *g) {
  for (uint32_t h = 0; !x->fused && x->comm == g->comm && h < g->comm->count; h++)
    if (g->end[h][0] > g->start[h][0] && x->end[h][0] > x->start[h][0] && g->comm->channels[h].published < x->end[h][0]) return 1;
  return 0;
}

/* a group's publications that need no wait, each channel's in order, a run of them one release: its earlier entries
   published, its steps' receives before them consumed */
static ncclResult_t issue_group(struct metal_program *program, struct inflight *g, int *published) {
  struct ncclComm *comm = g->comm;
  ncclResult_t result = ncclSuccess;
  touch(comm);
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

/* a group's scratch that crosses no link (a call that crosses none): the stream's workspace (the caller's allocator,
   ordered with its work), else a buffer the program keeps until its end */
static void *workspace(struct metal_program *program, const struct call *c, size_t bytes) {
  if (c->workspace) return c->workspace(bytes ? bytes : 16, c->context);
  if (metal_metal4(program) || c->encoder) {
    /* the Metal 4 form, and the classic form in the caller's encoder, which signal nothing at their end (a borrowed
       program keeps nothing): the communicator's scratch, the group's k-th request its k-th buffer, grown where short
       (the one it replaces kept until the communicator's end, as the GPU may yet read it); its groups run in the
       encoder's order, each complete at its end, so one group's scratch is free for the next */
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

/* whether a call reads its send buffer in place (mesh 95967e6; rdma/NCCL.md Memory), planned into its steps (struct
   steps read, fresh, needed) and its complement: a collective that crosses a link, not fused, premultiplied or merging
   rows, whose send buffer lies in this node's region apart from its result `own` (both paths: a where's address), every
   step's range in its buffer wholly written or wholly unwritten by the rank's steps before it (by round, then order:
   the order the walk encodes a receive before the SEND after it, mesh_steps_before).  Then a SEND of a range no
   earlier step wrote reads the send buffer (its entry names those pages), a receive's first combine of such a range
   writes buffer = send op piece, a scratch buffer no step writes is never made and none is copied, and the ranges of
   buffer 0 no step writes are copied from the send buffer after the publications (`complement`); else (0) the operand
   is made in place as ever (a premultiply, or a copy within the region) */
static int plan_direct(struct call *c, struct where own) {
  struct steps *st = c->bound;
  const size_t z = SIZE[c->type];
  uint64_t lo = 0, hi = elements(c);
  if (!st || p2p(c) || counted(c) || c->fused || c->premultiplied || unit_of(c) > 1 || !st->count) return 0;
  if (c->what == MESH_BROADCAST && c->root != c->comm->rank) return 0;
  if (c->what == MESH_ALLGATHER) { size_t length; lo = segment_at(c, c->comm->rank, &length); hi = lo + length; }
  const struct where send = c->send_at;
  const size_t sent = (size_t)(hi - lo) * z, made = elements(c) * z;
  const unsigned char *a = address_of(send), *b = address_of(own);
  if (!in_region(c->comm, send, sent) || !b || (a < b + made && b < a + sent)) return 0;
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

/* a group's results into recvbuff, once its entries are consumed (a self-addressed SEND's copy among them), loaded
   system-coherent where a COPY may have landed them in the same command buffer (METAL_LAND) */
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

/* the groups in flight whose entries are all published and consumed, in issue order: on each channel where the group
   sent, a wait for `sent` past its sends unless a wait encoded has awaited it (a fused crossing's, its caller's
   kernel's: nccl-mesh-kernels.h mesh_step_await, mesh_step_sent), so the caller and the next group may write what it
   sent; their results; and they are complete */
static ncclResult_t retire(struct metal_program *program) {
  ncclResult_t result = ncclSuccess;
  while (inflight && !result) {
    struct inflight *g = inflight;
    struct ncclComm *comm = g->comm;
    for (uint32_t h = 0; h < comm->count; h++)
      if (comm->channels[h].consumed < g->end[h][1] || comm->channels[h].published < g->end[h][0]) return ncclSuccess;
    touch(comm);
    metal_gate(program, comm->violation);
    for (uint32_t h = 0; h < comm->count && !result && !g->fused; h++) {
      struct channel *ch = comm->channels + h;
      if (g->end[h][0] == g->start[h][0]) continue;
      const mesh_pages last = out_at(g, h, g->end[h][0] - 1)->end;
      if (!pages_below(ch->sent_waited, last)) continue;
      const struct metal_wait wait = {ch->buffer, comm->violation, NULL, 0, ch->sent, 0, (mesh_pages){0}, last, g->end[h][0] - 1, h};
      if (!(result = gpu(metal_spin(program, &wait), "a wait for the group's SENDs"))) ch->sent_waited = last;
    }
    if (!result) result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g, 1);
  }
  return result;
}

/* a program's end, its result: the leases of each communicator it touched that it may name counted as used until the
   GPU has ended it (stamp_end: a stamp that fails fails the program, an earlier failure's message kept); where the
   library made its encoder and the group succeeded, its end signals `finished` at its count (`issued`, which a
   host-path group and a destroy await: a program the caller was told to commit), each communicator it touched keeping
   that count as its `last` and the command buffer as its `command` (gpu_finished); a borrowed encoder's, or a failed
   group's, signals nothing, so no wait of the library's is on a command buffer the caller may never commit.  Under
   group_lock */
static ncclResult_t program_end(struct metal_program *program, int borrowed, ncclResult_t result) {
  for (struct ncclComm *comm = touching; comm; comm = comm->next_touched) {
    const struct said kept = last;
    const ncclResult_t stamped = stamp_end(program, comm);
    if (result) last = kept;
    else result = stamped;
  }
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

/* a walk's failure: comm (the walk's own group's; NULL: none) and every communicator with a group in flight through
   `ticket` failed with it (comm_fail: its lists stand mid-group), their groups lost */
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

/* whether a group is a fused crossing's (nccl.h ncclMeshFused): its steps that cross at most one SEND and at most
   one COPY toward each peer (a publication and a landing record each; each peer's channel completed in the stream's
   order, whose order the caller's kernels keep: complete_track), each the whole of an
   ncclSend's, an ncclRecv's, an in-place ncclBroadcast's or an in-place ncclAllGather's (sendbuff recvbuff + rank x
   sendcount, NCCL's: the buffer the caller's kernel writes is the one sent), no combine (the caller's kernel combines:
   an all-reduce or a reduce is the caller's ncclSend and ncclRecv in one group), no scratch, and no SEND of bytes its
   call also receives (a forwarding rank, a broadcast tree's inner rank: a fused publication is released before the
   kernel that reads the landing, so it would send the buffer's old bytes; ncclMeshFusableRooted answers 0 there) */
static int fusable(const struct call *list, size_t n) {
  const int ranks = list[0].comm->nranks;
  uint8_t *seen = calloc((size_t)ranks, 1);
  if (!seen) return 0;
  int whole = 1;
  for (size_t i = 0; i < n && whole; i++) {
    const struct call *c = list + i;
    size_t length;
    const size_t before = c->what == MESH_ALLGATHER ? segment_at(c, c->comm->rank, &length) : 0;
    if (!crosses(c)) continue;
    if (!p2p(c) && !((c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER) &&
                     same(c->send_at, (struct where){c->recv_at.buffer, c->recv_at.offset + before * SIZE[c->type]}) && c->bound->buffers == 1))
      whole = 0;
    for (uint32_t k = 0; k < c->bound->count && whole; k++) {
      const struct mesh_step *step = c->bound->steps + k;
      if (!piece_bytes(step)) continue;
      const uint8_t bit = step->op == MESH_STEP_SEND ? 1 : 2;
      if (step->op == MESH_STEP_REDUCE || step->peer >= (uint32_t)ranks || seen[step->peer] & bit) whole = 0;
      else seen[step->peer] |= bit;
      for (uint32_t j = 0; j < c->bound->count && whole && step->op == MESH_STEP_SEND; j++) {
        const struct mesh_step *got = c->bound->steps + j;
        if (got->op != MESH_STEP_SEND && piece_bytes(got) && got->buffer == step->buffer && got->first < step->first + step->piece.elements &&
            step->first < got->first + got->piece.elements)
          whole = 0;
      }
    }
  }
  free(seen);
  return whole;
}

/* one group on the stream's command buffer (or, the Metal 4 form, its Metal 4 encoder: nccl-mesh-metal.h metal_begin4),
   taking its calls: each crossing call's send buffer a lease (operands_check), each collective's operand recvbuff
   itself where it holds the whole operand (an all-reduce's, a root's reduce, a broadcast's, an all-gather's), else a
   lease of the communicator's pool (a reduce-scatter's, a non-root's reduce), its scratch leases of the pool, read in
   place where it may (plan_direct); then the group scheduled and its entries presented (schedule: a piece off a page or
   outside the region fails the call as a value, nothing presented), its operands made in place, in flight after the
   others, issued (with issue; a fused one is recorded at once, its steps the caller's kernels', after the groups before
   it that it meets: meets, complete_fused) or completed with every group before it, its waits bounded by
   its calls' greatest polls; its results into recvbuff as it completes; the program's end signals `finished` where the
   library made its encoder.  The Metal 4 form makes its channels' counts and the pool's leases resident through the
   stream's residency set.  A fused group on the classic form takes the stream's residency set too (metal_begin),
   through which the buffers its deferred steps' bindings name are made resident for the caller's kernels. */
static ncclResult_t launch_metal(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  ncclMeshIssue *const issue = list[0].issue;
  ncclResult_t result = ncclSuccess;
  int remote = 0;
  /* the group's waits' bound: its calls' greatest (0, the most: nccl.h NCCL_MESH_POLLS, any 0 among them) */
  uint32_t polls = list[0].polls;
  if (issue) *issue = (ncclMeshIssue){0, 0};
  for (size_t i = 0; i < n && !result; i++) {
    polls = !polls || !list[i].polls ? 0 : polls > list[i].polls ? polls : list[i].polls;
    /* a group's dispatches are one program in one command buffer or encoder: a call encoded into another would be
       ordered by neither */
    if (list[i].command != list[0].command || list[i].encoder != list[0].encoder)
      result = fail(ncclInvalidUsage, "a group on several command buffers or encoders");
    /* Metal's shading language has no 64-bit floating type on Apple GPUs: a float64 combine has no kernel (a float64
       call that combines nothing moves its bytes) */
    else if (reducing(list[i].what) && !on_gpu(list[i].type))
      result = fail(ncclInvalidArgument, "a float64 reduction on the Metal path (Apple GPUs have no float64 arithmetic)");
    else if (reducing(list[i].what)) result = reduction(list + i);
    remote |= crosses(list + i);
  }
  const int metal4 = list[0].table != NULL;
  if (!result && metal4 && (!list[0].encoder || !list[0].residency))
    result = fail(ncclInvalidArgument, "a Metal 4 ncclMeshStream (argumentTable) without its encoder or residency set");
  else if (!result && !metal4 && !list[0].command) result = fail(ncclInvalidArgument, "an ncclMeshStream without a command buffer");
  else if (!result && !metal4 && list[0].fused && !list[0].residency)
    result = fail(ncclInvalidArgument, "a fused crossing on a classic ncclMeshStream without its residency set (the caller's kernels reach "
                                       "the library's buffers by address)");
  if (!result && revoked(comm)) result = revoked_fail(comm);
  pthread_mutex_lock(&group_lock);
  if (!result) group_numbered(comm);
  const uint64_t holder = comm->groups;
  if (!result && remote) result = comm_failure(comm);
  if (!result) result = plan_calls(comm, list, n);
  if (!result) result = operands_check(list, n);
  if (result) { pthread_mutex_unlock(&group_lock); calls_free(list, n); return result; }
  /* a group on a fused stream that is no fused crossing (fusable: one that crosses no link, or not ncclSend and ncclRecv
     toward each peer, an in-place broadcast or all-gather, at most one SEND and one landing of a whole buffer each,
     forwarding nothing) is an ordinary group on that stream, its steps the library's: no record is written and both
     counts are 0, so the caller's kernels publish and await nothing (a form choice, not a refusal) */
  ncclMeshFused *const fused = list[0].fused && remote && fusable(list, n) ? list[0].fused : NULL;
  if (list[0].fused && !fused) list[0].fused->publishes = list[0].fused->lands = 0;
  for (size_t i = 0; i < n && !fused; i++) list[i].fused = NULL;
  if (finished) metal_collect(finished);
  void *const violation = remote ? comm->violation : NULL;
  const char *message = NULL;
  struct metal_program *program = metal4 ? metal_begin4(list[0].encoder, list[0].table, list[0].residency, violation, polls, &message)
                                         : metal_begin(list[0].command, list[0].encoder, fused ? list[0].residency : NULL, violation,
                                                       polls, &message);
  if (!program) {
    pthread_mutex_unlock(&group_lock); calls_free(list, n);
    return fail(ncclSystemError, "%s", message);
  }
  comm->scratch_used = 0;
  if (remote) touch(comm);
  for (uint32_t h = 0; remote && h < comm->count; h++) metal_resident(program, comm->channels[h].buffer);
  const size_t r = (size_t)comm->rank;
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *st = c->bound;
    const size_t bytes = elements(c) * SIZE[c->type];
    if (counted(c)) { if (st) { st->buffer[0] = c->send_at; st->buffer[1] = c->recv_at; } continue; }
    if (p2p(c)) { if (st) st->buffer[0] = c->what == WHAT_SEND ? c->send_at : c->recv_at; continue; }
    const int in_place = c->what == MESH_ALLREDUCE || c->what == MESH_BROADCAST || c->what == MESH_ALLGATHER ||
                         (c->what == MESH_REDUCE && c->root == comm->rank);
    struct where own = c->recv_at;
    if (!in_place && st && !named(own = spare_where(program, comm, bytes, holder))) { result = fail(ncclSystemError, "its operand: %s", last.text); break; }
    if (!in_place && !st) {
      void *scratch = workspace(program, c, bytes);
      if (!scratch) { result = fail(ncclSystemError, "out of GPU memory"); break; }
      own = (struct where){scratch, 0};
    }
    c->sum = own;
    c->direct = plan_direct(c, own);
    if (st) st->buffer[0] = own;
    /* a program's scratch, a tree's or a ring's (mesh-plan.h mesh_compile: a forwarding node's partials, and its own
       operand where it sends that to two nodes or more), leases of the pool: its SENDs read them; the two-node direct
       exchange has none, its SEND reading the operand its REDUCE combines into once its bytes are sent (meeting) */
    for (uint32_t b = 1; st && b < st->buffers && !result; b++) {
      if (c->direct && !st->needed[b]) continue;
      if (!named(st->buffer[b] = spare_where(program, comm, bytes, holder))) result = fail(ncclSystemError, "a scratch buffer: %s", last.text);
    }
  }
  struct inflight *g = NULL;
  int published = 0, scheduled = 0;
  uint64_t ticket = 0;
  if (!result && remote) {
    if (!(g = calloc(1, sizeof *g)) || !(g->start = calloc(2 * (size_t)comm->count + 1, sizeof *g->start))) result = fail(ncclSystemError, "out of memory");
    else {
      g->comm = comm; g->holder = holder; g->fused = fused != NULL;
      g->end = g->start + comm->count;
      for (uint32_t h = 0; h < comm->count; h++) { g->start[h][0] = comm->channels[h].sends; g->start[h][1] = comm->channels[h].lands; }
      result = schedule(program, comm, list, n, holder, &g->plans);
    }
    if (result && g) { free(g->start); free(g); g = NULL; }
    if (!result) {
      for (uint32_t h = 0; h < comm->count; h++) { g->end[h][0] = comm->channels[h].sends; g->end[h][1] = comm->channels[h].lands; }
      g->lone = lone_of(g);
      g->list = list; g->n = n; g->ticket = ticket = ++tickets;
      if (inflight_last) inflight_last->next_group = g; else inflight = g;
      inflight_last = g;
      scheduled = 1;
    }
  }
  /* the operands made in place, after the group's entries are presented and before anything is published: a
     premultiply, or a copy within the region, where a call reads no send buffer in place; a scratch's copy; an
     all-to-all's own segment */
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
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
      result = gpu(metal_copy(program, METAL_PLAIN, st->buffer[b].buffer, st->buffer[b].offset, own.buffer, own.offset, bytes), "a scratch copy");
  }
  if (!result && g) {
    uint32_t lands = 0, publishes = 0;
    for (uint32_t h = 0; h < comm->count; h++) {
      publishes += (uint32_t)(g->end[h][0] - g->start[h][0]);
      lands += (uint32_t)(g->end[h][1] - g->start[h][1]);
    }
    /* a fused group is completed at once, issued or not: its steps are records its caller's kernels run, and the
       library encodes no wait of it.  The groups in flight before it are completed first through the last it meets
       (complete: every group before that one too, in issue order), and the groups after that one stay in flight to
       their own completion (a block-ahead broadcast's, at its caller's wait) */
    const int issuing = issue && !g->fused;
    struct inflight *met = NULL;
    for (struct inflight *x = inflight; g->fused && x != g; x = x->next_group)
      if (meets(x, g)) met = x;
    if (met) {
      result = complete(program, met, &published);
      if (!result) result = retire(program);
    }
    if (g->fused) {
      fusing.out = fused; fusing.room[0] = fusing.out->publishes; fusing.room[1] = fusing.out->lands;
      fusing.published = fusing.landed = 0;
    }
    if (!result) result = g->fused ? complete_fused(program, g, &published) : walk(program, issuing ? g : NULL, issuing ? NULL : g, &published);
    if (!result && g->fused && (fusing.published != publishes || fusing.landed != lands))
      result = fail(ncclInvalidUsage, "a fused crossing whose program is not this rank's publications and landings of its whole operands "
                                      "(ncclMeshFusableRooted)");
    if (g->fused) { fusing.out->publishes = fusing.published; fusing.out->lands = fusing.landed; }
    fusing.out = NULL;
    metal_gate(program, violation);
    for (size_t i = 0; i < n && !result; i++) result = complement_copy(program, list + i);
    if (!result) result = retire(program);
  } else if (!result) result = epilogue(program, list, n);
  result = program_end(program, list[0].encoder != NULL, result);
  if (result && scheduled) walk_failed(comm, issue && !fused ? 0 : ticket, result);
  if (issue && !result) { issue->ticket = !fused && ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  if (!scheduled) spares_give(comm, holder, 0);
  pthread_mutex_unlock(&group_lock);
  if (!scheduled) calls_free(list, n);
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
      for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group) {
        for (uint32_t h = 0; h < g->comm->count; h++) metal_resident(program, g->comm->channels[h].buffer);
        for (uint32_t k = 0; k < g->comm->nspares; k++) {
          void *buffer = NULL;
          size_t offset = 0;
          if (g->comm->spares[k].holder && !lease_buffer(g->comm->spares[k].address, &buffer, &offset)) metal_resident(program, buffer);
        }
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

void ncclMeshMetal4Records(uint64_t *issued, uint64_t *capacity) { metal_records(issued, capacity); }

/* entry e's time in a holding's time ring (mesh-call.h struct mesh_stream_view `posted_at`, `landed_at`), read as its
   counts are: its bridge's mach_absolute_time ticks; 0 where it has none */
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
  /* the bridges' ticks (mach_absolute_time) in ns of the same clock, CLOCK_UPTIME_RAW's */
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

/* whether a rank's moves of a program are at most one SEND and one receive toward each peer, each of every part of what
   it moves: the whole operand in one piece, or (`gather`, an all-gather's) one segment's every part, its own sent and
   another's received; a rank that forwards (a chain's middle, a broadcast tree's inner rank: it sends a part it also
   receives, `held` marking each part's SEND and receive) takes the unfused form, since a fused publication is released
   before the kernel that reads the landing; a move toward a peer this rank has no channel to (toward[peer] < 0: the
   communicator's map joins them by no link) has no fused step; its SENDs and receives counted into *sends and
   *receives */
static int whole_moves(const struct mesh_program *program, uint32_t rank, int gather, uint32_t ranks, const int *toward, uint32_t *sends,
                       uint32_t *receives) {
  uint8_t *seen = calloc(ranks, 1), *held = calloc(program->parts ? program->parts : 1, 1);
  int whole = seen && held;
  *sends = *receives = 0;
  for (uint32_t i = program->first[rank]; i < program->first[rank + 1] && whole; i++) {
    const struct mesh_move *m = program->move + i;
    const uint32_t s = program->segment[m->part];
    uint32_t parts = 0, within = 0;
    for (uint32_t p = 0; p < program->parts; p++) {
      const int of = !gather || program->segment[p] == s;
      parts += of;
      within += of && p >= m->part && p < m->part + m->parts;
    }
    const int send = m->op == MESH_STEP_SEND;
    const uint8_t bit = send ? 1 : 2;
    if (within != m->parts || m->parts != parts || (gather && send != (s == rank)) || m->peer >= ranks || toward[m->peer] < 0 || seen[m->peer] & bit)
      whole = 0;
    else {
      for (uint32_t p = m->part; p < m->part + m->parts; p++) {
        if (held[p] & (bit ^ 3)) whole = 0;
        held[p] |= bit;
      }
      seen[m->peer] |= bit; ++*(send ? sends : receives);
    }
  }
  free(seen);
  free(held);
  return whole;
}

ncclResult_t ncclMeshFusableRooted(ncclComm_t comm, ncclMeshFuseKind kind, int root, size_t count, ncclDataType_t type, ncclRedOp_t op,
                                   int *publishes, int *lands) {
  (void)op;
  if (!publishes || !lands) return fail(ncclInvalidArgument, "publishes or lands is NULL");
  *publishes = *lands = 0;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int what = kind == ncclMeshFuseBroadcast ? MESH_BROADCAST : kind == ncclMeshFuseReduce ? MESH_REDUCE
                 : kind == ncclMeshFuseAllGather ? MESH_ALLGATHER : MESH_ALLREDUCE;
  /* no crossing (one rank, no elements) has no fused step; a rooted kind's root is one of the ranks, and the type one
     NCCL names (SIZE) */
  if (comm->nranks < 2 || !count || (has_root(what) && (root < 0 || root >= comm->nranks)) || (int)type < 0 || (int)type >= ncclNumTypes)
    return ncclSuccess;
  if (!has_root(what)) root = 0;
  const int rooted = comm->rank == root;
  /* an all-reduce and a reduce are the caller's ncclSend and ncclRecv toward every peer (an all-reduce's one-shot form:
     each rank both, n - 1 of each; a reduce's root lands n - 1 and every other rank publishes one), whatever their
     size; a broadcast and an all-gather the communicator's program, which a configured
     form that cuts the operand (chunked) moves in several entries and so does not fuse: the answer is its table's
     (the caller's, or the structural one compiled on its map, here where no group compiled it yet), its operand an
     all-gather's every rank's piece */
  if (what == MESH_BROADCAST || what == MESH_ALLGATHER) {
    pthread_mutex_lock(&group_lock);
    const size_t bytes = (what == MESH_ALLGATHER ? (size_t)comm->nranks : 1) * count * SIZE[type];
    const struct mesh_program *program = comm_compile(comm) ? NULL : program_of(comm, what, root, bytes);
    uint32_t sends = 0, receives = 0;
    const int whole = program && !program->scratch &&
                      whole_moves(program, (uint32_t)comm->rank, what == MESH_ALLGATHER, (uint32_t)comm->nranks, comm->toward, &sends, &receives);
    pthread_mutex_unlock(&group_lock);
    if (whole) { *publishes = (int)sends; *lands = (int)receives; }
    return ncclSuccess;
  }
  /* the peers the form uses (an all-reduce's and a reduce root's every other rank, a reduce's other ranks the root),
     each joined to this rank by a channel (comm's map, toward), else the form has no fused step here: its ncclSend
     or ncclRecv toward an unjoined peer fails in plan_calls, and the caller takes the compiled program */
  for (int q = 0; q < comm->nranks; q++)
    if (q != comm->rank && (what == MESH_ALLREDUCE || rooted || q == root) && comm->toward[q] < 0) return ncclSuccess;
  const int peers = comm->nranks - 1;
  *publishes = what == MESH_ALLREDUCE ? peers : !rooted;
  *lands = what == MESH_ALLREDUCE ? peers : rooted ? peers : 0;
  return ncclSuccess;
}

const char *ncclMeshMetalPrelude(void) { return metal_prelude(); }

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

/* one communicator's calls of a group, launched as its own group: on the Metal path (launch_metal, which owns them) or
   the host path (launch) */
static ncclResult_t launch_one(struct call *list, size_t n) {
  if (list[0].stream) return launch_metal(list, n);
  const ncclResult_t result = launch(list, n);
  if (result) list[0].comm->async = result;
  calls_free(list, n);
  return result;
}

/* a group's calls of one communicator (ncclGroupEnd), launched as a group of their own (launch_one: on a thread of its
   own where they take the host path), its result and its message */
struct part { struct call *list; size_t n; ncclResult_t result; struct said said; pthread_t thread; int threaded, metal; };
static void *part_run(void *argument) {
  struct part *p = argument;
  p->result = launch_one(p->list, p->n);
  p->said = last;
  return NULL;
}
/* whether any of a part's calls crosses a link */
static int part_crosses(const struct part *p) {
  int any = 0;
  for (size_t i = 0; i < p->n; i++) any |= crosses(p->list + i);
  return any;
}

/* the group ended: a fused crossing's records zeroed (filled by its steps: ncclMeshFused), then each communicator's
   calls, in their order, a group of its own, the communicators in the order the group first names them, every one
   launched whatever another's result, the first failure (in that order) returned, as NCCL's group on several
   communicators has every one progress together: the Metal path's each issued (its stream's issue record, else one of
   the library's) and, where the stream asked no issue, all completed through the greatest ticket on the last one's
   stream (ncclMeshComplete: its walk interleaves their channels as tracks; a stream that asked issue gets that ticket
   and `published` any's); the host path's each on a thread of its own, joined; two that cross a link and name one
   fused record, which holds one crossing, the later failing as a value */
ncclResult_t ncclGroupEnd(void) {
  if (depth <= 0) return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart");
  if (--depth) return ncclSuccess;
  struct call *list = calls;
  const size_t n = call_count;
  calls = NULL; call_count = call_capacity = 0;
  if (!n) { free(list); return ncclSuccess; }
  for (size_t i = 0; i < n; i++) {
    ncclMeshFused *f = list[i].fused;
    if (!f) continue;
    if ((f->publishes && !f->publish) || (f->lands && !f->land)) {
      calls_free(list, n);
      return fail(ncclInvalidArgument, "an ncclMeshFused with room for records and no list to hold them");
    }
    if (f->publishes) memset(f->publish, 0, f->publishes * sizeof *f->publish);
    if (f->lands) memset(f->land, 0, f->lands * sizeof *f->land);
  }
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
  for (size_t b = 1; b < parts; b++)
    for (size_t a = 0; a < b && part[b].list; a++)
      if (part[a].list && part[b].list[0].fused && part[a].list[0].fused == part[b].list[0].fused && part_crosses(part + a) && part_crosses(part + b)) {
        part[b].result = fail(ncclInvalidUsage, "a fused crossing on several communicators: one ncclMeshFused holds one crossing's records");
        part[b].said = last;
        calls_free(part[b].list, part[b].n);
        part[b].list = NULL;
      }
  /* the caller's issue record, and the stream the Metal parts complete on where it asked none: the last Metal part's,
     made from its calls' fields (the caller's ncclMeshStream need not outlive its calls) */
  ncclMeshIssue *issue = NULL, internal = {0, 0};
  ncclMeshStream on = {0};
  int metals = 0;
  for (size_t k = 0; k < parts; k++) {
    const struct call *c = part[k].list;
    if (!c || !part[k].metal) continue;
    if (!metals++) issue = c->issue;
    uint32_t polls = c->polls;
    for (size_t i = 1; i < part[k].n; i++) polls = !polls || !c[i].polls ? 0 : polls > c[i].polls ? polls : c[i].polls;
    on = (ncclMeshStream){c->command, c->encoder, c->workspace, c->context, NULL, c->table, c->residency, polls, NULL};
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

/* nccl.h ncclMeshAlltoAllCounted (mesh 3981ca80^'s signature): refused as a value.  Its rows' places in the send and
   receive buffers are computed on the GPU from its counts, and a crossing's bytes are entries the host presents (a
   page's first byte and its bytes, design/transport.md section 14), so its rows could cross only through a copy into
   pages the host knows, which this library never makes (the operator, 2026-10-21: "zero copy methods must always be
   used over access by copy"; "code which can't perform well can fail instead of being run").  *arrived 2, as a failed
   call's */
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

/* ncclCommSplit: every rank's (color, key) and its report (reach_read; none for NCCL_SPLIT_NOCOLOR) all-gathered on comm
   (in a lease, each rank's row from a page), the ranks of this color ordered by key, then rank (two of a color on one
   node refused alike: comm_apart), each rank's streams toward its color's nodes held (comm_hold: the streams its parent holds are not free, EBUSY where none is), and a
   second all-gather on comm carrying each rank's holding's status and its holdings' pages posted, whose difference each
   rank keeps as a statistic as the child's channels are made (comm_join): NCCL's bootstrapSplit (the child's exchange over its parent's), with the init's
   two gathers.  A failure at any rank of a color fails that color's split on each of its ranks as a value, what it held
   let go (comm_free).  Inside a group, its all-gathers run at once, the group's calls set aside until it returns
   (aside) */
static ncclResult_t split(ncclComm_t comm, int color, int key, ncclComm_t *newcomm) {
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; struct report report; } mine = {color, key, {.rank = comm->rank}}, *every = calloc((size_t)n, sizeof *every);
  struct binding { int32_t status, padding; uint64_t posted[MESH_STREAMS][2]; } bound = {0, 0, {{0}}}, *binds = calloc((size_t)n, sizeof *binds);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  struct report *reports = calloc((size_t)n, sizeof *reports);
  /* the all-gathers' rows: a lease, each rank's row from a page of it (an entry starts on a page), in place */
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

/* the calling thread's group set aside (`to` 1: its depth and calls kept, none open) or taken back (0), so a call that
   runs its own collectives at once (ncclCommSplit, ncclCommGrow) runs them inside a group as it does outside one */
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

/* ncclCommShrink: comm's ranks but the excluded ones, in rank order, with no exchange among them (a departed member
   answers none), so holding no stream: a shrink to two ranks or more is made with no channel, its map none, so its
   first call that crosses a link fails as a value (comm_routes: no map of its members; plan_calls: no link toward the
   peer); a rank among those it excludes is in no shrunk communicator (*newcomm NULL, as NCCL_SPLIT_NOCOLOR's); with
   NCCL_SHRINK_ABORT, comm revoked first (ncclCommRevoke) */
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

/* ncclCommGetUniqueId: the id a grow of comm meets at, a listener of comm's root (ncclGetUniqueId) */
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
  return comm ? ncclGetUniqueId(uniqueId) : fail(ncclInvalidArgument, "comm is NULL");
}

/* ncclCommGrow: NCCL's bootstrap at the id of comm's root (ncclCommGetUniqueId), which the root broadcasts on comm to
   comm's other ranks (from a lease: a crossing's bytes are a lease's); comm's ranks keep their ranks (the `rank` they
   pass, NCCL's -1, not read) and a new one (comm NULL) takes `rank`, every rank reporting its node and the nodes its
   bridge reaches; the communicator, named by the id, holds its streams as every init does.  Inside a group, its
   broadcast runs at once, the group's calls set aside until it returns (aside) */
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

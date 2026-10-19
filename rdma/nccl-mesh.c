/* libnccl-mesh (nccl.h): NCCL's API on the bridges' prepared transfers.  The process is one client of its node's
   bridge (an attach, whose slot owns the process's leases of the window: ncclMemAlloc).  Each communicator holds a
   session on it (its program's own slot on that attach under the communicator's name, taken at its init with the link
   lanes its init leased for it, and a ring of slots each way to every other member), opened at its first group that
   crosses a link, on the map current then, once: a communicator whose session failed is made again, never because the
   map moved; a group's calls are planned by the one collective planner and streamed through the rings, the receives
   combined on the host or the GPU. */
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

/* the engine's partial piece [m: rows][l: rows][O: rows x dim] laid as the merge's rows (to_rows; a row with l not above
   0 the row [PARTIAL_NONE, 0, 0]), or rows laid back (nccl-mesh-metal.m partials_rows, partials_columns) */
static void partials_layout(int to_rows, float *dst, const float *src, size_t rows, size_t dim) {
  const size_t u = 2 + dim;
  for (size_t k = 0; k < rows * u; k++) {
    const size_t r = k / u, j = k % u, at = j == 0 ? r : j == 1 ? rows + r : 2 * rows + r * dim + j - 2;
    if (to_rows) dst[k] = src[rows + r] > 0 ? src[at] : j ? 0.0f : PARTIAL_NONE;
    else dst[at] = src[k];
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
__attribute__((visibility("hidden"))) void nccl_mesh_partials(int rows_out, float *dst, const float *src, size_t rows, size_t dim) {
  partials_layout(rows_out, dst, src, rows, dim);
}
#endif


/* -- communicators -------------------------------------------------------------------------------- */

/* an operator made on the communicator: PreMulSum's scalar (`value`), or the partials merge's row of `unit` elements */
struct redop { int used; ncclDataType_t type; unsigned char value[8]; uint32_t unit; };
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; };
struct session;

/* a communicator: rank r on the bridge of node members[r]; the bridges' map over its members as its session read it
   when it opened (mesh-plan.h mesh_map: its relays and routing table in `next`), the wire its session's rings make
   (mesh-plan.h mesh_wire) and the digest every member checks before the session runs (session_agree); its message
   paths, the caller's (count 0: none), else the map's shortest-path trees (`trees`); its table of programs
   (`entries`), the caller's (`listed`, kept for its life), else the structural one compiled along its paths or trees
   at each map read (comm_routes), its configured forms first (`forms`, ncclMeshCommForms, which its splits and shrinks
   keep); its name, and the count of splits and shrinks made of it, which name theirs; the process's attach it holds
   (`attach`, until ncclCommDestroy) and its session's program slot on it (`slot`, taken at its init, its lease words'
   owner word, and moved into its session as that opens: a slot is one program), with the lanes its init leased for
   that slot (`leased`, its bridge's report of its links, and `chosen`, the pair of each, UINT32_MAX none: mesh-call.h
   mesh_lanes_choose); its session (NULL until a group crosses a link); `opening` a session whose start and agreement
   run outside the sessions' lock, which a revoke ends; `revoked`, set by ncclCommRevoke and kept until ncclCommDestroy,
   failing every group after it; `peers`, its bootstrap's sockets, kept for its life once its init succeeds (NCCL's
   bootstrapState), which a revoke shuts down and comm_free closes; the four under `guard`.  A
   non-blocking init (nccl.h ncclMeshCommInitRank) runs on `init` (`initing`: ncclCommDestroy joins it), `async`
   ncclInProgress until it ends and `why` its failure's text */
struct attach;
struct ncclComm {
  struct mesh_link_map map;
  struct mesh_wire wire;
  uint64_t small;
  uint32_t *next, *mark;
  uint64_t digest;
  struct mesh_trees paths, trees;
  struct entry *entries;
  ncclMeshForm *forms;
  uint32_t programs, splits, *members, form_count;
  int nranks, rank, listed;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  struct redop ops[OPS];
  ncclResult_t async;
  struct session *session, *opening;
  pthread_mutex_t guard;
  int revoked, *peers, initing;
  pthread_t init;
  char why[sizeof last];
  void **scratch, **spent;
  size_t *scratch_bytes;
  uint32_t scratches, scratch_used, spents;
  void *merge;
  size_t merge_bytes;
  struct attach *attach;
  struct mesh_ctx slot;
  struct mesh_lane_report leased;
  uint32_t chosen[MESH_PORTS];
};

/* the sessions' one lock: a group, a completion, a session's opening and closing; `opened` is signalled when a session
   that opened outside it is done opening */
static pthread_mutex_t session_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t opened = PTHREAD_COND_INITIALIZER;

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

/* the map read, its trees and the programs compiled on it; with `all`, the caller's paths and programs too */
static void routes_free(struct ncclComm *comm, int all) {
  free(comm->map.link); free(comm->map.bandwidth); free(comm->next); free(comm->mark);
  comm->map = (struct mesh_link_map){0}; comm->next = NULL; comm->mark = NULL;
  trees_free(&comm->trees);
  if (all) trees_free(&comm->paths);
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
  if (comm->merge) metal_release(comm->merge);
  free(comm->scratch); free(comm->scratch_bytes); free(comm->spent);
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

/* the communicator's structural table on the map just read, where the caller listed no programs: its configured forms
   first, in their order (compile_form), then, along the caller's paths (checked against the map) or else the map's
   shortest-path trees (mesh_trees_shortest: each member's in-tree of the bridges' routes), the single-phase all-reduce
   (each member's sum up its own tree; the direct exchange where every two members are linked), the reduce-scatter and
   the all-gather (segment s up and down s's trees), and each root's reduce and broadcast, uncut and of any size
   (mesh_compile; mesh 2bf5a50d^ comm_routes's table, its entries along the routes): a call takes the first entry that
   fits it (program_of), so a collective no form names, or a size past every form's bound, runs the default */
static ncclResult_t comm_routes(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  if (comm->listed || n < 2) return ncclSuccess;
  const struct mesh_trees *along = &comm->paths;
  if (comm->paths.count && mesh_trees_check(&comm->map, &comm->paths))
    return fail(ncclInvalidArgument, "the paths are not trees of the links of the bridges' map (mesh_trees_check)");
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

/* nccl.h ncclMeshCommForms: the forms checked and copied, taken at the communicator's next map read (a table read before
   any session, by ncclMeshFusableRooted, dropped: its next read compiles them); refused once its session is open or
   opening (its members agree on its table there), or where its table is the caller's programs */
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
  pthread_mutex_lock(&session_lock);
  pthread_mutex_lock(&comm->guard);
  const int open = comm->session || comm->opening;
  pthread_mutex_unlock(&comm->guard);
  ncclResult_t result = ncclSuccess;
  if (comm->listed) result = fail(ncclInvalidUsage, "communicator %s runs the caller's programs (ncclMeshCommInitRank): they are its table", comm->name);
  else if (open) result = fail(ncclInvalidUsage, "communicator %s: its forms are set before its session opens (its members agree on its table there)",
                               comm->name);
  else {
    free(comm->forms);
    comm->forms = made; comm->form_count = (uint32_t)count; made = NULL;
    routes_free(comm, 0);
  }
  pthread_mutex_unlock(&session_lock);
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

/* the digest of what a communicator's programs and its session's flows are made from (FNV-1a over 64-bit words): the
   planner's rules (MESH_PLAN_VERSION), its members, the route between each two of them on `map` (mesh_route_path: its
   nodes' marks, the bridges' node numbers) and the bandwidth of each link it crosses (the cables between one pair),
   the session's wire, the caller's paths and the trees (each tree's root, parents and weight's bits: two builds of
   the library that make other trees of one map disagree here, not as a MISMATCH later) and the table of programs.
   Programs are a function of these alone (every hop and tree reads a member pair's route and its links), so members
   whose digests agree run the same programs on the same routes, whatever map serial each read and whatever else the
   map holds; 0 where a route does not reach.  The communicator's name too (its bytes), so ranks of two communicators
   that one bootstrap port paired disagree here as a value (metal-microbench docs/elastic.md#call-delivery) */
static uint64_t mix(uint64_t h, uint64_t word) {
  for (int b = 0; b < 8; b++) h = (h ^ ((word >> (8 * b)) & 0xff)) * UINT64_C(0x100000001b3);
  return h;
}
static uint64_t bits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }
static uint64_t trees_mix(uint64_t h, const struct mesh_trees *trees) {
  h = mix(h, trees->count);
  for (uint32_t t = 0; t < trees->count; t++) {
    h = mix(mix(h, trees->root[t]), bits(trees->log_weight[t]));
    for (uint32_t v = 0; v < trees->nodes; v++) h = mix(h, trees->parent[(size_t)t * trees->nodes + v]);
  }
  return h;
}
static uint64_t map_digest(const struct ncclComm *comm, const struct mesh_link_map *map) {
  const uint32_t n = (uint32_t)comm->nranks, N = map->nodes + map->relays;
  uint32_t *path = calloc((size_t)N + 1, sizeof *path);
  uint64_t h = mix(mix(mix(mix(mix(UINT64_C(0xcbf29ce484222325), MESH_PLAN_VERSION), comm->wire.slot), comm->wire.small), comm->wire.lag), n);
  for (const char *c = comm->name; *c; c++) h = (h ^ (unsigned char)*c) * UINT64_C(0x100000001b3);
  for (uint32_t i = 0; i < n; i++) h = mix(h, comm->members[i]);
  for (uint32_t a = 0; path && a < n; a++)
    for (uint32_t b = a + 1; b < n; b++) {
      const uint32_t hops = mesh_route_path(map, a, b, path);
      if (!hops) { free(path); return 0; }
      h = mix(h, hops);
      for (uint32_t i = 0; i < hops; i++) h = mix(h, map->mark ? map->mark[path[i]] : path[i]);
      for (uint32_t i = 0; i + 1 < hops; i++)
        for (uint32_t l = 0; l < map->links; l++)
          if ((map->link[l][0] == path[i] && map->link[l][1] == path[i + 1]) || (map->link[l][1] == path[i] && map->link[l][0] == path[i + 1]))
            h = map->bandwidth ? mix(h, map->bandwidth[l]) : mix(h, l);
    }
  free(path);
  h = trees_mix(trees_mix(h, &comm->paths), &comm->trees);
  for (uint32_t e = 0; e < comm->programs; e++) {
    const struct mesh_program *x = &comm->entries[e].program;
    h = mix(mix(mix(mix(mix(mix(h, comm->entries[e].below), x->what), x->root), x->nodes), x->parts), x->scratch);
    for (uint32_t p = 0; p < x->parts; p++) h = mix(mix(h, x->segment[p]), bits(x->log_weight[p]));
    for (uint32_t m = 0; m < x->first[x->nodes]; m++)
      h = mix(mix(mix(mix(mix(mix(h, x->move[m].op), x->move[m].peer), x->move[m].round), x->move[m].buffer), x->move[m].part), x->move[m].parts);
  }
  return h ? h : 1;
}

/* -- the process's client of its node's bridge (design/bridge-and-ipc.md#clients, design/prepared-machine.md#M09):
   one attach of the node's region (mesh_attach), made at the process's first lease or first communicator, whichever
   comes first.  Its slot owns every lease the process takes (ncclMemAlloc: a section of the window, mesh-call.h
   mesh_section_create, whose rows' owner is the slot), and each communicator takes its session's program slot on it
   at its init (mesh_program_attach: a slot is one execution's program, under the communicator's name).  The leases'
   bookkeeping is the region's, where the bridge sees it: the bridge retires the slot of a process gone and returns its
   leases' pages as it returns every retired client's rows; `leases` is this process's index of them by address.  A
   lease is an operand of the window it was taken from, and a bridge that stops takes its region with it: while the
   process holds a lease of a bridge that stopped (no longer registered on any device), its next lease, communicator and
   ncclCommRegister fail naming it (ESTALE, attach_fail), and ncclMeshMemLive says so, until the process has freed every
   such lease; the next attaches to the bridge that runs then, so a process never holds leases of two regions (a
   communicator made on the bridge that stopped opens no session: session_open).
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

/* the process's client refused (attach_current's or mesh_program_attach's status) for what `format` names: a lease of a
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

/* a communicator of `nranks` ranks named `name`, this process rank `rank`, its map read at the opening of its session
   (comm_read), holding nothing of its bridge's yet (comm_attach); NULL where out of memory */
static struct ncclComm *comm_new(int nranks, int rank, const char *name) {
  struct ncclComm *comm = calloc(1, sizeof *comm);
  if (comm && !(comm->members = calloc((size_t)nranks, sizeof *comm->members))) { free(comm); comm = NULL; }
  if (!comm) return NULL;
  comm->nranks = nranks; comm->rank = rank;
  for (uint32_t k = 0; k < MESH_PORTS; k++) comm->chosen[k] = UINT32_MAX;
  pthread_mutex_init(&comm->guard, NULL);
  snprintf(comm->name, sizeof comm->name, "%s", name);
  return comm;
}
/* comm's rank r on the bridge of node members[r] (NULL: its init's gather fills them); the process's client attached
   where it has none and held, and for two ranks or more with its members given its session's program slot taken there
   under its name (else by its init once the gather holds them: comm_slot), no lane leased for it yet; 0, or the
   attach's or the slot's refusal (comm_fail), comm then holding nothing of the bridge's */
static int comm_attach(struct ncclComm *comm, const uint32_t *members) {
  pthread_mutex_lock(&attach_lock);
  int status = attach_current(&comm->attach);
  if (!status && comm->nranks > 1 && members) status = mesh_program_attach(&comm->slot, &comm->attach->context, comm->name);
  if (!status) comm->attach->comms++;
  else comm->attach = NULL;
  pthread_mutex_unlock(&attach_lock);
  if (!status && members) memcpy(comm->members, members, (size_t)comm->nranks * sizeof *members);
  return status;
}
static ncclResult_t comm_fail(int status, const char *name) {
  return status == ENOMEM ? fail(ncclSystemError, "out of memory") : attach_fail(status, "communicator %s", name);
}
/* the session's program slot of a communicator attached without its members (comm_attach), under its name: 0, or the
   slot's refusal */
static int comm_slot(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  const int status = mesh_program_attach(&comm->slot, &comm->attach->context, comm->name);
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* a session's program slot given up on every path that ends it: retired (mesh-dataflow.c mesh_retire: its bridge ends
   its program), then its lanes' lease released (mesh-call.h mesh_lanes_release: each word its owner word's to 0, or to
   MESH_LANE_DIRTY alone where the bridge set it) */
static void slot_end(struct mesh_ctx *slot) {
  struct hdr *m = slot->M;
  const uint64_t owner = slot->client;
  if (!m) return;
  mesh_detach(slot);
  mesh_lanes_release(m, owner);
}

/* a communicator's slot no session took ended (slot_end), and its hold of its attach let go (detached where its bridge
   is gone and nothing else holds it) */
static void comm_leave(struct ncclComm *comm) {
  slot_end(&comm->slot);
  if (!comm->attach) return;
  pthread_mutex_lock(&attach_lock);
  comm->attach->comms--;
  attaches_prune();
  pthread_mutex_unlock(&attach_lock);
  comm->attach = NULL;
}

/* the bridges' map over the communicator's members as it is now (mesh-plan.h mesh_map: the other nodes its relays,
   with their marks and the routing table the bridges forward along), into `map` and its own storage (map_free) */
static void map_free(struct mesh_link_map *map) {
  free(map->link); free(map->bandwidth); free(map->next); free(map->mark);
  *map = (struct mesh_link_map){0};
}
static ncclResult_t map_read(const struct ncclComm *comm, struct mesh_link_map *map) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_link_map room = {.nodes = n, .links = n * (n - 1) / 2 + 1};
  for (int read = -ENOSPC;;) {
    *map = (struct mesh_link_map){MESH_LINKS_GRAPH, room.nodes, room.links, calloc(room.links + 1, sizeof *map->link),
                                  calloc(room.links + 1, sizeof *map->bandwidth), 0, calloc((size_t)room.nodes * room.nodes + 1, sizeof *map->next),
                                  calloc((size_t)room.nodes + 1, sizeof *map->mark)};
    if (!map->link || !map->bandwidth || !map->next || !map->mark) { map_free(map); return fail(ncclSystemError, "out of memory"); }
    if ((read = mesh_map(comm->members, n, NULL, NULL, map)) != -ENOSPC) {
      if (read < 0) {
        map_free(map);
        return fail(ncclRemoteError, "the map of this node's bridge over the %u members of %s: %s", n, comm->name, strerror(-read));
      }
      return ncclSuccess;
    }
    room = *map;
    map_free(map);
  }
}

/* the communicator's map read as it is now (map_read), every member reached; its trees and table (comm_routes) */
static ncclResult_t comm_read(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  routes_free(comm, 0);
  ncclResult_t result = map_read(comm, &comm->map);
  if (result) return result;
  comm->next = comm->map.next; comm->mark = comm->map.mark;
  uint32_t apart[2] = {0, 0};
  const int joined = n > 1 ? mesh_link_map_joined(&comm->map, apart) : 0;
  if (joined == ENOTCONN)
    return fail(ncclRemoteError, "node %u, rank %u of %s, has no route to node %u, rank %u, on the map of this node's bridge",
                comm->members[apart[0]], apart[0], comm->name, comm->members[apart[1]], apart[1]);
  if (joined) return fail(ncclInternalError, "the members' map of %s as this node's bridge gives it: %s", comm->name, strerror(joined));
  return comm_routes(comm);
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
   and it holds no slot of the bridge.  The second holds this rank's claims, lease words its bridge's other clients
   read, so each of its sends and receives is bounded by mesh.h MESH_PLATFORM_NS (ETIMEDOUT:
   design/recovery.md#five-seconds).  A rank's connect gives up MESH_PLATFORM_NS after its first try (ETIMEDOUT), so an
   init toward a root that does not listen fails as a value rather than trying for good.  A rank's report is
   its status (0, else why it cannot go on, so every rank fails alike), and its bridge's node and the lane pairs free
   there toward each peer node (mesh-call.h mesh_lanes_free) as its attach reads them; at the second gather its status
   is its slot's and its claim's.  The id is also the communicator's name where its init names none, which its session's
   slot takes on every member (mesh_program_attach). */
struct report { int32_t rank, status; struct mesh_lane_report lanes; };
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

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  memset(id->internal, 0, sizeof id->internal);
  const char *given = getenv("NCCL_COMM_ID");
  if (given && *given) { snprintf(id->internal, sizeof id->internal, "%s", given); return ncclSuccess; }
  char host[256] = {0};
  /* metal-microbench docs/asking-the-mesh.md: a cabled peer is reached on the fabric, never by a name that resolves to
     Wi-Fi or a LAN: the node's mesh identity (bin/mesh-router-init.sh, MESH_ULA_PREFIX on each fabric port) where it
     holds one, else its host name */
  struct ifaddrs *interfaces = NULL;
  if (!getifaddrs(&interfaces)) {
    for (struct ifaddrs *i = interfaces; i; i = i->ifa_next) {
      char text[INET6_ADDRSTRLEN];
      if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET6 || (i->ifa_flags & IFF_LOOPBACK) ||
          !inet_ntop(AF_INET6, &((struct sockaddr_in6 *)i->ifa_addr)->sin6_addr, text, sizeof text) ||
          strncmp(text, MESH_ULA_PREFIX, strlen(MESH_ULA_PREFIX)))
        continue;
      snprintf(host, sizeof host, "%s", text);
      break;
    }
    freeifaddrs(interfaces);
  }
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

/* design/recovery.md#five-seconds: the second gather's sockets, each send and receive bounded by mesh.h
   MESH_PLATFORM_NS: 0, or the errno */
static int bootstrap_bound(const int *peers, int nranks, int rank) {
  const struct timeval bound = {(time_t)(MESH_PLATFORM_NS / 1000000000u), 0};
  for (int r = rank ? 0 : 1; r < (rank ? 1 : nranks); r++)
    if (setsockopt(peers[r], SOL_SOCKET, SO_RCVTIMEO, &bound, sizeof bound) || setsockopt(peers[r], SOL_SOCKET, SO_SNDTIMEO, &bound, sizeof bound))
      return errno;
  return 0;
}

/* design/bridge-and-ipc.md#files: the one rule applied to the reports of `made`'s ranks, in its rank order (mesh-call.h
   mesh_lanes_choose: on each link the lowest lane pair free at both ends), into `chosen`, and its own row claimed at
   its bridge for its session's slot (mesh_lanes_claim, a compare-exchange of each lease word): 0, ENOSPC (a link with
   no pair free at both ends) or EAGAIN (a word taken first, its claims released); kept once the confirming gather finds
   every rank's 0 (lease_keep), else a failure on every rank alike (lease_fail) */
static int lease_claim(struct ncclComm *made, const struct mesh_lane_report *lanes, uint32_t *chosen) {
  const int status = mesh_lanes_choose(lanes, (uint32_t)made->nranks, chosen);
  return status ? status : mesh_lanes_claim(made->attach->context.M, made->slot.client, lanes + made->rank, chosen + (size_t)made->rank * MESH_PORTS);
}
static void lease_keep(struct ncclComm *made, const struct mesh_lane_report *lanes, const uint32_t *chosen) {
  made->leased = lanes[made->rank];
  memcpy(made->chosen, chosen + (size_t)made->rank * MESH_PORTS, sizeof made->chosen);
}
static ncclResult_t lease_fail(const char *name, int rank, int status) {
  return fail(ncclSystemError, "communicator %s: rank %d could not lease its lane pairs at its bridge (%s: ENOSPC, a link of the "
              "communicator with no pair free at both ends; EAGAIN, a pair taken first; else its session's slot there); a communicator "
              "made next reads the lanes as they then stand", name, rank, strerror(status));
}

static void comm_advise(const struct ncclComm *comm);
/* a communicator's table, the caller's operands (nccl.h ncclMeshCommInitRank): its small slot, its paths (NULL: none,
   the map's shortest-path trees) and its programs copied (none: the structural table, comm_routes); 0, or ENOMEM */
static int comm_table(struct ncclComm *made, const struct mesh_trees *paths, const ncclMeshProgram *programs, int count, uint64_t small) {
  made->small = small;
  made->listed = count > 0;
  if ((paths && trees_copy(&made->paths, paths)) || (count && !(made->entries = calloc((size_t)count, sizeof *made->entries)))) return ENOMEM;
  for (int e = 0; e < count; e++) {
    if (program_copy(&made->entries[e].program, programs[e].program)) return ENOMEM;
    made->entries[made->programs++].below = programs[e].below;
  }
  return 0;
}

/* design/bridge-and-ipc.md#files, a communicator's init on NCCL's bootstrap at its id, the one path of every init
   (ncclMeshCommInitRank, blocking or not): its bootstrap's sockets opened (bootstrap_open), then made attached
   (comm_attach) and its lanes read (mesh_lanes_free), then every rank's report gathered; every rank taking its
   session's slot at its own bridge (comm_slot, once the gather holds its members) and leasing its row of the table
   there (lease_claim); a second gather confirming every claim (NCCL's own init exchanges each peer's connection through
   its bootstrap: ncclTransportP2pSetup).  A rank that cannot go on, a slot refused, a link with no pair free at both
   ends (ENOSPC), a claim that found a word taken (EAGAIN), a peer silent past the second gather's bound (ETIMEDOUT:
   bootstrap_bound) or a revoke fails the init as a value, every claim released, made holding nothing of its bridge's
   (comm_leave) and its sockets closed; the next init reads the lease words as they then stand.  Made, it keeps its
   sockets (closed at comm_free); then the checker's thread (comm_advise) */
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
  *mine = (struct report){rank, making, {0}};
  if (!making) mine->status = mesh_lanes_free(made->attach->context.M, &mine->lanes);
  if (!status && nranks > 1) status = gather(peers, nranks, rank, reports, !rank);
  ncclResult_t result = status ? fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status)) : making ? comm_fail(making, made->name) : ncclSuccess;
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status)
      result = r == rank ? fail(ncclSystemError, "communicator %s: the lane pairs free at this node's bridge (mesh_lanes_free): %s", made->name,
                                strerror(reports[r].status))
                         : fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  for (int r = 0; !result && r < nranks; r++) { lanes[r] = reports[r].lanes; made->members[r] = reports[r].lanes.node; }
  if (!result && nranks > 1) {
    const int slotted = !(mine->status = comm_slot(made));
    if (slotted) mine->status = lease_claim(made, lanes, chosen);
    if ((status = bootstrap_bound(peers, nranks, rank)) && !mine->status) mine->status = status;
    if ((status = gather(peers, nranks, rank, reports, 0))) result = fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status));
    else if (!slotted) result = comm_fail(mine->status, made->name);
    for (int r = 0; !result && r < nranks; r++)
      if (reports[r].status) result = lease_fail(made->name, r, reports[r].status);
    if (!result) lease_keep(made, lanes, chosen);
  }
  free(reports); free(lanes); free(chosen);
  if (revoked(made)) result = revoked_fail(made);
  if (result) { bootstrap_close(made); comm_leave(made); }
  else comm_advise(made);
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
  if (!made || comm_table(made, paths, programs, count, small)) {
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

static void session_close(struct session *s);
static void release_all(struct session *s, int result, const char *format, ...);
static void session_retire(struct session *s);
/* ncclCommRevoke: comm revoked for good (`revoked`, until ncclCommDestroy): every socket of its bootstrap shut down, so
   an init still running (a non-blocking one's), its connect and accept reading the revoke at their next look
   (bootstrap_open), fails as a value with its claims released; its session's waits cancelled, its groups in flight failed (a host-path group returns the
   revoke, a GPU wait reads its word cancelled and ends, a completion through one returns the loss), a session opening
   (its start, its agreement awaiting its peers' digests) ended at once (its program's slot retired, which fails its
   peers' programs toward it; its agreement reads the revoke), a session not yet opening never opened (session_open
   reads the revoke where it begins and where it would wait), and every later group on comm failed at once.  All under
   the communicator's own lock alone, never the sessions' lock, which a group of any communicator holds while it waits
   on its peers: a revoke from another thread reaches a member waiting for an absent peer, or behind another
   communicator's slow group, at once.  The session is closed (its flows ended, its slot's rows returned) by
   ncclCommDestroy, once its GPU work has finished */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  if (!comm || revokeFlags != NCCL_REVOKE_DEFAULT) return fail(ncclInvalidArgument, "comm is NULL or revokeFlags is %d", revokeFlags);
  pthread_mutex_lock(&comm->guard);
  __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  if (comm->opening) session_retire(comm->opening);
  if (comm->session) release_all(comm->session, ncclRemoteError, "its communicator %s was revoked (ncclCommRevoke)", comm->name);
  pthread_mutex_unlock(&comm->guard);
  return ncclSuccess;
}

/* comm revoked, its init's thread joined (a non-blocking init's, which the revoke ends), then its session closed and
   comm freed once no session of it is opening (the opener reads the revoke and returns its failure): its session's slot
   retired and its lanes' lease released, by its session's close or, where no session took it, comm_free's (slot_end) */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
  if (comm->initing) pthread_join(comm->init, NULL);
  pthread_mutex_lock(&session_lock);
  while (comm->opening) pthread_cond_wait(&opened, &session_lock);
  session_close(comm->session);
  pthread_mutex_unlock(&session_lock);
  comm_free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

static ncclResult_t session_async(ncclComm_t comm);
/* a non-blocking init's state (ncclInProgress until it ends, then its failure, named again here) or a launch's failure,
   else the failure of the communicator's session (the progress thread's: a link lost, its bridge gone, or its program
   refused), which a group that had already returned, its work enqueued, could not report */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  const ncclResult_t async = __atomic_load_n(&comm->async, __ATOMIC_ACQUIRE);
  if (async != ncclInProgress && *comm->why) fail(async, "%s", comm->why);
  *asyncError = async ? async : session_async(comm);
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

/* design/recovery.md#five-seconds: `event` reached `value`, the GPU's completion awaited with no deadline (every wait a
   session's programs make ends: at its landing, its bound, or the session's ~0, progress_run's included), each look at
   most MESH_PLATFORM_NS and looked again: it decides nothing */
static void gpu_reached(void *event, uint64_t value) {
  while (!metal_wait(event, value, MESH_PLATFORM_NS)) {}
}

/* a host-path group's wait for the points recorded on the lease holding ptr (none where it is no lease): its writers'
   where the group reads it, every one where it writes it, each until the caller's GPU work reaches it (gpu_reached) */
static void lease_wait(const void *ptr, int write) {
  if (!ptr) return;
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
    if (!wait.event) return;
    gpu_reached(wait.event, wait.value);
    metal_release(wait.event);
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

/* -- the session: a communicator's prepared program on its node's bridge, attached under the communicator's name at
   its first group that crosses a link and kept until a failure or the communicator's end.  A channel to each other
   member: a ring of DEPTH slots each way, slot t % DEPTH carrying invocation t (a position), which the bridges carry
   along their map's route where no link joins the two nodes.  Both ends of a channel compute the same positions for a
   group (each collective's rounds in call order, then the point-to-point calls paired in order, a segment as long as
   its longer direction), each publishes every position (filler where it has nothing to send), and each publishes
   position t once it has consumed the peer's t - LAG: the peer then has consumed this end's t - 2 LAG = t - DEPTH, the
   slot t reuses, so no slot is overwritten before it is read (NCCL's Simple protocol's slots, its head credit carried
   by the reverse direction's positions; mesh-plan.h struct mesh_wire).  A
   position's SEND cell and completion word are its invocation's of a cycle of POSITIONS (M30: the word holds 1 plus its
   cycle, read as at least that), reused POSITIONS positions on, which the peer publishes only once this end has
   published POSITIONS - LAG on, after its wait on the position (a group spans any number of positions). ----------- */

#define LAG MESH_LAG
#define DEPTH (2 * LAG)
#define POSITIONS (2 * DEPTH)
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
  size_t slot, pitch;
  struct mesh_section out, in;
  unsigned char *sending, *receiving;
  uint64_t sent, consumed, end, waited, checked;
  void *ring_out, *ring_in;
  size_t cell, word, word_stride;
  uint64_t argument;
  _Atomic uint64_t target, signaled;
};

/* a communicator's session: its program's slot on the process's client of the node's bridge (`context` on `attach`), the
   bridge it opened under (`pid` and `boot`, the region's then), its channels, its progress thread and its failure once
   one is kept (`failed`, stored once `failure` is written, by the one release_all that `claimed` it); `last`, the count
   (`issued`) of the last program signalling `finished` that touched it (`touched` while one is encoded: program_end),
   which its close awaits; `next`, the process's other sessions, closed at its exit */
struct session {
  struct ncclComm *comm;
  struct mesh_ctx context;
  struct attach *attach;
  struct hdr *header;
  uint64_t pid, boot, last;
  struct channel *channels;
  uint32_t count;
  size_t slot, small;
  int running, touched;
  struct mesh_metal_transport transport;
  void *violation_buffer;
  _Atomic uint64_t *violation;
  pthread_t progress;
  pthread_mutex_t lock, park;
  pthread_cond_t wake, landed;
  _Atomic int stop, failed, claimed, runs;
  char failure[512];
  struct session *next;
};
static struct session *sessions;
static int registered;
static void *finished;
static uint64_t issued;

/* a position's completion word as M30 reads it: landed once at least 1 plus its cycle, ~0 its link cancelled */
static uint64_t cycle_of(uint64_t at) { return at / POSITIONS; }
static int landed(uint64_t word, uint64_t at) { return word != UINT64_MAX && word >= 1 + cycle_of(at); }

/* nccl.h ncclMeshSessionWaits */
static _Atomic uint64_t session_waits;

static double now_s(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}

/* every destination's pending words of the session ~0 (mesh_cancel), so every spin on them ends */
static void cancel_all(struct session *s) {
  struct mesh_client *client = mesh_client_at(s->header, s->context.client);
  for (uint32_t d = 0; s->transport.cancel && d < MESH_DESTINATIONS && client->tx[d].node; d++) mesh_cancel(s->header, s->transport.cancel, d);
}

/* a host-path group parked on the session (park) woken: a channel's landings moved (the progress thread) or the session
   failed (release_all) */
static void landed_tell(struct session *s) {
  pthread_mutex_lock(&s->park);
  pthread_cond_broadcast(&s->landed);
  pthread_mutex_unlock(&s->park);
}

/* every destination of the session with pending words cancelled (cancel_all) and the failure kept for its next group:
   its text written before `failed` is stored, so a reader that loads `failed` reads the whole of it */
static void release_all(struct session *s, int result, const char *format, ...) {
  int none = 0;
  if (atomic_compare_exchange_strong(&s->claimed, &none, 1)) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(s->failure, sizeof s->failure, format, arguments);
    va_end(arguments);
    atomic_store_explicit(&s->failed, result, memory_order_release);
  }
  cancel_all(s);
  for (uint32_t h = 0; h < s->count; h++) atomic_store(&s->channels[h].signaled, atomic_load(&s->channels[h].target));
  landed_tell(s);
}

/* the communicator's session's failure, where no group holds the sessions' lock (a group in progress returns its own) */
static void violated(struct session *s);
static int outstanding(struct session *s);
static ncclResult_t session_async(ncclComm_t comm) {
  pthread_mutex_lock(&session_lock);
  struct session *s = comm->session;
  if (s) violated(s);
  if (s && !atomic_load(&s->failed) && !outstanding(s))
    for (uint32_t h = 0; h < s->count; h++)
      if (mesh_host_arrived(&s->context, s->channels[h].in, (uint32_t)s->channels[h].end) == UINT64_MAX) {
        release_all(s, ncclRemoteError, "node %u's program of %s ended while the session was idle (its landings cancelled)", s->channels[h].node,
                    comm->name);
        break;
      }
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

/* a position whose role the two ends disagree on: a filler's slot carries its position's stamp (METAL_STAMP), which a
   receiver checks where it receives a piece (absent) and where it receives nothing (present) */
#define MISMATCH "position %llu on the channel to node %u: this rank receives %s there and its peer sent %s (send and " \
                 "receive calls that do not pair: an ncclSend and an ncclRecv issued separately in the same order on both " \
                 "ranks, or different counts or peers; or different collectives on the two ranks)"

/* a violation the Metal path's walk recorded (struct metal_check), a wait that reached its stream's bound
   (ncclMeshStream polls: words [2], [3] and [4], nccl-mesh-metal.h metal_begin), or a counted all-to-all's counts past
   its layout (word [5], nccl-mesh-metal.h METAL_COUNTED_ARRIVED), made the session's failure */
static void violated(struct session *s) {
  const uint64_t timed = s->violation ? atomic_load_explicit(s->violation + 2, memory_order_acquire) : 0;
  if (timed && !atomic_load(&s->failed)) {
    const uint32_t h = (uint32_t)atomic_load_explicit(s->violation + 3, memory_order_relaxed);
    release_all(s, ncclRemoteError, "position %llu on the channel to node %u landed nothing within %llu polls of a GPU wait (its "
                "stream's bound, ncclMeshStream polls): the peer's work there began that much later, or its link was lost",
                (unsigned long long)(timed - 1), h < s->count ? s->channels[h].node : UINT32_MAX,
                (unsigned long long)atomic_load_explicit(s->violation + 4, memory_order_relaxed));
    return;
  }
  const uint64_t over = s->violation ? atomic_load_explicit(s->violation + 5, memory_order_acquire) : 0;
  if (over && !atomic_load(&s->failed)) {
    release_all(s, ncclInvalidUsage, "a counted all-to-all's counts passed what its layout carries (ncclMeshAlltoAllCounted: a rank's rows "
                "past sendrows, a pair's past min(sendrows, capacity), or a rank's received rows past capacity); its rows past them were "
                "not moved and its arrived word is 2");
    return;
  }
  const uint64_t at = s->violation ? atomic_load_explicit(s->violation, memory_order_acquire) : 0;
  if (!at || atomic_load(&s->failed)) return;
  const uint64_t where = atomic_load_explicit(s->violation + 1, memory_order_relaxed);
  const uint32_t h = (uint32_t)(where >> 1);
  release_all(s, ncclInvalidUsage, MISMATCH, (unsigned long long)(at - 1), h < s->count ? s->channels[h].node : UINT32_MAX,
              where & 1 ? "a piece" : "nothing", where & 1 ? "nothing" : "a piece");
}

/* design/recovery.md#five-seconds: the session's bridge is no longer the one it opened under: the region's bridge_pid
   0 (its stop) or another, its boot another, or that pid no longer running (mesh_gone: an end by a fault stores no ~0,
   and as a bridge makes its region anew at each start, it changes no word of this one) */
static int bridge_gone(const struct session *s) {
  const uint64_t pid = atomic_load_explicit(&s->header->bridge_pid, memory_order_relaxed);
  return pid != s->pid || s->header->boot != s->boot || mesh_gone((pid_t)pid);
}

/* design/recovery.md#five-seconds: a destination of the session whose take its bridge refused, or whose program it
   failed (its status MESH_STOPPED with a code, told by the release of `prepared` after them: mesh-program.c
   status_failed), the node it was bound for, the node where it failed and the code (mesh-call.h mesh_transfers_start),
   which REFUSED names; 1, else 0 */
#define REFUSED "the session of %s toward node %u failed at node %u: %s"
static int start_refused(const struct session *s, uint32_t *node, uint32_t *where, int64_t *code) {
  struct mesh_client *client = mesh_client_at(s->header, s->context.client);
  for (uint32_t d = 0; d < MESH_DESTINATIONS && client->tx[d].node; d++) {
    struct mesh_port_info *status = &client->status[d];
    if (atomic_load_explicit(&status->prepared, memory_order_acquire) != s->context.client ||
        atomic_load_explicit(&status->phase, memory_order_relaxed) != MESH_STOPPED)
      continue;
    const int64_t failed = atomic_load_explicit(&status->code, memory_order_relaxed);
    if (failed) {
      *node = client->tx[d].node; *where = atomic_load_explicit(&status->failed, memory_order_relaxed); *code = failed;
      return 1;
    }
  }
  return 0;
}

/* a landing that read ~0 on the channel to `node`: the failure a destination's status names where its bridge stored
   one (start_refused: a take refused, or a flow failed, with its code), else its link lost */
static ncclResult_t cancelled(const struct session *s, uint32_t node) {
  uint32_t bound, where;
  int64_t code;
  if (start_refused(s, &bound, &where, &code)) return fail(ncclRemoteError, REFUSED, s->comm->name, bound, where, strerror((int)code));
  return fail(ncclRemoteError, "the link to node %u was lost (its landings cancelled)", node);
}

/* the progress thread, the session's watcher: it follows each channel's landings up to its target (the positions a
   group, or an agreement, waits for), and where a landing reads ~0 (cancelled: by its bridge at a link's loss, a flow's
   failure or its stop, or by a revoke) or, with positions pending and none landing, its bridge is gone (bridge_gone)
   or refused or failed a destination's program (start_refused), every destination of the session is cancelled
   (release_all), ending every wait on it.  No clock of its own decides (design/recovery.md#five-seconds: the bridge
   concludes a silent link, B2, and a held SEND, B4; a GPU wait ends at its bound, W3a).  A cancelled word stays ~0
   while the session stands, as a landing never lowers a word (mesh-link.c record_landed). */
/* how long the progress thread waits between its scans of a session's landings */
#define WATCH_NS 1000000L

static void *progress_run(void *argument) {
  struct session *s = argument;
  while (!atomic_load(&s->stop)) {
    violated(s);
    int waiting = 0, moved = 0;
    uint32_t node = 0, where = 0;
    int64_t code = 0;
    for (uint32_t h = 0; h < s->count && !atomic_load(&s->failed); h++) {
      struct channel *ch = s->channels + h;
      const uint64_t from = atomic_load(&ch->signaled), target = atomic_load(&ch->target);
      uint64_t at = from;
      for (; at < target; at++) {
        const uint64_t arrived = mesh_host_arrived(&s->context, ch->in, (uint32_t)at);
        if (arrived == UINT64_MAX) { release_all(s, cancelled(s, ch->node), "%s", last); break; }
        if (!landed(arrived, at)) { waiting = 1; break; }
      }
      if (at > from && !atomic_load(&s->failed)) { atomic_store(&ch->signaled, at); moved = 1; }
    }
    if (moved) landed_tell(s);
    if (waiting && !moved && !atomic_load(&s->failed)) {
      if (bridge_gone(s))
        release_all(s, ncclRemoteError, "this node's bridge (pid %llu) stopped or ended while the session waited (its waits cancelled)",
                    (unsigned long long)s->pid);
      else if (start_refused(s, &node, &where, &code)) release_all(s, ncclRemoteError, REFUSED, s->comm->name, node, where, strerror((int)code));
    }
    /* a watcher, never on a landing's path (the GPU and the host waits read the words themselves): between scans it
       waits WATCH_NS or a wake (new targets, a failure, the stop), so it takes no core from the progress threads that
       carry the crossings (metal-microbench docs/measurement.md 2026-10-17: a node's spinners descheduled ~13 ms) */
    if (waiting && !moved && !atomic_load(&s->failed)) {
      struct timespec until;
      clock_gettime(CLOCK_REALTIME, &until);
      until.tv_nsec += WATCH_NS;
      if (until.tv_nsec >= 1000000000L) { until.tv_sec += 1; until.tv_nsec -= 1000000000L; }
      pthread_mutex_lock(&s->lock);
      if (!atomic_load(&s->stop)) pthread_cond_timedwait(&s->wake, &s->lock, &until);
      pthread_mutex_unlock(&s->lock);
    }
    if (!waiting || atomic_load(&s->failed)) {
      pthread_mutex_lock(&s->lock);
      while (!atomic_load(&s->stop) && (atomic_load(&s->failed) || !outstanding(s))) pthread_cond_wait(&s->wake, &s->lock);
      pthread_mutex_unlock(&s->lock);
    }
  }
  return NULL;
}

/* each channel's target its end, `ahead` positions past it (an agreement's position, published outside a group), the
   progress thread woken to watch them */
static void progress_targets(struct session *s, uint64_t ahead) {
  pthread_mutex_lock(&s->lock);
  for (uint32_t h = 0; h < s->count; h++) atomic_store(&s->channels[h].target, s->channels[h].end + ahead);
  pthread_cond_signal(&s->wake);
  pthread_mutex_unlock(&s->lock);
}

/* every GPU program that touched session s finished (`last`, awaited on `finished`: program_end signals each program's
   count), before a host-path group runs on it (launch); another communicator's GPU work is not waited for */
static void drain(struct session *s) {
  if (!finished) return;
  if (s && s->last) gpu_reached(finished, s->last);
  metal_collect(finished);
}

static void inflight_lose(struct session *s);
/* the session ended: its groups in flight lost, every GPU program that touched it finished (`last`; no other
   session's, so a destroy under the sessions' lock never waits on another communicator's GPU work), its progress
   thread joined, its program's slot retired with its sections (its rings and violation words: the bridge releases
   them) and its lanes' lease released (slot_end); its communicator holds its attach until ncclCommDestroy */
static void session_close(struct session *s) {
  if (!s) return;
  pthread_mutex_lock(&s->comm->guard);
  if (s->comm->session == s) s->comm->session = NULL;
  pthread_mutex_unlock(&s->comm->guard);
  inflight_lose(s);
  if (finished && s->last) gpu_reached(finished, s->last);
  if (finished) metal_collect(finished);
  if (s->running) {
    pthread_mutex_lock(&s->lock);
    atomic_store(&s->stop, 1);
    pthread_cond_signal(&s->wake);
    pthread_mutex_unlock(&s->lock);
    pthread_join(s->progress, NULL);
  }
  for (uint32_t h = 0; h < s->count; h++) { metal_release(s->channels[h].ring_out); metal_release(s->channels[h].ring_in); }
  if (s->violation_buffer) metal_release(s->violation_buffer);
  mesh_metal_transport_destroy(&s->transport);
  slot_end(&s->context);
  for (struct session **at = &sessions; *at; at = &(*at)->next)
    if (*at == s) { *at = s->next; break; }
  pthread_mutex_destroy(&s->lock);
  pthread_cond_destroy(&s->wake);
  pthread_mutex_destroy(&s->park);
  pthread_cond_destroy(&s->landed);
  free(s->channels);
  free(s);
}

/* the session's program's slot retired (mesh-dataflow.c mesh_retire, once whoever calls it): its bridge fails its
   peers' programs toward it */
static void session_retire(struct session *s) { mesh_retire(s->header, s->context.client); }

/* the process's exit: every session closed but one a host-path group waits in (session_run: `runs`), which the process's
   exit releases with the rest of its memory */
static void session_exit(void) {
  pthread_mutex_lock(&session_lock);
  for (struct session *s = sessions, *next; s; s = next) {
    next = s->next;
    if (!atomic_load(&s->runs)) session_close(s);
  }
  pthread_mutex_unlock(&session_lock);
}

static size_t setting(const char *name, size_t otherwise) {
  const char *value = getenv(name);
  const long long x = value ? atoll(value) : 0;
  return x > 0 ? (size_t)x : otherwise;
}

/* design/collectives-dataflow.md#advisories, the checker at a communicator's init: its programs as its table compiles
   them on the bridges' map then (comm_read, on a copy of it) on its bridge's wire, read by mesh_program_advise at
   mesh-advise.c's operands, each below its entry's bound, every warning logged as NCCL logs one, on a thread of its
   own: nothing joins it, and it refuses nothing */
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
  if (comm_read(comm) == ncclSuccess)
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
  copy->nranks = comm->nranks; copy->rank = comm->rank; copy->small = comm->small; copy->listed = comm->listed;
  memcpy(copy->name, comm->name, sizeof copy->name);
  const struct hdr *m = comm->attach->context.M;
  mesh_wire_of((uint64_t)m->pgsz * m->block, setting("NCCL_BUFFSIZE", MESH_BUFFSIZE), comm->small, &copy->wire);
  int made = (copy->members = malloc((size_t)comm->nranks * sizeof *copy->members)) && !trees_copy(&copy->paths, &comm->paths) &&
             (!comm->form_count || (copy->forms = forms_copy(comm->forms, comm->form_count, (uint32_t)comm->nranks))) &&
             (!comm->listed || (copy->entries = calloc((size_t)comm->programs + 1, sizeof *copy->entries)));
  if (made) { memcpy(copy->members, comm->members, (size_t)comm->nranks * sizeof *copy->members); copy->form_count = comm->form_count; }
  for (uint32_t e = 0; made && comm->listed && e < comm->programs; e++)
    if ((made = !program_copy(&copy->entries[e].program, &comm->entries[e].program))) copy->entries[copy->programs++].below = comm->entries[e].below;
  pthread_t thread;
  if (made && !pthread_create(&thread, NULL, advise_run, copy)) pthread_detach(thread);
  else comm_free(copy);
}

/* design/bridge-and-ipc.md#files: the lane pair the communicator's init leased toward node `node` (UINT32_MAX none: no
   cable of this node joins it, or no bootstrap chose one), its report's first link to the node's, and how many links
   it leased that pair on (`cables`) */
static uint32_t lane_pair(const struct ncclComm *comm, uint32_t node, uint32_t *cables) {
  uint32_t pair = UINT32_MAX, n = 0;
  for (uint32_t k = 0; k < comm->leased.count && k < MESH_PORTS; k++) {
    const struct mesh_lane_link *link = comm->leased.link + k;
    if (link->peer != node || comm->chosen[k] == UINT32_MAX || (pair != UINT32_MAX && comm->chosen[k] != pair)) continue;
    pair = comm->chosen[k];
    n++;
  }
  if (cables) *cables = n;
  return pair;
}

/* a ring of DEPTH slots, each slot's first word cleared: a filler's stamp an earlier session left in the window at the
   same position is none of this session's (a piece of fewer bytes, a counted pair's with no rows, leaves it) */
static int ring_make(struct session *s, size_t slot, size_t pitch, struct mesh_section *section, unsigned char **memory) {
  const int status = mesh_section_create(&s->context, DEPTH * pitch, 1, section);
  if (status) return status;
  *memory = mesh_section_address(&s->context, *section, 0);
  for (size_t t = 0; t < DEPTH; t++) memset(*memory + t * pitch, 0, sizeof(uint64_t));
  section->bytes = slot;
  return 0;
}

/* a ring's transfer toward the member `node` on its queue `queue` (a queue's SENDs, and the receives its peer posts,
   run invocation by invocation through every transfer on it, so rings that advance apart need queues apart) */
static int ring_bind(struct session *s, uint32_t node, uint32_t queue, int receive, uint32_t identity, struct mesh_section section,
                     size_t pitch) {
  return mesh_transfer_bind(&s->context, node, queue, receive ? MESH_RECEIVE : MESH_SEND, identity, section,
                            (uint32_t)(pitch / s->header->pgsz), 0, UINT32_MAX, DEPTH);
}

/* a channel's Metal side once its transfers are prepared: the rings wrapped, its SEND cell, its completion words */
static int channel_metal(struct session *s, struct channel *ch) {
  ch->ring_out = metal_wrap(ch->sending, DEPTH * ch->pitch);
  ch->ring_in = metal_wrap(ch->receiving, DEPTH * ch->pitch);
  struct prepared_publication record;
  if (mesh_publication_prepare(&s->context, ch->out.first, NULL) != 1) return EINVAL;
  mesh_publication_prepare(&s->context, ch->out.first, &record);
  ch->cell = (size_t)(record.destination - ((uintptr_t)s->header + s->context.send_off));
  ch->argument = record.argument;
  const struct mesh_metal_input input = mesh_metal_receive_prepare(&s->context, &s->transport, ch->in);
  ch->word = input.offset; ch->word_stride = input.stride;
  return ch->ring_out && ch->ring_in ? 0 : ENOMEM;
}

/* a start's failure as its destination's status names it (start_refused: the node it was bound for, the node where it
   failed, that node's or a relay's on its route, and the code), else as mesh-call.h mesh_transfers_start returned it */
static ncclResult_t start_failure(struct session *s, int status) {
  uint32_t node, where;
  int64_t code;
  if (start_refused(s, &node, &where, &code)) return fail(ncclRemoteError, REFUSED, s->comm->name, node, where, strerror((int)code));
  return fail(status == ESRCH ? ncclRemoteError : ncclSystemError, "the session of %s did not start on this node's bridge: %s",
              s->comm->name, strerror(status));
}

/* the members' agreement, the session's position 0 on each channel: this rank's digest of its programs' making
   (map_digest) and how many cables its init leased its lane pair on toward the channel's node (lane_pair: counts from
   the gathered reports and the one rule, mesh-call.h mesh_lanes_choose, which both ends hold before the session
   opens), published on every channel and every peer's awaited there and compared, before any group runs, so the
   members of a session run the programs of one map or none, and the two ends of each link stripe its invocations over
   as many cables (4df30806 mesh-flow.c program_take's `f->stripes != n`, carried as data: position 0 rides the first
   cable at both ends, design/bridge-and-ipc.md#cables), each end comparing the same two counts.  A leased cable a
   bridge no longer pairs at the take (mesh-program.c program_take, after the start) is its link's loss, or leaves a
   SEND held that fails its flow (design/recovery.md#five-seconds B2, B4).  A session the map changed under as it
   opened, or whose ends leased different cables, fails as a value at both ends naming the disagreement.  It is
   awaited as a landing is: position 0 is the progress thread's target, so the wait ends at its landing, a
   cancellation (its link lost, its bridge gone or its destination's take refused: progress_run, cancelled) or a
   revoke; `serial`, the map serial this rank read its map at, names it */
struct agreement { uint64_t digest, serial; uint32_t version, node, cables, padding; };
_Static_assert(sizeof(struct agreement) <= MESH_SMALL_LEAST, "a small slot carries the agreement");
static ncclResult_t session_agree(struct session *s, uint64_t serial) {
  struct ncclComm *comm = s->comm;
  struct agreement mine = {comm->digest, serial, MESH_PLAN_VERSION, comm->members[comm->rank], 0, 0};
  for (uint32_t h = 0; h < s->count; h++) {
    lane_pair(comm, s->channels[h].node, &mine.cables);
    memcpy(s->channels[h].sending, &mine, sizeof mine);
    mesh_host_publish(&s->context, s->channels[h].out, 0);
  }
  progress_targets(s, 1);
  for (uint32_t h = 0; h < s->count;) {
    struct channel *ch = s->channels + h;
    const uint64_t arrived = mesh_host_arrived(&s->context, ch->in, 0);
    const ncclResult_t failed = (ncclResult_t)atomic_load(&s->failed);
    if (revoked(comm)) return revoked_fail(comm);
    if (failed) return fail(failed, "%s", s->failure);
    if (arrived == UINT64_MAX) return cancelled(s, ch->node);
    if (!landed(arrived, 0)) { usleep(20); continue; }
    struct agreement theirs;
    memcpy(&theirs, ch->receiving, sizeof theirs);
    if (theirs.version != mine.version || theirs.digest != mine.digest)
      return fail(ncclRemoteError, "the members' maps differ: this rank (node %u) planned %s on map serial %llu, digest %016llx, "
                  "planner version %llu, and node %u on serial %llu, digest %016llx, version %llu: the map changed as the session "
                  "opened, the members run different libraries, or ranks of two communicators met at one bootstrap (the digest holds "
                  "each one's name); the group fails, and the communicator is made again on the map then",
                  (unsigned)mine.node, comm->name, (unsigned long long)mine.serial, (unsigned long long)mine.digest,
                  (unsigned long long)mine.version, (unsigned)theirs.node, (unsigned long long)theirs.serial,
                  (unsigned long long)theirs.digest, (unsigned long long)theirs.version);
    uint32_t leased = 0;
    lane_pair(comm, ch->node, &leased);
    if (theirs.cables != leased)
      return fail(ncclRemoteError, "node %u leased its session of %s toward this rank (node %u) on %u cables, and this rank leased it on %u "
                  "toward that node: the two ends would stripe its invocations over different cables (a cable paired at one end only as "
                  "the communicator was made); the group fails, and the communicator is made again",
                  ch->node, comm->name, (unsigned)mine.node, theirs.cables, leased);
    ch->sent = ch->consumed = ch->end = ch->checked = ch->waited = 1;
    atomic_store(&ch->target, 1); atomic_store(&ch->signaled, 1);
    h++;
  }
  return ncclSuccess;
}

/* nccl.h ncclMeshCommAgree: the members' agreement on taking up a held communicator, one position of every channel
   (its `end`) carrying {MESH_PLAN_VERSION, digest, ticket, node}, published and awaited as a landing is, the progress
   thread's target, with session_lock released (a peer still preparing is waited for; a revoke, a cancelled landing,
   its bridge gone, its program refused or a disagreement ends it), every cursor then past it */
struct taken { uint64_t version, digest, ticket, node; };
_Static_assert(sizeof(struct taken) <= MESH_SMALL_LEAST, "a small slot carries the agreement");
ncclResult_t ncclMeshCommAgree(ncclComm_t comm, uint64_t ticket) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  if (revoked(comm)) return revoked_fail(comm);
  pthread_mutex_lock(&session_lock);
  struct session *s = comm->session;
  if (!s) { pthread_mutex_unlock(&session_lock); return ncclSuccess; }
  violated(s);
  const ncclResult_t failed = (ncclResult_t)atomic_load(&s->failed);
  if (failed) {
    const ncclResult_t result = fail(failed, "%s", s->failure);
    pthread_mutex_unlock(&session_lock);
    return result;
  }
  for (uint32_t h = 0; h < s->count; h++) {
    const struct channel *ch = s->channels + h;
    if (ch->sent != ch->end || ch->consumed != ch->end || ch->waited != ch->end || ch->checked != ch->end ||
        atomic_load(&ch->target) != ch->end || atomic_load(&ch->signaled) != ch->end) {
      const ncclResult_t result = fail(ncclRemoteError, "communicator %s: its session has positions in flight toward node %u (sent %llu, "
                                       "consumed %llu, landed %llu of %llu): it is not taken up", comm->name, ch->node,
                                       (unsigned long long)ch->sent, (unsigned long long)ch->consumed,
                                       (unsigned long long)atomic_load(&ch->signaled), (unsigned long long)ch->end);
      pthread_mutex_unlock(&session_lock);
      return result;
    }
  }
  const struct taken mine = {MESH_PLAN_VERSION, comm->digest, ticket, comm->members[comm->rank]};
  for (uint32_t h = 0; h < s->count; h++) {
    struct channel *ch = s->channels + h;
    memcpy(ch->sending + (ch->end % DEPTH) * ch->pitch, &mine, sizeof mine);
    mesh_host_publish(&s->context, ch->out, ch->end);
  }
  progress_targets(s, 1);
  pthread_mutex_unlock(&session_lock);
  for (uint32_t h = 0; h < s->count;) {
    struct channel *ch = s->channels + h;
    const uint64_t at = ch->end, arrived = mesh_host_arrived(&s->context, ch->in, (uint32_t)at);
    const ncclResult_t lost = (ncclResult_t)atomic_load(&s->failed);
    if (revoked(comm)) return revoked_fail(comm);
    if (lost) return fail(lost, "%s", s->failure);
    if (arrived == UINT64_MAX)
      return fail(ncclRemoteError, "node %u's program of %s ended while it was held (its landings cancelled): the communicator is made again",
                  ch->node, comm->name);
    if (!landed(arrived, at)) { usleep(20); continue; }
    struct taken theirs;
    memcpy(&theirs, ch->receiving + (at % DEPTH) * ch->pitch, sizeof theirs);
    if (theirs.version != mine.version || theirs.digest != mine.digest || theirs.ticket != ticket)
      return fail(ncclRemoteError, "node %u took up %s for another execution or program (ticket %016llx, digest %016llx, version %llu; this "
                  "rank's %016llx, %016llx, %llu): the communicator is made again", ch->node, comm->name,
                  (unsigned long long)theirs.ticket, (unsigned long long)theirs.digest, (unsigned long long)theirs.version,
                  (unsigned long long)ticket, (unsigned long long)mine.digest, (unsigned long long)mine.version);
    h++;
  }
  pthread_mutex_lock(&session_lock);
  for (uint32_t h = 0; h < s->count; h++) {
    struct channel *ch = s->channels + h;
    const uint64_t next = ch->end + 1;
    ch->sent = ch->consumed = ch->end = ch->checked = ch->waited = next;
    atomic_store(&ch->target, next); atomic_store(&ch->signaled, next);
  }
  pthread_mutex_unlock(&session_lock);
  return ncclSuccess;
}

/* the window's bytes of one session's leases and of one relayed direction of it, each lease a section of whole blocks
   (mesh-call.c mesh_section_create).  A ring: DEPTH slots (ring_make), bound as one transfer of DEPTH invocations of a
   slot each (ring_bind), which a relay mirrors at mesh.h mesh_relay_slot's bytes (the same DEPTH slots; a slot is
   whole blocks, so its pages counted in blocks are its bytes).  A destination's transfer lists: MESH_QUEUES each
   way, each first room for the transfers a block holds of them (mesh-call.c mesh_lists_grow), a ring's one each here.
   The SEND cells and targets of `sends` SEND transfers in one lease (mesh_transfers_prepare: POSITIONS invocations and
   a terminal cell a transfer, a target each). */
#define WHOLE(bytes, block) (((bytes) + (block) - 1) / (block) * (block))
static uint64_t ring_bytes(uint64_t slot, const struct mesh_wire *wire) {
  const struct mesh_transfer bound = {.count = 1, .invocation_pages = (uint32_t)(mesh_wire_pitch(wire, slot) / wire->block), .depth = DEPTH,
                                      .bytes = slot};
  return WHOLE(mesh_relay_slot(&bound, DEPTH, (uint32_t)wire->block), wire->block);
}
static uint64_t lists_bytes(const struct mesh_wire *wire) {
  const uint64_t room = wire->block / (MESH_QUEUES * 2 * sizeof(struct mesh_transfer));
  return WHOLE((uint64_t)MESH_QUEUES * 2 * (room ? room : 1) * sizeof(struct mesh_transfer), wire->block);
}
static uint64_t cells_bytes(uint64_t sends, const struct mesh_wire *wire) {
  return WHOLE((sends ? sends * (POSITIONS + 1) : 1) * sizeof(struct mesh_send) + sends * sizeof(struct mesh_target), wire->block);
}

/* one session toward `peers` other members (ncclMeshSessionBytes): toward each peer its four rings (two of the large
   class and two of the small, out and in) and its destination's lists; its SEND cells and targets (a ring of each
   class out toward each peer); its completion words (mesh_host_inputs: a word a received chunk of each of the
   POSITIONS invocations, each destination's from a 16-word boundary); its cancellation ranges (one a destination);
   and its violation page */
static uint64_t session_bytes(uint64_t peers, const struct mesh_wire *wire) {
  const uint64_t block = wire->block, page = (uint64_t)getpagesize();
  const uint64_t rings = 2 * ring_bytes(wire->slot, wire) + 2 * ring_bytes(wire->small, wire);
  const uint64_t frames = (mesh_wire_pitch(wire, wire->slot) / block + mesh_wire_pitch(wire, wire->small) / block) * POSITIONS;
  const uint64_t words = peers ? (peers - 1) * ((frames + 15) & ~UINT64_C(15)) + frames : 0;
  return peers * (rings + lists_bytes(wire)) + cells_bytes(peers * 2, wire) + WHOLE(sizeof(uint64_t) * (words ? words : 1), block) +
         WHOLE(sizeof(struct mesh_cancellation) + (peers ? peers : 1) * sizeof(struct mesh_cancel_range), block) + WHOLE(page, block);
}

/* one direction of a session a relay's bridge forwards (ncclMeshRelayBytes; mesh-program.c mirror): the direction's
   transfers, a ring of each class, mirrored; the lists of its two destinations, the sender and the receiver; and the
   SEND cells and targets of its two transfers */
static uint64_t relay_bytes(const struct mesh_wire *wire) {
  return ring_bytes(wire->slot, wire) + ring_bytes(wire->small, wire) + 2 * lists_bytes(wire) + cells_bytes(2, wire);
}
#undef WHOLE

ncclResult_t ncclMeshSessionBytes(int peers, const struct mesh_wire *wire, uint64_t *bytes) {
  if (peers < 0 || !wire || !wire->small || !wire->slot || !wire->block || !bytes) return fail(ncclInvalidArgument, "ncclMeshSessionBytes's arguments");
  *bytes = session_bytes((uint64_t)peers, wire);
  return ncclSuccess;
}

ncclResult_t ncclMeshRelayBytes(const struct mesh_wire *wire, uint64_t *bytes) {
  if (!wire || !wire->small || !wire->slot || !wire->block || !bytes) return fail(ncclInvalidArgument, "ncclMeshRelayBytes's arguments");
  *bytes = relay_bytes(wire);
  return ncclSuccess;
}

/* the communicator's session opened, where it was not revoked, on the program slot its init took and the lanes it
   leased (once: the slot is the session's from here, and a communicator whose session ended is made again): on the
   bridges' map as it is now (comm_read, at the map serial read before it), its wire the bridge's block's
   (mesh_wire_of, NCCL_BUFFSIZE) and its digest (map_digest), a channel of each class to every other member (a
   point-to-point call may name any, and every member binds the same channels whatever map it read, so the agreement
   below meets on them), each destination's lanes its lane pair's (struct mesh_tx `lane`: the large class's queue 0
   on the pair's first lane, the small class's queue 1 on its second, as mesh-stream.c; a destination with none fails
   the session before anything is handed to the bridge, ENOLCK), its transfers prepared and its progress thread
   started (under the bridge it opens under: `pid`, `boot`); then, with the sessions' lock released (and `opening` the
   session, which a revoke ends), its transfers started (mesh_transfers_start) and the members' agreement on their
   programs (session_agree, which the progress thread watches).  Called and returning with the lock held. */
static ncclResult_t session_open(ncclComm_t comm) {
  if (revoked(comm)) return revoked_fail(comm);
  if (!comm->slot.M || !attach_live(comm->attach))
    return fail(ncclRemoteError, "communicator %s: its session's slot, and its lanes' lease with it, went with its last session or the "
                "bridge it was made on (a failure, or that bridge's stop); it is made again (ncclCommDestroy, then an init)", comm->name);
  uint64_t serial = 0;
  mesh_map_wait(&serial, 0);
  ncclResult_t result = comm_read(comm);
  if (result) return result;
  struct session *s = calloc(1, sizeof *s);
  if (!s) return fail(ncclSystemError, "out of memory");
  s->context = comm->slot;
  comm->slot = (struct mesh_ctx){0};
  s->attach = comm->attach;
  pthread_mutex_init(&s->lock, NULL);
  pthread_cond_init(&s->wake, NULL);
  pthread_mutex_init(&s->park, NULL);
  pthread_cond_init(&s->landed, NULL);
  s->comm = comm;
  const struct hdr *m = s->header = s->context.M;
  s->pid = atomic_load(&s->header->bridge_pid); s->boot = m->boot;
  if (m->node != comm->members[comm->rank]) {
    result = fail(ncclInvalidUsage, "rank %d of %s runs on the bridge of node %u, not of node %u", comm->rank, comm->name, m->node,
                  comm->members[comm->rank]);
    session_close(s);
    return result;
  }
  mesh_wire_of((uint64_t)m->pgsz * m->block, setting("NCCL_BUFFSIZE", MESH_BUFFSIZE), comm->small, &comm->wire);
  comm->digest = map_digest(comm, &comm->map);
  s->slot = comm->wire.slot;
  s->small = comm->wire.small;
  const uint32_t N = (uint32_t)comm->nranks, me = (uint32_t)comm->rank;
  int status = (s->channels = calloc(2 * (size_t)N, sizeof *s->channels)) ? 0 : ENOMEM;
  for (uint32_t r = 0; r < N && !status; r++)
    for (uint32_t large = 0; r != me && large < 2 && !status; large++) {
      struct channel *ch = s->channels + s->count++;
      ch->node = comm->members[r];
      ch->large = (int)large;
      ch->slot = large ? s->slot : s->small;
      ch->pitch = mesh_wire_pitch(&comm->wire, ch->slot);
      status = ring_make(s, ch->slot, ch->pitch, &ch->out, &ch->sending);
      if (!status) status = ring_make(s, ch->slot, ch->pitch, &ch->in, &ch->receiving);
      if (!status) status = ring_bind(s, ch->node, !large, 0, IDENTITY + 2 * me + large, ch->out, ch->pitch);
      if (!status) status = ring_bind(s, ch->node, !large, 1, IDENTITY + 2 * r + large, ch->in, ch->pitch);
    }
  struct mesh_client *client = mesh_client_at(s->header, s->context.client);
  for (uint32_t h = 0; h < s->count && !status && !result; h++) {
    const uint32_t pair = lane_pair(comm, s->channels[h].node, NULL);
    struct mesh_tx *tx = &client->tx[mesh_destination(client, s->channels[h].node)];
    if (pair == UINT32_MAX)
      result = fail(ncclInvalidUsage, "communicator %s holds no lane pair toward node %u (ENOLCK): only an exchange leases them "
                    "(an init's bootstrap, ncclCommSplit over its parent), and only toward a node on a cable of this node's bridge",
                    comm->name, s->channels[h].node);
    else { tx->lane[0] = MESH_LANE_FIRST + 2 * pair; tx->lane[1] = MESH_LANE_FIRST + 2 * pair + 1; }
  }
  if (result) { session_close(s); return result; }
  if (!status) status = mesh_transfers_prepare(&s->context, 1, POSITIONS, DEPTH);
  if (!status) status = mesh_transfers_digest(&s->context, comm->digest);
  if (!status) status = mesh_transfers_cyclic(&s->context);
  if (!status) status = mesh_metal_transport_create(&s->context, metal_device(), &s->transport);
  for (uint32_t h = 0; h < s->count && !status; h++) status = channel_metal(s, s->channels + h);
  struct mesh_section words = {0};
  if (!status) status = mesh_section_create(&s->context, (size_t)getpagesize(), 1, &words);
  if (!status) {
    s->violation = mesh_section_address(&s->context, words, 0);
    memset((void *)s->violation, 0, (size_t)getpagesize());
    if (!(s->violation_buffer = metal_wrap((void *)s->violation, (size_t)getpagesize()))) status = ENOMEM;
  }
  if (!status && !finished && !(finished = metal_event())) status = ENOMEM;
  if (!status && (status = pthread_create(&s->progress, NULL, progress_run, s))) result = fail(ncclSystemError, "the progress thread: %s", strerror(status));
  else if (status) result = fail(ncclSystemError, "the session of %s on this node's bridge: %s", comm->name, strerror(status));
  s->running = !status;
  if (result) { session_close(s); return result; }
  pthread_mutex_lock(&comm->guard);
  const int before = revoked(comm);
  if (!before) comm->opening = s;
  pthread_mutex_unlock(&comm->guard);
  if (before) { session_close(s); return revoked_fail(comm); }
  pthread_mutex_unlock(&session_lock);
  const double began = now_s();
  if ((status = mesh_transfers_start(&s->context))) result = start_failure(s, status);
  if (!result) result = session_agree(s, serial);
  atomic_fetch_add(&session_waits, (uint64_t)((now_s() - began) * 1e9));
  pthread_mutex_lock(&session_lock);
  pthread_mutex_lock(&comm->guard);
  comm->opening = NULL;
  if (revoked(comm)) result = revoked_fail(comm);
  pthread_mutex_unlock(&comm->guard);
  pthread_cond_broadcast(&opened);
  if (result) { session_close(s); return result; }
  s->next = sessions;
  sessions = s;
  pthread_mutex_lock(&comm->guard);
  comm->session = s;
  if (revoked(comm)) release_all(s, ncclRemoteError, "its communicator %s was revoked (ncclCommRevoke)", comm->name);
  pthread_mutex_unlock(&comm->guard);
  if (!registered) registered = !atexit(session_exit);
  return ncclSuccess;
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

/* every call that crosses a link planned: a collective along the communicator's trees (an uneven one's segments its counts,
   mesh_program_steps_cut: its program's segments must be the ranks), its operand's element a merge's row where it
   merges, a point-to-point call one SEND or COPY step */
static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
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
      continue;
    }
    if (counted(c)) { s->count = counted_steps(comm, c, s->steps); s->borrowed = 1; continue; }
    s->count = mesh_program_steps_cut(program, (uint32_t)comm->rank, operand, s->segments, s->steps);
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
struct schedule { struct message *out, *in; size_t outs, ins, out_at, in_at; };

static size_t piece_bytes(const struct mesh_step *step) { return step->piece.elements * step->piece.element_bytes; }
static int in_class(const struct session *s, const struct mesh_step *step, int large) {
  return mesh_wire_large(&s->comm->wire, piece_bytes(step), step->piece.element_bytes) == large;
}

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
   collective's (mesh-plan.h mesh_lay, the one layout both ends make, `scratch` room for its placements) in
   call order, then each counted all-to-all's (mesh_lay of its steps toward the peer: both ends' alike, as its pair's
   messages are laid by their segments' and the bound's sizes alone; after the collectives, so a rank that cannot
   plan another's counted call, order_group, still lays that rank's collectives where they are), then its
   point-to-point calls, its j-th SEND to the peer beside its j-th receive from it; steps[i] and
   counts[i] are its steps of call i (a point-to-point call's only for this rank).  The count of positions; `made` gets
   the messages (no zero-piece step), *count of them */
static uint64_t lay(const struct session *ses, const struct call *list, size_t n, struct mesh_step *const *steps, const uint32_t *counts,
                    int peer, int large, struct mesh_place *scratch, struct placing *made, size_t *count) {
  const struct mesh_wire *wire = &ses->comm->wire;
  uint64_t at = 0;
  *count = 0;
  for (int pass = 0; pass < 2; pass++)
    for (size_t i = 0; i < n; i++) {
      if (p2p(list + i) || counted(list + i) != pass) continue;
      uint32_t placed = 0;
      at = mesh_lay(steps[i], counts[i], (uint32_t)peer, large, wire, at, scratch, &placed);
      for (uint32_t k = 0; k < placed; k++)
        made[(*count)++] = (struct placing){(uint32_t)i, scratch[k].step, (int)scratch[k].send, scratch[k].first, scratch[k].pieces};
    }
  for (size_t s = 0, r = 0;; s++, r++) {
    while (s < n && !(p2p(list + s) && counts[s] && list[s].what == WHAT_SEND && (int)steps[s][0].peer == peer && in_class(ses, steps[s], large))) s++;
    while (r < n && !(p2p(list + r) && counts[r] && list[r].what == WHAT_RECV && (int)steps[r][0].peer == peer && in_class(ses, steps[r], large))) r++;
    if (s >= n && r >= n) break;
    uint64_t out = at, in = at;
    if (s < n && piece_bytes(steps[s]))
      out += (made[(*count)++] = (struct placing){(uint32_t)s, 0, 1, out, mesh_wire_pieces(wire, piece_bytes(steps[s]), steps[s]->piece.element_bytes)}).pieces;
    if (r < n && piece_bytes(steps[r]))
      in += (made[(*count)++] = (struct placing){(uint32_t)r, 0, 0, in, mesh_wire_pieces(wire, piece_bytes(steps[r]), steps[r]->piece.element_bytes)}).pieces;
    at = out > in ? out : in;
  }
  return at;
}

/* each channel's positions for the group, from the last scheduled one's end (lay); each step's positions and the receive
   before it (mesh-plan.h mesh_steps_before) kept, a receive with none done */
static ncclResult_t schedule_group(struct session *s, struct call *list, size_t n, struct schedule *plans) {
  const ncclComm_t comm = list[0].comm;
  size_t total = 1;
  for (size_t i = 0; i < n; i++) total += list[i].bound ? list[i].bound->count : 0;
  struct mesh_step **steps = calloc(n + 1, sizeof *steps);
  uint32_t *counts = calloc(n + 1, sizeof *counts);
  struct placing *made = calloc(total, sizeof *made);
  struct mesh_place *scratch = calloc(total, sizeof *scratch);
  ncclResult_t result = steps && counts && made && scratch ? ncclSuccess : fail(ncclSystemError, "out of memory");
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
    const uint64_t positions = lay(s, list, n, steps, counts, rank_of(comm, ch->node), ch->large, scratch, made, &count);
    for (size_t k = 0; k < count; k++) {
      struct call *c = list + made[k].call;
      const struct mesh_step *step = c->bound->steps + made[k].step;
      const struct message m = {c, made[k].step, ch->end + made[k].first, made[k].pieces, step->first * step->piece.element_bytes, piece_bytes(step),
                                ch->slot - ch->slot % step->piece.element_bytes};
      if (made[k].send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
      c->bound->channel[m.step] = h; c->bound->first[m.step] = m.first; c->bound->pieces[m.step] = m.pieces;
    }
    ch->end += positions;
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct steps *st = list[i].bound;
    if (st && mesh_steps_before(st->steps, st->count, st->before)) result = fail(ncclSystemError, "out of memory");
    for (uint32_t k = 0; st && !result && k < st->count; k++)
      if (st->steps[k].op != MESH_STEP_SEND && !st->pieces[k]) st->done[k] = 1;
  }
  free(steps); free(counts); free(made); free(scratch);
  return result;
}

static void schedules_free(const struct session *s, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < s->count; h++) { free(plans[h].out); free(plans[h].in); }
  free(plans);
}

/* a step may run once the receive before it is consumed (mesh_steps_before: every earlier receive then is, by induction, as
   each receive's first piece waits on the one before it) */
static int ready(const struct steps *s, uint32_t k) { return s->before[k] < 0 || s->done[s->before[k]]; }

/* a host-path group with nothing to do until a landing: parked until the progress thread finds a channel's landings past
   what the group consumed, or the session fails (landed_tell), so a group waiting on a peer takes no core and holds no
   lock but its own */
static void park(struct session *s) {
  pthread_mutex_lock(&s->park);
  for (;;) {
    int ready = atomic_load(&s->failed) != 0;
    for (uint32_t h = 0; h < s->count && !ready; h++) ready = atomic_load(&s->channels[h].signaled) > s->channels[h].consumed;
    if (ready) break;
    pthread_cond_wait(&s->landed, &s->park);
  }
  pthread_mutex_unlock(&s->park);
}

/* every channel's positions published and consumed: a SEND's piece copied into its slot, a REDUCE's
   combined into the result, a COPY's copied into the operand (and the result, where they differ: a direct
   exchange's sums go to a copy, as its SENDs read the operand); the progress thread watches the landings */
static ncclResult_t run(struct session *ses, struct schedule *plans) {
  for (;;) {
    if (atomic_load(&ses->failed)) return fail((ncclResult_t)atomic_load(&ses->failed), "%s", ses->failure);
    int open = 0, moved = 0;
    for (uint32_t h = 0; h < ses->count; h++) {
      struct channel *ch = ses->channels + h;
      struct schedule *p = plans + h;
      while (ch->consumed < ch->end) {
        const uint64_t at = ch->consumed, arrived = mesh_host_arrived(&ses->context, ch->in, (uint32_t)at);
        if (arrived == UINT64_MAX) return cancelled(ses, ch->node);
        if (!landed(arrived, at)) break;
        struct message *m = p->in_at < p->ins ? p->in + p->in_at : NULL;
        const int data = m && at >= m->first;
        uint64_t seen;
        memcpy(&seen, ch->receiving + (at % DEPTH) * ch->pitch, sizeof seen);
        if (data == (seen == METAL_STAMP(at)))
          return fail(ncclInvalidUsage, MISMATCH, (unsigned long long)at, ch->node, data ? "a piece" : "nothing", data ? "nothing" : "a piece");
        if (at + 1 > ch->checked) ch->checked = at + 1;
        if (data) {
          struct steps *s = m->c->bound;
          if (at == m->first && !ready(s, m->step)) break;
          size_t length, offset = piece_at(m, at, &length);
          const unsigned char *from = ch->receiving + (at % DEPTH) * ch->pitch;
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
        struct message *m = p->out_at < p->outs ? p->out + p->out_at : NULL;
        if (m && at >= m->first) {
          if (at == m->first && !ready(m->c->bound, m->step)) break;
          size_t length, offset = piece_at(m, at, &length);
          if (counted(m->c) && m->c->bound->steps[m->step].buffer == 1) {
            size_t valid;
            const size_t place = counted_place(m->c, (int)m->c->bound->steps[m->step].peer, 0, offset, length, &valid);
            memcpy(ch->sending + (at % DEPTH) * ch->pitch, (const unsigned char *)m->c->send + place, valid);
          } else
            memcpy(ch->sending + (at % DEPTH) * ch->pitch, m->c->bound->host[m->c->bound->steps[m->step].buffer] + m->offset + offset, length);
          if (at + 1 == m->first + m->pieces) p->out_at++;
        } else {
          const uint64_t stamp = METAL_STAMP(at);
          memcpy(ch->sending + (at % DEPTH) * ch->pitch, &stamp, sizeof stamp);
        }
        mesh_host_publish(&ses->context, ch->out, at);
        ch->sent++;
        moved = 1;
      }
      open |= ch->sent < ch->end || ch->consumed < ch->end;
    }
    if (!open) return ncclSuccess;
    if (!moved) park(ses);
  }
}

/* the communicator's session, opened where it has none (after any other thread's opening of it is done), on the map
   current now, or closed where none of its groups is in flight and its bridge stopped; a failure its progress thread
   kept fails this group and closes the session, its slot and its lanes' lease with it (slot_end), so the caller's next
   group on the communicator fails and the communicator is made again (no group is tried again here).  A session runs
   on the map its members agreed on until it fails: a change of the map never closes it, so no member reopens at a group
   of its own and fails the group a peer had in flight on the old one.  A link or node its routes cross lost fails it on
   every member (the bridges cancel its flows, a member gone retires its program and its peers' flows toward it end),
   each member's next communicator made on the map then, which the agreement (session_agree) checks; a change that
   leaves its routes working (a link elsewhere, a node joining, a better route appearing) reaches the next communicator,
   and every call of metal-microbench makes its communicators at its first crossing (A28: the route resolved at each
   session or execution). */
static struct inflight *inflight;
static int session_issued(const struct session *s);
static ncclResult_t session_ensure(ncclComm_t comm) {
  while (comm->opening) pthread_cond_wait(&opened, &session_lock);
  struct session *s = comm->session;
  if (s) violated(s);
  if (s && atomic_load(&s->failed)) {
    const ncclResult_t result = fail((ncclResult_t)atomic_load(&s->failed), "%s", s->failure);
    session_close(s);
    return result;
  }
  if (s && !session_issued(s) && !attach_live(s->attach)) {
    session_close(s);
    s = NULL;
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

/* the group on the host, the progress thread watching its landings, the bridge's progress threads woken (mesh-call.h
   mesh_transfers_wake: they park after a while of nothing); its positions then count as landed */
static ncclResult_t session_run(struct session *s, struct call *list, size_t n) {
  struct schedule *plans;
  ncclResult_t result = session_schedule(s, list, n, &plans);
  if (result) return result;
  progress_targets(s, 0);
  mesh_transfers_wake(&s->context);
  atomic_fetch_add(&s->runs, 1);
  pthread_mutex_unlock(&session_lock);
  result = run(s, plans);
  pthread_mutex_lock(&session_lock);
  atomic_fetch_sub(&s->runs, 1);
  schedules_free(s, plans);
  if (result) { session_close(s); return result; }
  for (uint32_t h = 0; h < s->count; h++) {
    struct channel *ch = s->channels + h;
    atomic_store(&ch->target, ch->end); atomic_store(&ch->signaled, ch->end);
    ch->waited = ch->end;
  }
  return ncclSuccess;
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

/* one group on the host: the communicator's session ensured (its programs on the current map), its calls planned and
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
  if (revoked(comm)) return revoked_fail(comm);
  for (size_t i = 0; i < n; i++) {
    const struct call *c = list + i;
    lease_wait(c->send, 0);
    lease_wait(c->recv, 1);
    if (counted(c)) { lease_wait(c->counts, 0); lease_wait(c->received, 1); lease_wait(c->landed, 1); }
  }
  pthread_mutex_lock(&session_lock);
  if (remote) result = session_ensure(comm);
  if (!result && remote && session_issued(comm->session))
    result = fail(ncclInvalidUsage, "a host-path group while this communicator's GPU groups are issued and not complete (ncclMeshComplete)");
  if (!result && remote) drain(comm->session);
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

static int receives_at(struct session *s, uint32_t h, uint64_t at);

/* the positions of channel `ch` up to `count` not yet checked, each slot against its role once a wait covers it (its
   slot stays until this rank publishes the position LAG past it, which waits on the position after it first) */
static struct metal_check check_of(struct session *s, struct channel *ch, uint64_t count) {
  const uint32_t h = (uint32_t)(ch - s->channels);
  uint64_t first = ch->checked < count ? ch->checked : count;
  if (count - first > DEPTH) first = count - DEPTH;
  struct metal_check check = {ch->ring_in, s->violation_buffer, ch->pitch, first, (uint32_t)(count - first), 0, h, DEPTH, count ? count - 1 : 0};
  for (uint64_t t = first; t < count; t++)
    if (receives_at(s, h, t)) check.mask |= 1u << (t - first);
  if (count > ch->checked) ch->checked = count;
  return check;
}

/* the GPU waits until `count` of the channel's positions have landed: a spin on each one's completion word not yet
   awaited, in order, each position checked against its role as its wait covers it (a program over several cables
   lands its positions out of order, position t on cable t mod n: design/bridge-and-ipc.md#cables, so a later
   position's landing says nothing of an earlier one's) */
static ncclResult_t arrive(struct metal_program *program, struct session *s, struct channel *ch, uint64_t count) {
  ncclResult_t result = ncclSuccess;
  for (uint64_t at = ch->waited; at < count && !result; at++) {
    ch->waited = at + 1;
    const struct metal_check check = check_of(s, ch, at + 1);
    result = gpu(metal_spin(program, s->transport.inputs, ch->word + (at % POSITIONS) * ch->word_stride, cycle_of(at), &check), "a spin");
  }
  return result;
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
    free(list[i].segments);
    free(list[i].got);
    free(list[i].complement);
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
static int session_issued(const struct session *s) {
  for (const struct inflight *g = inflight; g; g = g->next_group)
    if (g->session == s) return 1;
  return 0;
}

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
    counted_failed(g->list, g->n);
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
  for (size_t i = 0; i < g->n; i++) {
    if (list[i].bound && list[i].bound->program && list[i].bound->program->first[N] + 1 > S) S = list[i].bound->program->first[N] + 1;
    if (list[i].bound && list[i].bound->count + 1 > S) S = list[i].bound->count + 1;
  }
  const size_t n = g->n, E = (size_t)N * (N - 1);
  struct mesh_step **steps = calloc((size_t)N * n + 1, sizeof *steps), *planned = calloc((size_t)N * n * S + 1, sizeof *planned);
  uint32_t *counts = calloc((size_t)N * n + 1, sizeof *counts), *lane = calloc((size_t)N * n * S + 1, sizeof *lane);
  uint64_t *first = calloc((size_t)N * n * S + 1, sizeof *first), *pieces = calloc((size_t)N * n * S + 1, sizeof *pieces);
  uint64_t *length = calloc(2 * E + 1, sizeof *length), *offset = calloc(E + 1, sizeof *offset);
  struct placing *made = calloc(n * S + 1, sizeof *made);
  struct mesh_place *scratch = calloc(S + 1, sizeof *scratch);
  int32_t *before = calloc((size_t)N * n * S + 1, sizeof *before);
  ncclResult_t result = steps && planned && counts && lane && first && pieces && length && offset && made && scratch && before
                          ? ncclSuccess : fail(ncclSystemError, "out of memory");
  /* a channel: the pair a < b (its index among the pairs) and the class; an end's side 0 for a, 1 for b */
#define CHANNEL(a, b, large) ((((size_t)(a) * (2 * N - (a) - 1) / 2 + ((b) - (a) - 1))) * 2 + (size_t)(large))
#define AT(r, i, k) (((size_t)(r) * n + (i)) * S + (k))
  for (int r = 0; r < N && !result; r++)
    for (size_t i = 0; i < n && !result; i++) {
      const struct steps *own = list[i].bound;
      if (!own) continue;
      if (r == me) { steps[r * n + i] = own->steps; counts[r * n + i] = own->count; }
      else if (!p2p(list + i) && !counted(list + i)) {
        steps[r * n + i] = planned + AT(r, i, 0);
        counts[r * n + i] = mesh_program_steps_cut(own->program, (uint32_t)r, own->operand, own->segments, steps[r * n + i]);
      }
    }
  for (int r = 0; r < N && !result; r++)
    for (int q = 0; q < N; q++)
      for (int large = 0; q != r && large < 2; large++) {
        size_t count;
        const size_t e = r < q ? CHANNEL(r, q, large) : CHANNEL(q, r, large);
        length[2 * e + (r > q)] = lay(s, list, n, steps + (size_t)r * n, counts + (size_t)r * n, q, large, scratch, made, &count);
        for (size_t m = 0; m < count; m++) {
          lane[AT(r, made[m].call, made[m].step)] = (uint32_t)e;
          first[AT(r, made[m].call, made[m].step)] = made[m].first;
          pieces[AT(r, made[m].call, made[m].step)] = made[m].pieces;
        }
      }
  for (int r = 0; r < N && !result; r++)
    for (size_t i = 0; i < n && !result; i++)
      if (counts[r * n + i] && mesh_steps_before(steps[r * n + i], counts[r * n + i], before + AT(r, i, 0))) result = fail(ncclSystemError, "out of memory");
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
        const int32_t j = before[AT(r, i, k)];
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
  free(steps); free(planned); free(counts); free(lane); free(first); free(pieces); free(length); free(offset); free(made); free(scratch); free(before);
  free(after_send); free(after_receive); free(sends); free(lands); free(pred); free(level); free(queue); free(degree); free(successors); free(edge);
  return result;
}

/* a counted all-to-all's end on the GPU, once every count landed (its counts receives consumed; at once where it has
   none): its own rows into place, the landed counts into `received` (stored system-coherent: the host reads them as they
   are, mesh 3981ca80^'s `received`) and `landed`, then its word released (1, or 2 where the counts pass its layout,
   which also fails the session through violation word [5]; nccl-mesh-metal.h METAL_COUNTED_ARRIVED), the rows maybe
   still on the wire; `violation` the session's words (NULL: none) */
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
   written in place of their dispatches, and whether each was; under session_lock, for one launch_metal */
static struct { ncclMeshFused *out; const struct call *call; int published, landed, publishes, lands; } fusing;

/* whether message m's piece at its channel's position (`offset`, `length`) is the fused call's whole operand */
static int fused_piece(const struct message *m, size_t offset, size_t length) {
  return fusing.out && m->c == fusing.call && !m->offset && !offset && length == m->bytes && m->bytes == elements(m->c) * SIZE[m->c->type];
}

/* channel h's positions consumed up to `end`, in order: a piece's landing awaited and combined or copied where it
   belongs (an empty position needs nothing), a receive's first piece after the receive before it; the fused call's
   landing of its whole operand recorded for its caller's kernel (metal_land_deferred) */
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
      size_t length, offset = piece_at(m, at, &length);
      const size_t from = (at % DEPTH) * ch->pitch;
      const int reduce = st->steps[m->step].op == MESH_STEP_REDUCE;
      const struct where to = st->gpu[st->steps[m->step].buffer];
      const int q = (int)st->steps[m->step].peer, n = m->c->comm->nranks;
      if (counted(m->c) && st->steps[m->step].buffer == 3) {
        const uint64_t *rseg = m->c->segments + n;
        result = arrive(program, s, ch, at + 1);
        if (!result)
          result = gpu(metal_counted(program, METAL_COUNTED_LAND, &(struct metal_counted){
            .d = to.buffer, .dst = to.offset, .s = ch->ring_in, .src = from, .bytes = length, .scalar = offset, .a = m->c->got_at.buffer,
            .aux = m->c->got_at.offset, .first = seg_first(rseg, q), .entries = (uint32_t)rseg[q], .row = row_bytes(m->c),
            .limit = m->c->capacity * row_bytes(m->c)}), "a counted landing");
      } else if (st->fresh && st->fresh[m->step]) {
        const struct where with = st->read[m->step];
        result = arrive(program, s, ch, at + 1);
        if (!result)
          result = gpu(metal_combine3(program, m->c->type, m->c->combine, to.buffer, to.offset + m->offset + offset, ch->ring_in, from, with.buffer,
                                      with.offset + m->offset + offset, length / SIZE[m->c->type]), "a first combine");
      } else if (!ch->large) {
        const struct metal_check check = check_of(s, ch, at + 1);
        if (fusing.lands && !fusing.landed && fused_piece(m, offset, length)) {
          result = gpu(metal_land_deferred(program, m->c->combine, to.buffer, to.offset, ch->ring_in, from,
                                           reduce ? length / SIZE[m->c->type] : length,
                                           s->transport.inputs, ch->word + (at % POSITIONS) * ch->word_stride, cycle_of(at), &check,
                                           fusing.out->land), "a fused landing's record");
          fusing.landed = !result;
        } else
          result = gpu(metal_land(program, reduce ? ktype(m->c) : -1, kop(m->c), to.buffer, to.offset + m->offset + offset, ch->ring_in, from,
                                  reduce ? kcount(m->c, length) : length, s->transport.inputs,
                                  ch->word + (at % POSITIONS) * ch->word_stride, cycle_of(at), &check), "a landing");
        if (at + 1 > ch->waited) ch->waited = at + 1;
      } else {
        result = arrive(program, s, ch, at + 1);
        if (!result && reduce)
          result = gpu(metal_combine(program, ktype(m->c), kop(m->c), to.buffer, to.offset + m->offset + offset,
                                     ch->ring_in, from, kcount(m->c, length)), "a combine");
        else if (!result)
          result = gpu(metal_copy(program, METAL_LAND, to.buffer, to.offset + m->offset + offset, ch->ring_in, from, length), "a landing");
      }
      if (result) break;
      if (at + 1 == m->first + m->pieces) {
        st->done[m->step] = 1; p->in_at++;
        if (counted(m->c) && st->steps[m->step].buffer == 2) result = counted_landed(program, m->c, q, s->violation_buffer);
      }
    }
    ch->consumed++;
  }
  return result;
}

/* channel h's next position published, after the landing of the peer's position LAG back, whose slot it fills: its
   piece copied into its slot and its cell released, or the cell alone (an empty position); the fused call's whole
   operand recorded for its caller's kernel, which stores it and releases the cell (metal_send_deferred) */
static ncclResult_t publish(struct metal_program *program, struct session *s, uint32_t h, int *published) {
  struct channel *ch = s->channels + h;
  const uint64_t at = ch->sent, need = at >= LAG ? at - LAG + 1 : 0;
  struct inflight *g = inflight_at(s, h, at);
  if (!g) return fail(ncclInternalError, "position %llu of channel %u is in no group", (unsigned long long)at, h);
  struct schedule *p = g->plans + h;
  struct message *m = p->out_at < p->outs && at >= p->out[p->out_at].first ? p->out + p->out_at : NULL;
  ncclResult_t result = arrive(program, s, ch, need);
  const size_t cell = ch->cell + (at % POSITIONS) * sizeof(struct mesh_send);
  const uint64_t value = ch->argument + cycle_of(at);
  int released = 0;
  if (m && !result) {
    const struct steps *st = m->c->bound;
    size_t length, offset = piece_at(m, at, &length);
    if (counted(m->c) && st->steps[m->step].buffer == 1) {
      const uint64_t *sseg = m->c->segments;
      const int q = (int)st->steps[m->step].peer;
      result = gpu(metal_counted(program, METAL_COUNTED_SEND, &(struct metal_counted){
        .d = ch->ring_out, .dst = (at % DEPTH) * ch->pitch, .s = m->c->send_at.buffer, .src = m->c->send_at.offset, .bytes = length,
        .scalar = offset, .a = m->c->counts_at.buffer, .aux = m->c->counts_at.offset, .first = seg_first(sseg, q), .entries = (uint32_t)sseg[q],
        .row = row_bytes(m->c), .limit = m->c->rows * row_bytes(m->c)}), "a counted piece into its slot");
    } else if (!ch->large && fusing.publishes && !fusing.published && fused_piece(m, offset, length)) {
      result = gpu(metal_send_deferred(program, ch->ring_out, (at % DEPTH) * ch->pitch, length, s->transport.publication, cell, value,
                                       fusing.out->publish), "a fused publication's record");
      fusing.published = !result;
      released = 1;
    } else if (!ch->large) {
      const struct where source = st->read && st->read[m->step].buffer ? st->read[m->step] : st->gpu[st->steps[m->step].buffer];
      result = gpu(metal_send_small(program, ch->ring_out, (at % DEPTH) * ch->pitch, source.buffer, source.offset + m->offset + offset,
                                    length, s->transport.publication, cell, value), "a piece into its slot and its release");
      released = 1;
    } else {
      const struct where source = st->read && st->read[m->step].buffer ? st->read[m->step] : st->gpu[st->steps[m->step].buffer];
      result = gpu(metal_copy(program, METAL_SEND, ch->ring_out, (at % DEPTH) * ch->pitch, source.buffer, source.offset + m->offset + offset,
                              length), "a piece into its slot");
    }
    if (at + 1 == m->first + m->pieces) p->out_at++;
  }
  if (!result && !released && m) result = gpu(metal_publish(program, s->transport.publication, cell, value), "a publication");
  else if (!result && !released)
    result = gpu(metal_publish_filler(program, s->transport.publication, cell, value, ch->ring_out, (at % DEPTH) * ch->pitch, at),
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
  s->touched = 1;
  metal_gate(program, s->violation_buffer);
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
  s->touched = 1;
  metal_gate(program, s->violation_buffer);
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
   no wait, so a group issued encodes no wait at all; each group's dispatches gated on its own session's violation
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
    for (uint32_t h = 0; h < g->session->count; h++)
      if (g->session->channels[h].consumed < g->end[h]) return ncclSuccess;
    metal_gate(program, g->session->violation_buffer);
    result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g);
  }
  return result;
}

/* a program's end: where the library made its encoder and the group succeeded, its end signals `finished` at its count
   (`issued`, which drain awaits: a program the caller was told to commit), each session it touched keeping that count
   as its `last` (session_close); a borrowed encoder's, or a failed group's, signals nothing, so no wait of the
   library's is on a command buffer the caller may never commit.  Under the sessions' lock */
static void program_end(struct metal_program *program, int borrowed, ncclResult_t result) {
  const uint64_t count = borrowed || result ? 0 : ++issued;
  metal_end(program, count ? (finished ? finished : (finished = metal_event())) : NULL, count);
  for (struct session *s = sessions; s; s = s->next)
    if (s->touched) { s->touched = 0; if (count) s->last = count; }
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

/* one group on the stream's command buffer (or, the Metal 4 form, its Metal 4 encoder: nccl-mesh-metal.h metal_begin4),
   taking its calls: each collective's operand recvbuff itself where it holds the whole operand (an all-reduce's, a
   root's reduce, a broadcast's, an all-gather's), else a scratch one (a reduce-scatter's, a non-root's reduce); a
   direct exchange's sums apart; then the group in flight after the others, issued (with issue) or completed with every
   group before it; its results into recvbuff as it completes; the program's end signals `finished` where the library
   made its encoder.  The Metal 4 form makes the session's rings, cells and words resident through the stream's
   residency set; without a workspace its scratch is the communicator's (workspace), so it is not issued. */
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
  pthread_mutex_lock(&session_lock);
  if (!result && remote) result = session_ensure(comm);
  if (!result) result = plan_calls(comm, list, n);
  if (!result && list[0].fused && (!remote || !list[0].bound ||
                                   list[0].bound->buffers != 1 + (list[0].what == MESH_ALLREDUCE && comm->nranks == 2)))
    result = fail(ncclInvalidUsage, "a fused crossing whose program copies its operand before publishing it (ncclMeshFusable)");
  if (result) { pthread_mutex_unlock(&session_lock); counted_failed(list, n); calls_free(list, n); return result; }
  if (finished) metal_collect(finished);
  struct session *const s = comm->session;
  void *const violation = remote ? s->violation_buffer : NULL;
  struct metal_program *program = metal4 ? metal_begin4(list[0].encoder, list[0].table, list[0].residency, violation, list[0].polls)
                                         : metal_begin(list[0].command, list[0].encoder, violation, list[0].polls);
  if (!program) {
    pthread_mutex_unlock(&session_lock); counted_failed(list, n); calls_free(list, n);
    return fail(ncclSystemError, "%s", *metal_error() ? metal_error() : "out of memory");
  }
  comm->scratch_used = 0;
  if (remote) s->touched = 1;
  for (uint32_t h = 0; metal4 && remote && h < s->count; h++) { metal_resident(program, s->channels[h].ring_out); metal_resident(program, s->channels[h].ring_in); }
  if (metal4 && remote) { metal_resident(program, s->transport.publication); metal_resident(program, s->transport.inputs); }
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
        progress_targets(s, 0);
        mesh_transfers_wake(&s->context);
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
  if (result && scheduled) walk_failed(s, issue ? 0 : ticket, result);
  if (issue && !result) { issue->ticket = ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  pthread_mutex_unlock(&session_lock);
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
  pthread_mutex_lock(&session_lock);
  const ncclResult_t loss = losses_through(ticket);
  ncclResult_t result = ncclSuccess;
  if (inflight && ticket >= inflight->ticket) {
    struct inflight *through = inflight;
    while (through->next_group && through->next_group->ticket <= ticket) through = through->next_group;
    if (finished) metal_collect(finished);
    struct metal_program *program = stream->argumentTable
      ? metal_begin4(stream->commandEncoder, stream->argumentTable, stream->residencySet, inflight->session->violation_buffer, stream->polls)
      : metal_begin(stream->commandBuffer, stream->commandEncoder, inflight->session->violation_buffer, stream->polls);
    if (!program) result = fail(ncclSystemError, "%s", *metal_error() ? metal_error() : "out of memory");
    else {
      mesh_transfers_wake(&inflight->session->context);
      result = walk(program, NULL, through, &any);
      if (!result) result = retire(program);
      program_end(program, stream->commandEncoder != NULL, result);
      if (result) walk_failed(NULL, ticket, result);
    }
  }
  pthread_mutex_unlock(&session_lock);
  if (published) *published = any;
  return result ? result : loss;
}

uint64_t ncclMeshRetired(void) { return atomic_load(&retired); }

void ncclMeshMetal4Records(uint64_t *issued, uint64_t *capacity) { metal_records(issued, capacity); }

void ncclMeshSessionWaits(uint64_t *ns) { *ns = atomic_load(&session_waits); }

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
  pthread_mutex_lock(&session_lock);
  /* the answer is the communicator's program, its table's (the caller's, or the structural one comm_routes compiles on
     the bridges' map, read here where no session has read it yet: a call records its crossings before its first group
     opens the session) */
  if (!comm->programs && comm_read(comm) != ncclSuccess) { pthread_mutex_unlock(&session_lock); return ncclSuccess; }
  if (!comm->wire.small) {
    /* the wire the session will take (session_open): its bridge's block and NCCL_BUFFSIZE */
    pthread_mutex_lock(&attach_lock);
    struct attach *a = NULL;
    const int attached = attach_current(&a);
    if (!attached) mesh_wire_of((uint64_t)a->context.M->pgsz * a->context.M->block, setting("NCCL_BUFFSIZE", MESH_BUFFSIZE), comm->small, &comm->wire);
    pthread_mutex_unlock(&attach_lock);
    if (attached) { pthread_mutex_unlock(&session_lock); return ncclSuccess; }
  }
  const struct mesh_program *program = program_of(comm, what, root, bytes);
  /* a two-rank all-reduce's one scratch is its copy of the operand, which its program sends from while the sum is made
     in place; its fused steps send from the producer's own store and land into the consumer's, so neither reads it; a
     configured form that cuts the operand (two-phase, chunked) moves it in several pieces and is not fusable */
  const int fusable = program && program->scratch <= (what == MESH_ALLREDUCE ? 1u : 0u) && bytes <= comm->wire.small &&
                     !mesh_wire_large(&comm->wire, bytes, SIZE[type]) && whole_moves(program, (uint32_t)comm->rank);
  pthread_mutex_unlock(&session_lock);
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

/* nccl.h ncclMeshAllReducePartials, ncclMeshReducePartials: the engine's piece laid as the merge's rows into a scratch
   (partials_layout; the Metal path's partials_rows), merged there by the communicator's all-reduce (root -1) or reduce
   to root with the partials merge (ncclMeshRedOpCreatePartials), then laid back into recvbuff (partials_columns), the
   group completed in the stream between the two layouts, so not inside ncclGroupStart/End.  The Metal path's scratch is
   the stream's workspace, else the communicator's own (`merge`, grown, the one it replaces kept until its end), each
   layout a program of the library's on the stream under session_lock */
static ncclResult_t layout_on(const ncclMeshStream *given, int to_rows, struct where dst, struct where src, size_t rows, size_t dim) {
  if (!given) {
    partials_layout(to_rows, (float *)((char *)dst.buffer + dst.offset), (const float *)((const char *)src.buffer + src.offset), rows, dim);
    return ncclSuccess;
  }
  pthread_mutex_lock(&session_lock);
  struct metal_program *program = given->argumentTable ? metal_begin4(given->commandEncoder, given->argumentTable, given->residencySet, NULL, given->polls)
                                                       : metal_begin(given->commandBuffer, given->commandEncoder, NULL, given->polls);
  const ncclResult_t result = !program ? fail(ncclSystemError, "%s", *metal_error() ? metal_error() : "out of memory")
                                       : gpu(metal_partials(program, to_rows, dst.buffer, dst.offset, src.buffer, src.offset, rows, dim), "the partials' rows");
  if (program) program_end(program, given->commandEncoder != NULL, result);
  pthread_mutex_unlock(&session_lock);
  return result;
}
static ncclResult_t partials(const void *sendbuff, void *recvbuff, size_t rows, size_t dim, int root, ncclComm_t comm, cudaStream_t stream) {
  if (!comm || !sendbuff || !recvbuff) return fail(ncclInvalidArgument, "a partials merge's comm, sendbuff or recvbuff is NULL");
  if (root < -1 || root >= comm->nranks) return fail(ncclInvalidArgument, "root %d of %d ranks", root, comm->nranks);
  if (depth) return fail(ncclInvalidUsage, "a partials merge of the engine's piece lays its rows back after its group: not inside a group");
  const ncclMeshStream *given = stream;
  if (given && given->fused) return fail(ncclInvalidUsage, "a partials merge is no fused crossing");
  if (!rows) return ncclSuccess;
  const size_t n = rows * (2 + dim), bytes = n * sizeof(float);
  void *scratch = given ? NULL : malloc(bytes);
  pthread_mutex_lock(&session_lock);
  if (given && given->workspace) scratch = given->workspace(bytes, given->context);
  else if (given && comm->merge_bytes < bytes) {
    void **spent = comm->merge ? realloc(comm->spent, (comm->spents + 1) * sizeof *spent) : comm->spent;
    void *made = (comm->merge && !spent) ? NULL : metal_scratch(bytes);
    if (made && comm->merge) { comm->spent = spent; comm->spent[comm->spents++] = comm->merge; }
    if (made) { comm->merge = made; comm->merge_bytes = bytes; }
  }
  if (given && !given->workspace && comm->merge_bytes >= bytes) scratch = comm->merge;
  pthread_mutex_unlock(&session_lock);
  ncclRedOp_t op;
  ncclResult_t result = scratch ? ncclMeshRedOpCreatePartials(&op, dim, ncclFloat32, comm) : fail(ncclSystemError, "out of memory");
  ncclMeshStream inner = given ? *given : (ncclMeshStream){0};
  inner.issue = NULL;
  ncclMeshBuffer merged = {scratch, 0};
  void *x = given ? (void *)&merged : scratch;
  const struct where at = {scratch, 0};
  if (!result) {
    result = layout_on(given, 1, at, given ? where_of(sendbuff) : (struct where){(void *)(uintptr_t)sendbuff, 0}, rows, dim);
    if (!result) result = root < 0 ? ncclAllReduce(x, x, n, ncclFloat32, op, comm, given ? &inner : NULL)
                                   : ncclReduce(x, x, n, ncclFloat32, op, root, comm, given ? &inner : NULL);
    if (!result) result = layout_on(given, 0, given ? where_of(recvbuff) : (struct where){recvbuff, 0}, at, rows, dim);
    if (!result && given && given->issue) *given->issue = (ncclMeshIssue){0, 1};
    ncclRedOpDestroy(op, comm);
  }
  if (!given) free(scratch);
  return result;
}
ncclResult_t ncclMeshAllReducePartials(const void *sendbuff, void *recvbuff, size_t rows, size_t dim, ncclComm_t comm, cudaStream_t stream) {
  return partials(sendbuff, recvbuff, rows, dim, -1, comm, stream);
}
ncclResult_t ncclMeshReducePartials(const void *sendbuff, void *recvbuff, size_t rows, size_t dim, int root, ncclComm_t comm, cudaStream_t stream) {
  return partials(sendbuff, recvbuff, rows, dim, root < 0 ? -2 : root, comm, stream);
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

/* a communicator over comm's ranks `ranks` (count of them, in that order), on the bridges' map read at each of its
   sessions, named after comm and its `split`-th split or shrink and `color`, the same on every member, its session's
   slot taken (comm_make) and its lanes leased by its caller (ncclCommSplit) */
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
  ncclResult_t result = making ? comm_fail(making, name) : ncclSuccess;
  if (making && *made) { comm_free(*made); *made = NULL; }
  if (!result) (*made)->small = comm->small;
  if (!result && comm->form_count && !((*made)->forms = forms_copy(comm->forms, comm->form_count, (uint32_t)comm->nranks))) {
    comm_free(*made); *made = NULL;
    result = fail(ncclSystemError, "out of memory");
  } else if (!result) {
    (*made)->form_count = comm->form_count;
    comm_advise(*made);
  }
  free(members);
  return result;
}

/* ncclCommSplit: every rank's (color, key) and its report (the lane pairs free at its bridge, mesh_lanes_free; none for
   NCCL_SPLIT_NOCOLOR) all-gathered on comm, the ranks of this color ordered by key, then rank; the child's lanes
   chosen on its ranks' reports and each rank's row claimed at its bridge (lease_claim), and a second all-gather on comm
   confirming every claim: NCCL's bootstrapSplit (the child's exchange over its parent's), with DESIGN I2's two
   gathers.  A failure at any rank of a color fails that color's split on each of its ranks as a value, its claims
   released (comm_free) */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  if (depth) return fail(ncclInvalidUsage, "ncclCommSplit inside a group");
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; struct report report; } mine = {color, key, {comm->rank, 0, {0}}}, *every = calloc((size_t)n, sizeof *every);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  int32_t claim = 0, *claims = calloc((size_t)n, sizeof *claims);
  struct mesh_lane_report *lanes = calloc((size_t)n, sizeof *lanes);
  uint32_t *chosen = calloc((size_t)n * MESH_PORTS, sizeof *chosen);
  *newcomm = NULL;
  if (color != NCCL_SPLIT_NOCOLOR) mine.report.status = mesh_lanes_free(comm->attach->context.M, &mine.report.lanes);
  ncclResult_t result = every && ranks && claims && lanes && chosen ? ncclAllGather(&mine, every, sizeof mine, ncclUint8, comm, NULL)
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
  for (int i = 0; !result && i < count; i++) lanes[i] = every[ranks[i]].report.lanes;
  if (!result && count > 1) claim = lease_claim(*newcomm, lanes, chosen);
  else if (result && count) claim = ECANCELED;
  if (gathered) {
    const ncclResult_t confirmed = ncclAllGather(&claim, claims, 1, ncclInt32, comm, NULL);
    if (!result) result = confirmed;
  }
  for (int i = 0; !result && i < count; i++)
    if (claims[ranks[i]]) result = lease_fail((*newcomm)->name, i, claims[ranks[i]]);
  if (!result && count > 1) lease_keep(*newcomm, lanes, chosen);
  if (result && *newcomm) { comm_free(*newcomm); *newcomm = NULL; }
  free(every); free(ranks); free(claims); free(lanes); free(chosen);
  return result;
}

/* ncclCommShrink: comm's ranks but the excluded ones, in rank order, with no exchange among them (a departed member
   answers none), so leasing no lanes: a shrink to two ranks or more, whose session would run on none, is refused
   before its slot is taken (ENOLCK); with NCCL_SHRINK_ABORT, comm revoked first (ncclCommRevoke) */
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
                       "an exchange leases a communicator's lanes (an init's bootstrap, ncclCommSplit over its parent)", comm->name, count)
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
   reporting its node; the communicator, named by the id, reads the bridges' map at each session as every one does */
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

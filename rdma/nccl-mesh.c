#include "mesh-write-compiler.h"
/* design/whole-function.md#fabric */
#include "nccl.h"
#include "mesh-call.h"
#include "mesh-plan.h"
#include "mesh.h"
#include "mesh-span-plan.h"
#include "mesh-write-routes.h"
#include "nccl-mesh-metal.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
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
#include <dispatch/dispatch.h>
#include <fcntl.h>


enum { WHAT_COUNTED = ncclMeshCallAlltoAllV };
_Static_assert(ncclMeshCallAllReduce == MESH_ALLREDUCE && ncclMeshCallBroadcast == MESH_BROADCAST && ncclMeshCallReduce == MESH_REDUCE &&
               ncclMeshCallReduceScatter == MESH_REDUCE_SCATTER && ncclMeshCallAllGather == MESH_ALLGATHER,
               "nccl.h's declared kinds are the planner's collectives");
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN, COMBINE_MERGE };

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
/* a failure's message: its text, and `id`, where the message is kept with its failure (a communicator's, comm_fail; a
   non-blocking init's, init_run), the count of messages the process had kept when it was (`messages`), else 0 */
struct said { uint64_t id; char text[MESH_NCCL_DIAGNOSTIC_BYTES]; };
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

/* -- communicators -------------------------------------------------------------------------------- */

/* an operator made on the communicator: PreMulSum's scalar (`value`, or where it lies, `at`, read at each host-path
   call: ncclScalarDevice), or the partials merge's row of `unit` elements */
struct redop { int used; ncclDataType_t type; unsigned char value[8]; uint32_t unit; const void *at; };
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; struct mesh_trees trees; int whole; };

/* a channel (design/transport.md section 14): toward rank `peer` on node `node`, the holding of the stream toward that
   node its process holds (mesh-call.h mesh_stream_view: its two lists and its counts), and the Metal buffer over the
   page that holds the counts (`buffer`: `ready` and `landed` at those offsets in it; `ready_at` the region
   offset of `ready`, a landing's gate).  Its entries, each list in its own order: `sends` and `lands` laid by the
   groups scheduled, `out` and `in` their pages (the value each count reaches once they are whole), `published` send
   entries published, `consumed` the AOT consumer prefix (not the physical landing prefix),
   `waited` consumer entries awaited by the GPU work encoded */
struct channel {
  int peer;
  uint32_t node, stream, cable;
  const uint64_t *wire;
  struct mesh_stream_view view;
  void *buffer, *lists;
  size_t ready, landed;
  uint64_t ready_at, sends, lands, published, consumed, waited;
  mesh_pages out, in;
};
/* a generation of a group definition's storage (bind_slot), `held` by one or free; its last group in flight (`ticket`)
   and each channel's landing ends once that group was presented, which its next user awaits (guard) */
struct slot { void *address; size_t bytes; uint64_t ticket, *lands; mesh_pages *in; int held; };

/* design/whole-function.md#interface */
struct attach;
struct plan;
struct ncclComm {
  struct mesh_link_map map;
  struct mesh_fabric fabric;
  uint32_t *endpoint;
  struct mesh_paths *writes;
  unsigned char *write_overrides;
  uint64_t page;
  struct mesh_trees paths, trees, rooted;
  struct entry *entries, *given;
  ncclMeshForm *forms;
  uint32_t programs, splits, *members, form_count, count, nops;
  int nranks, rank, listed, compiled, binding, touched;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  struct redop *ops;
  ncclResult_t async, failed;
  pthread_mutex_t guard, prepare;
  int revoked, *peers, initing, listener;
  struct mesh_deadline until;
  pthread_t init;
  struct said why, failure;
  struct attach *attach;
  uint32_t held;
  struct channel *channels;
  uint32_t slot_count;
  struct slot *slots;
  void *gate, *gate_word;
  uint64_t gate_at, gates;
  _Atomic uint64_t *words;
  void *violation;
  uint64_t last, groups;
  void *command, *queue; /* the last GPU program's command buffer; the queue of its host groups' (launch_host) */
  struct ncclMeshWritePlan *compiled_writes; /* the plans holding definitions of it (write_definitions) */
  struct ncclComm *next_touched, *next_live;
  struct plan *plan;
  _Atomic int runs;
};

/* the groups' one lock: a group's planning, scheduling and encoding, a completion, a communicator's failure kept and its
   end; `finished`, the event each GPU program the library made the encoder of signals at its count (`issued`) */
static pthread_mutex_t group_lock = PTHREAD_MUTEX_INITIALIZER;
static void *finished;
static uint64_t issued;
/* the communicators realized and not freed (fabric_realize, comm_free), for the expiry's stop */
static pthread_mutex_t live_lock = PTHREAD_MUTEX_INITIALIZER;
static struct ncclComm *live;

/* design/transport.md section 5: the program's end at an expiry, its GPU work stopped first: every communicator's GPU
   waits ended through their failure words' claim (nccl-mesh-kernels.h mesh_step_failed, read every 2^10 loads, the
   tracks' too) and its stop (words[6]), and then the library's GPU programs awaited within MESH_PLATFORM_NS; the wait
   named (mesh.h mesh_expired) */
static void library_stop(void) {
  pthread_mutex_lock(&live_lock);
  for (struct ncclComm *c = live; c; c = c->next_live) {
    atomic_store_explicit(c->words + 6, 1, memory_order_release);
    atomic_store_explicit(c->words + 1, 1, memory_order_release);
  }
  pthread_mutex_unlock(&live_lock);
  const uint64_t until = mesh_clock_ns() + MESH_PLATFORM_NS, last = __atomic_load_n(&issued, __ATOMIC_ACQUIRE);
  void *const event = __atomic_load_n(&finished, __ATOMIC_ACQUIRE);
  while (mesh_clock_ns() < until && event && metal_signaled(event) < last) usleep(1000);
}
void mesh_library_expired(const char *wait, uint64_t ns) {
  library_stop();
  mesh_expired(wait, ns);
}
/* a call no declaration covers: a warning on stderr, and the program ends, its GPU work stopped first as at an expiry
   (design/eager-execution.md#what-follows) */
__attribute__((noreturn)) static void library_end(const char *why) {
  library_stop();
  char text[MESH_DIAGNOSTIC_BYTES];
  const int n = snprintf(text, sizeof text, "libnccl-mesh: warning: %s; the program ends (design/eager-execution.md)\n", why);
  if (n > 0) (void)!write(2, text, (size_t)n < sizeof text ? (size_t)n : sizeof text - 1);
  _exit(1);
}
/* the nanoseconds left before d's deadline; past it, the library's expiry (mesh_library_expired) */
static uint64_t deadline_left(const struct mesh_deadline *d) {
  const uint64_t now = mesh_clock_ns();
  if (now >= d->at) mesh_library_expired(d->wait, d->ns);
  return d->at - now;
}
/* the GPU wait whose bound a communicator's failure words record (METAL_BOUND, nccl-mesh-kernels.h mesh_step_check),
   named into `wait`; its deadline, 0 where they record none */
static uint64_t bound_reached(const struct ncclComm *comm, char *wait, size_t room) {
  const _Atomic uint64_t *v = comm->words;
  const uint64_t at = v ? atomic_load_explicit(v + 2, memory_order_acquire) : 0;
  if (!at || atomic_load_explicit(v, memory_order_relaxed) != METAL_BOUND) return 0;
  const uint64_t h = atomic_load_explicit(v + 3, memory_order_relaxed), polls = atomic_load_explicit(v + 4, memory_order_relaxed);
  snprintf(wait, room, "libnccl-mesh: communicator %s's GPU wait for entry %llu on the channel to node %u, through %llu polls (its stream's "
           "bound, ncclMeshStream polls: the wait's deadline, fixed when it was made)",
           comm->name, (unsigned long long)(at - 1), h < comm->count ? comm->channels[h].node : UINT32_MAX, (unsigned long long)polls);
  return MESH_PLATFORM_NS;
}
/* the device's GPU load period (nccl-mesh-metal.m metal_poll_seconds), read under group_lock, as its kernels are */
double mesh_poll_seconds(void *device) {
  pthread_mutex_lock(&group_lock);
  const double seconds = metal_poll_seconds(device);
  pthread_mutex_unlock(&group_lock);
  return seconds;
}

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

static void cache_drop(struct ncclComm *comm);
/* the trees and the table compiled on the map, and the WRITE plans' definitions compiled on them (cache_drop); with `all`,
   the map, the caller's paths and programs too */
static void routes_free(struct ncclComm *comm, int all) {
  cache_drop(comm);
  trees_free(&comm->trees);
  comm->rooted = (struct mesh_trees){0};
  comm->compiled = 0;
  for (uint32_t e = 0; e < comm->programs; e++) {
    program_free(&comm->entries[e].program); trees_free(&comm->entries[e].trees);
  }
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
  if (comm->listener >= 0) close(comm->listener);
  comm->listener = -1;
  pthread_mutex_unlock(&comm->guard);
}

static void comm_leave(struct ncclComm *comm);
static void plan_free(struct ncclComm *comm, struct plan *plan);
static void paths_free(struct mesh_paths *paths) {
  free((void *)paths->first); free((void *)paths->cable); free((void *)paths->weight);
  *paths = (struct mesh_paths){0};
}
/* comm's tables, its slots and its gate returned to the region (ncclMemFree), after its holdings are given back
   (comm_leave: the bridge posts nothing more of a given holding, so only rows a unit already posted names stay held,
   mesh-dataflow.c mesh_pinned) */
static void comm_storage_free(struct ncclComm *comm) {
  for (uint32_t k = 0; k < comm->slot_count; k++) { ncclMemFree(comm->slots[k].address); free(comm->slots[k].lands); free(comm->slots[k].in); }
  if (comm->gate_word) ncclMemFree(comm->gate_word);
  free(comm->slots);
  comm->slots = NULL; comm->slot_count = 0;
  comm->gate_word = comm->gate = NULL;
}
struct ncclMeshWritePlan;
static void write_definitions_drop(struct ncclComm *comm, struct ncclMeshWritePlan *plan);
static void comm_free(struct ncclComm *comm) {
  pthread_mutex_lock(&live_lock);
  for (struct ncclComm **at = &live; *at; at = &(*at)->next_live) if (*at == comm) { *at = comm->next_live; break; }
  pthread_mutex_unlock(&live_lock);
  bootstrap_close(comm);
  routes_free(comm, 1);
  for (uint32_t h = 0; h < comm->count; h++) { metal_release(comm->channels[h].buffer); metal_release(comm->channels[h].lists); }
  ncclMemFree((void *)comm->words);
  metal_release(comm->command); metal_release(comm->queue);
  plan_free(comm, comm->plan);
  comm_leave(comm);
  free(comm->channels); free(comm->endpoint); free((void *)comm->fabric.cable);
  for (int p = 0; comm->writes && p < comm->nranks * comm->nranks; p++) paths_free(&comm->writes[p]);
  free(comm->writes); free(comm->write_overrides);
  free(comm->ops);
  pthread_mutex_destroy(&comm->guard);
  pthread_mutex_destroy(&comm->prepare);
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

/* `what` (at `root`) along `trees` (mesh-plan.h mesh_compile: each step's piece one entry, which the bridge cuts into
   SENDs), an all-reduce single-phase where `whole`, compiled into the table's next entry, for calls of fewer bytes than
   `below` (0: any size); 0, or EINVAL */
static int compile_entry(struct ncclComm *comm, const struct mesh_trees *trees, uint32_t what, uint32_t root, int whole, uint64_t below) {
  uint32_t parts, moves;
  mesh_compile_room(trees, &parts, &moves);
  uint32_t *segment = malloc((parts + 1) * sizeof *segment), *first = malloc((trees->nodes + 1) * sizeof *first);
  double *weight = malloc((parts + 1) * sizeof *weight);
  struct mesh_move *move = malloc((moves + 1) * sizeof *move);
  struct entry *e = comm->entries + comm->programs;
  if (!segment || !first || !weight || !move || trees_copy(&e->trees, trees) ||
      mesh_compile(trees, what, root, whole, NULL, &e->program, segment, weight, first, move)) {
    free(segment); free(first); free(weight); free(move);
    trees_free(&e->trees);
    return EINVAL;
  }
  e->below = below; e->whole = whole;
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
   cut among a root's trees by the form's shares, else by the packing's flows; a tree of share 0 carries none of it; the
   communicator's trees cut alike where the form gives shares), whole or two-phase, below its bound; a rooted one at
   its root, or at each root (-1), none where its root is past the communicator's ranks (a split's).  A root every one
   of whose trees takes share 0 has no tree to carry its segment: the form fails there as a value */
static ncclResult_t compile_form(struct ncclComm *comm, const struct mesh_trees *along, const ncclMeshForm *x) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_trees packed = {n, 0, NULL, NULL, NULL}, weighed = {n, 0, NULL, NULL, NULL};
  const struct mesh_trees *trees = along;
  ncclResult_t result = ncclSuccess;
  /* the communicator's trees cut by the form's shares: root r's k-th tree in their order shares[r x most + k], a tree
     past `most` none */
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
    else if (!(packed.count = mesh_trees_pack(&comm->map, NULL, x->eps, x->most, root, parent, weight)))
      result = fail(ncclInvalidArgument, "the map's links pack no trees (mesh_trees_pack)");
    for (uint32_t t = 0, k = 0; x->shares && t < packed.count; t++) {
      k = t && root[t] == root[t - 1] ? k + 1 : 0;
      weight[t] = log(x->shares[(size_t)root[t] * x->most + k]);
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
    if (!result && compile_entry(comm, trees, (uint32_t)x->collective, root, x->whole, x->below))
      result = fail(ncclInternalError, "the configured form of collective %d at root %u does not compile", x->collective, root);
  }
  trees_free(&packed);
  trees_free(&weighed);
  return result;
}

/* the communicator's table on its map: its configured forms first, in their order (compile_form), then the caller's
   programs where it listed some (ncclMeshCommInitRank), else, along the caller's paths (checked against the map) or
   else the map's links packed (mesh_trees_pack: every member's trees at once for the all-reduce, the reduce-scatter and
   the all-gather, each root's alone for its reduce and broadcast, every link in use and each tree's part its flow's
   share, so a part crosses every route and the routes end together [Garg & Konemann 1998; Blink, Wang et al. 2020]),
   the all-reduce two-phase (the reduce-scatter and then the all-gather over every pair [Patarasuk & Yuan 2009]), the
   reduce-scatter and the all-gather (segment s up and down s's trees), and each root's reduce and broadcast, each
   tree's part one entry, a relay sending on each page as it lands (present_group, forwards), of any size (mesh_compile;
   mesh 2bf5a50d^ comm_routes's table, its entries along the routes): a
   call takes the first entry that fits it (program_of), so a collective no form names, or a size past every form's
   bound, runs the caller's program or the default.  Where trees are compiled (a form, or no programs listed), a map
   whose links do not join its members fails it, naming two: a route crosses ranks alone, so a node a route is to cross
   takes part as a rank, and in this table, compiled with no contributors (compile_entry), it combines its own operand
   into what it sends on; the identity at a rank with no work (design/data-model.md section 9) is only a caller's
   programs' (mesh-plan.h mesh_compile's contributors) */
static ncclResult_t comm_routes(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks;
  if (n < 2) return ncclSuccess;
  const struct mesh_trees *along = &comm->paths, *rooted = &comm->paths;
  if (!comm->listed || comm->form_count) {
    uint32_t apart[2] = {0, 0};
    const int joined = mesh_link_map_joined(&comm->map, apart);
    if (joined == ENOTCONN)
      return fail(ncclInvalidUsage, "rank %u (node %u) and rank %u (node %u) of %s are joined by no path of links between its members: a "
                  "route crosses ranks alone, so a node it is to cross takes part as a rank", apart[0], comm->members[apart[0]],
                  apart[1], comm->members[apart[1]], comm->name);
    if (joined) return fail(ncclInternalError, "the members' map of %s: %s", comm->name, strerror(joined));
    if (comm->paths.count && mesh_trees_check(&comm->map, &comm->paths))
      return fail(ncclInvalidArgument, "the paths are not trees of the links of the members' map (mesh_trees_check)");
    if (!comm->paths.count) {
      uint32_t *root = calloc((size_t)n * n * 2 + 1, sizeof *root), *parent = calloc((size_t)n * n * n * 2 + 1, sizeof *parent);
      double *weight = calloc((size_t)n * n * 2 + 1, sizeof *weight);
      uint64_t *only = calloc(n / 64 + 1, sizeof *only);
      comm->trees = (struct mesh_trees){n, 0, root, parent, weight};
      comm->rooted = (struct mesh_trees){n, 0, root + (size_t)n * n, parent + (size_t)n * n * n, weight + (size_t)n * n};
      if (!root || !parent || !weight || !only) { free(only); return fail(ncclSystemError, "out of memory"); }
      comm->trees.count = mesh_trees_pack(&comm->map, NULL, 0, n, root, parent, weight);
      for (uint32_t r = 0; r < n; only[r / 64] = 0, r++) {
        only[r / 64] = 1ull << r % 64;
        const struct mesh_trees *at = &comm->rooted;
        comm->rooted.count += mesh_trees_pack(&comm->map, only, 0, n, (uint32_t *)at->root + at->count, (uint32_t *)at->parent + (size_t)at->count * n,
                                              (double *)at->log_weight + at->count);
      }
      free(only);
      along = &comm->trees;
      rooted = &comm->rooted;
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
  if (compile_entry(comm, along, MESH_ALLREDUCE, 0, 0, 0) || compile_entry(comm, along, MESH_REDUCE_SCATTER, 0, 0, 0) ||
      compile_entry(comm, along, MESH_ALLGATHER, 0, 0, 0))
    return fail(ncclInternalError, "the all-reduce, reduce-scatter or all-gather along the communicator's trees does not compile");
  for (uint32_t r = 0; r < n; r++)
    if (compile_entry(comm, rooted, MESH_REDUCE, r, 0, 0) || compile_entry(comm, rooted, MESH_BROADCAST, r, 0, 0))
      return fail(ncclInternalError, "the reduce or broadcast of root %u along the communicator's trees does not compile", r);
  return ncclSuccess;
}

/* The communicator's logical programs name rank endpoints. Physical lowering
   expands every transaction over the configured cables and relay paths. */
static ncclResult_t comm_compile(struct ncclComm *comm) {
  if (comm->compiled) return ncclSuccess;
  ncclResult_t result = comm_routes(comm);
  for (uint32_t e = 0; !result && e < comm->programs; e++) {
    const struct mesh_program *x = &comm->entries[e].program;
    for (uint32_t r = 0; !result && r < x->nodes; r++)
      for (uint32_t m = x->first[r]; !result && m < x->first[r + 1]; m++)
        if (x->move[m].peer >= (uint32_t)comm->nranks || x->move[m].peer == r)
          result = fail(ncclInvalidUsage, "program %u of %s names an invalid transaction endpoint %u from rank %u", e, comm->name,
                        x->move[m].peer, r);
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
   compiled before, by an earlier group or ncclMeshCommPlan, dropped), ahead of the caller's programs where it
   listed some: its members compile its table alike from them, so every member sets them at the same point between its
   groups (a group in flight keeps its own steps, never the table) */
ncclResult_t ncclMeshCommForms(ncclComm_t comm, const ncclMeshForm *forms, int count) {
  if (!comm || count < 0 || (count && !forms)) return fail(ncclInvalidArgument, "ncclMeshCommForms's arguments");
  for (int f = 0; f < count; f++) {
    const ncclMeshForm *x = forms + f;
    int shared = !x->shares || x->most;
    for (size_t k = 0; x->shares && shared && k < (size_t)comm->nranks * x->most; k++) shared = x->shares[k] >= 0 && isfinite(x->shares[k]);
    if (x->collective < MESH_ALLREDUCE || x->collective > MESH_ALLGATHER || (has_root(x->collective) && x->root < -1) || !shared ||
        (x->trees != ncclMeshTreesOwn && x->trees != ncclMeshTreesPacked) || (x->trees == ncclMeshTreesPacked && (!x->most || !(x->eps > 0))))
      return fail(ncclInvalidArgument, "form %d: a collective (mesh-plan.h MESH_*), a rooted one's root (-1 every root), its trees "
                  "(shortest, or packed with eps > 0 and most > 0), shares of 0 or more (most of them a root) or none", f);
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

/* design/whole-function.md#interface */
ncclResult_t ncclMeshCommFabric(ncclComm_t comm, const struct mesh_fabric **fabric) {
  if (!comm || !fabric) return fail(ncclInvalidArgument, "ncclMeshCommFabric's arguments");
  *fabric = &comm->fabric;
  return ncclSuccess;
}

ncclResult_t ncclMeshCommWriteRoutes(ncclComm_t comm, int source, int destination, const struct mesh_paths *paths) {
  if (!comm || source < 0 || destination < 0 || mesh_paths_check(&comm->fabric, (uint32_t)source, (uint32_t)destination, paths))
    return fail(ncclInvalidArgument, "one-way write paths must connect their endpoints over distinct nodes");
  const uint32_t P = paths->count, E = paths->first[P];
  uint32_t *first = malloc((P + 1) * sizeof *first), *cable = malloc(E * sizeof *cable);
  uint64_t *weights = malloc(P * sizeof *weights);
  if (!first || !cable || !weights) { free(first); free(cable); free(weights); return fail(ncclSystemError, "out of memory"); }
  memcpy(first, paths->first, (P + 1) * sizeof *first); memcpy(cable, paths->cable, E * sizeof *cable);
  memcpy(weights, paths->weight, P * sizeof *weights);
  pthread_mutex_lock(&group_lock);
  if (!comm->writes) comm->writes = calloc((size_t)comm->nranks * comm->nranks, sizeof *comm->writes);
  if (!comm->writes) {
    pthread_mutex_unlock(&group_lock); free(first); free(cable); free(weights);
    return fail(ncclSystemError, "out of memory");
  }
  struct mesh_paths *to = comm->writes + source * comm->nranks + destination;
  paths_free(to); *to = (struct mesh_paths){P, first, cable, weights};
  comm->write_overrides[source * comm->nranks + destination] = 1;
  plan_free(comm, comm->plan); comm->plan = NULL;
  cache_drop(comm);
  pthread_mutex_unlock(&group_lock);
  return ncclSuccess;
}

ncclResult_t ncclMeshCommWritePaths(ncclComm_t comm, int source, int destination, const struct mesh_paths **paths) {
  if (!comm || !paths || source < 0 || destination < 0 || source >= comm->nranks || destination >= comm->nranks || source == destination)
    return fail(ncclInvalidArgument, "route inspection requires distinct endpoints of the physical communicator");
  *paths = comm->writes ? comm->writes + (size_t)source * comm->nranks + destination : NULL;
  return *paths ? ncclSuccess : fail(ncclInvalidUsage, "the communicator has no realized physical routes");
}

/* the program a call of `bytes` runs: its communicator's table's first entry of its collective (and root, a rooted
   one's) whose bound it is below (a configured form's, else the default's); NULL where none is (mesh 2bf5a50d^
   program_of; cc8a6f8e^'s by size class, the classes now the forms' bounds) */
static const struct entry *program_of(ncclComm_t comm, int what, int root, uint64_t bytes) {
  for (uint32_t e = 0; e < comm->programs; e++) {
    const struct entry *x = comm->entries + e;
    if ((int)x->program.what != what || ((what == MESH_REDUCE || what == MESH_BROADCAST) && (int)x->program.root != root)) continue;
    if (!x->below || bytes < x->below) return x;
  }
  return NULL;
}

/* -- the process's client of its node's bridge (design/transport.md):
   one attach of the node's region (mesh_attach), made at the process's first lease or first communicator, whichever
   comes first.  Its slot owns every lease the process takes (ncclMemAlloc: a section of the region, mesh-call.h
   mesh_section_create, whose rows' owner is the slot), and it holds a stream on each physical cable its communicators use
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
#include "mesh-lease-retire.h"
struct lease { void *address; uint32_t row; size_t bytes; void *buffer; struct mesh_lease_retire reads; int returned; };
/* design/transport.md section 14: the stream toward node `node` the process holds (`users` the communicator on it, 0
   none: the hold vacant), its holding's view, and the identity it was taken by (`wire`: mesh-call.h mesh_stream_held,
   comm_failure) */
struct hold { uint32_t node, stream, users, padding; struct mesh_stream_view view; uint64_t wire, generation; };
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

/* comm's link on channel h lost (its stream no longer its own: a peer gone ends its holding), recorded in its failure
   words where no wait has recorded a failure (their claim, nccl-mesh-kernels.h mesh_step_claim), so every GPU wait on
   them ends within 2^10 loads (mesh_step_failed) and the host reads the loss as a value (comm_failure) */
static void comm_lost(struct ncclComm *comm, uint32_t h) {
  if (atomic_exchange_explicit((_Atomic uint32_t *)(void *)(comm->words + 1), 1, memory_order_acq_rel)) return;
  atomic_store_explicit(comm->words, METAL_LOST, memory_order_relaxed);
  atomic_store_explicit(comm->words + 3, h, memory_order_relaxed);
  atomic_store_explicit(comm->words + 2, UINT64_MAX, memory_order_release);
}
/* design/transport.md section 5: every MESH_PLATFORM_NS / 20, each communicator's streams read: one no longer its own
   (a peer gone ends its holding: mesh_stream_held) is recorded as a loss (comm_lost), ending its GPU waits at once as a
   value; a bound a GPU wait records (bound_reached) ends the program at once, named.  No wait is extended: each GPU
   wait's bound and each host wait's deadline is fixed when it is made (design/eager-execution.md assumption A4).
   Then the library's own GPU programs: a communicator's last program committed and not ended, seen so at every look
   since `finished` last moved for twice MESH_PLATFORM_NS, the program ended there, the wait named; a look that finds none
   committed and not ended starts the count again, so time with no program does not count; group_lock only where free
   (a holder's own waits are bounded) */
static void watch_turn(void *unused) {
  (void)unused;
  static uint64_t seen, since;
  char wait[MESH_DIAGNOSTIC_BYTES];
  uint64_t idle = 0, now = mesh_clock_ns();
  pthread_mutex_lock(&live_lock);
  for (struct ncclComm *c = live; c && !idle; c = c->next_live) {
    int held = !__atomic_load_n(&c->revoked, __ATOMIC_ACQUIRE);
    for (uint32_t h = 0; held && h < c->count; h++)
      if (!(held = mesh_stream_held(&c->attach->context, c->channels[h].stream, *c->channels[h].wire))) comm_lost(c, h);
    idle = bound_reached(c, wait, sizeof wait);
  }
  pthread_mutex_unlock(&live_lock);
  if (!idle && !pthread_mutex_trylock(&group_lock)) {
    const uint64_t signaled = finished ? metal_signaled(finished) : 0;
    const struct ncclComm *pending = NULL;
    now = mesh_clock_ns();
    pthread_mutex_lock(&live_lock);
    for (const struct ncclComm *c = live; c && !pending; c = c->next_live)
      if (c->last > signaled && c->command && !metal_ended(c->command)) pending = c;
    if (!pending || signaled != seen || !since) { seen = signaled; since = pending ? now : 0; }
    else if (now - since > 2 * MESH_PLATFORM_NS) {
      idle = now - since;
      snprintf(wait, sizeof wait, "libnccl-mesh: communicator %s's GPU program %llu, committed and not ended (watch_turn)", pending->name,
               (unsigned long long)pending->last);
    }
    pthread_mutex_unlock(&live_lock);
    pthread_mutex_unlock(&group_lock);
  }
  if (idle) mesh_library_expired(wait, idle);
}
static void watch_start(void *unused) {
  (void)unused;
  mesh_expiry_set(mesh_library_expired);
  dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
  dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(MESH_PLATFORM_NS / 20)), MESH_PLATFORM_NS / 20, MESH_PLATFORM_NS / 200);
  dispatch_source_set_event_handler_f(timer, watch_turn);
  dispatch_resume(timer);
}
void mesh_library_watch(void) {
  static dispatch_once_t once;
  dispatch_once_f(&once, NULL, watch_start);
}
static uint64_t holding_generation;
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
  if (comm && !(comm->members = calloc((size_t)nranks, sizeof *comm->members))) {
    free(comm->members); free(comm); comm = NULL;
  }
  if (!comm) return NULL;
  comm->nranks = nranks; comm->rank = rank; comm->fabric.nodes = (uint32_t)nranks; comm->listener = -1;
  pthread_mutex_init(&comm->guard, NULL);
  pthread_mutex_init(&comm->prepare, NULL);
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
static int lease_read_sample(void *, uint32_t, struct mesh_lease_retire_sample *);
static void leases_reap(struct attach *);
static void comm_let_go(struct ncclComm *comm) {
  if (!comm->binding) return;
  comm->binding = 0;
  for (size_t i = 0; i < comm->attach->count; i++)
    mesh_lease_retire_ready(&comm->attach->leases[i].reads, lease_read_sample, comm->attach);
  for (uint32_t k = 0; k < MESH_STREAMS; k++) {
    struct hold *h = comm->attach->holds + k;
    if (!(comm->held >> k & 1) || !h->users || --h->users) continue;
    if (mesh_stream_held(&comm->attach->context, h->stream, h->wire)) mesh_stream_give(&comm->attach->context, h->stream);
    *h = (struct hold){0};
  }
  comm->held = 0;
  leases_reap(comm->attach);
}
static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  if (comm->attach) comm_let_go(comm);
  pthread_mutex_unlock(&attach_lock);
  comm_storage_free(comm);
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
   to take, each one's node and identity (`frees` of them: reach_read, mesh-call.h mesh_streams_free) */
struct report {
  int32_t rank, status;
  uint32_t node, count;
  struct mesh_endpoint endpoint[MESH_STREAMS];
};

/* design/whole-function.md#fabric */
static int cable_order(const void *left, const void *right) {
  const struct mesh_cable *a = left, *b = right;
  for (uint32_t k = 0; k < 2; k++) if (a->node[k] != b->node[k]) return a->node[k] < b->node[k] ? -1 : 1;
  return (a->wire > b->wire) - (a->wire < b->wire);
}

static int fabric_discover(struct ncclComm *comm, const struct report *reports) {
  const uint32_t n = (uint32_t)comm->nranks;
  struct mesh_cable *cables = calloc((size_t)n * MESH_STREAMS + 1, sizeof *cables);
  if (!cables) return ENOMEM;
  comm->fabric = (struct mesh_fabric){n, 0, cables};
  for (uint32_t a = 0; a < n; a++) for (uint32_t b = a + 1; b < n; b++) {
    const struct report *x = reports + a, *y = reports + b;
    for (uint32_t i = 0; i < x->count; i++) {
      const struct mesh_endpoint *one = x->endpoint + i;
      if (one->node != y->node) continue;
      uint32_t earlier = 0;
      for (uint32_t k = 0; k < i; k++) earlier |= x->endpoint[k].bundle == one->bundle;
      if (earlier) continue;
      uint64_t wire = 0, bandwidth = 0;
      int reciprocal = 0;
      for (uint32_t k = i; k < x->count; k++) {
        const struct mesh_endpoint *u = x->endpoint + k;
        if (u->bundle != one->bundle) continue;
        for (uint32_t j = 0; j < y->count; j++) {
          const struct mesh_endpoint *v = y->endpoint + j;
          if (v->node != x->node || v->wire != u->wire) continue;
          reciprocal = 1;
          if (!u->available || !v->available || (wire && wire < u->wire)) continue;
          wire = u->wire;
          bandwidth = u->bandwidth < v->bandwidth ? u->bandwidth : v->bandwidth;
        }
      }
      if (!wire) return reciprocal ? EBUSY : ENOLINK;
      cables[comm->fabric.cables++] = (struct mesh_cable){wire, bandwidth, {a, b}};
    }
    for (uint32_t j = 0; j < y->count; j++) {
      const struct mesh_endpoint *v = y->endpoint + j;
      if (v->node != x->node) continue;
      int reciprocal = 0;
      for (uint32_t i = 0; i < x->count; i++) reciprocal |= x->endpoint[i].node == y->node && x->endpoint[i].wire == v->wire;
      if (!reciprocal) return ENOLINK;
    }
  }
  qsort(cables, comm->fabric.cables, sizeof *cables, cable_order);
  const uint32_t links = comm->fabric.cables;
  struct mesh_link_map *map = &comm->map;
  *map = (struct mesh_link_map){.kind = MESH_LINKS_GRAPH, .nodes = n, .links = links,
    .link = calloc(links + 1, sizeof *map->link), .bandwidth = calloc(links + 1, sizeof *map->bandwidth),
    .mark = calloc(n + 1, sizeof *map->mark)};
  comm->endpoint = malloc((links + 1) * sizeof *comm->endpoint);
  if (!map->link || !map->bandwidth || !map->mark || !comm->endpoint) return ENOMEM;
  for (uint32_t e = 0; e < links; e++) {
    memcpy(map->link[e], cables[e].node, sizeof cables[e].node);
    map->bandwidth[e] = cables[e].bandwidth;
    comm->endpoint[e] = UINT32_MAX;
  }
  for (uint32_t r = 0; r < n; r++) map->mark[r] = reports[r].node;
  int checked = mesh_fabric_check(&comm->fabric);
  if (!checked) checked = mesh_link_map_joined(map, NULL);
  if (!checked) checked = mesh_write_routes_default(&comm->fabric, &comm->writes);
  if (!checked && !(comm->write_overrides = calloc((size_t)n * n, 1))) checked = ENOMEM;
  return checked;
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
   bridge's: a rank attaches once its bootstrap has connected, or accepted every rank (comm_init).  The first gather
   holds no stream of the bridge's; the second follows this rank's holds of its streams, on which its peers wait.
   Each phase's waits (the connect, the accepts, each gather) end by the phase's deadline, MESH_PREPARATION_NS from its
   start (the ranks' preparation skew), past which the program ends naming the phase and the rank awaited (setup_left).  A rank's report is struct report.  The id
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

/* A split owns a fresh configuration listener. It neither honors the parent's
   NCCL_COMM_ID nor replaces the unrelated process-global public bootstrap. */
static int bootstrap_child_id(struct ncclComm *comm, ncclUniqueId *id) {
  char host[256] = {0};
  if (fabric_address(host, sizeof host)) *host = 0;
  if (!*host && gethostname(host, sizeof host - 1)) return errno;
  uint16_t port = 0;
  const int fd = bootstrap_listen(host, &port);
  if (fd < 0) return errno;
  pthread_mutex_lock(&comm->guard);
  comm->listener = fd;
  pthread_mutex_unlock(&comm->guard);
  memset(id, 0, sizeof *id);
  snprintf(id->internal, sizeof id->internal, "%s:%u", host, port);
  return 0;
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
/* design/rubrics.md M3, C5: each wait of a bootstrap a slice of MESH_BOOTSTRAP_SLICE_MS (a poll's timeout, a placed socket's
   SO_RCVTIMEO and SO_SNDTIMEO), after which the revoke and the phase's deadline (`until`) are read again: no wait of an
   init lasts past its phase's deadline, blocking or not, the program ending there (design/eager-execution.md, assumption A4),
   and a revoke ends a non-blocking one within a slice */
/* what is left of the phase's deadline (`until`, MESH_PREPARATION_NS from the phase's start: it covers the ranks'
   preparation skew, so a peer still preparing is waited for); past it the program ends (mesh_library_expired), naming
   the phase, this rank and the rank awaited (-1: a connection not yet identified) */
static uint64_t setup_left(const struct ncclComm *comm, int peer) {
  const uint64_t now = mesh_clock_ns();
  if (now < comm->until.at) return comm->until.at - now;
  char wait[MESH_DIAGNOSTIC_BYTES];
  snprintf(wait, sizeof wait, "%s: communicator %s's rank %d of %d awaiting rank %d (MESH_PREPARATION_NS)", comm->until.wait, comm->name,
           comm->rank, comm->nranks, peer);
  mesh_library_expired(wait, comm->until.ns);
}
/* a poll's timeout in ms: a slice, or what is left of the phase's deadline where less (setup_left) */
static int bootstrap_slice(const struct ncclComm *comm, int peer) {
  const uint64_t left = setup_left(comm, peer);
  return left < (uint64_t)MESH_BOOTSTRAP_SLICE_MS * 1000000 ? (int)((left + 999999) / 1000000) : MESH_BOOTSTRAP_SLICE_MS;
}
/* `size` bytes sent or received whole on comm's bootstrap socket `fd` to or from rank `peer`: 0, ECANCELED (revoked),
   ECONNRESET (the peer closed), or the errno; past the phase's deadline the program ends (setup_left) */
static int whole(struct ncclComm *comm, int fd, int peer, void *bytes, size_t size, int sending) {
  for (size_t done = 0; done < size;) {
    const ssize_t moved = sending ? send(fd, (char *)bytes + done, size - done, 0) : recv(fd, (char *)bytes + done, size - done, 0);
    if (moved < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      if (revoked(comm)) return ECANCELED;
      setup_left(comm, peer);
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
   its listener's O_NONBLOCK; each send and receive returns within MESH_BOOTSTRAP_SLICE_MS: whole), under comm's guard, where
   a revoke shuts it down (ncclCommRevoke), a send on it after that or a peer's reset an error, never SIGPIPE */
static void bootstrap_put(struct ncclComm *comm, int r, int fd) {
  const struct timeval slice = {0, MESH_BOOTSTRAP_SLICE_MS * 1000};
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
   phase's deadline (`until`) is MESH_PREPARATION_NS after it began: a non-root's connect, then its first gather; the
   root's accepts, every rank's report and its first gather under the one deadline made as its accepts begin; past it
   the program ends: 0, ECANCELED (comm revoked), or the errno */
static int bootstrap_open(struct ncclComm *comm, const char *id, struct report *reports) {
  int *peers = comm->peers, status = 0;
  char host[NCCL_UNIQUE_ID_BYTES];
  snprintf(host, sizeof host, "%s", id);
  char *colon = strrchr(host, ':');
  if (!colon) return EINVAL;
  *colon = 0;
  comm->until = mesh_deadline("libnccl-mesh: the bootstrap's connect to its root (bootstrap_open)", MESH_PREPARATION_NS);
  if (comm->rank) {
    struct addrinfo hint = {.ai_socktype = SOCK_STREAM}, *addresses = NULL;
    while (peers[0] < 0 && !revoked(comm)) {
      if (!getaddrinfo(host, colon + 1, &hint, &addresses)) {
        for (struct addrinfo *a = addresses; a && peers[0] < 0 && !revoked(comm); a = a->ai_next) {
          const int fd = socket(a->ai_family, SOCK_STREAM, 0);
          if (fd < 0) continue;
          int error = fcntl(fd, F_SETFL, O_NONBLOCK) || connect(fd, a->ai_addr, a->ai_addrlen) ? errno : 0;
          struct pollfd look = {fd, POLLOUT, 0};
          socklen_t length = sizeof error;
          while (error == EINPROGRESS || error == EINTR)
            if (revoked(comm)) error = ECANCELED;
            else if (poll(&look, 1, bootstrap_slice(comm, 0)) < 0) error = errno == EINTR ? EINPROGRESS : errno;
            else if (look.revents && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length)) error = errno;
          if (error) close(fd);
          else bootstrap_put(comm, 0, fd);
        }
        freeaddrinfo(addresses);
      }
      if (peers[0] < 0) usleep((useconds_t)bootstrap_slice(comm, 0) * 1000);
    }
    comm->until = mesh_deadline("libnccl-mesh: the bootstrap's first gather, a rank's report and the table (gather)", MESH_PREPARATION_NS);
    return revoked(comm) ? ECANCELED : 0;
  }
  uint16_t port = (uint16_t)atoi(colon + 1);
  int listener = comm->listener >= 0 ? comm->listener : !strcmp(root.id, id) ? root.fd : -1;
  if (listener < 0 && (listener = bootstrap_listen(host, &port)) < 0) return errno;
  comm->until = mesh_deadline("libnccl-mesh: the bootstrap's root, every rank's connect and report and the first gather (bootstrap_open)", MESH_PREPARATION_NS);
  for (int r = 1; r < comm->nranks && !status;) {
    struct report heard;
    struct pollfd look = {listener, POLLIN, 0};
    int fd = -1, awaited = 1;
    while (awaited < comm->nranks && peers[awaited] >= 0) awaited++;
    while (fd < 0 && !status)
      if (revoked(comm)) status = ECANCELED;
      else if (poll(&look, 1, bootstrap_slice(comm, awaited)) < 0 && errno != EINTR) status = errno;
      else if ((fd = accept(listener, NULL, NULL)) < 0 && errno != EWOULDBLOCK && errno != ECONNABORTED && errno != EINTR) status = errno;
    if (!status) bootstrap_put(comm, 0, fd);
    if (!status && revoked(comm)) status = ECANCELED;
    const int read = status ? status : whole(comm, fd, -1, &heard, sizeof heard, 0);
    if (read == ECANCELED) status = read;
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
  if (listener == comm->listener) comm->listener = -1;
  if (listener == root.fd) root.fd = -1;
  return status;
}

/* rows (each rank's `bytes` at rows + rank * bytes, this rank's in place) made every rank's on comm's bootstrap sockets
   within its phase (`until`): each other rank's row to the root (`heard`: the root holds them already, as the first
   gather's root, whose bootstrap_open read each), then the whole table to each; `ranked`: a row leads with its int32
   rank (struct report), checked before the table is sent.  NCCL's bootstrapAllGather; never a payload's.  0, or the
   errno (whole) */
static int exchange(struct ncclComm *comm, void *rows, size_t bytes, int heard, int ranked) {
  unsigned char *table = rows;
  const int n = comm->nranks, *peers = comm->peers;
  if (n == 1) return 0;
  if (!peers) return ENOTCONN;
  int status = 0;
  if (comm->rank) {
    if (peers[0] < 0) return ENOTCONN;
    status = whole(comm, peers[0], 0, table + (size_t)comm->rank * bytes, bytes, 1);
    return status ? status : whole(comm, peers[0], 0, table, (size_t)n * bytes, 0);
  }
  for (int r = 1; r < n && !status && !heard; r++) {
    int32_t rank = r;
    status = peers[r] < 0 ? ENOTCONN : whole(comm, peers[r], r, table + (size_t)r * bytes, bytes, 0);
    if (!status && ranked) memcpy(&rank, table + (size_t)r * bytes, sizeof rank);
    if (!status && rank != r) status = EPROTO;
  }
  for (int r = 1; r < n && !status; r++) status = peers[r] < 0 ? ENOTCONN : whole(comm, peers[r], r, table, (size_t)n * bytes, 1);
  return status;
}

/* this rank's report of its bridge's node, the nodes it reaches (mesh-call.h mesh_nodes_reached) and the streams free to
   take (mesh_streams_free) */
static void reach_read(struct attach *a, struct report *report) {
  report->node = a->context.M->node;
  report->count = mesh_endpoints(a->context.M, report->endpoint, MESH_STREAMS);
}

/* design/whole-function.md#fabric */
static int comm_apart(const struct report *reports, uint32_t n) {
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = a + 1; b < n; b++)
      if (reports[a].node == reports[b].node) return EINVAL;
  return 0;
}
static ncclResult_t apart_fail(const char *name) {
  return fail(ncclInvalidUsage, "communicator %s: two of its ranks are on one node (its holdings are kept a node)", name);
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


/* design/whole-function.md#fabric */
static int fabric_acquire(struct ncclComm *comm) {
  struct attach *a = comm->attach;
  int status = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t e = 0; !status && e < comm->fabric.cables; e++) {
    const struct mesh_cable *c = comm->fabric.cable + e;
    if (c->node[0] != (uint32_t)comm->rank && c->node[1] != (uint32_t)comm->rank) continue;
    const uint32_t peer = c->node[c->node[0] == (uint32_t)comm->rank];
    uint32_t slot = 0, stream = 0;
    while (slot < MESH_STREAMS && a->holds[slot].users) slot++;
    if (slot == MESH_STREAMS) { status = ENOSPC; break; }
    if (holding_generation == UINT64_MAX) { status = EOVERFLOW; break; }
    if ((status = mesh_stream_take(&a->context, c->wire, &stream))) break;
    struct hold *h = a->holds + slot;
    *h = (struct hold){.node = comm->members[peer], .stream = stream, .wire = c->wire, .users = 1, .generation = ++holding_generation};
    comm->endpoint[e] = slot; comm->held |= UINT32_C(1) << slot; comm->binding = 1;
    status = mesh_stream_view(&a->context, stream, &h->view);
  }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* the Metal buffer over the region's page (or pages) that holds a channel's counts, each count's offset in it, and
   `ready`'s region offset */
static int channel_buffer(struct channel *ch, const struct hdr *m) {
  const uintptr_t page = (uintptr_t)getpagesize();
  const uintptr_t at[2] = {(uintptr_t)ch->view.ready, (uintptr_t)ch->view.landed};
  uintptr_t lo = at[0], hi = at[0] + sizeof(uint64_t);
  for (int k = 1; k < 2; k++) { lo = at[k] < lo ? at[k] : lo; hi = at[k] + sizeof(uint64_t) > hi ? at[k] + sizeof(uint64_t) : hi; }
  lo &= ~(page - 1);
  hi = (hi + page - 1) & ~(page - 1);
  ch->ready = at[0] - lo; ch->landed = at[1] - lo;
  ch->ready_at = at[0] - (uintptr_t)m;
  return (ch->buffer = metal_wrap((void *)lo, hi - lo)) ? 0 : ENOMEM;
}

static int lease_buffer(const void *ptr, void **buffer, size_t *offset);
/* design/whole-function.md#fabric */
static int fabric_realize(struct ncclComm *comm) {
  const struct hdr *m = comm->attach->context.M;
  size_t offset = 0;
  comm->page = (uint64_t)m->block * m->pgsz;
  int status = 0;
  void *words = NULL;
  if (ncclMemAlloc(&words, (size_t)getpagesize()) || lease_buffer(words, &comm->violation, &offset) || offset) status = ENOMEM;
  else { memset(words, 0, (size_t)getpagesize()); comm->words = words; }
  if (!status && !(comm->channels = calloc(comm->fabric.cables + 1, sizeof *comm->channels))) status = ENOMEM;
  if (!status && comm->nranks > 1 && (ncclMemAlloc(&comm->gate_word, sizeof(uint64_t)) || lease_buffer(comm->gate_word, &comm->gate, &offset) || offset))
    status = ENOMEM;
  if (!status && comm->gate_word) comm->gate_at = (uint64_t)((char *)comm->gate_word - (char *)m);
  for (uint32_t e = 0; !status && e < comm->fabric.cables; e++) {
    if (comm->endpoint[e] == UINT32_MAX) continue;
    const struct mesh_cable *c = comm->fabric.cable + e;
    const uint32_t peer = c->node[c->node[0] == (uint32_t)comm->rank];
    struct hold *h = comm->attach->holds + comm->endpoint[e];
    struct channel *ch = comm->channels + comm->count++;
    *ch = (struct channel){.peer = (int)peer, .node = comm->members[peer], .stream = h->stream,
      .cable = e, .wire = &h->wire, .view = h->view};
    if (!(status = channel_buffer(ch, m))) atomic_store(&comm->attach->wrapped, 1);
  }
  /* live once its channels are made, the watch reading them (watch_turn) */
  if (!status) {
    pthread_mutex_lock(&live_lock);
    comm->next_live = live; live = comm;
    pthread_mutex_unlock(&live_lock);
    mesh_library_watch();
  }
  return status;
}

/* design/transport.md section 14, a communicator's init on NCCL's bootstrap at its id, the one path of every init
   (ncclMeshCommInitRank, blocking or not): its bootstrap's sockets opened (bootstrap_open), then made attached
   (comm_attach) and the nodes its bridge reaches read (reach_read), then every rank's report gathered (two ranks on one
   node fail every rank alike: comm_apart, a bound); each rank holding the stream toward each member's node it is
   joined to (fabric_acquire); a second gather carrying each rank's holding's status (NCCL's own init exchanges each peer's
   connection through its bootstrap: ncclTransportP2pSetup), then its channels (fabric_realize).  A rank that cannot go on, a stream held by another (EBUSY) or gone (ENOLINK), a peer silent past
   its phase's end (bootstrap_open, exchange: MESH_PREPARATION_NS) or a revoke fails the init as a value, made holding nothing of its
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
  if (!status && nranks > 1) status = exchange(made, reports, sizeof *reports, !rank, 1);
  ncclResult_t result = status ? fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status)) : making ? comm_refused(making, made->name) : ncclSuccess;
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status && r != rank)
      result = fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  for (int r = 0; !result && r < nranks; r++) made->members[r] = reports[r].node;
  if (!result && nranks > 1 && comm_apart(reports, (uint32_t)nranks)) result = apart_fail(made->name);
  if (!result && nranks > 1) {
    mine->status = fabric_discover(made, reports);
    if (!mine->status) mine->status = fabric_acquire(made);
    made->until = mesh_deadline("libnccl-mesh: the init's second gather, after its holds (exchange)", MESH_PREPARATION_NS);
    if ((status = exchange(made, reports, sizeof *reports, 0, 1))) result = fail(ncclRemoteError, "the bootstrap at %s: %s", made->id, strerror(status));
    for (int r = 0; !result && r < nranks; r++)
      if (reports[r].status) result = hold_fail(made->name, r, reports[r].status);
    if (!result && (status = fabric_realize(made)))
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
   one that overruns; the stop that is the destroy's own does not end it, and a buffer not committed is not awaited:
   nothing would run it while its committer waits), its groups in flight lost, each of these waits with a deadline
   MESH_PLATFORM_NS from its start, past which the program ends (the operator, 2026-10-23: no wait runs longer than a
   deadline presented when it was made; design/eager-execution.md assumption A4); then its streams
   given back (its bridge posts nothing more of their holdings), its tables and slots returned, and comm freed.  Work encoded in a caller's encoder is the
   caller's to finish first (NCCL's: no communicator is destroyed under its own operations in flight): a kernel of it
   still queued would write the stream given back (rdma/NCCL.md "Failures") */
static int gpu_wait(struct ncclComm *comm, int group);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  comm_stop(comm);
  if (comm->initing) {
    const struct mesh_deadline ended = mesh_deadline("ncclCommDestroy: its non-blocking init's end (init_run's `async`)", 0);
    while (__atomic_load_n(&comm->async, __ATOMIC_ACQUIRE) == ncclInProgress) { deadline_left(&ended); usleep(20); }
    pthread_join(comm->init, NULL);
  }
  struct mesh_deadline returned = mesh_deadline("ncclCommDestroy: its host-path groups' return (`runs`), none returning", 0);
  pthread_mutex_lock(&group_lock);
  while (atomic_load(&comm->runs)) {
    pthread_mutex_unlock(&group_lock);
    deadline_left(&returned);
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

/* Local DMA completion, never a remote receipt. The holding generation prevents
   a reused stream's prefix from retiring an earlier source. attach_lock held. */
static int lease_read_sample(void *context, uint32_t stream, struct mesh_lease_retire_sample *sample) {
  struct attach *a = context;
  for (uint32_t k = 0; k < MESH_STREAMS; k++) {
    const struct hold *h = a->holds + k;
    if (!h->users || h->stream != stream) continue;
    if (!mesh_stream_held(&a->context, stream, h->wire)) return ENOLINK;
    *sample = (struct mesh_lease_retire_sample){.wire = h->wire, .holding = h->generation,
      .sent = atomic_load_explicit(h->view.sent, memory_order_acquire),
      .landed = atomic_load_explicit(h->view.landed, memory_order_acquire)};
    return mesh_stream_held(&a->context, stream, h->wire) ? 0 : ENOLINK;
  }
  return ENOENT;
}

/* Returning a dead tensor can hand outstanding posted reads to bridge-owned
   pins once every holding that could post FUTURE reads has been given back.
   This is not a proof allowing the application to overwrite the allocation. */
static int lease_detachable(void *context, const struct mesh_lease_retire_dep *d) {
  struct attach *a = context;
  for (uint32_t k = 0; k < MESH_STREAMS; k++) {
    const struct hold *h = a->holds + k;
    if (h->users && h->stream == d->stream && h->generation == d->holding) return EBUSY;
  }
  if (d->metadata) {
    if (d->stream >= MESH_STREAMS) return EINVAL;
    const uint64_t owner = atomic_load_explicit(&mesh_stream_at(a->context.M, d->stream)->owner, memory_order_acquire);
    /* client|GIVEN prevents new DMA posts, but the progress thread can still
       be reading a descriptor before unit_told notices the give. Only its
       subsequent owner transition proves those local CPU reads have ended. */
    if (owner != 0 && owner != MESH_STREAM_GIVEN) return EBUSY;
  }
  return 0;
}
static int lease_returnable(struct attach *a, struct lease *l) {
  return mesh_lease_retire_detachable(&l->reads, lease_read_sample, lease_detachable, a);
}
static void leases_reap(struct attach *a) {
  for (size_t i = 0; i < a->count;) {
    struct lease *l = a->leases + i;
    if (!l->returned || !lease_returnable(a, l)) { i++; continue; }
    if (mesh_lease_return(&a->context, l->row)) { i++; continue; }
    if (l->buffer) metal_release(l->buffer);
    mesh_lease_retire_destroy(&l->reads);
    *l = a->leases[--a->count];
  }
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
  if (!status) leases_reap(a);
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
    a->leases[a->count++] = (struct lease){.address = *ptr, .row = lease.first, .bytes = (size_t)lease.pages * a->context.M->pgsz};
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

/* Logical free is asynchronous. All scheduled DMA reads, including descriptors
   not yet taken by the bridge, keep the source alive without an ACK or wait. */
ncclResult_t ncclMemFree(void *ptr) {
  if (!ptr) return ncclSuccess;
  pthread_mutex_lock(&attach_lock);
  int found = 0;
  for (struct attach *a = attaches; a && !found; a = a->next)
    for (size_t i = 0; i < a->count && !found; i++)
      if (a->leases[i].address == ptr && !a->leases[i].returned) {
        a->leases[i].returned = found = 1;
        leases_reap(a);
      }
  attach_settle();
  pthread_mutex_unlock(&attach_lock);
  return found ? ncclSuccess : fail(ncclInvalidArgument, "ncclMemFree of %p: no live lease of this process", ptr);
}

/* the lease holding ptr and its attach (*held, where held is not NULL), under attach_lock (NULL: none) */
static struct lease *lease_at(const void *ptr, struct attach **held) {
  for (struct attach *a = attaches; a; a = a->next)
    for (size_t i = 0; i < a->count; i++) {
      const uintptr_t at = (uintptr_t)ptr, first = (uintptr_t)a->leases[i].address;
      if (at >= first && at < first + a->leases[i].bytes) {
        if (held) *held = a;
        return a->leases + i;
      }
    }
  return NULL;
}

/* the lease holding `ptr` (any byte of it): the Metal buffer over the whole lease (made at its first use, kept until
   ncclMemFree) and ptr's offset in it; 0, ENOENT where no lease of this process holds ptr, ENOMEM */
static int lease_buffer(const void *ptr, void **buffer, size_t *offset) {
  pthread_mutex_lock(&attach_lock);
  struct attach *a = NULL;
  struct lease *l = lease_at(ptr, &a);
  if (l && !l->buffer && (l->buffer = metal_wrap(l->address, l->bytes))) atomic_store(&a->wrapped, 1);
  const int status = !l ? ENOENT : l->buffer ? 0 : ENOMEM;
  if (!status) { *buffer = l->buffer; *offset = (size_t)((uintptr_t)ptr - (uintptr_t)l->address); }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* nccl.h ncclMeshMemBuffer */
ncclResult_t ncclMeshMemBuffer(const void *ptr, void **buffer, size_t *offset) {
  if (!ptr || !buffer || !offset) return fail(ncclInvalidArgument, "ncclMeshMemBuffer: a NULL argument");
  const int status = lease_buffer(ptr, buffer, offset);
  if (status == ENOMEM) return fail(ncclSystemError, "ncclMeshMemBuffer of %p: no Metal buffer over its lease", ptr);
  return status ? fail(ncclInvalidArgument, "ncclMeshMemBuffer: %p is in no lease of this process (ncclMemAlloc)", ptr) : ncclSuccess;
}

/* nccl.h ncclMeshCommPage */
ncclResult_t ncclMeshCommPage(ncclComm_t comm, size_t *bytes) {
  if (!comm || !bytes) return fail(ncclInvalidArgument, "ncclMeshCommPage: a NULL argument");
  *bytes = (size_t)comm->page;
  return ncclSuccess;
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

/* -- a communicator's channels and its failures (design/transport.md section 14). Every physical cable has its
   own channel on the holding init took toward its peer. Each direction is its own list: this
   rank's sends to the member are its holding's send list and the member's landing list from it, entry for entry in the
   order both ends lay alike (declared call order and compiled physical ordinals: mesh-plan.h mesh_group_lay),
   a step one entry whatever its bytes. A send entry names its exact byte spans; a landing entry names their
   destination spans (the destination itself for a COPY, the group's landing area for a combine).
   This end publishes its sends by raising its holding's `ready` past them, in
   order, and consumes its landings once `landed` covers them.  Nothing credits a publication: the hardware's credit
   holds a SEND until its receive is posted (design/transport.md).  A Metal-path landing entry is gated on its
   group's start in the stream (gate_of), so no receive lands in pages the stream's earlier work still reads.  A group
   presents its entries before it publishes any (schedule); a member that fails mid-group sends nothing more, and its
   holding's end with units outstanding takes its stream out of the pairing (design/transport.md section 14). ---------- */

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

/* the communicator's failure: the one it kept; a revoke; a stream it holds no longer its own (a bridge's restart, or a
   pairing lost and made again, opens every stream new: mesh-call.h mesh_stream_held); one its GPU work recorded in its
   failure words (its link lost, or a wait's bound reached after a revoke: nccl-mesh-metal.h METAL_*; a wait past its
   bound, its deadline, ends the program: bound_reached, design/eager-execution.md, assumption A4), kept now (comm_fail), so
   ncclCommGetAsyncError reads it before any group runs on it; ncclSuccess where none.  Under group_lock */
static ncclResult_t comm_failure(struct ncclComm *comm) {
  if (comm->failed) return comm_told(comm);
  if (revoked(comm))
    return comm_fail(comm, ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): no group runs on it before ncclCommDestroy", comm->name);
  struct attach *a = comm->attach;
  for (uint32_t k = 0; comm->held && k < MESH_STREAMS; k++)
    if ((comm->held >> k & 1) && !mesh_stream_held(&a->context, a->holds[k].stream, a->holds[k].wire))
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost and paired again: communicator %s's stream there is no "
                       "longer its own (design/transport.md section 14); it is made again (ncclCommDestroy, then an init)",
                       a->holds[k].node, comm->name);
  _Atomic uint64_t *v = comm->words;
  const uint64_t at = v ? atomic_load_explicit(v + 2, memory_order_acquire) : 0;
  if (at) {
    const uint64_t cause = atomic_load_explicit(v, memory_order_relaxed), h = atomic_load_explicit(v + 3, memory_order_relaxed);
    const uint32_t node = h < comm->count ? comm->channels[h].node : UINT32_MAX;
    if (cause == METAL_LOST)
      return comm_fail(comm, ncclRemoteError, "the link to node %u was lost (its holding's counts ~0) at entry %llu", node,
                       (unsigned long long)(at - 1));
    char wait[MESH_DIAGNOSTIC_BYTES];
    const uint64_t bound = bound_reached(comm, wait, sizeof wait);
    if (bound) mesh_library_expired(wait, bound);
    return comm_fail(comm, ncclRemoteError, "entry %llu on the channel to node %u was not done within %llu polls of a GPU wait (its "
                     "stream's bound, ncclMeshStream polls): its bytes had not landed, the peer's work there beginning that "
                     "much later, or its link was lost", (unsigned long long)(at - 1), node,
                     (unsigned long long)atomic_load_explicit(v + 4, memory_order_relaxed));
  }
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

/* A fixed physical binding and its shared logical packet cuts. Preparation
   exchanges placement metadata once; execution never exchanges control data.
   References from queued/inflight calls keep the plan alive. Payload ownership
   remains with its caller and the local lease retirement ledger. */
struct ncclMeshWritePlan {
  _Atomic uint32_t refs;
  struct mesh_span_plan *packets;
  size_t occurrences;
  struct mesh_dma_span *spans;
  size_t *span_first; /* step, binding occurrence -> immutable repeated descriptor train */
  mesh_pages *step_units; /* same step/occurrence index, one fused repetition */
  void *payload;
  struct mesh_lease_retire_dep *uses; /* fixed channel/direction indices; only end advances */
  uint32_t channels;
  int retained;
  struct where source, landing;
  struct where source_ready;
  uint64_t ready_value;
  void *ready_owner;
  void *land_records;
  ncclMeshTileLayout ready_layout, land_layout;
  struct mesh_write_layout layout;
  uint32_t land_capacity;
  ncclMeshBuffer source_buffer, landing_buffer;
  ncclComm_t comm;
  size_t count, stride, bytes, row_bytes, rows, pitch;
  ncclMeshSpan *view;
  uint64_t *view_first;
  size_t views, view_extent;
  uint32_t repeats;
  ncclDataType_t type;
  int from, to, metal;
  struct compiled *compiled; /* its definitions, an occurrence each, where it runs alone (write_definitions) */
  struct ncclMeshWritePlan *next_compiled;
};
static size_t write_span_index(const struct ncclMeshWritePlan *p, size_t step, size_t invocation) {
  return step*p->occurrences+invocation%p->occurrences;
}
static const struct mesh_span_plan *write_packets(const struct ncclMeshWritePlan *p, size_t invocation) {
  return p->packets+invocation%p->occurrences;
}
static void write_packets_free(struct ncclMeshWritePlan *p) {
  for(size_t k=0;p->packets&&k<p->occurrences;k++)mesh_span_plan_free(p->packets+k);
  free(p->packets);
}
static size_t write_tiles(const struct ncclMeshWritePlan *p) {
  return p->layout.tile_bytes && p->bytes ? (p->bytes - 1) / p->layout.tile_bytes + 1 : 1;
}
static size_t write_tile(const struct ncclMeshWritePlan *p, size_t logical) {
  return p->layout.tile_bytes ? logical / p->layout.tile_bytes : 0;
}
static size_t tile_offset(ncclMeshTileLayout layout, size_t tile, size_t invocation) {
  return (invocation % layout.instances) * layout.instance_stride +
    (tile / layout.columns) * layout.row_stride + (tile % layout.columns) * layout.tile_stride;
}
/* Bound the affine metadata table before any commands are recorded. */
static int tile_extent(ncclMeshTileLayout layout, size_t tiles, size_t record, size_t *extent) {
  if (!layout.columns || !layout.instances || !tiles ||
      layout.tile_stride % sizeof(uint64_t) || layout.row_stride % sizeof(uint64_t) ||
      layout.instance_stride % sizeof(uint64_t)) return EINVAL;
  // The final row can be partial. Bound it and the last complete row without
  // imposing a row-major order on the table's affine strides.
  const size_t row=(tiles-1)/layout.columns,column=(tiles-1)%layout.columns;
  size_t end=0;
  for(unsigned k=0;k<(row?2u:1u);k++) {
    const size_t r=k?row-1:row,c=k?layout.columns-1:column;
    size_t at=record;
    if(layout.tile_stride && c>(SIZE_MAX-at)/layout.tile_stride)return EOVERFLOW;
    at+=c*layout.tile_stride;
    if(layout.row_stride && r>(SIZE_MAX-at)/layout.row_stride)return EOVERFLOW;
    at+=r*layout.row_stride;
    if(at>end)end=at;
  }
  if(layout.instance_stride && layout.instances-1>(SIZE_MAX-end)/layout.instance_stride)return EOVERFLOW;
  *extent=end+(layout.instances-1)*layout.instance_stride;return 0;
}
static uint64_t write_displacement(const struct ncclMeshWritePlan *p, uint64_t logical) {
  if (p && p->views) {
    size_t low = 0, high = p->views;
    while (low < high) { const size_t at = low + (high - low) / 2;
      if (p->view_first[at + 1] <= logical) low = at + 1; else high = at;
    }
    return p->view[low].offset + logical - p->view_first[low];
  }
  return p && p->rows > 1 ? (logical / p->row_bytes) * p->pitch + logical % p->row_bytes : logical;
}
static void write_plan_drop(struct ncclMeshWritePlan *p) {
  if (p && atomic_fetch_sub_explicit(&p->refs, 1, memory_order_acq_rel) == 1) {
    if (p->retained) {
      /* Releasing a realized configuration transfers its final uses to the
         allocator. No execution performs an ownership search or reuse query. */
      pthread_mutex_lock(&attach_lock);
      struct lease *payload = lease_at(p->payload, NULL), *metadata = lease_at(p->spans, NULL);
      struct lease *ready = p->ready_owner ? lease_at(p->ready_owner, NULL) : NULL;
      int error = !payload || !metadata ? EINVAL : 0;
      for (uint32_t k = 0; !error && k < 2 * p->channels; k++) {
        const struct mesh_lease_retire_dep *d = p->uses + k;
        error = mesh_lease_retire_mark_use(&payload->reads, d->stream, d->wire, d->holding, d->end, d->direction, 0);
        if (!error) error = mesh_lease_retire_mark_use(&metadata->reads, d->stream, d->wire, d->holding, d->end, d->direction, 1);
        if (!error && ready && d->direction == MESH_LEASE_SEND)
          error = mesh_lease_retire_mark_use(&ready->reads, d->stream, d->wire, d->holding, d->end, d->direction, 1);
      }
      if (!error) {
        mesh_lease_retire_end(&payload->reads);
        mesh_lease_retire_end(&metadata->reads);
        if (ready) mesh_lease_retire_end(&ready->reads);
      }
      pthread_mutex_unlock(&attach_lock);
      if (error) fail(ncclInternalError, "releasing prepared write ownership: %s", strerror(error));
    }
    if (p->spans) ncclMemFree(p->spans);
    write_packets_free(p); free(p->uses); free(p->span_first); free(p->step_units); free(p->view); free(p->view_first); free(p);
  }
}
/* First cut strictly after a logical byte offset. */
static size_t span_after(const struct mesh_span_plan *p, uint64_t at) {
  size_t lo = 0, hi = p->count;
  while (lo < hi) { const size_t mid = lo + (hi - lo) / 2;
    if (p->cuts[mid] <= at) lo = mid + 1; else hi = mid;
  }
  return lo;
}
static uint64_t span_units(const struct mesh_span_plan *p, uint64_t offset, uint64_t bytes) {
  if (!bytes) return 0;
  return span_after(p, offset + bytes - 1) - span_after(p, offset) + 1;
}

struct steps {
  _Atomic uint32_t references;
  struct steps *source; /* immutable compiler tables retained by an issued binding */
  struct mesh_step *steps;
  uint32_t count;
  unsigned char *done;
  /* the buffers the steps name: 0 the operand, made in place; b >= 1 the rank's scratch b (`buffers` of them) */
  uint32_t buffers, initialized;
  struct where *buffer;
  /* Logical scratch retains the operation's initialization. Relay leases are
     compact, distinct generations populated solely by one-sided arrivals. */
  uint64_t *extent, *demand, *route_units;
  /* a call reading its send buffer in place (plan_direct): each step's source where it reads the send buffer (`read`,
     none: its own buffer) and whether a receive's combine is its range's first (`fresh`: buffer = send op piece) */
  struct where *read;
  unsigned char *fresh;
  /* each step's channel and entry as its group's issue lays them (schedule_laid), and the receive before it (by round,
     then plan order) that has any bytes, or -1 */
  uint32_t *channel;
  uint64_t *entry, *consumption;
  uint32_t *consume_order; /* compiler event order; NULL means physical order */
  uint32_t *retired_first, *retired; /* strict AOT arrival witnesses for prior SEND generations */
  uint32_t *cable, *prior_first, *prior;
  uint32_t priors;
  /* a reduce-scatter's partials of the other ranks' segments (partial_split; 0 none) */
  uint32_t partial;
  uint64_t *complement; /* a copy's room for its call's complement (plan_direct) */
};

/* a call: its collective (or declared WRITE, or WHAT_COUNTED: an all-to-all whose counts the host holds,
   nccl.h ncclMeshAlltoAllV), root, peer, buffers, count, type, operator and communicator; `segments`, an uneven
   reduce-scatter's or all-gather's count a rank (ncclMeshReduceScatterV, ncclMeshAllGatherV: the call's own copy,
   `total` their sum; NULL: `count` each), an all-to-all's elements to each rank then from each (`count` those
   crossing); a reduction's operator as the host runs it (reduction: `combine`, `post`, `pre`; the partials merge's row
   of `unit` elements); its buffers where they are (`send_at`, `recv_at`: struct where), its result's (`sum`, the
   operand made in place) and whether it reads its send buffer in place (`direct`: plan_direct); on the host path a
   buffer of its own where nothing crosses (`result`) and a send to itself's copy (`copy`) */
struct call {
  int what, root, target;
  const void *send;
  void *recv;
  size_t count, total, tile_bytes, invocation;
  ncclMeshProduction *production;
  uint64_t *segments, *matrix;
  uint64_t *contributors, *recipients;
  /* a declaration's: each rank's bindings' offsets within their registration extents ([2r] its send buffer's, [2r + 1]
     its receive buffer's; UINT64_MAX none), exchanged at ncclMeshCommPlan (nccl.h ncclMeshCall bindings) */
  uint64_t *phase;
  /* Configuration owns masks. Issued calls retain only their local roles and
     reduction cardinality, so an in-flight call never borrows a plan's masks. */
  uint32_t roles, participants;
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
  cudaStream_t stream;
  void *command, *encoder, *table, *residency;
  uint32_t polls;
  ncclMeshIssue *issue;
  ncclMeshFused *fused;
  struct where send_at, recv_at, sum;
  struct where *send_views, *receive_views; /* original/final segments, in rank order */
  struct ncclMeshWritePlan *write_plan;
};

static _Thread_local int depth;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;
/* a thread's calls' own parts at each call's place in its list: a V form's counts, its buffer views (an issue copies
   both: struct bound); kept from group to group, each grown only where a call needs more, so a call allocates none once its
   place has room.  Only its group's launch reads them (ncclGroupEnd): no group in flight names one after */
enum { ROOM_SEGMENTS, ROOM_SEND_VIEWS, ROOM_RECEIVE_VIEWS, ROOMS };
struct room { void *part[ROOMS]; size_t bytes[ROOMS]; };
static _Thread_local struct room *rooms;
static _Thread_local size_t room_count;
/* `bytes` of part k at the place the thread's next call takes in its list; NULL where memory runs out */
static void *room_of(int k, size_t bytes) {
  if (call_count >= room_count) {
    const size_t more = call_count + 16;
    struct room *grown = realloc(rooms, more * sizeof *grown);
    if (!grown) return NULL;
    memset(grown + room_count, 0, (more - room_count) * sizeof *grown);
    rooms = grown; room_count = more;
  }
  struct room *r = rooms + call_count;
  if (bytes > r->bytes[k]) {
    void *grown = realloc(r->part[k], bytes);
    if (!grown) return NULL;
    r->part[k] = grown; r->bytes[k] = bytes;
  }
  return r->part[k];
}

static int reducing(int what) { return what == MESH_ALLREDUCE || what == MESH_REDUCE || what == MESH_REDUCE_SCATTER; }
/* a reduction's operand element in the call's elements (the partials merge's row, else one), and a piece of `bytes` as
   the kernels take it: its type and operator (the merge's METAL_MERGE and its row's floats) and its
   elements (the merge's rows) */
static uint32_t unit_of(const struct call *c) { return c->combine == COMBINE_MERGE ? c->unit : 1; }
static int ktype(const struct call *c) { return c->combine == COMBINE_MERGE ? METAL_MERGE : (int)c->type; }
static int kop(const struct call *c) { return c->combine == COMBINE_MERGE ? (int)c->unit : c->combine; }
static size_t kcount(const struct call *c, size_t bytes) { return bytes / (SIZE[c->type] * unit_of(c)); }
static int counted(const struct call *c) { return c->what == WHAT_COUNTED; }
static int call_contributes(const struct call *c) {
  if (c->target) return c->comm->rank == c->root;
  if (c->what == MESH_BROADCAST) return c->comm->rank == c->root;
  return !c->roles || (c->roles & 2);
}
static int call_receives(const struct call *c) {
  if (c->target) return c->comm->rank == c->target - 1;
  if (c->what == MESH_REDUCE) return c->comm->rank == c->root;
  return !c->roles || (c->roles & 4);
}
static uint32_t call_participants(const struct call *c) {
  return c->roles ? c->participants : (uint32_t)c->comm->nranks;
}
static void call_bind_roles(struct call *c, const struct call *from) {
  c->roles = from->roles; c->participants = from->participants;
  if (c->op == ncclAvg && c->premultiplied)
    store(c->type, c->pre, 1.0 / call_participants(c));
}
static size_t elements(const struct call *c) {
  if (counted(c)) return c->count;
  if (c->segments) return c->total;
  return c->count * ((c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER) ? (size_t)c->comm->nranks : 1);
}
/* rank r's segment of a reduce-scatter's or all-gather's operand: the elements before it, and *length its own (r x
   count and count; an uneven call's, its counts') */
static size_t segment_at(const struct call *c, int r, size_t *length) {
  *length = c->segments ? c->segments[r] : c->count;
  return c->segments ? mesh_segment_first(c->segments, (uint32_t)r) : (size_t)r * c->count;
}
/* Logical collective offsets become final local bindings during realization. */
static struct where view_at(const struct call *c, int send, size_t offset) {
  const struct where *views = send ? c->send_views : c->receive_views;
  if (!views) return shifted(send ? c->send_at : counted(c) ? c->recv_at : c->sum, offset);
  const uint64_t *counts = c->segments ? c->segments + (counted(c) && !send ? c->comm->nranks : 0) : NULL;
  for (int rank = 0; rank < c->comm->nranks; rank++) {
    const size_t bytes = (counts ? counts[rank] : c->count) * SIZE[c->type];
    if (offset < bytes) return shifted(views[rank], offset);
    offset -= bytes;
  }
  return views[c->comm->rank]; /* empty local contribution */
}
/* the name of a call's kind, its failures' */
static const char *call_name(const struct call *c) {
  static const char *const names[] = {"ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclReduceScatter", "ncclAllGather"};
  return c->target ? "ncclMeshWriteExecute" : counted(c) ? "ncclMeshAlltoAllV" : names[c->what];
}

/* Whether a call has physical work on a communicator with multiple ranks. */
static int crosses(const struct call *c) {
  /* A counted call with no local payload can still forward another rank's
     transaction. Its complete graph, not its local degree, defines its work. */
  return (counted(c) || elements(c)) && c->comm->nranks > 1;
}

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
  if (!call_contributes(c)) return 0;
  if (c->what == MESH_ALLGATHER) { segment_at(c, c->comm->rank, &length); return length * z; }
  if (counted(c)) return mesh_segment_first(c->segments, (uint32_t)c->comm->nranks) * z;
  if (c->what == MESH_BROADCAST) return c->count * z;
  return elements(c) * z;
}
/* every crossing call's send buffer in this node's region: an operand that crosses a link is a lease, read in place or
   staged in the region, and nothing is copied into the region from outside it (nccl.h, Memory); else the call fails as
   a value, naming it */
static ncclResult_t operands_check(const struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const struct call *c = list + i;
    if (c->write_plan) continue; /* bindings and ownership realized with the page table */
    if (c->send_views) {
      for (int rank=0;rank<c->comm->nranks;rank++) {
        const size_t bytes=(c->segments?c->segments[rank]:c->count)*SIZE[c->type];
        if (bytes&&!in_region(c->comm,c->send_views[rank],bytes))
          return fail(ncclInvalidUsage,"collective source binding is outside its configured region");
      }
      continue;
    }
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
  if (atomic_fetch_sub_explicit(&s->references, 1, memory_order_acq_rel) != 1) return;
  free(s->buffer); free(s->done); free(s->channel); free(s->entry); free(s->read); free(s->fresh);
  free(s->consumption); free(s->complement);
  if (s->source) steps_free(s->source);
  else {
    free(s->steps); free(s->cable); free(s->prior_first); free(s->prior);
    free(s->extent); free(s->demand); free(s->route_units); free(s->consume_order);
    free(s->retired_first); free(s->retired);
  }
  free(s);
}

/* An issue owns mutable bindings only. Compiler instructions and dependency
   proofs remain shared, retaining their generation even after a plan swap. */
static struct steps *steps_copy(struct steps *from) {
  const size_t most = from->count + 1;
  struct steps *s = calloc(1, sizeof *s);
  if (s) {
    *s = (struct steps){.references = 1, .source = from, .count = from->count, .buffers = from->buffers,
      .initialized = from->initialized, .priors = from->priors, .partial = from->partial, .steps = from->steps, .cable = from->cable,
      .prior_first = from->prior_first, .prior = from->prior, .extent = from->extent,
      .consume_order = from->consume_order, .retired_first = from->retired_first, .retired = from->retired};
    atomic_fetch_add_explicit(&from->references, 1, memory_order_relaxed);
  }
  if (!s || !(s->done = malloc(most)) ||
      !(s->channel = calloc(most, sizeof *s->channel)) || !(s->entry = calloc(most, sizeof *s->entry)) ||
      !(s->buffer = calloc(s->buffers, sizeof *s->buffer)) ||
      !(s->consumption = calloc(most, sizeof *s->consumption))) {
    steps_free(s);
    return NULL;
  }
  memcpy(s->done, from->done, from->count);
  return s;
}
/* a copy made again as steps_copy makes it, its arrays kept (an issue laid again: issue_bind): an in-place read none (read and
   fresh all zero, as NULL reads) */
static void steps_reset(struct steps *s) {
  const size_t most = s->count + 1;
  memcpy(s->done, s->source->done, s->count);
  memset(s->channel, 0, most * sizeof *s->channel); memset(s->entry, 0, most * sizeof *s->entry);
  memset(s->consumption, 0, most * sizeof *s->consumption); memset(s->buffer, 0, s->buffers * sizeof *s->buffer);
  if (s->read) { memset(s->read, 0, s->count * sizeof *s->read); memset(s->fresh, 0, s->count); }
}

/* a copy bound again for its issue's same binding (issue_bind): its progress made again, its buffers and in-place
   reads (buffer, read, fresh) kept */
static void steps_rebind(struct steps *s) {
  const size_t most = s->count + 1;
  memcpy(s->done, s->source->done, s->count);
  memset(s->channel, 0, most * sizeof *s->channel); memset(s->entry, 0, most * sizeof *s->entry);
  memset(s->consumption, 0, most * sizeof *s->consumption);
}

/* a call's steps, `initial` with each per-step array made for `room` steps; NULL where memory runs out */
static struct steps *steps_make(size_t room, struct steps initial) {
  struct steps *s = calloc(1, sizeof *s);
  if (s) *s = initial;
  if (s && (s->steps = calloc(room, sizeof *s->steps)) && (s->done = calloc(room, 1)) && (s->channel = calloc(room, sizeof *s->channel)) &&
      (s->entry = calloc(room, sizeof *s->entry)) && (s->cable = calloc(room, sizeof *s->cable)) &&
      (s->consumption = calloc(room, sizeof *s->consumption)) && (s->prior_first = calloc(room, sizeof *s->prior_first))) return s;
  steps_free(s);
  return NULL;
}
/* design/whole-function.md#one-way-writes */
static ncclResult_t write_steps(struct call *c) {
  struct ncclComm *comm = c->comm;
  const struct mesh_paths *paths = comm->writes ? comm->writes + c->root * comm->nranks + c->target - 1 : NULL;
  const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], c->count};
  const int count = mesh_write_compile(&comm->fabric, (uint32_t)comm->rank, (uint32_t)c->root, (uint32_t)c->target - 1,
    paths, operand, comm->page, c->tile_bytes / SIZE[c->type], NULL, 0);
  if (count < 0) return fail(ncclInvalidUsage, "one-way write requires its compiled physical paths: %s", strerror(-count));
  const size_t room = (size_t)count + 1;
  struct steps *s = steps_make(room, (struct steps){.references = 1, .count = (uint32_t)count, .buffers = 1, .initialized = 1});
  struct mesh_transfer *transfers = malloc(room * sizeof *transfers);
  uint32_t *received=malloc(room*sizeof *received);
  if (!s || !transfers || !received || !(s->prior = malloc(room*sizeof *s->prior)) || !(s->extent = calloc(2, sizeof *s->extent))) {
    steps_free(s); free(transfers); free(received); return fail(ncclSystemError, "out of memory");
  }
  const int made = mesh_write_compile(&comm->fabric, (uint32_t)comm->rank, (uint32_t)c->root, (uint32_t)c->target - 1,
    paths, operand, comm->page, c->tile_bytes / SIZE[c->type], transfers, (uint32_t)room);
  if (made < 0) { steps_free(s); free(transfers); free(received); return fail(ncclInternalError, "one-way write compile: %s", strerror(-made)); }
  for (int k = 0; k < made; k++) { s->steps[k] = transfers[k].step; s->cable[k] = transfers[k].cable; }
  /* WRITE paths own disjoint payload intervals, in increasing interval order.
     A relay's receive and send have identical tile cuts. Join those intervals
     once; there is no need to compare each tile with every earlier tile. */
  uint32_t receives=0,at=0;
  for(uint32_t k=0;k<s->count;k++)if(s->steps[k].op!=MESH_STEP_SEND)received[receives++]=k;
  for(uint32_t k=0;k<s->count;k++) {
    s->prior_first[k]=s->priors;
    if(s->steps[k].op!=MESH_STEP_SEND||comm->rank==c->root)continue;
    while(at<receives&&s->steps[received[at]].first<s->steps[k].first)at++;
    if(at==receives||received[at]>=k||s->steps[received[at]].first!=s->steps[k].first||
       s->steps[received[at]].piece.elements!=s->steps[k].piece.elements) {
      steps_free(s);free(transfers);free(received);return fail(ncclInternalError,"WRITE relay tile has no matching incoming interval");
    }
    s->prior[s->priors++]=received[at];
  }
  s->prior_first[s->count] = s->priors;
  free(received); free(transfers);
  c->bound = s;
  return ncclSuccess;
}

/* whether rank r's buffer b of call c is the communicator's (bind_slot): scratch, relay generations, and an operand
   that is no result in place (a reduce-scatter's, a rank's that receives nothing) */
static int rank_member(const uint64_t *mask, uint32_t rank);
static int generation_buffer(const struct call *c, uint32_t r, uint32_t b) {
  if (counted(c)) return b >= 2;
  return b || c->what == MESH_REDUCE_SCATTER || !(c->what == MESH_REDUCE ? (int)r == c->root : rank_member(c->recipients, r));
}

/* Every rank derives the complete transaction graph before selecting its own
   physical steps. Ordinary collectives use exactly the same path table and
   fragment allocator as explicit WRITEs, including paths through ranks that
   neither send nor receive that logical transaction. */
static ncclResult_t bind_routed_steps(struct call *c, const uint32_t *first, const struct mesh_step *logical,
                                       const uint32_t *initialized, int routed) {
  struct ncclComm *comm = c->comm;
  struct mesh_routed_program physical = {0};
  struct mesh_paths *composed = NULL;
  int status = routed ? mesh_write_routes_compose(&comm->fabric, comm->writes, comm->write_overrides, &composed) : 0;
  if (!status) status = mesh_routed_compile(&comm->fabric, composed ? composed : comm->writes,
                                           first, logical, initialized, comm->page, &physical);
  mesh_write_routes_free(composed, comm->fabric.nodes);
  ncclResult_t result = status ? fail(ncclInvalidUsage, "physical collective lowering for %s: %s", comm->name, strerror(status)) : ncclSuccess;
  if (!result) {
    struct mesh_routed_rank *rank = physical.rank + comm->rank;
    const size_t room = (size_t)rank->count + 1;
    struct steps *s = steps_make(room, (struct steps){.references = 1, .count = rank->count, .buffers = rank->buffers, .initialized = rank->initialized});
    uint32_t *consume = rank->consume ? realloc(rank->consume, room * sizeof *consume) : NULL;
    if (consume) rank->consume = consume;
    uint32_t *retired_first = rank->retired_first ? realloc(rank->retired_first, room * sizeof *retired_first) : NULL;
    if (retired_first) rank->retired_first = retired_first;
    uint64_t *extent = realloc(rank->extent, ((size_t)rank->buffers + 1) * sizeof *extent);
    if (extent) rank->extent = extent;
    if (!s || !extent || (rank->consume && !consume) || (rank->retired_first && !retired_first)) {
      steps_free(s); result = fail(ncclSystemError, "out of memory");
    } else {
      s->extent = rank->extent; rank->extent = NULL;
      s->consume_order = rank->consume; rank->consume = NULL;
      s->retired_first = rank->retired_first; rank->retired_first = NULL;
      s->retired = rank->retired; rank->retired = NULL;
      s->demand = physical.demand; physical.demand = NULL;
      s->route_units = physical.route_units; physical.route_units = NULL;
      for (uint32_t k = 0; k < s->count; k++) { s->steps[k] = rank->transfer[k].step; s->cable[k] = rank->transfer[k].cable; }
      c->bound = s;
    }
  }
  mesh_routed_free(&physical);
  return result;
}

/* the operand's elements, in order, at which some rank's declared binding crosses a registration extent (c->phase):
   a binding's bytes run from its operand element `base` (an all-gather's send buffer and a reduce-scatter's receive
   buffer hold their rank's segment alone, of `cut` elements a rank; other bindings the whole operand), and a cut
   lies where an extent's boundary falls on an element; their count, at most `room` */
static size_t extent_cuts(const struct call *c, struct mesh_operand operand, const uint64_t *cut, uint64_t *out, size_t room) {
  const uint32_t N = (uint32_t)c->comm->nranks;
  const uint64_t z = operand.element_bytes;
  size_t n = 0;
  for (uint32_t r = 0; r < N; r++)
    for (int j = 0; j < 2; j++) {
      const uint64_t phase = c->phase[2 * r + j];
      if (phase == UINT64_MAX) continue;
      const int segment = cut && ((!j && c->what == MESH_ALLGATHER) || (j && c->what == MESH_REDUCE_SCATTER));
      uint64_t first = 0, length = operand.elements;
      for (uint32_t q = 0; segment && q < r; q++) first += cut[q];
      if (segment) length = cut[r];
      for (uint64_t at = MESH_DMA_EXTENT - phase; at < length * z && n < room; at += MESH_DMA_EXTENT)
        if (!(at % z) && first + at / z < operand.elements) out[n++] = first + at / z;
    }
  for (size_t i = 1; i < n; i++)
    for (size_t k = i; k && out[k - 1] > out[k]; k--) { const uint64_t t = out[k]; out[k] = out[k - 1]; out[k - 1] = t; }
  size_t kept = 0;
  for (size_t i = 0; i < n; i++) if (!kept || out[kept - 1] != out[i]) out[kept++] = out[i];
  return kept;
}

static ncclResult_t routed_steps(struct call *c, const struct mesh_program *program,
                                  struct mesh_operand operand, const uint64_t *cut) {
  const uint32_t N = (uint32_t)c->comm->nranks;
  uint32_t *first = calloc((size_t)N + 1, sizeof *first), *initialized = malloc(N * sizeof *initialized);
  struct mesh_step *logical = calloc((size_t)program->first[N] * program->parts + 1, sizeof *logical);
  ncclResult_t result = first && initialized && logical ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (uint32_t r = 0; r < N && !result; r++) {
    initialized[r] = 1 + program->scratch;
    first[r + 1] = first[r] + mesh_program_steps_cut(program, r, operand, cut, c->comm->page, logical + first[r]);
  }
  /* every rank's steps cut alike where some rank's declared binding crosses a registration extent, so no piece, and
     no DMA unit of one, spans an extent boundary at any rank (nccl.h ncclMeshCall bindings) */
  const size_t room = 2 * (size_t)N * (operand.elements * operand.element_bytes / MESH_DMA_EXTENT + 1);
  uint64_t *cuts = !result && c->phase ? malloc(room * sizeof *cuts) : NULL;
  if (!result && c->phase && !cuts) result = fail(ncclSystemError, "out of memory");
  const size_t n = cuts ? extent_cuts(c, operand, cut, cuts, room) : 0;
  struct mesh_step *split = n ? calloc((size_t)first[N] * (n + 1) + 1, sizeof *split) : NULL;
  if (!result && n && !split) result = fail(ncclSystemError, "out of memory");
  for (uint32_t r = 0, at = 0, k = 0; split && !result && r < N; r++) {
    for (; k < first[r + 1]; k++) {
      const struct mesh_step s = logical[k];
      uint64_t lo = s.first;
      for (size_t i = 0; i < n; i++)
        if (cuts[i] > lo && cuts[i] < s.first + s.piece.elements) {
          split[at] = s; split[at].first = lo; split[at++].piece.elements = cuts[i] - lo; lo = cuts[i];
        }
      split[at] = s; split[at].first = lo; split[at++].piece.elements = s.first + s.piece.elements - lo;
    }
    first[r + 1] = at;
  }
  if (!result) result = bind_routed_steps(c, first, split ? split : logical, initialized, 1);
  free(first); free(initialized); free(logical); free(split); free(cuts);
  return result;
}

static ncclResult_t counted_steps(struct call *c) {
  if (!c->matrix) return fail(ncclInvalidUsage, "prepare the complete AlltoAllV count graph with ncclMeshCommPlan before execution");
  const uint32_t N = (uint32_t)c->comm->nranks;
  uint32_t *first = calloc((size_t)N + 1, sizeof *first), *initialized = malloc(N * sizeof *initialized);
  uint64_t *segments = malloc(2 * (size_t)N * sizeof *segments);
  struct mesh_step *logical = calloc(2 * (size_t)N * N + 1, sizeof *logical);
  ncclResult_t result = first && initialized && segments && logical ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (uint32_t r = 0; r < N && !result; r++) {
    initialized[r] = 2;
    for (uint32_t q = 0; q < N; q++) { segments[q] = c->matrix[(size_t)r * N + q]; segments[N + q] = c->matrix[(size_t)q * N + r]; }
    first[r + 1] = first[r] + mesh_counted_steps(N, r, segments, (struct mesh_operand){(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], 0},
                                                logical + first[r]);
  }
  /* each transfer cut alike at its two ends where the sender's send binding or the receiver's receive binding crosses
     a registration extent (c->phase), its pieces' offsets in each buffer differing (nccl.h ncclMeshCall bindings) */
  struct mesh_step *split = NULL;
  uint32_t *at = NULL;
  if (!result && c->phase) {
    size_t room = 0;
    for (uint32_t k = 0; k < first[N]; k++) room += 1 + 4 * (logical[k].piece.elements * SIZE[c->type] / MESH_DMA_EXTENT + 1);
    split = calloc(room + 1, sizeof *split); at = calloc((size_t)N + 1, sizeof *at);
    if (!split || !at) result = fail(ncclSystemError, "out of memory");
  }
  for (uint32_t r = 0, made = 0, k = 0; split && !result && r < N; at[++r] = made)
    for (; k < first[r + 1]; k++) {
      const struct mesh_step s = logical[k];
      const uint32_t q = s.peer, from = s.op == MESH_STEP_SEND ? r : q, to = s.op == MESH_STEP_SEND ? q : r;
      /* the transfer's first element in its sender's send buffer and in its receiver's receive buffer */
      uint64_t sent = 0, landed = 0;
      for (uint32_t p = 0; p < to; p++) sent += c->matrix[(size_t)from * N + p];
      for (uint32_t p = 0; p < from; p++) landed += c->matrix[(size_t)p * N + to];
      uint64_t cuts[64];
      size_t n = 0;
      const uint64_t ends[2][3] = {{c->phase[2 * from], sent}, {c->phase[2 * to + 1], landed}};
      for (int e = 0; e < 2; e++) {
        if (ends[e][0] == UINT64_MAX) continue;
        for (uint64_t b = MESH_DMA_EXTENT - ends[e][0]; n < sizeof cuts / sizeof *cuts; b += MESH_DMA_EXTENT) {
          if (b % SIZE[c->type]) { if (b / SIZE[c->type] >= ends[e][1] + s.piece.elements) break; continue; }
          const uint64_t x = b / SIZE[c->type];
          if (x >= ends[e][1] + s.piece.elements) break;
          if (x > ends[e][1]) cuts[n++] = x - ends[e][1];
        }
      }
      for (size_t i = 1; i < n; i++)
        for (size_t j = i; j && cuts[j - 1] > cuts[j]; j--) { const uint64_t t = cuts[j]; cuts[j] = cuts[j - 1]; cuts[j - 1] = t; }
      uint64_t lo = 0;
      for (size_t i = 0; i < n; i++)
        if (cuts[i] > lo) { split[made] = s; split[made].first = s.first + lo; split[made++].piece.elements = cuts[i] - lo; lo = cuts[i]; }
      split[made] = s; split[made].first = s.first + lo; split[made++].piece.elements = s.piece.elements - lo;
    }
  if (!result) result = bind_routed_steps(c, split ? at : first, split ? split : logical, initialized, 0);
  free(first); free(initialized); free(segments); free(logical); free(split); free(at);
  return result;
}

/* Reuse the selected form's exact tree weights and phase structure. A caller's
   explicit program already specifies its contribution graph and is lowered
   unchanged; generated forms are specialized to the declared operand roles. */
static ncclResult_t collective_steps(struct call *c, const struct entry *entry,
                                      struct mesh_operand operand, const uint64_t *cut) {
  struct mesh_program subset = {0};
  const struct mesh_program *program = &entry->program;
  ncclResult_t result = ncclSuccess;
  if ((c->contributors || c->recipients) && entry->trees.count) {
    uint32_t parts, moves;
    mesh_compile_room(&entry->trees, &parts, &moves);
    uint32_t *segment = calloc((size_t)parts + 1, sizeof *segment);
    double *weight = calloc((size_t)parts + 1, sizeof *weight);
    uint32_t *first = calloc((size_t)c->comm->nranks + 1, sizeof *first);
    struct mesh_move *move = calloc((size_t)moves + 1, sizeof *move);
    if (!segment || !weight || !first || !move) result = fail(ncclSystemError, "out of memory");
    else if (mesh_compile_members(&entry->trees, (uint32_t)c->what, (uint32_t)c->root, entry->whole,
                                  c->contributors, c->recipients, &subset, segment, weight, first, move))
      result = fail(ncclInvalidArgument, "the declared contributors and recipients do not form %s", call_name(c));
    if (result) { free(segment); free(weight); free(first); free(move); return result; }
    program = &subset;
  }
  if (cut) {
    uint32_t segments = program->segments ? program->segments : 1;
    for (uint32_t p = 0; p < program->parts; p++) if (program->segment[p] + 1 > segments) segments = program->segment[p] + 1;
    if (segments != (uint32_t)c->comm->nranks)
      result = fail(ncclInvalidUsage, "a counted collective's program must retain every physical rank's segment index");
  }
  if (!result) {
    const int status = mesh_program_check_members(program, operand, cut, c->comm->page, c->contributors, c->recipients);
    if (status) result = fail(ncclInvalidArgument, "the declared %s reads an absent input or leaves a required result undefined: %s",
                              call_name(c), strerror(status));
  }
  if (!result) result = routed_steps(c, program, operand, cut);
  program_free(&subset);
  return result;
}

/* a reduce-scatter's own segment combined in recvbuff itself (buffer 0: result_of), the others' partials in a buffer
   of the communicator's (`partial`): no step spans two segments (mesh_program_steps_cut), so no result is copied */
static ncclResult_t partial_split(struct call *c) {
  struct steps *s = c->bound;
  size_t length;
  const uint64_t u = unit_of(c), before = segment_at(c, c->comm->rank, &length) / u, end = before + length / u;
  uint64_t *extent = realloc(s->extent, ((size_t)s->buffers + 1) * sizeof *extent);
  if (!extent) return fail(ncclSystemError, "out of memory");
  (s->extent = extent)[s->partial = s->buffers++] = elements(c) * SIZE[c->type];
  for (uint32_t k = 0; k < s->count; k++)
    if (!s->steps[k].buffer && (s->steps[k].first < before || s->steps[k].first >= end)) s->steps[k].buffer = s->partial;
  return ncclSuccess;
}

/* every call that crosses a link planned: a collective along the communicator's table (comm_compile), its steps cut on
   the communicator's page (mesh-plan.h mesh_program_steps_cut: the program's own cuts on pages; a reduce-scatter's and
   an all-gather's segments NCCL's, a rank's count each or its counts, so its program's segments must be the ranks), its
   operand's element a merge's row where it merges; an all-to-all to and
   from each other rank (mesh_counted_steps). Ordinary and counted collectives
   share the explicit WRITE physical route compiler. */
static ncclResult_t plan_calls(ncclComm_t comm, struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    if (!crosses(c)) continue;
    if (c->target) { const ncclResult_t result = write_steps(c); if (result) return result; continue; }
    if (counted(c)) { const ncclResult_t result = counted_steps(c); if (result) return result; continue; }
    const ncclResult_t compiled = comm_compile(comm);
    if (compiled) return compiled;
    const struct entry *entry = program_of(comm, c->what, c->root, elements(c) * SIZE[c->type]);
    if (!entry) return fail(ncclInvalidUsage, "the communicator has no program for collective %d at root %d", c->what, c->root);
    const int partitioned = c->what == MESH_REDUCE_SCATTER || c->what == MESH_ALLGATHER;
    const uint32_t u = unit_of(c);
    const struct mesh_operand operand = {(uint32_t)c->type + 1, (uint32_t)(SIZE[c->type] * u), elements(c) / u};
    uint64_t cut[partitioned ? comm->nranks : 1];
    for (int r = 0; partitioned && r < comm->nranks; r++) cut[r] = (c->segments ? c->segments[r] : c->count) / u;
    ncclResult_t result = collective_steps(c, entry, operand, partitioned ? cut : NULL);
    if (!result && c->what == MESH_REDUCE_SCATTER) result = partial_split(c);
    if (result) return result;
  }
  return ncclSuccess;
}

/* a step's entry on its channel: its call and step, its entry in its list, its pages (pages_of its bytes) and the
   value its list's count reaches once it is whole (`end`), its piece's byte offset in its buffer and its bytes, where
   its bytes lie (a SEND's source, a COPY's destination, a combine's landing area: `data`) and the first of the rows
   they lie in (`row`); a landing's `need` (the send entries a combine into bytes an earlier SEND of its call on its
   channel reads waits to be published: that entry + 1, 0 none) */
struct message { struct call *c; uint32_t step; uint64_t entry; mesh_pages pages, end; uint64_t need; size_t offset, bytes;
                 struct where data; struct mesh_dma_span *spans; uint64_t table; double priority;
                 uint64_t listed; mesh_pages laid; /* its entry and end in its issue's own lists (issue_lay), schedule_laid's base */ };
/* a group's entries on one channel, each list in its order from its first (`first`: [0] the send list's, [1] the
   landing list's), its landing entries' gate (`word` a region offset, 0 none; `base` the value it reaches: a page
   count of the holding's `ready`, or the number of a group at the communicator's gate word: gate_of), and each list's
   pages a repeat (`each`: issue_lay) */
struct schedule { struct message *out, *in; size_t outs, ins; uint64_t first[2], word, base; mesh_pages each[2];
                  size_t *consume; /* consumer ordinal -> physical landing index */ };
/* whether a SEND entry waits on its holding's `ready` (a publication): not a write's with producer words, which name
   their own word */
static int needs_publication(const struct message *m) {
  return !(m->c->write_plan && m->c->write_plan->ready_value);
}
static struct message *consumption_at(const struct schedule *p, size_t at) {
  return p->in + (p->consume ? p->consume[at] : at);
}
struct consumption_order { size_t wire, repeat; const struct call *call; uint32_t order; };
static int consumption_order_compare(const void *a, const void *b) {
  const struct consumption_order *x = a, *y = b;
  if (x->repeat != y->repeat) return x->repeat < y->repeat ? -1 : 1;
  if (x->call != y->call) return x->call < y->call ? -1 : 1;
  if (x->order != y->order) return x->order < y->order ? -1 : 1;
  return (x->wire > y->wire) - (x->wire < y->wire);
}
/* Physical arrivals retain their wire prefixes. Consumption has its own AOT
   sequence, shared by host execution and the existing GPU cursor tracks. */
static ncclResult_t schedule_consumption(struct schedule *p, uint32_t repeats) {
  int reordered = 0;
  for (size_t k = 0; k < p->ins; k++) reordered |= p->in[k].c->bound->consume_order != NULL;
  if (!reordered) return ncclSuccess;
  struct consumption_order *order = malloc((p->ins + 1) * sizeof *order);
  p->consume = malloc((p->ins + 1) * sizeof *p->consume);
  if (!order || !p->consume) { free(order); return fail(ncclSystemError, "out of memory"); }
  const size_t each = p->ins / repeats;
  for (size_t k = 0; k < p->ins; k++) {
    const struct message *m = p->in + k;
    order[k] = (struct consumption_order){k, k / each, m->c,
      m->c->bound->consume_order ? m->c->bound->consume_order[m->step] : (uint32_t)(k % each)};
  }
  qsort(order, p->ins, sizeof *order, consumption_order_compare);
  for (size_t k = 0; k < p->ins; k++) p->consume[k] = order[k].wire;
  free(order);
  return ncclSuccess;
}

/* a group's calls as mesh-plan.h's group, which mesh_group_lay lays (the one layout both ends make): call i and
   its steps at rank r, steps[i * N + r] and count[i * N + r] (none where the rank's are not known) */
static void group_of(size_t n, uint32_t N, const struct mesh_step **steps, uint32_t *count,
                     struct mesh_group_call *calls, uint64_t page, struct mesh_group *group) {
  for (size_t i = 0; i < n; i++) calls[i] = (struct mesh_group_call){count + i * N, steps + i * N};
  *group = (struct mesh_group){N, (uint32_t)n, calls, page};
}

static int meet(const struct mesh_step *x, const struct mesh_step *y) {
  return x->buffer == y->buffer && x->first < y->first + y->piece.elements && y->first < x->first + x->piece.elements;
}

/* the receive whose bytes SEND step k sends on unchanged, page by page as they land (design/data-model.md section 9: a
   relay is the identity at a rank), or -1: the last receive before it that meets its piece, a COPY into the same
   buffer whose piece holds its own, from a page of it (step k not reading the send buffer in place) */
static int32_t forwards(const struct steps *s, uint32_t k, uint64_t page) {
  if (s->prior_first[k] == s->prior_first[k + 1] || (s->read && named(s->read[k]))) return -1;
  const uint32_t r = s->prior[s->prior_first[k + 1] - 1];
  const struct mesh_step *a = s->steps + r, *b = s->steps + k;
  return a->op == MESH_STEP_COPY && a->first <= b->first && b->first + b->piece.elements <= a->first + a->piece.elements &&
         (b->first - a->first) * b->piece.element_bytes % page == 0 ? (int32_t)r : -1;
}

/* each channel's entries for the group, each list from its 0 (mesh_group_lay: schedule_laid lays them from the last
   scheduled entry and page), the overlapping receives before each step kept, a receive with no bytes done */
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
  if (!result) group_of(n, N, steps, counts, group_calls, comm->page, &group);
  for (uint32_t h = 0; h < comm->count && !result; h++) {
    const struct channel *ch = comm->channels + h;
    struct schedule *p = plans + h;
    *p = (struct schedule){calloc(total, sizeof *p->out), calloc(total, sizeof *p->in), 0, 0, {0, 0}, 0, 0, {{0}, {0}}, NULL};
    if (!p->out || !p->in) { result = fail(ncclSystemError, "out of memory"); break; }
    for (int send = 1; send >= 0; send--) {
      uint32_t placed = 0;
      mesh_group_lay(&group, me, (uint32_t)ch->peer, send, 0, scratch, made, &placed);
      mesh_pages end = {0};
      for (uint32_t k = 0; k < placed; k++) {
        const struct mesh_place *x = &made[k].place;
        struct call *c = list + made[k].call;
        const struct mesh_step *step = c->bound->steps + x->step;
        if (c->bound->cable[x->step] != ch->cable) continue;
        const mesh_pages pages = c->write_plan ? c->write_plan->step_units[write_span_index(c->write_plan,x->step,c->invocation)]
                                               : pages_of(mesh_step_bytes(step), comm->page);
        end = pages_add(end, pages);
        const struct message m = {.c = c, .step = x->step, .entry = send ? p->outs : p->ins, .pages = pages, .end = end,
                                  .offset = step->first * step->piece.element_bytes, .bytes = mesh_step_bytes(step)};
        if (send) p->out[p->outs++] = m; else p->in[p->ins++] = m;
      }
    }
  }
  for (size_t i = 0; i < n && !result; i++) {
    struct steps *st = list[i].bound;
    if (st && !list[i].target) {
      for (uint32_t k = 0; k < st->count; k++) {
        st->prior_first[k] = st->priors;
        for (uint32_t j = 0; j < k; j++)
          st->priors += st->steps[j].op != MESH_STEP_SEND && meet(st->steps + j, st->steps + k);
      }
      st->prior_first[st->count] = st->priors;
      st->prior = malloc((st->priors + 1) * sizeof *st->prior);
      if (!st->prior) result = fail(ncclSystemError, "out of memory");
      for (uint32_t k = 0; !result && k < st->count; k++) {
        uint32_t at = st->prior_first[k];
        for (uint32_t j = 0; j < k; j++)
          if (st->steps[j].op != MESH_STEP_SEND && meet(st->steps + j, st->steps + k)) st->prior[at++] = j;
      }
    }
    for (uint32_t k = 0; st && !result && k < st->count; k++)
      if (st->steps[k].op != MESH_STEP_SEND && !mesh_step_bytes(st->steps + k)) st->done[k] = 1;
  }
  free(steps); free(counts); free(group_calls); free(made); free(scratch);
  return result;
}

static void schedules_free(const struct ncclComm *comm, struct schedule *plans) {
  for (uint32_t h = 0; plans && h < comm->count; h++) {
    free(plans[h].out); free(plans[h].in); free(plans[h].consume);
  }
  free(plans);
}

/* Immutable WRITE groups use the ordinal of the global AOT streaming schedule.
   Parent fragments precede their relays in that order, so queue order cannot
   create a dependency cycle. No runtime arbitration is introduced. */
static int write_order(const void *left, const void *right) {
  const struct message *a = left, *b = right;
  if (a->priority != b->priority) return a->priority < b->priority ? -1 : 1;
  const uint32_t ar = a->c->bound->steps[a->step].round, br = b->c->bound->steps[b->step].round;
  if (ar != br) return ar < br ? -1 : 1;
  if (a->c != b->c) return a->c < b->c ? -1 : 1;
  return a->offset < b->offset ? -1 : a->offset > b->offset;
}

static int all_writes(const struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) if (!list[i].target) return 0;
  return n != 0;
}

#include "mesh-write-order.h"
struct write_fragment { const struct call *call; uint32_t source, cable, hop; size_t offset; uint32_t task; };
static int write_fragment_order(const void *left,const void *right) {
  const struct write_fragment *a=left,*b=right;
  if(a->call!=b->call)return a->call<b->call?-1:1;
  if(a->offset!=b->offset)return a->offset<b->offset?-1:1;
  if(a->hop!=b->hop)return a->hop<b->hop?-1:1;
  if(a->source!=b->source)return a->source<b->source?-1:1;
  return a->cable<b->cable?-1:a->cable>b->cable;
}
static uint32_t write_fragment_find(const struct write_fragment *fragments,uint32_t count,struct write_fragment key) {
  uint32_t low=0,high=count;
  while(low<high){uint32_t mid=low+(high-low)/2;if(write_fragment_order(fragments+mid,&key)<0)low=mid+1;else high=mid;}
  return low;
}

static ncclResult_t order_writes(const struct ncclComm *comm, const struct call *list, size_t calls, struct schedule *plans) {
  size_t room = 0;
  for (size_t i = 0; i < calls; i++) for (int r = 0; r < comm->nranks; r++) {
    const struct call *c = list + i;
    const struct mesh_paths *paths = comm->writes + c->root * comm->nranks + c->target - 1;
    const int n = mesh_write_compile(&comm->fabric, (uint32_t)r, (uint32_t)c->root, (uint32_t)c->target - 1,
      paths, (struct mesh_operand){(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], c->count}, comm->page, c->tile_bytes / SIZE[c->type], NULL, 0);
    if (n < 0 || (size_t)n > UINT32_MAX - room) return fail(ncclInvalidUsage, "write schedule extent");
    room += (size_t)n;
  }
  struct mesh_write_task *tasks = calloc(room + 1, sizeof *tasks);
  struct write_fragment *fragments = calloc(room + 1, sizeof *fragments);
  struct mesh_transfer *transfers = calloc(room + 1, sizeof *transfers);
  double *capacity = calloc(2 * comm->fabric.cables + 1, sizeof *capacity);
  ncclResult_t result = ncclSuccess;
  if (!tasks || !fragments || !transfers || !capacity) { result = fail(ncclSystemError, "out of memory"); goto done; }
  for (uint32_t e = 0; e < comm->fabric.cables; e++)
    capacity[2 * e] = capacity[2 * e + 1] = comm->fabric.cable[e].bandwidth / 8.0;
  uint32_t count = 0;
  for (size_t i = 0; i < calls; i++) for (int r = 0; r < comm->nranks; r++) {
    const struct call *c = list + i;
    const struct mesh_paths *paths = comm->writes + c->root * comm->nranks + c->target - 1;
    const int n = mesh_write_compile(&comm->fabric, (uint32_t)r, (uint32_t)c->root, (uint32_t)c->target - 1,
      paths, (struct mesh_operand){(uint32_t)c->type + 1, (uint32_t)SIZE[c->type], c->count}, comm->page, c->tile_bytes / SIZE[c->type], transfers, (uint32_t)room);
    if (n < 0) { result = fail(ncclInternalError, "write schedule realization"); goto done; }
    for (int k = 0; k < n; k++) {
      const struct mesh_transfer *t = transfers + k;
      if (t->step.op != MESH_STEP_SEND) continue;
      const size_t tile = c->tile_bytes ? t->step.first * SIZE[c->type] / c->tile_bytes : 0;
      const ncclMeshProduction production = c->production ? c->production[tile] : (ncclMeshProduction){0};
      fragments[count] = (struct write_fragment){c, (uint32_t)r, t->cable, t->step.round, t->step.first * SIZE[c->type],count};
      tasks[count++] = (struct mesh_write_task){
        .arc = 2 * t->cable + ((uint32_t)r == comm->fabric.cable[t->cable].node[1]), .parent = UINT32_MAX,
        .bytes = (double)mesh_step_bytes(&t->step),
        .first_bytes = c->write_plan ? fmin((double)mesh_step_bytes(&t->step),
          (double)(write_packets(c->write_plan,c->invocation)->cuts[span_after(write_packets(c->write_plan,c->invocation), t->step.first * SIZE[c->type])] -
                   t->step.first * SIZE[c->type])) : fmin((double)mesh_step_bytes(&t->step), (double)comm->page),
        .release = production.seconds, .production = production.order};
    }
  }
  qsort(fragments,count,sizeof *fragments,write_fragment_order);
  for (uint32_t k = 0; k < count; k++) if (fragments[k].hop) {
    const struct write_fragment *f=fragments+k;
    const uint32_t j=write_fragment_find(fragments,count,(struct write_fragment){.call=f->call,.offset=f->offset,.hop=f->hop-1});
    if(j==count||fragments[j].call!=f->call||fragments[j].offset!=f->offset||fragments[j].hop+1!=f->hop) {
      result=fail(ncclInternalError,"write relay has no upstream fragment");goto done;
    }
    tasks[f->task].parent=fragments[j].task;
  }
  const int status = mesh_write_order(tasks, count, capacity, 2 * comm->fabric.cables, (double)comm->page);
  if (status) { result = fail(ncclInternalError, "write schedule: %s", strerror(status)); goto done; }
  for (uint32_t h = 0; h < comm->count; h++) for (int send = 1; send >= 0; send--) {
    struct message *items = send ? plans[h].out : plans[h].in;
    const size_t n = send ? plans[h].outs : plans[h].ins;
    for (size_t k = 0; k < n; k++) {
      const struct mesh_step *step = items[k].c->bound->steps + items[k].step;
      const uint32_t source = send ? (uint32_t)comm->rank : step->peer;
      const struct write_fragment key={items[k].c,source,comm->channels[h].cable,step->round,items[k].offset,0};
      const uint32_t j=write_fragment_find(fragments,count,key);
      if (j == count || write_fragment_order(fragments+j,&key)) { result = fail(ncclInternalError, "write entry has no global fragment"); goto done; }
      items[k].priority = tasks[fragments[j].task].order;
    }
    qsort(items, n, sizeof *items, write_order);
    mesh_pages end = {0};
    for (size_t k = 0; k < n; k++) { end = pages_add(end, items[k].pages); items[k].entry = k; items[k].end = end; }
  }
done:
  free(tasks); free(fragments); free(transfers); free(capacity);
  return result;
}

/* a group's calls compiled: each crossing call's steps (plan_calls) and the group's entries on each channel, each list
   from its 0 (schedule_group); *plans NULL where it failed */
static ncclResult_t compile_group(struct ncclComm *comm, struct call *list, size_t n, struct schedule **plans) {
  ncclResult_t result = plan_calls(comm, list, n);
  if (!result && !(*plans = calloc(comm->count + 1, sizeof **plans))) result = fail(ncclSystemError, "out of memory");
  if (!result) result = schedule_group(comm, list, n, *plans);
  if (!result && all_writes(list, n)) result = order_writes(comm, list, n, *plans);
  for (uint32_t h = 0; h < comm->count && !result; h++) result = schedule_consumption(*plans + h, 1);
  if (result) { schedules_free(comm, *plans); *plans = NULL; }
  return result;
}

/* Group definitions and a compact sequence of repeated blocks. Geometry is
   compiled once per definition; invocation operands are bound separately.  A definition's invocations take turns
   at GENERATIONS slots of storage (the plan's in-flight depth: bind_slot), `uses` of them bound so far */
enum { GENERATIONS = 2 };
/* what a group of a definition binds at its issue (issue_bind): its calls, its schedules laid `repeats` times, its
   calls' steps copies and its record in flight, made at its generation's first issue and laid and placed for that
   issue's binding, then taken again by each later issue that finds it let go (`held` 0: issue_let, as its group is
   done), laid and placed again only where the binding changed; a group whose generation's is still held makes one of
   its own, as does a group of a dropped definition's still held (`dropped`), freed when let go */
struct inflight;
/* each call's binding an issue is laid and placed for (issue_bind): its buffers, views and a write's instance, and
   what binding them made (its result, whether it reads its send buffer in place, its complement's ranges) */
/* (an issue owns its calls' counts, `segments`, and views, copied at each bind, so a call in flight names nothing of
   the thread that issued it: room_of's storage is the thread's next call's) */
struct bound { struct where send, recv, sum, *views[2]; uint64_t occurrence, instance, *segments; uint32_t complements; int direct, fused; };
struct issue { struct schedule *plans; struct steps **steps; struct inflight *group; struct call *calls; size_t n; uint32_t repeats; int held, dropped;
               /* laid and placed (`laid`) for a binding (`bound`, `stride`), in
                  slot `slot` (its storage at `base`, its landing area at `area`), its span tables a lease (`tables`) */
               int laid; size_t stride; uint32_t slot; struct where base, area; struct bound *bound; void *tables; };
struct compiled { struct call *calls; size_t n; uint64_t repeat; struct schedule *plans; uint32_t slot[GENERATIONS], uses;
                  struct issue *issue[GENERATIONS]; };
/* a definition dropped: its slots back to the communicator's, for the next definition under the same guard, and its
   issues freed (one held, once let go) */
static void calls_free(struct call *list, size_t n);
static void issue_free(struct ncclComm *comm, struct issue *u);
static void issue_let(struct ncclComm *comm, struct issue *u);
static void compiled_drop(struct ncclComm *comm, struct compiled *x) {
  for (uint32_t g = 0; g < GENERATIONS; g++) {
    if (x->slot[g] && comm->slots) comm->slots[x->slot[g] - 1].held = 0;
    if (x->issue[g] && x->issue[g]->held) x->issue[g]->dropped = 1;
    else issue_free(comm, x->issue[g]);
  }
  calls_free(x->calls, x->n); schedules_free(comm, x->plans);
  *x = (struct compiled){0};
}
/* the WRITE plans' own definitions (write_definitions) dropped, compiled on routes or a map that changed or a
   communicator that ends; a plan compiles them again at its next execution (definition_of) */
static void cache_drop(struct ncclComm *comm) {
  while (comm->compiled_writes) write_definitions_drop(comm, comm->compiled_writes);
}
struct plan {
  ncclMeshPlan view;
  struct compiled *group;
  ncclMeshPlanRun *order;
  uint64_t *ends, (*entries)[2];
  size_t runs;
};
static struct compiled *group_at(const struct plan *plan, uint64_t holder) {
  const uint64_t at=(holder-plan->view.first)%plan->view.issues;
  size_t low=0,high=plan->runs;
  while(low<high){size_t mid=low+(high-low)/2;if(plan->ends[mid]<=at)low=mid+1;else high=mid;}
  const ncclMeshPlanRun run=plan->order[low];
  return plan->group+run.first+(at-(low?plan->ends[low-1]:0))%run.count;
}

/* message k of list m (n a repeat) laid repeat after repeat, `stride` bytes a repeat (nccl.h ncclMeshFused repeats) */
static struct message repeat_of(const struct message *m, size_t n, size_t k, size_t stride, struct call *list, const struct call *calls) {
  struct message x = m[k % n];
  x.c = list + (x.c - calls);
  x.entry += k / n * n;
  x.offset += k / n * (x.c->write_plan ? x.c->write_plan->stride : stride);
  x.end = pages_add(x.end, (mesh_pages){k / n * m[n - 1].end.n});
  return x;
}

/* a declared call's fused stream's records (declared, plan_calls) */
static ncclMeshFused fused_declared;
/* the counts a call of its kind holds a rank each (an all-to-all's to each rank, then from each), where it holds any */
static size_t call_segments(const struct call *c) {
  const size_t n = (size_t)c->comm->nranks;
  return counted(c) ? 2 * n : c->what == MESH_ALLGATHER || c->what == MESH_REDUCE_SCATTER ? n : 0;
}
/* whether issued call c is the call declared as d: its kind, type, operator, root, target and counts (a call that
   holds none has equal ones, `count` each: enqueue_counts) */
static int declared_as(const struct call *c, const struct call *d) {
  if (c->what != d->what || c->type != d->type || c->op != d->op || c->root != d->root || c->target != d->target) return 0;
  if (!counted(d) && c->count != d->count) return 0;
  for (size_t k = 0, v = call_segments(d); k < v && (c->segments || d->segments); k++)
    if ((c->segments ? c->segments[k] : c->count) != (d->segments ? d->segments[k] : d->count)) return 0;
  return 1;
}
static ncclResult_t write_definitions(struct ncclMeshWritePlan *plan);

/* the compiled form of the group numbered `holder` (*group): its plan's group (group_at), else, with no plan, a WRITE
   alone's plan's own (write_definitions, compiled again where routes or the map changed since), each call's roles its
   declaration's (nothing planned at the call: nccl.h ncclMeshCommPlan).  A group that no declaration covers, or that
   is not the group declared at its place (its calls' count, kinds, types, operators, roots and counts: declared_as),
   ends the program with a warning (library_end): nothing compiles a group at its issue
   (design/eager-execution.md#what-follows).  Under group_lock */
static ncclResult_t definition_of(struct ncclComm *comm, struct call *list, size_t n, uint64_t holder, struct compiled **group) {
  struct compiled *x = comm->plan ? group_at(comm->plan, holder) : NULL;
  struct ncclMeshWritePlan *alone = n == 1 && !list[0].fused ? list[0].write_plan : NULL;
  ncclResult_t result = !x && alone && !alone->compiled ? write_definitions(alone) : ncclSuccess;
  if (!x && alone && alone->compiled) x = alone->compiled + list[0].invocation % alone->occurrences;
  if (!result && !x) {
    char why[MESH_DIAGNOSTIC_BYTES];
    snprintf(why, sizeof why, "a group of %zu calls on communicator %s that no declaration covers (ncclMeshCommPlan, or a "
             "prepared WRITE alone)", n, comm->name);
    library_end(why);
  }
  if (!result && n != x->n) {
    char why[MESH_DIAGNOSTIC_BYTES];
    snprintf(why, sizeof why, "a group of %zu calls on communicator %s issued at the place of a declared group of %zu", n, comm->name, x->n);
    library_end(why);
  }
  for (size_t i = 0; !result && i < n; i++) {
    const struct call *c = list + i, *d = x->calls + i;
    if (!declared_as(c, d)) {
      char why[MESH_DIAGNOSTIC_BYTES];
      snprintf(why, sizeof why, "call %zu of a group on communicator %s is not the call declared at its place (kind %d/%d, type %d/%d, "
               "op %d/%d, root %d/%d, target %d/%d, count %zu/%zu, issued/declared, or their counts)", i, comm->name, c->what, d->what,
               (int)c->type, (int)d->type, (int)c->op, (int)d->op, c->root, d->root, c->target, d->target, c->count, d->count);
      library_end(why);
    }
    call_bind_roles(list + i, d);
  }
  *group = x;
  return result;
}
/* issue u's entries laid from group x's compiled ones `repeats` times (repeat_of), `stride` bytes a repeat, each
   message's call the issued call at its place in `list` and its entry and end in u's own lists (`listed`, `laid`:
   schedule_laid binds them on the channels) */
static struct issue *issue_make(struct ncclComm *comm, const struct compiled *x, uint32_t repeats);
static void issue_lay(struct ncclComm *comm, const struct compiled *x, struct issue *u, struct call *list, uint32_t repeats, size_t stride) {
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct schedule *p = x->plans + h;
    struct schedule *q = u->plans + h;
    *q = (struct schedule){q->out, q->in, p->outs * repeats, p->ins * repeats, {0, 0}, 0, 0, {p->outs ? p->out[p->outs - 1].end : p->each[0],
                           p->ins ? p->in[p->ins - 1].end : p->each[1]}, q->consume};
    for (size_t k = 0; k < q->outs; k++) q->out[k] = repeat_of(p->out, p->outs, k, stride, list, x->calls);
    for (size_t k = 0; k < q->ins; k++) q->in[k] = repeat_of(p->in, p->ins, k, stride, list, x->calls);
    for (int send = 1; send >= 0; send--) {
      struct message *items = send ? q->out : q->in;
      const size_t count = send ? q->outs : q->ins;
      mesh_pages end = {0};
      for (size_t k = 0; k < count; k++) {
        struct message *m = items + k;
        if (m->c->write_plan) m->pages = m->c->write_plan->step_units[write_span_index(m->c->write_plan,m->step,m->c->invocation)];
        end = pages_add(end, m->pages); m->end = end;
        m->listed = m->entry; m->laid = m->end;
        if (k + 1 == count / repeats) q->each[!send] = end;
      }
    }
    for (size_t k = 0; p->consume && k < q->ins; k++) q->consume[k] = k / p->ins * p->ins + p->consume[k % p->ins];
  }
}

/* the group's entries bound on each channel from its last scheduled entry and page (issue_lay's from each list's 0:
   `listed`, `laid`), each step's channel and entry kept, at every issue; each landing that combines into bytes an earlier SEND of its call on its
   channel reads (mesh_compile's two-node direct exchange; a SEND that reads the send buffer in place reads none,
   plan_direct) with that SEND's publication entry (struct message `need`) */
static void schedule_laid(struct ncclComm *comm, struct schedule *plans) {
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct channel *ch = comm->channels + h;
    struct schedule *p = plans + h;
    p->first[0] = ch->sends; p->first[1] = ch->lands;
    for (int send = 1; send >= 0; send--)
      for (size_t k = 0; k < (send ? p->outs : p->ins); k++) {
        struct message *m = (send ? p->out : p->in) + k;
        m->entry = m->listed + p->first[!send];
        m->end = pages_add(send ? ch->out : ch->in, m->laid);
        m->need = 0;
        m->c->bound->channel[m->step] = h; m->c->bound->entry[m->step] = m->entry;
      }
    for (size_t k = 0; k < p->ins; k++) {
      const struct message *m = consumption_at(p, k);
      m->c->bound->consumption[m->step] = p->first[1] + k;
    }
    for (size_t k = 0; k < p->ins; k++) {
      struct message *m = p->in + k;
      const struct steps *st = m->c->bound;
      if (st->steps[m->step].op != MESH_STEP_REDUCE) continue;
      for (size_t j = 0; j < p->outs; j++) {
        const struct message *o = p->out + j;
        if (o->c != m->c || o->step >= m->step || (st->read && named(st->read[o->step])) || !meet(st->steps + o->step, st->steps + m->step)) continue;
        if (o->entry + 1 > m->need) m->need = o->entry + 1;
      }
    }
  }
}

static size_t page_up(size_t bytes, uint64_t page) { return (size_t)((bytes + page - 1) / page * page); }

/* *items (a list of `size`-byte items, *room of them) with room for `count` + 1, its room doubled until it has:
   a list kept from call to call grows only to sizes it has not had; 0, or ENOMEM */
static int grow(void *items, size_t *room, size_t count, size_t size) {
  void **at = items;
  if (count < *room) return 0;
  size_t more = *room ? 2 * *room : 8;
  while (more <= count && more < SIZE_MAX / 2) more *= 2;
  void *made = more > SIZE_MAX / size ? NULL : realloc(*at, more * size);
  if (!made) return ENOMEM;
  *at = made; *room = more;
  return 0;
}

/* design/transport.md section 14 (region only): the bytes the GPU reads once a program has encoded them (present_group's
   entries; nccl-mesh-metal.m's tracks and argument records), carved from leases of the region, each a piece with its
   Metal buffer, made once and kept, so no Metal buffer is made at a call.  The carving piece is cut in order while it
   has room; a piece is taken again once every use cut from it is done (`done`: each of a use's `n` words at least
   `value`: the presented count past its entries, its tracks' ends) and none holds it whole (`held`: an argument
   records' piece, held until its residency set ends: metal_piece_give); a new one is leased only where none free is
   large enough.  Under group_lock (a give, from any thread, stores `held` alone) */
struct use { const _Atomic uint64_t *words; uint32_t n; uint64_t value; };
struct piece { void *address, *buffer; size_t bytes, used, uses, room; _Atomic int held; struct use *done; };
static struct piece **pieces, *carving;
static size_t piece_count, piece_room;
static int piece_free(const struct piece *p) {
  if (atomic_load_explicit(&p->held, memory_order_acquire)) return 0;
  for (size_t u = 0; u < p->uses; u++)
    for (uint32_t k = 0; k < p->done[u].n; k++)
      if (!p->done[u].words || atomic_load_explicit(p->done[u].words + k, memory_order_acquire) < p->done[u].value) return 0;
  return 1;
}
static int lease_buffer(const void *ptr, void **buffer, size_t *offset);
/* a free piece of at least `bytes` (the least such), else one leased; NULL where the region cannot meet it */
static struct piece *piece_take(size_t bytes) {
  struct piece *best = NULL;
  for (size_t k = 0; k < piece_count; k++)
    if (pieces[k] != carving && pieces[k]->bytes >= bytes && (!best || pieces[k]->bytes < best->bytes) && piece_free(pieces[k])) best = pieces[k];
  size_t offset;
  if (!best && (best = calloc(1, sizeof *best)) &&
      (grow(&pieces, &piece_room, piece_count, sizeof *pieces) || ncclMemAlloc(&best->address, bytes > MESH_COMMAND_DATA_BYTES ? bytes : MESH_COMMAND_DATA_BYTES) ||
       lease_buffer(best->address, &best->buffer, &offset))) {
    ncclMemFree(best->address); free(best); best = NULL;
  } else if (best && !best->bytes) {
    best->bytes = bytes > MESH_COMMAND_DATA_BYTES ? bytes : MESH_COMMAND_DATA_BYTES;
    pieces[piece_count++] = best;
  }
  if (best) best->used = best->uses = 0;
  return best;
}
/* nccl-mesh-metal.h metal_carve, metal_done, metal_piece_hold, metal_piece_give */
int metal_carve(size_t bytes, struct metal_bytes *made) {
  bytes = (bytes + 255) & ~(size_t)255;
  if ((!carving || carving->used + bytes > carving->bytes) && !(carving = piece_take(bytes))) return ENOMEM;
  if (grow(&carving->done, &carving->room, carving->uses, sizeof *carving->done)) return ENOMEM;
  carving->done[carving->uses++] = (struct use){NULL, 1, 0};
  *made = (struct metal_bytes){carving->buffer, carving->used, (char *)carving->address + carving->used};
  carving->used += bytes;
  return 0;
}
void metal_done(const _Atomic uint64_t *words, uint32_t n, uint64_t value) { carving->done[carving->uses - 1] = (struct use){words, n, value}; }
void *metal_piece_hold(struct metal_bytes *made, size_t *bytes) {
  struct piece *p = piece_take(MESH_COMMAND_DATA_BYTES);
  if (!p) return NULL;
  atomic_store_explicit(&p->held, 1, memory_order_release);
  *made = (struct metal_bytes){p->buffer, 0, p->address};
  *bytes = p->bytes;
  return p;
}
void metal_piece_give(void *piece) { atomic_store_explicit(&((struct piece *)piece)->held, 0, memory_order_release); }

/* a free slot of at least `bytes` (the least such), else a new one (ncclMemAlloc) of `bytes` rounded up to one of eight
   sizes an octave (at most an eighth more: so a slot made for one definition fits others, and the slots made of each
   size are never more than are held at once, whatever sizes the definitions name), held; its number + 1, 0 where the
   region cannot meet it (the failure formatted).  A free slot too small is returned to the region (ncclMemFree), its
   entry vacant for the next, once no GPU work of the library's can still reach it: its last group the host path's,
   every landing awaited (`waited`); or its last group a Metal path's (`ticket`), whose stream has stored that ticket in
   the slot's word of the communicator's failure words' page (word 8 + its index), as the stream does once it has run
   past that group where its generation let the slot go (launch_metal); else at ncclCommDestroy.  Under group_lock */
static uint32_t slot_take(struct ncclComm *comm, size_t bytes) {
  size_t step = comm->page ? comm->page : 1;
  if (bytes > SIZE_MAX / 4) { fail(ncclSystemError, "a slot of %zu bytes", bytes); return 0; }
  bytes = bytes ? bytes : 1;
  while (step <= (bytes - 1) / 8) step *= 2;
  bytes = (bytes + step - 1) / step * step;
  uint32_t best = 0, vacant = 0;
  for (uint32_t k = 0; k < comm->slot_count; k++) {
    struct slot *s = comm->slots + k;
    const int released = s->ticket && 8 + k < (size_t)getpagesize() / 8 && atomic_load_explicit(comm->words + 8 + k, memory_order_acquire) == s->ticket;
    int done = !s->held && (released || !s->ticket);
    for (uint32_t h = 0; done && !released && h < comm->count; h++) done = comm->channels[h].waited >= s->lands[h];
    if (!s->held && s->bytes >= bytes) { if (!best || s->bytes < comm->slots[best - 1].bytes) best = k + 1; }
    else if (done) { ncclMemFree(s->address); s->address = NULL; s->bytes = 0; vacant = k + 1; }
  }
  if (!best && !vacant) {
    struct slot *grown = realloc(comm->slots, (comm->slot_count + 1) * sizeof *grown);
    struct slot made = {NULL, 0, 0, calloc(comm->count + 1, sizeof *made.lands), calloc(comm->count + 1, sizeof *made.in), 0};
    if (grown) comm->slots = grown;
    if (!grown || !made.lands || !made.in) { free(made.lands); free(made.in); fail(ncclSystemError, "out of memory"); return 0; }
    comm->slots[comm->slot_count] = made;
    vacant = ++comm->slot_count;
  }
  /* a slot that fits one registration extent lies within one, as every lease that does (mesh-dataflow.c
     mesh_arena_alloc) */
  if (!best) {
    struct slot *s = comm->slots + vacant - 1;
    if (ncclMemAlloc(&s->address, bytes)) { s->address = NULL; return 0; }
    s->bytes = bytes; best = vacant;
  }
  comm->slots[best - 1].held = 1;
  return best;
}

/* group x's storage for `list` (its calls: x's own at its declaration, an issued group's at its issue), a page each: its
   calls' buffers that are the communicator's (generation_buffer, read on x's own calls: an issued call keeps its roles
   alone, never its declaration's recipients; a write's none), each laid from *base where given (st->buffer), then its
   landing area (*area, where given), a page for each combine's landing of `plans`; its bytes */
static size_t slot_laid(const struct ncclComm *comm, const struct compiled *x, const struct call *list, size_t n,
                        const struct schedule *plans, const struct where *base, struct where *area) {
  size_t used = 0;
  for (size_t i = 0; i < n; i++)
    for (uint32_t b = 0, z = (uint32_t)SIZE[list[i].type]; list[i].bound && b < list[i].bound->buffers; b++) {
      struct steps *st = list[i].bound;
      if (list[i].target || !generation_buffer(x->calls + i, (uint32_t)comm->rank, b) || (!b && st->partial)) continue;
      if (base) st->buffer[b] = shifted(*base, used);
      used += page_up(b < st->initialized ? elements(list + i) * z : st->extent[b], comm->page);
    }
  if (base && area) *area = shifted(*base, used);
  for (uint32_t h = 0; h < comm->count; h++)
    for (size_t k = 0; k < plans[h].ins; k++)
      if (plans[h].in[k].c->bound->steps[plans[h].in[k].step].op == MESH_STEP_REDUCE) used += page_up(plans[h].in[k].bytes, comm->page);
  return used;
}
/* group x's storage (slot_laid) for generation k in x's slot k (*slot, its storage at *base, its lease's Metal buffer):
   taken at x's declaration (ncclMeshCommPlan) for each of GENERATIONS, or a WRITE plan's at its first
   executions, and again where it must grow.  Under group_lock */
static ncclResult_t bind_slot(struct ncclComm *comm, struct compiled *x, uint32_t k, struct call *list, size_t n, const struct schedule *plans,
                              struct where *base, struct where *area, uint32_t *slot) {
  const size_t need = slot_laid(comm, x, list, n, plans, NULL, NULL);
  uint32_t *held = x->slot + k;
  if (*held && comm->slots[*held - 1].bytes < need) { comm->slots[*held - 1].held = 0; *held = 0; }
  if (!*held && !(*held = slot_take(comm, need))) return fail(ncclSystemError, "a group's storage: %s", last.text);
  void *address = comm->slots[(*slot = *held) - 1].address;
  *base = (struct where){NULL, (size_t)address};
  if (lease_buffer(address, &base->buffer, &base->offset)) return fail(ncclSystemError, "no Metal buffer over a lease of the communicator's");
  slot_laid(comm, x, list, n, plans, base, area);
  return ncclSuccess;
}


/* where each entry's pages lie (struct message `data`): a SEND's in the buffer its step reads (the send buffer, where
   it reads that in place), a COPY's in its step's buffer, a combine's in the group's landing area (`area`: bind_slot,
   each entry from a page of it); each in a live lease, its span table, a span a page, in its issue's lease of them
   (`tables`, made for its binding; the one it replaces let go, kept by the bridge's reads of the entries presented
   from it: present_group's marks).  0, else the call fails as a value, naming the piece (ncclInvalidUsage: the piece's
   buffer is no lease, or a span crosses a registration extent; nothing is copied to move it), nothing presented */
static ncclResult_t place_group(struct ncclComm *comm, struct issue *u, struct where area) {
  const unsigned char *region = (const unsigned char *)comm->attach->context.M;
  struct schedule *plans = u->plans;
  size_t used = 0, length = 0;
  const size_t align = MESH_RECORD_ALIGNMENT;
  for (int pass = 0; pass < 2; pass++) {
    if (pass) {
      if (u->tables) ncclMemFree(u->tables);
      u->tables = NULL;
      if (length && ncclMemAlloc(&u->tables, length)) return fail(ncclSystemError, "a group's span tables: %s", last.text);
      length = 0;
    }
    for (uint32_t h = 0; h < comm->count; h++)
      for (int send = 1; send >= 0; send--)
        for (size_t k = 0; k < (send ? plans[h].outs : plans[h].ins); k++) {
          struct message *x = (send ? plans[h].out : plans[h].in) + k;
          const struct steps *st = x->c->bound;
          const struct mesh_step *step = st->steps + x->step;
          const struct ncclMeshWritePlan *write = x->c->write_plan;
          if (!pass) {
            const size_t instance = write ? x->c->invocation % write->layout.instances : 0;
            const uint64_t displacement = write_displacement(write, x->offset) + (write ? instance * write->layout.instance_stride : 0);
            const struct where origin = send && st->read && named(st->read[x->step]) ? st->read[x->step] : st->buffer[step->buffer];
            if (!send && step->op == MESH_STEP_REDUCE) { x->data = shifted(area, used); used += page_up(x->bytes, comm->page); }
            else x->data = shifted(origin, displacement);
            if (counted(x->c) && step->buffer < 2) x->data = view_at(x->c, step->buffer == 0, x->offset);
            else if (x->c->receive_views && step->buffer == 0) x->data = view_at(x->c, 0, x->offset);
          }
          if (write) {
            const uint64_t logical = step->first * step->piece.element_bytes;
            const size_t repeat = write->stride ? (x->offset - logical) / write->stride : 0;
            x->spans = write->spans + write->span_first[write_span_index(write,x->step,x->c->invocation)] + repeat * x->pages.n;
            x->table = (uint64_t)((const unsigned char *)x->spans - region);
            continue;
          }
          const unsigned char *address = address_of(x->data);
          if (!pass) {
            pthread_mutex_lock(&attach_lock);
            const struct lease *held = in_region(comm, x->data, x->bytes) ? lease_at(address, NULL) : NULL;
            const int valid = held && !held->returned && x->bytes <= held->bytes - (size_t)(address - (const unsigned char *)held->address);
            pthread_mutex_unlock(&attach_lock);
            if (!valid) return fail(ncclInvalidUsage, "%s on %s: %zu bytes at byte %zu are outside a live local lease",
                                    call_name(x->c), comm->name, x->bytes, x->offset);
          }
          if (!x->pages.n) continue;
          if (x->pages.n > SIZE_MAX / 2 / sizeof *x->spans) return fail(ncclSystemError, "out of memory");
          struct mesh_dma_span *made = pass ? (struct mesh_dma_span *)((char *)u->tables + length) : NULL;
          length += (x->pages.n * sizeof *x->spans + align - 1) / align * align;
          if (!pass) continue;
          const uint64_t offset = (uint64_t)(address - region);
          for (uint64_t unit = 0, at = 0; unit < x->pages.n; unit++, at += comm->page) {
            made[unit] = (struct mesh_dma_span){offset + at, x->bytes - at < comm->page ? x->bytes - at : comm->page};
            if (!mesh_dma_span_valid(comm->attach->context.M, made[unit]))
              return fail(ncclInvalidUsage, "DMA span crosses a registration extent; prepare its common byte cuts before execution");
          }
          x->spans = made;
          x->table = (uint64_t)((unsigned char *)made - region);
        }
  }
  return ncclSuccess;
}

/* Bind only true overwrite hazards. Original-input bindings are disjoint from reduction results. A landing that writes
   bytes an earlier SEND of its call reads (a COPY landing in place, a combine into its buffer) takes the compiled proof
   that the SEND's bytes arrived first (mesh-routed.c: a causal witness of that arrival), else the call fails as a
   value, nothing presented: no wait reads `sent` (design/pages-and-functions.md, Freeing 6-7) */
static ncclResult_t bind_retirements(struct ncclComm *comm, const struct schedule *plans) {
  for (uint32_t h = 0; h < comm->count; h++) for (size_t k = 0; k < plans[h].ins; k++) {
    const struct message *m = plans[h].in + k;
    const struct steps *st = m->c->bound;
    const struct mesh_step *step = st->steps + m->step;
    /* Explicit WRITEs have one immutable generation per logical transaction;
       intermediate COPY/SEND pieces have identical, disjoint indexed views. */
    if (m->c->write_plan) continue;
    const struct where target = step->op == MESH_STEP_REDUCE ? shifted(st->buffer[step->buffer], m->offset) : m->data;
    const uintptr_t begin = (uintptr_t)address_of(target), end = begin + m->bytes;
    for (uint32_t q = 0; q < comm->count; q++) for (size_t j = 0; j < plans[q].outs; j++) {
      const struct message *o = plans[q].out + j;
      if (o->c != m->c || o->step >= m->step) continue;
      const uintptr_t first = (uintptr_t)address_of(o->data), last = first + o->bytes;
      if (first >= end || begin >= last) continue;
      int proven = 0;
      for (uint32_t at = st->retired_first ? st->retired_first[m->step] : 0;
           st->retired_first && at < st->retired_first[m->step + 1]; at++) proven |= st->retired[at] == o->step;
      if (!proven)
        return fail(ncclInvalidUsage, "%s step %u writes bytes SEND step %u reads, without a causal source-read witness; declare distinct "
                    "destination storage", step->op == MESH_STEP_REDUCE ? "the combine of" : "COPY", m->step, o->step);
    }
  }
  return ncclSuccess;
}

/* a Metal-path group's landing entries gated on its start in the stream (design/transport.md section 14: a landing
   entry's word and base), so no receive lands in pages the stream's earlier work still reads or writes (the caller's
   buffers; a slot an earlier group combines from): on each channel, the holding's `ready` past the first
   page of the group's first send entry there where that entry's step waits on no receive (published at the group's
   start, issue_group; one system fence before it), else the communicator's gate word at the group's value of it, which
   the group's first dispatch stores (*gate) */
static void gate_of(struct ncclComm *comm, struct schedule *plans, int *gate) {
  for (uint32_t h = 0; h < comm->count; h++) {
    struct schedule *p = plans + h;
    if (!p->ins) continue;
    const struct message *first = p->outs ? p->out : NULL;
    if (first && first->c->bound->prior_first[first->step] == first->c->bound->prior_first[first->step + 1] && needs_publication(first)) {
      p->word = comm->channels[h].ready_at;
      p->base = pages_to_word(pages_add(pages_sub(first->end, first->pages), (mesh_pages){1}));
      continue;
    }
    p->word = comm->gate_at; p->base = comm->gates + 1; *gate = 1;
  }
}

/* the communicators a program touched, each once, until its end (program_end); a program holds group_lock from its
   begin to its end, so the list is empty at every taking of the lock */
static struct ncclComm *touching;
static void touch(struct ncclComm *comm) {
  if (comm->touched) return;
  comm->touched = 1;
  comm->next_touched = touching;
  touching = comm;
}
/* each channel's entries presented by the program (metal_present: the GPU's present, its send list's and then its
   landing list's once the bridge has room for all of them, each list's entries presented and not taken fewer than
   MESH_STREAM_ENTRIES with them), so a group is presented whole or not at all; a send entry's bytes ready once `ready`
   passes them (word 0), a landing entry gated as gate_of says.  0; ncclSystemError where a list's entries are more than
   the bridge ever takes (nothing presented, the call's failure alone); a refused presentation fails comm (its lists
   stand mid-group).  Under group_lock */
static ncclResult_t present_group(struct metal_program *program, struct ncclComm *comm, struct schedule *plans) {
  struct mesh_ctx *context = &comm->attach->context;
  for (uint32_t h = 0; h < comm->count; h++)
    for (int list = 0; list < 2; list++) {
      const struct channel *ch = comm->channels + h;
      const uint64_t need = list ? plans[h].ins : plans[h].outs;
      if (need > MESH_STREAM_ENTRIES)
        return fail(ncclSystemError, "communicator %s: the group lays %llu entries on its %s list toward node %u, and its bridge takes at most %u "
                    "not yet done (mesh.h MESH_STREAM_ENTRIES); none was presented", comm->name, (unsigned long long)need, list ? "landing" : "send",
                    ch->node, MESH_STREAM_ENTRIES);
    }
  /* Retain payload AND immutable descriptor tables for both directions before
     presenting anything. Metadata additionally needs local bridge CPU-reader
     quiescence on cancellation; it cannot borrow the payload DMA-pin proof. */
  int retained = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t h = 0; h < comm->count && !retained; h++) {
    const struct channel *ch = comm->channels + h;
    const struct hold *hold = comm->attach->holds + comm->endpoint[ch->cable];
    for (int send = 1; send >= 0 && !retained; send--) {
      const size_t count = send ? plans[h].outs : plans[h].ins;
      const struct message *items = send ? plans[h].out : plans[h].in;
      for (size_t k = 0; k < count && !retained; k++) {
        const struct message *m = items + k;
        if (m->c->write_plan) {
          m->c->write_plan->uses[2 * h + !send].end = m->end.n;
          continue;
        }
        struct lease *payload = lease_at(address_of(m->data), NULL);
        struct lease *metadata = lease_at(m->spans, NULL);
        const unsigned direction = send ? MESH_LEASE_SEND : MESH_LEASE_LAND;
        retained = !payload || payload->returned || !metadata || metadata->returned ? EINVAL :
          mesh_lease_retire_mark_use(&payload->reads, ch->stream, hold->wire, hold->generation, m->end.n, direction, 0);
        if (!retained) retained = mesh_lease_retire_mark_use(&metadata->reads, ch->stream, hold->wire, hold->generation, m->end.n, direction, 1);
      }
    }
  }
  pthread_mutex_unlock(&attach_lock);
  if (retained) return comm_fail(comm, ncclSystemError, "registering DMA lifetimes: %s", strerror(retained));
  for (uint32_t h = 0; h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    const struct schedule *p = plans + h;
    int status = 0;
    if (!p->outs && !p->ins) continue;
    struct metal_bytes compiled = {0};
    if (metal_carve((p->outs + p->ins) * sizeof(struct mesh_entry), &compiled))
      return fail(ncclSystemError, "communicator %s: the region cannot hold a group's entries for its present", comm->name);
    struct mesh_entry *records = compiled.contents;

    for (size_t k = 0; k < p->outs; k++) {
      const struct message *m = p->out + k;
      struct mesh_entry entry = {m->table, pages_to_word(m->pages), 0, pages_to_word(pages_sub(m->end, m->pages))};
      const struct steps *s = m->c->bound;
      const int32_t dependency = m->c->target && comm->rank != m->c->root ? (int32_t)s->prior[s->prior_first[m->step]] : forwards(s, m->step, comm->page);
      const uint32_t repeats = m->c->fused && m->c->fused->repeats ? m->c->fused->repeats : 1;
      const size_t repeat = k / (p->outs / repeats);
      if (dependency >= 0) {
        const uint32_t incoming = s->channel[dependency];
        const struct schedule *received = plans + incoming;
        const struct message *from = received->in + s->entry[dependency] - received->first[1] -
                                     (repeats - 1 - repeat) * (received->ins / repeats);
        entry.word = (uint64_t)((char *)comm->channels[incoming].view.landed - (char *)context->M);
        const struct mesh_step *before = s->steps + dependency;
        const uint64_t delta = m->offset - from->offset;
        const mesh_pages units = m->c->write_plan ? (mesh_pages){span_units(write_packets(m->c->write_plan,m->c->invocation),
          before->first * before->piece.element_bytes, delta)} : pages_of(delta, comm->page);
        entry.base = pages_to_word(pages_add(pages_sub(from->end, from->pages), units));
      } else if (m->c->write_plan && m->c->write_plan->ready_value) {
        const struct ncclMeshWritePlan *plan = m->c->write_plan;
        entry.word = (uint64_t)(address_of(plan->source_ready) + tile_offset(plan->ready_layout, write_tile(plan, m->offset), m->c->invocation) - (const unsigned char *)context->M);
        entry.base = plan->ready_value - m->pages.n;
        metal_resident(program, plan->source_ready.buffer);
      }
      records[k] = entry;
    }
    for (size_t k = 0; k < p->ins; k++)
      records[p->outs + k] = (struct mesh_entry){p->in[k].table, pages_to_word(p->in[k].pages), p->word, p->base};
    if (!ch->lists) ch->lists = metal_wrap(ch->view.list[0], 2 * MESH_STREAM_ENTRIES * sizeof(struct mesh_entry));
    if (!ch->lists) return fail(ncclSystemError, "binding bridge descriptor lists");
    metal_resident(program, ch->lists); metal_resident(program, compiled.buffer);
    /* the records done with once the GPU has presented them all: the landing list's count past the group's */
    metal_done(ch->view.presented[p->ins ? 1 : 0], 1, p->first[p->ins ? 1 : 0] + (p->ins ? p->ins : p->outs));
    const char *words = (const char *)ch->view.ready - ch->ready;
    for (int list = 0; list < 2 && !status; list++) {
      const size_t count = list ? p->ins : p->outs;
      struct metal_result placed = metal_present(program, ch->lists, list * MESH_STREAM_ENTRIES * sizeof(struct mesh_entry),
        compiled.buffer, compiled.offset + (list ? p->outs * sizeof(struct mesh_entry) : 0), count, p->first[list], ch->buffer,
        (const char *)ch->view.taken[list] - words, (const char *)ch->view.presented[list] - words);
      status = placed.status;
    }
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
    if (p->outs) { ch->sends = p->first[0] + p->outs; ch->out = p->out[p->outs - 1].end; }
    if (p->ins) { ch->lands = p->first[1] + p->ins; ch->in = p->in[p->ins - 1].end; }
  }
}

/* the group's laid and placed entries (issue_bind) bound on each channel from its last scheduled entry and page
   (schedule_laid), its landings gated (gate_of, *gate whether its first dispatch stores the communicator's gate),
   presented (present_group) and counted (commit_group); a failure
   before any entry is presented is the call's alone, one after it the communicator's (its lists stand mid-group) */
static ncclResult_t schedule(struct metal_program *program, struct ncclComm *comm, struct schedule *plans, int *gate) {
  schedule_laid(comm, plans);
  gate_of(comm, plans, gate);
  ncclResult_t result = present_group(program, comm, plans);
  if (result) return result;
  if (*gate) comm->gates++;
  commit_group(comm, plans);
  return ncclSuccess;
}
/* a slot's last user: the group (`ticket`, 0 the host path's) and each channel's landing ends once it is presented */
static void slot_used(struct ncclComm *comm, uint32_t slot, uint64_t ticket) {
  struct slot *s = comm->slots + slot - 1;
  s->ticket = ticket;
  for (uint32_t h = 0; h < comm->count; h++) { s->lands[h] = comm->channels[h].lands; s->in[h] = comm->channels[h].in; }
}

/* design/whole-function.md#fabric: only overlapping earlier receives create data dependencies. */
static int ready(const struct steps *s, uint32_t k) {
  for (uint32_t at = s->prior_first[k]; at < s->prior_first[k + 1]; at++) if (!s->done[s->prior[at]]) return 0;
  return 1;
}


/* where a call's operand holds byte `at`: its result (c->sum: result_of), a reduce-scatter's partial of another's
   segment in its own buffer (partial_split) */
static struct where operand_at(const struct call *c, size_t at) {
  const struct steps *st = c->bound;
  size_t length;
  const size_t before = segment_at(c, c->comm->rank, &length) * SIZE[c->type];
  return st && st->partial && (at < before || at >= before + length * SIZE[c->type]) ? shifted(st->buffer[st->partial], at) : shifted(c->sum, at);
}
/* where a call that is not a write or an all-to-all holds its result: recvbuff where it holds the whole operand or
   crosses nothing, a reduce-scatter's recvbuff as its own segment of the operand (partial_split), else the
   communicator's (bind_slot) */
static struct where result_of(const struct call *c) {
  const struct steps *st = c->bound;
  size_t length;
  if (!st || (call_receives(c) && c->what != MESH_REDUCE_SCATTER)) return c->recv_at;
  return st->partial ? shifted(c->recv_at, (size_t)0 - segment_at(c, c->comm->rank, &length) * SIZE[c->type]) : st->buffer[0];
}

/* each call's buffers bound: a write's the producer's lease itself (not a broadcast's staging copy, which would run
   before a deferred producer makes data), an all-to-all's its own two, a collective's its result (result_of), read in
   place where it may (plan_direct) */
static int plan_direct(struct call *c, struct where own);
static void bind_calls(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    struct call *c = list + i;
    struct steps *st = c->bound;
    if (c->target) { c->sum = c->recv_at; c->direct = 1; }
    else if (!counted(c)) { c->sum = result_of(c); c->direct = plan_direct(c, c->sum); }
    if (st) st->buffer[0] = c->target ? (c->comm->rank == c->root ? c->send_at : c->recv_at) : counted(c) ? c->send_at : c->sum;
    if (st && counted(c)) st->buffer[1] = c->recv_at;
  }
}

/* whether issue u is laid and placed for the binding of `list` (struct bound): the same buffers and views, a write's
   same instance and occurrence, fused or not alike, at the same stride in the same slot */
static int issue_same(const struct ncclComm *comm, const struct issue *u, const struct call *list, size_t n, size_t stride, uint32_t slot) {
  if (!u->laid || u->stride != stride || u->slot != slot) return 0;
  for (size_t i = 0; i < n; i++) {
    const struct call *c = list + i;
    const struct bound *b = u->bound + i;
    const struct ncclMeshWritePlan *w = c->write_plan;
    if (!same(c->send_at, b->send) || !same(c->recv_at, b->recv) || !c->fused != !b->fused ||
        (w && (c->invocation % w->occurrences != b->occurrence || c->invocation % w->layout.instances != b->instance)))
      return 0;
    for (int j = 0; j < 2; j++) {
      const struct where *v = j ? c->receive_views : c->send_views;
      if (!v != !b->views[j] || (v && memcmp(v, b->views[j], (size_t)comm->nranks * sizeof *v))) return 0;
    }
  }
  return 1;
}
/* group x's `n` calls (*calls) bound in its next generation's issue (*issued; struct issue), *calls then the issue's copy
   of them (what they own moved there) and *plans its entries: an issue laid and placed for the same binding (issue_same)
   takes it again, its steps' progress made again (steps_rebind) and its calls' bound results restored, and its slot and
   landing area are the ones it was placed in; else it is laid for them (issue_lay, each call's steps that generation's
   copy made again), its storage bound (bind_slot, where it crosses: `remote`), its calls' buffers (bind_calls), its
   entries' bytes placed (place_group) and their overwrites proved (bind_retirements), and its binding kept.  Under
   group_lock */
static ncclResult_t issue_bind(struct ncclComm *comm, struct call **calls, size_t n, struct compiled *x, uint32_t repeats, size_t stride,
                               int remote, struct schedule **plans, struct issue **issued, struct where *area, uint32_t *slot) {
  const uint32_t k = x->uses++ % GENERATIONS;
  struct issue **at = x->issue + k, *u = *at;
  if (u && !u->held && u->repeats != repeats) { issue_free(comm, u); *at = u = NULL; }
  if (!u || u->held) {
    if (!(u = issue_make(comm, x, repeats))) return fail(ncclSystemError, "out of memory");
    if (*at) u->dropped = 1;
    else *at = u;
  }
  u->held = 1;
  struct call *const list = *calls = memcpy(u->calls, *calls, n * sizeof *u->calls);
  for (size_t i = 0; i < x->n; i++)
    if ((list[i].bound = u->steps[i])) atomic_fetch_add_explicit(&u->steps[i]->references, 1, memory_order_relaxed);
  for (size_t i = 0; i < n; i++) {
    const size_t v = call_segments(list + i);
    if (!list[i].segments || !v) continue;
    if (!u->bound[i].segments && !(u->bound[i].segments = malloc(v * sizeof *list[i].segments))) return fail(ncclSystemError, "out of memory");
    list[i].segments = memcpy(u->bound[i].segments, list[i].segments, v * sizeof *list[i].segments);
  }
  *plans = u->plans; *issued = u;
  if (issue_same(comm, u, list, n, stride, remote ? x->slot[k] : 0)) {
    for (size_t i = 0; i < n; i++) {
      const struct bound *b = u->bound + i;
      if (list[i].bound) steps_rebind(list[i].bound);
      list[i].sum = b->sum; list[i].direct = b->direct; list[i].complements = b->complements;
      list[i].complement = b->complements ? list[i].bound->complement : NULL;
      if (list[i].send_views) list[i].send_views = b->views[0];
      if (list[i].receive_views) list[i].receive_views = b->views[1];
    }
    *area = u->area; *slot = u->slot;
    return ncclSuccess;
  }
  u->laid = 0;
  for (size_t i = 0; i < n; i++) if (list[i].bound) steps_reset(list[i].bound);
  issue_lay(comm, x, u, list, repeats, stride);
  ncclResult_t result = remote ? bind_slot(comm, x, k, list, n, u->plans, &u->base, area, slot) : ncclSuccess;
  if (!result) bind_calls(list, n);
  if (!result && remote) result = place_group(comm, u, *area);
  if (!result && remote) result = bind_retirements(comm, u->plans);
  for (size_t i = 0; !result && i < n; i++) {
    const struct call *c = list + i;
    const struct ncclMeshWritePlan *w = c->write_plan;
    struct bound *b = u->bound + i;
    for (int j = 0; j < 2 && !result; j++) {
      const struct where *v = j ? c->receive_views : c->send_views;
      if (!v) { free(b->views[j]); b->views[j] = NULL; }
      else if (!b->views[j] && !(b->views[j] = malloc((size_t)comm->nranks * sizeof *v))) result = fail(ncclSystemError, "out of memory");
      else memcpy(b->views[j], v, (size_t)comm->nranks * sizeof *v);
    }
    if (!result && c->send_views) list[i].send_views = b->views[0];
    if (!result && c->receive_views) list[i].receive_views = b->views[1];
    b->send = c->send_at; b->recv = c->recv_at; b->sum = c->sum; b->direct = c->direct; b->complements = c->complements;
    b->fused = c->fused != NULL;
    b->occurrence = w ? c->invocation % w->occurrences : 0; b->instance = w ? c->invocation % w->layout.instances : 0;
  }
  if (!result) { u->laid = 1; u->stride = stride; u->slot = remote ? *slot : 0; u->area = *area; }
  return result;
}

/* an operator as the host runs it (premultiplier, combine, postoperation; the partials merge's row) */
static ncclResult_t reduction(struct call *c) {
  const ncclComm_t comm = c->comm;
  c->premultiplied = 0; c->post = 0;
  if ((int)c->op < ncclAvg) { c->combine = (int)c->op; return ncclSuccess; }
  c->combine = COMBINE_SUM;
  if (c->op == ncclAvg) {
    if (!floating(c->type)) c->post = 1;
    else { store(c->type, c->pre, 1.0 / call_participants(c)); c->premultiplied = 1; }
    return ncclSuccess;
  }
  const struct redop *made = comm->ops + ((int)c->op - ncclNumOps);
  /* NCCL's hostToDevRedOp refuses it alike ("Data type supplied to user-created ncclRedOp_t does not match"): a
     scalar of one type read as another's is a value no one gave (design/rubrics.md D2, the operator 10-19: "functions
     should throw ... before they try to guess parametric values"); and the merge's rows are float32 [m, l, O]
     (nccl-mesh-kernels.h merge_row), which another type's elements would be read and written past */
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
  *sent = mesh_segment_first(c->segments, (uint32_t)me) * SIZE[c->type]; *landed = mesh_segment_first(c->segments + n, (uint32_t)me) * SIZE[c->type];
  const uint64_t out = c->segments[me], in = c->segments[n + me];
  return (out < in ? out : in) * SIZE[c->type];
}

/* what a wait for a communicator's GPU work (gpu_wait) ended at: its last program's end (GPU_ENDED), its command buffer's
   error status (GPU_ERROR), that buffer not committed (GPU_UNCOMMITTED: nothing runs it while its committer waits), a
   revoke (GPU_REVOKED) or a holding's end (GPU_HOLDING) */
enum { GPU_ENDED, GPU_ERROR, GPU_UNCOMMITTED, GPU_REVOKED, GPU_HOLDING };
/* design/transport.md section 5: the wait for comm's last GPU program (its count `last` on `finished`, signalled at the
   end of its command buffer, `command`) with group_lock released for it, as a condition variable's wait releases its
   mutex (design/rubrics.md M3, M4: other communicators' groups and revokes go on meanwhile): the count and the buffer
   read under the lock; the wait in MESH_PAUSE_NS slices, ending at the signal or at the buffer's end as its status tells
   it (Completed, or Error, which the platform's watchdog gives a buffer that overruns: nccl-mesh-metal.m metal_ended),
   and for a group (`group`: comm kept from its destroy meanwhile by `runs`) at a revoke or a holding's end too, and
   past its deadline, MESH_PLATFORM_NS from its start, the program (design/eager-execution.md assumption A4); then the lock taken again, and a later program that
   touched comm meanwhile awaited too, under the same bound.  Under group_lock */
static int gpu_wait(struct ncclComm *comm, int group) {
  int ended = GPU_ENDED;
  struct mesh_deadline deadline = mesh_deadline("libnccl-mesh: a communicator's GPU programs' end (gpu_wait)", 0);
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
      const uint64_t left = deadline_left(&deadline);
      metal_wait(finished, last, left < MESH_PAUSE_NS ? left : MESH_PAUSE_NS);
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

/* -- the Metal path ------------------------------------------------------------------------------------ */

/* a step's result (nccl-mesh-metal.h struct metal_result): its errno a failure, named by `what` and the step's own message */
static ncclResult_t gpu(struct metal_result step, const char *what) {
  return step.status ? fail(ncclSystemError, "%s on the GPU: %s", what, step.message) : ncclSuccess;
}

/* a group issued and not complete: its communicator, its calls, its schedule, its entries [start, end) of each list on
   each of its communicator's channels ([h][0] the send list's, [h][1] the landing list's), the one channel they lie on
   (`lone`: lone_of, read where the group is planned; comm->count where it has none, METAL_NONE where they lie on two
   or more), whether it is a fused crossing, and
   its repeats; the groups in flight are in issue order, a communicator's holding consecutive entries on every list */
struct inflight {
  uint64_t ticket;
  struct ncclComm *comm;
  struct call *list;
  size_t n;
  struct schedule *plans;
  uint64_t (*start)[2], (*end)[2];
  uint32_t lone, repeats;
  int fused;
  struct inflight *next_group;
  struct issue *issue; /* the issue it is of (issue_bind), let go as it is done */
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

/* what an issued call owns: its write plan's reference and its steps copy's (its counts, matrix and views are its
   thread's: room_of) */
static void call_parts_free(struct call *c) {
  write_plan_drop(c->write_plan);
  steps_free(c->bound);
}
static void parts_free(struct call *list, size_t n) { for (size_t i = 0; i < n; i++) call_parts_free(list + i); }
/* a definition's calls (a declaration's, a WRITE plan's): what each owns besides (its counts and matrix, its
   declared masks and production), then the list */
static void calls_free(struct call *list, size_t n) {
  for (size_t i = 0; i < n; i++) {
    call_parts_free(list + i);
    free(list[i].segments); free(list[i].matrix); free(list[i].contributors); free(list[i].recipients); free(list[i].production);
    free(list[i].phase);
  }
  free(list);
}

/* an issue of definition x for `repeats` repeats (struct issue): each channel's lists' room, x's calls' steps copies and
   a record in flight; NULL where memory runs out */
static struct issue *issue_make(struct ncclComm *comm, const struct compiled *x, uint32_t repeats) {
  struct issue *u = calloc(1, sizeof *u);
  int made = u && (u->plans = calloc(comm->count + 1, sizeof *u->plans)) && (u->steps = calloc(x->n + 1, sizeof *u->steps)) &&
             (u->calls = calloc(x->n + 1, sizeof *u->calls)) && (u->bound = calloc(x->n + 1, sizeof *u->bound)) &&
             (u->group = calloc(1, sizeof *u->group)) &&
             (u->group->start = calloc(2 * (size_t)comm->count + 1, sizeof *u->group->start));
  if (u) { u->n = x->n; u->repeats = repeats; }
  for (uint32_t h = 0; made && h < comm->count; h++) {
    const struct schedule *p = x->plans + h;
    struct schedule *q = u->plans + h;
    made = (q->out = malloc((p->outs * repeats + 1) * sizeof *q->out)) && (q->in = malloc((p->ins * repeats + 1) * sizeof *q->in)) &&
           (!p->consume || (q->consume = malloc((p->ins * repeats + 1) * sizeof *q->consume)));
  }
  for (size_t i = 0; made && i < x->n; i++) made = !x->calls[i].bound || (u->steps[i] = steps_copy(x->calls[i].bound));
  if (!made) { issue_free(comm, u); u = NULL; }
  return u;
}
static void issue_free(struct ncclComm *comm, struct issue *u) {
  if (!u) return;
  schedules_free(comm, u->plans);
  for (size_t i = 0; u->steps && i < u->n; i++) steps_free(u->steps[i]);
  if (u->group) free(u->group->start);
  for (size_t i = 0; u->bound && i < u->n; i++) { free(u->bound[i].views[0]); free(u->bound[i].views[1]); free(u->bound[i].segments); }
  if (u->tables) ncclMemFree(u->tables);
  free(u->steps); free(u->calls); free(u->bound); free(u->group); free(u);
}
/* an issue's group done with: it is bound again by its definition's next group of its generation, or freed where it is
   no definition's (dropped).  Under group_lock */
static void issue_let(struct ncclComm *comm, struct issue *u) {
  if (!u) return;
  u->held = 0;
  if (u->dropped) issue_free(comm, u);
}

/* a group done with: what its calls own (the list is its issue's), and its issue let go */
static void inflight_free(struct inflight *g) {
  parts_free(g->list, g->n);
  issue_let(g->comm, g->issue);
}

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
    inflight_free(g);
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
static struct message *in_at(struct inflight *g, uint32_t h, uint64_t at) { return consumption_at(g->plans + h, at - g->start[h][1]); }
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
  return m && ((m->c->write_plan && m->c->write_plan->ready_value) || ready(m->c->bound, m->step));
}

/* the fused crossing of the group being encoded (nccl.h ncclMeshFused): its records' lists, the room the
   caller gave each (`room`: publications, landings), and the publications and landings written in place of their
   dispatches; under group_lock, for one launch_metal */
static struct { ncclMeshFused *out; uint32_t room[2], published, landed; } fusing;

/* Explicit producer words and relay landing words already name readiness.  The last entry of [first, end)
   on channel h that needs a publication (needs_publication), NULL none */
static const struct message *channel_publication(struct inflight *g, uint32_t h, uint64_t first, uint64_t end) {
  for (uint64_t at = end; at > first; at--) {
    const struct message *m = out_at(inflight_at(g->comm, h, 0, at - 1), h, at - 1);
    if (needs_publication(m)) return m;
  }
  return NULL;
}

/* channel h's send entries [published, upto) published, their steps' bytes written by the stream's work before:
   one release storing the holding's `ready` past the last (one system fence, nccl-mesh-kernels.h mesh_step_release),
   or, the fused crossing's, its record for its caller's kernel (metal_send_deferred), whose last threadgroup to write
   its operand stores it, its first repeat's alone (a later repeat's is the record's at its own j) */
static ncclResult_t publish(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint64_t upto, int *published) {
  struct channel *ch = comm->channels + h;
  struct inflight *g = inflight_at(comm, h, 0, upto - 1);
  if (!g) return outside_groups(comm, h, 0, upto - 1);
  const struct message *m = channel_publication(g, h, ch->published, upto);
  if (!m) { ch->published = upto; return ncclSuccess; }
  const struct metal_release release = {ch->buffer, ch->ready, m->end, g->plans[h].each[0]};
  ncclResult_t result = ncclSuccess;
  if (g->fused && (upto - 1 - g->start[h][0]) * g->repeats < g->end[h][0] - g->start[h][0]) {
    if (fusing.published >= fusing.room[0])
      return fail(ncclInvalidUsage, "a fused crossing's publications past the %u its ncclMeshFused has room for (ncclMeshPlan's entries)", fusing.room[0]);
    result = gpu(metal_send_deferred(program, m->data.buffer, m->data.offset, &release, fusing.out->publish[fusing.published]),
                 "a fused publication's record");
    fusing.published += !result;
  } else if (!g->fused) {
    void *operands[64];
    size_t n = 0;
    for (uint64_t at = ch->published; at < upto; at++) {
      if (n == sizeof operands / sizeof *operands) { metal_uses(program, operands, n, 0); n = 0; }
      operands[n++] = out_at(inflight_at(comm, h, 0, at), h, at)->data.buffer;
    }
    metal_uses(program, operands, n, 0);
    result = gpu(metal_publish(program, &release), "a publication");
  }
  if (!result) { ch->published = upto; *published = 1; }
  return result;
}

/* channel h's next AOT consumer entry (possibly a later physical landing), as its track's item (metal_record; rdma/NCCL.md "Per-channel progress") or,
   `base` METAL_NONE, as dispatches in the stream's order (complete_track): its first wait on the holding's `landed`
   past it; a combine's landed bytes combined
   into its buffer (its first one, a call reading its send buffer in place, after a wait: with op landed), a COPY's
   wait alone (it landed in place), the fused crossing's landing recorded for its caller's kernel
   (metal_land_deferred: its first repeat's alone, as publish); a receive depends on every earlier overlapping receive
   (metal_depend), while disjoint ranges on separate cables can proceed independently */
static ncclResult_t consume(struct metal_program *program, struct ncclComm *comm, uint32_t h, uint32_t base) {
  struct channel *ch = comm->channels + h;
  const uint64_t at = ch->consumed;
  struct inflight *g = inflight_at(comm, h, 1, at);
  if (!g) return outside_groups(comm, h, 1, at);
  const struct message *m = in_at(g, h, at);
  struct steps *st = m->c->bound;
  const struct mesh_step *step = st->steps + m->step;
  ncclResult_t result = ncclSuccess;
  for (uint32_t at = st->prior_first[m->step]; !result && base != METAL_NONE && at < st->prior_first[m->step + 1]; at++) {
    const uint32_t j = st->prior[at];
    result = gpu(metal_depend(program, base + st->channel[j], st->consumption[j] + 1), "a receive's dependency");
  }
  const struct metal_wait wait = {ch->buffer, comm->violation, m->data.buffer, ch->landed, m->data.offset, m->end, at, h, g->plans[h].each[1]};
  if (!result && step->op == MESH_STEP_REDUCE) {
    const struct where to = shifted(st->buffer[step->buffer], m->offset);
    if (st->fresh && st->fresh[m->step]) {
      const struct where with = shifted(st->read[m->step], m->offset);
      result = gpu(metal_spin(program, &wait), "a wait");
      if (!result)
        result = gpu(metal_combine3(program, m->c->type, m->c->combine, to.buffer, to.offset, m->data.buffer, m->data.offset, with.buffer, with.offset,
                                    m->bytes / SIZE[m->c->type]), "a first combine");
    } else result = gpu(metal_land(program, ktype(m->c), kop(m->c), to.buffer, to.offset, kcount(m->c, m->bytes), m->pages, comm->page, &wait), "a landing");
  } else if (!result && g->fused && (at - g->start[h][1]) * g->repeats < g->end[h][1] - g->start[h][1]) {
    if (fusing.landed >= fusing.room[1])
      return fail(ncclInvalidUsage, "a fused crossing's landings past the %u its ncclMeshFused has room for (ncclMeshPlan's entries)", fusing.room[1]);
    result = gpu(metal_land_deferred(program, m->data.buffer, m->data.offset, &wait, fusing.out->land[fusing.landed]), "a fused landing's record");
    fusing.landed += !result;
  } else if (!result && !g->fused) result = gpu(metal_spin(program, &wait), "a wait");
  if (result) return result;
  st->done[m->step] = 1;
  ch->consumed = at + 1;
  if (ch->waited < at + 1) ch->waited = at + 1;
  return ncclSuccess;
}

/* the channel every group in flight through `through` lies on alone, all of one communicator (each group's `lone`,
   read where it was planned: launch_metal), or METAL_NONE where they lie on two or more channels or are of two
   communicators; a group with no entry lies on none and joins any */
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
   (one_track): the GPU command stream's own order is the dependency (rdma/NCCL.md "One track").  Plain dispatches in
   one serial order every rank builds alike: the next send entries published as soon as their steps' receives before
   them are consumed (a run of them one release; the fused crossing's alone), else the next landing consumed (its
   step's receive before it consumed, and a combine into bytes an earlier SEND of its call reads only once that SEND is
   published); a program whose SEND waits on a receive that waits on that SEND
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
    while (upto < out && publishable(comm, h, upto)) {
      const int fused = inflight_at(comm, h, 0, upto)->fused;
      if (fused && upto > ch->published) break;
      upto++;
      if (fused) break;
    }
    if (upto > ch->published) { result = publish(program, comm, h, upto, published); continue; }
    struct inflight *x = ch->consumed < in ? inflight_at(comm, h, 1, ch->consumed) : NULL;
    const struct message *r = x ? in_at(x, h, ch->consumed) : NULL;
    if (r && ready(r->c->bound, r->step) && ch->published >= r->need) result = consume(program, comm, h, METAL_NONE);
    else
      result = fail(ncclInternalError, "channel %u's entries admit no order on one track (published %llu of %llu, consumed %llu of %llu)", h,
                    (unsigned long long)ch->published, (unsigned long long)out, (unsigned long long)ch->consumed, (unsigned long long)in);
  }
  return result;
}

/* every group in flight through `through` completed, on one track as plain dispatches (complete_track), else each
   channel progressing on its own (nccl-mesh-metal.h metal_record; rdma/NCCL.md "Per-channel progress"): channel h of
   the k-th communicator met is track base[k] + h, its publications to its communicator's last group through `through`
   each an item of its own (nothing credits it; a SEND's step after the consumption of the receive before it), then its
   consumptions (a combine into bytes an earlier SEND of its channel reads after that publication: its `need`); the
   tracks run as one (metal_tracks).  No rank orders anything another rank must share */
/* Only values addressed to this rank are inputs to its fused consumer. Relay
   arrivals have their own landed-word dependencies in the transport program;
   their immutable leases live through the enclosing data function. */
static uint64_t own_landing(const struct inflight *g, uint32_t h) {
  const struct schedule *p = g->plans + h;
  for (size_t k = p->ins / g->repeats; k; k--)
    if ((!p->in[k - 1].c->target || p->in[k - 1].c->target == g->comm->rank + 1) &&
        !(p->in[k - 1].c->write_plan && p->in[k - 1].c->write_plan->land_records)) return g->start[h][1] + k;
  return g->start[h][1];
}

/* group_lock's arrays, kept from call to call (plan_direct's ranges, complete()'s tracks and tiles): `bytes` of one,
   grown (its new bytes zeroed) where it has fewer, so none is allocated once each has its size; NULL where memory runs
   out */
enum { KEPT_SPANS, KEPT_COMMS, KEPT_BASES, KEPT_CONSUMED, KEPT_ENDS, KEPT_TILES, KEPTS };
static void *kept(int k, size_t bytes) {
  static struct { void *at; size_t room; } room[KEPTS];
  if (bytes > room[k].room) {
    unsigned char *grown = realloc(room[k].at, bytes);
    if (!grown) return NULL;
    memset(grown + room[k].room, 0, bytes - room[k].room);
    room[k].at = grown; room[k].room = bytes;
  }
  return room[k].at;
}

/* Configuration-only reduction to the last prefix for each consumer record
   on a cable. Several contributors can name the same consumer tile. */
struct tile_landing { uint64_t *record; size_t entry; };
static int tile_landing_order(const void *left,const void *right) {
  const struct tile_landing *a=left,*b=right;
  if(a->record!=b->record)return (uintptr_t)a->record<(uintptr_t)b->record?-1:1;
  return a->entry>b->entry?-1:a->entry<b->entry;
}
static ncclResult_t complete(struct metal_program *program, struct inflight *through, int *published) {
  if (through->fused) {
    /* A group of immutable one-way writes. Relay entries already name their
       incoming landed word; publishing a channel never releases relay bytes
       early. The producer releases each physical channel once. Its consumer
       needs only that channel's last landing addressed to this rank. A later
       relay must not delay local compute. All leases belong to the enclosing
       data function, including relay-only arrivals skipped by this consumer. */
    struct ncclComm *comm = through->comm;
    ncclResult_t result = ncclSuccess;
    touch(comm);
    metal_gate(program, comm->violation);
    for (uint32_t h = 0; h < comm->count && !result; h++) {
      struct channel *ch = comm->channels + h;
      const uint64_t first_end = through->start[h][0] +
                                 (through->end[h][0] - through->start[h][0]) / through->repeats;
      if (ch->published < first_end) result = publish(program, comm, h, first_end, published);
      if (!result) ch->published = through->end[h][0];
      const struct schedule *schedule = through->plans + h;
      struct tile_landing *tiles=kept(KEPT_TILES,(schedule->ins/through->repeats+1)*sizeof *tiles);
      if(!tiles)return fail(ncclSystemError,"allocating consumer prefix compilation indices");
      size_t count=0;
      for (size_t k = schedule->ins / through->repeats; !result && k; k--) {
        const struct message *m = schedule->in + k - 1;
        struct ncclMeshWritePlan *write = m->c->write_plan;
        if (!write || !write->land_records || write->to != comm->rank) continue;
        const size_t tile=write_tile(write,m->offset);
        uint64_t *record=(uint64_t *)((char *)write->land_records+tile_offset(write->land_layout,tile,m->c->invocation));
        tiles[count++]=(struct tile_landing){record,k-1};
      }
      qsort(tiles,count,sizeof *tiles,tile_landing_order);
      for(size_t t=0;!result&&t<count;t++) {
        if(t&&tiles[t].record==tiles[t-1].record)continue;
        const size_t k=tiles[t].entry;
        const struct message *m=schedule->in+k;
        uint64_t *record=tiles[t].record;
        if (*record == m->c->write_plan->land_capacity) {
          result=fail(ncclInvalidUsage, "WRITE consumer bindings exceed their prepared physical-channel capacity");break;
        }
        const struct metal_wait wait = {ch->buffer, comm->violation, m->data.buffer, ch->landed, m->data.offset,
          m->end, through->start[h][1] + k, h, schedule->each[1]};
        ncclMeshBinding *bindings=(ncclMeshBinding *)(record+1);
        result = gpu(metal_land_deferred(program, m->data.buffer, m->data.offset, &wait,
          bindings[(*record)++]), "a WRITE consumer's landing record");
      }
      const uint64_t own = own_landing(through, h);
      if (!result && ch->consumed < own) {
        ch->consumed = own - 1;
        result = consume(program, comm, h, METAL_NONE);
      }
      if (!result) ch->consumed = through->end[h][1];
    }
    return result;
  }
  const uint32_t lone = one_track(through);
  if (lone != METAL_NONE) return complete_track(program, through, lone, published);
  size_t comms = 0;
  struct ncclComm **comm = NULL;
  uint32_t *base = NULL, tracks = 0;
  ncclResult_t result = ncclSuccess;
  for (struct inflight *g = inflight; g && !result; g = g == through ? NULL : g->next_group) {
    size_t k = 0;
    while (k < comms && comm[k] != g->comm) k++;
    if (k < comms) continue;
    if (!(comm = kept(KEPT_COMMS, (comms + 1) * sizeof *comm)) || !(base = kept(KEPT_BASES, (comms + 1) * sizeof *base))) {
      result = fail(ncclSystemError, "out of memory"); break;
    }
    comm[comms] = g->comm; base[comms++] = tracks; tracks += g->comm->count;
  }
  uint64_t *consumed = result ? NULL : kept(KEPT_CONSUMED, ((size_t)tracks + 1) * sizeof *consumed), (*end)[2] = NULL;
  if (!result && !consumed) result = fail(ncclSystemError, "out of memory");
  for (size_t k = 0; k < comms && !result; k++) {
    struct ncclComm *const c = comm[k];
    touch(c);
    metal_gate(program, c->violation);
    if (!(end = kept(KEPT_ENDS, ((size_t)c->count + 1) * sizeof *end))) { result = fail(ncclSystemError, "out of memory"); break; }
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
        if (!needs_publication(m)) { ch->published++; continue; }
        result = gpu(metal_record(program, track, METAL_PUBLISH, at, 0), "a publication's record");
        const struct steps *prior = m->c->bound;
        for (uint32_t d = prior->prior_first[m->step]; !result && d < prior->prior_first[m->step + 1]; d++) {
          const uint32_t j = prior->prior[d];
          result = gpu(metal_depend(program, base[k] + prior->channel[j], prior->consumption[j] + 1), "a SEND's dependency");
        }
        if (!result) result = publish(program, c, h, at + 1, published);
      }
      while (!result && ch->consumed < end[h][1]) {
        const uint64_t at = ch->consumed;
        struct inflight *g = inflight_at(c, h, 1, at);
        if (!g || g->fused) { result = g ? fail(ncclInternalError, "a fused crossing's landing on two tracks") : outside_groups(c, h, 1, at); break; }
        result = gpu(metal_record(program, track, METAL_CONSUME, at, in_at(g, h, at)->need), "a consumption's record");
        if (!result) result = consume(program, c, h, base[k]);
      }
    }
  }
  if (!result) result = gpu(metal_record(program, METAL_NONE, 0, 0, 0), "the tracks' end");
  if (!result) result = gpu(metal_tracks(program, tracks, consumed), "the tracks");
  else metal_record(program, METAL_NONE, 0, 0, 0);
  return result;
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
   steps read, fresh) and its complement: a collective that crosses a link, not fused, premultiplied or merging
   rows, whose send buffer lies in this node's region apart from its result `own` (both paths: a where's address), every
   step's range in its buffer wholly written or wholly unwritten by the rank's steps before it (by round, then order:
   the local range dependency order). Then a SEND of a range no
   earlier step wrote reads the send buffer (its entry names those pages), a receive's first combine of such a range
   writes buffer = send op piece, no scratch buffer is copied into, and the ranges of
   buffer 0 no step writes are copied from the send buffer after the publications (`complement`); else (0) the operand
   is made in place as ever (a premultiply, or a copy within the region) */
static int plan_direct(struct call *c, struct where own) {
  struct steps *st = c->bound;
  const size_t z = SIZE[c->type];
  uint64_t lo = 0, hi = elements(c);
  if (c->receive_views || !st || !call_contributes(c) || counted(c) || c->fused || c->premultiplied || unit_of(c) > 1 || !st->count) return 0;
  if (c->what == MESH_BROADCAST && c->root != c->comm->rank) return 0;
  if (c->what == MESH_ALLGATHER) { size_t length; lo = segment_at(c, c->comm->rank, &length); hi = lo + length; }
  const struct where send = c->send_at;
  size_t length, made = elements(c) * z;
  const size_t sent = (size_t)(hi - lo) * z, before = segment_at(c, c->comm->rank, &length) * z;
  const unsigned char *a = address_of(send), *b = address_of(own);
  if (!b) return 0;
  /* a reduce-scatter's result is its own segment alone, which in place (NCCL's) is its own contribution's range */
  if (st->partial) { b += before; made = length * z; }
  const int aliased = st->partial && b == a + before;
  if (c->send_views) {
    for (int rank=0;rank<c->comm->nranks;rank++) {
      const size_t bytes=(c->segments?c->segments[rank]:c->count)*z;
      const unsigned char *source=address_of(c->send_views[rank]);
      if (bytes&&(!in_region(c->comm,c->send_views[rank],bytes)||(source<b+made&&b<source+bytes)))return 0;
    }
  } else if (!in_region(c->comm, send, sent) || (!aliased && a < b + made && b < a + sent)) return 0;
  const uint32_t B = st->buffers;
  struct spans *w = kept(KEPT_SPANS, B * sizeof *w);
  for (uint32_t b = 0; w && b < B; b++) w[b].count = 0;
  if (!st->read || !st->fresh) { free(st->read); free(st->fresh); st->read = calloc(st->count, sizeof *st->read); st->fresh = calloc(st->count, 1); }
  int direct = w && st->read && st->fresh;
  for (uint32_t k = 0; direct && k < st->count; k++) {
    const struct mesh_step *step = st->steps + k;
    const uint64_t from = step->first, to = step->first + step->piece.elements;
    const struct where source=view_at(c,1,(size_t)(from-lo)*z);
    const struct where map={source.buffer,source.offset-(size_t)from*z};
    const int meet = spans_meet(w + step->buffer, from, to), inside = from >= lo && to <= hi;
    if (meet < 0 || (!meet && step->op != MESH_STEP_COPY && !inside)) direct = 0;
    else if (step->op == MESH_STEP_SEND) { if (!meet) st->read[k] = map; }
    else {
      if (step->op == MESH_STEP_REDUCE && !meet) { st->fresh[k] = 1; st->read[k] = map; }
      if (spans_add(w + step->buffer, from, to)) direct = 0;
    }
  }
  /* the result's own ranges: a reduce-scatter's segment, nothing of a reduce off its root, else the whole */
  uint64_t first = lo, last = hi;
  if (c->what == MESH_REDUCE_SCATTER) { size_t length; first = segment_at(c, c->comm->rank, &length); last = first + length; }
  if (!call_receives(c)) last = first;
  if (direct) {
    /* at most one range a buffer-0 step leaves before it, and the last: the steps copy's room (struct steps) */
    if (!st->complement) st->complement = malloc(3 * ((size_t)st->count + 1) * sizeof *st->complement);
    c->complement = st->complement;
    direct = c->complement != NULL;
    for (size_t k = 0, at = first; direct && at < last; k++) {
      const uint64_t end = k < w[0].count ? (w[0].span[k][0] < last ? w[0].span[k][0] : last) : last;
      if (end > at) {
        uint64_t *x = c->complement + 3 * c->complements++;
        x[0] = own.offset + at * z; x[1] = (at - lo) * z; x[2] = (end - at) * z;
      }
      if (k >= w[0].count) break;
      if (w[0].span[k][1] > at) at = w[0].span[k][1];
    }
  }
  if (!direct && st->read && st->fresh) { memset(st->read, 0, st->count * sizeof *st->read); memset(st->fresh, 0, st->count); }
  if (!direct) { c->complement = NULL; c->complements = 0; }
  return direct;
}

/* a call reading its send buffer in place: its result's ranges no step wrote, copied from the send buffer */
static ncclResult_t complement_copy(struct metal_program *program, const struct call *c) {
  if (!call_receives(c)) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  for (uint32_t k = 0; k < c->complements && !result; k++) {
    const uint64_t *x = c->complement + 3 * k;
    const struct where source=view_at(c,1,x[1]);
    result = gpu(metal_copy(program, METAL_PLAIN, c->sum.buffer, x[0], source.buffer, source.offset, x[2]), "the operand's own ranges");
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

/* a group's results, once its entries are consumed: each already in recvbuff (result_of), an integer average's
   divided there */
static ncclResult_t epilogue(struct metal_program *program, const struct call *list, size_t n) {
  ncclResult_t result = ncclSuccess;
  for (size_t i = 0; i < n && !result; i++) {
    const struct call *c = list + i;
    if (c->target || c->receive_views || !call_receives(c) || counted(c) || !c->post) continue;
    size_t length = c->count;
    const size_t before = c->what == MESH_REDUCE_SCATTER ? segment_at(c, c->comm->rank, &length) * SIZE[c->type] : 0;
    result = gpu(metal_truncdiv(program, c->type, c->recv_at.buffer, c->recv_at.offset, c->sum.buffer, c->sum.offset + before, length,
                                call_participants(c)), "the result");
  }
  return result;
}

/* the groups in flight whose entries are all published and consumed, in issue order: on each channel where the group
   published and its landings consumed; their results are encoded and the groups retired. This does not retire
   outgoing DMA: source and relay lease lifetimes are the data program's (design/pages-and-functions.md, Freeing 6-7) */
static ncclResult_t retire(struct metal_program *program) {
  ncclResult_t result = ncclSuccess;
  while (inflight && !result) {
    struct inflight *g = inflight;
    struct ncclComm *comm = g->comm;
    for (uint32_t h = 0; h < comm->count; h++)
      if (comm->channels[h].consumed < g->end[h][1] || comm->channels[h].published < g->end[h][0]) return ncclSuccess;
    metal_gate(program, comm->violation);
    if (!result) result = epilogue(program, g->list, g->n);
    inflight = g->next_group;
    if (!inflight) inflight_last = NULL;
    atomic_store(&retired, g->ticket);
    inflight_free(g);
  }
  return result;
}

/* a program's end: where the library made its encoder and the group succeeded, its end signals `finished` at its count
   (`issued`, which a host-path group and a destroy await: a program the caller was told to commit), each communicator it
   touched keeping that count as its `last` and the command buffer as its `command` (gpu_finished); a borrowed
   encoder's, or a failed group's, signals nothing, so no wait of the
   library's is on a command buffer the caller may never commit.  Under group_lock */
static void program_end(struct metal_program *program, int borrowed, ncclResult_t result) {
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

/* The one place a reuse of storage is ordered: before a group writes a slot, the slot's last group's landings have
   landed, so its SENDs from it were read (witnessed: mesh-routed.c) and its consumers ran earlier in the
   stream; that group completed first where it is in flight, then `landed` awaited past the landings no work in the
   stream awaited (a fused crossing's relay landings).  With no slot (0), before a fused crossing's caller
   writes its operands, which any earlier group of the communicator may still be sending, whatever its definition (one
   since evicted, one sharing the source): every group in flight completed, then every landing presented on each
   channel.  No `sent`.  Under group_lock */
static ncclResult_t guard(struct metal_program *program, struct ncclComm *comm, uint32_t slot, int *published) {
  const struct slot *s = slot ? comm->slots + slot - 1 : NULL;
  struct inflight *g = s ? inflight : inflight_last;
  while (s && g && g->ticket != s->ticket) g = g->next_group;
  ncclResult_t result = g ? walk(program, NULL, g, published) : ncclSuccess;
  if (g && !result) result = retire(program);
  for (uint32_t h = 0; !result && h < comm->count; h++) {
    struct channel *ch = comm->channels + h;
    const uint64_t lands = s ? s->lands[h] : ch->lands;
    if (ch->waited >= lands) continue;
    const struct metal_wait wait = {ch->buffer, comm->violation, ch->buffer, ch->landed, ch->landed, s ? s->in[h] : ch->in, lands - 1, h, {0}};
    result = gpu(metal_spin(program, &wait), "the last landings of a reused generation");
    ch->waited = lands;
  }
  return result;
}

/* one group on the stream's command buffer (or, the Metal 4 form, its Metal 4 encoder: nccl-mesh-metal.h metal_begin4),
   taking its calls: each crossing call's send buffer a lease (operands_check), its result in recvbuff (result_of), its
   storage that is not the caller's its definition's next slot (bind_slot) once that slot's last group has landed
   (guard), the send buffer read in place where it may (plan_direct); then the group scheduled and its entries presented (schedule: a piece off a page or
   outside the region fails the call as a value, nothing presented), its operands made in place, its gate stored where
   its landings wait on the communicator's (gate_of), in flight after the others, issued (with issue; a fused one is
   completed at once, its steps the caller's kernels') or completed with every group before it, its waits bounded by
   its calls' greatest polls; its results into recvbuff as it completes; the program's end signals `finished` where the
   library made its encoder.  The Metal 4 form makes its channels' counts and its slot resident through the stream's
   residency set.  A fused group on the classic form takes the stream's residency set too (metal_begin),
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
    /* a host group's scalar in memory (launch_host: its call has no stream, so reduction takes it), read as it is encoded */
    if (!result && list[i].scalar_at) memcpy(list[i].pre, list[i].scalar_at, SIZE[list[i].type]);
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
  const uint32_t repeats = list[0].fused && list[0].fused->repeats ? list[0].fused->repeats : 1;
  const size_t stride = list[0].fused ? list[0].fused->stride : 0;
  pthread_mutex_lock(&group_lock);
  const uint64_t holder = result ? comm->groups : (comm->groups += repeats) - repeats + 1;
  if (!result && remote) result = comm_failure(comm);
  struct schedule *plans = NULL;
  struct compiled *x = NULL;
  struct issue *u = NULL;
  if (!result) result = definition_of(comm, list, n, holder, &x);
  if (!result) result = operands_check(list, n);
  if (result) { pthread_mutex_unlock(&group_lock); parts_free(list, n); return result; }
  /* The compiled declaration fixes publication and landing arithmetic. */
  const int fused_kind = list[0].fused && remote;
  ncclMeshFused *const fused = fused_kind ? list[0].fused : NULL;
  if (list[0].fused && !fused) list[0].fused->publishes = list[0].fused->lands = 0;
  for (size_t i = 0; i < n && !fused; i++) list[i].fused = NULL;
  void *const violation = remote ? comm->violation : NULL;
  const char *message = NULL;
  struct metal_program operands;
  struct metal_program *program = metal4 ? metal_begin4(&operands, list[0].encoder, list[0].table, list[0].residency, violation, polls, &message)
                                         : metal_begin(&operands, list[0].command, list[0].encoder, fused ? list[0].residency : NULL, violation,
                                                       polls, &message);
  if (!program) {
    pthread_mutex_unlock(&group_lock); parts_free(list, n);
    return fail(ncclSystemError, "%s", message);
  }
  /* its program kept as comm's last (program_end), so a host group, whose command buffer the library made and commits,
     returns once it ends whether or not it crosses (launch_host: gpu_finished) */
  if (remote || !list[0].stream) touch(comm);
  for (uint32_t h = 0; remote && h < comm->count; h++) metal_resident(program, comm->channels[h].buffer);
  if (remote && comm->gate) metal_resident(program, comm->gate);
  const size_t r = (size_t)comm->rank;
  int gate = 0, published = 0, scheduled = 0;
  struct where area = {NULL, 0};
  uint32_t slot = 0;
  /* a fused crossing's caller writes its operands after this group is encoded: every earlier group of its communicator
     first (guard); else the last group of the generation this one binds, so its issue is let go to bind again
     (issue_bind), and once it is bound the last user of the slot it took where that is another */
  const uint32_t last = remote ? x->slot[x->uses % GENERATIONS] : 0;
  if (remote && (fused || last)) result = guard(program, comm, fused ? 0 : last, &published);
  if (!result) result = issue_bind(comm, &list, n, x, repeats, stride, remote, &plans, &u, &area, &slot);
  if (!result && slot) metal_resident(program, u->base.buffer);
  if (!result && remote && !fused && slot != last) result = guard(program, comm, slot, &published);
  /* the slot this generation let go, too small (bind_slot): its last group ran in the stream before this point (guard,
     above), so the stream stores that group's ticket in the slot's word, and the slot is then returnable (slot_take) */
  if (!result && remote && !fused && last && slot != last && comm->slots[last - 1].ticket && 8 + last <= (size_t)getpagesize() / 8)
    result = gpu(metal_open_gate(program, comm->violation, (7 + (size_t)last) * 8, comm->slots[last - 1].ticket), "a slot let go");
  struct inflight *g = NULL, *before = NULL;
  uint64_t ticket = 0;
  if (!result && remote) {
    uint64_t (*const start)[2] = u->group->start;
    *(g = u->group) = (struct inflight){.comm = comm, .start = start, .end = start + comm->count, .repeats = repeats, .fused = fused_kind, .issue = u};
    for (uint32_t h = 0; h < comm->count; h++) { g->start[h][0] = comm->channels[h].sends; g->start[h][1] = comm->channels[h].lands; }
    /* every landing, a write's and a fused crossing's too, is gated on the stream's reaching the group: its storage is
       reused (guard), never a generation of its own; the issue let go with the group (inflight_free) */
    if (!(result = schedule(program, comm, plans, &gate))) { g->plans = plans; plans = NULL; u = NULL; }
    else g = NULL;
    if (!result) {
      for (uint32_t h = 0; h < comm->count; h++) { g->end[h][0] = comm->channels[h].sends; g->end[h][1] = comm->channels[h].lands; }
      g->lone = lone_of(g);
      g->list = list; g->n = n; g->ticket = ticket = ++tickets;
      slot_used(comm, slot, ticket);
      before = inflight_last;
      if (inflight_last) inflight_last->next_group = g; else inflight = g;
      inflight_last = g;
      scheduled = 1;
    }
  }
  /* the operands made in place, after the group's entries are presented and before its gate opens or anything is
     published: a premultiply, or a copy within the region, where a call reads no send buffer in place; a scratch's
     copy; an all-to-all's own segment */
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    struct steps *st = c->bound;
    const size_t z = SIZE[c->type], bytes = elements(c) * z, all = elements(c);
    const struct where send = c->send_at, own = c->sum;
    if (c->target) continue;
    if (counted(c)) {
      size_t sent, landed, part = own_segment(c, &sent, &landed);
      const struct where source=view_at(c,1,sent),destination=view_at(c,0,landed);
      if (part && !same(source,destination))
        result = gpu(metal_copy(program,METAL_PLAIN,destination.buffer,destination.offset,source.buffer,source.offset,part),
                     "an all-to-all's own segment");
      continue;
    }
    if (!call_contributes(c) || !bytes) continue;
    size_t length = c->count;
    const size_t before = c->what == MESH_ALLGATHER ? segment_at(c, (int)r, &length) : 0;
    const struct where segment = view_at(c, 0, before * z);
    if (!c->direct && reducing(c->what)) {
      const int each = c->send_views || (st && st->partial);
      size_t at = 0;
      for (int rank = 0; rank < (each ? comm->nranks : 1) && !result; rank++) {
        const size_t count = !each ? all : c->segments ? c->segments[rank] : c->count;
        const struct where source = c->send_views ? c->send_views[rank] : shifted(send, at), to = operand_at(c, at);
        if (count && c->premultiplied)
          result = gpu(metal_premultiply(program, c->type, to.buffer, to.offset, source.buffer, source.offset, count, scalar_bits(c)), "the operand");
        else if (count && !same(to, source))
          result = gpu(metal_copy(program, METAL_PLAIN, to.buffer, to.offset, source.buffer, source.offset, count * z), "the operand");
        at += count * z;
      }
    } else if (!c->direct && ((c->what == MESH_BROADCAST && c->root == comm->rank) || c->what == MESH_ALLGATHER) && !same(segment, send))
      result = gpu(metal_copy(program, METAL_PLAIN, segment.buffer, segment.offset, send.buffer, send.offset, length * z), "the operand");
    for (uint32_t b = 1; st && b < st->initialized && !result && !c->direct; b++) {
      const struct where from=c->receive_views?send:own;
      result = gpu(metal_copy(program, METAL_PLAIN, st->buffer[b].buffer, st->buffer[b].offset+(c->receive_views?before*z:0),
                             from.buffer, from.offset, c->receive_views?length*z:bytes), "a scratch copy");
    }
  }
  for (uint32_t h = 0; !result && gate && g && h < comm->count; h++)
    for (size_t k = 0; k < g->plans[h].ins; k++) metal_uses(program, &g->plans[h].in[k].data.buffer, 1, 1);
  if (!result && gate) result = gpu(metal_open_gate(program, comm->gate, 0, comm->gates), "the group's gate");
  if (!result && g) {
    uint32_t lands = 0, publishes = 0;
    for (uint32_t h = 0; h < comm->count; h++) {
      publishes += g->fused ? channel_publication(g, h, g->start[h][0], g->end[h][0]) != NULL : (uint32_t)((g->end[h][0] - g->start[h][0]) / repeats);
      lands += g->fused ? own_landing(g, h) > g->start[h][1] : (uint32_t)((g->end[h][1] - g->start[h][1]) / repeats);
    }
    /* a fused group is completed at once, issued or not: its steps are records its caller's kernels run, and the
       library encodes no wait of it; the groups in flight before it are completed first, so it completes alone on its
       one channel's track (complete_track: the caller's kernels keep that stream's order) */
    const int issuing = issue && !g->fused;
    if (g->fused && before) {
      result = complete(program, before, &published);
      if (!result) result = retire(program);
    }
    if (g->fused) {
      fusing.out = fused; fusing.room[0] = fusing.out->publishes; fusing.room[1] = fusing.out->lands;
      fusing.published = fusing.landed = 0;
    }
    if (!result) result = walk(program, issuing ? g : NULL, issuing ? NULL : g, &published);
    if (!result && g->fused && (fusing.published != publishes || fusing.landed != lands))
      result = fail(ncclInvalidUsage, "a fused crossing whose program is not this rank's publications and landings of its whole operands");
    if (g->fused) { fusing.out->publishes = fusing.published; fusing.out->lands = fusing.landed; }
    fusing.out = NULL;
    metal_gate(program, violation);
    for (size_t i = 0; i < n && !result; i++) result = complement_copy(program, list + i);
    if (!result) result = retire(program);
  } else if (!result) result = epilogue(program, list, n);
  program_end(program, list[0].encoder != NULL, result);
  if (result && scheduled) walk_failed(comm, issue && !fused ? 0 : ticket, result);
  if (issue && !result) { issue->ticket = ticket > atomic_load(&retired) ? ticket : 0; issue->published = published; }
  if (!scheduled) parts_free(list, n);
  issue_let(comm, u);
  pthread_mutex_unlock(&group_lock);
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
    struct metal_program operands;
    struct metal_program *program = stream->argumentTable
      ? metal_begin4(&operands, stream->commandEncoder, stream->argumentTable, stream->residencySet, inflight->comm->violation, stream->polls, &message)
      : metal_begin(&operands, stream->commandBuffer, stream->commandEncoder, NULL, inflight->comm->violation, stream->polls, &message);
    if (!program) result = fail(ncclSystemError, "%s", message);
    else {
      for (struct inflight *g = inflight; g; g = g == through ? NULL : g->next_group)
        for (uint32_t h = 0; h < g->comm->count; h++) metal_resident(program, g->comm->channels[h].buffer);
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

const char *ncclMeshMetalPrelude(void) { return metal_prelude(); }

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

/* a host binding as its lease's Metal buffer: 0, ncclInvalidUsage where it lies in no lease */
static ncclResult_t as_metal(struct where *w) {
  if (w->buffer || !w->offset) return ncclSuccess;
  void *buffer = NULL;
  size_t offset = 0;
  if (lease_buffer((const void *)w->offset, &buffer, &offset))
    return fail(ncclInvalidUsage, "a host buffer at %p is in no lease of this process (nccl.h, Memory)",
                (void *)w->offset);
  *w = (struct where){buffer, offset};
  return ncclSuccess;
}
static ncclResult_t launch_host(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  pthread_mutex_lock(&group_lock);
  ncclResult_t result = revoked(comm) ? revoked_fail(comm) : comm_failure(comm);
  if (!result && comm_issued(comm))
    result = fail(ncclInvalidUsage, "a host-path group while this communicator's GPU groups are issued and not complete (ncclMeshComplete)");
  if (!result && finished && comm->last) result = gpu_finished(comm);
  if (!result && !comm->queue && !(comm->queue = metal_queue())) result = fail(ncclSystemError, "a command queue for communicator %s", comm->name);
  void *command = result ? NULL : metal_queue_command(comm->queue);
  pthread_mutex_unlock(&group_lock);
  if (!result && !command) result = fail(ncclSystemError, "a command buffer for communicator %s", comm->name);
  for (size_t i = 0; i < n && !result; i++) {
    struct call *c = list + i;
    c->command = command;
    result = as_metal(&c->send_at);
    if (!result) result = as_metal(&c->recv_at);
    for (int j = 0; j < 2 && !result; j++)
      for (int r = 0; (j ? c->receive_views : c->send_views) && r < comm->nranks && !result; r++)
        result = as_metal((j ? c->receive_views : c->send_views) + r);
  }
  if (result) { parts_free(list, n); metal_release(command); return result; }
  result = launch_metal(list, n);
  if (!result) {
    metal_commit(command);
    pthread_mutex_lock(&group_lock);
    result = gpu_finished(comm);
    pthread_mutex_unlock(&group_lock);
  }
  metal_release(command);
  return result;
}

/* one communicator's calls of a group, launched as its own group on the Metal path: on their stream (launch_metal) or
   on its own queue (launch_host), each taking what the calls own (freed, or moved to the group's issue); the list stays
   its caller's */
static ncclResult_t launch_one(struct call *list, size_t n) {
  const ncclComm_t comm = list[0].comm;
  if (list[0].stream) return launch_metal(list, n);
  const ncclResult_t result = launch_host(list, n);
  if (result) comm->async = result;
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
/* a thread's parts of its last group on several communicators, and their calls, each part's a run of them: kept for its
   next such group (grow), so none is allocated once they have its size */
static _Thread_local struct part *parts_kept;
static _Thread_local struct call *sorted;
static _Thread_local size_t parts_room, sorted_room;
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
  /* the thread's list kept for its next group (its calls' own taken by their launch: launch_one) */
  struct call *list = calls;
  const size_t n = call_count;
  call_count = 0;
  if (!n) return ncclSuccess;
  for (size_t i = 0; i < n; i++) {
    ncclMeshFused *f = list[i].fused;
    if (!f) continue;
    if ((f->publishes && !f->publish) || (f->lands && !f->land)) {
      parts_free(list, n);
      return fail(ncclInvalidArgument, "an ncclMeshFused with room for records and no list to hold them");
    }
    if (f->publishes) memset(f->publish, 0, f->publishes * sizeof *f->publish);
    if (f->lands) memset(f->land, 0, f->lands * sizeof *f->land);
  }
  int several = 0;
  for (size_t i = 1; i < n; i++) several |= list[i].comm != list[0].comm;
  if (!several) return launch_one(list, n);
  if (grow(&parts_kept, &parts_room, n, sizeof *parts_kept) || grow(&sorted, &sorted_room, n, sizeof *sorted)) {
    parts_free(list, n);
    return fail(ncclSystemError, "out of memory");
  }
  /* each communicator's calls a run of `sorted`, in their order, the communicators in the order the group names them */
  struct part *part = parts_kept;
  size_t parts = 0, placed = 0;
  for (size_t i = 0; i < n; i++) {
    size_t k = 0;
    while (k < parts && part[k].list[0].comm != list[i].comm) k++;
    if (k < parts) continue;
    part[parts++] = (struct part){.list = sorted + placed, .metal = list[i].stream != NULL};
    for (size_t j = i; j < n; j++) if (list[j].comm == list[i].comm) part[k].list[part[k].n++] = list[j];
    placed += part[k].n;
  }
  for (size_t b = 1; b < parts; b++)
    for (size_t a = 0; a < b && part[b].list; a++)
      if (part[a].list && part[b].list[0].fused && part[a].list[0].fused == part[b].list[0].fused && part_crosses(part + a) && part_crosses(part + b)) {
        part[b].result = fail(ncclInvalidUsage, "a fused crossing on several communicators: one ncclMeshFused holds one crossing's records");
        part[b].said = last;
        parts_free(part[b].list, part[b].n);
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
    on = (ncclMeshStream){c->command, c->encoder, NULL, NULL, NULL, c->table, c->residency, polls, NULL};
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
      parts_free(p->list, p->n);
    }
  }
  ncclResult_t result = ncclSuccess;
  struct said first = {0, ""};
  for (size_t k = 0; k < parts; k++) {
    if (part[k].threaded) pthread_join(part[k].thread, NULL);
    if (part[k].result && !result) { result = part[k].result; first = part[k].said; }
  }
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
    c.command = stream->commandBuffer; c.encoder = stream->commandEncoder;
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
  if (c.target < 0 || c.target > comm->nranks || (c.target && c.target - 1 == c.root))
    return fail(ncclInvalidArgument, "one-way write's distinct endpoints must belong to the fabric");
  return ncclSuccess;
}

/* a call on the Metal path keeps its command buffer and its buffers' MTLBuffers and offsets, so neither the
   stream nor the ncclMeshBuffers need outlive the call (a lowering's are its own, gone before an enclosing
   group ends); what the call owns (an uneven call's counts) is freed with it, or here where it is refused */
static ncclResult_t enqueue(struct call c) {
  const ncclResult_t refused = admit(&c);
  if (refused) { call_parts_free(&c); return refused; }
  ncclGroupStart();
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) { depth--; call_parts_free(&c); return fail(ncclSystemError, "out of memory"); }
    calls = grown; call_capacity = capacity;
  }
  calls[call_count++] = c;
  return ncclGroupEnd();
}

/* Fixed-size configuration metadata, before any data program executes. Retained
   bootstrap sockets never carry acknowledgements or payload-dependent routing. */
/* The caller holds prepare across both the fixed report and any indexed-view
   report. Concurrent preparations cannot interleave those two exchanges. */
static int prepare_exchange_locked(ncclComm_t comm, const void *local, size_t bytes, void *reports) {
  memcpy((unsigned char *)reports + (size_t)comm->rank * bytes, local, bytes);
  return exchange(comm, reports, bytes, 0, 0);
}
static ncclResult_t prepare_exchange(struct ncclComm *comm, const void *local, size_t bytes, void *reports) {
  pthread_mutex_lock(&comm->prepare);
  atomic_fetch_add(&comm->runs, 1);
  comm->until = mesh_deadline("libnccl-mesh: fixed function configuration metadata", MESH_PREPARATION_NS);
  const int error = prepare_exchange_locked(comm, local, bytes, reports);
  atomic_fetch_sub(&comm->runs, 1);
  pthread_mutex_unlock(&comm->prepare);
  return error ? fail(ncclRemoteError, "communicator %s: its configuration exchange: %s", comm->name, strerror(error)) : ncclSuccess;
}

ncclResult_t ncclMeshCommConfigGather(ncclComm_t comm, const void *local, void *reports, size_t bytes) {
  if (!comm || (bytes && (!local || !reports)) || bytes > SIZE_MAX / (size_t)comm->nranks)
    return fail(ncclInvalidArgument, "configuration metadata requires equal, representable rank extents");
  return bytes ? prepare_exchange(comm, local, bytes, reports) : ncclSuccess;
}

ncclResult_t ncclMeshCommRealized(ncclComm_t comm) {
  if (!comm) return fail(ncclInvalidArgument, "realization requires a communicator");
  const size_t fields = 2 * (size_t)comm->fabric.cables;
  uint64_t *local = calloc(fields + 1, sizeof *local);
  uint64_t *reports = calloc(fields * (size_t)comm->nranks + 1, sizeof *reports);
  if (!local || !reports) { free(local); free(reports); return fail(ncclSystemError, "allocating compiled list ends"); }
  pthread_mutex_lock(&group_lock);
  for (uint32_t h = 0; h < comm->count; h++) {
    const struct channel *ch = comm->channels + h;
    local[2 * ch->cable] = ch->sends;
    local[2 * ch->cable + 1] = ch->lands;
  }
  pthread_mutex_unlock(&group_lock);
  ncclResult_t result = prepare_exchange(comm, local, fields * sizeof *local, reports);
  for (uint32_t e = 0; !result && e < comm->fabric.cables; e++) {
    const struct mesh_cable *c = comm->fabric.cable + e;
    const uint64_t *a = reports + fields * c->node[0] + 2 * e;
    const uint64_t *b = reports + fields * c->node[1] + 2 * e;
    if (a[0] != b[1] || b[0] != a[1])
      result = fail(ncclInvalidUsage, "compiled cable %u list ends differ: rank %u send/land %llu/%llu, rank %u send/land %llu/%llu",
                    e, c->node[0], (unsigned long long)a[0], (unsigned long long)a[1],
                    c->node[1], (unsigned long long)b[0], (unsigned long long)b[1]);
  }
  free(local); free(reports);
  return result;
}

static ncclResult_t prepare_counted(struct call *c, const size_t *given) {
  const size_t N = (size_t)c->comm->nranks;
  if (N > SIZE_MAX / N / (2 * sizeof(uint64_t))) return fail(ncclInvalidArgument, "count matrix extent overflow");
  c->matrix = malloc(N * N * sizeof *c->matrix);
  if (!c->matrix) return fail(ncclSystemError, "allocating count matrix");
  ncclResult_t result = ncclSuccess;
  if (given) for (size_t k = 0; k < N * N; k++) c->matrix[k] = given[k];
  else {
    uint64_t *reports = malloc(2 * N * N * sizeof *reports);
    if (!reports) return fail(ncclSystemError, "allocating count reports");
    result = prepare_exchange(c->comm, c->segments, 2 * N * sizeof *reports, reports);
    for (size_t s = 0; s < N && !result; s++) for (size_t d = 0; d < N && !result; d++) {
      const uint64_t sent = reports[2 * s * N + d], received = reports[2 * d * N + N + s];
      if (sent != received) result = fail(ncclInvalidUsage, "AlltoAllV declaration differs at source %zu destination %zu", s, d);
      else c->matrix[s * N + d] = sent;
    }
    free(reports);
  }
  for (size_t q = 0; q < N && !result; q++)
    if (c->segments[q] != c->matrix[(size_t)c->comm->rank * N + q] || c->segments[N + q] != c->matrix[q * N + c->comm->rank])
      result = fail(ncclInvalidUsage, "AlltoAllV local declaration differs from its complete count matrix");
  return result;
}

static int rank_member(const uint64_t *mask, uint32_t rank) {
  return !mask || ((mask[rank / 64] >> (rank % 64)) & 1);
}

static ncclResult_t declared_members(struct call *c, const ncclMeshCall *d) {
  if (!d->contributors && !d->recipients) return ncclSuccess;
  if (c->target)
    return fail(ncclInvalidArgument, "an explicit write names its endpoints, not collective membership masks");
  const uint32_t n = (uint32_t)c->comm->nranks, me = (uint32_t)c->comm->rank;
  const size_t bytes = ((size_t)n + 63) / 64 * sizeof(uint64_t);
  if (d->contributors) {
    c->contributors = malloc(bytes);
    if (!c->contributors) return fail(ncclSystemError, "out of memory");
    memcpy(c->contributors, d->contributors, bytes);
  }
  if (d->recipients) {
    c->recipients = malloc(bytes);
    if (!c->recipients) return fail(ncclSystemError, "out of memory");
    memcpy(c->recipients, d->recipients, bytes);
  }
  c->roles = 1 | (rank_member(c->contributors, me) ? 2 : 0) | (rank_member(c->recipients, me) ? 4 : 0);
  for (uint32_t r = 0; r < n; r++) c->participants += rank_member(c->contributors, r);
  if (reducing(c->what) && !c->participants)
    return fail(ncclInvalidArgument, "a reduction must declare at least one contributor");
  if (c->what == MESH_BROADCAST && !rank_member(c->contributors, (uint32_t)c->root))
    return fail(ncclInvalidArgument, "a broadcast's root must contribute its input");
  if (c->what == MESH_ALLGATHER || c->what == MESH_REDUCE_SCATTER) {
    const uint64_t *owners = c->what == MESH_ALLGATHER ? c->contributors : c->recipients;
    if (!c->segments) {
      c->segments = calloc(n, sizeof *c->segments);
      if (!c->segments) return fail(ncclSystemError, "out of memory");
      c->total = 0;
      for (uint32_t r = 0; r < n; r++) c->total += (c->segments[r] = rank_member(owners, r) ? c->count : 0);
      c->count = c->segments[me];
    }
    for (uint32_t r = 0; r < n; r++)
      if (!rank_member(owners, r) && c->segments[r])
        return fail(ncclInvalidArgument, "rank %u has a nonzero segment but is not a declared owner", r);
  }
  if (counted(c)) {
    if ((!call_contributes(c) && mesh_segment_first(c->segments, (uint32_t)n)) ||
        (!call_receives(c) && mesh_segment_first(c->segments + n, (uint32_t)n)))
      return fail(ncclInvalidArgument, "a counted transaction's local extents contradict its declared operand roles");
  }
  return ncclSuccess;
}

/* a declared call (nccl.h ncclMeshCall) as the call its function makes (collective, enqueue_counts, routed: a V form's
   counts its own copy), admitted and its operator taken (admit, reduction), with no buffers; one declared `fused` names
   `fused_declared` as its fused stream's records, so plan_calls compiles its steps as that stream's call's (no records
   are written to it: a group's are its issued stream's) */
static ncclResult_t declared(ncclComm_t comm, const ncclMeshCall *d, struct call *c) {
  const int n = comm->nranks, me = comm->rank,
            v = !d->counts ? 0 : d->what == WHAT_COUNTED ? 2 * n : d->what == MESH_ALLGATHER || d->what == MESH_REDUCE_SCATTER ? n : 0;
  *c = (struct call){.what = d->what, .root = d->peer, .count = d->count, .type = d->type, .op = d->op, .comm = comm,
                     .fused = d->fused ? &fused_declared : NULL, .tile_bytes = d->tile_bytes};
  if (d->what == ncclMeshCallWrite && (d->destination < 0 || d->destination >= n))
    return fail(ncclInvalidArgument, "a declared write's destination must belong to the fabric");
  if (d->what == ncclMeshCallWrite) { c->what = MESH_BROADCAST; c->target = d->destination + 1; }
  if ((d->what < MESH_ALLREDUCE || d->what > MESH_ALLGATHER) && (d->what != WHAT_COUNTED && d->what != ncclMeshCallWrite))
    return fail(ncclInvalidArgument, "a declared call's kind %d (mesh-plan.h MESH_*, nccl.h ncclMeshCall*)", d->what);
  if (d->tile_bytes && (d->what != ncclMeshCallWrite || (unsigned)d->type >= ncclNumTypes ||
      d->tile_bytes % SIZE[d->type]))
    return fail(ncclInvalidArgument,"a declared WRITE tile must contain whole typed elements");
  if ((d->tile_bytes && !d->production) || (d->production && d->what != ncclMeshCallWrite))
    return fail(ncclInvalidArgument,"a tiled WRITE requires its producer records");
  if (d->production) {
    if ((unsigned)d->type >= ncclNumTypes || d->count > SIZE_MAX / SIZE[d->type])
      return fail(ncclInvalidArgument,"producer payload extent");
    const size_t bytes = d->count * SIZE[d->type];
    const size_t tiles = d->tile_bytes && bytes ? (bytes - 1) / d->tile_bytes + 1 : 1;
    if (tiles > SIZE_MAX / sizeof *c->production || !(c->production = malloc(tiles * sizeof *c->production)))
      return fail(ncclSystemError,"allocating producer records");
    for (size_t k = 0; k < tiles; k++) {
      if (!isfinite(d->production[k].seconds) || d->production[k].seconds < 0 || d->production[k].order == UINT64_MAX)
        return fail(ncclInvalidArgument,"producer predictions must be finite and nonnegative");
      c->production[k] = d->production[k];
    }
  }
  if (d->what == WHAT_COUNTED && !d->counts) return fail(ncclInvalidArgument, "a declared all-to-all without its counts");
  /* its bindings' offsets within their registration extents, this rank's now, the others' at ncclMeshCommPlan */
  if (d->bindings && d->what != ncclMeshCallWrite) {
    if (!(c->phase = malloc(2 * (size_t)n * sizeof *c->phase))) return fail(ncclSystemError, "out of memory");
    const struct hdr *m = comm->attach->context.M;
    const uintptr_t data = (uintptr_t)m + m->data_off, end = data + (uintptr_t)mesh_arena_pages(m) * m->pgsz;
    for (int k = 0; k < 2 * n; k++) c->phase[k] = UINT64_MAX;
    for (int j = 0; j < 2; j++) {
      const uintptr_t at = (uintptr_t)address_of((struct where){d->bindings[j].buffer, d->bindings[j].offset});
      if (at >= data && at < end) c->phase[2 * me + j] = (uint64_t)(at - data) % MESH_DMA_EXTENT;
    }
  }
  if (v && !(c->segments = malloc((size_t)v * sizeof *c->segments))) return fail(ncclSystemError, "out of memory");
  for (int k = 0; k < v; k++) { c->segments[k] = d->counts[k]; c->total += d->counts[k]; }
  if (v && d->what != WHAT_COUNTED) c->count = d->counts[me];
  if (d->what == WHAT_COUNTED) c->count = c->total - c->segments[me] - c->segments[n + me];
  const ncclResult_t refused = admit(c);
  if (refused) return refused;
  /* the bindings as the issued call's buffers, set after admit, which gives a call with no stream its own (none):
     its group's generations laid and placed for them (prepared).  design/eager-execution.md#what-follows */
  if (d->bindings && d->what != ncclMeshCallWrite) {
    c->send_at = (struct where){d->bindings[0].buffer, d->bindings[0].offset};
    c->recv_at = (struct where){d->bindings[1].buffer, d->bindings[1].offset};
  }
  const ncclResult_t members = declared_members(c, d);
  if (members) return members;
  if (counted(c)) return prepare_counted(c, d->matrix);
  return reducing(c->what) ? reduction(c) : ncclSuccess;
}

/* a plan's own: its groups' calls and entries, their storage back to the communicator's (compiled_drop) */
static void plan_free(struct ncclComm *comm, struct plan *plan) {
  for (uint32_t g = 0; plan && plan->group && g < plan->view.groups; g++) compiled_drop(comm, plan->group + g);
  if (plan) { free(plan->group); free(plan->order); free(plan->ends); free(plan->entries); }
  free(plan);
}

/* a declared group whose every call declares its bindings (nccl.h ncclMeshCall bindings; none a WRITE's, whose plan
   comes at its issue) laid and placed for them in each of its generations (issue_bind on copies of its calls, as their
   issues will be: the Metal path's, a host binding as its lease's Metal buffer: launch_host; a float64 reduction is
   refused there, launch_metal), so an issue with those buffers binds only its entries.  Under group_lock */
static ncclResult_t prepared(struct ncclComm *comm, struct compiled *x, int remote) {
  for (size_t i = 0; i < x->n; i++)
    if (!x->calls[i].phase || x->calls[i].target || (reducing(x->calls[i].what) && !on_gpu(x->calls[i].type))) return ncclSuccess;
  struct call *list = malloc((x->n + 1) * sizeof *list);
  ncclResult_t result = list ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (uint32_t k = 0; !result && k < GENERATIONS; k++) {
    memcpy(list, x->calls, x->n * sizeof *list);
    for (size_t i = 0; i < x->n && !result; i++) {
      list[i].bound = NULL; list[i].fused = x->calls[i].fused;
      if (reducing(list[i].what)) result = reduction(list + i);
      if (!result) result = as_metal(&list[i].send_at);
      if (!result) result = as_metal(&list[i].recv_at);
    }
    struct call *issued = list;
    struct schedule *plans = NULL;
    struct issue *u = NULL;
    struct where area = {NULL, 0};
    uint32_t slot = 0;
    if (!result) result = issue_bind(comm, &issued, x->n, x, 1, 0, remote, &plans, &u, &area, &slot);
    for (size_t i = 0; u && i < x->n; i++) steps_free(issued[i].bound);
    issue_let(comm, u);
  }
  x->uses = 0;
  free(list);
  return result;
}

/* each declared call's bindings' extent offsets (struct call `phase`), every rank's, by one configuration exchange of
   those this rank declared, in the plan's order (the same calls declare bindings on every rank) */
static ncclResult_t phases_exchange(struct ncclComm *comm, struct plan *plan) {
  const size_t N = (size_t)comm->nranks;
  size_t k = 0;
  for (uint32_t g = 0; plan && g < plan->view.groups; g++)
    for (size_t i = 0; i < plan->group[g].n; i++) k += plan->group[g].calls[i].phase != NULL;
  if (!k || N == 1) return ncclSuccess;
  uint64_t *local = malloc(2 * k * sizeof *local), *reports = malloc(N * 2 * k * sizeof *reports);
  ncclResult_t result = local && reports ? ncclSuccess : fail(ncclSystemError, "out of memory");
  size_t at = 0;
  for (uint32_t g = 0; !result && g < plan->view.groups; g++)
    for (size_t i = 0; i < plan->group[g].n; i++)
      if (plan->group[g].calls[i].phase) memcpy(local + 2 * at++, plan->group[g].calls[i].phase + 2 * comm->rank, 2 * sizeof *local);
  if (!result) result = prepare_exchange(comm, local, 2 * k * sizeof *local, reports);
  at = 0;
  for (uint32_t g = 0; !result && g < plan->view.groups; g++)
    for (size_t i = 0; i < plan->group[g].n; i++) {
      uint64_t *phase = plan->group[g].calls[i].phase;
      if (!phase) continue;
      for (size_t r = 0; r < N; r++) memcpy(phase + 2 * r, reports + r * 2 * k + 2 * at, 2 * sizeof *reports);
      at++;
    }
  free(local); free(reports);
  return result;
}

/* Compile group definitions, then bind the function's repeated-block order.
   Replacing a declaration takes effect at the next group; failure keeps the
   prior declaration. This work precedes numerical submission. */
ncclResult_t ncclMeshCommPlan(ncclComm_t comm, const ncclMeshCall *calls, int count,
    const ncclMeshPlanRun *order, size_t runs, const ncclMeshPlan **plan) {
  if (plan) *plan = NULL;
  if (!comm || count < 0 || (count && !calls) || (runs && !order) || (!!count != !!runs))
    return fail(ncclInvalidArgument, "ncclMeshCommPlan's arguments");
  uint32_t groups = 0;
  for (int i = 0; i < count; i++) groups += !i || calls[i].group != calls[i - 1].group;
  struct plan *made = groups ? calloc(1, sizeof *made) : NULL;
  ncclResult_t result = !groups || (made && (made->group = calloc(groups, sizeof *made->group)) && (made->entries = calloc(groups, sizeof *made->entries)))
                        ? ncclSuccess : fail(ncclSystemError, "out of memory");
  if (made) made->view.groups = groups;
  for (int i = 0, g = -1; i < count && !result; i++) {
    if (!i || calls[i].group != calls[i - 1].group) {
      struct compiled *x = made->group + ++g;
      int n = 1;
      while (i + n < count && calls[i + n].group == calls[i].group) n++;
      if (!(x->calls = calloc((size_t)n, sizeof *x->calls))) { result = fail(ncclSystemError, "out of memory"); break; }
    }
    struct compiled *x = made->group + g;
    result = declared(comm, calls + i, x->calls + x->n++);
  }
  if (!result) result = phases_exchange(comm, made);
  if (made && !result) {
    if(runs>SIZE_MAX/sizeof *made->order || runs>SIZE_MAX/sizeof *made->ends ||
       !(made->order=malloc(runs*sizeof *made->order)) || !(made->ends=malloc(runs*sizeof *made->ends)))
      result=fail(ncclSystemError,"allocating compact compiled order");
    else {
      memcpy(made->order,order,runs*sizeof *order);made->runs=runs;
      for(size_t k=0;k<runs&&!result;k++) {
        const ncclMeshPlanRun run=order[k];
        if(!run.count || !run.repeat || run.first>=groups || run.count>groups-run.first ||
           run.repeat>(UINT64_MAX-made->view.issues)/run.count) {
          result=fail(ncclInvalidArgument,"compiled group block extent");break;
        }
        made->ends[k]=made->view.issues+=run.count*run.repeat;
        for(uint32_t g=run.first;g<run.first+run.count;g++)made->group[g].repeat+=run.repeat;
      }
    }
  }
  pthread_mutex_lock(&group_lock);
  for (uint32_t g = 0; g < groups && !result; g++) {
    struct compiled *x = made->group + g;
    result = compile_group(comm, x->calls, x->n, &x->plans);
    for (uint32_t h = 0; h < comm->count && !result; h++) { made->entries[g][0] += x->plans[h].outs; made->entries[g][1] += x->plans[h].ins; }
    /* its storage taken here, a slot for each generation (bind_slot finds them held): its issues take none */
    int remote = 0;
    for (size_t i = 0; !result && i < x->n; i++) remote |= crosses(x->calls + i);
    const size_t need = remote ? slot_laid(comm, x, x->calls, x->n, x->plans, NULL, NULL) : 0;
    for (uint32_t k = 0; remote && !result && k < GENERATIONS; k++)
      if (!(x->slot[k] = slot_take(comm, need))) result = fail(ncclSystemError, "a declared group's storage: %s", last.text);
    if (!result) result = prepared(comm, x, remote);
  }
  if (!result) {
    struct plan *was = comm->plan;
    if (made) { made->view.first = comm->groups + 1; made->view.entries = (const uint64_t (*)[2])made->entries; }
    comm->plan = made;
    made = was;
    if (plan) *plan = comm->plan ? &comm->plan->view : NULL;
  }
  /* the plan replaced (or the one not made): its definitions' slots back to comm's, under the groups' lock */
  plan_free(comm, made);
  pthread_mutex_unlock(&group_lock);
  return result;
}

static ncclResult_t plan_routes(ncclComm_t comm, uint64_t *demand, size_t entries, int resolution) {
  if (!comm || !demand || entries != (size_t)comm->nranks * comm->nranks)
    return fail(ncclInvalidArgument, "plan demands require one source-major entry per directed rank pair");
  pthread_mutex_lock(&group_lock);
  ncclResult_t result = comm->plan ? ncclSuccess : fail(ncclInvalidUsage, "declare the complete function before querying its demands");
  memset(demand, 0, entries * sizeof *demand);
  for (uint32_t g = 0; !result && g < comm->plan->view.groups; g++) {
    const struct compiled *group = comm->plan->group + g;
    if (!group->repeat) continue;
    for (size_t i = 0; !result && i < group->n; i++) {
      const struct call *c = group->calls + i;
      if (!crosses(c)) continue;
      for (size_t e = 0; !result && e < entries; e++) {
        uint64_t bytes = 0;
        if (c->target) {
          if (e == (size_t)c->root * comm->nranks + c->target - 1) {
            const uint64_t a = mesh_span_gcd(comm->page, SIZE[c->type]);
            bytes = resolution ? (c->count ? (c->count - 1) / (comm->page / a) + 1 : 0) : c->count * SIZE[c->type];
          }
        }
        else if (c->bound && c->bound->demand && (!resolution || c->bound->route_units))
          bytes = resolution ? c->bound->route_units[e] : c->bound->demand[e];
        else { result = fail(ncclInvalidUsage, "the declared function lacks the complete physical transaction graph"); break; }
        if (resolution) { if (bytes > demand[e]) demand[e] = bytes; }
        else if (bytes > (UINT64_MAX - demand[e]) / group->repeat)
          result = fail(ncclInvalidArgument, "the declared function's directed byte count overflows");
        else demand[e] += bytes * group->repeat;
      }
    }
  }
  pthread_mutex_unlock(&group_lock);
  return result;
}

ncclResult_t ncclMeshProductionTimes(ncclMeshProduction* records, size_t count, double seconds) {
  if (!isfinite(seconds) || seconds < 0 || (count && !records)) return ncclInvalidArgument;
  uint64_t extent = 0;
  for (size_t i = 0; i < count; i++) {
    if (records[i].order == UINT64_MAX) return ncclInvalidArgument;
    if (records[i].order + 1 > extent) extent = records[i].order + 1;
  }
  for (size_t i = 0; i < count; i++) records[i].seconds = seconds * ((double)(records[i].order + 1) / extent);
  return ncclSuccess;
}

ncclResult_t ncclMeshCommPlanDemands(ncclComm_t comm, uint64_t *demand, size_t entries) {
  return plan_routes(comm, demand, entries, 0);
}

ncclResult_t ncclMeshCommPlanRouteUnits(ncclComm_t comm, uint64_t *units, size_t entries) {
  return plan_routes(comm, units, entries, 1);
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

struct ncclMeshPageTable {
  ncclComm_t comm;
  struct where base;
  size_t pages, page_bytes;
  int metal;
};

/* A tensor adapter resolves its local address once. Everything below that
   boundary names this table and a page-local start, plus a compact selection.
   A row's page/mask at byte k is divmod(start + row*pitch + k, page_bytes);
   no array of those identical affine expressions needs to be materialized. */
struct write_page_index { size_t page, first; };

/* Fixed-size configuration reports. Addresses are local arena phases; no
   process pointer crosses a socket. Metadata on the retained bootstrap is
   serialized separately from GPU execution and remains revocable/deadlined. */
struct write_binding_report {
  uint64_t phase, stride, bytes, count, packet, row_bytes, rows, views;
  struct mesh_write_layout layout;
  uint32_t repeats, type, source, destination, status, indexed;
};

static int write_view_compare(const void *a, const void *b) {
  const ncclMeshSpan *x = a, *y = b;
  return (x->offset > y->offset) - (x->offset < y->offset);
}
/* Pure metadata validation. Source aliasing is repeated reading; ambiguous
   landing aliases would give concurrent logical writers the same bytes. */
static int write_view_validate(const ncclMeshSpan *view, size_t count, int landing,
                               uint64_t *bytes, uint64_t *extent) {
  if ((count && !view) || !bytes || !extent) return EINVAL;
  if (count > SIZE_MAX / sizeof *view || count == SIZE_MAX) return EOVERFLOW;
  uint64_t total = 0, last = 0;
  int ordered = 1;
  for (size_t k = 0; k < count; k++) {
    if (!view[k].bytes) return EINVAL;
    if (view[k].offset > SIZE_MAX || view[k].bytes > SIZE_MAX - view[k].offset ||
        view[k].bytes > SIZE_MAX - total) return EOVERFLOW;
    total += view[k].bytes;
    if (k && view[k].offset < view[k - 1].offset + view[k - 1].bytes) ordered = 0;
    const uint64_t end = view[k].offset + view[k].bytes;
    if (end > last) last = end;
  }
  if (landing && !ordered) {
    ncclMeshSpan *ordered = malloc(count * sizeof *ordered);
    if (!ordered) return ENOMEM;
    memcpy(ordered, view, count * sizeof *ordered);
    qsort(ordered, count, sizeof *ordered, write_view_compare);
    int error = 0;
    for (size_t k = 1; k < count; k++)
      if (ordered[k].offset < ordered[k - 1].offset + ordered[k - 1].bytes) { error = EINVAL; break; }
    free(ordered);
    if (error) return error;
  }
  *bytes = total; *extent = last;
  return 0;
}

/* Already-declared masks/spans can also describe affine rows. Recognize that
   algebra once; peers need only the fixed geometry report for such a view.
   Irregular, reversed and aliased sources retain their declared index list. */
static int write_view_rows(const ncclMeshSpan *view, size_t count, uint64_t bytes,
    size_t *row_bytes, size_t *rows, size_t *pitch) {
  if (!count) return 0;
  int contiguous = 1, affine = 1;
  const uint64_t width = view[0].bytes;
  const uint64_t stride = count > 1 && view[1].offset >= view[0].offset ? view[1].offset - view[0].offset : width;
  for (size_t k = 1; k < count; k++) {
    if (view[k].offset != view[k - 1].offset + view[k - 1].bytes) contiguous = 0;
    if (view[k].bytes != width || view[k].offset < view[k - 1].offset ||
        view[k].offset - view[k - 1].offset != stride) affine = 0;
  }
  if (contiguous) { *row_bytes = bytes; *rows = 1; *pitch = bytes; return 1; }
  if (affine && stride >= width) { *row_bytes = width; *rows = count; *pitch = stride; return 1; }
  return 0;
}

/* Common cuts for compact local selections. Each peer can use rows, indexed
   masks or a contiguous selection for the same logical bytes. Visit actual
   discontinuities and registration boundaries without expanding affine rows
   or pages. Contiguous selections across page/row/span edges stay contiguous.
   Publication tiles restart the packet grid. All operands were checked once
   before this occurrence's cuts are compiled. */
static int write_view_packets(const struct write_binding_report *reports, const ncclMeshSpan *views, size_t stride,
    const uint64_t *phases, const uint64_t *strides, uint64_t repeats, uint32_t participants, uint32_t packet, struct mesh_span_plan *out) {
  const uint64_t bytes = reports[0].bytes, extent = MESH_DMA_EXTENT, tile = reports[0].layout.tile_bytes;
  size_t *index = calloc(participants, sizeof *index);
  uint64_t *begin = calloc(participants, sizeof *begin), *flat = calloc(2 * (size_t)participants + 1, sizeof *flat);
  struct mesh_span_residues cuts = {0}, lattice = {0};
  uint32_t flats = 0;
  int error = index && begin && flat ? mesh_span_residue_add(&cuts, 0) : ENOMEM;
  /* an untiled payload's flat participants: every registration boundary any repeat meets (mesh_span_residues) */
  for (uint32_t r = 0; !error && !tile && r < participants; r++)
    if (!reports[r].indexed && !(reports[r].rows > 1 && reports[r].stride != reports[r].row_bytes)) {
      flat[flats] = phases[r]; flat[participants + flats++] = strides[r];
    }
  if (!error && bytes && flats) error = mesh_span_residues(flat, flat + participants, flats, bytes, repeats, packet, &lattice);
  for (uint64_t at = 0; !error && at < bytes;) {
    const uint64_t within = tile ? at % tile : at;
    uint64_t length = packet - within % packet;
    if (tile && length > tile - within) length = tile - within;
    if (length > bytes - at) length = bytes - at;
    for (uint32_t r = 0; r < participants; r++) {
      const struct write_binding_report *p = reports + r;
      const int strided = p->rows > 1 && p->stride != p->row_bytes;
      if (!tile && !p->indexed && !strided) continue; /* the lattice's */
      uint64_t offset = at, left = bytes - at;
      if (p->indexed) {
        const ncclMeshSpan *view = views + r * stride;
        while (at - begin[r] >= view[index[r]].bytes) begin[r] += view[index[r]++].bytes;
        const uint64_t into = at - begin[r];
        offset = view[index[r]].offset + into;
        left = view[index[r]].bytes - into;
        for (size_t k = index[r] + 1; length > left && k < p->views &&
            view[k - 1].offset + view[k - 1].bytes == view[k].offset; k++) left += view[k].bytes;
      } else if (p->rows && p->stride != p->row_bytes) {
        const uint64_t into = at % p->row_bytes;
        offset = (at / p->row_bytes) * p->stride + into;
        left = p->row_bytes - into;
      }
      if (length > left) length = left;
      const uint64_t phase = (phases[r] + offset % extent) % extent;
      if (length > extent - phase) length = extent - phase;
    }
    if (lattice.count) {
      const uint64_t base = at - at % extent, into = at % extent;
      size_t k = 0;
      while (k < lattice.count && lattice.at[k] <= into) k++;
      const uint64_t next = k < lattice.count ? base + lattice.at[k] : base + extent + lattice.at[0];
      if (length > next - at) length = next - at;
    }
    at += length;
    error = mesh_span_residue_add(&cuts, at);
  }
  free(index); free(begin); free(flat); free(lattice.at);
  if (error) { free(cuts.at); return error; }
  *out = (struct mesh_span_plan){cuts.at, cuts.count};
  return 0;
}

/* Physical bindings, paths and common cuts are all configuration operands.
   Materialize their local descriptor trains once, including every repetition.
   Execution indexes these immutable tables; the usual retirement ledger keeps
   them alive until the bridge has finished reading them after plan destruction. */
static ncclResult_t write_prepare_spans(struct ncclMeshWritePlan *plan) {
  const ncclComm_t comm = plan->comm;
  const struct mesh_paths *paths = comm->writes + plan->from * comm->nranks + plan->to;
  const struct mesh_operand operand = {(uint32_t)plan->type + 1, (uint32_t)SIZE[plan->type], plan->count};
  const int count = mesh_write_compile(&comm->fabric, (uint32_t)comm->rank, (uint32_t)plan->from,
    (uint32_t)plan->to, paths, operand, comm->page, plan->layout.tile_bytes / SIZE[plan->type], NULL, 0);
  if (count < 0) return fail(ncclInvalidUsage, "preparing physical write spans: %s", strerror(-count));
  if((size_t)count>(SIZE_MAX/sizeof(size_t)-1)/plan->occurrences)
    return fail(ncclInvalidUsage,"prepared write binding index extent");
  const size_t entries=(size_t)count*plan->occurrences;
  struct mesh_transfer *transfers = calloc((size_t)count + 1, sizeof *transfers);
  plan->span_first = calloc(entries + 1, sizeof *plan->span_first);
  plan->step_units = calloc(entries + 1, sizeof *plan->step_units);
  if (!transfers || !plan->span_first || !plan->step_units) { free(transfers); return fail(ncclSystemError, "allocating write span geometry"); }
  const int made = mesh_write_compile(&comm->fabric, (uint32_t)comm->rank, (uint32_t)plan->from,
    (uint32_t)plan->to, paths, operand, comm->page, plan->layout.tile_bytes / SIZE[plan->type], transfers, (uint32_t)count);
  ncclResult_t result = made == count ? ncclSuccess : fail(ncclInternalError, "preparing physical write fragments");
  for (int k = 0; !result && k < count; k++) for(size_t instance=0;!result&&instance<plan->occurrences;instance++) {
    const struct mesh_step *s = &transfers[k].step;
    const size_t index=write_span_index(plan,(size_t)k,instance);
    const uint64_t units = span_units(write_packets(plan,instance), s->first * s->piece.element_bytes, mesh_step_bytes(s));
    plan->step_units[index] = (mesh_pages){units};
    if (units > (SIZE_MAX / sizeof *plan->spans - plan->span_first[index]) / plan->repeats)
      result = fail(ncclInvalidUsage, "prepared write descriptors overflow address space");
    else plan->span_first[index + 1] = plan->span_first[index] + (size_t)units * plan->repeats;
  }
  if (!result && plan->span_first[entries])
    result = ncclMemAlloc((void **)&plan->spans, plan->span_first[entries] * sizeof *plan->spans);
  const struct hdr *m = comm->attach->context.M;
  const struct where binding = comm->rank == plan->from ? plan->source : plan->landing;
  const uint64_t origin = count ? (uint64_t)(address_of(binding) - (const unsigned char *)m) : 0;
  for (int k = 0; !result && k < count; k++) for(size_t instance=0;!result&&instance<plan->occurrences;instance++) {
    const struct mesh_step *s = &transfers[k].step;
    const uint64_t logical = s->first * s->piece.element_bytes, bytes = mesh_step_bytes(s);
    const size_t index=write_span_index(plan,(size_t)k,instance);
    const size_t units = plan->step_units[index].n;
    const struct mesh_span_plan *packets=write_packets(plan,instance);
    for (uint32_t repeat = 0; !result && repeat < plan->repeats; repeat++) {
      uint64_t at = 0;
      size_t cut = span_after(packets, logical);
      for (size_t unit = 0; unit < units; unit++) {
        uint64_t next = packets->cuts[cut++] - logical;
        if (next > bytes) next = bytes;
        struct mesh_dma_span span = {origin + (instance % plan->layout.instances) * plan->layout.instance_stride +
          write_displacement(plan, logical + at + repeat * plan->stride), next - at};
        if (!mesh_dma_span_valid(m, span)) {
          result = fail(ncclInvalidUsage, "prepared write span crosses a registration extent"); break;
        }
        plan->spans[plan->span_first[index] + repeat * units + unit] = span;
        at = next;
      }
      if (!result && at != bytes) result = fail(ncclInternalError, "prepared write spans do not cover their fragment");
    }
  }
  free(transfers);
  if (!result && count) {
    plan->uses = calloc(2 * comm->count, sizeof *plan->uses);
    if (!plan->uses) return fail(ncclSystemError, "allocating prepared channel ownership");
    plan->payload = address_of(binding);
    plan->channels = comm->count;
    pthread_mutex_lock(&attach_lock);
    struct lease *payload = lease_at(plan->payload, NULL), *metadata = lease_at(plan->spans, NULL);
    int error = !payload || !metadata ? EINVAL : mesh_lease_retire_begin(&payload->reads);
    if (!error) {
      error = mesh_lease_retire_begin(&metadata->reads);
      if (error) mesh_lease_retire_end(&payload->reads);
    }
    if (!error) {
      plan->retained = 1;
      for (uint32_t h = 0; h < comm->count; h++) {
        const struct channel *ch = comm->channels + h;
        const struct hold *hold = comm->attach->holds + comm->endpoint[ch->cable];
        for (unsigned d = 0; d < 2; d++) plan->uses[2 * h + d] = (struct mesh_lease_retire_dep){
          .stream = ch->stream, .wire = hold->wire, .holding = hold->generation, .direction = d};
      }
    }
    pthread_mutex_unlock(&attach_lock);
    if (error) result = fail(ncclInvalidUsage, "preparing write ownership: %s", strerror(error));
  }
  return result;
}

/* a WRITE executed alone in its group (and not on a fused stream) binds a definition its plan holds, compiled as the
   plan is prepared (compile_group on the call ncclMeshWriteExecute makes, one an occurrence of its packet cuts); its
   issues are laid and placed at its first executions, a generation each (issue_bind).  A tiled WRITE's comes from its
   declaration (ncclMeshCommPlan); one that cannot be compiled yet (no paths), or whose routes or map changed since,
   at its next execution (definition_of).  The definitions hold no reference to their plan: they are let go before
   it (ncclMeshWritePlanDestroy), or with routes or a map that changed or the communicator's end (cache_drop), which
   keeps the plans that hold some (`compiled_writes`).  0, or the compile's failure.  Under group_lock */
static ncclResult_t write_definitions(struct ncclMeshWritePlan *plan) {
  struct ncclComm *comm = plan->comm;
  if (plan->layout.tile_bytes || comm->nranks == 1) return ncclSuccess;
  struct compiled *made = calloc(plan->occurrences, sizeof *made);
  ncclResult_t result = made ? ncclSuccess : fail(ncclSystemError, "out of memory");
  for (size_t o = 0; !result && o < plan->occurrences; o++) {
    struct compiled *x = made + o;
    if (!(x->calls = calloc(1, sizeof *x->calls))) { result = fail(ncclSystemError, "out of memory"); break; }
    x->n = 1; x->repeat = 1;
    x->calls[0] = (struct call){.what = MESH_BROADCAST, .root = plan->from, .target = plan->to + 1, .count = plan->count, .type = plan->type,
                                .op = ncclSum, .comm = comm, .write_plan = plan, .invocation = o};
    result = compile_group(comm, x->calls, 1, &x->plans);
    x->calls[0].write_plan = NULL;
  }
  if (!result) { plan->compiled = made; plan->next_compiled = comm->compiled_writes; comm->compiled_writes = plan; return ncclSuccess; }
  for (size_t o = 0; made && o < plan->occurrences; o++) compiled_drop(comm, made + o);
  free(made);
  return result;
}
static void write_definitions_drop(struct ncclComm *comm, struct ncclMeshWritePlan *plan) {
  for (struct ncclMeshWritePlan **at = &comm->compiled_writes; *at; at = &(*at)->next_compiled)
    if (*at == plan) { *at = plan->next_compiled; break; }
  for (size_t o = 0; o < plan->occurrences; o++) compiled_drop(comm, plan->compiled + o);
  free(plan->compiled);
  plan->compiled = NULL;
}

static ncclResult_t write_prepare(ncclMeshPageTable_t table, struct write_page_index start, size_t count, ncclDataType_t datatype,
    int source, int destination, uint32_t repeats, size_t stride, size_t max_packet,
    size_t row_bytes, size_t rows, const ncclMeshSpan *view, size_t views, int indexed, struct mesh_write_layout layout, ncclMeshWritePlan_t *out) {
  if (!table || !out || *out || !table->comm || !table->comm->attach)
    return fail(ncclInvalidArgument, "write preparation requires a page table and an empty output handle");
  const ncclComm_t comm = table->comm;
  const int metal = table->metal;
  const size_t packet = max_packet ? max_packet : comm->page;
  struct write_binding_report local = {.stride = stride, .count = count, .packet = packet, .repeats = repeats,
    .type = (uint32_t)datatype, .source = (uint32_t)source, .destination = (uint32_t)destination,
    .row_bytes = row_bytes, .rows = rows, .views = views, .indexed = indexed != 0, .layout = layout};
  if ((unsigned)datatype >= ncclNumTypes || source < 0 || source >= comm->nranks || destination < 0 ||
      destination >= comm->nranks || source == destination || !repeats || !packet || packet > UINT32_MAX ||
      (metal != 0 && metal != 1) || !layout.instances || !layout.invocations) local.status = EINVAL;
  if (!local.status && count > SIZE_MAX / SIZE[datatype]) local.status = EOVERFLOW;
  if (!local.status) local.bytes = count * SIZE[datatype];
  uint64_t view_extent = 0, view_origin = 0;
  if (!local.status && indexed) {
    if (datatype != ncclUint8 || rows || repeats != 1 || stride) local.status = EINVAL;
    else local.status = write_view_validate(view, views, comm->rank != source, &local.bytes, &view_extent);
    local.count = count = local.bytes;
    if (!local.status && write_view_rows(view, views, local.bytes, &row_bytes, &rows, &stride)) {
      view_origin = view[0].offset;
      indexed = 0; views = 0;
      local.indexed = 0; local.views = 0;
      local.row_bytes = row_bytes; local.rows = rows; local.stride = stride;
    }
  }
  if (!local.status && (layout.tile_bytes % SIZE[datatype] ||
      (layout.tile_bytes && repeats != 1))) local.status = EINVAL;
  if (!local.status && stride && repeats - 1 > (SIZE_MAX - local.bytes) / stride) local.status = EOVERFLOW;
  if (!local.status && rows && (!row_bytes || repeats != 1 || stride < row_bytes ||
      rows > SIZE_MAX / row_bytes || local.bytes != rows * row_bytes ||
      (stride && rows - 1 > (SIZE_MAX - row_bytes) / stride))) local.status = EOVERFLOW;
  const size_t table_bytes = table->pages * table->page_bytes;
  size_t origin = 0;
  if (start.page > table->pages || start.first >= table->page_bytes ||
      (start.page == table->pages && start.first)) local.status = EINVAL;
  else origin = start.page * table->page_bytes + start.first;
  if (view_origin > table_bytes - origin) local.status = EINVAL;
  else origin += view_origin;
  const struct where binding = shifted(table->base, origin);
  const unsigned char *address = address_of(binding);
  if (!local.status && local.bytes) {
    size_t extent = indexed ? (size_t)view_extent : rows ? row_bytes + (rows - 1) * stride : local.bytes + (repeats - 1) * stride;
    if (layout.instance_stride && layout.instances - 1 > (SIZE_MAX - extent) / layout.instance_stride) local.status = EOVERFLOW;
    else extent += (layout.instances - 1) * layout.instance_stride;
    if (extent > table_bytes - origin) local.status = EINVAL;
    pthread_mutex_lock(&attach_lock);
    const struct lease *lease = !local.status && in_region(comm, binding, extent) ? lease_at(address, NULL) : NULL;
    if (!lease || lease->returned || extent > lease->bytes - (size_t)(address - (const unsigned char *)lease->address))
      local.status = EINVAL;
    else local.phase = ((uint64_t)(address - (unsigned char *)comm->attach->context.M) - comm->attach->context.M->data_off) % MESH_DMA_EXTENT;
    pthread_mutex_unlock(&attach_lock);
  }
  struct write_binding_report *reports = calloc(comm->nranks, sizeof *reports);
  uint64_t *phases = calloc(comm->nranks, sizeof *phases), *strides = calloc(comm->nranks, sizeof *strides);
  struct ncclMeshWritePlan *plan = calloc(1, sizeof *plan);
  if (!reports || !phases || !strides || !plan) {
    free(reports); free(phases); free(strides); free(plan); return fail(ncclSystemError, "allocating write binding metadata");
  }
  pthread_mutex_lock(&comm->prepare);
  atomic_fetch_add(&comm->runs, 1);
  comm->until = mesh_deadline("libnccl-mesh: configuration-time write binding metadata", MESH_PREPARATION_NS);
  int error = prepare_exchange_locked(comm, &local, sizeof local, reports);
  for (int r = 0; r < comm->nranks && !error; r++) {
    const struct write_binding_report *p = reports + r;
    error = p->status;
    if (!error && (p->bytes != local.bytes || p->count != local.count || p->packet != local.packet ||
        p->repeats != local.repeats || p->type != local.type || p->source != local.source || p->destination != local.destination ||
        p->layout.tile_bytes != local.layout.tile_bytes || p->layout.invocations != local.layout.invocations)) error = EPROTO;
    strides[r] = p->stride;
  }
  size_t most = 0;
  ncclMeshSpan *all_views = NULL, *local_views = NULL;
  if (!error) {
    for (int r = 0; r < comm->nranks; r++) if (reports[r].views > most) most = (size_t)reports[r].views;
    if (most > SIZE_MAX / (size_t)comm->nranks / sizeof *all_views || views > (SIZE_MAX / sizeof *plan->view_first) - 1)
      error = EOVERFLOW;
    if (!error && most) {
      local_views = calloc(most, sizeof *local_views);
      all_views = calloc(most * (size_t)comm->nranks, sizeof *all_views);
      if (!local_views || !all_views) error = ENOMEM;
      else {
        if (views) memcpy(local_views, view, views * sizeof *view);
        error = prepare_exchange_locked(comm, local_views, most * sizeof *local_views, all_views);
      }
    }
  }
  atomic_fetch_sub(&comm->runs, 1);
  pthread_mutex_unlock(&comm->prepare);
  // Binding periods can differ. Keep only the invoked prefix when their LCM
  // exceeds the function extent; never union cuts from different occurrences.
  plan->occurrences=1;
  for(int r=0;!error&&r<comm->nranks;r++) {
    const size_t a=mesh_span_gcd(plan->occurrences,reports[r].layout.instances);
    const size_t factor=reports[r].layout.instances/a;
    plan->occurrences=factor>layout.invocations/plan->occurrences?layout.invocations:plan->occurrences*factor;
  }
  if(!error && (plan->occurrences>SIZE_MAX/sizeof *plan->packets ||
      !(plan->packets=calloc(plan->occurrences,sizeof *plan->packets))))error=ENOMEM;
  for(size_t occurrence=0;!error&&occurrence<plan->occurrences;occurrence++) {
    for(int r=0;r<comm->nranks;r++)
      phases[r]=(reports[r].phase+(uint64_t)((__uint128_t)(occurrence%reports[r].layout.instances)*
        reports[r].layout.instance_stride%MESH_DMA_EXTENT))%MESH_DMA_EXTENT;
    error=write_view_packets(reports,all_views,most,phases,strides,repeats,(uint32_t)comm->nranks,(uint32_t)packet,plan->packets+occurrence);
  }
  if(!error && indexed && views) {
    plan->view=malloc(views*sizeof *plan->view);
    plan->view_first=calloc(views+1,sizeof *plan->view_first);
    if(!plan->view||!plan->view_first)error=ENOMEM;
    else {
      memcpy(plan->view,view,views*sizeof *view);
      for(size_t k=0;k<views;k++)plan->view_first[k+1]=plan->view_first[k]+view[k].bytes;
      plan->views=views;plan->view_extent=(size_t)view_extent;
    }
  }
  free(reports); free(phases); free(strides); free(all_views); free(local_views);
  if (error) { write_packets_free(plan); free(plan->view); free(plan->view_first); free(plan);
    return fail(ncclInvalidUsage, "preparing exact-byte write bindings: %s", strerror(error)); }
  atomic_init(&plan->refs, 1);
  plan->layout = layout;
  plan->source = binding; plan->landing = binding; plan->comm = comm;
  plan->source_buffer = (ncclMeshBuffer){binding.buffer, binding.offset};
  plan->landing_buffer = plan->source_buffer;
  plan->count = count; plan->stride = rows ? 0 : stride; plan->bytes = local.bytes; plan->repeats = repeats;
  plan->row_bytes = row_bytes; plan->rows = rows; plan->pitch = stride;
  plan->type = datatype; plan->from = source; plan->to = destination; plan->metal = metal;
  const ncclResult_t prepared = write_prepare_spans(plan);
  if (prepared) { write_plan_drop(plan); return prepared; }
  pthread_mutex_lock(&group_lock);
  (void)write_definitions(plan); /* else compiled at its first execution (definition_of) */
  pthread_mutex_unlock(&group_lock);
  *out = plan;
  return ncclSuccess;
}
ncclResult_t ncclMeshPageTableCreate(const void *local, ncclComm_t comm, int metal, ncclMeshPageTable_t *out) {
  if (!local || !out || *out || (metal != 0 && metal != 1))
    return fail(ncclInvalidArgument, "page table needs local storage and an empty result");
  const struct where binding=metal ? where_of(local) : (struct where){NULL,(size_t)local};
  const void *address=address_of(binding);
  pthread_mutex_lock(&attach_lock);
  struct attach *owner=NULL;
  struct lease *lease=lease_at(address,&owner);
  struct ncclMeshPageTable *table=NULL;
  if (lease && !lease->returned && (!comm || owner==comm->attach)) {
    table=calloc(1,sizeof *table);
    if(table) {
      const size_t offset=(uintptr_t)address-(uintptr_t)lease->address;
      table->base=metal ? (struct where){binding.buffer,binding.offset-offset} : (struct where){NULL,(size_t)lease->address};
      table->comm=comm;table->metal=metal;
      table->page_bytes=owner->context.M->pgsz;
      table->pages=lease->bytes/table->page_bytes;
    }
  }
  pthread_mutex_unlock(&attach_lock);
  if(!table)return fail(ncclInvalidArgument,"page table has no live local region allocation");
  *out=table;return ncclSuccess;
}
ncclResult_t ncclMeshPageTableShape(ncclMeshPageTable_t table,size_t *pages,size_t *page_bytes) {
  if(!table||!pages||!page_bytes)return fail(ncclInvalidArgument,"page table shape operands");
  *pages=table->pages;*page_bytes=table->page_bytes;return ncclSuccess;
}
ncclResult_t ncclMeshPageTableBinding(ncclMeshPageTable_t table,ncclMeshBuffer *binding) {
  if(!table||!binding)return fail(ncclInvalidArgument,"page table binding operands");
  return ncclMeshMemBuffer(address_of(table->base),&binding->buffer,&binding->offset);
}
ncclResult_t ncclMeshPageTableDestroy(ncclMeshPageTable_t table) {free(table);return ncclSuccess;}

/* Masks are validated once, then adjacent selections are coalesced. Page
   identity imposes no extra runtime packet, dispatch, or boundary wait. */
static ncclResult_t page_spans(ncclMeshPageTable_t table,const ncclMeshPageMask *masks,size_t count,
                                ncclMeshSpan **out,size_t *used) {
  if(!table||(count&&!masks)||count>SIZE_MAX/sizeof **out)
    return fail(ncclInvalidArgument,"page mask operands");
  ncclMeshSpan *spans=count?malloc(count*sizeof *spans):NULL;
  if(count&&!spans)return fail(ncclSystemError,"page mask metadata allocation");
  size_t n=0;
  for(size_t i=0;i<count;i++) {
    const ncclMeshPageMask m=masks[i];
    if(m.page>=table->pages||m.first>table->page_bytes||m.count>table->page_bytes-m.first) {
      free(spans);return fail(ncclInvalidArgument,"a mask must stay within its one local page");
    }
    if(!m.count)continue;
    const uint64_t offset=m.page*table->page_bytes+m.first;
    if(n&&spans[n-1].offset+spans[n-1].bytes==offset)spans[n-1].bytes+=m.count;
    else spans[n++]=(ncclMeshSpan){offset,m.count};
  }
  *out=spans;*used=n;return ncclSuccess;
}

static ncclResult_t write_pages_prepare(ncclMeshPageTable_t table,const ncclMeshPageMask *masks,size_t count,
    int source,int destination,size_t max_packet,struct mesh_write_layout layout,ncclMeshWritePlan_t *out) {
  ncclMeshSpan *spans=NULL;size_t used=0;
  ncclResult_t result=page_spans(table,masks,count,&spans,&used);
  if(!result)result=write_prepare(table,(struct write_page_index){0},0,ncclUint8,source,destination,1,0,
                                 max_packet,0,0,spans,used,1,layout,out);
  free(spans);return result;
}

ncclResult_t ncclMeshWritePagesPrepare(ncclMeshPageTable_t table,const ncclMeshPageMask *masks,size_t count,
    int source,int destination,size_t max_packet,ncclMeshWriteLayout layout,ncclMeshWritePlan_t *out) {
  if(!table || layout.instance_pages>SIZE_MAX/table->page_bytes)
    return fail(ncclInvalidArgument,"repeated page table geometry");
  return write_pages_prepare(table,masks,count,source,destination,max_packet,
    (struct mesh_write_layout){layout.tile_bytes,layout.instances,layout.instance_pages*table->page_bytes,layout.invocations},out);
}

/* Resolve the local tensor binding to a page index. The common compiler
   bounds the entire selection and all its instances in that page table.
   Affine expressions stay affine until final DMA descriptors are emitted. */
static ncclResult_t tensor_pages(const void *local,ncclComm_t comm,int metal,
                                  ncclMeshPageTable_t *table,struct write_page_index *start) {
  ncclResult_t result=ncclMeshPageTableCreate(local,comm,metal,table);
  if(result)return result;
  const struct where binding=metal ? where_of(local) : (struct where){NULL,(size_t)local};
  const size_t origin=(uintptr_t)address_of(binding)-(uintptr_t)address_of((*table)->base);
  *start=(struct write_page_index){origin/(*table)->page_bytes,origin%(*table)->page_bytes};
  return ncclSuccess;
}

ncclResult_t mesh_compile_spans(const void *source_buffer,void *landing_buffer,const ncclMeshSpan *spans,size_t nspans,
    int source,int destination,ncclComm_t comm,int metal,size_t max_packet,struct mesh_write_layout layout,ncclMeshWritePlan_t *out) {
  if(!comm)return fail(ncclInvalidArgument,"tensor view requires a communicator");
  ncclMeshPageTable_t table=NULL;struct write_page_index start={0};
  ncclResult_t result=tensor_pages(comm->rank==source?source_buffer:landing_buffer,comm,metal,&table,&start);
  if(!result)result=write_prepare(table,start,0,ncclUint8,source,destination,1,0,max_packet,
                                 0,0,spans,nspans,1,layout,out);
  ncclMeshPageTableDestroy(table);return result;
}
/* mesh-write-compiler.h mesh_view_layout */
long mesh_view_layout(const uint64_t *shape, const int64_t *strides, size_t rank, uint64_t itemsize,
                      int64_t *shift, uint64_t *rows, ncclMeshSpan *spans, size_t room) {
  *shift = 0; rows[0] = rows[1] = rows[2] = 0;
  if (rank && (!shape || !strides)) return -1;
  int empty = !itemsize;
  for (size_t i = 0; i < rank; i++) empty |= !shape[i];
  if (empty) return 0;
  /* the trailing dense dimensions are one row of `width` bytes, the `outer` dimensions above it its rows */
  uint64_t width = itemsize;
  size_t outer = rank;
  while (outer && (shape[outer - 1] == 1 || strides[outer - 1] == (int64_t)width)) { outer--; width *= shape[outer]; }
  /* affine: the outer dimensions longer than one nested at whole multiples of the innermost one's pitch */
  int64_t pitch = (int64_t)width;
  for (size_t i = 0; i < outer; i++) if (shape[i] > 1) pitch = strides[i];
  int affine = pitch >= (int64_t)width;
  uint64_t count = 1;
  for (size_t i = outer; i-- > 0;)
    if (shape[i] > 1) { affine = affine && strides[i] == pitch * (int64_t)count; count *= shape[i]; }
  if (affine) { rows[0] = width; rows[1] = count; rows[2] = (uint64_t)pitch; return 0; }
  int64_t lowest = 0;
  for (size_t i = 0; i < rank; i++) if (strides[i] < 0) lowest += (int64_t)(shape[i] - 1) * strides[i];
  *shift = lowest;
  /* every row in logical order (an odometer over the outer dimensions), adjacent rows one span */
  uint64_t index[rank ? rank : 1];
  memset(index, 0, sizeof index);
  long n = 0;
  uint64_t end = 0;
  for (int more = 1; more;) {
    int64_t at = -lowest;
    for (size_t i = 0; i < outer; i++) at += (int64_t)index[i] * strides[i];
    if (n && end == (uint64_t)at) { if ((size_t)n <= room) spans[n - 1].bytes += width; }
    else { if ((size_t)n < room) spans[n] = (ncclMeshSpan){(uint64_t)at, width}; n++; }
    end = (uint64_t)at + width;
    more = 0;
    for (size_t i = outer; i-- > 0;) { if (++index[i] < shape[i]) { more = 1; break; } index[i] = 0; }
  }
  return n;
}
ncclResult_t mesh_compile_rows(const void *source_buffer,void *landing_buffer,size_t row_bytes,size_t rows,
    size_t pitch,int source,int destination,ncclComm_t comm,int metal,size_t max_packet,struct mesh_write_layout layout,ncclMeshWritePlan_t *out) {
  if(!comm||!rows||!row_bytes||rows>SIZE_MAX/row_bytes)
    return fail(ncclInvalidArgument,"invalid tensor row view");
  ncclMeshPageTable_t table=NULL;struct write_page_index start={0};
  ncclResult_t result=tensor_pages(comm->rank==source?source_buffer:landing_buffer,comm,metal,&table,&start);
  if(!result)result=write_prepare(table,start,rows*row_bytes,ncclUint8,source,destination,1,pitch,max_packet,
                                 row_bytes,rows,NULL,0,0,layout,out);
  ncclMeshPageTableDestroy(table);return result;
}
ncclResult_t mesh_compile_write(const void *source_buffer,void *landing_buffer,size_t count,ncclDataType_t datatype,
    int source,int destination,ncclComm_t comm,int metal,uint32_t repeats,size_t stride,size_t max_packet,ncclMeshWritePlan_t *out) {
  if(!comm)return fail(ncclInvalidArgument,"tensor write requires a communicator");
  ncclMeshPageTable_t table=NULL;struct write_page_index start={0};
  ncclResult_t result=tensor_pages(comm->rank==source?source_buffer:landing_buffer,comm,metal,&table,&start);
  if(!result)result=write_prepare(table,start,count,datatype,source,destination,repeats,stride,max_packet,
                                 0,0,NULL,0,0,(struct mesh_write_layout){.instances=1,.invocations=1},out);
  ncclMeshPageTableDestroy(table);return result;
}
ncclResult_t ncclMeshWriteSourceReady(ncclMeshWritePlan_t plan, const void *words, uint64_t value, ncclMeshTileLayout layout) {
  if (!plan || !value || value < (plan->layout.tile_bytes && plan->layout.tile_bytes < plan->bytes ? plan->layout.tile_bytes : plan->bytes) || plan->ready_value)
    return fail(ncclInvalidArgument, "source readiness needs a fresh plan and a value covering one logical tile");
  if (plan->comm->rank == plan->from && plan->retained) {
    size_t extent=0;
    if(tile_extent(layout,write_tiles(plan),sizeof(uint64_t),&extent))
      return fail(ncclInvalidArgument,"source readiness table geometry");
    if (!words) return fail(ncclInvalidArgument, "source readiness has no words");
    const struct where ready = plan->metal ? where_of(words) : (struct where){NULL, (size_t)words};
    void *address = address_of(ready);
    pthread_mutex_lock(&attach_lock);
    struct attach *attachment = NULL;
    struct lease *owner = lease_at(address, &attachment);
    const size_t offset = owner ? (uintptr_t)address - (uintptr_t)owner->address : 0;
    int error = !owner || attachment != plan->comm->attach || owner->returned || offset > owner->bytes || owner->bytes - offset < extent ||
      (uintptr_t)address % sizeof(uint64_t)
      ? EINVAL : mesh_lease_retire_begin(&owner->reads);
    pthread_mutex_unlock(&attach_lock);
    if (error) return fail(ncclInvalidArgument, "source readiness table is not an aligned live region operand");
    plan->source_ready = ready; plan->ready_owner = address;
  }
  plan->ready_value = value; plan->ready_layout = layout;
  return ncclSuccess;
}
ncclResult_t ncclMeshWriteExecute(ncclMeshWritePlan_t plan, size_t invocation, cudaStream_t stream) {
  atomic_fetch_add_explicit(&plan->refs, 1, memory_order_relaxed);
  return enqueue((struct call){.what = MESH_BROADCAST, .root = plan->from, .target = plan->to + 1,
    .send = plan->metal ? &plan->source_buffer : (const void *)plan->source.offset,
    .recv = plan->metal ? &plan->landing_buffer : (void *)plan->landing.offset,
    .count = plan->count, .type = plan->type, .op = ncclSum, .comm = plan->comm, .stream = stream,
    .write_plan = plan, .tile_bytes = plan->layout.tile_bytes, .invocation = invocation});
}
ncclResult_t ncclMeshWriteLandings(ncclMeshWritePlan_t plan, void *records, uint32_t capacity, ncclMeshTileLayout layout) {
  if (!plan || !plan->metal || plan->repeats != 1 || plan->comm->rank != plan->to || !records || !capacity || plan->land_records)
    return fail(ncclInvalidArgument, "WRITE landing output needs a fresh Metal plan and physical-channel storage");
  size_t extent=0,tiles=write_tiles(plan);
  if(tile_extent(layout,tiles,sizeof(uint64_t)+(size_t)capacity*sizeof(ncclMeshBinding),&extent))
    return fail(ncclInvalidArgument,"WRITE landing table geometry");
  for(size_t i=0;i<layout.instances;i++)for(size_t t=0;t<tiles;t++) {
    *(uint64_t *)((char *)records+tile_offset(layout,t,i))=0;
  }
  plan->land_records=records;plan->land_capacity=capacity;plan->land_layout=layout;
  return ncclSuccess;
}
ncclResult_t ncclMeshWritePlanDestroy(ncclMeshWritePlan_t plan) {
  if (plan) {
    pthread_mutex_lock(&group_lock);
    if (plan->compiled) write_definitions_drop(plan->comm, plan);
    pthread_mutex_unlock(&group_lock);
  }
  write_plan_drop(plan);
  return ncclSuccess;
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
  if (!(c.segments = room_of(ROOM_SEGMENTS, (size_t)n * sizeof *c.segments))) return fail(ncclSystemError, "out of memory");
  for (int r = 0; r < n; r++) { c.segments[r] = counts[r]; c.total += counts[r]; }
  return enqueue(c);
}
ncclResult_t ncclMeshAllGatherV(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream) {
  return enqueue_counts((struct call){.what = MESH_ALLGATHER, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm,
                                      .stream = stream}, counts);
}
/* the call's buffers, a rank's each, as its views (room k of its place: room_of) */
static struct where *bind_views(int k,void *const *buffers,ncclComm_t comm,cudaStream_t stream) {
  struct where *views=room_of(k,(size_t)comm->nranks*sizeof *views);
  for(int rank=0;views&&rank<comm->nranks;rank++)
    views[rank]=stream?where_of(buffers[rank]):(struct where){NULL,(size_t)buffers[rank]};
  return views;
}
ncclResult_t ncclMeshAllGatherBuffers(const void *sendbuff, void *const *recvbuffs, const size_t *counts,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  if (!comm || !recvbuffs || !counts) return fail(ncclInvalidArgument, "gather requires its configured final bindings");
  struct where *views=bind_views(ROOM_RECEIVE_VIEWS,recvbuffs,comm,stream);
  if (!views) return fail(ncclSystemError, "out of memory");
  return enqueue_counts((struct call){.what=MESH_ALLGATHER,.send=sendbuff,.recv=recvbuffs[0],.type=datatype,
    .op=ncclSum,.comm=comm,.stream=stream,.receive_views=views}, counts);
}
ncclResult_t ncclMeshReduceScatterV(const void *sendbuff, void *recvbuff, const size_t *counts, ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm, cudaStream_t stream) {
  return enqueue_counts((struct call){.what = MESH_REDUCE_SCATTER, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = op, .comm = comm,
                                      .stream = stream}, counts);
}


ncclResult_t ncclMeshReduceScatterBuffers(void *const *sendbuffs,void *recvbuff,const size_t *counts,
    ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,cudaStream_t stream) {
  if(!comm||!sendbuffs||!counts)return fail(ncclInvalidArgument,"reduction requires its configured source bindings");
  struct where *views=bind_views(ROOM_SEND_VIEWS,sendbuffs,comm,stream);if(!views)return fail(ncclSystemError,"out of memory");
  return enqueue_counts((struct call){.what=MESH_REDUCE_SCATTER,.send=sendbuffs[0],.recv=recvbuff,.type=datatype,
    .op=op,.comm=comm,.stream=stream,.send_views=views},counts);
}

/* nccl.h ncclMeshAlltoAllV, and NCCL's ncclAlltoAll, ncclGather and ncclScatter as its forms: a counted all-to-all whose
   counts are the host's (hosted), the call's own copy of them (the elements to each rank, then from each: sendcounts
   and recvcounts where given, else `count` from rank `from` to rank `to`, -1 every rank), its `count` those crossing */
static ncclResult_t routed(const void *sendbuff, void *recvbuff, const size_t *sendcounts, const size_t *recvcounts, size_t count, int from, int to,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
                           void *const *send_views,void *const *receive_views) {
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  const int n = comm->nranks, me = comm->rank, root = from < 0 ? to : from;
  if (root < -1 || root >= n) return fail(ncclInvalidArgument, "root %d of %d ranks", root, n);
  struct call c = {.what = WHAT_COUNTED, .send = sendbuff, .recv = recvbuff, .type = datatype, .op = ncclSum, .comm = comm, .stream = stream};
  if (!(c.segments = room_of(ROOM_SEGMENTS, 2 * (size_t)n * sizeof *c.segments))) return fail(ncclSystemError, "out of memory");
  for (int q = 0; q < n; q++) {
    c.segments[q] = sendcounts ? sendcounts[q] : (from < 0 || from == me) && (to < 0 || to == q) ? count : 0;
    c.segments[n + q] = recvcounts ? recvcounts[q] : (from < 0 || from == q) && (to < 0 || to == me) ? count : 0;
    if (q != me) c.count += c.segments[q] + c.segments[n + q];
  }
  c.send_views=send_views?bind_views(ROOM_SEND_VIEWS,send_views,comm,stream):NULL;
  c.receive_views=receive_views?bind_views(ROOM_RECEIVE_VIEWS,receive_views,comm,stream):NULL;
  if((send_views&&!c.send_views)||(receive_views&&!c.receive_views))return fail(ncclSystemError,"out of memory");
  return enqueue(c);
}
ncclResult_t ncclMeshAlltoAllV(const void *sendbuff, const size_t *sendcounts, void *recvbuff, const size_t *recvcounts, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
  if (!sendcounts || !recvcounts) return fail(ncclInvalidArgument, "ncclMeshAlltoAllV: sendcounts or recvcounts is NULL");
  return routed(sendbuff, recvbuff, sendcounts, recvcounts, 0, -1, -1, datatype, comm, stream, NULL, NULL);
}
ncclResult_t ncclMeshAlltoAllBuffers(void *const *sendbuffs,const size_t *sendcounts,void *const *recvbuffs,
    const size_t *recvcounts,ncclDataType_t type,ncclComm_t comm,cudaStream_t stream) {
  if(!sendbuffs||!recvbuffs||!sendcounts||!recvcounts)return fail(ncclInvalidArgument,"exchange requires its configured bindings");
  return routed(sendbuffs[0],recvbuffs[0],sendcounts,recvcounts,0,-1,-1,type,comm,stream,sendbuffs,recvbuffs);
}
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, NULL, count, -1, -1, datatype, comm, stream, NULL, NULL);
}
ncclResult_t ncclAlltoAllConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}
ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, NULL, count, -1, root == -1 ? -2 : root, datatype, comm, stream, NULL, NULL);
}
ncclResult_t ncclGatherConfig(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                              cudaStream_t stream, const ncclCollConfig_t *config) {
  (void)config;
  return ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
}
ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  return routed(sendbuff, recvbuff, NULL, NULL, count, root == -1 ? -2 : root, -1, datatype, comm, stream, NULL, NULL);
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

/* ncclCommSplit: every rank's (color, key) and its report (reach_read; none for NCCL_SPLIT_NOCOLOR) exchanged on comm's
   configuration channel (prepare_exchange), the ranks of this color ordered by key, then rank (two of a color on one
   node refused alike: comm_apart), each rank's streams toward its color's nodes held (fabric_acquire: the streams its parent holds are not free, EBUSY where none is), and a
   second exchange on it carrying each rank's holding's status, then the child's channels (fabric_realize): NCCL's bootstrapSplit (the child's exchange over its parent's), with the init's
   two gathers.  A failure at any rank of a color fails that color's split on each of its ranks as a value, what it held
   let go (comm_free).  Neither exchange is a collective: no group of comm's carries them, inside a group or not
   (design/eager-execution.md: an undeclared collective ends the program) */
static ncclResult_t split(ncclComm_t comm, int color, int key, ncclComm_t *newcomm) {
  const int n = comm->nranks;
  const uint32_t split = comm->splits++;
  struct row { int64_t color, key; struct report report; } mine = {color, key, {.rank = comm->rank}}, *every = calloc((size_t)n, sizeof *every);
  struct binding { int32_t status, padding; ncclUniqueId id; } bound = {0}, *binds = calloc((size_t)n, sizeof *binds);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  struct report *reports = calloc((size_t)n, sizeof *reports);
  *newcomm = NULL;
  if (color != NCCL_SPLIT_NOCOLOR) reach_read(comm->attach, &mine.report);
  ncclResult_t result = !every || !binds || !ranks || !reports ? fail(ncclSystemError, "out of memory")
                                                               : prepare_exchange(comm, &mine, sizeof mine, every);
  const int gathered = !result;
  for (int r = 0; !result && color != NCCL_SPLIT_NOCOLOR && r < n; r++)
    if (every[r].color == color) ranks[count++] = r;
  for (int a = 1; a < count; a++)
    for (int b = a; b > 0 && every[ranks[b]].key < every[ranks[b - 1]].key; b--) { const int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
  if (!result && count) result = comm_sub(comm, ranks, count, split, color, newcomm);
  for (int i = 0; !result && i < count; i++) { reports[i] = every[ranks[i]].report; reports[i].rank = i; }
  if (!result && count > 1 && comm_apart(reports, (uint32_t)count)) result = apart_fail((*newcomm)->name);
  if (!result && count > 1) {
    bound.status = fabric_discover(*newcomm, reports);
    if (!bound.status) bound.status = fabric_acquire(*newcomm);
    /* A child owns its configuration channel too. Later exact-byte WritePrepare
       must not depend on its parent's membership or a payload-side fallback. */
    if (!bound.status && !((*newcomm)->peers = malloc((size_t)count * sizeof(int)))) bound.status = ENOMEM;
    if ((*newcomm)->peers) for (int r = 0; r < count; r++) (*newcomm)->peers[r] = -1;
    if (!bound.status && !(*newcomm)->rank) bound.status = bootstrap_child_id(*newcomm, &bound.id);
  }
  else if (result && count) bound.status = ECANCELED;
  if (gathered) {
    const ncclResult_t confirmed = prepare_exchange(comm, &bound, sizeof bound, binds);
    if (!result) result = confirmed;
  }
  for (int i = 0; !result && i < count; i++)
    if (binds[ranks[i]].status) result = hold_fail((*newcomm)->name, i, binds[ranks[i]].status);
  if (!result && count > 1) {
    struct ncclComm *child = *newcomm;
    memcpy(child->id, binds[ranks[0]].id.internal, NCCL_UNIQUE_ID_BYTES);
    child->id[NCCL_UNIQUE_ID_BYTES] = 0;
    int error = bootstrap_open(child, child->id, reports);
    if (!error) error = exchange(child, reports, sizeof *reports, !child->rank, 1);
    if (error) result = fail(ncclSystemError, "split configuration bootstrap: %s", strerror(error));
  }
  int status = 0;
  if (!result && count > 1 && (status = fabric_realize(*newcomm)))
    result = fail(ncclSystemError, "communicator %s: its channels on the streams it held: %s", (*newcomm)->name, strerror(status));
  if (result && *newcomm) { comm_free(*newcomm); *newcomm = NULL; }
  free(every); free(binds); free(ranks); free(reports);
  return result;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return fail(ncclInvalidArgument, "comm or newcomm is NULL");
  return split(comm, color, key, newcomm);
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

/* ncclCommGrow: NCCL's bootstrap at the id of comm's root (ncclCommGetUniqueId), which comm's other ranks take from
   the root's row of an exchange on comm's configuration channel (prepare_exchange: no collective); comm's ranks keep their ranks (the `rank` they
   pass, NCCL's -1, not read) and a new one (comm NULL) takes `rank`, every rank reporting its node and the nodes its
   bridge reaches; the communicator, named by the id, holds its streams as every init does */
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!newcomm || (comm ? nRanks < comm->nranks || (!comm->rank && !uniqueId) : !uniqueId))
    return fail(ncclInvalidArgument, "ncclCommGrow's arguments");
  if (!comm) return ncclCommInitRank(newcomm, nRanks, *uniqueId, rank);
  ncclUniqueId *ids = calloc((size_t)comm->nranks, sizeof *ids);
  const ncclUniqueId mine = comm->rank ? (ncclUniqueId){{0}} : *uniqueId;
  ncclResult_t result = ids ? prepare_exchange(comm, &mine, sizeof mine, ids) : fail(ncclSystemError, "out of memory");
  const ncclUniqueId given = ids && !result ? ids[0] : mine;
  free(ids);
  return result ? result : ncclCommInitRank(newcomm, nRanks, given, comm->rank);
}

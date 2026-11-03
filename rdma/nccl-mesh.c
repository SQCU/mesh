#include "mesh-write-compiler.h"
/* design/whole-function.md#fabric */
#include "nccl.h"
#include "mesh-call.h"
#include "mesh-plan.h"
#include "mesh-graph.h"
#include "mesh.h"
#include "mesh-span-plan.h"
#include "mesh-landing.h"
#include "metal.h"
/* a count of a holding's units (mesh.h struct mesh_stream ready, sent and landed) */
typedef struct mesh_pages mesh_pages;
/* the partials merge's operator code (nccl.h ncclMeshRedOpCreatePartials), beside an ncclDataType_t's */
#define MESH_MERGE 12
/* `*items`, an array of `count` elements of `size` bytes, grown to hold one more; 0, or ENOMEM */
static int grow(void *items, size_t *room, size_t count, size_t size) {
  if (count < *room) return 0;
  const size_t want = *room ? 2 * *room : 8;
  void *made = realloc(*(void **)items, want * size);
  if (!made) return ENOMEM;
  *(void **)items = made; *room = want;
  return 0;
}
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
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>


enum { WHAT_COUNTED = ncclMeshCallAlltoAllV };
_Static_assert(ncclMeshCallAllReduce == MESH_ALLREDUCE && ncclMeshCallBroadcast == MESH_BROADCAST && ncclMeshCallReduce == MESH_REDUCE &&
               ncclMeshCallReduceScatter == MESH_REDUCE_SCATTER && ncclMeshCallAllGather == MESH_ALLGATHER,
               "nccl.h's declared kinds are the planner's collectives");
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN, COMBINE_MERGE };

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
/* a failure's message: its text, and `id`, where the message is kept with its failure (a communicator's, comm_fail; a
   non-blocking init's, init_run), the count of messages the process had kept when it was (`messages`), else 0 */
struct said { uint64_t id; char text[MESH_NCCL_DIAGNOSTIC_BYTES]; };
/* the unit a WRITE is counted in, a byte: a WRITE moves page-local byte intervals, and the values they hold (of any type,
   sub-byte ones too) are its producer's and consumer's, never the transfer's */
#define WRITE_UNIT ncclUint8
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





/* -- communicators -------------------------------------------------------------------------------- */

/* an operator made on the communicator: PreMulSum's scalar (`value`, or where it lies, `at`, read at each host-path
   call: ncclScalarDevice), or the partials merge's row of `unit` elements */
struct redop { int used; ncclDataType_t type; unsigned char value[8]; uint32_t unit; const void *at; };
/* a compiled collective the communicator runs for the calls it fits (ncclMeshProgram), its own copy */
struct entry { uint64_t below; struct mesh_program program; struct mesh_trees trees; int whole; };

/* a channel (design/transport.md section 14): toward rank `peer` on node `node`, the holding of the stream toward that
   node its process holds (mesh-call.h mesh_stream_view: its two lists and its counts), and the Metal buffer over the
   page that holds the counts (`buffer`: `ready` and `landed` at those offsets in it; `ready_at` the region
   offset of `ready`).  Its entries, each list in its own order: `sends` and `lands` laid by the
   groups scheduled, `out` and `in` their pages (the value each count reaches once they are whole).  The host's mirror
   of what the GPU had published and consumed (`published`, `consumed`, `waited`) is deleted with the tracks pool:
   every group is published and awaited at its own call (the shared source-effect solve) */
struct channel {
  int peer;
  uint32_t node, stream, cable;
  const uint64_t *wire;
  struct mesh_stream_view view;
  void *buffer;
  size_t ready, landed, sent;
  uint64_t ready_at, sends, lands;
  mesh_pages out, in;
  /* zinth's link word for the next occurrence: none before the first (the stream's zwoggle), then the last table's
     next (ncclMeshCommLowerOccurrence) */
  void *link;
  uint64_t linked;
};

/* design/zorvek.md: a declared group of calls, routed once: every call's steps at this rank (mesh-plan.h struct
   mesh_step), in call order */
struct definition {
  uint64_t id;
  size_t calls, steps;
  ncclMeshCollective *call;
  struct mesh_step *step;
  uint32_t *step_call, *step_channel; /* each step's call, and its channel (spread: a pair's parallel cables) */
  struct definition *next;
};
/* design/whole-function.md#interface */
struct attach;
struct plan;
struct ncclComm {
  struct mesh_link_map map;
  struct mesh_fabric fabric;
  uint32_t *endpoint;
  struct mesh_paths *writes;
  unsigned char *write_overrides;
  /* `writes` composed with the cables (mesh_write_routes_compose): a pair no route was installed for that a cable
     joins takes its cables, any other pair its `writes` paths. Made at realization and at each installation
     (ncclMeshCommWriteRoutes), the only times either changes; the tree collectives' lowering reads it
     (bind_routed_steps) */
  struct mesh_paths *composed;
  uint64_t page;
  struct mesh_trees paths, trees, rooted;
  struct entry *entries, *given;
  ncclMeshForm *forms;
  uint32_t programs, splits, *members, form_count, count, nops;
  int nranks, rank, listed, compiled, binding;
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
  _Atomic uint64_t *words;
  void *violation;
  uint64_t groups, definitions;
  struct ncclComm *next_live;
  char *cpu_compiler,*cpu_resources;
  _Atomic int runs;
  /* the collective lowering's (design/zorvek.md): the declared groups (routed along `trees`, which own their arrays:
     routes_free) */
  struct definition *declared;
};

/* the groups' one lock: a group's planning, scheduling and encoding, a completion, a communicator's failure kept and its
   end. */
static pthread_mutex_t group_lock = PTHREAD_MUTEX_INITIALIZER;
/* the communicators realized and not freed (fabric_realize, comm_free), for the expiry's stop */
static pthread_mutex_t live_lock = PTHREAD_MUTEX_INITIALIZER;
static struct ncclComm *live;

/* design/transport.md section 5: the program's end at an expiry, its GPU work stopped first: every communicator's GPU
   waits ended through their failure words' claim (nccl-mesh-kernels.h mesh_step_failed, read every 2^10 loads, the
   tracks' too) and its stop (words[6]), and then the library's GPU programs awaited within MESH_PLATFORM_NS; the wait
   named (mesh.h mesh_expired) */
/* An unusable declaration or an exhausted execution observation: report the
   actual cause and stop GPU work before ending the program. A poll budget is
   not a measured wall-clock duration. */
/* the nanoseconds left before d's deadline; past it, the library's expiry (mesh_library_expired) */
static uint64_t deadline_left(const struct mesh_deadline *d) {
  const uint64_t now = mesh_clock_ns();
  if (now >= d->at) mesh_expired(d->wait, d->ns);
  return d->at - now;
}
/* Report an exhausted execution-local poll budget. No preparation or elapsed
   timestamp exists in these records. Every channel's counts accompany the
   exact entry and load budget that failed. */
static void program_free(struct mesh_program *program) {
  free((void *)program->segment); free((void *)program->log_weight); free((void *)program->first); free((void *)program->move);
}

static void trees_free(struct mesh_trees *trees) {
  free((void *)trees->root); free((void *)trees->parent); free((void *)trees->log_weight); free((void *)trees->latency);
  *trees = (struct mesh_trees){0};
}

/* trees' own copy into `made` (its arrays allocated even for none); 0, or ENOMEM */
static int trees_copy(struct mesh_trees *made, const struct mesh_trees *from) {
  uint32_t *root = calloc(from->count + 1, sizeof *root), *parent = calloc((size_t)from->count * from->nodes + 1, sizeof *parent);
  double *weight = calloc(from->count + 1, sizeof *weight), *latency = from->latency ? calloc(from->count + 1, sizeof *latency) : NULL;
  *made = (struct mesh_trees){from->nodes, from->count, root, parent, weight, latency, from->rate};
  if (!root || !parent || !weight || (from->latency && !latency)) return ENOMEM;
  if (from->count) {
    memcpy(root, from->root, from->count * sizeof *root); memcpy(weight, from->log_weight, from->count * sizeof *weight);
    if (latency) memcpy(latency, from->latency, from->count * sizeof *latency);
    memcpy(parent, from->parent, (size_t)from->count * from->nodes * sizeof *parent);
  }
  return 0;
}

static void map_free(struct mesh_link_map *map) {
  free(map->link); free(map->bandwidth); free(map->next); free(map->mark);
  *map = (struct mesh_link_map){0};
}

/* the trees and the table compiled on the map; with `all` (the communicator's end), the map, the caller's paths and
   programs */
static void routes_free(struct ncclComm *comm, int all) {
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
static void paths_free(struct mesh_paths *paths) {
  free((void *)paths->first); free((void *)paths->cable); free((void *)paths->weight);
  *paths = (struct mesh_paths){0};
}
/* comm's tables returned to the region (ncclMemFree), after its holdings are given back (comm_leave: the bridge posts
   nothing more of a given holding, so only rows a unit already posted names stay held, mesh-dataflow.c mesh_pinned).
   The landing gate's word is deleted with the gate (design/zorvek.md "Instances") */
static void comm_storage_free(struct ncclComm *comm) { (void)comm; }
static void comm_free(struct ncclComm *comm) {
  pthread_mutex_lock(&live_lock);
  for (struct ncclComm **at = &live; *at; at = &(*at)->next_live) if (*at == comm) { *at = comm->next_live; break; }
  pthread_mutex_unlock(&live_lock);
  bootstrap_close(comm);
  routes_free(comm, 1);
  for (struct definition *d = comm->declared, *next; d; d = next) {
    next = d->next; free(d->call); free(d->step); free(d->step_call); free(d->step_channel); free(d);
  }
  for (uint32_t h = 0; h < comm->count; h++) MetalRelease(comm->channels[h].buffer);
  ncclMemFree((void *)comm->words);
  comm_leave(comm);
  free(comm->channels); free(comm->endpoint); free((void *)comm->fabric.cable);
  for (int p = 0; comm->writes && p < comm->nranks * comm->nranks; p++) paths_free(&comm->writes[p]);
  free(comm->writes); free(comm->write_overrides);
  mesh_write_routes_free(comm->composed, (uint32_t)comm->nranks);
  free(comm->ops);
  pthread_mutex_destroy(&comm->guard);
  pthread_mutex_destroy(&comm->prepare);
  free(comm->members); free(comm->forms);
  free(comm->cpu_compiler);free(comm->cpu_resources);
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

/* `what` at `root` along `trees` compiled into *out, its arrays its own (mesh_compile_room): every member's
   (mesh_compile), or the declared `contributors` and `recipients` alone where either is given (mesh_compile_members);
   0, or ENOMEM or EINVAL with nothing kept */

/* `what` (at `root`) along `trees` (mesh-plan.h mesh_compile: each step's piece one entry, which the bridge cuts into
   SENDs), an all-reduce single-phase where `whole`, compiled into the table's next entry, for calls of fewer bytes than
   `below` (0: any size); 0, or EINVAL */

static int has_root(int what) { return what == MESH_REDUCE || what == MESH_BROADCAST; }


/* a configured form's entries (ncclMeshCommForms): its collective along the communicator's trees (`along`) or the map's
   packed trees (mesh_trees_pack at the form's eps and most, made here: mesh cc8a6f8e^ comm_routes' sets[1]; a segment
   cut among a root's trees by the form's shares, else by the packing's flows; a tree of share 0 carries none of it; the
   communicator's trees cut alike where the form gives shares), whole or two-phase, below its bound; a rooted one at
   its root, or at each root (-1), none where its root is past the communicator's ranks (a split's).  A root every one
   of whose trees takes share 0 has no tree to carry its segment: the form fails there as a value */

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

/* The communicator's logical programs name rank endpoints. Physical lowering
   expands every transaction over the configured cables and relay paths. */

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


/* whether this rank forwards a WRITE from `source` to `destination`: an endpoint of a cable of an installed path of
   nonzero weight, and neither end of the write; 1 where no path is installed (the routes are the library's to choose) */


/* nccl.h ncclMeshCommChannels */
ncclResult_t ncclMeshCommChannels(ncclComm_t comm, int *channels) {
  if (!comm || !channels) return fail(ncclInvalidArgument, "ncclMeshCommChannels: a NULL argument");
  *channels = (int)comm->count;
  return ncclSuccess;
}
/* nccl.h ncclMeshCommWriteEntries: every rank's transfers of each WRITE (write_transfers, the compile its steps take),
   their SENDs counted on each directed cable (2e + whether the sender is cable e's node[1]) */

/* the program a call of `bytes` runs: its communicator's table's first entry of its collective (and root, a rooted
   one's) whose bound it is below (a configured form's, else the default's); NULL where none is (mesh 2bf5a50d^
   program_of; cc8a6f8e^'s by size class, the classes now the forms' bounds) */

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
struct lease {
  void *address; uint32_t row; size_t bytes; void *buffer;
  struct mesh_lease_retire reads; int returned;
  /* A composed local view holds original leases, not new arena ownership.
     Its flattened physical runs and owners are resolved during preparation. */
  size_t pins, owners_count;
  const void **owners;
  struct mesh_page_runs backing;
};
/* design/transport.md section 14: the stream toward node `node` the process holds (`users` the communicator on it, 0
   none: the hold vacant), its holding's view, and the identity it was taken by (`wire`: mesh-call.h mesh_stream_held,
   comm_failure) */
struct hold { uint32_t node, stream, users, padding; struct mesh_stream_view view; uint64_t wire, generation; };
struct attach {
  struct mesh_ctx context;
  struct lease *leases;
  size_t count, room;
  /* each local virtual row's lease, its index + 1 (0: none), so finding a byte's lease is its row's load (lease_at); and the
     leases returned and not yet given back (leases_reap) */
  uint32_t *lease_of;
  size_t returned;
  /* the returned leases' addresses, so a reap visits those alone (leases_reap) */
  void **returning;
  size_t returning_room;
  uint64_t comms;
  /* whether the library made a Metal buffer over the region (a lease's, lease_buffer; a holding's counts', channel_buffer):
     the caller's GPU work, and its residency set, may hold those pages after the buffer is released, so such an attach
     is never let go and its process ends to leave the region (attach_settle) */
  _Atomic int wrapped;
  /* one Metal buffer over the region's stream pages (mesh.h mesh_layout: [stream_off, client_off), every holding's
     counts, which the bridge writes and a kernel observes), made at its first need (stream_page) */
  void *stream_buffer;
  /* the composed leases (owners_count: views over other leases' pages, each mapped where mmap put it, so no row of the
     client view finds one) by address, sorted, for lease_at's binary search: rebuilt at a lookup once one was added,
     given back or moved (`composed_stale`), its room grown where one is composed (ncclMeshPageTableCompose).  A scan of
     every lease for every byte no row finds took half of bandwidth.py's preparation on main, over 30 s with no call
     made (2026-11-01, sampled: lease_at's loop inlined at ncclMeshRegionRuns+260, a lease's 0x78 bytes a step) */
  struct composed_at { uintptr_t lo, hi; size_t lease; } *composed;
  size_t composed_count, composed_room, composed_total;
  int composed_stale;
  struct hold holds[MESH_STREAMS];
  struct attach *next;
};
/* the process's one attach of the node's region (`attaches`, made at its first lease or communicator), under attach_lock */
static struct attach *attaches;

/* comm's link on channel h lost (its stream no longer its own: a peer gone ends its holding), recorded in its failure
   words where no wait has recorded a failure (their claim, nccl-mesh-kernels.h mesh_step_claim), so every GPU wait on
   them ends within 2^10 loads (mesh_step_failed) and the host reads the loss as a value (comm_failure) */
/* Observe link loss and reported poll exhaustion. GPU/CPU load budgets begin
   inside their executing observations. This watcher neither starts nor renews
   those budgets, and preparation time is not an elapsed observation. */
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
  free(a->leases); free(a->lease_of); free(a->composed); free(a->returning);
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
static struct lease *lease_at(const void *ptr, struct attach **held);
static ncclResult_t mem_alloc(void **ptr, size_t size, enum mesh_backing backing);
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
  if (!checked) checked = mesh_write_routes_compose(&comm->fabric, comm->writes, comm->write_overrides, &comm->composed);
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
  int on = 1, off = 0, fd = socket(AF_INET6, SOCK_STREAM, 0);
  if (inet_pton(AF_INET6, host, &at.sin6_addr) != 1) at.sin6_addr = in6addr_any;
  /* an IPv4-mapped host (the tailnet's IPv4, tailnet_address) is bound on a dual-stack socket */
  if (fd >= 0 && !setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on) &&
      !setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof off) && !bind(fd, (struct sockaddr *)&at, sizeof at) &&
      !listen(fd, 64) && !getsockname(fd, (struct sockaddr *)&at, &length) && !fcntl(fd, F_SETFL, O_NONBLOCK)) {
    *port = ntohs(at.sin6_port);
    return fd;
  }
  const int error = errno;
  if (fd >= 0) close(fd);
  errno = error;
  return -1;
}

/* a bootstrap root's address: the node's tailnet IPv4 address (in the tailnet's 100.64.0.0/10, on its utun interface),
   written IPv4-mapped (::ffff:a.b.c.d) for the bootstrap's IPv6 sockets. The bootstrap carries the communicator's
   configuration only; its transfers are the bridge's. Changed by the LSTM demonstration session from the fabric address
   (MESH_ULA_PREFIX, bin/mesh-router-init.sh's host routes over each Thunderbolt port): with one of m5ult2's ports
   stopped, m5ult2 reached only the Mini's fabric address and every five-rank bootstrap expired awaiting it (2026-11-01
   17:1x). The tailnet's IPv6 range is no way around it: babel holds a /128 host route for each member's tailnet IPv6
   address over a Thunderbolt port, more specific than the tailnet's own /48 on utun0, so it too reached nothing. Its
   IPv4 range has no babel route: m5ult2 reached m5ult4 at ::ffff:100.107.135.46 in 0.12 s. The operator: "everything
   should be connected by tailnet". What must stay: a route the Thunderbolt fabric does not carry. */
static ncclResult_t tailnet_address(char *address, size_t room) {
  if (!address || !room) return fail(ncclInvalidArgument, "the tailnet address: no room");
  *address = 0;
  struct ifaddrs *interfaces = NULL;
  if (getifaddrs(&interfaces)) return fail(ncclSystemError, "the node's interfaces (getifaddrs): %s", strerror(errno));
  int short_room = 0;
  for (struct ifaddrs *i = interfaces; i && !*address && !short_room; i = i->ifa_next) {
    char text[INET_ADDRSTRLEN];
    if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || (i->ifa_flags & IFF_LOOPBACK)) continue;
    const uint32_t v4 = ntohl(((struct sockaddr_in *)i->ifa_addr)->sin_addr.s_addr);
    if ((v4 & 0xffc00000u) != 0x64400000u ||
        !inet_ntop(AF_INET, &((struct sockaddr_in *)i->ifa_addr)->sin_addr, text, sizeof text))
      continue;
    if (strlen("::ffff:") + strlen(text) < room) snprintf(address, room, "::ffff:%s", text);
    else short_room = 1;
  }
  freeifaddrs(interfaces);
  if (short_room) return fail(ncclInvalidArgument, "the tailnet address: %zu bytes hold no tailnet address", room);
  return *address ? ncclSuccess
                  : fail(ncclSystemError, "this node holds no tailnet address (no interface's address in 100.64.0.0/10)");
}

/* A split owns a fresh configuration listener. It neither honors the parent's
   NCCL_COMM_ID nor replaces the unrelated process-global public bootstrap. */

/* the id: NCCL_COMM_ID where set, else a listener on the node's tailnet address (tailnet_address) where it holds one,
   else on its host name */
ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  memset(id->internal, 0, sizeof id->internal);
  const char *given = getenv("NCCL_COMM_ID");
  if (given && *given) { snprintf(id->internal, sizeof id->internal, "%s", given); return ncclSuccess; }
  char host[256] = {0};
  if (tailnet_address(host, sizeof host)) *host = 0;
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
  mesh_expired(wait, comm->until.ns);
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
/* `bytes` at `data` made the root's on every rank, on comm's bootstrap sockets within its phase (`until`): a root
   other than rank 0 sends them to rank 0, which sends them to each other rank.  NCCL's bootstrapBroadcast; never a
   payload's.  0, or the errno (whole) */

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
  const uintptr_t at[] = {(uintptr_t)ch->view.ready, (uintptr_t)ch->view.landed, (uintptr_t)ch->view.sent};
  uintptr_t lo = at[0], hi = at[0] + sizeof(uint64_t);
  for (size_t k = 1; k < sizeof at/sizeof *at; k++) { lo = at[k] < lo ? at[k] : lo; hi = at[k] + sizeof(uint64_t) > hi ? at[k] + sizeof(uint64_t) : hi; }
  lo &= ~(page - 1);
  hi = (hi + page - 1) & ~(page - 1);
  ch->ready = at[0] - lo; ch->landed = at[1] - lo;ch->sent=at[2]-lo;
  ch->ready_at = at[0] - (uintptr_t)m;
  return (ch->buffer = MetalWrap((void *)lo, hi - lo)) ? 0 : ENOMEM;
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
  for (uint32_t e = 0; !status && e < comm->fabric.cables; e++) {
    if (comm->endpoint[e] == UINT32_MAX) continue;
    const struct mesh_cable *c = comm->fabric.cable + e;
    const uint32_t peer = c->node[c->node[0] == (uint32_t)comm->rank];
    struct hold *h = comm->attach->holds + comm->endpoint[e];
    struct channel *ch = comm->channels + comm->count++;
    *ch = (struct channel){.peer = (int)peer, .node = comm->members[peer], .stream = h->stream,
      .cable = e, .wire = &h->wire, .view = h->view};
    if (!(status = channel_buffer(ch, m))) {
      atomic_store(&comm->attach->wrapped, 1);
    }
  }
  /* live once its channels are made, the watch reading them (watch_turn) */
  if (!status) {
    pthread_mutex_lock(&live_lock);
    comm->next_live = live; live = comm;
    pthread_mutex_unlock(&live_lock);
    
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

/* Stop the communicator, finish its configuration transactions, then release
   channels and compiled declarations. The enclosing function recorder owns
   submitted work and completes it before the communicator is destroyed. */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  comm_stop(comm);
  if (comm->initing) {
    const struct mesh_deadline ended = mesh_deadline("ncclCommDestroy: its non-blocking init's end (init_run's `async`)", 0);
    while (__atomic_load_n(&comm->async, __ATOMIC_ACQUIRE) == ncclInProgress) { deadline_left(&ended); usleep(20); }
    pthread_join(comm->init, NULL);
  }
  struct mesh_deadline returned = mesh_deadline("ncclCommDestroy: configuration transactions still active", 0);
  pthread_mutex_lock(&group_lock);
  while (atomic_load(&comm->runs)) {
    pthread_mutex_unlock(&group_lock);
    deadline_left(&returned);
    usleep(20);
    pthread_mutex_lock(&group_lock);
  }
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
  *asyncError = ncclSuccess; /* a lost link fails the run that waits on its landing (metal.h MetalRunLanded) */
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
  struct attach *owner = NULL;
  const struct lease *lease = lease_at(buff, &owner);
  const size_t offset = lease ? (uintptr_t)buff - (uintptr_t)lease->address : 0;
  const int valid = lease && !lease->returned && owner == comm->attach && size <= lease->bytes - offset;
  pthread_mutex_unlock(&attach_lock);
  if (!valid)
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
      .landed = atomic_load_explicit(h->view.landed, memory_order_acquire),
      .left = {atomic_load_explicit(h->view.taken[0], memory_order_acquire), atomic_load_explicit(h->view.taken[1], memory_order_acquire)}};
    return mesh_stream_held(&a->context, stream, h->wire) ? 0 : ENOLINK;
  }
  return ENOENT;
}

/* design/reclamation-2026-10-30.md: a lease's dependency on a holding is done with once no hold of this attach still
   uses that holding and the bridge holds none of this client's pages on its stream: the holding acknowledged (its
   owner word moved: the bridge reads none of its tables) and, where it ended unfinished, its units done or its port
   child reaped (mesh-dataflow.c mesh_holding_retained) */
static int lease_detachable(void *context, const struct mesh_lease_retire_dep *d) {
  struct attach *a = context;
  for (uint32_t k = 0; k < MESH_STREAMS; k++) {
    const struct hold *h = a->holds + k;
    if (h->users && h->stream == d->stream && h->generation == d->holding) return EBUSY;
  }
  if (d->stream >= MESH_STREAMS) return EINVAL;
  return mesh_holding_retained(a->context.M, d->stream, a->context.client) ? EBUSY : 0;
}
static int lease_returnable(struct attach *a, struct lease *l) {
  return mesh_lease_retire_detachable(&l->reads, lease_read_sample, lease_detachable, a);
}
static uint64_t lease_page_offset(const struct attach *a,const struct lease *l,size_t offset) {
  return l->owners_count?mesh_page_run_offset(&l->backing,offset):mesh_page_offset(a->context.M,l->row,offset);
}
/* The prepared local view's row, independent of its physical backing. */
static uint32_t view_row(const struct attach *a, const void *ptr) {
  const struct hdr *m = a->context.M;
  const uintptr_t data = (uintptr_t)a->context.view, at = (uintptr_t)ptr;
  const uint64_t row = !data || at < data ? UINT64_MAX : (at - data) / ((size_t)m->block * m->pgsz);
  return row < mesh_rows(m) ? (uint32_t)row : UINT32_MAX;
}
/* lease `i`'s pages named `value` in the page index (i + 1: its own; 0: none) */
static void lease_mark(struct attach *a, size_t i, uint32_t value) {
  const struct lease *l = a->leases + i;
  if(l->owners_count)return;
  const uint32_t rows = (uint32_t)(l->bytes / ((size_t)a->context.M->block * a->context.M->pgsz));
  for (uint32_t r = 0; r < rows; r++) a->lease_of[l->row + r] = value;
}
/* returned lease `i` given back to the region where its reads are done (lease_returnable), the last lease moved to its
   place: 1, else 0 (it stays, returned) */
static int lease_give(struct attach *a, size_t i) {
  struct lease *l = a->leases + i;
  if (!l->returned || l->pins || !lease_returnable(a, l)) return 0;
  if(l->owners_count) {
    for(size_t k=0;k<l->owners_count;k++) {
      struct lease *owner=lease_at(l->owners[k],NULL);
      owner->pins--;
    }
  } else if(mesh_lease_return(&a->context,l->row))return 0;
  if (l->buffer) MetalRelease(l->buffer);
  if(l->owners_count)munmap(l->address,l->bytes);
  free(l->owners);free(l->backing.at);
  mesh_lease_retire_destroy(&l->reads);
  lease_mark(a, i, 0);
  if (l->owners_count) a->composed_total--;
  a->composed_stale = 1;
  *l = a->leases[--a->count];
  if (i < a->count) lease_mark(a, i, (uint32_t)i + 1);
  return 1;
}
/* every returned lease whose reads are done given back (a lease returned with reads outstanding waits here), the
   returned ones alone visited (`returning`), again while one is given back (a composed view's return unpins its owners:
   the graph module's mesh_graph_settle, design/rubrics.md C9).
   Changed by the LSTM demonstration session: each reap scanned every lease, and a lease returned with reads outstanding
   made every allocation scan them all, allocations x leases (mem_alloc's leases_reap, the 64-step LSTM's source
   generation, 2026-11-01). What must stay: each returned lease given back once its reads are done and none is pinned. */
static int lease_take(void *consumer, void *address) {
  struct attach *a = consumer;
  struct lease *l = lease_at(address, NULL);
  return l && lease_give(a, (size_t)(l - a->leases));
}
static void leases_reap(struct attach *a) {
  a->returned = mesh_graph_settle(a->returning, a->returned, lease_take, a);
}

/* a lease of the region: its whole pages covering `size` (one page for 0), already registered on every device of the
   node's bridge, owned by the process's client; never other memory */
static ncclResult_t mem_alloc(void **ptr, size_t size, enum mesh_backing backing) {
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
  else if (grow(&a->leases, &a->room, a->count, sizeof *a->leases) ||
           (!a->lease_of && !(a->lease_of = calloc(mesh_rows(a->context.M), sizeof *a->lease_of))))
    result = fail(ncclSystemError, "out of memory");
  else if ((status = mesh_section_create(&a->context, bytes, 1, backing, &lease))) {
    const struct hdr *m = a->context.M;
    const size_t quantum = (size_t)m->block * m->pgsz;
    result = fail(ncclSystemError, "ncclMemAlloc of %zu bytes (%zu pages of %u bytes): the region of this node's bridge has %u of "
                  "its %u pages free (%s); a client sizes its leases to the region its node's bridge configuration states "
                  "(etc/bridge/README.md)", size, (bytes + quantum - 1) / quantum * m->block, m->pgsz, mesh_arena_free(a->context.M),
                  mesh_arena_pages(m), strerror(status));
  } else {
    *ptr = mesh_section_address(&a->context, lease, 0);
    a->leases[a->count++] = (struct lease){.address = *ptr, .row = lease.first, .bytes = (size_t)lease.pages * a->context.M->pgsz};
    lease_mark(a, a->count - 1, (uint32_t)a->count);
  }
  pthread_mutex_unlock(&attach_lock);
  return result;
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) { return mem_alloc(ptr, size, MESH_BACKING_PAGES); }
/* design/page-maps-2026-11-02.md: a lease's physical page runs, read once by the allocator's binding (an effector) at
   the lease's allocation, so a value's map is handed out with its lease and never recovered from an address by a
   chooser. type: (a lease's own address, capacity) -> [Backing] covering the lease's pages in order: run k's `first`
   its first page in the lease, `count` pages, `page` the first physical page of the region's page space. *count in:
   the capacity; out: the runs the lease has (runs NULL: count only) */
/* the region's page size, in bytes: a chooser's figure for page arithmetic on identities (design/page-maps-2026-11-02.md) */
ncclResult_t ncclMeshRegionPageBytes(uint64_t *bytes) {
  if (!bytes) return fail(ncclInvalidArgument, "the page size needs somewhere to go");
  pthread_mutex_lock(&attach_lock);
  struct attach *a = NULL;
  const int status = attach_current(&a);
  if (!status) *bytes = a->context.M->pgsz;
  pthread_mutex_unlock(&attach_lock);
  return status ? fail(ncclSystemError, "no region attached: %s", strerror(status)) : ncclSuccess;
}
ncclResult_t ncclMeshLeasePages(const void *lease, ncclMeshPageBacking *runs, size_t *count) {
  if (!lease || !count) return fail(ncclInvalidArgument, "a lease's pages need a lease and somewhere to go");
  pthread_mutex_lock(&attach_lock);
  struct attach *a = NULL;
  const struct lease *l = lease_at(lease, &a);
  if (!l || l->address != lease) {
    pthread_mutex_unlock(&attach_lock);
    return fail(ncclInvalidArgument, "%p is no lease's own address", lease);
  }
  const size_t page = a->context.M->pgsz, pages = l->bytes / page;
  size_t written = 0;
  for (size_t p = 0; p < pages;) {
    const uint64_t first = lease_page_offset(a, l, p * page) / page;
    size_t n = 1;
    while (p + n < pages && lease_page_offset(a, l, (p + n) * page) / page == first + n) n++;
    if (runs && written < *count) runs[written] = (ncclMeshPageBacking){p, n, first};
    written++; p += n;
  }
  pthread_mutex_unlock(&attach_lock);
  *count = written;
  return ncclSuccess;
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
  struct attach *a = NULL;
  struct lease *l = lease_at(ptr, &a);
  const int found = l && l->address == ptr && !l->returned;
  const int recorded = found && !grow((void **)&a->returning, &a->returning_room, a->returned, sizeof *a->returning);
  if (recorded) { l->returned = 1; a->returning[a->returned++] = ptr; leases_reap(a); }
  attach_settle();
  pthread_mutex_unlock(&attach_lock);
  return recorded ? ncclSuccess : found ? fail(ncclSystemError, "ncclMemFree of %p: out of memory recording its return", ptr)
                 : fail(ncclInvalidArgument, "ncclMemFree of %p: no live lease of this process", ptr);
}

/* the lease holding ptr and its attach (*held, where held is not NULL), under attach_lock (NULL: none) */
static int composed_order(const void *x, const void *y) {
  const uintptr_t a = ((const struct composed_at *)x)->lo, b = ((const struct composed_at *)y)->lo;
  return a < b ? -1 : a > b;
}
/* the attach's composed leases sorted by address (their room is composed_total's, grown where one is composed) */
static void composed_index(struct attach *a) {
  a->composed_count = 0;
  for (size_t i = 0; i < a->count; i++) if (a->leases[i].owners_count)
    a->composed[a->composed_count++] = (struct composed_at){(uintptr_t)a->leases[i].address,
                                                            (uintptr_t)a->leases[i].address + a->leases[i].bytes, i};
  qsort(a->composed, a->composed_count, sizeof *a->composed, composed_order);
  a->composed_stale = 0;
}
static struct lease *lease_at(const void *ptr, struct attach **held) {
  for (struct attach *a = attaches; a; a = a->next) {
    const uint32_t row = a->lease_of ? view_row(a, ptr) : UINT32_MAX, k = row == UINT32_MAX ? 0 : a->lease_of[row];
    if (k) {
      if (held) *held = a;
      return a->leases + k - 1;
    }
    if (a->composed_stale) composed_index(a);
    const uintptr_t at = (uintptr_t)ptr;
    size_t lo = 0, hi = a->composed_count;
    while (lo < hi) { const size_t mid = lo + (hi - lo) / 2; if (a->composed[mid].hi <= at) lo = mid + 1; else hi = mid; }
    if (lo < a->composed_count && a->composed[lo].lo <= at) {
      if (held) *held = a;
      return a->leases + a->composed[lo].lease;
    }
  }
  return NULL;
}

/* Preparation resolves a local view to its bridge page. A descriptor table
   explicitly requests linear backing before adding any table-relative offset. */
/* Changed by the LSTM demonstration session (2026-11-02): a word of the region's own records lies in no lease, e.g. a
   channel's `landed`, which landing observations name as {channel->buffer, channel->landed}. Its offset is its place in
   the region's own mapping. lease_at's NULL was read through, and bandwidth's bind faulted in ncclMeshCommSourceInputs
   on a SEND whose source landed earlier (metal-microbench output_data/lstm-2026-11-02/d-main3, m5ult2). What must
   stay: a lease's page offset where the address is in one. The joint change of design/page-maps-2026-11-02.md takes
   this pointer-to-offset step out of chooser code. */

/* the leases a prepared owner's entries read (a placed issue's, a prepared WRITE's), each a registration scope open
   from its preparation to its end (owned_begin, owned_end), so no present registers its reads (mesh-lease-retire.h: an
   open scope covers the uses not yet enumerated).  `uses`, a channel's sends and landings each, are fixed when it is
   prepared but `end`, which present_group advances.  Each lease's `kind`: OWNED_TABLE a descriptor table's (metadata),
   OWNED_SENDS read by its sends alone (a source readiness table).  Under attach_lock */
struct owned_lease { const void *address; unsigned char kind; };
struct owned { struct owned_lease *leases; size_t count, room; struct mesh_lease_retire_dep *uses; uint32_t channels; };
enum { OWNED_TABLE = 1, OWNED_SENDS = 2 };
/* `o`'s uses at the holdings `comm` has now: each channel's two directions (2h + d: units sent or landed), then its two
   walks' (2 count + 2h + d: the calls each walk has left, design/zorvek.md zinth: the issue's linked tables, read
   until the walk leaves them, so a table lease alone takes them: owned_end).  0, ENOMEM.  Under attach_lock */
/* a scope opened on each of `o`'s leases from `from` on.  0, else EINVAL (an address no live lease of this
   process's) or the ledger's error, the scopes this call opened closed and its leases no longer `o`'s.  Under
   attach_lock */
/* `o`'s leases let go: each scope ended after the uses its presents advanced are handed to it (`marks`), so a lease
   is reused only once the reads of its entries complete (mesh_lease_retire_ready).  A use that cannot be recorded
   leaves its scope open: the lease is never reused.  Under attach_lock */

/* the lease holding `ptr` (any byte of it): the Metal buffer over the whole lease (made at its first use, kept until
   ncclMemFree) and ptr's offset in it; 0, ENOENT where no lease of this process holds ptr, ENOMEM */
/* [ptr, ptr + bytes) within an attach's stream pages, the region's linear mapping of every holding's counts (mesh.h
   mesh_layout: [stream_off, client_off), page-aligned): its attach and its offset from the pages' first byte, under
   attach_lock.  A landing word is one (channel_buffer's `landed`): region memory like a lease's, the bridge's to
   write, so its Metal binding, its page table and its runs are found here as a lease's are (the chain's first WRITE,
   2026-11-01: every landing word's page operand refused as "no live local region allocation") */
static struct attach *stream_page(const void *ptr, size_t bytes, size_t *offset) {
  for (struct attach *a = attaches; a; a = a->next) {
    const struct hdr *m = a->context.M;
    const uintptr_t lo = (uintptr_t)m + m->stream_off, hi = (uintptr_t)m + m->client_off, at = (uintptr_t)ptr;
    if (at >= lo && at < hi && bytes <= hi - at) { *offset = at - lo; return a; }
  }
  return NULL;
}
static int lease_buffer(const void *ptr, void **buffer, size_t *offset) {
  pthread_mutex_lock(&attach_lock);
  struct attach *a = NULL;
  struct lease *l = lease_at(ptr, &a);
  int status;
  if (l) {
    if (!l->buffer && (l->buffer = MetalWrap(l->address, l->bytes))) atomic_store(&a->wrapped, 1);
    status = l->buffer ? 0 : ENOMEM;
    if (!status) { *buffer = l->buffer; *offset = (size_t)((uintptr_t)ptr - (uintptr_t)l->address); }
  } else if ((a = stream_page(ptr, 1, offset))) {
    const struct hdr *m = a->context.M;
    if (!a->stream_buffer && (a->stream_buffer = MetalWrap((char *)m + m->stream_off, m->client_off - m->stream_off)))
      atomic_store(&a->wrapped, 1);
    status = a->stream_buffer ? 0 : ENOMEM;
    if (!status) *buffer = a->stream_buffer;
  } else status = ENOENT;
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* nccl.h ncclMeshMemBuffer */
ncclResult_t ncclMeshMemBuffer(const void *ptr, void **buffer, size_t *offset) {
  if (!ptr || !buffer || !offset) return fail(ncclInvalidArgument, "ncclMeshMemBuffer: a NULL argument");
  const int status = lease_buffer(ptr, buffer, offset);
  if (status == ENOMEM) return fail(ncclSystemError, "ncclMeshMemBuffer of %p: no Metal buffer over its lease", ptr);
  return status ? fail(ncclInvalidArgument, "ncclMeshMemBuffer: %p is in no lease of this process (ncclMemAlloc) and no stream page", ptr) : ncclSuccess;
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
  struct redop made = {.used=1,.type=datatype,.at=immediate?NULL:scalar};
  if (immediate) memcpy(made.value, scalar, SIZE[datatype]);
  return op_make(comm, made, op);
}

/* nccl.h ncclMeshRedOpCreatePartials: an operator of the communicator's, as PreMulSum's, its row 2 + dim floats */
ncclResult_t ncclMeshRedOpCreatePartials(ncclRedOp_t *op, size_t dim, ncclDataType_t datatype, ncclComm_t comm) {
  if (!op || !comm || datatype != ncclFloat32 || dim > UINT32_MAX - 2)
    return fail(ncclInvalidArgument, "the partials merge's arguments: its operator, its communicator, float32 and a row's dim");
  return op_make(comm, (struct redop){.used=1,.type=datatype,.unit=(uint32_t)(2+dim)}, op);
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
   holds a SEND until its receive is posted (design/transport.md).  No landing entry is gated: the bridge reads none
   before its group's link (zinth), stored after everything the stream's earlier work does with those pages.  A group
   presents its entries before it publishes any (schedule); a member that fails mid-group sends nothing more, and its
   holding's end with units outstanding takes its stream out of the pairing (design/transport.md section 14). ---------- */

/* a failed communicator's kept result, its kept message named (ncclGetLastError: told, nothing formatted again).  Under
   group_lock */

/* the communicator failed for good, made again by its caller (ncclCommDestroy, then an init): its first failure's
   result and message kept, the message formatted here once (comm_failure reads them), then comm_told.  Under
   group_lock */

/* every channel's counts, compactly: its peer node, the units landed from it, and this rank's units ready and sent toward
   it ("" where the channels are not bound), so a failure names where each rank's stream stood */

/* the communicator's failure: the one it kept; a revoke; a stream it holds no longer its own (a bridge's restart, or a
   pairing lost and made again, opens every stream new: mesh-call.h mesh_stream_held); one its GPU work recorded in its
   failure words (its link lost, or a wait's bound reached after a revoke: nccl-mesh-metal.h METAL_*; a wait past its
   bound, its deadline, ends the program: bound_reached, design/eager-execution.md, assumption A4), kept now (comm_fail), so
   ncclCommGetAsyncError reads it before any group runs on it; ncclSuccess where none.  Under group_lock */

/* -- a group ------------------------------------------------------------------------------------------- */

/* where a step's bytes are: on the Metal path a buffer and a byte offset into it; on the host path an address (buffer
   NULL, offset the address); {NULL, 0} none */
/* space: a value's space (design/page-maps-2026-11-02.md), its bytes from `offset`; buffer unused where it is set */
struct ncclMeshSpace;
struct where { void *buffer; size_t offset; const struct ncclMeshSpace *space; };
/* a where's host address: NULL where it names none, or a buffer the host cannot read (a private one, never a lease's) */
/* design/page-maps-2026-11-02.md: a value's space, bound per occurrence by the run's lowering to a lease's bytes from
   first_byte (ncclMeshSpaceBind); `lease` NULL while unbound */
struct ncclMeshSpace { uint64_t serial, bytes; const unsigned char *lease; uint64_t first_byte; int placed; };

/* nccl.h ncclMeshSpaceCreate */
ncclResult_t ncclMeshSpaceCreate(uint64_t serial, uint64_t bytes, int placed, ncclMeshSpace_t *out) {
  if (!out) return fail(ncclInvalidArgument, "a space needs somewhere to go");
  struct ncclMeshSpace *space = calloc(1, sizeof *space);
  if (!space) return fail(ncclSystemError, "allocating a space");
  *space = (struct ncclMeshSpace){serial, bytes, NULL, 0, placed};
  *out = space;
  return ncclSuccess;
}
/* nccl.h ncclMeshSpaceBind: the lowering's call, at an occurrence, before the occurrence is lowered */
ncclResult_t ncclMeshSpaceBind(ncclMeshSpace_t space, const void *lease, uint64_t first_byte) {
  if (!space || !lease) return fail(ncclInvalidArgument, "bind a space to a lease's bytes");
  space->lease = lease; space->first_byte = first_byte;
  return ncclSuccess;
}
/* nccl.h ncclMeshSpaceDestroy */
ncclResult_t ncclMeshSpaceDestroy(ncclMeshSpace_t space) { free(space); return ncclSuccess; }

/* Immutable WRITE operands owned by their ordinary compiled call. Geometry,
   packet cuts and page descriptors are prepared before the group is realized.
   Issued calls borrow them; the definition and its retirement ledger own them. */

struct steps {
  _Atomic uint32_t references;
  struct steps *source; /* immutable compiler tables retained by an issued binding */
  struct mesh_step *steps;
  uint32_t count;
  /* Complete declared rank tables retained for the group's one capacity and
     dependency ordering pass. Issued bindings share this preparation data. */
  uint32_t *order_first;
  struct mesh_transfer *order_transfers;
  struct mesh_packet_view *views;
  struct mesh_transaction *transaction;
  uint32_t transactions;
  struct mesh_span_residues *packets;
  uint32_t domains;
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
struct routing;
struct call {
  int what, root, target;
  size_t count, total, tile_bytes, invocation;
  uint32_t frames;
  ncclMeshProduction *production;
  uint64_t *segments, *matrix;
  uint64_t *contributors, *recipients;
  int declared_bindings;
  ncclMeshOperand bindings[2];
  ncclMeshCoverage *coverage;
  size_t coverage_count;
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
  uint32_t polls;
  struct where send_at, recv_at, sum, scalar;
  /* Preparation-owned source generation, only for a real transform or alias. */
  struct where input_from;
  void *input_pages;
  /* A preserved WRITE publishes each copied frame/tile separately. */
  void *input_ready;
  struct where copied_ready;
  struct routing *routing; /* borrowed from the compiled group */
  uint32_t routing_scope; /* explicit preparation choice scope */
  /* each own buffer's space (ncclMeshCommOwnSpace), a deferred definition's: the run's solve's pages, bound per
     occurrence; the definition's array, which its issues' copies of the call borrow */
  struct ncclMeshSpace **own_space;
};



/* a reduction's operand element in the call's elements (the partials merge's row, else one), and a piece of `bytes` as
   the kernels take it: its type and operator (the merge's MESH_MERGE and its row's floats) and its
   elements (the merge's rows) */
/* the counts a call of kind `what` holds on n ranks where it holds any: an all-to-all's to each rank, then from each;
   a partitioned kind's a rank each; else none (struct call segments, ncclMeshCall counts) */
/* rank r's segment of a reduce-scatter's or all-gather's operand: the elements before it, and *length its own (r x
   count and count; an uneven call's, its counts') */


/* Snapshot a declared local binding in the shared page relation. Registered
   offsets, not its host pointer, determine its transport cuts. */


/* An issue owns mutable bindings only. Compiler instructions and dependency
   proofs remain shared, retaining their generation even after a plan swap. */
/* a copy's progress made again as steps_copy makes it, its arrays kept (issue_bind): no channel, entry or
   consumption.  `binding` 1 keeps its buffers and in-place reads (buffer, read, fresh: its issue's same
   binding again); 0 makes them none as well (an issue laid again: read and fresh all zero, as NULL reads) */

/* rank `rank`'s transfers of the WRITE `root` to `destination` of `operand`, cut at every `tile` elements (0: none):
   mesh_write_compile over the communicator's fabric, the pair's installed paths and its page, the one evaluation that
   every reader of a WRITE's transfers makes (write_steps, order_transfers, write_prepare_spans), into *out (malloc'd,
   its count + 1 room, the caller's to free) and *count.  0, else the compile's errno */


/* whether rank r's buffer b of call c is the communicator's (an issue's own pages: zappo, design/zorvek.md): scratch, relay generations, and an operand
   that is no result in place (a reduce-scatter's, a rank's that receives nothing) */

/* Every rank derives the complete transaction graph before selecting its own physical steps, with the fragment
   allocator explicit WRITEs use (mesh_write_compile), over `paths`:
   - a tree collective's program (each transaction a tree edge, which a cable joins): the composed table, so an edge
     keeps to its cables unless a route was installed for its pair;
   - an all-to-all's transactions, between any pair: its group's paths as the group's solution weighs them
     (plan_calls, writes_weigh), relays included. */



/* Reuse the selected form's exact tree weights and phase structure. A caller's
   explicit program already specifies its contribution graph and is lowered
   unchanged; generated forms are specialized to the declared operand roles. */

/* a reduce-scatter's own segment combined in recvbuff itself (buffer 0: result_of), the others' partials in a buffer
   of the communicator's (`partial`): no step spans two segments (mesh_program_steps_cut), so no result is copied */

/* design/measurement-regressions-2026-10-30.md: a group's calls routed at once by the planner's one solver (mesh-plan.h
   mesh_solve, b7311501's, which 3f72cbde's reset of main to an older tree removed and a tree packer and every-path
   weights of 1 then stood in for): each call's traffic its demands (demands_of), the group's demands solved together
   in finite time, and every crossing compiled from the solution, a collective along its own demands' trees and shares
   (solved_steps), an all-to-all's pairs over the write paths weighed by the flow the solution sends along each, in the
   group's own paths table (writes_weigh). A WRITE keeps the paths its fragments were compiled over as its plan was
   prepared, until a declared WRITE carries its memory and is compiled once, here. A configuration's forms and a
   caller's programs or paths are its own (comm_routes). */
#define MESH_SOLVE_EPS 0.1
struct flow { uint32_t count, *demand, *parent; double *share; };

/* call c's traffic as demands: a WRITE's and an all-to-all's each pair, its bytes up from the source to the destination;
   a reduce's up to its root and a broadcast's down from it; every other collective a demand a rank, its segment, up to
   it (a reduce-scatter's, the all-reduce's first phase) and down from it (an all-gather's, its second) */

/* collective c compiled along its own demands' trees, [first, last) of the group's, with their shares (mesh_flow_trees),
   the all-reduce in two phases, its reduce-scatter up every segment's trees and its all-gather down them */

/* the group's own paths table (*out, the caller's to free with weighed_free): each pair's write paths as the
   communicator's (comm->writes, which nothing here writes), weighed where the group's all-to-alls send between it
   (`weighs` of its demands) by the flow the solution sends along each: a solved tree's route from the pair's source up
   to its destination, its share times its demand's bytes, split over each hop's parallel cables by their bandwidths; a
   pair a configuration routed keeps its routes (ncclMeshCommWriteRoutes).  The solution is this compile's operand
   alone: a later compile weighs its own, and a WRITE plan's fragments, compiled as the plan was prepared over the
   communicator's paths, stay those its group lays */
/* a write path's hops as mesh-graph.h mesh_graph_tree_path takes them: hop k's cable (`cable`, the path's), its far end
   from `at`, and its share of the bandwidth of the cables joining its two ends */
struct path_hops { const struct mesh_fabric *fabric; const uint32_t *cable; };

/* Source identities, occurrence relations and choice scopes are preparation
   operands. Physical groups are merely competition/recording boundaries. */
struct routing {
  ncclMeshRouting view;
  struct mesh_demand *demands;
  uint32_t *lo,*hi;
  struct flow flow;
  /* flow's trees are laid demand by demand (routing_prepare): demand d's are [tree_first[d], tree_first[d + 1]) */
  uint32_t *tree_first;
  struct mesh_paths **paths[2];
  uint32_t **path_first[2];
};
struct route_prior { const ncclMeshRouting *route; uint32_t index; };
struct route_catalog { const ncclMeshRouting *at[2]; uint32_t count[2]; };
/* route_demand_equal's fields as a total order: equal demands are adjacent in it, so a plan finds every demand equal to
   one by a sort and a binary search instead of comparing every pair. Two demands with no scope order as equal here but
   are never equal (route_scope_equal), so a run is confirmed with route_demand_equal. Ties keep the demands' indices in
   order. Added by the LSTM demonstration session: the pairwise searches (the representative of each demand in a plan,
   and each demand against every earlier routing's) held the Mini for a minute in a 64-step LSTM's plan while the other
   ranks waited in its configuration exchange (sample of 2026-11-01 17:53). */



/* A preparation's stages as one JSON line each in MESH_PREPARE_RECORD's directory, a file a process (none where it is
   unset): the stage, its group, its call where it is one call's, the call's kind and sizes, the steps it made, the
   group's solved trees, routing demands and catalog trees, and its microseconds. A log of preparation, read by people
   and by extras/prepare_report.py, never by the plan or the transport. Added for the LSTM demonstration, whose 64-step
   plan sampled at 40% in write_steps and 45% in collective_steps with no sizes to say why (2026-11-01). */

/* every call that crosses a link planned, its group's traffic routed at once (above): a WRITE's steps over its pair's
   paths (write_steps), an all-to-all's over the group's weighed table (counted_steps), a collective along a configured form or the caller's
   programs where one fits it (comm_compile, program_of), else along its solved trees (solved_steps); its steps cut on
   the communicator's page (mesh-plan.h mesh_program_steps_cut), a reduce-scatter's and an all-gather's segments NCCL's,
   its operand's element a merge's row where it merges */


/* a step's entry on its channel: its call and step, its entry in its list, its pages (pages_of its bytes) and the
   value its list's count reaches once it is whole (`end`), its piece's byte offset in its buffer and its bytes, where
   its bytes lie (a SEND's source, a COPY's destination, a combine's landing area: `data`) and the first of the rows
   they lie in (`row`) */
/* a group's entries on one channel, each list in its order from its first (`first`: [0] the send list's, [1] the
   landing list's), and each list's pages a repeat (`each`: issue_lay) */
/* whether a SEND entry waits on its holding's `ready` (a publication): not a write's with producer words, which name
   their own word */
/* Original and produced values share the same page bindings. The prepared
   step names which generation it reads; a peer/root number does not. */
/* Physical arrivals retain their wire prefixes. Consumption has its own AOT
   sequence, shared by host execution and the existing GPU cursor tracks. */

/* a group's calls as mesh-plan.h's group, which mesh_group_lay lays (the one layout both ends make): call i and
   its steps at rank r, steps[i * N + r] and count[i * N + r] (none where the rank's are not known) */


/* the receive whose bytes SEND step k sends on as they are made (design/data-model.md section 9: a relay is the identity
   at a rank, or the sum where it adds partials as they pass), or -1: the last receive before it that meets its piece,
   whose piece holds its own (step k not reading the send buffer in place): a COPY into the same buffer, each unit as it
   lands, or, where `summed`, a combine there, each unit once summed (that receive's prepared progress word). The form
   63a3fe3d made and bd83c006 reverted for its count, a device atomic the bridge did not see; the count is stored
   system-coherent (nccl-mesh-kernels.h TRACK_KERNEL).  The walk to that receive is the graph module's
   (mesh-graph.h mesh_graph_forward: its row's last edge); whether it holds step k's piece is forwarded()'s */

/* each channel's entries for the group, each list from its 0 (mesh_group_lay: schedule_laid lays them from the last
   scheduled entry and page), the overlapping receives before each step kept, a receive with no bytes done */


/* Every group uses the ordinal of the common AOT transfer schedule. Sources,
   combines and relays retain its dependency order without runtime arbitration. */


/* a group's calls compiled: each crossing call's steps (plan_calls) and the group's entries on each channel, each list
   from its 0 (schedule_group), *seconds its schedule's finish (mesh_transfer_order); *plans NULL where it failed */

/* Definitions share compiled geometry. Each recorded invocation owns its page
   bindings and tables until the enclosing function releases them. Preparation
   never observes execution progress to choose a reusable instance. */
struct inflight;
struct compiled;


/* whether issued call c is the call declared as d: its kind, type, operator, root, target and counts (a call that
   holds none has equal ones, `count` each: enqueue_counts); a WRITE's, its bytes (WRITE_UNIT) and its tile, which cuts
   its transfers (write_transfers): the declaration's steps and its plan's spans are the same fragments only alike */

/* The same prepared definition is used by the access declaration and encoding.
   Nothing in this selection advances the function's issue order. */

/* The declared group at this position supplies each call's roles and operands.
   Preparation-only recording requires a complete bound declaration. */

/* issue u's entries laid from group x's compiled ones `repeats` times (repeat_of), `stride` bytes a repeat, each
   message's call the issued call at its place in `list` and its entry and end in u's own lists (`listed`, `laid`:
   schedule_laid binds them on the channels) */

/* the group's entries bound on each channel from its last scheduled entry and page (issue_lay's from each list's 0:
   `listed`, `laid`), each step's channel and entry kept, at every issue; each landing that combines into bytes an earlier SEND of its call on its
   channel reads (mesh_compile's two-node direct exchange; a SEND that reads the send buffer in place reads none,
   plan_direct) with that SEND's publication entry (struct message `need`) */


/* *items (a list of `size`-byte items, *room of them) with room for `count` + 1, its room doubled until it has:
   a list kept from call to call grows only to sizes it has not had; 0, or ENOMEM */

/* Argument and dependency pages belong to the prepared function. Their owner
   releases them with the schedule; execution supplies no reuse counters. */
static int lease_buffer(const void *ptr,void **buffer,size_t *offset);

/* group x's storage for `list` (its calls: x's own at its declaration, an issued group's at its issue), a page each: its
   calls' buffers that are the communicator's (generation_buffer, read on x's own calls: an issued call keeps its roles
   alone, never its declaration's recipients; a write's none), each laid from *base where given (st->buffer), then its
   landing area (*area, where given), a page for each combine's landing of `plans`; its bytes */
/* an issue's own buffers and landing area on their spaces, which the run's solve placed and the lowering bound to this
   occurrence's pages (design/page-maps-2026-11-02.md, integration step 4): nothing is allocated here */


/* Bind local operand views before collecting physical cuts. This is also the
   binding used to emit the DMA descriptors; no second address calculation. */

/* Compile numerical readiness once with the placed DMA spans. Regular page
   trains remain one run, regardless of their page count. */

/* Consumer order can differ from physical landing order. Each combine therefore
   publishes its own prefix; another receive can neither satisfy nor lower it.
   Later uses of this prepared issue retain increasing global packet values. */

/* where each entry's pages lie (struct message `data`): a SEND's in the buffer its step reads (the send buffer, where
   it reads that in place), a COPY's in its step's buffer, a combine's in the group's landing area (`area`: zappo,
   each entry from a page of it); each in a live lease, its common byte-packet span table in its issue's lease of them
   (`tables`, made for its binding; the one it replaces let go, kept by the bridge's reads of the entries presented
   from it: the uses owned_end hands it), the leases held by the issue from here to its next placement or its end
   (owned_begin).  0, else the call fails as a value, naming the piece (ncclInvalidUsage: the piece's
   buffer is no lease, or a span crosses a registration extent; nothing is copied to move it), nothing presented */

/* each SEND's dependency, fixed while the issue is laid and placed (its steps' priors, its lists' positions, a write's
   occurrence's packet cuts): the landing it forwards, on channel `dep_h`, its entry's base `dep_rel` past that channel's
   landing count before the group, which present_group adds.  Found once here, not at each issue's presentation */

/* a SEND's receive j: its message in the issue's lists (the frame the SEND's), the word its consumption counts, and
   its place on its channel (mesh-graph.h mesh_graph_frontier's key and rank) */



/* each channel's entries laid as the group's tables on its stream (present_group: mesh.h struct mesh_zwoggle, the
   bridge's alone to read), linked once the program reaches the group (zinth), so the bridge takes a group whole and
   has no ring to fill: a send entry's bytes ready once `ready` passes them (word 0), a landing entry gated as gate_of
   says.  The GPU present that copied each group's entries into the stream's 4,096-entry rings, and waited for their
   room and its predecessor's count, is deleted (design/zorvek.md zarpers): two calls' presents filling both ends'
   rings deadlocked the chain service (test-ov3, 2026-10-31).  0; ncclSystemError where the region cannot hold the
   tables.  Under group_lock */
/* Preparation matches physical page spans, including distinct virtual views
   of the same pages. A COPY landing that overwrites an outgoing SEND carries
   that SEND's completion operand. A REDUCE lands in separate scratch pages;
   its arithmetic store receives the overwrite relation instead. The shared
   transfer compiler has already ordered the corresponding value versions. */
/* design/zorvek.md zinth: a call's tables as the bridge reads them, one JSON line in MESH_PLAN_RECORD's directory a
   process (none where it is unset): each channel's peer, cable, words (sent, landed, ready: region offsets) and the
   call's channel positions; each entry of each list its units, word, absolute readiness base and
   its `after` list's (word, base) pairs; for a whole program's wait-for graph across its ranks (rdma/plan_cycles.py),
   read once at a call's present, never by the transport */



/* Bind packet ordinals, immutable descriptors and channel predecessors once.
   This is declaration lowering; backend recording consumes these results. */

/* where a call's operand holds byte `at`: its result (c->sum: result_of), a reduce-scatter's partial of another's
   segment in its own buffer (partial_split) */
/* where a call that is not a write or an all-to-all holds its result: recvbuff where it holds the whole operand or
   crosses nothing, a reduce-scatter's recvbuff as its own segment of the operand (partial_split), else the
   communicator's (zappo) */

/* each call's buffers bound: a write's the producer's lease itself (not a broadcast's staging copy, which would run
   before a deferred producer makes data), an all-to-all's its own two, a collective's its result (result_of), read in
   place where it may (plan_direct) */

/* issue u laid for `list` (issue_lay, `repeats` times `stride` bytes apart, each call's steps that issue's copy made
   again) and, where it crosses (`remote`), its own pages (zappo: its landing area *area), its calls' buffers
   (bind_calls) and its entries' bytes (place_data) */
/* a crossing issue's entries' pages placed (place_group), its SENDs' dependencies found (send_dependencies,
   send_requirements) and its sums' plan made (zabbadoo_plan); its landings' lists once their readers are
   (landing_lists) */



/* an all-to-all's own segment: its byte offsets in the send and the receive buffer, and its bytes, the lesser of
   what it sends itself and what it receives from itself (each in its own segment's bytes) */

/* -- the Metal path ------------------------------------------------------------------------------------ */

/* Link each prepared channel program after earlier readers of its destination
   pages. Each publication declares its own page relation; unrelated channels
   and computations have no collective-wide ordering edge. */


/* a group being encoded: its communicator, its calls, its schedule, its entries [start, end) of each list on each of
   its communicator's channels ([h][0] the send list's, [h][1] the landing list's), whether it is a fused crossing, and
   its repeats.  Every group completes at its own call, so none is in flight past it: the list of groups in flight,
   their retirement and the tickets that ordered their failures are deleted with the tracks pool */

/* Each recorded invocation has its declared pages. Preparation never polls
   execution progress to select a binding or rewrite a still-live program. */

/* an issue of definition x for `repeats` repeats (struct issue): each channel's lists' room, x's calls' steps copies and
   a record in flight; NULL where memory runs out */
struct zplan;
/* group_lock's arrays, kept from call to call (plan_direct's ranges, complete()'s tracks and tiles): `bytes` of one,
   grown (its new bytes zeroed) where it has fewer, so none is allocated once each has its size; NULL where memory runs
   out */
enum { KEPT_SPANS, KEPTS };

/* Every landing is a declared data producer. Its frame and the transfer
   compiler's consumption order identify its value version. COPY and REDUCE
   enter the same command graph; a receive need not share another's packet cuts. */
/* Adjacent folds can share one store when no intervening transfer observes
   an intermediate value. This is a preparation rewrite of the same leaf, not
   another executor; unequal packet partitions use their common cell refinement. */

/* The common command compiler receives exact accesses for each producer. No
   whole-channel join, pool selector, or secondary dependency interpreter. */




/* Two operands' relation as values (design/page-maps-2026-11-02.md): `same` where b's `made` bytes are a's from
   `before`, `overlap` where their bytes meet. Distinct spaces never meet: the run's solve keeps distinct live items
   apart, and leases are distinct storages; in one space the byte ranges decide. No page is resolved. */
/* Materialize a specifically selected original value generation: a transform,
   shifted alias, or the source chosen by the compiled overwrite-cycle solve. */



/* a group's results, once its entries are consumed: each already in recvbuff (result_of), an integer average's
   divided there */

/* a group's failure once its entries are presented: its communicator failed with it (comm_fail: its lists stand
   mid-group) */


/* the operands a group makes in place, after its entries are presented and before anything is
   published: a premultiply, or a copy within the region, where a call reads no send buffer in place; a scratch's copy;
   an all-to-all's own segment.  Encoded into `program`, or, with none, only counted into *copies: a group that makes
   some publishes once they are made (source_group) */
/* The source generation is copied in its declared logical tile order. Each
   tile observes its actual producer and publishes only its own copied bytes. */
/* Both layouts use the same typed source observations. A copied generation
   has its own publication word; producing its input cannot publish its output. */
/* An empty contributor set is the reduction's declared identity, emitted by
   the same output-page command graph. No source or physical transfer is needed. */
#define COPIED(step, what) (program ? gpu(step, what) : ((*copies)++, ncclSuccess))
#undef COPIED

/* Lower one declaration occurrence. Its issue owns all selected pages and
   transfer facts before the numerical/backend binding phase starts. The
   communicator cursor advances here; recording reads only the saved issue. */

/* Native numerical declarations join the enclosing compiler's source graph.
   No encoder or pipeline is bound while these operands are constructed. */
/* Recording consumes the exact specialized source program. It neither lowers
   operations nor solves their page relations again. */




/* Physical span projections exist only inside preparation. Programs declare
   semantic byte intervals; neither a physical ordinal nor a source-array
   insertion position crosses the public interface. */
/* Source input spans join the same declaration-owned SENDs. No numerical
   command is invented to carry a dependency whose only consumer is the DMA. */



/* Fixed-size configuration metadata, before any data program executes. Retained
   bootstrap sockets never carry acknowledgements or payload-dependent routing. */
/* The caller holds prepare across the fixed report and its selection/backing
   reports. Concurrent preparations cannot interleave those exchanges. */
/* one configuration phase under prepare, within MESH_PREPARATION_NS: every rank's `bytes` gathered into `data` from
   `local` (`root` -1), or the root's `bytes` at `data` on every rank */







/* A declared call owns its counts, page roles and bindings before lowering.
   Its operator and dependencies enter the common retained source graph. */


/* A page can hold successive values. A produced SEND reads its preceding
   producer, not the older generation that producer overwrote. Only later
   writes constrain that read. Original inputs have no such local producer. */
/* an occurrence's order is its declaration's (compile_group's schedules, copied by issue_lay): its entries numbered
   and its units and consumption laid. No alias is exchanged and no order predicted again: versions keep a SEND's
   source apart from the landings that would overwrite it (design/page-maps-2026-11-02.md) */
/* Every endpoint and relay contributes physical backing cuts in the declared
   operand or transaction coordinates. Union every laid issue's placement (each binding instance) before
   emitting either end's descriptors. These exchanges belong solely to declaration. */

/* design/zorvek.md zoblimp: an issue of definition x for binding instance `instance`, laid over pages of its own: its
   calls, their operands' views for that instance, its entries and its pages (issue_layout); kept among x's laid
   issues, so x's drop frees it */

/* Lay one issue for each binding instance before recording (zoblimp), their packets cut together and exchanged once
   (packets_prepare), then placed. WRITE and collective calls share the same issue storage and packet preparation. */
/* a definition's validation at declaration; it is laid at each occurrence, once the run's solve placed its values and
   the lowering bound its spaces (zorvek): nothing is laid here (design/page-maps-2026-11-02.md, integration step 1) */

/* Compile and bind every issue before command recording. */
/* Resolve declared source references against the complete logical graph once.
   A rank outside a transaction's route still has this same schema; absence of
   a local physical message must never turn a bad reference into an unmasked
   transfer or reject a valid remote value. */

/* Compile group definitions, then bind the function's repeated-block order.
   Replacing a declaration takes effect at the next group; failure keeps the
   prior declaration. This work precedes numerical submission. */


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




/* A tensor adapter resolves its local address once. Everything below that
   boundary names this table and a page-local start, plus a compact selection.
   A row's page/mask at byte k is divmod(start + row*pitch + k, page_bytes);
   no array of those identical affine expressions needs to be materialized. */

/* Fixed-size configuration reports. Storage locations are arena phases; no
   process pointer crosses a socket. Metadata on the retained bootstrap is
   serialized separately from GPU execution and remains revocable/deadlined. */

/* Pure metadata validation. Source aliasing is repeated reading; ambiguous
   landing aliases would give concurrent logical writers the same bytes. */

/* Already-declared masks/spans can also describe affine rows. Recognize that
   algebra once; peers need only the fixed geometry report for such a view.
   Irregular, reversed and aliased sources retain their declared index list. */

/* The next contiguous interval of selected logical bytes, up to limit. Both
   packet placement and physical backing composition consume this traversal. */

/* Common cuts for compact local selections. Each peer can use rows, indexed
   masks or a contiguous selection for the same logical bytes. Visit actual
   discontinuities and registration boundaries without expanding affine rows
   or pages. Contiguous selections across page/row/span edges stay contiguous.
   Publication tiles restart the packet grid. All operands were checked once
   before this occurrence's cuts are compiled. */

/* Compose selected logical bytes with each peer's actual page backing. One
   compact mapping exchange serves every occurrence. Registration phases alone
   suffice for a single run; several runs also contribute their discontinuities.
   The result is the same common packet table, independent of physical layout. */

/* Physical bindings, paths and common cuts are all configuration operands.
   Materialize their local descriptor trains once, including every repetition.
   Execution indexes these immutable tables; the usual retirement ledger keeps
   them alive until the bridge has finished reading them after plan destruction. */

/* ncclMeshPageTableCompose, which mapped selected pages of several leases into one remapped view (mach_vm_remap), is
   deleted with its only callers (MetalTensorPages compose:, page_storage; effect fields' composed view) by the joint
   change of design/page-maps-2026-11-02.md (2026-11-02): a chooser composes page maps, never views. The composed-lease
   bookkeeping it fed (`composed`, `composed_total`, `composed_stale`, a lease's `owners`) holds nothing now and goes
   with nccl-mesh.c's own move onto values. */

/* Lower both endpoint declarations through the same byte-selection algebra.
   Slot and frame relations are independent operands; neither is inferred from
   an allocation's shape or from a backend name. */




/* a communicator over comm's ranks `ranks` (count of them, in that order), named after comm and its `split`-th split or
   shrink and `color`, the same on every member, attached (comm_attach), its forms comm's; its streams held and its
   channels made by its caller (ncclCommSplit) */

/* ncclCommSplit: every rank's (color, key) and its report (reach_read; none for NCCL_SPLIT_NOCOLOR) exchanged on comm's
   configuration channel (prepare_exchange), the ranks of this color ordered by key, then rank (two of a color on one
   node refused alike: comm_apart), each rank's streams toward its color's nodes held (fabric_acquire: the streams its parent holds are not free, EBUSY where none is), and a
   second exchange on it carrying each rank's holding's status, then the child's channels (fabric_realize): NCCL's bootstrapSplit (the child's exchange over its parent's), with the init's
   two gathers.  A failure at any rank of a color fails that color's split on each of its ranks as a value, what it held
   let go (comm_free).  Neither exchange is a collective: no group of comm's carries them, inside a group or not
   (design/eager-execution.md: an undeclared collective ends the program) */


/* ncclCommShrink: comm's ranks but the excluded ones, in rank order, with no exchange among them (a departed member
   answers none), so holding no stream: a shrink to two ranks or more is made with no channel, its map none, so its
   first call that crosses a link fails as a value (comm_routes: no map of its members; plan_calls: no link toward the
   peer); a rank among those it excludes is in no shrunk communicator (*newcomm NULL, as NCCL_SPLIT_NOCOLOR's); with
   NCCL_SHRINK_ABORT, comm revoked first (ncclCommRevoke) */

/* ncclCommGetUniqueId: the id a grow of comm meets at, a listener of comm's root (ncclGetUniqueId) */
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
  return comm ? ncclGetUniqueId(uniqueId) : fail(ncclInvalidArgument, "comm is NULL");
}

/* ncclCommGrow: NCCL's bootstrap at the id of comm's root (ncclCommGetUniqueId), which comm's other ranks take from
   the root's row of an exchange on comm's configuration channel (prepare_exchange: no collective); comm's ranks keep their ranks (the `rank` they
   pass, NCCL's -1, not read) and a new one (comm NULL) takes `rank`, every rank reporting its node and the nodes its
   bridge reaches; the communicator, named by the id, holds its streams as every init does */


/* ==== The collective's lowering (design/zorvek.md) ====
   A group of calls is declared once (ncclMeshCommDeclare): each call routed along the communicator's trees by
   libmesh's planner (mesh-plan.h mesh_trees_pack, mesh_compile, mesh_program_steps_cut). Its steps are the run's
   transport accesses (ncclMeshCommSteps): the solve places their bytes and orders their events. An occurrence is
   lowered once its bytes are placed (ncclMeshCommLowerOccurrence): zoblimp lays a channel's table (mesh.h struct
   mesh_zwoggle) of its send and landing entries, a unit a page from each piece's start, at absolute positions of the
   channel's prefixes; the run's nodes are zinth's link store a channel, a release store of `ready` a SEND entry whose
   bytes are this rank's, and a landing a receive. A COPY's consumption needs no kernel: its landing is the copy. A
   SEND of bytes an earlier COPY landed is gated on that landing (zirrupel, the bridge's forward), with no release. */
#define MESH_TREE_EPS 0.1 /* Garg and Konemann's eps for the trees (mesh-plan.h mesh_trees_pack): a choice to measure */
#define MESH_TREES_MOST 4 /* each root's heaviest trees kept */
#define MESH_HOP_NS 8200.0 /* a hop's price: a release to its landing one-way, 8.2 us (metal-microbench docs/measurement.md u4) */
/* The communicator's trees: both of the map's packings, the unrestricted one and each root's shortest-path one
   (mesh-graph.h mesh_graph_pack `shallow`), each root's shares halved (the two flows' mean, a flow the links carry),
   every tree priced at its depth's hops (mesh_graph_tree_depth x MESH_HOP_NS) and the trees' joint rate from the
   cables' bandwidths (mesh_graph_trees_rate). A call's segment is then cut over its root's trees by earliest finish
   (mesh-plan.h struct mesh_trees): a unit takes the tree it lands soonest by, many spread over every route.
   Changed from f6d9ad29, the shortest-path trees alone, which left routes unused (m5ult2 -> Mini at 0.67 of the
   busiest direction while m5ult1 -> Mini and m5ult3 -> Mini were full) and so a long transfer slower: what must stay
   is a small segment's unit on a shortest tree (run-23: 575 -> 231-255 us a tick, a sub-page segment on a 3-4 deep
   tree before), which the earliest-finish cut keeps. */
static ncclResult_t comm_trees(ncclComm_t comm) {
  if (comm->trees.count) return ncclSuccess;
  const uint32_t nodes = comm->map.nodes, most = nodes * MESH_TREES_MOST, all = 2 * most;
  uint32_t *root = calloc(all + 1, sizeof *root), *parent = calloc((size_t)all * nodes + 1, sizeof *parent), count = 0;
  double *weight = calloc(all + 1, sizeof *weight), *latency = calloc(all + 1, sizeof *latency);
  double *share = calloc(all + 1, sizeof *share), *capacity = calloc((size_t)nodes * nodes + 1, sizeof *capacity);
  if (root && parent && weight && latency && share && capacity)
    for (int shallow = 0; shallow < 2; shallow++) {
      const uint32_t made = mesh_trees_pack(&comm->map, NULL, MESH_TREE_EPS, MESH_TREES_MOST, shallow, root + count,
                                            parent + (size_t)count * nodes, weight + count);
      if (!made) { count = 0; break; }
      count += made;
    }
  if (!count) {
    free(root); free(parent); free(weight); free(latency); free(share); free(capacity);
    return fail(ncclInvalidUsage, "the communicator's trees (mesh-plan.h mesh_trees_pack: its map joins its ranks)");
  }
  for (uint32_t t = 0; t < count; t++) {
    weight[t] += mesh_graph_log(0.5);
    share[t] = mesh_graph_exp(weight[t]);
    latency[t] = mesh_graph_tree_depth(nodes, root[t], parent + (size_t)t * nodes) * MESH_HOP_NS;
  }
  for (uint32_t l = 0; l < comm->map.links; l++) {
    const uint32_t a = comm->map.link[l][0], b = comm->map.link[l][1];
    if (a >= nodes || b >= nodes || a == b) continue;
    const double bytes_a_ns = (double)comm->map.bandwidth[l] / 8e9;
    capacity[(size_t)a * nodes + b] += bytes_a_ns; capacity[(size_t)b * nodes + a] += bytes_a_ns;
  }
  const double rate = mesh_graph_trees_rate(nodes, count, root, parent, share, capacity);
  free(share); free(capacity);
  comm->trees = (struct mesh_trees){nodes, count, root, parent, weight, latency, rate};
  return ncclSuccess;
}

static struct definition *definition_find(ncclComm_t comm, uint64_t id) {
  for (struct definition *d = comm->declared; d; d = d->next) if (d->id == id) return d;
  return NULL;
}

static int channel_of(ncclComm_t comm, uint32_t peer) {
  for (uint32_t h = 0; h < comm->count; h++) if ((uint32_t)comm->channels[h].peer == peer) return (int)h;
  return -1;
}

/* whether the receiving end of a transfer sends it on: a SEND of the receiver's, in a later round, over any of its
   elements (the receiver's steps from the program, alike at both ends) */
static int forwarded(const struct mesh_step *steps, uint32_t count, uint32_t round, uint64_t first, uint64_t end) {
  for (uint32_t k = 0; k < count; k++)
    if (steps[k].op == MESH_STEP_SEND && steps[k].round > round && steps[k].first < end &&
        first < steps[k].first + steps[k].piece.elements) return 1;
  return 0;
}

/* a call's `made` steps at d->steps spread over their pairs' parallel cables: a SEND or COPY whose receiver does not send
   it on goes over every channel toward its peer, its piece cut in page quanta (the first ones one more for the
   remainder), in the channels' order, which is the cables' (fabric_realize: the one order both ends make), so its two
   ends cut alike; a forwarded transfer stays on the first channel, so a forward's SEND, cut or not, lies within one COPY
   (the gate the lowering takes). Every step's channel in d->step_channel. Measured need: a pair's parallel cables sat
   idle under one channel a peer (metal-microbench docs/facts.md, the bandwidth run: one port a peer carried bytes). */
static ncclResult_t spread(ncclComm_t comm, struct definition *d, const struct mesh_program *program,
                           struct mesh_operand operand, const uint64_t *segments, uint32_t made, uint32_t call) {
  const uint64_t q = mesh_page_quantum(comm->page, operand.element_bytes);
  const uint32_t room = made * (comm->count ? comm->count : 1);
  struct mesh_step *own = malloc((made + 1) * sizeof *own), *peer = malloc(((size_t)room + 1) * sizeof *peer);
  struct mesh_step *step = realloc(d->step, (d->steps + room + 1) * sizeof *step);
  if (step) d->step = step;
  uint32_t *owner = realloc(d->step_call, (d->steps + room + 1) * sizeof *owner);
  if (owner) d->step_call = owner;
  uint32_t *channel = realloc(d->step_channel, (d->steps + room + 1) * sizeof *channel);
  if (channel) d->step_channel = channel;
  if (!own || !peer || !step || !owner || !channel) { free(own); free(peer); return fail(ncclSystemError, "a spread's memory"); }
  memcpy(own, d->step + d->steps, made * sizeof *own);
  size_t at = d->steps;
  for (uint32_t s = 0; s < made; s++) {
    const struct mesh_step x = own[s];
    int lanes[comm->count + 1], n = 0;
    for (uint32_t h = 0; h < comm->count; h++) if ((uint32_t)comm->channels[h].peer == x.peer) lanes[n++] = (int)h;
    int cut = n > 1 && (x.op == MESH_STEP_SEND || x.op == MESH_STEP_COPY) && x.peer != (uint32_t)comm->rank;
    if (cut) {
      /* the receiver's steps: this rank's own for a COPY, the peer's for a SEND */
      const struct mesh_step *theirs = own;
      uint32_t count = made;
      if (x.op == MESH_STEP_SEND) { count = mesh_program_steps_cut(program, x.peer, operand, segments, comm->page, peer); theirs = peer; }
      cut = !forwarded(theirs, count, x.round, x.first, x.first + x.piece.elements);
    }
    if (!cut) { d->step[at] = x; d->step_call[at] = call; d->step_channel[at] = x.peer == (uint32_t)comm->rank ? 0 : (uint32_t)channel_of(comm, x.peer); at++; continue; }
    const uint64_t quanta = (x.piece.elements + q - 1) / q;
    uint64_t from = x.first;
    for (int j = 0; j < n; j++) {
      const uint64_t share = quanta / (uint64_t)n + ((uint64_t)j < quanta % (uint64_t)n);
      uint64_t elements = share * q;
      if (from + elements > x.first + x.piece.elements) elements = x.first + x.piece.elements - from;
      if (!elements) continue;
      d->step[at] = x; d->step[at].first = from; d->step[at].piece.elements = elements;
      d->step_call[at] = call; d->step_channel[at] = (uint32_t)lanes[j]; at++;
      from += elements;
    }
  }
  d->steps = at;
  free(own); free(peer);
  return ncclSuccess;
}

ncclResult_t ncclMeshCommDeclare(ncclComm_t comm, const ncclMeshCollective *calls, size_t count, uint64_t *definition) {
  if (!comm || !definition || (count && !calls)) return fail(ncclInvalidArgument, "a declaration needs its comm, calls and result");
  ncclResult_t result = comm_trees(comm);
  if (result) return result;
  struct definition *d = calloc(1, sizeof *d);
  if (!d || !(d->call = malloc((count + 1) * sizeof *d->call))) { free(d); return fail(ncclSystemError, "a declaration's memory"); }
  memcpy(d->call, calls, count * sizeof *calls);
  for (size_t k = 0; k < count; k++) d->call[k].segments = NULL; /* read at declaration alone */
  d->calls = count;
  for (size_t k = 0; !result && k < count; k++) {
    const ncclMeshCollective *call = calls + k;
    if (call->type >= ncclNumTypes) { result = fail(ncclInvalidArgument, "call %zu: no such element type", k); break; }
    uint32_t parts = 0, moves = 0;
    mesh_compile_room(&comm->trees, &parts, &moves);
    uint32_t *segment = calloc(parts + 1, sizeof *segment), *first = calloc((size_t)comm->map.nodes + 1, sizeof *first);
    double *weight = calloc(parts + 1, sizeof *weight), *latency = calloc(parts + 1, sizeof *latency);
    struct mesh_move *move = calloc(moves + 1, sizeof *move);
    struct mesh_program program;
    if (!segment || !first || !weight || !latency || !move) result = fail(ncclSystemError, "a call's program memory");
    else if (mesh_compile(&comm->trees, call->what, call->root, 0, NULL, &program, segment, weight, first, move, latency))
      result = fail(ncclInvalidArgument, "call %zu: no program for its collective (mesh-plan.h mesh_compile)", k);
    else {
      const size_t room = (size_t)(first[comm->rank + 1] - first[comm->rank]) * program.parts;
      struct mesh_step *step = realloc(d->step, (d->steps + room + 1) * sizeof *step);
      uint32_t *owner = realloc(d->step_call, (d->steps + room + 1) * sizeof *owner);
      if (step) d->step = step;
      if (owner) d->step_call = owner;
      if (!step || !owner) result = fail(ncclSystemError, "a call's steps' memory");
      else {
        const struct mesh_operand operand = {call->type, (uint32_t)SIZE[call->type], call->elements};
        const uint32_t made = mesh_program_steps_cut(&program, (uint32_t)comm->rank, operand, call->segments,
                                                     comm->page, d->step + d->steps);
        if (!made && call->elements && call->segments)
          result = fail(ncclInvalidArgument, "call %zu: its segments do not sum to its elements", k);
        for (uint32_t s = 0; s < made; s++) {
          d->step_call[d->steps + s] = (uint32_t)k;
          if (d->step[d->steps + s].peer != (uint32_t)comm->rank && channel_of(comm, d->step[d->steps + s].peer) < 0)
            result = fail(ncclInvalidUsage, "call %zu: a step toward rank %u, which no channel reaches", k,
                          d->step[d->steps + s].peer);
        }
        if (!result) result = spread(comm, d, &program, operand, call->segments, made, (uint32_t)k);
        else d->steps += made;
      }
    }
    free(segment); free(first); free(weight); free(latency); free(move);
  }
  if (result) { free(d->call); free(d->step); free(d->step_call); free(d->step_channel); free(d); return result; }
  d->id = ++comm->definitions;
  d->next = comm->declared; comm->declared = d;
  *definition = d->id;
  return ncclSuccess;
}

ncclResult_t ncclMeshCommSteps(ncclComm_t comm, uint64_t definition, ncclMeshTransfer *out, size_t capacity,
                               size_t *count) {
  const struct definition *d = comm ? definition_find(comm, definition) : NULL;
  if (!d || !count) return fail(ncclInvalidArgument, "no such declaration");
  *count = d->steps;
  for (size_t s = 0; s < d->steps && s < capacity; s++) {
    const struct mesh_step *step = d->step + s;
    const uint64_t width = step->piece.element_bytes;
    out[s] = (ncclMeshTransfer){d->step_call[s], d->step_channel[s], step->op, step->round,
                                step->buffer, step->first * width, mesh_step_bytes(step)};
  }
  return ncclSuccess;
}

/* a placed byte's region offset: its lease's page table (lease_page_offset); UINT64_MAX where it lies in no lease */
static uint64_t region_offset(const void *address) {
  struct attach *a = NULL;
  const struct lease *l = lease_at(address, &a);
  return l ? lease_page_offset(a, l, (size_t)((const char *)address - (const char *)l->address)) : UINT64_MAX;
}

ncclResult_t ncclMeshCommLowerOccurrence(ncclComm_t comm, uint64_t definition, void *const *address,
                                         const uint32_t (*after)[2], size_t afters, const uint32_t (*forward)[2],
                                         size_t forwards, ncclMeshNode *node, size_t capacity, size_t *count) {
  const struct definition *d = comm ? definition_find(comm, definition) : NULL;
  if (!d || !count || (d->steps && !address)) return fail(ncclInvalidArgument, "no such declaration, or no placed bytes");
  const struct hdr *m = comm->attach->context.M;
  const uint64_t page = comm->page;
  /* each step's units, its entry's base on its channel's prefix (sends: ready; landings: landed), its list's place */
  uint64_t *base = calloc(d->steps + 1, sizeof *base), *units = calloc(d->steps + 1, sizeof *units);
  size_t *unit_first = calloc(d->steps + 2, sizeof *unit_first);
  int64_t *gate = calloc(d->steps + 1, sizeof *gate); /* the COPY a forwarding SEND follows, else -1 */
  if (!base || !units || !gate || !unit_first) {
    free(base); free(units); free(gate); free(unit_first);
    return fail(ncclSystemError, "a lowering's memory");
  }
  ncclResult_t result = ncclSuccess;
  for (size_t s = 0; s < d->steps; s++) {
    units[s] = (mesh_step_bytes(d->step + s) + page - 1) / page;
    unit_first[s + 1] = unit_first[s] + units[s];
  }
  uint64_t *unit_at = calloc(unit_first[d->steps] + 1, sizeof *unit_at);
  if (!unit_at) result = fail(ncclSystemError, "a lowering's unit offsets' memory");
  for (size_t s = 0; !result && s < d->steps; s++) {
    const struct mesh_step *step = d->step + s;
    const uint64_t bytes = mesh_step_bytes(step);
    gate[s] = -1;
    /* a unit is a page of the piece from its start, the same cut at both ends (TN3205: a SEND and its receive cover
       the same frames); a span may start inside a page (design/transport.md section 1), and each unit's bytes lie
       contiguous in the region, one SGE */
    for (uint64_t u = 0; !result && u < units[s]; u++) {
      const char *first = (const char *)address[s] + u * page;
      const uint64_t length = bytes - u * page < page ? bytes - u * page : page;
      const uint64_t a = region_offset(first), b = region_offset(first + length - 1);
      unit_at[unit_first[s] + u] = a;
      if (a == UINT64_MAX || b == UINT64_MAX)
        result = fail(ncclInvalidUsage, "step %zu's bytes lie in no region lease of this process", s);
      else if (b != a + length - 1 || a / MESH_DMA_EXTENT != b / MESH_DMA_EXTENT)
        result = fail(ncclInvalidUsage, "step %zu's unit %llu crosses a region page or extent its next bytes do not "
                      "continue: its piece starts %llu bytes into a page (place it on a page)", s,
                      (unsigned long long)u, (unsigned long long)(region_offset(address[s]) % page));
    }
  }
  for (size_t f = 0; !result && f < forwards; f++) {
    const size_t s = forward[f][0], r = forward[f][1];
    if (s >= d->steps || r >= d->steps || d->step[s].op != MESH_STEP_SEND || d->step[r].op != MESH_STEP_COPY ||
        d->step[s].first < d->step[r].first || (d->step[s].first - d->step[r].first) * d->step[s].piece.element_bytes % page)
      result = fail(ncclInvalidArgument, "forward %zu does not name a SEND within a COPY's pages", f);
    else gate[s] = (int64_t)r;
  }
  /* every list's order (mesh_lay: by round, then step order, the one order both ends make) and every entry's base on
     its channel's prefix, before any entry is written: a forward and a landing's after name other channels' bases */
  struct mesh_place *placed = calloc(2 * (size_t)comm->count * (d->steps + 1), sizeof *placed);
  uint32_t *listed = calloc(2 * (size_t)comm->count + 1, sizeof *listed);
  struct mesh_step *lay_step = malloc((d->steps + 1) * sizeof *lay_step);
  uint32_t *lay_index = malloc((d->steps + 1) * sizeof *lay_index);
  if (!placed || !listed || !lay_step || !lay_index) result = fail(ncclSystemError, "a lowering's lists' memory");
  for (uint32_t h = 0; !result && h < comm->count; h++) for (int land = 0; land < 2; land++) {
    struct channel *ch = comm->channels + h;
    struct mesh_place *list = placed + (2 * (size_t)h + land) * (d->steps + 1);
    /* a channel's own steps (spread: its pair's transfers cut over the pair's cables, each step one channel's), laid
       by mesh_lay over them alone and mapped back to the definition's indices */
    uint32_t mine = 0;
    for (size_t s = 0; s < d->steps; s++) if (d->step_channel[s] == h && d->step[s].peer == (uint32_t)ch->peer) { lay_step[mine] = d->step[s]; lay_index[mine++] = (uint32_t)s; }
    if (!mine) continue;
    mesh_lay(lay_step, mine, (uint32_t)ch->peer, !land, 0, list, listed + 2 * h + land);
    for (uint32_t i = 0; i < listed[2 * h + land]; i++) list[i].step = lay_index[list[i].step];
    uint64_t position = land ? ch->lands : ch->sends;
    for (uint32_t i = 0; i < listed[2 * h + land]; i++) { base[list[i].step] = position; position += units[list[i].step]; }
  }
  /* the occurrence's tables, one lease: a channel's table, its two lists, their spans, the landings' after lists and
     one complete word every static descriptor names */
  size_t bytes = sizeof(uint64_t);
  for (uint32_t h = 0; h < comm->count; h++) bytes += sizeof(struct mesh_zwoggle);
  for (size_t s = 0; s < d->steps; s++)
    bytes += sizeof(struct mesh_zwoggle_entry) + units[s] * sizeof(struct mesh_dma_span) + sizeof(uint64_t);
  bytes += afters * sizeof(struct mesh_zwoggle_after);
  void *tables = NULL;
  if (!result) result = mem_alloc(&tables, bytes, MESH_BACKING_LINEAR);
  size_t written = 0;
  #define NODE(...) do { if (written < capacity) node[written] = (ncclMeshNode){__VA_ARGS__}; written++; } while (0)
  if (!result) {
    memset(tables, 0, bytes);
    char *at = tables;
    uint64_t *complete = (uint64_t *)at; at += sizeof *complete;
    *complete = UINT64_MAX - 1;
    const uint64_t complete_at = region_offset(complete);
    for (uint32_t h = 0; !result && h < comm->count; h++) {
      struct channel *ch = comm->channels + h;
      struct mesh_zwoggle *table = (struct mesh_zwoggle *)at; at += sizeof *table;
      for (int land = 0; land < 2; land++) {
        uint64_t own_end = 0;
        const struct mesh_place *list = placed + (2 * (size_t)h + land) * (d->steps + 1);
        const uint32_t n_listed = listed[2 * h + land];
        struct mesh_zwoggle_entry *entry = (struct mesh_zwoggle_entry *)at; at += n_listed * sizeof *entry;
        table->entries[land] = region_offset(entry);
        table->count[land] = n_listed;
        uint64_t *position = land ? &ch->lands : &ch->sends;
        for (uint32_t i = 0; i < n_listed; i++) {
          const size_t s = list[i].step;
          const struct mesh_step *step = d->step + s;
          struct mesh_dma_span *span = (struct mesh_dma_span *)at; at += units[s] * sizeof *span;
          const uint64_t piece = mesh_step_bytes(step);
          for (uint64_t u = 0; u < units[s]; u++) {
            const uint64_t first = u * page, length = piece - first < page ? piece - first : page;
            span[u] = (struct mesh_dma_span){unit_at[unit_first[s] + u], length, *position + u + MESH_STREAM_LANES};
          }
          entry[i].entry = (struct mesh_entry){.spans = region_offset(span), .units = units[s], .step = 1,
                                               .spans_word = complete_at, .spans_base = 0, .base = *position};
          if (!land && gate[s] >= 0) { /* a forward: each unit once the COPY's unit under it has landed */
            const int g = (int)d->step_channel[gate[s]];
            entry[i].entry.word = (uint64_t)((const char *)comm->channels[g].view.landed - (const char *)m);
            entry[i].entry.base = base[gate[s]] + (step->first - d->step[gate[s]].first) * step->piece.element_bytes / page;
          }
          if (land) { /* each SEND that must complete before this landing is posted (the solve's edges) */
            uint64_t *list = (uint64_t *)at, n = 0;
            at += sizeof *list;
            for (size_t a = 0; a < afters; a++) if (after[a][1] == s) {
              const size_t k = after[a][0];
              const int g = (int)d->step_channel[k];
              ((struct mesh_zwoggle_after *)(list + 1))[n++] = (struct mesh_zwoggle_after){
                (uint64_t)((const char *)comm->channels[g].view.sent - (const char *)m), base[k] + units[k]};
            }
            at += n * sizeof(struct mesh_zwoggle_after);
            *list = n;
            entry[i].after = n ? region_offset(list) : 0;
          }
          *position += units[s];
          if (!land && gate[s] < 0) own_end = *position;
        }
        /* the channel's nodes, one each (a consumer reads the occurrence's landings whole; `ready` is one prefix):
           its landing at its landings' end, its SENDs' completion at their end, its release at its own units' end */
        if (land && n_listed) NODE(ncclMeshNodeLanding, h, h, (void *)ch->view.landed, *position);
        if (!land && n_listed) NODE(ncclMeshNodeSent, h, h, (void *)ch->view.sent, *position);
        if (!land && own_end) NODE(ncclMeshNodeRelease, h, h, (void *)ch->view.ready, own_end);
      }
      if (!table->count[0] && !table->count[1]) continue;
      /* zinth: this table stored where the channel's chain left off, the stream's root or the last table's next */
      NODE(ncclMeshNodeLink, UINT32_MAX, h, ch->link ? ch->link : (void *)ch->view.zwoggle, region_offset(table));
      ch->link = &table->next;
    }
  }
  #undef NODE
  free(base); free(units); free(gate); free(placed); free(listed); free(unit_at); free(unit_first); free(lay_step); free(lay_index);
  *count = written;
  return result;
}

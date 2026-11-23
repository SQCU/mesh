/* libnccl-mesh: NCCL's calls over the node's page table (nccl.h is the contract, rdma/NCCL.md how to build and use it;
   design/PAGE-TABLE-SPEC-20261122.md section 6).  Every call is blocking and moves host memory: what a rank sends it
   copies into source pages of its own and writes to the indices its peers took for it (mesh-call.h mesh_write), and
   what lands at its own indices it copies out, or combines rank by rank, into the call's recvbuff, as it landed, then
   invalidates. */
#include "nccl.h"
#include "mesh.h"
#include "mesh-call.h"
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
#include <sys/sysctl.h>

enum { CALL_ALLREDUCE, CALL_BROADCAST, CALL_REDUCE, CALL_REDUCE_SCATTER, CALL_ALLGATHER, CALL_ALLTOALL, CALL_GATHER,
       CALL_SCATTER };
static const char *const CALL_NAME[] = {"ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclReduceScatter",
                                        "ncclAllGather", "ncclAlltoAll", "ncclGather", "ncclScatter"};
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN };
/* A communicator's pages in CHANNELS channels, every rank cutting them alike: page j of a message, and a reduction's
   position j, are channel j % CHANNELS's.  Each channel has indices of its own on each rank's node: a receive ring of
   the communicator's `depth` indices for each source rank (at most RECEIVES: comm_depth), the channel's k-th page from
   rank s to rank d landing at d's base of the channel + s depth + k mod depth, that index's (k / depth + 1)-th
   landing; and SOURCES source pages of each class, which what the channel sends is copied into.  A rank moves its
   channels on the threads of its process's pool (T of them, at most CHANNELS), channel c on thread c % T with the
   indices of slot c % T: the layout is the same at every rank, whatever its threads */
#define CHANNELS 8u
#define RECEIVES 512u
#define SOURCES 128u

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
/* a failure's message: its text, and `id`, where the message is a communicator's (comm_fail), the count of messages the
   process had recorded when it was (`messages`), else 0 */
struct said { uint64_t id; char text[512]; };
static uint64_t messages;
/* this thread's last failure's message (ncclGetLastError): formatted by fail (id 0), or a copy of a communicator's
   (told) */
static _Thread_local struct said last;

static ncclResult_t fail(ncclResult_t result, const char *format, ...) __attribute__((format(printf, 2, 3)));
static ncclResult_t fail(ncclResult_t result, const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(last.text, sizeof last.text, format, arguments);
  va_end(arguments);
  last.id = 0;
  return result;
}

/* `result` beside the communicator's message `said`: `last` its copy, copied only where `last` has another (its id
   the match), so a failure read again formats and copies nothing */
static ncclResult_t told(ncclResult_t result, const struct said *said) {
  if (last.id != said->id) last = *said;
  return result;
}

/* a communicator's message's identity, a new one */
static uint64_t message_id(void) { return __atomic_fetch_add(&messages, 1, __ATOMIC_RELAXED) + 1; }

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

#define FLOAT_LOOP(S) do { \
    S *d = (S *)dst; const S *x = (const S *)src; \
    switch (combine) { \
      case COMBINE_SUM: for (size_t i = 0; i < n; i++) d[i] = (S)((float)d[i] + (float)x[i]); break; \
      case COMBINE_PROD: for (size_t i = 0; i < n; i++) d[i] = (S)((float)d[i] * (float)x[i]); break; \
      case COMBINE_MAX: for (size_t i = 0; i < n; i++) if ((float)x[i] > (float)d[i]) d[i] = x[i]; break; \
      default: for (size_t i = 0; i < n; i++) if ((float)x[i] < (float)d[i]) d[i] = x[i]; break; } } while (0)

/* dst[i] = dst[i] combine src[i], n elements */
static void combine_into(ncclDataType_t t, int combine, void *dst, const void *src, size_t n) {
  switch (t) {
  case ncclInt8: INTEGER_LOOP(int8_t, uint8_t); return;
  case ncclUint8: INTEGER_LOOP(uint8_t, uint8_t); return;
  case ncclInt32: INTEGER_LOOP(int32_t, uint32_t); return;
  case ncclUint32: INTEGER_LOOP(uint32_t, uint32_t); return;
  case ncclInt64: INTEGER_LOOP(int64_t, uint64_t); return;
  case ncclUint64: INTEGER_LOOP(uint64_t, uint64_t); return;
  /* float32 and float16: the arithmetic below ((float)a op (float)b, the load into double exact) as typed loops, which
     vectorize, in place of one load, combine and store through double per element */
  case ncclFloat32: FLOAT_LOOP(float); return;
  case ncclFloat16: FLOAT_LOOP(_Float16); return;
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

/* dst[i] = src[i] * scalar in the type (FuncPreMulSum, src/device/reduce_kernel.h; a floating average's 1 / nranks);
   dst may be src */
static void premultiply(ncclDataType_t t, void *dst, const void *src, const unsigned char *scalar, size_t n) {
  if (dst != src) memmove(dst, src, n * SIZE[t]);
  const size_t z = SIZE[t];
  const double s = load(t, scalar);
  unsigned char *d = dst;
  for (size_t i = 0; i < n; i++, d += z) store(t, d, arithmetic(t, load(t, d), s, COMBINE_PROD));
}

/* FuncSumPostDiv::divide (src/device/reduce_kernel.h, an integer average): the wrapped sum's magnitude over n, with the
   sum's sign */
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

/* -- the process's attach and its pool --------------------------------------------------------------------------------
   The process is a client of its node's bridge in one slot a thread: one attach of the node's region (mesh-call.h
   mesh_attach), made at its first communicator, and as many more slots of the same mapping as it has threads
   (`contexts`, T: threads_wanted; mesh.h mesh_slot_take, each a write ring of its own), whose slots own every
   communicator's indices and pages.  The attach's pool is its T threads: the calling thread, of slot 0, and T - 1
   workers, worker t of slot t (`worker`), started with the attach (pool_start).  A worker never sleeps: it spins on
   the pool's `generation` word, the count of calls told the pool in its high bits and the threads the latest call is
   moved on in its low 8, and joins a call only where its number is below that call's threads, moving the call's
   channels c with c % T its number (channels_progress) and then counting itself off `pending`, the countdown the
   call's end reads; a generation of no threads ends it (pool_end).  One call is moved on the pool at a time (`lock`,
   work_launch), so each slot's write ring has one producer at a time (mesh-call.h mesh_write) and calls on different
   communicators are made one after the other.  A region a stop released, or a bridge of another layout replaced, is
   expired (its layout no longer MESH_LAYOUT): while the process has a communicator of it, a new communicator fails
   naming the expiry, and once it has none the attach is let go (its workers ended, then mesh_detach of each slot: its
   indices freed and their pages dropped), so the next communicator attaches the bridge's next region and no process
   has two.  An attach that is never let go ends with the process, its workers with it. */
struct attach;
struct work;
struct worker { pthread_t thread; struct attach *attach; uint32_t t; };
struct attach {
  struct mesh_ctx context[CHANNELS];
  uint32_t contexts;
  uint64_t comms;
  struct worker worker[CHANNELS];
  pthread_mutex_t lock;
  struct work *work;
  _Alignas(128) uint64_t generation;
  _Alignas(128) uint32_t pending;
};
static struct attach *attached;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;

static void channels_progress(struct work *work, uint32_t t);
/* worker t of a pool: from the generation its pool started at (0, no call yet told), each new generation read; at one
   whose call is moved on more than t threads, its channels of the call moved and its count taken off `pending`; at one
   of no threads, its end */
static void *worker_main(void *arg) {
  const struct worker *self = arg;
  struct attach *a = self->attach;
  for (uint64_t seen = 0;;) {
    uint64_t g;
    while ((g = __atomic_load_n(&a->generation, __ATOMIC_ACQUIRE)) == seen) {}
    seen = g;
    if (!(g & 255)) return NULL;
    if (self->t < (g & 255)) {
      channels_progress(a->work, self->t);
      __atomic_fetch_sub(&a->pending, 1, __ATOMIC_RELEASE);
    }
  }
}
/* the pool's workers started, one for each slot past the first, before any communicator takes an index of the attach;
   where worker t cannot be started, slot t and every later one are let go, so each slot's ring is its own thread's */
static void pool_start(struct attach *a) {
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INTERACTIVE, 0);
  for (uint32_t t = 1; t < a->contexts; t++) {
    a->worker[t] = (struct worker){.attach = a, .t = t};
    if (!pthread_create(&a->worker[t].thread, &attributes, worker_main, a->worker + t)) continue;
    for (uint32_t w = a->contexts; w-- > t;) mesh_detach(&a->context[w]);
    a->contexts = t;
  }
  pthread_attr_destroy(&attributes);
}
/* the pool's workers ended (a generation of no threads) and joined; no call is under way on it */
static void pool_end(struct attach *a) {
  if (a->contexts < 2) return;
  __atomic_store_n(&a->generation, ((__atomic_load_n(&a->generation, __ATOMIC_RELAXED) >> 8) + 1) << 8, __ATOMIC_RELEASE);
  for (uint32_t t = 1; t < a->contexts; t++) pthread_join(a->worker[t].thread, NULL);
}

static int attach_expired(const struct attach *a) {
  return __atomic_load_n(&a->context[0].M->version, __ATOMIC_ACQUIRE) != MESH_LAYOUT;
}
/* the attach let go where its region is expired and it has no communicator of it (so no call is under way on its
   pool): its workers ended (pool_end), then its slots left; under attach_lock */
static void attach_settle(void) {
  struct attach *a = attached;
  if (!a || a->comms || !attach_expired(a)) return;
  pool_end(a);
  for (uint32_t w = a->contexts; w-- > 0;) mesh_detach(&a->context[w]);
  pthread_mutex_destroy(&a->lock);
  free(a);
  attached = NULL;
}
/* the threads of the process's pool (T): NCCL_MESH_THREADS where set (1 to CHANNELS), else THREADS_DEFAULT, at most
   half the node's top-level cores (its bridge's port children spin on the others).
   THREADS_DEFAULT is the fastest measured on the fleet (b80ab9e22; ten top-level cores, five or six port children):
   four ranks moved 12 MB all-gathers at 101, 112, 104 and 98 GB/s on 3, 4, 5 and 8 threads, all-reduces at 78, 91, 81
   and 73 */
#define THREADS_DEFAULT 4
static uint32_t threads_wanted(void) {
  const char *set = getenv("NCCL_MESH_THREADS");
  long n = set && *set ? strtol(set, NULL, 10) : 0;
  if (n <= 0) {
    int cores = 0;
    size_t size = sizeof cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, NULL, 0) || cores <= 0) cores = 2;
    n = cores / 2 < THREADS_DEFAULT ? cores / 2 : THREADS_DEFAULT;
  }
  return n < 1 ? 1 : n > (long)CHANNELS ? CHANNELS : (uint32_t)n;
}
/* the process's attach, made at its first use and again after an expired one is let go: its region attached, then a
   slot of the same mapping for each further thread wanted while one is vacant (fewer threads where none is), then its
   pool's workers started (pool_start); 0, ESTALE where the process has communicators of an expired region, ENOMEM, or
   mesh_attach's error; under attach_lock */
static int attach_current(struct attach **out) {
  attach_settle();
  if (attached) { *out = attached; return attach_expired(attached) ? ESTALE : 0; }
  struct attach *a = NULL;
  if (posix_memalign((void **)&a, _Alignof(struct attach), sizeof *a)) return ENOMEM;
  memset(a, 0, sizeof *a);
  const int status = mesh_attach(&a->context[0]);
  if (status) { free(a); return status; }
  a->contexts = 1;
  for (const uint32_t wanted = threads_wanted(); a->contexts < wanted; a->contexts++) {
    struct mesh_ctx *c = &a->context[a->contexts];
    *c = (struct mesh_ctx){.fd = -1};
    if (mesh_slot_take(c, a->context[0].M)) { *c = (struct mesh_ctx){0}; break; }
    /* the expiry read again after the slot's compare-exchange, as mesh_attach reads it */
    if (__atomic_load_n(&c->M->version, __ATOMIC_SEQ_CST) != MESH_LAYOUT) { mesh_detach(c); break; }
  }
  pthread_mutex_init(&a->lock, NULL);
  pool_start(a);
  *out = attached = a;
  return 0;
}
/* communicator `name`'s attach not made (attach_current's error) */
static ncclResult_t attach_fail(int status, const char *name) {
  if (status == ENOMEM) return fail(ncclSystemError, "communicator %s: out of memory", name);
  if (status == EBUSY) return fail(ncclSystemError, "communicator %s: every client slot of this node's bridge is taken (mesh.h MESH_CLIENTS)", name);
  if (status == ESTALE)
    return fail(ncclRemoteError, "communicator %s: the region of this node's bridge was released or replaced while this process has "
                "communicators of it; once it has destroyed them (ncclCommDestroy), its next communicator attaches the bridge's "
                "next region", name);
  return fail(ncclSystemError, "communicator %s: this process attaches no region of this node's bridge (%s); the node's bridge "
              "must be up (etc/bridge/README.md)", name, strerror(status));
}

/* -- communicators ------------------------------------------------------------------------------------------------- */

/* A channel of a communicator at this rank: the slot its indices are of (`context`, the attach's), its receive indices
   (`receives`, a ring of the communicator's depth for each rank) and its source pages (`source[c]` the first of its
   SOURCES of class c, 0 short and 1 long, and `next[c]` the one it looks at first); a cache line each, since each is
   its own thread's */
struct channel { _Alignas(128) struct mesh_ctx *context; uint32_t receives, source[2], next[2]; };

/* A communicator: its ranks; each rank's node (`node`), the first receive index of each of its channels (`base`, rank
   r channel c at r CHANNELS + c) and its standing listener's address (`address`, where a split's or a shrink's
   communicator whose first rank it is meets), gathered at its init; the pages each channel of this rank has written to
   each rank (`sent`) and taken from each (`received`, channel c rank r at c nranks + r); its channels here
   (`channel`); the depth of each pair's receive ring (`depth`, comm_depth; 0 for one rank, which has no ring); the
   splits and shrinks made of it (`splits`, which name them); its bootstrap's sockets (`peers`: the root's to every
   rank, a rank's to the root) and its standing listener; the scratch a reduction's own contribution is made in; its
   life (`state`, enum comm_state: its failure, `failed` and `failure`, recorded with it, and `guard`, its moves') and
   its `references` (comm_release) */
struct ncclComm {
  int nranks, rank;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  uint32_t *node, *base;
  char (*address)[NCCL_UNIQUE_ID_BYTES];
  uint64_t *sent, *received;
  struct channel channel[CHANNELS];
  uint32_t depth, splits;
  struct attach *attach;
  int *peers, standing;
  unsigned char *scratch;
  size_t scratch_bytes;
  ncclResult_t failed;
  struct said failure;
  pthread_mutex_t guard;
  _Atomic uint32_t state, references;
};
/* A communicator's life, its `state` word, moved by comm_turn alone, under its guard, and read by one acquire load:
   LIVE from its init; FAILED once a call records its failure, from LIVE alone (its result and message, `failed` and
   `failure`, written before the word and never again); REVOKED once revoked or destroyed, from LIVE or FAILED, which
   ends each call and exchange under way on it at its next look and refuses every later one.  Its references: its
   own, which ncclCommDestroy drops, and one for each call queued or under way on it and each split, shrink or grow of
   it while it reads comm; whoever drops the last frees it (comm_release), so a destroy waits for nothing */
enum comm_state { COMM_LIVE, COMM_FAILED, COMM_REVOKED };

/* a communicator of `nranks` ranks named `name`, this process rank `rank`, with nothing of its bridge's yet; NULL where
   out of memory */
static struct ncclComm *comm_new(int nranks, int rank, const char *name) {
  struct ncclComm *comm = NULL;
  if (posix_memalign((void **)&comm, _Alignof(struct ncclComm), sizeof *comm)) return NULL;
  memset(comm, 0, sizeof *comm);
  const size_t n = (size_t)nranks;
  comm->node = calloc(n, sizeof *comm->node); comm->base = calloc(n * CHANNELS, sizeof *comm->base);
  comm->address = calloc(n, sizeof *comm->address);
  comm->sent = calloc(n * CHANNELS, sizeof *comm->sent); comm->received = calloc(n * CHANNELS, sizeof *comm->received);
  comm->peers = malloc(n * sizeof *comm->peers);
  if (!comm->node || !comm->base || !comm->address || !comm->sent || !comm->received || !comm->peers) {
    free(comm->node); free(comm->base); free(comm->address); free(comm->sent); free(comm->received); free(comm->peers);
    free(comm);
    return NULL;
  }
  for (size_t r = 0; r < n; r++) comm->peers[r] = -1;
  comm->nranks = nranks; comm->rank = rank; comm->standing = -1;
  for (uint32_t c = 0; c < CHANNELS; c++)
    comm->channel[c] = (struct channel){NULL, MESH_ABSENT, {MESH_ABSENT, MESH_ABSENT}, {0, 0}};
  pthread_mutex_init(&comm->guard, NULL);
  atomic_init(&comm->state, COMM_LIVE);
  atomic_init(&comm->references, 1);
  snprintf(comm->name, sizeof comm->name, "%s", name);
  return comm;
}

/* the process's attach taken for comm and counted (attach_current): 0, or its error */
static int comm_attach(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  const int status = attach_current(&comm->attach);
  if (!status) comm->attach->comms++;
  else comm->attach = NULL;
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* comm's source pages of this node's table (none for one rank, which sends nothing), each channel's of the slot of
   the thread that moves it (channel c, the attach's context c % its contexts): SOURCES of each class; 0, or
   mesh_pages' errno */
static int comm_sources(struct ncclComm *comm) {
  if (comm->nranks == 1) return 0;
  struct attach *a = comm->attach;
  int status = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t c = 0; !status && c < CHANNELS; c++) {
    struct channel *x = comm->channel + c;
    x->context = &a->context[c % a->contexts];
    if ((x->source[0] = mesh_pages(x->context, SOURCES, MESH_SHORT_PAYLOAD)) == MESH_ABSENT) status = errno;
    else if ((x->source[1] = mesh_pages(x->context, SOURCES, MESH_LONG_PAYLOAD)) == MESH_ABSENT) status = errno;
  }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* comm's receive indices of this node's table, each channel's of its source pages' slot: a ring of comm's depth for
   each rank, nranks depth indices, naming no page and landed 0 until one lands; 0, or mesh_pages' errno */
static int comm_receives(struct ncclComm *comm) {
  const uint32_t receives = (uint32_t)comm->nranks * comm->depth;
  int status = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t c = 0; !status && c < CHANNELS; c++) {
    struct channel *x = comm->channel + c;
    if ((x->receives = mesh_pages(x->context, receives, 0)) == MESH_ABSENT) status = errno;
  }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* comm's indices freed (their pages dropped: a page a SEND still reads goes back to the pool at its completion) and its
   count on the process's attach taken off (attach_settle) */
static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  struct attach *a = comm->attach;
  if (a) {
    for (uint32_t c = 0; c < CHANNELS; c++) {
      struct channel *x = comm->channel + c;
      if (x->receives != MESH_ABSENT) mesh_invalidate(x->context, x->receives, (uint32_t)comm->nranks * comm->depth, 1);
      for (int k = 0; k < 2; k++)
        if (x->source[k] != MESH_ABSENT) mesh_invalidate(x->context, x->source[k], SOURCES, 1);
      x->receives = x->source[0] = x->source[1] = MESH_ABSENT;
    }
    a->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
}

/* comm's bootstrap's sockets released, under its guard */
static void bootstrap_free(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) close(comm->peers[r]);
  free(comm->peers);
  comm->peers = NULL;
  pthread_mutex_unlock(&comm->guard);
}

static void comm_free(struct ncclComm *comm) {
  bootstrap_free(comm);
  if (comm->standing >= 0) close(comm->standing);
  comm_leave(comm);
  pthread_mutex_destroy(&comm->guard);
  free(comm->node); free(comm->base); free(comm->address); free(comm->sent); free(comm->received); free(comm->scratch);
  free(comm);
}

/* one reference to comm taken, or dropped (enum comm_state names the holders): the last drop frees it (comm_free) */
static void comm_hold(struct ncclComm *comm) { atomic_fetch_add_explicit(&comm->references, 1, memory_order_relaxed); }
static void comm_release(struct ncclComm *comm) {
  if (atomic_fetch_sub_explicit(&comm->references, 1, memory_order_acq_rel) == 1) comm_free(comm);
}

/* comm's state moved toward `to` (enum comm_state), under its guard: to FAILED from LIVE alone, `result` and the
   message `said` recorded before the word; to REVOKED from LIVE or FAILED, every socket of its bootstrap shut down
   after the word, so an exchange under way ends; any other move changes nothing */
static void comm_turn(struct ncclComm *comm, enum comm_state to, ncclResult_t result, const struct said *said) {
  pthread_mutex_lock(&comm->guard);
  const uint32_t from = atomic_load_explicit(&comm->state, memory_order_relaxed);
  if (to == COMM_FAILED && from == COMM_LIVE) {
    comm->failed = result;
    comm->failure = *said;
    comm->failure.id = message_id();
    atomic_store_explicit(&comm->state, COMM_FAILED, memory_order_release);
  } else if (to == COMM_REVOKED && from != COMM_REVOKED) {
    atomic_store_explicit(&comm->state, COMM_REVOKED, memory_order_seq_cst);
    for (int r = 0; comm->peers && r < comm->nranks; r++)
      if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  }
  pthread_mutex_unlock(&comm->guard);
}

/* whether comm was revoked (one acquire load of its state) */
static int revoked(ncclComm_t comm) { return atomic_load_explicit(&comm->state, memory_order_acquire) == COMM_REVOKED; }
static ncclResult_t revoked_fail(ncclComm_t comm) {
  return fail(ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): every call on it is refused until ncclCommDestroy", comm->name);
}

/* comm's state as a call reads it, one acquire load: ncclSuccess where LIVE; its failure, the one it recorded
   (comm_fail), named again (ncclGetLastError) where FAILED; its revoke's where REVOKED */
static ncclResult_t comm_failure(struct ncclComm *comm) {
  switch (atomic_load_explicit(&comm->state, memory_order_acquire)) {
  case COMM_LIVE: return ncclSuccess;
  case COMM_FAILED: return told(comm->failed, &comm->failure);
  default: return revoked_fail(comm);
  }
}

/* comm failed for good (comm_turn to FAILED: its result and its message, formatted here, recorded where it is the
   first), and the failure named that comm_failure then reads: the first one recorded, or its revoke's */
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) __attribute__((format(printf, 3, 4)));
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) {
  struct said said = {0, ""};
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(said.text, sizeof said.text, format, arguments);
  va_end(arguments);
  comm_turn(comm, COMM_FAILED, result, &said);
  return comm_failure(comm);
}

/* -- NCCL's bootstrap (NVIDIA/nccl src/bootstrap.cc) --------------------------------------------------------------------
   The unique id is the root's address, host:port (NCCL_COMM_ID where set, else a listener this process opens on the
   node's fabric address, else on its host name), and the root listens on that host alone where it is an IPv6 address
   (the fabric's ULA), else on every address; rank 0 takes every rank's report there and sends each the whole table,
   once for each of the init's two rounds, on sockets the communicator has for its later exchanges (a split's, a
   grow's) and releases where its init fails.  A non-root's connect does not block: it is looked at every
   BOOTSTRAP_SLICE_MS and tried again a slice after it fails; the root looks for a connection every slice; each wait
   reads the communicator's revoke (ncclCommRevoke, ncclCommDestroy) at each look, and the revoke shuts the sockets
   down, so an exchange under way ends.  Nothing else bounds a wait of the bootstrap. */
static struct { int fd; char id[NCCL_UNIQUE_ID_BYTES]; } root = {-1, {0}};
#define BOOTSTRAP_SLICE_MS 100

/* a rank's report in a communicator's init's first round: its status (0, else why it cannot go on, so every rank fails
   alike), its bridge's node, its node's pool (`pages`: the fewer pages of its two classes, mesh.h struct mesh_pool)
   and its standing listener's address */
struct report { int32_t rank, status; uint32_t node, pages; char address[NCCL_UNIQUE_ID_BYTES]; };
/* its report in the second round: its status (its receive indices taken, else why not) and the first receive index of
   each of its channels */
struct base_report { int32_t status; uint32_t base[CHANNELS]; };

/* the depth of each pair's receive ring, alike at every rank (each computes it from the same first-round reports;
   nranks > 1): the pages each pair of a channel numbers before it comes back to an index, at most RECEIVES, and the
   most for which, at every node v of the communicator, its c_v ranks' source pages and a page at each index of their
   channels' rings, c_v CHANNELS (SOURCES + (nranks - 1) depth) pages of either class, fit in half the node's pool (the
   other half its bridge's lanes' receives, its relays' pages and its other clients'); 0 where no depth fits */
static uint32_t comm_depth(const struct report *reports, int nranks) {
  uint32_t depth = RECEIVES;
  for (int r = 0; r < nranks; r++) {
    uint64_t here = 0;
    for (int q = 0; q < nranks; q++) here += reports[q].node == reports[r].node;
    const uint64_t room = reports[r].pages / 2, sources = here * CHANNELS * SOURCES;
    const uint64_t most = room > sources ? (room - sources) / (here * CHANNELS * (uint64_t)(nranks - 1)) : 0;
    if (most < depth) depth = (uint32_t)most;
  }
  return depth;
}

/* a listener on `host` alone where it is an IPv6 address, else on every address, at *port (0: any, *port then the
   one bound), non-blocking (each accept is looked for first): its descriptor, or -1 and errno */
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

/* the host a listener of this node's is reached at: a peer is reached on the fabric, never by a name that resolves to
   Wi-Fi or a LAN, so the first address under MESH_ULA_PREFIX of an interface other than loopback (bin/mesh-router-init.sh
   puts one on each fabric port), else, on a node with none, its host name; 0, or the errno of gethostname */
static int node_host(char *host, size_t room) {
  memset(host, 0, room);
  struct ifaddrs *interfaces = NULL;
  if (!getifaddrs(&interfaces)) {
    for (struct ifaddrs *i = interfaces; i && !*host; i = i->ifa_next) {
      char text[INET6_ADDRSTRLEN];
      if (i->ifa_addr && i->ifa_addr->sa_family == AF_INET6 && !(i->ifa_flags & IFF_LOOPBACK) &&
          inet_ntop(AF_INET6, &((struct sockaddr_in6 *)i->ifa_addr)->sin6_addr, text, sizeof text) &&
          !strncmp(text, MESH_ULA_PREFIX, strlen(MESH_ULA_PREFIX)) && strlen(text) < room)
        snprintf(host, room, "%s", text);
    }
    freeifaddrs(interfaces);
  }
  return *host || !gethostname(host, room - 1) ? 0 : errno;
}

/* a listener of this node's at a port the system gives, its host:port in `address`: its descriptor, or -1 and errno */
static int listener_open(char *address, size_t room) {
  char host[256];
  uint16_t port = 0;
  const int status = node_host(host, sizeof host);
  if (status) { errno = status; return -1; }
  const int fd = bootstrap_listen(host, &port);
  if (fd >= 0) snprintf(address, room, "%s:%u", host, port);
  return fd;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return fail(ncclInvalidArgument, "uniqueId is NULL");
  memset(id->internal, 0, sizeof id->internal);
  const char *given = getenv("NCCL_COMM_ID");
  if (given && *given) { snprintf(id->internal, sizeof id->internal, "%s", given); return ncclSuccess; }
  if (root.fd >= 0) close(root.fd);
  if ((root.fd = listener_open(id->internal, sizeof id->internal)) < 0)
    return fail(ncclSystemError, "the bootstrap root's listener: %s", strerror(errno));
  memcpy(root.id, id->internal, sizeof root.id);
  return ncclSuccess;
}

/* `size` bytes sent or received whole on comm's bootstrap socket `fd`, its waits in slices (the socket's SO_RCVTIMEO
   and SO_SNDTIMEO) that read the revoke: 0, ECANCELED (revoked), ECONNRESET (the peer ended the connection), or the
   errno */
static int whole(struct ncclComm *comm, int fd, void *bytes, size_t size, int sending) {
  for (size_t done = 0; done < size;) {
    const ssize_t moved = sending ? send(fd, (char *)bytes + done, size - done, 0) : recv(fd, (char *)bytes + done, size - done, 0);
    if (moved < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      if (revoked(comm)) return ECANCELED;
      continue;
    }
    if (moved <= 0) return moved ? errno : ECONNRESET;
    done += (size_t)moved;
  }
  return 0;
}

/* socket `fd` of comm's init placed at peers[r] (-1: the one there released), blocking in slices (an accepted one takes
   its listener's O_NONBLOCK), under comm's guard, where a revoke shuts it down, a send on it after that or a peer's
   reset an error, never SIGPIPE */
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

/* the bootstrap's sockets of comm's init at `id`: a non-root's to the root (peers[0], placed once connected), or the
   root's to each rank r (peers[r], placed by the rank its first-round report names, that report put in `reports`; a
   socket not yet placed waits at peers[0]; one whose report fails or names no rank still to meet is released alone,
   the root accepting the next).  The root accepts on `standing` where it is a listener (a parent's standing listener,
   left open), else on the listener ncclGetUniqueId opened for `id`, else on one it opens at id's port, releasing
   either once every rank has met: 0, ECANCELED (comm revoked), or the errno */
static int bootstrap_open(struct ncclComm *comm, const char *id, struct report *reports, int standing) {
  int *peers = comm->peers, status = 0;
  char host[NCCL_UNIQUE_ID_BYTES + 1];
  snprintf(host, sizeof host, "%s", id);
  char *colon = strrchr(host, ':');
  if (!colon) return EINVAL;
  *colon = 0;
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
            else if (poll(&look, 1, BOOTSTRAP_SLICE_MS) < 0) error = errno == EINTR ? EINPROGRESS : errno;
            else if (look.revents && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length)) error = errno;
          if (error) close(fd);
          else bootstrap_put(comm, 0, fd);
        }
        freeaddrinfo(addresses);
      }
      if (peers[0] < 0) usleep(BOOTSTRAP_SLICE_MS * 1000);
    }
    return revoked(comm) ? ECANCELED : 0;
  }
  uint16_t port = (uint16_t)atoi(colon + 1);
  int listener = standing >= 0 ? standing : !strcmp(root.id, id) ? root.fd : -1;
  if (listener < 0 && (listener = bootstrap_listen(host, &port)) < 0) return errno;
  for (int r = 1; r < comm->nranks && !status;) {
    struct report heard;
    struct pollfd look = {listener, POLLIN, 0};
    int fd = -1;
    while (fd < 0 && !status)
      if (revoked(comm)) status = ECANCELED;
      else if (poll(&look, 1, BOOTSTRAP_SLICE_MS) < 0 && errno != EINTR) status = errno;
      else if ((fd = accept(listener, NULL, NULL)) < 0 && errno != EWOULDBLOCK && errno != ECONNABORTED && errno != EINTR) status = errno;
    if (!status) bootstrap_put(comm, 0, fd);
    if (!status && revoked(comm)) status = ECANCELED;
    const int read = status ? status : whole(comm, fd, &heard, sizeof heard, 0);
    if (read == ECANCELED) status = read;
    if (status) { bootstrap_put(comm, 0, -1); break; }
    /* a connection whose report does not come, or names no rank still to meet, is released alone */
    if (read || heard.rank <= 0 || heard.rank >= comm->nranks || peers[heard.rank] >= 0) { bootstrap_put(comm, 0, -1); continue; }
    reports[heard.rank] = heard;
    pthread_mutex_lock(&comm->guard);
    peers[heard.rank] = fd; peers[0] = -1;
    pthread_mutex_unlock(&comm->guard);
    r++;
  }
  if (listener != standing) close(listener);
  if (listener == root.fd) root.fd = -1;
  return status;
}

/* one round of the init's gather on comm's bootstrap sockets, each rank's `bytes` in `table` (rank r's at table + r
   bytes, this rank's there already): each other rank's to the root (`sent`: there already, as the root's
   bootstrap_open read each first-round report), then the whole table to each; 0, or the errno (whole) */
static int gather(struct ncclComm *comm, void *table, size_t bytes, int sent) {
  const int *peers = comm->peers, nranks = comm->nranks, rank = comm->rank;
  unsigned char *at = table;
  const size_t size = (size_t)nranks * bytes;
  int status = 0;
  if (rank) {
    if (!sent) status = whole(comm, peers[0], at + (size_t)rank * bytes, bytes, 1);
    return status ? status : whole(comm, peers[0], at, size, 0);
  }
  for (int r = 1; r < nranks && !status && !sent; r++) status = whole(comm, peers[r], at + (size_t)r * bytes, bytes, 0);
  for (int r = 1; r < nranks && !status; r++) status = whole(comm, peers[r], at, size, 1);
  return status;
}

/* `bytes` of every rank's gathered over comm's bootstrap (rank r's at reports + r bytes, this rank's `local`): the
   configuration a split or a grow exchanges before the communicator it makes exists, under the split's or grow's
   reference to comm, made by one thread at a time as comm's calls are (NCCL's rule), and ended by comm's revoke */
static ncclResult_t exchange(struct ncclComm *comm, const void *local, size_t bytes, void *reports) {
  memcpy((unsigned char *)reports + (size_t)comm->rank * bytes, local, bytes);
  int error = comm->nranks > 1 && !comm->peers ? ENOTCONN : 0;
  if (!error && comm->nranks > 1 && comm->rank) {
    if (comm->peers[0] < 0) error = ENOTCONN;
    if (!error) error = whole(comm, comm->peers[0], (void *)local, bytes, 1);
    if (!error) error = whole(comm, comm->peers[0], reports, (size_t)comm->nranks * bytes, 0);
  } else if (!error && comm->nranks > 1) {
    for (int r = 1; r < comm->nranks && !error; r++)
      error = comm->peers[r] < 0 ? ENOTCONN : whole(comm, comm->peers[r], (unsigned char *)reports + (size_t)r * bytes, bytes, 0);
    for (int r = 1; r < comm->nranks && !error; r++) error = whole(comm, comm->peers[r], reports, (size_t)comm->nranks * bytes, 1);
  }
  return error ? fail(ncclRemoteError, "communicator %s: an exchange over its bootstrap: %s", comm->name, strerror(error)) : ncclSuccess;
}

/* A communicator's init at its id, the one path of every init (ncclCommInitRank, a split's, a shrink's, a grow's), in
   two rounds of its bootstrap's gather.  First the process attached (comm_attach), the communicator's source pages
   taken (comm_sources) and its standing listener opened, its bootstrap's sockets opened (bootstrap_open, the root
   accepting on `standing` where it is a listener), and every rank's report gathered: its status, node, pool and
   standing listener, from which every rank computes the same depth (comm_depth).  Then each rank takes its receive
   rings, nranks depth indices a channel (comm_receives), and the second round gathers each rank's status and the
   first receive index of each of its channels: with the first, the only exchange its calls need.  A rank that cannot
   go on, or a depth of 0, fails every rank alike, made with nothing of its bridge's and its sockets released; a revoke
   fails it as a value */
static ncclResult_t comm_init(struct ncclComm *made, int standing) {
  const int nranks = made->nranks, rank = made->rank;
  struct report *reports = calloc((size_t)nranks, sizeof *reports);
  struct base_report *bases = calloc((size_t)nranks, sizeof *bases);
  if (!reports || !bases) { free(reports); free(bases); return fail(ncclSystemError, "communicator %s: out of memory", made->name); }
  struct report *mine = reports + rank;
  const int attaching = comm_attach(made);
  int status = attaching ? attaching : comm_sources(made);
  if (!status && nranks > 1 && (made->standing = listener_open(mine->address, sizeof mine->address)) < 0) status = errno;
  mine->rank = rank; mine->status = status;
  if (!attaching) {
    struct hdr *m = made->attach->context[0].M;
    const struct mesh_pool *pool = mesh_pool(m);
    mine->node = m->node;
    mine->pages = pool->cls[0].pages < pool->cls[1].pages ? pool->cls[0].pages : pool->cls[1].pages;
  }
  int met = nranks > 1 ? bootstrap_open(made, made->id, reports, standing) : 0;
  if (!met && nranks > 1) met = gather(made, reports, sizeof *reports, !rank);
  ncclResult_t result = ncclSuccess;
  if (attaching) result = attach_fail(attaching, made->name);
  else if (status)
    result = fail(ncclSystemError, "communicator %s: its %u source pages of each class of this node's bridge, or its listener: %s",
                  made->name, SOURCES * CHANNELS, strerror(status));
  else if (met == ECANCELED || revoked(made)) result = revoked_fail(made);
  else if (met) result = fail(ncclRemoteError, "communicator %s: the bootstrap at %s: %s", made->name, made->id, strerror(met));
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status)
      result = fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  if (!result && nranks > 1 && !(made->depth = comm_depth(reports, nranks)))
    result = fail(ncclSystemError, "communicator %s: half the pool of a node of its ranks has room for their source pages and no "
                  "receive page for each pair: %s", made->name, strerror(ENOSPC));
  if (!result && nranks > 1) {
    struct base_report *own = bases + rank;
    own->status = comm_receives(made);
    for (uint32_t c = 0; c < CHANNELS; c++) own->base[c] = made->channel[c].receives;
    met = gather(made, bases, sizeof *bases, 0);
    if (own->status)
      result = fail(ncclSystemError, "communicator %s: its %u receive indices of this node's bridge: %s", made->name,
                    (uint32_t)nranks * made->depth * CHANNELS, strerror(own->status));
    else if (met == ECANCELED || revoked(made)) result = revoked_fail(made);
    else if (met) result = fail(ncclRemoteError, "communicator %s: the bootstrap at %s: %s", made->name, made->id, strerror(met));
    for (int r = 0; !result && r < nranks; r++)
      if (bases[r].status)
        result = fail(ncclRemoteError, "communicator %s: rank %d could not take its receive indices: %s", made->name, r,
                      strerror(bases[r].status));
  }
  for (int r = 0; !result && r < nranks; r++) {
    made->node[r] = reports[r].node;
    for (uint32_t c = 0; c < CHANNELS; c++) made->base[(size_t)r * CHANNELS + c] = bases[r].base[c];
    memcpy(made->address[r], reports[r].address, sizeof made->address[r]);
    made->address[r][sizeof made->address[r] - 1] = 0;
  }
  free(reports); free(bases);
  if (result) { bootstrap_free(made); comm_leave(made); }
  return result;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return fail(ncclInvalidArgument, "the init's arguments: rank %d of %d", rank, nranks);
  char id[sizeof commId.internal + 1];
  memcpy(id, commId.internal, sizeof commId.internal);
  id[sizeof commId.internal] = 0;
  struct ncclComm *made = comm_new(nranks, rank, id);
  if (!made) return fail(ncclSystemError, "out of memory");
  memcpy(made->id, id, sizeof made->id);
  const ncclResult_t result = comm_init(made, -1);
  if (result) comm_release(made);
  else *comm = made;
  return result;
}

/* NCCL's revoke: comm's calls under way end as a value at their next look and every later one is refused
   (ncclInvalidUsage), its indices and pages left until ncclCommDestroy; ncclCommGetAsyncError reads ncclSuccess after
   it; from any thread.  revokeFlags is not read */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  (void)revokeFlags;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  comm_turn(comm, COMM_REVOKED, ncclSuccess, NULL);
  return ncclSuccess;
}

/* comm revoked (comm_turn) and its own reference dropped, returning at once: comm is freed (comm_free: its indices
   freed and their pages dropped, its sockets released) here where nothing else holds it, else by the last call, split,
   shrink or grow of it to end, each of which ends at its next look */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  comm_turn(comm, COMM_REVOKED, ncclSuccess, NULL);
  comm_release(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

/* comm's own failure (a call's, comm_fail), named again, where one load of its state reads FAILED; else ncclSuccess
   (none, or a revoke) */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  const int failed = atomic_load_explicit(&comm->state, memory_order_acquire) == COMM_FAILED;
  *asyncError = failed ? told(comm->failed, &comm->failure) : ncclSuccess;
  return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  if (!comm || !count) return fail(ncclInvalidArgument, "comm or count is NULL");
  *count = comm->nranks;
  return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  if (!comm || !rank) return fail(ncclInvalidArgument, "comm or rank is NULL");
  *rank = comm->rank;
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

const char *ncclGetLastError(ncclComm_t comm) { (void)comm; return last.text; }

/* host memory, aligned to the system's page: the calls copy what they send into the bridge's pages and what lands out
   of them, so a call's sendbuff and recvbuff may be any memory */
ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  if (!ptr) return fail(ncclInvalidArgument, "ptr is NULL");
  const int status = posix_memalign(ptr, (size_t)sysconf(_SC_PAGESIZE), size ? size : 1);
  return status ? fail(ncclSystemError, "ncclMemAlloc of %zu bytes: %s", size, strerror(status)) : ncclSuccess;
}

ncclResult_t ncclMemFree(void *ptr) {
  free(ptr);
  return ncclSuccess;
}

/* -- the calls ------------------------------------------------------------------------------------------------------- */

/* a call as its group has it: its kind, root, sendbuff, recvbuff and elements, its type, and its operator as the host
   computes it (`combine`; a floating average's premultiplier 1 / nranks, `pre`; an integer average's divide by nranks,
   `post`); the calling thread's group: its nesting (ncclGroupStart less ncclGroupEnd) and its calls */
struct call {
  int what, root;
  const void *send;
  void *recv;
  size_t count;
  ncclDataType_t type;
  ncclRedOp_t op;
  ncclComm_t comm;
  int combine, post, premultiplied;
  unsigned char pre[8];
};
static _Thread_local int nesting;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;

/* A phase of a call as this rank moves it: the messages it sends (`out`: `bytes` at `data`, premultiplied where `pre`,
   to rank `to`, or to every other rank where `to` is -1, and copied to `local` too where it is set: this rank's own
   part, made from each page as it is copied in) and those it receives (`in`: `bytes` from rank `from`, copied to
   `to`), and, where `acc` is set, its reduction: `bytes` at `acc` made the combine, rank by rank, of every rank's,
   this rank's at `own` and every other's a message to it.  A pair of ranks carries at most one message in a phase,
   so a message's first page is its pair's count of pages as the phase begins */
struct out { const unsigned char *data; size_t bytes; int to, pre; unsigned char *local; };
struct in { int from; size_t bytes; unsigned char *to; };
struct phase { struct out *out; struct in *in; uint32_t outs, ins; unsigned char *acc; const unsigned char *own; size_t bytes; };

/* a message's pages, every rank cutting it alike (design/PAGE-TABLE-SPEC-20261122.md section 6): one short page where
   it fits one, else long pages, each MESH_LONG_PAYLOAD but its last; page j's payload bytes, *offset its first byte's
   offset in the message */
static size_t page_payload(size_t bytes) { return bytes <= MESH_SHORT_PAYLOAD ? MESH_SHORT_PAYLOAD : MESH_LONG_PAYLOAD; }
static uint64_t pages_of(size_t bytes) { return (bytes + page_payload(bytes) - 1) / page_payload(bytes); }
static size_t page_at(size_t bytes, uint64_t j, size_t *offset) {
  const size_t p = page_payload(bytes);
  *offset = (size_t)j * p;
  return bytes - *offset < p ? bytes - *offset : p;
}

/* A channel's progress through a phase under way, toward and from each rank r, every count in the channel's own pages
   (its q-th page of a message the message's page id + q CHANNELS, id the channel's number): its pair's first page of
   the channel in the phase toward it (`first_out[r]`) and from it (`first_in[r]`), the channel's pages due from it
   (`due[r]`: those of its in-message, `from[r]`, or the reduction's positions), the first of them not TAKEN
   (`low[r]`) and each one's state (`state`, from `at[r]`: enum page_state); the page it sends next (out-message `o`,
   its page `j`, its destinations from the g-th of them) and the source page that page's payload was copied into and
   not yet written (`copied`, MESH_ABSENT none); each of the channel's reduction positions' next rank to combine
   (`next`); the in-message pages still to take and the reduction's contributions still to combine (`left`) */
struct channel_phase {
  struct call *call;
  const struct phase *p;
  struct channel *channel;
  uint32_t id;
  uint64_t *first_out, *first_in, *due, *low, *at;
  const struct in **from;
  uint32_t o, g, copied;
  uint64_t j;
  size_t copied_bytes;
  unsigned char *state;
  uint64_t positions, left;
  uint32_t *next;
};
/* A page due to a channel, moved by page_turn alone, on the thread that moves the channel: WAITING until its index
   has counted its landing; then TAKEN, its bytes copied to their place in recvbuff, or combined into the result at its
   rank's turn, and its index invalidated (nothing copied where the index names no page); or, a reduction page that
   lands before its rank's turn, HELD: left at its index, which still names it, until that turn, when it is combined
   from there and TAKEN */
enum page_state { PAGE_WAITING, PAGE_TAKEN, PAGE_HELD };

/* the pages of a message of `pages` pages that are channel id's */
static uint64_t channel_pages(uint64_t pages, uint32_t id) { return pages > id ? (pages - id + CHANNELS - 1) / CHANNELS : 0; }

/* a source page of channel x of class `cls` free to copy into: its uses 1, its word's alone (no write, SEND or landing
   of it under way: the use-checker, design/PAGE-TABLE-SPEC-20261122.md section 6), looked for from the class's `next`;
   MESH_ABSENT where none is */
static uint32_t source_free(struct channel *x, int cls) {
  for (uint32_t i = 0; i < SOURCES; i++) {
    const uint32_t at = (x->next[cls] + i) % SOURCES;
    struct mesh_view_entry entry;
    if (mesh_entry(x->context, x->source[cls] + at, &entry) || entry.uses != 1) continue;
    x->next[cls] = (at + 1) % SOURCES;
    /* the bridge's last drop of a use (release) before this rank's stores into the page */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return x->source[cls] + at;
  }
  return MESH_ABSENT;
}

/* the g-th of the ranks other than `me`, counting up from rank 0 */
static int other(int me, uint32_t g) { return (int)g + ((int)g >= me); }

/* the channel's pages of the phase sent one after another (each message's pages of the channel; each page to its
   destinations, one write to each MESH_DESTS of them by rank, every write from a source page of the channel's own,
   each destination the index its own pair's page number for the page names in its receive ring, of the channel's
   indices there; and, where the message has a `local`, the page copied there from its source page once its first
   write is on the ring, while its bytes are still in the core's cache), while a source page is free and the
   channel's write ring takes them: 0 (the rest wait for a later pass), or -1 with the call's failure in *result */
static int send_some(struct channel_phase *m, ncclResult_t *result) {
  struct call *c = m->call;
  struct ncclComm *comm = c->comm;
  struct channel *channel = m->channel;
  const struct phase *p = m->p;
  const int me = comm->rank;
  for (; m->o < p->outs; m->o++, m->j = m->id) {
    const struct out *x = p->out + m->o;
    const uint64_t pages = pages_of(x->bytes);
    const uint32_t dests = x->to < 0 ? (uint32_t)comm->nranks - 1 : 1;
    for (; m->j < pages; m->j += CHANNELS, m->g = 0)
      while (m->g < dests) {
        if (m->copied == MESH_ABSENT) {
          const uint32_t a = source_free(channel, x->bytes > MESH_SHORT_PAYLOAD);
          if (a == MESH_ABSENT) return 0;
          size_t offset;
          const size_t bytes = page_at(x->bytes, m->j, &offset);
          unsigned char *data = mesh_page_data(channel->context, a);
          if (x->pre) premultiply(c->type, data, x->data + offset, c->pre, bytes / SIZE[c->type]);
          else memcpy(data, x->data + offset, bytes);
          m->copied = a; m->copied_bytes = bytes;
        }
        struct mesh_dest dest[MESH_DESTS];
        const uint32_t n = dests - m->g < MESH_DESTS ? dests - m->g : MESH_DESTS;
        for (uint32_t i = 0; i < n; i++) {
          const int d = x->to < 0 ? other(me, m->g + i) : x->to;
          const uint64_t k = m->first_out[d] + m->j / CHANNELS;
          dest[i] = (struct mesh_dest){comm->node[d], comm->base[(size_t)d * CHANNELS + m->id] + (uint32_t)me * comm->depth +
                                                      (uint32_t)(k % comm->depth)};
        }
        const int status = mesh_write(channel->context, m->copied, (uint32_t)m->copied_bytes, dest, n);
        if (status == EAGAIN) return 0;
        if (status) {
          *result = comm_fail(comm, ncclInternalError, "communicator %s: %s's write of its source page %u: %s", comm->name,
                              CALL_NAME[c->what], m->copied, strerror(status));
          return -1;
        }
        if (x->local && !m->g) {
          size_t offset;
          page_at(x->bytes, m->j, &offset);
          memcpy(x->local + offset, mesh_page_data(channel->context, m->copied), m->copied_bytes);
        }
        m->copied = MESH_ABSENT;
        m->g += n;
      }
  }
  return 0;
}

/* the index the channel's page q from rank r lands at: the pair's page k of the channel at k mod depth of r's ring of
   the channel's receive indices (depth that of the pair's receive ring) */
static uint32_t receive_index(const struct channel_phase *m, int r, uint64_t q) {
  const struct ncclComm *comm = m->call->comm;
  return m->channel->receives + (uint32_t)r * comm->depth + (uint32_t)((m->first_in[r] + q) % comm->depth);
}

/* the channel's page q of the phase's pages from rank r moved on (enum page_state), its index read: a WAITING page
   once its index has counted its landing (the pair's page k of the channel is the (k / depth + 1)-th to land at its
   index), whatever landed, and a HELD one at its rank's turn.  Where the index names a page, an in-message's bytes due
   are copied from it as they are (a page of either class is followed by a long page's frames of the region, mesh.h)
   to their place in recvbuff (design/DEFRAGMENT-pseudocode-20261122.md), and a reduction page's combined straight into
   the result at its rank's turn at its position (rank 0's copied), then the index is invalidated: TAKEN; a reduction
   page that lands before its rank's turn is left at its index: HELD.  Where the index names none (its page was taken
   under an earlier count), nothing is copied: TAKEN.  0, or -1 with the call's failure in *result */
static int page_turn(struct channel_phase *m, int r, uint64_t q, ncclResult_t *result) {
  struct call *c = m->call;
  struct ncclComm *comm = c->comm;
  const struct phase *p = m->p;
  const struct in *x = m->from[r];
  unsigned char *state = m->state + m->at[r] + q;
  const uint32_t index = receive_index(m, r, q);
  struct mesh_view_entry entry;
  if (mesh_entry(m->channel->context, index, &entry)) {
    *result = comm_fail(comm, ncclInternalError, "communicator %s: its index %u could not be read", comm->name, index);
    return -1;
  }
  if (*state == PAGE_WAITING && entry.landed < (m->first_in[r] + q) / comm->depth + 1) return 0;
  const int turn = !x && m->next[q] == (uint32_t)r;
  if (!x && !turn && entry.data) { *state = PAGE_HELD; return 0; }
  if (entry.data) {
    size_t offset;
    const size_t bytes = page_at(x ? x->bytes : p->bytes, m->id + q * CHANNELS, &offset);
    if (x) memcpy(x->to + offset, entry.data, bytes);
    else if (!r) memcpy(p->acc + offset, entry.data, bytes);
    else combine_into(c->type, c->combine, p->acc + offset, entry.data, bytes / SIZE[c->type]);
    mesh_invalidate(m->channel->context, index, 1, 0);
  }
  *state = PAGE_TAKEN;
  return 0;
}

/* the channel's reduction position q combined on, rank by rank, while the next rank's contribution is here: this
   rank's own, a page TAKEN at its turn or before it (one whose index named none adds nothing), or a HELD one, combined
   from its index now (page_turn); rank 0's copied, each later one's combined.  0, or -1 with the call's failure in
   *result */
static int advance(struct channel_phase *m, uint64_t q, ncclResult_t *result) {
  struct call *c = m->call;
  const struct phase *p = m->p;
  const int n = c->comm->nranks, me = c->comm->rank;
  for (int r; (r = (int)m->next[q]) < n; m->next[q]++, m->left--) {
    if (r == me) {
      size_t offset;
      const size_t bytes = page_at(p->bytes, m->id + q * CHANNELS, &offset);
      if (!r) memcpy(p->acc + offset, p->own + offset, bytes);
      else combine_into(c->type, c->combine, p->acc + offset, p->own + offset, bytes / SIZE[c->type]);
      continue;
    }
    const unsigned char state = m->state[m->at[r] + q];
    if (state == PAGE_WAITING) return 0;
    if (state == PAGE_HELD && page_turn(m, r, q, result)) return -1;
  }
  return 0;
}

/* the channel's WAITING page q from rank r looked at (page_turn), and once TAKEN counted: an in-message's page off
   `left`, a reduction page by its position's combine (advance: at its turn it goes on to the position's later ranks
   already here).  0 (taken, held, or not landed yet), or -1 with the call's failure in *result */
static int take(struct channel_phase *m, int r, uint64_t q, ncclResult_t *result) {
  if (page_turn(m, r, q, result)) return -1;
  if (m->state[m->at[r] + q] != PAGE_TAKEN) return 0;
  if (m->from[r]) { m->left--; return 0; }
  return advance(m, q, result);
}

/* the channel's landed pages of the phase taken (take), each in the pass that finds it, every page independent of the
   others: from each rank, its pages from the first not TAKEN through the depth of its receive ring, each at an index
   of its own (a page further on lands at the index of one of these, and is looked for once that one is TAKEN, so no
   index is read for a later page while an earlier page there is due or HELD): 0, or -1 with the call's failure in
   *result */
static int take_some(struct channel_phase *m, ncclResult_t *result) {
  const int n = m->call->comm->nranks;
  const uint32_t depth = m->call->comm->depth;
  for (int r = 0; r < n; r++) {
    const unsigned char *state = m->state + m->at[r];
    while (m->low[r] < m->due[r] && state[m->low[r]] == PAGE_TAKEN) m->low[r]++;
    const uint64_t end = m->due[r] - m->low[r] > depth ? m->low[r] + depth : m->due[r];
    for (uint64_t q = m->low[r]; q < end; q++)
      if (state[q] == PAGE_WAITING && take(m, r, q, result)) return -1;
  }
  return 0;
}

static void channel_phase_end(struct channel_phase *m) {
  free(m->first_out); free(m->from); free(m->next); free(m->state);
  m->first_out = NULL; m->from = NULL; m->next = NULL; m->state = NULL;
}
/* channel id's progress through phase p begun: each pair's pages of the channel numbered on from its count (comm's
   `sent` to each rank, `received` from each, of the channel, the same at both ends, since every rank makes the same
   calls in the same sequence and cuts each message alike), and this rank's own contribution combined where it is rank
   0 (advance, which reads no index while every page is WAITING); ncclSuccess, or the failure (out of memory) */
static ncclResult_t channel_phase_begin(struct channel_phase *m, struct call *c, const struct phase *p, uint32_t id) {
  struct ncclComm *comm = c->comm;
  const int n = comm->nranks, me = comm->rank;
  *m = (struct channel_phase){.call = c, .p = p, .channel = comm->channel + id, .id = id, .copied = MESH_ABSENT, .j = id,
                              .positions = p->acc ? channel_pages(pages_of(p->bytes), id) : 0};
  m->first_out = calloc(5 * (size_t)n, sizeof *m->first_out);
  m->from = calloc((size_t)n, sizeof *m->from);
  m->next = calloc((size_t)m->positions + 1, sizeof *m->next);
  uint64_t pages = 0;
  if (m->first_out && m->from) {
    m->first_in = m->first_out + n; m->due = m->first_in + n; m->low = m->due + n; m->at = m->low + n;
    for (uint32_t i = 0; i < p->ins; i++) {
      m->from[p->in[i].from] = p->in + i;
      m->due[p->in[i].from] = channel_pages(pages_of(p->in[i].bytes), id);
    }
    for (int r = 0; p->acc && r < n; r++)
      if (r != me) m->due[r] = m->positions;
    for (int r = 0; r < n; r++) { m->at[r] = pages; pages += m->due[r]; }
  }
  m->state = calloc((size_t)pages + 1, 1);
  if (!m->first_out || !m->from || !m->next || !m->state) {
    channel_phase_end(m);
    return fail(ncclSystemError, "communicator %s: out of memory", comm->name);
  }
  m->left = pages + m->positions;
  uint64_t *sent = comm->sent + (size_t)id * n, *received = comm->received + (size_t)id * n;
  for (int r = 0; r < n; r++) { m->first_out[r] = sent[r]; m->first_in[r] = received[r]; received[r] += m->due[r]; }
  for (uint32_t o = 0; o < p->outs; o++)
    for (int d = 0; d < n; d++)
      if (d != me && (p->out[o].to < 0 || p->out[o].to == d)) sent[d] += channel_pages(pages_of(p->out[o].bytes), id);
  ncclResult_t result = ncclSuccess;
  for (uint64_t q = 0; q < m->positions && !advance(m, q, &result); q++) {}
  if (result) channel_phase_end(m);
  return result;
}

/* A call's work descriptor (call_move), which every thread that moves a channel of it reads: its phases (a second for
   an all-reduce of more than one page: the all-gather of the reduced segments); this rank's one copy of its own part
   where it lies apart from where it goes and no out-message carries it (`local`; an out-message's own: struct out),
   and the reduction's own contribution where comm's scratch makes it (`fill`, the bytes it is copied, or
   premultiplied, from), each made by the channels, page by page; the channels with pages (`active`: channels 0 to
   active - 1) and the threads that move them (`threads`, thread t channels t, t + threads, ...); each thread's result
   and its message */
struct local { const unsigned char *from; unsigned char *to; size_t bytes; };
struct work {
  struct call *call;
  struct phase phase[2];
  uint32_t phases, active, threads;
  struct local local;
  const unsigned char *fill;
  ncclResult_t result[CHANNELS];
  struct said said[CHANNELS];
};

/* channel id's pages of the reduction's own contribution made in comm's scratch from `fill`, before anything reads
   them */
static void channel_fill(struct work *work, uint32_t id) {
  struct call *c = work->call;
  const struct phase *p = work->phase;
  if (!work->fill) return;
  for (uint64_t j = id, pages = pages_of(p->bytes); j < pages; j += CHANNELS) {
    size_t offset;
    const size_t bytes = page_at(p->bytes, j, &offset);
    if (c->premultiplied) premultiply(c->type, c->comm->scratch + offset, work->fill + offset, c->pre, bytes / SIZE[c->type]);
    else memcpy(c->comm->scratch + offset, work->fill + offset, bytes);
  }
}
/* channel id's pages of this rank's local copy */
static void channel_local(struct work *work, uint32_t id) {
  const struct local *l = &work->local;
  for (uint64_t j = id, pages = pages_of(l->bytes); j < pages; j += CHANNELS) {
    size_t offset;
    const size_t bytes = page_at(l->bytes, j, &offset);
    memcpy(l->to + offset, l->from + offset, bytes);
  }
}
/* channel id's positions of the reduction's result divided by nranks (an integer average) */
static void channel_divide(struct work *work, uint32_t id) {
  struct call *c = work->call;
  const struct phase *p = work->phase;
  for (uint64_t j = id, pages = pages_of(p->bytes); j < pages; j += CHANNELS) {
    size_t offset;
    const size_t bytes = page_at(p->bytes, j, &offset);
    truncdiv(c->type, p->acc + offset, p->acc + offset, bytes / SIZE[c->type], (uint64_t)c->comm->nranks);
  }
}

/* Thread t's channels of a call moved, each through the call's phases (`state`, each channel's progress through its
   phase, `phase`, the index of that phase), each pass moving every one of them that is not done: its pages sent and
   its landed pages taken (send_some, take_some); a channel done with its first phase divides its positions where the
   call averages integers, then begins the second where there is one; the thread's part of the local copy made after
   its first pass's writes.  The waits are the call's own: they end once the channels' pages have landed, or at the
   communicator's revoke, which each pass reads and which ends it as a value.  The thread's result and its message
   are put in the work descriptor */
static void channels_progress(struct work *work, uint32_t t) {
  struct call *c = work->call;
  struct ncclComm *comm = c->comm;
  struct channel_phase state[CHANNELS];
  uint32_t phase[CHANNELS], channels[CHANNELS], count = 0, done = 0;
  ncclResult_t result = ncclSuccess;
  for (uint32_t id = t; id < work->active; id += work->threads) {
    channel_fill(work, id);
    if ((result = channel_phase_begin(state + count, c, work->phase, id))) break;
    phase[count] = 0;
    channels[count++] = id;
  }
  int local = !work->local.bytes;
  while (!result && done < count) {
    for (uint32_t i = 0; i < count; i++) {
      if (phase[i] == work->phases) continue;
      struct channel_phase *m = state + i;
      if (send_some(m, &result) || take_some(m, &result)) break;
      if (m->o < m->p->outs || m->left) continue;
      channel_phase_end(m);
      if (!phase[i] && c->post && work->phase[0].acc) channel_divide(work, channels[i]);
      if (++phase[i] < work->phases) { if ((result = channel_phase_begin(m, c, work->phase + phase[i], channels[i]))) break; }
      else done++;
    }
    if (result) break;
    if (!local) { for (uint32_t i = 0; i < count; i++) channel_local(work, channels[i]); local = 1; }
    if (done < count && revoked(comm)) { result = revoked_fail(comm); break; }
  }
  for (uint32_t i = 0; i < count; i++)
    if (phase[i] < work->phases) channel_phase_end(state + i);
  work->result[t] = result;
  if (result) work->said[t] = last;
}

/* a call's work moved on the process's pool, one call at a time (the pool's `lock`, which the calling thread takes
   once the call before it has ended, a wait of the call's own that reads its communicator's revoke at each look, so a
   revoke or a destroy ends it as a value): its threads past the first told by a new generation (the count raised and
   the call's threads put in its low 8 bits), thread 0's channels moved here, then each told worker waited for
   (`pending`, the call's own end): the first thread's failure, by thread number, named again (ncclGetLastError), or
   ncclSuccess */
static ncclResult_t work_launch(struct work *work) {
  struct ncclComm *comm = work->call->comm;
  struct attach *a = comm->attach;
  while (pthread_mutex_trylock(&a->lock))
    if (revoked(comm)) return revoked_fail(comm);
  if (work->threads > 1) {
    a->work = work;
    __atomic_store_n(&a->pending, work->threads - 1, __ATOMIC_RELAXED);
    __atomic_store_n(&a->generation, (((__atomic_load_n(&a->generation, __ATOMIC_RELAXED) >> 8) + 1) << 8) | work->threads,
                     __ATOMIC_RELEASE);
  }
  channels_progress(work, 0);
  while (__atomic_load_n(&a->pending, __ATOMIC_ACQUIRE)) {}
  pthread_mutex_unlock(&a->lock);
  for (uint32_t t = 0; t < work->threads; t++)
    if (work->result[t]) { last = work->said[t]; return work->result[t]; }
  return ncclSuccess;
}

/* rank q's segment of an all-reduce's elements, split as evenly as the ranks allow (the first count % nranks one
   more): its first element, *length its elements */
static size_t segment(size_t count, int n, int q, size_t *length) {
  const size_t each = count / (size_t)n, more = count % (size_t)n, r = (size_t)q;
  *length = each + (r < more);
  return r * each + (r < more ? r : more);
}

/* a reduction's own contribution, `bytes` at `from`: read in place where it needs no premultiply and lies apart from
   the result `acc`, else made in comm's scratch (premultiplied, or copied, by the channels: channel_fill, each position
   before the result's first store there); NULL where out of memory */
static const unsigned char *own_of(struct work *work, const unsigned char *from, const unsigned char *acc, size_t bytes) {
  struct call *c = work->call;
  const uintptr_t f = (uintptr_t)from, a = (uintptr_t)acc;
  if (!c->premultiplied && (f + bytes <= a || a + bytes <= f)) return from;
  struct ncclComm *comm = c->comm;
  if (comm->scratch_bytes < bytes) {
    unsigned char *grown = realloc(comm->scratch, bytes);
    if (!grown) return NULL;
    comm->scratch = grown; comm->scratch_bytes = bytes;
  }
  work->fill = from;
  return comm->scratch;
}
/* this rank's own part, `bytes` at `from`, copied to `to` where the two lie apart: from each page of out-message
   `sent` as it is copied in (send_some), where `sent` is given (a message of those bytes to other ranks), else by the
   channels page by page (`local`, channel_local); here at once where they overlap; nothing where they are one */
static void local_of(struct work *work, const unsigned char *from, unsigned char *to, size_t bytes, struct out *sent) {
  const uintptr_t f = (uintptr_t)from, t = (uintptr_t)to;
  if (from == to) return;
  if (f + bytes > t && t + bytes > f) memmove(to, from, bytes);
  else if (sent) sent->local = to;
  else work->local = (struct local){from, to, bytes};
}
/* the most pages any of a phase's messages, or its reduction, has */
static uint64_t phase_pages(const struct phase *p) {
  uint64_t most = p->acc ? pages_of(p->bytes) : 0;
  for (uint32_t o = 0; o < p->outs; o++) if (pages_of(p->out[o].bytes) > most) most = pages_of(p->out[o].bytes);
  for (uint32_t i = 0; i < p->ins; i++) if (pages_of(p->in[i].bytes) > most) most = pages_of(p->in[i].bytes);
  return most;
}

/* A call moved on its communicator (design/PAGE-TABLE-SPEC-20261122.md section 6), B its count's bytes:
     AllGather      this rank's B to every other rank (one write a page), landing at recvbuff + rank B
     Broadcast      the root's B to every other rank
     Reduce         every other rank's B to the root, which reduces them with its own
     ReduceScatter  segment d of every rank's to rank d, which reduces them with its own
     AllReduce      where B is one page, this rank's B to every other rank, each reducing them all (one round of pages);
                    else a reduce-scatter of the count split as evenly as the ranks allow, then an all-gather of the
                    reduced segments
     AlltoAll       segment d of this rank's to rank d, landing at recvbuff + rank B
     Gather         every other rank's B to the root, landing at its recvbuff + rank B
     Scatter        the root's segment d to rank d
   This rank's own part is copied in place (none where it is there already), from each page as it is copied in where
   the call sends those bytes (an all-gather, a broadcast's root: local_of); a reduction goes rank by rank, so every
   rank computes the same bits; a floating average premultiplies every contribution by 1 / nranks, and an integer one
   divides the reduced sum by nranks.  Its pages are moved in channels, on as many of the pool's threads as it has
   channels with pages (work_launch): a call of one page on the calling thread alone */
static ncclResult_t call_move(struct call *c) {
  struct ncclComm *comm = c->comm;
  const int n = comm->nranks, me = comm->rank, root = c->root;
  const size_t z = SIZE[c->type], bytes = c->count * z;
  const unsigned char *send = c->send;
  unsigned char *recv = c->recv;
  if (!bytes) return ncclSuccess;
  struct out *out = calloc(2 * (size_t)n, sizeof *out);
  struct in *in = calloc(2 * (size_t)n, sizeof *in);
  if (!out || !in) { free(out); free(in); return fail(ncclSystemError, "communicator %s: out of memory", comm->name); }
  struct work work = {.call = c, .phases = 1};
  struct phase *p = work.phase;
  *p = (struct phase){out, in, 0, 0, NULL, NULL, 0};
  ncclResult_t result = ncclSuccess;
  switch (c->what) {
  case CALL_ALLGATHER:
    out[p->outs++] = (struct out){send, bytes, -1, 0, NULL};
    local_of(&work, send, recv + (size_t)me * bytes, bytes, n > 1 ? out : NULL);
    for (int s = 0; s < n; s++)
      if (s != me) in[p->ins++] = (struct in){s, bytes, recv + (size_t)s * bytes};
    break;
  case CALL_BROADCAST:
    if (me != root) { in[p->ins++] = (struct in){root, bytes, recv}; break; }
    out[p->outs++] = (struct out){send, bytes, -1, 0, NULL};
    local_of(&work, send, recv, bytes, n > 1 ? out : NULL);
    break;
  case CALL_REDUCE:
    if (me != root) { out[p->outs++] = (struct out){send, bytes, root, c->premultiplied, NULL}; break; }
    p->acc = recv; p->bytes = bytes;
    break;
  case CALL_REDUCE_SCATTER:
    for (int d = 0; d < n; d++)
      if (d != me) out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, c->premultiplied, NULL};
    p->acc = recv; p->bytes = bytes;
    break;
  case CALL_ALLREDUCE:
    if (bytes <= MESH_LONG_PAYLOAD) { p->acc = recv; p->bytes = bytes; break; }
    for (int d = 0; d < n; d++) {
      size_t length;
      const size_t first = segment(c->count, n, d, &length);
      if (d != me) out[p->outs++] = (struct out){send + first * z, length * z, d, c->premultiplied, NULL};
      else { p->acc = recv + first * z; p->bytes = length * z; }
    }
    break;
  case CALL_ALLTOALL:
    local_of(&work, send + (size_t)me * bytes, recv + (size_t)me * bytes, bytes, NULL);
    for (int d = 0; d < n; d++)
      if (d != me) {
        out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, 0, NULL};
        in[p->ins++] = (struct in){d, bytes, recv + (size_t)d * bytes};
      }
    break;
  case CALL_GATHER:
    if (me != root) { out[p->outs++] = (struct out){send, bytes, root, 0, NULL}; break; }
    local_of(&work, send, recv + (size_t)root * bytes, bytes, NULL);
    for (int s = 0; s < n; s++)
      if (s != root) in[p->ins++] = (struct in){s, bytes, recv + (size_t)s * bytes};
    break;
  default: /* CALL_SCATTER */
    if (me != root) { in[p->ins++] = (struct in){root, bytes, recv}; break; }
    local_of(&work, send + (size_t)root * bytes, recv, bytes, NULL);
    for (int d = 0; d < n; d++)
      if (d != root) out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, 0, NULL};
    break;
  }
  if (p->acc) {
    const size_t at = (size_t)(p->acc - recv);
    const unsigned char *own = c->what == CALL_REDUCE_SCATTER ? send + (size_t)me * bytes : send + at;
    if (!(p->own = own_of(&work, own, p->acc, p->bytes))) result = fail(ncclSystemError, "communicator %s: out of memory", comm->name);
    /* the single-phase all-reduce sends what it reduces with, made once */
    if (c->what == CALL_ALLREDUCE && bytes <= MESH_LONG_PAYLOAD) out[p->outs++] = (struct out){p->own, bytes, -1, 0, NULL};
  }
  if (!result && c->what == CALL_ALLREDUCE && bytes > MESH_LONG_PAYLOAD) {
    struct phase *q = work.phase + 1;
    *q = (struct phase){out + n, in + n, 0, 0, NULL, NULL, 0};
    q->out[q->outs++] = (struct out){p->acc, p->bytes, -1, 0, NULL};
    for (int s = 0; s < n; s++)
      if (s != me) {
        size_t length;
        const size_t first = segment(c->count, n, s, &length);
        q->in[q->ins++] = (struct in){s, length * z, recv + first * z};
      }
    work.phases = 2;
  }
  if (!result) {
    uint64_t most = phase_pages(p), local = pages_of(work.local.bytes);
    if (work.phases > 1 && phase_pages(work.phase + 1) > most) most = phase_pages(work.phase + 1);
    if (local > most) most = local;
    const uint32_t threads = comm->attach->contexts;
    work.active = most < CHANNELS ? (uint32_t)most : CHANNELS;
    work.threads = work.active < threads ? work.active : threads;
    if (work.active) result = work_launch(&work);
  }
  free(out); free(in);
  return result;
}

/* a call moved on its communicator, there and then: refused where its communicator failed (its failure named again)
   or was revoked (comm_failure, one load of its state) */
static ncclResult_t call_one(struct call *c) {
  const ncclResult_t result = comm_failure(c->comm);
  return result ? result : call_move(c);
}

ncclResult_t ncclGroupStart(void) { nesting++; return ncclSuccess; }

/* the group's calls made one after another, as they were queued, each to its end, every one made whatever another's
   result, and each one's reference to its communicator dropped at its end (comm_release: the last frees a destroyed
   one); the first failure returned */
ncclResult_t ncclGroupEnd(void) {
  if (nesting <= 0) return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart");
  if (--nesting) return ncclSuccess;
  struct call *list = calls;
  const size_t n = call_count;
  calls = NULL; call_count = call_capacity = 0;
  ncclResult_t result = ncclSuccess;
  struct said first = {0, ""};
  for (size_t i = 0; i < n; i++) {
    const ncclResult_t made = call_one(list + i);
    if (made && !result) { result = made; first = last; }
    comm_release(list[i].comm);
  }
  free(list);
  if (result) last = first;
  return result;
}

/* whether a call reads sendbuff, or writes recvbuff, on this rank */
static int call_reads(const struct call *c) {
  return (c->what != CALL_BROADCAST && c->what != CALL_SCATTER) || c->comm->rank == c->root;
}
static int call_writes(const struct call *c) {
  return (c->what != CALL_REDUCE && c->what != CALL_GATHER) || c->comm->rank == c->root;
}

/* a call refused before it joins its group (its arguments), else its operator taken and the call queued, holding a
   reference to its communicator until its end (comm_hold), and its group ended (a call outside a group is a group of
   one) */
static ncclResult_t enqueue(struct call c, cudaStream_t stream) {
  const char *name = CALL_NAME[c.what];
  if (!c.comm) return fail(ncclInvalidArgument, "%s: comm is NULL", name);
  const int n = c.comm->nranks;
  if (stream) return fail(ncclInvalidArgument, "%s: a stream: every call is a blocking call on host memory, its stream NULL", name);
  if ((unsigned)c.type >= ncclNumTypes) return fail(ncclInvalidArgument, "%s: datatype %d", name, (int)c.type);
  if (c.count > SIZE_MAX / SIZE[c.type] / (size_t)n) return fail(ncclInvalidArgument, "%s: count %zu past the address space", name, c.count);
  const int reducing = c.what == CALL_ALLREDUCE || c.what == CALL_REDUCE || c.what == CALL_REDUCE_SCATTER;
  if (reducing && ((int)c.op < 0 || (int)c.op >= ncclNumOps)) return fail(ncclInvalidArgument, "%s: operator %d", name, (int)c.op);
  const int rooted = c.what == CALL_BROADCAST || c.what == CALL_REDUCE || c.what == CALL_GATHER || c.what == CALL_SCATTER;
  if (rooted && (c.root < 0 || c.root >= n)) return fail(ncclInvalidArgument, "%s: root %d of %d ranks", name, c.root, n);
  if (c.count && ((call_reads(&c) && !c.send) || (call_writes(&c) && !c.recv)))
    return fail(ncclInvalidArgument, "%s: %s is NULL, and this rank %s it", name, call_reads(&c) && !c.send ? "sendbuff" : "recvbuff",
                call_reads(&c) && !c.send ? "reads" : "writes");
  c.combine = c.op == ncclAvg ? COMBINE_SUM : (int)c.op;
  if (reducing && c.op == ncclAvg && floating(c.type)) { store(c.type, c.pre, 1.0 / n); c.premultiplied = 1; }
  else if (reducing && c.op == ncclAvg) c.post = 1;
  ncclGroupStart();
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) { nesting--; return fail(ncclSystemError, "%s: out of memory", name); }
    calls = grown; call_capacity = capacity;
  }
  calls[call_count++] = c;
  comm_hold(c.comm);
  return ncclGroupEnd();
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_ALLREDUCE, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype, .op = op,
                               .comm = comm}, stream);
}
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_REDUCE, .root = root, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype,
                               .op = op, .comm = comm}, stream);
}
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_BROADCAST, .root = root, .send = sendbuff, .recv = recvbuff, .count = count,
                               .type = datatype, .comm = comm}, stream);
}
ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_REDUCE_SCATTER, .send = sendbuff, .recv = recvbuff, .count = recvcount, .type = datatype,
                               .op = op, .comm = comm}, stream);
}
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_ALLGATHER, .send = sendbuff, .recv = recvbuff, .count = sendcount, .type = datatype,
                               .comm = comm}, stream);
}
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_ALLTOALL, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype,
                               .comm = comm}, stream);
}
ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_GATHER, .root = root, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype,
                               .comm = comm}, stream);
}
ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  return enqueue((struct call){.what = CALL_SCATTER, .root = root, .send = sendbuff, .recv = recvbuff, .count = count, .type = datatype,
                               .comm = comm}, stream);
}

/* -- communicators made of communicators ------------------------------------------------------------------------------ */

/* a communicator of comm's ranks `ranks` (count of them, in that sequence), named after comm, its split's number and
   `color` alike on every member, made at the standing listener of its first rank (its root: the address every rank of
   comm gathered at comm's init), so it needs no exchange over comm */
static ncclResult_t comm_sub(ncclComm_t comm, const int *ranks, int count, uint32_t split, int color, ncclComm_t *made) {
  char name[NCCL_UNIQUE_ID_BYTES + 1];
  int rank = 0;
  for (int i = 0; i < count; i++)
    if (ranks[i] == comm->rank) rank = i;
  snprintf(name, sizeof name, "%s/%u.%d", comm->name, split, color);
  struct ncclComm *sub = comm_new(count, rank, name);
  if (!sub) return fail(ncclSystemError, "out of memory");
  memcpy(sub->id, comm->address[ranks[0]], sizeof comm->address[0]);
  const ncclResult_t result = comm_init(sub, rank ? -1 : comm->standing);
  if (result) comm_release(sub);
  else *made = sub;
  return result;
}

/* NCCL's split: every rank's (color, key) gathered over comm's bootstrap, then the ranks of this rank's color, sorted
   by key and then rank, a communicator of their own (comm_sub; NCCL_SPLIT_NOCOLOR: none, *newcomm NULL), all under a
   reference to comm (comm_hold).  config is not read */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm || (color < 0 && color != NCCL_SPLIT_NOCOLOR)) return fail(ncclInvalidArgument, "ncclCommSplit's arguments");
  *newcomm = NULL;
  comm_hold(comm);
  const int n = comm->nranks;
  struct pick { int32_t color, key; } mine = {color, key}, *every = calloc((size_t)n, sizeof *every);
  int *ranks = calloc((size_t)n, sizeof *ranks), count = 0;
  ncclResult_t result = every && ranks ? exchange(comm, &mine, sizeof mine, every) : fail(ncclSystemError, "out of memory");
  const uint32_t split = comm->splits++;
  for (int r = 0; !result && color != NCCL_SPLIT_NOCOLOR && r < n; r++)
    if (every[r].color == color) ranks[count++] = r;
  for (int a = 1; a < count; a++)
    for (int b = a; b > 0 && every[ranks[b]].key < every[ranks[b - 1]].key; b--) { const int t = ranks[b]; ranks[b] = ranks[b - 1]; ranks[b - 1] = t; }
  if (!result && count) result = comm_sub(comm, ranks, count, split, color, newcomm);
  free(every); free(ranks);
  comm_release(comm);
  return result;
}

/* NCCL's shrink: comm's ranks but the excluded ones, by rank, a communicator of their own (comm_sub, which
   exchanges nothing over comm, so an excluded rank, alive or not, takes no part); a rank among those excluded is in
   none (*newcomm NULL); with NCCL_SHRINK_ABORT, comm revoked first; all under a reference to comm (comm_hold).
   config is not read */
ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t *config,
                            int shrinkFlags) {
  (void)config;
  if (!comm || !newcomm || excludeRanksCount < 0 || (excludeRanksCount && !excludeRanksList))
    return fail(ncclInvalidArgument, "ncclCommShrink's arguments");
  *newcomm = NULL;
  int *ranks = calloc((size_t)comm->nranks, sizeof *ranks), count = 0, member = 0;
  if (!ranks) return fail(ncclSystemError, "out of memory");
  comm_hold(comm);
  for (int r = 0; r < comm->nranks; r++) {
    int excluded = 0;
    for (int e = 0; e < excludeRanksCount; e++) excluded |= excludeRanksList[e] == r;
    if (!excluded) ranks[count++] = r;
    member |= !excluded && r == comm->rank;
  }
  if (shrinkFlags & NCCL_SHRINK_ABORT) comm_turn(comm, COMM_REVOKED, ncclSuccess, NULL);
  const uint32_t split = comm->splits++;
  const ncclResult_t result = member ? comm_sub(comm, ranks, count, split, -1, newcomm) : ncclSuccess;
  free(ranks);
  comm_release(comm);
  return result;
}

/* the id a grow of comm meets at: a listener of comm's root (ncclGetUniqueId) */
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
  return comm ? ncclGetUniqueId(uniqueId) : fail(ncclInvalidArgument, "comm is NULL");
}

/* NCCL's grow: an init at the id of comm's root (ncclCommGetUniqueId), which the root gives comm's other ranks over
   comm's bootstrap under a reference to comm (comm_hold); comm's ranks stay at their ranks (the `rank` they pass,
   NCCL's -1, not read) and a new one (comm NULL) takes `rank`.  config is not read */
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!newcomm || (comm ? nRanks < comm->nranks || (!comm->rank && !uniqueId) : !uniqueId))
    return fail(ncclInvalidArgument, "ncclCommGrow's arguments");
  if (!comm) return ncclCommInitRank(newcomm, nRanks, *uniqueId, rank);
  comm_hold(comm);
  const int me = comm->rank;
  const ncclUniqueId mine = me ? (ncclUniqueId){{0}} : *uniqueId;
  ncclUniqueId *ids = calloc((size_t)comm->nranks, sizeof *ids);
  const ncclResult_t result = ids ? exchange(comm, &mine, sizeof mine, ids) : fail(ncclSystemError, "out of memory");
  const ncclUniqueId given = ids ? ids[0] : mine;
  free(ids);
  comm_release(comm);
  return result ? result : ncclCommInitRank(newcomm, nRanks, given, me);
}

/* libnccl-mesh: NCCL's calls over the node's page table (nccl.h is the contract, rdma/NCCL.md how to run it;
   design/PAGE-TABLE-SPEC-20261122.md section 6).  Every call is blocking and moves host memory: what a rank sends it
   copies into source pages of its own and writes to the indices its peers hold for it (mesh-call.h mesh_write), and
   what lands at its own indices it copies out, or combines in rank order, into its buffers, as it landed, then
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
#include <os/os_sync_wait_on_address.h>

enum { CALL_ALLREDUCE, CALL_BROADCAST, CALL_REDUCE, CALL_REDUCE_SCATTER, CALL_ALLGATHER, CALL_ALLTOALL, CALL_GATHER,
       CALL_SCATTER };
static const char *const CALL_NAME[] = {"ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclReduceScatter",
                                        "ncclAllGather", "ncclAlltoAll", "ncclGather", "ncclScatter"};
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN };
/* A communicator's pages in SHARES shares, every rank cutting them alike: page j of a message, and a reduction's
   position j, are share j % SHARES's.  Each share has indices of its own on each rank's node: RECEIVES receive
   indices for each source rank, of which each pair uses the communicator's window W (at most RECEIVES:
   comm_window), the share's k-th page from rank s to rank d landing at d's base of the share + s RECEIVES + k mod W,
   that index's (k / W + 1)-th landing; and SOURCES source pages of each class, which what the share sends is copied
   into.  A rank moves its shares on its threads (the process's attach's, at most SHARES), share s on thread s % its
   threads: the layout is the same at every rank, whatever its threads */
#define SHARES 8u
#define RECEIVES 512u
#define SOURCES 128u
/* a thread with no call to move looks for one for WORKER_SPIN_NS, then sleeps until one comes */
#define WORKER_SPIN_NS 2000000u

static const size_t SIZE[ncclNumTypes] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
/* a failure's message: its text, and `id`, where the message is a communicator's (comm_fail), the count of messages the
   process had recorded when it was (`messages`), else 0 */
struct said { uint64_t id; char text[512]; };
static _Atomic uint64_t messages;
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

/* `result` beside the communicator's message `said`: `last` its copy, copied only where `last` holds another (its id
   the match), so a failure read again formats and copies nothing */
static ncclResult_t told(ncclResult_t result, const struct said *said) {
  if (last.id != said->id) last = *said;
  return result;
}

/* a communicator's message's identity, a new one */
static uint64_t message_id(void) { return atomic_fetch_add_explicit(&messages, 1, memory_order_relaxed) + 1; }

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
  /* float32 and float16: the arithmetic below ((float)a op (float)b, the load into double exact) as typed loops the
     compiler vectorizes, in place of one load, combine and store through double per element */
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

/* FuncSumPostDiv::divide (src/device/reduce_kernel.h, an integer average): the wrapped sum's magnitude over n, its sign
   kept */
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

/* -- the process's attach ---------------------------------------------------------------------------------------------
   The process is a client of its node's bridge in one slot a thread: one attach of the node's region (mesh-call.h
   mesh_attach), made at its first communicator, and as many more slots of the same mapping as it has threads
   (`contexts`, threads_wanted: mesh.h mesh_slot_take, each a write ring of its own), whose slots own every
   communicator's indices and pages.  A region a stop released, or a bridge of another layout replaced, is expired (its
   layout no longer MESH_LAYOUT): while the process holds a communicator of it, a new communicator fails naming the
   expiry, and once it holds none the attach is let go (mesh_detach of each slot: its indices freed and their pages
   dropped), so the next communicator attaches the bridge's next region and no process holds two.  A write ring has
   one producer (mesh-call.h mesh_write): each slot's writes, of all the process's communicators and threads, are made
   one at a time (`writing`). */
struct attach { struct mesh_ctx context[SHARES]; pthread_mutex_t writing[SHARES]; uint32_t contexts; uint64_t comms; };
static struct attach *attached;
static pthread_mutex_t attach_lock = PTHREAD_MUTEX_INITIALIZER;

static int attach_expired(const struct attach *a) {
  return __atomic_load_n(&a->context[0].M->version, __ATOMIC_ACQUIRE) != MESH_LAYOUT;
}
/* the attach let go where its region is expired and it holds no communicator of it; under attach_lock */
static void attach_settle(void) {
  struct attach *a = attached;
  if (!a || a->comms || !attach_expired(a)) return;
  for (uint32_t w = a->contexts; w-- > 0;) mesh_detach(&a->context[w]);
  for (uint32_t w = 0; w < SHARES; w++) pthread_mutex_destroy(&a->writing[w]);
  free(a);
  attached = NULL;
}
/* the threads a process moves a communicator's shares on: NCCL_MESH_THREADS where set (1 to SHARES), else
   THREADS_DEFAULT, at most half the node's top-level cores (its bridge's port children spin on the others).
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
  return n < 1 ? 1 : n > (long)SHARES ? SHARES : (uint32_t)n;
}
/* the process's attach, made at its first use and again after an expired one is let go: its region attached, then a
   slot of the same mapping for each further thread wanted while one is vacant (fewer threads where none is); 0, ESTALE
   where the process holds communicators of an expired region, ENOMEM, or mesh_attach's error; under attach_lock */
static int attach_current(struct attach **out) {
  attach_settle();
  if (attached) { *out = attached; return attach_expired(attached) ? ESTALE : 0; }
  struct attach *a = calloc(1, sizeof *a);
  if (!a) return ENOMEM;
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
  for (uint32_t w = 0; w < SHARES; w++) pthread_mutex_init(&a->writing[w], NULL);
  *out = attached = a;
  return 0;
}
/* communicator `name`'s attach not made (attach_current's error) */
static ncclResult_t attach_fail(int status, const char *name) {
  if (status == ENOMEM) return fail(ncclSystemError, "communicator %s: out of memory", name);
  if (status == EBUSY) return fail(ncclSystemError, "communicator %s: every client slot of this node's bridge is held (mesh.h MESH_CLIENTS)", name);
  if (status == ESTALE)
    return fail(ncclRemoteError, "communicator %s: the region of this node's bridge was released or replaced while this process holds "
                "communicators of it; once it has destroyed them (ncclCommDestroy), its next communicator attaches the bridge's "
                "next region", name);
  return fail(ncclSystemError, "communicator %s: this process attaches no region of this node's bridge (%s); the node's bridge "
              "must run (etc/bridge/README.md)", name, strerror(status));
}

/* -- communicators ------------------------------------------------------------------------------------------------- */

/* A share of a communicator at this rank: the slot its indices are of (`context`, the attach's, and its writes' lock
   `writing`), its receive indices (`receives`, its RECEIVES for each rank) and its source pages (`source[c]` the
   first of its SOURCES of class c, 0 short and 1 long, and `next[c]` the one it looks at first); a cache line each,
   since each is its own thread's */
struct share { _Alignas(128) struct mesh_ctx *context; pthread_mutex_t *writing; uint32_t receives, source[2], next[2]; };
struct plan;
/* a thread of a communicator's past its first (the calling thread): its number */
struct self { struct ncclComm *comm; uint32_t t; };

/* A communicator: its ranks; each rank's node (`node`), the first receive index of each of its shares (`base`, rank r
   share s at r SHARES + s) and its standing listener's address (`address`, where a split's or a shrink's communicator
   whose first rank it is meets), gathered at its init; the pages each share of this rank has written to each rank
   (`sent`) and taken from each (`received`, share s rank r at s nranks + r); its shares here (`share`); the indices each
   pair's pages of a share use (`window`, comm_window); the splits and shrinks made of it (`splits`, which name them);
   its bootstrap's sockets (`peers`: the root's to every rank, a rank's to the root) and its standing listener; the
   scratch a reduction's own contribution is made in; its first failure (`failed`, `failure`), its revoke, and the
   calls and exchanges under way on it (`active`).  Its threads (`threads`, the first the calling thread's, the others
   `thread`): a call's plan (`plan`) is told them by `job`, the count of calls told in its high bits and the threads
   the call is moved on in its low 8, the threads asleep counted (`asleep`), and the told threads still moving it
   counted (`running`); `stopping` ends them */
struct ncclComm {
  int nranks, rank;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  uint32_t *node, *base;
  char (*address)[NCCL_UNIQUE_ID_BYTES];
  uint64_t *sent, *received;
  struct share share[SHARES];
  uint32_t window, splits, threads;
  struct attach *attach;
  int *peers, standing;
  unsigned char *scratch;
  size_t scratch_bytes;
  ncclResult_t failed;
  struct said failure;
  pthread_mutex_t guard, exchanging;
  int revoked;
  _Atomic int active;
  pthread_t thread[SHARES];
  struct self self[SHARES];
  struct plan *plan;
  _Atomic uint64_t job;
  _Atomic uint32_t asleep, running;
  _Atomic int stopping;
};

/* a communicator of `nranks` ranks named `name`, this process rank `rank`, holding nothing yet; NULL where out of
   memory */
static struct ncclComm *comm_new(int nranks, int rank, const char *name) {
  struct ncclComm *comm = NULL;
  if (posix_memalign((void **)&comm, _Alignof(struct ncclComm), sizeof *comm)) return NULL;
  memset(comm, 0, sizeof *comm);
  const size_t n = (size_t)nranks;
  comm->node = calloc(n, sizeof *comm->node); comm->base = calloc(n * SHARES, sizeof *comm->base);
  comm->address = calloc(n, sizeof *comm->address);
  comm->sent = calloc(n * SHARES, sizeof *comm->sent); comm->received = calloc(n * SHARES, sizeof *comm->received);
  comm->peers = malloc(n * sizeof *comm->peers);
  if (!comm->node || !comm->base || !comm->address || !comm->sent || !comm->received || !comm->peers) {
    free(comm->node); free(comm->base); free(comm->address); free(comm->sent); free(comm->received); free(comm->peers);
    free(comm);
    return NULL;
  }
  for (size_t r = 0; r < n; r++) comm->peers[r] = -1;
  comm->nranks = nranks; comm->rank = rank; comm->standing = -1; comm->threads = 1;
  for (uint32_t s = 0; s < SHARES; s++)
    comm->share[s] = (struct share){NULL, NULL, MESH_ABSENT, {MESH_ABSENT, MESH_ABSENT}, {0, 0}};
  pthread_mutex_init(&comm->guard, NULL);
  pthread_mutex_init(&comm->exchanging, NULL);
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

/* comm's indices of this node's table (none for one rank, which sends nothing), each share's of the slot of the thread
   that moves it (share s, the attach's context s % its contexts): RECEIVES receive indices for each rank, naming no
   page and landed 0 until one lands, and SOURCES source pages of each class; 0, or mesh_pages' errno */
static int comm_reserve(struct ncclComm *comm) {
  if (comm->nranks == 1) return 0;
  struct attach *a = comm->attach;
  const uint64_t receives = (uint64_t)comm->nranks * RECEIVES;
  int status = 0;
  pthread_mutex_lock(&attach_lock);
  for (uint32_t s = 0; !status && s < SHARES; s++) {
    struct share *x = comm->share + s;
    x->context = &a->context[s % a->contexts];
    x->writing = &a->writing[s % a->contexts];
    if ((x->receives = mesh_pages(x->context, (uint32_t)receives, 0)) == MESH_ABSENT) status = errno;
    else if ((x->source[0] = mesh_pages(x->context, SOURCES, MESH_SHORT_PAYLOAD)) == MESH_ABSENT) status = errno;
    else if ((x->source[1] = mesh_pages(x->context, SOURCES, MESH_LONG_PAYLOAD)) == MESH_ABSENT) status = errno;
  }
  pthread_mutex_unlock(&attach_lock);
  return status;
}

/* comm's indices freed (their pages dropped: a page a SEND still reads goes back to the pool at its completion) and its
   hold of the process's attach let go */
static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  struct attach *a = comm->attach;
  if (a) {
    for (uint32_t s = 0; s < SHARES; s++) {
      struct share *x = comm->share + s;
      if (x->receives != MESH_ABSENT) mesh_invalidate(x->context, x->receives, (uint32_t)comm->nranks * RECEIVES, 1);
      for (int c = 0; c < 2; c++)
        if (x->source[c] != MESH_ABSENT) mesh_invalidate(x->context, x->source[c], SOURCES, 1);
      x->receives = x->source[0] = x->source[1] = MESH_ABSENT;
    }
    a->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
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

static void comm_threads_stop(struct ncclComm *comm);
static void comm_free(struct ncclComm *comm) {
  comm_threads_stop(comm);
  bootstrap_close(comm);
  if (comm->standing >= 0) close(comm->standing);
  comm_leave(comm);
  pthread_mutex_destroy(&comm->guard);
  pthread_mutex_destroy(&comm->exchanging);
  free(comm->node); free(comm->base); free(comm->address); free(comm->sent); free(comm->received); free(comm->scratch);
  free(comm);
}

/* comm failed for good: its first failure's result and message recorded (comm_failure reads them), the message
   formatted here once, and named (ncclGetLastError) */
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) __attribute__((format(printf, 3, 4)));
static ncclResult_t comm_fail(struct ncclComm *comm, ncclResult_t result, const char *format, ...) {
  pthread_mutex_lock(&comm->guard);
  if (!comm->failed) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(comm->failure.text, sizeof comm->failure.text, format, arguments);
    va_end(arguments);
    comm->failure.id = message_id();
    comm->failed = result;
  }
  const struct said said = comm->failure;
  const ncclResult_t failed = comm->failed;
  pthread_mutex_unlock(&comm->guard);
  return told(failed, &said);
}

/* comm's failure, the one it recorded (comm_fail), named again (ncclGetLastError); ncclSuccess where none */
static ncclResult_t comm_failure(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  const ncclResult_t failed = comm->failed;
  const struct said said = comm->failure;
  pthread_mutex_unlock(&comm->guard);
  return failed ? told(failed, &said) : ncclSuccess;
}

/* -- NCCL's bootstrap (NVIDIA/nccl src/bootstrap.cc) --------------------------------------------------------------------
   The unique id is the root's address, host:port (NCCL_COMM_ID where set, else a listener this process opens on the
   node's fabric address, else on its host name), and the root listens on that host alone where it is an IPv6 address
   (the fabric's ULA), else on every address; rank 0 takes every rank's report there and sends each the whole table, on
   sockets the communicator keeps for its exchanges (a split's, a grow's) and closes where its init fails.  A
   non-root's connect does not block: it is looked at every BOOTSTRAP_SLICE_MS and tried again a slice after it fails;
   the root looks for a connection every slice; each wait reads the communicator's revoke (ncclCommRevoke,
   ncclCommDestroy) at each look, and the revoke shuts the sockets down, so an exchange under way ends.  Nothing else
   bounds a wait of the bootstrap. */
static struct { int fd; char id[NCCL_UNIQUE_ID_BYTES]; } root = {-1, {0}};
#define BOOTSTRAP_SLICE_MS 100

/* a rank's report at a communicator's init: its status (0, else why it cannot go on, so every rank fails alike), its
   bridge's node, its node's pool (`pages`: the fewer pages of its two classes, mesh.h struct mesh_pool), the first
   receive index of each of its shares and its standing listener's address */
struct report { int32_t rank, status; uint32_t node, pages, base[SHARES]; char address[NCCL_UNIQUE_ID_BYTES]; };

/* the communicator's window, alike at every rank (each computes it from the same reports): the pages each pair of a
   share numbers before it comes back to an index, at most RECEIVES, and the most for which, at every node v of the
   communicator, its c_v ranks' source pages and a page at each index of their shares' windows, c_v SHARES (SOURCES +
   (nranks - 1) W) pages of either class, fit in half the node's pool (the other half its bridge's lanes' receives,
   its relays' pages and its other clients'); 0 where no window fits.  A communicator of one rank writes to no pair:
   RECEIVES */
static uint32_t comm_window(const struct report *reports, int nranks) {
  uint32_t window = RECEIVES;
  for (int r = 0; nranks > 1 && r < nranks; r++) {
    uint64_t here = 0;
    for (int q = 0; q < nranks; q++) here += reports[q].node == reports[r].node;
    const uint64_t room = reports[r].pages / 2, held = here * SHARES * SOURCES;
    const uint64_t most = room > held ? (room - held) / (here * SHARES * (uint64_t)(nranks - 1)) : 0;
    if (most < window) window = (uint32_t)most;
  }
  return window;
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

/* whether comm was revoked */
static int revoked(ncclComm_t comm) { return __atomic_load_n(&comm->revoked, __ATOMIC_SEQ_CST); }
static ncclResult_t revoked_fail(ncclComm_t comm) {
  return fail(ncclInvalidUsage, "its communicator %s was revoked (ncclCommRevoke): no call runs on it before ncclCommDestroy", comm->name);
}

/* `size` bytes sent or received whole on comm's bootstrap socket `fd`, its waits in slices (the socket's SO_RCVTIMEO
   and SO_SNDTIMEO) that read the revoke: 0, ECANCELED (revoked), ECONNRESET (the peer closed), or the errno */
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

/* socket `fd` of comm's init placed at peers[r] (-1: the one there closed), blocking in slices (an accepted one takes
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
   root's to each rank r (peers[r], placed by the rank its first report names, that report kept in `reports`; a socket
   not yet placed waits at peers[0]; one whose report fails or names no rank still to meet is closed alone, the root
   accepting the next).  The root accepts on `standing` where it is a listener (a parent's standing listener, kept
   open), else on the listener ncclGetUniqueId opened for `id`, else on one it opens at id's port, closing either once
   every rank has met: 0, ECANCELED (comm revoked), or the errno */
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
    /* a connection whose report does not come, or names no rank still to meet, is closed alone */
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

/* the init's gather on comm's bootstrap sockets: each other rank's report to the root (`sent`: there already, as the
   root's bootstrap_open read each), then the whole table to each; 0, or the errno (whole) */
static int gather(struct ncclComm *comm, struct report *reports, int sent) {
  const int *peers = comm->peers, nranks = comm->nranks, rank = comm->rank;
  const size_t table = (size_t)nranks * sizeof *reports;
  int status = 0;
  if (rank) {
    const struct report mine = reports[rank];
    if (!sent) status = whole(comm, peers[0], reports + rank, sizeof *reports, 1);
    if (!status) status = whole(comm, peers[0], reports, table, 0);
    reports[rank] = mine;
    return status;
  }
  for (int r = 1; r < nranks && !status && !sent; r++)
    if (!(status = whole(comm, peers[r], reports + r, sizeof *reports, 0)) && reports[r].rank != r) status = EPROTO;
  for (int r = 1; r < nranks && !status; r++) status = whole(comm, peers[r], reports, table, 1);
  return status;
}

/* `bytes` of every rank's gathered over comm's bootstrap (rank r's at reports + r bytes, this rank's `local`): the
   configuration a split or a grow exchanges before the communicator it makes exists */
static ncclResult_t exchange(struct ncclComm *comm, const void *local, size_t bytes, void *reports) {
  pthread_mutex_lock(&comm->exchanging);
  atomic_fetch_add(&comm->active, 1);
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
  atomic_fetch_sub(&comm->active, 1);
  pthread_mutex_unlock(&comm->exchanging);
  return error ? fail(ncclRemoteError, "communicator %s: an exchange over its bootstrap: %s", comm->name, strerror(error)) : ncclSuccess;
}

/* A communicator's init at its id, the one path of every init (ncclCommInitRank, a split's, a shrink's, a grow's): the
   process attached (comm_attach), the communicator's indices taken (comm_reserve) and its standing listener opened, its
   bootstrap's sockets opened (bootstrap_open, the root accepting on `standing` where it is a listener), then every
   rank's report gathered: each rank's node, base, pool and standing listener, the only exchange its calls need, from
   which every rank computes the same window (comm_window).  A rank that cannot go on, or a window of 0, fails every
   rank alike, made holding nothing of its bridge's and its sockets closed; a revoke fails it as a value.  A
   communicator made starts its threads (comm_threads_start) */
static void comm_threads_start(struct ncclComm *comm);
static ncclResult_t comm_init(struct ncclComm *made, int standing) {
  const int nranks = made->nranks, rank = made->rank;
  struct report *reports = calloc((size_t)nranks, sizeof *reports);
  if (!reports) return fail(ncclSystemError, "communicator %s: out of memory", made->name);
  struct report *mine = reports + rank;
  const int attaching = comm_attach(made);
  int status = attaching ? attaching : comm_reserve(made);
  if (!status && nranks > 1 && (made->standing = listener_open(mine->address, sizeof mine->address)) < 0) status = errno;
  mine->rank = rank; mine->status = status;
  for (uint32_t s = 0; s < SHARES; s++) mine->base[s] = made->share[s].receives;
  if (!attaching) {
    struct hdr *m = made->attach->context[0].M;
    const struct mesh_pool *pool = mesh_pool(m);
    mine->node = m->node;
    mine->pages = pool->cls[0].pages < pool->cls[1].pages ? pool->cls[0].pages : pool->cls[1].pages;
  }
  int met = nranks > 1 ? bootstrap_open(made, made->id, reports, standing) : 0;
  if (!met && nranks > 1) met = gather(made, reports, !rank);
  ncclResult_t result = ncclSuccess;
  if (attaching) result = attach_fail(attaching, made->name);
  else if (status)
    result = fail(ncclSystemError, "communicator %s: its %u receive indices and %u source pages of each class of this node's bridge, "
                  "or its listener: %s", made->name, (uint32_t)nranks * RECEIVES * SHARES, SOURCES * SHARES, strerror(status));
  else if (met == ECANCELED || revoked(made)) result = revoked_fail(made);
  else if (met) result = fail(ncclRemoteError, "communicator %s: the bootstrap at %s: %s", made->name, made->id, strerror(met));
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status)
      result = fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  made->window = !result && nranks > 1 ? comm_window(reports, nranks) : RECEIVES;
  if (!made->window)
    result = fail(ncclSystemError, "communicator %s: half the pool of a node of its ranks holds their source pages and no "
                  "receive page for each pair: %s", made->name, strerror(ENOSPC));
  for (int r = 0; !result && r < nranks; r++) {
    made->node[r] = reports[r].node;
    for (uint32_t s = 0; s < SHARES; s++) made->base[(size_t)r * SHARES + s] = reports[r].base[s];
    memcpy(made->address[r], reports[r].address, sizeof made->address[r]);
    made->address[r][sizeof made->address[r] - 1] = 0;
  }
  free(reports);
  if (result) { bootstrap_close(made); comm_leave(made); }
  else comm_threads_start(made);
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
  if (result) comm_free(made);
  else *comm = made;
  return result;
}

/* comm stopped for good (a revoke's and a destroy's): `revoked` set, failing every call after it and ending each wait
   of one under way at its next look, and every socket of its bootstrap shut down, so an exchange under way ends; under
   its guard */
static void comm_stop(struct ncclComm *comm) {
  pthread_mutex_lock(&comm->guard);
  __atomic_store_n(&comm->revoked, 1, __ATOMIC_SEQ_CST);
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) shutdown(comm->peers[r], SHUT_RDWR);
  pthread_mutex_unlock(&comm->guard);
}

/* NCCL's revoke: comm's calls under way end as a value at their next look and every later one is refused
   (ncclInvalidUsage), its indices and pages kept until ncclCommDestroy; ncclCommGetAsyncError reads ncclSuccess after
   it; from any thread.  revokeFlags is not read */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  (void)revokeFlags;
  if (!comm) return fail(ncclInvalidArgument, "comm is NULL");
  comm_stop(comm);
  return ncclSuccess;
}

/* comm stopped (comm_stop), the calls and exchanges under way on it returned (each at its next look), then its indices
   freed and their pages dropped, its sockets closed and comm freed */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!comm) return ncclSuccess;
  comm_stop(comm);
  while (atomic_load(&comm->active)) usleep(20);
  comm_free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

/* ncclSuccess after a revoke; else comm's own failure (comm_failure: a call's), ncclSuccess where none */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (!comm || !asyncError) return fail(ncclInvalidArgument, "comm or asyncError is NULL");
  *asyncError = revoked(comm) ? ncclSuccess : comm_failure(comm);
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
   of them, so a buffer is any memory */
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

/* a call as its group holds it: its kind, root, buffers and elements, its type, and its operator as the host runs it
   (`combine`; a floating average's premultiplier 1 / nranks, `pre`; an integer average's divide by nranks, `post`) */
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
static _Thread_local int depth;
static _Thread_local struct call *calls;
static _Thread_local size_t call_count, call_capacity;

/* A phase of a call as this rank moves it: the messages it sends (`out`: `bytes` at `data`, premultiplied where `pre`,
   to rank `to`, or to every other rank where `to` is -1) and those it receives (`in`: `bytes` from rank `from`,
   copied to `to`), and, where `acc` is set, its reduction: `bytes` at `acc` made the combine, in rank order, of every
   rank's, this rank's at `own` and every other's a message to it.  A pair of ranks carries at most one message in a
   phase, so a message's first page is its pair's count of pages as the phase begins */
struct out { const unsigned char *data; size_t bytes; int to, pre; };
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

/* A share's moves of a phase under way, toward and from each rank r, every count in the share's own pages (its q-th
   page of a message the message's page s + q SHARES, s the share): its pair's first page of the share in the phase
   toward it (`first_out[r]`) and from it (`first_in[r]`), the share's pages due from it (`due[r]`: those of its
   in-message, `from[r]`, or the reduction's positions), the first of them not yet taken (`low[r]`) and each one's state
   (`taken`, from `at[r]`: WAITING, TAKEN, or PARKED in the parked scratch until its turn); the page it sends next
   (out-message `o`, its page `j`, its destinations from the g-th of them) and the source page that page's payload was
   copied into and not yet written (`copied`, MESH_ABSENT none); each of the share's reduction positions' next rank to
   combine (`next`) and the scratch a page that lands before its rank's turn is copied into (`parked`, NULL until the
   first); the in-message pages still to take and the reduction's contributions still to combine (`left`) */
enum { WAITING, TAKEN, PARKED };
struct moving {
  struct call *call;
  const struct phase *p;
  struct share *share;
  uint32_t s;
  uint64_t *first_out, *first_in, *due, *low, *at;
  const struct in **from;
  uint32_t o, g, copied;
  uint64_t j;
  size_t copied_bytes;
  unsigned char *taken, *parked;
  uint64_t positions, left;
  uint32_t *next;
};

/* the pages of a message of `pages` pages that are share s's */
static uint64_t share_pages(uint64_t pages, uint32_t s) { return pages > s ? (pages - s + SHARES - 1) / SHARES : 0; }

/* a source page of share x of class `cls` free to copy into: its uses 1, its word's alone (no write, SEND or landing
   of it under way: the use-checker, design/PAGE-TABLE-SPEC-20261122.md section 6), looked for from the class's `next`;
   MESH_ABSENT where none is */
static uint32_t source_free(struct share *x, int cls) {
  for (uint32_t i = 0; i < SOURCES; i++) {
    const uint32_t at = (x->next[cls] + i) % SOURCES;
    struct mesh_view_entry entry;
    if (mesh_entry(x->context, x->source[cls] + at, &entry) || entry.uses != 1) continue;
    x->next[cls] = (at + 1) % SOURCES;
    /* the bridge's last drop of a use (release) before this rank's stores into the page */
    atomic_thread_fence(memory_order_acquire);
    return x->source[cls] + at;
  }
  return MESH_ABSENT;
}

/* the g-th of the ranks other than `me`, in rank order */
static int other(int me, uint32_t g) { return (int)g + ((int)g >= me); }

/* the share's pages of the phase sent in order (each message's pages of the share; each page to its destinations,
   one write to each MESH_DESTS of them in rank order, every write from a source page of the share's own, each
   destination the index its own pair's page number for the page names, in the share's indices there), while a source
   page is free and the share's write ring takes them: 0 (the rest wait for a later pass), or -1 with the call's
   failure in *result */
static int send_some(struct moving *m, ncclResult_t *result) {
  struct call *c = m->call;
  struct ncclComm *comm = c->comm;
  struct share *share = m->share;
  const struct phase *p = m->p;
  const int me = comm->rank;
  for (; m->o < p->outs; m->o++, m->j = m->s) {
    const struct out *x = p->out + m->o;
    const uint64_t pages = pages_of(x->bytes);
    const uint32_t dests = x->to < 0 ? (uint32_t)comm->nranks - 1 : 1;
    for (; m->j < pages; m->j += SHARES, m->g = 0)
      while (m->g < dests) {
        if (m->copied == MESH_ABSENT) {
          const uint32_t a = source_free(share, x->bytes > MESH_SHORT_PAYLOAD);
          if (a == MESH_ABSENT) return 0;
          size_t offset;
          const size_t bytes = page_at(x->bytes, m->j, &offset);
          unsigned char *data = mesh_page_data(share->context, a);
          if (x->pre) premultiply(c->type, data, x->data + offset, c->pre, bytes / SIZE[c->type]);
          else memcpy(data, x->data + offset, bytes);
          m->copied = a; m->copied_bytes = bytes;
        }
        struct mesh_dest dest[MESH_DESTS];
        const uint32_t n = dests - m->g < MESH_DESTS ? dests - m->g : MESH_DESTS;
        for (uint32_t i = 0; i < n; i++) {
          const int d = x->to < 0 ? other(me, m->g + i) : x->to;
          const uint64_t k = m->first_out[d] + m->j / SHARES;
          dest[i] = (struct mesh_dest){comm->node[d], comm->base[(size_t)d * SHARES + m->s] + (uint32_t)me * RECEIVES +
                                                      (uint32_t)(k % comm->window)};
        }
        pthread_mutex_lock(share->writing);
        const int status = mesh_write(share->context, m->copied, (uint32_t)m->copied_bytes, dest, n);
        pthread_mutex_unlock(share->writing);
        if (status == EAGAIN) return 0;
        if (status) {
          *result = comm_fail(comm, ncclInternalError, "communicator %s: %s's write of its source page %u: %s", comm->name,
                              CALL_NAME[c->what], m->copied, strerror(status));
          return -1;
        }
        m->copied = MESH_ABSENT;
        m->g += n;
      }
  }
  return 0;
}

/* the parked scratch's part for rank r's reduction pages: one p->bytes a rank, for each rank but rank 0 (whose page is
   always in its turn) and this one (a share touches only its own pages of it) */
static unsigned char *parked_at(const struct moving *m, int r) {
  const int me = m->call->comm->rank;
  return m->parked + (size_t)(r - 1 - (me && me < r)) * m->p->bytes;
}

/* the share's reduction position q combined on in rank order while the next rank's contribution is here: this rank's
   own, or a page parked before its turn (rank 0's copied, each later one's combined); a page taken before its turn
   from an index that named none adds nothing */
static void advance(struct moving *m, uint64_t q) {
  struct call *c = m->call;
  const struct phase *p = m->p;
  const int n = c->comm->nranks, me = c->comm->rank;
  size_t offset;
  const size_t bytes = page_at(p->bytes, m->s + q * SHARES, &offset);
  for (int r; (r = (int)m->next[q]) < n; m->next[q]++, m->left--) {
    const unsigned char *from = p->own + offset;
    if (r != me) {
      const int state = m->taken[m->at[r] + q];
      if (state == WAITING) return;
      if (state == TAKEN) continue;
      from = parked_at(m, r) + offset;
    }
    if (!r) memcpy(p->acc + offset, from, bytes);
    else combine_into(c->type, c->combine, p->acc + offset, from, bytes / SIZE[c->type]);
  }
}

/* the share's page q of the phase's pages from rank r taken once its index has counted its landing (the pair's page k
   of the share is the (k / W + 1)-th to land at its index, W the communicator's window), whatever landed.  Where the
   index names a page, the bytes due are copied from it as they are (a page of either class is followed by a long
   page's frames of the region, mesh.h): an in-message's to its place in the buffer
   (design/DEFRAGMENT-pseudocode-20261122.md), a reduction page's combined straight into the result where its rank's
   turn at its position has come (rank 0's copied; then the position's later ranks already here after it, advance),
   else copied into the share's parked scratch (allocated at the first page parked, sized from the phase); then the
   index is invalidated.  Where it names none (its page was taken under an earlier count), nothing is copied.  0
   (taken, or not landed yet), or -1 with the call's failure in *result */
static int take(struct moving *m, int r, uint64_t q, ncclResult_t *result) {
  struct call *c = m->call;
  struct ncclComm *comm = c->comm;
  struct share *share = m->share;
  const struct phase *p = m->p;
  const struct in *x = m->from[r];
  const uint64_t k = m->first_in[r] + q;
  const uint32_t index = share->receives + (uint32_t)r * RECEIVES + (uint32_t)(k % comm->window);
  struct mesh_view_entry entry;
  if (mesh_entry(share->context, index, &entry)) {
    *result = comm_fail(comm, ncclInternalError, "communicator %s: its index %u could not be read", comm->name, index);
    return -1;
  }
  if (entry.landed < k / comm->window + 1) return 0;
  size_t offset;
  const size_t bytes = page_at(x ? x->bytes : p->bytes, m->s + q * SHARES, &offset);
  const int turn = !x && m->next[q] == (uint32_t)r;
  unsigned char state = TAKEN;
  if (entry.data) {
    if (x) memcpy(x->to + offset, entry.data, bytes);
    else if (turn && !r) memcpy(p->acc + offset, entry.data, bytes);
    else if (turn) combine_into(c->type, c->combine, p->acc + offset, entry.data, bytes / SIZE[c->type]);
    else {
      if (!m->parked && !(m->parked = malloc((size_t)(comm->nranks - 1 - (comm->rank != 0)) * p->bytes))) {
        *result = comm_fail(comm, ncclSystemError, "communicator %s: out of memory for the pages that land before their turn", comm->name);
        return -1;
      }
      memcpy(parked_at(m, r) + offset, entry.data, bytes);
      state = PARKED;
    }
    mesh_invalidate(share->context, index, 1, 0);
  }
  m->taken[m->at[r] + q] = state;
  if (x || turn) m->left--;
  if (turn) { m->next[q]++; advance(m, q); }
  return 0;
}

/* the share's landed pages of the phase taken (take), each in the pass that finds it, every page independent of the
   others: from each rank, its pages from the first not yet taken through the communicator's window of them, each at
   an index of its own (a page further on lands at the index of one of these, and is looked for once that one is
   taken, so no index is read for a later page while an earlier page there is due): 0, or -1 with the call's failure
   in *result */
static int take_some(struct moving *m, ncclResult_t *result) {
  const int n = m->call->comm->nranks;
  const uint32_t window = m->call->comm->window;
  for (int r = 0; r < n; r++) {
    const unsigned char *taken = m->taken + m->at[r];
    while (m->low[r] < m->due[r] && taken[m->low[r]]) m->low[r]++;
    const uint64_t end = m->due[r] - m->low[r] > window ? m->low[r] + window : m->due[r];
    for (uint64_t q = m->low[r]; q < end; q++)
      if (!taken[q] && take(m, r, q, result)) return -1;
  }
  return 0;
}

static void moving_close(struct moving *m) {
  free(m->first_out); free(m->from); free(m->next); free(m->taken); free(m->parked);
  m->first_out = NULL; m->from = NULL; m->next = NULL; m->taken = NULL; m->parked = NULL;
}
/* share s's moves of phase p begun: each pair's pages of the share numbered on from its count (comm's `sent` to each
   rank, `received` from each, of the share, the same at both ends, since every rank makes the same calls in the same
   order and cuts each message alike), and this rank's own contribution combined where it is rank 0; ncclSuccess, or
   the failure (out of memory) */
static ncclResult_t moving_open(struct moving *m, struct call *c, const struct phase *p, uint32_t s) {
  struct ncclComm *comm = c->comm;
  const int n = comm->nranks, me = comm->rank;
  *m = (struct moving){.call = c, .p = p, .share = comm->share + s, .s = s, .copied = MESH_ABSENT, .j = s,
                       .positions = p->acc ? share_pages(pages_of(p->bytes), s) : 0};
  m->first_out = calloc(5 * (size_t)n, sizeof *m->first_out);
  m->from = calloc((size_t)n, sizeof *m->from);
  m->next = calloc((size_t)m->positions + 1, sizeof *m->next);
  uint64_t pages = 0;
  if (m->first_out && m->from) {
    m->first_in = m->first_out + n; m->due = m->first_in + n; m->low = m->due + n; m->at = m->low + n;
    for (uint32_t i = 0; i < p->ins; i++) {
      m->from[p->in[i].from] = p->in + i;
      m->due[p->in[i].from] = share_pages(pages_of(p->in[i].bytes), s);
    }
    for (int r = 0; p->acc && r < n; r++)
      if (r != me) m->due[r] = m->positions;
    for (int r = 0; r < n; r++) { m->at[r] = pages; pages += m->due[r]; }
  }
  m->taken = calloc((size_t)pages + 1, 1);
  if (!m->first_out || !m->from || !m->next || !m->taken) {
    moving_close(m);
    return fail(ncclSystemError, "communicator %s: out of memory", comm->name);
  }
  m->left = pages + m->positions;
  uint64_t *sent = comm->sent + (size_t)s * n, *received = comm->received + (size_t)s * n;
  for (int r = 0; r < n; r++) { m->first_out[r] = sent[r]; m->first_in[r] = received[r]; received[r] += m->due[r]; }
  for (uint32_t o = 0; o < p->outs; o++)
    for (int d = 0; d < n; d++)
      if (d != me && (p->out[o].to < 0 || p->out[o].to == d)) sent[d] += share_pages(pages_of(p->out[o].bytes), s);
  for (uint64_t q = 0; q < m->positions; q++) advance(m, q);
  return ncclSuccess;
}

/* A call's moves (call_move), told every thread that moves a share of it: its phases (a second for an all-reduce of
   more than one page: the all-gather of the reduced segments); this rank's one copy of its own part where it lies
   apart from where it goes (`local`), and the reduction's own contribution where comm's scratch makes it (`fill`, the
   bytes it is copied, or premultiplied, from), each made by the shares, page by page; the shares with pages (`active`:
   shares 0 to active - 1) and the threads that move them (`threads`, thread t shares t, t + threads, ...); each
   thread's result and its message, and `stop`, which a thread's failure sets, so the others end too */
struct local { const unsigned char *from; unsigned char *to; size_t bytes; };
struct plan {
  struct call *call;
  struct phase phase[2];
  uint32_t phases, active, threads;
  struct local local;
  const unsigned char *fill;
  _Atomic int stop;
  ncclResult_t result[SHARES];
  struct said said[SHARES];
};

/* share s's pages of the reduction's own contribution made in comm's scratch from `fill`, before anything reads them */
static void share_fill(struct plan *plan, uint32_t s) {
  struct call *c = plan->call;
  const struct phase *p = plan->phase;
  if (!plan->fill) return;
  for (uint64_t j = s, pages = pages_of(p->bytes); j < pages; j += SHARES) {
    size_t offset;
    const size_t bytes = page_at(p->bytes, j, &offset);
    if (c->premultiplied) premultiply(c->type, c->comm->scratch + offset, plan->fill + offset, c->pre, bytes / SIZE[c->type]);
    else memcpy(c->comm->scratch + offset, plan->fill + offset, bytes);
  }
}
/* share s's pages of this rank's local copy */
static void share_local(struct plan *plan, uint32_t s) {
  const struct local *l = &plan->local;
  for (uint64_t j = s, pages = pages_of(l->bytes); j < pages; j += SHARES) {
    size_t offset;
    const size_t bytes = page_at(l->bytes, j, &offset);
    memcpy(l->to + offset, l->from + offset, bytes);
  }
}
/* share s's positions of the reduction's result divided by nranks (an integer average) */
static void share_divide(struct plan *plan, uint32_t s) {
  struct call *c = plan->call;
  const struct phase *p = plan->phase;
  for (uint64_t j = s, pages = pages_of(p->bytes); j < pages; j += SHARES) {
    size_t offset;
    const size_t bytes = page_at(p->bytes, j, &offset);
    truncdiv(c->type, p->acc + offset, p->acc + offset, bytes / SIZE[c->type], (uint64_t)c->comm->nranks);
  }
}

/* Thread t's shares of a plan moved, each through the call's phases, each pass moving every one of them that is not
   done: its pages sent and its landed pages taken (send_some, take_some); a share done with its first phase divides
   its positions where the call averages integers, then begins the second where there is one; the thread's part of the
   local copy made after its first pass's writes.  The waits are the call's own: they end once the shares' pages have
   landed, at another thread's failure (`stop`), or at the communicator's revoke, which each pass reads and which ends
   it as a value.  The thread's result and its message are put in the plan */
static void shares_run(struct plan *plan, uint32_t t) {
  struct call *c = plan->call;
  struct ncclComm *comm = c->comm;
  struct moving run[SHARES];
  uint32_t stage[SHARES], shares[SHARES], runs = 0, done = 0;
  ncclResult_t result = ncclSuccess;
  for (uint32_t s = t; s < plan->active; s += plan->threads) {
    share_fill(plan, s);
    if ((result = moving_open(run + runs, c, plan->phase, s))) break;
    stage[runs] = 0;
    shares[runs++] = s;
  }
  int local = !plan->local.bytes;
  while (!result && done < runs) {
    for (uint32_t i = 0; i < runs; i++) {
      if (stage[i] == plan->phases) continue;
      struct moving *m = run + i;
      if (send_some(m, &result) || take_some(m, &result)) break;
      if (m->o < m->p->outs || m->left) continue;
      moving_close(m);
      if (!stage[i] && c->post && plan->phase[0].acc) share_divide(plan, shares[i]);
      if (++stage[i] < plan->phases) { if ((result = moving_open(m, c, plan->phase + stage[i], shares[i]))) break; }
      else done++;
    }
    if (result) break;
    if (!local) { for (uint32_t i = 0; i < runs; i++) share_local(plan, shares[i]); local = 1; }
    if (atomic_load_explicit(&plan->stop, memory_order_relaxed)) break;
    if (done < runs && revoked(comm)) { result = revoked_fail(comm); break; }
  }
  for (uint32_t i = 0; i < runs; i++)
    if (stage[i] < plan->phases) moving_close(run + i);
  plan->result[t] = result;
  if (result) { plan->said[t] = last; atomic_store_explicit(&plan->stop, 1, memory_order_relaxed); }
}

/* a thread of comm past its first: told each call (`job`), it moves its shares of the call where the call is moved on
   it (shares_run), then counts itself done (`running`); between calls it looks for the next for WORKER_SPIN_NS, then
   sleeps on `job` (counted `asleep`, so a call wakes it), until comm stops */
static void *comm_thread(void *arg) {
  const struct self *self = arg;
  struct ncclComm *comm = self->comm;
  uint64_t seen = atomic_load_explicit(&comm->job, memory_order_acquire), job;
  for (;;) {
    uint64_t until = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + WORKER_SPIN_NS;
    for (uint32_t looks = 0; (job = atomic_load_explicit(&comm->job, memory_order_acquire)) == seen;) {
      if (++looks & 255 || clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < until) continue;
      atomic_fetch_add(&comm->asleep, 1);
      if (atomic_load(&comm->job) == seen) os_sync_wait_on_address((void *)&comm->job, seen, sizeof comm->job, OS_SYNC_WAIT_ON_ADDRESS_NONE);
      atomic_fetch_sub(&comm->asleep, 1);
      until = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + WORKER_SPIN_NS;
    }
    seen = job;
    if (atomic_load(&comm->stopping)) return NULL;
    if (self->t < (job & 255)) {
      shares_run(comm->plan, self->t);
      atomic_fetch_sub_explicit(&comm->running, 1, memory_order_release);
    }
  }
}

/* comm's threads started, one for each slot of the process's attach past the first (fewer where one cannot be made) */
static void comm_threads_start(struct ncclComm *comm) {
  const uint32_t wanted = comm->attach->contexts;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INTERACTIVE, 0);
  for (comm->threads = 1; comm->threads < wanted; comm->threads++) {
    comm->self[comm->threads] = (struct self){comm, comm->threads};
    if (pthread_create(comm->thread + comm->threads, &attributes, comm_thread, comm->self + comm->threads)) break;
  }
  pthread_attr_destroy(&attributes);
}
/* comm's threads ended and joined (no call is under way: ncclCommDestroy, or an init that failed) */
static void comm_threads_stop(struct ncclComm *comm) {
  if (comm->threads < 2) return;
  atomic_store(&comm->stopping, 1);
  atomic_fetch_add(&comm->job, 256);
  os_sync_wake_by_address_all((void *)&comm->job, sizeof comm->job, OS_SYNC_WAKE_BY_ADDRESS_NONE);
  for (uint32_t t = 1; t < comm->threads; t++) pthread_join(comm->thread[t], NULL);
  comm->threads = 1;
}

/* a plan moved: told its threads past the first (the job's count raised and its threads put in its low bits, those
   asleep woken), thread 0's shares moved here, then each told thread waited for (the call's own end): the first
   thread's failure, in thread order, named again (ncclGetLastError), or ncclSuccess */
static ncclResult_t plan_run(struct plan *plan) {
  struct ncclComm *comm = plan->call->comm;
  if (plan->threads > 1) {
    comm->plan = plan;
    atomic_store_explicit(&comm->running, plan->threads - 1, memory_order_relaxed);
    atomic_store(&comm->job, (((atomic_load_explicit(&comm->job, memory_order_relaxed) >> 8) + 1) << 8) | plan->threads);
    if (atomic_load(&comm->asleep)) os_sync_wake_by_address_all((void *)&comm->job, sizeof comm->job, OS_SYNC_WAKE_BY_ADDRESS_NONE);
  }
  shares_run(plan, 0);
  while (atomic_load_explicit(&comm->running, memory_order_acquire)) {}
  for (uint32_t t = 0; t < plan->threads; t++)
    if (plan->result[t]) { last = plan->said[t]; return plan->result[t]; }
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
   the result `acc`, else made in comm's scratch (premultiplied, or copied, by the shares: share_fill, each position
   before the result's first store there); NULL where out of memory */
static const unsigned char *own_of(struct plan *plan, const unsigned char *from, const unsigned char *acc, size_t bytes) {
  struct call *c = plan->call;
  const uintptr_t f = (uintptr_t)from, a = (uintptr_t)acc;
  if (!c->premultiplied && (f + bytes <= a || a + bytes <= f)) return from;
  struct ncclComm *comm = c->comm;
  if (comm->scratch_bytes < bytes) {
    unsigned char *grown = realloc(comm->scratch, bytes);
    if (!grown) return NULL;
    comm->scratch = grown; comm->scratch_bytes = bytes;
  }
  plan->fill = from;
  return comm->scratch;
}
/* this rank's own part, `bytes` at `from`, copied to `to`: by the shares where the two lie apart (`local`), else here
   at once; nothing where they are one */
static void local_of(struct plan *plan, const unsigned char *from, unsigned char *to, size_t bytes) {
  const uintptr_t f = (uintptr_t)from, t = (uintptr_t)to;
  if (from == to) return;
  if (f + bytes <= t || t + bytes <= f) plan->local = (struct local){from, to, bytes};
  else memmove(to, from, bytes);
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
   This rank's own part is copied in place (none where it is there already); a reduction is in rank order, so every
   rank computes the same bits; a floating average premultiplies every contribution by 1 / nranks, and an integer one
   divides the reduced sum by nranks.  Its pages are moved in shares, on as many of comm's threads as it has shares
   with pages (plan_run): a call of one page on the calling thread alone */
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
  struct plan plan = {.call = c, .phases = 1};
  struct phase *p = plan.phase;
  *p = (struct phase){out, in, 0, 0, NULL, NULL, 0};
  ncclResult_t result = ncclSuccess;
  switch (c->what) {
  case CALL_ALLGATHER:
    local_of(&plan, send, recv + (size_t)me * bytes, bytes);
    out[p->outs++] = (struct out){send, bytes, -1, 0};
    for (int s = 0; s < n; s++)
      if (s != me) in[p->ins++] = (struct in){s, bytes, recv + (size_t)s * bytes};
    break;
  case CALL_BROADCAST:
    if (me != root) { in[p->ins++] = (struct in){root, bytes, recv}; break; }
    local_of(&plan, send, recv, bytes);
    out[p->outs++] = (struct out){send, bytes, -1, 0};
    break;
  case CALL_REDUCE:
    if (me != root) { out[p->outs++] = (struct out){send, bytes, root, c->premultiplied}; break; }
    p->acc = recv; p->bytes = bytes;
    break;
  case CALL_REDUCE_SCATTER:
    for (int d = 0; d < n; d++)
      if (d != me) out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, c->premultiplied};
    p->acc = recv; p->bytes = bytes;
    break;
  case CALL_ALLREDUCE:
    if (bytes <= MESH_LONG_PAYLOAD) { p->acc = recv; p->bytes = bytes; break; }
    for (int d = 0; d < n; d++) {
      size_t length;
      const size_t first = segment(c->count, n, d, &length);
      if (d != me) out[p->outs++] = (struct out){send + first * z, length * z, d, c->premultiplied};
      else { p->acc = recv + first * z; p->bytes = length * z; }
    }
    break;
  case CALL_ALLTOALL:
    local_of(&plan, send + (size_t)me * bytes, recv + (size_t)me * bytes, bytes);
    for (int d = 0; d < n; d++)
      if (d != me) {
        out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, 0};
        in[p->ins++] = (struct in){d, bytes, recv + (size_t)d * bytes};
      }
    break;
  case CALL_GATHER:
    if (me != root) { out[p->outs++] = (struct out){send, bytes, root, 0}; break; }
    local_of(&plan, send, recv + (size_t)root * bytes, bytes);
    for (int s = 0; s < n; s++)
      if (s != root) in[p->ins++] = (struct in){s, bytes, recv + (size_t)s * bytes};
    break;
  default: /* CALL_SCATTER */
    if (me != root) { in[p->ins++] = (struct in){root, bytes, recv}; break; }
    local_of(&plan, send + (size_t)root * bytes, recv, bytes);
    for (int d = 0; d < n; d++)
      if (d != root) out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, 0};
    break;
  }
  if (p->acc) {
    const size_t at = (size_t)(p->acc - recv);
    const unsigned char *own = c->what == CALL_REDUCE_SCATTER ? send + (size_t)me * bytes : send + at;
    if (!(p->own = own_of(&plan, own, p->acc, p->bytes))) result = fail(ncclSystemError, "communicator %s: out of memory", comm->name);
    /* the single-phase all-reduce sends what it reduces with, made once */
    if (c->what == CALL_ALLREDUCE && bytes <= MESH_LONG_PAYLOAD) out[p->outs++] = (struct out){p->own, bytes, -1, 0};
  }
  if (!result && c->what == CALL_ALLREDUCE && bytes > MESH_LONG_PAYLOAD) {
    struct phase *q = plan.phase + 1;
    *q = (struct phase){out + n, in + n, 0, 0, NULL, NULL, 0};
    q->out[q->outs++] = (struct out){p->acc, p->bytes, -1, 0};
    for (int s = 0; s < n; s++)
      if (s != me) {
        size_t length;
        const size_t first = segment(c->count, n, s, &length);
        q->in[q->ins++] = (struct in){s, length * z, recv + first * z};
      }
    plan.phases = 2;
  }
  if (!result) {
    uint64_t most = phase_pages(p), local = pages_of(plan.local.bytes);
    if (plan.phases > 1 && phase_pages(plan.phase + 1) > most) most = phase_pages(plan.phase + 1);
    if (local > most) most = local;
    plan.active = most < SHARES ? (uint32_t)most : SHARES;
    plan.threads = plan.active < comm->threads ? plan.active : comm->threads;
    if (plan.active) result = plan_run(&plan);
  }
  free(out); free(in);
  return result;
}

/* a call run on its communicator, there and then: refused where its communicator failed (its failure named again) or
   was revoked */
static ncclResult_t call_one(struct call *c) {
  struct ncclComm *comm = c->comm;
  atomic_fetch_add(&comm->active, 1);
  ncclResult_t result = comm_failure(comm);
  if (!result) result = revoked(comm) ? revoked_fail(comm) : call_move(c);
  atomic_fetch_sub(&comm->active, 1);
  return result;
}

ncclResult_t ncclGroupStart(void) { depth++; return ncclSuccess; }

/* the group's calls run in the order they were made, each to its end, every one run whatever another's result; the
   first failure returned */
ncclResult_t ncclGroupEnd(void) {
  if (depth <= 0) return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart");
  if (--depth) return ncclSuccess;
  struct call *list = calls;
  const size_t n = call_count;
  calls = NULL; call_count = call_capacity = 0;
  ncclResult_t result = ncclSuccess;
  struct said first = {0, ""};
  for (size_t i = 0; i < n; i++) {
    const ncclResult_t ran = call_one(list + i);
    if (ran && !result) { result = ran; first = last; }
  }
  free(list);
  if (result) last = first;
  return result;
}

/* whether a call reads its send buffer, or writes its receive buffer, on this rank */
static int call_reads(const struct call *c) {
  return (c->what != CALL_BROADCAST && c->what != CALL_SCATTER) || c->comm->rank == c->root;
}
static int call_writes(const struct call *c) {
  return (c->what != CALL_REDUCE && c->what != CALL_GATHER) || c->comm->rank == c->root;
}

/* a call refused before it joins its group (its arguments), else its operator taken and the call queued and its group
   ended (a call outside a group is a group of one) */
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
    return fail(ncclInvalidArgument, "%s: a NULL buffer this rank %s", name, call_reads(&c) && !c.send ? "reads" : "writes");
  c.combine = c.op == ncclAvg ? COMBINE_SUM : (int)c.op;
  if (reducing && c.op == ncclAvg && floating(c.type)) { store(c.type, c.pre, 1.0 / n); c.premultiplied = 1; }
  else if (reducing && c.op == ncclAvg) c.post = 1;
  ncclGroupStart();
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) { depth--; return fail(ncclSystemError, "%s: out of memory", name); }
    calls = grown; call_capacity = capacity;
  }
  calls[call_count++] = c;
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

/* a communicator of comm's ranks `ranks` (count of them, in that order), named after comm, its split's number and
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
  if (result) comm_free(sub);
  else *made = sub;
  return result;
}

/* NCCL's split: every rank's (color, key) gathered over comm's bootstrap, then the ranks of this rank's color, ordered
   by key and then rank, a communicator of their own (comm_sub; NCCL_SPLIT_NOCOLOR: none, *newcomm NULL).  config is
   not read */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm || (color < 0 && color != NCCL_SPLIT_NOCOLOR)) return fail(ncclInvalidArgument, "ncclCommSplit's arguments");
  *newcomm = NULL;
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
  return result;
}

/* NCCL's shrink: comm's ranks but the excluded ones, in rank order, a communicator of their own (comm_sub, which
   exchanges nothing over comm, so an excluded rank, gone or not, takes no part); a rank among those excluded is in
   none (*newcomm NULL); with NCCL_SHRINK_ABORT, comm revoked first.  config is not read */
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
  if (shrinkFlags & NCCL_SHRINK_ABORT) comm_stop(comm);
  const uint32_t split = comm->splits++;
  const ncclResult_t result = kept ? comm_sub(comm, ranks, count, split, -1, newcomm) : ncclSuccess;
  free(ranks);
  return result;
}

/* the id a grow of comm meets at: a listener of comm's root (ncclGetUniqueId) */
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
  return comm ? ncclGetUniqueId(uniqueId) : fail(ncclInvalidArgument, "comm is NULL");
}

/* NCCL's grow: an init at the id of comm's root (ncclCommGetUniqueId), which the root gives comm's other ranks over
   comm's bootstrap; comm's ranks keep their ranks (the `rank` they pass, NCCL's -1, not read) and a new one (comm NULL)
   takes `rank`.  config is not read */
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!newcomm || (comm ? nRanks < comm->nranks || (!comm->rank && !uniqueId) : !uniqueId))
    return fail(ncclInvalidArgument, "ncclCommGrow's arguments");
  if (!comm) return ncclCommInitRank(newcomm, nRanks, *uniqueId, rank);
  const ncclUniqueId mine = comm->rank ? (ncclUniqueId){{0}} : *uniqueId;
  ncclUniqueId *ids = calloc((size_t)comm->nranks, sizeof *ids);
  const ncclResult_t result = ids ? exchange(comm, &mine, sizeof mine, ids) : fail(ncclSystemError, "out of memory");
  const ncclUniqueId given = ids ? ids[0] : mine;
  free(ids);
  return result ? result : ncclCommInitRank(newcomm, nRanks, given, comm->rank);
}

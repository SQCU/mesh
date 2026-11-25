/* libnccl-mesh: NCCL's calls over the node's page table (nccl.h is the contract, rdma/NCCL.md how to build and use it;
   design/PAGE-TABLE-SPEC-20261122.md section 6).  Every call is blocking and moves host memory: what a rank sends it
   copies into source pages of its own and writes to the indices its peers took for it (mesh-call.h mesh_write), and
   what lands at its own indices it copies out, or combines rank by rank, into the call's recvbuff, as it landed, then
   invalidates. */
#include "nccl.h"
#include "mesh.h"
#include <dispatch/dispatch.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
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
#include <sys/sysctl.h>

enum { CALL_ALLREDUCE, CALL_BROADCAST, CALL_REDUCE, CALL_REDUCE_SCATTER, CALL_ALLGATHER, CALL_ALLTOALL, CALL_GATHER,
       CALL_SCATTER };
static const char *const CALL_NAME[] = {"ncclAllReduce", "ncclBroadcast", "ncclReduce", "ncclReduceScatter",
                                        "ncclAllGather", "ncclAlltoAll", "ncclGather", "ncclScatter"};
enum { COMBINE_SUM, COMBINE_PROD, COMBINE_MAX, COMBINE_MIN };
/* Page j of each message runs on channel j % CHANNELS. Each writer addresses the receiver's allocated extent. */
#define CHANNELS 4u
#define BLOCK_PAGES 8u
#define EXTENT_PAGES 1024u
#define SEND_BATCH 64u
#define ALLREDUCE_SINGLE_PHASE_MAX 36864u

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
#if defined(__ARM_NEON)
  if (t == ncclFloat32) {
    float *d = (float *)dst;
    const float *s = (const float *)src;
    float sc; memcpy(&sc, scalar, 4);
    const float32x4_t vsc = vdupq_n_f32(sc);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(d + i, vmulq_f32(vld1q_f32(s + i), vsc));
    for (; i < n; i++) d[i] = s[i] * sc;
    return;
  }
  if (t == ncclFloat16) {
    __fp16 *d = (__fp16 *)dst;
    const __fp16 *s = (const __fp16 *)src;
    const float sc = (float)load(t, scalar);
    const float32x4_t vsc = vdupq_n_f32(sc);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      const float16x8_t v = vld1q_f16(s + i);
      const float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(v)), vsc);
      const float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(v)), vsc);
      vst1q_f16(d + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
    }
    for (; i < n; i++) d[i] = (__fp16)((float)s[i] * sc);
    return;
  }
  if (t == ncclBfloat16) {
    uint16_t *d = (uint16_t *)dst;
    const uint16_t *s = (const uint16_t *)src;
    const float sc = (float)load(t, scalar);
    const float32x4_t vsc = vdupq_n_f32(sc);
    const uint32x4_t rbias = vdupq_n_u32(0x7fffu), one = vdupq_n_u32(1u);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      const float32x4_t fv = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(s + i)), 16));
      const uint32x4_t ur = vreinterpretq_u32_f32(vmulq_f32(fv, vsc));
      const uint32x4_t rnd = vaddq_u32(ur, vaddq_u32(rbias, vandq_u32(vshrq_n_u32(ur, 16), one)));
      vst1_u16(d + i, vmovn_u32(vshrq_n_u32(rnd, 16)));
    }
    for (; i < n; i++) d[i] = bf16_from(bf16_to(s[i]) * sc);
    return;
  }
#endif
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
  struct mesh_ctx context;
  uint32_t threads;
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
static void pool_start(struct attach *a) {
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INTERACTIVE, 0);
  for (uint32_t t = 1; t < a->threads; t++) {
    a->worker[t] = (struct worker){.attach = a, .t = t};
    if (!pthread_create(&a->worker[t].thread, &attributes, worker_main, a->worker + t)) continue;
    a->threads = t;
    break;
  }
  pthread_attr_destroy(&attributes);
}
static void pool_end(struct attach *a) {
  if (a->threads < 2) return;
  __atomic_store_n(&a->generation, ((__atomic_load_n(&a->generation, __ATOMIC_RELAXED) >> 8) + 1) << 8, __ATOMIC_RELEASE);
  for (uint32_t t = 1; t < a->threads; t++) pthread_join(a->worker[t].thread, NULL);
}

static int attach_expired(const struct attach *a) {
  return __atomic_load_n(&a->context.M->version, __ATOMIC_ACQUIRE) != MESH_LAYOUT;
}
static void attach_settle(void) {
  struct attach *a = attached;
  if (!a || a->comms || !attach_expired(a)) return;
  pool_end(a);
  mesh_detach(&a->context);
  pthread_mutex_destroy(&a->lock);
  free(a);
  attached = NULL;
}
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
static int attach_current(struct attach **out) {
  attach_settle();
  if (attached) { *out = attached; return attach_expired(attached) ? ESTALE : 0; }
  struct attach *a = NULL;
  if (posix_memalign((void **)&a, _Alignof(struct attach), sizeof *a)) return ENOMEM;
  memset(a, 0, sizeof *a);
  const int status = mesh_attach(&a->context);
  if (status) { free(a); return status; }
  a->threads = threads_wanted();
  pthread_mutex_init(&a->lock, NULL);
  pool_start(a);
  *out = attached = a;
  return 0;
}
static ncclResult_t attach_fail(int status, const char *name) {
  if (status == ENOMEM) return fail(ncclSystemError, "communicator %s: out of memory", name);
  if (status == ESTALE)
    return fail(ncclRemoteError, "communicator %s: the region of this node's bridge was released or replaced while this process has "
                "communicators of it; once it has destroyed them (ncclCommDestroy), its next communicator attaches the bridge's "
                "next region", name);
  return fail(ncclSystemError, "communicator %s: this process attaches no region of this node's bridge (%s); the node's bridge "
              "must be up (etc/bridge/README.md)", name, strerror(status));
}

/* -- communicators ------------------------------------------------------------------------------------------------- */

struct out { const unsigned char *data; size_t bytes; int to, pre; };
struct in { int from; size_t bytes; unsigned char *to; };
struct extent { uint32_t *idx; uint32_t used; };
struct peer_pages {
  struct extent send, receive;
  uint32_t *next_receive;
  int has_receive, has_next, advertised;
  const struct in *message;
  uint64_t due, low, at;
};
struct channel {
  _Alignas(128) struct mesh_ctx *context;
  struct peer_pages *peer;
  uint32_t *mailbox;
  void *positions;
  size_t capacity;
};

#define RED_SLOTS 2u
#define RED_WINDOW_PAGES 2048u

struct ncclComm {
  int nranks, rank;
  char name[NCCL_UNIQUE_ID_BYTES + 1], id[NCCL_UNIQUE_ID_BYTES + 1];
  uint32_t *node, *mailbox, *mailbox_buf, *idx_buf, *red_all;
  struct out *out;
  struct in *in;
  char (*address)[NCCL_UNIQUE_ID_BYTES];
  struct channel channel[CHANNELS];
  uint32_t extent_pages, red_pages, red_gen, splits;
  int has_red;
  struct attach *attach;
  int *peers, standing;
  unsigned char *scratch;
  size_t scratch_bytes;
  ncclResult_t failed;
  struct said failure;
  pthread_mutex_t guard;
  _Atomic uint32_t state, references;
};
enum comm_state { COMM_LIVE, COMM_FAILED, COMM_REVOKED };

static struct ncclComm *comm_new(int nranks, int rank, const char *name) {
  struct ncclComm *comm = NULL;
  if (posix_memalign((void **)&comm, _Alignof(struct ncclComm), sizeof *comm)) return NULL;
  memset(comm, 0, sizeof *comm);
  const size_t n = (size_t)nranks;
  comm->node = calloc(n, sizeof *comm->node);
  comm->mailbox = calloc(n * CHANNELS, sizeof *comm->mailbox);
  comm->mailbox_buf = malloc(n * CHANNELS * sizeof *comm->mailbox_buf);
  comm->address = calloc(n, sizeof *comm->address);
  struct peer_pages *peers = calloc(n * CHANNELS, sizeof *peers);
  comm->peers = malloc(n * sizeof *comm->peers);
  comm->out = malloc(2 * n * sizeof *comm->out);
  comm->in = malloc(2 * n * sizeof *comm->in);
  if (!comm->node || !comm->mailbox || !comm->mailbox_buf || !comm->address || !peers || !comm->peers || !comm->out || !comm->in) {
    free(comm->out); free(comm->in);
    free(comm->node); free(comm->mailbox); free(comm->mailbox_buf); free(comm->address); free(peers); free(comm->peers);
    free(comm);
    return NULL;
  }
  for (size_t r = 0; r < n; r++) comm->peers[r] = -1;
  for (size_t i = 0; i < n * CHANNELS; i++) comm->mailbox_buf[i] = MESH_ABSENT;
  comm->nranks = nranks; comm->rank = rank; comm->standing = -1;
  for (uint32_t c = 0; c < CHANNELS; c++)
    comm->channel[c] = (struct channel){.peer = peers + c * n, .mailbox = comm->mailbox_buf + c * n};
  pthread_mutex_init(&comm->guard, NULL);
  atomic_init(&comm->state, COMM_LIVE);
  atomic_init(&comm->references, 1);
  snprintf(comm->name, sizeof comm->name, "%s", name);
  return comm;
}

static int comm_attach(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  const int status = attach_current(&comm->attach);
  if (!status) comm->attach->comms++;
  else comm->attach = NULL;
  pthread_mutex_unlock(&attach_lock);
  return status;
}

static int comm_receives(struct ncclComm *comm) {
  const uint32_t n = (uint32_t)comm->nranks, count = comm->extent_pages;
  comm->idx_buf = calloc(3 * (size_t)n * CHANNELS * count, sizeof *comm->idx_buf);
  if (!comm->idx_buf) return ENOMEM;
  uint32_t *cur = comm->idx_buf;
  for (uint32_t c = 0; c < CHANNELS; c++) {
    struct channel *ch = comm->channel + c;
    ch->context = &comm->attach->context;
    for (uint32_t r = 0; r < n; r++) {
      ch->peer[r].send.idx = cur; cur += count;
      ch->peer[r].receive.idx = cur; cur += count;
      ch->peer[r].next_receive = cur; cur += count;
      if ((int)r == comm->rank) continue;
      int rc = mesh_alloc(ch->context, 1, 0, &ch->mailbox[r], NULL);
      if (rc) return rc;
      rc = mesh_alloc(ch->context, count, 0, ch->peer[r].receive.idx, NULL);
      if (rc) return rc;
      ch->peer[r].has_receive = 1;
    }
  }
  return 0;
}

static void comm_leave(struct ncclComm *comm) {
  pthread_mutex_lock(&attach_lock);
  struct attach *a = comm->attach;
  if (a) {
    if (comm->has_red && comm->red_all && comm->channel[0].context) {
      mesh_free_indices(comm->channel[0].context,
                        comm->red_all + (size_t)comm->rank * RED_SLOTS * comm->red_pages,
                        RED_SLOTS * comm->red_pages);
      comm->has_red = 0;
    }
    for (uint32_t c = 0; c < CHANNELS; c++) {
      struct channel *x = comm->channel + c;
      if (x->context) {
        for (int r = 0; r < comm->nranks; r++) {
          struct peer_pages *pages = x->peer + r;
          if (pages->has_receive) { mesh_free_indices(x->context, pages->receive.idx, comm->extent_pages); pages->has_receive = 0; }
          if (pages->has_next) { mesh_free_indices(x->context, pages->next_receive, comm->extent_pages); pages->has_next = 0; }
          if (x->mailbox[r] != MESH_ABSENT) { mesh_free_indices(x->context, &x->mailbox[r], 1); x->mailbox[r] = MESH_ABSENT; }
        }
      }
    }
    a->comms--;
    comm->attach = NULL;
    attach_settle();
  }
  pthread_mutex_unlock(&attach_lock);
}

static void bootstrap_free(struct ncclComm *comm) {
  for (int r = 0; comm->peers && r < comm->nranks; r++)
    if (comm->peers[r] >= 0) close(comm->peers[r]);
  free(comm->peers);
  comm->peers = NULL;
}

static void comm_free(struct ncclComm *comm) {
  bootstrap_free(comm);
  if (comm->standing >= 0) close(comm->standing);
  comm_leave(comm);
  pthread_mutex_destroy(&comm->guard);
  for (uint32_t c = 0; c < CHANNELS; c++) free(comm->channel[c].positions);
  free(comm->out); free(comm->in);
  free(comm->node); free(comm->mailbox); free(comm->mailbox_buf); free(comm->idx_buf); free(comm->red_all);
  free(comm->address); free(comm->channel[0].peer); free(comm->scratch);
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
   grow's) and releases where its init fails.  Every wait is a plain blocking call on its own socket: a non-root's
   connect (where no address of the root takes it, tried again BOOTSTRAP_RETRY_MS later: an NCCL_COMM_ID may name a
   root not listening yet, where a listener this library opens listens before any rank has its id), the root's
   accept, and each send and receive.  A communicator under init is no other thread's until its init returns it, so
   no revoke reaches its init, whose waits its peers alone end; an exchange on a communicator made ends at its revoke
   too (ncclCommRevoke, ncclCommDestroy), which shuts its sockets down.  Nothing else bounds a wait of the
   bootstrap. */
static struct { int fd; char id[NCCL_UNIQUE_ID_BYTES]; } root = {-1, {0}};
#define BOOTSTRAP_RETRY_MS 100

struct report { int32_t rank, status; uint32_t node, pages; char address[NCCL_UNIQUE_ID_BYTES]; };

static uint32_t extent_pages(const struct report *reports, int nranks) {
  uint32_t count = EXTENT_PAGES;
  for (int r = 0; r < nranks; r++) {
    uint64_t here = 0;
    for (int q = 0; q < nranks; q++) here += reports[q].node == reports[r].node;
    const uint64_t denom = 4 * here * CHANNELS * (uint64_t)(nranks - 1);
    const uint64_t room_p = reports[r].pages / denom;
    const uint64_t room_i = (uint64_t)MESH_TABLE / denom;
    const uint64_t room = room_p < room_i ? room_p : room_i;
    if (count > room) count = (uint32_t)room;
  }
  return count;
}

/* a listener on `host` alone where it is an IPv6 address, else on every address, at *port (0: any, *port then the
   one bound), blocking (an accept waits for its connection): its descriptor, or -1 and errno */
static int bootstrap_listen(const char *host, uint16_t *port) {
  struct sockaddr_in6 at = {.sin6_len = sizeof at, .sin6_family = AF_INET6, .sin6_port = htons(*port)};
  socklen_t length = sizeof at;
  int on = 1, fd = socket(AF_INET6, SOCK_STREAM, 0);
  if (inet_pton(AF_INET6, host, &at.sin6_addr) != 1) at.sin6_addr = in6addr_any;
  if (fd >= 0 && !setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on) && !bind(fd, (struct sockaddr *)&at, sizeof at) &&
      !listen(fd, 64) && !getsockname(fd, (struct sockaddr *)&at, &length)) {
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

/* `size` bytes sent or received whole on comm's bootstrap socket `fd`, each a plain blocking call, which comm's revoke
   ends by shutting the socket down: 0, ECANCELED (revoked), ECONNRESET (the peer ended the connection), or the
   errno */
static int whole(struct ncclComm *comm, int fd, void *bytes, size_t size, int sending) {
  for (size_t done = 0; done < size;) {
    const ssize_t moved = sending ? send(fd, (char *)bytes + done, size - done, 0) : recv(fd, (char *)bytes + done, size - done, 0);
    if (moved < 0 && errno == EINTR) continue;
    if (moved <= 0) {
      const int error = moved ? errno : ECONNRESET;
      return revoked(comm) ? ECANCELED : error;
    }
    done += (size_t)moved;
  }
  return 0;
}

/* a bootstrap socket `fd` whose sends never raise SIGPIPE: one after its peer's reset, or after its communicator's
   revoke shut it down, is an error (EPIPE) */
static int bootstrap_socket(int fd) {
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &(int){1}, sizeof(int));
  return fd;
}

/* the bootstrap's sockets of comm's init at `id`: a non-root's to the root (peers[0], placed once connected), or the
   root's to each rank r (peers[r], placed by the rank its first-round report names, that report put in `reports`; a
   connection whose report fails or names no rank still to meet is released alone, the root accepting the next).  The
   root accepts on `standing` where it is a listener (a parent's standing listener, left open), else on the listener
   ncclGetUniqueId opened for `id`, else on one it opens at id's port, releasing either once every rank has met.
   peers[] is written here and at the communicator's free alone, while no other thread holds it: 0, or the errno */
static int bootstrap_open(struct ncclComm *comm, const char *id, struct report *reports, int standing) {
  int *peers = comm->peers, status = 0;
  char host[NCCL_UNIQUE_ID_BYTES + 1];
  snprintf(host, sizeof host, "%s", id);
  char *colon = strrchr(host, ':');
  if (!colon) return EINVAL;
  *colon = 0;
  if (comm->rank) {
    struct addrinfo hint = {.ai_socktype = SOCK_STREAM}, *addresses = NULL;
    while (peers[0] < 0) {
      if (!getaddrinfo(host, colon + 1, &hint, &addresses)) {
        for (struct addrinfo *a = addresses; a && peers[0] < 0; a = a->ai_next) {
          const int fd = socket(a->ai_family, SOCK_STREAM, 0);
          if (fd < 0) continue;
          if (connect(fd, a->ai_addr, a->ai_addrlen)) close(fd);
          else peers[0] = bootstrap_socket(fd);
        }
        freeaddrinfo(addresses);
      }
      if (peers[0] < 0) usleep(BOOTSTRAP_RETRY_MS * 1000);
    }
    return 0;
  }
  uint16_t port = (uint16_t)atoi(colon + 1);
  int listener = standing >= 0 ? standing : !strcmp(root.id, id) ? root.fd : -1;
  if (listener < 0 && (listener = bootstrap_listen(host, &port)) < 0) return errno;
  for (int r = 1; r < comm->nranks && !status;) {
    const int fd = accept(listener, NULL, NULL);
    if (fd < 0) {
      if (errno != EINTR && errno != ECONNABORTED) status = errno;
      continue;
    }
    struct report heard;
    /* a connection whose report does not come, or names no rank still to meet, is released alone */
    if (whole(comm, bootstrap_socket(fd), &heard, sizeof heard, 0) || heard.rank <= 0 || heard.rank >= comm->nranks ||
        peers[heard.rank] >= 0) {
      close(fd);
      continue;
    }
    reports[heard.rank] = heard;
    peers[heard.rank] = fd;
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

static ncclResult_t comm_init(struct ncclComm *made, int standing) {
  const int nranks = made->nranks, rank = made->rank;
  struct report *reports = calloc((size_t)nranks, sizeof *reports);
  if (!reports) return fail(ncclSystemError, "communicator %s: out of memory", made->name);
  struct report *mine = reports + rank;
  const int attaching = comm_attach(made);
  int status = attaching;
  if (!status && nranks > 1 && (made->standing = listener_open(mine->address, sizeof mine->address)) < 0) status = errno;
  mine->rank = rank; mine->status = status;
  if (!attaching) {
    struct hdr *m = made->attach->context.M;
    mine->node = m->node;
    mine->pages = mesh_flat_pages(m);
  }
  int met = nranks > 1 ? bootstrap_open(made, made->id, reports, standing) : 0;
  if (!met && nranks > 1) met = gather(made, reports, sizeof *reports, !rank);
  ncclResult_t result = ncclSuccess;
  if (attaching) result = attach_fail(attaching, made->name);
  else if (status) result = fail(ncclSystemError, "communicator %s: listener: %s", made->name, strerror(status));
  else if (met) result = fail(ncclRemoteError, "communicator %s: the bootstrap at %s: %s", made->name, made->id, strerror(met));
  for (int r = 0; !result && r < nranks; r++)
    if (reports[r].status)
      result = fail(ncclRemoteError, "communicator %s: rank %d could not join it: %s", made->name, r, strerror(reports[r].status));
  if (!result && nranks > 1 && !(made->extent_pages = extent_pages(reports, nranks)))
    result = fail(ncclSystemError, "communicator %s: the node arena has no space for a "
                  "receive page for each pair: %s", made->name, strerror(ENOSPC));
  uint32_t *bases = NULL;
  size_t per_peer = 0, base_words = 0, red_off = 0;
  if (!result && nranks > 1) {
    made->red_pages = RED_WINDOW_PAGES;
    per_peer = 1u + (size_t)made->extent_pages;
    red_off = 1u + (size_t)nranks * CHANNELS * per_peer;
    base_words = red_off + (size_t)RED_SLOTS * made->red_pages;
    bases = calloc((size_t)nranks, base_words * sizeof *bases);
    made->red_all = calloc((size_t)nranks * RED_SLOTS * made->red_pages, sizeof(uint32_t));
    if (!bases || !made->red_all) {
      result = fail(ncclSystemError, "communicator %s: out of memory", made->name);
    } else {
      int own_status = comm_receives(made);
      uint32_t *own_red = made->red_all + (size_t)rank * RED_SLOTS * made->red_pages;
      if (!own_status) {
        own_status = mesh_alloc(made->channel[0].context, RED_SLOTS * made->red_pages, 0, own_red, NULL);
        if (!own_status) made->has_red = 1;
      }
      uint32_t *own = bases + (size_t)rank * base_words;
      own[0] = (uint32_t)own_status;
      if (!own_status) {
        for (uint32_t c = 0; c < CHANNELS; c++) {
          for (int r = 0; r < nranks; r++) {
            if (r == rank) continue;
            uint32_t *slot = own + 1u + ((size_t)c * (size_t)nranks + (size_t)r) * per_peer;
            slot[0] = made->channel[c].mailbox[r];
            memcpy(slot + 1, made->channel[c].peer[r].receive.idx, made->extent_pages * sizeof(uint32_t));
          }
        }
        memcpy(own + red_off, own_red, (size_t)RED_SLOTS * made->red_pages * sizeof(uint32_t));
      }
      met = gather(made, bases, base_words * sizeof *bases, 0);
      if (own_status)
        result = fail(ncclSystemError, "communicator %s: its %u receive indices of this node's bridge: %s", made->name,
                      (uint32_t)(nranks - 1) * made->extent_pages * CHANNELS, strerror(own_status));
      else if (met) result = fail(ncclRemoteError, "communicator %s: the bootstrap at %s: %s", made->name, made->id, strerror(met));
      for (int r = 0; !result && r < nranks; r++) {
        const int32_t rs = (int32_t)bases[(size_t)r * base_words];
        if (rs) result = fail(ncclRemoteError, "communicator %s: rank %d could not take its receive indices: %s", made->name, r, strerror(rs));
      }
    }
  }
  for (int r = 0; !result && r < nranks; r++) {
    made->node[r] = reports[r].node;
    if (nranks > 1) {
      if (r != rank) {
        for (uint32_t c = 0; c < CHANNELS; c++) {
          const uint32_t *slot = bases + (size_t)r * base_words + 1u + ((size_t)c * (size_t)nranks + (size_t)rank) * per_peer;
          made->mailbox[(size_t)r * CHANNELS + c] = slot[0];
          memcpy(made->channel[c].peer[r].send.idx, slot + 1, made->extent_pages * sizeof(uint32_t));
        }
      }
      memcpy(made->red_all + (size_t)r * RED_SLOTS * made->red_pages,
             bases + (size_t)r * base_words + red_off,
             (size_t)RED_SLOTS * made->red_pages * sizeof(uint32_t));
    }
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
   to rank `to`, or to every other rank where `to` is -1) and those it receives (`in`: `bytes` from rank `from`,
   copied to `to`), and, where `acc` is set, its reduction: `bytes` at `acc` made the combine, rank by rank, of every
   rank's, this rank's at `own` and every other's a message to it.  A pair of ranks carries at most one message in a
   phase, so a message's first page is its pair's count of pages as the phase begins */
struct phase { struct out *out; struct in *in; uint32_t outs, ins; unsigned char *acc; const unsigned char *own; size_t bytes; };

static size_t page_payload(size_t bytes) { (void)bytes; return MESH_PAGE_BYTES; }
static uint64_t pages_of(size_t bytes) { return (bytes + page_payload(bytes) - 1) / page_payload(bytes); }
static size_t page_at(size_t bytes, uint64_t j, size_t *offset) {
  const size_t p = page_payload(bytes);
  *offset = (size_t)j * p;
  return bytes - *offset < p ? bytes - *offset : p;
}

struct channel_phase {
  struct call *call;
  const struct phase *p;
  struct channel *channel;
  uint32_t id;
  uint32_t o;
  uint64_t q;
  unsigned char *taken;
  uint64_t positions, left;
  uint32_t *next;
};

static uint64_t page_of_channel(uint64_t q, uint32_t id) {
  return ((q / BLOCK_PAGES) * CHANNELS + id) * BLOCK_PAGES + (q % BLOCK_PAGES);
}

static uint64_t channel_pages(uint64_t pages, uint32_t id) {
  const uint64_t full = pages / BLOCK_PAGES, tail = pages % BLOCK_PAGES;
  const uint64_t my_full = full > id ? (full - 1u - id) / CHANNELS + 1u : 0u;
  const uint64_t my_tail = (full % CHANNELS == id) ? tail : 0u;
  return my_full * BLOCK_PAGES + my_tail;
}

static int destination(struct channel_phase *m, int rank) {
  struct ncclComm *comm = m->call->comm;
  struct extent *send = &m->channel->peer[rank].send;
  if (send->used == comm->extent_pages) {
    const uint32_t mailbox = m->channel->mailbox[rank];
    struct mesh_entry_info e;
    int rc = mesh_entry(m->channel->context, mailbox, &e);
    if (rc) return rc;
    if (!e.data) return EAGAIN;
    const uint32_t want = comm->extent_pages * (uint32_t)sizeof(uint32_t);
    if (e.bytes != want) return EIO;
    memcpy(send->idx, e.data, want);
    send->used = 0;
    mesh_clear_indices(m->channel->context, &mailbox, 1);
  }
  return 0;
}

static int extend_receive(struct channel_phase *m, int r) {
  struct ncclComm *comm = m->call->comm;
  struct channel *ch = m->channel;
  struct peer_pages *peer = ch->peer + r;
  const uint32_t count = comm->extent_pages;
  const uint32_t used = peer->receive.used;
  if (used < (count + 1) / 2 || peer->due - peer->low <= count - used) return 0;
  if (!peer->has_next) {
    int rc = mesh_alloc(ch->context, count, 0, peer->next_receive, NULL);
    if (rc) return rc == ENOMEM ? EAGAIN : rc;
    peer->has_next = 1;
  }
  if (!peer->advertised) {
    uint32_t pg = MESH_PAGE_NONE;
    if (mesh_alloc(ch->context, 1, 1, NULL, &pg)) return EAGAIN;
    const uint32_t offer_bytes = count * (uint32_t)sizeof(uint32_t);
    memcpy(mesh_page_ptr(ch->context, pg), peer->next_receive, offer_bytes);
    const struct mesh_xfer x = {
      .src_page = pg, .bytes = offer_bytes,
      .dst_node = comm->node[r], .dst_index = comm->mailbox[(size_t)r * CHANNELS + m->id]
    };
    int rc = mesh_write(ch->context, &x, 1);
    mesh_release(ch->context, pg);
    if (rc) return rc == ENOMEM ? EAGAIN : rc;
    peer->advertised = 1;
  }
  if (peer->receive.used == count) {
    mesh_free_indices(ch->context, peer->receive.idx, count);
    uint32_t *tmp = peer->receive.idx;
    peer->receive.idx = peer->next_receive;
    peer->next_receive = tmp;
    peer->receive.used = 0;
    peer->has_next = 0;
    peer->advertised = 0;
  }
  return 0;
}

static int other(int me, uint32_t g) { return (int)g + ((int)g >= me); }

static int send_some(struct channel_phase *m, ncclResult_t *result) {
  struct call *c = m->call;
  struct ncclComm *comm = c->comm;
  struct channel *channel = m->channel;
  const struct phase *p = m->p;
  const int me = comm->rank;
  for (; m->o < p->outs; m->o++, m->q = 0) {
    const struct out *x = p->out + m->o;
    const uint64_t total_pages = pages_of(x->bytes);
    const uint64_t ch_pages = channel_pages(total_pages, m->id);
    const uint32_t dests = x->to < 0 ? (uint32_t)comm->nranks - 1 : 1;
    const uint32_t max_b = dests <= MESH_DESC_FLUSH ? (MESH_DESC_FLUSH / dests < SEND_BATCH ? MESH_DESC_FLUSH / dests : SEND_BATCH) : 1u;
    while (m->q < ch_pages) {
      const uint64_t rem = ch_pages - m->q;
      uint32_t min_left = comm->extent_pages;
      for (uint32_t i = 0; i < dests; i++) {
        const int d = x->to < 0 ? other(me, i) : x->to;
        const int rc = destination(m, d);
        if (rc == EAGAIN) return 0;
        if (rc) { *result = comm_fail(comm, ncclInternalError, "receive extent from rank %d: %s", d, strerror(rc)); return -1; }
        const uint32_t left = comm->extent_pages - channel->peer[d].send.used;
        if (left < min_left) min_left = left;
      }
      uint32_t batch = max_b;
      if ((uint64_t)batch > rem) batch = (uint32_t)rem;
      if (batch > min_left) batch = min_left;

      /* Bulk 32 KiB O(1) SWAR 8-page run path (zero contig_pages contention). */
      if ((m->q & (BLOCK_PAGES - 1u)) == 0 && batch >= BLOCK_PAGES) {
        static _Thread_local uint32_t run_base, run_cur;
        if (!run_base) {
          uint64_t tid = 0;
          pthread_threadid_np(NULL, &tid);
          run_base = 1u + ((uint32_t)((tid ^ (tid >> 32) ^ (uint64_t)getpid()) * 2654435761u) >> 16 & 1023u) * 32u;
          run_cur = run_base;
        }
        batch &= ~(BLOCK_PAGES - 1u);
        const uint32_t want_blks = batch / BLOCK_PAGES;
        uint32_t sp_blk[SEND_BATCH / BLOCK_PAGES], got_blks = 0;
        for (; got_blks < want_blks; got_blks++) {
          const uint32_t w = run_cur;
          run_cur = run_base + ((run_cur - run_base + 1u) & 31u);
          const uint32_t sp = mesh_page_alloc_run(channel->context->M, BLOCK_PAGES, w);
          if (sp == MESH_PAGE_NONE) break;
          sp_blk[got_blks] = sp;
        }
        if (got_blks > 0u) {
          batch = got_blks * BLOCK_PAGES;
          struct mesh_xfer stack_xfers[SEND_BATCH * MESH_DESTS];
          struct mesh_xfer *xp = dests <= MESH_DESTS ? stack_xfers : malloc((size_t)batch * dests * sizeof *xp);
          if (!xp) {
            for (uint32_t b = 0; b < got_blks; b++) mesh_page_release_run(channel->context->M, sp_blk[b], BLOCK_PAGES);
            *result = comm_fail(comm, ncclSystemError, "communicator %s: out of memory", comm->name);
            return -1;
          }
          uint32_t nx = 0;
          for (uint32_t bi = 0; bi < got_blks; bi++) {
            const uint32_t blk = bi * BLOCK_PAGES;
            const uint32_t sp0 = sp_blk[bi];
            const uint64_t j0 = page_of_channel(m->q + blk, m->id);
            const size_t blk_off = (size_t)j0 * MESH_PAGE_BYTES;
            const size_t blk_bytes = x->bytes - blk_off < BLOCK_PAGES * MESH_PAGE_BYTES ? x->bytes - blk_off : BLOCK_PAGES * MESH_PAGE_BYTES;
            unsigned char *dst_ptr = mesh_page_at(channel->context->M, sp0);
            if (x->pre) premultiply(c->type, dst_ptr, x->data + blk_off, c->pre, blk_bytes / SIZE[c->type]);
            else memcpy(dst_ptr, x->data + blk_off, blk_bytes);
            for (uint32_t i = 0; i < dests; i++) {
              const int d = x->to < 0 ? other(me, i) : x->to;
              const struct extent *send = &channel->peer[d].send;
              for (uint32_t k = 0; k < BLOCK_PAGES; k++) {
                size_t poff;
                const size_t pb = page_at(x->bytes, j0 + k, &poff);
                xp[nx++] = (struct mesh_xfer){
                  .src_page = sp0 + k, .bytes = (uint32_t)pb,
                  .dst_node = comm->node[d], .dst_index = send->idx[send->used + blk + k]
                };
              }
            }
          }
          const int status = mesh_write(channel->context, xp, nx);
          if (xp != stack_xfers) free(xp);
          for (uint32_t b = 0; b < got_blks; b++) mesh_page_release_run(channel->context->M, sp_blk[b], BLOCK_PAGES);
          if (status == ENOMEM) return 0;
          if (status) {
            *result = comm_fail(comm, ncclInternalError, "communicator %s: %s's write: %s", comm->name, CALL_NAME[c->what], strerror(status));
            return -1;
          }
          for (uint32_t i = 0; i < dests; i++) {
            const int d = x->to < 0 ? other(me, i) : x->to;
            channel->peer[d].send.used += batch;
          }
          m->q += batch;
          continue;
        }
      }

      /* Scattered / tiny (< 8 pages) path */
      if (batch > BLOCK_PAGES) batch = BLOCK_PAGES;
      uint32_t sp[SEND_BATCH];
      if (mesh_alloc(channel->context, batch, 1, NULL, sp)) {
        batch = 1;
        if (mesh_alloc(channel->context, 1, 1, NULL, sp)) return 0;
      }
      struct mesh_xfer stack_xfers[SEND_BATCH * MESH_DESTS];
      struct mesh_xfer *xp = dests <= MESH_DESTS ? stack_xfers : malloc((size_t)batch * dests * sizeof *xp);
      if (!xp) {
        for (uint32_t b = 0; b < batch; b++) mesh_release(channel->context, sp[b]);
        *result = comm_fail(comm, ncclSystemError, "communicator %s: out of memory", comm->name);
        return -1;
      }
      uint32_t nx = 0;
      for (uint32_t b = 0; b < batch; b++) {
        size_t offset;
        const size_t bytes = page_at(x->bytes, page_of_channel(m->q + b, m->id), &offset);
        unsigned char *data = mesh_page_ptr(channel->context, sp[b]);
        if (x->pre) premultiply(c->type, data, x->data + offset, c->pre, bytes / SIZE[c->type]);
        else memcpy(data, x->data + offset, bytes);
        for (uint32_t i = 0; i < dests; i++) {
          const int d = x->to < 0 ? other(me, i) : x->to;
          const struct extent *send = &channel->peer[d].send;
          xp[nx++] = (struct mesh_xfer){
            .src_page = sp[b], .bytes = (uint32_t)bytes,
            .dst_node = comm->node[d], .dst_index = send->idx[send->used + b]
          };
        }
      }
      const int status = mesh_write(channel->context, xp, nx);
      if (xp != stack_xfers) free(xp);
      for (uint32_t b = 0; b < batch; b++) mesh_release(channel->context, sp[b]);
      if (status == ENOMEM) return 0;
      if (status) {
        *result = comm_fail(comm, ncclInternalError, "communicator %s: %s's write: %s", comm->name, CALL_NAME[c->what], strerror(status));
        return -1;
      }
      for (uint32_t i = 0; i < dests; i++) {
        const int d = x->to < 0 ? other(me, i) : x->to;
        channel->peer[d].send.used += batch;
      }
      m->q += batch;
    }
  }
  return 0;
}

static void own_contribution(struct channel_phase *m, uint64_t q) {
  if (m->next[q] != (uint32_t)m->call->comm->rank) return;
  size_t offset;
  size_t bytes = page_at(m->p->bytes, page_of_channel(q, m->id), &offset);
  if (!m->next[q]) memcpy(m->p->acc + offset, m->p->own + offset, bytes);
  else combine_into(m->call->type, m->call->combine, m->p->acc + offset, m->p->own + offset, bytes / SIZE[m->call->type]);
  m->next[q]++;
}

static int take(struct channel_phase *m, int r, uint64_t q, uint32_t index, uint32_t *clr, uint32_t *nclr, ncclResult_t *result) {
  struct call *c = m->call;
  struct ncclComm *comm = c->comm;
  struct channel *channel = m->channel;
  const struct phase *p = m->p;
  const struct peer_pages *peer = channel->peer + r;
  const struct in *x = peer->message;
  if (!x && m->next[q] != (uint32_t)r) return 0;
  struct mesh_entry_info entry;
  if (mesh_entry(channel->context, index, &entry)) {
    *result = comm_fail(comm, ncclInternalError, "communicator %s: its index %u could not be read", comm->name, index);
    return -1;
  }
  if (!entry.data) return 0;
  size_t offset;
  const size_t bytes = page_at(x ? x->bytes : p->bytes, page_of_channel(q, m->id), &offset);
  if (entry.bytes != bytes) {
    *result = comm_fail(comm, ncclInternalError, "page %u contains %u bytes; expected %zu", index, entry.bytes, bytes);
    return -1;
  }
  if (x) memcpy(x->to + offset, entry.data, bytes);
  else {
    if (!r) memcpy(p->acc + offset, entry.data, bytes);
    else combine_into(c->type, c->combine, p->acc + offset, entry.data, bytes / SIZE[c->type]);
    m->next[q]++;
    own_contribution(m, q);
  }
  clr[(*nclr)++] = index;
  if (*nclr == 64u) { mesh_clear_indices(channel->context, clr, *nclr); *nclr = 0; }
  m->taken[peer->at + q] = 1;
  m->left--;
  return 0;
}

static int take_some(struct channel_phase *m, ncclResult_t *result) {
  struct ncclComm *comm = m->call->comm;
  uint32_t clr[64], nclr = 0;
  for (int r = 0; r < comm->nranks; r++) {
    struct peer_pages *peer = m->channel->peer + r;
    if (peer->low == peer->due) continue;
    struct extent *recv = &peer->receive;
    uint32_t room = comm->extent_pages - recv->used;
    uint64_t low = peer->low, end = peer->due - low > room ? low + room : peer->due;
    uint64_t scan_end = low + 64u < end ? low + 64u : end;
    const unsigned char *taken = m->taken + peer->at;
    for (uint64_t q = low; q < scan_end; q++)
      if (!taken[q] && take(m, r, q, recv->idx[recv->used + (uint32_t)(q - low)], clr, &nclr, result)) {
        if (nclr) mesh_clear_indices(m->channel->context, clr, nclr);
        return -1;
      }
    if (nclr) { mesh_clear_indices(m->channel->context, clr, nclr); nclr = 0; }
    while (peer->low < end && taken[peer->low]) { peer->low++; recv->used++; }
    int rc = extend_receive(m, r);
    if (rc && rc != EAGAIN) {
      *result = comm_fail(comm, ncclSystemError, "receive extent for rank %d: %s", r, strerror(rc));
      return -1;
    }
  }
  return 0;
}

static ncclResult_t channel_phase_begin(struct channel_phase *m, struct call *c, const struct phase *p, uint32_t id) {
  struct ncclComm *comm = c->comm;
  const int n = comm->nranks, me = comm->rank;
  *m = (struct channel_phase){.call = c, .p = p, .channel = comm->channel + id, .id = id, .q = 0,
                              .positions = p->acc ? channel_pages(pages_of(p->bytes), id) : 0};
  struct channel *ch = m->channel;
  for (int r = 0; r < n; r++) ch->peer[r].message = NULL;
  for (uint32_t i = 0; i < p->ins; i++) ch->peer[p->in[i].from].message = p->in + i;
  size_t pages = 0;
  for (int r = 0; r < n; r++) {
    struct peer_pages *peer = ch->peer + r;
    peer->due = peer->message ? channel_pages(pages_of(peer->message->bytes), id) : (r == me ? 0 : m->positions);
    peer->low = 0; peer->at = pages; pages += peer->due;
  }
  size_t bytes = m->positions * sizeof *m->next + pages;
  if (bytes > ch->capacity) {
    void *grown = realloc(ch->positions, bytes);
    if (!grown) return fail(ncclSystemError, "communicator %s: out of memory", comm->name);
    ch->positions = grown; ch->capacity = bytes;
  }
  if (bytes) {
    memset(ch->positions, 0, bytes);
    m->next = ch->positions;
    m->taken = (unsigned char *)(m->next + m->positions);
  }
  m->left = pages;
  for (uint64_t q = 0; q < m->positions; q++) own_contribution(m, q);
  return ncclSuccess;
}

/* A call's work descriptor (call_move), which every thread that moves a channel of it reads: its phases (a second for
   an all-reduce of more than one page: the all-gather of the reduced segments); this rank's one copy of its own part
   where it lies apart from where it goes (`local`), and the reduction's own contribution where comm's scratch makes it
   (`fill`, the bytes it is copied, or premultiplied, from), each made by the channels, page by page; the channels with
   pages (`active`: channels 0 to active - 1) and the threads that move them (`threads`, thread t channels t,
   t + threads, ...); each thread's result and its message */
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

static void channel_fill(struct work *work, uint32_t id) {
  struct call *c = work->call;
  const struct phase *p = work->phase;
  if (!work->fill) return;
  for (uint64_t q = 0, n = channel_pages(pages_of(p->bytes), id); q < n; q++) {
    size_t offset;
    const size_t bytes = page_at(p->bytes, page_of_channel(q, id), &offset);
    if (c->premultiplied) premultiply(c->type, c->comm->scratch + offset, work->fill + offset, c->pre, bytes / SIZE[c->type]);
    else memcpy(c->comm->scratch + offset, work->fill + offset, bytes);
  }
}
static void channel_local(struct work *work, uint32_t id) {
  const struct local *l = &work->local;
  for (uint64_t q = 0, n = channel_pages(pages_of(l->bytes), id); q < n; q++) {
    size_t offset;
    const size_t bytes = page_at(l->bytes, page_of_channel(q, id), &offset);
    memcpy(l->to + offset, l->from + offset, bytes);
  }
}
static void channel_divide(struct work *work, uint32_t id) {
  struct call *c = work->call;
  const struct phase *p = work->phase;
  for (uint64_t q = 0, n = channel_pages(pages_of(p->bytes), id); q < n; q++) {
    size_t offset;
    const size_t bytes = page_at(p->bytes, page_of_channel(q, id), &offset);
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
      if (!phase[i] && c->post && work->phase[0].acc) channel_divide(work, channels[i]);
      if (++phase[i] < work->phases) { if ((result = channel_phase_begin(m, c, work->phase + phase[i], channels[i]))) break; }
      else done++;
    }
    if (result) break;
    if (!local) { for (uint32_t i = 0; i < count; i++) channel_local(work, channels[i]); local = 1; }
    if (done < count && revoked(comm)) { result = revoked_fail(comm); break; }
  }
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
  if (work->threads > 1) {
    a->work = work;
    __atomic_store_n(&a->pending, work->threads - 1, __ATOMIC_RELAXED);
    __atomic_store_n(&a->generation, (((__atomic_load_n(&a->generation, __ATOMIC_RELAXED) >> 8) + 1) << 8) | work->threads,
                     __ATOMIC_RELEASE);
  }
  channels_progress(work, 0);
  while (__atomic_load_n(&a->pending, __ATOMIC_ACQUIRE)) {}
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
/* this rank's own part, `bytes` at `from`, copied to `to`: by the channels where the two lie apart (`local`), else
   here at once; nothing where they are one */
static void local_of(struct work *work, const unsigned char *from, unsigned char *to, size_t bytes) {
  const uintptr_t f = (uintptr_t)from, t = (uintptr_t)to;
  if (from == to) return;
  if (f + bytes <= t || t + bytes <= f) work->local = (struct local){from, to, bytes};
  else memmove(to, from, bytes);
}
/* the most pages any of a phase's messages, or its reduction, has */
static uint64_t phase_pages(const struct phase *p) {
  uint64_t most = p->acc ? pages_of(p->bytes) : 0;
  for (uint32_t o = 0; o < p->outs; o++) if (pages_of(p->out[o].bytes) > most) most = pages_of(p->out[o].bytes);
  for (uint32_t i = 0; i < p->ins; i++) if (pages_of(p->in[i].bytes) > most) most = pages_of(p->in[i].bytes);
  return most;
}

static uint8_t nccl_red_op(ncclDataType_t type, int combine) {
  switch (type) {
  case ncclFloat32:
    if (combine == COMBINE_SUM) return MESH_RED_SUM_F32;
    if (combine == COMBINE_MAX) return MESH_RED_MAX_F32;
    if (combine == COMBINE_MIN) return MESH_RED_MIN_F32;
    return MESH_RED_NONE;
  case ncclFloat16:
    if (combine == COMBINE_SUM) return MESH_RED_SUM_F16;
    if (combine == COMBINE_MAX) return MESH_RED_MAX_F16;
    if (combine == COMBINE_MIN) return MESH_RED_MIN_F16;
    return MESH_RED_NONE;
  case ncclBfloat16:
    if (combine == COMBINE_SUM) return MESH_RED_SUM_BF16;
    if (combine == COMBINE_MAX) return MESH_RED_MAX_BF16;
    if (combine == COMBINE_MIN) return MESH_RED_MIN_BF16;
    return MESH_RED_NONE;
  case ncclInt32:
    return combine == COMBINE_SUM ? MESH_RED_SUM_I32 : MESH_RED_NONE;
  case ncclInt64:
    return combine == COMBINE_SUM ? MESH_RED_SUM_I64 : MESH_RED_NONE;
  default:
    return MESH_RED_NONE;
  }
}

struct rs_mt_ctx {
  struct call *c;
  struct ncclComm *comm;
  struct mesh_ctx *ctx;
  const unsigned char *const *seg_src;
  const size_t *seg_bytes;
  unsigned char *my_dst;
  uint64_t win_p;
  uint32_t win_n, my_n, win_cap, sl, nthreads;
  uint8_t red_op;
  uint16_t my_tag, done_tag;
  _Atomic int err;
  ncclResult_t res;
};

static void rs_mt_worker(void *arg, size_t tid) {
  struct rs_mt_ctx *ctx_w = (struct rs_mt_ctx *)arg;
  struct call *c = ctx_w->c;
  struct ncclComm *comm = ctx_w->comm;
  struct mesh_ctx *ctx = ctx_w->ctx;
  struct hdr *m = ctx->M;
  const int n = comm->nranks, me = comm->rank;
  const uint32_t win_cap = ctx_w->win_cap, sl = ctx_w->sl;
  const uint64_t win_p = ctx_w->win_p;
  const uint32_t stride_blk = ctx_w->nthreads * BLOCK_PAGES;

  uint32_t rs_base = 1u + ((uint32_t)((((uint64_t)tid + 1u) ^ (uint64_t)getpid()) * 2654435761u) >> 16 & 1023u) * 32u;
  uint32_t rs_cur = rs_base;

  struct mesh_red_xfer xf[MESH_DESC_FLUSH];
  uint32_t rel_pages[MESH_DESC_FLUSH];
  uint32_t nx = 0, nr = 0;

  for (uint32_t blk = (uint32_t)tid * BLOCK_PAGES; blk < ctx_w->win_n; blk += stride_blk) {
    if (atomic_load_explicit(&ctx_w->err, memory_order_relaxed)) break;
    for (int d = 0; d < n; d++) {
      const uint64_t dp = pages_of(ctx_w->seg_bytes[d]);
      if (win_p + blk >= dp) continue;
      const uint32_t cnt = (uint32_t)(dp - (win_p + blk) < BLOCK_PAGES ? dp - (win_p + blk) : BLOCK_PAGES);
      if (nx + cnt > MESH_DESC_FLUSH) {
        int st;
        while ((st = mesh_reduce_write(ctx, xf, nx)) == ENOMEM) {
          if (revoked(comm)) {
            for (uint32_t k = 0; k < nr; k++) mesh_release(ctx, rel_pages[k]);
            if (!atomic_exchange(&ctx_w->err, 1)) ctx_w->res = revoked_fail(comm);
            return;
          }
        }
        for (uint32_t k = 0; k < nr; k++) mesh_release(ctx, rel_pages[k]);
        nx = 0; nr = 0;
        if (st) {
          if (!atomic_exchange(&ctx_w->err, 1))
            ctx_w->res = comm_fail(comm, ncclInternalError, "communicator %s: reduce_write: %s", comm->name, strerror(st));
          return;
        }
      }
      const uint32_t *didx = comm->red_all + ((size_t)d * RED_SLOTS + sl) * win_cap + blk;
      const size_t blk_off = (size_t)(win_p + blk) * MESH_PAGE_BYTES;
      const size_t blk_bytes = ctx_w->seg_bytes[d] - blk_off < (size_t)cnt * MESH_PAGE_BYTES
                                   ? ctx_w->seg_bytes[d] - blk_off : (size_t)cnt * MESH_PAGE_BYTES;

      uint32_t sp0 = MESH_PAGE_NONE;
      if (cnt == BLOCK_PAGES) {
        const uint32_t w = rs_cur;
        rs_cur = rs_base + ((rs_cur - rs_base + 1u) & 31u);
        sp0 = mesh_page_alloc_run(m, BLOCK_PAGES, w);
      }
      if (sp0 != MESH_PAGE_NONE) {
        unsigned char *dst_ptr = mesh_page_at(m, sp0);
        if (c->premultiplied) premultiply(c->type, dst_ptr, ctx_w->seg_src[d] + blk_off, c->pre, blk_bytes / SIZE[c->type]);
        else memcpy(dst_ptr, ctx_w->seg_src[d] + blk_off, blk_bytes);
        for (uint32_t k = 0; k < BLOCK_PAGES; k++) {
          const size_t poff = blk_off + (size_t)k * MESH_PAGE_BYTES;
          const uint32_t pb = (uint32_t)(ctx_w->seg_bytes[d] - poff < MESH_PAGE_BYTES ? ctx_w->seg_bytes[d] - poff : MESH_PAGE_BYTES);
          xf[nx++] = (struct mesh_red_xfer){ sp0 + k, pb, comm->node[d], didx[k], ctx_w->red_op, 0, ctx_w->my_tag };
          rel_pages[nr++] = sp0 + k;
        }
      } else {
        uint32_t sp[BLOCK_PAGES];
        while (mesh_alloc(ctx, cnt, 1, NULL, sp) == ENOMEM) {
          if (revoked(comm)) {
            for (uint32_t k = 0; k < nr; k++) mesh_release(ctx, rel_pages[k]);
            if (!atomic_exchange(&ctx_w->err, 1)) ctx_w->res = revoked_fail(comm);
            return;
          }
        }
        for (uint32_t k = 0; k < cnt; k++) {
          const size_t poff = blk_off + (size_t)k * MESH_PAGE_BYTES;
          const uint32_t pb = (uint32_t)(ctx_w->seg_bytes[d] - poff < MESH_PAGE_BYTES ? ctx_w->seg_bytes[d] - poff : MESH_PAGE_BYTES);
          unsigned char *dst_ptr = mesh_page_ptr(ctx, sp[k]);
          if (c->premultiplied) premultiply(c->type, dst_ptr, ctx_w->seg_src[d] + poff, c->pre, pb / SIZE[c->type]);
          else memcpy(dst_ptr, ctx_w->seg_src[d] + poff, pb);
          xf[nx++] = (struct mesh_red_xfer){ sp[k], pb, comm->node[d], didx[k], ctx_w->red_op, 0, ctx_w->my_tag };
          rel_pages[nr++] = sp[k];
        }
      }
    }
  }
  if (nx) {
    int st;
    while ((st = mesh_reduce_write(ctx, xf, nx)) == ENOMEM) {
      if (revoked(comm)) {
        for (uint32_t k = 0; k < nr; k++) mesh_release(ctx, rel_pages[k]);
        if (!atomic_exchange(&ctx_w->err, 1)) ctx_w->res = revoked_fail(comm);
        return;
      }
    }
    for (uint32_t k = 0; k < nr; k++) mesh_release(ctx, rel_pages[k]);
    if (st) {
      if (!atomic_exchange(&ctx_w->err, 1))
        ctx_w->res = comm_fail(comm, ncclInternalError, "communicator %s: reduce_write: %s", comm->name, strerror(st));
      return;
    }
  }

  /* Drain this thread's disjoint blocks from shared W[my_idx] in parallel! */
  const uint32_t *my_idx = comm->red_all + ((size_t)me * RED_SLOTS + sl) * win_cap;
  for (uint32_t blk = (uint32_t)tid * BLOCK_PAGES; blk < ctx_w->my_n; blk += stride_blk) {
    const uint32_t cnt = ctx_w->my_n - blk < BLOCK_PAGES ? ctx_w->my_n - blk : BLOCK_PAGES;
    for (uint32_t k = 0; k < cnt; k++) {
      const uint32_t pi = blk + k;
      struct mesh_entry_info e;
      uint16_t tag = 0;
      for (;;) {
        mesh_entry_red(ctx, my_idx[pi], &e, &tag);
        if (e.data && tag == ctx_w->done_tag) break;
        if (revoked(comm)) {
          if (!atomic_exchange(&ctx_w->err, 1)) ctx_w->res = revoked_fail(comm);
          return;
        }
      }
      const size_t poff = (size_t)(win_p + pi) * MESH_PAGE_BYTES;
      const size_t pb = ctx_w->seg_bytes[me] - poff < MESH_PAGE_BYTES ? ctx_w->seg_bytes[me] - poff : MESH_PAGE_BYTES;
      if (c->post) truncdiv(c->type, ctx_w->my_dst + poff, (const unsigned char *)e.data, pb / SIZE[c->type], (uint64_t)n);
      else memcpy(ctx_w->my_dst + poff, e.data, pb);
    }
  }
}

/* Multi-threaded algebraic in-switch ReduceScatter phase over mesh_reduce_write + shared W[dst_idx] landings. */
static ncclResult_t reduce_scatter_in_switch(struct call *c, uint8_t red_op,
                                             const unsigned char *const *seg_src,
                                             const size_t *seg_bytes,
                                             unsigned char *my_dst) {
  struct ncclComm *comm = c->comm;
  struct mesh_ctx *ctx = comm->channel[0].context;
  const int n = comm->nranks, me = comm->rank;
  const uint32_t win_cap = comm->red_pages;
  const uint16_t my_tag = MESH_TAG_WEIGHT(1);
  const uint16_t done_tag = MESH_TAG_WEIGHT(n);

  uint64_t max_pages = 0;
  for (int d = 0; d < n; d++) {
    const uint64_t dp = pages_of(seg_bytes[d]);
    if (dp > max_pages) max_pages = dp;
  }
  const uint64_t my_total_pages = pages_of(seg_bytes[me]);

  for (uint64_t win_p = 0; win_p < max_pages; win_p += win_cap) {
    const uint32_t sl = (comm->red_gen++) & (RED_SLOTS - 1u);
    const uint32_t win_n = (uint32_t)(max_pages - win_p < win_cap ? max_pages - win_p : win_cap);
    const uint32_t my_n = win_p < my_total_pages
                              ? (uint32_t)(my_total_pages - win_p < win_cap ? my_total_pages - win_p : win_cap)
                              : 0u;
    const uint32_t nblks = (win_n + BLOCK_PAGES - 1u) / BLOCK_PAGES;
    const uint32_t max_t = comm->attach->threads;
    const uint32_t nthreads = nblks < max_t ? (nblks ? nblks : 1u) : max_t;
    struct rs_mt_ctx wctx = {
      .c = c, .comm = comm, .ctx = ctx, .seg_src = seg_src, .seg_bytes = seg_bytes, .my_dst = my_dst,
      .win_p = win_p, .win_n = win_n, .my_n = my_n, .win_cap = win_cap, .sl = sl, .nthreads = nthreads,
      .red_op = red_op, .my_tag = my_tag, .done_tag = done_tag, .err = 0, .res = ncclSuccess
    };
    if (nthreads > 1u)
      dispatch_apply_f(nthreads, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), &wctx, rs_mt_worker);
    else
      rs_mt_worker(&wctx, 0);
    if (atomic_load(&wctx.err)) return wctx.res;
    if (my_n) mesh_clear_indices(ctx, comm->red_all + ((size_t)me * RED_SLOTS + sl) * win_cap, my_n);
  }
  return ncclSuccess;
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
   This rank's own part is copied in place (none where it is there already); a reduction goes rank by rank, so every
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
  struct out *out = comm->out;
  struct in *in = comm->in;
  struct work work = {.call = c, .phases = 1};
  struct phase *p = work.phase;
  *p = (struct phase){out, in, 0, 0, NULL, NULL, 0};
  ncclResult_t result = ncclSuccess;
  const uint8_t red_op = comm->has_red && n > 1 ? nccl_red_op(c->type, c->combine) : MESH_RED_NONE;
  if (red_op != MESH_RED_NONE &&
      (c->what == CALL_REDUCE_SCATTER || (c->what == CALL_ALLREDUCE && bytes > ALLREDUCE_SINGLE_PHASE_MAX))) {
    const unsigned char *seg_src[64];
    size_t seg_bytes[64];
    int all_nonempty = (n <= 64);
    unsigned char *my_dst = recv;
    if (c->what == CALL_REDUCE_SCATTER) {
      for (int d = 0; d < n && all_nonempty; d++) {
        seg_src[d] = send + (size_t)d * bytes;
        seg_bytes[d] = bytes;
      }
      my_dst = recv;
    } else {
      for (int d = 0; d < n && all_nonempty; d++) {
        size_t length = 0;
        const size_t first = segment(c->count, n, d, &length);
        if (!length) { all_nonempty = 0; break; }
        seg_src[d] = send + first * z;
        seg_bytes[d] = length * z;
        if (d == me) my_dst = recv + first * z;
      }
    }
    if (all_nonempty) {
      result = reduce_scatter_in_switch(c, red_op, seg_src, seg_bytes, my_dst);
      if (result || c->what == CALL_REDUCE_SCATTER) return result;
      /* CALL_ALLREDUCE phase 2: Steiner multicast AllGather of the in-switch reduced segments! */
      *p = (struct phase){out, in, 0, 0, NULL, NULL, 0};
      p->out[p->outs++] = (struct out){my_dst, seg_bytes[me], -1, 0};
      for (int s = 0; s < n; s++)
        if (s != me) {
          size_t length = 0;
          const size_t first = segment(c->count, n, s, &length);
          p->in[p->ins++] = (struct in){s, length * z, recv + first * z};
        }
      uint64_t most = phase_pages(p);
      const uint64_t blocks = (most + BLOCK_PAGES - 1u) / BLOCK_PAGES;
      const uint32_t threads = comm->attach->threads;
      work.active = blocks < CHANNELS ? (uint32_t)blocks : CHANNELS;
      work.threads = work.active < threads ? work.active : threads;
      return work.active ? work_launch(&work) : ncclSuccess;
    }
  }
  switch (c->what) {
  case CALL_ALLGATHER:
    local_of(&work, send, recv + (size_t)me * bytes, bytes);
    out[p->outs++] = (struct out){send, bytes, -1, 0};
    for (int s = 0; s < n; s++)
      if (s != me) in[p->ins++] = (struct in){s, bytes, recv + (size_t)s * bytes};
    break;
  case CALL_BROADCAST:
    if (me != root) { in[p->ins++] = (struct in){root, bytes, recv}; break; }
    local_of(&work, send, recv, bytes);
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
    if (bytes <= ALLREDUCE_SINGLE_PHASE_MAX) { p->acc = recv; p->bytes = bytes; break; }
    for (int d = 0; d < n; d++) {
      size_t length;
      const size_t first = segment(c->count, n, d, &length);
      if (d != me) out[p->outs++] = (struct out){send + first * z, length * z, d, c->premultiplied};
      else { p->acc = recv + first * z; p->bytes = length * z; }
    }
    break;
  case CALL_ALLTOALL:
    local_of(&work, send + (size_t)me * bytes, recv + (size_t)me * bytes, bytes);
    for (int d = 0; d < n; d++)
      if (d != me) {
        out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, 0};
        in[p->ins++] = (struct in){d, bytes, recv + (size_t)d * bytes};
      }
    break;
  case CALL_GATHER:
    if (me != root) { out[p->outs++] = (struct out){send, bytes, root, 0}; break; }
    local_of(&work, send, recv + (size_t)root * bytes, bytes);
    for (int s = 0; s < n; s++)
      if (s != root) in[p->ins++] = (struct in){s, bytes, recv + (size_t)s * bytes};
    break;
  default: /* CALL_SCATTER */
    if (me != root) { in[p->ins++] = (struct in){root, bytes, recv}; break; }
    local_of(&work, send + (size_t)root * bytes, recv, bytes);
    for (int d = 0; d < n; d++)
      if (d != root) out[p->outs++] = (struct out){send + (size_t)d * bytes, bytes, d, 0};
    break;
  }
  if (p->acc) {
    const size_t at = (size_t)(p->acc - recv);
    const unsigned char *own = c->what == CALL_REDUCE_SCATTER ? send + (size_t)me * bytes : send + at;
    if (!(p->own = own_of(&work, own, p->acc, p->bytes))) result = fail(ncclSystemError, "communicator %s: out of memory", comm->name);
    if (c->what == CALL_ALLREDUCE && bytes <= ALLREDUCE_SINGLE_PHASE_MAX) out[p->outs++] = (struct out){p->own, bytes, -1, 0};
  }
  if (!result && c->what == CALL_ALLREDUCE && bytes > ALLREDUCE_SINGLE_PHASE_MAX) {
    struct phase *q = work.phase + 1;
    *q = (struct phase){out + n, in + n, 0, 0, NULL, NULL, 0};
    q->out[q->outs++] = (struct out){p->acc, p->bytes, -1, 0};
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
    const uint64_t blocks = (most + BLOCK_PAGES - 1u) / BLOCK_PAGES;
    const uint32_t threads = comm->attach->threads;
    work.active = blocks < CHANNELS ? (uint32_t)blocks : CHANNELS;
    work.threads = work.active < threads ? work.active : threads;
    if (work.active) result = work_launch(&work);
  }
  return result;
}

/* a call moved on its communicator, there and then: refused where its communicator failed (its failure named again)
   or was revoked (comm_failure, one load of its state) */
static ncclResult_t call_one(struct call *c) {
  struct ncclComm *comm = c->comm;
  while (pthread_mutex_trylock(&comm->attach->lock))
    if (revoked(comm)) return revoked_fail(comm);
  ncclResult_t result = comm_failure(comm);
  if (!result) result = call_move(c);
  pthread_mutex_unlock(&comm->attach->lock);
  return result;
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
  if (!nesting) {
    comm_hold(c.comm);
    ncclResult_t result = call_one(&c);
    comm_release(c.comm);
    return result;
  }
  if (call_count == call_capacity) {
    const size_t capacity = call_capacity ? 2 * call_capacity : 16;
    struct call *grown = realloc(calls, capacity * sizeof *grown);
    if (!grown) return fail(ncclSystemError, "%s: out of memory", name);
    calls = grown; call_capacity = capacity;
  }
  calls[call_count++] = c;
  comm_hold(c.comm);
  return ncclSuccess;
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

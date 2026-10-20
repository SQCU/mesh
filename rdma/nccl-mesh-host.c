/* The host executor (mesh 800fe6bc^, on HEAD's libnccl-mesh; design/transport.md): the walk's kernels
   (nccl-mesh-metal.h) on the CPU, so libnccl-mesh's Metal path runs on host threads: N ranks on one host over
   mesh-flow-loop's bridges, with no GPU kernel anywhere (several ranks' spinning kernels on one GPU have no
   forward-progress guarantee: the no-cross-queue-spin rule).  Each kernel runs as the walk encodes it, in the walk's
   order, its body transcribed from nccl-mesh-metal.m's: copies and combines in host memory (the host path's arithmetic,
   nccl-mesh.c combine_into, premultiply and truncdiv, which nccl-mesh.c exports hidden to this file where
   MESH_HOST_EXECUTOR is defined), a run's ready words release-stored, a wait for a landing a poll of its run's last
   unit's landed word (mesh.metal mesh_wait_bounded: until it reaches 1 + the cycle or the bridge stores ~0, for at most
   the program's polls looks, 0 or more than NCCL_MESH_POLLS being NCCL_MESH_POLLS, at most MESH_PLATFORM_NS, and until
   the communicator's stop word [6] is set, design/transport.md), its failure (the bound reached, the link lost, a dud,
   a revoke) recorded in the failure words [0], [3], [4] and the gate [2], and every later step gated on them doing
   nothing; a step naming no words is gated on the program's (metal_gate), as nccl-mesh-metal.m dispatch_as binds them.
   A buffer is a host address (metal_wrap returns its address; metal_contents it); what metal_scratch, metal_shared and
   metal_event made is counted and freed with its last release.  The Metal 4 form's own steps (a caller's kernel running
   a step: metal_send_deferred, metal_land_deferred) are refused as the classic form refuses them.  Built as
   .build/host/libnccl-mesh.dylib (make host), the GPU library's install name, beside .build/host/libmesh.dylib (the
   loop client's libmesh). */
#include "mesh-collective.h"
#include "nccl-mesh-metal.h"
#include "nccl.h"
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void nccl_mesh_combine(int t, int combine, void *dst, const void *src, size_t n);
void nccl_mesh_premultiply(int t, void *dst, const void *src, uint64_t bits, size_t n);
void nccl_mesh_truncdiv(int t, void *dst, const void *src, size_t n, uint64_t divisor);

/* ncclDataType_t's order and sizes: int8 uint8 int32 uint32 int64 uint64 float16 float32 float64 bfloat16 e4m3 e5m2; the
   walk takes every type but float64, as the kernels do, and the partials merge's rows (METAL_MERGE) */
static const size_t SIZE[12] = {1, 1, 4, 4, 8, 8, 2, 4, 8, 2, 1, 1};
static int walked(int type) { return type >= 0 && type <= METAL_MERGE && type != 8; }

static char error_[512];
static pthread_mutex_t owned_lock = PTHREAD_MUTEX_INITIALIZER;
struct held { void *object; int count; };
static struct held *owned_;
static size_t owned_count_, owned_capacity_;
struct metal_program { void **kept; size_t count, capacity; _Atomic uint64_t *violation; uint32_t polls; };
struct host_event { _Atomic uint64_t value; };

/* what metal_scratch and metal_event made, counted: the caller's reference and each program's or retainer's, freed
   with the last */
static void own(void *object) {
  pthread_mutex_lock(&owned_lock);
  if (owned_count_ == owned_capacity_) {
    const size_t grown = owned_capacity_ ? 2 * owned_capacity_ : 64;
    struct held *bigger = realloc(owned_, grown * sizeof *bigger);
    if (bigger) { owned_ = bigger; owned_capacity_ = grown; }
  }
  if (owned_count_ < owned_capacity_) owned_[owned_count_++] = (struct held){object, 1};
  pthread_mutex_unlock(&owned_lock);
}

static int hold(void *object) {
  int held = 0;
  pthread_mutex_lock(&owned_lock);
  for (size_t i = 0; i < owned_count_ && !held; i++)
    if (owned_[i].object == object) { owned_[i].count++; held = 1; }
  pthread_mutex_unlock(&owned_lock);
  return held;
}

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static unsigned char *at(void *buffer, size_t offset) { return (unsigned char *)buffer + offset; }

void *metal_device(void) { return (void *)1; }
const char *metal_error(void) { return error_; }
void metal_records(uint64_t *issued, uint64_t *capacity) { *issued = 0; *capacity = 0; }
void *metal_wrap(void *address, size_t bytes) { (void)bytes; return address; }
void *metal_contents(void *buffer) { return buffer; }

void *metal_scratch(size_t bytes) {
  void *made = calloc(1, bytes ? bytes : 16);
  if (made) own(made);
  return made;
}

void *metal_shared(size_t bytes) { return metal_scratch(bytes); }

void *metal_event(void) {
  struct host_event *made = calloc(1, sizeof *made);
  if (made) own(made);
  return made;
}

uint64_t metal_signaled(void *event) { return atomic_load_explicit(&((struct host_event *)event)->value, memory_order_acquire); }

int metal_wait(void *event, uint64_t value, uint64_t ns) {
  const uint64_t deadline = now_ns() + ns;
  for (uint64_t polls = 0; metal_signaled(event) < value; polls++) {
    if (now_ns() >= deadline) return 0;
    if (polls > 4096) usleep(20);
    else if (polls > 256) sched_yield();
  }
  return 1;
}

void metal_release(void *object) {
  pthread_mutex_lock(&owned_lock);
  for (size_t i = 0; i < owned_count_; i++)
    if (owned_[i].object == object) {
      if (!--owned_[i].count) { owned_[i] = owned_[--owned_count_]; free(object); }
      break;
    }
  pthread_mutex_unlock(&owned_lock);
}

void *metal_retain(void *object) { hold(object); return object; }

/* nccl-mesh-metal.m bound_of: 0, or more than NCCL_MESH_POLLS, is NCCL_MESH_POLLS */
static uint32_t bound_of(uint32_t polls) { return polls && polls < NCCL_MESH_POLLS ? polls : NCCL_MESH_POLLS; }

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls) {
  (void)command_buffer; (void)encoder;
  struct metal_program *program = calloc(1, sizeof *program);
  if (program) { program->violation = violation; program->polls = bound_of(polls); }
  return program;
}

/* no Metal 4 on the host: the walk runs its classic steps here */
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls) {
  (void)table; (void)residency;
  return metal_begin(NULL, encoder, violation, polls);
}
int metal_metal4(const struct metal_program *program) { (void)program; return 0; }
void metal_resident(struct metal_program *program, void *buffer) { (void)program; (void)buffer; }
void metal_gate(struct metal_program *program, void *violation) { program->violation = violation; }

/* a step's failure words: those it names (a wait's), else the program's, and the bound that holds with them
   (nccl-mesh-metal.m dispatch_as: p.polls is the program's where there are such words) */
static _Atomic uint64_t *words_of(struct metal_program *program, void *given, uint32_t *polls) {
  _Atomic uint64_t *v = given ? given : program->violation;
  *polls = v ? program->polls : 0;
  return v;
}
/* mesh_step_enter: a step whose failure words' gate [2] holds a failure does nothing; nor does one that writes a lane
   once the stop word [6] is set (nccl-mesh-metal.m STOP) */
static int halted(_Atomic uint64_t *v, uint32_t polls) { return polls && v && atomic_load_explicit(v + 2, memory_order_acquire); }
static int stopped(_Atomic uint64_t *v, uint32_t polls) { return halted(v, polls) || (polls && v && atomic_load_explicit(v + 6, memory_order_acquire)); }

/* mesh_step_wait: the word until it reaches 1 + cycle (or ~0), at most `polls` looks (0: one), at most MESH_PLATFORM_NS
   (a host look being far slower than a GPU load, design/transport.md) and while the stop word [6] is clear */
static uint64_t step_wait(_Atomic uint64_t *word, uint64_t cycle, uint32_t polls, _Atomic uint64_t *v) {
  const uint64_t target = 1 + cycle, deadline = now_ns() + MESH_PLATFORM_NS;
  uint64_t seen = atomic_load_explicit(word, memory_order_acquire);
  for (uint64_t k = 1; seen < target && k < polls; k++) {
    if (!(k & 63) && (now_ns() > deadline || (v && atomic_load_explicit(v + 6, memory_order_acquire)))) break;
    /* N ranks share the host's cores with N bridges: a wait that runs long yields them */
    if (k > 4096) usleep(20);
    else if (k > 256) sched_yield();
    seen = atomic_load_explicit(word, memory_order_acquire);
  }
  return seen;
}

/* mesh_step_check: a wait's failure recorded where the gate is clear: a revoke, the link lost (~0), the bound reached,
   or a dud (its run's first eight bytes all ones) */
static void step_check(const struct metal_wait *w, _Atomic uint64_t *v, uint32_t polls, uint64_t status) {
  if (!v || !polls || atomic_load_explicit(v + 2, memory_order_acquire)) return;
  atomic_thread_fence(memory_order_acquire);
  uint64_t head;
  memcpy(&head, at(w->buffer, w->slot), sizeof head);
  const uint64_t cause = atomic_load_explicit(v + 6, memory_order_acquire) ? METAL_REVOKED : status == UINT64_MAX ? METAL_LOST
                       : status < 1 + w->cycle ? METAL_BOUND : head == UINT64_MAX ? METAL_DUD : 0;
  if (!cause) return;
  atomic_store_explicit(v, cause, memory_order_relaxed);
  atomic_store_explicit(v + 3, w->channel, memory_order_relaxed);
  atomic_store_explicit(v + 4, polls, memory_order_relaxed);
  atomic_store_explicit(v + 2, w->at + 1, memory_order_release);
}

/* a step's wait (step_wait) and its check (step_check): whether its landing came and nothing failed */
static int step_await(const struct metal_wait *w, _Atomic uint64_t *v, uint32_t polls) {
  step_check(w, v, polls, step_wait((_Atomic uint64_t *)at(w->buffer, w->word), w->cycle, polls, v));
  atomic_thread_fence(memory_order_acquire);
  return !halted(v, polls);
}

/* mesh_step_release: the run's ready words stored, each after every store before it */
static void step_release(const struct metal_release *r) {
  atomic_thread_fence(memory_order_release);
  for (uint32_t u = 0; u < r->count; u++)
    atomic_store_explicit((_Atomic uint64_t *)at(r->buffer, r->word) + u, r->value, memory_order_release);
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if ((mode == METAL_SEND ? stopped(v, polls) : halted(v, polls)) || !bytes) return 0;
  if (mode == METAL_LAND) atomic_thread_fence(memory_order_acquire);
  memmove(at(dst, dst_offset), at(src, src_offset), bytes);
  if (mode == METAL_SEND) atomic_thread_fence(memory_order_release);
  return 0;
}

int metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (!walked(type)) return EINVAL;
  if (halted(v, polls) || !n) return 0;
  atomic_thread_fence(memory_order_acquire);
  nccl_mesh_combine(type, op, at(dst, dst_offset), at(src, src_offset), n);
  return 0;
}

int metal_combine3(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, void *with,
                   size_t with_offset, size_t n) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (!walked(type) || type == METAL_MERGE) return EINVAL;
  if (halted(v, polls) || !n) return 0;
  atomic_thread_fence(memory_order_acquire);
  memmove(at(dst, dst_offset), at(with, with_offset), n * SIZE[type]);
  nccl_mesh_combine(type, op, at(dst, dst_offset), at(src, src_offset), n);
  return 0;
}

int metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (!walked(type)) return EINVAL;
  if (halted(v, polls) || !n) return 0;
  nccl_mesh_premultiply(type, at(dst, dst_offset), at(src, src_offset), scalar, n);
  return 0;
}

int metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (type < 0 || type > 5) return EINVAL;
  if (halted(v, polls) || !n) return 0;
  nccl_mesh_truncdiv(type, at(dst, dst_offset), at(src, src_offset), n, divisor);
  return 0;
}

int metal_publish(struct metal_program *program, const struct metal_release *r) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (!stopped(v, polls)) step_release(r);
  return 0;
}

int metal_publish_filler(struct metal_program *program, const struct metal_release *r) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (stopped(v, polls)) return 0;
  memset(at(r->buffer, r->slot), 0, sizeof(uint64_t));
  step_release(r);
  return 0;
}

int metal_spin(struct metal_program *program, const struct metal_wait *w) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, w->violation, &polls);
  if (!halted(v, polls)) step_await(w, v, polls);
  return 0;
}

int metal_send_small(struct metal_program *program, void *src, size_t src_offset, size_t bytes, const struct metal_release *r) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (stopped(v, polls)) return 0;
  if (bytes < sizeof(uint64_t)) memset(at(r->buffer, r->slot), 0, sizeof(uint64_t));
  memmove(at(r->buffer, r->slot), at(src, src_offset), bytes);
  step_release(r);
  return 0;
}

int metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, w->violation, &polls);
  if (type >= 0 && !walked(type)) return EINVAL;
  if (halted(v, polls) || !step_await(w, v, polls) || !n) return 0;
  if (type < 0) memmove(at(dst, dst_offset), at(w->buffer, w->slot), n);
  else nccl_mesh_combine(type, op, at(dst, dst_offset), at(w->buffer, w->slot), n);
  return 0;
}

int metal_send_deferred(struct metal_program *program, size_t bytes, const struct metal_release *r, uint64_t bound[6]) {
  (void)program; (void)bytes; (void)r; (void)bound;
  snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's");
  return EINVAL;
}

int metal_land_deferred(struct metal_program *program, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w,
                        uint64_t bound[6]) {
  (void)program; (void)op; (void)dst; (void)dst_offset; (void)n; (void)w; (void)bound;
  snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's");
  return EINVAL;
}

/* nccl-mesh-metal.m's counted kernels (counted_start .. counted_own), one step each */
static uint64_t counted_rows(const int64_t *c, uint64_t first, uint64_t count) {
  uint64_t rows = 0;
  for (uint64_t k = 0; k < count; k++) rows += c[first + k] > 0 ? (uint64_t)c[first + k] : 0;
  return rows;
}
static uint64_t least(uint64_t a, uint64_t b) { return a < b ? a : b; }
int metal_counted(struct metal_program *program, int kind, const struct metal_counted *k) {
  if (kind < METAL_COUNTED_START || kind > METAL_COUNTED_OWN) return EINVAL;
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, kind == METAL_COUNTED_ARRIVED ? k->a : NULL, &polls);
  if (kind == METAL_COUNTED_SEND ? stopped(v, polls) : halted(v, polls)) return 0;
  unsigned char *d = k->d, *s = k->s, *a = k->a;
  atomic_thread_fence(memory_order_acquire);
  if (kind == METAL_COUNTED_START) {
    int64_t *got = (int64_t *)(d + k->dst);
    const int64_t *c = (const int64_t *)(s + k->src);
    for (uint64_t i = 0; i < k->bytes; i++) got[i] = c[k->first + i];
    uint64_t *tail = (uint64_t *)(d + k->aux);
    tail[0] = counted_rows(c, 0, k->first);
    tail[1] = counted_rows(c, 0, k->scalar);
    tail[2] = 0;
  } else if (kind == METAL_COUNTED_CHECK) {
    if (counted_rows((const int64_t *)d, k->first, k->entries) > k->scalar) ((uint64_t *)(d + k->dst))[2] = 1;
  } else if (kind == METAL_COUNTED_ARRIVED) {
    const uint64_t *tail = (const uint64_t *)(s + k->src) + k->first;
    const int over = counted_rows((const int64_t *)(s + k->src), 0, k->bytes) > k->aux || tail[1] > k->scalar || tail[2] != 0;
    _Atomic uint64_t *word = (_Atomic uint64_t *)(d + k->dst);
    if (over && a) atomic_store_explicit((_Atomic uint64_t *)a + 5, 1, memory_order_release);
    if (!atomic_load_explicit(word, memory_order_acquire)) atomic_store_explicit(word, over ? 2 : 1, memory_order_release);
  } else if (kind == METAL_COUNTED_OWN) {
    const int64_t *c = (const int64_t *)(a + k->aux);
    const uint64_t to = counted_rows(c, 0, k->first) * k->row, from = ((const uint64_t *)(a + k->aux))[k->scalar] * k->row;
    uint64_t bytes = counted_rows(c, k->first, k->entries) * k->row;
    bytes = to >= k->limit || from >= k->bytes ? 0 : least(bytes, least(k->limit - to, k->bytes - from));
    memmove(d + k->dst + to, s + k->src + from, bytes);
  } else {
    const int64_t *c = (const int64_t *)(a + k->aux);
    const uint64_t place = counted_rows(c, 0, k->first) * k->row + k->scalar, rows = counted_rows(c, k->first, k->entries) * k->row;
    const uint64_t bytes = rows > k->scalar ? least(rows - k->scalar, k->bytes) : 0, length = place >= k->limit ? 0 : least(bytes, k->limit - place);
    if (kind == METAL_COUNTED_SEND) {
      if (length < sizeof(uint64_t)) memset(d + k->dst, 0, sizeof(uint64_t));
      memmove(d + k->dst, s + k->src + place, length);
      atomic_thread_fence(memory_order_release);
    }
    else memmove(d + k->dst + place, s + k->src, length);
  }
  return 0;
}

const char *metal_steps(void) { return ""; }

void metal_keep(struct metal_program *program, void *object) {
  if (!hold(object)) return;
  if (program->count == program->capacity) {
    const size_t grown = program->capacity ? 2 * program->capacity : 8;
    void **bigger = realloc(program->kept, grown * sizeof *bigger);
    if (!bigger) { metal_release(object); return; }
    program->kept = bigger;
    program->capacity = grown;
  }
  program->kept[program->count++] = object;
}

/* the program's end: its event signalled (its steps have run), what it kept released */
void metal_end(struct metal_program *program, void *event, uint64_t value) {
  if (event) atomic_store_explicit(&((struct host_event *)event)->value, value, memory_order_release);
  for (size_t i = 0; i < program->count; i++) metal_release(program->kept[i]);
  free(program->kept);
  free(program);
}

void metal_collect(void *event) { (void)event; }

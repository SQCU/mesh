/* The host executor (mesh 800fe6bc^, on HEAD's libnccl-mesh; design/bridge-dataflow.md#the-loopback-fabric): the walk's
   kernels (nccl-mesh-metal.h) and the transport's Metal wrappers (mesh-metal.h) on the CPU, so libnccl-mesh's Metal
   path runs on host threads: N ranks on one host over mesh-flow-loop's bridges, with no GPU kernel anywhere (several
   ranks' spinning kernels on one GPU have no forward-progress guarantee: the no-cross-queue-spin rule).  Each kernel
   runs as the walk encodes it, in the walk's order, its body transcribed from nccl-mesh-metal.m's: copies and combines
   in host memory (the host path's arithmetic, nccl-mesh.c combine_into, premultiply, truncdiv and partials_layout, which
   nccl-mesh.c exports hidden to this file where MESH_HOST_EXECUTOR is defined), publications release-stored, a wait for
   a landing a poll of its word (mesh.metal mesh_remote_wait: until it reaches 1 + the cycle or the session's canceller
   stores ~0; bounded by the program's polls where it has violation words, mesh_remote_wait_bounded, past which the
   position, the channel and the bound go to violation words [2], [3] and [4] and every later step of the program does
   nothing), a landed wait's positions checked against the filler protocol (violation words [0] and [1]).  An unbounded
   wait that sees nothing for twice MESH_SILENT_NS (the session's canceller ends it first) fails ETIMEDOUT.  A buffer is a host
   address (metal_wrap returns its address; metal_contents it); what metal_scratch and metal_event made is counted and
   freed with its last release.  The Metal 4 form's own steps (a caller's kernel running a step: metal_send_deferred,
   metal_land_deferred) are refused as the classic form refuses them.  Built as .build/host/libnccl-mesh.dylib (make
   host), the GPU library's install name, beside .build/host/libmesh.dylib (the loop client's libmesh). */
#include "mesh-metal.h"
#include "mesh-collective.h"
#include "nccl-mesh-metal.h"
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
void nccl_mesh_partials(int rows_out, float *dst, const float *src, size_t rows, size_t dim);

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

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls) {
  (void)command_buffer; (void)encoder;
  struct metal_program *program = calloc(1, sizeof *program);
  if (program) { program->violation = violation; program->polls = polls; }
  return program;
}

/* no Metal 4 on the host: the walk runs its classic steps here */
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls) {
  (void)table; (void)residency;
  return metal_begin(NULL, encoder, violation, polls);
}
int metal_metal4(const struct metal_program *program) { (void)program; return 0; }
void metal_resident(struct metal_program *program, void *buffer) { (void)program; (void)buffer; }

/* a step's violation words: those it names (a wait's check), else the program's where its waits are bounded, and the
   bound that holds with them (nccl-mesh-metal.m dispatch_as: p.polls is the program's where there are such words) */
static _Atomic uint64_t *words_of(struct metal_program *program, void *given, uint32_t *polls) {
  _Atomic uint64_t *v = given ? given : program->polls ? program->violation : NULL;
  *polls = v ? program->polls : 0;
  return v;
}
/* mesh_step_enter: a bounded program whose violation word [2] holds a failure does nothing more */
static int halted(_Atomic uint64_t *v, uint32_t polls) { return polls && v && atomic_load_explicit(v + 2, memory_order_acquire); }

/* mesh_step_wait: the word until it reaches 1 + cycle (or ~0), at most `polls` looks where bounded */
static uint64_t step_wait(_Atomic uint64_t *word, uint64_t cycle, uint32_t polls, int *timed_out) {
  const uint64_t target = 1 + cycle, deadline = now_ns() + 2 * MESH_SILENT_NS;
  uint64_t seen = atomic_load_explicit(word, memory_order_acquire);
  for (uint64_t k = 1; seen < target; k++) {
    if (polls && k >= polls) break;
    /* N ranks share the host's cores with N bridges: a wait that runs long yields them */
    if (k > 4096) usleep(20);
    else if (k > 256) sched_yield();
    if (!polls && !(k & 63) && now_ns() > deadline) {
      snprintf(error_, sizeof error_, "a wait for a landing: the word stayed at %llu below %llu", (unsigned long long)seen,
               (unsigned long long)target);
      *timed_out = 1;
      break;
    }
    seen = atomic_load_explicit(word, memory_order_acquire);
  }
  return seen;
}

/* mesh_step_check: a word short of its target past the bound recorded in violation words [2], [3] and [4]; else a
   landed word's positions checked against their roles (nccl-mesh-metal.h struct metal_check) */
static void step_check(const struct metal_check *check, _Atomic uint64_t *v, uint32_t polls, uint64_t status, uint64_t cycle) {
  if (!v) return;
  if (polls && status != UINT64_MAX && status < 1 + cycle) {
    if (!atomic_load_explicit(v + 2, memory_order_acquire)) {
      atomic_store_explicit(v + 3, check ? check->channel : 0, memory_order_relaxed);
      atomic_store_explicit(v + 4, polls, memory_order_relaxed);
      atomic_store_explicit(v + 2, (check ? check->at : 0) + 1, memory_order_release);
    }
    return;
  }
  if (status == UINT64_MAX || !check || !check->count) return;
  atomic_thread_fence(memory_order_acquire);
  for (uint32_t c = 0; c < check->count; c++) {
    const uint64_t t = check->first + c, data = (check->mask >> c) & 1;
    uint64_t seen;
    memcpy(&seen, at(check->ring, (t % check->depth) * check->slot), sizeof seen);
    if (data == (seen == METAL_STAMP(t)) && !atomic_load_explicit(v, memory_order_acquire)) {
      atomic_store_explicit(v + 1, (uint64_t)check->channel << 1 | data, memory_order_relaxed);
      atomic_store_explicit(v, t + 1, memory_order_release);
    }
  }
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (halted(v, polls) || !bytes) return 0;
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

int metal_publish(struct metal_program *program, void *cells, size_t offset, uint64_t argument) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (halted(v, polls)) return 0;
  atomic_thread_fence(memory_order_release);
  atomic_store_explicit((_Atomic uint64_t *)at(cells, offset), argument, memory_order_release);
  return 0;
}

int metal_publish_filler(struct metal_program *program, void *cells, size_t offset, uint64_t argument, void *slot, size_t slot_offset,
                         uint64_t position) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (halted(v, polls)) return 0;
  const uint64_t stamp = METAL_STAMP(position);
  memcpy(at(slot, slot_offset), &stamp, sizeof stamp);
  return metal_publish(program, cells, offset, argument);
}

int metal_spin(struct metal_program *program, void *words, size_t offset, uint64_t cycle, const struct metal_check *check) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, check ? check->violation : NULL, &polls);
  if (halted(v, polls)) return 0;
  int timed_out = 0;
  const uint64_t status = step_wait((_Atomic uint64_t *)at(words, offset), cycle, polls, &timed_out);
  if (timed_out) return ETIMEDOUT;
  step_check(check, v, polls, status, cycle);
  return 0;
}

int metal_send_small(struct metal_program *program, void *slot, size_t slot_offset, void *src, size_t src_offset, size_t bytes,
                     void *cells, size_t cell_offset, uint64_t value) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (halted(v, polls)) return 0;
  memmove(at(slot, slot_offset), at(src, src_offset), bytes);
  return metal_publish(program, cells, cell_offset, value);
}

int metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
               void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, check ? check->violation : NULL, &polls);
  if (type >= 0 && !walked(type)) return EINVAL;
  if (halted(v, polls)) return 0;
  int timed_out = 0;
  const uint64_t status = step_wait((_Atomic uint64_t *)at(words, word_offset), cycle, polls, &timed_out);
  if (timed_out) return ETIMEDOUT;
  step_check(check, v, polls, status, cycle);
  if (halted(v, polls) || !n) return 0;
  atomic_thread_fence(memory_order_acquire);
  if (type < 0) memmove(at(dst, dst_offset), at(slot, slot_offset), n);
  else nccl_mesh_combine(type, op, at(dst, dst_offset), at(slot, slot_offset), n);
  return 0;
}

int metal_send_deferred(struct metal_program *program, void *slot, size_t slot_offset, size_t bytes, void *cells, size_t cell_offset,
                        uint64_t value, uint64_t bound[6]) {
  (void)program; (void)slot; (void)slot_offset; (void)bytes; (void)cells; (void)cell_offset; (void)value; (void)bound;
  snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's");
  return EINVAL;
}

int metal_land_deferred(struct metal_program *program, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
                        void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check, uint64_t bound[6]) {
  (void)program; (void)op; (void)dst; (void)dst_offset; (void)slot; (void)slot_offset; (void)n; (void)words; (void)word_offset;
  (void)cycle; (void)check; (void)bound;
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
  if (halted(v, polls)) return 0;
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
    if (kind == METAL_COUNTED_SEND) { memmove(d + k->dst, s + k->src + place, length); atomic_thread_fence(memory_order_release); }
    else memmove(d + k->dst + place, s + k->src, length);
  }
  return 0;
}

/* nccl-mesh-metal.m's partials_rows and partials_columns: nccl-mesh.c's partials_layout */
int metal_partials(struct metal_program *program, int rows_out, void *dst, size_t dst_offset, void *src, size_t src_offset, uint64_t rows,
                   uint64_t dim) {
  uint32_t polls;
  _Atomic uint64_t *v = words_of(program, NULL, &polls);
  if (!halted(v, polls)) nccl_mesh_partials(rows_out, (float *)at(dst, dst_offset), (const float *)at(src, src_offset), rows, dim);
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

/* mesh-metal.m's transport, its words and cells as host addresses (mesh-collective.h mesh_host_inputs lays the words) */
int mesh_metal_transport_create(struct mesh_ctx *context, void *device, struct mesh_metal_transport *transport) {
  (void)device;
  *transport = (struct mesh_metal_transport){0};
  transport->publication = (char *)context->M + context->send_off;
  const int status = mesh_host_inputs(context);
  if (status) return status;
  transport->inputs = (char *)context->M + context->inputs_off;
  transport->cancel = (void *)((char *)context->M + mesh_client_at(context->M, context->client)->tx[0].cancel);
  return 0;
}

struct mesh_metal_input mesh_metal_receive_prepare(struct mesh_ctx *context, struct mesh_metal_transport *transport, struct mesh_section operand) {
  const struct mesh_delivery *delivery = mesh_delivery_at(context->M, operand.first + mesh_row_chunks(context->M, operand.first, operand.bytes) - 1);
  return (struct mesh_metal_input){(uint64_t)((uintptr_t)context->M + delivery->device_input - (uintptr_t)transport->inputs), delivery->device_stride};
}

void mesh_metal_transport_destroy(struct mesh_metal_transport *transport) { *transport = (struct mesh_metal_transport){0}; }

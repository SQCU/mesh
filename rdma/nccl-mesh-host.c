/* The walk's kernels (nccl-mesh-metal.h) and the transport's Metal wrappers (mesh-metal.h) on the CPU, for running
   libnccl-mesh's GPU walk on host threads: N ranks on one host over mesh-flow-loop's bridges, with no GPU kernel
   anywhere (several ranks' spinning kernels on one GPU have no forward-progress guarantee).  Each kernel runs as the
   walk encodes it, in the walk's order: copies and combines in host memory (the host path's arithmetic,
   nccl_mesh_combine), publications release-stored, a wait for a landing a bounded poll of its word.  A buffer is a host
   address; metal_wrap returns its address, and only what metal_scratch and metal_event made is released.  Built as
   .build/host/libnccl-mesh.dylib (make host), the same install name as the GPU library, so a process takes it with
   DYLD_LIBRARY_PATH. */
#include "mesh-metal.h"
#include "nccl-mesh-metal.h"
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

void nccl_mesh_combine(int t, int combine, void *dst, const void *src, size_t n);
void nccl_mesh_premultiply(int t, void *dst, const void *src, uint64_t bits, size_t n);
void nccl_mesh_truncdiv(int t, void *dst, const void *src, size_t n, uint64_t divisor);

static char error_[512];
static pthread_mutex_t owned_lock = PTHREAD_MUTEX_INITIALIZER;
struct held { void *object; int count; };
static struct held *owned_;
static size_t owned_count_, owned_capacity_;
struct metal_program { void **kept; size_t count, capacity; };
struct host_event { _Atomic uint64_t value; };

/* what metal_scratch and metal_event made, counted: the caller's reference and each program's that keeps it (a
   command buffer's, on the GPU), freed with the last */
static void own(void *object) {
  pthread_mutex_lock(&owned_lock);
  if (owned_count_ == owned_capacity_) {
    owned_capacity_ = owned_capacity_ ? 2 * owned_capacity_ : 64;
    owned_ = realloc(owned_, owned_capacity_ * sizeof *owned_);
  }
  owned_[owned_count_++] = (struct held){object, 1};
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

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* the bound of a wait: twice the progress thread's (MESH_REMOTE_BOUND, default 10 s), which cancels a silent link
   (its words ~0, which ends every wait on them) */
static double bound(void) {
  const char *value = getenv("MESH_REMOTE_BOUND");
  const double seconds = value && atof(value) > 0 ? atof(value) : 10.0;
  return 2 * seconds;
}

static int await(_Atomic uint64_t *word, uint64_t expected, const char *what) {
  if (atomic_load_explicit(word, memory_order_acquire) >= expected) return 0;
  const double deadline = now_s() + bound();
  for (uint64_t polls = 0; atomic_load_explicit(word, memory_order_acquire) < expected; polls++) {
    /* N ranks share the host's cores with N bridges: a wait that runs long yields them */
    if (polls > 4096) usleep(20);
    else if (polls > 256) sched_yield();
    if ((polls & 63) == 0 && now_s() > deadline) {
      snprintf(error_, sizeof error_, "%s: the word stayed at %llu below %llu", what, (unsigned long long)atomic_load(word),
               (unsigned long long)expected);
      return ETIMEDOUT;
    }
  }
  return 0;
}

static unsigned char *at(void *buffer, size_t offset) { return (unsigned char *)buffer + offset; }

/* MESH_HOST_TRACE=1: each kernel a line on stderr, as it runs (its buffers' addresses, offsets, sizes, a wait's word,
   expected and seen values) */
static void trace(const char *format, ...) __attribute__((format(printf, 1, 2)));
static void trace(const char *format, ...) {
  static int on = -1;
  if (on < 0) on = getenv("MESH_HOST_TRACE") && atoi(getenv("MESH_HOST_TRACE")) > 0;
  if (!on) return;
  va_list arguments;
  va_start(arguments, format);
  vfprintf(stderr, format, arguments);
  va_end(arguments);
}

void *metal_device(void) { return (void *)1; }
const char *metal_error(void) { return error_; }
void *metal_wrap(void *address, size_t bytes) { (void)bytes; return address; }

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

void metal_signal(void *event, uint64_t value) { atomic_store_explicit(&((struct host_event *)event)->value, value, memory_order_release); }
uint64_t metal_signaled(void *event) { return atomic_load_explicit(&((struct host_event *)event)->value, memory_order_acquire); }

void metal_release(void *object) {
  pthread_mutex_lock(&owned_lock);
  for (size_t i = 0; i < owned_count_; i++)
    if (owned_[i].object == object) {
      if (!--owned_[i].count) { owned_[i] = owned_[--owned_count_]; free(object); }
      break;
    }
  pthread_mutex_unlock(&owned_lock);
}

struct metal_program *metal_begin(void *command_buffer, void *encoder) {
  (void)command_buffer; (void)encoder;
  return calloc(1, sizeof(struct metal_program));
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  (void)program;
  trace("copy %d %p+%zu <- %p+%zu %zu\n", mode, dst, dst_offset, src, src_offset, bytes);
  if (mode == METAL_LAND) atomic_thread_fence(memory_order_acquire);
  memmove(at(dst, dst_offset), at(src, src_offset), bytes);
  if (mode == METAL_SEND) atomic_thread_fence(memory_order_release);
  return 0;
}

int metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n) {
  (void)program;
  trace("combine %d %d %p+%zu <- %p+%zu %zu\n", type, op, dst, dst_offset, src, src_offset, n);
  atomic_thread_fence(memory_order_acquire);
  nccl_mesh_combine(type, op, at(dst, dst_offset), at(src, src_offset), n);
  return 0;
}

int metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar) {
  (void)program;
  nccl_mesh_premultiply(type, at(dst, dst_offset), at(src, src_offset), scalar, n);
  return 0;
}

int metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor) {
  (void)program;
  nccl_mesh_truncdiv(type, at(dst, dst_offset), at(src, src_offset), n, divisor);
  return 0;
}

int metal_publish(struct metal_program *program, void *cells, size_t offset, uint64_t argument) {
  (void)program;
  trace("publish %p+%zu %llu\n", cells, offset, (unsigned long long)argument);
  atomic_thread_fence(memory_order_release);
  atomic_store_explicit((_Atomic uint64_t *)at(cells, offset), argument, memory_order_release);
  return 0;
}

int metal_spin(struct metal_program *program, void *words, size_t offset, uint64_t expected) {
  (void)program;
  _Atomic uint64_t *word = (_Atomic uint64_t *)at(words, offset);
  trace("spin %p+%zu %llu seen %llu\n", words, offset, (unsigned long long)expected, (unsigned long long)atomic_load(word));
  return await(word, expected, "a wait for a landing");
}

int metal_send_small(struct metal_program *program, void *slot, size_t slot_offset, void *src, size_t src_offset, size_t bytes,
                     void *cells, size_t cell_offset, uint64_t value) {
  metal_copy(program, METAL_SEND, slot, slot_offset, src, src_offset, bytes);
  return metal_publish(program, cells, cell_offset, value);
}

int metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
               void *words, size_t word_offset, uint64_t expected) {
  int status = metal_spin(program, words, word_offset, expected);
  if (status) return status;
  if (type < 0) return metal_copy(program, METAL_LAND, dst, dst_offset, slot, slot_offset, n);
  return metal_combine(program, type, op, dst, dst_offset, slot, slot_offset, n);
}

void metal_wait(struct metal_program *program, void *event, uint64_t value) {
  (void)program;
  await(&((struct host_event *)event)->value, value, "a wait for an event");
}

void metal_keep(struct metal_program *program, void *object) {
  if (!hold(object)) return;
  if (program->count == program->capacity) {
    program->capacity = program->capacity ? 2 * program->capacity : 8;
    program->kept = realloc(program->kept, program->capacity * sizeof *program->kept);
  }
  program->kept[program->count++] = object;
}

void metal_end(struct metal_program *program, void *event, uint64_t value) {
  if (event) metal_signal(event, value);
  for (size_t i = 0; i < program->count; i++) metal_release(program->kept[i]);
  free(program->kept);
  free(program);
}

void metal_collect(void *event) { (void)event; }

/* mesh-metal.m's transport, its words and cells as host addresses (the same layout arithmetic) */
int mesh_metal_transport_create(struct mesh_ctx *context, void *device, struct mesh_metal_transport *transport) {
  (void)device;
  *transport = (struct mesh_metal_transport){0};
  transport->publication = (char *)context->M + context->send_off;
  struct hdr *m = context->M;
  struct mesh_section stop, words;
  uint64_t inputs = 0, frames[m->links ? m->links : 1];
  for (uint32_t p = 0; p < m->links; p++) {
    inputs = (inputs + 15) & ~UINT64_C(15);
    struct mesh_tx *tx = (void *)mesh_events(m, mesh_notice_queue(m, context->client, p));
    frames[p] = 0;
    for (uint32_t q = p * m->qps; q < (p + 1) * m->qps; q++) {
      struct mesh_transfer *in = mesh_transfers(m, context->client, q, MESH_RECEIVE);
      for (uint32_t i = 0; i < atomic_load(mesh_order_length(m, context->client, q, MESH_RECEIVE)); i++)
        frames[p] += (uint64_t)mesh_row_chunks(m, in[i].local_row, in[i].bytes) * in[i].count * mesh_transfer_active(in + i, tx->invocations);
    }
    inputs += frames[p];
  }
  int status = mesh_section_create(context, sizeof(_Atomic uint64_t) * (inputs ? inputs : 1), 1, 0, &words);
  if (status) return status;
  void *input = mesh_section_address(context, words, 0);
  memset(input, 0, words.bytes);
  transport->inputs = input;
  status = mesh_section_create(context, sizeof(struct mesh_cancellation) + m->links * sizeof(struct mesh_cancel_range), 1, 0, &stop);
  if (status) return status;
  struct mesh_cancellation *address = mesh_section_address(context, stop, 0);
  memset(address, 0, stop.bytes);
  uint64_t first = 0;
  for (uint32_t p = 0; p < m->links; p++) {
    first = (first + 15) & ~UINT64_C(15);
    struct mesh_tx *tx = (void *)mesh_events(m, mesh_notice_queue(m, context->client, p));
    tx->cancel = (uintptr_t)address - (uintptr_t)m;
    address->ranges[p] = (struct mesh_cancel_range){.offset = (uintptr_t)input - (uintptr_t)m + first * sizeof(_Atomic uint64_t), .count = frames[p]};
    for (uint32_t q = p * m->qps; q < (p + 1) * m->qps; q++) {
      struct mesh_transfer *in = mesh_transfers(m, context->client, q, MESH_RECEIVE);
      for (uint32_t i = 0; i < atomic_load(mesh_order_length(m, context->client, q, MESH_RECEIVE)); i++) {
        uint32_t chunks = mesh_row_chunks(m, in[i].local_row, in[i].bytes);
        for (uint32_t s = 0; s < in[i].count; s++)
          for (uint32_t k = 0; k < chunks; k++) {
            struct mesh_publication *delivery = mesh_publication_at(m, in[i].local_row + s * in[i].stride + k);
            delivery->device_input = (uintptr_t)input - (uintptr_t)m + sizeof(_Atomic uint64_t) * (first + in[i].first + (uint64_t)s * chunks + k);
            delivery->device_stride = sizeof(_Atomic uint64_t) * (uint64_t)in[i].count * chunks;
          }
      }
    }
    first += frames[p];
  }
  transport->cancel = address;
  return 0;
}

int mesh_metal_receive_prepare(struct mesh_ctx *context, struct mesh_metal_transport *transport, struct mesh_section operand,
                               struct mesh_metal_input *input) {
  struct mesh_publication *delivery = mesh_publication_at(context->M, operand.first + mesh_row_chunks(context->M, operand.first, operand.bytes) - 1);
  *input = (struct mesh_metal_input){.completion = transport->inputs,
                                     .offset = (uintptr_t)context->M + delivery->device_input - (uintptr_t)transport->inputs,
                                     .stride = delivery->device_stride};
  return 0;
}

void mesh_metal_transport_destroy(struct mesh_metal_transport *transport) { *transport = (struct mesh_metal_transport){0}; }

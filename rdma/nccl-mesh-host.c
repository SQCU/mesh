/* The host executor (mesh 800fe6bc^, on HEAD's libnccl-mesh; design/transport.md): the walk's kernels
   (nccl-mesh-metal.h) on the CPU, so libnccl-mesh's Metal path runs on host threads: N ranks on one host over
   mesh-flow-loop's bridges, with no GPU kernel anywhere (several ranks' spinning kernels on one GPU have no
   forward-progress guarantee: the no-cross-queue-spin rule).  Each kernel runs as the walk encodes it, in the walk's
   order, its body transcribed from nccl-mesh-kernels.h's: copies and combines in host memory (the host path's arithmetic,
   nccl-mesh.c combine_into, premultiply and truncdiv, which nccl-mesh.c exports hidden to this file where
   MESH_HOST_EXECUTOR is defined), a run's ready words release-stored, a wait for a landing a poll of its run's last
   unit's landed word (mesh.metal mesh_wait_bounded: until it reaches 1 + the cycle or the bridge stores ~0, for at most
   the program's polls looks, 0 or more than NCCL_MESH_POLLS being NCCL_MESH_POLLS, and at most MESH_PLATFORM_NS,
   design/transport.md), its failure (the bound reached, the link lost; a revoke named where the stop word [6] is set)
   recorded in the failure words [0], [3], [4] and [2], which the library reads after the program, every later step
   running as made; a step naming no words records in the program's (metal_gate), as nccl-mesh-metal.m dispatch_as binds them.
   A buffer is a host address (metal_wrap returns its address; metal_contents it); what metal_scratch, metal_shared and
   metal_event made is counted and freed with its last release.  A step run by a caller's kernel (metal_send_deferred,
   metal_land_deferred: a fused crossing's, on either form) is refused, since no kernel of a caller's runs here.  Built as
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

static pthread_mutex_t owned_lock = PTHREAD_MUTEX_INITIALIZER;
struct held { void *object; int count; };
static struct held *owned_;
static size_t owned_count_, owned_capacity_;
struct recording;
struct metal_program { void **kept; size_t count, capacity; _Atomic uint64_t *violation; uint32_t polls; struct recording *recording; };
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

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *residency, void *violation, uint32_t polls, const char **message) {
  (void)command_buffer; (void)encoder; (void)residency;
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) { *message = "out of memory"; return NULL; }
  program->violation = violation;
  program->polls = bound_of(polls);
  return program;
}

/* no Metal 4 on the host: the walk runs its classic steps here */
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls, const char **message) {
  (void)table; (void)residency;
  return metal_begin(NULL, encoder, NULL, violation, polls, message);
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
/* mesh_step_wait: the word until it reaches 1 + cycle (or ~0), at most `polls` looks (0: one) and at most
   MESH_PLATFORM_NS (a host look being far slower than a GPU load, design/transport.md), reading no other word */
static uint64_t step_wait(_Atomic uint64_t *word, uint64_t cycle, uint32_t polls) {
  const uint64_t target = 1 + cycle, deadline = now_ns() + MESH_PLATFORM_NS;
  uint64_t seen = atomic_load_explicit(word, memory_order_acquire);
  for (uint64_t k = 1; seen < target && k < polls; k++) {
    if (!(k & 63) && now_ns() > deadline) break;
    /* N ranks share the host's cores with N bridges: a wait that runs long yields them */
    if (k > 4096) usleep(20);
    else if (k > 256) sched_yield();
    seen = atomic_load_explicit(word, memory_order_acquire);
  }
  return seen;
}

/* mesh_step_check: a wait's failure recorded where none is yet ([2] clear): a revoke, the link lost (~0) or the bound reached;
   a landing is none, whatever its bytes */
static void step_check(const struct metal_wait *w, _Atomic uint64_t *v, uint32_t polls, uint64_t status) {
  if (!v || !polls || atomic_load_explicit(v + 2, memory_order_acquire)) return;
  const uint64_t cause = atomic_load_explicit(v + 6, memory_order_acquire) ? METAL_REVOKED : status == UINT64_MAX ? METAL_LOST
                       : status < 1 + w->cycle ? METAL_BOUND : 0;
  if (!cause) return;
  atomic_store_explicit(v, cause, memory_order_relaxed);
  atomic_store_explicit(v + 3, w->channel, memory_order_relaxed);
  atomic_store_explicit(v + 4, polls, memory_order_relaxed);
  atomic_store_explicit(v + 2, w->at + 1, memory_order_release);
}

/* a step's wait (step_wait) and its check (step_check) */
static void step_await(const struct metal_wait *w, _Atomic uint64_t *v, uint32_t polls) {
  step_check(w, v, polls, step_wait((_Atomic uint64_t *)at(w->buffer, w->word), w->cycle, polls));
  atomic_thread_fence(memory_order_acquire);
}

/* mesh_step_release: the run's ready words stored, each after every store before it */
static void step_release(const struct metal_release *r) {
  atomic_thread_fence(memory_order_release);
  for (uint32_t u = 0; u < r->count; u++)
    atomic_store_explicit((_Atomic uint64_t *)at(r->buffer, r->word) + u, r->value, memory_order_release);
}

/* a step as the walk encodes it: its kind and arguments, and the failure words and bound it names (words_of, read
   where it is made, as a dispatch binds them where it is encoded) */
enum { OP_COPY, OP_COMBINE, OP_COMBINE3, OP_PREMUL, OP_TRUNCDIV, OP_PUBLISH, OP_SPIN, OP_SEND_SMALL, OP_LAND, OP_COUNTED };
struct op {
  int kind, mode, type, op;
  void *dst, *src, *with;
  size_t dst_offset, src_offset, with_offset, n;
  uint64_t scalar;
  struct metal_release release;
  struct metal_wait wait;
  struct metal_counted counted;
  _Atomic uint64_t *v;
  uint32_t polls;
};

/* nccl-mesh-kernels.h's counted kernels (counted_start .. counted_own), one step each */
static uint64_t counted_rows(const int64_t *c, uint64_t first, uint64_t count) {
  uint64_t rows = 0;
  for (uint64_t k = 0; k < count; k++) rows += c[first + k] > 0 ? (uint64_t)c[first + k] : 0;
  return rows;
}
static uint64_t least(uint64_t a, uint64_t b) { return a < b ? a : b; }
static void run_counted(int kind, const struct metal_counted *k) {
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
    const int failed = a && atomic_load_explicit((_Atomic uint64_t *)a + 2, memory_order_acquire);
    if (over && a) atomic_store_explicit((_Atomic uint64_t *)a + 5, 1, memory_order_release);
    if (!atomic_load_explicit(word, memory_order_acquire)) atomic_store_explicit(word, over || failed ? 2 : 1, memory_order_release);
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
      memmove(d + k->dst, s + k->src + place, length);
      atomic_thread_fence(memory_order_release);
    }
    else memmove(d + k->dst + place, s + k->src, length);
  }
}

/* a step's body, nccl-mesh-kernels.h's kernel of its kind, run as made */
static void run_op(const struct op *o) {
  _Atomic uint64_t *v = o->v;
  const uint32_t polls = o->polls;
  switch (o->kind) {
  case OP_COPY:
    if (!o->n) return;
    if (o->mode == METAL_LAND) atomic_thread_fence(memory_order_acquire);
    memmove(at(o->dst, o->dst_offset), at(o->src, o->src_offset), o->n);
    if (o->mode == METAL_SEND) atomic_thread_fence(memory_order_release);
    return;
  case OP_COMBINE:
    if (!o->n) return;
    atomic_thread_fence(memory_order_acquire);
    nccl_mesh_combine(o->type, o->op, at(o->dst, o->dst_offset), at(o->src, o->src_offset), o->n);
    return;
  case OP_COMBINE3:
    if (!o->n) return;
    atomic_thread_fence(memory_order_acquire);
    memmove(at(o->dst, o->dst_offset), at(o->with, o->with_offset), o->n * SIZE[o->type]);
    nccl_mesh_combine(o->type, o->op, at(o->dst, o->dst_offset), at(o->src, o->src_offset), o->n);
    return;
  case OP_PREMUL:
    if (o->n) nccl_mesh_premultiply(o->type, at(o->dst, o->dst_offset), at(o->src, o->src_offset), o->scalar, o->n);
    return;
  case OP_TRUNCDIV:
    if (o->n) nccl_mesh_truncdiv(o->type, at(o->dst, o->dst_offset), at(o->src, o->src_offset), o->n, o->scalar);
    return;
  case OP_PUBLISH: step_release(&o->release); return;
  case OP_SPIN: step_await(&o->wait, v, polls); return;
  case OP_SEND_SMALL:
    memmove(at(o->release.buffer, o->release.slot), at(o->src, o->src_offset), o->n);
    step_release(&o->release);
    return;
  case OP_LAND:
    step_await(&o->wait, v, polls);
    if (!o->n) return;
    if (o->type < 0) memmove(at(o->dst, o->dst_offset), at(o->wait.buffer, o->wait.slot), o->n);
    else nccl_mesh_combine(o->type, o->op, at(o->dst, o->dst_offset), at(o->wait.buffer, o->wait.slot), o->n);
    return;
  case OP_COUNTED: run_counted(o->mode, &o->counted); return;
  }
}

/* the tracks being recorded (nccl-mesh-metal.h metal_record): each item's position, credit (`need`), dependencies,
   steps, and the landing a consumption awaits (its first wait) */
struct dep { uint32_t other; uint64_t count; };
struct item { uint64_t position, need; struct dep *deps; uint32_t ndeps; struct op *ops; size_t nops, room; int awaits; struct metal_wait wait;
              _Atomic uint64_t *v; uint32_t polls; };
struct cursor { struct item *item; size_t count, room, next; };
struct track { struct cursor cursor[2]; };
struct recording { struct track *track; uint32_t tracks, at; int cursor, on; };
static struct recording *recording_of(struct metal_program *program);

static void *grown(void *items, size_t *room, size_t count, size_t size) {
  if (count < *room) return items;
  const size_t more = *room ? 2 * *room : 8;
  void *bigger = realloc(items, more * size);
  if (bigger) *room = more;
  return bigger;
}

/* a step made: its failure words and bound read now, then run, or appended to the item being recorded */
static struct metal_result step(struct metal_program *program, struct op *o, void *given) {
  o->v = words_of(program, given, &o->polls);
  struct recording *r = recording_of(program);
  if (!r || !r->on) { run_op(o); return METAL_DONE; }
  struct cursor *c = r->track[r->at].cursor + r->cursor;
  struct item *it = c->item + c->count - 1;
  struct op *ops = grown(it->ops, &it->room, it->nops, sizeof *ops);
  if (!ops) return metal_failed(ENOMEM);
  it->ops = ops;
  it->ops[it->nops++] = *o;
  if (!it->v) { it->v = o->v; it->polls = o->polls; }
  if (r->cursor == METAL_CONSUME && !it->awaits && (o->kind == OP_SPIN || o->kind == OP_LAND)) {
    it->awaits = 1; it->wait = o->wait; it->v = o->v; it->polls = o->polls;
  }
  return METAL_DONE;
}

struct metal_result metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  struct op o = {.kind = OP_COPY, .mode = mode, .dst = dst, .dst_offset = dst_offset, .src = src, .src_offset = src_offset, .n = bytes};
  return step(program, &o, NULL);
}

struct metal_result metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                  size_t n) {
  if (!walked(type)) return metal_failed(EINVAL);
  struct op o = {.kind = OP_COMBINE, .type = type, .op = op, .dst = dst, .dst_offset = dst_offset, .src = src, .src_offset = src_offset, .n = n};
  return step(program, &o, NULL);
}

struct metal_result metal_combine3(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                   void *with, size_t with_offset, size_t n) {
  if (!walked(type) || type == METAL_MERGE) return metal_failed(EINVAL);
  struct op o = {.kind = OP_COMBINE3, .type = type, .op = op, .dst = dst, .dst_offset = dst_offset, .src = src, .src_offset = src_offset,
                 .with = with, .with_offset = with_offset, .n = n};
  return step(program, &o, NULL);
}

struct metal_result metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      uint64_t scalar) {
  if (!walked(type) || type == METAL_MERGE) return metal_failed(EINVAL);
  struct op o = {.kind = OP_PREMUL, .type = type, .dst = dst, .dst_offset = dst_offset, .src = src, .src_offset = src_offset, .n = n, .scalar = scalar};
  return step(program, &o, NULL);
}

struct metal_result metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   uint64_t divisor) {
  if (type < 0 || type > 5) return metal_failed(EINVAL);
  struct op o = {.kind = OP_TRUNCDIV, .type = type, .dst = dst, .dst_offset = dst_offset, .src = src, .src_offset = src_offset, .n = n, .scalar = divisor};
  return step(program, &o, NULL);
}

struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  struct op o = {.kind = OP_PUBLISH, .release = *r};
  return step(program, &o, NULL);
}

struct metal_result metal_spin(struct metal_program *program, const struct metal_wait *w) {
  struct op o = {.kind = OP_SPIN, .wait = *w};
  return step(program, &o, w->violation);
}

struct metal_result metal_send_small(struct metal_program *program, void *src, size_t src_offset, size_t bytes, const struct metal_release *r) {
  struct op o = {.kind = OP_SEND_SMALL, .src = src, .src_offset = src_offset, .n = bytes, .release = *r};
  return step(program, &o, NULL);
}

struct metal_result metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w) {
  if (type >= 0 && !walked(type)) return metal_failed(EINVAL);
  struct op o = {.kind = OP_LAND, .type = type, .op = op, .dst = dst, .dst_offset = dst_offset, .n = n, .wait = *w};
  return step(program, &o, w->violation);
}

struct metal_result metal_send_deferred(struct metal_program *program, size_t bytes, const struct metal_release *r, uint64_t bound[6]) {
  (void)program; (void)bytes; (void)r; (void)bound;
  return (struct metal_result){EINVAL, "a step run by its caller's kernel: no kernel of a caller's runs on the host executor"};
}

struct metal_result metal_land_deferred(struct metal_program *program, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w,
                                        uint64_t bound[6]) {
  (void)program; (void)op; (void)dst; (void)dst_offset; (void)n; (void)w; (void)bound;
  return (struct metal_result){EINVAL, "a step run by its caller's kernel: no kernel of a caller's runs on the host executor"};
}

struct metal_result metal_counted(struct metal_program *program, int kind, const struct metal_counted *k) {
  if (kind < METAL_COUNTED_START || kind > METAL_COUNTED_OWN) return metal_failed(EINVAL);
  struct op o = {.kind = OP_COUNTED, .mode = kind, .counted = *k};
  return step(program, &o, kind == METAL_COUNTED_ARRIVED ? k->a : NULL);
}

static struct recording *recording_of(struct metal_program *program) { return program->recording; }

struct metal_result metal_record(struct metal_program *program, uint32_t track, int cursor, uint64_t position, uint64_t need) {
  if (track == METAL_NONE) { if (program->recording) program->recording->on = 0; return METAL_DONE; }
  if (!program->recording && !(program->recording = calloc(1, sizeof *program->recording))) return metal_failed(ENOMEM);
  struct recording *r = program->recording;
  if (track >= r->tracks) {
    struct track *bigger = realloc(r->track, ((size_t)track + 1) * sizeof *bigger);
    if (!bigger) return metal_failed(ENOMEM);
    memset(bigger + r->tracks, 0, ((size_t)track + 1 - r->tracks) * sizeof *bigger);
    r->track = bigger; r->tracks = track + 1;
  }
  struct cursor *c = r->track[track].cursor + cursor;
  struct item *items = grown(c->item, &c->room, c->count, sizeof *items);
  if (!items) return metal_failed(ENOMEM);
  c->item = items;
  c->item[c->count++] = (struct item){.position = position, .need = need};
  r->at = track; r->cursor = cursor; r->on = 1;
  return METAL_DONE;
}

struct metal_result metal_depend(struct metal_program *program, uint32_t other, uint64_t count) {
  struct recording *r = program->recording;
  if (!r || !r->on) return metal_failed(EINVAL);
  struct item *it = r->track[r->at].cursor[r->cursor].item + r->track[r->at].cursor[r->cursor].count - 1;
  struct dep *bigger = realloc(it->deps, ((size_t)it->ndeps + 1) * sizeof *bigger);
  if (!bigger) return metal_failed(ENOMEM);
  it->deps = bigger;
  it->deps[it->ndeps++] = (struct dep){other, count};
  return METAL_DONE;
}

/* whether track tr's next item of `cursor` may run: its own `need` holds (a publication's, its track has consumed
   `need` positions; a consumption's, its track's publications have passed position `need` - 1), every dependency holds
   and, a consumption, its landing came (its word at 1 + the cycle, or ~0) */
static int item_ready(const struct track *tr, int cursor, const uint64_t *count, uint32_t self) {
  const struct cursor *k = tr->cursor + cursor, *out = tr->cursor + METAL_PUBLISH;
  const struct item *it = k->item + k->next;
  if (cursor == METAL_PUBLISH ? count[self] < it->need : out->next < out->count && out->item[out->next].position < it->need) return 0;
  for (uint32_t d = 0; d < it->ndeps; d++)
    if (count[it->deps[d].other] < it->deps[d].count) return 0;
  if (cursor != METAL_CONSUME || !it->awaits) return 1;
  return atomic_load_explicit((_Atomic uint64_t *)at(it->wait.buffer, it->wait.word), memory_order_acquire) >= 1 + it->wait.cycle;
}

/* a track's item that cannot run, its bound reached: its wait's failure recorded (mesh_step_check: the bound, the
   link's ~0, a revoke), or, an item waiting on its credit or a dependency alone, the bound at its position */
static void item_expired(const struct item *it) {
  if (!it->v || !it->polls || atomic_load_explicit(it->v + 2, memory_order_acquire)) return;
  if (it->awaits) { step_check(&it->wait, it->v, it->polls, atomic_load_explicit((_Atomic uint64_t *)at(it->wait.buffer, it->wait.word), memory_order_acquire)); return; }
  atomic_store_explicit(it->v, METAL_BOUND, memory_order_relaxed);
  atomic_store_explicit(it->v + 4, it->polls, memory_order_relaxed);
  atomic_store_explicit(it->v + 2, it->position + 1, memory_order_release);
}

static void tracks_free(struct recording *r) {
  for (uint32_t t = 0; t < r->tracks; t++)
    for (int c = 0; c < 2; c++) {
      for (size_t i = 0; i < r->track[t].cursor[c].count; i++) { free(r->track[t].cursor[c].item[i].ops); free(r->track[t].cursor[c].item[i].deps); }
      free(r->track[t].cursor[c].item);
    }
  free(r->track);
  *r = (struct recording){0};
}

/* the tracks run (nccl-mesh-metal.h metal_tracks) as one cooperative loop: each pass advances every cursor of every
   track while its next item may run; a pass that advances none is one look, the looks bounded as a wait's are
   (step_wait: the program's polls, MESH_PLATFORM_NS since the last advance, with its backoff), after which each cursor
   that cannot run has its failure recorded and the loop ends once nothing more can run */
struct metal_result metal_tracks(struct metal_program *program, uint32_t tracks, const uint64_t *consumed) {
  struct recording *r = program->recording;
  if (r) r->on = 0;
  if (!r || !r->tracks) return METAL_DONE;
  uint64_t *count = malloc(((size_t)r->tracks + 1) * sizeof *count);
  if (!count) { tracks_free(r); return metal_failed(ENOMEM); }
  for (uint32_t t = 0; t < r->tracks; t++) count[t] = t < tracks ? consumed[t] : 0;
  uint64_t looks = 0, since = now_ns();
  int expired = 0;
  for (;;) {
    int left = 0, moved = 0;
    for (uint32_t t = 0; t < r->tracks; t++)
      for (int c = 0; c < 2; c++) {
        struct cursor *k = r->track[t].cursor + c;
        while (k->next < k->count) {
          const struct item *it = k->item + k->next;
          if (!item_ready(r->track + t, c, count, t)) { left = 1; break; }
          for (size_t s = 0; s < it->nops; s++) run_op(it->ops + s);
          if (c == METAL_CONSUME) count[t] = it->position + 1;
          k->next++; moved = 1;
        }
      }
    if (!left) break;
    if (moved) { looks = 0; since = now_ns(); expired = 0; continue; }
    if (expired) break;
    if (++looks >= program->polls || now_ns() - since > MESH_PLATFORM_NS) {
      for (uint32_t t = 0; t < r->tracks; t++)
        for (int c = 0; c < 2; c++)
          if (r->track[t].cursor[c].next < r->track[t].cursor[c].count) item_expired(r->track[t].cursor[c].item + r->track[t].cursor[c].next);
      expired = 1;
      continue;
    }
    if (looks > 4096) usleep(20);
    else if (looks > 256) sched_yield();
  }
  free(count);
  tracks_free(r);
  return METAL_DONE;
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
  if (program->recording) { tracks_free(program->recording); free(program->recording); }
  if (event) atomic_store_explicit(&((struct host_event *)event)->value, value, memory_order_release);
  for (size_t i = 0; i < program->count; i++) metal_release(program->kept[i]);
  free(program->kept);
  free(program);
}

void metal_collect(void *event) { (void)event; }

/* Immutable host-owned payload-prefix accounting. One writer, concurrent
   observers; append before publishing the corresponding transport entry.
   Each captured shape compresses equal adjacent lengths, never rounds payload bytes.
   Completed history can be pruned only while the caller excludes all readers;
   an active trace retains its entire observation until it joins. */
#ifndef MESH_BYTE_ACCOUNT_H
#define MESH_BYTE_ACCOUNT_H
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct mesh_byte_run { uint64_t end, bytes, each; };
struct mesh_byte_shape {
  _Atomic size_t references;
  uint64_t units, bytes;
  size_t runs;
  struct mesh_byte_run run[];
};
struct mesh_byte_chunk {
  struct mesh_byte_chunk *previous;
  uint64_t first, end, base;
  struct mesh_byte_shape *shape;
};
struct mesh_byte_account {
  _Atomic(struct mesh_byte_chunk *) head;
  uint64_t first, bytes; /* exact absolute prefix of reclaimed history */
  uint64_t checkpoint, checkpoint_bytes; /* one unchanged publication may lag relay completion */
};

static inline int mesh_byte_shape_make(struct mesh_byte_shape **out, const void *lengths, uint64_t units, size_t stride) {
  if (!units || !lengths || stride < sizeof(uint64_t) || units > SIZE_MAX / stride) return EINVAL;
  size_t runs = 0;
  uint64_t bytes = 0, last = 0;
  for (uint64_t i = 0; i < units; i++) {
    uint64_t length;
    memcpy(&length, (const char *)lengths + (size_t)i * stride, sizeof length);
    if (!length || length > UINT64_MAX - bytes) return EOVERFLOW;
    bytes += length;
    runs += length != last;
    last = length;
  }
  if (runs > (SIZE_MAX - sizeof(struct mesh_byte_shape)) / sizeof(struct mesh_byte_run)) return EOVERFLOW;
  struct mesh_byte_shape *shape = malloc(sizeof *shape + runs * sizeof *shape->run);
  if (!shape) return ENOMEM;
  atomic_init(&shape->references, 1);
  shape->units = units; shape->bytes = bytes; shape->runs = runs;
  size_t at = 0;
  bytes = last = 0;
  for (uint64_t i = 0; i < units; i++) {
    uint64_t length;
    memcpy(&length, (const char *)lengths + (size_t)i * stride, sizeof length);
    if (last && length != last) at++;
    bytes += length;
    shape->run[at] = (struct mesh_byte_run){i + 1, bytes, length};
    last = length;
  }
  *out = shape;
  return 0;
}

static inline void mesh_byte_shape_release(struct mesh_byte_shape *shape) {
  if (shape && atomic_fetch_sub_explicit(&shape->references, 1, memory_order_acq_rel) == 1) free(shape);
}

static inline int mesh_byte_account_view(struct mesh_byte_account *account, uint64_t first, struct mesh_byte_shape *shape) {
  struct mesh_byte_chunk *previous = atomic_load_explicit(&account->head, memory_order_acquire);
  if (first != (previous ? previous->end : account->first) || first == UINT64_MAX) return EINVAL;
  if (!shape) return 0;
  if (shape->units >= UINT64_MAX - first) return EINVAL;
  const uint64_t base = previous ? previous->base + previous->shape->bytes : account->bytes;
  if (shape->bytes > UINT64_MAX - base) return EOVERFLOW;
  struct mesh_byte_chunk *chunk = malloc(sizeof *chunk);
  if (!chunk) return ENOMEM;
  *chunk = (struct mesh_byte_chunk){.previous = previous, .first = first, .end = first + shape->units, .base = base, .shape = shape};
  atomic_fetch_add_explicit(&shape->references, 1, memory_order_relaxed);
  atomic_store_explicit(&account->head, chunk, memory_order_release);
  return 0;
}

static inline int mesh_byte_account_append(struct mesh_byte_account *account, uint64_t first,
                                           const void *lengths, uint64_t units, size_t stride) {
  struct mesh_byte_shape *shape = NULL;
  int status = units ? mesh_byte_shape_make(&shape, lengths, units, stride) : 0;
  if (!status) status = mesh_byte_account_view(account, first, shape);
  mesh_byte_shape_release(shape);
  return status;
}

static inline int mesh_byte_account_at(const struct mesh_byte_account *account, uint64_t units, uint64_t *bytes) {
  if (units == account->checkpoint) { *bytes = account->checkpoint_bytes; return 0; }
  if (units < account->first) return EINVAL;
  if (units == account->first) { *bytes = account->bytes; return 0; }
  const struct mesh_byte_chunk *chunk = atomic_load_explicit(&account->head, memory_order_acquire);
  while (chunk && units < chunk->first) chunk = chunk->previous;
  if (!chunk || units > chunk->end) return EINVAL;
  size_t lo = 0, hi = chunk->shape->runs;
  const uint64_t within = units - chunk->first;
  while (lo < hi) {
    const size_t middle = lo + (hi - lo) / 2;
    if (chunk->shape->run[middle].end < within) lo = middle + 1;
    else hi = middle;
  }
  const struct mesh_byte_run *run = chunk->shape->run + lo;
  *bytes = chunk->base + run->bytes - (run->end - within) * run->each;
  return 0;
}

static inline int mesh_byte_account_checkpoint(struct mesh_byte_account *account, uint64_t units) {
  uint64_t bytes;
  const int error = mesh_byte_account_at(account, units, &bytes);
  if (!error) { account->checkpoint = units; account->checkpoint_bytes = bytes; }
  return error;
}

/* A relative prefix vector, out[0]=0, out[i+1]-out[i] is unit i's length.
   Original region descriptors may already have been reclaimed. */
static inline int mesh_byte_account_prefix(const struct mesh_byte_account *account, uint64_t first,
                                           uint64_t units, uint64_t *out) {
  if (units > UINT64_MAX - first) return EOVERFLOW;
  if (first < account->first) return EINVAL;
  const uint64_t end = first + units;
  const struct mesh_byte_chunk *chunk = atomic_load_explicit(&account->head, memory_order_acquire);
  uint64_t through = end;
  while (chunk && through > first) {
    if (through > chunk->end) return EINVAL;
    if (through <= chunk->first) { chunk = chunk->previous; continue; }
    uint64_t before = chunk->first;
    for (size_t r = 0; r < chunk->shape->runs; r++) {
      const uint64_t after = chunk->first + chunk->shape->run[r].end;
      const uint64_t lo = before > first ? before : first;
      const uint64_t hi = after < through ? after : through;
      for (uint64_t u = lo; u < hi; u++) out[u - first + 1] = chunk->shape->run[r].each;
      before = after;
    }
    through = chunk->first > first ? chunk->first : first;
    chunk = chunk->previous;
  }
  if (through != first) return EINVAL;
  out[0] = 0;
  for (uint64_t i = 0; i < units; i++) out[i + 1] += out[i];
  return 0;
}

/* Exclusive operation: no concurrent append, sample, or trace observer.
   Retain the chunk containing a partially completed entry; remove only whole
   chunks at/before through. Storage follows outstanding work and live traces,
   not the lifetime number of completed functions. */
static inline void mesh_byte_account_prune(struct mesh_byte_account *account, uint64_t through) {
  if (through == UINT64_MAX || through <= account->first) return;
  struct mesh_byte_chunk *head = atomic_load_explicit(&account->head, memory_order_relaxed);
  struct mesh_byte_chunk *kept = NULL, *old = head;
  while (old && old->end > through) { kept = old; old = old->previous; }
  if (!old) return;
  account->first = old->end;
  account->bytes = old->base + old->shape->bytes;
  if (kept) kept->previous = NULL;
  else atomic_store_explicit(&account->head, NULL, memory_order_relaxed);
  while (old) { struct mesh_byte_chunk *previous = old->previous; mesh_byte_shape_release(old->shape); free(old); old = previous; }
}

static inline void mesh_byte_account_destroy(struct mesh_byte_account *account) {
  struct mesh_byte_chunk *chunk = atomic_load_explicit(&account->head, memory_order_relaxed);
  while (chunk) { struct mesh_byte_chunk *previous = chunk->previous; mesh_byte_shape_release(chunk->shape); free(chunk); chunk = previous; }
  atomic_store_explicit(&account->head, NULL, memory_order_relaxed);
  account->first = account->bytes = account->checkpoint = account->checkpoint_bytes = 0;
}
#endif

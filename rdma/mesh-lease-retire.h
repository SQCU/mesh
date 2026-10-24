/* One-sided source ownership, independent of destination consumer lifetime.
   mesh_stream.sent is a contiguous COMPLETED PAGE prefix (mesh-link.c,
   link_send/walk_count), including both lanes. It is not a byte prefix and a
   presented/taken/ready prefix is not a completion. Register every future SEND
   range before presenting any entry; otherwise not-yet-posted source reads
   would disappear from the ownership proof. No ACK, wait or payload copy.

   `wire` identifies a QP pairing, not a holding. The caller assigns a unique,
   nonzero `holding` on each local acquisition, even if the same client takes
   the same stream and wire again. Query never substitutes a later holding's
   prefix for an old one's. Keep the generation identity until old dependencies
   retire. Caller serializes these metadata operations; this file adds no lock.
   Loss/timeout is not DMA completion. Invalid dependencies stay retained until
   an explicit driver/quiescent retirement proof is supplied. */
#ifndef MESH_LEASE_RETIRE_H
#define MESH_LEASE_RETIRE_H
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

struct mesh_lease_retire_dep {
  uint32_t stream;
  uint64_t wire, holding, end;
  unsigned done, invalid;
};
struct mesh_lease_retire {
  struct mesh_lease_retire_dep *deps;
  size_t count, room;
  uint64_t registering;
};
struct mesh_lease_retire_sample { uint64_t wire, holding, sent; };
/* Return zero only when sample describes this stream's currently held words. */
typedef int (*mesh_lease_retire_query)(void *, uint32_t, struct mesh_lease_retire_sample *);

/* A registration scope covers dependencies not yet enumerated. A failed scope
   stays open until the owner cancels unpresented work or records all its uses. */
static inline int mesh_lease_retire_begin(struct mesh_lease_retire *r) {
  if (r->registering == UINT64_MAX) return EOVERFLOW;
  r->registering++;
  return 0;
}
static inline int mesh_lease_retire_end(struct mesh_lease_retire *r) {
  if (!r->registering) return EINVAL;
  r->registering--;
  return 0;
}

static inline int mesh_lease_retire_mark(struct mesh_lease_retire *r, uint32_t stream,
                                        uint64_t wire, uint64_t holding, uint64_t end) {
  if (!wire || !holding || end == UINT64_MAX) return EINVAL;
  if (!end) return 0;
  for (size_t i = 0; i < r->count; i++) {
    struct mesh_lease_retire_dep *d = r->deps + i;
    if (d->stream == stream && d->wire == wire && d->holding == holding) {
      if (d->invalid) return ESTALE;
      if (end > d->end) { d->end = end; d->done = 0; }
      return 0;
    }
  }
  if (r->count == r->room) {
    if (r->room > SIZE_MAX / 2 / sizeof *r->deps) return EOVERFLOW;
    const size_t room = r->room ? r->room * 2 : 4;
    struct mesh_lease_retire_dep *grown = realloc(r->deps, room * sizeof *grown);
    if (!grown) return ENOMEM;
    r->deps = grown; r->room = room;
  }
  r->deps[r->count++] = (struct mesh_lease_retire_dep){.stream=stream, .wire=wire, .holding=holding, .end=end};
  return 0;
}

/* A nonblocking snapshot; latch every proved dependency even if another is
   incomplete. Existing completion evidence does not expire when a stream later
   changes generation. A lost or changed generation supplies no new proof. */
static inline int mesh_lease_retire_ready(struct mesh_lease_retire *r,
                                         mesh_lease_retire_query query, void *context) {
  int ready = r->registering == 0;
  for (size_t i = 0; i < r->count; i++) {
    struct mesh_lease_retire_dep *d = r->deps + i;
    if (d->done) continue;
    struct mesh_lease_retire_sample sample;
    if (d->invalid || !query || query(context, d->stream, &sample)) { ready = 0; continue; }
    if (sample.wire != d->wire || sample.holding != d->holding || sample.sent == UINT64_MAX) {
      d->invalid = 1; ready = 0; continue;
    }
    if (sample.sent >= d->end) d->done = 1;
    else ready = 0;
  }
  return ready;
}

/* An explicit proof that this holding cannot issue or retain any DMA source
   read: e.g. all its actual requests completed and presentation ended, or its
   QPs ceased to exist with their owning process. NOT link failure, timeout,
   a ~0 prefix, a local GPU completion, or merely giving the stream back. */
static inline void mesh_lease_retire_drained(struct mesh_lease_retire *r, uint32_t stream,
                                            uint64_t wire, uint64_t holding) {
  for (size_t i = 0; i < r->count; i++) {
    struct mesh_lease_retire_dep *d = r->deps + i;
    if (d->stream == stream && d->wire == wire && d->holding == holding) d->done = 1;
  }
}

/* Metadata destruction does not establish ownership. Caller must already have
   retired the underlying allocation or transferred it to driver-owned pins. */
static inline void mesh_lease_retire_destroy(struct mesh_lease_retire *r) {
  free(r->deps);
  *r = (struct mesh_lease_retire){0};
}
#endif

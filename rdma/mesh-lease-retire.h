/* One-sided DMA and immutable-descriptor ownership.
   mesh_stream.sent/landed are contiguous COMPLETED UNIT prefixes (mesh-link.c,
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
  unsigned direction, metadata, done, invalid;
};
struct mesh_lease_retire {
  struct mesh_lease_retire_dep *deps;
  size_t count, room;
  uint64_t registering;
};
enum { MESH_LEASE_SEND = 0, MESH_LEASE_LAND = 1 };
struct mesh_lease_retire_sample { uint64_t wire, holding, sent, landed; };
/* Return zero only when sample describes this stream's currently held words. */
typedef int (*mesh_lease_retire_query)(void *, uint32_t, struct mesh_lease_retire_sample *);
/* Zero proves no future access from this holding; payload DMA may be handed
   to posted-request pins. Metadata additionally requires local bridge CPU
   reader quiescence, since payload pins do not cover descriptor-table reads. */
typedef int (*mesh_lease_retire_detach_query)(void *, const struct mesh_lease_retire_dep *);

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

static inline int mesh_lease_retire_mark_use(struct mesh_lease_retire *r, uint32_t stream,
                                            uint64_t wire, uint64_t holding, uint64_t end,
                                            unsigned direction, unsigned metadata) {
  if (!wire || !holding || end == UINT64_MAX || direction > MESH_LEASE_LAND || metadata > 1) return EINVAL;
  if (!end) return 0;
  for (size_t i = 0; i < r->count; i++) {
    struct mesh_lease_retire_dep *d = r->deps + i;
    if (d->stream == stream && d->wire == wire && d->holding == holding &&
        d->direction == direction && d->metadata == metadata) {
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
  r->deps[r->count++] = (struct mesh_lease_retire_dep){.stream=stream, .wire=wire, .holding=holding, .end=end,
                                                    .direction=direction, .metadata=metadata};
  return 0;
}
static inline int mesh_lease_retire_mark(struct mesh_lease_retire *r, uint32_t stream,
                                        uint64_t wire, uint64_t holding, uint64_t end) {
  return mesh_lease_retire_mark_use(r, stream, wire, holding, end, MESH_LEASE_SEND, 0);
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
    const uint64_t completed = d->direction == MESH_LEASE_LAND ? sample.landed : sample.sent;
    if (sample.wire != d->wire || sample.holding != d->holding || completed == UINT64_MAX) {
      d->invalid = 1; ready = 0; continue;
    }
    if (completed >= d->end) d->done = 1;
    else ready = 0;
  }
  return ready;
}

/* Detachment is weaker than overwrite/reuse: outstanding payload DMA remains
   protected by posted-request pins. It never turns into completion evidence. */
static inline int mesh_lease_retire_detachable(struct mesh_lease_retire *r, mesh_lease_retire_query query,
                                              mesh_lease_retire_detach_query detach, void *context) {
  if (mesh_lease_retire_ready(r, query, context)) return 1;
  if (r->registering) return 0;
  for (size_t i = 0; i < r->count; i++)
    if (!r->deps[i].done && (!detach || detach(context, r->deps + i))) return 0;
  return 1;
}

/* An explicit proof that this holding cannot issue or retain any DMA source
   read: e.g. all its actual requests completed and presentation ended, or its
   QPs ceased to exist with their owning process. NOT link failure, timeout,
   a ~0 prefix, a local GPU completion, or merely giving the stream back.
   For metadata, the proof must also cover the bridge's CPU descriptor reads. */
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

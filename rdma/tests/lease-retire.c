#include "../mesh-lease-retire.h"
#include <assert.h>
#include <stdio.h>

struct observations {
  struct mesh_lease_retire_sample streams[8];
  int absent[8];
  int future_disabled[8], cpu_quiescent[8];
};
static int observe(void *context, uint32_t stream, struct mesh_lease_retire_sample *out) {
  struct observations *o = context;
  if (stream >= 8 || o->absent[stream]) return ENOENT;
  *out = o->streams[stream];
  return 0;
}
static int detachable(void *context, const struct mesh_lease_retire_dep *dependency) {
  struct observations *o = context;
  return o->future_disabled[dependency->stream] && (!dependency->metadata || o->cpu_quiescent[dependency->stream]) ? 0 : EBUSY;
}

int main(void) {
  struct observations o = {.streams={{.wire=10,.holding=100,.sent=0}, {.wire=20,.holding=200,.sent=0}}};
  struct mesh_lease_retire r = {0};
  assert(mesh_lease_retire_ready(&r, observe, &o));

  /* Register the full planned prefix before any pages are posted. */
  assert(!mesh_lease_retire_begin(&r));
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(!mesh_lease_retire_mark(&r, 0, 10, 100, 1000));
  assert(!mesh_lease_retire_mark(&r, 0, 10, 100, 500));
  assert(!mesh_lease_retire_mark(&r, 1, 20, 200, 300));
  assert(r.count == 2 && r.deps[0].end == 1000);
  o.streams[0].sent = 999;
  o.streams[1].sent = 300;
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(!mesh_lease_retire_end(&r));
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  o.streams[0].sent = 1000;
  assert(mesh_lease_retire_ready(&r, observe, &o));

  /* Reusing the immutable source for another SEND extends its lifetime. */
  assert(!mesh_lease_retire_mark(&r, 0, 10, 100, 1500));
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  o.streams[0].sent = UINT64_MAX;
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  o.streams[0].sent = 9999;
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(mesh_lease_retire_mark(&r, 0, 10, 100, 1600) == ESTALE);
  mesh_lease_retire_drained(&r, 0, 10, 101); /* different holding proves nothing */
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  mesh_lease_retire_drained(&r, 0, 10, 100);
  assert(mesh_lease_retire_ready(&r, observe, &o));
  mesh_lease_retire_destroy(&r);

  /* Same cable/QPs and client, but a later holding: no prefix substitution. */
  assert(!mesh_lease_retire_mark(&r, 0, 10, 100, 800));
  o.streams[0] = (struct mesh_lease_retire_sample){.wire=10, .holding=101, .sent=900};
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(r.deps[0].invalid);
  mesh_lease_retire_destroy(&r);

  assert(!mesh_lease_retire_mark(&r, 0, 10, 102, 800));
  o.streams[0] = (struct mesh_lease_retire_sample){.wire=11, .holding=102, .sent=900};
  assert(!mesh_lease_retire_ready(&r, observe, &o)); /* pairing changed */
  mesh_lease_retire_destroy(&r);

  /* Observe every dependency even when an earlier one remains incomplete. */
  o.streams[0] = (struct mesh_lease_retire_sample){.wire=10, .holding=103, .sent=99};
  o.streams[1] = (struct mesh_lease_retire_sample){.wire=20, .holding=201, .sent=100};
  assert(!mesh_lease_retire_mark(&r, 0, 10, 103, 100));
  assert(!mesh_lease_retire_mark(&r, 1, 20, 201, 100));
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(r.deps[1].done);
  o.streams[1].holding++;
  o.streams[0].sent = 100;
  assert(mesh_lease_retire_ready(&r, observe, &o));
  mesh_lease_retire_destroy(&r);

  assert(mesh_lease_retire_mark(&r, 0, 0, 1, 1) == EINVAL);
  assert(mesh_lease_retire_mark(&r, 0, 1, 0, 1) == EINVAL);
  assert(mesh_lease_retire_mark(&r, 0, 1, 1, UINT64_MAX) == EINVAL);
  assert(mesh_lease_retire_end(&r) == EINVAL);
  assert(!mesh_lease_retire_mark(&r, 0, 10, 103, 100));
  o.absent[0] = 1;
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  o.absent[0] = 0;
  assert(mesh_lease_retire_ready(&r, observe, &o));
  mesh_lease_retire_destroy(&r);

  /* SEND and LAND prefixes prove different obligations, including tables. */
  o.streams[0] = (struct mesh_lease_retire_sample){.wire=10, .holding=104, .sent=9, .landed=6};
  for (unsigned metadata = 0; metadata < 2; metadata++) {
    assert(!mesh_lease_retire_mark_use(&r, 0, 10, 104, 9, MESH_LEASE_SEND, metadata));
    assert(!mesh_lease_retire_mark_use(&r, 0, 10, 104, 7, MESH_LEASE_LAND, metadata));
  }
  assert(r.count == 4);
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(r.deps[0].done && !r.deps[1].done && r.deps[2].done && !r.deps[3].done);
  o.streams[0].landed = 7;
  assert(mesh_lease_retire_ready(&r, observe, &o));
  mesh_lease_retire_destroy(&r);

  /* A pending descriptor CPU read is not protected by payload DMA pins. */
  o.streams[0] = (struct mesh_lease_retire_sample){.wire=10, .holding=105};
  assert(!mesh_lease_retire_mark(&r, 0, 10, 105, 9));
  assert(!mesh_lease_retire_detachable(&r, observe, detachable, &o));
  o.future_disabled[0] = 1; /* client|GIVEN: no future DMA posts */
  assert(mesh_lease_retire_detachable(&r, observe, detachable, &o));
  assert(!mesh_lease_retire_ready(&r, observe, &o)); /* not an overwrite proof */
  assert(!mesh_lease_retire_mark_use(&r, 0, 10, 105, 9, MESH_LEASE_SEND, 1));
  assert(!mesh_lease_retire_mark_use(&r, 0, 10, 105, 7, MESH_LEASE_LAND, 1));
  o.absent[0] = 1; /* local holder cleared; bridge may still read its table */
  assert(!mesh_lease_retire_detachable(&r, observe, detachable, &o));
  o.cpu_quiescent[0] = 1; /* bridge owner 0/GIVEN-alone, not client|GIVEN */
  assert(mesh_lease_retire_detachable(&r, observe, detachable, &o));
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(!mesh_lease_retire_begin(&r));
  assert(!mesh_lease_retire_detachable(&r, observe, detachable, &o));
  assert(!mesh_lease_retire_end(&r));
  mesh_lease_retire_destroy(&r);
  assert(mesh_lease_retire_mark_use(&r, 0, 10, 105, 1, 2, 0) == EINVAL);
  assert(mesh_lease_retire_mark_use(&r, 0, 10, 105, 1, 0, 2) == EINVAL);
  puts("lease retirement: source, landing, descriptor CPU lifetime, loss and generation checks passed");
}

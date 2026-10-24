#include "../mesh-lease-retire.h"
#include <assert.h>
#include <stdio.h>

struct observations {
  struct mesh_lease_retire_sample streams[8];
  int absent[8];
};
static int observe(void *context, uint32_t stream, struct mesh_lease_retire_sample *out) {
  struct observations *o = context;
  if (stream >= 8 || o->absent[stream]) return ENOENT;
  *out = o->streams[stream];
  return 0;
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
  o.streams[0] = (struct mesh_lease_retire_sample){10, 101, 900};
  assert(!mesh_lease_retire_ready(&r, observe, &o));
  assert(r.deps[0].invalid);
  mesh_lease_retire_destroy(&r);

  assert(!mesh_lease_retire_mark(&r, 0, 10, 102, 800));
  o.streams[0] = (struct mesh_lease_retire_sample){11, 102, 900};
  assert(!mesh_lease_retire_ready(&r, observe, &o)); /* pairing changed */
  mesh_lease_retire_destroy(&r);

  /* Observe every dependency even when an earlier one remains incomplete. */
  o.streams[0] = (struct mesh_lease_retire_sample){10, 103, 99};
  o.streams[1] = (struct mesh_lease_retire_sample){20, 201, 100};
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
  puts("lease retirement: all source-prefix, loss, generation and presentation checks passed");
}

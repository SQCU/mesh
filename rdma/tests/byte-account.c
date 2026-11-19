#include "../mesh-byte-account.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>

struct concurrent {
  struct mesh_byte_account account;
  _Atomic uint64_t published;
};
static void *observe(void *context) {
  struct concurrent *c = context;
  uint64_t through;
  do {
    through = atomic_load_explicit(&c->published, memory_order_acquire);
    uint64_t bytes;
    assert(!mesh_byte_account_at(&c->account, through, &bytes));
    assert(bytes == through * (through + 1) / 2);
  } while (through != 1000);
  return NULL;
}

int main(void) {
  struct mesh_byte_account account = {0};
  struct span { uint64_t offset, bytes; } spans[] = {{100, 1}, {101, 4097}, {5000, 17}, {5017, 17}};
  assert(!mesh_byte_account_append(&account, 0, &spans[0].bytes, 4, sizeof *spans));
  assert(atomic_load(&account.head)->shape->runs == 3);
  uint64_t bytes;
  const uint64_t expected[] = {0, 1, 4098, 4115, 4132};
  for (uint64_t i = 0; i < 5; i++) {
    assert(!mesh_byte_account_at(&account, i, &bytes));
    assert(bytes == expected[i]);
  }
  const uint64_t more[] = {3, 3, 8191};
  assert(!mesh_byte_account_append(&account, 4, more, 3, sizeof *more));
  assert(atomic_load(&account.head)->shape->runs == 2);
  assert(!mesh_byte_account_at(&account, 7, &bytes) && bytes == 12329);
  uint64_t prefix[6];
  assert(!mesh_byte_account_prefix(&account, 1, 5, prefix));
  const uint64_t range[] = {0, 4097, 4114, 4131, 4134, 4137};
  for (size_t i = 0; i < 6; i++) assert(prefix[i] == range[i]);
  assert(mesh_byte_account_at(&account, 8, &bytes) == EINVAL);
  assert(mesh_byte_account_append(&account, 6, more, 1, sizeof *more) == EINVAL);
  assert(mesh_byte_account_prefix(&account, 6, 2, prefix) == EINVAL);
  assert(!mesh_byte_account_at(&account, 1, &bytes) && bytes == 1);
  mesh_byte_account_prune(&account, 3); /* incomplete first entry retained */
  assert(account.first == 0 && atomic_load(&account.head)->previous);
  mesh_byte_account_prune(&account, 5);
  assert(account.first == 4 && account.bytes == 4132 && !atomic_load(&account.head)->previous);
  assert(mesh_byte_account_at(&account, 3, &bytes) == EINVAL);
  assert(!mesh_byte_account_at(&account, 4, &bytes) && bytes == 4132);
  assert(!mesh_byte_account_at(&account, 5, &bytes) && bytes == 4135);
  assert(!mesh_byte_account_prefix(&account, 4, 3, prefix) && prefix[3] == 8197);
  assert(!mesh_byte_account_checkpoint(&account, 5));
  mesh_byte_account_prune(&account, UINT64_MAX); /* lost wire is not completion */
  assert(atomic_load(&account.head));
  mesh_byte_account_prune(&account, 7);
  assert(!atomic_load(&account.head) && account.first == 7 && account.bytes == 12329);
  assert(!mesh_byte_account_at(&account, 7, &bytes) && bytes == 12329);
  assert(!mesh_byte_account_at(&account, 5, &bytes) && bytes == 4135);
  /* A persistent communicator accumulates exact scalar totals, not history. */
  for (uint64_t i = 0; i < 100000; i++) {
    uint64_t length = 3 + i % 7;
    assert(!mesh_byte_account_append(&account, 7 + i, &length, 1, sizeof length));
    assert(!atomic_load(&account.head)->previous);
    assert(!mesh_byte_account_at(&account, 8 + i, &bytes));
    mesh_byte_account_prune(&account, 8 + i);
    assert(!atomic_load(&account.head) && account.bytes == bytes);
    assert(!mesh_byte_account_at(&account, 5, &bytes) && bytes == 4135);
  }
  mesh_byte_account_destroy(&account);

  struct mesh_byte_shape *shape = NULL;
  uint64_t sizes[] = {8, 16, 4};
  assert(!mesh_byte_shape_make(&shape, sizes, 3, sizeof *sizes));
  for (uint64_t start=0; start<9; start+=3) {
    assert(!mesh_byte_account_view(&account, start, shape));
    assert(atomic_load(&account.head)->shape == shape);
  }
  mesh_byte_shape_release(shape);
  mesh_byte_account_prune(&account, 3);
  assert(!mesh_byte_account_at(&account, 7, &bytes) && bytes == 64);
  assert(!mesh_byte_account_at(&account, 9, &bytes) && bytes == 84);
  mesh_byte_account_destroy(&account);

  struct concurrent c = {0};
  pthread_t reader;
  assert(!pthread_create(&reader, NULL, observe, &c));
  for (uint64_t i = 1; i <= 1000; i++) {
    assert(!mesh_byte_account_append(&c.account, i - 1, &i, 1, sizeof i));
    atomic_store_explicit(&c.published, i, memory_order_release);
  }
  assert(!pthread_join(reader, NULL));
  mesh_byte_account_destroy(&c.account);
  puts("byte accounting: ragged spans, compressed runs, old prefixes and concurrent observations passed");
}

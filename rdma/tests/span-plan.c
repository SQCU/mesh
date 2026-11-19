/* cc -std=c11 -O2 -Wall -Wextra -Werror -Irdma rdma/tests/span-plan.c
      -o /tmp/mesh-span-plan-test */
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t attempts, fail_at = SIZE_MAX, live;
static void *test_malloc(size_t bytes) {
  if (attempts++ == fail_at) return NULL;
  void *p = malloc(bytes);
  if (p) live++;
  return p;
}
static void *test_realloc(void *p, size_t bytes) {
  if (attempts++ == fail_at) return NULL;
  void *grown = realloc(p, bytes);
  if (grown && !p) live++;
  return grown;
}
static void test_free(void *p) {
  if (p) { assert(live); live--; }
  free(p);
}
#define malloc test_malloc
#define realloc test_realloc
#define free test_free
#include "mesh-span-plan.h"
#undef malloc
#undef realloc
#undef free

/* Deliberately enumerate each repeat in this independent small-case oracle. */
static void verify(const uint64_t *phases, const uint64_t *strides, uint32_t participants,
                   uint64_t bytes, uint64_t repeats, uint32_t packet) {
  struct mesh_span_plan plan = {0};
  assert(!mesh_span_plan_create(phases, strides, participants, bytes, repeats, packet, &plan));
  uint64_t expected[32768];
  size_t count = 0;
  expected[count++] = 0;
  if (bytes) expected[count++] = bytes;
  for (uint64_t at = packet; at < bytes; at += packet) {
    assert(count < sizeof expected / sizeof *expected); expected[count++] = at;
  }
  for (uint32_t p = 0; p < participants; p++) for (uint64_t k = 0; k < repeats; k++) {
    const uint64_t phase = (phases[p] + k * strides[p]) % MESH_DMA_EXTENT;
    for (uint64_t at = MESH_DMA_EXTENT - phase; at < bytes; at += MESH_DMA_EXTENT) {
      assert(count < sizeof expected / sizeof *expected); expected[count++] = at;
    }
    for (size_t i = 1; i < plan.count; i++) {
      assert(plan.cuts[i] > plan.cuts[i - 1]);
      const uint64_t length = plan.cuts[i] - plan.cuts[i - 1];
      assert(length <= packet);
      assert(length <= MESH_DMA_EXTENT - (phase + plan.cuts[i - 1]) % MESH_DMA_EXTENT);
    }
  }
  qsort(expected, count, sizeof *expected, mesh_span_cut_compare);
  size_t unique = 1;
  for (size_t i = 1; i < count; i++) if (expected[i] != expected[unique - 1]) expected[unique++] = expected[i];
  assert(plan.count == unique && !memcmp(plan.cuts, expected, unique * sizeof *expected));
  mesh_span_plan_free(&plan);
  assert(!live && !plan.cuts && !plan.count);
}

static void edges(void) {
  const uint64_t phase[] = {0, MESH_DMA_EXTENT - 3, MESH_DMA_EXTENT - 3, 17};
  const uint64_t stride[] = {64, 17, 19, MESH_DMA_EXTENT - 1};
  for (uint32_t bytes = 0; bytes < 100; bytes++) for (uint32_t packet = 1; packet < 19; packet++)
    verify(phase, stride, 4, bytes, 7, packet);
  const uint64_t aligned[] = {0, 16384}, same[] = {16384, 32768};
  verify(aligned, same, 2, 3 * MESH_DMA_EXTENT + 17, 8, 16384 * 32);
  verify(phase, stride, 4, 4 * MESH_DMA_EXTENT + 17, 3, UINT32_MAX);
  verify(phase, stride, 4, 2 * MESH_DMA_EXTENT + 19, 3, 1000000007);
}

static void large_repeat_orbits(void) {
  struct mesh_span_plan plan = {0};
  const uint64_t phase[] = {0, 17}, stride[] = {1, MESH_DMA_EXTENT - 1};
  assert(!mesh_span_plan_create(phase, stride, 2, 17, MESH_DMA_EXTENT, 16, &plan));
  assert(plan.count == 18);
  for (uint64_t i = 0; i < plan.count; i++) assert(plan.cuts[i] == i);
  mesh_span_plan_free(&plan);
  const uint64_t still_phase[] = {MESH_DMA_EXTENT - 3}, still_stride[] = {0};
  assert(!mesh_span_plan_create(still_phase, still_stride, 1, 17, UINT64_MAX, 16, &plan));
  const uint64_t expected[] = {0, 3, 16, 17};
  assert(plan.count == 4 && !memcmp(plan.cuts, expected, sizeof expected));
  mesh_span_plan_free(&plan);
  /* The complete orbit is already on the packet grid; do not enumerate it. */
  const uint64_t aligned_phase[] = {0}, aligned_stride[] = {16};
  assert(!mesh_span_plan_create(aligned_phase, aligned_stride, 1, 65537, MESH_DMA_EXTENT, 16, &plan));
  assert(plan.count == 4098 && plan.cuts[plan.count - 1] == 65537);
  mesh_span_plan_free(&plan);
  assert(!live);
}

static void errors(void) {
  const uint64_t phase[] = {MESH_DMA_EXTENT - 1}, stride[] = {13}, bad_phase[] = {MESH_DMA_EXTENT};
  struct mesh_span_plan plan = {0};
  assert(mesh_span_plan_create(phase, stride, 1, 17, 0, 16, &plan) == EINVAL);
  assert(mesh_span_plan_create(phase, stride, 1, 17, 2, 0, &plan) == EINVAL);
  assert(mesh_span_plan_create(bad_phase, stride, 1, 17, 2, 16, &plan) == EINVAL);
  assert(mesh_span_plan_create(phase, stride, 0, 17, 2, 16, &plan) == EINVAL);
  assert(mesh_span_plan_create(phase, stride, 1, UINT64_MAX, 1, 1, &plan) == EOVERFLOW);
  assert(mesh_span_plan_create(phase, stride, 1, 17, UINT64_MAX, 16, &plan) == EOVERFLOW);
  assert(!plan.cuts && !plan.count && !live);
  for (fail_at = 0; ; fail_at++) {
    attempts = 0;
    const int error = mesh_span_plan_create(phase, stride, 1, 1000, 100, 31, &plan);
    if (!error) { mesh_span_plan_free(&plan); break; }
    assert(error == ENOMEM && !plan.cuts && !plan.count && !live);
  }
  assert(fail_at >= 2 && !live);
  fail_at = SIZE_MAX;
  assert(!mesh_span_plan_create(phase, stride, 1, 17, 2, 16, &plan));
  uint64_t *old = plan.cuts;
  assert(mesh_span_plan_create(phase, stride, 1, 17, 2, 16, &plan) == EINVAL && plan.cuts == old);
  mesh_span_plan_free(&plan);
}

static uint32_t random_state = 0x72941;
static uint32_t random_word(void) { random_state = random_state * 1664525u + 1013904223u; return random_state; }

static void random_cases(void) {
  for (unsigned sample = 0; sample < 500; sample++) {
    const uint32_t participants = 1 + random_word() % 4;
    uint64_t phase[4], stride[4];
    for (uint32_t p = 0; p < participants; p++) {
      phase[p] = sample % 2 ? random_word() % MESH_DMA_EXTENT : MESH_DMA_EXTENT - 1 - random_word() % 1000;
      stride[p] = sample % 3 ? random_word() % 1000 : random_word() % MESH_DMA_EXTENT;
    }
    const uint64_t bytes = sample % 2 ? (uint64_t)(random_word() % 3) * MESH_DMA_EXTENT + random_word() % 1000 : random_word() % 1000;
    const uint32_t packet = sample % 2 ? 100000000 + random_word() % 1000000000 : 1 + random_word() % 127;
    verify(phase, stride, participants, bytes, 1 + random_word() % 11, packet);
  }
}

int main(void) {
  edges(); large_repeat_orbits(); errors(); random_cases();
  puts("span plan: 2303 oracle cases, per-rank strides, billion-repeat closure, exact cuts, overflow and allocation cleanup passed");
}

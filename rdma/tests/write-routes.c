/* cc -std=c11 -O2 -Wall -Wextra -Werror -Irdma rdma/tests/write-routes.c
      rdma/mesh-fabric.c -o /tmp/mesh-write-routes-test */
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
static void *test_calloc(size_t count, size_t bytes) {
  if (attempts++ == fail_at) return NULL;
  void *p = calloc(count, bytes);
  if (p) live++;
  return p;
}
static void test_free(void *p) {
  if (p) { assert(live); live--; }
  free(p);
}
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#include "mesh-write-routes.h"
#undef malloc
#undef calloc
#undef free

/* Independent subset-DP count: arrivals at destination are terminal. Parallel
   edges multiply the count even when they share the same next node. */
static uint64_t count_paths(const struct mesh_fabric *f, uint32_t source, uint32_t destination) {
  assert(f->nodes <= 7);
  uint64_t count[128][7] = {{0}}, result = 0;
  count[1u << source][source] = 1;
  for (uint32_t mask = 0; mask < (1u << f->nodes); mask++) for (uint32_t at = 0; at < f->nodes; at++) {
    if (at == destination) { result += count[mask][at]; continue; }
    if (!count[mask][at]) continue;
    for (uint32_t e = 0; e < f->cables; e++) {
      const struct mesh_cable *c = f->cable + e;
      if (c->node[0] != at && c->node[1] != at) continue;
      const uint32_t other = c->node[c->node[0] == at];
      if (!(mask & (1u << other))) count[mask | (1u << other)][other] += count[mask][at];
    }
  }
  return result;
}

static void verify(const struct mesh_fabric *f, const struct mesh_paths *routes) {
  for (uint32_t source = 0; source < f->nodes; source++) for (uint32_t destination = 0; destination < f->nodes; destination++) {
    const struct mesh_paths *p = routes + (size_t)source * f->nodes + destination;
    if (source == destination) {
      assert(!p->count && !p->first && !p->cable && !p->log_weight);
      continue;
    }
    assert(!mesh_paths_check(f, source, destination, p));
    assert(p->count == count_paths(f, source, destination));
    for (uint32_t a = 0; a < p->count; a++) {
      assert(p->log_weight[a] == 0);
      for (uint32_t b = a + 1; b < p->count; b++) {
        const uint32_t length = p->first[a + 1] - p->first[a];
        assert(length != p->first[b + 1] - p->first[b] ||
               memcmp(p->cable + p->first[a], p->cable + p->first[b], length * sizeof *p->cable));
      }
    }
  }
}

static const struct mesh_cable parallel[] = {
  {11, 1, {0, 1}}, {12, 200, {1, 0}}, {13, 3, {0, 2}},
  {14, 400, {2, 1}}, {15, 5, {1, 2}}
};

static void defaults(void) {
  const struct mesh_fabric f = {3, 5, parallel};
  struct mesh_paths *routes = NULL;
  assert(!mesh_write_routes_default(&f, &routes));
  verify(&f, routes);
  const struct mesh_paths *p = routes + 1;  /* 0 -> 1 */
  const uint32_t first[] = {0, 1, 2, 4, 6}, cable[] = {0, 1, 2, 3, 2, 4};
  assert(p->count == 4 && !memcmp(p->first, first, sizeof first) && !memcmp(p->cable, cable, sizeof cable));
  struct mesh_paths *same = routes;
  assert(mesh_write_routes_default(&f, &routes) == EINVAL && routes == same);
  /* Existing nccl-mesh.c paths_free ownership, including a pair replaced by a
     solver. Other pairs stay valid and no global address identity is needed. */
  struct mesh_paths *replace = routes + 1;
  test_free((void *)replace->first); test_free((void *)replace->cable); test_free((void *)replace->log_weight);
  *replace = (struct mesh_paths){0};
  assert(!mesh_paths_check(&f, 1, 0, routes + 3));
  mesh_write_routes_free(routes, f.nodes);
  assert(!live);
}

static void failures(void) {
  struct mesh_paths *routes = NULL;
  assert(mesh_write_routes_default(NULL, &routes) == EINVAL && !routes);
  const struct mesh_fabric singleton = {1, 0, NULL}, disconnected = {3, 1, parallel};
  assert(!mesh_write_routes_default(&singleton, &routes));
  assert(!routes[0].count && !routes[0].first);
  mesh_write_routes_free(routes, 1); routes = NULL;
  assert(mesh_write_routes_default(&disconnected, &routes) == ENOLINK && !routes && !live);
  const struct mesh_fabric too_large = {UINT32_MAX, 0, NULL};
  assert(mesh_write_routes_default(&too_large, &routes) == EOVERFLOW && !routes && !live);
  const struct mesh_cable invalid[] = {{1, 1, {0, 0}}};
  const struct mesh_fabric bad = {2, 1, invalid};
  assert(mesh_write_routes_default(&bad, &routes) == EINVAL && !routes && !live);
  const struct mesh_fabric f = {3, 5, parallel};
  for (fail_at = 0; ; fail_at++) {
    attempts = 0;
    const int error = mesh_write_routes_default(&f, &routes);
    if (!error) { mesh_write_routes_free(routes, f.nodes); routes = NULL; break; }
    assert(error == ENOMEM && !routes && !live);
  }
  assert(fail_at == 5 + 6 * 3 && !live);
  fail_at = SIZE_MAX;
}

static uint32_t random_state = 0x72941;
static uint32_t random_word(void) {
  random_state = random_state * 1664525u + 1013904223u;
  return random_state;
}

static void geometries(void) {
  for (uint32_t n = 2; n <= 7; n++) {
    struct mesh_cable cables[42];
    uint32_t edges = 0;
    for (uint32_t a = 0; a < n; a++) for (uint32_t b = a + 1; b < n; b++) {
      cables[edges] = (struct mesh_cable){edges + 1, 1, {a, b}}; edges++;
    }
    const struct mesh_fabric f = {n, edges, cables};
    struct mesh_paths *routes = NULL;
    assert(!mesh_write_routes_default(&f, &routes)); verify(&f, routes);
    /* Complete graphs include maximum-length paths without penalizing them. */
    int longest = 0;
    for (uint32_t p = 0; p < routes[1].count; p++) longest |= routes[1].first[p + 1] - routes[1].first[p] == n - 1;
    assert(longest);
    mesh_write_routes_free(routes, n); assert(!live);
  }
  for (unsigned sample = 0; sample < 100; sample++) {
    struct mesh_cable cables[42];
    const uint32_t n = 2 + random_word() % 5;
    uint32_t edges = 0;
    for (uint32_t a = 0; a < n; a++) for (uint32_t b = a + 1; b < n; b++) {
      const uint32_t parallel_count = (b == a + 1) ? 1 + random_word() % 2 : random_word() % 3;
      for (uint32_t k = 0; k < parallel_count; k++) {
        cables[edges] = (struct mesh_cable){edges + 1, 1 + random_word() % 1000, {a, b}}; edges++;
      }
    }
    const struct mesh_fabric f = {n, edges, cables};
    struct mesh_paths *routes = NULL;
    assert(!mesh_write_routes_default(&f, &routes)); verify(&f, routes);
    mesh_write_routes_free(routes, n); assert(!live);
  }
}

int main(void) {
  defaults(); failures(); geometries();
  puts("default physical routes: all simple paths, parallel cables, zero hop tax, failure cleanup, 106 geometries passed");
}

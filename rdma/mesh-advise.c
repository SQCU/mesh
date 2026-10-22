/* design/transport.md: mesh_program_advise over every collective mesh_compile makes on the loopback
   fabric's maps (mesh-loop.c family_cables: the pair, line 4, ring 5, star 8, torus 6 and 8), along each map's
   shortest-path trees and its packed trees (as libnccl-mesh makes them, nccl-mesh.c comm_routes, compile_form), at
   chunks 1 to 4, every root of a rooted collective, the all-reduce whole and two-phase, every node or the even nodes
   contributing, over operands of 1, 4,096 and 1,000,003 float32 and 2^20 half, on mesh_wire_of's wire at a 16 KiB
   block; mesh_group_advise over a group on each map and tree set (collectives along the trees, and the exchanges
   libnccl-mesh forwards along a routing table, mesh_forward_steps: an all-to-all, a gather, a scatter, a counted
   all-to-all and two groups' point exchanges, along the solver's table, mesh_routing_pack, or the shortest trees' one
   route a pair, with the count of their steps between ranks the map does not link and of the SENDs members between a
   block's ends forward).  It prints each one's advisories and a summary line, then the totals; it refuses nothing,
   exits 0.  `mesh-advise mutate`: programs and groups broken on purpose, each named with the check it must fire and
   whether it did. */
#include "mesh-plan.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHECKS = 7, KINDS = 5, SHOWN = 4, MOST = 8 };
static const char *const checks[CHECKS] = {"projection", "priorities", "depth", "causality", "hazard", "invocation", "bound"};
static const char *const kinds[KINDS] = {"allreduce", "broadcast", "reduce", "reduce-scatter", "allgather"};

struct tally { const char *label; uint32_t by_check[CHECKS]; };
static void warned(void *arg, const char *line) {
  struct tally *t = arg;
  for (int c = 0; c < CHECKS; c++) {
    const size_t length = strlen(checks[c]);
    if (strncmp(line, checks[c], length) || line[length] != ':') continue;
    if (t->by_check[c]++ < SHOWN) printf("  %s %s\n", t->label, line);
  }
}
static void summary(const struct tally *t, uint32_t warnings, uint64_t *total, const char *verdict) {
  printf("%s: %u warnings (", t->label, warnings);
  for (int c = 0; c < CHECKS; c++) {
    printf("%s%s %u", c ? ", " : "", checks[c], t->by_check[c]);
    if (total) total[c] += t->by_check[c];
    if (t->by_check[c] > SHOWN) printf(" (%u more not shown)", t->by_check[c] - SHOWN);
  }
  printf(")%s\n", verdict);
}

/* mesh-loop.c family_cables' rule */
static uint32_t family_cables(const char *family, uint32_t n, uint32_t (*link)[2]) {
  uint32_t k = 0;
  if (!strcmp(family, "line") || !strcmp(family, "ring")) {
    for (uint32_t i = 0; i + 1 < n; i++) { link[k][0] = i; link[k][1] = i + 1; k++; }
    if (!strcmp(family, "ring") && n > 2) { link[k][0] = n - 1; link[k][1] = 0; k++; }
  } else if (!strcmp(family, "star")) {
    for (uint32_t i = 1; i < n; i++) { link[k][0] = 0; link[k][1] = i; k++; }
  } else if (!strcmp(family, "torus")) {
    const uint32_t columns = n / 2;
    for (uint32_t r = 0; r < 2; r++)
      for (uint32_t c = 0; c < columns && columns > 1; c++) {
        if (columns == 2 && c == 1) break;
        link[k][0] = r * columns + c; link[k][1] = r * columns + (c + 1) % columns; k++;
      }
    for (uint32_t c = 0; c < columns; c++) { link[k][0] = c; link[k][1] = columns + c; k++; }
  }
  return k;
}
static const struct { const char *name, *family; uint32_t n; } maps[] = {
  {"pair", "line", 2}, {"line4", "line", 4}, {"ring5", "ring", 5}, {"star8", "star", 8}, {"torus6", "torus", 6}, {"torus8", "torus", 8}};
static uint32_t map_trees(size_t m, int packed, uint32_t (*link)[2], uint32_t **root, uint32_t **parent, double **weight) {
  const uint32_t n = maps[m].n, most = packed ? 2 : 1;
  struct mesh_link_map map = {MESH_LINKS_GRAPH, n, 0, link, NULL, 0, NULL, NULL, NULL};
  map.links = family_cables(maps[m].family, n, link);
  *root = calloc((size_t)n * most + 1, sizeof **root); *parent = calloc((size_t)n * n * most + 1, sizeof **parent);
  *weight = calloc((size_t)n * most + 1, sizeof **weight);
  if (mesh_link_map_check(&map) || !*root || !*parent || !*weight) return 0;
  const uint32_t count = packed ? mesh_trees_pack(&map, 0.1, most, *root, *parent, *weight) : mesh_trees_shortest(&map, *root, *parent, *weight);
  for (uint32_t t = 0; packed && t < count; t++) (*weight)[t] = 0;
  return count;
}

/* a call's steps at every rank, the rows a group call names */
struct rows { uint32_t count[MOST]; struct mesh_step *steps[MOST]; };
static void rows_free(struct rows *c) { for (int r = 0; r < MOST; r++) free(c->steps[r]); memset(c, 0, sizeof *c); }
static int rows_collective(struct rows *c, const struct mesh_trees *trees, uint32_t what, uint32_t root, int whole, uint32_t chunks,
                           struct mesh_operand operand) {
  uint32_t parts, moves;
  mesh_compile_room(trees, chunks, &parts, &moves);
  uint32_t *segment = malloc((parts + 1) * sizeof *segment), *first = malloc((trees->nodes + 1) * sizeof *first);
  double *log_weight = malloc((parts + 1) * sizeof *log_weight);
  struct mesh_move *move = malloc((moves + 1) * sizeof *move);
  struct mesh_program program;
  int failed = !segment || !first || !log_weight || !move || mesh_compile(trees, what, root, whole, chunks, NULL, &program, segment, log_weight, first, move);
  for (uint32_t r = 0; !failed && r < trees->nodes; r++) {
    failed = !(c->steps[r] = malloc((first[r + 1] - first[r] + 1) * sizeof **c->steps));
    if (!failed) c->count[r] = mesh_program_steps(&program, r, operand, c->steps[r]);
  }
  free(segment); free(first); free(log_weight); free(move);
  return failed;
}
/* a map's routing table (libnccl-mesh's exchanges' operand, nccl.h ncclMeshCommSetRouting): the solver's, its packed
   trees (mesh_routing_pack at the members' count of trees a root, as libnccl-mesh's routing_solve makes it), or its
   shortest-path trees' (mesh_routing_trees: one route a pair) */
struct table { struct mesh_routing routing; uint32_t *pair, *start, *node; double *share; };
static void table_free(struct table *t) { free(t->pair); free(t->start); free(t->node); free(t->share); memset(t, 0, sizeof *t); }
static int table_of(size_t m, int packed, const struct mesh_trees *shortest, uint32_t (*link)[2], struct table *t) {
  const uint32_t n = maps[m].n;
  struct mesh_link_map map = {MESH_LINKS_GRAPH, n, family_cables(maps[m].family, n, link), link, NULL, 0, NULL, NULL, NULL};
  uint32_t routes, hops;
  mesh_routing_room(n, n, &routes, &hops);
  t->pair = calloc((size_t)n * n + 1, sizeof *t->pair); t->start = calloc((size_t)routes + 1, sizeof *t->start);
  t->node = calloc((size_t)hops + 1, sizeof *t->node); t->share = calloc((size_t)routes + 1, sizeof *t->share);
  if (!t->pair || !t->start || !t->node || !t->share) return 1;
  const uint32_t made = packed ? mesh_routing_pack(&map, 0, n, &t->routing, t->pair, t->start, t->node, t->share)
                               : mesh_routing_trees(shortest, &t->routing, t->pair, t->start, t->node, t->share);
  return made == UINT32_MAX || mesh_routing_check(&map, &t->routing);
}
/* an exchange of `count` blocks at every rank along a routing table (mesh_forward_steps), its relay buffer `relay` */
static int rows_exchange(struct rows *c, const struct mesh_routing *routing, uint32_t n, const struct mesh_block *blocks, uint32_t count,
                         uint32_t relay) {
  const uint32_t room = mesh_forward_room(routing, blocks, count) + 1;
  for (uint32_t r = 0; r < n; r++) {
    if (!(c->steps[r] = malloc(room * sizeof **c->steps))) return 1;
    c->count[r] = mesh_forward_steps(routing, r, blocks, count, relay, c->steps[r], NULL, NULL);
    if (c->count[r] == UINT32_MAX) return 1;
  }
  return 0;
}
/* ragged sizes, each pair's alike at both ends: an all-to-all's elements from a to b, a counted one's entries */
static uint64_t ragged(int hosted, uint32_t a, uint32_t b) { return a == b ? 0 : hosted ? 1 + (a * 5 + b * 3) % 7 * 700 : 1 + (a + 2 * b) % 3; }
/* an all-to-all's blocks (libnccl-mesh exchange_blocks: buffer 0 to buffer 1, the relay 2) of ragged sizes: the all-to-all
   (root -1), or a gather to `root`, or a scatter from it (`scatter`) */
static uint32_t blocks_all(uint32_t n, int32_t root, int scatter, struct mesh_block *blocks) {
  uint64_t sizes[MOST * MOST];
  uint32_t k = 0;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = 0; b < n; b++) sizes[a * n + b] = root < 0 || (int32_t)(scatter ? a : b) == root ? ragged(1, a, b) : 0;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = 0; b < n; b++) {
      if (a == b || !sizes[a * n + b]) continue;
      uint64_t row = 0, column = 0;
      for (uint32_t q = 0; q < b; q++) row += sizes[a * n + q];
      for (uint32_t q = 0; q < a; q++) column += sizes[q * n + b];
      blocks[k++] = (struct mesh_block){a, b, 0, 1, 1, 4, row, column, sizes[a * n + b]};
    }
  return k;
}
/* a counted all-to-all's blocks (exchange_blocks: each pair's counts, buffer 0 to 2, then its rows' room, buffer 1 to 3,
   64 rows of 8 float32; the relay 4) */
static uint32_t blocks_counted(uint32_t n, struct mesh_block *blocks) {
  const uint64_t room = 64 * 8;
  uint32_t k = 0;
  for (uint32_t pass = 0; pass < 2; pass++)
    for (uint32_t a = 0; a < n; a++)
      for (uint32_t b = 0; b < n; b++) {
        if (a == b) continue;
        uint64_t row = 0, column = 0, apart = 0;
        for (uint32_t q = 0; q < b; q++) row += ragged(0, a, q);
        for (uint32_t q = 0; q < a; q++) { column += ragged(0, q, b); apart += q != b ? room : 0; }
        blocks[k++] = pass ? (struct mesh_block){a, b, 1, 3, 1, 4, 0, apart, room} : (struct mesh_block){a, b, 0, 2, 8, 8, row, column, ragged(0, a, b)};
      }
  return k;
}
/* the point exchange of a group of point-to-point calls at every rank (libnccl-mesh group_lower: of bytes, what a sends
   b from buffer 1 + b to buffer 1 + n + a, the relay 0): each rank sends the rank `shift` on `bytes` */
static uint32_t blocks_point(uint32_t n, uint32_t shift, uint64_t bytes, struct mesh_block *blocks) {
  uint32_t k = 0;
  for (uint32_t a = 0; a < n; a++)
    for (uint32_t b = 0; b < n; b++)
      if (a != b && b == (a + shift) % n) blocks[k++] = (struct mesh_block){a, b, 1 + b, 1 + n + a, 1, 1, 0, 0, bytes};
  return k;
}
static uint32_t group_advise(const char *label, struct rows *rows, const uint32_t *kind, uint32_t calls, uint32_t n, uint32_t polls,
                             uint32_t bound, const struct mesh_wire *wire, uint64_t *total) {
  struct mesh_group_call call[16];
  for (uint32_t i = 0; i < calls; i++) call[i] = (struct mesh_group_call){kind[i], 0, rows[i].count, (const struct mesh_step *const *)rows[i].steps};
  const struct mesh_group group = {n, calls, polls, bound, call};
  struct tally t = {label, {0}};
  const uint32_t warnings = mesh_group_advise(&group, wire, warned, &t);
  summary(&t, warnings, total, "");
  return warnings;
}

static int programs(const struct mesh_wire *wire) {
  static const struct mesh_operand operands[] = {{0, 4, 1}, {0, 4, 4096}, {0, 4, 1000003}, {1, 2, 1u << 20}};
  uint64_t count = 0, failed = 0, unchecked = 0, priorities_free = 0, admissible = 0, total[CHECKS] = {0}, by_kind[KINDS][CHECKS] = {{0}};
  uint64_t of_kind[KINDS] = {0};
  for (size_t m = 0; m < sizeof maps / sizeof *maps; m++)
    for (int packed = 0; packed < 2; packed++) {
      const uint32_t n = maps[m].n;
      uint32_t link[64][2], *root, *parent;
      double *weight;
      const uint32_t trees_count = map_trees(m, packed, link, &root, &parent, &weight);
      const struct mesh_trees trees = {n, trees_count, root, parent, weight};
      if (!trees_count) printf("%s %s: no trees\n", maps[m].name, packed ? "packed" : "shortest");
      for (uint32_t what = 0; trees_count && what < KINDS; what++) {
        const int rooted = what == MESH_REDUCE || what == MESH_BROADCAST;
        for (int whole = 0; whole < (what == MESH_ALLREDUCE ? 2 : 1); whole++)
          for (uint32_t r = 0; r < (rooted ? n : 1); r++)
            for (uint32_t chunks = 1; chunks <= 4; chunks++)
              for (int some = 0; some < ((what == MESH_ALLREDUCE || what == MESH_ALLGATHER) && n > 2 ? 2 : 1); some++)
                for (size_t o = 0; o < sizeof operands / sizeof *operands; o++) {
                  const uint64_t even = 0x5555555555555555ull & (n < 64 ? (1ull << n) - 1 : ~0ull);
                  uint32_t parts, moves;
                  mesh_compile_room(&trees, chunks, &parts, &moves);
                  uint32_t *segment = malloc((parts + 1) * sizeof *segment), *first = malloc((n + 1) * sizeof *first);
                  double *log_weight = malloc((parts + 1) * sizeof *log_weight);
                  struct mesh_move *move = malloc((moves + 1) * sizeof *move);
                  struct mesh_program program;
                  char label[160];
                  snprintf(label, sizeof label, "%s %s %s%s root=%u chunks=%u contributors=%s operand=%llux%u", maps[m].name,
                           packed ? "packed" : "shortest", kinds[what], what == MESH_ALLREDUCE ? (whole ? " whole" : " two-phase") : "", r, chunks,
                           some ? "even" : "all", (unsigned long long)operands[o].elements, operands[o].element_bytes);
                  count++; of_kind[what]++;
                  if (!segment || !first || !log_weight || !move ||
                      mesh_compile(&trees, what, r, whole, chunks, some ? &even : NULL, &program, segment, log_weight, first, move)) {
                    printf("%s: does not compile\n", label);
                    failed++;
                  } else {
                    struct tally t = {label, {0}};
                    const uint32_t warnings = mesh_program_advise(&program, wire, operands[o], warned, &t);
                    for (int c = 0; c < CHECKS; c++) by_kind[what][c] += t.by_check[c];
                    const int free_by_priorities = !t.by_check[0] && !t.by_check[1], checked = !t.by_check[0] && warnings != UINT32_MAX;
                    priorities_free += free_by_priorities; admissible += checked && !t.by_check[3]; unchecked += !checked;
                    char verdict[96];
                    snprintf(verdict, sizeof verdict, "; deadlock-free by priorities: %s; admissible on the wire: %s", free_by_priorities ? "yes" : "no",
                             checked ? (t.by_check[3] ? "no" : "yes") : "unchecked");
                    summary(&t, warnings, total, verdict);
                  }
                  free(segment); free(first); free(log_weight); free(move);
                }
      }
      free(root); free(parent); free(weight);
    }
  printf("\nprograms %llu, not compiled %llu, deadlock-free by priorities %llu, admissible on the wire %llu, wire unchecked %llu\n",
         (unsigned long long)count, (unsigned long long)failed, (unsigned long long)priorities_free, (unsigned long long)admissible,
         (unsigned long long)unchecked);
  printf("program warnings:");
  for (int c = 0; c < CHECKS; c++) printf(" %s %llu", checks[c], (unsigned long long)total[c]);
  printf("\n");
  for (int k = 0; k < KINDS; k++) {
    printf("%s (%llu programs):", kinds[k], (unsigned long long)of_kind[k]);
    for (int c = 0; c < CHECKS; c++) printf(" %s %llu", checks[c], (unsigned long long)by_kind[k][c]);
    printf("\n");
  }
  return 0;
}

/* a group on each map: its collectives along the map's trees (shortest or packed), its exchanges (an all-to-all, a gather
   to rank 0, a scatter from rank n - 1, a counted all-to-all, and two groups' point exchanges: each rank to the next at
   3,000 float32 and to the one before at 1,000,003) along the map's routing table (the solver's from the packed trees,
   or the shortest trees' one route a pair), every step's peer checked against the map's links, and the group's
   advisories (mesh_group_advise: projection, causality with each forwarded SEND piece after its own landing, hazards) */
static void groups(const struct mesh_wire *wire) {
  static const uint32_t kind[] = {MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE, MESH_GROUP_EXCHANGE, MESH_GROUP_EXCHANGE,
                                  MESH_GROUP_EXCHANGE, MESH_GROUP_EXCHANGE, MESH_GROUP_EXCHANGE, MESH_GROUP_EXCHANGE};
  enum { CALLS = sizeof kind / sizeof *kind };
  uint64_t count = 0, total[CHECKS] = {0}, steps = 0, relayed = 0;
  for (size_t m = 0; m < sizeof maps / sizeof *maps; m++)
    for (int packed = 0; packed < 2; packed++) {
      const uint32_t n = maps[m].n;
      uint32_t link[64][2], *root, *parent;
      double *weight;
      const uint32_t trees_count = map_trees(m, packed, link, &root, &parent, &weight);
      const struct mesh_trees trees = {n, trees_count, root, parent, weight};
      struct rows rows[CALLS];
      struct table table = {0};
      struct mesh_block blocks[2 * MOST * MOST];
      memset(rows, 0, sizeof rows);
      int failed = !trees_count || table_of(m, packed, &trees, link, &table) ||
                   rows_collective(rows + 0, &trees, MESH_ALLREDUCE, 0, 0, 2, (struct mesh_operand){0, 4, 4096}) ||
                   rows_collective(rows + 1, &trees, MESH_ALLGATHER, 0, 0, 1, (struct mesh_operand){0, 4, 1000003}) ||
                   rows_collective(rows + 2, &trees, MESH_BROADCAST, n - 1, 0, 3, (struct mesh_operand){1, 2, 1u << 20});
      failed = failed || rows_exchange(rows + 3, &table.routing, n, blocks, blocks_all(n, -1, 0, blocks), 2) ||
               rows_exchange(rows + 4, &table.routing, n, blocks, blocks_all(n, 0, 0, blocks), 2) ||
               rows_exchange(rows + 5, &table.routing, n, blocks, blocks_all(n, (int32_t)n - 1, 1, blocks), 2) ||
               rows_exchange(rows + 6, &table.routing, n, blocks, blocks_counted(n, blocks), 4) ||
               rows_exchange(rows + 7, &table.routing, n, blocks, blocks_point(n, 1, 3000 * 4, blocks), 0) ||
               rows_exchange(rows + 8, &table.routing, n, blocks, blocks_point(n, n - 1, 1000003 * 4, blocks), 0);
      /* every step of the exchanges between two ranks the map links, and how many a member between a block's ends makes */
      struct mesh_link_map map = {MESH_LINKS_GRAPH, n, family_cables(maps[m].family, n, link), link, NULL, 0, NULL, NULL, NULL};
      uint32_t unlinked = 0, forwarded = 0, routes = 0;
      for (int i = 3; !failed && i < CALLS; i++)
        for (uint32_t r = 0; r < n; r++)
          for (uint32_t k = 0; k < rows[i].count[r]; k++) {
            unlinked += !mesh_link_between(&map, r, rows[i].steps[r][k].peer);
            forwarded += rows[i].steps[r][k].op == MESH_STEP_SEND && rows[i].steps[r][k].buffer == (i == 6 ? 4u : i >= 7 ? 0u : 2u);
            steps++;
          }
      for (uint32_t k = 0; !failed && k < table.routing.routes; k++) routes += table.routing.start[k + 1] - table.routing.start[k] > 2;
      relayed += forwarded;
      if (!failed) printf("group %s %s: routes %u (%u through a member), exchange steps between unlinked ranks %u, forwarded SENDs %u\n",
                          maps[m].name, packed ? "packed" : "shortest", table.routing.routes, routes, unlinked, forwarded);
      for (int gpu = 0; gpu < 2; gpu++) {
        char label[192];
        snprintf(label, sizeof label, "group %s %s (allreduce, allgather, broadcast; all-to-all, gather, scatter, counted and two point "
                 "exchanges routed) %s", maps[m].name, packed ? "packed" : "shortest", gpu ? "GPU polls NCCL_MESH_POLLS" : "host");
        if (failed) { printf("%s: does not compile\n", label); break; }
        group_advise(label, rows, kind, CALLS, n, gpu ? NCCL_MESH_POLLS : 0, gpu ? NCCL_MESH_POLLS : 0, wire, total);
        count++;
      }
      for (int i = 0; i < CALLS; i++) rows_free(rows + i);
      table_free(&table);
      free(root); free(parent); free(weight);
    }
  printf("\ngroups %llu, exchange steps %llu, forwarded SENDs %llu, warnings:", (unsigned long long)count, (unsigned long long)steps,
         (unsigned long long)relayed);
  for (int c = 0; c < CHECKS; c++) printf(" %s %llu", checks[c], (unsigned long long)total[c]);
  printf("\n");
}

/* each mutation's tally and whether its expected check fired */
static uint32_t tried, fired;
static void expect(const struct tally *t, int check) {
  tried++; fired += t->by_check[check] > 0;
  printf("%s: %s %s\n\n", t->label, checks[check], t->by_check[check] ? "fired" : "DID NOT FIRE");
}
static int mutate(const struct mesh_wire *wire) {
  uint32_t link[64][2], *root, *parent;
  double *weight;
  const uint32_t trees_count = map_trees(0, 0, link, &root, &parent, &weight);
  const struct mesh_trees trees = {2, trees_count, root, parent, weight};
  uint32_t parts, moves, segment[64], first[3];
  double log_weight[64];
  struct mesh_move move[256], kept[256];
  struct mesh_program p;
  mesh_compile_room(&trees, 1, &parts, &moves);
  if (!trees_count || moves > 256 || parts > 64 || mesh_compile(&trees, MESH_ALLREDUCE, 0, 0, 1, NULL, &p, segment, log_weight, first, move)) return 1;
  memcpy(kept, move, sizeof move);
  const struct mesh_operand small = {0, 4, 4096}, big = {0, 4, 1000003}, wide = {0, 600000, 3};
  struct tally t;
#define PROGRAM(name, operand, check) do { t = (struct tally){name, {0}}; mesh_program_advise(&p, wire, operand, warned, &t); expect(&t, check); \
    memcpy(move, kept, sizeof move); } while (0)
  move[0].parts = 2; move[0].part = 0;
  PROGRAM("M1 rank 0's first SEND carries two parts", small, 0);
  for (uint32_t i = first[1]; i < first[2]; i++) if (move[i].op != MESH_STEP_SEND) { move[i].round++; break; }
  PROGRAM("M2 rank 1's first receive a round later", small, 1);
  for (uint32_t r = 0; r < 2; r++) { struct mesh_move *a = move + first[r]; const struct mesh_move s = a[0]; a[0] = a[1]; a[1] = s; }
  PROGRAM("M3 each rank's round-0 receive before its SEND", big, 3);
  /* a receive into the part a SEND reads at the SEND's own positions is the executors' in-place order (one position
     published before it is consumed: mesh_compile's two-node direct exchange), no hazard; the receive here covers
     the whole operand, so its pieces meeting the SEND's lie at later positions, with no order from the SEND's */
  for (uint32_t i = first[0]; i < first[2]; i++)
    if (!move[i].round && (i < first[1]) != (move[i].op == MESH_STEP_SEND)) { move[i].part = 0; move[i].parts = p.parts; }
  PROGRAM("M4 rank 0's round-0 receive into the whole operand, the part it sends among it, past that SEND's positions", big, 4);
  PROGRAM("M5 an element wider than a slot", wide, 2);
#undef PROGRAM
  struct rows rows[3];
  uint32_t kind[3] = {MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE};
  memset(rows, 0, sizeof rows);
#define GROUP(name, calls, n, polls, check) do { t = (struct tally){name, {0}}; struct mesh_group_call call[3]; \
    for (int i = 0; i < (calls); i++) call[i] = (struct mesh_group_call){kind[i], 0, rows[i].count, (const struct mesh_step *const *)rows[i].steps}; \
    const struct mesh_group group = {n, calls, polls, NCCL_MESH_POLLS, call}; mesh_group_advise(&group, wire, warned, &t); expect(&t, check); \
    for (int i = 0; i < 3; i++) rows_free(rows + i); } while (0)
  struct rows reduce, gather;
  memset(&reduce, 0, sizeof reduce); memset(&gather, 0, sizeof gather);
  if (rows_collective(&reduce, &trees, MESH_ALLREDUCE, 0, 0, 1, small) || rows_collective(&gather, &trees, MESH_ALLGATHER, 0, 0, 1, small)) return 1;
  for (int i = 0; i < 2; i++)
    for (int r = 0; r < 2; r++) {
      const struct rows *from = (i == 0) == (r == 0) ? &reduce : &gather;
      rows[i].count[r] = from->count[r];
      rows[i].steps[r] = malloc((from->count[r] + 1) * sizeof **rows[i].steps);
      if (rows[i].steps[r]) memcpy(rows[i].steps[r], from->steps[r], from->count[r] * sizeof **rows[i].steps);
    }
  GROUP("G1 rank 1 calls the all-gather before the all-reduce, rank 0 after", 2, 2, NCCL_MESH_POLLS, 0);
  struct table pair = {0};
  struct mesh_block blocks[2 * MOST * MOST];
  kind[0] = MESH_GROUP_EXCHANGE;
  if (table_of(0, 0, &trees, link, &pair) || rows_exchange(rows + 0, &pair.routing, 2, blocks, blocks_point(2, 1, 4096, blocks), 0)) return 1;
  for (uint32_t k = 0; k < rows[0].count[1]; k++)
    if (rows[0].steps[1][k].op != MESH_STEP_SEND) { rows[0].steps[1][k] = rows[0].steps[1][--rows[0].count[1]]; break; }
  GROUP("G2 rank 0 sends rank 1 a point exchange's block, rank 1 receives none", 1, 2, NCCL_MESH_POLLS, 0);
  uint32_t three_link[64][2], *three_root, *three_parent;
  double *three_weight;
  const uint32_t three_count = map_trees(1, 0, three_link, &three_root, &three_parent, &three_weight);
  const struct mesh_trees three = {4, three_count, three_root, three_parent, three_weight};
  struct table line_table = {0};
  if (!three_count || table_of(1, 0, &three, three_link, &line_table)) return 1;
  const uint32_t made = blocks_all(4, -1, 0, blocks);
  for (uint32_t r = 0; r < 4; r++) {
    struct mesh_block skewed[2 * MOST * MOST];
    memcpy(skewed, blocks, made * sizeof *skewed);
    for (uint32_t i = 0; r == 1 && i < made; i++) skewed[i].elements += skewed[i].from == 0 && skewed[i].to == 1;
    if (!(rows[0].steps[r] = malloc((mesh_forward_room(&line_table.routing, skewed, made) + 1) * sizeof **rows[0].steps))) return 1;
    rows[0].count[r] = mesh_forward_steps(&line_table.routing, r, skewed, made, 2, rows[0].steps[r], NULL, NULL);
  }
  GROUP("G3 rank 1 counts one element more from rank 0 than rank 0 sends it", 1, 4, NCCL_MESH_POLLS, 0);
  kind[0] = MESH_GROUP_COLLECTIVE;
  if (rows_collective(rows + 0, &trees, MESH_ALLREDUCE, 0, 0, 1, small)) return 1;
  GROUP("G4 a GPU group of polls 0", 1, 2, 0, 6);
  if (rows_collective(rows + 0, &trees, MESH_ALLREDUCE, 0, 0, 1, small)) return 1;
  GROUP("G5 a GPU group of polls past NCCL_MESH_POLLS", 1, 2, NCCL_MESH_POLLS + 1, 6);
  kind[0] = MESH_GROUP_EXCHANGE;
  if (rows_exchange(rows + 0, &line_table.routing, 4, blocks, blocks_all(4, -1, 0, blocks), 2)) return 1;
  for (uint32_t r = 0; r < 4; r++)
    for (uint32_t k = 0; k < rows[0].count[r]; k++) rows[0].steps[r][k].round -= rows[0].steps[r][k].round > 0;
  GROUP("G6 line4's all-to-all, each forwarded part sent on in the round it lands", 1, 4, NCCL_MESH_POLLS, 4);
  if (rows_exchange(rows + 0, &line_table.routing, 4, blocks, blocks_all(4, -1, 0, blocks), 2)) return 1;
  for (uint32_t r = 0; r < 4; r++)
    for (uint32_t k = 0; k < rows[0].count[r]; k++)
      if (rows[0].steps[r][k].op == MESH_STEP_SEND && rows[0].steps[r][k].buffer == 2) rows[0].steps[r][k].first++;
  GROUP("G7 line4's all-to-all, each forwarded part sent on from an element past what landed", 1, 4, NCCL_MESH_POLLS, 4);
  table_free(&line_table); table_free(&pair);
  free(three_root); free(three_parent); free(three_weight);
#undef GROUP
  rows_free(&reduce); rows_free(&gather);
  free(root); free(parent); free(weight);
  printf("mutations %u, each firing its check: %u\n", tried, fired);
  return 0;
}

int main(int argc, char **argv) {
  struct mesh_wire wire;
  mesh_wire_of(16384, &wire);
  printf("wire: slot %llu, small %llu, block %llu, lag %u\n", (unsigned long long)wire.slot, (unsigned long long)wire.small,
         (unsigned long long)wire.block, wire.lag);
  if (argc > 1 && !strcmp(argv[1], "mutate")) return mutate(&wire);
  programs(&wire);
  groups(&wire);
  return 0;
}

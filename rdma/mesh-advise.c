/* design/transport.md: mesh_program_advise over every collective mesh_compile makes on the loopback
   fabric's maps (mesh-loop.c family_cables: the pair, line 4, ring 5, star 8, torus 6 and 8), along each map's
   shortest-path trees and its packed trees (as libnccl-mesh makes them, nccl-mesh.c comm_routes, compile_form), at
   chunks 1 to 4, every root of a rooted collective, the all-reduce whole and two-phase, every node or the even nodes
   contributing, over operands of 1, 4,096 and 1,000,003 float32 and 2^20 half, cut on pages of 16 KiB (the region's on
   Apple silicon: libnccl-mesh's own cuts lie on them, and a SEND's or COPY's piece off one fails its call); mesh_group_advise
   over a group on each map and tree set (collectives, both counted all-to-alls, point-to-point calls each way at two
   sizes).  It prints each one's advisories and a summary line, then the totals; it refuses
   nothing, exits 0.  `mesh-advise mutate`: programs and groups broken on purpose, each named with the check it must
   fire and whether it did. */
#include "mesh-plan.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHECKS = 7, KINDS = 5, SHOWN = 4, MOST = 8 };
static const char *const checks[CHECKS] = {"projection", "priorities", "alignment", "causality", "hazard", "invocation", "bound"};
enum { PAGE = 16384 };
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
  struct mesh_link_map map = {MESH_LINKS_GRAPH, n, 0, link, NULL, 0, NULL, NULL};
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
  uint64_t sizes[MOST];
  for (uint32_t s = 0; s < trees->nodes; s++) sizes[s] = operand.elements / trees->nodes + (s < operand.elements % trees->nodes);
  const int nccl = what == MESH_REDUCE_SCATTER || what == MESH_ALLGATHER;
  for (uint32_t r = 0; !failed && r < trees->nodes; r++) {
    failed = !(c->steps[r] = malloc((first[r + 1] - first[r] + 1) * sizeof **c->steps));
    if (!failed) c->count[r] = mesh_program_steps_cut(&program, r, operand, nccl ? sizes : NULL, PAGE, c->steps[r]);
  }
  free(segment); free(first); free(log_weight); free(move);
  return failed;
}
/* ragged sizes, each pair's alike at both ends: a hosted all-to-all's elements from a to b, a counted one's entries */
static uint64_t ragged(int hosted, uint32_t a, uint32_t b) { return a == b ? 0 : hosted ? 1 + (a * 5 + b * 3) % 7 * 700 : 1 + (a + 2 * b) % 3; }
static int rows_counted(struct rows *c, uint32_t n, int hosted, uint32_t skew) {
  const struct mesh_operand element = {1, 4, hosted ? 0 : 64 * 8}, counts = {8, 8, 0};
  for (uint32_t r = 0; r < n; r++) {
    uint64_t segments[2 * MOST];
    for (uint32_t q = 0; q < n; q++) { segments[q] = ragged(hosted, r, q); segments[n + q] = ragged(hosted, q, r) + (r == 1 && q == 0 ? skew : 0); }
    if (!(c->steps[r] = malloc(4 * (size_t)n * sizeof **c->steps))) return 1;
    c->count[r] = mesh_counted_steps(n, r, segments, hosted, element, counts, c->steps[r]);
  }
  return 0;
}
/* a point-to-point call at every rank: a SEND to the rank `shift` on, or a receive from the rank `shift` back */
static int rows_point(struct rows *c, uint32_t n, int send, uint32_t shift, struct mesh_operand operand) {
  for (uint32_t r = 0; r < n; r++) {
    if (!(c->steps[r] = malloc(sizeof **c->steps))) return 1;
    c->steps[r][0] = (struct mesh_step){send ? MESH_STEP_SEND : MESH_STEP_COPY, send ? (r + shift) % n : (r + n - shift % n) % n, 0, 0, 0, operand};
    c->count[r] = 1;
  }
  return 0;
}
static uint32_t group_advise(const char *label, struct rows *rows, const uint32_t *kind, uint32_t calls, uint32_t n, uint32_t polls,
                             uint32_t bound, uint64_t *total) {
  struct mesh_group_call call[16];
  for (uint32_t i = 0; i < calls; i++) call[i] = (struct mesh_group_call){kind[i], 0, rows[i].count, (const struct mesh_step *const *)rows[i].steps};
  const struct mesh_group group = {n, calls, polls, bound, call, PAGE};
  struct tally t = {label, {0}};
  const uint32_t warnings = mesh_group_advise(&group, warned, &t);
  summary(&t, warnings, total, "");
  return warnings;
}

static int programs(void) {
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
                    const uint32_t warnings = mesh_program_advise(&program, operands[o], PAGE, warned, &t);
                    for (int c = 0; c < CHECKS; c++) by_kind[what][c] += t.by_check[c];
                    const int free_by_priorities = !t.by_check[0] && !t.by_check[1], checked = !t.by_check[0] && warnings != UINT32_MAX;
                    priorities_free += free_by_priorities; admissible += checked && !t.by_check[3]; unchecked += !checked;
                    char verdict[96];
                    snprintf(verdict, sizeof verdict, "; deadlock-free by priorities: %s; admissible on the lists: %s", free_by_priorities ? "yes" : "no",
                             checked ? (t.by_check[3] ? "no" : "yes") : "unchecked");
                    summary(&t, warnings, total, verdict);
                  }
                  free(segment); free(first); free(log_weight); free(move);
                }
      }
      free(root); free(parent); free(weight);
    }
  printf("\nprograms %llu, not compiled %llu, deadlock-free by priorities %llu, admissible on the lists %llu, lists unchecked %llu\n",
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

static void groups(void) {
  static const uint32_t kind[] = {MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE, MESH_GROUP_HOSTED, MESH_GROUP_COUNTED,
                                  MESH_GROUP_POINT, MESH_GROUP_POINT, MESH_GROUP_POINT, MESH_GROUP_POINT};
  enum { CALLS = sizeof kind / sizeof *kind };
  uint64_t count = 0, total[CHECKS] = {0};
  for (size_t m = 0; m < sizeof maps / sizeof *maps; m++)
    for (int packed = 0; packed < 2; packed++) {
      const uint32_t n = maps[m].n;
      uint32_t link[64][2], *root, *parent;
      double *weight;
      const uint32_t trees_count = map_trees(m, packed, link, &root, &parent, &weight);
      const struct mesh_trees trees = {n, trees_count, root, parent, weight};
      struct rows rows[CALLS];
      memset(rows, 0, sizeof rows);
      const int failed = !trees_count || rows_collective(rows + 0, &trees, MESH_ALLREDUCE, 0, 0, 2, (struct mesh_operand){0, 4, 4096}) ||
                         rows_collective(rows + 1, &trees, MESH_ALLGATHER, 0, 0, 1, (struct mesh_operand){0, 4, 1000003}) ||
                         rows_collective(rows + 2, &trees, MESH_BROADCAST, n - 1, 0, 3, (struct mesh_operand){1, 2, 1u << 20}) ||
                         rows_counted(rows + 3, n, 1, 0) || rows_counted(rows + 4, n, 0, 0) ||
                         rows_point(rows + 5, n, 1, 1, (struct mesh_operand){0, 4, 3000}) || rows_point(rows + 6, n, 0, 1, (struct mesh_operand){0, 4, 3000}) ||
                         rows_point(rows + 7, n, 1, n - 1, (struct mesh_operand){0, 4, 1000003}) ||
                         rows_point(rows + 8, n, 0, n - 1, (struct mesh_operand){0, 4, 1000003});
      for (int gpu = 0; gpu < 2; gpu++) {
        char label[160];
        snprintf(label, sizeof label, "group %s %s (allreduce, allgather, broadcast, all-to-all hosted and counted, 4 point-to-point) %s", maps[m].name,
                 packed ? "packed" : "shortest", gpu ? "GPU polls NCCL_MESH_POLLS" : "host");
        if (failed) { printf("%s: does not compile\n", label); break; }
        group_advise(label, rows, kind, CALLS, n, gpu ? NCCL_MESH_POLLS : 0, gpu ? NCCL_MESH_POLLS : 0, total);
        count++;
      }
      for (int i = 0; i < CALLS; i++) rows_free(rows + i);
      free(root); free(parent); free(weight);
    }
  printf("\ngroups %llu, warnings:", (unsigned long long)count);
  for (int c = 0; c < CHECKS; c++) printf(" %s %llu", checks[c], (unsigned long long)total[c]);
  printf("\n");
}

/* each mutation's tally and whether its expected check fired */
static uint32_t tried, fired;
static void expect(const struct tally *t, int check) {
  tried++; fired += t->by_check[check] > 0;
  printf("%s: %s %s\n\n", t->label, checks[check], t->by_check[check] ? "fired" : "DID NOT FIRE");
}
static int mutate(void) {
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
  const struct mesh_operand small = {0, 4, 4096}, big = {0, 4, 1000003};
  struct tally t;
#define PROGRAM(name, program, operand, check) do { t = (struct tally){name, {0}}; mesh_program_advise(&(program), operand, PAGE, warned, &t); \
    expect(&t, check); memcpy(move, kept, sizeof move); } while (0)
  move[0].parts = 2; move[0].part = 0;
  PROGRAM("M1 rank 0's first SEND carries two parts", p, small, 0);
  for (uint32_t i = first[1]; i < first[2]; i++) if (move[i].op != MESH_STEP_SEND) { move[i].round++; break; }
  PROGRAM("M2 rank 1's first receive a round later", p, big, 1);
  for (uint32_t r = 0; r < 2; r++) { struct mesh_move *a = move + first[r]; const struct mesh_move s = a[0]; a[0] = a[1]; a[1] = s; }
  PROGRAM("M3 each rank's round-0 receive before its SEND", p, big, 3);
  /* a combine into the part a SEND reads on the same channel waits for that SEND's bytes sent (the executors' order:
     mesh_compile's two-node direct exchange), no hazard; a COPY lands in place as its peer publishes it, which nothing
     orders after this rank's SEND has landed at that peer */
  for (uint32_t i = first[0]; i < first[2]; i++)
    if (!move[i].round && (i < first[1]) != (move[i].op == MESH_STEP_SEND)) {
      move[i].part = 0; move[i].parts = p.parts;
      if (i < first[1]) move[i].op = MESH_STEP_COPY;
    }
  PROGRAM("M4 rank 0's round-0 receive a COPY into the whole operand, the part it sends among it", p, big, 4);
  uint32_t split_segment[64], split_first[3];
  double split_weight[64];
  struct mesh_move split_move[256];
  struct mesh_program split;
  if (mesh_compile(&trees, MESH_ALLGATHER, 0, 0, 1, NULL, &split, split_segment, split_weight, split_first, split_move)) return 1;
  PROGRAM("M5 an all-gather whose second segment starts off a page (1,000 float32)", split, ((struct mesh_operand){0, 4, 1000}), 2);
#undef PROGRAM
  struct rows rows[3];
  uint32_t kind[3] = {MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE, MESH_GROUP_COLLECTIVE};
  memset(rows, 0, sizeof rows);
#define GROUP(name, calls, n, polls, check) do { t = (struct tally){name, {0}}; struct mesh_group_call call[3]; \
    for (int i = 0; i < (calls); i++) call[i] = (struct mesh_group_call){kind[i], 0, rows[i].count, (const struct mesh_step *const *)rows[i].steps}; \
    const struct mesh_group group = {n, calls, polls, NCCL_MESH_POLLS, call, PAGE}; mesh_group_advise(&group, warned, &t); expect(&t, check); \
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
  kind[0] = kind[1] = MESH_GROUP_POINT;
  if (rows_point(rows + 0, 2, 1, 1, small) || rows_point(rows + 1, 2, 1, 1, small)) return 1;
  rows[1].count[1] = 0;
  for (int r = 0; r < 2; r++) rows[0].steps[r][0].op = r ? MESH_STEP_COPY : MESH_STEP_SEND;
  GROUP("G2 rank 0 sends rank 1 two messages, rank 1 receives one", 2, 2, NCCL_MESH_POLLS, 0);
  kind[0] = MESH_GROUP_HOSTED;
  if (rows_counted(rows + 0, 3, 1, 1)) return 1;
  GROUP("G3 rank 1 counts one element more from rank 0 than rank 0 sends it", 1, 3, NCCL_MESH_POLLS, 0);
  kind[0] = MESH_GROUP_COLLECTIVE;
  if (rows_collective(rows + 0, &trees, MESH_ALLREDUCE, 0, 0, 1, small)) return 1;
  GROUP("G4 a GPU group of polls 0", 1, 2, 0, 6);
  if (rows_collective(rows + 0, &trees, MESH_ALLREDUCE, 0, 0, 1, small)) return 1;
  GROUP("G5 a GPU group of polls past NCCL_MESH_POLLS", 1, 2, NCCL_MESH_POLLS + 1, 6);
#undef GROUP
  rows_free(&reduce); rows_free(&gather);
  free(root); free(parent); free(weight);
  printf("mutations %u, each firing its check: %u\n", tried, fired);
  return 0;
}

int main(int argc, char **argv) {
  printf("pages of %u bytes\n", (unsigned)PAGE);
  if (argc > 1 && !strcmp(argv[1], "mutate")) return mutate();
  programs();
  groups();
  return 0;
}

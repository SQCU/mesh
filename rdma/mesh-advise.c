/* design/collectives-dataflow.md#advisories: mesh_program_advise over every collective mesh_compile makes on the loopback
   fabric's maps (mesh-loop.c family_cables: the pair, line 4, ring 5, star 8, torus 6 and 8), along each map's
   shortest-path trees and its packed trees (as libnccl-mesh makes them, nccl-mesh.c comm_routes, compile_form), at
   chunks 1 to 4, every root of a rooted collective, the all-reduce whole and two-phase, every node or the even nodes
   contributing, over operands of 1, 4,096 and 1,000,003 float32 and 2^20 half, on mesh_wire_of's wire at a 16 KiB
   block.  It prints each program's advisories and a summary line, then the totals; it refuses nothing, exits 0. */
#include "mesh-plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHECKS = 5, KINDS = 5, SHOWN = 4 };
static const char *const checks[CHECKS] = {"projection", "priorities", "depth", "causality", "hazard"};
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

int main(void) {
  static const struct { const char *name, *family; uint32_t n; } maps[] = {
    {"pair", "line", 2}, {"line4", "line", 4}, {"ring5", "ring", 5}, {"star8", "star", 8}, {"torus6", "torus", 6}, {"torus8", "torus", 8}};
  static const struct mesh_operand operands[] = {{0, 4, 1}, {0, 4, 4096}, {0, 4, 1000003}, {1, 2, 1u << 20}};
  struct mesh_wire wire;
  mesh_wire_of(16384, MESH_BUFFSIZE, 0, &wire);
  printf("wire: slot %llu, small %llu, block %llu, lag %u\n", (unsigned long long)wire.slot, (unsigned long long)wire.small,
         (unsigned long long)wire.block, wire.lag);
  uint64_t programs = 0, failed = 0, unchecked = 0, priorities_free = 0, admissible = 0, total[CHECKS] = {0}, by_kind[KINDS][CHECKS] = {{0}};
  uint64_t of_kind[KINDS] = {0};
  for (size_t m = 0; m < sizeof maps / sizeof *maps; m++) {
    const uint32_t n = maps[m].n;
    uint32_t link[64][2];
    struct mesh_link_map map = {MESH_LINKS_GRAPH, n, 0, link, NULL, 0, NULL, NULL};
    map.links = family_cables(maps[m].family, n, link);
    if (mesh_link_map_check(&map)) { printf("%s: not a map\n", maps[m].name); continue; }
    for (int packed = 0; packed < 2; packed++) {
      const uint32_t most = packed ? 2 : 1;
      uint32_t *root = calloc((size_t)n * most + 1, sizeof *root), *parent = calloc((size_t)n * n * most + 1, sizeof *parent);
      double *weight = calloc((size_t)n * most + 1, sizeof *weight);
      if (!root || !parent || !weight) return 0;
      uint32_t count = packed ? mesh_trees_pack(&map, 0.1, most, root, parent, weight) : mesh_trees_shortest(&map, root, parent, weight);
      for (uint32_t t = 0; packed && t < count; t++) weight[t] = 0;
      const struct mesh_trees trees = {n, count, root, parent, weight};
      if (!count) { printf("%s %s: no trees\n", maps[m].name, packed ? "packed" : "shortest"); free(root); free(parent); free(weight); continue; }
      for (uint32_t what = 0; what < KINDS; what++) {
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
                           packed ? "packed" : "shortest", kinds[what], what == MESH_ALLREDUCE ? (whole ? " whole" : " two-phase") : "", r, chunks, some ? "even" : "all", (unsigned long long)operands[o].elements, operands[o].element_bytes);
                  programs++; of_kind[what]++;
                  if (!segment || !first || !log_weight || !move ||
                      mesh_compile(&trees, what, r, whole, chunks, some ? &even : NULL, &program, segment, log_weight, first, move)) {
                    printf("%s: does not compile\n", label);
                    failed++;
                  } else {
                    struct tally t = {label, {0}};
                    const uint32_t warnings = mesh_program_advise(&program, &wire, operands[o], warned, &t);
                    for (int c = 0; c < CHECKS; c++) {
                      total[c] += t.by_check[c]; by_kind[what][c] += t.by_check[c];
                      if (t.by_check[c] > SHOWN) printf("  %s %s: %u more\n", label, checks[c], t.by_check[c] - SHOWN);
                    }
                    const int free_by_priorities = !t.by_check[0] && !t.by_check[1];
                    const int checked = !t.by_check[0] && warnings != UINT32_MAX;
                    priorities_free += free_by_priorities; admissible += checked && !t.by_check[3]; unchecked += !checked;
                    printf("%s: %u warnings (projection %u, priorities %u, depth %u, causality %u, hazard %u); deadlock-free by priorities: %s; "
                           "admissible on the wire: %s\n", label, warnings, t.by_check[0], t.by_check[1], t.by_check[2], t.by_check[3],
                           t.by_check[4], free_by_priorities ? "yes" : "no", checked ? (t.by_check[3] ? "no" : "yes") : "unchecked");
                  }
                  free(segment); free(first); free(log_weight); free(move);
                }
      }
      free(root); free(parent); free(weight);
    }
  }
  printf("\nprograms %llu, not compiled %llu, deadlock-free by priorities %llu, admissible on the wire %llu, wire unchecked %llu\n",
         (unsigned long long)programs, (unsigned long long)failed, (unsigned long long)priorities_free, (unsigned long long)admissible,
         (unsigned long long)unchecked);
  printf("warnings:");
  for (int c = 0; c < CHECKS; c++) printf(" %s %llu", checks[c], (unsigned long long)total[c]);
  printf("\n");
  for (int k = 0; k < KINDS; k++) {
    printf("%s (%llu programs):", kinds[k], (unsigned long long)of_kind[k]);
    for (int c = 0; c < CHECKS; c++) printf(" %s %llu", checks[c], (unsigned long long)by_kind[k][c]);
    printf("\n");
  }
  return 0;
}

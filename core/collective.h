/* collective.h: the collective library. A program declares its whole function's calls before it runs; the library
   compiles them into one plan of page writes and hands each node's bridge its page lists. Sources: design/data-model.md
   section 5 (the routing function and its plan; "It asks no peer to choose a place or grant permission between
   steps"; reuse from the work peers send back, "We do not ask for a separate receipt"), section 6 (page lists and
   steps); the rubric's interface table: between collective and program cross "the NCCL API, the device-function
   header, memory through ncclMemAlloc". */
#ifndef CORE_COLLECTIVE_H
#define CORE_COLLECTIVE_H
#include "bridge.h"
#include <stddef.h>

/* Memory: every buffer a call touches is pages of the region. */
void *collective_alloc(size_t bytes);    /* ncclMemAlloc */
void collective_free(void *memory);      /* ncclMemFree */

/* A declared call (the NCCL API's collectives, declared, never eager). */
enum { COLLECTIVE_ALLGATHER, COLLECTIVE_ALLREDUCE, COLLECTIVE_BROADCAST, COLLECTIVE_REDUCESCATTER };
struct collective_call {
  uint32_t what;
  const void *send; void *receive;       /* the program's buffers (region pages) */
  uint64_t elements; uint32_t element_bytes;
  const uint64_t *segments;              /* each rank's elements of an all-gather or reduce-scatter, or NULL */
};

/* The routing function's inputs (data-model.md section 5): the nodes, the links that join them and their rates. Its
   output is the plan. */
struct collective_link { uint32_t a, b; double bytes_a_second; };
struct collective_fabric { uint32_t nodes, rank; const struct collective_link *links; uint32_t count; };

/* Compile the whole function into each node's page lists and hand them to its bridge (bridge_load). The routing over
   the links, the pages' placement and the page writes' order are the graph module's (rdma/mesh-graph.h, design/
   rubrics.md:422, C9: graph algorithms in one module), composed here with this rank's calls; the same functions of the
   same inputs at every node, so both ends of every link make the same lists. 0, or an errno. */
int collective_plan(const struct collective_fabric *fabric, const struct collective_call *calls, uint32_t count);

/* The device-function header's two operations, in the program's own GPU work: a producer's publication (one system
   fence, then one store of its link's supplied word), and a consumer's presence (its link's landed word reaching the
   count the plan gives). */
#endif

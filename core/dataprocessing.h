/* dataprocessing.h: functions over pages, the program side. A function reads named pages and writes named pages as GPU
   work; a program is an ordered set of functions and declared collective calls, compiled once by collective_plan and
   then run with no decision at run time. Any DNN is such a program (an LSTM, an FFN, Gemma): GOAL.md, "the lowest
   attainable overhead collective operations for arbitrary programs on Apple GPUs ... on any number of Macs in any
   cabling"; acceptance: no overhead over a single program moving the same pages in the same order (design/
   page-maps-2026-11-02.md:849). Sources: design/data-model.md sections 5-6 (each step reads its named inputs and writes
   its named results when they are ready); the rubric's interface table (between program and bridge: nothing). */
#ifndef CORE_DATAPROCESSING_H
#define CORE_DATAPROCESSING_H
#include "collective.h"

/* A function over pages: its kernel, the pages it reads and writes (in the order it uses them), and its grid. A
   contiguous-typed operand names its pages as one run; every other operand is any pages. */
struct page_function {
  const char *kernel;                    /* the GPU function's name in the program's library */
  const bridge_page *reads, *writes; uint32_t read_count, write_count;
  uint32_t grid[3], group[3];
};

/* A program: functions and declared collective calls in program order; `after` names, for each item, the items whose
   results it reads (the dependences the plan orders). */
struct program_item { uint32_t is_call; union { struct page_function function; struct collective_call call; }; };
struct program { const struct program_item *items; uint32_t count; const char *library; };

/* Compile the program with its calls into one plan (collective_plan) and record its GPU work once: 0, or an errno. */
int program_prepare(const struct program *program, const struct collective_fabric *fabric);
/* Run it `repeats` times; the microseconds a run, measured on this rank. */
double program_run(const struct program *program, uint32_t repeats);
#endif

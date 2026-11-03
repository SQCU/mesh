/* dataprocessing.h: the program, the demonstration on the collective: an LSTM tick of `layers` layers, each rank
   holding its share of the units, its gates' products and cell as GPU work over region pages, and every rank's h
   gathered each tick by a declared all-gather. It is the specification's acceptance reference (design/
   page-maps-2026-11-02.md:849: a program "hardcoded to make the same pages transact in the same order without any
   other wrappers"). It uses the collective only through collective.h. */
#ifndef CORE_DATAPROCESSING_H
#define CORE_DATAPROCESSING_H
#include "collective.h"

struct lstm { uint32_t layers, width, rows, units; };   /* units: this rank's share of the width */

/* Declare the program's calls (each tick's gather) and compile the plan: once, before the first tick. */
int lstm_prepare(const struct lstm *shape, const struct collective_fabric *fabric);
/* Run `ticks` ticks; the microseconds a tick, measured on this rank. */
double lstm_run(const struct lstm *shape, uint32_t ticks);
#endif

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Ported for the mesh library (design/page-maps-2026-11-02.md, "The run's solve", 10) from TensorFlow's
   tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc (GreedyBySizeAssignment) and
   internal.cc (CompareBySize), master as fetched 2026-11-02; the algorithm of Y. Pisarchyk and J. Lee, "Efficient
   memory management for deep neural net inference", 2020. Changes from the original:
   - C in place of C++, status codes in place of absl::Status, the records as index and size arrays;
   - the test "tensor allocated_id's usage interval does not intersect the current one's" is apart(context,
     allocated_id, current), the run's exclusion relation over a partial order rather than intervals of one;
   - alignment 1 (the units are pages), so AlignByN is the identity;
   - the "Total size is wrong" consistency check is dropped, and a total past `capacity` is ENOMEM;
   - items already placed (pins) may be given, occupying the line before any item is assigned;
   - stable_sort by CompareBySize is a sort by size descending, then by the given order. */
#include "mesh-graph.h"
#include <errno.h>
#include <stdlib.h>

struct by_size { const uint64_t *size; const size_t *rank; };
/* CompareBySize: the larger first; the given order among equal sizes (stable_sort) */
static int greedy_by_size_order(void *context, const void *left, const void *right) {
  const struct by_size *s = context;
  const size_t a = *(const size_t *)left, b = *(const size_t *)right;
  if (s->size[a] != s->size[b]) return s->size[a] > s->size[b] ? -1 : 1;
  return (s->rank[a] > s->rank[b]) - (s->rank[a] < s->rank[b]);
}

/* mesh-graph.h */
int mesh_graph_greedy_by_size(size_t count, const size_t *items, size_t placed, const size_t *pinned,
                              const uint64_t *size, int (*apart)(const void *context, size_t k, size_t l),
                              const void *context, uint64_t capacity, uint64_t *offset, uint64_t *total) {
  size_t *ordered_records = malloc((count + 1) * sizeof *ordered_records),
         *ordered_allocs = malloc((count + placed + 1) * sizeof *ordered_allocs), *rank = NULL, allocs = 0;
  size_t largest = 0;
  for (size_t i = 0; i < count; i++) if (items[i] + 1 > largest) largest = items[i] + 1;
  for (size_t i = 0; i < placed; i++) if (pinned[i] + 1 > largest) largest = pinned[i] + 1;
  rank = malloc((largest + 1) * sizeof *rank);
  if (!ordered_records || !ordered_allocs || !rank) {
    free(ordered_records); free(ordered_allocs); free(rank);
    return ENOMEM;
  }
  *total = 0;
  /* the line's prior occupants, ordered by offset */
  for (size_t i = 0; i < placed; i++) {
    size_t at = allocs;
    while (at > 0 && offset[ordered_allocs[at - 1]] > offset[pinned[i]]) { ordered_allocs[at] = ordered_allocs[at - 1]; at--; }
    ordered_allocs[at] = pinned[i];
    allocs++;
    if (offset[pinned[i]] + size[pinned[i]] > *total) *total = offset[pinned[i]] + size[pinned[i]];
  }
  for (size_t i = 0; i < count; i++) { ordered_records[i] = items[i]; rank[items[i]] = i; }
  struct by_size order = {size, rank};
  qsort_r(ordered_records, count, sizeof *ordered_records, &order, greedy_by_size_order);
  int result = 0;
  for (size_t r = 0; r < count && !result; r++) {
    const size_t rec = ordered_records[r];
    uint64_t best_diff = UINT64_MAX, best_offset = UINT64_MAX, prev_offset = 0;
    for (size_t a = 0; a < allocs; a++) {
      const size_t allocated_id = ordered_allocs[a];
      if (apart(context, allocated_id, rec)) continue;
      const uint64_t cur_offset = offset[allocated_id];
      if (cur_offset >= prev_offset) {
        const uint64_t diff = cur_offset - prev_offset;
        /* the gap directly left of allocated_id fits, and is the smallest suitable gap so far */
        if (diff >= size[rec] && diff < best_diff) { best_diff = diff; best_offset = prev_offset; }
      }
      if (cur_offset + size[allocated_id] > prev_offset) prev_offset = cur_offset + size[allocated_id];
    }
    /* no suitable gap: after the rightmost item that is not apart from this one */
    if (best_offset == UINT64_MAX) best_offset = prev_offset;
    size_t at = 0;
    while (at < allocs && offset[ordered_allocs[at]] <= best_offset) at++;
    for (size_t y = allocs; y > at; y--) ordered_allocs[y] = ordered_allocs[y - 1];
    ordered_allocs[at] = rec;
    allocs++;
    offset[rec] = best_offset;
    if (best_offset + size[rec] > *total) *total = best_offset + size[rec];
    if (*total > capacity) result = ENOMEM;
  }
  free(ordered_records); free(ordered_allocs); free(rank);
  return result;
}

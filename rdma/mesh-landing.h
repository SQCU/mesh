/* Preparation: packet arrivals projected onto complete numerical elements.
   Equal-width packets become one arithmetic readiness run. Execution consumes
   the resulting constants, never this cursor or the DMA span list. */
#ifndef MESH_LANDING_H
#define MESH_LANDING_H
#include "mesh.h"

struct mesh_landing_cursor { uint64_t packet, bytes, elements; };
struct mesh_landing_run {
  uint64_t first, elements, packet, phase, step;
  uint32_t shift;
};

/* Element j within a run needs packet + ((phase + j * step) >> shift).
   A row split across packets belongs to the run containing its final byte.
   Packet offsets are deliberately irrelevant: the local page view has already
   bound the numerical operand in its logical order. */
static inline int mesh_landing_next(const struct mesh_dma_span *spans, uint64_t count, uint64_t width,
    struct mesh_landing_cursor *cursor, struct mesh_landing_run *out) {
  while (cursor->packet < count) {
    const uint64_t begin = cursor->packet, start = cursor->bytes, first = cursor->elements;
    const uint64_t quantum = spans[begin].bytes;
    const int regular = quantum && !(quantum & (quantum - 1));
    do {
      cursor->bytes += spans[cursor->packet++].bytes;
    } while (regular && cursor->packet < count && (spans[cursor->packet].bytes == quantum ||
             (cursor->packet + 1 == count && spans[cursor->packet].bytes < quantum)));
    cursor->elements = cursor->bytes / width;
    if (cursor->elements == first) continue;
    *out = (struct mesh_landing_run){.first = first, .elements = cursor->elements - first, .packet = begin + 1};
    if (regular) {
      const uint32_t shift = (uint32_t)__builtin_ctzll(quantum), aligned = (uint32_t)__builtin_ctzll(width);
      const uint32_t common = shift < aligned ? shift : aligned;
      const uint64_t last = (first + 1) * width - start - 1;
      out->packet += last >> shift;
      out->phase = (last & (quantum - 1)) >> common;
      out->step = width >> common;
      out->shift = shift - common;
    }
    return 1;
  }
  return 0;
}
#endif

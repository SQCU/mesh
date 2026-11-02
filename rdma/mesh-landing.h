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

/* Common refinement of independently packetized operands. Every emitted cell
   lies in one run of every input. Its thresholds are constants of preparation;
   a numerical block loads those words directly, without walking run tables.
   Each input describes the same logical extent, irrespective of wire cuts. */
struct mesh_landing_map { const struct mesh_landing_run *runs; size_t count; uint64_t base; };
typedef void (*mesh_landing_emit)(void *context,uint64_t first,uint64_t elements,const uint64_t *through);
/* Preparation restriction of the same periodic arrival relation to one value
   interval. No new packet numbering or runtime mapping is introduced. */
static inline size_t mesh_landing_slice(const struct mesh_landing_run *runs,size_t count,
    uint64_t first,uint64_t elements,struct mesh_landing_run *out) {
  size_t made=0;const uint64_t end=first+elements;
  for(size_t k=0;k<count;k++) {
    const struct mesh_landing_run r=runs[k];
    const uint64_t begin=r.first>first?r.first:first,stop=r.first+r.elements<end?r.first+r.elements:end;
    if(begin>=stop)continue;
    const uint64_t at=r.phase+(begin-r.first)*r.step;
    out[made++]=(struct mesh_landing_run){.first=begin-first,.elements=stop-begin,
      .packet=r.packet+(at>>r.shift),.phase=at&((UINT64_C(1)<<r.shift)-1),.step=r.step,.shift=r.shift};
  }
  return made;
}
static inline uint64_t mesh_landing_cells(const struct mesh_landing_map *maps,size_t inputs,uint64_t width,
    size_t *cursor,uint64_t *through,mesh_landing_emit emit,void *context) {
  if(!inputs||!maps[0].count)return 0;
  for(size_t k=0;k<inputs;k++)cursor[k]=0;
  const struct mesh_landing_run *last=maps[0].runs+maps[0].count-1;
  const uint64_t end=last->first+last->elements;
  uint64_t count=0;
  for(uint64_t first=0;first<end;) {
    uint64_t stop=end-first<width?end:first+width;
    for(size_t k=0;k<inputs;k++) {
      const struct mesh_landing_run *r=maps[k].runs+cursor[k];
      while(first>=r->first+r->elements)r=maps[k].runs+(++cursor[k]);
      const uint64_t boundary=r->first+r->elements;
      if(boundary<stop)stop=boundary;
      /* A packet is also a possible activity/publication cut. Keep complete
         values together, but never join values from distinct packets merely
         because their backing addresses form one regular run. */
      if(r->step) {
        const uint64_t at=r->phase+(first-r->first)*r->step;
        const uint64_t next=(((at>>r->shift)+1)<<r->shift)-r->phase;
        const uint64_t packet=r->first+(next+r->step-1)/r->step;
        if(packet>first&&packet<stop)stop=packet;
      }
    }
    for(size_t k=0;k<inputs;k++) {
      const struct mesh_landing_run *r=maps[k].runs+cursor[k];
      through[k]=maps[k].base+r->packet+((r->phase+(stop-r->first-1)*r->step)>>r->shift);
    }
    if(emit)emit(context,first,stop-first,through);
    count++;first=stop;
  }
  return count;
}

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

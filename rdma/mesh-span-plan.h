/* AOT common packet cuts for exact byte DMA. A rank contributes its arena
   offset modulo the registration extent, never a process virtual address.
   The union covers every participant and every repeat so all physical paths
   share one unit sequence. No payload, transport word, or network is read. */
#ifndef MESH_SPAN_PLAN_H
#define MESH_SPAN_PLAN_H
#include "mesh.h"
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

struct mesh_span_plan { uint64_t *cuts; size_t count; };

static inline void mesh_span_plan_free(struct mesh_span_plan *plan) {
  free(plan->cuts);
  *plan = (struct mesh_span_plan){0};
}

static inline uint64_t mesh_span_gcd(uint64_t a, uint64_t b) {
  while (b) { const uint64_t r = a % b; a = b; b = r; }
  return a;
}

/* Coprime a/modulus, modulus <= UINT32_MAX. Unsigned modular coefficients
   keep products representable even when a signed coefficient would not be. */
static inline uint64_t mesh_span_inverse(uint64_t a, uint64_t modulus) {
  if (modulus == 1) return 0;
  uint64_t r = modulus, next = a % modulus, t = 0, u = 1;
  while (next) {
    const uint64_t q = r / next, remainder = r % next;
    const uint64_t v = (t + modulus - (q * u) % modulus) % modulus;
    r = next; next = remainder; t = u; u = v;
  }
  return t;
}

static inline int mesh_span_cut_compare(const void *a, const void *b) {
  const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

struct mesh_span_residues { uint64_t *at; size_t count, room; };
static inline void mesh_span_residues_sort(struct mesh_span_residues *r) {
  if (!r->count) return;
  qsort(r->at, r->count, sizeof *r->at, mesh_span_cut_compare);
  size_t n = 1;
  for (size_t k = 1; k < r->count; k++)
    if (r->at[k] != r->at[n - 1]) r->at[n++] = r->at[k];
  r->count = n;
}
static inline int mesh_span_residue_add(struct mesh_span_residues *r, uint64_t at) {
  if (r->count == r->room) {
    if (r->room > SIZE_MAX / 2 / sizeof *r->at) return EOVERFLOW;
    const size_t room = r->room ? r->room * 2 : 16;
    uint64_t *grown = realloc(r->at, room * sizeof *grown);
    if (!grown) return ENOMEM;
    r->at = grown; r->room = room;
  }
  r->at[r->count++] = at;
  return 0;
}

/* A storage page list, coalesced only where its backing addresses are adjacent.
   Logical positions are bytes in the local operand; offsets name the bridge's
   registered region. These runs exist during preparation, never during issue. */
struct mesh_page_run { uint64_t first, end, offset; };
struct mesh_page_runs { struct mesh_page_run *at; size_t count, room; };
static inline int mesh_page_run_add(struct mesh_page_runs *runs,uint64_t first,uint64_t bytes,uint64_t offset) {
  if(!bytes)return 0;
  struct mesh_page_run *last=runs->count?runs->at+runs->count-1:NULL;
  if(last&&last->end==first&&last->offset+last->end-last->first==offset){last->end+=bytes;return 0;}
  if(runs->count==runs->room) {
    if(runs->room>SIZE_MAX/2/sizeof *runs->at)return EOVERFLOW;
    size_t room=runs->room?2*runs->room:1;
    struct mesh_page_run *grown=realloc(runs->at,room*sizeof *grown);
    if(!grown)return ENOMEM;
    runs->at=grown;runs->room=room;
  }
  runs->at[runs->count++]=(struct mesh_page_run){first,first+bytes,offset};return 0;
}
static inline int mesh_page_runs_read(struct hdr *m,uint32_t row,uint64_t first,uint64_t bytes,struct mesh_page_runs *out) {
  const uint64_t quantum=(uint64_t)m->block*m->pgsz;
  for(uint64_t at=0;at<bytes;) {
    const uint64_t offset=first+at;
    uint64_t length=quantum-offset%quantum;
    if(length>bytes-at)length=bytes-at;
    int error=mesh_page_run_add(out,at,length,mesh_page_offset(m,row,offset));
    if(error)return error;
    at+=length;
  }
  return 0;
}
/* the run holding logical byte `logical`, NULL where none does (past the last run's end, or before the first's start):
   never the slot past the table's end */
static inline const struct mesh_page_run *mesh_page_run_at(const struct mesh_page_runs *runs,uint64_t logical) {
  size_t first=0,end=runs->count;
  while(first<end){size_t middle=first+(end-first)/2;if(runs->at[middle].end<=logical)first=middle+1;else end=middle;}
  return first<runs->count&&runs->at[first].first<=logical?runs->at+first:NULL;
}
/* logical byte `logical`'s region offset; UINT64_MAX where no run holds it, which no DMA span takes (mesh_dma_span_valid) */
static inline uint64_t mesh_page_run_offset(const struct mesh_page_runs *runs,uint64_t logical) {
  const struct mesh_page_run *run=mesh_page_run_at(runs,logical);
  return run?run->offset+logical-run->first:UINT64_MAX;
}
/* Add the physical discontinuities and registration boundaries of one selected
   interval. logical follows message order, including repeats; previous is the
   prior interval's physical end. Adjacent selections introduce no extra cut. */
static inline int mesh_page_run_cuts(const struct mesh_page_runs *runs,uint64_t first,uint64_t bytes,
    uint64_t logical,uint64_t registration_origin,uint64_t extent,uint64_t *previous,struct mesh_span_residues *cuts) {
  while(bytes) {
    const struct mesh_page_run *run=mesh_page_run_at(runs,first);
    if(!run)return ERANGE;   /* the interval runs past the table's pages */
    uint64_t offset=run->offset+first-run->first,phase=(offset-registration_origin)%extent;
    if(logical&&(offset!=*previous||!phase)) {
      int error=mesh_span_residue_add(cuts,logical);if(error)return error;
    }
    uint64_t length=run->end-first;
    if(length>bytes)length=bytes;
    if(length>extent-phase)length=extent-phase;
    *previous=offset+length;first+=length;logical+=length;bytes-=length;
  }
  return 0;
}

static inline size_t mesh_span_lower_bound(const uint64_t *cuts,size_t count,uint64_t value) {
  size_t first=0;
  while(first<count){size_t middle=first+(count-first)/2;if(cuts[middle]<value)first=middle+1;else count=middle;}
  return first;
}

/* Common byte packets in transaction coordinates. Page size bounds each DMA
   interval; backing breaks and fragment boundaries cut that same sequence.
   Numerical row widths do not participate in this calculation. */
struct mesh_packet_cursor { uint64_t at, end; size_t cut; };
static inline struct mesh_packet_cursor mesh_packet_begin(const struct mesh_span_residues *cuts,uint64_t first,uint64_t bytes) {
  return (struct mesh_packet_cursor){first,first+bytes,mesh_span_lower_bound(cuts->at,cuts->count,first+!!bytes)};
}
static inline uint64_t mesh_packet_next(const struct mesh_span_residues *cuts,uint64_t page,struct mesh_packet_cursor *p) {
  if(p->at==p->end)return 0;
  uint64_t bytes=page-p->at%page;
  if(bytes>p->end-p->at)bytes=p->end-p->at;
  if(p->cut<cuts->count&&cuts->at[p->cut]-p->at<=bytes)bytes=cuts->at[p->cut++]-p->at;
  p->at+=bytes;
  return bytes;
}
/* Keep only distinct registration-boundary residues that can occur inside
   this payload. Enumerate the smaller of the repeat orbit and the candidate
   residue lattice; a billion repeats of a tiny payload need not be visited.
   For a candidate residue r, modular inversion determines its first repeat. */
static inline int mesh_span_residues(const uint64_t *phases, const uint64_t *strides, uint32_t participants,
                                     uint64_t bytes, uint64_t repeats, uint32_t packet,
                                     struct mesh_span_residues *result) {
  const uint64_t extent = MESH_DMA_EXTENT;
  const uint64_t limit = bytes < extent ? bytes - 1 : extent - 1;
  for (uint32_t p = 0; p < participants; p++) {
    const uint64_t phase = phases[p], step = strides[p] % extent;
    const uint64_t grain = mesh_span_gcd(step, extent), period = extent / grain;
    const uint64_t turns = repeats < period ? repeats : period;
    const uint64_t inverse = mesh_span_inverse(step / grain, period);
    /* These boundaries are already all on the common packet grid. */
    if (!(extent % packet) && !(step % packet) && !(phase % packet)) continue;
    const uint64_t remainder = (grain - phase % grain) % grain;
    const uint64_t first = remainder ? remainder : grain;
    const uint64_t zero = !remainder && bytes > extent;
    const uint64_t candidates = zero + (first <= limit ? (limit - first) / grain + 1 : 0);
    if (turns <= candidates) {
      uint64_t at = phase;
      for (uint64_t k = 0; k < turns; k++) {
        const uint64_t residue = at ? extent - at : 0;
        if ((residue ? residue : extent) < bytes) {
          const int error = mesh_span_residue_add(result, residue);
          if (error) return error;
        }
        at = (at + step) % extent;
      }
    } else {
      for (uint64_t k = 0; k < candidates; k++) {
        const uint64_t residue = zero && !k ? 0 : first + (k - zero) * grain;
        const uint64_t value = (phase + residue) / grain % period;
        const uint64_t turn = ((period - value) % period * inverse) % period;
        if (turn < turns) {
          const int error = mesh_span_residue_add(result, residue);
          if (error) return error;
        }
      }
    }
  }
  mesh_span_residues_sort(result);
  return 0;
}

#endif

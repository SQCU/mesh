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

/* The elements x, lo < x < hi, of a buffer whose byte 0 lies `phase` bytes into a registration extent, at which an
   extent's boundary falls on an element of z bytes (x * z + phase a multiple of MESH_DMA_EXTENT), ascending; at most
   `room` written, their count. */
static inline size_t mesh_span_extent_cuts(uint64_t phase, uint64_t z, uint64_t lo, uint64_t hi, uint64_t *out, size_t room) {
  size_t n = 0;
  uint64_t b = MESH_DMA_EXTENT - phase;
  if (b <= lo * z) b += (lo * z - b) / MESH_DMA_EXTENT * MESH_DMA_EXTENT + MESH_DMA_EXTENT;
  for (; n < room && b / z < hi; b += MESH_DMA_EXTENT)
    if (!(b % z) && b / z > lo) out[n++] = b / z;
  return n;
}
static inline int mesh_span_cut_compare(const void *a, const void *b) {
  const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

struct mesh_span_residues { uint64_t *at; size_t count, room; };
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
  if (result->count) {
    qsort(result->at, result->count, sizeof *result->at, mesh_span_cut_compare);
    size_t n = 1;
    for (size_t k = 1; k < result->count; k++)
      if (result->at[k] != result->at[n - 1]) result->at[n++] = result->at[k];
    result->count = n;
  }
  return 0;
}

#endif

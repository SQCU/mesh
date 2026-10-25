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

/* Positive errno on failure, leaving out unchanged. out starts empty. packet
   is a positive maximum packet length (the native caller supplies its page
   default); repeats is positive. bytes==0 produces the single endpoint 0.
   The plan is valid for the declared phases/repeat stride only. A differently
   placed lease requires a configuration-time rebind, not a local hot-path cut.
   Actual allocation bounds/generations are checked by the binding layer. */
static inline int mesh_span_plan_create(const uint64_t *phases, const uint64_t *strides, uint32_t participants,
                                        uint64_t bytes, uint64_t repeats, uint32_t packet,
                                        struct mesh_span_plan *out) {
  if (!out || out->cuts || out->count || !phases || !strides || !participants || !repeats || !packet) return EINVAL;
  for (uint32_t p = 0; p < participants; p++) {
    if (phases[p] >= MESH_DMA_EXTENT) return EINVAL;
    if (strides[p] && repeats - 1 > (UINT64_MAX - bytes) / strides[p]) return EOVERFLOW;
  }
  const uint64_t grid = bytes ? (bytes - 1) / packet : 0;
  const size_t endpoints = bytes ? 2 : 1;
  const size_t maximum = SIZE_MAX / sizeof(uint64_t);
  if (grid > maximum - endpoints) return EOVERFLOW;
  size_t count = (size_t)grid + endpoints;
  struct mesh_span_residues residues = {0};
  int error = bytes ? mesh_span_residues(phases, strides, participants, bytes, repeats, packet, &residues) : 0;
  if (error) { free(residues.at); return error; }
  const uint64_t extent = MESH_DMA_EXTENT, common = mesh_span_gcd(extent, packet);
  const uint64_t modulus = packet / common, inverse = mesh_span_inverse(extent / common, modulus);
  /* Count the exact union before allocation. Registration residues are unique;
     subtract their intersections with the packet grid using modular arithmetic. */
  for (size_t k = 0; k < residues.count; k++) {
    const uint64_t first = residues.at[k] ? residues.at[k] : extent;
    const uint64_t terms = (bytes - 1 - first) / extent + 1;
    uint64_t overlap = 0;
    if (!(first % common)) {
      const uint64_t value = (first / common) % modulus;
      const uint64_t begin = ((modulus - value) % modulus * inverse) % modulus;
      if (begin < terms) overlap = (terms - 1 - begin) / modulus + 1;
    }
    const uint64_t additional = terms - overlap;
    if (additional > maximum - count) { free(residues.at); return EOVERFLOW; }
    count += (size_t)additional;
  }
  uint64_t *cuts = malloc(count * sizeof *cuts);
  if (!cuts) { free(residues.at); return ENOMEM; }
  cuts[0] = 0;
  if (bytes) {
    uint64_t next_grid = packet < bytes ? packet : bytes, base = 0, next_mr = UINT64_MAX;
    size_t index = 0, written = 1;
    if (residues.count) {
      if (!residues.at[0]) index++;
      if (index == residues.count) { index = 0; base = extent; }
      next_mr = base + residues.at[index];
    }
    for (;;) {
      uint64_t at = next_grid < next_mr ? next_grid : next_mr;
      if (at > bytes) at = bytes;
      cuts[written++] = at;
      if (at == bytes) break;
      if (next_grid == at) next_grid = bytes - at > packet ? at + packet : bytes;
      if (next_mr == at) {
        if (++index == residues.count) {
          index = 0;
          if (UINT64_MAX - base < extent) { next_mr = UINT64_MAX; continue; }
          base += extent;
        }
        next_mr = UINT64_MAX - base < residues.at[index] ? UINT64_MAX : base + residues.at[index];
      }
    }
    if (written != count) { free(cuts); free(residues.at); return EPROTO; }
  }
  free(residues.at);
  *out = (struct mesh_span_plan){cuts, count};
  return 0;
}
#endif

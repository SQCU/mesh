#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* The GPU's side of the plan's words (design/disjoint-pages-change-request.md, Appendix C, C3.1, C3.2, C3.6 and C3.7;
   design/disjoint-pages-revision-coordination.md, "The GPU's side of the plan's words", D1, D2 and D7).
   2026-11-04, sub-assignment 8: reversing sub-assignment 9's 2026-11-03 forms: the transport's two layout areas (the
   supplied words, hdr word_off, and the landed list, hdr landed_off) and the program's cancellation (`cancel`, its
   cancel words stored UINT64_MAX by mesh_cancel); an input's landed list retained with its execution records at
   offset + t x stride and its range's bit run; the publication's records (struct prepared_publication) and statuses
   (struct mesh_output_status, one a range and invocation) in sections laid out at the call (mesh_section_create).  Why:
   the areas and the cancel word are gone (mesh.h 109); the plan's state section is its own pages, placed by the plan
   (C3.2, C3.4); a receive's state is a 2-bit phase a copy and its range's count a copy (C3.7, D1); cancellations arrive
   on the counts, and a stop is mesh_metal_waits_stop (C3.1); the recorder binds a supplied word itself and raises it by
   one (D9); its restoring counter replaces the statuses (the contract, "Publications").  What must stay: the plan's
   words wrapped once, by address, with no copy; one input a received range and one supplied word a source range, as
   byte offsets in that one buffer; nothing here creates a section. */
/* the plan's state section wrapped once for the GPU: `state` an MTLBuffer over its pages (no copy), `words` its struct
   mesh_state, `offset` its region offset (the context's `state`, mesh-dataflow.h struct mesh_ctx), `bytes` its
   length; EINVAL where the context has no standing plan.  2026-11-04, owner L (design/compilation-coordination.md, "The
   region"): every function here takes the plan from the context's `state`, never from the region's header, where it
   was hdr `state`, the one plan's.  What must stay: the plan's words wrapped once, by address, with no copy */
struct mesh_metal_transport { void *state; struct mesh_state *words; uint64_t offset,bytes; };
int mesh_metal_transport_create(struct mesh_ctx *,void *device,struct mesh_metal_transport *);
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
/* a received range's words, byte offsets in the transport's buffer: copy c's phase group at phases + c x group, its
   count word at count + 8 c; pages P; copies u */
struct mesh_metal_input { uint64_t phases,group,count; uint32_t pages,copies; };
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,struct mesh_metal_input *);
/* a source range's supplied word, its byte offset in the transport's buffer */
struct mesh_metal_publication { uint64_t supplied; };
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,
                                   struct mesh_metal_publication *);
/* one repetition's wait (design/disjoint-pages-rulings-coordination.md, R11 as review 10 amended it): pass `pass` of a
   repetition's `passes` reads a received range whose count word of copy 0 is at byte offset `count` of the transport's
   buffer (copy c's at count + 8 c), `pages` P a copy, `copies` u.  Repetition t's wait is {count word count + 8 (t mod
   copies), target (t / copies + 1) x pages, value t x passes + pass + 1} */
struct mesh_metal_wait { uint64_t count; uint32_t pages,copies,pass,padding; };
/* the client's signalling of the queue's event (C3.1): the waits given are one repetition's, `pass` nondecreasing and
   each below `passes`; the waiter copies them once (one repetition's, never the run's) and walks repetitions t = 0 ..
   `repetitions` - 1, computing each wait's count word, target and value as it goes.  A wait is met once its count word
   reads at least its target; the event (an id<MTLSharedEvent>) is set to each value once every wait of that value or
   less is met, in order, and a value no wait has (a pass that reads nothing) as soon as the values before it are.
   The thread waits on each count word with os_sync_wait_on_address_with_timeout (OS_SYNC_WAIT_ON_ADDRESS_SHARED),
   woken by the bridge's arrivals; a value not met within MESH_PLATFORM_NS of the last value set, or a stop, sets the
   event to the final value, repetitions x passes, at once and ends.  EINVAL where a wait's count word for any copy
   c < copies lies outside the transport's buffer or is misaligned, `copies` or `pages` is 0, `pass` >= `passes`, the
   passes are out of order, or `passes` is 0 with waits given.
   2026-11-04, sub-assignment 2 (review 10's finding 1, the coordinator's ruling): reversing sub-assignment 8's form,
   every repetition's waits {count, target, value} given at once for the run and copied whole, and `final` an
   argument.  Why: a run's waits table was steps x passes x reads, though each wait is affine in t: the trip-count
   unrolling C1 and C3.3 rejected (design/disjoint-pages-change-request.md, D.3.7: nothing bound grows with the steps).
   What must stay: the event set to each value once every wait of that value or less is met, in order; the bounded
   wait and the stop setting the final value at once */
struct mesh_metal_waiter;
int mesh_metal_waits_start(struct mesh_metal_transport *,void *event,const struct mesh_metal_wait *,size_t waits,
                           uint32_t passes,uint64_t repetitions,struct mesh_metal_waiter **);
void mesh_metal_waits_stop(struct mesh_metal_waiter *);  /* async-signal-safe: one store and one wake */
int mesh_metal_waits_end(struct mesh_metal_waiter *);    /* joined and freed: 0, ETIMEDOUT (a deadline), ECANCELED (a stop) */
/* 2026-11-04, sub-assignment 8, where the contract's mesh_metal_waits_end names three causes: also EINVAL, where the
   system's wait refuses a count word (os_sync_wait_on_address.h), so the thread ends at once rather than spin to the
   deadline (mesh-metal.m waits_run) */
/* the GPU's status after the run (struct mesh_status): 0 where no pass failed; where one found a field behind, `cause`
   where it is not 0, else ENOLINK (the bridge cancelled), whatever `ahead` holds; EPROTO where one found a field ahead
   and none behind; *first the first failed execution.
   2026-11-04, sub-assignment 8, the coordinator's ruling on owner 6's finding (mesh-flow.c 04b5e561): reversing the
   contract's order, EPROTO first where any pass found a field ahead.  Why: a cancelled receive's field is never
   written, so a reader a round later finds it two behind and counts it `ahead` (three rounds later it reads as
   landed): after a cancellation `ahead` is the cancellation's, not a write ahead of the allocator's order.  What must
   stay: EPROTO where no pass found a field behind and one found it ahead. */
int mesh_metal_status(struct mesh_metal_transport *,int cause,uint64_t *first);
_Static_assert(sizeof(struct mesh_metal_transport)==32,"mesh_metal_transport");
_Static_assert(sizeof(struct mesh_metal_input)==32 && offsetof(struct mesh_metal_input,group)==8 &&
  offsetof(struct mesh_metal_input,count)==16 && offsetof(struct mesh_metal_input,pages)==24 &&
  offsetof(struct mesh_metal_input,copies)==28,"mesh_metal_input: metal_recording.h struct MetalRemote");
_Static_assert(sizeof(struct mesh_metal_publication)==8,"mesh_metal_publication");
_Static_assert(sizeof(struct mesh_metal_wait)==24 && offsetof(struct mesh_metal_wait,pages)==8 &&
  offsetof(struct mesh_metal_wait,copies)==12 && offsetof(struct mesh_metal_wait,pass)==16,
  "mesh_metal_wait: metal_recording.h struct MetalWait");
#endif

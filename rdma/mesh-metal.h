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
   mesh_state, `offset` its region offset (hdr state), `bytes` its length; EINVAL where no plan is written */
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
/* the client's signalling of the queue's event (C3.1): a wait of `value` is met once the count word at byte offset
   `count` of the transport's buffer reads at least `target`; the event (an id<MTLSharedEvent>) is set to each value once
   every wait of that value or less is met, in order.  The thread waits on each count word with
   os_sync_wait_on_address_with_timeout (OS_SYNC_WAIT_ON_ADDRESS_SHARED), woken by the bridge's arrivals; a value not
   met within MESH_PLATFORM_NS of the last value set, or a stop, sets the event to `final` at once and ends */
struct mesh_metal_wait { uint64_t count,target,value; };
struct mesh_metal_waiter;
int mesh_metal_waits_start(struct mesh_metal_transport *,void *event,const struct mesh_metal_wait *,size_t waits,
                           uint64_t final,struct mesh_metal_waiter **);
void mesh_metal_waits_stop(struct mesh_metal_waiter *);  /* async-signal-safe: one store and one wake */
int mesh_metal_waits_end(struct mesh_metal_waiter *);    /* joined and freed: 0, ETIMEDOUT (a deadline), ECANCELED (a stop) */
/* 2026-11-04, sub-assignment 8, where the contract's mesh_metal_waits_end names three causes: also EINVAL, where the
   system's wait refuses a count word (os_sync_wait_on_address.h), so the thread ends at once rather than spin to the
   deadline (mesh-metal.m waits_run) */
/* the GPU's status after the run (struct mesh_status): 0 where no pass failed; EPROTO where one found a field ahead;
   else `cause` where it is not 0, else ENOLINK (the bridge cancelled); *first the first failed execution */
int mesh_metal_status(struct mesh_metal_transport *,int cause,uint64_t *first);
_Static_assert(sizeof(struct mesh_metal_transport)==32,"mesh_metal_transport");
_Static_assert(sizeof(struct mesh_metal_input)==32 && offsetof(struct mesh_metal_input,group)==8 &&
  offsetof(struct mesh_metal_input,count)==16 && offsetof(struct mesh_metal_input,pages)==24 &&
  offsetof(struct mesh_metal_input,copies)==28,"mesh_metal_input: metal_recording.h struct MetalRemote");
_Static_assert(sizeof(struct mesh_metal_publication)==8,"mesh_metal_publication");
_Static_assert(sizeof(struct mesh_metal_wait)==24 && offsetof(struct mesh_metal_wait,target)==8 &&
  offsetof(struct mesh_metal_wait,value)==16,"mesh_metal_wait: metal_recording.h struct MetalWait");
#endif

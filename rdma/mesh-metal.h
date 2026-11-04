#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* design/prepared-machine.md#M07 */
/* the plan's two areas for the GPU (mesh.h struct mesh_hop, "Supplied words" and "The landed list"), wrapped once:
   `publication` the supplied words' area (hdr word_off, word_capacity), `inputs` the landed list's area (hdr landed_off,
   landed_capacity); `cancel` the program's cancellation of its executions */
struct mesh_metal_transport { void *publication,*inputs; struct mesh_cancellation *cancel; };
/* design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", item 4: a received range's input,
   read by mask from the landed list: `landed` the landed list's buffer (the transport's `inputs`, retained), `offset`
   the byte offset in it of execution 0's landed record and `stride` hdr landed_stride, so execution t's record is at
   offset + t x stride; [first, first + count) the range's bits in a record (bit b is bit b % 64 of word 1 + b / 64; word
   0 the execution's cancel word).
   2026-11-03, sub-assignment 9: reversing {completion, offset, stride}, one input a page, its own landed word at a
   256-byte cell an execution (hdr hop_stride).  Why: every reading thread fenced one word a page for a landing the
   bridge already learns from its completion (the change request, "The turns").  What must stay: execution t's word at
   offset + t x stride of one buffer, as the recorder binds an input. */
struct mesh_metal_input { void *landed; uint64_t offset,stride; uint32_t first,count; };
/* design/prepared-machine.md#M10 */
struct mesh_metal_publication { void *records,*completion; uint32_t count,stride; };
/* design/prepared-machine.md#M28 */
/* a range's countdown of one invocation, MESH_WORD_STRIDE bytes, so the direct publication binds the statuses and the
   supplied words at one per-execution stride.
   2026-11-03, sub-assignment 9: reversing the 256-byte, 128-aligned cell (the size of struct mesh_send, the plan's
   hop_stride), which the plan's cells set.  Why: the cells are gone (the change request, proposal item 3) and a
   supplied word's executions are MESH_WORD_STRIDE apart (mesh.h mesh_supplied).  What must stay: sizeof this struct is
   the publication's `stride`. */
struct mesh_output_status { uint32_t remaining,padding; };
_Static_assert(sizeof(struct mesh_output_status)==MESH_WORD_STRIDE,"M28");
_Static_assert(sizeof(struct mesh_metal_publication)==24,"prepared publication binding");
_Static_assert(sizeof(struct mesh_metal_transport)==24,"prepared Metal transport");
_Static_assert(sizeof(struct mesh_metal_input)==32,"prepared numerical input");
/* design/algorithm-sources.md#resident-metal */
/* the supplied words' area and the landed list's area wrapped for the GPU, and the program's cancellation: one range,
   every execution's cancel word ({landed_off, executions, landed_stride}); EINVAL where no plan is written (hdr `hops`
   0) */
int mesh_metal_transport_create(struct mesh_ctx *,void *,struct mesh_metal_transport *);
/* design/algorithm-sources.md#resident-metal */
/* the input of a declared destination section (mesh-collective.h mesh_plan, its count the plan's executions): its
   range's bit run, from instance 0's rows' publications; EINVAL where a page has no receive or the bits are not one
   run */
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,struct mesh_metal_input *);
/* design/algorithm-sources.md#resident-metal */
/* the publication of `ranges` source sections: records relative to the supplied words' buffer, `stride`
   MESH_WORD_STRIDE; with `invocations`, the statuses (struct mesh_output_status), one a range and invocation */
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,const struct mesh_section *,uint32_t,uint32_t,struct mesh_metal_publication *);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
#endif

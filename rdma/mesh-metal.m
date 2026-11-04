#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
/* The plan's two areas for the GPU (mesh.h struct mesh_hop, "Supplied words" and "The landed list"), which the layout
   places once with the region, page-aligned, and mesh_plan zeroes as far as it uses them: `publication` wraps the
   supplied words (hdr word_off, word_capacity), for the publications' stores, and `inputs` the landed list (hdr
   landed_off, landed_capacity), for the receives' masks.  Each is wrapped once, its length rounded up to the region's
   page, the padding the layout leaves before the next area, so the transport reads no transfer list and counts no
   frames of a link.  The cancellation is one range of a wire-0 section, every execution's cancel word (word 0 of
   its landed record): mesh_cancel(m, cancel, 0) stores UINT64_MAX into each still 0, so every step not yet whole ends
   as cancelled; a supplied word is not cancelled, since 0 releases nothing.  The bridge cancels its own receives not
   landed by the same cancel words where it ends the plan (mesh.h), so it reads no link's cancellation (the --link
   form's per-link notice, removed at layout 107).  EINVAL where no plan was ever written (hdr `plan` 0: the generation,
   never 0 once written); a plan that gives this node no hop still has both areas wrapped (review 13, finding 12).
   2026-11-03, sub-assignment 9 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", items
   2 and 3): reversing one section of the plan's words at the context's send_off, its 256-byte cells holding both the
   supplied words and a landed word a page (`inputs` the same buffer retained), whose cancellation stored UINT64_MAX in
   every word still 0, supplied words among them.  Why: the cells are gone, and the landed list is a contiguous array
   in its own area of the region (the change request, "The turns").  What must stay: both areas wrapped once, by
   address, with no copy; the program's cancellation a wire-0 section that mesh_cancel(m, cancel, 0) reads. */
int mesh_metal_transport_create(struct mesh_ctx *context,void *device,struct mesh_metal_transport *transport){
  *transport=(struct mesh_metal_transport){0};
  struct hdr *m=context->M;
  if(!atomic_load_explicit(&m->plan,memory_order_acquire))return EINVAL;
  const uint64_t page=m->pgsz;
  /* design/prepared-machine.md#M17 */
  transport->publication=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)m+m->word_off
    length:(m->word_capacity+page-1)/page*page options:MTLResourceStorageModeShared deallocator:nil];
  transport->inputs=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)m+m->landed_off
    length:(m->landed_capacity+page-1)/page*page options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->publication || !transport->inputs){mesh_metal_transport_destroy(transport);return ENOMEM;}
  /* design/prepared-machine.md#M12 */
  struct mesh_section stop;
  int status=mesh_section_create(context,sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range),1,0,&stop);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  struct mesh_cancellation *address=mesh_section_address(context,stop,0);memset(address,0,stop.bytes);
  address->ranges[0]=(struct mesh_cancel_range){.offset=m->landed_off,.count=m->executions,.stride=m->landed_stride};
  transport->cancel=address;
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
/* The input of a declared destination section (mesh-collective.h mesh_plan: count the plan's executions, P = ceil(bytes
   / (block x pgsz)) pages an instance), read by mask from the landed list (mesh.h struct mesh_hop, "The landed list";
   design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", items 2 and 4).  mesh_plan names
   each of instance 0's rows' receive in its publication: `landed` its bit b, `device_input` the region offset of
   execution 0's landed record, `device_stride` hdr landed_stride.  A range's pages are consecutive receive hops, so
   rows first + k, k < P, hold bits first .. first + P - 1 of one record: the input is that run, `offset` execution 0's
   record in the landed list's buffer and `stride` the records' stride, and a reader learns its step's range has landed
   from the record's cancel word and the words holding the run, not from a word a page.  The run's words lie inside a
   record, and every execution's record inside the buffer.  EINVAL where a page has no receive of the plan (its
   publication naming other than execution 0's record, hdr landed_off, at hdr landed_stride: the contract fixes both;
   review 13, finding 10), or where the bits are not one run.
   2026-11-03, sub-assignment 9: reversing one input a received page, its own landed word (`device_input`'s cell at
   hop_stride), a range of more than one page refused (mesh.h mesh_row_chunks): "one page a range".  Why: a page's
   own word had every reading thread fence one word a page; the mask tells of the whole range in a few words (the change
   request, "The turns").  What must stay: one input a received range, `landed` retained once a call; EINVAL for a page
   no receive of the plan lands in. */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section operand,struct mesh_metal_input *input){
  struct hdr *m=context->M;
  const uint64_t quantum=(uint64_t)m->block*m->pgsz,pages=(operand.bytes+quantum-1)/quantum;
  if(!pages || pages>UINT32_MAX-operand.first)return EINVAL;
  const struct mesh_publication *head=mesh_publication_at(m,operand.first);
  const uint64_t record=head->device_input,stride=head->device_stride;
  const uint32_t first=head->landed;
  for(uint32_t k=1;k<pages;k++){
    const struct mesh_publication *delivery=mesh_publication_at(m,operand.first+k);
    if(delivery->device_input!=record || delivery->device_stride!=stride || delivery->landed!=first+k)return EINVAL;
  }
  id<MTLBuffer> memory=transport->inputs;
  if(record!=m->landed_off || stride!=m->landed_stride || !stride)return EINVAL;
  const uint64_t offset=record-m->landed_off;
  if((1+((uint64_t)first+pages+63)/64)*sizeof(uint64_t)>stride || offset+(uint64_t)m->executions*stride>memory.length)
    return EINVAL;
  *input=(struct mesh_metal_input){.landed=[memory retain],.offset=offset,.stride=stride,.first=first,.count=(uint32_t)pages};
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M10 */
/* The publication of `ranges` source sections (mesh.h struct mesh_hop, "Supplied words"): each range's first row
   names, a target each, the supplied words of execution 0 its sends wait on, and its records (mesh-dataflow.h
   mesh_publication_prepare) are their destinations relative to the supplied words' buffer (the transport's
   `publication`, hdr word_off) with the value MESH_SUPPLIED.  `stride` is MESH_WORD_STRIDE, a supplied word's
   executions apart, so execution t's store is at the record's destination + t x stride: one store of a range's word
   releases every send of the range's pages (the change request, proposal item 3).  With `invocations`, the statuses
   (struct mesh_output_status), one a range and invocation, range j's of invocation t at j x (invocations + 1) + t,
   each seeded with its range's bytes: the direct publication binds them and the supplied words at the one
   per-execution stride.  Every record's word of every execution lies in the supplied words' buffer, else EINVAL; and
   EINVAL where `invocations` passes hdr executions, since a store at t >= executions would land in the next release's
   words and release another range's sends early (review 13, finding 11).
   2026-11-03, sub-assignment 9 (design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", item
   3): reversing `stride` hdr hop_stride (256-byte cells, a plan of another stride refused) and the records relative to
   the context's send_off.  Why: the cells and send_off are gone; a supplied word's executions are MESH_WORD_STRIDE
   apart (mesh.h mesh_supplied).  What must stay: one store still releases a range; the statuses' stride is the
   publication's (sizeof struct mesh_output_status == MESH_WORD_STRIDE, mesh-metal.h). */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  const struct mesh_section *sections,uint32_t ranges,uint32_t invocations,struct mesh_metal_publication *publication){
  struct hdr *m=context->M;
  uint32_t count=mesh_publication_prepare(m,sections[0].first,NULL);
  *publication=(struct mesh_metal_publication){.count=count*ranges,.stride=MESH_WORD_STRIDE};
  if(invocations>m->executions)return EINVAL;
  if(!count)return 0;
  struct mesh_section records;
  /* design/prepared-machine.md#M10 */
  int status=mesh_section_create(context,count*ranges*sizeof(struct prepared_publication),1,0,&records);
  if(status)return status;
  struct prepared_publication *prepared=mesh_section_address(context,records,0);
  for(uint32_t j=0;j<ranges;j++)mesh_publication_prepare(m,sections[j].first,prepared+j*count);
  /* design/prepared-machine.md#M17 */
  id<MTLBuffer> memory=transport->publication;
  const uint64_t span=(uint64_t)m->executions*MESH_WORD_STRIDE;
  for(uint32_t i=0;i<count*ranges;i++){
    prepared[i].destination-=(uintptr_t)m+m->word_off;
    if(prepared[i].destination>memory.length || span>memory.length-prepared[i].destination)return EINVAL;
  }
  id<MTLDevice> device=memory.device;
  publication->records=[device newBufferWithBytesNoCopy:prepared length:records.pages*m->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!publication->records)return ENOMEM;
  /* design/prepared-machine.md#M28 */
  if(invocations && sections[0].bytes<=UINT32_MAX){
    struct mesh_section words;
    /* design/prepared-machine.md#M28 */
    status=mesh_section_create(context,sizeof(struct mesh_output_status)*(uint64_t)(invocations+1)*ranges,1,0,&words);
    if(status){[(id)publication->records release];publication->records=NULL;return status;}
    struct mesh_output_status *address=mesh_section_address(context,words,0);
    for(uint32_t j=0;j<ranges;j++)for(uint32_t t=0;t<=invocations;t++)
      address[(uint64_t)j*(invocations+1)+t]=(struct mesh_output_status){.remaining=(uint32_t)sections[j].bytes};
    publication->completion=[device newBufferWithBytesNoCopy:address length:words.pages*m->pgsz
      options:MTLResourceStorageModeShared deallocator:nil];
    if(!publication->completion){[(id)publication->records release];publication->records=NULL;return ENOMEM;}
  }
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->publication release];[(id)transport->inputs release];
  *transport=(struct mesh_metal_transport){0};
}

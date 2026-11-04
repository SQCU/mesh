#import <Metal/Metal.h>
#include "mesh-metal.h"

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
/* The plan's words for the GPU (mesh.h struct mesh_hop, "Execution t's word"): mesh_plan allocates them, one section
   outside the window, zeroed, at the context's `send_off` (mesh-dataflow.h struct mesh_ctx), and records each hop
   page's words in its row's publication; the transport wraps that section once, for the publications' stores and for
   the receives' waits alike (`publication`, and `inputs` the same buffer retained), so it reads no transfer list and
   counts no frames of a link.  The cancellation covers the same words: mesh_cancel(m, cancel, 0) stores UINT64_MAX
   in every word of the plan still 0, so every wait on a landing not landed ends, and a release word not yet stored
   releases nothing until its producer stores it.  The bridge cancels its own landed words where it ends the plan
   (mesh.h), so it reads no link's cancellation (struct mesh_tx `cancel`, which nothing of this form writes).  EINVAL
   where no plan's words are there (mesh_plan not called) */
int mesh_metal_transport_create(struct mesh_ctx *context,void *device,struct mesh_metal_transport *transport){
  *transport=(struct mesh_metal_transport){0};
  struct hdr *m=context->M;
  if(!context->send_bytes)return EINVAL;
  /* design/prepared-machine.md#M17 */
  transport->publication=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)m+context->send_off
    length:context->send_bytes
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!transport->publication)return ENOMEM;
  transport->inputs=[(id)transport->publication retain];
  /* design/prepared-machine.md#M12 */
  struct mesh_section stop;
  int status=mesh_section_create(context,sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range),1,0,&stop);
  if(status){mesh_metal_transport_destroy(transport);return status;}
  struct mesh_cancellation *address=mesh_section_address(context,stop,0);memset(address,0,stop.bytes);
  address->ranges[0]=(struct mesh_cancel_range){.offset=context->send_off,.count=context->send_bytes/sizeof(_Atomic uint64_t)};
  transport->cancel=address;
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M07 */
/* One input per received page (mesh.h struct mesh_hop, "What a receive's landing stores"): the page's own landed word,
   which its row's publication names (`device_input`, its execution-0 word's region offset, and `device_stride`, the
   plan's hop_stride), not the last chunk's word of a range.  A range a page: a transfer's pages spread over every
   cable and queue pair and land in any order, so a range of several pages has no one word that tells of them all,
   and the GPU waits one word a range.  A range of more than one page (mesh.h mesh_row_chunks), or a page no receive
   of the plan lands in, is refused with EINVAL */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  struct mesh_section operand,struct mesh_metal_input *input){
  struct hdr *m=context->M;
  if(mesh_row_chunks(m,operand.first,operand.bytes)!=1)return EINVAL;
  struct mesh_publication *delivery=mesh_publication_at(m,operand.first);
  id<MTLBuffer> memory=transport->inputs;
  const uint64_t offset=(uintptr_t)m+delivery->device_input-(uintptr_t)memory.contents;
  if(!delivery->device_input || !delivery->device_stride || offset>=memory.length)return EINVAL;
  *input=(struct mesh_metal_input){.completion=[memory retain],.offset=offset,.stride=delivery->device_stride};
  return 0;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M10 */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  const struct mesh_section *sections,uint32_t ranges,uint32_t invocations,struct mesh_metal_publication *publication){
  uint32_t count=mesh_publication_prepare(context->M,sections[0].first,NULL);
  /* the plan's word stride (mesh.h struct mesh_hop, hdr `hop_stride`), which the completion statuses below share for
     the GPU's direct publication: a plan of another stride than theirs is refused */
  *publication=(struct mesh_metal_publication){.count=count*ranges,.stride=(uint32_t)context->M->hop_stride};
  if(!count)return 0;
  if(context->M->hop_stride!=sizeof(struct mesh_output_status))return EINVAL;
  struct mesh_section records;
  /* design/prepared-machine.md#M10 */
  int status=mesh_section_create(context,count*ranges*sizeof(struct prepared_publication),1,0,&records);
  if(status)return status;
  struct prepared_publication *prepared=mesh_section_address(context,records,0);
  for(uint32_t j=0;j<ranges;j++)mesh_publication_prepare(context->M,sections[j].first,prepared+j*count);
  /* design/prepared-machine.md#M17 */
  id<MTLBuffer> memory=transport->publication;
  for(uint32_t i=0;i<count*ranges;i++)prepared[i].destination-=((uintptr_t)context->M+context->send_off);
  id<MTLDevice> device=memory.device;
  publication->records=[device newBufferWithBytesNoCopy:prepared length:records.pages*context->M->pgsz
    options:MTLResourceStorageModeShared deallocator:nil];
  /* design/prepared-machine.md#M28 */
  if(invocations && sections[0].bytes<=UINT32_MAX){
    struct mesh_section words;
    /* design/prepared-machine.md#M28 */
    status=mesh_section_create(context,sizeof(struct mesh_output_status)*(uint64_t)(invocations+1)*ranges,1,0,&words);
    if(status){[(id)publication->records release];return status;}
    struct mesh_output_status *address=mesh_section_address(context,words,0);
    for(uint32_t j=0;j<ranges;j++)for(uint32_t t=0;t<=invocations;t++)
      address[(uint64_t)j*(invocations+1)+t]=(struct mesh_output_status){.remaining=(uint32_t)sections[j].bytes};
    publication->completion=[device newBufferWithBytesNoCopy:address length:words.pages*context->M->pgsz
      options:MTLResourceStorageModeShared deallocator:nil];
    if(!publication->completion){[(id)publication->records release];return ENOMEM;}
  }
  return publication->records?0:ENOMEM;
}

/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->publication release];[(id)transport->inputs release];
  *transport=(struct mesh_metal_transport){0};
}

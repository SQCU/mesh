#include "mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* mesh_transfers_start: mesh-dataflow.c, beside the map reader whose copy of the map it reads (its configured latch and
   unbounded spin on `prepared` gave way to a bounded wait for the ports the plan's hops reach); mesh_plan_erased beside
   it, the ports' `prepared` its one input too */

/* design/algorithm-sources.md#programtensor */
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 "What it is" and A4 (sub-assignment 7): a
   section's rows and pages are no longer taken at its call (main's form: mesh_rows_alloc, then mesh_arena_alloc in the
   window or out of it, first fit, with a row plane and owner words for their reclaim).  Its rows are the
   next of the program's claim (mesh-dataflow.h struct mesh_ctx `row_cursor`), cleaned as mesh_rows_alloc cleaned them;
   a `wire` 1 section's pages are the plan's allocator's, inside the claim's `window` (mesh-collective.h mesh_plan), so
   none is bound here; a `wire` 0 section's are the next of the claim's `local` run (`local_cursor`), bound and touched
   as before.  No row holds an owner word or `pages` but the claim's heads (mesh-dataflow.c mesh_claim), which its
   release reads.  What must stay: nothing allocated while the program runs; a section never outside its claim */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,int wire,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  const uint64_t quantum=(uint64_t)m->block*m->pgsz;
  if(bytes/quantum>=UINT32_MAX/m->block)return ENOMEM;
  const uint64_t span=(bytes+quantum-1)/quantum*m->block,stride=span/m->block;
  const uint64_t rows=(uint64_t)context->rows.first+context->rows.count-context->row_cursor,
                 local=(uint64_t)context->local.first+context->local.count-context->local_cursor;
  if(stride>rows/count || (!wire && span>local/count))return ENOMEM;
  const uint32_t first=context->row_cursor,named=(uint32_t)stride*count;
  for(uint32_t r=first;r<first+named;r++){
    struct mesh_page_entry *entry=&mesh_page(m)[r];
    atomic_store_explicit(&entry->mapping,MESH_ABSENT,memory_order_relaxed);
    atomic_store_explicit(&entry->address,0,memory_order_relaxed);
    atomic_store_explicit(&entry->device,0,memory_order_relaxed);
    struct mesh_buffer *buffer=&mesh_buffers(m)[r];
    atomic_store_explicit(&buffer->owner,0,memory_order_relaxed);
    atomic_store_explicit(&buffer->closed,0,memory_order_relaxed);
    buffer->pages=buffer->constant=0;
    struct mesh_publication *publication=mesh_publication_at(m,r);
    publication->sends=publication->landed=0;
    publication->device_input=publication->device_stride=0;
    atomic_store_explicit(&publication->argument,0,memory_order_relaxed);
  }
  context->row_cursor+=named;
  if(!wire){
    const uint32_t page=context->local_cursor;
    context->local_cursor+=(uint32_t)span*count;
    /* design/prepared-machine.md#M01 */
    /* Registration wires its pages eagerly, so until now every operand was resident by the time the
       bridge came up.  Unregistered pages are not, and their first fault would land in the measured
       region; touch the span here, at preparation, where it costs nothing that is being timed. */
    for(uint32_t p=0;p<(uint32_t)span*count;p++)*(volatile unsigned char *)mesh_at(m,page+p)=0;
    for(uint32_t s=0;s<count;s++)
      mesh_backing_bind(context,first+s*(uint32_t)stride,(uint32_t)span,page+s*(uint32_t)span,s);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,(uint32_t)stride};
  return 0;
}

/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){
  const uintptr_t address=atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)].address,memory_order_relaxed);
  return address?(char *)context->M+address:NULL;
}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffers(context->M)[section.first].constant=1;
}

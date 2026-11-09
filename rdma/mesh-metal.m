#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mesh-metal.h"
#include "mesh-collective.h"

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7 */
static int state_area(uint64_t from,uint64_t at,uint64_t count,uint64_t size,uint64_t align,uint64_t len,uint64_t *end){
  if(at<from || at%align || at>len || count>(len-at)/size)return EINVAL;
  if(at+count*size>*end)*end=at+count*size;
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7: one plan state's section, [at, *end) */
static int state_section(const struct mesh_ctx *context,uint64_t at,uint64_t *end){
  const struct hdr *m=context->M;
  const uint64_t len=context->len;
  if(!at || at%m->pgsz || at<m->data_off || at>len || len-at<sizeof(struct mesh_state))return EINVAL;
  const struct mesh_state *words=(const struct mesh_state *)((const char *)m+at);
  const uint64_t from=at+sizeof *words,copies=words->copies,ranges=words->ranges,word=sizeof(uint64_t);
  *end=from;
  if(!copies || state_area(from,words->status,1,sizeof(struct mesh_status),sizeof(uint64_t),len,end) ||
     (ranges && state_area(from,words->phases,0,1,MESH_LANE_BYTES,len,end)) ||
     (ranges && state_area(from,words->counts,ranges*copies,word,word,len,end)) ||
     (words->releases && state_area(from,words->supplied,words->releases,word,word,len,end)))
    return EINVAL;
  return 0;
}
/* design/aot-compilation-change-request.md, appendix item 6; design/compilation-composition-change-request.md item 6
   (a delivered range read by another function's capture): every plan state of the program in one view, every word
   offset relative to its lowest state (mesh-collective.c plan_layout: one item a function, all in region M) */
int mesh_metal_transport_create(struct mesh_ctx *context,const struct mesh_planned *planned,void *device,
  struct mesh_metal_transport *transport){
  *transport=(struct mesh_metal_transport){0};
  struct hdr *m=context->M;
  const uint64_t page=m->pgsz;
  uint64_t low=UINT64_MAX,high=0;
  for(uint32_t j=0;j<planned->calls;j++){
    const uint64_t at=planned->call[j].plan;
    uint64_t end;
    const int status=state_section(context,at,&end);
    if(status)return status;
    if(at<low)low=at;
    if(end>high)high=end;
  }
  if(!high)return EINVAL;
  const uint64_t bytes=(high-low+page-1)/page*page;
  if(bytes>context->len-low)return EINVAL;
  id<MTLBuffer> state=[(id<MTLDevice>)device newBufferWithBytesNoCopy:(char *)m+low length:bytes
    options:MTLResourceStorageModeShared deallocator:nil];
  if(!state)return ENOMEM;
  *transport=(struct mesh_metal_transport){.state=state,.offset=low,.bytes=bytes};
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.7 */
int mesh_metal_receive_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  uint32_t transfer,struct mesh_metal_input *input){
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_publication *entry=mesh_transfer_at(context->M,words,transfer);
  const uint64_t at=transport->offset;
  *input=(struct mesh_metal_input){.phases=entry->phases-at,.group=mesh_group_bytes(entry->pages),.count=entry->count-at,
    .pages=entry->pages,.copies=words->copies,
    .time=((const double *)(const void *)((const char *)context->M+words->times))[transfer]};
  return 0;
}

/* design/torch-capture-change-request.md item 3; mesh.h struct mesh_hop, mesh_hops; mesh-call.h mesh_section_address */
static int row_order(const void *a,const void *b){
  const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
  return (x>y)-(x<y);
}
/* copy 0's rows of `section`, {region offset, q} pairs ordered by offset */
static uint64_t *rows_sorted(struct mesh_ctx *context,struct mesh_section section){
  uint64_t *rows=malloc(2*sizeof *rows*(section.stride?section.stride:1));
  for(uint32_t q=0;rows && q<section.stride;q++)
    rows[2*q]=(uint64_t)((char *)mesh_section_address(context,section,q)-(char *)context->M),rows[2*q+1]=q;
  if(rows)qsort(rows,section.stride,2*sizeof *rows,row_order);
  return rows;
}
/* the copy-0 row whose block holds the hop's byte data_off + page x pgsz + offset, *at the byte's place in it */
static const uint64_t *row_of(const struct hdr *m,const uint64_t *rows,uint32_t count,const struct mesh_hop *hop,
  uint64_t *at){
  const uint64_t landed=m->data_off+(uint64_t)hop->page*m->pgsz+hop->offset;
  const uint64_t key=landed-(landed-m->data_off)%((uint64_t)m->block*m->pgsz);
  *at=landed-key;
  return bsearch(&key,rows,count,2*sizeof *rows,row_order);
}
int mesh_metal_receive_fields(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  const struct mesh_metal_input *input,struct mesh_section section,uint32_t *fields){
  struct hdr *m=context->M;
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_hop *hops=mesh_hops(m,words);
  const uint64_t group=transport->offset+input->phases,count=transport->offset+input->count;
  uint64_t *rows=rows_sorted(context,section),at;
  if(!rows)return ENOMEM;
  memset(fields,0,(size_t)input->pages*sizeof *fields);
  for(uint64_t e=0;e<words->entries;e++){
    const struct mesh_hop *hop=hops+e;
    if(hop->direction!=MESH_RECEIVE || hop->count!=count)continue;
    const uint64_t *found=row_of(m,rows,section.stride,hop,&at);
    if(!found){free(rows);return EINVAL;}
    fields[(hop->word-group)/sizeof(uint64_t)*32+hop->field/2]=(uint32_t)found[1];
  }
  free(rows);
  return 0;
}

/* design/problems-of-14b9a757-change-request.md, third stops 7; MPI 4.1 §3.7.3; rdma/mesh-collective.c mesh_plan's
   SEND hops (count the transfer's completion word of the hop's copy) */
int mesh_metal_sent_extents(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  uint32_t transfer,struct mesh_section section,const uint32_t *members,uint32_t count,struct mesh_graph_access *extents,
  size_t room,size_t *made){
  (void)transport;
  struct hdr *m=context->M;
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_hop *hops=mesh_hops(m,words);
  const uint64_t completion=mesh_transfer_at(m,words,transfer)->completion,block=(uint64_t)m->block*m->pgsz;
  uint64_t *rows=rows_sorted(context,section),at;
  uint32_t *member=count?malloc((section.stride?section.stride:1)*sizeof *member):NULL;
  if(!rows || (count && !member)){free(rows);free(member);return ENOMEM;}
  for(uint32_t q=0;member && q<section.stride;q++)member[q]=UINT32_MAX;
  for(uint32_t n=0;n<count;n++)if(members[n]<section.stride)member[members[n]]=n;
  size_t found=0;
  int status=0;
  for(uint64_t e=0;e<words->entries && !status;e++){
    const struct mesh_hop *hop=hops+e;
    if(hop->direction!=MESH_SEND || hop->count!=completion)continue;
    const uint64_t *row=row_of(m,rows,section.stride,hop,&at);
    const uint32_t key=row && member?member[row[1]]:0;
    if(!row || key==UINT32_MAX){status=EINVAL;break;}
    const uint64_t base=member?0:row[1]*block;
    if(found<room)extents[found]=(struct mesh_graph_access){.key=key,.start=base+at,.end=base+at+hop->bytes};
    found++;
  }
  free(rows);free(member);
  *made=found;
  return status?status:found>room?ERANGE:0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.2 and C3.7 */
int mesh_metal_publication_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  uint32_t transfer,struct mesh_metal_publication *publication){
  publication->supplied=mesh_transfer_at(context->M,mesh_state_at(context->M,plan),transfer)->targets[0].word-transport->offset;
  return 0;
}

/* design/problems-of-14b9a757-change-request.md, third stops 7; rdma/mesh-collective.c mesh_host_sent */
int mesh_metal_sent_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  uint32_t transfer,struct mesh_metal_sent *sent){
  (void)transport;
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_publication *entry=mesh_transfer_at(context->M,words,transfer);
  *sent=(struct mesh_metal_sent){.counter=(char *)context->M+entry->completion,.sends=entry->sends,.copies=words->copies};
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1; metal_recording.h state:buffer:status: */
int mesh_metal_status_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  uint64_t *status){
  *status=mesh_state_at(context->M,plan)->status-transport->offset;
  return 0;
}
int mesh_metal_status(struct mesh_ctx *context,uint64_t plan,int cause,uint64_t *first){
  if(first)*first=UINT64_MAX;
  if(!plan)return EINVAL;
  struct mesh_status *status=(struct mesh_status *)((char *)context->M+mesh_state_at(context->M,plan)->status);
  const uint32_t ahead=atomic_load_explicit(&status->ahead,memory_order_acquire);
  const uint32_t behind=atomic_load_explicit(&status->behind,memory_order_acquire);
  const uint64_t failed=atomic_load_explicit(&status->first,memory_order_acquire);
  if(failed && first)*first=~failed;
  if(behind)return cause?cause:ENOLINK;
  if(ahead)return EPROTO;
  return 0;
}

void mesh_metal_transport_destroy(struct mesh_metal_transport *transport){
  [(id)transport->state release];
  *transport=(struct mesh_metal_transport){0};
}

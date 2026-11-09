#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/sysctl.h>
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

/* design/disjoint-pages-change-request.md, Appendix C, C3.1 and C3.7: a received range's words */
static void receive_words(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,uint32_t transfer,
  struct mesh_metal_input *input){
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_publication *entry=mesh_transfer_at(context->M,words,transfer);
  const uint64_t at=transport->offset;
  *input=(struct mesh_metal_input){.phases=entry->phases-at,.group=mesh_group_bytes(entry->pages),.count=entry->count-at,
    .pages=entry->pages,.copies=words->copies,
    .time=((const double *)(const void *)((const char *)context->M+words->times))[transfer]};
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
/* design/problems-of-14b9a757-change-request.md, third stops 7; MPI 4.1 §3.7.3: the plan's hops of `direction` whose
   count word is `word`, each one struct mesh_graph_access of its bytes in the copy-0 rows of `section`, row q at q x
   block (mesh-metal.h struct mesh_metal_sending `extent`), `vertex` its transport field where `phases` is the range's
   phase fields (else 0); ERANGE with *made the number needed past `room`, EINVAL for a hop in no copy-0 row */
static int hop_extents(struct mesh_ctx *context,uint64_t plan,uint32_t direction,uint64_t word,uint64_t phases,
  struct mesh_section section,struct mesh_graph_access *extents,size_t room,size_t *made){
  struct hdr *m=context->M;
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_hop *hops=mesh_hops(m,words);
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  uint64_t *rows=rows_sorted(context,section);
  if(!rows)return ENOMEM;
  size_t found=0;
  int status=0;
  for(uint64_t e=0;e<words->entries && !status;e++){
    const struct mesh_hop *hop=hops+e;
    if(hop->direction!=direction || hop->count!=word)continue;
    const uint64_t landed=m->data_off+(uint64_t)hop->page*m->pgsz+hop->offset,key=landed-(landed-m->data_off)%block;
    const uint64_t *row=bsearch(&key,rows,section.stride,2*sizeof *rows,row_order);
    if(!row){status=EINVAL;break;}
    const uint64_t start=row[1]*block+landed-key;
    const uint32_t field=phases?(uint32_t)((hop->word-phases)/sizeof(uint64_t)*32+hop->field/2):0;
    if(found<room)extents[found]=(struct mesh_graph_access){.start=start,.end=start+hop->bytes,.vertex=field};
    found++;
  }
  free(rows);
  *made=found;
  return status?status:found>room?ERANGE:0;
}
/* design/torch-capture-change-request.md item 3; MPI 4.1 §4.2.2 MPI_PARRIVED: each RECEIVE hop's field names its row */
static int receive_fields(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  const struct mesh_metal_input *input,struct mesh_section section,uint32_t *fields){
  const uint64_t block=(uint64_t)((struct hdr *)context->M)->block*((struct hdr *)context->M)->pgsz;
  size_t made=input->pages,room=0;
  struct mesh_graph_access *extents=NULL;
  int status=ERANGE;
  while(status==ERANGE){
    free(extents);
    room=made;
    if(!(extents=malloc((room+1)*sizeof *extents)))return ENOMEM;
    status=hop_extents(context,plan,MESH_RECEIVE,transport->offset+input->count,transport->offset+input->phases,section,
                       extents,room,&made);
  }
  memset(fields,0,(size_t)input->pages*sizeof *fields);
  for(size_t k=0;!status && k<made;k++)
    if(extents[k].vertex<input->pages)fields[extents[k].vertex]=(uint32_t)(extents[k].start/block);
    else status=EINVAL;
  free(extents);
  return status;
}

/* J. Lee, Ph.D. dissertation, University of Michigan, 2015, §3.2.2: threadgroup g's footprint, its terms evaluated at
   g; its partitions [*lo, *hi] of the range, none where *lo > *hi */
static void mask_bounds(const uint32_t g[3],const struct mesh_graph_footprint *a,const struct mesh_graph_term *term,
  const struct mesh_metal_range *r,uint32_t *lo,uint32_t *hi){
  __int128 low=a->start-(__int128)r->at;
  for(uint32_t k=0;k<a->terms;k++){
    const struct mesh_graph_term *t=term+a->first+k;
    const uint64_t q=g[t->dim]/(t->divisor?t->divisor:1);
    low+=(__int128)t->coefficient*(__int128)(t->modulus?q%t->modulus:q);
  }
  __int128 high=low+a->extent;
  const __int128 end=(__int128)r->partitions*r->page;
  if(low<0)low=0;
  if(high>end)high=end;
  if(high<=low || !r->page){*lo=1;*hi=0;return;}
  *lo=(uint32_t)(low/r->page);*hi=(uint32_t)((high-1)/r->page);
}
struct mask_rows { const uint32_t *bound; uint32_t width; const uint32_t *probe; };
static int mask_same(const void *context,uint32_t k){
  const struct mask_rows *x=context;
  return !memcmp(x->bound+(size_t)k*x->width,x->probe,x->width*sizeof *x->probe);
}
/* design/torch-capture-change-request.md item 3; mesh-graph.h mesh_graph_keyed_find and mesh_graph_keyed_add over
   the rows' hash */
int mesh_metal_mask(const uint32_t groups[3],size_t footprints,const struct mesh_graph_footprint *footprint,
  const struct mesh_graph_term *term,const uint32_t *range,struct mesh_metal_readiness *readiness){
  const uint64_t grid=(uint64_t)groups[0]*groups[1]*groups[2];
  const uint32_t G=(uint32_t)grid,R=readiness->ranges,width=2*R;
  if(readiness->groups!=grid || !G)return EINVAL;
  for(size_t a=0;a<footprints;a++){
    if(range[a]!=UINT32_MAX && range[a]>=R)return EINVAL;
    for(uint32_t k=0;k<footprint[a].terms;k++)if(term[footprint[a].first+k].dim>2)return EINVAL;
  }
  uint32_t *bound=malloc(((size_t)G*width+1)*sizeof *bound),*head=malloc(((size_t)G+1)*sizeof *head),
           *next=malloc(((size_t)G+1)*sizeof *next),*first=malloc(((size_t)(G>R?G:R)+1)*sizeof *first);
  if(!bound || !head || !next || !first){free(bound);free(head);free(next);free(first);return ENOMEM;}
  for(uint32_t d=0;d<R;d++)first[d]=0;
  for(size_t a=0;a<footprints;a++)if(range[a]!=UINT32_MAX)first[range[a]]=1;
  for(uint32_t w=0;w<G;w++){
    const uint32_t g[3]={w%groups[0],w/groups[0]%groups[1],w/groups[0]/groups[1]};
    uint32_t *row=bound+(size_t)w*width;
    for(uint32_t d=0;d<R;d++){row[2*d]=1;row[2*d+1]=0;}
    for(uint32_t d=0;d<R;d++)
      if(!first[d] && readiness->range[d].partitions){row[2*d]=0;row[2*d+1]=readiness->range[d].partitions-1;}
    for(size_t a=0;a<footprints;a++){
      if(range[a]==UINT32_MAX)continue;
      uint32_t lo,hi,*at=row+2*range[a];
      mask_bounds(g,footprint+a,term,readiness->range+range[a],&lo,&hi);
      if(lo>hi)continue;
      if(at[0]>at[1]){at[0]=lo;at[1]=hi;}
      else{if(lo<at[0])at[0]=lo;if(hi>at[1])at[1]=hi;}
    }
    head[w]=UINT32_MAX;
  }
  uint32_t classes=0;
  for(uint32_t w=0;w<G;w++){
    const uint32_t *row=bound+(size_t)w*width;
    uint64_t hash=1469598103934665603u;
    for(uint32_t x=0;x<width;x++)hash=(hash^row[x])*1099511628211u;
    const struct mask_rows same={bound,width,row};
    const uint32_t key=(uint32_t)(hash%G),k=mesh_graph_keyed_find(head,next,key,mask_same,&same);
    if(k!=UINT32_MAX){readiness->class_of[w]=readiness->class_of[k];continue;}
    mesh_graph_keyed_add(head,next,key,w);
    readiness->class_of[w]=classes;
    first[classes++]=w;
  }
  for(uint32_t c=0;c<classes;c++){
    const uint32_t *row=bound+(size_t)first[c]*width;
    for(uint32_t d=0;d<R;d++){
      const struct mesh_metal_range *r=readiness->range+d;
      uint32_t a=UINT32_MAX,b=0;
      for(uint32_t f=0;row[2*d]<=row[2*d+1] && f<r->input.pages;f++)
        if(r->field[f]!=UINT32_MAX && r->field[f]>=row[2*d] && r->field[f]<=row[2*d+1]){if(f<a)a=f;b=f+1;}
      if(a>=b)a=b=0;
      readiness->span[2*((size_t)c*R+d)]=a;readiness->span[2*((size_t)c*R+d)+1]=b;
    }
  }
  readiness->classes=classes;
  free(bound);free(head);free(next);free(first);
  return 0;
}

/* design/disjoint-pages-change-request.md, Appendix C, C3.1; metal_recording.h state:buffer:status: */
int mesh_metal_status_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,uint64_t plan,
  uint64_t *status){
  *status=mesh_state_at(context->M,plan)->status-transport->offset;
  return 0;
}

/* design/compilation-composition-change-request.md, "Complete replacement workgroup", values and effects: the records
   of mesh-metal.h from the plan, one calling sequence for every producer of the compiler's transport operands */
int mesh_metal_range_declare(struct mesh_ctx *context,const uint32_t *calls,uint32_t ncalls,uint32_t call,
  uint32_t owner,uint32_t transfer,uint64_t bytes,uint64_t at,struct mesh_metal_range *range){
  if(call>=ncalls || owner>=ncalls)return EINVAL;
  const struct hdr *m=context->M;
  const uint64_t page=(uint64_t)m->block*m->pgsz;
  uint32_t every=0,producer=0,ordinal=0;
  for(uint32_t k=0;k<ncalls;k++){
    every+=calls[k]==calls[call];
    producer+=calls[k]==calls[owner];
    ordinal+=k<owner && calls[k]==calls[owner];
  }
  *range=(struct mesh_metal_range){.generation=owner==call?(struct mesh_graph_generation){1,1,0}:
    (struct mesh_graph_generation){every,producer,ordinal},.partitions=(uint32_t)((bytes+page-1)/page),.page=page,
    .at=at,.transfer=transfer,.owner=owner};
  return 0;
}
int mesh_metal_range_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  const struct mesh_planned *planned,struct mesh_section section,uint32_t *field,uint32_t room,
  struct mesh_metal_range *range){
  if(range->owner>=planned->calls || range->partitions!=section.stride)return EINVAL;
  const uint64_t plan=planned->call[range->owner].plan;
  range->field=field;
  receive_words(context,transport,plan,range->transfer,&range->input);
  return range->input.pages>room?ERANGE:receive_fields(context,transport,plan,&range->input,section,field);
}
int mesh_metal_release_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  const struct mesh_planned *planned,uint32_t call,struct mesh_metal_release *release){
  if(call>=planned->calls)return EINVAL;
  const uint64_t plan=planned->call[call].plan;
  release->supplied=mesh_transfer_at(context->M,mesh_state_at(context->M,plan),release->transfer)->targets[0].word-
                    transport->offset;
  return mesh_metal_status_prepare(context,transport,plan,&release->status);
}
int mesh_metal_sending_prepare(struct mesh_ctx *context,struct mesh_metal_transport *transport,
  const struct mesh_planned *planned,uint32_t call,uint32_t transfer,struct mesh_section section,
  struct mesh_graph_access *extent,size_t room,struct mesh_metal_sending *sending){
  (void)transport;
  if(call>=planned->calls)return EINVAL;
  const uint64_t plan=planned->call[call].plan;
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_publication *entry=mesh_transfer_at(context->M,words,transfer);
  *sending=(struct mesh_metal_sending){.sent={.counter=(char *)context->M+entry->completion,.sends=entry->sends,
                                              .copies=words->copies},.extent=extent,.transfer=transfer};
  return entry->sends?hop_extents(context,plan,MESH_SEND,entry->completion,0,section,extent,room,&sending->extents):0;
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

/* S. Williams, A. Waterman and D. Patterson, Roofline, CACM 52(4), 2009, §3: one resource's recorded operands */
static int machine_read(id given,struct mesh_graph_machine *x){
  if(![given isKindOfClass:[NSDictionary class]])return EINVAL;
  double *field[]={&x->peak,&x->bandwidth,&x->block_bandwidth,&x->l2,&x->l1,&x->launch,&x->l2_speedup,&x->l1_speedup,
                   &x->overlap};
  NSString *name[]={@"peak",@"bandwidth",@"block_bandwidth",@"l2",@"l1",@"launch",@"l2_speedup",@"l1_speedup",
                    @"overlap"};
  for(int k=0;k<9;k++){
    id number=((NSDictionary *)given)[name[k]];
    if(![number isKindOfClass:[NSNumber class]])return EINVAL;
    *field[k]=[number doubleValue];
  }
  id cores=((NSDictionary *)given)[@"cores"],fpus=((NSDictionary *)given)[@"fpus"];
  uint32_t physical=0;
  size_t size=sizeof physical;
  if(!cores && sysctlbyname("hw.physicalcpu",&physical,&size,NULL,0))return errno;
  if((cores && ![cores isKindOfClass:[NSNumber class]]) || ![fpus isKindOfClass:[NSNumber class]])return EINVAL;
  x->cores=cores?[cores unsignedIntValue]:physical;x->fpus=[fpus unsignedIntValue];
  return 0;
}
/* design/torch-capture-change-request.md, 2026-11-09 appendix: this node's model's GPU and CPU records */
int mesh_metal_machines(const char *path,struct mesh_graph_machine *gpu,struct mesh_graph_machine *cpu){
  char model[256];
  size_t size=sizeof model;
  if(sysctlbyname("hw.model",model,&size,NULL,0))return errno;
  int status=ENOENT;
  @autoreleasepool{
    NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    id records=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]:nil;
    id models=[records isKindOfClass:[NSDictionary class]]?((NSDictionary *)records)[@"models"]:nil;
    id entry=[models isKindOfClass:[NSDictionary class]]?((NSDictionary *)models)[[NSString stringWithUTF8String:model]]:nil;
    id machines=[entry isKindOfClass:[NSDictionary class]]?((NSDictionary *)entry)[@"machines"]:nil;
    if(data && !records)status=EINVAL;
    else if([machines isKindOfClass:[NSDictionary class]]){
      status=machine_read(((NSDictionary *)machines)[@"gpu"],gpu);
      if(!status)status=machine_read(((NSDictionary *)machines)[@"cpu"],cpu);
    }
  }
  return status;
}

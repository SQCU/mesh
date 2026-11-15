#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/sysctl.h>
#include <dlfcn.h>
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

/* design/compilation-composition-change-request.md items 3, 6.4 and 7; mesh-collective.h struct mesh_planned_transfer:
   planned call `call`'s transfer `transfer`'s rows, ENODATA where the plan holds none (loaded from an archive saved
   without them) */
static int planned_rows(const struct mesh_planned *planned,uint32_t call,uint32_t transfer,
  struct mesh_planned_transfer *rows){
  if(!planned->first)return ENODATA;
  const uint32_t f=planned->call[call].function;
  if(transfer>=planned->first[f+1]-planned->first[f])return EINVAL;
  *rows=planned->transfer[planned->first[f]+transfer];
  return 0;
}
/* design/torch-capture-change-request.md item 3; MPI 4.1 §4.2.2 MPI_PARRIVED: each RECEIVE row's phase field names the
   block its logical bytes start in, a field no row names 0 */
static int receive_fields(struct mesh_ctx *context,const struct mesh_planned *planned,struct mesh_planned_transfer rows,
  const struct mesh_metal_input *input,uint32_t *fields){
  const uint64_t block=(uint64_t)((struct hdr *)context->M)->block*((struct hdr *)context->M)->pgsz;
  memset(fields,0,(size_t)input->pages*sizeof *fields);
  for(size_t k=rows.received;k<rows.sending;k++){
    const struct mesh_graph_access *row=planned->extent+k;
    if(row->vertex>=input->pages)return EINVAL;
    fields[row->vertex]=(uint32_t)(row->start/block);
  }
  return 0;
}

/* J. Lee, Ph.D. dissertation, University of Michigan, 2015, §3.2.2: threadgroup g's footprint, its terms evaluated at
   g; its partitions [*lo, *hi] of the range, none where *lo > *hi */
/* the partitions [*lo, *hi] of range r that bytes [low, high) of its binding reach (r->at the range's first byte there):
   *lo > *hi none */
static void mask_partitions(__int128 low,__int128 high,const struct mesh_metal_range *r,uint32_t *lo,uint32_t *hi){
  low-=(__int128)r->at;high-=(__int128)r->at;
  const __int128 end=(__int128)r->partitions*r->page;
  if(low<0)low=0;
  if(high>end)high=end;
  if(high<=low || !r->page){*lo=1;*hi=0;return;}
  *lo=(uint32_t)(low/r->page);*hi=(uint32_t)((high-1)/r->page);
}
/* footprint a's bytes at threadgroup g: [start + terms(g), + extent) */
static __int128 mask_start(const uint32_t g[3],const struct mesh_graph_footprint *a,const struct mesh_graph_term *term){
  __int128 low=a->start;
  for(uint32_t k=0;k<a->terms;k++){
    const struct mesh_graph_term *t=term+a->first+k;
    const uint64_t q=g[t->dim]/(t->divisor?t->divisor:1);
    low+=(__int128)t->coefficient*(__int128)(t->modulus?q%t->modulus:q);
  }
  return low;
}
struct mask_rows { const uint32_t *bound; uint32_t width; const uint32_t *probe; };
// mesh callback (preparation; registered with mesh_graph_keyed_find): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
static int mask_same(const void *context,uint32_t k){ void (*mesh_callback_disabled)(void) = &mesh_callbacks_are_outside_spec;
  const struct mask_rows *x=context;
  return !memcmp(x->bound+(size_t)k*x->width,x->probe,x->width*sizeof *x->probe);
}
/* design/torch-capture-change-request.md item 3; mesh-graph.h mesh_graph_keyed_find and mesh_graph_keyed_add over
   the rows' hash. A threadgroup's row: for each range d its fields' read bits (bit 2 (f mod 16) of word w f / 16,
   words[d] of them from offset[d]), the fields whose partition an access of d reaches, every field of a range no
   access names */
int mesh_metal_mask_partitions(const uint32_t groups[3],size_t accesses,const size_t *first,const uint32_t *partition,
  const uint32_t *range,struct mesh_metal_readiness *readiness){
  readiness->member=NULL;readiness->members=0;
  const uint64_t grid=(uint64_t)groups[0]*groups[1]*groups[2];
  const uint32_t G=(uint32_t)grid,R=readiness->ranges;
  if(readiness->groups!=grid || !G)return EINVAL;
  for(size_t a=0;a<accesses;a++)if(range[a]!=UINT32_MAX && range[a]>=R)return EINVAL;
  uint32_t width=0,most=1;
  uint32_t *offset=malloc(((size_t)R+1)*sizeof *offset),*named=calloc((size_t)R+1,sizeof *named);
  if(!offset || !named){free(offset);free(named);return ENOMEM;}
  for(uint32_t d=0;d<R;d++){
    offset[d]=width;width+=(readiness->range[d].input.pages+15)/16;
    if(readiness->range[d].partitions>most)most=readiness->range[d].partitions;
  }
  for(size_t a=0;a<accesses;a++)if(range[a]!=UINT32_MAX)named[range[a]]=1;
  uint32_t *bound=calloc((size_t)G*width+1,sizeof *bound),*head=malloc(((size_t)G+1)*sizeof *head),
           *next=malloc(((size_t)G+1)*sizeof *next),*rep=malloc(((size_t)G+1)*sizeof *rep);
  unsigned char *reached=malloc((size_t)most+1);
  if(!bound || !head || !next || !rep || !reached){
    free(offset);free(named);free(bound);free(head);free(next);free(rep);free(reached);return ENOMEM;
  }
  for(uint32_t w=0;w<G;w++){
    uint32_t *row=bound+(size_t)w*width;
    for(uint32_t d=0;d<R;d++){
      const struct mesh_metal_range *r=readiness->range+d;
      memset(reached,!named[d],(size_t)r->partitions+1);
      for(size_t a=0;named[d] && a<accesses;a++){
        if(range[a]!=d)continue;
        for(size_t k=first[(size_t)w*accesses+a];k<first[(size_t)w*accesses+a+1];k++)
          if(partition[k]<r->partitions)reached[partition[k]]=1;
      }
      for(uint32_t f=0;f<r->input.pages;f++)
        if(r->field[f]!=UINT32_MAX && r->field[f]<r->partitions && reached[r->field[f]])
          row[offset[d]+f/16]|=1u<<(2*(f%16));
    }
    head[w]=UINT32_MAX;
  }
  uint32_t classes=0;
  size_t entries=0;
  for(uint32_t w=0;w<G;w++){
    const uint32_t *row=bound+(size_t)w*width;
    uint64_t hash=1469598103934665603u;
    for(uint32_t x=0;x<width;x++)hash=(hash^row[x])*1099511628211u;
    const struct mask_rows same={bound,width,row};
    const uint32_t key=(uint32_t)(hash%G),k=mesh_graph_keyed_find(head,next,key,mask_same,&same);
    if(k!=UINT32_MAX){readiness->class_of[w]=readiness->class_of[k];continue;}
    mesh_graph_keyed_add(head,next,key,w);
    readiness->class_of[w]=classes;
    rep[classes++]=w;
    for(uint32_t x=0;x<width;x++)entries+=row[x]!=0;
  }
  /* member: E = 2 + classes R; first[classes R + 1] from member[1]; entries {word, mask} from member[E] */
  const size_t E=2+(size_t)classes*R,words=E+2*entries;
  uint32_t *member=words<=UINT32_MAX?malloc(words*sizeof *member):NULL;
  int status=member?0:ENOMEM;
  if(member){
    member[0]=(uint32_t)E;
    size_t at=0;
    for(uint32_t c=0;c<classes;c++){
      const uint32_t *row=bound+(size_t)rep[c]*width;
      for(uint32_t d=0;d<R;d++){
        member[1+(size_t)c*R+d]=(uint32_t)at;
        const uint32_t n=(readiness->range[d].input.pages+15)/16;
        for(uint32_t x=0;x<n;x++)
          if(row[offset[d]+x]){member[E+2*at]=x;member[E+2*at+1]=row[offset[d]+x];at++;}
      }
    }
    member[1+(size_t)classes*R]=(uint32_t)at;
    readiness->member=member;readiness->members=(uint32_t)words;readiness->classes=classes;
  }
  free(offset);free(named);free(bound);free(head);free(next);free(rep);free(reached);
  return status;
}
/* each footprint's partitions at each threadgroup ([start + terms(g), + extent) of its range), then
   mesh_metal_mask_partitions */
int mesh_metal_mask(const uint32_t groups[3],size_t footprints,const struct mesh_graph_footprint *footprint,
  const struct mesh_graph_term *term,const uint32_t *range,struct mesh_metal_readiness *readiness){
  const uint64_t grid=(uint64_t)groups[0]*groups[1]*groups[2];
  if(readiness->groups!=grid || !grid)return EINVAL;
  for(size_t a=0;a<footprints;a++){
    if(range[a]!=UINT32_MAX && range[a]>=readiness->ranges)return EINVAL;
    for(uint32_t k=0;k<footprint[a].terms;k++)if(term[footprint[a].first+k].dim>2)return EINVAL;
  }
  size_t total=0;
  size_t *first=malloc(((size_t)grid*footprints+1)*sizeof *first);
  uint32_t *partition=NULL;
  if(!first)return ENOMEM;
  for(int pass=0;pass<2;pass++){
    size_t at=0;
    for(uint64_t w=0;w<grid;w++){
      const uint32_t g[3]={(uint32_t)(w%groups[0]),(uint32_t)(w/groups[0]%groups[1]),(uint32_t)(w/groups[0]/groups[1])};
      for(size_t a=0;a<footprints;a++){
        first[w*footprints+a]=at;
        if(range[a]==UINT32_MAX)continue;
        uint32_t lo,hi;
        const __int128 low=mask_start(g,footprint+a,term);
        mask_partitions(low,low+footprint[a].extent,readiness->range+range[a],&lo,&hi);
        for(uint32_t q=lo;lo<=hi && q<=hi;q++){if(pass)partition[at]=q;at++;}
      }
    }
    first[grid*footprints]=at;
    if(!pass){total=at;if(!(partition=malloc((total+1)*sizeof *partition))){free(first);return ENOMEM;}}
  }
  const int status=mesh_metal_mask_partitions(groups,footprints,first,partition,range,readiness);
  free(first);free(partition);
  return status;
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
  struct mesh_planned_transfer rows;
  const int status=planned_rows(planned,range->owner,range->transfer,&rows);
  if(status)return status;
  const uint64_t plan=planned->call[range->owner].plan;
  range->field=field;
  receive_words(context,transport,plan,range->transfer,&range->input);
  return range->input.pages>room?ERANGE:receive_fields(context,planned,rows,&range->input,field);
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
  const struct mesh_planned *planned,uint32_t call,uint32_t transfer,struct mesh_metal_sending *sending){
  (void)transport;
  if(call>=planned->calls)return EINVAL;
  struct mesh_planned_transfer rows;
  const int status=planned_rows(planned,call,transfer,&rows);
  if(status)return status;
  const uint64_t plan=planned->call[call].plan;
  const struct mesh_state *words=mesh_state_at(context->M,plan);
  const struct mesh_publication *entry=mesh_transfer_at(context->M,words,transfer);
  *sending=(struct mesh_metal_sending){.sent={.counter=(char *)context->M+entry->completion,.sends=entry->sends,
                                              .copies=words->copies},.extent=planned->extent+rows.sending,
                                       .extents=rows.end-rows.sending,.transfer=transfer};
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

/* S. Williams, A. Waterman and D. Patterson, Roofline, CACM 52(4), 2009, §3: one resource's recorded operands */
static int machine_read(id given,struct mesh_graph_machine *x){
  if(![given isKindOfClass:[NSDictionary class]])return EINVAL;
  double *field[]={&x->rate[MESH_GRAPH_SCALAR][MESH_GRAPH_BITS32].peak,&x->bandwidth,&x->block_bandwidth,&x->l2,&x->l1,&x->launch,&x->l2_speedup,&x->l1_speedup,
                   &x->overlap,&x->direct,&x->blit};
  NSString *name[]={@"peak",@"bandwidth",@"block_bandwidth",@"l2",@"l1",@"launch",@"l2_speedup",@"l1_speedup",
                    @"overlap",@"direct",@"blit"};
  for(int k=0;k<11;k++){
    id number=((NSDictionary *)given)[name[k]];
    if(number==[NSNull null]){*field[k]=0;continue;}
    if(![number isKindOfClass:[NSNumber class]])return EINVAL;
    *field[k]=[number doubleValue];
  }
  id cores=((NSDictionary *)given)[@"cores"],fpus=((NSDictionary *)given)[@"fpus"];
  uint32_t physical=0;
  size_t size=sizeof physical;
  if(!cores && sysctlbyname("hw.physicalcpu",&physical,&size,NULL,0))return errno;
  if((cores && ![cores isKindOfClass:[NSNumber class]]) || (fpus!=[NSNull null] && ![fpus isKindOfClass:[NSNumber class]]))
    return EINVAL;
  x->cores=cores?[cores unsignedIntValue]:physical;
  x->rate[MESH_GRAPH_SCALAR][MESH_GRAPH_BITS32].lanes=fpus==[NSNull null]?0:[fpus unsignedIntValue];
  id compute=((NSDictionary *)given)[@"compute"];
  if(compute){
    if(![compute isKindOfClass:[NSDictionary class]])return EINVAL;
    NSArray *resources=@[@"scalar",@"matrix",@"integer"],*widths=@[@"16",@"32",@"64"];
    for(uint32_t r=0;r<MESH_GRAPH_RESOURCES;r++){
      id resource=compute[resources[r]];
      if(!resource)continue;
      if(![resource isKindOfClass:[NSDictionary class]])return EINVAL;
      for(uint32_t w=0;w<MESH_GRAPH_WIDTHS;w++){
        id rate=resource[widths[w]];
        if(!rate)continue;
        if(![rate isKindOfClass:[NSDictionary class]])return EINVAL;
        id peak=rate[@"peak"],lanes=rate[@"lanes"];
        if((peak!=[NSNull null] && ![peak isKindOfClass:[NSNumber class]]) ||
           (lanes!=[NSNull null] && ![lanes isKindOfClass:[NSNumber class]]))return EINVAL;
        x->rate[r][w].peak=peak==[NSNull null]?0:[peak doubleValue];
        x->rate[r][w].lanes=lanes==[NSNull null]?0:[lanes unsignedIntValue];
      }
    }
  }
  return 0;
}
/* design/torch-capture-change-request.md, 2026-11-09 appendix: this node's model's GPU and CPU records */
int mesh_metal_machines(const char *path,struct mesh_graph_machine *gpu,struct mesh_graph_machine *cpu){
  char model[256];
  size_t size=sizeof model;
  if(sysctlbyname("hw.model",model,&size,NULL,0))return errno;
  int status=ENOENT;
  @autoreleasepool{
    Dl_info loaded;
    NSString *file=path?[NSString stringWithUTF8String:path]:
      dladdr((const void *)mesh_graph_actor_time,&loaded) && loaded.dli_fname?
      [[[[NSString stringWithUTF8String:loaded.dli_fname] stringByDeletingLastPathComponent]
        stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"etc/mesh-capacity.json"]:nil;
    NSData *data=file?[NSData dataWithContentsOfFile:file]:nil;
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
